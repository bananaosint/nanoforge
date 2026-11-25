import json
import sys
import threading
import time
from unittest.mock import patch
import httpx
import pytest
import torch
from fastapi.testclient import TestClient
from backend.gemini_client import GeminiClient,GeminiError,GeminiService,BackgroundCall,strict_json
from backend.main import create_app
from backend.ml.dataset import DatasetManager
from backend.ml.trainer import Trainer
from backend.storage import Store
from backend.teaching import TeachingJobs

SMALL={'n_layer':1,'n_head':2,'n_embd':16,'block_size':32,'dropout':0.0}
CONFIG={'model':SMALL,'tokenizer':'char','datasets':[{'name':'starter-sample.txt','weight':100}],
        'max_steps':4,'batch_size':1,'eval_interval':100,'precision':'float32'}
CONTINUE={'action':'CONTINUE','reason':'Numeric evidence is insufficient for an intervention.','confidence':1.0}


@pytest.mark.parametrize('text',[
    '{"action":"CONTINUE","action":"EARLY_STOP"}',
    '{"confidence":NaN}', '{"confidence":Infinity}',
    'prefix {"valid":true}', '{} trailing', '[{}]', '```json\n{}\n``` extra',
])
def test_strict_json_rejects_ambiguous_or_nonfinite_content(text):
    with pytest.raises(GeminiError):strict_json(text)


def test_mock_and_fenced_cli_are_deterministic_without_keys(monkeypatch):
    monkeypatch.delenv('GEMINI_API_KEY',raising=False)
    client=GeminiClient({'mode':'mock'})
    assert client.request('watchdog',{})['action']=='CONTINUE'
    assert client.request('teach',{'prompt':'Explain tokens.'})==client.request('teach',{'prompt':'Explain tokens.'})
    script='import sys,json; request=json.load(sys.stdin); assert request["task"]=="watchdog"; print("```json\\n"+json.dumps('+repr(CONTINUE)+')+"\\n```")'
    cli=GeminiClient({'mode':'cli','cli_command':[sys.executable,'-c',script]})
    assert cli.request('watchdog',{})==CONTINUE


@pytest.mark.parametrize('response',[
    {**CONTINUE,'confidence':True},{**CONTINUE,'confidence':'0.9'},
    {**CONTINUE,'confidence':1.01},{**CONTINUE,'action':'SET_LR'},
    {**CONTINUE,'learning_rate':.1},{**CONTINUE,'reason':''},
])
def test_cli_response_schema_is_strict(response):
    client=GeminiClient({'mode':'cli','cli_command':[sys.executable,'-c','print('+repr(json.dumps(response))+')']})
    with pytest.raises(GeminiError):client.request('watchdog',{})


def test_cli_timeout_and_failure_never_expose_stderr():
    client=GeminiClient({'mode':'cli','cli_command':[sys.executable,'-c','import time; time.sleep(10)'],'timeout_seconds':1})
    began=time.monotonic()
    with pytest.raises(GeminiError,match='timed out'):client.request('watchdog',{})
    assert time.monotonic()-began<4
    failed=GeminiClient({'mode':'cli','cli_command':[sys.executable,'-c','import sys;sys.stderr.write("secret-value");sys.exit(1)']})
    with pytest.raises(GeminiError) as error:failed.request('watchdog',{})
    assert 'secret-value' not in str(error.value)


def test_cli_output_limit_stops_a_verbose_process():
    client=GeminiClient({'mode':'cli','timeout_seconds':5,'cli_command':[sys.executable,'-c',
        'import sys,time;sys.stdout.write("x"*400000);sys.stdout.flush();time.sleep(10)']})
    began=time.monotonic()
    with pytest.raises(GeminiError,match='size limit'):client.request('watchdog',{})
    assert time.monotonic()-began<4


def test_api_contract_uses_header_and_structured_response_schema(monkeypatch):
    monkeypatch.setenv('GEMINI_API_KEY','test-secret')
    def respond(request):
        assert request.headers['x-goog-api-key']=='test-secret' and not request.url.query
        payload=json.loads(request.content)
        text_format=payload['generationConfig']['responseFormat']['text']
        assert text_format['mimeType']=='APPLICATION_JSON'
        assert text_format['schema']['additionalProperties'] is False
        assert 'test-secret' not in request.content.decode()
        return httpx.Response(200,json={'candidates':[{'content':{'parts':[{'text':json.dumps(CONTINUE)}]}}]})
    client=GeminiClient({'mode':'api'},transport=httpx.MockTransport(respond))
    assert client.request('watchdog',{})==CONTINUE
    client.transport=httpx.MockTransport(lambda request:httpx.Response(403,text='secret provider body'))
    with pytest.raises(GeminiError,match='HTTP 403') as error:client.request('watchdog',{})
    assert 'secret provider body' not in str(error.value)


def test_settings_key_stays_out_of_database_and_responses(tmp_path,monkeypatch):
    monkeypatch.delenv('GEMINI_API_KEY',raising=False)
    store=Store(tmp_path/'settings.sqlite3')
    try:
        service=GeminiService(store)
        status=service.configure({'mode':'api','api_key':'session-secret','setup_complete':True})
        assert status['key_configured'] and status['key_source']=='session'
        assert 'session-secret' not in json.dumps(status)+json.dumps(store.setting('gemini'))
        assert not GeminiService(store).status()['key_configured']
        service.configure({'mode':'api','clear_api_key':True})
        assert not service.status()['key_configured']
    finally:store.close()


def test_nonsecret_config_file_selects_provider(tmp_path,monkeypatch):
    config=tmp_path/'gemini.json';config.write_text(json.dumps({'mode':'cli','cli_command':['my-gemini-wrapper']}))
    monkeypatch.setenv('NANOFORGE_GEMINI_CONFIG',str(config))
    store=Store(tmp_path/'settings.sqlite3')
    try:
        assert GeminiService(store).status()['mode']=='cli'
        config.write_text(json.dumps({'mode':'api','api_key':'never-store-this'}))
        with pytest.raises(ValueError,match='Keep API keys out'):GeminiService(store)
    finally:store.close()


@pytest.fixture
def trainer(tmp_path):
    store=Store(tmp_path/'metadata.sqlite3')
    trainer=Trainer(DatasetManager(tmp_path/'data'),store,tmp_path/'checkpoints','cpu')
    trainer.create(CONFIG)
    yield trainer
    trainer.shutdown();store.close()


def test_slow_provider_does_not_block_training_or_shutdown(trainer):
    entered=threading.Event();release=threading.Event()
    def delayed(self,task,payload):
        entered.set();release.wait(10);return CONTINUE
    try:
        with patch.object(GeminiClient,'request',delayed):
            trainer.start({**CONFIG,'max_steps':12,'watchdog_enabled':True,'watchdog_interval_steps':1})
            assert entered.wait(5)
            trainer.thread.join(5)
            assert not trainer.thread.is_alive() and trainer.state=='completed',trainer.error
            began=time.monotonic();trainer.shutdown()
            assert time.monotonic()-began<.5
            assert trainer.watchdog.worker.busy()
            assert any(event['action']=='SKIPPED' for event in trainer.watchdog.events)
    finally:release.set()


def test_provider_failure_does_not_change_training_state(trainer):
    with patch.object(GeminiClient,'request',side_effect=GeminiError('Gemini timed out.')):
        trainer.start({**CONFIG,'max_steps':5,'watchdog_enabled':True,'watchdog_interval_steps':1})
        trainer.thread.join(10)
        assert trainer.state=='completed',trainer.error
        if trainer.watchdog.worker.thread:trainer.watchdog.worker.thread.join(1)
        assert any(event['action']=='ERROR' for event in trainer.status()['watchdog']['events'])


def prepare_recovery(trainer):
    trainer.start({**CONFIG,'max_steps':2},background=False,evaluate=False)
    trainer.config['watchdog_enabled']=True
    trainer.val_loss=1.0;trainer.watchdog.evaluated();trainer.save()
    return torch.load(trainer.directory/trainer.watchdog.best_file,weights_only=True)


def decision(trainer,action,confidence=1.0,epoch=None):
    trainer.watchdog.worker.result={'epoch':epoch or trainer.watchdog.epoch,'updates':trainer.watchdog.updates,
                                    'provider_revision':trainer.gemini.revision,
                                    'decision':{'action':action,'reason':'Structured metrics support this action.','confidence':confidence}}
    trainer.watchdog._consume()


def test_rollback_restores_weights_optimizer_and_step_then_dampens_lr(trainer):
    checkpoint=prepare_recovery(trainer)
    with torch.no_grad():trainer.model.token_embedding.weight.add_(5)
    trainer.step=9;trainer.target_step=20
    trainer.watchdog.validations.append({'step':9,'loss':2.0})
    decision(trainer,'ROLLBACK_AND_DAMPEN')
    assert trainer.step==checkpoint['step'] and trainer.target_step==20
    assert trainer.optimizer.param_groups[0]['lr']==pytest.approx(.00015)
    torch.testing.assert_close(trainer.model.token_embedding.weight,checkpoint['model_state_dict']['token_embedding.weight'])
    actual=next(iter(trainer.optimizer.state.values()))
    expected=next(iter(checkpoint['optimizer_state_dict']['state'].values()))
    torch.testing.assert_close(actual['exp_avg'],expected['exp_avg'])
    assert trainer.watchdog.rollbacks==1
    trainer.save();name=trainer.checkpoint;trainer.load(name)
    assert trainer.watchdog.best_file and trainer.watchdog.best_val_loss==1.0


def test_stale_low_confidence_and_unsupported_numeric_advice_is_ignored(trainer):
    prepare_recovery(trainer)
    original=trainer.optimizer.param_groups[0]['lr']
    trainer.watchdog.validations.append({'step':9,'loss':2.0})
    decision(trainer,'ROLLBACK_AND_DAMPEN',epoch='stale')
    decision(trainer,'ROLLBACK_AND_DAMPEN',confidence=.1)
    assert trainer.optimizer.param_groups[0]['lr']==original
    decision(trainer,'BUMP_BALANCE_LOSS')  # Dense model cannot change an MoE coefficient.
    assert not trainer.watchdog.events[-1]['applied']
    trainer.watchdog.validations.clear()
    decision(trainer,'EARLY_STOP')
    assert not trainer.stop_event.is_set()


def test_balance_intervention_requires_two_collapsed_routing_checkpoints(trainer):
    trainer.create({**CONFIG,'model':{**SMALL,'architecture':'moe','n_experts':4,'experts_per_token':1}})
    trainer.config['watchdog_enabled']=True
    trainer.watchdog.routing.append({'step':1,'expert_load_pct':[[100.,0.,0.,0.]]})
    decision(trainer,'BUMP_BALANCE_LOSS')
    assert trainer.model.config.router_aux_loss==.01
    trainer.watchdog.routing.append({'step':2,'expert_load_pct':[[100.,0.,0.,0.]]})
    decision(trainer,'BUMP_BALANCE_LOSS')
    assert trainer.model.config.router_aux_loss==.02 and trainer.model.blocks[0].mlp.aux_coef==.02


def test_early_stop_requires_five_regressed_evaluations_and_rollback_is_bounded(trainer):
    prepare_recovery(trainer)
    trainer.watchdog.validations.clear()
    trainer.watchdog.validations.extend({'step':step,'loss':1.5} for step in range(5))
    decision(trainer,'EARLY_STOP')
    assert trainer.stop_event.is_set() and trainer.watchdog.events[-1]['applied']
    trainer.watchdog.rollbacks=3
    assert not trainer.watchdog.guard('ROLLBACK_AND_DAMPEN')


def test_failed_restore_keeps_current_model_and_optimizer(trainer):
    prepare_recovery(trainer)
    trainer.watchdog.validations.append({'step':9,'loss':2.0})
    model,optimizer=trainer.model,trainer.optimizer
    with patch('backend.ml.trainer.torch.load',side_effect=OSError('Bad recovery file')):
        decision(trainer,'ROLLBACK_AND_DAMPEN')
    assert trainer.model is model and trainer.optimizer is optimizer
    assert not trainer.watchdog.events[-1]['applied']


def test_recovery_copy_fallback_and_session_key_never_enter_checkpoint(trainer):
    trainer.gemini.configure({'mode':'api','api_key':'UNIQUE_SESSION_SECRET_DO_NOT_STORE'})
    with patch('backend.watchdog.os.link',side_effect=OSError('Links unsupported')):
        checkpoint=prepare_recovery(trainer)
    assert checkpoint['watchdog_state']['best_file']==trainer.watchdog.best_file
    assert b'UNIQUE_SESSION_SECRET_DO_NOT_STORE' not in (trainer.directory/trainer.checkpoint).read_bytes()


def test_replica_telemetry_matches_single_device_moe():
    from backend.ml.model import GPT,ModelConfig
    from backend.ml.parallel import ReplicaLoss,reduced_loss
    model=GPT(ModelConfig(vocab_size=40,**SMALL,architecture='moe',n_experts=4,experts_per_token=2))
    ids=torch.randint(0,40,(2,12))
    expected=model(ids,ids)
    result=ReplicaLoss(model,telemetry=True)(ids,ids)
    torch.testing.assert_close(reduced_loss(result),expected['loss'])
    torch.testing.assert_close(result[2].sum()/result[5].sum(),expected['balance_loss'])
    torch.testing.assert_close(result[4].sum(0)/result[5].sum(),expected['expert_load_pct'])


def test_moe_telemetry_in_payload_is_numeric_and_checkpointed(trainer):
    trainer.create({**CONFIG,'model':{**SMALL,'architecture':'moe','n_experts':4,'experts_per_token':2}})
    trainer.start({**CONFIG,'watchdog_enabled':True,'watchdog_interval_steps':2,'eval_interval':1},background=False)
    assert trainer.state=='completed',trainer.error
    payload=trainer.watchdog.payload()
    assert len(payload['validation_loss_history'])==5 and len(payload['routing_checkpoints'])>=2
    assert payload['best_val_loss'] is not None and payload['gradient_norm_window']
    assert 'prompts' not in payload and 'samples' not in payload
    for layer in payload['routing_checkpoints'][-1]['expert_load_pct']:assert sum(layer)==pytest.approx(200,abs=1e-4)
    assert trainer.metrics[-1]['balance_loss']>0 and trainer.metrics[-1]['z_loss']>0


def test_teacher_draft_requires_idempotent_approval_then_sft_trains(tmp_path,monkeypatch):
    monkeypatch.setenv('NANOFORGE_DEVICE','cpu')
    app=create_app(tmp_path)
    with TestClient(app) as client:
        settings=client.get('/api/gemini/settings').json()
        assert not settings['setup_complete'] and settings['mode']=='mock'
        assert client.put('/api/gemini/settings',json={'mode':'mock','setup_complete':True}).status_code==200
        result=client.post('/api/teaching',json={'prompt':'Explain tokens.','student_output':'noise'}).json()
        app.state.teaching.worker.thread.join(2)
        job=client.get('/api/teaching/'+result['id']).json()
        assert job['state']=='draft' and client.get('/api/feedback').json()==[]
        target='A token is a piece of text represented by an integer.'
        url='/api/teaching/'+job['id']+'/approve'
        first=client.post(url,json={'corrected_output':target})
        assert first.status_code==200,first.text
        assert client.post(url,json={'corrected_output':target}).json()['id']==first.json()['id']
        feedback=client.get('/api/feedback').json()
        assert len(feedback)==1 and feedback[0]['corrected_output']==target
        assert feedback[0]['teacher_provenance']['human_edited']
        assert client.post('/api/runs',json=CONFIG).status_code==200
        response=client.post('/api/training/start',json={'mode':'sft','max_steps':2,'batch_size':1,'base_mix':0})
        assert response.status_code==200,response.text
        app.state.trainer.thread.join(10)
        assert app.state.trainer.state=='completed' and app.state.trainer.mode=='sft'
        assert app.state.trainer.checkpoint


def test_teacher_failure_creates_no_feedback_and_restart_preserves_drafts(tmp_path):
    store=Store(tmp_path/'teacher.sqlite3');service=GeminiService(store)
    try:
        jobs=TeachingJobs(store,service)
        with patch.object(GeminiClient,'request',side_effect=GeminiError('Malformed teacher response.')):
            failed=jobs.start({'prompt':'Hello','student_output':'noise','checkpoint':None})
            jobs.worker.thread.join(2)
        assert jobs.get(failed['id'])['state']=='error' and not store.feedback()
        good=jobs.start({'prompt':'Hello','student_output':'noise','checkpoint':None});jobs.worker.thread.join(2)
        assert jobs.get(good['id'])['state']=='draft'
        resumed=TeachingJobs(store,service)
        assert resumed.get(good['id'])['state']=='draft' and not store.feedback()
    finally:store.close()
