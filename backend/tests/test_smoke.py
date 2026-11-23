import json
import time
from dataclasses import asdict
import pytest
import torch
from fastapi.testclient import TestClient
from backend.ml.model import GPT, ModelConfig
from backend.ml.tokenizer import Tokenizer
from backend.ml.dataset import DatasetManager, WeightedMixer, completion_batch
from backend.ml.trainer import Trainer
from backend.storage import Store
from backend.main import create_app

SMALL = dict(n_layer=1, n_head=2, n_embd=32, block_size=32, dropout=0.0)
SOURCES = [dict(name='starter-sample.txt', weight=100)]


def wait(trainer):
    trainer.thread.join(timeout=30)
    assert not trainer.thread.is_alive(), 'Training thread did not stop'
    assert trainer.state == 'completed', trainer.error


def test_forward_and_inspection():
    torch.manual_seed(9)
    model = GPT(ModelConfig(vocab_size=24, **SMALL)).eval()
    ids = torch.randint(0, 24, (2, 12))
    fast = model(ids, ids)
    manual = model(ids, ids, inspect=True)
    assert fast['logits'].shape == (2, 12, 24)
    assert torch.isfinite(fast['loss'])
    assert len(manual['attentions']) == 1
    attention = manual['attentions'][0]
    assert attention.shape == (2, 2, 12, 12)
    assert torch.all(attention.triu(1) == 0)
    torch.testing.assert_close(attention.sum(-1), torch.ones(2, 2, 12))
    torch.testing.assert_close(fast['logits'], manual['logits'], atol=1e-5, rtol=1e-5)
    assert model.token_embedding.weight is model.lm_head.weight
    a = model.generate(ids[:1], max_new_tokens=3, telemetry=True, seed=6)
    b = model.generate(ids[:1], max_new_tokens=3, telemetry=True, seed=6)
    assert torch.equal(a['ids'], b['ids'])
    assert len(a['steps']) == 3
    assert len(a['steps'][0]['top5']) == 5
    assert a['attention'][0].shape == (1, 2, 15, 15)


def test_tokenizers_and_prompt_mask():
    tokenizer = Tokenizer(text='hello world')
    assert tokenizer.decode(tokenizer.encode('hello')) == 'hello'
    assert Tokenizer(state=tokenizer.state()).encode('world') == tokenizer.encode('world')
    x, y = completion_batch(tokenizer, 'hello', 'world', 100, 'cpu')
    assert (y == -100).sum() > 0
    assert (y != -100).sum() == len(tokenizer.encode('world\n<|end|>\n'))
    assert x.shape == y.shape
    bpe = Tokenizer('bpe')
    assert bpe.encode('hello world') == [31373, 995]
    assert bpe.decode(bpe.encode('hello 🌍')) == 'hello 🌍'
    with pytest.raises(ValueError):
        ModelConfig(vocab_size=24, n_head=3, n_embd=32)


def test_pretrain_checkpoint_sft_dpo(tmp_path):
    store = Store(tmp_path/'test.sqlite3')
    trainer = Trainer(DatasetManager(tmp_path/'data'), store, tmp_path/'checkpoints', 'cpu')
    try:
        trainer.create(dict(model=SMALL, tokenizer='char', datasets=SOURCES, batch_size=2))
        before = trainer.model.token_embedding.weight.detach().clone()
        trainer.start(dict(mode='pretrain', datasets=SOURCES, batch_size=2, max_steps=5, eval_interval=5, gradient_accumulation=2))
        wait(trainer)
        assert trainer.step == 5
        assert not torch.equal(before, trainer.model.token_embedding.weight)
        assert len(trainer.metrics) == 5
        assert store.rows('SELECT step FROM timeline WHERE step=0')
        assert store.rows('SELECT step FROM timeline WHERE step=5')
        name = trainer.checkpoint
        saved_weights = trainer.model.token_embedding.weight.detach().clone()
        path = trainer.checkpoint_path(name)
        payload = torch.load(path, weights_only=True)
        assert {'model_config','tokenizer_state','model_state_dict','optimizer_state_dict','step','val_loss','version_tag'} <= set(payload)
        parent_run = trainer.run_id
        trainer.load(name)
        torch.testing.assert_close(saved_weights, trainer.model.token_embedding.weight)
        assert trainer.step == 5
        store.save_feedback('sft', dict(prompt='hello', original_output='noise', corrected_output='world'))
        trainer.start(dict(mode='sft', batch_size=1, base_mix=0, max_steps=2, eval_interval=2))
        wait(trainer)
        assert trainer.step == 7
        assert trainer.run_id != parent_run
        assert trainer.config['parent_checkpoint'] == name
        sft_weights = trainer.model.token_embedding.weight.detach().clone()
        store.save_feedback('preference', dict(prompt='hello', response_a='world', response_b='bad', winner='a', checkpoint_a=name, checkpoint_b=name))
        trainer.start(dict(mode='dpo', batch_size=1, max_steps=2, eval_interval=2))
        wait(trainer)
        assert trainer.step == 9
        assert trainer.metrics[0]['train_loss'] == pytest.approx(0.693147, abs=0.0001)
        assert not torch.equal(sft_weights, trainer.model.token_embedding.weight)
        assert all(torch.isfinite(p).all() for p in trainer.model.parameters())
        assert trainer.checkpoint.startswith('dpo-')
    finally:
        trainer.shutdown()
        store.close()


def test_pause_resume_stop_and_nan(tmp_path):
    store=Store(tmp_path/'test.sqlite3')
    trainer=Trainer(DatasetManager(tmp_path/'data'),store,tmp_path/'checkpoints','cpu')
    try:
        trainer.create(dict(model=SMALL,datasets=SOURCES))
        trainer.start(dict(mode='pretrain',datasets=SOURCES,batch_size=1,max_steps=10000,eval_interval=10000))
        trainer.pause()
        # Wait for an already-started step to complete, then confirm it stays paused.
        with trainer.lock:
            step=trainer.step
        time.sleep(.2)
        assert trainer.step == step
        trainer.resume()
        trainer.stop()
        trainer.thread.join(30)
        assert trainer.state == 'stopped'
        assert trainer.checkpoint
        with torch.no_grad():
            trainer.model.token_embedding.weight.fill_(float('nan'))
        trainer.start(dict(mode='pretrain',datasets=SOURCES,batch_size=1,max_steps=1))
        trainer.thread.join(30)
        assert trainer.state=='error'
        assert 'Non-finite' in trainer.error
    finally:
        trainer.shutdown()
        store.close()


def test_dataset_split_and_offline_fallback(tmp_path, monkeypatch):
    manager=DatasetManager(tmp_path)
    tokenizer=Tokenizer(text=manager.text('starter-sample.txt'))
    mixer=WeightedMixer([(manager.text('starter-sample.txt'),100)],tokenizer,32,'cpu')
    train,val,_=mixer.sources[0]
    assert len(train)/(len(train)+len(val)) == pytest.approx(.9,abs=.001)
    assert mixer.batch(2)[0].shape==(2,32)
    def offline(*args,**kwargs):
        raise OSError('offline')
    monkeypatch.setattr('backend.ml.dataset.urlopen',offline)
    assert manager.download()['offline_fallback']
    assert manager.text('tinyshakespeare.txt')
    with pytest.raises(ValueError):
        WeightedMixer([('a',100)],tokenizer,32,'cpu')


def test_api_endpoints(tmp_path,monkeypatch):
    monkeypatch.setenv('NANOFORGE_DEVICE','cpu')
    app=create_app(tmp_path)
    with TestClient(app) as client:
        assert client.get('/api/health').json()['device']=='cpu'
        assert client.get('/api/status').json()['state']=='idle'
        assert client.post('/api/generate',json={'prompt':'hello'}).status_code==400
        assert client.post('/api/runs',json={'datasets':[]}).status_code==400
        assert client.post('/api/datasets/upload',files={'file':('empty.txt',b'')}).status_code==400
        assert client.post('/api/datasets/upload',files={'file':('example.txt',b'hello world '*100)}).status_code==200
        assert client.get('/api/datasets/example.txt/preview').json()['characters']==1200
        assert client.post('/api/datasets/upload',files={'file':('bad.jsonl',b'{bad}')}).status_code==400
        assert client.post('/api/datasets/upload',files={'file':('good.jsonl',b'{"prompt":"hi","completion":"hello"}\n')}).status_code==200
        partial=client.post('/api/runs',json={'preset':'Micro','model':{'n_layer':1},'datasets':SOURCES})
        assert partial.status_code==200,partial.text
        assert partial.json()['model_config']['n_embd']==256
        assert partial.json()['model_config']['n_layer']==1
        response=client.post('/api/runs',json={'model':SMALL,'datasets':SOURCES})
        assert response.status_code==200,response.text
        assert client.post('/api/training/start',json={'max_steps':5,'batch_size':1,'eval_interval':5,'datasets':SOURCES}).status_code==200
        wait(app.state.trainer)
        with client.websocket_connect('/ws/training') as ws:
            assert ws.receive_json()['step']==5
        checkpoints=client.get('/api/checkpoints').json()
        assert checkpoints
        assert client.post('/api/checkpoints/select',json={'name':checkpoints[0]['name']}).status_code==200
        gen=client.post('/api/generate',json={'prompt':'First Citizen:','max_new_tokens':3,'layer':0,'head':0})
        assert gen.status_code==200,gen.text
        assert len(gen.json()['steps'])==3
        assert len(gen.json()['attention'])==1
        assert len(gen.json()['attention'][0])==1
        assert client.post('/api/generate',json={'prompt':'x','layer':30,'max_new_tokens':1}).status_code==400
        assert client.post('/api/generate',json={'prompt':'x','max_new_tokens':-1}).status_code==422
        arena=client.post('/api/arena',json={'prompt':'First','max_new_tokens':3,'checkpoint_a':checkpoints[0]['name'],'checkpoint_b':checkpoints[0]['name']})
        assert arena.status_code==200
        assert arena.json()['a']['seed']!=arena.json()['b']['seed']
        data={'kind':'sft','prompt':'Hi','original_output':'Bad','corrected_output':'Hello'}
        feedback_id=client.post('/api/feedback',json=data).json()['id']
        assert client.put('/api/feedback/'+str(feedback_id),json={**data,'corrected_output':'Hello there'}).status_code==200
        exported=client.get('/api/feedback-export').text
        assert json.loads(exported)['corrected_output']=='Hello there'
        assert client.delete('/api/feedback/'+str(feedback_id)).status_code==200
        assert client.post('/api/training/start',json={'mode':'dpo'}).status_code==400
        assert client.put('/api/prompts',json={'prompts':['one','two','three']}).status_code==200
        assert client.get('/api/timeline').json()
        assert client.get('/api/runs').json()
        assert client.delete('/api/datasets/example.txt').status_code==200
        cors=client.options('/api/status',headers={'Origin':'http://localhost:5173','Access-Control-Request-Method':'GET'})
        assert cors.headers['access-control-allow-origin']=='http://localhost:5173'
        assert client.post('/api/checkpoints/select',json={'name':'../../evil.pt'}).status_code==400
