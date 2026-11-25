"""CPU coverage for file state shared between ZeroGPU's short-lived workers."""
import json
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch
import pytest
import torch
from backend import zerogpu as engine
from backend.checkpoints import register_checkpoint
from backend.ml.dataset import DatasetManager
from backend.ml.trainer import Trainer
from backend.storage import Store
from scripts.export_zerogpu import export

SMALL=dict(n_layer=1,n_head=2,n_embd=16,block_size=32,dropout=0.0)


def options(**overrides):
    return {**engine.PROFILE['training'],'tokenizer':'char','model':SMALL,'batch_size':1,**overrides}


@pytest.mark.parametrize('architecture',['dense','moe'])
def test_prepare_chunks_portable_resume_and_generation(tmp_path,architecture):
    root=tmp_path/'sessions'
    config=options(model={**SMALL,'architecture':architecture,'n_experts':4,'experts_per_token':2})
    with patch('torch.cuda.is_available',side_effect=AssertionError('CPU preparation probed CUDA')):
        session,message,initial=engine.prepare_session(root,config)
    path=engine.session_path(root,session)
    assert 'GPU not requested' in message and Path(initial).is_file()
    assert list((path/'data/.tokens').glob('*.u16'))
    before=torch.load(initial,weights_only=True)['model_state_dict']['token_embedding.weight']
    # A leased GPU function must finish training and saving on its calling thread.
    with patch('backend.ml.trainer.threading.Thread',side_effect=AssertionError('Background worker started')):
        for step in (2,4,6):
            metric,latest=engine.train_chunk(root,session,2,device='cpu')
            assert metric['step']==step and metric['state']=='completed'
    payload=torch.load(latest,weights_only=True)
    assert not torch.equal(before,payload['model_state_dict']['token_embedding.weight'])
    assert all(t.device.type=='cpu' and t.dtype==torch.float32 for t in payload['model_state_dict'].values())
    assert len(list((path/'checkpoints').glob('*.pt')))==2
    assert engine.generate(root,session,'Hello',2,device='cpu')
    # Register/load in the ordinary Studio, independently of the Space session.
    store=Store(tmp_path/'studio.sqlite3')
    studio=Trainer(DatasetManager(tmp_path/'studio-data'),store,tmp_path/'studio-checkpoints','cpu')
    try:
        registered=register_checkpoint(latest,studio.directory)
        studio.load(registered['name'])
        assert studio.step==6 and studio.model.config.architecture==architecture
        imported,_,_=engine.prepare_session(root,config,checkpoint=latest)
        assert engine.train_chunk(root,imported,1,device='cpu')[0]['step']==7
    finally:studio.shutdown();store.close()


@pytest.mark.parametrize('mode',['sft','dpo'])
def test_normalized_feedback_objectives(tmp_path,mode):
    dataset=tmp_path/'feedback.jsonl'
    row={'prompt':'Hello','completion':'A good answer.'} if mode=='sft' else {'prompt':'Hello','chosen':'A good answer.','rejected':'A bad answer.'}
    dataset.write_text(json.dumps(row)+'\n',encoding='utf-8')
    config=options(mode=mode,base_mix=0)
    session,_,_=engine.prepare_session(tmp_path/'sessions',config,dataset=dataset)
    metrics,checkpoint=engine.train_chunk(tmp_path/'sessions',session,2,device='cpu')
    assert metrics['step']==2 and metrics['state']=='completed'
    assert torch.load(checkpoint,weights_only=True)['mode']==mode


def test_failed_or_expired_lease_preserves_committed_checkpoint(tmp_path,monkeypatch):
    root=tmp_path/'sessions';session,_,initial=engine.prepare_session(root,options())
    path=engine.session_path(root,session);previous=(path/'session.json').read_bytes()
    with patch.object(Trainer,'_save',side_effect=RuntimeError('Interrupted save')):
        with pytest.raises(RuntimeError,match='Interrupted save'):engine.train_chunk(root,session,1,device='cpu')
    assert (path/'session.json').read_bytes()==previous and Path(initial).is_file()
    monkeypatch.setitem(engine.PROFILE,'work_seconds',0)
    metrics,checkpoint=engine.train_chunk(root,session,2,device='cpu')
    assert metrics['state']=='stopped' and metrics['step']==0 and Path(checkpoint).is_file()


def test_invalid_requests_are_bounded_and_sessions_locked(tmp_path):
    root=tmp_path/'sessions'
    with patch('backend.ml.trainer.GPT',side_effect=AssertionError('Allocated oversized model')):
        with pytest.raises(ValueError,match='10M'):
            engine.prepare_session(root,options(tokenizer='bpe',model={**SMALL,'n_embd':256}))
    assert not list(root.iterdir())
    session,_,_=engine.prepare_session(root,options())
    path=engine.session_path(root,session)
    with engine.session_lock(path):
        with pytest.raises(ValueError,match='running request'):engine.train_chunk(root,session,1,device='cpu')
    for invalid in ('../outside','',None):
        with pytest.raises(ValueError):engine.session_path(root,invalid)
    for steps in (0,26,1.5):
        with pytest.raises(ValueError):engine.train_chunk(root,session,steps,device='cpu')
    bad=tmp_path/'bad.txt';bad.write_bytes(b'\xff\xfe')
    with pytest.raises(UnicodeDecodeError):engine.prepare_session(root,options(),dataset=bad)
    assert len(list(root.iterdir()))==1


def test_export_is_self_contained_and_preserves_existing_files(tmp_path):
    bundle=export(tmp_path/'space')
    assert (bundle/'backend/assets/magic8k/tokenizer.json').is_file()
    assert (bundle/'configs/zerogpu.json').is_file()
    assert not (bundle/'backend/tests').exists() and not (bundle/'.git').exists()
    assert not list(bundle.rglob('*.sqlite3')) and not list(bundle.rglob('*.pt'))
    before=(bundle/'README.md').read_bytes()
    with pytest.raises(ValueError,match='empty'):export(bundle)
    assert (bundle/'README.md').read_bytes()==before
    with pytest.raises(ValueError,match='separate'):export(Path(__file__).resolve().parents[2]/'exported-space')


def test_exported_gradio_ui_builds_without_cuda(tmp_path):
    pytest.importorskip('gradio');pytest.importorskip('spaces')
    bundle=export(tmp_path/'space')
    code="""
from unittest.mock import patch
import torch
with patch('torch.cuda.is_available',side_effect=AssertionError('Probed CUDA at import')):
    import app
assert len(app.demo.config['dependencies'])==4
assert app.SIZE=='large'
print('Gradio UI built without CUDA initialization')
"""
    result=subprocess.run([sys.executable,'-c',code],cwd=bundle,text=True,capture_output=True,timeout=60,
                          env={**os.environ,'PYTHONPATH':str(bundle),'NANOFORGE_ZEROGPU_HOME':str(tmp_path/'state'),'NANOFORGE_ZERO_GPU_SIZE':'large'})
    assert result.returncode==0,result.stdout+result.stderr
