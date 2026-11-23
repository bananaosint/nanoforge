"""CPU parity tests for TPU batching, objectives and exports; no mocked TPU claims."""
from contextlib import nullcontext
from dataclasses import asdict
import json
import random
import torch
import pytest
from backend.train_xla import prepare
from backend.ml.xla_training import (fixed_completion_batch,fixed_pretrain_batch,
    initialize_model,log_probabilities,objective,run_replica)
from backend.ml.dataset import WeightedMixer,completion_batch,completion_log_probability
from backend.ml.tokenizer import Tokenizer
from backend.ml.model import GPT,ModelConfig
from backend.ml.trainer import Trainer
from backend.ml.dataset import DatasetManager
from backend.checkpoints import register_checkpoint
from backend.storage import Store

SMALL=dict(n_layer=1,n_head=2,n_embd=16,block_size=32,dropout=0.0)


class CpuRuntime:
    device='cpu';rank=0;world_size=1
    def optimizer(self,parameters,lr):return torch.optim.AdamW(parameters,lr=lr,weight_decay=0.1,foreach=False)
    def autocast(self,precision):return nullcontext()
    def broadcast(self,model):pass
    def seed(self,seed):pass
    def reduce_gradients(self,optimizer):pass
    def mean(self,value):return value
    def step_optimizer(self,optimizer,found_inf):
        if not found_inf.item():optimizer.step()
    def sync(self):pass
    def barrier(self,tag,error=''):
        if error:raise ValueError('Distributed checkpoint failed: '+error)


def test_fixed_shapes_masking_and_dpo_probability():
    tokenizer=Tokenizer(text='a long prompt short completion')
    model=GPT(ModelConfig(vocab_size=tokenizer.vocab_size,**SMALL)).eval()
    pairs=[dict(prompt='a',corrected_output='short'),dict(prompt='a long prompt'*8,corrected_output='completion'*8)]
    x,y=fixed_completion_batch(tokenizer,pairs,32)
    assert x.shape==y.shape==(2,32)
    for index,pair in enumerate(pairs):
        a,b=completion_batch(tokenizer,pair['prompt'],pair['corrected_output'],32,'cpu')
        assert torch.equal(x[index,:a.shape[1]],a[0])
        assert torch.equal(y[index,:b.shape[1]],b[0])
        assert (y[index,b.shape[1]:]==-100).all()
        torch.testing.assert_close(log_probabilities(model,x[index:index+1],y[index:index+1]),
            completion_log_probability(model,tokenizer,pair['prompt'],pair['corrected_output'],'cpu'))
    pool=[dict(prompt='a',response_a='short',response_b='completion',winner='a')]
    options=dict(batch_size=2,beta=0.1)
    loss,_=objective(model,tokenizer,'dpo',None,pool,model,options,random.Random(1),'cpu')
    assert loss.item()==pytest.approx(0.693147,abs=1e-6)
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in model.parameters())


def test_pretrain_shapes_and_rank_sampling_are_stable():
    tokenizer=Tokenizer(text='a b c d')
    mixer=WeightedMixer([('a b '*60,1),('c d '*200,1)],tokenizer,64,'cpu')
    shapes=set()
    for validation in (False,True):
        rng=random.Random(42)
        for _ in range(10):
            x,y=fixed_pretrain_batch(mixer,4,rng,validation)
            shapes.add(tuple(x.shape));assert x.shape==y.shape
    assert len(shapes)==1
    a=fixed_pretrain_batch(mixer,4,random.Random(42))[0]
    b=fixed_pretrain_batch(mixer,4,random.Random(1_000_045))[0]
    assert not torch.equal(a,b)


def test_initialization_reproducible_without_changing_cpu_rng():
    config=asdict(ModelConfig(vocab_size=24,**SMALL))
    before=torch.random.get_rng_state().clone()
    first=initialize_model(config,42,'cpu')
    assert torch.equal(before,torch.random.get_rng_state())
    second=initialize_model(config,42,'cpu')
    for a,b in zip(first.parameters(),second.parameters()):torch.testing.assert_close(a,b)


def test_xla_engine_export_cpu_studio_resume_and_objectives(tmp_path):
    torch.set_num_threads(2)
    config=dict(preset='Custom',model=SMALL,tokenizer='char',precision='bfloat16',batch_size=2,
                max_steps=2,eval_interval=2,gradient_accumulation=2)
    plan=prepare(config,tmp_path)
    assert list((tmp_path/'data/.tokens').glob('*.u16'))
    path=run_replica(plan,CpuRuntime())
    payload=torch.load(path,weights_only=True)
    assert payload['precision']=='bfloat16'
    assert payload['step']==2
    assert payload['model_state_dict']['lm_head.weight'].data_ptr()==payload['model_state_dict']['token_embedding.weight'].data_ptr()
    assert all(t.device.type=='cpu' and t.dtype==torch.float32 for t in payload['model_state_dict'].values())
    resumed=prepare(config,tmp_path,resume=path)
    next_path=run_replica(resumed,CpuRuntime())
    assert torch.load(next_path,weights_only=True)['step']==4
    store=Store(tmp_path/'nanoforge.sqlite3')
    trainer=Trainer(DatasetManager(tmp_path/'data'),store,tmp_path/'studio','cpu')
    try:
        metadata=register_checkpoint(next_path,trainer.directory)
        trainer.load(metadata['name'])
        assert trainer.precision=='float32'
        assert trainer.model.config.n_embd==16
        assert len(trainer.generate(prompt='a',max_new_tokens=2,telemetry=False)['tokens'])==2
        store.save_feedback('sft',dict(prompt='hello',corrected_output='world'))
        store.save_feedback('preference',dict(prompt='hello',response_a='world',response_b='bad',winner='a'))
    finally:trainer.shutdown();store.close()
    for mode in ('sft','dpo'):
        options={**config,'mode':mode,'base_mix':0,'max_steps':1,'eval_interval':1}
        plan=prepare(options,tmp_path,resume=next_path)
        exported=run_replica(plan,CpuRuntime())
        result=torch.load(exported,weights_only=True)
        assert result['mode']==mode and result['step']==5
        assert all(torch.isfinite(t).all() for t in result['model_state_dict'].values())


def test_tpu_setup_validation_and_no_silent_cpu_fallback(tmp_path):
    config=dict(preset='Custom',model=SMALL)
    with pytest.raises(ValueError,match='not CUDA float16'):
        prepare({**config,'precision':'float16'},tmp_path)
    with pytest.raises(ValueError,match='No usable SFT'):
        prepare({**config,'mode':'sft','base_mix':0},tmp_path)
    plan=prepare(config,tmp_path,expected_replicas=8)
    assert plan['config']['precision']=='bfloat16'
    with pytest.raises(ValueError,match='Expected 8 TPU replicas'):
        run_replica(plan,CpuRuntime())
    store=Store(tmp_path/'test.sqlite3')
    trainer=Trainer(DatasetManager(tmp_path/'data'),store,tmp_path/'local','cpu')
    try:
        with pytest.raises(ValueError,match='backend.train_xla'):
            trainer.create({**plan['config'],'precision':'bfloat16'})
    finally:trainer.shutdown();store.close()


def test_checkpoint_retention_preserves_other_runs(tmp_path):
    config=dict(preset='Custom',model=SMALL,tokenizer='char',precision='float32',batch_size=1,
                max_steps=3,eval_interval=1)
    directory=tmp_path/'checkpoints';directory.mkdir()
    foreign=directory/'pretrain-another-run-s1.pt';foreign.write_bytes(b'keep')
    plan=prepare(config,tmp_path,keep_last=1)
    exported=run_replica(plan,CpuRuntime())
    assert exported.exists() and exported.name.endswith('-s3.pt')
    assert len(list(directory.glob(f"pretrain-{plan['run_id']}-s*.pt")))==1
    assert foreign.read_bytes()==b'keep'
    assert json.loads((tmp_path/'latest-checkpoint.json').read_text())['checkpoint']==str(exported)


def test_save_failure_propagates_to_replicas(tmp_path,monkeypatch):
    config=dict(preset='Custom',model=SMALL,batch_size=1,max_steps=1,eval_interval=1)
    plan=prepare(config,tmp_path)
    def disk_full(*args,**kwargs):raise OSError('No space left on device')
    monkeypatch.setattr('backend.ml.xla_training.export_checkpoint',disk_full)
    with pytest.raises(ValueError,match='Distributed checkpoint failed: OSError: No space'):
        run_replica(plan,CpuRuntime())
