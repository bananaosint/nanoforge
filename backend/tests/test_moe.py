import copy
from dataclasses import asdict
import pytest
import torch
from backend.ml.model import GPT,ModelConfig,parameter_count,active_parameter_count,export_config,resolve_model
from backend.ml.moe import SparseMoE
from backend.ml.dataset import DatasetManager
from backend.ml.trainer import Trainer
from backend.storage import Store
from backend.checkpoints import register_checkpoint
from backend.train_xla import prepare

SMALL=dict(n_layer=1,n_head=2,n_embd=16,block_size=32,dropout=0.0,architecture='moe',n_experts=4,experts_per_token=2)
SOURCE=[dict(name='starter-sample.txt',weight=100)]


@pytest.mark.parametrize('top_k',[1,2,4])
def test_sparse_routing_matches_manual_experts_and_backprop(top_k):
    torch.manual_seed(14)
    config=ModelConfig(vocab_size=24,**{**SMALL,'experts_per_token':top_k})
    layer=SparseMoE(config)
    x=torch.randn(2,7,16,requires_grad=True)
    result,aux=layer(x)
    flat=x.reshape(-1,16)
    probs=layer.router(flat).float().softmax(-1)
    weights,indices=probs.topk(top_k,dim=-1)
    if top_k>1:weights=weights/weights.sum(-1,keepdim=True)
    expected=torch.zeros_like(flat)
    for token in range(len(flat)):
        for slot in range(top_k):
            expert=layer.experts[indices[token,slot]]
            expected[token]+=expert(flat[token:token+1])[0]*weights[token,slot]
    torch.testing.assert_close(result,expected.reshape_as(x),atol=1e-6,rtol=1e-5)
    (result.square().mean()+aux).backward()
    assert torch.isfinite(aux) and aux>0
    assert layer.router.weight.grad is not None and layer.router.weight.grad.abs().sum()>0
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_total_budget_and_dense_legacy_export():
    config=ModelConfig(vocab_size=8192,**SMALL)
    with torch.device('meta'):model=GPT(config)
    assert parameter_count(config)==sum(p.numel() for p in model.parameters())
    assert active_parameter_count(config)<parameter_count(config)
    sized=resolve_model('Custom',dict(architecture='moe',n_experts=8,experts_per_token=2,n_layer=12,n_head=8),8192,50_000_000)
    assert parameter_count(sized)<=50_000_000
    with pytest.raises(ValueError):ModelConfig(vocab_size=8192,**{**SMALL,'experts_per_token':5})
    with pytest.raises(ValueError,match='1,000,000,000'):ModelConfig(vocab_size=8192,n_layer=24,n_head=16,n_embd=1024,architecture='moe',n_experts=16)
    dense=ModelConfig(vocab_size=24,n_layer=1,n_head=2,n_embd=16)
    assert 'architecture' not in export_config(dense)
    legacy=GPT(ModelConfig(**export_config(dense)))
    legacy.load_state_dict(GPT(dense).state_dict())


def test_moe_ce_auxiliary_and_causal_inspection():
    model=GPT(ModelConfig(vocab_size=24,**SMALL))
    ids=torch.randint(0,24,(2,12))
    out=model(ids,ids,inspect=True)
    torch.testing.assert_close(out['loss'],out['lm_loss']+out['aux_loss'])
    assert torch.isfinite(out['loss']) and out['attentions'][0].shape==(2,2,12,12)
    model.eval();out=model(ids,ids)
    torch.testing.assert_close(out['loss'],out['lm_loss'])
    assert model.generate(ids[:1],max_new_tokens=2,telemetry=True)['ids'].shape==(1,14)


def exercise_training(home,device,precision,mode='pretrain'):
    store=Store(home/'metadata.sqlite3')
    trainer=Trainer(DatasetManager(home/'data'),store,home/'checkpoints',device)
    try:
        store.save_feedback('sft',dict(prompt='Hello',corrected_output='A useful answer.'))
        store.save_feedback('preference',dict(prompt='Hello',response_a='A useful answer.',response_b='bad',winner='a'))
        trainer.create(dict(model=SMALL,tokenizer='magic8k',precision=precision,datasets=SOURCE))
        before=trainer.model.blocks[0].mlp.router.weight.detach().clone()
        trainer.start(dict(mode=mode,precision=precision,max_steps=2,batch_size=2,eval_interval=2,base_mix=0))
        trainer.thread.join(90)
        assert trainer.state=='completed' and not trainer.thread.is_alive(),trainer.error
        assert not torch.equal(before,trainer.model.blocks[0].mlp.router.weight)
        path=trainer.checkpoint_path(trainer.checkpoint)
        payload=torch.load(path,weights_only=True,map_location='cpu')
        assert payload['model_config']['architecture']=='moe'
        assert all(t.device.type=='cpu' and t.dtype==torch.float32 and torch.isfinite(t).all() for t in payload['model_state_dict'].values())
        register_checkpoint(path,home/'portable')
        local=Trainer(DatasetManager(home/'local-data'),store,home/'portable','cpu')
        try:
            local.load(path.name)
            assert len(local.generate(prompt='Hello',max_new_tokens=2,telemetry=True)['tokens'])==2
            local.start(dict(mode='pretrain',precision='auto',max_steps=1,batch_size=1,eval_interval=1))
            local.thread.join(60);assert local.state=='completed',local.error
        finally:local.shutdown()
    finally:trainer.shutdown();store.close()


@pytest.mark.parametrize('mode',['pretrain','sft','dpo'])
def test_cpu_moe_training_and_portable_resume(tmp_path,mode):
    exercise_training(tmp_path,'cpu','float32',mode)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='Requires real CUDA hardware; run from the G4 launcher.')
@pytest.mark.parametrize('precision',['float16','bfloat16'])
@pytest.mark.parametrize('mode',['pretrain','sft','dpo'])
def test_cuda_moe_training_export(tmp_path,precision,mode):
    if precision=='bfloat16' and (torch.cuda.get_device_capability()[0]<8 or not torch.cuda.is_bf16_supported()):pytest.skip('Native BF16 GPU required.')
    exercise_training(tmp_path,'cuda',precision,mode)


def test_tpu_explicitly_rejects_sparse_moe(tmp_path):
    with pytest.raises(ValueError,match='MoE is not supported'):
        prepare(dict(model=SMALL,tokenizer='magic8k'),tmp_path)
