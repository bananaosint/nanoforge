"""Replica loss parity on CPU; real two-CUDA-device integration is opt-in by hardware."""
import copy
import pytest
import torch
from torch.nn import functional as F
from backend.ml.model import GPT,ModelConfig
from backend.ml.parallel import ReplicaLoss,reduced_loss
from backend.ml.dataset import DatasetManager
from backend.ml.trainer import Trainer
from backend.storage import Store
from backend.checkpoints import register_checkpoint

SMALL=dict(n_layer=1,n_head=2,n_embd=32,block_size=32,dropout=0.0)


def test_unequal_masked_replica_losses_and_gradients_match_global_batch():
    torch.manual_seed(11)
    model=GPT(ModelConfig(vocab_size=24,**SMALL));reference=copy.deepcopy(model)
    x=torch.randint(0,24,(3,12));y=x.clone();y[0,:8]=-100;y[1,:2]=-100
    a=ReplicaLoss(model)(x[:2],y[:2]);b=ReplicaLoss(model)(x[2:],y[2:])
    loss=reduced_loss((torch.cat((a[0],b[0])),torch.cat((a[1],b[1]))))
    expected=reference(x,y)['loss']
    torch.testing.assert_close(loss,expected)
    loss.backward();expected.backward()
    for p,q in zip(model.parameters(),reference.parameters()):torch.testing.assert_close(p.grad,q.grad,atol=2e-6,rtol=2e-5)


def test_replica_dpo_uses_fixed_reference():
    model=GPT(ModelConfig(vocab_size=24,**SMALL));reference=copy.deepcopy(model).requires_grad_(False)
    x=torch.randint(0,24,(3,12));y=x.clone();y[:,:4]=-100
    rejected_x=x.flip(1);rejected_y=rejected_x.clone()
    replica=ReplicaLoss(model,reference=reference)
    loss=reduced_loss(replica(x,y,rejected_x,rejected_y))
    assert loss.item()==pytest.approx(0.693147,abs=1e-6)
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in model.parameters())
    assert all(p.grad is None for p in reference.parameters())


def test_cpu_rejects_two_gpu_training_clearly(tmp_path):
    store=Store(tmp_path/'test.sqlite3');trainer=Trainer(DatasetManager(tmp_path/'data'),store,tmp_path/'checkpoints','cpu')
    try:
        trainer.create({'model':SMALL,'datasets':[{'name':'starter-sample.txt','weight':100}]})
        with pytest.raises(ValueError,match='at least two CUDA GPUs'):trainer.start({'multi_gpu':True,'max_steps':1})
    finally:trainer.shutdown();store.close()


@pytest.mark.skipif(torch.cuda.device_count()<2,reason='Requires two actual CUDA GPUs; run from the Kaggle T4 x2 launcher.')
@pytest.mark.parametrize('mode',['pretrain','sft','dpo'])
def test_real_two_gpu_training_and_portable_export(tmp_path,mode):
    store=Store(tmp_path/'metadata.sqlite3');trainer=Trainer(DatasetManager(tmp_path/'data'),store,tmp_path/'checkpoints','cuda')
    try:
        store.save_feedback('sft',{'prompt':'Hello','corrected_output':'A friendly story.'})
        store.save_feedback('preference',{'prompt':'Hello','response_a':'A friendly story.','response_b':'bad','winner':'a'})
        trainer.create({'model':SMALL,'tokenizer':'magic8k','precision':'float16','datasets':[{'name':'starter-sample.txt','weight':100}]})
        before=trainer.model.token_embedding.weight.detach().clone()
        trainer.start({'precision':'float16','multi_gpu':True,'mode':mode,'max_steps':2,'batch_size':3,'eval_interval':2,'base_mix':0})
        seen=set()
        hook=trainer.parallel.module.register_forward_pre_hook(lambda module,args:seen.add(args[0].device.index))
        trainer.thread.join(120);hook.remove()
        assert not trainer.thread.is_alive() and trainer.state=='completed',trainer.error
        assert seen=={0,1} and trainer.status()['training_gpus']==2
        assert not torch.equal(before,trainer.model.token_embedding.weight)
        payload=torch.load(trainer.checkpoint_path(trainer.checkpoint),map_location='cpu',weights_only=True)
        assert not any(name.startswith(('module.','model.')) for name in payload['model_state_dict'])
        assert all(t.device.type=='cpu' and t.dtype==torch.float32 and torch.isfinite(t).all() for t in payload['model_state_dict'].values())
        registered=register_checkpoint(trainer.checkpoint_path(trainer.checkpoint),tmp_path/'portable')
        local=Trainer(DatasetManager(tmp_path/'local-data'),store,tmp_path/'portable','cpu')
        try:
            local.load(registered['name'])
            assert len(local.generate(prompt='Hello',max_new_tokens=2,telemetry=False)['tokens'])==2
            local.start({'multi_gpu':False,'mode':'pretrain','precision':'auto','max_steps':1,'batch_size':1,'eval_interval':1})
            local.thread.join(60);assert local.state=='completed',local.error
        finally:local.shutdown()
    finally:trainer.shutdown();store.close()
