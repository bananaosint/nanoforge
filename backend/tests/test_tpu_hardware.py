"""Opt-in real eight-core TPU smoke test, run by the Kaggle notebook in subprocesses."""
import json
import os
from pathlib import Path
import subprocess
import sys
import pytest
import torch
from backend.checkpoints import register_checkpoint
from backend.ml.dataset import DatasetManager
from backend.ml.trainer import Trainer
from backend.ml.xla_training import initialize_model
from backend.storage import Store


@pytest.mark.skipif(os.environ.get('NANOFORGE_TEST_TPU')!='1',reason='Requires real Kaggle TPU hardware and matching PyTorch/XLA.')
def test_real_tpu_replicated_training_and_studio_export(tmp_path):
    config={'preset':'Custom','model':{'n_layer':1,'n_head':2,'n_embd':16,'block_size':32,'dropout':0.0},
            'tokenizer':'magic8k','precision':'bfloat16','batch_size':1,'gradient_accumulation':2,
            'max_steps':2,'eval_interval':2}
    path=tmp_path/'smoke.json';path.write_text(json.dumps(config))
    home=tmp_path/'run'
    env={**os.environ,'PJRT_DEVICE':'TPU'}
    subprocess.run([sys.executable,'-m','backend.train_xla','--config',str(path),'--home',str(home),
                    '--expected-replicas','8','--log-every','1'],check=True,env=env)
    manifest=json.loads((home/'latest-checkpoint.json').read_text())
    checkpoint=Path(manifest['checkpoint'])
    payload=torch.load(checkpoint,map_location='cpu',weights_only=True)
    assert payload['world_size']==8 and payload['precision']=='bfloat16' and payload['step']==2
    assert all(t.device.type=='cpu' and t.dtype==torch.float32 and torch.isfinite(t).all()
               for t in payload['model_state_dict'].values())
    initial=initialize_model(payload['model_config'],42,'cpu')
    assert not torch.equal(initial.token_embedding.weight,payload['model_state_dict']['token_embedding.weight'])
    store=Store(tmp_path/'studio.sqlite3')
    trainer=Trainer(DatasetManager(tmp_path/'studio-data'),store,tmp_path/'studio-checkpoints','cpu')
    try:
        metadata=register_checkpoint(checkpoint,trainer.directory)
        trainer.load(metadata['name'])
        assert len(trainer.generate(prompt='Hello',max_new_tokens=2,telemetry=False)['tokens'])==2
        trainer.start({'precision':'auto','mode':'pretrain','max_steps':1,'batch_size':1,'eval_interval':1})
        trainer.thread.join(60)
        assert not trainer.thread.is_alive() and trainer.state=='completed',trainer.error
        assert trainer.step==3
    finally:trainer.shutdown();store.close()
