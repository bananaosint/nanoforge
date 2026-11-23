"""CPU CLI integration for the L4 profile; CUDA execution is tested in Colab."""
import ast
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import pytest
import torch
from backend.api.schemas import NewRun
from backend.checkpoints import register_checkpoint
from backend.ml.dataset import DatasetManager
from backend.ml.trainer import Trainer
from backend.storage import Store

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('colab_l4',ROOT/'scripts/colab_l4.py')
startup=importlib.util.module_from_spec(spec);spec.loader.exec_module(startup)


def test_l4_profile_and_custom_settings_validate_without_installing():
    config=startup.training_config(ROOT,100,4,8)
    validated=NewRun.model_validate(config)
    assert validated.precision=='float16' and validated.allow_tf32
    assert validated.model.block_size==512
    assert validated.batch_size*validated.gradient_accumulation==32
    custom=dict(n_layer=16,n_head=16,n_embd=1024,block_size=512,dropout=0.1)
    sized=startup.training_config(ROOT,100,1,8,custom)
    assert sized['preset']=='Custom' and sized['model']==custom
    NewRun.model_validate(sized)
    assert startup.training_config(ROOT,100,4,8)['preset']=='Mini'
    with pytest.raises(ValueError,match='steps must be an integer'):
        startup.run_l4(repo=ROOT,steps=0)
    with pytest.raises(ValueError,match='batch_size must be an integer'):
        startup.run_l4(repo=ROOT,batch_size=129)
    with pytest.raises(ValueError,match='Resume checkpoint does not exist'):
        startup.run_l4(repo=ROOT,resume=ROOT/'does-not-exist.pt')


def test_l4_cli_size_preview_and_portable_manifest(tmp_path):
    config=startup.training_config(ROOT,2,1,1,dict(n_layer=1,n_head=2,n_embd=16,block_size=32,dropout=0.0))
    # Hardware-specific FP16 is exercised by the CUDA test, not simulated here.
    config.update(precision='float32',datasets=[dict(name='starter-sample.txt',weight=100)],eval_interval=2)
    path=tmp_path/'config.json';path.write_text(json.dumps(config))
    home=tmp_path/'run'
    command=[sys.executable,'-m','backend.train_cli','--home',str(home),'--config',str(path),'--device','cpu']
    env={**os.environ,'NANOFORGE_DEVICE':'cpu'}
    preview=subprocess.run([*command,'--preview'],cwd=ROOT,env=env,capture_output=True,text=True,check=True)
    count=json.loads(preview.stdout)['parameters']
    assert count>0 and not list((home/'checkpoints').glob('*.pt'))
    assert not (home/'latest-checkpoint.json').exists()
    result=subprocess.run(command,cwd=ROOT,env=env,capture_output=True,text=True,check=True)
    manifest=json.loads((home/'latest-checkpoint.json').read_text())
    assert manifest==json.loads(result.stdout.splitlines()[-1])
    assert manifest['state']=='completed' and manifest['step']==2
    checkpoint=Path(manifest['checkpoint'])
    payload=torch.load(checkpoint,weights_only=True,map_location='cpu')
    assert payload['training_config']['allow_tf32']
    assert payload['precision']=='float32'
    assert all(t.device.type=='cpu' and t.dtype==torch.float32 for t in payload['model_state_dict'].values())
    metadata=register_checkpoint(checkpoint,tmp_path/'studio-checkpoints')
    assert metadata['parameters']==count
    store=Store(tmp_path/'studio.sqlite3')
    trainer=Trainer(DatasetManager(tmp_path/'studio-data'),store,tmp_path/'studio-checkpoints','cpu')
    try:
        trainer.load(metadata['name'])
        assert not trainer.allow_tf32
        assert len(trainer.generate(prompt='Hello',max_new_tokens=2,telemetry=False)['tokens'])==2
    finally:trainer.shutdown();store.close()


def test_l4_notebook_uses_committed_startup_function():
    notebook=json.loads((ROOT/'notebooks/NanoForge_L4.ipynb').read_text())
    assert notebook['metadata']['colab']['gpuType']=='L4'
    for cell in notebook['cells']:
        if cell['cell_type']=='code':
            assert not cell['outputs'];source=''.join(cell['source']);ast.parse(source)
            assert 'scripts/colab_l4.py' in source and "startup['run_l4']" in source
