"""One-cell Colab L4 setup. Execute in the notebook kernel; training runs in a child.

No work is performed on import. The caller clones/updates the repository first.
"""
import json
import os
from pathlib import Path
import subprocess
import sys

GPU_CHECK = r'''
import torch
assert torch.cuda.is_available(), 'Select Runtime > Change runtime type > L4 GPU, then rerun.'
name = torch.cuda.get_device_name(0)
assert name.upper().split()[-1] == 'L4', f'Expected an L4 GPU, got {name}. Select L4 in Colab.'
assert torch.cuda.get_device_capability(0)[0] >= 8, 'L4 requires a recent CUDA PyTorch installation.'
print('GPU:', name, '| VRAM:', round(torch.cuda.get_device_properties(0).total_memory/1024**3, 1), 'GiB', flush=True)
print('PyTorch:', torch.__version__, '| CUDA:', torch.version.cuda, flush=True)
'''


def training_config(repo,steps,batch_size,accumulation,custom_model=None):
    config=json.loads((Path(repo)/'configs/colab-l4.json').read_text(encoding='utf-8'))
    config.update(max_steps=steps,batch_size=batch_size,gradient_accumulation=accumulation)
    if custom_model is not None:config.update(preset='Custom',model=custom_model)
    return config


def run_l4(repo='/content/nanoforge',home='/content/nanoforge-l4-run',steps=1000,
           batch_size=32,accumulation=2,custom_model=None,resume=None):
    """Install, check real L4 training, prepare Dolly and train; return exact outputs."""
    repo,home=Path(repo).resolve(),Path(home).resolve()
    if not (repo/'configs/colab-l4.json').is_file():
        raise ValueError('Update the NanoForge repository before running the L4 startup cell.')
    for name,value,maximum in (('steps',steps,1_000_000),('batch_size',batch_size,128),('accumulation',accumulation,64)):
        if not isinstance(value,int) or isinstance(value,bool) or not 1<=value<=maximum:
            raise ValueError(f'{name} must be an integer between 1 and {maximum}.')
    if resume is not None and not Path(resume).is_file():raise ValueError('Resume checkpoint does not exist.')
    env=os.environ.copy()
    env['NANOFORGE_DEVICE']='cuda'
    env['PIP_NO_CACHE_DIR']='1'
    temporary=home/'install-tmp';temporary.mkdir(parents=True,exist_ok=True)
    env['TMPDIR']=str(temporary)

    def run(*args):
        subprocess.run(list(args),cwd=str(repo),env=env,check=True)

    # Colab already supplies CUDA PyTorch; preserve it, avoiding CPU/XLA wheels.
    run(sys.executable,'-m','pip','install','--no-cache-dir','-r','backend/requirements.txt')
    run(sys.executable,'-c',GPU_CHECK)
    print('Checking FP16 training, TF32 policy and portable checkpoint export on your GPU...',flush=True)
    run(sys.executable,'-m','pytest','backend/tests/test_expanded.py::test_cuda_fp16_training_and_cpu_studio_export','-q','-s')
    run(sys.executable,'-m','backend.importers','--dataset','dolly','--home',str(home))
    run(sys.executable,'-m','backend.pretokenize','--home',str(home),'--tokenizer','magic8k','--dataset','dolly.jsonl')
    config=training_config(repo,steps,batch_size,accumulation,custom_model)
    config_file=home/'l4-training-config.json'
    config_file.write_text(json.dumps(config,indent=2),encoding='utf-8')
    if resume is None:
        print('Exact size preview (no model tensors allocated):',flush=True)
        run(sys.executable,'-m','backend.train_cli','--home',str(home),'--config',str(config_file),'--device','cuda','--preview')
    command=[sys.executable,'-m','backend.train_cli','--home',str(home),'--config',str(config_file),'--device','cuda']
    if resume is not None:command.extend(['--resume',str(Path(resume).resolve())])
    run(*command)
    result=json.loads((home/'latest-checkpoint.json').read_text(encoding='utf-8'))
    if result['state']!='completed':raise RuntimeError('Training did not finish. Inspect the logs and saved checkpoints.')
    print('Studio checkpoint:',result['checkpoint'],flush=True)
    return result
