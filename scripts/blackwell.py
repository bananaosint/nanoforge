"""Fresh-process Blackwell stack probes, with one pinned CUDA wheel repair attempt."""
import subprocess
import sys

PROBE=r'''
import sys,subprocess
try:
    report=subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version,memory.total','--format=csv,noheader'],text=True,stderr=subprocess.STDOUT)
except Exception as exc:
    print('No NVIDIA GPU/driver detected:',str(exc),flush=True);sys.exit(2)
print(report,flush=True)
if not ('RTX PRO 6000' in report.upper() and 'BLACKWELL' in report.upper()):
    print('Choose the G4 RTX PRO 6000 Blackwell runtime; other GPUs need their matching profile.',flush=True);sys.exit(2)
try:
    import torch
    if not torch.cuda.is_available():raise RuntimeError('GPU hardware exists but this PyTorch CUDA installation is unavailable.')
    name=torch.cuda.get_device_name(0)
    major,minor=torch.cuda.get_device_capability(0)
    if not ('RTX PRO 6000' in name.upper() and 'BLACKWELL' in name.upper() and major==12):
        print('The visible CUDA device is not the assigned G4 Blackwell. Check runtime/device visibility.',flush=True);sys.exit(2)
    print('GPU:',name,'| capability:',(major,minor),'| VRAM:',round(torch.cuda.get_device_properties(0).total_memory/1024**3,1),'GiB',flush=True)
    version=tuple(int(p) for p in torch.__version__.split('.')[:2])
    cuda=tuple(int(p) for p in (torch.version.cuda or '0.0').split('.')[:2])
    if version<(2,7) or cuda<(12,8):raise RuntimeError('A Blackwell CUDA build is required.')
    print('PyTorch:',torch.__version__,'CUDA:',torch.version.cuda,'architectures:',torch.cuda.get_arch_list(),flush=True)
    model=torch.nn.Linear(64,64,device='cuda')
    opt=torch.optim.AdamW(model.parameters(),fused=True)
    x=torch.randn(8,32,64,device='cuda')
    with torch.autocast('cuda',dtype=torch.bfloat16):
        q=model(x).view(8,4,8,64)
        y=torch.nn.functional.scaled_dot_product_attention(q,q,q,is_causal=True)
        loss=y.float().square().mean()
    loss.backward();opt.step();torch.cuda.synchronize()
    assert torch.isfinite(loss).item()
    print('Blackwell CUDA kernel probe passed.',flush=True)
except Exception as exc:
    print('CUDA stack probe failed:',repr(exc),flush=True);sys.exit(3)
'''



def ensure_blackwell(repo,env,run=subprocess.run):
    command=[sys.executable,'-c',PROBE]
    result=run(command,cwd=repo,env=env)
    if result.returncode==2:raise RuntimeError('No matching G4 GPU is assigned. Change the Colab runtime before setup.')
    if result.returncode==3:
        print('Installing the pinned Blackwell-capable PyTorch 2.10.0 / CUDA 12.8 wheel...',flush=True)
        run([sys.executable,'-m','pip','install','--no-cache-dir','--upgrade','torch==2.10.0',
             '--index-url','https://download.pytorch.org/whl/cu128'],cwd=repo,env=env,check=True)
        result=run(command,cwd=repo,env=env)
    if result.returncode:
        raise RuntimeError('Blackwell CUDA checks failed. Inspect driver/runtime logs; restart the Colab runtime after a package repair if necessary.')
