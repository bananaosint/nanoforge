"""Password-protected Kaggle dual-T4 Studio; nothing trains until the UI starts it."""
import hashlib
import json
import os
from pathlib import Path
import re
import runpy
import secrets
import subprocess
import sys
import time
from urllib.request import urlopen
from urllib.error import HTTPError

GPU_CHECK=r'''
import torch
assert torch.cuda.is_available(), 'Enable GPU T4 x2 in Kaggle Notebook settings.'
assert torch.cuda.device_count()==2, f'Expected two GPUs, got {torch.cuda.device_count()}. Select GPU T4 x2.'
for i in range(2):
    name=torch.cuda.get_device_name(i)
    assert 'T4' in name, f'Expected T4 GPUs, got {name}.'
    print('GPU',i,name,round(torch.cuda.get_device_properties(i).total_memory/1024**3,1),'GiB',flush=True)
'''


def cloudflared_binary(directory):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    # Pin the official release digest, avoiding shared Kaggle IPs' GitHub API rate limits.
    version='2026.11.2'
    expected='56f13335143ec0e22772963290683a84a04114a35ec64cbc794a4d39ff9bb8cf'
    download='https://github.com/cloudflare/cloudflared/releases/download/'+version+'/cloudflared-linux-amd64'
    target=directory/('cloudflared-'+expected[:16])
    if target.is_file() and hashlib.sha256(target.read_bytes()).hexdigest()==expected:return target
    temporary=target.with_suffix('.partial')
    checksum=hashlib.sha256()
    try:
        with urlopen(download,timeout=60) as source,temporary.open('wb') as output:
            while chunk:=source.read(1024*1024):checksum.update(chunk);output.write(chunk)
        if checksum.hexdigest()!=expected:raise ValueError('cloudflared download checksum mismatch.')
        temporary.chmod(0o755);temporary.replace(target)
    finally:temporary.unlink(missing_ok=True)
    return target


def stop_tunnel(process):
    if process.poll() is None:
        process.terminate()
        try:process.wait(timeout=35)
        except subprocess.TimeoutExpired:
            process.kill();process.wait(timeout=5)  # Only the owned network tunnel, never the trainer.


class KaggleSession:
    def __init__(self,studio,tunnel,handle,log,url,password):
        self.studio,self.tunnel,self._handle,self.log=studio,tunnel,handle,log
        self.url,self.password=url,password

    def logs(self):
        return self.studio.logs()+'\n'+self.log.read_text(errors='replace')[-6000:]

    def stop(self):
        stop_tunnel(self.tunnel)
        self._handle.close()
        self.studio.stop()


def launch_kaggle(repo='/kaggle/working/nanoforge',home='/kaggle/working/nanoforge-run',port=8765):
    repo,home=Path(repo).resolve(),Path(home).resolve()
    temporary=home/'install-tmp';temporary.mkdir(parents=True,exist_ok=True)
    env={**os.environ,'PIP_NO_CACHE_DIR':'1','TMPDIR':str(temporary),'NANOFORGE_DEVICE':'cuda',
         'npm_config_cache':str(temporary/'npm-cache')}
    def run(*args,cwd=repo):subprocess.run(list(args),cwd=cwd,env=env,check=True)
    print('Installing dependencies and verifying both T4 GPUs...',flush=True)
    run(sys.executable,'-m','pip','install','--no-cache-dir','-r','backend/requirements.txt')
    run(sys.executable,'-c',GPU_CHECK)
    run(sys.executable,'-m','pytest','backend/tests/test_dual_gpu.py::test_real_two_gpu_training_and_portable_export','-q','-s')
    common=runpy.run_path(str(repo/'scripts/colab_ui.py'))
    env=common['node_environment'](temporary/'node',env)
    print('Building Studio...',flush=True)
    run('npm','ci',cwd=repo/'frontend');run('npm','run','build',cwd=repo/'frontend')
    binary=cloudflared_binary(temporary/'cloudflared')
    password=secrets.token_urlsafe(24)
    studio=common['start_server'](repo,home,port=port,profile='kaggle-t4x2',password=password)
    log=home/'tunnel.log';handle=log.open('wb')
    try:
        tunnel=subprocess.Popen([str(binary),'tunnel','--no-autoupdate','--protocol','http2','--url',f'http://127.0.0.1:{port}'],
                                stdin=subprocess.DEVNULL,stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
        deadline=time.monotonic()+60
        url=None
        while time.monotonic()<deadline:
            if tunnel.poll() is not None:raise RuntimeError('Tunnel failed:\n'+log.read_text(errors='replace')[-6000:])
            match=re.search(r'https://[a-z0-9-]+\.trycloudflare\.com',log.read_text(errors='replace'))
            if match:url=match.group();break
            time.sleep(0.5)
        if not url:raise RuntimeError('No tunnel URL was issued. Inspect '+str(log))
        deadline=time.monotonic()+45
        protected=False
        while time.monotonic()<deadline:
            try:
                with urlopen(url+'/api/health',timeout=5) as response:
                    raise RuntimeError('Tunnel endpoint unexpectedly allowed access without a password.')
            except HTTPError as exc:
                if exc.code==401 and 'NanoForge' in exc.headers.get('WWW-Authenticate',''):
                    protected=True;break
                if exc.code not in (502,503,504,530):raise
            except OSError:pass
            time.sleep(1)
        if not protected:raise RuntimeError('Tunnel did not become ready. Check Kaggle Internet access and tunnel logs.')
    except Exception:
        if 'tunnel' in locals():stop_tunnel(tunnel)
        handle.close();studio.stop();raise
    session=KaggleSession(studio,tunnel,handle,log,url,password)
    print('\nOPEN STUDIO:',url,'\nUsername: nanoforge\nPassword:',password,flush=True)
    print('Keep the password private. Choose data/settings in the UI, then click Start training.',flush=True)
    return session
