"""Launch the real NanoForge Studio in Colab; model/training settings live in the UI."""
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import runpy
import shutil
import signal
import socket
import subprocess
import sys
import tarfile
import time
from urllib.request import urlopen, Request
import base64


def node_environment(cache,env):
    """Use a compatible installed Node, otherwise a verified official Node 22 bundle."""
    env=dict(env)
    node=shutil.which('node',path=env.get('PATH'))
    npm=shutil.which('npm',path=env.get('PATH'))
    if node and npm:
        version=subprocess.check_output([node,'--version'],text=True).strip().lstrip('v')
        major,minor,*_=map(int,version.split('.'))
        if (major==20 and minor>=19) or (major==22 and minor>=12) or major>=24:
            return env
    if platform.system()!='Linux':raise ValueError('Automatic Node installation is supported in Linux Colab runtimes.')
    architecture={'x86_64':'x64','aarch64':'arm64'}.get(platform.machine())
    if not architecture:raise ValueError('Unsupported Colab CPU architecture.')
    cache=Path(cache);cache.mkdir(parents=True,exist_ok=True)
    base='https://nodejs.org/dist/latest-v22.x/'
    with urlopen(base+'SHASUMS256.txt',timeout=60) as response:
        manifest=response.read().decode('utf-8')
    pattern=rf'^([a-f0-9]{{64}})\s+(node-v22\.\d+\.\d+-linux-{architecture}\.tar\.xz)$'
    match=re.search(pattern,manifest,re.MULTILINE)
    if not match:raise RuntimeError('Could not find the official Node 22 package checksum.')
    checksum,filename=match.groups()
    installed=cache/filename.removesuffix('.tar.xz')
    if not (installed/'bin/node').is_file():
        archive=cache/filename
        digest=hashlib.sha256()
        try:
            with urlopen(base+filename,timeout=60) as response,archive.open('wb') as target:
                while chunk:=response.read(1024*1024):digest.update(chunk);target.write(chunk)
            if digest.hexdigest()!=checksum:raise ValueError('Node download checksum mismatch. Rerun setup.')
            if not hasattr(tarfile,'data_filter'):raise RuntimeError('A recent Python with safe tar extraction is required.')
            with tarfile.open(archive) as package:package.extractall(cache,filter='data')
        finally:archive.unlink(missing_ok=True)
    env['PATH']=str(installed/'bin')+os.pathsep+env.get('PATH','')
    return env


class StudioSession:
    def __init__(self,process,port,home,log,handle):
        self.process,self.port,self.home,self.log,self._handle=process,port,home,log,handle

    def logs(self):
        with self.log.open('rb') as source:
            source.seek(max(0,self.log.stat().st_size-12000))
            return source.read().decode('utf-8',errors='replace')

    def stop(self,timeout=60):
        if self.process.poll() is None:
            self.process.send_signal(signal.SIGINT)
            try:self.process.wait(timeout=timeout)
            except subprocess.TimeoutExpired as exc:
                raise RuntimeError('Studio is still finishing its current step/checkpoint. Call stop() again before restarting.') from exc
        self._handle.close()


def start_server(repo,home,port=8000,device='cuda',profile='colab-l4',timeout=60,password=None):
    """Start only the owned backend process, with built UI and no model/run initialized."""
    repo,home=Path(repo).resolve(),Path(home).resolve()
    if not (repo/'frontend/dist/index.html').is_file():raise ValueError('Build the frontend before starting Studio.')
    if not isinstance(port,int) or not 1024<=port<=65535:raise ValueError('Use a port between 1024 and 65535.')
    with socket.socket() as probe:
        probe.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
        try:probe.bind(('127.0.0.1',port))
        except OSError as exc:raise ValueError(f'Port {port} is already in use. Stop the previous Studio before restarting.') from exc
    home.mkdir(parents=True,exist_ok=True)
    env={**os.environ,'NANOFORGE_HOME':str(home),'NANOFORGE_DEVICE':device,
         'NANOFORGE_FRONTEND_DIR':str(repo/'frontend/dist'),'NANOFORGE_UI_PROFILE':profile}
    env.pop('NANOFORGE_UI_PASSWORD',None)
    if password:env['NANOFORGE_UI_PASSWORD']=password
    log=home/'studio-server.log'
    handle=log.open('ab')
    process=subprocess.Popen([sys.executable,'-m','uvicorn','backend.main:app','--host','127.0.0.1',
                              '--port',str(port),'--workers','1'],cwd=repo,env=env,
                             stdin=subprocess.DEVNULL,stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
    session=StudioSession(process,port,home,log,handle)
    deadline=time.monotonic()+timeout
    while time.monotonic()<deadline:
        if process.poll() is not None:
            handle.close();raise RuntimeError('Studio failed to start:\n'+session.logs())
        try:
            request=Request(f'http://127.0.0.1:{port}/api/health')
            if password:request.add_header('Authorization','Basic '+base64.b64encode(('nanoforge:'+password).encode()).decode())
            with urlopen(request,timeout=1) as response:
                health=json.load(response)
            if health.get('pid')==process.pid and health['device']==device:
                return session
        except (OSError,ValueError):pass
        time.sleep(0.25)
    session.stop()
    raise RuntimeError('Studio startup timed out:\n'+session.logs())


def launch_studio(repo='/content/nanoforge',home='/content/nanoforge-l4-run',port=8000):
    repo,home=Path(repo).resolve(),Path(home).resolve()
    temporary=home/'install-tmp';temporary.mkdir(parents=True,exist_ok=True)
    env={**os.environ,'PIP_NO_CACHE_DIR':'1','TMPDIR':str(temporary),'NANOFORGE_DEVICE':'cuda',
         'npm_config_cache':str(temporary/'npm-cache')}
    def run(*args,cwd=repo):subprocess.run(list(args),cwd=cwd,env=env,check=True)
    print('Installing Python dependencies and checking the L4 GPU...',flush=True)
    run(sys.executable,'-m','pip','install','--no-cache-dir','-r','backend/requirements.txt')
    gpu_check=runpy.run_path(str(repo/'scripts/colab_l4.py'))['GPU_CHECK']
    run(sys.executable,'-c',gpu_check)
    run(sys.executable,'-m','pytest','backend/tests/test_expanded.py::test_cuda_fp16_training_and_cpu_studio_export','-q')
    print('Building the NanoForge web interface...',flush=True)
    env=node_environment(temporary/'node',env)
    run('npm','ci',cwd=repo/'frontend')
    run('npm','run','build',cwd=repo/'frontend')
    session=start_server(repo,home,port)
    print('Studio is ready. Choose datasets and settings in the UI, then click Start training.',flush=True)
    return session


def show_studio(session,height=1100):
    from google.colab import output
    # The supported iframe proxy keeps notebook authentication; the old new-tab helper is deprecated.
    output.serve_kernel_port_as_iframe(session.port,height=height,cache_in_notebook=False)
