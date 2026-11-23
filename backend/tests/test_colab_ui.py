"""Production UI + API, Colab defaults, polling/download contract and owned process lifecycle."""
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import socket
import sys
import tarfile
from types import ModuleType,SimpleNamespace
from urllib.request import urlopen
import pytest
import torch
from fastapi.testclient import TestClient
from backend.main import create_app

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('colab_ui',ROOT/'scripts/colab_ui.py')
launcher=importlib.util.module_from_spec(spec);spec.loader.exec_module(launcher)


def test_served_ui_defaults_idle_and_download(tmp_path,monkeypatch):
    frontend=tmp_path/'web';frontend.mkdir();(frontend/'index.html').write_text('<html>NanoForge Studio</html>')
    monkeypatch.setenv('NANOFORGE_DEVICE','cpu')
    monkeypatch.setenv('NANOFORGE_UI_PROFILE','colab-l4')
    monkeypatch.setenv('NANOFORGE_FRONTEND_DIR',str(frontend))
    app=create_app(tmp_path/'run')
    with TestClient(app) as client:
        assert 'NanoForge Studio' in client.get('/').text
        info=client.get('/api/runtime').json()
        assert info['environment']=='colab' and info['transport']=='poll'
        assert info['defaults']['preset']=='Mini' and info['defaults']['model']['block_size']==512
        assert info['defaults']['precision']=='float16' and info['defaults']['allow_tf32']
        assert info['defaults']['datasets']==[{'name':'starter-sample.txt','weight':100}]
        assert client.get('/api/status').json()['state']=='idle'
        assert client.get('/api/runs').json()==[]
        assert app.state.trainer.model is None and not app.state.imports.status()['active']
        client.post('/api/datasets/upload',files={'file':('dolly.jsonl',b'{"prompt":"hi","completion":"hello"}\n')})
        assert client.get('/api/runtime').json()['defaults']['datasets'][0]['name']=='dolly.jsonl'
        tiny={'preset':'Custom','tokenizer':'char','precision':'float32','allow_tf32':True,
              'model':{'n_layer':1,'n_head':2,'n_embd':16,'block_size':32,'dropout':0.0},
              'datasets':[{'name':'starter-sample.txt','weight':100}]}
        assert client.post('/api/runs',json=tiny).status_code==200
        assert client.post('/api/training/start',json={'precision':'float32','max_steps':2,'batch_size':1,'eval_interval':2}).status_code==200
        app.state.trainer.thread.join(30)
        status=client.get('/api/status').json()
        assert status['state']=='completed' and status['step']==2
        response=client.get('/api/checkpoints/'+status['checkpoint']+'/download')
        assert response.status_code==200 and status['checkpoint'] in response.headers['content-disposition']
        payload=torch.load(io.BytesIO(response.content),map_location='cpu',weights_only=True)
        assert payload['step']==2
        assert all(t.device.type=='cpu' and t.dtype==torch.float32 for t in payload['model_state_dict'].values())
        assert client.get('/api/checkpoints/missing.pt/download').status_code==400
        assert client.get('/api/checkpoints/..%2Fnanoforge.sqlite3/download').status_code in (400,404)
        assert client.get('/api/does-not-exist').status_code==404
        assert client.get('/%2e%2e/nanoforge.sqlite3').status_code==404


def test_local_runtime_still_uses_websocket(tmp_path,monkeypatch):
    monkeypatch.setenv('NANOFORGE_DEVICE','cpu');monkeypatch.delenv('NANOFORGE_UI_PROFILE',raising=False)
    with TestClient(create_app(tmp_path)) as client:
        assert client.get('/api/runtime').json()=={'environment':'local','label':'Local','transport':'websocket','defaults':None}
        with client.websocket_connect('/ws/training') as ws:assert ws.receive_json()['state']=='idle'


def test_launcher_uses_notebook_iframe_without_public_tunnel(monkeypatch):
    calls=[]
    colab=ModuleType('google.colab')
    colab.output=SimpleNamespace(serve_kernel_port_as_iframe=lambda *a,**kw:calls.append((a,kw)))
    monkeypatch.setitem(sys.modules,'google.colab',colab)
    launcher.show_studio(SimpleNamespace(port=8123))
    assert calls==[((8123,),{'height':1100,'cache_in_notebook':False})]


def test_server_process_starts_idle_and_restarts_without_killing_others(tmp_path):
    if not (ROOT/'frontend/dist/index.html').exists():pytest.skip('Build frontend before production server integration test.')
    with socket.socket() as probe:probe.bind(('127.0.0.1',0));port=probe.getsockname()[1]
    session=launcher.start_server(ROOT,tmp_path,port=port,device='cpu',timeout=30)
    try:
        with urlopen(f'http://127.0.0.1:{port}/api/health') as response:health=json.load(response)
        assert health['pid']==session.process.pid and health['device']=='cpu'
        with urlopen(f'http://127.0.0.1:{port}/api/status') as response:assert json.load(response)['state']=='idle'
        with urlopen(f'http://127.0.0.1:{port}/') as response:assert b'<div id="root">' in response.read()
        with pytest.raises(ValueError,match='already in use'):launcher.start_server(ROOT,tmp_path,port=port,device='cpu')
        assert session.process.poll() is None
    finally:session.stop()
    assert session.process.poll() is not None
    restarted=launcher.start_server(ROOT,tmp_path,port=port,device='cpu',timeout=30)
    restarted.stop()


def test_node_download_checksum_and_safe_extraction(tmp_path,monkeypatch):
    package=io.BytesIO()
    with tarfile.open(fileobj=package,mode='w:xz') as archive:
        item=tarfile.TarInfo('node-v22.16.0-linux-x64/bin/node');content=b'node placeholder';item.size=len(content)
        archive.addfile(item,io.BytesIO(content))
    data=package.getvalue()
    checksum=hashlib.sha256(data).hexdigest()
    manifest=f'{checksum}  node-v22.16.0-linux-x64.tar.xz\n'.encode()
    monkeypatch.setattr(launcher.shutil,'which',lambda *a,**kw:None)
    monkeypatch.setattr(launcher.platform,'system',lambda:'Linux')
    monkeypatch.setattr(launcher.platform,'machine',lambda:'x86_64')
    monkeypatch.setattr(launcher,'urlopen',lambda url,**kw:io.BytesIO(manifest if url.endswith('.txt') else data))
    env=launcher.node_environment(tmp_path,{'PATH':'/original'})
    assert env['PATH'].startswith(str(tmp_path/'node-v22.16.0-linux-x64/bin'))
    assert not list(tmp_path.glob('*.tar.xz'))
    monkeypatch.setattr(launcher,'urlopen',lambda url,**kw:io.BytesIO(manifest if url.endswith('.txt') else b'corrupt'))
    with pytest.raises(ValueError,match='checksum mismatch'):launcher.node_environment(tmp_path/'corrupt',{'PATH':''})
