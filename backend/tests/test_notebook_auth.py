import base64
import pytest
from fastapi.testclient import TestClient
from backend.main import create_app

PASSWORD='test-only-random-password-at-least-32-chars'
def header(user='nanoforge',password=PASSWORD):
    return {'Authorization':'Basic '+base64.b64encode((user+':'+password).encode()).decode()}


def test_kaggle_ui_protects_every_route_and_blocks_cross_origin_writes(tmp_path,monkeypatch):
    monkeypatch.setenv('NANOFORGE_DEVICE','cpu');monkeypatch.setenv('NANOFORGE_UI_PROFILE','kaggle-t4x2')
    monkeypatch.setenv('NANOFORGE_UI_PASSWORD',PASSWORD)
    web=tmp_path/'web';web.mkdir();(web/'index.html').write_text('private UI')
    monkeypatch.setenv('NANOFORGE_FRONTEND_DIR',str(web))
    with TestClient(create_app(tmp_path/'run')) as client:
        for route in ('/','/api/health','/api/status','/docs','/openapi.json'):
            assert client.get(route).status_code==401
            assert client.get(route,headers=header()).status_code==200
        assert client.get('/api/health',headers=header(password='wrong')).status_code==401
        assert client.get('/api/health',headers=header(user='你')).status_code==401
        info=client.get('/api/runtime',headers=header()).json()
        assert info['environment']=='kaggle' and info['transport']=='poll' and info['defaults']['multi_gpu']
        assert client.get('/api/status',headers=header()).json()['state']=='idle'
        response=client.post('/api/prompts',headers={**header(),'Origin':'https://other.example'},json={})
        assert response.status_code==403
        response=client.put('/api/prompts',headers={**header(),'Origin':'http://testserver'},json={'prompts':['a','b','c']})
        assert response.status_code==200


def test_weak_notebook_password_rejected(tmp_path,monkeypatch):
    monkeypatch.setenv('NANOFORGE_DEVICE','cpu');monkeypatch.setenv('NANOFORGE_UI_PASSWORD','weak')
    with pytest.raises(ValueError,match='at least 20 characters'):create_app(tmp_path)


def test_authenticated_server_startup_and_idle_ui(tmp_path):
    import importlib.util
    import json
    from pathlib import Path
    import socket
    from urllib.request import Request,urlopen
    from urllib.error import HTTPError
    root=Path(__file__).resolve().parents[2]
    if not (root/'frontend/dist/index.html').exists():pytest.skip('Build frontend first.')
    spec=importlib.util.spec_from_file_location('ui_start',root/'scripts/colab_ui.py')
    helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    with socket.socket() as probe:probe.bind(('127.0.0.1',0));port=probe.getsockname()[1]
    session=helper.start_server(root,tmp_path,port=port,device='cpu',profile='kaggle-t4x2',password=PASSWORD,timeout=30)
    try:
        with pytest.raises(HTTPError) as exc:urlopen(f'http://127.0.0.1:{port}/api/health')
        assert exc.value.code==401
        request=Request(f'http://127.0.0.1:{port}/api/status',headers=header())
        with urlopen(request) as response:assert json.load(response)['state']=='idle'
    finally:session.stop()
