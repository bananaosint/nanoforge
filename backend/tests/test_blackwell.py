import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from fastapi.testclient import TestClient
from backend.main import create_app

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('blackwell',ROOT/'scripts/blackwell.py')
helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)


def test_stack_repair_only_for_software_failure():
    calls=[];results=iter([3,0,0])
    def run(command,**kwargs):calls.append(command);return SimpleNamespace(returncode=next(results))
    helper.ensure_blackwell(ROOT,{},run=run)
    assert len(calls)==3 and 'torch==2.10.0' in calls[1] and 'https://download.pytorch.org/whl/cu128' in calls[1]
    calls=[]
    def wrong_gpu(command,**kw):calls.append(command);return SimpleNamespace(returncode=2)
    with pytest.raises(RuntimeError,match='No matching G4'):helper.ensure_blackwell(ROOT,{},run=wrong_gpu)
    assert len(calls)==1


def test_g4_ui_profile_moe_preview_and_total_cap(tmp_path,monkeypatch):
    monkeypatch.setenv('NANOFORGE_DEVICE','cpu');monkeypatch.setenv('NANOFORGE_UI_PROFILE','colab-g4')
    with TestClient(create_app(tmp_path)) as client:
        info=client.get('/api/runtime').json()
        assert info['label']=='Colab G4 Blackwell' and info['defaults']['precision']=='bfloat16'
        assert client.get('/api/status').json()['state']=='idle'
        body=dict(preset='Custom',tokenizer='magic8k',model=dict(architecture='moe',n_layer=2,n_head=4,n_embd=128,n_experts=4,experts_per_token=2))
        response=client.post('/api/models/preview',json=body)
        assert response.status_code==200,response.text
        assert response.json()['active_parameters']<response.json()['parameters']
        assert client.post('/api/models/preview',json={**body,'target_parameters':20_000_000}).json()['parameters']<=20_000_000
