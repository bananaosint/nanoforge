import ast
import importlib.util
import json
from pathlib import Path
import pytest
from fastapi.testclient import TestClient
from backend.main import create_app
from backend.api.schemas import NewRun

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('colab_start',ROOT/'scripts/colab_ui.py')
helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)


def test_a100_ui_defaults_and_idle_start(tmp_path,monkeypatch):
    monkeypatch.setenv('NANOFORGE_DEVICE','cpu');monkeypatch.setenv('NANOFORGE_UI_PROFILE','colab-a100')
    app=create_app(tmp_path)
    with TestClient(app) as client:
        info=client.get('/api/runtime').json()
        assert info['label']=='Colab A100' and info['environment']=='colab' and info['transport']=='poll'
        config=NewRun.model_validate(info['defaults'])
        assert config.precision=='float16' and config.allow_tf32 and not config.multi_gpu
        assert config.model.block_size==512 and config.batch_size==64 and config.gradient_accumulation==1
        assert config.datasets[0].name=='starter-sample.txt'
        assert client.get('/api/status').json()['state']=='idle'
        assert app.state.trainer.model is None and not app.state.imports.status()['active']


def test_gpu_checks_fail_for_wrong_runtime_without_installing():
    from types import SimpleNamespace
    import sys
    from unittest.mock import patch
    gpu=SimpleNamespace(is_available=lambda:True,get_device_name=lambda index:'NVIDIA L4',
                        get_device_capability=lambda index:(8,9),get_device_properties=lambda index:SimpleNamespace(total_memory=24*1024**3))
    fake=SimpleNamespace(cuda=gpu,__version__='test',version=SimpleNamespace(cuda='test'))
    with patch.dict(sys.modules,{'torch':fake}):
        with pytest.raises(AssertionError,match='Expected A100'):exec(helper.gpu_check('colab-a100'),{})
        exec(helper.gpu_check('colab-l4'),{})
        gpu.get_device_name=lambda index:'NVIDIA A100-SXM4-40GB'
        exec(helper.gpu_check('colab-a100'),{})
    with pytest.raises(ValueError):helper.gpu_check('unknown')


def test_a100_notebook_launches_ui_not_training():
    nb=json.loads((ROOT/'notebooks/NanoForge_A100.ipynb').read_text())
    assert nb['metadata']['colab']['gpuType']=='A100'
    code=''.join(''.join(cell['source']) for cell in nb['cells'] if cell['cell_type']=='code')
    ast.parse(code)
    assert "profile='colab-a100'" in code and "startup['show_studio']" in code
    assert 'run_l4' not in code
