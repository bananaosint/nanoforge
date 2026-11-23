"""Custom sizing must agree with real Torch models and reject oversize runs early."""
from dataclasses import asdict, replace
from types import SimpleNamespace
from unittest.mock import patch
import pytest
import torch
from fastapi.testclient import TestClient
from backend.main import create_app
from backend.ml.model import GPT, ModelConfig, MAX_PARAMETERS, parameter_count, resolve_model
from backend.ml.corpus import FileCorpus


@pytest.mark.parametrize('dimensions', [
    dict(n_layer=1,n_head=1,n_embd=1,block_size=8),
    dict(n_layer=13,n_head=4,n_embd=16,block_size=1024),
    dict(n_layer=24,n_head=16,n_embd=1024,block_size=2048),
])
def test_exact_count_matches_unique_torch_parameters(dimensions):
    config=ModelConfig(vocab_size=8192,**dimensions)
    with torch.device('meta'):
        model=GPT(config)
    assert parameter_count(config)==sum(p.numel() for p in model.parameters())
    assert model.token_embedding.weight is model.lm_head.weight


def test_strict_billion_boundary_and_dimensions():
    # Width one lets the vocabulary express the exact parameter boundary.
    config=ModelConfig(vocab_size=999_999_964,n_layer=1,n_head=1,n_embd=1,block_size=8)
    assert parameter_count(config)==MAX_PARAMETERS
    with pytest.raises(ValueError,match='1,000,000,000'):
        replace(config,vocab_size=config.vocab_size+1)
    with pytest.raises(ValueError,match='positive'):
        replace(config,n_head=0)
    with pytest.raises(ValueError,match='divisible'):
        ModelConfig(vocab_size=10,n_head=3,n_embd=32)


@pytest.mark.parametrize('budget', [1_000_000, 500_000_000, MAX_PARAMETERS])
def test_fit_is_largest_width_within_budget(budget):
    config=resolve_model('Custom',dict(n_layer=12,n_head=8,block_size=1024),8192,budget)
    assert config.n_embd % config.n_head == 0
    assert parameter_count(config)<=budget
    larger=SimpleNamespace(**{**asdict(config),'n_embd':config.n_embd+config.n_head})
    assert parameter_count(larger)>budget
    with pytest.raises(ValueError,match='too small'):
        resolve_model('Custom',dict(n_layer=12,n_head=8),8192,1)


def test_preview_no_allocation_and_create_cap_before_allocation(tmp_path,monkeypatch):
    monkeypatch.setenv('NANOFORGE_DEVICE','cpu')
    app=create_app(tmp_path)
    with TestClient(app) as client:
        body=dict(preset='Custom',tokenizer='magic8k',model=dict(n_layer=24,n_head=16,n_embd=1024,block_size=2048))
        with patch('backend.ml.trainer.GPT',side_effect=AssertionError('Must not allocate weights')):
            preview=client.post('/api/models/preview',json=body)
            assert preview.status_code==200,preview.text
            assert preview.json()['parameters']>300_000_000
            assert preview.json()['model_config']['vocab_size']==8192
            assert app.state.trainer.model is None
            rejected=client.post('/api/runs',json={**body,'model':{**body['model'],'n_layer':100}})
            assert rejected.status_code==400
            assert '1,000,000,000' in rejected.json()['detail']
        fitted=client.post('/api/models/preview',json={**body,'target_parameters':500_000_000})
        assert fitted.status_code==200,fitted.text
        assert fitted.json()['parameters']<=500_000_000
        assert client.post('/api/models/preview',json={**body,'target_parameters':1_000_000_000}).status_code==422
        assert client.post('/api/models/preview',json={'preset':'Micro','model':{'n_head':3}}).status_code==400
        tiny={**body,'tokenizer':'char','model':dict(n_layer=13,n_head=1,n_embd=8,block_size=8)}
        expected=client.post('/api/models/preview',json=tiny).json()
        created=client.post('/api/runs',json=tiny)
        assert created.status_code==200,created.text
        assert created.json()['parameters']==expected['parameters']
        assert created.json()['model_config']==expected['model_config']
        before=app.state.trainer.model
        rejected=client.post('/api/runs',json={**body,'model':{**body['model'],'n_layer':100}})
        assert rejected.status_code==400
        assert app.state.trainer.model is before
        assert app.state.trainer.config['preset']=='Custom'


def test_character_preview_updates_after_dataset_changes(tmp_path):
    path=tmp_path/'vocab.txt'
    path.write_text('aaa',encoding='utf-8')
    assert FileCorpus(path).vocabulary()=={'a'}
    path.write_text('aaa你🌍',encoding='utf-8')
    assert FileCorpus(path).vocabulary()=={'a','你','🌍'}
