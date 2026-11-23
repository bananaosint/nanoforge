from types import SimpleNamespace
import pytest
import torch
from backend.ml.model import detect_device


def test_auto_detect_selects_intel_xpu(monkeypatch):
    monkeypatch.delenv('NANOFORGE_DEVICE',raising=False)
    monkeypatch.setattr(torch.cuda,'is_available',lambda:False)
    monkeypatch.setattr(torch,'xpu',SimpleNamespace(is_available=lambda:True))
    assert detect_device()=='xpu'


def test_explicit_xpu_does_not_fall_back_to_cpu(monkeypatch):
    monkeypatch.setenv('NANOFORGE_DEVICE','xpu')
    monkeypatch.setattr(torch,'xpu',SimpleNamespace(is_available=lambda:False))
    with pytest.raises(ValueError,match='XPU is unavailable'):
        detect_device()


def test_explicit_cpu_and_older_pytorch_without_xpu(monkeypatch):
    monkeypatch.setenv('NANOFORGE_DEVICE','cpu')
    assert detect_device()=='cpu'
    monkeypatch.setenv('NANOFORGE_DEVICE','auto')
    monkeypatch.delattr(torch,'xpu',raising=False)
    monkeypatch.setattr(torch.cuda,'is_available',lambda:False)
    monkeypatch.setattr(torch.backends.mps,'is_available',lambda:False)
    assert detect_device()=='cpu'
