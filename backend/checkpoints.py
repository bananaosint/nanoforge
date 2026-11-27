"""Register portable NanoForge .pt files, including Colab CUDA/AMP exports."""
import json
import math
import shutil
from pathlib import Path
import torch
from .ml.model import GPT, ModelConfig
from .ml.tokenizer import Tokenizer
from .storage import now


def register_checkpoint(source, directory, name=None, expected_mode=None):
    source=Path(source)
    directory=Path(directory)
    name=name or source.name
    if Path(name).name!=name or '\\' in name or not name.endswith('.pt') or name.startswith('.'):
        raise ValueError('Use a plain .pt checkpoint filename.')
    target=directory/name
    if target.exists() and target.resolve()!=source.resolve():
        raise ValueError('A checkpoint with that filename already exists. Rename the upload first.')
    try:
        payload=torch.load(source,map_location='cpu',weights_only=True)
        required={'model_config','tokenizer_state','model_state_dict','optimizer_state_dict','step','val_loss','version_tag'}
        if not isinstance(payload,dict) or not required <= payload.keys():
            raise ValueError('Expected a complete NanoForge checkpoint, not a bare state_dict.')
        if expected_mode is not None and (payload.get('mode') == 'chess') != (expected_mode == 'chess'):
            raise ValueError('Choose a checkpoint from the current studio mode.')
        if payload.get('checkpoint_format_version',1)!=1:
            raise ValueError('Unsupported NanoForge checkpoint format version.')
        config=ModelConfig(**payload['model_config'])
        tokenizer=Tokenizer(state=payload['tokenizer_state'])
        if config.vocab_size!=tokenizer.vocab_size:
            raise ValueError('Checkpoint tokenizer and model vocabulary sizes differ.')
        with torch.device('meta'):
            model=GPT(config)
        expected=model.state_dict()
        actual=payload['model_state_dict']
        if expected.keys()!=actual.keys() or any(not torch.is_tensor(actual[key]) or actual[key].shape!=tensor.shape for key,tensor in expected.items()):
            raise ValueError('Checkpoint weight names or shapes do not match the model configuration.')
        if not isinstance(payload['optimizer_state_dict'],dict) or not isinstance(payload['step'],int) or payload['step']<0:
            raise ValueError('Invalid optimizer state or training step.')
        if payload['val_loss'] is not None and not math.isfinite(float(payload['val_loss'])):
            raise ValueError('Validation loss is non-finite.')
        metadata={key:payload.get(key) for key in ('model_config','step','val_loss','version_tag','mode','run_id','precision')}
        metadata.update(name=name,created_at=now(),parameters=sum(parameter.numel() for parameter in model.parameters()))
    except Exception as exc:
        if isinstance(exc,ValueError):raise
        raise ValueError('Could not read this NanoForge checkpoint: '+str(exc)) from exc
    directory.mkdir(parents=True,exist_ok=True)
    if target.resolve()!=source.resolve():
        pending=target.with_suffix('.pt.pending')
        shutil.copyfile(source,pending)
        pending.replace(target)
    sidecar=target.with_suffix('.json')
    pending=sidecar.with_suffix('.json.pending')
    pending.write_text(json.dumps(metadata),encoding='utf-8')
    pending.replace(sidecar)
    return metadata
