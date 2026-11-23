"""Verify the actual accelerator with a real Transformer backward/optimizer step."""
import argparse
import json
import os
import sys
import torch
from .ml.model import GPT, ModelConfig, detect_device, device_name


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', choices=['auto','cpu','cuda','xpu','mps'], default='auto')
    args = parser.parse_args()
    os.environ['NANOFORGE_DEVICE'] = args.device
    try:
        device = detect_device()
        model = GPT(ModelConfig(vocab_size=32,n_layer=1,n_head=2,n_embd=32,block_size=16,dropout=0)).to(device)
        optimizer = torch.optim.AdamW(model.parameters(),lr=0.001,foreach=False)
        tokens = torch.arange(16,device=device).unsqueeze(0)
        loss = model(tokens,tokens)['loss']
        loss.backward()
        optimizer.step()
        model.eval()
        result = model.generate(tokens,max_new_tokens=2,telemetry=True)
        if device in ('cuda','xpu'):
            getattr(torch,device).synchronize()
        assert str(next(model.parameters()).device).split(':')[0] == device
        print(json.dumps({'ok':True,'device':device,'device_name':device_name(device),'torch_version':torch.__version__,'loss':float(loss.detach().cpu()),'generation_steps':len(result['steps'])}))
    except Exception as exc:
        print(json.dumps({'ok':False,'requested':args.device,'error':str(exc),'torch_version':torch.__version__,'wsl_dxg_present':os.path.exists('/dev/dxg')}),file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
