"""A decoder-only GPT, with every architectural component visible in PyTorch."""
from dataclasses import dataclass
from types import SimpleNamespace
import math
import torch
from torch import nn
from torch.nn import functional as F

PRESETS = {
    'Nano': dict(n_layer=4, n_head=4, n_embd=128, block_size=128, dropout=0.1),
    'Micro': dict(n_layer=6, n_head=4, n_embd=256, block_size=256, dropout=0.1),
    'Mini': dict(n_layer=8, n_head=6, n_embd=384, block_size=256, dropout=0.1),
}
MAX_PARAMETERS = 999_999_999


def parameter_count(config):
    """Exact unique parameter count, including biases and tied token embeddings."""
    width = config.n_embd
    return ((config.vocab_size + config.block_size) * width
            + config.n_layer * (12 * width * width + 13 * width) + 2 * width)


def resolve_model(preset, dimensions, vocab_size, target_parameters=None):
    if preset not in (*PRESETS, 'Custom'):
        raise ValueError('Unknown model preset.')
    values = {**PRESETS.get(preset, PRESETS['Nano']), **dimensions, 'vocab_size': vocab_size}
    if target_parameters is not None:
        if not 1 <= target_parameters <= MAX_PARAMETERS:
            raise ValueError('Parameter budget must be between 1 and 999,999,999.')
        # Layers, heads and context stay fixed; fit the largest whole head width.
        heads = values['n_head']
        if heads < 1:
            raise ValueError('The number of heads must be positive.')
        low, high, best = 1, target_parameters // heads, None
        while low <= high:
            middle = (low + high) // 2
            values['n_embd'] = middle * heads
            # Check the arithmetic before constructing a validated configuration.
            count = parameter_count(SimpleNamespace(**values))
            if count <= target_parameters:
                best, low = values['n_embd'], middle + 1
            else:
                high = middle - 1
        if best is None:
            raise ValueError('This budget is too small for the selected layers, heads, context and tokenizer.')
        values['n_embd'] = best
    return ModelConfig(**values)


def detect_device():
    """Select a usable accelerator; an explicit request never silently falls back."""
    import os
    requested = os.environ.get('NANOFORGE_DEVICE', 'auto').lower()
    available = {
        'cuda': lambda: torch.cuda.is_available(),
        'xpu': lambda: hasattr(torch, 'xpu') and torch.xpu.is_available(),
        'mps': lambda: torch.backends.mps.is_available(),
        'cpu': lambda: True,
    }
    if requested != 'auto':
        if requested not in available:
            raise ValueError('NANOFORGE_DEVICE must be auto, cuda, xpu, mps, or cpu.')
        if not available[requested]():
            raise ValueError(f'Requested {requested.upper()} is unavailable. For Intel GPUs, install the XPU PyTorch build and verify the Intel driver/WSL GPU support.')
        return requested
    for device in ('cuda', 'xpu', 'mps', 'cpu'):
        if available[device]():
            return device


def device_name(device):
    kind = str(device).split(':')[0]
    if kind in ('cuda', 'xpu'):
        return getattr(torch, kind).get_device_name(0)
    return 'Apple Metal' if kind == 'mps' else 'CPU'


@dataclass
class ModelConfig:
    vocab_size: int
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    block_size: int = 128
    dropout: float = 0.1

    def __post_init__(self):
        if self.vocab_size < 2 or min(self.n_layer, self.n_head, self.n_embd, self.block_size) < 1:
            raise ValueError('Model dimensions must be positive, with at least two tokens.')
        if self.n_embd % self.n_head:
            raise ValueError('Embedding width must be divisible by the number of heads.')
        if not 0 <= self.dropout <= 0.8:
            raise ValueError('Dropout must be between 0 and 0.8.')
        count = parameter_count(self)
        if count > MAX_PARAMETERS:
            raise ValueError(f'Model has {count:,} parameters. Choose fewer than 1,000,000,000.')


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.heads = config.n_head
        self.dropout = config.dropout
        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self, x, inspect=False):
        batch, time, width = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q, k, v = [t.view(batch, time, self.heads, width // self.heads).transpose(1, 2) for t in (q, k, v)]
        attention = None
        if inspect or x.device.type == 'xla':
            # Explicit scores expose [batch, head, query, key] for the inspector.
            scores = q @ k.transpose(-2, -1) / math.sqrt(k.size(-1))
            mask = torch.ones(time, time, device=x.device, dtype=torch.bool).tril()
            attention = scores.float().masked_fill(~mask, float('-inf')).softmax(dim=-1)
            values = F.dropout(attention, self.dropout, self.training).to(v.dtype) @ v
            if not inspect:
                attention = None
        else:
            values = F.scaled_dot_product_attention(q, k, v, is_causal=True,
                        dropout_p=self.dropout if self.training else 0.0)
        values = values.transpose(1, 2).contiguous().view(batch, time, width)
        return self.resid_dropout(self.proj(values)), attention


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.mlp = nn.Sequential(nn.Linear(config.n_embd, 4 * config.n_embd), nn.GELU(),
                                 nn.Linear(4 * config.n_embd, config.n_embd), nn.Dropout(config.dropout))

    def forward(self, x, inspect=False):
        update, attention = self.attn(self.ln1(x), inspect)
        x = x + update
        return x + self.mlp(self.ln2(x)), attention


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding = nn.Embedding(config.block_size, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.norm = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight  # One shared token matrix.
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)

    def forward(self, ids, targets=None, inspect=False):
        time = ids.size(1)
        if time > self.config.block_size:
            raise ValueError('Input exceeds the context window.')
        positions = torch.arange(time, device=ids.device)
        x = self.dropout(self.token_embedding(ids) + self.position_embedding(positions))
        attentions = []
        for block in self.blocks:
            x, weights = block(x, inspect)
            if inspect:
                attentions.append(weights)
        logits = self.lm_head(self.norm(x))
        loss = None if targets is None else F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-100)
        return {'logits': logits, 'loss': loss, 'attentions': attentions}

    @torch.no_grad()
    def generate(self, ids, max_new_tokens=64, temperature=0.8, top_k=40, telemetry=False, seed=42):
        if ids.size(1) == 0:
            raise ValueError('Enter a nonempty prompt.')
        was_training = self.training
        self.eval()
        generator = torch.Generator(device='cpu').manual_seed(seed)
        steps = []
        try:
            for _ in range(max_new_tokens):
                context = ids[:, -self.config.block_size:]
                logits = self(context)['logits'][:, -1, :].float() / max(temperature, 0.01)
                if top_k > 0:
                    cutoff = logits.topk(min(top_k, logits.size(-1))).values[:, -1:]
                    logits = logits.masked_fill(logits < cutoff, float('-inf'))
                probabilities = logits.softmax(-1)
                token = torch.multinomial(probabilities.cpu(), 1, generator=generator).to(ids.device)
                if telemetry:
                    values, indices = probabilities[0].topk(min(5, probabilities.size(-1)))
                    steps.append({'token_id': token.item(), 'top5': [
                        {'id': i, 'probability': p} for i, p in zip(indices.tolist(), values.tolist())]})
                ids = torch.cat([ids, token], dim=1)
            # Recompute on the final sequence: includes the final generated token.
            context = ids[:, -self.config.block_size:]
            attention = self(context, inspect=True)['attentions'] if telemetry else []
            return {'ids': ids, 'steps': steps, 'attention': attention, 'context_ids': context[0].tolist()}
        finally:
            self.train(was_training)
