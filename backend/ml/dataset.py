"""Local corpora and a source-weighted 90/10 autoregressive batch sampler."""
import json
import random
from pathlib import Path
from urllib.request import urlopen
import torch
import numpy as np
from .corpus import FileCorpus
from .tokenizer import format_prompt, END

FALLBACK = ('First Citizen:\nBefore we proceed any further, hear me speak.\n'
            'All:\nSpeak, speak.\nFirst Citizen:\nYou are all resolved rather to die than to famish?\n'
            'All:\nResolved. Resolved.\n' * 40)
SHAKESPEARE_URL = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'


class DatasetManager:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        sample = self.directory / 'starter-sample.txt'
        if not sample.exists():
            sample.write_text(FALLBACK)

    def path(self, name):
        if Path(name).name != name or Path(name).suffix.lower() not in ('.txt', '.jsonl'):
            raise ValueError('Use a local .txt or .jsonl dataset filename.')
        path = self.directory / name
        if not path.is_file():
            raise ValueError('Dataset not found.')
        return path

    def text(self, name):
        raw = self.path(name).read_text(encoding='utf-8')
        if Path(name).suffix.lower() == '.txt':
            return raw
        texts = []
        for number, line in enumerate(raw.splitlines(), 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
                if isinstance(item, str):
                    texts.append(item)
                elif isinstance(item, dict) and isinstance(item.get('text'), str):
                    texts.append(item['text'])
                elif isinstance(item, dict) and isinstance(item.get('prompt'), str):
                    target = item.get('corrected_output', item.get('completion', item.get('target')))
                    if not isinstance(target, str):
                        raise ValueError('Missing completion or corrected_output')
                    texts.append(format_prompt(item['prompt']) + target + END)
                else:
                    raise ValueError('Expected text or a prompt/completion pair')
            except (ValueError, TypeError) as exc:
                raise ValueError(f'Invalid JSONL at line {number}: {exc}') from exc
        return '\n'.join(texts)

    def list(self):
        return [{'name': p.name, 'bytes': p.stat().st_size} for p in sorted(self.directory.iterdir())
                if p.is_file() and p.suffix.lower() in ('.txt', '.jsonl')]

    def download(self):
        fallback = False
        try:
            with urlopen(SHAKESPEARE_URL, timeout=15) as response:
                text = response.read(8_000_000).decode('utf-8')
            if not text.strip():
                raise ValueError('Empty download')
        except Exception:
            text, fallback = FALLBACK, True
        (self.directory / 'tinyshakespeare.txt').write_text(text, encoding='utf-8')
        return {'name': 'tinyshakespeare.txt', 'offline_fallback': fallback}


class WeightedMixer:
    def __init__(self, texts, tokenizer, block_size, device):
        self.sources = []
        self.block_size, self.device = block_size, device
        for text, weight in texts:
            if weight <= 0:
                continue
            ids = text.tokens(tokenizer) if isinstance(text, FileCorpus) else torch.tensor(tokenizer.encode(text), dtype=torch.long)
            split = int(len(ids) * 0.9)
            if min(split, len(ids) - split) < 2:
                raise ValueError('Every selected dataset needs at least 20 tokens for a 90/10 split.')
            self.sources.append((ids[:split], ids[split:], weight))
        if not self.sources:
            raise ValueError('Select at least one nonempty dataset with a positive weight.')

    def batch(self, batch_size, validation=False):
        xs, ys = [], []
        sources = random.choices(self.sources, weights=[s[2] for s in self.sources], k=batch_size)
        # A common sequence length lets short corpora coexist with longer ones.
        length = min(self.block_size, min(len(s[int(validation)]) - 1 for s in sources))
        for source in sources:
            ids = source[int(validation)]
            start = random.randrange(len(ids) - length)
            xs.append(torch.tensor(np.asarray(ids[start:start + length]), dtype=torch.long))
            ys.append(torch.tensor(np.asarray(ids[start + 1:start + length + 1]), dtype=torch.long))
        return torch.stack(xs).to(self.device), torch.stack(ys).to(self.device)


def completion_batch(tokenizer, prompt, completion, block_size, device):
    """Preserve completion tokens when truncating; -100 masks all prompt targets."""
    prefix = tokenizer.encode(format_prompt(prompt))
    target = tokenizer.encode(completion + END)
    target = target[:block_size]
    prefix = prefix[-max(1, block_size + 1 - len(target)):]
    ids = (prefix + target)[-(block_size + 1):]
    boundary = len(ids) - len(target)
    x = torch.tensor([ids[:-1]], device=device)
    y = torch.tensor([ids[1:]], device=device)
    y[:, :max(0, boundary - 1)] = -100
    return x, y


def completion_log_probability(model, tokenizer, prompt, completion, device):
    x, y = completion_batch(tokenizer, prompt, completion, model.config.block_size, device)
    logits = model(x)['logits'].log_softmax(-1)
    mask = y != -100
    # DPO uses the sum of completion token log probabilities, not a mean.
    selected = logits.gather(-1, y.clamp_min(0).unsqueeze(-1)).squeeze(-1)
    return (selected * mask).sum(-1)
