"""Two intentionally small tokenizer adapters, fully serialized with each model."""
import os
from pathlib import Path
import tiktoken

# Bundle the unmodified GPT-2 vocabulary so first use also works offline.
os.environ.setdefault('TIKTOKEN_CACHE_DIR', str(Path(__file__).resolve().parents[1] / 'assets' / 'tokenizer'))

PROMPT_PREFIX = '<|prompt|>\n'
COMPLETION_PREFIX = '\n<|completion|>\n'
END = '\n<|end|>\n'


def format_prompt(prompt: str) -> str:
    return PROMPT_PREFIX + prompt + COMPLETION_PREFIX


class Tokenizer:
    def __init__(self, kind='char', text='', state=None):
        self.kind = state['kind'] if state else kind
        if self.kind == 'bpe':
            self.encoding = tiktoken.get_encoding('gpt2')
            self.vocab_size = self.encoding.n_vocab
        else:
            # Reserve 0 for unknown characters; include the feedback delimiters.
            self.chars = state['chars'] if state else ['�'] + sorted(set(text + PROMPT_PREFIX + COMPLETION_PREFIX + END) - {'�'})
            self.stoi = {c: i for i, c in enumerate(self.chars)}
            self.vocab_size = len(self.chars)

    def encode(self, text):
        if self.kind == 'bpe':
            return self.encoding.encode(text, disallowed_special=())
        return [self.stoi.get(c, 0) for c in text]

    def decode(self, ids):
        ids = [int(i) for i in ids]
        return self.encoding.decode(ids) if self.kind == 'bpe' else ''.join(self.chars[i] for i in ids)

    def decode_token(self, token_id):
        return self.decode([token_id])

    def state(self):
        return {'kind': self.kind, **({'chars': self.chars} if self.kind == 'char' else {'encoding': 'gpt2'})}
