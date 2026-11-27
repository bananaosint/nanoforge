"""Character, GPT-2 BPE, and Magic 8k adapters with portable checkpoint state."""
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
    def __init__(self, kind='char', text='', state=None, structural_tokens=()):
        self.kind = state['kind'] if state else kind
        if self.kind not in ('char', 'bpe', 'magic8k'):
            raise ValueError('Unknown tokenizer. Choose char, bpe, or magic8k.')
        if self.kind == 'magic8k':
            from tokenizers import Tokenizer as FastTokenizer
            self.tokenizer_json = state['tokenizer_json'] if state else (Path(__file__).resolve().parents[1] / 'assets/magic8k/tokenizer.json').read_text(encoding='utf-8')
            self.encoding = FastTokenizer.from_str(self.tokenizer_json)
            if structural_tokens and not state:
                self.encoding.add_special_tokens(list(structural_tokens))
                self.tokenizer_json = self.encoding.to_str()
            self.vocab_size = self.encoding.get_vocab_size(with_added_tokens=True)
        elif self.kind == 'bpe':
            self.encoding = tiktoken.get_encoding('gpt2')
            self.vocab_size = self.encoding.n_vocab
        else:
            # Reserve 0 for unknown characters; include the feedback delimiters.
            self.chars = state['chars'] if state else ['�'] + sorted(set(text + PROMPT_PREFIX + COMPLETION_PREFIX + END) - {'�'})
            self.stoi = {c: i for i, c in enumerate(self.chars)}
            self.vocab_size = len(self.chars)

    def encode(self, text):
        if self.kind == 'magic8k':
            return self.encoding.encode(text, add_special_tokens=False).ids
        if self.kind == 'bpe':
            return self.encoding.encode(text, disallowed_special=())
        return [self.stoi.get(c, 0) for c in text]

    def decode(self, ids):
        ids = [int(i) for i in ids]
        if self.kind == 'magic8k':
            return self.encoding.decode(ids, skip_special_tokens=False)
        return self.encoding.decode(ids) if self.kind == 'bpe' else ''.join(self.chars[i] for i in ids)

    def decode_token(self, token_id):
        return self.decode([token_id])

    def state(self):
        if self.kind == 'magic8k':
            return {'kind': 'magic8k', 'tokenizer_json': self.tokenizer_json, 'source': 'mjbommar/magic-bert-tokenizer-8k', 'revision': 'b00d91a6be28c97dea4d11255de3bc95dd8c6aa7'}
        return {'kind': self.kind, **({'chars': self.chars} if self.kind == 'char' else {'encoding': 'gpt2'})}
