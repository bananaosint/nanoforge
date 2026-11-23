"""Streaming UTF-8 sources and reusable, memory-mapped token caches."""
import hashlib
import json
import os
from array import array
from functools import lru_cache
from pathlib import Path
import numpy as np
from .tokenizer import format_prompt, END


def record_text(item):
    if isinstance(item, str):
        return item
    if isinstance(item, dict) and isinstance(item.get('text'), str):
        return item['text']
    if isinstance(item, dict) and isinstance(item.get('prompt'), str):
        target = item.get('corrected_output', item.get('completion', item.get('target')))
        if isinstance(target, str):
            return format_prompt(item['prompt']) + target + END
    raise ValueError('Expected text or a prompt/completion pair.')


class FileCorpus:
    def __init__(self, path):
        self.path = Path(path)

    def chunks(self):
        with self.path.open(encoding='utf-8-sig') as file:
            if self.path.suffix.lower() == '.txt':
                while text := file.read(65536):
                    yield text
            else:
                for number, line in enumerate(file, 1):
                    if line.strip():
                        try:
                            yield record_text(json.loads(line)) + '\n'
                        except (ValueError, TypeError) as exc:
                            raise ValueError(f'{self.path.name}, line {number}: {exc}') from exc

    def vocabulary(self):
        stat = self.path.stat()
        return file_vocabulary(str(self.path.resolve()), stat.st_size, stat.st_mtime_ns)

    def preview(self, limit=5000):
        result = ''
        for chunk in self.chunks():
            result += chunk[:limit - len(result)]
            if len(result) >= limit:
                break
        return result

    def tokens(self, tokenizer):
        stat = self.path.stat()
        identity = json.dumps([str(self.path.resolve()), stat.st_size, stat.st_mtime_ns, tokenizer.state()], sort_keys=True)
        use_u16 = tokenizer.vocab_size <= 65536
        dtype = np.uint16 if use_u16 else np.uint32
        suffix = '.u16' if use_u16 else '.u32'
        key = hashlib.sha256(('packed-v2:' + identity).encode()).hexdigest()
        directory = self.path.parent / '.tokens'
        directory.mkdir(exist_ok=True)
        path = directory / (key + suffix)
        if not path.exists():
            temporary = path.with_suffix('.partial')
            with temporary.open('wb') as output:
                for chunk in self.chunks():
                    # BPE is encoded in bounded chunks. Boundaries may split a merge,
                    # but no text is dropped and character tokenization stays exact.
                    ids = tokenizer.encode(chunk)
                    if ids and (min(ids) < 0 or max(ids) >= tokenizer.vocab_size):
                        raise ValueError('Tokenizer produced an ID outside its vocabulary.')
                    array('H' if use_u16 else 'I', ids).tofile(output)
            temporary.replace(path)
        if path.stat().st_size == 0:
            raise ValueError(f'Dataset {self.path.name} is empty.')
        return np.memmap(path, dtype=dtype, mode='r')


@lru_cache(maxsize=32)
def file_vocabulary(path, size, modified):
    """Reuse vocabulary scans until a dataset changes (no token/model allocation)."""
    chars = set()
    for chunk in FileCorpus(Path(path)).chunks():
        chars.update(chunk)
    return frozenset(chars)


def vocabulary_text(sources):
    chars = set()
    for text, _ in sources:
        chars.update(text.vocabulary() if isinstance(text, FileCorpus) else text)
    return ''.join(sorted(chars))


class PairPool:
    """Random-access feedback without loading large imported JSONL files into RAM."""
    def __init__(self, feedback, corpora, mode):
        self.feedback = feedback
        self.parts = []
        self.mode = mode
        self.total = len(feedback)
        for corpus in corpora:
            if corpus.path.suffix.lower() != '.jsonl':
                continue
            stat = corpus.path.stat()
            key = hashlib.sha256(f'{corpus.path.resolve()}:{stat.st_size}:{stat.st_mtime_ns}:{mode}'.encode()).hexdigest()
            folder = corpus.path.parent / '.tokens'
            folder.mkdir(exist_ok=True)
            index = folder / (key + '.offsets')
            if not index.exists():
                temp = index.with_suffix('.partial')
                with corpus.path.open('rb') as source, temp.open('wb') as target:
                    while True:
                        offset = source.tell()
                        line = source.readline()
                        if not line:
                            break
                        if not line.strip():
                            continue
                        row = json.loads(line)
                        if self.convert(row):
                            array('Q', [offset]).tofile(target)
                temp.replace(index)
            if index.stat().st_size:
                offsets = np.memmap(index, dtype=np.uint64, mode='r')
                self.parts.append((corpus.path, offsets))
                self.total += len(offsets)

    def convert(self, row):
        if not isinstance(row, dict) or not row.get('prompt'):
            return None
        if self.mode == 'sft':
            target = row.get('corrected_output', row.get('completion'))
            if target:
                return {'prompt': row['prompt'], 'corrected_output': target}
        elif row.get('chosen') and row.get('rejected'):
            return {'prompt': row['prompt'], 'response_a': row['chosen'], 'response_b': row['rejected'], 'winner': 'a'}
        return None

    def __len__(self):
        return self.total

    def __getitem__(self, index):
        if index < len(self.feedback):
            return self.feedback[index]
        index -= len(self.feedback)
        for path, offsets in self.parts:
            if index < len(offsets):
                with path.open('rb') as file:
                    file.seek(int(offsets[index]))
                    return self.convert(json.loads(file.readline()))
            index -= len(offsets)
        raise IndexError(index)
