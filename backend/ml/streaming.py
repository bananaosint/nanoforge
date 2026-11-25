"""Pinned Hugging Face pretraining streams with resumable, bounded token buffers."""
from collections.abc import Iterator
import queue
import random
import re
import threading
from urllib.parse import quote
from urllib.request import urlopen

import numpy as np
import pyarrow.parquet as parquet
import torch

from ..importers.hub import BY_ID, authenticated_request, get_json, normalize, supports_streaming
from ..importers.expanded import PRETRAINING_IDS
from ..importers.ranges import remote_file
from .corpus import record_text
from .dataset import WeightedMixer

PREFIX = 'hf:'


def is_stream(name):
    return name.startswith(PREFIX)


def stream_definition(name):
    source = BY_ID.get(name.removeprefix(PREFIX))
    if source is None or not supports_streaming(source):
        raise ValueError(f'{name} is unavailable for streaming. Choose a supported dataset or Local cache.')
    return source


def resolve_manifest(source):
    repo = source['repo']
    config = source.get('config', 'default')
    # Read the revision on both sides of discovery to avoid pinning a moving listing.
    revision_url = f'https://huggingface.co/api/datasets/{repo}/revision/refs%2Fconvert%2Fparquet'
    revision = get_json(revision_url)['sha']
    metadata = get_json('https://datasets-server.huggingface.co/parquet?dataset=' + quote(repo, safe=''))
    if metadata.get('partial') or get_json(revision_url)['sha'] != revision:
        raise ValueError('Hugging Face Parquet listing is incomplete or changing. Retry or use Local cache.')
    files = [(item['filename'], item['size']) for item in metadata['parquet_files']
             if item['config'] == config and item['split'] == 'train']
    return {'revision': revision, 'files': sorted(files)}


def manifest_shards(source, manifest):
    # Checkpoints hold filenames, never remotely supplied URLs or credentials.
    revision = manifest['revision']
    if not re.fullmatch(r'[a-f0-9]{40}', revision) or not 0 < len(manifest['files']) <= 10000:
        raise ValueError('Invalid pinned dataset revision or no train shards.')
    for name, size in manifest['files']:
        if not re.fullmatch(r'[a-zA-Z0-9_-]+\.parquet', name) or type(size) is not int or size <= 0:
            raise ValueError('Invalid pinned Parquet shard.')
        root = f"https://huggingface.co/datasets/{source['repo']}/resolve/{revision}"
        yield {'url': f"{root}/{source.get('config', 'default')}/train/{name}", 'size': size}


class StreamingError(ValueError):
    pass


class RemoteTokens:
    def __init__(self, name, tokenizer, width, stop, state=None):
        self.source = stream_definition(name)
        self.tokenizer = tokenizer
        self.width = width
        self.dtype = np.uint16 if tokenizer.vocab_size <= 65536 else np.uint32
        self.stop = stop
        self.closed = threading.Event()
        self.prefetch = queue.Queue(maxsize=4)
        self.worker = None
        self.capacity = max(2, min(64, 131072 // width))
        self.rng = random.Random(f'nanoforge:{name}')
        self.manifest = None
        self.position = (0, 0, 0, 0, 0)
        self.buffer = []
        self.validation = []
        if state:
            self.manifest = state['manifest']
            self.position = tuple(state['position'])
            if len(self.position) != 5 or any(type(n) is not int or n < 0 for n in self.position):
                raise ValueError('Invalid stream position.')
            saved_buffer = state['buffer']
            if len(saved_buffer) > self.capacity or any(
                type(window) is not bytes or len(window) != self.width * np.dtype(self.dtype).itemsize
                for window in saved_buffer
            ):
                raise ValueError('Invalid stream shuffle buffer.')
            self.buffer = list(saved_buffer)
            self.rng.setstate(state['rng'])

    def cancelled(self):
        return self.closed.is_set() or self.stop.is_set()

    def _texts(self, manifest, cursor, validation) -> Iterator:
        first_shard, first_group, first_row, _, _ = cursor
        source_id = self.source['id']
        columns = None
        if source_id in PRETRAINING_IDS or source_id in ('tinystories', 'cosmopedia-v2'):
            columns = ['text']
        elif source_id == 'wikipedia-en':
            columns = ['title', 'text']
        for shard_index, shard in enumerate(manifest_shards(self.source, manifest)):
            if shard_index < first_shard:
                continue
            start_group = first_group if shard_index == first_shard else 0
            with remote_file(shard, authenticated_request, urlopen, self.cancelled) as file:
                reader = parquet.ParquetFile(file)
                for group in range(start_group, reader.num_row_groups):
                    start_row = first_row if (shard_index, group) == (first_shard, first_group) else 0
                    row_index = 0
                    for batch in reader.iter_batches(batch_size=32, row_groups=[group], columns=columns, use_threads=False):
                        if self.cancelled():
                            raise InterruptedError()
                        if row_index + batch.num_rows <= start_row:
                            row_index += batch.num_rows
                            continue
                        for row in batch.to_pylist():
                            row_index += 1
                            index = row_index - 1
                            if index < start_row or (index % 10 == 0) != validation:
                                continue
                            try:
                                if source_id == 'tinystories':
                                    records = {'text': row['text']}
                                else:
                                    records = normalize(source_id, row)
                                records = records if isinstance(records, list) else [records]
                                for_text = '\n'.join(record_text(record) for record in records if record)
                            except (KeyError, TypeError, ValueError):
                                continue
                            if for_text.strip():
                                yield (shard_index, group, index), for_text + '\n'

    def _windows(self, manifest, cursor, validation=False):
        carry = []
        for location, text in self._texts(manifest, cursor, validation):
            char_start, token_start = cursor[3:] if location == cursor[:3] else (0, 0)
            for offset in range(char_start, len(text), 16384):
                tokens = self.tokenizer.encode(text[offset:offset + 16384])
                start = token_start if offset == char_start else 0
                while start < len(tokens):
                    end = min(len(tokens), start + self.width - len(carry))
                    carry.extend(tokens[start:end])
                    start = end
                    if len(carry) == self.width:
                        encoded = np.asarray(carry, dtype=self.dtype).tobytes()
                        yield encoded, (*location, offset, start)
                        carry = []

    def _put(self, item):
        while not self.cancelled():
            try:
                self.prefetch.put(item, timeout=.1)
                return True
            except queue.Full:
                pass
        return False

    def _produce(self):
        from itertools import islice

        try:
            manifest = self.manifest or resolve_manifest(self.source)
            origin = (0, 0, 0, 0, 0)
            with_validation = self._windows(manifest, origin, validation=True)
            try:
                validation = [window for window, _ in islice(with_validation, 16)]
            finally:
                with_validation.close()
            if not validation:
                raise StreamingError('Not enough held-out tokens for this context size.')
            if not self._put((manifest, validation)):
                return
            cursor = self.position
            while not self.cancelled():
                produced = False
                for encoded, position in self._windows(manifest, cursor):
                    produced = True
                    if not self._put((encoded, position)):
                        return
                if not produced and cursor == origin:
                    raise StreamingError('Not enough training tokens for this context size.')
                cursor = origin
        except InterruptedError:
            return
        except Exception as exc:
            self._put(StreamingError(f"{self.source['label']}: {exc}"))

    def _get(self):
        while not self.cancelled():
            try:
                item = self.prefetch.get(timeout=.1)
            except queue.Empty:
                continue
            if isinstance(item, Exception):
                raise item
            return item
        raise InterruptedError()

    def _start(self):
        if self.worker is None:
            self.worker = threading.Thread(target=self._produce, daemon=True, name='dataset-prefetch')
            self.worker.start()
            self.manifest, self.validation = self._get()

    def window(self, validation_rng=None):
        self._start()
        if validation_rng is not None:
            return np.frombuffer(validation_rng.choice(self.validation), dtype=self.dtype)
        while len(self.buffer) < self.capacity:
            encoded, self.position = self._get()
            self.buffer.append(encoded)
        return np.frombuffer(self.buffer.pop(self.rng.randrange(len(self.buffer))), dtype=self.dtype)

    def state(self):
        if self.manifest is None:
            return None
        return {'manifest': self.manifest, 'position': self.position,
                'buffer': tuple(self.buffer), 'rng': self.rng.getstate()}


class StreamingMixer:
    """Combine streamed windows and existing cached datasets in one weighted batch."""
    def __init__(self, sources, texts, tokenizer, block_size, device, stop, state=None):
        self.local = WeightedMixer(texts, tokenizer, block_size, device).sources if texts else []
        self.remote_specs = tuple((source['name'], source['weight']) for source in sources)
        self.tokenizer, self.device, self.stop = tokenizer, device, stop
        self.block_size = block_size
        self.rng = random.Random(42)
        self.reset_validation()
        self.remote = {}
        self.restore(state)

    def reset_validation(self):
        self.validation_rng = random.Random(77)

    def restore(self, state):
        state = state or {}
        rng = random.Random(42)
        if state.get('weights') == self.remote_specs:
            rng.setstate(state['rng'])
        remote = {name: RemoteTokens(name, self.tokenizer, self.block_size + 1, self.stop,
                                     state.get('sources', {}).get(name))
                  for name, _ in self.remote_specs}
        self.close()
        self.remote, self.rng = remote, rng
        self.names = [*[self.remote[name] for name, _ in self.remote_specs], *self.local]
        self.weights = [*[weight for _, weight in self.remote_specs], *[source[2] for source in self.local]]

    def batch(self, batch_size, validation=False):
        rng = self.validation_rng if validation else self.rng
        sources = rng.choices(self.names, weights=self.weights, k=batch_size)
        length = min([self.block_size, *(len(source[int(validation)]) - 1 for source in sources
                                        if not isinstance(source, RemoteTokens))])
        windows = []
        for source in sources:
            if isinstance(source, RemoteTokens):
                window = source.window(rng if validation else None)
            else:
                ids = source[int(validation)]
                start = rng.randrange(len(ids) - length)
                window = ids[start:start + length + 1]
            windows.append(np.asarray(window[:length + 1], dtype=np.int64))
        tensor = torch.from_numpy(np.stack(windows))
        if str(self.device).startswith('cuda'):
            tensor = tensor.pin_memory()
        tensor = tensor.to(self.device, non_blocking=True)
        return tensor[:, :-1].contiguous(), tensor[:, 1:].contiguous()

    def state(self):
        return {'sources': {name: source.state() for name, source in self.remote.items()},
                'weights': self.remote_specs, 'rng': self.rng.getstate()}

    def close(self):
        for source in self.remote.values():
            source.closed.set()
