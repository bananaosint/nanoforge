"""Exercise streaming from Parquet bytes through the training checkpoint boundary."""
from io import BytesIO
import json
from threading import Event

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from backend.ml.dataset import DatasetManager
from backend.ml.streaming import StreamingMixer
from backend.ml.tokenizer import Tokenizer
from backend.ml.trainer import Trainer
from backend.storage import Store


REVISION = 'a' * 40
STREAM = {'name': 'hf:tinystories-original', 'weight': 100}


@pytest.fixture
def remote(monkeypatch):
    # Several row groups and a following shard expose cursor errors at boundaries.
    data = BytesIO()
    pq.write_table(pa.table({'text': [
        ('Validation only. ' if i % 10 == 0 else f'Story number {i}. ') * 10
        for i in range(40)
    ]}), data, row_group_size=20)
    raw = data.getvalue()
    discovery = []
    requests = []

    def get_json(url):
        discovery.append(url)
        if '/revision/' in url:
            return {'sha': REVISION}
        return {'parquet_files': [
            {'filename': f'{i:04}.parquet', 'size': len(raw), 'config': 'default', 'split': split}
            for i, split in enumerate(('train', 'train', 'validation'))
        ]}

    class Response(BytesIO):
        status = 206

    def open_range(request, timeout):
        requests.append(request.full_url)
        assert f'/resolve/{REVISION}/default/train/' in request.full_url
        assert '0002.parquet' not in request.full_url
        start, end = map(int, request.get_header('Range')[6:].split('-'))
        response = Response(raw[start:end+1])
        response.headers = {'Content-Range': f'bytes {start}-{end}/{len(raw)}'}
        return response

    monkeypatch.setattr('backend.ml.streaming.get_json', get_json)
    monkeypatch.setattr('backend.ml.streaming.urlopen', open_range)
    return discovery, requests


def test_checkpointed_stream_replays_unconsumed_prefetch_and_fixed_validation(remote):
    tokenizer = Tokenizer('magic8k')
    local = [('Local file tokens. ' * 100, 30)]
    source = dict(STREAM, weight=70)
    mixer = StreamingMixer([source], local, tokenizer, 32, 'cpu', Event())
    try:
        first_validation = mixer.batch(6, validation=True)
        for _ in range(17):
            mixer.batch(5)
        # Use the real portable checkpoint serializer.
        file = BytesIO()
        torch.save(mixer.state(), file)
        file.seek(0)
        state = torch.load(file, weights_only=True)
        expected = [mixer.batch(8) for _ in range(20)]
    finally:
        mixer.close()
    discovery_count = len(remote[0])
    resumed = StreamingMixer([source], local, tokenizer, 32, 'cpu', Event(), state)
    try:
        assert all(torch.equal(a, b) for a, b in zip(first_validation, resumed.batch(6, validation=True)))
        for batch in expected:
            assert all(torch.equal(a, b) for a, b in zip(batch, resumed.batch(8)))
        assert len(remote[0]) == discovery_count  # Resume uses the saved immutable manifest.
        assert remote[1]
    finally:
        resumed.close()


def test_trainer_streams_without_corpus_files_and_restores_cursor(tmp_path, remote):
    manager = DatasetManager(tmp_path/'data')
    store = Store(tmp_path/'runs.sqlite3')
    trainer = Trainer(manager, store, tmp_path/'checkpoints', 'cpu')
    options = {'tokenizer': 'magic8k', 'datasets': [STREAM],
               'model': {'n_layer': 1, 'n_head': 1, 'n_embd': 16, 'block_size': 16},
               'batch_size': 1, 'max_steps': 2}
    try:
        with pytest.raises(ValueError, match='Character tokenization'):
            trainer.preview_model(dict(options, tokenizer='char'))
        trainer.create(options)
        assert not remote[0]  # Model setup and sizing don't contact HF.
        trainer.start(options, background=False, evaluate=False)
        assert trainer.state == 'completed', trainer.error
        name = trainer.checkpoint
        saved = torch.load(trainer.directory/name, weights_only=True)
        assert saved['stream_state']['sources'][STREAM['name']]['manifest']['revision'] == REVISION
        assert not list(manager.directory.glob('.tokens/*'))
        assert {path.name for path in manager.directory.iterdir()} == {'starter-sample.txt'}
        trainer.load(name)
        trainer.start(options, background=False, evaluate=False)
        assert trainer.step == 4 and trainer.state == 'completed'
        assert len(remote[0]) == 3  # Initial discovery only.
    finally:
        trainer.shutdown()
        store.close()


def test_stream_error_leaves_a_resumable_checkpoint(tmp_path, remote, monkeypatch):
    store = Store(tmp_path/'runs.sqlite3')
    trainer = Trainer(DatasetManager(tmp_path/'data'), store, tmp_path/'checkpoints', 'cpu')
    options = {'tokenizer': 'magic8k', 'datasets': [STREAM],
               'model': {'n_layer': 1, 'n_head': 1, 'n_embd': 16, 'block_size': 16},
               'batch_size': 1, 'max_steps': 2}

    def offline(request, timeout):
        raise OSError('test network error')

    monkeypatch.setattr('backend.ml.streaming.urlopen', offline)
    try:
        trainer.create(options)
        trainer.start(options, background=False, evaluate=False)
        assert trainer.state == 'error' and 'test network error' in trainer.error
        assert trainer.step == 0 and trainer.checkpoint_path(trainer.checkpoint).is_file()
        assert not json.loads((trainer.directory/trainer.checkpoint).with_suffix('.json').read_text())['step']
    finally:
        trainer.shutdown()
        store.close()
