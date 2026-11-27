from io import BytesIO
import json
import tarfile

import chess
import pyarrow as pa
import pyarrow.parquet as parquet
import pytest
import torch
from fastapi.testclient import TestClient

from backend import chess_pipeline
from backend.chess_pipeline import ChessExampleStream, adapt_stream_row, format_example
from backend.importers import chess_sources
from backend.importers.chess_sources import CHESS_STREAMS, ORIGIN, chess_rows, resolve_chess_manifest
from backend.main import create_app
from backend.ml.model import PRESETS
from backend.ml.streaming import RemoteTokens
from backend.ml.tokenizer import Tokenizer

REVISION = 'a' * 40
GAMES = ['e2e4', 'e7e5', 'g1f3', 'b8c6', 'f1b5', 'a7a6', 'b5a4', 'g8f6']
FIXTURES = {
    'strategic-game-chess': ('chess_game_0001.parquet', {'Moves': GAMES}),
    '400m-full-chess-match': ('data/shard_000001.parquet', {'moves': repr(GAMES)}),
    'chess-position-evaluations': ('data/data_0000.parquet',
                                  {'fen': chess.STARTING_FEN, 'line': 'e2e4 e7e5', 'cp': 19, 'mate': None}),
    'chess-explained': ('ChessExplained_2500k_qwen3.parquet',
                       {'fen': chess.STARTING_FEN, 'move': 'e2e4', 'explanation': 'White plays e4.'}),
    'chess-positions-lichess-sf': ('data/train-src00000-00000.parquet',
                                  {'fen': chess.STARTING_FEN, 'best_move': 'e2e4', 'eval_type': 'cp', 'eval_value': 19}),
    'waterhorse-chess-data': ('chessclip_data/annotated_pgn/annotated_pgn_free.tar.gz',
                             {'fen': chess.STARTING_FEN, 'move': 'e2e4', 'explanation': 'The pawn moves to e4.'}),
}


def install_fixture(monkeypatch, preset):
    path, row = FIXTURES[preset]
    data = BytesIO()
    if preset == 'waterhorse-chess-data':
        with tarfile.open(fileobj=data, mode='w:gz') as archive:
            text = b'1. e4 {The pawn moves to e4.} e5 {Black plays e5.} *\n'
            member = tarfile.TarInfo('annotated/lesson.pgn')
            member.size = len(text)
            archive.addfile(member, BytesIO(text))
    else:
        parquet.write_table(pa.Table.from_pylist([row, row]), data)
    data = data.getvalue()

    def respond(url, timeout):
        url = url.full_url
        assert url.endswith(f'/resolve/{REVISION}/{path}')
        return BytesIO(data)

    def remote(shard, *_, **__):
        assert shard['size'] == len(data)
        assert shard['url'].endswith(f'/resolve/{REVISION}/{path}')
        return BytesIO(data)

    def tree(repo, revision, request):
        assert revision == REVISION
        assert repo == CHESS_STREAMS[preset]['repo']
        yield {'type': 'file', 'path': path, 'size': len(data)}
        yield {'type': 'file', 'path': 'data/test-src00000-00000.parquet', 'size': 42}

    monkeypatch.setattr('backend.importers.hub.get_json', lambda url: {'sha': REVISION})
    monkeypatch.setattr(chess_sources, 'tree_files', tree)
    monkeypatch.setattr(chess_sources, 'remote_file', remote)
    monkeypatch.setattr(chess_sources, 'urlopen', respond)


@pytest.mark.parametrize('preset', FIXTURES)
def test_streamed_sources_read_pinned_training_files(monkeypatch, preset):
    install_fixture(monkeypatch, preset)
    manifest = resolve_chess_manifest(preset, lambda: False)
    assert [name for name, _ in manifest['files']] == [FIXTURES[preset][0]]
    rows = list(chess_rows(preset, manifest, ORIGIN, lambda: False))
    assert [position for position, _ in rows] == [(0, 0, 0), (0, 0, 1)]
    assert list(chess_rows(preset, manifest, (0, 0, 1), lambda: False)) == rows[1:]
    examples = list(adapt_stream_row(preset, rows[0][1]))
    assert examples
    for example in examples:
        assert chess.Board(example.fen).is_legal(chess.Move.from_uci(example.move))
        assert example.explanation
    if preset in ('chess-position-evaluations', 'chess-positions-lichess-sf'):
        prompt, _ = format_example(examples[0].context())
        assert '<EVAL> archive line=+19 cp' in prompt
    if preset == 'waterhorse-chess-data':
        assert examples[0].explanation == 'The pawn moves to e4.'


def test_catalog_stream_and_resume(monkeypatch, tmp_path):
    preset = 'strategic-game-chess'
    install_fixture(monkeypatch, preset)
    monkeypatch.setitem(PRESETS, 'Nano', {'n_layer': 1, 'n_head': 2, 'n_embd': 24, 'block_size': 128, 'dropout': 0.0})
    app = create_app(tmp_path)
    with TestClient(app) as client:
        expected_repos = {source['repo'] for source in CHESS_STREAMS.values()}
        text_catalog = client.get('/api/imports').json()['catalog']
        assert expected_repos <= {row['repo'] for row in text_catalog if row['streaming']}
        chess_catalog = client.get('/api/chess/datasets').json()
        assert expected_repos == {row['label'] for row in chess_catalog if row.get('streaming')}
        assert client.post('/api/imports/start', json={'dataset_id': preset}).status_code == 400

        remote = RemoteTokens(f'hf:{preset}', Tokenizer('magic8k'), 32, app.state.trainer.stop_event)
        manifest = resolve_chess_manifest(preset, lambda: False)
        rows = list(remote._texts(manifest, (0, 0, 0, 0, 0), validation=False))
        assert rows[0][0] == (0, 0, 1) and '<FEN>' in rows[0][1]

        settings = {'dataset': f'hf:{preset}', 'max_steps': 1, 'batch_size': 1, 'benchmark_size': 1}
        assert client.post('/api/chess/runs', json=settings).status_code == 200
        trainer = app.state.trainer
        trainer.thread.join(30)
        assert trainer.state == 'completed', trainer.error
        first = trainer.checkpoint
        state = torch.load(trainer.directory / first, weights_only=True)['stream_state']
        assert state['position'] == (0, 0, 0, 1)

        def unexpected_discovery(*_):
            pytest.fail('Resumption should use the checkpoint’s pinned revision.')

        monkeypatch.setattr(chess_pipeline, 'resolve_chess_manifest', unexpected_discovery)
        assert client.post('/api/chess/runs', json={**settings, 'checkpoint': first}).status_code == 200
        trainer.thread.join(30)
        assert trainer.state == 'completed' and trainer.step == 2, trainer.error
        restored = torch.load(trainer.directory / trainer.checkpoint, weights_only=True)['stream_state']
        assert restored['position'] == (0, 0, 0, 2)
        assert restored['manifest']['revision'] == REVISION


def test_stream_excludes_reserved_positions(monkeypatch):
    preset = 'chess-explained'
    install_fixture(monkeypatch, preset)
    monkeypatch.setattr(chess_pipeline, 'held_out_positions', lambda: {chess_pipeline.position_key(chess.STARTING_FEN)})
    stream = ChessExampleStream(preset, lambda: False)
    with pytest.raises(ValueError, match='no usable training examples'):
        next(stream)
    stream.close()


def test_archived_line_normalizes_castling_and_retains_signed_mate():
    row = {'fen': '4k3/8/8/8/8/8/8/4K2R w K -', 'line': 'e1h1', 'cp': None, 'mate': -3}
    example, = adapt_stream_row('chess-position-evaluations', row)
    assert example.move == 'e1g1'
    assert example.context()['move']['san'] == 'O-O'
    assert example.record()['source_eval'] == 'mate -3'


def test_native_file_discovery_follows_paginated_tree(monkeypatch):
    repo = CHESS_STREAMS['strategic-game-chess']['repo']
    base = f'https://huggingface.co/api/datasets/{repo}/tree/{REVISION}'
    first = base + '?recursive=true'
    second = first + '&cursor=page2'
    observed = []

    def open_page(url, timeout):
        observed.append(url)
        page = BytesIO(json.dumps([{'type': 'file', 'path': f'chess_game_{len(observed):04}.parquet', 'size': 8}]).encode())
        page.headers = {'Link': f'<{second}>; rel="next"'} if url == first else {}
        return page

    monkeypatch.setattr(chess_sources, 'urlopen', open_page)
    entries = list(chess_sources.tree_files(repo, REVISION, lambda url: url))
    assert observed == [first, second]
    assert len(entries) == 2
