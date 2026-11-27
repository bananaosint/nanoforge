from contextlib import contextmanager
from io import BytesIO
import json
from pathlib import Path

import chess
import chess.engine
import pytest
import pyarrow as pa
import pyarrow.parquet as parquet
import torch
from fastapi.testclient import TestClient

from backend import chess_pipeline
from backend.chess_eval import evaluate_checkpoint, theme_scores, verify_explanation
from backend.chess_pipeline import (
    BENCHMARK_MAX, STRUCTURAL_TOKENS, adapt_hub_row, benchmark_examples, extract_board_context,
    format_example, open_stockfish, parse_local, position_key, score_data, stockfish_path,
)
from backend.main import create_app
from backend.ml.model import GPT, ModelConfig, PRESETS
from backend.ml.trainer import Trainer
from backend.ml.tokenizer import Tokenizer


@pytest.mark.parametrize(('fen', 'move', 'san', 'event', 'value'), [
    (chess.STARTING_FEN, 'e2e4', 'e4', 'piece', 'pawn'),
    ('4k3/8/8/3q4/2B5/8/8/4K3 w - - 0 1', 'c4d5', 'Bxd5', 'capture', True),
    ('4k3/8/8/8/8/8/R7/6K1 w - - 0 1', 'a2e2', 'Re2+', 'check', True),
    ('6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1', 'a1a8', 'Ra8#', 'checkmate', True),
    ('4k3/8/8/8/8/8/8/4K2R w K - 0 1', 'e1g1', 'O-O', 'castling', True),
    ('4k3/P7/8/8/8/8/8/4K3 w - - 0 1', 'a7a8q', 'a8=Q+', 'promotion', 'queen'),
    ('4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1', 'e5d6', 'exd6', 'en_passant', True),
])
def test_extract_board_context(fen, move, san, event, value):
    context = extract_board_context(fen, move)
    assert context['side_to_move'] == 'white'
    assert context['move']['san'] == san
    assert context['move']['origin'] == move[:2]
    assert context['move']['destination'] == move[2:4]
    assert context['move'][event] == value


def test_board_relationships_and_material():
    fen = '4k3/4n3/8/8/8/8/R7/6K1 w - - 0 1'
    context = extract_board_context(fen, 'a2e2')
    knight = {'piece': 'knight', 'color': 'black', 'square': 'e7'}
    assert knight in context['pinned_pieces']
    assert knight in context['attacked_high_value_targets']
    assert not context['move']['check']
    hanging = extract_board_context('4k3/8/8/3q4/8/8/2B5/4K3 w - - 0 1', 'c2e4')
    queen = {'piece': 'queen', 'color': 'black', 'square': 'd5'}
    assert queen in hanging['hanging_pieces']
    assert queen in hanging['undefended_pieces']
    context = extract_board_context('4k3/8/8/3q4/2B5/8/8/4K3 w - - 0 1', 'c4d5')
    assert context['move']['captured_piece'] == 'queen'
    assert context['material_delta'] == 9
    with pytest.raises(ValueError, match='legal move'):
        extract_board_context(chess.STARTING_FEN, 'a1a8')


def test_adapt_lichess_row():
    example = adapt_hub_row('lichess', {
        'FEN': chess.STARTING_FEN, 'Moves': 'f2f3 e7e5 g2g4 d8h4', 'Rating': 1300,
        'Themes': ['mateIn2'], 'OpeningTags': ['Kings_Pawn_Game'],
    })
    assert example.fen.split()[1] == 'b'
    assert example.move == 'e7e5'
    assert example.pv == ('e7e5', 'g2g4', 'd8h4')
    assert example.themes == ('mateIn2',)
    assert example.rating == 1300
    assert example.opening_tags == ('Kings_Pawn_Game',)
    assert verify_explanation(example.context(), example.explanation)['passed']


def test_adapt_chess_explained_and_tokenizer():
    example = adapt_hub_row('chess-explained', {
        'fen': chess.STARTING_FEN, 'move': 'Nf3', 'explanation': 'The knight moves from g1 to f3.',
    })
    assert example.move == 'g1f3'
    prompt, target = format_example(example.context(), example.explanation)
    assert all(marker in prompt for marker in ('<FEN>', '<MOVE>', '<FACTS>', '<EXPLAIN>'))
    assert target == example.explanation and target not in prompt
    tokenizer = Tokenizer('magic8k', structural_tokens=STRUCTURAL_TOKENS)
    restored = Tokenizer(state=tokenizer.state())
    for token in STRUCTURAL_TOKENS:
        assert len(tokenizer.encode(token)) == 1
        assert restored.encode(token) == tokenizer.encode(token)
    assert Tokenizer('magic8k').vocab_size < tokenizer.vocab_size


@pytest.mark.parametrize(('preset', 'record'), [
    ('lichess', {'FEN': chess.STARTING_FEN, 'Moves': 'e2e4 e7e5', 'Rating': 1200,
                 'Themes': ['opening'], 'OpeningTags': ['Kings_Pawn_Game']}),
    ('chess-explained', {'fen': chess.STARTING_FEN, 'move': 'e2e4',
                         'explanation': 'White plays e4.', 'messages': [{'content': 'unused'}]}),
])
def test_hub_import_loads_pinned_train_shards(tmp_path, monkeypatch, preset, record):
    revision = 'b' * 40
    buf = BytesIO()
    parquet.write_table(pa.Table.from_pylist([record]), buf)
    data = buf.getvalue()
    shard = {'config': 'default', 'split': 'train', 'filename': '0000.parquet', 'size': len(data)}
    monkeypatch.setattr(chess_pipeline, 'get_json', lambda url:
                        {'sha': revision} if '/revision/' in url else {'parquet_files': [shard]})

    @contextmanager
    def parquet_file(source, *_):
        assert source['url'].endswith(f'/resolve/{revision}/default/train/0000.parquet')
        yield BytesIO(data)

    monkeypatch.setattr(chess_pipeline, 'remote_file', parquet_file)
    app = create_app(tmp_path)
    with TestClient(app) as client:
        result = client.post('/api/chess/datasets/import', json={'preset': preset, 'limit': 1})
        assert result.status_code == 200, result.text
        assert result.json() == {'id': preset, 'examples': 1}
        examples = app.state.trainer.chess_datasets.examples(preset)
        assert len(examples) == 1
        assert examples[0].move == ('e7e5' if preset == 'lichess' else 'e2e4')


def test_local_adapters_and_pv_validation():
    record = {'fen': chess.STARTING_FEN, 'pv': ['e2e4', 'e7e5', 'g1f3'], 'explanation': 'The pawn moves to e4.'}
    example, = parse_local(json.dumps(record), '.jsonl')
    assert example.move == 'e2e4'
    assert example.pv == tuple(record['pv'])
    assert extract_board_context(chess.STARTING_FEN, 'e2e4', ['e7e5', 'g1f3'])['pv_start'] == 'after'
    row, = parse_local('1. e4 {The white pawn moves to e4.} e5 2. Nf3 *', '.pgn')
    assert row.fen == chess.STARTING_FEN and row.move == 'e2e4'
    with pytest.raises(ValueError, match='JSONL line 2'):
        list(parse_local(json.dumps(record) + '\n{"fen": "bad"}', '.jsonl'))


@pytest.mark.parametrize(('text', 'category'), [
    ('A knight on f4 controls the center.', 'grounding'),
    ('The white queen is on h4.', 'grounding'),
    ('The pawns are on a2, b2 and c5.', 'grounding'),
    ('A knight on f9 controls the center.', 'grounding'),
    ('Bg5 is the best move.', 'legality'),
    ('White should play e2e5.', 'legality'),
    ('White’s pawn moves from e2 to d2.', 'legality'),
    ('e4 gives checkmate.', 'tactical'),
    ('e4# is decisive.', 'tactical'),
    ('White castles.', 'tactical'),
    ('The move captures a pawn.', 'tactical'),
    ('It pins the knight on g1.', 'tactical'),
    ('A lovely idea.', 'coverage'),
])
def test_verifier_reports_errors_with_offsets(text, category):
    context = extract_board_context(chess.STARTING_FEN, 'e2e4')
    result = verify_explanation(context, text)
    assert not result['passed']
    matching = [error for error in result['errors'] if error['category'] == category]
    assert matching
    assert all(error['reason'] and text[error['start']:error['end']] == error['claim'] for error in matching)


@pytest.mark.parametrize('text', [
    "White's pawn moves from e2 to e4, without giving check.",
    'e4 develops a pawn. Black can respond with e5, followed by Nf3.',
    "Black's pawns on a7, b7 and c7 remain on the seventh rank.",
    'The pawn on e4 is White’s. White does not castle.',
])
def test_verifier_accepts_supported_explanations(text):
    context = extract_board_context(chess.STARTING_FEN, 'e2e4', ['e2e4', 'e7e5', 'g1f3'])
    assert verify_explanation(context, text)['passed']


def test_verifier_ground_truth_tactical_positions_and_themes():
    for example in benchmark_examples(50):
        result = verify_explanation(example.context(), example.explanation, example.themes)
        assert result['passed'], result['errors']
        if result['themes']:
            assert result['themes']['fn'] == 0
    scores = theme_scores('It forks the king and queen and skewers the rook.', ('fork', 'pin'))
    assert scores == {'expected': ['fork', 'pin'], 'predicted': ['fork', 'skewer'],
                      'tp': 1, 'fp': 1, 'fn': 1}


def test_stockfish_polarity_and_recommendations():
    class Engine:
        def centipawn_loss(self, board, move):
            assert board.fen() == chess.STARTING_FEN and move.uci() == 'g1f3'
            return 23

    before = score_data(chess.engine.PovScore(chess.engine.Cp(10), chess.WHITE))
    after = score_data(chess.engine.PovScore(chess.engine.Cp(-250), chess.WHITE))
    context = extract_board_context(chess.STARTING_FEN, 'e2e4', stockfish_info={
        'eval_before': before, 'eval_after': after, 'eval_swing_cp': -260, 'multipv': [],
    })
    result = verify_explanation(context, 'e4 is a blunder. White is winning. The best move is Nf3.', engine=Engine())
    verdicts = [item for item in result['checks'] if item['category'] == 'stockfish']
    assert [(item['claim'], item['passed']) for item in verdicts] == [('blunder', True), ('White is winning', False)]
    assert result['recommendations'][0]['centipawn_loss'] == 23
    assert before['mate_color'] is None
    assert score_data(chess.engine.PovScore(chess.engine.Mate(0), chess.BLACK))['mate_color'] == 'white'


def test_real_engine_when_installed():
    path = stockfish_path()
    if not path:
        pytest.skip('Stockfish is optional.')
    with open_stockfish(path) as engine:
        assert engine
        result = engine.analyse(chess.STARTING_FEN, 'e2e4')
        assert result == engine.analyse(chess.STARTING_FEN, 'e2e4')
        assert len(result['multipv']) == 3
        assert isinstance(result['eval_after']['cp'], int)
        assert engine.centipawn_loss(chess.Board(), chess.Move.from_uci('g1f3')) >= 0
    report = evaluate_checkpoint(lambda _: 'Ra8# is a back-rank mate in one. White wins.', 'native.pt', None, 1, 1, path)
    assert report['stockfish_available']
    assert report['metrics']['stockfish_polarity_agreement'] == 100
    assert report['samples'][0]['verification']['passed']


def test_batched_chess_decoding_matches_single_position():
    tokenizer = Tokenizer('magic8k', structural_tokens=(*STRUCTURAL_TOKENS, '<|end|>'))
    torch.manual_seed(11)
    model = GPT(ModelConfig(vocab_size=tokenizer.vocab_size, n_layer=1, n_head=2,
                            n_embd=24, block_size=512, dropout=0.0)).eval()
    prompts = [format_example(example.context(), themes=example.themes)[0] for example in benchmark_examples(3)]
    expected = [Trainer._generate_chess(model, tokenizer, prompt) for prompt in prompts]
    assert Trainer._generate_chess_batch(model, tokenizer, prompts) == expected


def test_fixed_benchmark_is_held_out_and_deterministic(tmp_path):
    app = create_app(tmp_path)
    with TestClient(app):
        datasets = app.state.trainer.chess_datasets
        reserved = benchmark_examples(BENCHMARK_MAX)
        assert reserved[:50] == benchmark_examples(50)
        assert len({position_key(example.fen) for example in reserved}) == BENCHMARK_MAX
        assert not {position_key(example.fen) for example in reserved} & {
            position_key(example.fen) for example in datasets.examples('starter')
        }
    generate = lambda prompt: 'White plays a knight on z9.'
    first = evaluate_checkpoint(generate, 'test.pt', 'run', 10, 2, None)
    second = evaluate_checkpoint(generate, 'test.pt', 'run', 10, 2, None)
    assert first['benchmark_id'] == second['benchmark_id']
    assert first['metrics'] == second['metrics']
    assert first['metrics']['piece_hallucination_rate'] == 100
    assert first['metrics']['stockfish_polarity_agreement'] is None
    assert len(first['samples']) == 2


def test_chess_run_checkpoint_benchmark_and_sandbox_without_stockfish(tmp_path, monkeypatch):
    monkeypatch.setenv('NANOFORGE_DEVICE', 'cpu')
    monkeypatch.setenv('STOCKFISH_PATH', str(tmp_path/'missing-stockfish'))
    monkeypatch.setitem(PRESETS, 'Nano', {'n_layer': 1, 'n_head': 2, 'n_embd': 24, 'block_size': 128, 'dropout': 0.0})
    app = create_app(tmp_path)
    with TestClient(app) as client:
        assert not client.get('/api/chess/settings').json()['stockfish_available']
        assert client.get('/api/checkpoints').json() == []
        assert client.get('/api/chess/prompt-preview').json()['prompt'].endswith('<EXPLAIN>')
        assert client.post('/api/chess/runs', json={'dataset': 'starter-sample.txt'}).status_code == 400
        run = client.post('/api/chess/runs', json={
            'max_steps': 3, 'batch_size': 1, 'eval_interval': 2, 'benchmark_size': 2, 'use_stockfish': True,
        })
        assert run.status_code == 200, run.text
        run_id = run.json()['run_id']
        trainer = app.state.trainer
        trainer.thread.join(30)
        assert not trainer.thread.is_alive()
        assert trainer.state == 'completed', trainer.error
        assert client.get('/api/runs').json() == []
        assert client.get('/api/runs/' + run_id + '/metrics').json() == []
        assert client.get('/api/chess/runs').json()[0]['mode'] == 'chess'
        assert client.get('/api/checkpoints').json() == []
        checkpoints = client.get('/api/chess/checkpoints').json()
        assert {c['step'] for c in checkpoints} == {2, 3}
        checkpoint = checkpoints[0]['name']
        assert client.post('/api/checkpoints/select', json={'name': checkpoint}).status_code == 400
        history = client.get('/api/chess/accuracy', params={'run_id': run_id}).json()
        assert [(row['step'], row['source']) for row in history] == [(2, 'automatic'), (3, 'automatic')]
        url = f'/api/chess/checkpoints/{checkpoint}/accuracy'
        automatic = client.get(url).json()
        assert automatic['benchmark_id'] == history[-1]['benchmark_id']
        assert automatic['protocol']['size'] == 2
        assert automatic['metrics']['stockfish_polarity_agreement'] is None
        assert 0 <= automatic['metrics']['composite_chess_accuracy'] <= 100
        assert len(automatic['samples']) == 2
        assert list((tmp_path/'checkpoints'/'.chess-eval'/checkpoint).glob('*.json'))
        manual = client.post(url, json={'benchmark_size': 2}).json()
        assert manual['source'] == 'manual'
        assert manual['benchmark_id'] == automatic['benchmark_id']
        assert manual['metrics'] == automatic['metrics']
        assert [row['explanation'] for row in manual['samples']] == [row['explanation'] for row in automatic['samples']]
        variant = client.post(url, json={'benchmark_size': 1}).json()
        assert variant['benchmark_id'] != manual['benchmark_id']
        retained = client.get(url, params={'source': 'manual', 'benchmark_id': manual['benchmark_id']}).json()
        assert retained['metrics'] == manual['metrics']

        sandbox = client.post('/api/chess/test-position', json={
            'checkpoint': checkpoint, 'pgn': '1. e4 e5 2. Nf3 Nc6', 'ply': 3, 'use_stockfish': True,
        })
        assert sandbox.status_code == 200, sandbox.text
        body = sandbox.json()
        assert isinstance(body['explanation'], str)
        assert body['context']['move']['uci'] == 'g1f3'
        assert not body['stockfish_available']
        assert {'passed', 'checks', 'errors', 'themes', 'recommendations'} <= body['verification'].keys()
        assert client.post('/api/chess/test-position', json={'checkpoint': checkpoint, 'fen': chess.STARTING_FEN, 'move': 'e2e5'}).status_code == 400

        position = {'fen': chess.STARTING_FEN, 'move': 'e2e4', 'checkpoint': checkpoint}
        bad = client.post('/api/chess/corrections', json={**position, 'explanation': 'A knight is on f4.'}).json()
        assert not bad['saved']
        good = client.post('/api/chess/corrections', json={**position, 'explanation': "White's pawn moves from e2 to e4."}).json()
        assert good['saved']
        assert client.get('/api/feedback').json() == []
        assert trainer.chess_datasets.examples('starter')[-1].move == 'e2e4'
        weights = (tmp_path/'checkpoints'/checkpoint).read_bytes()
        assert client.post('/api/checkpoints/upload', files={'file': ('portable.pt', weights)}).status_code == 400
        imported = client.post('/api/checkpoints/upload?mode=chess', files={'file': ('portable.pt', weights)})
        assert imported.status_code == 200 and imported.json()['mode'] == 'chess'
        resumed = client.post('/api/chess/runs', json={
            'checkpoint': 'portable.pt', 'max_steps': 1, 'batch_size': 1, 'benchmark_size': 1,
        })
        assert resumed.status_code == 200
        trainer.thread.join(30)
        assert trainer.state == 'completed' and trainer.step == 4
        assert trainer.config['parent_checkpoint'] == 'portable.pt'
        assert client.post('/api/runs', json={'preset': 'Nano'}).json()['mode'] == 'pretrain'


def test_chess_upload_is_isolated(tmp_path):
    app = create_app(tmp_path)
    with TestClient(app) as client:
        route = '/api/chess/datasets/upload'
        pgn = b'1. e4 {White plays e4.} e5 2. Nf3 *'
        result = client.post(route, files={'file': ('lesson.pgn', pgn)})
        assert result.json() == {'id': 'local:lesson', 'examples': 1}
        assert client.post(route, files={'file': ('lesson.pgn', pgn)}).status_code == 400
        assert client.post(route, files={'file': ('lesson2.pgn', b'1. e4 e5 *')}).status_code == 400
        assert 'local:lesson' in {row['id'] for row in client.get('/api/chess/datasets').json()}
        assert {row['name'] for row in client.get('/api/datasets').json()} == {'starter-sample.txt'}
        assert not (Path(tmp_path)/'data'/'.chess'/'lesson2.jsonl').exists()
