"""Chess-only studio routes and request validation."""

from collections.abc import Callable
from io import StringIO
import os
from pathlib import Path
import re
from typing import Annotated, Literal
from urllib.error import URLError

import chess
import chess.engine
import chess.pgn
from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from pydantic import Field, model_validator
from starlette.concurrency import run_in_threadpool

from .schemas import StrictModel
from ..ml.trainer import Trainer
from ..storage import Store
from ..chess_eval import evaluate_checkpoint, verify_explanation
from ..chess_pipeline import (
    BENCHMARK_MAX, ChessExample, benchmark_examples, extract_board_context, format_example, hub_examples,
    legal_move, open_stockfish, parse_local, position_key, read_board, stockfish_path,
)


class ChessRun(StrictModel):
    dataset: str = Field(default='starter', max_length=90)
    preset: Literal['Nano', 'Micro', 'Mini'] = 'Nano'
    checkpoint: str | None = Field(default=None, max_length=200)
    precision: Literal['auto', 'float32', 'float16', 'bfloat16'] = 'auto'
    max_steps: int = Field(default=250, ge=1, le=1000000)
    batch_size: int = Field(default=4, ge=1, le=64)
    learning_rate: float = Field(default=0.0003, gt=0, le=0.1)
    eval_interval: int = Field(default=50, ge=1, le=10000)
    benchmark_size: int = Field(default=50, ge=1, le=BENCHMARK_MAX)
    use_stockfish: bool = False


class BenchmarkRequest(StrictModel):
    benchmark_size: int | None = Field(default=None, ge=1, le=BENCHMARK_MAX)
    use_stockfish: bool | None = None


class HubRequest(StrictModel):
    preset: Literal['lichess', 'chess-explained']
    limit: int = Field(default=2000, ge=1, le=50000)


class StockfishSettings(StrictModel):
    stockfish_path: str = Field(default='', max_length=1024)


class Position(StrictModel):
    fen: str | None = Field(default=None, min_length=1, max_length=200)
    move: str | None = Field(default=None, min_length=1, max_length=16)
    pgn: str | None = Field(default=None, min_length=1, max_length=200000)
    ply: int | None = Field(default=None, ge=1, le=2000)
    pv: list[str] = Field(default_factory=list, max_length=80)

    @model_validator(mode='after')
    def validate_position(self):
        if bool(self.fen) == bool(self.pgn):
            raise ValueError('Supply a FEN or a PGN.')
        if self.fen and self.ply is not None:
            raise ValueError('Ply applies to PGN input.')
        if self.pgn and self.move:
            raise ValueError('Use the PGN ply to choose the move.')
        return self

    def resolve(self):
        if self.fen:
            board = read_board(self.fen)
            return board, legal_move(board, self.move) if self.move else None, []
        stream = StringIO(self.pgn)
        game = chess.pgn.read_game(stream)
        board = game.board() if game else None
        if type(board) is not chess.Board or board.chess960 or game.errors or not board.is_valid() or chess.pgn.read_game(stream):
            raise ValueError('Enter one valid standard-chess PGN game.')
        moves = []
        selected = None
        for index, move in enumerate(game.mainline_moves(), 1):
            if index == (self.ply or 1):
                selected = board.copy(stack=False), move
            moves.append({'ply': index, 'san': board.san(move), 'uci': move.uci()})
            board.push(move)
        if selected is None:
            raise ValueError(f'The PGN contains {len(moves)} plies. Select an existing ply.')
        return *selected, moves

    def example(self) -> ChessExample:
        board, move, _ = self.resolve()
        if move is None:
            raise ValueError('Select a legal move.')
        return ChessExample(board.fen(), move.uci(), '', tuple(self.pv))


class TestPosition(Position):
    checkpoint: str = Field(min_length=1, max_length=200)
    use_stockfish: bool = False


class Correction(Position):
    explanation: str = Field(min_length=1, max_length=50000)
    original_output: str = Field(default='', max_length=50000)
    checkpoint: str | None = Field(default=None, max_length=200)
    use_stockfish: bool = False


def chess_router(trainer: Trainer, store: Store) -> APIRouter:
    router = APIRouter(prefix='/api/chess', tags=['chess'])
    datasets = trainer.chess_datasets
    reports = trainer.chess_reports

    def current_engine() -> str | None:
        return stockfish_path(store.setting('stockfish_path', ''))

    def settings():
        override = store.setting('stockfish_path', '')
        path = stockfish_path(override)
        return {'stockfish_path': override, 'resolved_path': path, 'stockfish_available': path is not None,
                'source': 'setting' if override else 'environment' if os.environ.get('STOCKFISH_PATH') else 'path'}

    def position_verification(example: ChessExample, use_stockfish: bool, explain: Callable[[str], str]):
        def run(engine):
            context = example.context(engine)
            prompt, _ = format_example(context)
            text = explain(prompt)
            return {'context': context, 'prompt': prompt, 'explanation': text, 'stockfish_available': engine is not None,
                    'verification': verify_explanation(context, text, engine=engine)}
        with open_stockfish(current_engine() if use_stockfish else None) as engine:
            try:
                return run(engine)
            except (OSError, chess.engine.EngineError, TimeoutError):
                if not engine:
                    raise
                return run(None)

    def checkpoint_exists(name: str):
        trainer.checkpoint_path(name)
        if not any(entry['name'] == name for entry in trainer.checkpoints('chess')):
            raise ValueError('Choose a chess checkpoint.')

    @router.get('/settings')
    def get_settings():
        return settings()

    @router.put('/settings')
    def put_settings(body: StockfishSettings):
        value = body.stockfish_path.strip()
        if value:
            path = stockfish_path(value)
            if not path or not re.fullmatch(r'stockfish(?:[-_.][\w.-]+)?(?:\.exe)?', Path(path).resolve().name, re.I):
                raise ValueError('Enter the path to an installed Stockfish executable.')
        store.set_setting('stockfish_path', value)
        return settings()

    @router.get('/datasets')
    def catalog():
        return datasets.catalog()

    @router.post('/datasets/import')
    def import_dataset(body: HubRequest):
        try:
            return datasets.save(body.preset, hub_examples(body.preset, body.limit), replace=True)
        except (URLError, OSError, KeyError) as exc:
            raise HTTPException(502, f'Hugging Face dataset is unavailable: {exc}') from exc

    @router.post('/datasets/upload')
    async def upload(file: UploadFile = File(...)):
        filename = Path(file.filename or '')
        suffix = filename.suffix.lower()
        if filename.name != file.filename or suffix not in ('.jsonl', '.pgn'):
            raise ValueError('Upload a UTF-8 .jsonl or annotated .pgn file.')
        if filename.stem in ('starter', 'lichess', 'chess-explained'):
            raise ValueError('Use a name distinct from the built-in chess presets.')
        raw = await file.read(20 * 1024 * 1024 + 1)
        if len(raw) > 20 * 1024 * 1024:
            raise ValueError('Files must be smaller than 20 MB.')
        try:
            text = raw.decode('utf-8-sig')
        except UnicodeError as exc:
            raise ValueError('Use a UTF-8 encoded file.') from exc
        return await run_in_threadpool(datasets.save, filename.stem, parse_local(text, suffix))

    @router.get('/prompt-preview')
    def prompt_preview():
        context = extract_board_context(chess.STARTING_FEN, 'e2e4', ['e2e4', 'e7e5', 'g1f3'])
        prompt, target = format_example(context, 'White’s pawn moves from e2 to e4, taking space in the center.')
        return {'prompt': prompt, 'target': target}

    @router.post('/runs')
    def launch(body: ChessRun):
        options = body.model_dump(exclude={'checkpoint'})
        return trainer.start_chess(options, body.checkpoint, current_engine() if body.use_stockfish else None)

    @router.post('/training/{action}')
    def control(action: Literal['pause', 'resume', 'stop']):
        if trainer.mode != 'chess':
            raise ValueError('There is no active chess run.')
        return {'pause': trainer.pause, 'resume': trainer.resume, 'stop': trainer.stop}[action]()

    @router.get('/runs')
    def runs():
        return trainer.run_history('chess')

    @router.get('/checkpoints')
    def checkpoints():
        return trainer.checkpoints('chess')

    @router.get('/accuracy')
    def accuracy(run_id: str | None = None):
        return reports.history(trainer.checkpoints('chess'), run_id)

    @router.get('/checkpoints/{name}/accuracy')
    def report(name: str, source: Literal['automatic', 'manual'] | None = None,
               benchmark_id: Annotated[str | None, Query(pattern=r'^[a-f0-9]{16}$')] = None):
        with trainer.lock:
            checkpoint_exists(name)
            result = reports.get(name, source, benchmark_id)
        if result is None:
            raise HTTPException(404, 'This checkpoint has not been evaluated.')
        return result

    @router.post('/checkpoints/{name}/accuracy')
    def benchmark(name: str, body: BenchmarkRequest):
        generate, generate_batch, meta = trainer.chess_generator(name)
        config = meta['config']
        use_stockfish = body.use_stockfish if body.use_stockfish is not None else config.get('use_stockfish', False)
        engine_path = current_engine() if use_stockfish else None
        size = body.benchmark_size or config.get('benchmark_size', 50)
        result = evaluate_checkpoint(generate, name, meta['run_id'], meta['step'], size, engine_path,
                                     generate_batch=generate_batch)
        with trainer.lock:
            checkpoint_exists(name)
            return reports.save(result, 'manual')

    @router.post('/position')
    def position(body: Position):
        board, move, pgn_moves = body.resolve()
        moves = [{'uci': candidate.uci(), 'san': board.san(candidate)}
                 for candidate in sorted(board.legal_moves, key=chess.Move.uci)]
        context = extract_board_context(board.fen(), move.uci(), body.pv) if move else None
        return {'fen': board.fen(), 'legal_moves': moves, 'pgn_moves': pgn_moves, 'context': context}

    @router.post('/test-position')
    def test_position(body: TestPosition):
        example = body.example()
        generate, _, _ = trainer.chess_generator(body.checkpoint)
        return {'checkpoint': body.checkpoint, **position_verification(example, body.use_stockfish, generate)}

    @router.post('/corrections')
    def save_correction(body: Correction):
        example = body.example()
        if body.checkpoint:
            checkpoint_exists(body.checkpoint)
        if any(position_key(example.fen) == position_key(item.fen) for item in benchmark_examples(BENCHMARK_MAX)):
            raise ValueError('This position is held out for the accuracy benchmark.')
        text = body.explanation.strip()
        result = position_verification(example, body.use_stockfish, lambda _: text)
        verification = result['verification']
        if not verification['passed']:
            return {'saved': False, 'verification': verification}
        identifier = store.save_feedback('chess-sft', {
            'fen': example.fen, 'move': example.move, 'pv': example.pv, 'explanation': text,
            'original_output': body.original_output, 'checkpoint': body.checkpoint, 'mode': 'chess',
        })
        return {'saved': True, 'id': identifier, 'verification': verification}

    return router
