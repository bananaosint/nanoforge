"""Normalize chess examples, derive position facts, and format completion prompts."""

from __future__ import annotations

import ast
from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from functools import lru_cache
from io import StringIO
from itertools import islice
import json
import os
from pathlib import Path
import random
import re
import shutil
import tempfile
from typing import Any, TypedDict
from urllib.parse import quote
from urllib.request import urlopen

import chess
import chess.engine
import chess.pgn

from .importers.chess_sources import CHESS_STREAMS, ORIGIN, chess_rows, resolve_chess_manifest
from .importers.hub import authenticated_request, get_json
from .importers.ranges import remote_file

STRUCTURAL_TOKENS = ('<FEN>', '<MOVE>', '<EVAL>', '<PV>', '<FACTS>', '<THEMES>', '<EXPLAIN>')
HUB_PRESETS = {
    'lichess': 'Lichess/chess-puzzles',
    'chess-explained': 'aicrowd/ChessExplained',
}
VALUES = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}
PV_LIMIT = 12
BENCHMARK_VERSION = 'nanoforge-chess-v1'
BENCHMARK_MAX = 200


class MoveFacts(TypedDict):
    color: str
    piece: str
    origin: str
    destination: str
    uci: str
    san: str
    capture: bool
    captured_piece: str | None
    en_passant: bool
    castling: bool
    promotion: str | None
    check: bool
    checkmate: bool


class BoardPiece(TypedDict):
    color: str
    piece: str
    square: str


class EngineScore(TypedDict):
    cp: int | None
    mate: int | None
    mate_color: str | None


class EngineLine(TypedDict):
    rank: int
    score: EngineScore
    pv: list[str]


class EngineAnalysis(TypedDict):
    eval_before: EngineScore
    eval_after: EngineScore
    eval_swing_cp: int | None
    multipv: list[EngineLine]


class _OptionalEngineAnalysis(TypedDict, total=False):
    source_eval: str
    eval_before: EngineScore
    eval_after: EngineScore
    eval_swing_cp: int | None
    multipv: list[EngineLine]


class BoardContext(_OptionalEngineAnalysis):
    fen: str
    after_fen: str
    side_to_move: str
    move: MoveFacts
    pv: list[str]
    pv_start: str
    undefended_pieces: list[BoardPiece]
    hanging_pieces: list[BoardPiece]
    attacked_high_value_targets: list[BoardPiece]
    pinned_pieces: list[BoardPiece]
    material_before: int
    material_after: int
    material_delta: int


def read_board(fen: str) -> chess.Board:
    board = chess.Board(fen.strip())
    if not board.is_valid():
        raise ValueError('The FEN does not describe a legal standard-chess position.')
    return board


def legal_move(board: chess.Board, notation: str) -> chess.Move:
    notation = notation.strip()
    try:
        move = board.parse_uci(notation.lower()) if re.fullmatch(r'[a-h][1-8][a-h][1-8][qrbn]?', notation, re.I) else board.parse_san(notation)
        if move and board.is_legal(move):
            return move
    except ValueError:
        pass
    raise ValueError(f'{notation!r} is not a legal move from this position.')


def move_line(board: chess.Board, line: Iterable[str]) -> tuple[list[str], list[chess.Board]]:
    board = board.copy(stack=False)
    moves, positions = [], [board.copy(stack=False)]
    for token in line:
        move = legal_move(board, token)
        moves.append(move.uci())
        board.push(move)
        positions.append(board.copy(stack=False))
    return moves, positions


def variation(board: chess.Board, after: chess.Board, line: Iterable[str]) -> tuple[str, list[str], list[chess.Board]]:
    tokens = list(line)
    try:
        moves, positions = move_line(board, tokens)
    except ValueError:
        moves, positions = move_line(after, tokens)
        return 'after', moves, positions
    return 'before', moves, positions


def move_facts(board: chess.Board, move: chess.Move) -> MoveFacts:
    piece = board.piece_at(move.from_square)
    if piece is None:
        raise ValueError('The move origin is empty.')
    captured = board.piece_at(move.to_square)
    if board.is_en_passant(move):
        captured = chess.Piece(chess.PAWN, not board.turn)
    san = board.san(move)
    after = board.copy(stack=False)
    after.push(move)
    return {
        'color': chess.COLOR_NAMES[board.turn],
        'piece': chess.piece_name(piece.piece_type),
        'origin': chess.square_name(move.from_square),
        'destination': chess.square_name(move.to_square),
        'uci': move.uci(),
        'san': san,
        'capture': board.is_capture(move),
        'captured_piece': chess.piece_name(captured.piece_type) if captured else None,
        'en_passant': board.is_en_passant(move),
        'castling': board.is_castling(move),
        'promotion': chess.piece_name(move.promotion) if move.promotion else None,
        'check': after.is_check(),
        'checkmate': after.is_checkmate(),
    }


def describe_piece(piece: chess.Piece, square: chess.Square) -> BoardPiece:
    return {'color': chess.COLOR_NAMES[piece.color], 'piece': chess.piece_name(piece.piece_type),
            'square': chess.square_name(square)}


def material(board: chess.Board) -> int:
    return sum(value * (len(board.pieces(piece, chess.WHITE)) - len(board.pieces(piece, chess.BLACK)))
               for piece, value in VALUES.items())


def extract_board_context(
    fen: str,
    move_uci: str,
    pv_uci: list[str] | tuple[str, ...] | str | None = None,
    stockfish_info: EngineAnalysis | None = None,
) -> BoardContext:
    board = read_board(fen)
    move = legal_move(board, move_uci)
    facts = move_facts(board, move)
    after = board.copy(stack=False)
    after.push(move)
    start, pv, _ = variation(board, after, words(pv_uci))
    undefended, hanging, targets, pins = [], [], [], []
    for square, piece in sorted(after.piece_map().items()):
        description = describe_piece(piece, square)
        if after.is_pinned(piece.color, square):
            pins.append(description)
        if piece.piece_type == chess.KING:
            continue
        attackers = after.attackers(not piece.color, square)
        defenders = after.attackers(piece.color, square)
        if not any(square in after.pin(piece.color, defender) for defender in defenders):
            undefended.append(description)
            if any(square in after.pin(not piece.color, attacker) for attacker in attackers):
                hanging.append(description)
        if piece.color != board.turn and square in after.attacks(move.to_square) and VALUES[piece.piece_type] >= 3:
            targets.append(description)
    result: BoardContext = {
        'fen': board.fen(), 'after_fen': after.fen(), 'side_to_move': facts['color'],
        'move': facts, 'pv': pv, 'pv_start': start, 'undefended_pieces': undefended,
        'hanging_pieces': hanging, 'attacked_high_value_targets': targets, 'pinned_pieces': pins,
        'material_before': material(board), 'material_after': material(after),
        'material_delta': material(after) - material(board),
    }
    if stockfish_info:
        result['eval_before'] = stockfish_info['eval_before']
        result['eval_after'] = stockfish_info['eval_after']
        result['eval_swing_cp'] = stockfish_info['eval_swing_cp']
        result['multipv'] = stockfish_info['multipv']
    return result


def words(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return tuple(value.split())
    if not isinstance(value, (list, tuple)) or not all(isinstance(item, str) for item in value):
        raise ValueError('Moves and tags must be strings or lists of strings.')
    return tuple(value)


@dataclass(frozen=True)
class ChessExample:
    fen: str
    move: str
    explanation: str
    pv: tuple[str, ...] = ()
    themes: tuple[str, ...] = ()
    rating: int | None = None
    opening_tags: tuple[str, ...] = ()
    source_eval: str | None = None

    def record(self) -> dict[str, Any]:
        record = asdict(self)
        if self.source_eval is None:
            del record['source_eval']
        return record

    def context(self, engine: Stockfish | None = None) -> BoardContext:
        evaluation = engine.analyse(self.fen, self.move) if engine else None
        context = extract_board_context(self.fen, self.move, self.pv, evaluation)
        if self.source_eval is not None:
            context['source_eval'] = self.source_eval
        return context


def canonical_example(row: Mapping[str, Any]) -> ChessExample:
    if not isinstance(row, Mapping) or not isinstance(row.get('fen'), str):
        raise ValueError('Each chess record needs a FEN.')
    explanation = row.get('explanation')
    if not isinstance(explanation, str) or not explanation.strip() or len(explanation) > 50000:
        raise ValueError('Each chess record needs a nonempty explanation of at most 50,000 characters.')
    pv = words(row.get('pv'))
    move = row.get('move') or (pv[0] if pv else None)
    if not isinstance(move, str):
        raise ValueError('Each chess record needs a move or a PV.')
    board = read_board(row['fen'])
    parsed = legal_move(board, move)
    if pv:
        after = board.copy(stack=False)
        after.push(parsed)
        _, line, _ = variation(board, after, pv)
        pv = tuple(line)
    source_eval = row.get('source_eval')
    if source_eval is not None and (not isinstance(source_eval, str) or not re.fullmatch(r'(?:[+-]\d+ cp|mate [+-]\d+)', source_eval)):
        raise ValueError('Invalid archived chess evaluation.')
    return ChessExample(board.fen(), parsed.uci(), explanation.strip(), pv, words(row.get('themes')),
                        int(row['rating']) if row.get('rating') is not None else None, words(row.get('opening_tags')),
                        source_eval)


def explain_facts(context: Mapping[str, Any]) -> str:
    move = context['move']
    subject = f"{move['color'].capitalize()}'s {move['piece']} moves from {move['origin']} to {move['destination']}"
    if move['castling']:
        subject = f"{move['color'].capitalize()} castles {'queenside' if move['san'].startswith('O-O-O') else 'kingside'}"
    events = []
    if move['capture']:
        events.append(f"captures a {move['captured_piece']}" + (' en passant' if move['en_passant'] else ''))
    if move['promotion']:
        events.append(f"promotes to a {move['promotion']}")
    if move['checkmate']:
        events.append('gives checkmate')
    elif move['check']:
        events.append('gives check')
    return ', '.join([subject, *events]) + '.'


def adapt_hub_row(preset: str, row: Mapping[str, Any]) -> ChessExample:
    if preset == 'chess-explained':
        return canonical_example(row)
    if preset != 'lichess':
        raise ValueError('Unknown chess dataset preset.')
    board = read_board(row['FEN'])
    moves = words(row['Moves'])
    if len(moves) < 2:
        raise ValueError('A Lichess puzzle needs a setup move and a solution.')
    setup, *solution = moves
    board.push(legal_move(board, setup))
    fen = board.fen()
    context = extract_board_context(fen, solution[0], solution)
    explanation = explain_facts(context)
    if len(solution) > 1:
        line = board.variation_san([chess.Move.from_uci(move) for move in context['pv']])
        explanation += f' The puzzle solution is {line}.'
    return ChessExample(fen, context['move']['uci'], explanation, tuple(context['pv']), words(row.get('Themes')),
                        int(row['Rating']) if row.get('Rating') is not None else None, words(row.get('OpeningTags')))


def adapt_stream_row(preset: str, row: Mapping[str, Any]) -> Iterator[ChessExample]:
    if preset in ('chess-explained', 'waterhorse-chess-data'):
        yield canonical_example(row)
        return
    if preset in ('strategic-game-chess', '400m-full-chess-match'):
        moves = row['Moves'] if preset == 'strategic-game-chess' else row['moves']
        if isinstance(moves, str) and moves.lstrip().startswith('['):
            if len(moves) > 32000:
                raise ValueError('Oversized match move list.')
            try:
                moves = ast.literal_eval(moves)
            except SyntaxError as exc:
                raise ValueError('Malformed match move list.') from exc
        moves, boards = move_line(chess.Board(), words(moves))
        # Spread the examples across the game to avoid replaying the same opening each row.
        for index in sorted({len(moves) * part // 4 for part in (1, 2, 3)}):
            if moves:
                pv = moves[index:index + PV_LIMIT]
                fen = boards[index].fen()
                context = extract_board_context(fen, pv[0], pv)
                yield ChessExample(fen, pv[0], explain_facts(context), tuple(pv))
        return

    if preset == 'chess-position-evaluations':
        pv = words(row['line'])
        score_kind, score = ('mate', row['mate']) if row['mate'] is not None else ('cp', row['cp'])
    elif preset == 'chess-positions-lichess-sf':
        pv = (row['best_move'],)
        score_kind, score = row['eval_type'], row['eval_value']
    else:
        raise ValueError('Unknown chess stream.')
    if not pv or score_kind not in ('cp', 'mate') or type(score) is not int:
        raise ValueError('The engine row needs a move and an evaluation.')
    fen = row['fen']
    context = extract_board_context(fen, pv[0], pv)
    if context['pv_start'] != 'before':
        raise ValueError('The archive principal variation must start at its FEN.')
    label = f'{score:+d} cp' if score_kind == 'cp' else f'mate {score:+d}'
    explanation = explain_facts(context) + f' The archived engine score for this line is {label}.'
    yield ChessExample(context['fen'], context['move']['uci'], explanation, tuple(context['pv']), source_eval=label)


class ChessExampleStream:
    """Reads completion examples on demand and checkpoints the last consumed row."""

    def __init__(self, preset, cancelled, state=None):
        self.preset = preset
        self.cancelled = cancelled
        self.manifest = state.get('manifest') if state else None
        self.cursor = tuple(state.get('position', (*ORIGIN, 0))) if state else (*ORIGIN, 0)
        if len(self.cursor) != 4 or any(type(part) is not int or part < 0 for part in self.cursor):
            raise ValueError('Invalid chess stream position.')
        self.iterator = self._examples()

    def _examples(self):
        self.manifest = self.manifest or resolve_chess_manifest(self.preset, self.cancelled)
        reserved = held_out_positions()
        cursor = self.cursor
        while not self.cancelled():
            produced = False
            rows = chess_rows(self.preset, self.manifest, cursor[:3], self.cancelled)
            try:
                for location, row in rows:
                    try:
                        examples = list(adapt_stream_row(self.preset, row))
                    except (ValueError, TypeError, KeyError):
                        continue
                    start = cursor[3] if location == cursor[:3] else 0
                    for index, example in enumerate(examples[start:], start):
                        if position_key(example.fen) not in reserved:
                            produced = True
                            self.cursor = (*location, index + 1)
                            yield example
            finally:
                rows.close()
            if not produced and cursor == (*ORIGIN, 0):
                raise ValueError('The chess stream has no usable training examples.')
            cursor = (*ORIGIN, 0)
        raise InterruptedError()

    def __next__(self):
        return next(self.iterator)

    def close(self):
        self.iterator.close()

    def state(self):
        return {'dataset': f'hf:{self.preset}', 'manifest': self.manifest, 'position': self.cursor}


def parse_local(text: str, suffix: str) -> Iterator[ChessExample]:
    if suffix == '.jsonl':
        for number, line in enumerate(text.splitlines(), 1):
            if line.strip():
                try:
                    yield canonical_example(json.loads(line))
                except (ValueError, TypeError, KeyError) as exc:
                    raise ValueError(f'JSONL line {number}: {exc}') from exc
        return
    stream = StringIO(text)
    while (game := chess.pgn.read_game(stream)) is not None:
        board = game.board()
        if game.errors or type(board) is not chess.Board or board.chess960 or not board.is_valid():
            raise ValueError('The PGN contains an invalid position or illegal move.')
        for node in game.mainline():
            if node.comment.strip():
                yield canonical_example({'fen': board.fen(), 'move': node.move.uci(), 'explanation': node.comment})
            board.push(node.move)


def format_example(context: Mapping[str, Any], explanation: str | None = None,
                   themes: Iterable[str] = ()) -> tuple[str, str | None]:
    move = context['move']
    def squares(key: str) -> str:
        return ' '.join(f"{p['color'][0]}{'n' if p['piece'] == 'knight' else p['piece'][0]}@{p['square']}"
                        for p in context[key]) or '-'

    events = [key for key in ('capture', 'check', 'checkmate', 'castling', 'en_passant') if move[key]]
    if move['captured_piece']:
        events.append(f"captured={move['captured_piece']}")
    if move['promotion']:
        events.append(f"promoted={move['promotion']}")
    lines = [
        f"<FEN> {context['fen']}",
        f"<MOVE> {move['uci']} {move['san']} {move['color']} {move['piece']} {move['origin']}→{move['destination']}",
        f"<FACTS> events={','.join(events) or '-'}; white material={context['material_before']:+d}→{context['material_after']:+d}",
        f"After move: undefended={squares('undefended_pieces')}; hanging={squares('hanging_pieces')}; "
        f"targets={squares('attacked_high_value_targets')}; pinned={squares('pinned_pieces')}",
    ]
    if context.get('eval_before') is not None and context.get('eval_after') is not None:
        before, after = context['eval_before'], context['eval_after']
        swing = context.get('eval_swing_cp')
        change = f' swing={swing:+d} cp' if swing is not None else ''
        lines.append(f"<EVAL> White POV before={score_label(before)} after={score_label(after)}{change}")
    if context.get('source_eval') is not None:
        lines.append(f"<EVAL> archive line={context['source_eval']}")
    if context['pv']:
        lines.append(f"<PV> {context['pv_start']} " + ' '.join(context['pv'][:PV_LIMIT]))
    for entry in context.get('multipv', []):
        lines.append(f"<PV> engine {entry['rank']} {score_label(entry['score'])} " + ' '.join(entry['pv'][:PV_LIMIT]))
    if tags := tuple(themes):
        lines.append('<THEMES> ' + ' '.join(tags))
    lines.append('<EXPLAIN>')
    return '\n'.join(lines), explanation


def score_label(score: Mapping[str, Any]) -> str:
    if score['mate'] is not None:
        return f"mate {score['mate_color']} {abs(score['mate'])}"
    return f"{score['cp']:+d} cp"


def stockfish_path(setting: str = '') -> str | None:
    command = setting or os.environ.get('STOCKFISH_PATH') or 'stockfish'
    return shutil.which(os.path.expanduser(command))


def score_data(score: chess.engine.PovScore) -> EngineScore:
    white = score.white()
    mate = white.mate()
    return {'cp': white.score(), 'mate': mate,
            'mate_color': ('white' if white > chess.engine.Cp(0) else 'black') if mate is not None else None}


class Stockfish:
    DEPTH = 12
    MULTIPV = 3

    def __init__(self, engine: chess.engine.SimpleEngine):
        self.engine = engine

    def query(self, board: chess.Board, **kwargs):
        # A new UCI game for each search clears the transposition table.
        return self.engine.analyse(board, chess.engine.Limit(depth=self.DEPTH), game=object(), **kwargs)

    def analyse(self, fen: str, notation: str) -> EngineAnalysis:
        board = read_board(fen)
        move = legal_move(board, notation)
        info = self.query(board, multipv=self.MULTIPV)
        after = board.copy()
        after.push(move)
        after_info = self.query(after)
        before_score, after_score = info[0]['score'], after_info['score']
        before_cp, after_cp = before_score.white().score(), after_score.white().score()
        return {'eval_before': score_data(before_score), 'eval_after': score_data(after_score),
                'eval_swing_cp': after_cp - before_cp if before_cp is not None and after_cp is not None else None,
                'multipv': [{'rank': i, 'score': score_data(item['score']),
                             'pv': [m.uci() for m in item.get('pv', [])[:PV_LIMIT]]}
                            for i, item in enumerate(info, 1)]}

    def centipawn_loss(self, board: chess.Board, move: chess.Move) -> int | None:
        best = self.query(board)['score'].pov(board.turn)
        candidate = self.query(board, root_moves=[move])['score'].pov(board.turn)
        best_cp, candidate_cp = best.score(), candidate.score()
        return max(0, best_cp - candidate_cp) if best_cp is not None and candidate_cp is not None else None


@contextmanager
def open_stockfish(path: str | None) -> Iterator[Stockfish | None]:
    engine = None
    try:
        if path:
            try:
                engine = chess.engine.SimpleEngine.popen_uci([path], timeout=8)
                engine.configure({key: value for key, value in {'Threads': 1, 'Hash': 16}.items() if key in engine.options})
            except (OSError, chess.engine.EngineError, TimeoutError):
                if engine:
                    engine.close()
                engine = None
        yield Stockfish(engine) if engine else None
    finally:
        if engine:
            engine.close()


def position_key(fen: str) -> str:
    return ' '.join(fen.split()[:4])


def fixture_examples(seed: int) -> Iterator[ChessExample]:
    """Produce reproducible standard positions with legal, literal explanations."""
    rng = random.Random(seed)
    seen = set()
    while True:
        board = chess.Board()
        target = rng.randrange(6, 64)
        for _ in range(target):
            moves = sorted(board.legal_moves, key=chess.Move.uci)
            if not moves:
                break
            board.push(rng.choice(moves))
        moves = sorted(board.legal_moves, key=chess.Move.uci)
        key = position_key(board.fen())
        if not moves or key in seen:
            continue
        seen.add(key)
        priorities = [m for m in moves if board.is_capture(m) or board.gives_check(m) or board.is_castling(m) or m.promotion]
        move = rng.choice(priorities if priorities and rng.randrange(4) else moves)
        context = extract_board_context(board.fen(), move.uci())
        themes = ('mateIn1',) if context['move']['checkmate'] else ()
        yield ChessExample(board.fen(), move.uci(), explain_facts(context), themes=themes)


# Fixed positions supply common tactical cases alongside seeded middlegames.
TACTICAL_FIXTURES = (
    ('6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1', 'a1a8', ('backRankMate', 'mateIn1'),
     "Ra8# is a back-rank mate in one. Black's pawns on f7, g7 and h7 leave the king without an escape square."),
    ('4k3/P7/8/8/8/8/8/4K3 w - - 0 1', 'a7a8q', ('promotion',), None),
    ('4k3/8/8/8/8/8/8/4K2R w K - 0 1', 'e1g1', ('castling',), None),
    ('4k3/4n3/8/8/8/8/R7/6K1 w - - 0 1', 'a2e2', ('pin',),
     'Re2 pins the black knight on e7 to its king.'),
    ('q3k3/8/8/3N4/8/8/8/4K3 w - - 0 1', 'd5c7', ('fork',),
     'Nc7+ forks the king and the queen on a8.'),
    ('4k3/8/8/3q4/2B5/8/8/4K3 w - - 0 1', 'c4d5', ('hangingPiece',),
     'Bxd5 captures the hanging queen on d5.'),
    ('4q3/4k3/8/8/8/8/R7/6K1 w - - 0 1', 'a2e2', ('skewer',),
     'Re2+ skewers the king and queen. The king must move, leaving the queen on e8 exposed.'),
    ('4k3/8/8/8/4N3/8/8/K3R3 w - - 0 1', 'e4d6', ('discoveredAttack',),
     'Nd6+ opens the e-file for a discovered check from the rook on e1.'),
)


@lru_cache(maxsize=8)
def benchmark_examples(size: int) -> tuple[ChessExample, ...]:
    if not 1 <= size <= BENCHMARK_MAX:
        raise ValueError(f'Benchmark size must be between 1 and {BENCHMARK_MAX}.')
    anchors = []
    for fen, move, themes, explanation in TACTICAL_FIXTURES:
        context = extract_board_context(fen, move)
        anchors.append(ChessExample(context['fen'], move, explanation or explain_facts(context), themes=themes))
    return tuple((anchors + list(islice(fixture_examples(647281), BENCHMARK_MAX - len(anchors))))[:size])


@lru_cache(maxsize=1)
def held_out_positions() -> frozenset[str]:
    return frozenset(position_key(example.fen) for example in benchmark_examples(BENCHMARK_MAX))


def hub_examples(preset: str, limit: int) -> Iterator[ChessExample]:
    import pyarrow.parquet as parquet

    repo = HUB_PRESETS[preset]
    revision_url = f'https://huggingface.co/api/datasets/{repo}/revision/refs%2Fconvert%2Fparquet'
    revision = get_json(revision_url)['sha']
    metadata = get_json('https://datasets-server.huggingface.co/parquet?dataset=' + quote(repo, safe=''))
    if not re.fullmatch(r'[a-f0-9]{40}', revision) or metadata.get('partial') or get_json(revision_url)['sha'] != revision:
        raise ValueError('The Hugging Face dataset listing is incomplete or changed during discovery.')
    files = sorted((item for item in metadata['parquet_files'] if item['split'] == 'train'),
                   key=lambda item: (item['config'], item['filename']))
    if not files:
        raise ValueError(f'{repo} has no available train Parquet shards.')
    required = {'FEN', 'Moves'} if preset == 'lichess' else {'fen', 'move', 'explanation'}
    optional = {'Rating', 'Themes', 'OpeningTags'} if preset == 'lichess' else {'pv', 'themes', 'rating', 'opening_tags'}
    remaining = limit
    for item in files:
        config, name = quote(item['config'], safe=''), quote(item['filename'], safe='')
        shard = {'url': f'https://huggingface.co/datasets/{repo}/resolve/{revision}/{config}/train/{name}',
                 'size': item['size']}
        with remote_file(shard, authenticated_request, urlopen, lambda: False) as source:
            reader = parquet.ParquetFile(source)
            names = set(reader.schema_arrow.names)
            if not required <= names:
                raise ValueError(f'{repo} is missing required chess fields.')
            columns = sorted((required | optional) & names)
            for batch in reader.iter_batches(batch_size=min(64, remaining), columns=columns):
                for row in batch.to_pylist():
                    try:
                        example = adapt_hub_row(preset, row)
                    except (ValueError, KeyError, TypeError):
                        continue
                    yield example
                    remaining -= 1
                    if not remaining:
                        return


class ChessDatasets:
    def __init__(self, directory: Path, store):
        self.directory = directory
        self.store = store

    def catalog(self) -> list[dict[str, Any]]:
        datasets = [{'id': 'starter', 'label': 'Built-in chess examples', 'ready': True}]
        datasets.extend({'id': key, 'label': label, 'ready': (self.directory / f'{key}.jsonl').is_file(), 'hub': True}
                        for key, label in HUB_PRESETS.items())
        datasets.extend({'id': f'hf:{key}', 'label': source['repo'], 'ready': True, 'streaming': True}
                        for key, source in CHESS_STREAMS.items())
        datasets.extend({'id': f'local:{p.stem}', 'label': p.stem, 'ready': True}
                        for p in sorted(self.directory.glob('*.jsonl')) if p.stem not in HUB_PRESETS)
        return datasets

    def save(self, dataset: str, examples: Iterable[ChessExample], replace: bool = False) -> dict[str, Any]:
        if not re.fullmatch(r'[a-zA-Z0-9_-]{1,80}', dataset):
            raise ValueError('Use a dataset name with letters, digits, underscores or hyphens.')
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.directory / f'{dataset}.jsonl'
        temp = tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=self.directory, delete=False)
        try:
            with temp:
                count = 0
                for example in examples:
                    temp.write(json.dumps(example.record(), ensure_ascii=False) + '\n')
                    count += 1
            if not count:
                raise ValueError('No usable chess examples. PGN uploads require move comments.')
            if replace:
                Path(temp.name).replace(path)
            else:
                try:
                    os.link(temp.name, path)
                except FileExistsError as exc:
                    raise ValueError('A chess dataset with that name already exists.') from exc
        finally:
            Path(temp.name).unlink(missing_ok=True)
        return {'id': dataset if dataset in HUB_PRESETS else f'local:{dataset}', 'examples': count}

    def load(self, dataset: str) -> Iterator[ChessExample]:
        if dataset == 'starter':
            yield from islice(fixture_examples(307281), 256)
            return
        name = dataset if dataset in HUB_PRESETS else dataset.removeprefix('local:')
        if not re.fullmatch(r'[a-zA-Z0-9_-]{1,80}', name) or (dataset not in HUB_PRESETS and dataset != f'local:{name}'):
            raise ValueError('Choose a chess dataset preset or a chess upload.')
        path = self.directory / f'{name}.jsonl'
        if not path.is_file():
            raise ValueError('Import or upload the selected chess dataset first.')
        with path.open(encoding='utf-8') as handle:
            for line in handle:
                yield canonical_example(json.loads(line))

    def examples(self, dataset: str) -> list[ChessExample]:
        stream = dataset.startswith('hf:')
        if stream and dataset.removeprefix('hf:') not in CHESS_STREAMS:
            raise ValueError('Choose a chess streaming dataset.')
        reserved = held_out_positions()
        corrections = (canonical_example(row) for row in self.store.feedback('chess-sft'))
        cached = () if stream else self.load(dataset)
        examples = [example for source in (cached, corrections) for example in source
                    if position_key(example.fen) not in reserved]
        if not stream and not examples:
            raise ValueError('The dataset has no examples outside the held-out chess benchmark.')
        return examples
