"""Conservative pattern-based chess verification and checkpoint benchmark reports."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from hashlib import sha256
import json
from pathlib import Path
import re
import shutil
from tempfile import NamedTemporaryFile
from typing import Any, TypedDict

import chess
import chess.engine

from .chess_pipeline import (
    BENCHMARK_VERSION, Stockfish, benchmark_examples, format_example, legal_move,
    move_facts, move_line, open_stockfish, read_board, variation,
)
from .storage import now

PIECES = 'king|queen|rook|bishop|knight|pawn'
SQUARE = '[a-h][1-8]'
CLAIM_SQUARE = '[a-z][0-9]{1,2}'
FLAGS = re.IGNORECASE
PIECE = rf"(?:(?P<color>white|black)(?:['’]s)?\s+)?(?P<piece>{PIECES})"
PIECE_AT = re.compile(
    rf"\b{PIECE}s?\s+(?:(?:is\s+|are\s+|is\s+not\s+)?(?:on|at|from|to)\s+)?"
    rf"(?P<square>{CLAIM_SQUARE})\b(?P<others>(?:(?:,\s*(?:and\s+)?|\s+and\s+){CLAIM_SQUARE}\b)*)", FLAGS,
)
SQUARE_PIECE = re.compile(rf"\b(?P<square>{CLAIM_SQUARE})\s+{PIECE}\b", FLAGS)
PIECE_ACTION = re.compile(
    rf"\b{PIECE}\s+(?:(?:on|at)\s+(?P<origin_at>{CLAIM_SQUARE})\s+)?"
    rf"(?:moves?|goes?|jumps?|slides?|lands?|advances?|retreats?|captures?|takes?)\s+"
    rf"(?:(?:from\s+(?P<origin>{CLAIM_SQUARE})\s+)?(?:to|on)\s+)?(?P<square>{CLAIM_SQUARE})\b", FLAGS,
)
MOVE_TOKEN = re.compile(
    rf"(?<![\w@])(?:{SQUARE}{SQUARE}[qrbnQRBN]?|[O0]-[O0](?:-[O0])?|"
    rf"[KQRBN][a-h]?[1-8]?x?{SQUARE}|(?:[a-h]x)?{SQUARE}(?:=?[QRBN])?)[+#]?(?!\w)"
)
TACTICAL = {
    'checkmate': re.compile(r'\b(?:checkmat(?:e[sd]?|ing)|mat(?:e[sd]?|ing))\b(?!\s+in\s+(?:[2-9]|two|three|four|five)\b)', FLAGS),
    'check': re.compile(r'\bcheck(?:s|ed|ing)?\b', FLAGS),
    'capture': re.compile(r'\b(?:captur(?:e[sd]?|ing)|takes|took)\b(?!\s+(?:place|time|advantage|control|space|the initiative)\b)', FLAGS),
    'promotion': re.compile(r'\bpromot(?:e[sd]?|ing|ion)\b', FLAGS),
    'castling': re.compile(r'\bcastl(?:e[sd]?|ing)\b', FLAGS),
    'pin': re.compile(r'\bpin(?:s|ned|ning)?\b', FLAGS),
}
THEMES = {
    'fork': r'\bfork(?:s|ed|ing)?\b|\bdouble attack\b',
    'pin': TACTICAL['pin'].pattern,
    'skewer': r'\bskewer(?:s|ed|ing)?\b',
    'hangingPiece': r'\b(?:hang(?:s|ing)?|en prise|undefended (?:piece|pawn|queen|rook|bishop|knight)|free (?:piece|pawn|queen|rook|bishop|knight))\b',
    'discoveredAttack': r'\bdiscovered (?:attack|check)\b|\buncover(?:s|ed|ing)?\s+(?:a |an )?(?:check|attack)\b',
    'backRankMate': r'\bback[- ]rank (?:mate|checkmate)\b|\bmating on the back rank\b',
    'mateIn1': r'\b(?:check)?mate in (?:one|1)\b|\bimmediate (?:check)?mate\b|\bcheckmat(?:e[sd]?|ing)\b',
    'promotion': TACTICAL['promotion'].pattern,
    'castling': TACTICAL['castling'].pattern,
    'sacrifice': r'\bsacrific(?:e[sd]?|ing)\b|\bgives? up material\b',
    'deflection': r'\bdeflect(?:s|ed|ing|ion)?\b',
    'attraction': r'\b(?:attraction|draws? the king out)\b',
    'clearance': r'\bclearance\b|\bclear(?:s|ed|ing)? (?:a|the) (?:file|rank|diagonal|square)\b',
    'interference': r'\binterfer(?:e[sd]?|ing|ence)\b',
    'doubleCheck': r'\bdouble check\b',
    'enPassant': r'\ben passant\b',
    'underPromotion': r'\bunder[- ]?promot(?:e[sd]?|ion|ing)\b|\bpromot(?:e[sd]?|ing) to a (?:rook|bishop|knight)\b',
    'trappedPiece': r'\btraps? (?:the |a )?(?:queen|rook|knight|bishop|piece)\b|\btrapped (?:queen|rook|knight|bishop|piece)\b',
    'intermezzo': r'\b(?:intermezzo|zwischenzug|in-between move)\b',
    'xRayAttack': r'\bx[- ]ray(?: attack)?\b',
    'mateIn2': r'\b(?:check)?mate in (?:two|2)\b',
    'mateIn3': r'\b(?:check)?mate in (?:three|3)\b',
    'mateIn4': r'\b(?:check)?mate in (?:four|4)\b',
    'mateIn5': r'\b(?:check)?mate in (?:five|5)\b',
}
NEGATION = re.compile(r"\b(?:no|not|never|without|neither|cannot|can’t|can't|doesn['’]t|isn['’]t)\b"
                      r"(?:\s+[\w'’-]+){0,4}\s*$", FLAGS)
HYPOTHETICAL = re.compile(r'\b(?:could|might|would|threatens?|threatening|prevents?|prevents?\s+a|'
                          r'avoids?|avoiding|sets? up|looks? for|if)\s+(?:[\w-]+\s+){0,4}$', FLAGS)
CLAUSE = re.compile(r'(?<!\d)[.!?]|[;\n]|,\s*(?:but|while|whereas)\b', FLAGS)
POLARITY = {
    'white_winning': re.compile(r"\bwhite\s+(?:(?:is|was|will be|remains)\s+winning|wins|has\s+(?:a\s+)?(?:decisive|winning|clear)\s+advantage)\b", FLAGS),
    'black_winning': re.compile(r"\bblack\s+(?:(?:is|was|will be|remains)\s+winning|wins|has\s+(?:a\s+)?(?:decisive|winning|clear)\s+advantage)\b", FLAGS),
    'equal': re.compile(r"\b(?:equal(?:ity|izes|ises)?|balanced|roughly even|dead even)\b", FLAGS),
    'mate': re.compile(r'\b(?:forced|forces?|forcing)\s+(?:a\s+)?(?:check)?mate\b', FLAGS),
    'blunder': re.compile(r'\bblunder(?:s|ed|ing)?\b', FLAGS),
}
RECOMMENDATION = re.compile(r'\b(?:best|better|alternative|strongest|recommended|prefer|recommend|should(?:\s+play)?)\b', FLAGS)
WEIGHTS = {'grounding_accuracy': 25, 'move_legality_rate': 25, 'tactical_fact_accuracy': 25,
           'theme_f1': 15, 'stockfish_polarity_agreement': 10}
MAX_NEW_TOKENS = 64
EVAL_BATCH_SIZE = 8


class Claim(TypedDict):
    category: str
    claim: str
    start: int
    end: int
    passed: bool
    reason: str | None


def claim(category: str, text: str, start: int, end: int, passed: bool, reason: str) -> Claim:
    return {'category': category, 'claim': text[start:end], 'start': start, 'end': end,
            'passed': bool(passed), 'reason': None if passed else reason}


def board_positions(context: Mapping[str, Any]) -> list[chess.Board]:
    board = read_board(context['fen'])
    after = board.copy(stack=False)
    after.push(legal_move(board, context['move']['uci']))
    boards = [board, after]
    if context.get('pv'):
        _, _, line = variation(board, after, context['pv'])
        boards.extend(line[1:])
    for engine_line in context.get('multipv', []):
        _, line = move_line(board, engine_line['pv'])
        boards.extend(line[1:])
    return list({b.fen(): b for b in boards}.values())


def piece_claims(text: str, boards: list[chess.Board]) -> tuple[list[Claim], list[tuple[int, int]]]:
    results, spans, seen = [], [], set()
    for pattern in (PIECE_AT, SQUARE_PIECE, PIECE_ACTION):
        for match in pattern.finditer(text):
            spans.append(match.span())
            fields = match.groupdict()
            piece = match['piece'].lower()
            piece_type = chess.PIECE_NAMES.index(piece)
            color = (match['color'] or '').lower()
            description = ' '.join(filter(None, (color, piece)))
            start, end = match.span()
            origin = fields.get('origin') or fields.get('origin_at')
            squares = [(match['square'], start, end)]
            if origin:
                squares.insert(0, (origin, start, end))
            if fields.get('others'):
                offset = match.start('others')
                squares.extend((tag[0], offset + tag.start(), offset + tag.end())
                               for tag in re.finditer(CLAIM_SQUARE, fields['others'], FLAGS))
            prefix = text[max(0, start - 35):start]
            negated = bool(NEGATION.search(prefix)) or bool(re.search(r"\b(?:isn['’]t|is\s+not)\b", match[0], FLAGS))
            piece_boards = [board for board in boards if not color or chess.COLOR_NAMES[board.turn] == color]
            for square, square_start, square_end in squares:
                square = square.lower()
                key = piece, color, square, match.start(), negated
                if key in seen:
                    continue
                seen.add(key)
                if square not in chess.SQUARE_NAMES:
                    results.append(claim('grounding', text, square_start, square_end, False, f'{square} is not a valid chess square.'))
                    continue
                destination = chess.parse_square(square)
                found = any((p := board.piece_at(destination)) and p.piece_type == piece_type
                            and (not color or chess.COLOR_NAMES[p.color] == color) for board in boards)
                if not found and pattern is PIECE_ACTION and square != (origin or '').lower() and piece_type == chess.PAWN:
                    # A promotion reaches the last rank as a new piece.
                    found = any(move.promotion and move.to_square == destination
                                and (origin is None or chess.square_name(move.from_square) == origin.lower())
                                for board in piece_boards for move in board.legal_moves)
                reason = (f'A {description} occupies {square} in a reference position.' if negated else
                          f'No {description} occupies {square} before the move, after it, or along a reference PV.')
                results.append(claim('grounding', text, square_start, square_end, found != negated, reason))
            if origin and pattern is PIECE_ACTION and not negated and not HYPOTHETICAL.search(prefix):
                notation = (origin + match['square']).lower()
                legal = False
                if all(square in chess.SQUARE_NAMES for square in (notation[:2], notation[2:])):
                    src, dest = chess.parse_square(notation[:2]), chess.parse_square(notation[2:])
                    legal = any(board.piece_type_at(src) == piece_type and any(move.to_square == dest for move in board.generate_legal_moves(from_mask=chess.BB_SQUARES[src]))
                                for board in piece_boards)
                results.append(claim('legality', text, start, end, legal, f'No legal {description} move matches {notation}.'))
    return results, spans


def mentioned_moves(text: str, boards: list[chess.Board], grounding_spans: list[tuple[int, int]]):
    mentions = []
    for match in MOVE_TOKEN.finditer(text):
        token = match[0]
        if re.fullmatch(SQUARE, token) and (
            any(start <= match.start() < end for start, end in grounding_spans)
            or re.search(r'\b(?:on|at|from|to|square|occupies|attacks|controls|defends|protects)\s*$', text[:match.start()], FLAGS)
        ):
            continue
        clean = token.rstrip('+#').replace('0', 'O')
        candidates = []
        for board in boards:
            try:
                candidates.append((board, legal_move(board, clean)))
            except ValueError:
                continue
        mentions.append((match, candidates))
    return mentions


def clause_prefix(text: str, start: int) -> tuple[int, str]:
    boundary = 0
    for separator in CLAUSE.finditer(text, 0, start):
        boundary = separator.end()
    return boundary, text[boundary:start]


def moves_and_notation(text: str, boards: list[chess.Board], mentions) -> list[Claim]:
    results = []
    for match, candidates in mentions:
        start, end = match.span()
        token = match[0]
        # Pawn squares can denote moves or board locations. Only unambiguous SAN contributes grounding.
        notation = re.match(rf'(?P<piece>[KQRBN])(?:[a-h]?[1-8]?x?)(?P<square>{SQUARE})', token)
        promotion = re.search(rf'({SQUARE})=?([QRBN])[+#]?$', token)
        if notation or promotion:
            symbol, square = (notation['piece'], notation['square']) if notation else (promotion[2], promotion[1])
            found = any((piece := board.piece_at(chess.parse_square(square))) and piece.symbol().upper() == symbol for board in boards)
            if not found:
                # A legal hypothetical SAN move establishes its arrival square.
                found = bool(candidates)
            results.append(claim('grounding', text, start, end, found,
                                 f'No {chess.piece_name(chess.PIECE_SYMBOLS.index(symbol.lower()))} is grounded on {square}.'))
        results.append(claim('legality', text, start, end, bool(candidates),
                             f'{token} is illegal from every reference position.'))
        if candidates:
            facts = [move_facts(board, move) for board, move in candidates]
            for marker, event in (('#', 'checkmate'), ('+', 'check'), ('x', 'capture'), ('=', 'promotion')):
                if marker in token:
                    results.append(claim('tactical', text, start, end, any(f[event] for f in facts),
                                         f'{token} does not represent {event} in any legal reference position.'))
    return results


def tactical_claims(text: str, context: Mapping[str, Any], boards: list[chess.Board], mentions) -> list[Claim]:
    results = []
    for event, pattern in TACTICAL.items():
        for match in pattern.finditer(text):
            start, end = match.span()
            boundary, prefix = clause_prefix(text, start)
            if HYPOTHETICAL.search(prefix) or (event == 'checkmate' and re.search(r'\b(?:forced|forces?|forcing)\s+(?:a\s+)?$', prefix, FLAGS)):
                continue
            negated = bool(NEGATION.search(prefix))
            previous = [(token, candidates) for token, candidates in mentions if boundary <= token.start() and token.end() <= start]
            facts = [move_facts(board, move) for board, move in previous[-1][1]] if previous else [context['move']]
            if event == 'pin':
                states = boards[:2]
                if previous:
                    states = [b.copy(stack=False) for b, _ in previous[-1][1]]
                    for state, (_, move) in zip(states, previous[-1][1]):
                        state.push(move)
                tail = text[end:CLAUSE.search(text, end).start() if CLAUSE.search(text, end) else len(text)]
                target = re.search(rf"\b{PIECE}\b(?:\s+(?:on|at)\s+(?P<square>{SQUARE})\b)?", tail, FLAGS)
                if not target:
                    target = re.search(rf"\b{PIECE}\b(?:\s+(?:on|at)\s+(?P<square>{SQUARE})\b)?\s+(?:(?:is|was)\s+)?$", prefix, FLAGS)
                actual = any(state.is_pinned(piece.color, square)
                             and (not target or chess.piece_name(piece.piece_type) == target['piece'].lower()
                                  and (not target['square'] or chess.square_name(square) == target['square'].lower())
                                  and (not target['color'] or chess.COLOR_NAMES[piece.color] == target['color'].lower()))
                             for state in states for square, piece in state.piece_map().items())
            else:
                field = 'captured_piece' if event == 'capture' else event
                tail = text[end:]
                target = re.match(rf'\s+(?:to\s+)?(?:(?:the|a|an|white|black)\s+)*(?P<piece>{PIECES})\b', tail, FLAGS)
                expected = target['piece'].lower() if target and event in ('capture', 'promotion') else None
                actual = any(f[field] == expected if expected else bool(f[event]) for f in facts)
            results.append(claim('tactical', text, start, end, actual != negated,
                                 f'Reference boards contradict the claim about {event}.'))
    return results


def theme_scores(text: str, tags: Iterable[str]) -> dict[str, Any] | None:
    tags = tuple(tags)
    if not tags:
        return None
    canonical = {key.lower(): key for key in THEMES}
    aliases = {'discoveredcheck': 'discoveredAttack'}
    expected = {aliases.get(tag.lower(), canonical.get(tag.lower())) for tag in tags} - {None}
    if not expected:
        return None
    predicted = {key for key, pattern in THEMES.items() if any(
        not NEGATION.search(clause_prefix(text, match.start())[1]) for match in re.finditer(pattern, text, FLAGS))}
    # Chess explanations usually say "gives checkmate" without counting plies.
    if 'mateIn1' in predicted and 'mateIn1' not in expected and not re.search(r'\bmate in (?:1|one)\b', text, FLAGS):
        predicted.remove('mateIn1')
    return {'expected': sorted(expected), 'predicted': sorted(predicted), 'tp': len(expected & predicted),
            'fp': len(predicted - expected), 'fn': len(expected - predicted)}


def engine_checks(text: str, context: Mapping[str, Any], mentions, engine: Stockfish | None):
    score = context.get('eval_after')
    if score is None:
        return [], []
    results, recommendations = [], []
    cp, mate_color = score['cp'], score['mate_color']
    mover = context['move']['color']
    swing = context.get('eval_swing_cp')
    before_mate = context['eval_before']['mate_color']
    actual = {
        'white_winning': mate_color == 'white' or cp is not None and cp >= 150,
        'black_winning': mate_color == 'black' or cp is not None and cp <= -150,
        'equal': cp is not None and abs(cp) <= 50,
        'mate': mate_color is not None,
        'blunder': (swing is not None and swing * (1 if mover == 'white' else -1) <= -200)
                   or (mate_color is not None and mate_color != mover and before_mate != mate_color)
                   or (before_mate == mover and mate_color != mover),
    }
    for verdict, pattern in POLARITY.items():
        for match in pattern.finditer(text):
            _, prefix = clause_prefix(text, match.start())
            if HYPOTHETICAL.search(prefix):
                continue
            supported = actual[verdict]
            if verdict == 'mate':
                side = re.search(r'\b(white|black)\b', prefix, FLAGS)
                if not side:
                    side = re.match(r'\s+(?:in\s+\w+\s+)?(?:for|by)\s+(white|black)\b', text[match.end():], FLAGS)
                if side:
                    supported = mate_color == side[1].lower()
            results.append(claim('stockfish', text, *match.span(), bool(supported) != bool(NEGATION.search(prefix)),
                                 f'Stockfish does not support the {verdict.replace("_", " ")} verdict.'))
    if engine:
        seen = set()
        for match, candidates in mentions:
            _, prefix = clause_prefix(text, match.start())
            suffix = re.split(CLAUSE, text[match.end():], maxsplit=1)[0]
            if not RECOMMENDATION.search(prefix[-64:]) and not re.match(r'\s+is\s+(?:the\s+)?(?:best|better|recommended)\b', suffix, FLAGS):
                continue
            # The earliest legal position is the original position or the first point at which a reply is possible.
            if candidates:
                board, move = candidates[0]
                key = (board.fen(), move.uci())
                if key not in seen:
                    seen.add(key)
                    recommendations.append({'move': move.uci(), 'fen': board.fen(),
                                            'centipawn_loss': engine.centipawn_loss(board, move)})
    return results, recommendations


def verify_explanation(context: Mapping[str, Any], text: str, themes: Iterable[str] = (),
                       engine: Stockfish | None = None) -> dict[str, Any]:
    boards = board_positions(context)
    checks, grounding_spans = piece_claims(text, boards)
    mentions = mentioned_moves(text, boards, grounding_spans)
    checks.extend(moves_and_notation(text, boards, mentions))
    checks.extend(tactical_claims(text, context, boards, mentions))
    engine_claims, recommendations = engine_checks(text, context, mentions, engine)
    checks.extend(engine_claims)
    errors = [item for item in checks if not item['passed']]
    themes_result = theme_scores(text, themes)
    if themes_result:
        expected, predicted = set(themes_result['expected']), set(themes_result['predicted'])
        if expected - predicted:
            missing = ', '.join(sorted(expected - predicted))
            errors.append(claim('theme', text, 0, 0, False, f'Reference themes missing from the explanation: {missing}.'))
        for theme in sorted(predicted - expected):
            match = re.search(THEMES[theme], text, FLAGS)
            if match:
                errors.append(claim('theme', text, *match.span(), False, f'{theme} is absent from the reference theme tags.'))
    if not checks:
        errors.append(claim('coverage', text, 0, 0, False, 'No verifiable chess claim was found.'))
    return {'passed': not errors, 'checks': sorted(checks, key=lambda item: item['start']), 'errors': errors,
            'themes': themes_result, 'recommendations': recommendations}


def percent(count: int, total: int) -> float:
    return round(100 * count / total, 2) if total else 0.0


def aggregate(samples: list[dict[str, Any]], engine_available: bool) -> dict[str, Any]:
    claims = [item for sample in samples for item in sample['verification']['checks']]
    def accuracy(category: str, absent: float | None = 0.0) -> float | None:
        relevant = [item for item in claims if item['category'] == category]
        return percent(sum(item['passed'] for item in relevant), len(relevant)) if relevant else absent

    themes = [sample['verification']['themes'] for sample in samples if sample['verification']['themes'] is not None]
    tp, fp, fn = (sum(theme[key] for theme in themes) for key in ('tp', 'fp', 'fn'))
    losses = [move['centipawn_loss'] for sample in samples for move in sample['verification']['recommendations']
              if move['centipawn_loss'] is not None]
    metrics = {
        'sample_count': len(samples),
        'passed_samples': sum(sample['verification']['passed'] for sample in samples),
        'grounding_accuracy': accuracy('grounding'),
        'piece_hallucination_rate': percent(sum(any(item['category'] == 'grounding' for item in sample['verification']['errors'])
                                                for sample in samples), len(samples)),
        'move_legality_rate': accuracy('legality'),
        'tactical_fact_accuracy': accuracy('tactical'),
        'theme_precision': percent(tp, tp + fp) if themes else None,
        'theme_recall': percent(tp, tp + fn) if themes else None,
        'theme_f1': percent(2 * tp, 2 * tp + fp + fn) if themes else None,
        'stockfish_polarity_agreement': accuracy('stockfish') if engine_available else None,
        'average_centipawn_loss': round(sum(losses) / len(losses), 2) if losses else None,
    }
    eligible = {key: weight for key, weight in WEIGHTS.items() if metrics[key] is not None}
    metrics['composite_chess_accuracy'] = round(sum(metrics[key] * weight for key, weight in eligible.items()) /
                                                sum(eligible.values()), 2)
    return metrics


def evaluate_checkpoint(generate: Callable[[str], str], checkpoint: str, run_id: str | None, step: int,
                        size: int, engine_path: str | None, *,
                        generate_batch: Callable[[list[str]], list[str]] | None = None) -> dict[str, Any]:
    examples = benchmark_examples(size)

    def run(engine: Stockfish | None):
        samples = []
        for offset in range(0, len(examples), EVAL_BATCH_SIZE):
            batch = examples[offset:offset + EVAL_BATCH_SIZE]
            contexts = [example.context(engine) for example in batch]
            prompts = [format_example(context, themes=example.themes)[0] for example, context in zip(batch, contexts)]
            outputs = generate_batch(prompts) if generate_batch else [generate(prompt) for prompt in prompts]
            for example, context, text in zip(batch, contexts, outputs, strict=True):
                samples.append({'fen': example.fen, 'move': example.move, 'reference': example.explanation,
                                'explanation': text, 'context': context,
                                'verification': verify_explanation(context, text, example.themes, engine)})
        return samples

    with open_stockfish(engine_path) as engine:
        engine_name = engine.engine.id.get('name', 'Stockfish') if engine else None
        try:
            samples = run(engine)
        except (OSError, chess.engine.EngineError, TimeoutError):
            if not engine:
                raise
            samples = run(None)
            engine_name = None
    protocol = {'benchmark': BENCHMARK_VERSION, 'size': size, 'decoding': 'greedy', 'max_new_tokens': MAX_NEW_TOKENS,
                'generation_batch_size': EVAL_BATCH_SIZE if generate_batch else 1, 'engine': engine_name,
                'engine_depth': Stockfish.DEPTH if engine_name else None, 'engine_multipv': Stockfish.MULTIPV if engine_name else None}
    benchmark_id = sha256(json.dumps([protocol, [example.record() for example in examples]], sort_keys=True).encode()).hexdigest()[:16]
    return {'checkpoint': checkpoint, 'run_id': run_id, 'step': step, 'created_at': now(),
            'benchmark_id': benchmark_id, 'protocol': protocol, 'stockfish_available': engine_name is not None,
            'metrics': aggregate(samples, engine_name is not None), 'samples': samples}


class ChessReports:
    def __init__(self, directory: Path):
        self.directory = directory / '.chess-eval'

    @staticmethod
    def write_json(path: Path, payload: dict[str, Any]) -> None:
        with NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent, delete=False) as temp:
            try:
                json.dump(payload, temp, ensure_ascii=False)
                temp.flush()
                Path(temp.name).replace(path)
            finally:
                Path(temp.name).unlink(missing_ok=True)

    def save(self, report: dict[str, Any], source: str) -> dict[str, Any]:
        report['source'] = source
        directory = self.directory / report['checkpoint']
        directory.mkdir(parents=True, exist_ok=True)
        stem = f"{source}-{report['benchmark_id']}"
        self.write_json(directory / f'{stem}.json', report)
        self.write_json(directory / f'{stem}.summary.json', {key: value for key, value in report.items() if key != 'samples'})
        return report

    def get(self, checkpoint: str, source: str | None = None, benchmark_id: str | None = None) -> dict[str, Any] | None:
        directory = self.directory / checkpoint
        candidates = [path for path in directory.glob(f'{source or "*"}-{benchmark_id or "*"}.json')
                      if not path.name.endswith('.summary.json')]
        if not candidates:
            return None
        path = max(candidates, key=lambda item: item.stat().st_mtime_ns)
        return json.loads(path.read_text(encoding='utf-8'))

    def history(self, checkpoints: Iterable[Mapping[str, Any]], run_id: str | None) -> list[dict[str, Any]]:
        history = []
        for checkpoint in checkpoints:
            if run_id and checkpoint.get('run_id') != run_id:
                continue
            for path in (self.directory / checkpoint['name']).glob('*.summary.json'):
                try:
                    history.append(json.loads(path.read_text(encoding='utf-8')))
                except FileNotFoundError:
                    continue
        return sorted(history, key=lambda row: (row['step'], row['checkpoint'], row['created_at']))

    def delete(self, checkpoint: str) -> None:
        directory = self.directory / checkpoint
        if directory.is_dir():
            shutil.rmtree(directory)
