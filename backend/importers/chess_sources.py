"""Pinned, bounded readers for the chess datasets in the streaming catalog."""

from codecs import getreader
from collections.abc import Callable, Iterator, Mapping
from fnmatch import fnmatchcase
import json
import re
import tarfile
from typing import Any
from urllib.parse import quote, urlsplit
from urllib.request import urlopen

import chess.pgn
import pyarrow.parquet as parquet

from .ranges import remote_file


CHESS_STREAMS = {
    'strategic-game-chess': {
        'repo': 'laion/strategic_game_chess', 'label': 'LAION Strategic Game Chess', 'size': '838 GB',
        'license': 'CC BY 4.0', 'files': 'chess_game_*.parquet', 'columns': ('Moves',),
        'note': 'Replays UCI game records into verified positions and move explanations.',
    },
    '400m-full-chess-match': {
        'repo': 'Yousap123/400m_full_chess_match', 'label': '400M Full Chess Match', 'size': '178 GB',
        'license': 'MIT', 'files': 'data/shard_*.parquet', 'columns': ('moves',),
        'note': 'Replays match move lists into verified positions and move explanations.',
    },
    'chess-position-evaluations': {
        'repo': 'Lichess/chess-position-evaluations', 'label': 'Lichess Position Evaluations', 'size': '44.1 GB',
        'license': 'CC0 1.0', 'files': 'data/data_*.parquet', 'columns': ('fen', 'line', 'cp', 'mate'),
        'note': 'Provides legal principal variations and archived Stockfish scores.',
    },
    'chess-explained': {
        'repo': 'aicrowd/ChessExplained', 'label': 'ChessExplained', 'size': '1.04 GB',
        'license': 'MIT', 'files': 'ChessExplained_*.parquet', 'columns': ('fen', 'move', 'explanation'),
        'note': 'Uses the supplied move explanations.',
    },
    'waterhorse-chess-data': {
        'repo': 'Waterhorse/chess_data', 'label': 'Waterhorse Chess Data', 'size': '57.6 MB annotated PGN',
        'license': 'See annotated PGN source terms',
        'files': 'chessclip_data/annotated_pgn/annotated_pgn_free.tar.gz',
        'note': 'Streams the published annotated PGN archive and uses move comments.',
    },
    'chess-positions-lichess-sf': {
        'repo': 'avewright/chess-positions-lichess-sf', 'label': 'Chess Positions Lichess Stockfish', 'size': '~19 GB',
        'license': 'See source card', 'files': 'data/train-*.parquet',
        'columns': ('fen', 'best_move', 'eval_type', 'eval_value'),
        'note': 'Uses the training shards with legal moves and archived engine scores.',
    },
}

Cursor = tuple[int, int, int]
ORIGIN: Cursor = (0, 0, 0)


def tree_files(repo: str, revision: str, request) -> Iterator[dict[str, Any]]:
    root = f'/api/datasets/{repo}/tree/{revision}'
    url = 'https://huggingface.co' + root + '?recursive=true'
    seen = set()
    while url:
        target = urlsplit(url)
        if target.scheme != 'https' or target.netloc != 'huggingface.co' or target.path != root or url in seen:
            raise ValueError('Invalid Hugging Face tree pagination link.')
        seen.add(url)
        with urlopen(request(url), timeout=15) as response:
            yield from json.load(response)
            link = re.search(r'<([^>]+)>;\s*rel="next"', response.headers.get('Link', ''))
            url = link[1] if link else ''


def resolve_chess_manifest(preset: str, cancelled: Callable[[], bool]) -> dict[str, Any]:
    from .hub import authenticated_request, get_json

    source = CHESS_STREAMS[preset]
    repo = source['repo']
    revision = get_json(f'https://huggingface.co/api/datasets/{repo}/revision/main')['sha']
    if not re.fullmatch(r'[a-f0-9]{40}', revision):
        raise ValueError('Invalid pinned revision.')
    with_entries = tree_files(repo, revision, authenticated_request)
    files = []
    for entry in with_entries:
        if cancelled():
            raise InterruptedError()
        if entry['type'] == 'file' and fnmatchcase(entry['path'], source['files']):
            files.append((entry['path'], entry['size']))
    if not files:
        raise ValueError(f'{repo} has no matching training files.')
    return {'revision': revision, 'files': sorted(files)}


def annotated_games(stream, cancelled: Callable[[], bool]) -> Iterator[dict[str, Any]]:
    with tarfile.open(fileobj=stream, mode='r|gz') as archive:
        for entry in archive:
            if cancelled():
                raise InterruptedError()
            if not entry.isfile() or not entry.name.lower().endswith('.pgn'):
                continue
            with archive.extractfile(entry) as contents:
                handle = getreader('utf-8-sig')(contents, errors='replace')
                while not cancelled():
                    game = chess.pgn.read_game(handle)
                    if game is None:
                        break
                    board = game.board()
                    if game.errors or type(board) is not chess.Board or board.chess960 or not board.is_valid():
                        continue
                    for node in game.mainline():
                        text = node.comment.strip()
                        if text:
                            yield {'fen': board.fen(), 'move': node.move.uci(), 'explanation': text}
                        board.push(node.move)
                if cancelled():
                    raise InterruptedError()


def chess_rows(preset: str, manifest: Mapping[str, Any], cursor: Cursor,
               cancelled: Callable[[], bool]) -> Iterator[tuple[Cursor, dict[str, Any]]]:
    from .hub import authenticated_request

    source = CHESS_STREAMS[preset]
    revision = manifest['revision']
    if not re.fullmatch(r'[a-f0-9]{40}', revision) or not 0 < len(manifest['files']) <= 10000:
        raise ValueError('Invalid pinned chess manifest.')
    for file_index, (path, size) in enumerate(manifest['files']):
        if file_index < cursor[0]:
            continue
        if (not re.fullmatch(r'[\w./-]+', path, re.ASCII) or any(part in ('', '.', '..') for part in path.split('/'))
                or not fnmatchcase(path, source['files']) or type(size) is not int or size <= 0):
            raise ValueError('Invalid pinned chess data file.')
        if cancelled():
            raise InterruptedError()
        url = f"https://huggingface.co/datasets/{source['repo']}/resolve/{revision}/{quote(path)}"
        first_group, first_row = cursor[1:] if file_index == cursor[0] else (0, 0)
        if preset == 'waterhorse-chess-data':
            with urlopen(authenticated_request(url), timeout=15) as stream:
                for index, row in enumerate(annotated_games(stream, cancelled)):
                    if index >= first_row:
                        yield (file_index, 0, index), row
            continue
        shard = {'url': url, 'size': size}
        # Native game shards can contain larger compressed column chunks than converted samples.
        with remote_file(shard, authenticated_request, urlopen, cancelled, max_read_size=512 * 1024 * 1024) as file:
            reader = parquet.ParquetFile(file)
            columns = source['columns']
            if not set(columns) <= set(reader.schema_arrow.names):
                raise ValueError(f"{source['repo']} is missing required chess fields.")
            for group in range(first_group, reader.num_row_groups):
                minimum = first_row if group == first_group else 0
                index = 0
                batches = reader.iter_batches(batch_size=32, row_groups=[group], columns=columns, use_threads=False)
                for batch in batches:
                    if cancelled():
                        raise InterruptedError()
                    if index + batch.num_rows <= minimum:
                        index += batch.num_rows
                        continue
                    for row in batch.to_pylist():
                        if index >= minimum:
                            yield (file_index, group, index), row
                        index += 1
