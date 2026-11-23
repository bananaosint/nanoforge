"""Resumable direct Hugging Face imports; complete train splits, bounded memory.

Original downloads, partial conversions and provenance are hidden under .imports.
Only completed, normalized UTF-8 corpora become visible to the training mixer.
"""
import hashlib
import json
import os
import sqlite3
import threading
import time
from pathlib import Path
from urllib.parse import quote
from urllib.request import Request, urlopen
from .translate import LocalTranslator, CJK

CATALOG = [
    {'id': 'tinystories', 'repo': 'roneneldan/TinyStoriesInstruct', 'label': 'TinyStories Instruct', 'kind': 'text', 'size': '2.66 GB', 'license': 'See source card'},
    {'id': 'hh-rlhf', 'repo': 'Anthropic/hh-rlhf', 'label': 'HH-RLHF', 'kind': 'preference', 'size': '~172 MB download', 'license': 'MIT'},
    {'id': 'oasst1', 'repo': 'OpenAssistant/oasst1', 'label': 'OpenAssistant OASST1', 'kind': 'sft', 'size': '~40 MB download', 'license': 'Apache 2.0', 'note': 'English conversations only'},
    {'id': 'no-robots', 'repo': 'HuggingFaceH4/no_robots', 'label': 'No Robots', 'kind': 'sft', 'size': '~10 MB download', 'license': 'CC BY-NC 4.0'},
    {'id': 'goat', 'repo': 'tiedong/goat', 'label': 'GOAT arithmetic', 'kind': 'sft', 'size': '~206 MB download', 'license': 'Apache 2.0'},

]
# Explicit CLI/API opt-in only; excluded from the catalog and Import all.
OPTIONAL_CHINESE = {'id': 'chinese-r1-en', 'repo': 'Congliu/Chinese-DeepSeek-R1-Distill-data-110k', 'label': 'R1 Distill · translated to English', 'kind': 'translated-sft', 'size': '~376 MB + 75 MB translator', 'license': 'Apache 2.0', 'note': 'Local translation of prompt, reasoning and answer; can take many hours'}
BY_ID = {item['id']: item for item in [*CATALOG, OPTIONAL_CHINESE]}


def get_json(url):
    with urlopen(Request(url, headers={'User-Agent': 'NanoForge/1.1'}), timeout=45) as response:
        return json.load(response)


def normalize(key, row, parent_lookup=None, translate=None):
    """Return one training record, or None for explicit language/deletion filters."""
    if key == 'hh-rlhf':
        chosen, rejected = row['chosen'], row['rejected']
        marker = '\n\nAssistant:'
        win_prompt, sep, win = chosen.rpartition(marker)
        lose_prompt, sep2, lose = rejected.rpartition(marker)
        if not sep or not sep2 or win_prompt != lose_prompt:
            raise ValueError('Preference branches do not share a final assistant prompt.')
        return {'prompt': win_prompt.lstrip(), 'completion': win.strip(), 'chosen': win.strip(), 'rejected': lose.strip()}
    if key == 'oasst1':
        if row.get('role') != 'assistant' or row.get('lang') != 'en' or row.get('deleted'):
            return None
        turns, current, seen = [], row.get('parent_id'), set()
        while current:
            if current in seen:
                raise ValueError('Cycle in OASST message tree.')
            seen.add(current)
            parent = parent_lookup(current)
            if not parent or parent.get('deleted') or parent.get('lang') != 'en':
                return None
            turns.append(('User' if parent['role'] == 'prompter' else 'Assistant') + ': ' + parent['text'])
            current = parent.get('parent_id')
        if not turns:
            return None
        return {'prompt': '\n\n'.join(reversed(turns)), 'completion': row['text']}
    if key == 'no-robots':
        messages = row['messages']
        last = max((i for i, message in enumerate(messages) if message['role'] == 'assistant'), default=-1)
        if last < 1:
            return None
        prompt = '\n\n'.join(message['role'].capitalize() + ': ' + message['content'] for message in messages[:last])
        return {'prompt': prompt, 'completion': messages[last]['content']}
    if key == 'goat':
        prompt = row.get('instruction') or row.get('input')
        context = row.get('input', '')
        if context and context != prompt:
            prompt += '\n' + context
        return {'prompt': prompt, 'completion': row.get('output') or row['answer']}
    if key == 'chinese-r1-en':
        if translate is None:
            raise ValueError('Chinese data requires translation before import.')
        fields = {key: translate(row.get(key, '')) for key in ('input', 'reasoning_content', 'content')}
        if any(CJK.search(text) for text in fields.values()):
            raise ValueError('Translation retained Chinese characters; excluded for review.')
        completion = ('<think>\n' + fields['reasoning_content'] + '\n</think>\n\n' if fields['reasoning_content'] else '') + fields['content']
        return {'prompt': fields['input'], 'completion': completion, 'reasoning': fields['reasoning_content'], 'answer': fields['content'], 'translation': 'Argos zh-en 1.9 / CTranslate2 int8; machine translated, review recommended'}
    raise ValueError('Unknown dataset adapter.')


class HubImports:
    def __init__(self, datasets):
        self.datasets = datasets
        self.root = datasets.directory / '.imports'
        self.root.mkdir(exist_ok=True)
        self.manifest = self.root / 'jobs.json'
        self.lock = threading.RLock()
        self.cancel = threading.Event()
        self.thread = None
        self.jobs = json.loads(self.manifest.read_text()) if self.manifest.exists() else {}
        for job in self.jobs.values():
            if job['state'] in ('queued', 'downloading', 'converting', 'translating'):
                job['state'] = 'paused'
                job['error'] = 'Previous process stopped. Resume to continue.'

    def _persist(self):
        temporary = self.manifest.with_suffix('.tmp')
        temporary.write_text(json.dumps(self.jobs, ensure_ascii=False, indent=2), encoding='utf-8')
        temporary.replace(self.manifest)

    def update(self, key, **fields):
        with self.lock:
            self.jobs.setdefault(key, {}).update(fields, updated_at=time.time())
            self._persist()

    def status(self):
        with self.lock:
            return {'catalog': CATALOG, 'jobs': json.loads(json.dumps(self.jobs)), 'active': bool(self.thread and self.thread.is_alive())}

    def start(self, key, max_records=0):
        if key != 'all' and key not in BY_ID:
            raise ValueError('Choose a dataset from the import catalog.')
        with self.lock:
            if self.thread and self.thread.is_alive():
                raise ValueError('An import is already running. Pause it before starting another.')
            keys = [item['id'] for item in CATALOG] if key == 'all' else [key]
            keys = [k for k in keys if self.jobs.get(k, {}).get('state') not in ('completed', 'sampled')]
            for k in keys:
                previous = self.jobs.get(k, {})
                if previous and previous.get('max_records', 0) != max_records:
                    raise ValueError('Resume with the original record limit. Use a separate NanoForge data directory for a different sample/full import.')
                self.jobs[k] = {**previous, 'state': 'queued', 'repo': BY_ID[k]['repo'], 'max_records': max_records, 'error': None}
            self._persist()
            self.cancel.clear()
            self.thread = threading.Thread(target=self._worker, args=(keys,), name='nanoforge-import', daemon=False)
            self.thread.start()
            return self.status()

    def forget_output(self, name):
        with self.lock:
            for key, job in list(self.jobs.items()):
                if job.get('output') == name:
                    self.jobs[key] = {'state':'paused', 'repo':job['repo'], 'max_records':job.get('max_records',0), 'error':'Output was deleted. Resume to import it again.'}
            self._persist()

    def pause(self):
        self.cancel.set()
        return self.status()

    def shutdown(self):
        self.cancel.set()
        if self.thread:
            self.thread.join()

    def _worker(self, keys):
        for key in keys:
            if self.cancel.is_set():
                self.update(key, state='paused')
                continue
            try:
                self._import(key)
            except InterruptedError:
                self.update(key, state='paused')
            except Exception as exc:
                self.update(key, state='error', error=str(exc))

    def _download(self, key, url, path, expected=None):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and (expected is None or path.stat().st_size == expected):
            return path
        partial = path.with_suffix(path.suffix + '.partial')
        if expected is not None and partial.exists() and partial.stat().st_size == expected:
            partial.replace(path)
            return path
        last_error = None
        for attempt in range(3):
            if self.cancel.is_set():
                raise InterruptedError()
            start = partial.stat().st_size if partial.exists() else 0
            request = Request(url + ('&' if '?' in url else '?') + 'nanoforge=' + str(time.time_ns()),
                              headers={'Range': f'bytes={start}-', 'User-Agent': 'NanoForge/1.1'} if start else {'User-Agent': 'NanoForge/1.1'})
            try:
                with urlopen(request, timeout=45) as response:
                    append = response.status == 206 and start > 0
                    if append and not response.headers.get('Content-Range', '').startswith(f'bytes {start}-'):
                        raise ValueError('Server returned an invalid resume range.')
                    downloaded = start if append else 0
                    previous = 0.0
                    with partial.open('ab' if append else 'wb') as target:
                        while chunk := response.read(1024 * 1024):
                            if self.cancel.is_set():
                                raise InterruptedError()
                            target.write(chunk)
                            downloaded += len(chunk)
                            if time.monotonic() - previous > 0.8:
                                self.update(key, state='downloading', file=path.name, bytes_downloaded=downloaded, bytes_total=expected)
                                previous = time.monotonic()
                if expected is not None and partial.stat().st_size != expected:
                    raise ValueError(f'Incomplete download: expected {expected} bytes, received {partial.stat().st_size}.')
                partial.replace(path)
                return path
            except InterruptedError:
                raise
            except Exception as exc:
                last_error = exc
                if self.cancel.wait(1 + attempt):
                    raise InterruptedError()
        raise ValueError(f'Download failed; resume will retry: {last_error}')

    @staticmethod
    def _rows(files):
        import pyarrow.parquet as parquet
        for path in files:
            for batch in parquet.ParquetFile(path).iter_batches(batch_size=64):
                yield from batch.to_pylist()

    def _import(self, key):
        definition = BY_ID[key]
        job = self.jobs[key]
        folder = self.root / key
        folder.mkdir(exist_ok=True)
        info = {'sha':job['revision'], 'cardData':{'license':job.get('license')}} if job.get('revision') else get_json('https://huggingface.co/api/datasets/' + definition['repo'])
        revision = job.get('revision') or info['sha']
        self.update(key, revision=revision, source_url='https://huggingface.co/datasets/' + definition['repo'], license=info.get('cardData', {}).get('license') or definition['license'])
        if key == 'tinystories' and not job.get('max_records'):
            name = 'TinyStoriesInstruct.txt'
            target = self.datasets.directory / name
            if not target.exists():
                url = f"https://huggingface.co/datasets/{definition['repo']}/resolve/{revision}/TinyStories-Instruct-train.txt"
                tree = get_json(f"https://huggingface.co/api/datasets/{definition['repo']}/tree/{revision}?recursive=true")
                size = next(entry['size'] for entry in tree if entry['path'] == 'TinyStories-Instruct-train.txt')
                self._download(key, url, target, size)
            self.update(key, state='completed', output=name, rows=None, note='Complete original training text; original validation file is not imported.')
            return
        parquet_info = {'partial':False,'parquet_files':[dict(s,config='default',split='train') for s in job['parquet_sources']]} if job.get('parquet_sources') else get_json('https://datasets-server.huggingface.co/parquet?dataset=' + quote(definition['repo'], safe=''))
        if parquet_info.get('partial'):
            raise ValueError('Hugging Face reports only partial Parquet coverage. Refusing an incomplete full import.')
        shards = [item for item in parquet_info['parquet_files'] if item['config'] == 'default' and item['split'] == 'train']
        if not shards:
            raise ValueError('No public training split is available.')
        converted = job.get('parquet_revision') or get_json('https://huggingface.co/api/datasets/' + definition['repo'] + '/revision/refs%2Fconvert%2Fparquet')['sha']
        self.update(key, parquet_revision=converted)
        for shard in shards:
            shard['url'] = shard['url'].replace('resolve/refs%2Fconvert%2Fparquet/', 'resolve/' + converted + '/')
        files = []
        for i, shard in enumerate(shards):
            # Limited verification samples use the public rows endpoint instead;
            # a sample should not require downloading an entire large shard.
            if job.get('max_records'):
                break
            files.append(self._download(key, shard['url'], folder / f'shard-{i}.parquet', shard['size']))
        self.update(key, parquet_sources=[{'url':s['url'],'size':s['size']} for s in shards])
        rows = lambda: self._sample_rows(definition['repo'], job['max_records']) if job.get('max_records') else self._rows(files)
        if key == 'tinystories':
            # Dataset viewer rows are individual lines; only used for explicit samples.
            adapter = lambda row: {'text': row['text']}
            lookup_db = None
        elif key == 'oasst1':
            lookup_db = sqlite3.connect(folder / 'parents.sqlite3')
            lookup_db.execute('CREATE TABLE IF NOT EXISTS messages(id TEXT PRIMARY KEY,payload TEXT)')
            for row in rows():
                lookup_db.execute('INSERT OR REPLACE INTO messages VALUES(?,?)', (row['message_id'], json.dumps(row)))
            lookup_db.commit()
            def lookup(message_id):
                found = lookup_db.execute('SELECT payload FROM messages WHERE id=?', (message_id,)).fetchone()
                return json.loads(found[0]) if found else None
            adapter = lambda row: normalize(key, row, parent_lookup=lookup)
        else:
            lookup_db = None
            adapter = lambda row: normalize(key, row)
        translator = None
        if key == 'chinese-r1-en':
            translator = LocalTranslator(self.root / 'translator', lambda url,path,expected: self._download(key,url,path,expected), self.cancel.is_set)
            adapter = lambda row: normalize(key, row, translate=translator)
        name = key + ('-sample' if job.get('max_records') else '') + '.jsonl'
        output = folder / 'normalized.partial'
        review = folder / 'review.jsonl'
        checkpoint = dict(job.get('checkpoint', {'seen':0,'written':0,'skipped':0,'review':0,'bytes':0,'review_bytes':0}))
        try:
            with output.open('r+b' if output.exists() else 'w+b') as target, review.open('r+b' if review.exists() else 'w+b') as rejected:
                target.truncate(checkpoint['bytes']); target.seek(0,2)
                rejected.truncate(checkpoint.get('review_bytes',0)); rejected.seek(0,2)
                for number, row in enumerate(rows()):
                    if number < checkpoint['seen']:
                        continue
                    if self.cancel.is_set():
                        raise InterruptedError()
                    if translator or number % 100 == 0:
                        self.update(key, state='translating' if translator else 'converting', current_record=number + 1)
                    try:
                        record = adapter(row)
                        if record:
                            record.update(source=definition['repo'], source_row=number)
                            target.write((json.dumps(record,ensure_ascii=False)+'\n').encode('utf-8'))
                            checkpoint['written'] += 1
                        else:
                            checkpoint['skipped'] += 1
                    except InterruptedError:
                        raise
                    except ValueError as exc:
                        rejected.write((json.dumps({'source_row':number,'error':str(exc),'original':row},ensure_ascii=False)+'\n').encode('utf-8'))
                        checkpoint['review'] += 1
                    # Commit after every row for slow translations; batch for normal imports.
                    if translator or (number + 1) % 100 == 0:
                        target.flush(); rejected.flush()
                        checkpoint.update(seen=number+1,bytes=target.tell(),review_bytes=rejected.tell())
                        self.update(key, checkpoint=dict(checkpoint), rows=checkpoint['written'], excluded=checkpoint['skipped'], needs_review=checkpoint['review'])
                target.flush(); rejected.flush()
                checkpoint.update(seen=number+1 if 'number' in locals() else 0,bytes=target.tell(),review_bytes=rejected.tell())
            if not checkpoint['written']:
                raise ValueError('No usable records were produced. Review the language filters or review.jsonl.')
            output.replace(self.datasets.directory / name)
            self.update(key, state='sampled' if job.get('max_records') else 'completed', output=name, checkpoint=dict(checkpoint), rows=checkpoint['written'], excluded=checkpoint['skipped'], needs_review=checkpoint['review'], note='Machine-translated records require quality review.' if translator else 'Original train split; OASST filters English conversations.')
            # Normalized output is sufficient for training. Keep translator/cache and
            # provenance, but release large intermediate Parquet shards.
            for path in files:
                path.unlink(missing_ok=True)
        finally:
            if translator:
                translator.close()
            if lookup_db:
                lookup_db.close()

    def _sample_rows(self, repo, limit):
        offset = 0
        while offset < limit:
            page = get_json('https://datasets-server.huggingface.co/rows?dataset=' + quote(repo,safe='') + f'&config=default&split=train&offset={offset}&length={min(100,limit-offset)}')
            items = page.get('rows', [])
            if not items:
                return
            for item in items:
                if item.get('truncated_cells'):
                    raise ValueError('Dataset viewer truncated a sample record. Use a full Parquet import instead.')
                yield item['row']
            offset += len(items)
