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
from .expanded import EXTRA_CATALOG, EXTRA_IDS, normalize_extra
from .raw_sources import persona_rows, cornell_rows, CORNELL_URL
from urllib.error import HTTPError

CATALOG = [
    {'id': 'tinystories', 'repo': 'roneneldan/TinyStoriesInstruct', 'label': 'TinyStories Instruct', 'kind': 'text', 'size': '2.66 GB', 'license': 'See source card'},
    {'id': 'hh-rlhf', 'repo': 'Anthropic/hh-rlhf', 'label': 'HH-RLHF', 'kind': 'preference', 'size': '~172 MB download', 'license': 'MIT'},
    {'id': 'oasst1', 'repo': 'OpenAssistant/oasst1', 'label': 'OpenAssistant OASST1', 'kind': 'sft', 'size': '~40 MB download', 'license': 'Apache 2.0', 'note': 'English conversations only'},
    {'id': 'no-robots', 'repo': 'HuggingFaceH4/no_robots', 'label': 'No Robots', 'kind': 'sft', 'size': '~10 MB download', 'license': 'CC BY-NC 4.0'},
    {'id': 'goat', 'repo': 'tiedong/goat', 'label': 'GOAT arithmetic', 'kind': 'sft', 'size': '~206 MB download', 'license': 'Apache 2.0'},

]
CATALOG.extend(EXTRA_CATALOG)

# Explicit CLI/API opt-in only; excluded from the catalog and Import all.
OPTIONAL_CHINESE = {'id': 'chinese-r1-en', 'repo': 'Congliu/Chinese-DeepSeek-R1-Distill-data-110k', 'label': 'R1 Distill · translated to English', 'kind': 'translated-sft', 'size': '~376 MB + 75 MB translator', 'license': 'Apache 2.0', 'note': 'Local translation of prompt, reasoning and answer; can take many hours'}
BY_ID = {item['id']: item for item in [*CATALOG, OPTIONAL_CHINESE]}


def authenticated_request(url,headers=None):
    from urllib.parse import urlparse
    request=Request(url,headers={'User-Agent':'NanoForge/1.2',**(headers or {})})
    token=os.environ.get('HF_TOKEN')
    if token and urlparse(url).hostname in ('huggingface.co','datasets-server.huggingface.co'):
        request.add_unredirected_header('Authorization','Bearer '+token)
    return request


def get_json(url):
    for attempt in range(3):
        try:
            with urlopen(authenticated_request(url),timeout=45) as response:
                return json.load(response)
        except HTTPError as exc:
            if exc.code not in (429,500,502,503,504) or attempt==2:
                raise
            time.sleep(attempt+1)


def normalize(key, row, parent_lookup=None, translate=None):
    """Return one training record, or None for explicit language/deletion filters."""
    if key in EXTRA_IDS:
        return normalize_extra(key,row)
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
            if job['state'] in ('queued', 'downloading', 'converting', 'translating', 'finalizing'):
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
        keys = list(dict.fromkeys(key)) if isinstance(key,list) else ([item['id'] for item in CATALOG] if key == 'all' else [key])
        if not keys or any(k not in BY_ID for k in keys):
            raise ValueError('Choose datasets from the import catalog.')
        with self.lock:
            if self.thread and self.thread.is_alive():
                raise ValueError('An import is already running. Pause it before starting another.')
            pending = {}
            for k in keys:
                previous = dict(self.jobs.get(k, {}))
                state = previous.get('state')
                if state == 'completed' or (state == 'sampled' and previous.get('max_records') == max_records):
                    continue
                if previous and previous.get('max_records',0) != max_records:
                    if state == 'sampled' and max_records == 0:
                        history = dict(previous.get('previous_outputs',{}))
                        history[previous['output']] = {'sft':previous.get('sft_pairs',0),'dpo':previous.get('dpo_pairs',0)}
                        previous = {k:v for k,v in previous.items() if k in ('revision','parquet_revision','parquet_sources','license','source_url')}
                        previous['previous_outputs'] = history
                    else:
                        raise ValueError('Resume with the original record limit. A completed sample can be upgraded to a full import.')
                pending[k] = {**previous,'state':'queued','repo':BY_ID[k]['repo'],'max_records':max_records,'error':None}
            self.jobs.update(pending)
            self._persist()
            self.cancel.clear()
            self.thread = threading.Thread(target=self._worker,args=(list(pending),),name='nanoforge-import',daemon=False)
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
            except HTTPError as exc:
                blocked=exc.code in (401,403)
                self.update(key,state='blocked' if blocked else 'error',error=('Hugging Face access denied. Verify the repository ID or set HF_TOKEN for authorized private/gated access.' if blocked else str(exc)))
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
            request = authenticated_request(url + ('&' if '?' in url else '?') + 'nanoforge=' + str(time.time_ns()),
                              headers={'Range': f'bytes={start}-', 'User-Agent': 'NanoForge/1.1'} if start else {'User-Agent': 'NanoForge/1.1'})
            try:
                with urlopen(request, timeout=45) as response:
                    append = response.status == 206 and start > 0
                    if append and not response.headers.get('Content-Range', '').startswith(f'bytes {start}-'):
                        raise ValueError('Server returned an invalid resume range.')
                    downloaded = start if append else 0
                    response_size=response.headers.get('Content-Length')
                    response_expected=downloaded+int(response_size) if response_size and response_size.isdigit() else None
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
                if response_expected is not None and downloaded!=response_expected:
                    raise ValueError('The server closed an incomplete download; resume will retry.')
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

    def _stream_shards(self,key,shards,folder,skip,boundary):
        """Retain only the current compressed shard; commit output before deletion."""
        offset=0
        for i,shard in enumerate(shards):
            if self.cancel.is_set():
                raise InterruptedError()
            counts=dict(self.jobs[key].get('shard_rows',{}))
            known=counts.get(str(i))
            if known is not None and offset+known <= skip:
                (folder/f'shard-{i}.parquet').unlink(missing_ok=True)
                offset+=known
                continue
            path=self._download(key,shard['url'],folder/f'shard-{i}.parquet',shard['size'])
            count=0
            for row in self._rows([path]):
                yield offset+count,row
                count+=1
            counts[str(i)]=count
            self.update(key,shard_rows=counts)
            boundary(offset+count)
            path.unlink(missing_ok=True)
            offset+=count

    def _raw_rows(self,key,definition,revision,folder,limit):
        from itertools import islice
        if definition['raw']=='persona':
            file='train_self_original.txt'
            tree=get_json(f"https://huggingface.co/api/datasets/{definition['repo']}/tree/{revision}?recursive=true")
            expected=next(item['size'] for item in tree if item['path']==file)
            url=f"https://huggingface.co/datasets/{definition['repo']}/resolve/{revision}/{file}"
            path=self._download(key,url,folder/file,expected)
            factory=lambda:persona_rows(path)
        else:
            url=CORNELL_URL
            path=self._download(key,url,folder/'cornell.zip')
            factory=lambda:cornell_rows(path,folder/'utterances.sqlite3',self.cancel.is_set)
        self.update(key,raw_source=url,variant='train_self_original' if key=='persona-chat' else 'original Cornell archive')
        return (lambda:islice(factory(),limit) if limit else factory()),[path]

    def _disk_sample_rows(self,key,shards,folder,limit,files):
        """Explicit catalog opt-in for a source with an oversized Parquet row group.

        Download to disk, never join hundreds of MB of HTTP ranges in Python.
        Only requested rows are normalized; successful imports remove the shard.
        """
        from itertools import islice
        remaining=limit
        for index,shard in enumerate(shards):
            if remaining<=0:return
            path=self._download(key,shard['url'],folder/f'shard-{index}.parquet',shard['size'])
            files.append(path)
            iterator=self._rows([path])
            try:
                for row in islice(iterator,remaining):
                    if self.cancel.is_set():raise InterruptedError()
                    remaining-=1
                    yield row
            finally:iterator.close()

    def _import(self, key):
        definition=BY_ID[key]
        job=self.jobs[key]
        folder=self.root/key
        folder.mkdir(exist_ok=True)
        limit=job.get('max_records',0)
        name=key+('-sample' if limit else '')+'.jsonl'
        output=folder/'normalized.partial'
        final=self.datasets.directory/name
        # Recover an interruption between the final output rename and metadata save.
        if job.get('finalize_pending'):
            if output.exists():
                output.replace(final)
            if not final.exists() or final.stat().st_size != job['checkpoint']['bytes']:
                raise ValueError('Finalized import is missing or changed; refusing a corrupt resume.')
            self.update(key,state='sampled' if limit else 'completed',finalize_pending=False,output=name)
            return
        info={'sha':job['revision'],'cardData':{'license':job.get('license')}} if job.get('revision') else get_json('https://huggingface.co/api/datasets/'+definition['repo'])
        revision=job.get('revision') or info['sha']
        self.update(key,revision=revision,source_url='https://huggingface.co/datasets/'+definition['repo'],license=info.get('cardData',{}).get('license') or definition['license'],config=definition.get('config','default'),split='train')
        if key=='tinystories' and not limit:
            name='TinyStoriesInstruct.txt'
            target=self.datasets.directory/name
            url=f"https://huggingface.co/datasets/{definition['repo']}/resolve/{revision}/TinyStories-Instruct-train.txt"
            expected=job.get('source_bytes')
            if expected is None:
                tree=get_json(f"https://huggingface.co/api/datasets/{definition['repo']}/tree/{revision}?recursive=true")
                expected=next(entry['size'] for entry in tree if entry['path']=='TinyStories-Instruct-train.txt')
                self.update(key,source_bytes=expected)
            self._download(key,url,target,expected)
            self.update(key,state='completed',output=name,rows=None,sft_pairs=0,dpo_pairs=0,note='Complete original training text; original validation is not imported.')
            return
        checkpoint=dict(job.get('checkpoint',{'seen':0,'written':0,'skipped':0,'review':0,'bytes':0,'review_bytes':0}))
        checkpoint.setdefault('sft',checkpoint['written'] if definition['kind'] in ('sft','preference','translated-sft') else 0)
        checkpoint.setdefault('dpo',checkpoint['written'] if key=='hh-rlhf' else 0)
        files=[]
        shards=[]
        lookup_db=None
        translator=None
        adapter=lambda row:normalize(key,row)
        try:
            if definition.get('raw'):
                rows,files=self._raw_rows(key,definition,revision,folder,limit)
            else:
                config=definition.get('config','default')
                parquet_info={'partial':False,'parquet_files':[dict(s,config=config,split='train') for s in job['parquet_sources']]} if job.get('parquet_sources') else get_json('https://datasets-server.huggingface.co/parquet?dataset='+quote(definition['repo'],safe=''))
                if parquet_info.get('partial'):
                    raise ValueError('Hugging Face reports partial Parquet coverage; refusing an incomplete import.')
                shards=[item for item in parquet_info['parquet_files'] if item['config']==config and item['split']=='train']
                if not shards:
                    raise ValueError('No public training split is available for '+config)
                converted=job.get('parquet_revision') or get_json('https://huggingface.co/api/datasets/'+definition['repo']+'/revision/refs%2Fconvert%2Fparquet')['sha']
                for shard in shards:
                    shard['url']=shard['url'].replace('resolve/refs%2Fconvert%2Fparquet/','resolve/'+converted+'/')
                self.update(key,parquet_revision=converted,parquet_sources=[{'url':s['url'],'size':s['size']} for s in shards],download_bytes=sum(s['size'] for s in shards),shard_count=len(shards))
                rows=lambda:self._sample_rows(definition['repo'],limit,config,shards)
                if limit and definition.get('sample_download_shard'):
                    self.update(key,sample_method='disk-backed shard',note='This source sample downloads its Parquet shard to disk before reading the requested rows.')
                    rows=lambda:self._disk_sample_rows(key,shards,folder,limit,files)
                if key=='oasst1' and not limit:
                    files=[self._download(key,s['url'],folder/f'shard-{i}.parquet',s['size']) for i,s in enumerate(shards)]
                    rows=lambda:self._rows(files)
            if key=='tinystories':
                adapter=lambda row:{'text':row['text']}
            if key=='oasst1':
                lookup_db=sqlite3.connect(folder/'parents.sqlite3')
                lookup_db.execute('CREATE TABLE IF NOT EXISTS messages(id TEXT PRIMARY KEY,payload TEXT)')
                for row in rows():
                    if self.cancel.is_set():raise InterruptedError()
                    lookup_db.execute('INSERT OR REPLACE INTO messages VALUES(?,?)',(row['message_id'],json.dumps(row)))
                lookup_db.commit()
                def lookup(message_id):
                    found=lookup_db.execute('SELECT payload FROM messages WHERE id=?',(message_id,)).fetchone()
                    return json.loads(found[0]) if found else None
                adapter=lambda row:normalize(key,row,parent_lookup=lookup)
            if key=='chinese-r1-en':
                translator=LocalTranslator(self.root/'translator',lambda url,path,expected:self._download(key,url,path,expected),self.cancel.is_set)
                adapter=lambda row:normalize(key,row,translate=translator)
            interval=1000 if definition.get('massive') else 100
            review=folder/'review.jsonl'
            with output.open('r+b' if output.exists() else 'w+b') as target, review.open('r+b' if review.exists() else 'w+b') as rejected:
                target.truncate(checkpoint['bytes']);target.seek(0,2)
                rejected.truncate(checkpoint.get('review_bytes',0));rejected.seek(0,2)
                def commit(seen):
                    target.flush();rejected.flush()
                    checkpoint.update(seen=seen,bytes=target.tell(),review_bytes=rejected.tell())
                    self.update(key,checkpoint=dict(checkpoint),rows=checkpoint['written'],source_records=seen,excluded=checkpoint['skipped'],needs_review=checkpoint['review'],sft_pairs=checkpoint['sft'],dpo_pairs=checkpoint['dpo'])
                indexed=enumerate(rows()) if limit or definition.get('raw') or key=='oasst1' else self._stream_shards(key,shards,folder,checkpoint['seen'],commit)
                seen=checkpoint['seen']
                for number,row in indexed:
                    seen=number+1
                    if number < checkpoint['seen']:
                        continue
                    if self.cancel.is_set():raise InterruptedError()
                    if translator or number % interval==0:
                        self.update(key,state='translating' if translator else 'converting',current_record=number+1)
                    try:
                        normalized=adapter(row)
                        records=normalized if isinstance(normalized,list) else [normalized] if normalized else []
                        if not records:checkpoint['skipped']+=1
                        for record in records:
                            record.update(source=definition['repo'],source_row=number)
                            target.write((json.dumps(record,ensure_ascii=False)+'\n').encode('utf-8'))
                            checkpoint['written']+=1
                            checkpoint['sft']+=int(bool(record.get('prompt') and record.get('completion')))
                            checkpoint['dpo']+=int(bool(record.get('prompt') and record.get('chosen') and record.get('rejected')))
                    except InterruptedError:
                        raise
                    except (ValueError,KeyError,TypeError) as exc:
                        rejected.write((json.dumps({'source_row':number,'error':str(exc),'original':row},ensure_ascii=False)+'\n').encode('utf-8'))
                        checkpoint['review']+=1
                    if translator or (number+1)%interval==0:commit(seen)
                commit(seen)
            if not checkpoint['written']:
                raise ValueError('No usable records were produced. Check access, language filters, or review.jsonl.')
            self.update(key,state='finalizing',finalize_pending=True,output=name)
            output.replace(final)
            self.update(key,state='sampled' if limit else 'completed',finalize_pending=False,note='Explicit sample; upgrade to full when ready.' if limit else 'Original train split normalized; exclusions/review counts are recorded.')
            for path in files:path.unlink(missing_ok=True)
        finally:
            if translator:translator.close()
            if lookup_db:lookup_db.close()

    def _range_rows(self,shards):
        import pyarrow.parquet as parquet
        from .ranges import RangeReader
        for shard in shards:
            def fetch(start,end):
                if self.cancel.is_set():raise InterruptedError()
                url=shard['url']+('&' if '?' in shard['url'] else '?')+'nanoforge_range='+str(time.time_ns())
                request=authenticated_request(url,{'Range':f'bytes={start}-{end}'})
                with urlopen(request,timeout=45) as response:
                    if response.status==206:
                        expected=f'bytes {start}-{end}/'
                        if not response.headers.get('Content-Range','').startswith(expected):
                            raise OSError('Server returned the wrong Parquet byte range.')
                        return response.read(end-start+2)
                    if response.status==200 and shard['size']<=4*1024*1024:
                        data=response.read(shard['size']+1)
                        return data[start:end+1]
                    raise OSError('Server ignored HTTP range requests; a full import is required.')
            with RangeReader(shard['size'],fetch) as file:
                for batch in parquet.ParquetFile(file).iter_batches(batch_size=64):
                    yield from batch.to_pylist()

    def _sample_rows(self,repo,limit,config='default',shards=None):
        if shards:
            # Pin samples to the same immutable Parquet revision as full imports.
            from itertools import islice
            yield from islice(self._range_rows(shards),limit)
            return
        offset=0
        while offset<limit:
            if self.cancel.is_set():raise InterruptedError()
            try:
                page=get_json('https://datasets-server.huggingface.co/rows?dataset='+quote(repo,safe='')+'&config='+quote(config,safe='')+f'&split=train&offset={offset}&length={min(100,limit-offset)}')
                items=page.get('rows',[])
                if any(item.get('truncated_cells') for item in items):
                    raise ValueError('Dataset viewer truncated a sample record.')
            except (HTTPError,ValueError):
                if not shards:raise
                # The public viewer can be unavailable or truncate long articles.
                # Read the same pinned Parquet source through bounded byte ranges.
                for number,row in enumerate(self._range_rows(shards)):
                    if number<offset:continue
                    if number>=limit:return
                    yield row
                return
            if not items:return
            for item in items:yield item['row']
            offset+=len(items)
