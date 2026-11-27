import json
import threading
from pathlib import Path
import numpy as np
import pytest
from fastapi.testclient import TestClient
from backend.importers.hub import HubImports, normalize
from backend.ml.corpus import FileCorpus, PairPool, vocabulary_text
from backend.ml.dataset import DatasetManager, WeightedMixer
from backend.ml.tokenizer import Tokenizer
from backend.main import create_app


def test_adapters_preserve_preference_and_conversation_structure():
    row={'chosen':'\n\nHuman: hello\n\nAssistant: good','rejected':'\n\nHuman: hello\n\nAssistant: bad'}
    result=normalize('hh-rlhf',row)
    assert result['chosen']==result['completion']=='good'
    assert result['rejected']=='bad'
    assert result['prompt']=='Human: hello'
    parents={'root':{'role':'prompter','lang':'en','text':'Hello','parent_id':None}}
    answer={'role':'assistant','lang':'en','text':'Hi','parent_id':'root'}
    assert normalize('oasst1',answer,parent_lookup=parents.get)=={'prompt':'User: Hello','completion':'Hi'}
    assert normalize('oasst1',{**answer,'lang':'de'},parent_lookup=parents.get) is None
    assert normalize('oasst1',{**answer,'parent_id':'missing'},parent_lookup=parents.get) is None
    robot=normalize('no-robots',{'messages':[{'role':'user','content':'Hi'},{'role':'assistant','content':'Hello'},{'role':'user','content':'Next'},{'role':'assistant','content':'Done'}]})
    assert robot['prompt']=='User: Hi\n\nAssistant: Hello\n\nUser: Next'
    assert robot['completion']=='Done'
    assert normalize('goat',{'instruction':'1+1','input':'1+1','output':'2'})=={'prompt':'1+1','completion':'2'}


def test_translation_gate_excludes_untranslated_chinese():
    row={'input':'你好','content':'答案','reasoning_content':'思考'}
    with pytest.raises(ValueError,match='requires translation'):
        normalize('chinese-r1-en',row)
    with pytest.raises(ValueError,match='retained Chinese'):
        normalize('chinese-r1-en',row,translate=lambda t:t)
    result=normalize('chinese-r1-en',row,translate=lambda t:{'你好':'Hello','答案':'Answer','思考':'Reasoning'}[t])
    assert result['prompt']=='Hello'
    assert result['completion']=='<think>\nReasoning\n</think>\n\nAnswer'
    assert result['reasoning']=='Reasoning'


def test_disk_backed_corpus_and_pair_sampling(tmp_path):
    path=tmp_path/'mixed.jsonl'
    rows=[{'prompt':'hello','completion':'world','chosen':'world','rejected':'noise'} for _ in range(30)]
    path.write_text('\n'.join(json.dumps(r) for r in rows)+'\n')
    corpus=FileCorpus(path)
    tokenizer=Tokenizer(text=vocabulary_text([(corpus,100)]))
    ids=corpus.tokens(tokenizer)
    assert isinstance(ids,np.memmap)
    assert ids.tolist()==tokenizer.encode(''.join(corpus.chunks()))
    assert len(corpus.preview(12))==12
    pool=PairPool([], [corpus], 'dpo')
    assert len(pool)==30
    assert pool[7]['response_a']=='world'
    assert pool[29]['winner']=='a'
    sft=PairPool([{'prompt':'a','corrected_output':'b'}], [corpus], 'sft')
    assert len(sft)==31
    assert sft[0]['corrected_output']=='b'
    assert sft[30]['corrected_output']=='world'
    x,y=WeightedMixer([(corpus,100)],tokenizer,16,'cpu').batch(3)
    assert x.shape==y.shape==(3,16)
    with pytest.raises(IndexError):
        pool[30]


def test_partial_import_is_hidden_and_resume_drops_uncommitted_bytes(tmp_path,monkeypatch):
    manager=HubImports(DatasetManager(tmp_path/'data'))
    def metadata(url):
        if '/parquet?' in url:
            return {'partial':False,'parquet_files':[{'config':'default','split':'train','url':'https://example.test/source.parquet','size':100}]}
        return {'sha':'revision123','cardData':{'license':'apache-2.0'}}
    monkeypatch.setattr('backend.importers.hub.get_json',metadata)
    monkeypatch.setattr(manager,'_download',lambda key,url,path,expected=None:Path(path))
    count=0
    def rows(files):
        for i in range(205):
            yield {'instruction':str(i),'input':'','output':'answer'}
            if i==124:
                manager.cancel.set()
    monkeypatch.setattr(manager,'_rows',rows)
    manager.start('goat');manager.thread.join(10)
    assert manager.jobs['goat']['state']=='paused'
    assert manager.jobs['goat']['checkpoint']['seen']==100
    assert not any(d['name']=='goat.jsonl' for d in manager.datasets.list())
    monkeypatch.setattr(manager,'_rows',lambda files:iter([{'instruction':str(i),'input':'','output':'answer'} for i in range(205)]))
    manager.start('goat');manager.thread.join(10)
    assert manager.jobs['goat']['state']=='completed'
    imported=[json.loads(line) for line in (tmp_path/'data/goat.jsonl').read_text().splitlines()]
    assert len(imported)==205
    assert [r['source_row'] for r in imported]==list(range(205))
    assert manager.jobs['goat']['rows']==205


def test_import_api_catalog_and_validation(tmp_path):
    app=create_app(tmp_path)
    with TestClient(app) as client:
        catalog = client.get('/api/imports').json()['catalog']
        assert sum(not source.get('stream_only') for source in catalog) == 26
        assert client.post('/api/imports/start',json={'dataset_id':'arbitrary/repo'}).status_code==400
        assert client.post('/api/imports/start',json={'dataset_id':'goat','max_records':-1}).status_code==422
        assert client.post('/api/imports/pause').status_code==200


def test_sft_and_dpo_can_train_directly_on_imported_pairs(tmp_path):
    from backend.ml.trainer import Trainer
    from backend.storage import Store
    manager=DatasetManager(tmp_path/'data')
    rows=[{'prompt':'hi','completion':'hello','chosen':'hello','rejected':'noise'} for _ in range(30)]
    (manager.directory/'external.jsonl').write_text('\n'.join(json.dumps(row) for row in rows))
    store=Store(tmp_path/'test.sqlite3')
    trainer=Trainer(manager,store,tmp_path/'checkpoints','cpu')
    sources=[{'name':'external.jsonl','weight':100}]
    try:
        trainer.create({'model':{'n_layer':1,'n_head':2,'n_embd':32,'block_size':32,'dropout':0},'datasets':sources})
        for mode in ['sft','dpo']:
            trainer.start({'mode':mode,'datasets':sources,'max_steps':2,'batch_size':1,'base_mix':0,'eval_interval':2})
            trainer.thread.join(20)
            assert trainer.state=='completed',trainer.error
        assert trainer.step==4
        assert not store.feedback()
    finally:
        trainer.shutdown();store.close()


def test_download_resumes_at_exact_byte_offset(tmp_path,monkeypatch):
    import io
    manager=HubImports(DatasetManager(tmp_path/'data'))
    target=manager.root/'download.bin'
    target.with_suffix('.bin.partial').write_bytes(b'abc')
    class Response(io.BytesIO):
        status=206
        headers={'Content-Range':'bytes 3-5/6'}
    requests=[]
    def response(request,timeout):
        requests.append(request)
        return Response(b'def')
    monkeypatch.setattr('backend.importers.hub.urlopen',response)
    manager._download('goat','https://example.test/data',target,6)
    assert target.read_bytes()==b'abcdef'
    assert requests[0].headers['Range']=='bytes=3-'
    assert not target.with_suffix('.bin.partial').exists()
