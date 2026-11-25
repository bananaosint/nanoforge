"""Keep large pretraining subsets exact and source metadata out of SFT targets."""
import json
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from backend.importers.hub import BY_ID, HubImports, normalize
from backend.ml.corpus import FileCorpus, PairPool
from backend.ml.dataset import DatasetManager, WeightedMixer
from backend.ml.tokenizer import Tokenizer

SOURCES={
    'fineweb-edu-10bt':('HuggingFaceFW/fineweb-edu','sample-10BT'),
    'cosmopedia-100m':('krisbailey/cosmopedia-100M','default'),
    'finemath-4plus':('HuggingFaceTB/finemath','finemath-4plus'),
    'tinystories-original':('roneneldan/TinyStories','default'),
}


def test_exact_catalog_subsets_and_existing_tinystories_instruct():
    for key,(repo,config) in SOURCES.items():
        assert BY_ID[key]['repo']==repo and BY_ID[key].get('config','default')==config
        assert BY_ID[key]['kind']=='text'
    assert BY_ID['fineweb-edu-10bt']['massive'] and BY_ID['finemath-4plus']['massive']
    assert BY_ID['tinystories']['repo']=='roneneldan/TinyStoriesInstruct'


@pytest.mark.parametrize('key',SOURCES)
def test_text_and_metadata_preserved_without_inventing_pairs(key):
    text='A lesson with Café, 中文, and code.\n    x = 1\n'
    row={'text':text,'url':'https://example.test/article','score':4.5,'metadata':'{"math": true}',
         'source':'upstream','source_row':12,'prompt':'metadata prompt','completion':'metadata answer'}
    result=normalize(key,row)
    assert result=={'text':text,'source_metadata':{name:value for name,value in row.items() if name!='text'}}
    for invalid in (None,123,'','  '):
        with pytest.raises(ValueError):normalize(key,{'text':invalid})


@pytest.mark.parametrize('key',SOURCES)
@pytest.mark.parametrize('limit',[0,1])
def test_full_and_sample_import_use_only_selected_train_subset(tmp_path,monkeypatch,key,limit):
    repo,config=SOURCES[key]
    manager=HubImports(DatasetManager(tmp_path/'data'))
    shard={'config':config,'split':'train','url':'https://example.test/resolve/refs%2Fconvert%2Fparquet/train','size':123}
    def metadata(url):
        assert '/rows?' not in url  # Samples must work when the public viewer fails.
        if '/parquet?' in url:
            return {'partial':False,'parquet_files':[
                {**shard,'config':'unwanted','url':'https://example.test/wrong-config'},
                {**shard,'split':'validation','url':'https://example.test/held-out'},shard]}
        return {'sha':'pinned123','cardData':{}}
    monkeypatch.setattr('backend.importers.hub.get_json',metadata)
    row={'text':'An educational story with Café and a mathematical explanation. '*20,
         'url':'https://example.test/article','score':4.5}
    rows=[row,{**row,'text':None}]
    downloaded=[]
    def download(source,url,path,expected):
        downloaded.append(url)
        assert url=='https://example.test/resolve/pinned123/train'
        pq.write_table(pa.Table.from_pylist(rows),path)
        return path
    def ranges(shards):
        assert shards==[{**shard,'url':'https://example.test/resolve/pinned123/train'}]
        yield from rows
    monkeypatch.setattr(manager,'_download',download)
    monkeypatch.setattr(manager,'_range_rows',ranges)
    try:
        manager.start(key,limit);manager.thread.join(15)
        job=manager.jobs[key]
        assert job['state']==('sampled' if limit else 'completed'),job.get('error')
        assert job['config']==config and job['split']=='train'
        disk_sample=bool(limit and BY_ID[key].get('sample_download_shard'))
        assert len(downloaded)==(1 if not limit or disk_sample else 0)
        assert job['source_records']==(1 if limit else 2)
        assert job['rows']==1 and job['sft_pairs']==job['dpo_pairs']==0
        assert job['needs_review']==(0 if limit else 1)
        if disk_sample:
            assert job['sample_method']=='disk-backed shard'
            assert not list((manager.root/key).glob('shard-*.parquet'))
        output=manager.datasets.path(job['output'])
        record=json.loads(output.read_text(encoding='utf-8'))
        assert record['text']==row['text'] and record['source']==repo and record['source_row']==0
        assert record['source_metadata']=={'url':row['url'],'score':4.5}
        corpus=FileCorpus(output);tokenizer=Tokenizer('magic8k')
        assert corpus.tokens(tokenizer).dtype.name=='uint16'
        assert not PairPool([], [corpus],'sft') and not PairPool([], [corpus],'dpo')
        x,y=WeightedMixer([(corpus,100)],tokenizer,32,'cpu').batch(1)
        assert x.shape==y.shape==(1,32)
    finally:manager.shutdown()
