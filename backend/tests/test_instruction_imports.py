"""Requested chat/instruction sources use the same durable Parquet import path."""
import json
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from backend.importers.hub import BY_ID, CATALOG, HubImports, normalize
from backend.ml.corpus import FileCorpus, PairPool
from backend.ml.dataset import DatasetManager
from backend.ml.tokenizer import Tokenizer

SOURCES={
    'smoltalk':('HuggingFaceTB/smoltalk','all'),
    'tulu-3-sft-mixture':('allenai/tulu-3-sft-mixture','default'),
    'tulu-3-sft-personas-math-grade':('allenai/tulu-3-sft-personas-math-grade','default'),
    'tulu-3-sft-personas-instruction-following':('allenai/tulu-3-sft-personas-instruction-following','default'),
    'tulu-3-sft-personas-algebra':('allenai/tulu-3-sft-personas-algebra','default'),
    'codealpaca-20k':('HuggingFaceH4/CodeAlpaca_20K','default'),
}
MESSAGES=[{'role':'system','content':'Keep Café and code whitespace.'},
          {'role':'user','content':'Write a function.'},
          {'role':'assistant','content':'def answer():\n    return 42\n'},
          {'role':'user','content':'Explain the return value.'},
          {'role':'assistant','content':'It returns the integer forty-two.'}]


def test_catalog_exact_repositories_and_smoltalk_configuration():
    assert len({entry['id'] for entry in CATALOG})==len(CATALOG)
    for key,(repo,config) in SOURCES.items():
        assert BY_ID[key]['repo']==repo and BY_ID[key].get('config','default')==config
        assert BY_ID[key]['kind']=='sft'


@pytest.mark.parametrize('key',[key for key in SOURCES if key!='codealpaca-20k'])
def test_assistant_turns_keep_context_and_metadata_without_future_leakage(key):
    result=normalize(key,{'messages':MESSAGES,'id':'row-id','source':'subset','constraints':['Use code.'],'prompt':'Original instruction'})
    assert len(result)==2
    assert result[0]['prompt']=='System: Keep Café and code whitespace.\n\nUser: Write a function.'
    assert result[0]['completion']==MESSAGES[2]['content']
    assert 'Explain' not in result[0]['prompt'] and 'forty-two' not in result[1]['prompt']
    assert 'Assistant: def answer():\n    return 42\n' in result[1]['prompt']
    assert result[1]['prompt'].endswith('User: Explain the return value.')
    assert [row['source_turn'] for row in result]==[2,4]
    assert result[1]['example_id']=='row-id' and result[1]['source_subset']=='subset'
    assert result[1]['constraints']==['Use code.'] and result[1]['original_prompt']=='Original instruction'


@pytest.mark.parametrize('messages',[
    [],None,[{'role':'assistant','content':'No question'}],
    [{'role':'user','content':'Unanswered'}],
    MESSAGES+[{'role':'assistant','content':None}],
    [{'role':'other','content':'Bad role'}],
    [{'role':['user'],'content':'Bad role type'}],
    [{'role':'user','content':['Not text']}],
])
def test_malformed_chat_is_rejected_before_partial_examples(messages):
    with pytest.raises(ValueError):normalize('smoltalk',{'messages':messages})


@pytest.mark.parametrize('key',SOURCES)
def test_full_import_filters_config_and_split_and_produces_trainable_pairs(tmp_path,monkeypatch,key):
    repo,config=SOURCES[key]
    manager=HubImports(DatasetManager(tmp_path/'data'))
    selected={'config':config,'split':'train','url':'https://example.test/resolve/refs%2Fconvert%2Fparquet/train','size':123}
    def metadata(url):
        if '/parquet?' in url:
            return {'partial':False,'parquet_files':[selected,{**selected,'split':'test','url':'https://example.test/test'},
                                                      {**selected,'config':'other-component','url':'https://example.test/duplicate'}]}
        return {'sha':'pinned123','cardData':{}}
    monkeypatch.setattr('backend.importers.hub.get_json',metadata)
    if key=='codealpaca-20k':
        good={'prompt':'Write a function with Café.','completion':MESSAGES[2]['content']}
        bad={**good,'completion':''};expected=1
    else:
        good={'messages':MESSAGES,'id':'row-id','source':'subset'}
        bad={**good,'messages':MESSAGES+[{'role':'assistant','content':None}]};expected=2
    downloads=[]
    def download(source,url,path,expected_size):
        downloads.append(url)
        assert url=='https://example.test/resolve/pinned123/train'
        pq.write_table(pa.Table.from_pylist([good,bad]),path)
        return path
    monkeypatch.setattr(manager,'_download',download)
    try:
        manager.start(key);manager.thread.join(15)
        job=manager.jobs[key]
        assert job['state']=='completed',job.get('error')
        assert job['config']==config and job['split']=='train'
        assert len(downloads)==1 and job['source_records']==2
        assert job['rows']==job['sft_pairs']==expected and job['needs_review']==1 and job['dpo_pairs']==0
        output=manager.datasets.path(job['output'])
        records=[json.loads(line) for line in output.read_text(encoding='utf-8').splitlines()]
        assert all(row['source']==repo and row['source_row']==0 for row in records)
        if key!='codealpaca-20k':assert all(row['source_subset']=='subset' for row in records)
        assert len(PairPool([], [FileCorpus(output)],'sft'))==expected
        assert FileCorpus(output).tokens(Tokenizer('magic8k')).dtype.name=='uint16'
    finally:manager.shutdown()
