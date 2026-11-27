import io
import json
from pathlib import Path
from urllib.error import HTTPError
import zipfile
import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient
from backend.importers.hub import HubImports,normalize,authenticated_request,CATALOG
from backend.importers.raw_sources import persona_rows,cornell_rows
from backend.ml.dataset import DatasetManager
from backend.ml.corpus import FileCorpus
from backend.ml.tokenizer import Tokenizer
from backend.ml.trainer import Trainer
from backend.storage import Store
from backend.main import create_app

SMALL={'n_layer':1,'n_head':2,'n_embd':32,'block_size':32,'dropout':0}


def test_catalog_selects_exact_requested_configs():
    sources={s['id']:s for s in CATALOG}
    assert sum(not source.get('stream_only') for source in sources.values()) == 26
    assert sources['cosmopedia-v2']['config']=='cosmopedia-v2'
    assert sources['wikipedia-en']['config']=='20231101.en'
    assert sources['daily-dialog']['needs_access']
    assert sources['persona-chat']['raw']=='persona'
    assert sources['cornell-movie']['raw']=='cornell'


def test_expanded_adapters():
    assert normalize('cosmopedia-v2',{'text':'textbook','prompt':'seed'})['text']=='textbook'
    assert normalize('wikipedia-en',{'title':'Title','text':'Article','id':'1','url':'https://en.wikipedia.org/wiki/Title'})['text']=='Title\n\nArticle'
    assert normalize('magicoder',{'problem':'Code task','solution':'def answer():\n    return 1','lang':'py'})['completion'].startswith('def ')
    assert normalize('dolly',{'instruction':'Question','context':'Passage','response':'Answer'})['prompt']=='Question\n\nContext:\nPassage'
    assert normalize('science-facts',{'text':'Fact','source_url':'https://source','category':'space'})['url']=='https://source'
    assert normalize('philosopher-quotes',{'quote':'Know thyself.','author':'Author'})['text']=='Know thyself.\n— Author'
    pairs=normalize('daily-dialog',{'dialog':['Hello','Hi','How are you?','Fine']})
    assert len(pairs)==2
    assert pairs[1]['prompt']=='User: Hello\n\nAssistant: Hi\n\nUser: How are you?'
    assert pairs[1]['completion']=='Fine'
    riddles=normalize('altered-riddles',{'original_riddle':'A?','original_answer':'A','altered_riddle':'B?','altered_answer':'B','altered_reasoning':'Because B.'})
    assert len(riddles)==2
    assert riddles[0]['completion']=='Answer: A'
    assert riddles[1]['completion']=='Answer: B\nReasoning: Because B.'


def test_lichess_setup_move_and_legal_solution():
    import chess
    fen='r6k/pp2r2p/4Rp1Q/3p4/8/1N1P2R1/PqP2bPP/7K b - - 0 24'
    moves='f2g3 e6e7 b2b1 b3c1 b1c1 h6c1'
    result=normalize('chess-puzzles',{'FEN':fen,'Moves':moves,'PuzzleId':'00008','Rating':1811})
    board=chess.Board(fen);board.push_uci('f2g3')
    assert result['prompt'].endswith(board.fen())
    assert result['completion']=='e6e7 b2b1 b3c1 b1c1 h6c1'
    with pytest.raises(ValueError):
        normalize('chess-puzzles',{'FEN':fen,'Moves':'a1a8 e6e7'})


def test_original_persona_and_cornell_parsers(tmp_path):
    persona=tmp_path/'persona.txt'
    persona.write_text('1 your persona: I like books.\n1 hi\thello\twrong candidate|another wrong\n2 question\tanswer\tdistractor\n1 your persona: I like chess.\n1 hello\thi\n')
    rows=list(persona_rows(persona))
    assert len(rows)==2
    assert rows[0]['turns']==['hi','hello','question','answer']
    result=normalize('persona-chat',rows[0])
    assert result[0]['prompt'].startswith('Persona:\nI like books.')
    assert 'wrong candidate' not in json.dumps(result)
    path=tmp_path/'cornell.zip'
    with zipfile.ZipFile(path,'w') as archive:
        archive.writestr('cornell movie-dialogs corpus/movie_lines.txt','L1 +++$+++ u1 +++$+++ m1 +++$+++ A +++$+++ Hello\nL2 +++$+++ u2 +++$+++ m1 +++$+++ B +++$+++ Hi\n')
        archive.writestr('cornell movie-dialogs corpus/movie_conversations.txt',"u1 +++$+++ u2 +++$+++ m1 +++$+++ ['L1', 'L2']\n")
    movie=list(cornell_rows(path,tmp_path/'parents.sqlite3',lambda:False))
    assert movie[0]['turns']==['Hello','Hi']
    assert normalize('cornell-movie',movie[0])[0]['completion']=='Hi'


def test_magic8k_is_exact_and_checkpoint_self_contained():
    from tokenizers import Tokenizer as RustTokenizer
    tokenizer=Tokenizer('magic8k')
    reference=RustTokenizer.from_str(tokenizer.state()['tokenizer_json'])
    text='def example(x):\n    return x + 1\nCafé 中文 😀 <|end|>'
    assert tokenizer.vocab_size==8192
    assert tokenizer.encode(text)==reference.encode(text,add_special_tokens=False).ids
    assert tokenizer.decode(tokenizer.encode(text))==text
    restored=Tokenizer(state=tokenizer.state())
    assert restored.encode(text)==tokenizer.encode(text)
    assert max(tokenizer.encode(text))<8192
    long_text='Preserve all tokens beyond a BERT-style length limit. '*500
    assert len(tokenizer.encode(long_text))>512
    assert tokenizer.decode(tokenizer.encode(long_text))==long_text
    with pytest.raises(ValueError):Tokenizer('unknown')


def test_uint16_cache_and_uint32_fallback(tmp_path):
    path=tmp_path/'text.txt';path.write_text('Hello world! '*100)
    corpus=FileCorpus(path)
    tokenizer=Tokenizer('magic8k')
    ids=corpus.tokens(tokenizer)
    assert ids.dtype==np.uint16
    assert ids.nbytes==len(ids)*2
    assert ids.tolist()==tokenizer.encode(path.read_text())
    class Wide:
        vocab_size=70000
        def state(self):return {'kind':'wide-test'}
        def encode(self,text):return [69999]*len(text)
    wide=corpus.tokens(Wide())
    assert wide.dtype==np.uint32
    assert int(wide[0])==69999


def make_trainer(root,tokenizer='magic8k'):
    store=Store(root/'metadata.sqlite3')
    trainer=Trainer(DatasetManager(root/'data'),store,root/'checkpoints','cpu')
    trainer.create({'model':SMALL,'tokenizer':tokenizer,'datasets':[{'name':'starter-sample.txt','weight':100}]})
    return trainer,store


def test_portable_magic_checkpoint_import_and_legacy_reload(tmp_path,monkeypatch):
    monkeypatch.setenv('NANOFORGE_DEVICE','cpu')
    source=tmp_path/'source';source.mkdir()
    trainer,store=make_trainer(source)
    try:
        trainer.start({'mode':'pretrain','max_steps':2,'batch_size':1,'eval_interval':2,'precision':'auto'})
        trainer.thread.join(20)
        assert trainer.state=='completed',trainer.error
        assert trainer.precision=='float32'
        payload=torch.load(trainer.checkpoint_path(trainer.checkpoint),weights_only=True,map_location='cpu')
        assert all(t.device.type=='cpu' and t.dtype==torch.float32 for t in payload['model_state_dict'].values())
        assert payload['model_state_dict']['token_embedding.weight'].data_ptr()==payload['model_state_dict']['lm_head.weight'].data_ptr()
        assert isinstance(payload['torch_version'],str)
        payload['optimizer_state_dict']['param_groups'][0]['fused']=True
        payload['training_config']['precision']='float16'
        payload['precision']='float16'
        app=create_app(tmp_path/'studio')
        with TestClient(app) as client:
            buffer=io.BytesIO();torch.save(payload,buffer)
            result=client.post('/api/checkpoints/upload',files={'file':('colab.pt',buffer.getvalue(),'application/octet-stream')})
            assert result.status_code==200,result.text
            assert result.json()['parameters']==trainer.status()['parameters']
            assert client.post('/api/checkpoints/select',json={'name':'colab.pt'}).status_code==200
            assert app.state.trainer.tokenizer.kind=='magic8k'
            assert app.state.trainer.precision=='float32'
            assert app.state.trainer.optimizer.param_groups[0]['fused'] is False
            generated=client.post('/api/generate',json={'prompt':'First Citizen:','max_new_tokens':2})
            assert generated.status_code==200,generated.text
            assert len(generated.json()['steps'])==2
            resumed=client.post('/api/training/start',json={'max_steps':1,'batch_size':1,'eval_interval':1,'precision':'auto'})
            assert resumed.status_code==200,resumed.text
            app.state.trainer.thread.join(20)
            assert app.state.trainer.state=='completed',app.state.trainer.error
            assert app.state.trainer.step==3
            assert client.post('/api/checkpoints/upload',files={'file':('colab.pt',buffer.getvalue())}).status_code==400
            for key in ['precision','scaler_state_dict','torch_version','checkpoint_format_version']:payload.pop(key,None)
            legacy=io.BytesIO();torch.save(payload,legacy)
            assert client.post('/api/checkpoints/upload',files={'file':('legacy.pt',legacy.getvalue())}).status_code==200
            assert client.post('/api/checkpoints/select',json={'name':'legacy.pt'}).status_code==200
            assert client.post('/api/checkpoints/upload',files={'file':('bad.pt',b'not a checkpoint')}).status_code==400
        with pytest.raises(ValueError,match='FP16 training requires CUDA'):
            trainer.start({'precision':'float16','max_steps':1})
    finally:trainer.shutdown();store.close()


def test_hf_credentials_are_not_sent_to_other_hosts(monkeypatch):
    monkeypatch.setenv('HF_TOKEN','hf_test_token')
    request=authenticated_request('https://huggingface.co/api/datasets/private/data')
    assert request.get_header('Authorization')=='Bearer hf_test_token'
    assert 'Authorization' in request.unredirected_hdrs
    assert authenticated_request('https://www.cs.cornell.edu/corpus.zip').get_header('Authorization') is None


def test_private_or_missing_dataset_is_explicitly_blocked(tmp_path,monkeypatch):
    manager=HubImports(DatasetManager(tmp_path/'data'))
    def denied(url):raise HTTPError(url,401,'Unauthorized',{},None)
    monkeypatch.setattr('backend.importers.hub.get_json',denied)
    manager.start('daily-dialog');manager.thread.join(5)
    assert manager.jobs['daily-dialog']['state']=='blocked'
    assert 'HF_TOKEN' in manager.jobs['daily-dialog']['error']
    assert not (tmp_path/'data/daily-dialog.jsonl').exists()


def test_streamed_shards_commit_before_deleting_and_resume(tmp_path,monkeypatch):
    import pyarrow as pa
    import pyarrow.parquet as pq
    manager=HubImports(DatasetManager(tmp_path/'data'))
    shards=[{'config':'default','split':'train','url':f'https://example.test/{i}','size':123} for i in range(2)]
    def metadata(url):
        return {'partial':False,'parquet_files':shards} if '/parquet?' in url else {'sha':'revision123','cardData':{}}
    monkeypatch.setattr('backend.importers.hub.get_json',metadata)
    downloads=[];interrupted=[False]
    def download(key,url,path,expected=None):
        index=int(url.rsplit('/',1)[-1]);downloads.append(index)
        if index==1 and not interrupted[0]:
            interrupted[0]=True;raise InterruptedError()
        path.parent.mkdir(parents=True,exist_ok=True)
        pq.write_table(pa.Table.from_pylist([{'instruction':str(index*3+i),'input':'','output':'answer'} for i in range(3)]),path)
        return path
    monkeypatch.setattr(manager,'_download',download)
    manager.start('goat');manager.thread.join(10)
    assert manager.jobs['goat']['state']=='paused'
    assert manager.jobs['goat']['checkpoint']['seen']==3
    assert not (manager.root/'goat/shard-0.parquet').exists()
    manager.start('goat');manager.thread.join(10)
    assert manager.jobs['goat']['state']=='completed'
    assert downloads.count(0)==1
    records=[json.loads(line) for line in (manager.datasets.directory/'goat.jsonl').read_text().splitlines()]
    assert [r['source_row'] for r in records]==list(range(6))


@pytest.mark.skipif(not torch.cuda.is_available(),reason='Requires an actual CUDA GPU; run this test in the Colab GPU notebook.')
@pytest.mark.parametrize('allow_tf32',[False,True])
def test_cuda_fp16_training_and_cpu_studio_export(tmp_path,allow_tf32):
    store=Store(tmp_path/'metadata.sqlite3')
    trainer=Trainer(DatasetManager(tmp_path/'data'),store,tmp_path/'checkpoints','cuda')
    try:
        trainer.create({'model':SMALL,'tokenizer':'magic8k','precision':'float16','allow_tf32':allow_tf32,'datasets':[{'name':'starter-sample.txt','weight':100}]})
        trainer.start({'precision':'float16','allow_tf32':allow_tf32,'max_steps':2,'batch_size':2,'eval_interval':2})
        trainer.thread.join(60)
        assert trainer.state=='completed',trainer.error
        assert trainer.amp_enabled and trainer.precision=='float16'
        assert trainer.allow_tf32==(allow_tf32 and torch.cuda.get_device_capability()[0]>=8)
        assert torch.backends.cuda.matmul.allow_tf32==trainer.allow_tf32
        payload=torch.load(trainer.checkpoint_path(trainer.checkpoint),map_location='cpu',weights_only=True)
        assert payload['scaler_state_dict']
        assert all(t.device.type=='cpu' and t.dtype==torch.float32 for t in payload['model_state_dict'].values())
        from backend.checkpoints import register_checkpoint
        registered=register_checkpoint(trainer.checkpoint_path(trainer.checkpoint),tmp_path/'studio-checkpoints')
        assert registered['step']==2
        local=Trainer(DatasetManager(tmp_path/'local-data'),store,tmp_path/'studio-checkpoints','cpu')
        try:
            local.load(registered['name'])
            assert local.precision=='float32' and not local.allow_tf32
            assert len(local.generate(prompt='Hello',max_new_tokens=2,telemetry=False)['tokens'])==2
            local.start({'precision':'auto','max_steps':1,'batch_size':1,'eval_interval':1})
            local.thread.join(60)
            assert local.state=='completed',local.error
            assert local.step==3
        finally:local.shutdown()
    finally:trainer.shutdown();store.close()


def test_range_reader_bounds_and_cross_block_seeks():
    from backend.importers.ranges import RangeReader
    data=bytes(range(100))
    requests=[]
    def fetch(start,end):
        requests.append((start,end));return data[start:end+1]
    with RangeReader(len(data),fetch,block_size=16) as file:
        file.seek(12)
        assert file.read(20)==data[12:32]
        file.seek(-7,2)
        assert file.read()==data[-7:]
        assert len(file.blocks)<=2
        file.seek(0)
        assert file.read(5)==data[:5]
    assert all(end-start<16 for start,end in requests)


def test_finalization_can_resume_after_output_rename(tmp_path,monkeypatch):
    manager=HubImports(DatasetManager(tmp_path/'data'))
    content=b'{"prompt":"Question","completion":"Answer"}\n'
    (manager.datasets.directory/'dolly.jsonl').write_bytes(content)
    manager.jobs['dolly']={'state':'paused','repo':'databricks/databricks-dolly-15k','max_records':0,'finalize_pending':True,'rows':1,'checkpoint':{'bytes':len(content)}}
    def unexpected(url):raise AssertionError('Finalization recovery must not use the network.')
    monkeypatch.setattr('backend.importers.hub.get_json',unexpected)
    manager.start('dolly');manager.thread.join(5)
    assert manager.jobs['dolly']['state']=='completed'
    assert manager.jobs['dolly']['rows']==1
    assert not manager.jobs['dolly']['finalize_pending']
