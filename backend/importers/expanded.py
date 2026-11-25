"""Explicit adapters for text, instruction, dialogue, and chess sources."""
import re

EXTRA_CATALOG = [
    {'id':'cosmopedia-v2','repo':'HuggingFaceTB/smollm-corpus','config':'cosmopedia-v2','label':'Cosmopedia v2','kind':'text','size':'122.4 GB compressed','license':'ODC-BY','massive':True,'note':'Cosmopedia v2 only. Full corpus and token caches require substantially more disk space.'},
    {'id':'wikipedia-en','repo':'wikimedia/wikipedia','config':'20231101.en','label':'Wikipedia · English','kind':'text','size':'11.6 GB compressed','license':'CC BY-SA 3.0 / GFDL','massive':True,'note':'Pinned English 2023-11-01 snapshot; article URLs and titles retained.'},
    {'id':'magicoder','repo':'ise-uiuc/Magicoder-OSS-Instruct-75K','label':'Magicoder OSS Instruct 75K','kind':'sft','size':'88 MB compressed','license':'MIT'},
    {'id':'dolly','repo':'databricks/databricks-dolly-15k','label':'Dolly 15K','kind':'sft','size':'8 MB compressed','license':'CC BY-SA 3.0'},
    {'id':'daily-dialog','repo':'myname426/li2017dailydialog-bucket','label':'DailyDialog bucket','kind':'sft','size':'Access required / verify repository','license':'See source card','needs_access':True,'note':'The supplied repository returns HTTP 401. Confirm its name or set HF_TOKEN for authorized access.'},
    {'id':'persona-chat','repo':'awsaf49/persona-chat','label':'Persona Chat','kind':'sft','size':'Canonical train_self_original.txt','license':'MIT','raw':'persona','note':'Imports the self-persona/original training variant once; excludes duplicated variants and test data.'},
    {'id':'cornell-movie','repo':'cornell-movie-dialog/cornell_movie_dialog','label':'Cornell Movie Dialogs','kind':'sft','size':'~10 MB ZIP','license':'See original Cornell terms','raw':'cornell','note':'Reads actual utterance text from the original archive; no dataset scripts are executed.'},
    {'id':'altered-riddles','repo':'marcodsn/altered-riddles','label':'Altered Riddles','kind':'sft','size':'~262 KB compressed','license':'Apache 2.0','note':'Original and altered riddles become separate examples. Training on this benchmark makes it unsuitable as an independent evaluation set.'},
    {'id':'philosopher-quotes','repo':'datastax/philosopher-quotes','label':'Philosopher Quotes','kind':'text','size':'~41 KB compressed','license':'CC · see source card'},
    {'id':'science-facts','repo':'Royal-lobster/10001-Science-Facts','label':'10,001 Science Facts','kind':'text','size':'~821 KB compressed','license':'MIT','note':'Source URLs and category metadata are preserved; source claims are not independently fact-checked.'},
    {'id':'chess-puzzles','repo':'Lichess/chess-puzzles','label':'Lichess Chess Puzzles','kind':'sft','size':'~885 MB compressed','license':'CC0 1.0','note':'Applies the opponent setup move before presenting the FEN; target is the remaining legal UCI solution.'},
    {'id':'smoltalk','repo':'HuggingFaceTB/smoltalk','config':'all','label':'SmolTalk','kind':'sft','size':'2.01 GB compressed','license':'Mixed · see subset terms','note':'Uses all/train once, not its duplicated component configurations. Preserves each assistant turn and its preceding conversation.'},
    {'id':'tulu-3-sft-mixture','repo':'allenai/tulu-3-sft-mixture','label':'Tulu 3 SFT Mixture','kind':'sft','size':'1.41 GB compressed','license':'ODC-BY · subset terms apply','note':'Includes multilingual data and persona subsets also listed separately. Avoid double-weighting overlapping sources. Some subsets are noncommercial.'},
    {'id':'tulu-3-sft-personas-math-grade','repo':'allenai/tulu-3-sft-personas-math-grade','label':'Tulu 3 Personas · Grade Math','kind':'sft','size':'52 MB compressed','license':'See source card','note':'Exact requested unfiltered repository. Overlaps the Tulu mixture; select weights deliberately.'},
    {'id':'tulu-3-sft-personas-instruction-following','repo':'allenai/tulu-3-sft-personas-instruction-following','label':'Tulu 3 Personas · Instruction Following','kind':'sft','size':'39 MB compressed','license':'ODC-BY','note':'Instruction constraints are retained as metadata. Overlaps the Tulu mixture.'},
    {'id':'tulu-3-sft-personas-algebra','repo':'allenai/tulu-3-sft-personas-algebra','label':'Tulu 3 Personas · Algebra','kind':'sft','size':'36 MB compressed','license':'See source card','note':'Uses the exact requested sft-personas-algebra repository; no similarly named dataset is substituted.'},
    {'id':'codealpaca-20k','repo':'HuggingFaceH4/CodeAlpaca_20K','label':'CodeAlpaca 20K','kind':'sft','size':'3 MB compressed','license':'CC · see source card','note':'Imports the H4 training split only. Preserves code whitespace; never executes downloaded code.'},
    {'id':'fineweb-edu-10bt','repo':'HuggingFaceFW/fineweb-edu','config':'sample-10BT','label':'FineWeb-Edu · 10BT','kind':'text','size':'28.5 GB compressed','license':'ODC-BY','massive':True,'note':'Only sample/10BT (sample-10BT/train), not the full FineWeb-Edu corpus. Text and original document metadata are preserved.'},
    {'id':'cosmopedia-100m','repo':'krisbailey/cosmopedia-100M','label':'Cosmopedia · 100M','kind':'text','size':'350 MB compressed','license':'ODC-BY','sample_download_shard':True,'note':'Exact 100M-token repository, separate from Cosmopedia v2. Samples also download its 350 MB Parquet shard to disk because of its large row group; only the requested rows are normalized.'},
    {'id':'finemath-4plus','repo':'HuggingFaceTB/finemath','config':'finemath-4plus','label':'FineMath · 4+','kind':'text','size':'18.4 GB compressed','license':'ODC-BY','massive':True,'note':'Only finemath-4plus/train from main; excludes FineMath-3+ and InfiWebMath variants. Original text, quality scores and document metadata are preserved.'},
    {'id':'tinystories-original','repo':'roneneldan/TinyStories','label':'TinyStories · Original','kind':'text','size':'991 MB compressed','license':'CDLA-Sharing 1.0','note':'Original default/train, separate from TinyStories Instruct. Excludes validation and GPT-4-only alternative files.'},
]
EXTRA_IDS = {item['id'] for item in EXTRA_CATALOG}
PRETRAINING_IDS = {'fineweb-edu-10bt','cosmopedia-100m','finemath-4plus','tinystories-original'}
CHAT_IDS = {'smoltalk','tulu-3-sft-mixture','tulu-3-sft-personas-math-grade',
            'tulu-3-sft-personas-instruction-following','tulu-3-sft-personas-algebra'}


def pair(prompt, completion, **metadata):
    if not isinstance(prompt,str) or not prompt.strip() or not isinstance(completion,str) or not completion.strip():
        raise ValueError('Expected a nonempty prompt and completion.')
    return {'prompt':prompt,'completion':completion,**metadata}


def dialogue_pairs(turns, persona=None):
    if not isinstance(turns,list) or any(not isinstance(turn,str) for turn in turns):
        raise ValueError('Dialogue must contain a list of utterance strings.')
    history=[]
    records=[]
    for i,turn in enumerate(turns):
        if not turn.strip():
            raise ValueError('Dialogue contains an empty utterance.')
        # Alternating speakers: learn the assistant turns with the entire earlier
        # context. Never turn distractor/candidate responses into positive targets.
        if i % 2 == 1:
            prefix=('Persona:\n'+'\n'.join(persona)+'\n\n') if persona else ''
            records.append(pair(prefix+'\n\n'.join(history),turn,source_turn=i))
        history.append(('User' if i % 2 == 0 else 'Assistant')+': '+turn)
    return records


def conversation_pairs(messages,**metadata):
    """Train each observed assistant turn using only its preceding role-tagged text."""
    roles={'system':'System','developer':'Developer','user':'User','assistant':'Assistant','tool':'Tool','function':'Function'}
    if not isinstance(messages,list) or not messages:
        raise ValueError('Expected a nonempty messages list.')
    # Validate the whole row before emitting any examples into the import stream.
    for message in messages:
        if (not isinstance(message,dict) or not isinstance(message.get('role'),str)
            or message['role'] not in roles or not isinstance(message.get('content'),str)
            or not message['content'].strip()):
            raise ValueError('Messages need a recognized role and nonempty text content.')
    history=[];records=[];has_user=False
    for index,message in enumerate(messages):
        role,content=message['role'],message['content']
        if role=='user':has_user=True
        if role=='assistant':
            if not has_user:raise ValueError('Assistant response has no preceding user instruction.')
            records.append(pair('\n\n'.join(history),content,source_turn=index,**metadata))
        history.append(roles[role]+': '+content)
    if not records:raise ValueError('Conversation has no assistant response.')
    return records


def normalize_extra(key,row):
    if key in PRETRAINING_IDS:
        if not isinstance(row.get('text'),str) or not row['text'].strip():
            raise ValueError('Pretraining record needs nonempty text.')
        # Keep source fields without interpreting metadata prompts/answers as SFT
        # or overwriting the importer's own source/source_row provenance.
        record={'text':row['text']}
        metadata={name:value for name,value in row.items() if name!='text'}
        if metadata:record['source_metadata']=metadata
        return record
    if key in CHAT_IDS:
        metadata={}
        for source,target in (('id','example_id'),('source','source_subset'),('constraints','constraints'),('prompt','original_prompt')):
            if source in row:metadata[target]=row[source]
        return conversation_pairs(row['messages'],**metadata)
    if key=='codealpaca-20k':
        return pair(row['prompt'],row['completion'])
    if key=='cosmopedia-v2':
        if not isinstance(row.get('text'),str) or not row['text'].strip():
            raise ValueError('Cosmopedia row has no text.')
        return {'text':row['text'],'original_prompt':row.get('prompt'),'audience':row.get('audience'),'format':row.get('format')}
    if key=='wikipedia-en':
        return {'text':row['title']+'\n\n'+row['text'],'title':row['title'],'article_id':row.get('id'),'url':row.get('url')}
    if key=='magicoder':
        return pair(row['problem'],row['solution'],language=row.get('lang'))
    if key=='dolly':
        context=row.get('context') or ''
        return pair(row['instruction']+('\n\nContext:\n'+context if context else ''),row['response'],category=row.get('category'))
    if key=='daily-dialog':
        turns=row.get('dialog',row.get('dialogue',row.get('utterances')))
        if isinstance(turns,str):
            turns=[part.strip() for part in re.split(r'__eou__|\n',turns) if part.strip()]
        return dialogue_pairs(turns)
    if key in ('persona-chat','cornell-movie'):
        return dialogue_pairs(row['turns'],row.get('persona'))
    if key=='altered-riddles':
        result=[]
        for variant in ('original','altered'):
            reasoning=row.get(variant+'_reasoning') or ''
            result.append(pair(row[variant+'_riddle'],'Answer: '+row[variant+'_answer']+('\nReasoning: '+reasoning if reasoning else ''),variant=variant,riddle_id=row.get('id'),accepted_answers=row.get(variant+'_accepted_answers')))
        return result
    if key=='philosopher-quotes':
        return {'text':row['quote']+'\n— '+row['author'],'author':row['author'],'tags':row.get('tags')}
    if key=='science-facts':
        return {'text':row['text'],'url':row.get('source_url'),'category':row.get('category'),'original_source':row.get('source'),'source_file':row.get('source_file')}
    if key=='chess-puzzles':
        import chess
        board=chess.Board(row['FEN'])
        moves=row['Moves'].split()
        if len(moves)<2:
            raise ValueError('Puzzle needs a setup move and a solution.')
        board.push_uci(moves[0])
        position=board.fen()
        for move in moves[1:]:
            board.push_uci(move)  # Validates legality before exposing a target.
        return pair('Solve this chess puzzle. Return the solution line in UCI notation.\nFEN: '+position,' '.join(moves[1:]),puzzle_id=row.get('PuzzleId'),rating=row.get('Rating'),themes=row.get('Themes'),setup_move=moves[0],original_fen=row['FEN'])
    raise ValueError('Unknown expanded dataset adapter.')
