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
]
EXTRA_IDS = {item['id'] for item in EXTRA_CATALOG}


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


def normalize_extra(key,row):
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
