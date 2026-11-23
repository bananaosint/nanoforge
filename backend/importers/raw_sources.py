"""Parse original dialogue files without running remote Python dataset scripts."""
import ast
import io
import sqlite3
import zipfile

CORNELL_URL='https://www.cs.cornell.edu/~cristian/data/cornell_movie_dialogs_corpus.zip'


def persona_rows(path):
    persona=[]
    turns=[]
    active=False
    with open(path,encoding='utf-8') as file:
        for line in file:
            number,sep,text=line.rstrip('\r\n').partition(' ')
            if not sep:
                continue
            if number=='1' and active:
                yield {'persona':persona,'turns':turns}
                persona=[];turns=[];active=False
            if text.startswith('your persona: '):
                persona.append(text[len('your persona: '):])
                continue
            fields=text.split('\t')
            if len(fields)<2:
                raise ValueError('PersonaChat dialogue line has no response.')
            turns.extend(fields[:2]);active=True
    if active:
        yield {'persona':persona,'turns':turns}


def cornell_rows(path,database,cancelled):
    db=sqlite3.connect(database)
    try:
        db.execute('CREATE TABLE IF NOT EXISTS utterances(id TEXT PRIMARY KEY,text TEXT)')
        with zipfile.ZipFile(path) as archive:
            names=archive.namelist()
            lines_name=next(name for name in names if name.endswith('/movie_lines.txt'))
            conversations_name=next(name for name in names if name.endswith('/movie_conversations.txt'))
            with archive.open(lines_name) as source:
                for raw in source:
                    if cancelled():raise InterruptedError()
                    parts=raw.decode('latin-1').rstrip('\r\n').split(' +++$+++ ',4)
                    if len(parts)!=5:
                        raise ValueError('Malformed Cornell utterance record.')
                    db.execute('INSERT OR REPLACE INTO utterances VALUES(?,?)',(parts[0],parts[4]))
            db.commit()
            with archive.open(conversations_name) as source:
                for raw in source:
                    if cancelled():raise InterruptedError()
                    fields=raw.decode('latin-1').rstrip('\r\n').split(' +++$+++ ')
                    ids=ast.literal_eval(fields[-1])
                    if not isinstance(ids,list) or any(not isinstance(i,str) for i in ids):
                        raise ValueError('Invalid Cornell conversation IDs.')
                    turns=[]
                    for line_id in ids:
                        found=db.execute('SELECT text FROM utterances WHERE id=?',(line_id,)).fetchone()
                        if not found:raise ValueError('Missing Cornell utterance '+line_id)
                        turns.append(found[0])
                    yield {'turns':turns,'movie_id':fields[2]}
    finally:
        db.close()
