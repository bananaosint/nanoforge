"""Small, CPU-only Chinese→English data preparation; independent of NanoForge GPT."""
import hashlib
import json
import re
import sqlite3
import zipfile
from pathlib import Path

MODEL_URL = 'https://argos-net.com/v1/translate-zh_en-1_9.argosmodel'
CJK = re.compile(r'[\u3400-\u9fff\uf900-\ufaff]')


class LocalTranslator:
    def __init__(self, directory, download, cancelled):
        import ctranslate2
        import sentencepiece
        self.cancelled = cancelled
        root = Path(directory)
        root.mkdir(parents=True, exist_ok=True)
        folder = root / 'translate-zh_en-1_9'
        if not (folder / '.nanoforge-ready').exists():
            archive = root / 'zh-en.argosmodel'
            download(MODEL_URL, archive, expected=74481402)
            with zipfile.ZipFile(archive) as package:
                for member in package.infolist():
                    target = (root / member.filename).resolve()
                    if not target.is_relative_to(root.resolve()):
                        raise ValueError('Invalid translation model archive.')
                package.extractall(root)
            (folder / '.nanoforge-ready').write_text('zh-en-1.9 complete\n')
            archive.unlink(missing_ok=True)
        self.model = ctranslate2.Translator(str(folder / 'model'), device='cpu', compute_type='int8', inter_threads=1, intra_threads=2)
        self.tokenizer = sentencepiece.SentencePieceProcessor(model_file=str(folder / 'sentencepiece.model'))
        self.cache = sqlite3.connect(root / 'translation-cache.sqlite3')
        self.cache.execute('CREATE TABLE IF NOT EXISTS translations(key TEXT PRIMARY KEY, result TEXT)')

    def __call__(self, text):
        text = text or ''
        if not CJK.search(text):
            return text
        key = hashlib.sha256(('zh-en-1.9-int8-v2:' + text).encode()).hexdigest()
        row = self.cache.execute('SELECT result FROM translations WHERE key=?', (key,)).fetchone()
        if row:
            return row[0]
        result = []
        # Preserve code fences verbatim. Chinese inside code will be flagged for
        # review by the importer, never silently trained as English.
        code = False
        for line in text.splitlines(keepends=True):
            if line.lstrip().startswith('```'):
                code = not code
                result.append(line)
                continue
            if code or not CJK.search(line):
                result.append(line)
                continue
            translated = []
            for segment in re.split(r'(?<=[。！？])', line.rstrip('\r\n')):
                if not segment:
                    continue
                if not CJK.search(segment):
                    translated.append(segment)
                    continue
                tokens = self.tokenizer.encode(segment, out_type=str)
                chunks = [tokens[i:i + 192] for i in range(0, len(tokens), 192)]
                for chunk in chunks:
                    if self.cancelled():
                        raise InterruptedError('Import paused.')
                    prediction = self.model.translate_batch([chunk], beam_size=2, max_decoding_length=768)[0]
                    output = prediction.hypotheses[0]
                    # A full decoding budget can indicate truncation; do not feed it.
                    if len(output) >= 768:
                        raise ValueError('Translation reached its output limit; record needs review.')
                    translated.append(self.tokenizer.decode(output).replace("▁", " "))
            result.append(' '.join(translated) + ('\n' if line.endswith('\n') else ''))
        output = ''.join(result)
        self.cache.execute('INSERT OR REPLACE INTO translations VALUES(?,?)', (key, output))
        self.cache.commit()
        return output

    def close(self):
        self.cache.close()
