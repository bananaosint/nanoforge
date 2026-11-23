# Optional Chinese translation

### Chinese dataset skipped by default

As requested, Chinese R1 Distill is excluded from the UI catalog and **Import all five**. No translation model or translation dependencies are installed by the normal setup. The remaining translation code is optional and is not used by any of the five imports.

If you later explicitly want to try it, install `backend/requirements-translation.txt` and run `.venv/bin/python -m backend.importers --dataset chinese-r1-en`. The following describes that optional workflow.

The separate Argos zh→en 1.9 translation model runs locally with CTranslate2 on CPU/int8. It is used **only for data preparation**. NanoForge's GPT still initializes from random weights and never loads pretrained language-model weights. The translator downloads once, then can run offline. Its code and model README are stored in `data/.imports/translator/`.

The importer translates `input`, `reasoning_content`, and `content`. Reasoning is retained in `<think>...</think>` followed by the final answer. English-only strings and code blocks are preserved. Long prose is split into bounded translation segments so text is not silently truncated. A record with conversion errors, a decoding-limit problem, or Chinese characters remaining after translation is excluded from the training corpus and written to `data/.imports/chinese-r1-en/review.jsonl` with its original fields. The UI shows review counts.

Translation progress is committed after each record; a persistent SQLite translation cache also saves completed fields. A full 110k-record translation can take **days**, depending on CPU and record lengths. This is machine translation, not a guarantee of mathematical or linguistic correctness; review samples before investing in a full training run. `examples/chinese-r1-english-preview.jsonl` contains one actual translated record, explicitly marked as a preview. The full translated corpus is not bundled in the ZIP and is never silently replaced with this sample.

The Chinese source is not exposed to the mixer while translation is incomplete. Its finalized English dataset becomes available only after the pass completes; held-for-review records are not fed to the model.

