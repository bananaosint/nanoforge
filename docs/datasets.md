# Dataset catalog and large imports

The Data & Mixer panel contains 22 sources. Select sources with their checkboxes, choose a full training split or a 1,000/10,000-source-row sample, and click **Import selected**. Each card also has individual **Import full** and **Sample 1,000** controls. Chinese R1 remains excluded from the default catalog. The ZeroGPU Gradio adapter exposes the same source catalog for bounded 1,000-source-row CPU samples, subject to its separate 20 MiB preparation limit; full imports remain in Studio/CLI.

Cosmopedia v2 and Wikipedia begin unselected because full downloads are large. DailyDialog begins unselected because the supplied repository is currently inaccessible. The cards remain visible, so none of the requested sources is silently replaced or hidden.

| Import ID | Repository / configuration | Normalization |
| --- | --- | --- |
| cosmopedia-v2 | HuggingFaceTB/smollm-corpus / cosmopedia-v2 | Generated textbook text; original prompt/audience/format retained as metadata. Text pretraining, not invented SFT pairs. |
| wikipedia-en | wikimedia/wikipedia / 20231101.en | English article title + body; article ID, title and URL preserved. |
| magicoder | ise-uiuc/Magicoder-OSS-Instruct-75K / default | Problem → solution; code indentation and language retained. Code is never executed. |
| dolly | databricks/databricks-dolly-15k / default | Instruction + optional context → response. |
| daily-dialog | myname426/li2017dailydialog-bucket / default | Dialogue utterances become assistant-turn examples with preceding context, when accessible. This exact repository returned HTTP 401 during verification; its live schema could not be verified. |
| persona-chat | awsaf49/persona-chat / train_self_original.txt | One canonical training variant; persona + preceding conversation → observed response. Candidate/distractor responses and duplicated variants are excluded. |
| cornell-movie | cornell-movie-dialog/cornell_movie_dialog / original Cornell archive | Decode original Latin-1 files, join line IDs through an on-disk index, and train on actual utterances. No remote dataset script is executed. |
| altered-riddles | marcodsn/altered-riddles / default | Separate original and altered examples with their matching answer and reasoning; accepted answers and variant retained. |
| philosopher-quotes | datastax/philosopher-quotes / default | Quote + author as pretraining text, with author/tags retained. |
| science-facts | Royal-lobster/10001-Science-Facts / default | Fact text with category, source URL and original source fields retained. |
| chess-puzzles | Lichess/chess-puzzles / default | Apply the first opponent setup move to the FEN, then prompt for the remaining UCI solution line. Every move is checked for legality. Rating, themes, puzzle ID, setup move and original FEN are retained. |
| smoltalk | HuggingFaceTB/smoltalk / all | Each assistant message becomes a completion with only its preceding role-tagged conversation as context; system messages and source subset are retained. Select all/train once, without also importing its component configurations. |
| tulu-3-sft-mixture | allenai/tulu-3-sft-mixture / default | Same assistant-turn conversion; example ID and upstream source subset retained. Includes multilingual data, preserved without translation/filtering. |
| tulu-3-sft-personas-math-grade | allenai/tulu-3-sft-personas-math-grade / default | Exact unfiltered repository, with assistant responses, example IDs and original prompts. |
| tulu-3-sft-personas-instruction-following | allenai/tulu-3-sft-personas-instruction-following / default | Same chat conversion; original prompt, example ID and constraints retained. |
| tulu-3-sft-personas-algebra | allenai/tulu-3-sft-personas-algebra / default | Exact requested sft-personas-algebra repository, retaining prompt/ID metadata; no substitution with similarly named datasets. |
| codealpaca-20k | HuggingFaceH4/CodeAlpaca_20K / default | Existing prompt → completion; whitespace/indentation preserved. Only H4's train split is imported; its held-out test split stays excluded. |

The original `tinystories`, `hh-rlhf`, `oasst1`, `no-robots`, and `goat` IDs continue to work. Only original training splits are imported. Wikipedia explicitly selects English; SmolLM explicitly selects Cosmopedia v2, not the other SmolLM corpus configurations.

The Tulu mixture includes persona material and other sources also available separately in the catalog. Imports preserve repository identity and do not deduplicate across corpora: weighting a mixture alongside its components can repeat examples. Choose weights accordingly. Multi-turn rows produce one example per assistant response, without later turns leaking into earlier prompts. Code is imported as text and never executed. Malformed conversations are held in the existing review file before any examples from that row are emitted.

## Size and memory

At the verified revisions, Cosmopedia v2 has 104 Parquet shards totaling **122,361,137,711 compressed bytes**. English Wikipedia has 41 shards totaling **11,630,929,031 compressed bytes**. Lichess is approximately 885 MB compressed. Normalized JSONL and token caches require additional space; compressed size is not the final training footprint.

The six new train sources total approximately 3.55 GB compressed: SmolTalk `all` 2,008,727,125 bytes (9 shards); Tulu mixture 1,412,954,868 bytes (6 shards); grade math 51,970,185 bytes; instruction following 39,171,921 bytes; algebra 36,167,938 bytes; H4 CodeAlpaca 3,008,277 bytes. These are metadata sizes for full train splits, not amounts downloaded by a sample. Normalizing multi-turn conversations repeats context for each target and can expand the final output.

Full imports now download, process and release **one Parquet shard at a time**, with a durable row checkpoint before deleting each completed shard. Resume skips committed shards using recorded row counts. OASST's small parent index and original raw dialogue sources have separate parsing paths. Samples use bounded HTTP byte-range reads to fetch real rows from the same pinned Parquet revision as full imports, without depending on the public viewer or its truncation limits. Samples never silently become full downloads. A server that ignores byte ranges produces a clear error instead of downloading a massive file into RAM.

Only finalized UTF-8 files enter the mixer. Interrupted output remains under `data/.imports/`; resume truncates uncommitted bytes, preserving a duplicate-free committed prefix. A completed sample can be upgraded to a full import. The sample file remains available separately; select the sample or full corpus in the mixer rather than unintentionally weighting both copies.

Source-row limits count original rows/conversations, not normalized records: ten riddle rows produce twenty examples; a dialogue row can produce several assistant-turn examples. Text-only corpora correctly contribute zero SFT/DPO pairs. Imported instruction examples are available to SFT, and HH-RLHF preserves DPO preference pairs, without flooding the human feedback table.

## Access and provenance

The exact DailyDialog ID returned Unauthorized; it may be private, missing, or misspelled. Its card reports this state. Check its URL, or supply `HF_TOKEN` in the backend environment if your account is authorized. Tokens are used only for Hugging Face requests and are not forwarded to redirected CDN/third-party hosts. No alternate dataset is substituted automatically.

The importer records repository/config/split, source revision, pinned converted-Parquet revision, download URLs, counts, exclusions and license metadata in `data/.imports/jobs.json`. Preserve that metadata with exported corpora. Wikipedia and Dolly have share-alike terms; No Robots is noncommercial; other sources have their own terms linked on each card. Cornell uses its original archive/terms. Do not treat the catalog's short license labels as replacing the full source terms.

SmolTalk combines subsets with different terms; its Apache 2.0 label covers the newly generated subsets, not every component. Tulu's mixture is ODC-BY at collection level with separate subset terms, including noncommercial portions. The grade-math and exact algebra cards did not declare a top-level license during verification, so those entries point to the source card rather than assuming one. H4 CodeAlpaca declares the generic `cc` label; consult its linked upstream terms for the exact conditions.

Altered Riddles is a benchmark. If used for training, it is no longer an independent evaluation set for that model. Science-fact claims and machine-generated code are retained as provided, not independently fact-checked.

## CLI

```bash
# A modest source, complete train split
.venv/bin/python -m backend.importers --dataset dolly

# Explicit sample of a very large corpus
.venv/bin/python -m backend.importers --dataset cosmopedia-v2 --max-records 1000

# New instruction/chat sources use the same resumable downloader
.venv/bin/python -m backend.importers --dataset smoltalk --max-records 1000
.venv/bin/python -m backend.importers --dataset tulu-3-sft-personas-algebra
.venv/bin/python -m backend.importers --dataset codealpaca-20k

# Upgrade the completed sample to the full corpus only when you intend the large download
.venv/bin/python -m backend.importers --dataset cosmopedia-v2
```

`--dataset all` explicitly includes the massive sources and the inaccessible DailyDialog ID; prefer individual IDs or the UI selection controls. Do not run a CLI import and app import into the same data directory simultaneously.

See `expanded-import-verification.json` for real-source checks and their scope. Large corpora were sampled, not downloaded in full in the development workspace.

See `instruction-import-verification.json` for live pinned-Parquet checks of the six new repositories. Tests also exercise split/config selection, assistant-turn boundaries, code whitespace, invalid-row review, uint16 pretraining caches and SFT pair indexing through the full import pipeline.
