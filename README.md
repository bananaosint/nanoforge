# NanoForge

A local studio for building a small GPT from scratch, watching it learn, and improving it with human feedback. Python/PyTorch handles the model; React handles the studio. No pretrained weights, Hugging Face Trainer, hosted inference, account, or API key.

## Start

Requires **Python 3.10+**, **Node.js 20.19+ or 22.12+**, npm, and a POSIX shell (macOS/Linux; use WSL on Windows). Allow at least 3 GB free for the CPU environment, frontend packages, and temporary installation files; exact usage varies by platform and package versions. NVIDIA CUDA installations can require several additional gigabytes. Initial dependency installation needs internet. After setup, training, inference, feedback, bundled GPT-2 tokenization, and the UI work offline.

```bash
chmod +x start.sh
./start.sh
```

Open **http://localhost:5173**. Interactive API documentation: **http://localhost:8000/docs**. Ctrl+C stops both servers, finishes the current training step, and saves a checkpoint before SQLite closes. Use one backend process and one worker: the active model lives in process memory.

`start.sh` creates `.venv`, installs Python requirements and locked frontend packages on first launch, then starts both servers bound to loopback. Fresh Linux/WSL installations use CPU-only PyTorch by default, avoiding NVIDIA CUDA downloads. The installer keeps temporary files inside `.venv/install-tmp` on the project filesystem instead of a potentially small RAM-backed `/tmp`, and disables pip’s duplicate download cache. Existing working PyTorch installations are preserved. macOS uses the ordinary PyPI wheel, including Apple silicon MPS support. If `python3` is not your preferred interpreter, use `PYTHON_BIN=python3.12 ./start.sh`.

### Manual setup and separate servers

From the repository root:

```bash
python3 -m venv .venv
mkdir -p .venv/install-tmp
export TMPDIR="$PWD/.venv/install-tmp"
# Linux / WSL: install CPU-only PyTorch first to avoid large CUDA downloads.
.venv/bin/python -m pip install --no-cache-dir 'torch>=2.4,<3' --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install --no-cache-dir -r backend/requirements.txt
cd frontend
npm ci
cd ..
```

Backend, from the repository root:

```bash
.venv/bin/python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

Frontend, in another terminal:

```bash
cd frontend
npm run dev -- --host 127.0.0.1 --port 5173 --strictPort
```

For an NVIDIA CUDA installation on Linux/WSL, opt in on a fresh environment (large download; requires a compatible GPU and driver):

```bash
NANOFORGE_TORCH=cuda ./start.sh
```

On macOS, skip the CPU-index command in the manual setup and install the requirements from PyPI. On Apple silicon, ordinary PyPI PyTorch provides MPS. Device selection is CUDA → XPU → MPS → CPU. Intel XPU is optional; see `docs/intel-xpu.md` for hardware requirements. The Core 3 100U integrated GPU is outside the officially validated XPU hardware list, so keep the CPU build on that laptop. CPU threads are capped at four to avoid overhead for tiny models. If ports 8000 or 5173 are occupied, stop the existing process; changing API ports also requires updating the proxy in `frontend/vite.config.ts`.

## Recovering from a WSL no space error

`[Errno 28] No space left on device` refers to the filesystem receiving a write, which may be a temporary filesystem rather than your main Windows/Linux disk. Check it from the NanoForge folder:

```bash
df -h . /tmp
```

A small `tmpfs` at `/tmp` can fill during PyTorch wheel downloads/unpacking even when the project filesystem has hundreds of gigabytes free. Pip often removes its temporary files after a failed download, so `/tmp` may appear empty afterward. The updated `start.sh` automatically uses the project filesystem for temporary downloads and chooses CPU-only PyTorch on Linux/WSL.

To recover using an older copy of `start.sh`, run these commands inside the NanoForge folder:

```bash
mkdir -p .venv/install-tmp
export TMPDIR="$PWD/.venv/install-tmp"
export PIP_NO_CACHE_DIR=1
.venv/bin/python -m pip install --no-cache-dir 'torch>=2.4,<3' --index-url https://download.pytorch.org/whl/cpu
./start.sh
```

These commands do not delete your data, checkpoints, feedback, or existing environment. An interrupted installation can usually be retried. The `.venv/install-tmp` directory holds only installation temporary files. `NANOFORGE_SETUP_ONLY=1 ./start.sh` installs/checks dependencies without starting either server.

## Direct Hugging Face dataset imports

Start the updated app and open **Data & Mixer → Direct dataset imports**. Select source checkboxes and click **Import selected**, or use an individual card’s **Import full** / **Sample 1,000** buttons. The catalog contains 16 requested sources; see `docs/datasets.md` for the full source/configuration table. Cosmopedia v2 and English Wikipedia start unselected because of their size, and the supplied DailyDialog ID reports its access issue. Files download directly onto the machine running the backend, bypassing the browser's 20 MB upload limit. Completed sources appear in the mixer at 0%: move their weight sliders to select them. A download in this chat's cloud workspace does not modify your own WSL installation.

| Source | Imported material | Approximate download |
| --- | --- | --- |
| roneneldan/TinyStoriesInstruct | Entire original training text, preserving story/instruction boundaries | 2.66 GB |
| Anthropic/hh-rlhf | Full train split; common conversation prompt plus chosen/rejected completions; mismatched branches held for review | 172 MB compressed |
| OpenAssistant/oasst1 | Full train split scanned; reconstruct English assistant responses with their available English ancestor turns; non-English/deleted/orphaned turns excluded | 40 MB compressed |
| HuggingFaceH4/no_robots | Full train split, conversation context and final assistant answer | 10 MB compressed |
| tiedong/goat | Full train split, arithmetic instruction/input and worked answer | 206 MB compressed |


These are original **train** splits. Original test/validation splits are not mixed into training. NanoForge makes its own internal 90/10 split after normalization. Dataset source links, licensing labels, source revision, converted Parquet revision, and download URLs are recorded under `data/.imports/jobs.json`. No Robots is licensed **CC BY-NC 4.0**; preserve source attribution and its noncommercial restriction. The UI links every dataset card for full terms.

**Pause imports** saves progress. Resume with the same dataset button after restarting. Interrupted downloads use HTTP range requests when supported. Normalization uses row checkpoints and truncates uncommitted output on resume, preventing duplicate records. Incomplete output stays hidden under `data/.imports/`; the mixer sees only finalized files. Standard imports stream through Arrow batches and release intermediate Parquet files after conversion. Converted files can still take several GB. Allow additional disk space for token caches: token caches use two bytes per token when the vocabulary fits uint16 (including Magic 8k and GPT-2 BPE), with uint32 fallback for larger vocabularies. Your WSL project filesystem has much more room than the small `/tmp`; imports write to the project filesystem.

For pretraining, only the chosen HH-RLHF completion is used as text; rejected completions remain available as preference targets. SFT/DPO also read imported JSONL pairs directly from positively weighted sources in the mixer, alongside your saved human feedback. They use disk indexes rather than copying millions of pairs into the HITL feedback table. Collected human edits remain independently editable/exportable in the feedback library. Select the imported source in the mixer before using **Train on feedback**.

### Chinese dataset skipped

Chinese R1 Distill is excluded from the UI catalog and the normal import selection, as requested. Normal setup installs no translation dependencies and downloads no translation model. Optional experimental translation code is documented separately in `docs/optional-translation.md` and does not run automatically.

### Large-corpus training

File-backed corpora now scan their vocabulary incrementally and tokenize into reusable `data/.tokens/*.u16` (or `.u32` for large vocabularies) memory-mapped caches. Training samples only the requested windows into RAM. Raw `.txt` and normalized `.jsonl` corpora no longer need to fit entirely in memory. Initial vocabulary/token preparation can take minutes; subsequent runs with the same source/tokenizer reuse the cache. BPE encoding happens in bounded chunks; a merge can differ at a chunk boundary, but no characters are dropped. Importing a dataset does not automatically change an existing checkpoint's vocabulary: with a character tokenizer, choose the complete mix **before creating a new experiment**, or use Magic 8k / GPT-2 BPE for a fixed broad vocabulary.

Old caches are not automatically deleted, so deleting or replacing large datasets may leave unused files in `data/.tokens/`. With the app stopped, that directory can be removed to reclaim space; it contains reproducible caches, not checkpoints or feedback.

### Terminal alternative

From the NanoForge root, with the updated dependencies installed:

```bash
.venv/bin/python -m backend.importers --dataset all
```

Or choose `tinystories`, `hh-rlhf`, `oasst1`, `no-robots`, or `goat` individually with `--dataset`. Ctrl+C pauses safely; rerun the same command to resume. Run one importer process at a time; do not run a CLI import while the app is importing into the same data directory. Advanced verification can use `--max-records 200 --home /path/to/separate/sample-home`; this produces explicitly named sample files and never claims a full import. Completed samples can be upgraded to full imports; interrupted imports must resume with the same row limit. Sample and full files remain distinct.

### Updating an existing copy

Stop NanoForge first. Extract the updated ZIP into a temporary folder, then copy its `backend/`, `frontend/`, `examples/`, `package.json`, `start.sh`, and `README.md` into your existing NanoForge folder, replacing code files. Keep your existing `.venv/`, `data/`, `checkpoints/`, and `nanoforge.sqlite3*` files. Run `./start.sh` again: it installs the additional import/translation dependencies without recreating your environment. The ZIP contains no trained checkpoints, user feedback, or full downloaded corpora.

## Kaggle GPU T4 x2 — full Studio UI

Select **GPU T4 x2** and enable **Internet**, then run [the Kaggle dual-T4 notebook](notebooks/NanoForge_Kaggle_T4x2.ipynb). It verifies both devices, runs the real two-GPU smoke tests and opens a temporary password-protected Studio link. All model/dataset/training controls stay in the UI. FP16 training can use both GPUs for pretraining, SFT and DPO; checkpoints remain portable to local Studio. Each GPU holds a complete model replica, so GPU memory is not pooled. See [setup, authentication and memory details](docs/kaggle-t4x2.md).

## Kaggle TPU x8

Import [the Kaggle TPU notebook](notebooks/NanoForge_Kaggle_TPU.ipynb) into Kaggle, enable Internet, select **TPU x8**, and run its cells. It installs/reuses matching PyTorch/XLA, runs an actual eight-core smoke test, prepares uint16 tokens, trains with **BF16**, and exports Studio-compatible CPU `.pt` weights. `configs/kaggle-tpu.json` uses Mini, batch 4 per core, accumulation 2 (effective batch 64). Use `preset: "Custom"` for larger architectures below one billion parameters.

This is data parallel training: each core holds a full model; core memories are not pooled. The separate `python -m backend.train_xla` CLI supports pretraining, SFT and DPO. The local Studio server remains on CPU/CUDA/MPS/XPU. See [Kaggle setup, resume and memory details](docs/kaggle-tpu.md). Actual TPU kernels require the notebook's hardware test; development checks here run on CPU.

## Colab A100 GPU — full Studio UI

[Open the A100 Studio notebook](https://colab.research.google.com/github/bananaosint/nanoforge/blob/main/notebooks/NanoForge_A100.ipynb). Select **A100 GPU** and run the launcher cell. It installs the Python/frontend dependencies, checks the actual GPU, runs CUDA checkpoint smoke tests and displays the full UI below the cell through Colab's notebook proxy. Model size, datasets and all training settings stay in the UI; startup does not import data or start training.

The A100 profile uses Mini, Magic 8k, FP16 + GradScaler, TF32, context 512, batch 64 and accumulation 1. Caches remain uint16 when IDs fit, and `.pt` files retain portable CPU FP32 weights. Use smaller batches for larger Custom models. These defaults are a starting point, not an A100 benchmark. See [A100 setup and storage](docs/colab-a100.md).

## Colab L4 GPU — full Studio UI

[Open the L4 Studio notebook](https://colab.research.google.com/github/bananaosint/nanoforge/blob/main/notebooks/NanoForge_L4.ipynb). Select **L4 GPU** and run its launcher cell. It verifies CUDA, builds the frontend and displays the complete Studio below the cell through Colab's notebook proxy. **Model size, datasets, hyperparameters, training controls, feedback and checkpoint downloads are in the UI.** Startup leaves training idle and imports no corpora automatically.

The L4 defaults are Mini, Magic 8k, FP16 + GradScaler, context 512, batch 32, accumulation 2 and TF32. Custom architectures stay below one billion parameters; adjust batch size to fit memory. Outputs retain the portable CPU FP32 checkpoint format. No public tunnel or tunnel account is needed. Colab uses HTTP polling for live progress; the desktop app retains WebSockets. See [Colab UI startup, session controls and storage](docs/colab-l4.md). The earlier immediate-training cell remains in [the optional headless notebook](notebooks/NanoForge_L4_Headless.ipynb).

## Colab Tesla T4

[Open the T4 notebook in Colab](https://colab.research.google.com/github/bananaosint/nanoforge/blob/main/notebooks/NanoForge_T4.ipynb)

Choose a T4 GPU runtime and run the notebook. `configs/colab-t4.json` uses Mini, Magic 8k, FP16 autocast + gradient scaling, fused AdamW, context 256, batch 32 and gradient accumulation 2. Pretokenization stores uint16 IDs; only sampled batches become int64 CUDA tensors. Parameters and exported checkpoint weights remain FP32 for stability and CPU compatibility.

Download the notebook’s checkpoint ZIP and use **Train & Monitor → Import .pt checkpoint**, or copy its `.pt` and `.json` into your local `checkpoints/` folder. The studio restores the embedded tokenizer/model/optimizer state on its own device. See `docs/colab-t4.md` for setup, resume, precision behavior and verification limits; see `docs/tokenizers.md` for the exact Magic tokenizer semantics.

## First experiment

1. **Train & Monitor:** choose Nano and character-level tokenization. The offline starter corpus is selected initially. Set 250 steps, batch size 8, and evaluation interval 50. Start training initializes random weights if necessary. “New experiment” explicitly replaces the current model with fresh weights and a new run ID.
2. **Data & Mixer:** download TinyShakespeare or upload your own UTF-8 `.txt`/`.jsonl`. Assign percentages with the sliders. Downloading adds a source at 0%; move its slider to include it. The trainer snapshots the source mix at the start of a run, so changes apply to the next run. Failed downloads use a clearly reported, small offline fallback.
3. **HITL Testing Studio:** generate, edit a completion, and save the ideal target. Or compare two seeds/two checkpoints in the Arena and vote. “Train on feedback” starts SFT or DPO and opens the live monitor. On completion, the new checkpoint is automatically active; return to the Studio to generate again.
4. **Glass-Box Inspector:** generate a sequence, click a token to examine its top-5 sampling alternatives, then choose an attention layer and head. Hover cells for exact probabilities and token positions. “View context tokens” exposes the full axis mapping.

Short runs prove the pipeline; they do not produce fluent language. The deliberately repetitive offline starter is for sanity checks. Download TinyShakespeare and train for thousands of steps for a meaningful learning curve. This is a small autoregressive language model, not a pretrained assistant.

## Architecture and learning code

```text
backend/
  main.py                  FastAPI routes, CORS, WebSocket, shutdown lifecycle
  storage.py               SQLite feedback, settings, run history, metrics, snapshots
  api/schemas.py           Validated request contracts
  ml/tokenizer.py          Character, GPT-2 BPE, or bundled Magic 8k BPE
  ml/model.py              Pre-norm decoder-only Transformer and sampling
  ml/dataset.py            Corpus handling, weighted batches, completion masks
  ml/corpus.py             Streaming files, memory-mapped tokens and feedback indexes
  importers/               Direct Hugging Face imports and local Chinese→English preparation
  ml/trainer.py            Background thread, CE / SFT / DPO, checkpoints
  assets/tokenizer/        Verified original GPT-2 vocabulary cache (no model weights)
  tests/                   Model, import, tokenizer, device, precision and API tests
  train_cli.py             Headless training and resume (same Studio trainer)
  pretokenize.py           Packed uint16/uint32 cache preparation
frontend/
  src/App.tsx              Four studio workspaces
  src/types.ts             Shared UI data contracts
  src/style.css            Responsive dark studio + Tailwind
  public/fonts/            Self-hosted fonts and licenses
  vite.config.ts           Same-origin API/WebSocket development proxies
data/                      Local text and JSONL corpora
checkpoints/               .pt weights + readable JSON metadata
nanoforge.sqlite3          Created on first backend launch
start.sh                   Setup and one-command local launcher
```

The model exposes learned token and positional embeddings, explicit Q/K/V projections, causal multi-head attention, GELU MLPs, pre-norm residual blocks, final LayerNorm, and a tied output projection. The regular path uses PyTorch scaled-dot-product attention. `inspect=True` explicitly computes masked softmax attention and returns one `[B, heads, T, T]` tensor per layer. Generation preserves the model's previous train/eval state and uses independent seeded sampling.

| Preset | Layers | Heads | Width | Context | Approximate parameters* |
| --- | ---: | ---: | ---: | ---: | ---: |
| Nano | 4 | 4 | 128 | 128 | 0.8M |
| Micro | 6 | 4 | 256 | 256 | 4.8M |
| Mini | 8 | 6 | 384 | 256 | 14.3M |

*With a small character vocabulary. GPT-2 BPE has 50,257 tokens, so its tied embedding matrix substantially increases the parameter count (Nano BPE is about 7.2M). The status bar shows the actual count. Custom widths must be divisible by head count. UI architecture/tokenizer changes require a **new experiment**; a loaded checkpoint always retains its original architecture and tokenizer.

### Custom model size (below 1 billion parameters)

In **Train & Monitor → Model preset → Custom**, enter layers, heads, embedding width, context and dropout. The previous 12-layer / 768-width limits are removed. Width must divide evenly into heads. The live preview includes tokenizer embeddings, learned positions, biases and the shared output matrix counted once. Character vocabulary is measured from the selected datasets; changing data or tokenizer can change the count.

For a desired size, enter **Parameter budget (millions)** and choose **Fit width to budget**. It keeps layers, heads and context fixed and chooses the largest head-divisible width at or below the budget. Discrete dimensions mean not every exact integer parameter count can be achieved. Click **New experiment** to initialize the chosen architecture; loaded models retain their saved dimensions.

All new models, API/CLI configurations and checkpoint model configurations must have **at most 999,999,999 parameters**. The backend checks this before allocating model weights. Preview and budget fitting allocate no model tensors. The preview shows FP32 weight storage and an approximate 16 bytes per parameter for weights, gradients and AdamW state. Activations, temporary buffers and the extra DPO reference model need more memory; the parameter limit does not guarantee a model fits a particular GPU. Checkpoint uploads allow up to 16 GiB for larger weights and optimizer states.

For headless/Colab training, set `"preset": "Custom"` and supply dimensions in the JSON config, for example:

```json
"model": {"n_layer": 12, "n_head": 8, "n_embd": 768, "block_size": 256, "dropout": 0.1}
```

The same count check applies regardless of training precision. Start with a small batch when increasing size on a T4.

### Objectives

- **Pretrain / continue:** next-token cross-entropy on source-weighted random windows. Each selected source gets a contiguous 90/10 train/validation split. The sampler chooses a source per example in proportion to its weight. Very short sources use a shorter common context; sources with fewer than 20 tokens are rejected.
- **SFT:** samples corrected `(prompt, target)` pairs, formats `<|prompt|>\n…\n<|completion|>\n…\n<|end|>\n`, and sets prompt-token labels to `-100`. Long examples retain a prompt suffix and truncate the completion to the context limit. Optionally replace 10–20% of batches with the base data mixture. Generation from SFT/DPO checkpoints automatically applies the same prompt delimiters. The exported text includes generated delimiters if the model predicts them.
- **DPO:** clones a frozen reference at the beginning of each pass. Computes **summed completion-token** log probabilities for chosen and rejected outputs, with prompt positions masked. Optimizes `-logsigmoid(beta * ((pi_chosen - ref_chosen) - (pi_rejected - ref_rejected)))`. Dropout is disabled for policy/reference comparisons. Gradient accumulation and clipping apply to all modes. The reference is discarded after the pass; a later DPO pass clones the newly active model.

Character vocabularies are fixed at creation and saved in checkpoints. Unseen correction characters map to the reserved `�` unknown token. Create a new model with a broader corpus or choose BPE if you need arbitrary Unicode feedback. GPT-2 BPE files are bundled and hash-verified by tiktoken; tokenization needs no network.

Three configurable pinned prompts are sampled before pretraining (step 0 for a new model) and at each evaluation, including the final step. SQLite stores snapshots with their experiment and step; the timeline selector can display previous experiments. Validation loss is available for pretraining; SFT and DPO display their own training objective without a stale validation loss.

### Controls and persistence

- **Pause / Resume** keep the worker, optimizer, and DPO reference in memory.
- **Stop** finishes the current optimizer step and saves a version. A stopped/completed run can continue with another step budget.
- **Load / branch** restores model, tokenizer, optimizer, and absolute step; subsequent training saves a new run/version. DPO's reference resets for a new tuning pass.
- `.pt` checkpoints contain `model_config`, `tokenizer_state`, `model_state_dict`, `optimizer_state_dict`, `step`, `val_loss`, `version_tag`, training config, mode, and run ID. Files and metadata are written via temporary files then replaced atomically.
- Active model selection is in memory. After restarting, select a saved checkpoint to restore it. SQLite feedback, metrics, run configuration, pinned prompts, and timeline history remain on disk. Use `NANOFORGE_HOME=/some/path` for a different runtime data/checkpoint/SQLite directory.
- Checkpoints are local PyTorch files loaded with `weights_only=True`. Use checkpoints created by this application. No training data, weights, or prompts are uploaded.
- Dataset weights and ordinary UI sampling controls are session state. Dataset files and collected feedback are durable.

NaN/Inf loss or gradients stop training with a visible error. Empty/invalid datasets, invalid model dimensions, missing feedback, invalid layer/head selections, and generation before one training step return readable errors. Generation and model operations acquire the same lock as a training step; an inference request can wait for the current step, but the API event loop and telemetry remain responsive. Shutdown waits for that step and the final checkpoint.

## Data formats

Plain text is read as a continuous corpus. Each nonempty JSONL line can be a JSON string, an object with `text`, or a `prompt` plus `completion`, `target`, or `corrected_output`:

```jsonl
{"text":"A small model can still teach us a lot."}
{"prompt":"What is a token?","completion":"A piece of text represented by an integer."}
```

The API rejects malformed JSONL, empty files, duplicates, non-UTF-8 files, and uploads over 20 MB. It never overwrites an existing corpus. The Data tab previews source text and supports deletion. Feedback corrections and preference pairs have independent SQLite CRUD and JSONL export.

## API surface

Direct imports use `GET /api/imports`, `POST /api/imports/start` (`dataset_id` or `dataset_ids`, optional `max_records=0` for full import), and `POST /api/imports/pause`.

All REST paths below are prefixed by `/api`; `/docs` provides complete OpenAPI schemas.

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/health`, `/runtime`, `/status`, `/presets` | Health, UI environment/defaults/transport, active model/metrics, preset definitions |
| POST | `/models/preview` | Exact parameter/storage preview; optional `target_parameters` fits width to budget without creating a model |
| GET / POST | `/runs` | Run history / initialize a model |
| GET | `/runs/{id}/metrics` | Durable loss and throughput history |
| GET | `/datasets` | List corpora |
| POST | `/datasets/upload`, `/datasets/tinyshakespeare` | Upload / download corpus |
| GET / DELETE | `/datasets/{name}/preview`, `/datasets/{name}` | Preview / remove corpus |
| POST | `/training/start`, `/training/pause`, `/training/resume`, `/training/stop` | Lifecycle; `start.mode` selects pretrain, sft, dpo |
| GET | `/checkpoints` | Backend checkpoint metadata |
| GET | `/checkpoints/{name}/download` | Download a portable .pt checkpoint |
| POST | `/checkpoints/save`, `/checkpoints/select`, `/checkpoints/upload` | Save / restore / import portable .pt weights and optimizer |
| DELETE | `/checkpoints/{name}` | Remove an inactive checkpoint |
| GET / PUT | `/timeline`, `/prompts` | Experiment snapshots / exactly three pinned prompts |
| POST | `/generate`, `/arena` | Completion with optional telemetry / dual sampling |
| GET / POST | `/feedback` | List / collect human feedback |
| PUT / DELETE | `/feedback/{id}` | Edit / remove feedback |
| GET | `/feedback-export` | Download JSONL, optionally `?kind=sft` or `preference` |
| WS | `/ws/training` (no `/api` prefix) | Status and telemetry every 600 ms |

`/generate` supports `prompt`, `max_new_tokens`, `temperature`, `top_k` (0 disables truncation), `seed`, `telemetry`, and optional zero-based `layer` / `head` filters. Returned candidate probabilities reflect the selected sampling settings. Attention covers the final generated sequence, cropped to the context window. `/arena` accepts optional `checkpoint_a`, `checkpoint_b`, and `seed_b`; null checkpoints use the active model. Responses include the exact checkpoint and seed used.

## Verification

```bash
# From repository root
.venv/bin/python -m pytest backend/tests -q

# Frontend typecheck and production bundle
cd frontend
npm run build
```

Smoke coverage: regular/manual forward agreement and causal masks; seeded generation and telemetry; all three tokenizers; uint16 packing and uint32 fallback; portable checkpoint upload; prompt loss masking; five pretraining steps with accumulation and checkpoint reload; two SFT and two DPO steps; parameter updates; pause/resume/stop; NaN protection; offline corpus fallback; dataset splits; FastAPI endpoints, CORS, WebSocket, feedback CRUD/export, and invalid requests.

The implementation was checked with CPU PyTorch, Python 3.12, and Node 24. Real CUDA/FP16 and eight-replica TPU/BF16 tests are included for their respective notebooks and skipped without hardware. CUDA/MPS/XPU/TPU execution was not exercised on this CPU-only development host. The frontend production build passes. Headless Chromium checks exercised custom sizing, budget fitting, the oversized-model guard and creating a custom experiment; desktop/mobile screenshots were inspected. CPU tests also exercise TPU-path static batching, objectives, resume, retention and portable Studio export.

## Third-party assets

DM Sans and Space Grotesk are served from `frontend/public/fonts/` under the SIL Open Font License; license files are included. The original GPT-2 tokenizer files are distributed under the included MIT license, with canonical SHA-256 digests documented in `backend/assets/tokenizer/README.txt`. TinyShakespeare is obtained from Karpathy's public char-rnn dataset only when you click Download; the built-in public-domain Shakespeare excerpt is available offline.

Magic 8k tokenizer assets are bundled under Apache 2.0 with their pinned source revision and checksum. See `backend/assets/magic8k/`.
