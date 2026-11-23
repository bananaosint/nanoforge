# Kaggle TPU x8 training

Import [`notebooks/NanoForge_Kaggle_TPU.ipynb`](../notebooks/NanoForge_Kaggle_TPU.ipynb) into Kaggle using its GitHub URL, or upload that notebook. Enable **Internet** and choose **TPU x8** in Notebook settings, then run the cells in order. TPU access and quotas depend on your Kaggle account. The notebook detects the actual replica count and requires eight; it does not assume a specific TPU generation.

GitHub notebook URL:

```
https://github.com/bananaosint/nanoforge/blob/main/notebooks/NanoForge_Kaggle_TPU.ipynb
```

## Setup and launch

The TPU path is `backend.train_xla`, a fresh-process CLI separate from the local FastAPI background trainer. It uses PJRT and `torch_xla.launch` to discover and launch all available TPU devices. Do not initialize an XLA device or TensorFlow TPU strategy in the notebook/launcher before spawning. If another cell already claimed the TPU runtime, restart the session. `--expected-replicas 8` fails instead of silently training on one core or falling back to CPU.

The notebook preserves installed matching PyTorch / PyTorch-XLA major/minor versions >=2.8 and <3. Otherwise it installs a documented pair. To install that pair explicitly in a fresh Kaggle session:

```python
import subprocess, sys
subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-cache-dir', '--upgrade',
                'torch==2.8.0', '--index-url', 'https://download.pytorch.org/whl/cpu'], check=True)
subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-cache-dir', '--upgrade',
                'torch_xla[tpu]==2.8.0', '-f', 'https://storage.googleapis.com/libtpu-releases/index.html'], check=True)
subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', 'backend/requirements.txt'], check=True)
```

No CUDA PyTorch or Intel extension is required. Torch/XLA and libtpu are optional dependencies used only in the Kaggle process; local Studio installations do not install them. ABI/undefined-symbol errors normally indicate mismatched Torch/XLA binaries. TensorFlow TPU runtime/library conflicts may require a fresh notebook session without TensorFlow initialization.

From the repository root in a fresh process after preparing the selected data:

```bash
PJRT_DEVICE=TPU python -m backend.train_xla \
  --config configs/kaggle-tpu.json \
  --home /kaggle/working/nanoforge-run \
  --expected-replicas 8 --keep-last 1
```

The default config imports no data automatically; the notebook imports Dolly and pre-tokenizes it first. Custom datasets must be in `RUN_HOME/data`. `.txt` and normalized `.jsonl` formats are the same as Studio. `hitl` reads feedback from `RUN_HOME/nanoforge.sqlite3` if present. The CLI prepares shared token caches/pair indexes in the parent before workers start, preventing concurrent partial-file writes.

## Precision, sizes and eight-core behavior

The profile uses Mini, Magic 8k, context 256, **BF16 autocast**, batch **4 per core**, accumulation **2**, and 1,000 updates. Effective batch is `batch_per_core × accumulation × replicas`, or 64 on eight replicas. This is an initial profile, not a hardware benchmark. CUDA FP16 is rejected; `auto` selects BF16 and `float32` is also available. BF16 needs no GradScaler. Legacy `XLA_USE_BF16` / `XLA_DOWNCAST_BF16` global downcasting flags must be unset: master weights and optimizer state stay FP32.

All replicas start with the same weights. Initialization protects the CPU RNG used by TPU v2/v3 device threads, followed by an explicit master-parameter broadcast. Each replica has its own CPU sampling RNG and XLA dropout seed. Gradients are averaged with `xm.reduce_gradients` **before** norm clipping; XLA's sync-free AdamW applies the same update on each core. A globally detected non-finite loss/gradient skips the update and stops training. Shapes stay fixed: pretraining uses a common source/split length, and completion batches pad the tail with ignored `-100` targets. Attention uses explicit causal math on XLA, avoiding CUDA-only fused kernels; softmax/loss reductions use FP32. Initial XLA compilation can take minutes.

Pretraining, SFT (including optional base-data mixing), and DPO are supported. SFT is token-weighted over nonmasked completion targets. DPO sums completion log probabilities, disables policy/reference dropout, and holds a frozen reference for the pass. Choose a selected preference dataset or saved HITL preferences for DPO.

Choose `preset: "Custom"` with explicit dimensions to train larger models. The [custom sizing guide](../README.md#custom-model-size-below-1-billion-parameters) describes exact parameter counting and budget fitting. Every model must have fewer than one billion parameters. **Data parallelism replicates the entire model and AdamW state on each core; it does not pool eight cores' RAM.** Weights, gradients and AdamW state alone use about 16 bytes per parameter per core, plus activations/temporary buffers. DPO adds a reference model. Start small and reduce per-core batch size for larger models. This backend does not implement FSDP/model sharding.

Caches remain memory-mapped uint16 when vocabulary IDs fit (Magic 8k and GPT-2 do); wider vocabularies use uint32. Only small batches expand to int64. TPU throughput logs count scheduled input token slots, including padded prompt/completion positions. These logs are in `RUN_HOME/tpu-metrics.jsonl`, separate from Studio's SQLite timeline.

## Resume, storage and Studio compatibility

```bash
PJRT_DEVICE=TPU python -m backend.train_xla \
  --config configs/kaggle-tpu.json --home /kaggle/working/nanoforge-run \
  --expected-replicas 8 --resume /kaggle/input/my-checkpoint/model.pt
```

Resume retains checkpoint architecture/tokenizer and optimizer history; `max_steps` adds updates. It is not an exact replay of the previous stochastic sample/dropout sequence. Keep the tokenizer name in the config consistent with the checkpoint. Fresh runs always initialize random weights. `--single-core --expected-replicas 1` is available for troubleshooting.

Only global rank zero writes checkpoints, with barriers around saving. Each file retains the NanoForge required fields and exact tokenizer JSON; model and optimizer tensors are moved to CPU, model tensors are FP32, and tied embeddings share serialized storage. No XLA objects or libtpu dependencies are required to load it locally. `latest-checkpoint.json` points to the latest fully saved `.pt`/`.json` pair. By default `--keep-last 1` prunes older completed checkpoints **from this invocation only**, after a newer one is safely saved. `--keep-last 0` retains all versions; earlier invocations are preserved. Allow disk space for old and temporary new files during an atomic save.

Kaggle may terminate an interrupted distributed job immediately. Resume the most recent completed checkpoint; an unfinished `.pending` file is not a checkpoint. Download/preserve outputs before the session ends, or save a Kaggle notebook version with outputs. Full Cosmopedia/Wikipedia downloads are unsuitable for typical notebook disk quotas.

In the updated Studio choose **Train & Monitor → Import .pt checkpoint** (up to 16 GiB), or copy both `.pt` and `.json` to local `checkpoints/` with Studio stopped. The Studio switches back to local compute precision; choose local datasets before continued training. The checkpoint does not embed corpora or the Kaggle metrics history. To resume headlessly with the ordinary CPU/GPU CLI, change the training config's precision from `bfloat16` to `auto` or `float32`.

## Validation and limits

CPU tests exercise the same fixed-shape batches, masked SFT/DPO objectives, parameter counting, cache preparation, resume, export and CPU Studio loading. They do not execute XLA kernels or collectives. `backend/tests/test_tpu_hardware.py` is opt-in with `NANOFORGE_TEST_TPU=1`; the notebook runs it in a subprocess before training. It requires eight real replicas, performs two BF16 updates, verifies changed/finite weights and CPU-only serialization, then imports the checkpoint, generates and resumes one update through the CPU Studio backend. It is skipped on the CPU-only development host. Actual Kaggle/TPU performance and package/runtime compatibility remain subject to that hardware test.

The integration follows upstream [PJRT multiprocessing](https://docs.pytorch.org/xla/release/r2.8/learn/pjrt.html), [TPU automatic mixed precision](https://docs.pytorch.org/xla/release/r2.8/perf/amp.html), and [`reduce_gradients` / master parameter broadcast APIs](https://github.com/pytorch/xla/blob/r2.8/torch_xla/core/xla_model.py).
