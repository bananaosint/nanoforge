# Colab L4: one cell to train NanoForge

[Open the L4 notebook in Colab](https://colab.research.google.com/github/bananaosint/nanoforge/blob/main/notebooks/NanoForge_L4.ipynb). Choose **Runtime → Change runtime type → L4 GPU**, then run its single code cell. It clones/updates the repository, loads `scripts/colab_l4.py` into the notebook kernel, and runs installation, hardware checks, data preparation, training and download. Training itself runs in a fresh CUDA subprocess. No Kaggle account, TPU/XLA package, frontend server or Node installation is needed.

## L4 profile

`configs/colab-l4.json` uses Mini, Magic 8k, **FP16** autocast with GradScaler, context **512**, batch **32**, accumulation **2**, learning rate 0.0003, 1,000 additional updates and evaluation/checkpoints every 250 updates. Effective batch is 64. Fused CUDA AdamW and pinned/nonblocking batch transfers are shared with the existing trainer. PyTorch selects an SDPA implementation supported by the actual CUDA stack; no particular attention kernel is forced. These defaults are a practical starting point, not a measured throughput or maximum-memory claim.

L4 supports TF32, enabled in this profile by `allow_tf32: true`. The trainer applies that flag inside the actual training subprocess for CUDA matmul/cuDNN FP32 operations on compute capability >=8. FP16 remains the primary compute precision; TF32 is not another model tensor/storage dtype. Other profiles default to TF32 off. On CPU/MPS/XPU the flag is ignored. T4 does not enable it. Model master weights, AdamW state and exported weights remain FP32.

The startup helper preserves a compatible CUDA PyTorch already installed in Colab and installs the app dependencies. It does not install CPU PyTorch or Torch/XLA. A GPU preflight requires an actual L4 and prints its name, memory, PyTorch and CUDA versions. Selecting a CPU/T4/TPU runtime stops with a clear message. Tests then perform real FP16 updates with TF32 off/on, export CPU tensors, register the checkpoint in Studio, generate on CPU and resume an optimizer update there.

## Settings and custom size

Edit the top of the notebook cell:

```python
TRAINING_STEPS = 1000
BATCH_SIZE = 32
GRADIENT_ACCUMULATION = 2
CUSTOM_MODEL = None  # Uses Mini from the L4 profile
RESUME = None
RUN_HOME = '/content/nanoforge-l4-run'
```

For example, set `CUSTOM_MODEL = dict(n_layer=12, n_head=8, n_embd=768, block_size=512, dropout=0.1)` and begin with a smaller batch. Width must be divisible by heads, and all architectures must have fewer than one billion parameters. The default batch 32 is for Mini, not an arbitrary larger model. The pre-training size preview allocates no model tensors and prints weight/optimizer-memory estimates; activations and temporary buffers need more GPU memory. If you run out of GPU memory, reduce `BATCH_SIZE` to 16, 8 or 1 and increase accumulation as needed. Refer to [custom sizing](../README.md#custom-model-size-below-1-billion-parameters).

The helper imports **Dolly only** and pre-tokenizes it with Magic 8k. Caches store uint16 IDs and are memory-mapped. Training expands only sampled batches to int64 tensors. This example initializes random weights; it does not download pretrained weights or attempt all the large catalog sources. To choose other datasets or SFT/DPO, edit a config and use the ordinary headless CLI. Character vocabularies above 65,536 use uint32 instead of overflowing.

Set `RESUME` to a complete NanoForge Magic 8k checkpoint to continue a saved model. The checkpoint's architecture and tokenizer are retained; `TRAINING_STEPS` means additional optimizer updates. Local dataset files must be available. The CLI rejects a tokenizer mismatch. New architecture settings apply only to a new run.

## Manual CLI and output

From the repository root after preparing data:

```bash
python -m backend.train_cli --config configs/colab-l4.json \
  --home /content/nanoforge-l4-run --device cuda --preview

python -m backend.train_cli --config configs/colab-l4.json \
  --home /content/nanoforge-l4-run --device cuda
```

`--preview` prints exact size without model allocation. It sizes new models and cannot be combined with `--resume`. After a completed training run, `latest-checkpoint.json` is atomically written with the exact checkpoint and JSON metadata paths, state and step. The startup helper uses this manifest rather than guessing which file is newest. Training errors stop the cell; an interrupted CLI exits with code 130 after attempting a safe save, so the helper cannot mistake an old manifest for the new run's success.

The cell downloads the `.pt`. In the updated Studio choose **Train & Monitor → Import .pt checkpoint**. The checkpoint embeds architecture, exact tokenizer, model/optimizer state, step and scaler metadata. Its tensors are on CPU; model tensors are FP32. The Studio needs no CUDA to deserialize it and switches compute precision for its own device. Select local datasets before continuing. Alternatively copy both `.pt` and `.json` into local `checkpoints/` with Studio stopped. The upload limit is 16 GiB.

No ZIP is made automatically, avoiding another large copy. If the browser blocks the download or the file is too large, use Colab's Files panel or an already mounted Drive run directory. `/content` is ephemeral: preserve checkpoints before the runtime ends. Interval checkpoint files accumulate; allow adequate disk space for large runs and remove unwanted old versions yourself. Dataset files and SQLite history are not embedded in `.pt`.

## Validation

CPU tests cover the new profile/custom settings, schema validation, CLI size preview, the completed-run manifest and exporting/importing the resulting checkpoint into Studio. Notebook code is parsed with empty saved outputs. The real CUDA/FP16 smoke tests run in the L4 startup helper and skip on the CPU-only development host. Actual L4 kernels, memory usage and throughput could not be measured here; the hardware test in your Colab session is the required live verification.
