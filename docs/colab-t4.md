# Train on a Colab Tesla T4 and return the model to Studio

Open `notebooks/NanoForge_T4.ipynb` in Colab and choose **Runtime → Change runtime type → T4 GPU**. The notebook installs dependencies, checks the actual CUDA device, runs the GPU smoke test, imports Dolly, pre-tokenizes with Magic 8k, trains, and downloads a portable checkpoint ZIP. It uses the same PyTorch architecture and background trainer as the local studio.

[Open in Colab](https://colab.research.google.com/github/bananaosint/nanoforge/blob/main/notebooks/NanoForge_T4.ipynb)

## T4 profile

`configs/colab-t4.json` selects Mini, the 8,192-token Magic BPE tokenizer, a 256-token context, batch 32, accumulation 2, learning rate 0.0003, 1,000 additional steps, and evaluation every 250 steps. Effective batch size is 64. Reduce batch size to 16 or 8 if other work uses GPU memory. This profile is a practical starting point, not a claim of a measured maximum throughput.

CUDA training uses **FP16 autocast** for eligible operations and **GradScaler**. Parameters/master optimizer state remain FP32 for stability. Gradients are unscaled before clipping; overflowed updates are skipped with scale adjustment and do not count as optimizer steps. Persistent overflows or non-finite loss stop with an error. DPO completion log-probability reductions and generation probability calculations explicitly use FP32. CUDA uses fused AdamW and pinned/nonblocking transfers for the small sampled batches.

T4 does not support BF16 or TF32 acceleration. The notebook uses FP16, and PyTorch chooses an attention implementation supported by the installed CUDA stack; FlashAttention is not forced on the T4. CPU/MPS/XPU automatic precision remains FP32. Explicit `float16` training on a non-CUDA device is rejected rather than silently changing its meaning.

## Pretokenization

```bash
python -m backend.pretokenize --home /content/nanoforge-run --tokenizer magic8k --dataset dolly.jsonl
```

Caches use unsigned **uint16** when the vocabulary fits IDs 0–65,535. This covers Magic 8k and GPT-2 BPE. Larger character vocabularies fall back to uint32, protecting against ID overflow. Each sampled batch is expanded to int64 for PyTorch embeddings/cross-entropy. Caches are memory-mapped; the full token array is not copied to RAM or GPU. `.u16` caches use two bytes per token. `.u32` caches use four. Older uint32 caches remain separate and can be removed with the app stopped to reclaim space.

The pretokenizer and trainer use identical source/tokenizer fingerprints, so training reuses prepared caches. A tokenizer change requires different caches and a new model, not relabeling an existing checkpoint's embeddings.

## Headless training and resume

```bash
python -m backend.train_cli --home /content/nanoforge-run --config configs/colab-t4.json --device cuda

python -m backend.train_cli --home /content/nanoforge-run --config configs/colab-t4.json --device cuda --resume /content/nanoforge-run/checkpoints/YOUR_CHECKPOINT.pt
```

`max_steps` is the number of additional updates on resume. The checkpoint retains its architecture and tokenizer. Tokenizer mismatches are rejected; resume preserves checkpoint architecture and ignores new-model architecture overrides. Imported prompt/completion pairs also support `mode: "sft"`; pretraining from scratch first is recommended for a new model.

The notebook imports a small example corpus; it does not attempt a 122 GB Cosmopedia download on Colab's ordinary temporary disk. `/content` is ephemeral. Download outputs before disconnecting, or use the optional Google Drive run folder shown in the notebook. Drive can be slower for random cache reads.

## Studio-compatible .pt outputs

Checkpoints retain the original required fields: `model_config`, `tokenizer_state`, `model_state_dict`, `optimizer_state_dict`, `step`, `val_loss`, and `version_tag`. Extra backward-compatible fields record training config, precision, scaler state, PyTorch version and format version. The exact Magic tokenizer JSON is embedded in the checkpoint.

Model tensors are exported as **CPU FP32 tensors** and optimizer tensors are copied to CPU. Tied token/output embedding storage is preserved. The file does not require CUDA to deserialize. FP16 describes the computation used to train; it does not force CPU inference to perform unsupported half-precision operations. Compatible saved optimizer settings are reset for the destination device on resume.

In Studio, use **Train & Monitor → Import .pt from Colab**. The backend streams uploads up to 2 GiB, uses `torch.load(..., map_location="cpu", weights_only=True)`, checks model/tokenizer consistency and tensor shapes, and reconstructs the JSON listing metadata. The frontend then selects the imported model. Existing filenames are not overwritten. A bare state dictionary is not a complete NanoForge checkpoint.

Alternatively, copy both the notebook ZIP's `.pt` and `.json` into your local `checkpoints/` directory with the app stopped. After loading, choose locally available training datasets before continuing. Model/optimizer state is portable; dataset files and the separate SQLite timeline history are not embedded in `.pt`.

## Verification limits

CPU training, uint16 caches, exact tokenizer round-trip, API upload, legacy checkpoint loading, CUDA-style optimizer-flag reset and headless CLI workflows were exercised locally. An actual CUDA/FP16 smoke test is included and run by the notebook; it is skipped on the CPU-only development machine. This environment could not measure T4 performance or verify its CUDA kernels. The notebook's first GPU test is the hardware verification step on your Colab runtime.
