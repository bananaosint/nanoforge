# Kaggle GPU T4 x2 — full Studio UI

Choose **GPU T4 x2** and enable **Internet** in Kaggle Notebook settings. Run the single cell in [`NanoForge_Kaggle_T4x2.ipynb`](../notebooks/NanoForge_Kaggle_T4x2.ipynb). It installs dependencies while preserving CUDA PyTorch, checks that exactly two T4 devices are visible, runs real two-GPU tests, builds Studio and prints a temporary browser link and random login password. Open the link in a new tab and use username `nanoforge` plus the printed password.

Kaggle does not use Colab's notebook port proxy. This launcher uses an account-free Cloudflare Quick Tunnel. The link is reachable over the Internet, but **all** Studio pages, assets, APIs and downloads require HTTP Basic authentication. A fresh 32-character random password is generated per launch, passed through the server environment rather than command arguments, and never stored in source or checkpoints. Keep the password and notebook output private. Cross-origin writes are blocked. The helper verifies that the public endpoint requires authentication before returning it. If the tunnel service or Kaggle Internet is unavailable, startup stops rather than exposing an unprotected alternative.

## Both GPUs, with settings in the UI

The app shows **KAGGLE STUDIO** and the detected devices. The defaults are Mini, Magic 8k, FP16 + GradScaler, context 256, global batch 32, accumulation 2 and **Use both GPUs** enabled. Effective batch is 64; with a full batch, each GPU processes approximately half the examples. T4 TF32 is off. Startup leaves training idle and imports no datasets automatically. In **Data & Mixer**, import sources and set weights; in **Model configuration**, choose presets or **Custom**, dimensions and parameter budget. Click **New experiment**, then **Start training**. Advanced settings include precision, accumulation, TF32 and the two-GPU checkbox.

The trainer uses PyTorch `DataParallel` with replica-local losses. Each replica returns only loss sums and target counts, avoiding gathering the full vocabulary logits on GPU 0. Losses are combined with their true counts, including uneven shards and masked completions; autograd reduces gradients to the primary model before clipping/AdamW. FP16 autocast is explicitly applied inside each replica thread. Pretraining, batched SFT and DPO all use both devices. Multi-GPU SFT uses a global token-weighted completion loss. DPO compares policy and frozen-reference completion log probabilities and holds an extra reference model on each device.

A batch of at least two is required when both GPUs are enabled. Smaller batches may favor one GPU because replication/communication have overhead; two GPUs are not a promise of 2× speed. This is data parallelism: **each GPU holds the full model**, so two 16 GB T4s do not become one 32 GB model-memory pool. GPU 0 also owns optimizer state. Choose a smaller batch/model if memory is exhausted. All model architectures remain below one billion parameters, subject to actual memory capacity.

Evaluation and interactive generation use the primary GPU. Model/optimizer exports use the canonical unwrapped model, with CPU FP32 tensors and no `module.` state-dict prefix. Studio `.pt` imports remain portable to a single GPU or CPU. The UI disables two-GPU training automatically when loading the checkpoint on a machine with fewer than two CUDA devices. For headless resume elsewhere, explicitly set `multi_gpu: false` in that config.

## Files, downloads and stop

Default data, caches, SQLite and checkpoints live in `/kaggle/working/nanoforge-run`. Use **Save checkpoint** / **Download .pt checkpoint** from the UI, or preserve files using Kaggle's output panel/version outputs. Kaggle session storage is temporary. Tunnel providers can limit large uploads/downloads; large existing checkpoints can also be attached as Kaggle input and copied with their JSON sidecars into the run's `checkpoints/` directory before launching.

The cell stores its owned processes in `kaggle_studio`:

```python
print(kaggle_studio.logs())  # Backend and tunnel logs
kaggle_studio.stop()         # Close tunnel and safely stop/save the trainer
```

Rerunning the cell stops only that earlier launcher instance. The server binds loopback and another occupied port is reported rather than killing unrelated processes. Cloudflared is downloaded from its official GitHub release and verified against the pinned official release SHA-256 digest; Node 22 is installed from the official verified bundle only when the installed Node is incompatible. No ngrok account/token is needed.

## Verification

CPU tests verify replica loss/gradient equivalence for unequal masked shards, DPO reference behavior, one-device errors, authentication across all routes, and cross-origin write protection. Existing CPU training/checkpoint/UI tests remain in the suite. `test_real_two_gpu_training_and_portable_export` requires two CUDA GPUs and runs from the Kaggle launcher: it exercises pretraining/SFT/DPO, checks that both replica devices execute, verifies finite changed weights and canonical CPU-only checkpoint tensors, then loads/generates/resumes in the CPU Studio. Actual dual-T4 execution, memory usage and throughput cannot be measured on the CPU-only development host; the runtime test is required before the launcher opens the UI.
