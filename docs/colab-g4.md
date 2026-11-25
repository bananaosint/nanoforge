# Colab G4 Blackwell Studio

[Open the G4 notebook](https://colab.research.google.com/github/bananaosint/nanoforge/blob/main/notebooks/NanoForge_G4.ipynb). Select the **G4 / NVIDIA RTX PRO 6000 Blackwell** accelerator in Colab, then run its launcher cell. GPU availability depends on your account. Startup installs dependencies and opens the full Studio below the cell through Colab's notebook proxy; it stays idle and imports no datasets automatically.

## Checked CUDA setup

The launcher verifies an RTX PRO 6000 Blackwell GPU with compute capability 12.x, prints actual memory/driver-stack information, and runs CUDA forward/backward, BF16 matmul, causal SDPA and fused AdamW kernels in a fresh subprocess. Device visibility alone is not considered a successful check.

A working preinstalled Blackwell-compatible PyTorch/CUDA stack is preserved. If the software probe fails, the launcher makes **one** repair attempt using the official `torch==2.10.0` CUDA 12.8 wheel from `https://download.pytorch.org/whl/cu128`, then repeats the probe. The pinned Linux/Python 3.12 wheel was verified in the official index. An absent/wrong GPU is rejected without installing replacement Torch wheels. If kernel/driver checks still fail, startup stops with logs rather than falling back to CPU. Drivers remain managed by Colab; changing pip packages cannot repair an incompatible host driver. Use a fresh runtime if package repair leaves notebook-imported libraries stale.

After the kernel probe, the launcher runs actual dense FP16/TF32 and MoE FP16/BF16 training/export tests on the assigned GPU before serving the UI. These verify CPU-only checkpoint tensors, CPU generation and optimizer resume. It then builds the frontend using compatible Node/npm (an official checksummed Node bundle is installed only if needed). No public tunnel or tunnel account is required.

## Profile and MoE settings

`configs/colab-g4.json` starts with **Dense Mini**, Magic 8k, **CUDA BF16**, TF32, context 512, batch 32, accumulation 2, learning rate 0.0003 and 1,000 steps. Effective batch is 64. These are conservative initial values, not a measured maximum-throughput profile for a full 96 GB GPU. Actual memory is read from the assigned device, including a smaller allocation if the provider partitions it.

BF16 runs eligible operations in bfloat16 with FP32 master parameters/AdamW state. GradScaler is used for FP16; BF16 does not need it. FP16 remains available in the UI. Native BF16 capability is checked before enabling it; CPU/MPS/XPU should use auto/float32. Existing CUDA/T4 FP16 behavior is unchanged.

All training/model settings remain in the UI. Select **Architecture → Mixture of Experts (MoE)** to enable top-k experts and router losses; see [MoE architecture and limitations](moe.md). Dense remains available and is the default. The size limit includes every expert/router parameter, not just active experts. Use larger batches or models only after measuring memory and throughput; a sub-billion parameter count does not guarantee any context/batch will fit.

Use **Data & Mixer** to import corpora and assign weights. The profile selects Dolly if already present, otherwise the included starter sample. Token caches remain memory-mapped uint16 when IDs fit (Magic 8k/GPT-2), with uint32 fallback. Training transfers only sampled batches as int64. Fast local data/cache reads matter for GPU utilization; a small Mini model and slow preprocessing can leave a large GPU underused.

## Files and session control

The default run directory is `/content/nanoforge-g4-run`. Pass `home=` to an already mounted Drive folder to retain files across runtime replacement, or reuse an earlier run directory and explicitly select its checkpoint in the UI. Changing a dense model into MoE requires a fresh model; resuming keeps the saved architecture/tokenizer/experts.

Use **Save checkpoint** and **Download .pt checkpoint**. Model/router/expert weights are exported as CPU FP32, and the updated desktop Studio validates and loads them without CUDA. Dense exports remain compatible with the existing dense checkpoint layout. Datasets and feedback/history are stored separately. `/content` is temporary: preserve outputs before ending the runtime. Large downloads can also be retrieved from the Colab Files panel.

`print(nanoforge_studio.logs())` displays backend diagnostics. `nanoforge_studio.stop()` requests a safe stop/checkpoint. Rerunning the cell stops only its earlier owned server. The UI uses HTTP polling through the notebook proxy and clearly identifies the Colab G4 backend. No Blackwell kernels or Colab authentication were executed in the development environment; the live startup tests are the hardware verification step.
