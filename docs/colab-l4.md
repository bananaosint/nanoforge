# NanoForge Studio UI on a Colab L4

[Open the L4 Studio notebook](https://colab.research.google.com/github/bananaosint/nanoforge/blob/main/notebooks/NanoForge_L4.ipynb). Select **Runtime → Change runtime type → L4 GPU**, stop any earlier headless training cell, and run the notebook's launcher cell. The full UI appears **below the cell** through Colab's notebook proxy. The launcher does not start a training run or import corpora; choose those in the UI.

## All model and training settings are in the UI

The Studio defaults to the L4 profile: Mini, Magic 8k, FP16, context 512, batch 32, accumulation 2, TF32 enabled, and 1,000 training steps. If Dolly already exists it is selected; otherwise the included starter sample is selected. These defaults apply only to a fresh Studio and can all be changed before creating a model. The header shows **COLAB STUDIO**, the connection and the detected device, so you can tell this backend apart from your local desktop Studio.

- **Model configuration:** choose Nano/Micro/Mini or **Custom**. Custom exposes layers, heads, width, context, dropout, a parameter budget and the exact parameter count/memory estimate. Every model stays below one billion parameters. Dimensions are discrete and width must divide evenly into heads.
- **Data & Mixer:** import sources from the existing Hugging Face catalog, upload UTF-8 text/JSONL, preview files and set weights. Imports run in the Colab runtime, so large source files do not need to pass through your browser. The importer still reports upstream access errors when a source is unavailable.
- **Training:** change learning rate, batch, steps and evaluation interval. **Advanced hyperparameters** contains accumulation, compute precision and TF32. Use **New experiment** to apply architecture/tokenizer changes, then **Start training**. Pause/resume/stop remain available. Imports/tokenization and training use the same backend as the local app; token caches use uint16 when IDs fit and uint32 otherwise.
- **Studio/Inspector:** generate, collect SFT/DPO feedback and inspect attention/token candidates using the active Colab model.
- **Checkpoints:** select/upload a checkpoint, save manually while idle, or choose **Download .pt checkpoint** after saving/training. The `.pt` retains tokenizer, architecture, CPU FP32 weights and optimizer state. Import it in your desktop Studio using **Import .pt checkpoint**.

The default batch is intended for Mini. For a larger Custom model, reduce batch size, often to 1, and increase accumulation as needed. A sub-billion parameter count does not guarantee the model fits L4 memory. Model/master optimizer weights remain FP32; FP16 is compute precision. TF32 affects supported remaining FP32 CUDA matmuls and is ignored on unsupported devices. PyTorch selects a supported SDPA kernel. These defaults are an initial profile, not a measured maximum throughput claim.

## Launcher and connection

The launcher loads `scripts/colab_ui.py`. It preserves compatible CUDA PyTorch, installs Python dependencies, verifies an actual L4, runs the CUDA FP16/TF32 checkpoint smoke tests, builds the React app, and starts a **single** FastAPI process on `127.0.0.1`. FastAPI serves both the production frontend and its API. A compatible installed Node is reused; otherwise the helper downloads an official Node 22 Linux bundle with SHA-256 verification into the run directory's install cache.

`google.colab.output.serve_kernel_port_as_iframe` displays the app through the notebook's authenticated proxy. No public tunnel/account/key is used. The deprecated separate-window Colab helper is not used. Frontend assets, API requests and downloads support proxy path prefixes. Colab sessions use one-second HTTP status polling, so live plots/controls do not depend on WebSocket proxy support. Local desktop sessions retain WebSockets with HTTP fallback.

The embedded app operates on the Colab backend and its GPU/files. It does not attach your separately running desktop UI to Colab. Keep the notebook/runtime connected while using the app; normal Colab allocation and idle policies apply.

## Session control and storage

The cell keeps its owned server in `nanoforge_studio`. For diagnostics:

```python
print(nanoforge_studio.logs())
```

To shut down safely:

```python
nanoforge_studio.stop()
```

Shutdown requests the normal backend stop, waits for the current step/checkpoint and closes the process. Rerunning the launcher cell stops only that previous server before rebuilding; it does not kill unrelated processes. If shutdown needs more time, wait for the checkpoint and call `stop()` again. An occupied port is reported instead of silently reusing or killing another server.

By default, data, feedback, caches, SQLite and checkpoints live in `/content/nanoforge-l4-run`; model files are in its `checkpoints/` directory. Earlier headless runs using that directory remain visible through the selectors. Startup leaves the trainer idle: choose a previous checkpoint explicitly to resume it.

`/content` is temporary. Download/preserve checkpoints before disconnecting. For Drive persistence, mount Drive first and pass an already mounted folder as `home=` to `launch_studio`; model hyperparameters still stay in the UI. Drive random cache reads can be slower. Large proxy/browser downloads may be limited; Colab's Files panel or Drive provides an alternative. JSON listing metadata can be rebuilt by the desktop `.pt` import control. Dataset files/history are not embedded in `.pt`.

## Validation limits

CPU integration tests verify production UI serving, L4 defaults, idle startup, API training controls, checkpoint downloads, safe owned-process start/stop/restart, the iframe helper contract, and the Node download checksum/extraction path. Headless Chromium exercises the UI behind a reverse-proxy path prefix, including custom model settings, upload/mixer, training/polling, checkpoint save/download and generation. Actual Colab proxy authentication and L4 kernels cannot be executed in this development environment. The launcher runs real CUDA smoke tests on your assigned GPU before serving the UI.

## Optional headless workflow

The previous auto-training cell remains in [NanoForge_L4_Headless.ipynb](../notebooks/NanoForge_L4_Headless.ipynb), with [headless CLI/setup documentation](colab-l4-headless.md). The default L4 notebook now launches the UI instead.
