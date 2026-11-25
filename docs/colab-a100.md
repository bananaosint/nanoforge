# NanoForge Studio on a Colab A100

[Open the A100 notebook](https://colab.research.google.com/github/bananaosint/nanoforge/blob/main/notebooks/NanoForge_A100.ipynb), choose **Runtime → Change runtime type → A100 GPU**, and run its launcher cell. GPU allocation depends on your Colab account. Stop an earlier training/Studio cell first. The launcher installs backend requirements, preserves compatible Colab CUDA PyTorch, installs/reuses compatible Node, builds the frontend, verifies an actual A100 and runs the CUDA checkpoint smoke tests. It then serves the UI below the cell using Colab's notebook proxy. No public tunnel or tunnel account is needed.

## Settings are in the UI

`configs/colab-a100.json` defaults to Mini, Magic 8k, **FP16** autocast + GradScaler, TF32, context 512, global batch 64, accumulation 1, learning rate 0.0003, 1,000 updates and evaluation every 250 updates. The effective batch is 64. The profile keeps the previously requested FP16 compute; model/master optimizer state and exported weights remain FP32. TF32 applies to supported remaining FP32 CUDA operations. PyTorch chooses a supported SDPA implementation without forcing a particular attention kernel.

Startup leaves the trainer idle and imports no corpora. Dolly is selected if already present; otherwise the included starter sample is selected. Use **Data & Mixer** to import sources and assign weights. Choose **Custom** for layers, heads, width, context and parameter-budget fitting. Change batch, steps, learning rate, precision, accumulation and TF32 in the UI. Apply architecture/tokenizer changes with **New experiment**, then click **Start training**. The A100 profile uses one GPU by default.

The batch 64 default is for Mini. For a larger Custom architecture, reduce batch size to 16, 8 or 1 and adjust accumulation. Every model must stay below one billion parameters, and must also fit actual GPU memory. The launcher prints the assigned A100's actual memory; it does not assume a 40 GB versus 80 GB card. The profile has not been throughput/memory benchmarked on physical A100 hardware in development.

Token caches remain memory-mapped uint16 for Magic 8k/GPT-2, with uint32 fallback for wider vocabularies. Only sampled batches expand to int64. Use **Save checkpoint** / **Download .pt checkpoint** to export the canonical model and optimizer without GPU-specific wrappers. Your desktop Studio can load the `.pt` via **Import .pt checkpoint**, including on CPU. Choose local dataset files before continued training.

## Launcher and persistence

The existing helper accepts an explicit profile:

```python
nanoforge_studio = startup['launch_studio'](repo=REPO, profile='colab-a100')
startup['show_studio'](nanoforge_studio)
```

It defaults to `/content/nanoforge-a100-run`, keeping the A100 run separate from earlier L4 runs. To reuse an earlier run's files, pass `home='/content/nanoforge-l4-run'`, or choose an already mounted Drive directory. Then select a checkpoint explicitly in the UI. Rerunning the notebook cell safely stops only its previous owned Studio process before restarting.

For diagnostics, use `print(nanoforge_studio.logs())`; to stop safely, use `nanoforge_studio.stop()`. Keep the runtime connected and preserve checkpoints before ending it: `/content` is temporary. Large browser/proxy downloads can also be retrieved from the Colab Files panel. Feedback/history and datasets are stored in the runtime and are not embedded in `.pt`.

## Verification

CPU tests validate the profile and idle UI defaults, A100 notebook syntax and launcher selection, and rejection of a mismatched GPU name. Existing training, checkpoint portability, L4, Kaggle and UI tests remain in the suite. Actual A100 kernels and Colab proxy authentication require the live runtime. The launcher runs the real CUDA FP16/TF32 smoke tests on your assigned GPU before showing the UI, including CPU generation and optimizer resume from its exported checkpoint.
