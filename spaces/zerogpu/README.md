---
title: NanoForge ZeroGPU
emoji: ⚒️
colorFrom: green
colorTo: gray
sdk: gradio
sdk_version: 6.9.0
python_version: '3.12'
app_file: app.py
pinned: false
---

# NanoForge ZeroGPU Blackwell

Select **ZeroGPU** hardware in this Space's Settings. The platform currently supplies RTX PRO 6000 Blackwell allocations: `large` is 48 GB; `xlarge` is 96 GB and consumes twice the quota. Set the Space variable `NANOFORGE_ZERO_GPU_SIZE=xlarge` only if needed; default is `large`. A dedicated Google Cloud G4 VM is a different service.

This Gradio UI supports CPU model/dataset preparation, short resumable training chunks and generation. It uses the same dense/MoE model and portable `.pt` format as NanoForge Studio. All GPU work completes synchronously within the decorated Gradio handler; there is no background GPU trainer or Colab tunnel. CUDA is not probed or initialized by the app outside GPU handlers. The small dynamically configured training model is loaded per request, which has transfer overhead; this mutable-checkpoint workflow does not use module-scope weight prepacking.

The demo caps total parameters at 10M and GPU calls at 60 seconds, with a 35-second soft training budget and at most 25 updates. The soft limit is checked between updates; loading, one slow update or saving can still exhaust the scheduler's hard lease. If killed, resume from the previous completed checkpoint. Daily quota/queue/tier restrictions are enforced by Hugging Face and cannot be changed by this profile. Use dedicated GPU/Colab Studio for sustained or larger training. The last two committed checkpoint versions are retained per session; DPO starts a fresh frozen reference for each resumed chunk.

Prepare a session on CPU, then click **Train next chunk**. The direct-download picker offers CPU samples limited to 1,000 source rows from the Studio catalog; Dolly is the default. Cosmopedia 100M samples first download a 350 MB temporary shard to disk because of its large row group; other Parquet samples use bounded byte ranges. Samples must fit the 20 MiB preparation limit. Change architecture/tokenizer by creating a new session; an imported checkpoint retains its model settings. The dataset/mode and learning rate chosen at preparation are used for subsequent chunks. SFT requires normalized prompt/completion JSONL; DPO requires prompt/chosen/rejected JSONL. Dataset upload cap: 20 MiB. Checkpoint cap: 128 MiB. Preserve outputs before Space restart. A public Space is a public app; use a private Space for private model/data work.

This Space relies on platform-managed Gradio/Spaces/PyTorch. Do not run the Colab Blackwell wheel-repair installer here, pin a generic CUDA wheel over ZeroGPU, start FastAPI GPU background workers, or use torch.compile. Local CPU tests do not validate the live scheduler, its CUDA emulation or account quota.

Source: https://github.com/bananaosint/nanoforge
ZeroGPU runtime documentation: https://huggingface.co/docs/hub/spaces-zerogpu
