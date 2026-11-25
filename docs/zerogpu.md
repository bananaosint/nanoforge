# Hugging Face ZeroGPU Blackwell Space

NanoForge's separate Gradio adapter targets ZeroGPU's RTX PRO 6000 Blackwell allocation. Google Cloud G4 and ZeroGPU are different services. ZeroGPU lends a GPU only while a decorated Gradio event runs; the full Studio's continuous background trainer cannot run there unchanged.

The default profile is `configs/zerogpu.json`: Nano, Magic 8k, BF16, context 128, batch 2, 10 updates per click. CPU preparation builds packed uint16 pretraining caches. The UI exposes dense/MoE architecture, dimensions, tokenizer, objective, learning rate, batch, precision and chunk length. Imported checkpoints retain their architecture/tokenizer. The adapter uses the shared Trainer synchronously, with evaluation snapshots disabled during GPU leases, and exports the ordinary CPU FP32 `.pt` format for Studio.

## Deploy with one Python notebook cell

Run this in a notebook such as Colab, with **CPU runtime sufficient for deployment**. It prompts for Hugging Face login through the SDK; use an account/token authorized to write the target Space. Don't put a token into the cell or chat. The deployment uploads source/dependencies, not your local datasets, checkpoints, database or `.venv`. It updates matching source paths in the Space and leaves unrelated remote paths alone.

```python
import subprocess, sys, tempfile
from pathlib import Path

subprocess.check_call([
    sys.executable, "-m", "pip", "install", "--quiet", "huggingface_hub>=1,<2"
])
work = Path(tempfile.mkdtemp(prefix="nanoforge-space-"))
source, bundle = work / "source", work / "space"
subprocess.check_call([
    "git", "clone", "--depth", "1",
    "https://github.com/bananaosint/nanoforge.git", str(source)
])
subprocess.check_call([
    sys.executable, str(source / "scripts/export_zerogpu.py"),
    "--output", str(bundle)
])

from huggingface_hub import HfApi, SpaceHardware, login
login()  # Secure SDK login prompt; never hardcode the token.
api = HfApi()
space = "chaoticbanana/nanoforge"
api.create_repo(
    repo_id=space, repo_type="space", space_sdk="gradio",
    private=True, exist_ok=True
)
api.upload_folder(
    repo_id=space, repo_type="space", folder_path=str(bundle),
    commit_message="Deploy NanoForge ZeroGPU Blackwell UI"
)
# The API retains this historical identifier; it now means ZeroGPU Blackwell.
try:
    api.request_space_hardware(space, hardware=SpaceHardware.ZERO_A10G)
except Exception as exc:
    print("Source uploaded; enable ZeroGPU in Space Settings if eligible:", exc)
print(f"Open https://huggingface.co/spaces/{space}")
```

For an existing Space, it must use the **Gradio SDK**. `private=True` applies when creating a new Space; this cell does not change an existing Space's visibility. If the hardware request is unavailable for the account, open the Space's Settings and check ZeroGPU eligibility. No paid dedicated hardware is requested. Hosting/runtime dependencies are installed by the Space build from its `requirements.txt`; the platform's PyTorch/Gradio/Spaces stack is preserved. Don't run `start.sh`, the Colab wheel-repair installer or the full backend requirements inside this Space.

For a local export without uploading anything:

```bash
python scripts/export_zerogpu.py --output /tmp/nanoforge-space
```

The exporter requires a separate empty destination and includes bundled tokenizer assets. Commit/upload the contents of that directory at the Space repository root. The generated `README.md` sets Gradio 6.9.0 and Python 3.12.

## Use the UI

1. Upload a UTF-8 `.txt` / normalized `.jsonl`, or choose a source from **Direct dataset download** and click **Import dataset sample** for up to 1,000 source rows on CPU. The picker includes the Studio catalog (SmolTalk, Tulu and CodeAlpaca included); Dolly is the default. The prepared dataset must still fit the 20 MiB limit. The bundled starter corpus is available immediately.
2. Choose settings and click **Create / import session on CPU**. To continue a previous session after restart, upload its `.pt` and matching tokenizer. For SFT, JSONL needs `prompt`/`completion`; DPO needs `prompt`/`chosen`/`rejected`.
3. Click **Train next chunk**. The completed chunk reports absolute step, loss, precision and elapsed time; download its checkpoint. Repeated clicks resume model and optimizer from files shared between the short-lived GPU workers. Each DPO chunk begins a new pass with a fresh frozen reference, as a normal Studio checkpoint resume does.
4. Click **Generate with current checkpoint** for a separate short GPU request.

## Bounds and persistence

Default `large` allocation is 48 GB. Optional Space variable `NANOFORGE_ZERO_GPU_SIZE=xlarge` requests a full 96 GB card and consumes twice the quota; the default small model does not need it. Queues, daily quota and per-tier limits remain controlled by Hugging Face.

This adapter deliberately caps models at **10M total parameters**, context 256, batch 8 and 25 updates per request. Each GPU event requests 60 seconds; training allows a 35-second soft budget, checked between updates. Model loading, one slow update or saving can still exceed a hard lease, so completed checkpoints are the recovery boundary. A file lock serializes each session, GPU events share a single queue, and the checkpoint manifest changes only after a successful save. The last two committed checkpoint versions are retained per session; download older versions before continuing if you need them.

Dataset upload cap: 20 MiB; checkpoint cap: 128 MiB. Runtime files are temporary under `/tmp/nanoforge-zerogpu`; preserve downloads before the Space restarts or loses storage. Separate sessions consume separate storage. A public Space is a public app; use a private Space for private work. This adapter is for short experiments; use dedicated GPU/Colab Studio for continuous or larger training.

The model is dynamically loaded inside each GPU handler to support configurable mutable training sessions. That has transfer overhead compared with ZeroGPU's recommended module-level prepacked inference models. The app does not use `torch.compile`, load a GPU model in a background thread, or assume Python object mutations survive worker forks.

## Validation

`backend/tests/test_zerogpu.py` covers CPU preparation, uint16 caches, synchronous dense/MoE updates, SFT/DPO, successive file-based resumes, interrupted saves, zero work budget, session locks, bounds, checkpoint import into Studio and self-contained bundle export. When Gradio/Spaces are installed, it also builds the exported UI with CUDA probing blocked. A local Gradio API smoke check created a session and returned a downloadable checkpoint. The regular backend suite also passes. Live ZeroGPU CUDA execution, scheduler timeouts and account quotas require a deployed authenticated Space and were not verified on the CPU development host.

Runtime reference: https://huggingface.co/docs/hub/spaces-zerogpu
