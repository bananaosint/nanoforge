# Intel XPU support

The Intel Core 3 100U integrated GPU is outside PyTorch's officially validated XPU client hardware list. This laptop should keep the CPU build. Software detection alone does not make an unsupported GPU compatible. Native Intel XPU execution was not tested in the development workspace, which has no Intel GPU.

Official hardware and driver guidance:
https://docs.pytorch.org/docs/stable/notes/get_start_xpu.html
https://www.intel.com/content/www/us/en/developer/articles/tool/pytorch-prerequisites-for-intel-gpu.html

For a compatible Intel GPU with working Windows/WSL drivers only, the explicit setup option is:

```bash
NANOFORGE_TORCH=xpu ./start.sh
```

This replaces an existing CPU PyTorch wheel with the official Intel XPU wheel in the project's .venv, keeps temporary downloads on the project disk, and runs a real tiny-Transformer backward/optimizer/generation test on the GPU before launching. It does not install or change graphics drivers. A missing XPU produces an error instead of silently falling back to CPU. It can be a substantial download.

Normal startup still preserves a working existing PyTorch build. Device auto-selection is CUDA, XPU, MPS, CPU. Override it with NANOFORGE_DEVICE=cpu/cuda/xpu/mps or auto. An explicit unavailable accelerator raises an actionable error. All model parameters, training batches and inference tensors use the selected device; the UI badge displays XPU and its tooltip gives the GPU name. Checkpoint weights and optimizer state load onto the selected device.

To test a compatible installed setup without launching servers:

```bash
.venv/bin/python -m backend.check_device --device xpu
```

CPU validation remains available:

```bash
.venv/bin/python -m backend.check_device --device cpu
```

The new device selection and installer paths are covered by automated tests; actual GPU execution still requires compatible hardware and drivers on the user's computer.
