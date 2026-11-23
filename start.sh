#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
PYTHON_BIN="${PYTHON_BIN:-python3}"
if ! command -v "$PYTHON_BIN" >/dev/null; then
  echo 'Python 3.10+ is required. Install Python, then run ./start.sh again.' >&2
  exit 1
fi
if ! command -v npm >/dev/null; then
  echo 'Node.js 20.19+ (or 22.12+) and npm are required.' >&2
  exit 1
fi
NANOFORGE_TORCH="${NANOFORGE_TORCH:-cpu}"
case "$NANOFORGE_TORCH" in
  cpu|cuda|xpu) ;;
  *) echo 'NANOFORGE_TORCH must be cpu, cuda, or xpu.' >&2; exit 1 ;;
esac
if [ ! -d .venv ]; then "$PYTHON_BIN" -m venv .venv; fi
# WSL may mount /tmp as a small RAM-backed filesystem. Keep wheel downloads and
# unpacking on the same filesystem as the project instead. No global changes.
mkdir -p .venv/install-tmp
export TMPDIR="$PWD/.venv/install-tmp"
export PIP_NO_CACHE_DIR=1

# XPU is an explicit opt-in, including for an environment already marked ready.
# Force replacement is required when an existing CPU wheel has a newer version.
if [ "$NANOFORGE_TORCH" = 'xpu' ]; then
  export NANOFORGE_DEVICE=xpu
  if ! .venv/bin/python -c "import torch; assert '+xpu' in torch.__version__.lower()" >/dev/null 2>&1; then
    echo 'Installing Intel XPU PyTorch into this project virtual environment.'
    .venv/bin/python -m pip install --no-cache-dir --force-reinstall 'torch>=2.6,<3' --index-url https://download.pytorch.org/whl/xpu
  fi
fi

# Only install dependencies on first launch or when requirements change.
if [ ! -f .venv/.nanoforge-ready ] || ! cmp -s backend/requirements.txt .venv/.nanoforge-ready; then
  if ! .venv/bin/python -c 'import torch' >/dev/null 2>&1; then
    if [ "$(uname -s)" = 'Linux' ] && [ "$NANOFORGE_TORCH" = 'cpu' ]; then
      echo 'Installing CPU-only PyTorch (skips the large NVIDIA CUDA packages).'
      .venv/bin/python -m pip install --no-cache-dir 'torch>=2.4,<3' --index-url https://download.pytorch.org/whl/cpu
    fi
  else
    echo 'Keeping the existing PyTorch installation.'
  fi
  .venv/bin/python -m pip install --no-cache-dir -r backend/requirements.txt
  cp backend/requirements.txt .venv/.nanoforge-ready
fi
if [ ! -d frontend/node_modules ] || [ frontend/package-lock.json -nt frontend/node_modules/.package-lock.json ]; then
  (cd frontend && npm ci)
fi
if [ "$NANOFORGE_TORCH" = 'xpu' ]; then
  .venv/bin/python -m backend.check_device --device xpu
fi
if [ "${NANOFORGE_SETUP_ONLY:-0}" = '1' ]; then
  echo 'NanoForge dependencies are ready.'
  exit 0
fi
BACKEND_PID=''
FRONTEND_PID=''
cleanup() {
  trap - EXIT INT TERM
  if [ -n "$FRONTEND_PID" ]; then kill "$FRONTEND_PID" 2>/dev/null || true; fi
  if [ -n "$BACKEND_PID" ]; then kill -TERM "$BACKEND_PID" 2>/dev/null || true; wait "$BACKEND_PID" 2>/dev/null || true; fi
}
trap cleanup EXIT INT TERM
.venv/bin/python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 &
BACKEND_PID=$!
(cd frontend && exec node node_modules/vite/bin/vite.js --host 127.0.0.1 --port 5173 --strictPort) &
FRONTEND_PID=$!
echo 'NanoForge: http://localhost:5173 | API: http://localhost:8000/docs'
echo 'Press Ctrl+C to stop. In-flight training finishes its step and saves a checkpoint.'
# Portable on macOS Bash 3.2 (which lacks wait -n).
while kill -0 "$BACKEND_PID" 2>/dev/null && kill -0 "$FRONTEND_PID" 2>/dev/null; do sleep 1; done
