#!/usr/bin/env bash
# Start the anomaly-detection web UI on http://127.0.0.1:8000 (Mac, local only).
# Every model file / cache stays inside this project folder.
set -euo pipefail
cd "$(dirname "$0")"
ROOT="$PWD"
export TORCH_HOME="$ROOT/models_cache/torch"          # X3D-S (torch.hub repo + weights)
export HF_HOME="$ROOT/models_cache/huggingface"       # CLIP ViT-B/16
export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1    # never re-download at runtime
export PIP_CACHE_DIR="$ROOT/models_cache/pip"
export MPLCONFIGDIR="$ROOT/models_cache/matplotlib"
export PYTORCH_ENABLE_MPS_FALLBACK=1
export PYTHONPATH="$ROOT"
PORT="${PORT:-8000}"
echo "Open http://127.0.0.1:$PORT  (Ctrl+C to stop)"
exec venv/bin/python -m uvicorn webui.server:app --host 127.0.0.1 --port "$PORT" "$@"
