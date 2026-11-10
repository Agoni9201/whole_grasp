#!/usr/bin/env bash
set -euo pipefail
MGPC_PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$MGPC_PROJECT_DIR"
export PYTHONNOUSERSITE=1
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mgpc_matplotlib}"
export HF_HOME="$MGPC_PROJECT_DIR/third_party/MGPC/pretrained/hf_cache"
export TORCH_HOME="$MGPC_PROJECT_DIR/third_party/MGPC/pretrained/torch_cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
exec "$MGPC_PROJECT_DIR/.venv_mgpc/bin/python" tools/compare_mgpc_reconstruction.py "$@"
