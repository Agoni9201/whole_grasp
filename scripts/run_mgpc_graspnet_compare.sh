#!/usr/bin/env bash
set -euo pipefail
GRASP_COMPARE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$GRASP_COMPARE_ROOT"
export PYTHONNOUSERSITE=1
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mgpc_matplotlib}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export MUJOCO_VIEWER=0
exec "$GRASP_COMPARE_ROOT/.venv_mgpc/bin/python" tools/compare_mgpc_graspnet.py "$@"
