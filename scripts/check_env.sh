#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${CONDA_PREFIX:?Activate grasp_yolo26 before running this script}"
export PYTHONNOUSERSITE=1
cd "$PROJECT_DIR"
"$CONDA_PREFIX/bin/python" - <<'PY'
import importlib
import sys
print('Python:', sys.executable)
failed=[]
for name in ['numpy','scipy','cv2','can','pyAgxArm','pyrealsense2','open3d','ultralytics','torch','pinocchio','mujoco','pytest','trac_ik']:
    try:
        module=importlib.import_module(name)
        print(name, getattr(module,'__file__',None))
    except Exception as exc:
        print(name, type(exc).__name__, str(exc)); failed.append(name)
if 'torch' not in failed:
    import torch
    print('CUDA build:',torch.version.cuda,'available:',torch.cuda.is_available())
    if not torch.cuda.is_available(): failed.append('CUDA')
print('Failed:',failed)
raise SystemExit(bool(failed))
PY
