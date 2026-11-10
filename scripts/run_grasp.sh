#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${CONDA_PREFIX:?Activate grasp_yolo26 before running this script}"
export PYTHONNOUSERSITE=1
# ROS Noetic may export its Python 3.8-era packages through PYTHONPATH.  Keep
# them available, but prefer the active Conda environment so xacrodoc loads the
# compatible xacro installed alongside roboticstoolbox.
CONDA_SITE_PACKAGES="$CONDA_PREFIX/lib/python3.10/site-packages"
export PYTHONPATH="$CONDA_SITE_PACKAGES${PYTHONPATH:+:$PYTHONPATH}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"
export YOLO_CONFIG_DIR="${YOLO_CONFIG_DIR:-$PROJECT_DIR/.ultralytics}"
mkdir -p "$MPLCONFIGDIR" "$YOLO_CONFIG_DIR"
cd "$PROJECT_DIR"
exec "$CONDA_PREFIX/bin/python" main.py "$@"
