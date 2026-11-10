#!/usr/bin/env bash
# SDK camera-only preview. No ROS driver or CAN connection is needed.
set -euo pipefail
cd "$(dirname "$0")/.."
exec python scripts/preview_rgbd_camera.py --config config/pipeline_orbbec.yaml "$@"
