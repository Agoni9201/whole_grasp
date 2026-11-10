#!/usr/bin/env bash
# T01 real-robot trial: measured-first MGPC/GenX + bounded side observations.
# Executes observation moves and, if qualified, closes/lifts to the configured
# joint hold. This entry does not enable handover or automatic object release.
set -euo pipefail

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  printf '%s\n' \
    'Usage: bash scripts/test_active_grasp_feedback.sh' \
    'Requires grasp_yolo26, configured can0, and the banana on the table.' \
    'Executes center + 1..2 side attempts, then qualified grasp/lift/joint hold.' \
    'Uses Orbbec CP0T963000S4, sept01 trial calibration and MGPC + GraspGenX.' \
    'Close other arm/camera programs first. If already holding an object, do not restart grasp.'
  exit 0
fi
if [[ $# -ne 0 ]]; then
  printf '%s\n' 'No positional arguments supported; use --help.' >&2
  exit 2
fi
: "${CONDA_PREFIX:?Activate grasp_yolo26 before running this script}"
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"
export MOTION_SUPPORT_RELEASE_HEIGHT=0.025

exec bash scripts/run_grasp_safe.sh \
  --config config/pipeline_orbbec.yaml --phase grasp --execute \
  --camera-backend orbbec --camera-serial CP0T963000S4 \
  --camera-self-mask \
  --camera-self-mask-calibration config/orbbec_self_mask_20261109_100mm_reference_trial.json \
  --calib-json config/sept01_joint_refined_candidate.json \
  --nero-firmware V112 \
  --target-class banana \
  --yolo-weights models/food5_visible_v3_full_reviewed_corrected_best.pt \
  --active-view --no-active-multiview-gate \
  --active-center-first --active-restore-center --save-active-raw-views \
  --active-fusion-register --active-fusion-reject-bad-views \
  --active-center-height 0.35 --active-center-roll-plan-candidates 1 \
  --no-active-center-optical-relaxation --active-capture-settle-seconds 1.0 \
  --active-side-view-count 2 --grasp-required-view-count 2 \
  --grasp-table-reference-z 0 --preview-cloud \
  --trajectory-executor mit --mit-q-log \
  --mit-max-joint-vel-deg-s 8 --mit-max-joint-accel-deg-s2 8 \
  --grasp-generator graspgenx --nero-grasp-refine --graspgenx-contact-advance auto \
  --grasp-close-pos-tol 0.004 \
  --mgpc --mgpc-category banana \
  --active-grasp-feedback --feedback-max-attempts 2 \
  --feedback-probe-preflight-max-candidates 6 --feedback-probe-refine-timeout-s 15 \
  --save-dir logs/active_grasp_feedback_20261109/banana_orbbec_sept_joint
