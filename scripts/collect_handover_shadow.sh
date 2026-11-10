#!/usr/bin/env bash
# Full grasp -> wrist delivery -> external RGB/seven-axis shadow collection.
# No pull model is loaded; the fusion handover never opens the held object.
set -euo pipefail

usage() {
  printf '%s\n' 'Usage: bash scripts/collect_handover_shadow.sh [Left|Right] [--held] [--group GRASP_ID --split train|validation|test]' \
    'Side is the MediaPipe label in the external unmirrored camera view.' \
    'Requires grasp_yolo26 and configured can0. Default: execute grasp and delivery; shadow handover.' \
    '--held: already holding an object; enter servo directly, skipping initial pose, opening and grasp.' \
    'The previous controller must have exited. --held still starts arm hold/follow control.' \
    '--group identifies one physical grasp, including restarts; keep it in one split.'
}

receiver_side=Left
receiver_side_set=0
collection_phase=all
collection_group=
collection_split=
while [[ $# -gt 0 ]]; do
  arg="$1"
  case "$arg" in
    Left|Right)
      if [[ "$receiver_side_set" == 1 ]]; then
        printf '%s\n' 'Specify receiver side only once: Left or Right.' >&2
        exit 2
      fi
      receiver_side="$arg"
      receiver_side_set=1
      ;;
    --held)
      if [[ "$collection_phase" == servo ]]; then
        printf '%s\n' 'Specify --held only once.' >&2
        exit 2
      fi
      collection_phase=servo
      ;;
    --group|--split)
      if [[ $# -lt 2 || -z "$2" || "$2" == --* ]]; then
        printf '%s needs a value.\n' "$arg" >&2
        exit 2
      fi
      if [[ "$arg" == --group ]]; then
        if [[ -n "$collection_group" ]]; then printf '%s\n' 'Specify --group only once.' >&2; exit 2; fi
        collection_group="$2"
      else
        if [[ -n "$collection_split" ]]; then printf '%s\n' 'Specify --split only once.' >&2; exit 2; fi
        collection_split="$2"
      fi
      shift
      ;;
    --help|-h) usage; exit 0 ;;
    *) printf 'Unknown argument: %s\n' "$arg" >&2; usage >&2; exit 2 ;;
  esac
  shift
done
collection_args=()
save_dir=logs/handover_fusion_20261109/orbbec_d457_shadow
if [[ -n "$collection_group" || -n "$collection_split" ]]; then
  if [[ ! "$collection_group" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$ || ! "$collection_split" =~ ^(train|validation|test)$ ]]; then
    printf '%s\n' 'Supply --group GRASP_ID and --split train|validation|test together; group uses letters, digits, _, . or -.' >&2
    exit 2
  fi
  for other_split in train validation test; do
    if [[ "$other_split" != "$collection_split" && -d "$(dirname "${BASH_SOURCE[0]}")/../logs/handover_dataset_20261109/$other_split/$collection_group" ]]; then
      printf 'Physical grasp %s already exists in %s; keep restarts in that split.\n' "$collection_group" "$other_split" >&2
      exit 2
    fi
  done
  collection_args=(--fusion-collection-group "$collection_group" --fusion-collection-split "$collection_split")
  save_dir="logs/handover_dataset_20261109/$collection_split/$collection_group"
fi
: "${CONDA_PREFIX:?Activate grasp_yolo26 before running this script}"
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"
"$CONDA_PREFIX/bin/python" tools/check_handover_usb.py --wrist-serial CP0T963000S4
export MOTION_SUPPORT_RELEASE_HEIGHT=0.025
printf '[collect] receiver=%s phase=%s handover=shadow\n' "$receiver_side" "$collection_phase"
printf '[collect] save_dir=%s group=%s split=%s\n' "$save_dir" "${collection_group:-unassigned}" "${collection_split:-unassigned}"

exec bash scripts/run_grasp_safe.sh \
  --config config/pipeline_orbbec.yaml --phase "$collection_phase" --execute \
  --camera-backend orbbec --camera-serial CP0T963000S4 \
  --camera-self-mask \
  --camera-self-mask-calibration config/orbbec_self_mask_20261109_100mm_reference_trial.json \
  --calib-json config/sept01_joint_refined_candidate.json \
  --handeye config/sept01_joint_refined_candidate.json \
  --nero-firmware V112 \
  --target-class banana \
  --yolo-weights models/food5_visible_v3_full_reviewed_corrected_best.pt \
  --active-view --no-active-multiview-gate \
  --active-center-height 0.35 --active-center-roll-plan-candidates 1 \
  --no-active-center-optical-relaxation --active-capture-settle-seconds 1.0 \
  --active-side-view-count 0 --grasp-required-view-count 1 \
  --grasp-table-reference-z 0 --preview-cloud \
  --trajectory-executor mit --mit-q-log \
  --mit-max-joint-vel-deg-s 8 --mit-max-joint-accel-deg-s2 8 \
  --grasp-generator graspgenx --nero-grasp-refine --graspgenx-contact-advance auto \
  --grasp-close-pos-tol 0.004 --no-mgpc \
  --enable-servo-delivery --target-source mediapipe --hand-camera-center \
  --servo-preview --servo-diagnostics --hand-depth-radius 8 \
  --fusion-handover shadow --fusion-global-serial 234222300443 \
  --fusion-receiver-side "$receiver_side" \
  "${collection_args[@]}" \
  --delivery-distance 0.25 --fusion-ready-distance 0.28 \
  --fusion-ready-seconds 0.15 --fusion-baseline-seconds 3 \
  --max-linear-vel 0.03 --max-joint-vel 0.08 --max-angular-vel 0.15 \
  --impedance-target-linear-vel 0.03 --impedance-target-angular-vel 0.15 \
  --save-dir "$save_dir"
