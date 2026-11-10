#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VISIBLE_WEIGHTS="${VISIBLE_WEIGHTS:-$PROJECT_DIR/models/food5_visible_v3_full_reviewed_corrected_best.pt}"
AMODAL_WEIGHTS="${AMODAL_WEIGHTS:-$PROJECT_DIR/models/food5_amodal_real_v2_best.pt}"
if [[ "${TARGET_APPROACH_FINGER_ENVIRONMENT_GUARD:-1}" == "1" ]]; then
  TARGET_APPROACH_FINGER_ENVIRONMENT_GUARD_FLAG="--target-approach-finger-environment-guard"
else
  TARGET_APPROACH_FINGER_ENVIRONMENT_GUARD_FLAG="--no-target-approach-finger-environment-guard"
fi
if [[ "${DIRECT_PICKUP_CONTACT_OVERRIDE:-0}" == "1" ]]; then
  DIRECT_PICKUP_CONTACT_OVERRIDE_FLAG="--direct-pickup-contact-override"
else
  DIRECT_PICKUP_CONTACT_OVERRIDE_FLAG="--no-direct-pickup-contact-override"
fi
if [[ "${ACTIVE_FUSION_RIGID_ICP:-0}" == "1" ]]; then
  ACTIVE_FUSION_RIGID_ICP_FLAG="--active-fusion-rigid-icp"
else
  ACTIVE_FUSION_RIGID_ICP_FLAG="--no-active-fusion-rigid-icp"
fi
if [[ "${ACTIVE_SCENE_TABLE_REGISTRATION:-0}" == "1" ]]; then
  ACTIVE_SCENE_TABLE_REGISTRATION_FLAG="--active-scene-table-registration"
  ACTIVE_REG_APPLY_TO_SCENE_FLAG="--active-reg-apply-to-scene"
else
  ACTIVE_SCENE_TABLE_REGISTRATION_FLAG="--no-active-scene-table-registration"
  ACTIVE_REG_APPLY_TO_SCENE_FLAG="--no-active-reg-apply-to-scene"
fi
if [[ "${ACTIVE_ROUND_TABLE_REGISTRATION:-0}" == "1" ]]; then
  ACTIVE_ROUND_TABLE_REGISTRATION_FLAG="--active-round-table-registration"
else
  ACTIVE_ROUND_TABLE_REGISTRATION_FLAG="--no-active-round-table-registration"
fi

if [[ ! -f "$VISIBLE_WEIGHTS" ]]; then
  echo "Visible weights not found: $VISIBLE_WEIGHTS" >&2
  exit 2
fi

if [[ ! -f "$AMODAL_WEIGHTS" ]]; then
  echo "Amodal weights not found: $AMODAL_WEIGHTS" >&2
  echo "Train the food5 amodal model before enabling amodal clear-first decisions." >&2
  exit 2
fi

exec bash "$PROJECT_DIR/run_real_nero_safe.sh" \
  --yolo-weights "$VISIBLE_WEIGHTS" \
  --amodal-weights "$AMODAL_WEIGHTS" \
  --enable-clear-obstacle \
  --active-view \
  --active-center-height "${ACTIVE_CENTER_HEIGHT:-0.30}" \
  --active-direct-return-min-clearance "${ACTIVE_RETURN_MIN_CLEARANCE:-0.020}" \
  --active-direct-return-runtime-downward-tolerance "${ACTIVE_RETURN_RUNTIME_DOWNWARD_TOLERANCE:-0.070}" \
  --active-return-execution-waypoint-deg "${ACTIVE_RETURN_EXECUTION_WAYPOINT_DEG:-15}" \
  --active-orbit-max-angle-deg "${ACTIVE_ORBIT_MAX_ANGLE_DEG:-45}" \
  --active-nbv-azimuth-step-deg "${ACTIVE_NBV_AZIMUTH_STEP_DEG:-22.5}" \
  --active-nbv-orbit-degs "${ACTIVE_NBV_ORBIT_DEGS:-20,30,35,45}" \
  --active-nbv-plan-candidate-num "${ACTIVE_NBV_PLAN_CANDIDATE_NUM:-32}" \
  --active-nbv-fast-plan-candidates "${ACTIVE_NBV_FAST_PLAN_CANDIDATES:-8}" \
  --active-nbv-fast-roll-candidates "${ACTIVE_NBV_FAST_ROLL_CANDIDATES:-2}" \
  --active-nbv-fast-ik-seed-limit "${ACTIVE_NBV_FAST_IK_SEED_LIMIT:-2}" \
  --active-nbv-min-valid-candidates "${ACTIVE_NBV_MIN_VALID_CANDIDATES:-8}" \
  --active-nbv-motion-weight "${ACTIVE_NBV_MOTION_WEIGHT:-0.0}" \
  --active-side-roll-samples "${ACTIVE_SIDE_ROLL_SAMPLES:-16}" \
  --active-side-roll-plan-candidates "${ACTIVE_SIDE_ROLL_PLAN_CANDIDATES:-8}" \
  --active-side-roll-ik-seed-limit "${ACTIVE_SIDE_ROLL_IK_SEED_LIMIT:-1}" \
  --active-side-roll-keep-candidates "${ACTIVE_SIDE_ROLL_KEEP_CANDIDATES:-2}" \
  --active-side-max-capture-attempts "${ACTIVE_SIDE_MAX_CAPTURE_ATTEMPTS:-3}" \
  --active-side-view-count "${ACTIVE_SIDE_VIEW_COUNT:-1}" \
  --active-path-guard \
  --active-path-obstacle-cap "${ACTIVE_PATH_OBSTACLE_CAP:-200000}" \
  --motion-path-obstacle-cap "${MOTION_PATH_OBSTACLE_CAP:-200000}" \
  --motion-transition-max-segments "${MOTION_TRANSITION_MAX_SEGMENTS:-4}" \
  "$ACTIVE_REG_APPLY_TO_SCENE_FLAG" \
  "$ACTIVE_SCENE_TABLE_REGISTRATION_FLAG" \
  "$ACTIVE_ROUND_TABLE_REGISTRATION_FLAG" \
  --active-table-reg-max-tilt-deg "${ACTIVE_TABLE_REG_MAX_TILT_DEG:-6.0}" \
  --active-confirm-each-view \
  --orbbec-depth-median-frames "${DEPTH_MEDIAN_FRAMES:-7}" \
  --orbbec-depth-median-min-valid "${DEPTH_MEDIAN_MIN_VALID:-5}" \
  --orbbec-depth-median-max-mad "${DEPTH_MEDIAN_MAX_MAD:-0.004}" \
  --camera-self-mask-expand-px "${CAMERA_SELF_MASK_EXPAND_PX:-12}" \
  --camera-self-edge-island-filter \
  --camera-self-edge-island-distance-px "${CAMERA_SELF_EDGE_ISLAND_DISTANCE_PX:-10}" \
  --camera-self-edge-island-max-area "${CAMERA_SELF_EDGE_ISLAND_MAX_AREA:-24}" \
  --camera-self-edge-island-max-diameter "${CAMERA_SELF_EDGE_ISLAND_MAX_DIAMETER:-12}" \
  --camera-near-field-ghost-filter \
  --camera-near-field-ghost-roi "${CAMERA_NEAR_GHOST_XMIN:-0.05}" "${CAMERA_NEAR_GHOST_XMAX:-0.30}" "${CAMERA_NEAR_GHOST_YMIN:-0.35}" "${CAMERA_NEAR_GHOST_YMAX:-1.00}" \
  --camera-near-field-ghost-seed-depth "${CAMERA_NEAR_GHOST_SEED_DEPTH:-0.220}" \
  --camera-near-field-ghost-grow-depth "${CAMERA_NEAR_GHOST_GROW_DEPTH:-0.260}" \
  --camera-near-field-ghost-min-seed-pixels "${CAMERA_NEAR_GHOST_MIN_SEED_PIXELS:-1}" \
  --depth-hole-max-area "${DEPTH_HOLE_MAX_AREA:-36}" \
  --depth-hole-max-diameter "${DEPTH_HOLE_MAX_DIAMETER:-7}" \
  --depth-hole-max-span "${DEPTH_HOLE_MAX_SPAN:-0.015}" \
  --voxel-size "${TARGET_VOXEL_SIZE:-0.0010}" \
  --active-fusion-voxel "${ACTIVE_FUSION_VOXEL:-0.0025}" \
  --active-target-fusion-mode "${ACTIVE_TARGET_FUSION_MODE:-surfel}" \
  --active-surfel-merge-radius "${ACTIVE_SURFEL_MERGE_RADIUS:-0.006}" \
  --active-surfel-normal-angle-deg "${ACTIVE_SURFEL_NORMAL_ANGLE_DEG:-55.0}" \
  "$ACTIVE_FUSION_RIGID_ICP_FLAG" \
  --active-reg-max-median "${ACTIVE_REG_MAX_MEDIAN:-0.006}" \
  --active-reg-max-p95 "${ACTIVE_REG_MAX_P95:-0.020}" \
  --active-reg-min-inlier-1cm "${ACTIVE_REG_MIN_INLIER_1CM:-0.70}" \
  --active-reg-icp-estimation "${ACTIVE_REG_ICP_ESTIMATION:-point_to_point}" \
  --active-reg-translation-iters "${ACTIVE_REG_TRANSLATION_ITERS:-3}" \
  --active-reg-icp-robust-kernel "${ACTIVE_REG_ICP_ROBUST_KERNEL:-0.005}" \
  --active-reg-icp-thresholds "${ACTIVE_REG_ICP_THRESHOLDS:-0.015,0.008}" \
  --active-reg-icp-max-trans "${ACTIVE_REG_ICP_MAX_TRANS:-0.010}" \
  --active-reg-icp-max-rot-deg "${ACTIVE_REG_ICP_MAX_ROT_DEG:-3.0}" \
  --active-fusion-overlap-max-median "${ACTIVE_FUSION_OVERLAP_MAX_MEDIAN:-0.004}" \
  --active-fusion-overlap-max-p95 "${ACTIVE_FUSION_OVERLAP_MAX_P95:-0.020}" \
  --active-fusion-overlap-min-inlier-1cm "${ACTIVE_FUSION_OVERLAP_MIN_INLIER_1CM:-0.75}" \
  --scene-voxel-size "${SCENE_VOXEL_SIZE:-0.0030}" \
  --scene-cloud-cap "${SCENE_CLOUD_CAP:-200000}" \
  --preview-point-size "${PREVIEW_POINT_SIZE:-1.5}" \
  --grasp-angle-deg "${GRASP_ANGLE_DEG:-75}" \
  --graspnet-collision-approach-dist "${GRASPNET_COLLISION_APPROACH_DIST:-0.04}" \
  --max-grasp-width "${MAX_GRASP_WIDTH:-0.110}" \
  --graspnet-decode-max-width "${GRASPNET_DECODE_MAX_WIDTH:-0.110}" \
  --graspnet-main-component-filter \
  --clear-max-rounds "${CLEAR_MAX_ROUNDS:-2}" \
  --clear-empty-grasp-retries "${CLEAR_EMPTY_GRASP_RETRIES:-1}" \
  --clear-score-threshold "${CLEAR_SCORE_THRESHOLD:-0.35}" \
  --clear-amodal-occlusion-threshold "${CLEAR_OCCLUSION_THRESHOLD:-0.25}" \
  --clear-amodal-min-hidden-overlap "${CLEAR_MIN_HIDDEN_OVERLAP:-0.05}" \
  --clear-largest-visible-fallback \
  --clear-largest-fallback-occlusion "${CLEAR_LARGEST_FALLBACK_OCCLUSION:-0.70}" \
  --post-clear-reidentify-max-center-distance "${POST_CLEAR_REIDENTIFY_MAX_CENTER_DISTANCE:-0.10}" \
  --post-clear-reidentify-center-ambiguity-margin "${POST_CLEAR_REIDENTIFY_CENTER_AMBIGUITY_MARGIN:-0.03}" \
  --post-clear-reidentify-min-confidence "${POST_CLEAR_REIDENTIFY_MIN_CONFIDENCE:-0.50}" \
  --post-clear-center-relocalize \
  "$@" \
  --no-active-up-and-over \
  --no-motion-up-and-over \
  --no-use-intermediate-motion \
  "$DIRECT_PICKUP_CONTACT_OVERRIDE_FLAG" \
  "$TARGET_APPROACH_FINGER_ENVIRONMENT_GUARD_FLAG"
