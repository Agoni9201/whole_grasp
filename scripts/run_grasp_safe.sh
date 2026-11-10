#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ "${GRASP_VALIDATION_MODE:-1}" == "1" ]]; then
  GRASP_VALIDATION_MODE_FLAG="--grasp-validation-mode"
else
  GRASP_VALIDATION_MODE_FLAG="--no-grasp-validation-mode"
fi
if [[ "${GRASP_VALIDATION_FIRST:-1}" == "1" ]]; then
  GRASP_VALIDATION_FIRST_FLAG="--grasp-validation-first"
else
  GRASP_VALIDATION_FIRST_FLAG="--no-grasp-validation-first"
fi
if [[ "${GRASP_SELECTIVE_IK_RESCUE:-1}" == "1" ]]; then
  GRASP_SELECTIVE_IK_RESCUE_FLAG="--grasp-selective-ik-rescue"
else
  GRASP_SELECTIVE_IK_RESCUE_FLAG="--no-grasp-selective-ik-rescue"
fi
if [[ "${ACTIVE_NBV_SELECTIVE_IK_RESCUE:-1}" == "1" ]]; then
  ACTIVE_NBV_SELECTIVE_IK_RESCUE_FLAG="--active-nbv-selective-ik-rescue"
else
  ACTIVE_NBV_SELECTIVE_IK_RESCUE_FLAG="--no-active-nbv-selective-ik-rescue"
fi
if [[ "${MOVE_CONTROLLER_AWARE_RETRY:-1}" == "1" ]]; then
  MOVE_CONTROLLER_AWARE_RETRY_FLAG="--move-controller-aware-retry"
else
  MOVE_CONTROLLER_AWARE_RETRY_FLAG="--no-move-controller-aware-retry"
fi

exec bash "$PROJECT_DIR/run_grasp.sh" \
  "$@" \
  --nero-speed-percent 5 \
  --grasp-motion-path-guard \
  --motion-runtime-collision-guard \
  --post-grasp-runtime-replan \
  --post-grasp-runtime-replan-min-clearance "${POST_GRASP_RUNTIME_REPLAN_MIN_CLEARANCE:-0.015}" \
  --post-grasp-runtime-replan-timeout "${POST_GRASP_RUNTIME_REPLAN_TIMEOUT:-1.5}" \
  --post-grasp-runtime-replan-max-nodes "${POST_GRASP_RUNTIME_REPLAN_MAX_NODES:-200}" \
  --motion-runtime-torque-guard \
  --motion-runtime-torque-delta "${MOTION_RUNTIME_TORQUE_DELTA:-8.0}" \
  --motion-runtime-torque-confirm-samples "${MOTION_RUNTIME_TORQUE_CONFIRM_SAMPLES:-2}" \
  --motion-runtime-torque-retract-speed-percent "${MOTION_RUNTIME_TORQUE_RETRACT_SPEED_PERCENT:-2}" \
  --motion-path-min-clearance "${MOTION_PATH_CLEARANCE:-0.005}" \
  --motion-path-max-joint-delta-deg "${MOTION_PATH_MAX_JOINT_DELTA_DEG:-180}" \
  --max-candidate-ik-joint-delta "${MAX_CANDIDATE_IK_JOINT_DELTA_RAD:-3.141592653589793}" \
  --ik-solution-topk "${IK_SOLUTION_TOPK:-6}" \
  --ik-solution-dedup-deg "${IK_SOLUTION_DEDUP_DEG:-0.5}" \
  --ik-lift-joint7-weight "${IK_LIFT_JOINT7_WEIGHT:-0.0}" \
  --grasp-stage-beam-search \
  --grasp-stage-beam-width "${GRASP_STAGE_BEAM_WIDTH:-6}" \
  --grasp-progressive-beam \
  --no-grasp-progressive-full-fallback \
  --grasp-fast-stage-beam-width "${GRASP_FAST_STAGE_BEAM_WIDTH:-2}" \
  --grasp-fast-ik-solution-topk "${GRASP_FAST_IK_SOLUTION_TOPK:-2}" \
  --grasp-fast-ik-seed-limit "${GRASP_FAST_IK_SEED_LIMIT:-5}" \
  "$GRASP_SELECTIVE_IK_RESCUE_FLAG" \
  --grasp-selective-ik-rescue-candidates "${GRASP_SELECTIVE_IK_RESCUE_CANDIDATES:-3}" \
  --grasp-selective-ik-rescue-seed-limit "${GRASP_SELECTIVE_IK_RESCUE_SEED_LIMIT:-12}" \
  --grasp-selective-ik-rescue-all-joint-seeds "${GRASP_SELECTIVE_IK_RESCUE_ALL_JOINT_SEEDS:-8}" \
  --grasp-selective-ik-rescue-topk "${GRASP_SELECTIVE_IK_RESCUE_TOPK:-3}" \
  --grasp-selective-ik-rescue-beam-width "${GRASP_SELECTIVE_IK_RESCUE_BEAM_WIDTH:-3}" \
  --grasp-preflight-max-candidates "${GRASP_PREFLIGHT_MAX_CANDIDATES:-12}" \
  --grasp-target-valid-plans "${GRASP_TARGET_VALID_PLANS:-3}" \
  --grasp-micro-lift-height "${GRASP_MICRO_LIFT_HEIGHT:-0.008}" \
  --grasp-micro-lift-ik-seed-limit "${GRASP_MICRO_LIFT_IK_SEED_LIMIT:-5}" \
  --no-motion-neutral-pregrasp-fallback \
  --ik-all-joint-seed-count "${IK_ALL_JOINT_SEED_COUNT:-0}" \
  "$ACTIVE_NBV_SELECTIVE_IK_RESCUE_FLAG" \
  --active-nbv-selective-ik-rescue-candidates "${ACTIVE_NBV_SELECTIVE_IK_RESCUE_CANDIDATES:-4}" \
  --active-nbv-selective-ik-rescue-all-joint-seeds "${ACTIVE_NBV_SELECTIVE_IK_RESCUE_ALL_JOINT_SEEDS:-8}" \
  --active-nbv-selective-ik-rescue-seed-limit "${ACTIVE_NBV_SELECTIVE_IK_RESCUE_SEED_LIMIT:-12}" \
  --motion-runtime-stop-clearance "${MOTION_RUNTIME_STOP_CLEARANCE:-0.005}" \
  --motion-runtime-open-finger-clearance-slack "${MOTION_RUNTIME_OPEN_FINGER_CLEARANCE_SLACK:-0.002}" \
  --motion-runtime-target-approach-finger-clearance-slack "${MOTION_RUNTIME_TARGET_APPROACH_FINGER_CLEARANCE_SLACK:-0.002}" \
  --motion-runtime-finger-clearance-confirm-samples "${MOTION_RUNTIME_FINGER_CLEARANCE_CONFIRM_SAMPLES:-2}" \
  --motion-runtime-joint-path-residual-deg "${MOTION_RUNTIME_JOINT_PATH_RESIDUAL_DEG:-8.0}" \
  --motion-runtime-joint-path-confirm-samples "${MOTION_RUNTIME_JOINT_PATH_CONFIRM_SAMPLES:-2}" \
  --grasp-close-exact-retry \
  --grasp-execution-waypoint-deg "${GRASP_EXECUTION_WAYPOINT_DEG:-20}" \
  --carrying-execution-waypoint-deg "${CARRYING_EXECUTION_WAYPOINT_DEG:-30}" \
  --released-return-execution-waypoint-deg "${RELEASED_RETURN_EXECUTION_WAYPOINT_DEG:-20}" \
  --place-release-retry-count "${PLACE_RELEASE_RETRY_COUNT:-1}" \
  --place-release-step-count "${PLACE_RELEASE_STEP_COUNT:-3}" \
  --place-release-step-settle "${PLACE_RELEASE_STEP_SETTLE:-0.25}" \
  --motion-target-non-tool-clearance "${MOTION_TARGET_NON_TOOL_CLEARANCE:-0.005}" \
  --motion-target-gripper-body-clearance "${MOTION_TARGET_GRIPPER_BODY_CLEARANCE:-0.002}" \
  --motion-path-tool-radius "${MOTION_GRIPPER_PALM_RADIUS:-0.035}" \
  --motion-path-open-finger-radius "${MOTION_GRIPPER_FINGER_RADIUS:-0.007}" \
  --motion-path-open-finger-back "${MOTION_GRIPPER_FINGER_BACK:-0.065}" \
  --motion-path-open-finger-front "${MOTION_GRIPPER_FINGER_FRONT:-0.005}" \
  --motion-self-filter-finger-radius "${MOTION_GRIPPER_FINGER_RADIUS:-0.007}" \
  --motion-closed-gripper-empty-width "${MOTION_CLOSED_GRIPPER_EMPTY_WIDTH:-0.000}" \
  --motion-closed-gripper-width-padding "${MOTION_CLOSED_GRIPPER_WIDTH_PADDING:-0.010}" \
  --motion-closed-finger-contact-tolerance "${MOTION_CLOSED_FINGER_CONTACT_TOLERANCE:-0.010}" \
  --move-wait-timeout "${MOVE_WAIT_TIMEOUT:-60}" \
  --move-poll-interval "${MOTION_POLL_INTERVAL:-0.10}" \
  "$MOVE_CONTROLLER_AWARE_RETRY_FLAG" \
  --move-unstarted-settle-timeout "${MOVE_UNSTARTED_SETTLE_TIMEOUT:-2.5}" \
  --move-unstarted-settle-samples "${MOVE_UNSTARTED_SETTLE_SAMPLES:-2}" \
  --motion-attached-object-clearance "${MOTION_ATTACHED_CLEARANCE:-0.000}" \
  --motion-payload-support-plane-clearance "${MOTION_PAYLOAD_SUPPORT_PLANE_CLEARANCE:-0.000}" \
  --motion-support-release-height "${MOTION_SUPPORT_RELEASE_HEIGHT:-0.020}" \
  --motion-support-strict-rise "${MOTION_SUPPORT_STRICT_RISE:-0.018}" \
  --motion-runtime-support-xy-drift-slack "${MOTION_RUNTIME_SUPPORT_XY_DRIFT_SLACK:-0.015}" \
  --motion-runtime-support-tilt-slack-deg "${MOTION_RUNTIME_SUPPORT_TILT_SLACK_DEG:-5.0}" \
  --motion-runtime-support-drop-slack "${MOTION_RUNTIME_SUPPORT_DROP_SLACK:-0.005}" \
  --motion-local-support-enable \
  --motion-local-support-max-gap "${MOTION_LOCAL_SUPPORT_MAX_GAP:-0.025}" \
  --motion-local-support-contact-distance "${MOTION_LOCAL_SUPPORT_CONTACT_DISTANCE:-0.008}" \
  --motion-local-support-plane-max-mad "${MOTION_LOCAL_SUPPORT_PLANE_MAX_MAD:-0.004}" \
  --motion-local-support-strict-rise "${MOTION_LOCAL_SUPPORT_STRICT_RISE:-0.040}" \
  --motion-transit-vertical-clearance "${MOTION_TRANSIT_CLEARANCE:-0.080}" \
  --motion-transit-max-tcp-z "${MOTION_TRANSIT_MAX_TCP_Z:-0.50}" \
  --gripper-force "${GRIPPER_FORCE:-0.50}" \
  --gripper-open-width "${GRIPPER_OPEN_WIDTH:-0.100}" \
  --gripper-hardware-max-width 0.100 \
  --gripper-hardware-max-force 0.50 \
  --manual-topk "${MANUAL_TOPK:-24}" \
  --max-grasp-width "${MAX_GRASP_WIDTH:-0.110}" \
  --grasp-physical-aperture-margin "${GRASP_PHYSICAL_APERTURE_MARGIN:-0.0005}" \
  --grasp-physical-recenter-max "${GRASP_PHYSICAL_RECENTER_MAX:-0.020}" \
  --grasp-physical-min-contact-inset "${GRASP_PHYSICAL_MIN_CONTACT_INSET:-0.010}" \
  --no-allow-oversize-grasp \
  --gripper-close-width 0.000 \
  --slow-close-gripper \
  --verbose-gripper-close \
  --slow-close-step 0.002 \
  --slow-close-interval 0.25 \
  --gripper-stop-force "${GRIPPER_STOP_FORCE:-0.40}" \
  --gripper-contact-confirm-samples 2 \
  --gripper-contact-confirm-interval 0.10 \
  --gripper-open-confirm-samples 2 \
  --require-gripper-contact \
  --require-gripper-status \
  --no-allow-empty-grasp-lift \
  --no-skip-empty-grasp-return \
  --table-guard-enable \
  --table-z "${TABLE_Z:-0.0}" \
  --table-tcp-clearance "${TABLE_TCP_CLEARANCE:-0.00}" \
  --table-tool-bottom-offset "${TABLE_TOOL_BOTTOM_OFFSET:-0.005}" \
  --motion-direct-release-retreat "${MOTION_DIRECT_RELEASE_RETREAT:-0.020}" \
  --motion-release-retreat-xy-relax "${MOTION_RELEASE_RETREAT_XY_RELAX:-0.010}" \
  --motion-release-retreat-tilt-relax-deg "${MOTION_RELEASE_RETREAT_TILT_RELAX_DEG:-5.0}" \
  --motion-release-retreat-max-attempts "${MOTION_RELEASE_RETREAT_MAX_ATTEMPTS:-9}" \
  --transport-place-relax \
  --transport-place-xy-relax "${TRANSPORT_PLACE_XY_RELAX:-0.020}" \
  --transport-place-z-relax "${TRANSPORT_PLACE_Z_RELAX:-0.020}" \
  --transport-place-tilt-relax-deg "${TRANSPORT_PLACE_TILT_RELAX_DEG:-8.0}" \
  --transport-place-relax-max-attempts "${TRANSPORT_PLACE_RELAX_MAX_ATTEMPTS:-9}" \
  --pregrasp-offset 0.10 \
  --adaptive-lift-heights \
  --adaptive-lift-step "${ADAPTIVE_LIFT_STEP:-0.020}" \
  --adaptive-lift-min-height "${ADAPTIVE_LIFT_MIN_HEIGHT:-0.030}" \
  --adaptive-lift-attempts "${ADAPTIVE_LIFT_ATTEMPTS:-10}" \
  --grasp-lift-endpoint-precheck \
  --grasp-lift-min-joint-margin-deg "${GRASP_LIFT_MIN_JOINT_MARGIN_DEG:-0.5}" \
  --grasp-lift-feasible-region \
  --grasp-lift-region-xy "${GRASP_LIFT_REGION_XY:-0.030}" \
  --grasp-lift-region-max-attempts "${GRASP_LIFT_REGION_MAX_ATTEMPTS:-24}" \
  "$GRASP_VALIDATION_MODE_FLAG" \
  "$GRASP_VALIDATION_FIRST_FLAG" \
  --grasp-validation-lift-height "${GRASP_VALIDATION_LIFT_HEIGHT:-0.070}" \
  --lift-orientation-relax \
  --lift-orientation-relax-degrees 3 6 10 15 20 \
  --lift-height "${LIFT_HEIGHT:-0.12}"
