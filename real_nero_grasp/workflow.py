"""Top-level real Nero grasp-session orchestration.

This module owns only workflow ordering and result aggregation.  Hardware,
perception, calibration, active observation, clearing, and grasp planning are
provided explicitly through :class:`WorkflowHooks`.
"""

from dataclasses import dataclass
from typing import Any, Callable, Optional

import numpy as np

from .experiment_log import (
    mark_experiment_stage,
    record_experiment_views,
    set_experiment_session,
)
from .types import (
    GraspAttemptResult,
    GraspAttemptStatus,
    UnsafeMotionTarget,
)


@dataclass(frozen=True)
class WorkflowHooks:
    """Injected collaborators required by :func:`run_single_grasp`."""

    arm_factory: Callable[..., Any]
    configure_amodal_clear_model: Callable[[Any], Any]
    test_current_flange_ik: Callable[[Any, Any], Any]
    prepare_initial_pose_for_capture: Callable[[Any, Any], Any]
    capture_target_cloud_with_retries: Callable[[Any], Any]
    preview_reconstructed_cloud: Callable[[Any, Any, Any], Any]
    T_base_camera_from_flange: Callable[[Any, Any], Any]
    maybe_clear_obstacles_before_active: Callable[[Any, Any, Any], Any]
    run_optional_active_view: Callable[..., Any]
    plan_and_maybe_execute_grasp: Callable[..., GraspAttemptResult]
    record_workflow_motion: Callable[[Any, Any], Any]
    with_workflow_motion: Callable[[Any, GraspAttemptResult], GraspAttemptResult]
    recapture_original_target_after_clear: Callable[[Any, Any, Any], Any] = None
    after_grasp_lifted: Optional[Callable[[Any, Any], Any]] = None


__all__ = [
    "WorkflowHooks",
    "record_workflow_motion",
    "with_workflow_motion",
    "run_single_grasp",
]

def record_workflow_motion(args, motion_sent):
    """Keep the final session result honest across clear/active/grasp stages."""
    if bool(motion_sent):
        setattr(args, "_workflow_motion_sent", True)


def with_workflow_motion(args, result):
    motion_sent = bool(
        result.motion_sent or getattr(args, "_workflow_motion_sent", False)
    )
    if motion_sent == bool(result.motion_sent):
        return result
    return GraspAttemptResult(
        result.status,
        motion_sent,
        result.detail,
        result.empty_grasp_recovered_to_initial,
    )

def run_single_grasp(args, *, hooks):
    if not isinstance(hooks, WorkflowHooks):
        raise TypeError("run_single_grasp requires a WorkflowHooks instance")

    RealNeroGraspArm = hooks.arm_factory
    configure_amodal_clear_model = hooks.configure_amodal_clear_model
    test_current_flange_ik = hooks.test_current_flange_ik
    prepare_initial_pose_for_capture = hooks.prepare_initial_pose_for_capture
    capture_target_cloud_with_retries = hooks.capture_target_cloud_with_retries
    preview_reconstructed_cloud = hooks.preview_reconstructed_cloud
    T_base_camera_from_flange = hooks.T_base_camera_from_flange
    maybe_clear_obstacles_before_active = hooks.maybe_clear_obstacles_before_active
    run_optional_active_view = hooks.run_optional_active_view
    plan_and_maybe_execute_grasp = hooks.plan_and_maybe_execute_grasp
    record_workflow_motion = hooks.record_workflow_motion
    with_workflow_motion = hooks.with_workflow_motion
    recapture_same_target = hooks.recapture_original_target_after_clear

    args._workflow_motion_sent = False
    args._force_active_multiview = False
    args._force_active_center_relocalize = False
    args._last_clear_failure_detail = None
    mark_experiment_stage(args, "arm_setup")
    configure_amodal_clear_model(args)
    with RealNeroGraspArm(
        args.nero_interface,
        args.nero_channel,
        args.nero_bitrate,
        args.nero_firmware,
        args.nero_enable_timeout,
        args.nero_speed_percent,
        gripper_enabled=not args.disable_gripper,
        gripper_max_width=args.gripper_hardware_max_width,
        gripper_max_force=args.gripper_hardware_max_force,
        ordinary_stop_settle_timeout=getattr(
            args, "ordinary_stop_settle_timeout", 3.0
        ),
        ordinary_stop_trend_extension=getattr(
            args, "ordinary_stop_trend_extension", 2.0
        ),
    ) as arm:
        arm.set_tcp_offset(args.tcp_offset)
        arm._stop_confirm_timeout = float(
            getattr(args, "stop_confirm_timeout", 2.0)
        )
        print("[tcp] offset =", np.round(np.asarray(args.tcp_offset, dtype=float), 6).tolist())
        if args.ik_test_current:
            test_current_flange_ik(arm, args)
            return GraspAttemptResult(GraspAttemptStatus.SUCCESS, False, "ik_test_complete")
        if args.gripper_test is not None:
            if args.gripper_test == "open":
                arm.open_gripper(args.gripper_open_width, args.gripper_force, args.gripper_settle)
                record_workflow_motion(args, True)
            elif args.gripper_test == "close":
                arm.close_gripper(args.gripper_close_width, args.gripper_force, args.gripper_settle)
                record_workflow_motion(args, True)
            elif args.gripper_test == "status":
                arm.print_gripper_status()
            return with_workflow_motion(
                args,
                GraspAttemptResult(GraspAttemptStatus.SUCCESS, False, "gripper_test_complete"),
            )
        mark_experiment_stage(args, "initial_pose")
        try:
            prepare_initial_pose_for_capture(arm, args)
        except UnsafeMotionTarget as exc:
            detail = f"initial motion table preflight failed: {exc}"
            print(f"[init][abort] {detail}")
            print("[init][abort] No initial motion command was sent.")
            return GraspAttemptResult(
                GraspAttemptStatus.PLANNING_FAILED,
                False,
                detail,
            )
        record_workflow_motion(args, True)
        args._empty_grasp_recovery_joints = np.asarray(
            arm.get_joint_angles(),
            dtype=float,
        ).reshape(7)
        print(
            "[init] saved empty-grasp recovery joints deg =",
            np.round(np.rad2deg(args._empty_grasp_recovery_joints), 3).tolist(),
        )

        mark_experiment_stage(args, "initial_capture")
        captured = capture_target_cloud_with_retries(args)
        if captured is None:
            return with_workflow_motion(
                args,
                GraspAttemptResult(GraspAttemptStatus.CAPTURE_FAILED, False, "initial_capture_failed"),
            )
        bgr, target_cloud, scene_cloud, out_dir, inst, intr = captured
        set_experiment_session(
            args,
            out_dir,
            target_label=inst.get("label", "unknown"),
        )
        preview_reconstructed_cloud(target_cloud, scene_cloud, args)
        T_base_camera = T_base_camera_from_flange(arm, args)
        mark_experiment_stage(args, "clear_policy")
        captured = maybe_clear_obstacles_before_active(arm, args, captured)
        if captured is None:
            clear_failure_detail = str(
                getattr(args, "_last_clear_failure_detail", None)
                or "required obstacle was not cleared; target grasp skipped"
            )
            return with_workflow_motion(
                args,
                GraspAttemptResult(
                    GraspAttemptStatus.CLEAR_FAILED,
                    False,
                    clear_failure_detail,
                ),
            )
        bgr, target_cloud, scene_cloud, out_dir, inst, intr = captured
        reference_mask = getattr(args, "_last_target_mask", None)
        if reference_mask is not None:
            reference_mask = np.asarray(reference_mask).copy()
        reference_inst = dict(inst) if isinstance(inst, dict) else {}
        target_empty_retries_used = 0
        target_empty_retry_limit = max(
            int(getattr(args, "target_empty_grasp_retries", 1)), 0
        )

        while True:
            T_base_camera = T_base_camera_from_flange(arm, args)
            if bool(getattr(args, 'active_grasp_feedback', False)):
                from .active_grasp_feedback import ActiveGraspFeedback
                args._active_feedback_session = ActiveGraspFeedback(args, plan_and_maybe_execute_grasp)
            mark_experiment_stage(args, "active_view")
            active_result = run_optional_active_view(
                arm,
                args,
                target_cloud,
                scene_cloud,
                T_base_camera,
                inst,
                out_dir,
            )
            record_experiment_views(args, active_result)
            record_workflow_motion(args, active_result.motion_sent)
            if bool(getattr(args, 'active_grasp_feedback', False)):
                feedback = args._active_feedback_session
                feedback.write_summary(out_dir)
                if (feedback.valid_sides < 1 or active_result.kept_view_count < 2
                        or not active_result.restore_ok or feedback.expired):
                    return with_workflow_motion(args, GraspAttemptResult(
                        GraspAttemptStatus.CAPTURE_FAILED, False,
                        'feedback requires one retained side at a verified center within the observation/time budget'))
            if bool(
                getattr(args, "_force_active_center_relocalize", False)
            ):
                center_relocalized = bool(
                    active_result.reason in {"center_view_only", "fused"}
                    and active_result.kept_view_count >= 1
                    and active_result.restore_ok
                )
                if not center_relocalized:
                    detail = (
                        "post-clear target-centered relocation was not "
                        "verified: "
                        f"reason={active_result.reason} "
                        f"kept_views={active_result.kept_view_count} "
                        f"restore_ok={active_result.restore_ok}"
                    )
                    print(
                        f"[active][memory][abort] {detail}; target grasp is "
                        "forbidden"
                    )
                    return with_workflow_motion(
                        args,
                        GraspAttemptResult(
                            GraspAttemptStatus.CLEAR_FAILED,
                            False,
                            detail,
                        ),
                    )
                args._force_active_center_relocalize = False
                print(
                    "[active][memory] target-centered YOLO relocation verified; "
                    "updated centered cloud will be used for grasp planning"
                )
            if bool(getattr(args, "_force_active_multiview", False)) and (
                not active_result.used_multiview
                or active_result.kept_view_count < 2
            ):
                detail = (
                    "amodal occlusion required active multi-view, but at least "
                    "two valid views were not obtained: "
                    f"reason={active_result.reason} "
                    f"kept_views={active_result.kept_view_count}"
                )
                print(f"[active][abort] {detail}; target grasp is forbidden")
                return with_workflow_motion(
                    args,
                    GraspAttemptResult(
                        GraspAttemptStatus.CLEAR_FAILED, False, detail
                    ),
                )
            if active_result.motion_sent and not active_result.restore_ok:
                detail = (
                    "active-view workflow did not restore a verified "
                    f"observation configuration: reason={active_result.reason}"
                )
                print(f"[active][abort] {detail}; target grasp is forbidden")
                return with_workflow_motion(
                    args,
                    GraspAttemptResult(
                        GraspAttemptStatus.EXECUTION_FAILED,
                        True,
                        detail,
                    ),
                )
            required_views = int(getattr(args, "grasp_required_view_count", 0))
            if required_views and (
                active_result.kept_view_count < required_views
                or active_result.reason not in {"center_view_only", "fused"}
                or not active_result.restore_ok
            ):
                detail = (
                    f"comparison requires {required_views} retained target view(s) "
                    f"at a verified center, got {active_result.kept_view_count}: "
                    f"{active_result.reason}; grasp skipped"
                )
                print(f"[active][comparison][abort] {detail}")
                return with_workflow_motion(
                    args, GraspAttemptResult(GraspAttemptStatus.CAPTURE_FAILED, False, detail)
                )
            planned_target_cloud = active_result.target_cloud
            planned_scene_cloud = active_result.scene_cloud
            target_plan_kwargs = {}
            active_environment_map = getattr(
                active_result, "motion_environment_map", None
            )
            if active_environment_map is not None:
                target_plan_kwargs["motion_environment_map"] = (
                    active_environment_map
                )
            retry_suffix = (
                ""
                if target_empty_retries_used == 0
                else f"_retry{target_empty_retries_used}"
            )
            mark_experiment_stage(args, "grasp_planning")
            target_result = plan_and_maybe_execute_grasp(
                arm,
                args,
                planned_target_cloud,
                planned_scene_cloud,
                active_result.T_base_camera,
                bgr,
                intr,
                out_dir,
                inst,
                plan_label=f"target{retry_suffix}",
                place_xyz_override=None,
                confirmation_word="GRASP",
                motion_scene_cloud=(
                    active_result.motion_scene_cloud
                    if active_result.motion_scene_cloud is not None
                    else active_result.scene_cloud
                ),
                motion_environment_cloud=(
                    active_result.motion_environment_cloud
                    if active_result.motion_environment_cloud is not None
                    else getattr(args, "_last_obstacle_cloud", None)
                ),
                **target_plan_kwargs,
            )
            target_result = with_workflow_motion(args, target_result)
            print(
                f"[result] target={inst.get('label', 'unknown')} "
                f"status={target_result.status.value} "
                f"motion_sent={target_result.motion_sent} "
                f"detail={target_result.detail}"
            )
            if (
                target_result.status is GraspAttemptStatus.SUCCESS
                and callable(hooks.after_grasp_lifted)
            ):
                log_state = getattr(args, "_experiment_log_state", None)
                if log_state is not None and log_state["grasp_stability"]["stable"] is not True:
                    print(
                        "[handoff][abort] post-lift gripper feedback does not confirm "
                        "object retention; keep the current hold and skip visual servo"
                    )
                    return GraspAttemptResult(
                        GraspAttemptStatus.EXECUTION_FAILED,
                        target_result.motion_sent,
                        "post-lift object retention unconfirmed; visual servo not started",
                    )
                try:
                    hooks.after_grasp_lifted(arm, args)
                except Exception as exc:
                    print(
                        "[handoff][warn] visual servo after successful lift "
                        f"failed: {type(exc).__name__}: {exc}"
                    )
            if not (
                target_result.status is GraspAttemptStatus.EXECUTION_FAILED
                and target_result.empty_grasp_recovered_to_initial
            ):
                return target_result
            if target_empty_retries_used >= target_empty_retry_limit:
                print(
                    "[target][retry][abort] confirmed empty grasp returned "
                    "safely, but the automatic replan limit "
                    f"{target_empty_retry_limit} is exhausted"
                )
                return target_result
            if not callable(recapture_same_target):
                print(
                    "[target][retry][abort] same-target recapture hook is "
                    "unavailable; refusing to replan from stale perception"
                )
                return target_result

            target_empty_retries_used += 1
            print(
                "[target][retry] empty grasp recovered to the verified initial "
                "observation pose; recapture the same selected target and "
                f"restart active-view/grasp planning "
                f"({target_empty_retries_used}/{target_empty_retry_limit})"
            )
            recaptured = recapture_same_target(
                args,
                reference_mask,
                reference_inst,
            )
            if recaptured is None:
                print(
                    "[target][retry][abort] the original target could not be "
                    "strictly re-identified; no further motion is allowed"
                )
                return target_result
            bgr, target_cloud, scene_cloud, out_dir, inst, intr = recaptured
