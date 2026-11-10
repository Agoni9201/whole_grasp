"""Execution of frozen, collision-prevalidated grasp motion plans."""

import json
import os

import numpy as np

from .stop_policy import request_ordinary_trajectory_stop
from .types import (
    ControlledMotionStopTriggered,
    MotionNotReached,
    PlanNotExecutable,
    SafetyStopTriggered,
    TorqueCollisionStopTriggered,
    UnsafeMotionTarget,
)


__all__ = [
    "_collision_context_for_planned_stage",
    "execute_prevalidated_stage",
    "execute_prevalidated_stages",
    "recover_after_torque_stop",
    "save_grasp_motion_plan",
]


_SMOOTH_HANDOFF_PHASES = {
    "free_open",
    "pickup_lift",
    "carrying",
    "released_free",
}


def _smooth_handoff_pair_eligible(previous_stage, stage, args):
    """Allow non-stopping handoff only across two proven open-space edges."""
    if not bool(getattr(args, "smooth_open_space_handoff", True)):
        return False
    previous_phase = str(getattr(previous_stage, "phase", ""))
    phase = str(getattr(stage, "phase", ""))
    if phase != previous_phase or phase not in _SMOOTH_HANDOFF_PHASES:
        return False
    threshold = max(
        float(getattr(args, "smooth_handoff_min_extra_clearance", 0.020)),
        0.0,
    )
    margins = (
        getattr(previous_stage, "path_clearance_margin", None),
        getattr(stage, "path_clearance_margin", None),
    )
    if any(value is None for value in margins):
        return False
    margins = tuple(float(value) for value in margins)
    return bool(
        all(np.isfinite(value) and value >= threshold for value in margins)
    )


def _install_smooth_handoff_state(args, previous_stage, enabled):
    """Install per-command state and return values needed for restoration."""
    marker = object()
    names = (
        "_smooth_open_space_handoff_active",
        "_smooth_open_space_handoff_previous_target",
    )
    previous = {name: getattr(args, name, marker) for name in names}
    if enabled:
        args._smooth_open_space_handoff_active = True
        args._smooth_open_space_handoff_previous_target = np.asarray(
            previous_stage.joints, dtype=float
        ).reshape(7).copy()
    else:
        args._smooth_open_space_handoff_active = False
        if hasattr(args, "_smooth_open_space_handoff_previous_target"):
            delattr(args, "_smooth_open_space_handoff_previous_target")
    return marker, previous


def _restore_smooth_handoff_state(args, marker, previous):
    for name, value in previous.items():
        if value is marker:
            if hasattr(args, name):
                delattr(args, name)
        else:
            setattr(args, name, value)


def _resolve_motion_planning_hook(name, hook):
    if hook is not None:
        return hook
    from . import motion_planning

    return getattr(motion_planning, name)


def _resolve_collision_hook(name, hook):
    if hook is not None:
        return hook
    from . import collision

    return getattr(collision, name)


def _resolve_arm_execution_hook(name, hook):
    if hook is not None:
        return hook
    from . import arm_execution

    return getattr(arm_execution, name)


def _resolve_pointcloud_hook(name, hook):
    if hook is not None:
        return hook
    from . import pointcloud_fusion

    return getattr(pointcloud_fusion, name)


def _resolve_safe_stop_hook(hook):
    if hook is not None:
        return hook
    from real_nero_point_test import safe_stop

    return safe_stop


def _collision_context_for_planned_stage(
    plan, stage, *, motion_context_hook=None
):
    motion_context_fn = _resolve_motion_planning_hook(
        "_motion_context", motion_context_hook
    )
    phase = str(stage.phase)
    return motion_context_fn(
        plan.safety_scene,
        phase,
        attached_target_tcp=(
            plan.attached_target_tcp
            if phase in {"carrying", "pickup_lift", "support_release"}
            else None
        ),
        placed_target_base=(
            plan.placed_target_base
            if phase in {"released_retreat", "released_free"}
            else None
        ),
        min_tcp_z=stage.min_tcp_z,
        runtime_min_tcp_z_slack=stage.runtime_min_tcp_z_slack,
        initial_support_release=(
            plan.initial_support_release
            if phase == "support_release"
            else None
        ),
        closed_gripper_width=getattr(plan, "closed_gripper_width", None),
        require_support_release_latch_at_endpoint=bool(
            getattr(
                stage,
                "require_support_release_latch_at_endpoint",
                True,
            )
        ),
    )


def execute_prevalidated_stage(
    arm,
    stage,
    expected_start_joints,
    plan,
    args,
    *,
    collision_context_for_planned_stage_hook=None,
    validate_motion_joint_path_hook=None,
    move_flange_pose_hook=None,
    safe_stop_hook=None,
):
    """Recheck from measured joints, then execute only the stored TRAC-IK target."""
    collision_context_fn = (
        _collision_context_for_planned_stage
        if collision_context_for_planned_stage_hook is None
        else collision_context_for_planned_stage_hook
    )
    validate_path_fn = _resolve_collision_hook(
        "validate_motion_joint_path", validate_motion_joint_path_hook
    )
    move_flange_pose_fn = _resolve_arm_execution_hook(
        "move_flange_pose", move_flange_pose_hook
    )
    safe_stop_fn = _resolve_safe_stop_hook(safe_stop_hook)
    actual_start = np.asarray(arm.get_joint_angles(), dtype=float).reshape(7)
    expected_start = np.asarray(expected_start_joints, dtype=float).reshape(7)
    start_error_deg = float(
        np.rad2deg(np.max(np.abs(actual_start - expected_start)))
    )
    allowed_start_error = float(
        getattr(args, "motion_stage_start_joint_match_deg", 2.0)
    )
    if start_error_deg > allowed_start_error:
        raise MotionNotReached(
            f"{stage.label} measured start joints differ from the prevalidated node by "
            f"{start_error_deg:.2f}deg > {allowed_start_error:.2f}deg; "
            "no new target was sent"
        )
    context = collision_context_fn(plan, stage)
    from .gripper_geometry import urdf_gripper_enabled, read_sdk_gripper_opening
    if urdf_gripper_enabled(args):
        from dataclasses import replace
        context = replace(context, gripper_mesh_opening=read_sdk_gripper_opening(arm, args))
    validate_path_fn(
        actual_start,
        stage.joints,
        context,
        args,
        f"{stage.label}_live_preflight",
    )
    marker = object()
    previous_context = getattr(args, "_runtime_motion_collision_context", marker)
    previous_support_state = getattr(
        args, "_runtime_support_release_state", marker
    )
    diagnostic_names = (
        "_runtime_motion_stage_label",
        "_runtime_motion_stage_start_joints",
        "_runtime_motion_stage_target_joints",
        "_runtime_torque_guard_state",
        "_runtime_torque_unavailable_stage",
        "_runtime_finger_clearance_guard_state",
        "_runtime_joint_path_guard_state",
    )
    previous_diagnostic = {
        name: getattr(args, name, marker) for name in diagnostic_names
    }
    args._runtime_motion_collision_context = context
    args._runtime_motion_stage_label = str(stage.label)
    args._runtime_motion_stage_start_joints = actual_start.copy()
    args._runtime_motion_stage_target_joints = np.asarray(
        stage.joints, dtype=float
    ).reshape(7).copy()
    if context.initial_support_release is not None:
        args._runtime_support_release_state = {
            "context_id": id(context),
            "last_rise": None,
            "max_rise": -float("inf"),
            "strict_latched": False,
        }
    elif hasattr(args, "_runtime_support_release_state"):
        delattr(args, "_runtime_support_release_state")
    try:
        try:
            return move_flange_pose_fn(
                arm,
                stage.flange_pose,
                args,
                stage.label,
                planned_joint_target=stage.joints,
            )
        except MotionNotReached as exc:
            # A timeout or an unstarted/retried command does not prove the
            # controller stopped moving.  Keep the frozen runtime map installed
            # while requesting an ordinary terminate-and-hold.  Electronic
            # e-stop is reserved for failure to confirm that ordinary stop.
            msg = (
                f"{stage.label} did not reach its prevalidated joint target; "
                "requesting ordinary trajectory termination before aborting the workflow"
            )
            request_ordinary_trajectory_stop(
                arm,
                msg,
                safe_stop_hook=safe_stop_fn,
            )
            raise ControlledMotionStopTriggered(msg) from exc
    finally:
        if previous_context is marker:
            if hasattr(args, "_runtime_motion_collision_context"):
                delattr(args, "_runtime_motion_collision_context")
        else:
            args._runtime_motion_collision_context = previous_context
        if previous_support_state is marker:
            if hasattr(args, "_runtime_support_release_state"):
                delattr(args, "_runtime_support_release_state")
        else:
            args._runtime_support_release_state = previous_support_state
        for name in diagnostic_names:
            previous_value = previous_diagnostic[name]
            if previous_value is marker:
                if hasattr(args, name):
                    delattr(args, name)
            else:
                setattr(args, name, previous_value)


def execute_prevalidated_stages(
    arm,
    stages,
    expected_start_joints,
    plan,
    args,
    *,
    execute_prevalidated_stage_hook=None,
    collision_context_for_planned_stage_hook=None,
    validate_motion_joint_path_hook=None,
    modeled_tcp_pose_for_joints_hook=None,
    wait_for_joint_settle_hook=None,
    safe_stop_hook=None,
):
    execute_stage_fn = (
        execute_prevalidated_stage
        if execute_prevalidated_stage_hook is None
        else execute_prevalidated_stage_hook
    )
    expected = np.asarray(expected_start_joints, dtype=float).reshape(7)
    last_flange = None
    stages = tuple(stages)
    smooth_handoffs = 0
    for stage_index, stage in enumerate(stages):
        previous_stage = stages[stage_index - 1] if stage_index > 0 else None
        smooth_handoff = bool(
            previous_stage is not None
            and _smooth_handoff_pair_eligible(previous_stage, stage, args)
        )
        marker, previous_handoff_state = _install_smooth_handoff_state(
            args,
            previous_stage if previous_stage is not None else stage,
            smooth_handoff,
        )
        boundary_marker = object()
        previous_grasp_boundary = getattr(
            args, "_grasp_close_action_boundary_active", boundary_marker
        )
        args._grasp_close_action_boundary_active = bool(
            stage_index == len(stages) - 1
            and str(getattr(stage, "phase", "")) == "target_approach"
        )
        try:
            try:
                if smooth_handoff:
                    smooth_handoffs += 1
                    margins_mm = (
                        float(previous_stage.path_clearance_margin) * 1000.0,
                        float(stage.path_clearance_margin) * 1000.0,
                    )
                    print(
                        f"[motion][handoff] {previous_stage.label} -> "
                        f"{stage.label} phase={stage.phase} extra_clearance="
                        f"[{margins_mm[0]:.1f}, {margins_mm[1]:.1f}]mm; "
                        "skip internal full-stop barrier"
                    )
                last_flange = execute_stage_fn(
                    arm, stage, expected, plan, args
                )
                expected = np.asarray(stage.joints, dtype=float).reshape(7)
            finally:
                _restore_smooth_handoff_state(
                    args, marker, previous_handoff_state
                )
                if previous_grasp_boundary is boundary_marker:
                    if hasattr(args, "_grasp_close_action_boundary_active"):
                        delattr(args, "_grasp_close_action_boundary_active")
                else:
                    args._grasp_close_action_boundary_active = (
                        previous_grasp_boundary
                    )
        except UnsafeMotionTarget as primary_error:
            if (
                str(getattr(stage, "phase", "")) != "carrying"
                or not bool(getattr(args, "post_grasp_runtime_replan", False))
                or bool(getattr(plan, "validation_only", False))
            ):
                raise
            actual = np.asarray(arm.get_joint_angles(), dtype=float).reshape(7)
            validate_path_fn = _resolve_collision_hook(
                "validate_motion_joint_path", validate_motion_joint_path_hook
            )
            context_fn = (
                _collision_context_for_planned_stage
                if collision_context_for_planned_stage_hook is None
                else collision_context_for_planned_stage_hook
            )
            context = context_fn(plan, stage)
            modeled_tcp_fn = modeled_tcp_pose_for_joints_hook
            if modeled_tcp_fn is None:
                modeled_tcp_fn = _resolve_motion_planning_hook(
                    "_modeled_tcp_pose_for_joints", None
                )
            from .post_grasp_rrt import replan_post_grasp_remaining_route

            final_stage = stages[-1]
            try:
                replacement = replan_post_grasp_remaining_route(
                    arm,
                    actual,
                    final_stage.joints,
                    context,
                    args,
                    f"{stage.label}_runtime_replan",
                    validate_motion_joint_path=validate_path_fn,
                    modeled_tcp_pose_for_joints=modeled_tcp_fn,
                )
            except (PlanNotExecutable, UnsafeMotionTarget, ValueError) as replan_error:
                print(
                    f"[motion][replan][reject] primary stage={stage.label}; "
                    f"remaining-route RRT failed: {replan_error}"
                )
                raise primary_error
            print(
                f"[motion][replan] primary stage={stage.label} failed live "
                f"preflight; planned {len(replacement)} replacement stage(s) "
                "from measured joints to the frozen place node"
            )
            for replacement_stage in replacement:
                last_flange = execute_stage_fn(
                    arm, replacement_stage, actual, plan, args
                )
                actual = np.asarray(
                    replacement_stage.joints, dtype=float
                ).reshape(7)
            expected = actual
            break
    # Every hardware route ends at a real action boundary (capture, gripper
    # command, or the next phase), so require measured encoder settling even
    # when no internal smooth handoff happened.  Offline/direct callers that
    # do not inject the live settle helper retain the old behavior unless they
    # explicitly used smooth handoff.
    if stages and (wait_for_joint_settle_hook is not None or smooth_handoffs):
        settle_fn = _resolve_arm_execution_hook(
            "_wait_for_joint_settle", wait_for_joint_settle_hook
        )
        final_stage = stages[-1]
        context_fn = (
            _collision_context_for_planned_stage
            if collision_context_for_planned_stage_hook is None
            else collision_context_for_planned_stage_hook
        )
        marker = object()
        previous_context = getattr(
            args, "_runtime_motion_collision_context", marker
        )
        # The runtime facade injects a settle helper with the measured-joint
        # point-cloud guard.  Direct/offline callers still get the settle
        # barrier, but do not install an unusable live context.
        if wait_for_joint_settle_hook is not None:
            args._runtime_motion_collision_context = context_fn(plan, final_stage)
        try:
            try:
                settle_fn(
                    arm,
                    args,
                    f"{final_stage.label}_phase_boundary",
                )
            except MotionNotReached as exc:
                msg = (
                    f"{final_stage.label} did not settle at the action/phase "
                    "boundary"
                )
                request_ordinary_trajectory_stop(
                    arm,
                    msg,
                    safe_stop_hook=_resolve_safe_stop_hook(safe_stop_hook),
                )
                raise ControlledMotionStopTriggered(msg) from exc
        finally:
            if previous_context is marker:
                if hasattr(args, "_runtime_motion_collision_context"):
                    delattr(args, "_runtime_motion_collision_context")
            else:
                args._runtime_motion_collision_context = previous_context
        print(
            f"[motion][boundary] route completed with {smooth_handoffs} "
            "continuous open-space internal handoff(s); endpoint encoder "
            "motion fully settled before the next capture/gripper/action"
        )
    return last_flange, expected


def recover_after_torque_stop(
    arm,
    stop,
    args,
    *,
    validate_motion_joint_path_hook=None,
    wait_until_joints_reached_hook=None,
):
    """Retract only over the just-completed stage's verified incoming edge.

    The normal stop has already been confirmed by the torque guard.  This is
    deliberately not a free-space replan: it validates the measured-current
    to the recorded stage start against the same frozen map, lowers controller
    speed temporarily, and keeps all geometric/table guards live.  A failed
    recovery remains terminal rather than issuing a speculative second move.
    """
    if not isinstance(stop, TorqueCollisionStopTriggered):
        raise TypeError("recover_after_torque_stop requires TorqueCollisionStopTriggered")
    validate_path_fn = _resolve_collision_hook(
        "validate_motion_joint_path", validate_motion_joint_path_hook
    )
    wait_fn = _resolve_arm_execution_hook(
        "wait_until_joints_reached", wait_until_joints_reached_hook
    )
    current = np.asarray(arm.get_joint_angles(), dtype=float).reshape(7)
    retreat = np.asarray(stop.retreat_joints, dtype=float).reshape(7)
    label = f"{stop.stage_label}_torque_retract"
    validate_path_fn(current, retreat, stop.context, args, f"{label}_live_preflight")

    marker = object()
    previous_context = getattr(args, "_runtime_motion_collision_context", marker)
    previous_label = getattr(args, "_runtime_motion_stage_label", marker)
    previous_start = getattr(args, "_runtime_motion_stage_start_joints", marker)
    previous_target = getattr(args, "_runtime_motion_stage_target_joints", marker)
    previous_suppressed = getattr(args, "_runtime_torque_guard_suppressed", marker)
    previous_speed = None
    speed_changed = False
    try:
        args._runtime_motion_collision_context = stop.context
        args._runtime_motion_stage_label = label
        args._runtime_motion_stage_start_joints = current.copy()
        args._runtime_motion_stage_target_joints = retreat.copy()
        args._runtime_torque_guard_suppressed = True
        allow_recovery = getattr(arm, "allow_controlled_recovery", None)
        if not callable(allow_recovery):
            raise RuntimeError("arm adapter cannot unlock a confirmed controlled stop")
        allow_recovery()
        requested_speed = float(
            getattr(args, "motion_runtime_torque_retract_speed_percent", 2.0)
        )
        driver = getattr(arm, "arm", None)
        speed_setter = getattr(driver, "set_speed_percent", None)
        if requested_speed > 0.0 and callable(speed_setter):
            previous_speed = float(getattr(args, "nero_speed_percent", requested_speed))
            speed_setter(requested_speed)
            speed_changed = True
        print(
            f"[torque][retract] {label}: reversing verified edge at "
            f"{requested_speed if requested_speed > 0.0 else float(getattr(args, 'nero_speed_percent', 0.0)):.1f}% speed"
        )
        arm.move_j(retreat)
        ok, _last_joints, detail = wait_fn(arm, retreat, args, label)
        if not ok:
            raise MotionNotReached(f"{label} did not reach its verified retreat target: {detail}")
        print(f"[torque][retract] {label} reached: {detail}")
        return True
    except Exception as exc:
        # A recovery move may have been sent.  Stop it normally first; only
        # escalate when the adapter cannot confirm that physical motion ceased.
        try:
            response = arm.terminate_motion(fallback_joints=arm.get_joint_angles())
            print(f"[torque][retract] ordinary termination confirmed after recovery failure: {response}")
        except Exception as stop_exc:
            from real_nero_point_test import safe_stop

            msg = (
                f"{label} failed ({type(exc).__name__}: {exc}); ordinary stop "
                f"was unconfirmed ({type(stop_exc).__name__}: {stop_exc})"
            )
            safe_stop(arm, msg)
            raise SafetyStopTriggered(msg) from stop_exc
        raise
    finally:
        if speed_changed and previous_speed is not None:
            try:
                getattr(arm, "arm").set_speed_percent(previous_speed)
            except Exception as exc:
                print(f"[torque][retract][warn] failed to restore speed: {exc}")
        finish_recovery = getattr(arm, "finish_controlled_recovery", None)
        if callable(finish_recovery):
            finish_recovery()
        for name, previous in (
            ("_runtime_motion_collision_context", previous_context),
            ("_runtime_motion_stage_label", previous_label),
            ("_runtime_motion_stage_start_joints", previous_start),
            ("_runtime_motion_stage_target_joints", previous_target),
            ("_runtime_torque_guard_suppressed", previous_suppressed),
        ):
            if previous is marker:
                if hasattr(args, name):
                    delattr(args, name)
            else:
                setattr(args, name, previous)


def save_grasp_motion_plan(
    out_dir,
    plan,
    *,
    save_cloud_hook=None,
    make_cloud_from_points_colors_hook=None,
):
    save_cloud_fn = _resolve_pointcloud_hook(
        "save_cloud", save_cloud_hook
    )
    make_cloud_fn = _resolve_pointcloud_hook(
        "_make_cloud_from_points_colors",
        make_cloud_from_points_colors_hook,
    )

    def serialize_stage(stage):
        return {
            "label": str(stage.label),
            "phase": str(stage.phase),
            "tcp_pose": np.asarray(stage.tcp_pose, dtype=float).reshape(6).tolist(),
            "flange_pose": np.asarray(stage.flange_pose, dtype=float).reshape(6).tolist(),
            "joints_deg": np.rad2deg(
                np.asarray(stage.joints, dtype=float).reshape(7)
            ).tolist(),
            "min_tcp_z": (
                None if stage.min_tcp_z is None else float(stage.min_tcp_z)
            ),
            "require_support_release_latch_at_endpoint": bool(
                getattr(
                    stage,
                    "require_support_release_latch_at_endpoint",
                    True,
                )
            ),
            "path_clearance_margin": (
                None
                if getattr(stage, "path_clearance_margin", None) is None
                else float(stage.path_clearance_margin)
            ),
        }

    environment_map = getattr(plan.safety_scene, "environment_map", None)
    generation = int(getattr(plan.safety_scene, "generation", 0))
    payload = {
        "map_id": str(plan.safety_scene.map_id),
        "map_generation": generation,
        "dynamic_environment": (
            None
            if environment_map is None
            else {
                "map_id": str(
                    getattr(environment_map, "map_id", "dynamic_environment")
                ),
                "generation": int(
                    getattr(environment_map, "generation", generation)
                ),
                "frame_id": str(
                    getattr(environment_map, "frame_id", "base_link")
                ),
                "voxel_size": float(
                    getattr(environment_map, "voxel_size", 0.0)
                ),
                "source_view_names": list(
                    getattr(environment_map, "source_view_names", ())
                ),
                "hard_environment": int(
                    np.asarray(
                        getattr(
                            environment_map,
                            "hard_environment_base",
                            np.zeros((0, 3)),
                        )
                    ).reshape(-1, 3).shape[0]
                ),
                "uncertain_environment": int(
                    np.asarray(
                        getattr(
                            environment_map,
                            "uncertain_environment_base",
                            np.zeros((0, 3)),
                        )
                    ).reshape(-1, 3).shape[0]
                ),
            }
        ),
        "start_joints_deg": np.rad2deg(
            np.asarray(plan.start_joints, dtype=float).reshape(7)
        ).tolist(),
        "validation_only": bool(
            getattr(plan, "validation_only", False)
        ),
        "closed_gripper_width": (
            None
            if getattr(plan, "closed_gripper_width", None) is None
            else float(plan.closed_gripper_width)
        ),
        "safety_points": {
            "full_scene": int(
                np.asarray(plan.safety_scene.full_scene_base).reshape(-1, 3).shape[0]
            ),
            "environment": int(
                np.asarray(plan.safety_scene.environment_base).reshape(-1, 3).shape[0]
            ),
            "target": int(
                np.asarray(plan.safety_scene.target_base).reshape(-1, 3).shape[0]
            ),
        },
        "before_close": [serialize_stage(stage) for stage in plan.before_close],
        "after_close": [serialize_stage(stage) for stage in plan.after_close],
        "after_release": [serialize_stage(stage) for stage in plan.after_release],
        "empty_recovery": [serialize_stage(stage) for stage in plan.empty_recovery],
        "initial_support_release": (
            None
            if plan.initial_support_release is None
            else {
                "support_points": int(
                    np.asarray(
                        plan.initial_support_release.support_points_base
                    ).reshape(-1, 3).shape[0]
                ),
                "strict_environment_points": int(
                    np.asarray(
                        plan.initial_support_release.strict_environment_base
                    ).reshape(-1, 3).shape[0]
                ),
                "tool_contact_environment_points": (
                    None
                    if plan.initial_support_release.tool_contact_environment_base
                    is None
                    else int(
                        np.asarray(
                            plan.initial_support_release.tool_contact_environment_base
                        ).reshape(-1, 3).shape[0]
                    )
                ),
                "target_contact_points": (
                    None
                    if plan.initial_support_release.target_contact_points_base
                    is None
                    else int(
                        np.asarray(
                            plan.initial_support_release.target_contact_points_base
                        ).reshape(-1, 3).shape[0]
                    )
                ),
                "start_tcp_pose": np.asarray(
                    plan.initial_support_release.start_tcp_pose, dtype=float
                ).reshape(6).tolist(),
                "strict_after_rise": float(
                    plan.initial_support_release.strict_after_rise
                ),
            }
        ),
    }
    path = os.path.join(out_dir, "whole_workflow_motion_plan.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(f"[motion][plan] saved full joint/TCP sequence={path}")
    try:
        save_cloud_fn(
            os.path.join(out_dir, "motion_safety_environment_base.ply"),
            make_cloud_fn(plan.safety_scene.environment_base),
        )
        save_cloud_fn(
            os.path.join(out_dir, "motion_safety_target_base.ply"),
            make_cloud_fn(plan.safety_scene.target_base),
        )
        if environment_map is not None:
            save_cloud_fn(
                os.path.join(
                    out_dir, "motion_environment_hard_base.ply"
                ),
                make_cloud_fn(environment_map.hard_environment_base),
            )
            save_cloud_fn(
                os.path.join(
                    out_dir, "motion_environment_uncertain_base.ply"
                ),
                make_cloud_fn(environment_map.uncertain_environment_base),
            )
    except Exception as exc:
        print(
            f"[motion][plan][warn] failed to save safety point clouds: "
            f"{type(exc).__name__}: {exc}"
        )
