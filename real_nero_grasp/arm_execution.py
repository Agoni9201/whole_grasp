"""Nero flange-pose execution and idempotent joint-command retry.

The workflow facade injects its live collision checker into the guarded wait
helpers.  Keeping that dependency explicit avoids importing the high-level
runtime here while preserving fail-closed execution when a frozen collision
context is active.
"""

from copy import copy
import time

import numpy as np
from scipy.spatial.transform import Rotation as R

from real_nero_grasp.collision import (
    table_guard_enabled,
    table_guard_min_tcp_z,
    validate_motion_joint_path,
)
from real_nero_grasp.ik_solver import (
    _clip_numerical_joint_limit_overshoot,
    _get_ik_solver,
    _solve_ik_for_flange_pose,
)
from real_nero_grasp.stop_policy import request_ordinary_trajectory_stop
from real_nero_grasp.types import (
    ControlledMotionStopTriggered,
    MotionNotReached,
    PlanNotExecutable,
    SafetyStopTriggered,
    UnsafeMotionTarget,
)
from tools.real_nero_point_test import (
    pose_error,
    safe_stop,
    wait_until_joints_reached as _base_wait_until_joints_reached,
    wait_until_pose_reached as _base_wait_until_pose_reached,
    wrap_angle_error,
)


__all__ = [
    "check_table_guard_current",
    "wait_until_pose_reached",
    "_enum_numeric",
    "_arm_motion_status_snapshot",
    "_wait_for_joint_settle",
    "_wait_for_joint_motion_start",
    "_send_joint_target_with_start_retry",
    "wait_until_joints_reached",
    "move_flange_pose_ik_joint",
    "move_flange_pose",
]


def _runtime_collision_hook_required(arm, args, label, joints=None):
    """Fail closed if a facade forgets the live collision-check dependency."""
    context = getattr(args, "_runtime_motion_collision_context", None)
    if context is None or not bool(
        getattr(args, "motion_runtime_collision_guard", True)
    ):
        return None
    msg = (
        f"{label} runtime collision guard hook is unavailable while a frozen "
        "motion collision context is active"
    )
    setattr(arm, "_safety_stop_reason", msg)
    request_ordinary_trajectory_stop(
        arm, msg, safe_stop_hook=safe_stop, fallback_joints=joints,
        fault_context=args,
    )
    raise ControlledMotionStopTriggered(msg)


def check_table_guard_current(
    arm,
    args,
    label,
    *,
    table_guard_enabled_hook=None,
    table_guard_min_tcp_z_hook=None,
    safe_stop_hook=None,
):
    table_guard_is_enabled = (
        table_guard_enabled
        if table_guard_enabled_hook is None
        else table_guard_enabled_hook
    )
    table_guard_min_z = (
        table_guard_min_tcp_z
        if table_guard_min_tcp_z_hook is None
        else table_guard_min_tcp_z_hook
    )
    safe_stop_impl = safe_stop if safe_stop_hook is None else safe_stop_hook
    if not table_guard_is_enabled(args):
        return
    min_z = table_guard_min_z(args)
    try:
        tcp_pose = np.asarray(arm.get_tcp_pose(), dtype=float).reshape(6)
    except Exception as exc:
        if bool(getattr(args, "table_guard_stop_on_read_error", False)):
            msg = f"{label} failed to read TCP pose during table guard: {type(exc).__name__}: {exc}"
            request_ordinary_trajectory_stop(
                arm,
                msg,
                safe_stop_hook=safe_stop_impl,
            )
            raise ControlledMotionStopTriggered(msg) from exc
        print(f"[table_guard][warn] failed to read TCP pose: {type(exc).__name__}: {exc}")
        return
    tcp_z = float(tcp_pose[2])
    if tcp_z < min_z:
        msg = (
            f"{label} current TCP z={tcp_z:.4f}m below table guard min_tcp_z={min_z:.4f}m. "
            "Electronic emergency stop was requested."
        )
        safe_stop_impl(arm, msg)
        raise SafetyStopTriggered(msg)


def wait_until_pose_reached(
    nero,
    target_pose,
    args,
    label,
    *,
    table_guard_enabled_hook=None,
    base_wait_until_pose_reached_hook=None,
    check_table_guard_current_hook=None,
    check_runtime_collision_current_hook=None,
    pose_error_hook=None,
):
    table_guard_is_enabled = (
        table_guard_enabled
        if table_guard_enabled_hook is None
        else table_guard_enabled_hook
    )
    base_wait = (
        _base_wait_until_pose_reached
        if base_wait_until_pose_reached_hook is None
        else base_wait_until_pose_reached_hook
    )
    check_table = (
        check_table_guard_current
        if check_table_guard_current_hook is None
        else check_table_guard_current_hook
    )
    check_collision = (
        _runtime_collision_hook_required
        if check_runtime_collision_current_hook is None
        else check_runtime_collision_current_hook
    )
    compute_pose_error = pose_error if pose_error_hook is None else pose_error_hook
    if not table_guard_is_enabled(args) and getattr(
        args, "_runtime_motion_collision_context", None
    ) is None:
        return base_wait(nero, target_pose, args, label)
    deadline = time.time() + float(args.move_wait_timeout)
    last_pose = None
    last_pos_err = float("inf")
    last_rpy_err = float("inf")
    while time.time() < deadline:
        last_pose = nero.get_flange_pose()
        check_table(nero, args, label)
        check_collision(nero, args, label)
        last_pos_err, last_rpy_err = compute_pose_error(last_pose, target_pose)
        if last_pos_err <= float(args.move_pos_tol) and last_rpy_err <= float(args.move_rpy_tol):
            print(
                f"[{label}] reached: pos_err={last_pos_err:.4f}m "
                f"rpy_err={np.rad2deg(last_rpy_err):.2f}deg"
            )
            return True, last_pose, last_pos_err, last_rpy_err
        time.sleep(float(args.move_poll_interval))

    print(
        f"[{label}][warn] not reached before timeout: "
        f"pos_err={last_pos_err:.4f}m rpy_err={np.rad2deg(last_rpy_err):.2f}deg"
    )
    if last_pose is not None:
        print(f"[{label}][warn] current_pose =", np.round(last_pose, 6).tolist())
        print(f"[{label}][warn] target_pose  =", np.round(np.asarray(target_pose, dtype=float), 6).tolist())
    return False, last_pose, last_pos_err, last_rpy_err


def _enum_numeric(value):
    raw = getattr(value, "value", value)
    try:
        return int(raw)
    except Exception:
        return None


def _arm_motion_status_snapshot(nero, args=None):
    """Return a compact, fail-closed status used to gate an idempotent retry."""
    driver = getattr(nero, "arm", None)
    getter = getattr(driver, "get_arm_status", None)
    if not callable(getter):
        return {"available": False, "normal": False, "reason": "status_api_unavailable"}
    try:
        raw = getter()
    except Exception as exc:
        return {
            "available": False,
            "normal": False,
            "reason": f"status_read_failed:{type(exc).__name__}",
        }
    msg = getattr(raw, "msg", raw)
    if msg is None:
        return {"available": False, "normal": False, "reason": "status_missing"}
    ctrl_mode = getattr(msg, "ctrl_mode", None)
    arm_status = getattr(msg, "arm_status", None)
    mode_feedback = getattr(msg, "mode_feedback", None)
    teach_status = getattr(msg, "teach_status", None)
    motion_status = getattr(msg, "motion_status", None)
    error_names = (
        "joint_1_angle_limit",
        "joint_2_angle_limit",
        "joint_3_angle_limit",
        "joint_4_angle_limit",
        "joint_5_angle_limit",
        "joint_6_angle_limit",
        "joint_7_angle_limit",
        "communication_status_joint_1",
        "communication_status_joint_2",
        "communication_status_joint_3",
        "communication_status_joint_4",
        "communication_status_joint_5",
        "communication_status_joint_6",
        "communication_status_joint_7",
    )
    err_status = getattr(msg, "err_status", None)
    active_errors = [
        name for name in error_names if bool(getattr(err_status, name, False))
    ]
    ctrl_code = _enum_numeric(ctrl_mode)
    arm_code = _enum_numeric(arm_status)
    mode_code = _enum_numeric(mode_feedback)
    teach_code = _enum_numeric(teach_status)
    motion_code = _enum_numeric(motion_status)

    status_timestamp = getattr(raw, "timestamp", None)
    try:
        status_timestamp = float(status_timestamp)
    except Exception:
        status_timestamp = float("nan")
    max_age_s = max(float(getattr(args, "move_status_max_age", 1.0)), 0.05)
    status_age_s = float("inf")
    if np.isfinite(status_timestamp) and status_timestamp > 0.0:
        status_age_s = float(time.time() - status_timestamp)
    status_fresh = bool(-0.5 <= status_age_s <= max_age_s)

    enable_getter = getattr(driver, "get_joints_enable_status_list", None)
    joints_enabled = []
    if callable(enable_getter):
        try:
            joints_enabled = [bool(value) for value in enable_getter()]
        except Exception:
            joints_enabled = []
    all_joints_enabled = bool(len(joints_enabled) == 7 and all(joints_enabled))

    safe_checks = {
        "status_fresh": status_fresh,
        "can_control": ctrl_code == 1,
        "arm_normal": arm_code == 0,
        "move_j_mode": mode_code == 1,
        "teaching_disabled": teach_code == 0,
        "motion_idle": motion_code == 0,
        "no_active_errors": not active_errors,
        "all_joints_enabled": all_joints_enabled,
    }
    normal = bool(all(safe_checks.values()))
    failed_checks = [name for name, passed in safe_checks.items() if not passed]
    return {
        "available": True,
        "normal": normal,
        "ctrl_mode": str(ctrl_mode),
        "ctrl_mode_code": ctrl_code,
        "arm_status": str(arm_status),
        "arm_status_code": arm_code,
        "mode_feedback": str(mode_feedback),
        "mode_feedback_code": mode_code,
        "teach_status": str(teach_status),
        "teach_status_code": teach_code,
        "motion_status": str(motion_status),
        "motion_status_code": motion_code,
        "status_age_s": None if not np.isfinite(status_age_s) else status_age_s,
        "status_max_age_s": max_age_s,
        "joints_enabled": joints_enabled,
        "active_errors": active_errors,
        "reason": "normal" if normal else "unsafe_for_retry:" + ",".join(failed_checks),
    }


def _wait_for_joint_motion_start(
    nero,
    q_before,
    q_target,
    args,
    label,
    *,
    wrap_angle_error_hook=None,
    check_table_guard_current_hook=None,
    check_runtime_collision_current_hook=None,
):
    wrap_error = (
        wrap_angle_error if wrap_angle_error_hook is None else wrap_angle_error_hook
    )
    check_table = (
        check_table_guard_current
        if check_table_guard_current_hook is None
        else check_table_guard_current_hook
    )
    check_collision = (
        _runtime_collision_hook_required
        if check_runtime_collision_current_hook is None
        else check_runtime_collision_current_hook
    )
    q_before = np.asarray(q_before, dtype=float).reshape(7)
    q_target = np.asarray(q_target, dtype=float).reshape(7)
    timeout = max(float(getattr(args, "move_start_timeout", 1.5)), 0.0)
    start_threshold = np.deg2rad(
        max(float(getattr(args, "move_start_threshold_deg", 0.20)), 0.01)
    )
    joint_tol = float(getattr(args, "move_joint_tol", np.deg2rad(2.0)))
    poll = max(float(getattr(args, "move_poll_interval", 0.25)), 0.02)
    deadline = time.time() + timeout
    last_joints = q_before.copy()
    moved = 0.0
    forward_progress = 0.0
    target_direction = wrap_error(q_target, q_before)
    target_direction_norm = float(np.linalg.norm(target_direction))
    if target_direction_norm > 1e-9:
        target_direction = target_direction / target_direction_norm
    target_err = float(np.linalg.norm(wrap_error(q_before, q_target)))
    while True:
        last_joints = np.asarray(nero.get_joint_angles(), dtype=float).reshape(7)
        check_table(nero, args, f"{label}_start")
        check_collision(nero, args, f"{label}_start", joints=last_joints)
        displacement = wrap_error(last_joints, q_before)
        moved = float(np.max(np.abs(displacement)))
        forward_progress = float(np.dot(displacement, target_direction))
        target_err = float(np.linalg.norm(wrap_error(last_joints, q_target)))
        # Merely seeing encoder motion is not proof that this command started:
        # the preceding waypoint can still be settling.  Require measurable
        # progress in the direction of the newly commanded joint target.
        if target_err <= joint_tol or forward_progress >= start_threshold:
            return True, last_joints, np.rad2deg(moved), np.rad2deg(target_err)
        if time.time() >= deadline:
            return False, last_joints, np.rad2deg(moved), np.rad2deg(target_err)
        time.sleep(poll)


def _wait_for_joint_settle(
    nero,
    args,
    label,
    *,
    check_table_guard_current_hook=None,
    check_runtime_collision_current_hook=None,
    wrap_angle_error_hook=None,
):
    """Wait until encoder motion from the preceding command has settled.

    A new ``move_j`` must not be sent while the previous waypoint is still
    coasting inside its relatively loose reached tolerance.  Otherwise that
    residual motion can be mistaken for acknowledgement of the new command.
    """
    check_table = (
        check_table_guard_current
        if check_table_guard_current_hook is None
        else check_table_guard_current_hook
    )
    check_collision = (
        _runtime_collision_hook_required
        if check_runtime_collision_current_hook is None
        else check_runtime_collision_current_hook
    )
    wrap_error = (
        wrap_angle_error if wrap_angle_error_hook is None else wrap_angle_error_hook
    )
    timeout = max(
        float(getattr(args, "move_precommand_settle_timeout", 1.5)), 0.0
    )
    required_samples = max(
        int(getattr(args, "move_precommand_settle_samples", 3)), 1
    )
    delta_limit = np.deg2rad(
        max(float(getattr(args, "move_precommand_settle_delta_deg", 0.05)), 0.001)
    )
    poll = max(
        float(
            getattr(
                args,
                "move_precommand_settle_poll",
                min(float(getattr(args, "move_poll_interval", 0.10)), 0.10),
            )
        ),
        0.02,
    )
    previous = np.asarray(nero.get_joint_angles(), dtype=float).reshape(7)
    deadline = time.time() + timeout
    stable_samples = 0
    last_delta = float("inf")
    while True:
        check_table(nero, args, f"{label}_settle")
        check_collision(nero, args, f"{label}_settle", joints=previous)
        if stable_samples >= required_samples:
            return previous
        if time.time() >= deadline:
            raise MotionNotReached(
                f"{label} previous joint motion did not settle before the next "
                f"command: last_delta={np.rad2deg(last_delta):.3f}deg"
            )
        time.sleep(poll)
        current = np.asarray(nero.get_joint_angles(), dtype=float).reshape(7)
        check_table(nero, args, f"{label}_settle")
        check_collision(nero, args, f"{label}_settle", joints=current)
        last_delta = float(np.max(np.abs(wrap_error(current, previous))))
        stable_samples = stable_samples + 1 if last_delta <= delta_limit else 0
        previous = current


def _send_joint_target_with_start_retry(
    nero,
    q_target,
    args,
    label,
    *,
    wait_for_joint_motion_start_hook=None,
    wait_for_joint_settle_hook=None,
    arm_motion_status_snapshot_hook=None,
    validate_motion_joint_path_hook=None,
):
    """Send a joint target and retry it once only when it was clearly dropped."""
    wait_for_start = (
        _wait_for_joint_motion_start
        if wait_for_joint_motion_start_hook is None
        else wait_for_joint_motion_start_hook
    )
    wait_for_settle = (
        _wait_for_joint_settle
        if wait_for_joint_settle_hook is None
        else wait_for_joint_settle_hook
    )
    status_snapshot = (
        _arm_motion_status_snapshot
        if arm_motion_status_snapshot_hook is None
        else arm_motion_status_snapshot_hook
    )
    validate_path = (
        validate_motion_joint_path
        if validate_motion_joint_path_hook is None
        else validate_motion_joint_path_hook
    )
    q_target = np.asarray(q_target, dtype=float).reshape(7)
    handoff_requested = bool(
        getattr(args, "_smooth_open_space_handoff_active", False)
    )
    handoff_context = getattr(args, "_runtime_motion_collision_context", None)
    if handoff_requested and handoff_context is not None:
        # The preceding endpoint has already entered its normal joint/pose
        # tolerance.  Re-read the actual joints and validate the short edge
        # that the controller will really receive, then retarget without the
        # legacy three-sample zero-speed barrier.
        q_before = np.asarray(nero.get_joint_angles(), dtype=float).reshape(7)
        validate_path(
            q_before,
            q_target,
            handoff_context,
            args,
            f"{label}_smooth_handoff_preflight",
        )
        print(
            f"[{label}][controller][smooth] open-space handoff from measured "
            "joints; full settle barrier skipped after short-edge revalidation"
        )
    else:
        # Fail closed if a caller accidentally requests handoff without the
        # frozen stage collision context needed for measured-edge validation.
        handoff_requested = False
        q_before = np.asarray(
            wait_for_settle(nero, args, label), dtype=float
        ).reshape(7)
    receipt = {
        "label": str(label),
        "q_before": q_before.copy(),
        "q_target": q_target.copy(),
        "retry_used": False,
        "smooth_handoff": handoff_requested,
    }
    previous_handoff_target = getattr(
        args, "_smooth_open_space_handoff_previous_target", None
    )
    if handoff_requested and previous_handoff_target is not None:
        receipt["handoff_previous_target"] = np.asarray(
            previous_handoff_target, dtype=float
        ).reshape(7).copy()
    setattr(nero, "_last_joint_command_receipt", receipt)
    response = nero.move_j(q_target)
    print(f"[{label}][ik] move_j response =", response)
    started, current, moved_deg, target_err_deg = wait_for_start(
        nero, q_before, q_target, args, label
    )
    if started:
        return response

    status = status_snapshot(nero, args)
    print(
        f"[{label}][controller][warn] joint command did not start: "
        f"moved={moved_deg:.3f}deg target_err={target_err_deg:.2f}deg "
        f"status={status}"
    )
    retry_enabled = bool(getattr(args, "move_retry_unstarted", True))
    if not retry_enabled or not bool(status.get("normal", False)):
        raise MotionNotReached(
            f"{label} joint command did not start and controller status does not permit retry"
        )

    controller_aware_retry = bool(
        getattr(args, "move_controller_aware_retry", False)
    )
    if controller_aware_retry:
        # A normal controller may still be finishing the preceding waypoint or
        # may publish its MOVE_J acknowledgement later than encoder feedback.
        # This branch runs only after the normal start timeout has expired, so
        # successful commands pay no additional latency.
        retry_args = copy(args)
        retry_args.move_precommand_settle_timeout = max(
            float(
                getattr(
                    args,
                    "move_unstarted_settle_timeout",
                    getattr(args, "move_precommand_settle_timeout", 1.5),
                )
            ),
            0.0,
        )
        retry_args.move_precommand_settle_samples = max(
            int(getattr(args, "move_unstarted_settle_samples", 2)), 1
        )
        print(
            f"[{label}][controller] command acknowledgement may be delayed; "
            f"wait up to {retry_args.move_precommand_settle_timeout:.1f}s "
            "for measured joints to settle before the one allowed resend"
        )
        q_retry_before = np.asarray(
            wait_for_settle(nero, retry_args, f"{label}_delayed_retry"),
            dtype=float,
        ).reshape(7)
    else:
        retry_delay = max(
            float(getattr(args, "move_retry_delay", 0.50)), 0.0
        )
        if retry_delay > 0.0:
            time.sleep(retry_delay)
        q_retry_before = np.asarray(
            nero.get_joint_angles(), dtype=float
        ).reshape(7)
    print(
        f"[{label}][controller] controller is normal; re-send the exact same "
        "validated joint target once"
    )
    retry_context = getattr(args, "_runtime_motion_collision_context", None)
    if retry_context is not None:
        validate_path(
            q_retry_before,
            q_target,
            retry_context,
            args,
            f"{label}_retry_preflight",
        )
    retry_response = nero.move_j(q_target)
    receipt["retry_used"] = True
    receipt["q_retry_before"] = q_retry_before.copy()
    print(f"[{label}][ik] retry move_j response =", retry_response)
    retry_started, _, retry_moved_deg, retry_target_err_deg = wait_for_start(
        nero, q_retry_before, q_target, args, f"{label}_retry"
    )
    if not retry_started:
        retry_status = status_snapshot(nero, args)
        raise MotionNotReached(
            f"{label} exact joint-target retry did not start: "
            f"moved={retry_moved_deg:.3f}deg target_err={retry_target_err_deg:.2f}deg "
            f"status={retry_status}"
        )
    return retry_response


def wait_until_joints_reached(
    nero,
    target_joints,
    args,
    label,
    *,
    table_guard_enabled_hook=None,
    base_wait_until_joints_reached_hook=None,
    check_table_guard_current_hook=None,
    check_runtime_collision_current_hook=None,
    wrap_angle_error_hook=None,
):
    table_guard_is_enabled = (
        table_guard_enabled
        if table_guard_enabled_hook is None
        else table_guard_enabled_hook
    )
    base_wait = (
        _base_wait_until_joints_reached
        if base_wait_until_joints_reached_hook is None
        else base_wait_until_joints_reached_hook
    )
    check_table = (
        check_table_guard_current
        if check_table_guard_current_hook is None
        else check_table_guard_current_hook
    )
    check_collision = (
        _runtime_collision_hook_required
        if check_runtime_collision_current_hook is None
        else check_runtime_collision_current_hook
    )
    wrap_error = (
        wrap_angle_error if wrap_angle_error_hook is None else wrap_angle_error_hook
    )
    if not table_guard_is_enabled(args) and getattr(
        args, "_runtime_motion_collision_context", None
    ) is None:
        return base_wait(nero, target_joints, args, label)
    target_joints = np.asarray(target_joints, dtype=float).reshape(7)
    deadline = time.time() + float(args.move_wait_timeout)
    last_joints = None
    previous_joints = None
    last_err = float("inf")
    stable_samples = 0
    started_at = time.time()
    setattr(nero, "_last_wait_failure_reason", None)
    stall_grace = max(float(getattr(args, "move_stall_grace", 1.0)), 0.0)
    stall_samples = max(int(getattr(args, "move_stall_confirm_samples", 5)), 1)
    stall_delta = np.deg2rad(
        max(float(getattr(args, "move_stall_delta_deg", 0.02)), 0.001)
    )
    stall_far_error = np.deg2rad(
        max(float(getattr(args, "move_stall_far_error_deg", 5.0)), 0.0)
    )
    while time.time() < deadline:
        last_joints = np.asarray(nero.get_joint_angles(), dtype=float).reshape(7)
        check_table(nero, args, label)
        check_collision(nero, args, label, joints=last_joints)
        last_err = float(np.linalg.norm(wrap_error(last_joints, target_joints)))
        if last_err <= float(args.move_joint_tol):
            print(f"[{label}] reached: joint_err={np.rad2deg(last_err):.2f}deg")
            setattr(nero, "_last_wait_failure_reason", None)
            return True, last_joints, last_err
        if previous_joints is not None and time.time() - started_at >= stall_grace:
            sample_delta = float(
                np.max(np.abs(wrap_error(last_joints, previous_joints)))
            )
            stable_samples = stable_samples + 1 if sample_delta <= stall_delta else 0
            if stable_samples >= stall_samples and last_err >= stall_far_error:
                print(
                    f"[{label}][warn] joint command stalled far from target: "
                    f"joint_err={np.rad2deg(last_err):.2f}deg "
                    f"last_delta={np.rad2deg(sample_delta):.3f}deg"
                )
                print(
                    f"[{label}][warn] joint_err_deg="
                    f"{np.round(np.rad2deg(wrap_error(last_joints, target_joints)), 3).tolist()} "
                    f"current_joints_deg={np.round(np.rad2deg(last_joints), 3).tolist()}"
                )
                setattr(nero, "_last_wait_failure_reason", "stalled")
                return False, last_joints, last_err
        else:
            stable_samples = 0
        previous_joints = last_joints.copy()
        time.sleep(float(args.move_poll_interval))

    print(f"[{label}][warn] joints not reached before timeout: joint_err={np.rad2deg(last_err):.2f}deg")
    if last_joints is not None:
        print(f"[{label}][warn] current_joints_deg =", np.round(np.rad2deg(last_joints), 3).tolist())
        print(f"[{label}][warn] target_joints_deg  =", np.round(np.rad2deg(target_joints), 3).tolist())
    setattr(nero, "_last_wait_failure_reason", "timeout")
    return False, last_joints, last_err


def move_flange_pose_ik_joint(
    arm,
    flange_pose,
    args,
    label,
    planned_joint_target=None,
    *,
    solve_ik_for_flange_pose_hook=None,
    get_ik_solver_hook=None,
    clip_numerical_joint_limit_overshoot_hook=None,
    pose_error_hook=None,
    send_joint_target_with_start_retry_hook=None,
    wait_until_joints_reached_hook=None,
    wait_until_pose_reached_hook=None,
):
    solve_ik = (
        _solve_ik_for_flange_pose
        if solve_ik_for_flange_pose_hook is None
        else solve_ik_for_flange_pose_hook
    )
    get_solver = _get_ik_solver if get_ik_solver_hook is None else get_ik_solver_hook
    clip_limit = (
        _clip_numerical_joint_limit_overshoot
        if clip_numerical_joint_limit_overshoot_hook is None
        else clip_numerical_joint_limit_overshoot_hook
    )
    compute_pose_error = pose_error if pose_error_hook is None else pose_error_hook
    send_joint_target = (
        _send_joint_target_with_start_retry
        if send_joint_target_with_start_retry_hook is None
        else send_joint_target_with_start_retry_hook
    )
    wait_joints = (
        wait_until_joints_reached
        if wait_until_joints_reached_hook is None
        else wait_until_joints_reached_hook
    )
    wait_pose = (
        wait_until_pose_reached
        if wait_until_pose_reached_hook is None
        else wait_until_pose_reached_hook
    )
    arm._residual_retry_used = False
    flange_pose = np.asarray(flange_pose, dtype=float).reshape(6)
    # The last target_approach command is an action boundary: reaching the
    # generic motion tolerance is not sufficient permission to close the
    # gripper.  Use the same strict pose/joint thresholds here that the close
    # gate uses later, so a stage cannot first report success and then time out
    # without any corrective command under a different definition of
    # "reached".
    grasp_action_boundary = bool(
        getattr(args, "_grasp_close_action_boundary_active", False)
    )
    motion_args = args
    if grasp_action_boundary:
        motion_args = copy(args)
        motion_args.move_pos_tol = float(
            getattr(args, "grasp_close_pos_tol", 0.003)
        )
        motion_args.move_rpy_tol = float(
            getattr(args, "grasp_close_rpy_tol", np.deg2rad(3.0))
        )
        motion_args.move_joint_tol = float(
            getattr(args, "grasp_close_joint_tol", np.deg2rad(0.5))
        )
        motion_args.move_wait_timeout = max(
            float(getattr(args, "move_wait_timeout", 20.0)),
            float(getattr(args, "grasp_close_wait_timeout", 8.0)),
        )
        print(
            f"[{label}][action_boundary] strict grasp target: "
            f"joint<={np.rad2deg(motion_args.move_joint_tol):.2f}deg "
            f"position<={motion_args.move_pos_tol * 1000.0:.1f}mm "
            f"orientation<={np.rad2deg(motion_args.move_rpy_tol):.2f}deg"
        )
    print(f"[{label}] target_flange_pose =", np.round(flange_pose, 6).tolist())
    if args.min_flange_z is not None and float(flange_pose[2]) < float(args.min_flange_z):
        raise UnsafeMotionTarget(
            f"{label} flange z={float(flange_pose[2]):.4f}m below --min-flange-z={float(args.min_flange_z):.4f}m. "
            "Raise --min-grasp-z or verify tcp_offset/tool geometry."
        )
    if planned_joint_target is None:
        q_target = solve_ik(arm, flange_pose, args, label)
    else:
        stored_q_target = np.asarray(
            planned_joint_target, dtype=float
        ).reshape(7)
        solver = get_solver(args)
        q_target = clip_limit(stored_q_target, solver)
        if q_target is None:
            raise PlanNotExecutable(f"{label} stored joint plan is invalid or outside joint limits")
        clipped_indices = np.flatnonzero(q_target != stored_q_target)
        if clipped_indices.size:
            max_clip_rad = float(
                np.max(np.abs(q_target - stored_q_target))
            )
            print(
                f"[{label}][ik] clamped numerical joint-limit rounding on "
                f"joint(s)={(clipped_indices + 1).tolist()} by at most "
                f"{np.rad2deg(max_clip_rad):.9f}deg"
            )
        fk = np.asarray(solver.solve_fk(q_target), dtype=float).reshape(7)
        fk_pose = np.r_[
            fk[:3],
            R.from_quat([fk[4], fk[5], fk[6], fk[3]]).as_euler("xyz", degrees=False),
        ]
        fk_pos_err, fk_rpy_err = compute_pose_error(fk_pose, flange_pose)
        if fk_pos_err > 0.003 or fk_rpy_err > np.deg2rad(1.0):
            raise PlanNotExecutable(
                f"{label} stored joint plan no longer matches flange target: "
                f"pos_err={fk_pos_err:.4f}m rpy_err={np.rad2deg(fk_rpy_err):.2f}deg"
            )
        current_joints = np.asarray(arm.get_joint_angles(), dtype=float).reshape(7)
        delta = q_target - current_joints
        print(f"[{label}][ik] using prevalidated joint target deg =", np.round(np.rad2deg(q_target), 3).tolist())
        print(
            f"[{label}][ik] joint_delta_deg={np.round(np.rad2deg(delta), 3).tolist()} "
            f"max_delta={np.rad2deg(np.max(np.abs(delta))):.2f}deg"
        )
    send_joint_target(arm, q_target, motion_args, label)
    joint_reached, current_joints, joint_err = wait_joints(
        arm, q_target, motion_args, f"{label}_joint"
    )
    if not joint_reached and current_joints is not None:
        receipt = getattr(arm, "_last_joint_command_receipt", None)
        retry_stalled = bool(getattr(args, "move_retry_stalled", True))
        if isinstance(receipt, dict) and retry_stalled and not bool(
            receipt.get("retry_used", False)
        ):
            receipt_start = np.asarray(receipt.get("q_before"), dtype=float).reshape(7)
            actual_displacement = float(
                np.max(np.abs(wrap_angle_error(current_joints, receipt_start)))
            )
            max_displacement = np.deg2rad(
                max(
                    float(
                        getattr(
                            args,
                            "move_late_retry_max_displacement_deg",
                            0.50,
                        )
                    ),
                    0.0,
                )
            )
            handoff_previous = receipt.get("handoff_previous_target")
            stalled_near_handoff_node = False
            handoff_node_error = float("inf")
            if handoff_previous is not None:
                handoff_node_error = float(
                    np.max(
                        np.abs(
                            wrap_angle_error(
                                current_joints,
                                np.asarray(handoff_previous, dtype=float).reshape(7),
                            )
                        )
                    )
                )
                handoff_node_tol = np.deg2rad(
                    max(
                        float(
                            getattr(
                                args,
                                "smooth_handoff_previous_target_tol_deg",
                                3.0,
                            )
                        ),
                        0.0,
                    )
                )
                stalled_near_handoff_node = bool(
                    handoff_node_error <= handoff_node_tol
                )
            if actual_displacement <= max_displacement or stalled_near_handoff_node:
                reason = (
                    f"stopped near the previous handoff node "
                    f"(error={np.rad2deg(handoff_node_error):.2f}deg)"
                    if stalled_near_handoff_node
                    else f"remained at the command start "
                    f"(moved={np.rad2deg(actual_displacement):.3f}deg)"
                )
                print(
                    f"[{label}][controller] target {reason}; "
                    "perform one final exact-target resend"
                )
                receipt["retry_used"] = True
                send_joint_target(
                    arm, q_target, motion_args, f"{label}_late_retry"
                )
                joint_reached, current_joints, joint_err = wait_joints(
                    arm,
                    q_target,
                    motion_args,
                    f"{label}_joint_after_late_retry",
                )
    if (
        not joint_reached
        and current_joints is not None
        and bool(getattr(args, "move_residual_retry", True))
        and not bool(getattr(arm, "_residual_retry_used", False))
        and getattr(arm, "_last_wait_failure_reason", None) == "stalled"
    ):
        residual = float(np.linalg.norm(wrap_angle_error(current_joints, q_target)))
        limit = np.deg2rad(
            max(float(getattr(args, "move_residual_retry_max_deg", 12.0)), 0.0)
        )
        if residual <= limit:
            retry_context = getattr(args, "_runtime_motion_collision_context", None)
            if retry_context is not None:
                validate_motion_joint_path(
                    current_joints,
                    q_target,
                    retry_context,
                    motion_args,
                    f"{label}_residual_retry_preflight",
                )
            print(
                f"[{label}][residual-retry] stalled at joint_err="
                f"{np.rad2deg(residual):.2f}deg <= {np.rad2deg(limit):.2f}deg; "
                "resend the same prevalidated target once"
            )
            arm._residual_retry_used = True
            send_joint_target(
                arm, q_target, motion_args, f"{label}_residual_retry"
            )
            joint_reached, current_joints, joint_err = wait_joints(
                arm,
                q_target,
                motion_args,
                f"{label}_joint_after_residual_retry",
            )
    reached, current_pose, pos_err, rpy_err = wait_pose(
        arm, flange_pose, motion_args, label
    )
    if grasp_action_boundary and (not joint_reached or not reached):
        receipt = getattr(arm, "_last_joint_command_receipt", None)
        retry_unused = not (
            isinstance(receipt, dict) and bool(receipt.get("retry_used", False))
        )
        status = _arm_motion_status_snapshot(arm, motion_args)
        if (
            bool(getattr(args, "grasp_close_exact_retry", True))
            and retry_unused
            and bool(status.get("normal", False))
        ):
            # This is not a re-plan and cannot move toward a new solution.  It
            # is one bounded, live-path-revalidated resend of the exact frozen
            # joint target, after controller feedback proves the prior motion
            # is idle and healthy.
            retry_args = copy(motion_args)
            retry_args.move_retry_unstarted = False
            retry_args.move_retry_stalled = False
            retry_args.move_wait_timeout = max(
                float(getattr(args, "grasp_close_wait_timeout", 8.0)),
                0.1,
            )
            print(
                f"[{label}][action_boundary] strict target not reached; "
                "controller is idle/normal, re-send the exact validated "
                "grasp target once"
            )
            send_joint_target(
                arm,
                q_target,
                retry_args,
                f"{label}_grasp_exact_retry",
            )
            retry_receipt = getattr(arm, "_last_joint_command_receipt", None)
            if isinstance(retry_receipt, dict):
                retry_receipt["retry_used"] = True
                retry_receipt["grasp_close_retry_used"] = True
            joint_reached, current_joints, joint_err = wait_joints(
                arm,
                q_target,
                retry_args,
                f"{label}_joint_after_grasp_exact_retry",
            )
            reached, current_pose, pos_err, rpy_err = wait_pose(
                arm,
                flange_pose,
                retry_args,
                f"{label}_after_grasp_exact_retry",
            )
        else:
            print(
                f"[{label}][action_boundary][warn] exact retry not permitted: "
                f"enabled={bool(getattr(args, 'grasp_close_exact_retry', True))} "
                f"retry_unused={retry_unused} controller_status={status}"
            )
    if current_pose is not None:
        pose_delta = np.asarray(current_pose, dtype=float).reshape(6) - flange_pose
        pose_delta[3:] = (pose_delta[3:] + np.pi) % (2.0 * np.pi) - np.pi
        print(f"[{label}] theoretical_flange_pose =", np.round(flange_pose, 6).tolist())
        print(f"[{label}] actual_flange_pose      =", np.round(current_pose, 6).tolist())
        print(
            f"[{label}] actual_error: xyz={np.round(pose_delta[:3], 6).tolist()}m "
            f"rpy={np.round(np.rad2deg(pose_delta[3:]), 3).tolist()}deg "
            f"norm_pos={float(pos_err):.4f}m norm_rpy={np.rad2deg(float(rpy_err)):.2f}deg"
        )
    if not joint_reached:
        raise MotionNotReached(f"{label} IK joint target not reached: joint_err={np.rad2deg(joint_err):.2f}deg")
    if not reached:
        raise MotionNotReached(
            f"{label} IK flange target not reached: pos_err={pos_err:.4f}m "
            f"rpy_err={np.rad2deg(rpy_err):.2f}deg"
        )
    return flange_pose


def move_flange_pose(
    arm,
    flange_pose,
    args,
    label,
    planned_joint_target=None,
    *,
    move_flange_pose_ik_joint_hook=None,
    wait_until_pose_reached_hook=None,
):
    move_ik_joint = (
        move_flange_pose_ik_joint
        if move_flange_pose_ik_joint_hook is None
        else move_flange_pose_ik_joint_hook
    )
    wait_pose = (
        wait_until_pose_reached
        if wait_until_pose_reached_hook is None
        else wait_until_pose_reached_hook
    )
    flange_pose = np.asarray(flange_pose, dtype=float).reshape(6)
    if getattr(args, "motion_mode", "cartesian") == "ik_joint":
        return move_ik_joint(
            arm,
            flange_pose,
            args,
            label,
            planned_joint_target=planned_joint_target,
        )
    if planned_joint_target is not None:
        raise PlanNotExecutable(
            f"{label} prevalidated active joint plan requires --motion-mode ik_joint"
        )
    print(f"[{label}] target_flange_pose =", np.round(flange_pose, 6).tolist())
    if args.min_flange_z is not None and float(flange_pose[2]) < float(args.min_flange_z):
        raise UnsafeMotionTarget(
            f"{label} flange z={float(flange_pose[2]):.4f}m below --min-flange-z={float(args.min_flange_z):.4f}m. "
            "Raise --min-grasp-z or verify tcp_offset/tool geometry."
        )
    resp = arm.move_p(flange_pose)
    print(f"[{label}] move_p response =", resp)
    reached, current_pose, pos_err, rpy_err = wait_pose(arm, flange_pose, args, label)
    if current_pose is not None:
        pose_delta = np.asarray(current_pose, dtype=float).reshape(6) - np.asarray(flange_pose, dtype=float).reshape(6)
        pose_delta[3:] = (pose_delta[3:] + np.pi) % (2.0 * np.pi) - np.pi
        print(f"[{label}] theoretical_flange_pose =", np.round(flange_pose, 6).tolist())
        print(f"[{label}] actual_flange_pose      =", np.round(current_pose, 6).tolist())
        print(
            f"[{label}] actual_error: xyz={np.round(pose_delta[:3], 6).tolist()}m "
            f"rpy={np.round(np.rad2deg(pose_delta[3:]), 3).tolist()}deg "
            f"norm_pos={float(pos_err):.4f}m norm_rpy={np.rad2deg(float(rpy_err)):.2f}deg"
        )
    if not reached:
        raise MotionNotReached(
            f"{label} not reached: pos_err={pos_err:.4f}m "
            f"rpy_err={np.rad2deg(rpy_err):.2f}deg"
        )
    return flange_pose
