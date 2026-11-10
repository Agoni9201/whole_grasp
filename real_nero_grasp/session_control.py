"""Interactive session setup, gripper confirmation, and recovery helpers."""
# 话题控制
from dataclasses import dataclass
from typing import Any, Callable, Optional
import time

import numpy as np

from .types import (
    GraspNotDetected,
    GripperCloseResult,
    SafetyStopTriggered,
)


@dataclass(frozen=True)
class SessionControlHooks:
    """Runtime-owned collaborators used by session-control helpers."""

    print_status: Callable[[Any], Any]
    preflight_initial_table_path: Callable[[Any, Any, Any], Any]
    wait_until_joints_reached: Callable[..., Any]
    wrap_angle_error: Callable[[Any, Any], Any]
    safe_stop: Callable[[Any, str], Any]
    terminate_motion: Optional[Callable[[Any, str], Any]] = None
    confirm_motion: Optional[Callable[[str], bool]] = None
    wait_until_gripper_open: Optional[Callable[[Any, Any], bool]] = None
    move_initial: Optional[Callable[..., Any]] = None


__all__ = [
    "SessionControlHooks",
    "confirm_motion",
    "set_tcp_offset",
    "move_initial",
    "return_initial_with_open_gripper",
    "recover_after_empty_grasp",
    "verify_empty_grasp_recovery_at_initial_observation",
    "wait_until_gripper_open",
    "prepare_initial_pose_for_capture",
    "assert_grasp_detected",
]


def _require_hooks(hooks):
    if not isinstance(hooks, SessionControlHooks):
        raise TypeError("a SessionControlHooks instance is required")
    return hooks

def confirm_motion(action_description):
    """Return True only for an empty line; ``no`` cancels the pending motion."""
    while True:
        answer = input(
            f"[confirm] {action_description} -- press Enter to execute, "
            "or type no to cancel: "
        ).strip().lower()
        if answer == "":
            return True
        if answer == "no":
            return False
        print("[confirm] unrecognized input; press Enter to execute or type no to cancel")

def set_tcp_offset(arm, args, *, hooks):
    hooks = _require_hooks(hooks)
    offset = np.asarray(args.tcp_offset, dtype=float).reshape(6)
    print("[tcp] set_tcp_offset =", np.round(offset, 6).tolist())
    arm.set_tcp_offset(offset)
    hooks.print_status(arm)

def move_initial(arm, args, require_confirm=True, *, hooks):
    hooks = _require_hooks(hooks)
    confirm_motion_fn = hooks.confirm_motion or confirm_motion
    joints = np.deg2rad(np.asarray(args.initial_joints_deg, dtype=float).reshape(7))
    print("[init] target joints deg =", np.round(np.rad2deg(joints), 3).tolist())
    hooks.preflight_initial_table_path(arm, joints, args)
    if require_confirm and not args.yes_initial:
        if not confirm_motion_fn("move to the initial joint pose"):
            print("[init] skipped")
            return None
    arm.move_j(joints)
    reached, _, _ = hooks.wait_until_joints_reached(arm, joints, args, "init")
    return bool(reached)

def return_initial_with_open_gripper(arm, args, *, hooks):
    hooks = _require_hooks(hooks)
    wait_open_fn = hooks.wait_until_gripper_open or wait_until_gripper_open
    move_initial_fn = hooks.move_initial
    if move_initial_fn is None:
        move_initial_fn = lambda arm_, args_, require_confirm=True: move_initial(
            arm_, args_, require_confirm=require_confirm, hooks=hooks
        )
    print("[return] opening gripper and confirming release before returning to initial joint pose")
    arm.open_gripper(args.gripper_open_width, args.gripper_force, args.gripper_settle)
    if not wait_open_fn(arm, args):
        print("[return][abort] gripper did not reach open width; initial return was not commanded")
        return False
    return bool(move_initial_fn(arm, args, require_confirm=False))

def recover_after_empty_grasp(arm, args, *, hooks):
    hooks = _require_hooks(hooks)
    wait_open_fn = hooks.wait_until_gripper_open or wait_until_gripper_open
    move_initial_fn = hooks.move_initial
    if move_initial_fn is None:
        move_initial_fn = lambda arm_, args_, require_confirm=True: move_initial(
            arm_, args_, require_confirm=require_confirm, hooks=hooks
        )
    delay = max(float(args.empty_grasp_return_delay), 0.0)
    if delay > 0.0:
        print(f"[return] empty grasp recovery delay {delay:.2f}s before returning")
        time.sleep(delay)
    print("[return] opening gripper and confirming width before recovery motion")
    arm.open_gripper(args.gripper_open_width, args.gripper_force, args.gripper_settle)
    if not wait_open_fn(arm, args):
        print("[return][abort] gripper did not reopen; initial-pose recovery was not commanded")
        return False
    if args.skip_empty_grasp_return:
        print("[return] empty grasp auto-return disabled; gripper opened, hold current pose")
        return True
    recovery_joints = getattr(args, "_empty_grasp_recovery_joints", None)
    if recovery_joints is not None:
        recovery_joints = np.asarray(recovery_joints, dtype=float).reshape(7)
        print("[return] returning to saved pre-workflow joint pose")
        arm.move_j(recovery_joints)
        reached, _, _ = hooks.wait_until_joints_reached(
            arm,
            recovery_joints,
            args,
            "empty_grasp_return",
        )
        return bool(reached)
    if args.skip_initial_move:
        print("[return][warn] no saved recovery joints and --skip-initial-move is set; hold current pose")
        return False
    print("[return] empty grasp confirmed; returning to initial joint pose")
    return bool(move_initial_fn(arm, args, require_confirm=False))

def verify_empty_grasp_recovery_at_initial_observation(
    arm,
    args,
    recovered,
    *,
    hooks,
):
    """Verify that a retry capture would be taken from the saved first-view pose."""
    hooks = _require_hooks(hooks)
    if not bool(recovered):
        return False
    if bool(getattr(args, "skip_empty_grasp_return", False)):
        print(
            "[return][retry] empty-grasp auto-return is disabled; "
            "semantic retry is not allowed"
        )
        return False
    target_joints = getattr(args, "_empty_grasp_recovery_joints", None)
    if target_joints is None:
        print(
            "[return][retry] saved initial observation joints are unavailable; "
            "semantic retry is not allowed"
        )
        return False
    try:
        target_joints = np.asarray(target_joints, dtype=float).reshape(7)
        current_joints = np.asarray(arm.get_joint_angles(), dtype=float).reshape(7)
    except Exception as exc:
        print(
            "[return][retry] could not verify the recovered joint pose: "
            f"{type(exc).__name__}: {exc}"
        )
        return False
    if not np.isfinite(target_joints).all() or not np.isfinite(current_joints).all():
        print(
            "[return][retry] recovered joint feedback is invalid; "
            "semantic retry is not allowed"
        )
        return False
    joint_error = float(
        np.linalg.norm(hooks.wrap_angle_error(current_joints, target_joints))
    )
    # The normal motion-reached check has already passed.  Allow a small extra
    # settling margin here while keeping the camera close enough for strict
    # mask-based target reassociation to validate the identity again.
    verify_tol = max(
        2.0 * float(getattr(args, "move_joint_tol", np.deg2rad(2.0))),
        float(np.deg2rad(4.0)),
    )
    verified = bool(joint_error <= verify_tol)
    print(
        f"[return][retry] initial observation pose verification: "
        f"joint_err={np.rad2deg(joint_error):.2f}deg "
        f"limit={np.rad2deg(verify_tol):.2f}deg verified={verified}"
    )
    return verified

def wait_until_gripper_open(arm, args):
    if arm.gripper is None:
        return True
    target = min(
        float(args.gripper_open_width),
        float(getattr(args, "gripper_hardware_max_width", 0.100)),
    )
    tol = float(args.gripper_open_tol)
    deadline = time.time() + float(args.gripper_open_timeout)
    required_samples = max(
        int(getattr(args, "gripper_open_confirm_samples", 2)),
        1,
    )
    confirmed_samples = 0
    last_value = None
    while time.time() < deadline:
        msg = arm.print_gripper_status()
        valid_sample = False
        if msg is not None and getattr(msg, "value", None) is not None:
            try:
                last_value = float(msg.value)
            except Exception:
                last_value = None
            feedback_mode = str(getattr(msg, "mode", "")).strip().lower()
            valid_sample = bool(
                feedback_mode == "width"
                and last_value is not None
                and np.isfinite(last_value)
                # The collision model and calibrated camera silhouette use
                # this requested gap. A larger opening is not equivalent.
                and abs(last_value - target) <= tol
            )
        confirmed_samples = confirmed_samples + 1 if valid_sample else 0
        if confirmed_samples >= required_samples:
            print(
                f"[init] gripper open confirmed: value={last_value:.4f}m "
                f"target={target:.4f}m samples={confirmed_samples}/{required_samples}"
            )
            return True
        time.sleep(float(args.move_poll_interval))
    print(
        f"[init][warn] gripper did not reach a confirmed open width before timeout: "
        f"last={last_value} target={target:.4f}m "
        f"samples={confirmed_samples}/{required_samples}"
    )
    return False

def prepare_initial_pose_for_capture(arm, args, *, hooks):
    hooks = _require_hooks(hooks)
    wait_open_fn = hooks.wait_until_gripper_open or wait_until_gripper_open
    move_initial_fn = hooks.move_initial
    if move_initial_fn is None:
        move_initial_fn = lambda arm_, args_: move_initial(
            arm_, args_, hooks=hooks
        )
    if args.skip_initial_move:
        print("[init] skipped initial move; opening gripper before capture")
        arm.open_gripper(args.gripper_open_width, args.gripper_force, args.gripper_settle)
        if not wait_open_fn(arm, args) and bool(
            getattr(args, "require_gripper_open", True)
        ):
            raise GraspNotDetected("gripper failed to open before capture")
        return
    reached = move_initial_fn(arm, args)
    if reached is None:
        print("[init][warn] initial move was skipped by user; continue without requiring initial pose.")
    elif not reached:
        reason = "initial joint pose did not reach before timeout"
        # This timeout is handled as a controlled workflow cancellation.  Do
        # not let context-manager shutdown convert it into an electronic
        # emergency stop; motion remains latched and no capture/grasp command
        # is permitted after this point.
        setattr(arm, "_suppress_program_exit_estop", True)
        terminate_fn = hooks.terminate_motion
        if terminate_fn is None:
            method = getattr(arm, "terminate_motion", None)
            if callable(method):
                terminate_fn = lambda arm_, _reason: method()
        if terminate_fn is not None:
            try:
                response = terminate_fn(arm, reason)
                confirmed = bool(
                    response is True
                    or (
                        isinstance(response, dict)
                        and bool(response.get("confirmed", False))
                    )
                )
                if confirmed:
                    msg = (
                        f"{reason}; ordinary trajectory termination was "
                        "confirmed before capture. Electronic emergency "
                        "stop was not requested"
                    )
                    setattr(arm, "_safety_stop_reason", msg)
                    print(f"[motion_stop] {msg}; details={response}")
                    raise SafetyStopTriggered(msg)
                termination_error = RuntimeError(
                    f"ordinary termination returned unconfirmed response={response}"
                )
            except SafetyStopTriggered:
                raise
            except Exception as exc:
                termination_error = exc
        else:
            termination_error = RuntimeError(
                "arm exposes no ordinary trajectory termination method"
            )
        msg = (
            f"{reason}; ordinary trajectory termination was requested but "
            f"settling was not confirmed ({type(termination_error).__name__}: "
            f"{termination_error}). The workflow is terminal and electronic "
            "emergency stop is disabled for this initial-timeout branch"
        )
        setattr(arm, "_safety_stop_reason", msg)
        print(f"[motion_stop][warn] {msg}")
        raise SafetyStopTriggered(msg)
    print("[init] opening gripper before capture")
    arm.open_gripper(args.gripper_open_width, args.gripper_force, args.gripper_settle)
    if not wait_open_fn(arm, args) and args.require_gripper_open:
        raise GraspNotDetected("gripper failed to open before capture")

def assert_grasp_detected(close_status, args):
    allow_empty = bool(args.allow_empty_grasp_lift)
    if isinstance(close_status, GripperCloseResult):
        if close_status.contact_detected:
            # Slow-close contact has already passed mode/width/force validation,
            # consecutive force confirmation, and the physical-close-limit test.
            return
        else:
            known_empty = close_status.stop_reason in {
                "close_limit_without_contact",
                "force_at_closed_limit",
            }
            if bool(getattr(args, "require_gripper_contact", True)) or not (allow_empty and known_empty):
                raise GraspNotDetected(
                    "slow close ended without confirmed force contact; "
                    f"stop_reason={close_status.stop_reason}",
                    retryable_empty=known_empty,
                    reason_code=close_status.stop_reason,
                )
        close_status = close_status.status
    if close_status is None:
        if bool(getattr(args, "require_gripper_status", True)):
            raise GraspNotDetected("gripper returned no status after close; contact cannot be verified")
        return
    value = getattr(close_status, "value", None)
    if value is None:
        if bool(getattr(args, "require_gripper_status", True)):
            raise GraspNotDetected("gripper status has no width value; contact cannot be verified")
        return
    value = float(value)
    if not np.isfinite(value):
        raise GraspNotDetected("gripper status returned a non-finite width value")
    if value <= float(args.empty_grasp_width_threshold) and not allow_empty:
        raise GraspNotDetected(
            f"gripper closed to {value * 1000.0:.2f}mm <= "
            f"empty threshold {float(args.empty_grasp_width_threshold) * 1000.0:.2f}mm; "
            "likely no object between fingers",
            retryable_empty=True,
            reason_code="fully_closed_without_object",
        )
