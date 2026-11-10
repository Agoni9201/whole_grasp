"""Shared ordinary-stop-first policy for recoverable execution failures."""

from datetime import datetime
import time
from zoneinfo import ZoneInfo

import numpy as np

from .types import SafetyStopTriggered


__all__ = ["request_ordinary_trajectory_stop"]


def request_ordinary_trajectory_stop(
    arm,
    reason,
    *,
    safe_stop_hook,
    fallback_joints=None,
    fault_context=None,
):
    """Terminate and hold normally, escalating only when stop is unconfirmed.

    ``RealNeroGraspArm.terminate_motion`` performs the physical settling check
    and raises when it cannot prove that the arm stopped.  Keeping that check
    here makes timeout/read/recovery failures share one escalation policy.
    """
    message = str(reason)
    setattr(arm, "_safety_stop_reason", message)
    event = {
        "timestamp": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="milliseconds"),
        "monotonic_time": time.monotonic(),
        "level": "R2", "reason": message, "action": "controlled_stop",
        "stop_confirmed": False, "emergency_stop_requested": False,
    }
    setattr(arm, "_last_fault_decision", event)
    state = getattr(fault_context, "_experiment_log_state", None)
    if isinstance(state, dict):
        state.setdefault("fault_events", []).append(event)
    if fallback_joints is None:
        try:
            fallback_joints = np.asarray(
                arm.get_joint_angles(), dtype=float
            ).reshape(7)
        except Exception:
            fallback_joints = None

    try:
        response = None
        owner = getattr(arm, "_mit_control_owner", None)
        idle = getattr(arm, "_confirm_idle_for_program_exit", None)
        if callable(owner) and owner() is None and callable(idle):
            # Never use the ordinary controller's idle bit to judge MIT.
            response = idle(first_joints=fallback_joints)
            if isinstance(response, dict) and response.get("confirmed") is True:
                event.update(level="R1", action="stationary_task_abort")
        terminate = getattr(arm, "terminate_motion", None)
        if response is None and not callable(terminate):
            raise RuntimeError(
                "arm exposes no ordinary trajectory termination method"
            )
        if response is None:
            if fallback_joints is None:
                response = terminate()
            else:
                response = terminate(fallback_joints=fallback_joints)
        if not isinstance(response, dict) or response.get("confirmed") is not True:
            raise RuntimeError("ordinary stop returned without explicit confirmation")
        event["stop_confirmed"] = True
        event["stop_method"] = response.get("method")
    except Exception as exc:
        failure_message = (
            f"{message}; ordinary trajectory termination could not be confirmed: "
            f"{type(exc).__name__}: {exc}; escalating because physical motion "
            "stop is unconfirmed"
        )
        setattr(arm, "_safety_stop_reason", failure_message)
        event.update(level="R3", action="electronic_stop_fallback", stop_error=str(exc),
                     emergency_stop_requested=True)
        # A lower-level terminal fault may already have requested e-stop.
        # Do not issue another request merely because the exception unwound.
        if getattr(arm, "_safety_stop_request_confirmed", False) is not True:
            try:
                safe_stop_hook(arm, failure_message)
            except Exception as stop_error:
                event["stop_request_error"] = str(stop_error)
        raise SafetyStopTriggered(failure_message) from exc
    try:
        print(f"[{event['level']}][motion_stop] {message}")
        print("[motion_stop] stop/hold confirmed; "
              f"details={response}; electronic emergency stop was not requested")
        print("[motion_stop] current workflow is terminal; old targets remain blocked")
    except Exception:
        pass  # Terminal/log I/O failure is not a failed physical stop.
    return response
