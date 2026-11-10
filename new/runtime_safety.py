"""Measured-joint runtime collision monitoring and stop escalation policy.

The guard evaluates each measured configuration against the frozen stage map.
Positive-clearance tolerance violations first request ordinary trajectory
termination. Electronic emergency stop is reserved for penetration, invalid
constraints, guard failures, or an unconfirmed ordinary stop.
"""

import json
import os
import time
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from .types import (
    ControlledMotionStopTriggered,
    MotionCollisionContext,
    SafetyStopTriggered,
    TorqueCollisionStopTriggered,
    UnsafeMotionTarget,
)


@dataclass(frozen=True)
class RuntimeSafetyHooks:
    """Explicit dependencies used by measured-joint runtime protection."""

    as_pose_list: Optional[Callable] = None
    tcp_transform_from_joints: Optional[Callable] = None
    motion_configuration_metrics: Optional[Callable] = None
    persist_runtime_collision_diagnostic: Optional[Callable] = None
    safe_stop: Optional[Callable] = None


def _require_hook(hooks, name):
    if hooks is None:
        raise RuntimeError(f"runtime-safety dependency hook '{name}' was not supplied")
    value = getattr(hooks, name, None)
    if not callable(value):
        raise RuntimeError(f"runtime-safety dependency hook '{name}' is unavailable")
    return value


__all__ = [
    "RuntimeSafetyHooks",
    "_persist_runtime_collision_diagnostic",
    "check_runtime_collision_current",
]


def _persist_runtime_collision_diagnostic(
    args,
    label,
    joints,
    context,
    metrics,
    *,
    hooks,
):
    """Persist the exact measured stop sample for offline trajectory audit."""
    _as_pose_list = _require_hook(hooks, "as_pose_list")
    _tcp_transform_from_joints = _require_hook(
        hooks, "tcp_transform_from_joints"
    )
    q = np.asarray(joints, dtype=float).reshape(7)
    payload = {
        "timestamp": time.strftime("%Y-%m-%d_%H-%M-%S"),
        "time_unix": float(time.time()),
        "label": str(label),
        "map_id": str(context.scene.map_id),
        "phase": str(context.phase),
        "measured_joints_deg": np.rad2deg(q).tolist(),
        "metrics": dict(metrics),
    }
    q_start = getattr(args, "_runtime_motion_stage_start_joints", None)
    q_target = getattr(args, "_runtime_motion_stage_target_joints", None)
    if q_start is not None and q_target is not None:
        q_start = np.asarray(q_start, dtype=float).reshape(7)
        q_target = np.asarray(q_target, dtype=float).reshape(7)
        delta = q_target - q_start
        denom = float(np.dot(delta, delta))
        alpha_raw = (
            0.0
            if denom <= 1e-12
            else float(np.dot(q - q_start, delta) / denom)
        )
        alpha = float(np.clip(alpha_raw, 0.0, 1.0))
        projected = q_start + alpha * delta
        per_axis_progress = [
            (
                None
                if abs(float(axis_delta)) <= 1e-6
                else float((q[index] - q_start[index]) / axis_delta)
            )
            for index, axis_delta in enumerate(delta)
        ]
        payload.update(
            {
                "stage_label": str(
                    getattr(args, "_runtime_motion_stage_label", label)
                ),
                "stage_start_joints_deg": np.rad2deg(q_start).tolist(),
                "stage_target_joints_deg": np.rad2deg(q_target).tolist(),
                "path_alpha_raw": alpha_raw,
                "path_alpha_clamped": alpha,
                "per_axis_progress": per_axis_progress,
                "max_joint_path_residual_deg": float(
                    np.rad2deg(np.max(np.abs(q - projected)))
                ),
            }
        )
    try:
        T_base_tcp = _tcp_transform_from_joints(q, args)
        payload["modeled_tcp_pose"] = np.asarray(
            _as_pose_list(T_base_tcp), dtype=float
        ).tolist()
    except Exception as exc:
        payload["modeled_tcp_error"] = f"{type(exc).__name__}: {exc}"

    def json_safe(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, dict):
            return {str(key): json_safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [json_safe(item) for item in value]
        return value

    destinations = []
    save_dir = str(getattr(args, "save_dir", "")).strip()
    if save_dir:
        destinations.append(
            os.path.join(save_dir, "runtime_collision_diagnostics.jsonl")
        )
    latest_capture = str(
        getattr(args, "_last_capture_out_dir", "") or ""
    ).strip()
    if latest_capture:
        destinations.append(
            os.path.join(latest_capture, "runtime_collision_diagnostics.jsonl")
        )
    written = []
    for path in dict.fromkeys(destinations):
        try:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            with open(path, "a", encoding="utf-8") as stream:
                stream.write(json.dumps(json_safe(payload), sort_keys=True) + "\n")
            written.append(path)
        except Exception as exc:
            print(
                f"[protect][diag][warn] failed to save {path}: "
                f"{type(exc).__name__}: {exc}"
            )
    if written:
        print(f"[protect][diag] saved runtime stop sample={written}")
    print(
        f"[protect][diag] measured_joints_deg="
        f"{np.round(np.rad2deg(q), 3).tolist()}"
    )


def _check_runtime_joint_torque(
    arm,
    args,
    label,
    joints,
    context,
    *,
    hooks,
):
    """Stop on sustained *measured* joint-torque contact.

    The joint-side torque estimate naturally changes with configuration and
    payload.  Therefore it is compared to the first valid sample for the
    current frozen stage rather than to a global absolute value.  A missed
    torque interface never disables the geometric/table guard: it is logged
    once and this additional layer is simply unavailable for that stage.
    """
    if (
        not bool(getattr(args, "motion_runtime_torque_guard", True))
        or bool(getattr(args, "_runtime_torque_guard_suppressed", False))
    ):
        return None
    getter = getattr(arm, "get_joint_torques", None)
    if not callable(getter):
        return None
    try:
        torque = getter()
        if torque is None:
            raise RuntimeError("Nero did not return seven finite joint torques")
        torque = np.asarray(torque, dtype=float).reshape(7)
        if not np.all(np.isfinite(torque)):
            raise RuntimeError("Nero returned non-finite joint torque feedback")
    except Exception as exc:
        warned_stage = getattr(args, "_runtime_torque_unavailable_stage", None)
        if warned_stage != str(label):
            print(
                f"[torque][warn] {label}: measured torque protection is unavailable "
                f"({type(exc).__name__}: {exc}); point-cloud/table protection remains active"
            )
            args._runtime_torque_unavailable_stage = str(label)
        return None

    delta_limit = float(getattr(args, "motion_runtime_torque_delta", 8.0))
    confirm_samples = max(
        int(getattr(args, "motion_runtime_torque_confirm_samples", 2)), 1
    )
    if not np.isfinite(delta_limit) or delta_limit <= 0.0:
        return None
    state = getattr(args, "_runtime_torque_guard_state", None)
    stage_label = str(getattr(args, "_runtime_motion_stage_label", label))
    if not isinstance(state, dict) or state.get("stage_label") != stage_label:
        state = {
            "stage_label": stage_label,
            "baseline_nm": torque.copy(),
            "consecutive": 0,
        }
        args._runtime_torque_guard_state = state
        print(
            f"[torque] {stage_label} baseline_nm="
            f"{np.round(torque, 3).tolist()} delta_limit={delta_limit:.2f}N.m "
            f"confirm_samples={confirm_samples}"
        )
        return None

    baseline = np.asarray(state["baseline_nm"], dtype=float).reshape(7)
    delta = np.abs(torque - baseline)
    peak_index = int(np.argmax(delta))
    peak_delta = float(delta[peak_index])
    if peak_delta >= delta_limit:
        state["consecutive"] = int(state.get("consecutive", 0)) + 1
    else:
        state["consecutive"] = 0
    if int(state["consecutive"]) < confirm_samples:
        return None

    metrics = {
        "margin": -peak_delta,
        "nearest_clearance": -peak_delta,
        "required_clearance": delta_limit,
        "primitive": f"joint{peak_index + 1}_torque",
        "source": "joint_torque",
        "measured_torque_nm": torque,
        "baseline_torque_nm": baseline,
        "torque_delta_nm": delta,
        "confirm_samples": int(state["consecutive"]),
    }
    _persist_runtime_collision_diagnostic = _require_hook(
        hooks, "persist_runtime_collision_diagnostic"
    )
    _persist_runtime_collision_diagnostic(args, label, joints, context, metrics)
    msg = (
        f"{label} runtime torque-contact stop: joint{peak_index + 1} delta="
        f"{peak_delta:.2f}N.m >= {delta_limit:.2f}N.m for "
        f"{int(state['consecutive'])} samples; map={context.scene.map_id} "
        f"phase={context.phase}"
    )
    setattr(arm, "_safety_stop_reason", msg)
    try:
        response = arm.terminate_motion(fallback_joints=joints)
        print(
            "[torque] ordinary trajectory termination confirmed; "
            f"details={response}; attempting verified low-speed retreat"
        )
    except Exception as exc:
        failure_msg = (
            f"{msg}; ordinary trajectory termination could not be confirmed: "
            f"{type(exc).__name__}: {exc}; escalating because physical motion stop is unconfirmed"
        )
        setattr(arm, "_safety_stop_reason", failure_msg)
        _require_hook(hooks, "safe_stop")(arm, failure_msg)
        raise SafetyStopTriggered(failure_msg) from exc
    retreat_joints = getattr(args, "_runtime_motion_stage_start_joints", None)
    if retreat_joints is None:
        retreat_joints = np.asarray(joints, dtype=float).reshape(7)
    raise TorqueCollisionStopTriggered(
        msg,
        retreat_joints=retreat_joints,
        context=context,
        stage_label=stage_label,
    )


def check_runtime_collision_current(
    arm,
    args,
    label,
    joints=None,
    *,
    hooks,
):
    """Second-line guard using measured joints and the frozen stage map."""
    _motion_configuration_metrics = _require_hook(
        hooks, "motion_configuration_metrics"
    )
    _persist_runtime_collision_diagnostic = _require_hook(
        hooks, "persist_runtime_collision_diagnostic"
    )
    safe_stop = _require_hook(hooks, "safe_stop")
    context = getattr(args, "_runtime_motion_collision_context", None)
    if context is None or not bool(
        getattr(args, "motion_runtime_collision_guard", True)
    ):
        return None
    controlled_floor_stop_allowed = False
    floor_stop_detail = ""
    path_guard_metrics = None
    try:
        q = (
            np.asarray(arm.get_joint_angles(), dtype=float).reshape(7)
            if joints is None
            else np.asarray(joints, dtype=float).reshape(7)
        )
        _check_runtime_joint_torque(
            arm, args, label, q, context, hooks=hooks
        )
        q_start = getattr(args, "_runtime_motion_stage_start_joints", None)
        q_target = getattr(args, "_runtime_motion_stage_target_joints", None)
        if q_start is not None and q_target is not None:
            q_start = np.asarray(q_start, dtype=float).reshape(7)
            q_target = np.asarray(q_target, dtype=float).reshape(7)
            edge = q_target - q_start
            edge_norm_sq = float(np.dot(edge, edge))
            alpha_raw = (
                0.0
                if edge_norm_sq <= 1e-12
                else float(np.dot(q - q_start, edge) / edge_norm_sq)
            )
            alpha = float(np.clip(alpha_raw, 0.0, 1.0))
            projected = q_start + alpha * edge
            residual_deg = float(
                np.rad2deg(np.max(np.abs(q - projected)))
            )
            residual_limit_deg = max(
                float(
                    getattr(
                        args,
                        "motion_runtime_joint_path_residual_deg",
                        8.0,
                    )
                ),
                0.0,
            )
            confirm_samples = max(
                int(
                    getattr(
                        args,
                        "motion_runtime_joint_path_confirm_samples",
                        2,
                    )
                ),
                1,
            )
            stage_label = str(
                getattr(args, "_runtime_motion_stage_label", label)
            )
            state = getattr(args, "_runtime_joint_path_guard_state", None)
            if not isinstance(state, dict) or state.get("stage_label") != stage_label:
                state = {"stage_label": stage_label, "consecutive": 0}
                args._runtime_joint_path_guard_state = state
            if residual_limit_deg > 0.0 and residual_deg > residual_limit_deg:
                state["consecutive"] = int(state.get("consecutive", 0)) + 1
            else:
                state["consecutive"] = 0
            if int(state["consecutive"]) >= confirm_samples:
                path_guard_metrics = {
                    "margin": residual_limit_deg - residual_deg,
                    "nearest_clearance": residual_deg,
                    "required_clearance": residual_limit_deg,
                    "primitive": "measured_joint_path_residual",
                    "source": "path_constraint",
                    "path_alpha_raw": alpha_raw,
                    "path_alpha_clamped": alpha,
                    "path_residual_deg": residual_deg,
                    "path_residual_limit_deg": residual_limit_deg,
                    "confirm_samples": int(state["consecutive"]),
                }
        from .gripper_geometry import urdf_gripper_enabled, read_sdk_gripper_opening
        if urdf_gripper_enabled(args):
            # Read cached SDK feedback without printing or commanding the jaws.
            # Never carry an old successful sample across a failed read.
            args._runtime_gripper_mesh_opening = None
            args._runtime_gripper_mesh_opening = read_sdk_gripper_opening(arm, args)
        support_state = getattr(args, "_runtime_support_release_state", None)
        if context.initial_support_release is not None and (
            not isinstance(support_state, dict)
            or support_state.get("context_id") != id(context)
        ):
            support_state = {
                "context_id": id(context),
                "last_rise": None,
                "max_rise": -float("inf"),
                "strict_latched": False,
            }
            args._runtime_support_release_state = support_state
        metrics = _motion_configuration_metrics(
            q,
            context,
            args,
            runtime=True,
            support_strict_latched=bool(
                support_state.get("strict_latched", False)
            )
            if isinstance(support_state, dict)
            else False,
        )
        if context.initial_support_release is not None:
            rise = float(metrics.get("support_rise", float("nan")))
            if not np.isfinite(rise):
                raise UnsafeMotionTarget(
                    "runtime support-release TCP rise is unavailable"
                )
            previous_rise = support_state.get("last_rise", None)
            max_rise = float(
                support_state.get("max_rise", -float("inf"))
            )
            tolerance = float(
                context.initial_support_release.downward_tolerance
            )
            runtime_drop_slack = max(
                float(
                    getattr(
                        args,
                        "motion_runtime_support_drop_slack",
                        0.005,
                    )
                ),
                0.0,
            )
            allowed_runtime_drop = tolerance + runtime_drop_slack
            if (
                previous_rise is not None
                and rise < max_rise - allowed_runtime_drop
            ):
                drop = max_rise - rise
                excess = drop - allowed_runtime_drop
                metrics = {
                    "margin": -excess,
                    "nearest_clearance": -excess,
                    "required_clearance": 0.0,
                    "primitive": "support_release_runtime_drop",
                    "source": "initial_support_constraint",
                    "support_rise": rise,
                    "runtime_drop": drop,
                    "runtime_drop_allowance": allowed_runtime_drop,
                }
            support_state["last_rise"] = rise
            support_state["max_rise"] = max(
                float(support_state.get("max_rise", -float("inf"))), rise
            )
            if rise >= float(
                context.initial_support_release.strict_after_rise
            ):
                support_state["strict_latched"] = True

        # A one-command active-view return is planned against a strict TCP
        # height floor, but Nero's move_j controller does not necessarily
        # advance all seven axes at the same fraction of the commanded edge.
        # That can produce a small, short-lived modeled TCP dip even when the
        # sampled joint-linear plan rises monotonically.  Only contexts that
        # explicitly carry runtime slack may tolerate that floor violation.
        # Re-evaluate without the supplemental height floor before continuing
        # so the point-cloud environment, robot geometry, and table guard stay
        # fully active.
        runtime_floor_slack = max(
            float(getattr(context, "runtime_min_tcp_z_slack", 0.0)), 0.0
        )
        strict_downward_tolerance = max(
            float(
                getattr(
                    args,
                    "active_direct_return_downward_tolerance",
                    0.002,
                )
            ),
            0.0,
        )
        floor_margin = float(metrics.get("margin", float("nan")))
        is_explicit_runtime_floor_violation = (
            runtime_floor_slack > 0.0
            and np.isfinite(floor_margin)
            and floor_margin < 0.0
            and metrics.get("source") == "path_constraint"
            and metrics.get("primitive") == "stage_min_tcp_z"
        )
        if is_explicit_runtime_floor_violation:
            floor_metrics = metrics
            environment_context = MotionCollisionContext(
                scene=context.scene,
                phase=context.phase,
                allow_target_tool_contact=context.allow_target_tool_contact,
                attached_target_tcp=context.attached_target_tcp,
                placed_target_base=context.placed_target_base,
                min_tcp_z=None,
                runtime_min_tcp_z_slack=0.0,
                initial_support_release=context.initial_support_release,
                closed_gripper_width=getattr(
                    context, "closed_gripper_width", None
                ),
                require_support_release_latch_at_endpoint=bool(
                    getattr(
                        context,
                        "require_support_release_latch_at_endpoint",
                        True,
                    )
                ),
            )
            environment_metrics = _motion_configuration_metrics(
                q,
                environment_context,
                args,
                runtime=True,
                support_strict_latched=bool(
                    support_state.get("strict_latched", False)
                )
                if isinstance(support_state, dict)
                else False,
            )
            environment_margin = float(
                environment_metrics.get("margin", float("nan"))
            )
            environment_is_safe = (
                np.isfinite(environment_margin) and environment_margin >= 0.0
            )
            if not environment_is_safe:
                # The height floor may be the numerically worst constraint,
                # but it must never hide a simultaneous physical collision.
                # Preserve the environment result; the stop policy below
                # distinguishes a positive buffer shortfall from penetration.
                metrics = environment_metrics
            elif floor_margin >= -runtime_floor_slack:
                # The supplemental height-floor excursion is within the
                # runtime envelope; continue using the independently safe
                # environment result.
                metrics = environment_metrics
            else:
                # Environment is still clear, but the active-return height
                # envelope itself was exceeded.  Keep the original floor
                # result so the ordinary trajectory-termination branch below
                # aborts this workflow without entering electronic e-stop.
                runtime_excess = max(
                    -floor_margin - runtime_floor_slack, 0.0
                )
                metrics = dict(floor_metrics)
                metrics.update(
                    {
                        "runtime_floor_slack": runtime_floor_slack,
                        "runtime_total_downward_tolerance": (
                            strict_downward_tolerance
                            + runtime_floor_slack
                        ),
                        "runtime_floor_excess": runtime_excess,
                    }
                )
                floor_stop_detail = (
                    f"strict_floor_violation="
                    f"{-floor_margin * 1000.0:.1f}mm"
                    f" extra_runtime_slack="
                    f"{runtime_floor_slack * 1000.0:.1f}mm"
                    f" runtime_envelope_excess="
                    f"{runtime_excess * 1000.0:.1f}mm "
                )
                controlled_floor_stop_allowed = True
            warned_context_id = getattr(
                args, "_runtime_min_tcp_z_slack_warned_context_id", None
            )
            if (
                environment_is_safe
                and floor_margin >= -runtime_floor_slack
                and warned_context_id != id(context)
            ):
                print(
                    f"[protect][motion][warn] {label} modeled TCP crossed the "
                    f"strict planned return floor by {-floor_margin * 1000.0:.1f}mm; "
                    f"within extra runtime slack={runtime_floor_slack * 1000.0:.1f}mm "
                    f"(total allowed drop from the side-view start="
                    f"{(runtime_floor_slack + strict_downward_tolerance) * 1000.0:.1f}mm). "
                    "Point-cloud collision and table guards remain active."
                )
                args._runtime_min_tcp_z_slack_warned_context_id = id(context)
        if path_guard_metrics is not None and float(
            metrics.get("margin", float("nan"))
        ) >= 0.0:
            # A large asynchronous-axis excursion can move the TCP, camera or
            # fingers away from the collision-validated joint edge even when
            # the endpoint itself is safe.  Stop before it grows into a
            # geometric collision, but never mask a simultaneous physical
            # clearance/table violation that needs its own stricter policy.
            metrics = path_guard_metrics
    except (SafetyStopTriggered, TorqueCollisionStopTriggered):
        raise
    except Exception as exc:
        msg = (
            f"{label} runtime collision guard failed closed: "
            f"{type(exc).__name__}: {exc}"
        )
        setattr(arm, "_safety_stop_reason", msg)
        safe_stop(arm, msg)
        raise SafetyStopTriggered(msg) from exc
    finger_clearance_deferred = False
    soft_finger_clearance = bool(
        float(metrics.get("margin", float("nan"))) < 0.0
        and str(context.phase) in {"free_open", "target_approach"}
        and metrics.get("source") == "environment"
        and str(metrics.get("primitive", "")).startswith("gripper_finger_")
        and np.isfinite(float(metrics.get("nearest_clearance", float("nan"))))
        and float(metrics.get("nearest_clearance", 0.0)) >= 0.0
        and float(metrics.get("required_clearance", 0.0)) > 0.0
    )
    if soft_finger_clearance:
        confirm_required = max(
            int(
                getattr(
                    args,
                    "motion_runtime_finger_clearance_confirm_samples",
                    2,
                )
            ),
            1,
        )
        state = getattr(args, "_runtime_finger_clearance_guard_state", None)
        state_key = (
            id(context),
            str(getattr(args, "_runtime_motion_stage_label", label)),
            str(metrics.get("primitive", "")),
        )
        if not isinstance(state, dict) or state.get("key") != state_key:
            state = {"key": state_key, "consecutive": 0}
            args._runtime_finger_clearance_guard_state = state
        state["consecutive"] = int(state.get("consecutive", 0)) + 1
        metrics = dict(metrics)
        metrics["runtime_confirmation_samples"] = int(state["consecutive"])
        metrics["runtime_confirmation_required"] = confirm_required
        finger_clearance_deferred = bool(
            int(state["consecutive"]) < confirm_required
        )
        if finger_clearance_deferred:
            print(
                f"[protect][finger][confirm] {label} "
                f"{metrics['primitive']} clearance="
                f"{float(metrics['nearest_clearance']) * 1000.0:.1f}mm < "
                f"{float(metrics['required_clearance']) * 1000.0:.1f}mm; "
                f"sample {int(state['consecutive'])}/{confirm_required}, "
                "waiting for confirmation"
            )
    else:
        state = getattr(args, "_runtime_finger_clearance_guard_state", None)
        if isinstance(state, dict):
            state["consecutive"] = 0
    if finger_clearance_deferred:
        return metrics
    if not np.isfinite(float(metrics["margin"])) or float(metrics["margin"]) < 0.0:
        _persist_runtime_collision_diagnostic(
            args, label, q, context, metrics
        )
        nearest_clearance = float(
            metrics.get("nearest_clearance", float("nan"))
        )
        required_clearance = float(
            metrics.get("required_clearance", float("nan"))
        )
        # Point-cloud geometry can report a small negative clearance because
        # of depth noise, self-mask leakage, capsule inflation, or measured
        # joint-path deviation even when no physical contact occurred.  Treat
        # every finite point-cloud clearance violation as a controlled motion
        # stop first.  Non-point-cloud constraints and modeled payload-vs-arm
        # penetration retain the stricter branch; an unconfirmed ordinary stop
        # is still escalated below because physical motion was not proven to
        # have stopped.
        pointcloud_clearance_source = metrics.get("source") in {
            "environment",
            "selected_target",
            "placed_target",
        }
        controlled_clearance_stop_allowed = bool(
            metrics.get("source")
            in {
                "environment",
                "selected_target",
                "placed_target",
                "attached_target_vs_robot",
            }
            and np.isfinite(nearest_clearance)
            and np.isfinite(required_clearance)
            and required_clearance > 0.0
            and nearest_clearance < required_clearance
            and (nearest_clearance > 0.0 or pointcloud_clearance_source)
        )
        if metrics.get("primitive") == "measured_joint_path_residual":
            msg = (
                f"{label} runtime trajectory stop: measured joint-path "
                f"residual={float(metrics['path_residual_deg']):.2f}deg > "
                f"{float(metrics['path_residual_limit_deg']):.2f}deg for "
                f"{int(metrics['confirm_samples'])} samples; "
                f"map={context.scene.map_id} phase={context.phase}"
            )
        elif metrics["source"] in {
            "initial_support_constraint",
            "path_constraint",
        }:
            msg = (
                f"{label} runtime motion stop: {metrics['primitive']} "
                f"margin={float(metrics['nearest_clearance']) * 1000.0:.1f}mm "
                f"{floor_stop_detail}"
                f"map={context.scene.map_id} phase={context.phase}"
            )
        else:
            stop_kind = (
                "runtime clearance stop"
                if controlled_clearance_stop_allowed
                else "runtime collision stop"
            )
            msg = (
                f"{label} {stop_kind}: {metrics['primitive']} clearance="
                f"{nearest_clearance * 1000.0:.1f}mm to "
                f"{metrics['source']}, required >= "
                f"{required_clearance * 1000.0:.1f}mm "
                f"map={context.scene.map_id} phase={context.phase}"
            )
        setattr(arm, "_safety_stop_reason", msg)
        controlled_floor_stop = bool(
            controlled_floor_stop_allowed
            and metrics.get("source") == "path_constraint"
            and metrics.get("primitive") == "stage_min_tcp_z"
        )
        controlled_support_stop = bool(
            metrics.get("source") == "initial_support_constraint"
            and metrics.get("primitive")
            in {
                "support_release_xy_drift",
                "support_release_tilt",
                "support_release_runtime_drop",
            }
        )
        controlled_path_residual_stop = bool(
            metrics.get("source") == "path_constraint"
            and metrics.get("primitive") == "measured_joint_path_residual"
        )
        if (
            controlled_floor_stop
            or controlled_clearance_stop_allowed
            or controlled_support_stop
            or controlled_path_residual_stop
        ):
            print(f"[motion_stop] {msg}")
            try:
                response = arm.terminate_motion(fallback_joints=q)
                print(
                    "[motion_stop] ordinary trajectory termination confirmed; "
                    f"details={response}"
                )
                print(
                    "[motion_stop] electronic emergency stop was not requested; "
                    "the current workflow is terminal and all later commands are blocked"
                )
            except Exception as exc:
                failure_msg = (
                    f"{msg}; ordinary trajectory termination could not be confirmed: "
                    f"{type(exc).__name__}: {exc}; escalating because physical "
                    "motion stop is unconfirmed"
                )
                setattr(arm, "_safety_stop_reason", failure_msg)
                print(f"[motion_stop][critical] {failure_msg}")
                safe_stop(arm, failure_msg)
                raise SafetyStopTriggered(failure_msg) from exc
            raise ControlledMotionStopTriggered(msg)
        safe_stop(arm, msg)
        raise SafetyStopTriggered(msg)
    return metrics
