"""One auditable result record for each top-level grasp run."""

from __future__ import annotations

import json
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np


SCHEMA_VERSION = 1


def _now_iso():
    return datetime.now().astimezone().isoformat(timespec="milliseconds")


def _finite(value):
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _state(args):
    return getattr(args, "_experiment_log_state", None)


def initialize_experiment_log(args, argv=None):
    """Create in-memory telemetry before preflight or hardware access."""
    now = datetime.now().astimezone()
    run_id = now.strftime("%Y-%m-%d_%H-%M-%S_%f") + f"_p{os.getpid()}"
    base_dir = Path(getattr(args, "save_dir", "logs") or "logs")
    run_dir = base_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    # Every camera view and diagnostic produced by this process is nested
    # below the same run directory.  This makes one run distinct from one
    # camera capture during active multi-view.
    args.save_dir = str(run_dir)
    state = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "started_at": now.isoformat(timespec="milliseconds"),
        "stage": "startup",
        "stage_events": [
            {"stage": "startup", "timestamp": now.isoformat(timespec="milliseconds")}
        ],
        "target": {
            "requested_class": getattr(args, "target_class", None),
            "selected_label": None,
        },
        "method": {
            "contact_check_only": bool(getattr(args, "grasp_contact_check_only", False)),
            "physical_table_reference_z_m": _finite(getattr(args, "grasp_table_reference_z", None)),
            "grasp_generator": str(getattr(args, "grasp_generator", "graspnet")),
            "nero_grasp_refine": bool(getattr(args, "nero_grasp_refine", False)),
            "graspgenx_table_repair_pool": (
                getattr(args, "grasp_generator", "graspnet") == "graspgenx"
                and bool(getattr(args, "nero_grasp_refine", False))
            ),
            "mgpc": bool(getattr(args, "mgpc", False)),
            "mgpc_seed": int(getattr(args, "mgpc_seed", 0)),
            "graspgenx_seed": int(getattr(args, "graspgenx_seed", 42)),
            "post_grasp_hold_joints_deg": getattr(args, "post_grasp_hold_joints_deg", None),
        },
        "artifacts": {
            "run_dir": str(run_dir),
            "primary_capture_dir": None,
        },
        "views": {
            "used_for_final_reconstruction": None,
            "multiview_used": False,
            "active_view_reason": None,
        },
        "path_clearance": {
            "minimum_planned_extra_margin_m": None,
            "minimum_live_preflight_clearance_m": None,
            "required_clearance_m_at_minimum": None,
            "minimum_live_preflight_margin_m": None,
            "primitive_at_minimum": None,
            "source_at_minimum": None,
            "stage_at_minimum": None,
            "minimum_runtime_modeled_clearance_from_measured_joints_m": None,
            "required_runtime_clearance_m_at_minimum": None,
            "minimum_runtime_modeled_margin_m": None,
            "runtime_primitive_at_minimum": None,
            "runtime_source_at_minimum": None,
            "runtime_stage_at_minimum": None,
            "runtime_phase_at_minimum": None,
            "runtime_minimum_timestamp": None,
            "runtime_measured_joints_deg_at_minimum": None,
            "runtime_measured_sample_count": 0,
        },
        "corrections": {
            "width_recenter_m": None,
            "abs_width_recenter_m": None,
            "jaw_orientation_delta_deg": None,
        },
        "selected_grasp": {},
        "gripper_close": {},
        "grasp_stability": {
            "available": False,
            "stable": None,
            "proxy_score": None,
            "measurement": "post_lift_grip_retention_proxy",
            "definition": (
                "width drift within tolerance and post-lift median force at least "
                "half of close-contact force when force feedback is available"
            ),
        },
        "command": list(sys.argv if argv is None else argv),
        "configuration": {
            "phase": getattr(args, "phase", "all"),
            "execute": bool(getattr(args, "execute", False)),
            "active_view": bool(getattr(args, "active_view", False)),
            "active_side_view_count": getattr(args, "active_side_view_count", None),
            "yolo_weights": getattr(args, "yolo_weights", None),
            "calib_json": getattr(args, "calib_json", None),
        },
        "_started_monotonic": time.monotonic(),
        "_base_dir": str(base_dir),
        "_run_dir": str(run_dir),
        "_finalized": False,
    }
    args._experiment_log_state = state
    return run_id


def mark_experiment_stage(args, stage, *, failure=False):
    # Workflow labels describe the slow task. Only the trajectory executor may
    # change the real control phase (hold/trajectory/stop); overwriting "hold"
    # here disabled fixed-hold recovery during perception/planning.
    state = _state(args)
    if state is None:
        return
    stage = str(stage)
    state["stage"] = stage
    state["stage_events"].append({"stage": stage, "timestamp": _now_iso()})
    if failure and state.get("failure_stage") is None:
        state["failure_stage"] = stage


def set_experiment_session(args, out_dir, *, target_label=None):
    state = _state(args)
    if state is None:
        return
    if state["artifacts"].get("primary_capture_dir") is None:
        state["artifacts"]["primary_capture_dir"] = str(
            Path(out_dir).resolve()
        )
    if target_label is not None:
        state["target"]["selected_label"] = str(target_label)


def record_experiment_views(args, active_result):
    state = _state(args)
    if state is None:
        return
    state["views"] = {
        "used_for_final_reconstruction": int(active_result.kept_view_count),
        "multiview_used": bool(active_result.used_multiview),
        "active_view_reason": str(active_result.reason),
    }


def record_selected_grasp(args, row, width_shift, plan=None):
    state = _state(args)
    if state is None:
        return
    row = {} if row is None else row
    selected = {}
    for key in (
        "idx",
        "raw_score",
        "recommendation_score",
        "center_dist",
        "graspnet_width",
        "recommended_contact_points",
        "recommended_robust_span",
        "genx_selected_mesh_table_clearance_m",
    ):
        value = row.get(key)
        selected[key] = int(value) if key in {"idx", "recommended_contact_points"} and value is not None else _finite(value)
    selected["generator"] = str(row.get("generator", getattr(args, "grasp_generator", "graspnet")))
    selected["nero_refinement_applied"] = bool(row.get("nero_refinement_applied", False))
    selected["nero_refinement_family"] = row.get("nero_refinement_family")
    selected["nero_refinement_advance_m"] = _finite(row.get("nero_refinement_advance_m"))
    if row.get("generator") == "graspgenx":
        selected["genx_native_table_safe"] = row.get("genx_native_table_safe")
        selected["genx_repaired_from_table_rejection"] = bool(
            row.get("genx_native_table_safe") is False
            and row.get("nero_refinement_applied", False))
    selected["generator_index"] = (
        None if row.get("generator_index") is None else int(row["generator_index"])
    )
    state["selected_grasp"] = selected
    shift = _finite(width_shift)
    state["corrections"] = {
        "width_recenter_m": shift,
        "abs_width_recenter_m": None if shift is None else abs(shift),
        "jaw_orientation_delta_deg": _finite(
            row.get("jaw_selected_orientation_delta_deg")
        ),
    }
    if plan is not None:
        margins = []
        for group in ("before_close", "after_close", "after_release", "empty_recovery"):
            for stage in getattr(plan, group, ()):
                value = _finite(getattr(stage, "path_clearance_margin", None))
                if value is not None:
                    margins.append(value)
        if margins:
            state["path_clearance"]["minimum_planned_extra_margin_m"] = min(margins)


def record_live_path_clearance(args, metrics, stage_label):
    state = _state(args)
    if state is None or not isinstance(metrics, dict):
        return
    clearance = _finite(metrics.get("nearest_clearance"))
    if clearance is None:
        return
    current = state["path_clearance"].get("minimum_live_preflight_clearance_m")
    if current is not None and clearance >= current:
        return
    state["path_clearance"].update(
        {
            "minimum_live_preflight_clearance_m": clearance,
            "required_clearance_m_at_minimum": _finite(
                metrics.get("required_clearance")
            ),
            "minimum_live_preflight_margin_m": _finite(metrics.get("margin")),
            "primitive_at_minimum": str(metrics.get("primitive", "unknown")),
            "source_at_minimum": str(metrics.get("source", "unknown")),
            "stage_at_minimum": str(stage_label),
        }
    )


def record_runtime_measured_clearance(
    args,
    metrics,
    stage_label,
    phase,
    measured_joints,
):
    """Keep the smallest modeled gap evaluated at an actual encoder sample."""
    state = _state(args)
    if state is None or not isinstance(metrics, dict):
        return
    path = state["path_clearance"]
    path["runtime_measured_sample_count"] = int(
        path.get("runtime_measured_sample_count", 0)
    ) + 1
    clearance = _finite(metrics.get("geometric_nearest_clearance"))
    if clearance is None:
        return
    current = path.get(
        "minimum_runtime_modeled_clearance_from_measured_joints_m"
    )
    if current is not None and clearance >= current:
        return
    required = _finite(metrics.get("geometric_required_clearance"))
    joints = np.asarray(measured_joints, dtype=float).reshape(7)
    path.update(
        {
            "minimum_runtime_modeled_clearance_from_measured_joints_m": clearance,
            "required_runtime_clearance_m_at_minimum": required,
            "minimum_runtime_modeled_margin_m": (
                None if required is None else clearance - required
            ),
            "runtime_primitive_at_minimum": str(
                metrics.get("geometric_primitive", "unknown")
            ),
            "runtime_source_at_minimum": str(
                metrics.get("geometric_source", "unknown")
            ),
            "runtime_stage_at_minimum": str(stage_label),
            "runtime_phase_at_minimum": str(phase),
            "runtime_minimum_timestamp": _now_iso(),
            "runtime_measured_joints_deg_at_minimum": np.rad2deg(joints).tolist(),
        }
    )


def record_gripper_close(args, close_result):
    state = _state(args)
    if state is None:
        return
    status = getattr(close_result, "status", close_result)
    state["gripper_close"] = {
        "contact_detected": getattr(close_result, "contact_detected", None),
        "stop_reason": getattr(close_result, "stop_reason", None),
        "contact_width_m": _finite(getattr(close_result, "contact_width", None)),
        "contact_force_n": _finite(getattr(close_result, "contact_force", None)),
        "final_width_m": _finite(getattr(status, "value", None)),
        "final_force_n": _finite(getattr(status, "force", None)),
    }


def sample_grasp_stability(arm, args, close_result):
    """Measure width retention after lift while the gripper remains closed."""
    state = _state(args)
    if state is None:
        return None
    requested = max(int(getattr(args, "grasp_stability_samples", 5)), 1)
    interval = max(float(getattr(args, "grasp_stability_interval", 0.10)), 0.0)
    tolerance = max(
        float(getattr(args, "grasp_stability_max_width_drift", 0.002)), 1e-6
    )
    widths = []
    forces = []
    for index in range(requested):
        status = arm.print_gripper_status()
        width = _finite(getattr(status, "value", None))
        force = _finite(getattr(status, "force", None))
        if width is not None:
            widths.append(width)
        if force is not None:
            forces.append(abs(force))
        if index + 1 < requested and interval > 0.0:
            time.sleep(interval)

    reference = _finite(getattr(close_result, "contact_width", None))
    if reference is None:
        status = getattr(close_result, "status", close_result)
        reference = _finite(getattr(status, "value", None))
    if reference is None and widths:
        reference = widths[0]
    drift = (
        None
        if reference is None or not widths
        else max(abs(value - reference) for value in widths)
    )
    valid_fraction = float(len(widths)) / float(requested)
    proxy = (
        None
        if drift is None
        else valid_fraction * max(0.0, 1.0 - drift / tolerance)
    )
    contact_force = _finite(getattr(close_result, "contact_force", None))
    force_ratio = (
        None
        if contact_force is None or contact_force <= 0.0 or not forces
        else float(np.median(forces)) / abs(contact_force)
    )
    empty_threshold = float(getattr(args, "empty_grasp_width_threshold", 0.001))
    stable = bool(
        len(widths) == requested
        and drift is not None
        and drift <= tolerance
        and min(widths) > empty_threshold
        and (force_ratio is None or force_ratio >= 0.5)
    )
    result = {
        "available": bool(widths),
        "stable": stable if widths else None,
        "proxy_score": proxy,
        "definition": state["grasp_stability"]["definition"],
        "measurement": "post_lift_grip_retention_proxy",
        "sample_point": "after_lift_gripper_still_closed",
        "requested_samples": requested,
        "valid_width_samples": len(widths),
        "configured_drift_tolerance_m": tolerance,
        "reference_width_m": reference,
        "width_mean_m": None if not widths else float(np.mean(widths)),
        "width_std_m": None if not widths else float(np.std(widths)),
        "max_abs_width_drift_m": drift,
        "force_mean_n": None if not forces else float(np.mean(forces)),
        "force_min_n": None if not forces else float(np.min(forces)),
        "force_retention_ratio": force_ratio,
        "width_samples_m": widths,
        "force_samples_n": forces,
    }
    state["grasp_stability"] = result
    return result


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def finalize_experiment_log(args, result=None, *, terminal_status=None, exception=None):
    """Write exactly one global JSONL row and one per-main-capture JSON file."""
    state = _state(args)
    if state is None or state.get("_finalized"):
        return None
    state["_finalized"] = True
    status = terminal_status
    detail = None
    motion_sent = bool(getattr(args, "_workflow_motion_sent", False))
    if result is not None:
        raw_status = getattr(result, "status", None)
        status = getattr(raw_status, "value", raw_status)
        detail = getattr(result, "detail", None)
        motion_sent = bool(getattr(result, "motion_sent", False) or motion_sent)
    status = "internal_error" if status is None else str(status)
    non_failure = {"success", "planned_only", "contact_checked"}
    finished = _now_iso()
    record = {
        key: value for key, value in state.items() if not key.startswith("_")
    }
    record.update(
        {
            "finished_at": finished,
            "total_time_s": float(time.monotonic() - state["_started_monotonic"]),
            "outcome": {
                "status": status,
                "success": status == "success",
                "motion_sent": motion_sent,
                "failure_stage": (
                    None
                    if status in non_failure
                    else state.get("failure_stage", state.get("stage"))
                ),
                "detail": detail,
                "exception_type": None if exception is None else type(exception).__name__,
                "exception_message": None if exception is None else str(exception),
            },
        }
    )
    record = _json_safe(record)
    root = Path(state["_base_dir"])
    run_dir = Path(state["_run_dir"])
    try:
        root.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record, ensure_ascii=False, allow_nan=False)
        with (root / "experiment_results.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        output = run_dir / "experiment_result.json"
        temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(record, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, output)
        print(
            f"[experiment] run_id={record['run_id']} status={status} "
            f"duration={record['total_time_s']:.3f}s log={root / 'experiment_results.jsonl'}"
        )
        return record
    except Exception as write_exc:
        print(
            "[experiment][warn] failed to persist final result: "
            f"{type(write_exc).__name__}: {write_exc}"
        )
        return None
