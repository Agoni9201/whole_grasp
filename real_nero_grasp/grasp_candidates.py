"""GraspNet candidate metadata, physical validation, and selection helpers.

The functions in this module deliberately keep the runner's existing
candidate policy and log messages intact.  Stateful collaborators are exposed
as keyword-only hooks so the compatibility facade can continue to honour
tests and deployments that monkeypatch legacy private names.
"""

import copy
import json
import os

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R

from .types import (
    GraspSelectionDeclined,
    InvalidGraspSelection,
    PlanNotExecutable,
)


__all__ = [
    "_candidate_to_jsonable",
    "grasp_camera_summary",
    "save_selected_grasp_pose",
    "save_grasp_candidates_json",
    "grasp_width_m",
    "annotate_grasp_candidate_physical_aperture",
    "grasp_candidate_width_shift",
    "filter_grasp_candidates_by_width",
    "filter_grasp_candidates_by_reachability",
    "filter_grasp_candidates_by_workflow_preflight",
    "filter_grasp_candidates_by_table_guard",
    "grasp_candidate_too_wide",
    "_ordered_equivalent_jaw_width_signs",
    "_filter_equivalent_jaw_width_signs",
    "_candidate_preflight_width_signs",
    "_defer_candidate_endpoint_reachability",
    "annotate_grasp_candidate_reachability",
    "execution_gripper_open_width",
    "_project_camera_point",
    "_draw_grasp_pose_axes",
    "recommend_grasp_candidate",
    "draw_grasp_candidate_overlay",
    "select_grasp_candidate",
    "choose_default_grasp_candidate",
    "focus_graspnet_target_indices",
]


def focus_graspnet_target_indices(
    points,
    *,
    T_base_camera=None,
    enabled=True,
    lower_quantile=0.25,
    upright_lower_quantile=0.15,
    elongated_longitudinal_range=(0.15, 0.85),
    elongated_threshold=1.5,
    upright_axis_cos=0.75,
    min_retained_ratio=0.50,
    min_retained_points=1000,
):
    """Keep the GraspNet-visible portion of a target, dropping table-adjacent bulk.

    Returns (indices, info).  If the crop would be too small, keep every point.
    """
    points = np.asarray(points, dtype=float)
    info = {
        "enabled": bool(enabled),
        "applied": False,
        "fallback": "none",
        "input_points": int(points.shape[0]) if points.ndim == 2 else 0,
        "retained_points": int(points.shape[0]) if points.ndim == 2 else 0,
        "elongation": 1.0,
        "upright": False,
        "elongated": False,
        "lower_quantile": float(lower_quantile),
    }
    if points.ndim != 2 or points.shape[0] == 0:
        info["fallback"] = "empty"
        return np.zeros((0,), dtype=int), info
    all_indices = np.arange(points.shape[0], dtype=int)
    if not bool(enabled):
        info["fallback"] = "disabled"
        return all_indices, info
    if T_base_camera is None:
        info["fallback"] = "missing_transform"
        return all_indices, info

    T = np.asarray(T_base_camera, dtype=float).reshape(4, 4)
    ones = np.ones((points.shape[0], 1), dtype=float)
    base = (T @ np.hstack([points, ones]).T).T[:, :3]
    finite = np.isfinite(base).all(axis=1)
    if int(np.count_nonzero(finite)) < 16:
        info["fallback"] = "too_few_finite_points"
        return all_indices, info

    work = base[finite]
    center = np.median(work, axis=0)
    centered = work - center.reshape(1, 3)
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(np.cov(centered, rowvar=False))
    except (ValueError, np.linalg.LinAlgError):
        info["fallback"] = "pca_failed"
        return all_indices, info
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(np.asarray(eigenvalues)[order], 0.0)
    long_axis = np.asarray(eigenvectors[:, order[0]], dtype=float).reshape(3)
    long_norm = float(np.linalg.norm(long_axis)) + 1e-12
    long_axis = long_axis / long_norm
    if long_axis[2] < 0.0:
        long_axis = -long_axis
    elongation = float(
        np.sqrt(eigenvalues[0] / max(float(eigenvalues[1]), 1e-12))
    )
    upright = bool(
        elongation >= float(elongated_threshold)
        and abs(float(long_axis[2])) >= float(upright_axis_cos)
    )
    elongated = bool(elongation >= float(elongated_threshold) and not upright)
    info["elongation"] = elongation
    info["upright"] = upright
    info["elongated"] = elongated

    keep = np.ones((work.shape[0],), dtype=bool)
    if upright:
        heights = work[:, 2]
        cutoff = float(np.quantile(heights, float(upright_lower_quantile)))
        keep = heights >= cutoff
        info["lower_quantile"] = float(upright_lower_quantile)
    elif elongated:
        projection = centered @ long_axis
        lo, hi = elongated_longitudinal_range
        q_lo = float(np.quantile(projection, float(lo)))
        q_hi = float(np.quantile(projection, float(hi)))
        keep = (projection >= q_lo) & (projection <= q_hi)
        info["lower_quantile"] = float(lo)
    else:
        heights = work[:, 2]
        cutoff = float(np.quantile(heights, float(lower_quantile)))
        keep = heights >= cutoff
        info["lower_quantile"] = float(lower_quantile)

    finite_indices = np.flatnonzero(finite)
    focused = finite_indices[keep]
    retained = int(focused.size)
    info["retained_points"] = retained
    min_count = max(int(min_retained_points), 1)
    min_ratio = float(np.clip(min_retained_ratio, 0.0, 1.0))
    if retained < min_count or retained < min_ratio * float(points.shape[0]):
        info["fallback"] = "too_few_retained"
        info["retained_points"] = int(points.shape[0])
        return all_indices, info
    info["applied"] = True
    return focused.astype(int), info


def _candidate_open_finger_min_z(row, args):
    """Return the lowest modeled open-finger surface at the grasp pose.

    This is a recommendation diagnostic, not another hard planning gate. The
    complete workflow preflight remains authoritative.
    """
    pose = row.get("grasp_tcp_preview", None)
    if pose is None:
        return None
    try:
        pose = np.asarray(pose, dtype=float).reshape(6)
        if not np.isfinite(pose).all():
            return None
        from .collision import _motion_open_gripper_finger_capsules
        from .transforms import _make_T_from_pose

        capsules = _motion_open_gripper_finger_capsules(
            _make_T_from_pose(pose),
            args,
            opening=float(getattr(args, "gripper_open_width", 0.100)),
            primitive_prefix="recommendation_finger",
        )
        return float(
            min(
                min(float(a[2]), float(b[2])) - float(radius)
                for a, b, radius, _name in capsules
            )
        )
    except Exception:
        return None


def recommend_grasp_candidate(rows, args, *, annotate=True):
    """Choose a geometrically credible grasp among fully executable rows.

    Executability alone is not grasp quality.  In particular, an elongated
    target must be clamped *across* its principal axis.  The recommendation
    also uses GraspNet confidence and actual target support inside the closing
    slab.  If no row satisfies the quality gates, the best executable row is
    retained only as an explicitly low-confidence fallback.
    """
    rows = list(rows)
    if not rows:
        return None

    table_z = getattr(args, "table_z", None)
    try:
        table_z = float(table_z)
        if not np.isfinite(table_z):
            table_z = None
    except (TypeError, ValueError):
        table_z = None
    required_margin = max(
        float(getattr(args, "recommended_grasp_finger_table_margin", 0.003)),
        0.0,
    )
    min_cross_axis = float(
        np.clip(
            getattr(args, "recommended_grasp_min_cross_axis_score", 0.45),
            0.0,
            1.0,
        )
    )
    min_raw_score = max(
        float(getattr(args, "recommended_grasp_min_raw_score", 0.10)), 0.0
    )
    min_contact_points = max(
        int(getattr(args, "recommended_grasp_min_contact_points", 80)), 0
    )

    def _selected_aperture_metrics(row):
        variants = [
            variant
            for variant in row.get("physical_aperture_variants", [])
            if bool(variant.get("ok", False))
        ]
        if not variants:
            return {}
        selected_sign = row.get("tcp_width_sign", None)
        if selected_sign is not None:
            matches = [
                variant
                for variant in variants
                if abs(
                    float(variant.get("width_sign", 0.0))
                    - float(selected_sign)
                )
                < 0.5
            ]
            if matches:
                variants = matches
        variant = max(
            variants,
            key=lambda item: int(item.get("points", 0)),
        )
        return {
            "points": int(variant.get("points", 0)),
            "span": variant.get("robust_span", None),
            "left_points": int(variant.get("left_contact_points", 0)),
            "right_points": int(variant.get("right_contact_points", 0)),
            "bilateral_balance": float(
                variant.get("bilateral_balance", 0.5)
            ),
            "bilateral_score": float(
                variant.get("bilateral_support_score", 0.5)
            ),
            "width_shift": float(variant.get("tcp_width_shift", 0.0)),
        }

    finite_raw_scores = [
        float(row.get("raw_score", 0.0))
        for row in rows
        if np.isfinite(float(row.get("raw_score", 0.0)))
    ]
    raw_min = min(finite_raw_scores) if finite_raw_scores else 0.0
    raw_max = max(finite_raw_scores) if finite_raw_scores else 1.0
    raw_range = raw_max - raw_min

    for row in rows:
        finger_min_z = _candidate_open_finger_min_z(row, args)
        table_margin = (
            None
            if finger_min_z is None or table_z is None
            else float(finger_min_z - table_z)
        )
        table_safe = (
            None if table_margin is None else bool(table_margin >= required_margin)
        )
        row["recommended_open_finger_min_z"] = finger_min_z
        row["recommended_table_margin"] = table_margin
        row["recommended_table_safe"] = table_safe
        row["recommended_grasp"] = False
        direction_applicable = bool(
            row.get("target_direction_applicable", False)
        )
        cross_axis = row.get("target_cross_axis_score", None)
        try:
            cross_axis = float(cross_axis)
            if not np.isfinite(cross_axis):
                cross_axis = None
        except (TypeError, ValueError):
            cross_axis = None
        direction_ok = bool(
            not direction_applicable
            or (cross_axis is not None and cross_axis >= min_cross_axis)
        )
        raw_available = "raw_score" in row
        raw_score = float(row.get("raw_score", 0.0))
        raw_ok = bool(not raw_available or raw_score >= min_raw_score)
        aperture = _selected_aperture_metrics(row)
        contact_points = aperture.get("points", None)
        robust_span = aperture.get("span", None)
        if robust_span is not None:
            try:
                robust_span = float(robust_span)
                if not np.isfinite(robust_span):
                    robust_span = None
            except (TypeError, ValueError):
                robust_span = None
        contact_ok = bool(
            contact_points is None or contact_points >= min_contact_points
        )
        table_quality_ok = bool(table_safe is not False)
        quality_ok = bool(
            direction_ok and raw_ok and contact_ok and table_quality_ok
        )

        direction_score = (
            float(np.clip(cross_axis, 0.0, 1.0))
            if cross_axis is not None
            else 0.5
        )
        raw_score_normalized = (
            float(np.clip((raw_score - raw_min) / raw_range, 0.0, 1.0))
            if raw_range > 1e-9
            else float(np.clip(raw_score, 0.0, 1.0))
        )
        bilateral_score = float(
            np.clip(aperture.get("bilateral_score", 0.5), 0.0, 1.0)
        )
        nominal_width = row.get("graspnet_nominal_width", None)
        try:
            nominal_width = float(nominal_width)
        except (TypeError, ValueError):
            nominal_width = None
        fit_score = 0.5
        if (
            robust_span is not None
            and nominal_width is not None
            and np.isfinite(nominal_width)
            and nominal_width > 1e-6
        ):
            fit_score = float(
                np.clip(robust_span / nominal_width, 0.0, 1.0)
            )

        route_l2 = row.get("workflow_preflight_route_l2_deg", None)
        route_max_step = row.get(
            "workflow_preflight_route_max_step_deg", None
        )
        wrist_sweep = row.get(
            "workflow_preflight_wrist_sweep_deg", None
        )

        def _inverse_cost(value, scale):
            if value is None:
                return 0.5
            try:
                value = float(value)
            except (TypeError, ValueError):
                return 0.5
            if not np.isfinite(value):
                return 0.0
            return float(np.clip(1.0 - value / float(scale), 0.0, 1.0))

        route_score = (
            0.40 * _inverse_cost(route_l2, 360.0)
            + 0.30 * _inverse_cost(route_max_step, 60.0)
            + 0.30 * _inverse_cost(wrist_sweep, 180.0)
        )

        orientation_delta = row.get(
            "jaw_selected_orientation_delta_deg", None
        )
        if orientation_delta is None:
            orientation_delta = (
                180.0
                if str(row.get("jaw_selected_role", "primary"))
                == "fallback_180"
                else 0.0
            )
        orientation_score = float(
            np.clip(
                1.0 - abs(float(orientation_delta)) / 180.0,
                0.0,
                1.0,
            )
        )
        center_score = float(
            np.clip(row.get("center_score", 0.5), 0.0, 1.0)
        )
        table_score = (
            0.5
            if table_margin is None
            else float(np.clip(table_margin / 0.020, 0.0, 1.0))
        )
        recommendation_score = (
            0.10 * raw_score_normalized
            + 0.25 * bilateral_score
            + 0.25 * direction_score
            + 0.10 * fit_score
            + 0.10 * route_score
            + 0.15 * orientation_score
            + 0.025 * center_score
            + 0.025 * table_score
        )
        row["recommended_direction_ok"] = direction_ok
        row["recommended_raw_score_ok"] = raw_ok
        row["recommended_contact_ok"] = contact_ok
        row["recommended_contact_points"] = contact_points
        row["recommended_robust_span"] = robust_span
        row["recommended_quality_ok"] = quality_ok
        row["recommended_raw_score_normalized"] = raw_score_normalized
        row["recommended_left_contact_points"] = aperture.get(
            "left_points", None
        )
        row["recommended_right_contact_points"] = aperture.get(
            "right_points", None
        )
        row["recommended_bilateral_balance"] = aperture.get(
            "bilateral_balance", None
        )
        row["recommended_bilateral_score"] = bilateral_score
        row["recommended_width_fit_score"] = fit_score
        row["recommended_route_score"] = route_score
        row["recommended_orientation_score"] = orientation_score
        row["recommended_center_score"] = center_score
        row["recommended_table_score"] = table_score
        row["recommendation_score"] = float(recommendation_score)

    # Quality is ranked continuously. Only the table-safety preference remains
    # categorical here; the earlier planning filters remain authoritative.
    table_safe_rows = [
        row
        for row in rows
        if row.get("recommended_table_safe", None) is not False
    ]
    pool = table_safe_rows if table_safe_rows else rows

    recommended = max(
        pool,
        key=lambda row: (
            float(row.get("recommendation_score", 0.0)),
            float(row.get("raw_score", 0.0)),
            float(row.get("priority", 0.0)),
        ),
    )
    if annotate:
        recommended["recommended_grasp"] = True
        role = str(recommended.get("jaw_selected_role", "primary"))
        margin = recommended.get("recommended_table_margin", None)
        margin_text = (
            "unknown" if margin is None else f"{float(margin) * 1000.0:.1f}mm"
        )
        quality_label = (
            "quality_gates_passed"
            if bool(recommended.get("recommended_quality_ok", False))
            else "low_confidence_fallback"
        )
        recommended["recommended_reason"] = (
            "passed complete workflow preflight; "
            f"label={quality_label}; "
            f"jaw_role={role}; open_finger_table_margin={margin_text}; "
            f"cross_axis={recommended.get('target_cross_axis_score', None)}; "
            f"contact_points="
            f"{recommended.get('recommended_left_contact_points', None)}/"
            f"{recommended.get('recommended_right_contact_points', None)}; "
            f"bilateral="
            f"{float(recommended.get('recommended_bilateral_score', 0.0)):.4f}; "
            f"raw_score={float(recommended.get('raw_score', 0.0)):.4f}; "
            f"raw_normalized="
            f"{float(recommended.get('recommended_raw_score_normalized', 0.0)):.4f}; "
            f"route={float(recommended.get('recommended_route_score', 0.0)):.4f}; "
            f"orientation="
            f"{float(recommended.get('recommended_orientation_score', 0.0)):.4f}; "
            f"recommendation_score={float(recommended.get('recommendation_score', 0.0)):.4f}"
        )
    return recommended


def _resolve_hook(module_name, name, hook):
    if hook is not None:
        return hook
    if module_name == "collision":
        from . import collision as owner
    elif module_name == "grasp_geometry":
        from . import grasp_geometry as owner
    elif module_name == "ik_solver":
        from . import ik_solver as owner
    elif module_name == "transforms":
        from . import transforms as owner
    else:
        raise ValueError(f"unknown grasp-candidate dependency owner: {module_name}")
    return getattr(owner, name)


def _candidate_to_jsonable(row, *, grasp_camera_summary_hook=None):
    grasp_camera_summary_fn = (
        grasp_camera_summary
        if grasp_camera_summary_hook is None
        else grasp_camera_summary_hook
    )
    out = {}
    for key, value in row.items():
        # Private keys may hold an already validated motion-plan object for
        # reuse after interactive selection.  They are process-local caches,
        # not candidate metadata, and must never be stringified into JSON.
        if key == "grasp" or str(key).startswith("_"):
            continue
        if isinstance(value, np.ndarray):
            out[key] = value.astype(float).reshape(-1).tolist()
        elif isinstance(value, (np.floating, np.integer)):
            out[key] = value.item()
        else:
            try:
                json.dumps(value)
                out[key] = value
            except TypeError:
                out[key] = str(value)
    grasp = row.get("grasp", None)
    if grasp is not None:
        out.update(grasp_camera_summary_fn(grasp))
    return out


def grasp_camera_summary(grasp):
    R_cg = np.asarray(grasp.rotation_matrix, dtype=float).reshape(3, 3)
    t_cg = np.asarray(grasp.translation, dtype=float).reshape(3)
    U, _, Vt = np.linalg.svd(R_cg)
    R_cg = U @ Vt
    if np.linalg.det(R_cg) < 0:
        U[:, -1] *= -1
        R_cg = U @ Vt
    summary = {
        "graspnet_translation_camera": t_cg.astype(float).tolist(),
        "graspnet_rotation_matrix_camera": R_cg.astype(float).tolist(),
        "graspnet_rpy_camera": R.from_matrix(R_cg).as_euler("xyz", degrees=False).astype(float).tolist(),
        "graspnet_rpy_camera_deg": R.from_matrix(R_cg).as_euler("xyz", degrees=True).astype(float).tolist(),
    }
    for attr in ("score", "width", "height", "depth"):
        if hasattr(grasp, attr):
            try:
                summary[f"graspnet_{attr}"] = float(getattr(grasp, attr))
            except Exception:
                pass
    return summary


def save_selected_grasp_pose(
    out_dir,
    grasp,
    T_base_camera,
    pre_tcp,
    grasp_tcp,
    lift_tcp,
    selected_row=None,
    *,
    candidate_to_jsonable_hook=None,
    grasp_camera_summary_hook=None,
):
    candidate_to_jsonable_fn = (
        _candidate_to_jsonable
        if candidate_to_jsonable_hook is None
        else candidate_to_jsonable_hook
    )
    grasp_camera_summary_fn = (
        grasp_camera_summary
        if grasp_camera_summary_hook is None
        else grasp_camera_summary_hook
    )
    R_cg = np.asarray(grasp.rotation_matrix, dtype=float).reshape(3, 3)
    t_cg = np.asarray(grasp.translation, dtype=float).reshape(3)
    T_camera_grasp = np.eye(4)
    T_camera_grasp[:3, :3] = R_cg
    T_camera_grasp[:3, 3] = t_cg
    T_base_grasp = np.asarray(T_base_camera, dtype=float).reshape(4, 4) @ T_camera_grasp
    payload = {
        "selected_candidate": None if selected_row is None else candidate_to_jsonable_fn(selected_row),
        "camera_grasp": grasp_camera_summary_fn(grasp),
        "T_camera_grasp": T_camera_grasp.astype(float).tolist(),
        "T_base_grasp_from_graspnet": T_base_grasp.astype(float).tolist(),
        "base_grasp_translation_from_graspnet": T_base_grasp[:3, 3].astype(float).tolist(),
        "base_grasp_rpy_from_graspnet": R.from_matrix(T_base_grasp[:3, :3]).as_euler("xyz", degrees=False).astype(float).tolist(),
        "base_grasp_rpy_from_graspnet_deg": R.from_matrix(T_base_grasp[:3, :3]).as_euler("xyz", degrees=True).astype(float).tolist(),
        "pre_tcp_command": np.asarray(pre_tcp, dtype=float).reshape(6).tolist(),
        "grasp_tcp_command": np.asarray(grasp_tcp, dtype=float).reshape(6).tolist(),
        "lift_tcp_command": np.asarray(lift_tcp, dtype=float).reshape(6).tolist(),
    }
    if selected_row is not None and selected_row.get("generator") == "graspgenx":
        payload["generator"] = "graspgenx"
        payload["T_camera_graspgenx_asset"] = selected_row.get("genx_pose_camera")
        payload["T_camera_graspgenx_tcp"] = selected_row.get("genx_tcp_pose_camera")
    path = os.path.join(out_dir, "selected_grasp_pose.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"[grasp] saved selected GraspNet pose={path}")
    return path


def save_grasp_candidates_json(rows, out_dir, *, candidate_to_jsonable_hook=None):
    candidate_to_jsonable_fn = (
        _candidate_to_jsonable
        if candidate_to_jsonable_hook is None
        else candidate_to_jsonable_hook
    )
    path = os.path.join(out_dir, "grasp_candidates.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump([candidate_to_jsonable_fn(row) for row in rows], f, indent=2)
    print(f"[grasp] saved candidates json={path}")


def grasp_width_m(row, *, grasp_camera_summary_hook=None):
    grasp = row.get("grasp", None) if isinstance(row, dict) else None
    if grasp is None:
        return None
    grasp_camera_summary_fn = (
        grasp_camera_summary
        if grasp_camera_summary_hook is None
        else grasp_camera_summary_hook
    )
    summary = grasp_camera_summary_fn(grasp)
    width = summary.get("graspnet_width", None)
    return None if width is None else float(width)

def annotate_grasp_candidate_physical_aperture(
    rows,
    target_cloud,
    T_base_camera,
    current_tcp_pose,
    args,
    *,
    verbose=True,
    grasp_to_tcp_poses_hook=None,
    make_T_from_pose_hook=None,
    axis_vector_hook=None,
    motion_open_gripper_finger_capsules_hook=None,
    point_segment_distances_hook=None,
    grasp_width_m_hook=None,
):
    """Validate candidates against the actual gripper opening.

    GraspNet inflates its predicted jaw width by 1.2 during decoding.  Keep
    that model width unchanged, but validate the target points in each
    candidate's closing slab against the real hardware opening before allowing
    an inflated-width candidate through selection. GenX supplies a fixed asset
    opening; GraspNet's inflation and prediction-saturation rules do not apply.
    """
    grasp_to_tcp_poses_fn = _resolve_hook(
        "grasp_geometry", "grasp_to_tcp_poses", grasp_to_tcp_poses_hook
    )
    make_T_from_pose_fn = _resolve_hook(
        "transforms", "_make_T_from_pose", make_T_from_pose_hook
    )
    axis_vector_fn = _resolve_hook(
        "grasp_geometry", "_axis_vector", axis_vector_hook
    )
    motion_open_gripper_finger_capsules_fn = _resolve_hook(
        "collision",
        "_motion_open_gripper_finger_capsules",
        motion_open_gripper_finger_capsules_hook,
    )
    point_segment_distances_fn = _resolve_hook(
        "transforms", "_point_segment_distances", point_segment_distances_hook
    )
    grasp_width_m_fn = grasp_width_m if grasp_width_m_hook is None else grasp_width_m_hook

    rows = [dict(row) for row in rows]
    try:
        points_camera = np.asarray(
            target_cloud.points, dtype=float
        ).reshape(-1, 3)
        T_base_camera = np.asarray(
            T_base_camera, dtype=float
        ).reshape(4, 4)
        points_base = (
            points_camera @ T_base_camera[:3, :3].T
            + T_base_camera[:3, 3].reshape(1, 3)
        )
    except Exception:
        points_base = np.zeros((0, 3), dtype=float)

    opening = min(
        float(getattr(args, "gripper_open_width", 0.100)),
        float(getattr(args, "gripper_hardware_max_width", 0.100)),
    )
    margin = max(
        float(getattr(args, "grasp_physical_aperture_margin", 0.001)), 0.0
    )
    effective_opening = max(opening - 2.0 * margin, 0.0)
    half_effective = 0.5 * effective_opening
    recenter_max = max(
        float(getattr(args, "grasp_physical_recenter_max", 0.010)), 0.0
    )
    max_finger_penetration = max(
        float(
            getattr(
                args,
                "grasp_physical_max_finger_penetration",
                0.001,
            )
        ),
        0.0,
    )
    inflation = max(
        float(getattr(args, "graspnet_width_inflation", 1.2)), 1.0
    )
    decode_max = max(
        float(getattr(args, "graspnet_decode_max_width", 0.100)), 0.0
    )
    quantile = float(
        np.clip(
            getattr(args, "grasp_physical_aperture_quantile", 0.01),
            0.0,
            0.20,
        )
    )
    min_points = max(
        int(getattr(args, "grasp_physical_aperture_min_points", 50)), 1
    )
    contact_band = max(
        float(getattr(args, "recommended_grasp_contact_band", 0.006)),
        1e-4,
    )
    contact_side_points = max(
        int(getattr(args, "recommended_grasp_contact_side_points", 25)),
        1,
    )
    finger_radius = max(
        float(getattr(args, "motion_path_open_finger_radius", 0.007)),
        0.003,
    )
    finger_back = max(
        float(getattr(args, "motion_path_open_finger_back", 0.065)),
        0.010,
    )
    finger_front = max(
        float(getattr(args, "motion_path_open_finger_front", 0.020)),
        0.0,
    )
    min_contact_inset = float(
        getattr(args, "grasp_physical_min_contact_inset", 0.0)
    )
    if not np.isfinite(min_contact_inset) or min_contact_inset < 0.0:
        raise ValueError("grasp physical minimum contact inset must be finite and nonnegative")

    kept = 0
    for row in rows:
        row["physical_aperture_ok"] = False
        row["physical_aperture_variants"] = []
        row["physical_aperture_ok_signs"] = []
        row["physical_aperture_opening"] = opening
        row["physical_aperture_effective_opening"] = effective_opening
        grasp = row.get("grasp", None)
        model_width = grasp_width_m_fn(row)
        if (
            grasp is None
            or model_width is None
            or not np.isfinite(model_width)
            or model_width <= 0.0
        ):
            row["physical_aperture_reason"] = "missing or invalid candidate width"
            continue

        fixed_genx_width = row.get("generator") == "graspgenx"
        nominal_width = float(model_width) if fixed_genx_width else float(model_width) / inflation
        row["graspnet_nominal_width"] = None if fixed_genx_width else nominal_width
        saturated = bool(
            not fixed_genx_width and decode_max > 0.0 and model_width >= decode_max - 1e-6
        )
        row["graspnet_width_saturated"] = saturated
        if saturated:
            row["physical_aperture_reason"] = (
                f"GraspNet width saturated at {model_width * 1000.0:.1f}mm; "
                "the pre-inflation width is ambiguous"
            )
            continue
        # GraspNet's decoded width is an inference hint; GenX's is its fixed
        # asset opening. Permit a nominal/fixed width up to the real
        # configured opening, then use the measured target cloud below to enforce
        # the stricter per-finger margin.
        if nominal_width > opening + 1e-9:
            width_source = "fixed GenX asset width" if fixed_genx_width else "de-inflated model width"
            row["physical_aperture_reason"] = (
                f"{width_source} {nominal_width * 1000.0:.1f}mm "
                f"> physical opening {opening * 1000.0:.1f}mm"
            )
            continue
        if points_base.shape[0] == 0:
            row["physical_aperture_reason"] = (
                "target cloud or base-camera transform unavailable for "
                "physical aperture validation"
            )
            continue

        width_signs = [1.0]
        variants = []
        valid_signs = []
        for width_sign in width_signs:
            variant = {"width_sign": float(width_sign), "ok": False}
            try:
                _, raw_grasp_tcp, _ = grasp_to_tcp_poses_fn(
                    grasp,
                    T_base_camera,
                    args,
                    current_tcp_pose,
                    width_sign=width_sign,
                )
                T_base_tcp = make_T_from_pose_fn(raw_grasp_tcp)
                tcp = np.asarray(
                    T_base_tcp[:3, 3], dtype=float
                ).reshape(3)
                tcp_rotation = np.asarray(
                    T_base_tcp[:3, :3], dtype=float
                ).reshape(3, 3)
                approach_axis = tcp_rotation @ axis_vector_fn(
                    args.tool_approach_axis
                )
                width_axis = tcp_rotation @ axis_vector_fn(
                    args.tcp_gripper_width_axis
                )
                approach_axis /= np.linalg.norm(approach_axis) + 1e-12
                width_axis /= np.linalg.norm(width_axis) + 1e-12
                height_axis = np.cross(approach_axis, width_axis)
                height_norm = float(np.linalg.norm(height_axis))
                if height_norm < 1e-6:
                    raise ValueError(
                        "tool approach and gripper width axes are collinear"
                    )
                height_axis /= height_norm

                delta = points_base - tcp.reshape(1, 3)
                along = delta @ approach_axis
                jaw_values_all = delta @ width_axis
                height_values = delta @ height_axis
                closing_mask = (
                    (along > -finger_back)
                    & (along < finger_front)
                    & (np.abs(height_values) < finger_radius)
                )
                jaw_values = np.asarray(
                    jaw_values_all[closing_mask], dtype=float
                )
                variant["points"] = int(jaw_values.size)
                if jaw_values.size < min_points:
                    variant["reason"] = (
                        f"only {jaw_values.size} target points in actual "
                        f"finger slab < required {min_points}"
                    )
                    variants.append(variant)
                    continue
                lower, upper = np.quantile(
                    jaw_values, [quantile, 1.0 - quantile]
                )
                lower = float(lower)
                upper = float(upper)
                variant["lower"] = lower
                variant["upper"] = upper
                variant["robust_span"] = upper - lower
                robust_span = upper - lower
                if robust_span > effective_opening + 1e-9:
                    variant["reason"] = (
                        f"actual-TCP target span {robust_span * 1000.0:.1f}mm "
                        f"> effective physical opening "
                        f"{effective_opening * 1000.0:.1f}mm"
                    )
                    variants.append(variant)
                    continue

                # GraspNet can place a good parallel-jaw grasp a few
                # millimetres off the target midpoint.  Recenter only along
                # the final commanded TCP width axis; every corrected pose is
                # subsequently re-run through IK and full-scene preflight.
                width_shift = 0.5 * (lower + upper)
                variant["tcp_width_shift"] = float(width_shift)
                if abs(width_shift) > recenter_max + 1e-9:
                    variant["reason"] = (
                        f"required width-axis recenter "
                        f"{width_shift * 1000.0:.1f}mm exceeds limit "
                        f"{recenter_max * 1000.0:.1f}mm"
                    )
                    variants.append(variant)
                    continue

                _, corrected_grasp_tcp, _ = grasp_to_tcp_poses_fn(
                    grasp,
                    T_base_camera,
                    args,
                    current_tcp_pose,
                    width_sign=width_sign,
                    width_shift=width_shift,
                )
                T_base_tcp = make_T_from_pose_fn(corrected_grasp_tcp)
                tcp = np.asarray(
                    T_base_tcp[:3, 3], dtype=float
                ).reshape(3)
                tcp_rotation = np.asarray(
                    T_base_tcp[:3, :3], dtype=float
                ).reshape(3, 3)
                approach_axis = tcp_rotation @ axis_vector_fn(
                    args.tool_approach_axis
                )
                width_axis = tcp_rotation @ axis_vector_fn(
                    args.tcp_gripper_width_axis
                )
                approach_axis /= np.linalg.norm(approach_axis) + 1e-12
                width_axis /= np.linalg.norm(width_axis) + 1e-12
                height_axis = np.cross(approach_axis, width_axis)
                height_axis /= np.linalg.norm(height_axis) + 1e-12
                delta = points_base - tcp.reshape(1, 3)
                along = delta @ approach_axis
                jaw_values_all = delta @ width_axis
                height_values = delta @ height_axis
                closing_mask = (
                    (along > -finger_back)
                    & (along < finger_front)
                    & (np.abs(height_values) < finger_radius)
                )
                corrected_jaw_values = np.asarray(
                    jaw_values_all[closing_mask], dtype=float
                )
                if corrected_jaw_values.size < min_points:
                    variant["reason"] = (
                        f"only {corrected_jaw_values.size} target points in "
                        f"recentered finger slab < required {min_points}"
                    )
                    variants.append(variant)
                    continue
                corrected_lower, corrected_upper = np.quantile(
                    corrected_jaw_values, [quantile, 1.0 - quantile]
                )
                corrected_lower = float(corrected_lower)
                corrected_upper = float(corrected_upper)
                variant["corrected_lower"] = corrected_lower
                variant["corrected_upper"] = corrected_upper

                # This is a soft ranking measurement. Aperture and collision
                # checks below remain the hard geometric filters.
                corrected_span = max(
                    corrected_upper - corrected_lower,
                    1e-6,
                )
                used_contact_band = min(
                    contact_band,
                    0.25 * corrected_span,
                )
                left_contact_points = int(
                    np.count_nonzero(
                        corrected_jaw_values
                        <= corrected_lower + used_contact_band
                    )
                )
                right_contact_points = int(
                    np.count_nonzero(
                        corrected_jaw_values
                        >= corrected_upper - used_contact_band
                    )
                )
                minimum_side_score = float(
                    np.clip(
                        min(left_contact_points, right_contact_points)
                        / float(contact_side_points),
                        0.0,
                        1.0,
                    )
                )
                bilateral_balance = float(
                    2.0 * min(left_contact_points, right_contact_points)
                    / max(left_contact_points + right_contact_points, 1)
                )
                variant["contact_band"] = float(used_contact_band)
                variant["left_contact_points"] = left_contact_points
                variant["right_contact_points"] = right_contact_points
                variant["bilateral_balance"] = bilateral_balance
                variant["bilateral_support_score"] = float(
                    minimum_side_score * bilateral_balance
                )
                # Width support alone accepts a surface that only grazes the
                # distal finger edge. Both predicted contact bands must also
                # sit inside the straight finger segment, away from its front
                # endpoint. Capsule endcaps are collision padding, not usable
                # contact depth. Measure the final, recentered execution pose.
                contact_along = along[closing_mask]
                contact_insets = {}
                for side, side_mask in (
                    ("left", corrected_jaw_values <= corrected_lower + used_contact_band),
                    ("right", corrected_jaw_values >= corrected_upper - used_contact_band),
                ):
                    contact_insets[side] = (
                        float(finger_front - np.median(contact_along[side_mask]))
                        if np.any(side_mask) else None
                    )
                variant["contact_inset_required"] = min_contact_inset
                variant["left_contact_inset"] = contact_insets["left"]
                variant["right_contact_inset"] = contact_insets["right"]
                if min_contact_inset > 0.0 and any(
                    inset is None or inset < min_contact_inset - 1e-9
                    for inset in contact_insets.values()
                ):
                    measured = "/".join(
                        "missing" if inset is None else f"{inset * 1000.0:.1f}"
                        for inset in contact_insets.values()
                    )
                    variant["reason"] = (
                        f"shallow bilateral contact: left/right inset={measured}mm "
                        f"from finger front; each side requires >= "
                        f"{min_contact_inset * 1000.0:.1f}mm"
                    )
                    variants.append(variant)
                    continue
                if (
                    corrected_lower < -half_effective - 1e-9
                    or corrected_upper > half_effective + 1e-9
                ):
                    variant["reason"] = (
                        f"recentered target bounds "
                        f"[{corrected_lower * 1000.0:.1f}, "
                        f"{corrected_upper * 1000.0:.1f}]mm exceed physical "
                        f"inner bounds [-{half_effective * 1000.0:.1f}, "
                        f"{half_effective * 1000.0:.1f}]mm"
                    )
                    variants.append(variant)
                    continue

                # The motion guard intentionally permits the selected target
                # to touch a finger during final approach.  Therefore the
                # aperture check must explicitly validate the complete
                # physical finger capsules, including both spherical endcaps.
                finger_clearances = []
                for capsule_a, capsule_b, capsule_radius, primitive in (
                    motion_open_gripper_finger_capsules_fn(T_base_tcp, args)
                ):
                    surface_clearance = (
                        point_segment_distances_fn(
                            points_base, capsule_a, capsule_b
                        )
                        - float(capsule_radius)
                    )
                    robust_clearance = float(
                        np.quantile(surface_clearance, quantile)
                    )
                    absolute_min_clearance = float(
                        np.min(surface_clearance)
                    )
                    finger_clearances.append(
                        {
                            "primitive": str(primitive),
                            "robust_clearance": robust_clearance,
                            "absolute_min_clearance": (
                                absolute_min_clearance
                            ),
                        }
                    )
                variant["finger_clearances"] = finger_clearances
                min_finger_clearance = min(
                    item["robust_clearance"]
                    for item in finger_clearances
                )
                absolute_min_finger_clearance = min(
                    item["absolute_min_clearance"]
                    for item in finger_clearances
                )
                variant["min_finger_clearance"] = float(
                    min_finger_clearance
                )
                variant["absolute_min_finger_clearance"] = float(
                    absolute_min_finger_clearance
                )
                if (
                    absolute_min_finger_clearance
                    < -max_finger_penetration - 1e-9
                ):
                    variant["reason"] = (
                        f"target point penetrates an open finger capsule by "
                        f"{-absolute_min_finger_clearance * 1000.0:.1f}mm; "
                        f"hard limit is "
                        f"{max_finger_penetration * 1000.0:.1f}mm"
                    )
                    variants.append(variant)
                    continue
                if min_finger_clearance < margin - 1e-9:
                    variant["reason"] = (
                        f"target gets within "
                        f"{min_finger_clearance * 1000.0:.1f}mm of an open "
                        f"finger capsule; required >= "
                        f"{margin * 1000.0:.1f}mm including endcaps"
                    )
                    variants.append(variant)
                    continue
                variant["ok"] = True
                variant["reason"] = "ok"
                valid_signs.append(float(width_sign))
                variants.append(variant)
            except Exception as exc:
                variant["reason"] = (
                    f"{type(exc).__name__}: {exc}"
                )
                variants.append(variant)

        row["physical_aperture_variants"] = variants
        row["physical_aperture_ok_signs"] = valid_signs
        if valid_signs:
            row["physical_aperture_ok"] = True
            row["physical_aperture_reason"] = "ok"
            kept += 1
        else:
            row["physical_aperture_reason"] = " | ".join(
                f"sign={int(item['width_sign']):+d}:{item.get('reason', 'unknown')}"
                for item in variants
            )

    if not verbose:
        return rows
    print(
        "[grasp][aperture] actual-opening validation kept "
        f"{kept}/{len(rows)} candidates; command={opening * 1000.0:.1f}mm "
        f"effective={effective_opening * 1000.0:.1f}mm "
        f"recenter_max={recenter_max * 1000.0:.1f}mm "
        f"max_penetration={max_finger_penetration * 1000.0:.1f}mm "
        f"min_contact_inset={min_contact_inset * 1000.0:.1f}mm "
        f"graspnet_model_inflation={inflation:.2f}"
    )
    for row in rows:
        model_width = grasp_width_m_fn(row)
        if model_width is None:
            continue
        nominal = row.get("graspnet_nominal_width", float("nan"))
        width_text = (
            f"fixed_asset_opening={model_width * 1000.0:.1f}mm "
            if row.get("generator") == "graspgenx"
            else f"model={model_width * 1000.0:.1f}mm nominal={float(nominal) * 1000.0:.1f}mm "
        )
        contact_text = ""
        for variant in row.get("physical_aperture_variants", []):
            if variant.get("ok") and variant.get("left_contact_inset") is not None:
                contact_text = (
                    f"contact_inset={variant['left_contact_inset'] * 1000.0:.1f}/"
                    f"{variant['right_contact_inset'] * 1000.0:.1f}mm "
                )
                break
        print(
            f"[grasp][aperture] candidate={int(row.get('idx', 0))} "
            f"{width_text}"
            f"ok={bool(row.get('physical_aperture_ok', False))} "
            f"{contact_text}"
            f"reason={row.get('physical_aperture_reason', 'unknown')}"
        )
    return rows


def grasp_candidate_width_shift(row, width_sign):
    """Return the aperture-validated local TCP width-axis recenter shift."""
    if row is None or "physical_aperture_ok_signs" not in row:
        return 0.0
    requested_sign = 1.0 if float(width_sign) >= 0.0 else -1.0
    for variant in row.get("physical_aperture_variants", []):
        variant_sign = float(variant.get("width_sign", 0.0))
        if (
            bool(variant.get("ok", False))
            and abs(variant_sign - requested_sign) < 0.5
        ):
            shift = float(variant.get("tcp_width_shift", 0.0))
            if np.isfinite(shift):
                return shift
            break
    raise PlanNotExecutable(
        f"jaw-axis orientation sign={int(requested_sign):+d} has no "
        "aperture-validated TCP recenter correction"
    )


def filter_grasp_candidates_by_width(rows, args, *, grasp_width_m_hook=None):
    grasp_width_m_fn = grasp_width_m if grasp_width_m_hook is None else grasp_width_m_hook
    rows = list(rows)
    if args.max_grasp_width is None:
        return rows
    max_width = float(args.max_grasp_width)
    kept = []
    physically_validated = []
    oversized = []
    invalid = []
    for row in rows:
        width = grasp_width_m_fn(row)
        aperture_annotated = "physical_aperture_ok" in row
        aperture_ok = bool(row.get("physical_aperture_ok", False))
        if aperture_annotated and not aperture_ok:
            invalid.append((row, width))
        elif width is not None and np.isfinite(width) and 0.0 < width <= max_width:
            kept.append(row)
        elif (
            width is not None
            and np.isfinite(width)
            and width > max_width
            and aperture_ok
        ):
            kept.append(row)
            physically_validated.append((row, width))
        elif width is not None and np.isfinite(width) and width > max_width:
            oversized.append((row, width))
        else:
            invalid.append((row, width))
    if len(invalid) > 0:
        print(
            f"[grasp][warn] width filter rejected {len(invalid)} candidate(s) with "
            "invalid width or failed actual-opening validation"
        )
    if len(physically_validated) > 0:
        print(
            f"[grasp] physical-aperture validation retained "
            f"{len(physically_validated)} candidate(s) above the "
            f"{max_width * 1000.0:.1f}mm inflated-model tolerance; "
            "the hardware opening remains capped"
        )
    if len(oversized) > 0:
        print(f"[grasp] width filter: kept {len(kept)}/{len(rows)} candidates with width <= {max_width * 1000.0:.1f}mm")
        if bool(getattr(args, "allow_oversize_grasp", False)):
            print(
                "[grasp][warn] --allow-oversize-grasp set; retain all candidates for "
                "diagnostic selection while physical opening remains hardware-capped"
            )
            allowed_ids = {id(row) for row in kept}
            allowed_ids.update(id(row) for row, _ in oversized)
            return [row for row in rows if id(row) in allowed_ids]
    if len(kept) == 0:
        msg = (
            f"no candidate passed inflated-model width tolerance "
            f"{max_width * 1000.0:.1f}mm and actual-opening validation"
        )
        print(
            f"[grasp][abort] {msg}; reject candidates that cannot be proven "
            "to fit the physical gripper"
        )
        return []
    return kept


def filter_grasp_candidates_by_reachability(rows, strict=False):
    rows = list(rows)
    annotated = [row for row in rows if "reachability_ok" in row]
    if len(annotated) == 0:
        return rows
    kept = [row for row in rows if bool(row.get("reachability_ok", True))]
    if len(kept) < len(rows):
        print(
            f"[grasp] reachability filter: kept {len(kept)}/{len(rows)} "
            "candidates with a valid pregrasp/grasp/lift endpoint sequence"
        )
    if len(kept) > 0:
        return kept
    if strict:
        return []
    print("[grasp][warn] no candidate passed reachability precheck; keeping hard candidates for review only")
    return rows


def filter_grasp_candidates_by_workflow_preflight(rows, strict=False):
    """Keep only candidates validated through the complete motion workflow.

    Endpoint IK is necessary but not sufficient: the final approach, lift,
    transport, release and recovery can still collide.  When workflow
    annotations are present, execution selection must use them.
    """
    rows = list(rows)
    annotated = [row for row in rows if "workflow_preflight_ok" in row]
    if len(annotated) == 0:
        if strict:
            return []
        return rows
    kept = [
        row for row in rows if bool(row.get("workflow_preflight_ok", False))
    ]
    if len(kept) < len(rows):
        print(
            f"[grasp] whole-workflow filter: kept {len(kept)}/{len(rows)} "
            "candidates whose complete collision-checked route is executable"
        )
    if len(kept) > 0 or strict:
        return kept
    print(
        "[grasp][warn] no candidate passed whole-workflow preflight; "
        "keeping rejected candidates for diagnostics only"
    )
    return rows


def filter_grasp_candidates_by_table_guard(
    rows,
    args=None,
    *,
    table_guard_enabled_hook=None,
):
    """Always remove candidates whose pre/grasp/lift TCP sequence is unsafe."""
    table_guard_enabled_fn = _resolve_hook(
        "collision", "table_guard_enabled", table_guard_enabled_hook
    )
    rows = list(rows)
    if not table_guard_enabled_fn(args):
        return rows
    kept = [row for row in rows if row.get("table_guard_ok", None) is True]
    if len(kept) < len(rows):
        print(
            f"[grasp] table-guard filter: kept {len(kept)}/{len(rows)} candidates "
            "whose annotated full TCP sequence stays above the protected height"
        )
    return kept


def grasp_candidate_too_wide(row, args, *, grasp_width_m_hook=None):
    if args is None or args.max_grasp_width is None:
        return False
    if bool(row.get("physical_aperture_ok", False)):
        return False
    grasp_width_m_fn = grasp_width_m if grasp_width_m_hook is None else grasp_width_m_hook
    width = grasp_width_m_fn(row)
    return width is not None and width > float(args.max_grasp_width)



def _ordered_equivalent_jaw_width_signs(
    grasp,
    T_base_camera,
    current_tcp_pose,
    args,
    *,
    graspnet_rotation_to_tcp_rotation_hook=None,
):
    """Return only the native GraspNet jaw orientation.

    Reachability and collision checks may reject this pose, but must not
    replace it with a generated 180-degree tool flip.
    """
    graspnet_rotation_to_tcp_rotation_fn = _resolve_hook(
        "grasp_geometry",
        "graspnet_rotation_to_tcp_rotation",
        graspnet_rotation_to_tcp_rotation_hook,
    )

    if (
        bool(args.topdown_grasp)
        or bool(args.use_current_rpy)
        or args.fixed_rpy is not None
    ):
        return [1.0], {1.0: 0.0}

    # Some diagnostic/unit callers construct a minimal namespace instead of
    # using parse_args().  Preserve the real runner defaults in that case.
    axis_args = args
    axis_defaults = {
        "graspnet_approach_axis": "x",
        "graspnet_width_axis": "y",
        "tool_approach_axis": "x",
        "tcp_gripper_width_axis": "z",
    }
    if any(not hasattr(args, name) for name in axis_defaults):
        axis_args = copy.copy(args)
        for name, default in axis_defaults.items():
            if not hasattr(axis_args, name):
                setattr(axis_args, name, default)

    R_base_camera = np.asarray(T_base_camera, dtype=float).reshape(4, 4)[:3, :3]
    R_camera_grasp = np.asarray(
        grasp.rotation_matrix, dtype=float
    ).reshape(3, 3)
    R_base_grasp = R_base_camera @ R_camera_grasp
    current = np.asarray(current_tcp_pose, dtype=float).reshape(6)
    current_rotation = R.from_euler("xyz", current[3:])
    width_sign = 1.0
    tcp_rotation = graspnet_rotation_to_tcp_rotation_fn(
        R_base_grasp,
        axis_args,
        width_sign=width_sign,
    )
    angle_by_sign = {
        width_sign: float(
            (R.from_matrix(tcp_rotation) * current_rotation.inv()).magnitude()
        )
    }
    return [width_sign], angle_by_sign


def _filter_equivalent_jaw_width_signs(
    ordered_signs,
    allowed_signs,
):
    if allowed_signs is None:
        return list(ordered_signs)
    allowed = [float(value) for value in allowed_signs]
    return [
        float(sign)
        for sign in ordered_signs
        if any(abs(float(sign) - value) < 0.5 for value in allowed)
    ]


def _candidate_preflight_width_signs(
    row,
    grasp,
    T_base_camera,
    current_tcp_pose,
    args,
    *,
    ordered_equivalent_jaw_width_signs_hook=None,
    filter_equivalent_jaw_width_signs_hook=None,
):
    """Build the hard-filtered native GraspNet jaw-orientation plan order."""
    ordered_fn = (
        _ordered_equivalent_jaw_width_signs
        if ordered_equivalent_jaw_width_signs_hook is None
        else ordered_equivalent_jaw_width_signs_hook
    )
    filter_fn = (
        _filter_equivalent_jaw_width_signs
        if filter_equivalent_jaw_width_signs_hook is None
        else filter_equivalent_jaw_width_signs_hook
    )
    ordered_signs, angle_by_sign = ordered_fn(
        grasp,
        T_base_camera,
        current_tcp_pose,
        args,
    )
    eligible_signs = list(ordered_signs)
    if row is not None and "physical_aperture_ok_signs" in row:
        eligible_signs = filter_fn(
            eligible_signs,
            row.get("physical_aperture_ok_signs", []),
        )
    if (
        row is not None
        and "reachability_ok_signs" in row
        and not bool(getattr(args, "allow_hard_grasp_candidate", False))
    ):
        eligible_signs = filter_fn(
            eligible_signs,
            row.get("reachability_ok_signs", []),
        )
    return ordered_signs, eligible_signs, angle_by_sign


def _defer_candidate_endpoint_reachability(args):
    """Let the complete planner own endpoint IK when it can search branches.

    The legacy endpoint precheck greedily kept one IK result for pregrasp,
    grasp, and lift.  With Top-K route search enabled that can incorrectly
    discard a candidate whose best-scored endpoint branch is a dead end while
    another branch is collision-free.  The complete guarded workflow checks
    the same endpoints together with every motion edge, so it is both stricter
    and better informed.  Single-solution direct mode keeps the inexpensive
    legacy precheck.
    """
    return bool(
        getattr(args, "grasp_motion_path_guard", True)
        and (
            getattr(args, "motion_up_and_over", False)
            or max(int(getattr(args, "ik_solution_topk", 1)), 1) > 1
        )
    )


def annotate_grasp_candidate_reachability(
    rows,
    T_base_camera,
    current_tcp_pose,
    args,
    arm=None,
    *,
    ordered_equivalent_jaw_width_signs_hook=None,
    filter_equivalent_jaw_width_signs_hook=None,
    grasp_candidate_width_shift_hook=None,
    grasp_to_tcp_poses_hook=None,
    assert_table_guard_target_tcp_hook=None,
    pose_rpy_error_norm_hook=None,
    solve_ik_for_flange_pose_hook=None,
    defer_candidate_endpoint_reachability_hook=None,
):
    ordered_fn = (
        _ordered_equivalent_jaw_width_signs
        if ordered_equivalent_jaw_width_signs_hook is None
        else ordered_equivalent_jaw_width_signs_hook
    )
    filter_signs_fn = (
        _filter_equivalent_jaw_width_signs
        if filter_equivalent_jaw_width_signs_hook is None
        else filter_equivalent_jaw_width_signs_hook
    )
    width_shift_fn = (
        grasp_candidate_width_shift
        if grasp_candidate_width_shift_hook is None
        else grasp_candidate_width_shift_hook
    )
    grasp_to_tcp_poses_fn = _resolve_hook(
        "grasp_geometry", "grasp_to_tcp_poses", grasp_to_tcp_poses_hook
    )
    assert_table_guard_target_tcp_fn = _resolve_hook(
        "collision",
        "assert_table_guard_target_tcp",
        assert_table_guard_target_tcp_hook,
    )
    pose_rpy_error_norm_fn = _resolve_hook(
        "transforms", "_pose_rpy_error_norm", pose_rpy_error_norm_hook
    )
    solve_ik_for_flange_pose_fn = _resolve_hook(
        "ik_solver", "_solve_ik_for_flange_pose", solve_ik_for_flange_pose_hook
    )
    defer_fn = (
        _defer_candidate_endpoint_reachability
        if defer_candidate_endpoint_reachability_hook is None
        else defer_candidate_endpoint_reachability_hook
    )

    rows = [dict(row) for row in rows]
    if T_base_camera is None or current_tcp_pose is None:
        return rows
    defer_to_whole_workflow = defer_fn(args)
    for row in rows:
        grasp = row.get("grasp", None)
        if grasp is None:
            continue
        variants = []
        variant_errors = []
        table_safe_variant_count = 0
        raw_width_signs, orientation_angles = (
            ordered_fn(
                grasp,
                T_base_camera,
                current_tcp_pose,
                args,
            )
        )
        primary_sign = float(raw_width_signs[0])
        fallback_sign = (
            None
            if len(raw_width_signs) < 2
            else float(raw_width_signs[1])
        )
        row["jaw_primary_width_sign"] = primary_sign
        row["jaw_fallback_width_sign"] = fallback_sign
        row["jaw_primary_orientation_delta_deg"] = float(
            np.rad2deg(orientation_angles[primary_sign])
        )
        row["jaw_fallback_orientation_delta_deg"] = (
            None
            if fallback_sign is None
            else float(np.rad2deg(orientation_angles[fallback_sign]))
        )
        width_signs = list(raw_width_signs)
        if "physical_aperture_ok_signs" in row:
            width_signs = filter_signs_fn(
                width_signs,
                row.get("physical_aperture_ok_signs", []),
            )
            if len(width_signs) == 0:
                row["table_guard_ok"] = False
                row["reachability_ok"] = False
                row["reachability_reason"] = (
                    "no jaw-axis orientation fits the actual physical aperture"
                )
                continue
        for width_sign in width_signs:
            try:
                width_shift = width_shift_fn(row, width_sign)
                pre_tcp, grasp_tcp, lift_tcp = grasp_to_tcp_poses_fn(
                    grasp,
                    T_base_camera,
                    args,
                    current_tcp_pose,
                    width_sign=width_sign,
                    width_shift=width_shift,
                )
                for stage, tcp_pose in zip(
                    ("pre", "grasp", "lift"),
                    (pre_tcp, grasp_tcp, lift_tcp),
                ):
                    assert_table_guard_target_tcp_fn(
                        tcp_pose,
                        args,
                        f"candidate_{int(row.get('idx', 0))}_{'pos' if width_sign > 0 else 'neg'}_{stage}",
                    )
                table_safe_variant_count += 1
                rpy_err = pose_rpy_error_norm_fn(current_tcp_pose, pre_tcp)
                pos_err = float(
                    np.linalg.norm(
                        np.asarray(pre_tcp[:3], dtype=float)
                        - np.asarray(current_tcp_pose[:3], dtype=float)
                    )
                )
                variant = {
                    "width_sign": float(width_sign),
                    "pre_tcp": np.asarray(pre_tcp, dtype=float).reshape(6),
                    "grasp_tcp": np.asarray(grasp_tcp, dtype=float).reshape(6),
                    "lift_tcp": np.asarray(lift_tcp, dtype=float).reshape(6),
                    "rpy_err": float(rpy_err),
                    "pos_err": float(pos_err),
                    "pos_ok": bool(
                        defer_to_whole_workflow
                        or pos_err
                        <= float(args.max_candidate_pre_pos_delta)
                    ),
                    "ik_ok": False,
                    "ik_reason": "not_checked",
                    "max_step": float("inf"),
                    "total_step": float("inf"),
                }
                if defer_to_whole_workflow:
                    # A direct current->pregrasp IK jump is not the trajectory
                    # that will be executed.  Keep both table-safe jaw-axis
                    # variants and let the seeded up-and-over planner solve and
                    # collision-check the actual waypoint sequence.
                    variant["ik_ok"] = True
                    variant["ik_reason"] = "deferred_to_whole_workflow_guard"
                    variant["max_step"] = float(rpy_err)
                    variant["total_step"] = float(rpy_err)
                elif arm is not None and str(getattr(args, "motion_mode", "cartesian")) == "ik_joint":
                    current_joints = np.asarray(
                        arm.get_joint_angles(), dtype=float
                    ).reshape(7)
                    flange_poses = [
                        np.asarray(arm.get_tcp2flange_pose(pose), dtype=float).reshape(6)
                        for pose in (pre_tcp, grasp_tcp, lift_tcp)
                    ]
                    q_sequence = []
                    q_previous = current_joints
                    step_deltas = []
                    for stage, flange_pose in zip(
                        ("pre", "grasp", "lift"), flange_poses
                    ):
                        q_stage = solve_ik_for_flange_pose_fn(
                            arm,
                            flange_pose,
                            args,
                            f"candidate_{int(row.get('idx', 0))}_{'pos' if width_sign > 0 else 'neg'}_{stage}",
                            current_joints_override=q_previous,
                            quiet=True,
                        )
                        delta = np.asarray(q_stage, dtype=float) - np.asarray(
                            q_previous, dtype=float
                        )
                        step_deltas.append(delta)
                        q_sequence.append(np.asarray(q_stage, dtype=float).reshape(7))
                        q_previous = q_stage
                    max_step = float(
                        max(np.max(np.abs(delta)) for delta in step_deltas)
                    )
                    total_step = float(
                        sum(np.linalg.norm(delta) for delta in step_deltas)
                    )
                    max_allowed = float(args.max_candidate_ik_joint_delta)
                    if max_step > max_allowed:
                        raise PlanNotExecutable(
                            f"sequence max joint step={np.rad2deg(max_step):.1f}deg "
                            f"> {np.rad2deg(max_allowed):.1f}deg"
                        )
                    variant.update(
                        {
                            "ik_ok": True,
                            "ik_reason": "ok",
                            "max_step": max_step,
                            "total_step": total_step,
                            "start_joints": current_joints,
                            "flange_poses": flange_poses,
                            "q_sequence": q_sequence,
                        }
                    )
                else:
                    variant["ik_ok"] = bool(
                        rpy_err
                        <= np.deg2rad(float(args.max_candidate_pre_rpy_delta_deg))
                    )
                    variant["ik_reason"] = (
                        "ok"
                        if variant["ik_ok"]
                        else f"rpy_delta={np.rad2deg(rpy_err):.1f}deg"
                    )
                    variant["max_step"] = float(rpy_err)
                    variant["total_step"] = float(rpy_err)
                variants.append(variant)
            except Exception as exc:
                variant_errors.append(
                    f"sign={int(width_sign):+d}:{type(exc).__name__}:{exc}"
                )

        valid_variants = [
            variant
            for variant in variants
            if bool(variant["ik_ok"]) and bool(variant["pos_ok"])
        ]
        selected = (
            # ``variants`` were generated in primary -> 180-degree-fallback
            # order.  Do not prefer the fallback merely because its joint
            # motion score is lower; it is selected only if the primary failed
            # an aperture/table/position/IK hard check.
            valid_variants[0]
            if valid_variants
            else (variants[0] if variants else None)
        )
        if selected is None:
            row["table_guard_ok"] = bool(table_safe_variant_count > 0)
            row["reachability_ok"] = False
            row["reachability_reason"] = (
                "both equivalent jaw-axis orientations failed: "
                + " | ".join(variant_errors)
            )
            continue

        pos_ok = bool(selected["pos_ok"])
        selected_sign = float(selected["width_sign"])
        selected_as_fallback = bool(
            fallback_sign is not None
            and abs(selected_sign - primary_sign) >= 0.5
        )
        row["tcp_width_sign"] = float(selected["width_sign"])
        row["jaw_selected_as_fallback"] = selected_as_fallback
        row["jaw_selected_role"] = (
            "fallback_180" if selected_as_fallback else "primary"
        )
        row["jaw_selected_orientation_delta_deg"] = float(
            np.rad2deg(orientation_angles[selected_sign])
        )
        row["reachability_ok_signs"] = [
            float(variant["width_sign"]) for variant in valid_variants
        ]
        row["table_guard_ok"] = True
        row["pre_tcp_preview"] = selected["pre_tcp"]
        row["grasp_tcp_preview"] = selected["grasp_tcp"]
        row["lift_tcp_preview"] = selected["lift_tcp"]
        row["pre_pos_delta"] = float(selected["pos_err"])
        row["pre_rpy_delta"] = float(selected["rpy_err"])
        row["pre_ik_ok"] = bool(selected["ik_ok"])
        row["pre_ik_reason"] = str(selected["ik_reason"])
        row["pre_ik_max_joint_delta"] = float(selected["max_step"])
        row["pre_ik_joint_delta_norm"] = float(selected["total_step"])
        if "flange_poses" in selected:
            row["pre_flange_preview"] = selected["flange_poses"][0]
            row["grasp_flange_preview"] = selected["flange_poses"][1]
            row["lift_flange_preview"] = selected["flange_poses"][2]
            row["planned_start_joints"] = selected["start_joints"]
            row["planned_pre_joints"] = selected["q_sequence"][0]
            row["planned_grasp_joints"] = selected["q_sequence"][1]
            row["planned_lift_joints"] = selected["q_sequence"][2]
        row["reachability_deferred"] = bool(defer_to_whole_workflow)
        row["reachability_ok"] = bool(pos_ok and selected["ik_ok"])
        row["reachability_reason"] = (
            "deferred_to_whole_workflow_guard"
            if defer_to_whole_workflow
            else "ok"
            if row["reachability_ok"]
            else (
                f"pre_delta pos={float(selected['pos_err']):.3f}m "
                f"rpy={np.rad2deg(float(selected['rpy_err'])):.1f}deg "
                f"ik={selected['ik_reason']} variants={' | '.join(variant_errors)}"
            )
        )
        if selected_as_fallback and row["reachability_ok"]:
            print(
                f"[grasp][orientation][fallback] candidate={int(row.get('idx', 0))} "
                f"primary width_sign={int(primary_sign):+d} did not pass endpoint "
                f"checks; enable 180deg fallback width_sign={int(selected_sign):+d}"
            )
    return rows


def execution_gripper_open_width(grasp, args):
    command_width = min(
        float(args.gripper_open_width),
        float(getattr(args, "gripper_hardware_max_width", 0.100)),
    )
    try:
        grasp_width = float(getattr(grasp, "width"))
    except Exception:
        grasp_width = None
    if grasp_width is not None and np.isfinite(grasp_width) and grasp_width > command_width:
        print(
            f"[gripper][warn] GraspNet width={grasp_width * 1000.0:.1f}mm exceeds "
            f"physical command width={command_width * 1000.0:.1f}mm; "
            "accepting it only as model-width tolerance"
        )
    return command_width

def _project_camera_point(point_camera, intr, width, height):
    point_camera = np.asarray(point_camera, dtype=float).reshape(3)
    if not np.isfinite(point_camera).all() or point_camera[2] <= 1e-6:
        return None
    u = int(round(float(intr.fx) * point_camera[0] / point_camera[2] + float(intr.cx)))
    v = int(round(float(intr.fy) * point_camera[1] / point_camera[2] + float(intr.cy)))
    if not (0 <= u < int(width) and 0 <= v < int(height)):
        return None
    return u, v


def _draw_grasp_pose_axes(
    canvas,
    center_camera,
    grasp,
    intr,
    image_width,
    image_height,
    args=None,
    *,
    axis_vector_hook=None,
):
    axis_vector_fn = _resolve_hook(
        "grasp_geometry", "_axis_vector", axis_vector_hook
    )
    R_cg = np.asarray(grasp.rotation_matrix, dtype=float).reshape(3, 3)
    t_cg = np.asarray(center_camera, dtype=float).reshape(3)
    center_uv = _project_camera_point(t_cg, intr, image_width, image_height)
    if center_uv is None:
        return

    width = None
    try:
        width = float(getattr(grasp, "width"))
    except Exception:
        pass
    width = 0.04 if width is None or not np.isfinite(width) or width <= 0.0 else min(width, 0.12)

    approach_axis = axis_vector_fn(getattr(args, "graspnet_approach_axis", "x"))
    width_axis = axis_vector_fn(getattr(args, "graspnet_width_axis", "y"))
    approach_dir = R_cg @ approach_axis
    width_dir = R_cg @ width_axis

    approach_uv = _project_camera_point(t_cg + approach_dir * 0.045, intr, image_width, image_height)
    open_a_uv = _project_camera_point(t_cg - width_dir * width * 0.5, intr, image_width, image_height)
    open_b_uv = _project_camera_point(t_cg + width_dir * width * 0.5, intr, image_width, image_height)

    if open_a_uv is not None and open_b_uv is not None:
        cv2.line(canvas, open_a_uv, open_b_uv, (255, 210, 0), 3, cv2.LINE_AA)
        cv2.circle(canvas, open_a_uv, 4, (255, 210, 0), -1, cv2.LINE_AA)
        cv2.circle(canvas, open_b_uv, 4, (255, 210, 0), -1, cv2.LINE_AA)
    if approach_uv is not None:
        cv2.arrowedLine(canvas, center_uv, approach_uv, (0, 0, 255), 3, cv2.LINE_AA, tipLength=0.28)


def draw_grasp_candidate_overlay(
    bgr,
    rows,
    intr,
    out_dir,
    args=None,
    *,
    grasp_candidate_too_wide_hook=None,
    draw_grasp_pose_axes_hook=None,
    grasp_camera_summary_hook=None,
):
    grasp_candidate_too_wide_fn = (
        grasp_candidate_too_wide
        if grasp_candidate_too_wide_hook is None
        else grasp_candidate_too_wide_hook
    )
    draw_grasp_pose_axes_fn = (
        _draw_grasp_pose_axes
        if draw_grasp_pose_axes_hook is None
        else draw_grasp_pose_axes_hook
    )
    grasp_camera_summary_fn = (
        grasp_camera_summary
        if grasp_camera_summary_hook is None
        else grasp_camera_summary_hook
    )
    img = np.asarray(bgr, dtype=np.uint8).copy()
    hit_regions = []
    colors = [
        (0, 255, 0),
        (0, 180, 255),
        (255, 120, 0),
        (255, 0, 180),
        (180, 255, 0),
        (0, 120, 255),
        (255, 0, 0),
        (120, 0, 255),
    ]
    h, w = img.shape[:2]
    panel_w = 520
    canvas = np.full((h, w + panel_w, 3), 245, dtype=np.uint8)
    canvas[:, :w] = img
    cv2.putText(canvas, "GraspNet candidates", (w + 18, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (30, 30, 30), 2, cv2.LINE_AA)
    app_axis = getattr(args, "graspnet_approach_axis", "x") if args is not None else "x"
    width_axis = getattr(args, "graspnet_width_axis", "y") if args is not None else "y"
    tcp_width_axis = getattr(args, "tcp_gripper_width_axis", "z") if args is not None else "z"
    cv2.putText(canvas, f"red=GraspNet {app_axis} approach; cyan=GraspNet {width_axis} width", (w + 18, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (70, 70, 70), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"exec maps cyan width to TCP {tcp_width_axis} axis", (w + 18, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (70, 70, 70), 1, cv2.LINE_AA)
    recommended_rows = [
        row for row in rows if bool(row.get("recommended_grasp", False))
    ]
    recommended_idx = (
        int(recommended_rows[0].get("idx", 0)) if recommended_rows else None
    )
    recommended_quality_ok = bool(
        recommended_rows
        and recommended_rows[0].get("recommended_quality_ok", True)
    )
    if recommended_idx is not None:
        recommendation_title = (
            "RECOMMENDED" if recommended_quality_ok else "FALLBACK (LOW CONFIDENCE)"
        )
        cv2.putText(
            canvas,
            f"{recommendation_title}: candidate {recommended_idx} (Enter)",
            (w + 18, 106),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 125, 215),
            2,
            cv2.LINE_AA,
        )
    cv2.putText(canvas, "idx width(mm) priority score center_dist", (w + 18, 132), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (70, 70, 70), 1, cv2.LINE_AA)
    for display_i, row in enumerate(rows):
        t = np.asarray(row.get("translation", [np.nan, np.nan, np.nan]), dtype=float).reshape(3)
        if not np.isfinite(t).all() or t[2] <= 0:
            continue
        u = int(round(float(intr.fx) * t[0] / t[2] + float(intr.cx)))
        v = int(round(float(intr.fy) * t[1] / t[2] + float(intr.cy)))
        if not (0 <= u < w and 0 <= v < h):
            continue
        idx = int(row.get("idx", display_i))
        color = colors[display_i % len(colors)]
        too_wide = grasp_candidate_too_wide_fn(row, args)
        reachability_ok = bool(row.get("reachability_ok", True))
        workflow_ok = bool(row.get("workflow_preflight_ok", True))
        is_recommended = bool(row.get("recommended_grasp", False))
        draw_color = (
            (0, 0, 180)
            if (too_wide or not reachability_ok or not workflow_ok)
            else color
        )
        if row.get("grasp", None) is not None:
            draw_grasp_pose_axes_fn(canvas, t, row["grasp"], intr, w, h, args)
        cv2.circle(canvas, (u, v), 8, draw_color, 2)
        cv2.drawMarker(canvas, (u, v), draw_color, cv2.MARKER_CROSS, 22, 2)
        if is_recommended:
            cv2.circle(canvas, (u, v), 17, (0, 215, 255), 4, cv2.LINE_AA)
        cam = grasp_camera_summary_fn(row["grasp"]) if row.get("grasp", None) is not None else {}
        width_m = cam.get("graspnet_width", None)
        width_mm = float(width_m) * 1000.0 if width_m is not None else float("nan")
        label = f"{idx}" if width_m is None else f"{idx} {width_mm:.0f}mm"
        if too_wide:
            label = f"{label} X"
        if not reachability_ok:
            label = f"{label} IK"
        if not workflow_ok:
            label = f"{label} PATH"
        if is_recommended:
            label = (
                f"BEST {label}"
                if bool(row.get("recommended_quality_ok", True))
                else f"FALLBACK {label}"
            )
        cv2.putText(canvas, label, (u + 10, max(20, v - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, draw_color, 2, cv2.LINE_AA)
        hit_regions.append({"idx": idx, "uv": (u, v), "bbox": (u - 24, v - 24, u + 42, v + 24)})
        y = 162 + 34 * display_i
        if y + 20 < h:
            priority = float(row.get("priority", 0.0))
            score = float(row.get("raw_score", 0.0))
            center_dist = float(row.get("center_dist", 0.0))
            row_text = f"{idx:>2}  {width_mm:>8.1f}  {priority:>7.3f}  {score:>5.3f}  {center_dist:>9.4f}"
            if too_wide:
                row_text = f"{row_text}  too wide"
            if not reachability_ok:
                row_text = f"{row_text}  hard pose"
            if is_recommended:
                row_text = (
                    f"{row_text}  RECOMMENDED"
                    if bool(row.get("recommended_quality_ok", True))
                    else f"{row_text}  LOW_CONFIDENCE"
                )
            cv2.putText(canvas, row_text, (w + 18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, draw_color, 2, cv2.LINE_AA)
            rpy_deg = np.asarray(cam.get("graspnet_rpy_camera_deg", [0.0, 0.0, 0.0]), dtype=float)
            pre_pos = row.get("pre_pos_delta", None)
            ik_delta = row.get("pre_ik_max_joint_delta", None)
            if ik_delta is not None and np.isfinite(float(ik_delta)):
                reach_text = f" pre={float(pre_pos):.2f}m ik_max={np.rad2deg(float(ik_delta)):.0f}deg"
            else:
                pre_rpy = row.get("pre_rpy_delta", None)
                reach_text = "" if pre_rpy is None else f" pre_delta={float(pre_pos):.2f}m/{np.rad2deg(float(pre_rpy)):.0f}deg"
            cross_axis = row.get("target_cross_axis_score", None)
            cross_text = (
                ""
                if cross_axis is None
                else f" cross={float(cross_axis):.2f}"
            )
            rpy_text = f"    rpy_cam_deg={np.round(rpy_deg, 1).tolist()}{reach_text}{cross_text}"
            cv2.putText(canvas, rpy_text, (w + 18, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (75, 75, 75), 1, cv2.LINE_AA)
    path = os.path.join(out_dir, "grasp_candidates.png")
    cv2.imwrite(path, canvas)
    graspnet_path = os.path.join(out_dir, "graspnet_candidates.png")
    if graspnet_path != path:
        cv2.imwrite(graspnet_path, canvas)
    print(f"[grasp] saved GraspNet candidate image={path}")
    return canvas, hit_regions, path

def select_grasp_candidate(
    rows,
    bgr,
    intr,
    out_dir,
    args,
    *,
    save_grasp_candidates_json_hook=None,
    filter_grasp_candidates_by_width_hook=None,
    filter_grasp_candidates_by_table_guard_hook=None,
    filter_grasp_candidates_by_reachability_hook=None,
    filter_grasp_candidates_by_workflow_preflight_hook=None,
    defer_candidate_endpoint_reachability_hook=None,
    draw_grasp_candidate_overlay_hook=None,
    grasp_camera_summary_hook=None,
    grasp_candidate_too_wide_hook=None,
    grasp_width_m_hook=None,
):
    save_json_fn = (
        save_grasp_candidates_json
        if save_grasp_candidates_json_hook is None
        else save_grasp_candidates_json_hook
    )
    filter_width_fn = (
        filter_grasp_candidates_by_width
        if filter_grasp_candidates_by_width_hook is None
        else filter_grasp_candidates_by_width_hook
    )
    filter_table_fn = (
        filter_grasp_candidates_by_table_guard
        if filter_grasp_candidates_by_table_guard_hook is None
        else filter_grasp_candidates_by_table_guard_hook
    )
    filter_reachability_fn = (
        filter_grasp_candidates_by_reachability
        if filter_grasp_candidates_by_reachability_hook is None
        else filter_grasp_candidates_by_reachability_hook
    )
    filter_workflow_fn = (
        filter_grasp_candidates_by_workflow_preflight
        if filter_grasp_candidates_by_workflow_preflight_hook is None
        else filter_grasp_candidates_by_workflow_preflight_hook
    )
    defer_fn = (
        _defer_candidate_endpoint_reachability
        if defer_candidate_endpoint_reachability_hook is None
        else defer_candidate_endpoint_reachability_hook
    )
    draw_overlay_fn = (
        draw_grasp_candidate_overlay
        if draw_grasp_candidate_overlay_hook is None
        else draw_grasp_candidate_overlay_hook
    )
    grasp_camera_summary_fn = (
        grasp_camera_summary
        if grasp_camera_summary_hook is None
        else grasp_camera_summary_hook
    )
    too_wide_fn = (
        grasp_candidate_too_wide
        if grasp_candidate_too_wide_hook is None
        else grasp_candidate_too_wide_hook
    )
    width_m_fn = grasp_width_m if grasp_width_m_hook is None else grasp_width_m_hook

    if len(rows) == 0:
        raise PlanNotExecutable("No grasp candidates returned by the selected generator")
    rows_all = list(rows)[: max(int(args.manual_topk), 1)]
    save_json_fn(rows_all, out_dir)
    rows_width_preferred = filter_width_fn(rows_all, args)
    if len(rows_width_preferred) == 0:
        overlay, hits, overlay_path = draw_overlay_fn(bgr, rows_all, intr, out_dir, args)
        max_width = None if args.max_grasp_width is None else float(args.max_grasp_width) * 1000.0
        raise PlanNotExecutable(
            "no GraspNet candidate passed both the inflated model-width gate "
            f"(--max-grasp-width={max_width:.1f}mm) and the actual gripper-aperture "
            f"validation. Review {overlay_path}; use a physically narrower "
            "candidate or obtain additional views."
        )
    rows_table_safe = filter_table_fn(rows_width_preferred, args)
    if len(rows_table_safe) == 0:
        save_json_fn(rows_all, out_dir)
        raise PlanNotExecutable(
            "all width-valid GraspNet candidates violate the configured table guard "
            "during pregrasp/grasp/lift; no motion is allowed. Verify TABLE_Z and "
            "request a different grasp/view without disabling protection."
        )
    whole_workflow_guard = bool(getattr(args, "grasp_motion_path_guard", True))
    defer_endpoint_ik = defer_fn(args)
    rows_reachable_strict = (
        rows_table_safe
        if defer_endpoint_ik
        else filter_reachability_fn(rows_table_safe, strict=True)
    )
    if defer_endpoint_ik:
        print(
            "[grasp] direct endpoint reachability is diagnostic only; the "
            "selected candidate will be decided by full up-and-over preflight"
        )
    elif whole_workflow_guard:
        print(
            "[grasp] direct-route endpoint IK filter is active; each selectable "
            "candidate will still receive full-scene workflow preflight"
        )
    if (
        len(rows_reachable_strict) == 0
        and not bool(args.allow_hard_grasp_candidate)
        and not defer_endpoint_ik
    ):
        save_json_fn(rows_all, out_dir)
        overlay, hits, overlay_path = draw_overlay_fn(bgr, rows_all, intr, out_dir, args)
        print(f"[grasp] saved all hard GraspNet candidates for review: {overlay_path}")
        sample = rows_all[0].get("reachability_reason", "unknown") if len(rows_all) > 0 else "unknown"
        raise PlanNotExecutable(
            "all GraspNet candidates are marked unreachable by the precheck. "
            f"sample_reason={sample}. Both equivalent jaw-axis orientations and the full "
            "pregrasp/grasp/lift IK sequence were checked; obtain another safe view or replan."
        )

    rows_endpoint_selectable = (
        rows_table_safe
        if defer_endpoint_ik or bool(args.allow_hard_grasp_candidate)
        else rows_reachable_strict
    )
    rows_selectable = rows_endpoint_selectable
    if whole_workflow_guard:
        rows_selectable = filter_workflow_fn(
            rows_endpoint_selectable,
            strict=True,
        )
    workflow_annotated = any(
        "workflow_preflight_ok" in row for row in rows_endpoint_selectable
    )
    if whole_workflow_guard and len(rows_selectable) == 0:
        save_json_fn(rows_all, out_dir)
        overlay, hits, overlay_path = draw_overlay_fn(
            bgr, rows_endpoint_selectable, intr, out_dir, args
        )
        reasons = [
            f"{int(row.get('idx', index))}:"
            f"{row.get('workflow_preflight_reason', 'unknown')}"
            for index, row in enumerate(rows_endpoint_selectable)
        ]
        if workflow_annotated:
            detail = (
                "no endpoint-reachable GraspNet candidate passed complete "
                "collision-checked workflow preflight; "
                + " | ".join(reasons)
            )
        else:
            detail = (
                "complete workflow preflight annotations are missing; "
                "refusing to advertise or execute an unvalidated candidate"
            )
        raise PlanNotExecutable(detail + f". Review {overlay_path}.")
    default_row = recommend_grasp_candidate(rows_selectable, args)
    default_idx = int(default_row.get("idx", 0))
    # Persist recommendation diagnostics together with the other candidate
    # annotations so the later choice can be audited from the session log.
    save_json_fn(rows_all, out_dir)
    overlay, hits, overlay_path = draw_overlay_fn(bgr, rows_selectable, intr, out_dir, args)
    if len(rows_selectable) < len(rows_all):
        print(
            "[grasp] non-executable candidates hidden from selection: "
            f"kept {len(rows_selectable)}/{len(rows_all)}"
        )
    print(
        "[grasp] review complete-workflow executable GraspNet candidates "
        f"before selecting: {overlay_path}"
    )
    print(
        f"[grasp][recommended] candidate={default_idx} "
        f"quality={'recommended' if bool(default_row.get('recommended_quality_ok', True)) else 'low_confidence_fallback'} "
        f"reason={default_row.get('recommended_reason', 'complete-workflow priority')}"
    )
    print(
        "[grasp] select by clicking a fully prevalidated candidate on the image, "
        "pressing 0-9, "
        f"or pressing Enter for recommended candidate {default_idx}."
    )
    if args.print_grasp_candidates:
        print("[grasp] candidate list:")
        for row in rows_all:
            cam = grasp_camera_summary_fn(row["grasp"]) if row.get("grasp", None) is not None else {}
            rpy_deg = np.asarray(cam.get("graspnet_rpy_camera_deg", [0.0, 0.0, 0.0]), dtype=float)
            width_m = cam.get("graspnet_width", float("nan"))
            width_mm = float(width_m) * 1000.0 if np.isfinite(float(width_m)) else float("nan")
            marker = (
                " [RECOMMENDED]"
                if bool(row.get("recommended_grasp", False))
                else ""
            )
            print(
                f"  [{int(row.get('idx', 0))}]{marker} priority={float(row.get('priority', 0.0)):.4f} "
                f"vertical={float(row.get('vertical_align', 0.0)):.4f} "
                f"center_dist={float(row.get('center_dist', 0.0)):.4f} "
                f"obs={float(row.get('obstacle_score', 0.0)):.4f} "
                f"score={float(row.get('raw_score', 0.0)):.4f} "
                f"width={float(width_m):.4f}m/{width_mm:.1f}mm "
                f"t_cam={np.round(np.asarray(row.get('translation', [0, 0, 0]), dtype=float), 4).tolist()} "
                f"rpy_cam_deg={np.round(rpy_deg, 1).tolist()}"
            )

    if args.grasp_candidate_index is not None:
        chosen_idx = int(args.grasp_candidate_index)
    elif args.select_grasp_candidate:
        clicked = {"idx": None}
        window_name = "Review GraspNet candidates, then click/digit/Enter"
        chosen_idx = None
        terminal_fallback = False

        def _prompt_candidate_in_terminal():
            valid_idxs = sorted(int(row.get("idx", i)) for i, row in enumerate(rows_selectable))
            while True:
                raw = input(f"[grasp] select candidate index {valid_idxs} (Enter={default_idx}): ").strip()
                if raw == "":
                    return default_idx
                try:
                    idx = int(raw)
                except ValueError:
                    print(f"[grasp][warn] invalid candidate index {raw!r}; try again.")
                    continue
                if idx in valid_idxs:
                    return idx
                print(f"[grasp][warn] candidate index {idx} is not in {valid_idxs}; try again.")

        def cb(event, x, y, flags, param):
            if event != cv2.EVENT_LBUTTONDOWN:
                return
            for hit in hits:
                x0, y0, x1, y1 = hit["bbox"]
                if x0 <= x <= x1 and y0 <= y <= y1:
                    clicked["idx"] = int(hit["idx"])
                    return

        try:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.imshow(window_name, overlay)
            cv2.setMouseCallback(window_name, cb)
            while True:
                try:
                    visible = cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE)
                    if visible < 1:
                        terminal_fallback = True
                        print("[grasp] candidate window closed; falling back to terminal input.")
                        break
                except cv2.error:
                    terminal_fallback = True
                    print("[grasp] candidate window unavailable/closed; falling back to terminal input.")
                    break
                key = cv2.waitKey(30) & 0xFF
                if clicked["idx"] is not None:
                    chosen_idx = int(clicked["idx"])
                    break
                if key in (13, 10):
                    chosen_idx = default_idx
                    break
                if key in (27, ord("q")):
                    raise GraspSelectionDeclined("grasp candidate selection cancelled")
                if ord("0") <= key <= ord("9"):
                    chosen_idx = int(key - ord("0"))
                    break
            try:
                cv2.destroyWindow(window_name)
            except cv2.error:
                pass
        except cv2.error as exc:
            print(f"[grasp][warn] OpenCV click window unavailable: {type(exc).__name__}: {exc}")
            terminal_fallback = True
        if terminal_fallback or chosen_idx is None:
            chosen_idx = _prompt_candidate_in_terminal()
    else:
        chosen_idx = default_idx

    by_idx = {int(row.get("idx", i)): row for i, row in enumerate(rows_selectable)}
    if chosen_idx not in by_idx:
        raise InvalidGraspSelection(
            f"selected grasp index {chosen_idx} is not in complete-workflow "
            f"executable candidates {sorted(by_idx)}"
        )
    chosen = by_idx[chosen_idx]
    if (
        not defer_endpoint_ik
        and (not bool(chosen.get("reachability_ok", True)))
        and not bool(args.allow_hard_grasp_candidate)
    ):
        fallback_rows = filter_reachability_fn(
            filter_table_fn(
                filter_width_fn(rows_all, args),
                args,
            ),
            strict=True,
        )
        if whole_workflow_guard:
            fallback_rows = filter_workflow_fn(
                fallback_rows,
                strict=True,
            )
        if len(fallback_rows) > 0:
            fallback = fallback_rows[0]
            fallback_idx = int(fallback.get("idx", 0))
            print(
                f"[grasp][warn] selected candidate {chosen_idx} is marked hard pose "
                f"({chosen.get('reachability_reason', 'unknown')}); using recommended reachable candidate {fallback_idx}. "
                "Add --allow-hard-grasp-candidate to override."
            )
            chosen_idx = fallback_idx
            chosen = fallback
        else:
            raise PlanNotExecutable(
                f"selected grasp candidate {chosen_idx} is hard to reach "
                f"({chosen.get('reachability_reason', 'unknown')}) and no reachable candidate is available; "
                "execution remains fail-closed."
            )
    if too_wide_fn(chosen, args):
        width = width_m_fn(chosen)
        print(
            f"[grasp][warn] selected candidate {chosen_idx} width={float(width) * 1000.0:.1f}mm "
            f"exceeds --max-grasp-width={float(args.max_grasp_width) * 1000.0:.1f}mm; "
            "execution gripper width will be capped"
        )
    if not bool(chosen.get("reachability_ok", True)):
        print(f"[grasp][warn] selected candidate {chosen_idx} may be hard to reach: {chosen.get('reachability_reason', 'unknown')}")
    selected_path = os.path.join(out_dir, "selected_grasp_candidate.json")
    with open(selected_path, "w", encoding="utf-8") as f:
        json.dump(_candidate_to_jsonable(chosen), f, indent=2)
    print(f"[grasp] selected candidate index={chosen_idx} saved={selected_path}")
    return chosen["grasp"], chosen


def choose_default_grasp_candidate(
    rows,
    gg,
    args,
    *,
    filter_grasp_candidates_by_width_hook=None,
    filter_grasp_candidates_by_table_guard_hook=None,
    defer_candidate_endpoint_reachability_hook=None,
    filter_grasp_candidates_by_reachability_hook=None,
    filter_grasp_candidates_by_workflow_preflight_hook=None,
):
    filter_width_fn = (
        filter_grasp_candidates_by_width
        if filter_grasp_candidates_by_width_hook is None
        else filter_grasp_candidates_by_width_hook
    )
    filter_table_fn = (
        filter_grasp_candidates_by_table_guard
        if filter_grasp_candidates_by_table_guard_hook is None
        else filter_grasp_candidates_by_table_guard_hook
    )
    defer_fn = (
        _defer_candidate_endpoint_reachability
        if defer_candidate_endpoint_reachability_hook is None
        else defer_candidate_endpoint_reachability_hook
    )
    filter_reachability_fn = (
        filter_grasp_candidates_by_reachability
        if filter_grasp_candidates_by_reachability_hook is None
        else filter_grasp_candidates_by_reachability_hook
    )
    filter_workflow_fn = (
        filter_grasp_candidates_by_workflow_preflight
        if filter_grasp_candidates_by_workflow_preflight_hook is None
        else filter_grasp_candidates_by_workflow_preflight_hook
    )

    rows = list(rows)
    if len(rows) == 0:
        raise PlanNotExecutable("No grasp candidates returned by the selected generator")
    filtered = filter_width_fn(rows, args)
    filtered = filter_table_fn(filtered, args)
    if (
        not defer_fn(args)
        and not bool(getattr(args, "allow_hard_grasp_candidate", False))
    ):
        filtered = filter_reachability_fn(
            filtered, strict=True
        )
    if bool(getattr(args, "grasp_motion_path_guard", True)):
        filtered = filter_workflow_fn(
            filtered, strict=True
        )
    if len(filtered) == 0:
        raise PlanNotExecutable(
            "no grasp candidate remains after "
            "width/table/endpoint-IK/whole-workflow filtering"
        )
    chosen = recommend_grasp_candidate(filtered, args)
    return chosen["grasp"], chosen
