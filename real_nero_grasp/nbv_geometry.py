"""Numerical-only NBV geometry shared by the facade and isolated worker."""

import numpy as np
from scipy.spatial import cKDTree
from .active_view_scoring import (
    _active_spherical_azimuths, _pca_main_axes_xy, _normalize_dir_xy, _dir_from_angle_rad,
    _direction_context_penalty,
)
from .types import UnsafeMotionTarget


def _prepare_active_obstacle_cloud(scene_pts, target_pts, target_center, args, source_is_non_target_mask=False):
    """Build obstacle cloud as scene minus segmented target cloud and table."""
    scene = np.asarray(scene_pts, dtype=np.float32).reshape(-1, 3)
    target = np.asarray(target_pts, dtype=np.float32).reshape(-1, 3)
    center = np.asarray(target_center, dtype=np.float32).reshape(3)
    if scene.shape[0] == 0:
        return scene, np.zeros((0,), dtype=np.float32), {
            "raw": 0,
            "kept": 0,
            "target_removed": 0,
            "table_removed": 0,
            "target_z_lo": float(center[2] - 0.03),
            "target_z_hi": float(center[2] + 0.03),
            "height_keep_z": float(center[2] - 0.035),
        }

    if target.shape[0] > 0:
        target_z_lo = float(np.percentile(target[:, 2], 10))
        target_z_hi = float(np.percentile(target[:, 2], 90))
    else:
        target_z_lo = float(center[2] - 0.03)
        target_z_hi = float(center[2] + 0.03)

    xy_dist = np.linalg.norm(scene[:, :2] - center.reshape(1, 3)[:, :2], axis=1)
    z_keep = target_z_lo + float(args.active_obstacle_min_height_margin)
    if bool(source_is_non_target_mask):
        target_keep = np.ones((scene.shape[0],), dtype=bool)
        exclusion_mode = "mask_non_target"
    elif target.shape[0] > 0:
        target_tree = cKDTree(target.astype(np.float32))
        dist_to_target, _ = target_tree.query(scene.astype(np.float32), k=1, workers=-1)
        target_keep = np.asarray(dist_to_target, dtype=np.float32) > float(args.active_target_cloud_exclusion_dist)
        exclusion_mode = "target_cloud"
    else:
        dist_to_center = np.linalg.norm(scene - center.reshape(1, 3), axis=1)
        target_keep = dist_to_center > float(args.active_target_exclusion_radius)
        exclusion_mode = "center_fallback"
    height_keep = scene[:, 2] >= z_keep
    xy_keep = xy_dist <= float(args.active_obstacle_xy_radius)
    keep = target_keep & xy_keep & height_keep
    pts = scene[keep]
    if pts.shape[0] == 0:
        pts = scene[target_keep & height_keep]
    if pts.shape[0] == 0:
        pts = scene[target_keep]
    if pts.shape[0] == 0:
        pts = scene[height_keep]
    if pts.shape[0] == 0:
        pts = scene[:0]

    denom = max((target_z_hi - target_z_lo) + 0.02, 0.02)
    height_norm = np.clip((pts[:, 2] - z_keep) / denom, 0.0, 2.0)
    weights = (1.0 + 0.75 * height_norm).astype(np.float32)

    if pts.shape[0] > int(args.active_obstacle_cap):
        idx = np.random.choice(pts.shape[0], int(args.active_obstacle_cap), replace=False)
        pts = pts[idx]
        weights = weights[idx]

    return pts.astype(np.float32), weights.astype(np.float32), {
        "raw": int(scene.shape[0]),
        "kept": int(pts.shape[0]),
        "target_removed": int(np.count_nonzero(~target_keep)),
        "table_removed": int(np.count_nonzero(target_keep & ~height_keep)),
        "exclusion_mode": exclusion_mode,
        "target_z_lo": float(target_z_lo),
        "target_z_hi": float(target_z_hi),
        "height_keep_z": float(z_keep),
    }


def _build_scene_aware_nbv_records(target_pts, target_normals, obstacle_pts, target_center, args, preferred_dir_xy=None, obstacle_weights=None):
    """Build the azimuth part of the spherical grid.

    The final score is deliberately *not* computed here: it must use the real
    3-D view direction for each (azimuth, tilt), otherwise all tilt levels at
    one azimuth incorrectly inherit the same horizontal score.
    """
    fallback_num = int(max(getattr(args, "active_nbv_candidate_num", 16), 4))
    step_deg = float(
        getattr(args, "active_nbv_azimuth_step_deg", 360.0 / fallback_num)
    )
    candidate_angles = _active_spherical_azimuths(step_deg)
    major_xy, minor_xy = _pca_main_axes_xy(target_pts)
    preferred = [] if preferred_dir_xy is None else [_normalize_dir_xy(preferred_dir_xy)]
    for d in (major_xy, -major_xy, minor_xy, -minor_xy):
        preferred.append(_normalize_dir_xy(d))

    records = []
    for ang in candidate_angles:
        dir_xy = _dir_from_angle_rad(ang)
        occ_pen, clearance = _direction_context_penalty(target_center, obstacle_pts, dir_xy, weights=obstacle_weights)
        pref_bonus = 0.0
        if preferred:
            pref_bonus = float(max(0.0, max(float(np.dot(_normalize_dir_xy(p), dir_xy)) for p in preferred)))
        # This value is only a cheap ordering hint before the full 3-D
        # information score and IK checks.  It is never added to the final
        # candidate score.
        score = 0.80 * float(clearance) + 0.20 * float(pref_bonus)
        records.append({
            "name": f"nbv_{int(round(np.rad2deg(ang))) % 360:03d}",
            "angle_rad": float(ang),
            "dir_xy": dir_xy.astype(np.float32),
            "score": float(score),
            "detail": {
                "clearance": float(clearance),
                "occ_penalty": float(occ_pen),
                "preferred_dir_bonus": float(pref_bonus),
                "azimuth_deg": float(np.rad2deg(ang) % 360.0),
                "score_stage": "azimuth_prefilter_only",
            },
        })

    records.sort(key=lambda item: float(item["score"]), reverse=True)
    if bool(getattr(args, "active_nbv_debug", False)):
        print("[ActiveNBV][grid] azimuth directions before 3-D tilt scoring:")
        for rec in records[: min(6, len(records))]:
            d = rec["detail"]
            print(
                f"  {rec['name']} angle={np.rad2deg(rec['angle_rad']):.1f} "
                f"prefilter={rec['score']:.4f} clr={d['clearance']:.4f} "
                f"occ={d['occ_penalty']:.4f} "
                f"pref={d['preferred_dir_bonus']:.4f}"
            )
    return records


def _line_segment_obstacle_metrics(p0, p1, obstacle_pts, corridor_radius, weights=None):
    pts = np.asarray(obstacle_pts, dtype=np.float32).reshape(-1, 3)
    if pts.shape[0] == 0:
        return {"count": 0, "ratio": 0.0, "nearest": float("inf"), "weighted_count": 0.0, "penalty": 0.0}
    w = np.ones((pts.shape[0],), dtype=np.float32) if weights is None else np.asarray(weights, dtype=np.float32).reshape(-1)
    if w.shape[0] != pts.shape[0]:
        w = np.ones((pts.shape[0],), dtype=np.float32)
    a = np.asarray(p0, dtype=np.float32).reshape(3)
    b = np.asarray(p1, dtype=np.float32).reshape(3)
    ab = b - a
    length = float(np.linalg.norm(ab))
    if length < 1e-6:
        return {"count": 0, "ratio": 0.0, "nearest": float("inf"), "weighted_count": 0.0, "penalty": 0.0}
    direction = ab / length
    rel = pts - a.reshape(1, 3)
    axial = rel @ direction
    keep_axial = (axial > 0.015) & (axial < max(length - 0.01, 0.015))
    perp = np.linalg.norm(rel - axial[:, None] * direction.reshape(1, 3), axis=1)
    keep = keep_axial & (perp < float(corridor_radius))
    if not np.any(keep):
        return {"count": 0, "ratio": 0.0, "nearest": float("inf"), "penalty": 0.0}
    nearest = float(np.min(axial[keep]))
    ratio = float(np.mean(keep))
    count = int(np.count_nonzero(keep))
    weighted_count = float(np.sum(np.clip(w[keep], 0.05, 5.0)))
    near_risk = float(np.clip((0.20 - nearest) / 0.20, 0.0, 1.0))
    density_risk = float(np.clip(weighted_count / 350.0, 0.0, 1.0))
    ratio_risk = float(np.clip(ratio / 0.04, 0.0, 1.0))
    penalty = float(np.clip(0.45 * near_risk + 0.35 * density_risk + 0.20 * ratio_risk, 0.0, 1.0))
    return {"count": count, "ratio": ratio, "nearest": nearest, "weighted_count": weighted_count, "penalty": penalty}


def _nearest_obstacle_distance(point_base, obstacle_pts):
    pts = np.asarray(obstacle_pts, dtype=np.float32).reshape(-1, 3)
    if pts.shape[0] == 0:
        return float("inf")
    p = np.asarray(point_base, dtype=np.float32).reshape(1, 3)
    return float(np.min(np.linalg.norm(pts - p, axis=1)))


def _active_motion_scene_points(target_pts, scene_pts, args):
    clouds = []
    # Use the complete captured scene for motion safety.  The target/table
    # exclusions used for NBV scoring are intentionally not used here: the arm
    # must not sweep through the selected object, basket, mat, or table either.
    for cloud in (scene_pts, target_pts):
        pts = np.asarray(cloud, dtype=np.float32).reshape(-1, 3)
        pts = pts[np.isfinite(pts).all(axis=1)]
        if pts.shape[0] > 0:
            clouds.append(pts)
    if not clouds:
        if bool(getattr(args, "active_path_require_scene", True)):
            raise UnsafeMotionTarget("active path guard has no valid scene points")
        return np.zeros((0, 3), dtype=np.float32)
    points = np.concatenate(clouds, axis=0).astype(np.float32)
    # Prepare the complete safety cloud once.  Candidate scoring evaluates many
    # joint paths; repeatedly voxelising the raw RGB-D cloud would otherwise add
    # seconds of latency without changing the collision decision.
    voxel = float(getattr(args, "active_path_scene_voxel", 0.005))
    if voxel > 0.0 and points.shape[0] > 0:
        keys = np.floor(points / voxel).astype(np.int64)
        _, unique_indices = np.unique(keys, axis=0, return_index=True)
        points = points[np.sort(unique_indices)]
    cap = int(max(getattr(args, "active_path_obstacle_cap", 50000), 1))
    if points.shape[0] > cap:
        if bool(getattr(args, "motion_path_fail_on_obstacle_cap", True)):
            raise UnsafeMotionTarget(
                f"active safety cloud has {points.shape[0]} occupied voxels > "
                f"cap={cap}; refusing to discard sparse obstacles"
            )
        indices = np.linspace(0, points.shape[0] - 1, cap, dtype=np.int64)
        points = points[indices]
    return points.astype(np.float32, copy=False)


