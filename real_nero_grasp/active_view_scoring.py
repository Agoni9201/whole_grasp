"""Pure scoring helpers for target-centered spherical active views."""

import numpy as np

from .visualization import get_o3d


__all__ = [
    "_estimate_normals_toward_camera",
    "_target_visibility_score",
    "_clamp01",
    "_normalize_dir_xy",
    "_dir_from_angle_rad",
    "_smallest_angle_diff_deg",
    "_pca_main_axes_xy",
    "_predict_visibility_from_viewdir",
    "_direction_context_penalty",
    "_active_spherical_azimuths",
    "_parse_active_orbit_tilts",
    "_active_spherical_camera_position",
    "_active_fibonacci_surface_bins",
    "_active_cap_weights",
    "_active_balanced_contact_coverage",
    "_build_active_information_proxy",
    "_score_active_candidate_information",
]


def _estimate_normals_toward_camera(
    points_base, camera_pos_base, *, get_o3d_hook=None
):
    pts = np.asarray(points_base, dtype=np.float32).reshape(-1, 3)
    if pts.shape[0] < 10:
        return np.zeros_like(pts)
    o3d = (get_o3d if get_o3d_hook is None else get_o3d_hook)()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.025, max_nn=30))
    normals = np.asarray(pcd.normals, dtype=np.float32).reshape(-1, 3)
    cam = np.asarray(camera_pos_base, dtype=np.float32).reshape(3)
    to_cam = cam.reshape(1, 3) - pts
    flip = np.sum(normals * to_cam, axis=1) < 0.0
    normals[flip] *= -1.0
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = normals / np.maximum(norms, 1e-6)
    return normals.astype(np.float32)


def _target_visibility_score(points_base, normals_base, camera_pos_base):
    pts = np.asarray(points_base, dtype=np.float32).reshape(-1, 3)
    normals = np.asarray(normals_base, dtype=np.float32).reshape(-1, 3)
    if pts.shape[0] == 0 or normals.shape[0] != pts.shape[0]:
        return 0.0
    to_cam = np.asarray(camera_pos_base, dtype=np.float32).reshape(1, 3) - pts
    to_cam = to_cam / np.maximum(np.linalg.norm(to_cam, axis=1, keepdims=True), 1e-6)
    vis = np.clip(np.sum(normals * to_cam, axis=1), 0.0, 1.0)
    return float(np.mean(vis))


def _clamp01(x):
    return np.clip(np.asarray(x, dtype=np.float32), 0.0, 1.0)


def _normalize_dir_xy(dir_xy, fallback=(1.0, 0.0)):
    d = np.asarray(dir_xy, dtype=np.float32).reshape(2)
    n = float(np.linalg.norm(d))
    if n < 1e-6 or not np.isfinite(n):
        d = np.asarray(fallback, dtype=np.float32).reshape(2)
        n = float(np.linalg.norm(d))
    return (d / max(n, 1e-6)).astype(np.float32)


def _dir_from_angle_rad(angle_rad):
    return np.array([np.cos(float(angle_rad)), np.sin(float(angle_rad))], dtype=np.float32)


def _smallest_angle_diff_deg(a_rad, b_rad):
    d = (float(a_rad) - float(b_rad) + np.pi) % (2.0 * np.pi) - np.pi
    return abs(float(np.rad2deg(d)))


def _pca_main_axes_xy(points_base):
    pts = np.asarray(points_base, dtype=np.float32).reshape(-1, 3)
    if pts.shape[0] < 10:
        return np.array([1.0, 0.0], dtype=np.float32), np.array([0.0, 1.0], dtype=np.float32)
    xy = pts[:, :2]
    xy = xy - np.mean(xy, axis=0, keepdims=True)
    cov = (xy.T @ xy) / max(xy.shape[0] - 1, 1)
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1]
    major = _normalize_dir_xy(vecs[:, order[0]])
    minor = _normalize_dir_xy(vecs[:, order[1]], fallback=(-major[1], major[0]))
    return major, minor


def _predict_visibility_from_viewdir(normals_base, view_dir_base):
    normals = np.asarray(normals_base, dtype=np.float32).reshape(-1, 3)
    v = np.asarray(view_dir_base, dtype=np.float32).reshape(3)
    v = v / max(float(np.linalg.norm(v)), 1e-6)
    return _clamp01(normals @ v)


def _direction_context_penalty(center_base, context_points_base, dir_xy, weights=None, near_radius=0.32, sigma_deg=30.0):
    pts = np.asarray(context_points_base, dtype=np.float32).reshape(-1, 3)
    if pts.shape[0] == 0:
        return 0.0, 1.0
    w = np.ones((pts.shape[0],), dtype=np.float32) if weights is None else np.asarray(weights, dtype=np.float32).reshape(-1)
    if w.shape[0] != pts.shape[0]:
        w = np.ones((pts.shape[0],), dtype=np.float32)
    center = np.asarray(center_base, dtype=np.float32).reshape(3)
    d = _normalize_dir_xy(dir_xy)
    rel = pts[:, :2] - center.reshape(1, 3)[:, :2]
    dist = np.linalg.norm(rel, axis=1)
    keep = (dist > 0.02) & (dist < float(near_radius))
    if not np.any(keep):
        return 0.0, 1.0
    rel = rel[keep] / np.maximum(dist[keep, None], 1e-6)
    align = np.clip(rel @ d.reshape(2), -1.0, 1.0)
    sigma = np.deg2rad(float(sigma_deg))
    ang = np.arccos(align)
    directional = np.exp(-0.5 * (ang / max(sigma, 1e-6)) ** 2)
    distance_weight = np.clip(1.0 - dist[keep] / float(near_radius), 0.0, 1.0)
    ww = np.clip(w[keep], 0.05, 5.0)
    raw = directional * distance_weight
    penalty = float(np.clip(np.sum(raw * ww) / max(float(np.sum(ww)), 1e-6), 0.0, 1.0))
    return penalty, float(1.0 - penalty)


def _active_spherical_azimuths(step_deg=22.5):
    """Return a deterministic full-circle azimuth grid in radians."""
    try:
        step = float(step_deg)
    except (TypeError, ValueError):
        step = 22.5
    if not np.isfinite(step) or step <= 0.0 or step > 360.0:
        raise ValueError(f"active spherical azimuth step must be in (0, 360], got {step_deg!r}")
    values_deg = np.arange(0.0, 360.0 - 1e-9, step, dtype=float)
    if values_deg.size == 0:
        values_deg = np.array([0.0], dtype=float)
    return np.deg2rad(values_deg)


def _parse_active_orbit_tilts(args):
    """Parse polar tilts without clamping out-of-range entries into duplicates."""
    max_angle = max(float(getattr(args, "active_orbit_max_angle_deg", 45.0)), 0.0)
    text = str(getattr(args, "active_nbv_orbit_degs", "20,30,35,45")).strip()
    parsed = []
    if text:
        for item in text.replace(",", " ").split():
            try:
                value = abs(float(item))
            except ValueError:
                continue
            if (
                np.isfinite(value)
                and value > 1e-6
                and value <= max_angle + 1e-9
                and not any(abs(value - old) <= 1e-6 for old in parsed)
            ):
                parsed.append(float(value))
    if parsed:
        return parsed
    defaults = [20.0, 30.0, 35.0, 45.0]
    fallback = [value for value in defaults if value <= max_angle + 1e-9]
    if fallback:
        return fallback
    return [max_angle] if max_angle > 1e-6 else []


def _active_spherical_camera_position(center, radius, azimuth_rad, tilt_deg):
    """Place a camera on a fixed-radius sphere using polar tilt from +Z."""
    center = np.asarray(center, dtype=float).reshape(3)
    radius = float(radius)
    if not np.isfinite(radius) or radius <= 0.0:
        raise ValueError(f"active spherical radius must be positive, got {radius!r}")
    phi = float(azimuth_rad)
    theta = np.deg2rad(abs(float(tilt_deg)))
    offset = radius * np.array(
        [
            np.sin(theta) * np.cos(phi),
            np.sin(theta) * np.sin(phi),
            np.cos(theta),
        ],
        dtype=float,
    )
    return center + offset


def _active_fibonacci_surface_bins(count=512, min_z=-0.20):
    """Sample observable object-envelope normals over the upper/side sphere."""
    count = max(int(count), 32)
    min_z = float(np.clip(min_z, -0.95, 0.95))
    z = np.linspace(1.0, min_z, count, dtype=np.float64)
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    phase = golden_angle * np.arange(count, dtype=np.float64)
    radial = np.sqrt(np.maximum(1.0 - z * z, 0.0))
    bins = np.column_stack((radial * np.cos(phase), radial * np.sin(phase), z))
    return bins.astype(np.float32)


def _active_cap_weights(bin_normals, direction, sigma_deg):
    bins = np.asarray(bin_normals, dtype=np.float32).reshape(-1, 3)
    direction = np.asarray(direction, dtype=np.float32).reshape(3)
    direction = direction / max(float(np.linalg.norm(direction)), 1e-6)
    dots = np.clip(bins @ direction, -1.0, 1.0)
    angles = np.arccos(dots)
    sigma = max(np.deg2rad(float(sigma_deg)), 1e-6)
    return np.exp(-0.5 * (angles / sigma) ** 2).astype(np.float32)


def _active_balanced_contact_coverage(coverage, positive_weights, negative_weights):
    coverage = _clamp01(coverage)
    positive_weights = np.asarray(positive_weights, dtype=np.float32).reshape(-1)
    negative_weights = np.asarray(negative_weights, dtype=np.float32).reshape(-1)
    c_pos = float(
        np.sum(positive_weights * coverage)
        / max(float(np.sum(positive_weights)), 1e-6)
    )
    c_neg = float(
        np.sum(negative_weights * coverage)
        / max(float(np.sum(negative_weights)), 1e-6)
    )
    # Completing the missing opposite contact is more useful than repeatedly
    # improving only one side of a parallel-jaw grasp.
    return float(np.clip(0.25 * (c_pos + c_neg) + 0.50 * min(c_pos, c_neg), 0.0, 1.0))


def _build_active_information_proxy(
    target_pts,
    target_normals,
    target_center,
    current_camera_pos,
    args,
):
    """Build a light-weight surface and parallel-jaw information model.

    This intentionally does not claim to be a GraspNet result.  Active views
    are selected before GraspNet runs, so contact regions are PCA-based
    parallel-jaw proxies and sweep gain is a directional observability proxy.
    All later measured-cloud collision checks remain hard gates.
    """
    points = np.asarray(target_pts, dtype=np.float32).reshape(-1, 3)
    finite = np.isfinite(points).all(axis=1)
    points = points[finite]
    center = np.asarray(target_center, dtype=np.float32).reshape(3)
    current_camera = np.asarray(current_camera_pos, dtype=np.float32).reshape(3)
    bin_count = int(getattr(args, "active_nbv_surface_bin_count", 512))
    bins = _active_fibonacci_surface_bins(
        bin_count,
        min_z=float(getattr(args, "active_nbv_surface_min_normal_z", -0.20)),
    )
    current_dir = current_camera - center
    current_dir = current_dir / max(float(np.linalg.norm(current_dir)), 1e-6)
    current_visibility = np.clip(bins @ current_dir, 0.0, 1.0) ** 1.5

    normals = np.asarray(target_normals, dtype=np.float32).reshape(-1, 3)
    if normals.shape[0] == finite.shape[0]:
        normals = normals[finite]
    if normals.shape[0] != points.shape[0]:
        normals = np.zeros_like(points)
    radial = points - center.reshape(1, 3)
    radial_norm = np.linalg.norm(radial, axis=1, keepdims=True)
    valid_radial = radial_norm[:, 0] > 0.003
    radial = radial / np.maximum(radial_norm, 1e-6)
    normal_norm = np.linalg.norm(normals, axis=1, keepdims=True)
    valid_normal = normal_norm[:, 0] > 1e-5
    normals = normals / np.maximum(normal_norm, 1e-6)
    flip = np.sum(normals * radial, axis=1) < 0.0
    normals[flip] *= -1.0
    observation_dirs = np.where(valid_normal[:, None], normals, radial)
    observation_valid = valid_radial | valid_normal
    observation_dirs = observation_dirs[observation_valid]
    observation_strength = np.clip(
        observation_dirs @ current_dir,
        0.0,
        1.0,
    )
    useful = observation_strength > 0.02
    observation_dirs = observation_dirs[useful]
    observation_strength = observation_strength[useful]
    if observation_dirs.shape[0] > 1024:
        sample_index = np.linspace(
            0,
            observation_dirs.shape[0] - 1,
            1024,
            dtype=int,
        )
        observation_dirs = observation_dirs[sample_index]
        observation_strength = observation_strength[sample_index]

    observed_coverage = np.zeros((bins.shape[0],), dtype=np.float32)
    if observation_dirs.shape[0] > 0:
        dots = np.clip(bins @ observation_dirs.T, -1.0, 1.0)
        sigma = max(
            np.deg2rad(float(getattr(args, "active_nbv_surface_coverage_sigma_deg", 12.0))),
            1e-6,
        )
        kernel = np.exp(-0.5 * (np.arccos(dots) / sigma) ** 2)
        kernel *= observation_strength.reshape(1, -1)
        observed_coverage = np.max(kernel, axis=1).astype(np.float32)
    # The center view itself is known to have been captured.  Blend its ideal
    # envelope visibility conservatively so the same pose has zero novelty.
    baseline_coverage = np.maximum(
        observed_coverage,
        0.35 * current_visibility.astype(np.float32),
    )

    major_xy, minor_xy = _pca_main_axes_xy(points)
    max_opening = float(
        getattr(
            args,
            "max_grasp_width",
            None,
        )
        or getattr(args, "gripper_open_width", 0.100)
    )
    width_margin = max(float(getattr(args, "active_nbv_grasp_width_margin", 0.002)), 0.0)
    contact_sigma = float(getattr(args, "active_nbv_contact_sigma_deg", 25.0))
    grasp_proxies = []
    for axis_name, axis_xy in (("major", major_xy), ("minor", minor_xy)):
        axis = np.array([axis_xy[0], axis_xy[1], 0.0], dtype=np.float32)
        projections = (
            (points - center.reshape(1, 3)) @ axis
            if points.shape[0] > 0
            else np.zeros((0,), dtype=np.float32)
        )
        width = (
            float(np.percentile(projections, 95) - np.percentile(projections, 5))
            if projections.size
            else float("inf")
        )
        if not np.isfinite(width) or width > max(max_opening - width_margin, 0.0):
            continue
        tangent = np.array([-axis[1], axis[0], 0.0], dtype=np.float32)
        sweep_dirs = []
        for direction in (axis, -axis, tangent, -tangent):
            shoulder = direction + np.array([0.0, 0.0, 0.75], dtype=np.float32)
            shoulder /= max(float(np.linalg.norm(shoulder)), 1e-6)
            sweep_dirs.append(shoulder)
        sweep_weights = np.maximum.reduce(
            [_active_cap_weights(bins, direction, 28.0) for direction in sweep_dirs]
        )
        grasp_proxies.append(
            {
                "source": "pca_proxy",
                "axis_name": axis_name,
                "width_axis": axis,
                "width_m": float(width),
                "positive_contact_weights": _active_cap_weights(
                    bins, axis, contact_sigma
                ),
                "negative_contact_weights": _active_cap_weights(
                    bins, -axis, contact_sigma
                ),
                "sweep_weights": sweep_weights.astype(np.float32),
                "sweep_gain_mode": "directional_proxy_no_free_space",
            }
        )

    target_radius = (
        float(np.percentile(np.linalg.norm(points - center.reshape(1, 3), axis=1), 90))
        if points.shape[0] > 0
        else 0.0
    )
    return {
        "center": center.astype(np.float32),
        "current_camera": current_camera.astype(np.float32),
        "current_view_dir": current_dir.astype(np.float32),
        "bin_normals": bins,
        "observed_coverage": _clamp01(baseline_coverage).astype(np.float32),
        "current_visibility": _clamp01(current_visibility).astype(np.float32),
        "grasp_proxies": grasp_proxies,
        "contact_proxy_available": bool(grasp_proxies),
        "contact_source": "pca_proxy" if grasp_proxies else "unavailable_width",
        "sweep_gain_mode": (
            "directional_proxy_no_free_space" if grasp_proxies else "unavailable"
        ),
        "target_radius": float(target_radius),
        "reference_depth": float(np.linalg.norm(current_camera - center)),
        "current_los_penalty": 0.0,
        "current_direction_collision": 0.0,
    }


def _score_active_candidate_information(
    proxy,
    cam_pos,
    los_penalty,
    direction_collision,
    center_norm,
    depth,
    joint_max_delta_deg,
    joint_l2_delta_deg,
    args,
    *,
    trajectory_penalty=0.0,
    camera_occupancy=0.0,
    extra_visibility=None,
):
    """Score one true 3-D (azimuth, tilt) candidate with auditable terms."""
    bins = np.asarray(proxy["bin_normals"], dtype=np.float32).reshape(-1, 3)
    center = np.asarray(proxy["center"], dtype=np.float32).reshape(3)
    view_dir = np.asarray(cam_pos, dtype=np.float32).reshape(3) - center
    view_dir = view_dir / max(float(np.linalg.norm(view_dir)), 1e-6)
    visibility = (np.clip(bins @ view_dir, 0.0, 1.0) ** 1.5).astype(np.float32)
    baseline_visibility = np.asarray(
        proxy.get("current_visibility", np.zeros_like(visibility)),
        dtype=np.float32,
    ).reshape(-1)
    if extra_visibility is not None:
        extra = np.asarray(extra_visibility, dtype=np.float32).reshape(-1)
        if extra.shape == baseline_visibility.shape:
            baseline_visibility = np.maximum(baseline_visibility, _clamp01(extra))
    observed = np.asarray(
        proxy.get("observed_coverage", np.zeros_like(visibility)),
        dtype=np.float32,
    ).reshape(-1)
    before = np.maximum(_clamp01(observed), _clamp01(baseline_visibility))
    after = np.maximum(before, visibility)
    increment = np.maximum(after - before, 0.0)
    surface_gain = float(np.mean(increment)) if increment.size else 0.0

    contact_gain = 0.0
    sweep_gain = 0.0
    for grasp_proxy in proxy.get("grasp_proxies", []):
        before_balance = _active_balanced_contact_coverage(
            before,
            grasp_proxy["positive_contact_weights"],
            grasp_proxy["negative_contact_weights"],
        )
        after_balance = _active_balanced_contact_coverage(
            after,
            grasp_proxy["positive_contact_weights"],
            grasp_proxy["negative_contact_weights"],
        )
        contact_gain = max(contact_gain, after_balance - before_balance)
        sweep_weights = np.asarray(
            grasp_proxy.get("sweep_weights", np.zeros_like(increment)),
            dtype=np.float32,
        ).reshape(-1)
        if sweep_weights.shape == increment.shape:
            sweep_value = float(
                np.sum(sweep_weights * increment)
                / max(float(np.sum(sweep_weights)), 1e-6)
            )
            sweep_gain = max(sweep_gain, sweep_value)
    contact_gain = float(np.clip(contact_gain, 0.0, 1.0))
    sweep_gain = float(np.clip(sweep_gain, 0.0, 1.0))

    candidate_los = float(np.clip(los_penalty, 0.0, 1.0))
    candidate_direction_collision = float(np.clip(direction_collision, 0.0, 1.0))
    current_los = float(np.clip(proxy.get("current_los_penalty", 0.0), 0.0, 1.0))
    current_direction = float(
        np.clip(proxy.get("current_direction_collision", 0.0), 0.0, 1.0)
    )
    los_quality = 1.0 - candidate_los
    direction_quality = 1.0 - candidate_direction_collision
    los_improvement = float(
        np.clip((current_los - candidate_los) / max(current_los, 0.10), 0.0, 1.0)
    )
    direction_improvement = float(
        np.clip(current_direction - candidate_direction_collision, 0.0, 1.0)
    )
    # Quality distinguishes a clear candidate even when the center view was
    # already clear; improvement ensures an actual occluder-removal benefit is
    # rewarded when the baseline was blocked.
    occlusion_gain = float(
        np.clip(
            0.45 * los_quality
            + 0.20 * direction_quality
            + 0.25 * los_improvement
            + 0.10 * direction_improvement,
            0.0,
            1.0,
        )
    )

    max_center_norm = max(
        float(getattr(args, "active_view_max_center_norm", 0.75)),
        1e-6,
    )
    fov_quality = float(np.clip(1.0 - float(center_norm) / max_center_norm, 0.0, 1.0))
    reference_depth = max(float(proxy.get("reference_depth", depth)), 1e-6)
    depth_scale = max(0.15 * reference_depth, 0.05)
    depth_quality = float(
        np.exp(-((float(depth) - reference_depth) / depth_scale) ** 2)
    )
    projected_quality = float(np.clip(np.mean(visibility) * 2.0, 0.0, 1.0))
    image_quality = float(
        np.clip(
            0.45 * fov_quality + 0.30 * depth_quality + 0.25 * projected_quality,
            0.0,
            1.0,
        )
    )

    redundancy = float(
        np.sum(np.minimum(visibility, baseline_visibility))
        / max(float(np.sum(np.maximum(visibility, baseline_visibility))), 1e-6)
    )
    max_joint = max(
        float(getattr(args, "active_side_max_joint_delta_deg", 45.0)),
        1e-6,
    )
    max_l2 = max(
        float(getattr(args, "active_side_max_joint_l2_deg", 75.0)),
        1e-6,
    )
    joint_max_cost = (
        float(np.clip(float(joint_max_delta_deg) / max_joint, 0.0, 1.0))
        if np.isfinite(joint_max_delta_deg)
        else 1.0
    )
    joint_l2_cost = (
        float(np.clip(float(joint_l2_delta_deg) / max_l2, 0.0, 1.0))
        if np.isfinite(joint_l2_delta_deg)
        else 1.0
    )
    current_dir = np.asarray(proxy["current_view_dir"], dtype=np.float32).reshape(3)
    arc_angle = float(
        np.rad2deg(np.arccos(np.clip(float(np.dot(current_dir, view_dir)), -1.0, 1.0)))
    )
    max_arc = max(float(getattr(args, "active_orbit_max_angle_deg", 45.0)), 1e-6)
    arc_cost = float(np.clip(arc_angle / max_arc, 0.0, 1.0))
    motion_cost = float(
        np.clip(
            0.40 * joint_l2_cost
            + 0.25 * joint_max_cost
            + 0.15 * arc_cost
            + 0.10 * float(np.clip(trajectory_penalty, 0.0, 1.0))
            + 0.10 * float(np.clip(camera_occupancy, 0.0, 1.0)),
            0.0,
            1.0,
        )
    )

    surface_weight = float(getattr(args, "active_nbv_surface_weight", 0.35))
    contact_weight = float(getattr(args, "active_nbv_grasp_contact_weight", 0.30))
    sweep_weight = float(getattr(args, "active_nbv_sweep_weight", 0.15))
    occlusion_weight = float(getattr(args, "active_nbv_occlusion_weight", 0.10))
    image_weight = float(getattr(args, "active_nbv_image_weight", 0.10))
    redundancy_weight = float(getattr(args, "active_nbv_redundancy_weight", 0.10))
    motion_weight = float(getattr(args, "active_nbv_motion_weight", 0.10))
    score = (
        surface_weight * surface_gain
        + contact_weight * contact_gain
        + sweep_weight * sweep_gain
        + occlusion_weight * occlusion_gain
        + image_weight * image_quality
        - redundancy_weight * redundancy
        - motion_weight * motion_cost
    )
    return {
        "score": float(score),
        "surface_gain": surface_gain,
        "contact_gain": contact_gain,
        "sweep_gain": sweep_gain,
        "occlusion_gain": occlusion_gain,
        "image_quality": image_quality,
        "redundancy": float(np.clip(redundancy, 0.0, 1.0)),
        "motion_cost": motion_cost,
        "visibility_vector": visibility,
        "baseline_visibility_vector": baseline_visibility.astype(np.float32),
        "contact_proxy_available": bool(proxy.get("contact_proxy_available", False)),
        "contact_source": str(proxy.get("contact_source", "unavailable")),
        "sweep_gain_mode": str(proxy.get("sweep_gain_mode", "unavailable")),
        "los_improvement": los_improvement,
        "direction_improvement": direction_improvement,
    }

