"""High-level target-centered active-view selection and observation workflow.

This module owns camera look-at/roll enumeration, spherical next-best-view
candidate construction, candidate ranking, and the active observation state
machine. Hardware/perception/fusion operations that still live in the legacy
workflow facade are supplied explicitly through ActiveViewHooks.
The module intentionally never imports real_nero_grasp.runtime.
"""

import copy
import os
import time
from dataclasses import dataclass, replace
from typing import Callable, Optional

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as R

from .pointcloud_fusion import (
    _make_cloud_from_points_colors,
    _points_from_cloud,
    fuse_target_clouds,
    fuse_target_clouds_surfel,
    next_dynamic_environment_generation,
    transform_cloud,
)
from .transforms import _as_pose_list, _make_T_from_pose
from .types import (
    ActiveViewResult,
    GraspMotionPlan,
    GraspSelectionDeclined,
    MotionNotReached,
    PlanNotExecutable,
    PlannedJointStage,
    SafetyStopTriggered,
    UnsafeMotionTarget,
)


def _wait_for_active_capture_settle(
    args,
    label: str,
    *,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> float:
    """Wait after an active-view move before sampling pose and RGB-D data."""

    seconds = max(
        0.0,
        float(getattr(args, "active_capture_settle_seconds", 0.0)),
    )
    if seconds <= 0.0:
        return 0.0
    print(
        f"[active][settle] {label} reached; wait {seconds:.2f}s "
        "before reading the capture pose and RGB-D frame"
    )
    sleep_fn(seconds)
    return seconds


@dataclass(frozen=True)
class ActiveViewHooks:
    """Explicit dependencies retained by the high-level active-view workflow."""

    active_motion_scene_points: Optional[Callable] = None
    active_spherical_camera_position: Optional[Callable] = None
    build_active_information_proxy: Optional[Callable] = None
    build_scene_aware_nbv_records: Optional[Callable] = None
    direction_context_penalty: Optional[Callable] = None
    estimate_normals_toward_camera: Optional[Callable] = None
    line_segment_obstacle_metrics: Optional[Callable] = None
    make_cloud_from_points_colors: Optional[Callable] = None
    nearest_obstacle_distance: Optional[Callable] = None
    normalize_dir_xy: Optional[Callable] = None
    parse_active_orbit_tilts: Optional[Callable] = None
    prepare_active_obstacle_cloud: Optional[Callable] = None
    score_active_candidate_information: Optional[Callable] = None
    solve_ik_for_flange_pose: Optional[Callable] = None
    target_visibility_score: Optional[Callable] = None
    get_ik_solver: Optional[Callable] = None
    joint_limit_margin: Optional[Callable] = None
    handeye_transform: Optional[Callable] = None
    look_at_camera_pose_flange_roll_candidates: Optional[Callable] = None
    rank_active_lookat_roll_solutions: Optional[Callable] = None
    spread_equivalent_roll_candidates: Optional[Callable] = None

    active_center_joint_state_is_verified: Optional[Callable] = None
    active_motion_capture_base: Optional[Callable] = None
    active_side_capture_overlap_ok: Optional[Callable] = None
    active_side_selection_order: Optional[Callable] = None
    active_view_second_capture_args: Optional[Callable] = None
    build_active_view_candidates: Optional[Callable] = None
    capture_active_projected_target_cloud: Optional[Callable] = None
    capture_target_cloud: Optional[Callable] = None
    confirm_motion: Optional[Callable] = None
    crop_scene_cloud_near_target: Optional[Callable] = None
    diagnose_active_fusion: Optional[Callable] = None
    execute_prevalidated_stages: Optional[Callable] = None
    format_active_view_candidate: Optional[Callable] = None
    fuse_dynamic_environment_clouds: Optional[Callable] = None
    fuse_target_clouds: Optional[Callable] = None
    merge_clouds: Optional[Callable] = None
    modeled_tcp_pose_for_joints: Optional[Callable] = None
    move_flange_pose: Optional[Callable] = None
    move_tcp_pose: Optional[Callable] = None
    plan_active_direct_route: Optional[Callable] = None
    plan_active_up_and_over_route: Optional[Callable] = None
    points_from_cloud: Optional[Callable] = None
    preview_fused_cloud: Optional[Callable] = None
    print_active_view_candidates: Optional[Callable] = None
    register_active_fusion_views: Optional[Callable] = None
    save_cloud: Optional[Callable] = None
    should_use_active_multiview: Optional[Callable] = None
    single_view_active_result: Optional[Callable] = None
    smallest_angle_diff_deg: Optional[Callable] = None
    target_center_base_from_cloud: Optional[Callable] = None
    target_center_overhead_flange_pose_candidates: Optional[Callable] = None
    T_base_camera_from_flange: Optional[Callable] = None
    transform_cloud: Optional[Callable] = None
    urdf_chain_geometry: Optional[Callable] = None


def _require_hook(hooks, name):
    if hooks is None:
        raise RuntimeError(f"active-view dependency hook '{name}' was not supplied")
    value = getattr(hooks, name, None)
    if not callable(value):
        raise RuntimeError(f"active-view dependency hook '{name}' is unavailable")
    return value


def _resolve_dynamic_environment_fuser(hooks):
    """Resolve the new environment-map builder without breaking old facades."""
    hook = getattr(hooks, "fuse_dynamic_environment_clouds", None)
    if callable(hook):
        return hook
    try:
        from .pointcloud_fusion import fuse_dynamic_environment_clouds
    except (ImportError, AttributeError):
        return None
    return fuse_dynamic_environment_clouds


def _exclude_target_band_from_environment(environment_base, target_base, args):
    """Return the legacy tight-NN non-target fallback in ``base_link``."""
    environment_source = (
        environment_base.points
        if hasattr(environment_base, "points")
        else environment_base
    )
    target_source = (
        target_base.points if hasattr(target_base, "points") else target_base
    )
    environment_points = np.asarray(
        environment_source, dtype=np.float32
    ).reshape(-1, 3)
    target_points = np.asarray(target_source, dtype=np.float32).reshape(-1, 3)
    exclusion = max(
        float(getattr(args, "motion_target_exclusion_dist", 0.002)),
        0.0,
    )
    if (
        environment_points.shape[0] == 0
        or target_points.shape[0] == 0
        or exclusion <= 0.0
    ):
        return environment_points
    distances, _ = cKDTree(target_points).query(
        environment_points,
        k=1,
        workers=-1,
    )
    return environment_points[np.asarray(distances) > exclusion]


def _build_dynamic_motion_environment(
    view_names,
    environment_base_clouds,
    args,
    T_camera0_base,
    *,
    hooks,
    target_base=None,
    map_id="active_dynamic_environment",
    target_center_base=None,
):
    """Build one generation of the base-frame map and its camera0 facade.

    Motion occupancy is deliberately assembled from every raw capture before
    target-registration/quality rejection.  The returned Open3D cloud exists
    only for legacy consumers; the canonical map remains in ``base_link``.
    """
    if not bool(getattr(args, "motion_environment_map_enabled", True)):
        return None, None

    view_names = tuple(view_names)
    environment_base_clouds = tuple(environment_base_clouds)
    if len(view_names) != len(environment_base_clouds):
        raise ValueError(
            "dynamic environment source mismatch: "
            f"view_names={len(view_names)} clouds={len(environment_base_clouds)}"
        )

    fuser = _resolve_dynamic_environment_fuser(hooks)
    if fuser is None:
        return None, None
    names = []
    clouds = []
    for name, cloud in zip(view_names, environment_base_clouds):
        if cloud is None:
            continue
        names.append(str(name))
        clouds.append(cloud)
    if not clouds:
        return None, None

    generation = next_dynamic_environment_generation(args)
    environment_map = fuser(
        names,
        clouds,
        args,
        target_base=target_base,
        map_id=str(map_id),
        generation=generation,
        target_center_base=target_center_base,
    )
    hard = np.asarray(
        getattr(environment_map, "hard_environment_base", np.zeros((0, 3))),
        dtype=np.float32,
    ).reshape(-1, 3)
    uncertain = np.asarray(
        getattr(
            environment_map,
            "uncertain_environment_base",
            np.zeros((0, 3)),
        ),
        dtype=np.float32,
    ).reshape(-1, 3)
    compatibility_points = np.vstack([hard, uncertain])
    make_cloud = getattr(hooks, "make_cloud_from_points_colors", None)
    if not callable(make_cloud):
        # Production captures are Open3D clouds and can safely use the package
        # builder.  Lightweight unit-test cloud sentinels intentionally avoid
        # importing Open3D in a headless process.
        make_cloud = (
            _make_cloud_from_points_colors
            if any(callable(getattr(cloud, "transform", None)) for cloud in clouds)
            else None
        )
    if make_cloud is None:
        print(
            f"[active][motion][map] id={map_id} generation={generation} "
            "compatibility cloud deferred (no point-cloud builder hook)"
        )
        return environment_map, None
    transform_impl = getattr(hooks, "transform_cloud", None)
    if not callable(transform_impl):
        transform_impl = transform_cloud
    try:
        compatibility_cloud = transform_impl(
            make_cloud(compatibility_points),
            np.asarray(T_camera0_base, dtype=float).reshape(4, 4),
        )
    except Exception as exc:
        # The camera0 cloud is only a compatibility facade for legacy callers.
        # Never discard a successfully built canonical base-frame map because
        # Open3D allocation/display conversion failed.
        print(
            f"[active][motion][map][warn] compatibility cloud unavailable; "
            f"keep canonical map id={map_id} generation={generation}: "
            f"{type(exc).__name__}: {exc}"
        )
        return environment_map, None
    print(
        f"[active][motion][map] id={map_id} generation={generation} "
        f"views={names} hard={hard.shape[0]} uncertain={uncertain.shape[0]}"
    )
    return environment_map, compatibility_cloud


def _fuse_active_target_clouds(
    target_base_clouds,
    args,
    T_camera0_base,
    *,
    camera_origins_base=None,
    merge_clouds_impl,
    transform_cloud_impl,
    hooks,
):
    """Return raw diagnostics plus the target-specific weighted fusion."""
    raw_base = merge_clouds_impl(
        target_base_clouds,
        args.active_fusion_voxel,
        filter_outliers=False,
        outlier_neighbors=int(args.outlier_neighbors),
        outlier_std_ratio=float(args.outlier_std_ratio),
    )
    # Keep lightweight/offline callers and unit-test cloud sentinels on the
    # historical merge path.  Live captures are Open3D clouds and expose this
    # method; they use the confidence-weighted implementation below.
    if not all(
        cloud is not None
        and callable(getattr(cloud, "voxel_down_sample", None))
        for cloud in target_base_clouds
    ):
        print(
            "[target][fusion][warn] target clouds do not expose Open3D "
            "geometry; use compatibility merge without weighted fusion"
        )
        filtered_base = merge_clouds_impl(
            target_base_clouds,
            args.active_fusion_voxel,
            filter_outliers=bool(args.fusion_target_outlier_filter),
            outlier_neighbors=int(args.outlier_neighbors),
            outlier_std_ratio=float(args.outlier_std_ratio),
        )
        return (
            transform_cloud_impl(raw_base, T_camera0_base),
            transform_cloud_impl(filtered_base, T_camera0_base),
        )
    fusion_mode = str(
        getattr(args, "active_target_fusion_mode", "voxel")
    ).strip().lower()
    if fusion_mode == "surfel":
        fuser = fuse_target_clouds_surfel
    else:
        fuser = getattr(hooks, "fuse_target_clouds", None)
        if not callable(fuser):
            fuser = fuse_target_clouds
    extra_fusion_kwargs = {}
    if fusion_mode == "surfel":
        extra_fusion_kwargs.update(
            merge_radius=float(
                getattr(args, "active_surfel_merge_radius", 0.008)
            ),
            normal_angle_deg=float(
                getattr(args, "active_surfel_normal_angle_deg", 55.0)
            ),
            normal_neighbors=int(
                getattr(args, "active_surfel_normal_neighbors", 12)
            ),
            camera_origins=camera_origins_base,
        )
    fused_base, stats = fuser(
        target_base_clouds,
        args.active_fusion_voxel,
        single_view_radius=float(
            getattr(args, "target_fusion_single_view_radius", 0.008)
        ),
        single_view_min_neighbors=int(
            getattr(args, "target_fusion_single_view_min_neighbors", 2)
        ),
        filter_outliers=bool(args.fusion_target_outlier_filter),
        outlier_neighbors=int(args.outlier_neighbors),
        outlier_std_ratio=float(args.outlier_std_ratio),
        return_stats=True,
        **extra_fusion_kwargs,
    )
    print(
        f"[target][fusion] mode={fusion_mode} raw and fused clouds prepared; "
        f"retained={int(stats.get('retained_points', len(fused_base.points)))}"
    )
    return (
        transform_cloud_impl(raw_base, T_camera0_base),
        transform_cloud_impl(fused_base, T_camera0_base),
    )


__all__ = [
    "ActiveViewHooks",
    "_candidate_pose_center_norm",
    "_look_at_camera_pose",
    "_rotation_error_deg_between_poses",
    "_look_at_camera_pose_flange_roll_candidates",
    "_look_at_camera_pose_minimize_flange_rotation",
    "_spread_equivalent_roll_candidates",
    "_rank_active_lookat_roll_solutions",
    "_rot_z",
    "_limit_flange_orientation",
    "_target_center_base_from_cloud",
    "_should_use_active_multiview",
    "_target_center_overhead_flange_pose_candidates",
    "_target_center_overhead_flange_pose",
    "_target_center_optical_relaxation_pose_records",
    "_build_active_view_candidates",
    "_format_active_view_candidate",
    "_print_active_view_candidates",
    "_active_view_candidate_angle",
    "_active_target_shape_profile",
    "_active_side_selection_order",
    "_single_view_active_result",
    "_active_center_joint_state_is_verified",
    "run_optional_active_view",
]


def _candidate_pose_center_norm(T_base_camera, target_center_base):
    p_base = np.r_[np.asarray(target_center_base, dtype=float).reshape(3), 1.0]
    p_cam = np.linalg.inv(np.asarray(T_base_camera, dtype=float).reshape(4, 4)) @ p_base
    if not np.isfinite(p_cam[:3]).all() or p_cam[2] <= 1e-6:
        return float("inf"), float(p_cam[2])
    return float(np.linalg.norm(p_cam[:2]) / max(abs(float(p_cam[2])), 1e-6)), float(p_cam[2])


def _look_at_camera_pose(cam_pos_base, target_base):
    cam_pos = np.asarray(cam_pos_base, dtype=float).reshape(3)
    target = np.asarray(target_base, dtype=float).reshape(3)
    z_axis = target - cam_pos
    z_norm = float(np.linalg.norm(z_axis))
    if z_norm < 1e-6:
        raise ValueError("camera position is too close to target for look-at pose")
    z_axis = z_axis / z_norm
    up = np.array([0.0, 0.0, 1.0], dtype=float)
    x_axis = np.cross(z_axis, up)
    if np.linalg.norm(x_axis) < 1e-6:
        x_axis = np.array([1.0, 0.0, 0.0], dtype=float)
    x_axis = x_axis / (np.linalg.norm(x_axis) + 1e-9)
    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis / (np.linalg.norm(y_axis) + 1e-9)
    R_base_camera = np.column_stack([x_axis, y_axis, z_axis])
    U, _, Vt = np.linalg.svd(R_base_camera)
    R_base_camera = U @ Vt
    if np.linalg.det(R_base_camera) < 0:
        U[:, -1] *= -1
        R_base_camera = U @ Vt
    T = np.eye(4)
    T[:3, :3] = R_base_camera
    T[:3, 3] = cam_pos
    return T


def _rotation_error_deg_between_poses(pose_a, pose_b):
    a = np.asarray(pose_a, dtype=float).reshape(6)
    b = np.asarray(pose_b, dtype=float).reshape(6)
    Ra = R.from_euler("xyz", a[3:])
    Rb = R.from_euler("xyz", b[3:])
    return float(np.rad2deg((Rb * Ra.inv()).magnitude()))


def _look_at_camera_pose_flange_roll_candidates(
    cam_pos_base,
    target_base,
    T_camera_flange,
    current_flange_pose,
    roll_samples=72,
):
    """Return equivalent look-at rolls ordered by flange rotation cost.

    Camera roll does not change the optical axis, but it can completely change
    Nero reachability near a wrist limit.  Keep all sampled equivalents so the
    guarded motion planner, rather than orientation distance alone, can choose
    the first roll with a valid IK path and collision-free joint sweep.
    """
    base_T = _look_at_camera_pose(cam_pos_base, target_base)
    T_camera_flange = np.asarray(T_camera_flange, dtype=float).reshape(4, 4)
    if current_flange_pose is None:
        T_base_flange = base_T @ T_camera_flange
        return [
            {
                "T_base_camera": base_T,
                "flange_pose": _as_pose_list(T_base_flange),
                "roll_rad": 0.0,
                "rotation_error_deg": 0.0,
            }
        ]
    current = np.asarray(current_flange_pose, dtype=float).reshape(6)
    candidates = []
    # Do not include both -pi and +pi: they are the same camera orientation.
    for roll in np.linspace(-np.pi, np.pi, int(max(roll_samples, 8)), endpoint=False):
        T_roll = np.eye(4, dtype=float)
        T_roll[:3, :3] = R.from_euler("z", float(roll)).as_matrix()
        T_base_camera = base_T @ T_roll
        T_base_flange = T_base_camera @ T_camera_flange
        flange_pose = _as_pose_list(T_base_flange)
        err = _rotation_error_deg_between_poses(current, flange_pose)
        candidates.append(
            {
                "T_base_camera": T_base_camera,
                "flange_pose": flange_pose,
                "roll_rad": float(roll),
                "rotation_error_deg": float(err),
            }
        )
    candidates.sort(
        key=lambda item: (
            float(item["rotation_error_deg"]),
            abs(float(item["roll_rad"])),
        )
    )
    return candidates


def _look_at_camera_pose_minimize_flange_rotation(cam_pos_base, target_base, T_camera_flange, current_flange_pose, roll_samples=72):
    """Look at the target using the equivalent roll with least flange rotation."""
    return _look_at_camera_pose_flange_roll_candidates(
        cam_pos_base,
        target_base,
        T_camera_flange,
        current_flange_pose,
        roll_samples=roll_samples,
    )[0]["T_base_camera"]


def _spread_equivalent_roll_candidates(candidates, limit):
    """Keep the least-rotation roll plus rolls spread over the full circle."""
    candidates = list(candidates)
    limit = max(int(limit), 1)
    if len(candidates) <= limit:
        return candidates
    selected = [candidates[0]]
    remaining = candidates[1:]

    def _circular_roll_distance(a, b):
        delta = (float(a) - float(b) + np.pi) % (2.0 * np.pi) - np.pi
        return abs(delta)

    while remaining and len(selected) < limit:
        best_index = max(
            range(len(remaining)),
            key=lambda index: (
                min(
                    _circular_roll_distance(
                        remaining[index]["roll_rad"],
                        old["roll_rad"],
                    )
                    for old in selected
                ),
                -float(remaining[index]["rotation_error_deg"]),
            ),
        )
        selected.append(remaining.pop(best_index))
    return selected


def _rank_active_lookat_roll_solutions(
    arm,
    args,
    cam_pos,
    target_center,
    T_camera_flange,
    current_flange,
    current_joints,
    label,
    roll_plan_limit=None,
    quick_seed_limit=None,
    allow_full_retry=True,
    *,
    hooks,
):
    """Rank equivalent look-at rolls by the actual endpoint joint motion.

    The camera optical axis is identical for every returned record.  A bounded,
    full-circle roll set is solved with TRAC-IK, then hard joint constraints are
    applied before ranking by max-joint sweep, L2 sweep, and flange rotation.
    """
    _solve_ik_for_flange_pose = _require_hook(
        hooks, "solve_ik_for_flange_pose"
    )
    _get_ik_solver = _require_hook(hooks, "get_ik_solver")
    _joint_limit_margin = _require_hook(hooks, "joint_limit_margin")
    _look_at_camera_pose_flange_roll_candidates = _require_hook(
        hooks, "look_at_camera_pose_flange_roll_candidates"
    )
    _spread_equivalent_roll_candidates = _require_hook(
        hooks, "spread_equivalent_roll_candidates"
    )
    roll_samples = max(int(getattr(args, "active_side_roll_samples", 16)), 8)
    plan_limit = max(
        int(
            getattr(args, "active_side_roll_plan_candidates", 8)
            if roll_plan_limit is None
            else roll_plan_limit
        ),
        1,
    )
    pose_records = _look_at_camera_pose_flange_roll_candidates(
        cam_pos,
        target_center,
        T_camera_flange,
        current_flange,
        roll_samples=roll_samples,
    )
    pose_records = _spread_equivalent_roll_candidates(pose_records, plan_limit)
    if (
        str(getattr(args, "motion_mode", "cartesian")) != "ik_joint"
        or current_joints is None
    ):
        record = dict(pose_records[0])
        record.update(
            {
                "planned_joint_target": None,
                "joint_max_delta_deg": float("inf"),
                "joint_l2_delta_deg": float("inf"),
                "joint_limit_margin_deg": float("nan"),
            }
        )
        return [record]

    current_joints = np.asarray(current_joints, dtype=float).reshape(7)
    max_joint = float(getattr(args, "active_side_max_joint_delta_deg", 45.0))
    max_l2 = float(getattr(args, "active_side_max_joint_l2_deg", 75.0))
    min_margin = max(
        float(getattr(args, "active_direct_min_joint_limit_margin_deg", 0.0)),
        0.0,
    )
    feasible = []
    failures = []
    quick_seed_limit = max(
        int(
            getattr(args, "active_side_roll_ik_seed_limit", 1)
            if quick_seed_limit is None
            else quick_seed_limit
        ),
        1,
    )

    def _evaluate_roll(
        index, pose_record, seed_limit, phase, planning_args=None
    ):
        planning_args = args if planning_args is None else planning_args
        try:
            joints = _solve_ik_for_flange_pose(
                arm,
                pose_record["flange_pose"],
                planning_args,
                f"{label}_roll{index}_{phase}",
                current_joints_override=current_joints,
                quiet=True,
                seed_limit=seed_limit,
            )
            joints = np.asarray(joints, dtype=float).reshape(7)
            delta = joints - current_joints
            joint_max = float(np.rad2deg(np.max(np.abs(delta))))
            joint_l2 = float(np.rad2deg(np.linalg.norm(delta)))
            if joint_max > max_joint:
                failures.append(
                    f"roll={np.rad2deg(pose_record['roll_rad']):+.1f}:"
                    f"jmax={joint_max:.1f}>{max_joint:.1f}"
                )
                return
            if joint_l2 > max_l2:
                failures.append(
                    f"roll={np.rad2deg(pose_record['roll_rad']):+.1f}:"
                    f"jl2={joint_l2:.1f}>{max_l2:.1f}"
                )
                return
            margin_deg = float("nan")
            if min_margin > 0.0:
                margin_deg = float(
                    np.rad2deg(
                        _joint_limit_margin(
                            joints, _get_ik_solver(planning_args)
                        )
                    )
                )
                if margin_deg < min_margin:
                    failures.append(
                        f"roll={np.rad2deg(pose_record['roll_rad']):+.1f}:"
                        f"margin={margin_deg:.2f}<{min_margin:.2f}"
                    )
                    return
            record = dict(pose_record)
            record.update(
                {
                    "planned_joint_target": joints,
                    "joint_max_delta_deg": joint_max,
                    "joint_l2_delta_deg": joint_l2,
                    "joint_limit_margin_deg": margin_deg,
                }
            )
            feasible.append(record)
        except Exception as exc:
            failures.append(
                f"roll={np.rad2deg(pose_record['roll_rad']):+.1f}:"
                f"{type(exc).__name__}"
            )

    for index, pose_record in enumerate(pose_records):
        _evaluate_roll(index, pose_record, quick_seed_limit, "quick")
    if not feasible and bool(allow_full_retry):
        # The current-joint seed is normally the desired local solution.  Only
        # when every sampled roll misses do a few least-rotation rolls receive
        # the full legacy seed set, preserving reachability without multiplying
        # the common-case planning cost by every roll and every seed.
        full_retry_count = min(
            max(int(getattr(args, "active_side_roll_full_retry_candidates", 3)), 1),
            len(pose_records),
        )
        retry_args = args
        retry_seed_limit = None
        retry_phase = "full"
        if bool(getattr(args, "active_nbv_selective_ik_rescue", False)):
            retry_args = copy.copy(args)
            retry_args.ik_all_joint_seed_count = max(
                int(
                    getattr(
                        args,
                        "active_nbv_selective_ik_rescue_all_joint_seeds",
                        8,
                    )
                ),
                int(getattr(args, "ik_all_joint_seed_count", 0)),
            )
            retry_seed_limit = max(
                int(
                    getattr(
                        args,
                        "active_nbv_selective_ik_rescue_seed_limit",
                        12,
                    )
                ),
                1,
            )
            retry_phase = "bounded_all_joint_rescue"
        for index, pose_record in enumerate(pose_records[:full_retry_count]):
            _evaluate_roll(
                index,
                pose_record,
                retry_seed_limit,
                retry_phase,
                planning_args=retry_args,
            )
            if feasible:
                break
    feasible.sort(
        key=lambda item: (
            float(item["joint_max_delta_deg"]),
            float(item["joint_l2_delta_deg"]),
            float(item["rotation_error_deg"]),
        )
    )
    if not feasible:
        sample = " | ".join(failures[:4]) if failures else "no roll candidates"
        raise PlanNotExecutable(
            f"{label} has no feasible look-at roll among {len(pose_records)} "
            f"full-circle samples; sample={sample}"
        )
    return feasible


def _rot_z(angle_rad):
    c = float(np.cos(angle_rad))
    s = float(np.sin(angle_rad))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=float)


def _limit_flange_orientation(current_flange_pose, desired_flange_pose, max_delta_deg):
    current = np.asarray(current_flange_pose, dtype=float).reshape(6)
    desired = np.asarray(desired_flange_pose, dtype=float).reshape(6)
    R_current = R.from_euler("xyz", current[3:])
    R_desired = R.from_euler("xyz", desired[3:])
    R_delta = R_desired * R_current.inv()
    rotvec = R_delta.as_rotvec()
    angle = float(np.linalg.norm(rotvec))
    max_angle = np.deg2rad(float(max_delta_deg))
    if angle > max_angle > 0.0:
        rotvec = rotvec * (max_angle / max(angle, 1e-9))
    R_limited = R.from_rotvec(rotvec) * R_current
    out = desired.copy()
    out[3:] = R_limited.as_euler("xyz", degrees=False)
    return out


def _target_center_base_from_cloud(target_cloud, T_base_camera):
    target_base = transform_cloud(target_cloud, T_base_camera)
    target_pts = _points_from_cloud(target_base)
    if target_pts.shape[0] == 0:
        raise RuntimeError("[active] target cloud is empty; cannot estimate target center")
    return np.median(target_pts, axis=0).astype(float)


def _should_use_active_multiview(
    target_cloud,
    scene_cloud,
    T_base_camera,
    args,
    obstacle_cloud=None,
    target_center_override=None,
    *,
    hooks,
):
    _prepare_active_obstacle_cloud = _require_hook(
        hooks, "prepare_active_obstacle_cloud"
    )
    if not bool(args.active_multiview_gate):
        print("[active][gate] disabled; run active multi-view whenever --active-view is set")
        return True
    T_base_camera = np.asarray(T_base_camera, dtype=float).reshape(4, 4)
    target_base = transform_cloud(target_cloud, T_base_camera)
    target_pts = _points_from_cloud(target_base)
    if target_pts.shape[0] == 0:
        print("[active][gate][warn] target cloud empty; keep single view")
        return False
    if target_center_override is None:
        target_center = np.median(target_pts, axis=0).astype(float)
    else:
        target_center = np.asarray(target_center_override, dtype=float).reshape(3)
    target_shape_profile = _active_target_shape_profile(
        target_pts,
        target_center=target_center,
        target_label=getattr(args, "_active_target_label", None),
    )
    print(
        "[ActiveNBV][shape] "
        f"elongation={float(target_shape_profile['elongation']):.3f} "
        f"bilateral_confidence="
        f"{float(target_shape_profile['bilateral_confidence']):.3f} "
        "policy=continuous_1.4_to_1.7"
    )

    if obstacle_cloud is not None:
        obstacle_base = transform_cloud(obstacle_cloud, T_base_camera)
        source_pts = _points_from_cloud(obstacle_base)
        source_is_non_target_mask = True
    else:
        scene_base = transform_cloud(scene_cloud, T_base_camera)
        source_pts = _points_from_cloud(scene_base)
        source_is_non_target_mask = False

    obstacle_pts, obstacle_weights, obstacle_meta = _prepare_active_obstacle_cloud(
        source_pts,
        target_pts,
        target_center,
        args,
        source_is_non_target_mask=source_is_non_target_mask,
    )
    if obstacle_pts.shape[0] == 0:
        print(
            "[active][gate] use_single_view: no non-target obstacle points after target/table filtering "
            f"mode={obstacle_meta.get('exclusion_mode', 'unknown')}"
        )
        return False

    tree = cKDTree(target_pts.astype(np.float32))
    dists, _ = tree.query(obstacle_pts.astype(np.float32), k=1, workers=-1)
    dists = np.asarray(dists, dtype=np.float32).reshape(-1)
    near_dist = float(args.active_multiview_obstacle_distance)
    near = dists <= near_dist
    near_count = int(np.count_nonzero(near))
    weighted_near = float(np.sum(np.clip(obstacle_weights[near], 0.05, 5.0))) if near_count > 0 else 0.0
    nearest = float(np.min(dists)) if dists.size else float("inf")
    need = near_count >= int(args.active_multiview_min_obstacle_points)
    decision = "use_multi_view" if need else "use_single_view"
    print(
        f"[active][gate] {decision}: nearest_obstacle_to_target={nearest * 1000.0:.1f}mm "
        f"near_count={near_count} weighted_near={weighted_near:.1f} "
        f"threshold={near_dist * 1000.0:.1f}mm min_points={int(args.active_multiview_min_obstacle_points)} "
        f"mode={obstacle_meta.get('exclusion_mode', 'unknown')} kept={int(obstacle_meta.get('kept', obstacle_pts.shape[0]))} "
        f"target_removed={int(obstacle_meta.get('target_removed', 0))} table_removed={int(obstacle_meta.get('table_removed', 0))}"
    )
    return bool(need)


def _target_center_overhead_flange_pose_candidates(
    args,
    target_center_base,
    current_flange_pose=None,
    *,
    hooks,
):
    handeye_transform = _require_hook(hooks, "handeye_transform")
    _look_at_camera_pose_flange_roll_candidates = _require_hook(
        hooks, "look_at_camera_pose_flange_roll_candidates"
    )
    target_center = np.asarray(target_center_base, dtype=float).reshape(3)
    cam_pos = target_center + np.array([0.0, 0.0, float(args.active_center_height)], dtype=float)
    T_camera_flange = np.linalg.inv(handeye_transform(args.calib_json, args.handeye_convention))
    return _look_at_camera_pose_flange_roll_candidates(
        cam_pos,
        target_center,
        T_camera_flange,
        current_flange_pose,
        roll_samples=int(getattr(args, "active_center_roll_samples", 72)),
    )


def _target_center_overhead_flange_pose(
    args,
    target_center_base,
    current_flange_pose=None,
    *,
    hooks,
):
    candidate = _target_center_overhead_flange_pose_candidates(
        args,
        target_center_base,
        current_flange_pose,
        hooks=hooks,
    )[0]
    return candidate["flange_pose"], candidate["T_base_camera"]


def _parse_active_center_optical_tilts(args):
    """Return small nonzero optical-axis relaxations in ascending order."""
    raw = str(getattr(args, "active_center_optical_tilts", "2,4,6"))
    values = []
    for item in raw.replace(",", " ").split():
        try:
            tilt = float(item)
        except ValueError:
            continue
        # This stage is deliberately a same-position near-overhead recovery,
        # not a replacement for the wider spherical observation search.
        if 0.05 <= tilt <= 15.0 and all(
            abs(tilt - old) > 1e-6 for old in values
        ):
            values.append(tilt)
    return tuple(sorted(values))


def _target_center_optical_relaxation_pose_records(
    args,
    target_center_base,
    current_flange_pose,
    current_camera_position,
    *,
    hooks,
):
    """Keep the overhead camera center fixed while relaxing its optical axis.

    An exactly downward optical axis can place Nero on a wrist singularity or
    hard joint boundary even though the requested camera position is readily
    reachable.  Roll-only search cannot escape that condition.  These records
    add a bounded two-dimensional optical-axis cone while retaining the same
    camera center and checking that the target remains inside the configured
    image-centering allowance.
    """
    handeye_transform = _require_hook(hooks, "handeye_transform")
    lookat_candidates = _require_hook(
        hooks, "look_at_camera_pose_flange_roll_candidates"
    )
    target = np.asarray(target_center_base, dtype=float).reshape(3)
    current_camera = np.asarray(current_camera_position, dtype=float).reshape(3)
    camera_position = target + np.array(
        [0.0, 0.0, float(args.active_center_height)], dtype=float
    )
    look_distance = max(float(args.active_center_height), 0.05)
    rel_xy = current_camera[:2] - target[:2]
    preferred_azimuth = (
        float(np.arctan2(rel_xy[1], rel_xy[0]))
        if float(np.linalg.norm(rel_xy)) > 1e-6
        else 0.0
    )
    step_deg = float(
        np.clip(
            getattr(args, "active_center_optical_azimuth_step_deg", 45.0),
            5.0,
            180.0,
        )
    )
    azimuths = np.arange(0.0, 360.0, step_deg, dtype=float)
    if azimuths.size == 0:
        azimuths = np.array([0.0], dtype=float)

    def azimuth_distance(azimuth_deg):
        angle = np.deg2rad(float(azimuth_deg))
        return abs(
            float(
                np.arctan2(
                    np.sin(angle - preferred_azimuth),
                    np.cos(angle - preferred_azimuth),
                )
            )
        )

    azimuths = np.array(
        sorted(azimuths.tolist(), key=azimuth_distance), dtype=float
    )
    T_camera_flange = np.linalg.inv(
        handeye_transform(args.calib_json, args.handeye_convention)
    )
    max_center_norm = max(
        float(getattr(args, "active_view_max_center_norm", 0.75)), 0.0
    )
    records = []
    for tilt_deg in _parse_active_center_optical_tilts(args):
        tilt_rad = float(np.deg2rad(tilt_deg))
        for azimuth_deg in azimuths:
            azimuth_rad = float(np.deg2rad(azimuth_deg))
            optical_direction = np.array(
                [
                    np.sin(tilt_rad) * np.cos(azimuth_rad),
                    np.sin(tilt_rad) * np.sin(azimuth_rad),
                    -np.cos(tilt_rad),
                ],
                dtype=float,
            )
            aim_point = camera_position + look_distance * optical_direction
            flange_candidates = lookat_candidates(
                camera_position,
                aim_point,
                T_camera_flange,
                current_flange_pose,
                roll_samples=int(getattr(args, "active_center_roll_samples", 72)),
            )
            if not flange_candidates:
                continue
            center_norm, target_depth = _candidate_pose_center_norm(
                flange_candidates[0]["T_base_camera"], target
            )
            if (
                not np.isfinite(center_norm)
                or target_depth <= 0.0
                or center_norm > max_center_norm
            ):
                continue
            records.append(
                {
                    "tilt_deg": float(tilt_deg),
                    "azimuth_deg": float(azimuth_deg),
                    "camera_position": camera_position.copy(),
                    "aim_point": aim_point,
                    "center_norm": float(center_norm),
                    "target_depth": float(target_depth),
                    "flange_candidates": flange_candidates,
                }
            )
    limit = max(
        int(getattr(args, "active_center_optical_max_directions", 24)), 1
    )
    return records[:limit]


def _parse_active_center_fallback_tilts(args):
    """Return bounded upper-hemisphere offsets for the center-view fallback."""
    raw = str(getattr(args, "active_center_fallback_tilts", "15,20,30,35"))
    values = []
    for item in raw.replace(",", " ").split():
        try:
            tilt = float(item)
        except ValueError:
            continue
        # Zero is the exact overhead pose already tried first.  Never let a
        # recovery view fall below the target horizon.
        if 1.0 <= tilt < 90.0 and all(abs(tilt - old) > 1e-6 for old in values):
            values.append(tilt)
    return tuple(sorted(values))


def _target_center_spherical_fallback_pose_records(
    args,
    target_center_base,
    current_flange_pose,
    current_camera_position,
    *,
    hooks,
):
    """Generate equal-radius tilted look-at alternatives to an unreachable top view.

    The active workflow historically required one exactly-overhead observation
    before it could inspect side views.  That makes a valid spherical camera
    shell unusable whenever the single zenith pose is outside Nero's oriented
    workspace.  These are *observation anchors*, not unguarded side moves:
    every flange roll still receives the same full IK and path validation as
    the normal center node before it can be executed.
    """
    handeye_transform = _require_hook(hooks, "handeye_transform")
    lookat_candidates = _require_hook(
        hooks, "look_at_camera_pose_flange_roll_candidates"
    )
    target = np.asarray(target_center_base, dtype=float).reshape(3)
    current_camera = np.asarray(current_camera_position, dtype=float).reshape(3)
    radius = max(
        float(getattr(args, "active_center_height", 0.30)),
        float(getattr(args, "active_orbit_min_radius", 0.30)),
    )
    # Retain the requested center-view radius whenever possible.  The orbit
    # minimum is only a safety lower bound, not a reason to move closer.
    radius = max(radius, 0.05)
    rel_xy = current_camera[:2] - target[:2]
    preferred_azimuth = (
        float(np.arctan2(rel_xy[1], rel_xy[0]))
        if float(np.linalg.norm(rel_xy)) > 1e-6
        else 0.0
    )
    step_deg = float(
        np.clip(
            getattr(args, "active_center_fallback_azimuth_step_deg", 45.0),
            5.0,
            180.0,
        )
    )
    azimuths = np.arange(0.0, 360.0, step_deg, dtype=float)
    if azimuths.size == 0:
        azimuths = np.array([0.0], dtype=float)
    # Start near the current viewing azimuth and interleave both directions,
    # avoiding a deterministic global-axis bias in the fallback choice.
    def azimuth_distance(azimuth_deg):
        angle = np.deg2rad(float(azimuth_deg))
        return abs(float(np.arctan2(np.sin(angle - preferred_azimuth), np.cos(angle - preferred_azimuth))))

    azimuths = np.array(sorted(azimuths.tolist(), key=azimuth_distance), dtype=float)
    T_camera_flange = np.linalg.inv(
        handeye_transform(args.calib_json, args.handeye_convention)
    )
    records = []
    for tilt_deg in _parse_active_center_fallback_tilts(args):
        tilt_rad = float(np.deg2rad(tilt_deg))
        for azimuth_deg in azimuths:
            azimuth_rad = float(np.deg2rad(azimuth_deg))
            cam_pos = target + radius * np.array(
                [
                    np.sin(tilt_rad) * np.cos(azimuth_rad),
                    np.sin(tilt_rad) * np.sin(azimuth_rad),
                    np.cos(tilt_rad),
                ],
                dtype=float,
            )
            flange_candidates = lookat_candidates(
                cam_pos,
                target,
                T_camera_flange,
                current_flange_pose,
                roll_samples=int(getattr(args, "active_center_roll_samples", 72)),
            )
            records.append(
                {
                    "tilt_deg": float(tilt_deg),
                    "azimuth_deg": float(azimuth_deg),
                    "camera_position": cam_pos,
                    "flange_candidates": flange_candidates,
                }
            )
    limit = max(int(getattr(args, "active_center_fallback_max_positions", 16)), 1)
    return records[:limit]


def _build_active_view_candidates(
    arm,
    args,
    target_cloud,
    scene_cloud,
    T_base_camera0,
    orbit_lateral_axis=None,
    orbit_geometry=None,
    target_center_override=None,
    obstacle_cloud=None,
    *,
    hooks,
):
    transform_cloud = _require_hook(hooks, "transform_cloud")
    _points_from_cloud = _require_hook(hooks, "points_from_cloud")
    handeye_transform = _require_hook(hooks, "handeye_transform")
    _estimate_normals_toward_camera = _require_hook(
        hooks, "estimate_normals_toward_camera"
    )
    _prepare_active_obstacle_cloud = _require_hook(
        hooks, "prepare_active_obstacle_cloud"
    )
    _active_motion_scene_points = _require_hook(
        hooks, "active_motion_scene_points"
    )
    _build_active_information_proxy = _require_hook(
        hooks, "build_active_information_proxy"
    )
    _line_segment_obstacle_metrics = _require_hook(
        hooks, "line_segment_obstacle_metrics"
    )
    _direction_context_penalty = _require_hook(
        hooks, "direction_context_penalty"
    )
    _nearest_obstacle_distance = _require_hook(
        hooks, "nearest_obstacle_distance"
    )
    _target_visibility_score = _require_hook(
        hooks, "target_visibility_score"
    )
    _solve_ik_for_flange_pose = _require_hook(
        hooks, "solve_ik_for_flange_pose"
    )
    _score_active_candidate_information = _require_hook(
        hooks, "score_active_candidate_information"
    )
    _build_scene_aware_nbv_records = _require_hook(
        hooks, "build_scene_aware_nbv_records"
    )
    _active_spherical_camera_position = _require_hook(
        hooks, "active_spherical_camera_position"
    )
    _parse_active_orbit_tilts = _require_hook(
        hooks, "parse_active_orbit_tilts"
    )
    _normalize_dir_xy = _require_hook(hooks, "normalize_dir_xy")
    _rank_active_lookat_roll_solutions = _require_hook(
        hooks, "rank_active_lookat_roll_solutions"
    )
    T_base_camera0 = np.asarray(T_base_camera0, dtype=float).reshape(4, 4)
    target_base = transform_cloud(target_cloud, T_base_camera0)
    scene_base = transform_cloud(scene_cloud, T_base_camera0)
    target_pts = _points_from_cloud(target_base)
    scene_pts = _points_from_cloud(scene_base)
    if target_pts.shape[0] == 0:
        raise RuntimeError("[active] target cloud is empty; cannot select active view")

    if target_center_override is None:
        target_center = np.median(target_pts, axis=0).astype(float)
    else:
        target_center = np.asarray(target_center_override, dtype=float).reshape(3)
    # The multiview gate and the candidate builder are independent entry
    # points.  In particular, --no-active-multiview-gate returns before the
    # gate computes its diagnostic shape profile, and a local variable from
    # that function cannot be reused here anyway.  Build the profile from the
    # exact target cloud/center being scored so every endpoint-valid candidate
    # can carry the elongation and bilateral-view metadata without depending
    # on whether the optional gate ran.
    target_shape_profile = _active_target_shape_profile(
        target_pts,
        target_center=target_center,
        target_label=getattr(args, "_active_target_label", None),
    )
    current_tcp = np.asarray(arm.get_tcp_pose(), dtype=float).reshape(6)
    current_flange = np.asarray(arm.get_flange_pose(), dtype=float).reshape(6)
    T_flange_camera = handeye_transform(args.calib_json, args.handeye_convention)
    cam0 = np.asarray(T_base_camera0[:3, 3], dtype=float).reshape(3)
    target_normals = _estimate_normals_toward_camera(target_pts, cam0)

    if obstacle_cloud is not None:
        obstacle_base = transform_cloud(obstacle_cloud, T_base_camera0)
        obstacle_source_pts = _points_from_cloud(obstacle_base)
        source_is_non_target_mask = True
    else:
        obstacle_source_pts = scene_pts
        source_is_non_target_mask = False
    obstacle_pts, obstacle_weights, obstacle_meta = _prepare_active_obstacle_cloud(
        obstacle_source_pts,
        target_pts,
        target_center,
        args,
        source_is_non_target_mask=source_is_non_target_mask,
    )
    path_guard = bool(getattr(args, "active_path_guard", True))
    current_joints = (
        np.asarray(arm.get_joint_angles(), dtype=float).reshape(7)
        if str(getattr(args, "motion_mode", "cartesian")) == "ik_joint"
        else None
    )
    path_scene_pts = np.zeros((0, 3), dtype=np.float32)
    if path_guard:
        if str(getattr(args, "motion_mode", "cartesian")) != "ik_joint":
            raise UnsafeMotionTarget(
                "full-arm active-view path guard requires --motion-mode ik_joint"
            )
        path_scene_pts = _active_motion_scene_points(target_pts, scene_pts, args)
    if bool(args.active_obstacle_aware) or bool(args.active_nbv_debug):
        print(
            "[active][obstacle] "
            f"raw={int(obstacle_meta.get('raw', scene_pts.shape[0]))} kept={int(obstacle_meta.get('kept', obstacle_pts.shape[0]))} "
            f"target_removed={int(obstacle_meta.get('target_removed', 0))} table_removed={int(obstacle_meta.get('table_removed', 0))} "
            f"mode={str(obstacle_meta.get('exclusion_mode', 'unknown'))} "
            f"target_z=[{float(obstacle_meta.get('target_z_lo', 0.0)):.3f}, {float(obstacle_meta.get('target_z_hi', 0.0)):.3f}] "
            f"height_keep_z={float(obstacle_meta.get('height_keep_z', 0.0)):.3f} "
            f"target_cloud_exclude={float(args.active_target_cloud_exclusion_dist):.3f} "
            f"fallback_center_exclude={float(args.active_target_exclusion_radius):.3f} "
            f"xy_radius={float(args.active_obstacle_xy_radius):.3f}"
        )

    information_proxy = _build_active_information_proxy(
        target_pts,
        target_normals,
        target_center,
        cam0,
        args,
    )
    if bool(args.active_obstacle_aware):
        current_los = _line_segment_obstacle_metrics(
            cam0,
            target_center,
            obstacle_pts,
            float(args.active_los_corridor_radius),
            weights=obstacle_weights,
        )
        current_direction_collision, _ = _direction_context_penalty(
            target_center,
            obstacle_pts,
            cam0[:2] - target_center[:2],
            weights=obstacle_weights,
        )
        information_proxy["current_los_penalty"] = float(current_los["penalty"])
        information_proxy["current_direction_collision"] = float(
            current_direction_collision
        )
    if bool(getattr(args, "active_nbv_debug", False)):
        grasp_proxy_text = ", ".join(
            f"{item['axis_name']}:{item['width_m'] * 1000.0:.1f}mm"
            for item in information_proxy.get("grasp_proxies", [])
        ) or "none"
        print(
            "[ActiveNBV][proxy] "
            f"surface_bins={len(information_proxy['bin_normals'])} "
            f"contact_source={information_proxy['contact_source']} "
            f"grasp_axes=[{grasp_proxy_text}] "
            f"sweep_mode={information_proxy['sweep_gain_mode']}"
        )

    candidates = []

    anchor_view = np.asarray(cam0 - target_center, dtype=float).reshape(3)
    anchor_view_norm = float(np.linalg.norm(anchor_view))

    def _anchor_disparity_deg(camera_position):
        candidate_view = (
            np.asarray(camera_position, dtype=float).reshape(3) - target_center
        )
        candidate_norm = float(np.linalg.norm(candidate_view))
        if anchor_view_norm < 1e-6 or candidate_norm < 1e-6:
            return float("inf")
        cosine = float(
            np.clip(
                np.dot(anchor_view, candidate_view)
                / max(anchor_view_norm * candidate_norm, 1e-12),
                -1.0,
                1.0,
            )
        )
        return float(np.rad2deg(np.arccos(cosine)))

    def _append_candidate(
        name,
        tcp,
        flange_pose,
        T_base_camera,
        baseline,
        command_type,
        angle_deg=0.0,
        planned_joint_target_override=None,
        roll_record=None,
    ):
        tcp = None if tcp is None else np.asarray(tcp, dtype=float).reshape(6)
        flange_pose = np.asarray(flange_pose, dtype=float).reshape(6)
        T_base_flange = _make_T_from_pose(flange_pose)
        if T_base_camera is None:
            T_base_camera = T_base_flange @ T_flange_camera
        else:
            T_base_camera = np.asarray(T_base_camera, dtype=float).reshape(4, 4)
        cam_pos = T_base_camera[:3, 3]
        anchor_disparity_deg = _anchor_disparity_deg(cam_pos)
        center_norm, depth = _candidate_pose_center_norm(T_base_camera, target_center)
        z_margin = float(flange_pose[2] - float(args.min_flange_z if args.min_flange_z is not None else 0.0))
        clearance = _nearest_obstacle_distance(cam_pos, obstacle_pts) if args.active_obstacle_aware else float("inf")
        cam_occ = 0.0 if (not args.active_obstacle_aware or not np.isfinite(clearance)) else float(np.clip((float(args.active_view_min_clearance) - clearance) / max(float(args.active_view_min_clearance), 1e-6), 0.0, 1.0))
        traj = (
            _line_segment_obstacle_metrics(cam0, cam_pos, obstacle_pts, float(args.active_traj_corridor_radius), weights=obstacle_weights)
            if args.active_obstacle_aware else {"count": 0, "ratio": 0.0, "nearest": float("inf"), "penalty": 0.0}
        )
        los = (
            _line_segment_obstacle_metrics(cam_pos, target_center, obstacle_pts, float(args.active_los_corridor_radius), weights=obstacle_weights)
            if args.active_obstacle_aware else {"count": 0, "ratio": 0.0, "nearest": float("inf"), "penalty": 0.0}
        )
        visibility = _target_visibility_score(target_pts, target_normals, cam_pos)
        angle_gain = min(abs(float(angle_deg)) / max(float(args.active_orbit_max_angle_deg), 1e-6), 1.0)
        center_bonus = float(np.clip(1.0 - center_norm / max(float(args.active_view_max_center_norm), 1e-6), 0.0, 1.0))
        clearance_bonus = 1.0 - cam_occ
        if args.active_obstacle_aware:
            view_dir_xy = np.asarray(cam_pos[:2] - target_center[:2], dtype=np.float32)
            direction_collision, direction_clearance = _direction_context_penalty(
                target_center,
                obstacle_pts,
                view_dir_xy,
                weights=obstacle_weights,
            )
        else:
            direction_collision, direction_clearance = 0.0, 1.0
        reject = []
        if depth <= float(args.min_depth) or depth >= float(args.max_depth):
            reject.append(f"target_depth={depth:.3f}")
        if center_norm > float(args.active_view_max_center_norm):
            reject.append(f"center_norm={center_norm:.3f}")
        min_anchor_disparity = max(
            float(getattr(args, "active_anchor_min_disparity_deg", 30.0)),
            0.0,
        )
        if anchor_disparity_deg < min_anchor_disparity:
            reject.append(
                f"anchor_disparity={anchor_disparity_deg:.1f}deg<"
                f"{min_anchor_disparity:.1f}deg"
            )
        if args.min_flange_z is not None and flange_pose[2] < float(args.min_flange_z):
            reject.append(f"flange_z={flange_pose[2]:.3f}")
        if args.active_obstacle_aware and np.isfinite(clearance) and clearance < float(args.active_view_min_clearance):
            reject.append(f"clearance={clearance:.3f}")
        if args.active_obstacle_aware and traj["penalty"] >= float(args.active_max_traj_penalty):
            reject.append(f"traj_occ={traj['penalty']:.3f}")
        if args.active_obstacle_aware and los["penalty"] >= float(args.active_max_los_penalty):
            reject.append(f"los_occ={los['penalty']:.3f}")

        planned_joint_target = (
            None
            if planned_joint_target_override is None
            else np.asarray(planned_joint_target_override, dtype=float).reshape(7)
        )
        joint_max_delta_deg = float("inf")
        joint_l2_delta_deg = float("inf")
        arm_path = {
            "nearest_clearance": float("nan"),
            "near_count": 0,
            "samples": 0,
            "primitive": "unchecked",
            "sample_index": -1,
        }
        # With up-and-over enabled, candidate generation only scores the view
        # geometry.  The selected candidates are then solved sequentially from
        # high waypoints; rejecting them here from a single direct IK seed would
        # incorrectly discard poses that are reachable through a safe elbow-up
        # transition.
        defer_to_up_and_over = bool(
            getattr(args, "active_up_and_over", False)
        )
        if (
            planned_joint_target is not None
            and current_joints is not None
            and not reject
        ):
            raw_delta = (
                np.asarray(planned_joint_target, dtype=float).reshape(7)
                - np.asarray(current_joints, dtype=float).reshape(7)
            )
            joint_max_delta_deg = float(np.rad2deg(np.max(np.abs(raw_delta))))
            joint_l2_delta_deg = float(np.rad2deg(np.linalg.norm(raw_delta)))
            max_joint = float(getattr(args, "active_side_max_joint_delta_deg", 45.0))
            max_l2 = float(getattr(args, "active_side_max_joint_l2_deg", 75.0))
            if joint_max_delta_deg > max_joint:
                reject.append(
                    f"joint_delta={joint_max_delta_deg:.1f}deg>{max_joint:.1f}deg"
                )
            if joint_l2_delta_deg > max_l2:
                reject.append(
                    f"joint_l2={joint_l2_delta_deg:.1f}deg>{max_l2:.1f}deg"
                )
            if not reject:
                arm_path["primitive"] = "guarded_direct_pending"
        elif path_guard and not reject and not defer_to_up_and_over:
            try:
                planned_joint_target = _solve_ik_for_flange_pose(
                    arm,
                    flange_pose,
                    args,
                    f"active_candidate_{name}",
                    current_joints_override=current_joints,
                    quiet=True,
                )
                raw_delta = (
                    np.asarray(planned_joint_target, dtype=float).reshape(7)
                    - np.asarray(current_joints, dtype=float).reshape(7)
                )
                joint_max_delta_deg = float(np.rad2deg(np.max(np.abs(raw_delta))))
                joint_l2_delta_deg = float(np.rad2deg(np.linalg.norm(raw_delta)))
                max_joint = float(getattr(args, "active_side_max_joint_delta_deg", 45.0))
                max_l2 = float(getattr(args, "active_side_max_joint_l2_deg", 75.0))
                if joint_max_delta_deg > max_joint:
                    reject.append(
                        f"joint_delta={joint_max_delta_deg:.1f}deg>{max_joint:.1f}deg"
                    )
                if joint_l2_delta_deg > max_l2:
                    reject.append(
                        f"joint_l2={joint_l2_delta_deg:.1f}deg>{max_l2:.1f}deg"
                    )
                if not reject:
                    # The selected direct candidate is checked below by
                    # plan_active_direct_route using the canonical segmented
                    # arm/camera/gripper model and the complete scene.  Do not
                    # reject it here with the legacy 90 mm spherical gripper
                    # envelope, which double-counts free space and can mark the
                    # measured start configuration itself as colliding.
                    arm_path["primitive"] = "guarded_direct_pending"
            except Exception as exc:
                planned_joint_target = None
                reject.append(f"arm_plan={type(exc).__name__}")
        elif path_guard and not reject and defer_to_up_and_over:
            arm_path["primitive"] = "up_and_over_pending"

        information = _score_active_candidate_information(
            information_proxy,
            cam_pos,
            los_penalty=float(los["penalty"]),
            direction_collision=float(direction_collision),
            center_norm=float(center_norm),
            depth=float(depth),
            joint_max_delta_deg=float(joint_max_delta_deg),
            joint_l2_delta_deg=float(joint_l2_delta_deg),
            trajectory_penalty=float(traj["penalty"]),
            camera_occupancy=float(cam_occ),
            args=args,
        )
        score = float(getattr(args, "active_nbv_score_weight", 1.0)) * float(
            information["score"]
        )
        if reject:
            score -= 10.0
        target_lateral_axis = target_shape_profile.get("lateral_axis_base", None)
        if target_lateral_axis is None:
            lateral_side_coordinate = 0.0
        else:
            lateral_side_coordinate = float(
                np.dot(
                    cam_pos - target_center,
                    np.asarray(target_lateral_axis, dtype=float).reshape(3),
                )
            )
        candidates.append({
            "name": name,
            "tcp": tcp,
            "flange": flange_pose,
            "T_base_camera": T_base_camera,
            "command_type": command_type,
            "orientation_mode": str(getattr(args, "active_orbit_orientation", "current")),
            "angle_deg": float(angle_deg),
            "score": float(score),
            "nbv_score": float(information["score"]),
            "nbv_detail": {
                key: value
                for key, value in information.items()
                if not key.endswith("_vector")
            },
            "visibility_vector": np.asarray(
                information["visibility_vector"], dtype=np.float32
            ),
            "center_norm": float(center_norm),
            "target_depth": float(depth),
            "baseline": float(baseline),
            "target_shape_elongation": float(
                target_shape_profile["elongation"]
            ),
            "target_bilateral_confidence": float(
                target_shape_profile["bilateral_confidence"]
            ),
            "target_round_confidence": float(
                target_shape_profile.get("round_confidence", 0.0)
            ),
            "target_long_axis_base": (
                None
                if target_shape_profile.get("long_axis_base", None) is None
                else np.asarray(
                    target_shape_profile["long_axis_base"], dtype=float
                ).reshape(3)
            ),
            "target_lateral_axis_base": (
                None
                if target_lateral_axis is None
                else np.asarray(target_lateral_axis, dtype=float).reshape(3)
            ),
            "target_lateral_side_coordinate": lateral_side_coordinate,
            "target_observation_radius": float(
                np.linalg.norm(cam_pos - target_center)
            ),
            "anchor_disparity_deg": float(anchor_disparity_deg),
            "clearance": float(clearance),
            "obstacle_aware": bool(args.active_obstacle_aware),
            "visibility": float(visibility),
            "angle_gain": float(angle_gain),
            "center_bonus": float(center_bonus),
            "cam_occ": float(cam_occ),
            "direction_collision": float(direction_collision),
            "direction_clearance": float(direction_clearance),
            "los_occ": float(los["penalty"]),
            "los_count": int(los["count"]),
            "los_nearest": float(los["nearest"]),
            "los_weighted_count": float(los.get("weighted_count", 0.0)),
            "traj_occ": float(traj["penalty"]),
            "traj_count": int(traj["count"]),
            "traj_nearest": float(traj["nearest"]),
            "traj_weighted_count": float(traj.get("weighted_count", 0.0)),
            "planned_joint_target": planned_joint_target,
            "joint_max_delta_deg": float(joint_max_delta_deg),
            "joint_l2_delta_deg": float(joint_l2_delta_deg),
            "arm_path_clearance": float(arm_path["nearest_clearance"]),
            "arm_path_near_count": int(arm_path["near_count"]),
            "arm_path_samples": int(arm_path["samples"]),
            "arm_path_primitive": str(arm_path["primitive"]),
            "camera_roll_rad": (
                float(roll_record.get("roll_rad", 0.0))
                if isinstance(roll_record, dict)
                else float("nan")
            ),
            "camera_roll_rotation_error_deg": (
                float(roll_record.get("rotation_error_deg", float("nan")))
                if isinstance(roll_record, dict)
                else float("nan")
            ),
            "joint_limit_margin_deg": (
                float(roll_record.get("joint_limit_margin_deg", float("nan")))
                if isinstance(roll_record, dict)
                else float("nan")
            ),
            "obstacle_meta": dict(obstacle_meta),
            "reason": "ok" if not reject else ",".join(reject),
            "current_tcp": current_tcp,
            "current_flange": current_flange,
            "target_center_base": target_center,
            "_information_proxy": information_proxy,
        })

    rel0 = cam0 - target_center
    radius = max(float(np.linalg.norm(rel0)), float(args.active_orbit_min_radius))
    if radius < 1e-4:
        raise RuntimeError("[active][orbit] current camera is too close to target center")
    max_angle = float(args.active_orbit_max_angle_deg)
    half_angle = max(5.0, 0.5 * max_angle)
    fixed_angles = []
    for deg in (-half_angle, half_angle, -max_angle, max_angle):
        deg = float(np.clip(deg, -max_angle, max_angle))
        if all(abs(deg - x) > 1e-6 for x in fixed_angles):
            fixed_angles.append(deg)
    T_camera_flange = np.linalg.inv(T_flange_camera)
    use_spherical_from_top = str(orbit_geometry or "").lower() == "spherical_from_top"
    if use_spherical_from_top and max_angle < 25.0:
        print(
            f"[active][warn] --active-orbit-max-angle-deg={max_angle:.1f}deg is small; "
            "side-view motion will be limited. Use 35-45deg for more obvious side observations."
        )
    candidate_specs = []
    if use_spherical_from_top:
        lateral_axis = np.asarray(orbit_lateral_axis if orbit_lateral_axis is not None else [1.0, 0.0, 0.0], dtype=float).reshape(3)
        lateral_axis[2] = 0.0
        lateral_norm = float(np.linalg.norm(lateral_axis))
        if lateral_norm < 1e-6:
            lateral_axis = np.array([1.0, 0.0, 0.0], dtype=float)
        else:
            lateral_axis = lateral_axis / lateral_norm

        if str(args.active_nbv_mode) == "fixed":
            for deg in fixed_angles:
                candidate_specs.append({
                    "name": f"orbit_{deg:+.1f}deg",
                    "deg": float(deg),
                    "dir_xy": lateral_axis[:2].astype(np.float32),
                    "nbv_score": 0.0,
                    "nbv_detail": {},
                    "view_angle_rad": float(np.arctan2(lateral_axis[1] * np.sign(deg or 1.0), lateral_axis[0] * np.sign(deg or 1.0))),
                })
        else:
            nbv_records = _build_scene_aware_nbv_records(
                target_pts,
                target_normals,
                obstacle_pts,
                target_center,
                args,
                preferred_dir_xy=lateral_axis[:2],
                obstacle_weights=obstacle_weights,
            )
            if len(nbv_records) == 0:
                print("[ActiveNBV][real][warn] no scene-aware NBV records; falling back to fixed orbit candidates")
                for deg in fixed_angles:
                    candidate_specs.append({
                        "name": f"orbit_{deg:+.1f}deg",
                        "deg": float(deg),
                        "dir_xy": lateral_axis[:2].astype(np.float32),
                        "nbv_score": 0.0,
                        "nbv_detail": {},
                        "view_angle_rad": float(np.arctan2(lateral_axis[1] * np.sign(deg or 1.0), lateral_axis[0] * np.sign(deg or 1.0))),
                    })
            else:
                orbit_degs = _parse_active_orbit_tilts(args)
                for rec in nbv_records:
                    for deg_abs in orbit_degs:
                        candidate_specs.append({
                            "name": f"{rec['name']}_orb{deg_abs:.0f}",
                            "deg": float(deg_abs),
                            "dir_xy": _normalize_dir_xy(rec["dir_xy"]),
                            "nbv_score": float(rec["score"]),
                            "nbv_detail": rec.get("detail", {}),
                            "view_angle_rad": float(rec["angle_rad"]),
                        })
    else:
        for deg in fixed_angles:
            candidate_specs.append({
                "name": f"orbit_{deg:+.1f}deg",
                "deg": float(deg),
                "dir_xy": None,
                "nbv_score": 0.0,
                "nbv_detail": {},
                "view_angle_rad": float(np.deg2rad(deg)),
            })

    if use_spherical_from_top and candidate_specs:
        for spec in candidate_specs:
            spec["cam_pos"] = _active_spherical_camera_position(
                target_center,
                radius,
                float(spec.get("view_angle_rad", 0.0)),
                float(spec["deg"]),
            )
            cam_pos = np.asarray(spec["cam_pos"], dtype=float).reshape(3)
            clearance = (
                _nearest_obstacle_distance(cam_pos, obstacle_pts)
                if bool(args.active_obstacle_aware)
                else float("inf")
            )
            camera_occupancy = (
                float(
                    np.clip(
                        (
                            float(args.active_view_min_clearance) - clearance
                        )
                        / max(float(args.active_view_min_clearance), 1e-6),
                        0.0,
                        1.0,
                    )
                )
                if bool(args.active_obstacle_aware) and np.isfinite(clearance)
                else 0.0
            )
            trajectory = (
                _line_segment_obstacle_metrics(
                    cam0,
                    cam_pos,
                    obstacle_pts,
                    float(args.active_traj_corridor_radius),
                    weights=obstacle_weights,
                )
                if bool(args.active_obstacle_aware)
                else {"penalty": 0.0}
            )
            line_of_sight = (
                _line_segment_obstacle_metrics(
                    cam_pos,
                    target_center,
                    obstacle_pts,
                    float(args.active_los_corridor_radius),
                    weights=obstacle_weights,
                )
                if bool(args.active_obstacle_aware)
                else {"penalty": 0.0}
            )
            direction_collision, _ = (
                _direction_context_penalty(
                    target_center,
                    obstacle_pts,
                    cam_pos[:2] - target_center[:2],
                    weights=obstacle_weights,
                )
                if bool(args.active_obstacle_aware)
                else (0.0, 1.0)
            )
            preliminary = _score_active_candidate_information(
                information_proxy,
                cam_pos,
                los_penalty=float(line_of_sight["penalty"]),
                direction_collision=float(direction_collision),
                center_norm=0.0,
                depth=float(radius),
                joint_max_delta_deg=0.0,
                joint_l2_delta_deg=0.0,
                trajectory_penalty=float(trajectory["penalty"]),
                camera_occupancy=float(camera_occupancy),
                args=args,
            )
            preliminary_reject = []
            anchor_disparity_deg = _anchor_disparity_deg(cam_pos)
            min_anchor_disparity = max(
                float(getattr(args, "active_anchor_min_disparity_deg", 30.0)),
                0.0,
            )
            if anchor_disparity_deg < min_anchor_disparity:
                preliminary_reject.append(
                    f"anchor_disparity={anchor_disparity_deg:.1f}deg<"
                    f"{min_anchor_disparity:.1f}deg"
                )
            if (
                bool(args.active_obstacle_aware)
                and np.isfinite(clearance)
                and clearance < float(args.active_view_min_clearance)
            ):
                preliminary_reject.append(f"clearance={clearance:.3f}")
            if (
                bool(args.active_obstacle_aware)
                and float(trajectory["penalty"]) >= float(args.active_max_traj_penalty)
            ):
                preliminary_reject.append(
                    f"traj_occ={float(trajectory['penalty']):.3f}"
                )
            if (
                bool(args.active_obstacle_aware)
                and float(line_of_sight["penalty"]) >= float(args.active_max_los_penalty)
            ):
                preliminary_reject.append(
                    f"los_occ={float(line_of_sight['penalty']):.3f}"
                )
            spec["preliminary_score"] = float(preliminary["score"])
            spec["preliminary_reject"] = preliminary_reject
            spec["anchor_disparity_deg"] = float(anchor_disparity_deg)

        # Rank all 16 x tilt positions cheaply, while front-loading several
        # candidates from every tilt layer so 45 degrees cannot crowd out all
        # safer, shallower alternatives before IK.
        score_order = sorted(
            candidate_specs,
            key=lambda item: float(item.get("preliminary_score", -float("inf"))),
            reverse=True,
        )
        per_tilt = max(
            int(getattr(args, "active_nbv_shortlist_per_tilt", 3)),
            1,
        )
        ordered_specs = []
        for tilt in _parse_active_orbit_tilts(args):
            layer = [
                item
                for item in score_order
                if abs(float(item["deg"]) - float(tilt)) <= 1e-6
            ]
            ordered_specs.extend(layer[:per_tilt])
        seen = {id(item) for item in ordered_specs}
        ordered_specs.extend(item for item in score_order if id(item) not in seen)
        candidate_specs = ordered_specs
        print(
            "[ActiveNBV][grid] "
            f"fixed_radius={radius:.3f}m azimuth_step="
            f"{float(getattr(args, 'active_nbv_azimuth_step_deg', 22.5)):.1f}deg "
            f"tilts={_parse_active_orbit_tilts(args)} "
            f"positions={len(candidate_specs)} "
            f"path_guard={'on' if path_guard else 'off'}; "
            "ranking=observation_gain_only "
            f"motion_weight="
            f"{float(getattr(args, 'active_nbv_motion_weight', 0.0)):.3f}; "
            "information score never overrides hard candidate checks"
        )

    initial_plan_budget = min(
        max(
            int(getattr(args, "active_nbv_plan_candidate_num", 32)),
            1,
        ),
        len(candidate_specs),
    )
    fast_plan_budget = min(
        max(
            int(getattr(args, "active_nbv_fast_plan_candidates", 8)),
            1,
        ),
        len(candidate_specs),
    )
    fast_roll_budget = max(
        int(getattr(args, "active_nbv_fast_roll_candidates", 2)), 1
    )
    fast_seed_budget = max(
        int(getattr(args, "active_nbv_fast_ik_seed_limit", 2)), 1
    )
    desired_valid_viewpoints = max(
        int(getattr(args, "active_nbv_min_valid_candidates", 8)),
        int(getattr(args, "active_side_view_count", 1)) * 2,
    )
    # Count distinct camera positions, not equivalent rolls and not the same
    # position revisited by the complete IK pass.  NBV selection needs a pool
    # of executable high-information positions; the old first-hit condition
    # stopped as soon as one easy view was found and biased observations toward
    # the same low-motion side of the robot.
    valid_viewpoint_indices = set()
    recorded_preliminary_rejects = set()
    if use_spherical_from_top:
        # The fast pass cheaply surveys the highest-information positions.
        # The bounded full pass retries fast false-negatives and then expands
        # farther down the information ranking.  Fast successes are cached and
        # never solved a second time.
        planning_specs = [
            (spec_index, spec, True)
            for spec_index, spec in enumerate(
                candidate_specs[:fast_plan_budget]
            )
        ]
        planning_specs.extend(
            (spec_index, spec, False)
            for spec_index, spec in enumerate(
                candidate_specs[:initial_plan_budget]
            )
        )
    else:
        planning_specs = [
            (spec_index, spec, False)
            for spec_index, spec in enumerate(candidate_specs)
        ]
    for spec_index, spec, fast_phase in planning_specs:
        if use_spherical_from_top and not fast_phase:
            if spec_index in valid_viewpoint_indices:
                continue
            if len(valid_viewpoint_indices) >= desired_valid_viewpoints:
                print(
                    "[ActiveNBV][progressive] executable information pool "
                    f"ready: distinct_positions="
                    f"{len(valid_viewpoint_indices)}/"
                    f"{desired_valid_viewpoints}; stop bounded IK expansion"
                )
                break
        deg = float(spec["deg"])
        if use_spherical_from_top:
            cam_pos = np.asarray(
                spec.get(
                    "cam_pos",
                    _active_spherical_camera_position(
                        target_center,
                        radius,
                        float(spec.get("view_angle_rad", 0.0)),
                        deg,
                    ),
                ),
                dtype=float,
            ).reshape(3)
        else:
            Rz = _rot_z(np.deg2rad(deg))
            cam_pos = target_center + Rz @ rel0
            if args.active_orbit_lift != 0.0:
                cam_pos[2] += float(args.active_orbit_lift)
        if use_spherical_from_top and spec.get("preliminary_reject"):
            if spec_index not in recorded_preliminary_rejects:
                candidates.append(
                    {
                        "name": str(spec.get("name", f"orbit_{deg:+.1f}deg")),
                        "command_type": "flange",
                        "score": float(spec.get("preliminary_score", 0.0)) - 10.0,
                        "angle_deg": deg,
                        "view_angle_rad": float(spec.get("view_angle_rad", 0.0)),
                        "reason": ",".join(spec["preliminary_reject"]),
                    }
                )
                recorded_preliminary_rejects.add(spec_index)
            if (
                fast_phase
                and (spec_index + 1) >= fast_plan_budget
            ):
                valid_count = len(valid_viewpoint_indices)
                if valid_count >= desired_valid_viewpoints:
                    print(
                        "[ActiveNBV][progressive] fast pass filled the "
                        f"executable information pool: {valid_count}/"
                        f"{desired_valid_viewpoints}; skip full IK expansion"
                    )
                    break
                print(
                    "[ActiveNBV][progressive] fast pass produced only "
                    f"{valid_count}/{desired_valid_viewpoints} distinct "
                    "executable positions; start bounded full IK expansion"
                )
            continue
        try:
            if use_spherical_from_top:
                selective_rescue = bool(
                    getattr(args, "active_nbv_selective_ik_rescue", False)
                )
                rescue_position_budget = max(
                    int(
                        getattr(
                            args,
                            "active_nbv_selective_ik_rescue_candidates",
                            4,
                        )
                    ),
                    0,
                )
                roll_solutions = _rank_active_lookat_roll_solutions(
                    arm,
                    args,
                    cam_pos,
                    target_center,
                    T_camera_flange,
                    current_flange,
                    current_joints,
                    f"active_candidate_{spec.get('name', 'sphere')}",
                    roll_plan_limit=(
                        fast_roll_budget if fast_phase else None
                    ),
                    quick_seed_limit=(
                        fast_seed_budget if fast_phase else None
                    ),
                    allow_full_retry=(
                        not fast_phase
                        and (
                            not selective_rescue
                            or spec_index < rescue_position_budget
                        )
                    ),
                    hooks=hooks,
                )
                keep_rolls = max(
                    int(getattr(args, "active_side_roll_keep_candidates", 2)),
                    1,
                )
                if fast_phase:
                    keep_rolls = min(keep_rolls, fast_roll_budget)
                spec_valid = False
                baseline = float(np.linalg.norm((cam_pos - cam0)[:2]))
                for roll_index, roll_record in enumerate(roll_solutions[:keep_rolls]):
                    roll_suffix = (
                        ""
                        if roll_index == 0
                        else f"_rollfb{roll_index}"
                    )
                    _append_candidate(
                        name=(
                            str(spec.get("name", f"orbit_{deg:+.1f}deg"))
                            + roll_suffix
                        ),
                        tcp=None,
                        flange_pose=roll_record["flange_pose"],
                        T_base_camera=roll_record["T_base_camera"],
                        baseline=baseline,
                        command_type="flange",
                        angle_deg=deg,
                        planned_joint_target_override=roll_record.get(
                            "planned_joint_target"
                        ),
                        roll_record=roll_record,
                    )
                    candidates[-1]["view_angle_rad"] = float(
                        spec.get("view_angle_rad", np.deg2rad(deg))
                    )
                    candidates[-1]["orientation_mode"] = "lookat_centered"
                    spec_valid = spec_valid or candidates[-1]["reason"] == "ok"
                if spec_valid:
                    valid_viewpoint_indices.add(spec_index)
                if (
                    fast_phase
                    and (spec_index + 1) >= fast_plan_budget
                ):
                    valid_count = len(valid_viewpoint_indices)
                    if valid_count >= desired_valid_viewpoints:
                        print(
                            "[ActiveNBV][progressive] fast pass filled the "
                            f"executable information pool: {valid_count}/"
                            f"{desired_valid_viewpoints}; skip full IK expansion"
                        )
                        break
                    print(
                        "[ActiveNBV][progressive] fast pass produced only "
                        f"{valid_count}/{desired_valid_viewpoints} distinct "
                        "executable positions; start bounded full IK expansion"
                    )
                elif (
                    not fast_phase
                    and len(valid_viewpoint_indices)
                    >= desired_valid_viewpoints
                ):
                    print(
                        "[ActiveNBV][progressive] executable information pool "
                        f"ready after full checks: "
                        f"{len(valid_viewpoint_indices)}/"
                        f"{desired_valid_viewpoints}"
                    )
                    break
                continue
            elif args.active_orbit_orientation == "lookat":
                T_base_camera = _look_at_camera_pose_minimize_flange_rotation(
                    cam_pos,
                    target_center,
                    T_camera_flange,
                    current_flange,
                )
                T_base_flange = T_base_camera @ T_camera_flange
                flange_pose = _as_pose_list(T_base_flange)
            elif args.active_orbit_orientation == "limited_lookat":
                T_base_camera_lookat = _look_at_camera_pose_minimize_flange_rotation(
                    cam_pos,
                    target_center,
                    T_camera_flange,
                    current_flange,
                )
                T_base_flange_lookat = T_base_camera_lookat @ T_camera_flange
                flange_lookat = _as_pose_list(T_base_flange_lookat)
                flange_pose = _limit_flange_orientation(
                    current_flange,
                    flange_lookat,
                    args.active_orbit_max_rpy_delta_deg,
                )
                T_base_flange = _make_T_from_pose(flange_pose)
                T_base_camera = T_base_flange @ T_flange_camera
            else:
                flange_pose = current_flange.copy()
                T_base_flange0 = _make_T_from_pose(current_flange)
                cam_from_flange = T_base_camera0[:3, 3] - T_base_flange0[:3, 3]
                flange_pose[:3] = cam_pos - cam_from_flange
                T_base_flange = _make_T_from_pose(flange_pose)
                T_base_camera = T_base_flange @ T_flange_camera
        except Exception as exc:
            candidates.append(
                {
                    "name": str(spec.get("name", f"orbit_{deg:+.1f}deg")),
                    "command_type": "flange",
                    "score": float(spec.get("preliminary_score", 0.0)) - 10.0,
                    "angle_deg": deg,
                    "view_angle_rad": float(
                        spec.get("view_angle_rad", np.deg2rad(deg))
                    ),
                    "reason": (
                        f"orbit_pose:{type(exc).__name__}:{str(exc)[:240]}"
                    ),
                }
            )
            if (
                fast_phase
                and (spec_index + 1) >= fast_plan_budget
            ):
                valid_count = len(valid_viewpoint_indices)
                if valid_count >= desired_valid_viewpoints:
                    print(
                        "[ActiveNBV][progressive] fast pass filled the "
                        f"executable information pool: {valid_count}/"
                        f"{desired_valid_viewpoints}; skip full IK expansion"
                    )
                    break
                print(
                    "[ActiveNBV][progressive] fast pass produced only "
                    f"{valid_count}/{desired_valid_viewpoints} distinct "
                    "executable positions; start bounded full IK expansion"
                )
            continue
        baseline = float(np.linalg.norm((cam_pos - cam0)[:2]))
        _append_candidate(
            name=str(spec.get("name", f"orbit_{deg:+.1f}deg")),
            tcp=None,
            flange_pose=flange_pose,
            T_base_camera=T_base_camera,
            baseline=baseline,
            command_type="flange",
            angle_deg=deg,
        )
        candidates[-1]["view_angle_rad"] = float(spec.get("view_angle_rad", np.deg2rad(deg)))
    if use_spherical_from_top:
        print(
            "[ActiveNBV][pool] distinct_endpoint_valid="
            f"{len(valid_viewpoint_indices)}/"
            f"{desired_valid_viewpoints} checked_position_budget="
            f"{initial_plan_budget} fast_budget={fast_plan_budget}"
        )
    candidates.sort(key=lambda item: float(item.get("score", -float("inf"))), reverse=True)
    return candidates


def _format_active_view_candidate(cand, prefix=""):
    tcp = cand.get("tcp", None)
    tcp_text = None if tcp is None else np.round(np.asarray(tcp, dtype=float).reshape(6)[:3], 4).tolist()
    flange = cand.get("flange", None)
    flange_text = None if flange is None else np.round(np.asarray(flange, dtype=float).reshape(6)[:3], 4).tolist()
    nbv = cand.get("nbv_detail", {}) if isinstance(cand.get("nbv_detail", {}), dict) else {}
    return (
        f"{prefix}{cand.get('name')} score={float(cand.get('score', -float('inf'))):.4f} "
        f"nbv={float(cand.get('nbv_score', 0.0)):.4f} "
        f"mode={cand.get('command_type', 'tcp')} orient={cand.get('orientation_mode', 'unknown')} "
        f"angle={float(cand.get('angle_deg', 0.0)):.1f}deg "
        f"obs={'on' if cand.get('obstacle_aware', False) else 'off'} "
        f"baseline={float(cand.get('baseline', 0.0)) * 1000.0:.1f}mm "
        f"anchor_sep={float(cand.get('anchor_disparity_deg', float('nan'))):.1f}deg "
        f"center_norm={float(cand.get('center_norm', float('inf'))):.3f} "
        f"depth={float(cand.get('target_depth', float('nan'))):.3f} "
        f"vis={float(cand.get('visibility', 0.0)):.3f} "
        f"Snew={float(nbv.get('surface_gain', 0.0)):.3f} "
        f"Gcontact={float(nbv.get('contact_gain', 0.0)):.3f} "
        f"Wsweep={float(nbv.get('sweep_gain', 0.0)):.3f}"
        f"({nbv.get('sweep_gain_mode', 'NA')}) "
        f"Osep={float(nbv.get('occlusion_gain', 0.0)):.3f} "
        f"Qimg={float(nbv.get('image_quality', 0.0)):.3f} "
        f"R={float(nbv.get('redundancy', 0.0)):.3f} "
        f"Mdiag={float(nbv.get('motion_cost', 0.0)):.3f} "
        f"dircol={float(cand.get('direction_collision', 0.0)):.3f} "
        f"los={float(cand.get('los_occ', 0.0)):.3f}/{int(cand.get('los_count', 0))} "
        f"traj={float(cand.get('traj_occ', 0.0)):.3f}/{int(cand.get('traj_count', 0))} "
        f"roll={np.rad2deg(float(cand.get('camera_roll_rad', float('nan')))):.1f}deg "
        f"jmax={float(cand.get('joint_max_delta_deg', float('nan'))):.1f}deg "
        f"jl2={float(cand.get('joint_l2_delta_deg', float('nan'))):.1f}deg "
        f"armclr={float(cand.get('arm_path_clearance', float('nan'))) * 1000.0:.1f}mm@"
        f"{cand.get('arm_path_primitive', 'unchecked')} "
        f"clearance={float(cand.get('clearance', float('inf'))):.3f} "
        f"tcp_xyz={tcp_text} flange_xyz={flange_text} reason={cand.get('reason')}"
    )


def _print_active_view_candidates(candidates, args=None):
    ok = [c for c in candidates if str(c.get("reason", "")) == "ok" and np.isfinite(float(c.get("score", -float("inf"))))]
    angles = sorted({round(float(c.get("angle_deg", 0.0)), 3) for c in ok})
    baselines = [float(c.get("baseline", 0.0)) * 1000.0 for c in ok]
    if baselines:
        baseline_text = f"baseline_range=[{min(baselines):.1f}, {max(baselines):.1f}]mm"
    else:
        baseline_text = "baseline_range=[]"
    print(
        f"[active] equal-radius orbit candidates: total={len(candidates)} ok={len(ok)} "
        f"angles_deg={angles} {baseline_text}"
    )
    if args is None or not bool(getattr(args, "active_nbv_debug", False)):
        print("[active] full candidate list hidden; add --active-nbv-debug to print every candidate.")
        return
    print("[active] full equal-radius orbit candidate views:")
    for i, cand in enumerate(candidates):
        print(_format_active_view_candidate(cand, prefix=f"  [{i}] "))


def _active_view_candidate_angle(cand):
    return float(cand.get("view_angle_rad", np.deg2rad(float(cand.get("angle_deg", 0.0)))))


def _active_target_shape_profile(
    target_points,
    target_center=None,
    target_label=None,
):
    """Estimate whether target-centered observations need opposite side views."""
    points = np.asarray(target_points, dtype=float).reshape(-1, 3)
    points = points[np.isfinite(points).all(axis=1)]
    result = {
        "elongation": 1.0,
        "bilateral_confidence": 0.0,
        "round_confidence": 0.0,
        "long_axis_base": None,
        "lateral_axis_base": None,
        "target_center_base": None,
    }
    if points.shape[0] < 16:
        return result
    center = (
        np.median(points, axis=0)
        if target_center is None
        else np.asarray(target_center, dtype=float).reshape(3)
    )
    centered = points - center.reshape(1, 3)
    radial = np.linalg.norm(centered, axis=1)
    robust = centered[radial <= float(np.quantile(radial, 0.995))]
    if robust.shape[0] < 16:
        robust = centered
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(np.cov(robust, rowvar=False))
    except (ValueError, np.linalg.LinAlgError):
        return result
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(np.asarray(eigenvalues)[order], 0.0)
    long_axis = np.asarray(eigenvectors[:, order[0]], dtype=float).reshape(3)
    long_axis /= np.linalg.norm(long_axis) + 1e-12
    elongation = float(
        np.sqrt(eigenvalues[0] / max(float(eigenvalues[1]), 1e-12))
    )
    confidence = float(np.clip((elongation - 1.4) / 0.3, 0.0, 1.0))
    round_confidence = float(np.clip((1.40 - elongation) / 0.25, 0.0, 1.0))
    normalized_label = str(target_label or "").strip().lower().replace("_", " ")
    if normalized_label == "tomato":
        # A partially occluded tomato can expose an elongated cap even though
        # the physical target is round.  Keep this as a prior floor rather
        # than forcing certainty so the measured geometry remains visible in
        # diagnostics.
        round_confidence = max(round_confidence, 0.80)
    # For tabletop targets, the useful two-sided distinction is around the
    # horizontal long axis.  A nearly vertical PCA axis is not treated as a
    # reliable left/right instruction.
    long_horizontal = long_axis.copy()
    long_horizontal[2] = 0.0
    horizontal_norm = float(np.linalg.norm(long_horizontal))
    if horizontal_norm < 0.35:
        confidence = 0.0
        lateral_axis = None
    else:
        long_horizontal /= horizontal_norm
        lateral_axis = np.cross(
            np.array([0.0, 0.0, 1.0], dtype=float), long_horizontal
        )
        lateral_axis /= np.linalg.norm(lateral_axis) + 1e-12
    result.update(
        {
            "elongation": elongation,
            "bilateral_confidence": confidence,
            "round_confidence": round_confidence,
            "long_axis_base": long_horizontal if horizontal_norm >= 0.35 else long_axis,
            "lateral_axis_base": lateral_axis,
            "target_center_base": center,
        }
    )
    return result


def _active_side_selection_order(valid, args, *, hooks):
    _score_active_candidate_information = _require_hook(
        hooks, "score_active_candidate_information"
    )
    _smallest_angle_diff_deg = _require_hook(
        hooks, "smallest_angle_diff_deg"
    )
    valid = list(valid)
    if len(valid) <= 1 or int(args.active_side_view_count) <= 1:
        return valid

    round_confidence = float(
        np.clip(valid[0].get("target_round_confidence", 0.0), 0.0, 1.0)
    )
    if (
        bool(getattr(args, "active_round_view_policy", True))
        and round_confidence > 0.0
        and len(valid) >= 2
    ):
        preferred_separation = max(
            float(
                getattr(
                    args,
                    "active_round_pair_min_separation_deg",
                    70.0,
                )
            ),
            1.0,
        )
        pair_weight = float(
            np.clip(
                getattr(args, "active_round_pair_weight", 0.35),
                0.0,
                0.75,
            )
        ) * round_confidence

        def _pair_value(first_index, second_index):
            first_candidate = valid[first_index]
            second_candidate = valid[second_index]
            separation = _smallest_angle_diff_deg(
                _active_view_candidate_angle(first_candidate),
                _active_view_candidate_angle(second_candidate),
            )
            separation_score = float(
                np.clip(separation / preferred_separation, 0.0, 1.0)
            )
            individual = 0.5 * (
                float(first_candidate.get("score", 0.0))
                + float(second_candidate.get("score", 0.0))
            )
            visibility_first = first_candidate.get("visibility_vector")
            visibility_second = second_candidate.get("visibility_vector")
            union_coverage = individual
            if visibility_first is not None and visibility_second is not None:
                a = np.asarray(visibility_first, dtype=float).reshape(-1)
                b = np.asarray(visibility_second, dtype=float).reshape(-1)
                if a.size > 0 and a.size == b.size:
                    finite = np.isfinite(a) & np.isfinite(b)
                    if np.any(finite):
                        union_coverage = float(
                            np.mean(np.maximum(a[finite], b[finite]))
                        )
            joint_coverage = 0.70 * union_coverage + 0.30 * separation_score
            score = (1.0 - pair_weight) * individual + pair_weight * joint_coverage
            return score, separation, union_coverage

        pairs = []
        for first_index in range(len(valid) - 1):
            for second_index in range(first_index + 1, len(valid)):
                value, separation, union_coverage = _pair_value(
                    first_index, second_index
                )
                pairs.append(
                    (value, separation, union_coverage, first_index, second_index)
                )
        separated = [
            pair for pair in pairs if pair[1] >= preferred_separation - 1e-9
        ]
        pool = separated if separated else pairs
        best_pair = max(pool, key=lambda item: (item[0], item[1]))
        selected = [valid[best_pair[3]], valid[best_pair[4]]]
        selected.sort(
            key=lambda item: float(item.get("score", 0.0)), reverse=True
        )
        selected_ids = {id(item) for item in selected}
        remaining_round = [item for item in valid if id(item) not in selected_ids]
        remaining_round.sort(
            key=lambda item: float(item.get("score", -float("inf"))),
            reverse=True,
        )
        print(
            "[active][round] selected joint-coverage side-view pair: "
            f"round_confidence={round_confidence:.2f} "
            f"separation={best_pair[1]:.1f}deg "
            f"preferred>={preferred_separation:.1f}deg "
            f"union_visibility={best_pair[2]:.3f} "
            f"pair_score={best_pair[0]:.3f}"
        )
        return selected + remaining_round

    first = valid[0]
    first_angle = _active_view_candidate_angle(first)
    min_sep = float(args.active_preferred_side_separation_deg)
    remaining = valid[1:]
    shape_confidence = float(
        np.clip(first.get("target_bilateral_confidence", 0.0), 0.0, 1.0)
    )
    first_side = float(first.get("target_lateral_side_coordinate", 0.0))

    def _score_second(cand):
        ang = _active_view_candidate_angle(cand)
        sep = _smallest_angle_diff_deg(ang, first_angle)
        opposite_bonus = min(sep / max(min_sep, 1e-6), 1.0)
        conditional_score = float(cand.get("score", -float("inf")))
        proxy = cand.get("_information_proxy")
        first_visibility = first.get("visibility_vector")
        T_base_camera = cand.get("T_base_camera")
        if (
            isinstance(proxy, dict)
            and first_visibility is not None
            and T_base_camera is not None
        ):
            conditional = _score_active_candidate_information(
                proxy,
                np.asarray(T_base_camera, dtype=float).reshape(4, 4)[:3, 3],
                los_penalty=float(cand.get("los_occ", 0.0)),
                direction_collision=float(cand.get("direction_collision", 0.0)),
                center_norm=float(cand.get("center_norm", 0.0)),
                depth=float(cand.get("target_depth", proxy.get("reference_depth", 0.30))),
                joint_max_delta_deg=float(cand.get("joint_max_delta_deg", 0.0)),
                joint_l2_delta_deg=float(cand.get("joint_l2_delta_deg", 0.0)),
                trajectory_penalty=float(cand.get("traj_occ", 0.0)),
                camera_occupancy=float(cand.get("cam_occ", 0.0)),
                extra_visibility=first_visibility,
                args=args,
            )
            conditional_score = float(conditional["score"])
            cand["conditional_second_score"] = conditional_score
            cand["conditional_second_detail"] = {
                key: value
                for key, value in conditional.items()
                if not key.endswith("_vector")
            }
        angular_weight = (
            float(
                np.clip(
                    getattr(args, "active_opposite_bonus_weight", 0.08),
                    0.0,
                    1.0,
                )
            )
            if bool(getattr(args, "active_prefer_opposite_sides", True))
            else 0.0
        )
        candidate_side = float(
            cand.get("target_lateral_side_coordinate", 0.0)
        )
        opposite_lateral = bool(first_side * candidate_side < 0.0)
        first_radius = max(
            float(first.get("target_observation_radius", 0.0)), 1e-6
        )
        side_strength = float(
            np.clip(
                min(abs(first_side), abs(candidate_side))
                / max(0.20 * first_radius, 1e-6),
                0.0,
                1.0,
            )
        )
        bilateral_score = side_strength if opposite_lateral else 0.0
        bilateral_weight = float(
            np.clip(
                getattr(args, "active_elongated_bilateral_weight", 0.25),
                0.0,
                0.5,
            )
        ) * shape_confidence
        # Round targets retain pure conditional information-gain ordering.
        # As elongation becomes reliable, a genuinely opposite contact-side
        # view receives priority; angular separation is only a small tiebreaker.
        separation_weight = angular_weight * shape_confidence
        total_extra = min(bilateral_weight + separation_weight, 0.75)
        combined_extra = (
            bilateral_weight * bilateral_score
            + separation_weight * opposite_bonus
        ) / max(total_extra, 1e-12) if total_extra > 0.0 else 0.0
        selection_score = (
            (1.0 - total_extra) * conditional_score
            + total_extra * combined_extra
        )
        cand["bilateral_opposite"] = opposite_lateral
        cand["bilateral_side_strength"] = side_strength
        cand["bilateral_selection_score"] = float(selection_score)
        return float(selection_score)

    remaining.sort(key=_score_second, reverse=True)
    ordered = [first] + remaining
    if remaining:
        sep = _smallest_angle_diff_deg(_active_view_candidate_angle(remaining[0]), first_angle)
        bilateral_text = (
            f"elongation={float(first.get('target_shape_elongation', 1.0)):.2f} "
            f"bilateral_confidence={shape_confidence:.2f} "
            f"opposite={bool(remaining[0].get('bilateral_opposite', False))}"
        )
        print(
            f"[active] second-view ranking recomputed conditional surface/contact/sweep "
            f"gain after the first selected view; {bilateral_text}; "
            f"first-second separation={sep:.1f}deg preferred={min_sep:.1f}deg"
        )
    return ordered


def _reverse_active_observation_plan(
    outbound_plan,
    args,
    label,
    *,
    modeled_tcp_pose_for_joints,
    urdf_chain_geometry,
):
    """Build the exact joint-path reverse of one validated observation plan.

    In a static observation workflow the route that safely reached the camera
    pose is also the strongest return certificate.  Re-solving IK or asking
    the controller for a new direct center edge can switch kinematic branches
    and create a different intermediate arm shape.  Each reversed stage below
    uses the constraints of the same outbound edge in the opposite direction
    and keeps the outbound plan's frozen safety scene.
    """
    outbound_stages = tuple(outbound_plan.before_close)
    if len(outbound_stages) == 0:
        raise PlanNotExecutable(
            f"{label} cannot reverse an empty outbound observation plan"
        )
    outbound_start = np.asarray(
        outbound_plan.start_joints, dtype=float
    ).reshape(7)
    reverse_start = np.asarray(
        outbound_stages[-1].joints, dtype=float
    ).reshape(7)
    reverse_stages = []
    for reverse_index, outbound_index in enumerate(
        range(len(outbound_stages) - 1, -1, -1),
        start=1,
    ):
        edge_constraints = outbound_stages[outbound_index]
        if outbound_index == 0:
            target_joints = outbound_start
            target_tcp = np.asarray(
                modeled_tcp_pose_for_joints(target_joints, args),
                dtype=float,
            ).reshape(6)
            _, target_flange_T = urdf_chain_geometry(target_joints, args)
            target_flange = np.asarray(
                _as_pose_list(target_flange_T), dtype=float
            ).reshape(6)
        else:
            target_stage = outbound_stages[outbound_index - 1]
            target_joints = np.asarray(
                target_stage.joints, dtype=float
            ).reshape(7)
            target_tcp = np.asarray(
                target_stage.tcp_pose, dtype=float
            ).reshape(6)
            target_flange = np.asarray(
                target_stage.flange_pose, dtype=float
            ).reshape(6)
        reverse_stages.append(
            PlannedJointStage(
                label=f"{label}_reverse_edge{reverse_index}",
                tcp_pose=target_tcp.copy(),
                flange_pose=target_flange.copy(),
                joints=target_joints.copy(),
                phase=str(edge_constraints.phase),
                min_tcp_z=edge_constraints.min_tcp_z,
                runtime_min_tcp_z_slack=float(
                    edge_constraints.runtime_min_tcp_z_slack
                ),
                require_support_release_latch_at_endpoint=bool(
                    edge_constraints.require_support_release_latch_at_endpoint
                ),
            )
        )
    return GraspMotionPlan(
        start_joints=reverse_start.copy(),
        before_close=tuple(reverse_stages),
        after_close=tuple(),
        after_release=tuple(),
        empty_recovery=tuple(),
        safety_scene=outbound_plan.safety_scene,
        attached_target_tcp=outbound_plan.attached_target_tcp,
        placed_target_base=outbound_plan.placed_target_base,
        initial_support_release=outbound_plan.initial_support_release,
        closed_gripper_width=outbound_plan.closed_gripper_width,
        validation_only=bool(outbound_plan.validation_only),
    )


def _single_view_active_result(
    target_cloud,
    scene_cloud,
    T_base_camera,
    reason,
    motion_sent=False,
    motion_environment_cloud=None,
    motion_environment_map=None,
    restore_ok=None,
):
    if restore_ok is None:
        restore_ok = not bool(motion_sent)
    return ActiveViewResult(
        target_cloud=target_cloud,
        scene_cloud=scene_cloud,
        T_base_camera=T_base_camera,
        used_multiview=False,
        kept_view_count=1,
        reason=str(reason),
        restore_ok=bool(restore_ok),
        motion_sent=bool(motion_sent),
        motion_scene_cloud=scene_cloud,
        motion_environment_cloud=motion_environment_cloud,
        motion_environment_map=motion_environment_map,
    )


def _active_center_joint_state_is_verified(
    arm,
    expected_center_joints,
    args,
    context,
):
    """Read-only check used to report whether a failed capture stayed at center."""
    if expected_center_joints is None:
        return False
    try:
        current_joints = np.asarray(
            arm.get_joint_angles(), dtype=float
        ).reshape(7)
        expected_joints = np.asarray(
            expected_center_joints, dtype=float
        ).reshape(7)
        error_deg = float(
            np.rad2deg(np.max(np.abs(current_joints - expected_joints)))
        )
        allowed_deg = float(
            getattr(args, "active_center_joint_match_deg", 2.0)
        )
        if error_deg > allowed_deg:
            print(
                f"[active][warn] {context} did not remain at the saved center "
                f"joints: max_error={error_deg:.2f}deg > {allowed_deg:.2f}deg"
            )
            return False
        print(
            f"[active] {context} remained at the saved center joints: "
            f"max_error={error_deg:.2f}deg"
        )
        return True
    except Exception as exc:
        print(
            f"[active][warn] could not verify center joints after {context}: "
            f"{type(exc).__name__}: {exc}"
        )
        return False


def run_optional_active_view(
    arm,
    args,
    target_cloud,
    scene_cloud,
    T_base_camera0,
    first_inst,
    out_dir,
    *,
    hooks,
):
    _single_view_active_result_impl = _require_hook(
        hooks, "single_view_active_result"
    )
    _should_use_active_multiview = _require_hook(
        hooks, "should_use_active_multiview"
    )
    _target_center_base_from_cloud = _require_hook(
        hooks, "target_center_base_from_cloud"
    )
    _target_center_overhead_flange_pose_candidates = _require_hook(
        hooks, "target_center_overhead_flange_pose_candidates"
    )
    _active_motion_scene_points = _require_hook(
        hooks, "active_motion_scene_points"
    )
    _build_active_view_candidates = _require_hook(
        hooks, "build_active_view_candidates"
    )
    _print_active_view_candidates = _require_hook(
        hooks, "print_active_view_candidates"
    )
    _format_active_view_candidate = _require_hook(
        hooks, "format_active_view_candidate"
    )
    _active_side_selection_order = _require_hook(
        hooks, "active_side_selection_order"
    )
    _active_view_second_capture_args = _require_hook(
        hooks, "active_view_second_capture_args"
    )
    _active_center_joint_state_is_verified = _require_hook(
        hooks, "active_center_joint_state_is_verified"
    )
    _active_motion_capture_base = _require_hook(
        hooks, "active_motion_capture_base"
    )
    _active_side_capture_overlap_ok = _require_hook(
        hooks, "active_side_capture_overlap_ok"
    )
    _modeled_tcp_pose_for_joints = _require_hook(
        hooks, "modeled_tcp_pose_for_joints"
    )
    _urdf_chain_geometry = _require_hook(hooks, "urdf_chain_geometry")
    _smallest_angle_diff_deg = _require_hook(
        hooks, "smallest_angle_diff_deg"
    )
    capture_active_projected_target_cloud = _require_hook(
        hooks, "capture_active_projected_target_cloud"
    )
    capture_target_cloud = _require_hook(hooks, "capture_target_cloud")
    confirm_motion = _require_hook(hooks, "confirm_motion")
    crop_scene_cloud_near_target = _require_hook(
        hooks, "crop_scene_cloud_near_target"
    )
    diagnose_active_fusion = _require_hook(hooks, "diagnose_active_fusion")
    execute_prevalidated_stages = _require_hook(
        hooks, "execute_prevalidated_stages"
    )
    merge_clouds = _require_hook(hooks, "merge_clouds")
    move_flange_pose = _require_hook(hooks, "move_flange_pose")
    move_tcp_pose = _require_hook(hooks, "move_tcp_pose")
    plan_active_direct_route = _require_hook(
        hooks, "plan_active_direct_route"
    )
    plan_active_up_and_over_route = _require_hook(
        hooks, "plan_active_up_and_over_route"
    )
    preview_fused_cloud = _require_hook(hooks, "preview_fused_cloud")
    register_active_fusion_views = _require_hook(
        hooks, "register_active_fusion_views"
    )
    save_cloud = _require_hook(hooks, "save_cloud")
    T_base_camera_from_flange = _require_hook(
        hooks, "T_base_camera_from_flange"
    )
    _points_from_cloud = _require_hook(hooks, "points_from_cloud")
    transform_cloud = _require_hook(hooks, "transform_cloud")
    initial_obstacle_cloud = getattr(args, "_last_obstacle_cloud", None)
    try:
        initial_capture_joints = np.asarray(
            arm.get_joint_angles(), dtype=float
        ).reshape(7).copy()
    except Exception:
        initial_capture_joints = None
    single_environment_cache = {
        "built": False,
        "map": None,
        "cloud": None,
    }

    def _single_view_active_result(
        single_target_cloud,
        single_scene_cloud,
        single_T_base_camera,
        reason,
        motion_sent=False,
        motion_environment_cloud=None,
        restore_ok=None,
    ):
        """Preserve the legacy result hook while attaching one map generation."""
        if not bool(single_environment_cache["built"]):
            single_environment_cache["built"] = True
            source_cloud = (
                initial_obstacle_cloud
                if initial_obstacle_cloud is not None
                else single_scene_cloud
            )
            try:
                if initial_capture_joints is None:
                    raise RuntimeError(
                        "initial capture joints are unavailable for self filtering"
                    )
                environment_base = _active_motion_capture_base(
                    source_cloud,
                    single_T_base_camera,
                    initial_capture_joints,
                    args,
                    "active_motion_environment_single",
                )
                target_base_single = transform_cloud(
                    single_target_cloud, single_T_base_camera
                )
                target_points_single = _points_from_cloud(target_base_single)
                if (
                    initial_obstacle_cloud is None
                    and target_points_single.shape[0] > 0
                ):
                    # Old/offline captures may not provide the exact non-target
                    # mask partition.  In that case remove only the same tight
                    # target NN band used by the legacy collision fallback;
                    # never promote the selected object itself to environment.
                    environment_base = _exclude_target_band_from_environment(
                        environment_base,
                        target_base_single,
                        args,
                    )
                T_camera_base = np.linalg.inv(
                    np.asarray(single_T_base_camera, dtype=float).reshape(4, 4)
                )
                (
                    single_environment_cache["map"],
                    single_environment_cache["cloud"],
                ) = _build_dynamic_motion_environment(
                    ["single"],
                    [environment_base],
                    args,
                    T_camera_base,
                    hooks=hooks,
                    target_base=target_points_single,
                    map_id="active_single_view_environment",
                    target_center_base=(
                        None
                        if target_points_single.shape[0] == 0
                        else np.median(target_points_single, axis=0)
                    ),
                )
                # A package version without the dynamic fuser still returns
                # the exact non-target capture expected by old consumers.
                if (
                    single_environment_cache["cloud"] is None
                    and initial_obstacle_cloud is not None
                ):
                    single_environment_cache["cloud"] = transform_cloud(
                        environment_base, T_camera_base
                    )
            except Exception as exc:
                print(
                    "[active][motion][map][warn] single-view environment map "
                    f"unavailable; keep legacy result: {type(exc).__name__}: {exc}"
                )

        effective_environment_cloud = (
            motion_environment_cloud
            if motion_environment_cloud is not None
            else single_environment_cache["cloud"]
        )
        result = _single_view_active_result_impl(
            single_target_cloud,
            single_scene_cloud,
            single_T_base_camera,
            reason,
            motion_sent=motion_sent,
            motion_environment_cloud=effective_environment_cloud,
            restore_ok=restore_ok,
        )
        environment_map = single_environment_cache["map"]
        if environment_map is not None and hasattr(
            result, "motion_environment_map"
        ):
            result = replace(result, motion_environment_map=environment_map)
        return result

    active_motion_sent = False
    if isinstance(first_inst, dict):
        args._active_target_label = str(
            first_inst.get("label", first_inst.get("class_name", ""))
        )
    if not args.active_view:
        return _single_view_active_result(target_cloud, scene_cloud, T_base_camera0, "active_view_disabled")
    if (
        bool(getattr(args, "execute", False))
        and (
            not bool(getattr(args, "active_path_guard", True))
            or not bool(getattr(args, "motion_runtime_collision_guard", True))
            or str(getattr(args, "motion_mode", "cartesian")) != "ik_joint"
        )
    ):
        print(
            "[active][abort] executable active-view motion requires "
            "--motion-mode ik_joint, --active-path-guard, and "
            "--motion-runtime-collision-guard; unguarded branches are dry-run "
            "diagnostics only"
        )
        return _single_view_active_result(
            target_cloud,
            scene_cloud,
            T_base_camera0,
            "active_guard_required",
        )
    force_active_multiview = bool(getattr(args, "_force_active_multiview", False))
    force_center_relocalize = bool(
        getattr(args, "_force_active_center_relocalize", False)
    )
    if force_active_multiview:
        print("[active] amodal occlusion requires multi-view; bypass proximity gate")
    elif force_center_relocalize:
        print(
            "[active] post-clear 3-D memory requires one target-centered "
            "YOLO relocation; bypass proximity gate and skip side views"
        )
    elif not _should_use_active_multiview(
        target_cloud,
        scene_cloud,
        T_base_camera0,
        args,
        obstacle_cloud=initial_obstacle_cloud,
    ):
        return _single_view_active_result(target_cloud, scene_cloud, T_base_camera0, "multiview_gate_not_triggered")
    if args.active_center_first:
        try:
            target_center0 = _target_center_base_from_cloud(target_cloud, T_base_camera0)
            first_cam = np.asarray(T_base_camera0[:3, 3], dtype=float).reshape(3)
            lateral_axis = first_cam - target_center0
            lateral_axis[2] = 0.0
            if np.linalg.norm(lateral_axis) < 1e-6:
                lateral_axis = np.array([1.0, 0.0, 0.0], dtype=float)
            lateral_axis = lateral_axis / (np.linalg.norm(lateral_axis) + 1e-9)
            center_pose_candidates = _target_center_overhead_flange_pose_candidates(
                args,
                target_center0,
                arm.get_flange_pose(),
            )
            if len(center_pose_candidates) == 0:
                raise PlanNotExecutable("no equivalent center look-at roll was generated")
            center_flange_pose = center_pose_candidates[0]["flange_pose"]
            center_T_base_camera_plan = center_pose_candidates[0]["T_base_camera"]
            print("[active] target_center_base =", np.round(target_center0, 6).tolist())
            print(
                f"[active] center-over-target camera height={float(args.active_center_height):.3f}m "
                f"orientation=look-at target roll_candidates={len(center_pose_candidates)}"
            )
        except Exception as exc:
            print(f"[active][warn] failed to build target-centered active view: {type(exc).__name__}: {exc}")
            if bool(getattr(args, "active_path_guard", True)):
                print(
                    "[active][abort] full-arm path guard forbids the legacy direct-orbit fallback."
                )
                return _single_view_active_result(
                    target_cloud,
                    scene_cloud,
                    T_base_camera0,
                    "active_center_plan_failed",
                )
            print("[active][warn] fall back to old orbit active-view logic.")
        else:
            if not args.execute:
                print("[active][dry-run] --execute not set; target-centered active-view motion skipped")
                return _single_view_active_result(target_cloud, scene_cloud, T_base_camera0, "execute_disabled")
            center_joint_target = None
            center_start_joints = None
            center_motion_plan = None
            center_joints = None
            center_observation_mode = "overhead"
            target_base0 = transform_cloud(target_cloud, T_base_camera0)
            scene_base0 = transform_cloud(scene_cloud, T_base_camera0)
            args._active_fusion_target_shape_profile = _active_target_shape_profile(
                _points_from_cloud(target_base0),
                target_center=target_center0,
                target_label=getattr(args, "_active_target_label", None),
            )
            if bool(getattr(args, "active_path_guard", True)):
                try:
                    if str(getattr(args, "motion_mode", "cartesian")) != "ik_joint":
                        raise UnsafeMotionTarget(
                            "full-arm active-view path guard requires --motion-mode ik_joint"
                        )
                    center_start_joints = np.asarray(
                        arm.get_joint_angles(), dtype=float
                    ).reshape(7)
                    # The initial observation can contain a few gripper-edge
                    # depth pixels just outside the 2-D silhouette mask.  They
                    # must be removed with the joints that produced this frame
                    # before the frozen cloud is used to plan center motion.
                    scene_base0 = _active_motion_capture_base(
                        scene_cloud,
                        T_base_camera0,
                        center_start_joints,
                        args,
                        "active_motion_scene_initial",
                    )
                    center_path_scene = _active_motion_scene_points(
                        _points_from_cloud(target_base0),
                        _points_from_cloud(scene_base0),
                        args,
                    )
                    if bool(getattr(args, "active_up_and_over", False)):
                        center_motion_plan = plan_active_up_and_over_route(
                            arm,
                            arm.get_tcp_pose(),
                            center_flange_pose,
                            center_start_joints,
                            center_path_scene,
                            _points_from_cloud(target_base0),
                            args,
                            "active_center_plan",
                            float(
                                getattr(
                                    args,
                                    "active_center_max_joint_delta_deg",
                                    45.0,
                                )
                            ),
                            float(
                                getattr(
                                    args,
                                    "active_center_max_joint_l2_deg",
                                    90.0,
                                )
                            ),
                        )
                        center_joint_target = np.asarray(
                            center_motion_plan.before_close[-1].joints,
                            dtype=float,
                        ).reshape(7)
                    else:
                        plan_limit = min(
                            len(center_pose_candidates),
                            max(
                                1,
                                int(
                                    getattr(
                                        args,
                                        "active_center_roll_plan_candidates",
                                        72,
                                    )
                                ),
                            ),
                        )
                        direct_failures = []
                        for roll_rank, candidate in enumerate(
                            center_pose_candidates[:plan_limit], start=1
                        ):
                            roll_deg = float(
                                np.rad2deg(float(candidate["roll_rad"]))
                            )
                            try:
                                candidate_plan = plan_active_direct_route(
                                    arm,
                                    candidate["flange_pose"],
                                    center_start_joints,
                                    center_path_scene,
                                    _points_from_cloud(target_base0),
                                    args,
                                    "active_center_plan",
                                    float(getattr(args, "active_center_max_joint_delta_deg", 45.0)),
                                    float(getattr(args, "active_center_max_joint_l2_deg", 90.0)),
                                )
                            except (PlanNotExecutable, UnsafeMotionTarget) as exc:
                                direct_failures.append(
                                    f"roll={roll_deg:.1f}deg:{type(exc).__name__}:{exc}"
                                )
                                print(
                                    f"[active][center-roll] reject rank={roll_rank}/{plan_limit} "
                                    f"roll={roll_deg:.1f}deg: {type(exc).__name__}: {exc}"
                                )
                                continue
                            center_motion_plan = candidate_plan
                            center_flange_pose = candidate["flange_pose"]
                            center_T_base_camera_plan = candidate["T_base_camera"]
                            print(
                                f"[active][center-roll] selected rank={roll_rank}/{plan_limit} "
                                f"roll={roll_deg:.1f}deg "
                                f"rotation_cost={float(candidate['rotation_error_deg']):.2f}deg"
                            )
                            break
                        if center_motion_plan is None:
                            sample = " | ".join(direct_failures[:3])
                            optical_records = []
                            optical_failures = []
                            if bool(
                                getattr(
                                    args,
                                    "active_center_optical_relaxation",
                                    True,
                                )
                            ):
                                optical_records = (
                                    _target_center_optical_relaxation_pose_records(
                                        args,
                                        target_center0,
                                        arm.get_flange_pose(),
                                        np.asarray(
                                            T_base_camera0[:3, 3], dtype=float
                                        ).reshape(3),
                                        hooks=hooks,
                                    )
                                )
                                optical_roll_limit = max(
                                    int(
                                        getattr(
                                            args,
                                            "active_center_optical_roll_plan_candidates",
                                            12,
                                        )
                                    ),
                                    1,
                                )
                                print(
                                    "[active][center-optical] exact downward "
                                    "orientation has no verified IK/path; keep the "
                                    "camera center fixed and try "
                                    f"{len(optical_records)} small optical-axis directions"
                                )
                                for direction_rank, record in enumerate(
                                    optical_records, start=1
                                ):
                                    roll_candidates = _spread_equivalent_roll_candidates(
                                        record["flange_candidates"],
                                        optical_roll_limit,
                                    )
                                    # Always retain the canonical zero-roll
                                    # representative.  A sparse full-circle
                                    # spread can otherwise skip the only wrist
                                    # branch that escapes the exact-downward
                                    # singular boundary.
                                    zero_roll = min(
                                        record["flange_candidates"],
                                        key=lambda item: abs(
                                            float(item["roll_rad"])
                                        ),
                                    )
                                    if abs(
                                        float(roll_candidates[0]["roll_rad"])
                                        - float(zero_roll["roll_rad"])
                                    ) > 1e-9:
                                        roll_candidates = (
                                            [roll_candidates[0], zero_roll]
                                            + [
                                                item
                                                for item in roll_candidates[1:]
                                                if abs(
                                                    float(item["roll_rad"])
                                                    - float(zero_roll["roll_rad"])
                                                )
                                                > 1e-9
                                            ]
                                        )[:optical_roll_limit]
                                    for roll_rank, candidate in enumerate(
                                        roll_candidates, start=1
                                    ):
                                        roll_deg = float(
                                            np.rad2deg(float(candidate["roll_rad"]))
                                        )
                                        label = (
                                            "active_center_optical_"
                                            f"t{float(record['tilt_deg']):g}_"
                                            f"a{float(record['azimuth_deg']):g}_"
                                            f"r{roll_rank}"
                                        )
                                        try:
                                            candidate_plan = plan_active_direct_route(
                                                arm,
                                                candidate["flange_pose"],
                                                center_start_joints,
                                                center_path_scene,
                                                _points_from_cloud(target_base0),
                                                args,
                                                label,
                                                float(
                                                    getattr(
                                                        args,
                                                        "active_center_max_joint_delta_deg",
                                                        45.0,
                                                    )
                                                ),
                                                float(
                                                    getattr(
                                                        args,
                                                        "active_center_max_joint_l2_deg",
                                                        90.0,
                                                    )
                                                ),
                                            )
                                        except (
                                            PlanNotExecutable,
                                            UnsafeMotionTarget,
                                        ) as exc:
                                            if len(optical_failures) < 3:
                                                optical_failures.append(
                                                    "tilt="
                                                    f"{float(record['tilt_deg']):.1f}deg,"
                                                    f"azimuth={float(record['azimuth_deg']):.1f}deg,"
                                                    f"roll={roll_deg:.1f}deg:"
                                                    f"{type(exc).__name__}:{exc}"
                                                )
                                            continue
                                        center_motion_plan = candidate_plan
                                        center_flange_pose = candidate["flange_pose"]
                                        center_T_base_camera_plan = candidate[
                                            "T_base_camera"
                                        ]
                                        center_observation_mode = (
                                            "same_position_optical_relaxation"
                                        )
                                        print(
                                            "[active][center-optical] selected "
                                            f"direction={direction_rank}/{len(optical_records)} "
                                            f"tilt={float(record['tilt_deg']):.1f}deg "
                                            f"azimuth={float(record['azimuth_deg']):.1f}deg "
                                            f"roll={roll_deg:.1f}deg "
                                            f"center_norm={float(record['center_norm']):.3f}; "
                                            "camera position unchanged"
                                        )
                                        break
                                    if center_motion_plan is not None:
                                        break
                        if center_motion_plan is None:
                            if not bool(
                                getattr(
                                    args,
                                    "active_center_spherical_fallback",
                                    True,
                                )
                            ):
                                optical_sample = " | ".join(optical_failures)
                                raise PlanNotExecutable(
                                    f"no guarded direct center path among {plan_limit} "
                                    "exact-overhead rolls or same-position optical "
                                    f"relaxations; overhead_sample={sample}; "
                                    f"optical_sample={optical_sample}"
                                )
                            fallback_records = (
                                _target_center_spherical_fallback_pose_records(
                                    args,
                                    target_center0,
                                    arm.get_flange_pose(),
                                    np.asarray(
                                        T_base_camera0[:3, 3], dtype=float
                                    ).reshape(3),
                                    hooks=hooks,
                                )
                            )
                            fallback_roll_limit = max(
                                int(
                                    getattr(
                                        args,
                                        "active_center_fallback_roll_plan_candidates",
                                        12,
                                    )
                                ),
                                1,
                            )
                            fallback_failures = []
                            print(
                                "[active][center-fallback] exact overhead view has no "
                                f"verified IK/path; try {len(fallback_records)} equal-radius "
                                "tilted look-at positions"
                            )
                            for position_rank, record in enumerate(
                                fallback_records, start=1
                            ):
                                roll_candidates = _spread_equivalent_roll_candidates(
                                    record["flange_candidates"],
                                    fallback_roll_limit,
                                )
                                for roll_rank, candidate in enumerate(
                                    roll_candidates, start=1
                                ):
                                    roll_deg = float(
                                        np.rad2deg(
                                            float(candidate["roll_rad"])
                                        )
                                    )
                                    label = (
                                        "active_center_fallback_"
                                        f"t{float(record['tilt_deg']):.0f}_"
                                        f"a{float(record['azimuth_deg']):.0f}_"
                                        f"r{roll_rank}"
                                    )
                                    try:
                                        candidate_plan = plan_active_direct_route(
                                            arm,
                                            candidate["flange_pose"],
                                            center_start_joints,
                                            center_path_scene,
                                            _points_from_cloud(target_base0),
                                            args,
                                            label,
                                            float(
                                                getattr(
                                                    args,
                                                    "active_center_max_joint_delta_deg",
                                                    45.0,
                                                )
                                            ),
                                            float(
                                                getattr(
                                                    args,
                                                    "active_center_max_joint_l2_deg",
                                                    90.0,
                                                )
                                            ),
                                        )
                                    except (
                                        PlanNotExecutable,
                                        UnsafeMotionTarget,
                                    ) as exc:
                                        if len(fallback_failures) < 3:
                                            fallback_failures.append(
                                                "tilt="
                                                f"{float(record['tilt_deg']):.1f}deg,"
                                                f"azimuth={float(record['azimuth_deg']):.1f}deg,"
                                                f"roll={roll_deg:.1f}deg:"
                                                f"{type(exc).__name__}:{exc}"
                                            )
                                        continue
                                    center_motion_plan = candidate_plan
                                    center_flange_pose = candidate["flange_pose"]
                                    center_T_base_camera_plan = candidate[
                                        "T_base_camera"
                                    ]
                                    center_observation_mode = "spherical_fallback"
                                    print(
                                        "[active][center-fallback] selected "
                                        f"position={position_rank}/{len(fallback_records)} "
                                        f"tilt={float(record['tilt_deg']):.1f}deg "
                                        f"azimuth={float(record['azimuth_deg']):.1f}deg "
                                        f"roll={roll_deg:.1f}deg"
                                    )
                                    break
                                if center_motion_plan is not None:
                                    break
                            if center_motion_plan is None:
                                fallback_sample = " | ".join(fallback_failures)
                                optical_sample = " | ".join(optical_failures)
                                raise PlanNotExecutable(
                                    f"no guarded direct center path among {plan_limit} "
                                    "overhead rolls, "
                                    f"{len(optical_records)} same-position optical "
                                    "relaxations, or "
                                    f"{len(fallback_records)} tilted equal-radius positions; "
                                    f"overhead_sample={sample}; "
                                    f"optical_sample={optical_sample}; "
                                    f"fallback_sample={fallback_sample}"
                                )
                        center_joint_target = np.asarray(
                            center_motion_plan.before_close[-1].joints,
                            dtype=float,
                        ).reshape(7)
                    print(
                        "[active] center_target_flange_pose =",
                        np.round(center_flange_pose, 6).tolist(),
                    )
                    if center_observation_mode == "same_position_optical_relaxation":
                        print(
                            "[active] center observation keeps the exact overhead "
                            "camera position with a small optical-axis relaxation; "
                            "subsequent side views return to this verified anchor"
                        )
                    elif center_observation_mode != "overhead":
                        print(
                            "[active] center observation anchor uses a tilted "
                            "equal-radius fallback; subsequent side views return "
                            "to this verified anchor"
                        )
                except Exception as exc:
                    print(
                        f"[active][abort] center observation path is not verified safe: "
                        f"{type(exc).__name__}: {exc}"
                    )
                    return _single_view_active_result(
                        target_cloud,
                        scene_cloud,
                        T_base_camera0,
                        "active_center_path_unsafe",
                    )
            if not (args.yes or args.yes_active_view):
                if not confirm_motion(
                    "move along the verified full-arm path to the center observation and capture"
                ):
                    print("[active] cancelled by no; keep single-view cloud")
                    return _single_view_active_result(target_cloud, scene_cloud, T_base_camera0, "view_declined")
            try:
                active_motion_sent = True
                if center_motion_plan is None:
                    raise UnsafeMotionTarget(
                        "active center has no stored guarded motion plan; "
                        "no motion was sent"
                    )
                execute_prevalidated_stages(
                    arm,
                    center_motion_plan.before_close,
                    center_motion_plan.start_joints,
                    center_motion_plan,
                    args,
                )
                _wait_for_active_capture_settle(args, "active_center")
                center_joints = np.asarray(arm.get_joint_angles(), dtype=float).reshape(7)
                T_base_camera_center = T_base_camera_from_flange(arm, args)
                bgr_center, target_cloud_center, scene_cloud_center, obstacle_cloud_center, out_dir_center, inst_center, intr_center = capture_active_projected_target_cloud(
                    args,
                    first_inst,
                    T_base_camera_center,
                    target_center0,
                    "active_center",
                )
                if isinstance(inst_center, dict) and inst_center.get("label"):
                    args._active_target_label = str(inst_center["label"])
                print(f"[active] centered-view selected_label={inst_center.get('label', 'unknown')} log_dir={out_dir_center}")
                center_scene_base = _active_motion_capture_base(
                    scene_cloud_center,
                    T_base_camera_center,
                    center_joints,
                    args,
                    "active_motion_scene_center",
                )
                center_target_base = transform_cloud(
                    target_cloud_center, T_base_camera_center
                )
                args._active_fusion_target_shape_profile = _active_target_shape_profile(
                    _points_from_cloud(center_target_base),
                    target_center=np.median(
                        _points_from_cloud(center_target_base), axis=0
                    ),
                    target_label=getattr(args, "_active_target_label", None),
                )
                # Active motion must remain conservative across observations:
                # a basket rim visible only in the initial image must not
                # disappear merely because the centered image cannot see it.
                side_path_target_points = np.vstack(
                    [
                        _points_from_cloud(target_base0),
                        _points_from_cloud(center_target_base),
                    ]
                )
                side_path_scene_points = np.vstack(
                    [
                        _points_from_cloud(scene_base0),
                        _points_from_cloud(center_scene_base),
                    ]
                )
                side_captures = []
                side_motion_safety_captures = []
                if force_center_relocalize:
                    print(
                        "[active][memory] centered target relocation captured; "
                        "side-view candidate generation is intentionally skipped"
                    )
                    candidates = []
                else:
                    try:
                        side_path_scene = _active_motion_scene_points(
                            side_path_target_points,
                            side_path_scene_points,
                            args,
                        )
                        candidates = _build_active_view_candidates(
                            arm,
                            args,
                            target_cloud_center,
                            scene_cloud_center,
                            T_base_camera_center,
                            orbit_lateral_axis=lateral_axis,
                            orbit_geometry="spherical_from_top",
                            target_center_override=target_center0 if args.active_fixed_target_center else None,
                            obstacle_cloud=obstacle_cloud_center,
                        )
                    except (PlanNotExecutable, UnsafeMotionTarget) as exc:
                        # The center observation was already reached and captured.
                        # A side-path map that is too large or otherwise unsafe
                        # forbids any side motion, but it does not invalidate the
                        # centered RGB-D observation or the measured center node.
                        # Continue through the normal center-only fusion/restore
                        # verification instead of reporting a false restore failure.
                        print(
                            "[active][warn] side-view motion planning unavailable "
                            "after successful center capture: "
                            f"{type(exc).__name__}: {exc}"
                        )
                        print(
                            "[active][warn] no side motion was sent; fuse the "
                            "centered view only"
                        )
                        candidates = []
                _print_active_view_candidates(candidates, args)
                valid = [c for c in candidates if str(c.get("reason", "")) == "ok" and np.isfinite(float(c.get("score", -float("inf"))))]
                if len(valid) == 0:
                    print("[active][warn] no target-centered side-view candidate passed checks; fuse centered view only")
                else:
                    tried_angles = []
                    side_capture_attempts = 0
                    max_side_capture_attempts = max(
                        int(
                            getattr(
                                args,
                                "active_side_max_capture_attempts",
                                3,
                            )
                        ),
                        1,
                    )
                    valid_ordered = _active_side_selection_order(valid, args)
                    print(
                        "[active] every side candidate will start from and return to "
                        "the exact saved center joint configuration"
                    )
                    for cand in valid_ordered:
                        if len(side_captures) >= int(args.active_side_view_count):
                            break
                        if side_capture_attempts >= max_side_capture_attempts:
                            print(
                                "[active][capture-budget] reached maximum "
                                f"side-view attempts={max_side_capture_attempts}; "
                                f"accepted={len(side_captures)}/"
                                f"{int(args.active_side_view_count)}"
                            )
                            break
                        view_angle = float(cand.get("view_angle_rad", np.deg2rad(float(cand.get("angle_deg", 0.0)))))
                        if any(_smallest_angle_diff_deg(view_angle, prev_angle) < float(args.active_min_side_disparity_deg) for prev_angle in tried_angles):
                            continue
                        label = f"active_side{len(side_captures) + 1}"
                        print(_format_active_view_candidate(cand, prefix=f"[active] selected {label}: "))
                        planned_side_joints = cand.get("planned_joint_target", None)
                        use_active_up_and_over = bool(
                            getattr(args, "active_up_and_over", False)
                        )
                        if (
                            bool(getattr(args, "active_path_guard", True))
                            and planned_side_joints is None
                            and not use_active_up_and_over
                        ):
                            print(
                                f"[active][reject] {label} has no prevalidated joint path; "
                                "skip without motion"
                            )
                            continue
                        side_out_plan = None
                        side_return_plan = None
                        current_before_side = np.asarray(
                            arm.get_joint_angles(), dtype=float
                        ).reshape(7)
                        center_error_deg = float(
                            np.rad2deg(
                                np.max(np.abs(current_before_side - center_joints))
                            )
                        )
                        if center_error_deg > float(
                            getattr(args, "active_center_joint_match_deg", 2.0)
                        ):
                            raise UnsafeMotionTarget(
                                f"{label} did not start at saved center joints: "
                                f"max error={center_error_deg:.2f}deg"
                            )
                        if use_active_up_and_over:
                            try:
                                candidate_flange = np.asarray(
                                    cand["flange"], dtype=float
                                ).reshape(6)
                                side_out_plan = plan_active_up_and_over_route(
                                    arm,
                                    _modeled_tcp_pose_for_joints(
                                        center_joints, args
                                    ),
                                    candidate_flange,
                                    center_joints,
                                    side_path_scene,
                                    side_path_target_points,
                                    args,
                                    f"{label}_outbound",
                                    float(
                                        getattr(
                                            args,
                                            "active_side_max_joint_delta_deg",
                                            45.0,
                                        )
                                    ),
                                    float(
                                        getattr(
                                            args,
                                            "active_side_max_joint_l2_deg",
                                            75.0,
                                        )
                                    ),
                                    goal_joints=planned_side_joints,
                                    scene_capture_joints=center_joints,
                                )
                                planned_side_joints = np.asarray(
                                    side_out_plan.before_close[-1].joints,
                                    dtype=float,
                                ).reshape(7)
                                side_return_plan = _reverse_active_observation_plan(
                                    side_out_plan,
                                    args,
                                    f"{label}_return_center",
                                    modeled_tcp_pose_for_joints=(
                                        _modeled_tcp_pose_for_joints
                                    ),
                                    urdf_chain_geometry=_urdf_chain_geometry,
                                )
                            except (PlanNotExecutable, UnsafeMotionTarget) as exc:
                                print(
                                    f"[active][reject] {label} has no verified "
                                    f"up-and-over round trip: {type(exc).__name__}: {exc}"
                                )
                                continue
                        else:
                            try:
                                candidate_flange = np.asarray(
                                    cand["flange"], dtype=float
                                ).reshape(6)
                                side_out_plan = plan_active_direct_route(
                                    arm,
                                    candidate_flange,
                                    center_joints,
                                    side_path_scene,
                                    side_path_target_points,
                                    args,
                                    f"{label}_outbound",
                                    float(
                                        getattr(
                                            args,
                                            "active_side_max_joint_delta_deg",
                                            45.0,
                                        )
                                    ),
                                    float(
                                        getattr(
                                            args,
                                            "active_side_max_joint_l2_deg",
                                            75.0,
                                        )
                                    ),
                                    goal_joints=planned_side_joints,
                                    scene_capture_joints=center_joints,
                                )
                                planned_side_joints = np.asarray(
                                    side_out_plan.before_close[-1].joints,
                                    dtype=float,
                                ).reshape(7)
                                side_return_plan = _reverse_active_observation_plan(
                                    side_out_plan,
                                    args,
                                    f"{label}_return_center",
                                    modeled_tcp_pose_for_joints=(
                                        _modeled_tcp_pose_for_joints
                                    ),
                                    urdf_chain_geometry=_urdf_chain_geometry,
                                )
                            except (PlanNotExecutable, UnsafeMotionTarget) as exc:
                                print(
                                    f"[active][reject] {label} has no verified "
                                    f"guarded-direct round trip: "
                                    f"{type(exc).__name__}: {exc}"
                                )
                                continue
                        if bool(getattr(args, "active_confirm_each_view", True)) and not (
                            args.yes or args.yes_active_view
                        ):
                            if not confirm_motion(
                                f"move center -> {cand.get('name')} -> exact center "
                                "along the verified full-arm path"
                            ):
                                raise GraspSelectionDeclined(
                                    f"operator declined {label} active-view motion"
                                )
                        side_motion_attempted = False
                        side_attempt_error = None
                        try:
                            side_motion_attempted = True
                            side_capture_attempts += 1
                            # A physically attempted direction is no longer
                            # new information even when identity/overlap later
                            # rejects its capture.  Recording it here prevents
                            # another near-duplicate move from consuming the
                            # remaining capture budget.
                            if not any(
                                _smallest_angle_diff_deg(
                                    view_angle, previous_angle
                                )
                                < 1e-6
                                for previous_angle in tried_angles
                            ):
                                tried_angles.append(view_angle)
                            print(
                                f"[active][capture-budget] attempt="
                                f"{side_capture_attempts}/"
                                f"{max_side_capture_attempts} "
                                f"accepted={len(side_captures)}"
                            )
                            if side_out_plan is None:
                                raise UnsafeMotionTarget(
                                    f"{label} has no stored guarded outbound "
                                    "motion plan; no motion was sent"
                                )
                            execute_prevalidated_stages(
                                arm,
                                side_out_plan.before_close,
                                side_out_plan.start_joints,
                                side_out_plan,
                                args,
                            )
                            _wait_for_active_capture_settle(args, label)
                            T_base_camera_side = T_base_camera_from_flange(arm, args)
                            side_capture_joints = np.asarray(
                                arm.get_joint_angles(), dtype=float
                            ).reshape(7)
                            bgr_side, target_cloud_side, scene_cloud_side, obstacle_cloud_side, out_dir_side, inst_side, intr_side = capture_active_projected_target_cloud(
                                args,
                                first_inst,
                                T_base_camera_side,
                                target_center0,
                                label,
                            )
                            side_motion_safety_captures.append(
                                (
                                    scene_cloud_side,
                                    obstacle_cloud_side,
                                    T_base_camera_side,
                                    side_capture_joints.copy(),
                                    _active_motion_capture_base(
                                        scene_cloud_side,
                                        T_base_camera_side,
                                        side_capture_joints,
                                        args,
                                        "active_motion_scene_"
                                        f"{len(side_motion_safety_captures) + 1}",
                                    ),
                                )
                            )
                            print(f"[active] {label} selected_label={inst_side.get('label', 'unknown')} log_dir={out_dir_side}")
                            side_target_base_preview = transform_cloud(
                                target_cloud_side, T_base_camera_side
                            )
                            side_overlap_ok = _active_side_capture_overlap_ok(
                                center_target_base,
                                side_target_base_preview,
                                args,
                                label,
                            )
                            if side_overlap_ok:
                                side_captures.append(
                                    (
                                        label.replace("active_", ""),
                                        target_cloud_side,
                                        scene_cloud_side,
                                        obstacle_cloud_side,
                                        T_base_camera_side,
                                        side_motion_safety_captures[-1][4],
                                    )
                                )
                                # Preserve the accepted target surface for
                                # return planning and every later candidate.
                                side_path_target_points = np.vstack(
                                    [
                                        side_path_target_points,
                                        _points_from_cloud(
                                            side_target_base_preview
                                        ),
                                    ]
                                )
                            # Every successfully captured raw scene contributes
                            # to return-path safety, even when target overlap is
                            # too poor for GraspNet fusion.
                            side_path_scene_points = np.vstack(
                                [
                                    side_path_scene_points,
                                    _points_from_cloud(
                                        side_motion_safety_captures[-1][4]
                                    ),
                                ]
                            )
                            side_path_scene = _active_motion_scene_points(
                                side_path_target_points,
                                side_path_scene_points,
                                args,
                            )
                            print(
                                f"[active][return] {label} will reverse the "
                                "stored, validated outbound joint path; the "
                                "new capture is reserved for fusion and later "
                                "grasp planning"
                            )
                        except MotionNotReached as exc:
                            side_attempt_error = exc
                            print(f"[active][warn] {label} candidate {cand.get('name')} not reached; trying next candidate. {exc}")
                        except SafetyStopTriggered:
                            raise
                        except (PlanNotExecutable, UnsafeMotionTarget) as exc:
                            side_attempt_error = exc
                            # Includes fail-closed safety-cloud overflow after
                            # a new side capture.  Do not continue to another
                            # view using an older, incomplete occupancy map.
                            print(
                                f"[active][warn] {label} capture/scene update "
                                f"failed: {type(exc).__name__}: {exc}"
                            )
                            raise
                        except Exception as exc:
                            side_attempt_error = exc
                            print(f"[active][warn] {label} capture failed, skip this side view: {type(exc).__name__}: {exc}")
                        finally:
                            if side_motion_attempted and not bool(
                                getattr(arm, "_safety_stop_latched", False)
                            ):
                                print(
                                    f"[active] {label} returning to exact saved center joints "
                                    "before any other candidate"
                                )
                                try:
                                    if side_return_plan is None:
                                        failure_detail = (
                                            "unknown"
                                            if side_attempt_error is None
                                            else f"{type(side_attempt_error).__name__}: "
                                            f"{side_attempt_error}"
                                        )
                                        raise UnsafeMotionTarget(
                                            f"{label} has no verified return route "
                                            "for the latest scene; arm remains "
                                            "stationary at the side view; "
                                            f"original_failure={failure_detail}"
                                        )
                                    execute_prevalidated_stages(
                                        arm,
                                        side_return_plan.before_close,
                                        side_return_plan.start_joints,
                                        side_return_plan,
                                        args,
                                    )
                                except SafetyStopTriggered:
                                    raise
                                except Exception as restore_exc:
                                    raise UnsafeMotionTarget(
                                        f"{label} exact center-joint restore failed; "
                                        "further observation/grasp is forbidden: "
                                        f"{type(restore_exc).__name__}: {restore_exc}"
                                    ) from restore_exc
                    if len(side_captures) == 0:
                        print("[active][warn] no side-view candidate could be reached; fuse centered view only")
            except GraspSelectionDeclined as exc:
                print(f"[active] {exc}; stop active-view workflow")
                return ActiveViewResult(
                    target_cloud=target_cloud,
                    scene_cloud=scene_cloud,
                    T_base_camera=T_base_camera0,
                    used_multiview=False,
                    kept_view_count=1,
                    reason="view_declined",
                    restore_ok=True,
                    motion_sent=active_motion_sent,
                    motion_scene_cloud=scene_cloud,
                )
            except MotionNotReached as exc:
                print(f"[active][warn] active_center was not reached; skip active-view fusion and continue with first-view cloud. {exc}")
                return _single_view_active_result(
                    target_cloud,
                    scene_cloud,
                    T_base_camera0,
                    "active_center_not_reached",
                    active_motion_sent,
                    restore_ok=_active_center_joint_state_is_verified(
                        arm,
                        center_joints,
                        args,
                        "failed center observation",
                    ),
                )
            except SafetyStopTriggered:
                raise
            except UnsafeMotionTarget as exc:
                print(f"[active][warn] target-centered active view is unsafe: {exc}")
                print("[active][warn] skip active-view fusion and continue with first-view cloud.")
                return _single_view_active_result(
                    target_cloud,
                    scene_cloud,
                    T_base_camera0,
                    "active_view_unsafe",
                    active_motion_sent,
                    restore_ok=_active_center_joint_state_is_verified(
                        arm,
                        center_joints,
                        args,
                        "unsafe active observation",
                    ),
                )
            except Exception as exc:
                print(f"[active][warn] target-centered active-view capture failed: {type(exc).__name__}: {exc}")
                print("[active][warn] skip active-view fusion and continue with first-view cloud.")
                return _single_view_active_result(
                    target_cloud,
                    scene_cloud,
                    T_base_camera0,
                    "active_capture_failed",
                    active_motion_sent,
                    restore_ok=_active_center_joint_state_is_verified(
                        arm,
                        center_joints,
                        args,
                        "failed active capture",
                    ),
                )

            print(
                "[active] first observation is used only for target selection/center estimation; "
                "it is not included in active-view fusion."
            )
            target_base_clouds = [
                transform_cloud(target_cloud_center, T_base_camera_center),
            ]
            target_camera_origins_base = [
                np.asarray(T_base_camera_center, dtype=float).reshape(4, 4)[:3, 3].copy()
            ]
            target_view_names = ["center"]
            scene_base_clouds = [
                center_scene_base,
            ]
            for (
                side_name,
                target_cloud_side,
                scene_cloud_side,
                obstacle_cloud_side,
                T_base_camera_side,
                scene_base_side,
            ) in side_captures:
                target_base_clouds.append(transform_cloud(target_cloud_side, T_base_camera_side))
                target_camera_origins_base.append(
                    np.asarray(T_base_camera_side, dtype=float).reshape(4, 4)[:3, 3].copy()
                )
                target_view_names.append(side_name)
                scene_base_clouds.append(scene_base_side)
            # Keep an unregistered conservative union for robot motion.  Target-
            # cloud calibration is useful for GraspNet fusion, but applying that
            # correction to a mat/basket/background can move real obstacles.
            # Geometric robot self-points are the one exception: remove those
            # per capture with the arm configuration that produced the image,
            # before any scene/environment union is formed.
            initial_motion_scene_base = scene_base0
            raw_motion_scene_base_clouds = [
                center_scene_base
            ] + [
                filtered_scene_base
                for side_index, (
                    scene_side,
                    _environment_side,
                    T_side,
                    capture_joints,
                    filtered_scene_base,
                )
                in enumerate(side_motion_safety_captures, start=1)
            ]
            raw_motion_environment_base_clouds = [
                _active_motion_capture_base(
                    obstacle_cloud_center,
                    T_base_camera_center,
                    center_joints,
                    args,
                    "active_motion_environment_center",
                )
            ] + [
                _active_motion_capture_base(
                    environment_side,
                    T_side,
                    capture_joints,
                    args,
                    f"active_motion_environment_{side_index}",
                )
                for side_index, (
                    _scene_side,
                    environment_side,
                    T_side,
                    capture_joints,
                    _filtered_scene_base,
                )
                in enumerate(side_motion_safety_captures, start=1)
            ]
            raw_motion_environment_view_names = ["center"] + [
                f"side_capture_{side_index}"
                for side_index in range(1, len(side_motion_safety_captures) + 1)
            ]
            dynamic_target_base = np.vstack(
                [_points_from_cloud(target_base0)]
                + [_points_from_cloud(cloud) for cloud in target_base_clouds]
            )
            target_base_clouds, scene_base_clouds, reg_keep = register_active_fusion_views(
                target_view_names,
                target_base_clouds,
                scene_base_clouds,
                T_base_camera0,
                out_dir,
                args,
            )
            diag_keep = diagnose_active_fusion(target_view_names, target_base_clouds, T_base_camera0, out_dir, args)
            keep_views = [bool(a and b) for a, b in zip(reg_keep, diag_keep)]
            if bool(args.active_fusion_reject_bad_views) and len(keep_views) == len(target_base_clouds):
                target_base_clouds = [cloud for cloud, keep in zip(target_base_clouds, keep_views) if keep]
                scene_base_clouds = [cloud for cloud, keep in zip(scene_base_clouds, keep_views) if keep]
                target_camera_origins_base = [
                    origin
                    for origin, keep in zip(target_camera_origins_base, keep_views)
                    if keep
                ]
                kept_names = [name for name, keep in zip(target_view_names, keep_views) if keep]
                print(f"[active] fusion kept views={kept_names}")
                print(
                    "[active][motion] raw obstacle occupancy from every captured "
                    "view is retained even when target fusion rejects that view"
                )
                if len(kept_names) <= 1:
                    print("[active][warn] active fusion kept only one active view; side observations did not contribute to GraspNet input.")
            T_camera0_base = np.linalg.inv(np.asarray(T_base_camera0, dtype=float).reshape(4, 4))
            raw_fused_target, fused_target = _fuse_active_target_clouds(
                target_base_clouds,
                args,
                T_camera0_base,
                camera_origins_base=target_camera_origins_base,
                merge_clouds_impl=merge_clouds,
                transform_cloud_impl=transform_cloud,
                hooks=hooks,
            )
            raw_fused_scene = transform_cloud(
                merge_clouds(
                    scene_base_clouds,
                    args.scene_voxel_size,
                    filter_outliers=False,
                    outlier_neighbors=int(args.outlier_neighbors),
                    outlier_std_ratio=float(args.outlier_std_ratio),
                ),
                T_camera0_base,
            )
            fused_scene = transform_cloud(
                merge_clouds(
                    scene_base_clouds,
                    args.scene_voxel_size,
                    filter_outliers=bool(args.fusion_scene_outlier_filter),
                    outlier_neighbors=int(args.outlier_neighbors),
                    outlier_std_ratio=float(args.outlier_std_ratio),
                ),
                T_camera0_base,
            )
            fused_motion_scene = transform_cloud(
                merge_clouds(
                    [initial_motion_scene_base]
                    + raw_motion_scene_base_clouds,
                    args.scene_voxel_size,
                    # Safety occupancy must preserve sparse basket rims and
                    # one-view obstacles; statistical filtering is only for
                    # the GraspNet visualization/collision cloud above.
                    filter_outliers=False,
                    outlier_neighbors=int(args.outlier_neighbors),
                    outlier_std_ratio=float(args.outlier_std_ratio),
                ),
                T_camera0_base,
            )
            initial_environment_source = (
                initial_obstacle_cloud
                if initial_obstacle_cloud is not None
                else scene_cloud
            )
            initial_motion_environment_base = _active_motion_capture_base(
                initial_environment_source,
                T_base_camera0,
                center_start_joints,
                args,
                "active_motion_environment_initial",
            )
            if initial_obstacle_cloud is None:
                initial_motion_environment_base = (
                    _exclude_target_band_from_environment(
                        initial_motion_environment_base,
                        target_base0,
                        args,
                    )
                )
            environment_union = list(raw_motion_environment_base_clouds)
            environment_view_names = list(raw_motion_environment_view_names)
            environment_union.insert(0, initial_motion_environment_base)
            environment_view_names.insert(0, "initial")
            legacy_environment_union = list(raw_motion_environment_base_clouds)
            if initial_obstacle_cloud is not None:
                legacy_environment_union.insert(
                    0, initial_motion_environment_base
                )
            motion_environment_map = None
            fused_motion_environment = None
            try:
                (
                    motion_environment_map,
                    fused_motion_environment,
                ) = _build_dynamic_motion_environment(
                    environment_view_names,
                    environment_union,
                    args,
                    T_camera0_base,
                    hooks=hooks,
                    target_base=dynamic_target_base,
                    map_id="active_center_multiview_environment",
                    target_center_base=target_center0,
                )
            except Exception as exc:
                print(
                    "[active][motion][map][warn] dynamic environment fusion "
                    f"failed; keep legacy conservative union: "
                    f"{type(exc).__name__}: {exc}"
                )
            if (
                fused_motion_environment is None
                and motion_environment_map is None
                and legacy_environment_union
            ):
                fused_motion_environment = transform_cloud(
                    merge_clouds(
                        legacy_environment_union,
                        args.scene_voxel_size,
                        filter_outliers=False,
                        outlier_neighbors=int(args.outlier_neighbors),
                        outlier_std_ratio=float(args.outlier_std_ratio),
                    ),
                    T_camera0_base,
                )
            fused_scene = crop_scene_cloud_near_target(fused_scene, fused_target, args, label="active_fused_scene")

            save_cloud(os.path.join(out_dir, "active_fused_target_raw_camera0.ply"), raw_fused_target)
            save_cloud(os.path.join(out_dir, "active_fused_target_camera0.ply"), fused_target)
            save_cloud(os.path.join(out_dir, "active_fused_scene_raw_camera0.ply"), raw_fused_scene)
            save_cloud(os.path.join(out_dir, "active_fused_scene_camera0.ply"), fused_scene)
            print(
                f"[active] fused target_points={len(fused_target.points)} scene_points={len(fused_scene.points)} "
                f"views={len(target_base_clouds)} frame=camera0"
            )
            preview_fused_cloud(fused_target, fused_scene, args)

            restore_ok = True
            if args.active_restore_center:
                print(
                    "[active] verifying/restoring the exact saved center joint "
                    "configuration before grasp planning"
                )
                try:
                    current_center_check = np.asarray(
                        arm.get_joint_angles(), dtype=float
                    ).reshape(7)
                    restore_error_deg = float(
                        np.rad2deg(
                            np.max(
                                np.abs(current_center_check - center_joints)
                            )
                        )
                    )
                    allowed_restore_error = float(
                        getattr(args, "active_center_joint_match_deg", 2.0)
                    )
                    if restore_error_deg > allowed_restore_error:
                        raise UnsafeMotionTarget(
                            "active view did not finish at the prevalidated "
                            f"center node: error={restore_error_deg:.2f}deg > "
                            f"{allowed_restore_error:.2f}deg; no unplanned restore "
                            "motion was sent"
                        )
                    print(
                        f"[active] saved center already verified: "
                        f"max_joint_error={restore_error_deg:.2f}deg"
                    )
                except SafetyStopTriggered:
                    raise
                except Exception as exc:
                    restore_ok = False
                    print(f"[active][warn] target-center restore failed: {type(exc).__name__}: {exc}")
                    print("[active][abort] grasp planning is forbidden after a failed center restore.")
            kept_view_count = len(target_base_clouds)
            return ActiveViewResult(
                target_cloud=fused_target,
                scene_cloud=fused_scene,
                T_base_camera=T_base_camera0,
                used_multiview=kept_view_count >= 2,
                kept_view_count=kept_view_count,
                reason="fused" if kept_view_count >= 2 else "center_view_only",
                restore_ok=restore_ok,
                motion_sent=active_motion_sent,
                motion_scene_cloud=fused_motion_scene,
                motion_environment_cloud=fused_motion_environment,
                motion_environment_map=motion_environment_map,
            )

    if bool(getattr(args, "active_path_guard", True)):
        print(
            "[active][abort] full-arm path guard requires --active-center-first; "
            "legacy direct-orbit motion is disabled"
        )
        return _single_view_active_result(
            target_cloud,
            scene_cloud,
            T_base_camera0,
            "active_center_required",
        )
    candidates = _build_active_view_candidates(arm, args, target_cloud, scene_cloud, T_base_camera0)
    _print_active_view_candidates(candidates, args)
    valid = [c for c in candidates if str(c.get("reason", "")) == "ok" and np.isfinite(float(c.get("score", -float("inf"))))]
    if len(valid) == 0:
        print("[active][warn] no orbit active-view candidate passed checks; keep single-view cloud")
        return _single_view_active_result(target_cloud, scene_cloud, T_base_camera0, "no_valid_active_candidate")
    chosen = valid[0]
    print(_format_active_view_candidate(chosen, prefix="[active] selected candidate: "))
    if not args.execute:
        print("[active][dry-run] --execute not set; active-view motion skipped")
        return _single_view_active_result(target_cloud, scene_cloud, T_base_camera0, "execute_disabled")
    if not (args.yes or args.yes_active_view):
        if not confirm_motion(
            "move to the selected active-view pose and capture the second frame"
        ):
            print("[active] cancelled by no; keep single-view cloud")
            return _single_view_active_result(target_cloud, scene_cloud, T_base_camera0, "view_declined")

    strict_target_center = None
    if bool(getattr(args, "active_strict_target_identity", False)):
        try:
            strict_target_center = _target_center_base_from_cloud(
                target_cloud,
                T_base_camera0,
            )
        except Exception as exc:
            print(
                f"[active][abort] strict target identity has no valid 3D center: "
                f"{type(exc).__name__}: {exc}"
            )
            return _single_view_active_result(
                target_cloud,
                scene_cloud,
                T_base_camera0,
                "strict_target_center_unavailable",
            )

    first_capture_joints = np.asarray(
        arm.get_joint_angles(), dtype=float
    ).reshape(7)
    obstacle_cloud2 = None
    try:
        active_motion_sent = True
        if str(chosen.get("command_type", "tcp")) == "flange":
            move_flange_pose(arm, np.asarray(chosen["flange"], dtype=float), args, "active_view")
        else:
            move_tcp_pose(arm, np.asarray(chosen["tcp"], dtype=float), args, "active_view")
        _wait_for_active_capture_settle(args, "active_side")
        T_base_camera1 = T_base_camera_from_flange(arm, args)
        second_capture_joints = np.asarray(
            arm.get_joint_angles(), dtype=float
        ).reshape(7)
        if strict_target_center is not None:
            (
                bgr2,
                target_cloud2,
                scene_cloud2,
                obstacle_cloud2,
                out_dir2,
                inst2,
                intr2,
            ) = capture_active_projected_target_cloud(
                args,
                first_inst,
                T_base_camera1,
                strict_target_center,
                "active_side",
            )
        else:
            args2 = _active_view_second_capture_args(args, first_inst)
            bgr2, target_cloud2, scene_cloud2, out_dir2, inst2, intr2 = capture_target_cloud(args2)
            obstacle_cloud2 = getattr(args2, "_last_obstacle_cloud", None)
        print(f"[active] second-view selected_label={inst2.get('label', 'unknown')} log_dir={out_dir2}")
    except MotionNotReached as exc:
        print(f"[active][warn] selected active view was not reached; skip active-view fusion and continue with first-view cloud. {exc}")
        return _single_view_active_result(target_cloud, scene_cloud, T_base_camera0, "active_view_not_reached", active_motion_sent)
    except SafetyStopTriggered:
        raise
    except UnsafeMotionTarget as exc:
        print(f"[active][warn] selected active view is unsafe: {exc}")
        print("[active][warn] skip active-view fusion and continue with first-view cloud.")
        return _single_view_active_result(target_cloud, scene_cloud, T_base_camera0, "active_view_unsafe", active_motion_sent)
    except Exception as exc:
        print(f"[active][warn] active-view capture failed: {type(exc).__name__}: {exc}")
        print("[active][warn] skip active-view fusion and continue with first-view cloud.")
        return _single_view_active_result(target_cloud, scene_cloud, T_base_camera0, "active_capture_failed", active_motion_sent)

    target0_base = transform_cloud(target_cloud, T_base_camera0)
    target1_base = transform_cloud(target_cloud2, T_base_camera1)
    target0_points = _points_from_cloud(target0_base)
    if target0_points.shape[0] > 0:
        args._active_fusion_target_shape_profile = _active_target_shape_profile(
            target0_points,
            target_center=np.median(target0_points, axis=0),
            target_label=getattr(args, "_active_target_label", None),
        )
    scene0_base = transform_cloud(scene_cloud, T_base_camera0)
    scene1_base = transform_cloud(scene_cloud2, T_base_camera1)
    target_base_pair = [target0_base, target1_base]
    target_camera_origins_base = [
        np.asarray(T_base_camera0, dtype=float).reshape(4, 4)[:3, 3].copy(),
        np.asarray(T_base_camera1, dtype=float).reshape(4, 4)[:3, 3].copy(),
    ]
    scene_base_pair = [scene0_base, scene1_base]
    dynamic_target_base = np.vstack(
        [_points_from_cloud(target0_base), _points_from_cloud(target1_base)]
    )
    raw_motion_scene_base_pair = [
        copy.deepcopy(scene0_base),
        copy.deepcopy(scene1_base),
    ]
    first_environment_source = (
        initial_obstacle_cloud
        if initial_obstacle_cloud is not None
        else scene_cloud
    )
    second_environment_source = (
        obstacle_cloud2 if obstacle_cloud2 is not None else scene_cloud2
    )
    raw_motion_environment_base_pair = [
        _active_motion_capture_base(
            first_environment_source,
            T_base_camera0,
            first_capture_joints,
            args,
            "active_motion_environment_first",
        ),
        _active_motion_capture_base(
            second_environment_source,
            T_base_camera1,
            second_capture_joints,
            args,
            "active_motion_environment_side",
        ),
    ]
    dynamic_motion_environment_base_pair = list(
        raw_motion_environment_base_pair
    )
    if initial_obstacle_cloud is None:
        dynamic_motion_environment_base_pair[0] = (
            _exclude_target_band_from_environment(
                dynamic_motion_environment_base_pair[0],
                target0_base,
                args,
            )
        )
    if obstacle_cloud2 is None:
        dynamic_motion_environment_base_pair[1] = (
            _exclude_target_band_from_environment(
                dynamic_motion_environment_base_pair[1],
                target1_base,
                args,
            )
        )
    pair_names = ["first", "side"]
    target_base_pair, scene_base_pair, reg_keep = register_active_fusion_views(
        pair_names,
        target_base_pair,
        scene_base_pair,
        T_base_camera0,
        out_dir,
        args,
    )
    diag_keep = diagnose_active_fusion(pair_names, target_base_pair, T_base_camera0, out_dir, args)
    keep_views = [bool(a and b) for a, b in zip(reg_keep, diag_keep)]
    if bool(args.active_fusion_reject_bad_views) and len(keep_views) == len(target_base_pair):
        target_base_pair = [cloud for cloud, keep in zip(target_base_pair, keep_views) if keep]
        scene_base_pair = [cloud for cloud, keep in zip(scene_base_pair, keep_views) if keep]
        target_camera_origins_base = [
            origin
            for origin, keep in zip(target_camera_origins_base, keep_views)
            if keep
        ]
        print(f"[active] fusion kept views={[name for name, keep in zip(pair_names, keep_views) if keep]}")
        print(
            "[active][motion] raw obstacle occupancy from both captured views "
            "is retained even when target fusion rejects one"
        )
        if int(np.count_nonzero(keep_views)) <= 1:
            print("[active][warn] active fusion kept only the first view; active-view observation did not contribute to GraspNet input.")
    T_camera0_base = np.linalg.inv(np.asarray(T_base_camera0, dtype=float).reshape(4, 4))
    motion_environment_map = None
    fused_motion_environment = None
    try:
        (
            motion_environment_map,
            fused_motion_environment,
        ) = _build_dynamic_motion_environment(
            pair_names,
            dynamic_motion_environment_base_pair,
            args,
            T_camera0_base,
            hooks=hooks,
            target_base=dynamic_target_base,
            map_id="active_two_view_environment",
            target_center_base=(
                None
                if dynamic_target_base.shape[0] == 0
                else np.median(dynamic_target_base, axis=0)
            ),
        )
    except Exception as exc:
        print(
            "[active][motion][map][warn] dynamic two-view environment "
            f"fusion failed; keep legacy conservative union: "
            f"{type(exc).__name__}: {exc}"
        )
    if fused_motion_environment is None and motion_environment_map is None:
        fused_motion_environment = transform_cloud(
            merge_clouds(
                raw_motion_environment_base_pair,
                args.scene_voxel_size,
                filter_outliers=False,
                outlier_neighbors=int(args.outlier_neighbors),
                outlier_std_ratio=float(args.outlier_std_ratio),
            ),
            T_camera0_base,
        )
    raw_fused_target, fused_target = _fuse_active_target_clouds(
        target_base_pair,
        args,
        T_camera0_base,
        camera_origins_base=target_camera_origins_base,
        merge_clouds_impl=merge_clouds,
        transform_cloud_impl=transform_cloud,
        hooks=hooks,
    )
    raw_fused_scene = transform_cloud(
        merge_clouds(
            scene_base_pair,
            args.scene_voxel_size,
            filter_outliers=False,
            outlier_neighbors=int(args.outlier_neighbors),
            outlier_std_ratio=float(args.outlier_std_ratio),
        ),
        T_camera0_base,
    )
    fused_scene = transform_cloud(
        merge_clouds(
            scene_base_pair,
            args.scene_voxel_size,
            filter_outliers=bool(args.fusion_scene_outlier_filter),
            outlier_neighbors=int(args.outlier_neighbors),
            outlier_std_ratio=float(args.outlier_std_ratio),
        ),
        T_camera0_base,
    )
    fused_motion_scene = transform_cloud(
        merge_clouds(
            raw_motion_scene_base_pair,
            args.scene_voxel_size,
            filter_outliers=False,
            outlier_neighbors=int(args.outlier_neighbors),
            outlier_std_ratio=float(args.outlier_std_ratio),
        ),
        T_camera0_base,
    )
    fused_scene = crop_scene_cloud_near_target(fused_scene, fused_target, args, label="active_fused_scene")

    save_cloud(os.path.join(out_dir, "active_fused_target_raw_camera0.ply"), raw_fused_target)
    save_cloud(os.path.join(out_dir, "active_fused_target_camera0.ply"), fused_target)
    save_cloud(os.path.join(out_dir, "active_fused_scene_raw_camera0.ply"), raw_fused_scene)
    save_cloud(os.path.join(out_dir, "active_fused_scene_camera0.ply"), fused_scene)
    print(
        f"[active] fused target_points={len(fused_target.points)} scene_points={len(fused_scene.points)} "
        "frame=camera0"
    )
    preview_fused_cloud(fused_target, fused_scene, args)

    restore_ok = True
    if args.active_restore_center:
        print("[active] restoring first observation TCP pose before grasp planning")
        try:
            move_tcp_pose(arm, np.asarray(chosen["current_tcp"], dtype=float), args, "active_restore")
        except SafetyStopTriggered:
            raise
        except Exception as exc:
            restore_ok = False
            print(f"[active][warn] center restore failed: {type(exc).__name__}: {exc}")
            print("[active][warn] continue with fused cloud from current pose.")
    kept_view_count = len(target_base_pair)
    return ActiveViewResult(
        target_cloud=fused_target,
        scene_cloud=fused_scene,
        T_base_camera=T_base_camera0,
        used_multiview=kept_view_count >= 2,
        kept_view_count=kept_view_count,
        reason="fused" if kept_view_count >= 2 else "single_view_after_rejection",
        restore_ok=restore_ok,
        motion_sent=active_motion_sent,
        motion_scene_cloud=fused_motion_scene,
        motion_environment_cloud=fused_motion_environment,
        motion_environment_map=motion_environment_map,
    )
