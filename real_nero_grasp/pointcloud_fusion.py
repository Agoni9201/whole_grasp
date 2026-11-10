"""Point-cloud merging, registration, diagnostics, and active-view previews."""

import copy
import json
import os
from datetime import datetime

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from real_nero_grasp.types import DynamicEnvironmentMap, SafetyStopTriggered, ControlledMotionStopTriggered
from real_nero_grasp.visualization import draw_open3d_preview, get_o3d


__all__ = [
    "transform_cloud",
    "merge_clouds",
    "fuse_target_clouds",
    "fuse_target_clouds_surfel",
    "fuse_dynamic_environment_clouds",
    "next_dynamic_environment_generation",
    "save_cloud",
    "_cloud_stats",
    "_nn_stats",
    "_mutual_overlap_stats",
    "_nn_gate",
    "_apply_T_points",
    "_downsample_points",
    "_small_translation_registration",
    "_rigid_icp_registration",
    "_fit_table_plane",
    "_round_table_plane_correction",
    "_round_static_xyyaw_correction",
    "register_active_fusion_views",
    "_active_side_capture_overlap_ok",
    "diagnose_active_fusion",
    "preview_fused_cloud",
    "_make_cloud_from_points_colors",
    "small_isolated_component_keep_mask",
]


def next_dynamic_environment_generation(args):
    """Allocate a session-wide map generation shared by shallow arg copies."""
    owner = getattr(args, "_motion_environment_generation_owner", None)
    if not isinstance(owner, dict) or "value" not in owner:
        try:
            initial = int(getattr(args, "_motion_environment_generation", 0))
        except (TypeError, ValueError):
            initial = 0
        owner = {"value": max(initial, 0)}
        args._motion_environment_generation_owner = owner
    owner["value"] = int(owner["value"]) + 1
    args._motion_environment_generation = int(owner["value"])
    return int(owner["value"])


def _points_from_cloud(cloud):
    return np.asarray(cloud.points, dtype=np.float32).reshape(-1, 3)


def small_isolated_component_keep_mask(
    points,
    *,
    connection_radius=0.020,
    max_component_size=8,
):
    """Return a mask that drops only small disconnected 3D components.

    A thin basket rim remains one long connected component, whereas a compact
    flying-depth island remains small even when its few samples are mutually
    dense.  This complements, rather than replaces, local density filtering.
    """
    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    count = int(points.shape[0])
    keep = np.ones(count, dtype=bool)
    radius = max(float(connection_radius), 0.0)
    maximum = max(int(max_component_size), 0)
    if count == 0 or radius <= 0.0 or maximum <= 0:
        return keep, {"removed_points": 0, "removed_components": 0}

    tree = cKDTree(points)
    neighbours = tree.query_ball_point(points, radius)
    visited = np.zeros(count, dtype=bool)
    removed_components = 0
    for seed in range(count):
        if visited[seed]:
            continue
        stack = [seed]
        visited[seed] = True
        component = []
        while stack:
            index = stack.pop()
            component.append(index)
            for neighbour in neighbours[index]:
                neighbour = int(neighbour)
                if not visited[neighbour]:
                    visited[neighbour] = True
                    stack.append(neighbour)
        if len(component) <= maximum:
            keep[np.asarray(component, dtype=np.int64)] = False
            removed_components += 1

    return keep, {
        "removed_points": int(np.count_nonzero(~keep)),
        "removed_components": int(removed_components),
    }


def _make_cloud_from_points_colors(points, colors=None, *, get_o3d_hook=None):
    o3d = (get_o3d if get_o3d_hook is None else get_o3d_hook)()
    cloud = o3d.geometry.PointCloud()
    # Open3D's pybind Vector3dVector expects a contiguous double buffer.  A
    # boolean-indexed float32 fusion view can otherwise reach an ABI-dependent
    # segfault instead of raising a Python error.
    points = np.ascontiguousarray(
        np.asarray(points, dtype=np.float64).reshape(-1, 3)
    )
    cloud.points = o3d.utility.Vector3dVector(points)
    if colors is not None:
        colors = np.ascontiguousarray(
            np.asarray(colors, dtype=np.float64).reshape(-1, 3)
        )
        if colors.shape[0] == points.shape[0]:
            cloud.colors = o3d.utility.Vector3dVector(
                np.clip(colors, 0.0, 1.0)
            )
    return cloud


def filter_capture_cloud(points, colors, voxel, outlier=None, *, get_o3d_hook=None):
    """Original capture voxel/SOR operations, reusable by the numerical worker."""
    cloud = _make_cloud_from_points_colors(points, colors, get_o3d_hook=get_o3d_hook)
    if voxel > 0:
        cloud = cloud.voxel_down_sample(float(voxel))
    if outlier is not None:
        try:
            neighbors, ratio = outlier
            cloud, _ = cloud.remove_statistical_outlier(
                nb_neighbors=min(int(neighbors), max(len(cloud.points) - 1, 1)),
                std_ratio=float(ratio),
            )
        except Exception:
            pass  # preserve the existing optional target SOR policy
    return cloud


def transform_cloud(cloud, T):
    out = copy.deepcopy(cloud)
    out.transform(np.asarray(T, dtype=float).reshape(4, 4))
    return out


def merge_clouds(
    clouds,
    voxel_size=0.0015,
    filter_outliers=True,
    outlier_neighbors=20,
    outlier_std_ratio=2.0,
    *,
    get_o3d_hook=None,
):
    o3d = (get_o3d if get_o3d_hook is None else get_o3d_hook)()
    merged = o3d.geometry.PointCloud()
    for cloud in clouds:
        if cloud is not None:
            merged += cloud
    if (
        voxel_size is not None
        and float(voxel_size) > 0.0
        and len(merged.points) > 0
    ):
        merged = merged.voxel_down_sample(float(voxel_size))
    if bool(filter_outliers):
        try:
            if len(merged.points) > 50:
                neighbors = min(
                    int(outlier_neighbors),
                    max(len(merged.points) - 1, 1),
                )
                merged, _ = merged.remove_statistical_outlier(
                    nb_neighbors=neighbors,
                    std_ratio=float(outlier_std_ratio),
                )
        except Exception:
            pass
    return merged


def _cloud_points_and_colors(cloud):
    """Return finite point/color rows without manufacturing RGB values."""
    if cloud is None:
        return np.zeros((0, 3), dtype=np.float32), None
    points = _points_from_cloud(cloud)
    finite = np.isfinite(points).all(axis=1)
    points = points[finite]
    colors = None
    if hasattr(cloud, "colors"):
        candidate = np.asarray(cloud.colors, dtype=np.float32).reshape(-1, 3)
        if candidate.shape[0] == finite.shape[0]:
            colors = candidate[finite]
    return points, colors


def fuse_target_clouds(
    clouds,
    voxel_size=0.0025,
    *,
    single_view_radius=0.008,
    single_view_min_neighbors=2,
    filter_outliers=False,
    outlier_neighbors=20,
    outlier_std_ratio=2.0,
    view_weights=None,
    return_stats=False,
    get_o3d_hook=None,
):
    """Fuse target-only views without letting one dense view dominate.

    Each source view contributes at most one representative to an output
    voxel.  The final position and RGB are confidence-weighted means of those
    independent representatives, rather than a raw concatenation followed by
    downsampling.  A point is removed only when it is both single-view and
    spatially isolated; coherent partial surfaces seen by one view remain.
    """
    voxel = float(voxel_size)
    if not np.isfinite(voxel) or voxel <= 0.0:
        raise ValueError("target fusion voxel size must be finite and positive")
    clouds = tuple(clouds)
    if view_weights is None:
        weights = np.ones((len(clouds),), dtype=np.float64)
    else:
        weights = np.asarray(view_weights, dtype=np.float64).reshape(-1)
        if weights.size != len(clouds):
            raise ValueError("target fusion view_weights must match clouds")
        weights[~np.isfinite(weights)] = 0.0
        weights = np.maximum(weights, 0.0)

    per_view_points = []
    per_view_colors = []
    per_view_keys = []
    per_view_weights = []
    has_colors = False
    raw_points = 0
    for cloud, weight in zip(clouds, weights):
        points, colors = _cloud_points_and_colors(cloud)
        raw_points += int(points.shape[0])
        if points.shape[0] == 0 or weight <= 0.0:
            continue
        keys = np.floor(points.astype(np.float64) / voxel + 1e-5).astype(np.int64)
        unique_keys, inverse = np.unique(keys, axis=0, return_inverse=True)
        count = np.bincount(inverse, minlength=unique_keys.shape[0]).astype(np.float64)
        representative = np.column_stack(
            [
                np.bincount(inverse, weights=points[:, axis], minlength=unique_keys.shape[0])
                / count
                for axis in range(3)
            ]
        ).astype(np.float32)
        per_view_keys.append(unique_keys)
        per_view_points.append(representative)
        per_view_weights.append(np.full(unique_keys.shape[0], weight, dtype=np.float64))
        if colors is not None:
            has_colors = True
            per_view_colors.append(
                np.column_stack(
                    [
                        np.bincount(inverse, weights=colors[:, axis], minlength=unique_keys.shape[0])
                        / count
                        for axis in range(3)
                    ]
                ).astype(np.float32)
            )
        else:
            per_view_colors.append(None)

    o3d = (get_o3d if get_o3d_hook is None else get_o3d_hook)()
    if not per_view_keys:
        empty = o3d.geometry.PointCloud()
        stats = {
            "input_views": len(clouds),
            "raw_points": raw_points,
            "view_representatives": 0,
            "fused_voxels": 0,
            "removed_isolated": 0,
        }
        return (empty, stats) if return_stats else empty

    stacked_keys = np.vstack(per_view_keys)
    stacked_points = np.vstack(per_view_points).astype(np.float64)
    stacked_weights = np.concatenate(per_view_weights)
    unique_keys, inverse = np.unique(stacked_keys, axis=0, return_inverse=True)
    support = np.bincount(inverse, minlength=unique_keys.shape[0]).astype(np.int32)
    total_weight = np.bincount(
        inverse,
        weights=stacked_weights,
        minlength=unique_keys.shape[0],
    )
    total_weight = np.maximum(total_weight, 1e-12)
    fused_points = np.column_stack(
        [
            np.bincount(
                inverse,
                weights=stacked_points[:, axis] * stacked_weights,
                minlength=unique_keys.shape[0],
            )
            / total_weight
            for axis in range(3)
        ]
    ).astype(np.float32)

    fused_colors = None
    if has_colors:
        # Missing RGB from an individual source should not tint its geometry
        # black.  Only views that actually carried colors vote on RGB.
        color_rows = []
        color_inverse = []
        color_weights = []
        offset = 0
        for keys, colors, weights_for_view in zip(
            per_view_keys, per_view_colors, per_view_weights
        ):
            length = keys.shape[0]
            if colors is not None:
                color_rows.append(colors.astype(np.float64))
                color_inverse.append(inverse[offset : offset + length])
                color_weights.append(weights_for_view)
            offset += length
        stacked_colors = np.vstack(color_rows)
        color_inverse = np.concatenate(color_inverse)
        color_weights = np.concatenate(color_weights)
        color_total_weight = np.maximum(
            np.bincount(
                color_inverse,
                weights=color_weights,
                minlength=unique_keys.shape[0],
            ),
            1e-12,
        )
        fused_colors = np.column_stack(
            [
                np.bincount(
                    color_inverse,
                    weights=stacked_colors[:, axis] * color_weights,
                    minlength=unique_keys.shape[0],
                )
                / color_total_weight
                for axis in range(3)
            ]
        ).astype(np.float32)

    keep = np.ones((fused_points.shape[0],), dtype=bool)
    radius = max(_finite_optional_float(single_view_radius, 0.008), 0.0)
    min_neighbors = max(int(single_view_min_neighbors), 0)
    if (
        fused_points.shape[0] > 1
        and radius > 0.0
        and min_neighbors > 0
    ):
        tree = cKDTree(fused_points.astype(np.float64))
        local_count = np.asarray(
            tree.query_ball_point(
                fused_points.astype(np.float64),
                r=radius + 1e-9,
                workers=-1,
                return_length=True,
            ),
            dtype=np.int32,
        ) - 1
        keep &= ~((support <= 1) & (local_count < min_neighbors))

    removed_isolated = int(np.count_nonzero(~keep))
    fused = _make_cloud_from_points_colors(
        fused_points[keep],
        None if fused_colors is None else fused_colors[keep],
        get_o3d_hook=get_o3d_hook,
    )
    if bool(filter_outliers):
        try:
            if len(fused.points) > 50:
                fused, _ = fused.remove_statistical_outlier(
                    nb_neighbors=min(int(outlier_neighbors), len(fused.points) - 1),
                    std_ratio=float(outlier_std_ratio),
                )
        except Exception:
            pass
    stats = {
        "input_views": len(clouds),
        "raw_points": raw_points,
        "view_representatives": int(stacked_keys.shape[0]),
        "fused_voxels": int(fused_points.shape[0]),
        "retained_points": int(len(fused.points)),
        "multi_view_voxels": int(np.count_nonzero(support >= 2)),
        "removed_isolated": removed_isolated,
        "voxel_size": voxel,
    }
    print(
        f"[target][fusion] views={stats['input_views']} raw={stats['raw_points']} "
        f"view_voxels={stats['view_representatives']} fused={stats['fused_voxels']} "
        f"multi_view={stats['multi_view_voxels']} removed_isolated={removed_isolated} "
        f"retained={stats['retained_points']} voxel={voxel * 1000.0:.1f}mm"
    )
    return (fused, stats) if return_stats else fused


def _surfel_geometry(points, *, camera_origin=None, neighbors=12):
    """Estimate unoriented normals and conservative geometric confidence.

    The confidence is deliberately bounded away from zero: grazing-angle and
    boundary samples can complete an unseen surface, but they must not pull a
    well-supported frontal surface halfway toward a noisy silhouette.
    """
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    count = int(points.shape[0])
    if count < 4:
        return (
            np.zeros((count, 3), dtype=np.float64),
            np.full((count,), 0.25, dtype=np.float64),
        )
    k = min(max(int(neighbors), 4), count)
    tree = cKDTree(points)
    distance, index = tree.query(points, k=k, workers=-1)
    local = points[index] - points[:, None, :]
    covariance = np.einsum("nki,nkj->nij", local, local) / float(k)
    _values, vectors = np.linalg.eigh(covariance)
    normals = vectors[:, :, 0]

    support_distance = np.asarray(distance[:, -1], dtype=np.float64)
    finite_support = support_distance[
        np.isfinite(support_distance) & (support_distance > 1e-9)
    ]
    typical = (
        float(np.median(finite_support))
        if finite_support.size
        else 1.0
    )
    density = np.clip(typical / np.maximum(support_distance, 1e-9), 0.20, 1.0)
    confidence = 0.20 + 0.80 * density

    if camera_origin is not None:
        origin = np.asarray(camera_origin, dtype=np.float64).reshape(3)
        rays = origin[None, :] - points
        ray_norm = np.linalg.norm(rays, axis=1)
        valid = ray_norm > 1e-9
        rays[valid] /= ray_norm[valid, None]
        incidence = np.abs(np.einsum("ij,ij->i", normals, rays))
        incidence = np.clip((incidence - 0.10) / 0.80, 0.0, 1.0)
        confidence *= 0.20 + 0.80 * incidence
    return normals, np.clip(confidence, 0.05, 1.0)


def _surfel_view_representatives(
    cloud,
    voxel,
    *,
    point_weights=None,
    view_weight=1.0,
    camera_origin=None,
    normal_neighbors=12,
):
    points, colors = _cloud_points_and_colors(cloud)
    if points.shape[0] == 0:
        return points.astype(np.float64), colors, np.zeros((0, 3)), np.zeros((0,))
    if point_weights is None:
        raw_weights = np.ones((points.shape[0],), dtype=np.float64)
    else:
        raw_weights = np.asarray(point_weights, dtype=np.float64).reshape(-1)
        if raw_weights.size != points.shape[0]:
            raise ValueError("surfel point_weights must match finite cloud points")
        raw_weights[~np.isfinite(raw_weights)] = 0.0
        raw_weights = np.clip(raw_weights, 0.0, 1.0)

    keys = np.floor(points.astype(np.float64) / voxel + 1e-5).astype(np.int64)
    unique_keys, inverse = np.unique(keys, axis=0, return_inverse=True)
    weighted_count = np.bincount(
        inverse, weights=np.maximum(raw_weights, 1e-6), minlength=unique_keys.shape[0]
    )
    representatives = np.column_stack(
        [
            np.bincount(
                inverse,
                weights=points[:, axis] * np.maximum(raw_weights, 1e-6),
                minlength=unique_keys.shape[0],
            )
            / np.maximum(weighted_count, 1e-12)
            for axis in range(3)
        ]
    )
    voxel_confidence = np.bincount(
        inverse, weights=raw_weights, minlength=unique_keys.shape[0]
    ) / np.maximum(
        np.bincount(inverse, minlength=unique_keys.shape[0]), 1
    )
    representative_colors = None
    if colors is not None:
        representative_colors = np.column_stack(
            [
                np.bincount(
                    inverse,
                    weights=colors[:, axis] * np.maximum(raw_weights, 1e-6),
                    minlength=unique_keys.shape[0],
                )
                / np.maximum(weighted_count, 1e-12)
                for axis in range(3)
            ]
        )
    normals, geometry_confidence = _surfel_geometry(
        representatives,
        camera_origin=camera_origin,
        neighbors=normal_neighbors,
    )
    confidence = (
        np.clip(voxel_confidence, 0.05, 1.0)
        * geometry_confidence
        * max(float(view_weight), 0.0)
    )
    return representatives, representative_colors, normals, confidence


def fuse_target_clouds_surfel(
    clouds,
    voxel_size=0.0025,
    *,
    merge_radius=0.008,
    normal_angle_deg=55.0,
    normal_neighbors=12,
    single_view_radius=0.008,
    single_view_min_neighbors=2,
    view_weights=None,
    point_weights=None,
    camera_origins=None,
    filter_outliers=False,
    outlier_neighbors=20,
    outlier_std_ratio=2.0,
    return_stats=False,
    return_confidence=False,
    get_o3d_hook=None,
):
    """Fuse target views as confidence-weighted surface elements.

    Unlike same-voxel averaging, mutually-nearest compatible surface samples
    may merge across a small metric radius.  Unmatched samples remain in the
    result, so a side view can still add genuinely unseen surface rather than
    being forced onto the reference view.
    """
    clouds = tuple(clouds)
    count = len(clouds)
    voxel = float(voxel_size)
    radius = max(float(merge_radius), voxel)
    if not np.isfinite(voxel) or voxel <= 0.0:
        raise ValueError("surfel voxel size must be finite and positive")
    weights = np.ones((count,), dtype=np.float64) if view_weights is None else np.asarray(view_weights, dtype=np.float64).reshape(-1)
    if weights.size != count:
        raise ValueError("surfel view_weights must match clouds")
    if point_weights is None:
        point_weights = [None] * count
    if len(point_weights) != count:
        raise ValueError("surfel point_weights must match clouds")
    if camera_origins is None:
        camera_origins = [None] * count
    if len(camera_origins) != count:
        raise ValueError("surfel camera_origins must match clouds")

    prepared = [
        _surfel_view_representatives(
            cloud,
            voxel,
            point_weights=pw,
            view_weight=vw,
            camera_origin=origin,
            normal_neighbors=normal_neighbors,
        )
        for cloud, pw, vw, origin in zip(clouds, point_weights, weights, camera_origins)
    ]
    prepared = [item for item in prepared if item[0].shape[0] > 0]
    if not prepared:
        empty = _make_cloud_from_points_colors([], get_o3d_hook=get_o3d_hook)
        stats = {"input_views": count, "retained_points": 0, "merged_pairs": 0}
        if return_stats and return_confidence:
            return empty, stats, np.zeros((0,), dtype=np.float64)
        if return_stats:
            return empty, stats
        if return_confidence:
            return empty, np.zeros((0,), dtype=np.float64)
        return empty

    points, colors, normals, confidence = [np.asarray(value).copy() if value is not None else None for value in prepared[0]]
    support = np.ones((points.shape[0],), dtype=np.int32)
    merged_pairs = 0
    cos_limit = float(np.cos(np.deg2rad(float(normal_angle_deg))))
    for new_points, new_colors, new_normals, new_confidence in prepared[1:]:
        tree = cKDTree(points)
        distance, nearest = tree.query(new_points, k=1, workers=-1)
        reverse = cKDTree(new_points).query(points, k=1, workers=-1)[1]
        reciprocal = reverse[nearest] == np.arange(new_points.shape[0])
        normal_ok = np.abs(np.einsum("ij,ij->i", new_normals, normals[nearest])) >= cos_limit
        matched = np.flatnonzero((distance <= radius) & reciprocal & normal_ok)
        for source_index in matched:
            target_index = int(nearest[source_index])
            old_weight = float(confidence[target_index])
            new_weight = float(new_confidence[source_index])
            total = max(old_weight + new_weight, 1e-12)
            points[target_index] = (
                points[target_index] * old_weight
                + new_points[source_index] * new_weight
            ) / total
            combined_normal = (
                normals[target_index] * old_weight
                + np.sign(np.dot(normals[target_index], new_normals[source_index]))
                * new_normals[source_index]
                * new_weight
            )
            norm = float(np.linalg.norm(combined_normal))
            if norm > 1e-9:
                normals[target_index] = combined_normal / norm
            if colors is not None and new_colors is not None:
                colors[target_index] = (
                    colors[target_index] * old_weight
                    + new_colors[source_index] * new_weight
                ) / total
            confidence[target_index] = total
            support[target_index] += 1
        unmatched = np.ones((new_points.shape[0],), dtype=bool)
        unmatched[matched] = False
        if np.any(unmatched):
            points = np.vstack((points, new_points[unmatched]))
            normals = np.vstack((normals, new_normals[unmatched]))
            confidence = np.concatenate((confidence, new_confidence[unmatched]))
            support = np.concatenate((support, np.ones(np.count_nonzero(unmatched), dtype=np.int32)))
            if colors is not None and new_colors is not None:
                colors = np.vstack((colors, new_colors[unmatched]))
            elif colors is not None:
                colors = np.vstack((colors, np.zeros((np.count_nonzero(unmatched), 3))))
        merged_pairs += int(matched.size)

    keep = np.ones((points.shape[0],), dtype=bool)
    isolated_radius = max(float(single_view_radius), 0.0)
    minimum = max(int(single_view_min_neighbors), 0)
    if points.shape[0] > 1 and isolated_radius > 0.0 and minimum > 0:
        local_count = np.asarray(
            cKDTree(points).query_ball_point(
                points, isolated_radius + 1e-9, workers=-1, return_length=True
            ),
            dtype=np.int32,
        ) - 1
        keep &= ~((support <= 1) & (local_count < minimum))

    retained_confidence = confidence[keep]
    fused = _make_cloud_from_points_colors(
        points[keep], None if colors is None else colors[keep], get_o3d_hook=get_o3d_hook
    )
    if bool(filter_outliers):
        try:
            if len(fused.points) > 50:
                fused, retained_indices = fused.remove_statistical_outlier(
                    nb_neighbors=min(int(outlier_neighbors), len(fused.points) - 1),
                    std_ratio=float(outlier_std_ratio),
                )
                retained_confidence = retained_confidence[
                    np.asarray(retained_indices, dtype=np.int64)
                ]
        except Exception:
            pass
    stats = {
        "input_views": count,
        "view_representatives": int(sum(item[0].shape[0] for item in prepared)),
        "merged_pairs": int(merged_pairs),
        "multi_view_surfels": int(np.count_nonzero(support[keep] >= 2)),
        "removed_isolated": int(np.count_nonzero(~keep)),
        "retained_points": int(len(fused.points)),
        "voxel_size": voxel,
        "merge_radius": radius,
        "confidence_median": float(np.median(retained_confidence)) if retained_confidence.size else 0.0,
    }
    print(
        f"[target][surfel] views={count} representatives={stats['view_representatives']} "
        f"merged_pairs={merged_pairs} multi_view={stats['multi_view_surfels']} "
        f"retained={stats['retained_points']} radius={radius * 1000.0:.1f}mm"
    )
    if return_stats and return_confidence:
        return fused, stats, retained_confidence
    if return_stats:
        return fused, stats
    if return_confidence:
        return fused, retained_confidence
    return fused


def _finite_optional_float(value, default=None):
    if value is None:
        return default
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    return value if np.isfinite(value) else default


def _environment_points_base(cloud):
    """Read either an Open3D-like cloud or an ``(N, 3)`` point array."""
    if cloud is None:
        return np.zeros((0, 3), dtype=np.float32)
    source = cloud.points if hasattr(cloud, "points") else cloud
    points = np.asarray(source, dtype=np.float32).reshape(-1, 3)
    return points[np.isfinite(points).all(axis=1)]


def _dynamic_environment_roi_bounds(
    args,
    target_base,
    target_center_base,
    voxel_size,
):
    """Resolve a base-frame table/workspace ROI without requiring new CLI args."""
    target = np.asarray(target_base, dtype=np.float32).reshape(-1, 3)
    center = None
    if target_center_base is not None:
        candidate = np.asarray(target_center_base, dtype=float).reshape(3)
        if np.isfinite(candidate).all():
            center = candidate
    if center is None and target.shape[0] > 0:
        candidate = np.median(target.astype(float), axis=0)
        if np.isfinite(candidate).all():
            center = candidate

    required_positions = []
    if center is not None:
        required_positions.append(("target_center", center.copy()))
    placement_positions = []
    for attribute in ("place_xyz", "clear_place_xyz"):
        value = getattr(args, attribute, None)
        if value is None:
            continue
        try:
            place = np.asarray(value, dtype=float).reshape(-1)
        except (TypeError, ValueError):
            continue
        if place.size < 2 or not np.isfinite(place[:2]).all():
            continue
        placement_positions.append((attribute, place.copy()))
        if place.size >= 3 and np.isfinite(place[:3]).all():
            required_positions.append((attribute, place[:3].copy()))

    bounds = np.array(
        [
            [-np.inf, -np.inf, -np.inf],
            [np.inf, np.inf, np.inf],
        ],
        dtype=float,
    )
    radius_xy = _finite_optional_float(
        getattr(args, "motion_environment_map_radius_xy", 0.45),
        0.45,
    )
    automatic_xy = bool(
        center is not None and radius_xy is not None and radius_xy > 0.0
    )
    if automatic_xy:
        bounds[0, :2] = center[:2] - radius_xy
        bounds[1, :2] = center[:2] + radius_xy

        # Transport and clear placement are part of the same frozen workflow
        # map.  Do not crop their endpoints out merely because they fall just
        # beyond a target-centred ROI.
        place_margin = max(float(voxel_size), 0.05)
        for _attribute, place in placement_positions:
            bounds[0, :2] = np.minimum(
                bounds[0, :2], place[:2] - place_margin
            )
            bounds[1, :2] = np.maximum(
                bounds[1, :2], place[:2] + place_margin
            )

    # Explicit bounds deliberately win over all automatically expanded bounds.
    explicit = (
        (0, 0, "motion_environment_map_x_min"),
        (1, 0, "motion_environment_map_x_max"),
        (0, 1, "motion_environment_map_y_min"),
        (1, 1, "motion_environment_map_y_max"),
    )
    for row, axis, attribute in explicit:
        value = _finite_optional_float(getattr(args, attribute, None), None)
        if value is not None:
            bounds[row, axis] = value

    table_z = _finite_optional_float(getattr(args, "table_z", None), None)
    if table_z is not None:
        below = max(
            _finite_optional_float(
                getattr(args, "motion_environment_map_z_below_table", 0.02),
                0.02,
            ),
            0.0,
        )
        above = max(
            _finite_optional_float(
                getattr(args, "motion_environment_map_z_above_table", 0.60),
                0.60,
            ),
            0.0,
        )
        bounds[0, 2] = table_z - below
        bounds[1, 2] = table_z + above

    if np.any(bounds[0] > bounds[1]):
        raise ValueError(
            "dynamic environment ROI minimum exceeds maximum: "
            f"{bounds.tolist()}"
        )
    tolerance = max(float(voxel_size), 1e-6)
    for label, position in required_positions:
        position = np.asarray(position, dtype=float).reshape(-1)
        dimensions = min(position.size, 3)
        outside = [
            axis
            for axis in range(dimensions)
            if (
                position[axis] < bounds[0, axis] - tolerance
                or position[axis] > bounds[1, axis] + tolerance
            )
        ]
        if outside:
            axis_names = "".join("xyz"[axis] for axis in outside)
            raise ValueError(
                f"dynamic environment ROI excludes required {label} on "
                f"axis/axes {axis_names}: point={position[:dimensions].tolist()} "
                f"bounds={bounds[:, :dimensions].tolist()}"
            )
    return bounds


def fuse_dynamic_environment_clouds(
    view_names,
    environment_base_clouds,
    args,
    target_base=None,
    map_id="dynamic_environment",
    generation=0,
    target_center_base=None,
):
    """Build one conservative base-frame occupancy map from dynamic views.

    Each view is voxelised independently, so duplicate samples in a dense
    capture count once and ``view_support`` measures distinct observations.
    Multi-view voxels are hard occupancy.  A single-view voxel is also hard
    when it belongs to a locally continuous thin structure (for example a
    basket rim); isolated low-support samples remain in the uncertain layer.

    The input clouds are already in ``base_link``.  No target-derived
    registration transform is applied here by design.
    """
    if (str(getattr(args, "trajectory_executor", "move_j")) == "mit"
            and not bool(getattr(args, "_planning_worker", False))):
        from .planning_process import isolated_environment_map
        return isolated_environment_map(
            view_names, environment_base_clouds, args, target_base=target_base,
            map_id=map_id, generation=generation, target_center_base=target_center_base,
        )
    names = tuple(str(name) for name in view_names)
    clouds = tuple(environment_base_clouds)
    if len(names) != len(clouds):
        raise ValueError(
            "view_names and environment_base_clouds must have equal length: "
            f"{len(names)} != {len(clouds)}"
        )
    if len(set(names)) != len(names):
        raise ValueError("dynamic environment source view names must be unique")
    try:
        generation_value = float(generation)
    except (TypeError, ValueError) as exc:
        raise ValueError("dynamic environment generation must be an integer") from exc
    if not np.isfinite(generation_value):
        raise ValueError(
            "dynamic environment generation must be a finite non-negative integer"
        )
    if not np.isfinite(generation_value) or generation_value < 0.0:
        raise ValueError(
            "dynamic environment generation must be a finite non-negative integer"
        )
    generation_int = int(generation_value)
    if float(generation_int) != generation_value:
        raise ValueError(
            "dynamic environment generation must be a finite non-negative integer"
        )

    voxel_size = _finite_optional_float(
        getattr(args, "motion_environment_map_voxel", None),
        None,
    )
    if voxel_size is None:
        voxel_size = _finite_optional_float(
            getattr(args, "motion_path_scene_voxel", None),
            None,
        )
    if voxel_size is None:
        voxel_size = _finite_optional_float(
            getattr(args, "scene_voxel_size", None),
            0.005,
        )
    if voxel_size is None or voxel_size <= 0.0:
        raise ValueError("motion environment map voxel size must be positive")

    target = (
        np.zeros((0, 3), dtype=np.float32)
        if target_base is None
        else _environment_points_base(target_base)
    )
    roi_bounds = _dynamic_environment_roi_bounds(
        args,
        target,
        target_center_base,
        voxel_size,
    )

    per_view_keys = []
    # Keep a physically measured representative for each occupied voxel.
    # Voxel centres remain useful for support and neighborhood calculations,
    # but using a synthetic centre as collision geometry can shift a measured
    # surface by half a voxel.
    representative_by_key = {}
    raw_point_count = 0
    roi_point_count = 0
    for cloud in clouds:
        points = _environment_points_base(cloud)
        raw_point_count += int(points.shape[0])
        if points.shape[0] > 0:
            keep = np.all(
                (points >= roi_bounds[0].reshape(1, 3))
                & (points <= roi_bounds[1].reshape(1, 3)),
                axis=1,
            )
            points = points[keep]
        roi_point_count += int(points.shape[0])
        if points.shape[0] == 0:
            per_view_keys.append(np.zeros((0, 3), dtype=np.int64))
            continue
        scaled = points.astype(np.float64) / float(voxel_size)
        # Float32 RGB-D coordinates that lie exactly on a voxel boundary can
        # evaluate a few ulps below the integer (for example 2.9999999).  A
        # dimensionless epsilon keeps boundary ownership deterministic without
        # moving any physically distinguishable point.
        keys = np.floor(scaled + 1e-5).astype(np.int64)
        unique_keys, unique_indices = np.unique(
            keys,
            axis=0,
            return_index=True,
        )
        per_view_keys.append(unique_keys)
        for key, point_index in zip(unique_keys, unique_indices):
            key_tuple = tuple(int(value) for value in key)
            representative_by_key.setdefault(
                key_tuple,
                points[int(point_index)].copy(),
            )

    nonempty_keys = [keys for keys in per_view_keys if keys.shape[0] > 0]
    if nonempty_keys:
        voxel_keys, view_support = np.unique(
            np.vstack(nonempty_keys),
            axis=0,
            return_counts=True,
        )
        # Every per-view array was uniqued above, therefore this count is the
        # number of distinct source views rather than raw sample density.
        view_support = np.asarray(view_support, dtype=np.int32)
    else:
        voxel_keys = np.zeros((0, 3), dtype=np.int64)
        view_support = np.zeros((0,), dtype=np.int32)

    key_to_index = {
        tuple(int(value) for value in key): index
        for index, key in enumerate(voxel_keys)
    }

    min_view_support = max(
        int(getattr(args, "motion_environment_map_min_view_support", 2)),
        1,
    )
    hard_mask = view_support >= min_view_support
    continuous_single_view = np.zeros_like(hard_mask, dtype=bool)
    neighbor_supported_single_view = np.zeros_like(hard_mask, dtype=bool)

    local_radius = _finite_optional_float(
        getattr(
            args,
            "motion_environment_map_single_view_local_radius",
            0.020,
        ),
        0.020,
    )
    neighbor_radius = _finite_optional_float(
        getattr(
            args,
            "motion_environment_map_single_view_neighbor_radius",
            0.009,
        ),
        0.009,
    )
    local_radius = max(float(local_radius), 0.0)
    neighbor_radius = max(float(neighbor_radius), 0.0)
    min_neighbors = max(
        int(
            getattr(
                args,
                "motion_environment_map_single_view_min_neighbors",
                2,
            )
        ),
        1,
    )
    if local_radius > 0.0 and neighbor_radius > 0.0:
        for keys in per_view_keys:
            if keys.shape[0] <= min_neighbors:
                continue
            centers = (keys.astype(np.float64) + 0.5) * float(voxel_size)
            tree = cKDTree(centers)
            # Request counts rather than Python lists of every neighbor.  A
            # dense table plane can contain tens of thousands of voxels, and
            # materialising all neighbor lists would otherwise dominate map
            # memory even though classification only needs the count.
            local_neighbor_count = np.asarray(
                tree.query_ball_point(
                    centers,
                    r=local_radius + 1e-9,
                    workers=-1,
                    return_length=True,
                ),
                dtype=np.int32,
            ) - 1
            nearest_distance, _ = tree.query(centers, k=2, workers=-1)
            nearest_other = np.asarray(nearest_distance, dtype=float)[:, 1]
            for local_index, key in enumerate(keys):
                global_index = key_to_index[
                    tuple(int(value) for value in key)
                ]
                if hard_mask[global_index]:
                    continue
                if (
                    int(local_neighbor_count[local_index]) >= 1
                    and np.isfinite(nearest_other[local_index])
                    and nearest_other[local_index] <= neighbor_radius + 1e-9
                ):
                    neighbor_supported_single_view[global_index] = True
                if (
                    int(local_neighbor_count[local_index]) >= min_neighbors
                    and np.isfinite(nearest_other[local_index])
                    and nearest_other[local_index] <= neighbor_radius + 1e-9
                ):
                    continuous_single_view[global_index] = True

    # Ghost-offset suppression.  A single-view continuous surface is usually a
    # genuine thin structure (for example a basket rim), but when it sits
    # directly above or below multi-view occupancy at the same XY it is far
    # more likely the same physical surface re-observed at a biased height by
    # one view (a straight-down eye-in-hand height bias is the classic case).
    # Such duplicate copies inflate the obstacle map and can reject every
    # grasp approach as a false collision, so drop them instead of promoting
    # them to hard occupancy.  The vertical window is bounded by the
    # registration-error scale, not the map size, so genuinely stacked
    # structures well above/below the table are preserved.
    ghost_z_window = _finite_optional_float(
        getattr(args, "motion_environment_map_ghost_z_window", 0.015),
        0.015,
    )
    ghost_z_window = max(float(ghost_z_window), 0.0)
    if ghost_z_window > 0.0 and np.any(continuous_single_view):
        ghost_z_voxels = max(int(np.ceil(ghost_z_window / float(voxel_size))), 1)
        ghost_mask = np.zeros_like(hard_mask, dtype=bool)
        for global_index in np.flatnonzero(continuous_single_view):
            key = voxel_keys[global_index]
            ix, iy, iz = int(key[0]), int(key[1]), int(key[2])
            shadowed = False
            for dz in range(1, ghost_z_voxels + 1):
                below = key_to_index.get((ix, iy, iz - dz))
                if below is not None and hard_mask[below]:
                    shadowed = True
                    break
                above = key_to_index.get((ix, iy, iz + dz))
                if above is not None and hard_mask[above]:
                    shadowed = True
                    break
            if shadowed:
                ghost_mask[global_index] = True
        if np.any(ghost_mask):
            suppressed = int(np.count_nonzero(ghost_mask))
            keep = ~ghost_mask
            voxel_keys = voxel_keys[keep]
            view_support = view_support[keep]
            hard_mask = hard_mask[keep]
            continuous_single_view = continuous_single_view[keep]
            neighbor_supported_single_view = neighbor_supported_single_view[keep]
            print(
                f"[environment][ghost] map={map_id} dropped {suppressed} "
                f"single-view voxel(s) shadowed by multi-view occupancy within "
                f"{ghost_z_window * 1000.0:.0f}mm vertically"
            )

    hard_mask |= continuous_single_view
    isolated_filter = bool(
        getattr(args, "motion_environment_map_isolated_filter", True)
    )
    isolated_mask = (
        (view_support < min_view_support)
        & ~neighbor_supported_single_view
        & ~continuous_single_view
    )
    dropped_isolated_count = 0
    if isolated_filter and np.any(isolated_mask):
        keep = ~isolated_mask
        dropped_isolated_count = int(np.count_nonzero(isolated_mask))
        voxel_keys = voxel_keys[keep]
        view_support = view_support[keep]
        hard_mask = hard_mask[keep]
        continuous_single_view = continuous_single_view[keep]

    dropped_small_component_count = 0
    dropped_small_component_voxels = 0
    if (
        bool(getattr(args, "motion_small_isolated_component_filter", True))
        and voxel_keys.shape[0] > 0
    ):
        provisional_centers = (
            (voxel_keys.astype(np.float64) + 0.5) * float(voxel_size)
        ).astype(np.float32)
        component_keep, component_stats = small_isolated_component_keep_mask(
            provisional_centers,
            connection_radius=float(
                getattr(args, "motion_small_isolated_component_radius", 0.020)
            ),
            max_component_size=int(
                getattr(args, "motion_small_isolated_component_max_voxels", 8)
            ),
        )
        if np.any(component_keep):
            dropped_small_component_count = int(
                component_stats["removed_components"]
            )
            dropped_small_component_voxels = int(
                component_stats["removed_points"]
            )
            voxel_keys = voxel_keys[component_keep]
            view_support = view_support[component_keep]
            hard_mask = hard_mask[component_keep]
            continuous_single_view = continuous_single_view[component_keep]
        elif component_stats["removed_points"]:
            print(
                f"[environment][isolated][warn] map={map_id} small-component "
                "filter would remove the entire environment map; retain it "
                "fail-closed"
            )
    centers = (
        (voxel_keys.astype(np.float64) + 0.5) * float(voxel_size)
    ).astype(np.float32)
    representatives = np.asarray(
        [
            representative_by_key.get(
                tuple(int(value) for value in key),
                center,
            )
            for key, center in zip(voxel_keys, centers)
        ],
        dtype=np.float32,
    ).reshape(-1, 3)
    confidence = np.clip(
        view_support.astype(np.float32) / float(min_view_support),
        0.0,
        1.0,
    )
    confidence[continuous_single_view] = np.maximum(
        confidence[continuous_single_view],
        np.float32(0.75),
    )

    hard_points = representatives[hard_mask]
    uncertain_points = representatives[~hard_mask]
    result = DynamicEnvironmentMap(
        hard_environment_base=hard_points,
        uncertain_environment_base=uncertain_points,
        target_base=None if target_base is None else target,
        voxel_keys=voxel_keys,
        view_support=view_support,
        confidence=confidence,
        source_view_names=names,
        roi_bounds_base=roi_bounds,
        frame_id="base_link",
        map_id=str(map_id),
        generation=generation_int,
        voxel_size=float(voxel_size),
    )
    printable_bounds = [
        [
            None if not np.isfinite(value) else round(float(value), 6)
            for value in row
        ]
        for row in roi_bounds
    ]
    print(
        f"[environment][map] map={result.map_id} generation={result.generation} "
        f"frame={result.frame_id} views={len(names)} raw_points={raw_point_count} "
        f"roi_points={roi_point_count} voxels={voxel_keys.shape[0]} "
        f"hard={hard_points.shape[0]} uncertain={uncertain_points.shape[0]} "
        f"dropped_isolated={dropped_isolated_count} "
        f"dropped_small_components={dropped_small_component_count} "
        f"dropped_small_voxels={dropped_small_component_voxels} "
        f"voxel={voxel_size:.4f}m bounds={printable_bounds}"
    )
    return result


def save_cloud(path, cloud, *, get_o3d_hook=None):
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        o3d_loader = get_o3d if get_o3d_hook is None else get_o3d_hook
        o3d_loader().io.write_point_cloud(path, cloud)
        print(f"[cloud] saved {path}")
    except Exception as exc:
        print(
            f"[cloud][warn] failed to save {path}: "
            f"{type(exc).__name__}: {exc}"
        )


def _cloud_stats(cloud):
    pts = np.asarray(cloud.points, dtype=float).reshape(-1, 3)
    if pts.shape[0] == 0:
        return {
            "n": 0,
            "median": np.full(3, np.nan),
            "mean": np.full(3, np.nan),
            "min": np.full(3, np.nan),
            "max": np.full(3, np.nan),
            "extent": np.full(3, np.nan),
            "pca_axis": np.full(3, np.nan),
        }
    pmin = np.min(pts, axis=0)
    pmax = np.max(pts, axis=0)
    pca_axis = np.full(3, np.nan)
    if pts.shape[0] >= 6:
        centered = pts - np.mean(pts, axis=0, keepdims=True)
        cov = centered.T @ centered / max(float(pts.shape[0] - 1), 1.0)
        try:
            vals, vecs = np.linalg.eigh(cov)
            pca_axis = vecs[:, int(np.argmax(vals))]
            pca_axis = pca_axis / max(float(np.linalg.norm(pca_axis)), 1e-9)
        except Exception:
            pass
    return {
        "n": int(pts.shape[0]),
        "median": np.median(pts, axis=0),
        "mean": np.mean(pts, axis=0),
        "min": pmin,
        "max": pmax,
        "extent": pmax - pmin,
        "pca_axis": pca_axis,
    }


def _nn_stats(src_pts, tgt_pts):
    src = np.asarray(src_pts, dtype=np.float32).reshape(-1, 3)
    tgt = np.asarray(tgt_pts, dtype=np.float32).reshape(-1, 3)
    if src.shape[0] == 0 or tgt.shape[0] == 0:
        return {
            "median": float("nan"),
            "p95": float("nan"),
            "max": float("nan"),
            "inlier@1cm": 0.0,
        }
    tree = cKDTree(tgt)
    d, _ = tree.query(src, k=1, workers=-1)
    d = np.asarray(d, dtype=np.float32)
    return {
        "median": float(np.median(d)),
        "p95": float(np.quantile(d, 0.95)),
        "max": float(np.max(d)),
        "inlier@1cm": float(np.mean(d < 0.01)),
    }

def _mutual_overlap_stats(src_pts, tgt_pts, max_distance):
    src = np.asarray(src_pts, dtype=np.float32).reshape(-1, 3)
    tgt = np.asarray(tgt_pts, dtype=np.float32).reshape(-1, 3)
    empty = {
        "pairs": 0,
        "median": float("inf"),
        "p95": float("inf"),
        "max": float("inf"),
        "inlier@1cm": 0.0,
        "overlap_ratio": 0.0,
        "source_overlap_ratio": 0.0,
        "target_overlap_ratio": 0.0,
        "novel_source_ratio": 1.0,
    }
    if src.shape[0] == 0 or tgt.shape[0] == 0:
        return empty

    target_tree = cKDTree(tgt)
    distances, target_ids = target_tree.query(
        src,
        k=1,
        workers=-1,
    )
    distances = np.asarray(distances, dtype=float)
    target_ids = np.asarray(target_ids, dtype=int)

    valid = (
        np.isfinite(distances)
        & (distances <= float(max_distance))
    )
    source_ids = np.flatnonzero(valid)
    if source_ids.size == 0:
        return empty

    source_tree = cKDTree(src)
    _, reverse_source_ids = source_tree.query(
        tgt,
        k=1,
        workers=-1,
    )
    reverse_source_ids = np.asarray(reverse_source_ids, dtype=int)
    mutual = (
        reverse_source_ids[target_ids[source_ids]]
        == source_ids
    )
    source_ids = source_ids[mutual]
    if source_ids.size == 0:
        return empty

    mutual_distances = distances[source_ids]
    pair_count = int(mutual_distances.size)
    source_ratio = pair_count / float(max(src.shape[0], 1))
    target_ratio = pair_count / float(max(tgt.shape[0], 1))
    return {
        "pairs": pair_count,
        "median": float(np.median(mutual_distances)),
        "p95": float(np.quantile(mutual_distances, 0.95)),
        "max": float(np.max(mutual_distances)),
        "inlier@1cm": float(np.mean(mutual_distances <= 0.010)),
        "overlap_ratio": float(
            pair_count / max(min(src.shape[0], tgt.shape[0]), 1)
        ),
        "source_overlap_ratio": float(source_ratio),
        "target_overlap_ratio": float(target_ratio),
        "novel_source_ratio": float(np.clip(1.0 - source_ratio, 0.0, 1.0)),
    }


def _nn_gate(src_pts, tgt_pts, thr):
    src = np.asarray(src_pts, dtype=np.float32).reshape(-1, 3)
    tgt = np.asarray(tgt_pts, dtype=np.float32).reshape(-1, 3)
    if src.shape[0] == 0 or tgt.shape[0] == 0:
        return src, np.zeros((src.shape[0],), dtype=bool)
    tree = cKDTree(tgt)
    d, _ = tree.query(src, k=1, workers=-1)
    keep = np.isfinite(d) & (d <= float(thr))
    return src[keep], keep


def _apply_T_points(points, T):
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    T = np.asarray(T, dtype=float).reshape(4, 4)
    if pts.shape[0] == 0:
        return pts
    ph = np.c_[pts.astype(float), np.ones((pts.shape[0],), dtype=float)]
    return (ph @ T.T)[:, :3].astype(np.float32)


def _downsample_points(points, voxel, *, get_o3d_hook=None):
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    if pts.shape[0] == 0 or voxel is None or float(voxel) <= 0.0:
        return pts
    o3d = (get_o3d if get_o3d_hook is None else get_o3d_hook)()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts.astype(float))
    pcd = pcd.voxel_down_sample(float(voxel))
    return np.asarray(pcd.points, dtype=np.float32).reshape(-1, 3)


def _small_translation_registration(
    src_pts,
    tgt_pts,
    args,
    *,
    nn_stats_hook=None,
    apply_T_points_hook=None,
):
    nn_stats = _nn_stats if nn_stats_hook is None else nn_stats_hook
    apply_T_points = (
        _apply_T_points if apply_T_points_hook is None else apply_T_points_hook
    )
    src = np.asarray(src_pts, dtype=np.float32).reshape(-1, 3)
    tgt = np.asarray(tgt_pts, dtype=np.float32).reshape(-1, 3)
    before = nn_stats(src, tgt)
    T = np.eye(4, dtype=float)
    if (
        src.shape[0] < int(args.active_reg_min_pairs)
        or tgt.shape[0] < int(args.active_reg_min_pairs)
    ):
        return T, {
            "applied": False,
            "reason": "too_few_points",
            "pairs": 0,
            "shift_norm": 0.0,
            "before": before,
            "after": before,
        }

    tree = cKDTree(tgt)
    work_src = src.copy()
    total_delta = np.zeros(3, dtype=float)
    pair_count = 0
    trimmed_pair_count = 0
    iters_run = 0
    trim_q = float(
        np.clip(
            float(getattr(args, "active_reg_trim_quantile", 1.0)),
            0.2,
            1.0,
        )
    )
    max_total = float(args.active_reg_max_shift)
    max_xy = float(getattr(args, "active_reg_max_shift_xy", 0.0))
    max_z = float(getattr(args, "active_reg_max_shift_z", 0.0))

    for _ in range(
        max(int(getattr(args, "active_reg_translation_iters", 1)), 1)
    ):
        d, nn = tree.query(work_src, k=1, workers=-1)
        d = np.asarray(d, dtype=np.float32)
        keep = np.isfinite(d) & (d <= float(args.active_reg_nn_thr))
        pair_count = int(np.count_nonzero(keep))
        if pair_count < int(args.active_reg_min_pairs):
            if iters_run == 0:
                return T, {
                    "applied": False,
                    "reason": "too_few_pairs",
                    "pairs": pair_count,
                    "shift_norm": 0.0,
                    "before": before,
                    "after": before,
                }
            break

        matched_tgt = tgt[np.asarray(nn[keep], dtype=np.int64)]
        matched_src = work_src[keep]
        d_keep = d[keep]
        if trim_q < 0.999 and d_keep.size > int(args.active_reg_min_pairs):
            trim_thr = float(np.quantile(d_keep, trim_q))
            trim_keep = d_keep <= trim_thr
            if int(np.count_nonzero(trim_keep)) >= int(
                args.active_reg_min_pairs
            ):
                matched_tgt = matched_tgt[trim_keep]
                matched_src = matched_src[trim_keep]
                d_keep = d_keep[trim_keep]
        trimmed_pair_count = int(matched_src.shape[0])

        step_delta = np.median(
            matched_tgt.astype(float) - matched_src.astype(float),
            axis=0,
        )
        step_norm = float(np.linalg.norm(step_delta))
        if (not np.isfinite(step_norm)) or step_norm <= 1e-6:
            break

        unclipped_total = total_delta + step_delta
        clipped_total = unclipped_total.copy()
        total_norm = float(np.linalg.norm(clipped_total))
        if total_norm > max_total:
            clipped_total *= max_total / max(total_norm, 1e-9)
        if max_xy > 0.0:
            xy_norm = float(np.linalg.norm(clipped_total[:2]))
            if xy_norm > max_xy:
                clipped_total[:2] *= max_xy / max(xy_norm, 1e-9)
        if max_z > 0.0:
            clipped_total[2] = float(
                np.clip(clipped_total[2], -max_z, max_z)
            )

        if float(np.linalg.norm(clipped_total - total_delta)) <= 1e-6:
            break
        total_delta = clipped_total
        T[:3, 3] = total_delta.reshape(3)
        work_src = apply_T_points(src, T)
        iters_run += 1

    delta = total_delta
    shift_norm = float(np.linalg.norm(delta))
    if (not np.isfinite(shift_norm)) or shift_norm <= 1e-6:
        return T, {
            "applied": False,
            "reason": "zero_shift",
            "pairs": pair_count,
            "shift_norm": shift_norm,
            "before": before,
            "after": before,
        }
    T[:3, 3] = delta.reshape(3)
    calibrated = apply_T_points(src, T)
    after = nn_stats(calibrated, tgt)
    before_med = float(before.get("median", np.inf))
    after_med = float(after.get("median", np.inf))
    before_in = float(before.get("inlier@1cm", 0.0))
    after_in = float(after.get("inlier@1cm", 0.0))
    med_improve = (
        (before_med - after_med) / max(before_med, 1e-6)
        if np.isfinite(before_med)
        else 0.0
    )
    accept = bool(
        np.isfinite(after_med)
        and after_in + 0.03 >= before_in
        and (
            med_improve >= float(args.active_reg_min_improve)
            or after_in >= before_in + 0.08
            or (after_med <= 0.010 and after_in >= before_in)
        )
    )
    if not accept:
        T = np.eye(4, dtype=float)
    return T, {
        "applied": accept,
        "reason": "accepted" if accept else "no_improvement",
        "pairs": pair_count,
        "trimmed_pairs": trimmed_pair_count,
        "trim_quantile": trim_q,
        "iters": iters_run,
        "shift": delta.astype(np.float32),
        "shift_norm": shift_norm,
        "before": before,
        "after": after,
    }


def _rigid_icp_registration(
    src_pts,
    tgt_pts,
    args,
    *,
    get_o3d_hook=None,
    nn_stats_hook=None,
    nn_gate_hook=None,
    apply_T_points_hook=None,
):
    nn_stats = _nn_stats if nn_stats_hook is None else nn_stats_hook
    nn_gate = _nn_gate if nn_gate_hook is None else nn_gate_hook
    apply_T_points = (
        _apply_T_points if apply_T_points_hook is None else apply_T_points_hook
    )
    src = np.asarray(src_pts, dtype=np.float32).reshape(-1, 3)
    tgt = np.asarray(tgt_pts, dtype=np.float32).reshape(-1, 3)
    before = nn_stats(src, tgt)
    T_identity = np.eye(4, dtype=float)
    if (
        src.shape[0] < int(args.active_reg_min_pairs)
        or tgt.shape[0] < int(args.active_reg_min_pairs)
    ):
        return T_identity, {
            "applied": False,
            "reason": "too_few_points",
            "fit": 0.0,
            "rmse": 0.0,
            "trans": 0.0,
            "rot_deg": 0.0,
            "before": before,
            "after": before,
        }
    overlap_pts, _ = nn_gate(
        src,
        tgt,
        float(args.active_reg_icp_overlap_thr),
    )
    overlap_ratio = float(overlap_pts.shape[0] / max(src.shape[0], 1))
    if (
        overlap_pts.shape[0] < int(args.active_reg_min_pairs)
        or overlap_ratio < float(args.active_reg_icp_min_overlap_ratio)
    ):
        return T_identity, {
            "applied": False,
            "reason": "too_few_overlap_pairs",
            "fit": 0.0,
            "rmse": 0.0,
            "trans": 0.0,
            "rot_deg": 0.0,
            "overlap_ratio": overlap_ratio,
            "before": before,
            "after": before,
        }
    cov = (
        np.cov(overlap_pts.astype(float).T)
        if overlap_pts.shape[0] >= 6
        else np.eye(3)
    )
    eigvals = np.sort(np.linalg.eigvalsh(cov))[::-1]
    shape_ratio_21 = float(eigvals[1] / max(eigvals[0], 1e-9))
    shape_ratio_31 = float(eigvals[2] / max(eigvals[0], 1e-9))
    # ICP is unnecessary only when the view already satisfies the *fusion*
    # overlap gate.  The former 6.5 mm shortcut skipped views that were good
    # enough for coarse registration but still failed the stricter 3 mm
    # fusion-median requirement.
    already_good = (
        before["median"]
        <= float(getattr(args, "active_fusion_overlap_max_median", 0.003))
        and before["p95"]
        <= float(getattr(args, "active_fusion_overlap_max_p95", 0.018))
        and before["inlier@1cm"]
        >= float(getattr(args, "active_fusion_overlap_min_inlier_1cm", 0.80))
    )
    degenerate = shape_ratio_21 < 0.06 or shape_ratio_31 < 0.008
    if already_good or degenerate:
        reason = (
            "already_good"
            if already_good
            else f"degenerate_shape(r21={shape_ratio_21:.4f},"
            f"r31={shape_ratio_31:.4f})"
        )
        return T_identity, {
            "applied": False,
            "reason": reason,
            "fit": 0.0,
            "rmse": 0.0,
            "trans": 0.0,
            "rot_deg": 0.0,
            "overlap_ratio": overlap_ratio,
            "before": before,
            "after": before,
        }

    voxel = float(args.active_reg_icp_voxel)
    if voxel > 0.0:
        overlap_pts = _downsample_points(overlap_pts, voxel)
        tgt = _downsample_points(tgt, voxel)
    # Solve around the target itself rather than the base-link origin.  A
    # legitimate 2-degree correction around an object 0.4 m from the robot
    # otherwise appears as a large translation in the homogeneous matrix.
    pivot = np.median(tgt.astype(float), axis=0)
    overlap_local = overlap_pts.astype(float) - pivot.reshape(1, 3)
    target_local = tgt.astype(float) - pivot.reshape(1, 3)
    estimation_name = str(
        getattr(args, "active_reg_icp_estimation", "point_to_plane")
    )
    robust_scale = max(
        float(getattr(args, "active_reg_icp_robust_kernel", 0.005)), 0.0
    )
    target_tree = cKDTree(target_local)
    target_normals = None
    if estimation_name == "point_to_plane":
        # Open3D 0.18's robust point-to-plane binding segfaults in the deployed
        # NumPy/ABI combination.  Estimate target normals and solve the same
        # local robust least-squares update here so an offline registration
        # failure can never terminate the real robot process.
        neighbor_count = min(24, int(target_local.shape[0]))
        _, neighbor_ids = target_tree.query(
            target_local, k=neighbor_count, workers=-1
        )
        target_normals = np.zeros_like(target_local, dtype=float)
        for index, ids in enumerate(np.asarray(neighbor_ids)):
            neighborhood = target_local[np.asarray(ids, dtype=int)]
            centered = neighborhood - np.mean(neighborhood, axis=0, keepdims=True)
            covariance = centered.T @ centered
            values, vectors = np.linalg.eigh(covariance)
            normal = vectors[:, int(np.argmin(values))]
            target_normals[index] = normal / max(float(np.linalg.norm(normal)), 1e-9)

    def huber_weights(residual):
        absolute = np.abs(np.asarray(residual, dtype=float))
        if robust_scale <= 0.0:
            return np.ones_like(absolute)
        return np.where(
            absolute <= robust_scale,
            1.0,
            robust_scale / np.maximum(absolute, 1e-12),
        )

    T = np.eye(4, dtype=float)
    final_fit = 0.0
    final_rmse = 0.0
    thresholds = [
        float(x)
        for x in str(args.active_reg_icp_thresholds).split(",")
        if str(x).strip()
    ]
    if not thresholds:
        thresholds = [float(args.active_reg_icp_thr)]
    for thr in thresholds:
        threshold = float(thr)
        for _ in range(max(int(args.active_reg_icp_max_iter), 1)):
            work = apply_T_points(overlap_local, T).astype(float)
            distances, indices = target_tree.query(work, k=1, workers=-1)
            valid = np.isfinite(distances) & (distances <= threshold)
            if int(np.count_nonzero(valid)) < int(args.active_reg_min_pairs):
                break
            p = work[valid]
            q = target_local[np.asarray(indices[valid], dtype=int)]
            if estimation_name == "point_to_plane":
                normals = target_normals[np.asarray(indices[valid], dtype=int)]
                residual = np.sum(normals * (p - q), axis=1)
                weights = huber_weights(residual)
                A = np.column_stack((np.cross(p, normals), normals))
                sqrt_w = np.sqrt(weights).reshape(-1, 1)
                lhs = A * sqrt_w
                rhs = -residual * sqrt_w[:, 0]
                # A tiny identity prior prevents an elongated/partially seen
                # target from inventing a large unconstrained twist.
                regularizer = np.diag(
                    [2e-3, 2e-3, 2e-3, 5e-4, 5e-4, 5e-4]
                )
                step = np.linalg.solve(
                    lhs.T @ lhs + regularizer,
                    lhs.T @ rhs,
                )
                rot_step = np.asarray(step[:3], dtype=float)
                trans_step = np.asarray(step[3:], dtype=float)
                rot_norm = float(np.linalg.norm(rot_step))
                max_rot_step = float(np.deg2rad(0.5))
                if rot_norm > max_rot_step:
                    rot_step *= max_rot_step / max(rot_norm, 1e-12)
                trans_norm = float(np.linalg.norm(trans_step))
                if trans_norm > 0.002:
                    trans_step *= 0.002 / max(trans_norm, 1e-12)
                delta_T = np.eye(4, dtype=float)
                delta_T[:3, :3] = Rotation.from_rotvec(rot_step).as_matrix()
                delta_T[:3, 3] = trans_step
            else:
                weights = huber_weights(distances[valid])
                weight_sum = max(float(np.sum(weights)), 1e-12)
                p_center = np.sum(p * weights[:, None], axis=0) / weight_sum
                q_center = np.sum(q * weights[:, None], axis=0) / weight_sum
                covariance = (
                    (p - p_center).T
                    @ ((q - q_center) * weights[:, None])
                )
                U, _, Vt = np.linalg.svd(covariance)
                rotation = Vt.T @ U.T
                if np.linalg.det(rotation) < 0.0:
                    Vt[-1, :] *= -1.0
                    rotation = Vt.T @ U.T
                delta_T = np.eye(4, dtype=float)
                delta_T[:3, :3] = rotation
                delta_T[:3, 3] = q_center - rotation @ p_center
                rot_step = Rotation.from_matrix(rotation).as_rotvec()
                trans_step = delta_T[:3, 3]
            T = delta_T @ T
            # Project every numerical update into the conservative search
            # envelope instead of allowing ICP to converge to a distant but
            # visually similar portion of an elongated object and rejecting
            # only at the end.
            total_rotvec = Rotation.from_matrix(T[:3, :3]).as_rotvec()
            total_rot_norm = float(np.linalg.norm(total_rotvec))
            max_total_rot = float(
                np.deg2rad(args.active_reg_icp_max_rot_deg)
            )
            if total_rot_norm > max_total_rot:
                total_rotvec *= max_total_rot / max(total_rot_norm, 1e-12)
                T[:3, :3] = Rotation.from_rotvec(total_rotvec).as_matrix()
            total_trans_norm = float(np.linalg.norm(T[:3, 3]))
            max_total_trans = float(args.active_reg_icp_max_trans)
            if total_trans_norm > max_total_trans:
                T[:3, 3] *= max_total_trans / max(total_trans_norm, 1e-12)
            if (
                float(np.linalg.norm(rot_step)) < np.deg2rad(0.005)
                and float(np.linalg.norm(trans_step)) < 1e-5
            ):
                break
        work = apply_T_points(overlap_local, T).astype(float)
        distances, _ = target_tree.query(work, k=1, workers=-1)
        valid = np.isfinite(distances) & (distances <= threshold)
        final_fit = float(np.mean(valid))
        final_rmse = (
            float(np.sqrt(np.mean(np.square(distances[valid]))))
            if np.any(valid)
            else float("inf")
        )

    T_local = T.copy()
    to_pivot = np.eye(4, dtype=float)
    to_pivot[:3, 3] = pivot
    from_pivot = np.eye(4, dtype=float)
    from_pivot[:3, 3] = -pivot
    best_T = to_pivot @ T_local @ from_pivot
    best_fit = final_fit
    best_rmse = final_rmse

    R_icp = best_T[:3, :3]
    t_icp = T_local[:3, 3]
    tr = np.clip((np.trace(R_icp) - 1.0) * 0.5, -1.0, 1.0)
    rot_deg = float(np.rad2deg(np.arccos(tr)))
    trans = float(np.linalg.norm(t_icp))
    calibrated = apply_T_points(src, best_T)
    after = nn_stats(calibrated, tgt)
    before_med = float(before.get("median", np.inf))
    after_med = float(after.get("median", np.inf))
    before_p95 = float(before.get("p95", np.inf))
    after_p95 = float(after.get("p95", np.inf))
    before_in = float(before.get("inlier@1cm", 0.0))
    after_in = float(after.get("inlier@1cm", 0.0))
    med_improve = (
        (before_med - after_med) / max(before_med, 1e-6)
        if np.isfinite(before_med)
        else 0.0
    )
    p95_improve = (
        (before_p95 - after_p95) / max(before_p95, 1e-6)
        if np.isfinite(before_p95)
        else 0.0
    )
    accept = bool(
        best_fit >= float(args.active_reg_icp_min_fit)
        and trans <= float(args.active_reg_icp_max_trans) + 1e-9
        and rot_deg <= float(args.active_reg_icp_max_rot_deg) + 1e-6
        and after_in + 0.03 >= before_in
        and (
            med_improve >= float(args.active_reg_min_improve)
            or p95_improve >= float(args.active_reg_min_improve)
            or after_in >= before_in + 0.08
        )
    )
    if not accept:
        best_T = T_identity
    if trans > float(args.active_reg_icp_max_trans) + 1e-9:
        reason = "large_translation"
    elif rot_deg > float(args.active_reg_icp_max_rot_deg) + 1e-6:
        reason = "large_rotation"
    elif best_fit < float(args.active_reg_icp_min_fit):
        reason = "low_fit"
    elif not accept:
        reason = "no_improvement"
    else:
        reason = "accepted"
    return best_T, {
        "applied": accept,
        "reason": reason,
        "fit": float(best_fit),
        "rmse": float(best_rmse),
        "trans": trans,
        "rot_deg": rot_deg,
        "overlap_ratio": overlap_ratio,
        "estimation": estimation_name,
        "before": before,
        "after": after,
    }


def _round_fusion_confidence(args):
    profile = getattr(args, "_active_fusion_target_shape_profile", None)
    if not isinstance(profile, dict):
        return 0.0
    return float(np.clip(profile.get("round_confidence", 0.0), 0.0, 1.0))


def _fit_table_plane(points, args):
    """Fit the local tabletop without using target or side-surface geometry."""
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    points = points[np.isfinite(points).all(axis=1)]
    table_z = float(getattr(args, "table_z", 0.08))
    z_window = max(float(getattr(args, "active_table_reg_z_window", 0.040)), 0.005)
    inlier_threshold = max(
        float(getattr(args, "active_table_reg_inlier", 0.003)), 0.0005
    )
    minimum = max(int(getattr(args, "active_table_reg_min_points", 250)), 30)
    local = points[np.abs(points[:, 2] - table_z) <= z_window]
    if local.shape[0] < minimum:
        return None, {
            "reason": "too_few_table_points",
            "candidate_points": int(local.shape[0]),
        }
    if local.shape[0] > 12000:
        indices = np.linspace(0, local.shape[0] - 1, 12000, dtype=int)
        work = local[indices]
    else:
        work = local
    rng = np.random.default_rng(17)
    best_mask = None
    best_count = 0
    for _ in range(180):
        sample = work[rng.choice(work.shape[0], size=3, replace=False)]
        normal = np.cross(sample[1] - sample[0], sample[2] - sample[0])
        norm = float(np.linalg.norm(normal))
        if norm < 1e-8:
            continue
        normal /= norm
        if normal[2] < 0.0:
            normal = -normal
        if normal[2] < float(np.cos(np.deg2rad(20.0))):
            continue
        d = -float(np.dot(normal, sample[0]))
        mask = np.abs(work @ normal + d) <= inlier_threshold
        count = int(np.count_nonzero(mask))
        if count > best_count:
            best_count = count
            best_mask = mask
    if best_mask is None or best_count < minimum:
        return None, {
            "reason": "ransac_no_table_plane",
            "candidate_points": int(local.shape[0]),
            "best_inliers": int(best_count),
        }
    seed_inliers = work[best_mask]
    centroid = np.mean(seed_inliers, axis=0)
    _, _, vh = np.linalg.svd(seed_inliers - centroid, full_matrices=False)
    normal = np.asarray(vh[-1], dtype=float)
    normal /= max(float(np.linalg.norm(normal)), 1e-12)
    if normal[2] < 0.0:
        normal = -normal
    d = -float(np.dot(normal, centroid))
    full_mask = np.abs(local @ normal + d) <= inlier_threshold
    inliers = local[full_mask]
    if inliers.shape[0] < minimum:
        return None, {
            "reason": "refined_table_too_small",
            "candidate_points": int(local.shape[0]),
            "inliers": int(inliers.shape[0]),
        }
    centroid = np.mean(inliers, axis=0)
    _, _, vh = np.linalg.svd(inliers - centroid, full_matrices=False)
    normal = np.asarray(vh[-1], dtype=float)
    normal /= max(float(np.linalg.norm(normal)), 1e-12)
    if normal[2] < 0.0:
        normal = -normal
    d = -float(np.dot(normal, centroid))
    residual = np.abs(inliers @ normal + d)
    return {
        "normal": normal,
        "d": d,
        "centroid": centroid,
        "height": -d / max(float(normal[2]), 1e-9),
        "inliers": int(inliers.shape[0]),
        "rms": float(np.sqrt(np.mean(np.square(residual)))),
    }, {"reason": "ok", "candidate_points": int(local.shape[0])}


def _rotation_between_vectors(source, target):
    source = np.asarray(source, dtype=float).reshape(3)
    target = np.asarray(target, dtype=float).reshape(3)
    source /= max(float(np.linalg.norm(source)), 1e-12)
    target /= max(float(np.linalg.norm(target)), 1e-12)
    cross = np.cross(source, target)
    sine = float(np.linalg.norm(cross))
    cosine = float(np.clip(np.dot(source, target), -1.0, 1.0))
    if sine < 1e-10:
        if cosine > 0.0:
            return np.eye(3, dtype=float)
        axis = np.cross(source, np.array([1.0, 0.0, 0.0]))
        if np.linalg.norm(axis) < 1e-6:
            axis = np.cross(source, np.array([0.0, 1.0, 0.0]))
        axis /= max(float(np.linalg.norm(axis)), 1e-12)
        return Rotation.from_rotvec(np.pi * axis).as_matrix()
    axis = cross / sine
    return Rotation.from_rotvec(np.arctan2(sine, cosine) * axis).as_matrix()


def _round_table_plane_correction(
    reference_scene_points,
    source_scene_points,
    source_target_points,
    args,
):
    """Constrain only table-normal height/tilt; preserve complementary XY/yaw."""
    reference, reference_info = _fit_table_plane(reference_scene_points, args)
    source, source_info = _fit_table_plane(source_scene_points, args)
    identity = np.eye(4, dtype=float)
    if reference is None or source is None:
        return identity, {
            "applied": False,
            "reason": "table_fit_failed",
            "reference": reference_info,
            "source": source_info,
        }
    rotation = _rotation_between_vectors(source["normal"], reference["normal"])
    tilt_deg = float(
        np.rad2deg(
            np.arccos(
                np.clip(
                    np.dot(source["normal"], reference["normal"]), -1.0, 1.0
                )
            )
        )
    )
    max_tilt = float(getattr(args, "active_table_reg_max_tilt_deg", 5.0))
    if tilt_deg > max_tilt:
        return identity, {
            "applied": False,
            "reason": "tilt_outside_envelope",
            "tilt_deg": tilt_deg,
            "reference": reference,
            "source": source,
        }
    target_points = np.asarray(source_target_points, dtype=float).reshape(-1, 3)
    pivot = (
        np.median(target_points[np.isfinite(target_points).all(axis=1)], axis=0)
        if np.any(np.isfinite(target_points).all(axis=1))
        else np.asarray(source["centroid"], dtype=float)
    )
    rotation_translation = pivot - rotation @ pivot
    rotated_d = float(source["d"] - np.dot(reference["normal"], rotation_translation))
    normal_shift = rotated_d - float(reference["d"])
    max_shift = float(
        getattr(args, "active_table_reg_max_height_shift", 0.015)
    )
    if abs(normal_shift) > max_shift:
        return identity, {
            "applied": False,
            "reason": "height_shift_outside_envelope",
            "tilt_deg": tilt_deg,
            "height_shift": normal_shift,
            "reference": reference,
            "source": source,
        }
    transform = identity.copy()
    transform[:3, :3] = rotation
    transform[:3, 3] = (
        rotation_translation + normal_shift * reference["normal"]
    )
    corrected_source = _apply_T_points(source_scene_points, transform)
    after, after_info = _fit_table_plane(corrected_source, args)
    if after is None:
        return identity, {
            "applied": False,
            "reason": "corrected_table_fit_failed",
            "reference": reference,
            "source": source,
            "after": after_info,
        }
    residual_tilt = float(np.rad2deg(np.arccos(np.clip(
        np.dot(reference["normal"], after["normal"]), -1.0, 1.0
    ))))
    probe = pivot.copy()
    probe[2] = -(np.dot(after["normal"][:2], probe[:2]) + after["d"]) / after["normal"][2]
    residual_height = abs(float(np.dot(reference["normal"], probe) + reference["d"]))
    residual_ok = bool(
        residual_tilt <= float(getattr(args, "active_table_reg_max_residual_tilt_deg", 0.5))
        and residual_height <= float(getattr(args, "active_table_reg_inlier", 0.003))
    )
    if not residual_ok:
        transform = identity
    return transform, {
        "applied": residual_ok,
        "reason": "accepted" if residual_ok else "table_residual_outside_envelope",
        "tilt_deg": tilt_deg,
        "height_shift": normal_shift,
        "residual_tilt_deg": residual_tilt,
        "residual_height_m": residual_height,
        "reference": reference,
        "source": source,
        "after": after,
    }


def _static_registration_points(scene_points, target_points, table_plane, args):
    """Keep coarse non-table structure plus the observed platform boundary."""
    scene = np.asarray(scene_points, dtype=float).reshape(-1, 3)
    scene = scene[np.isfinite(scene).all(axis=1)]
    if scene.shape[0] == 0 or table_plane is None:
        return np.zeros((0, 3), dtype=np.float32)
    normal = np.asarray(table_plane["normal"], dtype=float).reshape(3)
    d = float(table_plane["d"])
    table_band = max(float(getattr(args, "active_static_reg_table_band", 0.008)), 0.002)
    signed_height = scene @ normal + d
    non_table = np.abs(signed_height) > table_band
    # Registration needs rims, platform boundaries and other stationary scene
    # structure, not distant walls/floor or sparse overhead returns.
    non_table &= signed_height >= -0.010
    non_table &= signed_height <= 0.300
    target_region = np.zeros(scene.shape[0], dtype=bool)
    target = np.asarray(target_points, dtype=float).reshape(-1, 3)
    target = target[np.isfinite(target).all(axis=1)]
    if target.shape[0] > 0:
        padding = max(
            float(getattr(args, "active_static_reg_target_padding", 0.035)), 0.010
        )
        low = np.quantile(target, 0.01, axis=0) - padding
        high = np.quantile(target, 0.99, axis=0) + padding
        target_region = np.all((scene >= low) & (scene <= high), axis=1)
        non_table &= ~target_region
    voxel = max(float(getattr(args, "active_static_reg_voxel", 0.008)), 0.003)
    # A flat interior cannot constrain planar translation or yaw, but its
    # observed occupancy boundary can.  Retain only boundary cells, not the
    # dense plane interior, so the fit cannot be dominated by arbitrary table
    # correspondences.
    plane_points = scene[(np.abs(signed_height) <= table_band) & (~target_region)]
    boundary_points = np.zeros((0, 3), dtype=float)
    if plane_points.shape[0] > 0:
        keys = np.floor(plane_points[:, :2] / voxel + 1e-6).astype(np.int64)
        unique_keys, inverse = np.unique(keys, axis=0, return_inverse=True)
        counts = np.bincount(inverse, minlength=unique_keys.shape[0]).astype(float)
        representatives = np.column_stack(
            [
                np.bincount(
                    inverse,
                    weights=plane_points[:, axis],
                    minlength=unique_keys.shape[0],
                )
                / np.maximum(counts, 1.0)
                for axis in range(3)
            ]
        )
        occupied = {tuple(key) for key in unique_keys.tolist()}
        boundary = np.array(
            [
                any(
                    (int(key[0] + dx), int(key[1] + dy)) not in occupied
                    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1))
                )
                for key in unique_keys
            ],
            dtype=bool,
        )
        boundary_points = representatives[boundary]
    static = np.vstack([scene[non_table], boundary_points])
    static = _downsample_points(static, voxel).astype(np.float32)
    if static.shape[0] > 18000:
        indices = np.linspace(0, static.shape[0] - 1, 18000, dtype=int)
        static = static[indices]
    return static


def _symmetric_nn_stats(source, target, max_distance=None):
    def _directional(a, b):
        raw = _nn_stats(a, b)
        if max_distance is None or len(a) == 0 or len(b) == 0:
            raw["overlap_ratio"] = 1.0 if len(a) > 0 and len(b) > 0 else 0.0
            return raw
        distances, _ = cKDTree(b).query(a, k=1, workers=-1)
        valid = np.isfinite(distances) & (distances <= float(max_distance))
        overlap = distances[valid]
        if overlap.size == 0:
            return {
                "median": float("inf"),
                "p95": float("inf"),
                "max": float("inf"),
                "inlier@1cm": 0.0,
                "overlap_ratio": 0.0,
            }
        return {
            "median": float(np.median(overlap)),
            "p95": float(np.quantile(overlap, 0.95)),
            "max": float(np.max(overlap)),
            "inlier@1cm": float(np.mean(overlap <= 0.010)),
            "overlap_ratio": float(np.mean(valid)),
        }

    forward = _directional(source, target)
    reverse = _directional(target, source)
    return {
        "median": 0.5 * (float(forward["median"]) + float(reverse["median"])),
        "p95": 0.5 * (float(forward["p95"]) + float(reverse["p95"])),
        "inlier@1cm": 0.5
        * (float(forward["inlier@1cm"]) + float(reverse["inlier@1cm"])),
        "overlap_ratio": 0.5
        * (float(forward["overlap_ratio"]) + float(reverse["overlap_ratio"])),
        "forward": forward,
        "reverse": reverse,
    }


def _round_static_xyyaw_correction(
    reference_scene_points,
    source_scene_points,
    reference_target_points,
    source_target_points,
    args,
):
    """Estimate bounded XY/yaw from static structure, never from the target."""
    identity = np.eye(4, dtype=float)
    reference_plane, _ = _fit_table_plane(reference_scene_points, args)
    source_plane, _ = _fit_table_plane(source_scene_points, args)
    reference = _static_registration_points(
        reference_scene_points, reference_target_points, reference_plane, args
    )
    source = _static_registration_points(
        source_scene_points, source_target_points, source_plane, args
    )
    minimum = max(int(getattr(args, "active_static_reg_min_pairs", 120)), 30)
    if reference.shape[0] < minimum or source.shape[0] < minimum:
        return identity, {
            "applied": False,
            "reason": "too_few_static_points",
            "reference_points": int(reference.shape[0]),
            "source_points": int(source.shape[0]),
        }
    max_xy = max(float(getattr(args, "active_static_reg_max_xy", 0.005)), 0.0)
    max_yaw = max(
        float(np.deg2rad(getattr(args, "active_static_reg_max_yaw_deg", 1.0))),
        0.0,
    )
    coarse = max(
        float(getattr(args, "active_static_reg_correspondence", 0.025)), 0.008
    )
    before = _symmetric_nn_stats(source, reference, coarse)
    thresholds = (coarse, max(0.60 * coarse, 0.010))
    transform = identity.copy()
    final_pairs = 0
    for threshold in thresholds:
        for _ in range(8):
            moved = _apply_T_points(source, transform).astype(float)
            ref_tree = cKDTree(reference)
            distances, indices = ref_tree.query(moved, k=1, workers=-1)
            valid = np.isfinite(distances) & (distances <= threshold)
            if int(np.count_nonzero(valid)) < minimum:
                break
            valid_ids = np.flatnonzero(valid)
            # Mutual nearest-neighbour pairs prevent a large one-sided surface
            # from pulling a small rim/edge onto the wrong repeated structure.
            moved_tree = cKDTree(moved)
            _, reverse_ids = moved_tree.query(reference, k=1, workers=-1)
            mutual = reverse_ids[np.asarray(indices[valid], dtype=int)] == valid_ids
            valid_ids = valid_ids[mutual]
            if valid_ids.size < minimum:
                break
            pair_distances = distances[valid_ids]
            trim_limit = float(np.quantile(pair_distances, 0.75))
            valid_ids = valid_ids[pair_distances <= trim_limit]
            if valid_ids.size < minimum:
                break
            p = moved[valid_ids, :2]
            q = reference[np.asarray(indices[valid_ids], dtype=int), :2]
            residual = np.linalg.norm(p - q, axis=1)
            huber = 0.006
            weights = np.where(
                residual <= huber, 1.0, huber / np.maximum(residual, 1e-12)
            )
            weight_sum = max(float(np.sum(weights)), 1e-12)
            p_center = np.sum(p * weights[:, None], axis=0) / weight_sum
            q_center = np.sum(q * weights[:, None], axis=0) / weight_sum
            covariance = (p - p_center).T @ ((q - q_center) * weights[:, None])
            u, _, vt = np.linalg.svd(covariance)
            rotation_2d = vt.T @ u.T
            if np.linalg.det(rotation_2d) < 0.0:
                vt[-1, :] *= -1.0
                rotation_2d = vt.T @ u.T
            step_yaw = float(np.arctan2(rotation_2d[1, 0], rotation_2d[0, 0]))
            step_yaw = float(np.clip(step_yaw, -np.deg2rad(0.25), np.deg2rad(0.25)))
            c, s = np.cos(step_yaw), np.sin(step_yaw)
            step_rotation = np.array([[c, -s], [s, c]], dtype=float)
            step_translation = q_center - step_rotation @ p_center
            step_norm = float(np.linalg.norm(step_translation))
            if step_norm > 0.0015:
                step_translation *= 0.0015 / step_norm
            step = identity.copy()
            step[:2, :2] = step_rotation
            step[:2, 3] = step_translation
            transform = step @ transform
            total_yaw = float(np.arctan2(transform[1, 0], transform[0, 0]))
            total_yaw = float(np.clip(total_yaw, -max_yaw, max_yaw))
            c, s = np.cos(total_yaw), np.sin(total_yaw)
            transform[:3, :3] = np.array(
                [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=float
            )
            xy_norm = float(np.linalg.norm(transform[:2, 3]))
            if xy_norm > max_xy and xy_norm > 0.0:
                transform[:2, 3] *= max_xy / xy_norm
            transform[2, 3] = 0.0
            final_pairs = int(valid_ids.size)
            if abs(step_yaw) < np.deg2rad(0.005) and step_norm < 2e-5:
                break
    corrected = _apply_T_points(source, transform)
    after = _symmetric_nn_stats(corrected, reference, coarse)
    yaw_deg = float(np.rad2deg(np.arctan2(transform[1, 0], transform[0, 0])))
    xy = np.asarray(transform[:2, 3], dtype=float)
    median_improve = (before["median"] - after["median"]) / max(
        before["median"], 1e-6
    )
    p95_improve = (before["p95"] - after["p95"]) / max(before["p95"], 1e-6)
    inlier_improve = after["inlier@1cm"] - before["inlier@1cm"]
    accepted = bool(
        final_pairs >= minimum
        and after["p95"] <= before["p95"] + 0.0005
        and after["inlier@1cm"] + 0.01 >= before["inlier@1cm"]
        and (
            median_improve >= 0.02
            or p95_improve >= 0.02
            or inlier_improve >= 0.02
        )
    )
    if not accepted:
        transform = identity
    return transform, {
        "applied": accepted,
        "reason": "accepted" if accepted else "no_robust_improvement",
        "xy_shift": xy,
        "xy_norm": float(np.linalg.norm(xy)),
        "yaw_deg": yaw_deg,
        "pairs": final_pairs,
        "reference_points": int(reference.shape[0]),
        "source_points": int(source.shape[0]),
        "before": before,
        "after": after,
    }


def _scene_with_registered_target(scene_cloud, raw_target, registered_target,
                                  T_scene, args, *, points_from_cloud_hook=None,
                                  transform_cloud_hook=None):
    """Replace only this view's old object samples, leaving background on T_scene."""
    points_fn = _points_from_cloud if points_from_cloud_hook is None else points_from_cloud_hook
    transform_fn = transform_cloud if transform_cloud_hook is None else transform_cloud_hook
    if not callable(getattr(scene_cloud, "select_by_index", None)):
        return transform_fn(scene_cloud, T_scene), 0
    scene_points = points_fn(scene_cloud)
    target_points = points_fn(raw_target)
    if scene_points.shape[0] == 0 or target_points.shape[0] == 0:
        return transform_fn(scene_cloud, T_scene), 0
    radius = max(1.5 * float(getattr(args, "scene_voxel_size", 0.004)),
                 2.0 * float(getattr(args, "voxel_size", 0.0015)), 0.003)
    distances, _ = cKDTree(target_points).query(scene_points, k=1, workers=-1)
    background_ids = np.flatnonzero(distances > radius)
    background = scene_cloud.select_by_index(background_ids.tolist())
    corrected_scene = transform_fn(background, T_scene) + registered_target
    return corrected_scene, int(scene_points.shape[0] - background_ids.size)


def _registration_report_value(value):
    if isinstance(value, dict):
        return {str(key): _registration_report_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_registration_report_value(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def register_active_fusion_views(
    view_names,
    target_base_clouds,
    scene_base_clouds,
    T_base_camera0,
    out_dir,
    args,
    *,
    points_from_cloud_hook=None,
    downsample_points_hook=None,
    nn_stats_hook=None,
    small_translation_registration_hook=None,
    apply_T_points_hook=None,
    rigid_icp_registration_hook=None,
    transform_cloud_hook=None,
    save_cloud_hook=None,
):
    points_from_cloud = (
        _points_from_cloud
        if points_from_cloud_hook is None
        else points_from_cloud_hook
    )
    downsample_points = (
        _downsample_points
        if downsample_points_hook is None
        else downsample_points_hook
    )
    nn_stats = _nn_stats if nn_stats_hook is None else nn_stats_hook
    small_translation_registration = (
        _small_translation_registration
        if small_translation_registration_hook is None
        else small_translation_registration_hook
    )
    apply_T_points = (
        _apply_T_points if apply_T_points_hook is None else apply_T_points_hook
    )
    rigid_icp_registration = (
        _rigid_icp_registration
        if rigid_icp_registration_hook is None
        else rigid_icp_registration_hook
    )
    transform_cloud_impl = (
        transform_cloud if transform_cloud_hook is None else transform_cloud_hook
    )
    save_cloud_impl = save_cloud if save_cloud_hook is None else save_cloud_hook

    if not bool(args.active_fusion_register) or len(target_base_clouds) <= 1:
        return (
            target_base_clouds,
            scene_base_clouds,
            [True] * len(target_base_clouds),
        )
    ref_pts = downsample_points(
        points_from_cloud(target_base_clouds[0]),
        float(args.active_reg_voxel),
    )
    ref_scene_pts = points_from_cloud(scene_base_clouds[0])
    round_confidence = _round_fusion_confidence(args)
    round_target_mode = bool(round_confidence >= 0.50)
    round_table_mode = bool(
        getattr(args, "active_round_table_registration", True)
        and round_target_mode
    )
    if ref_pts.shape[0] < int(args.active_reg_min_pairs):
        print(
            "[active][reg][warn] reference target cloud too small; "
            "skip registration"
        )
        return (
            target_base_clouds,
            scene_base_clouds,
            [True] * len(target_base_clouds),
        )

    corrected_targets = [target_base_clouds[0]]
    corrected_scenes = [scene_base_clouds[0]]
    keep = [True]
    registration_report = {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(timespec="milliseconds"),
        "frame": "base_link",
        "reference_view": view_names[0],
        "views": [],
    }
    save_cloud_impl(
        os.path.join(
            out_dir,
            f"active_{view_names[0]}_target_base_registered.ply",
        ),
        target_base_clouds[0],
    )
    save_cloud_impl(os.path.join(out_dir, f"active_{view_names[0]}_scene_base_raw.ply"), scene_base_clouds[0])
    save_cloud_impl(os.path.join(out_dir, f"active_{view_names[0]}_scene_base_registered.ply"), scene_base_clouds[0])

    if round_table_mode:
        print(
            "[active][reg][round] use tabletop-only Z/roll/pitch correction; "
            "target ICP/translation are disabled so complementary surfaces "
            "cannot be pulled together"
        )
    elif round_target_mode:
        print(
            "[active][reg][round] rotation registration disabled; keep "
            "complementary round-target views in their measured base frame"
        )
    else:
        print("[active][reg] target-cloud registration against first view:")
    for name, target_cloud, scene_cloud in zip(
        view_names[1:],
        target_base_clouds[1:],
        scene_base_clouds[1:],
    ):
        src_pts_full = points_from_cloud(target_cloud)
        src_pts = downsample_points(
            src_pts_full,
            float(args.active_reg_voxel),
        )
        T_total = np.eye(4, dtype=float)
        T_scene = np.eye(4, dtype=float)
        before = nn_stats(src_pts, ref_pts)

        table_info = {"applied": False, "reason": "not_round_target"}
        table_requested = bool(
            round_table_mode
            or (not round_target_mode and getattr(args, "active_scene_table_registration", True))
        )
        if round_table_mode:
            T_table, table_info = _round_table_plane_correction(
                ref_scene_pts,
                points_from_cloud(scene_cloud),
                src_pts_full,
                args,
            )
            T_total = T_table @ T_total
            T_scene = T_table @ T_scene
            src_pts = apply_T_points(src_pts, T_table)
        elif (
            not round_target_mode
            and bool(getattr(args, "active_scene_table_registration", True))
        ):
            # Target-derived ICP compensates partial visibility and segmentation
            # residuals.  Applying it to the whole scene can bend a correctly
            # observed table into a second plane.  Establish the shared static
            # frame first from the table, then let only the target continue
            # through local translation/ICP below.
            T_table, table_info = _round_table_plane_correction(
                ref_scene_pts,
                points_from_cloud(scene_cloud),
                src_pts_full,
                args,
            )
            T_total = T_table @ T_total
            T_scene = T_table @ T_scene
            src_pts = apply_T_points(src_pts, T_table)

        static_info = {"applied": False, "reason": "disabled"}
        if round_table_mode and bool(
            getattr(args, "active_round_static_xyyaw_registration", True)
        ):
            table_scene = apply_T_points(
                points_from_cloud(scene_cloud), T_total
            )
            table_target = apply_T_points(src_pts_full, T_total)
            T_static, static_info = _round_static_xyyaw_correction(
                ref_scene_pts,
                table_scene,
                points_from_cloud(target_base_clouds[0]),
                table_target,
                args,
            )
            T_total = T_static @ T_total
            T_scene = T_static @ T_scene
            src_pts = apply_T_points(src_pts, T_static)

        trans_info = {
            "applied": False,
            "reason": "round_target_no_local_registration"
            if round_target_mode
            else "disabled",
            "before": before,
            "after": before,
        }
        if bool(args.active_fusion_translation_calib) and not round_target_mode:
            T_trans, trans_info = small_translation_registration(
                src_pts,
                ref_pts,
                args,
            )
            T_total = T_trans @ T_total
            src_pts = apply_T_points(src_pts, T_trans)

        icp_info = {
            "applied": False,
            "reason": "round_target_no_local_registration"
            if round_target_mode
            else "disabled",
            "before": nn_stats(src_pts, ref_pts),
            "after": nn_stats(src_pts, ref_pts),
        }
        if bool(args.active_fusion_rigid_icp) and not round_target_mode:
            T_icp, icp_info = rigid_icp_registration(
                src_pts,
                ref_pts,
                args,
            )
            T_total = T_icp @ T_total
            src_pts = apply_T_points(src_pts, T_icp)

        corrected_target = transform_cloud_impl(target_cloud, T_total)
        replaced_target_points = 0
        if bool(args.active_reg_apply_to_scene):
            corrected_scene, replaced_target_points = _scene_with_registered_target(
                scene_cloud, target_cloud, corrected_target, T_scene, args,
                points_from_cloud_hook=points_from_cloud,
                transform_cloud_hook=transform_cloud_impl,
            )
        else:
            corrected_scene = scene_cloud
        # after = nn_stats(
        #     downsample_points(
        #         points_from_cloud(corrected_target),
        #         float(args.active_reg_voxel),
        #     ),
        #     ref_pts,
        corrected_points = downsample_points(
            points_from_cloud(corrected_target),
            float(args.active_reg_voxel),
        )
        after = nn_stats(corrected_points, ref_pts)
        overlap_after = _mutual_overlap_stats(
            corrected_points,
            ref_pts,
            float(getattr(args, "active_reg_icp_overlap_thr", 0.018)),
        )
        keep_view = True
        reasons = []
        if round_target_mode:
            corrected_points = points_from_cloud(corrected_target)
            center_delta = float(
                np.linalg.norm(
                    np.median(corrected_points, axis=0) - np.median(ref_pts, axis=0)
                )
            )
            median_ok = bool(
                np.isfinite(after["median"])
                and after["median"]
                <= float(getattr(args, "active_round_fusion_max_median", 0.015))
            )
            inlier_ok = bool(
                after["inlier@1cm"]
                >= float(getattr(args, "active_round_fusion_min_inlier_1cm", 0.20))
            )
            keep_view = bool(
                np.isfinite(center_delta)
                and center_delta
                <= float(getattr(args, "active_round_fusion_max_center_delta", 0.040))
                and np.isfinite(after["p95"])
                and after["p95"]
                <= float(getattr(args, "active_round_fusion_max_p95", 0.040))
                and (median_ok or inlier_ok)
            )
            if not keep_view:
                reasons.append(
                    f"round_sanity(center={center_delta:.4f},median={after['median']:.4f},"
                    f"p95={after['p95']:.4f},inlier={after['inlier@1cm']:.3f})"
                )
        else:
            if (
                int(overlap_after["pairs"])
                < int(args.active_reg_min_pairs)
            ):
                keep_view = False
                reasons.append(
                    f"overlap_pairs={int(overlap_after['pairs'])}"
                )
            if (
               float(overlap_after["overlap_ratio"])
                < float(
                    getattr(
                        args,
                        "active_reg_icp_min_overlap_ratio",
                        0.08,
                    )
                )
            ):
                keep_view = False
                reasons.append(
                    f"overlap_ratio={overlap_after['overlap_ratio']:.3f}"
                )
            if (
                not np.isfinite(overlap_after["median"])
                or overlap_after["median"]
                > float(args.active_reg_max_median)
            ):
                keep_view = False
                reasons.append(
                    f"overlap_median={overlap_after['median']:.4f}"
                )
            if (
                not np.isfinite(overlap_after["p95"])
                or overlap_after["p95"]
                > float(args.active_reg_max_p95)
            ):
                keep_view = False
                reasons.append(
                    f"overlap_p95={overlap_after['p95']:.4f}"
                )

        if table_requested and not bool(table_info.get("applied", False)):
            keep_view = False
            reasons.append(f"table_alignment={table_info.get('reason')}")

        save_cloud_impl(
            os.path.join(out_dir, f"active_{name}_target_base_raw.ply"),
            target_cloud,
        )
        save_cloud_impl(
            os.path.join(
                out_dir,
                f"active_{name}_target_base_registered.ply",
            ),
            corrected_target,
        )
        save_cloud_impl(os.path.join(out_dir, f"active_{name}_scene_base_raw.ply"), scene_cloud)
        save_cloud_impl(os.path.join(out_dir, f"active_{name}_scene_base_registered.ply"), corrected_scene)
        registration_report["views"].append({
            "name": name, "registration_keep": keep_view, "reject_reasons": reasons,
            "target_transform": T_total, "background_transform": T_scene,
            "table": table_info, "translation": trans_info, "icp": icp_info,
            "overlap": overlap_after, "replaced_scene_target_points": replaced_target_points,
            "background_correction_applied": bool(args.active_reg_apply_to_scene),
        })
        corrected_targets.append(corrected_target)
        corrected_scenes.append(corrected_scene)
        keep.append(keep_view)
        shift = np.asarray(
            trans_info.get("shift", np.zeros(3)),
            dtype=float,
        ).reshape(3)
        print(
            f"[active][reg] {name}: before med={before['median']:.4f} "
            f"p95={before['p95']:.4f} in1cm={before['inlier@1cm']:.3f}; "
            f"table={table_info.get('reason')} "
            f"applied={bool(table_info.get('applied', False))} "
            f"dZ={1000.0 * float(table_info.get('height_shift', 0.0)):.2f}mm "
            f"dTilt={float(table_info.get('tilt_deg', 0.0)):.2f}deg; "
            f"table_residual_tilt={float(table_info.get('residual_tilt_deg', float('nan'))):.2f}deg "
            f"table_residual_height={1000.0 * float(table_info.get('residual_height_m', float('nan'))):.2f}mm; "
            f"static_xyyaw={static_info.get('reason')} "
            f"applied={bool(static_info.get('applied', False))} "
            f"dXY={1000.0 * float(static_info.get('xy_norm', 0.0)):.2f}mm "
            f"dYaw={float(static_info.get('yaw_deg', 0.0)):.2f}deg "
            f"pairs={int(static_info.get('pairs', 0))}; "
            f"trans={trans_info.get('reason')} "
            f"applied={bool(trans_info.get('applied', False))} "
            f"pairs={int(trans_info.get('trimmed_pairs', trans_info.get('pairs', 0)))}/"
            f"{int(trans_info.get('pairs', 0))} "
            f"trim_q={float(trans_info.get('trim_quantile', 1.0)):.2f} "
            f"iters={int(trans_info.get('iters', 0))} "
            f"shift={np.round(shift, 5).tolist()}m; "
            f"icp={icp_info.get('reason')} "
            f"applied={bool(icp_info.get('applied', False))} "
            f"dT={float(icp_info.get('trans', 0.0)):.4f}m "
            f"dR={float(icp_info.get('rot_deg', 0.0)):.2f}deg; "
            f"scene_transform={'table/static-only' if bool(args.active_reg_apply_to_scene) else 'raw'}; "
            f"scene_target_replaced={replaced_target_points}; "
            f"after med={after['median']:.4f} p95={after['p95']:.4f} "
            f"in1cm={after['inlier@1cm']:.3f}; "
            f"overlap pairs={int(overlap_after['pairs'])} "
            f"ratio={float(overlap_after['overlap_ratio']):.3f} "
            f"med={float(overlap_after['median']):.4f} "
            f"p95={float(overlap_after['p95']):.4f} "
            f"novel={float(overlap_after['novel_source_ratio']):.3f} "
            f"keep={keep_view}"
            f"{'' if keep_view else ' reject=' + ','.join(reasons)}"
        )
    if save_cloud_hook is None or os.path.isdir(out_dir):
        report_path = os.path.join(out_dir, "active_fusion_registration.json")
        try:
            os.makedirs(out_dir, exist_ok=True)
            with open(report_path, "w", encoding="utf-8") as stream:
                json.dump(_registration_report_value(registration_report), stream,
                          ensure_ascii=False, indent=2, allow_nan=False)
            print(f"[active][reg] report={report_path}")
        except OSError as exc:
            print(f"[active][reg][warn] registration report could not be saved: {exc}")
    return corrected_targets, corrected_scenes, keep


def _active_side_capture_overlap_ok(
    reference_base_cloud,
    side_base_cloud,
    args,
    label,
    *,
    reference_scene_base_cloud=None,
    side_scene_base_cloud=None,
    points_from_cloud_hook=None,
    downsample_points_hook=None,
    small_translation_registration_hook=None,
    apply_T_points_hook=None,
    nn_stats_hook=None,
    mutual_overlap_stats_hook=None,
):
    """Screen a side capture immediately so rejected views do not consume the quota."""
    """Require registration support while preserving genuinely novel surfaces."""
    points_from_cloud = (
        _points_from_cloud
        if points_from_cloud_hook is None
        else points_from_cloud_hook
    )
    downsample_points = (
        _downsample_points
        if downsample_points_hook is None
        else downsample_points_hook
    )
    small_translation_registration = (
        _small_translation_registration
        if small_translation_registration_hook is None
        else small_translation_registration_hook
    )
    apply_T_points = (
        _apply_T_points if apply_T_points_hook is None else apply_T_points_hook
    )
    nn_stats = _nn_stats if nn_stats_hook is None else nn_stats_hook
    mutual_overlap_stats = (
        _mutual_overlap_stats
        if mutual_overlap_stats_hook is None
        else mutual_overlap_stats_hook
    )
    ref_pts = downsample_points(
        points_from_cloud(reference_base_cloud),
        float(args.active_reg_voxel),
    )
    side_pts = downsample_points(
        points_from_cloud(side_base_cloud),
        float(args.active_reg_voxel),
    )
    if (
        ref_pts.shape[0] < int(args.active_reg_min_pairs)
        or side_pts.shape[0] < int(args.active_reg_min_pairs)
    ):
        print(
            f"[active][capture-quality] {label} rejected: too few target points "
            f"ref={ref_pts.shape[0]} side={side_pts.shape[0]}"
        )
        return False
    T_preview = np.eye(4, dtype=float)
    round_mode = bool(_round_fusion_confidence(args) >= 0.50)
    table_info = {"reason": "scene_unavailable", "applied": False}
    table_requested = bool(
        getattr(args, "active_round_table_registration", True) if round_mode
        else getattr(args, "active_scene_table_registration", True)
    )
    if table_requested and reference_scene_base_cloud is not None and side_scene_base_cloud is not None:
        T_preview, table_info = _round_table_plane_correction(
            points_from_cloud(reference_scene_base_cloud),
            points_from_cloud(side_scene_base_cloud), side_pts, args,
        )
        if not bool(table_info.get("applied", False)):
            print(f"[active][capture-quality] {label} rejected: tabletop alignment {table_info.get('reason')}")
            return False
    if bool(args.active_fusion_translation_calib) and not round_mode:
        T_translation, _ = small_translation_registration(
            apply_T_points(side_pts, T_preview),
            ref_pts,
            args,
        )
        T_preview = T_translation @ T_preview
    registered = apply_T_points(side_pts, T_preview)
    after = nn_stats(registered, ref_pts)
    overlap = mutual_overlap_stats(
        registered,
        ref_pts,
        float(getattr(args, "active_reg_icp_overlap_thr", 0.018)),
    )
    if round_mode:
        center_delta = float(
            np.linalg.norm(np.median(registered, axis=0) - np.median(ref_pts, axis=0))
        )
        keep = bool(
            center_delta
            <= float(getattr(args, "active_round_fusion_max_center_delta", 0.040))
            and np.isfinite(float(after["p95"]))
            and float(after["p95"])
            <= float(getattr(args, "active_round_fusion_max_p95", 0.040))
            and (
                (
                    np.isfinite(float(after["median"]))
                    and float(after["median"])
                    <= float(getattr(args, "active_round_fusion_max_median", 0.015))
                )
                or float(after["inlier@1cm"])
                >= float(
                    getattr(args, "active_round_fusion_min_inlier_1cm", 0.20)
                )
            )
        )
    else:
        center_delta = float("nan")
        keep = bool(
            # np.isfinite(float(after["median"]))
            # and float(after["median"]) <= float(args.active_reg_max_median)
            # and np.isfinite(float(after["p95"]))
            # and float(after["p95"]) <= float(args.active_reg_max_p95)
            # and float(after["inlier@1cm"])
            # >= float(args.active_reg_min_inlier_1cm)
            int(overlap["pairs"]) >= int(args.active_reg_min_pairs)
            and float(overlap["overlap_ratio"])
            >= float(
                getattr(args, "active_reg_icp_min_overlap_ratio", 0.08)
            )
            and np.isfinite(float(overlap["median"]))
            and float(overlap["median"])
            <= float(args.active_reg_max_median)
            and np.isfinite(float(overlap["p95"]))
            and float(overlap["p95"])
            <= float(args.active_reg_max_p95)
        )
    print(
        f"[active][capture-quality] {label}: "
        f"all_med={float(after['median']):.4f}m "
        f"all_p95={float(after['p95']):.4f}m "
        f"overlap_pairs={int(overlap['pairs'])} "
        f"overlap_ratio={float(overlap['overlap_ratio']):.3f} "
        f"overlap_med={float(overlap['median']):.4f}m "
        f"overlap_p95={float(overlap['p95']):.4f}m "
        f"novel={float(overlap['novel_source_ratio']):.3f} "
        f"table={table_info.get('reason')} "
        f"round_mode={round_mode} center_delta={center_delta:.4f}m keep={keep}"
    )
    if not keep:
        print(
            f"[active][capture-quality] {label} does not count toward the "
            "side-view quota; return to center before trying another "
            "verified path"
        )
    return keep


def diagnose_active_fusion(
    view_names,
    target_base_clouds,
    T_base_camera0,
    out_dir,
    args,
    *,
    cloud_stats_hook=None,
    points_from_cloud_hook=None,
    downsample_points_hook=None,
    nn_stats_hook=None,
    mutual_overlap_stats_hook=None,
    save_cloud_hook=None,
    transform_cloud_hook=None,
):
    cloud_stats = _cloud_stats if cloud_stats_hook is None else cloud_stats_hook
    points_from_cloud = (
        _points_from_cloud
        if points_from_cloud_hook is None
        else points_from_cloud_hook
    )
    downsample_points = (
        _downsample_points
        if downsample_points_hook is None
        else downsample_points_hook
    )
    nn_stats = _nn_stats if nn_stats_hook is None else nn_stats_hook
    mutual_overlap_stats = (
        _mutual_overlap_stats
        if mutual_overlap_stats_hook is None
        else mutual_overlap_stats_hook
    )
    save_cloud_impl = save_cloud if save_cloud_hook is None else save_cloud_hook
    transform_cloud_impl = (
        transform_cloud if transform_cloud_hook is None else transform_cloud_hook
    )

    if not bool(args.active_fusion_diagnostics):
        return [True] * len(target_base_clouds)
    if len(target_base_clouds) == 0:
        return []
    T_camera0_base = np.linalg.inv(
        np.asarray(T_base_camera0, dtype=float).reshape(4, 4)
    )
    ref_stats = cloud_stats(target_base_clouds[0])
    ref_center = np.asarray(ref_stats["median"], dtype=float)
    ref_axis = np.asarray(ref_stats["pca_axis"], dtype=float)
    ref_points = downsample_points(
        points_from_cloud(target_base_clouds[0]),
        float(args.active_reg_voxel),
    )
    print("[active][diag] target cloud alignment in base frame:")
    round_mode = bool(_round_fusion_confidence(args) >= 0.50)
    keep = []
    for name, cloud in zip(view_names, target_base_clouds):
        stats = cloud_stats(cloud)
        center = np.asarray(stats["median"], dtype=float)
        delta = center - ref_center
        delta_norm = (
            float(np.linalg.norm(delta))
            if np.isfinite(delta).all()
            else float("nan")
        )
        axis = np.asarray(stats["pca_axis"], dtype=float)
        axis_angle = float("nan")
        if np.isfinite(ref_axis).all() and np.isfinite(axis).all():
            dot = abs(float(np.dot(ref_axis, axis)))
            axis_angle = float(
                np.rad2deg(np.arccos(np.clip(dot, -1.0, 1.0)))
            )
        is_first = len(keep) == 0
        if is_first:
            overlap = {
                "pairs": int(ref_points.shape[0]),
                "median": 0.0,
                "p95": 0.0,
                "inlier@1cm": 1.0,
                "overlap_ratio": 1.0,
                "novel_source_ratio": 0.0,
            }
        else:
            current_points = downsample_points(
                points_from_cloud(cloud),
                float(args.active_reg_voxel),
            )
            if mutual_overlap_stats_hook is None and nn_stats_hook is not None:
                # Preserve lightweight unit-test hooks.
                overlap = dict(nn_stats(current_points, ref_points))
                overlap.setdefault(
                    "pairs", int(getattr(args, "active_reg_min_pairs", 60))
                )
                overlap.setdefault("overlap_ratio", 1.0)
                overlap.setdefault("novel_source_ratio", 0.0)
            else:
                overlap = mutual_overlap_stats(
                    current_points,
                    ref_points,
                    float(
                        getattr(
                            args,
                            "active_reg_icp_overlap_thr",
                            0.018,
                        )
                    ),
                )
        center_limit = (
            float(getattr(args, "active_round_fusion_max_center_delta", 0.040))
            if round_mode
            else float(args.active_fusion_reject_delta)
        )
        center_close = bool(np.isfinite(delta_norm) and delta_norm <= center_limit)
        overlap_override = bool(
            (not is_first)
            and int(overlap.get("pairs", 0))
            >= int(getattr(args, "active_reg_min_pairs", 60))
            and float(overlap.get("overlap_ratio", 0.0))
            >= float(
                getattr(
                    args,
                    "active_reg_icp_min_overlap_ratio",
                    0.08,
                )
            )
            and np.isfinite(float(overlap["median"]))
            and float(overlap["median"])
            <= float(
                getattr(
                    args,
                    "active_round_fusion_max_median"
                    if round_mode
                    else "active_fusion_overlap_max_median",
                    0.015 if round_mode else 0.003,
                )
            )
            and np.isfinite(float(overlap["p95"]))
            and float(overlap["p95"])
            <= float(
                getattr(
                    args,
                    "active_round_fusion_max_p95"
                    if round_mode
                    else "active_fusion_overlap_max_p95",
                    0.040 if round_mode else 0.018,
                )
            )
        )
        keep_view = bool(
            is_first
            or overlap_override
            or (round_mode and center_close)
        )
        axis_reject = (
            (not is_first)
            and (not round_mode)
            and bool(args.active_fusion_reject_axis_mismatch)
            and np.isfinite(axis_angle)
            and axis_angle > float(args.active_fusion_reject_axis_deg)
        )
        if axis_reject:
            keep_view = False
        keep.append(keep_view)
        print(
            f"[active][diag] {name}: n={stats['n']} "
            f"median_base={np.round(center, 6).tolist()} "
            f"delta_from_first={np.round(delta, 6).tolist()}m "
            f"delta_norm={delta_norm:.4f}m "
            f"overlap_med={float(overlap['median']):.4f}m "
            f"overlap_p95={float(overlap['p95']):.4f}m "
            f"overlap_in1cm={float(overlap['inlier@1cm']):.3f} "
            f"overlap_pairs={int(overlap.get('pairs', 0))} "
            f"overlap_ratio={float(overlap.get('overlap_ratio', 0.0)):.3f} "
            f"novel_ratio={float(overlap.get('novel_source_ratio', 0.0)):.3f} "
            f"overlap_override={overlap_override} "
            f"center_shift_gate=diagnostic_only "
            f"pca_axis_angle={axis_angle:.1f}deg "
            f"extent={np.round(np.asarray(stats['extent'], dtype=float), 4).tolist()}m "
            f"fusion_keep={keep_view}"
        )
        save_cloud_impl(
            os.path.join(out_dir, f"active_{name}_target_base.ply"),
            cloud,
        )
        save_cloud_impl(
            os.path.join(out_dir, f"active_{name}_target_camera0.ply"),
            transform_cloud_impl(cloud, T_camera0_base),
        )
        if (
            np.isfinite(delta_norm)
            and delta_norm > float(args.active_fusion_warn_delta)
        ):
            print(
                f"[active][diag][warn] {name} target center differs from "
                f"first view by {delta_norm * 1000.0:.1f}mm. This can come "
                "from partial visibility; registered overlap quality is used "
                "before rejecting the view."
            )
        if (
            (not is_first)
            and bool(args.active_fusion_reject_bad_views)
            and not keep_view
        ):
            reasons = []
            if (
                np.isfinite(delta_norm)
                and delta_norm > center_limit
                and not overlap_override
            ):
                reasons.append(
                    f"target-center delta {delta_norm * 1000.0:.1f}mm > "
                    f"{center_limit * 1000.0:.1f}mm "
                    f"and registered overlap quality was not strong enough "
                    f"(med={float(overlap['median']) * 1000.0:.1f}mm, "
                    f"p95={float(overlap['p95']) * 1000.0:.1f}mm, "
                    f"in1cm={float(overlap['inlier@1cm']):.3f})"
                )
            if axis_reject:
                reasons.append(
                    f"PCA axis angle {axis_angle:.1f}deg > "
                    f"{float(args.active_fusion_reject_axis_deg):.1f}deg"
                )
            print(
                f"[active][diag][reject] {name} excluded from fusion: "
                f"{'; '.join(reasons)}"
            )
        if (
            (not is_first)
            and (not round_mode)
            and np.isfinite(axis_angle)
            and axis_angle > 35.0
            and np.isfinite(delta_norm)
            and delta_norm <= float(args.active_fusion_reject_delta)
        ):
            print(
                f"[active][diag][warn] {name} target center is close but PCA "
                f"axis differs by {axis_angle:.1f}deg. This points to "
                "rotation/frame-convention error rather than sphere-center "
                "offset."
            )
    return keep


def preview_fused_cloud(
    target_cloud,
    scene_cloud,
    args,
    *,
    get_o3d_hook=None,
    draw_open3d_preview_hook=None,
):
    if not args.preview_active_clouds:
        return
    o3d = (get_o3d if get_o3d_hook is None else get_o3d_hook)()
    draw_preview = (
        draw_open3d_preview
        if draw_open3d_preview_hook is None
        else draw_open3d_preview_hook
    )
    scene_vis = copy.deepcopy(scene_cloud)
    target_vis = copy.deepcopy(target_cloud)
    target_vis.paint_uniform_color([1.0, 0.05, 0.05])
    geoms = [scene_vis, target_vis]
    if args.preview_cloud_frame:
        geoms.append(
            o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.08)
        )
    print(
        "[active] showing fused point cloud. Close the Open3D window to continue."
    )
    print(
        "[active] fused target cloud is red; fused scene cloud keeps RGB colors."
    )
    try:
        draw_preview(
            geoms,
            "Fused active-view point cloud: red=target",
            args,
            width=1280,
            height=720,
        )
    except (SafetyStopTriggered, ControlledMotionStopTriggered):
        raise
    except Exception as exc:
        print(
            f"[active][warn] fused cloud preview unavailable: "
            f"{type(exc).__name__}: {exc}"
        )
