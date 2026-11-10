"""Measured geometry and explicitly unverified completion, without hardware imports.

All distances are metres. Support/visibility are geometric proxies, not calibrated
uncertainty. Missing depth and occluded predictions remain unknown.
"""

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import binary_erosion, maximum_filter, minimum_filter
from scipy.spatial import cKDTree


UNKNOWN, SUPPORTED, CONFLICT, INCONSISTENT = range(4)


def xyz(value):
    points = np.asarray(value, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("Expected finite N x 3 XYZ")
    return points


def transform(points, matrix):
    matrix = np.asarray(matrix, dtype=float).reshape(4, 4)
    return xyz(points) @ matrix[:3, :3].T + matrix[:3, 3]


def representatives(points, voxel=.0025):
    points = xyz(points)
    if not len(points):
        return np.empty(0, dtype=int)
    _, indices = np.unique(np.floor(points / voxel).astype(np.int64), axis=0, return_index=True)
    return np.sort(indices)


@dataclass
class DepthEvidence:
    name: str
    T_base_camera: np.ndarray
    depth: np.ndarray
    target_mask: np.ndarray
    intrinsics: dict

    def __post_init__(self):
        self.depth = np.asarray(self.depth, dtype=float)
        self.target_mask = np.asarray(self.target_mask, dtype=bool)
        if self.depth.ndim != 2 or self.depth.shape != self.target_mask.shape:
            raise ValueError("Evidence depth/mask shapes differ")
        self.T_base_camera = np.asarray(self.T_base_camera, dtype=float).reshape(4, 4)
        if not np.isfinite(self.T_base_camera).all():
            raise ValueError("Invalid evidence pose")
        if any(not np.isfinite(self.intrinsics[k]) for k in ('fx', 'fy', 'cx', 'cy')):
            raise ValueError("Invalid evidence intrinsics")
        if min(self.intrinsics['fx'], self.intrinsics['fy']) <= 0:
            raise ValueError("Invalid evidence focal length")
        valid = np.isfinite(self.depth) & (self.depth > .05) & (self.depth < 3.)
        # Reject depth discontinuities/invalid neighbors rather than carving
        # thin occlusion boundaries into free space.
        low = minimum_filter(np.where(valid, self.depth, np.inf), size=3)
        high = maximum_filter(np.where(valid, self.depth, -np.inf), size=3)
        self.valid = binary_erosion(valid, iterations=1) & ((high - low) <= .012)
        self.target_interior = binary_erosion(self.target_mask, iterations=2)


def classify_prediction(prediction, measured, observations=(), tolerance=.006):
    prediction, measured = xyz(prediction), xyz(measured)
    if tolerance <= 0 or not np.isfinite(tolerance):
        raise ValueError("Support tolerance must be positive")
    support = np.zeros(len(prediction), dtype=bool)
    conflict = np.zeros(len(prediction), dtype=bool)
    support_views = np.zeros(len(prediction), dtype=np.int32)
    conflict_views = np.zeros(len(prediction), dtype=np.int32)
    if len(measured) and len(prediction):
        support |= cKDTree(measured).query(prediction)[0] <= tolerance
    for view in observations:
        camera = transform(prediction, np.linalg.inv(view.T_base_camera))
        positive = camera[:, 2] > .05
        indices = np.flatnonzero(positive)
        projected = camera[indices]
        u = np.rint(view.intrinsics['fx'] * projected[:, 0] / projected[:, 2] + view.intrinsics['cx']).astype(int)
        v = np.rint(view.intrinsics['fy'] * projected[:, 1] / projected[:, 2] + view.intrinsics['cy']).astype(int)
        inside = (u >= 0) & (v >= 0) & (u < view.depth.shape[1]) & (v < view.depth.shape[0])
        indices, u, v = indices[inside], u[inside], v[inside]
        valid = view.valid[v, u]
        residual = camera[indices, 2] - view.depth[v, u]
        agrees = valid & view.target_interior[v, u] & (np.abs(residual) <= tolerance)
        disagrees = valid & (residual < -tolerance)
        support[indices[agrees]] = True
        conflict[indices[disagrees]] = True
        support_views[indices[agrees]] += 1
        conflict_views[indices[disagrees]] += 1
    states = np.full(len(prediction), UNKNOWN, dtype=np.uint8)
    states[support] = SUPPORTED
    states[conflict] = CONFLICT
    states[support & conflict] = INCONSISTENT
    return states, support_views, conflict_views


def measured_first_proposal(measured, prediction, observations=(), *, tolerance=.006,
                            duplicate_distance=.003, prediction_ratio=.5, voxel=.0025):
    measured, prediction = xyz(measured), xyz(prediction)
    if not len(measured):
        raise ValueError("Completion cannot replace an empty measurement")
    if not np.isfinite([duplicate_distance, prediction_ratio, voxel]).all() or min(duplicate_distance, voxel) <= 0 or not 0 <= prediction_ratio <= 1:
        raise ValueError("Invalid prediction sampling budget")
    states, support_views, conflict_views = classify_prediction(prediction, measured, observations, tolerance)
    distance = cKDTree(measured).query(prediction)[0]
    keep = (distance > duplicate_distance) & ~np.isin(states, [CONFLICT, INCONSISTENT])
    selected = np.flatnonzero(keep)
    selected = selected[representatives(prediction[selected], voxel)]
    budget = int(len(measured) * prediction_ratio)
    if len(selected) > budget:
        selected = selected[np.linspace(0, len(selected) - 1, budget, dtype=int)] if budget else selected[:0]
    points = np.vstack([measured, prediction[selected]])
    source = np.r_[np.zeros(len(measured), dtype=np.uint8), np.ones(len(selected), dtype=np.uint8)]
    report = dict(measured_points=len(measured), prediction_points=len(prediction),
                  added_prediction_points=len(selected), duplicate_prediction_points=int(np.sum(distance <= duplicate_distance)),
                  prediction_state_counts={name: int(np.sum(states == state)) for state, name in
                                           enumerate(('unknown', 'supported', 'conflict', 'inconsistent'))},
                  measured_coordinates_preserved=True, prediction_is_measurement=False,
                  support_distance_m=tolerance, duplicate_distance_m=duplicate_distance,
                  maximum_prediction_to_measured_ratio=prediction_ratio)
    return points, source, states, selected, report


def contact_evidence(T_base_tcp, measured, prediction, observations=(), *, opening=.100,
                     tolerance=.006, required_support=.75, finger_back=.065, finger_front=.005,
                     finger_radius=.007, band=.006):
    """Inspect local left/right surface bands; TCP uses Nero +X/+Z axes.

    This complements production contact/CAD/path gates and cannot qualify a
    grasp on its own. Prediction conflicts remain observation targets, not votes.
    """
    measured, prediction = xyz(measured), xyz(prediction)
    prediction_states, _, _ = classify_prediction(prediction, measured, observations, tolerance)
    # Rejected prediction must not reappear here and artificially enlarge the
    # contact span of an otherwise supported measured grasp.
    usable_prediction = prediction[~np.isin(prediction_states, [CONFLICT, INCONSISTENT])]
    geometry = np.vstack([measured, usable_prediction])
    geometry = geometry[representatives(geometry)]
    T = np.asarray(T_base_tcp, float).reshape(4, 4)
    local = (geometry - T[:3, 3]) @ T[:3, :3]
    slab = ((local[:, 0] > -finger_back) & (local[:, 0] < finger_front)
            & (np.abs(local[:, 1]) < finger_radius) & (np.abs(local[:, 2]) < opening / 2 + .01))
    indices = np.flatnonzero(slab)
    regions, targets, normals, weights = [], [], [], []
    if len(indices) < 6:
        return dict(sufficient=False, regions=[], target_points=np.empty((0, 3)),
                    target_normals=np.empty((0, 3)), target_weights=np.empty(0), reason='insufficient_local_geometry')
    z = local[indices, 2]
    lo, hi = np.quantile(z, [.05, .95])
    for side, edge, sign in [('left', lo, -1), ('right', hi, 1)]:
        chosen = indices[np.abs(z - edge) <= band]
        points = geometry[chosen]
        states, _, _ = classify_prediction(points, measured, observations, tolerance)
        supported = states == SUPPORTED
        bad = np.isin(states, [CONFLICT, INCONSISTENT])
        ratio = float(np.mean(supported)) if len(points) else 0.
        regions.append(dict(side=side, points=len(points), supported_fraction=ratio,
                            conflict_points=int(np.sum(bad)), sufficient=bool(len(points) >= 3 and ratio >= required_support and not np.any(bad))))
        unverified = ~supported
        requested = points[unverified]
        if len(requested) > 32:
            requested = requested[np.linspace(0, len(requested)-1, 32, dtype=int)]
        targets.extend(requested)
        normals.extend(np.tile(sign * T[:3, 2], (len(requested), 1)))
        weights.extend(np.full(len(requested), 1. / max(len(requested), 1)))
    return dict(sufficient=all(r['sufficient'] for r in regions), regions=regions,
                target_points=np.asarray(targets).reshape(-1, 3),
                target_normals=np.asarray(normals).reshape(-1, 3), target_weights=np.asarray(weights),
                reason='local_contact_evidence')


def visible_validation_gain(T_base_camera, points, normals, weights, occluders, tolerance=.008):
    """Coarse z-buffer proxy; this never certifies a free approach corridor."""
    points, occluders = xyz(points), xyz(occluders)
    if not len(points):
        return 0.
    inverse = np.linalg.inv(np.asarray(T_base_camera, float))
    camera = transform(points, inverse)
    toward = np.asarray(T_base_camera)[:3, 3] - points
    toward /= np.maximum(np.linalg.norm(toward, axis=1, keepdims=True), 1e-9)
    facing = np.clip(np.sum(toward * normals, axis=1), 0, 1)
    z = camera[:, 2]
    uv = camera[:, :2] / np.maximum(z[:, None], 1e-6)
    in_view = (z > .05) & (z < 2.) & (np.abs(uv).max(axis=1) < .65)
    bins = np.floor((uv + .7) / .014).astype(int)
    bins = np.clip(bins, 0, 99)
    depth_buffer = np.full(10000, np.inf)
    if len(occluders):
        cloud = transform(occluders, inverse)
        cloud = cloud[cloud[:, 2] > .05]
        projected = cloud[:, :2] / cloud[:, 2, None]
        inside = np.abs(projected).max(axis=1) < .7
        coords = np.floor((projected[inside] + .7) / .014).astype(int)
        np.minimum.at(depth_buffer, coords[:, 1] * 100 + coords[:, 0], cloud[inside, 2])
    visible = in_view & (z <= depth_buffer[bins[:, 1] * 100 + bins[:, 0]] + tolerance)
    return float(np.sum(np.asarray(weights) * facing * visible) / max(np.sum(weights), 1e-9))
