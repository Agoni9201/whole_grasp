"""Pure pose, transform, and point-to-segment geometry helpers."""

import numpy as np
from scipy.spatial.transform import Rotation as R, Slerp


__all__ = [
    "_as_pose_list",
    "_make_T_from_pose",
    "_transform_points",
    "_point_segment_distances",
    "_pose_rpy_error_norm",
    "interpolate_pose",
]


def _as_pose_list(T):
    T = np.asarray(T, dtype=float).reshape(4, 4)
    rpy = R.from_matrix(T[:3, :3]).as_euler("xyz", degrees=False)
    return np.r_[T[:3, 3], rpy].astype(float)


def _make_T_from_pose(pose):
    pose = np.asarray(pose, dtype=float).reshape(6)
    transform = np.eye(4, dtype=float)
    transform[:3, :3] = R.from_euler("xyz", pose[3:]).as_matrix()
    transform[:3, 3] = pose[:3]
    return transform


def _transform_points(T, points):
    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    if points.shape[0] == 0:
        return points.copy()
    homogeneous = np.c_[
        points,
        np.ones((points.shape[0], 1), dtype=np.float32),
    ]
    return (
        np.asarray(T, dtype=float).reshape(4, 4) @ homogeneous.T
    ).T[:, :3].astype(np.float32)


def _point_segment_distances(points, a, b):
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    a = np.asarray(a, dtype=np.float32).reshape(3)
    b = np.asarray(b, dtype=np.float32).reshape(3)
    ab = b - a
    denom = float(np.dot(ab, ab))
    if denom <= 1e-12:
        return np.linalg.norm(pts - a.reshape(1, 3), axis=1)
    t = np.clip(((pts - a.reshape(1, 3)) @ ab) / denom, 0.0, 1.0)
    closest = a.reshape(1, 3) + t[:, None] * ab.reshape(1, 3)
    return np.linalg.norm(pts - closest, axis=1)


def _pose_rpy_error_norm(pose_a, pose_b):
    a = np.asarray(pose_a, dtype=float).reshape(6)
    b = np.asarray(pose_b, dtype=float).reshape(6)
    R_a = R.from_euler("xyz", a[3:])
    R_b = R.from_euler("xyz", b[3:])
    return float((R_b * R_a.inv()).magnitude())


def interpolate_pose(pose_a, pose_b, t):
    a = np.asarray(pose_a, dtype=float).reshape(6)
    b = np.asarray(pose_b, dtype=float).reshape(6)
    out = np.zeros(6, dtype=float)
    out[:3] = (1.0 - float(t)) * a[:3] + float(t) * b[:3]
    key_rots = R.from_euler("xyz", np.vstack([a[3:], b[3:]]))
    slerp = Slerp([0.0, 1.0], key_rots)
    out[3:] = slerp([float(t)]).as_euler("xyz", degrees=False)[0]
    return out
