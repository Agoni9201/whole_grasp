"""ArUco eye-in-hand target source.

This module integrates the two old ROS pieces used during calibration:

    aruco_ros single
    real_visual_servo/aruco_pose_from_eye_in_hand.py

The old files are left untouched.  This project-local implementation detects
the marker from camera images, converts the camera-frame pose to base frame
with the verified hand-eye transform, and returns marker center / normal for
the Jacobian servo.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import cv2
import numpy as np
import numpy.typing as npt
import yaml

from models.nero_model import DEFAULT_TCP_OFFSET, NeroPinocchioModel


T_LINK7_TCP = np.eye(4, dtype=np.float64)
T_LINK7_TCP[:3, 3] = DEFAULT_TCP_OFFSET
T_TCP_LINK7 = np.linalg.inv(T_LINK7_TCP)


def quat_xyzw_to_matrix(quat_xyzw: Iterable[float]) -> npt.NDArray[np.float64]:
    x, y, z, w = [float(v) for v in quat_xyzw]
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm < 1e-12:
        raise ValueError("Quaternion norm is zero")
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def rodrigues_to_transform(
    rvec: npt.NDArray[np.float64],
    tvec: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    T = np.eye(4, dtype=np.float64)
    R, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))
    T[:3, :3] = R
    T[:3, 3] = np.asarray(tvec, dtype=np.float64).reshape(3)
    return T

def estimate_aruco_marker_pose(corners, marker_size, camera_matrix, dist_coeffs):
    """Pose of one ArUco marker. OpenCV >=4.7 dropped estimatePoseSingleMarkers."""
    camera_matrix = np.asarray(camera_matrix, dtype=np.float64).reshape(3, 3)
    dist_coeffs = np.asarray(dist_coeffs, dtype=np.float64).reshape(-1)
    points = np.asarray(corners, dtype=np.float64).reshape(4, 2)
    estimate = getattr(cv2.aruco, "estimatePoseSingleMarkers", None)
    if callable(estimate):
        rvecs, tvecs, _ = estimate(
            [points],
            float(marker_size),
            camera_matrix,
            dist_coeffs,
        )
        return (
            np.asarray(rvecs[0], dtype=np.float64).reshape(3),
            np.asarray(tvecs[0], dtype=np.float64).reshape(3),
        )
    half = float(marker_size) / 2.0
    object_points = np.array(
        [
            [-half, half, 0.0],
            [half, half, 0.0],
            [half, -half, 0.0],
            [-half, -half, 0.0],
        ],
        dtype=np.float64,
    )
    ok, rvec, tvec = cv2.solvePnP(
        object_points,
        points,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        return None
    return (
        np.asarray(rvec, dtype=np.float64).reshape(3),
        np.asarray(tvec, dtype=np.float64).reshape(3),
    )


def load_transform_file(path: str | Path) -> npt.NDArray[np.float64]:
    path = Path(path).expanduser()
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) if path.suffix.lower() in {".yaml", ".yml"} else json.load(f)

    if not isinstance(data, dict):
        raise ValueError(f"Hand-eye file must contain a mapping: {path}")

    candidates = []

    def add_pose(label, xyz, quat):
        pose = np.eye(4, dtype=np.float64)
        pose[:3, :3] = quat_xyzw_to_matrix(quat)
        pose[:3, 3] = np.asarray(xyz, dtype=np.float64).reshape(3)
        candidates.append((label, pose))

    if "position" in data and "orientation" in data:
        add_pose("position/orientation", data["position"], data["orientation"])
    if "translation" in data and "rotation_xyzw" in data:
        t = data["translation"]
        q = data["rotation_xyzw"]
        add_pose("translation/rotation_xyzw", [t["x"], t["y"], t["z"]],
                 [q["x"], q["y"], q["z"], q["w"]])
    # ChArUco exporter: metres and XYZW, with redundant representations of
    # the same camera-to-mount transform. Do not infer/invert frame direction
    # here; load_calibration() retains the explicit CLI convention handling.
    if "translation_m" in data and "quaternion_xyzw" in data:
        add_pose("translation_m/quaternion_xyzw", data["translation_m"], data["quaternion_xyzw"])
    if "flange_camera_xyzw" in data:
        pose = np.asarray(data["flange_camera_xyzw"], dtype=np.float64)
        if pose.shape != (7,):
            raise ValueError(f"flange_camera_xyzw must contain 7 values: {path}")
        add_pose("flange_camera_xyzw", pose[:3], pose[3:])
    if "matrix_4x4" in data:
        candidates.append(("matrix_4x4", np.asarray(data["matrix_4x4"], dtype=np.float64)))
    if not candidates:
        raise ValueError(f"Unsupported hand-eye file format: {path}; keys={sorted(data)}")

    for label, matrix in candidates:
        if (matrix.shape != (4, 4) or not np.isfinite(matrix).all()
                or not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-6, rtol=0)
                or not np.allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=1e-6, rtol=0)
                or not np.isclose(np.linalg.det(matrix[:3, :3]), 1., atol=1e-6, rtol=0)):
            raise ValueError(f"Invalid rigid hand-eye transform in {label}: {path}")
    T = candidates[0][1]
    for label, matrix in candidates[1:]:
        if not np.allclose(T, matrix, atol=1e-6, rtol=0):
            raise ValueError(f"Inconsistent hand-eye representations: {candidates[0][0]} and {label}: {path}")
    return T


def aruco_dictionary_names(name: str) -> list[str]:
    if name == "ARUCO_ROS_ALL":
        return [
            "DICT_ARUCO_ORIGINAL",
            "DICT_4X4_50",
            "DICT_4X4_100",
            "DICT_4X4_250",
            "DICT_4X4_1000",
            "DICT_5X5_50",
            "DICT_5X5_100",
            "DICT_5X5_250",
            "DICT_5X5_1000",
            "DICT_6X6_50",
            "DICT_6X6_100",
            "DICT_6X6_250",
            "DICT_6X6_1000",
            "DICT_7X7_50",
            "DICT_7X7_100",
            "DICT_7X7_250",
            "DICT_7X7_1000",
        ]
    if not hasattr(cv2.aruco, name):
        raise ValueError(f"OpenCV ArUco dictionary not found: {name}")
    return [name]


def aruco_dictionary(name: str):
    return cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, name))


@dataclass
class ArucoTargetResult:
    marker_center_base: npt.NDArray[np.float64]
    marker_normal_base: npt.NDArray[np.float64]
    marker_tangent_base: npt.NDArray[np.float64]
    T_base_marker: npt.NDArray[np.float64]
    T_camera_marker: npt.NDArray[np.float64]
    marker_id: int
    corners: npt.NDArray[np.float64]
    dictionary_name: str


class ArUcoEyeInHandTarget:
    """Detect wrist-camera ArUco and transform it into base frame."""

    def __init__(
        self,
        handeye_path: str,
        aruco_dict: str = "ARUCO_ROS_ALL",
        marker_id: int = 582,
        marker_size: float = 0.0677,
        handeye_convention: str = "link7_cam",
        handeye_mount_frame: str = "feedback_link7",
        offset_base: Optional[npt.ArrayLike] = None,
        offset_marker: Optional[npt.ArrayLike] = None,
        model: Optional[NeroPinocchioModel] = None,
    ) -> None:
        self.handeye_path = str(handeye_path)
        self.aruco_dict_name = aruco_dict
        self.marker_id = int(marker_id)
        self.marker_size = float(marker_size)
        self.handeye_convention = handeye_convention
        self.handeye_mount_frame = handeye_mount_frame
        self.offset_base = np.asarray(
            [0.0, 0.0, 0.0] if offset_base is None else offset_base,
            dtype=np.float64,
        ).reshape(3)
        self.offset_marker = np.asarray(
            [0.0, 0.0, 0.0] if offset_marker is None else offset_marker,
            dtype=np.float64,
        ).reshape(3)
        self.model = model if model is not None else NeroPinocchioModel()
        self.T_mount_cam = self.load_calibration()

        self._params = cv2.aruco.DetectorParameters()
        self._dictionary_names = aruco_dictionary_names(self.aruco_dict_name)
        self._detectors = [
            (name, cv2.aruco.ArucoDetector(aruco_dictionary(name), self._params))
            for name in self._dictionary_names
        ]

    def load_calibration(self) -> npt.NDArray[np.float64]:
        T = load_transform_file(self.handeye_path)
        if self.handeye_convention in {"cam_link7", "cam_tcp", "cam_mount"}:
            T = np.linalg.inv(T)
        return T

    def base_mount_transform(self, q_arm: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        q_arm = np.asarray(q_arm, dtype=np.float64).ravel()
        T_base_tcp = self.model.fk_to_4x4(q_arm)
        if self.handeye_mount_frame == "tcp_offset":
            return T_base_tcp
        if self.handeye_mount_frame == "feedback_link7":
            return T_base_tcp @ T_TCP_LINK7
        raise ValueError(f"Unsupported handeye_mount_frame: {self.handeye_mount_frame}")

    def detect_camera_marker(
        self,
        image: npt.NDArray[np.uint8],
        camera_matrix: npt.NDArray[np.float64],
        dist_coeffs: npt.NDArray[np.float64],
    ) -> Optional[tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], str]]:
        for dictionary_name, detector in self._detectors:
            corners, ids, _ = detector.detectMarkers(image)
            if ids is None or len(ids) == 0:
                continue

            ids_flat = ids.reshape(-1)
            matches = np.where(ids_flat == self.marker_id)[0]
            if len(matches) == 0:
                continue

            index = int(matches[0])
            marker_corners = corners[index]
            # rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
            #     [marker_corners],
            #     self.marker_size,
            #     np.asarray(camera_matrix, dtype=np.float64).reshape(3, 3),
            #     np.asarray(dist_coeffs, dtype=np.float64).ravel(),
            # )
            pose = estimate_aruco_marker_pose(
                marker_corners,
                self.marker_size,
                camera_matrix,
                dist_coeffs,
            )
            # T_camera_marker = rodrigues_to_transform(rvecs[0], tvecs[0])
            if pose is None:
                continue
            rvec, tvec = pose
            T_camera_marker = rodrigues_to_transform(rvec, tvec)
            return T_camera_marker, np.asarray(marker_corners, dtype=np.float64), dictionary_name
        return None

    def transform_to_base(
        self,
        q_arm: npt.NDArray[np.float64],
        T_camera_marker: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        T_base_mount = self.base_mount_transform(q_arm)
        T_base_marker = T_base_mount @ self.T_mount_cam @ T_camera_marker
        T_base_marker = T_base_marker.copy()
        T_base_marker[:3, 3] += T_base_marker[:3, :3] @ self.offset_marker
        T_base_marker[:3, 3] += self.offset_base
        return T_base_marker

    def detect(
        self,
        image: npt.NDArray[np.uint8],
        camera_matrix: npt.NDArray[np.float64],
        dist_coeffs: npt.NDArray[np.float64],
        q_arm: npt.NDArray[np.float64],
    ) -> Optional[ArucoTargetResult]:
        camera_result = self.detect_camera_marker(image, camera_matrix, dist_coeffs)
        if camera_result is None:
            return None

        T_camera_marker, corners, dictionary_name = camera_result
        T_base_marker = self.transform_to_base(q_arm, T_camera_marker)
        return ArucoTargetResult(
            marker_center_base=T_base_marker[:3, 3].copy(),
            marker_normal_base=T_base_marker[:3, 2].copy(),
            marker_tangent_base=T_base_marker[:3, 1].copy(),
            T_base_marker=T_base_marker,
            T_camera_marker=T_camera_marker,
            marker_id=self.marker_id,
            corners=corners,
            dictionary_name=dictionary_name,
        )
