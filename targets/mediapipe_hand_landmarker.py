"""MediaPipe Tasks hand landmarks with aligned RGB-D palm localization.

This module deliberately stops at the camera optical frame.  Hand-eye
calibration and robot-base transforms belong to the servo integration layer.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time

import cv2
import numpy as np


PALM_LANDMARK_INDICES = (0, 5, 9, 13, 17)
DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "hand_landmarker.task"


@dataclass(frozen=True)
class CameraHandObservation:
    """One detected hand expressed in the aligned color optical frame."""

    landmarks_normalized: np.ndarray
    landmarks_pixel: np.ndarray
    palm_pixel: tuple[int, int]
    palm_center_camera_m: np.ndarray | None
    palm_normal_camera: np.ndarray | None
    palm_tangent_camera: np.ndarray | None
    handedness: str
    handedness_score: float
    valid_depth_landmarks: int
    palm_depth_m: float
    palm_depth_mad_m: float


def robust_patch_depth(
    depth_m: np.ndarray,
    u: int,
    v: int,
    radius: int = 4,
    min_depth_m: float = 0.10,
    max_depth_m: float = 2.0,
) -> tuple[float, float, int]:
    """Return median depth, MAD, and valid-pixel count around one pixel."""
    if depth_m.ndim != 2:
        raise ValueError("depth_m must be HxW")
    if radius < 0 or not 0.0 < min_depth_m < max_depth_m:
        raise ValueError("invalid depth sampling parameters")
    height, width = depth_m.shape
    u = int(np.clip(u, 0, width - 1))
    v = int(np.clip(v, 0, height - 1))
    patch = np.asarray(
        depth_m[
            max(0, v - radius):min(height, v + radius + 1),
            max(0, u - radius):min(width, u + radius + 1),
        ],
        dtype=np.float64,
    )
    valid = patch[
        np.isfinite(patch)
        & (patch >= float(min_depth_m))
        & (patch <= float(max_depth_m))
    ]
    if valid.size == 0:
        return float("nan"), float("nan"), 0
    median = float(np.median(valid))
    mad = float(np.median(np.abs(valid - median)))
    return median, mad, int(valid.size)


def deproject_pinhole(u: float, v: float, depth_m: float, K: np.ndarray) -> np.ndarray:
    """Deproject an aligned color pixel into the color optical frame."""
    matrix = np.asarray(K, dtype=np.float64).reshape(3, 3)
    fx, fy = float(matrix[0, 0]), float(matrix[1, 1])
    if fx <= 0.0 or fy <= 0.0 or not np.isfinite(depth_m):
        raise ValueError("invalid intrinsics or depth")
    return float(depth_m) * np.array(
        [(float(u) - matrix[0, 2]) / fx, (float(v) - matrix[1, 2]) / fy, 1.0],
        dtype=np.float64,
    )


def _unit(vector: np.ndarray) -> np.ndarray | None:
    vector = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(vector))
    return None if norm < 1e-9 else vector / norm


class MediaPipeHandLandmarker:
    """Synchronous VIDEO-mode detector suitable for a persistent SDK stream."""

    def __init__(
        self,
        model_path: str | Path = DEFAULT_MODEL_PATH,
        num_hands: int = 1,
        min_detection_confidence: float = 0.5,
        min_presence_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        depth_radius: int = 4,
        min_depth_m: float = 0.10,
        max_depth_m: float = 2.0,
    ) -> None:
        if num_hands < 1:
            raise ValueError("num_hands must be positive")
        if depth_radius < 0 or not 0.0 < min_depth_m < max_depth_m:
            raise ValueError("invalid hand depth parameters")
        model = Path(model_path).expanduser().resolve()
        if not model.is_file():
            raise FileNotFoundError(f"MediaPipe hand model not found: {model}")

        try:
            import mediapipe as mp
        except ImportError as exc:
            raise ImportError("Install mediapipe in the active Python environment") from exc
        if not hasattr(mp, "tasks") or not hasattr(mp.tasks.vision, "HandLandmarker"):
            raise RuntimeError("MediaPipe Tasks HandLandmarker API is unavailable")

        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(model)),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_hands=int(num_hands),
            min_hand_detection_confidence=float(min_detection_confidence),
            min_hand_presence_confidence=float(min_presence_confidence),
            min_tracking_confidence=float(min_tracking_confidence),
        )
        self._mp = mp
        self._detector = mp.tasks.vision.HandLandmarker.create_from_options(options)
        self.depth_radius = int(depth_radius)
        self.min_depth_m = float(min_depth_m)
        self.max_depth_m = float(max_depth_m)
        self._last_timestamp_ms = -1

    def close(self) -> None:
        detector, self._detector = self._detector, None
        if detector is not None:
            detector.close()

    def __enter__(self) -> "MediaPipeHandLandmarker":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def detect(
        self,
        image_bgr: np.ndarray,
        depth_m: np.ndarray | None = None,
        K: np.ndarray | None = None,
        timestamp_ms: int | None = None,
    ) -> list[CameraHandObservation]:
        """Return RGB landmarks even without depth; metric fields then stay invalid."""
        if self._detector is None:
            raise RuntimeError("hand landmarker is closed")
        image_bgr = np.asarray(image_bgr)
        if image_bgr.ndim != 3 or image_bgr.shape[2] != 3 or image_bgr.dtype != np.uint8:
            raise ValueError("image_bgr must be HxWx3 uint8")
        matrix = None
        if depth_m is not None:
            depth_m = np.asarray(depth_m)
            if depth_m.shape != image_bgr.shape[:2]:
                raise ValueError("depth_m must be aligned to the color image")
            if K is None:
                raise ValueError("K is required with depth_m")
            matrix = np.asarray(K, dtype=np.float64).reshape(3, 3)

        if timestamp_ms is None:
            timestamp_ms = time.monotonic_ns() // 1_000_000
        timestamp_ms = max(int(timestamp_ms), self._last_timestamp_ms + 1)
        self._last_timestamp_ms = timestamp_ms
        rgb = np.ascontiguousarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        result = self._detector.detect_for_video(image, timestamp_ms)

        observations: list[CameraHandObservation] = []
        height, width = image_bgr.shape[:2]
        for hand_index, hand in enumerate(result.hand_landmarks):
            normalized = np.array(
                [[float(point.x), float(point.y), float(point.z)] for point in hand],
                dtype=np.float64,
            )
            pixels = np.empty((len(normalized), 2), dtype=np.int32)
            pixels[:, 0] = np.clip(np.rint(normalized[:, 0] * (width - 1)), 0, width - 1)
            pixels[:, 1] = np.clip(np.rint(normalized[:, 1] * (height - 1)), 0, height - 1)

            wrist_pixel = pixels[0].astype(np.float64)
            mcp_center_pixel = np.mean(pixels[[5, 9, 13, 17]], axis=0)
            palm_float = 0.35 * wrist_pixel + 0.65 * mcp_center_pixel
            palm_pixel = (
                int(np.clip(round(palm_float[0]), 0, width - 1)),
                int(np.clip(round(palm_float[1]), 0, height - 1)),
            )

            points_3d: dict[int, np.ndarray] = {}
            depth_samples: list[float] = []
            depth_mads: list[float] = []
            if depth_m is not None:
                center_depth, center_mad, center_count = robust_patch_depth(
                    depth_m, *palm_pixel, self.depth_radius,
                    self.min_depth_m, self.max_depth_m,
                )
                if center_count:
                    depth_samples.append(center_depth)
                    depth_mads.append(center_mad)
                for index in PALM_LANDMARK_INDICES:
                    u, v = (int(value) for value in pixels[index])
                    value, mad, count = robust_patch_depth(
                        depth_m, u, v, self.depth_radius,
                        self.min_depth_m, self.max_depth_m,
                    )
                    if count:
                        depth_samples.append(value)
                        depth_mads.append(mad)
                        points_3d[index] = deproject_pinhole(u, v, value, matrix)

            palm_depth = float("nan")
            palm_mad = float("nan")
            palm_center = None
            # Require agreement from several palm locations.  A single valid
            # patch can easily be the background seen through a depth hole.
            if len(depth_samples) >= 3:
                samples = np.asarray(depth_samples, dtype=np.float64)
                sample_median = float(np.median(samples))
                sample_mad = float(np.median(np.abs(samples - sample_median)))
                threshold = max(0.035, 3.0 * sample_mad)
                inliers = samples[np.abs(samples - sample_median) <= threshold]
                palm_depth = float(np.median(inliers))
                palm_mad = float(max(sample_mad, np.median(depth_mads)))
                palm_center = deproject_pinhole(*palm_pixel, palm_depth, matrix)

            normal = None
            tangent = None
            if all(index in points_3d for index in (0, 5, 9, 17)):
                wrist = points_3d[0]
                index_mcp = points_3d[5]
                middle_mcp = points_3d[9]
                pinky_mcp = points_3d[17]
                tangent = _unit(pinky_mcp - index_mcp)
                normal = _unit(np.cross(middle_mcp - wrist, pinky_mcp - index_mcp))
                if normal is not None and palm_center is not None and float(normal @ palm_center) > 0.0:
                    normal = -normal

            handedness = "Unknown"
            handedness_score = float("nan")
            if hand_index < len(result.handedness) and result.handedness[hand_index]:
                category = result.handedness[hand_index][0]
                handedness = str(
                    getattr(category, "category_name", None)
                    or getattr(category, "display_name", None)
                    or "Unknown"
                )
                handedness_score = float(getattr(category, "score", float("nan")))

            observations.append(CameraHandObservation(
                landmarks_normalized=normalized,
                landmarks_pixel=pixels,
                palm_pixel=palm_pixel,
                palm_center_camera_m=palm_center,
                palm_normal_camera=normal,
                palm_tangent_camera=tangent,
                handedness=handedness,
                handedness_score=handedness_score,
                valid_depth_landmarks=len(points_3d),
                palm_depth_m=palm_depth,
                palm_depth_mad_m=palm_mad,
            ))
        return observations

