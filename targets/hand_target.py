
"""Human-hand target for SDK RGB-D delivery servoing
MediaPipe Tasks supplies image landmarks. Aligned RealSense depth makes the
palm position metric in the color optical frame, then the validated wrist
hand-eye chain maps it into the robot base frame
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
# import cv2

# from targets.aruco_eye_in_hand_target import ArUcoEyeInHandTarget
from models.nero_model import NeroPinocchioModel

# try:
#     import mediapipe as mp
#     _MP = mp.solutions
# except ImportError:  # 未装 mediapipe 时给出明确错误
#     _MP = None

from targets.aruco_eye_in_hand_target import ArUcoEyeInHandTarget, T_TCP_LINK7
from targets.mediapipe_hand_landmarker import (
    DEFAULT_MODEL_PATH,
    MediaPipeHandLandmarker,
)
from targets.hand_selection import HandRoles, filter_hand_observations

# @dataclass
@dataclass(frozen=True)
class HandTargetResult:
    hand_center_base: np.ndarray
    hand_normal_base: np.ndarray
    hand_tangent_base: np.ndarray
    T_base_hand: np.ndarray
    # palm_pixel: tuple
    T_camera_hand: np.ndarray
    palm_pixel: tuple[int, int]
    handedness: str
    handedness_score: float
    valid_depth_landmarks: int


def _normalize(vector: np.ndarray, label: str) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(value))
    if norm < 1e-9 or not np.all(np.isfinite(value)):
        raise ValueError(f"invalid {label}")
    return value / norm


def _camera_hand_rotation(normal: np.ndarray, tangent: np.ndarray) -> np.ndarray:
    """Build a right-handed frame with hand normal as Z and tangent as Y."""
    z_axis = _normalize(normal, "hand normal")
    tangent = np.asarray(tangent, dtype=np.float64).reshape(3)
    tangent = tangent - float(tangent @ z_axis) * z_axis
    y_axis = _normalize(tangent, "hand tangent")
    x_axis = _normalize(np.cross(y_axis, z_axis), "hand binormal")
    y_axis = _normalize(np.cross(z_axis, x_axis), "orthogonal hand tangent")
    return np.column_stack((x_axis, y_axis, z_axis))


class HandTarget:
    def __init__(
        self,
        handeye_path: str,
        handeye_convention: str = "link7_cam",
        handeye_mount_frame: str = "feedback_link7",
        offset_base=None,
        model=None,
        model_path: str | Path = DEFAULT_MODEL_PATH,
        min_detection_confidence: float = 0.5,
    # ):
        # if _MP is None:
        #     raise ImportError("mediapipe not installed: pip install mediapipe")
        # # 复用 ArUco 目标的手眼标定加载 + 坐标链（含 link7_cam 约定 / feedback_link7 挂载）
        min_presence_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        depth_radius: int = 4,
        min_valid_depth_landmarks: int = 3,
        min_depth_m: float = 0.10,
        max_depth_m: float = 2.0,
        approach_mode: str = "camera_ray",
        lock_first_hand: bool = False,
    ) -> None:
        if approach_mode not in {"camera_ray", "palm_normal"}:
            raise ValueError(f"unsupported hand approach mode: {approach_mode}")
        if not 1 <= int(min_valid_depth_landmarks) <= 5:
            raise ValueError("min_valid_depth_landmarks must be in [1, 5]")
        self.approach_mode = approach_mode
        self.min_valid_depth_landmarks = int(min_valid_depth_landmarks)
        self.last_diagnostics = {"reason": "waiting_for_frame"}
        self.last_observation = None
        self.last_observations = []
        self.hand_roles = HandRoles() if lock_first_hand else None
        self._frame = ArUcoEyeInHandTarget(
            handeye_path=str(handeye_path),
            aruco_dict="ARUCO_ROS_ALL",
            marker_id=582,
            marker_size=0.05,
            handeye_convention=handeye_convention,
            handeye_mount_frame=handeye_mount_frame,
            offset_base=None if offset_base is None else np.asarray(offset_base, dtype=np.float64),
            model=model if model is not None else NeroPinocchioModel(),
        )
        # self._hands = _MP.hands.Hands(
        #     static_image_mode=False,
        #     max_num_hands=1,
        #     min_detection_confidence=float(min_detection_confidence),
        #     model_complexity=0,
        self._detector = MediaPipeHandLandmarker(
            model_path=model_path,
            num_hands=2 if lock_first_hand else 1,
            min_detection_confidence=min_detection_confidence,
            min_presence_confidence=min_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
            depth_radius=depth_radius,
            min_depth_m=min_depth_m,
            max_depth_m=max_depth_m,
        )
    def close(self) -> None:
        self._detector.close()

    def warmup(self, image_bgr, depth_m, camera_matrix):
        """Initialize inference before CAN control, without locking hand roles."""
        self._detector.detect(image_bgr, depth_m, camera_matrix)

    @property
    def tcp_camera_transform(self):
        """Rigid color-camera mounting, excluding synthetic target offsets."""
        frame = self._frame
        if frame.handeye_mount_frame == 'feedback_link7':
            return T_TCP_LINK7 @ frame.T_mount_cam
        if frame.handeye_mount_frame == 'tcp_offset':
            return frame.T_mount_cam.copy()
        raise ValueError(f'Unsupported camera mount: {frame.handeye_mount_frame}')

    def detect(self, image_bgr, depth_m, camera_matrix, dist_coeffs, q_arm):
        # """检测人手 → base 系手心位姿。depth_m 与图像同尺寸（米），无效值 NaN。"""
        # h, w = depth_m.shape[:2]
        # rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        # results = self._hands.process(rgb)
        # if results.multi_hand_landmarks is None:
        """Detect one hand and return its target frame in the robot base frame."""
        del dist_coeffs
        self.last_observation = None
        observations = self._detector.detect(image_bgr, depth_m, camera_matrix)
        self.last_observations = observations
        selection = {}
        if getattr(self, 'hand_roles', None) is not None:
            accepted, rejected = filter_hand_observations(observations, image_bgr.shape)
            observation = self.hand_roles.select(accepted, rejected)
            selection = dict(receiver_side=self.hand_roles.receiver,
                             receiver_presence=self.hand_roles.presence,
                             control_side=self.hand_roles.control,
                             candidate_count=len(observations), accepted_count=len(accepted),
                             rejected_candidates=rejected)
            if observation is None:
                reason = ('receiver_absent_2d' if self.hand_roles.presence == 'absent'
                          else 'receiver_acquiring_or_ambiguous')
                self.last_diagnostics = dict(reason=reason, **selection)
                return None
            observations = [observation]
        if not observations:
            self.last_diagnostics = {"reason": "no_hand_2d"}
            return None
        # lm = results.multi_hand_landmarks[0].landmark

        # def _to3d(i):
        #     u = int(round(lm[i].x * (w - 1)))
        #     v = int(round(lm[i].y * (h - 1)))
        #     u = int(np.clip(u, 0, w - 1))
        #     v = int(np.clip(v, 0, h - 1))
        #     d = self._median_depth(depth_m, u, v)
        #     if not np.isfinite(d):
        #         return None
        #     return self._unproject(u, v, d, camera_matrix)

        # p_wrist = _to3d(0)
        # p_index = _to3d(5)
        # p_middle = _to3d(9)
        # p_pinky = _to3d(17)
        # if any(p is None for p in (p_wrist, p_index, p_middle, p_pinky)):
        observation = observations[0]
        self.last_observation = observation
        self.last_diagnostics = {
            "reason": "invalid_palm_depth",
            "valid_depth_landmarks": observation.valid_depth_landmarks,
            "required_depth_landmarks": self.min_valid_depth_landmarks,
            **selection,
        }
        if (observation.palm_center_camera_m is None
                or observation.valid_depth_landmarks < self.min_valid_depth_landmarks):
            return None
        center = np.asarray(
            observation.palm_center_camera_m, dtype=np.float64
        ).reshape(3)


        # # 掌心中心：手腕点（递送时最稳定）
        # center = p_wrist
        # # 手面法向：虎口→中指 与 食指→小指 叉积；翻向相机一侧
        # normal = np.cross(p_middle - p_wrist, p_pinky - p_index)
        # n = np.linalg.norm(normal)
        # if n < 1e-9:
        #     return None
        # normal = normal / n
        # if float(normal @ center) > 0.0:  # 相机在原点，法向指向相机
        #     normal = -normal
        # tangent = p_pinky - p_index
        # t = np.linalg.norm(tangent)
        # if t < 1e-9:
        #     return None
        # tangent = tangent / t
        # bi = np.cross(normal, tangent)
        if self.approach_mode == "palm_normal":
            if (observation.palm_normal_camera is None
                    or observation.palm_tangent_camera is None):
                self.last_diagnostics["reason"] = "palm_orientation_invalid"
                return None
            normal = observation.palm_normal_camera
            tangent = observation.palm_tangent_camera
        else:
            # 首轮只跟位置：沿相机观察射线接近，不使用噪声较大的掌面姿态。
            normal = _normalize(-center, "camera-to-hand viewing ray")
            tangent_seed = np.array([0.0, 1.0, 0.0], dtype=np.float64)
            if abs(float(tangent_seed @ normal)) > 0.95:
                tangent_seed = np.array([1.0, 0.0, 0.0], dtype=np.float64)
            tangent = tangent_seed

        # R_camera_hand = np.column_stack([tangent, bi, normal])
        # T_camera_hand = np.eye(4)
        # T_camera_hand[:3, :3] = R_camera_hand
        T_camera_hand = np.eye(4, dtype=np.float64)
        T_camera_hand[:3, :3] = _camera_hand_rotation(normal, tangent)
        T_camera_hand[:3, 3] = center
        T_base_hand = self._frame.transform_to_base(q_arm, T_camera_hand)
        self.last_diagnostics.update(
            reason="ok", palm_camera_mm=np.round(center * 1000, 1).tolist(),
        )

        return HandTargetResult(
            hand_center_base=T_base_hand[:3, 3].copy(),
            hand_normal_base=T_base_hand[:3, 2].copy(),
            hand_tangent_base=T_base_hand[:3, 1].copy(),
            T_base_hand=T_base_hand,
            # palm_pixel=(int(lm[0].x * (w - 1)), int(lm[0].y * (h - 1))),
            T_camera_hand=T_camera_hand,
            palm_pixel=observation.palm_pixel,
            handedness=observation.handedness,
            handedness_score=observation.handedness_score,
            valid_depth_landmarks=observation.valid_depth_landmarks,
        )

    # @staticmethod
    # def _median_depth(depth_m, u, v, half=2):
    #     patch = depth_m[max(0, v - half):v + half + 1, max(0, u - half):u + half + 1]
    #     patch = patch[np.isfinite(patch)]
    #     return float(np.median(patch)) if patch.size else np.nan

    # @staticmethod
    # def _unproject(u, v, d, K):
    #     return d * np.array([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], 1.0], dtype=np.float64)
