#!/usr/bin/env python3
"""伺服递送主程序：腕部相机 + 人手目标 + PBVS + 200 Hz 笛卡尔阻抗。

两种运行方式：
  1) 由 runtime hook 调用（--phase all --enable-servo-delivery，backend 已由 handoff 建好）
  2) 独立调试（--phase servo）：backend=None，内部自建；需先把臂摆到安全位姿
"""
from __future__ import annotations

import argparse
import logging
import os
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import numpy as np
from cameras import sdk_loop
from cameras.sdk_loop import Node
from cameras.sdk_loop import CameraInfo, Image

from models.nero_model import NeroPinocchioModel
from servo.pbvs_pose_servo import PBVSPoseServo, PBVSParams
from servo.execution_safety import ExecutionSafetyMonitor, SafetyParams
from servo.hand_follow_safety import HandFollowSafetyMonitor, HandFollowLimits
from servo.hand_position_memory import HandPositionMemory, HandMemoryLimits
from control_backends.force_ctrl.data_types import ImpedanceConfig
from control_backends.force_ctrl.normal_shutdown import defer_sigint
from handoff.fusion_servo import FusionServoMixin
from control_backends.force_ctrl.impedance_profile import (
    VALIDATED_CARTESIAN_KP, VALIDATED_CARTESIAN_KD,
    VALIDATED_JOINT_WEIGHTS, VALIDATED_COULOMB,
    VALIDATED_TASK_TORQUE_LIMIT, VALIDATED_TOTAL_TORQUE_LIMIT, VALIDATED_VELOCITY_SIGNS,
)

logger = logging.getLogger(__name__)
_NO_NEW_FRAME = object()
_TARGET_TIMEOUT_S = 0.45
_TARGET_LOSS_DECAY_S = 0.30
_JUMP_REACQUIRE_S = 0.40
_JUMP_REACQUIRE_POSITION_M = 0.02

# --------------------------------------------------------------------------
# 名义目标轨迹生成/限幅/防饱和；超出偏差上限时也按参考限速恢复。
# --------------------------------------------------------------------------

def _bounded_rate_step(velocity, dt, max_rate):
    value = np.asarray(velocity, dtype=np.float64).ravel()
    duration = max(0.0, float(dt))
    limit = max(0.0, float(max_rate)) * duration
    step = value * duration
    norm = float(np.linalg.norm(step))
    if limit <= 0.0:
        return np.zeros(3, dtype=np.float64)
    if norm > limit and norm > 1e-12:
        step *= limit / norm
    return step


def integrate_position_target_from_twist(equilibrium_tcp, linear_velocity_world, dt, max_linear_vel):
    current = np.asarray(equilibrium_tcp, dtype=np.float64).ravel()
    return current + _bounded_rate_step(linear_velocity_world, dt, max_linear_vel)


def constrain_impedance_target_twist(tcp_twist, max_linear_vel, max_angular_vel, include_orientation=False):
    twist = np.asarray(tcp_twist, dtype=np.float64).ravel()
    constrained = twist.copy()
    constrained[:3] = _bounded_rate_step(constrained[:3], 1.0, max(0.0, float(max_linear_vel)))
    if include_orientation:
        constrained[3:] = _bounded_rate_step(constrained[3:], 1.0, max(0.0, float(max_angular_vel)))
    else:
        constrained[3:] = 0.0
    return constrained


def apply_position_target_anti_windup(previous_target, candidate_target, current_tcp, max_error):
    """Project onto the force leash, with rate-bounded recovery after a push.

    When measured motion puts an existing reference outside the leash, freezing
    every outward/tangential update can lock it there indefinitely. Move toward
    the projected candidate by at most the original integration step instead.
    Zero requested motion keeps the reference fixed, including on target loss.
    """
    previous = np.asarray(previous_target, dtype=np.float64).ravel()
    candidate = np.asarray(candidate_target, dtype=np.float64).ravel()
    current = np.asarray(current_tcp, dtype=np.float64).ravel()
    if max_error is None:
        return candidate.copy()
    bound = max(0.0, float(max_error))
    previous_error = float(np.linalg.norm(previous - current))
    candidate_delta = candidate - current
    candidate_error = float(np.linalg.norm(candidate_delta))
    tolerance = 1e-12
    projected = (candidate.copy() if candidate_error <= bound + tolerance
                 else current + candidate_delta * (bound / candidate_error))
    if previous_error <= bound + tolerance:
        return projected
    step_limit = float(np.linalg.norm(candidate - previous))
    recovery = projected - previous
    recovery_norm = float(np.linalg.norm(recovery))
    if recovery_norm > step_limit and recovery_norm > tolerance:
        recovery *= step_limit / recovery_norm
    return previous + recovery


def advance_orientation_reference(previous, angular_velocity, actual, dt, max_rate, max_error):
    """World-frame rotation integration with a rate-bounded orientation leash."""
    from scipy.spatial.transform import Rotation
    step = _bounded_rate_step(angular_velocity, dt, max_rate)
    candidate = Rotation.from_rotvec(step).as_matrix() @ previous
    error = Rotation.from_matrix(candidate @ actual.T).as_rotvec()
    norm = float(np.linalg.norm(error))
    bound = max(0., float(max_error))
    projected = (candidate if norm <= bound else
                 Rotation.from_rotvec(error * (bound / norm)).as_matrix() @ actual)
    recovery = Rotation.from_matrix(projected @ previous.T).as_rotvec()
    return Rotation.from_rotvec(_bounded_rate_step(
        recovery, 1., np.linalg.norm(step))).as_matrix() @ previous


def image_msg_to_bgr(msg: Image) -> np.ndarray:
    """sensor_msgs/Image → BGR uint8（支持 rgb8/bgr8/mono8，RealSense 默认 rgb8）。"""
    data = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
    if msg.encoding in ("rgb8",):
        return data[:, :, ::-1].copy()
    if msg.encoding in ("bgr8",):
        return data.copy()
    if msg.encoding in ("mono8",):
        return data.copy()
    raise ValueError(f"unsupported image encoding: {msg.encoding}")


class FakeBackend:
    """离线 dry-run 后端：不发 CAN，只记录配置。"""

    def __init__(self):
        self.q = np.zeros(7)
        self.cfg = None
        self.disabled = False

    def connect(self): pass
    def disconnect(self): pass
    def get_joint_angles(self): return self.q.copy()
    def get_joint_velocity(self): return np.zeros(7)
    def get_debug_state(self): return {}
    def set_cartesian_impedance(self, cfg): self.cfg = cfg
    def disable_cartesian_impedance(self): self.disabled = True
    def emergency_stop(self): pass
    def get_robot(self): raise RuntimeError("dry-run has no robot")


def build_target(args, model):
    if args.target_source == "aruco":
        from targets.aruco_eye_in_hand_target import ArUcoEyeInHandTarget
        return ArUcoEyeInHandTarget(
            handeye_path=args.handeye,
            aruco_dict=args.aruco_dict,
            marker_id=args.marker_id,
            marker_size=args.marker_size,
            handeye_convention=args.wrist_handeye_convention,
            handeye_mount_frame=args.wrist_handeye_mount_frame,
            offset_base=np.fromstring(args.offset_base, sep=","),
            model=model,
        )
    if args.target_source == "mediapipe":
        from targets.hand_target import HandTarget
        return HandTarget(
            handeye_path=args.handeye,
            handeye_convention=args.wrist_handeye_convention,
            handeye_mount_frame=args.wrist_handeye_mount_frame,
            # offset_base=np.fromstring(args.offset_base, sep=","),
            offset_base=np.fromstring(args.hand_offset_base, sep=","),
            model=model,
            model_path=args.hand_model,
            min_detection_confidence=args.hand_min_detection_confidence,
            min_presence_confidence=args.hand_min_presence_confidence,
            min_tracking_confidence=args.hand_min_tracking_confidence,
            depth_radius=args.hand_depth_radius,
            min_valid_depth_landmarks=args.hand_min_depth_landmarks,
            min_depth_m=args.hand_min_depth,
            max_depth_m=args.hand_max_depth,
            approach_mode=args.hand_approach_mode,
            lock_first_hand=bool(getattr(args, 'gesture_handover', False) or getattr(args, 'fusion_handover', None)),
        )
    raise ValueError(f"unknown --target-source: {args.target_source}")


class ServoDeliveryNode(FusionServoMixin, Node):
    """递送状态机：HOLD_LIFTED→TRACK_HAND→APPROACH→HANDOVER→RELEASE→RETREAT→DONE。"""

    def __init__(self, backend, args, model, target=None, preview=None, diagnostics=None):
        super().__init__("servo_delivery")
        self.backend = backend
        self.args = args
        self.model = model
        self.target = target if target is not None else build_target(args, model.clone())
        self._preview = preview
        self._diagnostics = diagnostics
        self._diag_sample = None
        self._preview_last_sent = 0.0
        self._preview_status = "WAITING"
        self._preview_reason = "waiting_for_frame"
        self._preview_base = None
        self._preview_error_mm = "--"
        self.servo = PBVSPoseServo(params=self._servo_params(), model=model.clone())
        if bool(getattr(args, 'hand_camera_center', False)):
            from servo.hand_camera_center import HandCameraCenterServo
            from control_backends.force_ctrl.mit_loop import NERO_JOINT_LIMITS, JOINT_LIMIT_BUFFER
            bounds = getattr(backend, 'get_servo_joint_bounds', lambda:
                             NERO_JOINT_LIMITS + [JOINT_LIMIT_BUFFER, -JOINT_LIMIT_BUFFER])()
            p = self._servo_params()
            p.max_linear_vel = min(p.max_linear_vel, float(args.impedance_target_linear_vel))
            p.max_angular_vel = min(p.max_angular_vel, float(args.impedance_target_angular_vel))
            self.servo = HandCameraCenterServo(
                p, model.clone(), self.target.tcp_camera_transform, bounds,
                gain=args.hand_center_gain, max_accel=args.hand_center_max_joint_accel)
            self.get_logger().info('[servo] whole-arm camera centering enabled; joint headroom checked before motion')
        self.safety = ExecutionSafetyMonitor(params=SafetyParams())
        if bool(getattr(args, 'hand_camera_center', False)):
            self.safety = HandFollowSafetyMonitor(model.clone(),
                position_leash=args.impedance_max_position_error,
                joint_leash=args.follow_max_error_rad)
        self.gripper = None
        if not isinstance(backend, FakeBackend):
            from handoff.grasp_to_servo_handoff import BackendGripper
            existing = getattr(args, "_servo_gripper", None)
            if existing is not None:
                self.gripper = existing
            else:
                self.gripper = BackendGripper(backend.get_robot())

        self.state = "HOLD_LIFTED"
        self.state_entered = time.time()
        self.stable_frames = 0
        self.retreat_target = None
        self._last_lost_log = 0.0 #限制 target lost 日志频率
        self._latest_depth = None
        self._consumed_sequence = None
        self._release_samples = 0
        self._latest_image = None
        self._latest_info = None
        self._latest_q = None
        self._last_target_time = 0.0
        self._target_lost = True
        self._reacquire_count = 0
        self._reacquire_target = None
        self._rejected_target = None
        self._detection_reason = "waiting_for_frame"
        self._last_tracking_log = 0.0
        self._shutdown_prepared = False
        self._emergency_stop_requested = False
        self._qdot_window = deque(maxlen=int(1.0 / max(0.01, float(args.servo_hz)) * 2))

        self._imp_engaged = False
        self._hold_q_ref = None
        self._hold_stable = 0
        self._imp_cfg = None
        self._imp_smoothed_tcp = None
        self._imp_smoothed_R = None
        self._imp_follow_q = None
        self._imp_follow_kp_active = np.zeros(7)
        self._imp_target_update_time = None
        # Keep the last accepted command so a short detector dropout can
        # decelerate toward a hold instead of switching the MIT reference off
        # in one outer-loop tick.
        self._imp_last_target_twist = np.zeros(6, dtype=np.float64)
        self._imp_last_follow_kp = np.zeros(7, dtype=np.float64)
        self._gesture = None
        self._hand_memory = None
        self._gesture_joint_history = deque(maxlen=40)
        if bool(getattr(args, 'gesture_handover', False)):
            if getattr(args, 'hand_memory_approach', True):
                self._hand_memory = HandPositionMemory()
            from handoff.gesture_handover import GestureHandover, wait_initial_opening
            if isinstance(backend, FakeBackend):
                width = float(args.gesture_preview_grip_width_mm) / 1000.
                self._gesture_sim_width = width
            else:
                width, _, _ = wait_initial_opening(self.gripper)
                if width <= float(getattr(args, 'gripper_closed_width_tol', .001)):
                    raise ValueError('gripper appears fully closed; gesture handover requires a held cup')
            previous_command = getattr(self.gripper, 'last_command', {}) if self.gripper is not None else {}
            self._gesture_force = min(float(getattr(args, 'gripper_force', .5)), .5,
                float(getattr(args, 'gripper_hardware_max_force', .5)),
                float(previous_command.get('force', .5)))
            if not np.isfinite(self._gesture_force) or self._gesture_force <= 0.:
                raise ValueError('invalid gesture gripper force limit')
            maximum = min(float(getattr(args, 'gripper_open_width', .085)),
                          float(getattr(args, 'gripper_hardware_max_width', .085)), .085)
            self._gesture = GestureHandover(width, distance=args.delivery_distance,
                ready_distance=args.gesture_ready_distance,
                maximum=maximum,
                tolerance=args.gesture_near_tolerance,
                stable_seconds=args.gesture_stable_seconds,
                still_tolerance=args.gesture_still_tolerance_mm/1000.,
                extra_close=args.gesture_extra_close_mm/1000.,
                open_seconds=args.gesture_open_seconds, close_seconds=args.gesture_close_seconds)
            self.get_logger().info('[handover] gesture mode; fixed cup grip opening %.1fmm; only opposite hand controls release', width*1000)

        self._init_fusion(simulated=isinstance(backend, FakeBackend))

        self.create_subscription(
            Image, args.wrist_image_topic, self._on_image, qos_profile=1)
        self.create_subscription(
            CameraInfo, args.wrist_info_topic, self._on_info, qos_profile=1)
        self.create_timer(1.0 / float(args.servo_hz), self._on_timer)

    # ---------- 参数 ----------
    def _servo_params(self):
        p = PBVSParams()
        p.servo_hz = float(self.args.servo_hz)
        p.desired_distance = float(self.args.delivery_distance)
        p.max_linear_vel = float(self.args.max_linear_vel)
        p.max_angular_vel = float(self.args.max_angular_vel)
        p.max_joint_vel = float(self.args.max_joint_vel)
        p.enable_orientation = bool(self.args.visual_orientation)
        p.marker_center_alpha = float(getattr(self.args, "target_filter_alpha", p.marker_center_alpha))
        p.marker_normal_alpha = float(getattr(self.args, "normal_filter_alpha", p.marker_normal_alpha))
        p.marker_tangent_alpha = p.marker_normal_alpha
        p.target_blend_alpha = float(getattr(self.args, "target_blend_alpha", p.target_blend_alpha))
        p.tcp_twist_smooth_alpha = float(getattr(self.args, "twist_smooth_alpha", p.tcp_twist_smooth_alpha))
        p.target_timeout_s = 0.60
        return p

    # ---------- 订阅 ----------
    def _on_image(self, msg):
        self._latest_image = msg
        self._latest_q = self.backend.get_joint_angles()

    def _on_info(self, msg):
        self._latest_info = msg

    # ---------- 检测 ----------
    def _detect_latest(self, q_now, now):
        if self._latest_image is None or self._latest_info is None:
            self._detection_reason = "waiting_for_frame"
            return None
        sequence = getattr(self, '_sdk_frame_sequence', None)
        if time.monotonic() - getattr(self, '_sdk_received_monotonic', 0.) > _TARGET_TIMEOUT_S:
            self._detection_reason = "camera_frame_stale"
            return None
        if sequence == self._consumed_sequence:
            return _NO_NEW_FRAME
        self._consumed_sequence = sequence
        K = np.asarray(self._latest_info.k, dtype=np.float64).reshape(3, 3)
        if bool(getattr(self.args, 'hand_camera_center', False)):
            self.servo.set_camera_intrinsics(K, self._latest_image.width, self._latest_image.height)
        D = np.asarray(self._latest_info.d, dtype=np.float64).reshape(-1)
        bgr = image_msg_to_bgr(self._latest_image)
        q_image = np.asarray(self._latest_q, dtype=np.float64).ravel()
        if q_image.shape != (7,) or not np.all(np.isfinite(q_image)):
            self._detection_reason = "image_joint_feedback_invalid"
            return None
        if self.args.target_source == "aruco":
            # res = self.target.detect(bgr, K, D, q_now)
            res = self.target.detect(bgr, K, D, q_image)
            if res is None:
                self._detection_reason = "aruco_not_detected"
                return None
            self._detection_reason = "ok"
            return res.marker_center_base, res.marker_normal_base, res.marker_tangent_base
        # mediapipe：需要深度图（与 ArUco 不同，另加一个深度订阅的场景见 README）
        # MediaPipe Tasks：SDK loop 已提供与彩色图对齐的米制深度。
        depth = self._latest_depth
        if depth is None and not bool(getattr(self.args, 'gesture_handover', False)):
            self._detection_reason = "aligned_depth_missing"
            return None
        from cameras.realsense import depth_with_current_support
        filtered_depth = depth
        raw_depth = getattr(self, '_latest_raw_depth', None)
        if raw_depth is not None and depth is not None:
            depth = depth_with_current_support(depth, raw_depth,
                                               self.args.hand_min_depth, self.args.hand_max_depth)
        # res = self.target.detect(bgr, depth, K, D, q_now)
        res = self.target.detect(bgr, depth, K, D, q_image)
        # Keep evidence from this exact detection/frame, never a later camera
        # frame or an observation left behind by a duplicate timer tick.
        if getattr(self, "_diagnostics", None) is not None:
            detail = dict(getattr(self.target, "last_diagnostics", {}))
            observation = getattr(self.target, "last_observation", None)
            self._diag_note(perception=detail, rgbd_frame={
                "bgr": bgr, "depth_m": depth, "raw_depth_m": raw_depth,
                "filtered_depth_m": filtered_depth, "K": K, "D": D,
                "q_image": q_image.copy(), "sequence": sequence,
                "received_monotonic": self._sdk_received_monotonic,
                "reason": detail.get("reason", "unknown"),
                "radius": int(self.args.hand_depth_radius),
                "minimum_m": float(self.args.hand_min_depth),
                "maximum_m": float(self.args.hand_max_depth),
                "landmarks_pixel": None if observation is None else observation.landmarks_pixel,
                "palm_pixel": None if observation is None else observation.palm_pixel,
            })
        if time.monotonic() - self._sdk_received_monotonic > _TARGET_TIMEOUT_S:
            self._detection_reason = "camera_frame_expired_during_detection"
            return None
        diagnostics = getattr(self.target, "last_diagnostics", {})
        if res is None:
            self._detection_reason = str(diagnostics.get("reason", "hand_not_detected"))
            return None
        self._detection_reason = "ok"
        return res.hand_center_base, res.hand_normal_base, res.hand_tangent_base

    def _pause_target_motion(self, now, *, immediate=False):
        self.stable_frames = 0
        pause = getattr(getattr(self, 'servo', None), 'pause', None)
        if bool(getattr(getattr(self, 'args', None), 'hand_camera_center', False)) and callable(pause):
            pause()
        self._imp_target_update_time = now
        if self._imp_cfg is not None:
            # A brief MediaPipe dropout is common in a live RGB-D stream.  A
            # linear decay preserves a bounded, predictable stop while
            # avoiding the visible kick caused by an immediate zero command.
            last_target_time = float(getattr(self, "_last_target_time", 0.0))
            loss_age = max(0.0, float(now) - last_target_time)
            decay = 0.0 if immediate else max(0.0, 1.0 - loss_age / _TARGET_LOSS_DECAY_S)
            last_twist = np.asarray(
                getattr(self, "_imp_last_target_twist", self._imp_cfg.target_twist),
                dtype=np.float64,
            ).reshape(6)
            last_follow_kp = np.asarray(
                getattr(self, "_imp_last_follow_kp", self._imp_cfg.follow_kp),
                dtype=np.float64,
            ).reshape(7)
            self._imp_cfg.target_twist = last_twist * decay
            self._imp_cfg.follow_kp = last_follow_kp * decay
            self._imp_follow_kp_active = self._imp_cfg.follow_kp.copy()
            if immediate:
                # A later detector dropout must not revive a pre-HOLD moving
                # reference from the soft-decay cache.
                self._imp_last_target_twist = np.zeros(6, dtype=np.float64)
                self._imp_last_follow_kp = np.zeros(7, dtype=np.float64)
            try:
                self.backend.set_cartesian_impedance(self._imp_cfg)
            except Exception:
                self._diag_note(reference_write_failed=True)
                raise

    def _forget_lost_target(self, q_now, now):
        if not self._target_lost:
            self.servo.update(q_now, None, stamp=now)
        self._target_lost = True
        self._reacquire_count = 0
        self._reacquire_target = None
        self._rejected_target = None
        # A long dropout is a genuine target reset.  Do not resurrect the
        # pre-loss velocity/reference when the first new frame arrives.
        self._imp_last_target_twist = np.zeros(6, dtype=np.float64)
        self._imp_last_follow_kp = np.zeros(7, dtype=np.float64)
        self._imp_follow_kp_active = np.zeros(7, dtype=np.float64)
        if self._imp_cfg is not None:
            self._imp_cfg.target_twist = np.zeros(6, dtype=np.float64)
            self._imp_cfg.follow_kp = np.zeros(7, dtype=np.float64)

    def _accept_reacquired_target(self, target):
        center, normal, _ = target
        center = np.asarray(center, dtype=float)
        normal = np.asarray(normal, dtype=float)
        previous = self._reacquire_target
        consistent = (
            previous is not None
            and np.linalg.norm(center - previous[0]) <= 0.02
            and float(normal @ previous[1]) >= np.cos(np.deg2rad(15.0))
        )
        self._reacquire_count = self._reacquire_count + 1 if consistent else 1
        self._reacquire_target = (center.copy(), normal.copy())
        if self._reacquire_count < 3:
            return False
        self._target_lost = False
        self.get_logger().warning("[servo][target] acquired after 3 consistent new RGB-D frames")
        return True

    def _try_reacquire_rejected_target(self, q_now, target, now):
        """Reconsider a stable new hand location without removing jump rejection.

        Only new valid RGB-D observations reach this method. A fixed spatial
        anchor prevents slow drift from being counted as a stable target.
        Ordinary PBVS retains its TCP-error envelope. Camera centering permits
        bounded approach to distant goals, including outside the reach sphere;
        the impedance reference itself is never teleported to the new goal.
        """
        if getattr(self.args, 'target_source', None) != 'mediapipe':
            return None
        center, normal, tangent = target
        center = np.asarray(center, dtype=float).reshape(3)
        normal = np.asarray(normal, dtype=float).reshape(3)
        if not np.isfinite(center).all() or not np.isfinite(normal).all():
            self._rejected_target = None
            return None
        length = float(np.linalg.norm(normal))
        if length < 1e-8:
            self._rejected_target = None
            return None
        normal = normal / length
        previous = getattr(self, '_rejected_target', None)
        consistent = (
            previous is not None and 0 <= now - previous['updated'] <= _TARGET_TIMEOUT_S
            and np.linalg.norm(center - previous['center']) <= _JUMP_REACQUIRE_POSITION_M
            and float(normal @ previous['normal']) >= np.cos(np.deg2rad(15.0))
        )
        if not consistent:
            self._rejected_target = dict(center=center.copy(), normal=normal.copy(),
                                         since=now, updated=now, count=1)
            return None
        previous['updated'] = now
        previous['count'] += 1
        if previous['count'] < 3 or now - previous['since'] < _JUMP_REACQUIRE_S:
            return None
        tcp_now = np.asarray(self.model.fk(q_now)[0], dtype=float).reshape(3)
        # Same unfiltered geometry as PBVS after a fresh target acquisition.
        if float((tcp_now - center) @ normal) < 0:
            normal = -normal
        candidate = center + normal * float(self.servo.params.desired_distance)
        max_error = float(self.safety.params.max_tracking_error)
        error = float(np.linalg.norm(candidate - tcp_now))
        outside = (not isinstance(self.safety, HandFollowSafetyMonitor)
                   and (max_error <= 0 or error > max_error))
        if outside:
            self._diag_note(jump_reacquire='outside_tracking_envelope',
                            jump_reacquire_error_m=error)
            return None
        self.servo.update(q_now, None, stamp=now)
        out = self.servo.update(q_now, center, normal, tangent, None, now)
        self._rejected_target = None
        self._imp_follow_q = np.asarray(q_now, dtype=float).copy()
        self._diag_note(jump_reacquire='accepted_stable_target',
                        jump_reacquire_error_m=error)
        self.get_logger().warning(
            '[servo][target] stable jumped hand target reacquired; TCP error=%.1fmm; reference remains rate limited',
            error * 1000)
        return out

    # ---------- 阻抗：engage / refresh（移植 VS _build_impedance_command 的核心）----------
    def _engage(self, q_now, now):
        if bool(getattr(self.args, 'hand_camera_center', False)) and hasattr(self, 'servo'):
            bounds = self.servo.joint_bounds
            margin = np.minimum(q_now - bounds[:, 0], bounds[:, 1] - q_now)
            if np.min(margin) <= 0:
                self._stop_task('camera centering start is outside software joint bounds')
                return
            self.servo.pause()
            self.get_logger().warning('[servo][center] joint headroom deg=%s',
                                      np.round(np.rad2deg(margin), 1).tolist())
        tcp_now, R_now = self.model.fk(q_now)
        self._imp_smoothed_tcp = np.asarray(tcp_now, dtype=np.float64).ravel()
        self._imp_smoothed_R = np.asarray(R_now, dtype=np.float64).reshape(3, 3)
        self._imp_follow_q = np.asarray(q_now, dtype=np.float64).copy()
        self._imp_follow_kp_active = np.zeros(7, dtype=np.float64)
        self._imp_target_update_time = float(now)
        self._imp_cfg = self._build_cfg(self._imp_smoothed_tcp, self._imp_smoothed_R, q_now)
        try:
            self.backend.set_cartesian_impedance(self._imp_cfg)
        except Exception as exc:
            self._diag_note(reference_write_failed=True)
            self._stop_task(f"impedance engage failed: {exc}")
            return
        self._imp_engaged = True
        self.get_logger().info("cartesian impedance ENGAGED at current TCP (200 Hz law)")

    def _build_cfg(self, target_tcp, R_target, q_rest):
        args = self.args
        coulomb = np.asarray(VALIDATED_COULOMB, dtype=np.float64)
        return ImpedanceConfig(
            active=True,
            target_tcp=np.asarray(target_tcp, dtype=np.float64),
            R_target=np.asarray(R_target, dtype=np.float64),
            max_position_error=float(args.impedance_max_position_error),
            max_orientation_error_rad=np.deg2rad(float(args.impedance_max_orientation_error_deg)),
            K=np.asarray(VALIDATED_CARTESIAN_KP, dtype=np.float64) * float(args.cart_k_scale),
            B=np.asarray(VALIDATED_CARTESIAN_KD, dtype=np.float64) * float(args.cart_b_scale),
            target_twist=np.zeros(6, dtype=np.float64),
            joint_damping=np.zeros(7, dtype=np.float64),
            # This clips task + gravity. The separate task_torque_limit below
            # clips the Cartesian task before gravity is added.
            tau_limit_per_joint=np.full(7, VALIDATED_TOTAL_TORQUE_LIMIT, dtype=np.float64),
            q_rest=np.asarray(q_rest, dtype=np.float64),
            q_follow=np.asarray(self._imp_follow_q, dtype=np.float64),
            follow_kp=self._imp_follow_kp_active.copy(),
            follow_max_error=float(getattr(args, "follow_max_error_rad", 0.03)),
            null_kq=0.0,
            dls_lambda=float(args.dls_lambda),
            use_dls_mapping=bool(args.use_dls),
            kp_joint=np.zeros(7, dtype=np.float64),
            kd_joint=np.array([1.0, 1.0, 1.0, 0.5, 0.3, 0.2, 0.2], dtype=np.float64),
            joint_hold_kp=np.zeros(7, dtype=np.float64),
            joint_hold_kd=np.zeros(7, dtype=np.float64),
            vel_filter_tau=float(args.vel_filter_tau),
            qdot_trip_fast=2.0,
            use_real_velocity=True,
            velocity_signs=np.asarray(VALIDATED_VELOCITY_SIGNS, dtype=np.float64),
            real_vel_filter_tau=0.0,
            joint_weights=np.asarray(VALIDATED_JOINT_WEIGHTS, dtype=np.float64),
            task_torque_limit=float(VALIDATED_TASK_TORQUE_LIMIT),
            coulomb=coulomb if np.any(coulomb > 0.0) else None,
            coulomb_smooth_width=0.0,
        )

    def _refresh(self, q_now, servo_out, now):
        cfg = self._imp_cfg
        if cfg is None:
            return
        nominal_dt = 1.0 / float(self.args.servo_hz)
        dt = min(nominal_dt, max(0.0, float(now) - float(self._imp_target_update_time)))
        self._imp_target_update_time = float(now)

        qdot = np.asarray(servo_out.get("qdot", np.zeros(7)), dtype=np.float64).ravel()
        self._imp_follow_q = np.clip(
            self._imp_follow_q + qdot * dt,
            NeroPinocchioModel.JOINT_LIMITS[:, 0], NeroPinocchioModel.JOINT_LIMITS[:, 1],
        )
        follow_bound = float(getattr(self.args, "follow_max_error_rad", 0.03))
        follow_error = self._imp_follow_q - q_now
        self._imp_follow_q = np.clip(
            q_now + np.clip(follow_error, -follow_bound, follow_bound),
            NeroPinocchioModel.JOINT_LIMITS[:, 0], NeroPinocchioModel.JOINT_LIMITS[:, 1],
        )
        if servo_out.get('center_control', False):
            bounds = self.servo.planning_bounds
            self._imp_follow_q = np.clip(self._imp_follow_q, bounds[:, 0], bounds[:, 1])
        min_qdot = float(self.args.follow_min_qdot)
        scale = (np.ones(7, dtype=np.float64) if min_qdot <= 1e-12
                 else np.clip(np.abs(qdot) / min_qdot, 0.0, 1.0))
        gain_limit = np.asarray(self.args.follow_kp, dtype=np.float64)
        desired_follow_kp = gain_limit * scale
        # 人被推超前时松固件弹簧，保留笛卡尔柔顺
        release_error = float(getattr(self.args, "follow_release_error", 0.02))
        if release_error > 0.0:
            follow_error = self._imp_follow_q - q_now
            pushed = ((np.abs(qdot) > 1e-6) & (qdot * follow_error < 0.0)
                      & (np.abs(follow_error) >= release_error))
            desired_follow_kp[pushed] = 0.0

        # Ramp only increases. Zero speed, push-ahead release and safety holds
        # must not wait for a gain-down ramp. Start from the installed config,
        # including any dropout decay, rather than a pre-loss gain cache.
        rise_time = float(getattr(self.args, "follow_kp_rise_time", 0.20))
        if rise_time > 0.0:
            previous_kp = getattr(cfg, "follow_kp", None)
            previous_kp = (np.zeros(7, dtype=np.float64) if previous_kp is None
                           else np.asarray(previous_kp, dtype=np.float64))
            desired_follow_kp = np.minimum(
                desired_follow_kp, previous_kp + gain_limit * dt / rise_time)
        self._imp_follow_kp_active = desired_follow_kp

        tcp_twist = np.asarray(servo_out.get("tcp_twist", np.zeros(6)), dtype=np.float64).ravel()
        tcp_now, R_now = self.model.fk(q_now)
        tcp_now = np.asarray(tcp_now, dtype=np.float64).ravel()
        previous_tcp = self._imp_smoothed_tcp.copy()
        previous_R = self._imp_smoothed_R.copy()
        candidate = integrate_position_target_from_twist(
            self._imp_smoothed_tcp, tcp_twist[:3], dt, float(self.args.impedance_target_linear_vel))
        self._imp_smoothed_tcp = apply_position_target_anti_windup(
            self._imp_smoothed_tcp, candidate, tcp_now,
            float(self.args.impedance_max_position_error))
        center_control = bool(servo_out.get('center_control', False))
        if center_control:
            self._imp_smoothed_R = advance_orientation_reference(
                previous_R, tcp_twist[3:], np.asarray(R_now), dt,
                float(self.args.impedance_target_angular_vel),
                float(cfg.max_orientation_error_rad))
        elif not bool(self.args.visual_orientation):
            target_tcp = np.asarray(servo_out.get("target_tcp", self._imp_smoothed_tcp), dtype=np.float64).ravel()
            # 未开视觉姿态：名义姿态保持在 engage 时的姿态
        else:
            # 视觉姿态：由 PBVS 输出目标旋转直接作为名义姿态（有 leash）
            target_rotation = np.asarray(servo_out.get("target_rotation", self._imp_smoothed_R), dtype=np.float64).reshape(3, 3)
            self._imp_smoothed_R = target_rotation

        cfg.target_tcp = self._imp_smoothed_tcp.copy()
        cfg.R_target = self._imp_smoothed_R
        cfg.q_follow = self._imp_follow_q.copy()
        cfg.follow_kp = self._imp_follow_kp_active.copy()
        cfg.target_twist = constrain_impedance_target_twist(
            tcp_twist, float(self.args.impedance_target_linear_vel),
            float(self.args.impedance_target_angular_vel),
            include_orientation=center_control or bool(self.args.visual_orientation))
        if center_control:
            # Damping follows the trajectory actually installed after both
            # leashes, rather than a velocity the reference could not execute.
            from scipy.spatial.transform import Rotation
            cfg.target_twist = np.zeros(6)
            if dt > 0:
                cfg.target_twist[:3] = (self._imp_smoothed_tcp - previous_tcp) / dt
                cfg.target_twist[3:] = Rotation.from_matrix(
                    self._imp_smoothed_R @ previous_R.T).as_rotvec() / dt
        self._imp_last_target_twist = cfg.target_twist.copy()
        self._imp_last_follow_kp = cfg.follow_kp.copy()
        try:
            self.backend.set_cartesian_impedance(cfg)
            self._diag_note(reference_updated=True)
        except Exception as exc:
            self._diag_note(reference_write_failed=True)
            self._stop_task(f"impedance refresh failed: {exc}")

    def _stop_task(self, reason):
        """R2: keep the sole sender alive until its position hold is confirmed."""
        self._diag_note(stop_reason=reason)
        try:
            self.prepare_shutdown()
        except Exception as exc:
            self._fail_closed(f"controlled stop unconfirmed: {reason}; {exc}")
            return
        try:
            self.get_logger().error(f"[R2][motion_stop] {reason}")
        except Exception:
            pass
        self._shutdown(exit_code=1)

    def _fail_closed(self, reason):
        self._diag_note(stop_reason=reason, fault=True)
        self._emergency_stop_requested = True
        try:
            # Deduplication belongs to the sole control owner, not a diagnostic
            # read that could block behind the failed control thread.
            self.backend.emergency_stop()
        except Exception as exc:
            reason += f"; electronic stop request failed, stop state unknown: {exc}"
        try:
            self.get_logger().error(f"[FAIL-CLOSED] {reason}")
        except Exception:
            pass
        self._shutdown(exit_code=1)

    def _shutdown(self, exit_code=0):
        sdk_loop.shutdown()
        raise SystemExit(exit_code)

    def prepare_shutdown(self):
        if self._shutdown_prepared or self._emergency_stop_requested:
            return
        prepare = getattr(self.backend, "prepare_shutdown", None)
        if callable(prepare):
            prepare()
        else:
            terminate = getattr(self.backend, "terminate_motion", None)
            if callable(terminate):
                terminate()
            elif not isinstance(self.backend, FakeBackend):
                raise RuntimeError("Backend has no confirmed controlled-stop capability")
        self._shutdown_prepared = True

    # ---------- 主循环 ----------
    def _gesture_arm_still(self):
        history = self._gesture_joint_history
        if len(history) < 5 or history[-1][0]-history[0][0] < .25:
            return False
        return np.max(np.ptp([q for _, q in history], axis=0)) <= np.deg2rad(.3)

    def _try_memory_approach(self, q_now, now):
        memory = getattr(self, '_hand_memory', None)
        gesture = getattr(self, '_gesture', None)
        if memory is None or gesture is None:
            return False
        if gesture.phase != 'FOLLOW_RECEIVER' or gesture.release_started:
            memory.invalidate('memory_disabled_after_readiness_or_release')
            return False
        mono = time.monotonic()
        observation = getattr(self.target, 'last_observation', None)
        roles = getattr(self.target, 'hand_roles', None)
        if (self._detection_reason != 'invalid_palm_depth' or observation is None
                or roles is None or roles.presence != 'visible'
                or observation.handedness != roles.receiver
                or mono-getattr(self, '_sdk_received_monotonic', 0.) > .25):
            memory.invalidate('memory_requires_fresh_visible_receiver_with_depth_only_loss')
            return False
        if not memory.available(mono):
            return False
        tcp, R = self.model.fk(q_now)
        tcp, R = np.asarray(tcp).copy(), np.asarray(R).copy()
        hand = memory.target[0]
        distance = float(np.linalg.norm(hand-tcp))
        if distance <= self.args.delivery_distance+.01:
            memory.invalidate('memory_already_at_follow_distance')
            return False
        T = self.target.tcp_camera_transform
        R_camera = R@T[:3, :3]
        camera_point = R_camera.T@(hand-tcp-R@T[:3, 3])
        K = np.asarray(self._latest_info.k).reshape(3, 3)
        if not memory.consistent_image(camera_point, observation.palm_pixel, K):
            return False
        if not memory.active:
            # A new current-pose impedance hold removes pre-loss reference lead
            # before granting a separate 3mm budget. No motion-mode/CAN takeover.
            self._engage(q_now, now)
            memory.start(tcp, self._imp_smoothed_tcp)
        remaining = memory.remaining(tcp)
        if memory.blocked:
            return False
        dt = min(1./self.args.servo_hz, max(0., now-self._imp_target_update_time))
        old_speed = self.servo.params.max_linear_vel
        old_angular = self.servo.params.max_angular_vel
        self.servo.params.max_linear_vel = min(old_speed, memory.limits.maximum_speed_m_s,
                                               remaining/max(dt, 1e-6))
        self.servo.params.max_angular_vel = min(old_angular, .05)
        goal = hand+(tcp-hand)/distance*self.args.delivery_distance
        try:
            out = self.servo.fixed_step(q_now, goal, self._imp_smoothed_R.copy(), now)
        finally:
            self.servo.params.max_linear_vel = old_speed
            self.servo.params.max_angular_vel = old_angular
        decision = self.safety.evaluate(q_now, out, now)
        if decision.fatal:
            self._fail_closed(f'memory approach safety: {decision.reason}')
            return True
        if out.get('control_hold') or decision.hold:
            memory.invalidate(out.get('control_hold') or decision.status)
            return False
        self._refresh(q_now, out, now)
        memory.reference_sent(self._imp_smoothed_tcp)
        memory.reason = 'bounded_depth_only_memory_approach'
        self._preview_status = 'MEMORY_APPROACH'
        self._preview_reason = 'last measured BASE position; not a new depth measurement'
        self._preview_base = (hand*1000).tolist()
        self._preview_error_mm = '--'
        self._diag_note(servo_out=out, hand_memory=memory.snapshot())
        gesture.invalidate_near('Memory approach cannot establish receiver stillness')
        self._gesture_report()
        return True

    def _gesture_report(self, **values):
        gesture = self._gesture
        roles = getattr(self.target, 'hand_roles', None)
        snapshot = dict(gesture.snapshot(), receiver_side=getattr(roles, 'receiver', None),
                        control_side=getattr(roles, 'control', None), **values)
        self._gesture_snapshot = snapshot
        self._diag_note(gesture_handover=snapshot)
        previous = getattr(self, '_gesture_last_phase', None)
        if previous != gesture.phase:
            self._gesture_last_phase = gesture.phase
            self.get_logger().warning('[handover] %s receiver=%s control=%s: %s',
                gesture.phase, snapshot['receiver_side'], snapshot['control_side'], gesture.reason)
            instructions = {
                'FOLLOW_RECEIVER': '先只伸出接杯手，锁定左右身份后开始跟随。',
                'GESTURE_LEARN': '接杯手已在准备距离内停稳，机械臂保持；松夹前A手移动会重新跟随。B手先捏合约1秒、再充分张开约1秒，学习期间不松杯。',
                'GESTURE_ARM': '请控制手再次捏合并停约0.4秒，确认开始比例开合，无需按键。',
                'GESTURE_READY': '已确认但未松夹，夹持指令保持不变；B手主动张开才松夹，A手移动优先恢复跟随并作废旧请求。',
                'GESTURE_CONTROL': '已开始松夹，本轮机械臂不再跟随；0%为原夹位附近（最多再紧1mm），100%为允许全开。',
                'GESTURE_PAUSED': '控制手丢失或不明确，停发夹爪目标，不自动回夹；同一只手连续识别3帧后从当前开口继续。',
                'HANDOVER_DONE': '已确认全开，本次松夹结束；机械臂原地保持。',
                'HANDOVER_CANCELLED': '松夹期间A手持续移走，停止新开合目标并保持原地、当前开口，不自动回夹。',
                'GESTURE_FAULT': '夹爪控制已暂停并锁定故障；机械臂继续保持，详细原因见上行。',
            }
            self.get_logger().warning('[handover] %s', instructions.get(gesture.phase, gesture.reason))
        if gesture.phase != 'FOLLOW_RECEIVER':
            self.state = gesture.phase
            self._preview_status, self._preview_reason = gesture.phase, gesture.reason

    def _activate_gesture_hold(self, q_now, now):
        # Freeze the ACTUAL pose once. Do not leave a moving reference ahead of
        # the arm, and never run the standalone J/CAN takeover inside MIT.
        self._pause_target_motion(now, immediate=True)
        self._engage(q_now, now)
        self._gesture_joint_history.clear()
        self._gesture_report()

    def _gesture_hold_step(self, q_now, now):
        from handoff.gesture_handover import read_opening
        from handoff.pinch_gripper import measure_pinch
        gesture = self._gesture
        if gesture.terminal:
            self._gesture_report()
            return  # No re-close, retreat, target tracking, or further gripper writes.
        previous_sequence = self._consumed_sequence
        try:
            detected = self._detect_latest(q_now, now)
            fresh = (detected is not _NO_NEW_FRAME and self._consumed_sequence != previous_sequence
                     and time.monotonic()-getattr(self, '_sdk_received_monotonic', 0.) <= .25)
        except Exception as exc:
            detected, fresh = None, False
            self._detection_reason = f'detector_error:{exc}'
        # Read A BEFORE B on the same frame. Before the first opening, receiver
        # motion revokes authorization even if B is simultaneously opening.
        if fresh and detected is not None and detected is not _NO_NEW_FRAME:
            point = np.asarray(detected[0], dtype=float)
            distance = float(np.linalg.norm(point-np.asarray(self.model.fk(q_now)[0])))
            try:
                pixel = self.servo.feature_system(q_now, point)[4]
                centered = np.linalg.norm(pixel-self.servo._image_center) < 12.
            except (AttributeError, ValueError, np.linalg.LinAlgError):
                centered = False
            gesture.observe_receiver('visible', distance, bool(centered), time.monotonic(),
                                     position_base=point)
        elif fresh and not gesture.release_started and gesture.receiver_anchor is not None:
            observation = getattr(self.target, 'last_observation', None)
            roles = getattr(self.target, 'hand_roles', None)
            if (observation is not None and roles is not None and roles.presence == 'visible'
                    and observation.handedness == roles.receiver):
                # A held camera can still see a clear image displacement even
                # when depth is invalid. Revoke permission rather than release
                # against an old BASE position; this is not a new 3D estimate.
                try:
                    tcp, R = self.model.fk(q_now)
                    T = self.target.tcp_camera_transform
                    camera_point = (R@T[:3, :3]).T@(gesture.receiver_anchor-tcp-R@T[:3, 3])
                    check = HandPositionMemory()
                    K = np.asarray(self._latest_info.k).reshape(3, 3)
                    if not check.consistent_image(camera_point, observation.palm_pixel, K):
                        gesture.invalidate_near('Receiver image moved before release; wait for fresh 3D and re-confirm')
                except (AttributeError, TypeError, ValueError):
                    gesture.invalidate_near('Receiver projection unavailable before release')
        if gesture.phase == 'FOLLOW_RECEIVER':
            self.state = 'TRACK_HAND'
            self._gesture_report()
            return detected  # Reuse this frame in follow; do not detect twice.
        if gesture.terminal:
            self._gesture_report()
            return
        if isinstance(getattr(self, 'safety', None), ExecutionSafetyMonitor):
            guard = ExecutionSafetyMonitor.evaluate(self.safety, q_now, {'error': np.zeros(3)}, now)
            if guard.fatal:
                self._fail_closed(f'servo safety: {guard.reason}')
                return
            if guard.hold:
                gesture.fault(f'arm safety hold during handover: {guard.status}')
                self._gesture_report()
                return
        pinch = None
        if fresh:
            roles = self.target.hand_roles
            try:
                pinch = measure_pinch(roles.control_observations,
                    (self._latest_image.height, self._latest_image.width), roles.control)
            except ValueError:
                pass  # A fresh invalid/ambiguous control hand pauses gripper targets.
        try:
            if isinstance(self.backend, FakeBackend):
                measured, feedback_force, stamp = self._gesture_sim_width, 0., time.time()
            else:
                measured, feedback_force, stamp = read_opening(self.gripper.status())
            command = gesture.tick(pinch, fresh_frame=fresh, measured_width=measured,
                feedback_stamp=stamp, now=time.monotonic(), arm_still=self._gesture_arm_still())
            if command is not None:
                if isinstance(self.backend, FakeBackend):
                    self._gesture_sim_width = command
                else:
                    # Keep MIT as the sole arm sender. Only the existing gripper
                    # handle is used; no second CAN driver and no mode writes.
                    robot = self.backend.get_robot()
                    state = robot.get_arm_status()
                    if (state is None or not 0. <= time.time()-state.timestamp <= .5
                            or int(state.msg.ctrl_mode) != 1 or int(state.msg.arm_status) != 0
                            or int(state.msg.err_code) != 0 or int(state.msg.mode_feedback) not in (4, 6)):
                        raise ValueError('MIT/CAN arm status not healthy; no gesture gripper write')
                    latest, _, _ = read_opening(self.gripper.status())
                    if abs(command-latest) > gesture.ramp.max_lead+1e-9:
                        raise ValueError('gripper target leads latest feedback too far')
                    if not gesture.minimum <= command <= gesture.maximum:
                        raise ValueError('gripper target violates fixed cup closure limit')
                    self.gripper.open(width=command, force=self._gesture_force)
                gesture.command_sent(command, time.monotonic())
            self._gesture_report(actual_width_mm=measured*1000., force_feedback=feedback_force,
                                 simulated=isinstance(self.backend, FakeBackend))
        except (ValueError, RuntimeError, AttributeError, TypeError) as exc:
            gesture.fault(exc)
            self._gesture_report()

    def _on_timer(self):
        preview = getattr(self, "_preview", None)
        if preview is not None:
            event = preview.poll()
            if event == "quit":
                self._shutdown(0)
            if event is not None:
                raise RuntimeError(f"Servo preview unavailable: {event}; exiting servo")
        try:
            self._servo_step()
        finally:
            self._record_fusion()
            try:
                self._publish_preview()
            finally:
                self._publish_diagnostics()

    def _publish_diagnostics(self):
        reporter = getattr(self, "_diagnostics", None)
        sample = getattr(self, "_diag_sample", None)
        if reporter is None or sample is None:
            return
        try:
            debug = sample.get("debug") or {}
            status = getattr(self, "_preview_status", "UNKNOWN")
            reason = getattr(self, "_preview_reason", "unknown")
            if sample.get("fault") or debug.get("last_error"):
                status, reason = "FAULT", sample.get("stop_reason") or debug.get("last_error")
            elif sample.get("stop_reason") or debug.get("controlled_stop_reason"):
                status, reason = "STOPPING", sample.get("stop_reason") or debug.get("controlled_stop_reason")
            reporter.observe(
                sample, model=self.model, cfg=getattr(self, "_imp_cfg", None),
                status=status, reason=reason, stage=self.state,
                frame_sequence=getattr(self, "_sdk_frame_sequence", None),
                frame_received_monotonic=getattr(self, "_sdk_received_monotonic", None),
                target_time=getattr(self, "_last_target_time", None),
            )
        except Exception:
            # Observation failure must not change control, mask its exception,
            # or become a new reason to disable/stop the arm.
            pass

    def _diag_note(self, **values):
        sample = getattr(self, "_diag_sample", None)
        if sample is not None:
            sample.update(values)

    def _publish_preview(self):
        preview = getattr(self, "_preview", None)
        if preview is None:
            return
        now = time.monotonic()
        if now - self._preview_last_sent < 1.0 / 30.0 or self._latest_image is None:
            return
        self._preview_last_sent = now
        image = self._latest_image
        observation = None
        if self._consumed_sequence == getattr(self, "_sdk_frame_sequence", None):
            observation = getattr(self.target, "last_observation", None)
        camera_xyz = None if observation is None else observation.palm_center_camera_m
        preview.publish({
            "width": image.width, "height": image.height,
            "encoding": image.encoding, "data": image.data,
            "received_monotonic": getattr(self, "_sdk_received_monotonic", now),
            "updated_monotonic": now, "stage": self.state,
            "status": self._preview_status, "reason": self._preview_reason,
            "landmarks": None if observation is None else observation.landmarks_pixel.tolist(),
            "palm_pixel": None if observation is None else observation.palm_pixel,
            "camera_xyz_mm": None if camera_xyz is None else (np.asarray(camera_xyz) * 1000).tolist(),
            "depth_points": 0 if observation is None else observation.valid_depth_landmarks,
            "required_depth_points": getattr(self.args, "hand_min_depth_landmarks", 3),
            "base_xyz_mm": self._preview_base,
            "distance_m": float(self.servo.params.desired_distance),
            "error_mm": self._preview_error_mm,
            "camera_center": (getattr(self, '_center_feedback', None)
                              if self._preview_status == 'TRACKING' else None),
            "gesture_handover": getattr(self, '_gesture_snapshot', None),
            "fusion_handover": (self._fusion.snapshot() if getattr(self, '_fusion', None) is not None else None),
            "hand_memory": (self._hand_memory.snapshot()
                            if getattr(self, '_hand_memory', None) is not None else None),
            "control_landmarks": (self.target.hand_roles.control_observations[0].landmarks_pixel.tolist()
                if (getattr(self.target, 'hand_roles', None) is not None
                    and len(self.target.hand_roles.control_observations) == 1
                    and self._consumed_sequence == getattr(self, '_sdk_frame_sequence', None)) else None),
        })

    def _servo_step(self):
        self._diag_sample = None
        now = time.time()
        q_now = self.backend.get_joint_angles()
        if q_now is None or len(q_now) != 7 or not np.isfinite(q_now).all():
            self._fail_closed('Missing or invalid joint feedback')
            return
        q_now = np.asarray(q_now, dtype=np.float64).ravel()
        if getattr(self, '_gesture', None) is not None:
            mono = time.monotonic()
            self._gesture_joint_history.append((mono, q_now.copy()))
            while self._gesture_joint_history and mono-self._gesture_joint_history[0][0] > .6:
                self._gesture_joint_history.popleft()
        if getattr(self, "_diagnostics", None) is not None:
            self._diag_sample = dict(q=q_now.copy(), dq=None, debug=None,
                                     servo_out=None, detection="not_run", reference_updated=False,
                                     sampled_monotonic=time.monotonic())
        if not isinstance(self.backend, FakeBackend):
            feedback = self.backend.get_robot().get_joint_angles()
            stamp = float(getattr(feedback, 'timestamp', 0.0))
            if not 0 <= time.time() - stamp <= .5:
                self._fail_closed('Stale SDK joint feedback')
                return

        debug = self.backend.get_debug_state() or {}
        self._diag_note(debug=debug)
        if getattr(self, '_diagnostics', None) is not None:
            diagnostic_now = time.monotonic()
            if (not isinstance(self.backend, FakeBackend)
                    and diagnostic_now >= getattr(self, '_device_diagnostic_next', 0.0)):
                self._device_diagnostic_next = diagnostic_now + 1.0
                try:
                    from control_backends.force_ctrl.device_feedback import read_device_feedback
                    self._device_diagnostic = read_device_feedback(self.backend.get_robot())
                except Exception as exc:
                    self._device_diagnostic = {'read_errors': [f'{type(exc).__name__}: {exc}']}
            self._diag_note(device_feedback=getattr(self, '_device_diagnostic', None))
        if debug.get("last_error"):
            self._fail_closed(f"inner MIT loop fault: {debug.get('last_error')}")
            return
        if debug.get("controlled_stop_reason"):
            self._stop_task(debug["controlled_stop_reason"])
            return
        dq = self.backend.get_joint_velocity()
        self._diag_note(dq=dq)
        if dq is not None:
            self._qdot_window.append(np.linalg.norm(dq))
            if len(self._qdot_window) >= 10 and float(np.mean(self._qdot_window)) > 0.3:
                self._fail_closed("impedance instability: RMS qdot > 0.3 rad/s")
                return

        if self._poll_fusion(q_now, now) is False:
            self._fusion_report()
            return  # Await initial MIT feedback or retain a latched fusion fault.
        if self._fusion_manual_stop(q_now, now):
            return  # Explicit shadow parking freezes actual pose; no gripper writes.
        if getattr(self, '_fusion', None) is not None and self._fusion.phase != 'FOLLOW_RECEIVER':
            self._fusion_hold_step(q_now, now)
            return  # Fusion owns the hold/opening path; never enter legacy RELEASE/RETREAT.

        # if self.state == "HOLD_LIFTED":
        #     if not self._imp_engaged:
        #         self._engage(q_now, now)
        #     else:
        #         self.state, self.state_entered = "TRACK_HAND", now
        #     return
        if self.state == "HOLD_LIFTED":
            self._preview_status, self._preview_reason = "HOLD", "waiting_for_joint_settling"
            if not self._imp_engaged:
                if self._hold_q_ref is None:
                    self._hold_q_ref = q_now.copy()
                    self._hold_stable = 0
                    return
                delta = float(np.max(np.abs(np.rad2deg(q_now - self._hold_q_ref))))
                if delta <= 0.5:
                    self._hold_stable += 1
                else:
                    self._hold_q_ref = q_now.copy()
                    self._hold_stable = 0
                if self._hold_stable < 3:
                    return
                self._engage(q_now, now)
            else:
                self.state, self.state_entered = "TRACK_HAND", now
            return

        handover_target = None
        if (getattr(self, '_gesture', None) is not None
                and self._gesture.phase != 'FOLLOW_RECEIVER'):
            handover_target = self._gesture_hold_step(q_now, now)
            if self._gesture.phase != 'FOLLOW_RECEIVER' or handover_target is None:
                return

        if self.state == "RELEASE":
            self._preview_status, self._preview_reason = "RELEASE", "waiting_for_gripper_feedback"
            if now - self.state_entered > 5.0:
                self._stop_task('Gripper release feedback timeout')
                return
            opened = self.gripper is None
            if self.gripper is not None:
                status = self.gripper.status()
                msg = getattr(status, 'msg', None)
                width = float(getattr(msg, 'value', float('nan')))
                opened = (getattr(msg, 'mode', '') == 'width' and np.isfinite(width) and width >= .080)
            self._release_samples = self._release_samples + 1 if opened else 0
            if self._release_samples >= 3:
                self.state, self.state_entered = 'RETREAT', now
                self.retreat_target = None
            return

        if self.state == "RETREAT":
            self._preview_status, self._preview_reason = "RETREAT", "fixed_retreat_target"
            if self.retreat_target is None:
                tcp_now, _ = self.model.fk(q_now)
                self.retreat_target = np.asarray(tcp_now, dtype=np.float64).ravel() + np.array([0.0, 0.0, 0.15])
            if np.linalg.norm(self.retreat_target - np.asarray(self.model.fk(q_now)[0])) < 0.01:
                self._shutdown(0)
            self._fixed_step(q_now, self.retreat_target, now)
            return

        # target = self._detect_latest(q_now, now)
        try:
            target = (handover_target if handover_target is not None
                      else self._detect_latest(q_now, now))
        except Exception as exc:
            self.get_logger().warning(
                "target detect failed; holding last impedance target: "
                f"{type(exc).__name__}: {exc}"
            )
            self._detection_reason = f"detector_error:{type(exc).__name__}:{exc}"
            target = None
        if target is _NO_NEW_FRAME:
            self._diag_note(detection="duplicate")
            memory = getattr(self, '_hand_memory', None)
            if memory is not None and memory.active:
                # Enforce the loss budget on timer ticks too. Waiting for a
                # new image must not extend a 0.3s memory command to the camera
                # stale timeout while the inner sender keeps its last twist.
                if memory.available(time.monotonic()):
                    memory.remaining(np.asarray(self.model.fk(q_now)[0]))
                if memory.blocked:
                    self._pause_target_motion(now, immediate=True)
                    self._preview_status, self._preview_reason = 'LOST', memory.reason
                    self._diag_note(hand_memory=memory.snapshot())
            return
        if target is None:
            if getattr(self, '_fusion', None) is not None:
                self._fusion.delivery_lost()
            self._rejected_target = None
            self._diag_note(detection="lost")
            if self._try_memory_approach(q_now, now):
                return
            if getattr(self, '_hand_memory', None) is not None:
                self._diag_note(hand_memory=self._hand_memory.snapshot())
            self._preview_status, self._preview_reason = "LOST", self._detection_reason
            self._preview_base, self._preview_error_mm = None, "--"
            self._pause_target_motion(now, immediate=(getattr(self, '_gesture', None) is not None
                                                      or getattr(self, '_fusion', None) is not None))
            if getattr(self, '_gesture', None) is not None:
                presence = ('absent' if self._detection_reason == 'receiver_absent_2d' else 'unknown')
                self._gesture.observe_receiver(presence, None, False, time.monotonic())
                if self._gesture.phase != 'FOLLOW_RECEIVER':
                    self._activate_gesture_hold(q_now, now)
                    return
                self._gesture_report()
            if now - self._last_target_time > _TARGET_TIMEOUT_S:
                self._forget_lost_target(q_now, now)
            else:
                self._reacquire_count = 0
                self._reacquire_target = None
            if now - self._last_lost_log >= 1.0:
                self._last_lost_log = now
                detail = getattr(self.target, "last_diagnostics", {})
                self.get_logger().warning(
                    "[servo][target] lost reason=%s details=%s; holding last impedance target",
                    self._detection_reason, detail,
                )
            return
        self._diag_note(detection="valid")
        # A long gap may end with a valid frame before a lost timer ever ran.
        if now - self._last_target_time > _TARGET_TIMEOUT_S:
            self._forget_lost_target(q_now, now)
        self._last_target_time = now
        self._preview_base = (np.asarray(target[0]) * 1000).tolist()
        if self._target_lost and not self._accept_reacquired_target(target):
            if getattr(self, '_fusion', None) is not None:
                self._fusion.delivery_lost()
            if getattr(self, '_gesture', None) is not None:
                self._gesture.invalidate_near()
                self._gesture_report()
            self._preview_status = "ACQUIRING"
            self._preview_reason = f"consistent_new_frames:{self._reacquire_count}/3"
            self._preview_error_mm = "--"
            self._pause_target_motion(now)
            return
        center, normal, tangent = target
        servo_out = self.servo.update(q_now, center, normal, tangent, None, now)
        if servo_out.get('target_rejected', False):
            recovered = self._try_reacquire_rejected_target(q_now, target, now)
            if recovered is not None:
                servo_out = recovered
        else:
            self._rejected_target = None
        self._diag_note(servo_out=servo_out)
        self._center_feedback = servo_out.get('camera_center')
        if servo_out.get('control_hold'):
            if getattr(self, '_fusion', None) is not None:
                self._fusion.delivery_lost()
            if getattr(self, '_hand_memory', None) is not None:
                self._hand_memory.invalidate('follow_control_hold')
            if getattr(self, '_gesture', None) is not None:
                self._gesture.invalidate_near()
                self._gesture_report()
            self._preview_status, self._preview_reason = 'HOLD', servo_out['control_hold']
            self._preview_error_mm = '--'
            self._pause_target_motion(now, immediate=True)
            return
        if isinstance(self.safety, HandFollowSafetyMonitor):
            decision = self.safety.evaluate(q_now, servo_out, now,
                reference_tcp=self._imp_smoothed_tcp, reference_q=self._imp_follow_q)
        else:
            decision = self.safety.evaluate(q_now, servo_out, now)
        if decision.fatal:
            self._fail_closed(f"servo safety: {decision.reason}")
            return
        if decision.hold:
            if getattr(self, '_fusion', None) is not None:
                self._fusion.delivery_lost()
            if getattr(self, '_hand_memory', None) is not None:
                self._hand_memory.invalidate('follow_safety_hold')
            if getattr(self, '_gesture', None) is not None:
                self._gesture.invalidate_near()
                self._gesture_report()
            self._preview_status = "HOLD"
            self._preview_reason = f"{decision.status}: {decision.reason}"
            self._preview_error_mm = "--"
            # Safety holds (jump rejection, leash violation, etc.) must take
            # effect immediately; only a camera dropout gets the soft decay.
            self._pause_target_motion(now, immediate=True)
            if now - self._last_lost_log >= 1.0:
                self._last_lost_log = now
                self.get_logger().warning("[servo][hold] %s: %s", decision.status, decision.reason)
            return
        err_pos = float(np.linalg.norm(servo_out.get("error", np.zeros(3))))
        err_ori = float(np.linalg.norm(servo_out.get("orientation_error", np.zeros(3))))
        ok_orientation = (
            not bool(self.args.visual_orientation)
            or err_ori < np.deg2rad(float(self.args.approach_error_deg))
        )
        if bool(getattr(self.args, 'hand_camera_center', False)):
            detail = servo_out.get('camera_center', {})
            ok_orientation = np.linalg.norm(detail.get('pixel_error', [float('inf')])) < 12.
        ok_err = err_pos < float(self.args.approach_error_m) and ok_orientation
        self.stable_frames = self.stable_frames + 1 if ok_err else 0
        if getattr(self, '_fusion', None) is not None:
            distance = float(np.linalg.norm(center-np.asarray(self.model.fk(q_now)[0])))
            preparation_region = bool(ok_orientation)
            if bool(getattr(self.args, 'hand_camera_center', False)):
                # Parking is before contact/occlusion, not another exact-centre
                # convergence requirement. The motion servo itself is unchanged.
                pixels = np.asarray(servo_out.get('camera_center', {}).get('pixel_error', []), dtype=float)
                image = getattr(self, '_latest_image', None)
                span = min(getattr(image, 'width', 0), getattr(image, 'height', 0))
                preparation_region = (pixels.shape == (2,) and np.isfinite(pixels).all() and span > 0
                    and np.linalg.norm(pixels) <= self._fusion.config.ready_image_radius_fraction*span)
            if self._fusion.observe_delivery(center, distance, bool(preparation_region), time.monotonic()):
                self._fusion_begin_hold(q_now, now)
                return  # Freeze before another moving reference is installed.
        if getattr(self, '_gesture', None) is not None:
            distance = float(np.linalg.norm(center-np.asarray(self.model.fk(q_now)[0])))
            if getattr(self, '_hand_memory', None) is not None:
                self._hand_memory.observe(target, getattr(self, '_sdk_received_monotonic', time.monotonic()))
                self._diag_note(hand_memory=self._hand_memory.snapshot())
            # Readiness range is independent of the 5cm destination. Only
            # image centering and measured BASE stillness qualify at <=15cm.
            self._gesture.observe_receiver('visible', distance, bool(ok_orientation), time.monotonic(),
                                           position_base=center)
            if self._gesture.phase != 'FOLLOW_RECEIVER':
                self._activate_gesture_hold(q_now, now)
                return  # Freeze before sending another moving follow reference.

        self._refresh(q_now, servo_out, now)
        self._preview_status, self._preview_reason = "TRACKING", "target_accepted"
        self._preview_error_mm = f"{np.linalg.norm(servo_out.get('error', np.zeros(3))) * 1000:.1f}"
        if now - self._last_tracking_log >= 1.0:
            self._last_tracking_log = now
            self.get_logger().warning(
                "[servo][tracking] palm_base_mm=%s error_mm=%.1f",
                np.round(np.asarray(center) * 1000, 1).tolist(),
                float(np.linalg.norm(servo_out.get("error", np.zeros(3)))) * 1000,
            )

        if getattr(self, '_gesture', None) is not None:
            self._gesture_report(tcp_to_palm_m=distance)
            return  # Bypass the old automatic release/retreat state machine.

        if getattr(self, '_fusion', None) is not None:
            self._fusion_report()
            return  # No old distance, gesture, open or retreat path in fusion mode.

        if bool(getattr(self.args, "servo_follow_only", False)):
            return

        if self.state == "TRACK_HAND" and self.stable_frames >= int(self.args.stable_frames):
            self.state, self.state_entered, self.stable_frames = "APPROACH", now, 0
            self.servo.params.desired_distance = float(self.args.handover_distance)
            self.get_logger().info("hand target acquired; approaching")
        elif self.state == "APPROACH" and ok_err and np.linalg.norm(center - np.asarray(self.model.fk(q_now)[0])) <= float(self.args.handover_distance) + .01:
            self.state, self.state_entered, self.stable_frames = "HANDOVER", now, 0
            self.get_logger().info("at handover distance; waiting for stable hand")
        elif self.state == "HANDOVER" and self.stable_frames >= int(self.args.handover_stable_frames) and np.linalg.norm(center - np.asarray(self.model.fk(q_now)[0])) <= float(self.args.handover_distance) + .01:
            self.state, self.state_entered = "RELEASE", now
            self.get_logger().info("hand stable; releasing gripper")
            if self.gripper is not None:
                self.gripper.open()
        elif self.state == "RELEASE":
            if self.gripper is not None:
                self.gripper.status()  # 反馈校验（v1 先以命令成功为准）
            self.state, self.state_entered = "RETREAT", now
            self.retreat_target = None

    def _fixed_step(self, q_now, target_tcp, now):
        """RETREAT：沿固定目标步进（复用 PBVS，目标固定）。"""
        if bool(getattr(self.args, 'hand_camera_center', False)):
            servo_out = self.servo.fixed_step(q_now, target_tcp, self._imp_smoothed_R, now)
            if servo_out.get('control_hold'):
                self._pause_target_motion(now, immediate=True)
                return
            decision = self.safety.evaluate(q_now, servo_out, now)
            if decision.fatal:
                self._fail_closed(decision.reason)
            elif decision.hold:
                self._pause_target_motion(now, immediate=True)
            else:
                self._refresh(q_now, servo_out, now)
            return
        distance = self.servo.params.desired_distance
        try:
            self.servo.params.desired_distance = 0.0
            servo_out = self.servo.update(q_now, target_tcp, None, None, None, now)
        finally:
            self.servo.params.desired_distance = distance
        decision = self.safety.evaluate(q_now, servo_out, now)
        if decision.fatal:
            self._fail_closed(decision.reason)
        elif not decision.hold:
            self._refresh(q_now, servo_out, now)


def parse_servo_args(argv):
    from real_nero_grasp.servo_cli import servo_parser
    from cameras.options import add_camera_arguments
    p = argparse.ArgumentParser(parents=[servo_parser()])
    add_camera_arguments(p)
    from cameras.options import apply_pipeline_defaults
    apply_pipeline_defaults(p, argv)
    return p.parse_args(argv)


def validate_servo_args(args):
    from pathlib import Path
    from real_nero_grasp.servo_cli import parse_follow_kp, nonnegative_finite_float, filter_alpha
    from handoff.fusion_cli import validate_fusion_args
    validate_fusion_args(args)
    if not getattr(args, 'handeye', None) or not Path(args.handeye).is_file():
        raise ValueError('Provide --handeye for the current wrist camera and its mounting position')
    args.follow_kp = parse_follow_kp(args.follow_kp)
    args.follow_min_qdot = nonnegative_finite_float(args.follow_min_qdot)
    args.follow_kp_rise_time = nonnegative_finite_float(getattr(args, "follow_kp_rise_time", 0.20))
    args.follow_max_error_rad = nonnegative_finite_float(getattr(args, "follow_max_error_rad", 0.03))
    args.max_joint_vel = nonnegative_finite_float(args.max_joint_vel)
    args.follow_release_error = nonnegative_finite_float(getattr(args, "follow_release_error", 0.02))
    if bool(getattr(args, 'hand_camera_center', False)):
        if args.target_source != 'mediapipe':
            raise ValueError('--hand-camera-center requires --target-source mediapipe')
        if args.visual_orientation:
            raise ValueError('Use --hand-camera-center without --visual-orientation')
        for key in ('hand_center_gain', 'hand_center_max_joint_accel', 'delivery_distance', 'max_linear_vel',
                    'max_angular_vel', 'impedance_target_linear_vel', 'impedance_target_angular_vel'):
            value = nonnegative_finite_float(getattr(args, key))
            if value <= 0:
                raise ValueError(f'{key} must be positive for camera centering')
        if np.any(np.fromstring(args.hand_offset_base, sep=',') != 0):
            raise ValueError('--hand-camera-center requires zero --hand-offset-base for true image centering')
    for name, default in (("target_filter_alpha", .34), ("normal_filter_alpha", .24),
                          ("target_blend_alpha", .82), ("twist_smooth_alpha", .72)):
        setattr(args, name, filter_alpha(getattr(args, name, default)))
    if args.servo_hz <= 0:
        raise ValueError('servo-hz must be positive')
    if bool(getattr(args, 'gesture_handover', False)):
        if args.target_source != 'mediapipe' or args.servo_follow_only:
            raise ValueError('--gesture-handover requires mediapipe and cannot be combined with --servo-follow-only')
        if not np.isfinite(args.delivery_distance) or abs(args.delivery_distance-.05) > 1e-9:
            raise ValueError('--gesture-handover requires --delivery-distance 0.05 (TCP to palm)')
        if not bool(getattr(args, 'hand_camera_center', False)):
            raise ValueError('--gesture-handover requires --hand-camera-center for receiver centering')
        if (not 0. < args.gesture_near_tolerance <= .02
                or not 0. <= args.gesture_extra_close_mm <= 1.
                or min(args.gesture_open_seconds, args.gesture_close_seconds) <= 0.
                or min(args.gesture_stable_seconds, args.gesture_still_tolerance_mm) <= 0.
                or args.gesture_ready_distance < args.delivery_distance
                or not np.isfinite([args.gesture_near_tolerance, args.gesture_extra_close_mm,
                                   args.gesture_open_seconds, args.gesture_close_seconds,
                                   args.gesture_stable_seconds, args.gesture_still_tolerance_mm,
                                   args.gesture_ready_distance]).all()):
            raise ValueError('invalid gesture timing/tolerance or closure margin (maximum 1mm)')
        simulated = args.gesture_preview_grip_width_mm
        if not bool(getattr(args, 'execute', False)) and (
                simulated is None or not np.isfinite(simulated) or not 0. < simulated < 85.):
            raise ValueError('dry-run gesture mode requires --gesture-preview-grip-width-mm in (0,85)')
        if bool(getattr(args, 'execute', False)) and simulated is not None:
            raise ValueError('--gesture-preview-grip-width-mm is dry-run only; hardware uses fresh measured width')
    if args.target_source == 'mediapipe':
        if not Path(args.hand_model).expanduser().is_file():
            raise ValueError(
                'Provide --hand-model pointing to hand_landmarker.task'
            )
        confidences = (
            args.hand_min_detection_confidence,
            args.hand_min_presence_confidence,
            args.hand_min_tracking_confidence,
        )
        if not all(0.0 <= value <= 1.0 for value in confidences):
            raise ValueError('hand confidence thresholds must be in [0, 1]')
        if (args.hand_depth_radius < 0
                or not 0.0 < args.hand_min_depth < args.hand_max_depth):
            raise ValueError('invalid hand depth parameters')
        if not 1 <= args.hand_min_depth_landmarks <= 5:
            raise ValueError('--hand-min-depth-landmarks must be in [1, 5]')
        if np.fromstring(args.hand_offset_base, sep=',').shape != (3,):
            raise ValueError('--hand-offset-base must contain exactly three values')
        if args.visual_orientation:
            raise ValueError(
                'MediaPipe visual orientation is not validated; '
                'omit --visual-orientation'
            )

def lift_tcp_with_impedance(backend, args, lift_tcp, model=None, timeout_s=20.0):
    """把阻抗名义 TCP 抬到 lift_tcp[:3]，姿态保持 engage 时的旋转。"""
    from servo_delivery import (  # 若就写在本文件则直接用
        integrate_position_target_from_twist,
        apply_position_target_anti_windup,
    )
    if model is None:
        model = NeroPinocchioModel()
    goal = np.asarray(lift_tcp, dtype=np.float64).ravel()[:3]
    q = np.asarray(backend.get_joint_angles(), dtype=np.float64).ravel()
    tcp_now, R_now = model.fk(q)
    tcp_now = np.asarray(tcp_now, dtype=np.float64).ravel()
    node = ServoDeliveryNode.__new__(ServoDeliveryNode)
    node.args = args
    node.backend = backend
    node.model = model
    node._imp_follow_q = q.copy()
    node._imp_follow_kp_active = np.zeros(7, dtype=np.float64)
    node._imp_smoothed_tcp = tcp_now.copy()
    node._imp_smoothed_R = np.asarray(R_now, dtype=np.float64).reshape(3, 3)
    node._imp_cfg = None
    node._imp_engaged = False
    node._logger = logging.getLogger("lift_impedance")
    node.get_logger = lambda: node._logger
    node._fail_closed = lambda reason: print(f"[lift][impedance][warn] {reason}")
    node._engage(q, time.time())
    deadline = time.time() + float(timeout_s)
    vel = float(getattr(args, "impedance_target_linear_vel", 0.03))
    dt = 1.0 / 30.0
    print(f"[lift][impedance] from {tcp_now.tolist()} toward {goal.tolist()}")
    while time.time() < deadline:
        q = np.asarray(backend.get_joint_angles(), dtype=np.float64).ravel()
        tcp_now, _ = model.fk(q)
        tcp_now = np.asarray(tcp_now, dtype=np.float64).ravel()
        err = goal - tcp_now
        if float(np.linalg.norm(err)) <= 0.012:
            print(f"[lift][impedance] reached pos_err={float(np.linalg.norm(err)):.4f}m")
            return True
        direction = err / max(float(np.linalg.norm(err)), 1e-9)
        twist = np.zeros(6, dtype=np.float64)
        twist[:3] = direction * vel
        node._refresh(q, {"qdot": np.zeros(7), "tcp_twist": twist, "target_tcp": goal}, time.time())
        time.sleep(dt)
    print("[lift][impedance][warn] timeout before reaching lift TCP")
    return False


def run_servo_delivery(args, backend=None):
    from cameras.realsense import camera_for, close_all
    node = None
    target = None
    preview = None
    diagnostics = None
    fusion_observer = fusion_recorder = None
    try:
        validate_servo_args(args)
        if getattr(args, 'fusion_handover', None):
            from handoff.fusion_servo import start_fusion_session
            fusion_observer, fusion_recorder = start_fusion_session(args)
        if bool(getattr(args, "servo_diagnostics", False)):
            try:
                from servo.diagnostics import ServoDiagnostics
                run_id = datetime.now().astimezone().strftime("%Y-%m-%d_%H-%M-%S_%f") + f"_p{os.getpid()}"
                path = Path(getattr(args, "save_dir", "logs") or "logs") / "servo_diagnostics" / run_id / "samples.jsonl"
                diagnostics = ServoDiagnostics(path, metadata={
                    "dry_run": not bool(getattr(args, "execute", False)),
                    "target_source": args.target_source, "handeye": args.handeye,
                    "camera_color_profile": [getattr(args, 'camera_width', 640),
                        getattr(args, 'camera_height', 480), getattr(args, 'camera_fps', 30)],
                    "camera_native_depth_profile": [getattr(args, 'camera_depth_width', None),
                        getattr(args, 'camera_depth_height', None), getattr(args, 'camera_depth_fps', None)],
                    "camera_depth_filters": getattr(args, 'camera_depth_filters', 'off'),
                    "hand_depth_support": 'current_raw_measurements_required_when_filtering',
                    "delivery_distance_m": args.delivery_distance,
                    "max_linear_vel_m_s": args.max_linear_vel,
                    "max_joint_vel_rad_s": args.max_joint_vel,
                    "impedance_target_linear_vel_m_s": args.impedance_target_linear_vel,
                    "impedance_leash_m": args.impedance_max_position_error,
                    "task_torque_limit_nm": VALIDATED_TASK_TORQUE_LIMIT,
                    "total_feedforward_torque_limit_nm": VALIDATED_TOTAL_TORQUE_LIMIT,
                    "jump_reacquire_stable_s": _JUMP_REACQUIRE_S,
                    "jump_reacquire_position_m": _JUMP_REACQUIRE_POSITION_M,
                    "cartesian_k": VALIDATED_CARTESIAN_KP * args.cart_k_scale,
                    "cartesian_b": VALIDATED_CARTESIAN_KD * args.cart_b_scale,
                    "follow_kp": args.follow_kp,
                    "follow_min_qdot_rad_s": args.follow_min_qdot,
                    "follow_kp_rise_time_s": args.follow_kp_rise_time,
                    "follow_max_error_rad": args.follow_max_error_rad,
                    "follow_release_error_rad": args.follow_release_error,
                    "hand_camera_center": bool(getattr(args, 'hand_camera_center', False)),
                    "hand_center_gain": getattr(args, 'hand_center_gain', 2.),
                    "hand_center_max_joint_accel_rad_s2": getattr(args, 'hand_center_max_joint_accel', .5),
                    "hand_center_planning_extra_margin_rad": .02,
                    "hand_follow_safety": (dict(
                        policy='installed_reference_and_measured_progress',
                        reach_bound='diagnostic only; does not block approach outside the outer bound',
                        **vars(HandFollowLimits()))
                        if bool(getattr(args, 'hand_camera_center', False)) else None),
                    "gesture_handover": bool(getattr(args, 'gesture_handover', False)),
                    "gesture_stable_seconds": args.gesture_stable_seconds,
                    "gesture_ready_distance_m": args.gesture_ready_distance,
                    "gesture_still_tolerance_mm": args.gesture_still_tolerance_mm,
                    "gesture_ready_policy": 'centered_stationary_base_within_ready_distance; receiver_motion_revokes_before_release',
                    "gesture_release_boundary": 'first_opening_request_latches_arm_hold_before_sdk_write',
                    "hand_memory_approach": (dict(enabled=True,
                        **vars(HandMemoryLimits()))
                        if args.gesture_handover and args.hand_memory_approach else dict(enabled=False)),
                    "gesture_control_hand_loss_policy": 'pause_keep_current_opening',
                    "gesture_release_policy": 'opposite_hand_pinch_only; torque_is_observation_only',
                    "pbvs_filters": {
                        "target_filter_alpha": args.target_filter_alpha,
                        "normal_filter_alpha": args.normal_filter_alpha,
                        "target_blend_alpha": args.target_blend_alpha,
                        "twist_smooth_alpha": args.twist_smooth_alpha,
                    },
                })
            except Exception:
                diagnostics = None
        model = NeroPinocchioModel()
        # Parse calibration and initialize detection before touching the arm.
        target = build_target(args, model.clone())
        # camera_for(args).read()
        try:
            first_frame = camera_for(args).read()
            warmup = getattr(target, 'warmup', None)
            if callable(warmup) and (backend is None or isinstance(backend, FakeBackend)):
                warmup(first_frame.bgr, first_frame.depth_m, first_frame.K)
        except Exception as exc:
            print(f"[servo][warn] wrist camera unavailable, holding impedance: {exc}")
            if backend is not None and not isinstance(backend, FakeBackend):
                print("[servo] MIT hold continues; electronic e-stop was not requested")
            raise
        if bool(getattr(args, "servo_preview", False)):
            from cameras.servo_preview import ServoPreview
            preview = ServoPreview()
            preview.start()
        if backend is None:
            if getattr(args, 'execute', False):
                from handoff.grasp_to_servo_handoff import build_servo_backend
                backend = build_servo_backend(channel=getattr(args, 'nero_channel', 'can0'),
                    firmware=getattr(args, 'nero_firmware', 'V111').lower())
            else:
                backend = FakeBackend()
        sdk_loop.init(args=[])
        if diagnostics is not None:
            start_log = getattr(backend, "start_control_diagnostics", None)
            if callable(start_log):
                try:
                    status = start_log(path.with_name("control_cycles.jsonl"))
                    print(f"[servo][diag] inner_control_log={status.get('path', path.with_name('control_cycles.jsonl'))}")
                except Exception as exc:
                    print(f"[servo][diag][warn] inner-control logging unavailable: {exc}")
        extra = {} if diagnostics is None else {"diagnostics": diagnostics}
        node = ServoDeliveryNode(backend, args, model, target=target, preview=preview, **extra)
        sdk_loop.spin(node)
    except KeyboardInterrupt:
        return 130
    except Exception:
        print("[servo][warn] servo failed; checking controlled shutdown (not disabling motors)")
        raise
    finally:
        try:
            _close_servo_session(node, backend, target, preview, close_all)
        finally:
            try:
                if fusion_observer is not None:
                    fusion_observer.close()
            finally:
                try:
                    if fusion_recorder is not None:
                        fusion_recorder.close()
                        if fusion_recorder.error:
                            print(f'[fusion][recording_error] {fusion_recorder.error}')
                finally:
                    if diagnostics is not None:
                        try:
                            diagnostics.close()
                        except Exception:
                            pass
    return 0


@defer_sigint()
def _close_servo_session(node, backend, target, preview, close_all):
    sdk_loop.shutdown()
    try:
        try:
            if node is not None:
                node.prepare_shutdown()
        finally:
            if backend is not None:
                backend.disconnect()
    finally:
        try:
            if target is not None and hasattr(target, 'close'):
                target.close()
        finally:
            try:
                close_all()
            finally:
                if preview is not None:
                    preview.close()


if __name__ == "__main__":
    args = parse_servo_args(None)
    raise SystemExit(run_servo_delivery(args, backend=None))
