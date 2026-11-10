#!/usr/bin/env python3
"""Camera ArUco visual servo using Panthera-style PBVS control.

This is the Nero execution path that matches Panthera's ArUco velocity
Jacobian script:

    marker pose -> PBVS TCP twist -> 6x7 Jacobian qdot
    -> small-step q_cmd -> Nero MIT position command + MuJoCo gravity compensation

Panthera sends qdot through Joint_Vel.  On this Nero setup the CPV/follower
path has shown unsafe feedback jumps, so the default path keeps Panthera's PBVS
control law but executes it through the already validated
NeroForceController / NeroMITLoop / NeroGravityCompensator chain.

With ``--control-mode mit-impedance``, the Cartesian equilibrium and a separate
``q_follow`` reference integrate the same joint-limited PBVS command used by the
MIT-position path.  The MIT loop combines a low-gain firmware position spring
for continuous motion with compliant, error-bounded Cartesian impedance torque
at 200 Hz.

Run the dry-run script first and keep emergency stop within reach.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(SCRIPT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from cameras import sdk_loop  # noqa: E402
import cv2  # noqa: E402
from cameras.sdk_loop import Node  # noqa: E402
from cameras.sdk_loop import CameraInfo, Image  # noqa: E402

from aruco_camera_jacobian_servo_sdk_dryrun import (  # noqa: E402
    DEFAULT_HAND_EYE,
    _format,
    _format_or_none,
    _vec7,
    draw_camera_overlay,
    image_msg_to_bgr,
)
from aruco_jacobian_servo_sdk_dryrun import SdkJointFeedbackReader  # noqa: E402
from aruco_start_pose import DEFAULT_ARUCO_START_Q_DEG, parse_deg7  # noqa: E402
from control_backends.base import BackendCommand  # noqa: E402
from control_backends.cartesian_impedance import (  # noqa: E402
    default_cartesian_kd,
)
from control_backends.sdk_cpv_velocity_backend import SdkCpvVelocityBackend  # noqa: E402
from control_backends.force_ctrl.controller import (  # noqa: E402
    ZERO_GRAV_KD,
)
from control_backends.force_ctrl.impedance_profile import (  # noqa: E402
    VALIDATED_CARTESIAN_KD,
    VALIDATED_CARTESIAN_KP,
    VALIDATED_GRAVITY_FACTORS,
    VALIDATED_TASK_TORQUE_LIMIT,
    VALIDATED_TOTAL_TORQUE_LIMIT,
)
from control_backends.sdk_force_ctrl_backend import (  # noqa: E402
    DEFAULT_KD,
    DEFAULT_KP,
    SdkForceCtrlBackend,
)
from servo.execution_safety import ExecutionSafetyMonitor, SafetyParams  # noqa: E402
from servo.jacobian_position_servo import JacobianPositionServo, ServoParams  # noqa: E402
from servo.pbvs_pose_servo import PBVSPoseServo, PBVSParams  # noqa: E402
from targets.aruco_eye_in_hand_target import ArUcoEyeInHandTarget  # noqa: E402


def image_is_fresh(
    last_image_time: Optional[float],
    now: float,
    timeout_s: float,
) -> bool:
    """Return whether a camera frame is recent enough to use for detection."""
    if last_image_time is None or timeout_s <= 0.0:
        return False
    age = float(now) - float(last_image_time)
    return 0.0 <= age <= float(timeout_s)


def _bounded_rate_step(
    velocity: np.ndarray,
    dt: float,
    max_rate: float,
) -> np.ndarray:
    """Integrate one finite 3D velocity sample under a norm rate limit."""
    value = np.asarray(velocity, dtype=np.float64).ravel()
    if value.shape != (3,) or not np.all(np.isfinite(value)):
        raise ValueError(
            f"velocity must have finite shape (3,), got {value.shape}"
        )
    duration = max(0.0, float(dt))
    limit = max(0.0, float(max_rate)) * duration
    step = value * duration
    norm = float(np.linalg.norm(step))
    if limit <= 0.0:
        return np.zeros(3, dtype=np.float64)
    if norm > limit and norm > 1e-12:
        step *= limit / norm
    return step


def integrate_position_target_from_twist(
    equilibrium_tcp: np.ndarray,
    linear_velocity_world: np.ndarray,
    dt: float,
    max_linear_vel: float,
) -> np.ndarray:
    """Advance a Cartesian equilibrium along the filtered PBVS velocity."""
    current = np.asarray(equilibrium_tcp, dtype=np.float64).ravel()
    if current.shape != (3,) or not np.all(np.isfinite(current)):
        raise ValueError(
            f"equilibrium_tcp must have finite shape (3,), got {current.shape}"
        )
    return current + _bounded_rate_step(
        linear_velocity_world, dt, max_linear_vel
    )


def integrate_orientation_target_from_twist(
    equilibrium_rotation: np.ndarray,
    angular_velocity_world: np.ndarray,
    dt: float,
    max_angular_vel: float,
) -> np.ndarray:
    """Advance a rotation using a world-frame PBVS angular velocity."""
    from scipy.spatial.transform import Rotation as Rot

    current = np.asarray(equilibrium_rotation, dtype=np.float64).reshape(3, 3)
    if not np.all(np.isfinite(current)):
        raise ValueError("equilibrium_rotation must be finite")
    step = _bounded_rate_step(
        angular_velocity_world, dt, max_angular_vel
    )
    return Rot.from_matrix(
        Rot.from_rotvec(step).as_matrix() @ current
    ).as_matrix()


def constrain_impedance_target_twist(
    tcp_twist: np.ndarray,
    max_linear_vel: float,
    max_angular_vel: float,
    include_orientation: bool = True,
) -> np.ndarray:
    """Apply the same norm caps used by the moving impedance equilibrium.

    PBVS returns the Cartesian velocity represented by its joint-limited
    command.  The in-loop damper consumes a desired TCP velocity in the same
    world-aligned frame, so keep the feed-forward bounded independently of the
    outer callback period.  Orientation feed-forward is disabled when the
    impedance target is intentionally held at the engagement orientation.
    """
    twist = np.asarray(tcp_twist, dtype=np.float64).ravel()
    if twist.shape != (6,) or not np.all(np.isfinite(twist)):
        raise ValueError(
            f"tcp_twist must have finite shape (6,), got {twist.shape}"
        )
    constrained = twist.copy()
    constrained[:3] = _bounded_rate_step(
        constrained[:3], 1.0, max(0.0, float(max_linear_vel))
    )
    if include_orientation:
        constrained[3:] = _bounded_rate_step(
            constrained[3:], 1.0, max(0.0, float(max_angular_vel))
        )
    else:
        constrained[3:] = 0.0
    return constrained


def apply_position_target_anti_windup(
    previous_target: np.ndarray,
    candidate_target: np.ndarray,
    current_tcp: np.ndarray,
    max_error: Optional[float],
) -> np.ndarray:
    """Stop an equilibrium update from accumulating beyond its force leash."""
    previous = np.asarray(previous_target, dtype=np.float64).ravel()
    candidate = np.asarray(candidate_target, dtype=np.float64).ravel()
    current = np.asarray(current_tcp, dtype=np.float64).ravel()
    if previous.shape != (3,) or candidate.shape != (3,) or current.shape != (3,):
        raise ValueError("position targets and current_tcp must have shape (3,)")
    if max_error is None:
        return candidate.copy()

    bound = max(0.0, float(max_error))
    previous_error = float(np.linalg.norm(previous - current))
    candidate_delta = candidate - current
    candidate_error = float(np.linalg.norm(candidate_delta))
    tolerance = 1e-12

    if candidate_error <= bound + tolerance:
        return candidate.copy()
    if previous_error > bound + tolerance:
        # An external push can put a fixed reference outside the leash. Keep the
        # reference fixed unless this update actively reduces that separation.
        if candidate_error < previous_error - tolerance:
            return candidate.copy()
        return previous.copy()
    if candidate_error <= tolerance:
        return current.copy()
    return current + candidate_delta * (bound / candidate_error)


def apply_orientation_target_anti_windup(
    previous_target: np.ndarray,
    candidate_target: np.ndarray,
    current_rotation: np.ndarray,
    max_error_rad: Optional[float],
) -> np.ndarray:
    """SO(3) counterpart of :func:`apply_position_target_anti_windup`."""
    from scipy.spatial.transform import Rotation as Rot

    previous = np.asarray(previous_target, dtype=np.float64).reshape(3, 3)
    candidate = np.asarray(candidate_target, dtype=np.float64).reshape(3, 3)
    current = np.asarray(current_rotation, dtype=np.float64).reshape(3, 3)
    if not (
        np.all(np.isfinite(previous))
        and np.all(np.isfinite(candidate))
        and np.all(np.isfinite(current))
    ):
        raise ValueError("orientation targets and current_rotation must be finite")
    if max_error_rad is None:
        return candidate.copy()

    bound = max(0.0, float(max_error_rad))
    candidate_rotvec = Rot.from_matrix(current.T @ candidate).as_rotvec()
    candidate_error = float(np.linalg.norm(candidate_rotvec))
    tolerance = 1e-12

    if candidate_error <= bound + tolerance:
        return candidate.copy()

    # Keep the nominal orientation itself inside the same leash used by the
    # 200 Hz impedance law.  Letting a candidate outside the leash through
    # merely because it is moving toward the marker creates an unbounded
    # reference/actual gap when the robot cannot rotate as fast as the visual
    # target.  That gap was visible as imp_ref_rot_err=30+ deg while the
    # effective inner error stayed at 5 deg, and it made the outer PBVS loop
    # keep saturating joint velocity.
    if bound <= tolerance or candidate_error <= tolerance:
        return current.copy()
    clipped_delta = Rot.from_rotvec(
        candidate_rotvec * (bound / candidate_error)
    ).as_matrix()
    return Rot.from_matrix(current @ clipped_delta).as_matrix()


def effective_task_torque_limit(
    legacy_limit: float,
    task_limit: float,
) -> Optional[float]:
    """Resolve the two compatibility flags to one conservative task limit."""
    positive_limits = [
        float(value)
        for value in (legacy_limit, task_limit)
        if float(value) > 0.0
    ]
    return min(positive_limits) if positive_limits else None


def effective_startup_limit_margin(
    control_mode: str,
    startup_limit_margin: float,
    impedance_safe_margin: float,
) -> float:
    """Use the stricter impedance margin before a compliant controller starts."""
    if control_mode == "mit-impedance":
        return max(float(startup_limit_margin), float(impedance_safe_margin))
    return float(startup_limit_margin)


class ArucoCameraServoExecuteNode(Node):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__("nero_aruco_camera_jacobian_servo_sdk_execute")
        self.args = args
        self.last_image = None
        # Bind every camera frame to one joint-state snapshot. Reusing an old
        # image with a newer q would make a stationary marker appear to move in
        # base coordinates whenever the arm moves or is pushed.
        self.last_image_q: Optional[np.ndarray] = None
        self.last_image_time: Optional[float] = None
        self._image_generation = 0
        self._detected_image_generation = -1
        self._cached_image_target = None
        self.camera_matrix: Optional[np.ndarray] = None
        self.dist_coeffs: Optional[np.ndarray] = None
        self.last_log_time = 0.0
        self.last_image_warn_time = 0.0
        self.last_detect_time: Optional[float] = None
        self.started = time.monotonic()
        self.count = 0
        self.display_enabled = bool(args.display)
        self.display_initialized = False
        self.display_window = "Nero ArUco Servo Execute"

        # Zero-gravity KD for preflight/impedance settle (damping-only).
        # Same values as test_zero_gravity.py and NeroForceController.
        self.zero_grav_kd = ZERO_GRAV_KD.copy()

        # MIT mode: construct all fallible camera/servo/ROS objects first, then
        # connect at the end of __init__ so gravity comp is active from the
        # first timer cycle. CPV still uses a reader during preflight.
        self._mit_backend_started = False
        if args.execution_backend == "cpv":
            self.reader = SdkJointFeedbackReader(
                channel=args.channel,
                interface=args.interface,
                firmware=args.firmware,
            )
            self.backend = None
        else:
            # MIT: prepare the zero-gravity backend; connection is deferred.
            self.reader = None
            # In-loop Cartesian impedance (mit-impedance) needs a Pinocchio model
            # inside the 200 Hz loop (fk_and_jacobian). Build it eagerly ONLY for
            # mit-impedance so mit-position startup is unchanged; mit-position
            # never calls set_cartesian_impedance, so pin_model=None is fine.
            pin_model = None
            if args.control_mode == "mit-impedance":
                from models.nero_model import NeroPinocchioModel
                self._imp_model = NeroPinocchioModel()
                # The 200 Hz torque loop and the ROS/visual outer loop both run
                # FK, but Pinocchio Data is mutable and not thread-safe.
                pin_model = self._imp_model.clone()
            self.backend = SdkForceCtrlBackend(
                dry_run=False,
                channel=args.channel,
                interface=args.interface,
                firmware=args.firmware,
                control_hz=args.control_hz,
                torque_limit=args.torque_limit,
                impedance_torque_limit=(
                    effective_task_torque_limit(
                        args.impedance_torque_limit,
                        args.impedance_task_torque_limit,
                    )
                    or args.torque_limit
                ),
                gravity_factor=args.gravity_factor,
                gravity_factors=args.joint_factors,
                zero_gravity=True,
                gravity_compensation=True,
                startup_joint_margin=effective_startup_limit_margin(
                    args.control_mode,
                    args.startup_limit_margin,
                    args.impedance_safe_margin,
                ),
                pin_model=pin_model,
            )
            # Connect only after all camera/servo/ROS objects below have been
            # constructed.  If any of those constructors raises, no real
            # hardware has been activated and there is nothing to leak.
            self.last_mit_q_cmd = None
        self.phase = "preflight"
        self.settle_started: Optional[float] = None
        self.settle_q_ref: Optional[np.ndarray] = None
        self.last_preflight_q: Optional[np.ndarray] = None
        self.progress_best_error: Optional[float] = None
        self.progress_last_improved_time: Optional[float] = None
        self.progress_last_marker: Optional[np.ndarray] = None
        self.progress_last_target: Optional[np.ndarray] = None
        if not hasattr(self, 'last_mit_q_cmd'):
            self.last_mit_q_cmd: Optional[np.ndarray] = None
        # Accumulated MIT position target: integrates qdot from the previous
        # target (not from q_now), so the MIT loop sees a growing position
        # error and generates enough torque to actually move the arm.
        # Panthera's 13_blue_circle_follow.py does the same via IK output.
        self._mit_target_q: Optional[np.ndarray] = None
        # Impedance mode state (mit-impedance)
        self._impedance_fallback_triggered = False
        self._impedance_fallback_reason = ""
        self._impedance_terminal_fault = False
        self._qdot_ringbuf: list = []                 # ring buffer of (t, q) for RMS
        self._imp_track_start_q: Optional[np.ndarray] = None  # excursion baseline
        self._imp_follow_q: Optional[np.ndarray] = None       # nominal PBVS q reference
        # Cached Pinocchio model for J^T force mapping (built lazily)
        if not hasattr(self, "_imp_model"):
            self._imp_model = None

        self.target_source = ArUcoEyeInHandTarget(
            handeye_path=args.handeye,
            aruco_dict=args.aruco_dict,
            marker_id=args.marker_id,
            marker_size=args.marker_size,
            handeye_convention=args.handeye_convention,
            handeye_mount_frame=args.handeye_mount_frame,
            offset_base=args.offset_base,
            offset_marker=args.offset_marker,
        )

        # In MIT position-follow mode, do NOT limit per-cycle joint step.
        # The servo already limits qdot via max_joint_vel; kp/kd gains in
        # the 200 Hz MIT loop provide natural smoothing.  A tight step limit
        # (e.g. 0.001 rad) makes q_cmd barely move ahead of q_now, producing
        # negligible torque (kp * 0.001 ~ 0.03 Nm) that cannot overcome
        # static friction.  Panthera's 13_blue_circle_follow.py uses IK
        # output directly as q_cmd with no step limiting.
        effective_max_joint_step = args.max_joint_step
        if args.execution_backend != "cpv":
            effective_max_joint_step = args.max_joint_vel  # one full second of motion

        if args.servo_mode == "pbvs":
            params = PBVSParams(
                servo_hz=args.servo_hz,
                desired_distance=args.desired_distance,
                linear_gain=args.linear_gain,
                linear_integral_gain=args.linear_integral_gain,
                linear_derivative_gain=args.linear_derivative_gain,
                angular_gain=args.angular_gain,
                # In compliant impedance mode, --no-impedance-visual-orientation
                # must also disable PBVS angular qdot. Otherwise the outer
                # servo keeps chasing marker attitude while the inner law holds
                # the engagement orientation, consuming joint-speed budget and
                # producing apparent stalls/upward wrist motion.
                enable_orientation=(
                    args.control_mode != "mit-impedance"
                    or args.impedance_visual_orientation
                ),
                orientation_twist_scale=float(
                    getattr(args, "pbvs_orientation_scale", 1.0)
                ),
                orientation_error_limit_rad=(
                    np.deg2rad(float(args.impedance_max_orientation_error_deg))
                    if (
                        args.control_mode == "mit-impedance"
                        and args.impedance_visual_orientation
                    )
                    else 0.0
                ),
                max_linear_vel=args.max_linear_vel,
                max_angular_vel=args.max_angular_vel,
                damping=args.damping,
                max_joint_vel=args.max_joint_vel,
                max_joint_accel=args.max_joint_accel,
                max_joint_step=effective_max_joint_step,
                marker_center_alpha=args.target_filter_alpha,
                marker_normal_alpha=args.normal_filter_alpha,
                marker_tangent_alpha=args.normal_filter_alpha,
                target_blend_alpha=float(
                    getattr(args, "target_blend_alpha", 0.82)
                ),
                tcp_twist_smooth_alpha=args.twist_smooth_alpha,
                orientation_blend_alpha=args.orientation_filter_alpha,
                target_timeout_s=args.target_timeout_s,
                max_target_jump=args.max_target_jump,
                position_deadband=args.position_deadband,
                orientation_deadband_rad=np.deg2rad(args.orientation_deadband_deg),
            )
            self.servo = PBVSPoseServo(params=params)
        else:
            params = ServoParams(
                servo_hz=args.servo_hz,
                desired_distance=args.desired_distance,
                linear_gain=args.linear_gain,
                max_linear_vel=args.max_linear_vel,
                damping=args.damping,
                max_joint_vel=args.max_joint_vel,
                max_joint_step=effective_max_joint_step,
                target_filter_alpha=args.target_filter_alpha,
                normal_filter_alpha=args.normal_filter_alpha,
                qdot_filter_alpha=args.qdot_filter_alpha,
                target_timeout_s=args.target_timeout_s,
                max_target_jump=args.max_target_jump,
                position_deadband=args.position_deadband,
            )
            self.servo = JacobianPositionServo(params=params)
        self.params = params
        # Cartesian impedance gains (mit-impedance mode)
        self.cart_kp = np.asarray(args.cartesian_kp, dtype=np.float64).ravel()
        self.cart_kd = (
            np.asarray(args.cartesian_kd, dtype=np.float64).ravel()
            if args.cartesian_kd is not None
            else default_cartesian_kd(self.cart_kp)
        )
        # In-loop impedance law + the pieces ported from the standalone tuning
        # harness (test_official_cartesian_impedance.py), so the in-loop law can
        # reproduce that exact validated feel. Defaults match the harness baseline.
        self.imp_use_dls = (args.impedance_law == "panthera-real")
        self.imp_coulomb = _vec7(args.impedance_coulomb) if isinstance(args.impedance_coulomb, str) \
            else np.asarray(args.impedance_coulomb, dtype=np.float64).ravel()
        self.imp_joint_weight = _vec7(args.impedance_joint_weights) if isinstance(args.impedance_joint_weights, str) \
            else np.asarray(args.impedance_joint_weights, dtype=np.float64).ravel()
        self.imp_task_torque_limit = effective_task_torque_limit(
            args.impedance_torque_limit,
            args.impedance_task_torque_limit,
        )
        self.imp_tau_limit_per_joint = (
            np.asarray(args.impedance_tau_limit_per_joint, dtype=np.float64).ravel()
            if args.impedance_tau_limit_per_joint is not None
            else np.full(7, float(args.torque_limit)))
        self.imp_velocity_signs = _vec7(args.impedance_velocity_signs) if isinstance(args.impedance_velocity_signs, str) \
            else np.asarray(args.impedance_velocity_signs, dtype=np.float64).ravel()
        self.imp_use_real_velocity = bool(args.use_real_velocity)
        self.imp_real_vel_filter_tau = float(args.impedance_real_vel_filter_tau)
        self.imp_qdot_trip_fast = float(args.impedance_qdot_trip_fast)
        # Optional direct joint hold is disabled by default for moving visual
        # impedance; enable selected gains explicitly only after validation.
        self.imp_joint_hold_kp = (
            _vec7(args.impedance_joint_hold_kp) if isinstance(args.impedance_joint_hold_kp, str)
            else np.asarray(args.impedance_joint_hold_kp, dtype=np.float64).ravel()
        )
        follow_kp_arg = getattr(args, "impedance_follow_kp", np.zeros(7))
        self.imp_follow_kp = (
            _vec7(follow_kp_arg)
            if isinstance(follow_kp_arg, str)
            else np.asarray(follow_kp_arg, dtype=np.float64).ravel()
        )
        self._imp_follow_kp_active = self.imp_follow_kp.copy()
        self.imp_follow_min_qdot = float(
            getattr(args, "impedance_follow_min_qdot", 0.08)
        )
        self.imp_follow_release_error = float(
            getattr(args, "impedance_follow_release_error", 0.02)
        )
        self.imp_joint_hold_kd = None
        if args.impedance_joint_hold_kd is not None:
            self.imp_joint_hold_kd = (
                _vec7(args.impedance_joint_hold_kd)
                if isinstance(args.impedance_joint_hold_kd, str)
                else np.asarray(
                    args.impedance_joint_hold_kd, dtype=np.float64
                ).ravel()
            )
        # Engagement state: the in-loop impedance cfg is installed ONCE on the
        # first tracking cycle; thereafter only its moving target is refreshed.
        self._imp_engaged = False
        self._imp_cfg = None
        self._imp_smoothed_tcp: Optional[np.ndarray] = None
        self._imp_smoothed_R: Optional[np.ndarray] = None
        self._imp_fixed_R: Optional[np.ndarray] = None
        self._imp_target_update_time: Optional[float] = None
        self._imp_engage_monotonic: Optional[float] = None
        self._imp_engage_cycle_baseline: Optional[int] = None
        self.safety = ExecutionSafetyMonitor(
            SafetyParams(
                startup_limit_margin=effective_startup_limit_margin(
                    args.control_mode,
                    args.startup_limit_margin,
                    args.impedance_safe_margin,
                ),
                runtime_limit_margin=args.runtime_limit_margin,
                max_tracking_error=args.max_tracking_error,
                max_tracking_error_s=args.max_tracking_error_s,
            )
        )
        self.kp = DEFAULT_KP.copy() * float(args.kp_scale)
        self.kd = DEFAULT_KD.copy() * float(args.kd_scale)

        self.stable_target_count = 0
        self.last_stable_marker: Optional[np.ndarray] = None
        self.start_pose_checked = False

        # Sensor frames are perishable; depth=1 avoids processing a backlog
        # with stale arm poses.
        self.create_subscription(Image, args.image_topic, self.on_image, 1)
        self.create_subscription(CameraInfo, args.camera_info_topic, self.on_camera_info, 5)
        self.create_timer(1.0 / float(args.servo_hz), self.on_timer)

        if self.reader is not None:
            self.reader.connect()
        elif self.backend is not None:
            try:
                self.backend.connect()
                self._mit_backend_started = True
                self.last_mit_q_cmd = self.backend.get_joint_angles()
            except Exception:
                # connect() can fail after partially creating the force
                # controller.  Its disconnect path is deliberately idempotent.
                cleanup_errors = []
                try:
                    self.backend.emergency_stop()
                except Exception as exc:
                    cleanup_errors.append(f"emergency stop failed: {exc}")
                try:
                    self.backend.disconnect()
                except Exception as exc:
                    cleanup_errors.append(f"disconnect failed: {exc}")
                if cleanup_errors:
                    raise RuntimeError(
                        "MIT backend construction failed and cleanup was not "
                        f"confirmed: {'; '.join(cleanup_errors)}")
                self.backend = None
                raise
        self.get_logger().warning(
            f"REAL EXECUTION requested: {args.execution_backend} control may start after preflight passes."
        )
        if self._mit_backend_started:
            self.get_logger().info(
                "MIT backend active in zero-gravity mode; arm is draggable during preflight."
            )
        self.get_logger().info(f"execution_backend={args.execution_backend} firmware={args.firmware}")
        self.get_logger().info(f"image_topic={args.image_topic}")
        self.get_logger().info(f"camera_info_topic={args.camera_info_topic}")
        self.get_logger().info(
            f"marker_id={args.marker_id} marker_size={args.marker_size:.4f}m "
            f"aruco_dict={args.aruco_dict}"
        )
        self.get_logger().info(f"handeye={args.handeye}")
        self.get_logger().info(
            f"servo: hz={args.servo_hz:.1f} max_linear_vel={args.max_linear_vel:.4f}m/s "
            f"max_joint_vel={args.max_joint_vel:.4f}rad/s "
            f"max_joint_accel={getattr(args, 'max_joint_accel', 0.0):.4f}rad/s^2 "
            f"max_joint_step={args.max_joint_step:.4f}rad"
            f" (effective={effective_max_joint_step:.4f}rad)"
        )
        if args.control_mode == "mit-impedance":
            self.get_logger().warning(
                "impedance profile: "
                f"law={args.impedance_law} "
                f"K={_format(self.cart_kp, 3)} B={_format(self.cart_kd, 3)} "
                f"weights={_format(self.imp_joint_weight, 3)} "
                f"gravity_factors={_format(args.joint_factors, 3)} "
                f"kd_scale={args.kd_scale:.2f} "
                f"task_limit={self.imp_task_torque_limit}Nm "
                f"total_limit={args.torque_limit:.1f}Nm "
                f"real_velocity={self.imp_use_real_velocity} "
                f"coulomb={_format(self.imp_coulomb, 3)} "
                f"coulomb_smooth_width_deg={max(0.0, float(getattr(args, 'impedance_coulomb_smooth_width_deg', 0.0))):.2f} "
                f"follow_kp={_format(self.imp_follow_kp, 3)} "
                f"follow_min_qdot={self.imp_follow_min_qdot:.3f} "
                f"follow_release_error={self.imp_follow_release_error:.3f}rad"
            )
            self.get_logger().warning(
                "impedance target guards: "
                f"linear_rate={args.impedance_target_linear_vel:.3f}m/s "
                f"position_leash={args.impedance_max_position_error:.3f}m "
                f"visual_orientation={args.impedance_visual_orientation} "
                f"orientation_leash={args.impedance_max_orientation_error_deg:.1f}deg "
                f"image_timeout={args.image_timeout_s:.3f}s"
            )

    def on_image(self, msg: Image) -> None:
        try:
            q_image = self._read_joint_angles()
            image = image_msg_to_bgr(msg)
            now = time.monotonic()
            self.last_image = image
            self.last_image_q = (
                None
                if q_image is None
                else np.asarray(q_image, dtype=np.float64).ravel().copy()
            )
            self.last_image_time = now
            self._image_generation += 1
        except Exception as exc:
            now = time.monotonic()
            if now - self.last_image_warn_time >= 1.0:
                self.get_logger().warning(
                    f"image/joint snapshot capture failed: {exc}"
                )
                self.last_image_warn_time = now

    def _detect_latest_image_target(self, now: float):
        """Detect at most once per image, using that image's joint snapshot."""
        if not image_is_fresh(
            self.last_image_time, now, self.args.image_timeout_s
        ):
            # A short camera gap must not reset the outer PBVS velocity to zero
            # and then restart it on the next frame.  Keep using the last
            # image/q-paired base-frame target for the configured target grace
            # period; the impedance equilibrium itself remains rate-limited.
            grace_s = max(
                0.0,
                float(getattr(self.args, "target_timeout_s", 0.0)),
            )
            if (
                self._cached_image_target is not None
                and self.last_detect_time is not None
                and now - self.last_detect_time <= grace_s
            ):
                return self._cached_image_target, False
            return None, False
        if self.last_image_q is None:
            if now - self.last_image_warn_time >= 1.0:
                self.get_logger().warning(
                    "fresh camera image has no paired joint snapshot; "
                    "ignoring this frame"
                )
                self.last_image_warn_time = now
            return None, False

        generation = int(self._image_generation)
        is_new = generation != self._detected_image_generation
        if is_new:
            detected_target = self.target_source.detect(
                image=self.last_image,
                camera_matrix=self.camera_matrix,
                dist_coeffs=self.dist_coeffs,
                q_arm=self.last_image_q.copy(),
            )
            self._detected_image_generation = generation
            if detected_target is not None:
                self._cached_image_target = detected_target
                self.last_detect_time = now
                return self._cached_image_target, True

            # A single ArUco miss is common during motion. Dropping straight
            # into the hold branch makes PBVS qdot jump to zero and restart on
            # reacquisition, which is perceived as stick-slip. Reuse the last
            # valid, image/q-paired target for the configured PBVS timeout;
            # once that grace expires the stale target is discarded.
            grace_s = max(0.0, float(getattr(self.args, "target_timeout_s", 0.0)))
            if (
                self._cached_image_target is not None
                and self.last_detect_time is not None
                and now - self.last_detect_time <= grace_s
            ):
                return self._cached_image_target, False

            self._cached_image_target = None
        return self._cached_image_target, False

    def on_camera_info(self, msg: CameraInfo) -> None:
        self.camera_matrix = np.asarray(msg.k, dtype=np.float64).reshape(3, 3)
        self.dist_coeffs = np.asarray(msg.d, dtype=np.float64)
        if len(self.dist_coeffs) == 0:
            self.dist_coeffs = np.zeros(5, dtype=np.float64)

    def on_timer(self) -> None:
        """Fail closed on any unexpected impedance-mode timer exception."""
        try:
            self._on_timer_impl()
        except Exception as exc:
            if self.args.control_mode == "mit-impedance":
                self._fail_closed_impedance(
                    f"unexpected impedance timer exception: "
                    f"{type(exc).__name__}: {exc}")
                if sdk_loop.ok():
                    sdk_loop.shutdown()
                return
            raise

    def _on_timer_impl(self) -> None:
        now = time.monotonic()
        if self._check_inner_impedance_fault():
            sdk_loop.shutdown()
            return
        if self._impedance_terminal_fault:
            sdk_loop.shutdown()
            return
        if self.args.duration > 0.0 and now - self.started >= self.args.duration:
            self.get_logger().info(f"duration reached: {self.args.duration:.1f}s")
            sdk_loop.shutdown()
            return

        q_now = self._read_joint_angles()
        if q_now is None:
            self._update_display("waiting for SDK joint feedback", None, None)
            self._log_waiting(now, "waiting for SDK joint feedback")
            return
        if (
            self.args.control_mode == "mit-impedance"
            and self._impedance_fallback_triggered
        ):
            # A fault latch is terminal for this run. Keep the 200 Hz loop and
            # gravity compensation alive, without consulting stale vision or
            # automatically re-entering any tracking controller.
            assert self.backend is not None
            cmd = self._safe_damping_hold_command(q_now, now)
            try:
                self.backend.send(cmd)
            except Exception as exc:
                self._fail_closed_impedance(
                    f"latched safe-hold command failed: {exc}")
                sdk_loop.shutdown()
                return
            self.last_mit_q_cmd = q_now.copy()
            self.count += 1
            self._update_display("impedance_safe_hold", None, None)
            if now - self.last_log_time >= self.args.log_period:
                self.get_logger().warning(
                    "impedance_safe_hold (latched): "
                    f"{self._impedance_fallback_reason}"
                )
                self.last_log_time = now
            return
        if self.last_image is None:
            self._log_waiting(now, "waiting for camera image")
            return
        if self.camera_matrix is None or self.dist_coeffs is None:
            self._update_display("waiting for camera_info", None, None)
            self._log_waiting(now, "waiting for camera_info")
            return

        target = None
        target_is_new = False
        if self.last_image_time is not None:
            target, target_is_new = self._detect_latest_image_target(now)
        if (
            not image_is_fresh(self.last_image_time, now, self.args.image_timeout_s)
            and now - self.last_image_warn_time >= 1.0
        ):
            image_age = (
                "unknown"
                if self.last_image_time is None
                else f"{now - self.last_image_time:.3f}s"
            )
            self.get_logger().warning(
                f"camera image stale (age={image_age}, "
                f"timeout={self.args.image_timeout_s:.3f}s); freezing target"
            )
            self.last_image_warn_time = now

        if self.phase == "preflight":
            display_status = "preflight: marker visible" if target is not None else "preflight: waiting for marker"
            self._update_display(display_status, target, None)
            self._send_zero_gravity(q_now, now)
            self._preflight_step(
                q_now, target, now, new_observation=target_is_new
            )
            return
        if self.phase == "settle":
            self._update_display(f"{self.args.execution_backend}_settle", target, None)
            self._settle_step(q_now, target, now)
            return

        servo_out, status = self._servo_step(q_now, target, now)
        decision = self.safety.evaluate(q_now, servo_out, now)
        if decision.status != "tracking":
            status = decision.status

        if decision.fatal:
            self._handle_fatal_safety_stop(status, decision.reason)
            sdk_loop.shutdown()
            return

        if decision.hold:
            prefer_last_hold = decision.status in {
                "marker_lost_hold",
                "target_rejected_hold",
            }
            if self.args.control_mode == "mit-impedance":
                self._release_impedance_follow_spring()
            cmd = self._tracking_hold_command(
                q_now, now, prefer_last=prefer_last_hold
            )
            # Reset accumulated target so it re-initializes from current position
            # when tracking resumes.
            self._mit_target_q = None
        else:
            if self.args.execution_backend == "cpv":
                cmd = BackendCommand(
                    q_cmd=q_now.copy(),
                    qdot_cmd=servo_out["qdot"],
                    mode="velocity",
                    stamp=now,
                )
            elif self.args.control_mode == "mit-impedance":
                cmd = self._build_impedance_command(q_now, servo_out, now)
            else:
                # MIT position-follow: accumulate q_cmd from previous target
                # (NOT from q_now), so the position error grows over cycles
                # and the MIT kp generates enough torque to move the arm.
                # This matches Panthera's approach: IK outputs the full target
                # joint position, MIT tracks it with kp/kd + gravity.
                qdot = np.asarray(servo_out["qdot"], dtype=np.float64).ravel()
                if self._mit_target_q is None:
                    self._mit_target_q = q_now.copy()
                self._mit_target_q += qdot * self.params.dt
                # Clamp: don't let the virtual target run away more than
                # max_joint_vel ahead of current position (1 second of motion).
                max_lead = float(self.args.max_joint_vel)
                diff = self._mit_target_q - q_now
                self._mit_target_q = q_now + np.clip(diff, -max_lead, max_lead)
                self._mit_target_q = self._model_clamp(self._mit_target_q)

                cmd = BackendCommand(
                    q_cmd=self._mit_target_q.copy(),
                    qdot_cmd=qdot,
                    kp=self.kp,
                    kd=self.kd,
                    mode="mit",
                    stamp=now,
                )

        if self._impedance_terminal_fault:
            sdk_loop.shutdown()
            return

        reason = decision.reason
        progress_reason = self._check_progress_guard(status, target, servo_out, cmd, now)
        if progress_reason:
            status = "nonconvergent_hold"
            reason = progress_reason
            cmd = self._hold_command(q_now, now, prefer_last=True)

        assert self.backend is not None
        try:
            self.backend.send(cmd)
        except Exception as exc:
            if self.args.control_mode == "mit-impedance":
                self._fail_closed_impedance(
                    f"impedance command send failed: {exc}")
                sdk_loop.shutdown()
                return
            raise
        if self.args.execution_backend != "cpv":
            self.last_mit_q_cmd = np.asarray(cmd.q_cmd, dtype=np.float64).ravel().copy()
        # Impedance instability detector: latch into gravity+damping safe hold
        # on oscillation, excursion, limit proximity, or instantaneous spike.
        if self.args.control_mode == "mit-impedance":
            self._check_impedance_instability(q_now, now)
            if self._impedance_terminal_fault:
                sdk_loop.shutdown()
                return
        self.count += 1
        self._update_display(status, target, servo_out)

        if now - self.last_log_time >= self.args.log_period:
            self._log_status(status, q_now, target, servo_out, now, reason, cmd)
            self.last_log_time = now

    def _check_inner_impedance_fault(self) -> bool:
        """Stop if the 200 Hz loop faults or misses the impedance handoff."""
        if self.args.control_mode != "mit-impedance" or self.backend is None:
            return False
        getter = getattr(self.backend, "get_debug_state", None)
        if getter is None:
            return False
        debug = getter() or {}
        error = debug.get("last_error")
        handoff_error = None
        if not error and self._imp_engaged:
            required = {
                "running",
                "impedance_active",
                "control_path",
                "impedance_cycle_count",
            }
            if required.issubset(debug):
                elapsed = (
                    0.0
                    if self._imp_engage_monotonic is None
                    else time.monotonic() - self._imp_engage_monotonic
                )
                baseline = (
                    int(debug["impedance_cycle_count"])
                    if self._imp_engage_cycle_baseline is None
                    else self._imp_engage_cycle_baseline
                )
                cycles_advanced = int(debug["impedance_cycle_count"]) > baseline
                handoff_ok = (
                    bool(debug["running"])
                    and bool(debug["impedance_active"])
                    and debug["control_path"] == "impedance"
                    and cycles_advanced
                )
                if elapsed >= 0.15 and not handoff_ok:
                    handoff_error = (
                        "in-loop impedance handoff not confirmed after "
                        f"{elapsed:.3f}s: running={debug['running']} "
                        f"active={debug['impedance_active']} "
                        f"path={debug['control_path']} "
                        f"cycles={debug['impedance_cycle_count']} "
                        f"baseline={baseline}"
                    )
        if handoff_error:
            self._fail_closed_impedance(handoff_error)
            return True
        if not error:
            return False

        self._impedance_fallback_triggered = True
        self._impedance_fallback_reason = f"hard inner trip: {error}"
        self._impedance_terminal_fault = True
        self._imp_engaged = False
        trip = debug.get("impedance_trip") or {}
        self.get_logger().error(
            f"{self._impedance_fallback_reason}; "
            f"kind={trip.get('trigger_kind')} "
            f"joint={trip.get('trigger_joint')}. "
            "The MIT loop has already issued electronic e-stop; ending run."
        )
        return True

    def _send_zero_gravity(self, q_now: np.ndarray, now: float) -> None:
        """Send zero-gravity command (kp=0, small kd, gravity comp on).

        This keeps the arm in a draggable state with gravity compensation,
        matching Panthera's behavior during non-tracking phases.
        Only applies to MIT backend; CPV backend sends zero velocity instead.
        """
        if self.backend is None:
            return
        if self.args.execution_backend == "cpv":
            self.backend.send(BackendCommand(
                q_cmd=q_now.copy(),
                qdot_cmd=np.zeros(7, dtype=np.float64),
                mode="velocity",
                stamp=now,
            ))
        else:
            self.backend.send(BackendCommand(
                q_cmd=q_now.copy(),
                qdot_cmd=np.zeros(7, dtype=np.float64),
                kp=np.zeros(7, dtype=np.float64),
                kd=self.zero_grav_kd,
                mode="mit",
                stamp=now,
            ))

    def _read_joint_angles(self) -> Optional[np.ndarray]:
        if self.backend is not None:
            value = self.backend.get_joint_angles()
        elif self.reader is not None:
            value = self.reader.read_joint_angles()
        else:
            return None
        if value is None:
            return None
        q = np.asarray(value, dtype=np.float64).ravel()
        if q.shape != (7,) or not np.all(np.isfinite(q)):
            raise RuntimeError(
                "invalid SDK joint feedback: "
                f"expected finite shape (7,), got shape={q.shape}, "
                f"values={q.tolist()}")
        return q

    def _preflight_step(
        self,
        q_now: np.ndarray,
        target,
        now: float,
        new_observation: bool = True,
    ) -> None:
        ok, reason = self.safety.check_start_posture(q_now)
        if not ok:
            self.get_logger().error(f"refusing to start: unsafe start posture: {reason}")
            sdk_loop.shutdown()
            return

        if not self._check_start_pose(q_now):
            sdk_loop.shutdown()
            return

        if target is None:
            self.stable_target_count = 0
            self.last_stable_marker = None
            self._log_waiting(now, "preflight: waiting for stable marker")
            return

        # Stability means distinct camera observations, not repeated timer
        # processing of one cached frame.
        if not new_observation:
            self._log_waiting(now, "preflight: waiting for next camera frame")
            return

        self.last_preflight_q = q_now.copy()
        marker = target.marker_center_base
        if self.last_stable_marker is None:
            self.stable_target_count = 1
        else:
            jump = float(np.linalg.norm(marker - self.last_stable_marker))
            if jump <= self.args.preflight_target_tolerance:
                self.stable_target_count += 1
            else:
                self.stable_target_count = 1
                self.get_logger().warning(f"preflight marker moved: {jump:.4f}m, restarting stability count")
        self.last_stable_marker = marker.copy()

        if now - self.last_log_time >= self.args.log_period:
            self.get_logger().info(
                f"preflight stable={self.stable_target_count}/{self.args.preflight_stable_samples} "
                f"marker_base={_format(marker, 4)} q_now_deg={_format(np.degrees(q_now), 2)}"
            )
            self.last_log_time = now

        if self.stable_target_count >= self.args.preflight_stable_samples:
            self._start_backend()

    def _check_start_pose(self, q_now: np.ndarray) -> bool:
        if not self.args.require_start_pose:
            return True
        q_ref = np.radians(np.asarray(self.args.start_q_deg, dtype=np.float64).ravel())
        diff_deg = np.abs(np.degrees(q_now - q_ref))
        max_diff = float(np.max(diff_deg))
        if max_diff <= self.args.start_tolerance_deg:
            if not self.start_pose_checked:
                self.get_logger().info(
                    f"start pose accepted: max_diff={max_diff:.2f}deg "
                    f"q_now_deg={_format(np.degrees(q_now), 2)}"
                )
                self.start_pose_checked = True
            return True

        self.get_logger().error(
            "refusing to start: current q is not near configured start pose; "
            f"max_diff={max_diff:.2f}deg tolerance={self.args.start_tolerance_deg:.2f}deg\n"
            f"  q_now_deg={_format(np.degrees(q_now), 2)}\n"
            f"  start_q_deg={_format(self.args.start_q_deg, 2)}"
        )
        return False

    def _start_backend(self) -> None:
        backend_name = self.args.execution_backend
        self.get_logger().warning(f"preflight passed; enabling {backend_name} backend")

        if self._mit_backend_started:
            # MIT backend already running since __init__; just validate posture.
            q_now = self.backend.get_joint_angles() if self.backend is not None else None
            if q_now is None:
                self.get_logger().error("MIT backend running but no joint feedback")
                sdk_loop.shutdown()
                return
            q_now = np.asarray(q_now, dtype=np.float64).ravel()
            ok, reason = self.safety.check_start_posture(q_now)
            if not ok:
                self.get_logger().error(f"unsafe posture: {reason}")
                sdk_loop.shutdown()
                return
            if not self._check_start_pose(q_now):
                sdk_loop.shutdown()
                return
            self.last_mit_q_cmd = q_now.copy()
            self.phase = "settle"
            self.settle_started = time.monotonic()
            self.settle_q_ref = q_now.copy()
            settle_mode = (
                "damping-only"
                if self.args.control_mode == "mit-impedance"
                else "position-hold"
            )
            self.get_logger().warning(
                f"{backend_name} settle ({settle_mode}); DO NOT TOUCH while "
                "verifying no self-motion before tracking"
            )
            return

        # CPV path: start backend after preflight (original logic)
        if backend_name == "cpv":
            enable_list = self._reader_enable_list()
            if enable_list is not None and not all(enable_list):
                self.get_logger().error(
                    "refusing CPV execute: joints are not already enabled; "
                    f"enable_list={enable_list}. Run move_to_aruco_start_pose.py first."
                )
                sdk_loop.shutdown()
                return
            robot = self.reader.take_robot()
            self.backend = SdkCpvVelocityBackend(
                dry_run=False,
                channel=self.args.channel,
                interface=self.args.interface,
                firmware=self.args.firmware,
                max_joint_vel=self.args.max_joint_vel,
                velocity_signs=self.args.cpv_signs,
                prepare_cpv=self.args.prepare_cpv,
                enable=self.args.enable_cpv,
                set_follower_mode=self.args.set_follower_mode,
                cpv_acc=self.args.cpv_acc,
                cpv_dcc=self.args.cpv_dcc,
                cpv_cv=self.args.cpv_cv,
                prepare_delay_s=self.args.cpv_prepare_delay_s,
                auto_set_follower_when_disabled=False,
                robot_factory=lambda: robot,
                robot_already_connected=True,
            )
        self.backend.connect()
        q_now = self._wait_backend_joint_feedback(self.last_preflight_q)
        if q_now is None:
            self.get_logger().error(
                f"{backend_name} backend started but joint feedback did not match preflight feedback"
            )
            sdk_loop.shutdown()
            return
        ok, reason = self.safety.check_start_posture(q_now)
        if not ok:
            self.get_logger().error(f"unsafe posture after {backend_name} start: {reason}")
            sdk_loop.shutdown()
            return
        if not self._check_start_pose(q_now):
            sdk_loop.shutdown()
            return
        if backend_name != "cpv":
            self.last_mit_q_cmd = q_now.copy()
        self.phase = "settle"
        self.settle_started = time.monotonic()
        self.settle_q_ref = q_now.copy()
        self.get_logger().warning(f"{backend_name} backend is active; verifying zero-motion hold before tracking")

    def _reader_enable_list(self) -> Optional[list[bool]]:
        robot = getattr(self.reader, "robot", None)
        if robot is None:
            return None
        method = getattr(robot, "get_joints_enable_status_list", None)
        if method is None:
            return None
        try:
            values = list(method())
        except Exception:
            return None
        if len(values) != 7:
            return None
        return [bool(value) for value in values]

    def _wait_backend_joint_feedback(self, reference_q: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if self.backend is None:
            return None

        deadline = time.monotonic() + float(self.args.backend_feedback_timeout)
        tolerance = np.radians(float(self.args.backend_feedback_tolerance_deg))
        last_q: Optional[np.ndarray] = None
        last_diff: Optional[float] = None

        while time.monotonic() < deadline:
            q_now = self.backend.get_joint_angles()
            if q_now is None:
                time.sleep(0.02)
                continue
            q_now = np.asarray(q_now, dtype=np.float64).ravel()
            if q_now.shape != (7,) or not np.all(np.isfinite(q_now)):
                raise RuntimeError(
                    "invalid backend joint feedback during startup: "
                    f"shape={q_now.shape}, values={q_now.tolist()}")
            last_q = q_now.copy()
            if self.args.execution_backend == "cpv":
                self.backend.send(self._hold_command(q_now, time.monotonic()))

            if reference_q is None:
                return q_now

            diff = float(np.max(np.abs(q_now - reference_q)))
            last_diff = diff
            if diff <= tolerance:
                return q_now
            time.sleep(0.02)

        reference_text = "None" if reference_q is None else _format(np.degrees(reference_q), 2)
        last_text = "None" if last_q is None else _format(np.degrees(last_q), 2)
        diff_text = "none" if last_diff is None else f"{np.degrees(last_diff):.2f}deg"
        self.get_logger().error(
            "backend joint feedback is not consistent with preflight feedback; "
            f"max_diff={diff_text} tolerance={self.args.backend_feedback_tolerance_deg:.2f}deg\n"
            f"  preflight_q_deg={reference_text}\n"
            f"  backend_q_deg={last_text}"
        )
        return None

    def _settle_step(self, q_now: np.ndarray, target, now: float) -> None:
        if self.backend is None or self.settle_started is None or self.settle_q_ref is None:
            self.get_logger().error("settle phase entered without backend/reference")
            sdk_loop.shutdown()
            return

        if self.args.control_mode == "mit-impedance":
            ok, reason = self.safety.check_start_posture(q_now)
            if not ok:
                self.get_logger().error(
                    "refusing impedance engagement during settle: "
                    f"unsafe limit margin: {reason}")
                sdk_loop.shutdown()
                return

        fake_servo_out = {
            "target_lost": False,
            "target_rejected": False,
            "error": np.zeros(3, dtype=np.float64),
        }
        decision = self.safety.evaluate(q_now, fake_servo_out, now)
        if decision.fatal or decision.hold:
            self.get_logger().error(f"settle safety stop: {decision.status} {decision.reason}")
            sdk_loop.shutdown()
            return

        drift = float(np.max(np.abs(q_now - self.settle_q_ref)))
        if drift > np.radians(self.args.settle_max_drift_deg):
            self.get_logger().error(
                f"{self.args.execution_backend} hold drift too large during settle: "
                f"{np.degrees(drift):.2f}deg > {self.args.settle_max_drift_deg:.2f}deg"
            )
            sdk_loop.shutdown()
            return

        if self.args.control_mode == "mit-impedance":
            settle_cmd = self._safe_damping_hold_command(q_now, now)
        else:
            settle_cmd = self._hold_command(
                self.settle_q_ref, now, prefer_last=False)
        self.backend.send(settle_cmd)
        self.count += 1
        elapsed = time.monotonic() - self.settle_started
        if now - self.last_log_time >= self.args.log_period:
            target_text = "None" if target is None else _format(target.marker_center_base, 4)
            self.get_logger().info(
                f"{self.args.execution_backend}_settle t={elapsed:.2f}/{self.args.settle_s:.2f}s "
                f"max_drift={np.degrees(drift):.3f}deg marker_base={target_text} "
                f"q_now_deg={_format(np.degrees(q_now), 2)}"
            )
            self.last_log_time = now

        if elapsed >= self.args.settle_s:
            self.phase = "tracking"
            self.get_logger().warning(
                f"{self.args.execution_backend} hold verified; starting low-speed ArUco follow"
            )

    def _servo_step(self, q_now: np.ndarray, target, now: float) -> tuple[dict, str]:
        if target is None:
            servo_out = self.servo.update(q_now=q_now, target_position=None, target_normal=None, stamp=now)
            return servo_out, "marker_lost_hold"

        if self.args.servo_mode == "pbvs":
            servo_out = self.servo.update(
                q_now=q_now,
                target_position=target.marker_center_base,
                target_normal=target.marker_normal_base,
                target_tangent=target.marker_tangent_base,
                stamp=now,
            )
        else:
            servo_out = self.servo.update(
                q_now=q_now,
                target_position=target.marker_center_base,
                target_normal=target.marker_normal_base,
                stamp=now,
            )
        if servo_out["target_rejected"]:
            return servo_out, "target_rejected_hold"
        if servo_out["target_lost"]:
            return servo_out, "marker_lost_hold"
        return servo_out, "tracking"

    def _hold_command(
        self,
        q_now: np.ndarray,
        now: float,
        prefer_last: bool = False,
    ) -> BackendCommand:
        if self.args.execution_backend == "cpv":
            return BackendCommand(
                q_cmd=q_now.copy(),
                qdot_cmd=np.zeros(7, dtype=np.float64),
                mode="velocity",
                stamp=now,
            )
        q_hold = q_now.copy()
        if prefer_last and self.last_mit_q_cmd is not None:
            q_hold = self.last_mit_q_cmd.copy()
        return BackendCommand(
            q_cmd=q_hold,
            qdot_cmd=np.zeros(7, dtype=np.float64),
            kp=self.kp,
            kd=self.kd,
            mode="mit",
            stamp=now,
        )

    def _tracking_hold_command(
        self,
        q_now: np.ndarray,
        now: float,
        prefer_last: bool = False,
    ) -> BackendCommand:
        """Select the non-tracking buffer command for the active controller."""
        if self.args.control_mode == "mit-impedance":
            # Before first engagement this avoids unexpectedly switching to a
            # stiff joint-position hold. After engagement the 200 Hz loop keeps
            # the frozen Cartesian target and ignores this safe buffer.
            return self._safe_damping_hold_command(q_now, now)
        return self._hold_command(q_now, now, prefer_last=prefer_last)

    def _release_impedance_follow_spring(self) -> None:
        """Drop only the visual joint-assist spring while holding a target.

        A marker-loss/hold decision intentionally leaves the Cartesian
        equilibrium frozen, so the 200 Hz impedance law can still return the
        arm after a push.  Any follow Kp left over from the previous moving
        frame would make that frozen hold feel like a rigid position servo.
        """
        if (
            self.backend is None
            or not self._imp_engaged
            or self._imp_cfg is None
        ):
            return
        current_follow_kp = getattr(self._imp_cfg, "follow_kp", None)
        current_target_twist = getattr(self._imp_cfg, "target_twist", None)
        release_follow = (
            current_follow_kp is not None and np.any(current_follow_kp)
        )
        release_twist = (
            current_target_twist is not None
            and np.any(np.abs(current_target_twist) > 1e-12)
        )
        if not release_follow and not release_twist:
            return
        self._imp_follow_kp_active = np.zeros(7, dtype=np.float64)
        self._imp_cfg.follow_kp = self._imp_follow_kp_active.copy()
        # A marker-loss/hold branch does not refresh the visual target. Clear
        # the desired-twist damper as well, otherwise the last moving command
        # would continue pushing the arm while the marker is absent.
        self._imp_cfg.target_twist = np.zeros(6, dtype=np.float64)
        try:
            self.backend.set_cartesian_impedance(self._imp_cfg)
        except Exception as exc:
            self._fail_closed_impedance(
                f"failed to release visual follow spring: {exc}"
            )

    def _handle_fatal_safety_stop(self, status: str, reason: str) -> None:
        """Immediately stop actuation on a fatal runtime safety decision."""
        message = f"{status}: {reason}"
        if self.args.control_mode == "mit-impedance":
            self._fail_closed_impedance(f"fatal safety stop: {message}")
        else:
            self.get_logger().error(message)

    def _safe_damping_hold_command(
        self,
        q_now: np.ndarray,
        now: float,
    ) -> BackendCommand:
        """Gravity compensation plus damping, with no position spring."""
        return BackendCommand(
            q_cmd=np.asarray(q_now, dtype=np.float64).ravel().copy(),
            qdot_cmd=np.zeros(7, dtype=np.float64),
            kp=np.zeros(7, dtype=np.float64),
            kd=self.zero_grav_kd.copy(),
            torque_ff=np.zeros(7, dtype=np.float64),
            mode="mit",
            stamp=now,
        )

    def _model_clamp(self, q: np.ndarray) -> np.ndarray:
        """Clamp joint angles to Nero joint limits."""
        from models.nero_model import NeroPinocchioModel
        return np.clip(q, NeroPinocchioModel.JOINT_LIMITS[:, 0], NeroPinocchioModel.JOINT_LIMITS[:, 1])

    def _imp_model_get(self):
        """Lazily build/cache the Pinocchio model for the impedance J^T mapping."""
        if self._imp_model is None:
            from models.nero_model import NeroPinocchioModel
            self._imp_model = NeroPinocchioModel()
        return self._imp_model

    def _build_impedance_config(self, target_tcp, R_target, q_rest):
        """Build the ImpedanceConfig handed to the 200 Hz in-loop impedance.

        Gains come from the CLI (Cartesian K/B, follow Kp, Coulomb,
        joint-weights, task-torque-limit, ...). The moving ``q_follow``
        reference is kept separate from the static ``q_rest`` posture. Both
        mappings retain firmware joint damping so the 7-DOF null-space
        direction is not left undamped.
        """
        from control_backends.force_ctrl.data_types import ImpedanceConfig
        # Firmware damping uses the controller's clean internal velocity and
        # covers both task and null-space motion.  The old plain-jtf branch
        # accidentally discarded --kd-scale, leaving one undamped DOF.
        kd_joint = (
            np.asarray(self.zero_grav_kd, dtype=np.float64)
            * float(self.args.kd_scale)
        )
        # Coulomb only when some joint is nonzero (all-zero = no-op -> None).
        coulomb = self.imp_coulomb if np.any(self.imp_coulomb > 0.0) else None
        return ImpedanceConfig(
            active=True,
            target_tcp=np.asarray(target_tcp, dtype=np.float64),
            R_target=np.asarray(R_target, dtype=np.float64),
            max_position_error=float(
                self.args.impedance_max_position_error
            ),
            max_orientation_error_rad=float(
                np.deg2rad(
                    self.args.impedance_max_orientation_error_deg
                )
            ),
            K=self.cart_kp,
            B=self.cart_kd,
            # The outer PBVS loop fills this with its constrained TCP twist on
            # each tracking refresh. Start with zero so engagement is a fixed-
            # target compliant hold, matching the standalone test.
            target_twist=np.zeros(6, dtype=np.float64),
            joint_damping=np.zeros(7, dtype=np.float64),   # finite-diff damping stays 0 (laggy/noisy)
            tau_limit_per_joint=self.imp_tau_limit_per_joint,
            q_rest=np.asarray(q_rest, dtype=np.float64),
            q_follow=(None if self._imp_follow_q is None
                      else np.asarray(self._imp_follow_q, dtype=np.float64)),
            follow_kp=getattr(
                self, "_imp_follow_kp_active", self.imp_follow_kp
            ),
            follow_max_error=float(getattr(self.args, "max_joint_vel", 0.10)),
            null_kq=0.0,
            dls_lambda=float(self.args.impedance_dls_lambda),
            use_dls_mapping=self.imp_use_dls,
            kp_joint=np.zeros(7, dtype=np.float64),
            kd_joint=kd_joint,
            joint_hold_kp=self.imp_joint_hold_kp,
            joint_hold_kd=self.imp_joint_hold_kd,
            vel_filter_tau=float(self.args.impedance_vel_filter_tau),
            qdot_trip_fast=self.imp_qdot_trip_fast,
            use_real_velocity=self.imp_use_real_velocity,
            velocity_signs=self.imp_velocity_signs,
            real_vel_filter_tau=self.imp_real_vel_filter_tau,
            joint_weights=self.imp_joint_weight,
            task_torque_limit=self.imp_task_torque_limit,
            coulomb=coulomb,
            coulomb_smooth_width=float(
                np.deg2rad(
                    max(0.0, float(
                        getattr(self.args, "impedance_coulomb_smooth_width_deg", 0.0)
                    ))
                )
            ),
        )

    def _build_impedance_command(self, q_now: np.ndarray, servo_out: dict, now: float) -> BackendCommand:
        """mit-impedance: engage the 200 Hz in-loop Cartesian impedance and feed
        it the PBVS nominal equilibrium trajectory.

        Engage at the CURRENT TCP (zero position+orientation error), then
        integrate the filtered PBVS world-frame twist into the cfg target. This
        is essential: a stiff
        Cartesian spring (K ~ 200 N/m) yanks the arm if it ever sees the full
        engage error (the arm settles ~0.27 m from the marker while the PBVS
        target is at desired_distance 0.15 m -> a ~0.12 m step -> large force ->
        fast yank -> RMS-qdot trip). Rate-limiting the target makes the impedance
        approach as controlled as the qdot-capped mit-position path.

        Loss-of-tracking freeze is automatic: this runs ONLY on the tracking
        branch of on_timer. The hold / marker-lost branches skip it, so the cfg
        target is NOT refreshed and the arm compliantly holds the LAST target
        (yields on push, returns to the CURRENT target on release).

        While impedance is active the 200 Hz loop ignores the buffered command
        path, so the returned BackendCommand is a benign no-op.
        """
        if self.backend is None or self._impedance_fallback_triggered:
            return self._safe_damping_hold_command(q_now, now)

        target_tcp = servo_out.get("target_tcp")
        target_rotation = servo_out.get("target_rotation")
        tcp_twist = servo_out.get("tcp_twist")
        if target_tcp is None:
            return self._safe_damping_hold_command(q_now, now)
        if self.args.impedance_visual_orientation and target_rotation is None:
            return self._safe_damping_hold_command(q_now, now)

        target_tcp = np.asarray(target_tcp, dtype=np.float64).ravel()
        if target_rotation is not None:
            target_rotation = np.asarray(target_rotation, dtype=np.float64).reshape(3, 3)
        if tcp_twist is not None:
            tcp_twist = np.asarray(tcp_twist, dtype=np.float64).ravel()
            if tcp_twist.shape != (6,) or not np.all(np.isfinite(tcp_twist)):
                raise ValueError(
                    "tcp_twist must have finite shape (6,), "
                    f"got {tcp_twist.shape}"
                )
        model = self._imp_model_get()
        nominal_dt = max(0.0, float(self.params.dt))
        tcp_now, R_now = model.fk(q_now)
        tcp_now = np.asarray(tcp_now, dtype=np.float64).ravel()
        R_now = np.asarray(R_now, dtype=np.float64).reshape(3, 3)

        if not self._imp_engaged:
            ok, reason = self.safety.check_start_posture(q_now)
            if not ok:
                self._impedance_fallback_triggered = True
                self._impedance_terminal_fault = True
                self._impedance_fallback_reason = (
                    f"refusing impedance engagement: {reason}")
                self.get_logger().error(self._impedance_fallback_reason)
                return self._safe_damping_hold_command(q_now, now)
            # Engage at the current pose: zero error so the spring does not yank
            # the arm on cycle 1. The target then ramps toward the visual marker
            # (rate-limited below).
            self._imp_smoothed_tcp = tcp_now.copy()
            self._imp_smoothed_R = R_now.copy()
            self._imp_fixed_R = R_now.copy()
            self._imp_target_update_time = float(now)
            self._imp_track_start_q = q_now.copy()
            self._imp_follow_q = q_now.copy()
            # Do not add a position spring during the engage handoff. It is
            # enabled only when the visual joint trajectory actually moves.
            self._imp_follow_kp_active = np.zeros(7, dtype=np.float64)
            self._imp_cfg = self._build_impedance_config(
                self._imp_smoothed_tcp, self._imp_smoothed_R, q_now)
            try:
                self.backend.set_cartesian_impedance(self._imp_cfg)
            except Exception as exc:
                self._fail_closed_impedance(
                    f"initial impedance engagement failed: {exc}")
                return self._safe_damping_hold_command(q_now, now)
            debug = self.backend.get_debug_state() or {}
            self._imp_engage_cycle_baseline = int(
                debug.get("impedance_cycle_count", 0)
            )
            self._imp_engage_monotonic = time.monotonic()
            self._imp_engaged = True
            self.get_logger().info(
                "in-loop Cartesian impedance ENGAGED (200 Hz law) at the current "
                "pose; rate-limiting and leashing the visual equilibrium target.")
        else:
            # Respect the declared m/s and rad/s limits even if callbacks run
            # faster than nominal. A late callback or marker reacquisition gets
            # at most one nominal step, so no catch-up jump is injected.
            if self._imp_target_update_time is None:
                dt = nominal_dt
            else:
                dt = min(
                    nominal_dt,
                    max(0.0, float(now) - self._imp_target_update_time),
                )
            self._imp_target_update_time = float(now)
            # PBVS already applies position PID, pose filtering, singularity
            # scaling, twist smoothing, and joint-velocity limiting. Follow that same
            # trajectory so impedance and the validated mit-position path share
            # one visual-servo command instead of independently chasing the raw
            # pose at a constant speed. Position-only diagnostics do not expose
            # tcp_twist and retain the legacy pose-chase fallback.
            previous_tcp = self._imp_smoothed_tcp.copy()
            # Reuse the old mit-position trajectory: PBVS already limits and
            # smooths qdot, so integrating it gives a continuous visual
            # reference.  The 200 Hz loop tracks this with low firmware Kp/Kd
            # while retaining Cartesian J^T impedance for compliant push/release.
            qdot = np.asarray(
                servo_out.get("qdot", np.zeros(7)), dtype=np.float64
            ).ravel()
            if qdot.shape != (7,) or not np.all(np.isfinite(qdot)):
                raise ValueError(
                    f"qdot must have finite shape (7,), got {qdot.shape}"
                )
            if self._imp_follow_q is None:
                self._imp_follow_q = q_now.copy()
            self._imp_follow_q = self._model_clamp(
                self._imp_follow_q + qdot * dt
            )
            # Do not let a saturated outer qdot accumulate an ever-growing
            # visual joint reference while the compliant inner loop is still
            # catching up.  The backend applies the same bound when forming its
            # firmware command, but clamping the stored reference here also
            # keeps Coulomb direction and diagnostics tied to a finite lead.
            follow_bound = max(0.0, float(self.args.max_joint_vel))
            if follow_bound > 0.0:
                follow_error = self._imp_follow_q - q_now
                self._imp_follow_q = self._model_clamp(
                    q_now + np.clip(follow_error, -follow_bound, follow_bound)
                )
            # At a settled visual target, release the firmware position spring
            # and leave only Cartesian impedance, matching the fixed-target
            # compliant feel. Scale it continuously while the PBVS trajectory
            # is moving so tracking remains smooth without a Kp discontinuity.
            min_qdot = max(0.0, float(self.imp_follow_min_qdot))
            if min_qdot <= 1e-12:
                follow_scale = np.ones(7, dtype=np.float64)
            else:
                follow_scale = np.clip(
                    np.abs(qdot) / min_qdot, 0.0, 1.0
                )
            self._imp_follow_kp_active = self.imp_follow_kp * follow_scale
            # A real operator displacement can make q run ahead of the
            # nominal PBVS reference.  In that case the low firmware position
            # spring must yield; otherwise it masks the Cartesian compliance
            # and the arm feels like a position servo.  A lag in the desired
            # direction is retained to overcome static friction during normal
            # visual motion.
            release_error = max(
                0.0, float(getattr(self, "imp_follow_release_error", 0.03))
            )
            if release_error > 0.0:
                follow_error = self._imp_follow_q - q_now
                moving = np.abs(qdot) > 1e-6
                commanded_side = qdot * follow_error
                pushed_ahead = (
                    moving
                    & (commanded_side < 0.0)
                    & (np.abs(follow_error) >= release_error)
                )
                self._imp_follow_kp_active[pushed_ahead] = 0.0
            if tcp_twist is not None:
                candidate_tcp = integrate_position_target_from_twist(
                    previous_tcp,
                    tcp_twist[:3],
                    dt,
                    self.args.impedance_target_linear_vel,
                )
            else:
                max_step = max(
                    0.0, float(self.args.impedance_target_linear_vel)
                ) * dt
                diff = target_tcp - previous_tcp
                step = float(np.linalg.norm(diff))
                if step > max_step and step > 1e-9:
                    candidate_tcp = previous_tcp + diff * (max_step / step)
                else:
                    candidate_tcp = target_tcp.copy()
            self._imp_smoothed_tcp = apply_position_target_anti_windup(
                previous_tcp,
                candidate_tcp,
                tcp_now,
                self.args.impedance_max_position_error,
            )
            if self.args.impedance_visual_orientation:
                from scipy.spatial.transform import Rotation as Rot

                previous_rotation = self._imp_smoothed_R.copy()
                if tcp_twist is not None:
                    candidate_rotation = integrate_orientation_target_from_twist(
                        previous_rotation,
                        tcp_twist[3:],
                        dt,
                        self.args.impedance_target_angular_vel,
                    )
                else:
                    rv = Rot.from_matrix(
                        previous_rotation.T @ target_rotation
                    ).as_rotvec()
                    ang = float(np.linalg.norm(rv))
                    max_ang = max(
                        0.0, float(self.args.impedance_target_angular_vel)
                    ) * dt
                    if ang > max_ang and ang > 1e-9:
                        candidate_rotation = (
                            previous_rotation
                            @ Rot.from_rotvec((max_ang / ang) * rv).as_matrix()
                        )
                    else:
                        candidate_rotation = target_rotation.copy()
                self._imp_smoothed_R = apply_orientation_target_anti_windup(
                    previous_rotation,
                    candidate_rotation,
                    R_now,
                    np.deg2rad(
                        self.args.impedance_max_orientation_error_deg
                    ),
                )
            else:
                # Keep the engagement orientation as the nominal equilibrium.
                # The 200 Hz law clamps its effective SO(3) spring error, so a
                # push cannot ratchet this reference or exceed the torque leash.
                self._imp_smoothed_R = self._imp_fixed_R.copy()

            cfg = self._imp_cfg
            cfg.target_tcp = self._imp_smoothed_tcp.copy()
            cfg.R_target = self._imp_smoothed_R
            cfg.q_follow = self._imp_follow_q.copy()
            # Feed the same filtered, joint-limited TCP velocity used to
            # advance the equilibrium into the Cartesian damper. This supplies
            # continuous motion assistance without reinstating a joint
            # position spring. When visual orientation is disabled, retain the
            # engagement orientation and remove angular velocity feed-forward.
            if (
                tcp_twist is None
                or not getattr(
                    self.args, "impedance_target_twist_feedforward", True
                )
            ):
                cfg.target_twist = np.zeros(6, dtype=np.float64)
            else:
                cfg.target_twist = constrain_impedance_target_twist(
                    tcp_twist,
                    self.args.impedance_target_linear_vel,
                    self.args.impedance_target_angular_vel,
                    include_orientation=self.args.impedance_visual_orientation,
                )
            # The config is owned by the backend after the first engage. Keep
            # the per-cycle release/ramp decision in sync with that owned
            # copy; otherwise the diagnostic Kp would change while the 200 Hz
            # MIT command stayed at the engage-time value.
            cfg.follow_kp = self._imp_follow_kp_active.copy()
            try:
                self.backend.set_cartesian_impedance(cfg)
            except Exception as exc:
                self._fail_closed_impedance(
                    f"impedance target refresh failed: {exc}")
                return self._safe_damping_hold_command(q_now, now)

        # Benign buffered command: ignored by the 200 Hz loop while impedance is
        # active. Pure zeros so nothing competes with the in-loop law.
        return BackendCommand(
            q_cmd=q_now.copy(),
            qdot_cmd=np.zeros(7, dtype=np.float64),
            kp=np.zeros(7, dtype=np.float64),
            kd=np.zeros(7, dtype=np.float64),
            torque_ff=np.zeros(7, dtype=np.float64),
            mode="mit",
            stamp=now,
        )

    def _check_impedance_instability(self, q_now: np.ndarray, now: float) -> None:
        """Latch into a gravity-compensated damping hold on instability.

        Triggers on (any of):
          - instantaneous joint velocity above hard cap (single cycle)
          - RMS joint velocity over a sliding window above threshold
          - optional POSITION-domain trip: any joint drifts more than a
            positive max_excursion from its tracking-start pose
          - joint-limit proximity closer than safe_margin (always active)
        Latches once. The 200 Hz loop and gravity compensation remain active,
        but no automatic transition to stiff visual position tracking is made.
        """
        if self._impedance_fallback_triggered:
            return
        self._qdot_ringbuf.append((now, q_now.copy()))
        window = float(self.args.impedance_fallback_window)
        cutoff = now - window
        while self._qdot_ringbuf and self._qdot_ringbuf[0][0] < cutoff:
            self._qdot_ringbuf.pop(0)

        triggered = False
        reason = ""
        # Instantaneous trip: any joint moving faster than the hard cap.
        if len(self._qdot_ringbuf) >= 2:
            t_prev, q_prev = self._qdot_ringbuf[-2]
            dt = max(now - t_prev, 1e-6)
            qdot_inst = np.abs(q_now - q_prev) / dt
            if float(np.max(qdot_inst)) > float(self.args.impedance_instant_qdot_cap):
                triggered = True
                reason = f"instant qdot {float(np.max(qdot_inst)):.2f} > cap"
        # RMS trip over window.
        if len(self._qdot_ringbuf) >= 3:
            qs = [q for _, q in self._qdot_ringbuf]
            ts = [t for t, _ in self._qdot_ringbuf]
            dqs = []
            for i in range(1, len(qs)):
                dt = max(ts[i] - ts[i - 1], 1e-6)
                dqs.append(np.abs(qs[i] - qs[i - 1]) / dt)
            if dqs:
                rms = float(np.sqrt(np.mean(np.square(np.max(np.vstack(dqs), axis=1)))))
                if rms > float(self.args.impedance_fallback_oscillation):
                    triggered = True
                    reason = f"RMS qdot {rms:.2f} > threshold"
        # POSITION-domain trips (catch slow sub-threshold drift the velocity
        # detector is blind to).
        max_excursion_limit = float(self.args.impedance_max_excursion)
        if (
            self._imp_track_start_q is not None
            and max_excursion_limit > 0.0
        ):
            max_exc = float(
                np.max(np.abs(q_now - self._imp_track_start_q))
            )
            if max_exc > max_excursion_limit:
                triggered = True
                reason = f"excursion {max_exc:.3f} > {max_excursion_limit}"

        if self._imp_track_start_q is not None and not triggered:
            # A start-relative excursion bound is not meaningful for a visual
            # controller whose legitimate target moves. Hard joint limits are
            # still monitored regardless of this optional experimental bound.
            from models.nero_model import NeroPinocchioModel
            lim = NeroPinocchioModel.JOINT_LIMITS
            margin = float(self.args.impedance_safe_margin)
            for i in range(7):
                if q_now[i] < lim[i, 0] + margin or q_now[i] > lim[i, 1] - margin:
                    triggered = True
                    reason = f"joint{i+1} within {margin:.2f} of limit"
                    break

        if triggered:
            self._impedance_fallback_triggered = True
            self._impedance_fallback_reason = reason
            try:
                if self.backend is not None and self._imp_engaged:
                    self.backend.disable_cartesian_impedance()
            except Exception as exc:
                self._fail_closed_impedance(
                    f"impedance disable failed after {reason}: {exc}")
                return
            self._imp_engaged = False
            try:
                if self.backend is not None:
                    self.backend.send(
                        self._safe_damping_hold_command(q_now, now)
                    )
            except Exception as exc:
                self._fail_closed_impedance(
                    f"safe damping hold failed after {reason}: {exc}")
                return
            self.get_logger().warning(
                f"impedance instability detected ({reason}); "
                f"latched gravity+damping safe hold "
                f"(kp=0, kd=ZERO_GRAV_KD; no automatic mode switch)"
            )

    def _fail_closed_impedance(self, reason: str) -> None:
        """Escalate a failed graceful fallback to an electronic e-stop."""
        self._impedance_fallback_triggered = True
        self._impedance_terminal_fault = True
        self._impedance_fallback_reason = str(reason)
        self._imp_engaged = False
        self.get_logger().error(
            f"{reason}; graceful impedance fallback is not confirmed, "
            "issuing electronic emergency stop.")
        try:
            if self.backend is not None:
                self.backend.emergency_stop()
        except Exception as exc:
            self.get_logger().error(f"electronic emergency stop call failed: {exc}")

    def _reset_progress_guard(self) -> None:
        self.progress_best_error = None
        self.progress_last_improved_time = None
        self.progress_last_marker = None
        self.progress_last_target = None

    def _check_progress_guard(
        self,
        status: str,
        target,
        servo_out: dict,
        cmd: BackendCommand,
        now: float,
    ) -> str:
        if not self.args.enable_progress_guard:
            return ""
        if status != "tracking" or target is None:
            self._reset_progress_guard()
            return ""

        cmd_qdot = np.asarray(cmd.qdot_cmd, dtype=np.float64).ravel()
        if np.max(np.abs(cmd_qdot)) < self.args.progress_guard_min_qdot:
            self._reset_progress_guard()
            return ""

        target_tcp = servo_out.get("target_tcp")
        if target_tcp is None:
            self._reset_progress_guard()
            return ""
        marker = np.asarray(target.marker_center_base, dtype=np.float64).ravel()
        target_tcp = np.asarray(target_tcp, dtype=np.float64).ravel()

        if self.progress_last_marker is not None:
            marker_delta = float(np.linalg.norm(marker - self.progress_last_marker))
            target_delta = float(np.linalg.norm(target_tcp - self.progress_last_target))
            if (
                marker_delta > self.args.progress_guard_marker_reset
                or target_delta > self.args.progress_guard_target_reset
            ):
                self._reset_progress_guard()

        self.progress_last_marker = marker.copy()
        self.progress_last_target = target_tcp.copy()

        error_norm = float(np.linalg.norm(servo_out["error"]))
        if error_norm < self.args.progress_guard_min_error:
            self._reset_progress_guard()
            return ""

        if self.progress_best_error is None:
            self.progress_best_error = error_norm
            self.progress_last_improved_time = now
            return ""

        if error_norm < self.progress_best_error - self.args.progress_guard_epsilon:
            self.progress_best_error = error_norm
            self.progress_last_improved_time = now
            return ""

        assert self.progress_last_improved_time is not None
        no_progress_s = now - self.progress_last_improved_time
        if no_progress_s >= self.args.progress_guard_s:
            return (
                f"error not converging for {no_progress_s:.2f}s: "
                f"err={error_norm:.4f}m best={self.progress_best_error:.4f}m"
            )
        return ""

    def _log_waiting(self, now: float, text: str) -> None:
        if now - self.last_log_time >= self.args.log_period:
            self.get_logger().info(text)
            self.last_log_time = now

    def _log_status(
        self,
        status: str,
        q_now: np.ndarray,
        target,
        servo_out: dict,
        now: float,
        reason: str,
        cmd: BackendCommand,
    ) -> None:
        q_step = servo_out["q_cmd"] - q_now
        detected_age = None if self.last_detect_time is None else now - self.last_detect_time
        age_text = "none" if detected_age is None else f"{detected_age:.3f}s"
        cmd_qdot = np.asarray(cmd.qdot_cmd, dtype=np.float64).ravel()
        servo_qdot = np.asarray(servo_out["qdot"], dtype=np.float64).ravel()
        target_text = "None"
        normal_text = "None"
        camera_text = "None"
        dict_text = "None"
        if target is not None:
            target_text = _format(target.marker_center_base, 4)
            normal_text = _format(target.marker_normal_base, 4)
            camera_text = _format(target.T_camera_marker[:3, 3], 4)
            dict_text = target.dictionary_name
        reason_text = "" if not reason else f" reason={reason}"
        impedance_text = ""
        impedance_detail_text = ""
        if (
            self.args.control_mode == "mit-impedance"
            and self._imp_engaged
            and self._imp_cfg is not None
            and self._imp_smoothed_tcp is not None
        ):
            reference_position_error = float(
                np.linalg.norm(
                    np.asarray(self._imp_smoothed_tcp, dtype=np.float64)
                    - np.asarray(servo_out["tcp_now"], dtype=np.float64)
                )
            )
            position_limit = self._imp_cfg.max_position_error
            effective_position_error = (
                reference_position_error
                if position_limit is None
                else min(reference_position_error, float(position_limit))
            )
            impedance_text = (
                f" imp_ref_pos_err={reference_position_error:.4f}m"
                f" imp_eff_pos_err={effective_position_error:.4f}m"
            )

            tcp_rotation = servo_out.get("tcp_rotation")
            if self._imp_smoothed_R is not None and tcp_rotation is not None:
                current_rotation = np.asarray(
                    tcp_rotation, dtype=np.float64
                ).reshape(3, 3)
                reference_rotation = np.asarray(
                    self._imp_smoothed_R, dtype=np.float64
                ).reshape(3, 3)
                rotation_delta = current_rotation.T @ reference_rotation
                cos_angle = float(
                    np.clip((np.trace(rotation_delta) - 1.0) * 0.5, -1.0, 1.0)
                )
                reference_rotation_error = float(np.arccos(cos_angle))
                orientation_limit = self._imp_cfg.max_orientation_error_rad
                effective_rotation_error = (
                    reference_rotation_error
                    if orientation_limit is None
                    else min(reference_rotation_error, float(orientation_limit))
                )
                impedance_text += (
                    f" imp_ref_rot_err={np.degrees(reference_rotation_error):.2f}deg"
                    f" imp_eff_rot_err={np.degrees(effective_rotation_error):.2f}deg"
                )

            debug = self.backend.get_debug_state() if self.backend is not None else {}
            tau_imp = np.asarray(
                debug.get("impedance_torque", np.zeros(7)), dtype=np.float64
            ).ravel()
            tau_gravity = np.asarray(
                debug.get("gravity_torque", np.zeros(7)), dtype=np.float64
            ).ravel()
            tau_total = np.asarray(
                debug.get("total_torque", np.zeros(7)), dtype=np.float64
            ).ravel()
            tau_feedback = np.asarray(
                debug.get("feedback_torque", np.zeros(7)), dtype=np.float64
            ).ravel()
            impedance_text += (
                f" inner_imp_active={debug.get('impedance_active')}"
                f" inner_path={debug.get('control_path')}"
                f" inner_cycles={debug.get('impedance_cycle_count')}"
                f" tau_imp_max={np.max(np.abs(tau_imp)):.3f}Nm"
                f" tau_g_max={np.max(np.abs(tau_gravity)):.3f}Nm"
                f" tau_total_max={np.max(np.abs(tau_total)):.3f}Nm"
            )
            target_twist = np.asarray(
                getattr(self._imp_cfg, "target_twist", np.zeros(6)),
                dtype=np.float64,
            ).ravel()
            if target_twist.shape == (6,):
                impedance_text += (
                    f" target_twist_max={np.max(np.abs(target_twist)):.4f}"
                )
            if self._imp_follow_q is not None:
                follow_error = np.asarray(self._imp_follow_q) - np.asarray(q_now)
                impedance_text += (
                    f" follow_q_err_max={np.max(np.abs(follow_error)):.4f}rad"
                    f" follow_kp_max={np.max(self._imp_follow_kp_active):.2f}"
                )
            impedance_detail_text = (
                f"\n  tau_imp={_format(tau_imp, 3)}"
                f" tau_g={_format(tau_gravity, 3)}"
                f" tau_total={_format(tau_total, 3)}"
                f" tau_feedback={_format(tau_feedback, 3)}"
            )

        self.get_logger().info(
            f"{status} count={self.count} detect_age={age_text} "
            f"err={np.linalg.norm(servo_out['error']):.4f}m "
            f"rot_err={np.degrees(np.linalg.norm(servo_out.get('orientation_error', np.zeros(3)))):.2f}deg "
            f"servo_qdot_max={np.max(np.abs(servo_qdot)):.5f}rad/s "
            f"cmd_qdot_max={np.max(np.abs(cmd_qdot)):.5f}rad/s "
            f"cmd_mode={cmd.mode} step_max={np.max(np.abs(q_step)):.5f}rad"
            f"{impedance_text}{reason_text}\n"
            f"  dict={dict_text} marker_camera={camera_text} marker_base={target_text} normal={normal_text}\n"
            f"  tcp_now={_format(servo_out['tcp_now'], 4)} target_tcp={_format_or_none(servo_out['target_tcp'])}\n"
            f"  q_now_deg={_format(np.degrees(q_now), 2)} "
            f"q_cmd_deg={_format(np.degrees(servo_out['q_cmd']), 2)}"
            f"{impedance_detail_text}"
        )

    def _update_display(self, status: str, target, servo_out: Optional[dict]) -> None:
        if not self.display_enabled or self.last_image is None:
            return
        try:
            display = draw_camera_overlay(
                image=self.last_image,
                target=target,
                servo_out=servo_out,
                status=status,
                camera_matrix=self.camera_matrix,
                dist_coeffs=self.dist_coeffs,
                marker_size=self.args.marker_size,
            )
            if not self.display_initialized:
                cv2.namedWindow(self.display_window, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(self.display_window, 960, 540)
                self.display_initialized = True
            cv2.imshow(self.display_window, display)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                self.get_logger().info("display requested shutdown")
                sdk_loop.shutdown()
        except Exception as exc:
            self.display_enabled = False
            self.get_logger().warning(f"camera display disabled: {exc}")

    def close(self) -> None:
        errors = []
        if self.reader is not None:
            try:
                self.reader.disconnect()
            except Exception as exc:
                errors.append(f"feedback reader disconnect failed: {exc}")
        if self.backend is not None:
            # Drop out of in-loop impedance gracefully (zero-G hold buffer) so
            # the loop stop does not leave a stale impedance torque on the arm.
            if self._imp_engaged:
                try:
                    self.backend.disable_cartesian_impedance()
                except Exception as exc:
                    errors.append(
                        f"impedance disable failed during close: {exc}")
                    try:
                        self.backend.emergency_stop()
                    except Exception as estop_exc:
                        errors.append(
                            f"electronic emergency stop during close failed: "
                            f"{estop_exc}")
            try:
                self.backend.disconnect()
                self.backend = None
            except Exception as exc:
                errors.append(f"backend disconnect failed: {exc}")
                try:
                    self.backend.emergency_stop()
                except Exception as estop_exc:
                    errors.append(
                        "electronic emergency stop after disconnect failure "
                        f"failed: {estop_exc}")
        if self.display_initialized:
            try:
                cv2.destroyWindow(self.display_window)
            except Exception as exc:
                errors.append(f"display cleanup failed: {exc}")
        if errors:
            message = "; ".join(errors)
            self.get_logger().critical(f"cleanup incomplete: {message}")
            raise RuntimeError(message)


def _vec7_floats6(text: str) -> np.ndarray:
    """Parse six floats (Cartesian stiffness/damping). Accepts comma- OR
    space-separated, e.g. '50,50,50,10,10,10' or '50 50 50 10 10 10'."""
    import re
    values = [float(item) for item in re.split(r"[,\s]+", text.strip()) if item]
    if len(values) != 6:
        raise argparse.ArgumentTypeError(f"expected six floats, got {len(values)}")
    return np.asarray(values, dtype=np.float64)


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="Required to enable real robot control.")
    parser.add_argument("--image-topic", default="/camera/camera/color/image_raw")
    parser.add_argument("--camera-info-topic", default="/camera/camera/color/camera_info")
    parser.add_argument(
        "--image-timeout-s",
        type=float,
        default=0.20,
        help="Maximum camera-frame age used for detection. Stale frames freeze "
             "the last impedance target (default: 0.20 s).",
    )
    parser.add_argument("--handeye", default=DEFAULT_HAND_EYE)
    parser.add_argument(
        "--handeye-convention",
        choices=("link7_cam", "cam_link7", "tcp_cam", "cam_tcp", "mount_cam", "cam_mount"),
        default="link7_cam",
    )
    parser.add_argument(
        "--handeye-mount-frame",
        choices=("feedback_link7", "tcp_offset"),
        default="feedback_link7",
    )
    parser.add_argument("--offset-base", nargs=3, type=float, default=[0.02, -0.025, 0.0])
    parser.add_argument("--offset-marker", nargs=3, type=float, default=[0.0, 0.0, 0.0])
    parser.add_argument("--aruco-dict", default="DICT_ARUCO_ORIGINAL")
    parser.add_argument("--marker-id", type=int, default=582)
    parser.add_argument("--marker-size", type=float, default=0.0677)

    parser.add_argument("--channel", default="can0")
    parser.add_argument("--interface", default="socketcan")
    parser.add_argument(
        "--cpv-signs",
        type=_vec7,
        default=np.ones(7, dtype=np.float64),
        help="Per-joint sign multiplier before SDK move_cpv_vel(), comma-separated 7 values.",
    )
    parser.add_argument(
        "--no-prepare-cpv",
        dest="prepare_cpv",
        action="store_false",
        help="Skip follower/enable/MOVE_CPV/profile preparation before CPV execution.",
    )
    parser.set_defaults(prepare_cpv=True)
    parser.add_argument("--no-enable-cpv", dest="enable_cpv", action="store_false")
    parser.set_defaults(enable_cpv=True)
    parser.add_argument(
        "--set-follower-mode",
        dest="set_follower_mode",
        action="store_true",
        help="Diagnostic only: can cause large jumps if the arm is already enabled in CPV.",
    )
    parser.add_argument("--no-set-follower-mode", dest="set_follower_mode", action="store_false")
    parser.set_defaults(set_follower_mode=False)
    parser.add_argument("--cpv-acc", type=float, default=0.50)
    parser.add_argument("--cpv-dcc", type=float, default=0.50)
    parser.add_argument("--cpv-cv", type=float, default=0.50)
    parser.add_argument("--cpv-prepare-delay-s", type=float, default=0.20)
    parser.add_argument(
        "--firmware",
        default="v111",
        help="Nero firmware driver. MIT force_ctrl uses v111; CPV diagnostics require v112.",
    )
    parser.add_argument(
        "--execution-backend",
        choices=("mit-position", "cpv"),
        default="mit-position",
        help="mit-position executes Panthera PBVS qdot as MIT+gravity q_cmd steps; cpv is diagnostic only.",
    )
    parser.add_argument(
        "--allow-mit-position-execute",
        action="store_true",
        help="Deprecated compatibility flag; MIT position execute is now the default main path.",
    )
    parser.add_argument(
        "--control-mode",
        choices=("mit-position", "mit-impedance"),
        default="mit-position",
        help="mit-position = stiff position-follow (default). "
             "mit-impedance = compliant Cartesian impedance in the 200 Hz MIT loop "
             "(only effective with --execution-backend mit-position).",
    )
    parser.add_argument(
        "--cartesian-kp",
        type=_vec7_floats6,
        default=VALIDATED_CARTESIAN_KP.copy(),
        help="Cartesian stiffness [tx ty tz rx ry rz] (N/m, Nm/rad), 6 floats.",
    )
    parser.add_argument(
        "--cartesian-kd",
        type=_vec7_floats6,
        default=VALIDATED_CARTESIAN_KD.copy(),
        help="Cartesian damping (6 floats); default is the validated profile.",
    )
    parser.add_argument(
        "--impedance-torque-limit",
        type=float,
        default=float(VALIDATED_TASK_TORQUE_LIMIT),
        help="Compatibility task-torque limit. The effective in-loop task limit "
             "is the smaller positive value of this and "
             "--impedance-task-torque-limit; <=0 disables this bound.",
    )
    parser.add_argument("--impedance-fallback-window", type=float, default=1.0)
    parser.add_argument("--impedance-fallback-oscillation", type=float, default=0.3)
    parser.add_argument("--impedance-instant-qdot-cap", type=float, default=1.0)
    parser.add_argument(
        "--impedance-max-excursion",
        type=float,
        default=0.0,
        help="Optional start-relative joint excursion trip (rad). <=0 disables "
             "it, which is the visual-tracking default because legitimate "
             "tracking may move far from the engagement posture. Hard joint "
             "limits and qdot/feedback watchdogs remain active.",
    )
    parser.add_argument(
        "--impedance-safe-margin",
        type=float,
        default=0.30,
        help="Latch if any joint comes within this many rad of a hard limit.",
    )
    # ---- in-loop Cartesian impedance law (mit-impedance) ----
    parser.add_argument(
        "--impedance-law",
        choices=("plain-jtf", "panthera-real"),
        default="plain-jtf",
        help="plain-jtf = plain J^T@F + Cartesian Bx via REAL velocity (matches the "
             "standalone task mapping) plus firmware joint damping. "
             "panthera-real = diagnostic DLS force map plus the same damping. "
             "Default plain-jtf preserves the power-conjugate force mapping.",
    )
    parser.add_argument(
        "--impedance-coulomb",
        type=_vec7,
        default=np.zeros(7, dtype=np.float64),
        help="Per-joint Coulomb friction feed-forward (Nm), 7 comma-sep. Adds "
             "coulomb*sign(q_follow-q) during visual follow (q_rest for fixed targets), "
             "with 0.5 deg deadband. Default all-0 (off). "
             "Standalone baseline: 0,0.4,0,0.4,0,0,0.",
    )
    parser.add_argument(
        "--impedance-coulomb-smooth-width-deg",
        type=float,
        default=0.0,
        help="Smooth Coulomb transition width in degrees. >0 uses tanh(error/width) "
             "continuously during motion and rest; 0 keeps the standalone hard-sign law.",
    )
    parser.add_argument(
        "--impedance-joint-weights",
        type=_vec7,
        default=np.ones(7, dtype=np.float64),
        help="Per-joint task-torque weighting (7 comma-sep). Safe production "
             "default is all ones; non-uniform post-J^T weights do not preserve "
             "the standard Cartesian passivity argument.",
    )
    parser.add_argument(
        "--impedance-task-torque-limit",
        type=float,
        default=float(VALIDATED_TASK_TORQUE_LIMIT),
        help="Spring/task torque clip before gravity (Nm). Effective limit is the "
             "smaller positive value of this and --impedance-torque-limit; pass "
             "<=0 to disable this bound.",
    )
    parser.add_argument(
        "--impedance-tau-limit-per-joint",
        type=_vec7,
        default=None,
        help="Per-joint total impedance torque clip (7 comma-sep). "
             "Default = --torque-limit on all joints.",
    )
    parser.add_argument("--impedance-dls-lambda", type=float, default=0.05,
                        help="DLS regularization (panthera-real law).")
    parser.add_argument("--impedance-target-alpha", type=float, default=0.50,
                        help="[SUPERSEDED] formerly a low-pass on the visual target. "
                             "PBVS mode now integrates its joint-limited tcp_twist with "
                             "impedance_target_linear_vel / impedance_target_angular_vel "
                             "as additional caps. Kept for CLI "
                             "backward-compat; ignored.")
    parser.add_argument(
        "--impedance-target-linear-vel",
        type=float,
        default=0.02,
        help="Additional cap on PBVS equilibrium-point advance speed in "
             "impedance mode (m/s).",
    )
    parser.add_argument(
        "--impedance-max-position-error",
        type=float,
        default=0.015,
        help="Maximum distance between the impedance equilibrium point and the "
             "current TCP (m).",
    )
    parser.add_argument("--impedance-target-angular-vel", type=float, default=0.30,
                        help="Max advance rate of the cfg orientation target toward the "
                             "visual marker (rad/s). MUST stay slow: a fast orientation target "
                             "outruns the arm into the wrist singular (joint5~0) and the stiff "
                             "rotational spring slams joint5/6 to catch up -> runaway. Default "
                             "0.30 rad/s (17 deg/s, same order as the stable mit-position "
                             "max_joint_vel=0.15 rad/s). Do NOT use max_angular_vel (2.20) here.")
    parser.add_argument(
        "--impedance-target-twist-feedforward",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use the filtered, joint-limited PBVS TCP twist as the desired "
             "velocity in the Cartesian damping term. This gives smooth "
             "motion without reintroducing a joint position spring; disable "
             "only for a pure fixed-target impedance A/B test.",
    )
    parser.add_argument(
        "--impedance-visual-orientation",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Follow the marker orientation in impedance mode. Disabled by "
             "default, preserving the TCP orientation at engagement.",
    )
    parser.add_argument(
        "--impedance-max-orientation-error-deg",
        type=float,
        default=5.0,
        help="Leash the impedance equilibrium orientation within this angle of "
             "the current TCP, for both fixed and visual orientation modes.",
    )
    parser.add_argument("--impedance-vel-filter-tau", type=float, default=0.008,
                        help="Finite-diff velocity low-pass tau (s); FALLBACK only.")
    parser.add_argument("--impedance-real-vel-filter-tau", type=float, default=0.0,
                        help="Low-pass tau (s) on REAL motor velocity. Default 0 "
                             "avoids adding phase lag to firmware-filtered feedback.")
    parser.add_argument("--impedance-qdot-trip-fast", type=float, default=2.0,
                        help="IN-LOOP sustained-qdot runaway trip (rad/s). Fires only "
                             "after 5 fresh feedback frames over threshold, so a compliant push-bounce "
                             "(joint7 can spike briefly) does NOT kill the loop.")
    parser.add_argument(
        "--impedance-joint-hold-kp", type=_vec7,
        default=np.zeros(7, dtype=np.float64),
        help="Optional direct joint-position hold gains (7 comma-sep, Nm/rad). "
             "Default all zero for a fully compliant visual task.",
    )
    parser.add_argument(
        "--impedance-joint-hold-kd", type=_vec7,
        default=None,
        help="Optional direct joint-hold damping gains (7 comma-sep). If "
             "omitted, joints selected by --impedance-joint-hold-kp inherit "
             "the normal firmware kd_joint values.",
    )
    parser.add_argument(
        "--impedance-follow-kp", type=_vec7,
        default=np.array([2.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0]),
        help="Low firmware Kp for the hybrid PBVS joint reference. The default "
             "only supplies static-friction assistance; Cartesian impedance "
             "remains the primary tracking/compliance law. Set all zero for "
             "pure torque.",
    )
    parser.add_argument(
        "--impedance-follow-min-qdot", type=float, default=0.08,
        help="PBVS joint-speed scale (rad/s) at which hybrid follow_kp reaches "
             "full strength. Near-zero visual speed releases the firmware "
             "position spring and leaves Cartesian impedance compliant.",
    )
    parser.add_argument(
        "--impedance-follow-release-error", type=float, default=0.02,
        help="Release hybrid follow_kp when the measured joint position runs "
             "ahead of the PBVS reference by this amount (rad). This lets an "
             "operator push against the Cartesian spring during visual motion.",
    )
    parser.add_argument(
        "--use-real-velocity", dest="use_real_velocity",
        action=argparse.BooleanOptionalAction, default=True,
        help="Use the REAL motor velocity (parser-direct, the gold key) for the "
             "Cartesian damper -B*J*qdot (default ON). "
             "--no-use-real-velocity falls back to finite-diff "
             "(B must then stay tiny; divergence risk at the wrist).",
    )
    parser.add_argument(
        "--impedance-velocity-signs", type=_vec7,
        default=np.array([-1.0, -1.0, -1.0, -1.0, -1.0, 1.0, -1.0]),
        help="Per-joint sign for the real motor velocity (joint6 correct, rest negated).",
    )
    parser.add_argument("--duration", type=float, default=10.0, help="Seconds to run; 0 means forever.")
    parser.add_argument("--log-period", type=float, default=0.5)
    parser.add_argument("--control-hz", type=float, default=200.0)
    parser.add_argument(
        "--torque-limit",
        type=float,
        default=float(VALIDATED_TOTAL_TORQUE_LIMIT),
    )
    parser.add_argument("--gravity-factor", type=float, default=1.0)
    parser.add_argument(
        "--joint-factors",
        type=_vec7,
        default=VALIDATED_GRAVITY_FACTORS.copy(),
        help="Per-joint gravity compensation factors from the validated "
             "Cartesian impedance profile.",
    )
    parser.add_argument("--verbose-backend", action="store_true")
    parser.add_argument("--no-display", dest="display", action="store_false")
    parser.set_defaults(display=True)

    parser.add_argument(
        "--servo-mode",
        choices=("position", "pbvs"),
        default="pbvs",
        help="pbvs matches Panthera 19 ArUco velocity-Jacobian control; position is diagnostic fallback.",
    )
    parser.add_argument("--servo-hz", type=float, default=30.0)
    parser.add_argument("--desired-distance", type=float, default=0.15)
    parser.add_argument("--linear-gain", type=float, default=5.0)
    parser.add_argument("--linear-integral-gain", type=float, default=1.2)
    parser.add_argument("--linear-derivative-gain", type=float, default=0.08)
    parser.add_argument("--angular-gain", type=float, default=2.6)
    parser.add_argument(
        "--pbvs-orientation-scale",
        type=float,
        default=1.0,
        help="Scale the outer PBVS visual angular velocity before its cap. "
             "Use a value below 1 when orientation should not consume the "
             "whole joint-velocity budget.",
    )
    parser.add_argument("--max-linear-vel", type=float, default=0.35)
    parser.add_argument("--max-angular-vel", type=float, default=2.20)
    parser.add_argument("--damping", type=float, default=0.02)
    parser.add_argument("--max-joint-vel", type=float, default=0.15)
    parser.add_argument(
        "--max-joint-accel",
        type=float,
        default=0.0,
        help="Optional PBVS joint-velocity slew limit (rad/s^2); <=0 disables it.",
    )
    parser.add_argument("--max-joint-step", type=float, default=0.006)
    parser.add_argument("--target-filter-alpha", type=float, default=0.34)
    parser.add_argument(
        "--target-blend-alpha",
        type=float,
        default=0.82,
        help="SO(3)/position blend for accepted PBVS target updates. Lower "
             "values suppress marker pose jumps in impedance mode.",
    )
    parser.add_argument("--normal-filter-alpha", type=float, default=0.24)
    parser.add_argument("--qdot-filter-alpha", type=float, default=0.35)
    parser.add_argument("--twist-smooth-alpha", type=float, default=0.72)
    parser.add_argument(
        "--orientation-filter-alpha",
        type=float,
        default=0.25,
        help="SO(3) blend alpha for accepted marker orientation updates.",
    )
    parser.add_argument("--target-timeout-s", type=float, default=0.60)
    parser.add_argument("--max-target-jump", type=float, default=0.08)
    parser.add_argument("--position-deadband", type=float, default=0.004)
    parser.add_argument("--orientation-deadband-deg", type=float, default=1.5)
    parser.add_argument("--kp-scale", type=float, default=1.0)
    parser.add_argument("--kd-scale", type=float, default=1.0)

    parser.add_argument("--startup-limit-margin", type=float, default=0.08)
    parser.add_argument("--runtime-limit-margin", type=float, default=0.04)
    parser.add_argument(
        "--max-tracking-error",
        type=float,
        default=0.0,
        help="Hold if position error stays above this value in meters; <=0 disables distance-based hold.",
    )
    parser.add_argument("--max-tracking-error-s", type=float, default=1.0)
    parser.add_argument("--enable-progress-guard", dest="enable_progress_guard", action="store_true")
    parser.add_argument("--disable-progress-guard", dest="enable_progress_guard", action="store_false")
    parser.set_defaults(enable_progress_guard=False)
    parser.add_argument("--progress-guard-s", type=float, default=2.0)
    parser.add_argument("--progress-guard-epsilon", type=float, default=0.004)
    parser.add_argument("--progress-guard-min-error", type=float, default=0.025)
    parser.add_argument("--progress-guard-min-qdot", type=float, default=0.01)
    parser.add_argument("--progress-guard-marker-reset", type=float, default=0.03)
    parser.add_argument("--progress-guard-target-reset", type=float, default=0.04)
    parser.add_argument("--preflight-stable-samples", type=int, default=15)
    parser.add_argument("--preflight-target-tolerance", type=float, default=0.01)
    parser.add_argument("--backend-feedback-timeout", type=float, default=2.0)
    parser.add_argument("--backend-feedback-tolerance-deg", type=float, default=2.0)
    parser.add_argument("--settle-s", type=float, default=2.0)
    parser.add_argument("--settle-max-drift-deg", type=float, default=2.0)
    parser.add_argument(
        "--start-q-deg",
        type=parse_deg7,
        default=DEFAULT_ARUCO_START_Q_DEG.copy(),
        help="Required start posture in degrees, comma-separated joint1..joint7.",
    )
    parser.add_argument("--start-tolerance-deg", type=float, default=5.0)
    parser.add_argument(
        "--no-require-start-pose",
        dest="require_start_pose",
        action="store_false",
        help="Allow starting from any posture that passes joint-limit safety checks.",
    )
    parser.set_defaults(require_start_pose=True)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO if args.verbose_backend else logging.WARNING,
        format="%(levelname)s: %(message)s",
    )
    if not args.execute:
        raise SystemExit("Refusing to run real control without --execute. Use the dry-run script first.")
    if args.execution_backend == "cpv" and args.firmware != "v112":
        raise SystemExit("CPV velocity diagnostics require --firmware v112.")
    if not Path(args.handeye).expanduser().exists():
        raise FileNotFoundError(args.handeye)

    sdk_loop.init()
    node = None
    try:
        node = ArucoCameraServoExecuteNode(args)
        sdk_loop.spin(node)
    except KeyboardInterrupt:
        if node is not None:
            node.get_logger().info("interrupted")
    finally:
        cleanup_error = None
        if node is not None:
            try:
                node.close()
            except Exception as exc:
                cleanup_error = exc
            try:
                node.destroy_node()
            except Exception as exc:
                cleanup_error = cleanup_error or exc
        if sdk_loop.ok():
            sdk_loop.shutdown()
        if cleanup_error is not None:
            raise RuntimeError(
                f"visual-servo cleanup was not confirmed: {cleanup_error}"
            ) from cleanup_error


if __name__ == "__main__":
    main()
