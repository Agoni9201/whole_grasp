"""Panthera-style ArUco PBVS pose servo.

This implements the control law used by Panthera
``19_aruco_marker_tracking_velocity_jacobian.py``:

    marker pose -> safe TCP pose
    -> PBVS TCP twist (position PID + orientation axis-angle)
    -> damped 6x7 TCP Jacobian pseudo-inverse
    -> joint velocity command

The primary algorithmic output is ``qdot``, matching Panthera ``Joint_Vel``.
For Nero's default MIT + gravity backend, ``qdot`` is integrated into a small
``q_cmd`` step and sent as continuous position targets because CPV/follower
mode has shown unsafe jumps on this setup.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from models.nero_model import NeroPinocchioModel

logger = logging.getLogger(__name__)


@dataclass
class PBVSParams:
    servo_hz: float = 30.0
    desired_distance: float = 0.15

    linear_gain: float = 5.0
    linear_integral_gain: float = 1.2
    linear_derivative_gain: float = 0.08
    angular_gain: float = 2.6
    # Keep the pose-servo orientation term independently switchable.  The
    # impedance controller can intentionally hold the engagement orientation;
    # in that mode the outer PBVS loop must not spend joint velocity budget on
    # a rotation that the inner loop will refuse to follow.
    enable_orientation: bool = True
    # Scale the visual angular command before the angular-speed cap.  This is
    # useful when the marker normal is noisy or the requested pose differs
    # substantially from the current TCP pose: translation should keep moving
    # instead of letting the orientation term consume all seven joint-speed
    # budget.
    orientation_twist_scale: float = 1.0
    # Optional outer-loop orientation leash, in radians.  Keeping this equal to
    # the inner impedance orientation leash prevents the PBVS reference from
    # running far ahead of the torque law.
    orientation_error_limit_rad: float = 0.0

    max_linear_vel: float = 0.10
    max_angular_vel: float = 2.20
    max_joint_vel: float = 0.15
    # Optional slew limit for the joint-velocity command.  A value <= 0 keeps
    # the legacy behavior; visual execution can enable this to prevent a
    # direction change from jumping straight from +max to -max.
    max_joint_accel: float = 0.0
    max_joint_step: float = 0.006

    damping: float = 0.02
    manipulability_threshold: float = 0.006

    marker_center_alpha: float = 0.34
    marker_normal_alpha: float = 0.24
    marker_tangent_alpha: float = 0.24
    target_jitter_translation: float = 0.004
    target_jitter_normal_angle_deg: float = 3.0
    target_blend_alpha: float = 0.82
    orientation_blend_alpha: float = 0.25
    fast_target_delta: float = 0.018

    position_deadband: float = 0.004
    orientation_deadband_rad: float = np.deg2rad(1.5)
    target_timeout_s: float = 0.60
    max_target_jump: float = 0.08

    tcp_twist_smooth_alpha: float = 0.72
    position_integral_clamp: float = 0.05
    position_derivative_alpha: float = 0.35

    @property
    def dt(self) -> float:
        return 1.0 / self.servo_hz


def _normalize(
    vector: npt.ArrayLike,
    fallback: npt.ArrayLike,
) -> npt.NDArray[np.float64]:
    vector = np.asarray(vector, dtype=np.float64).ravel()
    norm = np.linalg.norm(vector)
    if norm < 1e-8:
        return np.asarray(fallback, dtype=np.float64).ravel().copy()
    return vector / norm


def _clip_norm(
    vector: npt.NDArray[np.float64],
    max_norm: float,
) -> npt.NDArray[np.float64]:
    vector = np.asarray(vector, dtype=np.float64).copy()
    norm = np.linalg.norm(vector)
    if norm > max_norm > 0.0:
        vector *= max_norm / norm
    return vector


def _project_to_plane(
    vector: npt.ArrayLike,
    plane_normal: npt.ArrayLike,
) -> npt.NDArray[np.float64]:
    vector = np.asarray(vector, dtype=np.float64).ravel()
    plane_normal = _normalize(plane_normal, np.array([1.0, 0.0, 0.0]))
    return vector - np.dot(vector, plane_normal) * plane_normal


def _rotation_to_rotvec(rotation: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    rotation = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    cos_theta = float(np.clip((np.trace(rotation) - 1.0) * 0.5, -1.0, 1.0))
    theta = float(np.arccos(cos_theta))
    if theta < 1e-8:
        return np.zeros(3, dtype=np.float64)

    if np.pi - theta < 1e-5:
        diag = np.diag(rotation)
        largest = int(np.argmax(diag))
        axis = np.zeros(3, dtype=np.float64)
        if largest == 0:
            axis[0] = 0.5 * np.sqrt(max(0.0, 1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]))
            denom = max(4.0 * axis[0], 1e-8)
            axis[1] = (rotation[0, 1] + rotation[1, 0]) / denom
            axis[2] = (rotation[0, 2] + rotation[2, 0]) / denom
        elif largest == 1:
            axis[1] = 0.5 * np.sqrt(max(0.0, 1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]))
            denom = max(4.0 * axis[1], 1e-8)
            axis[0] = (rotation[0, 1] + rotation[1, 0]) / denom
            axis[2] = (rotation[1, 2] + rotation[2, 1]) / denom
        else:
            axis[2] = 0.5 * np.sqrt(max(0.0, 1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]))
            denom = max(4.0 * axis[2], 1e-8)
            axis[0] = (rotation[0, 2] + rotation[2, 0]) / denom
            axis[1] = (rotation[1, 2] + rotation[2, 1]) / denom
        axis = _normalize(axis, np.array([1.0, 0.0, 0.0], dtype=np.float64))
        return axis * theta

    axis = np.array(
        [
            rotation[2, 1] - rotation[1, 2],
            rotation[0, 2] - rotation[2, 0],
            rotation[1, 0] - rotation[0, 1],
        ],
        dtype=np.float64,
    )
    axis /= max(2.0 * np.sin(theta), 1e-8)
    axis = _normalize(axis, np.array([1.0, 0.0, 0.0], dtype=np.float64))
    return axis * theta


def _rotvec_to_rotation(rotvec: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Return the SO(3) exponential map without requiring SciPy."""
    vector = np.asarray(rotvec, dtype=np.float64).ravel()
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise ValueError("rotvec must be a finite vector with shape (3,)")
    theta = float(np.linalg.norm(vector))
    if theta < 1e-10:
        return np.eye(3, dtype=np.float64)
    axis = vector / theta
    skew = np.array(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ],
        dtype=np.float64,
    )
    return (
        np.eye(3, dtype=np.float64)
        + np.sin(theta) * skew
        + (1.0 - np.cos(theta)) * (skew @ skew)
    )


def _interpolate_rotation(
    previous: npt.ArrayLike,
    target: npt.ArrayLike,
    alpha: float,
) -> npt.NDArray[np.float64]:
    """Geodesically blend two rotation matrices on SO(3)."""
    previous_matrix = np.asarray(previous, dtype=np.float64).reshape(3, 3)
    target_matrix = np.asarray(target, dtype=np.float64).reshape(3, 3)
    weight = float(np.clip(alpha, 0.0, 1.0))
    delta = previous_matrix.T @ target_matrix
    delta_rotvec = _rotation_to_rotvec(delta)
    return previous_matrix @ _rotvec_to_rotation(weight * delta_rotvec)


def _orientation_error_axis_angle(
    rotation_desired: npt.NDArray[np.float64],
    rotation_current: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    rotation_error = rotation_current.T @ rotation_desired
    return rotation_current @ _rotation_to_rotvec(rotation_error)


def _damped_pinv(
    jacobian: npt.NDArray[np.float64],
    damping: float,
    manipulability_threshold: float,
) -> tuple[npt.NDArray[np.float64], float]:
    rows, _ = jacobian.shape
    jjt = jacobian @ jacobian.T
    determinant = np.linalg.det(jjt)
    manipulability = np.sqrt(max(0.0, determinant))
    twist_scale = 1.0
    effective_damping = damping

    if manipulability < manipulability_threshold:
        effective_damping = damping * (
            1.0 + (manipulability_threshold - manipulability) * 120.0
        )
        twist_scale = max(0.15, manipulability / manipulability_threshold)

    damping_matrix = (effective_damping ** 2) * np.eye(rows, dtype=np.float64)
    try:
        return jacobian.T @ np.linalg.inv(jjt + damping_matrix), twist_scale
    except np.linalg.LinAlgError:
        stronger = (effective_damping * 10.0) ** 2 * np.eye(rows, dtype=np.float64)
        return jacobian.T @ np.linalg.inv(jjt + stronger), twist_scale


def _scale_joint_velocity_to_limit(
    qdot: npt.NDArray[np.float64],
    max_joint_vel: float | npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    qdot = np.asarray(qdot, dtype=np.float64).copy()
    limits = np.asarray(max_joint_vel, dtype=np.float64)
    if limits.ndim == 0:
        limits = np.full_like(qdot, float(limits))
    safe_limits = np.maximum(limits, 1e-6)
    ratio = float(np.max(np.abs(qdot) / safe_limits))
    if ratio > 1.0:
        qdot /= ratio
    return qdot


class PBVSPoseServo:
    """Panthera-style 6D pose servo for ArUco tracking."""

    def __init__(
        self,
        params: Optional[PBVSParams] = None,
        model: Optional["NeroPinocchioModel"] = None,
    ) -> None:
        self.params = params or PBVSParams()
        if model is None:
            from models.nero_model import NeroPinocchioModel
            model = NeroPinocchioModel()
        self._model = model

        self._filtered_marker_center: Optional[npt.NDArray[np.float64]] = None
        self._filtered_marker_normal: Optional[npt.NDArray[np.float64]] = None
        self._filtered_marker_tangent: Optional[npt.NDArray[np.float64]] = None
        self._accepted_target_position: Optional[npt.NDArray[np.float64]] = None
        self._accepted_target_rotation: Optional[npt.NDArray[np.float64]] = None
        self._accepted_marker_normal: Optional[npt.NDArray[np.float64]] = None
        self._smoothed_target_position: Optional[npt.NDArray[np.float64]] = None
        self._smoothed_target_rotation: Optional[npt.NDArray[np.float64]] = None

        self._actual_tcp_twist = np.zeros(6, dtype=np.float64)
        self._position_error_integral = np.zeros(3, dtype=np.float64)
        self._previous_position_error = np.zeros(3, dtype=np.float64)
        self._filtered_position_error_derivative = np.zeros(3, dtype=np.float64)
        self._prev_q_cmd: Optional[npt.NDArray[np.float64]] = None
        self._previous_qdot: Optional[npt.NDArray[np.float64]] = None
        self._last_target_stamp: Optional[float] = None

    def update(
        self,
        q_now: npt.NDArray[np.float64],
        target_position: Optional[npt.NDArray[np.float64]],
        target_normal: Optional[npt.NDArray[np.float64]] = None,
        target_tangent: Optional[npt.NDArray[np.float64]] = None,
        target_orientation: Optional[npt.NDArray[np.float64]] = None,
        stamp: Optional[float] = None,
    ) -> dict:
        now = time.time() if stamp is None else float(stamp)
        q_now = np.asarray(q_now, dtype=np.float64).ravel()
        if len(q_now) != 7:
            raise ValueError(f"Expected 7 joint angles, got {len(q_now)}")

        tcp_now, tcp_rotation = self._model.fk(q_now)

        if target_position is None:
            self._reset_target_filters()
            return self._hold(q_now, tcp_now, now, target_lost=True)

        marker_center = np.asarray(target_position, dtype=np.float64).ravel()
        if len(marker_center) != 3:
            raise ValueError("target_position must have shape (3,)")

        if target_normal is None:
            if target_orientation is None:
                raise ValueError("target_normal or target_orientation is required")
            target_rotation = np.asarray(target_orientation, dtype=np.float64).reshape(3, 3)
            marker_normal = _normalize(target_rotation[:, 2], tcp_rotation[:, 0])
            marker_tangent = _normalize(target_rotation[:, 1], tcp_rotation[:, 1])
        else:
            marker_normal = _normalize(target_normal, np.array([1.0, 0.0, 0.0]))
            if np.dot(tcp_now - marker_center, marker_normal) < 0.0:
                marker_normal = -marker_normal
            fallback_tangent = tcp_rotation[:, 1]
            marker_tangent = _normalize(
                target_tangent if target_tangent is not None else fallback_tangent,
                fallback_tangent,
            )
            marker_tangent = _normalize(
                _project_to_plane(marker_tangent, marker_normal),
                fallback_tangent,
            )
            target_rotation = None

        marker_center, marker_normal, marker_tangent = self._filter_marker_measurement(
            marker_center,
            marker_normal,
            marker_tangent,
        )

        raw_target_position = marker_center + marker_normal * self.params.desired_distance
        if target_rotation is None:
            raw_target_rotation = self._build_target_tcp_rotation(
                marker_normal,
                marker_tangent,
                tcp_rotation,
            )
        else:
            raw_target_rotation = np.asarray(target_rotation, dtype=np.float64).reshape(3, 3)

        if (
            self._accepted_target_position is not None
            and np.linalg.norm(raw_target_position - self._accepted_target_position)
            > self.params.max_target_jump
        ):
            logger.warning(
                "Rejecting target jump: %.4f m",
                np.linalg.norm(raw_target_position - self._accepted_target_position),
            )
            return self._hold(q_now, tcp_now, now, target_rejected=True)

        self._last_target_stamp = now
        target_position_base, target_rotation_base = self._accept_and_smooth_target(
            raw_target_position,
            raw_target_rotation,
            marker_normal,
        )

        tcp_twist, position_error, orientation_error = self._compute_pbvs_tcp_twist(
            target_position_base,
            tcp_now,
            target_rotation_base,
            tcp_rotation,
        )

        J = self._model.jacobian(q_now)
        J_pinv, twist_scale = _damped_pinv(
            J,
            self.params.damping,
            self.params.manipulability_threshold,
        )
        target_twist_scaled = tcp_twist * twist_scale

        orientation_settled = (
            not self.params.enable_orientation
            or np.linalg.norm(orientation_error)
            < self.params.orientation_deadband_rad
        )
        if (
            np.linalg.norm(position_error) < self.params.position_deadband
            and orientation_settled
        ):
            self._actual_tcp_twist[:] = 0.0
        else:
            alpha = self.params.tcp_twist_smooth_alpha
            self._actual_tcp_twist = self._actual_tcp_twist + (
                target_twist_scaled - self._actual_tcp_twist
            ) * alpha

        qdot = J_pinv @ self._actual_tcp_twist
        qdot = _scale_joint_velocity_to_limit(qdot, self.params.max_joint_vel)
        qdot = self._limit_joint_velocity_slew(qdot)
        # This is the Cartesian motion represented by the same joint-limited
        # qdot that the validated MIT-position path integrates.
        commanded_tcp_twist = J @ qdot

        q_cmd = self._integrate_for_logging(q_now, qdot)
        return {
            "q_cmd": q_cmd,
            "qdot": qdot,
            "error": position_error,
            "orientation_error": orientation_error,
            "target_tcp": target_position_base,
            "target_rotation": target_rotation_base,
            "tcp_now": tcp_now,
            "tcp_rotation": tcp_rotation,
            "marker_center": marker_center,
            "marker_normal": marker_normal,
            "marker_tangent": marker_tangent,
            "v_base": tcp_twist[:3],
            "tcp_twist": commanded_tcp_twist.copy(),
            "twist_scale": twist_scale,
            "target_lost": False,
            "target_rejected": False,
            "stamp": now,
        }

    def _compute_pbvs_tcp_twist(
        self,
        target_position: npt.NDArray[np.float64],
        current_position: npt.NDArray[np.float64],
        target_rotation: npt.NDArray[np.float64],
        current_rotation: npt.NDArray[np.float64],
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        position_error = target_position - current_position
        orientation_error = _orientation_error_axis_angle(target_rotation, current_rotation)
        control_orientation_error = orientation_error.copy()
        orientation_limit = max(
            0.0,
            float(self.params.orientation_error_limit_rad),
        )
        if orientation_limit > 0.0:
            control_orientation_error = _clip_norm(
                control_orientation_error,
                orientation_limit,
            )

        if np.linalg.norm(position_error) >= self.params.position_deadband:
            proportional = self.params.linear_gain * position_error
            raw_derivative = (position_error - self._previous_position_error) / max(
                self.params.dt,
                1e-6,
            )
            alpha = self.params.position_derivative_alpha
            self._filtered_position_error_derivative = (
                alpha * raw_derivative
                + (1.0 - alpha) * self._filtered_position_error_derivative
            )
            derivative = (
                self.params.linear_derivative_gain
                * self._filtered_position_error_derivative
            )

            candidate_integral = self._position_error_integral + position_error * self.params.dt
            candidate_integral = np.clip(
                candidate_integral,
                -self.params.position_integral_clamp,
                self.params.position_integral_clamp,
            )
            integral = self.params.linear_integral_gain * candidate_integral
            linear_velocity = proportional + derivative + integral
            if np.linalg.norm(linear_velocity) > self.params.max_linear_vel:
                linear_velocity = _clip_norm(linear_velocity, self.params.max_linear_vel)
                # Do not retain an old integral bias while the Cartesian
                # command is saturated.  In the compliant inner loop that
                # bias otherwise survives a target change and appears as a
                # slow one-direction lift/creep.
                self._position_error_integral *= 0.95
            else:
                self._position_error_integral = candidate_integral
        else:
            self._position_error_integral *= 0.5
            self._filtered_position_error_derivative *= 0.5
            linear_velocity = np.zeros(3, dtype=np.float64)

        self._previous_position_error = position_error.copy()

        if (
            not self.params.enable_orientation
            or np.linalg.norm(control_orientation_error)
            < self.params.orientation_deadband_rad
        ):
            angular_velocity = np.zeros(3, dtype=np.float64)
        else:
            angular_velocity = (
                self.params.angular_gain
                * max(0.0, float(self.params.orientation_twist_scale))
                * control_orientation_error
            )
            angular_velocity = _clip_norm(angular_velocity, self.params.max_angular_vel)

        return np.concatenate([linear_velocity, angular_velocity]), position_error, orientation_error

    def _filter_marker_measurement(
        self,
        marker_center: npt.NDArray[np.float64],
        marker_normal: npt.NDArray[np.float64],
        marker_tangent: npt.NDArray[np.float64],
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        if self._filtered_marker_center is None:
            self._filtered_marker_center = marker_center.copy()
        else:
            alpha = self.params.marker_center_alpha
            self._filtered_marker_center = (
                alpha * marker_center
                + (1.0 - alpha) * self._filtered_marker_center
            )

        if self._filtered_marker_normal is None:
            self._filtered_marker_normal = marker_normal.copy()
        else:
            if np.dot(self._filtered_marker_normal, marker_normal) < 0.0:
                marker_normal = -marker_normal
            alpha = self.params.marker_normal_alpha
            self._filtered_marker_normal = (
                alpha * marker_normal
                + (1.0 - alpha) * self._filtered_marker_normal
            )
            self._filtered_marker_normal = _normalize(
                self._filtered_marker_normal,
                marker_normal,
            )

        marker_tangent = _normalize(
            _project_to_plane(marker_tangent, self._filtered_marker_normal),
            np.array([0.0, 1.0, 0.0]),
        )
        if self._filtered_marker_tangent is None:
            self._filtered_marker_tangent = marker_tangent.copy()
        else:
            if np.dot(self._filtered_marker_tangent, marker_tangent) < 0.0:
                marker_tangent = -marker_tangent
            alpha = self.params.marker_tangent_alpha
            self._filtered_marker_tangent = (
                alpha * marker_tangent
                + (1.0 - alpha) * self._filtered_marker_tangent
            )
            self._filtered_marker_tangent = _normalize(
                _project_to_plane(self._filtered_marker_tangent, self._filtered_marker_normal),
                marker_tangent,
            )

        return (
            self._filtered_marker_center.copy(),
            self._filtered_marker_normal.copy(),
            self._filtered_marker_tangent.copy(),
        )

    def _build_target_tcp_rotation(
        self,
        marker_normal: npt.NDArray[np.float64],
        marker_tangent: npt.NDArray[np.float64],
        fallback_rotation: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        x_axis = _normalize(-marker_normal, fallback_rotation[:, 0])
        tangent_projected = _project_to_plane(marker_tangent, x_axis)
        if np.linalg.norm(tangent_projected) < 1e-6:
            tangent_projected = _project_to_plane(fallback_rotation[:, 1], x_axis)
        if np.linalg.norm(tangent_projected) < 1e-6:
            tangent_projected = _project_to_plane(np.array([0.0, 1.0, 0.0]), x_axis)

        y_axis = _normalize(tangent_projected, fallback_rotation[:, 1])
        z_axis = _normalize(np.cross(x_axis, y_axis), fallback_rotation[:, 2])
        y_axis = _normalize(np.cross(z_axis, x_axis), y_axis)
        return np.column_stack((x_axis, y_axis, z_axis))

    def _accept_and_smooth_target(
        self,
        target_position: npt.NDArray[np.float64],
        target_rotation: npt.NDArray[np.float64],
        marker_normal: npt.NDArray[np.float64],
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        if (
            self._accepted_target_position is None
            or self._accepted_target_rotation is None
            or self._accepted_marker_normal is None
        ):
            self._accepted_target_position = target_position.copy()
            self._accepted_target_rotation = target_rotation.copy()
            self._accepted_marker_normal = marker_normal.copy()
        elif self._should_accept_target_update(
            self._accepted_target_position,
            self._accepted_target_rotation,
            target_position,
            target_rotation,
        ):
            self._accepted_target_position = target_position.copy()
            self._accepted_target_rotation = target_rotation.copy()
            self._accepted_marker_normal = marker_normal.copy()

        if self._smoothed_target_position is None:
            self._smoothed_target_position = self._accepted_target_position.copy()
        else:
            target_delta = float(
                np.linalg.norm(
                    self._accepted_target_position - self._smoothed_target_position
                )
            )
            alpha = np.clip(
                self.params.target_blend_alpha
                + target_delta / max(self.params.fast_target_delta, 1e-6) * 0.38,
                self.params.target_blend_alpha,
                0.90,
            )
            self._smoothed_target_position = (
                alpha * self._accepted_target_position
                + (1.0 - alpha) * self._smoothed_target_position
            )

        if self._smoothed_target_rotation is None:
            self._smoothed_target_rotation = self._accepted_target_rotation.copy()
        else:
            # The accepted rotation changes in discrete steps (the jitter gate
            # intentionally rejects tiny ArUco updates).  Blend on SO(3) so
            # those steps cannot inject a burst of angular qdot.
            alpha_rot = float(np.clip(self.params.orientation_blend_alpha, 0.0, 1.0))
            self._smoothed_target_rotation = _interpolate_rotation(
                self._smoothed_target_rotation,
                self._accepted_target_rotation,
                alpha_rot,
            )

        assert self._accepted_target_rotation is not None
        assert self._smoothed_target_rotation is not None
        return self._smoothed_target_position.copy(), self._smoothed_target_rotation.copy()

    def _limit_joint_velocity_slew(
        self,
        qdot: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Limit per-cycle joint-velocity changes when requested."""
        qdot = np.asarray(qdot, dtype=np.float64).ravel().copy()
        limit = float(self.params.max_joint_accel)
        previous = self._previous_qdot
        if limit > 0.0 and previous is not None:
            max_delta = limit * max(float(self.params.dt), 1e-6)
            qdot = previous + np.clip(qdot - previous, -max_delta, max_delta)
            qdot = _scale_joint_velocity_to_limit(qdot, self.params.max_joint_vel)
        self._previous_qdot = qdot.copy()
        return qdot

    def _should_accept_target_update(
        self,
        previous_target_position: npt.NDArray[np.float64],
        previous_target_rotation: npt.NDArray[np.float64],
        target_position: npt.NDArray[np.float64],
        target_rotation: npt.NDArray[np.float64],
    ) -> bool:
        translation_delta = float(np.linalg.norm(target_position - previous_target_position))
        rotation_delta = (
            np.asarray(previous_target_rotation, dtype=np.float64).reshape(3, 3).T
            @ np.asarray(target_rotation, dtype=np.float64).reshape(3, 3)
        )
        orientation_delta_deg = float(
            np.degrees(np.linalg.norm(_rotation_to_rotvec(rotation_delta)))
        )
        return (
            translation_delta >= self.params.target_jitter_translation
            or orientation_delta_deg >= self.params.target_jitter_normal_angle_deg
        )

    def _integrate_for_logging(
        self,
        q_now: npt.NDArray[np.float64],
        qdot: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        q_cmd = q_now + qdot * self.params.dt
        step = np.clip(
            q_cmd - q_now,
            -self.params.max_joint_step,
            self.params.max_joint_step,
        )
        q_cmd = q_now + step
        q_limited = self._model.clamp_to_limits(q_cmd)
        step = np.clip(
            q_limited - q_now,
            -self.params.max_joint_step,
            self.params.max_joint_step,
        )
        q_cmd = q_now + step
        self._prev_q_cmd = q_cmd.copy()
        return q_cmd

    def _reset_target_filters(self) -> None:
        self._filtered_marker_center = None
        self._filtered_marker_normal = None
        self._filtered_marker_tangent = None
        self._accepted_target_position = None
        self._accepted_target_rotation = None
        self._accepted_marker_normal = None
        self._smoothed_target_position = None
        self._smoothed_target_rotation = None
        self._actual_tcp_twist[:] = 0.0
        self._position_error_integral[:] = 0.0
        self._filtered_position_error_derivative[:] = 0.0
        self._previous_position_error[:] = 0.0
        self._previous_qdot = None

    def _hold(
        self,
        q_now: npt.NDArray[np.float64],
        tcp_now: npt.NDArray[np.float64],
        stamp: float,
        target_lost: bool = False,
        target_rejected: bool = False,
    ) -> dict:
        self._actual_tcp_twist[:] = 0.0
        self._prev_q_cmd = q_now.copy()
        self._previous_qdot = None
        return {
            "q_cmd": q_now.copy(),
            "qdot": np.zeros(7, dtype=np.float64),
            "error": np.zeros(3, dtype=np.float64),
            "orientation_error": np.zeros(3, dtype=np.float64),
            "target_tcp": None,
            "target_rotation": None,
            "tcp_now": tcp_now,
            "tcp_rotation": None,
            "marker_center": None,
            "marker_normal": None,
            "marker_tangent": None,
            "v_base": np.zeros(3, dtype=np.float64),
            "tcp_twist": np.zeros(6, dtype=np.float64),
            "twist_scale": 1.0,
            "target_lost": target_lost,
            "target_rejected": target_rejected,
            "stamp": stamp,
        }
