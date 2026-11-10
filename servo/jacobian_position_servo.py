"""ArUco position-only Jacobian servo.

Core control law (position only, no orientation):

    T_tcp = fk_tcp(q_now)
    tcp_now = T_tcp.translation

    T_marker = target pose in base frame
    marker_center = T_marker.translation
    marker_normal = T_marker.rotation[:, 2]

    target_tcp = marker_center + marker_normal * desired_distance
    error_base = target_tcp - tcp_now

    v_base = linear_gain * error_base
    v_base = clip_norm(v_base, max_linear_vel)

    J = tcp_jacobian(q_now)    # default: pinocchio
    qdot = damped_pinv(J, damping) @ v_base
    qdot = low_pass(qdot)
    qdot = clip_abs(qdot, max_joint_vel)

    qdot is the Panthera-style primary output for velocity backends.
    q_cmd = q_prev + qdot * dt is kept for dry-run logging and legacy
    position/MIT backends.

Reference: REFACTOR_PLAN.md Section 7.1
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
class ServoParams:
    """Parameters for the Jacobian position servo.

    Reference: REFACTOR_PLAN.md Section 7.1 recommended initial parameters.
    """

    servo_hz: float = 30.0
    desired_distance: float = 0.15        # standoff distance (m)
    linear_gain: float = 1.0
    max_linear_vel: float = 0.02          # m/s
    damping: float = 0.04                 # damped pseudo-inverse
    max_joint_vel: float = 0.05           # rad/s
    max_joint_step: float = 0.003         # rad per cycle
    target_filter_alpha: float = 0.20     # EMA for target position
    normal_filter_alpha: float = 0.20     # EMA for marker normal
    qdot_filter_alpha: float = 0.35       # EMA for joint velocity
    target_timeout_s: float = 0.45        # target lost threshold
    max_target_jump: float = 0.06         # reject large jumps (m)
    position_deadband: float = 0.002      # m
    target_jitter_translation: float = 0.004      # m, Panthera-style target update deadzone
    target_jitter_normal_angle_deg: float = 3.0   # deg, Panthera-style normal update deadzone
    target_blend_alpha: float = 0.82              # fast blend from accepted target to servo target
    fast_target_delta: float = 0.018              # m, increase target blend for larger target motion

    @property
    def dt(self) -> float:
        return 1.0 / self.servo_hz


def _normalize(
    vector: npt.NDArray[np.float64],
    fallback: npt.NDArray[np.float64],
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


def _damped_pinv(
    J: npt.NDArray[np.float64],
    damping: float,
    adaptive: bool = True,
    manipulability_threshold: float = 0.006,
) -> tuple[npt.NDArray[np.float64], float]:
    rows, _ = J.shape
    jjt = J @ J.T
    twist_scale = 1.0
    effective_damping = damping

    if adaptive:
        determinant = np.linalg.det(jjt)
        manipulability = np.sqrt(max(0.0, determinant))
        if manipulability < manipulability_threshold:
            effective_damping = damping * (
                1.0 + (manipulability_threshold - manipulability) * 120.0
            )
            twist_scale = max(0.15, manipulability / manipulability_threshold)

    damping_matrix = (effective_damping ** 2) * np.eye(rows, dtype=np.float64)
    try:
        return J.T @ np.linalg.inv(jjt + damping_matrix), twist_scale
    except np.linalg.LinAlgError:
        stronger = (effective_damping * 10.0) ** 2 * np.eye(rows)
        return J.T @ np.linalg.inv(jjt + stronger), twist_scale


class JacobianPositionServo:
    """Position-only Jacobian visual servo.

    Takes a target pose (marker center + normal) and current joint state,
    outputs qdot for velocity backends, plus q_cmd for dry-run logging and
    legacy position/MIT backends.

    This is the first execution-safe subset of the Panthera ArUco
    velocity-Jacobian servo: position only, no PBVS orientation term yet.
    """

    def __init__(
        self,
        params: Optional[ServoParams] = None,
        model: Optional["NeroPinocchioModel"] = None,
    ) -> None:
        self.params = params or ServoParams()
        self._filtered_marker_center: Optional[npt.NDArray[np.float64]] = None
        self._filtered_normal: Optional[npt.NDArray[np.float64]] = None
        self._accepted_target_tcp: Optional[npt.NDArray[np.float64]] = None
        self._accepted_normal: Optional[npt.NDArray[np.float64]] = None
        self._smoothed_servo_target: Optional[npt.NDArray[np.float64]] = None
        self._filtered_qdot: Optional[npt.NDArray[np.float64]] = None
        self._prev_q_cmd: Optional[npt.NDArray[np.float64]] = None
        self._last_target_stamp: Optional[float] = None

        if model is None:
            from models.nero_model import NeroPinocchioModel
            model = NeroPinocchioModel()
        self._model = model

    def update(
        self,
        q_now: npt.NDArray[np.float64],
        target_position: Optional[npt.NDArray[np.float64]],
        target_normal: Optional[npt.NDArray[np.float64]],
        stamp: Optional[float] = None,
    ) -> Optional[dict]:
        """Compute one servo cycle.

        Args:
            q_now: Current joint angles (7,) in radians.
            target_position: Marker center in base frame (3,). None if lost.
            target_normal: Marker normal in base frame (3,). None if lost.
            stamp: Current time in seconds.

        Returns:
            dict with q_cmd, qdot, error, target_tcp, and debug fields.
            When the target is lost, returns a hold command at q_now.
        """
        now = time.time() if stamp is None else float(stamp)
        q_now = np.asarray(q_now, dtype=np.float64).ravel()
        if len(q_now) != 7:
            raise ValueError(f"Expected 7 joint angles, got {len(q_now)}")

        tcp_now, _ = self._model.fk(q_now)

        if target_position is None or target_normal is None:
            self._reset_target_filters()
            return self._hold(q_now, tcp_now, now, target_lost=True)

        marker_center = np.asarray(target_position, dtype=np.float64).ravel()
        marker_normal = np.asarray(target_normal, dtype=np.float64).ravel()
        if len(marker_center) != 3 or len(marker_normal) != 3:
            raise ValueError("target_position and target_normal must have shape (3,)")

        marker_normal = _normalize(marker_normal, np.array([1.0, 0.0, 0.0]))
        if np.dot(tcp_now - marker_center, marker_normal) < 0.0:
            marker_normal = -marker_normal

        marker_center, marker_normal = self._filter_marker_measurement(marker_center, marker_normal)
        raw_target_tcp = marker_center + marker_normal * self.params.desired_distance
        if (
            self._accepted_target_tcp is not None
            and np.linalg.norm(raw_target_tcp - self._accepted_target_tcp) > self.params.max_target_jump
        ):
            logger.warning(
                "Rejecting target jump: %.4f m",
                np.linalg.norm(raw_target_tcp - self._accepted_target_tcp),
            )
            return self._hold(q_now, tcp_now, now, target_rejected=True)

        self._last_target_stamp = now
        target_tcp = self._accept_and_smooth_target(raw_target_tcp, marker_normal)
        error = target_tcp - tcp_now

        if np.linalg.norm(error) < self.params.position_deadband:
            v_base = np.zeros(3, dtype=np.float64)
            qdot = np.zeros(7, dtype=np.float64)
            self._filtered_qdot = qdot.copy()
            self._prev_q_cmd = q_now.copy()
            return {
                "q_cmd": q_now.copy(),
                "qdot": qdot,
                "error": error,
                "target_tcp": target_tcp,
                "tcp_now": tcp_now,
                "marker_center": marker_center,
                "marker_normal": marker_normal,
                "v_base": v_base,
                "twist_scale": 1.0,
                "target_lost": False,
                "target_rejected": False,
                "stamp": now,
            }
        else:
            v_base = self.params.linear_gain * error
            v_base = _clip_norm(v_base, self.params.max_linear_vel)

        J = self._model.jacobian(q_now)[:3, :]
        J_pinv, twist_scale = _damped_pinv(
            J,
            damping=self.params.damping,
            adaptive=True,
        )
        qdot = J_pinv @ (v_base * twist_scale)
        qdot = self._filter_qdot(qdot)
        qdot = np.clip(qdot, -self.params.max_joint_vel, self.params.max_joint_vel)

        q_base = q_now if self._prev_q_cmd is None else self._prev_q_cmd
        q_cmd_unclipped = q_base + qdot * self.params.dt
        q_cmd = self._clip_step(q_now, q_cmd_unclipped)
        q_cmd = self._clip_limits_without_large_step(q_now, q_cmd)
        self._prev_q_cmd = q_cmd.copy()

        return {
            "q_cmd": q_cmd,
            "qdot": qdot,
            "error": error,
            "target_tcp": target_tcp,
            "tcp_now": tcp_now,
            "marker_center": marker_center,
            "marker_normal": marker_normal,
            "v_base": v_base,
            "twist_scale": twist_scale,
            "target_lost": False,
            "target_rejected": False,
            "stamp": now,
        }

    def _filter_marker_measurement(
        self,
        marker_center: npt.NDArray[np.float64],
        marker_normal: npt.NDArray[np.float64],
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        if self._filtered_marker_center is None:
            self._filtered_marker_center = marker_center.copy()
        else:
            alpha = self.params.target_filter_alpha
            self._filtered_marker_center = (
                alpha * marker_center + (1.0 - alpha) * self._filtered_marker_center
            )

        if self._filtered_normal is None:
            self._filtered_normal = marker_normal.copy()
        else:
            if np.dot(self._filtered_normal, marker_normal) < 0.0:
                marker_normal = -marker_normal
            alpha = self.params.normal_filter_alpha
            self._filtered_normal = (
                alpha * marker_normal + (1.0 - alpha) * self._filtered_normal
            )
            self._filtered_normal = _normalize(self._filtered_normal, marker_normal)

        return self._filtered_marker_center.copy(), self._filtered_normal.copy()

    def _accept_and_smooth_target(
        self,
        target_tcp: npt.NDArray[np.float64],
        marker_normal: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        if self._accepted_target_tcp is None or self._accepted_normal is None:
            self._accepted_target_tcp = target_tcp.copy()
            self._accepted_normal = marker_normal.copy()
        elif self._should_accept_target_update(
            self._accepted_target_tcp,
            self._accepted_normal,
            target_tcp,
            marker_normal,
        ):
            self._accepted_target_tcp = target_tcp.copy()
            self._accepted_normal = marker_normal.copy()

        if self._smoothed_servo_target is None:
            self._smoothed_servo_target = self._accepted_target_tcp.copy()
        else:
            target_delta = float(np.linalg.norm(self._accepted_target_tcp - self._smoothed_servo_target))
            alpha = np.clip(
                self.params.target_blend_alpha
                + target_delta / max(self.params.fast_target_delta, 1e-6) * 0.38,
                self.params.target_blend_alpha,
                0.90,
            )
            self._smoothed_servo_target = (
                alpha * self._accepted_target_tcp
                + (1.0 - alpha) * self._smoothed_servo_target
            )
        return self._smoothed_servo_target.copy()

    def _should_accept_target_update(
        self,
        previous_target_tcp: npt.NDArray[np.float64],
        previous_normal: npt.NDArray[np.float64],
        target_tcp: npt.NDArray[np.float64],
        target_normal: npt.NDArray[np.float64],
    ) -> bool:
        translation_delta = float(np.linalg.norm(target_tcp - previous_target_tcp))
        previous_normal = _normalize(previous_normal, target_normal)
        target_normal = _normalize(target_normal, previous_normal)
        cos_angle = float(np.clip(np.dot(previous_normal, target_normal), -1.0, 1.0))
        normal_angle_deg = float(np.degrees(np.arccos(cos_angle)))
        return (
            translation_delta >= self.params.target_jitter_translation
            or normal_angle_deg >= self.params.target_jitter_normal_angle_deg
        )

    def _filter_qdot(
        self,
        qdot: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        if self._filtered_qdot is None:
            self._filtered_qdot = qdot.copy()
        else:
            alpha = self.params.qdot_filter_alpha
            self._filtered_qdot = alpha * qdot + (1.0 - alpha) * self._filtered_qdot
        return self._filtered_qdot.copy()

    def _clip_step(
        self,
        q_now: npt.NDArray[np.float64],
        q_cmd: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        step = np.clip(
            q_cmd - q_now,
            -self.params.max_joint_step,
            self.params.max_joint_step,
        )
        return q_now + step

    def _reset_target_filters(self) -> None:
        self._filtered_marker_center = None
        self._filtered_normal = None
        self._accepted_target_tcp = None
        self._accepted_normal = None
        self._smoothed_servo_target = None

    def _clip_limits_without_large_step(
        self,
        q_now: npt.NDArray[np.float64],
        q_cmd: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Apply joint limits without creating a command jump.

        If the current feedback is already outside the configured software
        limits, a plain clamp would command a large immediate move back to
        the limit.  Keep the per-cycle step limit as the final authority.
        """
        q_limited = self._model.clamp_to_limits(q_cmd)
        return self._clip_step(q_now, q_limited)

    def _hold(
        self,
        q_now: npt.NDArray[np.float64],
        tcp_now: npt.NDArray[np.float64],
        stamp: float,
        target_lost: bool = False,
        target_rejected: bool = False,
    ) -> dict:
        self._filtered_qdot = np.zeros(7, dtype=np.float64)
        self._prev_q_cmd = q_now.copy()
        return {
            "q_cmd": q_now.copy(),
            "qdot": np.zeros(7, dtype=np.float64),
            "error": np.zeros(3, dtype=np.float64),
            "target_tcp": None,
            "tcp_now": tcp_now,
            "marker_center": None,
            "marker_normal": None,
            "v_base": np.zeros(3, dtype=np.float64),
            "twist_scale": 1.0,
            "target_lost": target_lost,
            "target_rejected": target_rejected,
            "stamp": stamp,
        }
