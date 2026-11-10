"""Runtime safety checks for low-speed visual-servo execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import numpy.typing as npt

from models.nero_model import NeroPinocchioModel


@dataclass
class SafetyParams:
    """Conservative limits for the first real ArUco follow test."""

    startup_limit_margin: float = 0.08
    runtime_limit_margin: float = 0.04
    max_tracking_error: float = 0.20
    max_tracking_error_s: float = 1.0


@dataclass
class SafetyDecision:
    status: str
    hold: bool
    fatal: bool = False
    reason: str = ""


class ExecutionSafetyMonitor:
    """Small stateful guard around the servo output.

    The servo already limits velocity, joint step, target jumps, and joint
    commands.  This guard adds execution-level decisions: reject unsafe start
    postures, hold on target loss/rejection, hold near joint limits, and stop
    if the commanded target remains too far away.
    """

    def __init__(
        self,
        params: Optional[SafetyParams] = None,
        joint_limits: Optional[npt.NDArray[np.float64]] = None,
    ) -> None:
        self.params = params or SafetyParams()
        limits = NeroPinocchioModel.JOINT_LIMITS if joint_limits is None else joint_limits
        self.joint_limits = np.asarray(limits, dtype=np.float64)
        self._large_error_since: Optional[float] = None

    def check_start_posture(self, q_now: npt.NDArray[np.float64]) -> tuple[bool, str]:
        """Return whether the current posture is suitable for starting execute."""
        q_now = np.asarray(q_now, dtype=np.float64).ravel()
        margin = float(self.params.startup_limit_margin)
        lower = self.joint_limits[:, 0] + margin
        upper = self.joint_limits[:, 1] - margin
        bad = np.where((q_now < lower) | (q_now > upper))[0]
        if len(bad) == 0:
            return True, ""
        return False, self._format_joint_report(q_now, bad, margin)

    def evaluate(
        self,
        q_now: npt.NDArray[np.float64],
        servo_out: dict,
        now: float,
    ) -> SafetyDecision:
        """Decide whether the current servo output may be executed."""
        q_now = np.asarray(q_now, dtype=np.float64).ravel()
        hard_bad = np.where(
            (q_now < self.joint_limits[:, 0]) | (q_now > self.joint_limits[:, 1])
        )[0]
        if len(hard_bad) > 0:
            return SafetyDecision(
                status="hard_joint_limit_stop",
                hold=True,
                fatal=True,
                reason=self._format_joint_report(q_now, hard_bad, margin=0.0),
            )

        margin = float(self.params.runtime_limit_margin)
        soft_lower = self.joint_limits[:, 0] + margin
        soft_upper = self.joint_limits[:, 1] - margin
        soft_bad = np.where((q_now < soft_lower) | (q_now > soft_upper))[0]
        if len(soft_bad) > 0:
            return SafetyDecision(
                status="near_joint_limit_hold",
                hold=True,
                reason=self._format_joint_report(q_now, soft_bad, margin=margin),
            )

        if servo_out.get("target_lost", False):
            self._large_error_since = None
            return SafetyDecision("marker_lost_hold", hold=True)
        if servo_out.get("target_rejected", False):
            self._large_error_since = None
            return SafetyDecision("target_rejected_hold", hold=True)

        error = np.asarray(servo_out.get("error", np.zeros(3)), dtype=np.float64).ravel()
        error_norm = float(np.linalg.norm(error))
        max_tracking_error = float(self.params.max_tracking_error)
        if max_tracking_error > 0.0 and error_norm > max_tracking_error:
            if self._large_error_since is None:
                self._large_error_since = now
            if now - self._large_error_since >= self.params.max_tracking_error_s:
                return SafetyDecision(
                    status="large_error_hold",
                    hold=True,
                    reason=f"err={error_norm:.4f}m > {max_tracking_error:.4f}m",
                )
        else:
            self._large_error_since = None

        return SafetyDecision("tracking", hold=False)

    def _format_joint_report(
        self,
        q_now: npt.NDArray[np.float64],
        bad_indices: npt.NDArray[np.int64],
        margin: float,
    ) -> str:
        parts = []
        for index in bad_indices:
            lower, upper = self.joint_limits[index]
            parts.append(
                f"joint{index + 1}={np.degrees(q_now[index]):.2f}deg "
                f"allowed=[{np.degrees(lower + margin):.2f}, "
                f"{np.degrees(upper - margin):.2f}]deg"
            )
        return "; ".join(parts)
