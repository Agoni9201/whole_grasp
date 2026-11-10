"""Base class for all control backends.

The servo core outputs a unified BackendCommand, and each concrete backend
translates it to the appropriate low-level API call (ROS, SDK move_j, or
force_ctrl / move_mit_batch).

Reference: REFACTOR_PLAN.md Section 4.1
"""

from __future__ import annotations

import abc
import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)


@dataclass
class BackendCommand:
    """Unified command from servo core to control backend.

    Attributes:
        q_cmd: Target joint positions (7,) in radians.
        qdot_cmd: Target joint velocities (7,) in rad/s.
        torque_ff: Feed-forward torques (7,) in Nm.
        kp: Position gains (7,).
        kd: Velocity gains (7,).
        mode: Control mode string. Recognized values:
            "position" (default), "velocity", "mit" (stiff position-follow),
            "mit-impedance" (compliant Cartesian impedance via torque_ff,
            clipped by the backend's impedance_torque_limit before gravity
            is layered on by the MIT loop).
        stamp: Command timestamp (seconds).
    """

    q_cmd: npt.NDArray[np.float64]
    qdot_cmd: npt.NDArray[np.float64] = field(
        default_factory=lambda: np.zeros(7, dtype=np.float64)
    )
    torque_ff: npt.NDArray[np.float64] = field(
        default_factory=lambda: np.zeros(7, dtype=np.float64)
    )
    kp: npt.NDArray[np.float64] = field(
        default_factory=lambda: np.full(7, np.nan, dtype=np.float64)
    )
    kd: npt.NDArray[np.float64] = field(
        default_factory=lambda: np.full(7, np.nan, dtype=np.float64)
    )
    mode: str = "position"
    stamp: Optional[float] = None


class ControlBackend(abc.ABC):
    """Abstract base class for control backends.

    Subclasses must implement:
        - connect(): Establish communication with the arm.
        - send(cmd): Send a BackendCommand.
        - get_joint_angles(): Read current joint positions.
        - get_flange_pose(): Read current end-effector pose.
        - disconnect(): Clean up communication.
    """

    def __init__(self, name: str, dry_run: bool = True) -> None:
        self.name = name
        self.dry_run = dry_run
        self._connected = False

    @property
    def connected(self) -> bool:
        return self._connected

    @abc.abstractmethod
    def connect(self) -> None:
        """Connect to the robot arm."""

    @abc.abstractmethod
    def send(self, cmd: BackendCommand) -> None:
        """Send a unified command to the arm.

        In dry_run mode, only print/log the command without executing.
        """

    @abc.abstractmethod
    def get_joint_angles(self) -> Optional[npt.NDArray[np.float64]]:
        """Read current joint angles (7,) in radians. Returns None if unavailable."""

    @abc.abstractmethod
    def get_flange_pose(self) -> Optional[npt.NDArray[np.float64]]:
        """Read current flange/TCP pose [x, y, z, roll, pitch, yaw]. Returns None if unavailable."""

    @abc.abstractmethod
    def disconnect(self) -> None:
        """Disconnect and clean up."""

    def _log_command(self, cmd: BackendCommand) -> None:
        """Log a command for debugging."""
        logger.info(
            "[%s] cmd: mode=%s q_cmd=%s qdot=%s torque_ff=%s kp=%s kd=%s",
            self.name,
            cmd.mode,
            np.array2string(cmd.q_cmd, precision=4, suppress_small=True),
            np.array2string(cmd.qdot_cmd, precision=4, suppress_small=True),
            np.array2string(cmd.torque_ff, precision=4, suppress_small=True),
            np.array2string(cmd.kp, precision=2, suppress_small=True),
            np.array2string(cmd.kd, precision=2, suppress_small=True),
        )

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()
        return False
