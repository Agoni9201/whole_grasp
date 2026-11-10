"""ROS /control/move_j backend.

Compatibility backend for cross-checking servo core behavior against
the force_ctrl backend. NOT the main control path.

Reference: REFACTOR_PLAN.md Section 4.2
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import numpy.typing as npt

from .base import ControlBackend, BackendCommand

logger = logging.getLogger(__name__)


class RosMoveJBackend(ControlBackend):
    """Control backend using ROS /control/move_j topic.

    This backend:
    - Integrates qdot into q_cmd and sends via ROS.
    - Is used as a comparison against force_ctrl backend.
    - Is NOT the Panthera-style main control path.
    """

    def __init__(
        self,
        dry_run: bool = True,
        joint_step_limit: float = 0.003,
        **kwargs,
    ) -> None:
        super().__init__(name="ros_move_j", dry_run=dry_run)
        self.joint_step_limit = joint_step_limit
        # TODO: Add ROS publisher setup

    def connect(self) -> None:
        # TODO: Initialize ROS node, publishers, subscribers
        logger.warning("RosMoveJBackend.connect() not yet implemented")
        self._connected = True

    def send(self, cmd: BackendCommand) -> None:
        if self.dry_run:
            self._log_command(cmd)
            return
        # TODO: Publish to /control/move_j
        raise NotImplementedError("RosMoveJBackend.send() not yet implemented")

    def get_joint_angles(self) -> Optional[npt.NDArray[np.float64]]:
        # TODO: Subscribe to /feedback/joint_states
        return None

    def get_flange_pose(self) -> Optional[npt.NDArray[np.float64]]:
        # TODO: Subscribe to /feedback/tcp_pose
        return None

    def disconnect(self) -> None:
        logger.info("RosMoveJBackend disconnected")
        self._connected = False
