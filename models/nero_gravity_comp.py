"""Nero MuJoCo-based gravity compensation.

Uses ``mujoco.mj_inverse`` to compute the joint torques needed to
counteract gravity at a given joint configuration. Adapted from the
sjyylml/pyAgxArm fork's NeroGravityCompensator.

This module loads the same URDF as Pinocchio (nero_with_gripper.urdf)
so that mass/inertia properties are consistent for gravity compensation.

Reference: sjyylml/pyAgxArm force_ctrl/gravity_comp.py
"""

import os
import re
import logging
import numpy as np

try:
    import mujoco
except ImportError as exc:
    raise ImportError(
        "mujoco is required for gravity compensation. "
        "Install with: pip install mujoco"
    ) from exc

logger = logging.getLogger(__name__)

# Default URDF (same as used by Pinocchio model)
_DEFAULT_URDF = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "nero_with_gripper.urdf"
)

# Number of arm joints (7 revolute) — gripper prismatic joints are
# included in the model for mass but compensated too
N_ARM_JOINTS = 7


class NeroGravityCompensator:
    """MuJoCo-based gravity compensation for the Nero 7-DOF arm.

    Uses ``mujoco.mj_inverse`` to compute the joint torques needed to
    counteract gravity at a given joint configuration.

    The URDF is stripped of visual/collision elements (to avoid mesh path
    resolution issues) before loading into MuJoCo.

    Parameters
    ----------
    urdf_path : str, optional
        Path to the Nero URDF file. Defaults to the bundled model.
    gravity : array-like, optional
        Gravity vector in the world frame. Default ``[0, 0, -9.81]``.
    max_torque_sanity : float, optional
        If any computed torque exceeds this value, a RuntimeError is raised.
        Default 20.0 N·m.
    """

    def __init__(
        self,
        urdf_path: str = "",
        gravity: np.ndarray = None,
        max_torque_sanity: float = 20.0,
    ):
        if not urdf_path:
            urdf_path = os.path.normpath(_DEFAULT_URDF)

        if not os.path.isfile(urdf_path):
            raise FileNotFoundError(f"URDF not found: {urdf_path}")

        # Strip <visual> and <collision> elements so MuJoCo doesn't
        # try to resolve ROS package:// mesh URIs
        with open(urdf_path, "r") as f:
            urdf_xml = f.read()
        urdf_xml = re.sub(
            r"<visual>.*?</visual>", "", urdf_xml, flags=re.DOTALL
        )
        urdf_xml = re.sub(
            r"<collision>.*?</collision>", "", urdf_xml, flags=re.DOTALL
        )

        self._model = mujoco.MjModel.from_xml_string(urdf_xml)
        self._data = mujoco.MjData(self._model)
        self._max_torque = max_torque_sanity

        # Set gravity
        if gravity is not None:
            self._model.opt.gravity[:] = np.asarray(gravity, dtype=np.float64)
        else:
            self._model.opt.gravity[:] = [0.0, 0.0, -9.81]

        # Disable joint limits in MuJoCo (we enforce limits separately)
        self._model.jnt_limited[:] = 0

        self._nq = self._model.nq
        logger.info(
            f"NeroGravityCompensator loaded: {urdf_path}, "
            f"nq={self._nq}, gravity={self._model.opt.gravity.tolist()}"
        )

    @property
    def nq(self) -> int:
        """Total number of DOF in the MuJoCo model."""
        return self._nq

    def compute(self, q: np.ndarray) -> np.ndarray:
        """Compute gravity compensation torques for the 7 arm joints.

        Parameters
        ----------
        q : np.ndarray
            Arm joint angles in radians, shape ``(7,)``.
            Gripper joints are automatically set to 0.

        Returns
        -------
        np.ndarray
            Gravity compensation torques for the 7 arm joints, shape ``(7,)``.
            Positive torque counteracts downward gravity.
        """
        q = np.asarray(q, dtype=np.float64).ravel()
        if len(q) != N_ARM_JOINTS:
            raise ValueError(
                f"Expected {N_ARM_JOINTS} joint angles, got {len(q)}"
            )

        # Build full qpos (arm + gripper at 0)
        qpos = np.zeros(self._nq, dtype=np.float64)
        qpos[:N_ARM_JOINTS] = q

        self._data.qpos[:self._nq] = qpos
        self._data.qvel[:self._nq] = 0.0
        self._data.qacc[:self._nq] = 0.0

        mujoco.mj_inverse(self._model, self._data)
        tau = self._data.qfrc_inverse[:self._nq].copy()

        if np.max(np.abs(tau)) > self._max_torque:
            raise RuntimeError(
                f"Gravity compensation torque too large: "
                f"max={np.max(np.abs(tau)):.2f} > {self._max_torque:.2f} N·m. "
                f"Check URDF or joint configuration."
            )

        # Return only the 7 arm joint torques
        return tau[:N_ARM_JOINTS]
