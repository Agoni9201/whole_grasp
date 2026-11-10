"""Nero 7-DOF arm Pinocchio model with TCP frame.

Loads the Nero URDF (with gripper), adds a TCP operational frame at the
configured offset from link7, and provides FK + Jacobian computation.

Architecture follows the Panthera-HT approach:
    - FK: pinocchio.forwardKinematics + tool offset correction
    - Jacobian: pinocchio.getJointJacobian + skew correction for tool offset
    - Damped pseudo-inverse for 7-DOF redundancy management

Reference: Panthera_HT_SDK_Extensions, 19_aruco_marker_tracking_velocity_jacobian.py
"""

import os
import logging

import numpy as np
import pinocchio as pin

logger = logging.getLogger(__name__)

# Default TCP offset from link7 frame (meters).
# This is the gripper tip position relative to link7 origin, in link7 frame.
# Pure translation, no rotation.
DEFAULT_TCP_OFFSET = np.array([0.1750, 0.0, -0.0235], dtype=np.float64)

# Default URDF path (relative to this file)
_DEFAULT_URDF = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "nero_with_gripper.urdf"
)

# Number of arm joints (revolute) — excludes gripper prismatic joints
N_ARM_JOINTS = 7

# Arm joint names in URDF order
ARM_JOINT_NAMES = [f"joint{i}" for i in range(1, 8)]


def _skew(v: np.ndarray) -> np.ndarray:
    """Skew-symmetric matrix from a 3-vector."""
    return np.array([
        [0.0, -v[2], v[1]],
        [v[2], 0.0, -v[0]],
        [-v[1], v[0], 0.0],
    ], dtype=np.float64)


class NeroPinocchioModel:
    """Pinocchio model for the Nero 7-DOF arm with TCP frame.

    Loads the URDF, identifies the 7 arm joints, adds a TCP operational
    frame at the configured offset from link7, and provides FK/Jacobian
    computation for visual servoing.

    Parameters
    ----------
    urdf_path : str, optional
        Path to the Nero URDF file. Defaults to the bundled model.
    tcp_offset : array-like, optional
        TCP position offset from link7 in the link7 frame (meters).
        Default ``[0.175, 0, -0.0235]``.
    """

    def __init__(
        self,
        urdf_path: str = "",
        tcp_offset: np.ndarray = None,
    ):
        if not urdf_path:
            urdf_path = _DEFAULT_URDF
        if not os.path.isfile(urdf_path):
            raise FileNotFoundError(f"URDF not found: {urdf_path}")
        self._urdf_path = os.path.abspath(urdf_path)

        # Load URDF into Pinocchio
        self.model = pin.buildModelFromUrdf(self._urdf_path)
        self.data = self.model.createData()

        # Store TCP offset
        if tcp_offset is not None:
            self._tcp_offset = np.asarray(tcp_offset, dtype=np.float64).copy()
        else:
            self._tcp_offset = DEFAULT_TCP_OFFSET.copy()

        # Identify arm joint indices in the Pinocchio model
        self._arm_joint_ids = []
        self._arm_idx_q = []       # q indices in full Pinocchio q vector
        self._arm_idx_v = []       # velocity indices in full v vector
        for name in ARM_JOINT_NAMES:
            jid = self.model.getJointId(name)
            if jid >= self.model.njoints:
                raise ValueError(f"Joint '{name}' not found in URDF")
            self._arm_joint_ids.append(jid)
            self._arm_idx_q.append(self.model.joints[jid].idx_q)
            self._arm_idx_v.append(self.model.joints[jid].idx_v)

        # Last arm joint (joint7) — tool offset is relative to this
        self._last_arm_joint_id = self._arm_joint_ids[-1]

        # Total DOF in the URDF (7 arm + 2 gripper prismatic)
        self._nq_total = self.model.nq
        self._nv_total = self.model.nv

        logger.info(
            f"NeroPinocchioModel loaded: {self._urdf_path}\n"
            f"  arm joints: {N_ARM_JOINTS}, total nq={self._nq_total}\n"
            f"  TCP offset from link7: {self._tcp_offset.tolist()}"
        )

    def clone(self):
        """Return an equivalent model with independent mutable Pinocchio data.

        FK/Jacobian calls mutate ``data``.  A 200 Hz controller and an outer
        monitor/visual thread therefore must use separate instances.
        """
        return NeroPinocchioModel(
            urdf_path=self._urdf_path,
            tcp_offset=self._tcp_offset,
        )

    @property
    def tcp_offset(self) -> np.ndarray:
        """TCP offset from link7 (read-only copy)."""
        return self._tcp_offset.copy()

    @property
    def n_arm_joints(self) -> int:
        """Number of arm joints (7)."""
        return N_ARM_JOINTS

    def tcp_reach_bound(self) -> tuple[np.ndarray, float]:
        """Outer sphere containing every TCP pose, not an IK feasibility test.

        Revolute joints preserve each link's length. The triangle inequality
        gives a conservative radius about the first joint, including this
        instance's tool offset and any fixed transforms folded into the URDF.
        """
        radius = float(np.linalg.norm(self._tcp_offset))
        joint = self._last_arm_joint_id
        while int(self.model.parents[joint]) != 0:
            radius += float(np.linalg.norm(self.model.jointPlacements[joint].translation))
            joint = int(self.model.parents[joint])
        return self.model.jointPlacements[joint].translation.copy(), radius

    def _build_full_q(self, q_arm: np.ndarray) -> np.ndarray:
        """Build full Pinocchio q vector from 7 arm joint angles.

        Gripper prismatic joints are set to 0 (closed).
        """
        q_arm = np.asarray(q_arm, dtype=np.float64).ravel()
        if len(q_arm) != N_ARM_JOINTS:
            raise ValueError(
                f"Expected {N_ARM_JOINTS} arm joint angles, got {len(q_arm)}"
            )
        q_full = np.zeros(self._nq_total, dtype=np.float64)
        for i, idx_q in enumerate(self._arm_idx_q):
            q_full[idx_q] = q_arm[i]
        return q_full

    def _extract_arm_jacobian(self, J_full: np.ndarray) -> np.ndarray:
        """Extract the 7 arm-joint columns from the full Jacobian."""
        return J_full[:, self._arm_idx_v]

    # ==================================================================
    # FK
    # ==================================================================

    def fk(self, q_arm: np.ndarray):
        """Compute forward kinematics to TCP.

        Parameters
        ----------
        q_arm : array-like, shape (7,)
            Arm joint angles in radians.

        Returns
        -------
        position : np.ndarray, shape (3,)
            TCP position in base frame (meters).
        rotation : np.ndarray, shape (3, 3)
            TCP rotation in base frame.
        """
        q_full = self._build_full_q(q_arm)

        pin.forwardKinematics(self.model, self.data, q_full)
        pin.updateFramePlacements(self.model, self.data)

        # FK to the last arm joint (joint7 / link7)
        T_link7 = self.data.oMi[self._last_arm_joint_id]
        R_link7 = T_link7.rotation
        p_link7 = T_link7.translation

        # Apply tool offset (in link7 frame, rotated to world)
        tcp_offset_world = R_link7 @ self._tcp_offset
        tcp_position = p_link7 + tcp_offset_world
        tcp_rotation = R_link7.copy()

        return tcp_position, tcp_rotation

    def fk_link7(self, q_arm: np.ndarray):
        """Compute FK to link7 (before TCP offset).

        Returns
        -------
        position : np.ndarray, shape (3,)
        rotation : np.ndarray, shape (3, 3)
        """
        q_full = self._build_full_q(q_arm)
        pin.forwardKinematics(self.model, self.data, q_full)
        T_link7 = self.data.oMi[self._last_arm_joint_id]
        return T_link7.translation.copy(), T_link7.rotation.copy()

    def fk_to_4x4(self, q_arm: np.ndarray) -> np.ndarray:
        """Compute FK to TCP as a 4x4 homogeneous matrix.

        Returns
        -------
        T_base_tcp : np.ndarray, shape (4, 4)
        """
        pos, rot = self.fk(q_arm)
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = rot
        T[:3, 3] = pos
        return T

    # ==================================================================
    # Jacobian
    # ==================================================================

    def jacobian(self, q_arm: np.ndarray) -> np.ndarray:
        """Compute the 6x7 TCP Jacobian in LOCAL_WORLD_ALIGNED frame.

        The Jacobian maps arm joint velocities to TCP twist:
            v_tcp = J @ qdot_arm

        Tool offset correction is applied via the skew-symmetric trick:
            J_tcp[:3, :] = J_joint[:3, :] - skew(offset_world) @ J_joint[3:, :]

        Parameters
        ----------
        q_arm : array-like, shape (7,)
            Arm joint angles in radians.

        Returns
        -------
        J_tcp : np.ndarray, shape (6, 7)
            TCP Jacobian in world-aligned frame.
        """
        q_full = self._build_full_q(q_arm)

        pin.forwardKinematics(self.model, self.data, q_full)
        pin.updateFramePlacements(self.model, self.data)
        pin.computeJointJacobians(self.model, self.data, q_full)

        # Full Jacobian for the last arm joint (joint7)
        J_full = pin.getJointJacobian(
            self.model,
            self.data,
            self._last_arm_joint_id,
            pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
        )

        # Extract arm joint columns (drop gripper columns)
        J_arm = self._extract_arm_jacobian(J_full)

        # Get link7 rotation for tool offset
        T_link7 = self.data.oMi[self._last_arm_joint_id]
        R_link7 = T_link7.rotation
        tcp_offset_world = R_link7 @ self._tcp_offset

        # Apply tool offset correction to the linear part of the Jacobian
        # Reference: Panthera compute_tcp_fk_and_jacobian()
        J_tcp = J_arm.copy()
        J_tcp[:3, :] -= _skew(tcp_offset_world) @ J_arm[3:, :]

        return J_tcp

    def fk_and_jacobian(self, q_arm: np.ndarray):
        """Compute TCP FK AND Jacobian in ONE forwardKinematics pass.

        Returns (position, rotation, J_tcp) with the same conventions as
        :meth:`fk` and :meth:`jacobian`. Used by the 200 Hz in-loop Cartesian
        impedance so the position error and the Jacobian come from the SAME q
        (the standalone fk()/jacobian() each call forwardKinematics independently,
        which is wasteful and risks q-staleness under the loop lock).
        """
        q_full = self._build_full_q(q_arm)

        pin.forwardKinematics(self.model, self.data, q_full)
        pin.updateFramePlacements(self.model, self.data)
        pin.computeJointJacobians(self.model, self.data, q_full)

        T_link7 = self.data.oMi[self._last_arm_joint_id]
        R_link7 = T_link7.rotation
        tcp_offset_world = R_link7 @ self._tcp_offset
        tcp_position = T_link7.translation + tcp_offset_world
        tcp_rotation = R_link7.copy()

        J_full = pin.getJointJacobian(
            self.model,
            self.data,
            self._last_arm_joint_id,
            pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
        )
        J_arm = self._extract_arm_jacobian(J_full)
        J_tcp = J_arm.copy()
        J_tcp[:3, :] -= _skew(tcp_offset_world) @ J_arm[3:, :]

        return tcp_position, tcp_rotation, J_tcp

    # ==================================================================
    # Damped pseudo-inverse
    # ==================================================================

    @staticmethod
    def damped_pinv(
        J: np.ndarray,
        damping: float = 0.04,
        adaptive: bool = True,
        manipulability_threshold: float = 0.006,
    ) -> np.ndarray:
        """Compute damped pseudo-inverse of the Jacobian.

        Uses the formula: J_pinv = J^T (J J^T + lambda^2 I)^{-1}

        With optional adaptive damping based on manipulability measure:
            When manipulability < threshold, increase damping proportionally.

        Parameters
        ----------
        J : np.ndarray, shape (6, 7)
            Jacobian matrix.
        damping : float
            Base damping factor. Default 0.04.
        adaptive : bool
            If True, increase damping when near singularity.
        manipulability_threshold : float
            Manipulability threshold for adaptive damping.

        Returns
        -------
        J_pinv : np.ndarray, shape (7, 6)
        twist_scale : float
            Scaling factor for the twist (1.0 in normal operation,
            reduced near singularity). Only returned when adaptive=True.
        """
        rows, cols = J.shape
        jjt = J @ J.T

        twist_scale = 1.0
        effective_damping = damping

        if adaptive:
            # Compute manipulability: sqrt(det(J J^T))
            det = np.linalg.det(jjt)
            manipulability = np.sqrt(max(0.0, det))

            if manipulability < manipulability_threshold:
                effective_damping = damping * (
                    1.0 + (manipulability_threshold - manipulability) * 120.0
                )
                twist_scale = max(0.15, manipulability / manipulability_threshold)

        damping_matrix = (effective_damping ** 2) * np.eye(rows, dtype=np.float64)

        try:
            J_pinv = J.T @ np.linalg.inv(jjt + damping_matrix)
        except np.linalg.LinAlgError:
            # Fallback with stronger damping
            stronger = (effective_damping * 10.0) ** 2
            J_pinv = J.T @ np.linalg.inv(jjt + stronger * np.eye(rows))

        return J_pinv, twist_scale

    @staticmethod
    def manipulability(J: np.ndarray) -> float:
        """Compute manipulability measure sqrt(det(J J^T))."""
        jjt = J @ J.T
        return np.sqrt(max(0.0, np.linalg.det(jjt)))

    # ==================================================================
    # Null-space projection (for 7-DOF redundancy)
    # ==================================================================

    @staticmethod
    def null_space_projector(J: np.ndarray, J_pinv: np.ndarray) -> np.ndarray:
        """Compute null-space projection matrix for 7-DOF arm.

        N = I - J_pinv @ J

        This can be used to project secondary objectives (e.g. joint limit
        avoidance, comfortable posture) into the null space of the primary
        task.
        """
        n = J.shape[1]
        return np.eye(n, dtype=np.float64) - J_pinv @ J

    # ==================================================================
    # Joint limits
    # ==================================================================

    # Nero joint limits (radians) — from pyAgxArm api/constants.py
    JOINT_LIMITS = np.array([
        [-2.705261, 2.705261],   # joint1
        [-1.745330, 1.745330],   # joint2
        [-2.757621, 2.757621],   # joint3
        [-1.012291, 2.146755],   # joint4
        [-2.757621, 2.757621],   # joint5
        [-0.733039, 0.959932],   # joint6
        [-1.570797, 1.570797],   # joint7
    ])

    def clamp_to_limits(self, q_arm: np.ndarray) -> np.ndarray:
        """Clamp arm joint angles to joint limits."""
        return np.clip(
            q_arm,
            self.JOINT_LIMITS[:, 0],
            self.JOINT_LIMITS[:, 1],
        )
