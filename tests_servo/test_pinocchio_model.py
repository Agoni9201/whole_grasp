"""Tests for Nero Pinocchio model: FK, Jacobian, and gravity compensation.

Verification strategy:
    1. FK at q=0 → check TCP position is reasonable (~arm reach)
    2. FK symmetry: joints rotated → TCP moves in expected direction
    3. Jacobian vs numerical differentiation (central differences)
    4. Damped pseudo-inverse: J_pinv @ J ≈ I (in row space)
    5. Gravity compensation: non-zero torques at q=0, sign sanity
    6. Gravity compensation: torques increase when arm extends

Run:
    conda activate agilexrobot
    cd "/home/dpf/桌面/RealRobot_Test/Visual Servo"
    python -m pytest tests/test_pinocchio_model.py -v
"""

import sys
import os
import numpy as np
import numpy.testing as npt

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.nero_model import NeroPinocchioModel, N_ARM_JOINTS, DEFAULT_TCP_OFFSET
from models.nero_gravity_comp import NeroGravityCompensator


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _numerical_jacobian(model: NeroPinocchioModel, q: np.ndarray, eps: float = 1e-7):
    """Compute Jacobian by numerical central differences for verification."""
    J_num = np.zeros((6, N_ARM_JOINTS), dtype=np.float64)
    for i in range(N_ARM_JOINTS):
        q_plus = q.copy()
        q_minus = q.copy()
        q_plus[i] += eps
        q_minus[i] -= eps

        p_plus, _ = model.fk(q_plus)
        p_minus, _ = model.fk(q_minus)

        # Linear part
        J_num[:3, i] = (p_plus - p_minus) / (2 * eps)

        # Angular part: use rotation matrix difference
        _, R_plus = model.fk(q_plus)
        _, R_minus = model.fk(q_minus)
        dR = R_plus - R_minus
        # Extract angular velocity from rotation difference
        # omega = vec(R_diff) / (2*eps), using skew-symmetric extraction
        # For small eps, (R_plus - R_minus) / (2*eps) ≈ skew(omega) @ R
        # Simplified: use the skew extraction on (R_plus @ R_minus.T - I) / (2*eps)
        R_diff = R_plus @ R_minus.T
        # Extract rotation vector from R_diff
        from scipy.spatial.transform import Rotation as Rot
        omega = Rot.from_matrix(R_diff).as_rotvec() / (2 * eps)
        J_num[3:, i] = omega

    return J_num


# ---------------------------------------------------------------------------
# FK Tests
# ---------------------------------------------------------------------------

class TestFK:
    """Forward kinematics tests."""

    def setup_method(self):
        self.model = NeroPinocchioModel()

    def test_clone_has_independent_data_and_identical_kinematics(self):
        clone = self.model.clone()
        q = np.array([0.1, -0.2, 0.15, 0.3, -0.1, 0.2, -0.25])

        assert clone is not self.model
        assert clone.model is not self.model.model
        assert clone.data is not self.model.data
        npt.assert_allclose(clone.tcp_offset, self.model.tcp_offset)

        pos, rot, jac = self.model.fk_and_jacobian(q)
        clone_pos, clone_rot, clone_jac = clone.fk_and_jacobian(q)
        npt.assert_allclose(clone_pos, pos)
        npt.assert_allclose(clone_rot, rot)
        npt.assert_allclose(clone_jac, jac)

    def test_fk_at_zero(self):
        """FK at q=0 should give a reasonable TCP position."""
        q = np.zeros(N_ARM_JOINTS)
        pos, rot = self.model.fk(q)

        # At q=0 the arm is straight up, TCP should be near z-axis
        # Sum of link offsets along z: 0.138 + 0.31 + 0.27001 + 0.0235 = 0.74151
        # Plus TCP offset components projected to world
        print(f"FK at q=0: pos={pos}, rot diag={np.diag(rot)}")
        assert pos.shape == (3,), f"Position shape: {pos.shape}"
        assert rot.shape == (3, 3), f"Rotation shape: {rot.shape}"
        # TCP should be above the base (z > 0.5)
        assert pos[2] > 0.5, f"TCP z={pos[2]:.4f}, expected > 0.5"

    def test_fk_to_4x4(self):
        """fk_to_4x4 should return a valid homogeneous matrix."""
        q = np.zeros(N_ARM_JOINTS)
        T = self.model.fk_to_4x4(q)
        assert T.shape == (4, 4)
        # Bottom row should be [0, 0, 0, 1]
        npt.assert_array_almost_equal(T[3, :], [0, 0, 0, 1])
        # Rotation matrix should be orthogonal: R^T R = I
        R = T[:3, :3]
        npt.assert_array_almost_equal(R.T @ R, np.eye(3), decimal=10)
        # det(R) should be +1
        assert abs(np.linalg.det(R) - 1.0) < 1e-10

    def test_fk_joint1_moves_tcp(self):
        """Rotating joint1 should move TCP horizontally when arm is bent.

        At q=0 the arm is vertical, so joint1 only changes orientation.
        We use a bent config where joint1 rotation actually displaces TCP.
        """
        q_base = np.array([0.0, 0.5, 0.0, 0.5, 0.0, 0.0, 0.0])
        q_rotated = q_base.copy()
        q_rotated[0] = 0.5  # rotate joint1

        pos_base, _ = self.model.fk(q_base)
        pos_rotated, _ = self.model.fk(q_rotated)

        # joint1 rotates around z-axis, so TCP x,y should change
        xy_diff = np.linalg.norm(pos_rotated[:2] - pos_base[:2])
        assert xy_diff > 0.01, (
            f"joint1 rotation didn't move TCP horizontally: xy_diff={xy_diff:.6f}"
        )

    def test_fk_joint1_at_zero_only_rotation(self):
        """At q=0, joint1 rotation should NOT move TCP (arm is vertical)."""
        q_zero = np.zeros(N_ARM_JOINTS)
        q_rotated = np.zeros(N_ARM_JOINTS)
        q_rotated[0] = 0.5

        pos_zero, _ = self.model.fk(q_zero)
        pos_rotated, _ = self.model.fk(q_rotated)

        # Position should barely change (only numerical noise)
        pos_diff = np.linalg.norm(pos_rotated - pos_zero)
        assert pos_diff < 0.001, (
            f"At q=0, joint1 rotation shouldn't move TCP: diff={pos_diff:.6f}"
        )

    def test_fk_link7_vs_tcp(self):
        """TCP should be offset from link7 by tcp_offset in link7 frame."""
        q = np.array([0.0, 0.3, 0.0, 0.5, 0.0, 0.3, 0.0])
        pos_tcp, R_tcp = self.model.fk(q)
        pos_link7, R_link7 = self.model.fk_link7(q)

        # TCP rotation should equal link7 rotation (pure translation offset)
        npt.assert_array_almost_equal(R_tcp, R_link7, decimal=12)

        # TCP position = link7 position + R_link7 @ tcp_offset
        expected_tcp_pos = pos_link7 + R_link7 @ self.model.tcp_offset
        npt.assert_array_almost_equal(pos_tcp, expected_tcp_pos, decimal=12)


# ---------------------------------------------------------------------------
# Jacobian Tests
# ---------------------------------------------------------------------------

class TestJacobian:
    """Jacobian verification tests."""

    def setup_method(self):
        self.model = NeroPinocchioModel()

    def test_jacobian_shape(self):
        """Jacobian should be 6x7."""
        q = np.zeros(N_ARM_JOINTS)
        J = self.model.jacobian(q)
        assert J.shape == (6, N_ARM_JOINTS), f"Jacobian shape: {J.shape}"

    def test_jacobian_vs_numerical_at_zero(self):
        """Analytic Jacobian should match numerical differentiation at q=0."""
        q = np.zeros(N_ARM_JOINTS)
        J_analytic = self.model.jacobian(q)
        J_numerical = _numerical_jacobian(self.model, q)

        # Allow small tolerance for numerical differentiation error
        npt.assert_allclose(
            J_analytic, J_numerical, atol=1e-4,
            err_msg="Analytic Jacobian doesn't match numerical at q=0"
        )
        print(f"J_analytic at q=0:\n{J_analytic}")
        print(f"J_numerical at q=0:\n{J_numerical}")
        print(f"Max diff: {np.max(np.abs(J_analytic - J_numerical)):.2e}")

    def test_jacobian_vs_numerical_at_nonzero(self):
        """Analytic Jacobian should match numerical at a non-trivial config."""
        q = np.array([0.3, -0.5, 0.8, 0.2, -0.6, 0.4, 0.1])
        J_analytic = self.model.jacobian(q)
        J_numerical = _numerical_jacobian(self.model, q)

        npt.assert_allclose(
            J_analytic, J_numerical, atol=1e-4,
            err_msg="Analytic Jacobian doesn't match numerical at non-zero q"
        )
        print(f"Max diff at non-zero q: {np.max(np.abs(J_analytic - J_numerical)):.2e}")

    def test_jacobian_consistent_with_fk(self):
        """Jacobian * delta_q should approximate delta_TCP (first-order)."""
        q = np.array([0.2, 0.3, -0.4, 0.5, -0.3, 0.2, 0.1])
        dq = np.array([0.001, -0.001, 0.002, -0.002, 0.001, -0.001, 0.001])

        J = self.model.jacobian(q)

        # Predicted velocity from Jacobian
        v_predicted = J @ dq

        # Actual position change from FK
        pos0, R0 = self.model.fk(q)
        pos1, R1 = self.model.fk(q + dq)

        # Linear part should match
        delta_pos = pos1 - pos0
        npt.assert_allclose(v_predicted[:3], delta_pos, atol=1e-4,
                           err_msg="Jacobian linear velocity doesn't match FK delta")

    def test_damped_pinv_identity(self):
        """J_pinv @ J should approximate identity in row space (6x6)."""
        q = np.array([0.3, -0.5, 0.8, 0.2, -0.6, 0.4, 0.1])
        J = self.model.jacobian(q)
        J_pinv, _ = NeroPinocchioModel.damped_pinv(J, damping=0.001)

        # J_pinv @ J should be close to identity in the 6D task space
        JJ = J_pinv @ J
        # This is a 7x7 matrix; the 6D task space part should be identity
        # More precisely: J @ J_pinv ≈ I (6x6)
        npt.assert_allclose(
            J @ J_pinv, np.eye(6), atol=0.1,
            err_msg="J @ J_pinv not close to I(6x6)"
        )

    def test_manipulability_non_negative(self):
        """Manipulability should be non-negative."""
        q = np.zeros(N_ARM_JOINTS)
        J = self.model.jacobian(q)
        mu = NeroPinocchioModel.manipulability(J)
        assert mu >= 0, f"Manipulability={mu}"
        print(f"Manipulability at q=0: {mu:.6f}")


# ---------------------------------------------------------------------------
# Gravity Compensation Tests
# ---------------------------------------------------------------------------

class TestGravityCompensation:
    """MuJoCo gravity compensation tests."""

    def setup_method(self):
        self.grav = NeroGravityCompensator()

    def test_gravity_comp_at_zero(self):
        """At q=0 (straight up), gravity torques should be near zero or small."""
        q = np.zeros(N_ARM_JOINTS)
        tau = self.grav.compute(q)
        assert tau.shape == (N_ARM_JOINTS,)
        print(f"Gravity torques at q=0: {np.round(tau, 4).tolist()}")
        # At q=0, arm is vertical — gravity torques should be small
        # (not exactly zero due to link COM offsets)
        assert np.max(np.abs(tau)) < 10.0, "Gravity torques too large at q=0"

    def test_gravity_comp_nonzero_config(self):
        """At a non-trivial config, gravity torques should be non-trivial."""
        q = np.array([0.0, 0.5, 0.0, 0.5, 0.0, 0.0, 0.0])
        tau = self.grav.compute(q)
        print(f"Gravity torques at q=[0,0.5,0,0.5,0,0,0]: {np.round(tau, 4).tolist()}")
        # Some joints should have significant gravity compensation
        assert np.max(np.abs(tau)) > 0.1, "Gravity torques too small"

    def test_gravity_comp_symmetry(self):
        """Gravity torques should be symmetric when joint2/joint4 are mirrored."""
        q_pos = np.array([0.0, 0.5, 0.0, 0.3, 0.0, 0.0, 0.0])
        q_neg = np.array([0.0, -0.5, 0.0, -0.3, 0.0, 0.0, 0.0])
        tau_pos = self.grav.compute(q_pos)
        tau_neg = self.grav.compute(q_neg)
        # Torques should have opposite signs (symmetric arm)
        print(f"tau_pos: {np.round(tau_pos, 4)}")
        print(f"tau_neg: {np.round(tau_neg, 4)}")
        # At least the main load-bearing joints should flip sign
        assert np.sign(tau_pos[1]) != np.sign(tau_neg[1]) or abs(tau_pos[1]) < 0.01, \
            "joint2 gravity torque should flip sign when joint2 is mirrored"

    def test_gravity_comp_rejects_bad_q(self):
        """Should raise ValueError for wrong number of joints."""
        try:
            self.grav.compute(np.zeros(5))
            assert False, "Should have raised ValueError"
        except ValueError:
            pass

    def test_gravity_comp_sanity_check(self):
        """Torques should be within sanity limits."""
        q = np.array([0.0, 1.0, 0.0, 1.0, 0.0, 0.5, 0.0])
        tau = self.grav.compute(q)
        assert np.all(np.abs(tau) <= 20.0), f"Torques exceed sanity: {tau}"


# ---------------------------------------------------------------------------
# Consistency Tests
# ---------------------------------------------------------------------------

class TestConsistency:
    """Cross-module consistency tests."""

    def test_same_urdf_same_nq(self):
        """Pinocchio and MuJoCo models should have consistent joint counts."""
        pin_model = NeroPinocchioModel()
        mj_comp = NeroGravityCompensator()

        # Both should see 9 DOF total (7 arm + 2 gripper)
        assert pin_model.model.nq == 9
        assert mj_comp.nq == 9

    def test_tcp_offset_matches_config(self):
        """Default TCP offset should match the config value."""
        expected = np.array([0.1750, 0.0, -0.0235])
        npt.assert_array_almost_equal(DEFAULT_TCP_OFFSET, expected)


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v", "--tb=short"])
