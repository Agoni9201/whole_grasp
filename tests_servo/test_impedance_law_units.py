"""Unit tests for the in-loop impedance pieces ported from the standalone tuning
harness (Step 0 of the impedance + visual-servo integration):

  * cartesian_impedance.coulomb_compensation   (friction feed-forward)
  * real_velocity.read_real_qdot               (shared real motor-velocity reader)
  * mit_loop._compute_impedance_torque         (joint_weights / task_torque_limit
                                                / coulomb wiring; backward compat)

No hardware, no ROS. Pure offline.
"""

import os
import sys
from types import SimpleNamespace

import numpy as np
from scipy.spatial.transform import Rotation as Rot

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from control_backends.cartesian_impedance import (  # noqa: E402
    coulomb_compensation,
    COULOMB_DEADBAND_RAD,
    impedance_wrench,
)
from control_backends.force_ctrl.real_velocity import (  # noqa: E402
    read_real_qdot,
    read_real_qdot_sample,
    DEFAULT_VELOCITY_SIGNS,
)
from control_backends.force_ctrl.data_types import ImpedanceConfig  # noqa: E402
from control_backends.force_ctrl.mit_loop import NeroMITLoop  # noqa: E402


# ----------------------------- coulomb_compensation -----------------------------

def test_coulomb_zero_when_all_coulomb_zero():
    q_rest = np.zeros(7)
    q = np.array([0.1, -0.1, 0.2, 0.0, 0.0, 0.0, 0.0])
    tau = coulomb_compensation(q_rest, q, np.zeros(7))
    assert np.all(tau == 0.0)


def test_coulomb_direction_positive_when_q_below_rest():
    # err = q_rest - q = 0 - (-0.2) = +0.2 -> sign +1 -> tau = +coulomb
    q_rest = np.zeros(7)
    q = np.array([-0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    coulomb = np.array([0.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    tau = coulomb_compensation(q_rest, q, coulomb)
    assert np.isclose(tau[0], 0.4)
    assert np.all(tau[1:] == 0.0)


def test_coulomb_direction_negative_when_q_above_rest():
    q_rest = np.zeros(7)
    q = np.array([0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    coulomb = np.array([0.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    tau = coulomb_compensation(q_rest, q, coulomb)
    assert np.isclose(tau[0], -0.4)


def test_coulomb_deadband_suppresses_small_error():
    # 0.5 deg deadband: |err| below it -> 0
    q_rest = np.zeros(7)
    small = COULOMB_DEADBAND_RAD * 0.5
    q = np.array([-small, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    coulomb = np.array([0.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    tau = coulomb_compensation(q_rest, q, coulomb)
    assert tau[0] == 0.0


def test_coulomb_only_active_joints():
    q_rest = np.zeros(7)
    q = np.array([-0.2, -0.2, 0.0, -0.2, 0.0, 0.0, 0.0])  # joints 2,4 off rest
    coulomb = np.array([0.0, 0.4, 0.0, 0.4, 0.0, 0.0, 0.0])  # only joints 2,4
    tau = coulomb_compensation(q_rest, q, coulomb)
    assert np.isclose(tau[1], 0.4)
    assert np.isclose(tau[3], 0.4)
    assert tau[0] == 0.0


def test_coulomb_smooth_width_is_continuous_and_bounded():
    q_rest = np.zeros(7)
    coulomb = np.array([0.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    width = 0.02
    near = coulomb_compensation(
        q_rest, np.array([-0.001, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        coulomb, smooth_width_rad=width,
    )
    far = coulomb_compensation(
        q_rest, np.array([-0.20, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        coulomb, smooth_width_rad=width,
    )
    assert 0.0 < near[0] < 0.4
    assert np.isclose(far[0], 0.4, atol=1e-8)


def test_coulomb_default_remains_hard_sign_law():
    q_rest = np.zeros(7)
    q = np.array([-0.02, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    coulomb = np.array([0.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    tau = coulomb_compensation(q_rest, q, coulomb)
    assert np.isclose(tau[0], 0.4)


# ------------------------------- read_real_qdot --------------------------------

class _FakeMsg:
    def __init__(self, velocity):
        self.velocity = velocity


class _FakeMotorState:
    def __init__(self, velocity, timestamp=None):
        self.msg = _FakeMsg(velocity)
        if timestamp is not None:
            self.timestamp = timestamp


def _fake_robot(velocities, timestamps=None):
    """velocities: (7,) raw parser velocities (joint-side rad/s, sign-bugged)."""
    if timestamps is None:
        timestamps = [None] * 7
    parser = SimpleNamespace(**{
        f"motor_state_{i}": _FakeMotorState(
            velocities[i - 1], timestamps[i - 1]
        )
        for i in range(1, 8)
    })
    return SimpleNamespace(_parser=parser)


def test_read_real_qdot_applies_default_signs():
    robot = _fake_robot(np.ones(7))
    v = read_real_qdot(robot)
    assert v is not None
    assert np.allclose(v, DEFAULT_VELOCITY_SIGNS)


def test_read_real_qdot_joint6_sign_unique():
    # raw +1 everywhere -> joint6 stays +1, joints 1-5,7 flip to -1
    robot = _fake_robot(np.ones(7))
    v = read_real_qdot(robot)
    assert v[5] == 1.0
    assert np.all(v[[0, 1, 2, 3, 4, 6]] == -1.0)


def test_read_real_qdot_custom_signs():
    robot = _fake_robot(np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]))
    v = read_real_qdot(robot, signs=np.array([-1.0] * 7))
    assert np.allclose(v, -np.arange(1, 8))


def test_read_real_qdot_none_when_no_parser():
    assert read_real_qdot(SimpleNamespace()) is None


def test_read_real_qdot_none_when_field_missing():
    # one motor_state missing -> nan -> None (caller falls back to finite-diff)
    parser = SimpleNamespace(**{
        f"motor_state_{i}": _FakeMotorState(1.0) for i in range(1, 7)
    })  # motor_state_7 absent
    assert read_real_qdot(SimpleNamespace(_parser=parser)) is None


def test_read_real_qdot_none_when_signs_wrong_length():
    robot = _fake_robot(np.ones(7))
    assert read_real_qdot(robot, signs=np.array([-1.0, 1.0])) is None


def test_read_real_qdot_sample_returns_per_joint_timestamps():
    timestamps = np.arange(10.0, 17.0)
    sample = read_real_qdot_sample(
        _fake_robot(np.ones(7), timestamps=timestamps))
    assert sample is not None
    qdot, returned_timestamps = sample
    assert np.allclose(qdot, DEFAULT_VELOCITY_SIGNS)
    assert np.array_equal(returned_timestamps, timestamps)


def test_read_real_qdot_sample_keeps_velocity_without_timestamps():
    sample = read_real_qdot_sample(_fake_robot(np.ones(7)))
    assert sample is not None
    qdot, timestamps = sample
    assert np.allclose(qdot, DEFAULT_VELOCITY_SIGNS)
    assert timestamps is None


# ---------------------- _compute_impedance_torque wiring -----------------------

class _FakePin:
    """Stand-in for NeroPinocchioModel: returns fixed FK + Jacobian."""
    def __init__(self, pos, R, J):
        self._pos = np.asarray(pos, dtype=np.float64)
        self._R = np.asarray(R, dtype=np.float64)
        self._J = np.asarray(J, dtype=np.float64)

    def fk_and_jacobian(self, q):
        return self._pos.copy(), self._R.copy(), self._J.copy()


_POS = np.array([0.5, 0.0, 0.4])
_R = np.eye(3)
# (6,7) Jacobian; ones gives a predictable J^T @ F mapping for the tests.
_J = np.ones((6, 7))


def _bare_loop():
    """A NeroMITLoop instance without running __init__ (no robot/hardware needed).

    _compute_impedance_torque only touches self._pin and the class attr N_JOINTS,
    so a bare __new__ instance with a fake pin is enough to unit-test the law.
    """
    loop = NeroMITLoop.__new__(NeroMITLoop)
    loop._pin = _FakePin(_POS, _R, _J)
    return loop


def _base_cfg(**overrides):
    cfg = ImpedanceConfig(
        active=True,
        target_tcp=_POS.copy(),
        R_target=_R.copy(),
        K=np.array([200.0, 200.0, 200.0, 5.0, 5.0, 5.0]),
        B=np.array([5.0, 5.0, 5.0, 0.2, 0.2, 0.2]),
        joint_damping=np.zeros(7),
        tau_limit_per_joint=np.full(7, 14.0),
        q_rest=np.zeros(7),
        use_dls_mapping=False,   # plain J^T so tau_cart = J^T @ F is predictable
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def test_impedance_zero_torque_at_target():
    # At target, zero velocity -> spring, Cartesian damping, and coulomb
    # (within deadband, q == q_rest) are all zero.
    tau, _, _, _ = _bare_loop()._compute_impedance_torque(
        np.zeros(7), np.zeros(7), _base_cfg())
    assert np.allclose(tau, 0.0, atol=1e-9)


def test_impedance_backward_compatible_when_optional_none():
    # All new fields default None -> the old panthera-real law path is unchanged.
    cfg = _base_cfg()
    assert cfg.joint_weights is None
    assert cfg.task_torque_limit is None
    assert cfg.coulomb is None
    assert cfg.target_twist is None
    tau, _, _, _ = _bare_loop()._compute_impedance_torque(
        np.zeros(7), np.zeros(7), cfg)
    assert np.allclose(tau, 0.0)


def test_impedance_target_twist_adds_damping_feedforward():
    """A desired TCP twist changes only the Cartesian damping term."""
    K = np.zeros(6)
    B = np.array([5.0, 5.0, 5.0, 0.2, 0.2, 0.2])
    target_twist = np.array([0.10, -0.04, 0.02, 0.30, -0.20, 0.05])
    J = np.ones((6, 7))

    wrench = impedance_wrench(
        np.zeros(3),
        np.zeros(3),
        J,
        np.zeros(7),
        K,
        B,
        target_twist=target_twist,
    )
    expected_wrench = B * target_twist
    assert np.allclose(wrench, expected_wrench)

    cfg = _base_cfg(K=K, B=B, target_twist=target_twist)
    tau, _, _, _ = _bare_loop()._compute_impedance_torque(
        np.zeros(7), np.zeros(7), cfg
    )
    assert np.allclose(tau, np.sum(expected_wrench))


def test_impedance_target_twist_matching_tcp_velocity_has_no_damping_error():
    """Matching desired and measured TCP twists cancels the damper."""
    qdot = np.full(7, 0.1)
    target_twist = _J @ qdot
    cfg = _base_cfg(
        K=np.zeros(6),
        target_twist=target_twist,
    )
    tau, _, _, _ = _bare_loop()._compute_impedance_torque(
        np.zeros(7), qdot, cfg
    )
    assert np.allclose(tau, 0.0)


def test_moving_target_disables_discontinuous_coulomb_kick():
    """Visual motion uses continuous twist feed-forward, not sign flips."""
    cfg = _base_cfg(
        K=np.zeros(6),
        B=np.zeros(6),
        q_rest=np.zeros(7),
        q_follow=np.zeros(7),
        coulomb=np.array([0.0, 0.4, 0.0, 0.4, 0.0, 0.0, 0.0]),
        target_twist=np.array([0.02, 0.0, 0.0, 0.0, 0.0, 0.0]),
    )
    tau, _, _, _ = _bare_loop()._compute_impedance_torque(
        np.array([0.0, 0.2, 0.0, -0.2, 0.0, 0.0, 0.0]),
        np.zeros(7),
        cfg,
    )
    assert np.allclose(tau, 0.0)

    cfg.target_twist = np.zeros(6)
    tau_static, _, _, _ = _bare_loop()._compute_impedance_torque(
        np.array([0.0, 0.2, 0.0, -0.2, 0.0, 0.0, 0.0]),
        np.zeros(7),
        cfg,
    )
    assert np.isclose(tau_static[1], -0.4)
    assert np.isclose(tau_static[3], 0.4)


def test_impedance_joint_weights_scale_task_torque():
    loop = _bare_loop()
    offset = np.array([0.05, 0.0, 0.0])  # nonzero spring
    cfg_plain = _base_cfg(target_tcp=_POS + offset)
    cfg_weighted = _base_cfg(
        target_tcp=_POS + offset, joint_weights=np.full(7, 2.0))
    tau_plain, _, _, _ = loop._compute_impedance_torque(np.zeros(7), np.zeros(7), cfg_plain)
    tau_weighted, _, _, _ = loop._compute_impedance_torque(np.zeros(7), np.zeros(7), cfg_weighted)
    assert np.allclose(tau_weighted, 2.0 * tau_plain)


def test_impedance_task_torque_limit_clips_spring():
    # Large error + stiff K -> unclipped task torque far above the limit.
    cfg = _base_cfg(
        target_tcp=_POS + np.array([0.5, 0.0, 0.0]),  # 0.5 m * 200 N/m = 100 N
        task_torque_limit=1.0,
    )
    tau, _, _, _ = _bare_loop()._compute_impedance_torque(np.zeros(7), np.zeros(7), cfg)
    assert np.all(np.abs(tau) <= 1.0 + 1e-9)
    assert np.allclose(np.abs(tau), 1.0)   # driven to the clip rail


def test_impedance_clamps_position_error_inside_200hz_law():
    cfg = _base_cfg(
        target_tcp=_POS + np.array([0.50, 0.0, 0.0]),
        max_position_error=0.025,
    )

    tau, _, _, _ = _bare_loop()._compute_impedance_torque(
        np.zeros(7), np.zeros(7), cfg
    )

    assert np.allclose(tau, 200.0 * 0.025)


def test_impedance_clamps_orientation_error_inside_200hz_law():
    max_error = np.deg2rad(3.0)
    cfg = _base_cfg(
        R_target=Rot.from_euler("z", 30.0, degrees=True).as_matrix(),
        max_orientation_error_rad=max_error,
    )

    tau, _, _, _ = _bare_loop()._compute_impedance_torque(
        np.zeros(7), np.zeros(7), cfg
    )

    assert np.allclose(tau, 5.0 * max_error)


def test_impedance_coulomb_added_beyond_deadband():
    q = np.array([-0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # joint1 err 0.2 rad
    cfg = _base_cfg(coulomb=np.array([0.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
    tau, _, _, _ = _bare_loop()._compute_impedance_torque(q, np.zeros(7), cfg)
    # Spring is zero (at target); only coulomb remains: +0.4 on joint1.
    assert np.isclose(tau[0], 0.4)
    assert np.allclose(tau[1:], 0.0)


def test_impedance_coulomb_skipped_without_q_rest():
    # coulomb set but q_rest None -> skipped (no crash, no coulomb added).
    cfg = _base_cfg(coulomb=np.array([0.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), q_rest=None)
    tau, _, _, _ = _bare_loop()._compute_impedance_torque(
        np.array([-0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), np.zeros(7), cfg)
    assert np.allclose(tau, 0.0)


def test_nullspace_posture_term_uses_measured_velocity_damping():
    # The first two equal Jacobian columns make [1,-1,0,...] a null direction.
    qdot = np.array([1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    cfg = _base_cfg(
        K=np.zeros(6),
        B=np.zeros(6),
        null_kq=1.0,
        q_rest=np.zeros(7),
    )
    tau, _, _, _ = _bare_loop()._compute_impedance_torque(
        np.zeros(7), qdot, cfg)
    assert float(qdot @ tau) < 0.0


def test_unweighted_cartesian_damping_is_dissipative():
    rng = np.random.default_rng(7)
    B = np.diag(np.array([5.0, 5.0, 5.0, 0.2, 0.2, 0.2]))
    for _ in range(100):
        J = rng.normal(size=(6, 7))
        qdot = rng.normal(size=7)
        tau_damping = -J.T @ B @ J @ qdot
        assert float(qdot @ tau_damping) <= 1e-10
