"""Tests for the ArUco position-only Jacobian servo."""

import os
import sys

import numpy as np
import numpy.testing as npt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from servo.jacobian_position_servo import JacobianPositionServo, ServoParams


class FakeModel:
    JOINT_LIMITS = np.array([[-10.0, 10.0]] * 7)

    def __init__(self):
        self.tcp = np.array([0.0, 0.0, 0.0], dtype=np.float64)

    def fk(self, q):
        return self.tcp.copy(), np.eye(3)

    def jacobian(self, q):
        jac = np.zeros((6, 7), dtype=np.float64)
        jac[0, 0] = 1.0
        jac[1, 1] = 1.0
        jac[2, 2] = 1.0
        return jac

    def clamp_to_limits(self, q):
        return np.clip(q, self.JOINT_LIMITS[:, 0], self.JOINT_LIMITS[:, 1])


def _servo():
    params = ServoParams(
        servo_hz=10.0,
        desired_distance=0.15,
        linear_gain=1.0,
        max_linear_vel=0.02,
        damping=0.001,
        max_joint_vel=0.05,
        max_joint_step=0.003,
        target_filter_alpha=1.0,
        normal_filter_alpha=1.0,
        qdot_filter_alpha=1.0,
        max_target_jump=0.5,
    )
    return JacobianPositionServo(params=params, model=FakeModel())


def test_position_servo_generates_limited_command():
    servo = _servo()
    q_now = np.zeros(7)
    out = servo.update(
        q_now=q_now,
        target_position=np.array([0.10, 0.0, 0.0]),
        target_normal=np.array([1.0, 0.0, 0.0]),
        stamp=1.0,
    )

    npt.assert_allclose(out["target_tcp"], [-0.05, 0.0, 0.0], atol=1e-9)
    assert np.linalg.norm(out["v_base"]) <= servo.params.max_linear_vel + 1e-12
    assert np.max(np.abs(out["qdot"])) <= servo.params.max_joint_vel + 1e-12
    assert np.max(np.abs(out["q_cmd"] - q_now)) <= servo.params.max_joint_step + 1e-12
    assert out["q_cmd"][0] < 0.0
    assert not out["target_lost"]


def test_position_servo_flips_marker_normal_toward_tcp_side():
    servo = _servo()
    out = servo.update(
        q_now=np.zeros(7),
        target_position=np.array([0.10, 0.0, 0.0]),
        target_normal=np.array([-1.0, 0.0, 0.0]),
        stamp=1.0,
    )

    npt.assert_allclose(out["marker_normal"], [-1.0, 0.0, 0.0], atol=1e-9)
    npt.assert_allclose(out["target_tcp"], [-0.05, 0.0, 0.0], atol=1e-9)
    assert out["q_cmd"][0] < 0.0


def test_position_servo_holds_on_missing_target():
    servo = _servo()
    q_now = np.ones(7) * 0.1
    out = servo.update(q_now=q_now, target_position=None, target_normal=None, stamp=1.0)

    npt.assert_allclose(out["q_cmd"], q_now)
    npt.assert_allclose(out["qdot"], np.zeros(7))
    assert out["target_lost"]


def test_position_servo_rejects_large_target_jump():
    servo = _servo()
    q_now = np.zeros(7)
    servo.update(
        q_now=q_now,
        target_position=np.array([0.10, 0.0, 0.0]),
        target_normal=np.array([1.0, 0.0, 0.0]),
        stamp=1.0,
    )
    out = servo.update(
        q_now=q_now,
        target_position=np.array([1.00, 0.0, 0.0]),
        target_normal=np.array([1.0, 0.0, 0.0]),
        stamp=1.1,
    )

    npt.assert_allclose(out["q_cmd"], q_now)
    npt.assert_allclose(out["qdot"], np.zeros(7))
    assert out["target_rejected"]


def test_position_servo_ignores_small_target_jitter():
    servo = _servo()
    q_now = np.zeros(7)
    first = servo.update(
        q_now=q_now,
        target_position=np.array([0.10, 0.0, 0.0]),
        target_normal=np.array([1.0, 0.0, 0.0]),
        stamp=1.0,
    )
    second = servo.update(
        q_now=q_now,
        target_position=np.array([0.101, 0.0, 0.0]),
        target_normal=np.array([1.0, 0.0, 0.0]),
        stamp=1.1,
    )

    npt.assert_allclose(second["target_tcp"], first["target_tcp"], atol=1e-12)


def test_position_deadband_clears_velocity_filter_tail():
    params = ServoParams(
        servo_hz=10.0,
        desired_distance=0.15,
        linear_gain=1.0,
        max_linear_vel=0.02,
        damping=0.001,
        max_joint_vel=0.05,
        max_joint_step=0.003,
        target_filter_alpha=1.0,
        normal_filter_alpha=1.0,
        qdot_filter_alpha=0.35,
        max_target_jump=1.0,
        position_deadband=0.002,
    )
    servo = JacobianPositionServo(params=params, model=FakeModel())
    q_now = np.zeros(7)
    servo._filtered_qdot = np.ones(7) * 0.02

    stopped = servo.update(
        q_now=q_now,
        target_position=np.array([0.151, 0.0, 0.0]),
        target_normal=np.array([1.0, 0.0, 0.0]),
        stamp=1.1,
    )

    npt.assert_allclose(stopped["qdot"], np.zeros(7), atol=1e-12)
    npt.assert_allclose(stopped["q_cmd"], q_now, atol=1e-12)


def test_position_servo_does_not_jump_when_current_q_is_outside_limits():
    servo = _servo()
    servo._model.JOINT_LIMITS[1] = [-1.0, 1.0]
    q_now = np.zeros(7)
    q_now[1] = 1.2

    out = servo.update(
        q_now=q_now,
        target_position=np.array([0.10, 0.10, 0.0]),
        target_normal=np.array([1.0, 0.0, 0.0]),
        stamp=1.0,
    )

    step = out["q_cmd"] - q_now
    assert np.max(np.abs(step)) <= servo.params.max_joint_step + 1e-12
    assert out["q_cmd"][1] >= q_now[1] - servo.params.max_joint_step - 1e-12
