"""Tests for the Panthera-style PBVS pose servo."""

import os
import sys

import numpy as np
import numpy.testing as npt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from servo.pbvs_pose_servo import PBVSPoseServo, PBVSParams


class FakePBVSModel:
    JOINT_LIMITS = np.array([[-10.0, 10.0]] * 7)

    def __init__(self):
        self.tcp = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        self.rotation = np.eye(3, dtype=np.float64)

    def fk(self, q):
        return self.tcp.copy(), self.rotation.copy()

    def jacobian(self, q):
        jac = np.zeros((6, 7), dtype=np.float64)
        jac[:6, :6] = np.eye(6)
        return jac

    def clamp_to_limits(self, q):
        return np.clip(q, self.JOINT_LIMITS[:, 0], self.JOINT_LIMITS[:, 1])


def _servo():
    params = PBVSParams(
        servo_hz=10.0,
        desired_distance=0.15,
        linear_gain=5.0,
        linear_integral_gain=0.0,
        linear_derivative_gain=0.0,
        angular_gain=2.6,
        max_linear_vel=0.06,
        max_angular_vel=2.2,
        damping=0.001,
        max_joint_vel=0.15,
        max_joint_step=0.006,
        marker_center_alpha=1.0,
        marker_normal_alpha=1.0,
        marker_tangent_alpha=1.0,
        tcp_twist_smooth_alpha=1.0,
        target_jitter_translation=0.004,
        position_deadband=0.004,
        orientation_deadband_rad=np.deg2rad(1.5),
        max_target_jump=1.0,
    )
    return PBVSPoseServo(params=params, model=FakePBVSModel())


def test_pbvs_generates_position_joint_velocity():
    servo = _servo()
    out = servo.update(
        q_now=np.zeros(7),
        target_position=np.array([0.20, 0.0, 0.0]),
        target_normal=np.array([1.0, 0.0, 0.0]),
        target_tangent=np.array([0.0, 1.0, 0.0]),
        stamp=1.0,
    )

    npt.assert_allclose(out["target_tcp"], [0.05, 0.0, 0.0], atol=1e-12)
    assert np.linalg.norm(out["v_base"]) <= servo.params.max_linear_vel + 1e-12
    assert out["qdot"][0] > 0.0
    assert np.max(np.abs(out["qdot"])) <= servo.params.max_joint_vel + 1e-12


def test_pbvs_position_and_orientation_deadbands_stop_motion():
    servo = _servo()
    out = servo.update(
        q_now=np.zeros(7),
        target_position=np.array([0.151, 0.0, 0.0]),
        target_normal=np.array([1.0, 0.0, 0.0]),
        target_tangent=np.array([0.0, 1.0, 0.0]),
        stamp=1.0,
    )

    npt.assert_allclose(out["qdot"], np.zeros(7), atol=1e-12)
    npt.assert_allclose(out["q_cmd"], np.zeros(7), atol=1e-12)


def test_pbvs_can_hold_orientation_without_spending_joint_velocity_budget():
    params = PBVSParams(
        servo_hz=10.0,
        desired_distance=0.15,
        linear_gain=5.0,
        linear_integral_gain=0.0,
        linear_derivative_gain=0.0,
        angular_gain=2.6,
        enable_orientation=False,
        max_linear_vel=0.06,
        max_angular_vel=2.2,
        damping=0.001,
        max_joint_vel=0.15,
        max_joint_step=0.006,
        marker_center_alpha=1.0,
        marker_normal_alpha=1.0,
        marker_tangent_alpha=1.0,
        tcp_twist_smooth_alpha=1.0,
        target_jitter_translation=0.004,
        position_deadband=0.004,
        orientation_deadband_rad=np.deg2rad(1.5),
        max_target_jump=1.0,
    )
    servo = PBVSPoseServo(params=params, model=FakePBVSModel())
    out = servo.update(
        q_now=np.zeros(7),
        target_position=np.array([0.15, 0.0, 0.0]),
        target_normal=np.array([1.0, 0.0, 0.0]),
        target_tangent=np.array([0.0, 0.0, 1.0]),
        stamp=1.0,
    )

    # The marker tangent induces a nonzero pose orientation error, but the
    # position-only outer loop must not turn that error into qdot.
    assert np.linalg.norm(out["orientation_error"]) > 0.0
    npt.assert_allclose(out["qdot"], np.zeros(7), atol=1e-12)


def test_pbvs_orientation_leash_and_scale_limit_outer_angular_twist():
    params = PBVSParams(
        servo_hz=10.0,
        linear_gain=0.0,
        linear_integral_gain=0.0,
        linear_derivative_gain=0.0,
        angular_gain=2.0,
        orientation_twist_scale=0.5,
        orientation_error_limit_rad=np.deg2rad(4.0),
        max_linear_vel=0.06,
        max_angular_vel=2.2,
        max_joint_vel=0.15,
        marker_center_alpha=1.0,
        marker_normal_alpha=1.0,
        marker_tangent_alpha=1.0,
        tcp_twist_smooth_alpha=1.0,
        max_target_jump=1.0,
    )
    servo = PBVSPoseServo(params=params, model=FakePBVSModel())
    out = servo.update(
        q_now=np.zeros(7),
        target_position=np.zeros(3),
        target_orientation=np.diag([1.0, -1.0, -1.0]),
        stamp=1.0,
    )

    # Diagnostics retain the real pose error, while the control twist sees
    # only the 4-degree leash and the 0.5 orientation scale.
    assert np.isclose(
        np.linalg.norm(out["orientation_error"]),
        np.pi,
        atol=1e-6,
    )
    assert np.isclose(
        np.linalg.norm(out["tcp_twist"][3:]),
        2.0 * 0.5 * np.deg2rad(4.0),
        atol=1e-4,
    )


def test_pbvs_ignores_small_target_jitter():
    servo = _servo()
    first = servo.update(
        q_now=np.zeros(7),
        target_position=np.array([0.20, 0.0, 0.0]),
        target_normal=np.array([1.0, 0.0, 0.0]),
        target_tangent=np.array([0.0, 1.0, 0.0]),
        stamp=1.0,
    )
    second = servo.update(
        q_now=np.zeros(7),
        target_position=np.array([0.201, 0.0, 0.0]),
        target_normal=np.array([1.0, 0.0, 0.0]),
        target_tangent=np.array([0.0, 1.0, 0.0]),
        stamp=1.1,
    )

    npt.assert_allclose(second["target_tcp"], first["target_tcp"], atol=1e-12)


def test_pbvs_accepts_marker_roll_when_center_and_normal_are_unchanged():
    servo = _servo()
    first = servo.update(
        q_now=np.zeros(7),
        target_position=np.array([0.15, 0.0, 0.0]),
        target_normal=np.array([1.0, 0.0, 0.0]),
        target_tangent=np.array([0.0, 1.0, 0.0]),
        stamp=1.0,
    )
    second = servo.update(
        q_now=np.zeros(7),
        target_position=np.array([0.15, 0.0, 0.0]),
        target_normal=np.array([1.0, 0.0, 0.0]),
        target_tangent=np.array([0.0, 0.0, 1.0]),
        stamp=1.1,
    )

    assert not np.allclose(second["target_rotation"], first["target_rotation"])
    assert np.linalg.norm(second["orientation_error"]) > 0.0


def test_pbvs_handles_near_180deg_orientation_error():
    servo = _servo()
    # Provide a direct desired orientation so the test isolates the rotvec
    # branch that is numerically fragile near pi radians.
    target_rotation = np.diag([1.0, -1.0, -1.0])

    out = servo.update(
        q_now=np.zeros(7),
        target_position=np.array([0.15, 0.0, 0.0]),
        target_orientation=target_rotation,
        stamp=1.0,
    )

    assert np.isfinite(out["orientation_error"]).all()
    npt.assert_allclose(np.linalg.norm(out["orientation_error"]), np.pi, atol=1e-6)
    assert np.max(np.abs(out["qdot"])) <= servo.params.max_joint_vel + 1e-12
    expected_twist = servo._model.jacobian(np.zeros(7)) @ out["qdot"]
    npt.assert_allclose(out["tcp_twist"], expected_twist, atol=1e-12)
    assert np.max(np.abs(out["tcp_twist"])) <= servo.params.max_joint_vel + 1e-12


def test_pbvs_q_cmd_keeps_step_limit_when_current_q_is_outside_limits():
    servo = _servo()
    servo._model.JOINT_LIMITS[1] = [-1.0, 1.0]
    q_now = np.zeros(7)
    q_now[1] = 1.2

    out = servo.update(
        q_now=q_now,
        target_position=np.array([0.20, 0.10, 0.0]),
        target_normal=np.array([1.0, 0.0, 0.0]),
        target_tangent=np.array([0.0, 1.0, 0.0]),
        stamp=1.0,
    )

    step = out["q_cmd"] - q_now
    assert np.max(np.abs(step)) <= servo.params.max_joint_step + 1e-12
