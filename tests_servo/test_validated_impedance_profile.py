import argparse
import sys

import numpy as np
import pytest

from control_backends.force_ctrl import impedance_profile as profile
from control_backends.force_ctrl import real_velocity
from control_backends.force_ctrl.controller import ZERO_GRAV_KD
from scripts import test_impedance_fixed_target as fixed_target
from scripts import test_official_cartesian_impedance as official


def test_validated_profile_matches_successful_standalone_tuning():
    assert np.allclose(profile.VALIDATED_CARTESIAN_KP, [200, 200, 200, 5, 5, 5])
    assert np.allclose(profile.VALIDATED_CARTESIAN_KD, [5, 5, 5, 0.2, 0.2, 0.2])
    assert np.allclose(profile.VALIDATED_JOINT_WEIGHTS, [1, 1, 1, 1, 0.5, 0.5, 0.5])
    assert np.allclose(profile.VALIDATED_COULOMB, [0, 0.4, 0, 0.4, 0, 0, 0])
    assert np.allclose(profile.VALIDATED_GRAVITY_FACTORS, [1, 1.25, 1, 1.25, 1, 1, 1.5])
    assert profile.VALIDATED_TASK_TORQUE_LIMIT == 10.0
    assert profile.VALIDATED_TOTAL_TORQUE_LIMIT == 14.0
    assert np.allclose(profile.VALIDATED_VELOCITY_SIGNS, [-1, -1, -1, -1, -1, 1, -1])


def test_official_harness_uses_shared_velocity_reader_and_profile(monkeypatch, capsys):
    assert official.read_real_qdot is real_velocity.read_real_qdot
    monkeypatch.setattr(sys, "argv", ["test_official_cartesian_impedance.py"])

    official.main()

    output = capsys.readouterr().out
    assert "task_torque_limit=±10.0Nm" in output
    assert "joint_weights=[1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5]" in output
    assert "Kx=[200.0, 200.0, 200.0, 5.0, 5.0, 5.0]" in output
    assert "Bx=[5.0, 5.0, 5.0, 0.2, 0.2, 0.2]" in output
    assert "coulomb=[0.0, 0.4, 0.0, 0.4, 0.0, 0.0, 0.0]" in output


def test_fixed_target_defaults_build_safe_production_config(monkeypatch):
    parsed = {}
    original_parse_args = argparse.ArgumentParser.parse_args

    def capture_defaults(parser):
        args = original_parse_args(parser, [])
        parsed["args"] = args
        return args

    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", capture_defaults)
    assert fixed_target.main() == 1
    args = parsed["args"]

    cfg = fixed_target._build_impedance_config(
        args,
        target_tcp=np.zeros(3),
        R_target=np.eye(3),
        q_rest=np.zeros(7),
    )
    assert args.law == "plain-jtf"
    assert cfg.use_dls_mapping is False
    assert cfg.use_real_velocity is True
    # Bare fixed-target execution is the zero-wrench bring-up stage.  The
    # standalone validated K/B values must be supplied explicitly later.
    assert np.allclose(cfg.K, np.zeros(6))
    assert np.allclose(cfg.B, np.zeros(6))
    assert np.allclose(cfg.joint_weights, np.ones(7))
    assert cfg.task_torque_limit == profile.VALIDATED_TASK_TORQUE_LIMIT
    assert np.allclose(cfg.coulomb, np.zeros(7))
    assert np.allclose(cfg.velocity_signs, profile.VALIDATED_VELOCITY_SIGNS)
    assert np.allclose(cfg.kp_joint, 0.0)
    assert np.allclose(cfg.kd_joint, ZERO_GRAV_KD)
    assert np.allclose(cfg.joint_hold_kp, 0.0)
    assert cfg.real_vel_filter_tau == 0.0
    assert cfg.qdot_trip_fast == 2.0
    assert args.max_excursion == 0.20
    assert args.safe_margin == 0.30
    assert np.allclose(args.joint_factors, profile.VALIDATED_GRAVITY_FACTORS)
    assert args.torque_limit == profile.VALIDATED_TOTAL_TORQUE_LIMIT
    assert np.allclose(args.tau_limit_per_joint, profile.VALIDATED_TOTAL_TORQUE_LIMIT)


def test_fixed_target_resolves_relative_target_after_settle():
    tcp_engage = np.array([0.4, -0.1, 0.3])
    R_engage = np.eye(3)
    target, R_target = fixed_target._resolve_engagement_target(
        None,
        [0.01, 0.02, -0.03],
        0.0,
        tcp_engage,
        R_engage,
    )
    assert np.allclose(target, [0.41, -0.08, 0.27])
    assert np.allclose(R_target, R_engage)


def test_fixed_target_absolute_target_is_not_shifted_after_settle():
    absolute = np.array([0.5, 0.0, 0.4])
    target, _ = fixed_target._resolve_engagement_target(
        absolute,
        [9.0, 9.0, 9.0],
        10.0,
        [0.1, 0.2, 0.3],
        np.eye(3),
    )
    assert np.allclose(target, absolute)


def test_fixed_target_accepts_absolute_base_rpy():
    target, rotation = fixed_target._resolve_engagement_target(
        [0.5, -0.1, 0.4],
        [9.0, 9.0, 9.0],
        None,
        [0.1, 0.2, 0.3],
        np.eye(3),
        target_rpy_deg=[10.0, -20.0, 30.0],
    )
    expected = fixed_target.Rot.from_euler(
        "xyz", [10.0, -20.0, 30.0], degrees=True).as_matrix()
    assert np.allclose(target, [0.5, -0.1, 0.4])
    assert np.allclose(rotation, expected)


def test_fixed_target_normalizes_absolute_quaternion():
    _, rotation = fixed_target._resolve_engagement_target(
        None,
        [0.0, 0.0, 0.0],
        None,
        [0.1, 0.2, 0.3],
        np.eye(3),
        target_quaternion=[0.0, 0.0, 0.0, 2.0],
    )
    assert np.allclose(rotation, np.eye(3))


def test_fixed_target_rejects_two_absolute_orientation_inputs():
    with pytest.raises(ValueError, match="mutually exclusive"):
        fixed_target._resolve_engagement_target(
            None,
            [0.0, 0.0, 0.0],
            None,
            [0.1, 0.2, 0.3],
            np.eye(3),
            target_rpy_deg=[0.0, 0.0, 0.0],
            target_quaternion=[0.0, 0.0, 0.0, 1.0],
        )


@pytest.mark.parametrize(
    "value",
    [np.full(7, np.nan), np.zeros(6)],
)
def test_fixed_target_rejects_invalid_joint_vectors(value):
    with pytest.raises(RuntimeError, match="expected finite joint vector"):
        fixed_target._joint_vector(value, "test")
