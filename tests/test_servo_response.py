"""PBVS response checks, with synthetic frames and no hardware access."""
import argparse
from types import SimpleNamespace as NS
from unittest.mock import Mock

import numpy as np
import pytest

from real_nero_grasp.servo_cli import servo_parser
from servo_delivery import ServoDeliveryNode, parse_servo_args, validate_servo_args
from servo.pbvs_pose_servo import PBVSPoseServo


def params(*argv):
    node = ServoDeliveryNode.__new__(ServoDeliveryNode)
    node.args = servo_parser().parse_args(list(argv))
    return node._servo_params()


@pytest.mark.parametrize('option', ['--target-filter-alpha', '--normal-filter-alpha',
                                    '--target-blend-alpha', '--twist-smooth-alpha'])
@pytest.mark.parametrize('value', ['0', '-1', '1.1', 'nan', 'inf'])
def test_invalid_filter_settings_are_rejected(option, value):
    with pytest.raises(SystemExit):
        servo_parser().parse_args([f'{option}={value}'])


def test_programmatic_invalid_filter_is_rejected_before_model_or_camera():
    args = servo_parser().parse_args([])
    args.handeye = __file__
    args.normal_filter_alpha = float('nan')
    with pytest.raises(argparse.ArgumentTypeError):
        validate_servo_args(args)


def test_yaml_filter_settings_and_cli_override(tmp_path):
    path = tmp_path / 'pipeline.yaml'
    path.write_text('servo:\n  target_filter_alpha: 0.5\n  normal_filter_alpha: 0.4\n'
                    '  target_blend_alpha: 0.9\n  twist_smooth_alpha: 0.8\n')
    args = parse_servo_args(['--config', str(path), '--target-filter-alpha', '.65'])
    assert args.target_filter_alpha == .65
    assert args.normal_filter_alpha == .4
    assert args.target_blend_alpha == .9
    assert args.twist_smooth_alpha == .8


def test_filter_options_reach_controller_without_changing_motion_limits():
    p = params('--target-filter-alpha', '.6', '--normal-filter-alpha', '.5',
               '--target-blend-alpha', '.9', '--twist-smooth-alpha', '.8')
    assert p.marker_center_alpha == .6
    assert p.marker_normal_alpha == p.marker_tangent_alpha == .5
    assert p.target_blend_alpha == .9
    assert p.tcp_twist_smooth_alpha == .8
    assert p.max_joint_vel == .08
    assert p.max_linear_vel == .03
    assert not p.enable_orientation


def test_higher_measurement_weight_reduces_step_delay_with_same_speed_cap():
    model = NS(fk=lambda q: (np.zeros(3), np.eye(3)),
               jacobian=lambda q: np.eye(6, 7), clamp_to_limits=lambda q: q)
    results = []
    for alpha in (.34, .65):
        p = params('--target-filter-alpha', str(alpha), '--max-linear-vel', '.01')
        servo = PBVSPoseServo(p, model)
        q, normal = np.zeros(7), np.array([-1., 0., 0.])
        for t in range(10):
            servo.update(q, [.2, 0., 0.], normal, stamp=t/30.)
        out = servo.update(q, [.2, .03, 0.], normal, stamp=10/30.)
        assert np.linalg.norm(out['tcp_twist'][:3]) <= .01 + 1e-12
        assert np.max(np.abs(out['qdot'])) <= .08 + 1e-12
        results.append(out['target_tcp'][1])
    assert results[1] > results[0]


def test_safety_hold_cannot_be_undone_by_dropout_soft_decay():
    node = ServoDeliveryNode.__new__(ServoDeliveryNode)
    node._last_target_time = 1.
    node._imp_cfg = NS(target_twist=np.ones(6), follow_kp=np.ones(7))
    node._imp_last_target_twist = np.ones(6)
    node._imp_last_follow_kp = np.ones(7)
    node.backend = NS(set_cartesian_impedance=Mock())
    node._pause_target_motion(1.01, immediate=True)
    node._pause_target_motion(1.02)
    np.testing.assert_array_equal(node._imp_cfg.target_twist, 0)
    np.testing.assert_array_equal(node._imp_cfg.follow_kp, 0)
