"""State-machine tests only: no camera, CAN, model loading or SDK connection."""

from collections import deque
from argparse import ArgumentTypeError
import logging
import time
from types import SimpleNamespace as NS
from unittest.mock import Mock

import numpy as np
import pytest

from real_nero_grasp.servo_cli import servo_parser
from servo_delivery import FakeBackend, ServoDeliveryNode


def _tracking_node(state, follow_only):
    node = ServoDeliveryNode.__new__(ServoDeliveryNode)
    node.args = NS(visual_orientation=False, approach_error_m=.03,
                   approach_error_deg=5.0, stable_frames=2,
                   handover_stable_frames=2, handover_distance=.1,
                   servo_follow_only=follow_only)
    node.backend = FakeBackend()
    node.get_logger = lambda: logging.getLogger('follow-only-test')
    node._qdot_window = deque(maxlen=60)
    node._last_target_time = time.time()
    node._last_tracking_log = 0.0
    node._last_lost_log = 0.0
    node._target_lost = False
    node._reacquire_count = 0
    node._reacquire_target = None
    node._imp_cfg = None
    node._detection_reason = 'no_hand_2d'
    node.target = NS(last_diagnostics={'reason': 'no_hand_2d'})
    node.state = state
    node.state_entered = 0.0
    node.stable_frames = 100
    node.gripper = Mock()
    node.model = NS(fk=lambda q: (np.zeros(3), np.eye(3)))
    node.servo = NS(params=NS(desired_distance=.15), update=Mock(return_value={
        'error': np.zeros(3), 'orientation_error': np.zeros(3)}))
    node.safety = NS(evaluate=lambda *args: NS(fatal=False, hold=False))
    node._detect_latest = lambda *args: (np.array([0.0, 0.0, .1]), None, None)
    node._refresh = Mock()
    return node


@pytest.mark.parametrize('state', ['TRACK_HAND', 'APPROACH', 'HANDOVER'])
def test_follow_only_refreshes_tracking_without_handover_or_opening(state):
    node = _tracking_node(state, True)
    for _ in range(4):
        node._on_timer()
    assert node.state == state
    assert node._refresh.call_count == 4
    assert node.servo.params.desired_distance == .15
    node.gripper.open.assert_not_called()


def test_normal_delivery_state_machine_is_unchanged_without_flag():
    node = _tracking_node('HANDOVER', False)
    node._on_timer()
    assert node.state == 'RELEASE'
    node.gripper.open.assert_called_once_with()


def test_follow_only_flag_is_opt_in():
    parser = servo_parser()
    assert not parser.parse_args([]).servo_follow_only
    assert parser.parse_args(['--servo-follow-only']).servo_follow_only


def test_delivery_follow_defaults_keep_gain_caps_with_low_speed_assist_ramp():
    args = servo_parser().parse_args([])
    np.testing.assert_array_equal(args.follow_kp, [2, 2, 2, 2, 1, 1, 1])
    assert args.follow_min_qdot == .02
    assert args.follow_kp_rise_time == .20
    assert args.follow_release_error == .02
    assert args.max_joint_vel == .08
    assert args.follow_max_error_rad == .03


@pytest.mark.parametrize('text,expected', [
    ('0', [0] * 7),
    ('1.5', [1.5] * 7),
    ('2,2,2,2,1,1,1', [2, 2, 2, 2, 1, 1, 1]),
])
def test_follow_kp_accepts_scalar_or_seven_axis_gains(text, expected):
    args = servo_parser().parse_args(['--follow-kp', text])
    np.testing.assert_array_equal(args.follow_kp, expected)


@pytest.mark.parametrize('text', [
    'nan', 'inf', '-inf', '-1', '1,2', '1,2,3,4,5,6',
    '1,2,3,4,5,6,7,8', '1,2,3,4,5,6,nan',
    '1,2,3,4,5,6,-1', '1,2,3,4,5,6,inf',
])
def test_follow_kp_rejects_invalid_gains(text):
    with pytest.raises(SystemExit):
        servo_parser().parse_args([f'--follow-kp={text}'])


@pytest.mark.parametrize('option', [
    '--follow-min-qdot', '--follow-kp-rise-time', '--follow-release-error', '--follow-max-error-rad', '--max-joint-vel',
])
@pytest.mark.parametrize('text', ['-1', 'nan', 'inf', '-inf'])
def test_follow_assist_thresholds_must_be_finite_and_nonnegative(option, text):
    with pytest.raises(SystemExit):
        servo_parser().parse_args([f'{option}={text}'])


@pytest.mark.parametrize('name,value', [
    ('follow_kp', float('nan')),
    ('follow_kp', [1, 1, 1, 1, 1, 1, float('inf')]),
    ('follow_kp', [-1] * 7),
    ('follow_kp', [1, 2]),
    ('follow_min_qdot', -.1),
    ('follow_min_qdot', float('nan')),
    ('follow_kp_rise_time', -.1),
    ('follow_kp_rise_time', float('inf')),
    ('follow_max_error_rad', -.1),
    ('follow_max_error_rad', float('nan')),
    ('max_joint_vel', -.1),
    ('max_joint_vel', float('inf')),
    ('follow_release_error', -.1),
    ('follow_release_error', float('inf')),
])
def test_programmatic_invalid_follow_options_fail_before_model_camera_or_arm(
        monkeypatch, name, value):
    import cameras.realsense as camera_module
    import servo_delivery as delivery
    args = servo_parser().parse_args([])
    args.handeye = __file__  # Only file existence is needed before gain validation.
    args.execute = True
    setattr(args, name, value)
    model = Mock(side_effect=AssertionError('must not load a real model'))
    camera = Mock(side_effect=AssertionError('must not open a camera'))
    monkeypatch.setattr(delivery, 'NeroPinocchioModel', model)
    monkeypatch.setattr(camera_module, 'camera_for', camera)
    monkeypatch.setattr(camera_module, 'close_all', Mock())
    monkeypatch.setattr(delivery.sdk_loop, 'shutdown', Mock())
    with pytest.raises(ArgumentTypeError):
        delivery.run_servo_delivery(args)
    model.assert_not_called()
    camera.assert_not_called()


def test_programmatic_follow_scalar_is_normalized_and_missing_thresholds_use_defaults():
    from servo_delivery import validate_servo_args
    args = servo_parser().parse_args([])
    args.handeye = __file__
    args.follow_kp = 1.5
    del args.follow_release_error
    del args.follow_max_error_rad
    del args.follow_kp_rise_time
    validate_servo_args(args)
    np.testing.assert_array_equal(args.follow_kp, [1.5] * 7)
    assert args.follow_release_error == .02
    assert args.follow_max_error_rad == .03
    assert args.follow_kp_rise_time == .20


def _impedance_follow_node():
    node = ServoDeliveryNode.__new__(ServoDeliveryNode)
    node.args = servo_parser().parse_args([])
    node.args.follow_kp = np.array([2, 2, 2, 2, 1, 1, 1], dtype=float)
    node.args.follow_min_qdot = .08
    # These tests isolate the legacy speed/release curve. The default gain
    # rise and its interaction with dropouts are covered in test_servo_recovery.
    node.args.follow_kp_rise_time = 0.0
    node.args.follow_release_error = .02
    node.args.max_joint_vel = .03
    node.backend = NS(set_cartesian_impedance=Mock())
    node.model = NS(fk=lambda q: (np.zeros(3), np.eye(3)))
    node.get_logger = lambda: logging.getLogger('follow-assist-test')
    node._imp_follow_q = np.zeros(7)
    node._imp_follow_kp_active = np.zeros(7)
    node._imp_smoothed_tcp = np.zeros(3)
    node._imp_smoothed_R = np.eye(3)
    node._imp_target_update_time = 0.0
    node._imp_engaged = False
    node._imp_cfg = node._build_cfg(np.zeros(3), np.eye(3), np.zeros(7))
    return node


def test_follow_assist_ramps_with_pbvs_speed_and_releases_at_zero_speed():
    node = _impedance_follow_node()
    qdot = np.array([.04, -.02, .08, -.16, .01, 0, -.04])
    node._refresh(np.zeros(7), {'qdot': qdot}, .01)
    expected = node.args.follow_kp * np.clip(np.abs(qdot) / .08, 0, 1)
    np.testing.assert_allclose(node._imp_cfg.follow_kp, expected)
    np.testing.assert_allclose(node._imp_cfg.q_follow, qdot * .01)
    node.backend.set_cartesian_impedance.assert_called_once_with(node._imp_cfg)
    node._refresh(np.zeros(7), {'qdot': np.zeros(7)}, .02)
    np.testing.assert_array_equal(node._imp_cfg.follow_kp, 0)


@pytest.mark.parametrize('velocity_limit', [.01, .08, .2])
def test_joint_reference_angle_bound_does_not_change_with_velocity_limit(velocity_limit):
    node = _impedance_follow_node()
    node.args.max_joint_vel = velocity_limit
    node.args.follow_max_error_rad = .012
    node._imp_cfg = node._build_cfg(np.zeros(3), np.eye(3), np.zeros(7))
    q_now = np.array([.2, -.2, .1, .1, 0, 0, 0])
    node._imp_follow_q = q_now + np.array([.05, -.05, .006, -.006, 0, 0, 0])
    node._refresh(q_now, {'qdot': np.zeros(7)}, .01)
    np.testing.assert_allclose(
        node._imp_cfg.q_follow - q_now, [.012, -.012, .006, -.006, 0, 0, 0])
    assert node._imp_cfg.follow_max_error == .012


def test_zero_joint_reference_angle_bound_prevents_follow_spring_offset():
    node = _impedance_follow_node()
    node.args.follow_max_error_rad = 0.0
    node._imp_cfg = node._build_cfg(np.zeros(3), np.eye(3), np.zeros(7))
    node._imp_follow_q = np.full(7, .02)
    node._refresh(np.zeros(7), {'qdot': np.full(7, .08)}, .01)
    np.testing.assert_array_equal(node._imp_cfg.q_follow, 0)
    assert node._imp_cfg.follow_max_error == 0


def test_subthreshold_reference_speed_still_scales_gain_with_higher_velocity_limit():
    node = _impedance_follow_node()
    node.args.max_joint_vel = .08
    node._refresh(np.zeros(7), {'qdot': np.full(7, .02)}, .01)
    np.testing.assert_allclose(node._imp_cfg.follow_kp, node.args.follow_kp * .25)


@pytest.mark.parametrize('ahead,expected_gain', [(.0199, 2.0), (.0200, 0.0), (.025, 0.0)])
def test_follow_assist_yields_when_measured_joint_runs_ahead(ahead, expected_gain):
    node = _impedance_follow_node()
    node._imp_follow_q[0] = -ahead
    node._refresh(np.zeros(7), {'qdot': np.array([.08, 0, 0, 0, 0, 0, 0])}, 0.0)
    assert node._imp_cfg.follow_kp[0] == expected_gain


def test_follow_assist_retains_gain_when_actual_joint_lags_in_commanded_direction():
    node = _impedance_follow_node()
    node._imp_follow_q[0] = .025
    node._refresh(np.zeros(7), {'qdot': np.array([.08, 0, 0, 0, 0, 0, 0])}, 0.0)
    assert node._imp_cfg.follow_kp[0] == 2.0


def test_follow_release_uses_original_threshold_for_legacy_namespace():
    node = _impedance_follow_node()
    del node.args.follow_release_error
    node._imp_follow_q[0] = -.025
    node._refresh(np.zeros(7), {'qdot': np.array([.08, 0, 0, 0, 0, 0, 0])}, 0.0)
    assert node._imp_cfg.follow_kp[0] == 0.0


def test_zero_speed_scale_retains_original_explicit_full_gain_mode():
    node = _impedance_follow_node()
    node.args.follow_min_qdot = 0.0
    node._refresh(np.zeros(7), {'qdot': np.zeros(7)}, .01)
    np.testing.assert_array_equal(node._imp_cfg.follow_kp, node.args.follow_kp)


def test_zero_release_threshold_explicitly_disables_push_ahead_release():
    node = _impedance_follow_node()
    node.args.follow_release_error = 0.0
    node._imp_follow_q[0] = -.025
    node._refresh(np.zeros(7), {'qdot': np.array([.08, 0, 0, 0, 0, 0, 0])}, 0.0)
    assert node._imp_cfg.follow_kp[0] == 2.0


def test_engage_starts_at_measured_pose_without_follow_spring():
    node = _impedance_follow_node()
    node._imp_cfg = None
    q = np.array([.1, -.1, .2, .3, -.2, .1, .2])
    node._engage(q, 1.0)
    assert node._imp_engaged
    np.testing.assert_array_equal(node._imp_cfg.q_follow, q)
    np.testing.assert_array_equal(node._imp_cfg.target_tcp, 0)
    np.testing.assert_array_equal(node._imp_cfg.follow_kp, 0)
    node.backend.set_cartesian_impedance.assert_called_once_with(node._imp_cfg)


def test_duplicate_frame_does_not_erase_target_or_count_as_new_detection():
    from servo_delivery import _NO_NEW_FRAME
    node = _tracking_node('TRACK_HAND', True)
    node._detect_latest = lambda *args: _NO_NEW_FRAME
    node._on_timer()
    node.servo.update.assert_not_called()
    node._refresh.assert_not_called()
    assert not node._target_lost


def test_detection_loss_stops_twist_and_forgets_old_target_only_after_timeout():
    node = _tracking_node('TRACK_HAND', True)
    node._imp_cfg = NS(target_twist=np.ones(6), follow_kp=np.ones(7))
    node._detect_latest = lambda *args: None
    node._on_timer()
    node.servo.update.assert_not_called()
    # A short dropout decelerates from the last command instead of injecting
    # an immediate zero-reference kick.
    assert np.all((node._imp_cfg.target_twist > 0.0)
                  & (node._imp_cfg.target_twist < 1.0))
    assert np.all((node._imp_cfg.follow_kp > 0.0)
                  & (node._imp_cfg.follow_kp < 1.0))
    assert node.stable_frames == 0
    node._last_target_time -= 1.0
    node._on_timer()
    node._on_timer()
    node.servo.update.assert_called_once()
    assert node.servo.update.call_args.args[1] is None
    assert node._target_lost


def test_reacquisition_needs_three_distinct_consistent_new_frames():
    from servo_delivery import _NO_NEW_FRAME
    node = _tracking_node('TRACK_HAND', True)
    node._last_target_time -= 1.0
    target = (np.array([.4, 0., .3]), np.array([0., 0., -1.]), np.array([0., 1., 0.]))
    frames = iter([target, _NO_NEW_FRAME, target, _NO_NEW_FRAME, target])
    node._detect_latest = lambda *args: next(frames)
    for _ in range(4):
        node._on_timer()
        node._refresh.assert_not_called()
    node._on_timer()
    node._refresh.assert_called_once()
    assert not node._target_lost
    assert node.servo.update.call_count == 2  # one reset + one accepted target


def test_jittering_reacquisition_never_advances_to_tracking():
    node = _tracking_node('TRACK_HAND', True)
    node._target_lost = True
    for x in [.4, .5, .4, .5, .4]:
        node._detect_latest = lambda *args, x=x: (
            np.array([x, 0., .3]), np.array([0., 0., -1.]), np.array([0., 1., 0.]))
        node._on_timer()
    assert node._target_lost
    node._refresh.assert_not_called()


def test_rejected_jump_holds_motion_without_reaching_handover():
    node = _tracking_node('HANDOVER', False)
    node._imp_cfg = NS(target_twist=np.ones(6), follow_kp=np.ones(7))
    node.servo.update.return_value = {'target_rejected': True, 'error': np.zeros(3)}
    node.safety.evaluate = lambda *args: NS(
        fatal=False, hold=True, status='target_rejected_hold', reason='jump')
    node._on_timer()
    node._refresh.assert_not_called()
    assert node.stable_frames == 0
    assert node.state == 'HANDOVER'
    node.gripper.open.assert_not_called()
    np.testing.assert_array_equal(node._imp_cfg.target_twist, 0)
    np.testing.assert_array_equal(node._imp_cfg.follow_kp, 0)


def test_duplicate_rgbd_frame_is_not_run_through_detector_twice():
    from cameras.sdk_loop import Image, CameraInfo
    from servo_delivery import _NO_NEW_FRAME
    node = _tracking_node('TRACK_HAND', True)
    del node._detect_latest  # exercise actual implementation
    node.args.target_source = 'mediapipe'
    node._latest_image = Image(2, 2, 'bgr8', 6, bytes(12))
    node._latest_info = CameraInfo(np.eye(3).ravel().tolist(), [])
    node._latest_depth = np.ones((2, 2))
    node._latest_q = np.zeros(7)
    node._consumed_sequence = None
    node._sdk_frame_sequence = 1
    node._sdk_received_monotonic = time.monotonic()
    node.target.detect = Mock(return_value=None)
    assert node._detect_latest(None, time.time()) is None
    assert node._detection_reason == 'no_hand_2d'
    assert node._detect_latest(None, time.time()) is _NO_NEW_FRAME
    node.target.detect.assert_called_once()
    node._sdk_received_monotonic -= 1.0
    assert node._detect_latest(None, time.time()) is None
    assert node._detection_reason == 'camera_frame_stale'


@pytest.mark.parametrize('emergency', [False, True])
def test_shutdown_is_prepared_once_and_never_reenables_after_emergency(emergency):
    node = _tracking_node('TRACK_HAND', True)
    node._shutdown_prepared = False
    node._emergency_stop_requested = emergency
    node.backend.prepare_shutdown = Mock()
    node.prepare_shutdown()
    node.prepare_shutdown()
    assert node.backend.prepare_shutdown.call_count == (0 if emergency else 1)


def test_real_pbvs_rejects_live_jump_but_recovers_after_sustained_loss():
    from servo.pbvs_pose_servo import PBVSPoseServo, PBVSParams
    node = _tracking_node('TRACK_HAND', True)
    model = NS(fk=lambda q: (np.zeros(3), np.eye(3)),
               jacobian=lambda q: np.eye(6, 7), clamp_to_limits=lambda q: q)
    node.servo = PBVSPoseServo(PBVSParams(
        desired_distance=.2, max_target_jump=.08, marker_center_alpha=1.0,
        marker_normal_alpha=1.0, marker_tangent_alpha=1.0), model)
    normal, tangent = np.array([-1., 0., 0.]), np.array([0., 1., 0.])
    q = np.zeros(7)
    node.servo.update(q, np.array([.4, 0., 0.]), normal, tangent)
    jumped = node.servo.update(q, np.array([.55, 0., 0.]), normal, tangent)
    assert jumped['target_rejected']
    node._last_target_time -= 1.0
    node._detect_latest = lambda *args: (np.array([.55, 0., 0.]), normal, tangent)
    node._on_timer()
    node._on_timer()
    node._refresh.assert_not_called()
    node._on_timer()
    node._refresh.assert_called_once()
    accepted = node._refresh.call_args.args[1]
    assert not accepted['target_rejected']
    np.testing.assert_allclose(accepted['target_tcp'], [.35, 0., 0.])


@pytest.mark.parametrize('observations,reason', [
    ([], 'no_hand_2d'),
    ([NS(palm_center_camera_m=None, valid_depth_landmarks=2)], 'invalid_palm_depth'),
])
def test_hand_detection_reports_2d_vs_depth_loss(observations, reason):
    from targets.hand_target import HandTarget
    target = HandTarget.__new__(HandTarget)
    target.min_valid_depth_landmarks = 3
    target._detector = NS(detect=lambda *args: observations)
    assert target.detect(None, None, None, None, None) is None
    assert target.last_diagnostics['reason'] == reason
