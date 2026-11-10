"""Follow/retreat and recovery regressions with synthetic feedback only."""
from types import SimpleNamespace as NS
from unittest.mock import Mock

import numpy as np
import pytest

from real_nero_grasp.servo_cli import servo_parser
from servo_delivery import (FakeBackend, ServoDeliveryNode,
                            apply_position_target_anti_windup)
from servo.pbvs_pose_servo import PBVSPoseServo
from control_backends.force_ctrl.mit_loop import NeroMITLoop


class Model:
    def fk(self, q):
        return np.zeros(3), np.eye(3)

    def jacobian(self, q):
        return np.eye(6, 7)

    def fk_and_jacobian(self, q):
        return *self.fk(q), self.jacobian(q)

    def clamp_to_limits(self, q):
        return q

    def clone(self):
        return Model()


def node(*extra_args):
    args = servo_parser().parse_args([
        '--target-source', 'mediapipe', '--servo-follow-only',
        '--delivery-distance', '.2', '--max-linear-vel', '.02',
        '--max-joint-vel', '.08', '--follow-kp', '4',
        '--impedance-target-linear-vel', '.02', *extra_args])
    n = ServoDeliveryNode(FakeBackend(), args, Model(), target=NS(last_diagnostics={}))
    n._engage(np.zeros(7), 0.)
    return n


def hand(x, y=0.):
    return np.array([x, y, 0.]), np.array([-1., 0., 0.]), np.array([0., 1., 0.])


def test_task_limit_is_separate_from_total_limit_in_delivery_config():
    cfg = node()._imp_cfg
    assert cfg.task_torque_limit == 10.
    np.testing.assert_array_equal(cfg.tau_limit_per_joint, 14.)


@pytest.mark.parametrize('task_sign,expected,controller_limit', [
    (0, -13., 14.), (1, -3., 14.), (-1, -14., 14.), (0, -8., 8.),
])
def test_real_inner_law_adds_gravity_after_task_clip_then_clips_total(
        task_sign, expected, controller_limit):
    n = node()
    cfg = n._imp_cfg
    cfg.target_tcp = np.array([0., .015 * task_sign, 0.])
    cfg.K = np.array([1e5] * 3 + [0.] * 3)
    cfg.coulomb = None
    cfg.joint_weights = np.ones(7)

    robot = NS(get_joint_angles=lambda: NS(timestamp=1., msg=np.zeros(7)))
    gravity = np.array([0., -13., 0., 0., 0., 0., 0.])
    loop = NeroMITLoop(robot, NS(compute=lambda q: gravity.copy()),
                      torque_limit=controller_limit, pin_model=Model(),
                      joint_limit_buffer=0., feedback_guard_enabled=False)
    loop.set_cartesian_impedance(cfg)
    loop._read_impedance_joint_vel = lambda q, cfg: np.zeros(7)
    sent = []

    def send(**command):
        sent.append(command['torques'].copy())
        loop._running = False

    loop._send_mit_commands = send
    loop._running = True
    loop._control_loop()
    assert loop.get_debug_state()['last_error'] is None
    assert len(sent) == 1
    assert sent[0][1] == pytest.approx(expected)
    assert np.max(np.abs(sent[0])) <= controller_limit


def test_tiny_feedback_overshoot_does_not_lock_tangential_reference():
    previous = np.array([.01502, 0., 0.])
    candidate = previous + np.array([0., .0006, 0.])
    result = apply_position_target_anti_windup(previous, candidate, np.zeros(3), .015)
    assert result[1] > 0
    assert np.linalg.norm(result) <= .015 + 1e-12
    assert np.linalg.norm(result - previous) <= .0006 + 1e-12


def test_external_displacement_recovers_at_reference_rate_without_force_windup():
    reference = np.array([.025, 0., 0.])
    for _ in range(40):
        previous = reference.copy()
        reference = apply_position_target_anti_windup(
            previous, previous + [0., .0006, 0.], np.zeros(3), .015)
        assert np.linalg.norm(reference - previous) <= .0006 + 1e-12
        assert np.linalg.norm(reference) <= np.linalg.norm(previous) + 1e-12
    assert np.linalg.norm(reference) <= .015 + 1e-12


def test_zero_reference_motion_does_not_reanchor_a_lost_target():
    previous = np.array([.025, 0., 0.])
    np.testing.assert_array_equal(
        apply_position_target_anti_windup(previous, previous, np.zeros(3), .015), previous)


def test_saturation_projection_preserves_step_and_error_bounds_in_all_directions():
    random = np.random.default_rng(42)
    for _ in range(300):
        actual = random.uniform(-.1, .1, 3)
        previous = actual + random.uniform(-.02, .02, 3)
        step = random.uniform(-.0005, .0005, 3)
        result = apply_position_target_anti_windup(previous, previous + step, actual, .015)
        assert np.linalg.norm(result - previous) <= np.linalg.norm(step) + 1e-12
        assert np.linalg.norm(result - actual) <= max(.015, np.linalg.norm(previous - actual)) + 1e-12


def test_close_hand_generates_retreat_and_reverses_the_impedance_reference():
    n = node()
    for k in range(60):
        out = n.servo.update(np.zeros(7), *hand(.25), stamp=k / 30.)
        n._refresh(np.zeros(7), out, k / 30.)
    assert n._imp_cfg.target_tcp[0] > 0
    for k in range(60, 150):
        # Continuous approach avoids a synthetic target-jump rejection.
        x = max(.15, .25 - (k - 59) * .003)
        out = n.servo.update(np.zeros(7), *hand(x), stamp=k / 30.)
        assert not out['target_rejected']
        previous = n._imp_cfg.target_tcp.copy()
        n._refresh(np.zeros(7), out, k / 30.)
        assert np.linalg.norm(n._imp_cfg.target_tcp - previous) <= .02 / 30. + 1e-12
        assert np.max(np.abs(n._imp_cfg.q_follow)) <= .03 + 1e-12
    assert out['tcp_twist'][0] < 0
    assert n._imp_cfg.target_tcp[0] < 0
    assert np.linalg.norm(n._imp_cfg.target_tcp) <= .015 + 1e-12


def test_stable_new_close_target_recovers_without_a_reference_jump():
    n = node()
    n.servo.params.marker_center_alpha = 1.
    q = np.zeros(7)
    n.servo.update(q, *hand(.35), stamp=0.)
    target = hand(.15)
    assert n.servo.update(q, *target, stamp=1.)['target_rejected']
    previous = n._imp_cfg.target_tcp.copy()
    for k in range(12):
        assert n._try_reacquire_rejected_target(q, target, 1. + k / 30.) is None
    out = n._try_reacquire_rejected_target(q, target, 1.41)
    assert out is not None and not out['target_rejected']
    assert out['tcp_twist'][0] < 0
    np.testing.assert_array_equal(n._imp_cfg.target_tcp, previous)
    n._refresh(q, out, 1.41)
    assert np.linalg.norm(n._imp_cfg.target_tcp - previous) <= .02 / 30. + 1e-12


def test_jittering_or_drifting_jump_cannot_pass_stable_anchor_check():
    n = node()
    q = np.zeros(7)
    for k in range(40):
        assert n._try_reacquire_rejected_target(q, hand(.15, k * .005), k / 30.) is None
    assert n._rejected_target['count'] < 10


def test_stable_jump_outside_existing_tracking_envelope_stays_rejected():
    n = node()
    n.servo.update(np.zeros(7), *hand(.2), stamp=0.)
    accepted = n.servo._accepted_target_position.copy()
    for k in range(30):
        assert n._try_reacquire_rejected_target(np.zeros(7), hand(.8), 1. + k / 30.) is None
    np.testing.assert_array_equal(n.servo._accepted_target_position, accepted)


def test_gap_resets_stable_jump_confirmation():
    n = node()
    for k in range(10):
        assert n._try_reacquire_rejected_target(np.zeros(7), hand(.15), k / 30.) is None
    assert n._try_reacquire_rejected_target(np.zeros(7), hand(.15), 1.) is None
    assert n._rejected_target['count'] == 1


def test_jump_recovery_does_not_change_aruco_behavior():
    n = node()
    n.args.target_source = 'aruco'
    n.servo.update = Mock(side_effect=AssertionError('must not reset ArUco tracking'))
    for k in range(30):
        assert n._try_reacquire_rejected_target(np.zeros(7), hand(.15), k / 30.) is None


@pytest.mark.parametrize('old_distance,new_distance,alpha', [
    (.35, .15, 1.), (.4, .1, .34),
])
def test_timer_reacquires_only_after_stable_new_frames_and_retreat_stays_bounded(
        monkeypatch, old_distance, new_distance, alpha):
    import servo_delivery as delivery
    n = node()
    clock = NS(now=100.)
    monkeypatch.setattr(delivery, 'time', NS(time=lambda: clock.now,
                                           monotonic=lambda: clock.now))
    n.servo.params.marker_center_alpha = alpha
    n.servo.update(np.zeros(7), *hand(old_distance), stamp=clock.now)
    n._last_target_time = clock.now
    n._target_lost = False
    n.state = 'TRACK_HAND'
    n._detect_latest = lambda *args: hand(new_distance)
    for k in range(12):
        clock.now = 100. + k / 30.
        n._on_timer()
        assert n._preview_status == 'HOLD'
        np.testing.assert_array_equal(n._imp_cfg.target_twist, 0)
        np.testing.assert_array_equal(n._imp_cfg.target_tcp, 0)
    clock.now = 100.44
    n._on_timer()
    assert n._preview_status == 'TRACKING'
    assert n._imp_cfg.target_twist[0] < 0
    assert -.02 / 30. <= n._imp_cfg.target_tcp[0] < 0


def test_missing_hand_resets_pending_jump_confirmation(monkeypatch):
    import servo_delivery as delivery
    n = node()
    clock = NS(now=100.)
    monkeypatch.setattr(delivery, 'time', NS(time=lambda: clock.now,
                                           monotonic=lambda: clock.now))
    n.state = 'TRACK_HAND'
    n._last_target_time = clock.now
    n._target_lost = False
    n._try_reacquire_rejected_target(np.zeros(7), hand(.15), clock.now)
    assert n._rejected_target is not None
    n._detect_latest = lambda *args: None
    clock.now += .03
    n._on_timer()
    assert n._rejected_target is None


def assist_step(n, now, speed=.02):
    n._last_target_time = now
    n._refresh(np.zeros(7), {'qdot': np.full(7, speed)}, now)
    return n._imp_cfg.follow_kp.copy()


def warm_assist(n):
    for k in range(1, 7):
        assist_step(n, k / 30.)


def test_low_speed_assist_reaches_same_gain_cap_in_200ms_without_speed_increase():
    n = node('--max-linear-vel', '.01', '--impedance-target-linear-vel', '.01')
    for k in range(1, 7):
        gain = assist_step(n, k / 30.)
        np.testing.assert_allclose(gain, np.full(7, 4. * k / 6.), atol=1e-12)
    np.testing.assert_allclose(assist_step(n, 7 / 30.), 4.)
    assert n.args.max_linear_vel == n.args.impedance_target_linear_vel == .01
    assert n.args.max_joint_vel == .08
    assert n._imp_cfg.follow_max_error == .03
    assert n._imp_cfg.task_torque_limit == 10.
    np.testing.assert_array_equal(n._imp_cfg.tau_limit_per_joint, 14.)


def test_low_speed_threshold_change_increases_assist_without_increasing_gain_cap():
    gains = []
    for threshold in [.08, .02]:
        n = node()
        n.args.follow_min_qdot = threshold
        warm_assist(n)
        gains.append(n._imp_cfg.follow_kp.copy())
    np.testing.assert_allclose(gains[0], 1.)
    np.testing.assert_allclose(gains[1], 4.)


def test_assist_rises_per_axis_and_never_exceeds_requested_gain():
    n = node()
    limits = np.array([4., 2., 1., 0., 4., 2., 1.])
    n.args.follow_kp = limits
    previous = np.zeros(7)
    qdot = np.array([.02, -.02, .01, .02, 0., .005, -.01])
    for k in range(1, 10):
        n._refresh(np.zeros(7), {'qdot': qdot}, k / 30.)
        gain = n._imp_cfg.follow_kp.copy()
        assert np.all(gain - previous <= limits / 6. + 1e-12)
        assert np.all(gain <= limits * np.minimum(np.abs(qdot) / .02, 1.) + 1e-12)
        assert gain[3] == gain[4] == 0.
        previous = gain


def test_zero_reference_speed_releases_assist_immediately_then_resumes_with_ramp():
    n = node()
    warm_assist(n)
    np.testing.assert_array_equal(assist_step(n, 7 / 30., speed=0.), 0.)
    np.testing.assert_allclose(assist_step(n, 8 / 30.), 4. / 6.)


def test_safety_hold_releases_all_assist_and_restart_cannot_use_old_full_gain():
    n = node()
    warm_assist(n)
    n._pause_target_motion(7 / 30., immediate=True)
    np.testing.assert_array_equal(n._imp_cfg.follow_kp, 0.)
    np.testing.assert_array_equal(n._imp_follow_kp_active, 0.)
    np.testing.assert_array_equal(n._imp_last_follow_kp, 0.)
    np.testing.assert_allclose(assist_step(n, 8 / 30.), 4. / 6.)


def test_detector_dropout_decays_existing_gain_and_resumes_from_decayed_value():
    n = node()
    warm_assist(n)
    n._pause_target_motion(.35)
    np.testing.assert_allclose(n._imp_cfg.follow_kp, 2.)
    np.testing.assert_allclose(n._imp_follow_kp_active, 2.)
    np.testing.assert_allclose(assist_step(n, .35 + 1 / 30.), 2. + 4. / 6.)


def test_long_loss_clears_ramp_state_and_reengage_starts_without_assist():
    n = node()
    warm_assist(n)
    n._target_lost = False
    n._forget_lost_target(np.zeros(7), 1.)
    np.testing.assert_array_equal(n._imp_follow_kp_active, 0.)
    np.testing.assert_array_equal(n._imp_cfg.follow_kp, 0.)
    n._imp_follow_kp_active[:] = 4.  # stale prior engagement must be discarded
    n._engage(np.zeros(7), 2.)
    np.testing.assert_array_equal(n._imp_cfg.follow_kp, 0.)


def test_push_ahead_release_is_immediate_and_reengagement_is_ramped():
    n = node()
    warm_assist(n)
    n._imp_follow_q[0] = -.025
    n._refresh(np.zeros(7), {'qdot': np.full(7, .02)}, 7 / 30.)
    assert n._imp_cfg.follow_kp[0] == 0.
    n._refresh(np.zeros(7), {'qdot': np.full(7, -.02)}, 8 / 30.)
    assert n._imp_cfg.follow_kp[0] == pytest.approx(4. / 6.)


def test_lower_requested_assist_is_not_delayed_by_rise_limiter():
    n = node()
    warm_assist(n)
    np.testing.assert_allclose(assist_step(n, 7 / 30., speed=.005), 1.)


def test_long_outer_loop_gap_cannot_make_assist_jump_to_full_gain():
    n = node()
    np.testing.assert_allclose(assist_step(n, 1 / 30.), 4. / 6.)
    np.testing.assert_allclose(assist_step(n, 10.), 8. / 6.)


def test_zero_rise_time_is_an_explicit_legacy_override():
    n = node()
    n.args.follow_kp_rise_time = 0.
    np.testing.assert_array_equal(assist_step(n, 1 / 30.), 4.)


def test_yaml_assist_parameters_and_cli_overrides(tmp_path):
    from servo_delivery import parse_servo_args
    path = tmp_path / 'pipeline.yaml'
    path.write_text('servo:\n  follow_min_qdot: 0.03\n  follow_kp_rise_time: 0.4\n')
    args = parse_servo_args(['--config', str(path)])
    assert args.follow_min_qdot == .03
    assert args.follow_kp_rise_time == .4
    args = parse_servo_args(['--config', str(path), '--follow-min-qdot', '.02',
                            '--follow-kp-rise-time', '.2'])
    assert args.follow_min_qdot == .02
    assert args.follow_kp_rise_time == .2
