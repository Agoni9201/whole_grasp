"""Calibrated visual-task and execution checks; no camera/CAN/robot access."""
from types import SimpleNamespace as NS
from unittest.mock import Mock

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from models.nero_model import NeroPinocchioModel
from servo.hand_camera_center import HandCameraCenterServo
from servo.pbvs_pose_servo import PBVSParams
from control_backends.force_ctrl.mit_loop import NERO_JOINT_LIMITS, JOINT_LIMIT_BUFFER
from servo_delivery import (ServoDeliveryNode, FakeBackend, advance_orientation_reference,
                            parse_servo_args, validate_servo_args)
from real_nero_grasp.servo_cli import servo_parser


Q = np.array([.1, .2, -.1, 1.2, .05, -.1, .2])
K = np.array([[600., 0, 310.], [0, 610., 233.], [0, 0, 1.]])


def controller(*, rotation=None, translation=None):
    m = NeroPinocchioModel()
    T = np.eye(4)
    T[:3, :3] = Rotation.from_euler('xyz', [20, -35, 15], degrees=True).as_matrix() if rotation is None else rotation
    T[:3, 3] = [.035, -.02, .06] if translation is None else translation
    p = PBVSParams(desired_distance=.2, max_linear_vel=.03, max_angular_vel=.15,
                   max_joint_vel=.15, enable_orientation=False)
    p.marker_center_alpha = 1.
    c = HandCameraCenterServo(p, m, T, NERO_JOINT_LIMITS + [JOINT_LIMIT_BUFFER, -JOINT_LIMIT_BUFFER])
    c.set_camera_intrinsics(K, 640, 480)
    return c


def hand_in_camera(c, q, point):
    tcp, R = c._model.fk(q)
    return tcp + R @ (c.tcp_camera[:3, 3] + c.tcp_camera[:3, :3] @ point)


def features(c, q, hand):
    tcp, R = c._model.fk(q)
    camera = tcp + R @ c.tcp_camera[:3, 3]
    point = (R @ c.tcp_camera[:3, :3]).T @ (hand-camera)
    pixel = K[:2, :2] @ (point[:2]/point[2]) + K[:2, 2]
    return np.r_[(pixel - [319.5, 239.5]) / [600, 610] * .2,
                 np.linalg.norm(hand-tcp) - .2]


@pytest.mark.parametrize('camera_rotation', [np.eye(3), Rotation.from_euler('xyz', [90, 0, 90], degrees=True).as_matrix()])
def test_full_feature_jacobian_matches_fk_differences_with_camera_lever(camera_rotation):
    c = controller(rotation=camera_rotation)
    hand = hand_in_camera(c, Q, [.03, -.04, .3])
    A, error, *_ = c.feature_system(Q, hand)
    numeric = np.empty((3, 7))
    for axis in range(7):
        step = np.eye(7)[axis] * 1e-6
        numeric[:, axis] = (features(c, Q+step, hand) - features(c, Q-step, hand)) / 2e-6
    np.testing.assert_allclose(A, numeric, atol=2e-9)
    np.testing.assert_allclose(error, features(c, Q, hand), atol=1e-12)


@pytest.mark.parametrize('point', [[.035, -.025, .28], [-.025, .025, .24]])
def test_closed_loop_centers_and_approaches_using_translation_and_rotation(point):
    c = controller()
    q = Q.copy()
    hand = hand_in_camera(c, q, point)
    initial = features(c, q, hand)
    previous = np.zeros(7)
    twists = []
    initial_R = c._model.fk(q)[1].copy()
    for tick in range(300):
        out = c.update(q, hand, [-1, 0, 0], [0, 1, 0], stamp=tick/30.)
        assert not out.get('control_hold'), out.get('camera_center')
        assert not out['target_rejected']
        dq, twist = out['qdot'], out['tcp_twist']
        assert np.max(np.abs(dq)) <= .15+1e-8
        assert np.max(np.abs(dq-previous)) <= .5/30.+1e-8
        assert np.linalg.norm(twist[:3]) <= .03+1e-8
        assert np.linalg.norm(twist[3:]) <= .15+1e-8
        q += dq/30.
        assert np.all(q > c.joint_bounds[:, 0]) and np.all(q < c.joint_bounds[:, 1])
        previous = dq.copy()
        twists.append(twist)
    final = features(c, q, hand)
    assert np.linalg.norm(final[:2]) < np.linalg.norm(initial[:2])*.15
    assert abs(final[2]) < abs(initial[2])*.15
    assert np.max(np.linalg.norm(np.array(twists)[:, :3], axis=1)) > .005
    assert np.max(np.linalg.norm(np.array(twists)[:, 3:], axis=1)) > .02
    assert Rotation.from_matrix(c._model.fk(q)[1] @ initial_R.T).magnitude() > .03


@pytest.mark.parametrize('axis', range(7))
@pytest.mark.parametrize('side', [-1, 1])
def test_joint_damper_blocks_outward_motion_and_permits_return(axis, side):
    c = controller()
    q = np.mean(c.planning_bounds, axis=1)
    q[axis] = c.planning_bounds[axis, 1 if side == 1 else 0] + side*.005
    A = np.zeros((3, 7)); A[0, axis] = 1
    # A synthetic joint feature isolates each boundary, independent of geometry.
    J = np.zeros((6, 7))
    outward, status, _ = c._solve(q, A, [-side*.1, 0, 0], J, np.zeros(7), 1/30.)
    assert status == 'ok'
    assert outward[axis]*side <= 1e-9
    c.pause()
    inward, status, _ = c._solve(q, A, [side*.1, 0, 0], J, np.zeros(7), 1/30.)
    assert status == 'ok'
    assert inward[axis]*side < 0


def test_solver_failure_never_installs_unchecked_motion(monkeypatch):
    import servo.hand_camera_center as module
    c = controller()
    hand = hand_in_camera(c, Q, [.02, 0, .25])
    monkeypatch.setattr(module, 'minimize', lambda *a, **kw: NS(success=False, x=np.full(7, 99.)))
    out = c.update(Q, hand, [-1, 0, 0], stamp=1.)
    assert out['control_hold'] == 'solver_hold'
    np.testing.assert_array_equal(out['qdot'], 0)


def test_loss_freezes_and_reacquisition_has_no_old_velocity_jump():
    c = controller()
    hand = hand_in_camera(c, Q, [.02, .02, .26])
    for k in range(10):
        c.update(Q, hand, [-1, 0, 0], stamp=k/30.)
    lost = c.update(Q, None, stamp=.4)
    assert lost['target_lost']
    np.testing.assert_array_equal(lost['qdot'], 0)
    out = c.update(Q, hand, [-1, 0, 0], stamp=1.)
    assert not out.get('control_hold')
    assert np.max(np.abs(out['qdot'])) <= .5/30.+1e-8


def test_jump_rejection_and_reset_are_retained():
    c = controller()
    hand = hand_in_camera(c, Q, [.01, .01, .25])
    c.update(Q, hand, [-1, 0, 0], stamp=0.)
    out = c.update(Q, hand + [.4, 0, 0], [-1, 0, 0], stamp=1.)
    assert out['target_rejected']
    np.testing.assert_array_equal(out['qdot'], 0)


def test_bad_or_missing_camera_geometry_holds():
    c = controller()
    for point in [[0, 0, -.3], [0, 0, .01]]:
        c.update(Q, None, stamp=0.)
        out = c.update(Q, hand_in_camera(c, Q, point), [-1, 0, 0], stamp=1.)
        assert out.get('control_hold')
        np.testing.assert_array_equal(out['qdot'], 0)
    c.update(Q, None, stamp=2.)
    c._camera_K = None
    out = c.update(Q, hand_in_camera(c, Q, [0, 0, .25]), [-1, 0, 0], stamp=3.)
    assert out['control_hold'] == 'camera intrinsics unavailable'


def test_orientation_reference_has_rate_and_error_bounds_and_no_zero_speed_snap():
    previous = np.eye(3)
    actual = np.eye(3)
    for _ in range(100):
        result = advance_orientation_reference(previous, [0, .15, 0], actual, 1/30., .15, np.deg2rad(5))
        assert Rotation.from_matrix(result @ previous.T).magnitude() <= .15/30.+1e-12
        assert Rotation.from_matrix(result @ actual.T).magnitude() <= np.deg2rad(5)+1e-12
        previous = result
    pushed = Rotation.from_euler('x', 15, degrees=True).as_matrix()
    np.testing.assert_allclose(advance_orientation_reference(previous, [0, 0, 0], pushed, 1/30., .15, .08), previous)


def test_orientation_leash_handles_noncommuting_axes_without_rate_jumps():
    random = np.random.default_rng(23)
    for _ in range(200):
        actual = Rotation.random(random_state=random).as_matrix()
        previous = Rotation.from_rotvec(random.normal(size=3)*.03).as_matrix() @ actual
        result = advance_orientation_reference(previous, random.normal(size=3), actual, 1/30., .15, .08)
        assert Rotation.from_matrix(result @ previous.T).magnitude() <= .15/30.+1e-12
        old_error = Rotation.from_matrix(previous @ actual.T).magnitude()
        assert Rotation.from_matrix(result @ actual.T).magnitude() <= max(.08, old_error)+1e-12


def test_node_installs_rotating_reference_and_clips_follow_inside_planning_bounds():
    args = servo_parser().parse_args(['--target-source', 'mediapipe', '--hand-camera-center',
                                     '--servo-follow-only', '--max-joint-vel', '.15'])
    c = controller()
    target = NS(tcp_camera_transform=c.tcp_camera, last_diagnostics={})
    n = ServoDeliveryNode(FakeBackend(), args, NeroPinocchioModel(), target=target)
    n._engage(Q, 0.)
    n.servo.set_camera_intrinsics(K, 640, 480)
    hand = hand_in_camera(c, Q, [.02, -.02, .25])
    R0 = n._imp_cfg.R_target.copy()
    out = n.servo.update(Q, hand, [-1, 0, 0], stamp=1/30.)
    n._refresh(Q, out, 1/30.)
    assert Rotation.from_matrix(n._imp_cfg.R_target @ R0.T).magnitude() > 1e-7
    assert np.linalg.norm(n._imp_cfg.target_twist[3:]) > 0
    q = Q.copy(); q[6] = n.servo.planning_bounds[6, 1] - .001
    n._imp_follow_q = q + .03
    n._refresh(q, {'qdot': np.zeros(7), 'tcp_twist': np.zeros(6), 'center_control': True}, 2/30.)
    assert np.all(n._imp_cfg.q_follow <= n.servo.planning_bounds[:, 1])
    assert np.all(n._imp_cfg.q_follow >= n.servo.planning_bounds[:, 0])
    assert n._imp_cfg.task_torque_limit == 10.
    np.testing.assert_array_equal(n._imp_cfg.tau_limit_per_joint, 14.)
    n._pause_target_motion(.1, immediate=True)
    np.testing.assert_array_equal(n._imp_cfg.target_twist, 0)
    np.testing.assert_array_equal(n.servo._center_velocity, 0)


def test_center_mode_is_explicit_and_yaml_cli_override_works(tmp_path):
    assert not servo_parser().parse_args([]).hand_camera_center
    p = tmp_path/'pipeline.yaml'
    p.write_text('servo:\n  hand_camera_center: true\n  hand_center_gain: 1.5\n')
    args = parse_servo_args(['--config', str(p), '--hand-center-gain', '2.5'])
    assert args.hand_camera_center and args.hand_center_gain == 2.5


@pytest.mark.parametrize('extra', [[], ['--target-source', 'mediapipe', '--visual-orientation'],
    ['--target-source', 'mediapipe', '--hand-center-gain', '0'],
    ['--target-source', 'mediapipe', '--hand-center-max-joint-accel', '0'],
    ['--target-source', 'mediapipe', '--hand-offset-base', '0,.01,0']])
def test_invalid_center_configuration_rejected_before_hardware(extra):
    args = servo_parser().parse_args(['--hand-camera-center', *extra])
    args.handeye = __file__
    with pytest.raises(ValueError):
        validate_servo_args(args)


def test_camera_mount_property_matches_existing_base_transform_for_both_conventions():
    from targets.hand_target import HandTarget
    from targets.aruco_eye_in_hand_target import ArUcoEyeInHandTarget
    for mount in ['feedback_link7', 'tcp_offset']:
        c = controller()
        frame = ArUcoEyeInHandTarget.__new__(ArUcoEyeInHandTarget)
        frame.handeye_mount_frame, frame.T_mount_cam, frame.model = mount, c.tcp_camera, c._model
        t = HandTarget.__new__(HandTarget); t._frame = frame
        np.testing.assert_allclose(c._model.fk_to_4x4(Q) @ t.tcp_camera_transform,
                                   frame.base_mount_transform(Q) @ frame.T_mount_cam, atol=1e-12)


def test_backend_exposes_actual_configured_sender_buffer():
    from control_backends.sdk_force_ctrl_backend import SdkForceCtrlBackend
    b = SdkForceCtrlBackend.__new__(SdkForceCtrlBackend)
    b._force_ctrl = NS(_joint_limit_buffer=.17)
    np.testing.assert_allclose(b.get_servo_joint_bounds(), NERO_JOINT_LIMITS + [.17, -.17])


def test_retreat_uses_bounded_fixed_pose_task_without_camera_features():
    c = controller()
    tcp, R = c._model.fk(Q)
    R = R.copy()
    c._camera_K = None
    out = c.fixed_step(Q, tcp+[.01, 0, 0], R, 0.)
    assert not out.get('control_hold')
    assert out['center_control']
    assert out['tcp_twist'][0] > 0
    assert np.max(np.abs(out['qdot'])) <= .5/30.+1e-8


@pytest.mark.parametrize('camera_z', [.4, .8])
def test_far_hand_closed_loop_approach_keeps_existing_motion_limits_and_safety(camera_z):
    from servo.hand_follow_safety import HandFollowSafetyMonitor
    c = controller()
    c.params.desired_distance = .05
    m = HandFollowSafetyMonitor(c._model.clone(), position_leash=.015, joint_leash=.03)
    q = Q.copy()
    hand = hand_in_camera(c, q, [.035, -.025, camera_z])
    if camera_z == .8:
        center, radius = c._model.tcp_reach_bound()
        assert np.linalg.norm(hand-center) > radius+.05
    initial = None
    for tick in range(240):
        out = c.update(q, hand, stamp=tick/30.)
        assert not out.get('control_hold'), out.get('camera_center')
        decision = m.evaluate(q, out, tick/30., reference_tcp=c._model.fk(q)[0], reference_q=q)
        assert not decision.hold, decision
        if initial is None:
            initial = np.linalg.norm(out['error'])
        assert np.max(np.abs(out['qdot'])) <= .15+1e-8
        assert np.linalg.norm(out['tcp_twist'][:3]) <= .03+1e-8
        assert np.linalg.norm(out['tcp_twist'][3:]) <= .15+1e-8
        q += out['qdot']/30.
    assert initial > .2
    assert np.linalg.norm(out['error']) < initial-.10


@pytest.mark.parametrize('moving', [True, False])
@pytest.mark.parametrize('camera_z', [.4, .8])
def test_delivery_timer_uses_reference_tracking_and_stops_only_if_progress_stalls(monkeypatch, moving, camera_z):
    import servo_delivery as delivery
    c = controller()
    args = servo_parser().parse_args(['--target-source', 'mediapipe', '--hand-camera-center',
        '--gesture-handover', '--gesture-preview-grip-width-mm', '40',
        '--delivery-distance', '.05', '--max-joint-vel', '.15'])
    q = Q.copy()
    hand = hand_in_camera(c, q, [.035, -.025, camera_z])
    if camera_z == .8:
        center, radius = c._model.tcp_reach_bound()
        assert np.linalg.norm(hand-center) > radius+.05
    backend = FakeBackend()
    backend.get_joint_angles = lambda: q.copy()
    n = ServoDeliveryNode(backend, args, NeroPinocchioModel(),
                         target=NS(tcp_camera_transform=c.tcp_camera, last_diagnostics={}))
    n._engage(q, 100.)
    n.servo.set_camera_intrinsics(K, 640, 480)
    n.state, n._target_lost = 'TRACK_HAND', False
    n._last_target_time = 100.
    n._detect_latest = lambda *a: (hand.copy(), np.array([-1., 0, 0]), np.array([0., 1, 0]))
    clock = NS(now=100.)
    monkeypatch.setattr(delivery, 'time', NS(time=lambda: clock.now, monotonic=lambda: clock.now))
    for tick in range(150):
        clock.now = 100.+tick/30.
        n._servo_step()
        if moving:
            assert n._preview_status == 'TRACKING', n._preview_reason
            q += n.servo._center_velocity/30.
        elif tick >= 91:
            assert n._preview_reason.startswith('hand_approach_stalled_hold')
            np.testing.assert_array_equal(n._imp_cfg.target_twist, 0)
        tcp = n.model.fk(q)[0]
        assert np.linalg.norm(n._imp_cfg.target_tcp-tcp) <= args.impedance_max_position_error+.002
        assert np.max(np.abs(n._imp_cfg.q_follow-q)) <= args.follow_max_error_rad+.001
    assert n._gesture.phase == 'FOLLOW_RECEIVER'
    assert n.gripper is None  # Fake backend; never opens a device connection.


@pytest.mark.parametrize('camera_z', [.38, 1.])
def test_stable_far_jump_can_reacquire_without_teleporting_reference(camera_z):
    c = controller()
    args = servo_parser().parse_args(['--target-source', 'mediapipe', '--hand-camera-center',
                                     '--servo-follow-only', '--delivery-distance', '.05'])
    n = ServoDeliveryNode(FakeBackend(), args, NeroPinocchioModel(),
                         target=NS(tcp_camera_transform=c.tcp_camera, last_diagnostics={}))
    n._engage(Q, 0.)
    n.servo.set_camera_intrinsics(K, 640, 480)
    old = hand_in_camera(c, Q, [.02, 0., .25])
    new = hand_in_camera(c, Q, [.02, 0., camera_z])
    if camera_z == 1.:
        center, radius = n.model.tcp_reach_bound()
        assert np.linalg.norm(new-center) > radius+.05
    n.servo.params.marker_center_alpha = 1.
    normal, tangent = np.array([-1., 0, 0]), np.array([0., 1, 0])
    n.servo.update(Q, old, normal, tangent, stamp=0.)
    assert n.servo.update(Q, new, normal, tangent, stamp=1.)['target_rejected']
    previous = n._imp_cfg.target_tcp.copy()
    for tick in range(12):
        assert n._try_reacquire_rejected_target(Q, (new, normal, tangent), 1.+tick/30.) is None
    out = n._try_reacquire_rejected_target(Q, (new, normal, tangent), 1.45)
    assert out is not None and not out['target_rejected']
    assert np.linalg.norm(out['error']) > .2
    np.testing.assert_array_equal(n._imp_cfg.target_tcp, previous)


def test_diagnostics_expose_centering_and_angular_reference(tmp_path):
    from servo.diagnostics import ServoDiagnostics, _console_line
    writer = Mock(); writer.error = None; writer.dropped = 0
    d = ServoDiagnostics(tmp_path/'samples.jsonl', writer=writer)
    c = controller(); hand = hand_in_camera(c, Q, [.02, -.02, .25])
    out = c.update(Q, hand, [-1, 0, 0], stamp=0.)
    cfg = NS(target_tcp=c._model.fk(Q)[0], target_twist=out['tcp_twist'])
    d.observe(dict(q=Q, servo_out=out, debug={}), model=c._model, cfg=cfg,
              status='TRACKING', reason='test', stage='TRACK_HAND', now=0., wall_time=1.)
    row = writer.submit.call_args.args[0]
    np.testing.assert_array_equal(row['camera_center']['pixel_error'], out['camera_center']['pixel_error'])
    np.testing.assert_allclose(row['command_angular_rad_s'], out['tcp_twist'][3:])
    assert 'center_error_px=' in _console_line(row)
