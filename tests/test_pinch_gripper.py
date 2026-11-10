from types import SimpleNamespace as NS
import sys
import time

import numpy as np
import pytest

from handoff.pinch_gripper import (
    AutoPinchHand, AutoPinchMapping, GripperOnlyLink, GripperResponseWatch,
    GripperUnresponsive, Pinch, PinchMapping, WidthRamp, measure_pinch,
)
from scripts.test_pinch_gripper import joints_stationary, parse_args
from targets.mediapipe_hand_landmarker import MediaPipeHandLandmarker


def hand(ratio=.5, scale=1., side='Right'):
    points = np.zeros((21, 3))
    points[:, :2] = .5
    points[5, :2], points[17, :2] = [.4, .6], [.6, .6]
    points[4, :2], points[8, :2] = [.4, .4], [.4 + .2 * ratio, .4]
    points[:, :2] = .5 + scale * (points[:, :2] - .5)
    return NS(landmarks_normalized=points, handedness=side, handedness_score=.99)


def test_pinch_uses_rgb_only_and_normalizes_hand_scale():
    # Neither metric palm position nor valid depth is needed.
    a = measure_pinch([hand(.65)], (480, 640, 3))
    b = measure_pinch([hand(.65, .65)], (480, 640, 3))
    assert a.ratio == pytest.approx(.65)
    assert b.ratio == pytest.approx(a.ratio)


@pytest.mark.parametrize('observations,side', [([], None), ([hand(), hand()], None),
                         ([hand(side='Left')], 'Right')])
def test_ambiguous_missing_or_wrong_hand_cannot_control(observations, side):
    with pytest.raises(ValueError):
        measure_pinch(observations, (480, 640, 3), side)


def test_offscreen_and_edge_on_geometry_is_rejected():
    for observation in (hand(scale=.01), hand(ratio=4.)):
        with pytest.raises(ValueError):
            measure_pinch([observation], (480, 640, 3))


@pytest.mark.parametrize('ratio,fraction', [(-1., 0.), (.1, 0.), (.3, .25),
                         (.5, .5), (.7, .75), (.9, 1.), (2., 1.)])
def test_aperture_is_proportional_at_intermediate_positions(ratio, fraction):
    mapping = PinchMapping(.1, .9)
    assert mapping.fraction(Pinch(ratio, 'Right')) == pytest.approx(fraction)


def test_explicit_fixed_mapping_can_still_be_supplied():
    mapping = PinchMapping()
    for ratio in (0., .25, .5, .75, 1.):
        assert mapping.fraction(measure_pinch([hand(ratio)], (480, 640, 3))) == pytest.approx(ratio)


def test_bad_mapping_limits_are_rejected():
    for limits in ((.5, .5), (.8, .2), (-.1, .8), (0., float('nan'))):
        with pytest.raises(ValueError):
            PinchMapping(*limits)
    with pytest.raises(ValueError):
        PinchMapping().fraction(Pinch(float('nan'), 'Right'))


def test_automatic_endpoints_fix_closed_offset_and_early_full_open_without_keys():
    learner = AutoPinchMapping()
    # Starting with an open hand must not set the closed endpoint.
    for i in range(40):
        assert learner.observe(Pinch(1.5, 'Left'), i/30) is None
    assert learner.phase == 'close'
    for i in range(40, 70):
        assert learner.observe(Pinch(.23 + .005*(i % 2), 'Left'), i/30) is None
    assert learner.phase == 'open'
    for i in range(70, 100):
        fraction = learner.observe(Pinch(2.4 + .005*(i % 2), 'Left'), i/30)
    assert learner.phase == 'ready' and fraction == 1.
    assert learner.observe(Pinch(.23, 'Left'), 4.) == 0.
    assert 0. < learner.observe(Pinch(1., 'Left'), 4.1) < .5
    assert learner.observe(Pinch((learner.mapping.closed + learner.mapping.opened)/2, 'Left'),
                           4.2) == pytest.approx(.5)
    learned = learner.mapping
    learner.pause()
    learner.observe(Pinch(3., 'Left'), 5.)
    assert learner.mapping is learned  # No changing the scale during control/reacquisition.


def test_calibration_needs_consecutive_stable_holds_not_a_sweep_or_one_outlier():
    learner = AutoPinchMapping()
    for i in range(40):
        learner.observe(Pinch(.23, 'Right'), i/30)
    assert learner.phase == 'open'
    # Opening gradually should not capture a premature full-open endpoint.
    for i in range(40, 100):
        assert learner.observe(Pinch(.9 + .05*(i-40), 'Right'), i/30) is None
    assert learner.mapping is None
    learner.pause()  # Losing the hand interrupts the hold, not the learned closed pose.
    for i in range(100, 110):
        assert learner.observe(Pinch(2.5, 'Right'), i/30) is None
    learner.pause()
    for i in range(110, 120):
        assert learner.observe(Pinch(2.5, 'Right'), i/30) is None
    assert learner.mapping is None


def test_response_watch_latches_on_unmoving_feedback_but_accepts_slow_progress():
    watch = GripperResponseWatch()
    watch.observe(.080, .085, 0.)
    with pytest.raises(GripperUnresponsive, match='NO RESPONSE'):
        watch.observe(.080, .085, 1.)
    watch.pause()
    for i in range(200):
        actual = .085 - i*.00002
        watch.observe(actual - .005, actual, i/30)
    watch.observe(actual, actual, 8.)  # At target: holding a constant opening is valid.
    assert watch.since is None


def test_five_mm_lead_can_cross_a_deadband_that_blocked_the_old_two_mm_limit():
    actual = .084959  # Exact constant feedback in the user's log.
    ramp = WidthRamp()
    ramp.reset(actual, 0.)
    commands = []
    watch = GripperResponseWatch()
    for i in range(1, 100):
        command = ramp.advance(0., i/30, actual)
        commands.append(command)
        # A simulated 3mm deadband is a regression model, not a measured hardware spec.
        if actual - command > .003:
            actual = command
        watch.observe(command, actual, i/30)
    assert min(commands) < .02
    assert actual == pytest.approx(0., abs=.003)


def test_automatic_hand_selection_and_recovery_keep_the_same_hand():
    tracker = AutoPinchHand()
    right, left = Pinch(.5, 'Right'), Pinch(.5, 'Left')
    assert not tracker.observe(right)
    assert not tracker.observe(right)
    assert tracker.observe(right)
    assert tracker.side == 'Right'
    tracker.lost()
    for _ in range(5):
        assert not tracker.observe(left)
    assert tracker.side == 'Right'
    assert not tracker.observe(right)
    assert not tracker.observe(right)
    assert tracker.observe(right)


def test_first_hand_must_be_consistent_before_it_is_selected():
    tracker = AutoPinchHand()
    assert not tracker.observe(Pinch(.5, 'Left'))
    assert not tracker.observe(Pinch(.5, 'Right'))
    assert not tracker.observe(Pinch(.5, 'Left'))
    assert tracker.side is None


def test_full_stroke_takes_five_seconds_to_open_and_two_to_close():
    ramp = WidthRamp()
    ramp.reset(0., 0.)
    for i in range(1, 151):
        ramp.advance(1., i / 30)
    assert ramp.width == pytest.approx(.085)
    for i in range(1, 61):
        ramp.advance(0., 5 + i / 30)
    assert ramp.width == pytest.approx(0.)


def test_intermediate_gesture_stops_at_its_proportional_width():
    ramp = WidthRamp()
    ramp.reset(0., 0.)
    for i in range(1, 301):
        ramp.advance(.5, i/30)
    assert ramp.width == pytest.approx(.0425)


def test_stall_has_no_catchup_jump_and_resume_starts_from_actual_opening():
    ramp = WidthRamp()
    ramp.reset(.040, 0.)
    assert ramp.advance(1., 10.) <= .040 + .085/5*.1 + 1e-12
    ramp.reset(.043, 20.)
    assert ramp.advance(0., 20.) == pytest.approx(.043)


def test_pending_target_is_bounded_by_measured_width():
    ramp = WidthRamp(max_lead=.002)
    ramp.reset(.040, 0.)
    for i in range(1, 101):
        ramp.advance(1., i/30, measured_width=.040)
    assert ramp.width == pytest.approx(.042)
    with pytest.raises(ValueError):
        ramp.advance(1., 4., measured_width=.060)


class FakeRobot:
    def __init__(self):
        self.calls = []
        self._config = {'joint_limits': {f'joint{i}': [-np.pi, np.pi] for i in range(1, 8)}}
        self.OPTIONS = NS(EFFECTOR=NS(AGX_GRIPPER='agx_gripper'))
        wrap = lambda msg: NS(msg=msg, timestamp=time.time())
        self._parser = NS(**{name: wrap(NS(**{f'joint_{i}': .1 for i in indices}))
            for name, indices in [('joint_12', (1, 2)), ('joint_34', (3, 4)),
                                  ('joint_56', (5, 6)), ('joint_7', (7,))]})
        self.state = wrap(NS(ctrl_mode=1, arm_status=0, err_code=0, motion_status=0, mode_feedback=1))
        faults = {name: False for name in ('voltage_too_low', 'motor_overheating',
            'driver_overcurrent', 'driver_overheating', 'sensor_status', 'driver_error_status')}
        faults['driver_enable_status'] = True
        self.feedback = wrap(NS(value=.04, force=.01, mode='width', foc_status=NS(**faults)))
        self.gripper = NS(get_gripper_status=lambda: self.feedback,
                          move_gripper_m=lambda **kw: self.calls.append(('gripper', kw)))

    def connect(self):
        self.calls.append(('connect',))

    def init_effector(self, name):
        self.calls.append(('init_effector', name))
        return self.gripper

    def disconnect(self):
        self.calls.append(('disconnect',))

    def get_arm_status(self):
        return self.state

    def get_joints_enable_status_list(self):
        return [True] * 7


def test_sdk_lifecycle_writes_only_gripper_and_does_not_disable_on_exit(monkeypatch):
    robot = FakeRobot()
    monkeypatch.setitem(sys.modules, 'pyAgxArm', NS(
        AgxArmFactory=NS(create_arm=lambda cfg: robot), create_agx_arm_config=lambda **kw: kw))
    link = GripperOnlyLink.connect()
    link.anchor = link.read().joints.copy()
    link.send(.041)
    link.close()
    assert [call[0] for call in robot.calls] == ['connect', 'init_effector', 'gripper', 'disconnect']
    assert robot.calls[2][1] == {'value': .041, 'force': .50}


def test_installed_sdk_encodes_width_force_and_enable_without_arm_commands():
    pytest.importorskip('pyAgxArm')
    from pyAgxArm.protocols.can_protocol.drivers.effector.agx_gripper.default.driver import Driver
    from pyAgxArm.protocols.can_protocol.drivers.effector.agx_gripper.default.parser import Codec

    payloads = []
    driver = Driver.__new__(Driver)  # No connect, CAN socket or hardware object.
    driver._send_msg = lambda msg: payloads.append(bytes(Codec().encode_159_gripper_ctrl(msg)))
    driver.move_gripper_m(value=.079959, force=.50)
    payload, = payloads
    assert len(payload) == 8
    assert int.from_bytes(payload[:4], 'big', signed=True) == 79959
    assert int.from_bytes(payload[4:6], 'big') == 500
    assert payload[6:] == bytes([1, 0])  # Gripper enable/width; no zero calibration.


def test_installed_sdk_first_can_mode_packet_preserves_motion_mode_at_zero_speed():
    pytest.importorskip('pyAgxArm')
    from pyAgxArm.protocols.can_protocol.drivers.nero.versions.v111.driver import Driver
    from pyAgxArm.protocols.can_protocol.drivers.nero.default.parser import Codec
    from pyAgxArm.protocols.can_protocol.msgs.nero.versions import ArmMsgModeCtrlV111

    payloads = []
    driver = Driver.__new__(Driver)
    driver._msg_mode = ArmMsgModeCtrlV111()
    driver._send_msg = lambda msg: payloads.append(bytes(Codec().encode_151_mode_ctrl(msg)))
    driver.set_speed_percent(0)
    driver.set_motion_mode('j')
    assert payloads[0][:4] == bytes([1, 255, 0, 0])  # CAN, preserve mode, ZERO speed, position mode.
    assert payloads[1][:4] == bytes([1, 1, 0, 0])  # Explicit J selection still at ZERO speed.


def test_observed_control_frame_is_logged_but_never_treated_as_actuator_ack():
    robot = FakeRobot()
    # A conflicting control frame can be received independently of actual width.
    control = NS(timestamp=time.time(), msg=NS(value=.08, force=.5, status_code=1))
    robot.gripper.get_gripper_ctrl_states = lambda: control
    link = GripperOnlyLink(robot, robot.gripper)
    feedback = link.read()
    assert feedback.width == .04 and link.observed_control['width_mm'] == 80.
    assert link.device_status['gripper_enabled'] is True
    control.timestamp -= 1.
    assert link.read().width == .04
    assert link.observed_control is None  # Missing/stale echo does not prevent reading.


@pytest.mark.parametrize('failure', [None, 'moving', 'no_ack', 'drift_after_restore', 'clipped_target'])
def test_explicit_can_preparation_clears_old_targets_and_confirms_current_hold(monkeypatch, failure):
    import handoff.pinch_gripper as control
    robot = FakeRobot()
    robot.state.msg.ctrl_mode = 3
    robot.state.msg.motion_status = 1
    initial = np.full(7, .1)
    wall_base = time.time()
    clock = NS(now=0.)

    def sleep(seconds):
        clock.now += seconds
        if failure == 'moving' and not robot.calls:
            robot._parser.joint_7.msg.joint_7 += np.deg2rad(.03)
        for wrapped in (robot.state, robot.feedback, *vars(robot._parser).values()):
            wrapped.timestamp = wall_base + clock.now

    def speed(percent):
        robot.calls.append(('speed', percent))
        robot.state.msg.ctrl_mode = 1
        if failure == 'drift_after_restore' and percent == 10:
            robot._parser.joint_7.msg.joint_7 += np.deg2rad(.6)

    def move_j(target):
        robot.calls.append(('hold', target))
        if failure != 'no_ack':
            robot.state.msg.motion_status = 0

    robot.set_speed_percent = speed
    robot.set_motion_mode = lambda mode: robot.calls.append(('mode', mode))
    robot.move_j = move_j
    robot._send_msg = lambda msg: robot.calls.append(('terminate', msg.track_ctrl, msg.emergency_stop))
    if failure == 'clipped_target':
        robot._config['joint_limits']['joint7'] = [-.05, .05]
    monkeypatch.setattr(control, 'time', NS(time=lambda: wall_base+clock.now,
                                          monotonic=lambda: clock.now, sleep=sleep))
    link = GripperOnlyLink(robot, robot.gripper)
    events = []
    if failure:
        with pytest.raises((ValueError, RuntimeError)):
            link.prepare_can_hold(record=lambda event, **data: events.append((event, data)))
        if failure in ('moving', 'clipped_target'):
            assert not robot.calls  # Validate stationary positions/limits before ANY write.
        else:
            assert robot.calls[-1] == ('speed', 0)
        assert not any(call[0] == 'gripper' for call in robot.calls)
    else:
        link.prepare_can_hold(record=lambda event, **data: events.append((event, data)))
        assert robot.calls[:3] == [('terminate', 6, 0)]*3
        assert robot.calls[3:8] == [('speed', 0), *[('terminate', 6, 0)]*3, ('mode', 'j')]
        assert robot.calls[8] == ('hold', initial.tolist())
        assert robot.calls[9] == ('speed', 10)
        assert events[-1][1]['stage'] == 'confirmed'
        assert clock.now >= .8
        assert np.array_equal(link.anchor, initial)
        # Only after CAN mode AND current hold are confirmed may aperture be sent.
        link.send(.041)
        assert robot.calls[-1] == ('gripper', {'value': .041, 'force': .5})
    assert all(call[0] not in ('enable', 'disable', 'reset') for call in robot.calls)


def test_can_preparation_does_not_switch_an_already_working_can_hold():
    robot = FakeRobot()
    GripperOnlyLink(robot, robot.gripper).prepare_can_hold()
    assert not robot.calls


@pytest.mark.parametrize('fault', ['gripper_stale', 'missing_joint7', 'joint_stale',
                                  'mit_mode', 'arm_fault', 'gripper_fault', 'arm_drift', 'ethernet_control'])
def test_invalid_hardware_feedback_prevents_writes(fault):
    robot = FakeRobot()
    link = GripperOnlyLink(robot, robot.gripper)
    link.anchor = link.read().joints.copy()
    if fault == 'gripper_stale':
        robot.feedback.timestamp -= 2.
    elif fault == 'missing_joint7':
        robot._parser.joint_7 = None
    elif fault == 'joint_stale':
        robot._parser.joint_12.timestamp -= 2.
    elif fault == 'mit_mode':
        robot.state.msg.mode_feedback = 6
    elif fault == 'arm_fault':
        robot.state.msg.arm_status = 7
    elif fault == 'gripper_fault':
        robot.feedback.msg.foc_status.driver_overcurrent = True
    elif fault == 'ethernet_control':
        robot.state.msg.ctrl_mode = 3
    else:
        robot._parser.joint_7.msg.joint_7 += .1
    with pytest.raises((ValueError, AttributeError, TypeError)):
        link.send(.041)
    assert robot.calls == []


def test_not_at_target_flag_is_not_used_as_a_velocity_measurement():
    robot = FakeRobot()
    robot.state.msg.motion_status = 1
    link = GripperOnlyLink(robot, robot.gripper)
    link.anchor = link.read().joints.copy()
    assert link.device_status['motion_status'] == 1
    link.send(.041)
    assert robot.calls == [('gripper', {'value': .041, 'force': .50})]


def test_joint_history_distinguishes_static_jitter_from_actual_motion_and_stale_samples():
    history = [(i/30, np.zeros(7)) for i in range(8)]
    history[-1][1][6] = np.deg2rad(.005)  # Scale observed in the read-only hardware sample.
    assert joints_stationary(history, 7/30)
    history[-1][1][6] = np.deg2rad(.6)
    assert not joints_stationary(history, 7/30)
    assert not joints_stationary(history, 2.)
    assert not joints_stationary(history[:3], 2/30)


def test_missing_enable_flags_do_not_count_as_all_enabled():
    robot = FakeRobot()
    robot.get_joints_enable_status_list = lambda: []
    link = GripperOnlyLink(robot, robot.gripper)
    with pytest.raises(ValueError, match='joint_enabled'):
        link.read()


@pytest.mark.parametrize('width', [-.01, .09, float('nan'), .05])
def test_width_bounds_and_lead_cap_are_checked_at_send_boundary(width):
    robot = FakeRobot()
    link = GripperOnlyLink(robot, robot.gripper)
    link.anchor = link.read().joints.copy()
    with pytest.raises(ValueError):
        link.send(width)
    assert robot.calls == []


@pytest.mark.parametrize('args', [['--force', 'nan'], ['--force', '.6'],
    ['--max-width-mm', '100'], ['--open-seconds', '0'], ['--max-lead-mm', 'nan'],
    ['--pinch-open-ratio', '0'], ['--pinch-closed-ratio', 'nan'], ['--max-lead-mm', '6']])
def test_invalid_cli_limits_rejected_before_hardware(args):
    with pytest.raises(SystemExit) as exc:
        parse_args(args)
    assert exc.value.code == 2


def fake_detector():
    detector = MediaPipeHandLandmarker.__new__(MediaPipeHandLandmarker)
    observation = hand()
    observation.landmarks_normalized[0, :2] = [.5, .8]
    observation.landmarks_normalized[9, :2] = [.5, .5]
    landmarks = [NS(x=x, y=y, z=z) for x, y, z in observation.landmarks_normalized]
    result = NS(hand_landmarks=[landmarks], handedness=[[NS(category_name='Right', score=.99)]])
    detector._detector = NS(detect_for_video=lambda *args: result)
    detector._mp = NS(ImageFormat=NS(SRGB=1), Image=lambda **kwargs: kwargs)
    detector._last_timestamp_ms = -1
    detector.depth_radius, detector.min_depth_m, detector.max_depth_m = 1, .1, 2.
    return detector


def test_rgb_detector_returns_landmarks_without_inventing_metric_depth():
    observation, = fake_detector().detect(np.zeros((60, 80, 3), dtype=np.uint8))
    assert observation.landmarks_pixel.shape == (21, 2)
    assert observation.handedness == 'Right'
    assert observation.palm_center_camera_m is None
    assert observation.valid_depth_landmarks == 0
    assert np.isnan(observation.palm_depth_m)


def test_existing_rgbd_detector_still_returns_metric_palm():
    detector = fake_detector()
    observation, = detector.detect(np.zeros((60, 80, 3), dtype=np.uint8),
        np.full((60, 80), .4), np.array([[100., 0., 40.], [0., 100., 30.], [0., 0., 1.]]))
    assert observation.valid_depth_landmarks == 5
    assert observation.palm_center_camera_m[2] == pytest.approx(.4)


@pytest.mark.parametrize('execute,arm_drift,unresponsive,route', [(False, False, False, 'can'),
    (True, False, False, 'can'), (True, True, False, 'can'), (True, False, True, 'can'),
    (True, False, False, 'ethernet'), (True, False, False, 'prepare'),
    (True, False, False, 'prepare_fail'), (True, False, False, 'lost')])
def test_entrypoint_auto_follows_without_keys_and_handles_loss(
        monkeypatch, tmp_path, execute, arm_drift, unresponsive, route):
    import json
    import scripts.test_pinch_gripper as app

    clock = NS(frame=-1, now=0.)
    robot = FakeRobot()
    # Reproduce the actual report: no arm fault, static joints, arrival flag=1.
    robot.state.msg.motion_status = 1
    robot.state.msg.ctrl_mode = 3 if route in ('ethernet', 'prepare', 'prepare_fail') else 1
    robot.feedback.msg.value = .084959
    sent = []

    def move(**kw):
        sent.append((clock.frame, kw['value']))
        if not unresponsive:
            robot.feedback.msg.value = kw['value']  # Simulated feedback, not a real SDK.

    robot.gripper.move_gripper_m = move
    link = GripperOnlyLink(robot, robot.gripper)

    def prepare(record):
        robot.calls.append(('prepare_can_hold',))
        if route == 'prepare_fail':
            raise RuntimeError('CAN current-position hold not confirmed')
        robot.state.msg.ctrl_mode = 1
        robot.state.msg.motion_status = 0
        record('can_hold_prepare', stage='confirmed')

    link.prepare_can_hold = prepare  # Real preparation ordering is tested separately above.

    def connect(**kwargs):
        assert execute, 'preview must never connect to CAN'
        return link

    class Camera:
        def __init__(self, **kwargs):
            assert kwargs['enable_depth'] is False

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.close()

        def close(self):
            pass

        def read(self):
            clock.frame += 1
            clock.now = clock.frame / 30
            if route == 'lost' and clock.frame == 130:
                robot.state.msg.ctrl_mode = 3
            if clock.frame == 140:
                robot.feedback.msg.value = .044
                if arm_drift:
                    robot._parser.joint_7.msg.joint_7 += .1
            for wrapped in (robot.state, robot.feedback, *vars(robot._parser).values()):
                wrapped.timestamp = time.time()
            return NS(bgr=np.zeros((480, 640, 3), dtype=np.uint8), sequence=clock.frame,
                      hardware_ms=clock.now*1000, received_monotonic=clock.now)

    class Detector:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def detect(self, bgr, **kwargs):
            if 140 <= clock.frame < 150 or clock.frame >= 230:
                return []
            ratio = .23 if clock.frame <= 44 else (2.4 if clock.frame <= 79 else 1.315)
            observation = hand(ratio, side='Left' if 150 <= clock.frame < 160 else 'Right')
            observation.landmarks_pixel = np.rint(
                observation.landmarks_normalized[:, :2] * [639, 479]).astype(int)
            return [observation]

    keys = {235: ord('q')}  # No calibration/start/resume key is ever pressed.
    monkeypatch.setattr(app, 'create_camera', Camera)
    monkeypatch.setattr(app, 'MediaPipeHandLandmarker', Detector)
    monkeypatch.setattr(app.GripperOnlyLink, 'connect', connect)
    monkeypatch.setattr(app, 'time', NS(monotonic=lambda: clock.now))
    monkeypatch.setattr(app.cv2, 'imshow', lambda *args: None)
    monkeypatch.setattr(app.cv2, 'waitKey', lambda *args: keys.get(clock.frame, -1))
    monkeypatch.setattr(app.cv2, 'getWindowProperty', lambda *args: 1.)
    monkeypatch.setattr(app.cv2, 'destroyAllWindows', lambda: None)
    args = ['--log-dir', str(tmp_path)] + (['--execute'] if execute else [])
    if route in ('prepare', 'prepare_fail', 'lost'):
        args += ['--prepare-can-hold']
    assert app.main(args) == (1 if unresponsive or route in ('lost', 'prepare_fail') else 0)
    rows = [json.loads(line) for line in next(tmp_path.glob('*/samples.jsonl')).read_text().splitlines()]
    samples = [row for row in rows if row['type'] == 'sample']
    if route in ('ethernet', 'prepare_fail'):
        assert not sent and not any(row['active'] for row in samples)
        if route == 'ethernet':
            assert any('CAN control required: ctrl_mode=3' in row['reason'] for row in samples)
            assert robot.calls == [('disconnect',)]  # No takeover without the explicit flag.
        else:
            assert any(row['fault'] == 'CAN current-position hold not confirmed' for row in samples)
            assert robot.calls == [('prepare_can_hold',), ('disconnect',)]  # Failure is not retried.
        return
    assert any(row['active'] for row in samples)
    assert all(not row['active'] for row in samples if row['mapping_phase'] != 'ready')
    if execute and route != 'prepare':
        assert any(row['active'] and row['device_status']['motion_status'] == 1 for row in samples)
    assert all(not row['active'] for row in samples if 140 <= row['sequence'] < 160)
    assert samples[-1]['active'] is False
    if unresponsive or route == 'lost':
        faults = [row for row in samples if row['fault']]
        assert faults and faults[0]['reason'].startswith('CAN control lost' if route == 'lost' else 'NO RESPONSE')
        assert all(not row['active'] for row in faults)
        assert not any(frame > faults[0]['sequence'] for frame, _ in sent)
    else:
        assert samples[-1]['reason'] == 'show exactly one hand'
        assert samples[-1]['ramped_mm'] == pytest.approx(42.5, abs=.02)
    if execute:
        assert sent
        assert not any(140 <= frame < 162 for frame, _ in sent)
        assert min(frame for frame, _ in sent) >= 69  # No writes during automatic learning.
        if not unresponsive:
            assert sent[-1][1] == pytest.approx(.0425, abs=2e-5)
            plateau = [(frame, width) for frame, width in sent if 115 <= frame < 140]
            assert len(plateau) >= (3 if route == 'lost' else 5)
            assert all(width == pytest.approx(.0425, abs=2e-5) for _, width in plateau)
            assert max(np.diff([frame for frame, _ in plateau])) <= 4
        resumed = [(frame, width) for frame, width in sent if frame >= 160]
        if arm_drift or unresponsive or route == 'lost':
            assert not resumed  # Cannot hide arm motion by resetting its anchor.
        else:
            assert resumed[0] == pytest.approx((162, .044))
        expected = [('prepare_can_hold',)] if route == 'prepare' else []
        assert robot.calls == expected + [('disconnect',)]  # No automatic mode restoration on exit.
    else:
        assert not sent and not robot.calls
