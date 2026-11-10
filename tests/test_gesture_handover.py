"""Gesture policy/role regressions. No CAN sockets, cameras or physical motion."""
from types import SimpleNamespace as NS
import time

import numpy as np
import pytest

from handoff.gesture_handover import GestureHandover, read_opening
from handoff.pinch_gripper import Pinch, PinchMapping, measure_pinch
from targets.hand_selection import HandRoles, filter_hand_observations


def observation(side='Left', scale=1., ratio=.5):
    points = np.full((21, 3), .5)
    points[5, :2], points[17, :2] = [.4, .6], [.6, .6]
    points[4, :2], points[8, :2] = [.4, .4], [.4+.2*ratio, .4]
    points[:, :2] = .5+scale*(points[:, :2]-.5)
    return NS(handedness=side, handedness_score=.99, landmarks_normalized=points,
              landmarks_pixel=np.rint(points[:, :2]*[639, 479]).astype(int))


def test_tiny_false_hand_is_filtered_before_counting_but_two_plausible_hands_are_not():
    real, tiny = observation(), observation('Right', scale=.02)
    assert measure_pinch([real, tiny], (480, 640), 'Left').side == 'Left'
    accepted, rejected = filter_hand_observations([real, tiny], (480, 640))
    assert len(accepted) == 1 and rejected[0]['reason'] == 'palm too small or edge-on'
    # A real, smaller hand still passes the absolute geometry check.
    with pytest.raises(ValueError, match='exactly one'):
        measure_pinch([real, observation('Right', scale=.2)], (480, 640))


def test_first_hand_identity_is_retained_with_both_hands_and_after_receiver_loss():
    roles = HandRoles()
    left, right = observation(), observation('Right')
    assert roles.select([left], []) is None
    assert roles.select([right], []) is None  # Side flicker cannot establish a role.
    assert roles.select([left], []) is None
    assert roles.select([left], []) is None
    assert roles.select([left], []) is left
    assert roles.receiver == 'Left' and roles.control == 'Right'
    assert roles.select([right, left], []) is left
    assert roles.control_observations == [right]
    assert roles.select([right], []) is None and roles.presence == 'absent'
    assert roles.receiver == 'Left'
    assert roles.select([right], [dict(side='Left', reason='invalid hand landmarks')]) is None
    assert roles.presence == 'unknown'  # Invalid depth/geometry is not a release cue.


def enter_handover(policy):
    for i in range(int(np.ceil(policy.stable_seconds*30))+1):
        policy.observe_receiver('visible', .05, True, i/30, position_base=[0., 0., .05])
    assert policy.phase == 'GESTURE_LEARN'


@pytest.mark.parametrize('kind', ['far', 'depth_invalid', 'unknown', 'short_near', 'off_center'])
def test_handover_requires_visible_centered_near_receiver_with_valid_base_position(kind):
    policy = GestureHandover(.04)
    count = 60 if kind == 'short_near' else 121
    for i in range(count):
        policy.observe_receiver('unknown' if kind == 'unknown' else 'visible',
            .2 if kind == 'far' else .05, kind != 'off_center', i/30,
            position_base=None if kind == 'depth_invalid' else [0., 0., .05])
    for i in range(count, count+20):
        policy.observe_receiver('absent', None, False, i/30)
    assert policy.phase == 'FOLLOW_RECEIVER'


@pytest.mark.parametrize('rate', [10, 30, 60])
def test_three_seconds_of_stillness_latch_without_waiting_for_receiver_to_disappear(rate):
    p = GestureHandover(.04)
    for i in range(3*rate):
        # Small 3D measurement variation is permitted, rather than exact equality.
        offset = .002*(-1)**i
        p.observe_receiver('visible', .05, True, i/rate,
                           position_base=[offset, offset, .05+offset])
        assert p.phase == 'FOLLOW_RECEIVER'
    p.observe_receiver('visible', .05, True, 3., position_base=[0., 0., .05])
    assert p.phase == 'GESTURE_LEARN'
    assert p.snapshot()['handover_ready'] and p.still_elapsed == pytest.approx(3.)


@pytest.mark.parametrize('speed', [.005, .02])
def test_arm_tracking_a_moving_hand_cannot_make_the_hand_appear_stationary(speed):
    p = GestureHandover(.04)
    for i in range(180):
        # TCP follows at the same distance with zero image error. Only the BASE
        # position reveals continued hand movement, even at just 5mm/second.
        p.observe_receiver('visible', .05, True, i/30,
                           position_base=[speed*i/30, 0., .05])
    assert p.phase == 'FOLLOW_RECEIVER' and not p.snapshot()['handover_ready']


@pytest.mark.parametrize('interruption', ['far', 'absent', 'depth_invalid', 'off_center', 'frame_gap'])
def test_interruption_before_readiness_requires_three_new_continuous_seconds(interruption):
    p = GestureHandover(.04)
    for i in range(85):
        p.observe_receiver('visible', .05, True, i/30, position_base=[0., 0., .05])
    if interruption != 'frame_gap':
        p.observe_receiver('absent' if interruption == 'absent' else 'visible',
            .2 if interruption == 'far' else .05, interruption != 'off_center', 2.85,
            position_base=None if interruption == 'depth_invalid' else [0., 0., .05])
    # A 0.3s gap also resets continuity even if no lost tick ran during it.
    for i in range(90):
        p.observe_receiver('visible', .05, True, 3.2+i/30, position_base=[0., 0., .05])
        assert p.phase == 'FOLLOW_RECEIVER'
    p.observe_receiver('visible', .05, True, 6.2, position_base=[0., 0., .05])
    assert p.phase == 'GESTURE_LEARN'


@pytest.mark.parametrize('presence', ['absent', 'unknown', 'visible'])
def test_receiver_occlusion_depth_failure_or_reappearance_cannot_cancel_latched_readiness(presence):
    p = GestureHandover(.04)
    enter_handover(p)
    for i in range(200):
        p.observe_receiver(presence, None, False, 4.+i/30)
    assert p.phase == 'GESTURE_LEARN' and p.snapshot()['handover_ready']
    assert p.mapping.phase != 'ready'  # No control hand yet; no gripper release.


@pytest.mark.parametrize('point', [None, [0., 0.], [0., 0., float('nan')]])
def test_missing_or_invalid_base_position_cannot_establish_stillness(point):
    p = GestureHandover(.04)
    for i in range(120):
        p.observe_receiver('visible', .05, True, i/30, position_base=point)
    assert p.phase == 'FOLLOW_RECEIVER'


class Driver:
    """Instant-response simulation only; never evidence of real actuator timing."""
    def __init__(self):
        self.policy = GestureHandover(.04)
        enter_handover(self.policy)
        self.now, self.actual = 4., .04
        self.commands = []

    def feed(self, ratio, frames=1, *, still=True, fresh=True, move=True, repeat_stamp=False,
             side='Right'):
        for _ in range(frames):
            self.now += 1/30
            pinch = None if ratio is None else Pinch(ratio, side)
            command = self.policy.tick(pinch, fresh_frame=fresh, measured_width=self.actual,
                feedback_stamp=1. if repeat_stamp else self.now, now=self.now, arm_still=still)
            if command is not None:
                self.commands.append(command)
                self.policy.command_sent(command, self.now)
                if move:
                    self.actual = command

    def calibrate_and_arm(self, *, start_release=True):
        self.feed(.23, 35)
        self.feed(2.4, 35)
        assert self.policy.phase == 'GESTURE_ARM'
        assert not self.commands  # Full opening during calibration must NEVER release a cup.
        self.feed(.23, 25)
        assert self.policy.phase == 'GESTURE_READY' and not self.commands
        if start_release:
            self.feed(2.4, 2)
            assert self.policy.phase == 'GESTURE_CONTROL' and self.policy.release_started


def test_calibration_open_does_not_move_gripper_and_closed_confirmation_is_required():
    d = Driver()
    d.feed(.23, 35)
    d.feed(2.4, 80)
    assert d.policy.phase == 'GESTURE_ARM' and not d.commands
    d.feed(.23, 25)
    assert d.policy.phase == 'GESTURE_READY'
    assert not d.commands  # Learning AND closed confirmation leave the grip unchanged.


@pytest.mark.parametrize('distance,ready', [(.14, True), (.15, True), (.16, False)])
def test_preparation_distance_is_independent_of_five_centimeter_follow_goal(distance, ready):
    p = GestureHandover(.04)
    for tick in range(100):
        p.observe_receiver('visible', distance, True, tick/30., position_base=[0, 0, distance])
    assert p.distance == .05
    assert p.snapshot()['handover_ready'] == ready


@pytest.mark.parametrize('learned', [False, True])
def test_receiver_moves_before_release_revokes_old_gesture_and_requires_reconfirmation(learned):
    d = Driver()
    if learned:
        d.calibrate_and_arm(start_release=False)
    mapping = d.policy.mapping.mapping
    moved = [.02, 0., .05]
    d.policy.observe_receiver('visible', .06, True, d.now, position_base=moved)
    assert d.policy.phase == 'FOLLOW_RECEIVER' and not d.policy.release_started
    assert d.policy.mapping.mapping is mapping  # No need to relearn finished endpoints.
    d.feed(2.4, 30)
    assert not d.commands
    for tick in range(92):
        d.now += 1/30.
        d.policy.observe_receiver('visible', .06, True, d.now, position_base=moved)
    if learned:
        assert d.policy.phase == 'GESTURE_ARM'
        d.feed(2.4, 80)  # An old held-open hand must never become a delayed release.
        assert not d.commands and d.policy.phase == 'GESTURE_ARM'
        d.feed(.23, 25)
        assert d.policy.phase == 'GESTURE_READY' and not d.commands
        d.feed(2.4, 2)
        assert d.policy.release_started and d.commands
    else:
        assert d.policy.phase == 'GESTURE_LEARN'


def test_receiver_motion_cannot_restart_follow_after_the_first_opening_request():
    d = Driver()
    d.calibrate_and_arm()
    for i in range(120):
        d.policy.observe_receiver('visible', .07, True, d.now+i/30., position_base=[.02, 0., .05])
        d.policy.invalidate_near()
    assert d.policy.release_started and d.policy.phase == 'GESTURE_CONTROL'


def test_sustained_receiver_withdrawal_during_release_holds_both_without_reclosing():
    d = Driver()
    d.calibrate_and_arm()
    count, width = len(d.commands), d.actual
    for _ in range(18):
        d.now += 1/30.
        d.policy.observe_receiver('visible', .12, True, d.now, position_base=[.07, 0., .05])
    assert d.policy.phase == 'HANDOVER_CANCELLED'
    d.feed(2.4, 60)
    d.feed(.23, 60)
    assert len(d.commands) == count and d.actual == width


def test_transient_withdrawal_or_depth_loss_is_not_sustained_receiver_motion():
    d = Driver()
    d.calibrate_and_arm()
    for _ in range(10):
        d.now += 1/30.
        d.policy.observe_receiver('visible', .12, True, d.now, position_base=[.07, 0., .05])
    d.policy.observe_receiver('absent', None, False, d.now+.1)
    assert d.policy.phase == 'GESTURE_CONTROL'


def test_control_hand_loss_before_any_opening_requires_new_closed_confirmation():
    d = Driver()
    d.calibrate_and_arm(start_release=False)
    d.feed(None, 2)
    assert d.policy.phase == 'GESTURE_ARM' and not d.commands
    d.feed(2.4, 60)
    assert not d.commands


def test_same_frame_receiver_motion_beats_control_hand_opening(monkeypatch):
    import logging
    import servo_delivery as delivery
    from unittest.mock import Mock
    d = Driver(); d.calibrate_and_arm(start_release=False)
    node = delivery.ServoDeliveryNode.__new__(delivery.ServoDeliveryNode)
    node._gesture = d.policy
    node._consumed_sequence = 1
    node._latest_image = NS(height=480, width=640)
    node.get_logger = lambda: logging.getLogger('receiver-priority-test')
    node.model = NS(fk=lambda q: (np.zeros(3), np.eye(3)))
    node.servo = NS(feature_system=lambda q, p: (None, None, None, None, np.zeros(2)),
                    _image_center=np.zeros(2))
    roles = HandRoles(); roles.receiver = 'Left'
    roles.control_observations = [observation('Right', ratio=2.4)]
    node.target = NS(hand_roles=roles)
    node.gripper = Mock()
    moved = (np.array([.02, 0., .05]), None, None)

    def detect(*args):
        node._consumed_sequence += 1
        node._sdk_received_monotonic = time.monotonic()
        return moved

    node._detect_latest = detect
    assert node._gesture_hold_step(np.zeros(7), time.time()) is moved
    assert node._gesture.phase == 'FOLLOW_RECEIVER'
    assert not node._gesture.release_started
    node.gripper.open.assert_not_called()


def test_intermediate_aperture_maps_proportionally_and_does_not_rebase_closure_limit():
    d = Driver()
    d.calibrate_and_arm()
    mapping = d.policy.mapping.mapping
    d.feed((mapping.closed+mapping.opened)/2, 180)
    assert d.actual == pytest.approx((.039+.085)/2)
    d.feed(.23, 180)
    assert d.actual == pytest.approx(.039)
    assert d.policy.grip_width == .04 and d.policy.minimum == .039


@pytest.mark.parametrize('ratio', [.23, 1.2, 2.4])
def test_lost_control_hand_keeps_any_current_opening_without_waiting_for_full_open(ratio):
    d = Driver()
    d.calibrate_and_arm()
    d.feed(ratio, 60)
    assert d.actual < .085
    width, count = d.actual, len(d.commands)
    d.feed(None, 150)
    assert d.policy.phase == 'GESTURE_PAUSED'
    assert len(d.commands) == count and d.actual == width
    assert d.policy.snapshot()['control_hand_loss_policy'] == 'pause_keep_current_opening'


def test_same_control_hand_resumes_from_measured_opening_without_closed_rearming():
    d = Driver()
    d.calibrate_and_arm()
    d.feed(2.4, 60)
    d.feed(None, 100)
    count = len(d.commands)
    # The previously issued small target may finish while no new ones are sent.
    d.actual += .002
    recovered_width = d.actual
    d.feed(2.4, 20, side='Left')
    assert d.policy.phase == 'GESTURE_PAUSED' and len(d.commands) == count
    d.feed(2.4, 2)
    assert d.policy.phase == 'GESTURE_PAUSED' and len(d.commands) == count
    d.feed(2.4)
    assert d.policy.phase == 'GESTURE_CONTROL'
    assert d.commands[count:] == [pytest.approx(recovered_width)]
    d.feed(2.4)
    assert recovered_width < d.actual <= recovered_width+d.policy.ramp.open_rate/30+1e-9


def test_changed_hand_label_pauses_immediately_without_sending_a_return_target():
    d = Driver()
    d.calibrate_and_arm()
    d.feed(2.4, 60)
    width, count = d.actual, len(d.commands)
    d.feed(2.4, 30, side='Left')
    assert d.policy.phase == 'GESTURE_PAUSED'
    assert d.actual == width and len(d.commands) == count


@pytest.mark.parametrize('gap_has_ticks', [False, True])
def test_camera_gap_requires_three_new_frames_and_resets_the_target_ramp(gap_has_ticks):
    d = Driver()
    d.calibrate_and_arm()
    d.feed(2.4, 40)
    if gap_has_ticks:
        d.feed(None, 12, fresh=False)
        assert d.policy.phase == 'GESTURE_PAUSED'
    else:
        d.now += 1.  # Also cover a stalled timer returning directly with a new frame.
    width, count = d.actual, len(d.commands)
    d.feed(2.4, 2)
    assert d.policy.phase == 'GESTURE_PAUSED'
    assert len(d.commands) == count
    d.feed(2.4)
    assert d.policy.phase == 'GESTURE_CONTROL'
    assert d.commands[count:] == [pytest.approx(width)]


def test_full_open_confirmation_ends_session_and_loss_never_recloses():
    d = Driver()
    d.calibrate_and_arm()
    d.feed(2.4, 200)
    assert d.policy.phase == 'HANDOVER_DONE'
    assert d.actual >= .084
    count = len(d.commands)
    d.feed(None, 100)
    d.feed(.23, 100)
    assert d.policy.phase == 'HANDOVER_DONE' and len(d.commands) == count


def test_old_feedback_packet_cannot_confirm_full_open_multiple_times():
    d = Driver()
    d.calibrate_and_arm()
    d.feed(2.4, 200, repeat_stamp=True)
    assert d.actual >= .084 and d.policy.phase == 'GESTURE_CONTROL'
    assert d.policy.open_count <= 1
    d.feed(2.4, 4)
    assert d.policy.phase == 'HANDOVER_DONE'


@pytest.mark.parametrize('failure', ['arm_moving', 'no_response', 'camera_stale'])
def test_faults_and_camera_loss_do_not_continue_opening(failure):
    d = Driver()
    d.calibrate_and_arm()
    if failure == 'arm_moving':
        count = len(d.commands)
        d.feed(2.4, still=False)
        assert d.policy.phase == 'GESTURE_FAULT' and len(d.commands) == count
    elif failure == 'no_response':
        d.feed(2.4, 100, move=False)
        assert d.policy.phase == 'GESTURE_FAULT' and 'NO RESPONSE' in d.policy.reason
    else:
        d.feed(2.4, 40)
        d.feed(None, 12, fresh=False)
        assert d.policy.phase == 'GESTURE_PAUSED'


def test_gripper_feedback_requires_fresh_width_enabled_driver_and_finite_values():
    flags = NS(driver_enable_status=True, **{key: False for key in (
        'voltage_too_low', 'motor_overheating', 'driver_overcurrent',
        'driver_overheating', 'sensor_status', 'driver_error_status')})
    status = NS(timestamp=time.time(), msg=NS(value=.04, force=.4, mode='width', foc_status=flags))
    assert read_opening(status)[0] == .04
    status.timestamp -= 1.
    with pytest.raises(ValueError):
        read_opening(status)
    status.timestamp = time.time()
    status.msg.foc_status.driver_enable_status = False
    with pytest.raises(ValueError):
        read_opening(status)


def test_new_gripper_parser_may_receive_first_feedback_after_attachment(monkeypatch):
    import handoff.gesture_handover as module
    clock = NS(now=0.)
    replies = iter([None, None, NS(value=.04)])
    monkeypatch.setattr(module, 'read_opening', lambda status: (
        (_ for _ in ()).throw(ValueError('missing')) if status is None else (.04, .4, 123.)))
    monkeypatch.setattr(module, 'time', NS(monotonic=lambda: clock.now,
        sleep=lambda seconds: setattr(clock, 'now', clock.now+seconds)))
    assert module.wait_initial_opening(NS(status=lambda: next(replies)))[0] == .04
    assert clock.now == pytest.approx(.02)


def test_hand_target_selects_receiver_with_two_hands_and_retains_rgb_control_without_depth():
    from targets.hand_target import HandTarget
    left, right = observation(), observation('Right')
    for index, value in enumerate((left, right)):
        value.palm_center_camera_m = np.array([.01+index*.1, 0., .2])
        value.valid_depth_landmarks = 5
        value.palm_pixel = (320+index*20, 240)
    detected = [left]
    target = HandTarget.__new__(HandTarget)
    target.hand_roles = HandRoles()
    target.min_valid_depth_landmarks = 3
    target.approach_mode = 'camera_ray'
    target._frame = NS(transform_to_base=lambda q, T: T)
    target._detector = NS(detect=lambda *args: detected)
    args = [np.zeros((480, 640, 3), dtype=np.uint8), None, np.eye(3), None, np.zeros(7)]
    assert target.detect(*args) is None
    assert target.detect(*args) is None
    assert target.detect(*args).handedness == 'Left'
    detected[:] = [right, left]
    assert target.detect(*args).handedness == 'Left'
    assert target.hand_roles.control_observations == [right]
    left.palm_center_camera_m = None
    left.valid_depth_landmarks = 0
    assert target.detect(*args) is None
    assert target.last_diagnostics['receiver_presence'] == 'visible'
    assert target.last_diagnostics['reason'] == 'invalid_palm_depth'
    detected[:] = [right]
    right.palm_center_camera_m = None
    assert target.detect(*args) is None
    assert target.last_diagnostics['reason'] == 'receiver_absent_2d'
    assert target.hand_roles.control_observations == [right]


@pytest.mark.parametrize('mode', ['gesture', 'fusion'])
def test_loaded_command_is_preserved_during_grasp_to_servo_handoff(monkeypatch, mode):
    from handoff.grasp_to_servo_handoff import handoff_after_lift
    import real_nero_grasp.mit_trajectory_executor as executor
    writes = []
    gripper = NS(move_gripper_m=lambda **values: writes.append(values))
    arm = NS(gripper=gripper, _last_gripper_command=dict(width=.035, force=.5))
    session = NS(ensure_started=lambda: None, get_robot=lambda: NS())
    monkeypatch.setattr(executor, 'get_mit_session', lambda *args: session)
    args = NS(gesture_handover=mode == 'gesture', fusion_handover='shadow' if mode == 'fusion' else None,
              trajectory_executor='mit', gripper_force=.5)
    handoff_after_lift(arm, args, NS(contact_width=.040))
    assert writes and all(row == dict(value=.035, force=.5) for row in writes)
    assert args._servo_gripper.last_command == dict(width=.035, force=.5)


@pytest.mark.parametrize('arguments', [
    ['--servo-follow-only'], ['--delivery-distance', '.2'], ['--gesture-extra-close-mm', '2'],
    ['--gesture-open-seconds', '0'], ['--gesture-near-tolerance', '.1'],
    ['--gesture-stable-seconds', '0'], ['--gesture-still-tolerance-mm', '0'],
    ['--gesture-ready-distance', '.04'],
])
def test_invalid_integrated_options_fail_before_hardware(arguments):
    from real_nero_grasp.servo_cli import servo_parser
    from servo_delivery import validate_servo_args
    args = servo_parser().parse_args(['--gesture-handover', '--target-source', 'mediapipe',
        '--hand-camera-center', '--delivery-distance', '.05', '--execute', *arguments])
    args.handeye = __file__
    with pytest.raises(ValueError):
        validate_servo_args(args)


def test_gesture_handover_requires_camera_centering_before_hardware():
    from real_nero_grasp.servo_cli import servo_parser
    from servo_delivery import validate_servo_args
    args = servo_parser().parse_args(['--gesture-handover', '--target-source', 'mediapipe',
        '--delivery-distance', '.05', '--execute'])
    args.handeye = __file__
    with pytest.raises(ValueError, match='requires --hand-camera-center'):
        validate_servo_args(args)


def test_integrated_options_are_available_in_main_cli_and_pipeline_defaults(monkeypatch, tmp_path):
    import sys
    from real_nero_grasp.cli import parse_args
    from servo_delivery import parse_servo_args, validate_servo_args
    monkeypatch.setattr(sys, 'argv', ['main.py', '--phase', 'servo', '--handeye', __file__,
        '--target-source', 'mediapipe', '--gesture-handover', '--hand-camera-center',
        '--delivery-distance', '.05', '--execute'])
    args = parse_args()
    validate_servo_args(args)
    assert args.gesture_handover and not args.servo_follow_only
    assert args.gesture_stable_seconds == 3. and args.gesture_still_tolerance_mm == 10.
    assert args.gesture_ready_distance == .15 and args.hand_memory_approach
    config = tmp_path/'pipeline.yaml'
    config.write_text('servo:\n  target_source: mediapipe\n  gesture_handover: true\n'
        '  hand_camera_center: true\n  delivery_distance: 0.05\n'
        '  gesture_stable_seconds: 4\n  gesture_still_tolerance_mm: 12\n'
        '  gesture_ready_distance: 0.12\n  hand_memory_approach: false\n')
    preview = parse_servo_args(['--config', str(config), '--handeye', __file__,
                               '--gesture-preview-grip-width-mm', '40'])
    validate_servo_args(preview)
    assert preview.gesture_handover and preview.delivery_distance == .05 and not preview.execute
    assert preview.gesture_stable_seconds == 4. and preview.gesture_still_tolerance_mm == 12.
    assert preview.gesture_ready_distance == .12 and not preview.hand_memory_approach
    override = parse_servo_args(['--config', str(config), '--gesture-ready-distance', '.15',
                                 '--hand-memory-approach'])
    assert override.gesture_ready_distance == .15 and override.hand_memory_approach


def test_node_freezes_on_stillness_before_occlusion_and_never_runs_legacy_release_or_retreat(monkeypatch):
    from collections import deque
    import logging
    from unittest.mock import Mock
    import servo_delivery as delivery
    from real_nero_grasp.servo_cli import servo_parser
    node = delivery.ServoDeliveryNode.__new__(delivery.ServoDeliveryNode)
    node.args = servo_parser().parse_args(['--gesture-handover', '--target-source', 'mediapipe',
                                         '--hand-camera-center', '--delivery-distance', '.05'])
    node.backend = delivery.FakeBackend()
    node.get_logger = lambda: logging.getLogger('gesture-integration-test')
    node._gesture = GestureHandover(.04)
    node._gesture_joint_history = deque(maxlen=40)
    node._qdot_window = deque(maxlen=60)
    node._last_target_time = 1000.
    node._last_tracking_log = node._last_lost_log = 0.
    node._target_lost = False
    node._reacquire_count, node._reacquire_target = 0, None
    node._imp_cfg = None
    node._detection_reason = 'ok'
    node.target = NS(last_diagnostics={}, hand_roles=HandRoles())
    node.state, node.state_entered, node.stable_frames = 'TRACK_HAND', 0., 100
    node.gripper = Mock()
    node.model = NS(fk=lambda q: (np.zeros(3), np.eye(3)))
    node.servo = NS(params=NS(desired_distance=.05), update=Mock(return_value={
        'error': np.zeros(3), 'orientation_error': np.zeros(3),
        'camera_center': {'pixel_error': np.zeros(2)}}))
    node.safety = NS(evaluate=lambda *args: NS(fatal=False, hold=False))
    node._detect_latest = Mock(return_value=(np.array([0., 0., .05]), None, None))
    node._refresh, node._pause_target_motion, node._engage = Mock(), Mock(), Mock()
    clock = NS(now=0.)
    monkeypatch.setattr(delivery, 'time', NS(time=lambda: 1000.+clock.now, monotonic=lambda: clock.now))
    for i in range(90):
        clock.now = i/30
        node._servo_step()
    assert node.state == 'TRACK_HAND' and not node._gesture.snapshot()['handover_ready']
    prior = node._refresh.call_count
    clock.now = 3.
    node._servo_step()
    assert node.state == 'GESTURE_LEARN'
    node._engage.assert_called_once()
    assert node._refresh.call_count == prior  # No moving reference on the transition tick.
    node._gesture_hold_step = Mock()
    # Neither an occluded/depth-invalid hand nor its reappearance resumes following.
    for reason in ('receiver_absent_2d', 'invalid_palm_depth', 'ok'):
        node._detection_reason = reason
        node._detect_latest.return_value = None if reason != 'ok' else (
            np.array([0., 0., .05]), None, None)
        clock.now += .1
        node._servo_step()
    assert node._refresh.call_count == prior
    assert node._gesture_hold_step.call_count == 3
    node.gripper.open.assert_not_called()


def test_gesture_writer_reuses_mit_gripper_without_arm_modes_or_new_connections(monkeypatch):
    from collections import deque
    import logging
    import servo_delivery as delivery
    writes = []
    value = NS(width=.04)
    flags = NS(driver_enable_status=True, **{key: False for key in (
        'voltage_too_low', 'motor_overheating', 'driver_overcurrent',
        'driver_overheating', 'sensor_status', 'driver_error_status')})
    state = NS(msg=NS(ctrl_mode=1, arm_status=0, err_code=0, mode_feedback=6), timestamp=time.time())
    # No enable, disable, set_motion_mode, connect, move_j, or prepare_can_hold API is exposed.
    robot = NS(get_arm_status=lambda: state)
    node = delivery.ServoDeliveryNode.__new__(delivery.ServoDeliveryNode)
    node.backend = NS(get_robot=lambda: robot)
    node.get_logger = lambda: logging.getLogger('gesture-writer-test')
    node._gesture = GestureHandover(.04)
    node._gesture.phase = 'GESTURE_CONTROL'
    node._gesture.release_started = True
    node._gesture.mapping.mapping = PinchMapping(.23, 2.4)
    node._gesture.mapping.phase = 'ready'
    node._gesture.control_hand.side = 'Right'
    node._gesture.control_hand.candidate = 'Right'
    node._gesture.control_hand.good_frames = 3
    node._gesture.filtered = .5
    now = time.monotonic()
    node._gesture.ramp.reset(.04, now-.03)
    node._gesture_joint_history = deque([(now-.3, np.zeros(7)), (now-.2, np.zeros(7)),
        (now-.1, np.zeros(7)), (now-.05, np.zeros(7)), (now, np.zeros(7))])
    node._gesture_force = .5
    node._consumed_sequence = 1
    node._latest_image = NS(height=480, width=640)
    roles = HandRoles()
    roles.receiver = 'Left'
    roles.control_observations = [observation('Right', ratio=1.315)]
    node.target = NS(hand_roles=roles)
    node._diag_sample = {}

    def detect(*args):
        node._consumed_sequence += 1
        node._sdk_received_monotonic = time.monotonic()
        return None  # Receiver occluded; RGB control landmarks still available.

    def status():
        return NS(timestamp=time.time(), msg=NS(value=value.width, force=.4, mode='width', foc_status=flags))

    def send(**kwargs):
        writes.append(kwargs)
        value.width = kwargs['width']

    node._detect_latest = detect
    node.gripper = NS(status=status, open=send)
    node._gesture_hold_step(np.zeros(7), time.time())
    assert len(writes) == 1 and writes[0]['force'] == .5
    assert .039 <= writes[0]['width'] <= .045
    roles.control_observations = []
    for _ in range(5):
        node._gesture_hold_step(np.zeros(7), time.time())
    assert node._gesture.phase == 'GESTURE_PAUSED'
    assert len(writes) == 1  # Pausing must also stop periodic SDK target refreshes.
    roles.control_observations = [observation('Right', ratio=1.315)]
    for _ in range(2):
        node._gesture_hold_step(np.zeros(7), time.time())
    assert len(writes) == 1
    node._gesture_hold_step(np.zeros(7), time.time())
    assert node._gesture.phase == 'GESTURE_CONTROL'
    assert len(writes) == 2 and writes[-1]['width'] == writes[0]['width']
    node._gesture.phase = 'HANDOVER_DONE'
    node._gesture_hold_step(np.zeros(7), time.time())
    assert len(writes) == 2  # A finished session never re-closes or keeps sending.
