"""Release boundary, stale/duplicate streams and withdrawal; no hardware."""
from dataclasses import replace
from types import SimpleNamespace as NS
from unittest.mock import Mock

import numpy as np
import pytest

from handoff.fusion_handover import FusionConfig, FusionHandover, LinearPullModel, FEATURE_VERSION, torque_features
from handoff.fusion_observer import visual_evidence, validate_roi
from handoff.fusion_cli import validate_fusion_args
from real_nero_grasp.servo_cli import servo_parser


class ActionModel:
    # Synthetic detector only, for exercising transitions, not a trained model.
    def score(self, x):
        return .99 if x[8] > .8 else .01


class Trial:
    def __init__(self, mode='shadow', model=True):
        c = FusionConfig(settle_seconds=.2, baseline_seconds=.4,
            window_seconds=.4, max_gap_seconds=.15, min_window_samples=4,
            confirm_seconds=.15, visual_stable_seconds=.1)
        self.p = FusionHandover(c, mode=mode, model=ActionModel() if model else None)
        self.p.begin_hold(np.zeros(7))
        self.t, self.counter = 0., 0

    def tick(self, force=0., *, candidate=False, clear=True, confirm=False,
             duplicate=False, age=0., dq=0., cancel=False, sequence=None):
        self.t = round(self.t+.05, 8)
        if not duplicate:
            self.counter += 1
        sample = dict(q=np.zeros(7), dq_raw=np.full(7, dq), tau=np.array([0., force, 0., 0., 0., 0., 0.]),
            motor_timestamps=np.full(7, 1000.+self.counter*.05),
            position_timestamps=np.full(7, 1000.+self.counter*.05))
        visual = dict(received_monotonic=self.t-age, roi_confirmed=True,
            roi_revision=1, sequence=self.counter if sequence is None else sequence,
            in_roi=candidate, visible=candidate, clear=clear)
        self.p.tick(sample, visual, self.t, confirm_untouched=confirm, cancel=cancel)
        return self.p.phase

    def baseline(self):
        for _ in range(8):
            self.tick()
        assert self.p.phase == 'FUSION_WAIT_BASELINE'
        self.tick(confirm=True)
        for _ in range(10):
            self.tick()
        assert self.p.phase == 'FUSION_WAIT_TAKE'

    def arm(self):
        for _ in range(14):
            self.tick(candidate=True, clear=False)
        assert self.p._armed


@pytest.mark.parametrize('mode,expected', [('shadow', 'WOULD_RELEASE'), ('release', 'FUSION_OPENING')])
def test_baseline_then_current_visual_and_new_pull_only_crosses_selected_boundary(mode, expected):
    t = Trial(mode)
    t.baseline()
    t.arm()
    for _ in range(8):
        t.tick(1.5, candidate=True, clear=False)
    assert t.p.phase == expected


def test_no_model_collects_features_but_never_authorizes_release():
    t = Trial(model=False)
    t.baseline()
    for _ in range(100):
        t.tick(3., candidate=True, clear=False)
    assert t.p.phase == 'FUSION_WAIT_TAKE'
    assert t.p.features.shape == (28,) and t.p.pull_score is None
    with pytest.raises(ValueError, match='model'):
        FusionHandover(mode='release')


def test_operator_no_contact_confirmation_is_required_and_rejected_with_hand_in_roi():
    t = Trial()
    for _ in range(100):
        t.tick()
    assert t.p.phase == 'FUSION_WAIT_BASELINE' and t.p.baseline is None
    t.tick(candidate=True, clear=False, confirm=True)
    assert t.p.phase == 'FUSION_WAIT_BASELINE'
    assert t.p.reason == 'B rejected: hand_in_roi'
    assert t.p.snapshot()['baseline_readiness']['blockers'] == ['hand_in_roi']
    for _ in range(5):
        t.tick(candidate=False, clear=True)
    assert t.p.phase == 'FUSION_WAIT_BASELINE' and t.p.baseline is None
    assert t.p.baseline_readiness['ready']
    assert 'press B' in t.p.reason  # Earlier rejected B never becomes queued permission.
    t.tick(confirm=True)
    assert t.p.phase == 'FUSION_BASELINE'


def test_raw_velocity_even_with_constant_positions_prevents_settling():
    t = Trial()
    for _ in range(40):
        t.tick(dq=.04, confirm=True)
    assert t.p.phase == 'FUSION_SETTLING'


@pytest.mark.parametrize('cause', ['hand', 'stale_image', 'motion'])
def test_interrupted_baseline_requires_new_confirmation(cause):
    t = Trial()
    for _ in range(8): t.tick()
    t.tick(confirm=True)
    assert t.p.phase == 'FUSION_BASELINE'
    t.tick(clear=cause != 'hand', age=.4 if cause == 'stale_image' else 0., dq=.05 if cause == 'motion' else 0.)
    for _ in range(40): t.tick()
    assert t.p.phase == 'FUSION_WAIT_BASELINE' and t.p.baseline is None


def test_unstable_torque_cannot_become_a_valid_untouched_baseline():
    t = Trial()
    for _ in range(8): t.tick()
    t.tick(confirm=True)
    for i in range(12): t.tick((-1)**i)
    assert t.p.phase == 'FUSION_WAIT_BASELINE' and t.p.baseline is None


def test_old_high_score_before_visual_eligibility_cannot_arm():
    t = Trial()
    t.baseline()
    for _ in range(20): t.tick(2.)
    for _ in range(100): t.tick(2., candidate=True, clear=False)
    assert t.p.phase == 'FUSION_WAIT_TAKE' and not t.p._armed


def test_withdrawal_during_confirmation_cancels_and_old_pull_cannot_be_reused():
    t = Trial()
    t.baseline()
    t.arm()
    t.tick(2., candidate=True, clear=False)
    assert t.p.phase == 'FUSION_CONFIRM'
    t.tick(2., candidate=False)
    for _ in range(30): t.tick(2., candidate=True, clear=False)
    assert t.p.phase == 'FUSION_WAIT_TAKE' and not t.p._armed
    for _ in range(20): t.tick(candidate=True, clear=False)
    assert t.p._armed
    for _ in range(8): t.tick(2., candidate=True, clear=False)
    assert t.p.phase == 'WOULD_RELEASE'


def test_duplicate_motor_packets_cannot_complete_confirmation():
    t = Trial()
    t.baseline()
    t.arm()
    t.tick(2., candidate=True, clear=False)
    for _ in range(6): t.tick(2., candidate=True, clear=False, duplicate=True)
    assert t.p.phase != 'WOULD_RELEASE'


def test_one_camera_frame_cannot_complete_confirmation():
    t = Trial()
    t.baseline()
    t.arm()
    for _ in range(8): t.tick(2., candidate=True, clear=False, sequence=500)
    assert t.p.phase != 'WOULD_RELEASE'


def test_vision_loss_during_opening_latches_fault_and_never_resumes():
    t = Trial('release')
    t.baseline()
    t.arm()
    for _ in range(8): t.tick(2., candidate=True, clear=False)
    assert t.p.phase == 'FUSION_OPENING'
    t.tick(2., candidate=False)
    for _ in range(20): t.tick(2., candidate=True, clear=False)
    assert t.p.phase == 'FUSION_FAULT'


def test_shadow_trial_can_be_reset_but_opening_cannot():
    t = Trial()
    t.baseline()
    t.arm()
    for _ in range(8): t.tick(2., candidate=True, clear=False)
    assert t.p.phase == 'WOULD_RELEASE'
    t.tick(cancel=True)
    assert t.p.phase == 'FUSION_WAIT_BASELINE' and t.p.baseline is None
    t.p.phase = 'FUSION_OPENING'
    t.tick(cancel=True)
    assert t.p.phase == 'FUSION_FAULT'


def test_shadow_candidate_is_revoked_on_vision_loss_and_cannot_be_replayed():
    t = Trial()
    t.baseline()
    t.arm()
    for _ in range(8): t.tick(2., candidate=True, clear=False)
    assert t.p.phase == 'WOULD_RELEASE'
    t.tick(2., candidate=False)
    for _ in range(20): t.tick(2., candidate=True, clear=False)
    assert t.p.phase == 'FUSION_WAIT_TAKE' and not t.p._armed


def test_3d_delivery_loss_resets_ready_time_and_cannot_establish_hold():
    p = FusionHandover(replace(FusionConfig(), ready_seconds=.5))
    for i in range(9):
        assert not p.observe_delivery([0, 0, .15], .15, True, i*.05)
    p.delivery_lost()
    for i in range(9):
        assert not p.observe_delivery([0, 0, .15], .15, True, 1+i*.05)
    assert not p.observe_delivery([0, 0, .15], .15, False, 1.5)
    assert p.phase == 'FOLLOW_RECEIVER'


def test_short_approach_segment_can_park_before_occlusion_without_human_stillness():
    p = FusionHandover(replace(FusionConfig(), ready_distance=.28))
    assert not p.observe_delivery([0, 0, .27], .27, True, 1.)
    assert not p.observe_delivery([.01, 0, .27], .27, True, 1.075)
    # Person moved 20mm: this is preparation evidence, not a human-stillness test.
    assert p.observe_delivery([.02, 0, .27], .27, True, 1.15)
    p.begin_hold(np.zeros(7))
    p.delivery_lost()
    assert p.phase == 'FUSION_SETTLING' and p.baseline is None
    assert p.parking_source == 'visual_pre_occlusion'


def test_jumping_3d_target_or_too_few_frames_cannot_create_early_parking():
    p = FusionHandover(replace(FusionConfig(), ready_distance=.28))
    assert not p.observe_delivery([0, 0, .27], .27, True, 1.)
    assert not p.observe_delivery([.12, 0, .27], .27, True, 1.1)
    assert p.delivery_evidence['frames'] == 1
    assert not p.observe_delivery([.12, 0, .27], .27, True, 1.25)
    p.delivery_lost()
    assert not p.observe_delivery([.12, 0, .27], .27, True, 1.3)
    assert p.phase == 'FOLLOW_RECEIVER'


def test_manual_parking_is_shadow_only_and_does_not_establish_baseline():
    p = FusionHandover()
    p.begin_hold(np.zeros(7), source='manual_shadow')
    assert p.phase == 'FUSION_SETTLING' and p.baseline is None
    assert p.parking_source == 'manual_shadow'
    release = FusionHandover(mode='release', model=ActionModel())
    with pytest.raises(ValueError, match='parking'):
        release.begin_hold(np.zeros(7), source='manual_shadow')
    assert release.phase == 'FOLLOW_RECEIVER'


def test_feature_sign_trend_and_model_format():
    times = np.arange(10)/10
    delta = times[:, None]*np.arange(1, 8)[None, :]
    f = torque_features(times, delta)
    np.testing.assert_allclose(f[21:], np.arange(1, 8))
    np.testing.assert_allclose(torque_features(times, -delta)[:7], -f[:7])
    model = LinearPullModel(dict(version=1, feature_version=FEATURE_VERSION, mean=[0]*28,
        scale=[1]*28, weights=[0]*28, intercept=0))
    assert model.score(f) == .5
    with pytest.raises(ValueError): torque_features([0, 0, 1, 2], np.zeros((4, 7)))
    with pytest.raises(ValueError): LinearPullModel(dict(model.data, scale=[0]*28))


def hand(side='Left'):
    xy = np.full((21, 3), .5)
    xy[5, :2], xy[17, :2] = [.4, .6], [.6, .6]
    return NS(handedness=side, handedness_score=.99, landmarks_normalized=xy,
              landmarks_pixel=np.rint(xy[:, :2]*[639, 479]).astype(int))


def test_rgb_only_receiver_candidate_is_independent_of_opposite_control_hand():
    o = hand()
    result = visual_evidence([hand('Right'), o], (480, 640), [.3, .3, .7, .7], 'Left')
    assert result['visible'] and result['in_roi'] and not result['clear']
    assert not visual_evidence([hand('Right')], (480, 640), [.3, .3, .7, .7], 'Left')['visible']
    assert not visual_evidence([o, o], (480, 640), [.3, .3, .7, .7], 'Left')['visible']
    assert visual_evidence([], (480, 640), [.3, .3, .7, .7], 'Left')['clear']
    with pytest.raises(ValueError): validate_roi([.7, .3, .2, .7])


def test_fusion_cli_is_opt_in_and_rejects_legacy_command_owners():
    p = servo_parser()
    assert p.parse_args([]).fusion_handover is None
    args = p.parse_args(['--fusion-handover', 'shadow', '--gesture-handover'])
    with pytest.raises(ValueError, match='ownership'): validate_fusion_args(args)
    args = p.parse_args(['--fusion-handover', 'release', '--target-source', 'mediapipe',
        '--hand-camera-center', '--fusion-global-serial', 'external', '--fusion-receiver-side', 'Left'])
    with pytest.raises(ValueError, match='model'): validate_fusion_args(args)


def test_fusion_follow_path_never_falls_through_to_legacy_distance_release():
    from test_servo_follow_only import _tracking_node
    node = _tracking_node('HANDOVER', False)
    node._fusion = NS(phase='FOLLOW_RECEIVER', observe_delivery=Mock(return_value=False))
    node._poll_fusion = Mock()
    node._record_fusion = Mock()
    node._fusion_report = Mock()
    node._on_timer()
    assert node.state == 'HANDOVER'
    node._refresh.assert_called_once()
    node.gripper.open.assert_not_called()


def test_fusion_hold_path_bypasses_wrist_detection_and_all_legacy_release_states():
    from test_servo_follow_only import _tracking_node
    node = _tracking_node('RELEASE', False)
    node._fusion = NS(phase='FUSION_WAIT_TAKE')
    node._poll_fusion = Mock()
    node._record_fusion = Mock()
    node._fusion_hold_step = Mock()
    node._detect_latest = Mock(side_effect=AssertionError('wrist XYZ not required during external RGB hold'))
    node._on_timer()
    node._fusion_hold_step.assert_called_once()
    node.gripper.open.assert_not_called()


def test_coarse_preparation_image_region_can_park_without_exact_centering():
    from test_servo_follow_only import _tracking_node
    node = _tracking_node('TRACK_HAND', False)
    node.args.hand_camera_center = True
    node._latest_image = NS(width=1280, height=720)
    node.servo.update.return_value.update(camera_center={'pixel_error': [15., 105.]})
    node._fusion = NS(phase='FOLLOW_RECEIVER', config=FusionConfig(), observe_delivery=Mock(return_value=True))
    node._poll_fusion = node._record_fusion = Mock()
    node._fusion_begin_hold = Mock()
    node._on_timer()
    assert node._fusion.observe_delivery.call_args.args[2] is True
    node._fusion_begin_hold.assert_called_once()
    node._refresh.assert_not_called()
    node.gripper.open.assert_not_called()


@pytest.mark.parametrize('mode,age,engaged,feedback,expected', [
    ('shadow', 0., True, True, True), ('release', 0., True, True, False),
    ('shadow', .4, True, True, False), ('shadow', 0., False, True, False),
    ('shadow', 0., True, False, False),
])
def test_manual_stop_key_uses_current_pose_without_wrist_xyz_and_cannot_open(monkeypatch, mode, age, engaged, feedback, expected):
    from test_servo_follow_only import _tracking_node
    from handoff import fusion_servo
    monkeypatch.setattr(fusion_servo.time, 'monotonic', lambda: 10.)
    node = _tracking_node('TRACK_HAND', False)
    node._fusion = NS(phase='FOLLOW_RECEIVER', mode=mode, config=FusionConfig())
    node._imp_engaged = engaged
    node._fusion_sample = {'q': np.zeros(7)} if feedback else None
    node._fusion_events = [dict(action='stop', at=10.-age)]
    node._fusion_begin_hold = Mock()
    assert node._fusion_manual_stop(np.zeros(7), 100.) is expected
    if expected:
        node._fusion_begin_hold.assert_called_once()
        assert node._fusion_begin_hold.call_args.kwargs['source'] == 'manual_shadow'
    else:
        node._fusion_begin_hold.assert_not_called()
    node.gripper.open.assert_not_called()
