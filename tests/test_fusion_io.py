"""Effector bounds, feedback ownership and offline model fitting, all mocked."""
import json
import sys
from types import SimpleNamespace as NS
from unittest.mock import Mock
import time

import numpy as np
import pytest

from handoff.fusion_handover import FusionConfig, LinearPullModel
from handoff.fusion_servo import FusionServoMixin
from handoff.pinch_gripper import GripperResponseWatch


class Adapter(FusionServoMixin):
    pass


def opening_adapter(monkeypatch):
    from handoff import fusion_servo
    a, clock, measured = Adapter(), [10.], [.04]
    monkeypatch.setattr(fusion_servo.time, 'monotonic', lambda: clock[0])
    a._fusion = NS(config=FusionConfig(open_seconds=.5), phase='FUSION_OPENING')
    a._fusion_grip_width, a._fusion_force = .04, .5
    a._fusion_ramp = a._fusion_last_command = a._fusion_last_send = None
    a._fusion_open_started = a._fusion_last_gripper_stamp = None
    a._fusion_open_count = 0
    a._fusion_watch = GripperResponseWatch()
    a._fusion_simulated = False
    a._fusion_arm_status = Mock()
    a._fusion_visual = dict(received_monotonic=clock[0])
    a.gripper = NS(status=Mock(return_value=None), open=Mock(side_effect=lambda width, force: measured.__setitem__(0, width)))
    monkeypatch.setattr(fusion_servo, 'read_opening', lambda _: (measured[0], .1, 100.+clock[0]))
    return a, clock, measured


def test_opening_is_rate_and_feedback_lead_limited_until_physical_feedback_done(monkeypatch):
    a, clock, measured = opening_adapter(monkeypatch)
    sent = []
    for i in range(30):
        clock[0] = 10.+i*.05
        a._fusion_visual['received_monotonic'] = clock[0]
        before = measured[0]
        a._fusion_open_step(measured[0], 100.+clock[0], clock[0])
        if measured[0] != before:
            assert 0 < measured[0]-before <= .0045+1e-9
            sent.append(measured[0])
        if a._fusion.phase == 'HANDOVER_DONE':
            break
    assert a._fusion.phase == 'HANDOVER_DONE'
    assert sent and max(sent) <= .085 and np.all(np.diff(sent) > 0)
    assert a.gripper.open.call_count == len(sent)


def test_duplicate_gripper_feedback_does_not_advance_opening(monkeypatch):
    a, clock, measured = opening_adapter(monkeypatch)
    a._fusion_open_step(.04, 110., 10.)
    for _ in range(5):
        clock[0] += .05
        a._fusion_visual['received_monotonic'] = clock[0]
        a._fusion_open_step(.04, 110., clock[0])
    a.gripper.open.assert_not_called()


def test_expired_visual_evidence_is_checked_again_immediately_before_sdk_write(monkeypatch):
    a, clock, measured = opening_adapter(monkeypatch)
    a._fusion_open_step(.04, 110., 10.)
    clock[0] = 10.3
    with pytest.raises(ValueError, match='visual evidence expired'):
        a._fusion_open_step(.04, 110.3, clock[0])
    a.gripper.open.assert_not_called()


def test_feedback_reader_never_invokes_sdk_get_motor_states_or_zeros_shared_velocity():
    from control_backends.force_ctrl.feedback_health import FeedbackHealthMonitor
    wall = time.time()
    parser = NS()
    for name, indices in (('joint_12', [1, 2]), ('joint_34', [3, 4]), ('joint_56', [5, 6]), ('joint_7', [7])):
        setattr(parser, name, NS(timestamp=wall, msg=NS(**{f'joint_{i}': 0. for i in indices})))
    for i in range(1, 8):
        setattr(parser, f'motor_state_{i}', NS(timestamp=wall, msg=NS(velocity=.012, torque=.4, current=.2)))
    robot = NS(_parser=parser, get_motor_states=Mock(side_effect=AssertionError('mutating SDK getter forbidden')))
    a = Adapter()
    a._fusion = NS(fault=Mock())
    a._fusion_health = FeedbackHealthMonitor()
    a._fusion_observer = NS(poll=lambda: (None, []))
    a._fusion_recorder = NS(error=None)
    a._fusion_simulated = False
    a.backend = NS(get_robot=lambda: robot)
    a._fusion_arm_status = lambda: {}
    a._pause_target_motion = Mock()
    a._poll_fusion(np.zeros(7), wall)
    robot.get_motor_states.assert_not_called()
    a._fusion.fault.assert_not_called()
    np.testing.assert_allclose(a._fusion_sample['dq_raw'], .012)
    np.testing.assert_allclose(a._fusion_sample['motor_current'], .2)
    assert all(getattr(parser, f'motor_state_{i}').msg.velocity == .012 for i in range(1, 8))


def test_even_an_otherwise_allowed_mit_mode_change_invalidates_locked_baseline():
    a = Adapter()
    a._fusion_simulated = False
    a._fusion_hold_control_mode = (1, 6)
    msg = NS(ctrl_mode=1, mode_feedback=6, arm_status=0, err_code=0)
    a.backend = NS(get_robot=lambda: NS(get_arm_status=lambda: NS(timestamp=time.time(), msg=msg)))
    assert a._fusion_arm_status()['mode_feedback'] == 6
    msg.mode_feedback = 4
    with pytest.raises(ValueError, match='mode changed'):
        a._fusion_arm_status()


@pytest.mark.parametrize('depth', [False, True])
def test_rgb_only_camera_accepts_raw_image_distortion_but_pinhole_depth_contract_stays_strict(depth):
    from cameras.realsense import RealSenseCamera
    camera = RealSenseCamera(enable_depth=depth, warmup=0)
    intr = NS(fx=100., fy=100., ppx=1., ppy=1., coeffs=[.01, 0, 0, 0, 0])
    frame = NS(get_frame_number=lambda: 1, get_timestamp=lambda: 33.,
        get_data=lambda: np.zeros((2, 2, 3), np.uint8),
        profile=NS(as_video_stream_profile=lambda: NS(intrinsics=intr)))
    frames = NS(get_color_frame=lambda: frame, get_depth_frame=lambda: frame)
    camera.pipeline = NS(wait_for_frames=lambda _: frames, poll_for_frames=lambda: None)
    camera.align = NS(process=lambda f: f)
    if depth:
        with pytest.raises(RuntimeError, match='distortion'): camera.read()
    else:
        result = camera.read()
        assert result.depth_m is None and result.D[0] == .01


@pytest.mark.parametrize('name,depth,accepted', [
    ('RealSense D455', False, True), ('RealSense D455', True, False),
    ('RealSense D457', False, True), ('RealSense D457', True, False),
    ('RealSense D435i', False, True), ('RealSense D435i', True, True),
    ('RealSense D405', False, False),
])
def test_external_d455_d457_rgb_selection_does_not_expand_metric_depth_support(monkeypatch, name, depth, accepted):
    from cameras.realsense import RealSenseCamera
    streams = []
    device = NS(get_info=lambda key: name if key == 'name' else 'external')
    fake = NS(context=lambda: NS(query_devices=lambda: [device]),
        camera_info=NS(name='name', serial_number='serial'),
        config=lambda: NS(enable_device=lambda _: None, enable_stream=lambda *args: streams.append(args)),
        pipeline=lambda _: NS(start=lambda _: NS(get_device=lambda: NS(first_depth_sensor=lambda:
            NS(get_depth_scale=lambda: .001, supports=lambda _: False))), stop=lambda: None),
        stream=NS(color='color', depth='depth'), format=NS(bgr8='bgr8', z16='z16'),
        option=NS(visual_preset='visual_preset'), align=lambda _: None)
    monkeypatch.setitem(sys.modules, 'pyrealsense2', fake)
    camera = RealSenseCamera(serial='external', enable_depth=depth, warmup=0)
    if accepted:
        try:
            camera.start()
            assert len(streams) == (2 if depth else 1)
            assert streams[0][0] == 'color'
        finally:
            camera.close()
    else:
        with pytest.raises(RuntimeError, match='Expected one selected'):
            camera.start()
        assert not streams


def test_linear_fit_learns_selected_axes_and_never_uses_other_axis_coefficients():
    from tools.train_handover_pull import fit_binary, metrics
    rng = np.random.default_rng(17)
    x = rng.normal(size=(120, 28))
    y = (x[:, 1] > 0).astype(int)
    data = fit_binary(x, y, joints=(2,))
    model = LinearPullModel(data)
    active = np.zeros(28, bool)
    active[[1, 8, 15, 22]] = True
    assert np.all(np.array(data['weights'])[~active] == 0)
    result = metrics(model, x, y, threshold=.5)
    assert result['true_positive']+result['true_negative'] >= 110
    assert result['unit'] == 'sample_level_action_only_not_handover_success'


def test_training_refuses_overlapping_whole_sessions_before_loading_data(tmp_path):
    from tools.train_handover_pull import main
    with pytest.raises(SystemExit):
        main(['--train', str(tmp_path), '--test', str(tmp_path), '--out', str(tmp_path/'model.json')])
    assert not (tmp_path/'model.json').exists()


def test_shadow_recording_model_training_and_portable_loading_roundtrip(tmp_path):
    from tools.train_handover_pull import main
    for name, offset in [('train', 0), ('test', 10)]:
        directory = tmp_path/name
        directory.mkdir()
        (directory/'metadata.json').write_text(json.dumps(dict(dry_run=False, mode='shadow',
            control_mode='mit_cartesian_hold', config=dict(window_seconds=1.))))
        rows = []
        for i in range(30):
            positive = i % 2 == 1
            features = np.zeros(28)
            features[1] = 2. if positive else -2.
            now = offset+i*.1+1.
            rows.append(dict(monotonic=now,
                feedback=dict(motor_timestamps=[100.+now]*7),
                policy=dict(phase='FUSION_WAIT_TAKE', features=features.tolist()),
                visual=dict(received_monotonic=now, label='TAKE_PULL' if positive else 'POST_RELEASE')))
        (directory/'samples.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
    main(['--train', str(tmp_path/'train'), '--test', str(tmp_path/'test'),
          '--joints', '2', '--out', str(tmp_path/'model.json')])
    data = json.loads((tmp_path/'model.json').read_text())
    assert data['test_action_metrics']['false_positive'] == 0
    assert data['test_action_metrics']['true_positive'] == 15
    assert data['support_verified'] is None
    model = LinearPullModel.load(tmp_path/'model.json')
    assert model.score(np.zeros(28)) == pytest.approx(.5)


def test_async_recording_retains_metadata_and_raw_samples(tmp_path):
    from handoff.fusion_recording import FusionRecorder
    directory = tmp_path/'run'
    r = FusionRecorder(directory, dict(mode='shadow', support_verified=None))
    r.write(dict(tau=np.arange(7), missing=float('nan')))
    r.close()
    assert r.error is None
    assert json.loads((directory/'metadata.json').read_text())['support_verified'] is None
    saved = json.loads((directory/'samples.jsonl').read_text())
    assert saved['tau'] == list(range(7)) and saved['missing'] is None
