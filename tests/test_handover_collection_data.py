"""Capture evidence needed for offline model/vision/guard replay, no devices."""
from types import SimpleNamespace as NS
import json

import numpy as np

from handoff.fusion_handover import FusionHandover
from handoff.fusion_observer import record_hand_observations, visual_evidence
from tools.check_handover_dataset import inventory


def test_side_on_rejected_hand_still_retains_raw_landmarks_and_side():
    points = np.full((21, 3), .5)  # Projected palm is edge-on, so policy rejects it.
    hand = NS(handedness='Right', handedness_score=.95,
              landmarks_normalized=points, landmarks_pixel=points[:, :2]*[640, 480])
    evidence = visual_evidence([hand], (480, 640, 3), [.3, .2, .7, .8], 'Right')
    assert evidence['rejected'][0]['reason'] == 'palm too small or edge-on'
    assert not evidence['visible'] and evidence['landmarks'] is None
    raw = record_hand_observations([hand])
    assert raw[0]['handedness'] == 'Right' and len(raw[0]['landmarks_normalized']) == 21
    np.testing.assert_allclose(raw[0]['landmarks_pixel'], hand.landmarks_pixel)
    json.dumps(raw, allow_nan=False)
    # Even invalid detector values can be logged without killing RGB capture.
    hand.landmarks_normalized[0, 0] = np.nan
    assert record_hand_observations([hand])[0]['landmarks_normalized'][0][0] is None
    assert record_hand_observations([]) == []


def test_exact_parking_reference_survives_pose_fault_and_is_an_owned_copy():
    p = FusionHandover()
    q = np.arange(7)*.01
    p.begin_hold(q)
    expected = q.copy()
    q[:] = 99.
    p.fault('test pose fault')
    snap = p.snapshot()
    np.testing.assert_array_equal(snap['hold_q_rad'], expected)
    assert snap['hold_drift_limit_rad'] == p.config.hold_drift_rad
    snap['hold_q_rad'][0] = 99.
    assert p.hold_q[0] == 0.


def write_session(root, split, *, group='grasp01', broken=False):
    directory = root/split/group/'handover_fusion/run_p123'
    directory.mkdir(parents=True)
    (directory/'metadata.json').write_text(json.dumps(dict(control_mode='mit_cartesian_hold',
        global_serial='test_serial', mode='shadow', dry_run=False,
        collection=dict(group=group, split=split, labels_verified=False))))
    records = []
    for i in range(3):
        feedback = dict(q=[0.]*7, dq_raw=[0.]*7, tau=[1.]*7,
                        motor_timestamps=[100.+i]*7, position_timestamps=[100.+i]*7)
        if broken and i == 0:
            feedback['tau'][0] = None
        records.append(dict(feedback=feedback, policy=dict(
            phase='FUSION_FAULT' if i == 2 else 'FUSION_WAIT_TAKE',
            baseline_nm=[0.]*7, hold_q_rad=[.1]*7),
            visual=dict(sequence=i, raw_hand_observations=[]),
            events=[dict(action='label', label='TAKE_PULL')] if i == 1 else []))
    (directory/'samples.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
    (directory/'global').mkdir()
    (directory/'global/global_rgb.avi').write_bytes(b'fixture not an actual video')
    return directory


def test_inventory_distinguishes_post_fault_raw_data_and_unverified_labels(tmp_path):
    write_session(tmp_path, 'train')
    report = inventory(tmp_path)
    session = report['sessions'][0]
    assert session['unique_motor_snapshots'] == 3
    assert session['raw_snapshots_before_fault'] == 2
    assert session['raw_snapshots_at_or_after_fault'] == 1
    assert session['visual_frames_with_raw_hand_records'] == 3
    assert session['hold_reference_saved']
    assert not session['labels_verified'] and not session['ready_for_training']
    assert session['label_key_events'] == {'TAKE_PULL': 1}
    assert session['label_key_events_before_fault'] == {'TAKE_PULL': 1}
    assert session['label_key_events_at_or_after_fault'] == {}
    assert report['physical_grasps_by_split']['train'] == ['grasp01']


def test_same_physical_grasp_cannot_be_treated_as_independent_train_and_test(tmp_path):
    write_session(tmp_path, 'train')
    write_session(tmp_path, 'test')
    report = inventory(tmp_path)
    assert report['conflicting_group_assignments'] == {'grasp01': ['test', 'train']}
    assert not report['ready_for_training']


def test_inventory_reports_empty_capture_and_skips_invalid_feedback(tmp_path):
    directory = write_session(tmp_path, 'train', broken=True)
    assert inventory(tmp_path)['sessions'][0]['unique_motor_snapshots'] == 2
    (directory/'samples.jsonl').write_text('')
    session = inventory(tmp_path)['sessions'][0]
    assert session['sample_rows'] == 0
    assert 'no seven-axis raw feedback' in session['capture_issues']
    assert 'no completed baseline' in session['capture_issues']


def test_collection_source_snapshot_preserves_external_calibration_without_modifying_it(tmp_path):
    import hashlib
    from handoff.fusion_capture_snapshot import save_capture_snapshot
    calibration = tmp_path/'test_calibration.json'
    calibration.write_text('{"purpose": "offline fixture"}')
    before = calibration.read_bytes()
    result = save_capture_snapshot(NS(handeye=str(calibration)), tmp_path/'session')
    directory = tmp_path/'session'/result['directory']
    manifest = json.loads((directory/'manifest.json').read_text())
    assert result['files'] == len(manifest)
    assert (directory/'external/test_calibration.json').read_bytes() == before
    assert calibration.read_bytes() == before
    for entry in manifest:
        assert hashlib.sha256((directory/entry['saved_as']).read_bytes()).hexdigest() == entry['sha256']


def test_inventory_reports_inner_fault_even_when_outer_policy_never_faulted(tmp_path):
    directory = write_session(tmp_path, 'train')
    records = [json.loads(x) for x in (directory/'samples.jsonl').read_text().splitlines()]
    for r in records:
        r['policy']['phase'] = 'FUSION_WAIT_TAKE'
    (directory/'samples.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
    diagnostic = directory.parent.parent/'servo_diagnostics/session_p123'
    diagnostic.mkdir(parents=True)
    (diagnostic/'control_cycles.jsonl').write_text(json.dumps(dict(type='fault',
        reason='MIT wall clock changed by +417.1ms', wall_time=100., monotonic_time=1.))+'\n')
    report = inventory(tmp_path)['sessions'][0]
    assert 'FUSION_FAULT' not in report['phases']
    assert report['inner_control_faults'][0]['reason'] == 'MIT wall clock changed by +417.1ms'
    assert report['review_flags'] == ['inner_control_fault_recorded']
    assert not report['ready_for_training']
