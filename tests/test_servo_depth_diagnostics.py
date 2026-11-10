"""Raw-depth evidence only: no camera, detector, CAN, or display opened."""
import json
from types import SimpleNamespace as NS

import numpy as np
import pytest

from servo.depth_diagnostics import analyze_frame, depth_statistics, save_frame
from servo.diagnostics import ServoDiagnostics, _ServoDiagnosticWriter


def frame():
    depth = np.full((30, 40), 0.5, dtype=np.float32)
    depth[8:13, 8:13] = 0
    return dict(depth_m=depth, bgr=np.zeros((30, 40, 3), dtype=np.uint8),
                K=np.eye(3), D=np.zeros(5), q_image=np.arange(7.), sequence=123,
                received_monotonic=1.0, reason="invalid_palm_depth", radius=2,
                minimum_m=0.1, maximum_m=2.0,
                landmarks_pixel=np.full((21, 2), 10), palm_pixel=(10, 10))


def test_statistics_distinguish_missing_from_rejected_measurements():
    values = np.array([0, -1, np.nan, np.inf, .05, .1, .5, 2., 3.])
    summary = depth_statistics(values, .1, 2.)
    assert summary['zero_or_negative'] == 2
    assert summary['nonfinite'] == 2
    assert summary['below_minimum'] == 1
    assert summary['above_maximum'] == 1
    assert summary['valid'] == 3
    assert summary['positive_depth_p10_p50_p90_m'][1] == .5


def test_health_distinguishes_hand_holes_from_scene_wide_depth_loss():
    value = frame()
    summary = analyze_frame(value)
    assert summary['scene']['valid_percent'] > 90
    assert len(summary['regions']) == 6
    assert all(r['zero_or_negative'] == 25 for r in summary['regions'].values())
    value['depth_m'][:] = 0
    assert analyze_frame(value)['scene']['valid_percent'] == 0


def test_archive_preserves_raw_depth_intrinsics_and_frame_identity(tmp_path):
    import cv2
    value = frame()
    value['depth_m'][0, 0] = np.nan
    value['metadata_json'] = json.dumps(analyze_frame(value))
    path = save_frame(tmp_path, value, 1)
    with np.load(path, allow_pickle=False) as data:
        np.testing.assert_array_equal(data['depth_m'], value['depth_m'])
        np.testing.assert_array_equal(data['K'], value['K'])
        np.testing.assert_array_equal(data['landmarks_pixel'], value['landmarks_pixel'])
        assert json.loads(str(data['metadata_json']))['frame_sequence'] == 123
    assert cv2.imread(path.replace('.npz', '.png')).shape == (30, 80, 3)


def test_snapshot_collection_is_bounded_and_disabled_for_live_control(tmp_path):
    class Writer:
        def __init__(self): self.records = []
        def submit(self, record): self.records.append(record)

    for dry_run in (False, True):
        writer = Writer()
        reporter = ServoDiagnostics(tmp_path / 'samples.jsonl', writer=writer,
                                    metadata={'dry_run': dry_run})
        for tick in range(100):
            reporter.observe(dict(q=np.zeros(7), detection='lost', rgbd_frame=frame()),
                             model=NS(fk=lambda q: (np.zeros(3), np.eye(3))), cfg=None,
                             status='LOST', reason='invalid_palm_depth', stage='TRACK_HAND',
                             now=float(tick), wall_time=1000. + tick)
        captured = [r for r in writer.records if '_rgbd_frame' in r]
        assert len(captured) == (12 if dry_run else 0)
        assert all(b['monotonic_time'] - a['monotonic_time'] >= 5.
                   for a, b in zip(captured, captured[1:]))


def test_background_writer_logs_depth_evidence_and_strips_arrays(tmp_path):
    path = tmp_path / 'samples.jsonl'
    writer = _ServoDiagnosticWriter(path)
    writer.submit({'type': 'depth_test', 'timestamp': 'test',
                   '_rgbd_frame': frame(), '_rgbd_index': 1})
    # A bounded join waits only for this tiny offline test artifact.
    writer.stop.set()
    writer.thread.join(timeout=5.)
    assert not writer.thread.is_alive()
    record = json.loads(path.read_text())
    assert '_rgbd_frame' not in record
    assert record['depth_health']['regions']['palm']['valid'] == 0
    assert record['rgbd_snapshot'].endswith('frame_01_seq_123.npz')


@pytest.mark.parametrize('postprocess', [False, True])
def test_servo_evidence_uses_detected_frame_and_excludes_duplicate_ticks(postprocess):
    import time
    from unittest.mock import Mock
    from cameras.sdk_loop import Image, CameraInfo
    from servo_delivery import ServoDeliveryNode, _NO_NEW_FRAME
    value = frame()
    node = ServoDeliveryNode.__new__(ServoDeliveryNode)
    node._diagnostics = object()
    node._diag_sample = {}
    node._latest_image = Image(40, 30, 'bgr8', 120, value['bgr'].tobytes())
    node._latest_info = CameraInfo(np.eye(3).ravel().tolist(), [])
    node._latest_depth = np.full_like(value['depth_m'], .5) if postprocess else value['depth_m']
    node._latest_raw_depth = value['depth_m'] if postprocess else None
    node._latest_q = np.zeros(7)
    node._sdk_received_monotonic = time.monotonic()
    node._sdk_frame_sequence = 123
    node._consumed_sequence = None
    node.args = NS(target_source='mediapipe', hand_depth_radius=2,
                   hand_min_depth=.1, hand_max_depth=2.)
    node.target = NS(detect=Mock(return_value=None),
                     last_diagnostics={'reason': 'invalid_palm_depth'},
                     last_observation=NS(landmarks_pixel=value['landmarks_pixel'],
                                         palm_pixel=value['palm_pixel']))
    assert node._detect_latest(None, time.time()) is None
    evidence = node._diag_sample['rgbd_frame']
    assert evidence['sequence'] == 123
    if postprocess:
        assert evidence['raw_depth_m'] is value['depth_m']
        assert evidence['filtered_depth_m'] is node._latest_depth
        assert np.count_nonzero(evidence['depth_m']) == np.count_nonzero(value['depth_m'])
        np.testing.assert_array_equal(node.target.detect.call_args.args[1], evidence['depth_m'])
    else:
        assert evidence['depth_m'] is value['depth_m']
    assert evidence['reason'] == 'invalid_palm_depth'
    np.testing.assert_array_equal(evidence['bgr'], value['bgr'])
    node._diag_sample = {}
    assert node._detect_latest(None, time.time()) is _NO_NEW_FRAME
    assert 'rgbd_frame' not in node._diag_sample
    node.target.detect.assert_called_once()
