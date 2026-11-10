"""Camera-only reference format and independent-frame sampling; no hardware."""
import json
from types import SimpleNamespace as NS

import numpy as np

from scripts import capture_self_mask_reference as capture


def frame(sequence, stamp):
    return NS(bgr=np.zeros((4, 6, 3), np.uint8), depth_m=np.ones((4, 6), np.float32),
        raw_depth_m=np.ones((4, 6), np.float32), backend='orbbec', serial='test',
        intr=NS(fx=10., fy=11., cx=2., cy=2.), K=np.eye(3), D=np.zeros(8),
        sequence=sequence, hardware_ms=stamp*1000, received_monotonic=stamp)


def test_saved_metadata_preserves_geometry_without_claiming_opening_measurement(tmp_path):
    args = NS(camera_depth_width=8, camera_depth_height=5, camera_width=6, camera_height=4)
    meta = capture.save_reference(tmp_path/'sample', frame(7, 1.), args, 100.)
    assert meta['width'] == 6 and meta['native_depth_width'] == 8
    assert meta['opening_declared_mm'] == 100 and not meta['opening_verified_by_this_tool']
    assert not meta['arm_or_gripper_commands_sent']
    assert (tmp_path/'sample/color.png').exists()
    assert json.loads((tmp_path/'sample/capture_meta.json').read_text())['sequence'] == 7


def test_capture_requires_four_distinct_time_separated_frames_and_closes_camera(monkeypatch, tmp_path):
    frames = iter([frame(1, 1.), frame(1, 1.), frame(2, 1.2), frame(3, 2.), frame(4, 3.), frame(5, 4.)])
    current = [1.]
    def read():
        value = next(frames)
        current[0] = value.received_monotonic
        return value
    closed = []
    monkeypatch.setattr(capture, 'camera_for', lambda _: NS(read=read))
    monkeypatch.setattr(capture, 'close_all', lambda: closed.append(True))
    monkeypatch.setattr(capture.time, 'monotonic', lambda: current[0])
    capture.main(['--opening-mm', '100', '--samples', '4', '--save-dir', str(tmp_path)])
    manifest = json.loads(next(tmp_path.glob('*/manifest.json')).read_text())
    assert [r['sequence'] for r in manifest['samples']] == [1, 3, 4, 5]
    assert closed == [True] and not manifest['opening_verified_by_this_tool']
