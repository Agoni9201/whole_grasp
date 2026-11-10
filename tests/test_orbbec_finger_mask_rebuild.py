"""Fixed offline reference rebuilding, without USB/CAN or guard relaxation."""
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from tools.rebuild_orbbec_finger_mask import measured_fingers, rebuild
from real_nero_grasp.camera_self_mask import CameraSelfMaskMismatch, calibrated_self_mask


def setup_reference(tmp_path, shift_validation=0):
    c = dict(schema_version=2, profile=dict(serial='test', width=320, height=240, depth_width=320, depth_height=240),
        intrinsics=dict(fx=200., fy=200., cx=160., cy=120.), gripper_open_width_m=.1,
        rgb_guard=dict(max_gray=80, min_core_dark_fraction=.995,
                       min_silhouette_dark_fraction=.97, max_outer_dark_fraction=.15),
        mask_color_guard=dict(max_channel_spread=20), provenance=dict(roi_polygons_px=[
            [[60, 120], [135, 120], [135, 239], [60, 239]],
            [[220, 120], [300, 120], [300, 239], [220, 239]]]))
    previous = tmp_path/'previous.json'
    previous.write_text(json.dumps(c))
    captures = []
    for i in range(4):
        directory = tmp_path/str(i)
        directory.mkdir()
        image = np.full((240, 320, 3), 190, np.uint8)
        dx = shift_validation if i == 3 else 0
        image[140:240, 75+dx:125+dx] = 5+i
        image[135:240, 235+dx:290+dx] = 5+i
        cv2.line(image, (10, 170), (80+dx, 170), (0, 0, 0), 3)
        cv2.imwrite(str(directory/'color.png'), image)
        meta = dict(serial='test', width=320, height=240, native_depth_width=320, native_depth_height=240,
                    intrinsics=c['intrinsics'])
        (directory/'capture_meta.json').write_text(json.dumps(meta))
        captures.append(directory)
    return c, previous, captures


def test_rebuild_keeps_guards_fixed_and_removes_thin_cable_branches(tmp_path):
    c, previous, captures = setup_reference(tmp_path)
    before = previous.read_bytes()
    output, reports = tmp_path/'new.json', tmp_path/'report'
    result = rebuild(previous, captures[:3], captures[3:], output, reports)
    saved = json.loads(output.read_text())
    assert saved['rgb_guard'] == c['rgb_guard'] and previous.read_bytes() == before
    assert [r['split'] for r in result] == ['build']*3+['heldout']
    image = cv2.imread(str(captures[3]/'color.png'))
    meta = json.loads((captures[3]/'capture_meta.json').read_text())
    mask, report = calibrated_self_mask(image, output, meta)
    assert report['validation_passed']
    assert mask[200, 100] == 255 and mask[200, 260] == 255
    assert not mask[165:176, 10:55].any()  # Cable remains outside the finger mask.
    shifted = np.roll(image, 28, axis=1)
    with pytest.raises(CameraSelfMaskMismatch): calibrated_self_mask(shifted, output, meta)


def test_failed_validation_does_not_publish_a_calibration_or_change_prior(tmp_path):
    _, previous, captures = setup_reference(tmp_path, shift_validation=25)
    before = previous.read_bytes()
    output = tmp_path/'new.json'
    with pytest.raises(CameraSelfMaskMismatch):
        rebuild(previous, captures[:3], captures[3:], output, tmp_path/'report')
    assert not output.exists() and previous.read_bytes() == before


def test_rebuild_rejects_training_validation_reuse(tmp_path):
    _, previous, captures = setup_reference(tmp_path)
    with pytest.raises(ValueError, match='distinct'):
        rebuild(previous, captures[:3], captures[:1], tmp_path/'new.json', tmp_path/'report')


def test_roi_component_must_actually_connect_to_image_bottom(tmp_path):
    c, _, _ = setup_reference(tmp_path)
    image = np.full((240, 320, 3), 190, np.uint8)
    image[140:190, 75:125] = 0
    image[140:190, 235:290] = 0
    with pytest.raises(ValueError, match='bottom-connected'):
        measured_fingers(image, c)
