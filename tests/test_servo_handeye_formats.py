"""Calibration compatibility; no detector, model, camera, or robot started."""
import json
from pathlib import Path

import numpy as np
import pytest

from targets.aruco_eye_in_hand_target import ArUcoEyeInHandTarget, load_transform_file


def load(tmp_path, data):
    path = tmp_path / 'calibration.json'
    path.write_text(json.dumps(data))
    return load_transform_file(path)


@pytest.mark.parametrize('kind', ['legacy', 'mapping', 'metric', 'flange', 'matrix'])
def test_supported_pose_formats_preserve_translation_and_direction(tmp_path, kind):
    xyz, quat = [.03, -.07, -.01], [0., 0., 1., 0.]
    expected = np.diag([-1., -1., 1., 1.])
    expected[:3, 3] = xyz
    formats = {
        'legacy': dict(position=xyz, orientation=quat),
        'mapping': dict(translation=dict(zip('xyz', xyz)), rotation_xyzw=dict(zip('xyzw', quat))),
        'metric': dict(translation_m=xyz, quaternion_xyzw=quat),
        'flange': dict(flange_camera_xyzw=xyz + quat),
        'matrix': dict(matrix_4x4=expected.tolist()),
    }
    np.testing.assert_allclose(load(tmp_path, formats[kind]), expected)


def test_actual_charuco_export_matches_matrix_and_explicit_inverse():
    root = Path(__file__).resolve().parents[1]
    path = root / 'config/2026-10-19_16-15-29_calibration.json'
    data = json.loads(path.read_text())
    expected = np.asarray(data['matrix_4x4'])
    np.testing.assert_allclose(load_transform_file(path), expected, atol=1e-12)
    target = ArUcoEyeInHandTarget.__new__(ArUcoEyeInHandTarget)
    target.handeye_path = str(path)
    target.handeye_convention = 'link7_cam'
    np.testing.assert_allclose(target.load_calibration(), expected, atol=1e-12)
    target.handeye_convention = 'cam_mount'
    np.testing.assert_allclose(target.load_calibration(), np.linalg.inv(expected), atol=1e-12)


def test_conflicting_redundant_transforms_are_rejected(tmp_path):
    with pytest.raises(ValueError, match='Inconsistent'):
        load(tmp_path, dict(translation_m=[.1, 0, 0], quaternion_xyzw=[0, 0, 0, 1],
                            matrix_4x4=np.eye(4).tolist()))


@pytest.mark.parametrize('matrix', [np.zeros((4, 4)), np.diag([-1., 1., 1., 1.]),
                                   np.diag([2., 1., 1., 1.]), np.full((4, 4), np.nan), np.eye(3)])
def test_invalid_matrices_are_rejected(tmp_path, matrix):
    with pytest.raises(ValueError, match='Invalid rigid'):
        load(tmp_path, dict(matrix_4x4=matrix.tolist()))
