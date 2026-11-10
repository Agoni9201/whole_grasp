"""Offline contract checks; no camera, ROS, CAN, or GUI."""
from types import SimpleNamespace
import numpy as np
import pytest
from tools.test_d435i_grasp_pointcloud import xyz_from_frame


def frame():
    return np.zeros((2, 3, 3), dtype=np.uint8), SimpleNamespace(fx=100., fy=100., cx=1., cy=0.)


def test_known_metric_point_and_invalid_depth():
    bgr, intr = frame()
    depth = np.array([[0., 0.5, np.nan], [np.inf, 0.5, -1.]], dtype=np.float32)
    xyz = xyz_from_frame(bgr, depth, intr)
    assert xyz.shape == (2, 3, 3)
    assert xyz.dtype == np.float32
    np.testing.assert_allclose(xyz[0, 1], [0., 0., 0.5])
    np.testing.assert_allclose(xyz[1, 1], [0., 0.005, 0.5])
    for row, col in [(0, 0), (0, 2), (1, 0), (1, 2)]:
        assert np.isnan(xyz[row, col]).all()


def test_raw_integer_depth_cannot_be_mistaken_for_metres():
    bgr, intr = frame()
    with pytest.raises(ValueError, match="metres"):
        xyz_from_frame(bgr, np.full((2, 3), 500, dtype=np.uint16), intr)


def test_mismatched_sizes_are_rejected():
    bgr, intr = frame()
    with pytest.raises(ValueError, match="matching"):
        xyz_from_frame(bgr, np.ones((3, 2), dtype=np.float32), intr)


def test_bad_focal_length_is_rejected():
    bgr, intr = frame()
    intr.fx = 0.
    with pytest.raises(ValueError, match="intrinsics"):
        xyz_from_frame(bgr, np.ones((2, 3), dtype=np.float32), intr)
