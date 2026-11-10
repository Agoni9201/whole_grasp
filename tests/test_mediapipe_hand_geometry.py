import numpy as np

from targets.mediapipe_hand_landmarker import deproject_pinhole, robust_patch_depth


def test_robust_patch_depth_ignores_invalid_and_outlier_values():
    depth = np.full((9, 9), 0.50, dtype=np.float32)
    depth[3, 3] = 0.0
    depth[4, 4] = np.nan
    depth[5, 5] = 1.80
    median, mad, count = robust_patch_depth(depth, 4, 4, radius=2, max_depth_m=2.0)
    assert abs(median - 0.50) < 1e-6
    assert mad < 1e-6
    assert count == 23


def test_deproject_pinhole_uses_color_optical_axes_and_metres():
    K = np.array([[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]])
    np.testing.assert_allclose(deproject_pinhole(320, 240, 0.6, K), [0.0, 0.0, 0.6])
    np.testing.assert_allclose(deproject_pinhole(420, 190, 0.5, K), [0.1, -0.05, 0.5])
