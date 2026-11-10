import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from tools.calibrate_d435i_aruco import transform, solve, residuals


def test_eye_in_hand_direction_and_units():
    rng = np.random.default_rng(32)
    x = transform(Rotation.from_euler('xyz',[.15,-.25,.1]).as_matrix(),[.03,-.04,.08])
    board = transform(Rotation.from_euler('xyz',[.4,.1,-.2]).as_matrix(),[.5,.1,.4])
    samples = []
    for _ in range(16):
        b = transform(Rotation.from_rotvec(rng.normal(0,.4,3)).as_matrix(),rng.uniform(-.2,.2,3))
        c = np.linalg.inv(x) @ np.linalg.inv(b) @ board
        samples.append(dict(T_base_flange=b.tolist(),T_camera_marker=c.tolist()))
    actual, reference = solve(samples[:12])
    np.testing.assert_allclose(actual,x,atol=1e-8)
    report = residuals(samples[12:], actual, reference)
    assert max(report['translation_mm']) < 1e-5
    assert max(report['rotation_deg']) < 1e-5


def test_repeated_views_rejected():
    s = dict(T_base_flange=np.eye(4).tolist(),T_camera_marker=np.eye(4).tolist())
    with pytest.raises(ValueError,match='independent axes'):
        solve([s]*12)
