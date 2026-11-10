import unittest
from types import SimpleNamespace

import numpy as np

from real_nero_grasp.active_view import (
    ActiveViewHooks,
    _look_at_camera_pose_flange_roll_candidates,
    _target_center_optical_relaxation_pose_records,
)


class ActiveCenterOpticalRelaxationTest(unittest.TestCase):
    def test_relaxation_keeps_camera_position_and_moves_only_optical_axis(self):
        args = SimpleNamespace(
            active_center_height=0.30,
            active_center_optical_tilts="2,4",
            active_center_optical_azimuth_step_deg=90.0,
            active_center_optical_max_directions=8,
            active_center_roll_samples=8,
            active_view_max_center_norm=0.20,
            calib_json="unused.json",
            handeye_convention="flange_camera_xyzw",
        )
        target = np.array([-0.30, -0.02, 0.20], dtype=float)
        overhead_camera = target + np.array([0.0, 0.0, 0.30])
        hooks = ActiveViewHooks(
            handeye_transform=lambda *_args, **_kwargs: np.eye(4),
            look_at_camera_pose_flange_roll_candidates=(
                _look_at_camera_pose_flange_roll_candidates
            ),
        )

        records = _target_center_optical_relaxation_pose_records(
            args,
            target,
            np.zeros(6, dtype=float),
            np.array([0.10, 0.0, 0.50], dtype=float),
            hooks=hooks,
        )

        self.assertEqual(len(records), 8)
        for record in records:
            np.testing.assert_allclose(record["camera_position"], overhead_camera)
            self.assertAlmostEqual(
                record["center_norm"],
                np.tan(np.deg2rad(record["tilt_deg"])),
                places=6,
            )
            self.assertGreater(record["target_depth"], 0.0)
            self.assertEqual(len(record["flange_candidates"]), 8)


if __name__ == "__main__":
    unittest.main()
