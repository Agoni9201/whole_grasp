import types
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from real_nero_grasp.pointcloud_fusion import (
    _apply_T_points,
    _nn_stats,
    _rigid_icp_registration,
)


class ConservativeIcpTest(unittest.TestCase):
    @staticmethod
    def _args():
        return types.SimpleNamespace(
            active_reg_min_pairs=40,
            active_reg_icp_overlap_thr=0.018,
            active_reg_icp_min_overlap_ratio=0.08,
            active_fusion_overlap_max_median=0.003,
            active_fusion_overlap_max_p95=0.018,
            active_fusion_overlap_min_inlier_1cm=0.80,
            active_reg_icp_voxel=0.0,
            active_reg_icp_thr=0.008,
            active_reg_icp_thresholds="0.015,0.008",
            active_reg_icp_max_iter=25,
            active_reg_icp_max_trans=0.010,
            active_reg_icp_max_rot_deg=3.0,
            active_reg_icp_min_fit=0.10,
            active_reg_icp_estimation="point_to_point",
            active_reg_icp_robust_kernel=0.005,
            active_reg_min_improve=0.04,
        )

    def test_partial_view_improves_within_hard_envelope(self):
        rng = np.random.default_rng(4)
        target = rng.normal(size=(900, 3)) * np.array([0.045, 0.018, 0.012])
        target += np.array([-0.40, 0.02, 0.18])
        correction = np.eye(4)
        correction[:3, :3] = Rotation.from_euler("z", 1.5, degrees=True).as_matrix()
        pivot = np.median(target, axis=0)
        local = target - pivot
        source = (local @ correction[:3, :3]) + pivot
        source += np.array([0.004, -0.003, 0.002])
        source = source[source[:, 0] > np.quantile(source[:, 0], 0.15)]
        source += rng.normal(scale=0.0004, size=source.shape)

        before = _nn_stats(source, target)
        T, info = _rigid_icp_registration(source, target, self._args())
        after = _nn_stats(_apply_T_points(source, T), target)

        self.assertTrue(info["applied"])
        self.assertLess(after["median"], before["median"])
        self.assertLessEqual(info["trans"], 0.010000001)
        self.assertLessEqual(info["rot_deg"], 3.000001)


if __name__ == "__main__":
    unittest.main()
