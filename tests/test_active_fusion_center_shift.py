import tempfile
import unittest
from types import SimpleNamespace

import numpy as np

from real_nero_grasp.pointcloud_fusion import diagnose_active_fusion


class ActiveFusionCenterShiftTest(unittest.TestCase):
    def args(self):
        return SimpleNamespace(
            active_fusion_diagnostics=True,
            active_reg_voxel=0.0025,
            active_fusion_reject_delta=0.015,
            active_fusion_overlap_override_delta=0.025,
            active_fusion_overlap_max_median=0.006,
            active_fusion_overlap_max_p95=0.030,
            active_fusion_overlap_min_inlier_1cm=0.55,
            active_fusion_reject_axis_mismatch=False,
            active_fusion_reject_axis_deg=35.0,
            active_fusion_warn_delta=0.015,
            active_fusion_reject_bad_views=True,
        )

    @staticmethod
    def stats(cloud):
        center = np.asarray(cloud, dtype=float).reshape(-1, 3)[0]
        return {
            "n": 1000,
            "median": center,
            "pca_axis": np.array([1.0, 0.0, 0.0]),
            "extent": np.array([0.16, 0.09, 0.06]),
        }

    def run_gate(self, overlap):
        center = np.array([[0.0, 0.0, 0.20]])
        side = np.array([[0.040, 0.0, 0.20]])
        calls = {"count": 0}

        def nn_stats(_src, _target):
            calls["count"] += 1
            return dict(overlap)

        with tempfile.TemporaryDirectory() as out_dir:
            result = diagnose_active_fusion(
                ["center", "side1"],
                [center, side],
                np.eye(4),
                out_dir,
                self.args(),
                cloud_stats_hook=self.stats,
                points_from_cloud_hook=lambda cloud: cloud,
                downsample_points_hook=lambda points, _voxel: points,
                nn_stats_hook=nn_stats,
                save_cloud_hook=lambda *_args, **_kwargs: None,
                transform_cloud_hook=lambda cloud, _T: cloud,
            )
        self.assertEqual(calls["count"], 1)
        return result

    def test_large_partial_visibility_center_shift_is_kept_by_good_overlap(self):
        keep = self.run_gate(
            {"median": 0.0054, "p95": 0.0221, "inlier@1cm": 0.653}
        )
        self.assertEqual(keep, [True, True])

    def test_large_center_shift_is_rejected_when_overlap_is_bad(self):
        keep = self.run_gate(
            {"median": 0.012, "p95": 0.045, "inlier@1cm": 0.30}
        )
        self.assertEqual(keep, [True, False])


if __name__ == "__main__":
    unittest.main()
