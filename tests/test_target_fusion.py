import unittest

import numpy as np

from real_nero_grasp.pointcloud_fusion import fuse_target_clouds


class _FakeCloud:
    def __init__(self, points=None, colors=None):
        self.points = np.zeros((0, 3), dtype=np.float32) if points is None else np.asarray(points, dtype=np.float32).reshape(-1, 3)
        self.colors = np.zeros((0, 3), dtype=np.float32) if colors is None else np.asarray(colors, dtype=np.float32).reshape(-1, 3)

    def remove_statistical_outlier(self, **_kwargs):
        return self, np.arange(len(self.points))


class _FakeO3D:
    class geometry:
        PointCloud = _FakeCloud

    class utility:
        @staticmethod
        def Vector3dVector(points):
            return np.asarray(points, dtype=np.float32).reshape(-1, 3)


class TargetFusionTest(unittest.TestCase):
    def test_independent_views_are_weighted_once_per_voxel(self):
        center = _FakeCloud(
            [[0.001, 0.0, 0.10], [0.031, 0.0, 0.10]],
            [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        )
        side = _FakeCloud(
            [[0.003, 0.0, 0.10], [0.006, 0.0, 0.10]],
            [[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
        )

        cloud, stats = fuse_target_clouds(
            [center, side],
            voxel_size=0.01,
            single_view_radius=0.012,
            single_view_min_neighbors=1,
            return_stats=True,
            get_o3d_hook=lambda: _FakeO3D,
        )

        # The isolated 31 mm point disappears.  The two center/side samples
        # inside the first output voxel contribute equally despite the side
        # view carrying an extra point in that voxel.
        self.assertEqual(len(cloud.points), 1)
        np.testing.assert_allclose(cloud.points[0], [0.00275, 0.0, 0.10])
        np.testing.assert_allclose(cloud.colors[0], [0.5, 0.5, 0.0])
        self.assertEqual(stats["multi_view_voxels"], 1)
        self.assertEqual(stats["removed_isolated"], 1)

    def test_coherent_single_view_target_surface_is_retained(self):
        cloud, stats = fuse_target_clouds(
            [
                _FakeCloud(
                    [[0.000, 0.0, 0.10], [0.006, 0.0, 0.10], [0.012, 0.0, 0.10]],
                    [[1.0, 0.0, 0.0]] * 3,
                )
            ],
            voxel_size=0.005,
            single_view_radius=0.008,
            single_view_min_neighbors=1,
            return_stats=True,
            get_o3d_hook=lambda: _FakeO3D,
        )

        self.assertEqual(len(cloud.points), 3)
        self.assertEqual(stats["removed_isolated"], 0)


if __name__ == "__main__":
    unittest.main()
