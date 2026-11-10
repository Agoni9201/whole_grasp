import unittest
from types import SimpleNamespace

import numpy as np

from real_nero_grasp.grasp_workflow import _prepare_graspnet_target_cloud


class _FakeCloud:
    def __init__(self, points, labels):
        self.points = np.asarray(points, dtype=float)
        self._labels = np.asarray(labels, dtype=np.int64)

    def cluster_dbscan(self, eps, min_points, print_progress=False):
        return self._labels.copy()

    def select_by_index(self, indices):
        indices = np.asarray(indices, dtype=np.int64)
        return _FakeCloud(self.points[indices], self._labels[indices])


def _args(dominance=0.85):
    return SimpleNamespace(
        graspnet_main_component_filter=True,
        graspnet_component_eps=0.004,
        graspnet_component_min_points=8,
        graspnet_component_min_dominance=dominance,
    )


class GraspNetInputComponentFilterTest(unittest.TestCase):
    def test_dominant_component_removes_small_island(self):
        cloud = _FakeCloud(np.zeros((105, 3)), [0] * 100 + [1] * 5)
        cleaned = _prepare_graspnet_target_cloud(cloud, _args())
        self.assertEqual(len(cleaned.points), 100)

    def test_split_target_is_kept_when_no_component_dominates(self):
        cloud = _FakeCloud(np.zeros((100, 3)), [0] * 60 + [1] * 40)
        cleaned = _prepare_graspnet_target_cloud(cloud, _args())
        self.assertIs(cleaned, cloud)


if __name__ == "__main__":
    unittest.main()
