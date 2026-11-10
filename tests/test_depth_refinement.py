import unittest

import numpy as np

import real_nero_single_grasp_runner as runner
from tools.verify_handeye_point import Intrinsics, median_valid_depth_frames


class DepthMedianTest(unittest.TestCase):
    def test_ignores_invalid_values_and_rejects_single_frame_pixels(self):
        frames = np.array(
            [
                [[0.50, 0.00], [0.70, 0.80]],
                [[0.52, 0.00], [0.00, 0.82]],
                [[0.90, 0.60], [0.00, 0.84]],
            ],
            dtype=np.float32,
        )

        depth, count = median_valid_depth_frames(frames, min_valid_frames=2)

        self.assertAlmostEqual(float(depth[0, 0]), 0.52, places=5)
        self.assertEqual(float(depth[0, 1]), 0.0)
        self.assertEqual(float(depth[1, 0]), 0.0)
        self.assertAlmostEqual(float(depth[1, 1]), 0.82, places=5)
        np.testing.assert_array_equal(count, np.array([[3, 1], [1, 3]], dtype=np.uint8))

    def test_temporal_mad_rejects_flying_pixel_but_preserves_stable_surface(self):
        frames = np.array(
            [
                [[0.500, 0.500]],
                [[0.501, 0.535]],
                [[0.499, 0.470]],
                [[0.500, 0.545]],
                [[0.501, 0.465]],
            ],
            dtype=np.float32,
        )

        depth, count, mad, unstable = median_valid_depth_frames(
            frames,
            min_valid_frames=5,
            max_mad=0.004,
            return_stats=True,
        )

        self.assertEqual(int(count[0, 0]), 5)
        self.assertAlmostEqual(float(depth[0, 0]), 0.500, places=4)
        self.assertLess(float(mad[0, 0]), 0.004)
        self.assertFalse(bool(unstable[0, 0]))
        self.assertEqual(float(depth[0, 1]), 0.0)
        self.assertGreater(float(mad[0, 1]), 0.004)
        self.assertTrue(bool(unstable[0, 1]))


class SmallDepthHoleFillTest(unittest.TestCase):
    def setUp(self):
        self.intr = Intrinsics(fx=100.0, fy=100.0, cx=6.0, cy=6.0)
        self.mask = np.zeros((13, 13), dtype=np.uint8)
        self.mask[1:12, 1:12] = 1
        z = np.full((13, 13), 0.50, dtype=np.float32)
        self.xyz = runner.depth_to_xyz_image(z, self.intr)

    def fill(self, xyz):
        return runner.fill_small_xyz_holes_in_mask(
            xyz,
            self.mask,
            self.intr,
            min_depth=0.1,
            max_depth=1.0,
            max_area=9,
            max_diameter=3,
            ring_radius=2,
            min_neighbors=8,
            max_depth_span=0.015,
        )

    def test_fills_tiny_enclosed_consistent_hole(self):
        xyz = self.xyz.copy()
        xyz[6, 6] = np.nan

        refined, fill_mask, stats = self.fill(xyz)

        self.assertAlmostEqual(float(refined[6, 6, 2]), 0.50, places=5)
        self.assertEqual(int(fill_mask[6, 6]), 255)
        self.assertEqual(stats["filled_pixels"], 1)

    def test_does_not_fill_hole_touching_visible_mask_boundary(self):
        xyz = self.xyz.copy()
        xyz[1, 6] = np.nan

        refined, fill_mask, stats = self.fill(xyz)

        self.assertTrue(np.isnan(refined[1, 6]).all())
        self.assertEqual(int(fill_mask[1, 6]), 0)
        self.assertEqual(stats["filled_pixels"], 0)

    def test_does_not_bridge_depth_discontinuity(self):
        xyz = self.xyz.copy()
        xyz[:, 7:, 2] = 0.55
        xyz[6, 6] = np.nan

        refined, fill_mask, stats = self.fill(xyz)

        self.assertTrue(np.isnan(refined[6, 6]).all())
        self.assertEqual(int(fill_mask[6, 6]), 0)
        self.assertEqual(stats["filled_pixels"], 0)


if __name__ == "__main__":
    unittest.main()
