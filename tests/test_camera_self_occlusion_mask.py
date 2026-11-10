import unittest
from types import SimpleNamespace

import numpy as np

import real_nero_single_grasp_runner as runner


class CameraSelfOcclusionMaskTest(unittest.TestCase):
    def setUp(self):
        self.args = SimpleNamespace(
            camera_self_mask=True,
            camera_self_mask_expand_px=12,
            camera_self_edge_island_filter=True,
            camera_self_edge_island_distance_px=3,
            camera_self_edge_island_max_area=8,
            camera_self_edge_island_max_diameter=6,
            camera_self_edge_island_max_depth=0.22,
            min_depth=0.05,
        )
        self.image_shape = (720, 1280)
        # This first pixel is the measured left-finger leak from the
        # 2026-08-04 empty-grasp recovery diagnostic.
        self.self_pixels = ((485, 492), (557, 609), (1065, 644))
        self.basket_pixel = (650, 475)

    def test_known_gripper_pixels_are_masked_but_basket_is_not(self):
        mask = runner.camera_self_occlusion_mask(self.image_shape, self.args)

        self.assertEqual(mask.shape, self.image_shape)
        for x, y in self.self_pixels:
            self.assertGreater(mask[y, x], 0)
        basket_x, basket_y = self.basket_pixel
        self.assertEqual(mask[basket_y, basket_x], 0)

    def test_apply_sets_only_known_self_pixels_to_nan(self):
        xyz = np.ones((*self.image_shape, 3), dtype=np.float32)
        bgr = np.zeros((*self.image_shape, 3), dtype=np.uint8)

        filtered = runner.apply_camera_self_occlusion_mask(
            xyz,
            bgr,
            self.args,
            out_dir=None,
        )

        for x, y in self.self_pixels:
            self.assertTrue(np.isnan(filtered[y, x]).all())
        basket_x, basket_y = self.basket_pixel
        np.testing.assert_array_equal(filtered[basket_y, basket_x], xyz[basket_y, basket_x])

    def test_tiny_near_depth_island_one_pixel_outside_self_mask_is_removed(self):
        xyz = np.full((*self.image_shape, 3), np.nan, dtype=np.float32)
        bgr = np.zeros((*self.image_shape, 3), dtype=np.uint8)
        mask = runner.camera_self_occlusion_mask(self.image_shape, self.args)
        # Reproduce the 2026-08-17 right-finger edge leak: one near-depth
        # pixel immediately outside the calibrated mask.
        y, x = 556, 1181
        self.assertEqual(mask[y, x], 0)
        self.assertGreater(mask[y, x - 1], 0)
        xyz[y, x] = [0.1293, 0.0455, 0.1660]

        filtered = runner.apply_camera_self_occlusion_mask(
            xyz, bgr, self.args, out_dir=None
        )

        self.assertTrue(np.isnan(filtered[y, x]).all())

    def test_large_near_component_touching_self_mask_is_retained(self):
        xyz = np.full((*self.image_shape, 3), np.nan, dtype=np.float32)
        bgr = np.zeros((*self.image_shape, 3), dtype=np.uint8)
        mask = runner.camera_self_occlusion_mask(self.image_shape, self.args)
        y, x = 556, 1181
        self.assertEqual(mask[y, x], 0)
        # A real close surface extends well beyond the edge band and exceeds
        # the tiny-component limits.
        xyz[y - 5 : y + 6, x : x + 11, 0] = 0.12
        xyz[y - 5 : y + 6, x : x + 11, 1] = 0.04
        xyz[y - 5 : y + 6, x : x + 11, 2] = 0.18

        filtered = runner.apply_camera_self_occlusion_mask(
            xyz, bgr, self.args, out_dir=None
        )

        self.assertTrue(np.isfinite(filtered[y, x]).all())


if __name__ == "__main__":
    unittest.main()
