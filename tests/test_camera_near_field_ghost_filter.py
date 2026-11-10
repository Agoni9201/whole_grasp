import unittest
from types import SimpleNamespace

import numpy as np

import real_nero_single_grasp_runner as runner


class CameraNearFieldGhostFilterTest(unittest.TestCase):
    def setUp(self):
        self.shape = (720, 1280)
        self.args = SimpleNamespace(
            min_depth=0.05,
            camera_near_field_ghost_filter=True,
            camera_near_field_ghost_image_size=(1280, 720),
            camera_near_field_ghost_roi=(0.05, 0.28, 0.75, 1.00),
            camera_near_field_ghost_seed_depth=0.175,
            camera_near_field_ghost_grow_depth=0.200,
            camera_near_field_ghost_min_seed_pixels=8,
            camera_near_field_ghost_max_area=5000,
            camera_near_field_ghost_max_diameter=120,
            camera_near_field_preserve_dilate_px=8,
        )
        self.bgr = np.zeros((*self.shape, 3), dtype=np.uint8)

    def xyz_with_background(self, depth=0.50):
        xyz = np.zeros((*self.shape, 3), dtype=np.float32)
        xyz[:, :, 2] = float(depth)
        return xyz

    def test_removes_dense_near_plateau_and_its_connected_halo(self):
        xyz = self.xyz_with_background()
        # The failing real capture contained a dense 0.163 m block around this
        # normalized location.  A 0.19 m fringe must only be removed because it
        # is connected to that verified very-near seed.
        xyz[612:657, 155:220, 2] = 0.190
        xyz[618:651, 161:215, 2] = 0.163

        filtered = runner.apply_camera_near_field_ghost_filter(
            xyz, self.bgr, self.args, out_dir=None
        )

        self.assertTrue(np.isnan(filtered[620, 170]).all())
        self.assertTrue(np.isnan(filtered[613, 156]).all())
        self.assertFalse(np.isnan(filtered[600, 170]).any())
        self.assertAlmostEqual(float(filtered[600, 170, 2]), 0.50)

    def test_preserves_selected_target_and_dilated_border(self):
        xyz = self.xyz_with_background()
        xyz[610:660, 160:215, 2] = 0.163
        target = np.zeros(self.shape, dtype=np.uint8)
        target[620:650, 170:205] = 1

        filtered = runner.apply_camera_near_field_ghost_filter(
            xyz,
            self.bgr,
            self.args,
            out_dir=None,
            preserve_mask=target,
        )

        self.assertFalse(np.isnan(filtered[635, 185]).any())
        self.assertAlmostEqual(float(filtered[635, 185, 2]), 0.163)
        self.assertFalse(np.isnan(filtered[615, 165]).any())
        self.assertAlmostEqual(float(filtered[615, 165, 2]), 0.163)

    def test_removes_isolated_calibrated_near_field_pixel_when_configured(self):
        xyz = self.xyz_with_background()
        xyz[590, 229, 2] = 0.173
        self.args.camera_near_field_ghost_min_seed_pixels = 1

        filtered = runner.apply_camera_near_field_ghost_filter(
            xyz, self.bgr, self.args, out_dir=None
        )

        self.assertTrue(np.isnan(filtered[590, 229]).all())
        self.assertFalse(np.isnan(filtered[590, 230]).any())

    def test_retains_near_points_outside_calibrated_roi(self):
        xyz = self.xyz_with_background()
        xyz[620:650, 600:640, 2] = 0.163

        filtered = runner.apply_camera_near_field_ghost_filter(
            xyz, self.bgr, self.args, out_dir=None
        )

        self.assertFalse(np.isnan(filtered[635, 620]).any())
        self.assertAlmostEqual(float(filtered[635, 620, 2]), 0.163)

    def test_retains_long_or_large_close_component_fail_closed(self):
        xyz = self.xyz_with_background()
        xyz[600:620, 100:250, 2] = 0.163

        rejected, stats = runner.camera_near_field_ghost_mask(xyz, self.args)

        self.assertEqual(int(np.count_nonzero(rejected)), 0)
        self.assertEqual(stats["retained_seeded_components"], 1)

    def test_sizes_full_component_when_real_obstacle_crosses_roi(self):
        xyz = self.xyz_with_background()
        # Only a small slice lies inside the calibrated ROI, but the complete
        # close component is long and must therefore remain fail-closed.
        xyz[600:620, 250:500, 2] = 0.163

        rejected, stats = runner.camera_near_field_ghost_mask(xyz, self.args)

        self.assertEqual(int(np.count_nonzero(rejected)), 0)
        self.assertEqual(stats["retained_seeded_components"], 1)

    def test_profile_mismatch_disables_filter_fail_closed(self):
        xyz = self.xyz_with_background()
        xyz[620:650, 170:205, 2] = 0.163
        args = SimpleNamespace(**vars(self.args))
        args.camera_near_field_ghost_image_size = (848, 480)

        rejected, stats = runner.camera_near_field_ghost_mask(xyz, args)

        self.assertEqual(int(np.count_nonzero(rejected)), 0)
        self.assertTrue(stats["profile_mismatch"])

    def test_filter_can_be_disabled_without_changing_xyz(self):
        xyz = self.xyz_with_background()
        xyz[620:650, 170:205, 2] = 0.163
        args = SimpleNamespace(**vars(self.args))
        args.camera_near_field_ghost_filter = False

        filtered = runner.apply_camera_near_field_ghost_filter(
            xyz, self.bgr, args, out_dir=None
        )

        np.testing.assert_array_equal(filtered, xyz)


if __name__ == "__main__":
    unittest.main()
