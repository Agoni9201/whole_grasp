import json
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

try:
    import open3d as o3d
except ImportError:
    o3d = None

from real_nero_grasp.cli import parse_args
from real_nero_grasp import pointcloud_fusion as fusion


@unittest.skipIf(o3d is None, "Open3D is required for geometric registration")
class ActiveSceneRegistrationTest(unittest.TestCase):
    @staticmethod
    def args():
        return SimpleNamespace(
            active_fusion_register=True,
            active_scene_table_registration=True,
            active_reg_apply_to_scene=True,
            active_fusion_translation_calib=True,
            active_fusion_rigid_icp=False,
            active_reg_voxel=0.0025,
            active_reg_min_pairs=60,
            active_reg_max_median=0.008,
            active_reg_max_p95=0.020,
            active_reg_icp_overlap_thr=0.018,
            active_reg_icp_min_overlap_ratio=0.08,
            scene_voxel_size=0.004,
            voxel_size=0.0015,
            table_z=0.020,
            active_table_reg_z_window=0.040,
            active_table_reg_inlier=0.003,
            active_table_reg_min_points=250,
            active_table_reg_max_tilt_deg=5.0,
            active_table_reg_max_height_shift=0.015,
            active_table_reg_max_residual_tilt_deg=0.5,
        )

    @staticmethod
    def cloud(points):
        return o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))

    def captures(self, tilt_deg=3.0):
        x, y = np.meshgrid(np.linspace(-0.16, 0.16, 32), np.linspace(-0.12, 0.12, 28))
        table = np.column_stack((x.ravel(), y.ravel(), np.full(x.size, 0.020)))
        tx, ty = np.meshgrid(np.linspace(-0.030, 0.030, 15), np.linspace(-0.012, 0.012, 9))
        target = np.column_stack((tx.ravel(), ty.ravel(), np.full(tx.size, 0.065)))
        static = np.array([[0.12, 0.10, 0.05], [0.13, 0.10, 0.06]])
        reference_scene = np.vstack((table, static, target))
        T = np.eye(4)
        T[:3, :3] = Rotation.from_euler("y", tilt_deg, degrees=True).as_matrix()
        T[:3, 3] = [0.008, 0.0, 0.004]
        source_scene = fusion._apply_T_points(reference_scene, np.linalg.inv(T))
        source_target = fusion._apply_T_points(target, np.linalg.inv(T))
        return ([self.cloud(target), self.cloud(source_target)],
                [self.cloud(reference_scene), self.cloud(source_scene)], len(table) + len(static))

    @staticmethod
    def local_translation(source, reference, _args):
        T = np.eye(4)
        T[:3, 3] = np.median(reference, axis=0) - np.median(source, axis=0)
        return T, {"applied": True, "reason": "accepted", "shift": T[:3, 3]}

    def test_common_table_correction_and_scene_object_refresh_preserve_background(self):
        targets, scenes, background_count = self.captures()
        original = np.asarray(scenes[1].points).copy()
        with tempfile.TemporaryDirectory() as out_dir:
            corrected_targets, corrected_scenes, keep = fusion.register_active_fusion_views(
                ["center", "side1"], targets, scenes, np.eye(4), out_dir, self.args(),
                small_translation_registration_hook=self.local_translation,
                save_cloud_hook=lambda *_: None,
            )
            with open(out_dir + "/active_fusion_registration.json", encoding="utf-8") as stream:
                report = json.load(stream)
        self.assertEqual(keep, [True, True])
        view = report["views"][0]
        self.assertIn("created_at", report)
        self.assertTrue(view["table"]["applied"])
        self.assertLess(view["table"]["residual_tilt_deg"], 0.5)
        self.assertEqual(view["replaced_scene_target_points"], len(targets[1].points))
        T_scene = np.array(view["background_transform"])
        T_target = np.array(view["target_transform"])
        self.assertGreater(np.linalg.norm(T_target[:3, 3] - T_scene[:3, 3]), 0.005)
        expected_background = fusion._apply_T_points(original[:background_count], T_scene)
        expected = np.vstack((expected_background, np.asarray(corrected_targets[1].points)))
        actual = np.asarray(corrected_scenes[1].points)
        self.assertEqual(len(actual), len(expected))
        self.assertLess(float(cKDTree(actual).query(expected)[0].max()), 1e-6)
        np.testing.assert_array_equal(np.asarray(scenes[1].points), original)

    def test_table_correction_outside_envelope_rejects_view(self):
        targets, scenes, _ = self.captures(tilt_deg=8.0)
        with tempfile.TemporaryDirectory() as out_dir:
            _, _, keep = fusion.register_active_fusion_views(
                ["center", "side1"], targets, scenes, np.eye(4), out_dir, self.args(),
                small_translation_registration_hook=self.local_translation,
                save_cloud_hook=lambda *_: None,
            )
            with open(out_dir + "/active_fusion_registration.json", encoding="utf-8") as stream:
                report = json.load(stream)
        self.assertFalse(keep[1])
        self.assertIn("table_alignment=tilt_outside_envelope", report["views"][0]["reject_reasons"])

    def test_capture_and_final_registration_use_same_table_correction(self):
        targets, scenes, _ = self.captures()
        args = self.args()
        self.assertTrue(fusion._active_side_capture_overlap_ok(
            targets[0], targets[1], args, "active_side1",
            reference_scene_base_cloud=scenes[0], side_scene_base_cloud=scenes[1],
            small_translation_registration_hook=self.local_translation,
        ))
        targets_bad, scenes_bad, _ = self.captures(tilt_deg=8.0)
        self.assertFalse(fusion._active_side_capture_overlap_ok(
            targets_bad[0], targets_bad[1], args, "active_side1",
            reference_scene_base_cloud=scenes_bad[0], side_scene_base_cloud=scenes_bad[1],
            small_translation_registration_hook=self.local_translation,
        ))

    def test_residual_table_mismatch_rolls_back_correction(self):
        args = self.args()
        plane = {"normal": np.array([0., 0., 1.]), "d": -0.02,
                 "centroid": np.array([0., 0., 0.02])}
        after = dict(plane, normal=Rotation.from_euler("y", 1.0, degrees=True).apply(plane["normal"]))
        with mock.patch.object(fusion, "_fit_table_plane", side_effect=[
            (plane, {}), (plane, {}), (after, {}),
        ]):
            T, info = fusion._round_table_plane_correction(
                np.zeros((300, 3)), np.zeros((300, 3)), np.array([[0., 0., 0.06]]), args,
            )
        np.testing.assert_array_equal(T, np.eye(4))
        self.assertFalse(info["applied"])
        self.assertEqual(info["reason"], "table_residual_outside_envelope")


class ActiveSceneRegistrationDefaultsTest(unittest.TestCase):
    def test_table_registration_enabled_and_explicit_disable_preserved(self):
        for flags, expected in (([], True), (["--no-active-scene-table-registration"], False)):
            with self.subTest(flags=flags), mock.patch.object(
                sys, "argv", ["main.py", "--phase", "grasp", "--calib-json", "unit_calibration.json", *flags],
            ):
                self.assertEqual(parse_args().active_scene_table_registration, expected)
