import sys
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

from real_nero_grasp import cli, collision
from real_nero_grasp.types import DynamicEnvironmentMap, PlanNotExecutable


def motion_args(**overrides):
    values = {
        "motion_environment_map_enabled": True,
        "motion_path_scene_voxel": 0.0,
        "motion_path_obstacle_cap": 100,
        "motion_path_fail_on_obstacle_cap": True,
        "motion_path_require_scene": True,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class MotionEnvironmentMapCollisionTest(unittest.TestCase):
    def test_cli_table_map_defaults_and_explicit_overrides(self):
        with mock.patch.object(
            sys,
            "argv",
            ["runner", "--calib-json", "calib.json"],
        ):
            defaults = cli.parse_args()
        self.assertTrue(defaults.motion_environment_map_enabled)
        self.assertEqual(defaults.motion_environment_map_voxel, 0.005)
        self.assertEqual(defaults.motion_environment_map_radius_xy, 0.45)
        self.assertIsNone(defaults.motion_environment_map_x_min)
        self.assertIsNone(defaults.motion_environment_map_x_max)
        self.assertIsNone(defaults.motion_environment_map_y_min)
        self.assertIsNone(defaults.motion_environment_map_y_max)
        self.assertEqual(defaults.motion_environment_map_z_below_table, 0.02)
        self.assertEqual(defaults.motion_environment_map_z_above_table, 0.60)
        self.assertEqual(defaults.motion_environment_map_min_view_support, 2)
        self.assertEqual(
            defaults.motion_environment_map_single_view_local_radius,
            0.020,
        )
        self.assertEqual(
            defaults.motion_environment_map_single_view_neighbor_radius,
            0.009,
        )
        self.assertEqual(
            defaults.motion_environment_map_single_view_min_neighbors,
            2,
        )
        self.assertTrue(defaults.motion_collision_isolated_filter)
        self.assertEqual(defaults.motion_collision_isolated_radius, 0.020)
        self.assertEqual(defaults.motion_collision_isolated_min_support, 8)

        argv = [
            "runner",
            "--calib-json",
            "calib.json",
            "--no-motion-environment-map",
            "--motion-environment-map-x-min",
            "-0.60",
            "--motion-environment-map-x-max",
            "-0.10",
            "--motion-environment-map-y-min",
            "-0.30",
            "--motion-environment-map-y-max",
            "0.20",
        ]
        with mock.patch.object(sys, "argv", argv):
            explicit = cli.parse_args()
        self.assertFalse(explicit.motion_environment_map_enabled)
        self.assertEqual(explicit.motion_environment_map_x_min, -0.60)
        self.assertEqual(explicit.motion_environment_map_x_max, -0.10)
        self.assertEqual(explicit.motion_environment_map_y_min, -0.30)
        self.assertEqual(explicit.motion_environment_map_y_max, 0.20)

    def test_collision_filter_removes_isolated_voxels_but_keeps_continuous_rim(self):
        rim = np.column_stack(
            [
                np.linspace(0.0, 0.09, 10),
                np.zeros(10),
                np.full(10, 0.12),
            ]
        )
        ghost = np.array([[0.30, 0.30, 0.20], [0.31, 0.30, 0.20]])
        args = motion_args(
            motion_collision_isolated_filter=True,
            motion_collision_isolated_radius=0.021,
            motion_collision_isolated_min_support=3,
        )

        filtered = collision.filter_motion_isolated_environment_points(
            np.vstack([rim, ghost]), args, "unit"
        )

        self.assertEqual(filtered.shape[0], rim.shape[0])
        np.testing.assert_allclose(filtered, rim.astype(np.float32))

    def test_dynamic_map_is_consumed_directly_in_base_link(self):
        environment_map = DynamicEnvironmentMap(
            frame_id="base_link",
            hard_environment_base=np.array([[0.10, 0.00, 0.10]]),
            uncertain_environment_base=np.array([[0.20, 0.00, 0.10]]),
            target_base=np.array([[0.15, 0.00, 0.12]]),
            map_id="table_map",
            generation=7,
        )
        transform = mock.Mock(side_effect=AssertionError("must not transform base map"))
        scene = collision.prepare_motion_safety_scene(
            target_cloud=None,
            motion_scene_cloud=None,
            T_base_camera=None,
            args=motion_args(),
            map_id="candidate_preflight",
            environment_map=environment_map,
            transform_cloud_hook=transform,
        )
        transform.assert_not_called()
        np.testing.assert_allclose(
            scene.environment_base,
            np.array(
                [[0.10, 0.00, 0.10], [0.20, 0.00, 0.10]],
                dtype=np.float32,
            ),
        )
        np.testing.assert_allclose(
            scene.full_scene_base,
            np.array(
                [
                    [0.10, 0.00, 0.10],
                    [0.20, 0.00, 0.10],
                    [0.15, 0.00, 0.12],
                ],
                dtype=np.float32,
            ),
        )
        self.assertIs(scene.environment_map, environment_map)
        self.assertEqual(scene.generation, 7)
        self.assertEqual(scene.map_id, "candidate_preflight")

    def test_final_planning_target_overrides_stale_dynamic_map_target(self):
        class Cloud:
            def __init__(self, points):
                self.points = np.asarray(points, dtype=np.float32).reshape(-1, 3)

        stale_target = np.array([[0.15, 0.00, 0.071]], dtype=np.float32)
        final_target = Cloud([[0.16, 0.01, 0.091], [0.17, 0.01, 0.095]])
        environment_map = DynamicEnvironmentMap(
            frame_id="base_link",
            hard_environment_base=np.array([[0.10, 0.00, 0.10]]),
            uncertain_environment_base=np.zeros((0, 3)),
            target_base=stale_target,
            map_id="stale_target_map",
            generation=8,
        )

        scene = collision.prepare_motion_safety_scene(
            target_cloud=final_target,
            motion_scene_cloud=None,
            T_base_camera=np.eye(4),
            args=motion_args(),
            map_id="candidate_preflight",
            environment_map=environment_map,
            transform_cloud_hook=lambda cloud, _transform: cloud,
        )

        np.testing.assert_allclose(scene.target_base, final_target.points)
        self.assertFalse(np.any(np.all(scene.target_base == stale_target[0], axis=1)))
        np.testing.assert_allclose(
            scene.environment_base,
            environment_map.hard_environment_base,
        )

    def test_dynamic_map_rejects_non_base_frame(self):
        environment_map = SimpleNamespace(
            frame_id="camera0",
            hard_environment_base=np.zeros((0, 3)),
            uncertain_environment_base=np.zeros((0, 3)),
            target_base=np.ones((1, 3)),
            generation=1,
        )
        with self.assertRaisesRegex(PlanNotExecutable, "frame_id.*base_link"):
            collision.prepare_motion_safety_scene(
                None,
                None,
                None,
                motion_args(),
                environment_map=environment_map,
            )

    def test_dynamic_map_missing_contract_field_fails_closed(self):
        environment_map = SimpleNamespace(
            frame_id="base_link",
            hard_environment_base=np.ones((1, 3)),
            target_base=np.ones((1, 3)),
            generation=1,
        )
        with self.assertRaisesRegex(
            PlanNotExecutable,
            "missing required field.*uncertain_environment_base",
        ):
            collision.prepare_motion_safety_scene(
                None,
                None,
                None,
                motion_args(),
                environment_map=environment_map,
            )

    def test_single_view_clear_preflight_promotes_partition_to_dynamic_map(self):
        class Cloud:
            def __init__(self, points):
                self.points = np.asarray(points, dtype=np.float32).reshape(-1, 3)

        args = motion_args(
            motion_environment_map_voxel=0.01,
            motion_environment_map_radius_xy=0.20,
            motion_environment_map_min_view_support=2,
            motion_environment_map_single_view_local_radius=0.021,
            motion_environment_map_single_view_neighbor_radius=0.011,
            motion_environment_map_single_view_min_neighbors=2,
            table_z=0.0,
        )
        target = Cloud([[0.0, 0.0, 0.05]])
        environment = Cloud(
            [
                [0.05, 0.00, 0.05],
                [0.06, 0.00, 0.05],
                [0.07, 0.00, 0.05],
            ]
        )
        scene = Cloud(
            np.vstack(
                [
                    target.points,
                    environment.points,
                ]
            )
        )

        safety_scene = collision.prepare_motion_safety_scene(
            target,
            scene,
            np.eye(4),
            args,
            map_id="clear1_single_preflight",
            motion_environment_cloud=environment,
            transform_cloud_hook=lambda cloud, _transform: cloud,
        )

        self.assertIsInstance(
            safety_scene.environment_map,
            DynamicEnvironmentMap,
        )
        self.assertEqual(safety_scene.generation, 1)
        self.assertEqual(
            safety_scene.environment_map.source_view_names,
            ("clear1_single_preflight_single",),
        )
        np.testing.assert_allclose(
            safety_scene.environment_map.hard_environment_base,
            environment.points,
        )
        np.testing.assert_allclose(
            safety_scene.target_base,
            target.points,
        )


if __name__ == "__main__":
    unittest.main()
