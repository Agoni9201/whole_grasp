import copy
import unittest
from dataclasses import FrozenInstanceError
from types import SimpleNamespace
from unittest import mock

import numpy as np

from real_nero_grasp.active_view import (
    ActiveViewHooks,
    _build_dynamic_motion_environment,
)
from real_nero_grasp.pointcloud_fusion import (
    fuse_dynamic_environment_clouds,
    next_dynamic_environment_generation,
)
from real_nero_grasp.types import DynamicEnvironmentMap
from real_nero_grasp.types import (
    ActiveViewResult,
    GraspAttemptResult,
    GraspAttemptStatus,
)
from real_nero_grasp.workflow import (
    WorkflowHooks,
    record_workflow_motion,
    run_single_grasp,
    with_workflow_motion,
)


class FakeCloud:
    def __init__(self, points):
        self.points = np.asarray(points, dtype=np.float32).reshape(-1, 3)


def map_args(**overrides):
    values = dict(
        motion_environment_map_enabled=True,
        motion_environment_map_voxel=0.01,
        motion_environment_map_radius_xy=0.20,
        motion_environment_map_x_min=None,
        motion_environment_map_x_max=None,
        motion_environment_map_y_min=None,
        motion_environment_map_y_max=None,
        motion_environment_map_z_below_table=0.01,
        motion_environment_map_z_above_table=0.20,
        motion_environment_map_min_view_support=2,
        motion_environment_map_single_view_local_radius=0.021,
        motion_environment_map_single_view_neighbor_radius=0.011,
        motion_environment_map_single_view_min_neighbors=2,
        motion_environment_map_isolated_filter=True,
        table_z=0.0,
        place_xyz=None,
        clear_place_xyz=None,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


class DynamicEnvironmentMapTest(unittest.TestCase):
    def test_generation_counter_is_shared_by_shallow_workflow_copies(self):
        args = SimpleNamespace()
        self.assertEqual(next_dynamic_environment_generation(args), 1)
        copied_args = copy.copy(args)
        self.assertEqual(next_dynamic_environment_generation(copied_args), 2)
        self.assertEqual(
            args._motion_environment_generation_owner["value"],
            2,
        )

    def test_active_map_reaches_top_level_target_planner_unchanged(self):
        environment_map = DynamicEnvironmentMap(
            hard_environment_base=[[0.1, 0.0, 0.1]],
            uncertain_environment_base=np.zeros((0, 3)),
            target_base=[[0.0, 0.0, 0.1]],
            generation=4,
        )
        cloud = FakeCloud([[0.0, 0.0, 0.1]])
        capture = (
            np.zeros((2, 2, 3), dtype=np.uint8),
            cloud,
            cloud,
            "/tmp",
            {"label": "carrot"},
            object(),
        )
        planner_kwargs = {}

        class Arm:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def set_tcp_offset(self, _offset):
                return None

            def get_joint_angles(self):
                return np.zeros(7)

        def arm_factory(*_args, **_kwargs):
            return Arm()

        def target_planner(*_args, **kwargs):
            planner_kwargs.update(kwargs)
            return GraspAttemptResult(GraspAttemptStatus.PLANNED_ONLY)

        hooks = WorkflowHooks(
            arm_factory=arm_factory,
            configure_amodal_clear_model=lambda _args: None,
            test_current_flange_ik=lambda *_args: None,
            prepare_initial_pose_for_capture=lambda *_args: None,
            capture_target_cloud_with_retries=lambda _args: capture,
            preview_reconstructed_cloud=lambda *_args: None,
            T_base_camera_from_flange=lambda *_args: np.eye(4),
            maybe_clear_obstacles_before_active=lambda *_args: capture,
            run_optional_active_view=lambda *_args: ActiveViewResult(
                target_cloud=cloud,
                scene_cloud=cloud,
                T_base_camera=np.eye(4),
                motion_scene_cloud=cloud,
                motion_environment_cloud=cloud,
                motion_environment_map=environment_map,
            ),
            plan_and_maybe_execute_grasp=target_planner,
            record_workflow_motion=record_workflow_motion,
            with_workflow_motion=with_workflow_motion,
        )
        args = SimpleNamespace(
            nero_interface="can0",
            nero_channel="can0",
            nero_bitrate=1_000_000,
            nero_firmware="DEFAULT",
            nero_enable_timeout=1.0,
            nero_speed_percent=5,
            disable_gripper=False,
            gripper_hardware_max_width=0.085,
            gripper_hardware_max_force=0.5,
            tcp_offset=np.zeros(6),
            ik_test_current=False,
            gripper_test=None,
        )

        result = run_single_grasp(args, hooks=hooks)

        self.assertEqual(result.status, GraspAttemptStatus.PLANNED_ONLY)
        self.assertIs(
            planner_kwargs["motion_environment_map"],
            environment_map,
        )

    def test_active_view_map_builder_respects_disabled_switch(self):
        args = SimpleNamespace(
            motion_environment_map_enabled=False,
            _motion_environment_generation=4,
        )
        fuser = mock.Mock(side_effect=AssertionError("must not be called"))

        result = _build_dynamic_motion_environment(
            ["first"],
            [FakeCloud([[0.0, 0.0, 0.1]])],
            args,
            np.eye(4),
            hooks=ActiveViewHooks(fuse_dynamic_environment_clouds=fuser),
        )

        self.assertEqual(result, (None, None))
        self.assertEqual(args._motion_environment_generation, 4)
        fuser.assert_not_called()

    def test_active_view_map_builder_rejects_mismatched_sources(self):
        args = SimpleNamespace(motion_environment_map_enabled=True)
        fuser = mock.Mock()

        with self.assertRaisesRegex(ValueError, "view_names=2 clouds=1"):
            _build_dynamic_motion_environment(
                ["first", "side"],
                [FakeCloud([[0.0, 0.0, 0.1]])],
                args,
                np.eye(4),
                hooks=ActiveViewHooks(
                    fuse_dynamic_environment_clouds=fuser
                ),
            )

        fuser.assert_not_called()
        self.assertFalse(hasattr(args, "_motion_environment_generation"))

    def test_active_view_map_builder_preserves_two_view_order_and_facade(self):
        args = SimpleNamespace(
            motion_environment_map_enabled=True,
            _motion_environment_generation=2,
        )
        seen = {}

        def fake_fuser(names, clouds, _args, **kwargs):
            seen["names"] = tuple(names)
            seen["clouds"] = tuple(clouds)
            seen["kwargs"] = kwargs
            return SimpleNamespace(
                hard_environment_base=np.array(
                    [[0.1, 0.0, 0.2]], dtype=np.float32
                ),
                uncertain_environment_base=np.array(
                    [[0.2, 0.0, 0.2]], dtype=np.float32
                ),
            )

        first = FakeCloud([[0.1, 0.0, 0.2]])
        side = FakeCloud([[0.2, 0.0, 0.2]])
        hooks = ActiveViewHooks(
            fuse_dynamic_environment_clouds=fake_fuser,
            make_cloud_from_points_colors=FakeCloud,
            transform_cloud=lambda cloud, _T: cloud,
        )

        environment_map, compatibility_cloud = (
            _build_dynamic_motion_environment(
                ["first", "side"],
                [first, side],
                args,
                np.eye(4),
                hooks=hooks,
                target_base=np.array([[0.0, 0.0, 0.2]]),
                map_id="two_view_test",
                target_center_base=np.array([0.0, 0.0, 0.2]),
            )
        )

        self.assertIsNotNone(environment_map)
        self.assertEqual(seen["names"], ("first", "side"))
        self.assertEqual(seen["clouds"], (first, side))
        self.assertEqual(seen["kwargs"]["generation"], 3)
        self.assertEqual(seen["kwargs"]["map_id"], "two_view_test")
        self.assertEqual(args._motion_environment_generation, 3)
        np.testing.assert_allclose(
            compatibility_cloud.points,
            [[0.1, 0.0, 0.2], [0.2, 0.0, 0.2]],
        )

    def test_compatibility_cloud_failure_keeps_canonical_map(self):
        args = SimpleNamespace(motion_environment_map_enabled=True)
        expected_map = SimpleNamespace(
            hard_environment_base=np.array([[0.1, 0.0, 0.2]]),
            uncertain_environment_base=np.zeros((0, 3)),
        )
        hooks = ActiveViewHooks(
            fuse_dynamic_environment_clouds=lambda *_args, **_kwargs: expected_map,
            make_cloud_from_points_colors=mock.Mock(
                side_effect=RuntimeError("Open3D allocation failed")
            ),
            transform_cloud=lambda cloud, _transform: cloud,
        )

        environment_map, compatibility_cloud = (
            _build_dynamic_motion_environment(
                ["first"],
                [FakeCloud([[0.1, 0.0, 0.2]])],
                args,
                np.eye(4),
                hooks=hooks,
            )
        )

        self.assertIs(environment_map, expected_map)
        self.assertIsNone(compatibility_cloud)

    def test_fault_during_map_build_is_propagated_without_compatibility_cloud_or_extra_stop(self):
        from real_nero_grasp.types import SafetyStopTriggered
        state = {'fault': False}
        arm = mock.Mock(_safety_stop_latched=False)
        def check(_label):
            if state['fault']:
                raise SafetyStopTriggered('background MIT fault')
        owner = SimpleNamespace(owns_control=True, _arm=arm, _raise_if_terminal_fault=check)
        args = SimpleNamespace(_mit_session=owner, motion_environment_map_enabled=True)
        def fuse(*a, **kw):
            state['fault'] = True
            return SimpleNamespace(hard_environment_base=np.zeros((1, 3)),
                                   uncertain_environment_base=np.zeros((0, 3)))
        make = mock.Mock()
        with self.assertRaisesRegex(SafetyStopTriggered, 'background MIT fault'):
            _build_dynamic_motion_environment(
                ['first'], [FakeCloud([[0., 0., .1]])], args, np.eye(4),
                hooks=ActiveViewHooks(fuse_dynamic_environment_clouds=fuse,
                                      make_cloud_from_points_colors=make),
            )
        make.assert_not_called()
        arm.terminate_motion.assert_not_called()
        arm.electronic_emergency_stop.assert_not_called()

    def test_duplicate_samples_count_once_per_view_and_support_counts_views(self):
        point = [0.001, 0.001, 0.051]
        result = fuse_dynamic_environment_clouds(
            ["center", "side"],
            [FakeCloud([point, point, point]), FakeCloud([point])],
            map_args(),
            target_base=np.array([[0.0, 0.0, 0.05]], dtype=np.float32),
            map_id="unit_map",
            generation=7,
            target_center_base=np.array([0.0, 0.0, 0.05]),
        )

        self.assertIsInstance(result, DynamicEnvironmentMap)
        self.assertEqual(result.voxel_keys.shape, (1, 3))
        np.testing.assert_array_equal(result.view_support, [2])
        self.assertEqual(result.hard_environment_base.shape, (1, 3))
        self.assertEqual(result.uncertain_environment_base.shape, (0, 3))
        self.assertEqual(result.source_view_names, ("center", "side"))
        self.assertEqual(result.frame_id, "base_link")
        self.assertEqual(result.map_id, "unit_map")
        self.assertEqual(result.generation, 7)
        self.assertFalse(result.hard_environment_base.flags.writeable)
        self.assertFalse(result.uncertain_environment_base.flags.writeable)
        self.assertFalse(result.target_base.flags.writeable)
        self.assertFalse(result.voxel_keys.flags.writeable)
        self.assertFalse(result.view_support.flags.writeable)
        self.assertFalse(result.confidence.flags.writeable)
        self.assertFalse(result.roi_bounds_base.flags.writeable)
        with self.assertRaises(FrozenInstanceError):
            result.generation = 8

    def test_collision_layers_keep_measured_points_not_synthetic_centers(self):
        measured = np.array([[0.001, 0.002, 0.053]], dtype=np.float32)
        result = fuse_dynamic_environment_clouds(
            ["center", "side"],
            [FakeCloud(measured), FakeCloud(measured)],
            map_args(),
            target_center_base=[0.0, 0.0, 0.05],
        )

        np.testing.assert_allclose(result.hard_environment_base, measured)

    def test_duplicate_samples_in_one_view_do_not_fake_multiview_support(self):
        point = [0.001, 0.001, 0.051]
        result = fuse_dynamic_environment_clouds(
            ["center"],
            [FakeCloud([point, point, point])],
            map_args(),
            target_center_base=[0.0, 0.0, 0.05],
        )

        self.assertEqual(result.view_support.shape, (0,))
        self.assertEqual(result.hard_environment_base.shape, (0, 3))
        self.assertEqual(result.uncertain_environment_base.shape, (0, 3))

    def test_continuous_single_view_thin_rim_is_hard(self):
        rim = np.array(
            [
                [0.001, 0.001, 0.051],
                [0.011, 0.001, 0.051],
                [0.021, 0.001, 0.051],
                [0.031, 0.001, 0.051],
            ],
            dtype=np.float32,
        )
        result = fuse_dynamic_environment_clouds(
            ["side"],
            [FakeCloud(rim)],
            map_args(),
            target_center_base=[0.0, 0.0, 0.05],
        )

        self.assertEqual(result.hard_environment_base.shape, (4, 3))
        self.assertEqual(result.uncertain_environment_base.shape, (0, 3))
        np.testing.assert_array_equal(result.view_support, np.ones(4, dtype=int))
        np.testing.assert_allclose(result.confidence, np.full(4, 0.75))

    def test_isolated_single_view_voxel_is_dropped(self):
        result = fuse_dynamic_environment_clouds(
            ["side"],
            [FakeCloud([[0.001, 0.001, 0.051]])],
            map_args(),
            target_center_base=[0.0, 0.0, 0.05],
        )

        self.assertEqual(result.hard_environment_base.shape, (0, 3))
        self.assertEqual(result.uncertain_environment_base.shape, (0, 3))
        self.assertEqual(result.confidence.shape, (0,))

    def test_isolated_filter_can_be_disabled_for_compatibility(self):
        result = fuse_dynamic_environment_clouds(
            ["side"],
            [FakeCloud([[0.001, 0.001, 0.051]])],
            map_args(motion_environment_map_isolated_filter=False),
            target_center_base=[0.0, 0.0, 0.05],
        )

        self.assertEqual(result.hard_environment_base.shape, (0, 3))
        self.assertEqual(result.uncertain_environment_base.shape, (1, 3))

    def test_explicit_xy_and_table_roi_remove_outside_points(self):
        args = map_args(
            motion_environment_map_isolated_filter=False,
            motion_environment_map_x_min=-0.05,
            motion_environment_map_x_max=0.05,
            motion_environment_map_y_min=-0.04,
            motion_environment_map_y_max=0.04,
            motion_environment_map_z_below_table=0.01,
            motion_environment_map_z_above_table=0.10,
        )
        points = [
            [0.001, 0.001, 0.051],
            [0.061, 0.001, 0.051],
            [0.001, 0.051, 0.051],
            [0.001, 0.001, -0.021],
            [0.001, 0.001, 0.111],
        ]
        result = fuse_dynamic_environment_clouds(
            ["center"],
            [FakeCloud(points)],
            args,
            target_center_base=[0.0, 0.0, 0.05],
        )

        self.assertEqual(result.voxel_keys.shape[0], 1)
        np.testing.assert_allclose(
            result.roi_bounds_base,
            [[-0.05, -0.04, -0.01], [0.05, 0.04, 0.10]],
        )

    def test_auto_xy_roi_includes_place_and_clear_endpoints_with_margin(self):
        args = map_args(
            motion_environment_map_isolated_filter=False,
            motion_environment_map_radius_xy=0.10,
            place_xyz=[0.30, 0.0, 0.10],
            clear_place_xyz=[-0.25, -0.20, 0.10],
        )
        result = fuse_dynamic_environment_clouds(
            ["center"],
            [FakeCloud([[0.321, 0.001, 0.051], [-0.281, -0.221, 0.051]])],
            args,
            target_center_base=[0.0, 0.0, 0.05],
        )

        self.assertEqual(result.voxel_keys.shape[0], 2)
        self.assertAlmostEqual(float(result.roi_bounds_base[0, 0]), -0.30)
        self.assertAlmostEqual(float(result.roi_bounds_base[0, 1]), -0.25)
        self.assertAlmostEqual(float(result.roi_bounds_base[1, 0]), 0.35)
        self.assertAlmostEqual(float(result.roi_bounds_base[1, 1]), 0.10)

    def test_explicit_roi_cannot_crop_required_transport_endpoint(self):
        args = map_args(
            motion_environment_map_x_min=-0.10,
            motion_environment_map_x_max=0.10,
            motion_environment_map_y_min=-0.10,
            motion_environment_map_y_max=0.10,
            clear_place_xyz=[-0.30, 0.0, 0.10],
        )

        with self.assertRaisesRegex(
            ValueError,
            "excludes required clear_place_xyz",
        ):
            fuse_dynamic_environment_clouds(
                ["center"],
                [FakeCloud([[0.0, 0.0, 0.05]])],
                args,
                target_center_base=[0.0, 0.0, 0.05],
            )

    def test_mismatched_view_lists_raise_instead_of_zip_truncation(self):
        with self.assertRaisesRegex(ValueError, "equal length"):
            fuse_dynamic_environment_clouds(
                ["center", "side"],
                [FakeCloud([[0.0, 0.0, 0.05]])],
                map_args(),
            )


if __name__ == "__main__":
    unittest.main()
