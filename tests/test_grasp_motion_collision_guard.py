import json
import unittest
import sys
import tempfile
from types import SimpleNamespace
from unittest import mock

import numpy as np

import real_nero_single_grasp_runner as runner


def motion_args():
    return SimpleNamespace(
        motion_path_min_clearance=0.020,
        motion_runtime_stop_clearance=0.008,
        motion_target_non_tool_clearance=0.005,
        motion_target_gripper_body_clearance=0.002,
        motion_attached_object_clearance=0.012,
        motion_path_sample_joint_deg=2.0,
        motion_path_max_joint_delta_deg=60.0,
        motion_path_require_scene=True,
        motion_transit_vertical_clearance=0.080,
        motion_transit_max_tcp_z=0.50,
        motion_transit_height_step=0.020,
        motion_up_and_over=False,
        motion_direct_release_retreat=0.030,
        table_guard_enable=False,
    )


def safety_scene(environment, target=None):
    environment = np.asarray(environment, dtype=np.float32).reshape(-1, 3)
    target = (
        np.zeros((0, 3), dtype=np.float32)
        if target is None
        else np.asarray(target, dtype=np.float32).reshape(-1, 3)
    )
    return runner.MotionSafetyScene(
        full_scene_base=np.vstack([environment, target]),
        environment_base=environment,
        target_base=target,
        map_id="unit",
    )


class GraspMotionCollisionGuardTest(unittest.TestCase):
    def test_numerical_joint_limit_rounding_is_clipped_but_real_violation_fails(self):
        upper = np.full(7, 2.0, dtype=float)
        upper[6] = 1.570796251296997
        solver = mock.Mock()
        solver.get_joint_limits.return_value = (
            np.full(7, -2.0, dtype=float),
            upper,
        )

        configured_home = np.zeros(7, dtype=float)
        configured_home[6] = np.deg2rad(90.0)
        clipped = runner._clip_numerical_joint_limit_overshoot(
            configured_home, solver
        )
        self.assertIsNotNone(clipped)
        self.assertEqual(clipped[6], upper[6])
        self.assertLess(
            configured_home[6] - clipped[6],
            runner.JOINT_LIMIT_NUMERICAL_TOL_RAD,
        )

        real_violation = configured_home.copy()
        real_violation[6] = upper[6] + 2.0 * runner.JOINT_LIMIT_NUMERICAL_TOL_RAD
        self.assertIsNone(
            runner._clip_numerical_joint_limit_overshoot(
                real_violation, solver
            )
        )

    def test_stored_home_joint_rounding_is_clipped_before_move_command(self):
        upper = np.full(7, 2.0, dtype=float)
        upper[6] = 1.570796251296997
        solver = mock.Mock()
        solver.get_joint_limits.return_value = (
            np.full(7, -2.0, dtype=float),
            upper,
        )
        solver.solve_fk.return_value = np.array(
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            dtype=float,
        )
        arm = mock.Mock()
        arm.get_joint_angles.return_value = np.zeros(7, dtype=float)
        args = SimpleNamespace(min_flange_z=None)
        configured_home = np.zeros(7, dtype=float)
        configured_home[6] = np.deg2rad(90.0)

        with mock.patch.object(
            runner, "_get_ik_solver", return_value=solver
        ), mock.patch.object(
            runner, "_send_joint_target_with_start_retry"
        ) as send, mock.patch.object(
            runner,
            "wait_until_joints_reached",
            return_value=(True, upper.copy(), 0.0),
        ), mock.patch.object(
            runner,
            "wait_until_pose_reached",
            return_value=(True, np.zeros(6), 0.0, 0.0),
        ):
            runner.move_flange_pose_ik_joint(
                arm,
                np.zeros(6),
                args,
                "released_return",
                planned_joint_target=configured_home,
            )

        sent_target = np.asarray(send.call_args.args[1], dtype=float)
        self.assertEqual(sent_target[6], upper[6])

    def test_all_raised_avoidance_routes_are_disabled_by_default(self):
        with mock.patch.object(
            sys, "argv", ["runner", "--calib-json", "/tmp/fake_calib.json"]
        ):
            args = runner.parse_args()
        self.assertFalse(args.active_up_and_over)
        self.assertFalse(args.motion_up_and_over)
        self.assertFalse(args.use_intermediate_motion)
        self.assertEqual(args.active_direct_return_retreat_height, 0.0)
        self.assertEqual(args.active_direct_return_downward_tolerance, 0.002)
        self.assertEqual(
            args.active_direct_return_runtime_downward_tolerance, 0.010
        )
        self.assertEqual(args.active_direct_return_min_clearance, 0.030)
        self.assertEqual(args.active_return_execution_waypoint_deg, 10.0)
        self.assertEqual(args.released_return_execution_waypoint_deg, 20.0)
        self.assertEqual(args.camera_self_mask_expand_px, 12)
        self.assertEqual(args.empty_recovery_execution_waypoint_deg, 15.0)
        self.assertAlmostEqual(
            np.rad2deg(args.max_candidate_ik_joint_delta), 100.0
        )
        self.assertTrue(args.motion_target_approach_finger_environment_guard)
        self.assertFalse(args.direct_pickup_contact_override)

    def test_target_approach_finger_environment_override_is_narrow(self):
        args = motion_args()
        args.motion_target_approach_finger_environment_guard = False
        scene = safety_scene([[0.0, 0.0, 0.0]], [[1.0, 0.0, 0.0]])
        finger_only = [
            (np.zeros(3), np.zeros(3), 0.01, "gripper_finger_+1")
        ]
        body_only = [
            (np.zeros(3), np.zeros(3), 0.01, "gripper_body")
        ]

        with mock.patch.object(
            runner, "_motion_robot_capsules", return_value=finger_only
        ):
            approach_metrics = runner._motion_configuration_metrics(
                np.zeros(7),
                runner._motion_context(scene, "target_approach"),
                args,
            )
            runtime_metrics = runner._motion_configuration_metrics(
                np.zeros(7),
                runner._motion_context(scene, "target_approach"),
                args,
                runtime=True,
            )
            free_open_metrics = runner._motion_configuration_metrics(
                np.zeros(7),
                runner._motion_context(scene, "free_open"),
                args,
            )

        self.assertTrue(np.isinf(approach_metrics["margin"]))
        self.assertTrue(np.isinf(runtime_metrics["margin"]))
        self.assertLess(free_open_metrics["margin"], 0.0)
        self.assertEqual(free_open_metrics["primitive"], "gripper_finger_+1")

        with mock.patch.object(
            runner, "_motion_robot_capsules", return_value=body_only
        ):
            body_metrics = runner._motion_configuration_metrics(
                np.zeros(7),
                runner._motion_context(scene, "target_approach"),
                args,
            )
        self.assertLess(body_metrics["margin"], 0.0)
        self.assertEqual(body_metrics["primitive"], "gripper_body")

    def test_target_approach_finger_environment_override_cli(self):
        with mock.patch.object(
            sys,
            "argv",
            [
                "runner",
                "--calib-json",
                "/tmp/fake_calib.json",
                "--no-target-approach-finger-environment-guard",
            ],
        ):
            args = runner.parse_args()
        self.assertFalse(args.motion_target_approach_finger_environment_guard)

    def test_direct_pickup_contact_override_cli(self):
        with mock.patch.object(
            sys,
            "argv",
            [
                "runner",
                "--calib-json",
                "/tmp/fake_calib.json",
                "--direct-pickup-contact-override",
            ],
        ):
            args = runner.parse_args()
        self.assertTrue(args.direct_pickup_contact_override)

    def test_direct_pickup_override_covers_pickup_and_carrying_tool_contact(self):
        args = motion_args()
        args.direct_pickup_contact_override = True
        scene = safety_scene([[0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0]])
        pickup_context = runner._motion_context(
            scene,
            "pickup_lift",
            attached_target_tcp=np.zeros((1, 3), dtype=float),
        )
        carrying_context = runner._motion_context(
            scene,
            "carrying",
        )
        tool_only = [
            (np.zeros(3), np.zeros(3), 0.01, "gripper_tool")
        ]
        body_only = [
            (np.zeros(3), np.zeros(3), 0.01, "gripper_body")
        ]

        with mock.patch.object(
            runner, "_motion_robot_capsules", return_value=tool_only
        ), mock.patch.object(
            runner, "_tcp_transform_from_joints", return_value=np.eye(4)
        ):
            pickup_metrics = runner._motion_configuration_metrics(
                np.zeros(7), pickup_context, args
            )
            pickup_runtime_metrics = runner._motion_configuration_metrics(
                np.zeros(7), pickup_context, args, runtime=True
            )
            carrying_metrics = runner._motion_configuration_metrics(
                np.zeros(7), carrying_context, args
            )
            carrying_runtime_metrics = runner._motion_configuration_metrics(
                np.zeros(7), carrying_context, args, runtime=True
            )

        self.assertTrue(np.isinf(pickup_metrics["margin"]))
        self.assertTrue(np.isinf(pickup_runtime_metrics["margin"]))
        self.assertTrue(np.isinf(carrying_metrics["margin"]))
        self.assertTrue(np.isinf(carrying_runtime_metrics["margin"]))

        with mock.patch.object(
            runner, "_motion_robot_capsules", return_value=body_only
        ), mock.patch.object(
            runner, "_tcp_transform_from_joints", return_value=np.eye(4)
        ):
            body_metrics = runner._motion_configuration_metrics(
                np.zeros(7), pickup_context, args
            )
            carrying_body_metrics = runner._motion_configuration_metrics(
                np.zeros(7), carrying_context, args
            )
        self.assertLess(body_metrics["margin"], 0.0)
        self.assertEqual(body_metrics["primitive"], "gripper_body")
        self.assertLess(carrying_body_metrics["margin"], 0.0)
        self.assertEqual(carrying_body_metrics["primitive"], "gripper_body")

    def test_direct_pickup_override_keeps_carried_payload_environment_guard(self):
        args = motion_args()
        args.direct_pickup_contact_override = True
        scene = safety_scene([[0.0, 0.0, 0.0]], [[10.0, 0.0, 0.0]])
        context = runner._motion_context(
            scene,
            "carrying",
            attached_target_tcp=np.zeros((1, 3), dtype=float),
        )
        tool_only = [
            (np.ones(3), np.ones(3), 0.01, "gripper_tool")
        ]
        with mock.patch.object(
            runner, "_motion_robot_capsules", return_value=tool_only
        ), mock.patch.object(
            runner, "_tcp_transform_from_joints", return_value=np.eye(4)
        ):
            metrics = runner._motion_configuration_metrics(
                np.zeros(7), context, args
            )

        self.assertLess(metrics["margin"], 0.0)
        self.assertEqual(metrics["primitive"], "attached_target")
        self.assertEqual(metrics["source"], "environment")

    def test_direct_pickup_override_disables_short_support_release(self):
        args = SimpleNamespace(direct_pickup_contact_override=True)
        self.assertIsNone(
            runner._build_initial_support_release(
                safety_scene([[0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0]]),
                np.zeros(6),
                args,
            )
        )

    def test_pickup_payload_table_guard_ignores_sparse_low_depth_tail(self):
        args = motion_args()
        args.direct_pickup_contact_override = True
        args.table_guard_enable = True
        args.table_z = 0.08
        args.table_tcp_clearance = 0.0
        args.motion_payload_table_percentile = 10.0
        args.motion_pickup_table_support_tolerance = 0.005
        payload = np.zeros((100, 3), dtype=float)
        payload[:, 2] = 0.08
        payload[0, 2] = 0.03
        scene = safety_scene([[10.0, 10.0, 10.0]])
        context = runner._motion_context(
            scene,
            "pickup_lift",
            attached_target_tcp=payload,
        )
        far_capsule = [
            (
                np.array([20.0, 20.0, 20.0]),
                np.array([20.0, 20.0, 20.0]),
                0.01,
                "link1_2",
            )
        ]

        with mock.patch.object(
            runner, "_motion_robot_capsules", return_value=far_capsule
        ), mock.patch.object(
            runner, "_tcp_transform_from_joints", return_value=np.eye(4)
        ):
            metrics = runner._motion_configuration_metrics(
                np.zeros(7), context, args
            )

        self.assertGreaterEqual(metrics["margin"], 0.0)

    def test_payload_table_precheck_runs_before_any_ik_route(self):
        args = motion_args()
        args.table_guard_enable = True
        args.table_z = 0.08
        args.table_tcp_clearance = 0.0
        args.motion_payload_table_percentile = 10.0
        args.motion_pickup_table_support_tolerance = 0.005
        target = np.zeros((100, 3), dtype=np.float32)
        target[:, 2] = 0.03
        scene = safety_scene([[10.0, 10.0, 10.0]], target)

        with self.assertRaisesRegex(
            runner.UnsafeMotionTarget,
            "attached_target.*table_plane.*sample 0",
        ):
            runner.assert_pickup_payload_table_start(
                scene,
                args,
                "candidate_pickup_start",
            )

    def test_pickup_payload_table_guard_accepts_bounded_start_contact_only(self):
        args = motion_args()
        args.table_guard_enable = True
        args.table_z = 0.08
        args.table_tcp_clearance = 0.0
        args.motion_payload_table_percentile = 10.0
        args.motion_support_target_bottom_tolerance = 0.020
        args.motion_support_downward_tolerance = 0.001
        payload = np.zeros((100, 3), dtype=float)
        payload[:, 2] = 0.0712
        scene = safety_scene([[10.0, 10.0, 10.0]], payload)

        raw = runner.assert_pickup_payload_table_start(
            scene,
            args,
            "candidate_pickup_start",
        )
        self.assertAlmostEqual(raw, -0.0088, places=5)

        context = runner._motion_context(
            scene,
            "pickup_lift",
            attached_target_tcp=payload,
        )
        far_capsule = [
            (
                np.array([20.0, 20.0, 20.0]),
                np.array([20.0, 20.0, 20.0]),
                0.01,
                "link1_2",
            )
        ]
        with mock.patch.object(
            runner, "_motion_robot_capsules", return_value=far_capsule
        ), mock.patch.object(
            runner, "_tcp_transform_from_joints", return_value=np.eye(4)
        ):
            baseline = runner._motion_configuration_metrics(
                np.zeros(7), context, args
            )
        self.assertGreaterEqual(baseline["margin"], 0.0)

        moved_down = np.eye(4)
        moved_down[2, 3] = -0.003
        with mock.patch.object(
            runner, "_motion_robot_capsules", return_value=far_capsule
        ), mock.patch.object(
            runner, "_tcp_transform_from_joints", return_value=moved_down
        ):
            deeper = runner._motion_configuration_metrics(
                np.zeros(7), context, args
            )
        self.assertLess(deeper["margin"], 0.0)
        self.assertEqual(deeper["source"], "table_plane")

    def test_pickup_payload_table_guard_rejects_a_genuinely_low_body(self):
        args = motion_args()
        args.direct_pickup_contact_override = True
        args.table_guard_enable = True
        args.table_z = 0.08
        args.table_tcp_clearance = 0.0
        args.motion_payload_table_percentile = 10.0
        args.motion_pickup_table_support_tolerance = 0.005
        payload = np.zeros((100, 3), dtype=float)
        payload[:, 2] = 0.03
        scene = safety_scene([[10.0, 10.0, 10.0]])
        context = runner._motion_context(
            scene,
            "pickup_lift",
            attached_target_tcp=payload,
        )
        far_capsule = [
            (
                np.array([20.0, 20.0, 20.0]),
                np.array([20.0, 20.0, 20.0]),
                0.01,
                "link1_2",
            )
        ]

        with mock.patch.object(
            runner, "_motion_robot_capsules", return_value=far_capsule
        ), mock.patch.object(
            runner, "_tcp_transform_from_joints", return_value=np.eye(4)
        ):
            metrics = runner._motion_configuration_metrics(
                np.zeros(7), context, args
            )

        self.assertLess(metrics["margin"], 0.0)
        self.assertEqual(metrics["primitive"], "attached_target")
        self.assertEqual(metrics["source"], "table_plane")

    def test_first_empty_recovery_edge_uses_reverse_target_approach_policy(self):
        args = motion_args()
        args.direct_pickup_contact_override = True
        args.motion_mode = "ik_joint"
        args.skip_initial_move = True
        args.skip_empty_grasp_return = False
        scene = safety_scene([[0.0, 0.0, 0.0]], [[1.0, 0.0, 0.0]])
        route_stage = runner.PlannedJointStage(
            label="to_pregrasp_direct",
            tcp_pose=np.zeros(6),
            flange_pose=np.zeros(6),
            joints=np.zeros(7),
            phase="free_open",
        )
        grasp_stage = runner.PlannedJointStage(
            label="grasp",
            tcp_pose=np.zeros(6),
            flange_pose=np.zeros(6),
            joints=np.ones(7) * 0.01,
            phase="target_approach",
        )
        calls = []

        def record_path(_q0, _q1, context, _args, label):
            calls.append((label, context.phase))

        with mock.patch.object(
            runner, "validate_motion_joint_path", side_effect=record_path
        ), mock.patch.object(
            runner,
            "_plan_guarded_route",
            return_value=[route_stage],
        ), mock.patch.object(
            runner,
            "_plan_tcp_motion_stage",
            return_value=grasp_stage,
        ), mock.patch.object(
            runner,
            "prepare_motion_safety_scene",
            return_value=scene,
        ), mock.patch.object(
            runner,
            "_target_points_in_tcp",
            return_value=np.zeros((0, 3), dtype=np.float32),
        ), mock.patch.object(
            runner,
            "_build_initial_support_release",
            return_value=None,
        ):
            arm = mock.Mock()
            arm.get_joint_angles.return_value = np.zeros(7)
            arm.get_tcp_pose.return_value = np.zeros(6)
            runner.plan_up_and_over_grasp_sequence(
                arm,
                np.zeros(6),
                np.zeros(6),
                np.zeros(6),
                None,
                mock.Mock(),
                mock.Mock(),
                np.eye(4),
                args,
                "unit",
            )

        empty_calls = [
            phase for label, phase in calls if "empty_reverse" in label
        ]
        self.assertEqual(empty_calls, ["target_approach"])

    def test_only_gripper_may_contact_selected_target_during_approach(self):
        args = motion_args()
        scene = safety_scene([[10.0, 0.0, 0.0]], [[0.0, 0.0, 0.0]])
        context = runner._motion_context(scene, "target_approach")
        tool_only = [(np.zeros(3), np.zeros(3), 0.01, "gripper_tool")]
        with mock.patch.object(runner, "_motion_robot_capsules", return_value=tool_only):
            metrics = runner._motion_configuration_metrics(
                np.zeros(7), context, args
            )
        self.assertGreaterEqual(metrics["margin"], 0.0)

        finger_only = [
            (np.zeros(3), np.zeros(3), 0.01, "gripper_finger_+1")
        ]
        with mock.patch.object(
            runner, "_motion_robot_capsules", return_value=finger_only
        ):
            metrics = runner._motion_configuration_metrics(
                np.zeros(7), context, args
            )
        self.assertGreaterEqual(metrics["margin"], 0.0)

        environment_and_target = safety_scene(
            [[0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0]]
        )
        with mock.patch.object(
            runner, "_motion_robot_capsules", return_value=finger_only
        ):
            metrics = runner._motion_configuration_metrics(
                np.zeros(7),
                runner._motion_context(environment_and_target, "target_approach"),
                args,
            )
        self.assertLess(metrics["margin"], 0.0)
        self.assertEqual(metrics["primitive"], "gripper_finger_+1")
        self.assertEqual(metrics["source"], "environment")

        link_contact = [
            (np.zeros(3), np.zeros(3), 0.01, "gripper_tool"),
            (np.zeros(3), np.zeros(3), 0.01, "link6_7"),
        ]
        with mock.patch.object(
            runner, "_motion_robot_capsules", return_value=link_contact
        ):
            metrics = runner._motion_configuration_metrics(
                np.zeros(7), context, args
            )
        self.assertLess(metrics["margin"], 0.0)
        self.assertEqual(metrics["primitive"], "link6_7")
        self.assertEqual(metrics["source"], "selected_target")

    def test_non_target_environment_always_blocks_gripper(self):
        args = motion_args()
        scene = safety_scene([[0.0, 0.0, 0.0]], [[1.0, 0.0, 0.0]])
        context = runner._motion_context(scene, "target_approach")
        capsules = [(np.zeros(3), np.zeros(3), 0.01, "gripper_tool")]
        with mock.patch.object(runner, "_motion_robot_capsules", return_value=capsules):
            metrics = runner._motion_configuration_metrics(
                np.zeros(7), context, args
            )
        self.assertLess(metrics["margin"], 0.0)
        self.assertEqual(metrics["source"], "environment")

    def test_gripper_body_is_not_part_of_target_contact_exception(self):
        args = motion_args()
        scene = safety_scene([[10.0, 0.0, 0.0]], [[0.0, 0.0, 0.0]])
        context = runner._motion_context(scene, "target_approach")
        capsules = [
            (np.zeros(3), np.zeros(3), 0.01, "gripper_body"),
            (np.ones(3), np.ones(3), 0.01, "gripper_tool"),
        ]
        with mock.patch.object(
            runner, "_motion_robot_capsules", return_value=capsules
        ):
            metrics = runner._motion_configuration_metrics(
                np.zeros(7), context, args
            )
        self.assertLess(metrics["margin"], 0.0)
        self.assertEqual(metrics["primitive"], "gripper_body")
        self.assertEqual(metrics["source"], "selected_target")

    def test_gripper_body_uses_narrow_target_clearance_only_on_final_approach(self):
        args = motion_args()
        scene = safety_scene([[10.0, 0.0, 0.0]], [[0.013, 0.0, 0.0]])
        capsule = [(np.zeros(3), np.zeros(3), 0.01, "gripper_body")]
        with mock.patch.object(
            runner, "_motion_robot_capsules", return_value=capsule
        ):
            approach = runner._motion_configuration_metrics(
                np.zeros(7),
                runner._motion_context(scene, "target_approach"),
                args,
            )
            free = runner._motion_configuration_metrics(
                np.zeros(7),
                runner._motion_context(scene, "free_open"),
                args,
            )
        self.assertGreater(approach["margin"], 0.0)
        self.assertEqual(approach["required_clearance"], 0.002)
        self.assertLess(free["margin"], 0.0)

    def test_open_finger_contact_exception_does_not_apply_to_placed_target(self):
        args = motion_args()
        scene = safety_scene([[10.0, 0.0, 0.0]], [[10.0, 0.0, 0.0]])
        context = runner._motion_context(
            scene,
            "target_approach",
            placed_target_base=np.zeros((1, 3), dtype=float),
        )
        finger_only = [
            (np.zeros(3), np.zeros(3), 0.01, "gripper_finger_+1")
        ]
        with mock.patch.object(
            runner, "_motion_robot_capsules", return_value=finger_only
        ):
            metrics = runner._motion_configuration_metrics(
                np.zeros(7), context, args
            )
        self.assertLess(metrics["margin"], 0.0)
        self.assertEqual(metrics["primitive"], "gripper_finger_+1")
        self.assertEqual(metrics["source"], "placed_target")

    def test_released_retreat_allows_open_finger_next_to_placed_target(self):
        args = motion_args()
        scene = safety_scene([[10.0, 0.0, 0.0]])
        context = runner._motion_context(
            scene,
            "released_retreat",
            placed_target_base=np.zeros((1, 3), dtype=float),
        )
        finger_only = [
            (np.zeros(3), np.zeros(3), 0.01, "gripper_finger_+1")
        ]
        with mock.patch.object(
            runner, "_motion_robot_capsules", return_value=finger_only
        ):
            metrics = runner._motion_configuration_metrics(
                np.zeros(7), context, args
            )

        self.assertGreaterEqual(metrics["margin"], 0.0)
        self.assertNotEqual(metrics["source"], "placed_target")

    def test_released_free_allows_open_finger_to_leave_placed_target(self):
        args = motion_args()
        scene = safety_scene([[10.0, 0.0, 0.0]])
        context = runner._motion_context(
            scene,
            "released_free",
            placed_target_base=np.zeros((1, 3), dtype=float),
        )
        finger_only = [
            (np.zeros(3), np.zeros(3), 0.01, "gripper_finger_-1")
        ]
        with mock.patch.object(
            runner, "_motion_robot_capsules", return_value=finger_only
        ):
            metrics = runner._motion_configuration_metrics(
                np.zeros(7), context, args
            )

        self.assertGreaterEqual(metrics["margin"], 0.0)
        self.assertNotEqual(metrics["source"], "placed_target")

    def test_gripper_body_endcap_stops_before_distal_contact_zone(self):
        args = SimpleNamespace(
            tcp_offset=[0.165, 0.0, 0.0, 0.0, 0.0, 0.0],
            motion_path_link_radius=0.060,
            motion_path_tool_radius=0.055,
            motion_path_camera_radius=0.055,
            motion_target_tool_contact_length=0.045,
            motion_path_require_camera_model=False,
        )
        T_flange = np.eye(4)
        with mock.patch.object(
            runner,
            "_urdf_chain_geometry",
            return_value=(np.array([[0.0, 0.0, 0.0]]), T_flange),
        ):
            capsules = runner._motion_robot_capsules(np.zeros(7), args)
        body = next(item for item in capsules if item[3] == "gripper_body")
        tcp = np.array([[0.165, 0.0, 0.0]])
        surface_clearance = float(
            runner._point_segment_distances(tcp, body[0], body[1])[0]
            - body[2]
        )
        self.assertAlmostEqual(surface_clearance, 0.045, places=6)

    def test_camera_collision_model_keeps_optical_front_clear(self):
        args = SimpleNamespace(
            calib_json="/tmp/fake_calib.json",
            handeye_convention="flange_camera_xyzw",
            camera_collision_body_rear_extent=0.120,
            camera_collision_mount_radius=0.025,
            camera_collision_mount_attachment_depth=0.060,
        )
        T_flange_camera = np.eye(4)
        T_flange_camera[:3, 3] = [0.0, 0.0, 0.200]
        with mock.patch.object(
            runner, "handeye_transform", return_value=T_flange_camera
        ):
            capsules = runner._camera_collision_capsules(
                np.eye(4), args, 0.055
            )

        optical_front = np.array([[0.0, 0.0, 0.230]])
        front_clearances = [
            float(
                runner._point_segment_distances(optical_front, a, b)[0]
                - radius
            )
            for a, b, radius, _name in capsules
        ]
        self.assertTrue(all(value > 0.0 for value in front_clearances))

        def colliding_names(point):
            point = np.asarray(point, dtype=float).reshape(1, 3)
            return {
                name
                for a, b, radius, name in capsules
                if float(runner._point_segment_distances(point, a, b)[0])
                <= float(radius)
            }

        # The body occupies the rear and lateral volume, while the narrow
        # bracket remains represented between the flange and camera body.
        self.assertIn("camera_body", colliding_names([0.0, 0.0, 0.145]))
        self.assertIn("camera_body", colliding_names([0.050, 0.0, 0.145]))
        self.assertEqual(colliding_names([0.0, 0.0, 0.070]), {"camera_mount"})

    def test_active_and_grasp_motion_share_camera_collision_geometry(self):
        args = SimpleNamespace(
            tcp_offset=[0.165, 0.0, 0.0, 0.0, 0.0, 0.0],
            active_path_link_radius=0.010,
            active_path_tool_radius=0.010,
            active_path_camera_radius=0.055,
            motion_path_link_radius=0.010,
            motion_path_tool_radius=0.010,
            motion_path_camera_radius=0.055,
            motion_target_tool_contact_length=0.045,
            motion_path_require_camera_model=True,
            active_path_require_camera_model=True,
            calib_json="/tmp/fake_calib.json",
            handeye_convention="flange_camera_xyzw",
            camera_collision_body_rear_extent=0.120,
            camera_collision_mount_radius=0.025,
            camera_collision_mount_attachment_depth=0.060,
        )
        T_flange_camera = np.eye(4)
        T_flange_camera[:3, 3] = [0.0, 0.0, 0.200]
        with mock.patch.object(
            runner,
            "_urdf_chain_geometry",
            return_value=(np.array([[-1.0, 0.0, 0.0]]), np.eye(4)),
        ), mock.patch.object(
            runner, "handeye_transform", return_value=T_flange_camera
        ):
            active = [
                item
                for item in runner._active_robot_capsules(np.zeros(7), args)
                if item[3].startswith("camera_")
            ]
            motion = [
                item
                for item in runner._motion_robot_capsules(np.zeros(7), args)
                if item[3].startswith("camera_")
            ]

        self.assertEqual([item[3] for item in active], ["camera_body", "camera_mount"])
        self.assertEqual([item[3] for item in motion], ["camera_body", "camera_mount"])
        for active_item, motion_item in zip(active, motion):
            np.testing.assert_allclose(active_item[0], motion_item[0])
            np.testing.assert_allclose(active_item[1], motion_item[1])
            self.assertAlmostEqual(active_item[2], motion_item[2])

    def test_free_open_uses_two_fingers_and_keeps_jaw_center_clear(self):
        args = motion_args()
        args.tcp_offset = [0.165, 0.0, 0.0, 0.0, 0.0, 0.0]
        args.motion_path_link_radius = 0.010
        args.motion_path_tool_radius = 0.055
        args.motion_path_camera_radius = 0.010
        args.motion_target_tool_contact_length = 0.045
        args.motion_path_require_camera_model = False
        args.tool_approach_axis = "x"
        args.tcp_gripper_width_axis = "z"
        args.gripper_open_width = 0.085
        args.gripper_hardware_max_width = 0.085
        args.motion_path_open_finger_radius = 0.007
        args.motion_path_open_finger_back = 0.065
        args.motion_path_open_finger_front = 0.020
        scene = safety_scene([[0.165, 0.0, 0.0]])
        T_flange = np.eye(4)
        with mock.patch.object(
            runner,
            "_urdf_chain_geometry",
            return_value=(np.array([[0.0, 0.0, 0.0]]), T_flange),
        ):
            open_metrics = runner._motion_configuration_metrics(
                np.zeros(7), runner._motion_context(scene, "free_open"), args
            )
            carrying_metrics = runner._motion_configuration_metrics(
                np.zeros(7), runner._motion_context(scene, "carrying"), args
            )
            open_capsules = runner._motion_robot_capsules(
                np.zeros(7), args, open_gripper=True
            )

        names = {item[3] for item in open_capsules}
        self.assertIn("gripper_finger_-1", names)
        self.assertIn("gripper_finger_+1", names)
        self.assertNotIn("gripper_tool", names)
        self.assertGreaterEqual(open_metrics["margin"], 0.0)
        # Empty-close finger centerlines overlap into one compact 14 mm
        # envelope.  The default 10 mm closed-finger contact tolerance still
        # permits this small contact during carrying.
        self.assertGreaterEqual(carrying_metrics["margin"], 0.0)
        with mock.patch.object(
            runner,
            "_urdf_chain_geometry",
            return_value=(np.array([[0.0, 0.0, 0.0]]), T_flange),
        ):
            closed_capsules = runner._motion_robot_capsules(
                np.zeros(7), args, open_gripper=False
            )
        closed_names = {item[3] for item in closed_capsules}
        self.assertIn("gripper_closed_finger_-1", closed_names)
        self.assertIn("gripper_closed_finger_+1", closed_names)
        self.assertNotIn("gripper_tool", closed_names)
        closed_fingers = [
            item
            for item in closed_capsules
            if item[3].startswith("gripper_closed_finger_")
        ]
        np.testing.assert_allclose(
            closed_fingers[0][0], closed_fingers[1][0]
        )
        self.assertAlmostEqual(closed_fingers[0][2], 0.007, places=6)

    def test_target_approach_uses_two_fingers_and_keeps_jaw_center_clear(self):
        args = motion_args()
        args.tcp_offset = [0.165, 0.0, 0.0, 0.0, 0.0, 0.0]
        args.motion_path_link_radius = 0.010
        args.motion_path_tool_radius = 0.055
        args.motion_path_camera_radius = 0.010
        args.motion_target_tool_contact_length = 0.045
        args.motion_path_require_camera_model = False
        args.tool_approach_axis = "x"
        args.tcp_gripper_width_axis = "z"
        args.gripper_open_width = 0.085
        args.gripper_hardware_max_width = 0.085
        args.motion_path_open_finger_radius = 0.014
        args.motion_path_open_finger_back = 0.065
        args.motion_path_open_finger_front = 0.020
        T_flange = np.eye(4)
        jaw_center = np.array([0.165, 0.0, 0.0])
        half_finger_spacing = 0.5 * 0.085 + 0.014
        finger_center = jaw_center + np.array(
            [0.0, 0.0, half_finger_spacing]
        )

        with mock.patch.object(
            runner,
            "_urdf_chain_geometry",
            return_value=(np.array([[0.0, 0.0, 0.0]]), T_flange),
        ):
            center_metrics = runner._motion_configuration_metrics(
                np.zeros(7),
                runner._motion_context(safety_scene([jaw_center]), "target_approach"),
                args,
            )
            finger_metrics = runner._motion_configuration_metrics(
                np.zeros(7),
                runner._motion_context(safety_scene([finger_center]), "target_approach"),
                args,
            )
            approach_capsules = runner._motion_robot_capsules(
                np.zeros(7), args, open_gripper=True
            )

        names = {item[3] for item in approach_capsules}
        self.assertIn("gripper_finger_-1", names)
        self.assertIn("gripper_finger_+1", names)
        self.assertNotIn("gripper_tool", names)
        fingers = [
            item
            for item in approach_capsules
            if item[3].startswith("gripper_finger_")
        ]
        centerline_spacing = float(
            np.linalg.norm(
                np.asarray(fingers[0][0]) - np.asarray(fingers[1][0])
            )
        )
        inner_face_gap = centerline_spacing - 2.0 * 0.014
        self.assertAlmostEqual(inner_face_gap, 0.085, places=6)
        self.assertGreaterEqual(center_metrics["margin"], 0.0)
        self.assertLess(finger_metrics["margin"], 0.0)
        self.assertTrue(
            finger_metrics["primitive"].startswith("gripper_finger_")
        )
        self.assertEqual(finger_metrics["source"], "environment")

    def test_post_close_collision_gap_is_not_capped_by_hardware_opening(self):
        args = motion_args()
        args.tcp_offset = [0.165, 0.0, 0.0, 0.0, 0.0, 0.0]
        args.motion_path_link_radius = 0.010
        args.motion_path_tool_radius = 0.055
        args.motion_path_camera_radius = 0.010
        args.motion_path_require_camera_model = False
        args.tool_approach_axis = "x"
        args.tcp_gripper_width_axis = "z"
        args.gripper_hardware_max_width = 0.085
        args.motion_path_open_finger_radius = 0.014
        args.motion_path_open_finger_back = 0.065
        args.motion_path_open_finger_front = 0.020
        with mock.patch.object(
            runner,
            "_urdf_chain_geometry",
            return_value=(np.array([[0.0, 0.0, 0.0]]), np.eye(4)),
        ):
            capsules = runner._motion_robot_capsules(
                np.zeros(7),
                args,
                open_gripper=False,
                closed_gripper_width=0.095,
            )
        fingers = [
            item
            for item in capsules
            if item[3].startswith("gripper_closed_finger_")
        ]
        self.assertEqual(len(fingers), 2)
        center_spacing = float(
            np.linalg.norm(
                np.asarray(fingers[0][0]) - np.asarray(fingers[1][0])
            )
        )
        inner_face_gap = center_spacing - 2.0 * 0.014
        self.assertAlmostEqual(inner_face_gap, 0.095, places=6)

    def test_post_close_fingers_allow_ten_mm_environment_contact_only(self):
        args = motion_args()
        args.motion_path_min_clearance = 0.005
        args.motion_closed_finger_contact_tolerance = 0.010
        context = runner._motion_context(
            safety_scene([[0.0, 0.0, 0.0]]),
            "carrying",
            closed_gripper_width=0.060,
        )
        closed_finger = [
            (np.zeros(3), np.zeros(3), 0.010, "gripper_closed_finger_+1")
        ]
        gripper_body = [
            (np.zeros(3), np.zeros(3), 0.010, "gripper_body")
        ]
        with mock.patch.object(
            runner, "_motion_robot_capsules", return_value=closed_finger
        ):
            finger_metrics = runner._motion_configuration_metrics(
                np.zeros(7), context, args
            )
        with mock.patch.object(
            runner, "_motion_robot_capsules", return_value=gripper_body
        ):
            body_metrics = runner._motion_configuration_metrics(
                np.zeros(7), context, args
            )
        self.assertAlmostEqual(
            finger_metrics["required_clearance"], -0.010, places=6
        )
        self.assertGreaterEqual(finger_metrics["margin"], -1e-7)
        self.assertAlmostEqual(
            body_metrics["required_clearance"], 0.005, places=6
        )
        self.assertLess(body_metrics["margin"], 0.0)

    def test_runtime_open_finger_gets_one_mm_tracking_slack_only(self):
        args = motion_args()
        args.motion_path_min_clearance = 0.005
        args.motion_runtime_stop_clearance = 0.005
        args.motion_runtime_open_finger_clearance_slack = 0.001
        context = runner._motion_context(
            safety_scene([[0.0, 0.0, 0.0]]), "free_open"
        )
        # Radius 10 mm with the point 14.9 mm from the centerline gives a
        # 4.9 mm surface clearance: planning rejects it against 5.0 mm, while
        # runtime accepts it against the finger-only 4.0 mm threshold.
        finger = [
            (
                np.array([0.0149, 0.0, 0.0]),
                np.array([0.0149, 0.0, 0.0]),
                0.010,
                "gripper_finger_-1",
            )
        ]
        body = [
            (
                np.array([0.0149, 0.0, 0.0]),
                np.array([0.0149, 0.0, 0.0]),
                0.010,
                "gripper_body",
            )
        ]
        with mock.patch.object(
            runner, "_motion_robot_capsules", return_value=finger
        ):
            planned = runner._motion_configuration_metrics(
                np.zeros(7), context, args, runtime=False
            )
            measured = runner._motion_configuration_metrics(
                np.zeros(7), context, args, runtime=True
            )
        with mock.patch.object(
            runner, "_motion_robot_capsules", return_value=body
        ):
            body_measured = runner._motion_configuration_metrics(
                np.zeros(7), context, args, runtime=True
            )
        self.assertLess(planned["margin"], 0.0)
        self.assertGreaterEqual(measured["margin"], 0.0)
        self.assertAlmostEqual(measured["required_clearance"], 0.004)
        self.assertLess(body_measured["margin"], 0.0)
        self.assertAlmostEqual(body_measured["required_clearance"], 0.005)

    def test_target_approach_tracking_slack_only_affects_runtime_open_fingers(self):
        args = motion_args()
        args.motion_path_min_clearance = 0.005
        args.motion_runtime_stop_clearance = 0.005
        args.motion_runtime_open_finger_clearance_slack = 0.001
        args.motion_runtime_target_approach_finger_clearance_slack = 0.002
        cases = [
            ("target_approach", True, "gripper_finger_+1", 0.003),
            ("target_approach", False, "gripper_finger_+1", 0.005),
            ("target_approach", True, "gripper_body", 0.005),
            ("target_approach", True, "camera_body", 0.005),
            ("target_approach", True, "urdf_link1", 0.005),
            ("carrying", True, "gripper_finger_+1", 0.005),
            ("free_open", True, "gripper_finger_+1", 0.004),
        ]
        for phase, runtime, primitive, required in cases:
            with self.subTest(phase=phase, runtime=runtime, primitive=primitive):
                capsule = [(np.array([.014, 0, 0]), np.array([.014, 0, 0]), .010, primitive)]
                context = runner._motion_context(safety_scene([[0, 0, 0]]), phase)
                with mock.patch.object(runner, "_motion_robot_capsules", return_value=capsule):
                    metrics = runner._motion_configuration_metrics(np.zeros(7), context, args, runtime=runtime)
                self.assertAlmostEqual(metrics["nearest_clearance"], .004, places=6)
                self.assertAlmostEqual(metrics["required_clearance"], required)
                self.assertAlmostEqual(metrics["margin"], .004 - required, places=6)

        context = runner._motion_context(safety_scene([[0, 0, 0]]), "target_approach")
        capsule = [(np.array([.0129, 0, 0]), np.array([.0129, 0, 0]), .010, "gripper_finger_-1")]
        with mock.patch.object(runner, "_motion_robot_capsules", return_value=capsule):
            metrics = runner._motion_configuration_metrics(np.zeros(7), context, args, runtime=True)
        self.assertAlmostEqual(metrics["required_clearance"], .003)
        self.assertLess(metrics["margin"], 0.)

    def test_contact_check_clearance_only_changes_approaching_open_fingers(self):
        args = motion_args()
        args.motion_path_min_clearance = .005
        args.motion_runtime_stop_clearance = .005
        args.motion_runtime_open_finger_clearance_slack = .001
        args.grasp_contact_check_only = True
        args.grasp_contact_check_clearance = .002
        cases = [
            ('target_approach', 'gripper_finger_+1', .002),
            ('target_approach', 'gripper_body', .005),
            ('target_approach', 'camera_body', .005),
            ('target_approach', 'urdf_link1', .005),
            ('carrying', 'gripper_finger_+1', .005),
        ]
        for runtime in [False, True]:
            for phase, primitive, required in cases:
                with self.subTest(runtime=runtime, phase=phase, primitive=primitive):
                    capsule = [(np.array([.013, 0, 0]), np.array([.013, 0, 0]), .010, primitive)]
                    context = runner._motion_context(safety_scene([[0, 0, 0]]), phase)
                    with mock.patch.object(runner, '_motion_robot_capsules', return_value=capsule):
                        metrics = runner._motion_configuration_metrics(np.zeros(7), context, args, runtime=runtime)
                    self.assertAlmostEqual(metrics['required_clearance'], required)
                    self.assertEqual(metrics['margin'] >= 0, primitive.startswith('gripper_finger') and phase == 'target_approach')
        context = runner._motion_context(safety_scene([[0, 0, 0]]), 'target_approach')
        penetrating = [(np.array([.009, 0, 0]), np.array([.009, 0, 0]), .010, 'gripper_finger_+1')]
        with mock.patch.object(runner, '_motion_robot_capsules', return_value=penetrating):
            for trial in [.002, 0.0]:
                args.grasp_contact_check_clearance = trial
                for runtime in [False, True]:
                    self.assertLess(runner._motion_configuration_metrics(np.zeros(7), context, args, runtime=runtime)['margin'], 0)
        args.grasp_contact_check_clearance = 0.0
        for runtime in [False, True]:
            for primitive in ['gripper_finger_+1', 'gripper_body', 'camera_body', 'urdf_link1']:
                capsule = [(np.array([.011, 0, 0]), np.array([.011, 0, 0]), .010, primitive)]
                with mock.patch.object(runner, '_motion_robot_capsules', return_value=capsule):
                    metrics = runner._motion_configuration_metrics(np.zeros(7), context, args, runtime=runtime)
                self.assertEqual(metrics['required_clearance'], 0.0 if primitive == 'gripper_finger_+1' else .005)
                self.assertEqual(metrics['margin'] >= 0, primitive == 'gripper_finger_+1')

    def test_negative_contact_trial_uses_cad_table_limit_and_keeps_other_obstacles_strict(self):
        args = motion_args()
        args.motion_path_min_clearance = .005
        args.motion_runtime_stop_clearance = .005
        args.grasp_contact_check_only = True
        args.grasp_contact_check_clearance = -.005
        args.grasp_table_reference_z = 0.0
        args.table_z = 0.0
        cases = [
            # The signed CAD/plane distance decides the table check, even
            # where a padded capsule would report a larger overlap.
            ([0, 0, 0], [.009, 0, 0], 'gripper_finger_+1', -.0035, -.005, True),
            ([0, 0, 0], [.004, 0, 0], 'gripper_finger_+1', -.0035, -.005, True),
            ([0, 0, 0], [.020, 0, 0], 'gripper_finger_+1', -.006, -.005, False),
            ([0, 0, .020], [.009, 0, .020], 'gripper_finger_+1', -.0035, 0.0, False),
            ([0, 0, 0], [.009, 0, 0], 'gripper_body', -.0035, .005, False),
            ([0, 0, 0], [.009, 0, 0], 'camera_body', -.0035, .005, False),
            ([0, 0, 0], [.009, 0, 0], 'urdf_link1', -.0035, .005, False),
        ]
        for runtime in [False, True]:
            for point, center, primitive, cad_height, required, accepted in cases:
                with self.subTest(runtime=runtime, primitive=primitive, point=point, center=center):
                    context = runner._motion_context(safety_scene([point]), 'target_approach')
                    capsule = [(np.array(center), np.array(center), .010, primitive)]
                    transform = np.eye(4)
                    transform[2, 3] = cad_height
                    with mock.patch.object(runner, '_motion_robot_capsules', return_value=capsule), \
                         mock.patch.object(runner, '_tcp_transform_from_joints', return_value=transform), \
                         mock.patch('real_nero_grasp.graspgenx_integration.open_gripper_vertices_tcp', return_value=np.zeros((1, 3))):
                        metrics = runner._motion_configuration_metrics(np.zeros(7), context, args, runtime=runtime)
                    self.assertEqual(metrics['required_clearance'], required)
                    self.assertEqual(metrics['margin'] >= 0, accepted)

    def test_negative_trial_checks_cad_table_distance_inside_the_joint_path(self):
        args = motion_args()
        args.grasp_contact_check_only = True
        args.grasp_contact_check_clearance = -.005
        args.grasp_table_reference_z = 0.0
        args.table_z = 0.0
        q0, q1 = np.zeros(7), np.zeros(7)
        q1[0] = np.deg2rad(4)
        context = runner._motion_context(safety_scene([]), 'target_approach')
        def transform(q, _args):
            value = np.eye(4)
            value[2, 3] = -.006 if .25 < float(q[0] / q1[0]) < .75 else -.0035
            return value
        with mock.patch.object(runner, '_motion_robot_capsules', return_value=[]), \
             mock.patch.object(runner, '_tcp_transform_from_joints', side_effect=transform), \
             mock.patch('real_nero_grasp.graspgenx_integration.open_gripper_vertices_tcp', return_value=np.zeros((1, 3))):
            metrics = runner._motion_joint_path_metrics(q0, q1, context, args)
        self.assertLess(metrics['margin'], 0)
        self.assertEqual(metrics['primitive'], 'gripper_cad_mesh')
        self.assertEqual(metrics['source'], 'table_plane')
        self.assertAlmostEqual(metrics['nearest_clearance'], -.006)

    def test_normal_confirmed_table_uses_full_cad_and_preserves_other_obstacle_guards(self):
        args = motion_args()
        args.motion_path_min_clearance = .005
        args.motion_runtime_stop_clearance = .005
        args.motion_runtime_target_approach_finger_clearance_slack = .002
        args.grasp_table_reference_z = 0.0
        args.table_z = 0.0
        args.grasp_contact_check_clearance = None
        for runtime in [False, True]:
            cases = [
                # Padded finger capsule overlaps a noisy table point, while
                # the actual CAD still has 6 mm of modeled clearance.
                ([0, 0, .002], [.009, 0, .002], 'gripper_finger_-1', .006, True),
                ([0, 0, 0], [.009, 0, 0], 'gripper_finger_-1', .004, runtime),
                ([0, 0, .02], [.009, 0, .02], 'gripper_finger_-1', .006, False),
                ([0, 0, 0], [.009, 0, 0], 'camera_body', .006, False),
                ([0, 0, 0], [.009, 0, 0], 'urdf_link1', .006, False),
                ([0, 0, 0], [.009, 0, 0], 'gripper_body', .006, False),
            ]
            for point, center, primitive, height, accepted in cases:
                with self.subTest(runtime=runtime, primitive=primitive, height=height):
                    context = runner._motion_context(safety_scene([point]), 'target_approach')
                    transform = np.eye(4)
                    transform[2, 3] = height
                    capsules = [(np.array(center), np.array(center), .010, primitive)]
                    with mock.patch.object(runner, '_motion_robot_capsules', return_value=capsules), \
                         mock.patch.object(runner, '_tcp_transform_from_joints', return_value=transform), \
                         mock.patch('real_nero_grasp.graspgenx_integration.open_gripper_vertices_tcp', return_value=np.zeros((1, 3))):
                        metrics = runner._motion_configuration_metrics(np.zeros(7), context, args, runtime=runtime)
                    self.assertEqual(metrics['margin'] >= 0, accepted)
                    if accepted:
                        self.assertEqual(metrics['source'], 'table_plane')
                        self.assertAlmostEqual(metrics['required_clearance'], .003 if runtime else .005)

    def test_live_approach_start_allows_tracking_slack_without_relaxing_future_path(self):
        args = motion_args()
        args.motion_path_min_clearance = 0.005
        args.motion_runtime_stop_clearance = 0.005
        args.motion_runtime_target_approach_finger_clearance_slack = 0.002
        context = runner._motion_context(safety_scene([[0, 0, 0]]), "target_approach")
        target = np.zeros(7)
        target[0] = np.deg2rad(4)
        cases = [
            ("gripper_finger_+1", .0045, False, True),
            ("gripper_finger_+1", .0029, False, False),
            ("gripper_body", .0045, False, False),
            ("camera_body", .0045, False, False),
            ("urdf_link1", .0045, False, False),
            ("gripper_finger_+1", .0045, True, False),
        ]
        for primitive, start_clearance, interior_shortfall, accepted in cases:
            with self.subTest(primitive=primitive, start=start_clearance,
                              interior_shortfall=interior_shortfall):
                def capsules(q, _args, **_kwargs):
                    alpha = float(q[0] / target[0])
                    clearance = start_clearance + alpha * .020
                    if interior_shortfall and .1 < alpha < .9:
                        clearance = .0035
                    point = np.array([.010 + clearance, 0, 0])
                    return [(point, point, .010, primitive)]

                with mock.patch.object(runner, "_motion_robot_capsules", side_effect=capsules):
                    # Original planning still rejects a sub-5 mm start.
                    with self.assertRaises(runner.UnsafeMotionTarget):
                        runner.validate_motion_joint_path(
                            np.zeros(7), target, context, args, "nominal_retreat")
                    if accepted:
                        runner.validate_motion_joint_path(
                            np.zeros(7), target, context, args, "live_retreat",
                            allow_runtime_start=True)
                    else:
                        with self.assertRaises(runner.UnsafeMotionTarget):
                            runner.validate_motion_joint_path(
                                np.zeros(7), target, context, args, "live_retreat",
                                allow_runtime_start=True)

    def test_two_finger_self_filter_keeps_free_jaw_center(self):
        args = SimpleNamespace(
            motion_self_filter=True,
            motion_self_filter_margin=0.0,
        )
        capsules = [
            (
                np.array([0.0, -0.05, 0.0]),
                np.array([0.1, -0.05, 0.0]),
                0.014,
                "self_finger_-1",
            ),
            (
                np.array([0.0, 0.05, 0.0]),
                np.array([0.1, 0.05, 0.0]),
                0.014,
                "self_finger_+1",
            ),
        ]
        points = np.array(
            [[0.05, -0.05, 0.0], [0.05, 0.05, 0.0], [0.05, 0.0, 0.0]]
        )
        with mock.patch.object(
            runner, "_motion_self_filter_capsules", return_value=capsules
        ):
            filtered = runner.filter_motion_robot_self_points(
                points, np.zeros(7), args, "self_test"
            )
        np.testing.assert_allclose(filtered, np.array([[0.05, 0.0, 0.0]]))

    def test_self_filter_finger_model_preserves_full_commanded_gap(self):
        args = SimpleNamespace(
            tcp_offset=[0.165, 0.0, 0.0, 0.0, 0.0, 0.0],
            motion_self_filter_link_radius=0.010,
            motion_self_filter_palm_radius=0.035,
            motion_self_filter_camera_radius=0.040,
            motion_self_filter_finger_radius=0.014,
            motion_self_filter_finger_back=0.065,
            motion_self_filter_finger_front=0.020,
            tool_approach_axis="x",
            tcp_gripper_width_axis="z",
            gripper_open_width=0.085,
            gripper_hardware_max_width=0.085,
            calib_json="/tmp/missing.json",
            handeye_convention="flange_camera_xyzw",
        )
        with mock.patch.object(
            runner,
            "_urdf_chain_geometry",
            return_value=(np.array([[0.0, 0.0, 0.0]]), np.eye(4)),
        ), mock.patch.object(
            runner,
            "handeye_transform",
            side_effect=FileNotFoundError,
        ):
            capsules = runner._motion_self_filter_capsules(
                np.zeros(7), args
            )
        fingers = [
            item for item in capsules if item[3].startswith("self_finger_")
        ]
        self.assertEqual(len(fingers), 2)
        centerline_spacing = float(
            np.linalg.norm(
                np.asarray(fingers[0][0]) - np.asarray(fingers[1][0])
            )
        )
        inner_face_gap = centerline_spacing - 2.0 * 0.014
        self.assertAlmostEqual(inner_face_gap, 0.085, places=6)

    def test_attached_target_collision_is_detected(self):
        args = motion_args()
        scene = safety_scene([[0.0, 0.0, 0.0]])
        context = runner._motion_context(
            scene,
            "carrying",
            attached_target_tcp=np.array([[0.0, 0.0, 0.0]], dtype=float),
        )
        far_capsule = [
            (
                np.array([10.0, 0.0, 0.0]),
                np.array([10.0, 0.0, 0.0]),
                0.01,
                "link1_2",
            )
        ]
        with mock.patch.object(
            runner, "_motion_robot_capsules", return_value=far_capsule
        ), mock.patch.object(
            runner, "_tcp_transform_from_joints", return_value=np.eye(4)
        ):
            metrics = runner._motion_configuration_metrics(
                np.zeros(7), context, args
            )
        self.assertLess(metrics["margin"], 0.0)
        self.assertEqual(metrics["primitive"], "attached_target")

    def test_attached_target_cannot_intersect_arm_or_camera(self):
        args = motion_args()
        args.motion_payload_robot_clearance = 0.005
        scene = safety_scene([[10.0, 10.0, 10.0]])
        context = runner._motion_context(
            scene,
            "carrying",
            attached_target_tcp=np.array([[0.0, 0.0, 0.0]], dtype=float),
        )
        capsules = [
            (np.zeros(3), np.zeros(3), 0.01, "camera_mount"),
            (np.ones(3), np.ones(3), 0.01, "gripper_tool"),
        ]
        with mock.patch.object(
            runner, "_motion_robot_capsules", return_value=capsules
        ), mock.patch.object(
            runner, "_tcp_transform_from_joints", return_value=np.eye(4)
        ):
            metrics = runner._motion_configuration_metrics(
                np.zeros(7), context, args
            )
        self.assertLess(metrics["margin"], 0.0)
        self.assertEqual(metrics["primitive"], "camera_mount")
        self.assertEqual(metrics["source"], "attached_target_vs_robot")

    def test_support_release_exempts_only_verified_plane_points(self):
        args = motion_args()
        support = np.array([[0.0, 0.0, 0.0]], dtype=float)
        strict = np.array([[1.0, 0.0, 0.0]], dtype=float)
        scene = safety_scene(np.vstack([support, strict]))
        release = runner.InitialSupportRelease(
            strict_environment_base=strict,
            support_points_base=support,
            start_tcp_pose=np.zeros(6),
        )
        context = runner._motion_context(
            scene,
            "support_release",
            attached_target_tcp=np.array([[0.0, 0.0, 0.0]]),
            initial_support_release=release,
        )
        far_capsule = [
            (
                np.array([10.0, 0.0, 0.0]),
                np.array([10.0, 0.0, 0.0]),
                0.01,
                "link1_2",
            )
        ]
        with mock.patch.object(
            runner, "_motion_robot_capsules", return_value=far_capsule
        ), mock.patch.object(
            runner, "_tcp_transform_from_joints", return_value=np.eye(4)
        ):
            metrics = runner._motion_configuration_metrics(
                np.zeros(7), context, args
            )
        self.assertGreaterEqual(metrics["margin"], 0.0)

        neighbor = np.array([[0.003, 0.0, 0.0]], dtype=float)
        blocked_scene = safety_scene(np.vstack([support, neighbor]))
        blocked_release = runner.InitialSupportRelease(
            strict_environment_base=neighbor,
            support_points_base=support,
            start_tcp_pose=np.zeros(6),
        )
        blocked_context = runner._motion_context(
            blocked_scene,
            "support_release",
            attached_target_tcp=np.array([[0.0, 0.0, 0.0]]),
            initial_support_release=blocked_release,
        )
        with mock.patch.object(
            runner, "_motion_robot_capsules", return_value=far_capsule
        ), mock.patch.object(
            runner, "_tcp_transform_from_joints", return_value=np.eye(4)
        ):
            metrics = runner._motion_configuration_metrics(
                np.zeros(7), blocked_context, args
            )
        self.assertLess(metrics["margin"], 0.0)
        self.assertEqual(metrics["primitive"], "attached_target")

    def test_support_release_restores_strict_scene_and_limits_xy_drift(self):
        args = motion_args()
        support = np.array([[0.0, 0.0, 0.0]], dtype=float)
        strict = np.array([[1.0, 0.0, 0.0]], dtype=float)
        scene = safety_scene(np.vstack([support, strict]))
        release = runner.InitialSupportRelease(
            strict_environment_base=strict,
            support_points_base=support,
            start_tcp_pose=np.zeros(6),
            strict_after_rise=0.018,
            max_xy_drift=0.005,
        )
        context = runner._motion_context(
            scene,
            "support_release",
            attached_target_tcp=np.array([[0.0, 0.0, 0.0]]),
            initial_support_release=release,
        )
        far_capsule = [
            (
                np.array([10.0, 0.0, 0.0]),
                np.array([10.0, 0.0, 0.0]),
                0.01,
                "link1_2",
            )
        ]
        raised = np.eye(4)
        raised[2, 3] = 0.020
        with mock.patch.object(
            runner, "_motion_robot_capsules", return_value=far_capsule
        ), mock.patch.object(
            runner, "_tcp_transform_from_joints", return_value=raised
        ):
            metrics = runner._motion_configuration_metrics(
                np.zeros(7), context, args
            )
        self.assertGreaterEqual(metrics["margin"], 0.0)

        drifted = np.eye(4)
        drifted[0, 3] = 0.006
        with mock.patch.object(
            runner, "_motion_robot_capsules", return_value=far_capsule
        ), mock.patch.object(
            runner, "_tcp_transform_from_joints", return_value=drifted
        ):
            metrics = runner._motion_configuration_metrics(
                np.zeros(7), context, args
            )
        self.assertLess(metrics["margin"], 0.0)
        self.assertEqual(metrics["source"], "initial_support_constraint")

        args.motion_runtime_support_xy_drift_slack = 0.002
        with mock.patch.object(
            runner, "_motion_robot_capsules", return_value=far_capsule
        ), mock.patch.object(
            runner, "_tcp_transform_from_joints", return_value=drifted
        ):
            runtime_metrics = runner._motion_configuration_metrics(
                np.zeros(7), context, args, runtime=True
            )
        self.assertGreaterEqual(runtime_metrics["margin"], 0.0)

        excessive = np.eye(4)
        excessive[0, 3] = 0.008
        with mock.patch.object(
            runner, "_motion_robot_capsules", return_value=far_capsule
        ), mock.patch.object(
            runner, "_tcp_transform_from_joints", return_value=excessive
        ):
            excessive_metrics = runner._motion_configuration_metrics(
                np.zeros(7), context, args, runtime=True
            )
        self.assertLess(excessive_metrics["margin"], 0.0)
        self.assertEqual(
            excessive_metrics["primitive"], "support_release_xy_drift"
        )

    def test_support_release_runtime_allows_small_tilt_tracking_error(self):
        args = motion_args()
        args.motion_runtime_support_tilt_slack_deg = 2.0
        release = runner.InitialSupportRelease(
            strict_environment_base=np.array([[1.0, 0.0, 0.0]]),
            support_points_base=np.array([[0.0, 0.0, 0.0]]),
            start_tcp_pose=np.zeros(6),
            max_tilt_rad=np.deg2rad(3.0),
        )
        context = runner._motion_context(
            safety_scene([[1.0, 0.0, 0.0]]),
            "support_release",
            attached_target_tcp=np.array([[0.0, 0.0, 0.0]]),
            initial_support_release=release,
        )
        far_capsule = [
            (
                np.array([10.0, 0.0, 0.0]),
                np.array([10.0, 0.0, 0.0]),
                0.01,
                "link1_2",
            )
        ]

        def tilted(degrees):
            angle = np.deg2rad(float(degrees))
            transform = np.eye(4)
            transform[:2, :2] = [
                [np.cos(angle), -np.sin(angle)],
                [np.sin(angle), np.cos(angle)],
            ]
            return transform

        with mock.patch.object(
            runner, "_motion_robot_capsules", return_value=far_capsule
        ), mock.patch.object(
            runner, "_tcp_transform_from_joints", return_value=tilted(3.5)
        ):
            planned = runner._motion_configuration_metrics(
                np.zeros(7), context, args
            )
            runtime = runner._motion_configuration_metrics(
                np.zeros(7), context, args, runtime=True
            )
        self.assertLess(planned["margin"], 0.0)
        self.assertEqual(planned["primitive"], "support_release_tilt")
        self.assertGreaterEqual(runtime["margin"], 0.0)

        with mock.patch.object(
            runner, "_motion_robot_capsules", return_value=far_capsule
        ), mock.patch.object(
            runner, "_tcp_transform_from_joints", return_value=tilted(5.5)
        ):
            excessive = runner._motion_configuration_metrics(
                np.zeros(7), context, args, runtime=True
            )
        self.assertLess(excessive["margin"], 0.0)
        self.assertEqual(excessive["primitive"], "support_release_tilt")

    def test_support_builder_keeps_neighbor_outside_target_hull_strict(self):
        args = motion_args()
        args.table_guard_enable = True
        args.table_z = 0.0
        args.motion_path_scene_voxel = 0.0
        args.motion_path_obstacle_cap = 1000
        args.motion_path_fail_on_obstacle_cap = True
        args.motion_support_xy_margin = 0.003
        args.motion_support_min_points = 8
        grid = np.linspace(-0.005, 0.005, 11)
        target = np.array(
            [[x, y, 0.001] for x in grid for y in grid], dtype=float
        )
        support = np.array(
            [
                [x, y, 0.0]
                for x in (-0.004, 0.0, 0.004)
                for y in (-0.004, 0.0, 0.004)
            ],
            dtype=float,
        )
        neighbor = np.array([[0.0, 0.019, 0.0]], dtype=float)
        scene = safety_scene(np.vstack([support, neighbor]), target)

        release = runner._build_initial_support_release(
            scene, np.zeros(6), args
        )

        self.assertIsNotNone(release)
        self.assertTrue(
            np.any(
                np.all(
                    np.isclose(
                        release.strict_environment_base,
                        neighbor.reshape(1, 3),
                    ),
                    axis=1,
                )
            )
        )
        self.assertFalse(
            np.any(
                np.all(
                    np.isclose(
                        release.support_points_base,
                        neighbor.reshape(1, 3),
                    ),
                    axis=1,
                )
            )
        )

    def test_support_builder_separates_target_leakage_for_closed_tool(self):
        args = motion_args()
        args.table_guard_enable = True
        args.table_z = 0.0
        args.motion_path_scene_voxel = 0.0
        args.motion_path_obstacle_cap = 1000
        args.motion_path_fail_on_obstacle_cap = True
        args.motion_support_min_points = 8
        grid = np.linspace(-0.005, 0.005, 5)
        target = np.array(
            [[x, y, z] for x in grid for y in grid for z in (0.001, 0.020, 0.040)],
            dtype=float,
        )
        support = np.array(
            [[x, y, 0.0] for x in grid[::2] for y in grid[::2]],
            dtype=float,
        )
        leaked_target = np.array([[0.0, 0.0, 0.020]], dtype=float)
        basket_wall = np.array([[0.030, 0.0, 0.020]], dtype=float)
        scene = safety_scene(
            np.vstack([support, leaked_target, basket_wall]), target
        )

        release = runner._build_initial_support_release(
            scene, np.zeros(6), args
        )

        self.assertIsNotNone(release)
        self.assertTrue(
            np.any(
                np.all(
                    np.isclose(
                        release.target_contact_points_base,
                        leaked_target.reshape(1, 3),
                    ),
                    axis=1,
                )
            )
        )
        self.assertFalse(
            np.any(
                np.all(
                    np.isclose(
                        release.tool_contact_environment_base,
                        leaked_target.reshape(1, 3),
                    ),
                    axis=1,
                )
            )
        )
        self.assertTrue(
            np.any(
                np.all(
                    np.isclose(
                        release.tool_contact_environment_base,
                        basket_wall.reshape(1, 3),
                    ),
                    axis=1,
                )
            )
        )

    def test_support_builder_infers_basket_bottom_above_table(self):
        args = motion_args()
        args.table_guard_enable = True
        args.table_z = 0.08
        args.motion_path_scene_voxel = 0.0
        args.motion_path_obstacle_cap = 1000
        args.motion_path_fail_on_obstacle_cap = True
        args.motion_support_xy_margin = 0.003
        args.motion_support_min_points = 8
        args.motion_local_support_enable = True
        grid = np.linspace(-0.012, 0.012, 13)
        target = np.array(
            [[x, y, 0.152 + 0.002 * (x / 0.012) ** 2]
             for x in grid for y in grid],
            dtype=float,
        )
        support = np.array(
            [[x, y, 0.149 + 0.0004 * ((i % 3) - 1)]
             for i, (x, y) in enumerate(
                 ([(x, y) for x in grid[::3] for y in grid[::3]])
             )],
            dtype=float,
        )
        basket_wall = np.array(
            [[0.030, y, z]
             for y in np.linspace(-0.02, 0.02, 7)
             for z in np.linspace(0.149, 0.22, 7)],
            dtype=float,
        )
        scene = safety_scene(np.vstack([support, basket_wall]), target)

        release = runner._build_initial_support_release(
            scene, np.zeros(6), args
        )

        self.assertIsNotNone(release)
        self.assertGreaterEqual(release.support_points_base.shape[0], 8)
        self.assertGreaterEqual(release.strict_after_rise, 0.040)
        self.assertTrue(
            np.any(
                np.isclose(
                    release.strict_environment_base[:, 0], 0.030
                )
            )
        )

    def test_support_release_omits_patch_for_tool_until_strict_latch(self):
        args = motion_args()
        args.table_guard_enable = False
        release = runner.InitialSupportRelease(
            strict_environment_base=np.array([[1.0, 0.0, 0.0]]),
            support_points_base=np.array([[0.0, 0.0, 0.0]]),
            start_tcp_pose=np.zeros(6),
            strict_after_rise=0.040,
            downward_tolerance=0.001,
        )
        context = runner._motion_context(
            safety_scene([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
            "support_release",
            attached_target_tcp=np.array([[10.0, 10.0, 10.0]]),
            initial_support_release=release,
        )
        tool_capsule = [
            (
                np.array([0.0, 0.0, 0.0]),
                np.array([0.0, 0.0, 0.0]),
                0.01,
                "gripper_tool",
            )
        ]
        with mock.patch.object(
            runner, "_motion_robot_capsules", return_value=tool_capsule
        ), mock.patch.object(
            runner, "_tcp_transform_from_joints", return_value=np.eye(4)
        ):
            before_latch = runner._motion_configuration_metrics(
                np.zeros(7),
                context,
                args,
                support_strict_latched=False,
            )
            after_latch = runner._motion_configuration_metrics(
                np.zeros(7),
                context,
                args,
                support_strict_latched=True,
            )

        self.assertGreaterEqual(before_latch["margin"], 0.0)
        self.assertLess(after_latch["margin"], 0.0)
        self.assertEqual(after_latch["primitive"], "gripper_tool")

    def test_support_release_target_leakage_exception_is_tool_only(self):
        args = motion_args()
        args.table_guard_enable = False
        target_leak = np.array([[0.0, 0.0, 0.0]])
        far = np.array([[1.0, 0.0, 0.0]])
        release = runner.InitialSupportRelease(
            strict_environment_base=target_leak,
            support_points_base=np.zeros((0, 3)),
            start_tcp_pose=np.zeros(6),
            tool_contact_environment_base=far,
            target_contact_points_base=target_leak,
            strict_after_rise=0.040,
        )
        context = runner._motion_context(
            safety_scene(target_leak),
            "support_release",
            attached_target_tcp=np.array([[10.0, 10.0, 10.0]]),
            initial_support_release=release,
        )
        at_contact = (
            np.array([0.0, 0.0, 0.0]),
            np.array([0.0, 0.0, 0.0]),
            0.01,
        )
        with mock.patch.object(
            runner,
            "_motion_robot_capsules",
            return_value=[(*at_contact, "gripper_tool")],
        ), mock.patch.object(
            runner, "_tcp_transform_from_joints", return_value=np.eye(4)
        ):
            tool_metrics = runner._motion_configuration_metrics(
                np.zeros(7), context, args
            )
        self.assertGreaterEqual(tool_metrics["margin"], 0.0)

        with mock.patch.object(
            runner,
            "_motion_robot_capsules",
            return_value=[(*at_contact, "camera_mount")],
        ), mock.patch.object(
            runner, "_tcp_transform_from_joints", return_value=np.eye(4)
        ):
            camera_metrics = runner._motion_configuration_metrics(
                np.zeros(7), context, args
            )
        self.assertLess(camera_metrics["margin"], 0.0)
        self.assertEqual(camera_metrics["primitive"], "camera_mount")

    def test_support_release_latches_strict_after_rise_and_rejects_drop(self):
        args = motion_args()
        args.motion_path_sample_joint_deg = 2.0
        release = runner.InitialSupportRelease(
            strict_environment_base=np.array([[1.0, 0.0, 0.0]]),
            support_points_base=np.array([[0.0, 0.0, 0.0]]),
            start_tcp_pose=np.zeros(6),
            strict_after_rise=0.018,
            downward_tolerance=0.001,
        )
        context = runner._motion_context(
            safety_scene([[0.0, 0.0, 0.0]]),
            "support_release",
            attached_target_tcp=np.array([[0.0, 0.0, 0.0]]),
            initial_support_release=release,
        )
        rises = iter([0.0, 0.005, 0.019, 0.010, 0.015, 0.020])
        latch_arguments = []

        def fake_metrics(*_args, **kwargs):
            latch_arguments.append(bool(kwargs["support_strict_latched"]))
            return {
                "margin": 1.0,
                "nearest_clearance": 1.0,
                "required_clearance": 0.0,
                "primitive": "none",
                "source": "none",
                "support_rise": next(rises),
            }

        q1 = np.zeros(7)
        q1[0] = np.deg2rad(10.0)
        with mock.patch.object(
            runner, "_motion_configuration_metrics", side_effect=fake_metrics
        ):
            metrics = runner._motion_joint_path_metrics(
                np.zeros(7), q1, context, args
            )

        self.assertLess(metrics["margin"], 0.0)
        self.assertEqual(
            metrics["primitive"], "support_release_non_monotonic"
        )
        self.assertEqual(
            latch_arguments, [False, False, False, True, True, True]
        )
        self.assertTrue(metrics["support_strict_latched"])

    def test_intermediate_support_release_may_end_before_strict_latch(self):
        args = motion_args()
        args.table_guard_enable = False
        release = runner.InitialSupportRelease(
            strict_environment_base=np.array([[1.0, 0.0, 0.0]]),
            support_points_base=np.array([[0.0, 0.0, 0.0]]),
            start_tcp_pose=np.zeros(6),
            strict_after_rise=0.040,
            downward_tolerance=0.001,
        )
        context = runner._motion_context(
            safety_scene([[0.0, 0.0, 0.0]]),
            "support_release",
            attached_target_tcp=np.array([[0.0, 0.0, 0.0]]),
            initial_support_release=release,
            require_support_release_latch_at_endpoint=False,
        )
        metrics = {
            "margin": 1.0,
            "nearest_clearance": 1.0,
            "required_clearance": 0.0,
            "primitive": "none",
            "source": "none",
            "samples": 2,
            "sample_index": 1,
            "final_support_rise": 0.005,
            "support_strict_latched": False,
        }
        with mock.patch.object(
            runner, "_motion_joint_path_metrics", return_value=metrics
        ):
            runner.validate_motion_joint_path(
                np.zeros(7), np.zeros(7), context, args, "segment1_5mm"
            )

    def test_support_policy_is_not_carried_into_full_lift(self):
        scene = safety_scene([[1.0, 0.0, 0.0]])
        release = runner.InitialSupportRelease(
            strict_environment_base=scene.environment_base,
            support_points_base=np.array([[0.0, 0.0, 0.0]]),
            start_tcp_pose=np.zeros(6),
        )
        plan = runner.GraspMotionPlan(
            start_joints=np.zeros(7),
            before_close=tuple(),
            after_close=tuple(),
            after_release=tuple(),
            empty_recovery=tuple(),
            safety_scene=scene,
            attached_target_tcp=np.array([[0.0, 0.0, 0.0]]),
            initial_support_release=release,
        )
        support_stage = runner.PlannedJointStage(
            label="support",
            tcp_pose=np.zeros(6),
            flange_pose=np.zeros(6),
            joints=np.zeros(7),
            phase="support_release",
        )
        lift_stage = runner.PlannedJointStage(
            label="lift",
            tcp_pose=np.zeros(6),
            flange_pose=np.zeros(6),
            joints=np.zeros(7),
            phase="carrying",
        )

        support_context = runner._collision_context_for_planned_stage(
            plan, support_stage
        )
        lift_context = runner._collision_context_for_planned_stage(
            plan, lift_stage
        )

        self.assertIs(support_context.initial_support_release, release)
        self.assertIsNone(lift_context.initial_support_release)
        self.assertIs(lift_context.attached_target_tcp, plan.attached_target_tcp)

    def test_mask_owned_environment_preserves_adjacent_obstacle(self):
        args = motion_args()
        args.motion_path_scene_voxel = 0.0
        args.motion_path_obstacle_cap = 100
        args.motion_path_fail_on_obstacle_cap = True
        args.motion_target_exclusion_dist = 0.008
        target = SimpleNamespace(points=np.array([[0.0, 0.0, 0.0]]))
        full = SimpleNamespace(
            points=np.array([[0.0, 0.0, 0.0], [0.003, 0.0, 0.0]])
        )
        exact_environment = SimpleNamespace(
            points=np.array([[0.003, 0.0, 0.0]])
        )
        with mock.patch.object(
            runner, "transform_cloud", side_effect=lambda cloud, _T: cloud
        ):
            scene = runner.prepare_motion_safety_scene(
                target,
                full,
                np.eye(4),
                args,
                motion_environment_cloud=exact_environment,
            )
        np.testing.assert_allclose(
            scene.environment_base,
            np.array([[0.003, 0.0, 0.0]], dtype=np.float32),
        )

    def test_safe_endpoints_but_colliding_joint_midpoint_is_rejected(self):
        args = motion_args()
        scene = safety_scene([[0.0, 0.0, 0.0]])
        context = runner._motion_context(scene, "carrying")

        def moving_sphere(q, _args, open_gripper=False):
            center = np.array([float(q[0]), 0.0, 0.0])
            return [(center, center, 0.01, "link1_2")]

        q_start = np.zeros(7)
        q_target = np.zeros(7)
        q_start[0] = -0.20
        q_target[0] = 0.20
        with mock.patch.object(
            runner, "_motion_robot_capsules", side_effect=moving_sphere
        ):
            self.assertGreater(
                runner._motion_configuration_metrics(q_start, context, args)[
                    "margin"
                ],
                0.0,
            )
            self.assertGreater(
                runner._motion_configuration_metrics(q_target, context, args)[
                    "margin"
                ],
                0.0,
            )
            with self.assertRaises(runner.UnsafeMotionTarget):
                runner.validate_motion_joint_path(
                    q_start, q_target, context, args, "midpoint"
                )

    def test_route_height_prefers_above_scene(self):
        args = motion_args()
        scene = safety_scene([[0.0, 0.0, 0.30]])
        start = np.array([0.0, 0.0, 0.35, 0.0, 0.0, 0.0])
        goal = np.array([0.2, 0.0, 0.20, 0.0, 0.0, 0.0])
        heights, scene_top = runner._motion_route_height_candidates(
            start, goal, scene, args
        )
        self.assertAlmostEqual(scene_top, 0.30, places=6)
        self.assertGreaterEqual(heights[0], 0.38 - 1e-9)
        self.assertGreater(heights[0], goal[2])

    def test_large_high_orientation_change_is_adaptively_slerped(self):
        args = motion_args()
        args.motion_transition_max_segments = 2
        args.motion_high_path_z_tolerance = 0.020
        scene = safety_scene([[10.0, 10.0, 10.0]])
        start = np.array([0.0, 0.0, 0.40, 0.0, 0.0, 0.0])
        goal = np.array([0.0, 0.0, 0.40, 0.0, 0.0, np.pi / 2.0])
        calls = []

        def fake_stage(_arm, tcp, q_prev, context, _args, label, fixed_joints=None):
            calls.append(label)
            if label.endswith("orient_high"):
                raise runner.UnsafeMotionTarget("single rotation too large")
            q = np.asarray(q_prev, dtype=float).copy()
            q[0] += 0.1
            return runner.PlannedJointStage(
                label=label,
                tcp_pose=np.asarray(tcp, dtype=float),
                flange_pose=np.asarray(tcp, dtype=float),
                joints=q,
                phase=context.phase,
                min_tcp_z=context.min_tcp_z,
            )

        with mock.patch.object(
            runner,
            "_motion_route_height_candidates",
            return_value=([0.40], 0.30),
        ), mock.patch.object(
            runner, "_plan_tcp_motion_stage", side_effect=fake_stage
        ):
            stages = runner._plan_up_and_over_route(
                mock.Mock(),
                start,
                goal,
                np.zeros(7),
                scene,
                args,
                "adaptive",
                transit_phase="free_open",
            )
        self.assertEqual(len(stages), 2)
        self.assertIn("adaptive_orient_high_1of2", calls)
        self.assertIn("adaptive_orient_high_2of2", calls)

    def test_guarded_direct_route_does_not_call_high_planner(self):
        args = motion_args()
        scene = safety_scene([[10.0, 10.0, 10.0]])
        start = np.array([0.0, 0.0, 0.30, 0.0, 0.0, 0.0])
        goal = np.array([0.2, 0.0, 0.25, 0.0, 0.0, 0.0])
        calls = []

        def fake_stage(_arm, tcp, _q_prev, context, _args, label, fixed_joints=None):
            calls.append((label, np.asarray(tcp, dtype=float).copy(), context.phase))
            return runner.PlannedJointStage(
                label=label,
                tcp_pose=np.asarray(tcp, dtype=float),
                flange_pose=np.asarray(tcp, dtype=float),
                joints=np.full(7, 0.1),
                phase=context.phase,
                min_tcp_z=context.min_tcp_z,
            )

        with mock.patch.object(
            runner, "_plan_up_and_over_route"
        ) as high_planner, mock.patch.object(
            runner, "_plan_tcp_motion_stage", side_effect=fake_stage
        ), mock.patch.object(
            runner, "validate_motion_joint_path"
        ) as validate:
            stages = runner._plan_guarded_route(
                mock.Mock(),
                start,
                goal,
                np.zeros(7),
                scene,
                args,
                "transport",
                transit_phase="carrying",
                goal_phase="carrying",
            )

        high_planner.assert_not_called()
        validate.assert_not_called()
        self.assertEqual(len(stages), 1)
        self.assertEqual(calls[0][0], "transport_direct")
        self.assertEqual(calls[0][2], "carrying")
        np.testing.assert_allclose(calls[0][1], goal)

    def test_guarded_direct_release_return_only_uses_local_retreat(self):
        args = motion_args()
        scene = safety_scene([[10.0, 10.0, 10.0]])
        start = np.array([0.0, 0.0, 0.20, 0.0, 0.0, 0.0])
        goal = np.array([0.3, 0.1, 0.25, 0.0, 0.0, 0.0])
        calls = []

        def fake_stage(_arm, tcp, q_prev, context, _args, label, fixed_joints=None):
            calls.append((label, np.asarray(tcp, dtype=float).copy(), context.phase))
            joints = np.asarray(q_prev, dtype=float).copy() + 0.1
            return runner.PlannedJointStage(
                label=label,
                tcp_pose=np.asarray(tcp, dtype=float),
                flange_pose=np.asarray(tcp, dtype=float),
                joints=joints,
                phase=context.phase,
                min_tcp_z=context.min_tcp_z,
            )

        with mock.patch.object(
            runner, "_plan_up_and_over_route"
        ) as high_planner, mock.patch.object(
            runner, "_plan_tcp_motion_stage", side_effect=fake_stage
        ), mock.patch.object(runner, "validate_motion_joint_path") as validate:
            stages = runner._plan_guarded_route(
                mock.Mock(),
                start,
                goal,
                np.zeros(7),
                scene,
                args,
                "released_return",
                transit_phase="released_free",
                first_phase="released_retreat",
            )

        high_planner.assert_not_called()
        self.assertEqual(len(stages), 2)
        self.assertEqual(calls[0][0], "released_return_direct_release_retreat")
        self.assertEqual(calls[0][2], "released_retreat")
        self.assertAlmostEqual(calls[0][1][2], start[2] + 0.030)
        self.assertEqual(calls[1][0], "released_return_direct")
        self.assertEqual(calls[1][2], "released_free")
        np.testing.assert_allclose(calls[1][1], goal)
        validate.assert_called_once()

    def test_release_return_retries_bounded_xy_after_exact_retreat_ik_failure(self):
        args = motion_args()
        args.motion_direct_release_retreat = 0.020
        args.motion_release_retreat_xy_relax = 0.010
        args.motion_release_retreat_tilt_relax_deg = 5.0
        args.motion_release_retreat_max_attempts = 9
        scene = safety_scene([[10.0, 10.0, 10.0]])
        start = np.array([0.0, 0.0, 0.20, 0.0, 0.0, 0.0])
        goal = np.array([0.3, 0.1, 0.25, 0.0, 0.0, 0.0])
        calls = []

        def fake_stage(_arm, tcp, q_prev, context, _args, label, fixed_joints=None):
            tcp = np.asarray(tcp, dtype=float).copy()
            calls.append((label, tcp, context.phase))
            if label == "released_return_direct_release_retreat":
                raise runner.PlanNotExecutable("exact retreat IK failed")
            return runner.PlannedJointStage(
                label=label,
                tcp_pose=tcp,
                flange_pose=tcp,
                joints=np.asarray(q_prev, dtype=float).copy() + 0.1,
                phase=context.phase,
                min_tcp_z=context.min_tcp_z,
            )

        with mock.patch.object(
            runner, "_plan_tcp_motion_stage", side_effect=fake_stage
        ), mock.patch.object(runner, "validate_motion_joint_path") as validate:
            stages = runner._plan_guarded_route(
                mock.Mock(),
                start,
                goal,
                np.zeros(7),
                scene,
                args,
                "released_return",
                transit_phase="released_free",
                first_phase="released_retreat",
            )

        self.assertEqual(len(stages), 2)
        self.assertEqual(
            calls[1][0], "released_return_direct_release_retreat_x_plus"
        )
        self.assertAlmostEqual(calls[1][1][0], start[0] + 0.010)
        self.assertAlmostEqual(calls[1][1][1], start[1])
        self.assertAlmostEqual(calls[1][1][2], start[2] + 0.020)
        validate.assert_called_once()

    def test_complete_workflow_preflight_hides_failed_candidate_and_keeps_next(self):
        args = motion_args()
        args.grasp_motion_path_guard = True
        args.motion_up_and_over = False
        args.allow_hard_grasp_candidate = False
        args.allow_oversize_grasp = False
        args.max_grasp_width = 0.090
        args.topdown_grasp = False
        args.use_current_rpy = False
        args.fixed_rpy = None
        args.table_guard_enable = False
        grasp_1 = SimpleNamespace(
            width=0.060,
            depth=0.010,
            rotation_matrix=np.eye(3),
            translation=np.zeros(3),
        )
        grasp_4 = SimpleNamespace(
            width=0.065,
            depth=0.010,
            rotation_matrix=np.eye(3),
            translation=np.ones(3) * 0.01,
        )
        rows = [
            {
                "idx": 1,
                "grasp": grasp_1,
                "reachability_ok": True,
                "table_guard_ok": True,
                "tcp_width_sign": 1.0,
            },
            {
                "idx": 4,
                "grasp": grasp_4,
                "reachability_ok": True,
                "table_guard_ok": True,
                "tcp_width_sign": 1.0,
            },
        ]
        pre = np.array([0.0, 0.0, 0.30, 0.0, 0.0, 0.0])
        grasp = np.array([0.0, 0.0, 0.20, 0.0, 0.0, 0.0])
        lift = np.array([0.0, 0.0, 0.40, 0.0, 0.0, 0.0])
        valid_plan = object()

        def fake_plan(*call_args, **_kwargs):
            label = str(call_args[9])
            if "candidate1" in label:
                raise runner.UnsafeMotionTarget("finger hits neighbor")
            return valid_plan

        with mock.patch.object(
            runner,
            "grasp_to_tcp_poses",
            return_value=(pre, grasp, lift),
        ), mock.patch.object(
            runner,
            "plan_up_and_over_grasp_sequence",
            side_effect=fake_plan,
        ):
            annotated = runner.annotate_grasp_candidate_workflow_preflight(
                rows,
                mock.Mock(),
                args,
                object(),
                object(),
                np.eye(4),
                np.zeros(6),
                "clear1",
            )

        self.assertFalse(annotated[0]["workflow_preflight_ok"])
        self.assertTrue(annotated[1]["workflow_preflight_ok"])
        self.assertIs(annotated[1]["_workflow_motion_plan"], valid_plan)
        executable = runner.filter_grasp_candidates_by_workflow_preflight(
            annotated, strict=True
        )
        self.assertEqual([row["idx"] for row in executable], [4])
        _, selected = runner.choose_default_grasp_candidate(
            annotated, [], args
        )
        self.assertEqual(selected["idx"], 4)
        self.assertNotIn(
            "_workflow_motion_plan",
            runner._candidate_to_jsonable(annotated[1]),
        )

    def test_full_mesh_table_failure_skips_candidate_before_joint_planning(self):
        args = motion_args()
        args.grasp_motion_path_guard = True
        args.grasp_table_reference_z = 0.
        args.motion_path_min_clearance = .005
        args.max_grasp_width = .1
        args.topdown_grasp = False
        args.use_current_rpy = False
        args.fixed_rpy = None
        rows = [dict(idx=index, grasp=SimpleNamespace(width=.06, depth=.03,
                    rotation_matrix=np.eye(3), translation=np.array([index, 0, .3])),
                    reachability_ok=True, table_guard_ok=True, tcp_width_sign=1.)
                for index in (1, 4)]
        def poses(grasp, *_a, **_k):
            x = grasp.translation[0]
            return tuple(np.array([x, 0, z, 0, 0, 0]) for z in (.4, .3, .45))
        def mesh(pre, grasp, local_args):
            self.assertEqual(local_args.graspgenx_table_margin, .005)
            if grasp[0] == 1:
                raise RuntimeError("[grasp][mesh-table] table clearance 3.2mm < required 5.0mm")
            return .00645
        valid_plan = object()
        planner = mock.Mock(return_value=valid_plan)
        with mock.patch.object(runner, "grasp_to_tcp_poses", side_effect=poses), \
             mock.patch.object(runner, "plan_up_and_over_grasp_sequence", planner), \
             mock.patch("real_nero_grasp.graspgenx_integration.selected_mesh_table_clearance",
                        side_effect=mesh):
            annotated = runner.annotate_grasp_candidate_workflow_preflight(
                rows, mock.Mock(), args, object(), object(), np.eye(4), np.zeros(6), "target")
        self.assertFalse(annotated[0]["workflow_preflight_ok"])
        self.assertIn("3.2mm", annotated[0]["workflow_preflight_reason"])
        self.assertTrue(annotated[1]["workflow_preflight_ok"])
        self.assertEqual(annotated[1]["workflow_preflight_mesh_table_clearance_m"], .00645)
        self.assertTrue(all("candidate4" in str(call.args[9]) for call in planner.call_args_list))
        _, selected = runner.choose_default_grasp_candidate(annotated, [], args)
        self.assertEqual(selected["idx"], 4)

    def test_workflow_does_not_plan_180_fallback_when_primary_route_passes(self):
        args = motion_args()
        args.grasp_motion_path_guard = True
        args.allow_hard_grasp_candidate = False
        args.allow_oversize_grasp = False
        args.max_grasp_width = 0.090
        args.topdown_grasp = False
        args.use_current_rpy = False
        args.fixed_rpy = None
        args.graspnet_approach_axis = "x"
        args.graspnet_width_axis = "y"
        args.tool_approach_axis = "x"
        args.tcp_gripper_width_axis = "z"
        candidate = SimpleNamespace(
            width=0.060,
            depth=0.010,
            rotation_matrix=np.eye(3),
            translation=np.zeros(3),
        )
        rows = [
            {
                "idx": 8,
                "grasp": candidate,
                "reachability_ok": True,
                "reachability_ok_signs": [1.0, -1.0],
                "table_guard_ok": True,
            }
        ]
        poses = (
            np.array([0.0, 0.0, 0.30, 0.0, 0.0, 0.0]),
            np.array([0.0, 0.0, 0.20, 0.0, 0.0, 0.0]),
            np.array([0.0, 0.0, 0.40, 0.0, 0.0, 0.0]),
        )
        pose_signs = []

        def fake_poses(*_args, **kwargs):
            pose_signs.append(float(kwargs["width_sign"]))
            return tuple(pose.copy() for pose in poses)

        valid_plan = object()
        with mock.patch.object(
            runner, "grasp_to_tcp_poses", side_effect=fake_poses
        ), mock.patch.object(
            runner,
            "plan_up_and_over_grasp_sequence",
            return_value=valid_plan,
        ) as planner:
            annotated = runner.annotate_grasp_candidate_workflow_preflight(
                rows,
                mock.Mock(),
                args,
                object(),
                object(),
                np.eye(4),
                np.zeros(6),
                "target",
            )

        self.assertEqual(pose_signs, [1.0])
        planner.assert_called_once()
        self.assertTrue(annotated[0]["workflow_preflight_ok"])
        self.assertEqual(annotated[0]["workflow_preflight_width_sign"], 1.0)
        self.assertEqual(
            annotated[0]["workflow_preflight_jaw_role"], "primary"
        )

    def test_workflow_rejects_candidate_when_native_route_fails(self):
        args = motion_args()
        args.grasp_motion_path_guard = True
        args.allow_hard_grasp_candidate = False
        args.allow_oversize_grasp = False
        args.max_grasp_width = 0.090
        args.topdown_grasp = False
        args.use_current_rpy = False
        args.fixed_rpy = None
        args.graspnet_approach_axis = "x"
        args.graspnet_width_axis = "y"
        args.tool_approach_axis = "x"
        args.tcp_gripper_width_axis = "z"
        candidate = SimpleNamespace(
            width=0.060,
            depth=0.010,
            rotation_matrix=np.eye(3),
            translation=np.zeros(3),
        )
        rows = [
            {
                "idx": 9,
                "grasp": candidate,
                "reachability_ok": True,
                "reachability_ok_signs": [1.0, -1.0],
                "table_guard_ok": True,
            }
        ]
        poses = (
            np.array([0.0, 0.0, 0.30, 0.0, 0.0, 0.0]),
            np.array([0.0, 0.0, 0.20, 0.0, 0.0, 0.0]),
            np.array([0.0, 0.0, 0.40, 0.0, 0.0, 0.0]),
        )
        pose_signs = []

        def fake_poses(*_args, **kwargs):
            pose_signs.append(float(kwargs["width_sign"]))
            return tuple(pose.copy() for pose in poses)

        with mock.patch.object(
            runner, "grasp_to_tcp_poses", side_effect=fake_poses
        ), mock.patch.object(
            runner,
            "plan_up_and_over_grasp_sequence",
            side_effect=runner.UnsafeMotionTarget("native route blocked"),
        ) as planner:
            annotated = runner.annotate_grasp_candidate_workflow_preflight(
                rows,
                mock.Mock(),
                args,
                object(),
                object(),
                np.eye(4),
                np.zeros(6),
                "target",
            )

        self.assertEqual(pose_signs, [1.0])
        planner.assert_called_once()
        self.assertFalse(annotated[0]["workflow_preflight_ok"])
        self.assertNotIn("_workflow_motion_plan", annotated[0])

    def test_strict_workflow_filter_rejects_unannotated_candidates(self):
        rows = [{"idx": 0, "grasp": object()}]
        self.assertEqual(
            runner.filter_grasp_candidates_by_workflow_preflight(
                rows, strict=True
            ),
            [],
        )
        self.assertEqual(
            runner.filter_grasp_candidates_by_workflow_preflight(
                rows, strict=False
            ),
            rows,
        )

    def test_workflow_uses_only_aperture_valid_sign_and_recenter_shift(self):
        args = motion_args()
        args.grasp_motion_path_guard = True
        args.allow_hard_grasp_candidate = False
        args.allow_oversize_grasp = False
        args.max_grasp_width = 0.090
        args.topdown_grasp = False
        args.use_current_rpy = False
        args.fixed_rpy = None
        args.table_guard_enable = False
        candidate = SimpleNamespace(
            width=0.060,
            depth=0.010,
            rotation_matrix=np.eye(3),
            translation=np.zeros(3),
        )
        rows = [
            {
                "idx": 3,
                "grasp": candidate,
                "reachability_ok": True,
                "table_guard_ok": True,
                "tcp_width_sign": -1.0,
                "physical_aperture_ok": True,
                "physical_aperture_ok_signs": [1.0],
                "physical_aperture_variants": [
                    {
                        "width_sign": 1.0,
                        "tcp_width_shift": 0.004,
                        "ok": True,
                    }
                ],
            }
        ]
        poses = (
            np.array([0.0, 0.0, 0.30, 0.0, 0.0, 0.0]),
            np.array([0.0, 0.0, 0.20, 0.0, 0.0, 0.0]),
            np.array([0.0, 0.0, 0.40, 0.0, 0.0, 0.0]),
        )
        with mock.patch.object(
            runner, "grasp_to_tcp_poses", return_value=poses
        ) as pose_mock, mock.patch.object(
            runner,
            "plan_up_and_over_grasp_sequence",
            return_value=object(),
        ):
            annotated = runner.annotate_grasp_candidate_workflow_preflight(
                rows,
                mock.Mock(),
                args,
                object(),
                object(),
                np.eye(4),
                np.zeros(6),
                "target",
            )

        self.assertTrue(annotated[0]["workflow_preflight_ok"])
        self.assertEqual(annotated[0]["workflow_preflight_width_sign"], 1.0)
        self.assertEqual(annotated[0]["workflow_preflight_width_shift"], 0.004)
        pose_mock.assert_called_once()
        self.assertEqual(pose_mock.call_args.kwargs["width_sign"], 1.0)
        self.assertEqual(pose_mock.call_args.kwargs["width_shift"], 0.004)

    def test_workflow_reannotation_discards_stale_success_and_plan(self):
        args = motion_args()
        args.grasp_motion_path_guard = True
        args.allow_hard_grasp_candidate = False
        args.allow_oversize_grasp = False
        args.max_grasp_width = 0.090
        args.topdown_grasp = False
        args.use_current_rpy = False
        args.fixed_rpy = None
        args.table_guard_enable = False
        stale_plan = object()
        rows = [
            {
                "idx": 2,
                "grasp": SimpleNamespace(
                    width=0.060,
                    depth=0.010,
                    rotation_matrix=np.eye(3),
                    translation=np.zeros(3),
                ),
                "reachability_ok": True,
                "table_guard_ok": True,
                "tcp_width_sign": 1.0,
                "workflow_preflight_ok": True,
                "workflow_preflight_reason": "old scene",
                "_workflow_motion_plan": stale_plan,
                "_workflow_pre_tcp": np.ones(6),
            }
        ]
        poses = (
            np.array([0.0, 0.0, 0.30, 0.0, 0.0, 0.0]),
            np.array([0.0, 0.0, 0.20, 0.0, 0.0, 0.0]),
            np.array([0.0, 0.0, 0.40, 0.0, 0.0, 0.0]),
        )
        with mock.patch.object(
            runner, "grasp_to_tcp_poses", return_value=poses
        ), mock.patch.object(
            runner,
            "plan_up_and_over_grasp_sequence",
            side_effect=runner.UnsafeMotionTarget("new scene collision"),
        ):
            annotated = runner.annotate_grasp_candidate_workflow_preflight(
                rows,
                mock.Mock(),
                args,
                object(),
                object(),
                np.eye(4),
                np.zeros(6),
                "target",
            )
        self.assertFalse(annotated[0]["workflow_preflight_ok"])
        self.assertIn("new scene collision", annotated[0]["workflow_preflight_reason"])
        self.assertNotIn("_workflow_motion_plan", annotated[0])
        self.assertNotIn("_workflow_pre_tcp", annotated[0])

    def test_whole_sequence_failure_happens_before_any_command(self):
        with mock.patch.object(
            sys, "argv", ["runner", "--calib-json", "/tmp/fake_calib.json"]
        ):
            args = runner.parse_args()
        args.execute = True
        args.yes = True
        args.table_guard_enable = False
        arm = mock.Mock()
        arm.get_tcp_pose.return_value = np.zeros(6)
        arm.get_joint_angles.return_value = np.zeros(7)
        fake_grasp = SimpleNamespace(
            width=0.04,
            depth=0.01,
            rotation_matrix=np.eye(3),
            translation=np.zeros(3),
        )
        row = {
            "idx": 0,
            "grasp": fake_grasp,
            "reachability_ok": True,
            "tcp_width_sign": 1.0,
            "planned_start_joints": np.zeros(7),
            "planned_pre_joints": np.zeros(7),
            "planned_grasp_joints": np.zeros(7),
            "planned_lift_joints": np.zeros(7),
        }
        summary = {
            "graspnet_width": 0.04,
            "graspnet_depth": 0.01,
            "graspnet_translation_camera": np.zeros(3).tolist(),
            "graspnet_rpy_camera_deg": np.zeros(3).tolist(),
            "graspnet_rotation_matrix_camera": np.eye(3).tolist(),
        }
        axes = {
            "grasp_approach_base": np.array([1.0, 0.0, 0.0]),
            "grasp_width_base": np.array([0.0, 1.0, 0.0]),
            "tcp_approach_base": np.array([1.0, 0.0, 0.0]),
            "tcp_width_base": np.array([0.0, 1.0, 0.0]),
            "approach_angle_deg": 0.0,
            "width_angle_deg": 0.0,
        }
        pre = np.array([0.0, 0.0, 0.30, 0.0, 0.0, 0.0])
        grasp = np.array([0.0, 0.0, 0.20, 0.0, 0.0, 0.0])
        lift = np.array([0.0, 0.0, 0.40, 0.0, 0.0, 0.0])
        with tempfile.TemporaryDirectory() as out_dir, mock.patch.object(
            runner,
            "get_graspnet_helpers",
            return_value=(
                mock.Mock(return_value=(object(), object())),
                mock.Mock(return_value=([fake_grasp], [row])),
            ),
        ), mock.patch.object(
            runner,
            "annotate_grasp_candidate_physical_aperture",
            return_value=[
                {
                    **row,
                    "physical_aperture_ok": True,
                    "physical_aperture_ok_signs": [1.0, -1.0],
                    "physical_aperture_variants": [
                        {
                            "width_sign": 1.0,
                            "tcp_width_shift": 0.0,
                            "ok": True,
                        },
                        {
                            "width_sign": -1.0,
                            "tcp_width_shift": 0.0,
                            "ok": True,
                        },
                    ],
                }
            ],
        ), mock.patch.object(
            runner,
            "annotate_grasp_candidate_reachability",
            side_effect=lambda rows, *_args, **_kwargs: rows,
        ), mock.patch.object(
            runner, "preview_graspnet_pointcloud"
        ), mock.patch.object(
            runner, "choose_default_grasp_candidate", return_value=(fake_grasp, row)
        ), mock.patch.object(
            runner, "grasp_to_tcp_poses", return_value=(pre, grasp, lift)
        ), mock.patch.object(
            runner, "grasp_camera_summary", return_value=summary
        ), mock.patch.object(
            runner, "execution_gripper_open_width", return_value=0.085
        ), mock.patch.object(
            runner, "save_selected_grasp_pose"
        ), mock.patch.object(
            runner, "grasp_tcp_axis_diagnostics", return_value=axes
        ), mock.patch.object(
            runner,
            "plan_up_and_over_grasp_sequence",
            side_effect=runner.UnsafeMotionTarget("transport hits basket"),
        ):
            result = runner.plan_and_maybe_execute_grasp(
                arm,
                args,
                object(),
                object(),
                np.eye(4),
                np.zeros((4, 4, 3), dtype=np.uint8),
                object(),
                out_dir,
                {"label": "carrot"},
            )
            with open(out_dir + "/grasp_candidates.json", encoding="utf-8") as stream:
                rejected_rows = json.load(stream)
            self.assertFalse(rejected_rows[0]["workflow_preflight_ok"])
            self.assertIn("transport hits basket", rejected_rows[0]["workflow_preflight_reason"])
        self.assertEqual(result.status, runner.GraspAttemptStatus.PLANNING_FAILED)
        self.assertFalse(result.motion_sent)
        arm.open_gripper.assert_not_called()
        arm.move_j.assert_not_called()


if __name__ == "__main__":
    unittest.main()
