import unittest
from unittest import mock
from types import SimpleNamespace

import numpy as np

import real_nero_single_grasp_runner as runner


def path_args():
    return SimpleNamespace(
        ik_urdf=runner.DEFAULT_IK_URDF,
        ik_base_link="base_link",
        ik_tip_link="link7",
        tcp_offset=runner.DEFAULT_TCP_OFFSET,
        active_path_link_radius=0.060,
        active_path_tool_radius=0.090,
        active_path_camera_radius=0.055,
        active_path_min_clearance=0.015,
        active_path_sample_joint_deg=2.0,
        active_path_scene_voxel=0.0,
        active_path_obstacle_cap=50000,
        active_path_require_scene=True,
        active_path_require_camera_model=False,
    )


class ActiveMotionPathGuardTest(unittest.TestCase):
    def test_return_micro_waypoints_can_be_disabled_explicitly(self):
        args = SimpleNamespace(active_return_execution_waypoint_deg=0.0)
        stage = runner.PlannedJointStage(
            label="return_direct",
            tcp_pose=np.zeros(6),
            flange_pose=np.zeros(6),
            joints=np.full(7, 0.5),
            phase="free_open",
        )
        result = runner._subdivide_active_direct_stage(
            np.zeros(7), stage, args
        )
        self.assertEqual(result, (stage,))

    def test_return_micro_waypoints_limit_each_joint_sweep_to_ten_degrees(self):
        args = SimpleNamespace(
            active_return_execution_waypoint_deg=10.0,
            min_flange_z=None,
        )
        stage = runner.PlannedJointStage(
            label="return_direct",
            tcp_pose=np.zeros(6),
            flange_pose=np.zeros(6),
            joints=np.array([0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            phase="free_open",
        )

        def fake_geometry(joints, _args):
            T = np.eye(4)
            T[0, 3] = float(np.asarray(joints)[0])
            return None, T

        with mock.patch(
            "real_nero_grasp.runtime._urdf_chain_geometry",
            side_effect=fake_geometry,
        ), mock.patch(
            "real_nero_grasp.runtime._tcp_pose_for_flange_pose",
            side_effect=lambda pose, _args: pose,
        ):
            result = runner._subdivide_active_direct_stage(
                np.zeros(7), stage, args
            )

        self.assertEqual(len(result), 3)
        nodes = [np.zeros(7)] + [item.joints for item in result]
        for before, after in zip(nodes[:-1], nodes[1:]):
            self.assertLessEqual(
                float(np.rad2deg(np.max(np.abs(after - before)))),
                10.0 + 1e-9,
            )
        np.testing.assert_allclose(result[-1].joints, stage.joints)

    def test_center_roll_samples_cover_full_circle_without_duplicate_endpoint(self):
        candidates = runner._look_at_camera_pose_flange_roll_candidates(
            np.array([0.0, 0.0, 1.0]),
            np.zeros(3),
            np.eye(4),
            np.zeros(6),
            roll_samples=72,
        )

        rolls_deg = np.rad2deg(
            [float(candidate["roll_rad"]) for candidate in candidates]
        )
        self.assertEqual(len(candidates), 72)
        self.assertEqual(len(np.unique(np.round(rolls_deg, 6))), 72)
        self.assertLessEqual(float(np.min(rolls_deg)), -179.9)
        self.assertGreaterEqual(float(np.max(rolls_deg)), 174.9)
        rotation_costs = [
            float(candidate["rotation_error_deg"])
            for candidate in candidates
        ]
        self.assertEqual(rotation_costs, sorted(rotation_costs))

    def test_active_scene_overflow_fails_closed(self):
        args = path_args()
        args.active_path_scene_voxel = 0.0
        args.active_path_obstacle_cap = 2
        args.motion_path_fail_on_obstacle_cap = True
        with self.assertRaisesRegex(
            runner.UnsafeMotionTarget, "refusing to discard sparse obstacles"
        ):
            runner._active_motion_scene_points(
                np.array([[0.0, 0.0, 0.0]]),
                np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
                args,
            )

    def test_open_gripper_capsule_rejects_scene_contact(self):
        args = path_args()
        q = np.zeros(7, dtype=float)
        tool = next(
            capsule
            for capsule in runner._active_robot_capsules(q, args)
            if capsule[3] == "open_gripper"
        )
        a, b, _, _ = tool
        scene_point = 0.5 * (a + b)

        metrics = runner._active_robot_path_metrics(
            q, q, scene_point.reshape(1, 3), args
        )

        self.assertEqual(metrics["primitive"], "open_gripper")
        self.assertLess(metrics["nearest_clearance"], 0.0)
        with self.assertRaises(runner.UnsafeMotionTarget):
            runner._validate_active_joint_path(
                q,
                q,
                scene_point.reshape(1, 3),
                args,
                "contact",
                45.0,
                75.0,
            )

    def test_effective_open_gripper_envelope_is_105mm(self):
        args = path_args()
        q = np.zeros(7, dtype=float)
        a, b, _, _ = next(
            capsule
            for capsule in runner._active_robot_capsules(q, args)
            if capsule[3] == "open_gripper"
        )
        axis = b - a
        axis = axis / np.linalg.norm(axis)
        helper = np.array([0.0, 0.0, 1.0])
        if abs(float(np.dot(axis, helper))) > 0.9:
            helper = np.array([0.0, 1.0, 0.0])
        normal = np.cross(axis, helper)
        normal = normal / np.linalg.norm(normal)
        # 104 mm from the tool centerline leaves only 14 mm outside the
        # physical 90 mm capsule, below the required 15 mm extra clearance.
        scene_point = 0.5 * (a + b) + normal * 0.104

        with self.assertRaises(runner.UnsafeMotionTarget):
            runner._validate_active_joint_path(
                q,
                q,
                scene_point.reshape(1, 3),
                args,
                "tool_margin",
                45.0,
                75.0,
            )

    def test_bounded_revolute_joint_delta_is_not_wrapped(self):
        args = path_args()
        q_start = np.zeros(7, dtype=float)
        q_target = np.zeros(7, dtype=float)
        q_start[0] = 2.6
        q_target[0] = -2.6
        far_scene = np.array([[10.0, 10.0, 10.0]], dtype=float)

        with self.assertRaisesRegex(
            runner.UnsafeMotionTarget, "largest joint sweep"
        ):
            runner._validate_active_joint_path(
                q_start,
                q_target,
                far_scene,
                args,
                "raw_delta",
                45.0,
                400.0,
            )

    def test_far_scene_passes_static_path(self):
        args = path_args()
        q = np.zeros(7, dtype=float)
        metrics = runner._validate_active_joint_path(
            q,
            q,
            np.array([[10.0, 10.0, 10.0]], dtype=float),
            args,
            "far",
            45.0,
            75.0,
        )
        self.assertGreater(metrics["nearest_clearance"], 1.0)

    def test_logged_side1_to_side2_sweep_is_rejected(self):
        args = path_args()
        q_side1 = np.deg2rad(
            [-124.549, 44.877, 146.254, 116.283, -24.295, 1.595, 46.744]
        )
        q_side2 = np.deg2rad(
            [-40.077, -50.540, 29.098, 122.305, -4.856, -31.197, 32.179]
        )
        # Scene point reconstructed from the user's 2026-07-15 11:18 run;
        # the old side-to-side move swept the open gripper through this area.
        scene_point = np.array([[-0.28408, -0.03241, 0.16799]], dtype=float)

        metrics = runner._active_robot_path_metrics(
            q_side1, q_side2, scene_point, args
        )

        self.assertEqual(metrics["primitive"], "open_gripper")
        self.assertLess(metrics["nearest_clearance"], 0.015)
        with self.assertRaises(runner.UnsafeMotionTarget):
            runner._validate_active_joint_path(
                q_side1,
                q_side2,
                scene_point,
                args,
                "logged_side_to_side",
                45.0,
                75.0,
            )

    def test_active_up_and_over_validates_every_intermediate_edge(self):
        args = path_args()
        args.motion_path_scene_voxel = 0.0
        args.motion_path_obstacle_cap = 50000
        q0 = np.zeros(7, dtype=float)
        q1 = np.full(7, 0.05, dtype=float)
        q2 = np.full(7, 0.10, dtype=float)

        def stage(label, joints):
            return runner.PlannedJointStage(
                label=label,
                tcp_pose=np.zeros(6),
                flange_pose=np.zeros(6),
                joints=joints,
                phase="free_open",
            )

        with mock.patch.object(
            runner,
            "_plan_up_and_over_route",
            return_value=(stage("raise", q1), stage("observe", q2)),
        ), mock.patch.object(
            runner, "_validate_active_joint_path", return_value={}
        ) as validate:
            plan = runner.plan_active_up_and_over_route(
                mock.Mock(),
                np.zeros(6),
                np.zeros(6),
                q0,
                np.array([[10.0, 10.0, 10.0]]),
                np.array([[0.0, 0.0, 0.0]]),
                args,
                "active_test",
                45.0,
                75.0,
            )

        self.assertEqual(len(plan.before_close), 2)
        self.assertEqual(validate.call_count, 2)
        np.testing.assert_allclose(validate.call_args_list[0].args[0], q0)
        np.testing.assert_allclose(validate.call_args_list[0].args[1], q1)
        np.testing.assert_allclose(validate.call_args_list[1].args[0], q1)
        np.testing.assert_allclose(validate.call_args_list[1].args[1], q2)

    def test_active_guarded_direct_uses_canonical_collision_check(self):
        args = path_args()
        args.table_guard_enable = False
        args.min_flange_z = None
        q0 = np.zeros(7, dtype=float)
        q1 = np.full(7, 0.05, dtype=float)
        target = np.array([[0.0, 0.0, 0.0]], dtype=float)
        scene = np.array([[1.0, 0.0, 0.0]], dtype=float)
        arm = mock.Mock()
        arm.get_tcp2flange_pose.return_value = np.zeros(6)

        with mock.patch.object(
            runner, "_solve_ik_solutions_for_flange_pose", return_value=[q1]
        ), mock.patch.object(
            runner, "_check_motion_joint_margin"
        ), mock.patch.object(
            runner, "validate_motion_joint_path", return_value=None
        ) as generic_validate, mock.patch.object(
            runner, "_validate_active_joint_path"
        ) as legacy_active_validate:
            plan = runner.plan_active_direct_route(
                arm,
                np.zeros(6),
                q0,
                scene,
                target,
                args,
                "direct_test",
                45.0,
                75.0,
            )

        self.assertEqual(len(plan.before_close), 1)
        np.testing.assert_allclose(plan.before_close[0].joints, q1)
        generic_validate.assert_called_once()
        np.testing.assert_allclose(generic_validate.call_args.args[0], q0)
        np.testing.assert_allclose(generic_validate.call_args.args[1], q1)
        self.assertEqual(generic_validate.call_args.args[2].phase, "free_open")
        # The selected target is re-added to the full active safety map.
        self.assertEqual(plan.safety_scene.full_scene_base.shape[0], 2)
        legacy_active_validate.assert_not_called()

    def test_active_return_is_one_command_with_floor_and_extra_clearance(self):
        args = path_args()
        args.active_direct_return_downward_tolerance = 0.002
        args.active_direct_return_runtime_downward_tolerance = 0.010
        args.active_direct_return_min_clearance = 0.030
        args.active_direct_min_joint_limit_margin_deg = 0.0
        q0 = np.zeros(7, dtype=float)
        q_center = np.full(7, 0.04, dtype=float)
        scene = runner.MotionSafetyScene(
            full_scene_base=np.array([[10.0, 10.0, 10.0]]),
            environment_base=np.array([[10.0, 10.0, 10.0]]),
            target_base=np.zeros((0, 3)),
            map_id="return_retreat",
        )
        calls = []

        def fake_stage(
            _arm,
            tcp,
            q_previous,
            context,
            _args,
            label,
            fixed_joints=None,
        ):
            calls.append(
                (
                    label,
                    np.asarray(tcp, dtype=float).copy(),
                    np.asarray(q_previous, dtype=float).copy(),
                    context.min_tcp_z,
                    context.runtime_min_tcp_z_slack,
                )
            )
            return runner.PlannedJointStage(
                label=label,
                tcp_pose=np.asarray(tcp, dtype=float).copy(),
                flange_pose=np.zeros(6),
                joints=np.asarray(fixed_joints, dtype=float).copy(),
                phase="free_open",
                min_tcp_z=context.min_tcp_z,
            )

        with mock.patch.object(
            runner,
            "_prepare_active_route_safety_scene",
            return_value=(np.zeros((0, 3)), scene.full_scene_base, scene),
        ), mock.patch.object(
            runner,
            "_modeled_tcp_pose_for_joints",
            return_value=np.array([0.1, 0.2, 0.30, 0.0, 0.0, 0.0]),
        ), mock.patch.object(
            runner,
            "_tcp_pose_for_flange_pose",
            return_value=np.array([0.0, 0.0, 0.45, 0.0, 0.0, 0.0]),
        ), mock.patch.object(
            runner, "_plan_tcp_motion_stage", side_effect=fake_stage
        ), mock.patch.object(
            runner,
            "validate_motion_joint_path",
        ) as robust_validate:
            plan = runner.plan_active_direct_route(
                mock.Mock(),
                np.zeros(6),
                q0,
                scene.full_scene_base,
                np.zeros((0, 3)),
                args,
                "side_return",
                45.0,
                75.0,
                goal_joints=q_center,
                initial_retreat_height=0.0,
                enforce_single_stage_return=True,
            )

        self.assertEqual(len(plan.before_close), 1)
        self.assertEqual(len(calls), 1)
        self.assertIn("direct", calls[0][0])
        self.assertAlmostEqual(calls[0][3], 0.298)
        self.assertAlmostEqual(calls[0][4], 0.008)
        robust_validate.assert_called_once()
        np.testing.assert_allclose(robust_validate.call_args.args[0], q0)
        np.testing.assert_allclose(
            robust_validate.call_args.args[1], q_center
        )
        self.assertEqual(robust_validate.call_args.args[2].phase, "free_open")
        self.assertAlmostEqual(
            robust_validate.call_args.args[3].motion_path_min_clearance,
            0.030,
        )

    def test_active_guarded_direct_fails_closed_on_canonical_collision(self):
        args = path_args()
        q0 = np.zeros(7, dtype=float)
        with mock.patch.object(
            runner,
            "_plan_tcp_motion_stage",
            side_effect=runner.UnsafeMotionTarget("basket collision"),
        ):
            with self.assertRaisesRegex(
                runner.UnsafeMotionTarget, "basket collision"
            ):
                runner.plan_active_direct_route(
                    mock.Mock(),
                    np.zeros(6),
                    q0,
                    np.array([[1.0, 0.0, 0.0]]),
                    np.array([[0.0, 0.0, 0.0]]),
                    args,
                    "unsafe",
                    45.0,
                    75.0,
                )

    def test_active_guarded_direct_rejects_endpoint_too_close_to_joint_limit(self):
        args = path_args()
        args.active_direct_min_joint_limit_margin_deg = 1.5
        q0 = np.zeros(7, dtype=float)
        q1 = np.full(7, 0.05, dtype=float)
        stage = runner.PlannedJointStage(
            label="near_limit_direct",
            tcp_pose=np.zeros(6),
            flange_pose=np.zeros(6),
            joints=q1,
            phase="free_open",
        )
        safety_scene = runner.MotionSafetyScene(
            full_scene_base=np.array([[10.0, 10.0, 10.0]]),
            environment_base=np.array([[10.0, 10.0, 10.0]]),
            target_base=np.zeros((0, 3)),
            map_id="near_limit",
        )

        with mock.patch.object(
            runner,
            "_prepare_active_route_safety_scene",
            return_value=(
                np.zeros((0, 3)),
                np.array([[10.0, 10.0, 10.0]]),
                safety_scene,
            ),
        ), mock.patch.object(
            runner, "_plan_tcp_motion_stage", return_value=stage
        ), mock.patch.object(
            runner, "_get_ik_solver", return_value=mock.Mock()
        ), mock.patch.object(
            runner, "_joint_limit_margin", return_value=np.deg2rad(1.0)
        ):
            with self.assertRaisesRegex(
                runner.PlanNotExecutable,
                "joint-limit margin=1.00deg < required 1.50deg",
            ):
                runner.plan_active_direct_route(
                    mock.Mock(),
                    np.zeros(6),
                    q0,
                    np.array([[10.0, 10.0, 10.0]]),
                    np.zeros((0, 3)),
                    args,
                    "near_limit",
                    45.0,
                    75.0,
                )


if __name__ == "__main__":
    unittest.main()
