import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

from real_nero_grasp.post_grasp_rrt import rrt_connect, plan_post_grasp_joint_hold
from real_nero_grasp import grasp_planner
from real_nero_grasp.types import MotionSafetyScene, PlannedJointStage, PlanNotExecutable, UnsafeMotionTarget


def _stage(label, value, phase="carrying"):
    tcp = np.zeros(6)
    tcp[2] = 0.1 + 0.01 * value
    return PlannedJointStage(
        label=label,
        tcp_pose=tcp,
        flange_pose=tcp.copy(),
        joints=np.full(7, float(value)),
        phase=phase,
    )


class PostGraspRrtTests(unittest.TestCase):
    def test_exact_joint_hold_is_finely_checked_and_never_replaces_the_goal(self):
        from real_nero_grasp.motion_planning import _motion_context
        empty = np.zeros((0, 3), dtype=np.float32)
        context = _motion_context(MotionSafetyScene(empty, empty, empty), "carrying",
                                  attached_target_tcp=empty, closed_gripper_width=.05)
        goal = [-.002, 0, .002, 100.6, 0, 0, 0]
        args = SimpleNamespace(post_grasp_hold_joints_deg=goal, motion_path_sample_joint_deg=1.,
                               post_grasp_execution_waypoint_deg=20.)
        solver = SimpleNamespace(get_joint_limits=lambda: (np.full(7, -2.), np.full(7, 2.)))
        for failure in (None, "goal_collision", "fine_collision", "joint_limit"):
            with self.subTest(failure=failure):
                if failure == "joint_limit":
                    args.post_grasp_hold_joints_deg = [0, 0, 0, 200, 0, 0, 0]
                else:
                    args.post_grasp_hold_joints_deg = goal
                calls = []
                def validate(a, b, ctx, local_args, label):
                    calls.append((ctx, local_args.motion_path_sample_joint_deg))
                    if failure == "goal_collision" or (
                        failure == "fine_collision" and local_args is args
                    ):
                        raise UnsafeMotionTarget("test obstacle")
                    return dict(margin=.01)
                with mock.patch("real_nero_grasp.ik_solver._get_ik_solver", return_value=solver), \
                     mock.patch("real_nero_grasp.collision.validate_motion_joint_path", side_effect=validate), \
                     mock.patch("real_nero_grasp.collision._urdf_chain_geometry", return_value=(empty, np.eye(4))), \
                     mock.patch("real_nero_grasp.motion_planning._modeled_tcp_pose_for_joints",
                                side_effect=lambda q, _: np.array([0, 0, .2 + .1*q[3], 0, 0, 0])):
                    if failure is not None:
                        with self.assertRaises((PlanNotExecutable, UnsafeMotionTarget)):
                            plan_post_grasp_joint_hold(np.zeros(7), context, args, "hold")
                    else:
                        stages = plan_post_grasp_joint_hold(np.zeros(7), context, args, "hold")
                        np.testing.assert_allclose(stages[-1].joints, np.deg2rad(goal), atol=1e-12)
                        self.assertTrue(all(stage.phase == "carrying" for stage in stages))
                        self.assertTrue(all(ctx.attached_target_tcp is empty for ctx, _ in calls))
                        self.assertEqual(sum(step == 1. for _, step in calls), len(stages))
                        q = np.zeros(7)
                        for stage in stages:
                            self.assertLessEqual(np.rad2deg(abs(stage.joints - q)).max(), 20. + 1e-9)
                            q = stage.joints

    def test_opt_in_beam_stops_exact_specs_after_support_then_appends_rrt(self):
        class Arm:
            def get_tcp2flange_pose(self, pose):
                return np.asarray(pose, dtype=float)

        args = SimpleNamespace(
            motion_closed_gripper_empty_width=0.0,
            post_grasp_free_route=True,
            motion_support_release_height=0.020,
            grasp_stage_beam_width=2,
            skip_initial_move=True,
            skip_empty_grasp_return=True,
            motion_direct_release_retreat=0.020,
            tcp_offset=np.zeros(6),
        )
        empty = np.zeros((0, 3), dtype=np.float32)
        scene = MotionSafetyScene(empty, empty, empty)
        seen_spec_labels = []

        def beam(_arm, specs, _q, _args, _label, **_kwargs):
            seen_spec_labels.extend(spec["label"] for spec in specs)
            stages = tuple(
                _stage(spec["label"], index + 1, spec["context"].phase)
                for index, spec in enumerate(specs)
            )
            return [(stages, 0.0)]

        fake_context = lambda _scene, phase, **kwargs: SimpleNamespace(
            phase=phase,
            min_tcp_z=kwargs.get("min_tcp_z"),
            runtime_min_tcp_z_slack=0.0,
            require_support_release_latch_at_endpoint=True,
        )
        rrt_stages = (_stage("free_lift", 4), _stage("free_place", 5))
        with mock.patch(
            "real_nero_grasp.post_grasp_rrt.plan_post_grasp_free_route",
            return_value=(rrt_stages, np.array([-0.35, -0.2, 0.3, 0, 0, 0])),
        ) as free_route:
            plan = grasp_planner._plan_direct_grasp_sequence_beam(
                Arm(),
                np.zeros(6),
                np.zeros(7),
                np.array([0, 0, 0.2, 0, 0, 0], dtype=float),
                np.array([0, 0, 0.1, 0, 0, 0], dtype=float),
                np.array([0, 0, 0.3, 0, 0, 0], dtype=float),
                np.array([-0.35, -0.2, 0.3, 0, 0, 0], dtype=float),
                scene,
                args,
                "candidate",
                "direct",
                "direct",
                None,
                {},
                {},
                False,
                plan_guarded_route=lambda *_a, **_k: (),
                motion_context=fake_context,
                plan_tcp_motion_stage=lambda _arm, tcp, _q, context, _args, label: PlannedJointStage(
                    label, np.asarray(tcp), np.asarray(tcp), np.zeros(7), context.phase
                ),
                plan_tcp_stage_sequence_beam=beam,
                target_points_in_tcp=lambda *_a: empty,
                build_initial_support_release=lambda *_a: None,
                validate_motion_joint_path=lambda *_a, **_k: {},
                pose_rpy_error_norm=lambda *_a: 0.0,
                transform_points=lambda _T, points: points,
                tcp_transform_from_joints=lambda *_a: np.eye(4),
                modeled_tcp_pose_for_joints=lambda *_a: np.array([0, 0, 0.4, 0, 0, 0]),
            )

        self.assertEqual(len(seen_spec_labels), 3)
        self.assertTrue(any("strict_20mm_release" in item for item in seen_spec_labels))
        self.assertFalse(any("_lift" in item for item in seen_spec_labels))
        self.assertFalse(any("_transport" in item for item in seen_spec_labels))
        self.assertEqual(plan.after_close[-2:], rrt_stages)
        free_route.assert_called_once()

    def test_direct_edge_stays_two_points(self):
        start = np.zeros(7)
        goal = np.full(7, 0.1)
        path = rrt_connect(
            start,
            goal,
            np.full(7, -1.0),
            np.full(7, 1.0),
            lambda _a, _b: True,
            step_rad=0.2,
            timeout_s=0.2,
            max_nodes=100,
            goal_bias=0.2,
            seed=3,
        )
        self.assertEqual(len(path), 2)
        np.testing.assert_allclose(path[0], start)
        np.testing.assert_allclose(path[-1], goal)

    def test_routes_around_joint_space_wall_with_gap(self):
        start = np.zeros(7)
        start[0] = -0.8
        goal = np.zeros(7)
        goal[0] = 0.8

        def point_valid(q):
            # A wall in J1 with a gap reached by moving J2 upward.
            return not (abs(float(q[0])) < 0.12 and float(q[1]) < 0.35)

        def edge_valid(a, b):
            for alpha in np.linspace(0.0, 1.0, 12):
                if not point_valid(np.asarray(a) + alpha * (np.asarray(b) - a)):
                    return False
            return True

        path = rrt_connect(
            start,
            goal,
            np.full(7, -1.0),
            np.full(7, 1.0),
            edge_valid,
            step_rad=0.20,
            timeout_s=1.0,
            max_nodes=1200,
            goal_bias=0.2,
            seed=11,
        )
        self.assertIsNotNone(path)
        np.testing.assert_allclose(path[0], start)
        np.testing.assert_allclose(path[-1], goal)
        self.assertTrue(all(edge_valid(a, b) for a, b in zip(path, path[1:])))
        self.assertGreater(max(float(q[1]) for q in path), 0.35)

    def test_returns_none_when_start_and_goal_are_disconnected(self):
        start = np.zeros(7)
        start[0] = -0.8
        goal = np.zeros(7)
        goal[0] = 0.8

        def edge_valid(a, b):
            # No edge may cross J1=0.
            return float(a[0]) * float(b[0]) > 0.0

        path = rrt_connect(
            start,
            goal,
            np.full(7, -1.0),
            np.full(7, 1.0),
            edge_valid,
            step_rad=0.15,
            timeout_s=0.05,
            max_nodes=60,
            goal_bias=0.2,
            seed=7,
        )
        self.assertIsNone(path)


if __name__ == "__main__":
    unittest.main()
