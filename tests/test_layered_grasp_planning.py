import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

from real_nero_grasp import collision, grasp_planner, motion_planning
from real_nero_grasp.types import (
    InitialSupportRelease,
    MotionCollisionContext,
    PlanNotExecutable,
    PlannedJointStage,
)


def planner_args(**overrides):
    values = dict(
        grasp_motion_path_guard=True,
        allow_hard_grasp_candidate=False,
        motion_neutral_pregrasp_fallback=False,
        motion_up_and_over=False,
        pregrasp_offset=0.10,
        adaptive_pregrasp_offsets=False,
        adaptive_pregrasp_step=0.02,
        adaptive_pregrasp_attempts=1,
        adaptive_pregrasp_min=0.04,
        adaptive_pregrasp_max=0.16,
        adaptive_lift_heights=False,
        lift_orientation_relax=False,
        lift_orientation_relax_degrees=[],
        lift_orientation_relax_axis="tool_approach",
        tool_approach_axis="x",
        fixed_rpy=None,
        use_current_rpy=False,
        grasp_preflight_max_candidates=12,
        grasp_target_valid_plans=3,
        grasp_progressive_beam=False,
        grasp_stage_beam_search=False,
        place_rpy_deg=None,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def annotate(rows, args, plan_hook):
    pre = np.array([0.0, 0.0, 0.30, 0.0, 0.0, 0.0])
    grasp = np.array([0.1, 0.0, 0.30, 0.0, 0.0, 0.0])
    lift = np.array([0.1, 0.0, 0.40, 0.0, 0.0, 0.0])
    return grasp_planner.annotate_grasp_candidate_workflow_preflight(
        rows,
        object(),
        args,
        object(),
        object(),
        np.eye(4),
        np.zeros(6),
        "target",
        filter_grasp_candidates_by_width_hook=lambda value, _args: value,
        filter_grasp_candidates_by_table_guard_hook=lambda value, _args: value,
        defer_candidate_endpoint_reachability_hook=lambda _args: True,
        filter_grasp_candidates_by_reachability_hook=lambda value, strict: value,
        candidate_preflight_width_signs_hook=lambda *_args: (
            [1.0],
            [1.0],
            {1.0: 0.0},
        ),
        grasp_candidate_width_shift_hook=lambda *_args: 0.0,
        grasp_to_tcp_poses_hook=lambda *_args, **_kwargs: (
            pre.copy(),
            grasp.copy(),
            lift.copy(),
        ),
        assert_table_guard_target_tcp_hook=lambda *_args: None,
        plan_up_and_over_grasp_sequence_hook=plan_hook,
    )


class LayeredGraspPlanningTest(unittest.TestCase):
    def test_candidate_budget_stops_after_target_valid_plan_count(self):
        rows = [{"idx": index, "grasp": object()} for index in range(6)]
        calls = []

        def plan(*_args, **_kwargs):
            calls.append(1)
            return object()

        args = planner_args(
            grasp_preflight_max_candidates=4,
            grasp_target_valid_plans=2,
        )
        result = annotate(rows, args, plan)

        self.assertEqual(len(calls), 2)
        self.assertEqual(
            [row["workflow_preflight_ok"] for row in result],
            [True, True, False, False, False, False],
        )
        for row in result[2:]:
            self.assertEqual(
                row["workflow_preflight_reason"],
                "not preflighted due to planning budget",
            )
        self.assertEqual(args._planning_stats["full_preflight_candidate_count"], 2)
        self.assertEqual(args._planning_stats["valid_plan_count"], 2)

    def test_progressive_fast_seed_limit_and_full_fallback(self):
        seen = []

        def plan(*call_args, **_kwargs):
            planning_args = call_args[8]
            seen.append(
                (
                    planning_args.grasp_stage_beam_width,
                    planning_args.ik_solution_topk,
                    getattr(planning_args, "_planning_ik_seed_limit", None),
                )
            )
            if len(seen) == 1:
                raise PlanNotExecutable("fast pregrasp beam exhausted")
            return object()

        args = planner_args(
            grasp_progressive_beam=True,
            grasp_stage_beam_search=True,
            grasp_stage_beam_width=6,
            ik_solution_topk=6,
            grasp_fast_stage_beam_width=2,
            grasp_fast_ik_solution_topk=2,
            grasp_fast_ik_seed_limit=5,
        )
        result = annotate([{"idx": 0, "grasp": object()}], args, plan)

        self.assertTrue(result[0]["workflow_preflight_ok"])
        self.assertEqual(seen, [(2, 2, 5), (6, 6, None)])

    def test_micro_lift_is_exact_world_z_with_same_orientation(self):
        q_grasp = np.arange(7, dtype=float) * 0.01
        grasp_tcp = np.array([0.2, -0.1, 0.15, 0.4, -0.3, 1.2])
        arm = mock.Mock()
        arm.get_tcp2flange_pose.side_effect = lambda pose: np.asarray(pose)
        captured = {}

        def ranked(_arm, flange, _args, _label, **kwargs):
            captured["flange"] = np.asarray(flange, dtype=float).copy()
            captured["seed"] = np.asarray(
                kwargs["current_joints_override"], dtype=float
            ).copy()
            return [
                {
                    "limit_margin": np.deg2rad(8.0),
                    "delta": np.deg2rad(np.array([0, 0, 0, 0, 0, 0, 4.0])),
                    "score": 0.1,
                }
            ]

        with mock.patch.object(
            grasp_planner.ik_solver,
            "_ranked_ik_solution_records_for_flange_pose",
            side_effect=ranked,
        ):
            result = grasp_planner._micro_lift_feasibility(
                arm,
                q_grasp,
                grasp_tcp,
                planner_args(
                    grasp_micro_lift_height=0.008,
                    grasp_micro_lift_ik_seed_limit=5,
                ),
                "candidate",
            )

        expected = grasp_tcp.copy()
        expected[2] += 0.008
        np.testing.assert_allclose(captured["flange"], expected)
        np.testing.assert_allclose(captured["seed"], q_grasp)
        self.assertEqual(result["micro_lift_joint_margin"], 8.0)
        self.assertEqual(result["micro_lift_max_joint_delta"], 4.0)

    def test_payload_support_plane_does_not_use_tcp_clearance(self):
        scene = SimpleNamespace(
            target_base=np.array(
                [[0.0, 0.0, 0.081], [0.01, 0.0, 0.081]], dtype=float
            )
        )
        args = SimpleNamespace(
            table_guard_enable=True,
            table_z=0.080,
            table_tcp_clearance=0.030,
            motion_payload_support_plane_clearance=0.0,
            motion_payload_table_percentile=10.0,
            motion_support_target_bottom_tolerance=0.0,
        )
        clearance = collision.assert_pickup_payload_table_start(
            scene, args, "payload"
        )
        self.assertAlmostEqual(clearance, 0.001)

    def test_place_orientation_can_be_decoupled_from_grasp(self):
        lift = np.array([0.0, 0.0, 0.4, 1.0, 1.1, 1.2])
        result = grasp_planner._place_tcp_from_lift(
            lift,
            [-0.3, -0.2, 0.25],
            planner_args(place_rpy_deg=[10.0, 20.0, 30.0]),
        )
        np.testing.assert_allclose(result[:3], [-0.3, -0.2, 0.25])
        np.testing.assert_allclose(result[3:], np.deg2rad([10.0, 20.0, 30.0]))

    def test_legacy_post_support_lift_interface_calls_existing_planner(self):
        sentinel = object()
        planner = mock.Mock(return_value=sentinel)
        context = object()
        result = grasp_planner.plan_post_support_lift(
            np.zeros(6),
            context,
            "lift",
            planner=planner,
            arm=object(),
            q_previous=np.zeros(7),
            args=object(),
        )
        self.assertIs(result, sentinel)
        planner.assert_called_once()

    def test_segmented_support_release_keeps_xy_and_exact_orientation(self):
        grasp_tcp = np.array([0.2, -0.1, 0.15, 0.4, -0.3, 1.2])
        pre_tcp = grasp_tcp.copy()
        pre_tcp[0] -= 0.08
        lift_tcp = grasp_tcp.copy()
        lift_tcp[2] += 0.10
        scene = SimpleNamespace(target_base=np.zeros((1, 3)))
        support = InitialSupportRelease(
            strict_environment_base=np.zeros((1, 3)),
            support_points_base=np.zeros((1, 3)),
            start_tcp_pose=grasp_tcp.copy(),
            strict_after_rise=0.018,
            max_xy_drift=0.005,
            max_tilt_rad=np.deg2rad(3.0),
            downward_tolerance=0.001,
        )
        calls = []

        def beam(_arm, specs, _q, _args, _label, **_kwargs):
            calls.append(tuple(specs))
            if len(calls) == 1:
                raise PlanNotExecutable("candidate_support_release IK failed")
            stages = []
            for index, spec in enumerate(specs):
                stages.append(
                    PlannedJointStage(
                        label=spec["label"],
                        tcp_pose=np.asarray(spec["tcp_pose"], dtype=float),
                        flange_pose=np.asarray(spec["tcp_pose"], dtype=float),
                        joints=np.full(7, 0.01 * (index + 1)),
                        phase=spec["context"].phase,
                    )
                )
            return ((tuple(stages), 0.0),)

        context = lambda scene, phase, **kwargs: MotionCollisionContext(
            scene=scene,
            phase=phase,
            attached_target_tcp=kwargs.get("attached_target_tcp"),
            initial_support_release=kwargs.get("initial_support_release"),
        )
        args = SimpleNamespace(
            motion_support_release_height=0.020,
            grasp_stage_beam_width=2,
            skip_initial_move=True,
            skip_empty_grasp_return=True,
        )
        result = grasp_planner._plan_direct_grasp_sequence_beam(
            object(),
            np.zeros(6),
            np.zeros(7),
            pre_tcp,
            grasp_tcp,
            lift_tcp,
            None,
            scene,
            args,
            "candidate",
            "direct",
            "direct",
            None,
            {},
            {},
            plan_guarded_route=lambda *_args, **_kwargs: (),
            motion_context=context,
            plan_tcp_motion_stage=lambda *_args, **_kwargs: None,
            plan_tcp_stage_sequence_beam=beam,
            target_points_in_tcp=lambda *_args: np.zeros((1, 3)),
            build_initial_support_release=lambda *_args: support,
            validate_motion_joint_path=lambda *_args, **_kwargs: None,
            pose_rpy_error_norm=lambda *_args: 0.0,
            transform_points=lambda *_args: np.zeros((1, 3)),
            tcp_transform_from_joints=lambda *_args: np.eye(4),
            modeled_tcp_pose_for_joints=lambda *_args: np.zeros(6),
        )

        self.assertEqual(len(calls), 2)
        segmented = [
            spec
            for spec in calls[1]
            if "support_release_segment" in spec["label"]
        ]
        self.assertEqual(len(segmented), 4)
        self.assertEqual(len(result.after_close), 5)
        self.assertEqual(
            [
                spec["context"].require_support_release_latch_at_endpoint
                for spec in segmented
            ],
            [False, False, False, True],
        )
        for spec in segmented:
            np.testing.assert_allclose(spec["tcp_pose"][:2], grasp_tcp[:2])
            np.testing.assert_allclose(spec["tcp_pose"][3:], grasp_tcp[3:])
        self.assertEqual(
            [round((spec["tcp_pose"][2] - grasp_tcp[2]) * 1000) for spec in segmented],
            [5, 10, 15, 20],
        )

    def test_validation_only_plan_stops_after_support_release_and_omits_place(self):
        grasp_tcp = np.array([0.2, -0.1, 0.15, 0.4, -0.3, 1.2])
        pre_tcp = grasp_tcp.copy()
        pre_tcp[0] -= 0.08
        lift_tcp = grasp_tcp.copy()
        lift_tcp[2] += 0.045
        place_tcp = grasp_tcp.copy()
        place_tcp[:3] = [-0.3, -0.2, 0.25]
        scene = SimpleNamespace(target_base=np.zeros((1, 3)))
        support = InitialSupportRelease(
            strict_environment_base=np.zeros((1, 3)),
            support_points_base=np.zeros((1, 3)),
            start_tcp_pose=grasp_tcp.copy(),
            strict_after_rise=0.040,
            max_xy_drift=0.005,
            max_tilt_rad=np.deg2rad(3.0),
            downward_tolerance=0.001,
        )
        seen_specs = []

        def beam(_arm, specs, _q, _args, _label, **_kwargs):
            seen_specs.extend(specs)
            stages = tuple(
                PlannedJointStage(
                    label=spec["label"],
                    tcp_pose=np.asarray(spec["tcp_pose"], dtype=float),
                    flange_pose=np.asarray(spec["tcp_pose"], dtype=float),
                    joints=np.full(7, 0.01 * (index + 1)),
                    phase=spec["context"].phase,
                )
                for index, spec in enumerate(specs)
            )
            return ((stages, 0.0),)

        context = lambda scene, phase, **kwargs: MotionCollisionContext(
            scene=scene,
            phase=phase,
            attached_target_tcp=kwargs.get("attached_target_tcp"),
            initial_support_release=kwargs.get("initial_support_release"),
            min_tcp_z=kwargs.get("min_tcp_z"),
        )
        args = SimpleNamespace(
            motion_support_release_height=0.020,
            grasp_stage_beam_width=2,
            skip_initial_move=True,
            skip_empty_grasp_return=True,
        )
        plan = grasp_planner._plan_direct_grasp_sequence_beam(
            object(),
            np.zeros(6),
            np.zeros(7),
            pre_tcp,
            grasp_tcp,
            lift_tcp,
            place_tcp,
            scene,
            args,
            "validation",
            "direct",
            "direct",
            None,
            {},
            {},
            validation_only=True,
            plan_guarded_route=lambda *_args, **_kwargs: (),
            motion_context=context,
            plan_tcp_motion_stage=lambda *_args, **_kwargs: None,
            plan_tcp_stage_sequence_beam=beam,
            target_points_in_tcp=lambda *_args: np.zeros((1, 3)),
            build_initial_support_release=lambda *_args: support,
            validate_motion_joint_path=lambda *_args, **_kwargs: None,
            pose_rpy_error_norm=lambda *_args: 0.0,
            transform_points=lambda *_args: np.zeros((1, 3)),
            tcp_transform_from_joints=lambda *_args: np.eye(4),
            modeled_tcp_pose_for_joints=lambda *_args: np.zeros(6),
        )
        labels = [spec["label"] for spec in seen_specs]
        self.assertTrue(plan.validation_only)
        self.assertEqual(len(plan.after_close), 1)
        self.assertTrue(any("support_release" in label for label in labels))
        self.assertFalse(any("_lift" in label for label in labels))
        self.assertFalse(any("transport" in label for label in labels))

    def test_known_home_joints_try_direct_return_before_high_route(self):
        args = SimpleNamespace(
            motion_up_and_over=True,
            motion_direct_release_retreat=0.03,
        )
        scene = SimpleNamespace()
        goal_joints = np.full(7, 0.2)

        def stage(_arm, tcp, _q, context, _args, label, fixed_joints=None):
            return PlannedJointStage(
                label=label,
                tcp_pose=np.asarray(tcp, dtype=float),
                flange_pose=np.asarray(tcp, dtype=float),
                joints=np.asarray(fixed_joints, dtype=float),
                phase=context.phase,
            )

        context = lambda scene, phase, **kwargs: MotionCollisionContext(
            scene=scene, phase=phase
        )
        with mock.patch.object(
            motion_planning, "_plan_up_and_over_route"
        ) as guarded_fallback:
            result = motion_planning._plan_guarded_route(
                object(),
                np.zeros(6),
                np.ones(6),
                np.zeros(7),
                scene,
                args,
                "released_return",
                transit_phase="released_free",
                goal_joints=goal_joints,
                motion_context=context,
                plan_tcp_motion_stage=stage,
                validate_joint_path=lambda *_args: None,
            )
        guarded_fallback.assert_not_called()
        np.testing.assert_allclose(result[-1].joints, goal_joints)


if __name__ == "__main__":
    unittest.main()
