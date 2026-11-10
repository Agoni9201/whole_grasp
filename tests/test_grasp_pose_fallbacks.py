import unittest
from types import SimpleNamespace

import numpy as np

from real_nero_grasp.grasp_geometry import grasp_planning_pose_variants
from real_nero_grasp.grasp_planner import (
    _plan_grasp_pose_fallbacks,
    annotate_grasp_candidate_workflow_preflight,
)
from real_nero_grasp.types import PlanNotExecutable


def _args(**overrides):
    values = dict(
        grasp_motion_path_guard=True,
        allow_hard_grasp_candidate=False,
        motion_neutral_pregrasp_fallback=False,
        motion_up_and_over=False,
        pregrasp_offset=0.10,
        adaptive_pregrasp_offsets=True,
        adaptive_pregrasp_step=0.02,
        adaptive_pregrasp_attempts=4,
        adaptive_pregrasp_min=0.04,
        adaptive_pregrasp_max=0.16,
        lift_orientation_relax=True,
        lift_orientation_relax_degrees=[3.0, 6.0],
        lift_orientation_relax_axis="tool_approach",
        tool_approach_axis="x",
        fixed_rpy=None,
        use_current_rpy=False,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


class GraspPoseFallbackTest(unittest.TestCase):
    def setUp(self):
        self.pre = np.array([0.0, 0.0, 0.20, 0.0, 0.0, 0.0])
        self.grasp = np.array([0.10, 0.0, 0.20, 0.0, 0.0, 0.0])
        self.lift = np.array([0.10, 0.0, 0.40, 0.0, 0.0, 0.0])

    def test_variants_try_all_original_orientation_pregrasp_distances_first(self):
        variants = grasp_planning_pose_variants(
            self.pre, self.grasp, self.lift, _args()
        )

        self.assertEqual(len(variants), 20)
        self.assertEqual(
            [round(item["pregrasp_offset"], 3) for item in variants[:4]],
            [0.10, 0.08, 0.12, 0.14],
        )
        self.assertEqual(
            [item["lift_relax_deg"] for item in variants[:4]],
            [0.0, 0.0, 0.0, 0.0],
        )
        self.assertEqual(variants[4]["lift_relax_deg"], 3.0)
        self.assertEqual(variants[8]["lift_relax_deg"], -3.0)
        self.assertEqual(variants[12]["lift_relax_deg"], 6.0)
        self.assertEqual(variants[16]["lift_relax_deg"], -6.0)
        for item in variants:
            np.testing.assert_allclose(item["grasp_tcp"], self.grasp)
            np.testing.assert_allclose(item["lift_tcp"][:3], self.lift[:3])

    def test_adaptive_lift_height_descends_without_multiplying_pregrasp(self):
        variants = grasp_planning_pose_variants(
            self.pre,
            self.grasp,
            self.lift,
            _args(
                adaptive_pregrasp_offsets=False,
                adaptive_lift_heights=True,
                adaptive_lift_step=0.02,
                adaptive_lift_min_height=0.14,
                adaptive_lift_attempts=4,
                lift_orientation_relax=False,
            ),
        )

        self.assertEqual(len(variants), 4)
        self.assertEqual(
            [round(item["lift_height"], 3) for item in variants],
            [0.20, 0.18, 0.16, 0.14],
        )
        for item in variants:
            np.testing.assert_allclose(item["pre_tcp"], self.pre)
            np.testing.assert_allclose(item["grasp_tcp"], self.grasp)

    def test_global_lift_precheck_skips_full_plan_for_impossible_heights(self):
        args = _args(
            adaptive_pregrasp_offsets=False,
            adaptive_lift_heights=True,
            adaptive_lift_step=0.02,
            adaptive_lift_min_height=0.14,
            adaptive_lift_attempts=4,
            lift_orientation_relax=False,
        )
        full_plan_heights = []
        checked_heights = []
        winning_plan = object()

        def precheck(variant):
            height = round(float(variant["lift_height"]), 2)
            checked_heights.append(height)
            if height in (0.18, 0.16):
                raise PlanNotExecutable("lift endpoint global precheck failed")

        def plan_attempt(variant, _transport_mode, _attempt_index):
            height = round(float(variant["lift_height"]), 2)
            full_plan_heights.append(height)
            if height == 0.20:
                raise PlanNotExecutable("candidate_lift unavailable")
            return winning_plan

        selected, failures = _plan_grasp_pose_fallbacks(
            self.pre,
            self.grasp,
            self.lift,
            args,
            plan_attempt,
            lift_endpoint_precheck=precheck,
        )

        self.assertIs(selected["plan"], winning_plan)
        self.assertEqual(checked_heights, [0.20, 0.18, 0.16, 0.14])
        self.assertEqual(full_plan_heights, [0.20, 0.14])
        self.assertEqual(len(failures), 3)

    def test_sample_zero_payload_table_failure_skips_all_lift_fallbacks(self):
        args = _args(
            adaptive_pregrasp_offsets=False,
            adaptive_lift_heights=True,
            adaptive_lift_step=0.02,
            adaptive_lift_min_height=0.08,
            adaptive_lift_attempts=7,
            lift_orientation_relax=True,
        )
        attempted_heights = []

        def plan_attempt(variant, _transport_mode, _attempt_index):
            attempted_heights.append(float(variant["lift_height"]))
            raise PlanNotExecutable(
                "candidate_lift attached_target gets within -10.0mm of "
                "table_plane at sample 0"
            )

        selected, failures = _plan_grasp_pose_fallbacks(
            self.pre,
            self.grasp,
            self.lift,
            args,
            plan_attempt,
        )

        self.assertIsNone(selected)
        self.assertEqual(attempted_heights, [0.20])
        self.assertEqual(len(failures), 1)

    def test_validation_mode_accepts_safe_release_after_full_lift_failure(self):
        args = _args(
            adaptive_pregrasp_offsets=False,
            adaptive_lift_heights=False,
            lift_orientation_relax=False,
            grasp_validation_mode=True,
            grasp_validation_lift_height=0.045,
            grasp_lift_region_max_attempts=4,
        )
        attempts = []
        validation_plan = object()

        def plan_attempt(variant, _transport_mode, _attempt_index):
            attempts.append(bool(variant.get("validation_only", False)))
            if not bool(variant.get("validation_only", False)):
                raise PlanNotExecutable("candidate_lift IK failed")
            return validation_plan

        selected, failures = _plan_grasp_pose_fallbacks(
            self.pre,
            self.grasp,
            self.lift,
            args,
            plan_attempt,
        )

        self.assertEqual(attempts, [False, True])
        self.assertEqual(len(failures), 1)
        self.assertIs(selected["plan"], validation_plan)
        self.assertTrue(selected["variant"]["validation_only"])
        self.assertEqual(selected["transport_route_mode"], "validation_only")

    def test_lift_feasible_region_adds_bounded_xy_endpoints(self):
        variants = grasp_planning_pose_variants(
            self.pre,
            self.grasp,
            self.lift,
            _args(
                adaptive_pregrasp_offsets=False,
                adaptive_lift_heights=True,
                adaptive_lift_step=0.02,
                adaptive_lift_min_height=0.04,
                adaptive_lift_attempts=10,
                lift_orientation_relax=False,
                grasp_lift_feasible_region=True,
                grasp_lift_region_xy=0.02,
                grasp_validation_lift_height=0.045,
            ),
        )
        region = [item for item in variants if item["lift_region_fallback"]]
        self.assertEqual(len(region), 4)
        self.assertEqual(
            {tuple(np.round(item["lift_xy_offset"], 3)) for item in region},
            {(0.02, 0.0), (-0.02, 0.0), (0.0, 0.02), (0.0, -0.02)},
        )
        self.assertTrue(all(item["lift_height"] >= 0.045 for item in region))

    def test_transport_failure_uses_bounded_place_region_without_global_search(self):
        args = _args(
            adaptive_pregrasp_offsets=False,
            adaptive_lift_heights=False,
            lift_orientation_relax=False,
            transport_place_relax=True,
            transport_place_xy_relax=0.02,
            transport_place_z_relax=0.02,
            transport_place_tilt_relax_deg=8.0,
            transport_place_relax_max_attempts=9,
            motion_neutral_transport_fallback=True,
        )
        attempts = []
        winning_plan = object()

        def plan_attempt(variant, transport_mode, _attempt_index):
            kind = variant.get("transport_place_relax_kind", "exact")
            attempts.append((kind, transport_mode))
            if kind == "x_plus" and transport_mode == "direct":
                return winning_plan
            raise PlanNotExecutable("candidate_transport unavailable")

        selected, failures = _plan_grasp_pose_fallbacks(
            self.pre,
            self.grasp,
            self.lift,
            args,
            plan_attempt,
        )

        self.assertIs(selected["plan"], winning_plan)
        self.assertEqual(selected["transport_route_mode"], "direct_relaxed_place")
        self.assertEqual(
            attempts[:3],
            [("exact", "direct"), ("z_plus", "direct"), ("x_plus", "direct")],
        )
        self.assertFalse(any(mode == "neutral" for _kind, mode in attempts))
        self.assertEqual(len(failures), 2)

    def _annotate(self, args, fake_plan):
        row = {"idx": 2, "grasp": object()}
        return annotate_grasp_candidate_workflow_preflight(
            [row],
            object(),
            args,
            object(),
            object(),
            np.eye(4),
            np.zeros(6),
            "target",
            filter_grasp_candidates_by_width_hook=lambda rows, _args: rows,
            filter_grasp_candidates_by_table_guard_hook=lambda rows, _args: rows,
            defer_candidate_endpoint_reachability_hook=lambda _args: True,
            filter_grasp_candidates_by_reachability_hook=lambda rows, strict: rows,
            candidate_preflight_width_signs_hook=lambda *_args: (
                [1.0],
                [1.0],
                {1.0: 0.0},
            ),
            grasp_candidate_width_shift_hook=lambda *_args: 0.0,
            grasp_to_tcp_poses_hook=lambda *_args, **_kwargs: (
                self.pre.copy(),
                self.grasp.copy(),
                self.lift.copy(),
            ),
            assert_table_guard_target_tcp_hook=lambda *_args: None,
            plan_up_and_over_grasp_sequence_hook=fake_plan,
        )[0]

    def test_winning_pregrasp_distance_and_plan_are_cached_together(self):
        calls = []
        winning_plan = object()

        def fake_plan(_arm, pre, _grasp, _lift, *_args, **_kwargs):
            offset = float(np.linalg.norm(self.grasp[:3] - np.asarray(pre)[:3]))
            calls.append(offset)
            if not np.isclose(offset, 0.12):
                raise PlanNotExecutable("pregrasp branch unavailable")
            return winning_plan

        row = self._annotate(_args(), fake_plan)

        self.assertEqual(len(calls), 3)
        self.assertTrue(row["workflow_preflight_ok"])
        self.assertAlmostEqual(row["workflow_preflight_pregrasp_offset"], 0.12)
        self.assertEqual(row["workflow_preflight_lift_relax_deg"], 0.0)
        self.assertIs(row["_workflow_motion_plan"], winning_plan)
        np.testing.assert_allclose(row["_workflow_pre_tcp"][:3], [-0.02, 0.0, 0.20])

    def test_progressive_beam_uses_fast_complete_route_when_it_passes(self):
        calls = []
        winning_plan = object()

        def fake_plan(
            _arm,
            _pre,
            _grasp,
            _lift,
            _place,
            _target,
            _scene,
            _transform,
            planning_args,
            _label,
            **_kwargs,
        ):
            calls.append(
                (
                    planning_args.grasp_stage_beam_width,
                    planning_args.ik_solution_topk,
                )
            )
            return winning_plan

        row = self._annotate(
            _args(
                grasp_progressive_beam=True,
                grasp_stage_beam_search=True,
                grasp_stage_beam_width=6,
                ik_solution_topk=6,
                grasp_fast_stage_beam_width=2,
                grasp_fast_ik_solution_topk=2,
            ),
            fake_plan,
        )

        self.assertEqual(calls, [(2, 2)])
        self.assertIs(row["_workflow_motion_plan"], winning_plan)

    def test_progressive_beam_expands_to_full_route_after_fast_failure(self):
        calls = []
        winning_plan = object()

        def fake_plan(
            _arm,
            _pre,
            _grasp,
            _lift,
            _place,
            _target,
            _scene,
            _transform,
            planning_args,
            _label,
            **_kwargs,
        ):
            key = (
                planning_args.grasp_stage_beam_width,
                planning_args.ik_solution_topk,
            )
            calls.append(key)
            if key == (2, 2):
                raise PlanNotExecutable("fast beam exhausted")
            return winning_plan

        row = self._annotate(
            _args(
                grasp_progressive_beam=True,
                grasp_stage_beam_search=True,
                grasp_stage_beam_width=6,
                ik_solution_topk=6,
                grasp_fast_stage_beam_width=2,
                grasp_fast_ik_solution_topk=2,
            ),
            fake_plan,
        )

        self.assertEqual(calls, [(2, 2), (6, 6)])
        self.assertIs(row["_workflow_motion_plan"], winning_plan)

    def test_progressive_beam_can_disable_full_expansion(self):
        calls = []

        def fake_plan(
            _arm, _pre, _grasp, _lift, _place, _target, _scene,
            _transform, planning_args, _label, **_kwargs,
        ):
            calls.append(
                (
                    planning_args.grasp_stage_beam_width,
                    planning_args.ik_solution_topk,
                )
            )
            raise PlanNotExecutable("fast beam exhausted")

        row = self._annotate(
            _args(
                grasp_progressive_beam=True,
                grasp_progressive_full_fallback=False,
                grasp_stage_beam_search=True,
                grasp_stage_beam_width=6,
                ik_solution_topk=6,
                grasp_fast_stage_beam_width=2,
                grasp_fast_ik_solution_topk=2,
            ),
            fake_plan,
        )

        self.assertEqual(calls, [(2, 2)])
        self.assertFalse(row["workflow_preflight_ok"])

    def test_validation_first_never_plans_full_lift(self):
        calls = []
        winning_plan = object()

        def fake_plan(*_args, **kwargs):
            calls.append(bool(kwargs.get("validation_only", False)))
            return winning_plan

        row = self._annotate(
            _args(
                grasp_validation_mode=True,
                grasp_validation_first=True,
            ),
            fake_plan,
        )

        self.assertEqual(calls, [True])
        self.assertTrue(row["workflow_preflight_ok"])
        self.assertTrue(row["workflow_preflight_validation_only"])
        self.assertIs(row["_workflow_motion_plan"], winning_plan)

    def test_progressive_lift_expansion_reuses_validated_upstream_cache(self):
        calls = []
        winning_plan = object()

        def fake_plan(
            _arm,
            _pre,
            _grasp,
            _lift,
            _place,
            _target,
            _scene,
            _transform,
            planning_args,
            _label,
            **kwargs,
        ):
            cache = kwargs["planning_stage_cache"]
            key = (
                planning_args.grasp_stage_beam_width,
                planning_args.ik_solution_topk,
            )
            calls.append((key, "validated_pregrasp" in cache))
            if key == (2, 2):
                cache["validated_pregrasp"] = object()
                raise PlanNotExecutable(
                    "candidate_fast beam exhausted at stage 3/4 "
                    "(candidate_fast_lift): lift IK failed"
                )
            return winning_plan

        row = self._annotate(
            _args(
                grasp_progressive_beam=True,
                grasp_stage_beam_search=True,
                grasp_stage_beam_width=6,
                ik_solution_topk=6,
                grasp_fast_stage_beam_width=2,
                grasp_fast_ik_solution_topk=2,
            ),
            fake_plan,
        )

        self.assertEqual(calls, [((2, 2), False), ((6, 6), True)])
        self.assertIs(row["_workflow_motion_plan"], winning_plan)

    def test_progressive_beam_does_not_expand_invariant_pickup_failure(self):
        calls = []

        def fake_plan(
            _arm,
            _pre,
            _grasp,
            _lift,
            _place,
            _target,
            _scene,
            _transform,
            planning_args,
            _label,
            **_kwargs,
        ):
            calls.append(
                (
                    planning_args.grasp_stage_beam_width,
                    planning_args.ik_solution_topk,
                )
            )
            raise PlanNotExecutable(
                "candidate_lift attached_target gets within -10.0mm of "
                "table_plane at sample 0"
            )

        row = self._annotate(
            _args(
                grasp_progressive_beam=True,
                grasp_stage_beam_search=True,
                grasp_stage_beam_width=6,
                ik_solution_topk=6,
                grasp_fast_stage_beam_width=2,
                grasp_fast_ik_solution_topk=2,
            ),
            fake_plan,
        )

        self.assertEqual(calls, [(2, 2)])
        self.assertFalse(row["workflow_preflight_ok"])

    def test_progressive_beam_does_not_expand_shared_lift_start_contact(self):
        calls = []

        def fake_plan(
            _arm,
            _pre,
            _grasp,
            _lift,
            _place,
            _target,
            _scene,
            _transform,
            planning_args,
            _label,
            **_kwargs,
        ):
            calls.append(
                (
                    planning_args.grasp_stage_beam_width,
                    planning_args.ik_solution_topk,
                )
            )
            raise PlanNotExecutable(
                "candidate_fast beam exhausted at stage 3/4 "
                "(candidate_fast_lift): "
                "parent=1:UnsafeMotionTarget:candidate_fast_lift "
                "gripper_tool gets within -27.2mm of environment at "
                "sample=1 | parent=2:UnsafeMotionTarget:candidate_fast_lift "
                "gripper_tool gets within -27.5mm of environment at "
                "sample=2"
            )

        row = self._annotate(
            _args(
                grasp_progressive_beam=True,
                grasp_stage_beam_search=True,
                grasp_stage_beam_width=6,
                ik_solution_topk=6,
                grasp_fast_stage_beam_width=2,
                grasp_fast_ik_solution_topk=2,
            ),
            fake_plan,
        )

        self.assertEqual(calls, [(2, 2)])
        self.assertFalse(row["workflow_preflight_ok"])

    def test_lift_relaxation_runs_only_after_original_orientation_options_fail(self):
        seen = []
        winning_plan = object()

        def fake_plan(_arm, _pre, _grasp, lift, *_args, **_kwargs):
            roll_deg = float(np.rad2deg(np.asarray(lift)[3]))
            seen.append(roll_deg)
            if not np.isclose(roll_deg, 3.0):
                raise PlanNotExecutable("lift endpoint unavailable")
            return winning_plan

        row = self._annotate(
            _args(adaptive_pregrasp_attempts=2), fake_plan
        )

        # The lift failure must not trigger unrelated pregrasp distances.
        self.assertEqual([round(value) for value in seen], [0, 3])
        self.assertEqual(row["workflow_preflight_lift_relax_deg"], 3.0)
        self.assertEqual(row["workflow_preflight_lift_relax_axis"], "x")
        self.assertIs(row["_workflow_motion_plan"], winning_plan)

    def test_grasp_failure_does_not_expand_pregrasp_or_lift_variants(self):
        calls = []

        def fake_plan(*_args, **_kwargs):
            calls.append(1)
            raise PlanNotExecutable("target_candidate2_grasp collision")

        row = self._annotate(_args(), fake_plan)

        self.assertEqual(len(calls), 1)
        self.assertFalse(row["workflow_preflight_ok"])
        self.assertIn("'grasp': 1", row["workflow_preflight_reason"])

    def test_transport_failure_only_retries_neutral_transport(self):
        modes = []
        winning_plan = object()

        def fake_plan(*_args, **kwargs):
            mode = kwargs.get("transport_route_mode", "direct")
            modes.append(mode)
            if mode == "direct":
                raise PlanNotExecutable("target_candidate2_transport blocked")
            return winning_plan

        row = self._annotate(
            _args(motion_neutral_transport_fallback=True), fake_plan
        )

        self.assertEqual(modes, ["direct", "neutral"])
        self.assertTrue(row["workflow_preflight_ok"])
        self.assertEqual(
            row["workflow_preflight_transport_route"], "neutral"
        )
        self.assertIs(row["_workflow_motion_plan"], winning_plan)


if __name__ == "__main__":
    unittest.main()
