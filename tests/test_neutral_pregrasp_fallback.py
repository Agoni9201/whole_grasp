import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

import real_nero_single_grasp_runner as runner


class NeutralPregraspFallbackTest(unittest.TestCase):
    def _args(self):
        return SimpleNamespace(
            grasp_motion_path_guard=True,
            motion_neutral_pregrasp_fallback=True,
            motion_up_and_over=False,
            allow_hard_grasp_candidate=False,
        )

    def _rows(self, count=2):
        return [
            {
                "idx": idx,
                "grasp": object(),
                "reachability_ok": True,
                "table_guard_ok": True,
            }
            for idx in range(count)
        ]

    @staticmethod
    def _poses(*_args, **_kwargs):
        return (
            np.array([0.0, 0.0, 0.30, 0.0, 0.0, 0.0]),
            np.array([0.0, 0.0, 0.20, 0.0, 0.0, 0.0]),
            np.array([0.0, 0.0, 0.40, 0.0, 0.0, 0.0]),
        )

    @staticmethod
    def _width_signs(*_args, **_kwargs):
        return (
            [1.0, -1.0],
            [1.0, -1.0],
            {1.0: 0.0, -1.0: np.pi},
        )

    def _run_preflight(self, rows, planner):
        identity_filter = lambda candidates, _args: candidates
        with mock.patch.object(
            runner,
            "filter_grasp_candidates_by_width",
            side_effect=identity_filter,
        ), mock.patch.object(
            runner,
            "filter_grasp_candidates_by_table_guard",
            side_effect=identity_filter,
        ), mock.patch.object(
            runner,
            "_defer_candidate_endpoint_reachability",
            return_value=True,
        ), mock.patch.object(
            runner,
            "_candidate_preflight_width_signs",
            side_effect=self._width_signs,
        ), mock.patch.object(
            runner,
            "grasp_candidate_width_shift",
            return_value=0.0,
        ), mock.patch.object(
            runner,
            "grasp_to_tcp_poses",
            side_effect=self._poses,
        ), mock.patch.object(
            runner,
            "assert_table_guard_target_tcp",
        ), mock.patch.object(
            runner,
            "plan_up_and_over_grasp_sequence",
            side_effect=planner,
        ):
            return runner.annotate_grasp_candidate_workflow_preflight(
                rows,
                mock.Mock(),
                self._args(),
                object(),
                object(),
                np.eye(4),
                np.zeros(6),
                "target",
            )

    def test_any_direct_success_prevents_the_global_neutral_pass(self):
        direct_plan = object()
        calls = []

        def planner(*call_args, **kwargs):
            route = kwargs["pregrasp_route_mode"]
            label = str(call_args[9])
            calls.append((route, label))
            if route == "direct" and "candidate0" in label:
                return direct_plan
            raise runner.UnsafeMotionTarget("direct route blocked")

        annotated = self._run_preflight(self._rows(), planner)

        self.assertTrue(annotated[0]["workflow_preflight_ok"])
        self.assertIs(annotated[0]["_workflow_motion_plan"], direct_plan)
        self.assertEqual(
            annotated[0]["workflow_preflight_route_kind"], "direct"
        )
        self.assertTrue(calls)
        self.assertEqual({route for route, _label in calls}, {"direct"})
        self.assertFalse(annotated[1]["workflow_preflight_ok"])

    def test_neutral_pass_starts_only_after_every_direct_candidate_and_sign(self):
        calls = []

        def planner(*call_args, **kwargs):
            route = kwargs["pregrasp_route_mode"]
            label = str(call_args[9])
            calls.append((route, label))
            if route == "neutral":
                return object()
            raise runner.UnsafeMotionTarget("direct route blocked")

        annotated = self._run_preflight(self._rows(), planner)

        self.assertEqual(
            [route for route, _label in calls],
            ["direct", "direct", "direct", "direct", "neutral", "neutral"],
        )
        self.assertEqual(
            [label for route, label in calls if route == "direct"],
            [
                "target_candidate0",
                "target_candidate0",
                "target_candidate1",
                "target_candidate1",
            ],
        )
        self.assertTrue(all(row["workflow_preflight_ok"] for row in annotated))

    def test_neutral_success_caches_plan_and_route_kind(self):
        neutral_plan = object()
        calls = []

        def planner(*_call_args, **kwargs):
            route = kwargs["pregrasp_route_mode"]
            calls.append(route)
            if route == "neutral":
                return neutral_plan
            raise runner.PlanNotExecutable("direct IK branch exhausted")

        annotated = self._run_preflight(self._rows(count=1), planner)

        row = annotated[0]
        self.assertEqual(calls, ["direct", "direct", "neutral"])
        self.assertTrue(row["workflow_preflight_ok"])
        self.assertEqual(row["workflow_preflight_reason"], "ok")
        self.assertEqual(row["workflow_preflight_route_kind"], "neutral")
        self.assertIs(row["_workflow_motion_plan"], neutral_plan)

    def test_neutral_failure_preserves_direct_and_neutral_reasons(self):
        calls = []

        def planner(*_call_args, **kwargs):
            route = kwargs["pregrasp_route_mode"]
            calls.append(route)
            raise runner.UnsafeMotionTarget(f"{route} route blocked")

        annotated = self._run_preflight(self._rows(count=1), planner)

        row = annotated[0]
        self.assertEqual(
            calls, ["direct", "direct", "neutral", "neutral"]
        )
        self.assertFalse(row["workflow_preflight_ok"])
        self.assertIn("direct:", row["workflow_preflight_reason"])
        self.assertIn("direct route blocked", row["workflow_preflight_reason"])
        self.assertIn("neutral:", row["workflow_preflight_reason"])
        self.assertIn("neutral route blocked", row["workflow_preflight_reason"])
        self.assertIn(
            "direct route blocked", row["workflow_preflight_direct_reason"]
        )
        self.assertIn(
            "neutral route blocked", row["workflow_preflight_neutral_reason"]
        )
        self.assertNotIn("_workflow_motion_plan", row)


if __name__ == "__main__":
    unittest.main()
