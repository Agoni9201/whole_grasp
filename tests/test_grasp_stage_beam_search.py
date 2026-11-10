import dataclasses
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

import real_nero_single_grasp_runner as runner


class _Arm:
    def get_joint_angles(self):
        return np.zeros(7, dtype=float)

    def get_tcp_pose(self):
        return np.zeros(6, dtype=float)

    def get_tcp2flange_pose(self, tcp_pose):
        return np.asarray(tcp_pose, dtype=float).reshape(6)


def _args():
    return SimpleNamespace(
        motion_mode="ik_joint",
        motion_up_and_over=False,
        motion_direct_release_retreat=0.030,
        min_flange_z=None,
        tcp_offset=np.zeros(6, dtype=float),
        skip_initial_move=True,
        skip_empty_grasp_return=True,
        # Keep both pregrasp branches alive until the grasp edge is checked.
        ik_solution_topk=2,
        grasp_stage_beam_search=True,
        grasp_stage_beam_width=2,
    )


def _scene():
    empty = np.zeros((0, 3), dtype=np.float32)
    return runner.MotionSafetyScene(
        full_scene_base=empty,
        environment_base=empty,
        target_base=empty,
        map_id="beam-unit",
    )


def _q(value):
    joints = np.zeros(7, dtype=float)
    joints[0] = float(value)
    return joints


class GraspStageBeamSearchTest(unittest.TestCase):
    """Regression coverage for cross-stage IK branch search.

    The locally best pregrasp solution is collision-free from the start, but
    it cannot connect to grasp.  The second-ranked pregrasp solution can.  A
    stage-local greedy planner therefore fails this fixture, while a beam
    search must backtrack and return the complete second-branch plan.
    """

    def _plan(self):
        q_pre_top1 = _q(0.10)
        q_pre_top2 = _q(0.20)
        q_grasp = _q(0.30)
        q_lift = _q(0.40)
        validated_edges = []

        def solve_ik(
            _arm,
            flange_pose,
            _args,
            _label,
            current_joints_override=None,
            **_kwargs,
        ):
            del current_joints_override
            stage_id = float(np.asarray(flange_pose, dtype=float)[0])
            if np.isclose(stage_id, 0.10):
                return [q_pre_top1.copy(), q_pre_top2.copy()]
            if np.isclose(stage_id, 0.20):
                return [q_grasp.copy()]
            if np.isclose(stage_id, 0.30):
                return [q_lift.copy()]
            raise AssertionError(f"unexpected IK target: {flange_pose}")

        def validate(q_start, q_goal, _context, _args, label):
            edge = (
                round(float(np.asarray(q_start, dtype=float)[0]), 6),
                round(float(np.asarray(q_goal, dtype=float)[0]), 6),
            )
            validated_edges.append((edge, str(label)))
            if edge == (0.1, 0.3):
                raise runner.UnsafeMotionTarget(
                    "top-1 pregrasp cannot connect to grasp"
                )

        pre_tcp = np.array([0.10, 0.0, 0.30, 0.0, 0.0, 0.0])
        grasp_tcp = np.array([0.20, 0.0, 0.20, 0.0, 0.0, 0.0])
        lift_tcp = np.array([0.30, 0.0, 0.40, 0.0, 0.0, 0.0])

        with mock.patch.object(
            runner, "prepare_motion_safety_scene", return_value=_scene()
        ), mock.patch.object(
            runner, "assert_table_guard_target_tcp"
        ), mock.patch.object(
            runner, "_check_motion_joint_margin"
        ), mock.patch.object(
            runner,
            "_solve_ik_solutions_for_flange_pose",
            side_effect=solve_ik,
        ), mock.patch.object(
            runner, "validate_motion_joint_path", side_effect=validate
        ), mock.patch.object(
            runner,
            "_target_points_in_tcp",
            return_value=np.zeros((0, 3), dtype=np.float32),
        ), mock.patch.object(
            runner, "_build_initial_support_release", return_value=None
        ):
            plan = runner.plan_up_and_over_grasp_sequence(
                _Arm(),
                pre_tcp,
                grasp_tcp,
                lift_tcp,
                None,
                np.zeros((0, 3), dtype=np.float32),
                np.zeros((0, 3), dtype=np.float32),
                np.eye(4, dtype=float),
                _args(),
                "beam",
            )

        return plan, validated_edges

    def test_top1_dead_end_backtracks_to_top2_complete_path(self):
        plan, validated_edges = self._plan()

        np.testing.assert_allclose(plan.before_close[0].joints, _q(0.20))
        np.testing.assert_allclose(plan.before_close[-1].joints, _q(0.30))
        np.testing.assert_allclose(plan.after_close[0].joints, _q(0.40))

        checked = {edge for edge, _label in validated_edges}
        self.assertIn((0.1, 0.3), checked, "Top-1 dead end was not explored")
        self.assertIn((0.2, 0.3), checked, "planner did not backtrack to Top-2")

    def test_result_contains_only_one_frozen_winning_stage_sequence(self):
        plan, _validated_edges = self._plan()

        self.assertIsInstance(plan.before_close, tuple)
        self.assertIsInstance(plan.after_close, tuple)
        self.assertEqual(
            [round(float(stage.joints[0]), 2) for stage in plan.before_close],
            [0.20, 0.30],
        )
        self.assertEqual(
            [round(float(stage.joints[0]), 2) for stage in plan.after_close],
            [0.40],
        )
        self.assertFalse(
            any(
                np.isclose(float(stage.joints[0]), 0.10)
                for stage in plan.before_close + plan.after_close
            ),
            "the rejected branch leaked into the executable plan",
        )
        with self.assertRaises(dataclasses.FrozenInstanceError):
            plan.before_close[0].label = "mutated"

    def test_every_explored_and_selected_edge_is_collision_validated(self):
        plan, validated_edges = self._plan()
        checked = {edge for edge, _label in validated_edges}

        # Both first-stage alternatives, including the branch eventually
        # discarded, must receive the same edge collision validation.
        expected_explored = {
            (0.0, 0.1),
            (0.0, 0.2),
            (0.1, 0.3),
            (0.2, 0.3),
            (0.3, 0.4),
        }
        self.assertTrue(
            expected_explored.issubset(checked),
            f"missing collision checks: {sorted(expected_explored - checked)}",
        )

        # Independently derive the executable forward edges from the returned
        # frozen plan and prove none bypassed collision validation.
        stages = plan.before_close + plan.after_close
        previous = np.asarray(plan.start_joints, dtype=float)
        for stage in stages:
            edge = (
                round(float(previous[0]), 6),
                round(float(np.asarray(stage.joints, dtype=float)[0]), 6),
            )
            self.assertIn(edge, checked, f"selected edge {edge} was unchecked")
            previous = np.asarray(stage.joints, dtype=float)


if __name__ == "__main__":
    unittest.main()
