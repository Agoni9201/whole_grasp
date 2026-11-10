import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

import real_nero_single_grasp_runner as runner


class _Arm:
    def get_joint_angles(self):
        return np.zeros(7, dtype=float)


class _QueueIKSolver:
    num_joints = 7

    def __init__(self, solutions):
        self._solutions = [
            None if solution is None else np.asarray(solution, dtype=float)
            for solution in solutions
        ]
        self.calls = 0

    def solve_ik(self, **_kwargs):
        solution = self._solutions[self.calls]
        self.calls += 1
        return None if solution is None else solution.copy()

    def get_joint_limits(self):
        return -np.pi * np.ones(7), np.pi * np.ones(7)


def _args(**overrides):
    values = {
        "ik_max_joint_delta": None,
        "ik_joint7_weight": 0.35,
        "ik_solution_topk": 6,
        "ik_solution_dedup_deg": 0.5,
        "initial_joints_deg": [0.0] * 7,
        "ik_all_joint_seed_count": 24,
        "ik_seed_limit_margin_deg": 2.0,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _q(joint1):
    result = np.zeros(7, dtype=float)
    result[0] = float(joint1)
    return result


class IKTopKSolutionsTest(unittest.TestCase):
    def test_default_seed_set_deterministically_spans_every_joint(self):
        solver = _QueueIKSolver([])
        solver.get_joint_limits = mock.Mock(
            return_value=(-np.ones(7), np.ones(7))
        )
        args = _args(ik_all_joint_seed_count=8)

        first = runner._joint_seed_candidates(
            np.zeros(7, dtype=float), args, solver=solver
        )
        second = runner._joint_seed_candidates(
            np.zeros(7, dtype=float), args, solver=solver
        )

        self.assertEqual(len(first), len(second))
        for left, right in zip(first, second):
            np.testing.assert_allclose(left, right)
        # Global seeds are interleaved at positions 2, 4, 6, ... so even a
        # bounded side-view prefix explores complete seven-joint postures.
        global_prefix = np.vstack([first[index] for index in (2, 4, 6, 8)])
        self.assertTrue(np.all(np.ptp(global_prefix, axis=0) > 1e-3))
        margin = np.deg2rad(args.ik_seed_limit_margin_deg)
        self.assertTrue(np.all(global_prefix >= -1.0 + margin - 1e-12))
        self.assertTrue(np.all(global_prefix <= 1.0 - margin + 1e-12))

    def _solve(self, solver, args=None, **kwargs):
        seeds = [np.full(7, index, dtype=float) for index in range(len(solver._solutions))]
        with mock.patch.object(runner, "_get_ik_solver", return_value=solver), mock.patch.object(
            runner,
            "_joint_seed_candidates",
            return_value=seeds,
        ):
            return runner._solve_ik_solutions_for_flange_pose(
                _Arm(),
                np.zeros(6, dtype=float),
                _args() if args is None else args,
                "topk_test",
                quiet=True,
                **kwargs,
            )

    def test_multi_seed_solutions_are_score_sorted_and_deduplicated(self):
        solver = _QueueIKSolver(
            [
                _q(0.40),
                _q(0.10),
                _q(0.104),  # Same branch as 0.10 within the default 0.5 deg.
                _q(0.20),
            ]
        )

        solutions = self._solve(solver)

        self.assertEqual(solver.calls, 4)
        self.assertEqual(len(solutions), 3)
        np.testing.assert_allclose(solutions[0], _q(0.10))
        np.testing.assert_allclose(solutions[1], _q(0.20))
        np.testing.assert_allclose(solutions[2], _q(0.40))

    def test_top_k_can_be_configured_and_defaults_to_six(self):
        raw = [_q(value) for value in (0.60, 0.10, 0.50, 0.20, 0.40, 0.30, 0.70)]
        default_solver = _QueueIKSolver(raw)
        default_solutions = self._solve(default_solver)
        self.assertEqual(len(default_solutions), 6)
        np.testing.assert_allclose(
            [solution[0] for solution in default_solutions],
            [0.10, 0.20, 0.30, 0.40, 0.50, 0.60],
        )

        explicit_solver = _QueueIKSolver(raw)
        explicit_solutions = self._solve(explicit_solver, top_k=2)
        self.assertEqual(len(explicit_solutions), 2)
        np.testing.assert_allclose(
            [solution[0] for solution in explicit_solutions],
            [0.10, 0.20],
        )

    def test_legacy_single_solution_api_still_returns_best_ndarray(self):
        solver = _QueueIKSolver([_q(0.35), _q(0.08), _q(0.22)])
        seeds = [np.full(7, index, dtype=float) for index in range(3)]
        with mock.patch.object(runner, "_get_ik_solver", return_value=solver), mock.patch.object(
            runner,
            "_joint_seed_candidates",
            return_value=seeds,
        ):
            result = runner._solve_ik_for_flange_pose(
                _Arm(),
                np.zeros(6, dtype=float),
                _args(),
                "legacy_test",
                quiet=True,
            )

        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (7,))
        np.testing.assert_allclose(result, _q(0.08))

    def test_endpoint_margin_check_stops_at_first_viable_solution(self):
        near_limit = _q(np.pi - np.deg2rad(0.5))
        viable = _q(0.25)
        solver = _QueueIKSolver([near_limit, viable, _q(0.10)])
        seeds = [np.full(7, index, dtype=float) for index in range(3)]
        with mock.patch.object(
            runner, "_get_ik_solver", return_value=solver
        ), mock.patch.object(
            runner, "_joint_seed_candidates", return_value=seeds
        ):
            records = runner._ranked_ik_solution_records_for_flange_pose(
                _Arm(),
                np.zeros(6, dtype=float),
                _args(),
                "endpoint_test",
                quiet=True,
                early_accept_margin_rad=np.deg2rad(1.5),
            )

        self.assertEqual(solver.calls, 2)
        self.assertEqual(len(records), 1)
        np.testing.assert_allclose(records[0]["joints"], viable)


if __name__ == "__main__":
    unittest.main()
