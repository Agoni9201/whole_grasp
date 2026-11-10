import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

import real_nero_single_grasp_runner as runner


class _Arm:
    def get_tcp2flange_pose(self, tcp_pose):
        return np.asarray(tcp_pose, dtype=float).reshape(6)


def _args():
    return SimpleNamespace(
        min_flange_z=None,
        tcp_offset=np.zeros(6, dtype=float),
    )


def _context():
    return SimpleNamespace(
        phase="free_open",
        min_tcp_z=None,
        runtime_min_tcp_z_slack=0.0,
    )


class MotionStageIKFallbackTest(unittest.TestCase):
    def test_top1_collision_uses_top2_solution(self):
        q_previous = np.zeros(7, dtype=float)
        q_top1 = np.full(7, 0.10, dtype=float)
        q_top2 = np.full(7, 0.20, dtype=float)

        def validate(_start, goal, _context, _args, label):
            if np.allclose(goal, q_top1):
                raise runner.UnsafeMotionTarget("top-1 path collides")
            self.assertTrue(np.allclose(goal, q_top2))
            self.assertEqual(label, "stage_ik_option2")

        with mock.patch.object(
            runner,
            "assert_table_guard_target_tcp",
        ), mock.patch.object(
            runner,
            "_solve_ik_solutions_for_flange_pose",
            return_value=[q_top1, q_top2],
        ) as solve, mock.patch.object(
            runner,
            "_check_motion_joint_margin",
        ) as margin, mock.patch.object(
            runner,
            "validate_motion_joint_path",
            side_effect=validate,
        ) as path:
            stage = runner._plan_tcp_motion_stage(
                _Arm(),
                np.zeros(6, dtype=float),
                q_previous,
                _context(),
                _args(),
                "stage",
            )

        np.testing.assert_allclose(stage.joints, q_top2)
        self.assertEqual(solve.call_count, 1)
        self.assertEqual(margin.call_count, 2)
        self.assertEqual(path.call_count, 2)

    def test_all_topk_paths_rejected_fails_closed(self):
        q_previous = np.zeros(7, dtype=float)
        q_top1 = np.full(7, 0.10, dtype=float)
        q_top2 = np.full(7, 0.20, dtype=float)

        with mock.patch.object(
            runner,
            "assert_table_guard_target_tcp",
        ), mock.patch.object(
            runner,
            "_solve_ik_solutions_for_flange_pose",
            return_value=[q_top1, q_top2],
        ), mock.patch.object(
            runner,
            "_check_motion_joint_margin",
        ), mock.patch.object(
            runner,
            "validate_motion_joint_path",
            side_effect=[
                runner.UnsafeMotionTarget("top-1 collision"),
                runner.PlanNotExecutable("top-2 joint sweep"),
            ],
        ):
            with self.assertRaisesRegex(
                runner.PlanNotExecutable,
                r"none of 2 retained IK solution.*rank=1.*rank=2",
            ):
                runner._plan_tcp_motion_stage(
                    _Arm(),
                    np.zeros(6, dtype=float),
                    q_previous,
                    _context(),
                    _args(),
                    "stage",
                )

    def test_sample_zero_payload_table_failure_skips_remaining_topk(self):
        q_previous = np.zeros(7, dtype=float)
        q_top1 = np.full(7, 0.10, dtype=float)
        q_top2 = np.full(7, 0.20, dtype=float)

        with mock.patch.object(
            runner,
            "assert_table_guard_target_tcp",
        ), mock.patch.object(
            runner,
            "_solve_ik_solutions_for_flange_pose",
            return_value=[q_top1, q_top2],
        ), mock.patch.object(
            runner,
            "_check_motion_joint_margin",
        ), mock.patch.object(
            runner,
            "validate_motion_joint_path",
            side_effect=runner.UnsafeMotionTarget(
                "lift attached_target gets within -50.3mm of table_plane "
                "at sample 0"
            ),
        ) as path:
            with self.assertRaisesRegex(
                runner.UnsafeMotionTarget,
                "attached_target.*table_plane.*sample 0",
            ):
                runner._plan_tcp_motion_stage(
                    _Arm(),
                    np.zeros(6, dtype=float),
                    q_previous,
                    _context(),
                    _args(),
                    "lift_stage",
                )

        self.assertEqual(path.call_count, 1)

    def test_fixed_joint_stage_does_not_search_other_ik_solutions(self):
        q_previous = np.zeros(7, dtype=float)
        fixed_joints = np.full(7, 0.15, dtype=float)

        with mock.patch.object(
            runner,
            "assert_table_guard_target_tcp",
        ), mock.patch.object(
            runner,
            "_solve_ik_solutions_for_flange_pose",
        ) as solve, mock.patch.object(
            runner,
            "_get_ik_solver",
            return_value=object(),
        ), mock.patch.object(
            runner,
            "_clip_numerical_joint_limit_overshoot",
            return_value=fixed_joints.copy(),
        ), mock.patch.object(
            runner,
            "_urdf_chain_geometry",
            return_value=([], np.eye(4, dtype=float)),
        ), mock.patch.object(
            runner,
            "_check_motion_joint_margin",
        ) as margin, mock.patch.object(
            runner,
            "validate_motion_joint_path",
        ) as path:
            stage = runner._plan_tcp_motion_stage(
                _Arm(),
                np.zeros(6, dtype=float),
                q_previous,
                _context(),
                _args(),
                "fixed_stage",
                fixed_joints=fixed_joints,
            )

        solve.assert_not_called()
        np.testing.assert_allclose(stage.joints, fixed_joints)
        margin.assert_called_once()
        path.assert_called_once()
        np.testing.assert_allclose(path.call_args.args[1], fixed_joints)
        self.assertEqual(path.call_args.args[4], "fixed_stage")


if __name__ == "__main__":
    unittest.main()
