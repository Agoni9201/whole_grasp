import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

import real_nero_single_grasp_runner as runner


class FakeGrasp:
    def __init__(self, grasp_z):
        self.grasp_z = float(grasp_z)
        self.width = 0.070
        self.rotation_matrix = np.eye(3, dtype=float)
        self.translation = np.zeros(3, dtype=float)


class FakeArm:
    def get_joint_angles(self):
        return np.zeros(7, dtype=float)

    def get_tcp2flange_pose(self, pose):
        return np.asarray(pose, dtype=float)


def args_for_table_guard():
    return SimpleNamespace(
        graspnet_approach_axis="x",
        graspnet_width_axis="y",
        tool_approach_axis="x",
        tcp_gripper_width_axis="z",
        topdown_grasp=True,
        use_current_rpy=False,
        fixed_rpy=None,
        motion_mode="ik_joint",
        grasp_motion_path_guard=False,
        max_candidate_ik_joint_delta=np.deg2rad(95.0),
        max_candidate_pre_pos_delta=0.25,
        max_candidate_pre_rpy_delta_deg=60.0,
        table_guard_enable=True,
        table_z=0.080,
        table_tcp_clearance=0.040,
        table_tool_bottom_offset=0.020,
        max_grasp_width=0.090,
        allow_oversize_grasp=False,
    )


class GraspTableGuardFilterTest(unittest.TestCase):
    def test_unsafe_top_candidate_is_removed_before_selection(self):
        args = args_for_table_guard()
        arm = FakeArm()
        rows = [
            {"idx": 0, "grasp": FakeGrasp(0.1345)},
            {"idx": 1, "grasp": FakeGrasp(0.1458)},
        ]

        def fake_poses(grasp, *_args, **_kwargs):
            grasp_pose = np.array([0.0, 0.0, grasp.grasp_z, 0.0, 0.0, 0.0])
            pre_pose = grasp_pose.copy()
            pre_pose[2] += 0.10
            lift_pose = grasp_pose.copy()
            lift_pose[2] += 0.08
            return pre_pose, grasp_pose, lift_pose

        with mock.patch.object(
            runner, "grasp_to_tcp_poses", side_effect=fake_poses
        ), mock.patch.object(
            runner,
            "_solve_ik_for_flange_pose",
            return_value=np.zeros(7, dtype=float),
        ):
            annotated = runner.annotate_grasp_candidate_reachability(
                rows,
                np.eye(4),
                np.array([0.0, 0.0, 0.30, 0.0, 0.0, 0.0]),
                args,
                arm=arm,
            )

        self.assertFalse(annotated[0]["table_guard_ok"])
        self.assertFalse(annotated[0]["reachability_ok"])
        self.assertTrue(annotated[1]["table_guard_ok"])
        safe = runner.filter_grasp_candidates_by_table_guard(annotated, args)
        self.assertEqual([row["idx"] for row in safe], [1])

        partially_unannotated = [annotated[1], {"idx": 2, "grasp": FakeGrasp(0.160)}]
        safe = runner.filter_grasp_candidates_by_table_guard(
            partially_unannotated,
            args,
        )
        self.assertEqual([row["idx"] for row in safe], [1])

        _, chosen = runner.choose_default_grasp_candidate(annotated, [], args)
        self.assertEqual(chosen["idx"], 1)


if __name__ == "__main__":
    unittest.main()
