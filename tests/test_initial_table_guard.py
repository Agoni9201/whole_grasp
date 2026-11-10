import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

import real_nero_single_grasp_runner as runner


def guard_args():
    return SimpleNamespace(
        initial_joints_deg=[3.546, -0.494, 4.357, 69.346, 2.379, -13.027, 90.0],
        yes_initial=True,
        table_guard_enable=True,
        table_z=0.080,
        table_tcp_clearance=0.0,
        table_tool_bottom_offset=0.020,
        ik_urdf=runner.DEFAULT_IK_URDF,
        ik_base_link="base_link",
        ik_tip_link="link7",
        tcp_offset=np.asarray(runner.DEFAULT_TCP_OFFSET, dtype=float),
        initial_table_path_sample_joint_deg=0.5,
        initial_table_fk_live_z_tol=0.020,
        initial_path_max_joint_delta_deg=95.0,
    )


class FakeArm:
    def __init__(self, joints, tcp_z):
        self._joints = np.asarray(joints, dtype=float).reshape(7)
        self._tcp = np.array([0.0, 0.0, tcp_z, 0.0, 0.0, 0.0])
        self.move_j = mock.Mock(return_value=None)

    def get_joint_angles(self):
        return self._joints.copy()

    def get_tcp_pose(self):
        return self._tcp.copy()


class InitialTableGuardTest(unittest.TestCase):
    def test_below_guard_aborts_before_move_command(self):
        args = guard_args()
        target = np.deg2rad(args.initial_joints_deg)
        arm = FakeArm(target, 0.099)

        with self.assertRaises(runner.UnsafeMotionTarget):
            runner.move_initial(arm, args, require_confirm=False)

        arm.move_j.assert_not_called()

    def test_safe_static_initial_pose_sends_exact_target(self):
        args = guard_args()
        target = np.deg2rad(args.initial_joints_deg)
        profile = runner._joint_path_tcp_z_profile(target, target, args)
        arm = FakeArm(target, profile["start_z"])

        with mock.patch.object(
            runner,
            "wait_until_joints_reached",
            return_value=(True, target, 0.0),
        ):
            reached = runner.move_initial(arm, args, require_confirm=False)

        self.assertTrue(reached)
        arm.move_j.assert_called_once()
        np.testing.assert_allclose(arm.move_j.call_args.args[0], target)

    def test_sampled_path_dip_aborts_before_move_command(self):
        args = guard_args()
        target = np.deg2rad(args.initial_joints_deg)
        arm = FakeArm(target, 0.120)
        unsafe_profile = {
            "samples": 5,
            "start_z": 0.120,
            "target_z": 0.300,
            "min_z": 0.095,
            "min_sample": 2,
            "z_values": np.array([0.120, 0.110, 0.095, 0.200, 0.300]),
            "max_joint_delta_deg": 20.0,
        }

        with mock.patch.object(
            runner,
            "_joint_path_tcp_z_profile",
            return_value=unsafe_profile,
        ), self.assertRaises(runner.UnsafeMotionTarget):
            runner.move_initial(arm, args, require_confirm=False)

        arm.move_j.assert_not_called()

    def test_fk_live_mismatch_aborts_before_move_command(self):
        args = guard_args()
        target = np.deg2rad(args.initial_joints_deg)
        arm = FakeArm(target, 0.120)
        mismatched = {
            "samples": 2,
            "start_z": 0.200,
            "target_z": 0.300,
            "min_z": 0.200,
            "min_sample": 0,
            "z_values": np.array([0.200, 0.300]),
            "max_joint_delta_deg": 20.0,
        }

        with mock.patch.object(
            runner,
            "_joint_path_tcp_z_profile",
            return_value=mismatched,
        ), self.assertRaises(runner.UnsafeMotionTarget):
            runner.move_initial(arm, args, require_confirm=False)

        arm.move_j.assert_not_called()


if __name__ == "__main__":
    unittest.main()
