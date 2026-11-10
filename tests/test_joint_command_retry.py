import unittest
import time
from types import SimpleNamespace
from unittest import mock

import numpy as np

import real_nero_single_grasp_runner as runner
from real_nero_grasp import arm_execution


def normal_status(
    code=0,
    *,
    ctrl_mode=1,
    mode_feedback=1,
    teach_status=0,
    motion_status=0,
    timestamp=None,
):
    err = SimpleNamespace()
    msg = SimpleNamespace(
        ctrl_mode=ctrl_mode,
        arm_status=code,
        mode_feedback=mode_feedback,
        teach_status=teach_status,
        motion_status=motion_status,
        err_status=err,
    )
    return SimpleNamespace(
        msg=msg,
        timestamp=time.time() if timestamp is None else timestamp,
    )


class JointCommandRetryTest(unittest.TestCase):
    def args(self):
        return SimpleNamespace(
            move_retry_unstarted=True,
            move_retry_delay=0.0,
            move_status_max_age=1.0,
        )

    def test_retries_exact_target_once_when_controller_is_normal(self):
        target = np.linspace(0.01, 0.07, 7)
        arm = mock.Mock()
        arm.get_joint_angles.return_value = np.zeros(7)
        arm.arm.get_arm_status.return_value = normal_status(0)
        arm.arm.get_joints_enable_status_list.return_value = [True] * 7
        stopped = (False, np.zeros(7), 0.0, 10.0)
        started = (True, np.full(7, 0.001), 0.25, 9.0)
        with mock.patch.object(
            runner,
            "_wait_for_joint_motion_start",
            side_effect=[stopped, started],
        ):
            runner._send_joint_target_with_start_retry(
                arm, target, self.args(), "grasp"
            )
        self.assertEqual(arm.move_j.call_count, 2)
        np.testing.assert_allclose(arm.move_j.call_args_list[0].args[0], target)
        np.testing.assert_allclose(arm.move_j.call_args_list[1].args[0], target)

    def test_controller_error_forbids_retry(self):
        target = np.linspace(0.01, 0.07, 7)
        arm = mock.Mock()
        arm.get_joint_angles.return_value = np.zeros(7)
        arm.arm.get_arm_status.return_value = normal_status(7)
        arm.arm.get_joints_enable_status_list.return_value = [True] * 7
        stopped = (False, np.zeros(7), 0.0, 10.0)
        with mock.patch.object(
            runner,
            "_wait_for_joint_motion_start",
            return_value=stopped,
        ):
            with self.assertRaises(runner.MotionNotReached):
                runner._send_joint_target_with_start_retry(
                    arm, target, self.args(), "grasp"
                )
        self.assertEqual(arm.move_j.call_count, 1)

    def test_unsafe_or_stale_controller_state_forbids_retry(self):
        unsafe_statuses = {
            "not_can_control": normal_status(ctrl_mode=0),
            "wrong_motion_mode": normal_status(mode_feedback=2),
            "teaching_enabled": normal_status(teach_status=1),
            "motion_failed": normal_status(motion_status=1),
            "stale_status": normal_status(timestamp=time.time() - 10.0),
            "missing_timestamp": normal_status(timestamp=0.0),
        }
        target = np.linspace(0.01, 0.07, 7)
        stopped = (False, np.zeros(7), 0.0, 10.0)
        for label, status in unsafe_statuses.items():
            with self.subTest(label=label):
                arm = mock.Mock()
                arm.get_joint_angles.return_value = np.zeros(7)
                arm.arm.get_arm_status.return_value = status
                arm.arm.get_joints_enable_status_list.return_value = [True] * 7
                with mock.patch.object(
                    runner,
                    "_wait_for_joint_motion_start",
                    return_value=stopped,
                ):
                    with self.assertRaises(runner.MotionNotReached):
                        runner._send_joint_target_with_start_retry(
                            arm, target, self.args(), "grasp"
                        )
                self.assertEqual(arm.move_j.call_count, 1)

    def test_disabled_joint_forbids_retry(self):
        target = np.linspace(0.01, 0.07, 7)
        arm = mock.Mock()
        arm.get_joint_angles.return_value = np.zeros(7)
        arm.arm.get_arm_status.return_value = normal_status()
        arm.arm.get_joints_enable_status_list.return_value = [True] * 6 + [False]
        stopped = (False, np.zeros(7), 0.0, 10.0)
        with mock.patch.object(
            runner,
            "_wait_for_joint_motion_start",
            return_value=stopped,
        ):
            with self.assertRaises(runner.MotionNotReached):
                runner._send_joint_target_with_start_retry(
                    arm, target, self.args(), "grasp"
                )
        self.assertEqual(arm.move_j.call_count, 1)

    def test_start_requires_progress_toward_new_target(self):
        target = np.zeros(7)
        target[0] = np.deg2rad(10.0)
        residual = np.zeros(7)
        residual[1] = np.deg2rad(1.0)
        directed = residual.copy()
        directed[0] = np.deg2rad(0.30)

        class FakeArm:
            def __init__(self):
                self.samples = [residual, directed]

            def get_joint_angles(self):
                return self.samples.pop(0) if len(self.samples) > 1 else self.samples[0]

        args = SimpleNamespace(
            move_start_timeout=0.05,
            move_start_threshold_deg=0.20,
            move_joint_tol=np.deg2rad(2.0),
            move_poll_interval=0.001,
        )
        started, current, moved_deg, _ = arm_execution._wait_for_joint_motion_start(
            FakeArm(),
            np.zeros(7),
            target,
            args,
            "waypoint2",
            check_table_guard_current_hook=lambda *a, **k: None,
            check_runtime_collision_current_hook=lambda *a, **k: None,
        )
        self.assertTrue(started)
        self.assertGreater(moved_deg, 0.9)
        np.testing.assert_allclose(current, directed)

    def test_motion_away_from_target_is_not_a_start(self):
        target = np.zeros(7)
        target[0] = np.deg2rad(10.0)
        away = np.zeros(7)
        away[0] = np.deg2rad(-1.0)
        arm = mock.Mock()
        arm.get_joint_angles.return_value = away
        args = SimpleNamespace(
            move_start_timeout=0.005,
            move_start_threshold_deg=0.20,
            move_joint_tol=np.deg2rad(2.0),
            move_poll_interval=0.001,
        )
        started, _, moved_deg, _ = arm_execution._wait_for_joint_motion_start(
            arm,
            np.zeros(7),
            target,
            args,
            "waypoint2",
            check_table_guard_current_hook=lambda *a, **k: None,
            check_runtime_collision_current_hook=lambda *a, **k: None,
        )
        self.assertFalse(started)
        self.assertGreater(moved_deg, 0.9)

    def test_precommand_barrier_waits_for_encoder_settle(self):
        moving1 = np.full(7, np.deg2rad(1.0))
        moving2 = np.full(7, np.deg2rad(2.0))
        settled = np.full(7, np.deg2rad(2.01))

        class FakeArm:
            def __init__(self):
                self.samples = [np.zeros(7), moving1, moving2, settled, settled, settled]
                self.calls = 0

            def get_joint_angles(self):
                self.calls += 1
                return self.samples.pop(0) if len(self.samples) > 1 else self.samples[0]

        arm = FakeArm()
        args = SimpleNamespace(
            move_precommand_settle_timeout=0.1,
            move_precommand_settle_samples=2,
            move_precommand_settle_delta_deg=0.05,
            move_precommand_settle_poll=0.001,
        )
        result = arm_execution._wait_for_joint_settle(
            arm,
            args,
            "waypoint2",
            check_table_guard_current_hook=lambda *a, **k: None,
            check_runtime_collision_current_hook=lambda *a, **k: None,
        )
        self.assertGreaterEqual(arm.calls, 5)
        np.testing.assert_allclose(result, settled)


if __name__ == "__main__":
    unittest.main()
