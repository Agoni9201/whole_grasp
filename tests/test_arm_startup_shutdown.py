import unittest
import time
from unittest import mock
from types import SimpleNamespace

import numpy as np

from real_nero_grasp.arm import RealNeroGraspArm


class ArmStartupShutdownTests(unittest.TestCase):
    @staticmethod
    def _bare_arm(speed_percent=5):
        arm = object.__new__(RealNeroGraspArm)
        arm.arm = mock.Mock()
        arm.enable_timeout_s = 1.0
        arm.speed_percent = speed_percent
        arm._safety_stop_latched = False
        arm._safety_stop_reason = None
        arm._safety_stop_request_confirmed = False
        arm._controlled_stop_request_confirmed = False
        def fresh_sample(_monitor):
            stamp = time.time()
            return {"q": np.zeros(7), "position_timestamps": np.full(7, stamp),
                    "motor_timestamps": np.full(7, stamp)}
        arm._read_stop_feedback = fresh_sample
        return arm

    def test_startup_clears_stale_target_before_enable_and_holds_feedback(self):
        arm = self._bare_arm(speed_percent=5)
        measured = np.deg2rad(np.asarray([11, -7, 3, 80, 9, -12, 61], dtype=float))
        events = []

        arm.arm.connect.side_effect = lambda: events.append("connect")
        arm.arm.enable.side_effect = lambda: events.append("enable") or True
        arm.arm.set_speed_percent.side_effect = (
            lambda value: events.append(f"speed:{value}")
        )
        arm.arm.get_joint_angles.side_effect = (
            lambda: events.append("feedback") or measured.copy()
        )
        arm.arm.move_j.side_effect = lambda joints: events.append(
            ("hold", np.asarray(joints, dtype=float))
        )

        def terminate(frames=3):
            events.append(f"terminate:{frames}")
            return frames

        with mock.patch.object(
            arm, "_send_ordinary_trajectory_terminate", side_effect=terminate
        ), mock.patch("real_nero_grasp.arm.time.sleep", return_value=None):
            entered = arm.__enter__()

        self.assertIs(entered, arm)
        self.assertLess(events.index("terminate:3"), events.index("enable"))
        self.assertEqual(events.count("terminate:3"), 2)
        self.assertLess(events.index("speed:0"), events.index("speed:5"))
        hold_events = [event for event in events if isinstance(event, tuple)]
        self.assertEqual(len(hold_events), 1)
        np.testing.assert_allclose(hold_events[0][1], measured)
        np.testing.assert_allclose(arm._startup_hold_joints, measured)

    def test_startup_refuses_enable_if_pre_enable_terminate_fails(self):
        arm = self._bare_arm()
        with mock.patch.object(
            arm,
            "_send_ordinary_trajectory_terminate",
            side_effect=RuntimeError("CAN terminate unavailable"),
        ):
            with self.assertRaisesRegex(RuntimeError, "CAN terminate unavailable"):
                arm.__enter__()

        arm.arm.enable.assert_not_called()
        arm.arm.set_speed_percent.assert_not_called()
        arm.arm.disconnect.assert_called_once()

    def test_startup_interrupt_checks_hold_before_disconnect(self):
        arm = self._bare_arm()
        events = []
        arm.arm.enable.side_effect = KeyboardInterrupt()
        arm.stop_for_program_exit = mock.Mock(side_effect=lambda reason: events.append('stop') or True)
        arm.arm.disconnect.side_effect = lambda: events.append('disconnect')
        with mock.patch.object(arm, '_send_ordinary_trajectory_terminate'):
            with self.assertRaises(KeyboardInterrupt):
                arm.__enter__()
        self.assertEqual(events, ['stop', 'disconnect'])
        arm.arm.disable.assert_not_called()

    def test_normal_context_exit_terminates_before_disconnect(self):
        arm = self._bare_arm()
        events = []
        arm.stop_for_program_exit = mock.Mock(
            side_effect=lambda reason: events.append(("stop", reason)) or True
        )
        arm.arm.disconnect.side_effect = lambda: events.append(("disconnect", None))

        arm.__exit__(None, None, None)

        self.assertEqual(events[0][0], "stop")
        self.assertEqual(events[1][0], "disconnect")
        arm.stop_for_program_exit.assert_called_once_with("program workflow finished")

    def test_exit_does_not_overwrite_confirmed_emergency_stop(self):
        arm = self._bare_arm()
        arm._safety_stop_request_confirmed = True
        arm.stop_for_program_exit = mock.Mock()

        arm.__exit__(None, None, None)

        arm.stop_for_program_exit.assert_not_called()
        arm.arm.disconnect.assert_called_once()

    def test_ctrl_c_during_idle_planning_sends_no_controller_command(self):
        arm = self._bare_arm()
        arm.arm.get_joint_angles.return_value = np.zeros(7)
        arm.arm.get_arm_status.return_value = SimpleNamespace(
            msg=SimpleNamespace(motion_status=0), timestamp=time.time(),
        )
        arm.terminate_motion = mock.Mock()

        with mock.patch("real_nero_grasp.arm.time.sleep", return_value=None):
            stopped = arm.stop_for_keyboard_interrupt()

        self.assertTrue(stopped)
        self.assertTrue(arm._controlled_stop_request_confirmed)
        arm.terminate_motion.assert_not_called()
        arm.arm._send_msg.assert_not_called()
        arm.arm.set_speed_percent.assert_not_called()
        arm.arm.move_j.assert_not_called()

    def test_ctrl_c_while_controller_is_moving_uses_trajectory_stop(self):
        arm = self._bare_arm()
        arm.arm.get_joint_angles.return_value = np.zeros(7)
        arm.arm.get_arm_status.return_value = SimpleNamespace(
            msg=SimpleNamespace(motion_status=1), timestamp=time.time(),
        )
        arm.terminate_motion = mock.Mock(return_value={"confirmed": True})

        stopped = arm.stop_for_keyboard_interrupt()

        self.assertTrue(stopped)
        arm.terminate_motion.assert_called_once()

    def test_initial_timeout_exit_suppresses_estop_escalation(self):
        arm = self._bare_arm()
        arm._suppress_program_exit_estop = True
        arm.arm.get_joint_angles.return_value = np.zeros(7)
        arm._confirm_idle_for_program_exit = mock.Mock(return_value=None)
        arm.terminate_motion = mock.Mock(
            side_effect=RuntimeError("settling not confirmed")
        )
        arm.stop = mock.Mock()

        stopped = arm.stop_for_program_exit("initial timeout")

        self.assertFalse(stopped)
        arm.stop.assert_not_called()


if __name__ == "__main__":
    unittest.main()
