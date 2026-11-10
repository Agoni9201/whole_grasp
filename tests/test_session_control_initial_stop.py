import unittest
from types import SimpleNamespace

import numpy as np

from real_nero_grasp.session_control import (
    SessionControlHooks,
    prepare_initial_pose_for_capture,
)
from real_nero_grasp.types import SafetyStopTriggered


class InitialPoseTimeoutStopTest(unittest.TestCase):
    def _args(self):
        return SimpleNamespace(
            skip_initial_move=False,
            gripper_open_width=0.085,
            gripper_force=0.5,
            gripper_settle=0.0,
            require_gripper_open=True,
        )

    def _hooks(self, events, terminate):
        return SessionControlHooks(
            print_status=lambda *_args: None,
            preflight_initial_table_path=lambda *_args: None,
            wait_until_joints_reached=lambda *_args: (False, None, np.inf),
            wrap_angle_error=lambda a, b: np.asarray(a) - np.asarray(b),
            safe_stop=lambda *_args: events.append("electronic_stop"),
            terminate_motion=terminate,
            move_initial=lambda *_args: False,
        )

    def test_confirmed_ordinary_stop_does_not_request_electronic_stop(self):
        events = []
        hooks = self._hooks(
            events,
            lambda *_args: {
                "confirmed": True,
                "method": "track_terminate_speed_zero_hold",
            },
        )

        with self.assertRaisesRegex(
            SafetyStopTriggered,
            "ordinary trajectory termination was confirmed",
        ):
            prepare_initial_pose_for_capture(
                SimpleNamespace(), self._args(), hooks=hooks
            )

        self.assertEqual(events, [])

    def test_unconfirmed_ordinary_stop_still_does_not_request_estop(self):
        events = []
        hooks = self._hooks(
            events,
            lambda *_args: {"confirmed": False},
        )

        with self.assertRaisesRegex(
            SafetyStopTriggered,
            "electronic emergency stop is disabled",
        ):
            arm = SimpleNamespace()
            prepare_initial_pose_for_capture(arm, self._args(), hooks=hooks)

        self.assertEqual(events, [])
        self.assertTrue(arm._suppress_program_exit_estop)


if __name__ == "__main__":
    unittest.main()
