import unittest
from types import SimpleNamespace
from unittest import mock

import real_nero_single_grasp_runner as runner


class GripperContactConfirmationTest(unittest.TestCase):
    def test_force_confirmation_keeps_loaded_command_instead_of_unloading(self):
        arm = runner.RealNeroGraspArm.__new__(runner.RealNeroGraspArm)
        arm.gripper_max_width = 0.085
        arm.gripper_max_force = 0.30
        arm.gripper = mock.Mock()
        statuses = [
            SimpleNamespace(value=0.0848, force=0.05, mode="width"),
            SimpleNamespace(value=0.0822, force=0.30, mode="width"),
            SimpleNamespace(value=0.0821, force=0.29, mode="width"),
            SimpleNamespace(value=0.0820, force=0.29, mode="width"),
        ]
        arm.print_gripper_status = mock.Mock(side_effect=statuses)

        with mock.patch.object(runner.time, "sleep"):
            result = arm.close_gripper_slow(
                open_width=0.085,
                close_width=0.0,
                force=0.30,
                step_m=0.002,
                interval_s=0.0,
                settle_s=0.1,
                verbose=True,
                stop_force=0.28,
                contact_confirm_samples=2,
                contact_confirm_interval_s=0.0,
                empty_width_threshold=0.005,
            )

        self.assertTrue(result.contact_detected)
        self.assertEqual(result.stop_reason, "force_contact")
        commands = arm.gripper.move_gripper_m.call_args_list
        self.assertEqual(len(commands), 2)
        self.assertAlmostEqual(commands[0].kwargs["value"], 0.085, places=6)
        self.assertAlmostEqual(commands[1].kwargs["value"], 0.083, places=6)
        # In particular, no third command resets the target to the measured
        # contact width (0.0822 m), which previously unloaded the force.


if __name__ == "__main__":
    unittest.main()
