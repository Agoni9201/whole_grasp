"""Focused tests for measured-torque stopping and reverse recovery."""

import types
import unittest

import numpy as np

from real_nero_grasp.execution import recover_after_torque_stop
from real_nero_grasp.runtime_safety import (
    RuntimeSafetyHooks,
    check_runtime_collision_current,
)
from real_nero_grasp.types import (
    MotionCollisionContext,
    MotionSafetyScene,
    TorqueCollisionStopTriggered,
)


def _context():
    scene = MotionSafetyScene(
        full_scene_base=np.zeros((0, 3)),
        environment_base=np.zeros((0, 3)),
        target_base=np.zeros((0, 3)),
        map_id="torque-test-map",
    )
    return MotionCollisionContext(scene=scene, phase="free_open")


class _TorqueArm:
    def __init__(self):
        self.q = np.zeros(7)
        self.samples = [np.zeros(7), np.full(7, 9.0), np.full(7, 9.0)]
        self.terminate_calls = 0

    def get_joint_angles(self):
        return self.q.copy()

    def get_joint_torques(self):
        return self.samples.pop(0)

    def terminate_motion(self, fallback_joints=None):
        self.terminate_calls += 1
        return {"confirmed": True}


class RuntimeTorqueGuardTests(unittest.TestCase):
    def _args(self):
        return types.SimpleNamespace(
            motion_runtime_collision_guard=True,
            motion_runtime_torque_guard=True,
            motion_runtime_torque_delta=8.0,
            motion_runtime_torque_confirm_samples=2,
            _runtime_motion_collision_context=_context(),
            _runtime_motion_stage_label="stage",
            _runtime_motion_stage_start_joints=np.zeros(7),
            _runtime_motion_stage_target_joints=np.ones(7),
            save_dir="",
            _last_capture_out_dir="",
        )

    @staticmethod
    def _hooks(events):
        return RuntimeSafetyHooks(
            as_pose_list=lambda value: value,
            tcp_transform_from_joints=lambda joints, args: np.eye(4),
            motion_configuration_metrics=lambda joints, context, args, **_: {
                "margin": 1.0,
                "nearest_clearance": 1.0,
                "required_clearance": 0.01,
                "primitive": "none",
                "source": "environment",
            },
            persist_runtime_collision_diagnostic=lambda *values: events.append(values),
            safe_stop=lambda *values: events.append(("safe_stop", values)),
        )

    def test_sustained_torque_excursion_stops_normally(self):
        arm = _TorqueArm()
        args = self._args()
        events = []
        hooks = self._hooks(events)
        self.assertGreater(
            check_runtime_collision_current(arm, args, "stage", hooks=hooks)["margin"],
            0.0,
        )
        self.assertGreater(
            check_runtime_collision_current(arm, args, "stage", hooks=hooks)["margin"],
            0.0,
        )
        with self.assertRaises(TorqueCollisionStopTriggered) as caught:
            check_runtime_collision_current(arm, args, "stage", hooks=hooks)
        self.assertIn("joint1", str(caught.exception))
        self.assertEqual(arm.terminate_calls, 1)
        self.assertFalse(any(event[0] == "safe_stop" for event in events))

    def test_verified_torque_retreat_reverses_only_the_stage_edge(self):
        context = _context()
        stop = TorqueCollisionStopTriggered(
            "contact",
            retreat_joints=np.zeros(7),
            context=context,
            stage_label="stage",
        )

        class Driver:
            def __init__(self):
                self.speeds = []

            def set_speed_percent(self, speed):
                self.speeds.append(float(speed))

        class Arm:
            def __init__(self):
                self.arm = Driver()
                self.q = np.full(7, 0.2)
                self.moves = []
                self.recovery_enabled = False
                self.finished = False

            def get_joint_angles(self):
                return self.q.copy()

            def allow_controlled_recovery(self):
                self.recovery_enabled = True

            def finish_controlled_recovery(self):
                self.finished = True

            def move_j(self, target):
                self.moves.append(np.asarray(target).copy())
                self.q = np.asarray(target).copy()

            def terminate_motion(self, fallback_joints=None):
                raise AssertionError("successful retreat must not terminate again")

        arm = Arm()
        args = types.SimpleNamespace(
            nero_speed_percent=5.0,
            motion_runtime_torque_retract_speed_percent=2.0,
        )
        validated = []

        ok = recover_after_torque_stop(
            arm,
            stop,
            args,
            validate_motion_joint_path_hook=lambda start, target, *rest: validated.append(
                (np.asarray(start).copy(), np.asarray(target).copy())
            ),
            wait_until_joints_reached_hook=lambda *values: (True, np.zeros(7), 0.0),
        )
        self.assertTrue(ok)
        self.assertTrue(arm.recovery_enabled)
        self.assertTrue(arm.finished)
        np.testing.assert_allclose(arm.moves, [np.zeros(7)])
        self.assertEqual(arm.arm.speeds, [2.0, 5.0])
        self.assertEqual(len(validated), 1)


if __name__ == "__main__":
    unittest.main()
