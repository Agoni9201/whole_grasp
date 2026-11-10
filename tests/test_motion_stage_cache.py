import unittest
from types import SimpleNamespace

import numpy as np

from real_nero_grasp.motion_planning import _plan_tcp_stage_sequence_beam
from real_nero_grasp.types import (
    MotionCollisionContext,
    MotionSafetyScene,
    PlanNotExecutable,
    PlannedJointStage,
)


class MotionStageCacheTest(unittest.TestCase):
    def test_invariant_support_release_collision_skips_remaining_parents(self):
        empty = np.zeros((0, 3), dtype=np.float32)
        scene = MotionSafetyScene(empty, empty, empty, map_id="support-cache")
        context = MotionCollisionContext(scene=scene, phase="support_release")
        args = SimpleNamespace(
            grasp_stage_beam_width=2,
            ik_solution_dedup_deg=0.5,
            ik_joint7_weight=0.35,
            ik_lift_joint7_weight=0.0,
        )
        calls = []

        def plan_options(
            _arm, tcp_pose, _previous, motion_context, _args, label,
            fixed_joints=None,
        ):
            del fixed_joints
            if float(np.asarray(tcp_pose)[0]) < 1.5:
                return tuple(
                    PlannedJointStage(
                        label=label,
                        tcp_pose=np.asarray(tcp_pose, dtype=float),
                        flange_pose=np.asarray(tcp_pose, dtype=float),
                        joints=np.full(7, value, dtype=float),
                        phase=motion_context.phase,
                    )
                    for value in (0.1, 0.2)
                )
            calls.append(label)
            raise PlanNotExecutable(
                "support_release none of 2 IK solutions passed: "
                "rank=1: gripper_tool gets within -45.2mm of environment "
                "at sample 1 | rank=2: gripper_tool gets within -45.2mm "
                "of environment at sample 4"
            )

        with self.assertRaisesRegex(PlanNotExecutable, "gripper_tool"):
            _plan_tcp_stage_sequence_beam(
                object(),
                (
                    {
                        "tcp_pose": np.array([1.0, 0, 0, 0, 0, 0]),
                        "context": context,
                        "label": "grasp",
                    },
                    {
                        "tcp_pose": np.array([2.0, 0, 0, 0, 0, 0]),
                        "context": context,
                        "label": "support_release",
                    },
                ),
                np.zeros(7),
                args,
                "support-invariant",
                plan_tcp_motion_stage_options=plan_options,
            )

        self.assertEqual(calls, ["support_release_beam_parent1"])

    def test_invariant_payload_start_failure_skips_remaining_beam_parents(self):
        empty = np.zeros((0, 3), dtype=np.float32)
        scene = MotionSafetyScene(empty, empty, empty, map_id="cache-test")
        context = MotionCollisionContext(scene=scene, phase="pickup_lift")
        args = SimpleNamespace(
            grasp_stage_beam_width=2,
            ik_solution_dedup_deg=0.5,
            ik_joint7_weight=0.35,
            ik_lift_joint7_weight=0.0,
        )
        q0 = np.zeros(7, dtype=float)
        tcp_pre = np.array([1.0, 0.0, 0.3, 0.0, 0.0, 0.0])
        tcp_lift = np.array([2.0, 0.0, 0.4, 0.0, 0.0, 0.0])
        lift_calls = []

        def plan_options(
            _arm,
            tcp_pose,
            _previous,
            motion_context,
            _args,
            label,
            fixed_joints=None,
        ):
            del fixed_joints
            stage_id = int(round(float(np.asarray(tcp_pose)[0])))
            if stage_id == 1:
                return tuple(
                    PlannedJointStage(
                        label=label,
                        tcp_pose=np.asarray(tcp_pose, dtype=float),
                        flange_pose=np.asarray(tcp_pose, dtype=float),
                        joints=np.full(7, value, dtype=float),
                        phase=motion_context.phase,
                    )
                    for value in (0.1, 0.2)
                )
            lift_calls.append(label)
            raise PlanNotExecutable(
                "lift attached_target gets within -50.3mm of table_plane "
                "at sample 0"
            )

        with self.assertRaisesRegex(
            PlanNotExecutable,
            "attached_target.*table_plane.*sample 0",
        ):
            _plan_tcp_stage_sequence_beam(
                object(),
                (
                    {"tcp_pose": tcp_pre, "context": context, "label": "pre"},
                    {
                        "tcp_pose": tcp_lift,
                        "context": context,
                        "label": "lift",
                    },
                ),
                q0,
                args,
                "invariant",
                plan_tcp_motion_stage_options=plan_options,
            )

        self.assertEqual(lift_calls, ["lift_beam_parent1"])

    def test_identical_prefix_stage_reuses_collision_checked_options(self):
        empty = np.zeros((0, 3), dtype=np.float32)
        scene = MotionSafetyScene(empty, empty, empty, map_id="cache-test")
        context = MotionCollisionContext(scene=scene, phase="free_open")
        args = SimpleNamespace(
            grasp_stage_beam_width=2,
            ik_solution_dedup_deg=0.5,
            ik_joint7_weight=0.35,
            ik_lift_joint7_weight=0.0,
        )
        q0 = np.zeros(7, dtype=float)
        q1 = np.full(7, 0.1, dtype=float)
        tcp = np.array([0.1, 0.0, 0.3, 0.0, 0.0, 0.0])
        calls = []

        def plan_options(
            _arm,
            tcp_pose,
            _previous,
            motion_context,
            _args,
            label,
            fixed_joints=None,
        ):
            del fixed_joints
            calls.append(label)
            return (
                PlannedJointStage(
                    label=label,
                    tcp_pose=np.asarray(tcp_pose, dtype=float),
                    flange_pose=np.asarray(tcp_pose, dtype=float),
                    joints=q1.copy(),
                    phase=motion_context.phase,
                ),
            )

        cache = {}
        specs_a = ({"tcp_pose": tcp, "context": context, "label": "pre_a"},)
        specs_b = ({"tcp_pose": tcp, "context": context, "label": "pre_b"},)

        first = _plan_tcp_stage_sequence_beam(
            object(),
            specs_a,
            q0,
            args,
            "first",
            stage_option_cache=cache,
            plan_tcp_motion_stage_options=plan_options,
        )
        second = _plan_tcp_stage_sequence_beam(
            object(),
            specs_b,
            q0,
            args,
            "second",
            stage_option_cache=cache,
            plan_tcp_motion_stage_options=plan_options,
        )

        self.assertEqual(calls, ["pre_a"])
        self.assertEqual(first[0][0][0].label, "pre_a")
        self.assertEqual(second[0][0][0].label, "pre_b")

    def test_prefix_cache_survives_a_later_stage_failure(self):
        empty = np.zeros((0, 3), dtype=np.float32)
        scene = MotionSafetyScene(empty, empty, empty, map_id="cache-test")
        context = MotionCollisionContext(scene=scene, phase="free_open")
        args = SimpleNamespace(
            grasp_stage_beam_width=1,
            ik_solution_dedup_deg=0.5,
            ik_joint7_weight=0.35,
            ik_lift_joint7_weight=0.0,
        )
        q0 = np.zeros(7, dtype=float)
        calls = []

        def plan_options(
            _arm,
            tcp_pose,
            _previous,
            motion_context,
            _args,
            label,
            fixed_joints=None,
        ):
            del fixed_joints
            stage_id = int(round(float(np.asarray(tcp_pose)[0])))
            calls.append(stage_id)
            if stage_id == 3:
                raise PlanNotExecutable("first lift fails")
            joints = np.full(7, 0.1 * stage_id, dtype=float)
            return (
                PlannedJointStage(
                    label=label,
                    tcp_pose=np.asarray(tcp_pose, dtype=float),
                    flange_pose=np.asarray(tcp_pose, dtype=float),
                    joints=joints,
                    phase=motion_context.phase,
                ),
            )

        def spec(stage_id, label):
            return {
                "tcp_pose": np.array(
                    [float(stage_id), 0.0, 0.3, 0.0, 0.0, 0.0]
                ),
                "context": context,
                "label": label,
            }

        cache = {}
        with self.assertRaises(PlanNotExecutable):
            _plan_tcp_stage_sequence_beam(
                object(),
                (spec(1, "pre_a"), spec(2, "grasp_a"), spec(3, "lift_a")),
                q0,
                args,
                "first",
                stage_option_cache=cache,
                plan_tcp_motion_stage_options=plan_options,
            )
        result = _plan_tcp_stage_sequence_beam(
            object(),
            (spec(1, "pre_b"), spec(2, "grasp_b"), spec(4, "lift_b")),
            q0,
            args,
            "second",
            stage_option_cache=cache,
            plan_tcp_motion_stage_options=plan_options,
        )

        self.assertEqual(calls, [1, 2, 3, 4])
        self.assertEqual(len(result), 1)


if __name__ == "__main__":
    unittest.main()
