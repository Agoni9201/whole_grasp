import unittest
import json
import os
import tempfile
import time
from types import SimpleNamespace
from unittest import mock

import numpy as np

import real_nero_single_grasp_runner as runner
from real_nero_grasp import arm_execution
from real_nero_grasp.execution import execute_prevalidated_stage, execute_prevalidated_stages
from real_nero_grasp.types import MotionPreflightRejected


class ExecutionCollisionMonitorTest(unittest.TestCase):
    def test_live_approach_preflight_marks_measured_start_for_finger_allowance(self):
        arm = mock.Mock()
        arm.get_joint_angles.return_value = np.zeros(7)
        for phase in ("target_approach", "free_open", "carrying"):
            with self.subTest(phase=phase):
                stage = SimpleNamespace(label="return_edge", joints=np.zeros(7), phase=phase)
                validate = mock.Mock(return_value={})
                execute = mock.Mock(return_value=np.zeros(6))
                execute_prevalidated_stage(
                    arm, stage, np.zeros(7), object(), SimpleNamespace(),
                    collision_context_for_planned_stage_hook=lambda *_: (
                        SimpleNamespace(initial_support_release=None)),
                    validate_motion_joint_path_hook=validate,
                    execute_joint_stage_hook=execute,
                    safe_stop_hook=mock.Mock(),
                )
                self.assertEqual(validate.call_args.kwargs.get("allow_runtime_start", False),
                                 phase == "target_approach")
                execute.assert_called_once()

    def test_live_preflight_rejection_sends_no_stage_command(self):
        arm = mock.Mock()
        arm.get_joint_angles.return_value = np.zeros(7)
        stage = SimpleNamespace(label="active_side2", joints=np.ones(7))
        execute = mock.Mock()
        with self.assertRaises(MotionPreflightRejected) as raised:
            execute_prevalidated_stage(
                arm, stage, np.zeros(7), object(), SimpleNamespace(),
                collision_context_for_planned_stage_hook=lambda *_: object(),
                validate_motion_joint_path_hook=mock.Mock(
                    side_effect=runner.UnsafeMotionTarget("clearance below 5mm")),
                execute_joint_stage_hook=execute,
                safe_stop_hook=mock.Mock(),
            )
        self.assertFalse(raised.exception.motion_sent)
        execute.assert_not_called()
        arm.move_j.assert_not_called()

    def test_route_marks_preflight_rejection_after_an_executed_stage(self):
        for failed_stage in (0, 1):
            with self.subTest(failed_stage=failed_stage):
                stages = [SimpleNamespace(label=f"edge{index}", joints=np.zeros(7),
                                          phase="free_open") for index in range(2)]
                execute = mock.Mock(side_effect=(
                    [MotionPreflightRejected("live preflight rejected")]
                    if failed_stage == 0 else
                    [np.zeros(6), MotionPreflightRejected("live preflight rejected")]
                ))
                with self.assertRaises(MotionPreflightRejected) as raised:
                    execute_prevalidated_stages(
                        mock.Mock(), stages, np.zeros(7), object(), SimpleNamespace(),
                        execute_prevalidated_stage_hook=execute,
                    )
                self.assertEqual(raised.exception.motion_sent, failed_stage > 0)

    def test_mit_route_rejects_preflight_before_command_and_marks_partial_motion(self):
        for failed_stage in (0, 1):
            with self.subTest(failed_stage=failed_stage):
                arm = mock.Mock()
                arm.get_joint_angles.return_value = np.zeros(7)
                stages = [SimpleNamespace(label=f"active_edge{index}",
                                          joints=np.zeros(7), flange_pose=np.zeros(6),
                                          phase="free_open") for index in range(2)]
                reject = runner.UnsafeMotionTarget("clearance below 5mm")
                validate = mock.Mock(side_effect=(
                    [reject] if failed_stage == 0 else [{}, reject]
                ))
                session = mock.Mock()
                with mock.patch(
                    "real_nero_grasp.mit_trajectory_executor.get_mit_session",
                    return_value=session,
                ), self.assertRaises(MotionPreflightRejected) as raised:
                    execute_prevalidated_stages(
                        arm, stages, np.zeros(7), object(),
                        SimpleNamespace(trajectory_executor="mit"),
                        collision_context_for_planned_stage_hook=lambda *_: (
                            SimpleNamespace(initial_support_release=None)),
                        validate_motion_joint_path_hook=validate,
                        safe_stop_hook=mock.Mock(),
                    )
                self.assertEqual(session.execute_stages.call_count, failed_stage)
                self.assertEqual(raised.exception.motion_sent, failed_stage > 0)

    def context(self):
        scene = runner.MotionSafetyScene(
            full_scene_base=np.array([[1.0, 0.0, 0.0]], dtype=float),
            environment_base=np.array([[1.0, 0.0, 0.0]], dtype=float),
            target_base=np.zeros((0, 3), dtype=float),
            map_id="runtime_unit",
        )
        return runner._motion_context(scene, "carrying")

    def test_runtime_positive_clearance_shortfall_terminates_without_estop(self):
        arm = mock.Mock()
        arm.get_joint_angles.return_value = np.zeros(7)
        args = SimpleNamespace(
            motion_runtime_collision_guard=True,
            _runtime_motion_collision_context=self.context(),
        )
        violation = {
            "margin": -0.001,
            "nearest_clearance": 0.007,
            "required_clearance": 0.008,
            "primitive": "attached_target",
            "source": "environment",
        }
        with mock.patch.object(
            runner, "_motion_configuration_metrics", return_value=violation
        ), self.assertRaises(runner.ControlledMotionStopTriggered):
            runner.check_runtime_collision_current(arm, args, "transport")
        arm.terminate_motion.assert_called_once()
        arm.stop.assert_not_called()

    def test_cad_table_margin_shortfall_uses_confirmed_controlled_stop(self):
        for clearance in [.002, -.001]:
            with self.subTest(clearance=clearance):
                arm = mock.Mock()
                arm.get_joint_angles.return_value = np.zeros(7)
                context = runner._motion_context(self.context().scene, 'target_approach')
                args = SimpleNamespace(motion_runtime_collision_guard=True,
                    _runtime_motion_collision_context=context)
                metrics = dict(margin=clearance-.003, nearest_clearance=clearance,
                    required_clearance=.003, primitive='gripper_cad_mesh', source='table_plane')
                with mock.patch.object(runner, '_motion_configuration_metrics', return_value=metrics), \
                     mock.patch.object(runner, '_persist_runtime_collision_diagnostic'), \
                     self.assertRaises(runner.ControlledMotionStopTriggered):
                    runner.check_runtime_collision_current(arm, args, 'approach')
                arm.terminate_motion.assert_called_once()
                arm.stop.assert_not_called()

    def test_runtime_safe_sample_does_not_stop(self):
        arm = mock.Mock()
        arm.get_joint_angles.return_value = np.zeros(7)
        args = SimpleNamespace(
            motion_runtime_collision_guard=True,
            _runtime_motion_collision_context=self.context(),
        )
        safe = {
            "margin": 0.020,
            "nearest_clearance": 0.028,
            "required_clearance": 0.008,
            "primitive": "link1_2",
            "source": "environment",
        }
        with mock.patch.object(
            runner, "_motion_configuration_metrics", return_value=safe
        ):
            result = runner.check_runtime_collision_current(
                arm, args, "transport"
            )
        self.assertEqual(result, safe)
        arm.stop.assert_not_called()

    def test_active_return_small_floor_dip_rechecks_environment_and_continues(self):
        arm = mock.Mock()
        context = runner._motion_context(
            self.context().scene,
            "free_open",
            min_tcp_z=0.3356,
            runtime_min_tcp_z_slack=0.008,
        )
        args = SimpleNamespace(
            motion_runtime_collision_guard=True,
            _runtime_motion_collision_context=context,
            _runtime_motion_stage_label=(
                "active_side1_return_center_updated_scene_direct"
            ),
        )
        floor_violation = {
            "margin": -0.0012,
            "nearest_clearance": -0.0012,
            "required_clearance": 0.0,
            "primitive": "stage_min_tcp_z",
            "source": "path_constraint",
        }
        environment_safe = {
            "margin": 0.005,
            "nearest_clearance": 0.015,
            "required_clearance": 0.010,
            "primitive": "gripper_tool",
            "source": "environment",
        }
        observed_contexts = []

        def metrics_for_context(_q, observed_context, _args, **_kwargs):
            observed_contexts.append(observed_context)
            return (
                floor_violation
                if observed_context.min_tcp_z is not None
                else environment_safe
            )

        with mock.patch.object(
            runner,
            "_motion_configuration_metrics",
            side_effect=metrics_for_context,
        ), mock.patch.object(
            runner, "_persist_runtime_collision_diagnostic"
        ) as persist:
            result = runner.check_runtime_collision_current(
                arm,
                args,
                "active_side1_return_center_updated_scene_direct_start",
                joints=np.zeros(7),
            )

        self.assertEqual(result, environment_safe)
        self.assertEqual(len(observed_contexts), 2)
        self.assertIs(observed_contexts[1].scene, context.scene)
        self.assertEqual(observed_contexts[1].phase, context.phase)
        self.assertIsNone(observed_contexts[1].min_tcp_z)
        persist.assert_not_called()
        arm.stop.assert_not_called()

    def test_active_return_observed_43mm_drop_fits_new_60mm_total_envelope(self):
        arm = mock.Mock()
        context = runner._motion_context(
            self.context().scene,
            "free_open",
            min_tcp_z=0.2948243789077997,
            # 60 mm total tolerance minus the 2 mm strict planning floor.
            runtime_min_tcp_z_slack=0.058,
        )
        args = SimpleNamespace(
            motion_runtime_collision_guard=True,
            active_direct_return_downward_tolerance=0.002,
            _runtime_motion_collision_context=context,
            _runtime_motion_stage_label=(
                "active_side1_return_center_updated_scene_direct"
            ),
        )
        floor_violation = {
            "margin": -0.04065966532863746,
            "nearest_clearance": -0.04065966532863746,
            "required_clearance": 0.0,
            "primitive": "stage_min_tcp_z",
            "source": "path_constraint",
        }
        environment_safe = {
            "margin": 0.005,
            "nearest_clearance": 0.010,
            "required_clearance": 0.005,
            "primitive": "gripper_tool",
            "source": "environment",
        }
        with mock.patch.object(
            runner,
            "_motion_configuration_metrics",
            side_effect=[floor_violation, environment_safe],
        ), mock.patch.object(
            runner, "_persist_runtime_collision_diagnostic"
        ) as persist:
            result = runner.check_runtime_collision_current(
                arm,
                args,
                "active_side1_return_center_updated_scene_direct_joint",
                joints=np.zeros(7),
            )

        self.assertEqual(result, environment_safe)
        persist.assert_not_called()
        arm.terminate_motion.assert_not_called()
        arm.stop.assert_not_called()

    def test_active_return_floor_dip_beyond_runtime_slack_stops(self):
        arm = mock.Mock()
        context = runner._motion_context(
            self.context().scene,
            "free_open",
            min_tcp_z=0.3356,
            runtime_min_tcp_z_slack=0.008,
        )
        args = SimpleNamespace(
            motion_runtime_collision_guard=True,
            _runtime_motion_collision_context=context,
        )
        floor_violation = {
            "margin": -0.0081,
            "nearest_clearance": -0.0081,
            "required_clearance": 0.0,
            "primitive": "stage_min_tcp_z",
            "source": "path_constraint",
        }
        environment_safe = {
            "margin": 0.005,
            "nearest_clearance": 0.015,
            "required_clearance": 0.010,
            "primitive": "gripper_tool",
            "source": "environment",
        }
        with mock.patch.object(
            runner,
            "_motion_configuration_metrics",
            side_effect=[floor_violation, environment_safe],
        ) as metrics_call, mock.patch.object(
            runner, "_persist_runtime_collision_diagnostic"
        ) as persist, self.assertRaises(
            runner.ControlledMotionStopTriggered
        ):
            runner.check_runtime_collision_current(
                arm, args, "active_return", joints=np.zeros(7)
            )

        self.assertEqual(metrics_call.call_count, 2)
        persist.assert_called_once()
        arm.terminate_motion.assert_called_once()
        np.testing.assert_allclose(
            arm.terminate_motion.call_args.kwargs["fallback_joints"],
            np.zeros(7),
        )
        arm.stop.assert_not_called()

    def test_active_return_floor_slack_environment_penetration_uses_controlled_stop(self):
        arm = mock.Mock()
        context = runner._motion_context(
            self.context().scene,
            "free_open",
            min_tcp_z=0.3356,
            runtime_min_tcp_z_slack=0.008,
        )
        args = SimpleNamespace(
            motion_runtime_collision_guard=True,
            _runtime_motion_collision_context=context,
        )
        floor_violation = {
            "margin": -0.0081,
            "nearest_clearance": -0.0081,
            "required_clearance": 0.0,
            "primitive": "stage_min_tcp_z",
            "source": "path_constraint",
        }
        environment_violation = {
            "margin": -0.0115,
            "nearest_clearance": -0.0015,
            "required_clearance": 0.010,
            "primitive": "gripper_tool",
            "source": "environment",
        }
        with mock.patch.object(
            runner,
            "_motion_configuration_metrics",
            side_effect=[floor_violation, environment_violation],
        ) as metrics_call, mock.patch.object(
            runner, "_persist_runtime_collision_diagnostic"
        ) as persist, self.assertRaises(
            runner.ControlledMotionStopTriggered
        ):
            runner.check_runtime_collision_current(
                arm, args, "active_return", joints=np.zeros(7)
            )

        self.assertEqual(metrics_call.call_count, 2)
        self.assertEqual(persist.call_args.args[4]["source"], "environment")
        self.assertEqual(
            persist.call_args.args[4]["primitive"], "gripper_tool"
        )
        arm.terminate_motion.assert_called_once()
        arm.stop.assert_not_called()

    def test_zero_pointcloud_clearance_uses_controlled_stop(self):
        arm = mock.Mock()
        context = self.context()
        args = SimpleNamespace(
            motion_runtime_collision_guard=True,
            _runtime_motion_collision_context=context,
        )
        contact = {
            "margin": -0.005,
            "nearest_clearance": 0.0,
            "required_clearance": 0.005,
            "primitive": "gripper_finger_-1",
            "source": "environment",
        }
        with mock.patch.object(
            runner,
            "_motion_configuration_metrics",
            return_value=contact,
        ), mock.patch.object(
            runner, "_persist_runtime_collision_diagnostic"
        ), self.assertRaises(
            runner.ControlledMotionStopTriggered
        ):
            runner.check_runtime_collision_current(
                arm, args, "pregrasp", joints=np.zeros(7)
            )

        arm.terminate_motion.assert_called_once()
        arm.stop.assert_not_called()

    def test_floor_violation_without_explicit_runtime_slack_stops(self):
        arm = mock.Mock()
        context = runner._motion_context(
            self.context().scene,
            "free_open",
            min_tcp_z=0.3356,
        )
        args = SimpleNamespace(
            motion_runtime_collision_guard=True,
            _runtime_motion_collision_context=context,
        )
        floor_violation = {
            "margin": -0.0012,
            "nearest_clearance": -0.0012,
            "required_clearance": 0.0,
            "primitive": "stage_min_tcp_z",
            "source": "path_constraint",
        }
        with mock.patch.object(
            runner,
            "_motion_configuration_metrics",
            return_value=floor_violation,
        ) as metrics_call, mock.patch.object(
            runner, "_persist_runtime_collision_diagnostic"
        ), self.assertRaises(runner.SafetyStopTriggered):
            runner.check_runtime_collision_current(
                arm, args, "active_side1_outbound", joints=np.zeros(7)
            )

        metrics_call.assert_called_once()
        arm.terminate_motion.assert_not_called()
        arm.stop.assert_called_once()

    def test_unconfirmed_ordinary_floor_stop_escalates_to_estop(self):
        arm = mock.Mock()
        arm.terminate_motion.side_effect = RuntimeError("no stop feedback")
        context = runner._motion_context(
            self.context().scene,
            "free_open",
            min_tcp_z=0.3356,
            runtime_min_tcp_z_slack=0.008,
        )
        args = SimpleNamespace(
            motion_runtime_collision_guard=True,
            _runtime_motion_collision_context=context,
        )
        floor_violation = {
            "margin": -0.0081,
            "nearest_clearance": -0.0081,
            "required_clearance": 0.0,
            "primitive": "stage_min_tcp_z",
            "source": "path_constraint",
        }
        environment_safe = {
            "margin": 0.005,
            "nearest_clearance": 0.015,
            "required_clearance": 0.010,
            "primitive": "gripper_tool",
            "source": "environment",
        }
        with mock.patch.object(
            runner,
            "_motion_configuration_metrics",
            side_effect=[floor_violation, environment_safe],
        ), mock.patch.object(
            runner, "_persist_runtime_collision_diagnostic"
        ), self.assertRaises(runner.SafetyStopTriggered) as raised:
            runner.check_runtime_collision_current(
                arm, args, "active_return", joints=np.zeros(7)
            )

        self.assertNotIsInstance(
            raised.exception, runner.ControlledMotionStopTriggered
        )
        arm.terminate_motion.assert_called_once()
        arm.stop.assert_called_once()

    def test_unconfirmed_ordinary_clearance_stop_escalates_to_estop(self):
        arm = mock.Mock()
        arm.terminate_motion.side_effect = RuntimeError("no stop feedback")
        context = self.context()
        args = SimpleNamespace(
            motion_runtime_collision_guard=True,
            _runtime_motion_collision_context=context,
        )
        shortfall = {
            "margin": -0.0011,
            "nearest_clearance": 0.0039,
            "required_clearance": 0.0050,
            "primitive": "gripper_finger_-1",
            "source": "environment",
        }
        with mock.patch.object(
            runner,
            "_motion_configuration_metrics",
            return_value=shortfall,
        ), mock.patch.object(
            runner, "_persist_runtime_collision_diagnostic"
        ), self.assertRaises(runner.SafetyStopTriggered) as raised:
            runner.check_runtime_collision_current(
                arm, args, "pregrasp", joints=np.zeros(7)
            )

        self.assertNotIsInstance(
            raised.exception, runner.ControlledMotionStopTriggered
        )
        arm.terminate_motion.assert_called_once()
        arm.stop.assert_called_once()

    def test_runtime_violation_persists_measured_path_diagnostic(self):
        arm = mock.Mock()
        measured = np.full(7, 0.15, dtype=float)
        arm.get_joint_angles.return_value = measured
        context = self.context()
        violation = {
            "margin": -0.001,
            "nearest_clearance": 0.009,
            "required_clearance": 0.010,
            "primitive": "gripper_finger_+1",
            "source": "environment",
        }
        with tempfile.TemporaryDirectory() as save_dir:
            capture_dir = os.path.join(save_dir, "capture")
            args = SimpleNamespace(
                motion_runtime_collision_guard=True,
                _runtime_motion_collision_context=context,
                _runtime_motion_stage_label="active_return",
                _runtime_motion_stage_start_joints=np.zeros(7),
                _runtime_motion_stage_target_joints=np.full(7, 0.2),
                save_dir=save_dir,
                _last_capture_out_dir=capture_dir,
            )
            with mock.patch.object(
                runner, "_motion_configuration_metrics", return_value=violation
            ), mock.patch.object(
                runner, "_tcp_transform_from_joints", return_value=np.eye(4)
            ), self.assertRaises(runner.ControlledMotionStopTriggered):
                runner.check_runtime_collision_current(
                    arm, args, "active_return_joint", joints=measured
                )

            arm.terminate_motion.assert_called_once()
            arm.stop.assert_not_called()

            root_path = os.path.join(
                save_dir, "runtime_collision_diagnostics.jsonl"
            )
            capture_path = os.path.join(
                capture_dir, "runtime_collision_diagnostics.jsonl"
            )
            self.assertTrue(os.path.exists(root_path))
            self.assertTrue(os.path.exists(capture_path))
            with open(root_path, "r", encoding="utf-8") as stream:
                payload = json.loads(stream.readline())
            self.assertEqual(payload["stage_label"], "active_return")
            self.assertEqual(payload["metrics"]["primitive"], "gripper_finger_+1")
            self.assertAlmostEqual(payload["path_alpha_raw"], 0.75)
            self.assertAlmostEqual(payload["max_joint_path_residual_deg"], 0.0)

    def test_runtime_support_release_small_settling_drop_is_tolerated(self):
        arm = mock.Mock()
        arm.get_joint_angles.return_value = np.zeros(7)
        base_context = self.context()
        release = runner.InitialSupportRelease(
            strict_environment_base=base_context.scene.environment_base,
            support_points_base=np.array([[0.0, 0.0, 0.0]]),
            start_tcp_pose=np.zeros(6),
            strict_after_rise=0.018,
            downward_tolerance=0.001,
        )
        context = runner._motion_context(
            base_context.scene,
            "support_release",
            attached_target_tcp=np.array([[0.0, 0.0, 0.0]]),
            initial_support_release=release,
        )
        args = SimpleNamespace(
            motion_runtime_collision_guard=True,
            _runtime_motion_collision_context=context,
            motion_runtime_support_drop_slack=0.005,
        )
        safe_samples = [
            {
                "margin": 0.020,
                "nearest_clearance": 0.028,
                "required_clearance": 0.008,
                "primitive": "link1_2",
                "source": "environment",
                "support_rise": 0.010,
            },
            {
                "margin": 0.020,
                "nearest_clearance": 0.028,
                "required_clearance": 0.008,
                "primitive": "link1_2",
                "source": "environment",
                "support_rise": 0.008,
            },
        ]
        with mock.patch.object(
            runner,
            "_motion_configuration_metrics",
            side_effect=safe_samples,
        ):
            runner.check_runtime_collision_current(arm, args, "support")
            runner.check_runtime_collision_current(arm, args, "support")
        arm.terminate_motion.assert_not_called()
        arm.stop.assert_not_called()

    def test_runtime_support_release_large_settling_drop_uses_controlled_stop(self):
        arm = mock.Mock()
        arm.get_joint_angles.return_value = np.zeros(7)
        base_context = self.context()
        release = runner.InitialSupportRelease(
            strict_environment_base=base_context.scene.environment_base,
            support_points_base=np.array([[0.0, 0.0, 0.0]]),
            start_tcp_pose=np.zeros(6),
            strict_after_rise=0.018,
            downward_tolerance=0.001,
        )
        context = runner._motion_context(
            base_context.scene,
            "support_release",
            attached_target_tcp=np.array([[0.0, 0.0, 0.0]]),
            initial_support_release=release,
        )
        args = SimpleNamespace(
            motion_runtime_collision_guard=True,
            _runtime_motion_collision_context=context,
            motion_runtime_support_drop_slack=0.005,
        )
        samples = [
            {
                "margin": 0.020,
                "nearest_clearance": 0.028,
                "required_clearance": 0.008,
                "primitive": "link1_2",
                "source": "environment",
                "support_rise": 0.010,
            },
            {
                "margin": 0.020,
                "nearest_clearance": 0.028,
                "required_clearance": 0.008,
                "primitive": "link1_2",
                "source": "environment",
                "support_rise": 0.003,
            },
        ]
        with mock.patch.object(
            runner, "_motion_configuration_metrics", side_effect=samples
        ):
            runner.check_runtime_collision_current(arm, args, "support")
            with self.assertRaises(runner.ControlledMotionStopTriggered):
                runner.check_runtime_collision_current(arm, args, "support")
        arm.terminate_motion.assert_called_once()
        arm.stop.assert_not_called()

    def test_runtime_support_xy_drift_uses_controlled_stop(self):
        arm = mock.Mock()
        arm.get_joint_angles.return_value = np.zeros(7)
        context = self.context()
        args = SimpleNamespace(
            motion_runtime_collision_guard=True,
            _runtime_motion_collision_context=context,
        )
        violation = {
            "margin": -0.001,
            "nearest_clearance": -0.001,
            "required_clearance": 0.0,
            "primitive": "support_release_xy_drift",
            "source": "initial_support_constraint",
        }
        with mock.patch.object(
            runner, "_motion_configuration_metrics", return_value=violation
        ), mock.patch.object(
            runner, "_persist_runtime_collision_diagnostic"
        ), self.assertRaises(runner.ControlledMotionStopTriggered):
            runner.check_runtime_collision_current(
                arm, args, "support", joints=np.zeros(7)
            )
        arm.terminate_motion.assert_called_once()
        arm.stop.assert_not_called()

    def test_stop_latch_refuses_all_followup_commands(self):
        arm = object.__new__(runner.RealNeroGraspArm)
        arm.arm = mock.Mock()
        arm.gripper = mock.Mock()
        arm.gripper_enabled = True
        arm.gripper_max_width = 0.085
        arm.gripper_max_force = 0.50
        arm._safety_stop_latched = False
        arm._safety_stop_reason = None
        arm._safety_stop_request_confirmed = False

        arm.stop()
        arm.stop()
        arm.arm.electronic_emergency_stop.assert_called_once()

        with self.assertRaises(runner.SafetyStopTriggered):
            arm.move_j(np.zeros(7))
        with self.assertRaises(runner.SafetyStopTriggered):
            arm.move_p(np.zeros(6))
        with self.assertRaises(runner.SafetyStopTriggered):
            arm.open_gripper(0.085, 0.30, 0.0)
        with self.assertRaises(runner.SafetyStopTriggered):
            arm.close_gripper(0.0, 0.30, 0.0)
        arm.arm.move_j.assert_not_called()
        arm.arm.move_p.assert_not_called()
        arm.gripper.move_gripper_m.assert_not_called()

    def test_ordinary_trajectory_termination_never_sends_estop_bit(self):
        arm = object.__new__(runner.RealNeroGraspArm)
        arm.arm = mock.Mock()
        arm.arm.get_joint_angles.return_value = np.zeros(7)
        arm._safety_stop_latched = False
        arm._safety_stop_reason = None
        arm._controlled_stop_request_confirmed = False
        def fresh_sample(_monitor):
            stamp = time.time()
            return {"q": np.zeros(7), "position_timestamps": np.full(7, stamp),
                    "motor_timestamps": np.full(7, stamp)}
        arm._read_stop_feedback = fresh_sample

        result = arm.terminate_motion(fallback_joints=np.zeros(7))

        self.assertTrue(result["confirmed"])
        self.assertEqual(
            result["method"], "track_terminate_speed_zero_hold"
        )
        self.assertTrue(arm._safety_stop_latched)
        self.assertTrue(arm._controlled_stop_request_confirmed)
        self.assertEqual(arm.arm._send_msg.call_count, 3)
        for call in arm.arm._send_msg.call_args_list:
            message = call.args[0]
            self.assertEqual(message.emergency_stop, 0)
            self.assertEqual(message.track_ctrl, 0x06)
            self.assertEqual(message.grag_teach_ctrl, 0)
        arm.arm.set_speed_percent.assert_called_once_with(0)
        arm.arm.move_j.assert_called_once_with([0.0] * 7)
        arm.arm.electronic_emergency_stop.assert_not_called()
        with self.assertRaises(runner.SafetyStopTriggered):
            arm.move_j(np.zeros(7))

    def test_failed_stop_rpc_can_be_retried_while_motion_stays_latched(self):
        arm = object.__new__(runner.RealNeroGraspArm)
        arm.arm = mock.Mock()
        arm.arm.electronic_emergency_stop.side_effect = [
            RuntimeError("can timeout"),
            RuntimeError("can timeout"),
            RuntimeError("can timeout"),
            None,
        ]
        arm._safety_stop_latched = False
        arm._safety_stop_request_confirmed = False
        arm._safety_stop_max_attempts = 3

        with self.assertRaises(RuntimeError):
            arm.stop()
        self.assertTrue(arm._safety_stop_latched)
        self.assertFalse(arm._safety_stop_request_confirmed)
        with self.assertRaises(runner.SafetyStopTriggered):
            arm.move_j(np.zeros(7))

        arm.stop()
        self.assertTrue(arm._safety_stop_request_confirmed)
        self.assertEqual(arm.arm.electronic_emergency_stop.call_count, 4)

    def test_stage_timeout_ordinary_stops_before_runtime_context_is_removed(self):
        arm = mock.Mock()
        arm.get_joint_angles.return_value = np.zeros(7)
        arm.terminate_motion.return_value = {
            "confirmed": True,
            "method": "unit_test",
        }
        context = self.context()
        args = SimpleNamespace(
            motion_stage_start_joint_match_deg=2.0,
            _runtime_motion_collision_context="previous",
        )
        stage = runner.PlannedJointStage(
            label="transport",
            tcp_pose=np.zeros(6),
            flange_pose=np.zeros(6),
            joints=np.zeros(7),
            phase="carrying",
        )
        plan = runner.GraspMotionPlan(
            start_joints=np.zeros(7),
            before_close=(stage,),
            after_close=tuple(),
            after_release=tuple(),
            empty_recovery=tuple(),
            safety_scene=context.scene,
        )
        context_seen_by_stop = []

        def fake_terminate(*, fallback_joints=None):
            context_seen_by_stop.append(
                getattr(args, "_runtime_motion_collision_context", None)
            )
            return {"confirmed": True, "method": "unit_test"}

        arm.terminate_motion.side_effect = fake_terminate

        with mock.patch.object(
            runner, "validate_motion_joint_path"
        ), mock.patch.object(
            runner,
            "move_flange_pose",
            side_effect=runner.MotionNotReached("timeout"),
        ), mock.patch.object(runner, "safe_stop") as emergency_stop:
            with self.assertRaises(runner.ControlledMotionStopTriggered):
                runner.execute_prevalidated_stage(
                    arm, stage, np.zeros(7), plan, args
                )

        self.assertEqual(context_seen_by_stop, [context])
        emergency_stop.assert_not_called()
        self.assertEqual(args._runtime_motion_collision_context, "previous")

    def test_stage_timeout_escalates_only_when_ordinary_stop_is_unconfirmed(self):
        arm = mock.Mock()
        arm.get_joint_angles.return_value = np.zeros(7)
        arm.terminate_motion.side_effect = RuntimeError("settling not confirmed")
        context = self.context()
        args = SimpleNamespace(motion_stage_start_joint_match_deg=2.0)
        stage = runner.PlannedJointStage(
            label="pregrasp",
            tcp_pose=np.zeros(6),
            flange_pose=np.zeros(6),
            joints=np.zeros(7),
            phase="free_open",
        )
        plan = runner.GraspMotionPlan(
            start_joints=np.zeros(7),
            before_close=(stage,),
            after_close=tuple(),
            after_release=tuple(),
            empty_recovery=tuple(),
            safety_scene=context.scene,
        )

        with mock.patch.object(
            runner, "validate_motion_joint_path"
        ), mock.patch.object(
            runner,
            "move_flange_pose",
            side_effect=runner.MotionNotReached("retry did not start"),
        ), mock.patch.object(runner, "safe_stop") as emergency_stop:
            with self.assertRaises(runner.SafetyStopTriggered) as raised:
                runner.execute_prevalidated_stage(
                    arm, stage, np.zeros(7), plan, args
                )

        self.assertNotIsInstance(
            raised.exception, runner.ControlledMotionStopTriggered
        )
        arm.terminate_motion.assert_called_once()
        emergency_stop.assert_called_once()

    def test_table_guard_tcp_read_error_uses_ordinary_stop_first(self):
        arm = mock.Mock()
        arm.get_tcp_pose.side_effect = RuntimeError("tcp feedback timeout")
        arm.get_joint_angles.return_value = np.zeros(7)
        arm.terminate_motion.return_value = {
            "confirmed": True,
            "method": "unit_test",
        }
        args = SimpleNamespace(table_guard_stop_on_read_error=True)

        with mock.patch.object(runner, "safe_stop") as emergency_stop:
            with self.assertRaises(runner.ControlledMotionStopTriggered):
                arm_execution.check_table_guard_current(
                    arm,
                    args,
                    "pregrasp",
                    table_guard_enabled_hook=lambda _args: True,
                    table_guard_min_tcp_z_hook=lambda _args: 0.10,
                    safe_stop_hook=emergency_stop,
                )

        arm.terminate_motion.assert_called_once()
        emergency_stop.assert_not_called()


if __name__ == "__main__":
    unittest.main()
