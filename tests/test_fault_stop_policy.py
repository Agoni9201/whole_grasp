"""Fault injection only: no CAN connection, camera, motor enable or movement."""
import threading
import time
from types import SimpleNamespace as NS
from unittest.mock import Mock

import numpy as np
import pytest

from control_backends.force_ctrl.data_types import JointCommands
from control_backends.force_ctrl.mit_loop import NERO_JOINT_LIMITS, NeroMITLoop
from real_nero_grasp.arm import RealNeroGraspArm
from real_nero_grasp.arm_execution import _runtime_collision_hook_required
from real_nero_grasp.runtime_safety import RuntimeSafetyHooks, check_runtime_collision_current
from real_nero_grasp.stop_policy import request_ordinary_trajectory_stop
from real_nero_grasp.types import ControlledMotionStopTriggered, SafetyStopTriggered


def test_r2_confirms_stop_and_records_timestamp_without_estop():
    arm = NS(get_joint_angles=lambda: np.zeros(7),
             terminate_motion=Mock(return_value={"confirmed": True, "method": "test_hold"}))
    emergency = Mock()
    args = NS(_experiment_log_state={})
    request_ordinary_trajectory_stop(arm, "task failed", safe_stop_hook=emergency,
                                     fault_context=args)
    emergency.assert_not_called()
    event = args._experiment_log_state["fault_events"][0]
    assert event["level"] == "R2" and event["stop_confirmed"]
    assert event["timestamp"].endswith("+08:00")
    assert not event["emergency_stop_requested"]


@pytest.mark.parametrize("response", [None, {}, {"confirmed": False}, Mock()])
def test_stop_without_explicit_confirmation_is_r3(response):
    arm = NS(get_joint_angles=lambda: np.zeros(7),
             terminate_motion=Mock(return_value=response))
    emergency = Mock()
    with pytest.raises(SafetyStopTriggered):
        request_ordinary_trajectory_stop(arm, "failed", safe_stop_hook=emergency)
    emergency.assert_called_once()
    assert arm._last_fault_decision["level"] == "R3"
    assert not arm._last_fault_decision["stop_confirmed"]


def test_existing_emergency_request_is_not_repeated_on_unwind():
    arm = NS(_safety_stop_request_confirmed=True, get_joint_angles=lambda: np.zeros(7),
             terminate_motion=Mock(side_effect=RuntimeError("MIT terminal fault")))
    emergency = Mock()
    with pytest.raises(SafetyStopTriggered):
        request_ordinary_trajectory_stop(arm, "failed", safe_stop_hook=emergency)
    emergency.assert_not_called()


@pytest.mark.parametrize("stop_fails", [False, True])
def test_collision_calculation_failure_stops_before_any_retry(stop_fails):
    arm = NS(get_joint_angles=lambda: np.zeros(7),
             terminate_motion=Mock(return_value={"confirmed": True}))
    if stop_fails:
        arm.terminate_motion.side_effect = RuntimeError("feedback untrusted")
    emergency = Mock()
    hooks = RuntimeSafetyHooks(
        motion_configuration_metrics=Mock(side_effect=ValueError("bad map")),
        persist_runtime_collision_diagnostic=Mock(), safe_stop=emergency,
    )
    args = NS(motion_runtime_collision_guard=True, motion_runtime_torque_guard=False,
              _runtime_motion_collision_context=NS())
    with pytest.raises(SafetyStopTriggered) as raised:
        check_runtime_collision_current(arm, args, "stage", hooks=hooks)
    arm.terminate_motion.assert_called_once()
    assert isinstance(raised.value, ControlledMotionStopTriggered) is not stop_fails
    assert emergency.call_count == int(stop_fails)


def test_missing_collision_hook_uses_r2_stop_not_blind_continuation():
    arm = NS(get_joint_angles=lambda: np.zeros(7),
             terminate_motion=Mock(return_value={"confirmed": True}))
    args = NS(_runtime_motion_collision_context=object(), motion_runtime_collision_guard=True)
    from real_nero_grasp import arm_execution
    emergency = Mock()
    from unittest.mock import patch
    with patch.object(arm_execution, "safe_stop", emergency), pytest.raises(ControlledMotionStopTriggered):
        _runtime_collision_hook_required(arm, args, "stage")
    emergency.assert_not_called()
    arm.terminate_motion.assert_called_once()


def _loop(q, *, stop_gains=True):
    robot = NS(electronic_emergency_stop=Mock(), set_motion_mode=Mock(), move_mit=Mock(),
               get_joint_angles=lambda: NS(msg=q.copy(), timestamp=time.time()),
               get_motor_states=lambda i: NS(msg=NS(torque=0.), timestamp=time.time()))
    loop = NeroMITLoop(robot, NS(compute=lambda q: np.ones(7)),
                       feedback_guard_enabled=False, joint_limit_buffer=.1,
                       controlled_stop_kp=np.full(7, 5.) if stop_gains else None,
                       controlled_stop_kd=np.full(7, .2) if stop_gains else None)
    loop._running = True
    stamp = [time.time()]
    def read():
        stamp[0] += .005
        loop._joint_pos_timestamp = stamp[0]
        loop._feedback_health_sample = {
            "q": q.copy(), "position_timestamps": np.full(7, stamp[0]),
            "motor_timestamps": np.full(7, stamp[0])}
        return q.copy()
    loop._read_joint_pos = read
    count = [0]
    def sleep(_deadline):
        count[0] += 1
        if count[0] >= 5:
            loop._running = False
    loop._sleep_until = sleep
    return loop


def test_software_buffer_stops_old_targets_but_keeps_sender_and_gravity():
    q = np.zeros(7)
    q[6] = NERO_JOINT_LIMITS[6, 1] - .05
    loop = _loop(q)
    loop._control_loop()
    assert not loop.terminal_fault
    assert loop._buffer_stop_confirmed and loop._command_stop_latched
    assert loop._controlled_stop_reason.startswith("R2")
    loop._robot.electronic_emergency_stop.assert_not_called()
    assert loop._robot.move_mit.call_count == 35
    for call in loop._robot.move_mit.call_args_list:
        assert call.kwargs["kp"] == 5.
        assert call.kwargs["v_des"] == 0.
        assert call.kwargs["t_ff"] == 1.  # gravity still computed and sent
    with pytest.raises(RuntimeError, match="latched MIT controlled stop"):
        loop.update_commands(JointCommands.zeros())


@pytest.mark.parametrize("kind", ["physical_limit", "missing_hold_profile", "invalid_feedback"])
def test_unsafe_buffer_stop_still_escalates_without_sending_mit(kind):
    q = np.zeros(7)
    q[6] = NERO_JOINT_LIMITS[6, 1] + (.01 if kind == "physical_limit" else -.05)
    if kind == "invalid_feedback":
        q[0] = np.nan
    loop = _loop(q, stop_gains=kind != "missing_hold_profile")
    loop._control_loop()
    assert loop.terminal_fault
    loop._robot.electronic_emergency_stop.assert_called_once()
    loop._robot.move_mit.assert_not_called()


def test_buffer_stop_cannot_wait_forever_without_new_frames(monkeypatch):
    from control_backends.force_ctrl import mit_loop as module
    q = np.zeros(7)
    q[6] = NERO_JOINT_LIMITS[6, 1] - .05
    loop = _loop(q)
    loop._read_joint_pos()
    clock = NS(now=10.)
    monkeypatch.setattr(module.time, "monotonic", lambda: clock.now)
    loop._check_cycle_joint_limits(q)
    clock.now += 2.01
    with pytest.raises(RuntimeError, match="not confirmed"):
        loop._check_cycle_joint_limits(q)


def test_logging_errors_do_not_turn_into_control_faults():
    loop = _loop(np.zeros(7))
    loop._diagnostics_writer = NS(status=Mock(side_effect=OSError("disk error")),
                                  close=Mock(side_effect=OSError("disk error")))
    assert "disk error" in loop.get_diagnostics_status()["error"]
    assert "disk error" in loop.stop_diagnostics()["error"]
    loop._robot.electronic_emergency_stop.assert_not_called()


def test_sender_join_does_not_wait_for_log_writer():
    loop = _loop(np.zeros(7))
    loop._thread = threading.Thread(target=lambda: None)
    loop._thread.start()
    loop.stop_diagnostics = Mock()
    loop.stop_sender()
    loop.stop_diagnostics.assert_not_called()
    assert not loop.running


def test_stale_normal_mode_feedback_cannot_confirm_idle_or_stop(monkeypatch):
    from control_backends.force_ctrl.feedback_health import FeedbackStale
    arm = RealNeroGraspArm.__new__(RealNeroGraspArm)
    arm.arm = NS(get_arm_status=lambda: NS(msg=NS(motion_status=0)), move_j=Mock(),
                 _send_msg=Mock(), set_speed_percent=Mock())
    arm._controlled_stop_request_confirmed = False
    arm._read_stop_feedback = Mock(side_effect=FeedbackStale("stale"))
    assert arm._confirm_idle_for_program_exit() is None
    with pytest.raises(FeedbackStale):
        arm.terminate_motion()
    arm.arm.move_j.assert_not_called()


def test_servo_r2_prepares_hold_without_electronic_stop():
    from servo_delivery import ServoDeliveryNode
    node = ServoDeliveryNode.__new__(ServoDeliveryNode)
    node._shutdown_prepared = False
    node._emergency_stop_requested = False
    node.backend = NS(prepare_shutdown=Mock(), emergency_stop=Mock())
    node.get_logger = Mock()
    node._shutdown = Mock(side_effect=SystemExit(1))
    with pytest.raises(SystemExit):
        node._stop_task("invalid target configuration")
    node.backend.prepare_shutdown.assert_called_once()
    node.backend.emergency_stop.assert_not_called()
    assert node._shutdown_prepared


def test_r1_verified_idle_task_failure_sends_no_extra_motion():
    arm = NS(get_joint_angles=lambda: np.zeros(7), _mit_control_owner=lambda: None,
             _confirm_idle_for_program_exit=Mock(return_value={
                 "confirmed": True, "method": "controller_idle_no_command"}),
             terminate_motion=Mock())
    emergency = Mock()
    request_ordinary_trajectory_stop(arm, "no executable candidate", safe_stop_hook=emergency)
    assert arm._last_fault_decision["level"] == "R1"
    arm.terminate_motion.assert_not_called()
    emergency.assert_not_called()


def test_terminal_print_failure_does_not_change_a_confirmed_stop(monkeypatch):
    arm = NS(get_joint_angles=lambda: np.zeros(7),
             terminate_motion=Mock(return_value={"confirmed": True}))
    emergency = Mock()
    monkeypatch.setattr("builtins.print", Mock(side_effect=OSError("closed terminal")))
    result = request_ordinary_trajectory_stop(arm, "task error", safe_stop_hook=emergency)
    assert result["confirmed"]
    emergency.assert_not_called()


def test_terminal_print_failure_cannot_mask_r3_stop_request(monkeypatch):
    arm = NS(get_joint_angles=lambda: np.zeros(7),
             terminate_motion=Mock(side_effect=RuntimeError("feedback lost")))
    emergency = Mock()
    monkeypatch.setattr("builtins.print", Mock(side_effect=OSError("closed terminal")))
    with pytest.raises(SafetyStopTriggered):
        request_ordinary_trajectory_stop(arm, "task error", safe_stop_hook=emergency)
    emergency.assert_called_once()


@pytest.mark.parametrize("debug", [
    {"terminal_fault": True, "last_error": "control fault"},
    {"terminal_fault": True, "last_error": "electronic emergency stop call failed"},
    None,
])
def test_servo_r3_does_not_probe_diagnostics_before_stop(debug):
    from servo_delivery import ServoDeliveryNode
    node = ServoDeliveryNode.__new__(ServoDeliveryNode)
    node.backend = NS(get_debug_state=Mock(return_value=debug), emergency_stop=Mock())
    if debug is None:
        node.backend.get_debug_state.side_effect = RuntimeError("diagnostics broken")
    node.get_logger = Mock()
    node._shutdown = Mock(side_effect=SystemExit(1))
    with pytest.raises(SystemExit):
        node._fail_closed("control feedback invalid")
    node.backend.emergency_stop.assert_called_once()
    node.backend.get_debug_state.assert_not_called()


def test_frozen_frames_cannot_confirm_normal_mode_stop(monkeypatch):
    from real_nero_grasp import arm as module
    clock = NS(now=100.)
    monkeypatch.setattr(module, "time", NS(time=lambda: clock.now, monotonic=lambda: clock.now,
                        sleep=lambda dt: setattr(clock, "now", clock.now + dt)))
    arm = RealNeroGraspArm.__new__(RealNeroGraspArm)
    arm.arm = NS(move_j=Mock(), set_speed_percent=Mock())
    arm._controlled_stop_request_confirmed = False
    arm._stop_confirm_timeout = .25
    arm._send_ordinary_trajectory_terminate = lambda frames: frames
    arm._read_stop_feedback = lambda monitor: {
        "q": np.zeros(7), "position_timestamps": np.full(7, 100.),
        "motor_timestamps": np.full(7, 100.)}
    with pytest.raises(RuntimeError, match="settling was not confirmed"):
        arm.terminate_motion()
    assert not arm._controlled_stop_request_confirmed


def test_old_stop_confirmation_cannot_bypass_new_feedback_failure():
    arm = RealNeroGraspArm.__new__(RealNeroGraspArm)
    arm._controlled_stop_request_confirmed = True
    arm._confirm_idle_for_program_exit = lambda: None
    with pytest.raises(RuntimeError, match="no longer verified"):
        arm.terminate_motion()
    assert not arm._controlled_stop_request_confirmed


def test_sole_owner_deduplicates_completed_electronic_stop_and_forbids_restart():
    loop = _loop(np.zeros(7))
    loop.request_emergency_stop("first control fault")
    loop.request_emergency_stop("outer cleanup noticed the same fault")
    loop._robot.electronic_emergency_stop.assert_called_once()
    assert loop.last_error == "first control fault"
    with pytest.raises(RuntimeError, match="terminal fault"):
        loop.start()


def test_failed_electronic_request_can_still_be_retried():
    loop = _loop(np.zeros(7))
    loop._robot.electronic_emergency_stop.side_effect = [OSError("CAN unavailable"), None]
    with pytest.raises(RuntimeError, match="CAN unavailable"):
        loop.request_emergency_stop("control fault")
    loop.request_emergency_stop("retry failed stop request")
    assert loop._robot.electronic_emergency_stop.call_count == 2
    assert loop._electronic_stop_sent and loop.terminal_fault


def test_closed_terminal_does_not_turn_normal_exit_into_estop(monkeypatch):
    arm = RealNeroGraspArm.__new__(RealNeroGraspArm)
    arm.get_joint_angles = lambda: np.zeros(7)
    arm._confirm_idle_for_program_exit = lambda **kwargs: None
    arm.terminate_motion = lambda **kwargs: {"confirmed": True}
    arm.stop = Mock()
    monkeypatch.setattr("builtins.print", Mock(side_effect=OSError("closed terminal")))
    assert arm.stop_for_program_exit()
    arm.stop.assert_not_called()


def test_controller_refuses_fault_restart_before_connect_reset_or_enable():
    from control_backends.force_ctrl.controller import NeroForceController
    controller = NeroForceController.__new__(NeroForceController)
    controller._loop = NS(terminal_fault=True)
    controller._emergency_stopped = False
    controller._started = False
    controller._robot = Mock()
    with pytest.raises(RuntimeError, match="explicit recovery"):
        controller.start()
    controller._robot.connect.assert_not_called()
    controller._robot.reset.assert_not_called()
    controller._robot.enable.assert_not_called()
