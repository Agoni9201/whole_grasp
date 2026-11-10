"""Fault injection only: no CAN bus, camera, or robot connection."""

import time
import json
import threading
from types import SimpleNamespace as NS

import numpy as np
import pytest

from control_backends.force_ctrl.feedback_health import FeedbackHealthMonitor
from control_backends.force_ctrl.mit_loop import NeroMITLoop
from control_backends.force_ctrl.data_types import JointCommands


class Robot:
    def __init__(self, stamp=None):
        stamp = time.time() if stamp is None else stamp
        self._parser = NS()
        for name, fields in (("joint_12", (1, 2)), ("joint_34", (3, 4)),
                             ("joint_56", (5, 6)), ("joint_7", (7,))):
            setattr(self._parser, name, NS(timestamp=stamp,
                    msg=NS(**{f"joint_{i}": 0.0 for i in fields})))
        for i in range(1, 8):
            setattr(self._parser, f"motor_state_{i}", NS(
                timestamp=stamp, msg=NS(velocity=0.0, torque=0.0)))
        self.moves, self.estops = [], 0

    def set_motion_mode(self, mode):
        assert mode == "mit"

    def move_mit(self, **command):
        self.moves.append(command)

    def electronic_emergency_stop(self):
        self.estops += 1


def test_fresh_feedback_contains_all_position_groups_not_just_joint7_stamp():
    robot = Robot(100.0)
    sample = FeedbackHealthMonitor().read(robot, 100.01, 10.01)
    assert len(sample["position_timestamps"]) == 7
    assert len(sample["motor_timestamps"]) == 7
    np.testing.assert_allclose(sample["q"], 0.0)


def test_velocity_and_torque_are_preserved_from_same_validated_motor_frames():
    robot = Robot(100.0)
    for index in range(1, 8):
        wrapped = getattr(robot._parser, f"motor_state_{index}")
        wrapped.msg.velocity, wrapped.msg.torque = index * .01, index * -.2
    sample = FeedbackHealthMonitor().read(robot, 100.01, 10.01)
    np.testing.assert_allclose(sample["dq_raw"], np.arange(1, 8) * .01)
    np.testing.assert_allclose(sample["dq"], np.arange(1, 8) * .01 * [-1, -1, -1, -1, -1, 1, -1])
    np.testing.assert_allclose(sample["tau"], np.arange(1, 8) * -.2)
    assert np.all(sample["motor_timestamps"] == 100.0)


@pytest.mark.parametrize("field", ["joint_12", "joint_34", "joint_56", "joint_7"]
                         + [f"motor_state_{i}" for i in range(1, 8)])
def test_any_single_stale_group_or_axis_fails_even_when_others_are_fresh(field):
    robot = Robot(100.0)
    getattr(robot._parser, field).timestamp = 99.8
    with pytest.raises(RuntimeError, match="stale/future"):
        FeedbackHealthMonitor().read(robot, 100.01, 10.01)


def test_advancing_but_old_timestamps_do_not_count_as_fresh():
    monitor = FeedbackHealthMonitor()
    monitor.read(Robot(100.0), 100.01, 10.01)
    with pytest.raises(RuntimeError, match="stale/future"):
        monitor.read(Robot(100.02), 100.30, 10.30)


def test_all_frames_individually_young_can_still_be_incoherent():
    robot = Robot(100.0)
    robot._parser.joint_12.timestamp = 99.93
    with pytest.raises(RuntimeError, match="skew"):
        FeedbackHealthMonitor().read(robot, 100.01, 10.01)


def test_timestamp_regression_and_wall_clock_jump_fail_closed():
    monitor = FeedbackHealthMonitor()
    monitor.read(Robot(100.0), 100.01, 10.01)
    with pytest.raises(RuntimeError, match="backwards"):
        monitor.read(Robot(99.99), 100.02, 10.02)
    with pytest.raises(RuntimeError, match="clock changed"):
        monitor.read(Robot(101.0), 101.01, 10.03)


def test_ntp_backward_step_reports_delta_and_still_rejects_cached_feedback():
    monitor = FeedbackHealthMonitor()
    sample = monitor.read(Robot(100.0), 100.01, 10.01)
    # Advancing monotonic time, but wall time steps backwards during a MIT batch.
    with pytest.raises(RuntimeError, match=r"wall clock changed by -427\.6ms"):
        monitor.validate_age(sample, 100.015 - .4276, 10.015)


@pytest.mark.parametrize("budget", [0.0, -1.0, float("nan"), float("inf")])
def test_invalid_budgets_are_rejected(budget):
    with pytest.raises(ValueError):
        FeedbackHealthMonitor(max_age_s=budget)


@pytest.mark.parametrize("phase", ["hold", "gripper_close", "trajectory:grasp"])
def test_inner_guard_is_active_in_hold_and_close_without_outer_executor(phase):
    robot = Robot(time.time() - .3)
    loop = NeroMITLoop(robot, NS(compute=lambda q: np.zeros(7)), joint_limit_buffer=0.0)
    loop.set_diagnostics_phase(phase)
    loop._running = True
    loop._control_loop()
    assert loop.terminal_fault
    assert "stale" in loop.last_error
    assert not robot.moves
    assert robot.estops == 1


def test_slow_control_work_is_detected_before_any_mit_batch():
    robot = Robot()
    def slow(q):
        time.sleep(.03)
        return np.zeros(7)
    loop = NeroMITLoop(robot, NS(compute=slow), joint_limit_buffer=0.0,
                       max_control_gap_s=.02)
    loop._running = True
    loop._control_loop()
    assert loop.terminal_fault
    assert "deadline missed" in loop.last_error
    assert not robot.moves
    assert robot.estops == 1


def test_previous_cycle_gap_is_detected_before_reading_or_sending():
    robot = Robot()
    loop = NeroMITLoop(robot, NS(compute=lambda q: np.zeros(7)), joint_limit_buffer=0.0)
    loop._last_control_loop_start = time.perf_counter() - .5
    loop._running = True
    loop._control_loop()
    assert loop.terminal_fault
    assert "deadline missed" in loop.last_error
    assert not robot.moves


def test_healthy_inner_cycle_still_sends_all_seven_joints():
    robot = Robot()
    loop = NeroMITLoop(robot, NS(compute=lambda q: np.zeros(7)), joint_limit_buffer=0.0)
    loop._sleep_until = lambda _: setattr(loop, "_running", False)
    loop._running = True
    loop._control_loop()
    assert len(robot.moves) == 7
    assert robot.estops == 0


def test_slow_sdk_send_cannot_continue_the_rest_of_the_command_batch():
    robot = Robot()
    original = robot.move_mit
    def slow(**command):
        original(**command)
        time.sleep(.03)
    robot.move_mit = slow
    loop = NeroMITLoop(robot, NS(compute=lambda q: np.zeros(7)),
                       joint_limit_buffer=0.0, max_control_gap_s=.02)
    loop._running = True
    loop._control_loop()
    assert loop.terminal_fault
    assert "deadline missed" in loop.last_error
    assert len(robot.moves) == 1
    assert loop._cycle_timing()['sent_joint_count'] == 1
    assert robot.estops == 1


def test_fault_log_keeps_failed_read_sample_and_all_seven_axis_ages(tmp_path):
    robot = Robot()
    loop = NeroMITLoop(robot, NS(compute=lambda q: np.zeros(7)), joint_limit_buffer=0.0)
    path = tmp_path / 'cycles.jsonl'
    loop.start_diagnostics(path)
    robot._parser.motor_state_1.timestamp = time.time() - .2
    loop.set_diagnostics_phase('hold')
    loop._running = True
    loop._control_loop()
    loop.stop_diagnostics()
    record = next(x for x in map(json.loads, path.read_text().splitlines()) if x['type'] == 'fault')
    assert robot.estops == 1 and not robot.moves
    assert record['phase'] == 'hold'
    assert record['feedback_snapshot']['motor_age_s'][0] >= .2
    assert len(record['feedback_snapshot']['position_age_s']) == 7
    assert len(record['feedback_snapshot']['motor_age_s']) == 7
    assert record['timing']['active_stage'] == 'feedback_read'
    assert record['timing']['stage_durations_s']['feedback_read'] >= 0
    assert record['parser_cache_after_stop_request']['diagnostic_only'] is True
    assert record['monotonic_time'] <= record['parser_cache_after_stop_request']['captured_monotonic_time']


def test_fault_distinguishes_old_computation_snapshot_from_fresh_parser_cache(tmp_path):
    robot = Robot()
    def slow_gravity(q):
        time.sleep(.05)
        now = time.time()
        for value in vars(robot._parser).values():
            value.timestamp = now
        return np.zeros(7)
    loop = NeroMITLoop(robot, NS(compute=slow_gravity), joint_limit_buffer=0.0,
                       max_feedback_age_s=.03, max_control_gap_s=.5)
    path = tmp_path / 'cycles.jsonl'
    loop.start_diagnostics(path)
    for value in vars(robot._parser).values():
        value.timestamp = time.time()
    loop._running = True
    loop._control_loop()
    loop.stop_diagnostics()
    record = next(x for x in map(json.loads, path.read_text().splitlines()) if x['type'] == 'fault')
    assert robot.estops == 1 and not robot.moves
    assert 'stale' in loop.last_error
    assert record['timing']['active_stage'] == 'send_preflight'
    assert record['timing']['stage_durations_s']['gravity'] >= .05
    old = record['feedback_snapshot']['motor_timestamps'][0]
    fresh = record['parser_cache_after_stop_request']['motor_timestamps'][0]
    assert fresh - old >= .05
    assert record['feedback_snapshot']['motor_age_s'][0] >= .05


def test_missing_axis_and_broken_diagnostics_cannot_prevent_emergency_stop(tmp_path, monkeypatch):
    robot = Robot()
    del robot._parser.motor_state_3
    loop = NeroMITLoop(robot, NS(compute=lambda q: np.zeros(7)), joint_limit_buffer=0.0)
    path = tmp_path / 'cycles.jsonl'
    loop.start_diagnostics(path)
    loop._running = True
    loop._control_loop()
    loop.stop_diagnostics()
    record = next(x for x in map(json.loads, path.read_text().splitlines()) if x['type'] == 'fault')
    assert record['feedback_snapshot'] is None
    assert record['parser_cache_after_stop_request']['motor_age_s'][2] is None
    assert robot.estops == 1 and not robot.moves
    # Unwinding the same fault must not send another electronic-stop frame.
    assert loop._hard_stop_locked('outer cleanup')
    assert robot.estops == 1
    # A distinct fault still requests hardware stop before optional logging.
    robot = Robot()
    loop = NeroMITLoop(robot, NS(compute=lambda q: np.zeros(7)), joint_limit_buffer=0.0)
    def broken(*args):
        assert robot.estops == 1  # stop must be attempted before diagnostics
        raise RuntimeError('writer failed')
    monkeypatch.setattr(loop, '_record_fault_diagnostics', broken)
    monkeypatch.setattr(loop, '_cycle_timing', lambda: (_ for _ in ()).throw(RuntimeError('clock failed')))
    assert loop._hard_stop_locked('test second stop')
    assert robot.estops == 1


def _established_hold(budget=.5):
    robot = Robot()
    loop = NeroMITLoop(robot, NS(compute=lambda q: np.zeros(7)),
                       joint_limit_buffer=0.0, hold_recovery_timeout_s=budget)
    loop.update_commands(JointCommands(
        pos=np.zeros(7), vel=np.zeros(7), kp=np.full(7, 30.0),
        kd=np.ones(7), torques=np.zeros(7)))
    loop.set_diagnostics_phase('hold')
    loop._sleep_until = lambda _: setattr(loop, '_running', False)
    loop._running = True
    loop._control_loop()
    assert len(robot.moves) == 7 and robot.estops == 0
    assert loop._last_healthy_hold_time is not None
    robot.moves.clear()
    return robot, loop


def _interrupt_hold(robot, loop, gap=.23):
    loop._last_control_loop_start = time.perf_counter() - gap
    loop._last_healthy_hold_time = time.perf_counter() - gap
    for frame in vars(robot._parser).values():
        frame.timestamp = time.time() - gap


def _refresh_frames(robot, done, before_first=None):
    time.sleep(.015)
    if before_first is not None:
        before_first()
    while not done.is_set():
        now = time.time()
        for frame in vars(robot._parser).values():
            frame.timestamp = now
        done.wait(.008)


def test_fixed_hold_recovers_230ms_gap_without_stale_sends_or_estop(tmp_path):
    robot, loop = _established_hold()
    path = tmp_path / 'recovery.jsonl'
    loop.start_diagnostics(path)
    _interrupt_hold(robot, loop)
    done = threading.Event()
    thread = threading.Thread(target=_refresh_frames, args=(robot, done))
    thread.start()
    try:
        loop._running = True
        loop._control_loop()
    finally:
        done.set()
        thread.join()
        loop.stop_diagnostics()
    assert robot.estops == 0 and not loop.terminal_fault
    assert len(robot.moves) == 7
    assert all(command['p_des'] == 0.0 for command in robot.moves)
    assert all(command['v_des'] == 0.0 for command in robot.moves)
    rows = list(map(json.loads, path.read_text().splitlines()))
    event = next(row for row in rows if row['type'] == 'hold_recovery')
    assert .23 <= event['interruption_s'] < .5
    assert event['fresh_samples'] == 3 and event['target_unchanged']


def test_fixed_hold_waits_without_sending_commands_on_stale_feedback():
    robot, loop = _established_hold(.08)
    _interrupt_hold(robot, loop, gap=.04)
    loop._feedback_health.max_control_gap_s = .02
    loop._feedback_health.max_age_s = .02
    loop._running = True
    loop._control_loop()
    assert robot.estops == 1 and not robot.moves
    assert 'recovery failed within 0.080s' in loop.last_error


@pytest.mark.parametrize('phase', ['trajectory:grasp', 'gripper_close', 'initializing'])
def test_recovery_grace_is_not_used_for_moving_phases(phase):
    robot, loop = _established_hold()
    _interrupt_hold(robot, loop)
    loop.set_diagnostics_phase(phase)
    loop._running = True
    loop._control_loop()
    assert robot.estops == 1 and not robot.moves
    assert 'deadline missed' in loop.last_error


@pytest.mark.parametrize('moving_mode', ['impedance', 'velocity', 'zero_kp'])
def test_hold_label_does_not_bypass_active_control_safety(moving_mode):
    robot, loop = _established_hold()
    _interrupt_hold(robot, loop)
    if moving_mode == 'impedance':
        loop._impedance_cfg = NS(active=True)
    elif moving_mode == 'velocity':
        loop._commands.vel[0] = .01
    else:
        loop._commands.kp[0] = 0.0
    loop._running = True
    loop._control_loop()
    assert robot.estops == 1 and not robot.moves


def test_drift_during_fixed_hold_recovery_is_not_silently_recentered():
    robot, loop = _established_hold()
    _interrupt_hold(robot, loop)
    for frame in vars(robot._parser).values():
        frame.timestamp = time.time()
    robot._parser.joint_12.msg.joint_1 = np.deg2rad(.6)
    loop._running = True
    loop._control_loop()
    assert robot.estops == 1 and not robot.moves
    assert 'drift exceeds 0.5deg' in loop.last_error


def test_user_shutdown_during_feedback_wait_does_not_request_estop():
    robot, loop = _established_hold()
    _interrupt_hold(robot, loop)
    cancel = threading.Timer(.015, lambda: setattr(loop, '_running', False))
    cancel.start()
    try:
        loop._running = True
        loop._control_loop()
    finally:
        cancel.join()
    assert robot.estops == 0 and not robot.moves
    assert not loop._hold_recovering


@pytest.mark.parametrize('budget', [-1.0, 1.01, float('nan'), float('inf')])
def test_fixed_hold_recovery_budget_is_bounded(budget):
    with pytest.raises(ValueError, match='hold_recovery_timeout_s'):
        NeroMITLoop(Robot(), NS(compute=lambda q: np.zeros(7)),
                    hold_recovery_timeout_s=budget)

