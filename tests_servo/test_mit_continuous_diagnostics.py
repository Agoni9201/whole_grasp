"""Hardware-free continuous MIT diagnostics and stop ownership checks."""

import json
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from control_backends.force_ctrl.data_types import JointCommands
from control_backends.force_ctrl import mit_loop as module
from control_backends.force_ctrl.mit_loop import NeroMITLoop


class _Robot:
    def __init__(self):
        self.moves = []
        self.estops = 0
        self._parser = SimpleNamespace(**{
            f"motor_state_{i}": SimpleNamespace(
                timestamp=100.0 + i,
                msg=SimpleNamespace(torque=float(i), velocity=0.01 * i),
            ) for i in range(1, 8)
        })

    def get_joint_angles(self):
        return SimpleNamespace(timestamp=123.0, msg=np.zeros(7))

    def set_motion_mode(self, mode):
        assert mode == "mit"

    def move_mit(self, **kwargs):
        self.moves.append(kwargs)

    def electronic_emergency_stop(self):
        self.estops += 1


def _loop(robot=None):
    return NeroMITLoop(
        robot or _Robot(), SimpleNamespace(compute=lambda q: np.ones(7)),
        torque_limit=8.0, joint_limit_buffer=0.0,
        feedback_guard_enabled=False,  # synthetic historical timestamps; guard tested separately
    )


def _hold(pos=0.0):
    return JointCommands(
        pos=np.full(7, pos), vel=np.zeros(7),
        kp=np.full(7, 5.0), kd=np.full(7, 0.2), torques=np.zeros(7),
    )


def test_controlled_stop_is_atomic_fixed_hold_and_rejects_old_targets():
    loop = _loop()
    loop._running = True
    loop._impedance_cfg = SimpleNamespace(active=True)
    original = _hold(0.1)

    assert loop.hold_for_controlled_stop(original) is True
    original.pos[:] = 0.4
    assert np.allclose(loop._commands.pos, 0.1)
    assert loop._impedance_cfg.active is False
    assert loop._command_stop_latched is True
    assert loop.hold_for_controlled_stop(_hold(0.2)) is False
    assert np.allclose(loop._commands.pos, 0.1)
    loop.disable_cartesian_impedance()
    assert np.allclose(loop._commands.pos, 0.1)
    assert np.allclose(loop._commands.kp, 5.0)
    with pytest.raises(RuntimeError, match="latched MIT controlled stop"):
        loop.update_commands(_hold(0.3))
    with pytest.raises(RuntimeError, match="restart the workflow"):
        loop.start()


def test_controlled_stop_rejects_nonzero_velocity_before_latching():
    loop = _loop()
    loop._running = True
    commands = _hold()
    commands.vel[0] = 0.01
    with pytest.raises(ValueError, match="velocity and requested torque must be zero"):
        loop.hold_for_controlled_stop(commands)
    assert loop._command_stop_latched is False


def test_emergency_stop_serializes_and_reports_rpc_failure():
    loop = _loop()
    loop._running = True

    def fail():
        raise OSError("bus unavailable")

    loop._robot.electronic_emergency_stop = fail
    with pytest.raises(RuntimeError, match="bus unavailable"):
        loop.request_emergency_stop("test fault")
    assert loop.terminal_fault is True
    assert loop.running is False
    with pytest.raises(RuntimeError, match="terminal MIT fault"):
        loop.update_commands(_hold())


def test_emergency_stop_before_final_lock_prevents_late_mit_batch():
    loop = _loop()
    loop._running = True

    def gravity(q):
        loop.request_emergency_stop("between feedback and final send")
        return np.zeros(7)

    loop._grav.compute = gravity
    loop._control_loop()
    assert loop._robot.estops == 1
    assert loop._robot.moves == []


def test_continuous_log_spans_hold_and_stop_and_matches_sent_packets(tmp_path, monkeypatch):
    loop = _loop()
    loop.update_commands(_hold(0.1))
    path = tmp_path / "mit_control_cycles.jsonl"
    serialized_threads = []
    original_conversion = module._diagnostic_json_value

    def record_thread(value):
        serialized_threads.append(threading.current_thread().name)
        return original_conversion(value)

    monkeypatch.setattr(module, "_diagnostic_json_value", record_thread)
    loop.start_diagnostics(path)
    loop.set_diagnostics_phase("gripper_close")
    cycles = 0

    def finish_cycle(deadline):
        nonlocal cycles
        cycles += 1
        if cycles == 1:
            loop.hold_for_controlled_stop(_hold(0.0))
        elif cycles == 3:
            loop._running = False

    loop._sleep_until = finish_cycle
    loop._running = True
    loop._control_loop()
    loop.stop()
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert rows[0]["type"] == "metadata"
    assert "excludes firmware-internal" in rows[0]["torque_semantics"]
    samples = [row for row in rows if row["type"] == "cycle"]
    assert [row["phase"] for row in samples] == [
        "gripper_close", "controlled_stop_hold", "controlled_stop_hold"]
    assert samples[0]["q_command"] == [0.1] * 7
    assert samples[1]["q_command"] == [0.0] * 7
    assert samples[0]["tau_gravity"] == [1.0] * 7
    assert samples[0]["tau_ff_sent"] == [1.0] * 7
    assert samples[0]["tau_requested"] == [0.0] * 7
    assert samples[0]["tau_feedback"] == list(range(1, 8))
    assert samples[0]["torque_feedback_timestamps"] == list(range(101, 108))
    assert samples[0]["position_feedback_timestamp"] == 123.0
    assert samples[0]["wall_time"] > 0.0
    assert samples[0]["timestamp"].endswith("+00:00")
    assert samples[0]["loop_dt"] > 0.0
    timing = samples[0]['timing']
    assert timing['sent_joint_count'] == 7
    assert timing['active_stage'] == 'diagnostics'
    for name in ('feedback_read', 'state_safety', 'gravity', 'actuation_lock_wait',
                 'command_compute', 'send_preflight', 'send_mit', 'state_publish'):
        assert timing['stage_durations_s'][name] >= 0.0
    assert all(name == "nero_mit_log_writer" for name in serialized_threads)
    for sample, packets in zip(samples, [loop._robot.moves[i:i + 7] for i in range(0, 21, 7)]):
        for joint, packet in enumerate(packets):
            assert packet["p_des"] == sample["q_command"][joint]
            assert packet["v_des"] == sample["dq_command"][joint]
            assert packet["t_ff"] == sample["tau_ff_sent"][joint]
            assert packet["kp"] == sample["kp"][joint]
            assert packet["kd"] == sample["kd"][joint]
    status = loop.get_diagnostics_status()
    assert status["written_cycles"] == 3
    assert status["dropped_cycles"] == 0
    assert status["closed"] is True


def test_queue_overflow_drops_without_blocking_control_thread(tmp_path, monkeypatch):
    entered = threading.Event()
    release = threading.Event()
    original_conversion = module._diagnostic_json_value

    def blocked_conversion(value):
        if isinstance(value, dict) and value.get("type") == "cycle":
            entered.set()
            assert release.wait(timeout=2.0)
        return original_conversion(value)

    monkeypatch.setattr(module, "_diagnostic_json_value", blocked_conversion)
    writer = module._MITDiagnosticWriter(tmp_path / "bounded.jsonl", capacity=1)
    try:
        writer.submit({"type": "cycle", "sequence": 1})
        assert entered.wait(timeout=2.0)
        writer.submit({"type": "cycle", "sequence": 2})
        writer.submit({"type": "cycle", "sequence": 3})
        assert writer.status()["dropped_cycles"] == 1
        assert writer.records.qsize() == 1
    finally:
        release.set()
        status = writer.close(timeout=2.0)
    assert status["written_cycles"] == 2
    assert status["closed"] is True


def test_log_startup_failure_is_surfaced_without_hardware_action(tmp_path):
    loop = _loop()
    with pytest.raises(RuntimeError, match="Cannot start MIT diagnostics"):
        loop.start_diagnostics(tmp_path / "missing_directory" / "cycles.jsonl")
    assert loop._robot.moves == []
    assert loop._robot.estops == 0


def test_writer_failure_is_reported_without_becoming_a_control_fault(tmp_path, monkeypatch):
    loop = _loop()
    loop.update_commands(_hold())
    loop.start_diagnostics(tmp_path / "write_failure.jsonl")

    def fail_conversion(value):
        raise OSError("simulated disk full")

    monkeypatch.setattr(module, "_diagnostic_json_value", fail_conversion)
    loop._running = True
    loop._sleep_until = lambda deadline: setattr(loop, "_running", False)
    loop._control_loop()
    loop.stop()

    assert len(loop._robot.moves) == 7
    assert loop._robot.estops == 0
    assert loop.terminal_fault is False
    assert "simulated disk full" in loop.get_diagnostics_status()["error"]
