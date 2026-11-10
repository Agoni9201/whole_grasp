"""Offline checks of torque-continuous stop transitions and sent-batch ownership."""
from types import SimpleNamespace as NS

import numpy as np
import pytest

from control_backends.force_ctrl import mit_loop as module
from control_backends.force_ctrl.data_types import JointCommands
from control_backends.force_ctrl.mit_loop import NeroMITLoop


def command(pos=0.0, kp=30.0, kd=2.0, vel=0.0, torque=0.0):
    return JointCommands(*(np.full(7, value) for value in (pos, vel, kp, kd, torque)))


class Robot:
    def __init__(self):
        self.batches = []
        self.stops = 0

    def set_motion_mode(self, mode):
        assert mode == "mit"

    def move_mit(self, **packet):
        self.batches.append(packet)

    def electronic_emergency_stop(self):
        self.stops += 1


def loop_with_sent_command():
    robot = Robot()
    loop = NeroMITLoop(robot, NS(compute=lambda q: np.ones(7)),
                       feedback_guard_enabled=False, joint_limit_buffer=0.0)
    loop._running = True
    loop._last_sent_snapshot = {
        "pos": np.full(7, 0.03), "vel": np.full(7, 0.02),
        "kp": np.full(7, 0.75), "kd": np.full(7, 0.5),
        "non_gravity_torque": np.full(7, 0.6),
    }
    loop._impedance_cfg = NS(active=True)
    return loop


def test_ramp_blends_pd_effort_without_stiffness_times_error_overshoot():
    loop = loop_with_sent_command()
    loop.hold_for_controlled_stop(command())
    start, complete = loop._stop_transition_command(loop._commands, 100.0)
    assert not complete
    np.testing.assert_allclose(start.kp, .75)
    np.testing.assert_allclose(start.pos, .03)
    np.testing.assert_allclose(start.torques, .6)
    middle, complete = loop._stop_transition_command(loop._commands, 100.2)
    q, dq = np.full(7, .001), np.full(7, .01)
    actual_pd = middle.kp * (middle.pos - q) + middle.kd * (middle.vel - dq)
    expected_pd = .5 * (.75 * (.03 - q) + .5 * (.02 - dq)) + .5 * (30 * -q + 2 * -dq)
    np.testing.assert_allclose(actual_pd, expected_pd, atol=1e-12)
    np.testing.assert_allclose(middle.kp * middle.pos, .5 * .75 * .03)
    np.testing.assert_allclose(middle.torques, .3)
    assert not complete
    end, complete = loop._stop_transition_command(loop._commands, 100.5)
    np.testing.assert_allclose(end.pos, 0)
    np.testing.assert_allclose(end.vel, 0)
    np.testing.assert_allclose(end.kp, 30)
    np.testing.assert_allclose(end.torques, 0)
    assert complete
    # Computation alone is not acknowledgement of a successful CAN batch.
    assert not loop.get_controlled_stop_status()["transition_complete"]


def test_zero_initial_kp_and_kd_remain_finite_and_cannot_reintroduce_target():
    loop = loop_with_sent_command()
    loop._last_sent_snapshot["kp"][:] = 0
    loop._last_sent_snapshot["kd"][:] = 0
    loop.hold_for_controlled_stop(command())
    for now in (100.0, 100.1, 100.2, 100.5):
        out, _ = loop._stop_transition_command(loop._commands, now)
        assert all(np.isfinite(getattr(out, key)).all() for key in ("pos", "vel", "kp", "kd", "torques"))
        np.testing.assert_allclose(out.kp * out.pos, 0)
    with pytest.raises(RuntimeError, match="latched"):
        loop.update_commands(command(pos=.2))


def test_inner_loop_keeps_gravity_and_acknowledges_only_completed_final_batch(monkeypatch):
    clock = NS(now=100.0)
    monkeypatch.setattr(module, "time", NS(time=lambda: clock.now, monotonic=lambda: clock.now,
                                           perf_counter=lambda: clock.now))
    loop = loop_with_sent_command()
    loop._impedance_cfg.active = False
    loop._read_joint_pos = lambda: np.zeros(7)
    loop._read_joint_torque = lambda: np.zeros(7)
    loop.update_commands(command(pos=.03, kp=.75, kd=.5, vel=.02, torque=.6))
    observations = []

    def step(_deadline):
        observations.append(loop.get_controlled_stop_status()["transition_complete"])
        if len(observations) == 1:
            # Snapshot must describe the just-sent batch, before stop latching.
            np.testing.assert_allclose(loop._last_sent_snapshot["non_gravity_torque"], .6)
            loop._impedance_cfg.active = True
            loop.hold_for_controlled_stop(command())
        if len(observations) >= 6:
            loop._running = False
        clock.now += .2

    loop._sleep_until = step
    loop._control_loop()
    assert observations[:3] == [False, False, False]
    assert observations[-1] is True
    packets = loop._robot.batches
    np.testing.assert_allclose([p["t_ff"] for p in packets[:7]], 1.6)
    np.testing.assert_allclose([p["t_ff"] for p in packets[7:14]], 1.6)
    np.testing.assert_allclose([p["t_ff"] for p in packets[-7:]], 1.0)
    np.testing.assert_allclose([p["p_des"] for p in packets[-7:]], 0.0)
    assert loop._robot.stops == 0


def test_shared_status_has_no_stop_target_before_latch():
    loop = loop_with_sent_command()
    assert loop.get_controlled_stop_status()["target_q"] is None


def test_partial_final_batch_never_acknowledges_hold_or_replaces_complete_snapshot():
    loop = loop_with_sent_command()
    loop._impedance_cfg.active = False
    old_snapshot = loop._last_sent_snapshot
    loop.hold_for_controlled_stop(command())
    loop._read_joint_pos = lambda: np.zeros(7)
    loop._read_joint_torque = lambda: np.zeros(7)

    def fail_mid_batch(**packet):
        if packet["joint_index"] == 4:
            raise RuntimeError("synthetic CAN send failure")

    loop._robot.move_mit = fail_mid_batch
    loop._control_loop()
    assert not loop.get_controlled_stop_status()["transition_complete"]
    assert loop._last_sent_snapshot is old_snapshot
    assert loop.terminal_fault


def test_stop_latch_prevents_gravity_changes_during_transition():
    loop = loop_with_sent_command()
    loop.hold_for_controlled_stop(command())
    with pytest.raises(RuntimeError, match="Cannot change gravity"):
        loop.set_gravity_enabled(False)
    with pytest.raises(RuntimeError, match="Cannot change gravity"):
        loop.set_gravity_factors(np.zeros(7))
    assert loop._gravity_enabled


def test_control_diagnostics_reuses_existing_sender_logger(tmp_path):
    loop = loop_with_sent_command()
    original = tmp_path / "grasp" / "cycles.jsonl"
    try:
        first = loop.ensure_control_diagnostics(original)
        writer = loop._diagnostics_writer
        second = loop.ensure_control_diagnostics(tmp_path / "servo" / "unused.jsonl")
        assert loop._diagnostics_writer is writer
        assert first["path"] == second["path"] == str(original)
        assert not (tmp_path / "servo").exists()
    finally:
        loop.stop_diagnostics()
