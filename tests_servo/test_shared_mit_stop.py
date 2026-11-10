"""Shared stop checks use raw grouped feedback; no driver or camera is opened."""

from types import SimpleNamespace as NS

import numpy as np
import pytest

import control_backends.force_ctrl.normal_shutdown as shared


class Clock:
    def __init__(self):
        self.now = 1000.0

    def time(self):
        return self.now

    def monotonic(self):
        return self.now

    def sleep(self, delay):
        self.now += delay


class Robot:
    def __init__(self, clock):
        self.clock = clock
        self.q = np.zeros(7)
        self.q_source = lambda: self.q.copy()
        self.stamp_source = lambda: np.full(14, self.clock.now)

    @property
    def _parser(self):
        q, stamps = self.q_source(), self.stamp_source()
        parser = NS()
        for name, axes in (("joint_12", (0, 1)), ("joint_34", (2, 3)),
                           ("joint_56", (4, 5)), ("joint_7", (6,))):
            setattr(parser, name, NS(timestamp=stamps[axes[0]], msg=NS(
                **{f"joint_{axis + 1}": q[axis] for axis in axes})))
        for axis in range(7):
            setattr(parser, f"motor_state_{axis + 1}", NS(
                timestamp=stamps[7 + axis], msg=NS(velocity=0.0, torque=0.0)))
        return parser


class Loop:
    running = True

    def __init__(self, clock, duration=0.4):
        self.clock, self.duration = clock, duration
        self.target = None
        self.request_time = None
        self.holds = []
        self.never_acknowledge = False

    def hold_for_controlled_stop(self, command):
        if self.target is None:
            self.target = command.pos.copy()
            self.request_time = self.clock.now
            self.holds.append(command)
            return True
        return False

    def get_controlled_stop_status(self):
        completed = (self.target is not None and not self.never_acknowledge
                     and self.clock.now >= self.request_time + self.duration)
        return {"target_q": self.target, "transition_duration_s": self.duration,
                "transition_complete": completed,
                "transition_completed_monotonic": (
                    self.request_time + self.duration if completed else None),
                "phase": "hold" if completed else "transition"}


@pytest.fixture
def rig(monkeypatch):
    clock = Clock()
    monkeypatch.setattr(shared, "time", clock)
    robot, loop = Robot(clock), Loop(clock)
    stop = shared.MITControlledStop(robot, loop, np.full(7, 30.0), np.full(7, 1.0),
                                    check_fault=lambda: None)
    return clock, robot, loop, stop


def test_stable_feedback_cannot_finish_before_final_transition_command(rig):
    clock, _, loop, stop = rig
    stop.request()
    result = stop.wait()
    assert result["confirmed"] is True
    assert clock.now >= loop.request_time + loop.duration + 0.059
    assert result["last_joint_delta_deg"] == 0.0
    assert stop.phase == "confirmed"


def test_request_is_idempotent_and_keeps_original_stop_target(rig):
    _, robot, loop, stop = rig
    first = stop.request()
    robot.q[:] = 0.001
    np.testing.assert_array_equal(stop.request(), first)
    assert len(loop.holds) == 1


def test_existing_software_buffer_hold_is_not_replaced(rig):
    clock, robot, loop, stop = rig
    loop.target = np.full(7, 0.001)
    loop.request_time = clock.now
    robot.q[:] = 0.002
    np.testing.assert_array_equal(stop.request(), loop.target)
    assert loop.holds == []


def test_fresh_stationary_feedback_cannot_replace_missing_transition_ack(rig):
    clock, _, loop, stop = rig
    loop.never_acknowledge = True
    with pytest.raises(RuntimeError, match="final command was not acknowledged"):
        stop.wait()
    assert clock.now - loop.request_time > loop.duration + 0.5


def test_settling_timeout_starts_after_full_transition(rig):
    clock, robot, loop, stop = rig
    robot.q_source = lambda: np.full(7, np.deg2rad(0.1 if int(round(clock.now * 50)) % 2 else 0.0))
    with pytest.raises(RuntimeError, match="settling was not confirmed"):
        stop.wait(timeout_s=0.12)
    assert clock.now - loop.request_time >= loop.duration + 0.12


@pytest.mark.parametrize("frozen_axis", [0, 6, 7, 13])
def test_any_stale_group_or_motor_frame_blocks_controlled_stop(rig, frozen_axis):
    clock, robot, _, stop = rig
    frozen = clock.now
    def stamps():
        result = np.full(14, clock.now)
        result[frozen_axis] = frozen
        return result
    robot.stamp_source = stamps
    with pytest.raises(RuntimeError, match="stale|skew"):
        stop.wait()


def test_identical_cached_frames_do_not_confirm_stop(rig):
    clock, robot, loop, stop = rig
    loop.duration = 0.0
    stamp = clock.now
    robot.stamp_source = lambda: np.full(14, stamp)
    with pytest.raises(RuntimeError, match="stale"):
        stop.wait()
    assert stop.settling.stable_samples == 0


def test_drift_blocks_stop_confirmation(rig):
    _, robot, _, stop = rig
    stop.request()
    robot.q[0] = np.deg2rad(1.1)
    with pytest.raises(RuntimeError, match="drift exceeded"):
        stop.wait()


def test_fault_or_dead_worker_never_sends_a_hold(rig):
    _, _, loop, stop = rig
    loop.running = False
    with pytest.raises(RuntimeError, match="not running"):
        stop.request()
    assert loop.holds == []


def test_fault_during_transition_blocks_completion(rig):
    clock, _, loop, stop = rig
    def check():
        if loop.request_time is not None and clock.now - loop.request_time > 0.04:
            raise RuntimeError("injected terminal feedback fault")
    stop.check_fault = check
    with pytest.raises(RuntimeError, match="injected terminal"):
        stop.wait()
    assert stop.phase != "confirmed"


def test_latched_target_cannot_change_during_wait(rig):
    _, _, loop, stop = rig
    stop.request()
    loop.target = np.ones(7)
    with pytest.raises(RuntimeError, match="target changed"):
        stop.wait()


def _sample(q, position, motor=None):
    return {"q": np.asarray(q, dtype=float),
            "position_timestamps": np.asarray(position, dtype=float),
            "motor_timestamps": np.asarray(position if motor is None else motor, dtype=float)}


def test_settling_uses_each_joint_interval_not_oldest_frame_interval():
    detector = shared.MITSettlingCheck(np.zeros(7))
    detector.update(_sample(np.zeros(7), np.ones(7)))
    stamps = np.full(7, 1.1)
    stamps[6] = 1.01
    q = np.zeros(7)
    q[6] = np.deg2rad(0.05)
    assert not detector.update(_sample(q, stamps))
    assert detector.last_speed_deg_s == pytest.approx(5.0)
    assert detector.stable_samples == 0


def test_all_position_and_motor_frames_must_advance_before_counting():
    detector = shared.MITSettlingCheck(np.zeros(7))
    detector.update(_sample(np.zeros(7), np.ones(7)))
    for stamp in (1.02, 1.04, 1.06):
        motors = np.full(7, stamp)
        motors[6] = 1.0
        assert not detector.update(_sample(np.zeros(7), np.full(7, stamp), motors))
    assert detector.stable_samples == 0
    for stamp in (1.08, 1.10, 1.12):
        confirmed = detector.update(_sample(np.zeros(7), np.full(7, stamp)))
    assert confirmed


def test_settling_rejects_regressing_per_axis_timestamp():
    detector = shared.MITSettlingCheck(np.zeros(7))
    detector.update(_sample(np.zeros(7), np.full(7, 2.0)))
    stamps = np.full(7, 2.1)
    stamps[6] = 1.99
    with pytest.raises(RuntimeError, match="went backwards"):
        detector.update(_sample(np.zeros(7), stamps))
