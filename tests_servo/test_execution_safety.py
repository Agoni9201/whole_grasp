"""Tests for the low-speed execute safety monitor."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from servo.execution_safety import ExecutionSafetyMonitor, SafetyParams


LIMITS = np.array([[-1.0, 1.0]] * 7, dtype=np.float64)


def _monitor() -> ExecutionSafetyMonitor:
    return ExecutionSafetyMonitor(
        SafetyParams(
            startup_limit_margin=0.1,
            runtime_limit_margin=0.05,
            max_tracking_error=0.2,
            max_tracking_error_s=1.0,
        ),
        joint_limits=LIMITS,
    )


def test_start_posture_rejects_near_limit():
    monitor = _monitor()
    ok, reason = monitor.check_start_posture(np.array([0.95, 0, 0, 0, 0, 0, 0]))
    assert not ok
    assert "joint1" in reason


def test_start_posture_accepts_centered_joint_angles():
    monitor = _monitor()
    ok, reason = monitor.check_start_posture(np.zeros(7))
    assert ok
    assert reason == ""


def test_runtime_holds_on_target_lost():
    monitor = _monitor()
    decision = monitor.evaluate(
        np.zeros(7),
        {"target_lost": True, "target_rejected": False, "error": np.zeros(3)},
        now=1.0,
    )
    assert decision.hold
    assert decision.status == "marker_lost_hold"


def test_runtime_holds_on_near_limit():
    monitor = _monitor()
    decision = monitor.evaluate(
        np.array([0.98, 0, 0, 0, 0, 0, 0]),
        {"target_lost": False, "target_rejected": False, "error": np.zeros(3)},
        now=1.0,
    )
    assert decision.hold
    assert not decision.fatal
    assert decision.status == "near_joint_limit_hold"


def test_runtime_fatal_on_hard_limit_violation():
    monitor = _monitor()
    decision = monitor.evaluate(
        np.array([1.01, 0, 0, 0, 0, 0, 0]),
        {"target_lost": False, "target_rejected": False, "error": np.zeros(3)},
        now=1.0,
    )
    assert decision.hold
    assert decision.fatal
    assert decision.status == "hard_joint_limit_stop"


def test_runtime_holds_after_large_error_duration():
    monitor = _monitor()
    servo_out = {
        "target_lost": False,
        "target_rejected": False,
        "error": np.array([0.3, 0.0, 0.0]),
    }
    first = monitor.evaluate(np.zeros(7), servo_out, now=1.0)
    second = monitor.evaluate(np.zeros(7), servo_out, now=2.1)
    assert not first.hold
    assert second.hold
    assert second.status == "large_error_hold"


def test_large_error_hold_can_be_disabled():
    monitor = ExecutionSafetyMonitor(
        SafetyParams(
            startup_limit_margin=0.1,
            runtime_limit_margin=0.05,
            max_tracking_error=0.0,
            max_tracking_error_s=1.0,
        ),
        joint_limits=LIMITS,
    )
    servo_out = {
        "target_lost": False,
        "target_rejected": False,
        "error": np.array([10.0, 0.0, 0.0]),
    }
    first = monitor.evaluate(np.zeros(7), servo_out, now=1.0)
    second = monitor.evaluate(np.zeros(7), servo_out, now=20.0)
    assert not first.hold
    assert not second.hold
    assert second.status == "tracking"
