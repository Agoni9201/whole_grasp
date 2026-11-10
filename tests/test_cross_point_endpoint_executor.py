"""Offline endpoint execution checks: one move_j, not one per FK sample."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from tools import cross_point_check as cross


def fk(joints):
    return cross.pose_transform(np.asarray(joints, float)[:6])


class Clock:
    def __init__(self):
        self.elapsed = 0.0

    def time(self):
        return 1_800_000_000.0 + self.elapsed

    def monotonic(self):
        return self.elapsed

    def sleep(self, seconds):
        self.elapsed += max(float(seconds), 0.001)


class StreamingArm:
    """Fresh feedback progresses along the sent joint-space segment."""

    def __init__(self, clock, *, stall=False, sideways=False):
        self.clock = clock
        self.q = np.array([-0.4, 0.0, 0.25, 0, 0, 0, 0], dtype=float)
        self.start, self.target = self.q.copy(), None
        self.progress, self.stall, self.sideways = 0.0, stall, sideways
        self.commands = []
        self.arm = SimpleNamespace(
            get_flange_pose=self.pose, get_joint_angles=self.joints,
        )

    def pose(self):
        if self.target is not None:
            self.progress = min(1.0, self.progress + 0.2)
            fraction = 0.2 if self.stall else self.progress
            self.q = self.start + fraction * (self.target - self.start)
            if self.sideways:
                self.q[1] += 0.08
        return SimpleNamespace(msg=self.q[:6].copy(), timestamp=self.clock.time())

    def joints(self):
        return SimpleNamespace(msg=self.q.copy(), timestamp=self.clock.time())

    def move_j(self, target):
        self.start = self.q.copy()
        self.target = np.asarray(target).copy()
        self.progress = 0.0
        self.commands.append(self.target.copy())

    def _controller_motion_active(self):
        return self.target is not None and self.progress < 1.0


@pytest.fixture
def endpoint_case(monkeypatch):
    clock = Clock()
    monkeypatch.setattr(cross.time, "time", clock.time)
    monkeypatch.setattr(cross.time, "monotonic", clock.monotonic)
    monkeypatch.setattr(cross.time, "sleep", clock.sleep)
    args = SimpleNamespace(
        move_wait_timeout=0.2, min_x=-0.8, max_x=0.8,
        min_y=-0.6, max_y=0.6, min_z=0.02, max_z=1.2,
    )
    return clock, args


def test_inflight_feedback_is_compared_to_joint_segment_not_endpoint(endpoint_case):
    clock, args = endpoint_case
    arm = StreamingArm(clock)
    start = arm.q.copy()
    target = start + np.array([0.1, 0.04, 0.05, 0, 0, 0, 0.1])
    result = {"motion_sent": False, "samples": []}

    cross._execute_cross_steps(
        arm, [(target, fk(target))], np.eye(4), result, "to_hover",
        source=(start, fk(start)), fk=fk, args=args,
    )

    assert len(arm.commands) == 1
    assert result["motion_sent"] is True
    np.testing.assert_allclose(arm.q, target)
    assert len(result["samples"]) == 1
    assert result["samples"][0]["phase"] == "to_hover"
    np.testing.assert_allclose(result["samples"][0]["actual_joints_rad"], target)


def test_timeout_records_endpoint_and_actual_values_before_raising(endpoint_case):
    clock, args = endpoint_case
    args.move_wait_timeout = 0.05
    arm = StreamingArm(clock, stall=True)
    start = arm.q.copy()
    target = start + np.array([0.1, 0.04, 0.05, 0, 0, 0, 0.1])
    result = {"motion_sent": False, "samples": []}

    with pytest.raises(RuntimeError):
        cross._execute_cross_steps(
            arm, [(target, fk(target))], np.eye(4), result, "to_hover",
            source=(start, fk(start)), fk=fk, args=args,
        )

    assert len(arm.commands) == 1
    sample = result["failure_sample"]
    assert sample["phase"] == "to_hover"
    np.testing.assert_allclose(sample["planned_joints_rad"], target)
    np.testing.assert_allclose(sample["actual_joints_rad"], arm.q)
    np.testing.assert_allclose(sample["planned_tcp_xyz_m"], target[:3])
    np.testing.assert_allclose(sample["actual_tcp_xyz_m"], arm.q[:3])
    assert sample["joint_error_deg"] > 0
    assert sample["position_error_mm"] > 0


def test_feedback_off_verified_joint_segment_is_rejected(endpoint_case):
    clock, args = endpoint_case
    arm = StreamingArm(clock, sideways=True)
    start = arm.q.copy()
    target = start + np.array([0.1, 0, 0, 0, 0, 0, 0.1])
    result = {"motion_sent": False, "samples": []}

    with pytest.raises(RuntimeError):
        cross._execute_cross_steps(
            arm, [(target, fk(target))], np.eye(4), result, "to_hover",
            source=(start, fk(start)), fk=fk, args=args,
        )

    assert len(arm.commands) == 1


def test_unknown_controller_status_waits_for_sustained_stationary_feedback(endpoint_case):
    clock, args = endpoint_case
    args.move_wait_timeout = 0.7
    arm = StreamingArm(clock)
    arm._controller_motion_active = lambda: None
    start = arm.q.copy()
    target = start.copy()
    target[0] += 0.1
    result = dict(motion_sent=False, samples=[])
    cross._execute_cross_steps(arm, [(target, fk(target))], np.eye(4), result,
                               "to_hover", source=(start, fk(start)), fk=fk, args=args)
    assert clock.elapsed >= 0.25
    assert len(arm.commands) == 1
    np.testing.assert_allclose(arm.q, target)


def test_known_active_controller_cannot_pass_arrival_gate(endpoint_case):
    clock, args = endpoint_case
    arm = StreamingArm(clock)
    arm._controller_motion_active = lambda: True
    start = arm.q.copy()
    target = start.copy()
    target[0] += 0.01
    result = dict(motion_sent=False, samples=[])
    with pytest.raises(RuntimeError, match="未到位"):
        cross._execute_cross_steps(arm, [(target, fk(target))], np.eye(4), result,
                                   "to_hover", source=(start, fk(start)), fk=fk, args=args)
    assert result["failure_sample"]["controller_motion_active"] is True
    assert len(arm.commands) == 1


def test_stage_timeout_is_saved_and_never_sends_return(
    endpoint_case, tmp_path, monkeypatch,
):
    clock, args = endpoint_case
    args.execute = True
    args.nero_speed_percent = 1
    args.move_wait_timeout = 0.05
    args.tcp_offset = [0.0] * 6
    args.hover = 0.03
    args.cross_step_mm = 1.0
    arm = StreamingArm(clock, stall=True)
    stops = []
    arm.stop_for_program_exit = stops.append

    def solve_ik(target_position, target_wxyz, initial_guess, use_previous_solution):
        q = np.asarray(initial_guess).copy()
        q[:3] = target_position
        return q

    solver = SimpleNamespace(
        get_joint_limits=lambda: (np.full(7, -3.0), np.full(7, 3.0)),
        solve_ik=solve_ik,
    )
    point = np.array([-0.3, 0.0, 0.2])
    state = dict(
        observation_joints_rad=arm.q.copy(), flange_pose=arm.q[:6].copy(),
        P_base=point.copy(), out_dir=str(tmp_path),
        cross_kinematics=(solver, fk),
    )
    monkeypatch.setattr("builtins.input", lambda _: "MOVE")

    with pytest.raises(RuntimeError):
        cross.execute_cross_roundtrip(arm, args, state)

    assert len(arm.commands) == len(stops) == 1
    assert not state.get("cross_last_execution")
    report = json.loads((tmp_path / "cross_roundtrip.json").read_text())
    assert report["status"] == "failed"
    assert report["failure_sample"]["phase"] == "to_hover"
    assert report["failure_sample"]["position_error_mm"] > 0
