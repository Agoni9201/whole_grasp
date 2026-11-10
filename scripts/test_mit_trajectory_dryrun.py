#!/usr/bin/env python3
"""Offline dry-run tests for the unified MIT trajectory session (no hardware).

Covers:
  1. Quintic time-parameterization: boundary conditions, velocity/acceleration
     limits, endpoint arrival (pure functions, no robot).
  2. MITTrajectorySession.execute_stages() with a fake driver + fake MIT loop:
       - successful tracking -> arrival gate passes -> loop KEEPS running (no
         move_j restore);
       - the session is persistent: two routes share one loop / one start;
       - injected tracking error trips the tracking-error protection;
       - a wrong flange pose fails the arrival gate (MotionNotReached);
       - q(t) logging writes a JSONL file.
  3. Joint-limit buffer plumbing and pre-execution route rejection.
  4. Background MIT terminal faults latch the workflow safety stop immediately.

Run with the project interpreter, e.g.:
    /home/yx/miniconda3/envs/grasp_yolo26/bin/python scripts/test_mit_trajectory_dryrun.py
"""

import os
import sys
import time
import json
import shutil
import tempfile
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import real_nero_grasp.mit_trajectory_executor as executor_mod
from real_nero_grasp.trajectory_time_param import (
    build_joint_trajectory,
    time_parameterize_segment,
    quintic_s,
    quintic_s_dot,
    quintic_s_ddot,
)
from real_nero_grasp.types import (
    GraspMotionPlan,
    MotionNotReached,
    MotionSafetyScene,
    PlannedJointStage,
    SafetyStopTriggered,
)
from control_backends.force_ctrl.data_types import JointState


# --------------------------------------------------------------------------- #
# Fake plant / driver / MIT loop (no CAN, no mujoco, no pinocchio)
# --------------------------------------------------------------------------- #

class FakePlant:
    def __init__(self, start_joints, flange_pose, tracking_error_rad=0.0):
        self.q = np.asarray(start_joints, dtype=np.float64).reshape(7).copy()
        self.flange = np.asarray(flange_pose, dtype=np.float64).reshape(6).copy()
        self.tracking_error = float(tracking_error_rad)
        self.dq = np.zeros(7)

    def follow(self, q_des, dq_des):
        self.q = np.asarray(q_des, dtype=np.float64).reshape(7) + self.tracking_error
        self.dq = np.asarray(dq_des, dtype=np.float64).reshape(7)

    def joint_state(self):
        return JointState(
            pos=self.q.copy(), vel=self.dq.copy(),
            torque=np.zeros(7), timestamp=time.time(),
        )


class FakeDriver:
    def __init__(self):
        self.set_normal_called = False
        self.enable_called = False

    def set_normal_mode(self):
        self.set_normal_called = True

    def enable(self):
        self.enable_called = True
        return True

    def set_auto_set_motion_mode_enabled(self, value):
        pass


class FakeArm:
    def __init__(self, plant):
        self.arm = FakeDriver()
        self.plant = plant
        self.move_j_calls = 0

    def get_joint_angles(self):
        return self.plant.q.copy()

    def get_flange_pose(self):
        return self.plant.flange.copy()

    def _send_ordinary_trajectory_terminate(self, frames=3):
        return frames

    def _wait_for_stationary_joints(self, timeout_s=1.5):
        return self.plant.q.copy(), 0.0

    def move_j(self, q):
        self.plant.q = np.asarray(q, dtype=np.float64).reshape(7).copy()
        self.move_j_calls += 1


_PLANT = None


class FakeMITLoop:
    def __init__(self, **kwargs):
        self._running = False
        self._terminal_fault = False
        self._last_error = None
        self.joint_limit_buffer = kwargs.get("joint_limit_buffer")
        self.start_calls = 0
        assert _PLANT is not None
        self.plant = _PLANT

    @property
    def running(self):
        return self._running

    @property
    def terminal_fault(self):
        return self._terminal_fault

    @property
    def last_error(self):
        return self._last_error

    def start(self):
        self._running = True
        self._terminal_fault = False
        self._last_error = None
        self.start_calls += 1

    def stop(self):
        self._running = False

    def start_diagnostics(self, path, queue_capacity=4096):
        pass

    def set_diagnostics_phase(self, label):
        pass

    def update_commands(self, commands):
        if self._running:
            self.plant.follow(commands.pos, commands.vel)

    def get_joint_state(self):
        return self.plant.joint_state() if self._running else None


def _install_fakes():
    executor_mod.NeroMITLoop = FakeMITLoop
    executor_mod.NeroGravityCompensator = lambda *a, **k: None
    executor_mod._build_pin_model = lambda: None  # avoid pinocchio import


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #

def make_args(**overrides):
    base = dict(
        trajectory_executor="mit",
        mit_kp=[30.0, 30.0, 30.0, 15.0, 10.0, 5.0, 5.0],
        mit_kd=[2.0, 2.0, 2.0, 1.0, 0.5, 0.3, 0.3],
        mit_max_joint_vel_deg_s=57.3,
        mit_max_joint_accel_deg_s2=286.5,
        mit_tracking_error_deg=1.0,
        mit_tracking_error_confirm_samples=3,
        mit_feedback_timeout_s=0.5,
        mit_torque_limit=14.0,
        mit_joint_limit_buffer_deg=0.0,
        mit_joint_limit_planning_slack_deg=1.0,
        nbv_arrival_joint_error_deg=1.0,
        nbv_arrival_vel_deg_s=5.0,
        nbv_arrival_pose_error_mm=8.0,
        nbv_arrival_stable_samples=3,
        nbv_arrival_timeout_s=1.0,
        mit_q_log=True,
        mit_q_log_dir=None,
        save_dir=None,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def make_stage(label, q_goal, flange_pose):
    return PlannedJointStage(
        label=label,
        tcp_pose=np.zeros(6),
        flange_pose=np.asarray(flange_pose, dtype=np.float64).reshape(6),
        joints=np.asarray(q_goal, dtype=np.float64).reshape(7),
        phase="free_open",
    )


def make_plan(q_start, q_goal, flange_pose):
    stage = make_stage("active_center", q_goal, flange_pose)
    scene = MotionSafetyScene(
        full_scene_base=np.zeros((0, 3)),
        environment_base=np.zeros((0, 3)),
        target_base=np.zeros((0, 3)),
    )
    return GraspMotionPlan(
        start_joints=np.asarray(q_start, dtype=np.float64).reshape(7),
        before_close=(stage,),
        after_close=(),
        after_release=(),
        empty_recovery=(),
        safety_scene=scene,
    )


def _session(arm, args):
    global _PLANT
    _PLANT = arm.plant
    return executor_mod.get_mit_session(arm, args)


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #

def test_time_param():
    for tau in (0.0, 1.0):
        assert abs(quintic_s_dot(tau)) < 1e-12
        assert abs(quintic_s_ddot(tau)) < 1e-12
    assert abs(quintic_s(0.0)) < 1e-12 and abs(quintic_s(1.0) - 1.0) < 1e-12

    q0 = np.zeros(7)
    q1 = np.array([0.5, -0.4, 0.3, 0.2, -0.2, 0.1, 1.4])
    v_max, a_max = 0.5, 1.5
    seg = time_parameterize_segment(q0, q1, v_max, a_max)
    assert seg.duration_s > 0
    for t in np.linspace(0, seg.duration_s, 2001):
        dq = (seg.q_goal - seg.q_start) * quintic_s_dot(t / seg.duration_s) / seg.duration_s
        ddq = (seg.q_goal - seg.q_start) * quintic_s_ddot(t / seg.duration_s) / seg.duration_s ** 2
        assert np.max(np.abs(dq)) <= v_max + 1e-9
        assert np.max(np.abs(ddq)) <= a_max + 1e-9

    traj = build_joint_trajectory([q0, q1], v_max, a_max)
    qf, dqf, _ = traj.sample(traj.duration_s)
    assert np.allclose(qf, q1, atol=1e-9)
    assert np.allclose(dqf, 0.0, atol=1e-9)
    print("  [ok] test_time_param")


def test_executor_success_and_persistence(tmp_dir):
    q_start = np.array(
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, np.deg2rad(80.0)]
    )
    q_goal = q_start + np.array([0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    flange = np.array([0.4, 0.0, 0.3, 0.0, 0.0, 0.0])
    plan = make_plan(q_start, q_goal, flange)
    args = make_args(mit_q_log_dir=tmp_dir, save_dir=tmp_dir)

    plant = FakePlant(q_start, flange, tracking_error_rad=0.0)
    arm = FakeArm(plant)
    session = _session(arm, args)

    session.execute_stages(plan.before_close, plan.start_joints, args, "active_center")

    # loop keeps running (no move_j restore); move_j NOT re-issued on the arm
    assert session.running, "MIT loop must keep running after a route"
    assert arm.move_j_calls == 0, "move_j must not be issued in unified MIT mode"

    # second route on the same session -> same loop, no restart
    q_goal2 = q_goal - np.array([0.03, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    plan2 = make_plan(q_goal, q_goal2, flange)
    session.execute_stages(plan2.before_close, plan2.start_joints, args, "active_side")
    assert session.loop.start_calls == 1, "MIT loop must be started exactly once"

    # q(t) log written
    log_path = os.path.join(tmp_dir, "mit_trajectory_active_center.jsonl")
    assert os.path.isfile(log_path), f"missing q(t) log: {log_path}"
    with open(log_path) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    assert len(rows) >= 3
    assert abs(rows[-1]["tracking_error_deg"]) < 0.2
    print(f"  [ok] test_executor_success_and_persistence ({len(rows)} log rows)")


def test_executor_tracking_trip():
    q_start = np.zeros(7)
    q_goal = q_start + np.array([0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    flange = np.array([0.4, 0.0, 0.3, 0.0, 0.0, 0.0])
    plan = make_plan(q_start, q_goal, flange)
    args = make_args(mit_q_log=False)

    plant = FakePlant(q_start, flange, tracking_error_rad=0.05)  # ~2.86 deg > 1 deg
    arm = FakeArm(plant)
    session = _session(arm, args)
    try:
        session.execute_stages(plan.before_close, plan.start_joints, args, "active_center")
        raise AssertionError("expected MotionNotReached from tracking-error protection")
    except MotionNotReached as exc:
        assert "tracking error" in str(exc)
    # on error the loop keeps running (holding), and move_j is never issued
    assert session.running and arm.move_j_calls == 0
    print("  [ok] test_executor_tracking_trip")


def test_executor_arrival_gate_fail():
    q_start = np.zeros(7)
    q_goal = q_start + np.array([0.02, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    planned_flange = np.array([0.4, 0.0, 0.3, 0.0, 0.0, 0.0])
    plan = make_plan(q_start, q_goal, planned_flange)
    args = make_args(mit_q_log=False, nbv_arrival_timeout_s=0.4, nbv_arrival_stable_samples=2)

    wrong_flange = planned_flange + np.array([0.1, 0.0, 0.0, 0.0, 0.0, 0.0])
    plant = FakePlant(q_start, wrong_flange, tracking_error_rad=0.0)
    arm = FakeArm(plant)
    session = _session(arm, args)
    try:
        session.execute_stages(plan.before_close, plan.start_joints, args, "active_center")
        raise AssertionError("expected MotionNotReached from arrival gate")
    except MotionNotReached as exc:
        assert "arrival" in str(exc)
    assert session.running and arm.move_j_calls == 0
    print("  [ok] test_executor_arrival_gate_fail")


def test_joint_limit_buffer_plumbed():
    q_start = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.5708])
    flange = np.array([0.4, 0.0, 0.3, 0.0, 0.0, 0.0])
    args = make_args(mit_q_log=False, mit_joint_limit_buffer_deg=0.0)
    global _PLANT
    _PLANT = FakePlant(q_start, flange)
    arm = FakeArm(_PLANT)

    captured = {}

    class CapturingLoop(FakeMITLoop):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            captured["buffer"] = self.joint_limit_buffer

    executor_mod.NeroMITLoop = CapturingLoop
    executor_mod.MITTrajectorySession(arm, args)
    assert captured["buffer"] == 0.0, f"joint_limit_buffer={captured['buffer']!r}, expected 0.0"
    print("  [ok] test_joint_limit_buffer_plumbed")


def test_joint_limit_preflight_rejects_before_start():
    q_start = np.zeros(7)
    q_start[-1] = np.deg2rad(75.0)
    q_goal = q_start.copy()
    q_goal[-1] = np.deg2rad(85.0)
    flange = np.array([0.4, 0.0, 0.3, 0.0, 0.0, 0.0])
    args = make_args(
        mit_q_log=False,
        mit_joint_limit_buffer_deg=5.0,
        mit_tracking_error_deg=2.0,
        mit_joint_limit_planning_slack_deg=1.0,
    )
    plant = FakePlant(q_start, flange)
    arm = FakeArm(plant)
    session = _session(arm, args)
    stage = make_stage("active_side", q_goal, flange)

    try:
        session.execute_stages((stage,), q_start, args, "active_side")
        raise AssertionError("expected joint-limit preflight rejection")
    except MotionNotReached as exc:
        assert "joint-limit preflight rejected" in str(exc)
        assert "no motion command was sent" in str(exc)
    assert session.loop.start_calls == 0
    assert not session.running
    print("  [ok] test_joint_limit_preflight_rejects_before_start")


def test_terminal_fault_propagates_and_latches_arm():
    q_start = np.zeros(7)
    q_goal = q_start.copy()
    q_goal[0] = 0.02
    flange = np.array([0.4, 0.0, 0.3, 0.0, 0.0, 0.0])
    args = make_args(mit_q_log=False)
    plant = FakePlant(q_start, flange)
    arm = FakeArm(plant)
    session = _session(arm, args)
    session.ensure_started()
    session.loop._terminal_fault = True
    session.loop._last_error = "Joint limit exceeded, emergency stop"
    session.loop._running = False
    stage = make_stage("active_side", q_goal, flange)

    try:
        session.execute_stages((stage,), q_start, args, "active_side")
        raise AssertionError("expected terminal MIT fault propagation")
    except SafetyStopTriggered as exc:
        assert "MIT terminal safety fault" in str(exc)
    assert arm._safety_stop_latched
    assert arm._safety_stop_request_confirmed
    print("  [ok] test_terminal_fault_propagates_and_latches_arm")


def main():
    _install_fakes()
    tmp_dir = tempfile.mkdtemp(prefix="mit_traj_test_")
    try:
        test_time_param()
        test_executor_success_and_persistence(tmp_dir)
        test_executor_tracking_trip()
        test_executor_arrival_gate_fail()
        test_joint_limit_buffer_plumbed()
        test_joint_limit_preflight_rejects_before_start()
        test_terminal_fault_propagates_and_latches_arm()
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
    print("\nALL MIT-TRAJECTORY DRY-RUN TESTS PASSED")


if __name__ == "__main__":
    main()
