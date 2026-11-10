"""Offline regression tests for exclusive MIT stop ownership."""

import time
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest

from real_nero_grasp.arm import RealNeroGraspArm
from real_nero_grasp.mit_trajectory_executor import MITTrajectorySession


def _bare_arm():
    arm = object.__new__(RealNeroGraspArm)
    arm.arm = mock.Mock()
    arm.arm.get_joint_angles.return_value = np.zeros(7)
    arm._safety_stop_latched = False
    arm._safety_stop_request_confirmed = False
    arm._controlled_stop_request_confirmed = False
    arm._stop_confirm_timeout = 0.5
    return arm


def _assert_no_normal_motion(arm):
    arm.arm._send_msg.assert_not_called()
    arm.arm.set_speed_percent.assert_not_called()
    arm.arm.move_j.assert_not_called()


def _state(stamp, q=None):
    return SimpleNamespace(
        timestamp=stamp,
        pos=np.zeros(7) if q is None else np.asarray(q, dtype=float),
        vel=np.zeros(7),
        torque=np.zeros(7),
    )


class _Clock:
    def __init__(self):
        self.elapsed = 0.0
        self.epoch = time.time()

    def monotonic(self):
        return self.elapsed

    def time(self):
        return self.epoch + self.elapsed

    def sleep(self, seconds):
        self.elapsed += max(float(seconds), 0.001)


class _Loop:
    running = True
    terminal_fault = False
    last_error = None

    def __init__(self, states):
        self.states = iter(states)
        self.pending_state = next(self.states, None)
        self.last_state = None
        self.holds = []
        self.commands = []
        self.clock = None

    def get_joint_state(self):
        candidate = self.pending_state
        if candidate is not None:
            if self.clock is not None and candidate.timestamp > self.clock.time():
                return self.last_state
            self.last_state = candidate
            self.pending_state = next(self.states, None)
        return self.last_state

    def hold_for_controlled_stop(self, command):
        self.holds.append(command)

    def get_controlled_stop_status(self):
        return {'target_q': None if not self.holds else self.holds[0].pos.copy(),
                'transition_complete': bool(self.holds),
                'transition_duration_s': 0.0,
                'transition_completed_monotonic': 0.0,
                'phase': 'hold'}

    def update_commands(self, command):
        self.commands.append(command)

    def set_diagnostics_phase(self, phase):
        self.phase = phase


class _RawFeedbackDriver:
    """Expose scripted feedback through the same grouped cache as the SDK."""

    def __init__(self, driver, loop):
        self.driver, self.loop = driver, loop

    def __getattr__(self, name):
        return getattr(self.driver, name)

    @property
    def _parser(self):
        state = self.loop.get_joint_state()
        if state is None:
            return None
        parser = SimpleNamespace()
        for group, axes in (("joint_12", (0, 1)), ("joint_34", (2, 3)),
                            ("joint_56", (4, 5)), ("joint_7", (6,))):
            setattr(parser, group, SimpleNamespace(timestamp=state.timestamp,
                msg=SimpleNamespace(**{f"joint_{axis + 1}": state.pos[axis] for axis in axes})))
        for axis in range(7):
            setattr(parser, f"motor_state_{axis + 1}", SimpleNamespace(
                timestamp=state.timestamp, msg=SimpleNamespace(velocity=0., torque=0.)))
        return parser


def _session(arm, loop, clock=None):
    session = object.__new__(MITTrajectorySession)
    session._arm = arm
    session._driver = _RawFeedbackDriver(arm.arm, loop)
    loop.clock = clock
    session._loop = loop
    session._started = True
    session._owns_control = True
    session._feedback_timeout_s = 0.5
    session._kp = np.full(7, 5.0)
    session._kd = np.full(7, 0.3)
    session._period = 0.005
    session._phase = "hold"
    return session


def test_arm_termination_delegates_to_mit_without_normal_commands():
    arm = _bare_arm()
    result = {"confirmed": True, "method": "mit_hold"}
    owner = SimpleNamespace(
        owns_control=True,
        terminate_motion=mock.Mock(return_value=result),
    )
    arm._mit_session_owner = owner

    assert arm.terminate_motion(fallback_joints=np.ones(7)) == result

    assert arm._safety_stop_latched is True
    owner.terminate_motion.assert_called_once()
    _assert_no_normal_motion(arm)


def test_failed_mit_stop_never_falls_back_to_normal_joint_commands():
    arm = _bare_arm()
    arm._mit_session_owner = SimpleNamespace(
        owns_control=True,
        terminate_motion=mock.Mock(side_effect=RuntimeError("MIT feedback stale")),
    )

    with pytest.raises(RuntimeError, match="MIT feedback stale"):
        arm.terminate_motion(fallback_joints=np.zeros(7))

    assert arm._safety_stop_latched is True
    _assert_no_normal_motion(arm)


def test_exit_confirms_mit_stop_before_stopping_sender():
    arm = _bare_arm()
    events = []

    def terminate_motion(**_kwargs):
        events.append("mit_hold_confirmed")
        arm._controlled_stop_request_confirmed = True
        return {"confirmed": True, "method": "mit_hold"}

    owner = SimpleNamespace(
        owns_control=True,
        terminate_motion=terminate_motion,
        shutdown=lambda: events.append("sender_shutdown"),
    )
    arm._mit_session_owner = owner
    callback = mock.Mock()
    arm._register_pre_disconnect(callback)

    assert arm.stop_for_program_exit("test exit") is True

    assert events == ["mit_hold_confirmed", "sender_shutdown"]
    callback.assert_not_called()
    _assert_no_normal_motion(arm)
    arm.arm.electronic_emergency_stop.assert_not_called()


def test_emergency_stop_is_owned_by_mit_not_raw_driver():
    arm = _bare_arm()
    owner = SimpleNamespace(owns_control=True, emergency_stop=mock.Mock())
    arm._mit_session_owner = owner

    arm.stop()

    owner.emergency_stop.assert_called_once()
    arm.arm.electronic_emergency_stop.assert_not_called()
    _assert_no_normal_motion(arm)


def test_mit_owner_blocks_ordinary_joints_but_not_gripper_commands():
    arm = _bare_arm()
    arm._mit_session_owner = SimpleNamespace(owns_control=True)
    arm.gripper = mock.Mock()
    arm.gripper_max_width = 0.085
    arm.gripper_max_force = 0.5

    with pytest.raises(RuntimeError, match="MIT"):
        arm.move_j(np.zeros(7))
    with pytest.raises(RuntimeError, match="MIT"):
        arm.move_p(np.zeros(6))
    arm.open_gripper_nowait(0.08, 0.5)

    arm.gripper.move_gripper_m.assert_called_once_with(value=0.08, force=0.5)
    _assert_no_normal_motion(arm)


def test_mit_emergency_rpc_failure_remains_retryable():
    arm = _bare_arm()
    owner = SimpleNamespace(
        owns_control=True,
        emergency_stop=mock.Mock(side_effect=[RuntimeError("CAN unavailable")] * 3 + [None]),
    )
    arm._mit_session_owner = owner

    with pytest.raises(RuntimeError, match="CAN unavailable"):
        arm.stop()
    assert arm._safety_stop_request_confirmed is False

    arm.stop()

    assert owner.emergency_stop.call_count == 4
    assert arm._safety_stop_request_confirmed is True
    arm.arm.electronic_emergency_stop.assert_not_called()
    _assert_no_normal_motion(arm)


def test_confirmed_stop_never_disconnects_if_mit_sender_join_fails():
    arm = _bare_arm()
    arm._controlled_stop_request_confirmed = True
    owner = SimpleNamespace(
        owns_control=True,
        running=False,  # stop() clears this before joining; it is not proof of exit.
        shutdown=mock.Mock(side_effect=RuntimeError("MIT sender is still alive")),
        emergency_stop=mock.Mock(side_effect=RuntimeError("MIT sender is still alive")),
    )
    arm._mit_session_owner = owner
    arm._register_pre_disconnect(owner.shutdown)

    try:
        arm.__exit__(None, None, None)
    except RuntimeError:
        pass

    owner.shutdown.assert_called()
    arm.arm.disconnect.assert_not_called()
    _assert_no_normal_motion(arm)


def test_confirmed_mit_exit_joins_sender_before_driver_disconnect():
    arm = _bare_arm()
    arm._controlled_stop_request_confirmed = True
    events = []
    owner = SimpleNamespace(
        owns_control=True,
        shutdown=lambda: events.append("sender_joined"),
    )
    arm._mit_session_owner = owner
    arm.arm.disconnect.side_effect = lambda: events.append("driver_disconnected")

    arm.__exit__(None, None, None)

    assert events == ["sender_joined", "driver_disconnected"]
    _assert_no_normal_motion(arm)


def test_torque_guard_read_preserves_shared_parser_velocity():
    arm = _bare_arm()
    parser = SimpleNamespace()
    for index in range(1, 8):
        setattr(parser, f"motor_state_{index}", SimpleNamespace(
            msg=SimpleNamespace(torque=float(index), velocity=0.25),
            timestamp=time.time(),
        ))
    calls = []

    def mutating_sdk_getter(index):
        calls.append(index)
        state = getattr(parser, f"motor_state_{index}")
        state.msg.velocity = 0.0
        return state

    arm.arm = SimpleNamespace(_parser=parser, get_motor_states=mutating_sdk_getter)

    np.testing.assert_allclose(arm.get_joint_torques(), np.arange(1.0, 8.0))

    assert calls == []
    assert all(getattr(parser, f"motor_state_{index}").msg.velocity == 0.25
               for index in range(1, 8))


def test_mit_stop_holds_once_and_confirms_only_new_feedback():
    arm = _bare_arm()
    clock = _Clock()
    q = np.linspace(-0.1, 0.1, 7)
    loop = _Loop([
        _state(clock.epoch, q),
        _state(clock.epoch, q),
        _state(clock.epoch + 0.05, q),
        _state(clock.epoch + 0.05, q),
        _state(clock.epoch + 0.15, q),
        _state(clock.epoch + 0.20, q),
        _state(clock.epoch + 0.25, q),
    ])
    session = _session(arm, loop, clock)

    with mock.patch("real_nero_grasp.mit_trajectory_executor.time", clock), \
            mock.patch("control_backends.force_ctrl.normal_shutdown.time", clock):
        result = session.terminate_motion(timeout_s=0.5)

    assert result["confirmed"] is True
    assert arm._controlled_stop_request_confirmed is True
    assert len(loop.holds) == 1
    np.testing.assert_allclose(loop.holds[0].pos, q)
    np.testing.assert_allclose(loop.holds[0].vel, 0.0)
    np.testing.assert_allclose(loop.holds[0].torques, 0.0)
    assert clock.elapsed >= 0.20 - 1e-12
    _assert_no_normal_motion(arm)


@pytest.mark.parametrize("bad_kind", ["missing", "nonfinite", "stale", "dead", "terminal"])
def test_mit_stop_rejects_invalid_initial_feedback_before_hold(bad_kind):
    arm = _bare_arm()
    clock = _Clock()
    initial = _state(clock.epoch)
    if bad_kind == "missing":
        initial = None
    elif bad_kind == "nonfinite":
        initial.pos[0] = np.nan
    elif bad_kind == "stale":
        initial.timestamp -= 1.0
    loop = _Loop([initial])
    if bad_kind == "dead":
        loop.running = False
    elif bad_kind == "terminal":
        loop.terminal_fault = True
        loop.last_error = "joint-limit terminal stop"
    session = _session(arm, loop, clock)

    with mock.patch("real_nero_grasp.mit_trajectory_executor.time", clock), \
            mock.patch("control_backends.force_ctrl.normal_shutdown.time", clock):
        with pytest.raises((RuntimeError, ValueError)):
            session.terminate_motion(timeout_s=0.3)

    assert loop.holds == []
    assert arm._controlled_stop_request_confirmed is False
    _assert_no_normal_motion(arm)


def test_frozen_feedback_cannot_confirm_stationary_mit_stop():
    arm = _bare_arm()
    clock = _Clock()
    loop = _Loop([_state(clock.epoch)])
    session = _session(arm, loop, clock)

    with mock.patch("real_nero_grasp.mit_trajectory_executor.time", clock), \
            mock.patch("control_backends.force_ctrl.normal_shutdown.time", clock):
        with pytest.raises(RuntimeError):
            session.terminate_motion(timeout_s=0.3)

    assert len(loop.holds) == 1
    assert arm._controlled_stop_request_confirmed is False
    _assert_no_normal_motion(arm)


def test_regressing_feedback_timestamp_cannot_confirm_mit_stop():
    arm = _bare_arm()
    clock = _Clock()
    loop = _Loop([
        _state(clock.epoch),
        _state(clock.epoch + 0.05),
        _state(clock.epoch - 0.01),
    ])
    session = _session(arm, loop, clock)

    with mock.patch("real_nero_grasp.mit_trajectory_executor.time", clock), \
            mock.patch("control_backends.force_ctrl.normal_shutdown.time", clock):
        with pytest.raises(RuntimeError):
            session.terminate_motion(timeout_s=0.3)

    assert arm._controlled_stop_request_confirmed is False
    _assert_no_normal_motion(arm)


def test_small_per_frame_motion_still_requires_low_derived_stop_speed():
    arm = _bare_arm()
    clock = _Clock()
    states = []
    for index in range(8):
        q = np.zeros(7)
        q[0] = np.deg2rad(0.1 * index)
        states.append(_state(clock.epoch + 0.05 * index, q))
    loop = _Loop(states)
    session = _session(arm, loop, clock)

    with mock.patch("real_nero_grasp.mit_trajectory_executor.time", clock), \
            mock.patch("control_backends.force_ctrl.normal_shutdown.time", clock):
        with pytest.raises(RuntimeError):
            session.terminate_motion(timeout_s=0.3)

    assert len(loop.holds) == 1
    assert arm._controlled_stop_request_confirmed is False
    _assert_no_normal_motion(arm)


def test_execute_exception_does_not_replace_already_latched_stop_hold():
    arm = _bare_arm()
    loop = _Loop([_state(time.time())])
    session = _session(arm, loop)
    session.ensure_started = mock.Mock()
    session._validate_joint_limit_preflight = mock.Mock()
    session.hold = mock.Mock()
    stages = [SimpleNamespace(label="target", joints=np.zeros(7), flange_pose=np.zeros(6))]
    args = SimpleNamespace(mit_q_log=False)

    def guard(arm, *_args, **_kwargs):
        arm._safety_stop_latched = True
        arm._controlled_stop_request_confirmed = True
        raise RuntimeError("guard confirmed controlled stop")

    with pytest.raises(RuntimeError, match="guard confirmed controlled stop"):
        session.execute_stages(stages, np.zeros(7), args, "target", runtime_guard=guard)

    session.hold.assert_not_called()
    assert len(loop.commands) == 1
    _assert_no_normal_motion(arm)


@pytest.mark.parametrize("phase,joint_error_deg,position_mm,start_match_deg,accepted", [
    ("free_open", 1.12, 2.28, 2.0, True),
    ("free_open", 2.01, 2.28, 2.0, False),
    ("free_open", 1.12, 2.28, 1.0, False),
    ("free_open", 1.12, 8.1, 2.0, False),
    ("target_approach", 1.12, 2.28, 2.0, False),
    ("support_release", 1.12, 2.28, 2.0, False),
    ("carrying", 1.0216638649, 1.84, 2.0, True),
    ("carrying", 2.01, 1.84, 2.0, False),
    ("carrying", 1.0216638649, 1.84, 1.0, False),
    ("carrying", 1.0216638649, 8.1, 2.0, False),
])
def test_grasp_transit_arrival_uses_motion_tolerance_and_preserves_other_gates(
    phase, joint_error_deg, position_mm, start_match_deg, accepted,
):
    from real_nero_grasp.types import MotionNotReached

    arm = _bare_arm()
    clock = _Clock()
    q = np.zeros(7)
    q[0] = np.deg2rad(joint_error_deg)
    loop = _Loop([_state(clock.epoch + .005 * i, q) for i in range(100)])
    session = _session(arm, loop, clock)
    session.ensure_started = mock.Mock()
    session._validate_joint_limit_preflight = mock.Mock()
    session.hold = mock.Mock()
    session._arm.get_joint_angles = mock.Mock(return_value=q)
    session._arm.get_flange_pose = mock.Mock(
        return_value=[position_mm / 1000, 0, 0, 0, 0, np.deg2rad(1.39)])
    stages = [SimpleNamespace(label="target_pregrasp", joints=np.zeros(7),
                              flange_pose=np.zeros(6), phase=phase)]
    args = SimpleNamespace(mit_q_log=False, nbv_arrival_timeout_s=.08,
                           move_joint_tol=np.deg2rad(2),
                           motion_stage_start_joint_match_deg=start_match_deg,
                           mit_tracking_error_deg=10)
    trajectory = SimpleNamespace(duration_s=0, segments=(None,),
                                 sample=lambda _: (np.zeros(7), np.zeros(7), np.zeros(7)))
    with mock.patch("real_nero_grasp.mit_trajectory_executor.time", clock), \
         mock.patch("real_nero_grasp.mit_trajectory_executor.build_joint_trajectory",
                    return_value=trajectory):
        if accepted:
            session.execute_stages(stages, np.zeros(7), args, "target_pregrasp")
        else:
            with pytest.raises(MotionNotReached, match="MIT arrival gate failed"):
                session.execute_stages(stages, np.zeros(7), args, "target_pregrasp")
    _assert_no_normal_motion(arm)
