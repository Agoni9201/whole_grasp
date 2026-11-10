"""Offline safety and ownership tests for the 200 Hz MIT impedance loop."""

import threading
from types import SimpleNamespace

import numpy as np
import pytest

from control_backends.force_ctrl.data_types import ImpedanceConfig, JointCommands
from control_backends.force_ctrl.mit_loop import (
    JOINT_LIMIT_BUFFER,
    NERO_JOINT_LIMITS,
    QDOT_TRIP_FRESH_FRAMES,
    NeroMITLoop,
)


class _FakePin:
    def fk_and_jacobian(self, q):
        return np.array([0.5, 0.0, 0.4]), np.eye(3), np.zeros((6, 7))


def _valid_cfg(**overrides):
    cfg = ImpedanceConfig(
        target_tcp=np.array([0.5, 0.0, 0.4]),
        R_target=np.eye(3),
        K=np.array([200.0, 200.0, 200.0, 5.0, 5.0, 5.0]),
        B=np.array([5.0, 5.0, 5.0, 0.2, 0.2, 0.2]),
        joint_damping=np.zeros(7),
        tau_limit_per_joint=np.full(7, 14.0),
        q_rest=np.zeros(7),
        use_dls_mapping=False,
    )
    for name, value in overrides.items():
        setattr(cfg, name, value)
    return cfg


def _settable_loop():
    loop = NeroMITLoop.__new__(NeroMITLoop)
    loop._pin = _FakePin()
    loop._impedance_lock = threading.Lock()
    loop._impedance_cfg = None
    return loop


def test_set_cartesian_impedance_owns_deep_copy():
    loop = _settable_loop()
    cfg = _valid_cfg()
    original_target = cfg.target_tcp.copy()
    original_k = cfg.K.copy()

    loop.set_cartesian_impedance(cfg)
    cfg.target_tcp[:] = 99.0
    cfg.K[:] = 1.0
    cfg.active = False

    assert np.array_equal(loop._impedance_cfg.target_tcp, original_target)
    assert np.array_equal(loop._impedance_cfg.K, original_k)
    assert loop._impedance_cfg.target_tcp is not cfg.target_tcp
    assert loop._impedance_cfg.active is True


def test_debug_state_reports_inner_impedance_path_and_torque():
    loop = _settable_loop()
    loop.set_cartesian_impedance(_valid_cfg())
    loop._impedance_safety_lock = threading.RLock()
    loop._state_lock = threading.Lock()
    loop._running = True
    loop._last_error = None
    loop._gravity_enabled = True
    loop._last_control_path = "impedance"
    loop._impedance_cycle_count = 17
    loop._last_impedance_torque = np.arange(7, dtype=np.float64)
    loop._last_gravity_torque = np.ones(7)
    loop._last_total_torque = np.arange(7, dtype=np.float64) + 1.0
    loop._last_feedback_torque = np.full(7, 0.5)
    loop._last_impedance_trip = None

    debug = loop.get_debug_state()

    assert debug["impedance_active"] is True
    assert debug["control_path"] == "impedance"
    assert debug["impedance_cycle_count"] == 17
    assert np.array_equal(debug["impedance_torque"], np.arange(7))


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("target_tcp", np.zeros(2), "target_tcp must have shape"),
        ("R_target", np.full((3, 3), np.nan), "R_target must contain only finite"),
        ("target_twist", np.zeros(5), "target_twist must have shape"),
        ("joint_damping", None, "joint_damping is required"),
        ("tau_limit_per_joint", np.ones(7) * -1.0, "must contain positive"),
        ("joint_weights", np.zeros(7), "joint_weights must contain positive"),
        ("task_torque_limit", 0.0, "must be finite and positive"),
        ("kp_joint", np.full(7, -1.0), "kp_joint must be within"),
        ("joint_hold_kp", np.full(7, 501.0), "joint_hold_kp must be within"),
        ("kd_joint", np.full(7, 5.1), "kd_joint must be within"),
        ("joint_hold_kd", np.full(7, -0.1), "joint_hold_kd must be within"),
        ("follow_max_error", -0.01, "follow_max_error must be finite and non-negative"),
        ("max_position_error", -0.01, "max_position_error must be finite and non-negative"),
        (
            "max_orientation_error_rad",
            np.nan,
            "max_orientation_error_rad must be finite and non-negative",
        ),
    ],
)
def test_set_cartesian_impedance_rejects_invalid_config(field, value, message):
    loop = _settable_loop()
    cfg = _valid_cfg(**{field: value})

    with pytest.raises(ValueError, match=message):
        loop.set_cartesian_impedance(cfg)

    assert loop._impedance_cfg is None


def test_joint_limit_buffer_shrinks_safe_range():
    loop = NeroMITLoop.__new__(NeroMITLoop)
    middle = np.mean(NERO_JOINT_LIMITS, axis=1)

    near_lower = middle.copy()
    near_lower[0] = NERO_JOINT_LIMITS[0, 0] + JOINT_LIMIT_BUFFER / 2.0
    assert loop._check_joint_limits(near_lower) is False

    near_upper = middle.copy()
    near_upper[0] = NERO_JOINT_LIMITS[0, 1] - JOINT_LIMIT_BUFFER / 2.0
    assert loop._check_joint_limits(near_upper) is False

    inside = middle.copy()
    inside[0] = NERO_JOINT_LIMITS[0, 0] + JOINT_LIMIT_BUFFER + 1e-3
    assert loop._check_joint_limits(inside) is True


@pytest.mark.parametrize(
    "values",
    [
        np.array([0.0, 0.0, 0.0, 0.0, np.nan, 0.0, 0.0]),
        np.zeros(6),
    ],
)
def test_joint_position_reader_rejects_nonfinite_or_wrong_shape(values):
    loop = NeroMITLoop.__new__(NeroMITLoop)
    loop._robot = SimpleNamespace(
        get_joint_angles=lambda: SimpleNamespace(msg=values, timestamp=1.0))

    with pytest.raises(ValueError, match="Invalid joint-position feedback"):
        loop._read_joint_pos()


def test_impedance_target_refresh_does_not_reset_safety_history():
    loop = _settable_loop()
    cfg = _valid_cfg()
    loop.set_cartesian_impedance(cfg)
    loop._qdot_over_count[4] = 3
    loop._impedance_trace.append({"sentinel": True})

    cfg.target_tcp = cfg.target_tcp + np.array([0.001, 0.0, 0.0])
    loop.set_cartesian_impedance(cfg)

    assert loop._qdot_over_count[4] == 3
    assert list(loop._impedance_trace) == [{"sentinel": True}]


def test_terminal_fault_rejects_impedance_reengagement():
    loop = _settable_loop()
    loop._terminal_fault = True

    with pytest.raises(RuntimeError, match="terminal MIT fault"):
        loop.set_cartesian_impedance(_valid_cfg())


def test_hard_stop_latches_terminal_fault_and_deactivates_old_target():
    loop = _settable_loop()
    loop._terminal_fault = False
    loop._running = True
    loop._robot = SimpleNamespace(
        electronic_emergency_stop=lambda: None)
    loop.set_cartesian_impedance(_valid_cfg())

    with loop._get_actuation_lock():
        loop._hard_stop_locked("test hard fault")

    assert loop._terminal_fault is True
    assert loop._running is False
    assert loop._impedance_cfg.active is False
    assert loop._last_error == "test hard fault"


def test_joint_hold_kd_applies_only_on_joint_hold_kp_mask():
    loop = NeroMITLoop.__new__(NeroMITLoop)
    loop._pin = _FakePin()
    q_rest = np.zeros(7)
    q_rest[-1] = 0.4
    cfg = _valid_cfg(
        q_rest=q_rest,
        kd_joint=np.full(7, 0.1),
        joint_hold_kp=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.0]),
        joint_hold_kd=np.array([9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 0.2]),
    )

    _, kp, kd, cmd_pos = loop._compute_impedance_torque(
        np.zeros(7), np.zeros(7), cfg)

    assert np.array_equal(kp[:6], np.zeros(6))
    assert kp[-1] == 5.0
    assert np.allclose(kd[:6], 0.1)
    assert kd[-1] == 0.2
    assert cmd_pos[-1] == q_rest[-1]


def test_hybrid_visual_follow_sets_low_kp_and_clips_command_error():
    loop = NeroMITLoop.__new__(NeroMITLoop)
    loop._pin = _FakePin()
    q = np.zeros(7)
    cfg = _valid_cfg(
        q_follow=np.full(7, 0.5),
        follow_kp=np.array([8.0, 10.0, 8.0, 12.0, 4.0, 2.0, 2.0]),
        follow_max_error=0.1,
        kd_joint=np.full(7, 0.2),
    )

    _, kp, kd, cmd_pos = loop._compute_impedance_torque(q, np.zeros(7), cfg)

    assert np.allclose(kp, cfg.follow_kp)
    assert np.allclose(kd, cfg.kd_joint)
    assert np.allclose(cmd_pos, 0.1)


def test_hybrid_follow_does_not_rewrite_reference_when_pushed():
    loop = NeroMITLoop.__new__(NeroMITLoop)
    loop._pin = _FakePin()
    q = np.full(7, 0.4)
    q_follow = np.zeros(7)
    cfg = _valid_cfg(
        q_follow=q_follow,
        follow_kp=np.ones(7),
        follow_max_error=0.1,
    )

    _, _, _, cmd_pos = loop._compute_impedance_torque(q, np.zeros(7), cfg)

    assert np.allclose(cfg.q_follow, q_follow)
    assert np.allclose(cmd_pos, 0.3)


def test_moving_impedance_follow_clip_preserves_cartesian_task_torque():
    """The follow spring is bounded independently of Cartesian torque."""

    class _TaskPin:
        def fk_and_jacobian(self, q):
            J = np.zeros((6, 7))
            J[0, :] = 1.0
            return np.array([0.5, 0.0, 0.4]), np.eye(3), J

    loop = NeroMITLoop.__new__(NeroMITLoop)
    loop._pin = _TaskPin()
    q_now = np.zeros(7)
    q_follow = np.full(7, 0.20)
    cfg = _valid_cfg(
        q_follow=q_follow,
        follow_kp=np.full(7, 2.0),
        follow_max_error=0.03,
        target_tcp=np.array([0.55, 0.0, 0.4]),
    )

    tau, kp, _, cmd_pos = loop._compute_impedance_torque(
        q_now, np.zeros(7), cfg)
    cfg_without_follow = _valid_cfg(target_tcp=np.array([0.55, 0.0, 0.4]))
    tau_without_follow, _, _, _ = loop._compute_impedance_torque(
        q_now, np.zeros(7), cfg_without_follow)

    assert np.allclose(tau, tau_without_follow)
    assert np.any(np.abs(tau) > 0.0)
    assert np.allclose(kp, 2.0)
    assert np.allclose(cmd_pos, 0.03)


def test_impedance_velocity_keeps_finite_difference_backup_hot():
    loop = NeroMITLoop.__new__(NeroMITLoop)
    calls = []
    finite_dq = np.ones(7)
    real_dq = np.full(7, 2.0)

    def read_finite(q, tau_override):
        calls.append(("finite", tau_override))
        return finite_dq

    def read_real(cfg):
        calls.append(("real", None))
        return real_dq

    loop._read_joint_vel = read_finite
    loop._read_real_joint_vel = read_real
    cfg = _valid_cfg(use_real_velocity=True, vel_filter_tau=0.012)

    selected = loop._read_impedance_joint_vel(np.zeros(7), cfg)

    assert calls == [("finite", 0.012), ("real", None)]
    assert selected is real_dq


class _TorqueRobot:
    def __init__(self, missing=()):
        self.get_calls = []
        states = {}
        for i in range(1, 8):
            states[f"motor_state_{i}"] = (
                None if i in missing
                else SimpleNamespace(msg=SimpleNamespace(torque=float(i)))
            )
        self._parser = SimpleNamespace(**states)

    def get_motor_states(self, joint):
        self.get_calls.append(joint)
        return SimpleNamespace(msg=SimpleNamespace(torque=70.0 + joint))


def test_joint_torque_prefers_parser_without_sdk_getter():
    robot = _TorqueRobot()
    loop = NeroMITLoop.__new__(NeroMITLoop)
    loop._robot = robot

    torque = loop._read_joint_torque()

    assert np.array_equal(torque, np.arange(1.0, 8.0))
    assert robot.get_calls == []


def test_joint_torque_falls_back_only_for_missing_parser_joint():
    robot = _TorqueRobot(missing={4})
    loop = NeroMITLoop.__new__(NeroMITLoop)
    loop._robot = robot

    torque = loop._read_joint_torque()

    assert torque[3] == 74.0
    assert robot.get_calls == [4]


class _ModeRobot:
    def __init__(self):
        self.auto_calls = []
        self.mode_calls = []
        self.move_calls = []

    def set_auto_set_motion_mode_enabled(self, enabled):
        self.auto_calls.append(bool(enabled))

    def set_motion_mode(self, mode):
        self.mode_calls.append(mode)

    def move_mit(self, **kwargs):
        self.move_calls.append(kwargs)


def test_mit_mode_auto_is_disabled_and_mode_is_sent_once():
    robot = _ModeRobot()
    loop = NeroMITLoop.__new__(NeroMITLoop)
    loop._robot = robot
    loop._running = False
    loop._last_error = None
    loop._mit_mode_set = False
    loop._thread = None
    loop._mit_joint_count = 7
    loop._freq = 200
    loop._period = 1.0 / 200.0
    loop._state_lock = threading.Lock()
    loop._impedance_lock = threading.Lock()
    loop._impedance_cfg = None
    loop._impedance_generation = 0
    loop._actuation_lock = threading.Lock()
    loop._impedance_trace = []
    loop._last_impedance_trip = None
    loop._control_loop = lambda: None

    loop.start()
    loop.stop()
    assert robot.auto_calls == [False]

    zeros = np.zeros(7)
    loop._send_mit_commands(zeros, zeros, zeros, zeros, zeros)
    loop._send_mit_commands(zeros, zeros, zeros, zeros, zeros)
    assert robot.mode_calls == ["mit"]
    assert len(robot.move_calls) == 14


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("pos", np.full(7, np.nan), "invalid MIT pos"),
        ("kp", np.full(7, -1.0), "kp outside"),
        ("kd", np.full(7, 5.1), "kd outside"),
        ("torques", np.zeros(6), "invalid MIT torques"),
    ],
)
def test_mit_send_validates_entire_batch_before_first_can_frame(
    field, value, message
):
    robot = _ModeRobot()
    loop = NeroMITLoop.__new__(NeroMITLoop)
    loop._robot = robot
    loop._mit_mode_set = False
    loop._mit_joint_count = 7
    args = {
        "pos": np.zeros(7),
        "vel": np.zeros(7),
        "kp": np.zeros(7),
        "kd": np.zeros(7),
        "torques": np.zeros(7),
    }
    args[field] = value

    with pytest.raises(ValueError, match=message):
        loop._send_mit_commands(**args)

    assert robot.mode_calls == []
    assert robot.move_calls == []


def _trip_loop():
    loop = NeroMITLoop.__new__(NeroMITLoop)
    loop._reset_impedance_safety_state(clear_last_trip=False)
    return loop


def test_qdot_trip_does_not_recount_cached_timestamp():
    loop = _trip_loop()
    dq = np.zeros(7)
    dq[4] = 3.0
    stamps = np.arange(7, dtype=np.float64)

    speed_joint = None
    for step in range(10):
        speed_joint, stale_joint, _ = loop._update_qdot_trip(
            dq, 2.0, stamps, now=step * 0.005)
        assert stale_joint is None

    assert speed_joint is None
    assert loop._qdot_over_count[4] == 1


def test_qdot_trip_requires_five_fresh_frames_on_same_joint():
    loop = _trip_loop()
    dq = np.zeros(7)
    dq[4] = 3.0

    for step in range(QDOT_TRIP_FRESH_FRAMES):
        stamps = np.full(7, float(step))
        speed_joint, stale_joint, _ = loop._update_qdot_trip(
            dq, 2.0, stamps, now=step * 0.005)
        assert stale_joint is None
        if step < QDOT_TRIP_FRESH_FRAMES - 1:
            assert speed_joint is None

    assert speed_joint == 4
    assert loop._qdot_over_count[4] == QDOT_TRIP_FRESH_FRAMES


def test_qdot_trip_counts_are_per_joint_and_low_frame_resets():
    loop = _trip_loop()
    stamps = np.zeros(7)
    for step in range(4):
        joint = step % 2
        stamps[joint] += 1.0
        dq = np.zeros(7)
        dq[joint] = 3.0
        speed_joint, _, _ = loop._update_qdot_trip(
            dq, 2.0, stamps.copy(), now=step * 0.005)
        assert speed_joint is None
    assert np.array_equal(loop._qdot_over_count[:2], [2, 2])

    stamps[0] += 1.0
    loop._update_qdot_trip(
        np.zeros(7), 2.0, stamps.copy(), now=0.025)
    assert loop._qdot_over_count[0] == 0
    assert loop._qdot_over_count[1] == 2


def test_qdot_trip_falls_back_to_control_cycles_without_timestamps():
    loop = _trip_loop()
    dq = np.zeros(7)
    dq[2] = 3.0
    speed_joint = None
    for step in range(QDOT_TRIP_FRESH_FRAMES):
        speed_joint, stale_joint, fresh = loop._update_qdot_trip(
            dq, 2.0, timestamps=None, now=step * 0.005)
        assert stale_joint is None
        assert np.all(fresh)
    assert speed_joint == 2


def test_frozen_real_velocity_timestamp_has_watchdog():
    loop = _trip_loop()
    stamps = np.arange(7, dtype=np.float64)
    loop._update_qdot_trip(
        np.zeros(7), 2.0, stamps, now=0.0)
    speed_joint, stale_joint, fresh = loop._update_qdot_trip(
        np.zeros(7), 2.0, stamps, now=0.11)

    assert speed_joint is None
    assert stale_joint == 0
    assert not np.any(fresh)


class _VelocityRobot:
    def __init__(self, timestamps=True):
        states = {}
        for joint in range(1, 8):
            state = SimpleNamespace(
                msg=SimpleNamespace(velocity=float(joint)))
            if timestamps:
                state.timestamp = float(joint)
            states[f"motor_state_{joint}"] = state
        self._parser = SimpleNamespace(**states)


def _velocity_loop(robot):
    loop = NeroMITLoop.__new__(NeroMITLoop)
    loop._robot = robot
    loop._real_vel_filtered = np.zeros(7)
    loop._real_vel_last_time = None
    loop._real_vel_raw = np.zeros(7)
    loop._real_vel_timestamps = None
    loop._real_vel_filter_timestamps = None
    loop._real_vel_filter_update_time = np.zeros(7)
    loop._last_real_velocity_sample_time = None
    loop._real_velocity_unavailable_since = None
    return loop


def test_real_velocity_filter_only_updates_fresh_joint_frames():
    robot = _VelocityRobot(timestamps=True)
    loop = _velocity_loop(robot)
    cfg = _valid_cfg(
        velocity_signs=np.ones(7),
        real_vel_filter_tau=0.0,
    )

    first = loop._read_real_joint_vel(cfg)
    for joint in range(1, 8):
        state = getattr(robot._parser, f"motor_state_{joint}")
        state.msg.velocity = 100.0 + joint
    # Only joint5 received a new CAN frame.
    robot._parser.motor_state_5.timestamp += 1.0
    second = loop._read_real_joint_vel(cfg)

    assert np.array_equal(first, np.arange(1.0, 8.0))
    assert second[4] == 105.0
    assert np.array_equal(second[:4], first[:4])
    assert np.array_equal(second[5:], first[5:])


def test_real_velocity_without_frame_timestamps_is_unusable_for_impedance():
    loop = _velocity_loop(_VelocityRobot(timestamps=False))
    cfg = _valid_cfg(
        velocity_signs=np.ones(7),
        real_vel_filter_tau=0.0,
    )

    assert loop._read_real_joint_vel(cfg) is None
    assert loop._real_velocity_unavailable_since is not None


def test_real_velocity_loss_zeros_cartesian_damping_input():
    loop = NeroMITLoop.__new__(NeroMITLoop)
    loop._read_joint_vel = lambda q, tau_override: np.full(7, 3.0)
    loop._read_real_joint_vel = lambda cfg: None
    loop._last_real_velocity_sample_time = 1.0
    loop._real_vel_filtered = np.full(7, -4.0)
    cfg = _valid_cfg(use_real_velocity=True)

    selected = loop._read_impedance_joint_vel(np.zeros(7), cfg)

    assert np.array_equal(selected, np.zeros(7))
    assert loop._last_velocity_source == "real-unavailable"
    assert np.array_equal(loop._last_fdiff_velocity, np.full(7, 3.0))


def test_finite_difference_ignores_repeated_position_timestamp():
    loop = NeroMITLoop.__new__(NeroMITLoop)
    loop._vel_filter_tau = 0.0
    loop._vel_last_pos = None
    loop._vel_last_time = None
    loop._vel_last_pos_timestamp = None
    loop._vel_filtered = np.zeros(7)
    loop._last_fdiff_raw = np.zeros(7)
    loop._last_fdiff_velocity = np.zeros(7)
    loop._joint_pos_timestamp = 10.0

    loop._read_joint_vel(np.zeros(7))
    loop._vel_last_time -= 0.01
    repeated = loop._read_joint_vel(np.full(7, 0.01))
    assert np.array_equal(repeated, np.zeros(7))
    assert np.array_equal(loop._vel_last_pos, np.zeros(7))

    loop._joint_pos_timestamp = 11.0
    loop._vel_last_time -= 0.01
    fresh = loop._read_joint_vel(np.full(7, 0.01))
    assert np.all(fresh > 0.0)
    assert np.array_equal(loop._vel_last_pos, np.full(7, 0.01))


def test_disable_waits_for_actuation_boundary_and_always_installs_safe_buffer():
    loop = NeroMITLoop.__new__(NeroMITLoop)
    loop._impedance_lock = threading.Lock()
    loop._actuation_lock = threading.Lock()
    loop._impedance_cfg = _valid_cfg(active=True)
    loop._impedance_generation = 1
    loop._command_lock = threading.Lock()
    loop._commands = JointCommands(
        pos=np.ones(7),
        vel=np.ones(7),
        kp=np.full(7, 20.0),
        kd=np.zeros(7),
        torques=np.ones(7),
    )
    loop._state_lock = threading.Lock()
    loop._joint_state = None
    loop._robot = SimpleNamespace(get_joint_angles=lambda: None)
    loop._freq = 200.0
    loop._last_impedance_trip = None

    entered = threading.Event()
    finished = threading.Event()

    def disable():
        entered.set()
        loop.disable_cartesian_impedance()
        finished.set()

    loop._actuation_lock.acquire()
    worker = threading.Thread(target=disable)
    worker.start()
    assert entered.wait(timeout=1.0)
    assert not finished.wait(timeout=0.02)
    assert loop._impedance_cfg.active is True

    loop._actuation_lock.release()
    assert finished.wait(timeout=1.0)
    worker.join(timeout=1.0)

    assert loop._impedance_cfg.active is False
    assert np.array_equal(loop._commands.kp, np.zeros(7))
    assert np.array_equal(
        loop._commands.kd,
        np.array([1.0, 1.0, 1.0, 0.5, 0.3, 0.2, 0.2]),
    )


def test_disable_impedance_survives_invalid_live_and_cached_feedback():
    loop = NeroMITLoop.__new__(NeroMITLoop)
    loop._impedance_lock = threading.Lock()
    loop._actuation_lock = threading.Lock()
    loop._impedance_cfg = _valid_cfg(active=True)
    loop._impedance_generation = 1
    loop._command_lock = threading.Lock()
    loop._commands = JointCommands.zeros(7)
    loop._state_lock = threading.Lock()
    loop._joint_state = SimpleNamespace(pos=np.full(7, np.nan))
    loop._robot = SimpleNamespace(
        get_joint_angles=lambda: SimpleNamespace(msg=np.full(7, np.nan))
    )
    loop._freq = 200.0
    loop._last_impedance_trip = None

    loop.disable_cartesian_impedance()

    assert loop._impedance_cfg.active is False
    assert np.array_equal(loop._commands.pos, np.zeros(7))
    assert np.array_equal(loop._commands.kp, np.zeros(7))
    assert np.array_equal(
        loop._commands.kd,
        np.array([1.0, 1.0, 1.0, 0.5, 0.3, 0.2, 0.2]),
    )


def test_buffered_command_update_is_serialized_with_disable():
    loop = NeroMITLoop.__new__(NeroMITLoop)
    loop._actuation_lock = threading.Lock()
    loop._command_lock = threading.Lock()
    loop._commands = JointCommands.zeros(7)
    update = JointCommands(
        pos=np.ones(7),
        vel=np.zeros(7),
        kp=np.ones(7),
        kd=np.ones(7),
        torques=np.zeros(7),
    )
    entered = threading.Event()
    finished = threading.Event()

    def writer():
        entered.set()
        loop.update_commands(update)
        finished.set()

    loop._actuation_lock.acquire()
    worker = threading.Thread(target=writer)
    worker.start()
    assert entered.wait(timeout=1.0)
    assert not finished.wait(timeout=0.02)
    loop._actuation_lock.release()
    assert finished.wait(timeout=1.0)
    worker.join(timeout=1.0)
    assert np.array_equal(loop._commands.pos, np.ones(7))


def test_stop_retains_thread_reference_when_join_times_out():
    class StuckThread:
        def join(self, timeout):
            self.timeout = timeout

        def is_alive(self):
            return True

    robot = SimpleNamespace(estop_calls=0)

    def estop():
        robot.estop_calls += 1

    robot.electronic_emergency_stop = estop
    loop = NeroMITLoop.__new__(NeroMITLoop)
    loop._running = True
    loop._thread = StuckThread()
    loop._robot = robot

    with pytest.raises(RuntimeError, match="did not stop"):
        loop.stop()

    assert loop._thread is not None
    assert robot.estop_calls == 1
