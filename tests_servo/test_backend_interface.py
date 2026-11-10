"""Tests for control backend interface.

Verify that BackendCommand and ControlBackend ABC work as expected.
"""

import numpy as np
import numpy.testing as npt

from control_backends.base import BackendCommand, ControlBackend
from control_backends.force_ctrl.controller import NeroForceController
from control_backends.sdk_cpv_velocity_backend import SdkCpvVelocityBackend
from control_backends.sdk_force_ctrl_backend import (
    DEFAULT_KD,
    DEFAULT_KP,
    SdkForceCtrlBackend,
)


def test_backend_command_defaults():
    """BackendCommand defaults: qdot/torque_ff zero; kp/kd NaN (unset).

    kp/kd default to NaN so the backend can distinguish "unset" (-> default
    gains) from an explicit zero (-> zero-gravity / pure torque). This is the
    convention that lets _send_zero_gravity work.
    """
    cmd = BackendCommand(q_cmd=np.ones(7))
    npt.assert_array_equal(cmd.qdot_cmd, np.zeros(7))
    npt.assert_array_equal(cmd.torque_ff, np.zeros(7))
    assert np.all(np.isnan(cmd.kp))
    assert np.all(np.isnan(cmd.kd))
    assert cmd.mode == "position"


def test_backend_command_custom():
    """BackendCommand should accept custom values."""
    cmd = BackendCommand(
        q_cmd=np.zeros(7),
        qdot_cmd=np.ones(7) * 0.1,
        kp=np.ones(7) * 10.0,
        kd=np.ones(7) * 1.0,
        mode="mit",
    )
    npt.assert_array_almost_equal(cmd.qdot_cmd, np.ones(7) * 0.1)
    assert cmd.mode == "mit"


class FakeForceController:
    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.started = False
        self.stopped = False
        self.commands = []
        self.q = np.arange(7, dtype=np.float64) * 0.1
        self.robot = None
        FakeForceController.instances.append(self)

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True

    def command_joint_state(self, **kwargs):
        self.commands.append(kwargs)

    def get_joint_pos(self):
        return self.q.copy()

    def get_debug_state(self, include_trace=False):
        return {"running": True, "last_error": None}

    def emergency_stop(self):
        self.stopped = True


class FailingForceController(FakeForceController):
    def start(self):
        self.started = True
        raise RuntimeError("partial start failure")


class StopFailForceController(FakeForceController):
    def stop(self):
        self.stopped = True
        raise RuntimeError("stop not confirmed")


def test_sdk_force_backend_dry_run_does_not_create_controller():
    FakeForceController.instances.clear()
    backend = SdkForceCtrlBackend(dry_run=True, controller_cls=FakeForceController)
    backend.connect()
    backend.send(BackendCommand(q_cmd=np.zeros(7)))
    assert backend.connected
    assert FakeForceController.instances == []


def test_sdk_force_backend_calls_command_joint_state():
    FakeForceController.instances.clear()
    gravity_factors = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.5])
    backend = SdkForceCtrlBackend(
        dry_run=False,
        channel="can-test",
        control_hz=200,
        torque_limit=14.0,
        gravity_factors=gravity_factors,
        controller_cls=FakeForceController,
    )
    backend.connect()
    ctrl = FakeForceController.instances[-1]
    assert ctrl.started
    assert ctrl.kwargs["channel"] == "can-test"
    assert ctrl.kwargs["firmware"] == "v111"
    assert ctrl.kwargs["torque_limit"] == 14.0
    npt.assert_array_almost_equal(ctrl.kwargs["gravity_factors"], gravity_factors)

    cmd = BackendCommand(
        q_cmd=np.ones(7) * 0.2,
        qdot_cmd=np.ones(7) * 0.01,
        torque_ff=np.ones(7) * 0.1,
        mode="mit",
    )
    backend.send(cmd)
    sent = ctrl.commands[-1]
    npt.assert_array_almost_equal(sent["pos"], cmd.q_cmd)
    npt.assert_array_almost_equal(sent["vel"], cmd.qdot_cmd)
    npt.assert_array_almost_equal(sent["torques"], cmd.torque_ff)
    npt.assert_array_almost_equal(sent["kp"], DEFAULT_KP)
    npt.assert_array_almost_equal(sent["kd"], DEFAULT_KD)
    assert sent["gravity_enabled"] is True

    npt.assert_array_almost_equal(backend.get_joint_angles(), ctrl.q)
    assert backend.get_debug_state() == {"running": True, "last_error": None}
    backend.emergency_stop()
    assert ctrl.stopped
    backend.disconnect()
    assert ctrl.stopped


def test_sdk_force_backend_cleans_up_partial_controller_start():
    FakeForceController.instances.clear()
    backend = SdkForceCtrlBackend(
        dry_run=False,
        controller_cls=FailingForceController,
    )

    with np.testing.assert_raises_regex(RuntimeError, "partial start failure"):
        backend.connect()

    ctrl = FakeForceController.instances[-1]
    assert ctrl.started is True
    assert ctrl.stopped is True
    assert backend._force_ctrl is None
    assert backend.connected is False


def test_sdk_force_backend_retains_controller_when_disconnect_fails():
    FakeForceController.instances.clear()
    backend = SdkForceCtrlBackend(
        dry_run=False,
        controller_cls=StopFailForceController,
    )
    backend.connect()
    ctrl = FakeForceController.instances[-1]

    with np.testing.assert_raises_regex(RuntimeError, "stop not confirmed"):
        backend.disconnect()

    assert backend._force_ctrl is ctrl
    assert backend.connected is True


def test_force_controller_stop_cleans_up_partial_start_idempotently():
    class PartialLoop:
        running = False

        def __init__(self):
            self.stop_calls = 0

        def stop(self):
            self.stop_calls += 1

    class PartialRobot:
        def __init__(self):
            self.disable_calls = 0
            self.disconnect_calls = 0

        def disable(self):
            self.disable_calls += 1
            return True

        def disconnect(self):
            self.disconnect_calls += 1

    ctrl = NeroForceController.__new__(NeroForceController)
    ctrl._loop = PartialLoop()
    ctrl._robot = PartialRobot()
    ctrl._started = False
    ctrl._robot_connected = True
    ctrl._robot_enabled = True

    ctrl.stop()
    ctrl.stop()

    assert ctrl._robot.disable_calls == 1
    assert ctrl._robot.disconnect_calls == 1
    assert ctrl._robot_connected is False
    assert ctrl._robot_enabled is False


def test_force_controller_joint_pos_prefers_loop_snapshot_over_sdk_getter():
    snapshot = np.arange(7, dtype=np.float64) * 0.1

    class SnapshotLoop:
        def get_joint_state(self):
            return type("State", (), {"pos": snapshot})()

    class MutatingRobot:
        def get_joint_angles(self):
            raise AssertionError("SDK getter must not run beside control loop")

    ctrl = NeroForceController.__new__(NeroForceController)
    ctrl._loop = SnapshotLoop()
    ctrl._robot = MutatingRobot()

    result = ctrl.get_joint_pos()

    np.testing.assert_array_equal(result, snapshot)
    result[:] = 99.0
    assert not np.any(snapshot == 99.0)


def test_force_controller_stop_retains_driver_if_loop_does_not_stop():
    class StuckLoop:
        running = True

        def stop(self):
            raise RuntimeError("worker still alive")

    class TrackingRobot:
        def __init__(self):
            self.estop_calls = 0
            self.disable_calls = 0
            self.disconnect_calls = 0

        def electronic_emergency_stop(self):
            self.estop_calls += 1

        def disable(self):
            self.disable_calls += 1
            return True

        def disconnect(self):
            self.disconnect_calls += 1

    ctrl = NeroForceController.__new__(NeroForceController)
    ctrl._loop = StuckLoop()
    ctrl._robot = TrackingRobot()
    ctrl._started = True
    ctrl._robot_connected = True
    ctrl._robot_enabled = True

    with np.testing.assert_raises_regex(
        RuntimeError, "driver connection retained"
    ):
        ctrl.stop()

    assert ctrl._robot.estop_calls == 1
    assert ctrl._robot.disable_calls == 0
    assert ctrl._robot.disconnect_calls == 0
    assert ctrl._started is True
    assert ctrl._robot_connected is True
    assert ctrl._robot_enabled is True


def test_force_controller_stop_can_retry_after_loop_later_exits(monkeypatch):
    class EventuallyStoppedLoop:
        running = True

        def __init__(self):
            self.stop_calls = 0

        def stop(self):
            self.stop_calls += 1
            if self.stop_calls == 1:
                raise RuntimeError("worker still alive")
            self.running = False

    class TrackingRobot:
        def __init__(self):
            self.estop_calls = 0
            self.disable_calls = 0
            self.disconnect_calls = 0

        def electronic_emergency_stop(self):
            self.estop_calls += 1

        def disable(self):
            self.disable_calls += 1
            return True

        def disconnect(self):
            self.disconnect_calls += 1

    monkeypatch.setattr(
        "control_backends.force_ctrl.controller.time.sleep",
        lambda _: None,
    )
    ctrl = NeroForceController.__new__(NeroForceController)
    ctrl._loop = EventuallyStoppedLoop()
    ctrl._robot = TrackingRobot()
    ctrl._started = True
    ctrl._robot_connected = True
    ctrl._robot_enabled = True

    with np.testing.assert_raises_regex(
        RuntimeError, "driver connection retained"
    ):
        ctrl.stop()
    ctrl.stop()

    assert ctrl._loop.stop_calls == 2
    assert ctrl._robot.estop_calls == 1
    assert ctrl._robot.disable_calls == 1
    assert ctrl._robot.disconnect_calls == 1
    assert ctrl._started is False
    assert ctrl._robot_connected is False
    assert ctrl._robot_enabled is False


def test_force_controller_refuses_unsafe_margin_before_control_takeover(monkeypatch):
    class StartupLoop:
        running = False

        def __init__(self):
            self.started = False

        def start(self):
            self.started = True

        def stop(self):
            self.running = False

    class StartupRobot:
        def __init__(self):
            self.disabled = False
            self.disconnected = False

        def connect(self):
            return None

        def reset(self):
            raise AssertionError("Startup refusal must not reset the device")

        def set_normal_mode(self):
            raise AssertionError("Startup refusal must not change control mode")

        def enable(self):
            raise AssertionError("Startup refusal must not enable motors")

        def get_joint_angles(self):
            q = np.zeros(7)
            q[5] = -0.633  # only ~0.10 rad above joint6 lower limit
            return type("Angles", (), {"msg": q})()

        def disable(self):
            self.disabled = True
            return True

        def disconnect(self):
            self.disconnected = True

    class Monitor:
        def read(self, robot, *_):
            return {"q": robot.get_joint_angles().msg}

    monkeypatch.setattr("control_backends.force_ctrl.controller.FeedbackHealthMonitor", Monitor)
    ctrl = NeroForceController.__new__(NeroForceController)
    ctrl._robot = StartupRobot()
    ctrl._loop = StartupLoop()
    ctrl._channel = "fake"
    ctrl._zero_gravity_init = True
    ctrl._startup_joint_margin = 0.30
    ctrl._skip_reset = False
    ctrl._kp = DEFAULT_KP.copy()
    ctrl._kd = DEFAULT_KD.copy()
    ctrl._started = False
    ctrl._robot_connected = False
    ctrl._robot_enabled = False

    with np.testing.assert_raises_regex(RuntimeError, "Unsafe startup posture"):
        ctrl.start()

    assert ctrl._loop.started is False
    assert ctrl._robot.disabled is False
    assert ctrl._robot.disconnected is True


def test_force_controller_emergency_stop_hits_hardware_before_joining_loop():
    calls = []

    class StopRobot:
        def electronic_emergency_stop(self):
            calls.append("estop")

    class StopLoop:
        def stop(self):
            calls.append("loop-stop")

    ctrl = NeroForceController.__new__(NeroForceController)
    ctrl._robot = StopRobot()
    ctrl._loop = StopLoop()

    ctrl.emergency_stop()

    assert calls == ["estop", "loop-stop"]


def test_force_controller_emergency_stop_reports_hardware_failure():
    class StopRobot:
        def electronic_emergency_stop(self):
            raise RuntimeError("estop bus error")

    class StopLoop:
        def stop(self):
            return None

    ctrl = NeroForceController.__new__(NeroForceController)
    ctrl._robot = StopRobot()
    ctrl._loop = StopLoop()

    with np.testing.assert_raises_regex(RuntimeError, "estop bus error"):
        ctrl.emergency_stop()


class _Msg:
    def __init__(self, msg):
        self.msg = msg


class FakeCpvRobot:
    class OPTIONS:
        class MOTION_MODE:
            CPV = "CPV"

    def __init__(self):
        self.connected = False
        self.disconnected = False
        self.vel_commands = []
        self.calls = []
        self.q = np.arange(7, dtype=np.float64) * 0.1

    def connect(self, start_read_thread=True):
        self.connected = start_read_thread

    def disconnect(self):
        self.disconnected = True

    def move_cpv_vel(self, joint_index, vel):
        self.vel_commands.append((joint_index, vel))

    def get_joint_angles(self):
        return _Msg(self.q.copy())

    def set_follower_mode(self):
        self.calls.append(("set_follower_mode",))
        return True

    def enable(self, joint_id):
        self.calls.append(("enable", joint_id))
        return True

    def set_motion_mode(self, mode):
        self.calls.append(("set_motion_mode", mode))
        return True

    def set_cpv_acc(self, joint_index, value):
        self.calls.append(("set_cpv_acc", joint_index, value))
        return True

    def set_cpv_dcc(self, joint_index, value):
        self.calls.append(("set_cpv_dcc", joint_index, value))
        return True

    def set_cpv_cv(self, joint_index, value):
        self.calls.append(("set_cpv_cv", joint_index, value))
        return True


def test_cpv_velocity_backend_sends_clipped_joint_velocities():
    robot = FakeCpvRobot()
    backend = SdkCpvVelocityBackend(
        dry_run=False,
        firmware="v112",
        max_joint_vel=0.02,
        robot_factory=lambda: robot,
        stop_on_disconnect=False,
    )
    backend.connect()
    assert robot.connected is True

    cmd = BackendCommand(
        q_cmd=np.zeros(7),
        qdot_cmd=np.array([0.0, 0.01, 0.03, -0.04, 0.02, -0.02, 0.005]),
        mode="velocity",
    )
    backend.send(cmd)
    assert robot.vel_commands[-7:] == [
        (1, 0.0),
        (2, 0.01),
        (3, 0.02),
        (4, -0.02),
        (5, 0.02),
        (6, -0.02),
        (7, 0.005),
    ]
    npt.assert_array_almost_equal(backend.get_joint_angles(), robot.q)

    backend.disconnect()
    assert robot.disconnected is True


def test_cpv_velocity_backend_prepares_cpv_mode():
    robot = FakeCpvRobot()
    backend = SdkCpvVelocityBackend(
        dry_run=False,
        firmware="v112",
        max_joint_vel=0.02,
        cpv_acc=0.5,
        cpv_dcc=0.4,
        cpv_cv=0.3,
        prepare_delay_s=0.0,
        robot_factory=lambda: robot,
        stop_on_disconnect=False,
    )
    backend.connect()

    assert ("set_follower_mode",) not in robot.calls
    assert ("enable", 255) in robot.calls
    assert ("set_motion_mode", "CPV") in robot.calls
    for joint in range(1, 8):
        assert ("set_cpv_acc", joint, 0.5) in robot.calls
        assert ("set_cpv_dcc", joint, 0.4) in robot.calls
        assert ("set_cpv_cv", joint, 0.3) in robot.calls
    assert robot.vel_commands[:7] == [(joint, 0.0) for joint in range(1, 8)]


def test_cpv_velocity_backend_can_explicitly_set_follower_mode():
    robot = FakeCpvRobot()
    backend = SdkCpvVelocityBackend(
        dry_run=False,
        firmware="v112",
        max_joint_vel=0.02,
        set_follower_mode=True,
        prepare_delay_s=0.0,
        robot_factory=lambda: robot,
        stop_on_disconnect=False,
    )
    backend.connect()

    assert ("set_follower_mode",) in robot.calls


def test_cpv_velocity_backend_applies_configured_signs():
    robot = FakeCpvRobot()
    backend = SdkCpvVelocityBackend(
        dry_run=False,
        firmware="v112",
        max_joint_vel=0.02,
        velocity_signs=np.array([1, -1, 1, -1, 1, -1, 1], dtype=np.float64),
        robot_factory=lambda: robot,
        stop_on_disconnect=False,
    )
    backend.connect()

    cmd = BackendCommand(
        q_cmd=np.zeros(7),
        qdot_cmd=np.array([0.01, 0.01, -0.03, -0.01, 0.02, -0.02, 0.005]),
        mode="velocity",
    )
    backend.send(cmd)
    assert robot.vel_commands[-7:] == [
        (1, 0.01),
        (2, -0.01),
        (3, -0.02),
        (4, 0.01),
        (5, 0.02),
        (6, 0.02),
        (7, 0.005),
    ]


def test_cpv_velocity_backend_requires_v112():
    backend = SdkCpvVelocityBackend(
        dry_run=False,
        firmware="v111",
        robot_factory=lambda: FakeCpvRobot(),
    )
    try:
        backend.connect()
    except RuntimeError as exc:
        assert "firmware='v112'" in str(exc)
    else:
        raise AssertionError("Expected RuntimeError for non-v112 CPV backend")


if __name__ == "__main__":
    test_backend_command_defaults()
    test_backend_command_custom()
    print("All backend interface tests passed!")
