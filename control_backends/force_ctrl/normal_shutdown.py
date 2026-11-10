"""Shared normal exit: confirmed MIT stop -> enabled device position hold.

No new driver/connection is created. Fault recovery and motor enable are
deliberately outside this module; a failed handoff must not masquerade as success.
"""
from contextlib import contextmanager
import logging
import signal
import threading
import time

import numpy as np

from .feedback_health import FeedbackHealthMonitor
from .device_feedback import describe_device_feedback

logger = logging.getLogger(__name__)


def report_shutdown(message):
    """A closed terminal must not interrupt physical stop/hold confirmation."""
    try:
        print(message)
    except Exception:
        pass


@contextmanager
def defer_sigint():
    """Let one bounded cleanup finish instead of interrupting it with Ctrl+C."""
    if threading.current_thread() is not threading.main_thread():
        yield
        return
    previous = signal.getsignal(signal.SIGINT)
    if getattr(previous, "_normal_shutdown_handler", False):
        yield
        return
    repeated = [False]
    def waiting(_signum, _frame):
        # No logging/locks/I/O from a signal handler inside controller cleanup.
        repeated[0] = True
    waiting._normal_shutdown_handler = True
    signal.signal(signal.SIGINT, waiting)
    try:
        yield
    finally:
        try:
            if repeated[0]:
                try:
                    logger.warning("[shutdown] repeated Ctrl+C deferred until controlled cleanup finished")
                except Exception:
                    pass
        finally:
            signal.signal(signal.SIGINT, previous)


@defer_sigint()
def close_backend_normally(backend):
    """Never disable impedance into a zero-Kp buffer before establishing hold."""
    try:
        prepare = getattr(backend, "prepare_shutdown", None)
        if callable(prepare):
            prepare()
        else:
            terminate = getattr(backend, "terminate_motion", None)
            if callable(terminate):
                terminate()
    finally:
        backend.disconnect()


class MITSettlingCheck:
    """One settling rule for fixed MIT holds and device-hold acknowledgement.

    Callers validate feedback freshness before update(). Cached frames never
    advance the counter, and each joint uses its own position-frame interval.
    """

    def __init__(self, target_q, *, max_drift_deg=1.0, delta_deg=0.15,
                 speed_deg_s=1.0, required_samples=3):
        target = np.asarray(target_q, dtype=float)
        if target.shape != (7,) or not np.isfinite(target).all():
            raise ValueError("MIT settling target must be a finite 7-vector")
        self.target_q = target.copy()
        self.max_drift = np.deg2rad(float(max_drift_deg))
        self.max_delta = np.deg2rad(float(delta_deg))
        self.max_speed = np.deg2rad(float(speed_deg_s))
        self.required_samples = int(required_samples)
        if (not np.isfinite([self.max_drift, self.max_delta, self.max_speed]).all()
                or min(self.max_drift, self.max_delta, self.max_speed) <= 0
                or self.required_samples < 1):
            raise ValueError("MIT settling limits must be finite and positive")
        self.previous = None
        self.stable_samples = 0
        self.last_delta_deg = float("inf")
        self.last_speed_deg_s = float("inf")

    def update(self, sample):
        values = {key: np.asarray(sample[key], dtype=float).copy()
                  for key in ("q", "position_timestamps", "motor_timestamps")}
        if any(value.shape != (7,) or not np.isfinite(value).all()
               for value in values.values()):
            raise RuntimeError("Invalid per-axis feedback during MIT settling")
        if any(np.any(values[key] <= 0.0)
               for key in ("position_timestamps", "motor_timestamps")):
            raise RuntimeError("Invalid per-axis timestamps during MIT settling")
        if np.max(np.abs(values["q"] - self.target_q)) > self.max_drift:
            raise RuntimeError("MIT hold drift exceeded 1 degree during settling")
        previous = self.previous
        if previous is None:
            self.previous = values
            return False
        if any(np.any(values[key] < previous[key])
               for key in ("position_timestamps", "motor_timestamps")):
            raise RuntimeError("MIT per-axis feedback timestamp went backwards during stop")
        dt = values["position_timestamps"] - previous["position_timestamps"]
        if np.all(dt > 0.0) and np.all(
                values["motor_timestamps"] > previous["motor_timestamps"]):
            delta = np.abs(values["q"] - previous["q"])
            speed = delta / dt
            self.last_delta_deg = float(np.rad2deg(np.max(delta)))
            self.last_speed_deg_s = float(np.rad2deg(np.max(speed)))
            slow = np.all(delta <= self.max_delta) and np.all(speed <= self.max_speed)
            self.stable_samples = self.stable_samples + 1 if slow else 0
            self.previous = values
        return self.stable_samples >= self.required_samples


class MITControlledStop:
    """Freeze once, await the inner-loop transition, then confirm fixed hold.

    The coordinator sends no CAN commands itself. The existing MIT loop owns
    both the transition and its successful final-batch acknowledgement.
    """

    def __init__(self, robot, loop, kp, kd, *, check_fault, timeout_s=2.0):
        self.robot, self.loop, self.check_fault = robot, loop, check_fault
        self.kp, self.kd = np.asarray(kp, dtype=float).copy(), np.asarray(kd, dtype=float).copy()
        if any(value.shape != (7,) or not np.isfinite(value).all()
               for value in (self.kp, self.kd)):
            raise ValueError("MIT stop gains must be finite 7-vectors")
        if np.any(self.kp <= 0) or np.any(self.kp > 500) or np.any(self.kd < 0) or np.any(self.kd > 5):
            raise ValueError("MIT stop gains require positive kp and valid kd")
        self.timeout_s = self._timeout(timeout_s)
        health = getattr(loop, "_feedback_health", None)
        self.monitor = FeedbackHealthMonitor(
            max_age_s=float(getattr(health, "max_age_s", 0.1)),
            max_skew_s=float(getattr(health, "max_skew_s", 0.05)),
            max_control_gap_s=float(getattr(health, "max_control_gap_s", 0.1)))
        self.target_q = None
        self.requested_monotonic = None
        self.phase = "idle"
        self.settling = None
        self._settle_started = None

    @staticmethod
    def _timeout(value):
        result = float(value)
        if not np.isfinite(result) or result <= 0.0:
            raise ValueError("MIT stop timeout must be finite and positive")
        return result

    def _read(self):
        self.check_fault()
        if not self.loop.running:
            raise RuntimeError("MIT worker is not running; cannot confirm a controlled stop")
        sample = self.monitor.read(self.robot, time.time(), time.monotonic())
        from .mit_loop import NERO_JOINT_LIMITS
        q = sample["q"]
        if np.any(q < NERO_JOINT_LIMITS[:, 0]) or np.any(q > NERO_JOINT_LIMITS[:, 1]):
            raise RuntimeError("MIT stop joint feedback is outside physical limits")
        self.check_fault()
        return sample

    def _status(self):
        status = self.loop.get_controlled_stop_status()
        if not isinstance(status, dict):
            raise RuntimeError("MIT controlled-stop acknowledgement is unavailable")
        duration = float(status.get("transition_duration_s", 0.0))
        if not np.isfinite(duration) or duration < 0.0:
            raise RuntimeError("MIT controlled-stop transition duration is invalid")
        return status, duration

    def request(self):
        sample = self._read()
        if self.target_q is not None:
            return self.target_q.copy()
        status, _ = self._status()
        if status.get("target_q") is None:
            from .mit_loop import JointCommands
            self.loop.hold_for_controlled_stop(JointCommands(
                pos=sample["q"].copy(), vel=np.zeros(7),
                kp=self.kp.copy(), kd=self.kd.copy(), torques=np.zeros(7)))
            status, _ = self._status()
        target = np.asarray(status.get("target_q"), dtype=float)
        if target.shape != (7,) or not np.isfinite(target).all():
            raise RuntimeError("MIT controlled-stop target was not latched")
        self.target_q = target.copy()
        self.requested_monotonic = time.monotonic()
        self.phase = "transition"
        record = getattr(self.loop, "record_control_event", None)
        if callable(record):
            record("mit_hold_requested", target_q=self.target_q.copy())
        return self.target_q.copy()

    def wait(self, timeout_s=None):
        timeout = self.timeout_s if timeout_s is None else self._timeout(timeout_s)
        if self.target_q is None:
            self.request()
        while True:
            sample = self._read()
            status, duration = self._status()
            target = np.asarray(status.get("target_q"), dtype=float)
            if target.shape != (7,) or not np.array_equal(target, self.target_q):
                raise RuntimeError("MIT controlled-stop target changed after latching")
            if np.max(np.abs(sample["q"] - self.target_q)) > np.deg2rad(1.0):
                raise RuntimeError("MIT hold drift exceeded 1 degree during stop transition")
            now = time.monotonic()
            if status.get("transition_complete") is not True:
                if self._settle_started is not None:
                    raise RuntimeError("MIT completed stop transition lost its acknowledgement")
                if now - self.requested_monotonic > duration + 0.5:
                    raise RuntimeError("MIT stop transition final command was not acknowledged")
            else:
                if self._settle_started is None:
                    self._settle_started = now
                    self.settling = MITSettlingCheck(self.target_q)
                    self.phase = "settling"
                if self.settling.update(sample):
                    self.phase = "confirmed"
                    result = {"confirmed": True,
                              "method": "mit_fixed_hold_fresh_feedback",
                              "last_joint_delta_deg": self.settling.last_delta_deg,
                              "last_joint_speed_deg_s": self.settling.last_speed_deg_s}
                    record = getattr(self.loop, "record_control_event", None)
                    if callable(record):
                        record("mit_hold_confirmed", result=result)
                    return result
                if now - self._settle_started >= timeout:
                    raise RuntimeError(
                        f"MIT hold settling was not confirmed within {timeout:.1f}s; "
                        f"last_delta={self.settling.last_delta_deg:.3f}deg")
            time.sleep(0.02)


def handoff_to_position_hold(robot, stop_sender, *, check_fault, timeout_s=2.0,
                             observation_s=0.5, record_event=None):
    """Running MIT callers must first confirm a fixed stationary hold.

    An interrupted startup with no running MIT sender can also use this to
    cancel any old device trajectory at zero speed and confirm current hold.

    stop_sender must join the sole MIT sender before returning. Every normal
    mode/target frame is sent only afterwards; a terminal fault forbids handoff.
    The caller retains ownership of the driver until this function succeeds.
    Keep the same connection for a bounded post-switch observation, including
    measured q/dq/torque in diagnostics rather than ending at the first mode ack.
    """
    timeout_s, observation_s = float(timeout_s), float(observation_s)
    if (not np.isfinite([timeout_s, observation_s]).all()
            or not 0.0 <= observation_s < timeout_s):
        raise ValueError("Device hold observation must be finite and shorter than its timeout")

    def record(event, **values):
        if callable(record_event):
            try:
                record_event(event, **values)
            except Exception:
                pass  # Optional diagnostics must not interrupt physical hold.

    monitor = FeedbackHealthMonitor()
    from .mit_loop import NERO_JOINT_LIMITS
    def read():
        sample = monitor.read(robot, time.time(), time.monotonic())
        if np.any(sample["q"] < NERO_JOINT_LIMITS[:, 0]) or np.any(sample["q"] > NERO_JOINT_LIMITS[:, 1]):
            raise RuntimeError("Joint feedback outside physical limits; no position-hold handoff")
        return sample
    check_fault()
    before = read()
    record("device_handoff_feedback", stage="before_sender_join", sample=before)
    stop_sender()
    check_fault()
    current = read()
    hold = current["q"].copy()
    if np.max(np.abs(hold - before["q"])) > np.deg2rad(1.0):
        raise RuntimeError("Enabled-exit position changed before device handoff")
    record("device_handoff_feedback", stage="before_mode_switch", sample=current)

    from pyAgxArm.protocols.can_protocol.msgs.nero.default.transmit import ArmMsgMotionCtrl
    robot.set_speed_percent(0)
    terminate = ArmMsgMotionCtrl(emergency_stop=0, track_ctrl=0x06, grag_teach_ctrl=0)
    for _ in range(3):
        robot._send_msg(terminate)
    robot.set_motion_mode("j")
    robot.move_j(hold.tolist())
    sent_at = time.time()
    started = time.monotonic()
    deadline = started + timeout_s
    record("device_hold_commanded", target_q=hold.copy(), sent_wall_time=sent_at)
    settling = MITSettlingCheck(hold)
    settling.update(current)
    peak_drift_deg = 0.0
    peak_speed_deg_s = 0.0
    last_device_feedback = None
    while time.monotonic() < deadline:
        time.sleep(0.02)
        check_fault()
        current = read()
        drift = float(np.rad2deg(np.max(np.abs(current["q"] - hold))))
        peak_drift_deg = max(peak_drift_deg, drift)
        record("device_hold_sample", sample=current, target_q=hold.copy(),
               elapsed_s=time.monotonic() - started, drift_deg=drift)
        if drift > 1.0:
            raise RuntimeError("Device position hold drifted before acknowledgement")
        stationary = settling.update(current)
        if np.isfinite(settling.last_speed_deg_s):
            peak_speed_deg_s = max(peak_speed_deg_s, settling.last_speed_deg_s)
        if current.get("dq") is not None:
            peak_speed_deg_s = max(peak_speed_deg_s, float(np.rad2deg(np.max(np.abs(current["dq"])))))
        wrapped = robot.get_arm_status()
        status = getattr(wrapped, "msg", None)
        stamp = float(getattr(wrapped, "timestamp", 0.0))
        drivers = [robot.get_driver_states(i) for i in range(1, 8)]
        wall = time.time()
        enabled = all(
            item is not None and sent_at <= float(item.timestamp) <= wall
            and wall - float(item.timestamp) <= 0.5
            and bool(item.msg.foc_status.driver_enable_status)
            for item in drivers
        )
        try:
            last_device_feedback = describe_device_feedback(wrapped, drivers, wall)
        except Exception as exc:
            last_device_feedback = {'read_error': f'{type(exc).__name__}: {exc}'}
        record('device_hold_ack_sample', device_feedback=last_device_feedback,
               stationary=bool(stationary), enabled=bool(enabled),
               elapsed_s=time.monotonic() - started)
        observed = time.monotonic() - started
        if (observed >= observation_s and stationary
                and status is not None and sent_at <= stamp <= wall
                and wall - stamp <= 0.5 and enabled and status.ctrl_mode == 1
                and status.arm_status == 0 and status.mode_feedback == 1
                and status.motion_status == 0):
            return {"confirmed": True, "method": "device_enabled_position_hold", "joints": hold.tolist(),
                    "actual_joints": current["q"].tolist(),
                    "position_timestamps": current["position_timestamps"].tolist(),
                    "motor_timestamps": current["motor_timestamps"].tolist(),
                    "observation_s": observed,
                    "peak_joint_drift_deg": peak_drift_deg,
                    "peak_joint_speed_deg_s": peak_speed_deg_s,
                    "confirmed_wall_time": wall}
    record('device_hold_unconfirmed', device_feedback=last_device_feedback)
    raise RuntimeError("Device enabled position hold was not acknowledged; no disable sent; "
                       f"last_device_feedback={last_device_feedback}")
