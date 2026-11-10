"""Proportional pinch control and an SDK link for an EMPTY gripper.

Ordinary use writes only gripper targets. Explicit prepare_can_hold additionally
hands the controller to CAN and establishes a hold at measured joint positions.
It never enables/disables motors or commands a new arm pose.
Losing the hand pauses this isolated test; the same hand resumes automatically.
"""
from __future__ import annotations

import copy
from collections import deque
from dataclasses import dataclass
import math
import time

import numpy as np
from targets.hand_selection import filter_hand_observations

MAX_WIDTH_M = 0.085
MAX_FORCE_N = 0.50
DEFAULT_MAX_LEAD_M = 0.005


@dataclass(frozen=True)
class Pinch:
    ratio: float
    side: str


def measure_pinch(observations, image_shape, side=None) -> Pinch:
    """One unambiguous RGB hand; pixel-distance ratio avoids aspect distortion."""
    observations, rejected = filter_hand_observations(observations, image_shape, fingertips=True)
    if len(observations) != 1:
        if not observations and len(rejected) == 1:
            raise ValueError(rejected[0]['reason'])
        raise ValueError("show exactly one hand")
    observation = observations[0]
    label = observation.handedness
    score = float(observation.handedness_score)
    if label not in ("Left", "Right") or not math.isfinite(score) or score < .7:
        raise ValueError("hand side uncertain")
    if side is not None and label != side:
        raise ValueError(f"waiting for the selected {side} hand")
    points = np.asarray(observation.landmarks_normalized, dtype=float)
    if points.shape != (21, 3) or not np.isfinite(points).all():
        raise ValueError("invalid hand landmarks")
    xy = points[[4, 8, 5, 17], :2]
    if np.any(xy < 0.) or np.any(xy > 1.):
        raise ValueError("fingers or palm outside image")
    h, w = image_shape[:2]
    xy = xy * [w - 1, h - 1]
    palm = float(np.linalg.norm(xy[2] - xy[3]))
    if palm < 15.:
        raise ValueError("palm too small or edge-on")
    return Pinch(float(np.linalg.norm(xy[0] - xy[1])) / palm, label)


@dataclass(frozen=True)
class PinchMapping:
    """Linear aperture between observed (or explicitly supplied) endpoints."""

    closed: float = 0.
    opened: float = 1.

    def __post_init__(self):
        if (not math.isfinite(self.closed) or not math.isfinite(self.opened)
                or not 0. <= self.closed < self.opened):
            raise ValueError("pinch ratios must be finite and 0 <= closed < open")

    def fraction(self, pinch):
        if not math.isfinite(pinch.ratio):
            raise ValueError("invalid pinch ratio")
        return float(np.clip((pinch.ratio - self.closed) / (self.opened - self.closed), 0., 1.))


class AutoPinchMapping:
    """Learn closed, then fully open holds without keys or actuator commands.

    The user must actually show both endpoints: a partial opening cannot reveal
    a person's maximum aperture. Once learned, the endpoints stay fixed across
    hand loss instead of turning each new observed maximum into 100%.
    """

    def __init__(self, mapping=None, hold_seconds=.8):
        if not math.isfinite(hold_seconds) or hold_seconds <= 0.:
            raise ValueError("invalid endpoint hold duration")
        self.mapping = mapping
        self.hold_seconds = hold_seconds
        self.phase = 'ready' if mapping is not None else 'close'
        self.closed = None
        self.samples = deque()

    def pause(self):
        self.samples.clear()

    @property
    def instruction(self):
        if self.phase == 'close':
            return 'Touch thumb + index tips; hold still for 1 second'
        if self.phase == 'open':
            return 'Spread thumb + index FULLY; hold still for 1 second'
        return 'Proportional follow | Q/Esc=quit'

    def observe(self, pinch, now):
        if not math.isfinite(pinch.ratio) or not math.isfinite(now) or pinch.ratio < 0.:
            raise ValueError('invalid calibration sample')
        if self.mapping is not None:
            return self.mapping.fraction(pinch)
        # A starting open hand must not be silently learned as "closed".
        candidate = (pinch.ratio <= .45 if self.phase == 'close'
                     else pinch.ratio >= self.closed + .6)
        if not candidate:
            self.pause()
            return None
        if self.samples and (now <= self.samples[-1][0] or now - self.samples[-1][0] > .25):
            self.pause()
        self.samples.append((now, pinch.ratio))
        while self.samples and now - self.samples[0][0] > self.hold_seconds + .15:
            self.samples.popleft()
        if len(self.samples) < 8 or now - self.samples[0][0] < self.hold_seconds:
            return None
        ratios = np.array([ratio for _, ratio in self.samples])
        middle = float(np.median(ratios))
        # Ignore isolated landmark outliers, but require a stable hold rather
        # than capturing an endpoint while the fingers are still opening.
        if np.ptp(np.percentile(ratios, [10, 90])) > max(.04, .08 * middle):
            return None
        self.pause()
        if self.phase == 'close':
            self.closed = middle
            self.phase = 'open'
            return None
        # Small endpoint dead zones absorb fingertip-center offset and jitter.
        padding = .03 * (middle - self.closed)
        self.mapping = PinchMapping(self.closed + padding, middle - padding)
        self.phase = 'ready'
        return self.mapping.fraction(pinch)


class GripperUnresponsive(RuntimeError):
    """A sent target did not produce measurable opening progress."""


class GripperResponseWatch:
    def __init__(self, timeout=1., tolerance=.001, progress=.0002):
        if not all(math.isfinite(v) and v > 0. for v in (timeout, tolerance, progress)):
            raise ValueError('invalid response watchdog limits')
        self.timeout, self.tolerance, self.progress = timeout, tolerance, progress
        self.pause()

    def pause(self):
        self.direction, self.since, self.anchor = 0, None, None

    def observe(self, sent_width, measured_width, now):
        if not all(math.isfinite(v) for v in (sent_width, measured_width, now)):
            raise ValueError('invalid response sample')
        error = sent_width - measured_width
        if abs(error) <= self.tolerance:
            self.pause()
            return
        direction = 1 if error > 0. else -1
        if (direction != self.direction or self.since is None
                or direction * (measured_width - self.anchor) >= self.progress):
            self.direction, self.since, self.anchor = direction, now, measured_width
        elif now - self.since >= self.timeout:
            raise GripperUnresponsive(
                f'NO RESPONSE: sent={sent_width*1000:.1f}mm actual={measured_width*1000:.1f}mm; '
                f'no {self.progress*1000:g}mm progress for {self.timeout:g}s; restart required')


@dataclass
class AutoPinchHand:
    """Lock the first consistent hand and reacquire it without changing roles."""

    side: str | None = None
    candidate: str | None = None
    good_frames: int = 0

    def lost(self):
        self.candidate = None
        self.good_frames = 0

    def observe(self, pinch):
        if self.side is not None and pinch.side != self.side:
            self.lost()
            return False
        if self.candidate != pinch.side:
            self.candidate, self.good_frames = pinch.side, 0
        self.good_frames += 1
        if self.good_frames < 3:
            return False
        if self.side is None:
            self.side = pinch.side
        return True


class WidthRamp:
    """Absolute proportional target with full-stroke rate and feedback lead caps."""

    def __init__(self, minimum=0., maximum=MAX_WIDTH_M, open_seconds=5., close_seconds=2.,
                 max_lead=DEFAULT_MAX_LEAD_M):
        values = (minimum, maximum, open_seconds, close_seconds, max_lead)
        if (not all(math.isfinite(v) for v in values)
                or not 0. <= minimum < maximum <= MAX_WIDTH_M
                or min(open_seconds, close_seconds, max_lead) <= 0.):
            raise ValueError("invalid width/rate/lead limits")
        self.minimum, self.maximum = minimum, maximum
        self.open_rate = (maximum - minimum) / open_seconds
        self.close_rate = (maximum - minimum) / close_seconds
        self.max_lead = max_lead
        self.width = None
        self.last_time = None

    def reset(self, width, now):
        if not math.isfinite(width) or not self.minimum <= width <= self.maximum:
            raise ValueError("current opening is outside the selected test range")
        if not math.isfinite(now):
            raise ValueError("invalid clock")
        self.width, self.last_time = float(width), float(now)

    def advance(self, fraction, now, measured_width=None):
        if self.width is None:
            raise ValueError("ramp is not initialized")
        if not math.isfinite(fraction) or not math.isfinite(now):
            raise ValueError("invalid target or clock")
        # Never catch up a camera/inference stall with one large target jump.
        dt = float(np.clip(now - self.last_time, 0., .1))
        lower = max(self.minimum, self.width - self.close_rate * dt)
        upper = min(self.maximum, self.width + self.open_rate * dt)
        if measured_width is not None:
            if not math.isfinite(measured_width):
                raise ValueError("invalid measured opening")
            lower = max(lower, measured_width - self.max_lead)
            upper = min(upper, measured_width + self.max_lead)
        if lower > upper + 1e-12:
            raise ValueError("opening feedback diverged from command; pause and re-arm")
        target = self.minimum + float(np.clip(fraction, 0., 1.)) * (self.maximum - self.minimum)
        self.width = float(np.clip(target, lower, upper))
        self.last_time = float(now)
        return self.width


@dataclass(frozen=True)
class GripperFeedback:
    width: float
    force: float
    joints: np.ndarray


def _fresh(wrapped, wall_time, timeout):
    stamp = float(getattr(wrapped, "timestamp", float("nan")))
    if not math.isfinite(stamp) or not 0. <= wall_time - stamp <= timeout:
        raise ValueError("device feedback missing or stale")
    return wrapped.msg


class GripperOnlyLink:
    """Bound gripper targets; arm control preparation is an explicit separate step."""

    def __init__(self, robot, gripper, force=.50, timeout=.5, max_lead=DEFAULT_MAX_LEAD_M):
        if (not all(math.isfinite(v) for v in (force, timeout, max_lead))
                or not 0. < force <= MAX_FORCE_N or timeout <= 0. or max_lead <= 0.):
            raise ValueError("invalid force/feedback limits")
        self.robot, self.gripper = robot, gripper
        self.force, self.timeout, self.max_lead = force, timeout, max_lead
        self.anchor = None
        self.device_status = {}
        self.observed_control = None

    @classmethod
    def connect(cls, channel="can0", firmware="v111", **kwargs):
        # Imported only in --execute. connect/init_effector do not enable the arm.
        from pyAgxArm import AgxArmFactory, create_agx_arm_config
        cfg = create_agx_arm_config(robot="nero", comm="can", channel=channel,
                                    interface="socketcan", firmeware_version=firmware)
        robot = AgxArmFactory.create_arm(cfg)
        try:
            robot.connect()
            gripper = robot.init_effector(robot.OPTIONS.EFFECTOR.AGX_GRIPPER)
            return cls(robot, gripper, **kwargs)
        except BaseException:
            robot.disconnect()
            raise

    def read(self):
        self.device_status = {}
        state = _fresh(copy.deepcopy(self.robot.get_arm_status()), time.time(), self.timeout)
        self.device_status.update(
            arm_status=int(state.arm_status), err_code=int(state.err_code),
            motion_status=int(state.motion_status), mode_feedback=int(state.mode_feedback),
        )
        if hasattr(state, 'ctrl_mode'):
            self.device_status['ctrl_mode'] = int(state.ctrl_mode)
        if int(state.arm_status) != 0 or int(state.err_code) != 0:
            raise ValueError(f"arm fault: arm_status={int(state.arm_status)} err_code={int(state.err_code)}")
        # motion_status is a target-arrival flag, not a measured velocity.
        # A real static position hold was observed with motion_status=1.
        # The caller checks a window of actual joint positions before/during
        # gripper control; never change arm mode or resend a target to clear it.
        if int(state.mode_feedback) not in (0, 1, 2, 3):
            raise ValueError(f"arm position mode required: mode_feedback={int(state.mode_feedback)}; stop servo/MIT first")
        enabled = list(self.robot.get_joints_enable_status_list())
        self.device_status['joint_enabled'] = enabled
        if len(enabled) != 7 or not all(enabled):
            raise ValueError(f"arm not already enabled: joint_enabled={enabled}; no automatic enable")
        # The aggregate SDK getter can hide missing joint packets with zeros.
        parser = self.robot._parser
        joints = []
        for name, indices in (("joint_12", (1, 2)), ("joint_34", (3, 4)),
                              ("joint_56", (5, 6)), ("joint_7", (7,))):
            msg = _fresh(copy.deepcopy(getattr(parser, name, None)), time.time(), self.timeout)
            joints.extend(float(getattr(msg, f"joint_{i}")) for i in indices)
        joints = np.asarray(joints)
        if not np.isfinite(joints).all():
            raise ValueError("invalid arm position feedback")
        if self.anchor is not None and np.max(np.abs(joints - self.anchor)) > np.deg2rad(1.):
            raise ValueError("arm moved over 1 degree; pause gripper test")
        wrapped = copy.deepcopy(self.gripper.get_gripper_status())
        wall_time = time.time()
        msg = _fresh(wrapped, wall_time, self.timeout)
        self.device_status.update(
            gripper_enabled=bool(msg.foc_status.driver_enable_status),
            gripper_feedback_age_ms=(wall_time - wrapped.timestamp) * 1000.,
        )
        width, force = float(msg.value), float(msg.force)
        if (msg.mode != "width" or not math.isfinite(width) or not 0. <= width <= MAX_WIDTH_M
                or not math.isfinite(force)):
            raise ValueError("invalid gripper width/force feedback")
        faults = ("voltage_too_low", "motor_overheating", "driver_overcurrent",
                  "driver_overheating", "sensor_status", "driver_error_status")
        if any(getattr(msg.foc_status, name) for name in faults):
            raise ValueError("gripper reports a driver fault")
        # CAN 0x159 is an observed control frame, NOT an actuator acknowledgement.
        # Some setups never echo it; its absence must not block control.
        self.observed_control = None
        getter = getattr(self.gripper, 'get_gripper_ctrl_states', None)
        control = None if getter is None else copy.deepcopy(getter())
        if control is not None:
            age = time.time() - control.timestamp
            if (0. <= age <= self.timeout and math.isfinite(control.msg.value)
                    and math.isfinite(control.msg.force)):
                self.observed_control = dict(width_mm=float(control.msg.value)*1000.,
                    force=float(control.msg.force), status_code=int(control.msg.status_code),
                    age_ms=age*1000.)
        return GripperFeedback(width, force, joints)

    def send(self, width):
        if not math.isfinite(width) or not 0. <= width <= MAX_WIDTH_M:
            raise ValueError("gripper target out of bounds")
        feedback = self.read()  # Recheck freshness/arm state at the write boundary.
        self.require_can_control()
        if self.anchor is None:
            raise ValueError("gripper test not armed")
        if abs(width - feedback.width) > self.max_lead + 1e-9:
            raise ValueError("target leads measured gripper opening too far")
        self.gripper.move_gripper_m(value=float(width), force=self.force)

    def require_can_control(self):
        mode = self.device_status.get('ctrl_mode')
        if mode != 1:
            raise ValueError(f'CAN control required: ctrl_mode={mode} (3=Ethernet); '
                             'restart with --prepare-can-hold to establish a current-position hold')

    def prepare_can_hold(self, record=None, timeout=2.):
        """Explicit startup takeover, based on the working grasp startup sequence.

        First CAN mode message has ZERO speed, so an old target cannot resume
        at normal speed. Clear the queue again, command measured joint positions,
        and require fresh target-arrival feedback before raising speed to 10%.
        Failures are terminal for the caller; never retry, reset or disable here.
        """
        if not math.isfinite(timeout) or timeout <= .5:
            raise ValueError('invalid CAN hold preparation timeout')
        record = record or (lambda *_args, **_kwargs: None)
        feedback = self.read()
        if self.device_status.get('ctrl_mode') == 1:
            return  # Already CAN-controlled; no reason to change a working hold.
        if self.device_status.get('ctrl_mode') != 3 or self.device_status['mode_feedback'] != 1:
            raise ValueError('CAN preparation requires an already enabled, stationary Ethernet J hold')
        hold = feedback.joints.copy()
        # Avoid SDK joint-limit clipping turning "hold here" into a different
        # target. Check the configured limits without mutating the driver.
        limits = self.robot._config.get('joint_limits', {})
        for index, value in enumerate(hold, 1):
            bounds = limits.get(f'joint{index}')
            if not -math.pi <= value <= math.pi or (bounds is not None
                    and not bounds[0] <= value <= bounds[1]):
                raise ValueError('current joints outside SDK limits; no CAN takeover')
        self.anchor = hold.copy() if self.anchor is None else self.anchor
        record('can_hold_prepare', stage='before', joints=hold.tolist(),
               device_status=dict(self.device_status))
        start = time.monotonic()
        while time.monotonic() - start < .25:
            time.sleep(.02)
            feedback = self.read()
            if np.max(np.abs(feedback.joints - hold)) > np.deg2rad(.15):
                raise RuntimeError('arm is moving; no CAN takeover')
            if (self.device_status.get('ctrl_mode') != 3
                    or self.device_status['mode_feedback'] != 1):
                raise RuntimeError('arm control source changed; no CAN takeover')
        from pyAgxArm.protocols.can_protocol.msgs.nero.default.transmit import ArmMsgMotionCtrl
        terminate = ArmMsgMotionCtrl(emergency_stop=0, track_ctrl=0x06, grag_teach_ctrl=0)

        def clear_queue():
            for _ in range(3):
                self.robot._send_msg(terminate)
                time.sleep(.02)

        takeover_started = False
        try:
            clear_queue()
            takeover_started = True
            self.robot.set_speed_percent(0)  # First CAN selection: mode preserved, speed ZERO.
            clear_queue()
            current = self.read()
            if np.max(np.abs(current.joints - hold)) > np.deg2rad(.15):
                raise RuntimeError('arm drifted during zero-speed takeover; speed remains zero')
            self.robot.set_motion_mode('j')
            self.robot.move_j(hold.tolist())
            sent_at = time.time()
            record('can_hold_prepare', stage='hold_sent_at_zero_speed', joints=hold.tolist())

            def verify_hold(duration):
                deadline = time.monotonic() + timeout
                steady_since = None
                while time.monotonic() < deadline:
                    time.sleep(.02)
                    actual = self.read()
                    drift = float(np.max(np.abs(actual.joints - hold)))
                    if drift > np.deg2rad(.5):
                        raise RuntimeError('arm drifted over 0.5 degree during CAN hold; stop test')
                    state = self.device_status
                    stamps = [self.robot.get_arm_status().timestamp] + [
                        getattr(self.robot._parser, name).timestamp
                        for name in ('joint_12', 'joint_34', 'joint_56', 'joint_7')]
                    # motion_status is used here for its actual meaning: confirming
                    # arrival at the NEWLY commanded current-position hold.
                    confirmed = (state.get('ctrl_mode') == 1 and state['mode_feedback'] == 1
                                 and state['motion_status'] == 0 and min(stamps) >= sent_at)
                    now = time.monotonic()
                    if not confirmed:
                        steady_since = None
                    elif steady_since is None:
                        steady_since = now
                    record('can_hold_sample', joints=actual.joints.tolist(),
                           drift_deg=math.degrees(drift), device_status=dict(state), confirmed=confirmed)
                    if steady_since is not None and now - steady_since >= duration:
                        return
                raise RuntimeError('CAN current-position hold not confirmed; no gripper commands sent')

            verify_hold(.2)  # Do not raise speed until the replacement target is acknowledged.
            self.robot.set_speed_percent(10)
            sent_at = time.time()
            verify_hold(.3)
            record('can_hold_prepare', stage='confirmed', joints=hold.tolist(), speed_percent=10)
        except BaseException as error:
            if takeover_started:
                # Never leave a failed takeover at restored trajectory speed.
                # This does not reset, disable or restore an old controller target.
                try:
                    self.robot.set_speed_percent(0)
                except Exception as stop_error:
                    message = f'{error}; zero-speed fallback also failed: {stop_error}'
                    if isinstance(error, KeyboardInterrupt):
                        print(f'[pinch][WARN] {message}', flush=True)
                    else:
                        raise RuntimeError(message) from error
            raise

    def close(self):
        # Disconnect does not open/close the gripper or disable the arm.
        self.robot.disconnect()
