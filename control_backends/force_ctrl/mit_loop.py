import copy
import json
import time
import logging
import queue
import threading
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as Rot

from .data_types import JointCommands, JointState, ImpedanceConfig
from .gravity_comp import NeroGravityCompensator
from .real_velocity import read_real_qdot_sample
from .feedback_health import ControlDeadlineMissed, FeedbackHealthMonitor, FeedbackStale
from .normal_shutdown import MITSettlingCheck
from ..cartesian_impedance import (
    impedance_wrench,
    task_torque,
    dls_task_torque,
    joint_damping_torque,
    null_space_torque,
    coulomb_compensation,
)

logger = logging.getLogger(__name__)

# Nero joint limits (rad) from api/constants.py
NERO_JOINT_LIMITS = np.array([
    [-2.705261, 2.705261],   # joint1
    [-1.745330, 1.745330],   # joint2
    [-2.757621, 2.757621],   # joint3
    [-1.012291, 2.146755],   # joint4
    [-2.757621, 2.757621],   # joint5
    [-0.733039, 0.959932],   # joint6
    [-1.570797, 1.570797],   # joint7
])

# Safety buffer around joint limits (rad)
JOINT_LIMIT_BUFFER = 0.1

# Max position jump between consecutive frames (rad)
MAX_POS_JUMP = 0.5

# Report interval for loop statistics (seconds)
REPORT_INTERVAL = 30.0

# Number of fresh per-joint velocity feedback frames required for a runaway trip.
QDOT_TRIP_FRESH_FRAMES = 5

# Active impedance must not continue on a frozen parser velocity sample.
REAL_VELOCITY_STALE_TIMEOUT = 0.10

# Retain the last half-second for a compact post-trip diagnostic snapshot.
IMPEDANCE_TRACE_SECONDS = 0.50


def _clamp_vector_norm(vector, limit):
    value = np.asarray(vector, dtype=np.float64).ravel().copy()
    if limit is None:
        return value
    bound = float(limit)
    norm = float(np.linalg.norm(value))
    if norm > bound and norm > 1e-12:
        value *= bound / norm
    return value


def _diagnostic_json_value(value):
    if isinstance(value, np.ndarray):
        return _diagnostic_json_value(value.tolist())
    if isinstance(value, np.generic):
        return _diagnostic_json_value(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _diagnostic_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_diagnostic_json_value(item) for item in value]
    return value


class _MITDiagnosticWriter:
    """Bounded nonblocking producer; all JSON/file I/O belongs to the writer."""

    def __init__(self, path, capacity):
        self.path = str(Path(path))
        self.records = queue.Queue(maxsize=int(capacity))
        self.stop_requested = threading.Event()
        self.ready = threading.Event()
        self.accepting = True
        self.written = 0
        self.dropped = 0
        self.error = None
        self.closed = False
        self.thread = threading.Thread(
            target=self._run, daemon=True, name="nero_mit_log_writer")
        self.thread.start()
        if not self.ready.wait(timeout=5.0):
            self.accepting = False
            self.stop_requested.set()
            raise RuntimeError("MIT diagnostics writer did not start within 5.0s")
        if self.error is not None:
            raise RuntimeError(f"Cannot start MIT diagnostics: {self.error}")

    def submit(self, record):
        if not self.accepting:
            return
        try:
            self.records.put_nowait(record)
        except queue.Full:
            self.dropped += 1

    def status(self):
        return {
            "path": self.path,
            "accepting": self.accepting,
            "written_cycles": self.written,
            "dropped_cycles": self.dropped,
            "queued_cycles": self.records.qsize(),
            "error": self.error,
            "closed": self.closed,
        }

    def close(self, timeout):
        self.accepting = False
        self.stop_requested.set()
        self.thread.join(timeout=timeout)
        if self.thread.is_alive():
            self.error = self.error or "MIT diagnostics writer flush timed out"
            logger.error("%s: %s", self.path, self.error)
        if self.dropped:
            logger.warning("MIT diagnostics dropped %s cycles: %s", self.dropped, self.path)
        return self.status()

    def _run(self):
        try:
            with Path(self.path).open("x", encoding="utf-8") as stream:
                metadata = {
                    "type": "metadata", "schema_version": 1,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "units": {"q": "rad", "dq": "rad/s", "torque": "N.m", "time": "s"},
                    "torque_semantics": (
                        "tau_ff_sent is the clipped gravity plus requested feedforward sent to SDK; "
                        "it excludes firmware-internal kp/kd torque. Feedback torque is asynchronous feedback, "
                        "not a synchronized estimate of total commanded torque."),
                    "command_semantics": (
                        "Command fields are the completed host SDK batch, not device acknowledgement. "
                        "kp_12bit_estimate is a host estimate of V111 12-bit Kp encoding."),
                }
                stream.write(json.dumps(metadata, ensure_ascii=False) + "\n")
                stream.flush()
                self.ready.set()
                last_flush = time.monotonic()
                while not self.stop_requested.is_set() or not self.records.empty():
                    try:
                        record = self.records.get(timeout=0.05)
                    except queue.Empty:
                        continue
                    stream.write(json.dumps(
                        _diagnostic_json_value({
                            **record,
                            **({"timestamp": datetime.fromtimestamp(
                                record["wall_time"], timezone.utc).isoformat()}
                               if "wall_time" in record else {}),
                        }), ensure_ascii=False,
                        allow_nan=False, separators=(",", ":")) + "\n")
                    self.written += 1
                    if time.monotonic() - last_flush >= 1.0:
                        stream.flush()
                        last_flush = time.monotonic()
                stream.write(json.dumps({"type": "summary", **self.status(), "closed": True}) + "\n")
                stream.flush()
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            self.accepting = False
            logger.error("MIT diagnostics writer failed: %s: %s", self.path, self.error)
        finally:
            self.closed = True
            self.ready.set()


class NeroMITLoop:
    """200 Hz real-time MIT control loop for Nero 7-DOF arm.

    This loop runs in a background thread, continuously:
    1. Reads the latest joint angles from the pyAgxArm driver (parser cache)
    2. Computes gravity compensation torques via MuJoCo
    3. Combines user commands with gravity compensation
    4. Sends MIT control CAN frames for supported joints

    Parameters
    ----------
    robot : Driver
        The Nero pyAgxArm driver instance (already connected).
    gravity_comp : NeroGravityCompensator
        The gravity compensator instance.
    control_freq : int
        Control loop frequency in Hz. Default 200.
    gravity_factor : float
        Multiplier for gravity compensation torques. Default 1.0.
    torque_limit : float
        Maximum absolute torque per joint (N·m). Default 8.0.
    miss_limit : int
        Maximum consecutive feedback misses before emergency stop. Default 20.
    mit_joint_count : int
        Number of joints accepted by the installed SDK's move_mit().
    """

    N_JOINTS = 7

    def __init__(
        self,
        robot,
        gravity_comp: NeroGravityCompensator,
        control_freq: int = 200,
        gravity_factor: float = 1.0,
        gravity_factors=None,
        torque_limit: float = 8.0,
        miss_limit: int = 20,
        mit_joint_count: int = N_JOINTS,
        pin_model=None,
        joint_limit_buffer: float = JOINT_LIMIT_BUFFER,
        feedback_guard_enabled: bool = True,
        max_feedback_age_s: float = 0.1,
        max_feedback_skew_s: float = 0.05,
        max_control_gap_s: float = 0.1,
        hold_recovery_timeout_s: float = 0.0,
        controlled_stop_kp=None,
        controlled_stop_kd=None,
    ):
        self._robot = robot
        self._feedback_guard_enabled = bool(feedback_guard_enabled)
        self._feedback_health = FeedbackHealthMonitor(
            max_feedback_age_s, max_feedback_skew_s, max_control_gap_s)
        self._feedback_health_sample = None
        recovery_timeout = float(hold_recovery_timeout_s)
        if not np.isfinite(recovery_timeout) or not 0.0 <= recovery_timeout <= 1.0:
            raise ValueError("hold_recovery_timeout_s must be finite and within [0, 1]")
        self._hold_recovery_timeout_s = recovery_timeout
        self._hold_recovering = False
        self._last_healthy_hold_time = None
        self._grav = gravity_comp
        self._freq = control_freq
        self._period = 1.0 / control_freq
        self._gravity_factor = gravity_factor
        if gravity_factors is None:
            self._gravity_factors = np.ones(self.N_JOINTS, dtype=np.float64) * float(gravity_factor)
        else:
            self._gravity_factors = np.asarray(gravity_factors, dtype=np.float64).ravel() * float(gravity_factor)
            if len(self._gravity_factors) != self.N_JOINTS:
                raise ValueError(
                    f"gravity_factors must have shape ({self.N_JOINTS},), "
                    f"got {self._gravity_factors.shape}"
                )
        self._gravity_enabled = True
        self._torque_limit = torque_limit
        self._miss_limit = miss_limit
        self._mit_joint_count = mit_joint_count

        # Pinocchio model for in-loop Cartesian impedance (fk_and_jacobian). None
        # => impedance mode unavailable; the buffered command path (zero-gravity,
        # mit-position visual servo, etc.) is unaffected.
        self._pin = pin_model

        buffer = float(joint_limit_buffer)
        if not np.isfinite(buffer) or buffer < 0.0:
            raise ValueError("joint_limit_buffer must be finite and >= 0")
        self._joint_limit_buffer = buffer

        # Thread-safe command buffer
        self._command_lock = threading.Lock()
        self._commands = JointCommands.zeros(self.N_JOINTS)

        # Thread-safe in-loop Cartesian impedance config. None => disabled, the
        # loop runs the buffered command path. Set by the slow outer thread; read
        # every 200 Hz cycle under this lock.
        self._impedance_lock = threading.Lock()
        self._impedance_cfg: ImpedanceConfig = None
        self._impedance_generation = 0
        # Linearizes the final control decision/CAN send against engage and
        # disable.  Once disable_cartesian_impedance() returns, no frame from
        # the previous impedance generation can still be sent.
        self._actuation_lock = threading.Lock()

        # Thread-safe state buffer
        self._state_lock = threading.Lock()
        self._joint_state: JointState = None
        self._last_gravity_torque = np.zeros(self.N_JOINTS)
        self._last_impedance_torque = np.zeros(self.N_JOINTS)
        self._last_total_torque = np.zeros(self.N_JOINTS)
        self._last_feedback_torque = np.zeros(self.N_JOINTS)
        self._last_feedback_torque_timestamps = [None] * self.N_JOINTS
        self._last_feedback_torque_valid = np.zeros(self.N_JOINTS, dtype=bool)
        self._last_control_path = "buffered"
        self._impedance_cycle_count = 0
        self._diagnostics_writer = None
        self._diagnostics_phase = "initializing"
        self._cycle_stage = None
        self._cycle_stage_start = None
        self._cycle_stage_durations = {}
        self._cycle_loop_dt = None
        self._cycle_feedback_read_started = False
        self._cycle_sent_joints = 0

        self._running = False
        self._thread: threading.Thread = None
        # Per-joint sustained-exceedance state.  With parser velocity, only a
        # NEW CAN timestamp advances a counter, so one cached spike cannot be
        # counted five times by the faster 200 Hz control loop.
        self._qdot_over_count = np.zeros(self.N_JOINTS, dtype=np.int64)
        self._qdot_last_timestamp: np.ndarray = None
        self._qdot_timestamp_last_change = np.zeros(
            self.N_JOINTS, dtype=np.float64)
        self._last_error = None
        self._terminal_fault = False
        self._electronic_stop_sent = False
        self._command_stop_latched = False
        self._last_sent_snapshot = None
        self._controlled_stop_transition = None
        self._controlled_stop_complete = False
        self._controlled_stop_completed_monotonic = None
        self._controlled_stop_reason = None
        self._buffer_stop_started = None
        self._buffer_stop_previous = None
        self._buffer_stop_stable = 0
        self._buffer_stop_confirmed = False
        self._buffer_stop_settling = None
        self._buffer_stop_settle_started = None
        self._controlled_stop_kp = None
        self._controlled_stop_kd = None
        if controlled_stop_kp is not None or controlled_stop_kd is not None:
            kp = np.asarray(controlled_stop_kp, dtype=float)
            kd = np.asarray(controlled_stop_kd, dtype=float)
            if (kp.shape != (7,) or kd.shape != (7,)
                    or not np.isfinite(kp).all() or not np.isfinite(kd).all()
                    or np.any(kp <= 0) or np.any(kp > 500)
                    or np.any(kd < 0) or np.any(kd > 5)):
                raise ValueError("Controlled-stop gains require positive 7-axis kp and valid kd")
            self._controlled_stop_kp, self._controlled_stop_kd = kp.copy(), kd.copy()

        # Finite-difference velocity estimator state. The v111 driver does NOT
        # report joint velocity (get_motor_states().velocity is always 0), so we
        # synthesize it from consecutive get_joint_angles() reads with an
        # exponential low-pass. Used ONLY for published JointState.vel (external
        # readers: impedance damping, logging); the firmware computes its own
        # kd damping internally.
        # tau MUST stay small: a large low-pass lag turns the Cartesian damping
        # term (-Dx·J·qdot_m) into anti-damping at the ~5Hz wrist mode and the
        # impedance loop self-diverges. 8ms gives ~14deg phase at 5Hz (safe);
        # 30ms gave ~54deg and diverged.
        self._vel_filter_tau = 0.008
        self._vel_last_pos: np.ndarray = None
        self._vel_last_time: float = None
        self._joint_pos_timestamp: float = None
        self._vel_last_pos_timestamp: float = None
        self._vel_filtered: np.ndarray = np.zeros(self.N_JOINTS)
        self._last_fdiff_raw = np.zeros(self.N_JOINTS)
        self._last_fdiff_velocity = np.zeros(self.N_JOINTS)

        # REAL motor-velocity reader state (2026-06 gold key). The SDK zeros
        # get_motor_states().velocity (a TODO bug); the parser holds the real
        # value. _read_real_joint_vel() reads parser.motor_state_<i>.msg.velocity
        # directly + applies per-joint signs, giving a clean velocity for the
        # Cartesian damping -B*J*qdot term. Preferred over finite-diff when the
        # active impedance config has use_real_velocity=True. This is the
        # FALLBACK-agnostic state; the finite-diff path above is the fallback.
        self._real_vel_filtered: np.ndarray = np.zeros(self.N_JOINTS)
        self._real_vel_last_time: float = None
        self._real_vel_raw = np.zeros(self.N_JOINTS)
        self._real_vel_timestamps: np.ndarray = None
        self._real_vel_filter_timestamps: np.ndarray = None
        self._real_vel_filter_update_time = np.zeros(
            self.N_JOINTS, dtype=np.float64)
        self._last_real_velocity_sample_time: float = None
        self._real_velocity_unavailable_since: float = None
        self._last_velocity_source = "finite-diff"

        # No file I/O occurs in the 200 Hz thread.  On a trip this deque is
        # frozen into _last_impedance_trip and exposed through get_debug_state().
        trace_len = max(
            QDOT_TRIP_FRESH_FRAMES,
            int(round(float(control_freq) * IMPEDANCE_TRACE_SECONDS)),
        )
        self._impedance_trace = deque(maxlen=trace_len)
        self._last_impedance_trip = None
        self._last_control_loop_start: float = None
        self._impedance_safety_lock = threading.RLock()

        # MIT mode initialized flag
        self._mit_mode_set = False

    @property
    def running(self) -> bool:
        return self._running

    @property
    def last_error(self):
        return self._last_error

    @property
    def terminal_fault(self) -> bool:
        """Whether the loop stopped because a terminal safety check tripped."""
        return bool(getattr(self, "_terminal_fault", False))

    def start(self) -> None:
        """Start the control loop in a background thread."""
        if getattr(self, "_terminal_fault", False):
            raise RuntimeError("Cannot restart MIT after a terminal fault; explicit recovery is required")
        if getattr(self, "_command_stop_latched", False):
            raise RuntimeError("Cannot restart MIT after a controlled stop; restart the workflow")
        if self._running:
            logger.warning("MIT loop already running")
            return
        if self._thread is not None and self._thread.is_alive():
            raise RuntimeError("Cannot start: previous MIT control thread is still alive")

        # The v111 SDK calls _maybe_set_motion_mode() inside every move_mit().
        # Auto mode would therefore add 7 mode frames per 200 Hz cycle.  Match
        # the validated standalone path: disable auto mode and let
        # _send_mit_commands() set MIT once before the first command batch.
        self._robot.set_auto_set_motion_mode_enabled(False)

        with self._get_actuation_lock():
            self._terminal_fault = False
            with self._impedance_lock:
                if self._impedance_cfg is not None:
                    self._impedance_cfg.active = False
                    self._impedance_generation += 1
        self._running = True
        self._last_error = None
        self._mit_mode_set = False
        self._reset_impedance_safety_state(clear_last_trip=True)
        self._last_control_loop_start = None
        self._thread = threading.Thread(
            target=self._control_loop, daemon=True, name="nero_mit_loop"
        )
        self._thread.start()
        logger.info(f"MIT control loop started at {self._freq} Hz")

    def stop_sender(self) -> None:
        """Join the CAN sender without waiting for optional disk/log cleanup."""
        self.stop(finalize_diagnostics=False)

    def stop(self, *, finalize_diagnostics=True) -> None:
        """Stop the control loop, or raise if the worker cannot be joined."""
        self._running = False
        thread = self._thread
        if thread is not None:
            if thread is threading.current_thread():
                raise RuntimeError("MIT control thread cannot join itself")
            thread.join(timeout=2.0)
            if thread.is_alive():
                try:
                    self._robot.electronic_emergency_stop()
                except Exception as exc:
                    logger.critical(
                        "MIT thread join timed out and emergency stop failed: %s",
                        exc,
                    )
                raise RuntimeError(
                    "MIT control thread did not stop within 2.0s; "
                    "thread reference retained")
            self._thread = None
        with self._get_actuation_lock():
            with self._impedance_lock:
                if self._impedance_cfg is not None and self._impedance_cfg.active:
                    self._impedance_cfg.active = False
                    self._impedance_generation += 1
        self._mit_mode_set = False
        if finalize_diagnostics:
            self.stop_diagnostics()
        logger.info("MIT control loop stopped")

    def update_commands(self, commands: JointCommands) -> None:
        """Update the control commands (thread-safe)."""
        with self._get_actuation_lock():
            if getattr(self, "_terminal_fault", False):
                raise RuntimeError("Cannot update commands after a terminal MIT fault")
            if getattr(self, "_hold_recovering", False):
                raise RuntimeError("MIT hold feedback recovery in progress; no new target accepted")
            if getattr(self, "_command_stop_latched", False):
                raise RuntimeError("Cannot update commands after a latched MIT controlled stop")
            with self._command_lock:
                self._commands = copy.deepcopy(commands)

    def hold_for_controlled_stop(self, commands: JointCommands) -> bool:
        """Atomically replace all moving targets with one fixed MIT hold.

        Only the first call installs the hold. Later outer-loop updates cannot
        reintroduce a velocity reference or an old Cartesian impedance target.
        """
        owned = copy.deepcopy(commands)
        for name in ("pos", "vel", "kp", "kd", "torques"):
            value = np.asarray(getattr(owned, name), dtype=np.float64).ravel()
            if value.shape != (self.N_JOINTS,) or not np.all(np.isfinite(value)):
                raise ValueError(f"Controlled-stop {name} must be a finite 7-vector")
            setattr(owned, name, value.copy())
        if np.any(owned.vel != 0.0) or np.any(owned.torques != 0.0):
            raise ValueError("Controlled-stop velocity and requested torque must be zero")
        if np.any(owned.kp < 0.0) or np.any(owned.kp > 500.0):
            raise ValueError("Controlled-stop kp must be within [0, 500]")
        if np.any(owned.kd < 0.0) or np.any(owned.kd > 5.0):
            raise ValueError("Controlled-stop kd must be within [0, 5]")
        if np.any(owned.pos < NERO_JOINT_LIMITS[:, 0]) or np.any(
            owned.pos > NERO_JOINT_LIMITS[:, 1]
        ):
            raise ValueError("Controlled-stop position must be within physical joint limits")
        with self._get_actuation_lock():
            if getattr(self, "_terminal_fault", False):
                raise RuntimeError("Cannot hold after a terminal MIT fault")
            if not self._running:
                raise RuntimeError("Cannot hold with an inactive MIT control loop")
            if getattr(self, "_command_stop_latched", False):
                return False
            # This snapshot is installed under this same lock only after a
            # complete seven-axis command batch. Diagnostic/state publication
            # happens later and can still describe the preceding cycle.
            sent = getattr(self, "_last_sent_snapshot", None)
            with self._impedance_lock:
                was_impedance = bool(self._impedance_cfg is not None
                                     and self._impedance_cfg.active)
            if was_impedance and sent is not None:
                self._controlled_stop_transition = {
                    "start": copy.deepcopy(sent), "duration_s": 0.4,
                    "started_monotonic": None, "fraction": 0.0,
                }
            else:
                self._controlled_stop_transition = None
            self._controlled_stop_complete = False
            self._controlled_stop_completed_monotonic = None
            with self._command_lock:
                self._commands = owned
            with self._impedance_lock:
                if self._impedance_cfg is not None:
                    self._impedance_cfg.active = False
                self._impedance_generation = (
                    int(getattr(self, "_impedance_generation", 0)) + 1)
            self._command_stop_latched = True
        self.set_diagnostics_phase("controlled_stop_hold")
        return True

    def get_controlled_stop_status(self):
        """The ramp is complete only after its final CAN batch succeeded."""
        with self._get_actuation_lock():
            transition = getattr(self, "_controlled_stop_transition", None)
            with self._command_lock:
                target = (self._commands.pos.copy() if self._command_stop_latched
                          and self._commands is not None else None)
            complete = bool(getattr(self, "_controlled_stop_complete", False))
            return {
                "target_q": target,
                "transition_complete": complete,
                "transition_duration_s": 0.0 if transition is None else transition["duration_s"],
                "transition_completed_monotonic": getattr(
                    self, "_controlled_stop_completed_monotonic", None),
                "phase": "hold" if complete else "transition",
            }

    def _stop_transition_command(self, final, now):
        """Blend firmware PD effort, not independently Kp and angle error.

        Gain-weighted references make Kp*(p-q) and Kd*(v-dq) a convex
        combination of the outgoing and fixed-hold laws for any measured q/dq.
        Independently interpolating Kp and p can multiply the transient effort.
        Gravity is computed afresh by the existing inner loop throughout.
        """
        transition = getattr(self, "_controlled_stop_transition", None)
        if transition is None:
            return final, True
        if transition["started_monotonic"] is None:
            transition["started_monotonic"] = float(now)
        u = float(np.clip((now - transition["started_monotonic"])
                          / transition["duration_s"], 0.0, 1.0))
        weight = u * u * (3.0 - 2.0 * u)
        transition["fraction"] = weight
        start = transition["start"]
        kp = (1.0 - weight) * start["kp"] + weight * final.kp
        kd = (1.0 - weight) * start["kd"] + weight * final.kd
        pos = final.pos.copy()
        vel = final.vel.copy()
        np.divide((1.0 - weight) * start["kp"] * start["pos"]
                  + weight * final.kp * final.pos, kp, out=pos, where=kp > 0.0)
        np.divide((1.0 - weight) * start["kd"] * start["vel"]
                  + weight * final.kd * final.vel, kd, out=vel, where=kd > 0.0)
        return JointCommands(
            pos=pos, vel=vel, kp=kp, kd=kd,
            torques=(1.0 - weight) * start["non_gravity_torque"] + weight * final.torques,
        ), u >= 1.0

    def request_emergency_stop(self, reason: str) -> None:
        """Latch e-stop while serialized against the final MIT CAN send."""
        with self._get_actuation_lock():
            if not self._hard_stop_locked(str(reason)):
                raise RuntimeError(self._last_error)

    def start_diagnostics(self, path, queue_capacity: int = 4096) -> None:
        """Start optional continuous logging outside the control thread.

        Includes position holds and gripper-operation intervals because the
        producer belongs to the inner loop, not to individual trajectories.
        """
        if int(queue_capacity) < 1:
            raise ValueError("MIT diagnostics queue_capacity must be positive")
        writer = getattr(self, "_diagnostics_writer", None)
        if writer is not None:
            if writer.path == str(Path(path)) and writer.accepting:
                return
            raise RuntimeError("MIT diagnostics already configured for this loop")
        self._diagnostics_writer = _MITDiagnosticWriter(path, int(queue_capacity))

    def set_diagnostics_phase(self, label: str) -> None:
        self._diagnostics_phase = str(label)

    def record_control_event(self, event, **values):
        """Non-blocking owner-side events, including final device handoff."""
        writer = getattr(self, "_diagnostics_writer", None)
        if writer is not None and writer.accepting:
            try:
                writer.submit({"type": "control_event", "event": str(event),
                               "wall_time": time.time(), "monotonic_time": time.monotonic(),
                               **values})
            except Exception:
                pass  # A diagnostic failure cannot invalidate physical hold.

    def ensure_control_diagnostics(self, path):
        status = self.get_diagnostics_status()
        if not status.get("enabled"):
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            self.start_diagnostics(path)
        elif not status.get("accepting", False):
            raise RuntimeError(f"Existing MIT diagnostics are inactive: {status.get('error')}")
        self.set_diagnostics_phase("visual_servo")
        return self.get_diagnostics_status()

    def get_diagnostics_status(self) -> dict:
        writer = getattr(self, "_diagnostics_writer", None)
        try:
            return {"enabled": False} if writer is None else {"enabled": True, **writer.status()}
        except Exception as exc:
            return {"enabled": True, "error": f"Diagnostic status failed: {exc}", "closed": False}

    def stop_diagnostics(self, timeout: float = 5.0) -> dict:
        writer = getattr(self, "_diagnostics_writer", None)
        if writer is None:
            return {"enabled": False}
        try:
            return {"enabled": True, **writer.close(float(timeout))}
        except Exception as exc:
            return {"enabled": True, "error": f"Diagnostic close failed: {exc}", "closed": False}

    def _record_control_cycle(
        self, loop_dt, q, dq, cmd_pos, vel_cmd, kp_cmd, kd_cmd,
        requested_torque, grav_torque, total_torque, torque, control_path,
    ) -> None:
        writer = getattr(self, "_diagnostics_writer", None)
        if writer is None or not writer.accepting:
            return
        # Read-only parser sampling: do not invoke SDK getters which can mutate
        # velocity fields. Missing raw velocity remains missing, not zero.
        try:
            raw_sample = read_real_qdot_sample(self._robot, n_joints=self.N_JOINTS)
        except Exception:
            raw_sample = None
        writer.submit({
            "type": "cycle", "wall_time": time.time(),
            "monotonic_time": time.monotonic(), "loop_dt": float(loop_dt),
            "phase": getattr(self, "_diagnostics_phase", "unknown"),
            "control_path": control_path,
            "q_actual": np.asarray(q).copy(), "dq_actual": np.asarray(dq).copy(),
            "dq_source": (getattr(self, "_last_velocity_source", "unknown")
                          if control_path == "impedance" else "finite-diff"),
            "dq_finite_difference": np.asarray(self._last_fdiff_velocity).copy(),
            "dq_parser_sign_corrected": None if raw_sample is None else raw_sample[0].copy(),
            "velocity_feedback_timestamps": None if raw_sample is None else raw_sample[1],
            "position_feedback_timestamp": getattr(self, "_joint_pos_timestamp", None),
            "q_command": np.asarray(cmd_pos).copy(), "dq_command": np.asarray(vel_cmd).copy(),
            "kp": np.asarray(kp_cmd).copy(), "kd": np.asarray(kd_cmd).copy(),
            "q_command_error": np.asarray(cmd_pos) - np.asarray(q),
            "kp_12bit_estimate": np.floor(np.asarray(kp_cmd) * 4095.0 / 500.0) * (500.0 / 4095.0),
            "tau_requested": np.asarray(requested_torque).copy(),
            "tau_gravity": np.asarray(grav_torque).copy(),
            "tau_ff_sent": np.asarray(total_torque).copy(),
            "tau_feedback": np.asarray(torque).copy(),
            "torque_feedback_timestamps": list(getattr(
                self, "_last_feedback_torque_timestamps", [None] * self.N_JOINTS)),
            "torque_feedback_valid": np.asarray(getattr(
                self, "_last_feedback_torque_valid", np.zeros(self.N_JOINTS, dtype=bool))).copy(),
            "controlled_stop_latched": bool(getattr(self, "_command_stop_latched", False)),
            "controlled_stop_reason": getattr(self, "_controlled_stop_reason", None),
            "controlled_stop_confirmed": bool(getattr(self, "_buffer_stop_confirmed", False)),
            "stop_transition_complete": bool(getattr(self, "_controlled_stop_complete", False)),
            "stop_transition_fraction": (None if getattr(self, "_controlled_stop_transition", None) is None
                                         else self._controlled_stop_transition["fraction"]),
            "position_feedback_timestamps": (
                None if getattr(self, "_feedback_health_sample", None) is None
                else self._feedback_health_sample["position_timestamps"].copy()),
            "feedback_guard_enabled": bool(getattr(self, "_feedback_guard_enabled", False)),
            "timing": self._cycle_timing(),
        })

    def _begin_cycle_stage(self, name):
        now = time.perf_counter()
        previous = getattr(self, "_cycle_stage", None)
        if previous is not None:
            self._cycle_stage_durations[previous] = now - self._cycle_stage_start
        self._cycle_stage, self._cycle_stage_start = str(name), now

    def _cycle_timing(self, now=None):
        now = time.perf_counter() if now is None else now
        durations = dict(getattr(self, "_cycle_stage_durations", {}))
        stage = getattr(self, "_cycle_stage", None)
        if stage is not None:
            durations[stage] = now - self._cycle_stage_start
        started = getattr(self, "_last_control_loop_start", None)
        return {"loop_dt_s": getattr(self, "_cycle_loop_dt", None),
                "cycle_elapsed_s": None if started is None else now - started,
                "active_stage": stage, "stage_durations_s": durations,
                "sent_joint_count": getattr(self, "_cycle_sent_joints", 0)}

    def _record_fault_diagnostics(self, reason, fault_wall_time, fault_monotonic_time, timing):
        writer = getattr(self, "_diagnostics_writer", None)
        if writer is None or not writer.accepting:
            return
        monitor = getattr(self, "_feedback_health", None)
        attempted = (getattr(monitor, "last_read_sample", None)
                     if getattr(self, "_cycle_feedback_read_started", False) else None)
        # A new diagnostic cache read must never refresh the control snapshot
        # or make an old q/torque calculation pass the safety gate.
        raw_cache = FeedbackHealthMonitor.timestamp_snapshot(self._robot)
        cache_wall_time = time.time()
        writer.submit({
            "type": "fault", "wall_time": fault_wall_time,
            "monotonic_time": fault_monotonic_time, "reason": str(reason),
            "phase": getattr(self, "_diagnostics_phase", "unknown"),
            "timing": timing,
            "feedback_snapshot": FeedbackHealthMonitor.describe_snapshot(attempted, fault_wall_time),
            "parser_cache_after_stop_request": {
                **FeedbackHealthMonitor.describe_snapshot(raw_cache, cache_wall_time),
                "captured_monotonic_time": time.monotonic(), "diagnostic_only": True,
            },
        })

    def get_joint_state(self) -> JointState:
        """Get the latest joint state (thread-safe)."""
        with self._state_lock:
            if self._joint_state is None:
                return None
            return copy.deepcopy(self._joint_state)

    def set_gravity_enabled(self, enabled: bool) -> None:
        """Enable or disable automatic gravity torque overlay."""
        with self._get_actuation_lock():
            if getattr(self, "_command_stop_latched", False):
                raise RuntimeError("Cannot change gravity during a latched MIT controlled stop")
            self._gravity_enabled = bool(enabled)

    def set_gravity_factors(self, factors) -> None:
        values = np.asarray(factors, dtype=np.float64).ravel()
        if values.shape != (self.N_JOINTS,):
            raise ValueError(
                f"gravity_factors must have shape ({self.N_JOINTS},), got {values.shape}"
            )
        with self._get_actuation_lock():
            if getattr(self, "_command_stop_latched", False):
                raise RuntimeError("Cannot change gravity during a latched MIT controlled stop")
            self._gravity_factors = values.copy()

    def _get_impedance_safety_lock(self):
        lock = getattr(self, "_impedance_safety_lock", None)
        if lock is None:
            lock = threading.RLock()
            self._impedance_safety_lock = lock
        return lock

    def _get_actuation_lock(self):
        lock = getattr(self, "_actuation_lock", None)
        if lock is None:
            lock = threading.Lock()
            self._actuation_lock = lock
        return lock

    def get_debug_state(self, include_trace: bool = False) -> dict:
        """Get latest torque debug values."""
        with self._impedance_lock:
            impedance_active = bool(
                self._impedance_cfg is not None
                and self._impedance_cfg.active
            )
        with self._get_impedance_safety_lock():
            with self._state_lock:
                debug = {
                    "running": self._running,
                    "last_error": self._last_error,
                    "terminal_fault": bool(
                        getattr(self, "_terminal_fault", False)
                    ),
                    "controlled_stop_latched": bool(
                        getattr(self, "_command_stop_latched", False)
                    ),
                    "controlled_stop_reason": getattr(self, "_controlled_stop_reason", None),
                    "buffer_stop_confirmed": bool(getattr(self, "_buffer_stop_confirmed", False)),
                    "hold_recovering": bool(getattr(self, "_hold_recovering", False)),
                    "gravity_enabled": self._gravity_enabled,
                    "impedance_active": impedance_active,
                    "control_path": getattr(
                        self, "_last_control_path", "unknown"
                    ),
                    "impedance_cycle_count": int(getattr(
                        self, "_impedance_cycle_count", 0
                    )),
                    "impedance_torque": np.asarray(getattr(
                        self, "_last_impedance_torque",
                        np.zeros(self.N_JOINTS),
                    )).copy(),
                    "gravity_torque": self._last_gravity_torque.copy(),
                    "total_torque": self._last_total_torque.copy(),
                    "feedback_torque": self._last_feedback_torque.copy(),
                    "impedance_trip": copy.deepcopy(self._last_impedance_trip),
                    "last_sent_command": copy.deepcopy(getattr(self, "_last_sent_snapshot", None)),
                    "stop_transition_complete": bool(getattr(self, "_controlled_stop_complete", False)),
                    "stop_transition_fraction": (
                        None if getattr(self, "_controlled_stop_transition", None) is None
                        else self._controlled_stop_transition["fraction"]),
                    "diagnostics": self.get_diagnostics_status(),
                }
                if include_trace:
                    debug["impedance_trace"] = copy.deepcopy(
                        list(self._impedance_trace))
                return debug

    def _reset_impedance_safety_state(self, clear_last_trip: bool = False) -> None:
        """Reset per-engagement speed counters and the diagnostic trace."""
        with self._get_impedance_safety_lock():
            self._qdot_over_count = np.zeros(self.N_JOINTS, dtype=np.int64)
            self._qdot_last_timestamp = None
            self._qdot_timestamp_last_change = np.zeros(
                self.N_JOINTS, dtype=np.float64)
            self._real_vel_timestamps = None
            self._real_vel_filter_timestamps = None
            self._real_vel_filter_update_time = np.zeros(
                self.N_JOINTS, dtype=np.float64)
            self._real_vel_last_time = None
            self._last_real_velocity_sample_time = None
            self._real_velocity_unavailable_since = None
            trace = getattr(self, "_impedance_trace", None)
            if trace is None:
                trace = deque(maxlen=max(
                    QDOT_TRIP_FRESH_FRAMES,
                    int(round(float(getattr(self, "_freq", 200.0))
                              * IMPEDANCE_TRACE_SECONDS)),
                ))
                self._impedance_trace = trace
            trace.clear()
            if clear_last_trip:
                state_lock = getattr(self, "_state_lock", None)
                if state_lock is None:
                    self._last_impedance_trip = None
                else:
                    with state_lock:
                        self._last_impedance_trip = None

    # =================== In-loop Cartesian impedance ===================

    def set_cartesian_impedance(self, cfg: ImpedanceConfig) -> None:
        """Enable in-loop Cartesian impedance with the given config.

        Atomically installs an owned copy with active=True. The 200 Hz loop
        picks it up on the next cycle. Requires a pin_model to have been
        supplied to the constructor; otherwise raises.
        """
        if self._pin is None:
            raise RuntimeError(
                "In-loop Cartesian impedance requires a pin_model; construct "
                "NeroMITLoop with pin_model=<NeroPinocchioModel>.")
        if not isinstance(cfg, ImpedanceConfig):
            raise TypeError("cfg must be an ImpedanceConfig")

        with self._get_actuation_lock():
            if getattr(self, "_terminal_fault", False):
                raise RuntimeError(
                    "Cannot engage Cartesian impedance after a terminal MIT "
                    "fault; restart the controller first")
            if getattr(self, "_command_stop_latched", False):
                raise RuntimeError("Cannot engage Cartesian impedance after a latched MIT controlled stop")
            if getattr(self, "_hold_recovering", False):
                raise RuntimeError("Cannot engage Cartesian impedance during MIT hold recovery")
            with self._impedance_lock:
                was_active = bool(
                    self._impedance_cfg is not None
                    and self._impedance_cfg.active
                )
                owned_cfg = copy.deepcopy(cfg)

                required_arrays = {
                    "target_tcp": (3,),
                    "R_target": (3, 3),
                    "K": (6,),
                    "B": (6,),
                    "joint_damping": (self.N_JOINTS,),
                    "tau_limit_per_joint": (self.N_JOINTS,),
                }
                optional_arrays = {
                    "q_rest": (self.N_JOINTS,),
                    "target_twist": (6,),
                    "kp_joint": (self.N_JOINTS,),
                    "kd_joint": (self.N_JOINTS,),
                    "q_follow": (self.N_JOINTS,),
                    "follow_kp": (self.N_JOINTS,),
                    "joint_hold_kp": (self.N_JOINTS,),
                    "joint_hold_kd": (self.N_JOINTS,),
                    "velocity_signs": (self.N_JOINTS,),
                    "joint_weights": (self.N_JOINTS,),
                    "coulomb": (self.N_JOINTS,),
                }
                for name, shape in required_arrays.items():
                    value = getattr(owned_cfg, name)
                    if value is None:
                        raise ValueError(f"{name} is required")
                    array = np.asarray(value, dtype=np.float64)
                    if array.shape != shape:
                        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
                    if not np.all(np.isfinite(array)):
                        raise ValueError(f"{name} must contain only finite values")
                    setattr(owned_cfg, name, array.copy())
                for name, shape in optional_arrays.items():
                    value = getattr(owned_cfg, name)
                    if value is None:
                        continue
                    array = np.asarray(value, dtype=np.float64)
                    if array.shape != shape:
                        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
                    if not np.all(np.isfinite(array)):
                        raise ValueError(f"{name} must contain only finite values")
                    setattr(owned_cfg, name, array.copy())

                torque_limits = np.asarray(
                    owned_cfg.tau_limit_per_joint, dtype=np.float64)
                if np.any(torque_limits <= 0.0):
                    raise ValueError("tau_limit_per_joint must contain positive values")
                if np.any(owned_cfg.K < 0.0) or np.any(owned_cfg.B < 0.0):
                    raise ValueError("Cartesian K and B must contain non-negative values")
                if np.any(owned_cfg.joint_damping < 0.0):
                    raise ValueError("joint_damping must contain non-negative values")
                gain_limits = {
                    "kp_joint": 500.0,
                    "joint_hold_kp": 500.0,
                    "kd_joint": 5.0,
                    "joint_hold_kd": 5.0,
                    "follow_kp": 500.0,
                }
                for name, upper in gain_limits.items():
                    value = getattr(owned_cfg, name)
                    if value is None:
                        continue
                    if np.any(value < 0.0) or np.any(value > upper):
                        raise ValueError(
                            f"{name} must be within [0, {upper}]")
                if owned_cfg.joint_weights is not None:
                    weights = np.asarray(owned_cfg.joint_weights, dtype=np.float64)
                    if np.any(weights <= 0.0):
                        raise ValueError("joint_weights must contain positive values")
                smooth_width = float(getattr(owned_cfg, "coulomb_smooth_width", 0.0))
                if not np.isfinite(smooth_width) or smooth_width < 0.0:
                    raise ValueError(
                        "coulomb_smooth_width must be finite and non-negative"
                    )
                if owned_cfg.task_torque_limit is not None:
                    task_limit = float(owned_cfg.task_torque_limit)
                    if not np.isfinite(task_limit) or task_limit <= 0.0:
                        raise ValueError("task_torque_limit must be finite and positive")
                for name in (
                    "max_position_error",
                    "max_orientation_error_rad",
                ):
                    value = getattr(owned_cfg, name)
                    if value is None:
                        continue
                    value = float(value)
                    if not np.isfinite(value) or value < 0.0:
                        raise ValueError(
                            f"{name} must be finite and non-negative"
                        )
                    setattr(owned_cfg, name, value)
                if owned_cfg.follow_max_error is not None:
                    follow_max_error = float(owned_cfg.follow_max_error)
                    if not np.isfinite(follow_max_error) or follow_max_error < 0.0:
                        raise ValueError(
                            "follow_max_error must be finite and non-negative"
                        )
                if (
                    not np.isfinite(float(owned_cfg.qdot_trip_fast))
                    or float(owned_cfg.qdot_trip_fast) <= 0.0
                ):
                    raise ValueError("qdot_trip_fast must be finite and positive")

                owned_cfg.active = True
                if not was_active:
                    # Reset before publishing active so the control thread can
                    # never observe a new engagement with old counters/filter
                    # timestamps.
                    self._reset_impedance_safety_state(clear_last_trip=True)
                    self._impedance_generation = (
                        int(getattr(self, "_impedance_generation", 0)) + 1)
                self._impedance_cfg = owned_cfg

        if not was_active:
            weights = owned_cfg.joint_weights
            if (
                weights is not None
                and not np.allclose(weights, np.full(self.N_JOINTS, weights[0]))
            ):
                logger.warning(
                    "Non-uniform joint_weights post-multiply J^T torque and "
                    "do not preserve Cartesian passivity. Use all ones for the "
                    "safe production baseline.")

    def disable_cartesian_impedance(self) -> None:
        """Disable in-loop impedance and fall back to the buffered command path.

        Also writes a zero-gravity-hold JointCommand into the buffer so the
        fallback is graceful (no torque spike from whatever was last buffered).
        No-op if impedance was never set.
        """
        with self._get_actuation_lock():
            if getattr(self, "_command_stop_latched", False):
                # Shutdown/servo cleanup must not replace the fixed safety
                # hold with this method's zero-kp gravity-compensation buffer.
                return
            with self._impedance_lock:
                if self._impedance_cfg is None:
                    return

            # Install the safe buffer before publishing inactive.  kp=0 makes
            # the position field irrelevant, but use fresh feedback when
            # available for clean diagnostics.
            try:
                q = self._read_joint_pos()
            except Exception as exc:
                logger.warning(
                    "Ignoring invalid joint feedback while disabling "
                    "Cartesian impedance: %s",
                    exc,
                )
                q = None
            if q is None:
                with self._state_lock:
                    state = self._joint_state
                    q = None if state is None else state.pos.copy()
            if q is not None:
                q = np.asarray(q, dtype=np.float64).ravel()
            if (
                q is None
                or q.shape != (self.N_JOINTS,)
                or not np.all(np.isfinite(q))
            ):
                q = np.zeros(self.N_JOINTS, dtype=np.float64)
            with self._command_lock:
                self._commands = JointCommands(
                    pos=q.copy(),
                    vel=np.zeros(self.N_JOINTS),
                    kp=np.zeros(self.N_JOINTS),
                    kd=np.array([1.0, 1.0, 1.0, 0.5, 0.3, 0.2, 0.2]),
                    torques=np.zeros(self.N_JOINTS),
                )
            with self._impedance_lock:
                self._impedance_cfg.active = False
                self._impedance_generation = (
                    int(getattr(self, "_impedance_generation", 0)) + 1)
            self._reset_impedance_safety_state(clear_last_trip=False)

    def _compute_impedance_torque(self, q, dq, cfg: ImpedanceConfig):
        """Author-validated Panthera Cartesian impedance torque, computed fresh
        each 200 Hz cycle. Returns (tau, kp, kd, cmd_pos).

        Law (see cartesian_impedance.py + data_types.ImpedanceConfig):
            F       = K @ e  -  B @ (J @ dq)        (Cartesian; B tiny ~0.01)
            or, when ``target_twist`` is supplied,
            F       = K @ e  +  B @ (target_twist - J @ dq)
            tau_cart = DLS: J^T (J J^T + lam^2 I)^-1 F   (or plain J^T@F)
            tau_damp = -joint_damping @ dq          (PRIMARY, joint space)
            tau_null = N^T (kq (q_rest - q))        (7-DOF redundancy, optional)
            tau = tau_cart + tau_damp + tau_null     (gravity added by the caller)
            p_des = q_follow, kp = follow_kp       (optional visual feed-forward)
        Optional (default None => OFF; ports the standalone tuning-harness law,
        test_official_cartesian_impedance.py, so the in-loop law can reproduce
        that exact validated feel):
            tau_cart *= joint_weights               (per-joint task-torque scaling)
            clip(tau_cart, +-task_torque_limit)      (spring clip BEFORE gravity)
            tau += coulomb * sign(q_follow - q)      (or q_rest for fixed targets, 0.5 deg deadband)
        Orientation error uses the CORRECT single rotation to world frame
        (R_tcp @ rotvec(R_tcp^T @ R_target)), matching the validated C++ reference.
        """
        pos, R_tcp, J = self._pin.fk_and_jacobian(q)
        e_pos = _clamp_vector_norm(
            cfg.target_tcp - pos,
            cfg.max_position_error,
        )
        # World-frame orientation error (single rotation). NOTE: this differs from
        # the old test script which double-rotated (R_tcp @ (R_tcp @ rotvec));
        # that was masked by tiny K_ori. The validated reference uses one rotation.
        e_ori_world = _clamp_vector_norm(
            R_tcp @ Rot.from_matrix(R_tcp.T @ cfg.R_target).as_rotvec(),
            cfg.max_orientation_error_rad,
        )

        F = impedance_wrench(
            e_pos,
            e_ori_world,
            J,
            dq,
            cfg.K,
            cfg.B,
            target_twist=cfg.target_twist,
        )
        if cfg.use_dls_mapping:
            tau_cart = dls_task_torque(J, F, cfg.dls_lambda)
        else:
            tau_cart = task_torque(J, F)
        # Optional pieces ported from the standalone tuning harness
        # (test_official_cartesian_impedance.py); all default None => skipped, so
        # the existing panthera-real law is bit-identical unless opted in:
        #   joint_weights     -> per-joint scaling of the task torque (down-weight wrist)
        #   task_torque_limit -> spring clip BEFORE gravity/coulomb (harness uses +-10)
        #   coulomb           -> static-friction feed-forward (0.5 deg deadband)
        # Order matches the harness: weight -> spring clip on tau_cart, then the
        # joint-space damping / null-space / coulomb are layered on top.
        if cfg.joint_weights is not None:
            tau_cart = np.asarray(cfg.joint_weights, dtype=np.float64).ravel() * tau_cart
        if cfg.task_torque_limit is not None:
            tau_cart = np.clip(tau_cart, -float(cfg.task_torque_limit), float(cfg.task_torque_limit))

        tau = tau_cart + joint_damping_torque(dq, cfg.joint_damping)

        if cfg.null_kq > 0.0 and cfg.q_rest is not None:
            tau_null = null_space_torque(
                q_now=q, qdot_measured=dq, jacobian=J,
                q_rest=cfg.q_rest, kq=cfg.null_kq,
            )
            if np.all(np.isfinite(tau_null)):
                tau = tau + tau_null

        # Use the moving visual reference for friction assist as well as for a
        # fixed target.  When coulomb_smooth_width > 0 the helper uses a tanh
        # transition, so the same impedance parameters apply in motion and at
        # rest without the previous target-moving mode switch or sign kick.
        coulomb_ref = cfg.q_follow if cfg.q_follow is not None else cfg.q_rest
        if cfg.coulomb is not None and coulomb_ref is not None:
            tau_c = coulomb_compensation(
                coulomb_ref,
                q,
                cfg.coulomb,
                smooth_width_rad=float(
                    getattr(cfg, "coulomb_smooth_width", 0.0)
                ),
            )
            if np.all(np.isfinite(tau_c)):
                tau = tau + tau_c

        if not np.all(np.isfinite(tau)):
            tau = np.zeros(self.N_JOINTS)

        kp = (cfg.kp_joint.copy() if cfg.kp_joint is not None
              else np.zeros(self.N_JOINTS))
        kd = (cfg.kd_joint.copy() if cfg.kd_joint is not None
              else np.zeros(self.N_JOINTS))
        cmd_pos = q.copy()
        # Hybrid visual tracking: low firmware position Kp supplies continuous
        # motion through static friction while Cartesian J^T impedance remains
        # the compliant task-level force. Clip only this cycle's command; never
        # rewrite q_follow when an operator pushes the arm.
        if cfg.q_follow is not None and cfg.follow_kp is not None:
            follow_kp = np.asarray(cfg.follow_kp, dtype=np.float64).ravel()
            follow_ref = np.asarray(cfg.q_follow, dtype=np.float64).ravel()
            follow_error = follow_ref - q
            if cfg.follow_max_error is not None:
                bound = float(cfg.follow_max_error)
                if bound >= 0.0:
                    follow_error = np.clip(follow_error, -bound, bound)
            mask = follow_kp > 0.0
            if np.any(mask):
                kp[mask] = follow_kp[mask]
                cmd_pos[mask] = q[mask] + follow_error[mask]
        # Firmware-side position hold on selected joints. The pure Cartesian
        # law cannot hold joint7 (tool-roll): it drifts untouched (~10s, weak
        # task coupling) and oscillates to the limit on a push (light inertia,
        # high K_rot, no Cartesian velocity damping). Holding it with the
        # FIRMWARE position PD (kp*(q_rest-q) + kd*(0-v)) — the same mechanism
        # that makes mit-position/zero-G stable — fixes both: the firmware uses
        # its own clean velocity, so the hold is damped and stable, unlike a
        # t_ff torque hold. Other joints stay pure-torque Cartesian impedance.
        if cfg.joint_hold_kp is not None and cfg.q_rest is not None:
            hold = np.asarray(cfg.joint_hold_kp, dtype=np.float64).ravel()
            mask = hold > 0.0
            if np.any(mask):
                kp[mask] = hold[mask]
                cmd_pos[mask] = np.asarray(cfg.q_rest, dtype=np.float64).ravel()[mask]
                if cfg.joint_hold_kd is not None:
                    hold_kd = np.asarray(
                        cfg.joint_hold_kd, dtype=np.float64).ravel()
                    kd[mask] = hold_kd[mask]
        return tau, kp, kd, cmd_pos

    def _read_joint_pos(self) -> np.ndarray:
        """Read current joint positions from the driver's parser cache."""
        if getattr(self, "_feedback_guard_enabled", False):
            sample = self._feedback_health.read(
                self._robot, time.time(), time.monotonic())
            self._feedback_health_sample = sample
            self._joint_pos_timestamp = float(np.min(sample["position_timestamps"]))
            return sample["q"].copy()
        ja = self._robot.get_joint_angles()
        if ja is None:
            return None
        stamp = getattr(ja, "timestamp", None)
        try:
            stamp = float(stamp) if stamp is not None else None
        except (TypeError, ValueError):
            stamp = None
        self._joint_pos_timestamp = (
            stamp if stamp is not None and np.isfinite(stamp) else None)
        q = np.asarray(ja.msg, dtype=np.float64).ravel()
        if q.shape != (self.N_JOINTS,) or not np.all(np.isfinite(q)):
            raise ValueError(
                "Invalid joint-position feedback: "
                f"expected finite shape ({self.N_JOINTS},), got "
                f"shape={q.shape}, values={q.tolist()}")
        return q

    def _read_joint_vel(self, q: np.ndarray = None, tau_override: float = None) -> np.ndarray:
        """Joint velocity for external readers (impedance damping, logging).

        The v111 driver's get_motor_states().velocity is always 0 (not reported),
        so we synthesize velocity from finite differences of joint positions with
        an exponential low-pass (tau ~ 8 ms at the loop rate) to suppress
        encoder quantization noise. This is NOT used by the firmware kd term
        (that uses its own internal velocity); it only populates JointState.vel
        and feeds the in-loop impedance velocity term.

        ``tau_override`` lets the in-loop impedance use its own (possibly larger)
        low-pass tau without affecting the tau used for the published state.
        """
        if q is None:
            q = self._read_joint_pos()
            if q is None:
                return np.zeros(self.N_JOINTS)
        now = time.monotonic()
        position_timestamp = getattr(self, "_joint_pos_timestamp", None)
        if self._vel_last_pos is None or self._vel_last_time is None:
            self._vel_last_pos = q.copy()
            self._vel_last_time = now
            self._vel_last_pos_timestamp = position_timestamp
            self._last_fdiff_raw = np.zeros(self.N_JOINTS)
            self._last_fdiff_velocity = self._vel_filtered.copy()
            return self._vel_filtered.copy()
        if (
            position_timestamp is not None
            and self._vel_last_pos_timestamp is not None
            and position_timestamp == self._vel_last_pos_timestamp
        ):
            # The 200 Hz loop may poll the same cached position frame more than
            # once. Do not inject a zero sample or shorten dt before the next
            # real position update.
            return self._vel_filtered.copy()
        dt = now - self._vel_last_time
        if dt < 1e-4:
            return self._vel_filtered.copy()
        raw_vel = (q - self._vel_last_pos) / dt
        # Exponential low-pass. tau kept small (default 8ms) to avoid lag that
        # destabilizes the Cartesian damping term. See self._vel_filter_tau.
        tau = self._vel_filter_tau if tau_override is None else float(tau_override)
        alpha = dt / (tau + dt)
        self._vel_filtered = (alpha * raw_vel + (1.0 - alpha) * self._vel_filtered)
        self._vel_last_pos = q.copy()
        self._vel_last_time = now
        self._vel_last_pos_timestamp = position_timestamp
        self._last_fdiff_raw = raw_vel.copy()
        self._last_fdiff_velocity = self._vel_filtered.copy()
        return self._vel_filtered.copy()

    def _read_real_joint_vel(self, cfg: ImpedanceConfig):
        """Read REAL joint velocity straight off the parser's 0x5 high-speed
        feedback (bypassing get_motor_states, which zeros velocity via a SDK
        `# TODO: remove this after the bug is fixed` line), then apply this
        loop's light low-pass.

        The raw sign-corrected read is shared via ``real_velocity.read_real_qdot``
        (single source of truth, used by the standalone tuning harness too). The
        low-pass state is per-loop-instance (``_real_vel_filtered``); the real
        value is already firmware-filtered, so tau is kept tiny.

        The torque reader also prefers parser-direct feedback, avoiding the SDK
        getter that mutates parser velocity. This read still happens before
        torque publication in each control cycle.

        Returns the (7,) sign-corrected, lightly low-passed velocity, or None
        if parser fields/timestamps are unavailable. Production impedance then
        zeros Cartesian B and trips the source watchdog; finite difference is
        used only when explicitly selected in the config.
        """
        sample = read_real_qdot_sample(
            self._robot, cfg.velocity_signs, n_joints=self.N_JOINTS)
        if sample is None:
            now = time.monotonic()
            if self._real_velocity_unavailable_since is None:
                self._real_velocity_unavailable_since = now
            self._real_vel_raw = np.full(self.N_JOINTS, np.nan)
            self._real_vel_timestamps = None
            return None
        v, timestamps = sample
        if timestamps is None:
            # Production damping requires a frame-identity token for every
            # joint.  Otherwise a cached velocity cannot be distinguished from
            # five fresh samples and a frozen sensor has no watchdog.
            now = time.monotonic()
            if self._real_velocity_unavailable_since is None:
                self._real_velocity_unavailable_since = now
            self._real_vel_raw = v.copy()
            self._real_vel_timestamps = None
            return None
        self._last_real_velocity_sample_time = time.monotonic()
        self._real_velocity_unavailable_since = None
        self._real_vel_raw = v.copy()
        self._real_vel_timestamps = (
            None if timestamps is None else timestamps.copy())

        # Light low-pass: the real value is already firmware-filtered; tau is
        # kept tiny (cfg.real_vel_filter_tau) only to shave residual sensor
        # noise without adding damping-relevant lag.
        tau = float(getattr(cfg, "real_vel_filter_tau", 0.0))
        now = time.monotonic()
        previous = getattr(self, "_real_vel_filter_timestamps", None)
        update_times = getattr(
            self,
            "_real_vel_filter_update_time",
            np.zeros(self.N_JOINTS, dtype=np.float64),
        )
        first = previous is None
        if first:
            previous = np.full(self.N_JOINTS, np.nan)
        for joint in range(self.N_JOINTS):
            fresh = first or timestamps[joint] != previous[joint]
            if not fresh:
                continue
            last_update = float(update_times[joint])
            if last_update <= 0.0 or tau <= 0.0:
                self._real_vel_filtered[joint] = v[joint]
            else:
                dt = now - last_update
                if dt > 1e-4:
                    alpha = dt / (tau + dt)
                    self._real_vel_filtered[joint] = (
                        alpha * v[joint]
                        + (1.0 - alpha) * self._real_vel_filtered[joint]
                    )
            update_times[joint] = now
        self._real_vel_filter_timestamps = timestamps.copy()
        self._real_vel_filter_update_time = update_times
        self._real_vel_last_time = now
        return self._real_vel_filtered.copy()

    def _read_impedance_joint_vel(
        self, q: np.ndarray, cfg: ImpedanceConfig
    ) -> np.ndarray:
        """Update finite-difference backup, then prefer parser velocity."""
        fallback_dq = self._read_joint_vel(
            q, tau_override=cfg.vel_filter_tau)
        self._last_fdiff_velocity = fallback_dq.copy()
        if cfg.use_real_velocity:
            real_dq = self._read_real_joint_vel(cfg)
            if real_dq is not None:
                self._last_velocity_source = "real"
                return real_dq
            # Do not silently feed a delayed finite-difference estimate into a
            # B tuned for parser velocity. Zero the Cartesian velocity input;
            # the source-loss watchdog hard-stops after a short grace period
            # while firmware KD remains active.
            self._last_velocity_source = "real-unavailable"
            # Zeroing the Cartesian velocity term is safer than holding a
            # stale, possibly sign-opposed damping sample. Firmware joint KD
            # stays active during the short source-loss grace period.
            return np.zeros(self.N_JOINTS, dtype=np.float64)
        self._last_velocity_source = "finite-diff"
        return fallback_dq

    def _update_qdot_trip(
        self,
        dq: np.ndarray,
        limit: float,
        timestamps: np.ndarray = None,
        now: float = None,
    ):
        with self._get_impedance_safety_lock():
            return self._update_qdot_trip_unlocked(
                dq, limit, timestamps=timestamps, now=now)

    def _update_qdot_trip_unlocked(
        self,
        dq: np.ndarray,
        limit: float,
        timestamps: np.ndarray = None,
        now: float = None,
    ):
        """Advance per-joint runaway counters using fresh feedback only.

        Returns ``(speed_joint, stale_joint, fresh_mask)``.  Joint indices are
        zero-based and ``None`` means that condition has not tripped.
        """
        qdot = np.asarray(dq, dtype=np.float64).ravel()
        if qdot.shape != (self.N_JOINTS,):
            raise ValueError(
                f"dq must have shape ({self.N_JOINTS},), got {qdot.shape}")
        now = time.monotonic() if now is None else float(now)

        counts = getattr(self, "_qdot_over_count", None)
        if not isinstance(counts, np.ndarray) or counts.shape != (self.N_JOINTS,):
            counts = np.zeros(self.N_JOINTS, dtype=np.int64)
        last_change = getattr(
            self, "_qdot_timestamp_last_change",
            np.zeros(self.N_JOINTS, dtype=np.float64),
        )

        timestamp_array = None
        if timestamps is not None:
            candidate = np.asarray(timestamps, dtype=np.float64).ravel()
            if candidate.shape == (self.N_JOINTS,) and np.all(np.isfinite(candidate)):
                timestamp_array = candidate

        stale_joint = None
        if timestamp_array is None:
            # No timestamp support: preserve the original fail-safe behavior.
            fresh = np.ones(self.N_JOINTS, dtype=bool)
            self._qdot_last_timestamp = None
        else:
            previous = getattr(self, "_qdot_last_timestamp", None)
            if previous is None:
                fresh = np.ones(self.N_JOINTS, dtype=bool)
                last_change[:] = now
            else:
                # Exact comparison is intentional: these values identify CAN
                # frames; np.isclose on epoch-like timestamps would collapse
                # distinct frames because of its relative tolerance.
                fresh = timestamp_array != previous
                last_change[fresh] = now
            self._qdot_last_timestamp = timestamp_array.copy()

            stale = (now - last_change) > REAL_VELOCITY_STALE_TIMEOUT
            if np.any(stale):
                stale_joint = int(np.argmax(now - last_change))

        over = np.abs(qdot) > float(limit)
        for joint in np.flatnonzero(fresh):
            counts[joint] = counts[joint] + 1 if over[joint] else 0
        self._qdot_over_count = counts
        self._qdot_timestamp_last_change = last_change

        tripped = counts >= QDOT_TRIP_FRESH_FRAMES
        speed_joint = None
        if np.any(tripped):
            candidates = np.flatnonzero(tripped)
            speed_joint = int(candidates[np.argmax(np.abs(qdot[candidates]))])
        return speed_joint, stale_joint, fresh

    def _append_impedance_trace(
        self,
        loop_dt: float,
        q: np.ndarray,
        dq: np.ndarray,
        tau_imp: np.ndarray,
        grav_torque: np.ndarray,
        total_torque: np.ndarray,
        fresh_mask: np.ndarray,
    ) -> None:
        with self._get_impedance_safety_lock():
            self._append_impedance_trace_unlocked(
                loop_dt,
                q,
                dq,
                tau_imp,
                grav_torque,
                total_torque,
                fresh_mask,
            )

    def _append_impedance_trace_unlocked(
        self,
        loop_dt: float,
        q: np.ndarray,
        dq: np.ndarray,
        tau_imp: np.ndarray,
        grav_torque: np.ndarray,
        total_torque: np.ndarray,
        fresh_mask: np.ndarray,
    ) -> None:
        """Append one in-memory diagnostic sample; never performs file I/O."""
        timestamps = getattr(self, "_real_vel_timestamps", None)
        self._impedance_trace.append({
            "mono_time": time.monotonic(),
            "loop_dt": float(loop_dt),
            "q": np.asarray(q, dtype=np.float64).copy(),
            "qdot_real_raw": np.asarray(
                getattr(self, "_real_vel_raw", np.zeros(self.N_JOINTS)),
                dtype=np.float64,
            ).copy(),
            "qdot_selected": np.asarray(dq, dtype=np.float64).copy(),
            "qdot_fdiff": np.asarray(
                getattr(self, "_last_fdiff_velocity", np.zeros(self.N_JOINTS)),
                dtype=np.float64,
            ).copy(),
            "feedback_timestamps": (
                None if timestamps is None
                else np.asarray(timestamps, dtype=np.float64).copy()
            ),
            "fresh_mask": np.asarray(fresh_mask, dtype=bool).copy(),
            "tau_imp": np.asarray(tau_imp, dtype=np.float64).copy(),
            "tau_gravity": np.asarray(grav_torque, dtype=np.float64).copy(),
            "tau_total": np.asarray(total_torque, dtype=np.float64).copy(),
            "over_count": self._qdot_over_count.copy(),
            "source": self._last_velocity_source,
        })

    def _freeze_impedance_trip(
        self,
        reason: str,
        trigger_joint: int,
        trigger_kind: str,
    ) -> None:
        with self._get_impedance_safety_lock():
            self._freeze_impedance_trip_unlocked(
                reason, trigger_joint, trigger_kind)

    def _freeze_impedance_trip_unlocked(
        self,
        reason: str,
        trigger_joint: int,
        trigger_kind: str,
    ) -> None:
        """Freeze the pre-trip trace so the slow outer monitor can inspect it."""
        trace = copy.deepcopy(list(self._impedance_trace))
        trip = {
            "reason": str(reason),
            "trigger_joint": int(trigger_joint) + 1,
            "trigger_kind": str(trigger_kind),
            "trace": trace,
        }
        with self._state_lock:
            self._last_impedance_trip = trip

    def _read_joint_torque(self) -> np.ndarray:
        """Read parser torque directly; use the mutating SDK getter as fallback."""
        torque = np.zeros(self.N_JOINTS)
        timestamps = [None] * self.N_JOINTS
        valid = np.zeros(self.N_JOINTS, dtype=bool)
        parser = getattr(self._robot, "_parser", None)
        for i in range(1, self.N_JOINTS + 1):
            value = None
            if parser is not None:
                ms = getattr(parser, f"motor_state_{i}", None)
                msg = getattr(ms, "msg", None)
                raw = getattr(msg, "torque", None)
                if raw is not None:
                    candidate = float(raw)
                    if np.isfinite(candidate):
                        value = candidate
            if value is None:
                ms = self._robot.get_motor_states(i)
                msg = getattr(ms, "msg", None)
                raw = getattr(msg, "torque", None)
                if raw is not None:
                    candidate = float(raw)
                    if np.isfinite(candidate):
                        value = candidate
            if value is not None:
                torque[i - 1] = value
                valid[i - 1] = True
                try:
                    stamp = float(getattr(ms, "timestamp", None))
                    timestamps[i - 1] = stamp if np.isfinite(stamp) else None
                except (TypeError, ValueError):
                    pass
        self._last_feedback_torque_timestamps = timestamps
        self._last_feedback_torque_valid = valid
        return torque

    def _check_joint_limits(self, q: np.ndarray) -> bool:
        """Check if joint positions are within limits (with buffer).

        Returns True if safe, False if violated.
        """
        q = np.asarray(q, dtype=np.float64).ravel()
        if q.shape != (self.N_JOINTS,) or not np.all(np.isfinite(q)):
            logger.error(
                "Invalid joint vector in limit check: shape=%s values=%s",
                q.shape,
                q.tolist(),
            )
            return False
        buffer = float(getattr(self, "_joint_limit_buffer", JOINT_LIMIT_BUFFER))
        lower = NERO_JOINT_LIMITS[:, 0] + buffer
        upper = NERO_JOINT_LIMITS[:, 1] - buffer
        if np.any(q < lower) or np.any(q > upper):
            violations = []
            for i in range(self.N_JOINTS):
                if q[i] < lower[i] or q[i] > upper[i]:
                    violations.append(
                        f"joint{i+1}={q[i]:.4f} "
                        f"effective=[{lower[i]:.4f}, {upper[i]:.4f}] "
                        f"physical=[{NERO_JOINT_LIMITS[i,0]:.4f}, "
                        f"{NERO_JOINT_LIMITS[i,1]:.4f}] "
                        f"buffer={buffer:.4f}rad"
                    )
            logger.warning(f"Software joint buffer violation: {', '.join(violations)}")
            return False
        return True

    def _check_cycle_joint_limits(self, q):
        """R2 at the software buffer; physical/feedback failures still trip hard."""
        q = np.asarray(q, dtype=float)
        if (q.shape != (7,) or not np.isfinite(q).all()
                or np.any(q < NERO_JOINT_LIMITS[:, 0])
                or np.any(q > NERO_JOINT_LIMITS[:, 1])):
            raise RuntimeError("Invalid feedback or physical joint limit exceeded")
        if not self._command_stop_latched and not self._check_joint_limits(q):
            kp, kd = self._controlled_stop_kp, self._controlled_stop_kd
            if kp is None or kd is None:
                raise RuntimeError("Software joint buffer reached without a configured position hold")
            # Same fixed-hold primitive used by normal R2/Ctrl+C stopping.
            # Do not stop the sender or clear gravity compensation.
            self.hold_for_controlled_stop(JointCommands(
                pos=q.copy(), vel=np.zeros(7), kp=kp.copy(), kd=kd.copy(),
                torques=np.zeros(7),
            ))
            self._controlled_stop_reason = "R2: software joint-limit buffer reached"
            self._buffer_stop_started = time.monotonic()
        if self._buffer_stop_started is None:
            return
        status = self.get_controlled_stop_status()
        if np.max(np.abs(q - status["target_q"])) > np.deg2rad(1.0):
            raise RuntimeError("Software-buffer stop position hold drifted during transition")
        if not status["transition_complete"]:
            if time.monotonic() - self._buffer_stop_started > status["transition_duration_s"] + 0.5:
                raise RuntimeError("Software-buffer stop transition was not confirmed: final batch not acknowledged")
            return
        sample = self._feedback_health_sample
        if sample is None:
            raise RuntimeError("Software-buffer stop requires trusted per-axis feedback")
        if self._buffer_stop_settling is None:
            self._buffer_stop_settling = MITSettlingCheck(status["target_q"])
            self._buffer_stop_settle_started = time.monotonic()
        if self._buffer_stop_settling.update(sample):
            self._buffer_stop_confirmed = True
        self._buffer_stop_stable = self._buffer_stop_settling.stable_samples
        if not self._buffer_stop_confirmed and time.monotonic() - self._buffer_stop_settle_started > 2.0:
            raise RuntimeError("Software-buffer controlled stop was not confirmed within 2s")
        with self._command_lock:
            target = self._commands.pos.copy()
        if self._buffer_stop_confirmed and np.max(np.abs(q - target)) > np.deg2rad(1.0):
            raise RuntimeError("Confirmed software-buffer position hold was lost")

    def _hard_stop_locked(self, reason: str) -> bool:
        """Latch a terminal fault and attempt e-stop while actuation is locked."""
        if getattr(self, "_electronic_stop_sent", False):
            return True
        self._last_error = str(reason)
        self._terminal_fault = True
        self._running = False
        fault_wall_time, fault_monotonic_time = time.time(), time.monotonic()
        try:
            timing = self._cycle_timing()
        except Exception:
            timing = None
        with self._impedance_lock:
            if self._impedance_cfg is not None and self._impedance_cfg.active:
                self._impedance_cfg.active = False
                self._impedance_generation += 1
        try:
            self._robot.electronic_emergency_stop()
            self._electronic_stop_sent = True
            return True
        except Exception as exc:
            self._last_error += (
                f"; electronic emergency stop call failed: "
                f"{type(exc).__name__}: {exc}")
            logger.critical(self._last_error)
            return False
        finally:
            # Stop is requested first. Diagnostics are best-effort and all
            # serialization/file I/O stays in the existing writer thread.
            try:
                self._record_fault_diagnostics(
                    reason, fault_wall_time, fault_monotonic_time, timing)
            except Exception as exc:
                logger.error("MIT fault diagnostics unavailable: %s", exc)

    def _stationary_hold_command(self):
        """Only an established fixed position hold may use the recovery grace."""
        if self._diagnostics_phase != "hold" or self._command_stop_latched:
            return None
        with self._impedance_lock:
            if self._impedance_cfg is not None and self._impedance_cfg.active:
                return None
        with self._command_lock:
            command = copy.deepcopy(self._commands)
        if any(not np.isfinite(getattr(command, name)).all()
               for name in ("pos", "vel", "kp", "kd", "torques")):
            return None
        if (np.any(command.vel != 0.0) or np.any(command.torques != 0.0)
                or np.any(command.kp[:self._mit_joint_count] <= 0.0)):
            return None
        return command

    def _try_recover_stationary_hold(self, error):
        """Wait for fresh CAN frames, then recompute the unchanged hold.

        This is not trajectory/impedance recovery. No stale MIT batch is sent,
        no timestamp is refreshed synthetically, and no new hold target is made.
        The budget includes the interruption already observed, not just polling.
        """
        if not isinstance(error, (ControlDeadlineMissed, FeedbackStale)):
            return False
        budget = self._hold_recovery_timeout_s
        last_good = self._last_healthy_hold_time
        state = self.get_joint_state()
        if budget <= 0.0 or last_good is None or state is None:
            return False
        deadline = last_good + budget
        with self._get_actuation_lock():
            command = self._stationary_hold_command()
            if command is None or self._terminal_fault or not self._running:
                return False
            reference = np.asarray(state.pos, dtype=float).copy()
            # Do not apply a grace period to a moving/poorly settled hold.
            if (not np.isfinite(reference).all() or not np.isfinite(state.vel).all()
                    or np.max(np.abs(state.vel)) > np.deg2rad(1.0)
                    or np.max(np.abs(command.pos - reference)) > np.deg2rad(1.0)):
                return False
            self._hold_recovering = True
        previous = None
        stable = 0
        try:
            while time.perf_counter() < deadline:
                if not self._running or self._terminal_fault:
                    return False
                try:
                    sample = self._feedback_health.read(
                        self._robot, time.time(), time.monotonic())
                except FeedbackStale:
                    time.sleep(self._period)
                    continue
                q = sample["q"]
                if not self._check_joint_limits(q):
                    raise RuntimeError("MIT hold recovery: joint limit exceeded")
                if np.max(np.abs(q - reference)) > np.deg2rad(0.5):
                    raise RuntimeError("MIT hold recovery: measured drift exceeds 0.5deg")
                if previous is not None and all(
                    np.all(sample[key] > previous[key])
                    for key in ("position_timestamps", "motor_timestamps")
                ):
                    dt = float(np.min(sample["position_timestamps"])
                               - np.min(previous["position_timestamps"]))
                    speed = float(np.max(np.abs(q - previous["q"]))) / dt
                    stable = stable + 1 if speed <= np.deg2rad(1.0) else 0
                    previous = sample
                elif previous is None:
                    previous = sample
                if stable >= 3:
                    with self._get_actuation_lock():
                        current = self._stationary_hold_command()
                        if (not self._running or self._terminal_fault or current is None
                                or time.perf_counter() >= deadline
                                or any(not np.array_equal(getattr(current, name), getattr(command, name))
                                       for name in ("pos", "vel", "kp", "kd", "torques"))):
                            return False
                    # Next iteration re-reads q, recomputes gravity and validates
                    # age again. The failed cycle's torque is never reused.
                    self._last_control_loop_start = None
                    self._last_healthy_hold_time = None
                    writer = self._diagnostics_writer
                    if writer is not None and writer.accepting:
                        try:
                            writer.submit({
                                "type": "hold_recovery", "wall_time": time.time(),
                                "monotonic_time": time.monotonic(), "reason": str(error),
                                "interruption_s": time.perf_counter() - last_good,
                                "budget_s": budget, "fresh_samples": stable,
                                "target_unchanged": True,
                            })
                        except Exception:
                            # Optional logging cannot turn a recovered hold into
                            # an emergency-stop condition.
                            pass
                    return True
                time.sleep(self._period)
            raise RuntimeError(
                f"MIT fixed-hold recovery failed within {budget:.3f}s; "
                "fresh, stable feedback was not confirmed")
        finally:
            self._hold_recovering = False

    def _send_mit_commands(
        self, pos, vel, kp, kd, torques
    ) -> None:
        """Send MIT commands for configured MIT-capable joints.

        Uses the official SDK's ``move_mit()`` per joint (compatible with
        both official and fork pyAgxArm).  The MIT mode is set once at
        the start of the loop.
        """
        arrays = {
            "pos": np.asarray(pos, dtype=np.float64).ravel(),
            "vel": np.asarray(vel, dtype=np.float64).ravel(),
            "kp": np.asarray(kp, dtype=np.float64).ravel(),
            "kd": np.asarray(kd, dtype=np.float64).ravel(),
            "torques": np.asarray(torques, dtype=np.float64).ravel(),
        }
        for name, value in arrays.items():
            if value.shape != (self.N_JOINTS,) or not np.all(np.isfinite(value)):
                raise ValueError(
                    f"Refusing invalid MIT {name}: expected finite shape "
                    f"({self.N_JOINTS},), got shape={value.shape}, "
                    f"values={value.tolist()}")
        if np.any(arrays["kp"] < 0.0) or np.any(arrays["kp"] > 500.0):
            raise ValueError("Refusing MIT kp outside [0, 500]")
        if np.any(arrays["kd"] < 0.0) or np.any(arrays["kd"] > 5.0):
            raise ValueError("Refusing MIT kd outside [0, 5]")
        pos = arrays["pos"]
        vel = arrays["vel"]
        kp = arrays["kp"]
        kd = arrays["kd"]
        torques = arrays["torques"]

        if not self._mit_mode_set:
            self._robot.set_motion_mode('mit')
            self._mit_mode_set = True

        for i in range(self._mit_joint_count):
            if getattr(self, "_feedback_guard_enabled", False):
                self._feedback_health.check_cycle(
                    time.perf_counter() - self._last_control_loop_start)
                self._feedback_health.validate_age(
                    self._feedback_health_sample, time.time(), time.monotonic())
            self._robot.move_mit(
                joint_index=i + 1,
                p_des=float(pos[i]),
                v_des=float(vel[i]),
                kp=float(kp[i]),
                kd=float(kd[i]),
                t_ff=float(torques[i]),
            )
            self._cycle_sent_joints = i + 1

    def _control_loop(self) -> None:
        """Main control loop (runs in background thread)."""
        miss_count = 0
        prev_q = None
        step_count = 0
        exceed_count = 0
        report_start = time.time()

        while self._running:
            loop_start = time.perf_counter()
            if self._last_control_loop_start is None:
                loop_dt = self._period
            else:
                loop_dt = loop_start - self._last_control_loop_start
            self._last_control_loop_start = loop_start
            self._cycle_stage = "cycle_deadline"
            self._cycle_stage_start = loop_start
            self._cycle_stage_durations = {}
            self._cycle_loop_dt = float(loop_dt)
            self._cycle_feedback_read_started = False
            self._cycle_sent_joints = 0
            actuation_owned = False
            stop_batch_complete = False

            try:
                if getattr(self, "_feedback_guard_enabled", False):
                    self._feedback_health.check_cycle(loop_dt)
                # 1. Read latest joint state
                self._begin_cycle_stage("feedback_read")
                self._cycle_feedback_read_started = True
                q = self._read_joint_pos()

                if q is None:
                    miss_count += 1
                    if miss_count > self._miss_limit:
                        reason = (
                            f"Lost joint feedback for {miss_count} "
                            f"consecutive cycles, emergency stop"
                        )
                        logger.error(reason)
                        with self._get_actuation_lock():
                            self._hard_stop_locked(reason)
                        break
                    # Use previous state and skip control
                    self._sleep_until(loop_start + self._period)
                    continue

                miss_count = 0
                self._begin_cycle_stage("state_safety")

                # 2. Safety: position jump detection
                if prev_q is not None:
                    dq = np.abs(q - prev_q)
                    if np.any(dq > MAX_POS_JUMP):
                        logger.warning(
                            f"Position jump detected: max_dq="
                            f"{np.max(dq):.4f} rad"
                        )
                prev_q = q.copy()

                # 3. Safety: joint limit check
                self._check_cycle_joint_limits(q)

                # 4. Compute gravity compensation
                self._begin_cycle_stage("gravity")
                if self._gravity_enabled:
                    grav_torque = self._grav.compute(q) * self._gravity_factors
                else:
                    grav_torque = np.zeros(self.N_JOINTS)

                # 5. Resolve command. Two mutually exclusive paths:
                #    (a) in-loop Cartesian impedance (author-validated Panthera law,
                #        computed fresh each cycle) — active only when an impedance
                #        config with active=True is installed AND a pin_model exists;
                #    (b) the buffered command path (zero-gravity / mit-position
                #        visual servo / legacy outer-loop commands).
                self._begin_cycle_stage("actuation_lock_wait")
                actuation_lock = self._get_actuation_lock()
                actuation_lock.acquire()
                actuation_owned = True
                self._begin_cycle_stage("command_compute")
                if self._terminal_fault or not self._running:
                    actuation_lock.release()
                    actuation_owned = False
                    break
                with self._impedance_lock:
                    imp_cfg = copy.deepcopy(self._impedance_cfg)
                if imp_cfg is not None and imp_cfg.active and self._pin is not None:
                    # Velocity source for the Cartesian damper -B*J*qdot. Prefer
                    # the REAL motor velocity (parser-direct, sign-corrected): it
                    # is clean (firmware-filtered), so a PROPER B works — the
                    # thing finite-difference could never provide. If parser
                    # velocity is unavailable, Cartesian B is zeroed and
                    # firmware KD remains active; finite difference stays hot
                    # only as an independent runaway detector.
                    dq = self._read_impedance_joint_vel(q, imp_cfg)
                    tau_imp, kp_cmd, kd_cmd, cmd_pos = self._compute_impedance_torque(
                        q, dq, imp_cfg)
                    requested_torque = tau_imp
                    total_torque = tau_imp + grav_torque
                    # Per-joint impedance clip, then driver hard limit (+-16), then torque_limit.
                    total_torque = np.clip(
                        total_torque, -imp_cfg.tau_limit_per_joint, imp_cfg.tau_limit_per_joint)
                    total_torque = np.clip(total_torque, -16.0, 16.0)
                    total_torque = np.clip(total_torque, -self._torque_limit, self._torque_limit)
                    control_path = "impedance"
                    vel_cmd = np.zeros(self.N_JOINTS)
                    vel = dq  # reuse for state publish (already filtered)
                    # Fast in-loop divergence trip.  Parser velocity is checked
                    # per joint and only NEW CAN timestamps advance the five-
                    # frame counter.  A frozen timestamp has its own watchdog.
                    if self._last_velocity_source in (
                        "real", "real-unavailable"
                    ):
                        if self._last_velocity_source == "real-unavailable":
                            safety_dq = np.asarray(
                                self._last_fdiff_velocity, dtype=np.float64)
                            velocity_timestamps = (
                                None
                                if self._joint_pos_timestamp is None
                                else np.full(
                                    self.N_JOINTS,
                                    self._joint_pos_timestamp,
                                )
                            )
                        else:
                            safety_dq = dq
                            velocity_timestamps = self._real_vel_timestamps
                    elif self._joint_pos_timestamp is not None:
                        safety_dq = dq
                        velocity_timestamps = np.full(
                            self.N_JOINTS, self._joint_pos_timestamp)
                    else:
                        safety_dq = dq
                        velocity_timestamps = None

                    trip_joint = None
                    trip_kind = ""
                    trip_error = ""
                    with self._get_impedance_safety_lock():
                        speed_joint, stale_joint, fresh_mask = (
                            self._update_qdot_trip_unlocked(
                                safety_dq,
                                imp_cfg.qdot_trip_fast,
                                timestamps=velocity_timestamps,
                            )
                        )
                        self._append_impedance_trace_unlocked(
                            loop_dt=loop_dt,
                            q=q,
                            dq=dq,
                            tau_imp=tau_imp,
                            grav_torque=grav_torque,
                            total_torque=total_torque,
                            fresh_mask=fresh_mask,
                        )

                        unavailable_since = self._real_velocity_unavailable_since
                        source_unavailable = (
                            imp_cfg.use_real_velocity
                            and self._last_velocity_source == "real-unavailable"
                            and unavailable_since is not None
                            and time.monotonic() - unavailable_since
                            > REAL_VELOCITY_STALE_TIMEOUT
                        )
                        if source_unavailable:
                            trip_joint = 0
                            trip_kind = "real-velocity-unavailable"
                            trip_error = (
                                "in-loop impedance real velocity unavailable "
                                f">{REAL_VELOCITY_STALE_TIMEOUT:.3f}s; refusing "
                                "finite-difference fallback with active Cartesian "
                                "B, emergency stop")
                        elif speed_joint is not None:
                            trip_joint = speed_joint
                            trip_kind = "qdot"
                            raw = np.asarray(
                                self._real_vel_raw, dtype=np.float64)
                            fdiff = np.asarray(
                                self._last_fdiff_velocity, dtype=np.float64)
                            trip_error = (
                                f"in-loop impedance qdot joint{speed_joint + 1} "
                                f"safety={safety_dq[speed_joint]:.3f} rad/s "
                                f"damping={dq[speed_joint]:.3f} rad/s "
                                f"raw={raw[speed_joint]:.3f} "
                                f"fdiff={fdiff[speed_joint]:.3f} "
                                f"source={self._last_velocity_source} sustained "
                                f"{int(self._qdot_over_count[speed_joint])} fresh "
                                f"frames >{imp_cfg.qdot_trip_fast:.2f}, "
                                "emergency stop")
                        elif stale_joint is not None:
                            trip_joint = stale_joint
                            trip_kind = "stale-velocity"
                            trip_error = (
                                "in-loop impedance velocity feedback stale "
                                f"joint{stale_joint + 1} >"
                                f"{REAL_VELOCITY_STALE_TIMEOUT:.3f}s, "
                                "emergency stop")

                        if trip_joint is not None:
                            self._last_error = trip_error
                            self._freeze_impedance_trip_unlocked(
                                trip_error, trip_joint, trip_kind)

                    if trip_joint is not None:
                        logger.error(self._last_error)
                        self._hard_stop_locked(self._last_error)
                        actuation_lock.release()
                        actuation_owned = False
                        break
                else:
                    with self._command_lock:
                        cmds = copy.deepcopy(self._commands)
                    if self._command_stop_latched:
                        cmds, stop_batch_complete = self._stop_transition_command(
                            cmds, time.monotonic())
                    requested_torque = cmds.torques
                    total_torque = cmds.torques + grav_torque
                    total_torque = np.clip(
                        total_torque, -self._torque_limit, self._torque_limit)
                    tau_imp = np.zeros(self.N_JOINTS)
                    control_path = ("stop_transition" if self._command_stop_latched
                                    and not stop_batch_complete else "buffered")
                    cmd_pos = np.clip(
                        cmds.pos, NERO_JOINT_LIMITS[:, 0], NERO_JOINT_LIMITS[:, 1])
                    kp_cmd = cmds.kp
                    kd_cmd = cmds.kd
                    vel_cmd = cmds.vel
                    vel = self._read_joint_vel(q)

                # 9. Send MIT commands for all joints
                self._begin_cycle_stage("send_preflight")
                if getattr(self, "_feedback_guard_enabled", False):
                    self._feedback_health.check_cycle(time.perf_counter() - loop_start)
                    self._feedback_health.validate_age(
                        self._feedback_health_sample, time.time(), time.monotonic())
                self._begin_cycle_stage("send_mit")
                self._send_mit_commands(
                    pos=cmd_pos,
                    vel=vel_cmd,
                    kp=kp_cmd,
                    kd=kd_cmd,
                    torques=total_torque,
                )
                # Publish only completed batches, while the stop latch still
                # shares this actuation lock. No diagnostic I/O occurs here.
                self._last_sent_snapshot = {
                    "pos": cmd_pos.copy(), "vel": vel_cmd.copy(),
                    "kp": kp_cmd.copy(), "kd": kd_cmd.copy(),
                    "q_measured": q.copy(), "dq_measured": vel.copy(),
                    "non_gravity_torque": (total_torque - grav_torque).copy(),
                    "tau_ff": total_torque.copy(),
                    "wall_time": time.time(), "monotonic_time": time.monotonic(),
                    "control_path": control_path,
                }
                if self._command_stop_latched and stop_batch_complete:
                    if not self._controlled_stop_complete:
                        self._controlled_stop_completed_monotonic = time.monotonic()
                    self._controlled_stop_complete = True
                actuation_lock.release()
                actuation_owned = False

                # 10. Update joint state for external readers
                self._begin_cycle_stage("state_publish")
                torque = self._read_joint_torque()
                with self._state_lock:
                    self._last_control_path = control_path
                    if control_path == "impedance":
                        self._impedance_cycle_count += 1
                    self._last_impedance_torque = tau_imp.copy()
                    self._last_gravity_torque = grav_torque.copy()
                    self._last_total_torque = total_torque.copy()
                    self._last_feedback_torque = torque.copy()
                    feedback_timestamp = getattr(
                        self, "_joint_pos_timestamp", None
                    )
                    self._joint_state = JointState(
                        pos=q,
                        vel=vel,
                        torque=torque,
                        timestamp=(
                            0.0
                            if feedback_timestamp is None
                            else float(feedback_timestamp)
                        ),
                    )
                if self._stationary_hold_command() is not None:
                    self._last_healthy_hold_time = time.perf_counter()
                else:
                    self._last_healthy_hold_time = None
                try:
                    self._begin_cycle_stage("diagnostics")
                    self._record_control_cycle(
                        loop_dt, q, vel, cmd_pos, vel_cmd, kp_cmd, kd_cmd,
                        requested_torque, grav_torque, total_torque, torque, control_path,
                    )
                except Exception as exc:
                    # Optional diagnostics must never become a control-loop
                    # exception/e-stop. The slow owner observes this status.
                    writer = getattr(self, "_diagnostics_writer", None)
                    if writer is not None:
                        writer.error = f"Diagnostic capture failed: {type(exc).__name__}: {exc}"
                        writer.accepting = False
                        writer.stop_requested.set()

            except Exception as e:
                if actuation_owned:
                    actuation_lock.release()
                    actuation_owned = False
                try:
                    if self._try_recover_stationary_hold(e):
                        continue
                except Exception as recovery_error:
                    e = recovery_error
                if self._terminal_fault or not self._running:
                    break
                reason = f"Control loop error: {type(e).__name__}: {e}"
                if not actuation_owned:
                    actuation_lock = self._get_actuation_lock()
                    actuation_lock.acquire()
                    actuation_owned = True
                self._hard_stop_locked(reason)
                actuation_lock.release()
                actuation_owned = False
                logger.error(reason)
                break

            # 11. Rate limiting
            elapsed = time.perf_counter() - loop_start
            step_count += 1
            if elapsed > self._period * 1.5:
                exceed_count += 1

            # Periodic reporting
            now = time.time()
            if now - report_start >= REPORT_INTERVAL:
                freq = step_count / (now - report_start) if step_count > 0 else 0
                logger.info(
                    f"MIT loop: {freq:.1f} Hz, "
                    f"exceeded={exceed_count}/{step_count}"
                )
                step_count = 0
                exceed_count = 0
                report_start = now

            self._sleep_until(loop_start + self._period)

    def _sleep_until(self, target_time: float) -> None:
        """Sleep until target_time using a combination of busy-wait and sleep."""
        remaining = target_time - time.perf_counter()
        if remaining > 0.002:
            time.sleep(remaining - 0.001)
        # Busy-wait for the last ~1ms for precision
        while time.perf_counter() < target_time:
            pass
