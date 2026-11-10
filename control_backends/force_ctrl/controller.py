import time
import logging
import inspect
import numpy as np

from pyAgxArm import create_agx_arm_config, AgxArmFactory
from .gravity_comp import NeroGravityCompensator
from .mit_loop import NeroMITLoop, NERO_JOINT_LIMITS, JOINT_LIMIT_BUFFER
from .data_types import JointCommands, JointState
from .feedback_health import FeedbackHealthMonitor
from .normal_shutdown import MITControlledStop, defer_sigint, handoff_to_position_hold, report_shutdown

logger = logging.getLogger(__name__)

# Default PD gains (conservative starting values, tune on real hardware)
DEFAULT_KP = np.array([30.0, 30.0, 30.0, 15.0, 10.0, 5.0, 5.0])
DEFAULT_KD = np.array([2.0, 2.0, 2.0, 1.0, 0.5, 0.3, 0.3])

# Damping for zero-gravity mode (prevents free oscillation)
ZERO_GRAV_KD = np.array([1.0, 1.0, 1.0, 0.5, 0.3, 0.2, 0.2])

N_JOINTS = 7


def _infer_mit_joint_count(robot) -> int:
    """Infer how many joints the installed SDK accepts for move_mit()."""
    try:
        param = inspect.signature(robot.move_mit).parameters.get("joint_index")
        args = getattr(param.annotation, "__args__", ()) if param else ()
        joint_ids = [value for value in args if isinstance(value, int)]
        if joint_ids:
            return max(joint_ids)
    except Exception:
        pass
    return N_JOINTS


class NeroForceController:
    """High-level force control interface for the Nero 7-DOF arm.

    Wraps the pyAgxArm Nero driver with a real-time MIT control loop
    and MuJoCo-based gravity compensation.

    Parameters
    ----------
    channel : str
        CAN channel name (e.g. "can0").
    urdf_path : str, optional
        Path to Nero URDF for gravity compensation.
        Defaults to the bundled model.
    zero_gravity : bool, optional
        If True, start in zero-gravity (free-drag) mode.
        If False, start in position-hold mode at current pose.
        Default True.
    gravity_factor : float, optional
        Gravity compensation multiplier. Default 1.0.
    control_freq : int, optional
        MIT control loop frequency in Hz. Default 200.
    torque_limit : float, optional
        Maximum absolute torque per joint (N·m). Default 8.0.
    interface : str, optional
        CAN interface type. Default "socketcan".
    firmware : str, optional
        Nero firmware driver variant. Use "v111" for firmware >= 1.11.
    mit_joint_count : int, optional
        Number of joints to send MIT commands to. Defaults to SDK introspection.

    Examples
    --------
    Zero-gravity drag mode::

        ctrl = NeroForceController(channel="can0", zero_gravity=True)
        ctrl.start()
        # The arm can now be moved freely by hand.
        # ...
        ctrl.stop()

    Position hold with compliance::

        ctrl = NeroForceController(channel="can0", zero_gravity=False)
        ctrl.start()
        # Arm holds its initial position.
        ctrl.command_joint_pos(target_q, kp=[10]*7, kd=[1]*7)
        # ...
        ctrl.stop()
    """

    def __init__(
        self,
        channel: str = "can0",
        urdf_path: str = "",
        zero_gravity: bool = True,
        gravity_factor: float = 1.0,
        gravity_factors=None,
        control_freq: int = 200,
        torque_limit: float = 8.0,
        startup_joint_margin: float = 0.0,
        interface: str = "socketcan",
        firmware: str = "default",
        mit_joint_count: int = None,
        pin_model=None,
        skip_reset: bool = False,
        joint_limit_buffer: float = JOINT_LIMIT_BUFFER,
    ):
        self._channel = channel
        self._zero_gravity_init = zero_gravity
        self._gravity_factor = gravity_factor
        self._gravity_factors = gravity_factors
        self._control_freq = control_freq
        self._torque_limit = torque_limit
        self._startup_joint_margin = float(startup_joint_margin)
        self._skip_reset = bool(skip_reset)
        self._joint_limit_buffer = float(joint_limit_buffer)
        if (
            not np.isfinite(self._startup_joint_margin)
            or self._startup_joint_margin < 0.0
        ):
            raise ValueError("startup_joint_margin must be finite and non-negative")
        self._firmware = firmware

        # Create pyAgxArm Nero driver
        cfg = create_agx_arm_config(
            robot="nero",
            comm="can",
            channel=channel,
            interface=interface,
            firmeware_version=firmware,
        )
        self._robot = AgxArmFactory.create_arm(cfg)
        self._mit_joint_count = (
            _infer_mit_joint_count(self._robot)
            if mit_joint_count is None else int(mit_joint_count)
        )

        # Create gravity compensator
        self._grav = NeroGravityCompensator(
            urdf_path=urdf_path,
            gravity=np.array([0.0, 0.0, -9.81]),
            max_torque_sanity=20.0,
        )

        # Create MIT control loop
        self._loop = NeroMITLoop(
            robot=self._robot,
            gravity_comp=self._grav,
            control_freq=control_freq,
            gravity_factor=gravity_factor,
            gravity_factors=gravity_factors,
            torque_limit=torque_limit,
            mit_joint_count=self._mit_joint_count,
            pin_model=pin_model,
            joint_limit_buffer=self._joint_limit_buffer,
            controlled_stop_kp=DEFAULT_KP,
            controlled_stop_kd=DEFAULT_KD,
        )

        # Store default gains
        self._kp = DEFAULT_KP.copy()
        self._kd = DEFAULT_KD.copy()

        self._started = False
        self._robot_connected = False
        self._robot_enabled = False
        self._emergency_stopped = False
        self._exit_hold_q = None
        self._startup_control_touched = False
        self._startup_cleanup_complete = False
        self._startup_cleanup_error = None
        self._device_hold_confirmed = False

    @property
    def robot(self):
        """Access the underlying pyAgxArm Nero driver."""
        return self._robot

    @property
    def mit_joint_count(self) -> int:
        """Number of joints this controller sends MIT commands to."""
        return self._mit_joint_count

    def start(self) -> None:
        """Validate read-only feedback before enabling or changing control mode."""
        if (getattr(self, "_emergency_stopped", False) is True
                or getattr(self._loop, "terminal_fault", False) is True
                or getattr(self._loop, "_command_stop_latched", False) is True
                or getattr(self, "_startup_cleanup_error", None) is not None):
            raise RuntimeError("Controller stop is latched; explicit recovery/new session is required")
        if self._started:
            logger.warning("Controller already started")
            return

        self._startup_control_touched = False
        self._startup_cleanup_complete = False
        self._startup_cleanup_error = None
        self._device_hold_confirmed = False
        try:
            # SDK connect() starts reception only; rejection here must leave
            # the device's existing control/enable state untouched.
            self._robot.connect()
            self._robot_connected = True
            logger.info(f"Connected to Nero on {self._channel}")
            sample = self._wait_startup_feedback()
            self._validate_startup_posture(sample["q"])

            # A driver may transmit and then raise, so latch before the first
            # state-changing call, not after enable() acknowledges success.
            if not self._skip_reset:
                self._startup_control_touched = True
                self._robot.reset()
                time.sleep(0.2)
            else:
                logger.info("Skipping robot.reset() for grasp-to-servo handoff")
            self._startup_control_touched = True
            self._robot.set_normal_mode()
            deadline = time.monotonic() + 5.0
            while not self._robot.enable():
                if time.monotonic() > deadline:
                    raise RuntimeError("Failed to enable Nero arm within 5 seconds")
                time.sleep(0.01)
            self._robot_enabled = True
            enabled_at = time.time()
            logger.info("Nero arm enabled")

            # Do not initialize MIT from the pose cached before enable/reset.
            sample = self._wait_startup_feedback(after_wall_time=enabled_at)
            q = np.asarray(sample["q"], dtype=np.float64).copy()
            self._validate_startup_posture(q)
            logger.info(f"Initial joint angles: {np.round(q, 4).tolist()}")

            # Set initial mode
            if self._zero_gravity_init:
                cmds = JointCommands(
                    pos=q.copy(),
                    vel=np.zeros(N_JOINTS),
                    kp=np.zeros(N_JOINTS),
                    kd=ZERO_GRAV_KD.copy(),
                    torques=np.zeros(N_JOINTS),
                )
            else:
                cmds = JointCommands(
                    pos=q.copy(),
                    vel=np.zeros(N_JOINTS),
                    kp=self._kp.copy(),
                    kd=self._kd.copy(),
                    torques=np.zeros(N_JOINTS),
                )

            self._loop.update_commands(cmds)
            self._loop.start()
            self._started = True
            mode_name = "zero-gravity" if self._zero_gravity_init else "position-hold"
            logger.info(f"Control loop started in {mode_name} mode")
        except (Exception, KeyboardInterrupt) as start_exc:
            try:
                self.cleanup_failed_start()
            except Exception as cleanup_exc:
                raise RuntimeError(
                    f"Controller start failed ({start_exc}); cleanup also "
                    f"failed ({cleanup_exc})") from start_exc
            raise

    def _wait_startup_feedback(self, *, after_wall_time=None):
        monitor = FeedbackHealthMonitor()
        deadline = time.monotonic() + 3.0
        while True:
            try:
                sample = monitor.read(self._robot, time.time(), time.monotonic())
                if after_wall_time is not None and (
                    np.any(sample["position_timestamps"] <= after_wall_time)
                    or np.any(sample["motor_timestamps"] <= after_wall_time)
                ):
                    raise RuntimeError("Waiting for new per-axis feedback after enable")
                return sample
            except RuntimeError as exc:
                if time.monotonic() >= deadline:
                    raise RuntimeError(f"Startup feedback not ready within 3 seconds: {exc}") from exc
                time.sleep(0.02)

    def _validate_startup_posture(self, q):
        q = np.asarray(q, dtype=np.float64).ravel()
        if q.shape != (N_JOINTS,) or not np.all(np.isfinite(q)):
            raise RuntimeError(
                "Invalid initial joint feedback: "
                f"expected finite shape ({N_JOINTS},), got "
                f"shape={q.shape}, values={q.tolist()}")
        clearances = np.minimum(q - NERO_JOINT_LIMITS[:, 0], NERO_JOINT_LIMITS[:, 1] - q)
        joint = int(np.argmin(clearances))
        margin = self._startup_joint_margin
        if clearances[joint] < 0.0 or (margin > 0.0 and clearances[joint] <= margin):
            lower, upper = NERO_JOINT_LIMITS[joint]
            raise RuntimeError(
                "Unsafe startup posture before MIT loop: "
                f"joint{joint + 1}={q[joint]:.4f}rad "
                f"limit=[{lower:.4f}, {upper:.4f}] "
                f"margin={clearances[joint]:.4f} required {margin:.4f}rad; "
                "startup refused; configured limits were not changed")

    @defer_sigint()
    def cleanup_failed_start(self):
        """Idempotent startup unwind; never route a failure through disable()."""
        if getattr(self, "_startup_cleanup_complete", False):
            return
        previous_error = getattr(self, "_startup_cleanup_error", None)
        if previous_error is not None:
            raise RuntimeError(f"Startup cleanup remains unconfirmed: {previous_error}") from previous_error
        if not getattr(self, "_startup_control_touched", False):
            # This also releases a partially initialized CAN receiver when
            # connect() itself failed. No control frame has been sent by us.
            self._robot.disconnect()
            self._robot_connected = False
            self._startup_cleanup_complete = True
            report_shutdown("[servo][startup] rejected before control takeover; "
                            "existing device state left unchanged; no enable/disable/e-stop sent")
            return
        fault_stop_attempted = False
        try:
            if self._exit_has_fault() and getattr(self._loop, "_electronic_stop_sent", False) is not True:
                # A terminal flag alone is not proof that its CAN stop frame
                # reached the driver. Do not disconnect after a failed send.
                fault_stop_attempted = True
                self.emergency_stop()
            if self._loop.running:
                self._started = True  # start() may raise after launching its worker.
            self.stop_keep_enabled()
        except Exception as cleanup_exc:
            self._startup_cleanup_error = cleanup_exc
            if getattr(self, "_device_hold_confirmed", False):
                # A communication/diagnostic teardown error after a verified
                # device hold is not a reason to remove that support.
                raise
            if fault_stop_attempted or getattr(self, "_emergency_stopped", False):
                raise
            # Once control was touched, untrusted feedback/unconfirmed hold is
            # still an R3 fault. Do not claim a safe hold or fall back to disable.
            try:
                self.emergency_stop()
            except Exception as protection_exc:
                raise RuntimeError(
                    f"Startup hold unconfirmed ({cleanup_exc}); "
                    f"fault stop also failed ({protection_exc})") from cleanup_exc
            raise
        self._startup_cleanup_complete = True

    def stop(self) -> None:
        """Idempotently stop, disable, and disconnect; report any failure."""
        errors = []
        loop_stopped = not self._started
        try:
            if self._started or self._loop.running:
                self._loop.stop()
                time.sleep(0.1)
            loop_stopped = True
        except Exception as exc:
            loop_stopped = False
            errors.append(f"MIT loop stop failed: {exc}")
            try:
                self._robot.electronic_emergency_stop()
            except Exception as estop_exc:
                errors.append(
                    f"electronic emergency stop after loop failure failed: "
                    f"{estop_exc}")

        # A worker that failed to join may still be inside move_mit().  Do not
        # race that thread with disable()/disconnect(): keep the driver and
        # controller state intact so a later stop() can retry after the worker
        # actually exits.  The hardware e-stop above is the immediate safety
        # action.
        if not loop_stopped:
            errors.append(
                "driver connection retained because MIT loop termination "
                "was not confirmed")
            message = "; ".join(errors)
            logger.critical(
                "Nero force controller cleanup incomplete: %s", message)
            raise RuntimeError(message)

        disabled = not self._robot_enabled
        try:
            if self._robot_connected and self._robot_enabled:
                timeout = time.time() + 2.0
                while True:
                    if self._robot.disable():
                        disabled = True
                        break
                    if time.time() > timeout:
                        errors.append(
                            "motor disable was not confirmed within 2.0s")
                        break
                    time.sleep(0.01)
        except Exception as exc:
            errors.append(f"motor disable failed: {exc}")

        if self._robot_connected and not disabled:
            try:
                self._robot.electronic_emergency_stop()
            except Exception as exc:
                errors.append(
                    f"electronic emergency stop after unconfirmed disable "
                    f"failed: {exc}")

        disconnected = not self._robot_connected
        try:
            if self._robot_connected:
                self._robot.disconnect()
                disconnected = True
        except Exception as exc:
            errors.append(f"driver disconnect failed: {exc}")

        if loop_stopped:
            self._started = False
        if disabled:
            self._robot_enabled = False
        if disconnected:
            self._robot_connected = False
        if errors:
            message = "; ".join(errors)
            logger.critical("Nero force controller cleanup incomplete: %s", message)
            raise RuntimeError(message)
        logger.info("Nero force controller stopped and disconnected")

    def _exit_has_fault(self):
        return bool(
            getattr(self, "_emergency_stopped", False)
            or getattr(self._loop, "terminal_fault", False)
            or self._loop.get_debug_state().get("last_error")
        )

    def prepare_shutdown(self) -> None:
        """Latch a fresh, fixed joint hold before camera/detector teardown."""
        if not self._robot_connected or self._exit_has_fault():
            return
        if getattr(self, "_exit_hold_q", None) is not None:
            return
        if not self._started or not self._loop.running:
            raise RuntimeError("Cannot prepare enabled exit without a running MIT loop")
        def check_fault():
            if self._exit_has_fault():
                raise RuntimeError("MIT fault during enabled-exit hold")
        self._exit_stop = MITControlledStop(
            self._robot, self._loop, self._kp, self._kd, check_fault=check_fault)
        self._exit_hold_q = self._exit_stop.request()

    @defer_sigint()
    def stop_keep_enabled(self) -> None:
        """Normal servo exit: fixed MIT hold -> device joint hold, never disable.

        A failed handoff is reported, not retried through stop()/disable().
        An existing emergency/terminal fault is never cleared or re-enabled.
        """
        if not self._robot_connected:
            return
        if self._exit_has_fault():
            if getattr(self._loop, "_electronic_stop_sent", False) is not True:
                self.emergency_stop()
            else:
                self._loop.stop()
            self._started = False
            self._robot.disconnect()
            self._robot_connected = False
            logger.warning("Servo disconnected after a fault; no enable/reset/hold command sent")
            return
        if self._loop.running:
            self.prepare_shutdown()
            self._exit_stop.wait()
        elif self._started:
            raise RuntimeError("MIT sender stopped without a verified exit hold")
        elif not getattr(self, "_startup_control_touched", False):
            # A failed read-only startup must not create a new hold target.
            self._robot.disconnect()
            self._robot_connected = False
            return
        def check_fault():
            if self._exit_has_fault():
                raise RuntimeError("MIT fault forbids normal-mode exit handoff")
        stop_sender = getattr(self._loop, "stop_sender", self._loop.stop)
        record = getattr(self._loop, "record_control_event", None)
        if callable(record):
            record("device_handoff_begin")
        result = handoff_to_position_hold(
            self._robot, stop_sender, check_fault=check_fault, record_event=record)
        if not isinstance(result, dict) or result.get("confirmed") is not True:
            raise RuntimeError("Device enabled position hold was not confirmed")
        record = getattr(self._loop, "record_control_event", None)
        if callable(record):
            record("device_hold_confirmed", result=result)
        self._device_hold_confirmed = True
        self._robot_enabled = True
        self._started = False
        if callable(record):
            record("device_disconnect_begin")
        self._robot.disconnect()
        self._robot_connected = False
        if callable(record):
            record("device_disconnected")
        try:
            close = getattr(self._loop, "stop_diagnostics", None)
            if callable(close):
                close()
        except Exception as exc:
            logger.warning("Servo hold confirmed; diagnostic cleanup failed: %s", exc)
        logger.warning("Servo exit: device joint hold confirmed, motors left enabled; no disable/e-stop sent")

    # ========================= Control Modes =========================

    def zero_gravity_mode(self) -> None:
        """Switch to zero-gravity (free drag) mode.

        In this mode, only gravity compensation torques are applied.
        A small damping (kd) is used to prevent free oscillation.
        The arm can be freely moved by hand.
        """
        q = self.get_joint_pos()
        if q is None:
            raise RuntimeError("No joint feedback available")

        self._loop.set_gravity_enabled(True)
        cmds = JointCommands(
            pos=q.copy(),
            vel=np.zeros(N_JOINTS),
            kp=np.zeros(N_JOINTS),
            kd=ZERO_GRAV_KD.copy(),
            torques=np.zeros(N_JOINTS),
        )
        self._loop.update_commands(cmds)
        logger.info("Switched to zero-gravity mode")

    def set_gravity_factors(self, factors) -> None:
        self._loop.set_gravity_factors(factors)

    def zero_torque_mode(self) -> None:
        """Switch to zero-torque (fully passive) mode.

        All PD gains and torques are zero. The arm is completely passive
        with no gravity compensation. Use with caution - the arm will fall.
        """
        self._loop.set_gravity_enabled(False)
        cmds = JointCommands.zeros(N_JOINTS)
        self._loop.update_commands(cmds)
        logger.info("Switched to zero-torque mode (no gravity comp)")

    def command_joint_pos(
        self,
        pos: np.ndarray,
        kp: np.ndarray = None,
        kd: np.ndarray = None,
    ) -> None:
        """Position PD control with gravity compensation.

        Parameters
        ----------
        pos : array-like, shape (7,)
            Target joint angles in radians.
        kp : array-like, shape (7,), optional
            Proportional gains. Default uses DEFAULT_KP.
        kd : array-like, shape (7,), optional
            Derivative gains. Default uses DEFAULT_KD.
        """
        pos = np.asarray(pos, dtype=np.float64).ravel()
        if len(pos) != N_JOINTS:
            raise ValueError(f"Expected {N_JOINTS} joint angles, got {len(pos)}")

        # Clamp to joint limits
        pos = np.clip(pos, NERO_JOINT_LIMITS[:, 0], NERO_JOINT_LIMITS[:, 1])

        if kp is None:
            kp = self._kp.copy()
        else:
            kp = np.asarray(kp, dtype=np.float64).ravel()

        if kd is None:
            kd = self._kd.copy()
        else:
            kd = np.asarray(kd, dtype=np.float64).ravel()

        self._loop.set_gravity_enabled(True)
        cmds = JointCommands(
            pos=pos,
            vel=np.zeros(N_JOINTS),
            kp=kp,
            kd=kd,
            torques=np.zeros(N_JOINTS),
        )
        self._loop.update_commands(cmds)

    def command_joint_state(
        self,
        pos: np.ndarray,
        vel: np.ndarray = None,
        kp: np.ndarray = None,
        kd: np.ndarray = None,
        torques: np.ndarray = None,
        gravity_enabled: bool = True,
    ) -> None:
        """Full MIT control with optional gravity compensation.

        Parameters
        ----------
        pos : array-like, shape (7,)
            Target joint positions in radians.
        vel : array-like, shape (7,), optional
            Target joint velocities in rad/s. Default zeros.
        kp : array-like, shape (7,), optional
            Proportional gains. Default uses DEFAULT_KP.
        kd : array-like, shape (7,), optional
            Derivative gains. Default uses DEFAULT_KD.
        torques : array-like, shape (7,), optional
            User feed-forward torques in N·m (gravity compensation
            is added on top automatically). Default zeros.
        gravity_enabled : bool, optional
            If True, add model gravity compensation on top of torques.
            Set False for pure MIT torque-channel diagnostics.
        """
        pos = np.asarray(pos, dtype=np.float64).ravel()
        if len(pos) != N_JOINTS:
            raise ValueError(f"Expected {N_JOINTS} joint angles, got {len(pos)}")
        pos = np.clip(pos, NERO_JOINT_LIMITS[:, 0], NERO_JOINT_LIMITS[:, 1])

        if vel is None:
            vel = np.zeros(N_JOINTS)
        else:
            vel = np.asarray(vel, dtype=np.float64).ravel()

        if kp is None:
            kp = self._kp.copy()
        else:
            kp = np.asarray(kp, dtype=np.float64).ravel()

        if kd is None:
            kd = self._kd.copy()
        else:
            kd = np.asarray(kd, dtype=np.float64).ravel()

        if torques is None:
            torques = np.zeros(N_JOINTS)
        else:
            torques = np.asarray(torques, dtype=np.float64).ravel()

        self._loop.set_gravity_enabled(gravity_enabled)
        cmds = JointCommands(
            pos=pos,
            vel=vel,
            kp=kp,
            kd=kd,
            torques=torques,
        )
        self._loop.update_commands(cmds)

    def update_gains(
        self,
        kp: np.ndarray = None,
        kd: np.ndarray = None,
    ) -> None:
        """Update the default PD gains.

        Parameters
        ----------
        kp : array-like, shape (7,), optional
        kd : array-like, shape (7,), optional
        """
        if kp is not None:
            self._kp = np.asarray(kp, dtype=np.float64).ravel()
        if kd is not None:
            self._kd = np.asarray(kd, dtype=np.float64).ravel()

    # ===================== In-loop Cartesian impedance =====================

    def set_cartesian_impedance(self, cfg) -> None:
        """Enable in-loop Cartesian impedance (author-validated Panthera law)
        computed fresh at the 200 Hz loop rate. ``cfg`` is an
        ``ImpedanceConfig`` (control_backends.force_ctrl.data_types)."""
        self._loop.set_cartesian_impedance(cfg)

    def disable_cartesian_impedance(self) -> None:
        """Disable in-loop impedance and fall back to the buffered command path."""
        self._loop.disable_cartesian_impedance()

    # ========================= State Reading =========================

    def get_joint_state(self) -> JointState:
        """Get the latest joint state.

        Returns
        -------
        JointState or None
            Contains pos (7,), vel (7,), torque (7,), timestamp.
        """
        return self._loop.get_joint_state()

    def get_joint_pos(self) -> np.ndarray:
        """Get current joint positions.

        Returns
        -------
        np.ndarray, shape (7,) or None
        """
        # Once the 200 Hz loop is running, use its immutable snapshot.  Calling
        # the SDK getter concurrently from a monitor/ROS thread mutates the
        # driver's shared feedback message while the control thread reads it.
        state = self._loop.get_joint_state()
        if state is not None:
            q = np.asarray(state.pos, dtype=np.float64).ravel()
        else:
            ja = self._robot.get_joint_angles()
            if ja is None:
                return None
            q = np.asarray(ja.msg, dtype=np.float64).ravel()
        if q.shape != (N_JOINTS,) or not np.all(np.isfinite(q)):
            raise RuntimeError(
                "Invalid joint feedback: "
                f"expected finite shape ({N_JOINTS},), got "
                f"shape={q.shape}, values={q.tolist()}")
        return q.copy()

    def get_debug_state(self, include_trace: bool = False) -> dict:
        """Get latest force-control debug values."""
        return self._loop.get_debug_state(include_trace=include_trace)

    # ========================= Safety =========================

    def emergency_stop(self) -> None:
        """Issue the electronic e-stop immediately, then halt the loop."""
        self._emergency_stopped = True
        errors = []
        try:
            request = getattr(self._loop, "request_emergency_stop", None)
            if callable(request):
                request("Force controller emergency stop")
            elif not getattr(self._loop, "_electronic_stop_sent", False):
                self._robot.electronic_emergency_stop()
                self._loop._electronic_stop_sent = True
        except Exception as exc:
            errors.append(f"electronic emergency stop failed: {exc}")
        try:
            self._loop.stop()
        except Exception as exc:
            errors.append(f"MIT loop stop after emergency stop failed: {exc}")
        if errors:
            message = "; ".join(errors)
            logger.critical("Emergency stop incomplete: %s", message)
            raise RuntimeError(message)
        logger.warning("Emergency stop triggered")

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        return False
