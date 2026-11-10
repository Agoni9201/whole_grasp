"""SDK force_ctrl / MIT backend (main control path).

This is the Panthera-style main control backend that uses:
- move_mit() for sending 7-joint MIT control frames in a 200 Hz loop
- MuJoCo-based gravity compensation (NeroGravityCompensator)
- 200 Hz control loop (NeroMITLoop)

Reference: REFACTOR_PLAN.md Section 4.3
Reference: https://github.com/sjyylml/pyAgxArm (force_ctrl module)
"""

from __future__ import annotations

import logging
from typing import Optional, Type

import numpy as np
import numpy.typing as npt

from .base import ControlBackend, BackendCommand
from .force_ctrl.normal_shutdown import defer_sigint

logger = logging.getLogger(__name__)

DEFAULT_KP = np.array([30.0, 30.0, 30.0, 15.0, 10.0, 5.0, 5.0])
DEFAULT_KD = np.array([2.0, 2.0, 2.0, 1.0, 0.5, 0.3, 0.3])


class SdkForceCtrlBackend(ControlBackend):
    """Main control backend using SDK force_ctrl / MIT mode.

    Control flow:
        servo_core outputs q_cmd / qdot_cmd
        -> force backend overlays gravity compensation
        -> command_joint_state(pos=q_cmd, vel=qdot_cmd, kp=KP, kd=KD)
        -> move_mit_batch()

    IMPORTANT:
    - When using this backend, stop the ROS arm controller to avoid
      two control sources sending CAN commands simultaneously.
    - Gravity direction, gripper URDF inertia, torque sign, and TCP
      payload must be independently verified.
    """

    def __init__(
        self,
        dry_run: bool = True,
        channel: str = "can0",
        control_hz: float = 200.0,
        firmware: str = "v111",
        interface: str = "socketcan",
        torque_limit: float = 14.0,
        impedance_torque_limit: float = 4.0,
        gravity_factor: float = 1.0,
        gravity_factors: Optional[npt.NDArray[np.float64]] = None,
        zero_gravity: bool = False,
        urdf_path: str = "",
        default_kp: Optional[npt.NDArray[np.float64]] = None,
        default_kd: Optional[npt.NDArray[np.float64]] = None,
        gravity_compensation: bool = True,
        startup_joint_margin: float = 0.0,
        controller_cls: Optional[Type] = None,
        pin_model=None,
        pin_urdf_path: str = "",
        skip_reset: bool = False,
        keep_enabled_on_disconnect: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(name="sdk_force_ctrl", dry_run=dry_run)
        self.channel = channel
        self.control_hz = control_hz
        self.firmware = firmware
        self.interface = interface
        self.torque_limit = torque_limit
        # Symmetric clip applied to torque_ff BEFORE the MIT loop adds gravity
        # (mit_loop clips total = t_ff + grav to ±torque_limit AFTER the add).
        # The impedance Cartesian spring must not starve the gravity-comp budget.
        self.impedance_torque_limit = float(impedance_torque_limit)
        self.gravity_factor = gravity_factor
        self.gravity_factors = (
            None
            if gravity_factors is None
            else np.asarray(gravity_factors, dtype=np.float64).ravel()
        )
        self.zero_gravity = zero_gravity
        self.urdf_path = urdf_path
        self.gravity_compensation = gravity_compensation
        self.startup_joint_margin = float(startup_joint_margin)
        self.skip_reset = bool(skip_reset)
        self.keep_enabled_on_disconnect = bool(keep_enabled_on_disconnect)
        self._controller_cls = controller_cls

        # Pinocchio model for in-loop Cartesian impedance. None unless the caller
        # asks for impedance (pin_urdf_path set). When None, impedance mode is
        # unavailable and the buffered command path (visual servo, zero-gravity)
        # is entirely unaffected.
        self._pin_model = pin_model
        self._pin_urdf_path = pin_urdf_path

        self._default_kp = default_kp if default_kp is not None else DEFAULT_KP.copy()
        self._default_kd = default_kd if default_kd is not None else DEFAULT_KD.copy()

        # Will be initialized in connect()
        self._force_ctrl = None
        self._startup_failed = False

    def connect(self) -> None:
        """Connect to the arm via SDK and start the force controller."""
        if self.dry_run:
            logger.info("[DRY-RUN] Would connect to arm on channel=%s", self.channel)
            self._connected = True
            return

        controller_cls = self._controller_cls
        if controller_cls is None:
            from .force_ctrl.controller import NeroForceController
            controller_cls = NeroForceController

        # Build a Pinocchio model only if in-loop impedance is requested. Lazy
        # import so pure visual-servo users never pay the cost or take the dep.
        pin_model = self._pin_model
        if pin_model is None and self._pin_urdf_path:
            from models.nero_model import NeroPinocchioModel
            pin_model = NeroPinocchioModel(urdf_path=self._pin_urdf_path)
            self._pin_model = pin_model

        controller = controller_cls(
            channel=self.channel,
            urdf_path=self.urdf_path,
            zero_gravity=self.zero_gravity,
            gravity_factor=self.gravity_factor,
            gravity_factors=self.gravity_factors,
            control_freq=int(round(self.control_hz)),
            torque_limit=self.torque_limit,
            startup_joint_margin=self.startup_joint_margin,
            interface=self.interface,
            firmware=self.firmware,
            pin_model=pin_model,
            skip_reset=self.skip_reset,
        )
        self._force_ctrl = controller
        self._startup_failed = False
        try:
            controller.start()
        except KeyboardInterrupt:
            self._startup_failed = True
            with defer_sigint():
                try:
                    self._cleanup_failed_start(controller)
                except Exception:
                    self._connected = True
                    raise
                self._force_ctrl = None
                self._connected = False
            raise
        except Exception as start_exc:
            self._startup_failed = True
            try:
                self._cleanup_failed_start(controller)
            except Exception as cleanup_exc:
                self._connected = True
                raise RuntimeError(
                    f"Force controller start failed ({start_exc}); cleanup "
                    f"also failed ({cleanup_exc})") from start_exc
            self._force_ctrl = None
            self._connected = False
            raise
        self._connected = True
        logger.info("SdkForceCtrlBackend connected on channel=%s", self.channel)

    def _cleanup_failed_start(self, controller):
        cleanup = getattr(controller, "cleanup_failed_start", None)
        if callable(cleanup):
            cleanup()
        elif self.keep_enabled_on_disconnect:
            controller.stop_keep_enabled()
        else:
            controller.stop()

    def send(self, cmd: BackendCommand) -> None:
        """Send a unified command via MIT control."""
        if self.dry_run:
            self._log_command(cmd)
            return

        if self._force_ctrl is None:
            raise RuntimeError("SdkForceCtrlBackend is not connected")

        q_cmd = np.asarray(cmd.q_cmd, dtype=np.float64).ravel()
        qdot_cmd = np.asarray(cmd.qdot_cmd, dtype=np.float64).ravel()
        torque_ff = np.asarray(cmd.torque_ff, dtype=np.float64).ravel()
        kp = np.asarray(cmd.kp, dtype=np.float64).ravel()
        kd = np.asarray(cmd.kd, dtype=np.float64).ravel()
        for name, value in (
            ("q_cmd", q_cmd),
            ("qdot_cmd", qdot_cmd),
            ("torque_ff", torque_ff),
            ("kp", kp),
            ("kd", kd),
        ):
            if len(value) != 7:
                raise ValueError(f"{name} must have shape (7,), got {value.shape}")

        # Only fall back to defaults when kp/kd were not explicitly set
        # (all-NaN = unset).  Deliberate kp=0 / kd=0 (e.g. zero-gravity
        # mode) must be forwarded as-is.
        if np.all(np.isnan(kp)):
            kp = self._default_kp.copy()
        if np.all(np.isnan(kd)):
            kd = self._default_kd.copy()

        # Impedance mode: clip the Cartesian spring torque before the MIT loop
        # overlays gravity. Protects the gravity-compensation torque budget.
        if cmd.mode == "mit-impedance":
            lim = self.impedance_torque_limit
            torque_ff = np.clip(torque_ff, -lim, lim)

        self._force_ctrl.command_joint_state(
            pos=q_cmd,
            vel=qdot_cmd,
            kp=kp,
            kd=kd,
            torques=torque_ff,
            gravity_enabled=self.gravity_compensation,
        )

    def get_joint_angles(self) -> Optional[npt.NDArray[np.float64]]:
        """Read current joint angles from SDK."""
        if self._force_ctrl is None:
            return None
        q = self._force_ctrl.get_joint_pos()
        if q is None:
            return None
        return np.asarray(q, dtype=np.float64).ravel()

    def get_robot(self):
        """Access the underlying pyAgxArm Nero driver (gripper + 诊断)."""
        if self._force_ctrl is None:
            raise RuntimeError("backend not connected")
        return self._force_ctrl.robot


    def get_joint_velocity(self) -> Optional[npt.NDArray[np.float64]]:
        """Read filtered measured joint velocities (rad/s).

        The v111 driver does not report velocity directly, so the MIT loop
        synthesizes it from finite-differenced positions (low-pass filtered).
        Returns None until the loop has published its first JointState.
        """
        if self._force_ctrl is None:
            return None
        try:
            state = self._force_ctrl.get_joint_state()
        except Exception:
            return None
        if state is None or state.vel is None:
            return None
        return np.asarray(state.vel, dtype=np.float64).ravel()

    def get_servo_joint_bounds(self):
        """The same buffered position bounds enforced by the sole MIT sender."""
        from .force_ctrl.mit_loop import NERO_JOINT_LIMITS, JOINT_LIMIT_BUFFER
        buffer = float(getattr(self._force_ctrl, '_joint_limit_buffer', JOINT_LIMIT_BUFFER))
        return NERO_JOINT_LIMITS + np.array([buffer, -buffer])

    def get_debug_state(self, include_trace: bool = False) -> dict:
        """Return MIT-loop torque, status, and last impedance-trip diagnostics."""
        if self._force_ctrl is None:
            return {}
        return self._force_ctrl.get_debug_state(include_trace=include_trace)

    def start_control_diagnostics(self, path):
        if self._force_ctrl is None:
            raise RuntimeError("backend not connected")
        return self._force_ctrl._loop.ensure_control_diagnostics(path)

    # =================== In-loop Cartesian impedance ===================
    # The 200 Hz in-loop impedance (author-validated Panthera law). The slow
    # caller installs or refreshes the target at its own slower rate; the MIT
    # loop computes the force every cycle. ``send(mode="mit-impedance")`` above
    # is retained only for backward compatibility.

    def set_cartesian_impedance(self, cfg) -> None:
        """Enable in-loop Cartesian impedance. ``cfg`` is an ``ImpedanceConfig``
        (control_backends.force_ctrl.data_types). Requires the backend to have
        been constructed with pin_urdf_path (or pin_model)."""
        if self._force_ctrl is None:
            raise RuntimeError("SdkForceCtrlBackend is not connected")
        self._force_ctrl.set_cartesian_impedance(cfg)

    def disable_cartesian_impedance(self) -> None:
        """Disable in-loop impedance; fall back to the buffered command path."""
        if self._force_ctrl is None:
            return
        self._force_ctrl.disable_cartesian_impedance()

    def set_gravity_factors(self, factors) -> None:
        if self._force_ctrl is None:
            raise RuntimeError("SdkForceCtrlBackend is not connected")
        self._force_ctrl.set_gravity_factors(factors)

    def emergency_stop(self) -> None:
        """Stop the MIT loop and issue the controller's electronic e-stop."""
        if self._force_ctrl is None:
            return
        self._force_ctrl.emergency_stop()

    def get_flange_pose(self) -> Optional[npt.NDArray[np.float64]]:
        """Read current flange/TCP pose from SDK."""
        if self._force_ctrl is None:
            return None
        robot = getattr(self._force_ctrl, "robot", None)
        if robot is None:
            return None
        for method_name in ("get_flange_pose", "get_tcp_pose"):
            method = getattr(robot, method_name, None)
            if method is None:
                continue
            pose = method()
            if pose is None:
                continue
            msg = getattr(pose, "msg", pose)
            return np.asarray(msg, dtype=np.float64).ravel()
        return None

    def prepare_shutdown(self) -> None:
        if self._force_ctrl is not None and self.keep_enabled_on_disconnect:
            self._force_ctrl.prepare_shutdown()

    def disconnect(self) -> None:
        """Stop control loop and disconnect."""
        if self._force_ctrl is not None:
            controller = self._force_ctrl
            try:
                if self._startup_failed:
                    self._cleanup_failed_start(controller)
                elif self.keep_enabled_on_disconnect:
                    controller.stop_keep_enabled()
                else:
                    controller.stop()
            except Exception:
                # Retain the controller so the caller can retry cleanup or
                # issue another emergency stop; do not claim disconnection.
                self._connected = True
                raise
            self._force_ctrl = None
            self._connected = False
        logger.info("SdkForceCtrlBackend disconnected")
        self._connected = False
