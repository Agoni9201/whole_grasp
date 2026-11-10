"""SDK CPV joint-velocity backend.

This backend is the closest Nero SDK mapping to Panthera's ``Joint_Vel`` path:

    TCP velocity -> Jacobian pseudo-inverse -> joint velocity -> move_cpv_vel()

It requires the Nero v112 driver because CPV mode is only exposed there.
"""

from __future__ import annotations

import logging
import time
from typing import Callable, Optional

import numpy as np
import numpy.typing as npt

from .base import BackendCommand, ControlBackend

logger = logging.getLogger(__name__)


class SdkCpvVelocityBackend(ControlBackend):
    """Send joint velocity commands through Nero CPV mode."""

    def __init__(
        self,
        dry_run: bool = True,
        channel: str = "can0",
        interface: str = "socketcan",
        firmware: str = "v112",
        max_joint_vel: float | npt.NDArray[np.float64] = 0.02,
        velocity_signs: float | npt.NDArray[np.float64] = 1.0,
        prepare_cpv: bool = True,
        enable: bool = True,
        set_follower_mode: bool = False,
        cpv_acc: float = 0.5,
        cpv_dcc: float = 0.5,
        cpv_cv: float = 0.5,
        prepare_delay_s: float = 0.2,
        auto_set_follower_when_disabled: bool = True,
        robot_factory: Optional[Callable[[], object]] = None,
        robot_already_connected: bool = False,
        stop_on_disconnect: bool = True,
    ) -> None:
        super().__init__(name="sdk_cpv_velocity", dry_run=dry_run)
        self.channel = channel
        self.interface = interface
        self.firmware = firmware
        self.max_joint_vel = np.asarray(max_joint_vel, dtype=np.float64)
        if self.max_joint_vel.ndim == 0:
            self.max_joint_vel = np.full(7, float(self.max_joint_vel), dtype=np.float64)
        if self.max_joint_vel.shape != (7,):
            raise ValueError(f"max_joint_vel must be scalar or shape (7,), got {self.max_joint_vel.shape}")
        self.velocity_signs = np.asarray(velocity_signs, dtype=np.float64)
        if self.velocity_signs.ndim == 0:
            self.velocity_signs = np.full(7, float(self.velocity_signs), dtype=np.float64)
        if self.velocity_signs.shape != (7,):
            raise ValueError(
                f"velocity_signs must be scalar or shape (7,), got {self.velocity_signs.shape}"
            )
        if np.any(np.abs(self.velocity_signs) < 1e-9):
            raise ValueError("velocity_signs entries must be non-zero")
        self.velocity_signs = np.sign(self.velocity_signs)
        self.prepare_cpv = bool(prepare_cpv)
        self.enable = bool(enable)
        self.set_follower_mode = bool(set_follower_mode)
        self.cpv_acc = float(cpv_acc)
        self.cpv_dcc = float(cpv_dcc)
        self.cpv_cv = float(cpv_cv)
        self.prepare_delay_s = float(prepare_delay_s)
        self.auto_set_follower_when_disabled = bool(auto_set_follower_when_disabled)
        self._robot_factory = robot_factory
        self._robot_already_connected = bool(robot_already_connected)
        self._stop_on_disconnect = stop_on_disconnect
        self.robot = None

    def connect(self) -> None:
        if self.dry_run:
            logger.info("[DRY-RUN] Would connect CPV velocity backend on channel=%s", self.channel)
            self._connected = True
            return

        if self.firmware != "v112":
            raise RuntimeError("Nero CPV velocity backend requires pyAgxArm firmware='v112'")

        if self._robot_factory is not None:
            self.robot = self._robot_factory()
        else:
            from pyAgxArm import AgxArmFactory, create_agx_arm_config

            cfg = create_agx_arm_config(
                robot="nero",
                comm="can",
                channel=self.channel,
                interface=self.interface,
                firmeware_version=self.firmware,
            )
            self.robot = AgxArmFactory.create_arm(cfg)

        if not self._robot_already_connected:
            self.robot.connect(start_read_thread=True)
        if not hasattr(self.robot, "move_cpv_vel"):
            self.robot.disconnect()
            self.robot = None
            raise RuntimeError("Installed Nero SDK driver does not expose move_cpv_vel(); use firmware='v112'")
        if self.prepare_cpv:
            self._prepare_cpv()
        self._connected = True
        logger.info("SdkCpvVelocityBackend connected on channel=%s", self.channel)

    def _prepare_cpv(self) -> None:
        if self.robot is None:
            raise RuntimeError("Cannot prepare CPV before robot connection")

        should_set_follower = self.set_follower_mode or self._should_auto_set_follower_mode()
        if should_set_follower and hasattr(self.robot, "set_follower_mode"):
            logger.info("Setting Nero follower mode before CPV velocity control")
            self.robot.set_follower_mode()
            self._prepare_delay()

        if self.enable:
            logger.info("Enabling Nero joints before CPV velocity control")
            enabled = False
            for _ in range(20):
                enabled = bool(self.robot.enable(255))
                if enabled:
                    break
                if self._should_auto_set_follower_mode():
                    logger.info("Retrying Nero follower mode before enable")
                    self.robot.set_follower_mode()
                    self._prepare_delay()
            if not enabled:
                enable_text = self._format_enable_list()
                raise RuntimeError(
                    "Failed to enable all Nero joints for CPV velocity control"
                    + ("" if enable_text is None else f"; enable_list={enable_text}")
                )
            self._prepare_delay()

        self.robot.set_motion_mode(self.robot.OPTIONS.MOTION_MODE.CPV)
        self._send_zero_velocity_raw()
        self._prepare_delay()

        for joint_index in range(1, 8):
            if self.cpv_acc > 0.0:
                self.robot.set_cpv_acc(joint_index, self.cpv_acc)
            if self.cpv_dcc > 0.0:
                self.robot.set_cpv_dcc(joint_index, self.cpv_dcc)
            if self.cpv_cv > 0.0:
                self.robot.set_cpv_cv(joint_index, self.cpv_cv)
        self._send_zero_velocity_raw()

    def _prepare_delay(self) -> None:
        if self.prepare_delay_s > 0.0:
            time.sleep(self.prepare_delay_s)

    def _read_enable_list(self) -> Optional[list[bool]]:
        if self.robot is None:
            return None
        method = getattr(self.robot, "get_joints_enable_status_list", None)
        if method is None:
            return None
        try:
            values = list(method())
        except Exception:
            return None
        if len(values) != 7:
            return None
        return [bool(value) for value in values]

    def _format_enable_list(self) -> Optional[str]:
        values = self._read_enable_list()
        if values is None:
            return None
        return str(values)

    def _should_auto_set_follower_mode(self) -> bool:
        if not self.auto_set_follower_when_disabled:
            return False
        if self.set_follower_mode:
            return False
        if self.robot is None or not hasattr(self.robot, "set_follower_mode"):
            return False
        values = self._read_enable_list()
        if values is None:
            return False
        return not any(values)

    def _send_zero_velocity_raw(self, repeats: int = 3, period_s: float = 0.02) -> None:
        if self.robot is None:
            return
        for repeat in range(repeats):
            for joint_index in range(1, 8):
                self.robot.move_cpv_vel(joint_index, 0.0)
            if repeat + 1 < repeats and period_s > 0.0:
                time.sleep(period_s)

    def send(self, cmd: BackendCommand) -> None:
        qdot_cmd = np.asarray(cmd.qdot_cmd, dtype=np.float64).ravel()
        if len(qdot_cmd) != 7:
            raise ValueError(f"qdot_cmd must have shape (7,), got {qdot_cmd.shape}")

        qdot_cmd = np.clip(qdot_cmd, -self.max_joint_vel, self.max_joint_vel)
        sdk_qdot_cmd = qdot_cmd * self.velocity_signs

        if self.dry_run:
            self._log_command(
                BackendCommand(
                    q_cmd=np.asarray(cmd.q_cmd, dtype=np.float64).ravel(),
                    qdot_cmd=sdk_qdot_cmd,
                    torque_ff=np.asarray(cmd.torque_ff, dtype=np.float64).ravel(),
                    kp=np.asarray(cmd.kp, dtype=np.float64).ravel(),
                    kd=np.asarray(cmd.kd, dtype=np.float64).ravel(),
                    mode="velocity",
                    stamp=cmd.stamp,
                )
            )
            return

        if self.robot is None:
            raise RuntimeError("SdkCpvVelocityBackend is not connected")

        for joint_index, vel in enumerate(sdk_qdot_cmd, start=1):
            self.robot.move_cpv_vel(joint_index, float(vel))

    def get_joint_angles(self) -> Optional[npt.NDArray[np.float64]]:
        if self.robot is None:
            return None
        msg = self.robot.get_joint_angles()
        if msg is None:
            return None
        return np.asarray(msg.msg, dtype=np.float64).ravel()

    def get_flange_pose(self) -> Optional[npt.NDArray[np.float64]]:
        if self.robot is None:
            return None
        for method_name in ("get_flange_pose", "get_tcp_pose"):
            method = getattr(self.robot, method_name, None)
            if method is None:
                continue
            pose = method()
            if pose is None:
                continue
            msg = getattr(pose, "msg", pose)
            return np.asarray(msg, dtype=np.float64).ravel()
        return None

    def disconnect(self) -> None:
        if self.robot is not None:
            try:
                if self._stop_on_disconnect:
                    zero = BackendCommand(
                        q_cmd=np.zeros(7, dtype=np.float64),
                        qdot_cmd=np.zeros(7, dtype=np.float64),
                        mode="velocity",
                    )
                    self.send(zero)
            finally:
                self.robot.disconnect()
                self.robot = None
        logger.info("SdkCpvVelocityBackend disconnected")
        self._connected = False
