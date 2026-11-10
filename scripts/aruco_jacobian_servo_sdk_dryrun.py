#!/usr/bin/env python3
"""SDK-feedback dry-run for ArUco Jacobian servo.

This is the non-ROS dry-run path.  It connects to the Nero SDK only to read
joint feedback, then computes Panthera-style position-only visual-servo
commands for a supplied base-frame marker pose.

It does not enable the arm, does not switch motion mode, and does not send
CPV / MIT / move_j commands.

Run:
    conda activate agilexrobot
    python3 scripts/aruco_jacobian_servo_sdk_dryrun.py \
      --marker-center 0.25,0.0,0.45 \
      --marker-normal -1,0,0
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from typing import Optional

import numpy as np


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(SCRIPT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from control_backends.base import BackendCommand  # noqa: E402
from control_backends.sdk_cpv_velocity_backend import SdkCpvVelocityBackend  # noqa: E402
from servo.jacobian_position_servo import JacobianPositionServo, ServoParams  # noqa: E402


def _vec3(text: str) -> np.ndarray:
    values = [float(item) for item in text.split(",")]
    if len(values) != 3:
        raise argparse.ArgumentTypeError("expected three comma-separated floats")
    return np.asarray(values, dtype=np.float64)


def _vec7(text: str) -> np.ndarray:
    values = [float(item) for item in text.split(",")]
    if len(values) != 7:
        raise argparse.ArgumentTypeError("expected seven comma-separated floats")
    return np.asarray(values, dtype=np.float64)


def _format(values, precision: int = 4) -> str:
    return "[" + ", ".join(f"{float(v):.{precision}f}" for v in values) + "]"


class SdkJointFeedbackReader:
    """Read-only pyAgxArm joint feedback helper."""

    def __init__(
        self,
        channel: str = "can0",
        interface: str = "socketcan",
        firmware: str = "v111",
    ) -> None:
        self.channel = channel
        self.interface = interface
        self.firmware = firmware
        self.robot = None

    def connect(self) -> None:
        from pyAgxArm import AgxArmFactory, create_agx_arm_config

        cfg = create_agx_arm_config(
            robot="nero",
            comm="can",
            channel=self.channel,
            interface=self.interface,
            firmeware_version=self.firmware,
        )
        self.robot = AgxArmFactory.create_arm(cfg)
        self.robot.connect(start_read_thread=True)

    def read_joint_angles(self) -> Optional[np.ndarray]:
        if self.robot is None:
            return None
        msg = self.robot.get_joint_angles()
        if msg is None:
            return None
        return np.asarray(msg.msg, dtype=np.float64).ravel()

    def disconnect(self) -> None:
        if self.robot is None:
            return
        try:
            self.robot.disconnect()
        finally:
            self.robot = None

    def take_robot(self):
        robot = self.robot
        self.robot = None
        return robot


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--channel", default="can0")
    parser.add_argument("--interface", default="socketcan")
    parser.add_argument("--firmware", default="v111")
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--servo-hz", type=float, default=30.0)
    parser.add_argument("--log-period", type=float, default=0.5)
    parser.add_argument(
        "--q",
        type=_vec7,
        default=None,
        help="Optional manual current joint angles in radians; skips SDK feedback.",
    )
    parser.add_argument(
        "--marker-center",
        type=_vec3,
        required=True,
        help="Marker center in base frame, comma-separated meters.",
    )
    parser.add_argument(
        "--marker-normal",
        type=_vec3,
        required=True,
        help="Marker normal in base frame.",
    )
    parser.add_argument("--desired-distance", type=float, default=0.15)
    parser.add_argument("--linear-gain", type=float, default=1.0)
    parser.add_argument("--max-linear-vel", type=float, default=0.02)
    parser.add_argument("--damping", type=float, default=0.04)
    parser.add_argument("--max-joint-vel", type=float, default=0.05)
    parser.add_argument("--max-joint-step", type=float, default=0.003)
    parser.add_argument("--target-filter-alpha", type=float, default=0.20)
    parser.add_argument("--normal-filter-alpha", type=float, default=0.20)
    parser.add_argument("--qdot-filter-alpha", type=float, default=0.35)
    parser.add_argument("--target-timeout-s", type=float, default=0.45)
    parser.add_argument("--max-target-jump", type=float, default=0.06)
    parser.add_argument("--position-deadband", type=float, default=0.002)
    parser.add_argument("--kp-scale", type=float, default=1.0)
    parser.add_argument("--kd-scale", type=float, default=1.0)
    parser.add_argument(
        "--verbose-backend",
        action="store_true",
        help="Print every dry-run BackendCommand.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO if args.verbose_backend else logging.WARNING,
        format="%(levelname)s: %(message)s",
    )

    params = ServoParams(
        servo_hz=args.servo_hz,
        desired_distance=args.desired_distance,
        linear_gain=args.linear_gain,
        max_linear_vel=args.max_linear_vel,
        damping=args.damping,
        max_joint_vel=args.max_joint_vel,
        max_joint_step=args.max_joint_step,
        target_filter_alpha=args.target_filter_alpha,
        normal_filter_alpha=args.normal_filter_alpha,
        qdot_filter_alpha=args.qdot_filter_alpha,
        target_timeout_s=args.target_timeout_s,
        max_target_jump=args.max_target_jump,
        position_deadband=args.position_deadband,
    )
    servo = JacobianPositionServo(params=params)
    backend = SdkCpvVelocityBackend(
        dry_run=True,
        channel=args.channel,
        max_joint_vel=args.max_joint_vel,
    )
    backend.connect()

    reader = None
    if args.q is None:
        reader = SdkJointFeedbackReader(
            channel=args.channel,
            interface=args.interface,
            firmware=args.firmware,
        )
        print("Connecting SDK read-only feedback. No enable/mode/command will be sent.")
        reader.connect()
    else:
        print("Using manual --q. SDK feedback is not connected.")

    print("=== SDK ArUco Jacobian Servo Dry-Run ===")
    print("No real robot command will be sent.")
    print(f"marker_center={_format(args.marker_center, 4)}")
    print(f"marker_normal={_format(args.marker_normal, 4)}")
    print(
        f"servo_hz={params.servo_hz:.1f}, desired_distance={params.desired_distance:.3f}m, "
        f"max_linear_vel={params.max_linear_vel:.3f}m/s"
    )

    t_start = time.monotonic()
    last_print = 0.0
    count = 0
    try:
        while True:
            now = time.monotonic()
            if args.duration > 0.0 and now - t_start >= args.duration:
                break

            if args.q is not None:
                q_now = args.q.copy()
            else:
                q_now = reader.read_joint_angles()
                if q_now is None:
                    if now - last_print >= args.log_period:
                        print("waiting for SDK joint feedback")
                        last_print = now
                    time.sleep(0.02)
                    continue

            result = servo.update(
                q_now=q_now,
                target_position=args.marker_center,
                target_normal=args.marker_normal,
                stamp=now,
            )
            cmd = BackendCommand(
                q_cmd=q_now.copy(),
                qdot_cmd=result["qdot"],
                mode="velocity",
                stamp=now,
            )
            backend.send(cmd)
            count += 1

            if now - last_print >= args.log_period:
                q_step = result["q_cmd"] - q_now
                normal_side = float(
                    np.dot(
                        result["tcp_now"] - result["marker_center"],
                        result["marker_normal"],
                    )
                )
                print(
                    f"t={now - t_start:5.2f}s count={count} "
                    f"err={np.linalg.norm(result['error']):.4f}m "
                    f"qdot_max={np.max(np.abs(result['qdot'])):.5f}rad/s "
                    f"step_max={np.max(np.abs(q_step)):.5f}rad "
                    f"normal_side={normal_side:.4f}m"
                )
                print(
                    f"  tcp_now={_format(result['tcp_now'], 4)} "
                    f"target_tcp={_format(result['target_tcp'], 4)} "
                    f"normal={_format(result['marker_normal'], 4)}"
                )
                print(
                    f"  q_now_deg={_format(np.degrees(q_now), 2)} "
                    f"q_cmd_deg={_format(np.degrees(result['q_cmd']), 2)}"
                )
                last_print = now

            time.sleep(max(0.0, 1.0 / params.servo_hz))

    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        if reader is not None:
            reader.disconnect()
        backend.disconnect()
        print("Stopped dry-run")


if __name__ == "__main__":
    main()
