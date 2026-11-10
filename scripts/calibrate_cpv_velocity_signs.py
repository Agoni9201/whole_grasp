#!/usr/bin/env python3
"""Calibrate Nero CPV joint velocity signs.

This script sends a very small CPV velocity pulse to one joint at a time and
compares the SDK joint-angle feedback before/after the pulse.  The output is a
``--cpv-signs`` vector for the visual-servo execute script.

Run only with the arm in a clear, safe posture and emergency stop available.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(SCRIPT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from control_backends.base import BackendCommand  # noqa: E402
from control_backends.sdk_cpv_velocity_backend import SdkCpvVelocityBackend  # noqa: E402
from servo.execution_safety import ExecutionSafetyMonitor, SafetyParams  # noqa: E402


def _format(values, precision: int = 4) -> str:
    return "[" + ", ".join(f"{float(v):.{precision}f}" for v in values) + "]"


def _read_q(backend: SdkCpvVelocityBackend, timeout: float) -> np.ndarray:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        q = backend.get_joint_angles()
        if q is not None and len(q) == 7:
            return np.asarray(q, dtype=np.float64).ravel()
        time.sleep(0.02)
    raise RuntimeError("No SDK joint feedback")


def _send_qdot(backend: SdkCpvVelocityBackend, qdot: np.ndarray) -> None:
    q_now = backend.get_joint_angles()
    if q_now is None or len(q_now) != 7:
        q_now = np.zeros(7, dtype=np.float64)
    backend.send(
        BackendCommand(
            q_cmd=np.asarray(q_now, dtype=np.float64).ravel(),
            qdot_cmd=np.asarray(qdot, dtype=np.float64).ravel(),
            mode="velocity",
            stamp=time.monotonic(),
        )
    )


def _call_optional(robot, method_name: str, *args) -> object:
    method = getattr(robot, method_name, None)
    if method is None:
        return None
    return method(*args)


def _print_robot_status(robot) -> None:
    status = _call_optional(robot, "get_arm_status")
    status_msg = None if status is None else getattr(status, "msg", status)
    print(f"arm_status={status_msg}")

    enable_list = _call_optional(robot, "get_joints_enable_status_list")
    print(f"enable_list={enable_list}")

    auto_mode = _call_optional(robot, "get_auto_set_motion_mode_enabled")
    print(f"auto_set_motion_mode={auto_mode}")


def _prepare_cpv(robot, args: argparse.Namespace) -> None:
    if args.set_follower_mode:
        print("setting follower mode")
        _call_optional(robot, "set_follower_mode")
        time.sleep(0.2)

    if args.enable:
        print("enabling all joints")
        for _ in range(20):
            if robot.enable(255):
                break
            time.sleep(0.05)
        time.sleep(0.2)

    print("setting CPV motion mode")
    robot.set_motion_mode(robot.OPTIONS.MOTION_MODE.CPV)
    time.sleep(0.2)

    for joint in range(1, 8):
        if args.cpv_acc > 0.0:
            ok = robot.set_cpv_acc(joint, args.cpv_acc)
            print(f"  J{joint} set_cpv_acc({args.cpv_acc:.3f}) -> {ok}")
        if args.cpv_dcc > 0.0:
            ok = robot.set_cpv_dcc(joint, args.cpv_dcc)
            print(f"  J{joint} set_cpv_dcc({args.cpv_dcc:.3f}) -> {ok}")
        if args.cpv_cv > 0.0:
            ok = robot.set_cpv_cv(joint, args.cpv_cv)
            print(f"  J{joint} set_cpv_cv({args.cpv_cv:.3f}) -> {ok}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="Required to send real CPV velocity pulses.")
    parser.add_argument("--channel", default="can0")
    parser.add_argument("--interface", default="socketcan")
    parser.add_argument("--firmware", default="v112")
    parser.add_argument("--velocity", type=float, default=0.015, help="Pulse velocity in rad/s.")
    parser.add_argument("--pulse-s", type=float, default=0.7, help="Pulse duration per joint.")
    parser.add_argument("--settle-s", type=float, default=0.4, help="Zero-velocity settle after each pulse.")
    parser.add_argument("--feedback-timeout", type=float, default=3.0)
    parser.add_argument("--min-delta-deg", type=float, default=0.15)
    parser.add_argument("--startup-limit-margin", type=float, default=0.08)
    parser.add_argument("--no-enable", dest="enable", action="store_false")
    parser.set_defaults(enable=True)
    parser.add_argument(
        "--set-follower-mode",
        dest="set_follower_mode",
        action="store_true",
        help="Diagnostic only: can cause large jumps if the arm is already enabled in CPV.",
    )
    parser.add_argument("--no-set-follower-mode", dest="set_follower_mode", action="store_false")
    parser.set_defaults(set_follower_mode=False)
    parser.add_argument("--cpv-acc", type=float, default=0.30)
    parser.add_argument("--cpv-dcc", type=float, default=0.30)
    parser.add_argument("--cpv-cv", type=float, default=0.20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.execute:
        raise SystemExit("Refusing to send CPV pulses without --execute.")
    if args.firmware != "v112":
        raise SystemExit("CPV velocity calibration requires --firmware v112.")
    if args.velocity <= 0.0:
        raise SystemExit("--velocity must be positive.")

    backend = SdkCpvVelocityBackend(
        dry_run=False,
        channel=args.channel,
        interface=args.interface,
        firmware=args.firmware,
        max_joint_vel=args.velocity,
        prepare_cpv=False,
        stop_on_disconnect=True,
    )
    monitor = ExecutionSafetyMonitor(
        SafetyParams(
            startup_limit_margin=args.startup_limit_margin,
            runtime_limit_margin=0.04,
            max_tracking_error=0.0,
            max_tracking_error_s=1.0,
        )
    )

    signs = []
    print("=== CPV Velocity Sign Calibration ===")
    print("This sends one tiny positive velocity pulse per joint.")
    print("Keep emergency stop available.")
    print(f"pulse velocity={args.velocity:.4f}rad/s pulse_s={args.pulse_s:.2f}s")

    try:
        backend.connect()
        assert backend.robot is not None
        print("\nBefore CPV prepare:")
        _print_robot_status(backend.robot)
        _prepare_cpv(backend.robot, args)
        print("\nAfter CPV prepare:")
        _print_robot_status(backend.robot)

        q_start = _read_q(backend, args.feedback_timeout)
        ok, reason = monitor.check_start_posture(q_start)
        if not ok:
            raise RuntimeError(f"Unsafe start posture: {reason}")
        print(f"q_start_deg={_format(np.degrees(q_start), 3)}")

        for joint in range(7):
            q_before = _read_q(backend, args.feedback_timeout)
            pulse = np.zeros(7, dtype=np.float64)
            pulse[joint] = args.velocity
            print(f"\nJ{joint + 1}: command +{args.velocity:.4f}rad/s for {args.pulse_s:.2f}s")
            _send_qdot(backend, pulse)
            time.sleep(args.pulse_s)
            _send_qdot(backend, np.zeros(7, dtype=np.float64))
            time.sleep(args.settle_s)

            q_after = _read_q(backend, args.feedback_timeout)
            delta = q_after - q_before
            delta_deg = np.degrees(delta[joint])
            if abs(delta_deg) < args.min_delta_deg:
                sign = 0.0
                print(
                    f"  delta={delta_deg:+.3f}deg is below {args.min_delta_deg:.3f}deg; "
                    "repeat with a slightly larger --velocity or --pulse-s"
                )
            else:
                sign = 1.0 if delta[joint] > 0.0 else -1.0
                print(f"  q_before_deg={np.degrees(q_before[joint]):+.3f}")
                print(f"  q_after_deg ={np.degrees(q_after[joint]):+.3f}")
                print(f"  feedback delta={delta_deg:+.3f}deg -> cpv_sign={sign:+.0f}")
            signs.append(sign)

            ok, reason = monitor.check_start_posture(q_after)
            if not ok:
                raise RuntimeError(f"Post-pulse posture is unsafe: {reason}")

        if any(sign == 0.0 for sign in signs):
            print("\nCalibration incomplete: at least one joint did not move enough.")
        print("\nUse this with dry-run/execute after confirming all signs are non-zero:")
        print("--cpv-signs " + ",".join(f"{int(sign):d}" for sign in signs))
    finally:
        try:
            if backend.connected:
                _send_qdot(backend, np.zeros(7, dtype=np.float64))
        finally:
            backend.disconnect()


if __name__ == "__main__":
    main()
