#!/usr/bin/env python3
"""Diagnose Nero v112 CPV prepare drift.

This script does not run visual servo.  It connects to the SDK, then performs
CPV preparation one step at a time and prints joint drift after each step.
Keep emergency stop available.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Callable

import numpy as np


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(SCRIPT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def _format(values, precision: int = 3) -> str:
    return "[" + ", ".join(f"{float(v):.{precision}f}" for v in values) + "]"


def _read_q(robot, timeout_s: float = 3.0) -> np.ndarray:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        msg = robot.get_joint_angles()
        if msg is not None and len(msg.msg) == 7:
            return np.asarray(msg.msg, dtype=np.float64).ravel()
        time.sleep(0.02)
    raise RuntimeError(f"No joint feedback within {timeout_s:.1f}s")


def _zero_vel(robot, repeats: int = 3, period_s: float = 0.02) -> None:
    for repeat in range(repeats):
        for joint in range(1, 8):
            robot.move_cpv_vel(joint, 0.0)
        if repeat + 1 < repeats:
            time.sleep(period_s)


def _print_status(robot) -> None:
    get_status = getattr(robot, "get_arm_status", None)
    if get_status is not None:
        status = get_status()
        print(f"  arm_status={None if status is None else getattr(status, 'msg', status)}")
    get_enabled = getattr(robot, "get_joints_enable_status_list", None)
    if get_enabled is not None:
        print(f"  enable_list={get_enabled()}")


def _step(robot, name: str, fn: Callable[[], None], q_ref: np.ndarray, settle_s: float) -> np.ndarray:
    print(f"\n-- {name} --")
    q_before = _read_q(robot)
    print(f"before_q_deg={_format(np.degrees(q_before), 3)}")
    fn()
    time.sleep(settle_s)
    q_after = _read_q(robot)
    drift = q_after - q_ref
    step_drift = q_after - q_before
    print(f"after_q_deg ={_format(np.degrees(q_after), 3)}")
    print(f"step_diff_deg={_format(np.degrees(step_drift), 3)}")
    print(f"total_diff_deg={_format(np.degrees(drift), 3)} max={np.degrees(np.max(np.abs(drift))):.3f}deg")
    _print_status(robot)
    return q_after


def _zero_hold(robot, q_ref: np.ndarray, duration_s: float, command_hz: float, log_period_s: float) -> None:
    if duration_s <= 0.0:
        return
    period = 1.0 / float(command_hz)
    t_start = time.monotonic()
    last_log = t_start
    print(f"\n-- continuous zero move_cpv_vel hold {duration_s:.1f}s --")
    while time.monotonic() - t_start < duration_s:
        loop_start = time.monotonic()
        _zero_vel(robot, repeats=1, period_s=0.0)
        now = time.monotonic()
        if now - last_log >= log_period_s:
            q_now = _read_q(robot, timeout_s=1.0)
            drift = q_now - q_ref
            print(
                f"t={now - t_start:5.1f}s "
                f"q_deg={_format(np.degrees(q_now), 3)} "
                f"total_diff_deg={_format(np.degrees(drift), 3)} "
                f"max={np.degrees(np.max(np.abs(drift))):.3f}deg"
            )
            last_log = now
        sleep_s = period - (time.monotonic() - loop_start)
        if sleep_s > 0.0:
            time.sleep(sleep_s)
    q_final = _read_q(robot, timeout_s=1.0)
    drift = q_final - q_ref
    print(
        f"zero_hold_final_q_deg={_format(np.degrees(q_final), 3)} "
        f"total_diff_deg={_format(np.degrees(drift), 3)} "
        f"max={np.degrees(np.max(np.abs(drift))):.3f}deg"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="Required to touch the real SDK.")
    parser.add_argument("--channel", default="can0")
    parser.add_argument("--interface", default="socketcan")
    parser.add_argument("--firmware", default="v112")
    parser.add_argument("--settle-s", type=float, default=0.5)
    parser.add_argument("--cpv-acc", type=float, default=0.50)
    parser.add_argument("--cpv-dcc", type=float, default=0.50)
    parser.add_argument("--cpv-cv", type=float, default=0.50)
    parser.add_argument(
        "--include-follower-mode",
        action="store_true",
        help="Dangerous diagnostic step: may cause large jumps if the arm is already enabled in CPV.",
    )
    parser.add_argument("--zero-hold-s", type=float, default=10.0)
    parser.add_argument("--zero-hold-hz", type=float, default=30.0)
    parser.add_argument("--log-period", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.execute:
        raise SystemExit("Refusing to touch the real SDK without --execute.")
    if args.firmware != "v112":
        raise SystemExit("This diagnostic is for --firmware v112.")

    from pyAgxArm import AgxArmFactory, create_agx_arm_config

    cfg = create_agx_arm_config(
        robot="nero",
        comm="can",
        channel=args.channel,
        interface=args.interface,
        firmeware_version=args.firmware,
    )
    robot = AgxArmFactory.create_arm(cfg)
    robot.connect(start_read_thread=True)
    try:
        q0 = _read_q(robot)
        print("=== CPV Prepare Drift Diagnostic ===")
        print(f"q0_deg={_format(np.degrees(q0), 3)}")
        _print_status(robot)

        if args.include_follower_mode:
            _step(robot, "set_follower_mode", robot.set_follower_mode, q0, args.settle_s)
        else:
            print("\n-- set_follower_mode skipped by default --")
        _step(robot, "enable(255)", lambda: robot.enable(255), q0, args.settle_s)
        _step(
            robot,
            "set_motion_mode(CPV)",
            lambda: robot.set_motion_mode(robot.OPTIONS.MOTION_MODE.CPV),
            q0,
            args.settle_s,
        )
        _step(robot, "zero move_cpv_vel", lambda: _zero_vel(robot, repeats=8), q0, args.settle_s)
        _step(
            robot,
            "set_cpv_acc/dcc/cv",
            lambda: [
                (
                    robot.set_cpv_acc(joint, args.cpv_acc),
                    robot.set_cpv_dcc(joint, args.cpv_dcc),
                    robot.set_cpv_cv(joint, args.cpv_cv),
                )
                for joint in range(1, 8)
            ],
            q0,
            args.settle_s,
        )
        _step(robot, "final zero move_cpv_vel", lambda: _zero_vel(robot, repeats=12), q0, args.settle_s)
        _zero_hold(robot, q0, args.zero_hold_s, args.zero_hold_hz, args.log_period)
    finally:
        try:
            _zero_vel(robot, repeats=5)
        except Exception:
            pass
        robot.disconnect()
        print("Disconnected")


if __name__ == "__main__":
    main()
