#!/usr/bin/env python3
"""Move slowly to the ArUco visual-servo start posture, then exit.

This script is intentionally separate from visual servo tracking.  It uses the
same force_ctrl / MIT backend as the position-hold test, ramps joint commands
in small joint-space steps, briefly holds the target posture, then stops the
controller and exits.

Run the tracking script only after this script has completed successfully.
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
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from aruco_start_pose import DEFAULT_ARUCO_START_Q_DEG, parse_deg7  # noqa: E402
from control_backends.base import BackendCommand  # noqa: E402
from control_backends.force_ctrl.controller import (  # noqa: E402
    DEFAULT_KD,
    DEFAULT_KP,
    NeroForceController,
)
from control_backends.force_ctrl.mit_loop import NERO_JOINT_LIMITS  # noqa: E402
from control_backends.sdk_cpv_velocity_backend import SdkCpvVelocityBackend  # noqa: E402


def _format(values, precision: int = 2) -> str:
    return "[" + ", ".join(f"{float(v):.{precision}f}" for v in values) + "]"


def parse_vec7(text: str, default: np.ndarray) -> np.ndarray:
    if not text:
        return default.copy()
    values = [float(item) for item in text.split(",")]
    if len(values) != 7:
        raise argparse.ArgumentTypeError(f"expected 7 comma-separated values, got {len(values)}")
    return np.asarray(values, dtype=np.float64)


def parse_float7(text: str) -> np.ndarray:
    values = [float(item) for item in text.split(",")]
    if len(values) != 7:
        raise argparse.ArgumentTypeError(f"expected 7 comma-separated values, got {len(values)}")
    return np.asarray(values, dtype=np.float64)


def step_towards(current: np.ndarray, target: np.ndarray, max_step: float) -> np.ndarray:
    """Move current toward target by at most max_step per joint."""
    current = np.asarray(current, dtype=np.float64).ravel()
    target = np.asarray(target, dtype=np.float64).ravel()
    delta = target - current
    return current + np.clip(delta, -max_step, max_step)


def joint_space_waypoints(start: np.ndarray, target: np.ndarray, max_step: float) -> list[np.ndarray]:
    """Split a large joint-space move into small absolute waypoints."""
    start = np.asarray(start, dtype=np.float64).ravel()
    target = np.asarray(target, dtype=np.float64).ravel()
    distance = float(np.max(np.abs(target - start)))
    if distance < 1e-12:
        return [target.copy()]
    steps = max(1, int(np.ceil(distance / max_step)))
    return [start + (target - start) * (i / steps) for i in range(1, steps + 1)]


def check_hard_limits(q: np.ndarray, limits: np.ndarray = NERO_JOINT_LIMITS) -> tuple[bool, str]:
    q = np.asarray(q, dtype=np.float64).ravel()
    bad = np.where((q < limits[:, 0]) | (q > limits[:, 1]))[0]
    if len(bad) == 0:
        return True, ""
    parts = []
    for index in bad:
        parts.append(
            f"joint{index + 1}={np.degrees(q[index]):.2f}deg "
            f"limit=[{np.degrees(limits[index, 0]):.2f}, {np.degrees(limits[index, 1]):.2f}]deg"
        )
    return False, "; ".join(parts)


def read_joint_angles(robot) -> np.ndarray | None:
    msg = robot.get_joint_angles()
    if msg is None:
        return None
    return np.asarray(msg.msg, dtype=np.float64).ravel()


def wait_until_enabled(robot, timeout_s: float = 5.0) -> None:
    timeout = time.time() + timeout_s
    while not robot.enable():
        if time.time() > timeout:
            raise RuntimeError(f"Failed to enable Nero arm within {timeout_s:.1f}s")
        time.sleep(0.02)


def prepare_move_j_mode(robot, firmware: str) -> None:
    """Put Nero into the SDK state expected by move_j."""
    if firmware != "v112":
        robot.set_normal_mode()
    time.sleep(0.2)
    wait_until_enabled(robot)
    robot.set_motion_mode(robot.OPTIONS.MOTION_MODE.J)
    time.sleep(0.3)


def _read_backend_q(backend: SdkCpvVelocityBackend, timeout_s: float = 3.0) -> np.ndarray:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        q = backend.get_joint_angles()
        if q is not None and len(q) == 7:
            return np.asarray(q, dtype=np.float64).ravel()
        time.sleep(0.02)
    raise RuntimeError(f"No joint feedback received within {timeout_s:.1f} seconds")


def _send_cpv_qdot(backend: SdkCpvVelocityBackend, q_now: np.ndarray, qdot: np.ndarray) -> None:
    backend.send(
        BackendCommand(
            q_cmd=np.asarray(q_now, dtype=np.float64).ravel(),
            qdot_cmd=np.asarray(qdot, dtype=np.float64).ravel(),
            mode="velocity",
            stamp=time.time(),
        )
    )


def cpv_position_velocity(args: argparse.Namespace, err: np.ndarray) -> np.ndarray:
    abs_err = np.abs(err)
    near = abs_err <= np.radians(float(args.cpv_near_target_deg))
    max_vel = np.where(
        near,
        min(float(args.cpv_near_max_joint_vel), float(args.max_joint_vel)),
        float(args.max_joint_vel),
    )
    qdot = np.clip(float(args.cpv_position_gain) * err, -max_vel, max_vel)

    deadband = np.radians(float(args.cpv_deadband_deg))
    qdot[abs_err <= deadband] = 0.0

    min_vel = float(args.cpv_min_joint_vel)
    if min_vel > 0.0:
        active = abs_err > deadband
        qdot_abs = np.abs(qdot)
        small = active & (qdot_abs < min_vel)
        qdot[small] = np.sign(err[small]) * min_vel
    return qdot


def run_cpv(args: argparse.Namespace, target_q: np.ndarray) -> None:
    """Use Nero v112 CPV joint velocity to relocate to the start pose."""
    if args.firmware != "v112":
        raise RuntimeError("CPV start-pose move requires --firmware v112")

    backend = SdkCpvVelocityBackend(
        dry_run=False,
        channel=args.channel,
        interface=args.interface,
        firmware=args.firmware,
        max_joint_vel=args.max_joint_vel,
        velocity_signs=args.cpv_signs,
        cpv_acc=args.cpv_acc,
        cpv_dcc=args.cpv_dcc,
        cpv_cv=args.cpv_cv,
        prepare_delay_s=args.cpv_prepare_delay_s,
        stop_on_disconnect=True,
    )

    q_final = None
    try:
        backend.connect()
        q_now = _read_backend_q(backend)
        ok, reason = check_hard_limits(q_now)
        if not ok:
            raise RuntimeError(f"Current posture violates joint limits: {reason}")

        tolerance = np.radians(float(args.target_tolerance_deg))
        period = 1.0 / float(args.freq)
        t_start = time.time()
        last_print = t_start

        print("=== Move To ArUco Start Pose (SDK CPV velocity) ===")
        print(f"q_now_deg={_format(np.degrees(q_now), 2)}")
        print(f"target_q_deg={_format(args.start_q_deg, 2)}")
        print(
            f"max_joint_vel={args.max_joint_vel:.4f}rad/s "
            f"near_max_joint_vel={args.cpv_near_max_joint_vel:.4f}rad/s "
            f"cpv_position_gain={args.cpv_position_gain:.3f} "
            f"cpv_signs={_format(args.cpv_signs, 0)}"
        )
        print("Ctrl+C sends zero CPV velocity and exits.")

        while time.time() - t_start < args.timeout:
            loop_start = time.time()
            q_now = _read_backend_q(backend)
            q_final = q_now.copy()
            ok, reason = check_hard_limits(q_now)
            if not ok:
                raise RuntimeError(f"Current posture violates joint limits: {reason}")

            err = target_q - q_now
            max_err = float(np.max(np.abs(err)))
            if max_err <= tolerance:
                _send_cpv_qdot(backend, q_now, np.zeros(7, dtype=np.float64))
                print(f"Reached start pose: max_err={np.degrees(max_err):.2f}deg")
                if args.hold_after > 0.0:
                    hold_until = time.time() + float(args.hold_after)
                    print(f"Holding zero CPV velocity for {args.hold_after:.1f}s before exit")
                    while time.time() < hold_until:
                        q_hold = _read_backend_q(backend, timeout_s=1.0)
                        _send_cpv_qdot(backend, q_hold, np.zeros(7, dtype=np.float64))
                        time.sleep(0.05)
                return

            qdot = cpv_position_velocity(args, err)
            _send_cpv_qdot(backend, q_now, qdot)

            now = time.time()
            if now - last_print >= 1.0:
                print(
                    f"t={now - t_start:5.1f}s "
                    f"max_err={np.degrees(max_err):.2f}deg "
                    f"qdot_max={np.max(np.abs(qdot)):.4f}rad/s "
                    f"q_deg={_format(np.degrees(q_now), 2)}"
                )
                last_print = now

            sleep_s = period - (time.time() - loop_start)
            if sleep_s > 0.0:
                time.sleep(sleep_s)

        raise RuntimeError(f"Timed out before reaching start pose: timeout={args.timeout:.1f}s")
    finally:
        try:
            if backend.connected:
                q_stop = q_final
                if q_stop is None:
                    q_stop = np.zeros(7, dtype=np.float64)
                _send_cpv_qdot(backend, q_stop, np.zeros(7, dtype=np.float64))
        finally:
            backend.disconnect()
            if q_final is not None:
                print(f"final_q_deg={_format(np.degrees(q_final), 2)}")
            print("Stopped CPV velocity backend; script exited")


def run_move_j(args: argparse.Namespace, target_q: np.ndarray) -> None:
    """Use SDK move_j to relocate to the start pose, then return."""
    from pyAgxArm import AgxArmFactory, create_agx_arm_config

    speed_percent = int(round(float(args.speed_percent)))
    if speed_percent < 1 or speed_percent > 100:
        raise ValueError(f"speed_percent must be in [1, 100], got {args.speed_percent}")

    cfg = create_agx_arm_config(
        robot="nero",
        comm="can",
        channel=args.channel,
        interface=args.interface,
        firmeware_version=args.firmware,
    )
    robot = AgxArmFactory.create_arm(cfg)
    robot.connect()

    try:
        robot.reset()
        time.sleep(0.2)
        robot.set_speed_percent(speed_percent)
        robot.set_auto_set_motion_mode_enabled(True)
        prepare_move_j_mode(robot, args.firmware)

        q_start = None
        timeout = time.time() + 3.0
        while q_start is None:
            q_start = read_joint_angles(robot)
            if time.time() > timeout:
                raise RuntimeError("No joint feedback received within 3 seconds")
            time.sleep(0.02)

        ok, reason = check_hard_limits(q_start)
        if not ok:
            raise RuntimeError(f"Current posture violates joint limits: {reason}")

        waypoint_step = np.radians(float(args.move_j_waypoint_deg))
        waypoints = joint_space_waypoints(q_start, target_q, waypoint_step)
        tolerance = np.radians(float(args.target_tolerance_deg))
        print("=== Move To ArUco Start Pose (SDK move_j) ===")
        print(f"q_now_deg={_format(np.degrees(q_start), 2)}")
        print(f"target_q_deg={_format(args.start_q_deg, 2)}")
        print(
            f"speed_percent={speed_percent} waypoint_step={args.move_j_waypoint_deg:.2f}deg "
            f"waypoints={len(waypoints)}"
        )

        for index, waypoint in enumerate(waypoints, start=1):
            robot.move_j(waypoint.tolist())
            waypoint_start = time.time()
            q_last = q_start.copy()
            while time.time() - waypoint_start < args.waypoint_timeout:
                q_now = read_joint_angles(robot)
                if q_now is not None:
                    q_last = q_now
                err = float(np.max(np.abs(q_last - waypoint)))
                total_err = float(np.max(np.abs(q_last - target_q)))
                if err <= tolerance:
                    break
                time.sleep(0.05)
            else:
                raise RuntimeError(
                    f"move_j waypoint {index}/{len(waypoints)} timed out; "
                    f"err={np.degrees(err):.2f}deg q_deg={_format(np.degrees(q_last), 2)}"
                )

            print(
                f"waypoint {index:02d}/{len(waypoints):02d}: "
                f"max_target_err={np.degrees(total_err):.2f}deg "
                f"q_deg={_format(np.degrees(q_last), 2)}"
            )

        q_final = read_joint_angles(robot)
        if q_final is None:
            raise RuntimeError("No joint feedback after final move_j")
        final_err = float(np.max(np.abs(q_final - target_q)))
        if final_err > tolerance:
            raise RuntimeError(
                f"Final start-pose error too large: {np.degrees(final_err):.2f}deg"
            )

        print(f"Reached start pose: final_err={np.degrees(final_err):.2f}deg")
        if args.hold_after > 0.0:
            print(f"Holding in J mode for {args.hold_after:.1f}s before exit")
            time.sleep(float(args.hold_after))
        if args.disable_on_exit:
            robot.disable()
            print("Disabled arm on exit")
        else:
            print("Leaving arm enabled in J mode for handoff to tracking script")
    finally:
        try:
            robot.disconnect()
        except Exception:
            pass


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="Connect to real arm and move.")
    parser.add_argument("--method", choices=("cpv", "move_j", "mit"), default="cpv")
    parser.add_argument("--channel", default="can0")
    parser.add_argument("--firmware", default="v112")
    parser.add_argument("--interface", default="socketcan")
    parser.add_argument("--freq", type=float, default=30.0, help="Command ramp frequency in Hz.")
    parser.add_argument("--control-freq", type=int, default=200, help="MIT loop frequency in Hz.")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--hold-after", type=float, default=2.0, help="Seconds to hold target before stopping and exiting.")
    parser.add_argument("--speed-percent", type=float, default=10.0, help="SDK move_j speed percent.")
    parser.add_argument("--move-j-waypoint-deg", type=float, default=5.0)
    parser.add_argument("--waypoint-timeout", type=float, default=20.0)
    parser.add_argument("--disable-on-exit", action="store_true")
    parser.add_argument("--target-tolerance-deg", type=float, default=3.0)
    parser.add_argument("--max-joint-vel", type=float, default=0.03, help="Max ramp speed per joint, rad/s.")
    parser.add_argument("--max-joint-step", type=float, default=0.001, help="Hard max command step per cycle, rad.")
    parser.add_argument("--max-follow-error-deg", type=float, default=8.0)
    parser.add_argument(
        "--cpv-signs",
        type=parse_float7,
        default=np.ones(7, dtype=np.float64),
        help="Per-joint sign multiplier before SDK move_cpv_vel(), comma-separated 7 values.",
    )
    parser.add_argument("--cpv-position-gain", type=float, default=1.0)
    parser.add_argument("--cpv-deadband-deg", type=float, default=0.2)
    parser.add_argument("--cpv-near-target-deg", type=float, default=8.0)
    parser.add_argument("--cpv-near-max-joint-vel", type=float, default=0.015)
    parser.add_argument("--cpv-min-joint-vel", type=float, default=0.0)
    parser.add_argument("--cpv-acc", type=float, default=0.50)
    parser.add_argument("--cpv-dcc", type=float, default=0.50)
    parser.add_argument("--cpv-cv", type=float, default=0.50)
    parser.add_argument("--cpv-prepare-delay-s", type=float, default=0.20)
    parser.add_argument("--torque-limit", type=float, default=14.0)
    parser.add_argument("--gravity-factor", type=float, default=1.0)
    parser.add_argument("--kp-scale", type=float, default=1.0)
    parser.add_argument("--kd-scale", type=float, default=1.0)
    parser.add_argument("--kp", default="", help="Override KP, comma-separated 7 values.")
    parser.add_argument("--kd", default="", help="Override KD, comma-separated 7 values.")
    parser.add_argument(
        "--start-q-deg",
        type=parse_deg7,
        default=DEFAULT_ARUCO_START_Q_DEG.copy(),
        help="Target start posture in degrees, comma-separated joint1..joint7.",
    )
    args = parser.parse_args()

    target_q = np.radians(np.asarray(args.start_q_deg, dtype=np.float64).ravel())
    kp = parse_vec7(args.kp, DEFAULT_KP * args.kp_scale)
    kd = parse_vec7(args.kd, DEFAULT_KD * args.kd_scale)
    max_step = min(float(args.max_joint_step), float(args.max_joint_vel) / float(args.freq))
    tolerance = np.radians(float(args.target_tolerance_deg))
    max_follow_error = np.radians(float(args.max_follow_error_deg))

    ok, reason = check_hard_limits(target_q)
    if not ok:
        raise SystemExit(f"Target start posture violates joint limits: {reason}")

    if not args.execute:
        print("Dry-run only. Add --execute to connect to real Nero arm.")
        print(f"method={args.method}")
        print(f"target_q_deg={_format(args.start_q_deg, 2)}")
        print(f"freq={args.freq:.1f}Hz max_joint_vel={args.max_joint_vel:.4f}rad/s max_step={max_step:.5f}rad")
        print(f"target_tolerance={args.target_tolerance_deg:.2f}deg hold_after={args.hold_after:.1f}s")
        print(f"kp={_format(kp, 3)}")
        print(f"kd={_format(kd, 3)}")
        return

    if args.method == "cpv":
        run_cpv(args, target_q)
        return

    if args.method == "move_j":
        run_move_j(args, target_q)
        return

    ctrl = NeroForceController(
        channel=args.channel,
        zero_gravity=False,
        gravity_factor=args.gravity_factor,
        control_freq=args.control_freq,
        torque_limit=args.torque_limit,
        interface=args.interface,
        firmware=args.firmware,
    )

    q_cmd = None
    try:
        ctrl.start()
        q_now = ctrl.get_joint_pos()
        if q_now is None:
            raise RuntimeError("No joint feedback after force controller start")
        q_now = np.asarray(q_now, dtype=np.float64).ravel()
        ok, reason = check_hard_limits(q_now)
        if not ok:
            raise RuntimeError(f"Current posture violates joint limits: {reason}")

        q_cmd = q_now.copy()
        print("=== Move To ArUco Start Pose ===")
        print(f"q_now_deg={_format(np.degrees(q_now), 2)}")
        print(f"target_q_deg={_format(args.start_q_deg, 2)}")
        print(f"max_joint_vel={args.max_joint_vel:.4f}rad/s max_step={max_step:.5f}rad")
        print("Ctrl+C stops the MIT loop and disables the arm.")

        period = 1.0 / float(args.freq)
        t_start = time.time()
        last_print = t_start
        done = False

        while time.time() - t_start < args.timeout:
            loop_start = time.time()
            q_now = ctrl.get_joint_pos()
            if q_now is None:
                time.sleep(0.02)
                continue
            q_now = np.asarray(q_now, dtype=np.float64).ravel()
            ok, reason = check_hard_limits(q_now)
            if not ok:
                raise RuntimeError(f"Current posture violates joint limits: {reason}")

            follow_err = np.max(np.abs(q_cmd - q_now))
            if follow_err > max_follow_error:
                raise RuntimeError(
                    f"Command/feedback error too large: {np.degrees(follow_err):.2f}deg"
                )

            prev_cmd = q_cmd.copy()
            q_cmd = step_towards(q_cmd, target_q, max_step)
            qdot_cmd = np.clip((q_cmd - prev_cmd) / period, -args.max_joint_vel, args.max_joint_vel)

            ctrl.command_joint_state(
                pos=q_cmd,
                vel=qdot_cmd,
                kp=kp,
                kd=kd,
                torques=np.zeros(7),
                gravity_enabled=True,
            )

            target_err = np.max(np.abs(q_now - target_q))
            cmd_err = np.max(np.abs(q_cmd - target_q))
            now = time.time()
            if now - last_print >= 1.0:
                print(
                    f"t={now - t_start:5.1f}s "
                    f"max_target_err={np.degrees(target_err):.2f}deg "
                    f"max_cmd_err={np.degrees(cmd_err):.2f}deg "
                    f"max_follow_err={np.degrees(follow_err):.2f}deg "
                    f"q_deg={_format(np.degrees(q_now), 2)} "
                    f"q_cmd_deg={_format(np.degrees(q_cmd), 2)}"
                )
                dbg = ctrl.get_debug_state()
                gravity = dbg.get("gravity_torque", np.zeros(7))
                total = dbg.get("total_torque", np.zeros(7))
                print(
                    f"  tau_g={_format(gravity, 2)} tau={_format(total, 2)} "
                    f"mit_loop_error={getattr(ctrl._loop, 'last_error', None)}"
                )
                last_print = now

            if target_err <= tolerance and cmd_err <= tolerance:
                done = True
                break

            sleep_s = period - (time.time() - loop_start)
            if sleep_s > 0.0:
                time.sleep(sleep_s)

        if not done:
            raise RuntimeError(f"Timed out before reaching start pose: timeout={args.timeout:.1f}s")

        print(f"Reached start pose within {args.target_tolerance_deg:.2f}deg")
        hold_until = time.time() + max(0.0, float(args.hold_after))
        while time.time() < hold_until:
            ctrl.command_joint_state(
                pos=target_q,
                vel=np.zeros(7),
                kp=kp,
                kd=kd,
                torques=np.zeros(7),
                gravity_enabled=True,
            )
            time.sleep(0.05)

    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as exc:
        print(f"\nStopped early: {type(exc).__name__}: {exc}")
        print(f"MIT loop error: {getattr(ctrl._loop, 'last_error', None)}")
    finally:
        qf = ctrl.get_joint_pos()
        if qf is not None:
            print(f"final_q_deg={_format(np.degrees(qf), 2)}")
        ctrl.stop()
        print("Stopped force_ctrl/MIT; script exited")


if __name__ == "__main__":
    main()
