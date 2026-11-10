#!/usr/bin/env python3
"""Minimal official-style pyAgxArm Nero probe.

This script intentionally avoids project controllers, gravity compensation,
and background MIT loops. It uses only the public pyAgxArm SDK calls shown in
the official Nero demo.
"""

import argparse
import time
import sys
from platform import system

import numpy as np

from pyAgxArm import create_agx_arm_config, AgxArmFactory, ArmModel, NeroFW


def enum_text(value):
    if value is None:
        return "None"
    name = getattr(value, "name", None)
    try:
        raw = int(value)
    except Exception:
        raw = value
    return f"{name}({raw})" if name else str(raw)


def array_text(values):
    return "[" + ", ".join(f"{float(v): .3f}" for v in values) + "]"


def joint_text(label, q):
    deg = np.degrees(q)
    text = "  ".join(f"J{i+1}:{d:7.2f}deg" for i, d in enumerate(deg))
    print(f"  [{label}] {text}")


def arm_status_summary(robot):
    try:
        st = robot.get_arm_status()
    except Exception as exc:
        return f"read failed ({exc})"
    if st is None:
        return "no feedback"
    msg = st.msg
    return (
        f"ctrl={enum_text(getattr(msg, 'ctrl_mode', None))}, "
        f"arm={enum_text(getattr(msg, 'arm_status', None))}, "
        f"mode={enum_text(getattr(msg, 'mode_feedback', None))}, "
        f"teach={enum_text(getattr(msg, 'teach_status', None))}, "
        f"motion={enum_text(getattr(msg, 'motion_status', None))}"
    )


def firmware_value(name):
    if name == "default":
        return NeroFW.DEFAULT
    attr = name.upper()
    return getattr(NeroFW, attr, name)


def create_config(args):
    return create_agx_arm_config(
        robot=ArmModel.NERO,
        firmeware_version=firmware_value(args.firmware),
        interface=args.interface,
        channel=args.channel,
    )


def read_joint_angles(robot, timeout=3.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        ja = robot.get_joint_angles()
        if ja is not None:
            return np.array(ja.msg, dtype=np.float64)
        time.sleep(0.02)
    raise RuntimeError("No joint feedback received")


def read_torques(robot):
    torques = np.zeros(7)
    for i in range(7):
        ms = robot.get_motor_states(i + 1)
        if ms is not None:
            torques[i] = ms.msg.torque
    return torques


def enable_robot(robot):
    deadline = time.time() + 5.0
    while not robot.enable():
        if time.time() > deadline:
            try:
                enable_list = robot.get_joints_enable_status_list()
            except Exception as exc:
                enable_list = f"read failed ({exc})"
            raise RuntimeError(
                "enable() timed out; "
                f"ArmStatus={arm_status_summary(robot)}; "
                f"enable_list={enable_list}"
            )
        time.sleep(0.01)


def install_frame_trace(robot):
    """Print SDK TX frames before sending them."""
    original_send_msg = robot._send_msg

    def traced_send_msg(msg):
        try:
            frame = robot._parser.pack(msg)
            if frame is None:
                print(f"  TX {type(msg).__name__}: no frame")
            else:
                data = bytes(frame.data).hex(" ")
                print(
                    f"  TX {type(msg).__name__}: "
                    f"id=0x{frame.arbitration_id:03X} data={data}"
                )
        except Exception as exc:
            print(f"  TX {type(msg).__name__}: trace failed ({exc})")
        return original_send_msg(msg)

    robot._send_msg = traced_send_msg


def run_move_j(robot, args):
    q_start = read_joint_angles(robot)
    q_target = q_start.copy()
    q_target[args.joint - 1] += args.delta

    robot.set_speed_percent(args.speed)
    robot.set_motion_mode(robot.OPTIONS.MOTION_MODE.J)
    time.sleep(0.3)
    print(f"  after set_motion_mode(J): {arm_status_summary(robot)}")
    joint_text("start", q_start)
    joint_text("target", q_target)

    robot.move_j(q_target.tolist())
    time.sleep(0.2)
    print(f"  after move_j(): {arm_status_summary(robot)}")
    observe(robot, q_start, args)


def run_mit_position(robot, args):
    q_start = read_joint_angles(robot)
    q_target = q_start.copy()
    q_target[args.joint - 1] += args.delta

    robot.set_motion_mode(robot.OPTIONS.MOTION_MODE.MIT)
    time.sleep(0.3)
    print(f"  after set_motion_mode(MIT): {arm_status_summary(robot)}")
    joint_text("start", q_start)
    joint_text("target", q_target)

    start = time.time()
    q_last = q_start.copy()
    while time.time() - start < args.duration:
        q_now = robot.get_joint_angles()
        if q_now is not None:
            q_last = np.array(q_now.msg, dtype=np.float64)
        for i in range(7):
            p_des = q_target[i] if i == args.joint - 1 else q_start[i]
            kp = args.kp if i == args.joint - 1 else 0.0
            kd = args.kd if i == args.joint - 1 else 0.0
            robot.move_mit(i + 1, p_des=float(p_des), v_des=0.0, kp=kp, kd=kd, t_ff=0.0)
        print_progress(time.time() - start, q_last, q_start, args, read_torques(robot))
        time.sleep(0.02)
    print_result(robot, q_last, q_start, args)


def run_mit_torque(robot, args):
    q_start = read_joint_angles(robot)

    robot.set_motion_mode(robot.OPTIONS.MOTION_MODE.MIT)
    time.sleep(0.3)
    print(f"  after set_motion_mode(MIT): {arm_status_summary(robot)}")
    joint_text("start", q_start)

    start = time.time()
    q_last = q_start.copy()
    while time.time() - start < args.duration:
        q_now = robot.get_joint_angles()
        if q_now is not None:
            q_last = np.array(q_now.msg, dtype=np.float64)
        for i in range(7):
            t_ff = args.torque if i == args.joint - 1 else 0.0
            robot.move_mit(i + 1, p_des=float(q_start[i]), v_des=0.0, kp=0.0, kd=0.0, t_ff=t_ff)
        print_progress(time.time() - start, q_last, q_start, args, read_torques(robot))
        time.sleep(0.02)
    print_result(robot, q_last, q_start, args)


def observe(robot, q_start, args):
    q_last = q_start.copy()
    start = time.time()
    while time.time() - start < args.duration:
        q_now = robot.get_joint_angles()
        if q_now is not None:
            q_last = np.array(q_now.msg, dtype=np.float64)
        print_progress(time.time() - start, q_last, q_start, args, read_torques(robot))
        time.sleep(0.05)
    print_result(robot, q_last, q_start, args)


def print_progress(elapsed, q_now, q_start, args, torques):
    delta_deg = np.degrees(q_now - q_start)
    print(
        f"\r  [{elapsed:5.1f}s] "
        f"J{args.joint} d={delta_deg[args.joint - 1]: .3f}deg  "
        f"max|d|={np.max(np.abs(delta_deg)): .3f}deg  "
        f"tau={array_text(torques)}  ",
        end="",
    )
    sys.stdout.flush()


def print_result(robot, q_last, q_start, args):
    print()
    joint_text("end", q_last)
    delta = q_last - q_start
    print(f"  J{args.joint} delta: {np.degrees(delta[args.joint - 1]):.3f}deg")
    print(f"  max joint delta: {np.max(np.abs(np.degrees(delta))):.3f}deg")
    print(f"  final ArmStatus: {arm_status_summary(robot)}")
    print(f"  final torques: {array_text(read_torques(robot))}")


def main():
    parser = argparse.ArgumentParser(description="Official-style Nero SDK probe")
    parser.add_argument("--execute", action="store_true", help="connect to the real robot")
    parser.add_argument("--channel", default="can0")
    parser.add_argument("--interface", default="socketcan")
    parser.add_argument("--firmware", choices=["default", "v111", "v112"], default="v111")
    parser.add_argument("--mode", choices=["move_j", "mit_position", "mit_torque"], default="move_j")
    parser.add_argument("--joint", type=int, choices=range(1, 8), default=6)
    parser.add_argument("--delta", type=float, default=0.03)
    parser.add_argument("--duration", type=float, default=3.0)
    parser.add_argument("--speed", type=int, default=10)
    parser.add_argument("--kp", type=float, default=8.0)
    parser.add_argument("--kd", type=float, default=0.8)
    parser.add_argument("--torque", type=float, default=0.8)
    parser.add_argument(
        "--skip-normal",
        action="store_true",
        help="match the official demo more closely: do not call set_normal_mode() before enable",
    )
    parser.add_argument(
        "--trace-frames",
        action="store_true",
        help="print SDK CAN frames before sending",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Official-style Nero SDK probe")
    print("=" * 60)
    print(f"  platform: {system()}")
    print(f"  config: {create_config(args)}")
    print(f"  mode: {args.mode}")
    print(f"  joint: J{args.joint}")
    print(f"  execute: {args.execute}")

    if not args.execute:
        print("\n  dry-run only. Add --execute to connect to the robot.")
        return 0

    input("\nPress Enter to connect and run the probe, Ctrl+C to cancel...")

    robot = AgxArmFactory.create_arm(create_config(args))
    if args.trace_frames:
        install_frame_trace(robot)
    robot.connect()
    try:
        print(f"  driver: {robot.__class__.__module__}.{robot.__class__.__name__}")
        print(f"  initial ArmStatus: {arm_status_summary(robot)}")
        robot.set_auto_set_motion_mode_enabled(True)
        if args.skip_normal:
            print("  skip set_normal_mode() before enable")
        else:
            robot.set_normal_mode()
            time.sleep(0.3)
            print(f"  after set_normal_mode(): {arm_status_summary(robot)}")
        enable_robot(robot)
        time.sleep(0.5)
        print(f"  after enable(): {arm_status_summary(robot)}")
        print(f"  enable list: {robot.get_joints_enable_status_list()}")

        if args.mode == "move_j":
            run_move_j(robot, args)
        elif args.mode == "mit_position":
            run_mit_position(robot, args)
        else:
            run_mit_torque(robot, args)

    finally:
        try:
            if args.mode.startswith("mit"):
                for i in range(7):
                    robot.move_mit(i + 1, p_des=0.0, v_des=0.0, kp=0.0, kd=0.0, t_ff=0.0)
                time.sleep(0.1)
        except Exception:
            pass
        try:
            deadline = time.time() + 2.0
            while not robot.disable():
                if time.time() > deadline:
                    break
                time.sleep(0.01)
        except Exception:
            pass
        print("  done; disabled")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
