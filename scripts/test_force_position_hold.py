#!/usr/bin/env python3
"""Force-control position hold test.

This is the hard gate before real visual servo execution.  It starts the
local NeroForceController in MIT position-hold mode and continuously holds
the current joint position with gravity compensation.

Default is dry-run.  Add --execute only when the arm workspace is clear and
ROS arm controllers are stopped.

Run:
    conda activate agilexrobot
    cd "/home/dpf/桌面/RealRobot_Test/Visual Servo"
    python scripts/test_force_position_hold.py
    python scripts/test_force_position_hold.py --execute --duration 30
"""

import argparse
import os
import sys
import time

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(SCRIPT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from control_backends.force_ctrl.controller import DEFAULT_KD, DEFAULT_KP, NeroForceController


def rad2deg(v):
    return round(float(v) * 57.2958, 2)


def parse_vec7(text, default):
    if not text:
        return default.copy()
    values = [float(item) for item in text.split(",")]
    if len(values) != 7:
        raise argparse.ArgumentTypeError(f"expected 7 comma-separated values, got {len(values)}")
    return np.asarray(values, dtype=np.float64)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="Connect to real arm and hold position.")
    parser.add_argument("--channel", default="can0")
    parser.add_argument("--firmware", default="v111")
    parser.add_argument("--interface", default="socketcan")
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--freq", type=int, default=200)
    parser.add_argument("--torque-limit", type=float, default=14.0)
    parser.add_argument("--gravity-factor", type=float, default=1.0)
    parser.add_argument("--kp-scale", type=float, default=1.0)
    parser.add_argument("--kd-scale", type=float, default=1.0)
    parser.add_argument("--kp", default="", help="Override KP, comma-separated 7 values.")
    parser.add_argument("--kd", default="", help="Override KD, comma-separated 7 values.")
    args = parser.parse_args()

    kp = parse_vec7(args.kp, DEFAULT_KP * args.kp_scale)
    kd = parse_vec7(args.kd, DEFAULT_KD * args.kd_scale)

    if not args.execute:
        print("Dry-run only. Add --execute to connect to real Nero arm.")
        print(f"channel={args.channel}, firmware={args.firmware}, freq={args.freq}Hz")
        print(f"duration={args.duration}s, torque_limit=±{args.torque_limit} N·m")
        print(f"gravity_factor={args.gravity_factor}")
        print(f"kp={np.round(kp, 3).tolist()}")
        print(f"kd={np.round(kd, 3).tolist()}")
        return

    ctrl = NeroForceController(
        channel=args.channel,
        zero_gravity=False,
        gravity_factor=args.gravity_factor,
        control_freq=args.freq,
        torque_limit=args.torque_limit,
        interface=args.interface,
        firmware=args.firmware,
    )

    q0 = None
    try:
        ctrl.start()
        q0 = ctrl.get_joint_pos()
        if q0 is None:
            raise RuntimeError("No joint feedback after force controller start")
        q0 = np.asarray(q0, dtype=np.float64).ravel()
        print("=== Force Position Hold ===")
        print(f"hold q0(deg)={np.round(np.degrees(q0), 2).tolist()}")
        print(f"kp={np.round(kp, 3).tolist()}")
        print(f"kd={np.round(kd, 3).tolist()}")
        print(f"duration={args.duration}s")
        print("Ctrl+C stops the MIT loop and disables the arm.")

        ctrl.command_joint_state(
            pos=q0,
            vel=np.zeros(7),
            kp=kp,
            kd=kd,
            torques=np.zeros(7),
            gravity_enabled=True,
        )

        t_start = time.time()
        last_print = t_start
        while time.time() - t_start < args.duration:
            state = ctrl.get_joint_state()
            now = time.time()
            if state is not None and now - last_print >= 1.0:
                q = np.asarray(state.pos, dtype=np.float64)
                err_deg = np.degrees(q - q0)
                dbg = ctrl.get_debug_state()
                gravity = dbg.get("gravity_torque", np.zeros(7))
                total = dbg.get("total_torque", np.zeros(7))
                print(
                    f"t={now - t_start:5.1f}s "
                    f"max_err={np.max(np.abs(err_deg)):.3f}deg "
                    f"q(deg)={[rad2deg(v) for v in q]} "
                    f"tau_g={np.round(gravity, 2).tolist()} "
                    f"tau={np.round(total, 2).tolist()}"
                )
                last_print = now
            time.sleep(0.02)

    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        if q0 is not None:
            qf = ctrl.get_joint_pos()
            if qf is not None:
                diff = np.degrees(np.asarray(qf, dtype=np.float64) - q0)
                print(f"final q(deg)={np.round(np.degrees(qf), 2).tolist()}")
                print(f"total diff(deg)={np.round(diff, 3).tolist()}")
        ctrl.stop()
        print("Stopped")


if __name__ == "__main__":
    main()
