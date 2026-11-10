#!/usr/bin/env python3
"""Dry-run one ArUco Jacobian-servo command.

This script does not connect to the robot or camera.  It validates the
software chain:

    marker center / normal -> JacobianPositionServo -> BackendCommand
    -> SdkCpvVelocityBackend(dry_run=True)

Run:
    conda activate agilexrobot
    python scripts/test_aruco_jacobian_servo_dryrun.py
"""

import argparse
import logging
import os
import sys

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(SCRIPT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from control_backends.base import BackendCommand
from control_backends.sdk_cpv_velocity_backend import SdkCpvVelocityBackend
from servo.jacobian_position_servo import JacobianPositionServo, ServoParams


def _vec3(text):
    values = [float(item) for item in text.split(",")]
    if len(values) != 3:
        raise argparse.ArgumentTypeError("expected three comma-separated floats")
    return np.asarray(values, dtype=np.float64)


def _vec7(text):
    values = [float(item) for item in text.split(",")]
    if len(values) != 7:
        raise argparse.ArgumentTypeError("expected seven comma-separated floats")
    return np.asarray(values, dtype=np.float64)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--q",
        type=_vec7,
        default=np.array([0.0, 0.3, 0.0, 0.5, 0.0, 0.2, 0.0]),
        help="Current joint angles, comma-separated radians.",
    )
    parser.add_argument(
        "--marker-center",
        type=_vec3,
        default=np.array([0.25, 0.0, 0.45]),
        help="Marker center in base frame, comma-separated meters.",
    )
    parser.add_argument(
        "--marker-normal",
        type=_vec3,
        default=np.array([-1.0, 0.0, 0.0]),
        help="Marker normal in base frame.",
    )
    parser.add_argument("--desired-distance", type=float, default=0.15)
    parser.add_argument("--channel", default="can0")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    params = ServoParams(desired_distance=args.desired_distance)
    servo = JacobianPositionServo(params=params)
    result = servo.update(
        q_now=args.q,
        target_position=args.marker_center,
        target_normal=args.marker_normal,
    )

    cmd = BackendCommand(
        q_cmd=args.q.copy(),
        qdot_cmd=result["qdot"],
        mode="velocity",
    )
    backend = SdkCpvVelocityBackend(dry_run=True, channel=args.channel)
    backend.connect()
    backend.send(cmd)

    print("\n=== ArUco Jacobian Servo Dry-Run ===")
    print(f"q_now      = {np.round(args.q, 4).tolist()}")
    print(f"tcp_now    = {np.round(result['tcp_now'], 4).tolist()}")
    print(f"target_tcp = {np.round(result['target_tcp'], 4).tolist()}")
    print(f"error      = {np.round(result['error'], 4).tolist()}")
    print(f"v_base     = {np.round(result['v_base'], 4).tolist()}")
    print(f"qdot       = {np.round(result['qdot'], 5).tolist()}")
    print(f"q_cmd      = {np.round(result['q_cmd'], 5).tolist()}")
    print(f"twist_scale= {result['twist_scale']:.3f}")


if __name__ == "__main__":
    main()
