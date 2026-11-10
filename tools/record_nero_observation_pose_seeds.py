#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

"""Record manually chosen Nero observation pose seeds.

Move the robot to a good observation pose manually, then press Enter here.
The script records current joint angles, flange pose, and TCP pose.  Optional
ArUco verification can capture one RGB frame and confirm the marker is visible.
"""

import argparse
import json
import os
from datetime import datetime

import cv2
import numpy as np

from cameras.realsense import capture_rgbd as capture_sdk_rgbd
from tools.real_nero_point_test import NeroArm


def _detect_aruco(bgr, dictionary_name, marker_id=None):
    if not hasattr(cv2, "aruco"):
        return False, "cv2.aruco unavailable"
    aruco = cv2.aruco
    dict_id = getattr(aruco, str(dictionary_name), None)
    if dict_id is None:
        return False, f"unknown aruco dictionary {dictionary_name}"
    dictionary = aruco.getPredefinedDictionary(dict_id)
    try:
        params = aruco.DetectorParameters()
    except AttributeError:
        params = aruco.DetectorParameters_create()
    try:
        detector = aruco.ArucoDetector(dictionary, params)
        corners, ids, _ = detector.detectMarkers(bgr)
    except AttributeError:
        corners, ids, _ = aruco.detectMarkers(bgr, dictionary, parameters=params)
    if ids is None or len(ids) == 0:
        return False, "no marker detected"
    ids_flat = np.asarray(ids, dtype=int).reshape(-1).tolist()
    if marker_id is not None and int(marker_id) not in ids_flat:
        return False, f"detected ids={ids_flat}, target id={int(marker_id)} absent"
    return True, f"detected ids={ids_flat}"


def _capture_aruco(args):
    from tools.verify_handeye_point import capture_ros_rgbd_frame

    bgr, _, _, out_dir = capture_sdk_rgbd(args)
    visible, msg = _detect_aruco(bgr, args.aruco_dict, args.aruco_id)
    return visible, msg, out_dir


def _write_log(path, payload):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def parse_args():
    p = argparse.ArgumentParser(description="Record manually selected Nero observation pose seeds.")
    p.add_argument("--save-dir", default="handeye_pose_seed_logs")
    p.add_argument("--count", type=int, default=0, help="Stop after this many records; 0 means manual q")
    p.add_argument("--nero-interface", default="socketcan")
    p.add_argument("--nero-channel", default="can0")
    p.add_argument("--nero-bitrate", type=int, default=1000000)
    p.add_argument("--nero-firmware", default="V111", choices=["DEFAULT", "V111", "V112"])
    p.add_argument("--nero-enable-timeout", type=float, default=5.0)
    p.add_argument("--nero-speed-percent", type=int, default=3)
    p.add_argument("--verify-aruco", action="store_true", help="Capture RGB-D and verify ArUco visibility for each seed")
    p.add_argument("--aruco-dict", default="DICT_ARUCO_ORIGINAL")
    p.add_argument("--aruco-id", type=int, default=582)
    p.add_argument("--orbbec-color-topic", default="/camera/color/image_raw")
    p.add_argument("--orbbec-depth-topic", default="/camera/depth/image_raw")
    p.add_argument("--orbbec-camera-info-topic", default="/camera/color/camera_info")
    p.add_argument("--orbbec-depth-scale", type=float, default=None)
    p.add_argument("--orbbec-timeout", type=float, default=5.0)
    p.add_argument("--orbbec-warmup-frames", type=int, default=5)
    from cameras.options import add_camera_arguments
    add_camera_arguments(p)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_path = os.path.join(args.save_dir, f"nero_observation_pose_seeds_{timestamp}.json")
    payload = {
        "timestamp": timestamp,
        "description": "Manual Nero observation pose seeds.",
        "records": [],
    }

    print("[nero] connecting/enabling once; this script only records current poses.")
    with NeroArm(
        args.nero_interface,
        args.nero_channel,
        args.nero_bitrate,
        args.nero_firmware,
        args.nero_enable_timeout,
        args.nero_speed_percent,
    ) as arm:
        print("[run] Move the robot to a good observation pose.")
        print("[run] Enter=record current pose, q=finish, or type a short name then Enter to record with that name.")
        while True:
            if int(args.count) > 0 and len(payload["records"]) >= int(args.count):
                break
            raw = input(f"[seed_{len(payload['records']) + 1:02d}] Enter/name=record, q=finish: ").strip()
            if raw.lower() == "q":
                break
            name = raw if raw else f"seed_{len(payload['records']) + 1:02d}"

            flange_pose = np.asarray(arm.get_flange_pose(), dtype=float).reshape(6)
            tcp_pose = np.asarray(arm.get_tcp_pose(), dtype=float).reshape(6)
            joints = np.asarray(arm.get_joint_angles(), dtype=float).reshape(7)

            aruco_visible = None
            aruco_msg = "not checked"
            capture_dir = None
            if bool(args.verify_aruco):
                try:
                    aruco_visible, aruco_msg, capture_dir = _capture_aruco(args)
                except Exception as exc:
                    aruco_visible = False
                    aruco_msg = f"{type(exc).__name__}: {exc}"

            record = {
                "index": len(payload["records"]) + 1,
                "name": name,
                "recorded_at": datetime.now().isoformat(timespec="seconds"),
                "joint_angles_rad": joints.tolist(),
                "joint_angles_deg": np.rad2deg(joints).tolist(),
                "flange_pose": flange_pose.tolist(),
                "tcp_pose": tcp_pose.tolist(),
                "aruco_visible": aruco_visible,
                "aruco_msg": aruco_msg,
                "capture_dir": capture_dir,
            }
            payload["records"].append(record)
            _write_log(log_path, payload)

            print(f"[{name}] joints_deg =", np.round(record["joint_angles_deg"], 3).tolist())
            print(f"[{name}] flange_pose =", np.round(flange_pose, 6).tolist())
            print(f"[{name}] tcp_pose    =", np.round(tcp_pose, 6).tolist())
            if args.verify_aruco:
                print(f"[{name}] aruco_visible={aruco_visible} ({aruco_msg})")
            print(f"[log] updated {log_path}")

    print(f"[done] records={len(payload['records'])} log={log_path}")


if __name__ == "__main__":
    main()
