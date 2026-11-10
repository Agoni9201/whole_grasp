#!/usr/bin/env python3
"""Verify eye-in-hand calibration by projecting clicked depth pixels to robot base.

Example:
  python verify_handeye_point.py \
    --rgb color.png --depth depth.npy \
    --fx 615 --fy 615 --cx 320 --cy 240 \
    --flange-pose 0.30 0.10 0.85 3.14 0.0 1.57 \
    --measured-base 0.35 0.10 0.745

If --u/--v are omitted, click one point in the RGB image and press Enter.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R

from cameras.depth_fusion import median_valid_depth_frames


DEFAULT_CALIB_POS = [0.07861645260509609, -0.08356132565557191, 0.0013613808770194039]
DEFAULT_CALIB_QUAT_XYZW = [-0.12203699806428801, 0.6971853675253987, 0.1299244749511357, 0.6943768178882441]


@dataclass
class Intrinsics:
    fx: float
    fy: float
    cx: float
    cy: float


def make_T(R_mat, t_xyz):
    T = np.eye(4, dtype=float)
    T[:3, :3] = np.asarray(R_mat, dtype=float).reshape(3, 3)
    T[:3, 3] = np.asarray(t_xyz, dtype=float).reshape(3)
    return T


def inv_T(T):
    T = np.asarray(T, dtype=float).reshape(4, 4)
    out = np.eye(4, dtype=float)
    out[:3, :3] = T[:3, :3].T
    out[:3, 3] = -T[:3, :3].T @ T[:3, 3]
    return out


def load_calibration(path):
    if path is None:
        return np.asarray(DEFAULT_CALIB_POS, dtype=float), np.asarray(DEFAULT_CALIB_QUAT_XYZW, dtype=float)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # Accept the standalone ArUco calibrator schema as well as the legacy
    # grasp schema. Both store meters and xyzw quaternions.
    pos_value = data.get("position", data.get("translation_m"))
    quat_value = data.get("orientation", data.get("quaternion_xyzw"))
    if pos_value is None or quat_value is None:
        raise KeyError("calibration JSON requires position/orientation or translation_m/quaternion_xyzw")
    pos = np.asarray(pos_value, dtype=float).reshape(3)
    quat = np.asarray(quat_value, dtype=float).reshape(4)
    return pos, quat


def load_depth(path, depth_scale):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        depth = np.load(path)
    elif ext == ".npz":
        obj = np.load(path)
        key = "depth" if "depth" in obj else obj.files[0]
        depth = obj[key]
    else:
        depth = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if depth is None:
            raise FileNotFoundError(path)

    depth = np.asarray(depth)
    if depth.ndim == 3:
        depth = depth[:, :, 0]

    if depth_scale is None:
        scale = 0.001 if np.issubdtype(depth.dtype, np.integer) else 1.0
    else:
        scale = float(depth_scale)
    return depth.astype(np.float32) * scale


def capture_realsense_frame(save_dir, width, height, fps, warmup_frames):
    try:
        import pyrealsense2 as rs
    except ImportError as e:
        raise ImportError("pyrealsense2 is required for --live-realsense") from e

    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = os.path.join(save_dir, timestamp)
    os.makedirs(out_dir, exist_ok=True)

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, int(width), int(height), rs.format.bgr8, int(fps))
    config.enable_stream(rs.stream.depth, int(width), int(height), rs.format.z16, int(fps))

    try:
        profile = pipeline.start(config)
        align = rs.align(rs.stream.color)
        for _ in range(max(int(warmup_frames), 0)):
            pipeline.wait_for_frames()

        frames = align.process(pipeline.wait_for_frames())
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        if not color_frame or not depth_frame:
            raise RuntimeError("Failed to get aligned RealSense color/depth frames")

        rgb = np.asanyarray(color_frame.get_data())
        depth_raw = np.asanyarray(depth_frame.get_data())
        depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
        depth_m = depth_raw.astype(np.float32) * float(depth_scale)

        intr_rs = color_frame.profile.as_video_stream_profile().intrinsics
        intr = Intrinsics(float(intr_rs.fx), float(intr_rs.fy), float(intr_rs.ppx), float(intr_rs.ppy))

        rgb_path = os.path.join(out_dir, "color.png")
        depth_npy_path = os.path.join(out_dir, "depth_m.npy")
        depth_png_path = os.path.join(out_dir, "depth_mm.png")
        meta_path = os.path.join(out_dir, "capture_meta.json")

        cv2.imwrite(rgb_path, rgb)
        np.save(depth_npy_path, depth_m)
        cv2.imwrite(depth_png_path, np.clip(depth_m * 1000.0, 0, 65535).astype(np.uint16))
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "rgb": rgb_path,
                    "depth_m_npy": depth_npy_path,
                    "depth_mm_png": depth_png_path,
                    "intrinsics": {"fx": intr.fx, "fy": intr.fy, "cx": intr.cx, "cy": intr.cy},
                    "depth_scale": float(depth_scale),
                    "width": int(intr_rs.width),
                    "height": int(intr_rs.height),
                    "timestamp": timestamp,
                },
                f,
                indent=2,
            )
        print(f"[capture] saved RGB-D frame to {out_dir}")
        return rgb, depth_m, intr, out_dir
    finally:
        pipeline.stop()


def _ros_image_to_cv(msg):
    encoding = msg.encoding.lower()
    if encoding in ("bgr8", "rgb8"):
        arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.step)
        channels = 3
        img = arr[:, : msg.width * channels].reshape(msg.height, msg.width, channels)
        if encoding == "rgb8":
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img.copy()
    if encoding in ("bgra8", "rgba8"):
        arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.step)
        channels = 4
        img = arr[:, : msg.width * channels].reshape(msg.height, msg.width, channels)
        code = cv2.COLOR_RGBA2BGR if encoding == "rgba8" else cv2.COLOR_BGRA2BGR
        return cv2.cvtColor(img, code)
    if encoding in ("mono8", "8uc1"):
        arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.step)
        return arr[:, : msg.width].copy()
    if encoding in ("16uc1", "mono16"):
        dtype = ">u2" if msg.is_bigendian else "<u2"
        arr = np.frombuffer(msg.data, dtype=np.dtype(dtype)).reshape(msg.height, msg.step // 2)
        return arr[:, : msg.width].copy()
    if encoding == "32fc1":
        dtype = ">f4" if msg.is_bigendian else "<f4"
        arr = np.frombuffer(msg.data, dtype=np.dtype(dtype)).reshape(msg.height, msg.step // 4)
        return arr[:, : msg.width].copy()
    raise ValueError(f"Unsupported ROS image encoding: {msg.encoding}")


def capture_ros_rgbd_frame(*args, **kwargs):
    """Compatibility signature used by older tools; transport is D435 SDK only."""
    from cameras.realsense import capture_legacy
    return capture_legacy(*args, **kwargs)


def _response_msg(resp):
    if hasattr(resp, "msg"):
        return resp.msg
    return resp


def read_nero_flange_pose(interface, channel, bitrate, firmware, enable_timeout_s, speed_percent):
    try:
        from pyAgxArm import AgxArmFactory, ArmModel, create_agx_arm_config
        from pyAgxArm import NeroFW
    except ImportError as e:
        raise ImportError("pyAgxArm is required for --read-nero-flange") from e

    fw = getattr(NeroFW, str(firmware))
    cfg = create_agx_arm_config(
        robot=ArmModel.NERO,
        firmeware_version=fw,
        interface=str(interface),
        channel=str(channel),
        bitrate=int(bitrate),
    )
    arm = AgxArmFactory.create_arm(cfg)
    try:
        arm.connect()
        deadline = time.time() + float(enable_timeout_s)
        while True:
            ok = bool(arm.enable())
            if ok:
                break
            if time.time() > deadline:
                raise TimeoutError("Timed out waiting for Nero enable()")
            time.sleep(0.05)
        if speed_percent is not None:
            arm.set_speed_percent(int(speed_percent))
        pose = np.asarray(_response_msg(arm.get_flange_pose()), dtype=float).reshape(-1)
        if pose.size < 6:
            raise RuntimeError(f"get_flange_pose returned too few values: {pose.tolist()}")
        pose = pose[:6]
        print("[nero] flange_pose =", np.round(pose, 6).tolist())
        return pose
    finally:
        try:
            arm.disconnect()
        except Exception:
            pass


def read_ros_pose_stamped(topic, timeout_s, label):
    raise RuntimeError("ROS pose input removed. Use --read-nero-flange or --flange-pose.")


def robust_depth_at(depth_m, u, v, radius):
    h, w = depth_m.shape[:2]
    u = int(round(u))
    v = int(round(v))
    x0, x1 = max(0, u - radius), min(w, u + radius + 1)
    y0, y1 = max(0, v - radius), min(h, v + radius + 1)
    patch = depth_m[y0:y1, x0:x1]
    valid = patch[np.isfinite(patch) & (patch > 0)]
    if valid.size == 0:
        raise ValueError(f"No valid depth around pixel ({u}, {v}) with radius={radius}")
    return float(np.median(valid))


def pixel_to_camera(u, v, depth_m, intr):
    z = float(depth_m)
    x = (float(u) - intr.cx) * z / intr.fx
    y = (float(v) - intr.cy) * z / intr.fy
    return np.array([x, y, z], dtype=float)


def transform_point(T, p):
    return (np.asarray(T, dtype=float).reshape(4, 4) @ np.r_[np.asarray(p, dtype=float).reshape(3), 1.0])[:3]


def read_clicked_pixel(rgb):
    clicked = []

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            clicked[:] = [(x, y)]
            vis = rgb.copy()
            cv2.drawMarker(vis, (x, y), (0, 0, 255), cv2.MARKER_CROSS, 22, 2)
            cv2.imshow("click target point, then press Enter", vis)

    cv2.imshow("click target point, then press Enter", rgb)
    cv2.setMouseCallback("click target point, then press Enter", on_mouse)
    while True:
        key = cv2.waitKey(20) & 0xFF
        if key in (13, 10) and clicked:
            break
        if key in (27, ord("q")):
            raise KeyboardInterrupt("cancelled")
    cv2.destroyAllWindows()
    return clicked[0]


def print_candidate(name, T_base_flange, T_hand_cam, P_camera, measured_base):
    P_base = transform_point(T_base_flange @ T_hand_cam, P_camera)
    print(f"\n[{name}]")
    print("P_base_pred =", np.round(P_base, 6).tolist())
    print("base_distance_m =", round(float(np.linalg.norm(P_base)), 6))
    if measured_base is not None:
        err = P_base - measured_base
        print("err          =", np.round(err, 6).tolist())
        print("err_norm_m   =", round(float(np.linalg.norm(err)), 6))


def parse_args():
    p = argparse.ArgumentParser(description="Verify Nero eye-in-hand calibration with one clicked RGB-D point.")
    p.add_argument("--live-realsense", action="store_true", help="Capture and save one aligned RGB-D frame from RealSense")
    p.add_argument("--live-orbbec", action="store_true", help="Deprecated alias for D435 SDK capture (330 disabled)")
    p.add_argument("--save-dir", default="handeye_verify_captures", help="Directory for live camera captures")
    p.add_argument("--rs-width", type=int, default=1280)
    p.add_argument("--rs-height", type=int, default=720)
    p.add_argument("--rs-fps", type=int, default=30)
    p.add_argument("--rs-warmup-frames", type=int, default=30)
    p.add_argument("--orbbec-color-topic", default="/camera/color/image_raw")
    p.add_argument("--orbbec-depth-topic", default="/camera/depth/image_raw")
    p.add_argument("--orbbec-camera-info-topic", default="/camera/color/camera_info")
    p.add_argument("--orbbec-depth-scale", type=float, default=None, help="Depth scale to meters for ROS depth. Auto: int depth=0.001, float depth=1.0")
    p.add_argument("--orbbec-timeout", type=float, default=5.0)
    p.add_argument("--orbbec-warmup-frames", type=int, default=5)
    p.add_argument("--orbbec-depth-median-frames", type=int, default=1)
    p.add_argument("--orbbec-depth-median-min-valid", type=int, default=1)
    p.add_argument("--rgb", default=None, help="RGB/BGR image path for clicking, e.g. color.png")
    p.add_argument("--depth", default=None, help="Depth image path: .npy/.npz meters, or png/tiff usually millimeters")
    p.add_argument("--depth-scale", type=float, default=None, help="Depth scale to meters. Auto: int depth=0.001, float depth=1.0")
    p.add_argument("--fx", type=float, default=None)
    p.add_argument("--fy", type=float, default=None)
    p.add_argument("--cx", type=float, default=None)
    p.add_argument("--cy", type=float, default=None)
    p.add_argument("--calib-json", default=None, help="Hand-eye calibration json with position and orientation")
    p.add_argument("--read-nero-flange", action="store_true", help="Read current flange pose from Nero using pyAgxArm; SDK failure is fatal; no ROS fallback")
    p.set_defaults(read_nero_tcp_ros=False)
    p.add_argument("--nero-ros-pose-topic", default="/feedback/tcp_pose", help="ROS PoseStamped topic for --read-nero-tcp-ros or pyAgxArm fallback")
    p.add_argument("--nero-ros-pose-timeout", type=float, default=5.0)
    p.add_argument("--nero-interface", default="socketcan")
    p.add_argument("--nero-channel", default="can0")
    p.add_argument("--nero-bitrate", type=int, default=1000000)
    p.add_argument("--nero-firmware", default="V112", choices=["DEFAULT", "V111", "V112"])
    p.add_argument("--nero-enable-timeout", type=float, default=5.0)
    p.add_argument("--nero-speed-percent", type=int, default=20)
    p.add_argument("--flange-pose", type=float, nargs=6, default=None, metavar=("X", "Y", "Z", "ROLL", "PITCH", "YAW"), help="Current get_flange_pose(): x y z roll pitch yaw, meters/radians")
    p.add_argument("--measured-base", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"), help="Optional known target point in robot base, meters, used only to compute error")
    p.add_argument("--u", type=float, default=None, help="Pixel u. If omitted, click in image.")
    p.add_argument("--v", type=float, default=None, help="Pixel v. If omitted, click in image.")
    p.add_argument("--depth-radius", type=int, default=2, help="Median depth radius around clicked pixel")
    p.add_argument("--flange-rpy-order", default="xyz", help="Euler order for flange rpy from Nero API; default xyz")
    return p.parse_args()


def main():
    args = parse_args()

    if args.live_realsense and args.live_orbbec:
        raise ValueError("Use only one live camera mode: --live-realsense or --live-orbbec")

    if args.live_realsense:
        rgb, depth_m, intr, out_dir = capture_realsense_frame(
            args.save_dir,
            args.rs_width,
            args.rs_height,
            args.rs_fps,
            args.rs_warmup_frames,
        )
    elif args.live_orbbec:
        from cameras.realsense import capture_rgbd
        rgb, depth_m, intr, out_dir = capture_rgbd(args)
    else:
        if args.rgb is None or args.depth is None:
            raise ValueError("Provide --rgb and --depth, or use --live-realsense/--live-orbbec")
        if None in (args.fx, args.fy, args.cx, args.cy):
            raise ValueError("Provide --fx --fy --cx --cy when not using a live camera mode")
        intr = Intrinsics(args.fx, args.fy, args.cx, args.cy)
        rgb = cv2.imread(args.rgb, cv2.IMREAD_COLOR)
        if rgb is None:
            raise FileNotFoundError(args.rgb)
        depth_m = load_depth(args.depth, args.depth_scale)
        out_dir = None

    if depth_m.shape[:2] != rgb.shape[:2]:
        print(f"[warn] rgb shape {rgb.shape[:2]} != depth shape {depth_m.shape[:2]}; make sure they are aligned.")

    if args.read_nero_flange:
        try:
            flange_pose = read_nero_flange_pose(
                args.nero_interface,
                args.nero_channel,
                args.nero_bitrate,
                args.nero_firmware,
                args.nero_enable_timeout,
                args.nero_speed_percent,
            )
        except ImportError as e:
            # print(f"[warn] {e}; falling back to ROS PoseStamped topic {args.nero_ros_pose_topic}")
            # flange_pose = read_ros_pose_stamped(args.nero_ros_pose_topic, args.nero_ros_pose_timeout, "tcp_pose")
            raise RuntimeError("NERO SDK unavailable; ROS fallback disabled") from e
    elif args.read_nero_tcp_ros:
        flange_pose = read_ros_pose_stamped(args.nero_ros_pose_topic, args.nero_ros_pose_timeout, "tcp_pose")
    else:
        if args.flange_pose is None:
            raise ValueError("Provide --flange-pose, or use --read-nero-flange/--read-nero-tcp-ros")
        flange_pose = np.asarray(args.flange_pose, dtype=float)

    if args.u is None or args.v is None:
        u, v = read_clicked_pixel(rgb)
    else:
        u, v = float(args.u), float(args.v)

    depth = robust_depth_at(depth_m, u, v, args.depth_radius)
    P_camera = pixel_to_camera(u, v, depth, intr)

    flange_xyz = np.asarray(flange_pose[:3], dtype=float)
    flange_rpy = np.asarray(flange_pose[3:], dtype=float)
    T_base_flange = make_T(R.from_euler(args.flange_rpy_order, flange_rpy).as_matrix(), flange_xyz)

    calib_pos, calib_quat = load_calibration(args.calib_json)
    measured_base = None if args.measured_base is None else np.asarray(args.measured_base, dtype=float).reshape(3)

    print("pixel        =", [float(u), float(v)])
    print("depth_m      =", round(depth, 6))
    print("P_camera     =", np.round(P_camera, 6).tolist())
    print("flange_xyz   =", np.round(flange_xyz, 6).tolist())
    print("flange_rpy   =", np.round(flange_rpy, 6).tolist())
    if measured_base is not None:
        print("measured_base=", np.round(measured_base, 6).tolist())
    print("intrinsics   =", {"fx": intr.fx, "fy": intr.fy, "cx": intr.cx, "cy": intr.cy})
    if out_dir is not None:
        with open(os.path.join(out_dir, "verify_input.json"), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "pixel": [float(u), float(v)],
                    "depth_m": float(depth),
                    "P_camera": P_camera.tolist(),
                    "flange_pose": flange_pose.tolist(),
                    "measured_base": None if measured_base is None else measured_base.tolist(),
                    "intrinsics": {"fx": intr.fx, "fy": intr.fy, "cx": intr.cx, "cy": intr.cy},
                },
                f,
                indent=2,
            )

    # Try common conventions. If measured_base is provided, the correct one should have the smallest error.
    T_xyzw = make_T(R.from_quat(calib_quat).as_matrix(), calib_pos)
    q_wxyz_to_xyzw = np.array([calib_quat[1], calib_quat[2], calib_quat[3], calib_quat[0]], dtype=float)
    T_wxyz = make_T(R.from_quat(q_wxyz_to_xyzw).as_matrix(), calib_pos)

    print_candidate("calib is T_flange_camera, quat xyzw", T_base_flange, T_xyzw, P_camera, measured_base)
    print_candidate("calib is T_camera_flange, quat xyzw, using inverse", T_base_flange, inv_T(T_xyzw), P_camera, measured_base)
    print_candidate("calib is T_flange_camera, quat wxyz", T_base_flange, T_wxyz, P_camera, measured_base)
    print_candidate("calib is T_camera_flange, quat wxyz, using inverse", T_base_flange, inv_T(T_wxyz), P_camera, measured_base)


if __name__ == "__main__":
    main()
