#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
"""SDK RGB-D click-to-base point test; optional supervised cross compensation."""

import argparse
import hashlib
import json
import os
import time
from datetime import datetime

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R

from tools.verify_handeye_point import (
    Intrinsics,
    _ros_image_to_cv,
    capture_ros_rgbd_frame,
    inv_T,
    load_calibration,
    load_depth,
    make_T,
    pixel_to_camera,
    read_clicked_pixel,
    robust_depth_at,
    transform_point,
)


DEFAULT_INITIAL_POSE_MM_DEG = [-100.06, 278.0, 555.400, -105.0, 67.0, 102.035]
DEFAULT_INITIAL_JOINT_DEG = [3.546, -0.494, 4.357, 69.346, 2.379, -13.027, 75.000]
DEFAULT_TCP_OFFSET = [0.165, 0.0, -0.025, 0.0, 0.0, 0.0]


def _response_msg(resp):
    return resp.msg if hasattr(resp, "msg") else resp


class NeroArm:
    def __init__(self, interface, channel, bitrate, firmware, enable_timeout_s, speed_percent):
        from pyAgxArm import AgxArmFactory, ArmModel, NeroFW, create_agx_arm_config

        fw = getattr(NeroFW, str(firmware))
        cfg = create_agx_arm_config(
            robot=ArmModel.NERO,
            firmeware_version=fw,
            interface=str(interface),
            channel=str(channel),
            bitrate=int(bitrate),
        )
        self.arm = AgxArmFactory.create_arm(cfg)
        self.enable_timeout_s = float(enable_timeout_s)
        self.speed_percent = speed_percent

    def __enter__(self):
        self.arm.connect()
        deadline = time.time() + self.enable_timeout_s
        while True:
            if bool(self.arm.enable()):
                break
            if time.time() > deadline:
                raise TimeoutError("Timed out waiting for Nero enable()")
            time.sleep(0.05)
        if self.speed_percent is not None:
            self.arm.set_speed_percent(int(self.speed_percent))
        return self

    def __exit__(self, exc_type, exc, tb):
        try:
            self.arm.disconnect()
        except Exception:
            pass

    def get_flange_pose(self):
        pose = np.asarray(_response_msg(self.arm.get_flange_pose()), dtype=float).reshape(-1)
        if pose.size < 6:
            raise RuntimeError(f"get_flange_pose returned too few values: {pose.tolist()}")
        return pose[:6]

    def get_joint_angles(self):
        joints = np.asarray(_response_msg(self.arm.get_joint_angles()), dtype=float).reshape(-1)
        if joints.size < 7:
            raise RuntimeError(f"get_joint_angles returned too few values: {joints.tolist()}")
        return joints[:7]

    def get_tcp_pose(self):
        pose = np.asarray(_response_msg(self.arm.get_tcp_pose()), dtype=float).reshape(-1)
        if pose.size < 6:
            raise RuntimeError(f"get_tcp_pose returned too few values: {pose.tolist()}")
        return pose[:6]

    def set_tcp_offset(self, offset):
        return self.arm.set_tcp_offset([float(x) for x in np.asarray(offset, dtype=float).reshape(6)])

    def get_tcp2flange_pose(self, tcp_pose):
        pose = np.asarray(_response_msg(self.arm.get_tcp2flange_pose([float(x) for x in np.asarray(tcp_pose, dtype=float).reshape(6)])), dtype=float).reshape(-1)
        if pose.size < 6:
            raise RuntimeError(f"get_tcp2flange_pose returned too few values: {pose.tolist()}")
        return pose[:6]

    def move_p(self, pose):
        return self.arm.move_p([float(x) for x in np.asarray(pose, dtype=float).reshape(6)])

    def move_j(self, joints):
        return self.arm.move_j([float(x) for x in np.asarray(joints, dtype=float).reshape(7)])

    def stop(self):
        return self.arm.electronic_emergency_stop()


def handeye_transform(calib_json, convention):
    pos, quat = load_calibration(calib_json)
    if convention.endswith("wxyz"):
        quat = np.array([quat[1], quat[2], quat[3], quat[0]], dtype=float)
    T = make_T(R.from_quat(quat).as_matrix(), pos)
    if convention.startswith("camera_flange"):
        T = inv_T(T)
    return T


def load_offline_rgbd(args):
    if None in (args.fx, args.fy, args.cx, args.cy):
        raise ValueError("Offline mode requires --fx --fy --cx --cy")
    rgb = cv2.imread(args.rgb, cv2.IMREAD_COLOR)
    if rgb is None:
        raise FileNotFoundError(args.rgb)
    depth_m = load_depth(args.depth, args.depth_scale)
    intr = Intrinsics(args.fx, args.fy, args.cx, args.cy)
    return rgb, depth_m, intr, None


def workspace_error(point, args):
    x, y, z = np.asarray(point, dtype=float).reshape(3)
    if not (args.min_x <= x <= args.max_x):
        return f"x={x:.4f} outside [{args.min_x}, {args.max_x}]"
    if not (args.min_y <= y <= args.max_y):
        return f"y={y:.4f} outside [{args.min_y}, {args.max_y}]"
    if not (args.min_z <= z <= args.max_z):
        return f"z={z:.4f} outside [{args.min_z}, {args.max_z}]"
    return None


def assert_workspace(point, args):
    err = workspace_error(point, args)
    if err is not None:
        raise ValueError(f"Target {err}")


def wrap_angle_error(a, b):
    return (np.asarray(a, dtype=float) - np.asarray(b, dtype=float) + np.pi) % (2.0 * np.pi) - np.pi


def pose_error(current_pose, target_pose):
    current_pose = np.asarray(current_pose, dtype=float).reshape(6)
    target_pose = np.asarray(target_pose, dtype=float).reshape(6)
    pos_err = float(np.linalg.norm(current_pose[:3] - target_pose[:3]))
    rpy_err = float(np.linalg.norm(wrap_angle_error(current_pose[3:], target_pose[3:])))
    return pos_err, rpy_err


def wait_until_pose_reached(nero, target_pose, args, label):
    deadline = time.time() + float(args.move_wait_timeout)
    last_pose = None
    last_pos_err = float("inf")
    last_rpy_err = float("inf")
    while time.time() < deadline:
        last_pose = nero.get_flange_pose()
        last_pos_err, last_rpy_err = pose_error(last_pose, target_pose)
        if last_pos_err <= float(args.move_pos_tol) and last_rpy_err <= float(args.move_rpy_tol):
            print(
                f"[{label}] reached: pos_err={last_pos_err:.4f}m "
                f"rpy_err={np.rad2deg(last_rpy_err):.2f}deg"
            )
            return True, last_pose, last_pos_err, last_rpy_err
        time.sleep(float(args.move_poll_interval))

    print(
        f"[{label}][warn] not reached before timeout: "
        f"pos_err={last_pos_err:.4f}m rpy_err={np.rad2deg(last_rpy_err):.2f}deg"
    )
    if last_pose is not None:
        print(f"[{label}][warn] current_pose =", np.round(last_pose, 6).tolist())
        print(f"[{label}][warn] target_pose  =", np.round(np.asarray(target_pose, dtype=float), 6).tolist())
    return False, last_pose, last_pos_err, last_rpy_err


def wait_until_joints_reached(nero, target_joints, args, label):
    target_joints = np.asarray(target_joints, dtype=float).reshape(7)
    deadline = time.time() + float(args.move_wait_timeout)
    last_joints = None
    last_err = float("inf")
    while time.time() < deadline:
        last_joints = nero.get_joint_angles()
        last_err = float(np.linalg.norm(wrap_angle_error(last_joints, target_joints)))
        if last_err <= float(args.move_joint_tol):
            print(f"[{label}] reached: joint_err={np.rad2deg(last_err):.2f}deg")
            return True, last_joints, last_err
        time.sleep(float(args.move_poll_interval))

    print(f"[{label}][warn] joints not reached before timeout: joint_err={np.rad2deg(last_err):.2f}deg")
    if last_joints is not None:
        print(f"[{label}][warn] current_joints_deg =", np.round(np.rad2deg(last_joints), 3).tolist())
        print(f"[{label}][warn] target_joints_deg  =", np.round(np.rad2deg(target_joints), 3).tolist())
    return False, last_joints, last_err


def write_result(out_dir, data):
    if out_dir is None:
        out_dir = os.path.join("real_nero_point_test_logs", datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "point_test_result.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"[log] wrote {path}")


def capture_rgbd(args):
    if args.rgb is not None or args.depth is not None:
        if args.rgb is None or args.depth is None:
            raise ValueError("Provide both --rgb and --depth for offline mode")
        return load_offline_rgbd(args)
    from cameras.realsense import capture_rgbd as capture_sdk
    return capture_sdk(args)



def _pointcloud2_field_dtype(datatype):
    # sensor_msgs/PointField constants.
    mapping = {
        1: np.int8,
        2: np.uint8,
        3: np.int16,
        4: np.uint16,
        5: np.int32,
        6: np.uint32,
        7: np.float32,
        8: np.float64,
    }
    if datatype not in mapping:
        raise ValueError(f"Unsupported PointCloud2 field datatype: {datatype}")
    return mapping[datatype]


def pointcloud2_xyz_at(msg, u, v):
    if int(msg.height) <= 1:
        raise ValueError("PointCloud2 is unorganized; cannot index by image pixel")
    u = int(round(u))
    v = int(round(v))
    if not (0 <= u < int(msg.width) and 0 <= v < int(msg.height)):
        raise ValueError(f"Pixel ({u}, {v}) outside point cloud size {msg.width}x{msg.height}")

    fields = {field.name: field for field in msg.fields}
    for name in ("x", "y", "z"):
        if name not in fields:
            raise ValueError(f"PointCloud2 missing field {name!r}")

    base = int(v) * int(msg.row_step) + int(u) * int(msg.point_step)
    endian = ">" if msg.is_bigendian else "<"
    xyz = []
    for name in ("x", "y", "z"):
        field = fields[name]
        dtype = np.dtype(_pointcloud2_field_dtype(field.datatype)).newbyteorder(endian)
        xyz.append(np.frombuffer(msg.data, dtype=dtype, count=1, offset=base + int(field.offset))[0])
    xyz = np.asarray(xyz, dtype=np.float32)
    if not np.isfinite(xyz).all() or float(xyz[2]) <= 0.0:
        raise ValueError(f"Invalid XYZ at pixel ({u}, {v}): {xyz.tolist()}")
    return xyz.astype(float)


def capture_ros_color_pointcloud_frame(args):
    """Compatibility name: returns SDK organized XYZ, with no ROS dependency."""
    from cameras.realsense import organized_cloud
    return organized_cloud(args)


def print_status(nero, target_pose=None):
    try:
        pose = nero.get_flange_pose()
        print("[status] flange_pose =", np.round(pose, 6).tolist())
    except Exception as exc:
        print(f"[status] failed to read flange pose: {type(exc).__name__}: {exc}")
    try:
        tcp_pose = nero.get_tcp_pose()
        print("[status] tcp_pose    =", np.round(tcp_pose, 6).tolist())
    except Exception as exc:
        print(f"[status] failed to read tcp pose: {type(exc).__name__}: {exc}")
    try:
        joints = nero.get_joint_angles()
        print("[status] joint_angles =", np.round(joints, 6).tolist())
    except Exception as exc:
        print(f"[status] failed to read joint angles: {type(exc).__name__}: {exc}")
    if target_pose is not None:
        print("[status] target_pose =", np.round(target_pose, 6).tolist())


def safe_stop(nero, reason):
    print(f"[stop] {reason}")
    try:
        resp = nero.stop()
        print("[stop] electronic_emergency_stop response =", resp)
    except Exception as exc:
        print(f"[stop] electronic_emergency_stop failed: {type(exc).__name__}: {exc}")


def pose_mm_deg_to_m_rad(values):
    vals = np.asarray(values, dtype=float).reshape(6)
    return np.r_[vals[:3] / 1000.0, np.deg2rad(vals[3:])]


def initial_pose_from_args(args):
    return pose_mm_deg_to_m_rad(args.initial_pose_mm_deg)


def initial_joints_from_args(args):
    return np.deg2rad(np.asarray(args.initial_joints_deg, dtype=float).reshape(7))


def move_to_initial_pose(nero, args, require_confirm=True):
    if args.initial_mode == "joint":
        joints = initial_joints_from_args(args)
        print("\n[init] startup initial joint angles:")
        print("[init] joints_rad =", np.round(joints, 6).tolist())
        print("[init] joints_deg =", np.round(np.rad2deg(joints), 3).tolist())
        print("[init] control_mode = move_j joint-space command [j1..j7]")
        if require_confirm:
            ans = input("Type INIT to move Nero to startup initial joint pose, or press Enter to skip: ").strip()
            if ans != "INIT":
                print("[init] skipped")
                return False
        try:
            print("[init] commanding Nero move_j to startup initial joint pose...")
            resp = nero.move_j(joints)
            print("[init] response =", resp)
            wait_until_joints_reached(nero, joints, args, "init")
            print_status(nero)
            return True
        except KeyboardInterrupt:
            print("[init] Ctrl+C received. No emergency stop sent; use e/stop in the menu if needed.")
            return False

    pose = initial_pose_from_args(args)
    assert_workspace(pose[:3], args)
    print("\n[init] startup initial flange pose:")
    print("[init] pose_m_rad =", np.round(pose, 6).tolist())
    print("[init] pose_mm_deg =", np.round(np.r_[pose[:3] * 1000.0, np.rad2deg(pose[3:])], 3).tolist())
    print("[init] control_mode = move_p Cartesian flange pose [x,y,z,roll,pitch,yaw]")
    if require_confirm:
        ans = input("Type INIT to move Nero to startup initial pose, or press Enter to skip: ").strip()
        if ans != "INIT":
            print("[init] skipped")
            return False
    try:
        print("[init] commanding Nero move_p to startup initial pose...")
        resp = nero.move_p(pose)
        print("[init] response =", resp)
        wait_until_pose_reached(nero, pose, args, "init")
        print_status(nero, pose)
        return True
    except KeyboardInterrupt:
        print("[init] Ctrl+C received. No emergency stop sent; use e/stop in the menu if needed.")
        return False


def configure_tcp_offset(nero, args):
    if args.disable_tcp_offset:
        print("[tcp] skip set_tcp_offset (--disable-tcp-offset)")
        return
    offset = np.asarray(args.tcp_offset, dtype=float).reshape(6)
    print("[tcp] set_tcp_offset =", np.round(offset, 6).tolist())
    resp = nero.set_tcp_offset(offset)
    print("[tcp] response =", resp)
    try:
        print("[tcp] current tcp_pose =", np.round(nero.get_tcp_pose(), 6).tolist())
    except Exception as exc:
        print(f"[tcp] failed to read tcp_pose after set_tcp_offset: {type(exc).__name__}: {exc}")


def print_target_result(state):
    if state.get("target_pose") is None:
        print("[target] no target yet. Use c/capture first.")
        return
    print("\n[target]")
    print("P_camera          =", np.round(state["P_camera"], 6).tolist())
    print("P_base            =", np.round(state["P_base"], 6).tolist())
    if state.get("target_tcp_pose") is not None:
        print("target_tcp_pose   =", np.round(state["target_tcp_pose"], 6).tolist())
    print("target_flange_pose=", np.round(state["target_pose"], 6).tolist())
    print("point_source      =", state.get("point_source", "unknown"))
    if state.get("pointcloud_frame_id"):
        print("pointcloud_frame  =", state["pointcloud_frame_id"])
    print("speed_percent     =", int(state["speed_percent"]))
    print("target_frame      =", state.get("target_frame", "flange"))
    if state.get("cross_check", False):
        print("control_mode      = endpoint IK + move_j joint interpolation; m = hover / visual point / return")
    else:
        print("control_mode      = move_p flange command; TCP targets are converted by get_tcp2flange_pose()")
    if state.get("point_workspace_err") is not None:
        print(f"[warn] clicked point outside workspace bounds: {state['point_workspace_err']}")
    if state.get("target_workspace_err") is not None:
        print(f"[warn] target pose outside workspace bounds: {state['target_workspace_err']}")


def capture_and_select_target(nero, args):
    cross_check = bool(getattr(args, "cross_check", False))
    if cross_check:
        from tools.cross_point_check import fresh_pose, _joint_sample
        _, capture_before, _ = fresh_pose(nero)
        joints_before, _ = _joint_sample(nero)
    if args.use_registered_points:
        rgb, cloud_msg, intr, out_dir = capture_ros_color_pointcloud_frame(args)
        depth_m = None
    else:
        rgb, depth_m, intr, out_dir = capture_rgbd(args)
        cloud_msg = None
    flange_pose = nero.get_flange_pose()
    if cross_check:
        from tools.cross_point_check import orientation_error_deg
        flange_pose, capture_after, _ = fresh_pose(nero)
        observation_joints, observation_joint_stamp = _joint_sample(nero)
        if (np.linalg.norm(capture_after[:3, 3]-capture_before[:3, 3]) > 0.001
                or orientation_error_deg(capture_before, capture_after) > 0.3
                or not np.isfinite(joints_before).all() or not np.isfinite(observation_joints).all()
                or np.rad2deg(np.max(np.abs(observation_joints-joints_before))) > 0.2):
            raise ValueError("RGB-D采集期间机械臂移动，请停稳后重新c采集")
    print("[nero] captured flange_pose =", np.round(flange_pose, 6).tolist())

    if depth_m is not None and depth_m.shape[:2] != rgb.shape[:2]:
        print(f"[warn] rgb shape {rgb.shape[:2]} != depth shape {depth_m.shape[:2]}; depth must be aligned to color.")
    if cloud_msg is not None and (int(cloud_msg.width), int(cloud_msg.height)) != (int(rgb.shape[1]), int(rgb.shape[0])):
        print(
            f"[warn] rgb size {rgb.shape[1]}x{rgb.shape[0]} != point cloud size "
            f"{cloud_msg.width}x{cloud_msg.height}; pixel indexing may be invalid."
        )

    if args.u is None or args.v is None:
        u, v = read_clicked_pixel(rgb)
    else:
        u, v = float(args.u), float(args.v)

    if cloud_msg is not None:
        P_camera = pointcloud2_xyz_at(cloud_msg, u, v)
        depth = float(P_camera[2])
        print("[capture] P_camera from registered PointCloud2 pixel XYZ")
    else:
        if cross_check:
            from tools.calibrate_realsense_touch import sample_depth_point
            K = np.array([[intr.fx, 0, intr.cx], [0, intr.fy, intr.cy], [0, 0, 1]])
            P_camera, _ = sample_depth_point(depth_m, K, [u, v], radius=args.depth_radius)
            depth = float(P_camera[2])
        else:
            depth = robust_depth_at(depth_m, u, v, args.depth_radius)
            P_camera = pixel_to_camera(u, v, depth, intr)
        print("[capture] P_camera from depth image back-projection")

    T_flange_camera = handeye_transform(args.calib_json, args.handeye_convention)
    calibration_sha256 = hashlib.sha256(Path(args.calib_json).read_bytes()).hexdigest()
    T_base_flange = make_T(
        R.from_euler(args.flange_rpy_order, flange_pose[3:]).as_matrix(),
        flange_pose[:3],
    )
    T_base_camera = T_base_flange @ T_flange_camera
    P_base = transform_point(T_base_camera, P_camera)

    target_tcp_pose = None
    if args.target_frame == "tcp":
        current_tcp_pose = nero.get_tcp_pose()
        target_tcp_pose = np.array(current_tcp_pose, dtype=float)
        target_tcp_pose[:3] = P_base + np.array([0.0, 0.0, float(args.hover)])
        target_pose = nero.get_tcp2flange_pose(target_tcp_pose)
    else:
        target_pose = np.array(flange_pose, dtype=float)
        target_pose[:3] = P_base + np.array([0.0, 0.0, float(args.hover)])
    point_workspace_err = workspace_error(P_base, args)
    target_workspace_err = workspace_error(target_pose[:3], args)

    result = {
        "pixel": [float(u), float(v)],
        "depth_m": float(depth),
        "P_camera": P_camera.tolist(),
        "P_base": P_base.tolist(),
        "current_flange_pose": flange_pose.tolist(),
        "target_frame": args.target_frame,
        "target_tcp_pose": None if target_tcp_pose is None else target_tcp_pose.tolist(),
        "target_flange_pose": target_pose.tolist(),
        "point_workspace_error": point_workspace_err,
        "target_workspace_error": target_workspace_err,
        "hover_m": float(args.hover),
        "handeye_convention": args.handeye_convention,
        "intrinsics": {"fx": intr.fx, "fy": intr.fy, "cx": intr.cx, "cy": intr.cy},
        "point_source": "registered_pointcloud" if cloud_msg is not None else "depth_image",
        "pointcloud_frame_id": None if cloud_msg is None else cloud_msg.header.frame_id,
        "executed": False,
        "calibration_sha256": calibration_sha256,
        "T_base_flange": T_base_flange.tolist(),
        "T_flange_camera": T_flange_camera.tolist(),
    }
    if cross_check:
        result["observation_joints_rad"] = observation_joints.tolist()
        result["observation_joint_timestamp"] = observation_joint_stamp
    write_result(out_dir, result)

    state = {
        "target_pose": target_pose,
        "target_tcp_pose": target_tcp_pose,
        "P_camera": P_camera,
        "P_base": P_base,
        "flange_pose": flange_pose,
        "point_workspace_err": point_workspace_err,
        "target_workspace_err": target_workspace_err,
        "speed_percent": int(args.nero_speed_percent),
        "out_dir": out_dir,
        "result": result,
        "point_source": result["point_source"],
        "pointcloud_frame_id": result["pointcloud_frame_id"],
        "target_frame": args.target_frame,
        "T_flange_camera": T_flange_camera.copy(),
        "calibration_sha256": calibration_sha256,
        "cross_check": cross_check,
    }
    if cross_check:
        state["observation_joints_rad"] = observation_joints.copy()
    print_target_result(state)
    return state


def interactive_motion_menu(nero, args, state=None):
    if state is None:
        state = {}
    moved = False
    cross_check = bool(getattr(args, "cross_check", False))
    if not cross_check:
        print(
            "\nInteractive control:\n"
            "  i / init    command Nero to startup initial pose\n"
            "  c / capture capture one RGB-D frame, click a pixel, compute target\n"
            "  m / move    command Nero move_p to current target pose\n"
            "  t / target  print current target\n"
            "  s / status print current flange pose and joint angles\n"
            "  e / stop   electronic emergency stop\n"
            "  q / quit   exit this script\n"
            "Ctrl+C cancels terminal input only. Use e/stop for electronic emergency stop.\n"
        )
    if cross_check:
        print(
            "叉叉模式：m=到悬停点、确认后接近视觉点、自动原路返回拍照位置；"
            "r=记录示教实际轻触TCP；k=导出平移补偿候选。\n"
            "拍照后先不要示教移动；悬停／目标各求一次IK、各发一次move_j，不逐毫米点动。到悬停点后输入APPROACH才接近。\n"
            "返回的是本次拍照位置，不是固定初始位；不改变夹爪开度，目标朝向沿用拍照值。"
            "没有全局避障或可靠轻触检测；全程需人工确认路径无遮挡。\n"
            "正常返回后再切示教轻触同一点，用r读取TCP；示教朝向可以不同。"
            "\nc=重新采集，t=目标，s=状态，e=急停，q=退出；Ctrl+C退出并检查普通停机。"
        )
    while True:
        try:
            cmd = input("point-test> ").strip().lower()
        except KeyboardInterrupt:
            if cross_check:
                print("\n叉叉测试退出；复用抓取适配器的普通停机／退出检查。")
                return moved
            print("\n[input] Ctrl+C received. No emergency stop sent; use e/stop if needed.")
            continue
        except EOFError:
            print("[exit] stdin closed")
            return moved

        if cmd in {"q", "quit", "exit"}:
            print("[exit] leaving interactive control")
            return moved
        if cross_check and cmd in {"i", "init", "home"}:
            print("叉叉模式不去固定初始位；m正常完成后会原路返回本次拍照位置。")
            continue
        if cross_check and cmd in {"m", "move", "d", "down", "r", "reference", "k", "compensate"}:
            from tools.cross_point_check import record_taught_point, export_compensation, execute_cross_roundtrip
            try:
                if cmd in {"r", "reference"}:
                    record_taught_point(nero, args, state)
                elif cmd in {"k", "compensate"}:
                    export_compensation(args, state)
                elif cmd in {"d", "down"}:
                    print("不再逐次d下降；用m执行完整往返，到悬停点后确认APPROACH。")
                else:
                    execute_cross_roundtrip(nero, args, state)
                    if state.get("cross_roundtrip", {}).get("motion_sent", False):
                        moved = True
                        state["result"]["executed"] = True
                        write_result(state["out_dir"], state["result"])
            except KeyboardInterrupt:
                return moved
            except Exception as exc:
                print(f"[cross] {type(exc).__name__}: {exc}")
            continue
        if cmd in {"c", "capture", "select"}:
            try:
                state.clear()
                state.update(capture_and_select_target(nero, args))
            except KeyboardInterrupt:
                print("[capture] cancelled. No emergency stop sent.")
            except Exception as exc:
                print(f"[capture] failed: {type(exc).__name__}: {exc}")
            continue
        if cmd in {"t", "target"}:
            print_target_result(state)
            continue
        if cmd in {"i", "init", "home"}:
            try:
                move_to_initial_pose(nero, args, require_confirm=True)
            except Exception as exc:
                print(f"[init] failed: {type(exc).__name__}: {exc}")
            continue
        if cmd in {"s", "status", "p", "print"}:
            print_status(nero, state.get("target_pose"))
            continue
        if cmd in {"e", "stop", "estop"}:
            safe_stop(nero, "manual stop command")
            continue
        if cmd in {"m", "move"}:
            target_pose = state.get("target_pose")
            if target_pose is None:
                print("[move] no target yet. Use c/capture first.")
                continue
            assert_workspace(target_pose[:3], args)
            try:
                print("[move] commanding Nero move_p...")
                resp = nero.move_p(target_pose)
                print("[move] response =", resp)
                wait_until_pose_reached(nero, target_pose, args, "move")
                print_status(nero, target_pose)
                moved = True
                print("[move] done. Menu is still active; use s/status, e/stop, m/move again, or q/quit.")
                continue
            except KeyboardInterrupt:
                print("[move] Ctrl+C received. No emergency stop sent; use e/stop if needed.")
                continue
        print("unknown command; use i, c, m, t, s, e, or q")


def parse_args():
    p = argparse.ArgumentParser(description="SDK click-to-Nero-base point test.")
    p.add_argument("--calib-json", required=True, help="Hand-eye calibration JSON")
    p.add_argument(
        "--handeye-convention",
        default="flange_camera_xyzw",
        choices=[
            "flange_camera_xyzw",
            "camera_flange_xyzw",
            "flange_camera_wxyz",
            "camera_flange_wxyz",
        ],
        help="Use the convention that was verified with verify_handeye_point.py",
    )

    p.add_argument("--orbbec-color-topic", default="/camera/color/image_raw")
    p.add_argument("--orbbec-depth-topic", default="/camera/depth/image_raw")
    p.add_argument("--orbbec-camera-info-topic", default="/camera/color/camera_info")
    p.add_argument("--orbbec-points-topic", default="/camera/depth_registered/points")
    p.add_argument("--use-registered-points", action="store_true", help="Use organized /camera/depth_registered/points XYZ instead of depth image back-projection")
    p.add_argument("--orbbec-depth-scale", type=float, default=None)
    p.add_argument("--orbbec-timeout", type=float, default=5.0)
    p.add_argument("--orbbec-warmup-frames", type=int, default=5)
    p.add_argument("--save-dir", default="real_nero_point_test_logs")

    p.add_argument("--rgb", default=None, help="Optional offline RGB image")
    p.add_argument("--depth", default=None, help="Optional offline depth image")
    p.add_argument("--depth-scale", type=float, default=None)
    p.add_argument("--fx", type=float, default=None)
    p.add_argument("--fy", type=float, default=None)
    p.add_argument("--cx", type=float, default=None)
    p.add_argument("--cy", type=float, default=None)

    p.add_argument("--nero-interface", default="socketcan")
    p.add_argument("--nero-channel", default="can0")
    p.add_argument("--nero-bitrate", type=int, default=1000000)
    p.add_argument("--nero-firmware", default="V111", choices=["DEFAULT", "V111", "V112"])
    p.add_argument("--nero-enable-timeout", type=float, default=5.0)
    p.add_argument("--nero-speed-percent", type=int, default=10)
    p.add_argument("--flange-rpy-order", default="xyz")
    p.add_argument("--target-frame", default="tcp", choices=["tcp", "flange"], help="Move TCP or flange to clicked point plus hover")
    p.add_argument(
        "--tcp-offset",
        type=float,
        nargs=6,
        default=DEFAULT_TCP_OFFSET,
        metavar=("X", "Y", "Z", "ROLL", "PITCH", "YAW"),
        help="TCP offset in flange frame, meters/radians",
    )
    p.add_argument("--disable-tcp-offset", action="store_true", help="Do not call pyAgxArm set_tcp_offset() at startup")
    p.add_argument(
        "--initial-mode",
        default="joint",
        choices=["joint", "pose"],
        help="Use move_j joint angles or move_p Cartesian flange pose for startup initial move",
    )
    p.add_argument(
        "--initial-joints-deg",
        type=float,
        nargs=7,
        default=DEFAULT_INITIAL_JOINT_DEG,
        metavar=("J1", "J2", "J3", "J4", "J5", "J6", "J7"),
        help="Startup initial joint angles in degrees",
    )
    p.add_argument(
        "--initial-pose-mm-deg",
        type=float,
        nargs=6,
        default=DEFAULT_INITIAL_POSE_MM_DEG,
        metavar=("X_MM", "Y_MM", "Z_MM", "RX_DEG", "RY_DEG", "RZ_DEG"),
        help="Startup initial flange pose in mm/deg, used only with --initial-mode pose",
    )
    p.add_argument("--skip-initial-move", action="store_true", help="Do not prompt for startup initial move")
    p.add_argument("--yes-initial", action="store_true", help="Move to startup initial pose without typing INIT")

    p.add_argument("--u", type=float, default=None)
    p.add_argument("--v", type=float, default=None)
    p.add_argument("--depth-radius", type=int, default=2)
    p.add_argument("--hover", type=float, default=0.10, help="Move this many meters above clicked point")
    p.add_argument("--execute", action="store_true", help="Actually command Nero move_p after confirmation")
    p.add_argument("--cross-check", action="store_true", help="Supervised visual-point round trip, taught TCP comparison and translation candidate")
    p.add_argument("--cross-step-mm", type=float, default=1.0, help="Joint-interpolation FK check sampling density only; not execution waypoints")
    p.add_argument("--yes", action="store_true", help="Skip interactive confirmation when --execute is set")
    p.add_argument("--no-menu", action="store_true", help="Exit after dry-run instead of entering the safety menu")
    p.add_argument("--capture-on-start", action="store_true", help="Capture/select one point immediately before opening the menu")
    p.add_argument("--move-wait-timeout", type=float, default=None, help="Stage arrival timeout seconds (cross-check: 60, legacy: 20)")
    p.add_argument("--move-poll-interval", type=float, default=0.25, help="Seconds between get_flange_pose polls after move_p")
    p.add_argument("--move-pos-tol", type=float, default=None, help="Stage endpoint tolerance meters (cross-check: 0.002, legacy: 0.015)")
    p.add_argument("--move-rpy-tol", type=float, default=None, help="Orientation tolerance radians (cross-check: SO3 angle 1deg; legacy: RPY norm 5deg)")
    p.add_argument("--move-joint-tol", type=float, default=None, help="Joint tolerance radians (cross-check: max error 0.5deg; legacy: norm 2deg)")

    p.add_argument("--min-x", type=float, default=-0.80)
    p.add_argument("--max-x", type=float, default=0.80)
    p.add_argument("--min-y", type=float, default=-0.60)
    p.add_argument("--max-y", type=float, default=0.60)
    p.add_argument("--min-z", type=float, default=0.02)
    p.add_argument("--max-z", type=float, default=1.20)
    from cameras.options import add_camera_arguments
    add_camera_arguments(p)
    args = p.parse_args()
    defaults = (dict(move_wait_timeout=60.0, move_pos_tol=0.002,
                     move_rpy_tol=np.deg2rad(1.0), move_joint_tol=np.deg2rad(0.5))
                if args.cross_check else dict(move_wait_timeout=20.0, move_pos_tol=0.015,
                     move_rpy_tol=np.deg2rad(5.0), move_joint_tol=np.deg2rad(2.0)))
    for key, value in defaults.items():
        if getattr(args, key) is None:
            setattr(args, key, value)
    if args.cross_check:
        if (not args.execute or not args.skip_initial_move or args.yes or args.yes_initial
                or args.disable_tcp_offset or args.target_frame != "tcp" or args.flange_rpy_order != "xyz"
                or args.nero_speed_percent != 1 or not np.isfinite(args.hover) or args.hover < 0.03
                or not np.isfinite(args.cross_step_mm) or not 0 < args.cross_step_mm <= 1.0
                or not np.isfinite(args.move_wait_timeout) or args.move_wait_timeout <= 0
                or not np.isfinite(args.move_pos_tol) or not 0 < args.move_pos_tol <= 0.005
                or not np.isfinite(args.move_joint_tol) or not 0 < args.move_joint_tol <= np.deg2rad(1.0)
                or not np.isfinite(args.move_rpy_tol) or not 0 < args.move_rpy_tol <= np.deg2rad(3.0)
                or not np.isfinite(np.asarray(args.tcp_offset, float)).all()
                or args.rgb is not None or args.depth is not None):
            p.error("叉叉模式要求 --execute --skip-initial-move --nero-speed-percent 1、TCP/xyz、hover>=0.03、0<step<=1；禁止自动确认和离线图运动")
        args.orbbec_depth_median_frames = 15
        args.orbbec_depth_median_min_valid = 8
        args.orbbec_depth_median_max_mad = 0.002
    return args


def main():
    args = parse_args()
    arm_type, arm_options = NeroArm, {}
    if args.cross_check:
        from real_nero_grasp.arm import RealNeroGraspArm
        arm_type, arm_options = RealNeroGraspArm, {"gripper_enabled": False}
        print("[cross] SDK原始内参；外参 =", Path(args.calib_json).resolve())
    with arm_type(
        args.nero_interface,
        args.nero_channel,
        args.nero_bitrate,
        args.nero_firmware,
        args.nero_enable_timeout,
        args.nero_speed_percent,
        **arm_options,
    ) as nero:
        configure_tcp_offset(nero, args)

        if not args.skip_initial_move:
            move_to_initial_pose(nero, args, require_confirm=not args.yes_initial)

        state = {}
        if args.capture_on_start or args.execute or args.yes:
            state = capture_and_select_target(nero, args)

        if not args.execute:
            print("\n[ready] Nero is connected. Use c/capture to select a target, then m/move.")
            if not args.no_menu:
                interactive_motion_menu(nero, args, state)
            else:
                print("Omit --no-menu to use c/capture and m/move interactively.")
            return

        target_pose = state.get("target_pose")
        if target_pose is None:
            raise RuntimeError("No target available. Use --capture-on-start or run without --execute and use c/capture.")
        assert_workspace(target_pose[:3], args)

        if not args.yes:
            interactive_motion_menu(nero, args, state)
            return

        try:
            print("[move] commanding Nero move_p...")
            resp = nero.move_p(target_pose)
            print("[move] response =", resp)
            wait_until_pose_reached(nero, target_pose, args, "move")
            print_status(nero, target_pose)
            if not args.no_menu:
                interactive_motion_menu(nero, args, state)
        except KeyboardInterrupt:
            print("[move] Ctrl+C received. No emergency stop sent; use e/stop if needed.")
            if not args.no_menu:
                interactive_motion_menu(nero, args, state)


if __name__ == "__main__":
    main()
