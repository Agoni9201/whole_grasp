#!/usr/bin/env python3
"""Move Nero through hand-eye calibration observation poses.

The poses are generated around one table/marker center, and the camera optical
axis is aimed at that center using the current approximate hand-eye transform.
Use this together with the hand-eye calibration capture node: after each pose is
reached, capture one calibration sample there, then press Enter in this script
to continue to the next pose.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # 把工程根目录加入 import 路径


import argparse
import json
import math
import os
import threading
import time
from datetime import datetime
from types import SimpleNamespace

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R

from tools.real_nero_point_test import (
    DEFAULT_INITIAL_JOINT_DEG,
    DEFAULT_TCP_OFFSET,
    NeroArm,
    capture_rgbd,
    handeye_transform,
    make_T,
    pose_error,
    safe_stop,
    transform_point,
    wait_until_joints_reached,
    wait_until_pose_reached,
)
from real_nero_single_grasp_runner import (
    DEFAULT_IK_URDF,
    MotionNotReached,
    PlanNotExecutable,
    _as_pose_list,
    _get_ik_solver,
    _make_T_from_pose,
    _solve_ik_for_flange_pose,
)


class _HandeyePosePublisher:
    """Local pose JSONL recorder; old topic argument is now an output file path."""
    def __init__(self, topic, hz):
        path = Path(topic).expanduser()
        if path.suffix != '.jsonl':
            raise ValueError('Use --handeye-pose-log <path.jsonl>, not a ROS topic')
        path.parent.mkdir(parents=True, exist_ok=True)
        self.file = path.open('x', encoding='utf-8')
    def set_pose(self, pose6):
        pose = np.asarray(pose6, dtype=float).reshape(6)
        self.file.write(json.dumps({'wall_time': time.time(), 'flange_pose': pose.tolist()}) + '\n')
        self.file.flush()
    def close(self):
        self.file.close()


def _camera_look_at_T(cam_pos_base, target_base, roll_deg=0.0):
    cam_pos = np.asarray(cam_pos_base, dtype=float).reshape(3)
    target = np.asarray(target_base, dtype=float).reshape(3)
    z_axis = target - cam_pos
    z_norm = float(np.linalg.norm(z_axis))
    if z_norm < 1e-6:
        raise ValueError("camera pose is too close to the target center")
    z_axis = z_axis / z_norm

    up = np.array([0.0, 0.0, 1.0], dtype=float)
    x_axis = np.cross(z_axis, up)
    if np.linalg.norm(x_axis) < 1e-6:
        x_axis = np.array([1.0, 0.0, 0.0], dtype=float)
    x_axis = x_axis / (np.linalg.norm(x_axis) + 1e-9)
    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis / (np.linalg.norm(y_axis) + 1e-9)
    R_base_camera = np.column_stack([x_axis, y_axis, z_axis])

    if abs(float(roll_deg)) > 1e-9:
        R_base_camera = R_base_camera @ R.from_euler("z", math.radians(float(roll_deg))).as_matrix()

    T = np.eye(4, dtype=float)
    T[:3, :3] = R_base_camera
    T[:3, 3] = cam_pos
    return T


def _pose_delta_text(current, target):
    pos_err, rpy_err = pose_error(current, target)
    return f"pos_err={pos_err:.4f}m rpy_err={math.degrees(rpy_err):.2f}deg"


def _try_detect_aruco(bgr, dictionary_name, marker_id=None):
    if not hasattr(cv2, "aruco"):
        return False, "cv2.aruco unavailable"
    aruco = cv2.aruco
    dict_id = getattr(aruco, dictionary_name, None)
    if dict_id is None:
        return False, f"unknown aruco dictionary {dictionary_name}"
    dictionary = aruco.getPredefinedDictionary(dict_id)
    params = aruco.DetectorParameters()
    try:
        detector = aruco.ArucoDetector(dictionary, params)
        corners, ids, _ = detector.detectMarkers(bgr)
    except AttributeError:
        corners, ids, _ = aruco.detectMarkers(bgr, dictionary, parameters=params)
    if ids is None or len(ids) == 0:
        return False, "no marker detected"
    ids_flat = np.asarray(ids, dtype=int).reshape(-1)
    if marker_id is not None and int(marker_id) not in ids_flat.tolist():
        return False, f"detected ids={ids_flat.tolist()}, target id={int(marker_id)} absent"
    return True, f"detected ids={ids_flat.tolist()}"


def _click_target_base(arm, args, T_flange_camera):
    print("[target] capture current RGB-D frame; click ArUco/table-center point, then press Enter in image window")
    bgr, depth_m, intr, out_dir = capture_rgbd(args)
    from tools.verify_handeye_point import pixel_to_camera, read_clicked_pixel, robust_depth_at

    u, v = read_clicked_pixel(bgr)
    depth = robust_depth_at(depth_m, u, v, args.depth_radius)
    p_cam = pixel_to_camera(u, v, depth, intr)
    T_base_flange = _make_T_from_pose(arm.get_flange_pose())
    p_base = transform_point(T_base_flange @ T_flange_camera, p_cam)
    print(f"[target] clicked pixel={[u, v]} depth={depth:.4f}m p_camera={np.round(p_cam, 6).tolist()}")
    print("[target] estimated target_base =", np.round(p_base, 6).tolist())
    if out_dir is not None:
        with open(os.path.join(out_dir, "clicked_handeye_sweep_target.json"), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "pixel": [float(u), float(v)],
                    "depth_m": float(depth),
                    "p_camera": p_cam.tolist(),
                    "target_base": p_base.tolist(),
                },
                f,
                indent=2,
            )
    return p_base


def _generate_camera_poses(target_base, args):
    target = np.asarray(target_base, dtype=float).reshape(3)
    # Sample viewpoints on a sphere around the marker/table center.  The
    # default azimuths mirror the grasp active-view orbit angles: centered view
    # plus left/right side views at 25/35/45 degrees.
    azimuths = _parse_float_csv(args.orbit_azimuths_deg, "--orbit-azimuths-deg")
    if not azimuths:
        azimuths = [0.0]
    patterns = _parse_sphere_view_specs(args.sphere_view_specs)
    roll_offsets = _parse_float_csv(args.roll_offsets_deg, "--roll-offsets-deg")
    if not roll_offsets:
        roll_offsets = [0.0]
    base_angle = math.radians(float(args.base_azimuth_deg))
    poses = []
    idx = 0
    for row, (distance, elevation_deg, roll_deg) in enumerate(patterns):
        radius = float(distance) * math.cos(math.radians(float(elevation_deg)))
        height = float(distance) * math.sin(math.radians(float(elevation_deg)))
        for az in azimuths:
            theta = base_angle + math.radians(az)
            cam_pos = np.array(
                [
                    target[0] + radius * math.cos(theta),
                    target[1] + radius * math.sin(theta),
                    target[2] + height,
                ],
                dtype=float,
            )
            for roll_offset in roll_offsets:
                total_roll_deg = float(roll_deg) + float(roll_offset)
                T_base_camera = _camera_look_at_T(cam_pos, target, roll_deg=total_roll_deg)
                poses.append(
                    {
                        "index": idx,
                        "row": row,
                        "base_azimuth_deg": float(args.base_azimuth_deg),
                        "azimuth_deg": az,
                        "world_azimuth_deg": math.degrees(theta),
                        "radius": radius,
                        "distance": float(distance),
                        "elevation_deg": float(elevation_deg),
                        "height": height,
                        "roll_deg": total_roll_deg,
                        "is_topdown": bool(float(elevation_deg) >= float(args.topdown_elevation_threshold_deg)),
                        "roll_offset_deg": float(roll_offset),
                        "T_base_camera": T_base_camera,
                    }
                )
                idx += 1
    return poses


def _candidate_pose_count(target_base, args):
    return len(_generate_camera_poses(target_base, args))


def _current_camera_azimuth_deg(arm, target_base, T_flange_camera):
    T_base_flange = _make_T_from_pose(arm.get_flange_pose())
    T_base_camera = T_base_flange @ T_flange_camera
    cam_pos = np.asarray(T_base_camera[:3, 3], dtype=float).reshape(3)
    target = np.asarray(target_base, dtype=float).reshape(3)
    vec = cam_pos[:2] - target[:2]
    if float(np.linalg.norm(vec)) < 1e-6:
        return None
    return math.degrees(math.atan2(float(vec[1]), float(vec[0])))


def _apply_current_view_azimuth_preference(arm, args, target_base, T_flange_camera):
    if not args.prefer_current_view_azimuth:
        return
    current_azimuth = _current_camera_azimuth_deg(arm, target_base, T_flange_camera)
    if current_azimuth is None:
        print("[plan] current camera azimuth unavailable; using configured base azimuth candidates")
        return

    offsets = _parse_float_csv(args.current_view_azimuth_offsets_deg, "--current-view-azimuth-offsets-deg")
    local_candidates = [float(current_azimuth) + float(offset) for offset in offsets]
    configured_candidates = _parse_float_csv(args.base_azimuth_candidates, "--base-azimuth-candidates") if args.include_global_azimuth_fallback else []
    combined = []
    seen = set()
    for value in local_candidates + configured_candidates:
        key = round(float(value), 6)
        if key in seen:
            continue
        seen.add(key)
        combined.append(float(value))

    args.base_azimuth_deg = float(current_azimuth)
    args.base_azimuth_candidates = ",".join(f"{value:.3f}" for value in combined)
    print(
        f"[plan] prefer current initial view: current_camera_azimuth={current_azimuth:+.1f}deg "
        f"local_offsets={offsets} include_global_fallback={bool(args.include_global_azimuth_fallback)}"
    )


def _within_workspace(flange_pose, args):
    return _workspace_error(flange_pose, args) is None


def _workspace_error(flange_pose, args):
    x, y, z = np.asarray(flange_pose, dtype=float).reshape(6)[:3]
    checks = [
        ("x", float(x), float(args.min_x), float(args.max_x)),
        ("y", float(y), float(args.min_y), float(args.max_y)),
        ("z", float(z), float(args.min_z), float(args.max_z)),
    ]
    problems = []
    for name, value, lower, upper in checks:
        if value < lower:
            problems.append(f"{name}={value:.3f}<{lower:.3f}")
        elif value > upper:
            problems.append(f"{name}={value:.3f}>{upper:.3f}")
    if not problems:
        return None
    return ";".join(problems) + f" flange_xyz={[round(float(x), 3), round(float(y), 3), round(float(z), 3)]}"


def _parse_float_csv(value, label):
    values = []
    for token in str(value).replace(";", ",").split(","):
        token = token.strip()
        if not token:
            continue
        try:
            values.append(float(token))
        except ValueError as exc:
            raise ValueError(f"Invalid {label} value {token!r} in {value!r}") from exc
    return values


def _parse_sphere_view_specs(value):
    specs = []
    for token in str(value).replace(";", ",").split(","):
        token = token.strip()
        if not token:
            continue
        parts = [part.strip() for part in token.split(":")]
        if len(parts) != 3:
            raise ValueError(
                "Invalid --sphere-view-specs entry "
                f"{token!r}; expected DISTANCE_M:ELEVATION_DEG:ROLL_DEG"
            )
        try:
            distance, elevation_deg, roll_deg = [float(part) for part in parts]
        except ValueError as exc:
            raise ValueError(f"Invalid numeric value in --sphere-view-specs entry {token!r}") from exc
        if distance <= 0.0:
            raise ValueError(f"Invalid --sphere-view-specs distance {distance}; must be positive")
        specs.append((distance, elevation_deg, roll_deg))
    if not specs:
        raise ValueError("--sphere-view-specs produced no viewpoints")
    return specs


def _base_azimuth_candidates(args):
    values = [float(args.base_azimuth_deg)]
    values.extend(_parse_float_csv(args.base_azimuth_candidates, "--base-azimuth-candidates"))
    deduped = []
    seen = set()
    for value in values:
        key = round(float(value), 6)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(float(value))
    return deduped


def _wrap_joint_delta(q, q_ref):
    return (np.asarray(q, dtype=float) - np.asarray(q_ref, dtype=float) + np.pi) % (2.0 * np.pi) - np.pi


def _pose_to_wxyz(pose):
    pose = np.asarray(pose, dtype=float).reshape(6)
    quat_xyzw = R.from_euler("xyz", pose[3:]).as_quat()
    return np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]], dtype=float)


def _joint_limit_margin(q, solver):
    q = np.asarray(q, dtype=float).reshape(-1)
    low, high = solver.get_joint_limits()
    low = np.asarray(low, dtype=float).reshape(-1)[: q.size]
    high = np.asarray(high, dtype=float).reshape(-1)[: q.size]
    return float(np.min(np.minimum(q - low, high - q)))


def _j2_side_ok(q, current_joints, args):
    mode = str(getattr(args, "j2_side", "any")).strip().lower()
    if mode == "any":
        return True
    q = np.asarray(q, dtype=float).reshape(7)
    current_joints = np.asarray(current_joints, dtype=float).reshape(7)
    deadband = math.radians(float(getattr(args, "j2_side_deadband_deg", 3.0)))
    j2 = float(q[1])
    if mode == "positive":
        return j2 > deadband
    if mode == "negative":
        return j2 < -deadband
    if mode == "opposite-current":
        current_j2 = float(current_joints[1])
        if current_j2 >= 0.0:
            return j2 < -deadband
        return j2 > deadband
    raise ValueError(f"Unsupported --j2-side {mode!r}")


def _joint_seed_candidates_for_j2_side(current_joints, args):
    current = np.asarray(current_joints, dtype=float).reshape(7)
    seeds = [current]

    initial = np.deg2rad(np.asarray(args.initial_joints_deg, dtype=float).reshape(7))
    seeds.append(initial)

    j2_seed_degs = _parse_float_csv(args.j2_seed_degs, "--j2-seed-degs")
    for j2_deg in j2_seed_degs:
        seed = initial.copy()
        seed[1] = math.radians(float(j2_deg))
        seeds.append(seed)

        seed = current.copy()
        seed[1] = math.radians(float(j2_deg))
        seeds.append(seed)

    for j7_deg in (90.0, 60.0, 30.0, 0.0, -30.0, -60.0):
        seed = current.copy()
        seed[6] = np.deg2rad(j7_deg)
        seeds.append(seed)

    unique = []
    for seed in seeds:
        seed = np.asarray(seed, dtype=float).reshape(7)
        if not any(np.linalg.norm(_wrap_joint_delta(seed, old)) < 1e-6 for old in unique):
            unique.append(seed)
    return unique


def _solve_ik_for_flange_pose_sweep(arm, flange_pose, args, label):
    if str(getattr(args, "j2_side", "any")).strip().lower() == "any":
        return _solve_ik_for_flange_pose(arm, flange_pose, args, label)

    solver = _get_ik_solver(args)
    flange_pose = np.asarray(flange_pose, dtype=float).reshape(6)
    target_pos = flange_pose[:3]
    target_wxyz = _pose_to_wxyz(flange_pose)
    current_joints = np.asarray(arm.get_joint_angles(), dtype=float).reshape(7)

    rejected_side = 0
    solutions = []
    for seed in _joint_seed_candidates_for_j2_side(current_joints, args):
        try:
            q = solver.solve_ik(
                target_position=target_pos,
                target_wxyz=target_wxyz,
                initial_guess=np.asarray(seed, dtype=float).reshape(-1)[: solver.num_joints],
                use_previous_solution=False,
            )
        except Exception as exc:
            print(f"[{label}][ik][warn] solve failed for seed: {type(exc).__name__}: {exc}")
            continue
        if q is None:
            continue
        q = np.asarray(q, dtype=float).reshape(-1)
        if q.size != 7 or not np.isfinite(q).all():
            continue
        if not solver.check_joint_limits(q):
            continue
        if not _j2_side_ok(q, current_joints, args):
            rejected_side += 1
            continue
        delta = _wrap_joint_delta(q, current_joints)
        max_delta = float(np.max(np.abs(delta)))
        if args.ik_max_joint_delta is not None and max_delta > float(args.ik_max_joint_delta):
            continue
        margin = _joint_limit_margin(q, solver)
        joint7_delta = abs(float(delta[6]))
        score = float(np.linalg.norm(delta)) + float(args.ik_joint7_weight) * joint7_delta - 0.02 * margin
        solutions.append((score, q, delta, margin))

    if not solutions:
        raise PlanNotExecutable(
            f"{label} IK failed: no valid --j2-side {args.j2_side} solution "
            f"(rejected_side={rejected_side})"
        )

    solutions.sort(key=lambda item: item[0])
    score, q_best, delta, margin = solutions[0]
    print(f"[{label}][ik] selected joint target deg =", np.round(np.rad2deg(q_best), 3).tolist())
    print(
        f"[{label}][ik] j2_side={args.j2_side} "
        f"joint_delta_deg={np.round(np.rad2deg(delta), 3).tolist()} "
        f"max_delta={np.rad2deg(np.max(np.abs(delta))):.2f}deg "
        f"limit_margin={np.rad2deg(margin):.2f}deg score={score:.4f}"
    )
    return q_best


def _move_flange_with_correction(arm, target_flange_pose, args, label):
    target_flange_pose = np.asarray(target_flange_pose, dtype=float).reshape(6)
    q_target = _solve_ik_for_flange_pose_sweep(arm, target_flange_pose, args, label)
    print(f"[{label}] target_flange_pose =", np.round(target_flange_pose, 6).tolist())
    print(f"[{label}] target_joints_deg  =", np.round(np.rad2deg(q_target), 3).tolist())
    move_resp = arm.move_j(q_target)
    print(f"[{label}] move_j response = {move_resp}")
    reached, current_pose, pos_err, rpy_err = wait_until_pose_reached(arm, target_flange_pose, args, label)
    if current_pose is None:
        current_pose = arm.get_flange_pose()
    print(f"[{label}] actual_flange_pose =", np.round(current_pose, 6).tolist())
    print(f"[{label}] error after move: {_pose_delta_text(current_pose, target_flange_pose)}")
    if not reached:
        try:
            current_joints = np.asarray(arm.get_joint_angles(), dtype=float).reshape(7)
            joint_err = _wrap_joint_delta(current_joints, q_target)
            print(f"[{label}][warn] current_joints_deg =", np.round(np.rad2deg(current_joints), 3).tolist())
            print(f"[{label}][warn] joint_error_deg   =", np.round(np.rad2deg(joint_err), 3).tolist())
        except Exception as exc:
            print(f"[{label}][warn] failed to read current joints after move: {type(exc).__name__}: {exc}")

    if pos_err <= float(args.correct_pos_tol) and rpy_err <= math.radians(float(args.correct_rpy_tol_deg)):
        return current_pose, True

    for correction_idx in range(max(int(args.correction_attempts), 0)):
        print(f"[{label}] correction {correction_idx + 1}/{int(args.correction_attempts)}")
        q_target = _solve_ik_for_flange_pose_sweep(arm, target_flange_pose, args, f"{label}_correct{correction_idx + 1}")
        move_resp = arm.move_j(q_target)
        print(f"[{label}_correct{correction_idx + 1}] move_j response = {move_resp}")
        reached, current_pose, pos_err, rpy_err = wait_until_pose_reached(
            arm,
            target_flange_pose,
            args,
            f"{label}_correct{correction_idx + 1}",
        )
        if current_pose is None:
            current_pose = arm.get_flange_pose()
        print(f"[{label}] error after correction: {_pose_delta_text(current_pose, target_flange_pose)}")
        if not reached:
            try:
                current_joints = np.asarray(arm.get_joint_angles(), dtype=float).reshape(7)
                joint_err = _wrap_joint_delta(current_joints, q_target)
                print(f"[{label}_correct{correction_idx + 1}][warn] current_joints_deg =", np.round(np.rad2deg(current_joints), 3).tolist())
                print(f"[{label}_correct{correction_idx + 1}][warn] joint_error_deg   =", np.round(np.rad2deg(joint_err), 3).tolist())
            except Exception as exc:
                print(f"[{label}_correct{correction_idx + 1}][warn] failed to read current joints after move: {type(exc).__name__}: {exc}")
        if pos_err <= float(args.correct_pos_tol) and rpy_err <= math.radians(float(args.correct_rpy_tol_deg)):
            return current_pose, True
    return current_pose, False


def _build_planned_poses(arm, args, target_base, T_flange_camera):
    T_camera_flange = np.linalg.inv(T_flange_camera)
    planned = []
    skipped = []
    candidates = _generate_camera_poses(target_base, args)
    current_joints = np.asarray(arm.get_joint_angles(), dtype=float).reshape(7)
    for item in candidates:
        T_base_camera = item["T_base_camera"]
        T_base_flange = T_base_camera @ T_camera_flange
        flange_pose = _as_pose_list(T_base_flange)
        workspace_error = _workspace_error(flange_pose, args)
        if workspace_error is not None:
            skipped.append((item, f"workspace:{workspace_error}"))
            continue
        try:
            q = _solve_ik_for_flange_pose_sweep(arm, flange_pose, args, f"plan_{item['index'] + 1:02d}")
        except Exception as exc:
            skipped.append((item, f"ik:{type(exc).__name__}"))
            continue
        q = np.asarray(q, dtype=float).reshape(7)
        joint_delta = float(np.max(np.abs((q - current_joints + np.pi) % (2.0 * np.pi) - np.pi)))
        if joint_delta > math.radians(float(args.max_plan_joint_delta_deg)):
            skipped.append((item, f"joint_delta={math.degrees(joint_delta):.1f}deg"))
            continue
        item = dict(item)
        item["flange_pose"] = flange_pose
        item["joint_target"] = q
        item["plan_joint_delta_deg"] = math.degrees(joint_delta)
        planned.append(item)
    return planned, skipped


def _parse_seed_joint_offsets_deg(text):
    rows = []
    for chunk in str(text).split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        vals = [float(x) for x in chunk.replace(",", " ").split()]
        if len(vals) != 7:
            raise ValueError(f"Each --seed-joint-offsets-deg entry must contain 7 values, got {len(vals)}: {chunk!r}")
        rows.append(np.deg2rad(np.asarray(vals, dtype=float).reshape(7)))
    if not rows:
        rows.append(np.zeros(7, dtype=float))
    return rows


def _load_pose_seed_records(path, include_invisible=False):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    records = []
    for record in data.get("records", []):
        if not include_invisible and record.get("aruco_visible") is False:
            continue
        if "joint_angles_rad" not in record:
            continue
        item = dict(record)
        item["joint_angles_rad"] = np.asarray(record["joint_angles_rad"], dtype=float).reshape(7)
        item["joint_angles_deg"] = np.rad2deg(item["joint_angles_rad"]).tolist()
        if "flange_pose" in record:
            item["flange_pose"] = np.asarray(record["flange_pose"], dtype=float).reshape(6)
        if "tcp_pose" in record:
            item["tcp_pose"] = np.asarray(record["tcp_pose"], dtype=float).reshape(6)
        records.append(item)
    if not records:
        raise ValueError(f"No usable seed records found in {path}")
    return records


def _planned_seed_key(item):
    q = np.asarray(item["joint_target"], dtype=float).reshape(7)
    return tuple(np.round(q / math.radians(1.0)).astype(int).tolist())


def _build_seed_planned_poses(arm, args):
    records = _load_pose_seed_records(args.pose_seeds_json, args.seed_include_invisible)
    offsets = _parse_seed_joint_offsets_deg(args.seed_joint_offsets_deg)
    current_joints = np.asarray(arm.get_joint_angles(), dtype=float).reshape(7)
    planned = []
    skipped = []
    seen = set()
    max_offset = math.radians(float(args.seed_max_joint_offset_deg))
    idx = 0
    for seed_order, record in enumerate(records):
        seed_q = np.asarray(record["joint_angles_rad"], dtype=float).reshape(7)
        seed_name = str(record.get("name", f"seed_{int(record.get('index', seed_order + 1)):02d}"))
        for variant_idx, offset in enumerate(offsets):
            offset = np.asarray(offset, dtype=float).reshape(7)
            if float(np.max(np.abs(offset))) > max_offset + 1e-12:
                skipped.append(({"index": idx, "seed_name": seed_name, "variant": variant_idx}, "seed_offset_too_large"))
                idx += 1
                continue
            q = seed_q + offset
            item = {
                "index": idx,
                "source": "seed_joint",
                "seed_order": seed_order,
                "seed_index": int(record.get("index", seed_order + 1)),
                "seed_name": seed_name,
                "variant_index": variant_idx,
                "joint_target": q,
                "seed_joint_deg": np.rad2deg(seed_q).tolist(),
                "joint_offset_deg": np.rad2deg(offset).tolist(),
                "plan_joint_delta_deg": math.degrees(float(np.max(np.abs(_wrap_joint_delta(q, current_joints))))),
                "flange_pose": np.asarray(record.get("flange_pose", np.zeros(6)), dtype=float).reshape(6).tolist(),
                "seed_flange_pose": np.asarray(record.get("flange_pose", np.zeros(6)), dtype=float).reshape(6).tolist(),
                "seed_tcp_pose": np.asarray(record.get("tcp_pose", np.zeros(6)), dtype=float).reshape(6).tolist(),
                "aruco_visible_seed": record.get("aruco_visible"),
                "aruco_msg_seed": record.get("aruco_msg"),
            }
            key = _planned_seed_key(item)
            if key in seen:
                skipped.append((item, "duplicate_joint_target"))
                idx += 1
                continue
            seen.add(key)
            planned.append(item)
            idx += 1

    # Keep the selected set balanced across the user-provided seed poses.
    selected = []
    used = set()
    by_seed = {}
    for item in planned:
        by_seed.setdefault(int(item["seed_order"]), []).append(item)
    for values in by_seed.values():
        values.sort(key=lambda item: (int(item["variant_index"]), float(item["plan_joint_delta_deg"])))

    if bool(args.seed_execute_originals_first):
        for seed_order in sorted(by_seed):
            originals = [item for item in by_seed[seed_order] if int(item.get("variant_index", -1)) == 0]
            if not originals:
                continue
            item = originals[0]
            key = _planned_seed_key(item)
            if key in used:
                continue
            used.add(key)
            selected.append(item)
            if len(selected) >= int(args.count):
                break

    while len(selected) < int(args.count):
        added = False
        for seed_order in sorted(by_seed):
            for item in by_seed[seed_order]:
                key = _planned_seed_key(item)
                if key in used:
                    continue
                used.add(key)
                selected.append(item)
                added = True
                break
            if len(selected) >= int(args.count):
                break
        if not added:
            break

    print(
        f"[seed_plan] records={len(records)} offsets={len(offsets)} "
        f"generated={len(planned)} selected={len(selected)} skipped={len(skipped)}"
    )
    seed_counts = {}
    for item in selected:
        name = str(item.get("seed_name", item.get("seed_index", "?")))
        seed_counts[name] = seed_counts.get(name, 0) + 1
    print(f"[seed_plan] selected_per_seed={seed_counts}")
    args._last_plan_generated = len(planned)
    if bool(args.seed_execute_originals_first):
        originals = [item for item in selected if int(item.get("variant_index", -1)) == 0]
        perturbed = [item for item in selected if int(item.get("variant_index", -1)) != 0]
        return originals + perturbed, skipped
    return _order_planned_poses_by_joint_distance(arm, selected), skipped


def _move_seed_joint_pose(arm, item, args, label):
    q_target = np.asarray(item["joint_target"], dtype=float).reshape(7)
    print(f"[{label}] target_joints_deg =", np.round(np.rad2deg(q_target), 3).tolist())
    print(f"[{label}] seed={item.get('seed_name')} variant={item.get('variant_index')} offset_deg={np.round(item.get('joint_offset_deg', []), 3).tolist()}")
    move_resp = arm.move_j(q_target)
    print(f"[{label}] move_j response = {move_resp}")
    reached, current_joints, joint_err = wait_until_joints_reached(arm, q_target, args, label)
    actual_pose = arm.get_flange_pose()
    print(f"[{label}] actual_flange_pose =", np.round(actual_pose, 6).tolist())
    return actual_pose, bool(reached)


def _print_plan_quality(planned, skipped, args, label):
    bucket_width = float(args.direction_bucket_width_deg)

    def bucket(item):
        return _direction_bucket_deg(
            item.get("world_azimuth_deg", item.get("azimuth_deg", 0.0)),
            bucket_width,
        )

    def elev(item):
        return round(float(item.get("elevation_deg", 0.0)), 1)

    planned_by_elev = {}
    planned_by_bucket = {}
    for item in planned:
        planned_by_elev[elev(item)] = planned_by_elev.get(elev(item), 0) + 1
        planned_by_bucket[bucket(item)] = planned_by_bucket.get(bucket(item), 0) + 1

    skipped_by_elev_reason = {}
    for item, reason in skipped:
        key = (elev(item), str(reason).split(":", 1)[0].split("=", 1)[0])
        skipped_by_elev_reason[key] = skipped_by_elev_reason.get(key, 0) + 1

    topdown_planned = sum(1 for item in planned if bool(item.get("is_topdown", False)))
    print(f"[plan] {label} usable_by_elevation={dict(sorted(planned_by_elev.items()))}")
    print(f"[plan] {label} usable_by_direction_bucket={dict(sorted(planned_by_bucket.items()))}")
    print(f"[plan] {label} topdown_usable={topdown_planned}")
    if skipped_by_elev_reason:
        compact = {f"{e:.1f}:{r}": n for (e, r), n in sorted(skipped_by_elev_reason.items())}
        print(f"[plan] {label} skipped_by_elevation_reason={compact}")


def _planned_pose_key(item):
    pose = np.asarray(item["flange_pose"], dtype=float).reshape(6)
    xyz_key = tuple(np.round(pose[:3] / 0.025).astype(int).tolist())
    rpy_key = tuple(np.round(pose[3:] / math.radians(8.0)).astype(int).tolist())
    return xyz_key + rpy_key


def _direction_bucket_deg(angle_deg, bucket_width=45.0):
    wrapped = (float(angle_deg) + 180.0) % 360.0 - 180.0
    return int(round(wrapped / float(bucket_width)))


def _select_diverse_planned_poses(
    planned,
    count,
    min_topdown_count=0,
    direction_bucket_width_deg=30.0,
    min_per_height_layer=0,
):
    count = int(count)
    bucket_width = float(direction_bucket_width_deg)
    unique = []
    used_keys = set()
    for item in planned:
        key = _planned_pose_key(item)
        if key in used_keys:
            continue
        used_keys.add(key)
        unique.append(item)
    if len(unique) <= count:
        return unique

    selected = []
    selected_keys = set()
    rows = sorted({int(item.get("row", 0)) for item in unique}, reverse=True)
    buckets = sorted(
        {
            _direction_bucket_deg(
                item.get("world_azimuth_deg", item.get("azimuth_deg", 0.0)),
                bucket_width,
            )
            for item in unique
        }
    )
    bucket_counts = {bucket: 0 for bucket in buckets}

    def add_item(item):
        key = _planned_pose_key(item)
        if key in selected_keys:
            return False
        selected_keys.add(key)
        selected.append(item)
        bucket = _direction_bucket_deg(
            item.get("world_azimuth_deg", item.get("azimuth_deg", 0.0)),
            bucket_width,
        )
        bucket_counts[bucket] = bucket_counts.get(bucket, 0) + 1
        return True

    # Reserve a few poses from every height/elevation layer first.  This keeps
    # calibration observations from collapsing into one easy IK band.
    min_per_row = max(int(min_per_height_layer), 0)
    row_counts = {row: 0 for row in rows}
    if min_per_row > 0:
        for row in rows:
            if len(selected) >= count:
                break
            row_buckets = sorted(
                {
                    _direction_bucket_deg(
                        item.get("world_azimuth_deg", item.get("azimuth_deg", 0.0)),
                        bucket_width,
                    )
                    for item in unique
                    if int(item.get("row", 0)) == row
                }
            )
            while row_counts.get(row, 0) < min_per_row and len(selected) < count:
                added_for_row = False
                for bucket in sorted(
                    row_buckets,
                    key=lambda b: (
                        sum(
                            1
                            for selected_item in selected
                            if int(selected_item.get("row", 0)) == row
                            and _direction_bucket_deg(
                                selected_item.get("world_azimuth_deg", selected_item.get("azimuth_deg", 0.0)),
                                bucket_width,
                            )
                            == b
                        ),
                        bucket_counts.get(b, 0),
                        b,
                    ),
                ):
                    candidates = [
                        item
                        for item in unique
                        if int(item.get("row", 0)) == row
                        and _direction_bucket_deg(
                            item.get("world_azimuth_deg", item.get("azimuth_deg", 0.0)),
                            bucket_width,
                        )
                        == bucket
                        and _planned_pose_key(item) not in selected_keys
                    ]
                    if not candidates:
                        continue
                    candidates.sort(key=lambda item: float(item.get("plan_joint_delta_deg", 999.0)))
                    if add_item(candidates[0]):
                        row_counts[row] = row_counts.get(row, 0) + 1
                        added_for_row = True
                        break
                if not added_for_row:
                    break
            if row_counts.get(row, 0) < min_per_row:
                print(
                    f"[plan][warn] height layer row={row} reserved only "
                    f"{row_counts.get(row, 0)}/{min_per_row} usable poses"
                )

    # First balance by azimuth direction.  Within each direction bucket, cycle
    # through elevation rows so each direction gets side/mid/top variation.
    while len(selected) < count:
        added_this_round = False
        for bucket in sorted(buckets, key=lambda b: (bucket_counts.get(b, 0), b)):
            if len(selected) >= count:
                break
            row_order = sorted(
                rows,
                key=lambda row: (
                    sum(
                        1
                        for selected_item in selected
                        if int(selected_item.get("row", 0)) == row
                        and _direction_bucket_deg(
                            selected_item.get("world_azimuth_deg", selected_item.get("azimuth_deg", 0.0)),
                            bucket_width,
                        )
                        == bucket
                    ),
                    -row,
                ),
            )
            for row in row_order:
                candidates = [
                    item
                    for item in unique
                    if int(item.get("row", 0)) == row
                    and _direction_bucket_deg(
                        item.get("world_azimuth_deg", item.get("azimuth_deg", 0.0)),
                        bucket_width,
                    )
                    == bucket
                    and _planned_pose_key(item) not in selected_keys
                ]
                if not candidates:
                    continue
                candidates.sort(key=lambda item: float(item.get("plan_joint_delta_deg", 999.0)))
                added_this_round = add_item(candidates[0]) or added_this_round
                break
        if not added_this_round:
            break

    min_topdown = min(max(int(min_topdown_count), 0), count)
    if min_topdown > 0:
        topdown_items = [item for item in unique if bool(item.get("is_topdown", False))]
        topdown_buckets = sorted(
            {
                _direction_bucket_deg(
                    item.get("world_azimuth_deg", item.get("azimuth_deg", 0.0)),
                    bucket_width,
                )
                for item in topdown_items
            }
        )
        for bucket in topdown_buckets:
            if len(selected) >= count:
                break
            if len([item for item in selected if bool(item.get("is_topdown", False))]) >= min_topdown:
                break
            candidates = [
                item
                for item in topdown_items
                if _direction_bucket_deg(
                    item.get("world_azimuth_deg", item.get("azimuth_deg", 0.0)),
                    bucket_width,
                )
                == bucket
            ]
            if not candidates:
                continue
            candidates.sort(key=lambda item: float(item.get("plan_joint_delta_deg", 999.0)))
            add_item(candidates[0])

    # Safety net if some buckets are sparse.
    for row in rows:
        for bucket in buckets:
            if len(selected) >= count:
                break
            candidates = [
                item
                for item in unique
                if int(item.get("row", 0)) == row
                and _direction_bucket_deg(
                    item.get("world_azimuth_deg", item.get("azimuth_deg", 0.0)),
                    bucket_width,
                )
                == bucket
            ]
            if not candidates:
                continue
            candidates.sort(key=lambda item: float(item.get("plan_joint_delta_deg", 999.0)))
            add_item(candidates[0])
        if len(selected) >= count:
            break

    # Fill remaining slots with the easier plans while preserving uniqueness.
    for item in sorted(unique, key=lambda item: float(item.get("plan_joint_delta_deg", 999.0))):
        if len(selected) >= count:
            break
        add_item(item)

    rows_used = sorted({int(item.get("row", 0)) for item in selected})
    buckets_used = sorted(
        {
            _direction_bucket_deg(
                item.get("world_azimuth_deg", item.get("azimuth_deg", 0.0)),
                bucket_width,
            )
            for item in selected
        }
    )
    final_bucket_counts = {
        bucket: sum(
            1
            for item in selected
            if _direction_bucket_deg(
                item.get("world_azimuth_deg", item.get("azimuth_deg", 0.0)),
                bucket_width,
            )
            == bucket
        )
        for bucket in buckets_used
    }
    topdown_count = sum(1 for item in selected if bool(item.get("is_topdown", False)))
    elevations_used = sorted({round(float(item.get("elevation_deg", 0.0)), 1) for item in selected})
    row_bucket_counts = {}
    for item in selected:
        row = int(item.get("row", 0))
        bucket = _direction_bucket_deg(
            item.get("world_azimuth_deg", item.get("azimuth_deg", 0.0)),
            bucket_width,
        )
        row_bucket_counts[(row, bucket)] = row_bucket_counts.get((row, bucket), 0) + 1
    row_bucket_text = {f"r{row}/b{bucket}": count for (row, bucket), count in sorted(row_bucket_counts.items())}
    print(
        f"[plan] diverse selected={len(selected)} rows={rows_used} "
        f"elevations={elevations_used} topdown={topdown_count} "
        f"bucket_width={bucket_width:.1f} direction_counts={final_bucket_counts}"
    )
    print(f"[plan] selected row_direction_counts={row_bucket_text}")
    return selected


def _joint_distance(q_a, q_b):
    q_a = np.asarray(q_a, dtype=float).reshape(7)
    q_b = np.asarray(q_b, dtype=float).reshape(7)
    delta = (q_b - q_a + np.pi) % (2.0 * np.pi) - np.pi
    return float(np.max(np.abs(delta)))


def _order_planned_poses_by_joint_distance(arm, planned):
    remaining = list(planned)
    if len(remaining) <= 1:
        return remaining
    current_q = np.asarray(arm.get_joint_angles(), dtype=float).reshape(7)
    ordered = []
    max_step = 0.0
    while remaining:
        best_idx = min(
            range(len(remaining)),
            key=lambda idx: _joint_distance(current_q, remaining[idx]["joint_target"]),
        )
        item = remaining.pop(best_idx)
        step = _joint_distance(current_q, item["joint_target"])
        max_step = max(max_step, step)
        ordered.append(item)
        current_q = np.asarray(item["joint_target"], dtype=float).reshape(7)
    print(f"[plan] reordered poses by joint distance; max_neighbor_delta={math.degrees(max_step):.1f}deg")
    return ordered


def _sleep_with_message(seconds, message):
    seconds = float(seconds)
    if seconds <= 0.0:
        return
    print(f"{message} ({seconds:.1f}s)")
    time.sleep(seconds)


def _camera_relative_summary(items, target_base, label):
    if not items:
        print(f"[plan] {label}: no camera poses")
        return
    target = np.asarray(target_base, dtype=float).reshape(3)
    rels = []
    elevations = []
    azimuths = []
    for item in items:
        T_base_camera = item.get("T_base_camera")
        if T_base_camera is None:
            continue
        cam_pos = np.asarray(T_base_camera[:3, 3], dtype=float).reshape(3)
        rel = cam_pos - target
        xy = float(np.linalg.norm(rel[:2]))
        dist = float(np.linalg.norm(rel))
        elev = math.degrees(math.atan2(float(rel[2]), xy)) if dist > 1e-9 else 0.0
        az = math.degrees(math.atan2(float(rel[1]), float(rel[0]))) if xy > 1e-9 else 0.0
        rels.append(rel)
        elevations.append(elev)
        azimuths.append(az)
    if not rels:
        print(f"[plan] {label}: no camera transforms")
        return
    rel_arr = np.asarray(rels, dtype=float)
    print(
        f"[plan] {label} camera rel xyz range: "
        f"x=[{rel_arr[:,0].min():+.3f},{rel_arr[:,0].max():+.3f}] "
        f"y=[{rel_arr[:,1].min():+.3f},{rel_arr[:,1].max():+.3f}] "
        f"z=[{rel_arr[:,2].min():+.3f},{rel_arr[:,2].max():+.3f}] "
        f"elev=[{min(elevations):.1f},{max(elevations):.1f}]deg "
        f"az=[{min(azimuths):+.1f},{max(azimuths):+.1f}]deg"
    )


def _select_planned_poses(arm, args, target_base, T_flange_camera):
    if args.pose_seeds_json:
        return _build_seed_planned_poses(arm, args)
    per_base_count = _candidate_pose_count(target_base, args)
    if not args.auto_base_azimuth:
        args._last_plan_generated = per_base_count
        planned, skipped = _build_planned_poses(arm, args, target_base, T_flange_camera)
        planned = _select_diverse_planned_poses(
            planned,
            args.count,
            args.min_topdown_count,
            args.direction_bucket_width_deg,
            args.min_per_height_layer,
        )
        return _order_planned_poses_by_joint_distance(arm, planned), skipped

    scores = []
    best = None
    trial_records = []
    candidates = _base_azimuth_candidates(args)
    print(f"[plan] preplanning {len(candidates)} base azimuths x {per_base_count} poses = {len(candidates) * per_base_count} IK candidates")
    for azimuth_deg in candidates:
        trial_args = SimpleNamespace(**vars(args))
        trial_args.base_azimuth_deg = float(azimuth_deg)
        planned, skipped = _build_planned_poses(arm, trial_args, target_base, T_flange_camera)
        _print_plan_quality(planned, skipped, trial_args, f"base_az={azimuth_deg:+.1f}")
        trial_records.append((float(azimuth_deg), planned, skipped))
        scores.append((float(azimuth_deg), len(planned), len(skipped)))
        if best is None or len(planned) > len(best[1]):
            best = (float(azimuth_deg), planned, skipped)

    print("[plan] auto base azimuth scores:", ", ".join(f"{az:+.1f}:{usable}" for az, usable, _ in scores))
    args._last_plan_generated = per_base_count * len(trial_records)
    if best is None:
        return [], []

    if len(best[1]) < int(args.count) and trial_records:
        merged = []
        used_keys = set()
        skipped_all = []
        for _, _, skipped in trial_records:
            skipped_all.extend(skipped)
        for _, planned, _ in sorted(trial_records, key=lambda rec: (-len(rec[1]), abs(rec[0] - best[0]))):
            for item in planned:
                key = _planned_pose_key(item)
                if key in used_keys:
                    continue
                used_keys.add(key)
                merged.append(item)
        if len(merged) > len(best[1]):
            selected_bases = sorted({round(float(item.get("base_azimuth_deg", best[0])), 3) for item in merged})
            print(f"[plan] merged usable poses from base azimuths={selected_bases} usable={len(merged)}")
            if len(merged) >= int(args.count):
                args.base_azimuth_deg = float(best[0])
                _camera_relative_summary(merged, target_base, "usable merged")
                merged = _select_diverse_planned_poses(
                    merged,
                    args.count,
                    args.min_topdown_count,
                    args.direction_bucket_width_deg,
                    args.min_per_height_layer,
                )
                return _order_planned_poses_by_joint_distance(arm, merged), skipped_all

    args.base_azimuth_deg = float(best[0])
    print(f"[plan] selected base_azimuth_deg={args.base_azimuth_deg:+.1f} usable={len(best[1])}")
    _camera_relative_summary(best[1], target_base, "usable best")
    selected = _select_diverse_planned_poses(
        best[1],
        args.count,
        args.min_topdown_count,
        args.direction_bucket_width_deg,
        args.min_per_height_layer,
    )
    return _order_planned_poses_by_joint_distance(arm, selected), best[2]


def run(args):
    os.makedirs(args.save_dir, exist_ok=True)
    T_flange_camera = handeye_transform(args.calib_json, args.handeye_convention)
    pose_publisher = None
    with NeroArm(
        args.nero_interface,
        args.nero_channel,
        args.nero_bitrate,
        args.nero_firmware,
        args.nero_enable_timeout,
        args.nero_speed_percent,
    ) as arm:
        try:
            arm.set_tcp_offset(args.tcp_offset)
            if args.handeye_pose_pub_topic:
                pose_publisher = _HandeyePosePublisher(args.handeye_pose_pub_topic, args.handeye_pose_pub_hz)
            if args.pose_seeds_json:
                target_base = None
                print(f"[seed_plan] using pose seeds: {args.pose_seeds_json}")
            elif args.target_base is not None:
                target_base = np.asarray(args.target_base, dtype=float).reshape(3)
            elif args.click_target:
                target_base = _click_target_base(arm, args, T_flange_camera)
            else:
                raise ValueError("Provide --target-base X Y Z, or use --click-target to click marker center first.")

            if target_base is not None:
                print("[target] final target_base =", np.round(target_base, 6).tolist())
                _apply_current_view_azimuth_preference(arm, args, target_base, T_flange_camera)
            planned, skipped = _select_planned_poses(arm, args, target_base, T_flange_camera)
            generated_default = len(planned) if args.pose_seeds_json else _candidate_pose_count(target_base, args)
            print(f"[plan] generated={int(getattr(args, '_last_plan_generated', generated_default))} usable={len(planned)} skipped={len(skipped)}")
            for item, reason in skipped[:20]:
                if args.pose_seeds_json:
                    print(
                        f"[plan][skip] idx={int(item.get('index', -1)) + 1:02d} "
                        f"seed={item.get('seed_name', '?')} variant={item.get('variant', item.get('variant_index', '?'))} reason={reason}"
                    )
                else:
                    print(f"[plan][skip] idx={item['index'] + 1:02d} az={item['azimuth_deg']:+.1f} height={item['height']:.3f} reason={reason}")
            if len(planned) < int(args.count):
                raise PlanNotExecutable(
                    f"Only {len(planned)} usable poses; need {int(args.count)}. "
                    "Adjust --base-azimuth-deg/--base-azimuth-candidates or workspace limits."
                )
            planned = planned[: int(args.count)]
            if target_base is not None:
                _camera_relative_summary(planned, target_base, "final selected")

            log = {
                "timestamp": datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
                "target_base": None if target_base is None else target_base.tolist(),
                "pose_seeds_json": args.pose_seeds_json,
                "calib_json": args.calib_json,
                "handeye_convention": args.handeye_convention,
                "base_azimuth_deg": float(args.base_azimuth_deg),
                "handeye_pose_pub_topic": args.handeye_pose_pub_topic,
                "poses": [],
            }
            log_path = os.path.join(args.save_dir, f"nero_handeye_pose_sweep_{log['timestamp']}.json")

            print("\n[run] Start pose sweep. At each pose:")
            print("[run]   1. Confirm the ArUco is visible.")
            print("[run]   2. Capture one sample in your hand-eye calibration node.")
            print("[run]   3. Press Enter here to move to the next pose.")
            print("[run] Type q then Enter at a prompt to stop.")

            for i, item in enumerate(planned, start=1):
                label = f"pose_{i:02d}"
                if item.get("source") == "seed_joint":
                    print(
                        f"\n[{label}] seed={item.get('seed_name')} variant={item.get('variant_index')} "
                        f"plan_joint_delta={float(item.get('plan_joint_delta_deg', 0.0)):.1f}deg "
                        f"offset_deg={np.round(item.get('joint_offset_deg', []), 2).tolist()}"
                    )
                else:
                    print(
                        f"\n[{label}] base_az={float(item.get('base_azimuth_deg', args.base_azimuth_deg)):+.1f}deg "
                        f"rel_az={item['azimuth_deg']:+.1f}deg world_az={float(item.get('world_azimuth_deg', item['azimuth_deg'])):+.1f}deg "
                        f"elev={float(item.get('elevation_deg', 0.0)):.1f}deg dist={float(item.get('distance', 0.0)):.3f}m "
                        f"radius={item['radius']:.3f}m height={item['height']:.3f}m "
                        f"roll={item['roll_deg']:+.1f}deg topdown={bool(item.get('is_topdown', False))}"
                    )
                ans = input(f"[{label}] Press Enter to move, or q to stop: ").strip().lower()
                if ans == "q":
                    break
                try:
                    if item.get("source") == "seed_joint":
                        actual_pose, ok = _move_seed_joint_pose(arm, item, args, label)
                    else:
                        actual_pose, ok = _move_flange_with_correction(arm, item["flange_pose"], args, label)
                except KeyboardInterrupt:
                    print(f"[{label}] Ctrl+C received; requesting electronic emergency stop.")
                    safe_stop(arm, "pose sweep interrupted")
                    raise
                if pose_publisher is not None:
                    pose_publisher.set_pose(actual_pose)
                    print(f"[{label}] publishing actual flange pose for hand-eye capture on {pose_publisher.topic}")
                    _sleep_with_message(
                        args.pose_publish_settle_sec,
                        f"[{label}] waiting for initial hand-eye pose messages to publish",
                    )

                marker_visible = None
                marker_msg = "not checked"
                if args.verify_aruco:
                    try:
                        bgr, _, _, _ = capture_rgbd(args)
                        marker_visible, marker_msg = _try_detect_aruco(bgr, args.aruco_dict, args.aruco_id)
                        print(f"[{label}] aruco visible={marker_visible} ({marker_msg})")
                    except Exception as exc:
                        marker_visible = False
                        marker_msg = f"{type(exc).__name__}: {exc}"
                        print(f"[{label}][warn] aruco check failed: {marker_msg}")

                log["poses"].append(
                    {
                        "label": label,
                        "source": item.get("source", "spherical_camera"),
                        "seed_name": item.get("seed_name"),
                        "seed_index": item.get("seed_index"),
                        "variant_index": item.get("variant_index"),
                        "joint_offset_deg": item.get("joint_offset_deg"),
                        "target_joints_deg": None if "joint_target" not in item else np.rad2deg(np.asarray(item["joint_target"], dtype=float)).tolist(),
                        "azimuth_deg": item.get("azimuth_deg"),
                        "base_azimuth_deg": item.get("base_azimuth_deg"),
                        "world_azimuth_deg": item.get("world_azimuth_deg"),
                        "radius": item.get("radius"),
                        "distance": item.get("distance"),
                        "elevation_deg": item.get("elevation_deg"),
                        "is_topdown": item.get("is_topdown"),
                        "height": item.get("height"),
                        "roll_deg": item.get("roll_deg"),
                        "target_flange_pose": np.asarray(item["flange_pose"], dtype=float).tolist(),
                        "target_camera_position": None if item.get("T_base_camera") is None else np.asarray(item["T_base_camera"][:3, 3], dtype=float).tolist(),
                        "target_camera_rel_to_marker": (
                            None
                            if item.get("T_base_camera") is None or target_base is None
                            else (
                                np.asarray(item["T_base_camera"][:3, 3], dtype=float).reshape(3)
                                - np.asarray(target_base, dtype=float).reshape(3)
                            ).tolist()
                        ),
                        "actual_flange_pose": np.asarray(actual_pose, dtype=float).tolist(),
                        "reached_after_correction": bool(ok),
                        "aruco_visible": marker_visible,
                        "aruco_msg": marker_msg,
                    }
                )
                with open(log_path, "w", encoding="utf-8") as f:
                    json.dump(log, f, indent=2)
                print(f"[{label}] log updated: {log_path}")

                sample_ok = bool(ok) and (marker_visible is not False)
                if not sample_ok and not args.allow_unreached_or_invisible_sample:
                    reasons = []
                    if not ok:
                        reasons.append("pose not reached")
                    if marker_visible is False:
                        reasons.append("ArUco not visible")
                    print(f"[{label}][skip] do NOT capture calibration sample: {', '.join(reasons)}")
                    ans = input(f"[{label}] Press Enter to skip to next pose, or q to stop: ").strip().lower()
                    if ans == "q":
                        break
                    continue

                if i == len(planned):
                    ans = input(
                        f"[{label}] Capture the FINAL calibration sample now. "
                        "Confirm the SDK sample and pose log are saved, then Enter to finish; q to stop: "
                    ).strip().lower()
                    if pose_publisher is not None:
                        _sleep_with_message(
                            args.final_publish_hold_sec,
                            f"[{label}] keeping final hand-eye pose alive before shutdown",
                        )
                else:
                    ans = input(f"[{label}] Capture calibration sample now, then press Enter for next pose; q to stop: ").strip().lower()
                if ans == "q":
                    break
            print(f"[done] wrote {log_path}")
        finally:
            if pose_publisher is not None:
                pose_publisher.close()


def parse_args():
    p = argparse.ArgumentParser(description="Nero hand-eye calibration observation sweep.")
    p.add_argument("--calib-json", required=True, help="Existing approximate hand-eye json used only to aim the camera at the marker")
    p.add_argument("--handeye-convention", default="flange_camera_xyzw", choices=["flange_camera_xyzw", "camera_flange_xyzw", "flange_camera_wxyz", "camera_flange_wxyz"])
    p.add_argument("--target-base", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"), help="Marker/table center in robot base frame, meters")
    p.add_argument("--click-target", action="store_true", help="Capture current RGB-D frame and click marker center to estimate --target-base")
    p.add_argument("--count", type=int, default=32)
    p.add_argument("--pose-seeds-json", default=None, help="Use manually recorded Nero pose seeds and generate nearby joint-space observation poses")
    p.add_argument("--seed-include-invisible", action="store_true", help="Use seed records even if their recorded ArUco visibility was false")
    p.add_argument(
        "--seed-joint-offsets-deg",
        default=(
            "0,0,0,0,0,0,0;"
            "10,0,0,0,0,0,0;-10,0,0,0,0,0,0;"
            "0,10,0,0,0,0,0;0,-10,0,0,0,0,0;"
            "0,0,10,0,0,0,0;0,0,-10,0,0,0,0;"
            "0,0,0,10,0,0,0;0,0,0,-10,0,0,0;"
            "0,0,0,0,0,10,0;0,0,0,0,0,-10,0;"
            "0,0,0,0,0,0,12;0,0,0,0,0,0,-12;"
            "7,7,0,0,0,0,0;-7,-7,0,0,0,0,0;"
            "0,0,7,7,0,0,0;0,0,-7,-7,0,0,0"
        ),
        help="Semicolon-separated 7-joint offset rows in degrees applied around each seed",
    )
    p.add_argument("--seed-execute-originals-first", dest="seed_execute_originals_first", action="store_true", help="Select all original seed poses before nearby perturbations")
    p.add_argument("--no-seed-execute-originals-first", dest="seed_execute_originals_first", action="store_false")
    p.set_defaults(seed_execute_originals_first=True)
    p.add_argument("--seed-max-joint-offset-deg", type=float, default=12.0, help="Reject seed joint perturbations larger than this per joint")
    p.add_argument("--base-azimuth-deg", type=float, default=180.0, help="Central approach direction around marker in base XY plane")
    p.add_argument("--base-azimuth-candidates", default="0,45,-45", help="Comma-separated central azimuths to try when --include-global-azimuth-fallback is enabled")
    p.add_argument("--orbit-azimuths-deg", default="0,25,-25,35,-35,45,-45", help="Comma-separated orbit azimuths around each central direction; defaults mirror grasp active-view angles")
    p.add_argument("--roll-offsets-deg", default="0,-20,20", help="Comma-separated optical-axis roll offsets to try for each camera position")
    p.add_argument(
        "--sphere-view-specs",
        default="0.50:30:-16,0.48:42:8,0.46:54:16,0.44:64:-8,0.42:72:20",
        help="Comma-separated spherical view layers as DISTANCE_M:ELEVATION_DEG:ROLL_DEG",
    )
    p.add_argument("--topdown-elevation-threshold-deg", type=float, default=60.0, help="Candidate elevations at or above this are treated as top-down views")
    p.add_argument("--min-topdown-count", type=int, default=0, help="Optional minimum number of top-down-biased poses to reserve after direction balancing")
    p.add_argument("--min-per-height-layer", type=int, default=4, help="Reserve at least this many selected poses from each spherical height/elevation layer when possible")
    p.add_argument("--direction-bucket-width-deg", type=float, default=30.0, help="Azimuth bucket width used to balance the final selected poses by direction")
    p.add_argument("--prefer-current-view-azimuth", dest="prefer_current_view_azimuth", action="store_true", help="Use the current camera/initial grasp view direction as the center of azimuth candidates")
    p.add_argument("--no-prefer-current-view-azimuth", dest="prefer_current_view_azimuth", action="store_false")
    p.set_defaults(prefer_current_view_azimuth=True)
    p.add_argument("--current-view-azimuth-offsets-deg", default="0,15,-15,45,60,75", help="Offsets around the current camera azimuth when --prefer-current-view-azimuth is enabled")
    p.add_argument("--include-global-azimuth-fallback", dest="include_global_azimuth_fallback", action="store_true", help="Also try --base-azimuth-candidates after the current-view local azimuths; slower but broader")
    p.add_argument("--no-include-global-azimuth-fallback", dest="include_global_azimuth_fallback", action="store_false")
    p.set_defaults(include_global_azimuth_fallback=False)
    p.add_argument("--auto-base-azimuth", dest="auto_base_azimuth", action="store_true", help="Preplan several central azimuths and use the one with the most executable poses")
    p.add_argument("--no-auto-base-azimuth", dest="auto_base_azimuth", action="store_false", help="Use only --base-azimuth-deg")
    p.set_defaults(auto_base_azimuth=True)
    p.add_argument("--save-dir", default="handeye_pose_sweep_logs")
    p.add_argument("--handeye-pose-log", dest="handeye_pose_pub_topic", default=None, help="Save actual flange poses to a new JSONL file; no ROS publisher")
    p.add_argument("--handeye-pose-pub-hz", type=float, default=10.0)
    p.add_argument("--pose-publish-settle-sec", type=float, default=0.3, help="Short delay after each move so hand-eye pose topic publishes a few samples before capture")
    p.add_argument("--final-publish-hold-sec", type=float, default=5.0, help="Keep publishing the final pose for a short time before shutting down")

    p.add_argument("--orbbec-color-topic", default="/camera/color/image_raw")
    p.add_argument("--orbbec-depth-topic", default="/camera/depth/image_raw")
    p.add_argument("--orbbec-camera-info-topic", default="/camera/color/camera_info")
    p.add_argument("--orbbec-depth-scale", type=float, default=None)
    p.add_argument("--orbbec-timeout", type=float, default=5.0)
    p.add_argument("--orbbec-warmup-frames", type=int, default=5)
    p.add_argument("--rgb", default=None)
    p.add_argument("--depth", default=None)
    p.add_argument("--depth-scale", type=float, default=None)
    p.add_argument("--fx", type=float, default=None)
    p.add_argument("--fy", type=float, default=None)
    p.add_argument("--cx", type=float, default=None)
    p.add_argument("--cy", type=float, default=None)
    p.add_argument("--depth-radius", type=int, default=3)
    p.add_argument("--verify-aruco", action="store_true", help="After each move, capture one frame and check that ArUco is visible")
    p.add_argument("--aruco-dict", default="DICT_4X4_50")
    p.add_argument("--aruco-id", type=int, default=None)
    p.add_argument("--allow-unreached-or-invisible-sample", action="store_true", help="Allow manual capture prompt even when the pose was not reached or ArUco verification failed")

    p.add_argument("--nero-interface", default="socketcan")
    p.add_argument("--nero-channel", default="can0")
    p.add_argument("--nero-bitrate", type=int, default=1000000)
    p.add_argument("--nero-firmware", default="V111", choices=["DEFAULT", "V111", "V112"])
    p.add_argument("--nero-enable-timeout", type=float, default=5.0)
    p.add_argument("--nero-speed-percent", type=int, default=2)
    p.add_argument("--tcp-offset", type=float, nargs=6, default=DEFAULT_TCP_OFFSET)
    p.add_argument("--initial-joints-deg", type=float, nargs=7, default=DEFAULT_INITIAL_JOINT_DEG)

    p.add_argument("--ik-urdf", default=DEFAULT_IK_URDF)
    p.add_argument("--ik-base-link", default="base_link")
    p.add_argument("--ik-tip-link", default="link7")
    p.add_argument("--ik-timeout", type=float, default=0.02)
    p.add_argument("--ik-epsilon", type=float, default=0.00001)
    p.add_argument("--ik-solver-type", default="Distance", choices=["Speed", "Distance", "Manip1", "Manip2"])
    p.add_argument("--ik-max-joint-delta", type=float, default=None)
    p.add_argument("--ik-joint7-weight", type=float, default=0.35)
    p.add_argument("--j2-side", default="any", choices=["any", "positive", "negative", "opposite-current"], help="Restrict planned IK solutions by joint-2 side; use positive to force the other side when current selected J2 is negative")
    p.add_argument("--j2-side-deadband-deg", type=float, default=3.0, help="Minimum absolute J2 angle used when checking --j2-side")
    p.add_argument("--j2-seed-degs", default="35,55,75,95,115,-35,-55,-75", help="Extra J2 seed angles for finding the requested --j2-side IK branch")
    p.add_argument("--max-plan-joint-delta-deg", type=float, default=125.0)

    p.add_argument("--min-x", type=float, default=-0.85)
    p.add_argument("--max-x", type=float, default=0.25)
    p.add_argument("--min-y", type=float, default=-0.60)
    p.add_argument("--max-y", type=float, default=0.60)
    p.add_argument("--min-z", type=float, default=0.18)
    p.add_argument("--max-z", type=float, default=0.75)

    p.add_argument("--move-wait-timeout", type=float, default=18.0)
    p.add_argument("--move-poll-interval", type=float, default=0.2)
    p.add_argument("--move-pos-tol", type=float, default=0.012)
    p.add_argument("--move-rpy-tol", type=float, default=np.deg2rad(4.0))
    p.add_argument("--move-joint-tol", type=float, default=np.deg2rad(3.0))
    p.add_argument("--correction-attempts", type=int, default=1)
    p.add_argument("--correct-pos-tol", type=float, default=0.006)
    p.add_argument("--correct-rpy-tol-deg", type=float, default=3.0)
    from cameras.options import add_camera_arguments
    add_camera_arguments(p)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
