#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

"""Check hand-eye calibration consistency with live ArUco preview.

The marker absolute base-frame position is not required. Each S/Enter key press
captures one RGB-D frame, detects the same ArUco marker center, transforms it
to robot base using the current flange pose and hand-eye calibration, then
reports how tightly all predicted base-frame points cluster.
The preview displays marker IDs, selected center depth, and saved sample count.
"""

import argparse
import json
import math
import os
import select
from datetime import datetime

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R

from cameras.realsense import camera_for, close_all, depth_with_current_support
from cameras.realsense import capture_rgbd as capture_sdk_rgbd
from tools.verify_handeye_point import (
    capture_ros_rgbd_frame,
    load_calibration,
    make_T,
    inv_T,
    pixel_to_camera,
    robust_depth_at,
    transform_point,
)
from tools.calibrate_d435i_charuco import NeroReadOnlyFeedback


def _handeye_T(calib_json, convention):
    pos, quat = load_calibration(calib_json)
    convention = str(convention)
    if convention.endswith("wxyz"):
        quat_xyzw = np.array([quat[1], quat[2], quat[3], quat[0]], dtype=float)
    else:
        quat_xyzw = np.asarray(quat, dtype=float).reshape(4)
    T = make_T(R.from_quat(quat_xyzw).as_matrix(), pos)
    if convention.startswith("camera_flange"):
        return inv_T(T)
    return T


def _marker_detector(dictionary_name):
    if not hasattr(cv2, "aruco"):
        raise RuntimeError("cv2.aruco unavailable; install/use opencv-contrib-python")
    aruco = cv2.aruco
    dict_id = getattr(aruco, dictionary_name, None)
    if dict_id is None:
        raise ValueError(f"unknown aruco dictionary: {dictionary_name}")
    dictionary = aruco.getPredefinedDictionary(dict_id)
    return aruco.ArucoDetector(dictionary, aruco.DetectorParameters())


def _detect_aruco_center(bgr, dictionary_name, marker_id=None, *, detector=None):
    if detector is None:
        detector = _marker_detector(dictionary_name)
    corners, ids, _ = detector.detectMarkers(bgr)
    if ids is None or len(ids) == 0:
        raise RuntimeError("no ArUco marker detected")

    ids_flat = np.asarray(ids, dtype=int).reshape(-1)
    if marker_id is None:
        pick = 0
    else:
        matches = np.where(ids_flat == int(marker_id))[0]
        if matches.size == 0:
            raise RuntimeError(f"target marker id={int(marker_id)} absent; detected ids={ids_flat.tolist()}")
        pick = int(matches[0])

    pts = np.asarray(corners[pick], dtype=float).reshape(-1, 2)
    center = pts.mean(axis=0)
    return float(center[0]), float(center[1]), int(ids_flat[pick]), corners, ids


def _stats(points):
    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    centroid = pts.mean(axis=0)
    residual = pts - centroid.reshape(1, 3)
    residual_norm = np.linalg.norm(residual, axis=1)
    max_pairwise = 0.0
    if pts.shape[0] > 1:
        for i in range(pts.shape[0]):
            d = np.linalg.norm(pts[i + 1 :] - pts[i].reshape(1, 3), axis=1)
            if d.size:
                max_pairwise = max(max_pairwise, float(d.max()))
    return {
        "count": int(pts.shape[0]),
        "centroid": centroid,
        "axis_std": pts.std(axis=0),
        "residual_rms": float(math.sqrt(float(np.mean(residual_norm**2)))) if residual_norm.size else 0.0,
        "residual_mean": float(residual_norm.mean()) if residual_norm.size else 0.0,
        "residual_max": float(residual_norm.max()) if residual_norm.size else 0.0,
        "max_pairwise": float(max_pairwise),
    }


def _pnp_marker_center(corners, ids, marker_id, intr, length_m):
    """Independent RGB estimate; requires the actual printed marker side length."""
    index = int(np.flatnonzero(np.asarray(ids).reshape(-1) == marker_id)[0])
    pixels = np.asarray(corners[index], dtype=float).reshape(4, 2)
    half = float(length_m) / 2
    obj = np.array([[-half, half, 0], [half, half, 0],
                    [half, -half, 0], [-half, -half, 0]], dtype=float)
    K = np.array([[intr.fx, 0, intr.cx], [0, intr.fy, intr.cy], [0, 0, 1]], float)
    ok, _, tvec = cv2.solvePnP(obj, pixels, K, np.zeros(5), flags=cv2.SOLVEPNP_IPPE_SQUARE)
    point = np.asarray(tvec, float).reshape(3) if ok else None
    if point is None or not np.isfinite(point).all() or point[2] <= 0:
        raise RuntimeError("RGB marker PnP failed")
    return point


def _print_stats(points):
    if not points:
        return
    st = _stats(points)
    print(
        "[stats] "
        f"n={st['count']} "
        f"centroid={np.round(st['centroid'], 6).tolist()} "
        f"axis_std_mm={np.round(st['axis_std'] * 1000.0, 2).tolist()} "
        f"rms_mm={st['residual_rms'] * 1000.0:.2f} "
        f"max_to_mean_mm={st['residual_max'] * 1000.0:.2f} "
        f"max_pairwise_mm={st['max_pairwise'] * 1000.0:.2f}"
    )


def _jsonable_sample(sample):
    out = dict(sample)
    for key in ("P_camera", "P_base", "flange_pose", "P_camera_pnp", "P_base_pnp"):
        if key in out and out[key] is not None:
            out[key] = np.asarray(out[key], dtype=float).tolist()
    return out


def parse_args():
    p = argparse.ArgumentParser(description="Repeated ArUco hand-eye consistency check without known marker base position.")
    p.add_argument("--calib-json", required=True)
    p.add_argument(
        "--handeye-convention",
        default="flange_camera_xyzw",
        choices=["flange_camera_xyzw", "camera_flange_xyzw", "flange_camera_wxyz", "camera_flange_wxyz"],
    )
    p.add_argument("--aruco-dict", default="DICT_4X4_1000")
    p.add_argument("--aruco-id", type=int, default=None)
    p.add_argument("--marker-length-mm", type=float, default=None,
                   help="Actual printed marker side, excluding white border; also compare RGB PnP against depth")
    p.add_argument("--samples", type=int, default=0, help="Stop after this many samples; 0 means manual stop with q")
    p.add_argument("--no-preview", dest="preview", action="store_false",
                   help="Disable the default live camera/ArUco preview; use terminal Enter only")
    p.add_argument("--save-dir", default="handeye_aruco_consistency_logs")
    p.add_argument("--orbbec-color-topic", default="/camera/color/image_raw")
    p.add_argument("--orbbec-depth-topic", default="/camera/depth/image_raw")
    p.add_argument("--orbbec-camera-info-topic", default="/camera/color/camera_info")
    p.add_argument("--orbbec-depth-scale", type=float, default=None)
    p.add_argument("--orbbec-timeout", type=float, default=5.0)
    p.add_argument("--orbbec-warmup-frames", type=int, default=5)
    p.add_argument("--depth-radius", type=int, default=4)
    p.add_argument("--nero-interface", default="socketcan")
    p.add_argument("--nero-channel", default="can0")
    p.add_argument("--nero-bitrate", type=int, default=1000000)
    p.add_argument("--nero-firmware", default="V111", choices=["DEFAULT", "V111", "V112"])
    p.add_argument("--nero-enable-timeout", type=float, default=5.0)
    p.add_argument("--nero-speed-percent", type=int, default=3)
    from cameras.options import add_camera_arguments, apply_pipeline_defaults
    add_camera_arguments(p)
    apply_pipeline_defaults(p)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_path = os.path.join(args.save_dir, f"handeye_aruco_consistency_{timestamp}.json")
    T_flange_camera = _handeye_T(args.calib_json, args.handeye_convention)
    samples = []
    locked_id = args.aruco_id
    if args.nero_interface != "socketcan" or args.nero_bitrate != 1000000:
        raise ValueError("Read-only feedback currently supports socketcan at 1000000 bit/s")
    marker_length = getattr(args, "marker_length_mm", None)
    if marker_length is not None and (not np.isfinite(marker_length) or marker_length <= 0):
        raise ValueError("Marker length must be finite and positive")
    detector = _marker_detector(args.aruco_dict)
    preview = getattr(args, "preview", True)
    window = "ArUco RGB-D check | S/Enter save | Q/Esc exit"
    capture_status = "Stop and settle before saving. No robot motion commands."

    print("[nero] read-only feedback connection; no enable, motion or gripper commands.")
    try:
        with NeroReadOnlyFeedback(
            args.nero_channel,
            args.nero_firmware,
            args.nero_enable_timeout,
        ) as arm:
            print("[run] Move the robot to a view where the marker is visible.")
            if preview:
                camera = camera_for(args)
                cv2.namedWindow(window, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(window, 1100, 700)
                print("[run] Live preview: marker outlines/IDs and selected center depth.")
                print("[run] Window S/Enter or terminal Enter=record; Q/Esc/close window=finish.")
            else:
                print("[run] Press Enter to record one sample; type q then Enter to finish.")
            while True:
                if int(args.samples) > 0 and len(samples) >= int(args.samples):
                    break
                if preview:
                    frame = camera.read()
                    vis = frame.bgr.copy()
                    preview_corners, preview_ids, _ = detector.detectMarkers(frame.bgr)
                    seen = [] if preview_ids is None else np.asarray(preview_ids).reshape(-1).tolist()
                    active_id = (seen[0] if seen else None) if locked_id is None else locked_id
                    state, color = "NO MARKERS DETECTED", (0, 0, 255)
                    if seen:
                        cv2.aruco.drawDetectedMarkers(vis, preview_corners, preview_ids)
                        if active_id not in seen:
                            state, color = f"Target ID {active_id} missing; seen={seen}", (0, 180, 255)
                        else:
                            index = seen.index(active_id)
                            u, v = np.asarray(preview_corners[index]).reshape(4, 2).mean(axis=0)
                            x, y = int(round(u)), int(round(v))
                            radius = int(args.depth_radius)
                            current_depth = depth_with_current_support(frame.depth_m, frame.raw_depth_m, 0., np.inf)
                            patch = current_depth[max(0,y-radius):min(current_depth.shape[0],y+radius+1),
                                                  max(0,x-radius):min(current_depth.shape[1],x+radius+1)]
                            values = patch[np.isfinite(patch) & (patch > 0)]
                            color = (0, 220, 0) if values.size else (0, 180, 255)
                            state = (f"ID {active_id}: depth {np.median(values)*1000:.1f} mm" if values.size
                                     else f"ID {active_id}: RGB detected, DEPTH MISSING")
                            cv2.drawMarker(vis, (x, y), color, cv2.MARKER_CROSS, 24, 2)
                    cv2.rectangle(vis, (0, 0), (vis.shape[1], 112), (20, 20, 20), -1)
                    limit = str(args.samples) if args.samples > 0 else 'manual'
                    cv2.putText(vis, f"Saved {len(samples)}/{limit} | locked ID: {locked_id} | S/Enter save, Q/Esc exit",
                                (12, 26), cv2.FONT_HERSHEY_SIMPLEX, .6, (255,255,255), 2)
                    cv2.putText(vis, state, (12, 58), cv2.FONT_HERSHEY_SIMPLEX, .65, color, 2)
                    cv2.putText(vis, capture_status[:115], (12, 91), cv2.FONT_HERSHEY_SIMPLEX, .48, (255,255,255), 1)
                    cv2.imshow(window, vis)
                    key = cv2.waitKey(1) & 255
                    if key in (27, ord('q'), ord('Q')) or cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                        break
                    save = key in (10, 13, ord('s'), ord('S'))
                    if select.select([sys.stdin], [], [], 0)[0]:
                        line = sys.stdin.readline()
                        if not line or line.strip().lower() == 'q':
                            break
                        save = save or not line.strip()
                    if not save:
                        continue
                else:
                    ans = input(f"[sample_{len(samples) + 1:02d}] Enter=record, q=finish: ").strip().lower()
                    if ans == "q":
                        break

                try:
                    _, before, before_stamp = arm.read()
                    bgr, depth_m, intr, out_dir = capture_sdk_rgbd(args)
                    flange_pose, after, after_stamp = arm.read()
                    delta = np.linalg.inv(before) @ after
                    if (after_stamp < before_stamp or np.linalg.norm(delta[:3, 3]) > .001
                            or R.from_matrix(delta[:3, :3]).magnitude() > np.deg2rad(.3)):
                        raise RuntimeError("Robot moved or feedback clock changed during RGB-D capture; retry stationary")
                    u, v, detected_id, corners, ids = _detect_aruco_center(
                        bgr, args.aruco_dict, locked_id, detector=detector)
                    depth = robust_depth_at(depth_m, u, v, args.depth_radius)
                except (RuntimeError, ValueError) as exc:
                    print(f"[sample_{len(samples) + 1:02d}][skip] {type(exc).__name__}: {exc}")
                    capture_status = f"NOT SAVED: {exc}"
                    continue

                if locked_id is None:
                    locked_id = detected_id
                    print(f"[aruco] locked marker id={locked_id}; subsequent samples must use this same marker")
                flange_pose = np.asarray(flange_pose, dtype=float).reshape(6)
                print("[nero] flange_pose =", np.round(flange_pose, 6).tolist())
                T_base_flange = make_T(R.from_euler("xyz", np.asarray(flange_pose[3:], dtype=float)).as_matrix(), flange_pose[:3])
                P_camera = pixel_to_camera(u, v, depth, intr)
                P_base = transform_point(T_base_flange @ T_flange_camera, P_camera)

                vis = bgr.copy()
                cv2.aruco.drawDetectedMarkers(vis, corners, ids)
                cv2.drawMarker(vis, (int(round(u)), int(round(v))), (0, 0, 255), cv2.MARKER_CROSS, 24, 2)
                if out_dir is not None:
                    cv2.imwrite(os.path.join(out_dir, "aruco_center.png"), vis)

                sample = {
                    "index": len(samples) + 1,
                    "marker_id": int(detected_id),
                    "pixel": [float(u), float(v)],
                    "depth_m": float(depth),
                    "P_camera": P_camera,
                    "P_base": P_base,
                    "flange_pose": np.asarray(flange_pose, dtype=float),
                    "capture_dir": out_dir,
                    "feedback_stamps": [float(before_stamp), float(after_stamp)],
                }
                if marker_length is not None:
                    try:
                        pnp = _pnp_marker_center(corners, ids, locked_id, intr, marker_length / 1000)
                        sample["P_camera_pnp"] = pnp
                        sample["P_base_pnp"] = transform_point(T_base_flange @ T_flange_camera, pnp)
                        print("[rgb-vs-depth] camera XYZ depth-minus-PnP mm =",
                              np.round((P_camera - pnp) * 1000, 2).tolist())
                    except (RuntimeError, cv2.error) as exc:
                        print(f"[pnp][skip] {exc}")
                samples.append(sample)

                print(
                    f"[sample_{sample['index']:02d}] id={detected_id} "
                    f"uv={[round(u, 2), round(v, 2)]} depth={depth:.4f} "
                    f"P_base={np.round(P_base, 6).tolist()}"
                )
                _print_stats([s["P_base"] for s in samples])

                log = {
                    "timestamp": timestamp,
                    "calib_json": args.calib_json,
                    "handeye_convention": args.handeye_convention,
                    "aruco_dict": args.aruco_dict,
                    "aruco_id": locked_id,
                    "marker_length_mm": marker_length,
                    "robot_commands_sent": False,
                    "samples": [_jsonable_sample(s) for s in samples],
                }
                if samples:
                    st = _stats([s["P_base"] for s in samples])
                    log["stats"] = {
                        "count": st["count"],
                        "centroid": st["centroid"].tolist(),
                        "axis_std_m": st["axis_std"].tolist(),
                        "residual_rms_m": st["residual_rms"],
                        "residual_mean_m": st["residual_mean"],
                        "residual_max_m": st["residual_max"],
                        "max_pairwise_m": st["max_pairwise"],
                    }
                pnp_points = [s["P_base_pnp"] for s in samples if "P_base_pnp" in s]
                if pnp_points:
                    print("[rgb-pnp consistency; marker size must be correct]")
                    _print_stats(pnp_points)
                    st = _stats(pnp_points)
                    log["rgb_pnp_stats"] = dict(count=st["count"],
                                                residual_rms_m=st["residual_rms"],
                                                max_pairwise_m=st["max_pairwise"])
                with open(log_path, "w", encoding="utf-8") as f:
                    json.dump(log, f, indent=2)
                print(f"[log] updated {log_path}")
                capture_status = f"Saved sample {len(samples)}: ID {detected_id}, depth {depth*1000:.1f} mm"
    except KeyboardInterrupt:
        print("\n[run] Stopped; previously recorded samples are retained.")
    finally:
        close_all()
        if preview:
            cv2.destroyAllWindows()

    print("[done]")
    _print_stats([s["P_base"] for s in samples])
    print(f"[done] log={log_path}" if samples else "[done] No samples recorded.")


if __name__ == "__main__":
    main()
