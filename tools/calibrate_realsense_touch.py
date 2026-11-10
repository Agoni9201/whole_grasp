#!/usr/bin/env python3
"""Read-only TCP touch / SDK RGB-D hand-eye calibration, without robot motion.

P_parent = T_parent_child @ P_child. All saved positions are metres. The
camera coordinates are the production SDK's aligned color optical frame,
not native depth optical coordinates or the experimental ChArUco K/D.
"""

import argparse
import hashlib
import json
import sys
import tempfile
import time
import uuid
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

CAMERA_FRAME = "d435_color_optical_frame"
PROJECTION = "sdk_aligned_color_pinhole"
TCP_OFFSET_M = [0.165, 0.0, -0.025]


def now():
    return datetime.now().astimezone().isoformat()


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value, *, new=True):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    if new:
        with path.open("x", encoding="utf-8") as stream:
            stream.write(payload)
    else:
        # Only progress files in a newly created session directory use replace.
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, delete=False
        ) as stream:
            temporary = Path(stream.name)
            stream.write(payload)
        temporary.replace(path)


def check_transform(value):
    transform = np.asarray(value, dtype=float)
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise ValueError("Invalid finite 4x4 transform")
    rotation = transform[:3, :3]
    if (not np.allclose(transform[3], [0, 0, 0, 1], atol=1e-8)
            or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6)
            or not np.isclose(np.linalg.det(rotation), 1, atol=1e-6)):
        raise ValueError("Transform must be rigid, right-handed SE(3)")
    return transform


def point3(value):
    point = np.asarray(value, dtype=float)
    if point.shape != (3,) or not np.isfinite(point).all():
        raise ValueError("Expected a finite XYZ vector")
    return point


def transform_points(transform, points):
    transform = check_transform(transform)
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("Expected finite Nx3 points")
    return points @ transform[:3, :3].T + transform[:3, 3]


def tcp_position(transform, offset=TCP_OFFSET_M):
    return transform_points(transform, [point3(offset)])[0]


def check_geometry(points):
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 3:
        raise ValueError("Need at least three non-collinear 3D points")
    if not np.isfinite(points).all():
        raise ValueError("Points must be finite")
    spread = np.linalg.svd(points - points.mean(axis=0), compute_uv=False)
    if spread[0] < 0.02 or spread[1] < 0.005 or spread[1] / spread[0] < 0.05:
        raise ValueError("Points are too clustered or nearly collinear; spread the marks")
    return (spread / np.sqrt(len(points)) * 1000).tolist()


def fit_rigid(camera_points, flange_points):
    source, target = np.asarray(camera_points, float), np.asarray(flange_points, float)
    if source.shape != target.shape:
        raise ValueError("Corresponding point shapes differ")
    check_geometry(source)
    check_geometry(target)
    src_center, dst_center = source.mean(axis=0), target.mean(axis=0)
    u, _, vt = np.linalg.svd((source - src_center).T @ (target - dst_center))
    correction = np.eye(3)
    correction[2, 2] = np.linalg.det(vt.T @ u.T)
    rotation = vt.T @ correction @ u.T
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = dst_center - rotation @ src_center
    return check_transform(transform)


def load_references(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if (data.get("kind") != "tcp_touch_references" or data.get("schema_version") != 1
            or not data.get("complete") or data.get("unit") != "metre"):
        raise ValueError("References are incomplete or have an unsupported schema")
    point3(data["tcp_offset_m"])
    by_id = {}
    for row in data["points"]:
        identity = str(row["point_id"])
        if not identity or identity in by_id or row["role"] not in ("fit", "validation"):
            raise ValueError("Duplicate/invalid point ID or role")
        point3(row["P_base_m"])
        by_id[identity] = row
    if sum(row["role"] == "fit" for row in by_id.values()) < 6:
        raise ValueError("Need at least six fit reference points")
    if sum(row["role"] == "validation" for row in by_id.values()) < 2:
        raise ValueError("Reserve at least two validation points before collection")
    check_geometry([row["P_base_m"] for row in by_id.values() if row["role"] == "fit"])
    return data, by_id


def check_camera_profile(profile):
    if (profile.get("frame") != CAMERA_FRAME or profile.get("projection") != PROJECTION
            or not profile.get("serial") or min(profile["width"], profile["height"], profile["fps"]) <= 0):
        raise ValueError("Invalid SDK color optical camera profile")
    k, d = np.asarray(profile["K"], float), np.asarray(profile["D"], float)
    if (k.shape != (3, 3) or d.shape != (5,) or not np.isfinite(k).all()
            or not np.isfinite(d).all() or np.any(np.abs(d) > 1e-9)
            or min(k[0, 0], k[1, 1]) <= 0
            or not np.allclose(k[2], [0, 0, 1])
            or not np.allclose([k[0, 1], k[1, 0]], 0)
            or not 0 <= k[0, 2] < profile["width"]
            or not 0 <= k[1, 2] < profile["height"]):
        raise ValueError("Unsupported/invalid SDK pinhole K/D; do not inject candidate RGB intrinsics")


def same_camera(first, second):
    check_camera_profile(first)
    check_camera_profile(second)
    keys = ("serial", "width", "height", "fps", "frame", "projection")
    if (any(first[k] != second[k] for k in keys)
            or not np.allclose(first["K"], second["K"], rtol=0, atol=1e-6)
            or not np.allclose(first["D"], second["D"], rtol=0, atol=1e-9)):
        raise ValueError("Camera serial/profile/K/D mismatch; no implicit rescaling")


def load_views(paths, reference_path):
    _, references = load_references(reference_path)
    views, captures, images = [], set(), set()
    for path in paths:
        path = Path(path)
        view = json.loads(path.read_text(encoding="utf-8"))
        if (view.get("kind") != "tcp_touch_observations" or view.get("schema_version") != 1
                or not view.get("complete") or view.get("unit") != "metre"
                or view.get("references_sha256") != sha256(reference_path)):
            raise ValueError("Incomplete observations or different touch references")
        check_transform(view["T_base_flange"])
        check_camera_profile(view["camera"])
        capture_id, image_hash = view["capture_id"], view["image_sha256"]
        if not capture_id or not image_hash or capture_id in captures or image_hash in images:
            raise ValueError("Duplicate/missing capture or image identity")
        captures.add(capture_id)
        images.add(image_hash)
        if sha256(path.parent / "color.png") != image_hash:
            raise ValueError("Saved image SHA-256 mismatch")
        observed = set()
        for row in view["points"]:
            identity = str(row["point_id"])
            if identity not in references or identity in observed:
                raise ValueError("Unknown or duplicate observed point ID")
            observed.add(identity)
            if point3(row["P_camera_m"])[2] <= 0:
                raise ValueError("Camera point must have positive depth")
        if not observed:
            raise ValueError("No observed points")
        if views:
            same_camera(views[0]["camera"], view["camera"])
        views.append(view)
    if not views:
        raise ValueError("Need observations")
    return references, views


def residual_report(references, views, flange_camera):
    rows, by_point = [], {}
    for view in views:
        transform = check_transform(view["T_base_flange"]) @ check_transform(flange_camera)
        for observation in view["points"]:
            identity = str(observation["point_id"])
            predicted = transform_points(transform, [observation["P_camera_m"]])[0]
            reference = point3(references[identity]["P_base_m"])
            delta = (predicted - reference) * 1000
            rows.append(dict(
                capture_id=view["capture_id"], point_id=identity,
                role=references[identity]["role"], error_xyz_mm=delta.tolist(),
                error_mm=float(np.linalg.norm(delta)), P_base_predicted_m=predicted.tolist(),
            ))
            by_point.setdefault(identity, []).append(predicted)
    summary = {}
    for role in ("fit", "validation"):
        values = np.array([row["error_mm"] for row in rows if row["role"] == role])
        summary[role] = None if not len(values) else dict(
            observations=len(values), mean_mm=float(values.mean()),
            median_mm=float(np.median(values)), rmse_mm=float(np.sqrt(np.mean(values ** 2))),
            max_mm=float(values.max()),
        )
    spread = {}
    for identity, values in by_point.items():
        values = np.asarray(values)
        spread[identity] = dict(
            views=len(values), xyz_range_mm=(np.ptp(values, axis=0) * 1000).tolist(),
            max_pair_distance_mm=(None if len(values) < 2 else float(
                np.linalg.norm(values[:, None] - values[None, :], axis=2).max() * 1000)),
        )
    return dict(summary=summary, points=rows, cross_view_spread=spread)


def solve_calibration(reference_path, observation_paths):
    references, views = load_views(observation_paths, reference_path)
    fit_ids = {identity for identity, row in references.items() if row["role"] == "fit"}
    source, target = [], []
    for view in views:
        seen = {str(row["point_id"]) for row in view["points"]}
        if not fit_ids <= seen:
            raise ValueError("Each solving view must observe all fit points")
        inverse = np.linalg.inv(check_transform(view["T_base_flange"]))
        for observation in view["points"]:
            identity = str(observation["point_id"])
            if identity not in fit_ids:
                continue
            source.append(observation["P_camera_m"])
            target.append(transform_points(inverse, [references[identity]["P_base_m"]])[0])
    transform = fit_rigid(source, target)
    position = transform[:3, 3].tolist()
    orientation = Rotation.from_matrix(transform[:3, :3]).as_quat().tolist()
    return dict(
        schema_version=1, kind="tcp_touch_handeye", created_at=now(),
        status="candidate_not_approved_for_motion", method="rigid_3d3d_scale_fixed_one",
        transform="camera_to_flange", parent_frame="flange", child_frame=CAMERA_FRAME,
        unit="metre", matrix_4x4=transform.tolist(), position=position, orientation=orientation,
        translation_m=position, quaternion_xyzw=orientation,
        flange_camera_xyzw=position + orientation, camera=views[0]["camera"],
        tcp_offset_m=json.loads(Path(reference_path).read_text())["tcp_offset_m"],
        references_sha256=sha256(reference_path), fit_point_ids=sorted(fit_ids),
        validation_point_ids=sorted(set(references) - fit_ids),
        training_capture_ids=[view["capture_id"] for view in views],
        training_image_sha256=[view["image_sha256"] for view in views],
        training_observations=[str(Path(path).resolve()) for path in observation_paths],
        camera_point_spread_mm=check_geometry(source),
        report=residual_report(references, views, transform),
        note="Held-out point residuals in solving images are not independent-view validation. "
             "No scale, RGB intrinsics, robot geometry, or runtime defaults were modified.",
    )


def load_calibration(path, reference_path):
    calibration = json.loads(Path(path).read_text(encoding="utf-8"))
    _, references = load_references(reference_path)
    if (calibration.get("kind") != "tcp_touch_handeye" or calibration.get("schema_version") != 1
            or calibration.get("unit") != "metre" or calibration.get("transform") != "camera_to_flange"
            or calibration.get("parent_frame") != "flange" or calibration.get("child_frame") != CAMERA_FRAME
            or calibration.get("references_sha256") != sha256(reference_path)):
        raise ValueError("Wrong touch calibration, frame convention or references")
    fit_ids = {identity for identity, row in references.items() if row["role"] == "fit"}
    if set(calibration["fit_point_ids"]) != fit_ids:
        raise ValueError("Calibration fit-point split mismatch")
    if set(calibration["validation_point_ids"]) != set(references) - fit_ids:
        raise ValueError("Calibration validation-point split mismatch")
    recorded = json.loads(Path(reference_path).read_text(encoding="utf-8"))
    if not np.allclose(point3(calibration["tcp_offset_m"]), point3(recorded["tcp_offset_m"]), atol=1e-9, rtol=0):
        raise ValueError("Calibration physical TCP offset mismatch")
    transform = check_transform(calibration["matrix_4x4"])
    if (not np.allclose(transform[:3, 3], point3(calibration["position"]))
            or not np.allclose(transform[:3, :3], Rotation.from_quat(calibration["orientation"]).as_matrix())):
        raise ValueError("Calibration matrix and legacy position/orientation disagree")
    check_camera_profile(calibration["camera"])
    return calibration, transform


def validate_calibration(reference_path, observation_paths, calibration_path):
    references, views = load_views(observation_paths, reference_path)
    calibration, transform = load_calibration(calibration_path, reference_path)
    for view in views:
        same_camera(calibration["camera"], view["camera"])
        if (view["capture_id"] in calibration["training_capture_ids"]
                or view["image_sha256"] in calibration["training_image_sha256"]):
            raise ValueError("Validation requires independent captures, not solving images")
        if sum(references[str(row["point_id"])]["role"] == "validation" for row in view["points"]) < 2:
            raise ValueError("Each validation view needs at least two held-out points")
    return dict(
        created_at=now(), kind="tcp_touch_validation", calibration_sha256=sha256(calibration_path),
        independent_images=True, independent_point_ids=calibration["validation_point_ids"],
        report=residual_report(references, views, transform),
        status="measurement_only_not_motion_authorization",
    )


def check_stationary(transforms, max_translation_m=0.001, max_rotation_deg=0.3):
    transforms = [check_transform(value) for value in transforms]
    if len(transforms) < 2:
        raise ValueError("Need at least two fresh flange samples")
    inverse = np.linalg.inv(transforms[0])
    for transform in transforms[1:]:
        delta = inverse @ transform
        if (np.linalg.norm(delta[:3, 3]) > max_translation_m
                or np.rad2deg(Rotation.from_matrix(delta[:3, :3]).magnitude()) > max_rotation_deg):
            raise ValueError("Wrist moved during sampling; stop, settle and retry")


def stationary_feedback(reader, count=10, timeout=3.0):
    transforms, stamps = [], []
    deadline = time.monotonic() + timeout
    while len(transforms) < count and time.monotonic() < deadline:
        _, transform, stamp = reader.read()
        if not stamps or stamp > stamps[-1]:
            transforms.append(check_transform(transform))
            stamps.append(float(stamp))
        time.sleep(0.025)
    if len(transforms) < max(2, count):
        raise ValueError("Flange feedback did not advance enough; no point recorded")
    check_stationary(transforms)
    return transforms, stamps


def feedback_after(reader, stamp, timeout=0.2):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = reader.read()
        if np.isfinite(result[2]) and result[2] > stamp:
            return result
        time.sleep(0.005)
    raise ValueError("Flange feedback did not advance across the RGB-D frame; retry")


def sample_depth_point(depth, intrinsics, pixel, radius=2, min_depth=0.15, max_depth=1.5):
    depth = np.asarray(depth)
    k = np.asarray(intrinsics, dtype=float)
    uv = np.asarray(pixel, dtype=float)
    if (depth.ndim != 2 or k.shape != (3, 3) or not np.isfinite(k).all()
            or min(k[0, 0], k[1, 1]) <= 0 or uv.shape != (2,) or not np.isfinite(uv).all()
            or radius < 0 or int(radius) != radius or not 0 < min_depth < max_depth):
        raise ValueError("Invalid depth image/intrinsics/pixel/options")
    u, v = np.rint(uv).astype(int)
    h, w = depth.shape
    if not (radius <= u < w - radius and radius <= v < h - radius):
        raise ValueError("Click is outside the image or too near its border")
    patch = depth[v-radius:v+radius+1, u-radius:u+radius+1]
    valid = patch[np.isfinite(patch) & (patch >= min_depth) & (patch <= max_depth)]
    if valid.size < max(1, int(np.ceil(patch.size * 0.6))):
        raise ValueError("Too little valid depth at cross; do not fill holes")
    z = float(np.median(valid))
    mad = float(np.median(np.abs(valid - z)))
    if mad > 0.002 or float(np.ptp(valid)) > 0.010:
        raise ValueError("Depth patch is noisy or crosses a surface edge; choose a flat opaque surface")
    point = np.array([(u-k[0, 2])*z/k[0, 0], (v-k[1, 2])*z/k[1, 1], z])
    return point, dict(pixel=[int(u), int(v)], depth_m=z, valid_pixels=int(valid.size),
                       patch_pixels=int(patch.size), depth_mad_mm=mad*1000)


def read_only_reader(args):
    from tools.calibrate_d435i_charuco import NeroReadOnlyFeedback
    return NeroReadOnlyFeedback(args.channel, args.firmware, args.feedback_timeout)


def confirmed_input(prompt):
    answer = input(prompt).strip().lower()
    if answer in ("q", "quit"):
        raise KeyboardInterrupt("user cancelled")
    if answer:
        raise ValueError("Press Enter to confirm, or q to quit")


def run_touch(args):
    identities = [str(value) for value in args.point_ids]
    held_out = set(args.validation_ids)
    if (len(set(identities)) != len(identities) or not held_out <= set(identities)
            or len(held_out) < 2 or len(identities) - len(held_out) < 6):
        raise ValueError("Use unique IDs, at least six fit and two held-out validation points")
    offset = point3(np.asarray(args.tcp_offset_mm) / 1000)
    args.output.mkdir(parents=True, exist_ok=False)
    path = args.output / "references.json"
    data = dict(schema_version=1, kind="tcp_touch_references", unit="metre", created_at=now(),
                session_id=uuid.uuid4().hex, tcp_offset_m=offset.tolist(), complete=False, points=[])
    write_json(path, data)
    print("只读：程序不会使能、切模式、设置 TCP 或发送运动。请自行进入安全示教状态。")
    print("所有叉叉从触碰到视觉验收必须固定；尖点轻触，不要压弯纸面或推动方块。")
    with read_only_reader(args) as reader:
        for identity in identities:
            role = "validation" if identity in held_out else "fit"
            while True:
                try:
                    confirmed_input(f"尖点已轻触叉叉 {identity} [{role}]，停稳后 Enter；q 退出：")
                    transforms, stamps = stationary_feedback(reader)
                    positions = np.array([tcp_position(value, offset) for value in transforms])
                    center = np.median(positions, axis=0)
                    spread = float(np.linalg.norm(positions - center, axis=1).max())
                    if spread > 0.001:
                        raise ValueError("TCP moved more than 1 mm during this touch")
                    data["points"].append(dict(
                        point_id=identity, role=role, created_at=now(), P_base_m=center.tolist(),
                        sample_spread_mm=spread*1000, flange_timestamps=stamps,
                        T_base_flange_samples=[value.tolist() for value in transforms],
                    ))
                    write_json(path, data, new=False)
                    print(f"叉叉 {identity}: XYZ_mm={np.round(center*1000, 2).tolist()}，采样波动={spread*1000:.2f} mm")
                    break
                except (ValueError, RuntimeError) as exc:
                    print("未记录，请重试：", exc)
    check_geometry([row["P_base_m"] for row in data["points"] if row["role"] == "fit"])
    data["complete"] = True
    data["completed_at"] = now()
    write_json(path, data, new=False)
    print("触点坐标：", path)


def capture_still_rgbd(camera, reader, args):
    from tools.verify_handeye_point import median_valid_depth_frames

    stationary_feedback(reader)
    # Discard buffered images only after the operator confirms a settled view.
    for _ in range(10):
        camera.read()
    frames, transforms, stamps = [], [], []
    for _ in range(args.depth_frames):
        before = reader.read()
        frame = camera.read()
        after = feedback_after(reader, before[2])
        transforms.extend([before[1], after[1]])
        stamps.extend([before[2], after[2]])
        if frames and frame.sequence <= frames[-1].sequence:
            raise ValueError("Repeated or backward SDK depth frame sequence")
        if frames and (not np.allclose(frame.K, frames[0].K) or not np.allclose(frame.D, frames[0].D)):
            raise ValueError("Camera intrinsics changed during capture")
        if frame.depth_m is None or frame.depth_m.shape != frame.bgr.shape[:2]:
            raise ValueError("SDK depth must be aligned to the color image")
        frames.append(frame)
    check_stationary(transforms)
    depth, _ = median_valid_depth_frames(
        [frame.depth_m for frame in frames], min_valid_frames=(args.depth_frames+1)//2,
        max_mad=0.002,
    )
    frame = frames[-1]
    profile = dict(serial=frame.serial, width=frame.bgr.shape[1], height=frame.bgr.shape[0],
                   fps=args.fps, frame=CAMERA_FRAME, projection=PROJECTION,
                   K=frame.K.tolist(), D=frame.D.tolist())
    check_camera_profile(profile)
    # The last image is bracketed by these feedback poses; not a claim of hardware clock sync.
    transform = np.eye(4)
    transform[:3, :3] = Rotation.from_matrix([t[:3, :3] for t in transforms[-2:]]).mean().as_matrix()
    transform[:3, 3] = np.mean([t[:3, 3] for t in transforms[-2:]], axis=0)
    return frame.bgr, depth, profile, transform, dict(
        depth_sequences=[int(frame.sequence) for frame in frames],
        hardware_ms=[float(frame.hardware_ms) for frame in frames],
        received_monotonic=[float(frame.received_monotonic) for frame in frames],
        flange_timestamps=stamps, depth_frames=args.depth_frames,
        synchronization="stationary_feedback_bracketing_not_hardware_clock_sync",
    )


def collect_observation(args, reader, references, identities):
    import cv2
    from cameras.realsense import RealSenseCamera
    from tools.verify_handeye_point import read_clicked_pixel

    confirmed_input("确认所有叉叉仍固定；Enter 打开预览，q 退出：")
    with RealSenseCamera(
        serial=args.serial, width=args.width, height=args.height, fps=args.fps,
        timeout=args.camera_timeout, warmup=30, enable_depth=True,
        depth_preset="high_density", emitter_enabled=args.emitter_enabled, laser_power=args.laser_power,
    ) as camera:
        window = "SDK touch calibration: S=freeze settled view, Q=quit"
        print("预览中自行示教调整相机，确保所需叉叉均可见；停稳后在窗口按 s 冻结。")
        try:
            while True:
                preview = camera.read().bgr.copy()
                cv2.putText(preview, "Stop and settle, S=capture / Q=quit", (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
                cv2.imshow(window, preview)
                key = cv2.waitKey(20) & 0xFF
                if key in (27, ord("q")) or cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                    raise KeyboardInterrupt("camera preview cancelled")
                if key == ord("s"):
                    try:
                        bgr, depth, profile, transform, metadata = capture_still_rgbd(camera, reader, args)
                    except (ValueError, RuntimeError) as exc:
                        print("未冻结，请停稳后按 s 重试：", exc)
                        continue
                    break
        finally:
            cv2.destroyAllWindows()
    # Camera is closed; clicking refers to the saved frozen frame and its saved flange pose.
    if args.expected_profile is not None:
        same_camera(args.expected_profile, profile)
    args.output.mkdir(parents=True, exist_ok=False)
    if not cv2.imwrite(str(args.output / "color.png"), bgr):
        raise OSError("Cannot save color frame")
    np.save(args.output / "depth_m.npy", depth, allow_pickle=False)
    data = dict(
        schema_version=1, kind="tcp_touch_observations", unit="metre", created_at=now(),
        capture_id=uuid.uuid4().hex, references_sha256=sha256(args.references),
        camera=profile, T_base_flange=transform.tolist(), capture=metadata,
        image_sha256=sha256(args.output / "color.png"), complete=False, points=[],
    )
    path = args.output / "observations.json"
    write_json(path, data)
    visual = bgr.copy()
    try:
        for identity in identities:
            while True:
                image = visual.copy()
                label = f"Click cross {identity} [{references[identity]['role']}], Enter=accept, Q=quit"
                cv2.putText(image, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
                print(label)
                pixel = read_clicked_pixel(image)
                try:
                    point, stats = sample_depth_point(
                        depth, profile["K"], pixel, radius=args.depth_radius,
                        min_depth=args.min_depth, max_depth=args.max_depth,
                    )
                except ValueError as exc:
                    print("深度不合格，请重新点击：", exc)
                    continue
                data["points"].append(dict(point_id=identity, P_camera_m=point.tolist(), **stats))
                write_json(path, data, new=False)
                cv2.drawMarker(visual, tuple(stats["pixel"]), (0, 0, 255), cv2.MARKER_CROSS, 14, 1)
                cv2.putText(visual, identity, tuple(stats["pixel"]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
                print(f"叉叉 {identity} camera XYZ_mm={np.round(point*1000, 2).tolist()}")
                break
    finally:
        cv2.destroyAllWindows()
    data["complete"] = True
    write_json(path, data, new=False)
    print("视觉观测：", path)
    return data


def run_observe(args):
    _, references = load_references(args.references)
    identities = args.point_ids or list(references)
    if len(set(identities)) != len(identities) or not set(identities) <= set(references):
        raise ValueError("Unknown or duplicate point IDs")
    args.expected_profile = None
    with read_only_reader(args) as reader:
        collect_observation(args, reader, references, identities)


def hover_flange_pose(predicted_base, observation_flange, tcp_offset, hover_m):
    if not np.isfinite(hover_m) or hover_m < 0.03:
        raise ValueError("Read-only hover preview must remain at least 30 mm above predicted point")
    transform = check_transform(observation_flange).copy()
    transform[:3, 3] = point3(predicted_base) + [0, 0, hover_m] - transform[:3, :3] @ point3(tcp_offset)
    return transform


def run_verify(args):
    data, references = load_references(args.references)
    calibration, flange_camera = load_calibration(args.calibration, args.references)
    if args.point_id not in references or references[args.point_id]["role"] != "validation":
        raise ValueError("Physical verification must use a held-out validation cross")
    args.expected_profile = calibration["camera"]
    if args.hover_mm < 30:
        raise ValueError("Hover preview must be at least 30 mm; no motion is sent")
    with read_only_reader(args) as reader:
        view = collect_observation(args, reader, references, [args.point_id])
        if view["image_sha256"] in calibration["training_image_sha256"]:
            raise ValueError("Verification requires a new capture, not a solving image")
        point = view["points"][0]["P_camera_m"]
        predicted = transform_points(check_transform(view["T_base_flange"]) @ flange_camera, [point])[0]
        reference = point3(references[args.point_id]["P_base_m"])
        error = (predicted - reference) * 1000
        hover = hover_flange_pose(predicted, view["T_base_flange"], data["tcp_offset_m"], args.hover_mm/1000)
        result = dict(
            created_at=now(), kind="tcp_touch_physical_verification", point_id=args.point_id,
            calibration_sha256=sha256(args.calibration), P_base_predicted_m=predicted.tolist(),
            P_base_reference_m=reference.tolist(), error_xyz_mm=error.tolist(),
            error_norm_mm=float(np.linalg.norm(error)),
            hover_tcp_xyz_m=(predicted + [0, 0, args.hover_mm/1000]).tolist(),
            hover_flange_pose_m_rad=np.r_[hover[:3, 3], Rotation.from_matrix(hover[:3, :3]).as_euler("xyz")].tolist(),
            hover_collision_checked=False, motion_sent=False, physical_touch_confirmed=False,
        )
        path = args.output / "verification.json"
        write_json(path, result)
        print("视觉预测 XYZ_mm：", np.round(predicted*1000, 2).tolist())
        print("视觉 - 触点参考 XYZ_mm：", np.round(error, 2).tolist(),
              f"；空间误差={np.linalg.norm(error):.2f} mm")
        print("以上悬停位姿仅供核对，没有 IK/碰撞预检，不是可直接执行的运动指令。")
        print("程序不会运动。不要因屏幕坐标而盲目下压；自行在安全示教状态下轻触实际叉叉。")
        answer = input("尖点确已轻触实际叉叉并停稳，输入 TOUCH 记录；Enter 跳过，q 退出：").strip()
        if answer != "TOUCH":
            print("未记录实际触碰。结果：", path)
            return
        transforms, stamps = stationary_feedback(reader)
        positions = np.array([tcp_position(t, data["tcp_offset_m"]) for t in transforms])
        actual = np.median(positions, axis=0)
        if np.linalg.norm(positions - actual, axis=1).max() > 0.001:
            raise ValueError("TCP was not stationary during physical verification")
        result.update(
            physical_touch_confirmed=True, touched_at=now(), actual_touch_base_m=actual.tolist(),
            actual_touch_flange_timestamps=stamps,
            visual_minus_actual_touch_xyz_mm=((predicted-actual)*1000).tolist(),
            visual_minus_actual_touch_norm_mm=float(np.linalg.norm(predicted-actual)*1000),
            retouch_minus_reference_xyz_mm=((actual-reference)*1000).tolist(),
        )
        write_json(path, result, new=False)
        print("视觉 - 实际轻触 XYZ_mm：", np.round((predicted-actual)*1000, 2).tolist(),
              f"；空间误差={np.linalg.norm(predicted-actual)*1000:.2f} mm")
        print("实际轻触 - 原参考 XYZ_mm：", np.round((actual-reference)*1000, 2).tolist())
        print("验证结果：", path)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    def feedback_options(sub):
        sub.add_argument("--channel", default="can0")
        sub.add_argument("--firmware", choices=["DEFAULT", "V111", "V112"], default="V111")
        sub.add_argument("--feedback-timeout", type=float, default=5.0)

    def observation_options(sub):
        feedback_options(sub)
        sub.add_argument("--references", type=Path, required=True)
        sub.add_argument("--output", type=Path, required=True, help="New session directory")
        sub.add_argument("--serial", required=True)
        sub.add_argument("--width", type=int, default=1280)
        sub.add_argument("--height", type=int, default=720)
        sub.add_argument("--fps", type=int, default=30)
        sub.add_argument("--camera-timeout", type=float, default=10)
        sub.add_argument("--depth-frames", type=int, default=15)
        sub.add_argument("--depth-radius", type=int, default=2)
        sub.add_argument("--min-depth", type=float, default=0.15)
        sub.add_argument("--max-depth", type=float, default=1.5)
        sub.add_argument("--laser-power", type=float, default=None)
        sub.add_argument("--emitter-enabled", action="store_true", default=None)

    touch = commands.add_parser("touch", help="Record fixed crosses using the known physical TCP")
    feedback_options(touch)
    touch.add_argument("--output", type=Path, required=True)
    touch.add_argument("--point-ids", nargs="+", default=list("12345678"))
    touch.add_argument("--validation-ids", nargs="+", default=["7", "8"])
    touch.add_argument("--tcp-offset-mm", type=float, nargs=3, default=[165, 0, -25])
    touch.set_defaults(run=run_touch)

    observe = commands.add_parser("observe", help="Freeze SDK RGB-D and click matching cross IDs")
    observation_options(observe)
    observe.add_argument("--point-ids", nargs="+", default=None)
    observe.set_defaults(run=run_observe)

    solve = commands.add_parser("solve", help="Offline rigid 3D-3D solve, scale fixed at one")
    solve.add_argument("--references", type=Path, required=True)
    solve.add_argument("--observations", type=Path, nargs="+", required=True)
    solve.add_argument("--output", type=Path, required=True)
    solve.set_defaults(run=run_solve)

    validate = commands.add_parser("validate", help="Offline independent-view and held-out-point validation")
    validate.add_argument("--references", type=Path, required=True)
    validate.add_argument("--observations", type=Path, nargs="+", required=True)
    validate.add_argument("--calibration", type=Path, required=True)
    validate.add_argument("--output", type=Path, required=True)
    validate.set_defaults(run=run_validate)

    verify = commands.add_parser("verify", help="Read-only camera cross check, then manual physical re-touch")
    observation_options(verify)
    verify.add_argument("--calibration", type=Path, required=True)
    verify.add_argument("--point-id", default="7")
    verify.add_argument("--hover-mm", type=float, default=30)
    verify.set_defaults(run=run_verify)
    return parser


def run_solve(args):
    result = solve_calibration(args.references, args.observations)
    write_json(args.output, result)
    print(json.dumps(result["report"]["summary"], indent=2))
    print("候选外参：", args.output, "；尚未授权接入运动。请采新视角 validate / verify。")


def run_validate(args):
    result = validate_calibration(args.references, args.observations, args.calibration)
    write_json(args.output, result)
    print(json.dumps(result["report"], indent=2))
    print("仅报告误差，不自动判定安全或覆盖项目参数：", args.output)


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error(f"输出已存在，不覆盖：{args.output}；请选择新的目录/文件名")
    if hasattr(args, "feedback_timeout") and (not np.isfinite(args.feedback_timeout) or args.feedback_timeout <= 0):
        parser.error("Feedback timeout must be finite and positive")
    if hasattr(args, "hover_mm") and (not np.isfinite(args.hover_mm) or args.hover_mm < 30):
        parser.error("Hover preview must be finite and at least 30 mm")
    if hasattr(args, "depth_frames"):
        if (not 3 <= args.depth_frames <= 61 or args.depth_radius < 0
                or not 0 < args.min_depth < args.max_depth
                or min(args.width, args.height, args.fps, args.camera_timeout) <= 0):
            parser.error("Invalid RGB-D sampling options")
    try:
        args.run(args)
    except (KeyboardInterrupt, EOFError):
        print("\n采集已取消；没有发送任何机械臂使能/运动/停机命令。部分文件不会被当作完整数据。")
        return 130
    except (ValueError, RuntimeError, OSError, KeyError) as exc:
        print(f"失败：{exc}；没有发送任何机械臂运动命令。", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
