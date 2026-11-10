#!/usr/bin/env python3
"""SDK-only RealSense/Orbbec Nero eye-in-hand calibration with a ChArUco board.

The board is stationary for each capture session.  The wrist camera moves and
the script reads, but never commands or enables, the Nero arm.  Transforms use
``P_parent = T_parent_child @ P_child``; the result is ``T_flange_camera``.
"""

import argparse
import copy
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import yaml
from scipy.spatial.transform import Rotation


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.charuco_intrinsics import (
    check_intrinsics, estimate_intrinsics, intrinsics_provenance,
    load_intrinsics, validate_intrinsics, write_new_json,
)

HAND_EYE_METHODS = {
    "TSAI": cv2.CALIB_HAND_EYE_TSAI,
    "PARK": cv2.CALIB_HAND_EYE_PARK,
    "HORAUD": cv2.CALIB_HAND_EYE_HORAUD,
    "ANDREFF": cv2.CALIB_HAND_EYE_ANDREFF,
    "DANIILIDIS": cv2.CALIB_HAND_EYE_DANIILIDIS,
}


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def load_board_config(path):
    path = Path(path)
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    board = config["board"]
    quality = config["quality"]
    squares_x = int(board["squares_x"])
    squares_y = int(board["squares_y"])
    square_m = float(board["square_length_m"])
    marker_m = float(board["marker_length_m"])
    min_corners = int(quality["min_charuco_corners"])
    if min(squares_x, squares_y) < 3:
        raise ValueError("ChArUco board requires at least 3x3 squares")
    if not 0 < marker_m < square_m:
        raise ValueError("Require 0 < marker_length_m < square_length_m")
    maximum_corners = (squares_x - 1) * (squares_y - 1)
    if not 6 <= min_corners <= maximum_corners:
        raise ValueError(f"min_charuco_corners must be in [6, {maximum_corners}]")
    dictionary_id = getattr(cv2.aruco, str(board["dictionary"]), None)
    if dictionary_id is None:
        raise ValueError(f"Unknown ArUco dictionary: {board['dictionary']}")
    if bool(board.get("legacy_pattern", False)):
        raise ValueError("legacy_pattern=true is not supported by this board definition")
    dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id)
    target = cv2.aruco.CharucoBoard(
        (squares_x, squares_y), square_m, marker_m, dictionary
    )
    detector = cv2.aruco.CharucoDetector(target)
    normalized = {
        "squares_x": squares_x,
        "squares_y": squares_y,
        "square_length_m": square_m,
        "marker_length_m": marker_m,
        "dictionary": str(board["dictionary"]),
        "legacy_pattern": False,
    }
    normalized_quality = {
        "min_charuco_corners": min_corners,
        "warning_reprojection_error_px": float(
            quality["warning_reprojection_error_px"]
        ),
        "max_reprojection_error_px": float(quality["max_reprojection_error_px"]),
        "axis_length_m": float(quality["axis_length_m"]),
    }
    return target, detector, normalized, normalized_quality


def make_transform(rotation, translation):
    transform = np.eye(4)
    transform[:3, :3] = np.asarray(rotation, dtype=float).reshape(3, 3)
    transform[:3, 3] = np.asarray(translation, dtype=float).reshape(3)
    return transform


def check_transform(value, label="transform"):
    transform = np.asarray(value, dtype=float)
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise ValueError(f"Invalid {label}")
    if not np.allclose(transform[3], [0, 0, 0, 1], atol=1e-8):
        raise ValueError(f"Invalid homogeneous row in {label}")
    rotation = transform[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5):
        raise ValueError(f"Non-orthonormal rotation in {label}")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-5):
        raise ValueError(f"Improper rotation in {label}")
    return transform


def consistency_center_transform(transforms):
    values = [check_transform(value) for value in transforms]
    result = np.eye(4)
    result[:3, :3] = Rotation.from_matrix(
        np.asarray([value[:3, :3] for value in values])
    ).mean().as_matrix()
    # Match the proven ChArUco workflow: translation uses a robust componentwise
    # median while SO(3) uses the intrinsic rotation mean.
    result[:3, 3] = np.median([value[:3, 3] for value in values], axis=0)
    return result


def residual_values(samples, flange_camera, base_target):
    flange_camera = check_transform(flange_camera, "T_flange_camera")
    base_target = check_transform(base_target, "T_base_target")
    translation_mm = []
    rotation_deg = []
    for sample in samples:
        base_flange = check_transform(sample["T_base_flange"], "T_base_flange")
        camera_target = check_transform(sample["T_camera_target"], "T_camera_target")
        error = np.linalg.inv(base_target) @ base_flange @ flange_camera @ camera_target
        translation_mm.append(float(np.linalg.norm(error[:3, 3]) * 1000.0))
        rotation_deg.append(
            float(Rotation.from_matrix(error[:3, :3]).magnitude() * 180.0 / np.pi)
        )
    return {"translation_mm": translation_mm, "rotation_deg": rotation_deg}


def summarize_residuals(values):
    summary = {}
    for name, entries in values.items():
        array = np.asarray(entries, dtype=float)
        summary[name] = {
            "mean": float(np.mean(array)),
            "median": float(np.median(array)),
            "std": float(np.std(array)),
            "max": float(np.max(array)),
        }
    return summary


def solve_park(samples):
    """Solve ``B_i X C_i = Y`` using Park rotation alignment and translation LS."""
    if len(samples) < 12:
        raise ValueError("Need at least 12 included stationary views")
    base_flange = [
        check_transform(sample["T_base_flange"], "T_base_flange")
        for sample in samples
    ]
    camera_target = [
        check_transform(sample["T_camera_target"], "T_camera_target")
        for sample in samples
    ]
    absolute_rotations = np.asarray(
        [
            Rotation.from_matrix(base_flange[0][:3, :3].T @ value[:3, :3]).as_rotvec()
            for value in base_flange[1:]
        ]
    )
    singular = np.linalg.svd(absolute_rotations, compute_uv=False)
    if singular.size < 2 or singular[1] < 0.25:
        raise ValueError("Insufficient flange rotation about independent axes")
    pairs = [
        (
            np.linalg.inv(base_flange[j]) @ base_flange[i],
            camera_target[j] @ np.linalg.inv(camera_target[i]),
        )
        for i in range(len(samples))
        for j in range(i)
    ]
    vectors_a = np.asarray(
        [Rotation.from_matrix(a[:3, :3]).as_rotvec() for a, _ in pairs]
    )
    vectors_b = np.asarray(
        [Rotation.from_matrix(b[:3, :3]).as_rotvec() for _, b in pairs]
    )
    rotation = Rotation.align_vectors(vectors_a, vectors_b)[0].as_matrix()
    lhs = np.vstack([a[:3, :3] - np.eye(3) for a, _ in pairs])
    rhs = np.concatenate(
        [rotation @ b[:3, 3] - a[:3, 3] for a, b in pairs]
    )
    translation, _, rank, _ = np.linalg.lstsq(lhs, rhs, rcond=None)
    if rank < 3:
        raise ValueError("Degenerate hand-eye translation system")
    flange_camera = make_transform(rotation, translation)
    board_estimates = [
        base @ flange_camera @ camera
        for base, camera in zip(base_flange, camera_target)
    ]
    return flange_camera, consistency_center_transform(board_estimates), singular


def solve_opencv_methods(samples):
    """Run and rank the five methods used by the original ChArUco workflow."""
    if len(samples) < 12:
        raise ValueError("Need at least 12 included stationary views")
    if not hasattr(cv2, "calibrateHandEye"):
        raise RuntimeError(
            "OpenCV calibrateHandEye is unavailable; require "
            "opencv-python/opencv-contrib-python 4.13.0.92"
        )
    base_flange = [
        check_transform(sample["T_base_flange"], "T_base_flange")
        for sample in samples
    ]
    camera_target = [
        check_transform(sample["T_camera_target"], "T_camera_target")
        for sample in samples
    ]
    absolute_rotations = np.asarray(
        [
            Rotation.from_matrix(
                base_flange[0][:3, :3].T @ value[:3, :3]
            ).as_rotvec()
            for value in base_flange[1:]
        ]
    )
    singular = np.linalg.svd(absolute_rotations, compute_uv=False)
    if singular.size < 2 or singular[1] < 0.25:
        raise ValueError("Insufficient flange rotation about independent axes")

    results = {}
    failures = {}
    for name, method in HAND_EYE_METHODS.items():
        try:
            rotation, translation = cv2.calibrateHandEye(
                [value[:3, :3] for value in base_flange],
                [value[:3, 3] for value in base_flange],
                [value[:3, :3] for value in camera_target],
                [value[:3, 3] for value in camera_target],
                method=method,
            )
            flange_camera = check_transform(
                make_transform(rotation, translation), f"{name} T_flange_camera"
            )
            base_target = consistency_center_transform(
                [
                    base @ flange_camera @ camera
                    for base, camera in zip(base_flange, camera_target)
                ]
            )
            residuals = residual_values(samples, flange_camera, base_target)
            summary = summarize_residuals(residuals)
            score = (
                summary["translation_mm"]["std"]
                + summary["rotation_deg"]["std"]
            )
            results[name] = {
                "T_flange_camera": flange_camera,
                "T_base_target_training": base_target,
                "residuals": residuals,
                "residual_summary": summary,
                "score": float(score),
            }
        except (cv2.error, ValueError, FloatingPointError) as exc:
            failures[name] = str(exc)
    if not results:
        raise RuntimeError(f"All OpenCV hand-eye methods failed: {failures}")
    best = min(results, key=lambda name: results[name]["score"])
    return best, results, singular, failures


def detect_charuco(image, board, detector, intrinsics, distortion):
    charuco_corners, charuco_ids, marker_corners, marker_ids = detector.detectBoard(image)
    marker_count = 0 if marker_ids is None else int(len(marker_ids))
    corner_count = 0 if charuco_ids is None else int(len(charuco_ids))
    result = {
        "marker_corners": marker_corners,
        "marker_ids": marker_ids,
        "charuco_corners": charuco_corners,
        "charuco_ids": charuco_ids,
        "valid_markers": marker_count,
        "valid_corners": corner_count,
        "pnp_success": False,
    }
    if corner_count < 6:
        return result
    object_points, image_points = board.matchImagePoints(charuco_corners, charuco_ids)
    try:
        ok, rvec, tvec = cv2.solvePnP(
            object_points, image_points, intrinsics, distortion
        )
    except cv2.error:
        return result
    if not ok or float(np.asarray(tvec).reshape(3)[2]) <= 0:
        return result
    projected, _ = cv2.projectPoints(
        object_points, rvec, tvec, intrinsics, distortion
    )
    errors = np.linalg.norm(
        projected.reshape(-1, 2) - image_points.reshape(-1, 2), axis=1
    )
    result.update(
        pnp_success=True,
        rvec=np.asarray(rvec, dtype=float).reshape(3),
        tvec=np.asarray(tvec, dtype=float).reshape(3),
        T_camera_target=make_transform(
            cv2.Rodrigues(np.asarray(rvec, dtype=float))[0], tvec
        ),
        image_points=np.asarray(image_points, dtype=float).reshape(-1, 2),
        charuco_ids_flat=np.asarray(charuco_ids, dtype=int).reshape(-1),
        reprojection_mean_px=float(np.mean(errors)),
        reprojection_median_px=float(np.median(errors)),
        reprojection_max_px=float(np.max(errors)),
    )
    return result


def quality_reason(detection, quality):
    if detection["valid_corners"] < quality["min_charuco_corners"]:
        return "insufficient ChArUco corners"
    if not detection["pnp_success"]:
        return "solvePnP failed"
    if detection["reprojection_mean_px"] > quality["max_reprojection_error_px"]:
        return "reprojection error too large"
    return None


def draw_detection(image, detection, intrinsics, distortion, axis_length):
    visual = image.copy()
    if detection["marker_ids"] is not None:
        cv2.aruco.drawDetectedMarkers(
            visual, detection["marker_corners"], detection["marker_ids"]
        )
    if detection["charuco_ids"] is not None:
        # OpenCV 5 may return ChArUco corners as (N, 2), while its drawing
        # binding still requires the legacy (N, 1, 2) representation.
        draw_corners = np.asarray(
            detection["charuco_corners"], dtype=np.float32
        ).reshape(-1, 1, 2)
        draw_ids = np.asarray(
            detection["charuco_ids"], dtype=np.int32
        ).reshape(-1, 1)
        cv2.aruco.drawDetectedCornersCharuco(
            visual, draw_corners, draw_ids
        )
    if detection["pnp_success"]:
        try:
            cv2.drawFrameAxes(
                visual,
                intrinsics,
                distortion,
                detection["rvec"],
                detection["tvec"],
                float(axis_length),
            )
        except cv2.error:
            pass
    return visual


class NeroReadOnlyFeedback:
    """Connect to Nero feedback without enabling it or sending motion commands."""

    def __init__(self, channel, firmware, timeout):
        from pyAgxArm import AgxArmFactory, ArmModel, NeroFW, create_agx_arm_config

        config = create_agx_arm_config(
            robot=ArmModel.NERO,
            firmeware_version=getattr(NeroFW, firmware),
            interface="socketcan",
            channel=channel,
            bitrate=1000000,
        )
        self.arm = AgxArmFactory.create_arm(config)
        self.timeout = float(timeout)

    def __enter__(self):
        self.arm.connect()
        deadline = time.time() + self.timeout
        while time.time() < deadline:
            try:
                self.read()
                return self
            except RuntimeError:
                time.sleep(0.05)
        self.arm.disconnect()
        raise TimeoutError("No fresh Nero flange feedback; check can0 and firmware")

    def read(self):
        response = self.arm.get_flange_pose()
        if response is None:
            raise RuntimeError("No flange feedback")
        values = np.asarray(
            response.msg if hasattr(response, "msg") else response, dtype=float
        ).reshape(-1)
        if values.size < 6 or not np.isfinite(values[:6]).all():
            raise RuntimeError("Invalid flange feedback")
        timestamp = float(getattr(response, "timestamp", time.time()))
        age = time.time() - timestamp
        if not 0 <= age <= 0.5:
            raise RuntimeError(f"Stale flange feedback ({age:.3f}s)")
        pose = values[:6].copy()
        transform = make_transform(
            Rotation.from_euler("xyz", pose[3:]).as_matrix(), pose[:3]
        )
        return pose, transform, timestamp

    def read_joints(self):
        """Read/copy the joint feedback used to audit each camera/pose pair."""
        response = self.arm.get_joint_angles()
        if response is None:
            raise RuntimeError("No joint feedback")
        joints = np.asarray(response.msg, dtype=float).reshape(-1)
        timestamp = float(response.timestamp)
        age = time.time() - timestamp
        if joints.shape != (7,) or not np.isfinite(joints).all() or not 0 <= age <= .5:
            raise RuntimeError("Invalid or stale seven-joint feedback")
        return joints.copy(), timestamp

    def __exit__(self, *_):
        try:
            self.arm.disconnect()
        except Exception:
            pass


def stationary_pair(before, after, before_stamp, after_stamp):
    delta = np.linalg.inv(before) @ after
    translation = float(np.linalg.norm(delta[:3, 3]))
    rotation = float(Rotation.from_matrix(delta[:3, :3]).magnitude())
    if after_stamp <= before_stamp:
        raise RuntimeError("Flange feedback did not advance")
    if translation > 0.001 or rotation > np.deg2rad(0.3):
        raise RuntimeError(
            "Wrist moved during capture: "
            f"{translation * 1000.0:.2f}mm/{np.rad2deg(rotation):.2f}deg"
        )


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dataset_payload(output, board_config, camera, firmware, channel):
    payload = {
        "schema_version": 1,
        "session_id": Path(output).resolve().name,
        "target_type": "charuco",
        "board": board_config,
        "camera": {
            "serial": str(camera.serial),
            "width": int(camera.width),
            "height": int(camera.height),
            "fps": int(camera.fps),
            "frame": getattr(camera, "optical_frame", "d435_color_optical_frame"),
            "backend": getattr(camera, "backend", "realsense"),
            "image_space": "rectified_color" if getattr(camera, "backend", "realsense") == "orbbec" else "raw_color",
        },
        "robot": {
            "model": "NERO",
            "channel": str(channel),
            "firmware_profile": str(firmware),
            "pose_source": "pyAgxArm.get_flange_pose",
        },
        "samples": [],
    }
    override = getattr(camera, "intrinsics_override", None)
    if override is not None:
        payload["camera"]["intrinsics_override"] = override
    return payload


def load_dataset(path):
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("samples"), list):
        raise ValueError(f"Invalid ChArUco dataset: {path}")
    if data.get("target_type") != "charuco":
        raise ValueError("Dataset target_type is not charuco")
    return data


def dataset_metadata(data):
    metadata = {
        "target_type": data["target_type"],
        "board": data["board"],
        "camera_serial": data["camera"]["serial"],
        "camera_width": int(data["camera"]["width"]),
        "camera_height": int(data["camera"]["height"]),
        "camera_frame": data["camera"]["frame"],
    }
    if "intrinsics_override" in data["camera"]:
        metadata["camera_intrinsics"] = data["camera"]["intrinsics_override"]
    return metadata


def ensure_independent(calibration, dataset):
    training_values = calibration.get("training_image_sha256", [])
    current_values = [sample.get("image_sha256") for sample in dataset["samples"]]
    if not training_values or not current_values or any(not value for value in current_values):
        raise ValueError("Training and validation datasets require image SHA-256 identities")
    current_hashes = set(current_values)
    training_hashes = set(training_values)
    overlap = current_hashes & training_hashes
    if overlap:
        raise ValueError(
            f"Validation requires independent captures; {len(overlap)} image(s) overlap"
        )


class CalibrationIntrinsicsCamera:
    """Opt-in raw-color PnP model; never rectifies images or changes SDK settings."""

    def __init__(self, camera, path, model):
        self.camera = camera
        self.model = model
        self.intrinsics_override = intrinsics_provenance(path, model)

    def __getattr__(self, name):
        return getattr(self.camera, name)

    def __enter__(self):
        self.camera.__enter__()
        try:
            check_intrinsics(self.model, {
                "serial": self.camera.serial, "width": self.camera.width,
                "height": self.camera.height, "frame": "d435_color_optical_frame",
            })
        except BaseException:
            self.camera.__exit__(*sys.exc_info())
            raise
        print("[calibration] Using candidate K/D for raw-color PnP only; not runtime/depth")
        return self

    def read(self):
        frame = copy.copy(self.camera.read())
        if frame.bgr.shape[:2] != (self.model["camera"]["height"], self.model["camera"]["width"]):
            raise ValueError("Actual color frame size differs from candidate intrinsics")
        frame.sdk_K, frame.sdk_D = frame.K.copy(), frame.D.copy()
        frame.K, frame.D = check_intrinsics(self.model)
        return frame

    def __exit__(self, *args):
        return self.camera.__exit__(*args)


def camera_from_args(args):
    from cameras.realsense import create_camera

    backend = getattr(args, "backend", "realsense")
    path = getattr(args, "intrinsics", None)
    if backend == "orbbec" and path:
        raise ValueError("Orbbec calibration uses SDK-rectified RGB and matching K/D; omit raw-color --intrinsics")
    model = load_intrinsics(path, {
        "serial": args.serial, "width": args.width, "height": args.height,
        "frame": "d435_color_optical_frame",
    }) if path else None
    camera = create_camera(
        backend=backend,
        serial=args.serial,
        width=args.width,
        height=args.height,
        fps=args.fps,
        timeout=args.camera_timeout,
        warmup=args.warmup,
        enable_depth=False,
    )
    return CalibrationIntrinsicsCamera(camera, path, model) if path else camera


def run_test(args):
    board, detector, _, quality = load_board_config(args.board)
    save = Path(args.save_dir)
    save.mkdir(parents=True, exist_ok=True)
    with camera_from_args(args) as camera:
        while True:
            frame = camera.read()
            detection = detect_charuco(frame.bgr, board, detector, frame.K, frame.D)
            visual = draw_detection(
                frame.bgr, detection, frame.K, frame.D, quality["axis_length_m"]
            )
            reason = quality_reason(detection, quality)
            text = (
                f"markers={detection['valid_markers']} "
                f"corners={detection['valid_corners']} "
                f"quality={'OK' if reason is None else reason}"
            )
            cv2.putText(
                visual, text, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
            )
            cv2.imshow("RGB ChArUco test: s save / q quit", visual)
            key = cv2.waitKey(1) & 255
            if key in (27, ord("q")):
                break
            if key == ord("s"):
                path = save / f"charuco_{time.time_ns()}.png"
                if not cv2.imwrite(str(path), visual):
                    raise OSError(f"Cannot save {path}")
                print("Saved", path)
    cv2.destroyAllWindows()


def run_capture(args):
    board, detector, board_config, quality = load_board_config(args.board)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "images").mkdir()
    (output / "debug").mkdir()
    shutil.copy2(args.board, output / "board.yaml")
    try:
        with camera_from_args(args) as camera, NeroReadOnlyFeedback(
            args.channel, args.firmware, args.feedback_timeout
        ) as robot:
            data = dataset_payload(
                output, board_config, camera, args.firmware, args.channel
            )
            atomic_json(output / "samples.json", data)
            print(
                "Fixed rigid ChArUco board; move the wrist using existing controls, "
                "then stop and settle before saving. s=save, d=delete last, q=quit."
            )
            print("This process does not enable or command the robot.")
            while True:
                frame = camera.read()
                detection = detect_charuco(
                    frame.bgr, board, detector, frame.K, frame.D
                )
                visual = draw_detection(
                    frame.bgr,
                    detection,
                    frame.K,
                    frame.D,
                    quality["axis_length_m"],
                )
                reason = quality_reason(detection, quality)
                text = (
                    f"samples={len(data['samples'])} "
                    f"corners={detection['valid_corners']} "
                    f"quality={'OK' if reason is None else reason}"
                )
                cv2.putText(
                    visual,
                    text,
                    (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )
                cv2.imshow("RGB ChArUco capture: s save / d delete / q quit", visual)
                key = cv2.waitKey(1) & 255
                if key in (27, ord("q")):
                    break
                if key == ord("d") and data["samples"]:
                    removed = data["samples"].pop()
                    (output / removed["image"]).unlink(missing_ok=True)
                    (output / removed["debug_image"]).unlink(missing_ok=True)
                    atomic_json(output / "samples.json", data)
                    print("Deleted sample", removed["sample_id"])
                    continue
                if key != ord("s"):
                    continue
                try:
                    before_request = time.monotonic()
                    pose_before, before, before_stamp = robot.read()
                    joints_before, joint_stamp_before = robot.read_joints()
                    candidates = []
                    for _ in range(int(args.candidate_frames)):
                        candidate_frame = camera.read()
                        read_return = time.monotonic()
                        candidate_detection = detect_charuco(
                            candidate_frame.bgr,
                            board,
                            detector,
                            candidate_frame.K,
                            candidate_frame.D,
                        )
                        candidate_reason = quality_reason(
                            candidate_detection, quality
                        )
                        if candidate_reason is None:
                            candidates.append(
                                (
                                    candidate_detection["reprojection_mean_px"],
                                    -candidate_detection["valid_corners"],
                                    candidate_frame,
                                    candidate_detection,
                                    read_return,
                                )
                            )
                    after_request = time.monotonic()
                    joints_after, joint_stamp_after = robot.read_joints()
                    pose, after, after_stamp = robot.read()
                    stationary_pair(before, after, before_stamp, after_stamp)
                    if not candidates:
                        raise RuntimeError("No valid ChArUco frame among candidates")
                    _, _, selected_frame, selected, selected_read_return = min(
                        candidates, key=lambda item: (item[0], item[1])
                    )
                    sample_id = len(data["samples"])
                    image_rel = f"images/{sample_id:03d}.png"
                    debug_rel = f"debug/{sample_id:03d}.png"
                    image_path = output / image_rel
                    debug_path = output / debug_rel
                    if not cv2.imwrite(str(image_path), selected_frame.bgr):
                        raise OSError(f"Cannot save {image_path}")
                    debug_image = draw_detection(
                        selected_frame.bgr,
                        selected,
                        selected_frame.K,
                        selected_frame.D,
                        quality["axis_length_m"],
                    )
                    if not cv2.imwrite(str(debug_path), debug_image):
                        image_path.unlink(missing_ok=True)
                        raise OSError(f"Cannot save {debug_path}")
                    sample = {
                        "sample_id": sample_id,
                        "sample_uid": f"{data['session_id']}:{sample_id:03d}",
                        "captured_at_ns": time.time_ns(),
                        "image": image_rel,
                        "debug_image": debug_rel,
                        "image_sha256": file_sha256(image_path),
                        "T_base_flange": after.tolist(),
                        "flange_pose_xyzrpy": pose.tolist(),
                        "flange_pose_before_xyzrpy": pose_before.tolist(),
                        "joints_before_rad": joints_before.tolist(),
                        "joints_after_rad": joints_after.tolist(),
                        "joint_feedback_stamps": [joint_stamp_before, joint_stamp_after],
                        "T_camera_target": selected["T_camera_target"].tolist(),
                        "feedback_stamps": [before_stamp, after_stamp],
                        "camera_hardware_ms": float(selected_frame.hardware_ms),
                        "camera_sequence": int(selected_frame.sequence),
                        "capture_timing": {
                            "before_feedback_request_monotonic": before_request,
                            "after_feedback_request_monotonic": after_request,
                            "image_received_monotonic": float(selected_frame.received_monotonic),
                            "image_read_return_monotonic": selected_read_return,
                            "image_age_at_read_ms": (selected_read_return - selected_frame.received_monotonic) * 1000.,
                        },
                        "K": selected_frame.K.tolist(),
                        "D": selected_frame.D.tolist(),
                        "valid_markers": selected["valid_markers"],
                        "valid_corners": selected["valid_corners"],
                        "charuco_ids": selected["charuco_ids_flat"].tolist(),
                        "image_points": selected["image_points"].tolist(),
                        "reprojection_mean_px": selected[
                            "reprojection_mean_px"
                        ],
                        "reprojection_median_px": selected[
                            "reprojection_median_px"
                        ],
                        "reprojection_max_px": selected["reprojection_max_px"],
                    }
                    if hasattr(selected_frame, "sdk_K"):
                        sample["sdk_K"] = selected_frame.sdk_K.tolist()
                        sample["sdk_D"] = selected_frame.sdk_D.tolist()
                    data["samples"].append(sample)
                    atomic_json(output / "samples.json", data)
                    print(
                        f"Saved sample {sample_id:02d}: "
                        f"corners={sample['valid_corners']} "
                        f"reprojection={sample['reprojection_mean_px']:.3f}px "
                        f"image_age={sample['capture_timing']['image_age_at_read_ms']:.1f}ms "
                        f"joint_delta={np.degrees(np.max(np.abs(joints_after-joints_before))):.3f}deg"
                    )
                except RuntimeError as exc:
                    print("Rejected:", exc)
    except KeyboardInterrupt:
        print("Capture stopped; previously saved samples are retained.")
    finally:
        cv2.destroyAllWindows()
    print("Dataset:", output)


def run_reprocess(args):
    source_path = Path(args.dataset)
    source = load_dataset(source_path / "samples.json")
    board, detector, board_config, quality = load_board_config(args.board)
    intrinsics_path = getattr(args, "intrinsics", None)
    model = load_intrinsics(intrinsics_path, source["camera"]) if intrinsics_path else None
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "images").mkdir()
    (output / "debug").mkdir()
    shutil.copy2(args.board, output / "board.yaml")
    result = copy.deepcopy(source)
    result["session_id"] = output.resolve().name
    result["reprocessed_at_ns"] = time.time_ns()
    result["source_dataset"] = str(source_path.resolve())
    result["source_samples_json_sha256"] = file_sha256(source_path / "samples.json")
    result["board"] = board_config
    if model is not None:
        result["camera"]["intrinsics_override"] = intrinsics_provenance(intrinsics_path, model)
    result["samples"] = []
    for source_sample in source["samples"]:
        sample = copy.deepcopy(source_sample)
        image = cv2.imread(str(source_path / source_sample["image"]), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(source_path / source_sample["image"])
        if (file_sha256(source_path / source_sample["image"]) != source_sample["image_sha256"]
                or image.shape[:2] != (int(source["camera"]["height"]), int(source["camera"]["width"]))):
            raise ValueError("Original image SHA-256/size differs from dataset metadata")
        intrinsics = np.asarray(source_sample["K"], dtype=float)
        distortion = np.asarray(source_sample["D"], dtype=float)
        if model is not None:
            sample["sdk_K"] = source_sample.get("sdk_K", source_sample["K"])
            sample["sdk_D"] = source_sample.get("sdk_D", source_sample["D"])
            intrinsics, distortion = check_intrinsics(model)
        detection = detect_charuco(image, board, detector, intrinsics, distortion)
        reason = quality_reason(detection, quality)
        if reason:
            raise RuntimeError(
                f"sample {source_sample['sample_id']} reprocessing rejected: {reason}"
            )
        sample_id = int(source_sample["sample_id"])
        image_rel = f"images/{sample_id:03d}.png"
        debug_rel = f"debug/{sample_id:03d}.png"
        shutil.copy2(source_path / source_sample["image"], output / image_rel)
        visual = draw_detection(
            image, detection, intrinsics, distortion, quality["axis_length_m"]
        )
        if not cv2.imwrite(str(output / debug_rel), visual):
            raise OSError(f"Cannot save {output / debug_rel}")
        sample.update(
            sample_uid=f"{result['session_id']}:{sample_id:03d}",
            image=image_rel,
            debug_image=debug_rel,
            image_sha256=file_sha256(output / image_rel),
            K=intrinsics.tolist(),
            D=distortion.tolist(),
            T_camera_target=detection["T_camera_target"].tolist(),
            valid_markers=detection["valid_markers"],
            valid_corners=detection["valid_corners"],
            charuco_ids=detection["charuco_ids_flat"].tolist(),
            image_points=detection["image_points"].tolist(),
            reprojection_mean_px=detection["reprojection_mean_px"],
            reprojection_median_px=detection["reprojection_median_px"],
            reprojection_max_px=detection["reprojection_max_px"],
        )
        result["samples"].append(sample)
    atomic_json(output / "samples.json", result)
    print(f"Reprocessed {len(result['samples'])} samples")
    print("Dataset:", output)


def run_solve(args):
    data = load_dataset(Path(args.dataset) / "samples.json")
    excluded = set(int(value) for value in args.exclude)
    unknown = excluded - {int(sample["sample_id"]) for sample in data["samples"]}
    if unknown:
        raise ValueError(f"Unknown excluded sample IDs: {sorted(unknown)}")
    included = [
        sample for sample in data["samples"] if int(sample["sample_id"]) not in excluded
    ]
    best_method, method_results, singular, method_failures = (
        solve_opencv_methods(included)
    )
    best_result = method_results[best_method]
    flange_camera = best_result["T_flange_camera"]
    base_target = best_result["T_base_target_training"]
    residuals = best_result["residuals"]
    summary = best_result["residual_summary"]
    quaternion = Rotation.from_matrix(flange_camera[:3, :3]).as_quat()
    if quaternion[3] < 0:
        quaternion = -quaternion
    output = {
        "created_at_ns": time.time_ns(),
        "method": best_method,
        "solver": "OpenCV_calibrateHandEye_five_method_ranking",
        "opencv_version": cv2.__version__,
        "convention": "flange_camera_xyzw",
        "parent_frame": "SDK_feedback_flange",
        "child_frame": data["camera"]["frame"],
        "position": flange_camera[:3, 3].tolist(),
        "orientation": quaternion.tolist(),
        "flange_camera_xyzw": np.r_[flange_camera[:3, 3], quaternion].tolist(),
        "T_flange_camera": flange_camera.tolist(),
        "matrix_4x4": flange_camera.tolist(),
        "T_base_target_training": base_target.tolist(),
        "metadata": dataset_metadata(data),
        "training_dataset": str(Path(args.dataset).resolve()),
        "training_sample_ids": [sample["sample_uid"] for sample in included],
        "training_image_sha256": [sample["image_sha256"] for sample in included],
        "excluded_sample_ids": sorted(excluded),
        "rotation_diversity_singular_values": singular.tolist(),
        "residuals": residuals,
        "residual_summary": summary,
        "all_methods": {
            name: {
                "T_flange_camera": result["T_flange_camera"].tolist(),
                "matrix_4x4": result["T_flange_camera"].tolist(),
                "residual_summary": result["residual_summary"],
                "score": result["score"],
            }
            for name, result in method_results.items()
        },
        "method_failures": method_failures,
        "status": "unvalidated",
    }
    output_path = Path(args.output)
    with output_path.open("x", encoding="utf-8") as stream:
        json.dump(output, stream, indent=2)
    print("Method       Trans STD   Median error   Rot STD    Score")
    for name, result in sorted(
        method_results.items(), key=lambda item: item[1]["score"]
    ):
        method_summary = result["residual_summary"]
        print(
            f"{name:<12} "
            f"{method_summary['translation_mm']['std']:9.3f} mm "
            f"{method_summary['translation_mm']['median']:9.3f} mm "
            f"{method_summary['rotation_deg']['std']:8.3f} deg "
            f"{result['score']:8.3f}"
        )
    if method_failures:
        print("Method failures:", method_failures)
    print("Recommended method:", best_method)
    print("Outliers sorted by translation error:")
    print("Sample   Trans error   Rot error   Reproj")
    order = np.argsort(-np.asarray(residuals["translation_mm"]))
    for index in order:
        sample = included[int(index)]
        print(
            f"{int(sample['sample_id']):6d} "
            f"{residuals['translation_mm'][int(index)]:10.3f} mm "
            f"{residuals['rotation_deg'][int(index)]:9.3f} deg "
            f"{float(sample['reprojection_mean_px']):7.3f} px"
        )
    print("Best-method residual summary:")
    print(json.dumps(summary, indent=2))
    print("Saved:", output_path)
    print("flange_camera_xyzw:", output["flange_camera_xyzw"])


def run_validate(args):
    calibration = json.loads(Path(args.calibration).read_text(encoding="utf-8"))
    data = load_dataset(Path(args.dataset) / "samples.json")
    if len(data["samples"]) < 5:
        raise ValueError("Need at least 5 independent validation views")
    if calibration["metadata"] != dataset_metadata(data):
        raise ValueError("Calibration and validation board/camera metadata differ")
    ensure_independent(calibration, data)
    camera_model = calibration["metadata"].get("camera_intrinsics")
    if camera_model is not None:
        ensure_independent(camera_model, data)
    flange_camera = check_transform(
        calibration["T_flange_camera"], "T_flange_camera"
    )
    estimates = [
        check_transform(sample["T_base_flange"]) @ flange_camera
        @ check_transform(sample["T_camera_target"])
        for sample in data["samples"]
    ]
    validation_center = consistency_center_transform(estimates)
    internal = residual_values(data["samples"], flange_camera, validation_center)
    report = {
        "validation_dataset": str(Path(args.dataset).resolve()),
        "samples": len(data["samples"]),
        "internal_consistency": {
            "values": internal,
            "summary": summarize_residuals(internal),
        },
    }
    if args.same_board_pose:
        training_target = check_transform(
            calibration["T_base_target_training"], "T_base_target_training"
        )
        relative = residual_values(data["samples"], flange_camera, training_target)
        report["relative_to_training_board"] = {
            "valid_only_if_board_never_moved": True,
            "values": relative,
            "summary": summarize_residuals(relative),
        }
    print(json.dumps(report, indent=2))


def add_camera_arguments(parser):
    parser.add_argument("--backend", choices=("realsense", "orbbec"), default="realsense")
    parser.add_argument("--serial", default=None)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--camera-timeout", type=float, default=10.0)
    parser.add_argument("--intrinsics", help="Candidate raw-color K/D JSON, for calibration PnP only")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    default_board = str(ROOT / "config" / "charuco_board.yaml")

    test_parser = commands.add_parser("test")
    test_parser.add_argument("--board", default=default_board)
    test_parser.add_argument("--save-dir", default="calibration/charuco_debug")
    add_camera_arguments(test_parser)

    capture_parser = commands.add_parser("capture")
    capture_parser.add_argument("--output", required=True)
    capture_parser.add_argument("--board", default=default_board)
    capture_parser.add_argument("--channel", default="can0")
    capture_parser.add_argument("--firmware", choices=["V111", "V112"], default="V111")
    capture_parser.add_argument("--feedback-timeout", type=float, default=5.0)
    capture_parser.add_argument("--candidate-frames", type=int, default=5)
    add_camera_arguments(capture_parser)

    reprocess_parser = commands.add_parser("reprocess")
    reprocess_parser.add_argument("--dataset", required=True)
    reprocess_parser.add_argument("--output", required=True)
    reprocess_parser.add_argument("--board", default=default_board)
    reprocess_parser.add_argument("--intrinsics", help="Candidate raw-color K/D JSON")

    intrinsics_parser = commands.add_parser("intrinsics-solve")
    intrinsics_parser.add_argument("--dataset", required=True)
    intrinsics_parser.add_argument("--output", required=True)

    intrinsics_validate_parser = commands.add_parser("intrinsics-validate")
    intrinsics_validate_parser.add_argument("--dataset", required=True)
    intrinsics_validate_parser.add_argument("--intrinsics", required=True)
    intrinsics_validate_parser.add_argument("--output", required=True)

    solve_parser = commands.add_parser("solve")
    solve_parser.add_argument("--dataset", required=True)
    solve_parser.add_argument("--output", required=True)
    solve_parser.add_argument("--exclude", type=int, nargs="*", default=[])

    validate_parser = commands.add_parser("validate")
    validate_parser.add_argument("--dataset", required=True)
    validate_parser.add_argument("--calibration", required=True)
    validate_parser.add_argument(
        "--same-board-pose",
        action="store_true",
        help="Also compare with the training board pose; valid only if the board never moved",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.command == "test":
        return run_test(args)
    if args.command == "capture":
        if args.candidate_frames <= 0:
            raise ValueError("candidate-frames must be positive")
        return run_capture(args)
    if args.command == "reprocess":
        return run_reprocess(args)
    if args.command == "intrinsics-solve":
        report = estimate_intrinsics(args.dataset)
        write_new_json(args.output, report)
        print("Saved candidate (not promoted to runtime):", args.output)
        print("Training RMS px:", report["training_rms_px"])
        return
    if args.command == "intrinsics-validate":
        model = load_intrinsics(args.intrinsics)
        report = validate_intrinsics(args.dataset, model)
        report["intrinsics_file"] = str(Path(args.intrinsics).resolve())
        report["intrinsics_file_sha256"] = file_sha256(args.intrinsics)
        write_new_json(args.output, report)
        print(json.dumps({name: {key: value for key, value in result.items()
                                if key not in ("rotation_pairs", "per_view")}
                          for name, result in report["models"].items()}, indent=2))
        print("Saved diagnostic report:", args.output)
        return
    if args.command == "solve":
        return run_solve(args)
    return run_validate(args)


if __name__ == "__main__":
    main()
