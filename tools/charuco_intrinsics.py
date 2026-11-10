"""Offline RGB camera-model diagnostics; no camera/robot access or promotion."""

import hashlib
import json
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def camera_profile(camera):
    return {
        "serial": str(camera["serial"]),
        "width": int(camera["width"]),
        "height": int(camera["height"]),
        "frame": str(camera["frame"]),
    }


def check_intrinsics(model, profile=None):
    if (model.get("schema_version") != 1
            or model.get("kind") != "charuco_color_intrinsics"
            or model.get("image_space") != "raw_color"
            or model.get("model") != "radial_k1k2"):
        raise ValueError("Unsupported intrinsics schema/image space/model")
    own_profile = camera_profile(model["camera"])
    hashes = model.get("training_image_sha256")
    if not isinstance(hashes, list) or not hashes or any(not isinstance(h, str) or not h for h in hashes):
        raise ValueError("Intrinsics require training image SHA-256 provenance")
    if profile is not None and own_profile != camera_profile(profile):
        raise ValueError("Intrinsics camera serial/resolution/frame mismatch; no rescaling")
    width, height = own_profile["width"], own_profile["height"]
    K = np.asarray(model["K"], dtype=float)
    D = np.asarray(model["D"], dtype=float).reshape(-1)
    if (width <= 0 or height <= 0 or K.shape != (3, 3) or D.shape != (5,)
            or not np.isfinite(K).all() or not np.isfinite(D).all()
            or K[0, 0] <= 0 or K[1, 1] <= 0
            or not 0 <= K[0, 2] < width or not 0 <= K[1, 2] < height
            or not np.allclose(K[2], [0, 0, 1])
            or not np.allclose([K[0, 1], K[1, 0]], 0)
            or not np.allclose(D[2:], 0)):
        raise ValueError("Invalid K/D for the radial_k1k2 model")
    corners = np.array([[0, 0], [width - 1, 0], [0, height - 1],
                        [width - 1, height - 1]], dtype=float)
    radius_max = np.linalg.norm(
        (corners - [K[0, 2], K[1, 2]]) / [K[0, 0], K[1, 1]], axis=1
    ).max()
    radius2 = np.linspace(0, radius_max, 256) ** 2
    scale = 1 + D[0] * radius2 + D[1] * radius2 ** 2
    derivative = 1 + 3 * D[0] * radius2 + 5 * D[1] * radius2 ** 2
    if min(scale.min(), derivative.min()) <= 0:
        raise ValueError("Non-monotonic radial distortion within the image field")
    return K, D


def load_intrinsics(path, profile=None):
    model = json.loads(Path(path).read_text(encoding="utf-8"))
    check_intrinsics(model, profile)
    return model


def intrinsics_provenance(path, model):
    return {
        "model_sha256": sha256(path),
        "image_space": model["image_space"],
        "model": model["model"],
        "K": model["K"],
        "D": model["D"],
        "training_image_sha256": model["training_image_sha256"],
    }


def _summary(values):
    if not values:
        return None
    values = np.asarray(values, dtype=float)
    return {"mean": float(values.mean()), "median": float(np.median(values)),
            "std": float(values.std()), "max": float(values.max())}


def _observations(root):
    root = Path(root)
    data = json.loads((root / "samples.json").read_text(encoding="utf-8"))
    if data.get("target_type") != "charuco" or not data.get("samples"):
        raise ValueError("Need a nonempty ChArUco dataset")
    board_config = data["board"]
    if board_config.get("legacy_pattern", False):
        raise ValueError("Legacy board is unsupported")
    board = cv2.aruco.CharucoBoard(
        (int(board_config["squares_x"]), int(board_config["squares_y"])),
        float(board_config["square_length_m"]), float(board_config["marker_length_m"]),
        cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, board_config["dictionary"])),
    )
    all_points = board.getChessboardCorners()
    profile = camera_profile(data["camera"])
    views = []
    for sample in data["samples"]:
        image_path = (root / sample["image"]).resolve()
        if not image_path.is_relative_to(root.resolve()):
            raise ValueError("Image path must remain within its dataset")
        digest = sha256(image_path)
        if sample.get("image_sha256") != digest:
            raise ValueError("Image SHA-256 missing or does not match the original file")
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None or image.shape != (profile["height"], profile["width"]):
            raise ValueError("Original image size differs from dataset metadata")
        ids = np.asarray(sample["charuco_ids"], dtype=int).reshape(-1)
        pixels = np.asarray(sample["image_points"], dtype=np.float32).reshape(-1, 2)
        if (len(ids) < 6 or len(ids) != len(pixels) or len(set(ids)) != len(ids)
                or np.any(ids < 0) or np.any(ids >= len(all_points))
                or not np.isfinite(pixels).all()):
            raise ValueError("Invalid saved ChArUco corner correspondences")
        views.append({"sample": sample, "object_points": all_points[ids],
                      "image_points": pixels, "image_sha256": digest})
    return data, views


def _pnp(view, K, D):
    ok, rvec, tvec = cv2.solvePnP(view["object_points"], view["image_points"], K, D)
    if not ok or not np.isfinite(tvec).all() or tvec[2, 0] <= 0:
        raise ValueError("PnP failed for a saved view")
    projected, _ = cv2.projectPoints(view["object_points"], rvec, tvec, K, D)
    errors = np.linalg.norm(projected.reshape(-1, 2) - view["image_points"], axis=1)
    return Rotation.from_rotvec(rvec.reshape(3)), tvec.reshape(3), errors


def estimate_intrinsics(dataset):
    data, views = _observations(dataset)
    if len({view["image_sha256"] for view in views}) != len(views):
        raise ValueError("Duplicate training images are not independent views")
    if len(views) < 8:
        raise ValueError("Need at least 8 different tilted views; 15-25 recommended")
    # Use saved optical poses only to reject repeated/single-axis image sets.
    normals = np.array([np.asarray(v["sample"]["T_camera_target"])[:3, 2] for v in views])
    spread = np.linalg.svd(normals - normals.mean(axis=0), compute_uv=False)
    if spread[1] < 0.02:
        raise ValueError("Insufficient board tilt diversity for intrinsic estimation")
    sdk_K = np.asarray(views[0]["sample"].get("sdk_K", views[0]["sample"]["K"]), float)
    sdk_D = np.asarray(views[0]["sample"].get("sdk_D", views[0]["sample"]["D"]), float)
    for view in views:
        sample = view["sample"]
        if (not np.allclose(sample.get("sdk_K", sample["K"]), sdk_K)
                or not np.allclose(sample.get("sdk_D", sample["D"]), sdk_D)):
            raise ValueError("SDK K/D changed during the dataset")
    flags = cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_FIX_K3
    rms, K, D, _, _ = cv2.calibrateCamera(
        [v["object_points"].astype(np.float32) for v in views],
        [v["image_points"] for v in views],
        (data["camera"]["width"], data["camera"]["height"]),
        sdk_K.copy(), np.zeros(5), flags=flags,
    )
    pixels = np.concatenate([v["image_points"] for v in views])
    report = {
        "schema_version": 1, "kind": "charuco_color_intrinsics",
        "created_at": datetime.now().astimezone().isoformat(),
        "status": "unvalidated_candidate_not_for_runtime",
        "image_space": "raw_color", "model": "radial_k1k2",
        "distortion_order": ["k1", "k2", "p1", "p2", "k3"],
        "camera": camera_profile(data["camera"]), "board": data["board"],
        "K": K.tolist(), "D": D.reshape(-1).tolist(),
        "sdk_K": sdk_K.tolist(), "sdk_D": sdk_D.tolist(),
        "opencv_version": cv2.__version__, "calibration_flags": int(flags),
        "training_dataset": str(Path(dataset).resolve()),
        "training_samples_json_sha256": sha256(Path(dataset) / "samples.json"),
        "training_samples": len(views),
        "training_image_sha256": [v["image_sha256"] for v in views],
        "training_rms_px": float(rms),
        "observed_pixel_bounds": [pixels.min(axis=0).tolist(), pixels.max(axis=0).tolist()],
        "board_normal_spread_singular_values": spread.tolist(),
        "limitations": [
            "Training RMS is not independent validation or metric accuracy.",
            "Unobserved image regions and depth geometry are not validated.",
            "Board dimensions/flatness must be measured independently.",
            "No device calibration or runtime defaults have been changed.",
        ],
    }
    check_intrinsics(report)
    return report


def validate_intrinsics(dataset, model):
    data, views = _observations(dataset)
    K, D = check_intrinsics(model, data["camera"])
    training = set(model.get("training_image_sha256", []))
    if not training or training.intersection(v["image_sha256"] for v in views):
        raise ValueError("Intrinsic validation requires independent captures, not training images")
    models = {}
    for name in ("sdk", "candidate"):
        rotations = []
        per_view = []
        all_errors = []
        for view in views:
            sample = view["sample"]
            own_K = np.asarray(sample.get("sdk_K", sample["K"]), float) if name == "sdk" else K
            own_D = np.asarray(sample.get("sdk_D", sample["D"]), float) if name == "sdk" else D
            rotation, _, errors = _pnp(view, own_K, own_D)
            rotations.append(rotation)
            all_errors.extend(errors.tolist())
            per_view.append({"sample_id": sample["sample_id"],
                             "reprojection_mean_px": float(errors.mean())})
        pairs = []
        for i in range(len(views)):
            for j in range(i):
                B_i = np.asarray(views[i]["sample"]["T_base_flange"], float)[:3, :3]
                B_j = np.asarray(views[j]["sample"]["T_base_flange"], float)[:3, :3]
                robot_deg = np.rad2deg(Rotation.from_matrix(B_j.T @ B_i).magnitude())
                if robot_deg < 5.0:
                    continue
                vision_deg = np.rad2deg((rotations[j] * rotations[i].inv()).magnitude())
                pairs.append({"sample_ids": [views[j]["sample"]["sample_id"],
                                             views[i]["sample"]["sample_id"]],
                              "robot_deg": float(robot_deg), "vision_deg": float(vision_deg),
                              "absolute_difference_deg": float(abs(robot_deg - vision_deg))})
        models[name] = {
            "corner_reprojection_px": _summary(all_errors), "per_view": per_view,
            "rotation_angle_difference_deg": _summary([p["absolute_difference_deg"] for p in pairs]),
            "rotation_pairs": pairs,
        }
    return {
        "created_at": datetime.now().astimezone().isoformat(),
        "status": "diagnostics_only_not_metric_accuracy_certification",
        "validation_dataset": str(Path(dataset).resolve()), "samples": len(views),
        "validation_samples_json_sha256": sha256(Path(dataset) / "samples.json"),
        "candidate_model": {key: model[key] for key in (
            "model", "K", "D", "training_dataset", "training_image_sha256",
        ) if key in model},
        "camera": camera_profile(data["camera"]), "models": models,
        "assumptions": ["Board remains fixed throughout this dataset.",
                        "Rotation comparison uses pairs >=5 degrees; it is hand-eye independent.",
                        "Improved reprojection/J7 alone does not validate multi-axis hand-eye or depth."],
    }


def write_new_json(path, value):
    encoded = json.dumps(value, indent=2, allow_nan=False)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(encoded)
