"""Profile-bound RGB validation of a measured, camera-fixed robot silhouette.

Depth cannot identify self pixels after a bad return has put them elsewhere in
3-D.  Only the measured silhouette is excluded here; dark cables/objects outside
it remain obstacles. A mismatch aborts capture instead of guessing a new mask.
"""

import json
from pathlib import Path

import cv2
import numpy as np


class CameraSelfMaskMismatch(ValueError):
    """Expected capture rejection with the measured validation data."""

    def __init__(self, message, report):
        super().__init__(message)
        self.report = report


def _load(path):
    calibration = json.loads(Path(path).read_text(encoding="utf-8"))
    if calibration.get("schema_version") not in (1, 2):
        raise ValueError("Unsupported camera self-mask calibration schema")
    return calibration


def validate_self_mask_settings(args):
    """Reject known profile/installation mismatches before connecting the arm."""
    path = getattr(args, "camera_self_mask_calibration", None)
    if not (getattr(args, "camera_self_mask", False) and path):
        return
    calibration = _load(path)
    expected = calibration["profile"]
    for key in ("serial", "width", "height", "depth_width", "depth_height"):
        actual = getattr(args, "camera_" + key, None)
        if key.startswith("depth_") and actual is None:
            actual = getattr(args, "camera_" + key.removeprefix("depth_"), None)
        if str(actual) != str(expected[key]):
            raise ValueError(
                f"Camera self-mask {key} mismatch: configured {actual}, "
                f"calibrated {expected[key]}. Use a matching self-mask calibration."
            )
    if getattr(args, "use_registered_points", False):
        raise ValueError("Calibrated camera self mask requires SDK capture metadata")
    opening = getattr(args, "gripper_open_width", None)
    if opening is not None and abs(float(opening) - calibration["gripper_open_width_m"]) > 0.0005:
        raise ValueError(
            f"Camera self-mask opening mismatch: requested {float(opening)*1000:.1f} mm; "
            f"calibration requires {calibration['gripper_open_width_m']*1000:.1f} mm. "
            "Capture a new silhouette calibration before changing the working opening."
        )


def calibrated_self_mask(bgr, calibration_path, capture_meta):
    """Return a verified mask and audit data, or raise before any cloud is built."""
    calibration = _load(calibration_path)
    profile = calibration["profile"]
    image = np.asarray(bgr)
    if image.dtype != np.uint8 or image.shape != (profile["height"], profile["width"], 3):
        raise ValueError("Camera self-mask RGB profile mismatch; calibration must not be rescaled")
    for key, meta_key in (("serial", "serial"), ("width", "width"), ("height", "height"),
                          ("depth_width", "native_depth_width"),
                          ("depth_height", "native_depth_height")):
        if str(capture_meta.get(meta_key)) != str(profile[key]):
            raise ValueError(f"Camera self-mask measured {key} mismatch")
    for key, expected in calibration["intrinsics"].items():
        actual = float(capture_meta.get("intrinsics", {}).get(key, float("nan")))
        if not np.isfinite(actual) or abs(actual - expected) > 0.5:
            raise ValueError(f"Camera self-mask measured intrinsic {key} mismatch")

    def rasterize(polygons):
        region = np.zeros(image.shape[:2], dtype=np.uint8)
        for polygon in polygons:
            vertices = np.asarray(polygon, dtype=float)
            if (vertices.ndim != 2 or vertices.shape[1] != 2 or len(vertices) < 3
                    or not np.isfinite(vertices).all()
                    or np.any(vertices < 0)
                    or np.any(vertices >= [profile["width"], profile["height"]])):
                raise ValueError("Invalid camera self-mask polygon")
            cv2.fillPoly(region, [np.rint(vertices).astype(np.int32)], 1)
        return region

    reference = rasterize(calibration["polygons_px"])
    # Version 2 stores both the measured multi-frame envelope and its stable
    # intersection. Validate the stable region; test for foreign dark pixels
    # outside the envelope. No per-frame search, shifting or growth is allowed.
    stable = reference
    if calibration["schema_version"] == 2:
        stable = rasterize(calibration["validation_polygons_px"])
        if np.any((stable > 0) & (reference == 0)):
            raise ValueError("Camera self-mask stable region exceeds its measured envelope")
    guard = calibration["rgb_guard"]
    dark = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) <= int(guard["max_gray"])
    color_guard = calibration.get("mask_color_guard")
    colored = np.zeros(image.shape[:2], dtype=bool)
    if color_guard is not None:
        colored = image.max(axis=2) - image.min(axis=2) > int(color_guard["max_channel_spread"])
    finger_dark = dark & ~colored
    core = cv2.erode(stable, np.ones((13, 13), np.uint8)) > 0
    outer = ((cv2.dilate(reference, np.ones((17, 17), np.uint8)) > 0)
             & ~(cv2.dilate(reference, np.ones((5, 5), np.uint8)) > 0))
    if not np.any(core) or not np.any(outer):
        raise ValueError("Camera self-mask calibration has no validation region")
    core_fraction = float(np.mean(dark[core]))
    silhouette_fraction = float(np.mean(dark[stable > 0]))
    # Match the actual masking rule: a coloured dark object next to a finger
    # is not a displaced black silhouette. Neutral black extensions remain
    # subject to the same outer-area threshold.
    outer_fraction = float(np.mean(finger_dark[outer]))
    report = dict(
        calibration_file=str(calibration_path), calibration=calibration,
        core_dark_fraction=core_fraction, silhouette_dark_fraction=silhouette_fraction,
        outer_dark_fraction=outer_fraction,
        outer_gray_dark_fraction=float(np.mean(dark[outer])),
        outer_colored_dark_pixels=int(np.count_nonzero(outer & dark & colored)),
        outer_guard_mode="neutral_black" if color_guard is not None else "gray",
        envelope_dark_fraction=float(np.mean(dark[reference > 0])),
        stable_pixels=int(np.count_nonzero(stable)),
        envelope_pixels=int(np.count_nonzero(reference)),
    )
    if (core_fraction < guard["min_core_dark_fraction"]
            or silhouette_fraction < guard["min_silhouette_dark_fraction"]
            or outer_fraction > guard["max_outer_dark_fraction"]):
        report["validation_passed"] = False
        raise CameraSelfMaskMismatch(
            "Camera self-mask silhouette mismatch: "
            f"core={core_fraction:.3f} (min {guard['min_core_dark_fraction']:.3f}), "
            f"silhouette={silhouette_fraction:.3f} (min {guard['min_silhouette_dark_fraction']:.3f}), "
            f"outer={outer_fraction:.3f} (max {guard['max_outer_dark_fraction']:.3f}). "
            f"Calibration opening={calibration['gripper_open_width_m']*1000:.1f} mm. "
            "Check the measured opening and capture a matching finger silhouette; "
            "target detection and grasp planning have not run for this frame.", report,
        )
    mask = (reference > 0) & finger_dark
    report["colored_pixels_preserved"] = int(np.count_nonzero((reference > 0) & dark & colored))
    report["validation_passed"] = True
    report["masked_pixels"] = int(np.count_nonzero(mask))
    return mask.astype(np.uint8) * 255, report


def apply_calibrated_self_mask(xyz_img, bgr, args, out_dir):
    """Apply before target/scene/obstacle sampling and save the actual mask."""
    if out_dir is None:
        raise ValueError("Camera self-mask requires capture_meta.json from this capture")
    directory = Path(out_dir)
    metadata = json.loads((directory / "capture_meta.json").read_text(encoding="utf-8"))
    try:
        mask, report = calibrated_self_mask(bgr, args.camera_self_mask_calibration, metadata)
    except CameraSelfMaskMismatch as exc:
        overlay = np.asarray(bgr).copy()
        polygons = [np.rint(p).astype(np.int32) for p in exc.report["calibration"]["polygons_px"]]
        cv2.polylines(overlay, polygons, True, (0, 0, 255), 3)
        cv2.imwrite(str(directory / "camera_self_mask_mismatch.png"), overlay)
        (directory / "camera_self_mask_report.json").write_text(
            json.dumps(exc.report, indent=2) + "\n", encoding="utf-8")
        raise CameraSelfMaskMismatch(f"{exc} Diagnostics: {directory}", exc.report) from None
    xyz = np.asarray(xyz_img, dtype=np.float32).copy()
    if xyz.shape != (*mask.shape, 3):
        raise ValueError("Camera self-mask XYZ alignment mismatch")
    valid = np.isfinite(xyz).all(axis=2) & (xyz[:, :, 2] > 0)
    report["removed_valid_depth"] = int(np.count_nonzero(valid & (mask > 0)))
    xyz[mask > 0] = np.nan
    overlay = np.asarray(bgr).copy()
    selected = mask > 0
    overlay[selected] = (0.35 * overlay[selected] + 0.65 * np.array([0, 0, 255])).astype(np.uint8)
    for filename, pixels in (("camera_self_mask.png", mask), ("camera_self_mask_overlay.png", overlay)):
        if not cv2.imwrite(str(directory / filename), pixels):
            raise OSError(f"Cannot save {filename}")
    (directory / "camera_self_mask_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"[camera][self_mask] calibrated removed_valid_depth={report['removed_valid_depth']} "
          f"pixels={report['masked_pixels']} core_dark={report['core_dark_fraction']:.3f}")
    return xyz
