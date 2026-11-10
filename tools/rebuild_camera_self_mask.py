#!/usr/bin/env python3
"""Build a fixed multi-frame finger envelope from reviewed saved RGB-D captures."""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from real_nero_grasp.camera_self_mask import calibrated_self_mask


def polygons(mask):
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                  cv2.CHAIN_APPROX_SIMPLE)
    return [c.reshape(-1, 2).tolist() for c in contours if cv2.contourArea(c) >= 4]


def rebuild(previous, captures, output):
    calibration = json.loads(Path(previous).read_text())
    profile = calibration["profile"]
    base = np.zeros((profile["height"], profile["width"]), np.uint8)
    cv2.fillPoly(base, [np.asarray(p, np.int32) for p in calibration["polygons_px"]], 1)
    yy, xx = np.where(base > 0)
    x0, x1 = max(0, xx.min()-20), min(base.shape[1], xx.max()+21)
    y0, y1 = max(0, yy.min()-20), base.shape[0]
    masks, images, metadata = [], [], []
    for directory in map(Path, captures):
        image = cv2.imread(str(directory / "color.png"))
        meta = json.loads((directory / "capture_meta.json").read_text())
        if image is None or image.shape != (*base.shape, 3):
            raise ValueError(f"RGB profile mismatch: {directory}")
        for key, actual in (("serial", "serial"), ("width", "width"), ("height", "height"),
                            ("depth_width", "native_depth_width"), ("depth_height", "native_depth_height")):
            if str(meta[actual]) != str(profile[key]):
                raise ValueError(f"Profile mismatch {key}: {directory}")
        for key, expected in calibration["intrinsics"].items():
            if abs(meta["intrinsics"][key] - expected) > .5:
                raise ValueError(f"Intrinsic mismatch {key}: {directory}")
        dark = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) <= calibration["rgb_guard"]["max_gray"]
        count, labels, stats, _ = cv2.connectedComponentsWithStats(
            dark[y0:y1, x0:x1].astype(np.uint8))
        if count < 2:
            raise ValueError(f"No measured finger: {directory}")
        index = max(range(1, count), key=lambda i: stats[i, cv2.CC_STAT_AREA])
        component = labels == index
        if component[0].any() or component[:, 0].any() or component[:, -1].any() or not component[-1].any():
            raise ValueError(f"Finger crosses the reviewed ROI or is not bottom-connected: {directory}")
        mask = np.zeros_like(base)
        mask[y0:y1, x0:x1] = component
        overlap = np.count_nonzero(mask & base) / np.count_nonzero(mask | base)
        if overlap < .90:
            raise ValueError(f"Capture differs from the reviewed finger by too much: IoU={overlap:.3f}")
        masks.append(mask)
        images.append(image)
        metadata.append(meta)
    if len(masks) < 3:
        raise ValueError("At least three independently captured frames are required")
    envelope = np.maximum.reduce(masks)
    stable = np.minimum.reduce(masks)
    if np.count_nonzero(stable) / np.count_nonzero(envelope) < .90:
        raise ValueError("Finger variation exceeds the fixed calibration envelope budget")
    calibration.update(schema_version=2, polygons_px=polygons(envelope),
                       validation_polygons_px=polygons(stable),
                       sources=[str(Path(p) / "color.png") for p in captures],
                       description="Fixed 100 mm finger envelope and stable intersection from repeated saved captures. Online RGB checks use unchanged thresholds; no runtime mask growth or alignment.")
    core = cv2.erode(stable, np.ones((13, 13), np.uint8)) > 0
    if not core.any():
        raise ValueError("No stable finger core")
    calibration["mask_color_guard"] = {
        "max_channel_spread": max(int((im.max(axis=2)-im.min(axis=2))[core].max()) for im in images),
        "source": "Maximum observed RGB channel spread in the multi-frame eroded stable finger core",
    }
    calibration["provenance"] = {
        "previous_calibration": str(previous), "captures": len(masks),
        "stable_pixels": int(np.count_nonzero(stable)),
        "envelope_pixels": int(np.count_nonzero(envelope)),
        "boundary_variation_pixels": int(np.count_nonzero(envelope - stable)),
        "calibration_method": "reviewed bottom-connected finger component in a bounded ROI, offline only",
    }
    output = Path(output)
    output.write_text(json.dumps(calibration, indent=2) + "\n")
    reports = []
    for image, meta in zip(images, metadata):
        _, report = calibrated_self_mask(image, output, meta)
        reports.append({k: report[k] for k in ("core_dark_fraction", "silhouette_dark_fraction",
                                               "outer_dark_fraction", "masked_pixels")})
    return reports


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous", type=Path, required=True)
    parser.add_argument("--captures", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(rebuild(args.previous, args.captures, args.output), indent=2))
