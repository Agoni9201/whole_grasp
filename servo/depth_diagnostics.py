"""Read-only RGB-D evidence helpers; called by the diagnostic writer only."""
from pathlib import Path

import numpy as np


def depth_statistics(depth, minimum, maximum):
    """Separate missing depth from positive measurements rejected by range."""
    values = np.asarray(depth)
    finite = np.isfinite(values)
    positive = finite & (values > 0)
    valid = positive & (values >= minimum) & (values <= maximum)
    measured = values[positive]
    return {
        "pixels": int(values.size),
        "zero_or_negative": int(np.count_nonzero(finite & (values <= 0))),
        "nonfinite": int(np.count_nonzero(~finite)),
        "below_minimum": int(np.count_nonzero(positive & (values < minimum))),
        "above_maximum": int(np.count_nonzero(positive & (values > maximum))),
        "valid": int(np.count_nonzero(valid)),
        "valid_percent": 100.0 * np.count_nonzero(valid) / max(1, values.size),
        "positive_depth_p10_p50_p90_m": (
            np.percentile(measured, [10, 50, 90]).tolist() if measured.size else None),
    }


def analyze_frame(frame):
    """Inspect the full aligned image and the six regions actually sampled."""
    depth = frame["depth_m"]
    minimum, maximum = frame["minimum_m"], frame["maximum_m"]
    height, width = depth.shape
    radius = frame["radius"]
    regions = {}
    pixels = frame["landmarks_pixel"]
    if pixels is not None:
        centers = {"palm": frame["palm_pixel"]}
        centers.update({f"landmark_{i}": pixels[i] for i in (0, 5, 9, 13, 17)})
        for label, point in centers.items():
            u, v = map(int, point)
            patch = depth[max(0, v-radius):min(height, v+radius+1),
                          max(0, u-radius):min(width, u+radius+1)]
            regions[label] = {"pixel": [u, v], **depth_statistics(patch, minimum, maximum)}
    return {
        "frame_sequence": frame["sequence"], "reason": frame["reason"],
        "received_monotonic": frame["received_monotonic"],
        "minimum_m": minimum, "maximum_m": maximum, "radius": radius,
        "scene": depth_statistics(depth, minimum, maximum), "regions": regions,
    }


def save_frame(directory, frame, index):
    """Preserve exact aligned RGB/depth plus geometry for offline reproduction."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"frame_{index:02d}_seq_{frame['sequence']}.npz"
    with path.open("xb") as stream:
        extra = {key: frame[key] for key in ('raw_depth_m', 'filtered_depth_m')
                 if frame.get(key) is not None}
        np.savez(stream, bgr=frame["bgr"], depth_m=frame["depth_m"],
                 **extra,
                 K=frame["K"], D=frame["D"], q_image_rad=frame["q_image"],
                 landmarks_pixel=(np.empty((0, 2), dtype=int) if frame["landmarks_pixel"] is None
                                  else frame["landmarks_pixel"]),
                 palm_pixel=(np.empty(0, dtype=int) if frame["palm_pixel"] is None
                             else frame["palm_pixel"]),
                 metadata_json=np.asarray(frame["metadata_json"]))
    # A paired overview is for inspection only; the archive keeps unmodified
    # floating-point metres, including zeros/NaNs and out-of-range values.
    import cv2
    depth = frame["depth_m"]
    valid = (np.isfinite(depth) & (depth >= frame["minimum_m"])
             & (depth <= frame["maximum_m"]) & (depth > 0))
    scaled = np.zeros(depth.shape, dtype=np.uint8)
    scaled[valid] = np.clip((depth[valid] - frame["minimum_m"])
                            / (frame["maximum_m"] - frame["minimum_m"]) * 255, 0, 255)
    depth_image = cv2.applyColorMap(scaled, cv2.COLORMAP_TURBO)
    depth_image[~valid] = 0
    rejected = np.isfinite(depth) & (depth > 0) & ~valid
    depth_image[rejected] = (255, 0, 255)
    color = frame["bgr"].copy()
    if frame["landmarks_pixel"] is not None:
        for i in (0, 5, 9, 13, 17):
            center = tuple(map(int, frame["landmarks_pixel"][i]))
            for image in (color, depth_image):
                cv2.circle(image, center, frame["radius"], (0, 255, 0), 1)
    overview = np.hstack((color, depth_image))
    cv2.putText(overview, "RGB | aligned depth: black=missing, magenta=out of range",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    if not cv2.imwrite(str(path.with_suffix(".png")), overview):
        raise OSError("Cannot save RGB-D diagnostic overview")
    return str(path)
