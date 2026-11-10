"""Pure depth temporal fusion; safe to import without camera/GUI/robot SDKs."""

import warnings

import numpy as np


def median_valid_depth_frames(
    depth_frames,
    min_valid_frames=1,
    max_mad=None,
    *,
    return_stats=False,
):
    """Fuse settled depth frames and reject temporally unstable pixels.

    ``max_mad`` is expressed in metres.  A positive value removes pixels
    whose robust temporal median absolute deviation exceeds that value.  This
    targets High-Density flying depth without filling any holes or smoothing
    across a depth boundary.  The historical two-value return remains the
    default for offline callers.
    """
    stack = np.asarray(depth_frames, dtype=np.float32)
    if stack.ndim == 2:
        stack = stack[None, ...]
    if stack.ndim != 3 or stack.shape[0] == 0:
        raise ValueError(f"Expected depth frames shaped NxHxW, got {stack.shape}")

    required = int(np.clip(int(min_valid_frames), 1, stack.shape[0]))
    valid = np.isfinite(stack) & (stack > 0.0)
    valid_count = np.count_nonzero(valid, axis=0).astype(np.uint8)
    masked = np.where(valid, stack, np.nan)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="All-NaN slice encountered", category=RuntimeWarning)
        depth_m = np.nanmedian(masked, axis=0).astype(np.float32)
        deviations = np.where(
            valid,
            np.abs(stack - depth_m[None, ...]),
            np.nan,
        )
        temporal_mad = np.nanmedian(deviations, axis=0).astype(np.float32)
    depth_m[valid_count < required] = 0.0
    depth_m[~np.isfinite(depth_m)] = 0.0
    temporal_mad[~np.isfinite(temporal_mad)] = 0.0

    threshold = None
    if max_mad is not None:
        try:
            candidate = float(max_mad)
        except (TypeError, ValueError):
            candidate = 0.0
        if np.isfinite(candidate) and candidate > 0.0:
            threshold = candidate
    unstable = np.zeros(depth_m.shape, dtype=bool)
    if threshold is not None:
        # A pixel with too few valid samples was already invalidated above;
        # flag only valid pixels so the diagnostic describes actual temporal
        # disagreement rather than ordinary depth holes.
        unstable = (depth_m > 0.0) & (temporal_mad > threshold)
        depth_m[unstable] = 0.0
    if return_stats:
        return depth_m, valid_count, temporal_mad, unstable
    return depth_m, valid_count


def supported_depth_median(depth_frames, raw_depth_frames, min_valid_frames=1, max_mad=None):
    """Accept SDK denoising only where the SAME raw window supports a surface.

    Repeated SDK hole-fill/history values are not independent observations.
    Reuse the configured count/MAD thresholds on the unfiltered aligned frames;
    never turn rejected depth into free-space evidence or replace it with a plane.
    """
    processed = np.asarray(depth_frames, dtype=np.float32)
    raw = np.asarray(raw_depth_frames, dtype=np.float32)
    if processed.ndim != 3 or raw.shape != processed.shape:
        raise ValueError("Raw/processed temporal depth windows must have identical NxHxW shapes")
    sdk_depth, _ = median_valid_depth_frames(processed, min_valid_frames, max_mad)
    return apply_raw_depth_support(sdk_depth, raw, min_valid_frames, max_mad)


def apply_raw_depth_support(sdk_depth_m, raw_depth_frames, min_valid_frames=1, max_mad=None):
    """Apply the same production decision to an archived SDK median as well."""
    sdk_depth = np.asarray(sdk_depth_m, dtype=np.float32)
    raw = np.asarray(raw_depth_frames, dtype=np.float32)
    if sdk_depth.ndim != 2 or raw.ndim != 3 or raw.shape[1:] != sdk_depth.shape:
        raise ValueError("Raw depth window and SDK median shapes do not match")
    raw_depth, raw_count, _, raw_unstable = median_valid_depth_frames(
        raw, min_valid_frames, max_mad, return_stats=True,
    )
    valid = sdk_depth > 0
    unsupported = valid & (raw_depth <= 0)
    agreement = None
    if max_mad is not None and np.isfinite(float(max_mad)) and float(max_mad) > 0:
        agreement = 2 * float(max_mad)
    inconsistent = np.zeros(sdk_depth.shape, dtype=bool)
    if agreement is not None:
        inconsistent = valid & (raw_depth > 0) & (np.abs(sdk_depth - raw_depth) > agreement)
    rejected = unsupported | inconsistent
    output = sdk_depth.copy()
    output[rejected] = 0
    output[~np.isfinite(output) | (output <= 0)] = 0
    required = int(np.clip(int(min_valid_frames), 1, raw.shape[0]))
    report = {
        "mode": "same_window_raw_support", "frames": int(raw.shape[0]),
        "min_valid_frames": required, "max_mad_m": max_mad,
        "max_sdk_raw_median_difference_m": agreement,
        "sdk_valid_pixels": int(np.count_nonzero(valid)),
        "raw_supported_pixels": int(np.count_nonzero(raw_depth > 0)),
        "rejected_insufficient_raw_pixels": int(np.count_nonzero(valid & (raw_count < required))),
        "rejected_raw_unstable_pixels": int(np.count_nonzero(valid & raw_unstable)),
        "rejected_sdk_raw_difference_pixels": int(np.count_nonzero(inconsistent)),
        "rejected_pixels": int(np.count_nonzero(rejected)),
        "valid_pixels": int(np.count_nonzero(output > 0)),
        "missing_depth_is_free_space": False,
    }
    return {"depth_m": output, "sdk_depth_m": sdk_depth,
            "raw_valid_count": raw_count, "rejected_mask": rejected.astype(np.uint8),
            "report": report}
