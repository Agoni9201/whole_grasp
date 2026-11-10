"""Pure RGB-D image, depth-hole, and binary-mask processing helpers."""

import os

import cv2
import numpy as np


__all__ = [
    "pointcloud2_xyz_image",
    "depth_to_xyz_image",
    "fill_small_xyz_holes_in_mask",
    "refine_xyz_inside_visible_mask",
    "_resize_mask",
    "_mask_center",
    "pointcloud2_xyz_at_from_image",
    "_mask_bbox",
    "_expand_bbox_xyxy",
    "_bbox_mask",
    "_mask_iou",
    "_mask_nested_overlap",
    "_median_depth_for_mask",
]


def pointcloud2_xyz_image(cloud_msg):
    h, w = int(cloud_msg.height), int(cloud_msg.width)
    if h <= 1:
        raise ValueError("PointCloud2 is unorganized; cannot build XYZ image")
    fields = {field.name: field for field in cloud_msg.fields}
    for name in ("x", "y", "z"):
        if name not in fields:
            raise ValueError(f"PointCloud2 missing field {name!r}")
        if int(fields[name].datatype) != 7:
            raise ValueError(f"PointCloud2 field {name!r} is not FLOAT32")
    endian = ">" if cloud_msg.is_bigendian else "<"
    dtype = np.dtype(
        {
            "names": ["x", "y", "z"],
            "formats": [endian + "f4", endian + "f4", endian + "f4"],
            "offsets": [int(fields["x"].offset), int(fields["y"].offset), int(fields["z"].offset)],
            "itemsize": int(cloud_msg.point_step),
        }
    )
    arr = np.ndarray(
        shape=(h, w),
        dtype=dtype,
        buffer=cloud_msg.data,
        strides=(int(cloud_msg.row_step), int(cloud_msg.point_step)),
    )
    xyz = np.dstack([arr["x"], arr["y"], arr["z"]]).astype(np.float32, copy=True)
    xyz[~np.isfinite(xyz).all(axis=2)] = np.nan
    xyz[xyz[:, :, 2] <= 0.0] = np.nan
    return xyz

def depth_to_xyz_image(depth_m, intr):
    depth_m = np.asarray(depth_m, dtype=np.float32)
    if depth_m.ndim != 2:
        raise ValueError(f"depth image must be HxW, got shape {depth_m.shape}")
    height, width = depth_m.shape[:2]
    us, vs = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
    z = depth_m.astype(np.float32, copy=False)
    x = (us - float(intr.cx)) * z / float(intr.fx)
    y = (vs - float(intr.cy)) * z / float(intr.fy)
    xyz = np.dstack([x, y, z]).astype(np.float32, copy=False)
    xyz[~np.isfinite(xyz).all(axis=2)] = np.nan
    xyz[xyz[:, :, 2] <= 0.0] = np.nan
    return xyz

def fill_small_xyz_holes_in_mask(
    xyz_img,
    mask,
    intr,
    min_depth=0.05,
    max_depth=1.50,
    max_area=36,
    max_diameter=7,
    ring_radius=3,
    min_neighbors=8,
    max_depth_span=0.015,
):
    """Interpolate only tiny, enclosed depth dropouts inside a visible mask."""
    xyz = np.asarray(xyz_img, dtype=np.float32)
    mask_bool = np.asarray(mask).astype(bool)
    if xyz.ndim != 3 or xyz.shape[2] != 3 or mask_bool.shape != xyz.shape[:2]:
        raise ValueError(f"xyz/mask shape mismatch: xyz={xyz.shape}, mask={mask_bool.shape}")

    valid = (
        np.isfinite(xyz).all(axis=2)
        & (xyz[:, :, 2] > float(min_depth))
        & (xyz[:, :, 2] < float(max_depth))
    )
    holes = mask_bool & ~valid
    stats_out = {
        "invalid_before": int(np.count_nonzero(holes)),
        "components": 0,
        "filled_components": 0,
        "filled_pixels": 0,
    }
    fill_mask = np.zeros(mask_bool.shape, dtype=np.uint8)
    if not np.any(holes) or int(max_area) <= 0 or int(max_diameter) <= 0:
        return xyz.copy(), fill_mask, stats_out

    mask_u8 = mask_bool.astype(np.uint8)
    interior = cv2.erode(mask_u8, np.ones((3, 3), dtype=np.uint8), iterations=1) > 0
    count, labels, component_stats, _ = cv2.connectedComponentsWithStats(
        holes.astype(np.uint8),
        connectivity=8,
    )
    refined = xyz.copy()
    radius = max(int(ring_radius), 1)
    ring_kernel = np.ones((2 * radius + 1, 2 * radius + 1), dtype=np.uint8)

    for component_id in range(1, int(count)):
        component = labels == component_id
        area = int(component_stats[component_id, cv2.CC_STAT_AREA])
        width = int(component_stats[component_id, cv2.CC_STAT_WIDTH])
        height = int(component_stats[component_id, cv2.CC_STAT_HEIGHT])
        stats_out["components"] += 1
        if area > int(max_area) or max(width, height) > int(max_diameter):
            continue
        if not np.all(interior[component]):
            continue

        dilated = cv2.dilate(component.astype(np.uint8), ring_kernel, iterations=1) > 0
        neighbors = dilated & ~component & mask_bool & valid
        neighbor_depth = xyz[:, :, 2][neighbors]
        if neighbor_depth.size < int(min_neighbors):
            continue
        low, high = np.percentile(neighbor_depth, [10.0, 90.0])
        if not np.isfinite(low + high) or float(high - low) > float(max_depth_span):
            continue

        fill_depth = float(np.median(neighbor_depth))
        vs, us = np.nonzero(component)
        refined[vs, us, 2] = fill_depth
        refined[vs, us, 0] = (us.astype(np.float32) - float(intr.cx)) * fill_depth / float(intr.fx)
        refined[vs, us, 1] = (vs.astype(np.float32) - float(intr.cy)) * fill_depth / float(intr.fy)
        fill_mask[component] = 255
        stats_out["filled_components"] += 1
        stats_out["filled_pixels"] += area

    return refined, fill_mask, stats_out

def refine_xyz_inside_visible_mask(xyz_img, mask, intr, args, out_dir=None, label="target"):
    if not bool(getattr(args, "depth_small_hole_fill", True)):
        return np.asarray(xyz_img, dtype=np.float32)
    refined, fill_mask, stats = fill_small_xyz_holes_in_mask(
        xyz_img,
        mask,
        intr,
        min_depth=float(args.min_depth),
        max_depth=float(args.max_depth),
        max_area=int(args.depth_hole_max_area),
        max_diameter=int(args.depth_hole_max_diameter),
        ring_radius=int(args.depth_hole_ring_radius),
        min_neighbors=int(args.depth_hole_min_neighbors),
        max_depth_span=float(args.depth_hole_max_span),
    )
    print(
        f"[depth][holes] {label} invalid_inside_mask={stats['invalid_before']} "
        f"components={stats['components']} filled_components={stats['filled_components']} "
        f"filled_pixels={stats['filled_pixels']}"
    )
    if out_dir and stats["filled_pixels"] > 0:
        os.makedirs(out_dir, exist_ok=True)
        safe_label = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in str(label))
        cv2.imwrite(os.path.join(out_dir, f"{safe_label}_depth_small_hole_fill.png"), fill_mask)
        np.save(os.path.join(out_dir, f"{safe_label}_depth_refined_m.npy"), refined[:, :, 2])
    return refined

def _resize_mask(mask, width, height):
    mask = np.asarray(mask)
    if mask.shape[:2] != (height, width):
        mask = cv2.resize(mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST)
    return (mask > 0).astype(np.uint8)

def _mask_center(mask):
    ys, xs = np.nonzero(np.asarray(mask) > 0)
    if xs.size == 0:
        return None
    return int(np.median(xs)), int(np.median(ys))

def pointcloud2_xyz_at_from_image(xyz_img, u, v):
    u = int(round(u))
    v = int(round(v))
    if not (0 <= v < xyz_img.shape[0] and 0 <= u < xyz_img.shape[1]):
        raise ValueError(f"pixel outside image: {(u, v)}")
    p = xyz_img[v, u].astype(float)
    if not np.isfinite(p).all() or p[2] <= 0:
        raise ValueError(f"invalid point at {(u, v)}: {p.tolist()}")
    return p

def _mask_bbox(mask):
    ys, xs = np.nonzero(np.asarray(mask) > 0)
    if xs.size == 0:
        return np.array([0, 0, 0, 0], dtype=np.float32)
    return np.array([xs.min(), ys.min(), xs.max(), ys.max()], dtype=np.float32)

def _expand_bbox_xyxy(xyxy, width, height, margin):
    x1, y1, x2, y2 = np.asarray(xyxy, dtype=float).reshape(4)
    margin = float(margin)
    return np.array([
        max(0.0, x1 - margin),
        max(0.0, y1 - margin),
        min(float(width - 1), x2 + margin),
        min(float(height - 1), y2 + margin),
    ], dtype=np.float32)

def _bbox_mask(shape_hw, xyxy):
    h, w = shape_hw
    x1, y1, x2, y2 = np.round(np.asarray(xyxy, dtype=float).reshape(4)).astype(int).tolist()
    x1, x2 = max(0, x1), min(w - 1, x2)
    y1, y2 = max(0, y1), min(h - 1, y2)
    mask = np.zeros((h, w), dtype=bool)
    if x2 >= x1 and y2 >= y1:
        mask[y1:y2 + 1, x1:x2 + 1] = True
    return mask

def _mask_iou(a, b):
    ma = np.asarray(a) > 0
    mb = np.asarray(b) > 0
    inter = int(np.count_nonzero(ma & mb))
    union = int(np.count_nonzero(ma | mb))
    return float(inter / max(union, 1))

def _mask_nested_overlap(a, b):
    """Return intersection/smaller-area and smaller/larger-area mask ratios."""
    ma = np.asarray(a) > 0
    mb = np.asarray(b) > 0
    area_a = int(np.count_nonzero(ma))
    area_b = int(np.count_nonzero(mb))
    inter = int(np.count_nonzero(ma & mb))
    smaller = min(area_a, area_b)
    larger = max(area_a, area_b)
    return (
        float(inter / max(smaller, 1)),
        float(smaller / max(larger, 1)),
    )

def _median_depth_for_mask(xyz_img, mask, args):
    mask = np.asarray(mask) > 0
    valid = mask & np.isfinite(xyz_img).all(axis=2) & (xyz_img[:, :, 2] > args.min_depth) & (xyz_img[:, :, 2] < args.max_depth)
    vals = xyz_img[:, :, 2][valid]
    if vals.size == 0:
        return float("nan")
    return float(np.median(vals))

