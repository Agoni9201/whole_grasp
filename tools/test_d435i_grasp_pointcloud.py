#!/usr/bin/env python3
"""Camera-only D435i/recorded RGB-D probe; no ROS or arm connection."""
import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
from real_nero_grasp.depth_processing import depth_to_xyz_image
from real_nero_grasp.perception import (
    cloud_from_mask, scene_cloud_from_xyz, infer_yolo_instances, mask_for_instance,
)
from tools.verify_handeye_point import capture_realsense_frame


def xyz_from_frame(bgr, depth_m, intr):
    if bgr is None or bgr.ndim != 3 or bgr.shape[2] != 3 or bgr.dtype != np.uint8:
        raise ValueError("color must be HxWx3 uint8 BGR")
    if depth_m.ndim != 2 or depth_m.shape != bgr.shape[:2]:
        raise ValueError("depth must be aligned to color with matching H/W")
    if not np.issubdtype(depth_m.dtype, np.floating):
        raise ValueError("depth must be floating-point metres, not raw z16")
    values = np.array([intr.fx, intr.fy, intr.cx, intr.cy], dtype=float)
    if not np.isfinite(values).all() or intr.fx <= 0 or intr.fy <= 0:
        raise ValueError("invalid intrinsics")
    return depth_to_xyz_image(depth_m, intr)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--capture-dir", type=Path, help="Replay color.png/depth_m.npy/capture_meta.json")
    p.add_argument("--output", type=Path, required=True, help="New output directory; never overwrite")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--warmup", type=int, default=30)
    p.add_argument("--mask", type=Path, help="Optional HxW nonzero foreground PNG")
    p.add_argument("--yolo-weights", help="Optional existing segmentation checkpoint")
    p.add_argument("--target-class")
    p.add_argument("--target-index", type=int, default=0)
    p.add_argument("--view", action="store_true")
    p.add_argument("--min-depth", type=float, default=0.05)
    p.add_argument("--max-depth", type=float, default=1.50)
    p.add_argument("--min-target-points", type=int, default=250)
    p.add_argument("--voxel-size", type=float, default=0.0015)
    p.add_argument("--scene-voxel-size", type=float, default=0.004)
    p.add_argument("--no-target-outlier-filter", dest="target_outlier_filter", action="store_false")
    args = p.parse_args()
    if args.mask and args.yolo_weights:
        p.error("choose --mask or --yolo-weights")
    if args.yolo_weights and not Path(args.yolo_weights).is_file():
        p.error("YOLO checkpoint must already exist locally")
    if not 0 <= args.min_depth < args.max_depth or min(args.voxel_size, args.scene_voxel_size) < 0:
        p.error("invalid depth bounds or voxel sizes")
    args.yolo_classes, args.yolo_conf, args.yolo_imgsz = None, 0.25, 640
    args.box_depth_band = 0.045
    args.outlier_neighbors, args.outlier_std_ratio, args.scene_cloud_cap = 20, 2.0, 60000
    args.output.mkdir(parents=True, exist_ok=False)
    if args.capture_dir:
        root = args.capture_dir
        bgr = cv2.imread(str(root / "color.png"), cv2.IMREAD_COLOR)
        depth = np.load(root / "depth_m.npy", allow_pickle=False)
        meta = json.loads((root / "capture_meta.json").read_text())
        intr = SimpleNamespace(**{k: float(meta["intrinsics"][k]) for k in ("fx", "fy", "cx", "cy")})
    else:
        import pyrealsense2 as rs
        devices = list(rs.context().query_devices())
        if len(devices) != 1 or "D435I" not in devices[0].get_info(rs.camera_info.name).upper():
            raise RuntimeError("Connect exactly one RealSense D435i; no automatic device guessing")
        print("camera:", devices[0].get_info(rs.camera_info.name), devices[0].get_info(rs.camera_info.serial_number))
        bgr, depth, intr, root = capture_realsense_frame(
            str(args.output / "capture"), args.width, args.height, args.fps, args.warmup)
        print("RGB-D replay directory:", root)
    xyz = xyz_from_frame(bgr, depth, intr)
    mask = np.ones(depth.shape, dtype=bool)
    if args.mask:
        image = cv2.imread(str(args.mask), cv2.IMREAD_GRAYSCALE)
        if image is None or image.shape != depth.shape:
            raise ValueError("mask must match frame H/W")
        mask = image > 0
    elif args.yolo_weights:
        instances, _ = infer_yolo_instances(bgr, args, str(args.output / "yolo.png"))
        for i, item in enumerate(instances):
            print("instance", i, item["label"], item["score"])
        if not 0 <= args.target_index < len(instances):
            raise RuntimeError("No instance at target-index; inspect YOLO overlay")
        chosen = instances[args.target_index]
        if chosen.get("is_box_only", False):
            raise RuntimeError("Segmentation mask missing; refusing box-only target comparison")
        mask = mask_for_instance(chosen, xyz, args)
    else:
        print("No target mask: showing entire scene; this is not an object-specific evaluation")
    valid = np.isfinite(xyz).all(axis=2) & (depth > args.min_depth) & (depth < args.max_depth)
    print("frame H/W:", depth.shape, "XYZ:", xyz.shape, "metres")
    print("selected pixels:", int(mask.sum()), "valid:", int((valid & mask).sum()),
          "valid fraction:", float((valid & mask).sum()) / max(1, int(mask.sum())))
    raw_args = SimpleNamespace(**vars(args))
    raw_args.voxel_size, raw_args.target_outlier_filter = 0.0, False
    raw = cloud_from_mask(xyz, bgr, mask, raw_args)
    filtered = cloud_from_mask(xyz, bgr, mask, args)
    scene = scene_cloud_from_xyz(xyz, bgr, args)
    print("target points raw -> filtered:", len(raw.points), "->", len(filtered.points))
    if len(filtered.points) < 3000:
        print("Fewer than 3000 points: default GraspNet sampling will duplicate points")
    import open3d as o3d
    for name, cloud in (("raw", raw), ("filtered", filtered), ("scene", scene)):
        if not o3d.io.write_point_cloud(str(args.output / (name + ".ply")), cloud):
            raise OSError("PLY write failed: " + name)
    if not cv2.imwrite(str(args.output / "selected_mask.png"), mask.astype(np.uint8) * 255):
        raise OSError("mask write failed")
    if args.view:
        o3d.visualization.draw_geometries([raw], window_name="Raw camera-frame cloud")
        o3d.visualization.draw_geometries([filtered], window_name="Grasp filtered cloud")


if __name__ == "__main__":
    main()