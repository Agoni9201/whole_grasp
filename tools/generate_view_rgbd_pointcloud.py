#!/usr/bin/env python3
"""Generate and interactively display a colored point cloud from a saved RGB-D capture."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a colored single-view PLY from color.png, depth_m.npy and capture_meta.json."
    )
    parser.add_argument("log_dir", help="capture directory containing the three RGB-D files")
    parser.add_argument("--output", help="output PLY path (default: LOG_DIR/single_view_rgb_cloud.ply)")
    parser.add_argument("--min-depth", type=float, default=0.10, help="minimum depth in metres")
    parser.add_argument("--max-depth", type=float, default=2.00, help="maximum depth in metres")
    parser.add_argument("--stride", type=int, default=1, help="keep every Nth image row/column")
    parser.add_argument("--voxel", type=float, default=0.0, help="optional voxel size in metres")
    parser.add_argument("--point-size", type=float, default=2.0)
    parser.add_argument("--no-view", action="store_true", help="only save the PLY; do not open a window")
    args = parser.parse_args()

    log_dir = Path(args.log_dir).expanduser().resolve()
    color_path = log_dir / "color.png"
    depth_path = log_dir / "depth_m.npy"
    meta_path = log_dir / "capture_meta.json"
    for path in (color_path, depth_path, meta_path):
        if not path.is_file():
            raise SystemExit(f"Required input does not exist: {path}")

    bgr = cv2.imread(str(color_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise SystemExit(f"Could not read image: {color_path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    depth = np.asarray(np.load(depth_path), dtype=np.float32)
    with meta_path.open("r", encoding="utf-8") as handle:
        meta = json.load(handle)
    intr = meta["intrinsics"]
    fx, fy = float(intr["fx"]), float(intr["fy"])
    cx, cy = float(intr["cx"]), float(intr["cy"])

    if depth.shape != rgb.shape[:2]:
        raise SystemExit(f"RGB/depth size mismatch: rgb={rgb.shape[:2]}, depth={depth.shape}")
    stride = max(1, int(args.stride))
    rows = np.arange(0, depth.shape[0], stride)
    cols = np.arange(0, depth.shape[1], stride)
    uu, vv = np.meshgrid(cols, rows)
    z = depth[np.ix_(rows, cols)]
    colors = rgb[np.ix_(rows, cols)]
    valid = np.isfinite(z) & (z >= args.min_depth) & (z <= args.max_depth)
    z_valid = z[valid]
    x = (uu[valid].astype(np.float32) - cx) * z_valid / fx
    y = (vv[valid].astype(np.float32) - cy) * z_valid / fy
    points = np.column_stack((x, y, z_valid))
    colors = colors[valid].astype(np.float64) / 255.0

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    cloud.colors = o3d.utility.Vector3dVector(colors)
    if args.voxel > 0:
        cloud = cloud.voxel_down_sample(float(args.voxel))
    if cloud.is_empty():
        raise SystemExit("No valid depth points remain; check --min-depth/--max-depth")

    output = Path(args.output).expanduser().resolve() if args.output else log_dir / "single_view_rgb_cloud.ply"
    if not o3d.io.write_point_cloud(str(output), cloud):
        raise SystemExit(f"Failed to write point cloud: {output}")
    print(f"Saved {len(cloud.points)} colored points: {output}")
    print("Coordinate frame: camera optical frame (+X right, +Y down, +Z forward)")

    if args.no_view:
        return
    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.10)
    visualizer = o3d.visualization.Visualizer()
    if not visualizer.create_window("Single-view RGB Point Cloud", width=1280, height=800):
        raise SystemExit("Could not create an Open3D window; check desktop/DISPLAY access")
    visualizer.add_geometry(cloud)
    visualizer.add_geometry(axes)
    options = visualizer.get_render_option()
    options.background_color = np.asarray([1.0, 1.0, 1.0])
    options.point_size = max(1.0, float(args.point_size))
    options.show_coordinate_frame = True
    print("Controls: left-drag rotate, wheel zoom, Shift+left-drag pan, Q/Esc close")
    visualizer.run()
    visualizer.destroy_window()


if __name__ == "__main__":
    main()
