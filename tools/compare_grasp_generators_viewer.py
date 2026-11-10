#!/usr/bin/env python3
"""View saved GraspNet and GraspGenX proposals on the same object cloud.

This is a visual diagnostic. GraspNet uses its native parallel-jaw schematic;
GraspGenX uses the selected gripper's open mesh or stored open point cloud.
The two learned scores have different meanings and must not be compared.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parents[1]
GENX_ROOT = ROOT / "third_party" / "GraspGenX"
sys.path.insert(0, str(GENX_ROOT))

from graspgenx.utils.viser_utils import create_visualizer, matrix_to_wxyz_position, visualize_mesh, visualize_pointcloud
from graspgenx.x_grippers import resolve_gripper_info


def graspnet_schematic(width: float, depth: float) -> trimesh.Trimesh:
    """Same four-box dimensions as graspnetAPI.plot_gripper_pro_max."""
    finger = 0.004
    base_depth = 0.020
    tail_length = 0.040
    thickness = 0.004
    specs = (
        ((depth + base_depth + finger, finger, thickness),
         ((depth - base_depth - finger) / 2, -width / 2 - finger / 2, 0)),
        ((depth + base_depth + finger, finger, thickness),
         ((depth - base_depth - finger) / 2, width / 2 + finger / 2, 0)),
        ((finger, width, thickness), (-base_depth - finger / 2, 0, 0)),
        ((tail_length, finger, thickness),
         (-tail_length / 2 - finger - base_depth, 0, 0)),
    )
    pieces = []
    for size, center in specs:
        part = trimesh.creation.box(extents=size)
        part.apply_translation(center)
        pieces.append(part)
    return trimesh.util.concatenate(pieces)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-json", type=Path, required=True)
    parser.add_argument("--graspnet-json", type=Path, required=True)
    parser.add_argument("--graspgenx-json", type=Path, required=True)
    parser.add_argument("--table-preview-json", type=Path,
                        help="Saved final_grasp_preview.json for the shared table plane")
    parser.add_argument("--table-report-json", type=Path,
                        help="Output from check_graspgenx_table.py")
    parser.add_argument("--safe-only", action="store_true",
                        help="Browse only GenX candidates that passed the table-plane check")
    parser.add_argument("--port", type=int, default=8081)
    args = parser.parse_args()

    cloud_data = json.loads(args.input_json.read_text(encoding="utf-8"))
    net = json.loads(args.graspnet_json.read_text(encoding="utf-8"))
    genx = json.loads(args.graspgenx_json.read_text(encoding="utf-8"))
    points = np.asarray(cloud_data["pc"], dtype=np.float64)
    colors = np.asarray(cloud_data["pc_color"], dtype=np.uint8)
    if points.ndim != 2 or points.shape[1] != 3 or colors.shape != points.shape:
        raise ValueError("Input JSON must contain matching Nx3 pc and pc_color arrays")
    if not np.isfinite(points).all():
        raise ValueError("Input point cloud contains nonfinite coordinates")
    if Path(net["input_json"]).resolve() != args.input_json.resolve():
        raise ValueError("GraspNet result was generated from a different input JSON")
    if genx["source_name"] != args.input_json.stem or genx["input_point_count"] != len(points):
        raise ValueError("GraspGenX result does not match the requested object")
    if not np.allclose(genx["input_cloud_mean"], points.mean(axis=0), atol=1e-6):
        raise ValueError("GraspGenX result uses a different point cloud centroid")
    if net["seed"] != genx["seed"]:
        raise ValueError("Results use different seeds; choose matching files")
    if not net["candidates"] or not genx["scores"]:
        raise ValueError("Both result files must contain at least one candidate")

    gripper = resolve_gripper_info(genx["gripper_name"], str(GENX_ROOT / "assets"))
    open_points = np.asarray(gripper.open_pointcloud, dtype=np.float32)
    if open_points.ndim != 2 or open_points.shape[1] != 3 or not len(open_points):
        raise ValueError("Open-gripper point cloud is missing")
    genx_order = np.argsort(-np.asarray(genx["scores"], dtype=float))
    genx_poses = np.asarray(genx["poses_input_frame"], dtype=float)
    if genx_poses.shape != (len(genx_order), 4, 4):
        raise ValueError("Invalid GraspGenX pose array")
    table_report = None
    if args.table_report_json is not None:
        table_report = json.loads(args.table_report_json.read_text(encoding="utf-8"))
        if Path(table_report["candidate_json"]).resolve() != args.graspgenx_json.resolve():
            raise ValueError("Table report was generated for different GraspGenX candidates")
        if len(table_report["records"]) != len(genx_order):
            raise ValueError("Table report and GraspGenX candidate counts differ")
    if args.safe_only:
        if table_report is None:
            parser.error("--safe-only requires --table-report-json")
        genx_order = np.asarray(
            [i for i in genx_order if table_report["records"][int(i)]["table_safe"]],
            dtype=int,
        )
        if not len(genx_order):
            raise ValueError("No table-safe GraspGenX candidates to display")

    mean = points.mean(axis=0)
    spacing = max(float(np.ptp(points, axis=0).max()) * 1.8, 0.24)
    offsets = (np.array([-spacing / 2, 0, 0]), np.array([spacing / 2, 0, 0]))
    viewer = create_visualizer(port=args.port)
    for label, offset in (("graspnet", offsets[0]), ("graspgenx", offsets[1])):
        visualize_pointcloud(viewer, f"{label}/object", points - mean + offset, colors, size=0.002)
    if args.table_preview_json is not None:
        preview = json.loads(args.table_preview_json.read_text(encoding="utf-8"))
        T_base_camera = np.asarray(preview["T_base_camera0"], dtype=float)
        T_camera_base = np.linalg.inv(T_base_camera)
        center_base = (T_base_camera @ np.r_[mean, 1.0])[:3]
        table_z = table_report["table_z_m"] if table_report is not None else 0.0
        table_pose = T_camera_base.copy()
        table_pose[:3, 3] = (T_camera_base @ np.r_[center_base[0], center_base[1], table_z, 1.0])[:3]
        for label, offset in (("graspnet", offsets[0]), ("graspgenx", offsets[1])):
            shifted = table_pose.copy()
            shifted[:3, 3] += offset - mean
            wxyz, position = matrix_to_wxyz_position(shifted)
            viewer.scene.add_grid(
                f"{label}/table", width=0.32, height=0.32,
                cell_size=0.025, section_size=0.05,
                cell_color=(180, 80, 80), section_color=(140, 40, 40),
                plane_color=(235, 160, 160), plane_opacity=0.12,
                wxyz=wxyz, position=position,
            )
    explanation = (
        f"**左：GraspNet 通用夹爪示意；右：GraspGenX {genx['gripper_name']} 张开夹爪。** "
        "两边使用同一份目标点云；两种网络分数不可直接比较。"
    )
    if table_report is not None:
        explanation += "桌面状态只标注右侧候选，左侧是未执行生产流程筛选的原始提议。"
    viewer.gui.add_markdown(explanation)
    net_status = viewer.gui.add_markdown("")
    genx_status = viewer.gui.add_markdown("")
    handles = {"net": None, "genx": None}

    def show_net(rank: int) -> None:
        if handles["net"] is not None:
            handles["net"].remove()
        row = net["candidates"][rank - 1]
        pose = np.asarray(row["pose_input_frame"], dtype=float).copy()
        pose[:3, 3] += offsets[0] - mean
        mesh = graspnet_schematic(float(row["width_m"]), float(row["depth_m"]))
        handles["net"] = visualize_mesh(viewer, "graspnet/current_gripper", mesh,
                                        color=[255, 133, 43], transform=pose)
        net_status.content = f"**GraspNet 第 {rank} 名** · 网络分数 {row['score']:.3f} · 开口 {row['width_m'] * 1000:.1f} mm"

    def show_genx(rank: int) -> None:
        if handles["genx"] is not None:
            handles["genx"].remove()
        index = int(genx_order[rank - 1])
        pose = genx_poses[index].copy()
        pose[:3, 3] += offsets[1] - mean
        if len(gripper.collision_mesh.vertices):
            handles["genx"] = visualize_mesh(
                viewer, "graspgenx/current_gripper", gripper.collision_mesh,
                color=[40, 120, 255], transform=pose,
            )
        else:
            wxyz, position = matrix_to_wxyz_position(pose)
            handles["genx"] = viewer.scene.add_point_cloud(
                "graspgenx/current_gripper",
                points=open_points,
                colors=np.tile(np.array([[40, 120, 255]], dtype=np.uint8), (len(open_points), 1)),
                point_size=0.0014,
                wxyz=wxyz,
                position=position,
            )
        table_text = ""
        if table_report is not None:
            row = table_report["records"][index]
            state = "桌面通过" if row["table_safe"] else "桌面碰撞"
            table_text = f" · **{state}** · 最小余量 {row['minimum_table_clearance_m'] * 1000:.1f} mm"
        genx_status.content = (
            f"**GraspGenX 第 {rank} 名** · 模型分数 {genx['scores'][index]:.3f}"
            f" · {genx['gripper_name']} 张开状态{table_text}"
        )

    net_slider = viewer.gui.add_slider("GraspNet 名次", min=1,
                                       max=len(net["candidates"]), step=1, initial_value=1)
    genx_slider = viewer.gui.add_slider("GraspGenX 名次", min=1,
                                        max=len(genx_order), step=1, initial_value=1)

    @net_slider.on_update
    def _on_net_update(_):
        show_net(int(net_slider.value))

    @genx_slider.on_update
    def _on_genx_update(_):
        show_genx(int(genx_slider.value))

    show_net(1)
    show_genx(1)
    print(f"Open http://localhost:{args.port}; Ctrl+C to stop")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Viewer stopped")


if __name__ == "__main__":
    main()
