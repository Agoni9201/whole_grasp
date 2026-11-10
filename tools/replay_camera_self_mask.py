#!/usr/bin/env python3
"""Rebuild saved obstacle views with the production self mask, without hardware.

Reuses the fixed numerical joint paths in a collision-witness JSON. This checks
map/path clearance, not contact, closed-gripper motion or grasp success.
"""
import argparse
import copy
import json
from pathlib import Path
import sys

import cv2
import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--witness-json", required=True, type=Path)
    parser.add_argument("--calibration", type=Path,
                        default=Path(__file__).resolve().parents[1] / "config/2026-10-22_camera_self_mask.json")
    parser.add_argument("--output-dir", required=True, type=Path)
    opts = parser.parse_args()
    from real_nero_grasp import runtime as r
    import open3d as o3d

    run = opts.run_dir.resolve()
    saved_argv = sys.argv
    try:
        sys.argv = json.loads((run / "experiment_result.json").read_text())["command"]
        args = r.parse_args()
    finally:
        sys.argv = saved_argv
    args.execute = False
    capture_root = next(run.glob("*/grasp_candidates.json")).parent
    opts.output_dir.mkdir(parents=True, exist_ok=True)
    witnesses = json.loads(opts.witness_json.read_text())
    if Path(witnesses["run"]).resolve() != run:
        raise ValueError("Collision witnesses are from a different run")
    report = dict(run=str(run), calibration=str(opts.calibration.resolve()),
                  method="Rebuild all non-target clouds from saved depth with the production mask before voxelisation; same fixed numerical joint paths and collision settings, no IK/hardware.",
                  limitation="Saved paths were solved offline and may differ from online IK. No complete grasp/close/lift validation.",
                  views={}, cases=[])
    fixed_args = copy.copy(args)
    fixed_args.camera_self_mask = True
    fixed_args.camera_self_mask_calibration = str(opts.calibration.resolve())
    views = {}
    for path in (capture_root / "multiview_raw").glob("*/metadata.json"):
        meta = json.loads(path.read_text())
        label = meta["view_label"]
        source = path.parent
        image = cv2.imread(str(source / "color.png"))
        depth = np.load(source / "depth_m.npy")
        intrinsics = type("Intrinsics", (), meta["intrinsics"])()
        xyz = r.depth_to_xyz_image(depth, intrinsics)
        selected = cv2.imread(str(source / "selected_target_mask.png"), cv2.IMREAD_GRAYSCALE) > 0
        directory = opts.output_dir / label
        directory.mkdir(exist_ok=True)
        (directory / "capture_meta.json").write_bytes((source / "capture_meta.json").read_bytes())
        filtered = r.apply_camera_self_occlusion_mask(xyz, image, fixed_args, str(directory))
        mask = cv2.imread(str(directory / "camera_self_mask.png"), cv2.IMREAD_GRAYSCALE) > 0
        if np.any(mask & selected):
            raise ValueError(f"{label}: self mask overlaps selected target")
        np.testing.assert_array_equal(filtered[~mask], xyz[~mask])
        # Verify baseline reconstruction against archived clouds before comparing
        # the fix, rather than silently changing another processing stage.
        before_cloud = r.obstacle_cloud_from_non_target_mask(xyz, image, selected, args)
        fixed_cloud = r.obstacle_cloud_from_non_target_mask(filtered, image, selected, args)
        archived = o3d.io.read_point_cloud(str(source / "obstacle_camera.ply"))
        before = np.asarray(before_cloud.points)
        original = np.asarray(archived.points)
        maximum_error = float(np.max(cKDTree(original).query(before)[0]))
        reverse_error = float(np.max(cKDTree(before).query(original)[0]))
        if len(before) != len(original) or max(maximum_error, reverse_error) > 1e-6:
            raise ValueError(f"{label}: reconstructed baseline differs from archive; cannot attribute changes solely to self mask")
        T = np.asarray(meta["T_base_camera"])
        fixed_cloud.transform(T)
        info = dict(archived_points=len(original), fixed_points=len(fixed_cloud.points),
                    baseline_reconstruction_max_error_m=max(maximum_error, reverse_error),
                    target_overlap_pixels=int(np.count_nonzero(mask & selected)),
                    changed_pixels_outside_mask=0,
                    mask_report=json.loads((directory / "camera_self_mask_report.json").read_text()))
        report["views"][label] = info
        views[label] = dict(meta=meta, original=o3d.io.read_point_cloud(str(source / "obstacle_base.ply")),
                            fixed=fixed_cloud, target=o3d.io.read_point_cloud(str(source / "target_base.ply")))

    order = [name for name in ("initial", "center", "side1", "side2") if name in views]
    labels = [name if not name.startswith("side") else "side_capture_" + name[4:] for name in order]
    target_union = np.vstack([np.asarray(views[name]["target"].points) for name in order])
    # Match active_view._target_center_base_from_cloud: median of the initial
    # capture, not of all views or of the registered/fused target.
    target_center = np.median(np.asarray(views["initial"]["target"].points,
                                        dtype=np.float32), axis=0).astype(float)
    if "target_center_base" in witnesses:
        target_center = np.asarray(witnesses["target_center_base"])
    target = o3d.io.read_point_cloud(str(capture_root / "graspnet_target_full.ply"))
    captured_scene = o3d.io.read_point_cloud(str(capture_root / "active_fused_scene_camera0.ply"))
    T = np.asarray(views["initial"]["meta"]["T_base_camera"])
    scenes = {}
    for kind in ("original", "fixed"):
        occupancy = r.fuse_dynamic_environment_clouds(
            labels, [views[name][kind] for name in order], args,
            target_base=target_union, target_center_base=target_center,
            map_id="offline_self_mask_" + kind, generation=1)
        scenes[kind] = r.prepare_motion_safety_scene(target, captured_scene, T, args,
            map_id="offline_self_mask_" + kind, environment_map=occupancy)
    report["target_center_base"] = target_center.tolist()
    report["environment_points"] = {kind: len(scene.environment_base) for kind, scene in scenes.items()}
    if report["environment_points"]["original"] != witnesses["scene_points"]:
        raise ValueError("Original environment map does not match witness replay")
    output = opts.output_dir / "replay.json"
    for case in witnesses["cases"]:
        row = dict(idx=case["idx"], phase=case["phase"], q0=case["q0"], q1=case["q1"])
        for kind, scene in scenes.items():
            row[kind] = r._motion_joint_path_metrics(
                np.asarray(case["q0"]), np.asarray(case["q1"]),
                r.MotionCollisionContext(scene, case["phase"]), args)
        report["cases"].append(row)
        output.write_text(json.dumps(report, indent=2) + "\n")
        print(f"candidate={row['idx']} phase={row['phase']} "
              f"clearance_mm={row['original']['nearest_clearance']*1000:.2f} -> "
              f"{row['fixed']['nearest_clearance']*1000:.2f}", flush=True)
    o3d.io.write_point_cloud(str(opts.output_dir / "environment_fixed_base.ply"),
        o3d.geometry.PointCloud(o3d.utility.Vector3dVector(scenes["fixed"].environment_base)))
    print(output)


if __name__ == "__main__":
    main()
