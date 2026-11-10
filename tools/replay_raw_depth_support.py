#!/usr/bin/env python3
"""Replay raw-supported depth, cloud construction and endpoint checks offline."""

import argparse
import json
from pathlib import Path
import socket
import sys
import sysconfig

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [sysconfig.get_path("purelib"), str(ROOT)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--endpoint-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    options = parser.parse_args()
    original_socket = socket.socket
    class OfflineSocket(original_socket):
        def __init__(self, family=socket.AF_INET, *args, **kwargs):
            if family == getattr(socket, "AF_CAN", None):
                raise RuntimeError("CAN forbidden in raw-depth replay")
            super().__init__(family, *args, **kwargs)
    socket.socket = OfflineSocket
    import cv2
    import numpy as np
    from scipy.spatial import cKDTree
    from cameras.depth_fusion import apply_raw_depth_support
    from real_nero_grasp import runtime as runtime
    from real_nero_grasp.camera_self_mask import calibrated_self_mask

    run = options.run_dir.resolve()
    source = json.loads(options.endpoint_json.read_text())
    if Path(source["run"]).resolve() != run:
        raise ValueError("Saved joints belong to a different run")
    sys.argv = json.loads((run / "experiment_result.json").read_text())["command"]
    args = runtime.parse_args()
    args.execute = False
    args._planning_worker = True
    output = options.output_dir.resolve()
    output.mkdir(exist_ok=True, parents=True)
    capture = next(run.glob("*/grasp_candidates.json")).parent
    o3d = runtime.get_o3d()
    views, view_reports = {}, {}
    for path in (capture / "multiview_raw").glob("*/metadata.json"):
        metadata = json.loads(path.read_text())
        label, folder = metadata["view_label"], path.parent
        directory = output / label
        directory.mkdir(exist_ok=True)
        image = cv2.imread(str(folder / "color.png"))
        depth = np.load(folder / "depth_m.npy")
        with np.load(folder / "depth_raw_frames.npz") as arrays:
            raw = arrays["depth_m"]
            sequences = arrays["sequence"]
        if len(np.unique(sequences)) != len(sequences):
            raise ValueError("Raw frames must be distinct observations")
        support = apply_raw_depth_support(
            depth, raw, args.orbbec_depth_median_min_valid, args.orbbec_depth_median_max_mad,
        )
        capture_metadata = json.loads((folder / "capture_meta.json").read_text())
        self_mask, _ = calibrated_self_mask(image, args.camera_self_mask_calibration, capture_metadata)
        np.testing.assert_array_equal(self_mask, cv2.imread(str(folder / "camera_self_mask.png"), 0))
        intr = type("Intrinsics", (), metadata["intrinsics"])()
        selected = cv2.imread(str(folder / "selected_target_mask.png"), 0) > 0
        before = runtime.depth_to_xyz_image(depth, intr)
        fixed = runtime.depth_to_xyz_image(support["depth_m"], intr)
        before[self_mask > 0] = np.nan
        fixed[self_mask > 0] = np.nan
        baseline = runtime.obstacle_cloud_from_non_target_mask(before, image, selected, args)
        archived = o3d.io.read_point_cloud(str(folder / "obstacle_camera.ply"))
        maximum_error = max(
            float(cKDTree(archived.points).query(baseline.points)[0].max()),
            float(cKDTree(baseline.points).query(archived.points)[0].max()),
        )
        if len(baseline.points) != len(archived.points) or maximum_error > 1e-6:
            raise ValueError("Baseline cloud does not reproduce the archive")
        corrected = runtime.obstacle_cloud_from_non_target_mask(fixed, image, selected, args)
        transform = np.asarray(metadata["T_base_camera"])
        baseline.transform(transform)
        corrected.transform(transform)
        measured_target = o3d.io.read_point_cloud(str(folder / "target_base.ply"))
        # Independently verify that the target can still supply the grasp generator.
        target_xyz = runtime.refine_xyz_inside_visible_mask(
            fixed, selected, intr, args, out_dir=str(directory), label=label,
        )
        corrected_target = runtime.cloud_from_mask(target_xyz, image, selected, args)
        corrected_target.transform(transform)
        color = image.astype(float)
        red = (color[:, :, 2] > 100) & (color[:, :, 2] > 1.5 * color[:, :, 1]) & (color[:, :, 2] > 1.5 * color[:, :, 0])
        cable = (image.max(axis=2) < 65) & (self_mask == 0)
        cable[:, image.shape[1] // 4:] = False  # Visual proxy for left-side cables in these captures.
        object_stats = {}
        for name, region in (("banana_mask", selected), ("red_cube_color_proxy", red), ("left_cable_dark_proxy", cable)):
            valid_before = region & np.isfinite(before).all(axis=2)
            valid_after = region & np.isfinite(fixed).all(axis=2)
            object_stats[name] = {"before_pixels": int(np.count_nonzero(valid_before)),
                                  "after_pixels": int(np.count_nonzero(valid_after)),
                                  "retained_fraction": float(np.count_nonzero(valid_after) / max(np.count_nonzero(valid_before), 1))}
        overlay = image.copy()
        rejected = (support["rejected_mask"] > 0) & (self_mask == 0)
        overlay[rejected] = (0.35 * overlay[rejected] + 0.65 * np.array([255, 0, 255])).astype(np.uint8)
        cv2.imwrite(str(directory / "rejected_overlay.png"), overlay)
        cv2.imwrite(str(directory / "depth_support_rejected.png"), support["rejected_mask"] * 255)
        np.save(directory / "depth_m.npy", support["depth_m"])
        o3d.io.write_point_cloud(str(directory / "obstacle_fixed_base.ply"), corrected)
        o3d.io.write_point_cloud(str(directory / "target_fixed_base.ply"), corrected_target)
        view_reports[label] = {"support": support["report"], "objects": object_stats,
                               "obstacle_points_before": len(baseline.points), "obstacle_points_after": len(corrected.points),
                               "target_points_before": len(measured_target.points), "target_points_after": len(corrected_target.points),
                               "baseline_reconstruction_max_error_m": maximum_error}
        views[label] = {"before": baseline, "after": corrected, "target": measured_target, "meta": metadata}
    order = sorted(views, key=lambda name: (name != "initial", name != "center", name))
    target_union = np.vstack([np.asarray(views[label]["target"].points) for label in order])
    center = np.median(np.asarray(views["initial"]["target"].points), axis=0)
    target = o3d.io.read_point_cloud(str(capture / "graspnet_target_full.ply"))
    scene_cloud = o3d.io.read_point_cloud(str(capture / "active_fused_scene_camera0.ply"))
    scenes = {}
    for kind in ("before", "after"):
        environment = runtime.fuse_dynamic_environment_clouds(
            order, [views[label][kind] for label in order], args, target_base=target_union,
            target_center_base=center, map_id=f"raw_support_{kind}", generation=1,
        )
        scenes[kind] = runtime.prepare_motion_safety_scene(
            target, scene_cloud, np.asarray(views["initial"]["meta"]["T_base_camera"]),
            args, map_id=f"raw_support_{kind}", environment_map=environment,
        )
    if len(scenes["before"].environment_base) != source["environment_points"]:
        raise ValueError("Original collision map differs from the saved endpoint probe")
    witness = np.asarray(source["cases"][1]["records"][0]["nearest_point_base"])
    report = {
        "run": str(run), "hardware_connected": False, "arm_model": args.arm_collision_model,
        "count_mad_collision_thresholds_changed": False, "missing_depth_is_free_space": False,
        "views": view_reports, "environment_points": {kind: len(scene.environment_base) for kind, scene in scenes.items()},
        "witness_nearest_environment_distance_m": {
            kind: float(cKDTree(scene.environment_base).query(witness)[0]) for kind, scene in scenes.items()
        }, "cases": [],
        "limitations": "No point is replaced by an assumed table plane. Same saved joints and target collision cloud isolate the environment-map change; corrected per-view target clouds are checked separately. Object color masks are visual proxies. Endpoint results do not establish a complete executable grasp.",
    }
    for case in source["cases"]:
        context_phase = "free_open" if case["stage"] == "pre_tcp_preview" else "target_approach"
        for rank, record in enumerate(case["records"]):
            row = {"candidate": case["idx"], "stage": case["stage"], "rank": rank + 1}
            for kind, scene in scenes.items():
                row[kind] = runtime._motion_configuration_metrics(
                    record["joints"], runtime.MotionCollisionContext(scene, context_phase), args,
                )
            report["cases"].append(row)
    o3d.io.write_point_cloud(str(output / "environment_fixed_base.ply"), o3d.geometry.PointCloud(o3d.utility.Vector3dVector(scenes["after"].environment_base)))
    o3d.io.write_point_cloud(str(output / "environment_original_base.ply"), o3d.geometry.PointCloud(o3d.utility.Vector3dVector(scenes["before"].environment_base)))
    def plain(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        raise TypeError(type(value).__name__)
    (output / "replay.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=plain) + "\n")
    print(json.dumps({"environment_points": report["environment_points"], "witness_distance": report["witness_nearest_environment_distance_m"], "views": view_reports}, indent=2))
    for case in report["cases"]:
        if case["rank"] == 1:
            print(case["candidate"], case["stage"], [(kind, round(case[kind]["nearest_clearance"] * 1000, 2), case[kind]["primitive"]) for kind in ("before", "after")])


if __name__ == "__main__":
    main()
