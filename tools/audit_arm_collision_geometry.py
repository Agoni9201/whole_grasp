#!/usr/bin/env python3
"""Compare CAD-fitted/legacy robot geometry on saved endpoint joints; no hardware."""

import argparse
from copy import copy
import json
from pathlib import Path
import socket
import sys
import sysconfig
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [sysconfig.get_path("purelib"), str(ROOT)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--endpoint-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    options = parser.parse_args()
    # This script must never connect to the physical arm, even after future edits.
    original_socket = socket.socket
    class OfflineSocket(original_socket):
        def __init__(self, family=socket.AF_INET, *args, **kwargs):
            if family == getattr(socket, "AF_CAN", None):
                raise RuntimeError("CAN is forbidden in this offline geometry audit")
            super().__init__(family, *args, **kwargs)
    socket.socket = OfflineSocket
    import numpy as np
    from real_nero_grasp import runtime as runtime
    from real_nero_grasp import collision
    from real_nero_grasp.arm_collision_geometry import load_arm_geometry

    run = options.run_dir.resolve()
    source = json.loads(options.endpoint_json.read_text())
    result = json.loads((run / "experiment_result.json").read_text())
    sys.argv = result["command"]
    args = runtime.parse_args()
    args.execute = False
    args._planning_worker = True
    # Find the capture containing the exact saved grasp candidates.
    capture = next(path.parent for path in run.glob("*/grasp_candidates.json"))
    views = {}
    for path in (capture / "multiview_raw").glob("*/metadata.json"):
        metadata = json.loads(path.read_text())
        views[metadata["view_label"]] = (path.parent, metadata)
    labels = sorted(views, key=lambda label: (label != "initial", label != "center", label))
    o3d = runtime.get_o3d()
    obstacles = [o3d.io.read_point_cloud(str(views[label][0] / "obstacle_base.ply")) for label in labels]
    targets = [np.asarray(o3d.io.read_point_cloud(str(views[label][0] / "target_base.ply")).points) for label in labels]
    environment = runtime.fuse_dynamic_environment_clouds(
        labels, obstacles, args, target_base=np.vstack(targets),
        target_center_base=np.median(targets[labels.index("initial")], axis=0),
        map_id="arm_geometry_audit", generation=1,
    )
    scene = runtime.prepare_motion_safety_scene(
        o3d.io.read_point_cloud(str(capture / "graspnet_target_full.ply")),
        o3d.io.read_point_cloud(str(capture / "active_fused_scene_camera0.ply")),
        np.asarray(views["initial"][1]["T_base_camera"]), args,
        map_id="arm_geometry_audit", environment_map=environment,
    )
    geometry = load_arm_geometry(args, collision._load_urdf_joint_chain(args))
    compared = []
    for case in source["cases"]:
        phase = "free_open" if case["stage"] == "pre_tcp_preview" else "target_approach"
        context = runtime.MotionCollisionContext(scene, phase)
        for rank, record in enumerate(case["records"]):
            entry = {"candidate": case["idx"], "stage": case["stage"], "rank": rank + 1}
            for mode in ("legacy_capsules", "urdf_capsules"):
                mode_args = copy(args)
                mode_args.arm_collision_model = mode
                metrics = runtime._motion_configuration_metrics(record["joints"], context, mode_args)
                entry[mode] = metrics
            compared.append(entry)
    # Representative warm single-configuration cost; this is not a CAN timing test.
    q = np.asarray(views["center"][1]["capture_joints"])
    context = runtime.MotionCollisionContext(scene, "target_approach")
    timing = {}
    for mode in ("legacy_capsules", "urdf_capsules"):
        mode_args = copy(args)
        mode_args.arm_collision_model = mode
        runtime._motion_configuration_metrics(q, context, mode_args)
        samples = []
        for _ in range(20):
            start = time.monotonic()
            runtime._motion_configuration_metrics(q, context, mode_args)
            samples.append((time.monotonic() - start) * 1000)
        timing[mode] = {"median_ms": float(np.median(samples)), "max_ms": max(samples)}
    report = {
        "run": str(run), "hardware_connected": False, "endpoint_only": True,
        "environment_points": len(scene.environment_base), "target_points": len(scene.target_base),
        "mesh_padding_m": geometry["padding_m"], "urdf_sha256": geometry["urdf_sha256"],
        "links": geometry["report"], "configuration_timing": timing, "cases": compared,
        "limitations": "CAD coverage is verified; camera bracket and physical installation are not measured here. No point-cloud occupancy is removed. Endpoint comparisons do not validate full paths.",
    }
    options.output_dir.mkdir(parents=True, exist_ok=True)
    def plain(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        raise TypeError(type(value).__name__)
    output = options.output_dir / "arm_collision_audit.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=plain) + "\n")
    print(f"Saved {output}; {len(compared)} endpoint comparisons; no hardware connected")
    print(json.dumps(timing, indent=2))
    for case in compared:
        if case["rank"] == 1:
            print(case["candidate"], case["stage"], [(mode, round(case[mode]["nearest_clearance"] * 1000, 2), case[mode]["primitive"]) for mode in ("legacy_capsules", "urdf_capsules")])


if __name__ == "__main__":
    main()
