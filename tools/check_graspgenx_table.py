#!/usr/bin/env python3
"""Screen saved GraspGenX grasps against the measured Nero gripper and table.

This checks the complete open-gripper mesh at the grasp pose and along a
straight 10 cm pregrasp approach. It checks only the calibrated table plane;
the existing whole-workflow scene collision and IK checks remain necessary.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
GENX_ROOT = ROOT / "third_party" / "GraspGenX"
sys.path.insert(0, str(GENX_ROOT))

from graspgenx.x_grippers import resolve_gripper_info


def assess_table(
    poses_camera: np.ndarray,
    scores: np.ndarray,
    vertices_gripper: np.ndarray,
    T_base_camera: np.ndarray,
    *,
    table_z: float,
    margin: float,
    pregrasp_offset: float,
) -> list[dict]:
    """Minimum mesh Z is exact for a planar table and fixed-orientation line approach."""
    if poses_camera.shape != (len(scores), 4, 4):
        raise ValueError("Pose and score arrays have incompatible shapes")
    if vertices_gripper.ndim != 2 or vertices_gripper.shape[1] != 3 or not len(vertices_gripper):
        raise ValueError("A nonempty Nero gripper collision mesh is required")
    records = []
    for index, (pose, score) in enumerate(zip(poses_camera, scores)):
        world = T_base_camera @ pose
        # The extrema of a linear Z function over each triangle lie at a vertex.
        mesh_min_z_local = float(np.min(vertices_gripper @ world[2, :3]))
        grasp_min_z = mesh_min_z_local + float(world[2, 3])
        pregrasp_min_z = grasp_min_z - pregrasp_offset * float(world[2, 2])
        minimum_z = min(grasp_min_z, pregrasp_min_z)
        clearance = minimum_z - table_z
        records.append({
            "index": index,
            "score": float(score),
            "grasp_mesh_min_z_m": grasp_min_z,
            "pregrasp_mesh_min_z_m": pregrasp_min_z,
            "minimum_table_clearance_m": clearance,
            "table_safe": bool(clearance >= margin),
        })
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-json", type=Path, required=True)
    parser.add_argument("--preview-json", type=Path, required=True,
                        help="Saved final_grasp_preview.json containing T_base_camera0")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--table-z", type=float, default=0.0)
    parser.add_argument("--margin", type=float, default=0.005)
    parser.add_argument("--pregrasp-offset", type=float, default=0.10)
    args = parser.parse_args()
    if args.margin < 0 or args.pregrasp_offset < 0:
        parser.error("--margin and --pregrasp-offset must be nonnegative")

    data = json.loads(args.candidate_json.read_text(encoding="utf-8"))
    preview = json.loads(args.preview_json.read_text(encoding="utf-8"))
    info = resolve_gripper_info(data["gripper_name"], str(GENX_ROOT / "assets"))
    poses = np.asarray(data["poses_input_frame"], dtype=float)
    scores = np.asarray(data["scores"], dtype=float)
    transform = np.asarray(preview["T_base_camera0"], dtype=float)
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise ValueError("Invalid T_base_camera0")
    rows = assess_table(
        poses, scores, np.asarray(info.collision_mesh.vertices), transform,
        table_z=args.table_z, margin=args.margin,
        pregrasp_offset=args.pregrasp_offset,
    )
    safe = sorted((row for row in rows if row["table_safe"]), key=lambda row: -row["score"])
    report = {
        "candidate_json": str(args.candidate_json.resolve()),
        "preview_json": str(args.preview_json.resolve()),
        "gripper_name": data["gripper_name"],
        "source_name": data["source_name"],
        "seed": data["seed"],
        "table_z_m": args.table_z,
        "required_margin_m": args.margin,
        "pregrasp_retreat_m": args.pregrasp_offset,
        "checked_poses": len(rows),
        "table_safe_count": len(safe),
        "best_safe_index": safe[0]["index"] if safe else None,
        "records": rows,
        "scope": "open gripper versus calibrated table plane only; no other scene or robot-arm collision",
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Table-safe poses: {len(safe)}/{len(rows)}; best safe index: {report['best_safe_index']}")
    if safe:
        print(f"Best safe score: {safe[0]['score']:.3f}; clearance: {safe[0]['minimum_table_clearance_m']*1000:.1f} mm")
    print(f"Saved {args.output_json}")


if __name__ == "__main__":
    main()
