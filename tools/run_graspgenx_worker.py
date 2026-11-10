#!/usr/bin/env python3
"""Headless GraspGenX worker for one measured or MGPC proposal cloud."""

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
GENX = ROOT / "third_party/GraspGenX"
sys.path.insert(0, str(GENX))
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", type=Path, required=True)
    args = parser.parse_args()
    request = json.loads(args.request.read_text())
    gripper_name = request.get("gripper_name", "nero_2f_85")
    if gripper_name not in ("nero_2f_85", "nero_2f_100"):
        raise ValueError(f"Unsupported Nero gripper asset: {gripper_name}")
    import torch
    from graspgenx.grasp_server import GraspGenXSampler
    from graspgenx.samplers import run_planner_on_object
    from graspgenx.utils.checkpoint_io import load_model_cfg
    from graspgenx.x_grippers import resolve_gripper_info
    from tools.check_graspgenx_table import assess_table

    seed = int(request["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if not torch.cuda.is_available():
        raise RuntimeError("GraspGenX requires CUDA")
    points = np.load(request["input_npy"], allow_pickle=False)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 100 or not np.isfinite(points).all():
        raise ValueError("invalid nonempty Nx3 proposal cloud")
    checkpoint = GENX / "ext/graspgenx_checkpoints/release"
    cfg = load_model_cfg(str(checkpoint / "gen"), str(checkpoint / "dis"))
    sampler = GraspGenXSampler(cfg, gripper_name, assets_dir=str(GENX / "assets"))
    center = points.mean(axis=0)
    poses, scores, tags, _ = run_planner_on_object(
        (points - center).astype(np.float32), sampler,
        planner="diffusion", grasp_threshold=-1.0,
        num_grasps=int(request["num_grasps"]),
        topk_num_grasps=int(request["topk_per_pass"]),
    )
    poses = np.asarray(poses, dtype=float).reshape(-1, 4, 4)
    scores = np.asarray(scores, dtype=float).reshape(-1)
    if len(poses) != len(scores) or not len(scores) or not np.isfinite(poses).all() or not np.isfinite(scores).all():
        raise RuntimeError("GraspGenX produced no valid candidates")
    poses[:, :3, 3] += center
    gripper = resolve_gripper_info(gripper_name, str(GENX / "assets"))
    vertices = np.asarray(gripper.collision_mesh.vertices, dtype=float)
    if len(vertices) < 100:
        raise RuntimeError("Nero collision mesh is missing or empty")
    table = assess_table(
        poses, scores, vertices, np.asarray(request["T_base_camera"], dtype=float),
        table_z=float(request["table_z"]), margin=float(request["table_margin"]),
        pregrasp_offset=float(request["pregrasp_offset"]),
    )
    ranking = sorted(range(len(scores)), key=lambda index: -scores[index])
    payload = {
        "source": f"GraspGenX diffusion, {gripper_name}",
        "seed": seed, "input_points": len(points),
        "candidate_count": len(scores),
        "table_safe_count": sum(row["table_safe"] for row in table),
        "candidates": [
            {"index": int(index), "pose_camera": poses[index].tolist(),
             "score": float(scores[index]), "branch": tags[index],
             **table[index]}
            for index in ranking
        ],
    }
    output = Path(request["output_json"])
    output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    print(f"[graspgenx] inferred {len(scores)}; table-safe {payload['table_safe_count']}", flush=True)


if __name__ == "__main__":
    main()
