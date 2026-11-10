#!/usr/bin/env python3
"""Run raw GraspNet proposals on a GraspGenX object-PC JSON, without robot I/O.

This compares proposal generation on identical saved object points. The output
poses use GraspNet's native grasp frame, not the Nero TCP frame. Collision,
contact, IK, and whole-workflow checks are intentionally not inferred here.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for name in ("pointnet2", "models", "dataset", "utils", "graspnetAPI"):
    path = str(ROOT / "graspnet-baseline" / name)
    if path not in sys.path:
        sys.path.insert(0, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--num-points", type=int, default=3000)
    parser.add_argument("--topk", type=int, default=40)
    args = parser.parse_args()
    if args.num_points <= 0 or args.topk <= 0:
        parser.error("--num-points and --topk must be positive")

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this GraspNet checkpoint")
    # Compatibility with the installed graspnetAPI on NumPy 2.x.
    for name, value in (("float", float), ("int", int), ("bool", bool)):
        if name not in np.__dict__:
            setattr(np, name, value)
    from graspnet import GraspNet, pred_decode
    from graspnetAPI import GraspGroup

    obj = json.loads(args.input_json.read_text(encoding="utf-8"))
    points = np.asarray(obj["pc"], dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3 or not len(points):
        raise ValueError("Input JSON must contain a nonempty Nx3 'pc' array")
    if not np.isfinite(points).all():
        raise ValueError("Input point cloud contains nonfinite coordinates")

    checkpoint = ROOT / "logs/log_rs/checkpoint-rs.tar"
    torch.set_num_threads(4)
    model = GraspNet(
        input_feature_dim=0,
        num_view=300,
        num_angle=12,
        num_depth=4,
        cylinder_radius=0.05,
        hmin=-0.02,
        hmax_list=[0.01, 0.02, 0.03, 0.04],
        is_training=False,
    )
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state["model_state_dict"], strict=True)
    del state
    model = model.cuda().eval()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for seed in args.seeds:
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if len(points) >= args.num_points:
            indices = np.random.choice(len(points), args.num_points, replace=False)
        else:
            indices = np.r_[np.arange(len(points)), np.random.choice(
                len(points), args.num_points - len(points), replace=True
            )]
        tensor = torch.from_numpy(points[indices][None].copy()).cuda()
        started = time.monotonic()
        with torch.inference_mode():
            decoded = pred_decode(model({"point_clouds": tensor}))[0].cpu().numpy()
        torch.cuda.synchronize()
        all_grasps = GraspGroup(decoded)
        ranking = sorted(range(len(all_grasps)), key=lambda i: -float(all_grasps[i].score))
        rows = []
        for rank, index in enumerate(ranking[:args.topk], 1):
            grasp = all_grasps[index]
            pose = np.eye(4, dtype=float)
            pose[:3, :3] = grasp.rotation_matrix
            pose[:3, 3] = grasp.translation
            rows.append({
                "rank": rank,
                "decoded_index": index,
                "score": float(grasp.score),
                "pose_input_frame": pose.tolist(),
                "width_m": float(grasp.width),
                "height_m": float(grasp.height),
                "depth_m": float(grasp.depth),
            })
        result = {
            "method": "GraspNet raw decoded candidates, ranked by network score",
            "input_json": str(args.input_json.resolve()),
            "input_frame": "same coordinates as input JSON pc",
            "pose_convention": "GraspNet native grasp frame, not Nero TCP",
            "checkpoint": str(checkpoint),
            "seed": seed,
            "input_points": len(points),
            "network_points": args.num_points,
            "decoded_candidates": len(decoded),
            "seconds": time.monotonic() - started,
            "candidates": rows,
        }
        output = args.output_dir / f"{args.input_json.stem}__graspnet__seed{seed}.json"
        output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(f"seed={seed}: saved {len(rows)}/{len(decoded)} candidates to {output}", flush=True)


if __name__ == "__main__":
    main()
