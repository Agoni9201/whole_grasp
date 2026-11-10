#!/usr/bin/env python3
"""Offline, read-only diagnostic for active-view target-cloud layering.

Reads a previous run's ``active_fusion_registration.json`` and (optionally) the
per-view ``active_*_target_base{,_registered}.ply`` files, then reports:

1. per view — which registration stages actually applied (table / translation /
   rigid ICP) and their magnitudes, plus the post-registration overlap quality;
2. the dominant direction of the residual misalignment (XY translation, yaw
   rotation, or Z / height), i.e. what is making one surface become two layers.

This script never imports the robot / camera / GraspNet stack and never writes
to the run directory. It only reads saved artefacts, so it is safe to run while
the main program is being used for other work.

Run with the real runtime env (open3d only needed for the geometric part)::

    conda run -n grasp_yolo26 python tools/diagnose_fusion_layering.py [run_dir]

If ``run_dir`` is omitted, the newest log run that contains a registration JSON
is selected automatically.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

# Post-registration overlap thresholds used to label how severe the residual
# misalignment (the "two layers") is. These are diagnostic labels, not the
# exact runtime acceptance gates; they are chosen around the code's own
# fusion tolerances (surfel merge radius ~8 mm, fusion median gate ~3 mm).
MEDIAN_WARN_MM = 3.0
MEDIAN_STRONG_MM = 6.0
P95_WARN_MM = 8.0
P95_STRONG_MM = 12.0
GHOST_DIST_MM = 4.0  # points farther than this from the reference = "second layer"


def _find_newest(root: Path, name: str) -> Path | None:
    matches = sorted(
        (p for p in root.rglob(name) if p.is_file()),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return matches[0] if matches else None


def _rotation_to_rpy_xyz(R: np.ndarray) -> tuple[float, float, float]:
    """Intrinsic XYZ Euler angles (deg), matching scipy Rotation 'xyz'."""
    R = np.asarray(R, dtype=float).reshape(3, 3)
    sy = math.hypot(R[0, 0], R[1, 0])
    if sy > 1e-6:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0.0
    return math.degrees(x), math.degrees(y), math.degrees(z)


def _decompose_transform(T) -> dict:
    T = np.asarray(T, dtype=float).reshape(4, 4)
    t = T[:3, 3]
    R = T[:3, :3]
    roll, pitch, yaw = _rotation_to_rpy_xyz(R)
    return {
        "tx": float(t[0]),
        "ty": float(t[1]),
        "tz": float(t[2]),
        "trans_norm": float(np.linalg.norm(t)),
        "roll_deg": roll,
        "pitch_deg": pitch,
        "yaw_deg": yaw,
        "rot_norm_deg": float(
            np.rad2deg(np.arccos(np.clip((np.trace(R) - 1.0) * 0.5, -1.0, 1.0)))
        ),
    }


def _load_xyz(path: Path) -> np.ndarray | None:
    """Return (N,3) float64 points, using open3d if present else a tiny reader."""
    try:
        import open3d as o3d

        pcd = o3d.io.read_point_cloud(str(path))
        pts = np.asarray(pcd.points, dtype=np.float64).reshape(-1, 3)
        return pts if pts.shape[0] else None
    except Exception as exc:  # noqa: BLE001
        print(f"    [skip geometry] open3d unavailable for {path.name}: {type(exc).__name__}: {exc}")
        return None


def _analyze_registration_json(path: Path) -> None:
    data = json.loads(path.read_text(encoding="utf-8"))
    ref = data.get("reference_view", "?")
    print(f"registration JSON: {path}")
    print(f"frame={data.get('frame', '?')}  reference_view={ref}  created={data.get('created_at', '?')}")
    views = data.get("views", [])
    if not views:
        print("  (no views recorded)")
        return

    print()
    print("== per-view registration stages ==")
    header = (
        f"{'view':8s} {'keep':5s} | "
        f"{'table':8s} {'dZ(mm)':>7s} {'tilt':>6s} | "
        f"{'trans':8s} {'shiftXY':>8s} {'shiftZ':>7s} | "
        f"{'icp':12s} {'dR':>6s} {'dT(mm)':>7s} | "
        f"{'ovl_med':>7s} {'ovl_p95':>7s} {'ovl1cm':>6s}"
    )
    print(header)
    print("-" * len(header))

    layering_hits = 0
    for v in views:
        name = str(v.get("name", "?"))
        keep = "Y" if v.get("registration_keep", False) else "N"
        table = v.get("table", {})
        trans = v.get("translation", {})
        icp = v.get("icp", {})
        overlap = v.get("overlap", {})
        t_applied = bool(table.get("applied"))
        tr_applied = bool(trans.get("applied"))
        i_applied = bool(icp.get("applied"))
        tilt = float(table.get("tilt_deg", float("nan")))
        dz = float(table.get("height_shift", float("nan")))
        shift = np.asarray(trans.get("shift", [0.0, 0.0, 0.0]), dtype=float).reshape(3)
        shift_xy = float(np.linalg.norm(shift[:2]))
        shift_z = float(shift[2])
        dR = float(icp.get("rot_deg", float("nan")))
        dT = float(icp.get("trans", float("nan")))
        med = float(overlap.get("median", float("nan")))
        p95 = float(overlap.get("p95", float("nan")))
        in1 = float(overlap.get("inlier@1cm", float("nan")))

        t_str = "on " if t_applied else "off"
        tr_str = "on " if tr_applied else "off"
        i_str = f"{icp.get('reason', 'off')[:12]:12s}"

        print(
            f"{name:8s} {keep:5s} | "
            f"{t_str:8s} {dz * 1000:7.2f} {tilt:6.2f} | "
            f"{tr_str:8s} {shift_xy * 1000:8.2f} {shift_z * 1000:7.2f} | "
            f"{i_str:12s} {dR:6.2f} {dT * 1000:7.2f} | "
            f"{med * 1000:7.2f} {p95 * 1000:7.2f} {in1:6.2f}"
        )

        if np.isfinite(med) and med * 1000 > MEDIAN_WARN_MM:
            layering_hits += 1
        elif np.isfinite(p95) and p95 * 1000 > P95_WARN_MM:
            layering_hits += 1

        if not t_applied and not tr_applied and not i_applied:
            print(f"    {name}: no correction applied -> raw robot-pose prior only")
        if i_applied is False and icp.get("reason") not in ("disabled", None):
            print(f"    {name}: ICP rejected: {icp.get('reason')}")

    print()
    print("== applied correction per view (from target_transform) ==")
    for v in views:
        name = str(v.get("name", "?"))
        T = v.get("target_transform")
        if T is None:
            continue
        d = _decompose_transform(T)
        print(
            f"  {name:8s} t=({d['tx'] * 1000:+.2f},{d['ty'] * 1000:+.2f},{d['tz'] * 1000:+.2f})mm "
            f"|t|={d['trans_norm'] * 1000:.2f}mm  "
            f"rpy=({d['roll_deg']:+.2f},{d['pitch_deg']:+.2f},{d['yaw_deg']:+.2f})deg  "
            f"|r|={d['rot_norm_deg']:.2f}deg"
        )

    print()
    if layering_hits:
        print(
            f"[结论] 有 {layering_hits} 个视角的后配准重叠中位数/p95 超过诊断阈值 "
            f"(med>{MEDIAN_WARN_MM}mm / p95>{P95_WARN_MM}mm) -> 存在可见分层。"
        )
        print(
            "  方向判定见下方几何分析；若仅开了 table+translation 而 icp=disabled，"
            "则旋转（尤其 yaw）误差未被修正，最可能是分层主因。"
        )
    else:
        print("[结论] 各视角后配准重叠在阈值内，未检测到明显分层（至少按此阈值）。")


def _analyze_geometry(run_dir: Path, json_path: Path) -> None:
    """Cross-view NN decomposition inside the overlap region only.

    A side view legitimately sees surfaces the center view does not. Counting
    those novel points as "ghost" would inflate the layering signal, so we only
    consider side points that land within OVERLAP_GATE of the reference cloud.
    """
    data = json.loads(json_path.read_text(encoding="utf-8"))
    ref = data.get("reference_view", "center")
    side_names = [
        v.get("name") for v in data.get("views", []) if v.get("name") != ref
    ]

    def _cloud(view: str, registered: bool) -> np.ndarray | None:
        suffix = "target_base_registered.ply" if registered else "target_base.ply"
        p = _find_newest(run_dir, f"active_{view}_{suffix}")
        if p is None:
            return None
        return _load_xyz(p)

    ref_pts = _cloud(ref, registered=True)
    if ref_pts is None:
        ref_pts = _cloud(ref, registered=False)
    if ref_pts is None:
        print("\n[几何] 未找到参考视角点云，跳过方向分解（JSON 分析已完成）。")
        return

    OVERLAP_GATE = 0.015  # m: within this of the reference = overlap region
    GHOST_LO = 0.004      # m: above which an overlap point reads as "second layer"
    WELL = 0.002          # m: below which an overlap point is "coincident"

    print("\n== 几何方向分解（仅重叠区，避免把新看到的侧面误判成重影） ==")
    print(f"  参考视角 {ref}: n={ref_pts.shape[0]}")
    for name in side_names:
        side = _cloud(name, registered=True)
        if side is None:
            side = _cloud(name, registered=False)
        if side is None:
            print(f"  {name}: 找不到点云，跳过")
            continue
        ref_d = ref_pts[:: max(1, ref_pts.shape[0] // 15000)]
        side_d = side[:: max(1, side.shape[0] // 15000)]
        from scipy.spatial import cKDTree

        tree = cKDTree(ref_d)
        dist, idx = tree.query(side_d, k=1)
        dist = np.asarray(dist)
        overlap = dist < OVERLAP_GATE
        if not np.any(overlap):
            print(f"  {name}: 与参考视角几乎无重叠（视角差异大），跳过")
            continue
        d = dist[overlap]
        med = float(np.median(d))
        p90 = float(np.quantile(d, 0.90))
        well_frac = float(np.mean(d < WELL))
        ghost_frac = float(np.mean((d >= GHOST_LO) & (d < OVERLAP_GATE)))
        residual = ref_d[idx[overlap]] - side_d[overlap]
        med_res = np.median(residual, axis=0)
        axy = float(np.linalg.norm(med_res[:2]))
        az = float(abs(med_res[2]))
        print(
            f"  {name:8s} 重叠点={d.shape[0]:5d}  "
            f"NN med={med * 1000:5.2f}mm p90={p90 * 1000:5.2f}mm "
            f"贴合(<2mm)={well_frac * 100:4.1f}% "
            f"第二层(4~15mm)={ghost_frac * 100:4.1f}%  "
            f"残差 |XY|={axy * 1000:.2f}mm |Z|={az * 1000:.2f}mm"
        )
        dom = "XY（平移/yaw）" if axy >= az else "Z（roll/pitch/深度）"
        if ghost_frac > 0.20:
            print(f"      -> 检测到明显第二层（重叠区 {ghost_frac * 100:.0f}% 点在 4~15mm），方向以 {dom} 为主")
        elif med > GHOST_LO:
            print(f"      -> 重叠区整体偏置 {med * 1000:.1f}mm（方向 {dom}），偏置但未形成清晰两层")
        else:
            print("      -> 重叠区贴合良好，未检测到分层")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("run_dir", nargs="?", help="日志运行目录（含 active_fusion_registration.json）")
    ap.add_argument("--no-geometry", action="store_true", help="只做 JSON 分析，不读点云")
    args = ap.parse_args(argv)

    if args.run_dir:
        root = Path(args.run_dir)
    else:
        logs = Path(__file__).resolve().parents[1] / "logs"
        json_hit = _find_newest(logs, "active_fusion_registration.json")
        if json_hit is None:
            print("未找到任何 active_fusion_registration.json，请指定 run_dir。", file=sys.stderr)
            return 1
        root = json_hit.parent
        print(f"[auto] 使用最新包含配准 JSON 的运行目录: {root}")

    json_path = _find_newest(root, "active_fusion_registration.json")
    if json_path is None:
        print(f"目录 {root} 下未找到 active_fusion_registration.json", file=sys.stderr)
        return 1

    _analyze_registration_json(json_path)
    if not args.no_geometry:
        try:
            _analyze_geometry(root, json_path)
        except Exception as exc:  # noqa: BLE001
            print(f"[几何] 方向分解失败（不影响 JSON 结论）: {type(exc).__name__}: {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
