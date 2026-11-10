#!/usr/bin/env python3
"""Offline, read-only diagnostic for grasp candidate filtering / ranking.

Reads a previous run's ``grasp_candidates.json`` (and ``selected_grasp_pose.json``
if present), then reports:

1. the survival funnel across the hard filters that are applied *after* the
   candidate rows are built (physical aperture, table guard, reachability,
   whole-workflow preflight);
2. which candidates are actually executable, and which got killed by which rule;
3. how the network's raw score ranks against the hand-tuned ``priority``, so you
   can see whether a physically-sensible grasp is being overridden by the
   "center-ness / vertical-down" heuristics.

It never imports the robot / camera / GraspNet stack and never writes to the
run directory. It only reads saved artefacts, so it is safe to run while the
main program is being used for other work.

Run with the real runtime env::

    conda run -n grasp_yolo26 python tools/diagnose_grasp_candidates.py [run_dir]

If ``run_dir`` is omitted, the newest log run that contains a candidate JSON is
selected automatically.

Note on scope: the JSON only contains candidates that already survived the
pre-row filters inside ``generate_grasps`` (collision, target-occupancy,
vertical-angle, IK pre-check). Candidates deleted there are not present, so
their loss is visible only as "candidate pool is smaller than expected".
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# The two hand-tuned rankings used by the pipeline (for re-derivation / sanity).
PRIORITY_W = {"vertical_score": 0.28, "center_score": 0.30, "obstacle_score": 0.27, "ik_score": 0.10, "raw_score": 0.05}


def _find_newest(root: Path, name: str) -> Path | None:
    matches = sorted(
        (p for p in root.rglob(name) if p.is_file()),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return matches[0] if matches else None


def _f(value, default=float("nan")):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _bool(value):
    return bool(value)


def _recompute_priority(row) -> float:
    return sum(PRIORITY_W[k] * float(np.clip(_f(row.get(k, 0.0)), 0.0, 1.0)) for k in PRIORITY_W)


def _candidate_line(idx: int, row: dict) -> str:
    raw = _f(row.get("raw_score"))
    prio = _f(row.get("priority"))
    cross = _f(row.get("target_cross_axis_score"))
    elong = _f(row.get("target_shape_elongation"))
    width = _f(row.get("graspnet_width")) * 1000
    ap_ok = _bool(row.get("physical_aperture_ok"))
    tg_ok = row.get("table_guard_ok")
    reach_ok = row.get("reachability_ok")
    reach_def = row.get("reachability_deferred")
    wf_ok = row.get("workflow_preflight_ok")

    # executable = every hard gate that has a value is satisfied
    gates = []
    if "physical_aperture_ok" in row:
        gates.append(ap_ok)
    if "table_guard_ok" in row:
        gates.append(tg_ok is True)
    if "reachability_ok" in row and not reach_def:
        gates.append(reach_ok is True)
    if "workflow_preflight_ok" in row:
        gates.append(wf_ok is True)
    executable = bool(gates) and all(gates)

    mark = "OK " if executable else "-- "
    ap_s = "Y" if ap_ok else ("-" if "physical_aperture_ok" not in row else "N")
    tg_s = "Y" if tg_ok is True else ("-" if "table_guard_ok" not in row else "N")
    r_s = "Y" if reach_ok else ("D" if reach_def else ("-" if "reachability_ok" not in row else "N"))
    w_s = "Y" if wf_ok else ("-" if "workflow_preflight_ok" not in row else "N")
    cross_s = f"{cross:5.2f}" if np.isfinite(cross) else "    -"
    elong_s = f"{elong:4.1f}" if np.isfinite(elong) else " 0.0"
    width_s = f"{width:5.1f}" if np.isfinite(width) else "  nan"
    return (
        f"  {mark} idx={idx:2d} raw={raw:5.3f} prio={prio:5.3f} "
        f"cross={cross_s} elong={elong_s} width={width_s}mm "
        f"ap={ap_s} tbl={tg_s} reach={r_s} wf={w_s}"
    )


def _main(root: Path, candidates_path: Path, selected_path: Path | None) -> None:
    rows = json.loads(candidates_path.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not rows:
        print(f"候选 JSON 为空或格式异常: {candidates_path}")
        return
    print(f"candidates JSON: {candidates_path}")
    print(f"候选数（已过 generate_grasps 前置过滤）: {len(rows)}\n")

    # ---- survival funnel ----
    n_ap = sum("physical_aperture_ok" in r for r in rows)
    n_ap_ok = sum(_bool(r.get("physical_aperture_ok")) for r in rows)
    n_tg = sum("table_guard_ok" in r for r in rows)
    n_tg_ok = sum(r.get("table_guard_ok") is True for r in rows)
    n_reach = sum("reachability_ok" in r for r in rows)
    n_reach_ok = sum(r.get("reachability_ok") is True for r in rows)
    n_reach_def = sum(_bool(r.get("reachability_deferred")) for r in rows)
    n_wf = sum("workflow_preflight_ok" in r for r in rows)
    n_wf_ok = sum(r.get("workflow_preflight_ok") is True for r in rows)

    print("== 硬过滤漏斗（后置过滤） ==")
    print(f"  物理孔径 physical_aperture : {n_ap_ok}/{n_ap} 通过（未标注 {len(rows) - n_ap}）")
    print(f"  桌面保护 table_guard       : {n_tg_ok}/{n_tg} 通过（未标注 {len(rows) - n_tg}）")
    if n_reach_def:
        print(f"  可达性 reachability       : {n_reach_def} 个推迟到 whole-workflow（deferred）")
    print(f"  可达性 reachability       : {n_reach_ok}/{n_reach} 通过（未标注 {len(rows) - n_reach}）")
    print(f"  全流程预检 workflow       : {n_wf_ok}/{n_wf} 通过（未标注 {len(rows) - n_wf}）")
    print()

    # ---- ranking sanity: raw score vs priority ----
    if all("raw_score" in r for r in rows):
        by_raw = sorted(range(len(rows)), key=lambda i: -_f(rows[i].get("raw_score")))
        by_prio = sorted(range(len(rows)), key=lambda i: -_f(rows[i].get("priority")))
        top_raw = by_raw[0]
        top_raw_prio_rank = by_prio.index(top_raw) + 1
        print("== 排序 sanity ==")
        print(
            f"  网络最高分 raw_score 的候选是 idx={rows[top_raw]['idx']}，"
            f"在 priority 排序里排第 {top_raw_prio_rank} 位"
        )
        if top_raw_prio_rank > 1:
            print(
                "  -> 网络的最高分候选被 priority 启发式（竖直/中心/避障/IK）压了下去，"
                "这可能是‘选出来的不像人认为最优’的直接原因之一。"
            )
        print()

    # ---- per-candidate table ----
    print("== 候选逐条 ==")
    print("  OK=通过全部后置硬过滤；ap=物理孔径 tbl=桌面 reach=可达(wf=全流程预检；D=推迟")
    for i, row in enumerate(rows):
        print(_candidate_line(int(row.get("idx", i)), row))

    # ---- kill reasons ----
    print("\n== 被删原因汇总 ==")
    reasons: dict[str, int] = {}
    for row in rows:
        for key, label in (
            ("physical_aperture_reason", "ap"),
            ("reachability_reason", "reach"),
            ("workflow_preflight_reason", "wf"),
            ("workflow_preflight_direct_reason", "wf_direct"),
        ):
            if row.get(key):
                reasons.setdefault(f"{label}:{str(row[key])[:60]}", 0)
                reasons[f"{label}:{str(row[key])[:60]}"] += 1
    if reasons:
        for k, c in sorted(reasons.items(), key=lambda kv: -kv[1]):
            print(f"  [{c}x] {k}")
    else:
        print("  （无被删原因字段，可能本 run 未触发后置过滤删除）")

    # ---- selected ----
    if selected_path is not None:
        try:
            sel = json.loads(selected_path.read_text(encoding="utf-8"))
            cand = sel.get("selected_candidate") or {}
            print(f"\n== 最终选中（{selected_path.name}） ==")
            print(f"  idx={cand.get('idx')}  raw_score={cand.get('raw_score')}  priority={cand.get('priority')}")
            print(f"  recommended_reason={cand.get('recommended_reason')}")
        except Exception as exc:  # noqa: BLE001
            print(f"\n[selected] 读取失败: {type(exc).__name__}: {exc}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("run_dir", nargs="?", help="日志运行目录（含 grasp_candidates.json）")
    args = ap.parse_args(argv)

    if args.run_dir:
        root = Path(args.run_dir)
    else:
        logs = Path(__file__).resolve().parents[1] / "logs"
        hit = _find_newest(logs, "grasp_candidates.json")
        if hit is None:
            print("未找到任何 grasp_candidates.json，请指定 run_dir。", file=sys.stderr)
            return 1
        root = hit.parent
        print(f"[auto] 使用最新包含候选 JSON 的运行目录: {root}")

    candidates = _find_newest(root, "grasp_candidates.json")
    if candidates is None:
        print(f"目录 {root} 下未找到 grasp_candidates.json", file=sys.stderr)
        return 1
    selected = _find_newest(root, "selected_grasp_pose.json")
    _main(root, candidates, selected)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
