#!/usr/bin/env python3
"""Reweight saved, preflighted candidates. No inference or robot entrypoints.

Freeze candidate membership, poses, score components, normalization and table
preference. Only the network coefficient changes; all other coefficients are
scaled by (1 - network_weight) / .9. Saved outcomes are not training labels.
"""
import argparse
import copy
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
WEIGHTS = {"network": .10, "bilateral": .25, "direction": .25, "fit": .10,
           "route": .10, "orientation": .15, "center": .025, "table": .025}
FIELDS = {"network": "recommended_raw_score_normalized",
          "bilateral": "recommended_bilateral_score",
          "fit": "recommended_width_fit_score", "route": "recommended_route_score",
          "orientation": "recommended_orientation_score",
          "center": "recommended_center_score", "table": "recommended_table_score"}
POLICIES = (.10, .40, .70)
LABELS = {"A": "单视角实测", "B": "多视角融合实测",
          "C": "单视角 → MGPC", "D": "多视角融合 → MGPC"}


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items() if k != "grasp"}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [clean(v) for v in value]
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def dump(path, value):
    path.write_text(json.dumps(clean(value), ensure_ascii=False, indent=2,
                               allow_nan=False) + "\n", encoding="utf-8")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def coefficients(network_weight):
    if not np.isfinite(network_weight) or not 0 <= network_weight <= 1:
        raise ValueError("network weight must be finite and in [0, 1]")
    scale = (1 - network_weight) / .9
    return {k: network_weight if k == "network" else v * scale
            for k, v in WEIGHTS.items()}


def components(row):
    values = {k: float(row[field]) for k, field in FIELDS.items()}
    direction = row.get("target_cross_axis_score")
    values["direction"] = .5 if direction is None else float(np.clip(direction, 0, 1))
    if not all(np.isfinite(v) and 0 <= v <= 1 for v in values.values()):
        raise ValueError("missing, nonfinite or out-of-range frozen score components")
    return values


def rank_rows(rows, network_weight):
    """Preserve the production table preference and tie-break ordering."""
    coefs = coefficients(network_weight)
    safe = [r for r in rows if r.get("recommended_table_safe") is not False]
    pool = safe if safe else rows
    ranked = []
    for row in pool:
        terms = {k: coefs[k] * v for k, v in row["components"].items()}
        ranked.append(dict(candidate_id=int(row["idx"]), score=sum(terms.values()),
                           contributions=terms, raw_score=float(row["raw_score"]),
                           priority=float(row.get("priority", 0))))
    ranked.sort(key=lambda r: (r["score"], r["raw_score"], r["priority"]), reverse=True)
    for rank, row in enumerate(ranked, 1):
        row["rank"] = rank
    return ranked


def eligible(row):
    # Never infer a pass from missing annotations or a good score.
    return (row.get("workflow_preflight_ok") is True
            and row.get("physical_aperture_ok") is True
            and row.get("table_guard_ok") is True
            and row.get("reachability_ok") is True)


def freeze_case(key, title, path, all_rows, rows, provenance, args=None):
    if args is not None and rows:
        from real_nero_grasp.grasp_candidates import recommend_grasp_candidate
        recommend_grasp_candidate(rows, args)
    frozen = []
    for row in rows:
        item = clean(row)
        item["components"] = components(row)
        baseline = sum(WEIGHTS[k] * v for k, v in item["components"].items())
        saved = row.get("recommendation_score")
        if saved is not None and abs(baseline - float(saved)) > 1e-10:
            raise ValueError(f"{path} candidate {row['idx']}: baseline formula mismatch")
        item["baseline_reproduction_error"] = 0 if saved is None else abs(baseline-float(saved))
        frozen.append(item)
    rankings = {str(round(w * 100)): rank_rows(frozen, w) for w in POLICIES}
    tops = {k: v[0]["candidate_id"] if v else None for k, v in rankings.items()}
    pose_bytes = json.dumps([[r["idx"], r["raw_score"], r.get("grasp_tcp_preview")]
                             for r in frozen], sort_keys=True).encode()
    return dict(key=key, title=title, generator="graspnet", source=str(path.relative_to(ROOT)),
                source_sha256=digest(path), provenance=provenance,
                decoded_or_saved_count=len(all_rows), eligible_count=len(frozen),
                excluded_count=len(all_rows)-len(frozen),
                normalization="frozen production min-max over unchanged eligible pool",
                candidate_pose_fingerprint=hashlib.sha256(pose_bytes).hexdigest(),
                top_ids=tops, changed=bool(frozen and any(v != tops['10'] for v in tops.values())),
                informative=len(rankings['10']) >= 2,
                rows=frozen, rankings=rankings)


def collect_cases(audit_dir, log_dir):
    cases, skipped = [], []
    settings = json.loads((audit_dir / "evaluation_settings.json").read_text())
    args = SimpleNamespace(**settings["args"])
    for path in sorted(audit_dir.glob("*/audit_seed*.json")):
        audit = json.loads(path.read_text())
        letter, seed = audit["summary"]["case"], audit["summary"]["seed"]
        all_rows = audit["candidates"]
        rows = [copy.deepcopy(r["production"]) for r in all_rows
                if r["status"] == "workflow_pass" and eligible(r.get("production", {}))]
        case = freeze_case(f"{letter}_seed{seed}", f"{letter} · {LABELS[letter]} · seed {seed}",
                           path, all_rows, rows,
                           "saved numerical preflight; current production recommendation on saved rows",
                           args=args)
        case.update(kind="audit", letter=letter, seed=seed,
                    status_counts=dict(Counter(r["status"] for r in all_rows)),
                    notes=["原离线候选未保存方向和中心评分输入，沿用生产函数的中性默认值 0.5。",
                           "仅在原数值路径检查通过的集合内重排；未检查、碰撞和失败候选不进入排名。"])
        raw_path = path.parent / f"raw_seed{seed}.npy"
        case["raw_candidates_sha256"] = digest(raw_path)
        cases.append(case)
    for path in sorted(log_dir.rglob("grasp_candidates.json")):
        data = json.loads(path.read_text())
        if not isinstance(data, list):
            continue
        passing = [r for r in data if eligible(r)]
        if not passing:
            skipped.append(dict(source=str(path.relative_to(ROOT)), reason="no complete saved preflight pass"))
            continue
        if any(any(field not in r for field in FIELDS.values()) for r in passing):
            skipped.append(dict(source=str(path.relative_to(ROOT)), reason="missing frozen score components"))
            continue
        if any(r.get('generator', 'graspnet') != 'graspnet' for r in passing):
            skipped.append(dict(source=str(path.relative_to(ROOT)), reason="unsupported generator; do not mix score scales"))
            continue
        case = freeze_case("log_" + path.parent.name, "历史采集 · " + path.parent.name,
                           path, data, passing, "exact frozen production recommendation components")
        case.update(kind="historical", capture_dir=str(path.parent.relative_to(ROOT)),
                    saved_recommended_ids=[int(r["idx"]) for r in passing if r.get("recommended_grasp")],
                    notes=["历史碰撞与路径检查来自当时的配置；本次未重新认证其几何安全性。",
                           "保存的 10% 总分被逐候选精确复现；仅调整网络权重。"])
        cases.append(case)
    return cases, skipped, settings


def sample_cloud(points, rgb=None, cap=6000):
    points = np.asarray(points)
    rgb = np.asarray(rgb) if rgb is not None and len(rgb) == len(points) else None
    ids = np.linspace(0, len(points)-1, min(len(points), cap), dtype=int) if len(points) else []
    return dict(points=np.round(points[ids], 5),
                rgb=np.rint(np.clip(rgb[ids], 0, 1)*255).astype(int) if rgb is not None else None)


def read_cloud(path, cap):
    import open3d as o3d
    if not path.is_file():
        return dict(points=[], rgb=None)
    cloud = o3d.io.read_point_cloud(str(path))
    return sample_cloud(np.asarray(cloud.points), np.asarray(cloud.colors), cap)


def scene_payload(case, input_dir, settings):
    if case["kind"] == "audit":
        name = "single" if case["letter"] in "AC" else "multi"
        predicted = case["letter"] in "CD"
        with np.load(input_dir/"background.npz") as bg:
            environment = sample_cloud(bg[name], bg[name+"_rgb"], 6500)
        measured = np.load(input_dir/name/"observed_base.npy")
        target = sample_cloud(measured, np.load(input_dir/name/"observed_rgb.npy"), 3000)
        prediction = sample_cloud(np.load(input_dir/name/"prediction_base.npy"), cap=2500) if predicted else None
    else:
        folder = ROOT / case["capture_dir"]
        environment = read_cloud(folder/"motion_safety_environment_base.ply", 6500)
        target = read_cloud(folder/"motion_safety_target_base.ply", 3000)
        prediction = None
    visible = np.asarray(target["points"])
    center = np.median(visible, axis=0) if visible.size else np.zeros(3)
    return dict(environment=environment, target=target, prediction=prediction, center=center)


def render(cases, settings, input_dir, out_dir):
    from plotly.offline import get_plotlyjs
    geometry = {}
    for case in cases:
        key = ("audit_" + ("single" if case.get("letter") in "AC" else "multi")
               + ("_mgpc" if case.get("letter") in "CD" else "")
               if case["kind"] == "audit" else case["key"])
        if key not in geometry:
            geometry[key] = scene_payload(case, input_dir, settings)
        case["scene_key"] = key
    payload = json.dumps(clean(dict(cases=cases, scenes=geometry, weights=WEIGHTS)),
                         ensure_ascii=False, allow_nan=False).replace("</", "<\\/")
    template = (ROOT/"tools/grasp_weight_rerank_viewer.html").read_text()
    page = template.replace("/*PLOTLY_LIBRARY*/", get_plotlyjs()).replace("/*RERANK_DATA*/", payload)
    (out_dir/"comparison.html").write_text(page, encoding="utf-8")
    render_snapshot(cases, geometry, out_dir)


def render_snapshot(cases, geometry, out_dir):
    """Export a small, readable figure for the three distinct changed pairs."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from scipy.spatial.transform import Rotation
    selected = [c for c in cases if c['changed'] and
                (c['kind'] == 'historical' or c.get('seed') == 0 or c['key'] == 'D_seed1')]
    fig = plt.figure(figsize=(15, 4.5*len(selected)), facecolor='#f5f7fa')
    for row_index, case in enumerate(selected):
        scene = geometry[case['scene_key']]
        for col, policy in enumerate(('10', '40', '70')):
            entry = case['rankings'][policy][0]
            row = next(r for r in case['rows'] if r['idx'] == entry['candidate_id'])
            ax = fig.add_subplot(len(selected), 3, row_index*3+col+1, projection='3d')
            for key, color in [('environment', '#aab4bf'), ('target', '#60c8d2')]:
                cloud = scene[key]
                p = np.asarray(cloud['points'])
                if len(p):
                    rgb = np.asarray(cloud['rgb'])/255 if cloud['rgb'] is not None else color
                    ax.scatter(*p.T, c=rgb, s=1 if key == 'environment' else 3, alpha=.65,
                               depthshade=False, rasterized=True)
            pose = np.asarray(row['grasp_tcp_preview'])
            rot = Rotation.from_euler('xyz', pose[3:]).as_matrix()
            half = float(row['physical_aperture_opening'])/2
            def at(x, z):
                return pose[:3]+rot @ np.array([x, 0, z])
            ax.plot(*np.array([at(-.065, -half), at(-.065, half)]).T, color='#da9e00', linewidth=5)
            for sign in (-1, 1):
                ax.plot(*np.array([at(-.065, sign*half), at(.020, sign*half)]).T, color='#da9e00', linewidth=4)
            path = np.array([row[k][:3] for k in ('pre_tcp_preview','grasp_tcp_preview','lift_tcp_preview')])
            ax.plot(*path.T, color='#3679d1', linewidth=1.5, linestyle='--')
            center = np.asarray(scene['center'])
            ax.set(xlim=(center[0]-.15, center[0]+.15), ylim=(center[1]-.15, center[1]+.15),
                   zlim=(-.01,.24), xlabel='Base X / m', ylabel='Base Y / m', zlabel='Base Z / m')
            ax.set_box_aspect((1,1,.83))
            ax.view_init(elev=35, azim=-65)
            ax.tick_params(labelsize=7)
            label = case['key'].replace('log_', '')
            ax.set_title(f"{label} | network {policy}% | ID {row['idx']}\n"
                         f"raw {row['raw_score']:.3f} | bilateral {row['components']['bilateral']:.3f} | "
                         f"joint route {row['workflow_preflight_route_l2_deg']:.1f} deg", fontsize=9)
    fig.suptitle('Frozen candidates: yellow = TCP gripper sketch, blue = saved TCP keypoints', fontsize=13)
    fig.subplots_adjust(left=.02, right=.98, top=.94, bottom=.02, wspace=.05, hspace=.16)
    fig.savefig(out_dir/'changed_top1.png', dpi=150)
    plt.close(fig)


def write_reports(cases, skipped, output):
    comparable = [c for c in cases if c["informative"]]
    changed = [c for c in comparable if c["changed"]]
    summary = dict(pools=len(cases), nonempty=sum(c["eligible_count"] > 0 for c in cases),
                   informative_pools=len(comparable), changed_top1_pools=len(changed),
                   eligible_candidates=sum(c["eligible_count"] for c in cases),
                   hardware_connected=False, inference_run=False, live_defaults_modified=False,
                   genx_full_preflight_data_available=False,
                   baseline_max_error=max((r["baseline_reproduction_error"] for c in cases for r in c["rows"]), default=0))
    dump(output/"results.json", dict(summary=summary, policies={str(int(w*100)):coefficients(w) for w in POLICIES},
         cases=cases, skipped_sources=skipped,
         scoring_source_sha256=digest(ROOT/"real_nero_grasp/grasp_candidates.py"),
         limitations=["离线排序变化不是抓取成功率。保存位姿不得直接用于实机执行，须重新采集并检查。",
                      "香蕉日志尚无候选，GenX 尚无完整重排输入；本报告只含已有 GraspNet 数据。",
                      "A/B 三个种子生成的原始候选各自相同，不是独立场景；本次未扩大原预检预算。",
                      "同一候选池、姿态、归一化、过滤结果保持不变；缺失分量不从其他数据补造。",
                      "源记录的安全性只按历史标注保留，不代表当前代码或当前实机的安全认证。"] ))
    with (output/"rankings.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=["pool", "network_weight", "rank", "candidate_id", "score", "raw_score", *WEIGHTS])
        writer.writeheader()
        for case in cases:
            for policy, ranking in case["rankings"].items():
                for row in ranking:
                    writer.writerow(dict(pool=case["key"], network_weight=policy, rank=row["rank"],
                        candidate_id=row["candidate_id"], score=row["score"], raw_score=row["raw_score"], **row["contributions"]))
    text = ["# 网络权重离线重排", "", "同一候选池比较 10% / 40% / 70%；其余七项按原比例缩减。只重排保存检查通过的候选。", "",
            f"共 {len(cases)} 个池，{len(comparable)} 个至少有两项候选；其中 {len(changed)} 个池的首选发生变化。",
            "A/B 各三个种子原始候选相同；上述池数不是独立场景数。没有执行成功率结论。", "",
            "| 数据池 | 可排候选 | 10% 首选 | 40% 首选 | 70% 首选 |", "| --- | ---: | ---: | ---: | ---: |"]
    for case in cases:
        ids=[str(case['top_ids'][k]) if case['top_ids'][k] is not None else '无候选' for k in ('10','40','70')]
        text.append(f"| {case['title']} | {case['eligible_count']} | {' | '.join(ids)} |")
    text += ["", "40% 和 70% 在本批数据中的首选全部一致。后续可先对比 10% 与 40%，尚不能据此选择最佳权重。",
             "", "## 首选改变时的取舍", "",
             "| 候选对 | 网络原始分（10%→40%） | 双侧支持（10%→40%） | 关节路径代价 / 度（10%→40%） |",
             "| --- | --- | --- | --- |"]
    for case in changed:
        if case.get('letter') == 'A' and case.get('seed') != 0:
            continue
        pair = [next(r for r in case['rows'] if r['idx'] == case['top_ids'][key]) for key in ('10','40')]
        old, new = pair
        text.append(f"| {case['key']}：{old['idx']}→{new['idx']} | {old['raw_score']:.3f}→{new['raw_score']:.3f} | "
                    f"{old['components']['bilateral']:.3f}→{new['components']['bilateral']:.3f} | "
                    f"{old['workflow_preflight_route_l2_deg']:.1f}→{new['workflow_preflight_route_l2_deg']:.1f} |")
    text += ["", "关节路径代价是记录的关节变化量指标，不是执行时间。网络原始分不是成功概率。",
             "A 组提高网络权重会牺牲双侧实测支持；D seed1 和 23:11 历史组则选择关节运动代价更高的候选。",
             "![首选变化示意](changed_top1.png)"]
    text += ["", "## 后续真机验证", "", "优先验证首选发生变化的数据对应的场景，重点比较原首选与新首选的接触区、姿态和运动代价。",
             "本次未修改主流程权重，也未提供直接执行历史位姿的入口。重搭场景后须重新采集、生成候选和完成预检。",
             "需要在真机切换权重时，再给实时推荐函数增加经过验证的权重参数；历史候选 ID 不适用于新采集。", "",
             "## 数据边界", "", "香蕉新日志没有生成候选，GraspGenX 只有早期独立候选／桌面检查，未混入这次完整流程重排。",
             "历史采集使用保存的各分量；四组离线审计通过当前生产函数计算分量，缺少的方向／中心输入沿用中性默认值 0.5。",
             "没有重新推理、扩大候选预算或放宽任何过滤条件。预测点云只作可视化参考。", "",
             "## 打开", "", "```bash", "cd /home/yx/yx/panda_ws/src/whole-grasp",
             "python3 -m http.server 8082 --bind 127.0.0.1 --directory artifacts/grasp_weight_rerank", "```", "",
             "浏览器访问 http://127.0.0.1:8082/comparison.html 。", ""]
    (output/"SUMMARY.md").write_text("\n".join(text), encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-dir", type=Path, default=ROOT/"artifacts/mgpc_graspnet_compare")
    parser.add_argument("--input-dir", type=Path, default=ROOT/"artifacts/mgpc_compare")
    parser.add_argument("--log-dir", type=Path, default=ROOT/"logs")
    parser.add_argument("--output-dir", type=Path, default=ROOT/"artifacts/grasp_weight_rerank")
    opts=parser.parse_args()
    for name in ('audit_dir', 'input_dir', 'log_dir', 'output_dir'):
        setattr(opts, name, getattr(opts, name).resolve())
    opts.output_dir.mkdir(parents=True, exist_ok=True)
    cases, skipped, settings = collect_cases(opts.audit_dir, opts.log_dir)
    render(cases, settings, opts.input_dir, opts.output_dir)
    summary=write_reports(cases, skipped, opts.output_dir)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(opts.output_dir/"comparison.html")


if __name__ == "__main__":
    main()
