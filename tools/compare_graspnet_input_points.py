#!/usr/bin/env python3
"""Replay fixed-3000 versus every prepared target point, without hardware."""
import argparse
import copy
import hashlib
import html
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import compare_mgpc_graspnet as replay


def fingerprint_inputs(input_dir):
    files = [input_dir/'manifest.json', ROOT/'logs/log_rs/checkpoint-rs.tar']
    files += list(input_dir.glob('*/*base.npy'))
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(files)}


def snapshot(opts,pools):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from tools.render_mgpc_graspnet import preview_capsules
    settings=json.loads((opts.output_dir/'fixed/evaluation_settings.json').read_text())
    args=SimpleNamespace(**settings['args'])
    center=np.asarray(json.loads((opts.input_dir/'manifest.json').read_text())['target_center_base'])
    fig=plt.figure(figsize=(14,18),facecolor='#f5f7fa')
    background=np.load(opts.input_dir/'background.npz')
    for row_index,(letter,name,predicted,_) in enumerate(replay.CASES):
        measured=np.load(opts.input_dir/name/'observed_base.npy')
        rgb=np.load(opts.input_dir/name/'observed_rgb.npy')
        for col,mode in enumerate(('fixed','all')):
            p=next(p for p in pools if (p['case'],p['seed'],p['mode'])==(letter,0,mode))
            b=p['best'];ax=fig.add_subplot(4,2,row_index*2+col+1,projection='3d')
            env=background[name];colors=background[name+'_rgb'];stride=max(1,len(env)//7000)
            ax.scatter(*env[::stride].T,c=colors[::stride],s=.6,alpha=.55,rasterized=True)
            ax.scatter(*measured.T,c=rgb,s=3,alpha=.9,rasterized=True)
            if predicted:
                proposal=np.load(opts.input_dir/name/'prediction_base.npy')
                ax.scatter(*proposal[::3].T,c='#ed9336',s=1,alpha=.20,rasterized=True)
            caption='no preflight-passing grasp'
            if b:
                for a,z,_,label in preview_capsules(b['grasp_tcp_preview'],args):
                    ax.plot(*np.asarray([a,z]).T,c='#d2a000',linewidth=5 if label=='gripper_body' else 3)
                path=np.array([b[k][:3] for k in ('pre_tcp_preview','grasp_tcp_preview','lift_tcp_preview')])
                ax.plot(*path.T,c='#3977d1',linestyle='--',linewidth=1.3)
                caption=f"ID {b['idx']} | bilateral {b['recommended_bilateral_score']:.3f} | route {b['workflow_preflight_route_l2_deg']:.1f} deg"
            ax.set_title(f"{letter} / {mode} / {p['input']['sampled_points']} points\n{caption}",fontsize=10)
            ax.set(xlim=(center[0]-.16,center[0]+.16),ylim=(center[1]-.16,center[1]+.16),zlim=(-.01,.28),
                   xlabel='Base X (m)',ylabel='Base Y (m)',zlabel='Base Z (m)')
            ax.set_box_aspect((1,1,.9));ax.view_init(elev=33,azim=-65);ax.tick_params(labelsize=7)
    fig.suptitle('Seed 0 | default ranking | yellow: gripper capsule axes | blue: TCP keypoints',fontsize=13)
    fig.subplots_adjust(left=.02,right=.98,top=.95,bottom=.025,wspace=.02,hspace=.15)
    fig.savefig(opts.output_dir/'top_candidates.png',dpi=150);plt.close(fig)
    background.close()


def reports(opts):
    from real_nero_grasp.grasp_candidates import recommend_grasp_candidate
    from tools.rerank_grasp_weights import eligible
    pools=[]
    for mode in ('fixed','all'):
        directory=opts.output_dir/mode
        settings=json.loads((directory/'evaluation_settings.json').read_text())
        args=SimpleNamespace(**settings['args'])
        for letter,_,_,title in replay.CASES:
            for seed in opts.seeds:
                audit=json.loads((directory/letter/f'audit_seed{seed}.json').read_text())
                meta=json.loads((directory/letter/f'input_seed{seed}.json').read_text())
                rows=[copy.deepcopy(c['production']) for c in audit['candidates']
                      if c['status']=='workflow_pass' and eligible(c.get('production',{}))]
                best=recommend_grasp_candidate(rows,args)
                pools.append(dict(mode=mode,case=letter,title=title,seed=seed,input=meta,
                    summary=audit['summary'],passing=rows,best=best))
    fixed=json.loads((opts.output_dir/'fixed/evaluation_settings.json').read_text())
    full=json.loads((opts.output_dir/'all/evaluation_settings.json').read_text())
    if fixed != full:
        raise RuntimeError('Evaluation settings differ between input policies')
    with np.load(opts.output_dir/'fixed/safety_scene.npz') as a, np.load(opts.output_dir/'all/safety_scene.npz') as b:
        for key in a.files:
            if not np.array_equal(a[key],b[key]):
                raise RuntimeError(f'Measured safety scene changed: {key}')
    for letter,_,_,_ in replay.CASES:
        a=np.load(opts.output_dir/f'fixed/{letter}/network_target_camera.npy')
        b=np.load(opts.output_dir/f'all/{letter}/network_target_camera.npy')
        if not np.array_equal(a,b): raise RuntimeError(f'Prepared target changed: {letter}')
        for seed in opts.seeds:
            ids=np.load(opts.output_dir/f'all/{letter}/sampled_indices_seed{seed}.npy')
            if not np.array_equal(ids,np.arange(len(b))):
                raise RuntimeError(f'All-input policy sampled or padded points: {letter}')
    pair_checks={}
    for letter,_,_,_ in replay.CASES:
        pair_checks[letter]=dict(
            same_raw_candidates_seed0=np.array_equal(
                np.load(opts.output_dir/f'fixed/{letter}/raw_seed0.npy'),
                np.load(opts.output_dir/f'all/{letter}/raw_seed0.npy')),
            all_seeds_identical=all(np.array_equal(
                np.load(opts.output_dir/f'all/{letter}/raw_seed{opts.seeds[0]}.npy'),
                np.load(opts.output_dir/f'all/{letter}/raw_seed{seed}.npy')) for seed in opts.seeds))
    fixed_matches_saved=[]
    for letter,_,_,_ in replay.CASES:
        for seed in opts.seeds:
            old=ROOT/f'artifacts/mgpc_graspnet_compare/{letter}/raw_seed{seed}.npy'
            fixed_matches_saved.append(old.exists() and np.array_equal(np.load(old),
                np.load(opts.output_dir/f'fixed/{letter}/raw_seed{seed}.npy')))
    replay.write_json(opts.output_dir/'results.json',dict(pools=pools,
        validation=dict(same_settings=True,same_measured_safety_scene=True,
            same_prepared_targets=True,all_input_order_preserved=True,hardware_connected=False),
        candidate_comparison=pair_checks,fixed_matches_previous_saved_raw=all(fixed_matches_saved),
        limitations=['One saved short-bottle scene, not the new banana scene.',
            'Numerical preflight only, with the same per-pool planning budget.',
            'All mode preserves point order; seeds do not produce independent point sets.',
            'Direction/center recommendation features use the existing replay defaults.',
            'Inference timings exclude initialization, preprocessing and path planning.']))
    lines=['# GraspNet 输入点数离线对比','',
        '同一旧短瓶场景；固定 3000 点 vs 清理裁剪后全量输入，不补点、不打乱顺序。',
        '四组使用相同模型、裁剪、实测安全环境、过滤和默认 10% 网络评分权重。',
        f'每池最多对 {opts.preflight_topk} 个通过前置检查的候选做数值路径预检；未检查不计通过。',
        '全量方案的不同 seed 输入完全相同，不计作独立试验。没有实机抓取成功率结论。','',
        '| 组 / seed | 输入方式 | 裁剪后 / 输入点数 | 解码 | 简化碰撞后 | 路径通过 | 路径未查 | 首选 ID | 原始分数 | 双侧支持分 | 关节路程° | 推理 ms |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    def fmt(value): return '—' if value is None else f'{value:.3f}'
    for letter,_,_,_ in replay.CASES:
        for seed in opts.seeds:
            for mode in ('fixed','all'):
                p=next(p for p in pools if (p['case'],p['seed'],p['mode'])==(letter,seed,mode))
                m,s,b=p['input'],p['summary'],p['best'] or {}
                count={x['stage']:x['remaining'] for x in s['counts']}
                values=[f'{letter} / {seed}',mode,f"{m['prepared_points']} / {m['sampled_points']}",
                    s['decoded'],count['network_collision'],s['status_counts']['workflow_pass'],
                    s['status_counts']['workflow_not_checked'],b.get('idx','—'),fmt(b.get('raw_score')),
                    fmt(b.get('recommended_bilateral_score')),fmt(b.get('workflow_preflight_route_l2_deg')),
                    f"{m['median_seconds']*1000:.1f}"]
                lines.append('| '+' | '.join(map(str,values))+' |')
    lines += ['', '候选 ID 只在本方案、本组、本 seed 内有意义，跨池相同 ID 不代表同一姿态。',
        '完整拒绝原因、TCP 位姿、实测接触和路径碰撞位置在两个候选浏览器中查看。',
        '第一层固定采样 2048 点；输入不足时的实际重复索引数量记录在各池 input_seed*.json。',
        '推理耗时为预热后多次前向＋解码的中位数，不包含模型加载和规划；不同 seed 的速度小差异不可视为收益。','']
    lines += ['## 结果解读','',
        'A/B 两方案的原始候选完全相同，首选也相同，取消输入端重复补点没有改变本场景的抓取候选。',
        'B 组 1724 点已实际完成前向推理和重复一致性检查；第一层输出 2048 个索引，其中只有 1724 个不同索引，因此网络内部仍重复采样。','',
        '| 组 | 3000 点各 seed 的数值路径通过数 | 全量各 seed 的数值路径通过数 |',
        '|---|---|---|']
    for letter,_,_,_ in replay.CASES:
        counts={mode:[str(p['summary']['status_counts']['workflow_pass']) for p in pools
                      if p['mode']==mode and p['case']==letter] for mode in ('fixed','all')}
        lines.append(f"| {letter} | {' / '.join(counts['fixed'])} | {' / '.join(counts['all'])} |")
    c_full=next(p for p in pools if p['mode']=='all' and p['case']=='C')
    c_counts=c_full['summary']['status_counts']
    lines += ['',
        f"C 全量组 {c_full['summary']['decoded']} 个解码候选中，{c_counts['network_collision']} 个被简化碰撞排除，另 {c_counts['target_empty']} 个因实测目标闭合区为空被排除。",
        '全量 D 组增加了解码候选数量，但本场景没有增加数值路径通过候选的数量。其首选的双侧实测支持分较好，仍需同时考虑关节运动代价。',
        '这些结果不足以支持把主流程默认值统一改成全量。当前保留 3000 点，并在更多场景重复对比。全量与固定方案同时改变输入数量和点顺序策略，不能把差异全部归因于数量。','']
    lines += ['## 一致性检查','',
        f"固定 3000 点的 {4*len(opts.seeds)} 池候选是否与此前保存的原始输出逐元素相同：{all(fixed_matches_saved)}。",
        '两方案的裁剪后目标、实测安全场景、评估配置逐项相同；全量输入索引保持原顺序，未采样或补点。','',
        '| 组 | 两方案 seed 0 原始候选完全相同 | 全量三个 seed 原始候选完全相同 |',
        '|---|---|---|']
    for letter,checks in pair_checks.items():
        lines.append(f"| {letter} | {checks['same_raw_candidates_seed0']} | {checks['all_seeds_identical']} |")
    (opts.output_dir/'SUMMARY.md').write_text('\n'.join(lines))
    table=[]
    for line in lines:
        if line.startswith('|') and not line.startswith('|---'):
            cells=[c.strip() for c in line.strip('|').split('|')]
            table.append('<tr>'+''.join('<td>'+html.escape(c)+'</td>' for c in cells)+'</tr>')
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>GraspNet 输入点数对比</title>
<style>body{background:#101821;color:#dce7f0;font:15px system-ui;margin:22px}a{color:#5bd5dc}iframe{width:100%;height:1500px;border:1px solid #435769}table{border-collapse:collapse}td{border:1px solid #435769;padding:7px}p{max-width:1150px;line-height:1.7}</style>
<h1>固定 3000 点 / 全量目标点 · 离线对比</h1>
<p>相同的旧短瓶点云、模型和过滤规则。全量输入保留全部清理裁剪后的目标点，保持原顺序。四组 A/B/C/D 不变；全量方案三个 seed 不是三个独立样本。通过指保存场景的数值路径预检通过。</p>
<p><a href="SUMMARY.md">结果表</a> · <a href="results.json">完整数据</a> · <a href="fixed/comparison.html" target="_blank">单独打开 3000 点候选</a> · <a href="all/comparison.html" target="_blank">单独打开全量候选</a></p>
<div style="overflow:auto"><table>TABLE</table></div><p><a href="top_candidates.png">查看 seed 0 首选姿态对照图</a>。候选 ID 只在各自方案内有意义。推理耗时不含模型加载、预处理和规划。</p><h2>固定 3000 点</h2><iframe loading="lazy" src="fixed/comparison.html"></iframe><h2>全量输入</h2><iframe loading="lazy" src="all/comparison.html"></iframe></html>'''
    (opts.output_dir/'comparison.html').write_text(page.replace('TABLE',''.join(table)))
    snapshot(opts,pools)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir',type=Path,default=ROOT/'artifacts/mgpc_compare')
    parser.add_argument('--output-dir',type=Path,default=ROOT/'artifacts/graspnet_input_points_compare')
    parser.add_argument('--seeds',type=int,nargs='+',default=[0,1,2])
    parser.add_argument('--preflight-topk',type=int,default=24)
    parser.add_argument('--stage',choices=['infer','evaluate','report','all'],default='all')
    opts=parser.parse_args();opts.output_dir.mkdir(parents=True,exist_ok=True)
    before=fingerprint_inputs(opts.input_dir)
    for mode in ('fixed','all'):
        child=SimpleNamespace(**vars(opts));child.output_dir=opts.output_dir/mode
        child.output_dir.mkdir(exist_ok=True);child.sampling_mode=mode
        child.timing_repeats=5;child.force=False;child.rebuild_audits=False
        saved_settings=child.output_dir/'evaluation_settings.json'
        if saved_settings.exists():
            old_budget=json.loads(saved_settings.read_text())['args']['grasp_preflight_max_candidates']
            child.rebuild_audits=old_budget != opts.preflight_topk
        if opts.stage in ('infer','all'): replay.infer(child)
        if opts.stage in ('evaluate','all'): replay.evaluate(child)
        if opts.stage in ('report','all'):
            from tools.render_mgpc_graspnet import render
            render(child)
    if opts.stage in ('report','all'): reports(opts)
    after=fingerprint_inputs(opts.input_dir)
    if before != after: raise RuntimeError('Source inputs changed during replay')
    replay.write_json(opts.output_dir/f'provenance_{opts.stage}.json',dict(
        inputs_sha256=before,source_inputs_unchanged=True,hardware_connected=False,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        replay_sha256=hashlib.sha256(Path(replay.__file__).read_bytes()).hexdigest()))


if __name__=='__main__': main()
