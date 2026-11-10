#!/usr/bin/env python3
"""Compare bounded IK search using frozen saved poses and measured scenes.

This script never constructs a robot/camera driver or invokes execution. It
does not change production defaults. Reports distinguish original candidates
from their pose variants and retain the normal contact, CAD and route guards.
"""

import argparse
import contextlib
import copy
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.compare_mgpc_graspnet import NumericalArm, baseline_imports, cloud, plain
from real_nero_grasp import runtime as r, ik_solver
from real_nero_grasp.graspgenx_integration import generate_graspgenx_candidates
from real_nero_grasp.graspnet_process import _unpack
from real_nero_grasp.nero_refinement import geometry_variants, pack_grasp, qualify_hold
from real_nero_grasp.planning_process import close_planning_process, get_planning_process
from real_nero_grasp.types import DynamicEnvironmentMap


PROFILES = [('seeds5_top2', 5, 2, 0), ('seeds16_top2', 16, 2, 0),
            ('seeds16_top6', 16, 6, 0), ('global16_top6', 16, 6, 8)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, nargs='+', required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=1)
    opts = parser.parse_args()
    if opts.repeats < 1:
        parser.error('--repeats must be positive')
    opts.output_dir.mkdir(parents=True, exist_ok=True)
    baseline_imports()
    result = dict(hardware_connected=False, motion_commands_sent=False,
        changed_live_pipeline=False, frozen_network_proposals=True,
        full_route_definition='pregrasp + approach + support release + empty recovery + requested joint hold',
        profiles=[dict(name=n, seed_limit=s, topk=k, global_seed_count=g) for n,s,k,g in PROFILES],
        software_sha256={name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in (
            'real_nero_grasp/ik_solver.py', 'real_nero_grasp/grasp_planner.py',
            'real_nero_grasp/nero_refinement.py', 'config/pipeline.yaml')}, cases=[])
    for run in opts.run_dir:
        record = json.loads((run/'experiment_result.json').read_text())
        sys.argv = record['command']
        args = r.parse_args()
        args.execute = False
        args.table_z = float(args.grasp_table_reference_z)
        args.grasp_selective_ik_rescue = False
        args.grasp_progressive_full_fallback = False
        if args.mgpc:
            parser.error('Use measured non-MGPC A/B/E cases for this IK-only comparison')
        root = Path(record['artifacts']['primary_capture_dir'])
        output = opts.output_dir/run.name
        output.mkdir(exist_ok=True)
        report = json.loads((root/'table_reference/table_reference_report.json').read_text())
        with np.load(root/'table_reference/registered_scene.npz') as data:
            T = data['T_base_camera'].copy()
            scene = cloud(data['scene_points'])
            obstacles = cloud(data['environment_points'], data['environment_colors'])
            metadata = dict(report['environment_map_metadata'])
            metadata.update({key[4:]: data[key] for key in data.files if key.startswith('map_')})
            environment = DynamicEnvironmentMap(**metadata)
        target = r.get_o3d().io.read_point_cloud(str(root/'active_fused_target_camera0.ply'))
        plan_file = root/'whole_workflow_motion_plan.json'
        if plan_file.is_file():
            q = np.deg2rad(json.loads(plan_file.read_text())['start_joints_deg'])
        else:
            start_time = datetime.fromisoformat(next(e['timestamp'] for e in record['stage_events']
                if e['stage']=='grasp_planning')).timestamp()
            q = None
            with (run/'mit_control_cycles.jsonl').open() as cycles:
                for line in cycles:
                    cycle = json.loads(line)
                    if cycle['type']=='cycle' and cycle['wall_time'] <= start_time:
                        q = np.asarray(cycle['q_actual'])
            if q is None:
                raise ValueError('No saved measured start joints before grasp planning')
        args._empty_grasp_recovery_joints = np.deg2rad(args.initial_joints_deg)
        args.save_dir = str(output)
        with (output/'proposal_freeze.log').open('w') as log, contextlib.redirect_stdout(log):
            if args.grasp_generator == 'graspgenx':
                saved = json.loads((root/'graspgenx/candidates.json').read_text())
                def replay_worker(command, **kwargs):
                    request = json.loads((output/'graspgenx/request.json').read_text())
                    Path(request['output_json']).write_text(json.dumps(saved))
                _, rows = generate_graspgenx_candidates(target, T, args, output,
                    measured_target=target, runner=replay_worker)
            else:
                rows = _unpack(json.loads((root/'graspnet_worker/result.json').read_text())['rows'])
            # A successful saved run is a positive control: recheck the exact
            # selected pose without doing a new pose/depth search.
            if record['outcome']['success']:
                selected = record['selected_grasp']['idx']
                rows = [next(row for row in rows if row['idx']==selected)]
            variants = []
            for original in rows:
                variants.append(dict(parent_idx=int(original['idx']), family='original',
                    advance_m=float(getattr(original['grasp'], 'nero_tcp_contact_advance', args.grasp_advance))
                    if args.grasp_generator=='graspgenx' else args.grasp_advance,
                    row=copy.deepcopy(original)))
                if record['outcome']['success']:
                    continue
                geometry = geometry_variants(dict(idx=original['idx'], generator=args.grasp_generator,
                    grasp=pack_grasp(original['grasp'])), target, T, args)
                for proposal in geometry['proposals'][:args.nero_refine_max_route_variants]:
                    if proposal['family']=='native' and np.isclose(proposal['advance_m'], variants[-1]['advance_m']):
                        continue
                    row = copy.deepcopy(original)
                    row['grasp'] = SimpleNamespace(**proposal['grasp'])
                    variants.append(dict(parent_idx=int(original['idx']), family=proposal['family'],
                        advance_m=proposal['advance_m'], row=row))
        case = dict(source_run=str(run), generator=args.grasp_generator,
            start_joints_deg=np.rad2deg(q).tolist(), original_candidate_count=len(rows),
            frozen_pose_variant_count=len(variants), positive_control=bool(record['outcome']['success']),
            input_sha256={name:hashlib.sha256((root/name).read_bytes()).hexdigest()
                for name in ('active_fused_target_camera0.ply','table_reference/registered_scene.npz')},
            comparisons=[])
        result['cases'].append(case)
        print(f'CASE {run.name}: parents={len(rows)} frozen variants={len(variants)}', flush=True)
        for repeat in range(opts.repeats):
            for name, seeds, topk, global_count in PROFILES:
                close_planning_process()
                ik_solver._IK_SOLVER = ik_solver._IK_SOLVER_KEY = None
                cfg = copy.copy(args)
                cfg.grasp_fast_ik_seed_limit = seeds
                cfg.grasp_fast_ik_solution_topk = topk
                cfg.ik_all_joint_seed_count = global_count
                cfg.grasp_preflight_max_candidates = cfg.grasp_target_valid_plans = 1
                folder = output/f'{name}_repeat{repeat}'
                folder.mkdir(exist_ok=True)
                cfg.save_dir = str(folder)
                # Warm the same isolated solver in each profile before timing.
                solver = ik_solver._get_ik_solver(cfg)
                available = len(ik_solver._joint_seed_candidates(q, cfg, solver))
                arm = NumericalArm(r, cfg, q)
                entry = dict(profile=name, repeat=repeat, available_seed_count=available,
                    effective_seed_limit=min(seeds,available), attempts=[])
                case['comparisons'].append(entry)
                started = time.perf_counter()
                for index, item in enumerate(variants):
                    log_path = folder/f'pose{index}_{item["family"]}.log'
                    before = time.perf_counter()
                    with log_path.open('w') as log, contextlib.redirect_stdout(log):
                        probes = r.annotate_grasp_candidate_physical_aperture(
                            [copy.deepcopy(item['row'])], target, T, arm.get_tcp_pose(), cfg)
                        probes = r.annotate_grasp_candidate_reachability(probes,T,arm.get_tcp_pose(),cfg,arm=arm)
                        probes = r.annotate_grasp_candidate_workflow_preflight(probes,arm,cfg,
                            target,scene,T,arm.get_tcp_pose(),f'ik_compare_pose{index}',
                            motion_environment_cloud=obstacles,motion_environment_map=environment)
                        pickup_ok = bool(probes[0]['workflow_preflight_ok'])
                        stats = copy.deepcopy(cfg._planning_stats)
                        full_ok = qualify_hold(probes[0],cfg,f'ik_compare_pose{index}')
                    entry['attempts'].append(dict(parent_idx=item['parent_idx'],family=item['family'],
                        advance_m=item['advance_m'],pickup_route_ok=pickup_ok,full_route_ok=full_ok,
                        reason=probes[0]['workflow_preflight_reason'],seconds=time.perf_counter()-before,
                        planning_stats=plain(stats),log=str(log_path)))
                entry.update(seconds=time.perf_counter()-started,
                    full_route_pose_variants=sum(a['full_route_ok'] for a in entry['attempts']),
                    full_route_original_candidates=len({a['parent_idx'] for a in entry['attempts'] if a['full_route_ok']}),
                    ik_calls=sum(a['planning_stats']['ik_calls'] for a in entry['attempts']),
                    ik_seed_trials=sum(a['planning_stats']['ik_seed_trials'] for a in entry['attempts']))
                entry['numerical_worker'] = get_planning_process().request('ping', {})
                (opts.output_dir/'comparison.json').write_text(json.dumps(plain(result),indent=2,allow_nan=False)+'\n')
                print(f'{name} repeat={repeat}: actual seeds<={entry["effective_seed_limit"]}, '
                    f'full variants={entry["full_route_pose_variants"]}/{len(variants)}, '
                    f'parents={entry["full_route_original_candidates"]}/{len(rows)}, '
                    f'seconds={entry["seconds"]:.2f} seed_trials={entry["ik_seed_trials"]}',flush=True)
        close_planning_process()
    with (opts.output_dir/'comparison.csv').open('w',newline='') as stream:
        writer=csv.writer(stream)
        writer.writerow(['case','profile','repeat','effective_seed_limit','full_route_pose_variants',
                         'full_route_original_candidates','seconds','ik_calls','ik_seed_trials'])
        for case in result['cases']:
            for entry in case['comparisons']:
                writer.writerow([Path(case['source_run']).name]+[entry[k] for k in (
                    'profile','repeat','effective_seed_limit','full_route_pose_variants',
                    'full_route_original_candidates','seconds','ik_calls','ik_seed_trials')])


if __name__=='__main__':
    try:
        main()
    finally:
        close_planning_process()
