#!/usr/bin/env python3
"""Offline comparison of saved GraspNet poses and bounded Nero pose refinement.

No inference, hardware adapter, or execution entrypoint is called. Both methods
reuse identical measured clouds, occupancy, start joints and safety settings.
Refinement is an extra geometric search, not a new network prediction.
"""

import argparse
import contextlib
import copy
import csv
from datetime import datetime
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.compare_mgpc_graspnet import NumericalArm, baseline_imports, cloud, plain
from real_nero_grasp import runtime as r
from real_nero_grasp.grasp_candidates import annotate_grasp_candidate_physical_aperture
from real_nero_grasp.nero_refinement import orientation_variants
from real_nero_grasp.graspgenx_integration import open_gripper_vertices_tcp
from real_nero_grasp.graspnet_process import _unpack
from real_nero_grasp.post_grasp_rrt import plan_post_grasp_joint_hold
from real_nero_grasp.types import DynamicEnvironmentMap, PlanNotExecutable, UnsafeMotionTarget


def load_case(run):
    record = json.loads((run / 'experiment_result.json').read_text())
    sys.argv = record['command']
    args = r.parse_args()
    args.execute = False
    args._planning_worker = True
    if (args.grasp_generator != 'graspnet' or args.mgpc
            or args.tool_approach_axis != 'x' or args.tcp_gripper_width_axis != 'z'
            or args.topdown_grasp or args.fixed_rpy is not None or args.use_current_rpy
            or args.use_graspnet_depth_advance or args.min_grasp_z is not None
            or not np.isclose(min(args.gripper_open_width, args.gripper_hardware_max_width), .1)):
        raise ValueError('Replay requires native GraspNet, measured target and Nero 100 mm opening')
    root = Path(record['artifacts']['primary_capture_dir'])
    required = [root / name for name in ('graspnet_worker/result.json',
        'active_fused_target_camera0.ply', 'active_fused_scene_camera0.ply',
        'table_reference/table_reference_report.json', 'table_reference/registered_scene.npz')]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    rows = _unpack(json.loads(required[0].read_text())['rows'])
    if not rows:
        raise ValueError(f'No saved network candidates: {run}')
    saved_plan = root / 'whole_workflow_motion_plan.json'
    if saved_plan.is_file():
        q = np.deg2rad(json.loads(saved_plan.read_text())['start_joints_deg'])
        q_source = str(saved_plan)
    else:
        start_time = datetime.fromisoformat(next(event['timestamp']
            for event in record['stage_events'] if event['stage'] == 'grasp_planning')).timestamp()
        before = [json.loads(line) for line in (run / 'mit_control_cycles.jsonl').open()]
        q = np.asarray(next(item['q_actual'] for item in reversed(before)
            if item['type'] == 'cycle' and item['wall_time'] <= start_time))
        q_source = 'last measured control cycle before grasp_planning'
    report = json.loads(required[3].read_text())
    with np.load(required[4]) as data:
        T = data['T_base_camera'].copy()
        obstacle = cloud(data['environment_points'], data['environment_colors'])
        metadata = dict(report['environment_map_metadata'])
        metadata.update({key[4:]: data[key] for key in data.files if key.startswith('map_')})
        environment = DynamicEnvironmentMap(**metadata)
    target = r.get_o3d().io.read_point_cloud(str(required[1]))
    scene = r.get_o3d().io.read_point_cloud(str(required[2]))
    return dict(args=args, rows=rows, q=q, q_source=q_source, T=T,
                target=target, scene=scene, obstacle=obstacle, environment=environment)


def rotations(original, args, T):
    base_grasp = T[:3, :3] @ original['grasp'].rotation_matrix
    base_tcp = r.graspnet_rotation_to_tcp_rotation(base_grasp, args)
    yield from orientation_variants(base_tcp)


def geometry_proposals(original, case, max_depth_mm):
    args, T = case['args'], case['T']
    vertices = open_gripper_vertices_tcp(args)
    mapping = r.graspnet_rotation_to_tcp_rotation(np.eye(3), args)
    candidates = []
    samples = 0
    for name, tcp_rotation, angle in rotations(original, args, T):
        row = copy.deepcopy(original)
        row['grasp'].rotation_matrix = T[:3, :3].T @ tcp_rotation @ mapping.T
        cfg = copy.copy(args)
        for millimeters in range(max_depth_mm + 1):
            cfg.grasp_advance = millimeters / 1000
            probe = annotate_grasp_candidate_physical_aperture(
                [row], case['target'], T, None, cfg, verbose=False)[0]
            samples += 1
            if not probe['physical_aperture_ok']:
                continue
            variant = next(v for v in probe['physical_aperture_variants'] if v['ok'])
            pre, grasp, _ = r.grasp_to_tcp_poses(probe['grasp'], T, cfg,
                width_sign=variant['width_sign'], width_shift=variant['tcp_width_shift'])
            clearance = min(float(np.min(vertices @ pose[2, :3]) + pose[2, 3])
                for pose in (r._make_T_from_pose(pre), r._make_T_from_pose(grasp))) - args.table_z
            if clearance < args.motion_path_min_clearance:
                continue
            candidates.append(dict(row=row, family=name, angle_change_deg=angle,
                advance_m=cfg.grasp_advance, cad_table_clearance_m=clearance,
                left_contact_m=variant['left_contact_inset'], right_contact_m=variant['right_contact_inset']))
            break
    return candidates, samples


def check_route(row, case, advance, label, path):
    cfg = copy.copy(case['args'])
    cfg.grasp_advance = float(advance)
    cfg.grasp_preflight_max_candidates = 1
    cfg.grasp_target_valid_plans = 1
    cfg.save_dir = str(path.parent)
    arm = NumericalArm(r, cfg, case['q'])
    started = time.perf_counter()
    with path.open('w') as stream, contextlib.redirect_stdout(stream):
        rows = r.annotate_grasp_candidate_physical_aperture(
            [copy.deepcopy(row)], case['target'], case['T'], arm.get_tcp_pose(), cfg)
        rows = r.annotate_grasp_candidate_reachability(
            rows, case['T'], arm.get_tcp_pose(), cfg, arm=arm)
        rows = r.annotate_grasp_candidate_workflow_preflight(rows, arm, cfg,
            case['target'], case['scene'], case['T'], arm.get_tcp_pose(), label,
            motion_environment_cloud=case['obstacle'], motion_environment_map=case['environment'])
        selected = rows[0]
        pickup_ok = bool(selected['workflow_preflight_ok'])
        hold = []
        reason = selected['workflow_preflight_reason']
        if pickup_ok:
            plan = selected['_workflow_motion_plan']
            context = r._motion_context(plan.safety_scene, 'carrying',
                attached_target_tcp=plan.attached_target_tcp, closed_gripper_width=plan.closed_gripper_width)
            try:
                hold = plan_post_grasp_joint_hold(plan.after_close[-1].joints, context, cfg, label + '_hold')
            except (PlanNotExecutable, UnsafeMotionTarget) as exc:
                reason = 'requested joint hold failed: ' + str(exc)
        full_ok = pickup_ok and bool(hold)
        return dict(pickup_route_ok=pickup_ok, full_route_ok=full_ok, reason=reason,
            aperture_ok=bool(selected['physical_aperture_ok']), seconds=time.perf_counter() - started,
            advance_m=advance, log=str(path), row=plain(selected),
            planned_stage_counts=dict(before_close=0 if not pickup_ok else len(plan.before_close),
                support_and_lift=0 if not pickup_ok else len(plan.after_close),
                empty_recovery=0 if not pickup_ok else len(plan.empty_recovery),
                requested_joint_hold=len(hold)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, nargs='+', required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--max-route-variants', type=int, default=8)
    parser.add_argument('--max-depth-mm', type=int, default=30)
    opts = parser.parse_args()
    if opts.max_route_variants < 1 or not 0 <= opts.max_depth_mm <= 30:
        parser.error('Positive route budget and 0..30 mm depth envelope required')
    opts.output_dir.mkdir(parents=True, exist_ok=True)
    baseline_imports()
    result = dict(hardware_connected=False, motion_commands_sent=False,
        reused_saved_network_candidates=True, changed_live_pipeline=False,
        methods=dict(baseline='saved pose and fixed 15 mm advance',
            refined='retain baseline successes, then bounded jaw-leveling/tilt/flip and 0..30 mm depth search'),
        physical_opening_mm=100, max_route_variants=opts.max_route_variants,
        max_depth_mm=opts.max_depth_mm, cases=[])
    for run in opts.run_dir:
        case = load_case(run)
        output = opts.output_dir / run.name
        output.mkdir(exist_ok=True)
        report = dict(source_run=str(run), start_joint_source=case['q_source'],
            start_joints_deg=np.rad2deg(case['q']).tolist(), original_candidates=len(case['rows']), candidates=[])
        result['cases'].append(report)
        started = time.perf_counter()
        for original in case['rows']:
            index = int(original['idx'])
            baseline = check_route(original, case, case['args'].grasp_advance,
                f'base_{index}', output / f'base_{index}.log')
            entry = dict(candidate=index, baseline=baseline, attempts=[], geometry_variants=0, geometry_samples=0)
            report['candidates'].append(entry)
            if baseline['full_route_ok']:
                entry.update(refined_full_route_ok=True, reused_baseline=True)
            else:
                proposals, samples = geometry_proposals(original, case, opts.max_depth_mm)
                entry.update(geometry_variants=len(proposals), geometry_samples=samples,
                    refined_full_route_ok=False, reused_baseline=False)
                for proposal in proposals[:opts.max_route_variants]:
                    if (proposal['family'] == 'native'
                            and np.isclose(proposal['advance_m'], case['args'].grasp_advance)):
                        continue
                    number = len(entry['attempts'])
                    checked = check_route(proposal['row'], case, proposal['advance_m'],
                        f'refine_{index}_{number}', output / f'refine_{index}_{number}.log')
                    checked.update({key: value for key, value in proposal.items() if key != 'row'})
                    entry['attempts'].append(checked)
                    if checked['full_route_ok']:
                        entry['refined_full_route_ok'] = True
                        break
            print(f'{run.name} candidate={index}: baseline={baseline["full_route_ok"]} '
                  f'refined={entry["refined_full_route_ok"]} route_attempts={len(entry["attempts"])}', flush=True)
            (opts.output_dir / 'comparison.json').write_text(json.dumps(plain(result), indent=2, allow_nan=False) + '\n')
        report['wall_seconds'] = time.perf_counter() - started
        report['summary'] = dict(baseline_contact=sum(c['baseline']['aperture_ok'] for c in report['candidates']),
            baseline_pickup_routes=sum(c['baseline']['pickup_route_ok'] for c in report['candidates']),
            baseline_full_routes=sum(c['baseline']['full_route_ok'] for c in report['candidates']),
            refined_full_routes=sum(c['refined_full_route_ok'] for c in report['candidates']),
            extra_route_attempts=sum(len(c['attempts']) for c in report['candidates']),
            geometry_samples=sum(c['geometry_samples'] for c in report['candidates']))
        print('CASE RESULT', run.name, report['summary'], flush=True)
    (opts.output_dir / 'comparison.json').write_text(json.dumps(plain(result), indent=2, allow_nan=False) + '\n')
    with (opts.output_dir / 'comparison.csv').open('w') as stream:
        writer = csv.writer(stream)
        writer.writerow(['run', 'original_candidates', 'baseline_contact', 'baseline_full_routes',
                         'refined_full_routes', 'extra_route_attempts', 'geometry_samples', 'wall_seconds'])
        for case in result['cases']:
            summary = case['summary']
            writer.writerow([case['source_run'], case['original_candidates'], summary['baseline_contact'],
                summary['baseline_full_routes'], summary['refined_full_routes'], summary['extra_route_attempts'],
                summary['geometry_samples'], case['wall_seconds']])


if __name__ == '__main__':
    main()
