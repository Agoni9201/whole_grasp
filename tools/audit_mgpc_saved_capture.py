#!/usr/bin/env python3
"""Compare measured/MGPC proposals on saved captures; never connect to hardware.

Reuse each capture's deployed GraspNet options and seed. Re-run measured-input
inference, compare to its saved MGPC proposals, and apply the same measured
aperture and full open-gripper mesh/table checks. This does not certify IK,
whole-workflow feasibility, or physical grasp success. --no-focus instead
re-infers both saved clouds without the upper-focus crop.
"""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from real_nero_grasp import runtime as r
from real_nero_grasp.grasp_workflow import _prepare_graspnet_target_cloud
from real_nero_grasp.graspgenx_integration import selected_mesh_table_clearance
from real_nero_grasp.graspnet_process import _unpack, generate_candidates
from tools.compare_mgpc_graspnet import cloud, plain


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, nargs='+', required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--no-focus', action='store_true',
        help='Re-infer both inputs with upper-focus disabled; keep all safety checks')
    opts = parser.parse_args()
    runs = []
    for run in opts.run_dir:
        record = json.loads((run / 'experiment_result.json').read_text())
        saved_argv = sys.argv
        try:
            sys.argv = record['command']
            args = r.parse_args()
        finally:
            sys.argv = saved_argv
        if args.grasp_generator != 'graspnet' or not args.mgpc:
            parser.error(f'{run}: expected a saved MGPC -> GraspNet run')
        root = Path(record['artifacts']['primary_capture_dir'])
        request = json.loads((root / 'graspnet_worker/request.json').read_text())
        result = json.loads((root / 'graspnet_worker/result.json').read_text())
        if result['seed'] != request['seed'] or not result['ok']:
            parser.error(f'{run}: saved inference was not completed with its requested seed')
        args.execute = False
        args.graspnet_seed = request['seed']
        if opts.no_focus:
            args.graspnet_upper_focus = False
        args.graspgenx_table_margin = max(args.graspgenx_table_margin, args.motion_path_min_clearance)
        runs.append((run, root, args, request, result))

    opts.output_dir.mkdir(parents=True, exist_ok=True)
    overview = []
    for run, root, args, request, result in runs:
        destination = opts.output_dir / run.name
        destination.mkdir(exist_ok=True)
        with np.load(root / 'graspnet_worker/input.npz', allow_pickle=False) as data:
            measured = cloud(data['measured_points'], data['measured_colors'])
            scene = cloud(data['scene_points'], data['scene_colors'])
            saved_proposal_count = len(data['proposal_points'])
        with np.load(root / 'table_reference/registered_scene.npz', allow_pickle=False) as data:
            T = data['T_base_camera'].copy()
        capture = json.loads(next((root / 'multiview_raw').glob('center*/metadata.json')).read_text())
        current = r._modeled_tcp_pose_for_joints(capture['capture_joints'], args)
        # The saved MGPC proposals already include production cleanup/focus;
        # apply that same production preprocessing once to the measured input.
        proposal = _prepare_graspnet_target_cloud(measured, args, T_base_camera=T)
        _, measured_rows = generate_candidates(proposal, scene, args, destination / 'measured',
            _unpack(request['options']), measured_target=measured)
        if opts.no_focus:
            predicted_base = np.load(root / 'mgpc/prediction_base.npy', allow_pickle=False)
            inverse = np.linalg.inv(T)
            predicted = cloud(predicted_base @ inverse[:3, :3].T + inverse[:3, 3])
            predicted = _prepare_graspnet_target_cloud(predicted, args, T_base_camera=T)
            _, mgpc_rows = generate_candidates(predicted, scene, args, destination / 'mgpc',
                _unpack(request['options']), measured_target=measured)
            prediction_count = len(predicted.points)
        else:
            mgpc_rows, prediction_count = _unpack(result['rows']), saved_proposal_count
        cases = [('measured', measured_rows, len(proposal.points)),
                 ('mgpc', mgpc_rows, prediction_count)]
        summaries = []
        for name, rows, count in cases:
            rows = r.annotate_grasp_candidate_physical_aperture(rows, measured, T, current, args)
            checks = []
            for row in rows:
                entry = dict(idx=int(row['idx']), aperture_ok=row['physical_aperture_ok'],
                    aperture_reason=row['physical_aperture_reason'], mesh_table_checks=[])
                for variant in row['physical_aperture_variants']:
                    if not variant['ok']:
                        continue
                    pre, grasp, _ = r.grasp_to_tcp_poses(row['grasp'], T, args, current,
                        width_sign=variant['width_sign'], width_shift=variant['tcp_width_shift'])
                    # This is the production mesh check; preserve its refusal.
                    try:
                        clearance = selected_mesh_table_clearance(pre, grasp, args)
                    except RuntimeError as exc:
                        entry['mesh_table_checks'].append(dict(ok=False, reason=str(exc)))
                    else:
                        entry['mesh_table_checks'].append(dict(ok=True, clearance_mm=clearance * 1000))
                checks.append(entry)
            summary = dict(case=name, network_input_points=count, proposals=len(rows),
                aperture_pass=sum(item['aperture_ok'] for item in checks),
                aperture_and_mesh_pass=sum(any(v['ok'] for v in item['mesh_table_checks']) for item in checks))
            summaries.append(summary)
            (destination / f'{name}_checks.json').write_text(
                json.dumps(plain(dict(summary=summary, candidates=checks)), indent=2) + '\n')
            print(f"{run.name} {name}: proposals={summary['proposals']} "
                f"aperture={summary['aperture_pass']} mesh={summary['aperture_and_mesh_pass']}", flush=True)
        overview.append(dict(source_run=str(run), seed=request['seed'], cases=summaries))
    output = dict(hardware_connected=False, motion_commands_sent=False, same_capture=True,
        same_seed_per_pair=True, safety_geometry='saved measured target and registered tabletop',
        upper_focus_override='disabled for both inputs' if opts.no_focus else None,
        mgpc_inference='re-infer saved completed cloud without focus' if opts.no_focus
            else 'reuse saved output; no new completion',
        limitations='Single seed per capture. CUDA need not be bitwise deterministic. '
            'Aperture/mesh only; IK, path, fixed joint hold and physical grasp are not verified.',
        runs=overview)
    path = opts.output_dir / 'comparison.json'
    path.write_text(json.dumps(plain(output), ensure_ascii=False, indent=2) + '\n')
    print(f'Offline comparison saved: {path}', flush=True)


if __name__ == '__main__':
    main()
