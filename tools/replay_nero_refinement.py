#!/usr/bin/env python3
"""Replay the production GraspNet/GraspGenX workflow with saved proposals.

Only perception/inference/UI are substituted. Geometry, isolated IK, complete
route checks, fixed carrying goal, selection and pose artifacts are production
code. NumericalArm has no driver or motion methods; execution is disabled.
"""

import argparse
import contextlib
import copy
from dataclasses import replace
from datetime import datetime
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.compare_mgpc_graspnet import NumericalArm, baseline_imports, cloud, plain
from real_nero_grasp import runtime as r
from real_nero_grasp.grasp_workflow import plan_and_maybe_execute_grasp
from real_nero_grasp.graspgenx_integration import generate_graspgenx_candidates
from real_nero_grasp.graspnet_process import _unpack
from real_nero_grasp.planning_process import close_planning_process, get_planning_process
from real_nero_grasp.types import DynamicEnvironmentMap
from unittest import mock


class ReplayArm(NumericalArm):
    def get_flange_pose(self):
        return self.get_tcp2flange_pose(self.get_tcp_pose())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, nargs='+', required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    opts = parser.parse_args()
    opts.output_dir.mkdir(parents=True, exist_ok=True)
    baseline_imports()
    summary = dict(hardware_connected=False, motion_commands_sent=False,
                   reused_saved_network_proposals=True, cases=[])
    for run in opts.run_dir:
        record = json.loads((run/'experiment_result.json').read_text())
        sys.argv = record['command'] + ['--nero-grasp-refine']
        args = r.parse_args()
        args.execute = False
        args.preview_final_grasp = args.preview_graspnet_cloud = False
        args.select_grasp_candidate = False
        if args.mgpc:
            parser.error('Replay expects saved non-MGPC proposals; use the measured A/B/E cases')
        root = Path(record['artifacts']['primary_capture_dir'])
        output = opts.output_dir / run.name
        output.mkdir(exist_ok=True)
        args.save_dir = str(output)
        args._empty_grasp_recovery_joints = np.deg2rad(args.initial_joints_deg)
        args.table_z = float(args.grasp_table_reference_z)
        report = json.loads((root/'table_reference/table_reference_report.json').read_text())
        with np.load(root/'table_reference/registered_scene.npz') as data:
            T = data['T_base_camera'].copy()
            scene = cloud(data['scene_points'])
            obstacles = cloud(data['environment_points'], data['environment_colors'])
            metadata = dict(report['environment_map_metadata'])
            metadata.update({key[4:]: data[key] for key in data.files if key.startswith('map_')})
            environment = DynamicEnvironmentMap(**metadata)
        target = r.get_o3d().io.read_point_cloud(str(root/'active_fused_target_camera0.ply'))
        saved_plan = root/'whole_workflow_motion_plan.json'
        if saved_plan.is_file():
            q = np.deg2rad(json.loads(saved_plan.read_text())['start_joints_deg'])
        else:
            start_time = datetime.fromisoformat(next(event['timestamp'] for event in record['stage_events']
                if event['stage'] == 'grasp_planning')).timestamp()
            cycles = [json.loads(line) for line in (run/'mit_control_cycles.jsonl').open()]
            q = np.asarray(next(item['q_actual'] for item in reversed(cycles)
                if item['type'] == 'cycle' and item['wall_time'] <= start_time))
        arm = ReplayArm(r, args, q)
        hooks = replace(r._grasp_workflow_hooks(args),
            prepare_table_referenced_scene=lambda *a: (target, scene, T, scene, obstacles, environment),
            preview_graspnet_pointcloud=lambda *a: None,
            preview_final_grasp_candidate=lambda *a: True, print_status=lambda *a: None,
            release_vision_model_caches=lambda: None)
        if args.grasp_generator == 'graspnet':
            rows = _unpack(json.loads((root/'graspnet_worker/result.json').read_text())['rows'])
            hooks = replace(hooks, get_graspnet_helpers=lambda: (
                lambda *a, **k: ({}, None), lambda *a, **k: (None, copy.deepcopy(rows))))
            generator_context = contextlib.nullcontext()
        else:
            saved = json.loads((root/'graspgenx/candidates.json').read_text())
            def replay_worker(command, **kwargs):
                request = json.loads((output/'graspgenx/request.json').read_text())
                Path(request['output_json']).write_text(json.dumps(saved))
            def native_adapter(*a, **kwargs):
                return generate_graspgenx_candidates(*a, runner=replay_worker, **kwargs)
            generator_context = mock.patch('real_nero_grasp.graspgenx_integration.generate_graspgenx_candidates',
                side_effect=native_adapter)
        print(f'Replaying {args.grasp_generator}: {run.name}', flush=True)
        try:
            with (output/'main_workflow.log').open('w') as log, contextlib.redirect_stdout(log), generator_context:
                result = plan_and_maybe_execute_grasp(arm, args, target, scene, T,
                    None, None, str(output), dict(label='banana'), hooks=hooks)
                worker_state = get_planning_process().request('ping', {})
        finally:
            close_planning_process()
        case = dict(source_run=str(run), generator=args.grasp_generator, result=plain(result.__dict__),
                    numerical_worker=worker_state, output=str(output))
        refinement = output/'nero_refinement.json'
        if refinement.is_file():
            case['refinement'] = json.loads(refinement.read_text())
        summary['cases'].append(case)
        (opts.output_dir/'mainflow_replay.json').write_text(json.dumps(plain(summary), indent=2) + '\n')
        counts = {key: value for key, value in case.get('refinement', {}).items()
                  if key in ('native_full_routes', 'refined_full_routes', 'route_attempts', 'seconds')}
        print(result, 'refinement=', counts, flush=True)


if __name__ == '__main__':
    main()
