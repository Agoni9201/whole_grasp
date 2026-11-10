#!/usr/bin/env python3
"""Fresh GenX input-policy ablation on center-only measured/MGPC geometry.

Requires a completed --inference live feedback replay. A single freshly inferred
MGPC prediction is shared by the input policies so only proposal preparation
changes. EVERY GenX pool is inferred afresh; there is no candidate-cache mode.
No camera/robot object is created and CAN sockets are forbidden.
"""

import argparse
import contextlib
import copy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import socket
import sys
import time
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--live-replay', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44])
    parser.add_argument('--candidate-limit', type=int, default=6)
    parser.add_argument('--refine-budget-s', type=float, default=15.)
    opts = parser.parse_args()
    if opts.output_dir.exists():
        parser.error('Use a new output directory')
    if not 1 <= opts.candidate_limit <= 40 or opts.refine_budget_s <= 0:
        parser.error('Invalid planning budget')
    source = json.loads((opts.live_replay / 'summary.json').read_text())
    if source['inference'] != 'live' or not source['fresh_model_inference']:
        parser.error('A successful fresh-MGPC live replay is required')
    original_socket = socket.socket

    class NoCanSocket(original_socket):
        def __init__(self, family=socket.AF_INET, *a, **kw):
            if family == getattr(socket, 'AF_CAN', -1):
                raise RuntimeError('CAN forbidden in offline input comparison')
            super().__init__(family, *a, **kw)

    socket.socket = NoCanSocket
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('Fresh GenX inference needs an accessible GPU; no cached fallback')
    from tools.compare_mgpc_graspnet import NumericalArm, baseline_imports, cloud
    baseline_imports()
    from real_nero_grasp import runtime
    from real_nero_grasp.grasp_workflow import plan_and_maybe_execute_grasp, _prepare_graspnet_target_cloud
    from real_nero_grasp.active_grasp_feedback import ActiveGraspFeedback, FeedbackPlanningSnapshot, validate_feedback_args
    from real_nero_grasp.perception_evidence import transform
    from real_nero_grasp.planning_process import close_planning_process

    opts.output_dir.mkdir(parents=True)
    report = dict(hardware_connected=False, motion_commands_sent=False,
        genx_inference='fresh for every policy/seed; no cached candidates',
        mgpc_inference='shared fresh center-only live prediction for each recorded scene',
        source_replay=str(opts.live_replay.resolve()), source_replay_sha256=sha256(opts.live_replay/'summary.json'),
        device=torch.cuda.get_device_name(), seeds=opts.seeds,
        candidate_limit=opts.candidate_limit, target_valid_plans=1, refinement_budget_s=opts.refine_budget_s,
        num_grasps=64, topk_per_pass=20,
        comparison='same measured target/environment/q; independent pools, no old candidate memory',
        limitations=[
            'Two historical D435 scenes, not current Orbbec or physical retention.',
            'Center-only input ablation is not validation of mandatory side observation or NBV viewpoint selection.',
            'The common extra contact proxy uses the same M/P for all policies; production physical gates use M only.',
            'Complete-route field uses saved grasp-validation-first/support-release protocol; not full delivery.',
            'At most six retained candidates, first qualifying route target and 15 s refinement; failures do not prove no solution.',
            'Measured and full-prediction policies retain original focus; mixed policy preserves measured points.',
            'CUDA seeds control sampling but do not guarantee bitwise deterministic kernels.'],
        source_sha256={str(p):sha256(ROOT/p) for p in map(Path, [
            'tools/compare_genx_proposal_inputs.py', 'tools/run_graspgenx_worker.py',
            'real_nero_grasp/grasp_workflow.py', 'real_nero_grasp/perception_evidence.py',
            'real_nero_grasp/active_grasp_feedback.py', 'real_nero_grasp/graspgenx_integration.py'])},
        results=[])
    for case in source['cases']:
        run = Path(case['source_run'])
        original = json.loads((run/'experiment_result.json').read_text())
        capture = Path(original['artifacts']['primary_capture_dir'])
        live = opts.live_replay/run.name/'center'
        completion = json.loads((live/'mgpc/completion.json').read_text())
        worker = json.loads((live/'mgpc/worker_result.json').read_text())
        assert worker['strict_state_dict'] and worker['points'] == 8192
        measured = np.load(live/'mgpc/measured_camera.npy', allow_pickle=False)
        prediction = np.load(live/'mgpc/prediction_camera.npy', allow_pickle=False)
        with np.load(live/'mgpc/proposal_evidence.npz', allow_pickle=False) as data:
            mixed, labels = data['points_camera'].copy(), data['source'].copy()
        assert np.array_equal(mixed[:len(measured)], measured)
        assert np.all(labels[:len(measured)] == 0) and np.all(labels[len(measured):] == 1)
        T = np.asarray(completion['T_base_camera'])
        measured_base, prediction_base = transform(measured, T), transform(prediction, T)
        table = json.loads((live/'table_reference/table_reference_report.json').read_text())
        center_report = next(v for v in table['views'] if v['view'] == 'center')
        archives = {json.loads(p.read_text())['view_label']:p.parent for p in capture.glob('multiview_raw/*/metadata.json')}
        meta = json.loads((archives['center']/'metadata.json').read_text())
        with np.load(archives['center']/'clouds.npz', allow_pickle=False) as data:
            scene_base = transform(data['scene_base_points'], center_report['T_registration'])
            obstacles = transform(data['obstacle_base_points'], center_report['T_registration'])
        sys.argv = original['command']
        base_args = runtime.parse_args()
        base_args.execute = False
        base_args.active_grasp_feedback = True
        base_args.grasp_required_view_count = 0
        validate_feedback_args(base_args)
        base_args.feedback_candidate_limit = opts.candidate_limit
        base_args.nero_refine_timeout_s = opts.refine_budget_s
        base_args.grasp_preflight_max_candidates = opts.candidate_limit
        base_args.grasp_target_valid_plans = 1
        base_args.graspgenx_num_grasps = 64
        base_args.graspgenx_topk_per_pass = 20
        base_args.select_grasp_candidate = False
        base_args.preview_graspnet_cloud = base_args.preview_final_grasp = False
        base_args.active_side_view_count = 0
        base_args._experiment_log_state = None
        base_args._empty_grasp_recovery_joints = np.deg2rad(base_args.initial_joints_deg)
        scene = cloud(transform(scene_base, np.linalg.inv(T)))
        target = cloud(measured)
        environment = runtime.fuse_dynamic_environment_clouds(['center'], [cloud(obstacles)], base_args,
            target_base=measured_base, target_center_base=np.median(measured_base, axis=0),
            map_id='fresh_genx_input_comparison', generation=1)
        for seed in opts.seeds:
            # Rotate policy order, avoiding systematically assigning warm caches
            # and numerical planner startup to the same policy.
            policies = ['measured', 'prediction', 'mixed']
            shift = opts.seeds.index(seed) % len(policies)
            policies = policies[shift:] + policies[:shift]
            for policy in policies:
                output = opts.output_dir/run.name/f'seed_{seed}'/policy
                output.mkdir(parents=True)
                raw = output/'multiview_raw'
                raw.mkdir()
                (raw/'center').symlink_to(archives['center'].resolve(), target_is_directory=True)
                args = copy.copy(base_args)
                args.graspgenx_seed = seed
                args.save_dir = str(output)
                args._feedback_generate = True
                args._graspnet_target_focus_info = {}
                session = ActiveGraspFeedback(args)
                session.prediction_base = prediction_base.copy()
                args._active_feedback_session = session
                arm = NumericalArm(runtime, args, meta['capture_joints'])
                arm.get_flange_pose = lambda: arm.get_tcp2flange_pose(arm.get_tcp_pose())
                chosen = {'measured':measured, 'prediction':prediction, 'mixed':mixed}[policy]
                proposal = cloud(chosen)
                if policy != 'mixed':
                    proposal = _prepare_graspnet_target_cloud(proposal, args, T_base_camera=T)
                intended = np.asarray(proposal.points).copy()
                np.save(output/'intended_proposal_camera.npy', intended)
                write_json(output/'input_provenance.json', dict(policy=policy, seed=seed,
                    source_completion=str(live/'mgpc'), source_prediction_sha256=sha256(live/'mgpc/prediction_camera.npy'),
                    fresh_mgpc_shared=True, fresh_genx_required=True,
                    measured_points=len(measured), raw_policy_points=len(chosen), actual_input_points=len(intended),
                    focus=getattr(args, '_graspnet_target_focus_info', {}),
                    mandatory_side_observation_satisfied=False))

                def table_hook(probe_args, _target, _scene, _camera, directory):
                    folder = Path(directory)/'table_reference'
                    folder.mkdir(exist_ok=True)
                    path = folder/'table_reference_report.json'
                    write_json(path, table)
                    probe_args._grasp_table_reference_report = str(path)
                    return target, scene, T, scene, None, environment

                hooks = replace(runtime._grasp_workflow_hooks(args), prepare_table_referenced_scene=table_hook,
                    preview_graspnet_pointcloud=lambda *a:None, preview_final_grasp_candidate=lambda *a:True,
                    print_status=lambda *a:None, release_vision_model_caches=lambda:None)
                start = time.monotonic()
                print(f'Fresh GenX {run.name} seed={seed} {policy}: {len(intended)} points', flush=True)
                try:
                    # Only input preparation is injected. The GenX function,
                    # worker/subprocess, checkpoints and numerical checks are real.
                    with mock.patch('real_nero_grasp.mgpc_completion.complete_graspnet_target', return_value=proposal), \
                            (output/'planning.log').open('w') as log, contextlib.redirect_stdout(log):
                        snapshot = plan_and_maybe_execute_grasp(arm, args, target, scene, T, None, None,
                            str(output), {'label':'banana'}, planning_only=True, hooks=hooks,
                            plan_label=f'input_{policy}_{seed}')
                        if not isinstance(snapshot, FeedbackPlanningSnapshot):
                            raise RuntimeError(f'Planning-only did not return rows: {snapshot}')
                        session.assess(snapshot.rows, snapshot.T_base_camera, output)
                finally:
                    close_planning_process()
                request = json.loads((output/'graspgenx/request.json').read_text())
                consumed = np.load(request['input_npy'], allow_pickle=False)
                assert np.array_equal(consumed, intended.astype(np.float32))
                candidates = json.loads((output/'graspgenx/candidates.json').read_text())
                assert 'replay_provenance' not in candidates
                event = session.events[-1]
                result = dict(scene=run.name, policy=policy, seed=seed, directory=str(output),
                    proposal_sha256=sha256(request['input_npy']), candidates_sha256=sha256(output/'graspgenx/candidates.json'),
                    generated_candidates=candidates['candidate_count'], native_table_safe=candidates['table_safe_count'],
                    retained_candidates=len(snapshot.rows),
                    physical_aperture_pass=sum(bool(r.get('physical_aperture_ok')) for r in snapshot.rows),
                    endpoint_reachability_pass=sum(bool(r.get('reachability_ok')) for r in snapshot.rows),
                    route_pass=sum(r['complete_route_pass'] for r in event['candidates']),
                    evidence_sufficient=sum(r['evidence_sufficient'] for r in event['candidates']),
                    both_pass=sum(r['complete_route_pass'] and r['evidence_sufficient'] for r in event['candidates']),
                    wall_s=time.monotonic()-start,
                    grasp_validation_first=bool(args.grasp_validation_first), actual_genx_input_verified=True)
                report['results'].append(result)
                write_json(opts.output_dir/'summary.json', report)
                print(f"  table-safe={result['native_table_safe']}; aperture={result['physical_aperture_pass']}; "
                      f"route={result['route_pass']}; evidence={result['evidence_sufficient']}; both={result['both_pass']}; "
                      f"{result['wall_s']:.1f}s", flush=True)


if __name__ == '__main__':
    main()
