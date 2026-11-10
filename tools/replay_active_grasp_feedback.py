#!/usr/bin/env python3
"""Replay measured-first proposals and grasp evidence without robot/camera access.

The G single-view captures contain center-only MGPC/GenX outputs and a separately
archived initial view. Replay withholds that view's target/depth until the update.
It tests evidence updates, NOT whether NBV selected that archived viewpoint.
Saved mode reuses physical completion/candidate outputs, never claims new neural
inference on the mixed input. Use --inference live for fresh CUDA inference.
"""

import argparse
import contextlib
import copy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import shutil
import socket
import sys
import time
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, nargs='+', required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--inference', choices=['saved', 'live'], default='saved')
    parser.add_argument('--candidate-limit', type=int, default=6)
    parser.add_argument('--refine-budget-s', type=float, default=15.)
    opts = parser.parse_args()
    if not 1 <= opts.candidate_limit <= 40 or opts.refine_budget_s <= 0:
        parser.error('Invalid candidate/refinement budget')
    if opts.output_dir.exists():
        parser.error('Use a new output directory to avoid overwriting an earlier replay')
    opts.output_dir.mkdir(parents=True)
    # Parent and numerical subprocesses must not be able to create CAN sockets.
    original_socket = socket.socket
    class NoCanSocket(original_socket):
        def __init__(self, family=socket.AF_INET, *a, **kw):
            if family == getattr(socket, 'AF_CAN', -1):
                raise RuntimeError('CAN is forbidden in saved feedback replay')
            super().__init__(family, *a, **kw)
    socket.socket = NoCanSocket
    from tools.compare_mgpc_graspnet import NumericalArm, baseline_imports, cloud
    baseline_imports()
    from real_nero_grasp import runtime as runtime
    from real_nero_grasp.grasp_workflow import plan_and_maybe_execute_grasp
    from real_nero_grasp.active_grasp_feedback import ActiveGraspFeedback, validate_feedback_args
    from real_nero_grasp.perception_evidence import transform
    from real_nero_grasp.mgpc_completion import complete_graspnet_target
    from real_nero_grasp.graspgenx_integration import generate_graspgenx_candidates
    from real_nero_grasp.planning_process import close_planning_process
    import cv2
    summary = dict(hardware_connected=False, motion_commands_sent=False, inference=opts.inference,
                   fresh_model_inference=opts.inference == 'live', cases=[],
                   source_sha256={name: digest(ROOT/name) for name in (
                       'real_nero_grasp/perception_evidence.py', 'real_nero_grasp/active_grasp_feedback.py',
                       'real_nero_grasp/active_view.py', 'real_nero_grasp/grasp_workflow.py',
                       'real_nero_grasp/workflow.py', 'real_nero_grasp/mgpc_completion.py',
                       'real_nero_grasp/cli.py', 'tools/replay_active_grasp_feedback.py')},
                   limitations=[
                       'Withheld initial view tests one evidence update; it is not the selected NBV or a two-side experiment.',
                       'Saved mode reuses center-only physical model outputs, not neural inference on new mixed/fused inputs.',
                       'Measured contact, CAD and full-route checks do not validate friction or physical retention.',
                       'Historical D435 data do not establish current Orbbec performance.'])
    for run in opts.run_dir:
        run = run.resolve()
        record = json.loads((run/'experiment_result.json').read_text())
        if record['method']['grasp_generator'] != 'graspgenx' or not record['method']['mgpc']:
            parser.error('Replay requires an MGPC+GenX source capture')
        if record['views']['used_for_final_reconstruction'] != 1:
            parser.error('Use center-only completion; multiview prediction would leak held-out geometry')
        capture = Path(record['artifacts']['primary_capture_dir'])
        archives = {json.loads(p.read_text())['view_label']: p.parent for p in capture.glob('multiview_raw/*/metadata.json')}
        if not {'center', 'initial'} <= archives.keys():
            parser.error('Need center and independent initial RGB-D archives')
        output = opts.output_dir / run.name
        output.mkdir()
        visible = output/'observations/multiview_raw'
        visible.mkdir(parents=True)
        for name in ('center',):
            (visible/name).symlink_to(archives[name], target_is_directory=True)
        table = json.loads((capture/'table_reference/table_reference_report.json').read_text())
        info = json.loads((capture/'mgpc/completion.json').read_text())
        T = np.asarray(info['T_base_camera'])
        source_prediction = np.load(capture/'mgpc/prediction_base.npy', allow_pickle=False)
        source_candidates = json.loads((capture/'graspgenx/candidates.json').read_text())
        center_report = next(v for v in table['views'] if v['view'] == 'center')
        initial_report = next(v for v in table['views'] if v['view'] == 'initial')
        center_meta = json.loads((archives['center']/'metadata.json').read_text())
        with np.load(archives['center']/'clouds.npz', allow_pickle=False) as arrays:
            center_scene = transform(arrays['scene_base_points'], center_report['T_registration'])
            center_obstacles = transform(arrays['obstacle_base_points'], center_report['T_registration'])
        measured = np.load(capture/'mgpc/measured_camera.npy', allow_pickle=False)
        measured_base = transform(measured, T)
        sys.argv = record['command']
        args = runtime.parse_args()
        args.execute = False
        args.active_grasp_feedback = True
        args.feedback_candidate_limit = opts.candidate_limit
        args.grasp_required_view_count = 0
        validate_feedback_args(args)
        args.nero_refine_timeout_s = opts.refine_budget_s
        args.grasp_preflight_max_candidates = opts.candidate_limit
        args.grasp_target_valid_plans = 1
        args.select_grasp_candidate = False
        args.preview_graspnet_cloud = args.preview_final_grasp = False
        args.save_dir = str(output)
        args._empty_grasp_recovery_joints = np.deg2rad(args.initial_joints_deg)
        arm = NumericalArm(runtime, args, np.asarray(center_meta['capture_joints']))
        arm.get_flange_pose = lambda: arm.get_tcp2flange_pose(arm.get_tcp_pose())
        session = ActiveGraspFeedback(args)
        args._active_feedback_session = session
        current = dict(target=cloud(measured), scene=cloud(transform(center_scene, np.linalg.inv(T))),
                       environment=runtime.fuse_dynamic_environment_clouds(
                           ['center'], [cloud(center_obstacles)], args, target_base=measured_base,
                           target_center_base=np.median(measured_base, axis=0), map_id='feedback_replay_center', generation=1))
        def table_hook(probe_args, target, scene, camera, directory):
            path = Path(directory)/'table_reference'
            path.mkdir(exist_ok=True)
            # Only center geometry is available before update. Retain its
            # independent table correction, not the held-out target/depth.
            visible_table = dict(table, views=[center_report])
            (path/'table_reference_report.json').write_text(json.dumps(visible_table))
            probe_args._grasp_table_reference_report = str(path/'table_reference_report.json')
            return current['target'], current['scene'], T, current['scene'], None, current['environment']
        def probe(_arm, probe_args, *a, **kw):
            hooks = replace(runtime._grasp_workflow_hooks(probe_args), prepare_table_referenced_scene=table_hook,
                            preview_graspnet_pointcloud=lambda *a: None, preview_final_grasp_candidate=lambda *a: True,
                            print_status=lambda *a: None, release_vision_model_caches=lambda: None)
            return plan_and_maybe_execute_grasp(_arm, probe_args, *a, **kw, hooks=hooks)
        session.probe_runner = probe
        def saved_completion(command, **kw):
            request_path = Path(command[-1])
            req = json.loads(request_path.read_text())
            reference = transform(source_prediction, np.linalg.inv(req['T_base_reference']))
            norm = req['normalization']
            result = (reference - norm['centroid_camera_m']) / norm['bbox_diagonal_m']
            np.save(request_path.parent/'prediction_normalized.npy', result)
        def completion_adapter(target, camera, bgr, probe_args, directory, **kw):
            value = complete_graspnet_target(target, camera, bgr, probe_args, directory,
                                             runner=saved_completion, **kw)
            path = Path(directory)/'mgpc/completion.json'
            report = json.loads(path.read_text())
            report['inference_provenance'] = dict(fresh_inference=False,
                method='cached physical completion renormalized for current request; not new model inference',
                source=str(capture/'mgpc/completion.json'))
            path.write_text(json.dumps(report, indent=2)+'\n')
            return value
        def saved_genx(command, **kw):
            req = json.loads(Path(command[-1]).read_text())
            payload = dict(source_candidates, replay_provenance=dict(fresh_inference=False,
                source=str(capture/'graspgenx/candidates.json')))
            Path(req['output_json']).write_text(json.dumps(payload))
        def genx_adapter(*a, **kw):
            return generate_graspgenx_candidates(*a, runner=saved_genx, **kw)
        contexts = contextlib.ExitStack()
        if opts.inference == 'saved':
            contexts.enter_context(mock.patch('real_nero_grasp.mgpc_completion.complete_graspnet_target', side_effect=completion_adapter))
            contexts.enter_context(mock.patch('real_nero_grasp.graspgenx_integration.generate_graspgenx_candidates', side_effect=genx_adapter))
        print(f'Replaying {run.name}: center then withheld initial view', flush=True)
        start = time.monotonic()
        try:
            with contexts, (output/'replay.log').open('w') as log, contextlib.redirect_stdout(log):
                session.update(arm, current['target'], current['scene'], T, None, None, {'label': 'banana'},
                               output/'observations', output/'center')
                center_ready = session.ready
                memory_before = [r['row']['feedback_memory_id'] for r in session.memory]
                # The held-out view becomes available only here. It was not in
                # the model/evidence input of the center evaluation above.
                side = visible/'side1'
                side.mkdir()
                meta = json.loads((archives['initial']/'metadata.json').read_text())
                meta['view_label'] = 'side1'
                meta['T_base_camera'] = (np.asarray(initial_report['T_registration']) @ np.asarray(meta['T_base_camera'])).tolist()
                meta['status'] = dict(capture_quality_keep=True)
                (side/'metadata.json').write_text(json.dumps(meta))
                for name in ('color.png', 'depth_m.npy', 'selected_target_mask.png'):
                    (side/name).symlink_to(archives['initial']/name)
                # read_observations applies center correction to each raw pose.
                meta['T_base_camera'] = (np.linalg.inv(center_report['T_registration']) @ np.asarray(meta['T_base_camera'])).tolist()
                (side/'metadata.json').write_text(json.dumps(meta))
                with np.load(archives['initial']/'clouds.npz', allow_pickle=False) as arrays:
                    additional = transform(arrays['target_base_points'], initial_report['T_registration'])
                    additional_scene = transform(arrays['scene_base_points'], initial_report['T_registration'])
                    additional_obstacles = transform(arrays['obstacle_base_points'], initial_report['T_registration'])
                # Apply the same production target-registration gate as live
                # updates, rather than treating table alignment as target truth.
                update_dir = output/'after_update'
                update_dir.mkdir()
                registered, _, keep = runtime.register_active_fusion_views(
                    ['center', 'side1'], [cloud(measured_base), cloud(additional)],
                    [cloud(center_scene), cloud(additional_scene)], T, str(update_dir), args)
                registration_path = update_dir/'active_fusion_registration.json'
                registration = json.loads(registration_path.read_text())
                C = np.asarray(center_report['T_registration'])
                for item in registration['views']:
                    item['target_transform'] = (np.linalg.inv(C) @ np.asarray(item['target_transform']) @ C).tolist()
                registration_path.write_text(json.dumps(registration))
                fused, fusion_stats = runtime.fuse_target_clouds(
                    [value for value, accepted in zip(registered, keep) if accepted], args.active_fusion_voxel, return_stats=True)
                fused_base = np.asarray(fused.points)
                current['target'] = cloud(transform(fused_base, np.linalg.inv(T)))
                current['scene'] = cloud(transform(np.vstack([center_scene, additional_scene]), np.linalg.inv(T)))
                current['environment'] = runtime.fuse_dynamic_environment_clouds(
                    ['center', 'heldout'], [cloud(center_obstacles), cloud(additional_obstacles)], args,
                    target_base=fused_base, target_center_base=np.median(fused_base, axis=0),
                    map_id='feedback_replay_update', generation=2)
                session.accepted_names = [name for name, accepted in zip(['center', 'side1'], keep) if accepted]
                session.attempts = 1
                session.update(arm, current['target'], current['scene'], T, None, None, {'label': 'banana'},
                               output/'observations', output/'after_update')
                session.write_summary(output)
        finally:
            close_planning_process()
        case = dict(source_run=str(run), source_completion_sha256=digest(capture/'mgpc/completion.json'),
                    source_candidates_sha256=digest(capture/'graspgenx/candidates.json'),
                    center_ready=center_ready, after_update_ready=session.ready,
                    center_memory_ids=memory_before, events=session.events,
                    fusion_stats=fusion_stats, wall_s=time.monotonic()-start,
                    full_route_checks='production preflight and Nero refinement on CPU',
                    withheld_view='archived initial; additional viewpoint relative to centered source',
                    observation_attempts=session.attempts, valid_sides=session.valid_sides)
        summary['cases'].append(case)
        (opts.output_dir/'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
        print(f'  center_ready={center_ready}, after_update_ready={session.ready}; {case["wall_s"]:.1f}s', flush=True)


if __name__ == '__main__':
    main()
