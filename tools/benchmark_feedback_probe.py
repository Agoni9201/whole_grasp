#!/usr/bin/env python3
"""Compare route-probe budgets using one frozen, genuinely inferred capture.

No neural inference: this isolates CPU planning cost on identical measured
geometry, saved MGPC/GenX outputs, robot q and obstacle map. No device objects.
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

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture-dir',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    opts=parser.parse_args()
    if opts.output_dir.exists():parser.error('Use a new output directory')
    source=opts.capture_dir.resolve()
    original=json.loads((source.parent/'experiment_result.json').read_text())
    frozen=source/'feedback/round_1'
    original_socket=socket.socket
    class NoCanSocket(original_socket):
        def __init__(self,family=socket.AF_INET,*a,**kw):
            if family==getattr(socket,'AF_CAN',-1):raise RuntimeError('CAN forbidden in probe benchmark')
            super().__init__(family,*a,**kw)
    socket.socket=NoCanSocket
    from tools.compare_mgpc_graspnet import NumericalArm,baseline_imports,cloud
    baseline_imports()
    from real_nero_grasp import runtime
    from real_nero_grasp.grasp_workflow import plan_and_maybe_execute_grasp
    from real_nero_grasp.active_grasp_feedback import ActiveGraspFeedback
    from real_nero_grasp.graspgenx_integration import generate_graspgenx_candidates
    from real_nero_grasp.types import DynamicEnvironmentMap
    from real_nero_grasp.planning_process import close_planning_process
    sys.argv=original['command']
    base=runtime.parse_args()
    base.execute=False
    base._experiment_log_state=None
    completion=json.loads((frozen/'mgpc/completion.json').read_text())
    T=np.asarray(completion['T_base_camera'])
    target=cloud(np.load(frozen/'mgpc/measured_camera.npy',allow_pickle=False))
    proposal=cloud(np.load(frozen/'graspgenx/proposal_camera.npy',allow_pickle=False))
    prediction=np.load(frozen/'mgpc/prediction_base.npy',allow_pickle=False)
    payload=json.loads((frozen/'graspgenx/candidates.json').read_text())
    table=json.loads((frozen/'table_reference/table_reference_report.json').read_text())
    with np.load(frozen/'table_reference/registered_scene.npz',allow_pickle=False) as data:
        scene=cloud(data['scene_points'],data['scene_colors'])
        environment_cloud=cloud(data['environment_points'],data['environment_colors'])
        metadata=copy.deepcopy(table['environment_map_metadata'])
        metadata.update({key[4:]:data[key].copy() for key in data.files if key.startswith('map_')})
    environment=DynamicEnvironmentMap(**metadata)
    center=next(p for p in (source/'multiview_raw').glob('*/metadata.json')
                if json.loads(p.read_text())['view_label']=='center')
    q=np.asarray(json.loads(center.read_text())['capture_joints'])
    opts.output_dir.mkdir(parents=True)
    results=dict(hardware_connected=False,motion_commands_sent=False,fresh_neural_inference=False,
        source_capture=str(source),method='Frozen real MGPC/GenX outputs; identical q, target and canonical obstacle map; numerical route checks only',
        source_sha256={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in
            ['real_nero_grasp/active_grasp_feedback.py','real_nero_grasp/active_view.py',
             'real_nero_grasp/grasp_workflow.py','real_nero_grasp/cli.py','tools/benchmark_feedback_probe.py']},
        limitations=['No real-time MIT load in this offline benchmark.',
                     'Budgets checked between attempts; a single route check can overrun the refinement budget.',
                     'A smaller probe can miss useful candidates; final execution still uses the original budgets.'],cases=[])
    for name in ['old_probe_budget','bounded_probe']:
        directory=opts.output_dir/name
        args=copy.copy(base)
        args.save_dir=str(directory)
        if name=='old_probe_budget':
            args.feedback_probe_preflight_max_candidates=args.grasp_preflight_max_candidates
            args.feedback_probe_refine_timeout_s=args.nero_refine_timeout_s
        session=ActiveGraspFeedback(args)
        session.prediction_base=prediction.copy()
        args._active_feedback_session=session
        arm=NumericalArm(runtime,args,q)
        arm.get_flange_pose=lambda:arm.get_tcp2flange_pose(arm.get_tcp_pose())
        def prepare(probe_args,_target,_scene,_camera,folder):
            destination=Path(folder)/'table_reference'
            destination.mkdir(exist_ok=True)
            path=destination/'table_reference_report.json'
            path.write_text(json.dumps(table))
            probe_args._grasp_table_reference_report=str(path)
            return target,scene,T,scene,environment_cloud,environment
        def probe(_arm,probe_args,*a,**kw):
            if name=='old_probe_budget':probe_args.grasp_target_valid_plans=base.grasp_target_valid_plans
            hooks=replace(runtime._grasp_workflow_hooks(probe_args),prepare_table_referenced_scene=prepare,
                print_status=lambda *a:None,release_vision_model_caches=lambda:None,
                preview_graspnet_pointcloud=lambda *a:None,preview_final_grasp_candidate=lambda *a:True)
            return plan_and_maybe_execute_grasp(_arm,probe_args,*a,**kw,hooks=hooks)
        session.probe_runner=probe
        def worker(command,**kw):
            request=json.loads(Path(command[-1]).read_text())
            actual=np.load(request['input_npy'],allow_pickle=False)
            assert np.array_equal(actual,np.asarray(proposal.points).astype(np.float32))
            output=dict(payload,replay_provenance=dict(fresh_inference=False,source=str(frozen/'graspgenx/candidates.json')))
            Path(request['output_json']).write_text(json.dumps(output))
        def generator(*a,**kw):return generate_graspgenx_candidates(*a,runner=worker,**kw)
        directory.mkdir()
        started=time.monotonic()
        print('Benchmarking',name,flush=True)
        try:
            with mock.patch('real_nero_grasp.mgpc_completion.complete_graspnet_target',return_value=proposal), \
                 mock.patch('real_nero_grasp.graspgenx_integration.generate_graspgenx_candidates',side_effect=generator), \
                 (directory/'planning.log').open('w') as log,contextlib.redirect_stdout(log):
                session.update(arm,target,scene,T,None,None,{'label':'banana'},source,directory)
        finally:close_planning_process()
        refinement=json.loads((directory/'nero_refinement.json').read_text())
        event=session.events[-1]
        result=dict(name=name,wall_s=time.monotonic()-started,
            preflight_cap=args.feedback_probe_preflight_max_candidates,
            refinement_budget_s=args.feedback_probe_refine_timeout_s,
            refinement_route_attempts=refinement['route_attempts'],
            refinement_actual_s=refinement['seconds'],
            ready=session.ready,valid_sides=session.valid_sides,
            route_pass=sum(r['complete_route_pass'] for r in event['candidates']))
        results['cases'].append(result)
        (opts.output_dir/'summary.json').write_text(json.dumps(results,indent=2)+'\n')
        print(json.dumps(result),flush=True)


if __name__=='__main__':main()
