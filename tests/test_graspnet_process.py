"""Candidate IPC and control ownership during isolated GraspNet work; no CAN."""

import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace as NS
from dataclasses import fields
from unittest import mock

import numpy as np
import pytest

from real_nero_grasp import graspnet_process as worker
from real_nero_grasp.types import SafetyStopTriggered


RAW = [1., .05, .02, .03, 1., 0., 0., 0., 1., 0., 0., 0., 1., .01, .02, .4, -1.]


def setup_args():
    return NS(grasp_num_points=3000, graspnet_seed=42, graspnet_process_timeout_s=15.)


def cloud(offset=0.):
    return NS(points=np.tile([offset, 0., .4], (200, 1)),
              colors=np.tile([.2, .4, .6], (200, 1)))


def fake_process(behavior, processes, paths):
    bootstrap = '''
import ctypes, socket, sys, time
from pathlib import Path
from real_nero_grasp import graspnet_process as w
def infer(request):
    try:
        socket.socket(socket.AF_CAN, socket.SOCK_RAW, socket.CAN_RAW)
    except RuntimeError as exc:
        assert 'forbidden' in str(exc)
    else:
        raise AssertionError('CAN must be forbidden')
    assert not request['options']['visual'] and not request['options']['manual_select']
    Path(sys.argv[-1]).with_name('working').write_text('started')
    BEHAVIOR
w._infer = infer
raise SystemExit(w.main())
'''.replace('BEHAVIOR', behavior)
    def launch(command, **kwargs):
        paths.append(Path(command[-1]))
        p = subprocess.Popen([command[0], '-c', bootstrap, *command[3:]], **kwargs)
        processes.append(p)
        return p
    return launch


def test_numeric_candidate_roundtrip_preserves_pose_rows_and_infinite_metadata():
    grasp = worker._grasp_api()[0](np.asarray(RAW))
    original = {'grasp': grasp, 'translation': grasp.translation.copy(), 'ik_jump': float('inf')}
    decoded = worker._unpack(json.loads(json.dumps(worker._pack(original))))
    np.testing.assert_array_equal(decoded['grasp'].grasp_array, grasp.grasp_array)
    np.testing.assert_array_equal(decoded['translation'], original['translation'])
    assert np.isinf(decoded['ik_jump'])


def test_worker_preserves_measured_support_and_checks_health_during_blocking_work(tmp_path):
    processes, paths, checks = [], [], []
    args = setup_args()
    args._mit_session = NS(owns_control=True, _arm=NS(_safety_stop_latched=False),
                           _raise_if_terminal_fault=lambda label: checks.append(label))
    behavior = "ctypes.PyDLL(None).usleep(300000); return {'group': [RAW], 'rows': [{'idx': 0, 'grasp': {'__grasp__': RAW}, 'translation': {'__array__': [.01,.02,.4], 'dtype': 'float64'}}]}".replace('RAW', repr(RAW))
    group, rows = worker.generate_candidates(cloud(), cloud(.1), args, tmp_path,
        {'visual': True, 'manual_select': True}, measured_target=cloud(.2),
        popen_factory=fake_process(behavior, processes, paths))
    assert len(checks) >= 5
    assert processes[0].pid != os.getpid()
    assert group[0].width == .05
    assert rows[0]['idx'] == 0
    request = json.loads(paths[0].read_text())
    assert request['measured_target'] and request['seed'] == 42
    with np.load(request['input_file'], allow_pickle=False) as data:
        np.testing.assert_array_equal(data['measured_points'], cloud(.2).points)
        np.testing.assert_array_equal(data['scene_points'], cloud(.1).points)
    result = json.loads((tmp_path/'graspnet_worker/result.json').read_text())
    assert result['can_forbidden'] and not result['runtime_loaded']


def test_worker_failure_removes_stale_result_and_never_returns_candidates(tmp_path):
    folder = tmp_path/'graspnet_worker'; folder.mkdir()
    (folder/'result.json').write_text('stale success')
    processes, paths = [], []
    with pytest.raises(RuntimeError, match='without output'):
        worker.generate_candidates(cloud(), cloud(.1), setup_args(), tmp_path, {},
            popen_factory=fake_process("__import__('os')._exit(17)", processes, paths))
    assert not (folder/'result.json').exists()


def test_control_stop_during_inference_kills_child_and_propagates(tmp_path):
    args = setup_args(); processes, paths = [], []
    def check(_):
        if paths and paths[0].with_name('working').exists():
            raise SafetyStopTriggered('synthetic MIT fault')
    args._mit_session = NS(owns_control=True, _arm=NS(_safety_stop_latched=False),
                           _raise_if_terminal_fault=check)
    with pytest.raises(SafetyStopTriggered, match='synthetic MIT fault'):
        worker.generate_candidates(cloud(), cloud(.1), args, tmp_path, {},
            popen_factory=fake_process('ctypes.PyDLL(None).usleep(2000000)', processes, paths))
    assert processes[0].poll() is not None


def test_stage_logging_cannot_overwrite_the_stationary_control_phase(tmp_path):
    from real_nero_grasp.experiment_log import initialize_experiment_log, mark_experiment_stage
    session = mock.Mock(running=True)
    args = NS(save_dir=str(tmp_path), _mit_session=session)
    initialize_experiment_log(args, ['test'])
    mark_experiment_stage(args, 'grasp_planning')
    session.set_phase.assert_not_called()
    assert args._experiment_log_state['stage'] == 'grasp_planning'


def test_mit_workflow_skips_in_process_helpers_and_propagates_control_fault(tmp_path):
    from real_nero_grasp import grasp_workflow as workflow
    from real_nero_grasp.cli import parse_args
    from real_nero_grasp.grasp_geometry import _axis_vector
    with mock.patch.object(sys, 'argv', ['test','--calib-json','/tmp/test.json','--phase','grasp']):
        args = parse_args()
    args._mit_session = NS(owns_control=True, _arm=NS(_safety_stop_latched=False),
                           _raise_if_terminal_fault=lambda _: None)
    hooks = {field.name: mock.Mock() for field in fields(workflow.GraspWorkflowHooks)}
    hooks.update(axis_vector=_axis_vector,
                 get_graspnet_helpers=mock.Mock(side_effect=AssertionError('parent GraspNet forbidden')))
    arm = mock.Mock()
    with mock.patch.object(workflow, '_prepare_graspnet_target_cloud', side_effect=lambda value,*a,**kw:value), \
         mock.patch('real_nero_grasp.pointcloud_fusion.save_cloud'), \
         mock.patch.object(worker, 'generate_candidates', side_effect=SafetyStopTriggered('stop during inference')):
        with pytest.raises(SafetyStopTriggered, match='stop during inference'):
            workflow.plan_and_maybe_execute_grasp(arm, args, cloud(), cloud(.1), np.eye(4),
                None, None, str(tmp_path), {'label':'banana'}, hooks=workflow.GraspWorkflowHooks(**hooks))
    hooks['get_graspnet_helpers'].assert_not_called()
    hooks['annotate_grasp_candidate_physical_aperture'].assert_not_called()
    arm.move_j.assert_not_called()


def test_runtime_cache_release_defers_full_gc_while_mit_owns_control():
    from real_nero_grasp import runtime
    args = NS(_mit_session=NS(owns_control=True))
    with mock.patch.object(runtime, 'release_vision_model_caches') as release:
        hooks = runtime._grasp_workflow_hooks(args)
        hooks.release_vision_model_caches()
    release.assert_called_once_with(collect_garbage=False)
