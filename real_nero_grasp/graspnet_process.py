"""Isolated GraspNet inference/initial filtering, with plain geometry IPC only."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import sysconfig
import time

import numpy as np

from .paths import PROJECT_ROOT


def _grasp_api():
    path = str(PROJECT_ROOT / "graspnet-baseline/graspnetAPI")
    if path not in sys.path:
        sys.path.insert(0, path)
    from graspnetAPI import Grasp, GraspGroup
    return Grasp, GraspGroup


def _pack(value):
    if hasattr(value, "grasp_array"):
        return {"__grasp__": np.asarray(value.grasp_array, dtype=float).tolist()}
    if isinstance(value, np.ndarray):
        if value.dtype.kind not in 'fibu':
            raise TypeError('Only numeric arrays may cross GraspNet IPC')
        return {"__array__": value.tolist(), "dtype": str(value.dtype)}
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _pack(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_pack(item) for item in value]
    if value is None or type(value) in (str, float, int, bool):
        return value
    raise TypeError(f'Unsupported GraspNet IPC data: {type(value).__name__}')


def _unpack(value):
    if isinstance(value, dict):
        if set(value) == {'__grasp__'}:
            array = np.asarray(value['__grasp__'], dtype=float)
            if array.shape != (17,) or not np.isfinite(array).all():
                raise ValueError('Invalid returned GraspNet grasp')
            return _grasp_api()[0](array)
        if set(value) == {'__array__', 'dtype'}:
            array = np.asarray(value['__array__'], dtype=value['dtype'])
            if array.dtype.kind not in 'fibu':
                raise TypeError('Invalid returned GraspNet array dtype')
            return array
        return {key: _unpack(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_unpack(item) for item in value]
    return value


def _save_cloud(arrays, prefix, cloud):
    points = np.asarray(cloud.points, dtype=float)
    colors = np.asarray(getattr(cloud, 'colors', []), dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or not len(points) or not np.isfinite(points).all():
        raise ValueError(f'Invalid GraspNet {prefix} cloud')
    if colors.size and (colors.shape != points.shape or not np.isfinite(colors).all()):
        raise ValueError(f'Invalid GraspNet {prefix} colors')
    arrays[prefix + '_points'] = points
    arrays[prefix + '_colors'] = colors.reshape(-1, 3)


def generate_candidates(proposal, scene, args, out_dir, options, *,
                        measured_target=None, popen_factory=None):
    from .planning_process import planning_health_check
    from .preview_process import _stop_process
    def health():
        planning_health_check(args, 'GraspNet candidate generation')
    health()
    directory = Path(out_dir) / 'graspnet_worker'
    directory.mkdir(parents=True, exist_ok=True)
    output = directory / 'result.json'
    output.unlink(missing_ok=True)
    arrays = {}
    _save_cloud(arrays, 'proposal', proposal)
    _save_cloud(arrays, 'scene', scene)
    if measured_target is not None:
        _save_cloud(arrays, 'measured', measured_target)
    np.savez(directory / 'input.npz', **arrays)
    options = dict(options)
    # Robot-facing previews and selection are performed in the parent only
    # after shared physical-aperture/IK/workflow checks.
    options.update(visual=False, manual_select=False)
    seed = getattr(args, 'graspnet_seed', None)
    seed = int(np.random.randint(0, 2**31-1) if seed is None else seed)
    request = dict(input_file=str((directory / 'input.npz').resolve()),
                   output_file=str(output.resolve()), options=_pack(options),
                   num_points=int(args.grasp_num_points), seed=seed,
                   measured_target=measured_target is not None)
    request_path = directory / 'request.json'
    request_path.write_text(json.dumps(request, indent=2) + '\n')
    environment = os.environ.copy()
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        environment[key] = '1'
    environment['PYTHONNOUSERSITE'] = '1'
    # ROS Noetic's Python 3.8 packages must not shadow this interpreter's
    # compatible xacro/roboticstoolbox dependencies.
    environment['PYTHONPATH'] = os.pathsep.join(filter(None, (
        sysconfig.get_path('purelib'), str(PROJECT_ROOT), environment.get('PYTHONPATH'))))
    timeout = float(getattr(args, 'graspnet_process_timeout_s', 180.))
    if not np.isfinite(timeout) or timeout <= 0:
        raise ValueError('Invalid GraspNet process timeout')
    with (directory / 'worker.log').open('w') as log:
        process = (popen_factory or subprocess.Popen)(
            [sys.executable, '-m', 'real_nero_grasp.graspnet_process', '--request', str(request_path.resolve())],
            env=environment, stdin=subprocess.DEVNULL, close_fds=True,
            start_new_session=True, stdout=log, stderr=subprocess.STDOUT)
        print(f'[graspnet][worker] pid={process.pid}; isolated load/inference/filtering; no CAN ownership', flush=True)
        deadline = time.monotonic() + timeout
        try:
            while process.poll() is None:
                health()
                if time.monotonic() >= deadline:
                    raise RuntimeError(f'GraspNet worker timed out; see {directory / "worker.log"}')
                time.sleep(.02)
            health()
            if not output.is_file():
                raise RuntimeError(f'GraspNet worker exited {process.returncode} without output; see {directory / "worker.log"}')
            payload = json.loads(output.read_text())
            if payload.get('source') != 'isolated_graspnet':
                raise RuntimeError('Unexpected GraspNet worker result')
            if not payload.get('ok'):
                error = RuntimeError(payload.get('error', 'GraspNet worker failed'))
                if payload.get('diagnostic_candidate_rows'):
                    error.diagnostic_candidate_rows = _unpack(payload['diagnostic_candidate_rows'])
                    error.diagnostic_filter_stage = payload.get('diagnostic_filter_stage')
                raise error
            if process.returncode != 0:
                raise RuntimeError(f'GraspNet worker failed after producing output: {process.returncode}')
            rows = _unpack(payload['rows'])
            group = np.asarray(payload['group'], dtype=float).reshape(-1, 17)
            if not rows or not len(group) or not np.isfinite(group).all():
                raise RuntimeError('GraspNet worker returned invalid candidates')
            health()
            print(f'[graspnet][worker] {len(rows)} proposals ready; seed={seed}; shared safety preflight follows', flush=True)
            return _grasp_api()[1](group), rows
        finally:
            _stop_process(process)


def _infer(request):
    import torch
    torch.set_num_threads(1)
    torch.manual_seed(request['seed'])
    torch.cuda.manual_seed_all(request['seed'])
    np.random.seed(request['seed'])
    from .visualization import get_o3d, get_graspnet_helpers
    o3d = get_o3d()
    make_inputs, generate = get_graspnet_helpers()
    def cloud(data, prefix):
        result = o3d.geometry.PointCloud()
        result.points = o3d.utility.Vector3dVector(data[prefix + '_points'])
        colors = data[prefix + '_colors']
        if len(colors):
            result.colors = o3d.utility.Vector3dVector(colors)
        else:
            result.paint_uniform_color([.5, .5, .5])
        return result
    with np.load(request['input_file'], allow_pickle=False) as data:
        proposal = cloud(data, 'proposal')
        scene = cloud(data, 'scene')
        measured = cloud(data, 'measured') if request['measured_target'] else None
    options = _unpack(request['options'])
    options.update(visual=False, manual_select=False, collision_cloud=scene)
    if measured is not None:
        options['target_support_cloud'] = measured
    print('[graspnet-worker] prepare inputs and load/infer', flush=True)
    end_points, _ = make_inputs(proposal, num_point=request['num_points'])
    group, rows = generate(end_points, proposal, **options)
    return dict(group=np.asarray(group.grasp_group_array).tolist(), rows=_pack(rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, required=True)
    args = parser.parse_args()
    import socket
    original_socket = socket.socket
    class GraspNetSocket(original_socket):
        def __init__(self, family=socket.AF_INET, *a, **kw):
            if family == getattr(socket, 'AF_CAN', None):
                raise RuntimeError('CAN connections are forbidden in the GraspNet worker')
            super().__init__(family, *a, **kw)
    socket.socket = GraspNetSocket
    request = json.loads(args.request.read_text())
    payload = dict(source='isolated_graspnet', pid=os.getpid(), seed=request['seed'], can_forbidden=True)
    try:
        from threadpoolctl import threadpool_limits
        with threadpool_limits(limits=1):
            payload.update(_infer(request), ok=True)
    except Exception as exc:
        payload.update(ok=False, error=f'{exc}', error_type=type(exc).__name__)
        if getattr(exc, 'diagnostic_candidate_rows', None):
            payload.update(diagnostic_candidate_rows=_pack(exc.diagnostic_candidate_rows),
                           diagnostic_filter_stage=getattr(exc, 'diagnostic_filter_stage', None))
    payload['runtime_loaded'] = 'real_nero_grasp.runtime' in sys.modules
    Path(request['output_file']).write_text(json.dumps(payload, indent=2) + '\n')
    return 0 if payload['ok'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
