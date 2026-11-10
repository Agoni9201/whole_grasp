"""Exercise geometry IPC and blocked/crashed windows without opening a GUI/CAN."""

import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from real_nero_grasp import preview_process as preview
from real_nero_grasp.types import SafetyStopTriggered
from real_nero_grasp.visualization import get_o3d


def geometries():
    o3d = get_o3d()
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector([[0., 0., 0.], [0., 1., 2.]])
    cloud.paint_uniform_color([1., .2, 0.])
    mesh = o3d.geometry.TriangleMesh.create_box(.1, .2, .3)
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color([0., 0., 1.])
    lines = o3d.geometry.LineSet()
    lines.points = o3d.utility.Vector3dVector([[0., 0., 0.], [0., 1., 0.]])
    lines.lines = o3d.utility.Vector2iVector([[0, 1]])
    lines.colors = o3d.utility.Vector3dVector([[.5, .4, .3]])
    return [cloud, mesh, lines]


def test_cloud_mesh_and_lines_round_trip_preserves_coordinates_and_colors(tmp_path):
    original = geometries()
    arrays, specs = preview._pack_geometries(original)
    np.savez(tmp_path / "data.npz", **arrays)
    loaded = preview._load_geometries(specs, tmp_path / "data.npz")
    for before, after, spec in zip(original, loaded, specs):
        for name in spec["fields"]:
            np.testing.assert_array_equal(np.asarray(getattr(before, name)),
                                          np.asarray(getattr(after, name)))


def spawn_fake_window(behavior, processes, request_paths):
    # Replace only the child's display function. Production IPC, Open3D loading,
    # fresh-interpreter startup and AF_CAN prohibition all run unchanged.
    bootstrap = '''
import ctypes, os, socket, sys, time
from pathlib import Path
from real_nero_grasp import preview_process as p
def draw(geometries, request):
    try:
        socket.socket(socket.AF_CAN, socket.SOCK_RAW, socket.CAN_RAW)
    except RuntimeError as exc:
        assert 'forbidden' in str(exc)
    else:
        raise AssertionError('Preview worker unexpectedly opened CAN')
    Path(sys.argv[-1]).with_name('drawing').write_text(str(time.monotonic()))
    BEHAVIOR
p._draw = draw
raise SystemExit(p.main())
'''.replace('BEHAVIOR', behavior)
    def spawn(command, **kwargs):
        request_paths.append(Path(command[-1]))
        process = subprocess.Popen([command[0], "-c", bootstrap, *command[3:]], **kwargs)
        processes.append(process)
        return process
    return spawn


def test_child_holds_its_gil_while_parent_control_heartbeat_continues():
    processes, paths, ticks, checks, starts = [], [], [], [], []
    stop = threading.Event()
    def heartbeat():
        while not stop.wait(.003):
            ticks.append(time.monotonic())
    def health():
        if paths and paths[0].with_name('drawing').is_file():
            checks.append(time.monotonic())
            if not starts:
                starts.append(float(paths[0].with_name('drawing').read_text()))
    thread = threading.Thread(target=heartbeat)
    thread.start()
    try:
        result = preview.show_preview(geometries(), "test", 2., 640, 480,
            health_check=health, popen_factory=spawn_fake_window(
                'ctypes.PyDLL(None).usleep(400000)', processes, paths))
    finally:
        stop.set()
        thread.join()
    assert result['pid'] != os.getpid()
    assert result['can_forbidden']
    assert not result['runtime_loaded'] and not result['camera_loaded']
    assert len(checks) >= 5
    assert len([t for t in ticks if starts[0] <= t <= starts[0] + .4]) >= 20
    assert processes[0].poll() == 0


def test_control_fault_terminates_window_child_and_propagates():
    processes, paths = [], []
    def health():
        if paths and paths[0].with_name('drawing').is_file():
            raise SafetyStopTriggered('synthetic MIT fault')
    with pytest.raises(SafetyStopTriggered, match='synthetic MIT fault'):
        preview.show_preview(geometries(), "test", 2., 640, 480,
            health_check=health, popen_factory=spawn_fake_window(
                'ctypes.PyDLL(None).usleep(2000000)', processes, paths))
    assert processes[0].poll() is not None


def test_crashed_window_is_reported_without_inline_gui_fallback():
    processes, paths = [], []
    with pytest.raises(RuntimeError, match='preview worker failed'):
        preview.show_preview(geometries(), "test", 2., 640, 480,
            popen_factory=spawn_fake_window('os._exit(17)', processes, paths))
    assert processes[0].poll() == 17


def test_fused_preview_does_not_swallow_a_control_fault():
    from real_nero_grasp.pointcloud_fusion import preview_fused_cloud
    args = SimpleNamespace(preview_active_clouds=True, preview_cloud_frame=False)
    def fault(*a, **kw):
        raise SafetyStopTriggered('synthetic preview stop')
    cloud = geometries()[0]
    with pytest.raises(SafetyStopTriggered, match='synthetic preview stop'):
        preview_fused_cloud(cloud, cloud, args, draw_open3d_preview_hook=fault)
