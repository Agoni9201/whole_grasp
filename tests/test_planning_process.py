"""Numerical-process integration and failure tests; never connect hardware."""

import os
import threading
import time
from types import SimpleNamespace as NS
from unittest import mock

import numpy as np
import pytest

from real_nero_grasp import planning_process as module
from real_nero_grasp.paths import DEFAULT_IK_URDF


@pytest.fixture
def worker():
    item = module.PlanningProcess()
    item.request("ping", {}, timeout_s=30.0)  # production warms dependencies before CAN setup
    yield item
    item.close()
    module.close_planning_process()


def solver_key(timeout=.02):
    return (str(DEFAULT_IK_URDF), "base_link", "link7", timeout, 1e-5, "Distance")


def test_spawn_worker_is_a_distinct_persistent_numerical_process(worker):
    first = worker.request("ping", {}, timeout_s=20.0)
    second = worker.request("ping", {}, timeout_s=5.0)
    assert first["pid"] != os.getpid()
    assert first["can_forbidden"]
    assert first == second
    assert not first["runtime_loaded"]
    assert not first["camera_loaded"]


def test_worker_telemetry_creates_new_nested_capture_directory(worker, tmp_path):
    import json
    destination = tmp_path / "new_capture" / "contact_check"
    assert not destination.exists()
    result = worker.request("ping", {}, args=NS(save_dir=str(destination)))
    record = json.loads((destination / "planning_process.jsonl").read_text())
    assert record["operation"] == "ping"
    assert record["pid"] == result["pid"]
    assert worker.request("ping", {})["pid"] == result["pid"]


def test_table_reference_quality_rejection_keeps_worker_alive(worker, tmp_path):
    before = worker.request("ping", {})
    result = worker.request("physical_table_reference", dict(
        config=dict(active_view=True, active_side_view_count=0,
                    grasp_table_reference_z=0., table_z=0.),
        capture_dir=str(tmp_path / "missing"), T_base_camera=np.eye(4),
        output_dir=str(tmp_path / "output")))
    assert not result["ok"]
    assert "raw archives" in result["reason"]
    assert worker.request("ping", {})["pid"] == before["pid"]


def test_config_whitelist_never_serializes_driver_or_mit_session():
    args = NS(value=[1., 2.], _mit_session=object(), _camera=object())
    assert module.config_snapshot(args) == {"value": [1., 2.]}
    with pytest.raises(TypeError):
        module.config_snapshot(NS(arm=object()))
    frozen = module._FrozenArm({"joints": np.zeros(7), "tcp": np.zeros(6), "flange": np.zeros(6)})
    assert not hasattr(frozen, "arm")
    assert not hasattr(frozen, "move_j")


def test_pipeline_joint_hold_config_cli_override_and_disable(monkeypatch):
    from real_nero_grasp.cli import parse_args
    import sys
    base = ["test", "--config", "config/pipeline.yaml", "--phase", "grasp"]
    monkeypatch.setattr(sys, "argv", base)
    assert parse_args().post_grasp_hold_joints_deg == [-.002, 0, .002, 100.6, 0, 0, 0]
    monkeypatch.setattr(sys, "argv", base + ["--no-post-grasp-hold"])
    assert parse_args().post_grasp_hold_joints_deg is None
    monkeypatch.setattr(sys, "argv", base + ["--post-grasp-hold-joints-deg", "0", "0", "0", "90", "0", "0", "0"])
    assert parse_args().post_grasp_hold_joints_deg == [0, 0, 0, 90, 0, 0, 0]
    monkeypatch.setattr(sys, "argv", base + ["--post-grasp-hold-joints-deg", "nan", "0", "0", "90", "0", "0", "0"])
    with pytest.raises(SystemExit):
        parse_args()


def test_real_ik_runs_off_process_while_parent_heartbeat_keeps_running(worker):
    key = solver_key(.02)
    worker.request("ik_info", {"solver": key}, timeout_s=20)
    times, stop = [], threading.Event()
    def heartbeat():
        while not stop.is_set():
            times.append(time.monotonic())
            stop.wait(.002)
    thread = threading.Thread(target=heartbeat)
    thread.start()
    try:
        for _ in range(8):
            assert worker.request("ik", {"solver": key, "position": np.array([3., 3., 3.]),
                "wxyz": np.array([1., 0., 0., 0.]), "seed": np.zeros(7),
                "previous": True}, timeout_s=5) is None
    finally:
        stop.set()
        thread.join()
    # Proves concurrency, not a hard real-time scheduling guarantee.
    assert len(times) >= 30


def test_numerical_worker_fk_ik_round_trip_preserves_the_target(worker):
    key = solver_key()
    info = worker.request("ik_info", {"solver": key}, timeout_s=20)
    q = np.deg2rad([3.546, -.494, 4.357, 69.346, 2.379, -13.027, 75.])
    pose = worker.request("fk", {"solver": key, "joints": q})
    answer = worker.request("ik", {"solver": key, "position": pose[:3],
        "wxyz": pose[3:], "seed": q, "previous": True})
    assert answer is not None
    assert np.all(np.asarray(answer) >= info["limits"][0])
    assert np.all(np.asarray(answer) <= info["limits"][1])
    actual = worker.request("fk", {"solver": key, "joints": np.asarray(answer)})
    np.testing.assert_allclose(actual[:3], pose[:3], atol=1e-5)


def test_timeout_discards_worker_and_forbids_reusing_late_result(worker):
    worker.request("ik_info", {"solver": solver_key(.2)}, timeout_s=20)
    with pytest.raises(module.PlanningProcessError, match="timed out"):
        worker.request("ik", {"solver": solver_key(.2), "position": np.array([3., 3., 3.]),
            "wxyz": np.array([1., 0., 0., 0.]), "seed": np.zeros(7),
            "previous": True}, timeout_s=.03)
    assert worker._process.poll() is not None
    with pytest.raises(module.PlanningProcessError, match="unavailable"):
        worker.request("ping", {})


def test_worker_error_does_not_fall_back_to_inline_numerical_code(worker):
    with pytest.raises(module.PlanningProcessError, match="Unknown"):
        worker.request("not_a_numerical_operation", {}, timeout_s=20)
    assert worker._process.poll() is not None


def test_worker_crash_is_reported_without_hanging(worker):
    worker.request("ping", {}, timeout_s=20)
    worker._process.terminate()
    worker._process.wait(timeout=2)
    with pytest.raises(module.PlanningProcessError, match="unavailable"):
        worker.request("ping", {})


def test_health_failure_aborts_pending_computation(worker):
    worker.request("ik_info", {"solver": solver_key(.2)}, timeout_s=20)
    def fault():
        raise RuntimeError("control process safety fault")
    with pytest.raises(RuntimeError, match="control process safety fault"):
        worker.request("ik", {"solver": solver_key(.2), "position": np.array([3., 3., 3.]),
            "wxyz": np.array([1., 0., 0., 0.]), "seed": np.zeros(7),
            "previous": True}, health_check=fault, timeout_s=5)
    assert worker._process.poll() is not None


def test_failed_active_worker_stops_owner_before_returning_any_plan():
    from real_nero_grasp.types import ControlledMotionStopTriggered
    owner = NS(owns_control=True, _arm=mock.Mock())
    args = NS(_mit_session=owner)
    with mock.patch.object(module, "get_planning_process") as get:
        get.return_value.request.side_effect = module.PlanningProcessError("worker died")
        with pytest.raises(ControlledMotionStopTriggered):
            module.checked_request("nbv", {}, args)
    owner._arm.terminate_motion.assert_called_once()


@pytest.mark.parametrize("configured, expected", [(None, 60.), (12., 12.)])
def test_checked_request_allows_bounded_nbv_search_and_honors_explicit_deadline(configured, expected):
    args = NS() if configured is None else NS(planning_process_timeout_s=configured)
    with mock.patch.object(module, "get_planning_process") as get:
        get.return_value.request.return_value = []
        assert module.checked_request("nbv", {}, args) == []
    assert get.return_value.request.call_args.kwargs["timeout_s"] == expected


@pytest.mark.parametrize("kind", ["target", "scene", "obstacles"])
def test_capture_cloud_filter_preserves_local_geometry_and_colors(worker, kind):
    from real_nero_grasp import perception
    args = NS(trajectory_executor="mit", min_depth=.05, max_depth=1.5,
              min_target_points=10, voxel_size=.003, scene_voxel_size=.005,
              scene_cloud_cap=100000, target_outlier_filter=True,
              outlier_neighbors=12, outlier_std_ratio=2.0)
    x, y = np.meshgrid(np.linspace(-.1, .1, 64), np.linspace(-.08, .08, 48))
    xyz = np.stack([x, y, .4 + .1 * x], axis=-1).astype(np.float32)
    mask = x > 0
    bgr = np.zeros((48, 64, 3), dtype=np.uint8)
    bgr[:, :, 1] = 120
    functions = {
        "target": lambda: perception.cloud_from_mask(xyz, bgr, mask, args),
        "scene": lambda: perception.scene_cloud_from_xyz(xyz, bgr, args),
        "obstacles": lambda: perception.obstacle_cloud_from_non_target_mask(xyz, bgr, mask, args),
    }
    args._planning_worker = True
    expected = functions[kind]()
    args._planning_worker = False
    with mock.patch.object(module, "get_planning_process", return_value=worker):
        actual = functions[kind]()
    def ordered(cloud):
        points = np.asarray(cloud.points)
        return np.c_[points, np.asarray(cloud.colors)][np.lexsort(points.T)]
    np.testing.assert_allclose(ordered(actual), ordered(expected), atol=1e-8)
    assert not worker.request("ping", {})["runtime_loaded"]


@pytest.mark.parametrize("max_mad", [None, .004])
def test_depth_median_worker_preserves_depth_counts_and_rejection(worker, max_mad, tmp_path):
    from cameras.depth_fusion import median_valid_depth_frames
    from tools.verify_handeye_point import median_valid_depth_frames as compatibility_fn
    assert compatibility_fn is median_valid_depth_frames
    frames = np.array([
        [[.5, 0., .5, .5], [np.nan, np.inf, -.1, .5]],
        [[.501, .6, .535, .5], [np.nan, np.inf, -.1, .5]],
        [[.499, 0., .47, .5], [np.nan, np.inf, -.1, .5]],
        [[.5, .6, .545, .5], [np.nan, np.inf, -.1, .5]],
        [[.501, 0., .465, .5], [np.nan, np.inf, -.1, .5]],
        [[.5, .6, .54, .5], [np.nan, np.inf, -.1, .5]],
        [[.5, 0., .46, .5], [np.nan, np.inf, -.1, .5]],
    ], dtype=np.float32)
    original = frames.copy()
    expected = median_valid_depth_frames(frames, min_valid_frames=5, max_mad=max_mad)
    args = NS(orbbec_depth_median_min_valid=5, orbbec_depth_median_max_mad=max_mad,
              save_dir=str(tmp_path))
    with mock.patch.object(module, "get_planning_process", return_value=worker):
        actual = module.isolated_depth_median(list(frames), args)
    for a, b in zip(actual, expected):
        np.testing.assert_array_equal(a, b)
        assert a.dtype == b.dtype
    np.testing.assert_array_equal(frames, original)
    assert '"operation": "depth_median"' in (tmp_path / "planning_process.jsonl").read_text()
    assert not worker.request("ping", {})["camera_loaded"]


def test_raw_supported_depth_worker_matches_pure_function(worker, tmp_path):
    from cameras.depth_fusion import supported_depth_median
    sdk = np.full((7, 2, 3), .5, np.float32)
    raw = sdk.copy()
    raw[:, 0, 0] = 0
    raw[4:, 0, 1] = 0
    raw[5:, 0, 2] = 0
    sdk[:, 1, 0] = .4
    expected = supported_depth_median(sdk, raw, 5, .004)
    args = NS(orbbec_depth_median_min_valid=5, orbbec_depth_median_max_mad=.004,
              save_dir=str(tmp_path))
    with mock.patch.object(module, "get_planning_process", return_value=worker):
        actual = module.isolated_supported_depth_median(list(sdk), list(raw), args)
    assert actual['report'] == expected['report']
    for key in ('depth_m', 'sdk_depth_m', 'raw_valid_count', 'rejected_mask'):
        np.testing.assert_array_equal(actual[key], expected[key])
    assert not worker.request('ping', {})['camera_loaded']


def test_raw_depth_exchange_passes_paths_and_cleans_up_on_failure():
    from pathlib import Path
    captured = []
    def fail(operation, data, args):
        assert operation == 'supported_depth_median_files'
        assert not any(isinstance(value, np.ndarray) for value in data.values())
        folder = Path(data['directory'])
        captured.append(folder)
        assert (folder / 'raw.npy').is_file()
        assert (folder / 'processed.npy').is_file()
        raise module.PlanningProcessError('worker failed')
    with mock.patch.object(module, 'checked_request', side_effect=fail):
        with pytest.raises(module.PlanningProcessError, match='worker failed'):
            module.isolated_supported_depth_median(np.ones((7, 2, 2)), np.ones((7, 2, 2)), NS())
    assert len(captured) == 1 and not captured[0].exists()


@pytest.mark.parametrize("empty", [False, True])
def test_environment_map_worker_preserves_occupancy_metadata_and_readonly_arrays(worker, tmp_path, empty):
    from dataclasses import fields
    from real_nero_grasp import pointcloud_fusion as fusion
    args = NS(trajectory_executor="mit", _planning_worker=True,
              motion_environment_map_voxel=.005, motion_environment_map_radius_xy=.3,
              table_z=0., save_dir=str(tmp_path))
    x, y = np.meshgrid(np.linspace(-.09, .09, 32), np.linspace(-.07, .07, 24))
    points = np.c_[x.ravel(), y.ravel(), np.full(x.size, .05)].astype(np.float32)
    points = points[:0] if empty else points
    clouds = [NS(points=points), points + [0., 0., .0005]]
    kwargs = dict(target_base=None if empty else np.array([[0., 0., .1]]),
                  target_center_base=np.array([0., 0., .1]),
                  generation=7, map_id="isolated_test")
    expected = fusion.fuse_dynamic_environment_clouds(["center", "side"], clouds, args, **kwargs)
    args._planning_worker = False
    with mock.patch.object(module, "get_planning_process", return_value=worker), mock.patch.object(
        fusion, "small_isolated_component_keep_mask",
        side_effect=AssertionError("connected components must not run in the control process")
    ):
        actual = fusion.fuse_dynamic_environment_clouds(["center", "side"], clouds, args, **kwargs)
    for field in fields(expected):
        a, b = getattr(actual, field.name), getattr(expected, field.name)
        if isinstance(b, np.ndarray):
            np.testing.assert_array_equal(a, b)
            assert a.dtype == b.dtype and not a.flags.writeable
        else:
            assert a == b
    assert '"operation": "environment_map"' in (tmp_path / 'planning_process.jsonl').read_text()
    info = worker.request('ping', {})
    assert info['can_forbidden'] and not info['runtime_loaded'] and not info['camera_loaded']


def test_environment_worker_error_uses_controlled_stop_not_new_emergency_stop(worker):
    from real_nero_grasp import pointcloud_fusion as fusion
    from real_nero_grasp.types import ControlledMotionStopTriggered
    arm = mock.Mock(_safety_stop_latched=False)
    owner = NS(owns_control=True, _arm=arm, _raise_if_terminal_fault=mock.Mock())
    args = NS(trajectory_executor='mit', _mit_session=owner)
    with mock.patch.object(module, 'get_planning_process', return_value=worker):
        with pytest.raises(ControlledMotionStopTriggered, match='planning worker failed'):
            fusion.fuse_dynamic_environment_clouds(['first', 'missing'], [np.zeros((1, 3))], args)
    arm.terminate_motion.assert_called_once()
    arm.electronic_emergency_stop.assert_not_called()
    assert worker._process.poll() is not None


def test_environment_map_existing_safety_fault_sends_no_new_stop_or_worker_request():
    from real_nero_grasp import pointcloud_fusion as fusion
    from real_nero_grasp.types import SafetyStopTriggered
    arm = mock.Mock(_safety_stop_latched=True)
    owner = NS(owns_control=True, _arm=arm,
               _raise_if_terminal_fault=mock.Mock(side_effect=SafetyStopTriggered('existing fault')))
    args = NS(trajectory_executor='mit', _mit_session=owner)
    with mock.patch.object(module, 'get_planning_process') as get:
        with pytest.raises(SafetyStopTriggered, match='existing fault'):
            fusion.fuse_dynamic_environment_clouds(['first'], [np.zeros((1, 3))], args)
    get.assert_not_called()
    arm.terminate_motion.assert_not_called()
    arm.electronic_emergency_stop.assert_not_called()


@pytest.mark.parametrize("guarded", [False, True])
def test_nbv_snapshot_runs_same_numerical_ranking_in_spawn_worker(worker, tmp_path, guarded):
    import open3d as o3d
    from dataclasses import replace
    from threadpoolctl import threadpool_limits
    from real_nero_grasp import runtime, active_view
    with mock.patch("sys.argv", ["offline", "--calib-json", "/offline.json"]):
        args = runtime.parse_args()
    args.trajectory_executor = "mit"
    args.motion_mode = "ik_joint" if guarded else "cartesian"
    args.active_path_guard = guarded
    args.ik_all_joint_seed_count = 2
    args.active_side_roll_plan_candidates = 2
    args.active_side_roll_ik_seed_limit = 1
    args.active_side_roll_full_retry_candidates = 1
    args.active_nbv_mode = "fixed"
    args.active_obstacle_aware = False
    args.save_dir = str(tmp_path)
    x, y = np.meshgrid(np.linspace(-.025, .025, 8), np.linspace(-.04, .04, 10))
    points = np.c_[x.ravel(), y.ravel(), np.full(x.size, .3)]
    target = o3d.geometry.PointCloud()
    target.points = o3d.utility.Vector3dVector(points)
    scene = o3d.geometry.PointCloud(target)
    arm = module._FrozenArm({"joints": np.zeros(7), "tcp": np.zeros(6), "flange": np.zeros(6)})
    hooks = replace(runtime._active_view_hooks(), handeye_transform=lambda *_: np.eye(4))
    args._planning_worker = True  # evaluate the existing numerical implementation locally for comparison
    with threadpool_limits(limits=1):
        expected = active_view._build_active_view_candidates(
            arm, args, target, scene, np.eye(4), hooks=hooks)
    args._planning_worker = False
    with mock.patch.object(module, "get_planning_process", return_value=worker), mock.patch(
        "tools.real_nero_point_test.handeye_transform", return_value=np.eye(4)
    ):
        result = module.isolated_nbv(arm, args, target, scene, np.eye(4))
    assert result
    assert [row["name"] for row in result] == [row["name"] for row in expected]
    if not guarded:
        np.testing.assert_allclose([row["score"] for row in result], [row["score"] for row in expected])
    else:
        for row in result:
            assert isinstance(row["reason"], str)
            if row["reason"] == "ok":
                assert np.isfinite(row["planned_joint_target"]).all()
    records = (tmp_path / "planning_process.jsonl").read_text()
    assert '"operation": "nbv"' in records
