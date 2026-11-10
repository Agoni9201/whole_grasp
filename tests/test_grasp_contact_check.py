import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from real_nero_grasp.contact_check import capture_grasp_contact_snapshot
from real_nero_grasp.grasp_workflow import (
    _ContactCheckFinished, _capture_contact_check_or_continue, _contact_check_recovery_result,
)
from real_nero_grasp.types import GraspAttemptStatus, PlanNotExecutable, SafetyStopTriggered


class ReadOnlyArm:
    def __init__(self):
        self.joints = np.zeros(7)
    def get_joint_angles(self):
        return self.joints.copy()
    def get_flange_pose(self):
        return np.zeros(6)
    def get_tcp_pose(self):
        return np.array([.165, 0, -.025, 0, 0, 0])
    def print_gripper_status(self):
        return SimpleNamespace(value=.0998, force=.1, mode="width")
    def close_gripper(self, *_):
        raise AssertionError("diagnostic must not close")
    def move_j(self, *_):
        raise AssertionError("snapshot must not move")


def args():
    return SimpleNamespace(grasp_contact_check_only=True, rgb=None, depth=None,
                           save_dir="original", tcp_offset=[.165, 0, -.025, 0, 0, 0],
                           gripper_open_width=.1, grasp_advance=.015, calib_json="existing.json")


def save_capture(settings):
    path = Path(settings.save_dir) / "frame"
    path.mkdir(parents=True)
    (path / "color.png").write_bytes(b"camera-owned RGB")
    np.save(path / "depth_m.npy", np.full((2, 2), .2, dtype=np.float32))
    (path / "capture_meta.json").write_text("{}")
    return None, None, None, str(path)


def test_final_pose_snapshot_retains_live_health_context_and_original_args(tmp_path):
    settings = args()
    checks = []
    session = SimpleNamespace(owns_control=True, _arm=SimpleNamespace(_safety_stop_latched=False),
                              _raise_if_terminal_fault=lambda label: checks.append(label))
    settings._mit_session = session
    def capture(local):
        assert local._mit_session is session
        return save_capture(local)
    path = capture_grasp_contact_snapshot(ReadOnlyArm(), settings, tmp_path, np.zeros(6), capture_rgbd_hook=capture)
    report = json.loads(Path(path).read_text())
    assert report["stationary_during_capture"]
    assert not report["motion_commands_sent_by_snapshot"]
    assert report["grasp_advance_m"] == .015
    assert report["measured_opening_m"] == .0998
    assert settings.save_dir == "original"
    assert len(checks) == 3


def test_moving_capture_is_not_promoted_to_valid_contact_evidence(tmp_path):
    arm = ReadOnlyArm()
    def moving_capture(settings):
        result = save_capture(settings)
        arm.joints[0] += np.deg2rad(.5)
        return result
    with pytest.raises(PlanNotExecutable, match="Robot moved"):
        capture_grasp_contact_snapshot(arm, args(), tmp_path, np.zeros(6), capture_rgbd_hook=moving_capture)
    report = json.loads((tmp_path / "contact_check/frame/contact_pose.json").read_text())
    assert not report["stationary_during_capture"]


def test_diagnostic_leaves_workflow_before_any_close_or_lift():
    events = []
    def capture(*_):
        events.append("capture")
        return "contact_pose.json"
    with pytest.raises(_ContactCheckFinished) as stopped:
        _capture_contact_check_or_continue(ReadOnlyArm(), args(), "out", np.zeros(6), capture)
        events.append("close")
        events.append("lift")
    assert stopped.value.snapshot_ok
    assert events == ["capture"]


def test_capture_failure_leaves_for_recovery_instead_of_closing():
    def failed(*_):
        raise OSError("camera unavailable")
    with pytest.raises(_ContactCheckFinished) as stopped:
        _capture_contact_check_or_continue(ReadOnlyArm(), args(), "out", np.zeros(6), failed)
    assert not stopped.value.snapshot_ok


def test_control_fault_is_not_converted_to_a_normal_recovery_request():
    def fault(*_):
        raise SafetyStopTriggered("MIT fault")
    with pytest.raises(SafetyStopTriggered, match="MIT fault"):
        _capture_contact_check_or_continue(ReadOnlyArm(), args(), "out", np.zeros(6), fault)


@pytest.mark.parametrize("snapshot_ok,home_ok", [(True, True), (True, False), (False, True)])
def test_contact_result_requires_capture_and_verified_return_and_never_counts_as_grasp(snapshot_ok, home_ok):
    result = _contact_check_recovery_result(_ContactCheckFinished("check", snapshot_ok=snapshot_ok), True, home_ok)
    expected = GraspAttemptStatus.CONTACT_CHECKED if snapshot_ok and home_ok else GraspAttemptStatus.EXECUTION_FAILED
    assert result.status is expected
    assert not result.succeeded
    assert not result.empty_grasp_recovered_to_initial


def test_normal_grasp_mode_does_not_capture():
    settings = args()
    settings.grasp_contact_check_only = False
    _capture_contact_check_or_continue(ReadOnlyArm(), settings, "out", np.zeros(6), None)


@pytest.mark.parametrize("contact_confirmed,hold_requested,hold_blocked", [
    (False, False, False), (True, False, False), (True, True, False), (True, True, True),
])
def test_grasp_failure_hook_and_confirmed_contact_reference(
    tmp_path, monkeypatch, contact_confirmed, hold_requested, hold_blocked,
):
    from dataclasses import fields
    from unittest import mock
    from real_nero_grasp import grasp_workflow as workflow, runtime
    from real_nero_grasp.cli import parse_args
    from real_nero_grasp.grasp_geometry import _axis_vector
    from real_nero_grasp.types import (
        ControlledMotionStopTriggered, GraspMotionPlan, InitialSupportRelease,
        MotionNotReached, MotionSafetyScene, PlannedJointStage,
    )
    from real_nero_grasp.transforms import _make_T_from_pose, _transform_points

    monkeypatch.setattr("sys.argv", ["test", "--calib-json", "/tmp/offline.json",
                                    "--phase", "grasp", "--execute", "--yes"])
    settings = parse_args()
    settings.motion_mode = "ik_joint"
    settings.require_gripper_contact = False
    if hold_requested:
        settings.post_grasp_hold_joints_deg = [-.002, 0, .002, 100.6, 0, 0, 0]
    arm = mock.Mock()
    arm.get_tcp_pose.return_value = np.zeros(6)
    arm.get_joint_angles.return_value = np.zeros(7)
    pre, grasp, lift = [np.array([0, 0, z, 0, 0, 0]) for z in (.4, .3, .45)]
    stage = PlannedJointStage("pregrasp", pre, pre, np.zeros(7), "free_open")
    target = np.array([[0, 0, .01]], dtype=np.float32)
    environment = np.array([[10, 0, 0]], dtype=np.float32)
    scene = MotionSafetyScene(full_scene_base=np.vstack([target, environment]),
                              environment_base=environment, target_base=target, map_id="offline")
    release = InitialSupportRelease(environment, np.array([[0, 0, 0]]), grasp.copy())
    lift_stage = PlannedJointStage("support", lift, lift, np.zeros(7), "support_release")
    plan = GraspMotionPlan(np.zeros(7), (stage,), (lift_stage,), (), (), scene,
        attached_target_tcp=_transform_points(np.linalg.inv(_make_T_from_pose(grasp)), target),
        initial_support_release=release if contact_confirmed else None)
    proposal = SimpleNamespace(width=.04, depth=.03, translation=np.zeros(3),
                               rotation_matrix=np.eye(3))
    row = dict(idx=0, grasp=proposal, reachability_ok=True, tcp_width_sign=1,
               _workflow_motion_plan=plan, _workflow_pre_tcp=pre,
               _workflow_grasp_tcp=grasp, _workflow_lift_tcp=lift)
    hooks = {field.name: mock.Mock() for field in fields(workflow.GraspWorkflowHooks)}
    reason = "pregrasp MIT arrival gate failed: joint_err=1.12deg"
    executed_plans = []
    def execute(*values):
        executed_plans.append(values[3])
        if not contact_confirmed:
            raise MotionNotReached(reason)
        if len(executed_plans) == 1:
            return grasp, np.zeros(7)
        raise ControlledMotionStopTriggered("offline check after contact anchoring")
    hooks.update(
        axis_vector=_axis_vector,
        get_graspnet_helpers=lambda: (lambda *a, **k: ({}, object()),
                                     lambda *a, **k: ([proposal], [row])),
        annotate_grasp_candidate_physical_aperture=lambda rows, *a, **k: rows,
        annotate_grasp_candidate_reachability=lambda rows, *a, **k: rows,
        annotate_grasp_candidate_workflow_preflight=lambda rows, *a, **k: rows,
        defer_candidate_endpoint_reachability=lambda _: False,
        choose_default_grasp_candidate=lambda *a: (proposal, row),
        grasp_candidate_width_shift=lambda *a: 0.,
        grasp_to_tcp_poses=lambda *a, **k: (pre, grasp, lift),
        grasp_camera_summary=lambda _: dict(graspnet_translation_camera=[0, 0, 0],
            graspnet_rpy_camera_deg=[0, 0, 0], graspnet_rotation_matrix_camera=np.eye(3).tolist()),
        execution_gripper_open_width=lambda *a: .1,
        handle_motion_not_reached=runtime.handle_motion_not_reached,
        execute_prevalidated_stages=execute,
    )
    monkeypatch.setattr(workflow, "_prepare_graspnet_target_cloud", lambda value, *a, **k: value)
    monkeypatch.setattr("real_nero_grasp.pointcloud_fusion.save_cloud", mock.Mock())
    monkeypatch.setattr(runtime, "print_status", mock.Mock())
    actual_tcp = grasp.copy()
    actual_tcp[2] -= .0025
    actual_transform = _make_T_from_pose(actual_tcp)
    fk = mock.Mock(return_value=actual_transform)
    monkeypatch.setattr("real_nero_grasp.collision._tcp_transform_from_joints", fk)
    holding_stage = PlannedJointStage("joint_hold", lift, lift, np.ones(7), "carrying")
    def plan_hold(*values):
        arm.open_gripper.assert_not_called()
        arm.close_gripper.assert_not_called()
        np.testing.assert_array_equal(values[0], lift_stage.joints)
        if hold_blocked:
            raise PlanNotExecutable("hold route blocked")
        return (holding_stage,)
    hold_planner = mock.Mock(side_effect=plan_hold)
    monkeypatch.setattr("real_nero_grasp.post_grasp_rrt.plan_post_grasp_joint_hold", hold_planner)
    result = workflow.plan_and_maybe_execute_grasp(
        arm, settings, object(), object(), np.eye(4), None, None, str(tmp_path),
        {"label": "banana"}, hooks=workflow.GraspWorkflowHooks(**hooks))
    if hold_blocked:
        assert result.status is GraspAttemptStatus.PLANNING_FAILED
        assert result.detail == "hold route blocked"
        assert executed_plans == []
        arm.open_gripper.assert_not_called()
        arm.close_gripper.assert_not_called()
        return
    assert result.status is GraspAttemptStatus.EXECUTION_FAILED
    if not contact_confirmed:
        assert result.detail == reason
        arm.close_gripper.assert_not_called()
        arm.close_gripper_slow.assert_not_called()
        fk.assert_not_called()
        assert not (tmp_path / "post_close_contact_reference.json").exists()
    else:
        fk.assert_called_once()
        fixed = executed_plans[1]
        np.testing.assert_allclose(fixed.initial_support_release.start_tcp_pose, actual_tcp)
        np.testing.assert_allclose(release.start_tcp_pose, grasp)
        np.testing.assert_allclose(_transform_points(actual_transform, fixed.attached_target_tcp),
                                   target, atol=1e-7)
        assert fixed.before_close is plan.before_close
        if hold_requested:
            hold_planner.assert_called_once()
            assert fixed.after_close == plan.after_close + (holding_stage,)
        else:
            hold_planner.assert_not_called()
            assert fixed.after_close is plan.after_close
        assert fixed.initial_support_release.downward_tolerance == .001
        report = json.loads((tmp_path / "post_close_contact_reference.json").read_text())
        assert report["downward_tolerance_m"] == .001
        assert report["strict_after_rise_m"] == .018
        assert report["target_base_unchanged"]
        assert report["joint_targets_unchanged"]

        from real_nero_grasp import collision
        old_context = runtime._collision_context_for_planned_stage(plan, lift_stage)
        fixed_context = runtime._collision_context_for_planned_stage(fixed, lift_stage)
        def measured_transform(q, _args):
            transform = actual_transform.copy()
            transform[2, 3] += q[0]
            return transform
        def metrics(context, drop=0):
            return collision._motion_configuration_metrics(
                np.r_[drop, np.zeros(6)], context, settings,
                tcp_transform_from_joints_hook=measured_transform,
                motion_robot_capsules_hook=lambda *a, **k: [],
            )
        assert metrics(old_context)["primitive"] == "support_release_downward"
        assert metrics(old_context)["margin"] < 0
        assert metrics(fixed_context)["support_rise"] == 0
        assert metrics(fixed_context)["margin"] >= 0
        downward = metrics(fixed_context, -.0011)
        assert downward["primitive"] == "support_release_downward"
        assert downward["margin"] < 0


def test_contact_check_log_is_diagnostic_and_not_a_success_or_failure(tmp_path):
    from real_nero_grasp.experiment_log import initialize_experiment_log, finalize_experiment_log
    settings = args()
    settings.save_dir = str(tmp_path)
    initialize_experiment_log(settings, ["test", "--grasp-contact-check-only"])
    result = _contact_check_recovery_result(_ContactCheckFinished("saved", snapshot_ok=True), True, True)
    report = finalize_experiment_log(settings, result)
    assert report["method"]["contact_check_only"]
    assert report["outcome"]["status"] == "contact_checked"
    assert not report["outcome"]["success"]
    assert report["outcome"]["failure_stage"] is None


def test_snapshot_sdk_path_uses_real_depth_worker_before_image_directory_exists(tmp_path, monkeypatch):
    """Replay the failed call order, with actual numerical IPC and no devices."""
    from cameras import realsense
    from real_nero_grasp import planning_process
    settings = args()
    settings.trajectory_executor = "mit"
    settings.orbbec_depth_median_frames = 3
    settings.orbbec_depth_median_min_valid = 2
    settings.orbbec_depth_median_max_mad = .004
    intr = SimpleNamespace(fx=100., fy=100., cx=8., cy=6.)

    class Camera:
        depth_filters = "all"
        depth_width, depth_height = 16, 12
        sequence = 0
        def read(self):
            self.sequence += 1
            return realsense.Frame(
                np.zeros((12, 16, 3), dtype=np.uint8),
                np.full((12, 16), .2, dtype=np.float32), intr,
                np.array([[100., 0, 8.], [0, 100., 6.], [0, 0, 1.]]),
                np.zeros(5), self.sequence, float(self.sequence), float(self.sequence),
                "offline-test", np.full((12, 16), .2, dtype=np.float32),
            )

    camera = Camera()
    monkeypatch.setattr(realsense, "camera_for", lambda _: camera)
    out_dir = tmp_path / "previous_capture"
    assert not out_dir.exists()
    def capture(local):
        # Before this fix the raw-depth IPC tried to log here before mkdir.
        assert Path(local.save_dir).is_dir()
        return realsense.capture_rgbd(local)
    planning_process.close_planning_process()
    try:
        report_path = Path(capture_grasp_contact_snapshot(
            ReadOnlyArm(), settings, out_dir, np.zeros(6), capture_rgbd_hook=capture,
        ))
        report = json.loads(report_path.read_text())
        worker_log = out_dir / "contact_check/planning_process.jsonl"
        records = [json.loads(line) for line in worker_log.read_text().splitlines()]
        assert records[-1]["operation"] == "supported_depth_median_files"
        assert report["stationary_during_capture"]
        assert (report_path.parent / "color.png").is_file()
        np.testing.assert_allclose(np.load(report_path.parent / "depth_m.npy"), .2)
        assert (report_path.parent / "depth_raw_frames.npz").is_file()
    finally:
        planning_process.close_planning_process()
