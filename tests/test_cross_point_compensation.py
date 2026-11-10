import hashlib
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from tools import cross_point_check as cross
from tools import real_nero_point_test as point


def transform(rpy=(0, 0, 0), xyz=(0, 0, 0)):
    return cross.pose_transform(np.r_[xyz, rpy])


def test_compensation_rotates_base_error_back_into_observation_flange():
    observation = transform((0.2, -0.7, 1.4), (-0.3, 0.1, 0.4))
    original = transform((0.1, 0.7, -0.2), (0.06, -0.09, -0.01))
    camera_point = np.array([0.03, 0.02, 0.45])
    predicted = (observation @ original @ np.r_[camera_point, 1])[:3]
    taught = predicted + [0.01, -0.015, -0.02]
    corrected = cross.corrected_translation(original, observation, predicted, taught)
    after = (observation @ corrected @ np.r_[camera_point, 1])[:3]
    np.testing.assert_allclose(after, taught, atol=1e-12)
    np.testing.assert_allclose(corrected[:3, :3], original[:3, :3])
    assert not np.allclose(corrected[:3, 3], original[:3, 3]+taught-predicted)
    np.testing.assert_allclose(original[:3, 3], [0.06, -0.09, -0.01])


@pytest.fixture
def capture_state(tmp_path):
    source = tmp_path / "original.json"
    source.write_text(json.dumps({"position": [0.06, -0.09, -0.01],
                                  "orientation": [0, 0, 0, 1]}))
    directory = tmp_path / "capture"
    directory.mkdir()
    args = SimpleNamespace(calib_json=str(source), tcp_offset=[0.165, 0, -0.025, 0, 0, 0])
    state = dict(
        target_pose=np.zeros(6), target_tcp_pose=np.zeros(6),
        out_dir=str(directory), calibration_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        T_flange_camera=transform(xyz=(0.06, -0.09, -0.01)),
        flange_pose=np.array([-0.3, 0, 0.4, 0, 0, 0]),
        P_base=np.array([-0.45, 0.02, 0.12]), P_camera=np.array([0.01, 0.02, 0.4]),
        result={"pixel": [500, 300]},
    )
    return args, state


def test_reference_uses_taught_tcp_not_flange_and_exports_without_overwrite(capture_state, monkeypatch):
    args, state = capture_state
    state["cross_last_execution"] = dict(purpose="approach_step", actual_tcp_xyz_m=[-0.449, 0.02, 0.12])
    pose = np.array([-0.60, 0.02, 0.14, 0, 0, 0])
    stamps = iter(np.arange(100, 120, dtype=float))
    monkeypatch.setattr(cross, "fresh_pose", lambda _: (pose, cross.pose_transform(pose), next(stamps)))
    monkeypatch.setattr(cross.time, "sleep", lambda _: None)
    monkeypatch.setattr("builtins.input", lambda _: "TOUCH")
    source_bytes = Path(args.calib_json).read_bytes()
    cross.record_taught_point(SimpleNamespace(), args, state)
    result = state["cross_comparison"]
    np.testing.assert_allclose(result["P_base_taught_tcp_m"], [-0.435, 0.02, 0.115])
    np.testing.assert_allclose(result["correction_base_mm"], [15, 0, -5], atol=1e-10)
    np.testing.assert_allclose(result["executed_minus_taught_mm"], [-14, 0, 5], atol=1e-10)
    assert len(result["T_base_flange_teach_samples"]) == 10
    candidate_path = cross.export_compensation(args, state)
    candidate = json.loads(candidate_path.read_text())
    assert candidate["rotation_changed"] is False
    assert "requires_new_capture_validation" in candidate["status"]
    np.testing.assert_allclose(candidate["position"], [0.075, -0.09, -0.015])
    with pytest.raises(FileExistsError):
        cross.export_compensation(args, state)
    assert Path(args.calib_json).read_bytes() == source_bytes


def test_reference_records_changed_attitude_using_full_taught_tcp(capture_state, monkeypatch, capsys):
    args, state = capture_state
    pose = np.array([-0.6, 0, 0.2, 0, 0, np.deg2rad(10)])
    counter = iter(range(100))
    monkeypatch.setattr(cross, "fresh_pose", lambda _: (pose, cross.pose_transform(pose), next(counter)))
    monkeypatch.setattr(cross.time, "sleep", lambda _: None)
    monkeypatch.setattr("builtins.input", lambda _: "TOUCH")
    cross.record_taught_point(SimpleNamespace(), args, state)
    result = state["cross_comparison"]
    assert result["teach_attitude_difference_deg"] == pytest.approx(10)
    expected = (cross.pose_transform(pose) @ cross.pose_transform(args.tcp_offset))[:3, 3]
    np.testing.assert_allclose(result["P_base_taught_tcp_m"], expected)
    np.testing.assert_allclose(result["correction_base_mm"], (expected-state["P_base"])*1000)
    assert (Path(state["out_dir"]) / "cross_comparison.json").exists()
    assert "仅记录、不拒绝" in capsys.readouterr().out


def test_export_requires_reference_and_unchanged_source(capture_state):
    args, state = capture_state
    with pytest.raises(ValueError, match="先用 r"):
        cross.export_compensation(args, state)
    state["cross_comparison"] = {"calibration_sha256": "different"}
    with pytest.raises(ValueError, match="标定文件已改变"):
        cross.export_compensation(args, state)


def test_fresh_feedback_rejects_stale_and_invalid_poses():
    wrapper = SimpleNamespace(timestamp=time.time()-2, msg=[0]*6)
    arm = SimpleNamespace(arm=SimpleNamespace(get_flange_pose=lambda: wrapper))
    with pytest.raises(RuntimeError, match="fresh"):
        cross.fresh_pose(arm)
    wrapper.timestamp = time.time()
    wrapper.msg = [float("nan")]*6
    with pytest.raises(ValueError, match="Non-finite"):
        cross.fresh_pose(arm)


class FakeSolver:
    def get_joint_limits(self):
        return np.full(7, -3.0), np.full(7, 3.0)

    def solve_ik(self, target_position, target_wxyz, initial_guess, use_previous_solution):
        result = initial_guess.copy()
        result[:3] = target_position
        result[3:6] = Rotation.from_quat(np.asarray(target_wxyz)[[1, 2, 3, 0]]).as_euler("xyz")
        return result


def fake_fk(q):
    return transform(np.asarray(q)[3:6], np.asarray(q)[:3])


def test_jog_prevalidates_small_steps_and_preserves_attitude():
    start = transform(xyz=(-0.3, 0, 0.3))
    target = transform(xyz=(-0.296, 0, 0.302))
    q = np.array([-0.3, 0, 0.3, 0, 0, 0, 0])
    plan = cross._bounded_jog_plan(start, target, q, np.eye(4), FakeSolver(), fake_fk)
    assert len(plan) == 5
    np.testing.assert_allclose(plan[-1][1], target)
    for joint, tcp in plan:
        assert np.max(np.abs(joint-q)) <= np.deg2rad(2)
        np.testing.assert_allclose(tcp[:3, :3], start[:3, :3])
        q = joint


@pytest.mark.parametrize("case", ["far", "rotation", "joint_jump", "fk_mismatch"])
def test_jog_rejects_unsafe_or_inconsistent_plan(case):
    start = transform(xyz=(-0.3, 0, 0.3))
    target = transform(xyz=(-0.299, 0, 0.3))
    q = np.array([-0.3, 0, 0.3, 0, 0, 0, 0])
    solver, fk = FakeSolver(), fake_fk
    if case == "far":
        target[0, 3] += 0.1
    elif case == "rotation":
        target[:3, :3] = Rotation.from_euler("z", 20, degrees=True).as_matrix()
    elif case == "joint_jump":
        solver.solve_ik = lambda **kw: q+1
    else:
        fk = lambda _: transform(xyz=(0, 0, 0))
    with pytest.raises(ValueError):
        cross._bounded_jog_plan(start, target, q, np.eye(4), solver, fk)


def test_supervised_jog_refuses_to_descend_beyond_prediction_without_hardware():
    current = transform(xyz=(-0.4, 0, 0.2))
    offset = np.eye(4)
    args = SimpleNamespace(execute=True, tcp_offset=[0]*6, cross_step_mm=1)
    state = dict(target_tcp_pose=np.array([-0.4, 0, 0.23, 0, 0, 0]),
                 P_base=np.array([-0.4, 0, 0.2]), cross_hover_reached=True)
    arm = SimpleNamespace(arm=SimpleNamespace(get_flange_pose=lambda:
        SimpleNamespace(msg=[-0.4, 0, 0.2, 0, 0, 0], timestamp=time.time())))
    assert cross.supervised_jog(arm, args, state, down=True) is False


def test_cross_check_cli_requires_explicit_supervision_flags(monkeypatch):
    base = ["point", "--calib-json", "test.json", "--cross-check"]
    monkeypatch.setattr(sys, "argv", base)
    with pytest.raises(SystemExit):
        point.parse_args()
    valid = base+["--execute", "--skip-initial-move", "--nero-speed-percent", "1", "--hover", "0.03"]
    monkeypatch.setattr(sys, "argv", valid)
    args = point.parse_args()
    assert args.cross_step_mm == 1 and args.orbbec_depth_median_frames == 15
    assert args.move_wait_timeout == 60
    assert args.move_pos_tol == pytest.approx(0.002)
    assert args.move_joint_tol == pytest.approx(np.deg2rad(0.5))
    assert args.move_rpy_tol == pytest.approx(np.deg2rad(1.0))
    for bad in (["--yes"], ["--cross-step-mm", "2"], ["--hover", "0"], ["--target-frame", "flange"],
                ["--move-wait-timeout", "0"], ["--move-pos-tol", "0.015"], ["--move-joint-tol", "nan"]):
        monkeypatch.setattr(sys, "argv", valid+bad)
        with pytest.raises(SystemExit):
            point.parse_args()


def test_legacy_point_cli_defaults_remain_unchanged(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["point", "--calib-json", "test.json"])
    args = point.parse_args()
    assert args.move_wait_timeout == 20
    assert args.move_pos_tol == 0.015
    assert args.move_rpy_tol == pytest.approx(np.deg2rad(5.0))
    assert args.move_joint_tol == pytest.approx(np.deg2rad(2.0))


def test_current_config_selects_may_extrinsics_and_sdk_intrinsics():
    import yaml
    root = Path(__file__).resolve().parents[1]
    camera = yaml.safe_load((root / "config/pipeline.yaml").read_text())["wrist_camera"]
    assert camera["grasp_calib_json"].endswith("2026-05-19_14-45-49_calibration.json")
    assert camera["servo_calib_json"] == camera["grasp_calib_json"]
    assert not camera.get("intrinsics_json")
    source = root / "config/2026-05-19_14-45-49_calibration.json"
    assert hashlib.sha256(source.read_bytes()).hexdigest() == "a12e269a3b3d1a97a5339605007c95feb4aea013bf687ae1cd5c481d8c0836f1"


class FakeJogArm:
    def __init__(self):
        self.q = np.array([-0.4, 0, 0.23, 0, 0, 0, 0])
        self.commands = []
        self.stops = []
        self.arm = SimpleNamespace(get_flange_pose=self._pose)

    def _pose(self):
        return SimpleNamespace(msg=self.q[:6].copy(), timestamp=time.time())

    def get_joint_angles(self):
        return self.q.copy()

    def get_joint_torques(self):
        raise AssertionError("Point jogs must not request joint torque feedback")

    def move_j(self, q):
        self.commands.append(np.asarray(q).copy())
        self.q = np.asarray(q).copy()

    def stop_for_program_exit(self, reason):
        self.stops.append(reason)


def jog_inputs():
    args = SimpleNamespace(execute=True, tcp_offset=[0]*6, cross_step_mm=1,
                           min_x=-0.8, max_x=0.8, min_y=-0.6, max_y=0.6, min_z=0.02, max_z=1.2)
    state = dict(target_tcp_pose=np.array([-0.4, 0, 0.23, 0, 0, 0]),
                 P_base=np.array([-0.4, 0, 0.2]), cross_hover_reached=True,
                 cross_kinematics=(FakeSolver(), fake_fk))
    return args, state


def test_explicit_move_confirmation_and_actual_execution_record(monkeypatch):
    arm = FakeJogArm()
    args, state = jog_inputs()
    monkeypatch.setattr("builtins.input", lambda _: "MOVE")
    assert cross.supervised_jog(arm, args, state, down=True)
    assert len(arm.commands) == 1
    np.testing.assert_allclose(arm.q[:3], [-0.4, 0, 0.229])
    np.testing.assert_allclose(state["cross_last_execution"]["actual_tcp_xyz_m"], arm.q[:3])
    assert state["cross_last_execution"]["automatic_contact_detection"] is False
    assert not arm.stops


def test_declining_confirmation_sends_no_motion(monkeypatch):
    arm = FakeJogArm()
    args, state = jog_inputs()
    monkeypatch.setattr("builtins.input", lambda _: "")
    assert cross.supervised_jog(arm, args, state, down=True) is False
    assert not arm.commands


def test_motion_exception_uses_existing_exit_stop_policy(monkeypatch):
    arm = FakeJogArm()
    args, state = jog_inputs()
    monkeypatch.setattr("builtins.input", lambda _: "MOVE")

    def pose():
        age = 2.0 if arm.commands else 0.0
        return SimpleNamespace(msg=arm.q[:6].copy(), timestamp=time.time()-age)

    arm.arm.get_flange_pose = pose
    with pytest.raises(RuntimeError, match="fresh"):
        cross.supervised_jog(arm, args, state, down=True)
    assert len(arm.commands) == 1 and len(arm.stops) == 1


def test_jog_does_not_request_sdk_motor_feedback(monkeypatch):
    arm = FakeJogArm()
    args, state = jog_inputs()

    def forbidden_motor_read(_):
        raise AssertionError("Point jogs must not request SDK motor feedback")

    arm.arm.get_motor_states = forbidden_motor_read
    monkeypatch.setattr("builtins.input", lambda _: "MOVE")
    assert cross.supervised_jog(arm, args, state, down=True)
    assert len(arm.commands) == 1 and not arm.stops


def test_latched_stop_refuses_jog_before_motion():
    arm = FakeJogArm()
    arm._safety_stop_latched = True
    args, state = jog_inputs()
    with pytest.raises(RuntimeError, match="锁存"):
        cross.supervised_jog(arm, args, state, down=True)
    assert not arm.commands and not arm.stops


def test_teaching_mode_does_not_send_control_target(monkeypatch):
    arm = FakeJogArm()
    arm.arm.get_arm_status = lambda: SimpleNamespace(msg=SimpleNamespace(ctrl_mode="TEACHING_MODE"))
    args, state = jog_inputs()
    with pytest.raises(ValueError, match="示教模式"):
        cross.supervised_jog(arm, args, state, down=True)
    assert not arm.commands
