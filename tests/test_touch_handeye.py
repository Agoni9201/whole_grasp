import ast
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from tools import calibrate_realsense_touch as touch


def transform(rotation, position):
    result = np.eye(4)
    result[:3, :3] = Rotation.from_euler("xyz", rotation).as_matrix()
    result[:3, 3] = position
    return result


def dump(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


@pytest.fixture
def scene(tmp_path):
    points = np.array([
        [-0.52, -0.10, 0.04], [-0.40, -0.10, 0.04], [-0.28, -0.10, 0.07],
        [-0.52, 0.10, 0.07], [-0.40, 0.10, 0.04], [-0.28, 0.10, 0.07],
        [-0.46, 0.02, 0.05], [-0.34, 0.02, 0.08],
    ])
    reference_path = tmp_path / "refs" / "references.json"
    references = dict(
        kind="tcp_touch_references", schema_version=1, unit="metre", complete=True,
        tcp_offset_m=touch.TCP_OFFSET_M,
        points=[dict(point_id=str(i+1), role="fit" if i < 6 else "validation", P_base_m=p.tolist())
                for i, p in enumerate(points)],
    )
    dump(reference_path, references)
    x = transform([0.2, -0.4, 0.8], [0.06, -0.08, 0.015])
    profile = dict(
        serial="243622072771", width=1280, height=720, fps=30,
        frame=touch.CAMERA_FRAME, projection=touch.PROJECTION,
        K=[[910, 0, 630], [0, 910, 380], [0, 0, 1]], D=[0]*5,
    )

    def make_view(name, angle=0.0, identities=None):
        base_camera = transform([np.pi + angle, 0.08, 0.02], [-0.4, 0.01, 0.55])
        flange = base_camera @ np.linalg.inv(x)
        camera_points = touch.transform_points(np.linalg.inv(base_camera), points)
        assert np.all(camera_points[:, 2] > 0)
        path = tmp_path / name / "observations.json"
        path.parent.mkdir()
        (path.parent / "color.png").write_bytes(name.encode("ascii"))
        view = dict(
            kind="tcp_touch_observations", schema_version=1, unit="metre", complete=True,
            references_sha256=touch.sha256(reference_path), capture_id=name,
            image_sha256=touch.sha256(path.parent / "color.png"),
            T_base_flange=flange.tolist(), camera=profile,
            points=[dict(point_id=str(i+1), P_camera_m=p.tolist())
                    for i, p in enumerate(camera_points)
                    if identities is None or str(i+1) in identities],
        )
        dump(path, view)
        return path

    return SimpleNamespace(
        root=tmp_path, refs=reference_path, references=references,
        points=points, x=x, profile=profile, make_view=make_view,
    )


def test_recover_exact_handeye_with_nonidentity_wrist_pose(scene):
    first, second = scene.make_view("one"), scene.make_view("two", 0.23)
    calibration = touch.solve_calibration(scene.refs, [first, second])
    np.testing.assert_allclose(calibration["matrix_4x4"], scene.x, atol=1e-12)
    assert calibration["report"]["summary"]["fit"]["max_mm"] < 1e-8
    assert calibration["report"]["summary"]["validation"]["max_mm"] < 1e-8
    assert calibration["fit_point_ids"] == list("123456")
    assert calibration["validation_point_ids"] == ["7", "8"]
    assert calibration["status"] == "candidate_not_approved_for_motion"
    assert calibration["method"] == "rigid_3d3d_scale_fixed_one"


def test_held_out_points_never_affect_solution(scene):
    scene.references["points"][6]["P_base_m"][0] += 0.1
    scene.references["points"][7]["P_base_m"][1] -= 0.08
    dump(scene.refs, scene.references)
    calibration = touch.solve_calibration(scene.refs, [scene.make_view("one")])
    np.testing.assert_allclose(calibration["matrix_4x4"], scene.x, atol=1e-12)
    assert calibration["report"]["summary"]["fit"]["max_mm"] < 1e-8
    assert calibration["report"]["summary"]["validation"]["max_mm"] == pytest.approx(100)


def test_independent_validation_and_cross_view_spread(scene):
    training = scene.make_view("train")
    path = scene.root / "calibration.json"
    touch.write_json(path, touch.solve_calibration(scene.refs, [training]))
    left = scene.make_view("left", 0.15, {"7", "8"})
    right = scene.make_view("right", -0.15, {"7", "8"})
    result = touch.validate_calibration(scene.refs, [left, right], path)
    assert result["independent_images"]
    assert result["report"]["summary"]["fit"] is None
    assert result["report"]["summary"]["validation"]["max_mm"] < 1e-8
    assert result["report"]["cross_view_spread"]["7"]["max_pair_distance_mm"] < 1e-8


def test_reject_reusing_training_image_for_validation(scene):
    path = scene.make_view("train")
    calibration_path = scene.root / "calibration.json"
    touch.write_json(calibration_path, touch.solve_calibration(scene.refs, [path]))
    with pytest.raises(ValueError, match="independent captures"):
        touch.validate_calibration(scene.refs, [path], calibration_path)


def test_reject_same_image_with_a_different_capture_id(scene):
    train = scene.make_view("train")
    calibration_path = scene.root / "calibration.json"
    calibration = touch.solve_calibration(scene.refs, [train])
    touch.write_json(calibration_path, calibration)
    other = scene.make_view("other", 0.2)
    data = json.loads(other.read_text())
    (other.parent / "color.png").write_bytes((train.parent / "color.png").read_bytes())
    data["image_sha256"] = touch.sha256(other.parent / "color.png")
    dump(other, data)
    with pytest.raises(ValueError, match="independent captures"):
        touch.validate_calibration(scene.refs, [other], calibration_path)


def test_partial_validation_is_not_full_validation(scene):
    train = scene.make_view("train")
    calibration_path = scene.root / "calibration.json"
    touch.write_json(calibration_path, touch.solve_calibration(scene.refs, [train]))
    other = scene.make_view("other", 0.2, {"7"})
    with pytest.raises(ValueError, match="two held-out"):
        touch.validate_calibration(scene.refs, [other], calibration_path)


@pytest.mark.parametrize("mutation, message", [
    (lambda v: v.update(complete=False), "Incomplete"),
    (lambda v: v.update(references_sha256="wrong"), "different touch"),
    (lambda v: v["points"].append(v["points"][0]), "duplicate"),
    (lambda v: v["points"][0].update(point_id="unknown"), "Unknown"),
    (lambda v: v["points"][0].update(P_camera_m=[0, 0, -1]), "positive depth"),
    (lambda v: v["points"][0].update(P_camera_m=[float("nan"), 0, 1]), "finite XYZ"),
    (lambda v: v["camera"].update(frame="depth_optical"), "camera profile"),
    (lambda v: v["camera"].update(D=[0.1, 0, 0, 0, 0]), "K/D"),
])
def test_bad_observations_fail_closed(scene, mutation, message):
    path = scene.make_view("one")
    data = json.loads(path.read_text())
    mutation(data)
    dump(path, data)
    with pytest.raises(ValueError, match=message):
        touch.solve_calibration(scene.refs, [path])


def test_reject_missing_fit_point(scene):
    path = scene.make_view("one", identities=set("2345678"))
    with pytest.raises(ValueError, match="all fit points"):
        touch.solve_calibration(scene.refs, [path])


def test_reject_duplicate_capture_and_tampered_raw_image(scene):
    path = scene.make_view("one")
    with pytest.raises(ValueError, match="Duplicate"):
        touch.solve_calibration(scene.refs, [path, path])
    (path.parent / "color.png").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        touch.solve_calibration(scene.refs, [path])


@pytest.mark.parametrize("key, value", [("serial", "other"), ("width", 1920), ("fps", 15)])
def test_reject_mixed_camera_profiles(scene, key, value):
    one, two = scene.make_view("one"), scene.make_view("two", 0.2)
    data = json.loads(two.read_text())
    data["camera"][key] = value
    dump(two, data)
    with pytest.raises(ValueError, match="mismatch"):
        touch.solve_calibration(scene.refs, [one, two])


def test_legacy_loader_reads_new_output(scene):
    from tools.verify_handeye_point import load_calibration

    path = scene.root / "calibration.json"
    touch.write_json(path, touch.solve_calibration(scene.refs, [scene.make_view("one")]))
    position, quaternion = load_calibration(path)
    np.testing.assert_allclose(position, scene.x[:3, 3], atol=1e-12)
    np.testing.assert_allclose(Rotation.from_quat(quaternion).as_matrix(), scene.x[:3, :3], atol=1e-12)


def test_wrong_matrix_alias_cannot_be_used(scene):
    path = scene.root / "calibration.json"
    calibration = touch.solve_calibration(scene.refs, [scene.make_view("one")])
    calibration["position"][0] += 0.05
    touch.write_json(path, calibration)
    with pytest.raises(ValueError, match="disagree"):
        touch.load_calibration(path, scene.refs)


def test_planar_points_are_allowed_but_collinear_clustered_and_nonfinite_are_not():
    planar = np.array([[0, 0, 0], [0.1, 0, 0], [0, 0.1, 0], [0.1, 0.1, 0]])
    estimated = touch.fit_rigid(planar, planar + [0.2, -0.1, 0.3])
    np.testing.assert_allclose(estimated[:3, 3], [0.2, -0.1, 0.3])
    for points in (planar * 0.001, np.array([[0, 0, 0], [0.1, 0, 0], [0.2, 0, 0]]), planar * np.nan):
        with pytest.raises(ValueError):
            touch.fit_rigid(points, points)


def test_fit_never_silently_corrects_scale_or_returns_reflection(scene):
    source = scene.points.copy()
    result = touch.fit_rigid(source * 1.1, source)
    assert np.linalg.det(result[:3, :3]) == pytest.approx(1)
    assert np.linalg.norm(touch.transform_points(result, source * 1.1) - source, axis=1).max() > 0.01
    reflected = source * [-1, 1, 1]
    result = touch.fit_rigid(source, reflected)
    assert np.linalg.det(result[:3, :3]) == pytest.approx(1)


def test_known_tcp_is_rotated_from_flange_not_added_in_base_axes():
    pose = transform([0, 0, np.pi/2], [0.1, 0.2, 0.3])
    np.testing.assert_allclose(touch.tcp_position(pose), [0.1, 0.365, 0.275], atol=1e-12)
    hover = touch.hover_flange_pose([0.3, 0.1, 0.05], pose, touch.TCP_OFFSET_M, 0.03)
    np.testing.assert_allclose(touch.tcp_position(hover), [0.3, 0.1, 0.08], atol=1e-12)
    np.testing.assert_allclose(hover[:3, :3], pose[:3, :3])
    with pytest.raises(ValueError, match="30 mm"):
        touch.hover_flange_pose([0.3, 0.1, 0.05], pose, touch.TCP_OFFSET_M, 0.001)


def test_sample_depth_matches_production_pinhole_and_tolerates_a_small_hole():
    depth = np.full((30, 40), 0.5)
    depth[15, 20] = 0
    k = np.array([[100, 0, 15], [0, 100, 12], [0, 0, 1]])
    point, stats = touch.sample_depth_point(depth, k, [20, 15])
    np.testing.assert_allclose(point, [0.025, 0.015, 0.5])
    assert stats["valid_pixels"] == 24


@pytest.mark.parametrize("case", ["outside", "border", "holes", "nan", "edge", "far"])
def test_bad_depth_is_rejected(case):
    depth = np.full((30, 40), 0.5)
    k = np.array([[100, 0, 15], [0, 100, 12], [0, 0, 1]])
    pixel = [20, 15]
    if case == "outside":
        pixel = [41, 15]
    elif case == "border":
        pixel = [0, 0]
    elif case == "holes":
        depth[13:18, 18:23] = 0
    elif case == "nan":
        depth[13:18, 18:23] = np.nan
    elif case == "edge":
        depth[13:18, 18:20] += 0.03
    elif case == "far":
        depth[:] = 3
    with pytest.raises(ValueError):
        touch.sample_depth_point(depth, k, pixel)


def test_motion_during_feedback_or_rotation_is_rejected():
    identity = np.eye(4)
    touch.check_stationary([identity, identity.copy()])
    moved = identity.copy()
    moved[0, 3] = 0.002
    with pytest.raises(ValueError, match="Wrist moved"):
        touch.check_stationary([identity, moved])
    with pytest.raises(ValueError, match="Wrist moved"):
        touch.check_stationary([identity, transform([0, 0, 0.01], [0, 0, 0])])


def test_repeated_feedback_does_not_count_as_new_sample(monkeypatch):
    reader = SimpleNamespace(read=lambda: (np.zeros(6), np.eye(4), 1.0))
    with pytest.raises(ValueError, match="did not advance"):
        touch.stationary_feedback(reader, count=2, timeout=0.01)


class FakeReader:
    def __init__(self, pose=None):
        self.pose = np.eye(4) if pose is None else pose
        self.stamp = 0

    def read(self):
        self.stamp += 1
        return np.zeros(6), self.pose.copy(), float(self.stamp)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass


def test_rgbd_sampling_uses_fresh_aligned_frames_and_stationary_bracketing(monkeypatch):
    monkeypatch.setattr(touch.time, "sleep", lambda _: None)
    reader = FakeReader()
    sequence = 0

    def read():
        nonlocal sequence
        sequence += 1
        return SimpleNamespace(
            sequence=sequence, bgr=np.zeros((30, 40, 3), np.uint8),
            depth_m=np.full((30, 40), 0.5), K=np.array([[100, 0, 20], [0, 100, 15], [0, 0, 1]]),
            D=np.zeros(5), serial="test", hardware_ms=float(sequence), received_monotonic=float(sequence),
        )

    args = SimpleNamespace(depth_frames=3, fps=30)
    _, depth, profile, pose, meta = touch.capture_still_rgbd(SimpleNamespace(read=read), reader, args)
    assert meta["depth_sequences"] == [11, 12, 13]
    assert profile["projection"] == touch.PROJECTION
    np.testing.assert_allclose(depth, 0.5)
    np.testing.assert_allclose(pose, np.eye(4))


def test_cli_offline_commands_do_not_connect_hardware_and_never_overwrite(scene, monkeypatch):
    def fail(*_):
        pytest.fail("offline command attempted hardware access")

    monkeypatch.setattr(touch, "read_only_reader", fail)
    training = scene.make_view("train")
    path = scene.root / "candidate.json"
    command = ["solve", "--references", str(scene.refs), "--observations", str(training), "--output", str(path)]
    assert touch.main(command) == 0
    original = path.read_bytes()
    with pytest.raises(SystemExit):
        touch.main(command)
    assert path.read_bytes() == original
    validation = scene.make_view("validation", 0.3)
    assert touch.main([
        "validate", "--references", str(scene.refs), "--observations", str(validation),
        "--calibration", str(path), "--output", str(scene.root / "validation.json"),
    ]) == 0


def test_new_tool_has_no_motion_enable_mode_or_stop_calls():
    source = Path(touch.__file__).read_text(encoding="utf-8")
    forbidden = {"enable", "disable", "move_p", "move_j", "mit_ctrl", "set_mode", "set_tcp_offset",
                 "electronic_emergency_stop", "set_speed_percent"}
    calls = [node.func.attr for node in ast.walk(ast.parse(source))
             if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)]
    assert forbidden.isdisjoint(calls)


def test_physical_verify_records_actual_manual_touch_without_motion(scene, monkeypatch):
    training = scene.make_view("train")
    calibration = scene.root / "candidate.json"
    touch.write_json(calibration, touch.solve_calibration(scene.refs, [training]))
    observation = json.loads(scene.make_view("verify", 0.2, {"7"}).read_text())
    contact_pose = np.eye(4)
    contact_pose[:3, 3] = scene.points[6] - touch.TCP_OFFSET_M
    monkeypatch.setattr(touch, "read_only_reader", lambda _: FakeReader(contact_pose))
    monkeypatch.setattr(touch.time, "sleep", lambda _: None)

    def capture(args, *unused):
        args.output.mkdir()
        return observation

    monkeypatch.setattr(touch, "collect_observation", capture)
    monkeypatch.setattr("builtins.input", lambda _: "TOUCH")
    args = SimpleNamespace(
        references=scene.refs, calibration=calibration, point_id="7", hover_mm=30,
        output=scene.root / "physical",
    )
    touch.run_verify(args)
    result = json.loads((args.output / "verification.json").read_text())
    assert result["physical_touch_confirmed"] is True
    assert result["motion_sent"] is False
    assert result["hover_collision_checked"] is False
    assert result["visual_minus_actual_touch_norm_mm"] < 1e-8
    np.testing.assert_allclose(result["retouch_minus_reference_xyz_mm"], 0, atol=1e-8)


def test_verify_rejects_a_fitting_point_before_hardware_access(scene, monkeypatch):
    training = scene.make_view("train")
    calibration = scene.root / "candidate.json"
    touch.write_json(calibration, touch.solve_calibration(scene.refs, [training]))
    monkeypatch.setattr(touch, "read_only_reader", lambda _: pytest.fail("hardware accessed"))
    with pytest.raises(ValueError, match="held-out"):
        touch.run_verify(SimpleNamespace(references=scene.refs, calibration=calibration, point_id="1"))


def test_cancelled_touch_keeps_partial_reference_but_cannot_be_solved(tmp_path, monkeypatch):
    answers = iter(["", "q"])
    monkeypatch.setattr("builtins.input", lambda _: next(answers))
    monkeypatch.setattr(touch, "read_only_reader", lambda _: FakeReader())
    monkeypatch.setattr(touch.time, "sleep", lambda _: None)
    output = tmp_path / "partial_touch"
    assert touch.main(["touch", "--output", str(output)]) == 130
    data = json.loads((output / "references.json").read_text())
    assert not data["complete"]
    assert len(data["points"]) == 1
    np.testing.assert_allclose(data["points"][0]["P_base_m"], touch.TCP_OFFSET_M)
    with pytest.raises(ValueError, match="incomplete"):
        touch.load_references(output / "references.json")


def test_read_only_reader_only_constructs_existing_read_only_adapter(monkeypatch):
    from tools import calibrate_d435i_charuco as charuco

    called = []

    class Adapter:
        def __init__(self, *args):
            called.append(args)

    monkeypatch.setattr(charuco, "NeroReadOnlyFeedback", Adapter)
    args = SimpleNamespace(channel="can0", firmware="V111", feedback_timeout=5)
    assert isinstance(touch.read_only_reader(args), Adapter)
    assert called == [("can0", "V111", 5)]


def test_no_execute_flag_and_nonfinite_options_are_rejected_before_hardware(tmp_path, monkeypatch):
    monkeypatch.setattr(touch, "read_only_reader", lambda _: pytest.fail("hardware accessed"))
    with pytest.raises(SystemExit):
        touch.main(["touch", "--output", str(tmp_path / "new"), "--execute"])
    with pytest.raises(SystemExit):
        touch.main(["touch", "--output", str(tmp_path / "new"), "--feedback-timeout", "nan"])
    assert not (tmp_path / "new").exists()


def test_same_camera_rejects_experimental_changed_intrinsics(scene):
    changed = dict(scene.profile, K=[[903, 0, 632], [0, 903, 385], [0, 0, 1]])
    with pytest.raises(ValueError, match="K/D mismatch"):
        touch.same_camera(scene.profile, changed)


def test_post_frame_feedback_can_wait_for_an_actual_new_timestamp():
    reader = FakeReader()
    reader.stamp = 1
    result = touch.feedback_after(reader, 2.0)
    assert result[2] == 3.0
    stale = SimpleNamespace(read=lambda: (np.zeros(6), np.eye(4), 2.0))
    with pytest.raises(ValueError, match="did not advance"):
        touch.feedback_after(stale, 2.0, timeout=0.01)
