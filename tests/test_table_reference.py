from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from real_nero_grasp.table_reference import fit_table_reference


def scene(tilt_deg=3., height=.020):
    x, y = np.meshgrid(np.linspace(-.70, -.30, 45), np.linspace(-.20, .20, 40))
    table = np.c_[x.ravel(), y.ravel(), np.zeros(x.size)]
    obstacle = np.array([[-.50, 0, .03], [-.46, .02, .06]])
    rotation = Rotation.from_euler("xy", [tilt_deg, .5], degrees=True).as_matrix()
    points = np.vstack([table, obstacle]) @ rotation.T + [0, 0, height]
    return points, len(table)


@pytest.mark.parametrize("tilt_deg", [3., 5.1])
def test_confirmed_plane_corrects_scene_rigidly_without_deleting_obstacles(tilt_deg):
    points, table_count = scene(tilt_deg)
    transform, report = fit_table_reference(points, [-.5, 0], 0.)
    corrected = points @ transform[:3, :3].T + transform[:3, 3]
    assert corrected.shape == points.shape
    np.testing.assert_allclose(corrected[:table_count, 2], 0, atol=1e-6)
    np.testing.assert_allclose(corrected[table_count:, 2], [.03, .06], atol=1e-6)
    np.testing.assert_allclose(np.linalg.norm(corrected[0] - corrected[-1]),
                               np.linalg.norm(points[0] - points[-1]), atol=1e-12)
    assert report["points_deleted_by_registration"] == 0
    assert report["physical_reference_z_m"] == 0.
    assert report["plane_fit_rms_m"] < 1e-6
    assert report["inlier_ratio"] > .99
    assert report["max_correction_tilt_deg"] == 6.


def test_nonzero_physical_reference_is_not_silently_replaced_with_zero():
    points, count = scene(height=.030)
    transform, _ = fit_table_reference(points, [-.5, 0], .015)
    corrected = points @ transform[:3, :3].T + transform[:3, 3]
    np.testing.assert_allclose(corrected[:count, 2], .015, atol=1e-6)


@pytest.mark.parametrize("tilt,height,reason", [(6.1, .020, "tilt"), (9., .020, "tilt"), (0., .050, "envelope")])
def test_unsupported_table_geometry_is_rejected(tilt, height, reason):
    with pytest.raises(ValueError, match=reason):
        fit_table_reference(scene(tilt, height)[0], [-.5, 0], 0.)


def test_insufficient_or_narrow_plane_is_rejected():
    with pytest.raises(ValueError, match="Insufficient"):
        fit_table_reference(np.zeros((10, 3)), [0, 0], 0.)
    narrow = np.c_[np.linspace(-.7, -.3, 1000), np.zeros(1000), np.zeros(1000)]
    # Avoid RANSAC on perfectly collinear points by giving a 2 mm wide strip.
    narrow[:, 1] = np.sin(np.arange(1000)) * .001
    with pytest.raises(ValueError, match="too narrow"):
        fit_table_reference(narrow, [-.5, 0], 0.)


def test_table_reference_is_opt_in_cli(monkeypatch):
    from real_nero_grasp.cli import parse_args
    monkeypatch.setattr("sys.argv", ["runner", "--calib-json", "unused.json"])
    assert parse_args().grasp_table_reference_z is None


def test_reference_requires_active_centered_observations(tmp_path):
    from real_nero_grasp.table_reference import build_table_reference
    with pytest.raises(ValueError, match="target-centered"):
        build_table_reference(SimpleNamespace(active_view=False),
                              tmp_path, np.eye(4), tmp_path / "output")


def test_multiview_requires_side_archives_instead_of_silently_using_center_only(tmp_path):
    import json
    from real_nero_grasp.table_reference import build_table_reference
    for name in ("initial", "center"):
        folder = tmp_path / "multiview_raw" / name
        folder.mkdir(parents=True)
        (folder / "metadata.json").write_text(json.dumps(dict(view_label=name)))
    args = SimpleNamespace(active_view=True, active_side_view_count=1,
                           grasp_table_reference_z=0., table_z=0.)
    with pytest.raises(ValueError, match="archived side"):
        build_table_reference(args, tmp_path, np.eye(4), tmp_path / "output")


@pytest.mark.parametrize("side_labels,side_height,expected_error", [
    ((), .034, None), (("side1",), .034, None),
    (("side1", "side1"), .034, None),
    (("side1",), .060, "Table height correction"),
])
def test_worker_registers_target_and_all_obstacles_and_preserves_raw_archives(
    tmp_path, monkeypatch, side_labels, side_height, expected_error,
):
    import hashlib
    import json
    import open3d as o3d
    from real_nero_grasp import planning_process
    from real_nero_grasp.cli import parse_args
    from real_nero_grasp.table_reference import prepare_table_referenced_scene
    monkeypatch.setattr("sys.argv", ["runner", "--calib-json", "unused.json"])
    args = parse_args()
    args.active_view, args.active_side_view_count = True, int(bool(side_labels))
    args.grasp_table_reference_z = args.table_z = 0.
    args.save_dir = str(tmp_path / "run")
    tx, ty = np.meshgrid(np.linspace(-.70, -.30, 45), np.linspace(-.20, .20, 40))
    table = np.c_[tx.ravel(), ty.ravel(), np.zeros(tx.size)]
    gx, gy, gz = np.meshgrid(np.linspace(-.43, -.40, 8), np.linspace(.08, .11, 8),
                             np.linspace(.04, .07, 8))
    cube = np.c_[gx.ravel(), gy.ravel(), gz.ravel()]
    target = np.c_[np.linspace(-.52, -.48, 200), np.sin(np.arange(200))*.01,
                   .025 + np.cos(np.arange(200))*.005]
    original_files = []
    source_target = source_scene = None
    def cloud(points):
        return o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    observations = [("initial", 1., .015), ("center", 5.1, .022)]
    observations += [(name, 3.5 - i, side_height + .002 * i) for i, name in enumerate(side_labels)]
    for index, (name, angle, offset) in enumerate(observations):
        folder = tmp_path / "capture" / "multiview_raw" / f"{name}_{index}"
        folder.mkdir(parents=True)
        metadata = folder / "metadata.json"
        metadata.write_text(json.dumps(dict(
            view_label=name,
            status=dict(fusion_used=name != "initial" and not (len(side_labels) > 1 and index == len(observations) - 1)),
        )))
        original_files.append((metadata, hashlib.sha256(metadata.read_bytes()).hexdigest()))
        rotation = Rotation.from_euler("x", angle, degrees=True).as_matrix()
        view_cube = cube + ([.01 * (index - 2), -.10, .020 * (index - 1)]
                            if name.startswith("side") else [0, 0, 0])
        for kind, truth in (("scene", np.vstack([table, view_cube, target])),
                             ("target", target), ("obstacle", np.vstack([table, view_cube]))):
            points = truth @ rotation.T + [0, 0, offset]
            path = folder / f"{kind}_base.ply"
            o3d.io.write_point_cloud(str(path), cloud(points))
            original_files.append((path, hashlib.sha256(path.read_bytes()).hexdigest()))
            if name == "center":
                if kind == "target": source_target = cloud(points)
                if kind == "scene": source_scene = cloud(points)
    # The caller's fused target already passed its own view gates. Its local
    # coordinates and side-view contribution must survive absolute registration.
    expected_z = target[:, 2]
    if side_labels:
        source_target += cloud(np.asarray(source_target.points)[:40] + [.01, 0, 0])
        expected_z = np.r_[expected_z, target[:40, 2]]
    local_target = np.asarray(source_target.points).copy()
    planning_process.close_planning_process()
    try:
        if expected_error:
            from real_nero_grasp.types import PlanNotExecutable
            with pytest.raises(PlanNotExecutable, match=expected_error):
                prepare_table_referenced_scene(args, source_target, source_scene, np.eye(4), tmp_path / "capture")
            return
        result = prepare_table_referenced_scene(args, source_target, source_scene, np.eye(4), tmp_path / "capture")
        target_out, _, transform, motion_scene, _, environment = result
        assert target_out is source_target
        measured_target = np.asarray(target_out.points) @ transform[:3, :3].T + transform[:3, 3]
        np.testing.assert_array_equal(np.asarray(target_out.points), local_target)
        np.testing.assert_allclose(measured_target[:, 2], expected_z, atol=1e-6)
        assert len(motion_scene.points) == len(observations) * (len(table) + len(cube) + len(target))
        assert environment.source_view_names[:2] == ("initial", "center")
        assert len(set(environment.source_view_names)) == len(observations)
        assert np.max(environment.hard_environment_base[:, 2]) > .065  # the cube remains
        if side_labels:
            assert np.max(environment.hard_environment_base[:, 2]) > .085  # side-only obstacle remains
        if len(side_labels) > 1:
            assert np.max(environment.hard_environment_base[:, 2]) > .105  # rejected retry's obstacle remains
        for path, digest in original_files:
            assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
        assert (tmp_path / "capture/table_reference/table_reference_report.json").is_file()
        report = json.loads((tmp_path / "capture/table_reference/table_reference_report.json").read_text())
        for view in report["views"][2:]:
            assert view["center_anchor_applied"]
            assert view["height_at_target_m"] > .030
            assert abs(view["residual_height_before_m"]) < .015
    finally:
        planning_process.close_planning_process()
