"""Bad self depth must disappear without erasing a neighbouring obstacle."""
import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from real_nero_grasp.camera_self_mask import calibrated_self_mask, validate_self_mask_settings
from real_nero_grasp.perception import apply_camera_self_occlusion_mask, capture_rgbd_xyz_frame


CALIBRATION = Path(__file__).resolve().parents[1] / "config/2026-10-26_camera_self_mask.json"
MULTIFRAME = CALIBRATION.with_name("2026-10-26_camera_self_mask_100mm_multiframe.json")


@pytest.fixture
def capture(tmp_path):
    calibration = json.loads(CALIBRATION.read_text())
    profile = calibration["profile"]
    metadata = dict(serial=profile["serial"], width=1920, height=1080,
                    native_depth_width=1280, native_depth_height=720,
                    intrinsics=calibration["intrinsics"])
    (tmp_path / "capture_meta.json").write_text(json.dumps(metadata))
    image = np.full((1080, 1920, 3), 150, np.uint8)
    cv2.fillPoly(image, [np.array(p, np.int32) for p in calibration["polygons_px"]], (15, 20, 18))
    # Two real obstacles, including a dark object, outside the measured finger.
    image[900:930, 1000:1030] = 10
    image[740:820, 980:1070] = (180, 70, 10)
    args = SimpleNamespace(camera_self_mask=True, camera_self_mask_calibration=str(CALIBRATION),
                           camera_serial=profile["serial"], camera_width=1920, camera_height=1080,
                           camera_depth_width=1280, camera_depth_height=720,
                           gripper_open_width=.085, use_registered_points=False)
    return image, metadata, args, tmp_path


def test_known_bad_self_depth_removed_neighbouring_obstacles_and_inputs_preserved(capture):
    image, meta, args, directory = capture
    xyz = np.zeros((1080, 1920, 3), np.float32)
    xyz[:, :, 2] = .5
    xyz[1020, 832, 2] = .277  # Candidate 6: wrong depth, far from physical robot.
    xyz[1074, 782, 2] = .295  # Candidate 9.
    before = xyz.copy()
    filtered = apply_camera_self_occlusion_mask(xyz, image, args, directory)
    assert np.isnan(filtered[1020, 832]).all()
    assert np.isnan(filtered[1074, 782]).all()
    for x, y in [(1010, 910), (1000, 780), (765, 986), (1335, 965)]:
        np.testing.assert_array_equal(filtered[y, x], xyz[y, x])
    mask, _ = calibrated_self_mask(image, CALIBRATION, meta)
    np.testing.assert_array_equal(filtered[mask == 0], before[mask == 0])
    np.testing.assert_array_equal(xyz, before)
    report = json.loads((directory / "camera_self_mask_report.json").read_text())
    assert report["removed_valid_depth"] > 25000
    assert report["calibration"]["profile"]["serial"] == args.camera_serial


@pytest.mark.parametrize("change", ["serial", "profile", "intrinsics", "bright", "shift", "dark_occluder"])
def test_mismatched_capture_aborts_instead_of_using_legacy_or_rescaling(capture, change):
    image, meta, args, directory = capture
    if change == "serial":
        meta["serial"] = "different-camera"
    elif change == "profile":
        meta["native_depth_width"] = 848
    elif change == "intrinsics":
        meta["intrinsics"]["cx"] += 5
    elif change == "bright":
        image[850:, 730:950] = 150
    elif change == "shift":
        image = np.roll(image, 40, axis=1)
    else:
        image[820:, 700:975] = 10
    with pytest.raises(ValueError, match="self-mask"):
        calibrated_self_mask(image, CALIBRATION, meta)
    assert not (directory / "camera_self_mask.png").exists()


def test_black_cable_joining_finger_does_not_grow_mask(capture):
    image, meta, _, _ = capture
    # Thin cable touches the finger, but the mask cannot follow it to the left.
    cv2.line(image, (900, 1010), (600, 900), (12, 12, 12), 4)
    mask, _ = calibrated_self_mask(image, CALIBRATION, meta)
    assert mask[1010, 900] > 0
    assert mask[900, 600] == 0
    assert mask[955, 750] == 0


def test_failed_silhouette_saves_diagnostics_without_masking_xyz(capture):
    from real_nero_grasp.camera_self_mask import CameraSelfMaskMismatch
    image, _, args, directory = capture
    image = np.roll(image, 40, axis=1)
    xyz = np.full((*image.shape[:2], 3), .3, np.float32)
    before = xyz.copy()
    with pytest.raises(CameraSelfMaskMismatch, match="Diagnostics:"):
        apply_camera_self_occlusion_mask(xyz, image, args, directory)
    np.testing.assert_array_equal(xyz, before)
    assert (directory / "camera_self_mask_mismatch.png").is_file()
    report = json.loads((directory / "camera_self_mask_report.json").read_text())
    assert report["validation_passed"] is False
    assert not (directory / "camera_self_mask.png").exists()


def test_runtime_reports_expected_mask_rejection_without_traceback(monkeypatch, capsys):
    from real_nero_grasp import runtime
    from real_nero_grasp.camera_self_mask import CameraSelfMaskMismatch
    import cameras.preflight
    args = SimpleNamespace(phase="grasp", _experiment_log_state=object())
    recorded = {}
    monkeypatch.setattr(runtime, "parse_args", lambda: args)
    monkeypatch.setattr(runtime, "initialize_experiment_log", lambda *a: None)
    monkeypatch.setattr(runtime, "mark_experiment_stage", lambda *a: None)
    monkeypatch.setattr(cameras.preflight, "grasp_preflight", lambda *a: None)
    def reject(_):
        raise CameraSelfMaskMismatch("synthetic silhouette mismatch", {})
    monkeypatch.setattr(runtime, "run_single_grasp", reject)
    monkeypatch.setattr(runtime, "finalize_experiment_log", lambda *a, **kw: recorded.update(kw))
    assert runtime.main() == 2
    assert recorded["terminal_status"] == "capture_failed"
    output = capsys.readouterr().out
    assert "[capture][abort]" in output
    assert "Traceback" not in output


def test_known_configuration_mismatch_rejected_before_arm_preflight(capture):
    _, _, args, _ = capture
    validate_self_mask_settings(args)
    args.camera_width = 848
    from cameras.preflight import grasp_preflight
    args.execute = True
    with pytest.raises(ValueError, match="width mismatch"):
        grasp_preflight(args)  # Must fail before reading a handeye or opening SDKs.


def test_capture_masks_xyz_before_any_pointcloud_sampling(capture):
    image, _, args, directory = capture
    def capture_rgbd(_):
        return image, np.full(image.shape[:2], .3, np.float32), object(), str(directory)
    def depth_xyz(depth, _):
        result = np.zeros((*depth.shape, 3), np.float32)
        result[:, :, 2] = depth
        return result
    _, xyz, _, _ = capture_rgbd_xyz_frame(args, capture_rgbd_hook=capture_rgbd,
                                         depth_to_xyz_image_hook=depth_xyz)
    assert np.isnan(xyz[1020, 832]).all()
    assert xyz[910, 1010, 2] == np.float32(.3)


def test_pipeline_enables_current_calibration_and_cli_can_override(tmp_path):
    import argparse
    from cameras.options import add_camera_arguments, apply_pipeline_defaults
    parser = argparse.ArgumentParser()
    add_camera_arguments(parser)
    parser.add_argument("--camera-self-mask", dest="camera_self_mask", action="store_true")
    parser.add_argument("--no-camera-self-mask", dest="camera_self_mask", action="store_false")
    parser.add_argument("--camera-self-mask-calibration")
    config = CALIBRATION.parent / "pipeline.yaml"
    argv = ["--config", str(config)]
    apply_pipeline_defaults(parser, argv)
    args = parser.parse_args(argv)
    assert args.camera_self_mask
    assert Path(args.camera_self_mask_calibration) == MULTIFRAME
    validate_self_mask_settings(args)
    assert not parser.parse_args([*argv, "--no-camera-self-mask"]).camera_self_mask


@pytest.mark.parametrize("draw_envelope", [False, True])
def test_multiframe_edge_variation_is_bounded_and_never_masks_outside_envelope(draw_envelope):
    calibration = json.loads(MULTIFRAME.read_text())
    image = np.full((1080, 1920, 3), 150, np.uint8)
    envelope = np.zeros(image.shape[:2], np.uint8)
    cv2.fillPoly(envelope, [np.asarray(p, np.int32) for p in calibration['polygons_px']], 1)
    stable = np.zeros_like(envelope)
    cv2.fillPoly(stable, [np.asarray(p, np.int32) for p in calibration['validation_polygons_px']], 1)
    image[(envelope if draw_envelope else stable) > 0] = 10
    # A neighbouring cable connects to the finger but must not grow the mask.
    cv2.line(image, (800, 1020), (450, 950), (10, 10, 10), 2)
    profile = calibration['profile']
    meta = dict(serial=profile['serial'], width=1920, height=1080,
                native_depth_width=1280, native_depth_height=720,
                intrinsics=calibration['intrinsics'])
    mask, report = calibrated_self_mask(image, MULTIFRAME, meta)
    assert report['validation_passed']
    assert not mask[envelope == 0].any()
    assert mask[950, 450] == 0
    if not draw_envelope:
        assert not mask[(envelope > 0) & (stable == 0) & (cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) > 80)].any()


@pytest.mark.parametrize("change", ['shift', 'wrong_opening', 'foreign_occlusion'])
def test_multiframe_mask_still_rejects_installation_opening_and_occlusion_changes(change):
    calibration = json.loads(MULTIFRAME.read_text())
    image = np.full((1080, 1920, 3), 150, np.uint8)
    cv2.fillPoly(image, [np.asarray(p, np.int32) for p in calibration['polygons_px']], (10, 10, 10))
    if change == 'shift':
        image = np.roll(image, 40, axis=1)
    elif change == 'wrong_opening':
        image[:] = 150
        other = json.loads(CALIBRATION.read_text())
        cv2.fillPoly(image, [np.asarray(p, np.int32) for p in other['polygons_px']], (10, 10, 10))
    else:
        image[820:, 550:920] = 10
    profile = calibration['profile']
    meta = dict(serial=profile['serial'], width=1920, height=1080,
                native_depth_width=1280, native_depth_height=720,
                intrinsics=calibration['intrinsics'])
    with pytest.raises(ValueError, match='self-mask'):
        calibrated_self_mask(image, MULTIFRAME, meta)


def test_colored_dark_target_edge_inside_envelope_is_preserved():
    calibration = json.loads(MULTIFRAME.read_text())
    image = np.full((1080, 1920, 3), 150, np.uint8)
    cv2.fillPoly(image, [np.asarray(p, np.int32) for p in calibration['polygons_px']], (10, 10, 10))
    # The actual failure frame has yellow/brown target-edge pixels at the tip.
    # They meet the grey threshold but not the measured neutral-finger colour.
    image[900, 800] = [20, 60, 70]
    profile = calibration['profile']
    meta = dict(serial=profile['serial'], width=1920, height=1080,
                native_depth_width=1280, native_depth_height=720,
                intrinsics=calibration['intrinsics'])
    mask, report = calibrated_self_mask(image, MULTIFRAME, meta)
    assert mask[900, 800] == 0
    assert report['colored_pixels_preserved'] >= 1


@pytest.mark.parametrize('outside_color', [(30, 50, 70), (10, 10, 10)])
def test_outer_guard_distinguishes_colored_obstacle_from_black_finger_extension(outside_color):
    calibration = json.loads(MULTIFRAME.read_text())
    image = np.full((1080,1920,3),150,np.uint8)
    envelope = np.zeros(image.shape[:2],np.uint8)
    cv2.fillPoly(envelope,[np.asarray(p,np.int32) for p in calibration['polygons_px']],1)
    image[envelope>0] = 10
    profile = calibration['profile']
    meta = dict(serial=profile['serial'],width=1920,height=1080,
                native_depth_width=1280,native_depth_height=720,
                intrinsics=calibration['intrinsics'])
    baseline, _ = calibrated_self_mask(image,MULTIFRAME,meta)
    outer = (cv2.dilate(envelope,np.ones((17,17),np.uint8))>0) & ~(
        cv2.dilate(envelope,np.ones((5,5),np.uint8))>0)
    image[outer] = outside_color
    if outside_color == (10,10,10):
        # Core and silhouette still match; the outer guard must catch a black
        # extension even though a coloured object in that same ring is valid.
        with pytest.raises(ValueError,match='outer=1.000'):
            calibrated_self_mask(image,MULTIFRAME,meta)
    else:
        mask, report = calibrated_self_mask(image,MULTIFRAME,meta)
        assert report['validation_passed']
        assert report['outer_gray_dark_fraction']==1.
        assert report['outer_dark_fraction']==0.
        assert report['outer_colored_dark_pixels']==int(outer.sum())
        np.testing.assert_array_equal(mask,baseline)
        assert not mask[outer].any()


def test_calibrated_finger_mask_preserves_geometric_self_filter_and_target(capture):
    from real_nero_grasp.active_motion import _prepare_active_route_safety_scene
    from unittest.mock import Mock
    _, _, args, _ = capture
    target = np.array([[.4, .1, .1]], np.float32)
    obstacle = np.array([[.5, .2, .1]], np.float32)
    # Model a geometric filter that also removes a nearby target point. The
    # segmented target must be restored as occupied space after self filtering.
    self_filter = Mock(return_value=obstacle)
    _, full_scene, _ = _prepare_active_route_safety_scene(
        np.zeros(7), np.vstack([target, obstacle]), target, args, "test",
        bounded_motion_points_hook=lambda p, _: np.asarray(p),
        filter_motion_robot_self_points_hook=self_filter)
    self_filter.assert_called_once()
    assert any(np.array_equal(point, target[0]) for point in full_scene)
    assert any(np.array_equal(point, obstacle[0]) for point in full_scene)
