import json
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from tools import calibrate_d435i_charuco as calibration
from tools.charuco_intrinsics import (
    check_intrinsics, estimate_intrinsics, load_intrinsics,
    sha256, validate_intrinsics, write_new_json,
)


PROFILE = {"serial": "test", "width": 1280, "height": 720,
           "frame": "d435_color_optical_frame"}
TRUE_K = np.array([[902., 0., 626.], [0., 904., 380.], [0., 0., 1.]])
TRUE_D = np.array([0.097, -0.165, 0., 0., 0.])
SDK_K = np.array([[911., 0., 627.], [0., 911., 382.], [0., 0., 1.]])


def make_model():
    return {"schema_version": 1, "kind": "charuco_color_intrinsics",
            "image_space": "raw_color", "model": "radial_k1k2",
            "camera": dict(PROFILE), "K": TRUE_K.tolist(), "D": TRUE_D.tolist(),
            "training_image_sha256": ["independent-training-image"]}


def make_dataset(root, count=12, seed=11):
    root.mkdir()
    (root / "images").mkdir()
    board = cv2.aruco.CharucoBoard(
        (8, 6), .028729, .02095,
        cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100),
    )
    data = {"target_type": "charuco", "camera": dict(PROFILE), "session_id": root.name,
            "board": {"squares_x": 8, "squares_y": 6, "square_length_m": .028729,
                      "marker_length_m": .02095, "dictionary": "DICT_5X5_100",
                      "legacy_pattern": False}, "samples": []}
    rng = np.random.default_rng(seed)
    for i in range(count):
        rvec = rng.uniform([-.6, -.6, -.3], [.6, .6, .3])
        tvec = rng.uniform([-.18, -.12, .4], [0., 0., .75])
        pixels, _ = cv2.projectPoints(board.getChessboardCorners(), rvec, tvec, TRUE_K, TRUE_D)
        image_path = root / "images" / f"{i:03d}.png"
        # Synthetic pixels are supplied explicitly; files establish identity/size only.
        image = np.zeros((720, 1280), np.uint8)
        image[0, :2] = [seed, i]
        assert cv2.imwrite(str(image_path), image)
        C = np.eye(4)
        C[:3, :3] = Rotation.from_rotvec(rvec).as_matrix()
        C[:3, 3] = tvec
        data["samples"].append({
            "sample_id": i, "sample_uid": f"{root.name}:{i}",
            "image": f"images/{i:03d}.png", "image_sha256": sha256(image_path),
            "T_camera_target": C.tolist(), "T_base_flange": np.linalg.inv(C).tolist(),
            "K": SDK_K.tolist(), "D": [0.] * 5,
            "charuco_ids": list(range(35)), "image_points": pixels.reshape(-1, 2).tolist(),
        })
    (root / "samples.json").write_text(json.dumps(data))
    return data


def test_fit_recovers_synthetic_radial_camera_and_held_out_improvement(tmp_path):
    make_dataset(tmp_path / "train")
    make_dataset(tmp_path / "valid", seed=23)
    model = estimate_intrinsics(tmp_path / "train")
    np.testing.assert_allclose(model["K"], TRUE_K, atol=.02)
    np.testing.assert_allclose(model["D"], TRUE_D, atol=.001)
    assert model["training_rms_px"] < 1e-3
    assert "not_for_runtime" in model["status"]
    report = validate_intrinsics(tmp_path / "valid", model)
    assert report["models"]["candidate"]["corner_reprojection_px"]["mean"] < .001
    assert report["models"]["sdk"]["corner_reprojection_px"]["mean"] > .01
    assert report["models"]["candidate"]["rotation_angle_difference_deg"]["max"] < .01


@pytest.mark.parametrize("field,value", [("serial", "other"), ("width", 640),
                                          ("frame", "depth_optical")])
def test_intrinsics_reject_profile_mismatch(field, value):
    profile = dict(PROFILE)
    profile[field] = value
    with pytest.raises(ValueError, match="mismatch"):
        check_intrinsics(make_model(), profile)


@pytest.mark.parametrize("distortion", [[0., -5., 0., 0., 0.],
                                        [np.nan, 0., 0., 0., 0.],
                                        [0., 0., .01, 0., 0.]])
def test_intrinsics_reject_invalid_or_folded_model(distortion):
    model = make_model()
    model["D"] = distortion
    with pytest.raises(ValueError):
        check_intrinsics(model)


def test_validation_rejects_training_image_reuse(tmp_path):
    make_dataset(tmp_path / "train")
    model = estimate_intrinsics(tmp_path / "train")
    with pytest.raises(ValueError, match="independent captures"):
        validate_intrinsics(tmp_path / "train", model)


def test_estimation_rejects_hash_tampering_and_single_tilt(tmp_path):
    data = make_dataset(tmp_path / "train")
    sample_path = tmp_path / "train" / "samples.json"
    data["samples"][0]["image_sha256"] = "wrong"
    sample_path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="SHA-256"):
        estimate_intrinsics(tmp_path / "train")
    data["samples"][0]["image_sha256"] = sha256(tmp_path / "train" / "images/000.png")
    for sample in data["samples"]:
        sample["T_camera_target"] = np.eye(4).tolist()
    sample_path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="tilt diversity"):
        estimate_intrinsics(tmp_path / "train")


def test_new_json_never_overwrites_old_file(tmp_path):
    path = tmp_path / "candidate.json"
    write_new_json(path, make_model())
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        write_new_json(path, {"replacement": True})
    assert path.read_bytes() == before
    load_intrinsics(path, PROFILE)


def test_invalid_json_values_do_not_create_an_output_file(tmp_path):
    path = tmp_path / "invalid.json"
    with pytest.raises(ValueError):
        write_new_json(path, {"bad": float("nan")})
    assert not path.exists()


def test_calibration_camera_override_keeps_raw_frame_and_sdk_values(tmp_path):
    path = tmp_path / "candidate.json"
    write_new_json(path, make_model())
    frame = SimpleNamespace(bgr=np.zeros((720, 1280, 3), np.uint8),
                            K=SDK_K.copy(), D=np.zeros(5))

    class FakeCamera:
        serial, width, height = "test", 1280, 720
        entered = closed = False

        def __enter__(self):
            self.entered = True
            return self

        def __exit__(self, *_):
            self.closed = True

        def read(self):
            return frame

    source = FakeCamera()
    with calibration.CalibrationIntrinsicsCamera(source, path, make_model()) as camera:
        result = camera.read()
        assert result.bgr is frame.bgr  # no image warp/rectification
        np.testing.assert_allclose(result.K, TRUE_K)
        np.testing.assert_allclose(result.sdk_K, SDK_K)
        np.testing.assert_allclose(frame.K, SDK_K)  # no mutation of SDK frame
    assert source.entered and source.closed
    source.serial = "wrong-device"
    source.closed = False
    with pytest.raises(ValueError, match="mismatch"):
        with calibration.CalibrationIntrinsicsCamera(source, path, make_model()):
            pass
    assert source.closed


def test_reprocess_writes_pnp_model_and_provenance_without_changing_original(tmp_path, monkeypatch):
    source = tmp_path / "source"
    data = make_dataset(source, count=1)
    before = (source / "samples.json").read_bytes()
    candidate = tmp_path / "candidate.json"
    write_new_json(candidate, make_model())
    detection = {"T_camera_target": np.eye(4), "valid_markers": 20, "valid_corners": 35,
                 "charuco_ids_flat": np.arange(35), "image_points": np.zeros((35, 2)),
                 "reprojection_mean_px": .1, "reprojection_median_px": .1,
                 "reprojection_max_px": .2, "pnp_success": True}
    seen = []

    def fake_detect(image, board, detector, K, D):
        seen.append((K.copy(), D.copy()))
        return detection

    monkeypatch.setattr(calibration, "detect_charuco", fake_detect)
    monkeypatch.setattr(calibration, "draw_detection", lambda image, *args: image)
    output = tmp_path / "processed"
    args = SimpleNamespace(dataset=source, output=output,
                           board=calibration.ROOT / "config/charuco_board.yaml",
                           intrinsics=candidate)
    calibration.run_reprocess(args)
    result = json.loads((output / "samples.json").read_text())
    np.testing.assert_allclose(seen[0][0], TRUE_K)
    np.testing.assert_allclose(result["samples"][0]["K"], TRUE_K)
    np.testing.assert_allclose(result["samples"][0]["sdk_K"], SDK_K)
    assert result["samples"][0]["image_sha256"] == data["samples"][0]["image_sha256"]
    assert (source / "samples.json").read_bytes() == before
    assert result["reprocessed_at_ns"] > 0
    assert result["source_samples_json_sha256"] == sha256(source / "samples.json")
    old_metadata = calibration.dataset_metadata(data)
    new_metadata = calibration.dataset_metadata(result)
    assert "camera_intrinsics" not in old_metadata
    assert new_metadata["camera_intrinsics"]["model_sha256"] == sha256(candidate)
    with pytest.raises(ValueError, match="independent captures"):
        calibration.ensure_independent({"training_image_sha256": [
            data["samples"][0]["image_sha256"]]}, result)


def test_cli_new_options_are_calibration_only():
    args = calibration.parse_args(["intrinsics-solve", "--dataset", "data", "--output", "new.json"])
    assert args.command == "intrinsics-solve"
    args = calibration.parse_args(["reprocess", "--dataset", "data", "--output", "new",
                                   "--intrinsics", "candidate.json"])
    assert args.intrinsics == "candidate.json"
