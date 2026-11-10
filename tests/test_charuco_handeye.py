import numpy as np
import pytest
import json
import time
from types import SimpleNamespace
from scipy.spatial.transform import Rotation

from tools.calibrate_d435i_charuco import (
    camera_from_args,
    dataset_payload,
    draw_detection,
    ensure_independent,
    make_transform,
    residual_values,
    solve_opencv_methods,
    solve_park,
    parse_args,
)


def test_charuco_camera_uses_color_only_realsense(monkeypatch):
    captured = {}

    class FakeCamera:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr("cameras.realsense.RealSenseCamera", FakeCamera)
    args = type(
        "Args",
        (),
        {
            "serial": "test",
            "width": 640,
            "height": 480,
            "fps": 30,
            "camera_timeout": 10.0,
            "warmup": 2,
        },
    )()

    camera_from_args(args)

    assert captured["enable_depth"] is False


def test_orbbec_calibration_uses_runtime_rectified_rgb_and_records_camera_identity(monkeypatch):
    from cameras.orbbec import OrbbecCamera
    captured = {}
    class FakeCamera:
        backend = 'orbbec'
        optical_frame = OrbbecCamera.optical_frame
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.__dict__.update(kwargs)
    monkeypatch.setattr('cameras.orbbec.OrbbecCamera', FakeCamera)
    args = parse_args(['test', '--backend', 'orbbec', '--serial', 'OB123',
                       '--width', '1280', '--height', '720'])
    camera = camera_from_args(args)
    assert captured['enable_depth'] is False
    assert (camera.width,camera.height,camera.serial) == (1280,720,'OB123')
    metadata = dataset_payload('example',{},camera,'V111','can0')['camera']
    assert metadata['frame']=='orbbec_color_optical_frame'
    assert metadata['backend']=='orbbec' and metadata['image_space']=='rectified_color'
    args.intrinsics = 'old_d435_raw_intrinsics.json'
    with pytest.raises(ValueError,match='SDK-rectified'):
        camera_from_args(args)


def test_joint_audit_reads_fresh_feedback_without_commands_and_copies_sdk_buffer():
    from tools.calibrate_d435i_charuco import NeroReadOnlyFeedback
    response=SimpleNamespace(msg=[.1]*7,timestamp=time.time())
    reader=NeroReadOnlyFeedback.__new__(NeroReadOnlyFeedback)
    reader.arm=SimpleNamespace(get_joint_angles=lambda:response)
    q,stamp=reader.read_joints()
    response.msg[0]=.5
    assert q[0]==.1 and stamp==response.timestamp
    response.timestamp=time.time()-1.
    with pytest.raises(RuntimeError,match='stale'):
        reader.read_joints()


def test_capture_saves_joint_and_host_image_timing_without_moving_robot(tmp_path,monkeypatch):
    import cv2
    from cameras.realsense import Frame
    from tools import calibrate_d435i_charuco as app
    board,*_=app.load_board_config('config/charuco_board.yaml')
    image=cv2.cvtColor(board.generateImage((640,480),marginSize=20),cv2.COLOR_GRAY2BGR)
    K=np.array([[800.,0,320.],[0,800.,240.],[0,0,1.]])
    closed=[]
    class Camera:
        serial='test';width=640;height=480;fps=30;backend='orbbec';optical_frame='orbbec_color_optical_frame'
        def __enter__(self): return self
        def __exit__(self,*a): closed.append('camera')
        def read(self):
            return Frame(image,None,SimpleNamespace(),K,np.zeros(5),2,60.,time.monotonic()-.02,'test')
    class ReadOnly:
        def __init__(self,*a): self.counter=0
        def __enter__(self): return self
        def __exit__(self,*a): closed.append('feedback')
        def read(self):
            self.counter+=1
            return np.zeros(6),np.eye(4),float(self.counter)
        def read_joints(self): return np.zeros(7),time.time()
        # Any attempt to enable, move, stop, or operate a gripper fails.
    monkeypatch.setattr(app,'camera_from_args',lambda _:Camera())
    monkeypatch.setattr(app,'NeroReadOnlyFeedback',ReadOnly)
    monkeypatch.setattr(app.cv2,'imshow',lambda *a:None)
    keys=iter([ord('s'),ord('q')]);monkeypatch.setattr(app.cv2,'waitKey',lambda _:next(keys))
    monkeypatch.setattr(app.cv2,'destroyAllWindows',lambda:None)
    args=SimpleNamespace(board='config/charuco_board.yaml',output=str(tmp_path/'capture'),
                         channel='can0',firmware='V111',feedback_timeout=5.,candidate_frames=1)
    app.run_capture(args)
    record=json.loads((tmp_path/'capture/samples.json').read_text())['samples'][0]
    assert record['joints_before_rad']==record['joints_after_rad']==[0.]*7
    assert len(record['joint_feedback_stamps'])==2 and record['camera_sequence']==2
    assert record['capture_timing']['image_age_at_read_ms']>=19.
    assert record['capture_timing']['after_feedback_request_monotonic']>=record['capture_timing']['image_read_return_monotonic']
    assert set(closed)=={'camera','feedback'}


def test_park_solver_recovers_exact_eye_in_hand_transform():
    rng = np.random.default_rng(31)
    flange_camera = make_transform(
        Rotation.from_euler("xyz", [0.2, -0.8, 1.1]).as_matrix(),
        [0.07, -0.073, 0.002],
    )
    base_target = make_transform(
        Rotation.from_euler("xyz", [0.1, 0.0, -0.2]).as_matrix(),
        [-0.45, 0.05, 0.02],
    )
    samples = []
    for _ in range(18):
        base_flange = make_transform(
            Rotation.random(random_state=rng).as_matrix(),
            rng.uniform([-0.5, -0.3, 0.25], [-0.2, 0.3, 0.65]),
        )
        camera_target = np.linalg.inv(flange_camera) @ np.linalg.inv(base_flange) @ base_target
        samples.append(
            {
                "T_base_flange": base_flange.tolist(),
                "T_camera_target": camera_target.tolist(),
            }
        )

    estimated, estimated_target, _ = solve_park(samples)
    error = np.linalg.inv(flange_camera) @ estimated
    assert np.linalg.norm(error[:3, 3]) < 1e-10
    assert Rotation.from_matrix(error[:3, :3]).magnitude() < 1e-10
    values = residual_values(samples, estimated, estimated_target)
    assert max(values["translation_mm"]) < 1e-8
    assert max(values["rotation_deg"]) < 1e-8


def test_original_five_method_solver_recovers_exact_eye_in_hand_transform():
    rng = np.random.default_rng(37)
    flange_camera = make_transform(
        Rotation.from_euler("xyz", [0.3, -0.7, 1.0]).as_matrix(),
        [0.065, -0.08, 0.004],
    )
    base_target = make_transform(
        Rotation.from_euler("xyz", [0.0, 0.1, -0.15]).as_matrix(),
        [-0.5, 0.1, 0.01],
    )
    samples = []
    for sample_id in range(20):
        base_flange = make_transform(
            Rotation.random(random_state=rng).as_matrix(),
            rng.uniform([-0.5, -0.3, 0.25], [-0.2, 0.3, 0.65]),
        )
        camera_target = (
            np.linalg.inv(flange_camera)
            @ np.linalg.inv(base_flange)
            @ base_target
        )
        samples.append(
            {
                "sample_id": sample_id,
                "T_base_flange": base_flange.tolist(),
                "T_camera_target": camera_target.tolist(),
            }
        )

    best, results, _, failures = solve_opencv_methods(samples)

    assert best in results
    assert not failures
    assert set(results) == {"TSAI", "PARK", "HORAUD", "ANDREFF", "DANIILIDIS"}
    for result in results.values():
        error = np.linalg.inv(flange_camera) @ result["T_flange_camera"]
        assert np.linalg.norm(error[:3, 3]) < 1e-7
        assert Rotation.from_matrix(error[:3, :3]).magnitude() < 1e-7


def test_validation_rejects_reprocessed_training_image():
    calibration = {"training_image_sha256": ["same-image", "training-only"]}
    dataset = {"samples": [{"image_sha256": "same-image"}]}
    with pytest.raises(ValueError, match="independent captures"):
        ensure_independent(calibration, dataset)


def test_validation_accepts_distinct_images():
    calibration = {"training_image_sha256": ["training"]}
    dataset = {"samples": [{"image_sha256": "validation"}]}
    ensure_independent(calibration, dataset)


def test_validation_rejects_missing_image_identity():
    with pytest.raises(ValueError, match="SHA-256"):
        ensure_independent({"training_image_sha256": ["training"]}, {"samples": [{}]})


def test_draw_detection_accepts_opencv5_flat_charuco_corners():
    image = np.zeros((80, 100, 3), dtype=np.uint8)
    detection = {
        "marker_ids": None,
        "marker_corners": (),
        "charuco_corners": np.array([[20.0, 20.0], [40.0, 40.0]], np.float32),
        "charuco_ids": np.array([0, 1], np.int32),
        "pnp_success": False,
    }

    visual = draw_detection(image, detection, np.eye(3), np.zeros(5), 0.01)

    assert visual.shape == image.shape
