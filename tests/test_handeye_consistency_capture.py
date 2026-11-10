import json
import io
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from tools import verify_handeye_aruco_consistency as tool


def test_rgb_marker_center_uses_known_size_without_depth():
    length = .02095
    h = length / 2
    points = np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]])
    intr = SimpleNamespace(fx=900., fy=890., cx=640., cy=360.)
    K = np.array([[intr.fx, 0, intr.cx], [0, intr.fy, intr.cy], [0, 0, 1.]])
    expected = np.array([.03, -.02, .45])
    pixels, _ = cv2.projectPoints(points, np.array([.2, -.3, .1]), expected, K, np.zeros(5))
    actual = tool._pnp_marker_center([pixels], np.array([[7]]), 7, intr, length)
    np.testing.assert_allclose(actual, expected, atol=1e-7)


@pytest.mark.parametrize("moving_first", [False, True])
def test_capture_reads_only_locks_marker_and_rejects_motion(tmp_path, monkeypatch, moving_first):
    calibration = tmp_path / "calibration.json"
    calibration.write_text(json.dumps(dict(position=[0, 0, 0], orientation=[0, 0, 0, 1])))
    args = SimpleNamespace(
        save_dir=str(tmp_path / "capture"), calib_json=str(calibration),
        handeye_convention="flange_camera_xyzw", aruco_id=None,
        aruco_dict="DICT_5X5_100", marker_length_mm=None,
        nero_interface="socketcan", nero_bitrate=1000000,
        nero_channel="can0", nero_firmware="V111", nero_enable_timeout=5.,
        samples=1 if moving_first else 2, depth_radius=2,
        preview=False,
    )
    monkeypatch.setattr(tool, "parse_args", lambda: args)

    class FeedbackOnly:
        def __init__(self, channel, firmware, timeout):
            self.read_count = 0
        def __enter__(self):
            return self
        def __exit__(self, *_):
            pass
        def read(self):
            self.read_count += 1
            pose, transform = np.zeros(6), np.eye(4)
            if moving_first and self.read_count == 2:
                pose[0] = transform[0, 3] = .005
            return pose, transform, float(self.read_count)
        # No enable/move/stop/gripper methods: the test must complete without them.

    monkeypatch.setattr(tool, "NeroReadOnlyFeedback", FeedbackOnly)
    answers = iter(["", ""])
    monkeypatch.setattr("builtins.input", lambda _: next(answers))
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    intr = SimpleNamespace(fx=500., fy=500., cx=50., cy=50.)
    monkeypatch.setattr(tool, "capture_sdk_rgbd", lambda _: (image, np.full((100, 100), .5), intr, None))
    requested_ids = []
    def detect(_, dictionary, marker_id, **kwargs):
        requested_ids.append(marker_id)
        corners = [np.array([[[40, 40], [60, 40], [60, 60], [40, 60]]], dtype=np.float32)]
        return 50., 50., 7, corners, np.array([[7]], dtype=np.int32)
    monkeypatch.setattr(tool, "_detect_aruco_center", detect)
    tool.main()
    log = json.loads(next((tmp_path / "capture").glob("handeye_*.json")).read_text())
    assert log["robot_commands_sent"] is False
    assert log["aruco_id"] == 7
    assert requested_ids == ([None] if moving_first else [None, 7])
    assert len(log["samples"]) == args.samples
    np.testing.assert_allclose(log["samples"][0]["P_base"], [0, 0, .5])


@pytest.mark.parametrize('save_key,exit_mode', [(ord('s'),'q'), (13,'close'), (None,'ctrlc')])
def test_live_preview_precedes_sampling_displays_missing_marker_and_depth_and_closes(
        tmp_path, monkeypatch, save_key, exit_mode):
    calibration = tmp_path/'calibration.json'
    calibration.write_text(json.dumps(dict(position=[0,0,0],orientation=[0,0,0,1])))
    args = SimpleNamespace(save_dir=str(tmp_path/'capture'),calib_json=str(calibration),
        handeye_convention='flange_camera_xyzw',aruco_id=None,aruco_dict='DICT_5X5_100',
        marker_length_mm=None,nero_interface='socketcan',nero_bitrate=1000000,
        nero_channel='can0',nero_firmware='V111',nero_enable_timeout=5.,samples=0,
        depth_radius=2,preview=True)
    monkeypatch.setattr(tool,'parse_args',lambda:args)
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
    images = []
    for marker in (7,7,8,None,7):
        image = np.full((480,640,3),255,np.uint8)
        if marker is not None:
            gray = cv2.aruco.generateImageMarker(dictionary,marker,80)
            image[160:240,220:300] = cv2.cvtColor(gray,cv2.COLOR_GRAY2BGR)
        images.append(image)
    calls = dict(frame=-1,reads=0,captures=0,closed=False,disconnected=False,windows_closed=False)
    intr = SimpleNamespace(fx=500.,fy=500.,cx=320.,cy=240.)
    class Camera:
        def read(self):
            calls['frame'] += 1
            i = calls['frame']
            return SimpleNamespace(bgr=images[i],depth_m=np.full((480,640),.5),
                raw_depth_m=np.zeros((480,640)) if i==4 else np.full((480,640),.5))
        def close(self): calls['closed']=True
    camera = Camera()
    monkeypatch.setattr(tool,'camera_for',lambda _:camera)
    monkeypatch.setattr(tool,'close_all',camera.close)
    class FeedbackOnly:
        def __init__(self,*args): pass
        def __enter__(self): return self
        def __exit__(self,*args): calls['disconnected']=True
        def read(self):
            calls['reads']+=1
            return np.zeros(6),np.eye(4),float(calls['reads'])
    monkeypatch.setattr(tool,'NeroReadOnlyFeedback',FeedbackOnly)
    def capture(_):
        # Merely showing the first preview frame must not trigger sampling.
        assert calls['frame']==1
        calls['captures']+=1
        return images[1],np.full((480,640),.5),intr,None
    monkeypatch.setattr(tool,'capture_sdk_rgbd',capture)
    monkeypatch.setattr(tool.cv2,'namedWindow',lambda *a:None)
    monkeypatch.setattr(tool.cv2,'resizeWindow',lambda *a:None)
    shown=[]
    monkeypatch.setattr(tool.cv2,'imshow',lambda _,image:shown.append(image.copy()))
    monkeypatch.setattr(tool.cv2,'getWindowProperty',lambda *a:0 if calls['frame']==4 and exit_mode=='close' else 1)
    monkeypatch.setattr(tool.cv2,'destroyAllWindows',lambda:calls.update(windows_closed=True))
    def key(_):
        if calls['frame']==1: return -1 if save_key is None else save_key
        if calls['frame']==4:
            if exit_mode=='ctrlc': raise KeyboardInterrupt
            return ord('q') if exit_mode=='q' else -1
        return -1
    monkeypatch.setattr(tool.cv2,'waitKey',key)
    terminal=io.StringIO('\n')
    monkeypatch.setattr(tool.sys,'stdin',terminal)
    monkeypatch.setattr(tool.select,'select',lambda *a:([terminal],[],[]) if calls['frame']==1 and save_key is None else ([],[],[]))
    text=[]
    put_text=cv2.putText
    def observe(image,string,*a,**kw):
        text.append(string)
        return put_text(image,string,*a,**kw)
    monkeypatch.setattr(tool.cv2,'putText',observe)
    tool.main()
    log=json.loads(next((tmp_path/'capture').glob('handeye_*.json')).read_text())
    assert log['robot_commands_sent'] is False and log['aruco_id']==7 and len(log['samples'])==1
    assert calls['captures']==1 and calls['reads']==2 and len(shown)==5
    assert calls['closed'] and calls['disconnected'] and calls['windows_closed']
    assert any('ID 7: depth 500.0 mm' in t for t in text)
    assert any('Target ID 7 missing' in t for t in text)
    assert any(t=='NO MARKERS DETECTED' for t in text)
    assert any('DEPTH MISSING' in t for t in text)


def test_preview_is_default_and_can_be_disabled(monkeypatch):
    argv=['check','--calib-json','unused.json']
    monkeypatch.setattr(tool.sys,'argv',argv)
    assert tool.parse_args().preview is True
    monkeypatch.setattr(tool.sys,'argv',argv+['--no-preview'])
    assert tool.parse_args().preview is False
