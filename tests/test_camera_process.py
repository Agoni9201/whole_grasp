"""Shared-frame process/lifecycle checks; no USB, CAN or robot access."""
import json
import os
from pathlib import Path
import sys

import numpy as np
import pytest

from cameras import process_camera


@pytest.fixture
def camera_worker(monkeypatch, tmp_path):
    helper = tmp_path/'camera_probe.py'
    helper.write_text('''
import json, os, sys, time
from types import SimpleNamespace
from multiprocessing.connection import Connection
import numpy as np
from cameras.process_camera import _worker_main
from cameras.realsense import Frame
class Camera:
    def __init__(self, **settings):
        self.settings=settings; self.seq=0
        import socket
        try:
            socket.socket(socket.AF_CAN, socket.SOCK_RAW)
        except RuntimeError as exc:
            assert 'forbidden' in str(exc)
        else:
            raise AssertionError('worker must forbid CAN')
        assert all(os.environ[k]=='1' for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'))
    def __enter__(self): return self
    def __exit__(self,*args): self.close()
    def close(self): pass
    def read(self):
        if os.environ.get('CAMERA_PROBE_FAIL'):
            raise ValueError('synthetic capture failure')
        time.sleep(.005); self.seq+=1
        h,w=self.settings['height'],self.settings['width']
        raw=np.full((h,w),self.seq,np.float32); raw[0,0]=0.
        return Frame(np.full((h,w,3),self.seq%251,np.uint8),
                     np.full((h,w),self.seq,np.float32),
                     SimpleNamespace(fx=100.,fy=100.,cx=0.,cy=0.),np.eye(3),np.zeros(5),
                     self.seq,float(self.seq),time.monotonic(),'synthetic',raw,self.settings['backend'])
_worker_main(Connection(int(sys.argv[1])),json.loads(sys.argv[2]),json.loads(sys.argv[3]),Camera)
''')
    original = process_camera.subprocess.Popen
    def spawn(command, **kwargs):
        fd = command[command.index('--worker-fd')+1]
        slot_index = command.index('--slots')+1
        paths = command[slot_index:slot_index+2]
        settings = command[command.index('--settings')+1]
        return original([sys.executable, str(helper), fd, json.dumps(paths), settings], **kwargs)
    monkeypatch.setattr(process_camera.subprocess, 'Popen', spawn)
    return lambda **kwargs: process_camera.ProcessCamera(width=64, height=48, warmup=0,
                                               timeout=2., depth_filters='all', **kwargs)


def test_child_owns_capture_and_frames_are_consistent_independent_snapshots(camera_worker):
    camera = camera_worker()
    try:
        first = camera.read()
        assert camera._process.pid != os.getpid()
        assert first.bgr.shape == (48, 64, 3)
        assert np.all(first.bgr == first.sequence%251)
        assert np.all(first.depth_m == first.sequence)
        assert first.raw_depth_m[0,0] == 0 and first.raw_depth_m[1,1] == first.sequence
        frozen = first.depth_m.copy()
        second = camera.read()
        assert second.sequence > first.sequence
        np.testing.assert_array_equal(first.depth_m, frozen)
        directory = Path(camera._directory.name)
        process = camera._process
    finally:
        camera.close()
        camera.close()
    assert process.poll() == 0 and not directory.exists()
    with pytest.raises(RuntimeError, match='closed'):
        camera.read()  # No automatic restart after failure/teardown.


def test_child_error_propagates_and_resources_close_without_restarting(camera_worker, monkeypatch):
    monkeypatch.setenv('CAMERA_PROBE_FAIL', '1')
    camera = camera_worker()
    try:
        with pytest.raises(RuntimeError, match='synthetic capture failure'):
            camera.read()
        process = camera._process
    finally:
        camera.close()
    assert process.poll() == 0


def test_orbbec_backend_survives_worker_settings_and_frame_transport(camera_worker):
    with camera_worker(backend='orbbec') as camera:
        frame = camera.read()
        assert camera.backend == frame.backend == 'orbbec'
        assert frame.raw_depth_m.shape == frame.depth_m.shape


def test_hand_warmup_does_not_lock_or_update_receiver_roles():
    from types import SimpleNamespace
    from unittest.mock import Mock
    from targets.hand_target import HandTarget
    from targets.hand_selection import HandRoles
    target = HandTarget.__new__(HandTarget)
    target.hand_roles = HandRoles()
    target.last_observations = []
    target._detector = SimpleNamespace(detect=Mock(return_value=[object()]))
    target.warmup(np.zeros((2,2,3),np.uint8), np.ones((2,2)), np.eye(3))
    target._detector.detect.assert_called_once()
    assert target.hand_roles.receiver is None and target.hand_roles.count == 0
    assert target.last_observations == []
