"""Orbbec boundary tests: no real SDK, USB or robot commands."""
import argparse
import sys
import time
from types import SimpleNamespace as NS

import cv2
import numpy as np
import pytest

from cameras.orbbec import OrbbecCamera, color_to_bgr, select_profile
from cameras.realsense import camera_for, close_all, create_camera


class ProfileList:
    def __init__(self, profiles): self.profiles = profiles
    def get_count(self): return len(self.profiles)
    def get_stream_profile_by_index(self, i): return self.profiles[i]


def profile(w, h, fps, fmt, distortion=0.):
    p = NS(get_width=lambda:w, get_height=lambda:h, get_fps=lambda:fps, get_format=lambda:fmt,
           get_intrinsic=lambda:NS(fx=4., fy=4., cx=1.5, cy=1.),
           get_distortion=lambda:NS(k1=distortion, k2=0., p1=0., p2=0., k3=0.,
                                     k4=0., k5=0., k6=0., model='brown'))
    p.as_video_stream_profile = lambda:p
    return p


def sdk_frame(pixels, fmt, sequence=1, scale=2.):
    f = NS(get_data=lambda:pixels.tobytes(), get_format=lambda:fmt,
           get_height=lambda:pixels.shape[0], get_width=lambda:pixels.shape[1],
           get_index=lambda:sequence, get_timestamp=lambda:sequence*33., get_depth_scale=lambda:scale)
    f.as_depth_frame = lambda:f
    return f


def frames(sequence=1, shape=(3,4)):
    color = sdk_frame(np.full((3,4,3), [1,2,3], np.uint8), 'rgb', sequence)
    data = np.full(shape, 250, np.uint16)
    data[0,0] = 0
    f = NS(get_color_frame=lambda:color, get_depth_frame=lambda:sdk_frame(data,'y16',sequence))
    f.as_frame_set = lambda:f
    return f


@pytest.fixture
def sdk(monkeypatch):
    calls = []
    device = NS(get_device_info=lambda:NS(get_serial_number=lambda:'OB123', get_name=lambda:'Gemini 335'),
                get_available_preset_list=lambda:NS(get_count=lambda:1, get_name_by_index=lambda i:'High Density'),
                load_preset=lambda n:calls.append(('preset',n)),
                set_bool_property=lambda *a:calls.append(('emitter',a)))
    devices = NS(get_count=lambda:1, get_device_by_index=lambda i:device,
                 get_device_serial_number_by_index=lambda i:'OB123')
    class Pipe:
        def __init__(self, dev):
            assert dev is device
            self.color_profile = profile(4,3,30,'rgb')
            calls.append(self)
        def get_stream_profile_list(self, sensor):
            return ProfileList([self.color_profile] if sensor=='color' else [profile(2,2,30,'y16')])
        def enable_frame_sync(self): calls.append('sync')
        def start(self, cfg, callback):
            self.callback = callback
            callback(frames())
        def stop(self): calls.append('stop')
    class Noise:
        def enable(self, enabled): assert enabled
        def process(self, depth):
            array = np.frombuffer(depth.get_data(),np.uint16).reshape(3,4).copy()
            array[1,1] = 0
            return sdk_frame(array,'y16',depth.get_index())
    ob = NS(Context=lambda:NS(query_devices=lambda:devices), Pipeline=Pipe,
            Config=lambda:NS(enable_stream=lambda p:None, set_frame_aggregate_output_mode=lambda v:None),
            OBFormat=NS(BGR='bgr', RGB='rgb', MJPG='mjpg', RGBA='rgba', Y16='y16'),
            OBSensorType=NS(COLOR_SENSOR='color', DEPTH_SENSOR='depth'),
            OBStreamType=NS(COLOR_STREAM='color'), OBFrameAggregateOutputMode=NS(FULL_FRAME_REQUIRE=1),
            OBPropertyID=NS(OB_PROP_LASER_BOOL=1),
            OBCameraDistortionModel=NS(BROWN_CONRADY='brown', BROWN_CONRADY_K6='brown6'),
            AlignFilter=lambda **kw:NS(process=lambda f:f), NoiseRemovalFilter=Noise)
    monkeypatch.setitem(sys.modules,'pyorbbecsdk',ob)
    return ob, calls


def camera(**kwargs):
    return OrbbecCamera(width=4,height=3,depth_width=2,depth_height=2,warmup=0,timeout=.02,**kwargs)


def test_exact_modes_never_fall_back_in_resolution_or_fps():
    p = profile(1280,800,30,'y16')
    assert select_profile(ProfileList([p]),1280,800,30,('y16',)) is p
    with pytest.raises(ValueError,match='1280x800@30'):
        select_profile(ProfileList([p]),1280,720,30,('y16',))
    with pytest.raises(ValueError,match='unavailable'):
        select_profile(ProfileList([p]),1280,800,15,('y16',))


def test_sdk_alignment_units_color_raw_support_and_latest_queue(sdk):
    _, calls = sdk
    c = camera(depth_filters='all')
    try:
        c.start()
        c.pipeline.callback(frames(2))
        c.pipeline.callback(frames(3))
        before = time.monotonic()
        f = c.read()
        assert f.sequence == 3 and f.backend == 'orbbec' and f.serial == 'OB123'
        assert f.received_monotonic <= before
        np.testing.assert_array_equal(f.bgr[1,1],[3,2,1])
        assert f.depth_m.dtype == np.float32 and f.depth_m.shape == (3,4)
        assert f.depth_m[0,0] == f.raw_depth_m[0,0] == 0
        assert f.depth_m[1,1] == 0 and f.raw_depth_m[1,1] == pytest.approx(.5)
        assert f.depth_m[2,2] == pytest.approx(.5)  # 250 * 2 mm -> 0.5 m
        assert ('preset','High Density') in calls and 'sync' in calls
    finally:
        c.close()
    assert calls.count('stop') == 1


def test_alignment_shape_mismatch_is_not_hidden_by_resize(sdk):
    c = camera()
    with c:
        c.pipeline.callback(frames(2,shape=(2,2)))
        with pytest.raises(RuntimeError,match='D2C shape mismatch'):
            c.read()


def test_rgb_only_uses_color_and_does_not_enable_depth_or_presets(sdk):
    _, calls = sdk
    with camera(enable_depth=False) as c:
        f = c.read()
        assert f.depth_m is None and f.raw_depth_m is None
    assert 'sync' not in calls and ('preset','High Density') not in calls


def test_warmup_timeout_stops_pipeline(sdk,monkeypatch):
    ob, calls = sdk
    monkeypatch.setattr(ob.Pipeline,'start',lambda *a:None)
    c = OrbbecCamera(width=4,height=3,depth_width=2,depth_height=2,warmup=1,timeout=.005)
    with pytest.raises(TimeoutError): c.start()
    assert c.pipeline is None and calls.count('stop') == 1


def test_rgb_and_aligned_depth_share_rectification_without_interpolating_depth(sdk,monkeypatch):
    import cameras.orbbec as module
    ob,_ = sdk
    original = ob.Pipeline.get_stream_profile_list
    def profiles(self,sensor):
        if sensor == 'color':
            self.color_profile = profile(4,3,30,'rgb',distortion=.2)
        return original(self,sensor)
    monkeypatch.setattr(ob.Pipeline,'get_stream_profile_list',profiles)
    remap = cv2.remap
    modes = []
    def observe(image, x,y, mode):
        modes.append(mode)
        return remap(image,x,y,mode)
    monkeypatch.setattr(module.cv2,'remap',observe)
    with camera() as c:
        f = c.read()
        assert c._maps is not None
        assert modes == [cv2.INTER_LINEAR,cv2.INTER_NEAREST,cv2.INTER_NEAREST]
        assert set(np.unique(f.depth_m)).issubset({0.,.5})
        np.testing.assert_array_equal(f.D,0.)


def test_registry_dispatches_orbbec_to_camera_process_without_importing_sdk():
    from cameras.process_camera import ProcessCamera
    try:
        a = camera_for(NS(camera_backend='orbbec'))
        b = camera_for(NS(camera_backend='realsense'))
        assert isinstance(a,ProcessCamera) and a.backend=='orbbec' and a is not b
        assert a is camera_for(NS(camera_backend='orbbec')) and a.pipeline is None
        with pytest.raises(ValueError,match='laser_power'):
            create_camera(backend='orbbec',laser_power=360)
    finally:
        close_all()


def test_orbbec_config_does_not_inherit_d435_serial_calibration_or_mask():
    from pathlib import Path
    from cameras.options import add_camera_arguments,apply_pipeline_defaults
    p = argparse.ArgumentParser()
    add_camera_arguments(p)
    p.add_argument('--calib-json')
    p.add_argument('--handeye')
    p.add_argument('--camera-self-mask',action='store_true')
    p.add_argument('--camera-self-mask-calibration')
    argv = ['--config','config/pipeline_orbbec.yaml']
    apply_pipeline_defaults(p,argv)
    a = p.parse_args(argv)
    assert a.camera_backend=='orbbec' and a.camera_serial=='CP0T963000S4'
    # Orbbec now has its own measured trial calibration and 100mm mask;
    # their absence was only true before the November 3/5 setup updates.
    assert Path(a.calib_json).name == 'orbbec_handeye_20261103_trial.json'
    assert a.calib_json == a.handeye and Path(a.calib_json).is_file()
    assert a.camera_self_mask
    assert Path(a.camera_self_mask_calibration).name == 'orbbec_self_mask_20261105_100mm_threeframe_trial.json'
    assert Path(a.camera_self_mask_calibration).is_file()
    assert a.camera_laser_power is None and a.camera_depth_height==800
    assert p.parse_args(argv+['--camera-depth-height','720']).camera_depth_height==720
