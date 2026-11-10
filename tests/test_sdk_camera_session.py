"""SDK lifecycle/transport regression without USB or robot access."""
import sys
import time
from types import SimpleNamespace
import numpy as np
import pytest
from cameras import realsense, sdk_loop


def frame(seq=1):
    intr=SimpleNamespace(fx=100.,fy=100.,cx=0.,cy=0.)
    return realsense.Frame(np.zeros((2,2,3),np.uint8),np.full((2,2),.5,np.float32),
                          intr,np.eye(3),np.zeros(5),seq,10.,time.monotonic(),'test')


def test_registry_reuses_camera_across_shallow_copied_args():
    from copy import copy
    args=SimpleNamespace(camera_serial='offline')
    try:
        assert realsense.camera_for(args) is realsense.camera_for(copy(args))
        assert realsense.camera_for(args).pipeline is None
    finally:
        realsense.close_all()


def test_median_capture_and_organized_xyz_use_same_metric_contract(monkeypatch,tmp_path):
    class Fake:
        def read(self): return frame()
    monkeypatch.setattr(realsense,'camera_for',lambda args:Fake())
    args=SimpleNamespace(save_dir=str(tmp_path),orbbec_depth_median_frames=3,
                         orbbec_depth_median_min_valid=2)
    bgr,cloud,intr,out=realsense.organized_cloud(args)
    from real_nero_grasp.depth_processing import pointcloud2_xyz_image
    xyz=pointcloud2_xyz_image(cloud)
    np.testing.assert_allclose(xyz[1,1],[.005,.005,.5])
    assert bgr.shape==(2,2,3)
    assert (tmp_path / out).is_dir()


def test_mit_capture_offloads_only_depth_arrays_after_reading_same_camera(monkeypatch, tmp_path):
    from real_nero_grasp import planning_process
    from cameras import depth_fusion
    samples = [frame(seq=i) for i in range(8)]
    sequence = iter(samples)
    reads, jobs = [], []
    class Fake:
        def read(self):
            item = next(sequence)
            reads.append(item.sequence)
            return item
    monkeypatch.setattr(realsense, 'camera_for', lambda args: Fake())
    def offload(depths, args):
        jobs.append(depths)
        assert all(isinstance(item, np.ndarray) for item in depths)
        assert [id(item) for item in depths] == [id(item.depth_m) for item in samples[1:]]
        return original(depths, min_valid_frames=args.orbbec_depth_median_min_valid,
                        max_mad=args.orbbec_depth_median_max_mad)
    original = depth_fusion.median_valid_depth_frames
    monkeypatch.setattr(planning_process, 'isolated_depth_median', offload)
    monkeypatch.setattr(depth_fusion, 'median_valid_depth_frames',
                        lambda *a, **kw: pytest.fail('MIT must not fuse depth inline'))
    args = SimpleNamespace(trajectory_executor='mit', save_dir=str(tmp_path),
                           orbbec_depth_median_frames=7, orbbec_depth_median_min_valid=5,
                           orbbec_depth_median_max_mad=.004)
    bgr, depth, intr, out = realsense.capture_rgbd(args)
    assert reads == list(range(8)) and len(jobs) == 1
    np.testing.assert_array_equal(depth, .5)
    assert bgr is samples[-1].bgr and intr is samples[-1].intr
    import json
    from pathlib import Path
    meta = json.loads((Path(out) / 'capture_meta.json').read_text())
    assert meta['sequence'] == 7 and meta['temporal_frames'] == 7


def test_mit_depth_worker_failure_does_not_fall_back_inline_or_save_result(monkeypatch, tmp_path):
    from real_nero_grasp import planning_process
    from cameras import depth_fusion
    monkeypatch.setattr(realsense, 'camera_for', lambda args: SimpleNamespace(read=lambda: frame()))
    def fail(*a, **kw):
        raise planning_process.PlanningProcessError('offline worker failure')
    monkeypatch.setattr(planning_process, 'isolated_depth_median', fail)
    monkeypatch.setattr(depth_fusion, 'median_valid_depth_frames',
                        lambda *a, **kw: pytest.fail('must not fall back inline'))
    with pytest.raises(planning_process.PlanningProcessError, match='offline worker failure'):
        realsense.capture_rgbd(SimpleNamespace(trajectory_executor='mit', save_dir=str(tmp_path)))
    assert list(tmp_path.iterdir()) == []


def test_capture_keeps_raw_window_for_same_frames_as_processed_median(monkeypatch, tmp_path):
    import json
    from pathlib import Path
    samples = [frame(seq=i) for i in range(4)]
    for i, item in enumerate(samples):
        item.raw_depth_m = np.full((2, 2), .4 + i * .01, np.float32)
        item.raw_depth_m[0, 0] = 0.  # SDK filled/historic return is retained separately.
    iterator = iter(samples)
    monkeypatch.setattr(realsense, 'camera_for', lambda _: SimpleNamespace(read=lambda: next(iterator)))
    _, depth, _, out = realsense.capture_rgbd(SimpleNamespace(save_dir=str(tmp_path),
        orbbec_depth_median_frames=3, orbbec_depth_median_min_valid=2))
    meta = json.loads((Path(out) / 'capture_meta.json').read_text())
    assert meta['raw_temporal_frames'] == 3
    with np.load(Path(out) / meta['raw_temporal_file']) as arrays:
        np.testing.assert_array_equal(arrays['sequence'], [1, 2, 3])
        np.testing.assert_array_equal(arrays['depth_m'], [f.raw_depth_m for f in samples[1:]])
    assert depth[0, 0] == 0  # No raw return in the ENTIRE window: SDK filling is not evidence.
    assert depth[1, 1] == .5
    assert meta['depth_support_mode'] == 'same_window_raw_support'
    assert (Path(out) / 'depth_sdk_m.npy').exists()
    assert (Path(out) / 'depth_support_report.json').exists()


def test_mit_raw_depth_support_is_computed_off_process(monkeypatch, tmp_path):
    from cameras import depth_fusion
    from real_nero_grasp import planning_process
    original = depth_fusion.supported_depth_median
    samples = [frame(seq=i) for i in range(8)]
    for item in samples:
        item.raw_depth_m = item.depth_m.copy()
        item.raw_depth_m[0, 0] = 0
    sequence = iter(samples)
    monkeypatch.setattr(realsense, 'camera_for', lambda _: SimpleNamespace(read=lambda: next(sequence)))
    calls = []
    def offload(processed, raw, args):
        calls.append(len(raw))
        return original(processed, raw, args.orbbec_depth_median_min_valid, args.orbbec_depth_median_max_mad)
    monkeypatch.setattr(planning_process, 'isolated_supported_depth_median', offload)
    monkeypatch.setattr(depth_fusion, 'supported_depth_median', lambda *a, **kw: pytest.fail('MIT cannot fuse raw depth inline'))
    _, depth, _, _ = realsense.capture_rgbd(SimpleNamespace(
        trajectory_executor='mit', save_dir=str(tmp_path), orbbec_depth_median_frames=7,
        orbbec_depth_median_min_valid=5, orbbec_depth_median_max_mad=.004))
    assert calls == [7]
    assert depth[0, 0] == 0 and depth[1, 1] == .5


def test_partial_raw_window_is_rejected(monkeypatch, tmp_path):
    samples = [frame(seq=i) for i in range(4)]
    samples[1].raw_depth_m = samples[1].depth_m.copy()
    sequence = iter(samples)
    monkeypatch.setattr(realsense, 'camera_for', lambda _: SimpleNamespace(read=lambda: next(sequence)))
    with pytest.raises(RuntimeError, match='Incomplete raw'):
        realsense.capture_rgbd(SimpleNamespace(save_dir=str(tmp_path), orbbec_depth_median_frames=3))
    assert list(tmp_path.iterdir()) == []


def test_mit_supported_depth_failure_does_not_fall_back(monkeypatch, tmp_path):
    from real_nero_grasp import planning_process
    item = frame()
    item.raw_depth_m = item.depth_m.copy()
    monkeypatch.setattr(realsense, 'camera_for', lambda _: SimpleNamespace(read=lambda: item))
    def fail(*a, **kw):
        raise planning_process.PlanningProcessError('raw support worker failed')
    monkeypatch.setattr(planning_process, 'isolated_supported_depth_median', fail)
    with pytest.raises(planning_process.PlanningProcessError, match='raw support worker failed'):
        realsense.capture_rgbd(SimpleNamespace(trajectory_executor='mit', save_dir=str(tmp_path)))
    assert list(tmp_path.iterdir()) == []


def test_warmup_failure_stops_started_pipeline(monkeypatch):
    class Device:
        def get_info(self,key): return 'D435i' if key=='name' else 'serial'
    calls=[]
    class Pipe:
        def start(self,cfg): return SimpleNamespace(get_device=lambda:SimpleNamespace(
            first_depth_sensor=lambda:SimpleNamespace(get_depth_scale=lambda:.001)))
        def wait_for_frames(self,ms): raise RuntimeError('timeout')
        def stop(self): calls.append('stopped')
    cfg=SimpleNamespace(enable_device=lambda *a:None,enable_stream=lambda *a:None)
    fake=SimpleNamespace(context=lambda:SimpleNamespace(query_devices=lambda:[Device()]),
        camera_info=SimpleNamespace(name='name',serial_number='serial'),
        config=lambda:cfg,pipeline=lambda ctx:Pipe(),align=lambda x:None,
        stream=SimpleNamespace(color='color',depth='depth'),format=SimpleNamespace(bgr8=1,z16=2))
    monkeypatch.setitem(sys.modules,'pyrealsense2',fake)
    camera=realsense.RealSenseCamera(warmup=1)
    with pytest.raises(RuntimeError,match='timeout'): camera.start()
    assert camera.pipeline is None and calls==['stopped']


def test_sdk_loop_delivers_color_depth_and_info_without_ros(monkeypatch):
    class Fake:
        timeout=.1
        closed=False
        def read(self):
            time.sleep(.005)
            return frame()
        def close(self): self.closed=True
    camera=Fake()
    monkeypatch.setattr(sdk_loop,'camera_for',lambda args:camera)
    node=sdk_loop.Node('test');node.args=SimpleNamespace()
    received=[]
    node.create_subscription(sdk_loop.Image,'color',lambda img:received.append(img))
    def tick():
        if received: sdk_loop.shutdown()
    node.create_timer(.01,tick)
    sdk_loop.init()
    sdk_loop.spin(node)
    assert camera.closed and len(received)==1
    assert received[0].width==2
    np.testing.assert_allclose(node._latest_depth,.5)


def test_config_cli_overrides_yaml(tmp_path):
    import argparse
    from cameras.options import add_camera_arguments,apply_pipeline_defaults
    path=tmp_path/'pipeline.yaml'
    path.write_text('wrist_camera:\n  width: 848\n')
    args=['--config',str(path),'--camera-width','640']
    p=argparse.ArgumentParser();add_camera_arguments(p);apply_pipeline_defaults(p,args)
    assert p.parse_args(args).camera_width==640


def test_maximum_profiles_and_filters_are_independent_yaml_defaults(tmp_path):
    import argparse
    from cameras.options import add_camera_arguments, apply_pipeline_defaults
    path = tmp_path/'pipeline.yaml'
    path.write_text('wrist_camera:\n  width: 1920\n  height: 1080\n  fps: 30\n'
                    '  depth_width: 1280\n  depth_height: 720\n  depth_fps: 30\n  depth_filters: all\n')
    argv = ['--config', str(path), '--camera-width', '640', '--camera-height', '480']
    parser = argparse.ArgumentParser()
    add_camera_arguments(parser)
    apply_pipeline_defaults(parser, argv)
    args = parser.parse_args(argv)
    assert (args.camera_width, args.camera_height) == (640, 480)
    assert (args.camera_depth_width, args.camera_depth_height, args.camera_depth_fps) == (1280, 720, 30)
    assert args.camera_depth_filters == 'all'
    assert parser.parse_args(argv+['--camera-depth-filters', 'off']).camera_depth_filters == 'off'
    try:
        a = realsense.camera_for(args)
        args.camera_depth_filters = 'off'
        b = realsense.camera_for(args)
        assert a is not b and a.depth_width == 1280 and a.width == 640
    finally:
        realsense.close_all()


def test_filled_or_historic_depth_cannot_supply_missing_current_hand_measurements():
    raw = np.array([[0., np.nan, .05], [.5, .8, 3.]], dtype=np.float32)
    filtered = np.full(raw.shape, .6, dtype=np.float32)
    result = realsense.depth_with_current_support(filtered, raw, .1, 2.)
    np.testing.assert_allclose(result, [[0., 0., 0.], [.6, .6, 0.]])
    np.testing.assert_allclose(filtered, .6)  # The filled image is retained separately.
    assert realsense.depth_with_current_support(filtered, None, .1, 2.) is filtered
    with pytest.raises(ValueError, match='alignment'):
        realsense.depth_with_current_support(filtered, raw[:1], .1, 2.)


def test_mixed_resolution_streams_filter_history_and_unmodified_raw_depth(monkeypatch):
    calls, streams, constructed = [], [], []
    raw = np.array([[0, 500], [600, 0]], dtype=np.uint16)
    intr = SimpleNamespace(fx=100., fy=100., ppx=.5, ppy=.5, coeffs=np.zeros(5))
    class Depth:
        def __init__(self, pixels=raw, sequence=1): self.pixels, self.sequence = pixels, sequence
        def get_data(self): return self.pixels
        def get_frame_number(self): return self.sequence
        def get_timestamp(self): return self.sequence*33.
    class Frames:
        def __init__(self, seq): self.depth = Depth(sequence=seq)
        def get_depth_frame(self): return self.depth
        def get_color_frame(self): return SimpleNamespace(
            get_data=lambda: np.zeros((2, 2, 3), np.uint8),
            profile=SimpleNamespace(as_video_stream_profile=lambda:SimpleNamespace(intrinsics=intr)))
    class Block:
        def __init__(self, name): self.name = name; constructed.append(self)
        def set_option(self, *args): pass
        def process(self, depth):
            calls.append(self.name)
            return Depth(np.where(depth.pixels > 0, depth.pixels, 450).astype(np.uint16),
                         depth.sequence) if self.name == 'holes' else depth
    class Device:
        def get_info(self, key): return 'RealSense D435IF' if key == 'name' else 'test'
    class Pipe:
        seq = 0
        def start(self, cfg): return SimpleNamespace(get_device=lambda:SimpleNamespace(
            first_depth_sensor=lambda:SimpleNamespace(get_depth_scale=lambda:.001)))
        def wait_for_frames(self, ms):
            self.seq += 1
            first = Frames(self.seq)
            self.seq += 1
            self.queued = [Frames(self.seq)]
            return first
        def poll_for_frames(self): return self.queued.pop(0) if self.queued else None
        def stop(self): pass
    opts = SimpleNamespace(**{k:k for k in ('filter_magnitude', 'filter_smooth_alpha',
                                           'filter_smooth_delta', 'holes_fill')})
    fake = SimpleNamespace(context=lambda:SimpleNamespace(query_devices=lambda:[Device()]),
        camera_info=SimpleNamespace(name='name', serial_number='serial'),
        config=lambda:SimpleNamespace(enable_device=lambda *a:None,
                                      enable_stream=lambda *a:streams.append(a)),
        pipeline=lambda ctx:Pipe(), align=lambda _:SimpleNamespace(process=lambda frame:frame),
        stream=SimpleNamespace(color='color', depth='depth'), format=SimpleNamespace(bgr8=1, z16=2),
        option=opts, disparity_transform=lambda to:Block('to_disparity' if to else 'to_depth'),
        spatial_filter=lambda:Block('spatial'), temporal_filter=lambda:Block('temporal'),
        hole_filling_filter=lambda _:Block('holes'))
    monkeypatch.setitem(sys.modules, 'pyrealsense2', fake)
    with realsense.RealSenseCamera(width=1920, height=1080, depth_width=1280,
            depth_height=720, depth_filters='all', warmup=0) as camera:
        first, second = camera.read(), camera.read()
        assert streams == [('color', 1920, 1080, 1, 30), ('depth', 1280, 720, 2, 30)]
        assert calls == ['to_disparity', 'spatial', 'temporal', 'to_depth', 'holes']*2
        assert len(constructed) == 5  # Temporal state belongs to one camera session.
        assert first.sequence == 2 and second.sequence == 4  # Process latest queued frame only.
        np.testing.assert_allclose(first.raw_depth_m, raw*.001)
        assert first.depth_m[0, 0] == pytest.approx(.45) and first.raw_depth_m[0, 0] == 0.
        assert second.sequence != first.sequence
    assert camera._depth_filters == []


def test_handoff_stale_feedback_does_not_suppress_exit_stop():
    from handoff.grasp_to_servo_handoff import stop_grasp_control
    feedback=SimpleNamespace(timestamp=0.,msg=[0.]*7)
    arm=SimpleNamespace(get_joint_angles=lambda:[0.]*7,
        terminate_motion=lambda **kw:None,
        arm=SimpleNamespace(get_joint_angles=lambda:feedback))
    assert stop_grasp_control(arm) is False
    assert not getattr(arm,'_suppress_program_exit_estop',False)
