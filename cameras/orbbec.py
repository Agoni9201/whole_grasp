"""Orbbec SDK v2 wrist RGB-D, implementing the shared metric Frame contract.

The SDK aligns depth into the color optical frame. Both images are then
rectified together because grasp/hand consumers use pinhole back-projection.
No ROS, controller imports, temporal filling, or invented depth measurements.
"""
from collections import deque
import threading
import time
from types import SimpleNamespace

import cv2
import numpy as np

from .realsense import Frame, RealSenseCamera


def select_profile(profiles, width, height, fps, formats):
    """Select an exact native mode; never silently resize depth into alignment."""
    available = [profiles.get_stream_profile_by_index(i).as_video_stream_profile()
                 for i in range(profiles.get_count())]
    matching = [p for p in available
                if (p.get_width(), p.get_height(), p.get_fps()) == (width, height, fps)
                and p.get_format() in formats]
    if not matching:
        modes = sorted({f'{p.get_width()}x{p.get_height()}@{p.get_fps()} {p.get_format()}'
                        for p in available})
        raise ValueError(f'Orbbec mode {width}x{height}@{fps} unavailable; supported: {modes}')
    return min(matching, key=lambda p: formats.index(p.get_format()))


def color_to_bgr(frame, ob):
    width, height, fmt = frame.get_width(), frame.get_height(), frame.get_format()
    data = np.frombuffer(frame.get_data(), dtype=np.uint8)
    if fmt == ob.OBFormat.MJPG:
        image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    elif fmt == ob.OBFormat.BGR:
        image = data.reshape(height, width, 3)
    elif fmt == ob.OBFormat.RGB:
        image = cv2.cvtColor(data.reshape(height, width, 3), cv2.COLOR_RGB2BGR)
    elif fmt == ob.OBFormat.RGBA:
        image = cv2.cvtColor(data.reshape(height, width, 4), cv2.COLOR_RGBA2BGR)
    else:
        raise ValueError(f'Unsupported Orbbec color format: {fmt}')
    if image is None or image.shape != (height, width, 3):
        raise ValueError('Orbbec color decode failed')
    return image.copy()


def depth_metres(frame):
    # Orbbec depth scale is millimetres per raw unit, not metres per unit.
    return (np.frombuffer(frame.get_data(), dtype=np.uint16)
            .reshape(frame.get_height(), frame.get_width()).astype(np.float32)
            * float(frame.get_depth_scale()) * .001)


class OrbbecCamera(RealSenseCamera):
    """Reuse validated session settings and the context-manager interface only."""
    backend = 'orbbec'
    optical_frame = 'orbbec_color_optical_frame'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.laser_power is not None:
            raise ValueError('Orbbec cannot reuse D435 laser_power=360; omit camera-laser-power')
        self._pending = deque(maxlen=1)
        self._arrival = threading.Condition()
        self._maps = None

    def _on_frame(self, frames):
        with self._arrival:
            self._pending.append((frames, time.monotonic()))
            self._arrival.notify()

    def start(self):
        with self.lock:
            if self.pipeline is not None:
                return self
            import pyorbbecsdk as ob
            self._ob = ob
            self._context = ob.Context()
            devices = self._context.query_devices()
            selected = [i for i in range(devices.get_count())
                        if not self.serial or devices.get_device_serial_number_by_index(i) == self.serial]
            if len(selected) != 1:
                raise RuntimeError(f'Expected one selected Orbbec camera; found {len(selected)}')
            device = devices.get_device_by_index(selected[0])
            info = device.get_device_info()
            self.serial = info.get_serial_number()
            print(f'[camera] backend=orbbec model={info.get_name()} serial={self.serial}')
            if self.enable_depth and self.depth_preset == 'high_density':
                presets = device.get_available_preset_list()
                names = [presets.get_name_by_index(i) for i in range(presets.get_count())]
                if 'High Density' not in names:
                    raise ValueError(f'Orbbec High Density preset unavailable; presets={names}; '
                                     'use --camera-depth-preset unchanged')
                device.load_preset('High Density')
            if self.enable_depth and self.emitter_enabled is not None:
                device.set_bool_property(ob.OBPropertyID.OB_PROP_LASER_BOOL, self.emitter_enabled)
            pipe = ob.Pipeline(device)
            config = ob.Config()
            color = select_profile(pipe.get_stream_profile_list(ob.OBSensorType.COLOR_SENSOR),
                                   self.width, self.height, self.fps,
                                   (ob.OBFormat.BGR, ob.OBFormat.RGB, ob.OBFormat.MJPG, ob.OBFormat.RGBA))
            config.enable_stream(color)
            ci, dist = color.get_intrinsic(), color.get_distortion()
            self.K = np.array([[ci.fx, 0., ci.cx], [0., ci.fy, ci.cy], [0., 0., 1.]])
            self.D_source = np.array([dist.k1, dist.k2, dist.p1, dist.p2,
                                      dist.k3, dist.k4, dist.k5, dist.k6])
            if np.any(np.abs(self.D_source) > 1e-9) and dist.model not in (
                    ob.OBCameraDistortionModel.BROWN_CONRADY,
                    ob.OBCameraDistortionModel.BROWN_CONRADY_K6):
                raise ValueError(f'Orbbec color distortion model requires SDK-specific rectification: {dist.model}')
            self._maps = (cv2.initUndistortRectifyMap(self.K, self.D_source, None, self.K,
                                                    (self.width, self.height), cv2.CV_32FC1)
                          if np.any(np.abs(self.D_source) > 1e-9) else None)
            self.align = None
            if self.enable_depth:
                depth = select_profile(pipe.get_stream_profile_list(ob.OBSensorType.DEPTH_SENSOR),
                                       self.depth_width, self.depth_height, self.depth_fps, (ob.OBFormat.Y16,))
                config.enable_stream(depth)
                config.set_frame_aggregate_output_mode(ob.OBFrameAggregateOutputMode.FULL_FRAME_REQUIRE)
                pipe.enable_frame_sync()
                self.align = ob.AlignFilter(align_to_stream=ob.OBStreamType.COLOR_STREAM)
                # Match the old Gemini ROS setup's noise-only approach. Do not
                # activate spatial/temporal/hole filling from the D435 chain.
                self._depth_filters = [ob.NoiseRemovalFilter()] if self.depth_filters == 'all' else []
                for block in self._depth_filters:
                    block.enable(True)
            self._pending.clear()
            self.pipeline = pipe
            try:
                pipe.start(config, self._on_frame)
                for _ in range(self.warmup):
                    self.read()
            except BaseException:
                self.close()
                raise
            if self.enable_depth:
                print(f'[camera] RGB={self.width}x{self.height}@{self.fps}; '
                      f'depth={self.depth_width}x{self.depth_height}@{self.depth_fps}; '
                      f'align=software_D2C rectify={self._maps is not None}; '
                      f'filters={"noise_only" if self._depth_filters else "off"}; holes=off')
            else:
                print(f'[camera] RGB={self.width}x{self.height}@{self.fps}; '
                      f'depth=disabled; rectify={self._maps is not None} (RGB-only capture)')
            return self

    def read(self):
        with self.lock:
            self.start()
            deadline = time.monotonic() + self.timeout
            while time.monotonic() < deadline:
                with self._arrival:
                    if not self._arrival.wait_for(lambda: bool(self._pending), max(0., deadline-time.monotonic())):
                        break
                    frames, received = self._pending.pop()
                if self.enable_depth:
                    aligned = self.align.process(frames)
                    if aligned is None:
                        continue
                    frames = aligned.as_frame_set()
                color = frames.get_color_frame()
                depth = frames.get_depth_frame() if self.enable_depth else None
                if color is None or (self.enable_depth and depth is None):
                    continue
                source = depth if depth is not None else color
                sequence = source.get_index()
                if sequence == self.last_sequence:
                    continue
                bgr = color_to_bgr(color, self._ob)
                raw = depth_metres(depth) if self.enable_depth else None
                z = raw
                for block in self._depth_filters:
                    depth = block.process(depth).as_depth_frame()
                    z = depth_metres(depth)
                if self.enable_depth and z.shape != bgr.shape[:2]:
                    raise RuntimeError('Orbbec SDK D2C shape mismatch; refusing unaligned depth')
                if self._maps is not None:
                    bgr = cv2.remap(bgr, *self._maps, cv2.INTER_LINEAR)
                    if self.enable_depth:
                        raw = cv2.remap(raw, *self._maps, cv2.INTER_NEAREST)
                        z = cv2.remap(z, *self._maps, cv2.INTER_NEAREST)
                intr = SimpleNamespace(fx=self.K[0, 0], fy=self.K[1, 1], cx=self.K[0, 2], cy=self.K[1, 2])
                self.last_sequence = sequence
                return Frame(bgr, z, intr, self.K.copy(), np.zeros(8), sequence,
                             source.get_timestamp(), received, self.serial, raw, self.backend)
            raise TimeoutError('No new aligned Orbbec SDK frame')

    def close(self):
        with self.lock:
            pipe, self.pipeline = self.pipeline, None
            if pipe is not None:
                pipe.stop()
            self._pending.clear()
            self._depth_filters = []
            self._maps = None
            self.last_sequence = None
