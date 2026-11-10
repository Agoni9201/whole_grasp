"""Persistent D435 RGB-D / D435-or-D455/D457 RGB capture; lazy SDK import."""
import atexit
import json
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
import numpy as np


@dataclass
class Frame:
    bgr: np.ndarray
    depth_m: np.ndarray
    intr: object
    K: np.ndarray
    D: np.ndarray
    sequence: int
    hardware_ms: float
    received_monotonic: float
    serial: str
    raw_depth_m: np.ndarray | None = None
    backend: str = 'realsense'


def depth_with_current_support(depth_m, raw_depth_m, minimum, maximum):
    """Denoise current measurements without treating filled holes as evidence.

    Raw depth uses the exact same color alignment and frame as filtered depth.
    Missing/currently out-of-range pixels cannot acquire a valid hand distance
    solely through interpolation or a temporal filter's stored scene history.
    """
    if depth_m is None or raw_depth_m is None:
        return depth_m
    if raw_depth_m.shape != depth_m.shape:
        raise ValueError('Raw/filtered depth alignment shape mismatch')
    supported = (np.isfinite(raw_depth_m) & (raw_depth_m > 0)
                 & (raw_depth_m >= minimum) & (raw_depth_m <= maximum))
    return np.where(supported, depth_m, 0.).astype(np.float32, copy=False)


class RealSenseCamera:
    backend = 'realsense'
    optical_frame = 'd435_color_optical_frame'
    def __init__(self, serial=None, width=640, height=480, fps=30, timeout=10., warmup=30,
                 enable_depth=True, depth_preset='high_density', emitter_enabled=None,
                 laser_power=None, depth_width=None, depth_height=None, depth_fps=None,
                 depth_filters='off'):
        if min(width, height, fps) <= 0 or timeout <= 0 or warmup < 0:
            raise ValueError('Invalid camera profile/timeout/warmup')
        self.serial, self.width, self.height, self.fps = serial, width, height, fps
        self.timeout, self.warmup = timeout, warmup
        self.enable_depth = bool(enable_depth)
        self.depth_width = width if depth_width is None else depth_width
        self.depth_height = height if depth_height is None else depth_height
        self.depth_fps = fps if depth_fps is None else depth_fps
        if min(self.depth_width, self.depth_height, self.depth_fps) <= 0:
            raise ValueError('Invalid native depth profile')
        if depth_filters not in ('off', 'all'):
            raise ValueError('Invalid depth filters (expected off or all)')
        self.depth_filters = depth_filters
        self._depth_filters = []
        self.depth_preset = str(depth_preset).strip().lower()
        if self.depth_preset not in ('high_density', 'unchanged'):
            raise ValueError(f'Unsupported D400 depth preset: {depth_preset!r}')
        self.emitter_enabled = (
            None if emitter_enabled is None else bool(emitter_enabled)
        )
        self.laser_power = None if laser_power is None else float(laser_power)
        self.pipeline = None
        self.last_sequence = None
        self.lock = threading.RLock()

    def start(self):
        with self.lock:
            if self.pipeline is not None:
                return self
            import pyrealsense2 as rs
            ctx = rs.context()
            # D455/D457 are supported only as independent 2D RGB observers.
            # Keep the existing D435-only metric-depth/grasp contract intact.
            families = ('435',) if self.enable_depth else ('435', '455', '457')
            devices = [d for d in ctx.query_devices()
                       if any(family in d.get_info(rs.camera_info.name).upper() for family in families)
                       and (not self.serial or d.get_info(rs.camera_info.serial_number) == self.serial)]
            if len(devices) != 1:
                label = 'D435-family RGB-D' if self.enable_depth else 'D435/D455/D457 RGB'
                raise RuntimeError(f'Expected one selected {label} camera; found {len(devices)}')
            self.serial = devices[0].get_info(rs.camera_info.serial_number)
            config = rs.config()
            config.enable_device(self.serial)
            config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
            if self.enable_depth:
                config.enable_stream(rs.stream.depth, self.depth_width, self.depth_height,
                                     rs.format.z16, self.depth_fps)
            pipe = rs.pipeline(ctx)
            profile = pipe.start(config)
            self.pipeline = pipe
            try:
                if self.enable_depth:
                    depth_sensor = profile.get_device().first_depth_sensor()
                    self.scale = depth_sensor.get_depth_scale()
                    try:
                        if (
                            self.depth_preset == 'high_density'
                            and depth_sensor.supports(rs.option.visual_preset)
                        ):
                            depth_sensor.set_option(
                                rs.option.visual_preset,
                                int(rs.rs400_visual_preset.high_density),
                            )
                            print(
                                "[camera] visual_preset="
                                f"{int(depth_sensor.get_option(rs.option.visual_preset))} "
                                "(high_density)"
                            )
                    except (AttributeError, TypeError, RuntimeError) as exc:
                        print(f"[camera] visual_preset high_density skipped: {exc}")
                    try:
                        if (
                            self.emitter_enabled is not None
                            and depth_sensor.supports(rs.option.emitter_enabled)
                        ):
                            value = 1.0 if self.emitter_enabled else 0.0
                            depth_sensor.set_option(rs.option.emitter_enabled, value)
                            print(
                                '[camera] emitter_enabled='
                                f"{int(depth_sensor.get_option(rs.option.emitter_enabled))}"
                            )
                    except (AttributeError, TypeError, RuntimeError) as exc:
                        print(f"[camera] emitter setting skipped: {exc}")
                    try:
                        if (
                            self.laser_power is not None
                            and depth_sensor.supports(rs.option.laser_power)
                        ):
                            power_range = depth_sensor.get_option_range(rs.option.laser_power)
                            if not float(power_range.min) <= self.laser_power <= float(power_range.max):
                                raise ValueError(
                                    'camera laser power '
                                    f'{self.laser_power:g} is outside device range '
                                    f'[{float(power_range.min):g}, {float(power_range.max):g}]'
                                )
                            depth_sensor.set_option(rs.option.laser_power, self.laser_power)
                            print(
                                '[camera] laser_power='
                                f"{depth_sensor.get_option(rs.option.laser_power):g}/"
                                f"{float(power_range.max):g}"
                            )
                    except (AttributeError, TypeError, RuntimeError) as exc:
                        print(f"[camera] laser power setting skipped: {exc}")
                    self.align = rs.align(rs.stream.color)
                    if self.depth_filters == 'all':
                        spatial = rs.spatial_filter()
                        for option, value in ((rs.option.filter_magnitude, 2),
                                              (rs.option.filter_smooth_alpha, .5),
                                              (rs.option.filter_smooth_delta, 20),
                                              (rs.option.holes_fill, 2)):
                            spatial.set_option(option, value)
                        temporal = rs.temporal_filter()
                        temporal.set_option(rs.option.filter_smooth_alpha, .4)
                        temporal.set_option(rs.option.filter_smooth_delta, 20)
                        temporal.set_option(rs.option.holes_fill, 2)  # Valid in 2/last 3; never indefinite.
                        self._depth_filters = [rs.disparity_transform(True), spatial, temporal,
                                               rs.disparity_transform(False), rs.hole_filling_filter(2)]
                else:
                    self.scale = None
                    self.align = None
                print(f'[camera] RGB={self.width}x{self.height}@{self.fps}; '
                      f'depth={self.depth_width}x{self.depth_height}@{self.depth_fps}'
                      if self.enable_depth else
                      f'[camera] RGB={self.width}x{self.height}@{self.fps}; depth=disabled')
                if self.enable_depth:
                    print(f'[camera] depth_filters={self.depth_filters}; '
                          'decimation=off; raw aligned depth retained when filtering')
                for _ in range(self.warmup):
                    pipe.wait_for_frames(int(self.timeout * 1000))
            except BaseException:
                self.close()
                raise
            return self

    def read(self):
        with self.lock:
            self.start()
            deadline = time.monotonic() + self.timeout
            while time.monotonic() < deadline:
                frames = self.pipeline.wait_for_frames(
                    max(1, int((deadline-time.monotonic())*1000)))
                # Full-resolution filtering may exceed one frame period.
                # Discard queued old frames before processing the newest one.
                poll = getattr(self.pipeline, 'poll_for_frames', None)
                if callable(poll):
                    for _ in range(16):
                        newer = poll()
                        if not newer:
                            break
                        frames = newer
                received = time.monotonic()  # Include alignment/filter time in downstream frame age.
                if self.enable_depth:
                    frames = self.align.process(frames)
                color = frames.get_color_frame()
                depth = frames.get_depth_frame() if self.enable_depth else None
                if not color or (self.enable_depth and not depth):
                    continue
                source_frame = depth if depth is not None else color
                seq = source_frame.get_frame_number()
                if seq == self.last_sequence:
                    continue
                ci = color.profile.as_video_stream_profile().intrinsics
                # Existing grasp code uses pinhole back-projection. Reject unsupported raw distortion.
                if self.enable_depth and np.any(np.abs(ci.coeffs) > 1e-9):
                    raise RuntimeError('Nonzero color distortion: rectify before pinhole back-projection')
                bgr = np.asanyarray(color.get_data()).copy()
                z = None
                raw = None
                if self.enable_depth:
                    z = np.asanyarray(depth.get_data()).astype(np.float32) * float(self.scale)
                    if self._depth_filters:
                        raw = z.copy()
                        # All filters preserve the already-aligned color geometry.
                        # Keep instances across frames for temporal history.
                        for block in self._depth_filters:
                            depth = block.process(depth)
                        z = np.asanyarray(depth.get_data()).astype(np.float32) * float(self.scale)
                    if bgr.shape[:2] != z.shape:
                        raise RuntimeError('SDK color/depth alignment shape mismatch')
                self.last_sequence = seq
                intr = SimpleNamespace(fx=ci.fx, fy=ci.fy, cx=ci.ppx, cy=ci.ppy)
                return Frame(bgr, z, intr, np.array([[ci.fx,0,ci.ppx],[0,ci.fy,ci.ppy],[0,0,1.]]),
                             np.asarray(ci.coeffs), seq, source_frame.get_timestamp(), received, self.serial, raw)
            raise TimeoutError('No new aligned SDK frame')

    def close(self):
        with self.lock:
            pipe, self.pipeline = self.pipeline, None
            self.last_sequence = None
            self._depth_filters = []  # Do not reuse another session's temporal history.
            if pipe is not None:
                # pipe.stop()
                try:
                    pipe.stop()
                except Exception as exc:
                    print(f"[camera] pipeline stop ignored: {exc}")

    def __enter__(self): return self.start()
    def __exit__(self, *exc): self.close()


def create_camera(*args, backend='realsense', **kwargs):
    """Construct the selected adapter without opening USB or importing its SDK."""
    if backend == 'realsense':
        return RealSenseCamera(*args, **kwargs)
    if backend == 'orbbec':
        from .orbbec import OrbbecCamera
        return OrbbecCamera(*args, **kwargs)
    raise ValueError(f'Unknown camera backend: {backend}')


_sessions = {}

def camera_for(args):
    backend = getattr(args, 'camera_backend', 'realsense')
    settings = (getattr(args, 'camera_serial', None), getattr(args, 'camera_width', 640),
                getattr(args, 'camera_height', 480), getattr(args, 'camera_fps', 30),
                getattr(args, 'camera_timeout', 10.), getattr(args, 'camera_warmup', 30),
                True, getattr(args, 'camera_depth_preset', 'high_density'),
                getattr(args, 'camera_emitter_enabled', None),
                getattr(args, 'camera_laser_power', None),
                getattr(args, 'camera_depth_width', None),
                getattr(args, 'camera_depth_height', None),
                getattr(args, 'camera_depth_fps', None),
                getattr(args, 'camera_depth_filters', 'off'))
    key = (backend, *settings)
    if key not in _sessions:
        if backend == 'orbbec' or settings[-1] == 'all':
            from .process_camera import ProcessCamera
            _sessions[key] = ProcessCamera(*settings, backend=backend)
        else:
            _sessions[key] = create_camera(*settings, backend=backend)
    return _sessions[key]


def close_all():
    sessions = list(_sessions.values())
    _sessions.clear()
    errors = []
    for camera in sessions:
        try:
            camera.close()
        except Exception as exc:
            errors.append(exc)
    if errors:
        raise RuntimeError('Failed to close camera session') from errors[0]


atexit.register(close_all)


def capture_rgbd(args):
    from .depth_fusion import median_valid_depth_frames, supported_depth_median
    camera = camera_for(args)
    # Discard queued frame after robot motion; next read must have a new sequence.
    camera.read()
    count = max(1, int(getattr(args, 'orbbec_depth_median_frames', 1)))
    frames = [camera.read() for _ in range(count)]
    frame = frames[-1]
    offload = (str(getattr(args, 'trajectory_executor', 'move_j')) == 'mit'
               and not bool(getattr(args, '_planning_worker', False)))
    raw_present = [f.raw_depth_m is not None for f in frames]
    if any(raw_present) and not all(raw_present):
        raise RuntimeError('Incomplete raw depth window; refusing unsupported SDK fusion')
    support = None
    if all(raw_present):
        if offload:
            from real_nero_grasp.planning_process import isolated_supported_depth_median
            support = isolated_supported_depth_median(
                [f.depth_m for f in frames], [f.raw_depth_m for f in frames], args,
            )
        else:
            support = supported_depth_median(
                [f.depth_m for f in frames], [f.raw_depth_m for f in frames],
                min_valid_frames=getattr(args, 'orbbec_depth_median_min_valid', 1),
                max_mad=getattr(args, 'orbbec_depth_median_max_mad', None),
            )
        depth = support['depth_m']
    elif offload:
        from real_nero_grasp.planning_process import isolated_depth_median
        depth, _ = isolated_depth_median([f.depth_m for f in frames], args)
    else:
        depth, _ = median_valid_depth_frames(
            [f.depth_m for f in frames],
            min_valid_frames=getattr(args, 'orbbec_depth_median_min_valid', 1),
            max_mad=getattr(args, 'orbbec_depth_median_max_mad', None))
    import cv2
    from datetime import datetime
    out = Path(getattr(args, 'save_dir', None) or 'captures') / datetime.now().strftime('%Y-%m-%d_%H-%M-%S_%f')
    out.mkdir(parents=True, exist_ok=False)
    if not cv2.imwrite(str(out/'color.png'), frame.bgr):
        raise OSError('Cannot save RGB capture')
    np.save(out/'depth_m.npy', depth)
    if support is not None:
        np.save(out/'depth_sdk_m.npy', support['sdk_depth_m'])
        np.save(out/'depth_raw_valid_count.npy', support['raw_valid_count'])
        if not cv2.imwrite(str(out/'depth_support_rejected.png'), support['rejected_mask'] * 255):
            raise OSError('Cannot save raw depth support diagnostic')
        (out/'depth_support_report.json').write_text(json.dumps(support['report'], indent=2) + '\n')
        print(f"[camera][depth_support] raw_frames={count} "
              f"min_valid={support['report']['min_valid_frames']} "
              f"rejected_pixels={support['report']['rejected_pixels']} "
              f"valid_pixels={support['report']['valid_pixels']}")
    if frame.raw_depth_m is not None:
        np.save(out/'depth_raw_m.npy', frame.raw_depth_m)
    raw_window_saved = all(f.raw_depth_m is not None for f in frames)
    if raw_window_saved:
        # Keep the whole settled window: the last raw frame alone cannot tell
        # whether a processed median came from current returns or SDK history.
        np.savez_compressed(
            out/'depth_raw_frames.npz',
            depth_m=np.stack([f.raw_depth_m for f in frames]),
            sequence=np.asarray([f.sequence for f in frames]),
            hardware_ms=np.asarray([f.hardware_ms for f in frames]),
            received_monotonic=np.asarray([f.received_monotonic for f in frames]),
        )
    (out/'capture_meta.json').write_text(json.dumps(dict(
        intrinsics=vars(frame.intr), width=frame.bgr.shape[1], height=frame.bgr.shape[0],
        serial=frame.serial, backend=frame.backend, K=frame.K.tolist(), D=frame.D.tolist(),
        sequence=frame.sequence, hardware_ms=frame.hardware_ms,
        received_monotonic=frame.received_monotonic, depth_unit='metre',
        temporal_frames=count, depth_filters=getattr(camera, 'depth_filters', 'off'),
        raw_temporal_file='depth_raw_frames.npz' if raw_window_saved else None,
        raw_temporal_frames=count if raw_window_saved else 0,
        depth_support_mode='same_window_raw_support' if support is not None else 'unfiltered_window',
        depth_support_report='depth_support_report.json' if support is not None else None,
        native_depth_width=getattr(camera, 'depth_width', frame.bgr.shape[1]),
        native_depth_height=getattr(camera, 'depth_height', frame.bgr.shape[0])), indent=2))
    return frame.bgr, depth, frame.intr, str(out)


def capture_legacy(save_dir, color_topic=None, depth_topic=None, camera_info_topic=None,
                   depth_scale=None, timeout_s=10., warmup_frames=30,
                   depth_median_frames=1, depth_median_min_valid=1, depth_median_max_mad=None):
    """Old tools' positional signature; topic arguments do not select a transport."""
    return capture_rgbd(SimpleNamespace(save_dir=save_dir, camera_timeout=timeout_s,
        camera_warmup=warmup_frames, orbbec_depth_median_frames=depth_median_frames,
        orbbec_depth_median_min_valid=depth_median_min_valid,
        orbbec_depth_median_max_mad=depth_median_max_mad))


def organized_cloud(args):
    """Array-backed organized XYZ envelope for legacy pure-array consumers, not ROS."""
    from real_nero_grasp.depth_processing import depth_to_xyz_image
    bgr, depth, intr, out = capture_rgbd(args)
    xyz = np.ascontiguousarray(depth_to_xyz_image(depth, intr), dtype='<f4')
    h, w = depth.shape
    msg = SimpleNamespace(height=h, width=w, data=xyz.tobytes(), is_bigendian=False,
        point_step=12, row_step=12*w, fields=[SimpleNamespace(name=n, offset=i*4, datatype=7, count=1)
        for i,n in enumerate(('x','y','z'))], header=SimpleNamespace(frame_id='aligned_color_optical_frame'))
    return bgr, msg, intr, out
