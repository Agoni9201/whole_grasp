"""Local SDK callback host for ported camera controllers; no ROS middleware.

Image/CameraInfo are plain in-process envelopes preserving tested conversion code.
Topic strings are descriptive labels only, not network endpoints.
"""
import logging
import threading
import time
from dataclasses import dataclass, field
from types import SimpleNamespace
from .realsense import camera_for
from control_backends.force_ctrl.normal_shutdown import defer_sigint

@dataclass
class Image:
    width: int = 0
    height: int = 0
    encoding: str = 'bgr8'
    step: int = 0
    data: bytes = b''
    header: object = None

@dataclass
class CameraInfo:
    k: list = field(default_factory=list)
    d: list = field(default_factory=list)

_running = False

def init(args=None):
    global _running
    _running = True

def shutdown():
    global _running
    _running = False

def ok(): return _running


class Node:
    def __init__(self, name):
        self._logger = logging.getLogger(name)
        self._callbacks = []
        self._timers = []
    def get_logger(self): return self._logger
    def create_subscription(self, kind, label, callback, *args, **kwargs):
        if kind not in (Image, CameraInfo):
            raise ValueError('SDK host accepts image/calibration callbacks only')
        self._callbacks.append((kind, label, callback))
    def create_timer(self, period, callback):
        if period <= 0: raise ValueError('Timer period must be positive')
        self._timers.append([period, callback, time.monotonic()])
    def destroy_node(self): pass


def spin(node):
    camera = camera_for(node.args)
    stop = threading.Event()
    latest = [None]
    failure = []
    def produce():
        try:
            while not stop.is_set():
                latest[0] = camera.read()
        except Exception as exc:
            failure.append(exc)
    worker = threading.Thread(target=produce, name='rgbd-sdk-capture', daemon=True)
    worker.start()
    last = None
    try:
        while ok():
            if failure: raise RuntimeError('SDK camera stream failed') from failure[0]
            frame = latest[0]
            if frame is not None and frame.sequence != last:
                last = frame.sequence
                if time.monotonic()-frame.received_monotonic > .45:
                    raise TimeoutError('SDK frame stale')
                # This stamp is host receipt time, NOT hardware exposure wall time.
                now = time.time(); sec=int(now)
                stamp=SimpleNamespace(sec=sec, nanosec=int((now-sec)*1e9))
                image=Image(frame.bgr.shape[1], frame.bgr.shape[0], 'bgr8', frame.bgr.shape[1]*3,
                            frame.bgr.tobytes(), SimpleNamespace(stamp=stamp))
                info=CameraInfo(frame.K.reshape(-1).tolist(), frame.D.tolist())
                node._latest_depth=frame.depth_m
                node._latest_raw_depth=getattr(frame, 'raw_depth_m', None)
                node._sdk_frame_sequence=frame.sequence
                node._sdk_received_monotonic=frame.received_monotonic
                for kind, label, callback in node._callbacks:
                    callback(info if kind is CameraInfo else image)
            now = time.monotonic()
            for timer in node._timers:
                if now >= timer[2]:
                    timer[2] = now+timer[0]
                    timer[1]()
                    if not ok(): break
            stop.wait(.005)
    finally:
        with defer_sigint():
            try:
                prepare = getattr(node, 'prepare_shutdown', None)
                if callable(prepare):
                    prepare()
            finally:
                stop.set()
                worker.join(timeout=camera.timeout+2.)
                if worker.is_alive():
                    raise RuntimeError('SDK capture worker did not exit')
                camera.close()
