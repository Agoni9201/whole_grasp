"""Camera-only subprocess with bounded shared RGB-D slots; no CAN ownership.

RealSense processing blocks can hold the interpreter lock. A capture thread
therefore cannot isolate them from the parent's Python CAN receiver. Only tiny
frame descriptors use IPC; full-resolution images use two locked mmap slots.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import multiprocessing
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import time

import numpy as np

_HEADER = struct.Struct('=Q')
_FIELDS = ('backend', 'serial', 'width', 'height', 'fps', 'timeout', 'warmup', 'enable_depth',
           'depth_preset', 'emitter_enabled', 'laser_power', 'depth_width',
           'depth_height', 'depth_fps', 'depth_filters')


def _views(buffer, height, width):
    pixels = height*width
    color = np.ndarray((height, width, 3), np.uint8, buffer, offset=_HEADER.size)
    depth = np.ndarray((height, width), np.float32, buffer,
                       offset=_HEADER.size+pixels*3)
    raw = np.ndarray((height, width), np.float32, buffer,
                     offset=_HEADER.size+pixels*7)
    return color, depth, raw


def _open_slots(paths, height, width):
    size = _HEADER.size+height*width*11
    return [(os.open(path, os.O_RDWR), np.memmap(path, dtype=np.uint8, mode='r+', shape=(size,)))
            for path in paths]


def _worker_main(connection, paths, settings, camera_factory=None):
    # This fresh interpreter never imports a controller or inherits its CAN
    # descriptor. Enforce that ownership even if a future import changes.
    import socket
    original_socket = socket.socket
    class CameraSocket(original_socket):
        def __init__(self, family=socket.AF_INET, *args, **kwargs):
            if family == getattr(socket, 'AF_CAN', None):
                raise RuntimeError('CAN connections are forbidden in the camera worker')
            super().__init__(family, *args, **kwargs)
    socket.socket = CameraSocket
    slots = []
    camera = None
    try:
        from .realsense import create_camera
        camera = (camera_factory or create_camera)(**settings)
        slots = _open_slots(paths, settings['height'], settings['width'])
        publication = 0
        with camera:
            while True:
                if connection.poll():
                    if connection.recv() is None:
                        break
                frame = camera.read()
                index = publication % len(slots)
                fd, buffer = slots[index]
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    continue  # Drop a frame rather than block behind a reader.
                try:
                    bgr, depth, raw = _views(buffer, settings['height'], settings['width'])
                    bgr[:] = frame.bgr
                    depth[:] = frame.depth_m
                    raw_present = frame.raw_depth_m is not None
                    if raw_present:
                        raw[:] = frame.raw_depth_m
                    publication += 1
                    buffer[:_HEADER.size] = np.frombuffer(_HEADER.pack(publication), np.uint8)
                finally:
                    fcntl.flock(fd, fcntl.LOCK_UN)
                connection.send(('frame', dict(
                    slot=index, publication=publication, sequence=frame.sequence,
                    hardware_ms=frame.hardware_ms, received_monotonic=frame.received_monotonic,
                    serial=frame.serial, intr=vars(frame.intr), K=frame.K.tolist(),
                    D=frame.D.tolist(), raw_present=raw_present, backend=frame.backend)))
    except (EOFError, BrokenPipeError):
        pass
    except BaseException as exc:
        try:
            connection.send(('error', f'{type(exc).__name__}: {exc}'))
        except (EOFError, OSError):
            pass
    finally:
        if camera is not None:
            camera.close()
        for fd, _ in slots:
            os.close(fd)
        connection.close()


class ProcessCamera:
    """Same read/close contract as RealSenseCamera; child owns USB and filters."""
    def __init__(self, *args, **kwargs):
        from .realsense import create_camera
        prototype = create_camera(*args, **kwargs)  # Validate only; no SDK/USB access.
        for name in _FIELDS:
            setattr(self, name, getattr(prototype, name))
        self.pipeline = None
        self._process = self._connection = self._directory = None
        self._slots = []
        self._closed = False

    def start(self):
        if self._closed:
            raise RuntimeError('Camera process is closed')
        if self._process is not None:
            return self
        self._directory = tempfile.TemporaryDirectory(prefix='grasp-camera-')
        paths = [str(Path(self._directory.name)/f'frame-{i}') for i in range(2)]
        size = _HEADER.size+self.height*self.width*11
        for path in paths:
            with open(path, 'wb') as stream:
                stream.truncate(size)
        parent, child = multiprocessing.Pipe()
        self._connection = parent
        try:
            self._slots = _open_slots(paths, self.height, self.width)
            environment = os.environ.copy()
            for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
                environment[name] = '1'
            environment['PYTHONUNBUFFERED'] = '1'
            root = str(Path(__file__).resolve().parents[1])
            environment['PYTHONPATH'] = os.pathsep.join(filter(None, (root, environment.get('PYTHONPATH'))))
            self._process = subprocess.Popen(
                [sys.executable, '-m', 'cameras.process_camera', '--worker-fd', str(child.fileno()),
                 '--slots', *paths, '--settings', json.dumps({name:getattr(self,name) for name in _FIELDS})],
                pass_fds=(child.fileno(),), close_fds=True, start_new_session=True,
                stdin=subprocess.DEVNULL, env=environment)
            print(f'[camera] isolated capture/filter worker pid={self._process.pid}; no CAN ownership')
        except BaseException:
            self.close()
            raise
        finally:
            child.close()
        return self

    def _snapshot(self, packet):
        from types import SimpleNamespace
        from .realsense import Frame
        fd, buffer = self._slots[packet['slot']]
        try:
            fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError:
            return None
        try:
            if _HEADER.unpack(buffer[:_HEADER.size])[0] != packet['publication']:
                return None  # Slot was already replaced; await a newer descriptor.
            bgr, depth, raw = _views(buffer, self.height, self.width)
            return Frame(bgr.copy(), depth.copy(), SimpleNamespace(**packet['intr']),
                         np.array(packet['K']), np.array(packet['D']), packet['sequence'],
                         packet['hardware_ms'], packet['received_monotonic'], packet['serial'],
                         raw.copy() if packet['raw_present'] else None, packet['backend'])
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)

    def read(self):
        self.start()
        deadline = time.monotonic()+self.timeout+2.
        while time.monotonic() < deadline:
            if not self._connection.poll(min(.1, max(0., deadline-time.monotonic()))):
                if self._process.poll() is not None:
                    raise RuntimeError('Camera worker exited without a frame')
                continue
            try:
                kind, value = self._connection.recv()
                for _ in range(32):
                    if kind != 'frame' or not self._connection.poll():
                        break
                    kind, value = self._connection.recv()
            except (EOFError, OSError) as exc:
                raise RuntimeError('Camera worker connection closed') from exc
            if kind != 'frame':
                raise RuntimeError(f'Camera worker failed: {value}')
            result = self._snapshot(value)
            if result is not None:
                self.serial = result.serial
                return result
        raise TimeoutError('Camera worker frame timeout')

    def close(self):
        if self._closed:
            return
        self._closed = True
        try:
            if self._connection is not None:
                try:
                    self._connection.send(None)
                except (OSError, EOFError):
                    pass
            if self._process is not None:
                try:
                    self._process.wait(timeout=self.timeout+2.)
                except subprocess.TimeoutExpired:
                    self._process.terminate()
                    try:
                        self._process.wait(timeout=2.)
                    except subprocess.TimeoutExpired:
                        self._process.kill()
                        self._process.wait(timeout=2.)
        finally:
            if self._connection is not None:
                self._connection.close()
            for fd, buffer in self._slots:
                buffer._mmap.close()
                os.close(fd)
            self._slots = []
            if self._directory is not None:
                self._directory.cleanup()

    def __enter__(self):
        return self.start()

    def __exit__(self, *args):
        self.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker-fd', type=int, required=True)
    parser.add_argument('--slots', nargs=2, required=True)
    parser.add_argument('--settings', required=True)
    cli = parser.parse_args()
    from multiprocessing.connection import Connection
    _worker_main(Connection(cli.worker_fd), cli.slots, json.loads(cli.settings))
