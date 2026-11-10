"""Read-only servo display in a clean subprocess; no camera or CAN ownership."""
from __future__ import annotations

import os
from pathlib import Path
import pickle
import queue
import select
import struct
import subprocess
import sys
import threading
import time

_WINDOW = "Hand servo RGB-D | Q/Esc/close: controlled exit"
_PACKET_HEADER = struct.Struct("!I")
_MAX_PACKET_BYTES = 32 * 1024 * 1024
_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15),
    (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
)


class ServoPreview:
    """Bounded, non-blocking producer; drawing/serialization never runs in MIT."""

    def __init__(self):
        self._pending = queue.Queue(maxsize=1)
        self._ready = threading.Event()
        self._stop = threading.Event()
        self._event = None
        self._process = None
        self._threads = []

    def start(self):
        if sys.platform.startswith("linux") and not (os.getenv("DISPLAY") or os.getenv("WAYLAND_DISPLAY")):
            raise RuntimeError("--servo-preview needs a desktop display; omit it for headless use")
        # A clean interpreter, not fork/spawn of main.py: never inherit/rebuild
        # the robot driver, models, detector, or controller from the parent.
        self._process = subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), "--worker"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, close_fds=True,
            start_new_session=True,
            env={**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
        )
        self._threads = [threading.Thread(target=fn, daemon=True, name=name)
                         for fn, name in ((self._write, "servo-preview-send"),
                                          (self._read_events, "servo-preview-events"))]
        for thread in self._threads:
            thread.start()
        if not self._ready.wait(5.0) or self.poll() is not None:
            event = self.poll()
            self.close()
            raise RuntimeError(f"Servo preview could not open: {event or 'startup timeout'}")

    def _write(self):
        stream = self._process.stdin
        try:
            while not self._stop.is_set():
                try:
                    payload = self._pending.get(timeout=0.1)
                except queue.Empty:
                    continue
                data = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
                if len(data) > _MAX_PACKET_BYTES:
                    raise ValueError("display frame is too large")
                stream.write(_PACKET_HEADER.pack(len(data)) + data)
                stream.flush()
        except Exception as exc:
            if not self._stop.is_set():
                self._event = f"error: preview input: {exc}"
        finally:
            # EOF lets the GUI exit normally after parent controlled shutdown.
            try:
                stream.close()
            except OSError:
                pass

    def _read_events(self):
        for line in self._process.stdout:
            message = line.decode("utf-8", errors="replace").strip()
            if message == "ready":
                self._ready.set()
            elif message == "quit" or message.startswith("error:"):
                self._event = message
        if not self._stop.is_set() and self._event is None:
            self._event = "error: preview process exited"
        self._ready.set()

    def poll(self):
        if self._event is not None:
            return self._event
        if self._process is not None and self._process.poll() is not None:
            return "error: preview process exited"
        return None

    def publish(self, payload):
        if self._stop.is_set():
            return
        try:
            self._pending.put_nowait(payload)
        except queue.Full:
            # Keep the newest pending image, not the oldest unsent image.
            try:
                self._pending.get_nowait()
            except queue.Empty:
                pass
            try:
                self._pending.put_nowait(payload)
            except queue.Full:
                pass

    def close(self):
        self._stop.set()
        process = self._process
        if process is None:
            return
        try:
            process.wait(timeout=1.0)
        except subprocess.TimeoutExpired:
            process.terminate()  # Display only; no robot/control connection.
            try:
                process.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=1.0)
        for thread in self._threads:
            thread.join(timeout=1.0)
        # The writer owns stdin; never close its buffered stream concurrently.
        if not self._threads or not self._threads[0].is_alive():
            process.stdin.close()
        if len(self._threads) < 2 or not self._threads[1].is_alive():
            process.stdout.close()
        self._process = None


class _FrameReader:
    """Parse bounded packets using raw, non-blocking reads, never buffered stdin."""

    def __init__(self, fd):
        self._fd = fd
        self._buffer = bytearray()
        self._expected = None
        os.set_blocking(fd, False)

    def poll(self):
        latest = None
        # Keep GUI event handling responsive even if frames arrive continuously.
        remaining = 4 * 1024 * 1024
        while remaining > 0:
            try:
                data = os.read(self._fd, min(65536, remaining))
            except BlockingIOError:
                return latest, False
            if not data:
                if self._buffer or self._expected is not None:
                    raise EOFError("incomplete display frame")
                return latest, True
            remaining -= len(data)
            self._buffer.extend(data)
            while True:
                if self._expected is None:
                    if len(self._buffer) < _PACKET_HEADER.size:
                        break
                    self._expected = _PACKET_HEADER.unpack_from(self._buffer)[0]
                    del self._buffer[:_PACKET_HEADER.size]
                    if not 0 < self._expected <= _MAX_PACKET_BYTES:
                        raise ValueError("invalid display frame size")
                if len(self._buffer) < self._expected:
                    break
                latest = pickle.loads(self._buffer[:self._expected])
                del self._buffer[:self._expected]
                self._expected = None
        return latest, False


class _LatestFrameReceiver:
    """Drain the pipe independently of GUI rendering; retain one latest frame."""

    def __init__(self, fd):
        self._fd = fd
        self._reader = _FrameReader(fd)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._latest = None
        self._ended = False
        self._error = None
        self._thread = threading.Thread(
            target=self._receive, name="servo-preview-receive", daemon=True)

    def start(self):
        self._thread.start()

    def _receive(self):
        try:
            while not self._stop.is_set():
                readable, _, _ = select.select([self._fd], [], [], 0.05)
                if not readable:
                    continue
                update, ended = self._reader.poll()
                with self._lock:
                    if update is not None:
                        self._latest = update
                    self._ended = ended
                if ended:
                    return
        except Exception as exc:
            with self._lock:
                self._error = exc

    def poll(self):
        with self._lock:
            if self._error is not None:
                raise self._error
            latest, self._latest = self._latest, None
            return latest, self._ended

    def close(self):
        self._stop.set()
        # select is interruptible by this bounded timeout; no thread owns a
        # Python buffered-stdin lock when the subprocess interpreter exits.
        if self._thread.ident is not None:
            self._thread.join(timeout=1.0)


def overlay_lines(payload, now):
    """Pure status formatting, also used in offline tests."""
    age = max(0.0, now - payload["received_monotonic"])
    status_age = max(0.0, now - payload["updated_monotonic"])
    stale = age > 0.45 or status_age > 0.45
    status = "STALE DISPLAY - do not rely on old tracking status" if stale else payload["status"]
    def xyz(value):
        return "unavailable" if value is None else " ".join(f"{v:+.1f}" for v in value)
    lines = [
        f"Servo: {status} | stage: {payload['stage']}",
        f"Reason: {payload['reason']}",
        f"Frame age: {age * 1000:.0f} ms | status age: {status_age * 1000:.0f} ms",
        f"camera XYZ mm: {xyz(payload['camera_xyz_mm'])}",
        f"Depth points: {payload['depth_points']}/5 (need {payload['required_depth_points']})",
        f"base XYZ mm: {xyz(payload['base_xyz_mm'])}",
        f"Tracking distance: {payload['distance_m']:.2f} m | error: {payload['error_mm']} mm",
        "Q / Esc / close window: controlled exit (not emergency stop)",
    ]
    gesture = payload.get('gesture_handover')
    memory = payload.get('hand_memory')
    if memory and payload.get('status') == 'MEMORY_APPROACH':
        lines += [f"MEMORY ONLY: command {memory.get('command_travel_mm', 0.):.1f}/"
                  f"{memory.get('maximum_travel_mm', 3.):g} mm | {memory.get('reason')}"]
    if gesture:
        fraction = gesture.get('fraction')
        percent = '--' if fraction is None else f'{fraction*100:.0f}%'
        lines += [f"Handover phase: {gesture.get('phase', '--')}",
                  f"Receiver: {gesture.get('receiver_side')} | control: {gesture.get('control_side')}",
                  f"Receiver still: {gesture.get('receiver_still_seconds', 0.):.1f}/"
                  f"{gesture.get('receiver_still_required_seconds', 3.):g}s | "
                  f"handover ready: {bool(gesture.get('handover_ready'))}",
                  f"Prepare within: {gesture.get('ready_distance_mm', 150.):g} mm | "
                  f"release started: {bool(gesture.get('release_started'))}",
                  f"Handover: {gesture.get('reason', '--')}",
                  f"Pinch: {percent} | gripper actual: {gesture.get('actual_width_mm', '--')} mm",
                  'Control hand lost -> pause at current opening; no automatic re-close']
        center = payload.get('camera_center')
        if center and center.get('pixel_error') is not None:
            pixel_error = center['pixel_error']
            error_px = sum(float(value) ** 2 for value in pixel_error) ** 0.5
            lines.append(f"A hand center error: {error_px:.0f} px (need <12 px to learn gesture)")
    return lines


def draw_payload(payload, now):
    import cv2
    import numpy as np
    image = np.frombuffer(payload["data"], dtype=np.uint8).reshape(
        payload["height"], payload["width"], -1).copy()
    if payload["encoding"] == "rgb8":
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if payload["landmarks"] is not None:
        pixels = np.asarray(payload["landmarks"], dtype=int)
        for start, end in _CONNECTIONS:
            cv2.line(image, tuple(pixels[start]), tuple(pixels[end]), (40, 220, 40), 2)
        for pixel in pixels:
            cv2.circle(image, tuple(pixel), 3, (20, 80, 255), -1)
        cv2.circle(image, tuple(payload["palm_pixel"]), 7, (255, 180, 0), 2)
    if payload.get('control_landmarks') is not None:
        points = np.asarray(payload['control_landmarks'], dtype=int)
        for start, end in _CONNECTIONS:
            cv2.line(image, tuple(points[start]), tuple(points[end]), (255, 180, 20), 2)
        cv2.line(image, tuple(points[4]), tuple(points[8]), (0, 220, 255), 3)
    center = payload.get('camera_center')
    if center and center.get('desired_pixel') is not None:
        target_pixel = tuple(np.rint(center['desired_pixel']).astype(int))
        cv2.drawMarker(image, target_pixel, (0, 255, 255), cv2.MARKER_CROSS, 24, 2)
        if payload.get('palm_pixel') is not None:
            cv2.line(image, tuple(payload['palm_pixel']), target_pixel, (0, 255, 255), 1)
    # Draw on a separate panel: text never hides the palm/depth geometry.
    width = 960
    height = max(1, round(image.shape[0] * width / image.shape[1]))
    image = cv2.resize(image, (width, height))
    lines = overlay_lines(payload, now)
    panel = np.zeros((max(224, 8+27*len(lines)), width, 3), dtype=np.uint8)
    for index, line in enumerate(lines):
        color = (40, 220, 40) if index == 0 and payload["status"] == "TRACKING" else (0, 220, 255)
        if index == 0 and line.startswith("Servo: STALE"):
            color = (0, 80, 255)
        cv2.putText(panel, line, (10, 23 + index * 27), cv2.FONT_HERSHEY_SIMPLEX, 0.54, color, 1, cv2.LINE_AA)
    return np.vstack((image, panel))


def _worker():
    # The worker imports only GUI/numeric libraries and receives display data.
    # It does not import servo_delivery, pyAgxArm, RealSense, or MediaPipe.
    import cv2
    import numpy as np
    cv2.setNumThreads(1)
    latest = None
    reader = None
    try:
        reader = _LatestFrameReceiver(sys.stdin.fileno())
        reader.start()
        cv2.namedWindow(_WINDOW, cv2.WINDOW_NORMAL)
        cv2.imshow(_WINDOW, np.zeros((764, 960, 3), dtype=np.uint8))
        cv2.waitKey(1)
        print("ready", flush=True)
        while True:
            update, ended = reader.poll()
            if ended:
                return
            if update is not None:
                latest = update
            if latest is not None:
                cv2.imshow(_WINDOW, draw_payload(latest, time.monotonic()))
            key = cv2.waitKey(30) & 0xFF
            if key in (ord("q"), ord("Q"), 27) or cv2.getWindowProperty(_WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                print("quit", flush=True)
                return
    except Exception as exc:
        print(f"error: {type(exc).__name__}: {exc}", flush=True)
    finally:
        if reader is not None:
            reader.close()
        cv2.destroyAllWindows()


if __name__ == "__main__" and sys.argv[1:] == ["--worker"]:
    _worker()
