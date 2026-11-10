"""Fixed RGB hand observer in a fresh camera-only interpreter (no CAN).

R confirms/selects the exposed-object ROI AFTER arm hold. B independently
declares no contact and requests baseline capture. Neither key opens a gripper.
"""
from __future__ import annotations

import json
import multiprocessing
import os
from pathlib import Path
import select
import subprocess
import sys
import time

import numpy as np


def validate_roi(roi):
    a = np.asarray(roi, float)
    if (a.shape != (4,) or not np.isfinite(a).all()
            or not 0 <= a[0] < a[2] <= 1 or not 0 <= a[1] < a[3] <= 1):
        raise ValueError('fusion ROI must be normalized XMIN YMIN XMAX YMAX in [0,1]')
    return a.tolist()


def record_hand_observations(observations):
    """Retain every detected hand, including rejected hands, for offline replay."""
    from .fusion_recording import json_value
    return [json_value(dict(index=i, handedness=o.handedness,
        handedness_score=float(o.handedness_score),
        landmarks_normalized=np.asarray(o.landmarks_normalized),
        landmarks_pixel=np.asarray(o.landmarks_pixel))) for i, o in enumerate(observations)]


def visual_evidence(observations, image_shape, roi, receiver_side):
    """Image proximity only, deliberately not a grip/contact classifier."""
    from targets.hand_selection import filter_hand_observations
    accepted, rejected = filter_hand_observations(observations, image_shape, fingertips=True)
    selected = [o for o in accepted if o.handedness == receiver_side]
    # A rejected unknown-side detection could be the receiver: do not override it.
    ambiguous = len(selected) != 1 or any(r['side'] == receiver_side
        or r['reason'] == 'hand side uncertain' for r in rejected)
    result = dict(visible=not ambiguous, in_roi=False, clear=False,
                  receiver_side=receiver_side, rejected=rejected, landmarks=None)
    if roi is None:
        return result
    box = np.asarray(validate_roi(roi))
    def hits(o):
        xy = np.asarray(o.landmarks_normalized)[:, :2]
        return (xy[:, 0] >= box[0]) & (xy[:, 0] <= box[2]) & (xy[:, 1] >= box[1]) & (xy[:, 1] <= box[3])
    result['clear'] = not rejected and not any(np.any(hits(o)) for o in accepted)
    if not ambiguous:
        # Two finger landmarks in the ROI, including a fingertip. This is still
        # only a candidate; a palm passing through the image cannot suffice.
        o = selected[0]
        hit = hits(o)
        result['in_roi'] = bool(np.count_nonzero(hit[1:]) >= 2 and np.any(hit[[4, 8, 12, 16, 20]]))
        result['landmarks'] = np.asarray(o.landmarks_pixel).tolist()
    return result


class FusionObserver:
    def __init__(self, settings):
        self.settings = settings
        self.latest = None
        self._process = self._connection = None

    def start(self):
        parent, child = multiprocessing.Pipe()
        self._connection = parent
        env = os.environ.copy()
        for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
            env[key] = '1'
        root = str(Path(__file__).resolve().parents[1])
        env['PYTHONPATH'] = os.pathsep.join(filter(None, (root, env.get('PYTHONPATH'))))
        try:
            self._process = subprocess.Popen(
                [sys.executable, '-m', 'handoff.fusion_observer', '--worker-fd', str(child.fileno()),
                 '--settings', json.dumps(self.settings)], pass_fds=(child.fileno(),),
                close_fds=True, start_new_session=True, stdin=subprocess.DEVNULL, env=env)
        except BaseException:
            parent.close()
            self._connection = None
            raise
        finally:
            child.close()
        return self

    def control(self, kind, **fields):
        # Never queue unbounded UI updates while selectROI is waiting for a user.
        if self._connection is not None and select.select([], [self._connection], [], 0)[1]:
            self._connection.send(dict(kind=kind, **fields))

    def poll(self):
        events = []
        try:
            for _ in range(64):
                if not self._connection.poll():
                    break
                packet = self._connection.recv()
                if packet.get('kind') == 'error':
                    raise RuntimeError(packet['error'])
                if packet.get('kind') == 'frame':
                    self.latest = packet
                    events.extend(packet.get('events', []))
        except (EOFError, OSError) as exc:
            if any(e['action'] == 'quit' for e in events):
                return self.latest, events
            raise RuntimeError('global hand observer disconnected') from exc
        if self._process is not None and self._process.poll() is not None and not events:
            raise RuntimeError('global hand observer exited')
        return self.latest, events

    def close(self):
        try:
            if self._process is not None:
                try:
                    self.control('close')
                except (OSError, EOFError):
                    pass
                try:
                    self._process.wait(timeout=2.)
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
            self._connection = self._process = None


def worker(connection, settings):
    import socket
    original_socket = socket.socket
    class CameraSocket(original_socket):
        def __init__(self, family=socket.AF_INET, *args, **kwargs):
            if family == getattr(socket, 'AF_CAN', None):
                raise RuntimeError('CAN is forbidden in the fusion visual observer')
            super().__init__(family, *args, **kwargs)
    socket.socket = CameraSocket
    import cv2
    from cameras.realsense import RealSenseCamera
    from targets.mediapipe_hand_landmarker import MediaPipeHandLandmarker
    camera = detector = video = log = None
    title = 'Global handover | S=shadow park R=ROI B=baseline P=pull T=touch U=withdraw H=hold N=no-contact C=cancel Q=quit'
    try:
        directory = Path(settings['save_dir'])
        directory.mkdir(parents=True, exist_ok=True)
        log = (directory/'observations.jsonl').open('x', buffering=1)
        camera = RealSenseCamera(serial=settings['serial'], width=640, height=480, fps=30,
                                enable_depth=False, warmup=10, timeout=2.)
        detector = MediaPipeHandLandmarker(model_path=settings['hand_model'], num_hands=2,
            min_detection_confidence=.35, min_presence_confidence=.35, min_tracking_confidence=.35)
        camera.start()
        cv2.namedWindow(title, cv2.WINDOW_NORMAL)
        roi, confirmed, revision = settings.get('roi'), False, 0
        held, phase, reason, label = False, 'FOLLOW_RECEIVER', 'waiting for arm to hold', 'UNKNOWN'
        key_notice, key_notice_until = '', 0.
        running, video_index = True, 0
        while running:
            while connection.poll():
                command = connection.recv()
                if command['kind'] == 'close':
                    running = False
                    break
                if command['kind'] == 'hold':
                    held = True
                    confirmed = False
                    revision += 1
                if command['kind'] == 'status':
                    phase, reason = command['phase'], command['reason']
                    held = phase != 'FOLLOW_RECEIVER'
            if not running:
                break
            frame = camera.read()
            observations = detector.detect(frame.bgr)  # Deliberately works without XYZ/depth.
            evidence = visual_evidence(observations, frame.bgr.shape, roi, settings['receiver_side'])
            canvas = frame.bgr.copy()
            if roi:
                x0, y0, x1, y1 = np.rint(np.asarray(roi)*[639, 479, 639, 479]).astype(int)
                cv2.rectangle(canvas, (x0, y0), (x1, y1), (0, 255, 0) if confirmed else (0, 200, 255), 2)
            for o in observations:
                for x, y in o.landmarks_pixel:
                    cv2.circle(canvas, (int(x), int(y)), 2, (255, 100, 0), -1)
                px, py = o.palm_pixel
                cv2.putText(canvas, f'{o.handedness} {o.handedness_score:.2f}',
                    (max(0, int(px)-25), max(14, int(py)-12)),
                    cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 0), 1)
            for i, text in enumerate((phase, reason[:95],
                'S=park; R=ROI; hands OFF object, then B' if settings.get('allow_manual_stop')
                    else 'R=ROI; hands OFF object, then B',
                f"receiver={settings['receiver_side']} candidate={evidence['in_roi']} label={label}")):
                cv2.putText(canvas, text, (8, 24+24*i), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 255), 1)
            if time.monotonic() < key_notice_until:
                cv2.putText(canvas, key_notice, (8, 125), cv2.FONT_HERSHEY_SIMPLEX,
                            .45, (0, 80, 255), 1)
            cv2.imshow(title, canvas)
            key = cv2.waitKey(1) & 0xff
            if ord('A') <= key <= ord('Z'):
                key += ord('a')-ord('A')
            events = []
            now = time.monotonic()
            if key == ord('r') and (not held or phase in
                    ('FOLLOW_RECEIVER', 'FUSION_SETTLING', 'FUSION_OPENING', 'HANDOVER_DONE', 'FUSION_FAULT')):
                if phase == 'FUSION_FAULT':
                    key_notice = 'R blocked: fault latched. Q exits; ROI cannot reset fault.'
                elif phase in ('FUSION_OPENING', 'HANDOVER_DONE'):
                    key_notice = 'R blocked: opening/finished phase; ROI cannot be changed.'
                else:
                    key_notice = 'R not ready: park and wait for FUSION_WAIT_BASELINE.'
                key_notice_until = now+5.
                events.append(dict(action='roi_rejected', phase=phase, reason=key_notice, at=now))
            if key in (27, ord('q')) or cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1:
                events.append(dict(action='quit', at=now))
                running = False
            elif key == ord('s') and settings.get('allow_manual_stop') and phase == 'FOLLOW_RECEIVER':
                events.append(dict(action='stop', at=now))
            elif key == ord('r') and held:
                if phase in ('FUSION_OPENING', 'HANDOVER_DONE', 'FUSION_FAULT'):
                    events.append(dict(action='cancel', at=now))
                elif phase not in ('FOLLOW_RECEIVER', 'FUSION_SETTLING'):
                    chosen = roi if roi is not None and not confirmed else None
                    if chosen is None:
                        x, y, w, h = cv2.selectROI(title, frame.bgr, showCrosshair=True)
                        if w > 0 and h > 0:
                            chosen = validate_roi([x/640, y/480, (x+w)/640, (y+h)/480])
                    if chosen is not None:
                        roi = chosen
                        confirmed, revision = True, revision+1
                        events.append(dict(action='roi', at=time.monotonic()))
            elif key == ord('b') and held and confirmed and phase == 'FUSION_WAIT_BASELINE':
                events.append(dict(action='baseline', at=now))
                label = 'NO_CONTACT'
            elif key == ord('c'):
                events.append(dict(action='cancel', at=now))
                label = 'UNKNOWN'
            labels = {ord('n'): 'NO_CONTACT', ord('t'): 'BUMP_PUSH', ord('p'): 'TAKE_PULL',
                      ord('u'): 'POST_RELEASE', ord('h'): 'HOLD_NO_PULL'}
            if key in labels:
                label = labels[key]
                events.append(dict(action='label', label=label, at=now))
            evidence = visual_evidence(observations, frame.bgr.shape, roi, settings['receiver_side'])
            for event in events:
                event['roi_revision'] = revision
            packet = dict(kind='frame', sequence=frame.sequence,
                serial=frame.serial, hardware_ms=frame.hardware_ms,
                received_monotonic=frame.received_monotonic, detected_monotonic=time.monotonic(),
                raw_hand_observations=record_hand_observations(observations),
                roi=roi, roi_confirmed=confirmed, roi_revision=revision, label=label,
                events=events, **evidence)
            if video is None:
                video = cv2.VideoWriter(str(directory/'global_rgb.avi'),
                    cv2.VideoWriter_fourcc(*'MJPG'), 30., (640, 480))
                if not video.isOpened():
                    raise RuntimeError('cannot open global RGB recording')
            video.write(frame.bgr)
            packet['video_index'] = video_index
            video_index += 1
            # The source sequence and log order map frames to the recorded video;
            # AVI FPS is nominal, timestamps in JSONL remain authoritative.
            log.write(json.dumps(packet, allow_nan=False)+'\n')
            connection.send(packet)
    except (EOFError, BrokenPipeError):
        pass
    except BaseException as exc:
        try:
            connection.send(dict(kind='error', error=f'{type(exc).__name__}: {exc}'))
        except (OSError, EOFError):
            pass
    finally:
        for obj in (detector, camera):
            if obj is not None:
                obj.close()
        if video is not None:
            video.release()
        if log is not None:
            log.close()
        cv2.destroyAllWindows()
        connection.close()


if __name__ == '__main__':
    import argparse
    from multiprocessing.connection import Connection
    p = argparse.ArgumentParser()
    p.add_argument('--worker-fd', type=int, required=True)
    p.add_argument('--settings', required=True)
    args = p.parse_args()
    worker(Connection(args.worker_fd), json.loads(args.settings))
