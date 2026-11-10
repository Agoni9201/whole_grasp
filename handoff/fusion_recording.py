"""Bounded asynchronous handover recording; disk failures revoke permission."""
from __future__ import annotations

import json
from pathlib import Path
import queue
import threading

import numpy as np


def json_value(value):
    if isinstance(value, np.ndarray):
        return json_value(value.tolist())
    if isinstance(value, np.generic):
        return json_value(value.item())
    if isinstance(value, dict):
        return {key: json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_value(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


class FusionRecorder:
    def __init__(self, directory, metadata):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        (self.directory/'metadata.json').write_text(json.dumps(json_value(metadata), indent=2))
        self.queue = queue.Queue(maxsize=128)
        self.error = None
        self._closed = False
        self.thread = threading.Thread(target=self._run, name='handover-recorder', daemon=True)
        self.thread.start()

    def write(self, record, wrist=None):
        if self.error or self._closed:
            raise RuntimeError(f'handover recorder unavailable: {self.error or "closed"}')
        try:
            self.queue.put_nowait((json_value(record), wrist))
        except queue.Full as exc:
            raise RuntimeError('handover recording queue full; permission cancelled') from exc

    def _run(self):
        try:
            import cv2
            from servo_delivery import image_msg_to_bgr
            with (self.directory/'samples.jsonl').open('x', buffering=1) as stream:
                while True:
                    item = self.queue.get()
                    if item is None:
                        return
                    record, wrist = item
                    if wrist is not None:
                        msg, sequence = wrist
                        path = self.directory/'wrist_rgb'/f'{sequence}.jpg'
                        path.parent.mkdir(exist_ok=True)
                        if not cv2.imwrite(str(path), image_msg_to_bgr(msg)):
                            raise RuntimeError('cannot record wrist RGB')
                        record['wrist_rgb'] = str(path.relative_to(self.directory))
                    stream.write(json.dumps(record, allow_nan=False)+'\n')
        except BaseException as exc:
            self.error = f'{type(exc).__name__}: {exc}'

    def close(self):
        if self._closed:
            return
        self._closed = True
        try:
            self.queue.put(None, timeout=2.)
        except queue.Full:
            self.error = self.error or 'recorder did not drain on close'
        self.thread.join(timeout=3.)
        if self.thread.is_alive():
            self.error = self.error or 'recorder close timeout'
