"""Short depth-only dropout budget. No camera, robot, or actuator ownership."""
from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class HandMemoryLimits:
    maximum_age_s: float = .3
    maximum_travel_m: float = .003
    maximum_speed_m_s: float = .01
    stable_seconds: float = .3
    stable_span_m: float = .01
    maximum_frame_gap_s: float = .25
    maximum_ray_error: float = .03


class HandPositionMemory:
    def __init__(self):
        self.limits = HandMemoryLimits()
        self.history = deque()
        self.target = self.stamp = None
        self.active = self.blocked = False
        self.command_travel = self.measured_travel = 0.
        self.last_tcp = self.last_reference = None
        self.reason = 'waiting_for_stable_3d_measurements'

    def observe(self, target, stamp):
        point = np.asarray(target[0], dtype=float).reshape(3)
        if not np.isfinite(point).all() or not np.isfinite(stamp):
            self.invalidate('invalid_memory_measurement')
            return
        if self.stamp is not None and not 0 < stamp-self.stamp <= self.limits.maximum_frame_gap_s:
            self.history.clear()
        self.target = tuple(np.asarray(v, dtype=float).copy() for v in target)
        self.stamp = float(stamp)
        self.history.append((stamp, point.copy()))
        while self.history and stamp-self.history[0][0] > self.limits.stable_seconds+.1:
            self.history.popleft()
        self.active = self.blocked = False
        self.command_travel = self.measured_travel = 0.
        self.last_tcp = self.last_reference = None
        self.reason = 'fresh_3d_measurement'

    def invalidate(self, reason):
        self.blocked = True  # Only a NEW measured 3D observation can re-enable.
        self.reason = reason

    def available(self, now):
        if self.target is None or self.blocked:
            return False
        if not 0 <= now-self.stamp <= self.limits.maximum_age_s:
            self.invalidate('memory_expired')
            return False
        points = [p for _, p in self.history]
        if (len(points) < 2 or self.stamp-self.history[0][0] < self.limits.stable_seconds-1e-9
                or np.linalg.norm(np.ptp(points, axis=0)) > self.limits.stable_span_m):
            self.invalidate('receiver_was_not_stable_before_depth_loss')
            return False
        return True

    def consistent_image(self, camera_point, palm_pixel, K):
        p = np.asarray(camera_point, dtype=float).reshape(3)
        uv = np.asarray(palm_pixel, dtype=float).reshape(2)
        K = np.asarray(K, dtype=float).reshape(3, 3)
        if (not all(np.isfinite(v).all() for v in (p, uv, K)) or p[2] <= .03
                or min(K[0, 0], K[1, 1]) <= 0):
            self.invalidate('invalid_memory_projection')
            return False
        observed = (uv-K[:2, 2])/[K[0, 0], K[1, 1]]
        if np.linalg.norm(observed-p[:2]/p[2]) > self.limits.maximum_ray_error:
            self.invalidate('receiver_image_disagrees_with_memory')
            return False
        return True

    def start(self, tcp, reference):
        self.active = True
        self.last_tcp = np.asarray(tcp, dtype=float).copy()
        self.last_reference = np.asarray(reference, dtype=float).copy()

    def remaining(self, tcp):
        tcp = np.asarray(tcp, dtype=float)
        if self.last_tcp is not None:
            self.measured_travel += float(np.linalg.norm(tcp-self.last_tcp))
        self.last_tcp = tcp.copy()
        left = self.limits.maximum_travel_m-max(self.command_travel, self.measured_travel)
        if left <= 1e-8:
            self.invalidate('memory_travel_budget_exhausted')
        return max(0., left)

    def reference_sent(self, reference):
        reference = np.asarray(reference, dtype=float)
        self.command_travel += float(np.linalg.norm(reference-self.last_reference))
        self.last_reference = reference.copy()

    def snapshot(self):
        return dict(active=self.active and not self.blocked, reason=self.reason,
                    command_travel_mm=self.command_travel*1000,
                    measured_travel_mm=self.measured_travel*1000,
                    maximum_age_s=self.limits.maximum_age_s,
                    maximum_travel_mm=self.limits.maximum_travel_m*1000)
