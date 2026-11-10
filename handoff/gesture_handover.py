"""Two-hand cup handover policy, independent of camera, SDK and arm control.

Distance is TCP-to-palm, not a measured cup surface distance. Neither gestures
nor joint-current torque feedback establish that a person supports the cup.
"""
from __future__ import annotations

import math
import copy
import time
import numpy as np
from handoff.pinch_gripper import (
    AutoPinchHand, AutoPinchMapping, GripperResponseWatch, WidthRamp,
)


def read_opening(status):
    """Snapshot fresh physical width feedback; no enable/reset/config writes."""
    wrapped = copy.deepcopy(status)
    stamp = float(getattr(wrapped, 'timestamp', float('nan')))
    if not math.isfinite(stamp) or not 0. <= time.time()-stamp <= .5:
        raise ValueError('gripper feedback missing/stale')
    msg = wrapped.msg
    width, force = float(msg.value), float(msg.force)
    if msg.mode != 'width' or not math.isfinite(width) or not 0. <= width <= .085 or not math.isfinite(force):
        raise ValueError('invalid gripper opening/force feedback')
    foc = msg.foc_status
    if not foc.driver_enable_status or any(getattr(foc, name) for name in (
            'voltage_too_low', 'motor_overheating', 'driver_overcurrent',
            'driver_overheating', 'sensor_status', 'driver_error_status')):
        raise ValueError('gripper disabled or reports a driver fault')
    return width, force, stamp


def wait_initial_opening(gripper, timeout=.5):
    """Allow a newly attached SDK parser to receive its first status packet."""
    deadline = time.monotonic()+timeout
    while True:
        try:
            return read_opening(gripper.status())
        except (ValueError, AttributeError, TypeError) as exc:
            if time.monotonic() >= deadline:
                raise ValueError(f'initial cup grip feedback unavailable: {exc}') from exc
            time.sleep(.01)


class GestureHandover:
    def __init__(self, grip_width, *, distance=.05, tolerance=.01,
                 stable_seconds=3., still_tolerance=.01,
                 ready_distance=.15, extra_close=.001, maximum=.085,
                 open_seconds=5., close_seconds=2.):
        values = (grip_width, distance, tolerance, stable_seconds, still_tolerance,
                  extra_close, maximum, open_seconds, close_seconds, ready_distance)
        if (not all(math.isfinite(v) for v in values) or not 0. < grip_width < maximum <= .085
                or not 0. <= extra_close <= .001 or min(values[1:5]) <= 0.
                or min(open_seconds, close_seconds) <= 0. or ready_distance < distance):
            raise ValueError('invalid gesture handover geometry or timing')
        self.grip_width = grip_width  # Fixed MEASURED opening while holding this cup.
        self.minimum, self.maximum = max(0., grip_width-extra_close), maximum
        self.distance, self.tolerance = distance, tolerance
        self.ready_distance = ready_distance
        self.stable_seconds, self.still_tolerance = stable_seconds, still_tolerance
        self.phase, self.reason = 'FOLLOW_RECEIVER', 'follow the locked receiver hand'
        self.near_since = self.near_updated = self.ready_at = None
        self.still_lower = self.still_upper = None
        self.still_elapsed = self.motion_span = 0.
        self.mapping = AutoPinchMapping()
        self.control_hand = AutoPinchHand()
        self.ramp = WidthRamp(self.minimum, maximum, open_seconds, close_seconds)
        self.watch = GripperResponseWatch()
        self.closed_since = self.last_control_at = self.last_sent_at = self.last_sent = None
        self.filtered = self.fraction = None
        self.last_feedback_stamp = None
        self.open_count = 0
        self.full_open_requested = False
        self.release_started = False
        self.receiver_anchor = None
        self.receiver_last_position = None
        self.withdraw_since = self.withdraw_updated = None

    @property
    def terminal(self):
        return self.phase in ('HANDOVER_DONE', 'HANDOVER_CANCELLED', 'GESTURE_FAULT')

    def invalidate_near(self, reason='Receiver acquisition interrupted; restart stability timer'):
        if self.release_started or self.terminal:
            return  # Once opening is offered, never return to arm tracking.
        self.phase = 'FOLLOW_RECEIVER'
        self.ready_at = self.receiver_anchor = None
        self.near_since = self.near_updated = None
        self.still_lower = self.still_upper = None
        self.still_elapsed = self.motion_span = 0.
        self.mapping.pause()  # Preserve learned endpoints, discard pending gestures.
        self.control_hand.lost()
        self.closed_since = self.last_control_at = self.fraction = self.filtered = None
        self.reason = reason

    def observe_receiver(self, presence, distance, accepted, now, *, position_base=None):
        """Latch readiness from fresh centered/ranged, stationary BASE positions.

        Image centering alone cannot establish stillness: the arm could be
        following a moving hand. The bounding-box diagonal bounds the distance
        between every pair of raw palm positions in this uninterrupted window.
        """
        if self.terminal:
            return
        point = np.asarray(position_base, dtype=float)
        valid = (presence == 'visible' and math.isfinite(now)
                 and point.shape == (3,) and np.isfinite(point).all())
        if self.release_started:
            # Fresh, sustained measured displacement may cancel release. Loss
            # itself is not evidence that the receiver has withdrawn.
            moved = (valid and self.receiver_anchor is not None
                     and np.linalg.norm(point-self.receiver_anchor) > .05)
            if not moved:
                self.withdraw_since = self.withdraw_updated = None
            else:
                if self.withdraw_updated is None or not 0 < now-self.withdraw_updated <= .25:
                    self.withdraw_since = now
                self.withdraw_updated = now
                if now-self.withdraw_since >= .5:
                    self.phase = 'HANDOVER_CANCELLED'
                    self.reason = 'Receiver moved away during release; hold arm and current opening; no re-close'
                    self.watch.pause()
            return
        if self.ready_at is not None:
            if not valid:
                return  # Occlusion/depth failure alone cannot revoke readiness.
            self.receiver_last_position = point.copy()
            if (np.linalg.norm(point-self.receiver_anchor) > self.still_tolerance
                    or not accepted or distance is None or not math.isfinite(distance)
                    or distance > self.ready_distance):
                self.invalidate_near('Receiver moved before release; follow again; discard opening request and re-confirm after settling')
            return
        if presence != 'visible' or not math.isfinite(now):
            self.invalidate_near('Waiting for visible receiver with valid depth; stability timer reset')
            return
        if point.shape != (3,) or not np.isfinite(point).all():
            self.invalidate_near('Waiting for valid receiver BASE position; stability timer reset')
            return
        if distance is None or not math.isfinite(distance) or not 0. < distance <= self.ready_distance:
            self.invalidate_near('Waiting for receiver at handover distance; stability timer reset')
            return
        if not accepted:
            self.invalidate_near('Waiting for centered, safety-accepted receiver; stability timer reset')
            return
        restart = (self.near_updated is None or not 0. <= now-self.near_updated <= .25)
        lower = point if restart else np.minimum(self.still_lower, point)
        upper = point if restart else np.maximum(self.still_upper, point)
        self.motion_span = float(np.linalg.norm(upper-lower))
        moved = self.motion_span > self.still_tolerance+1e-12
        if restart or moved:
            self.near_since = now
            self.still_lower, self.still_upper = point.copy(), point.copy()
        else:
            self.still_lower, self.still_upper = lower, upper
        self.near_updated = now
        self.receiver_last_position = point.copy()
        self.still_elapsed = max(0., now-self.near_since)
        if self.still_elapsed >= self.stable_seconds:
            self.ready_at = now
            self.receiver_anchor = point.copy()
            self.phase = 'GESTURE_ARM' if self.mapping.phase == 'ready' else 'GESTURE_LEARN'
            self.closed_since = self.last_control_at = self.fraction = self.filtered = None
            self.control_hand.lost()
            self.reason = ('Receiver stable again; pinch CLOSED to re-confirm' if self.mapping.phase == 'ready'
                           else 'Receiver stable; arm holds; ' + self.mapping.instruction)
        elif moved:
            self.reason = 'Receiver moved in BASE coordinates; stability timer restarted'
        else:
            self.reason = f'Receiver still: {self.still_elapsed:.1f}/{self.stable_seconds:g}s'

    def fault(self, reason):
        self.phase, self.reason = 'GESTURE_FAULT', str(reason)
        self.watch.pause()

    def command_sent(self, width, now):
        if not self.minimum <= width <= self.maximum:
            raise ValueError('handover opening outside fixed cup limits')
        self.last_sent, self.last_sent_at = float(width), float(now)

    def tick(self, pinch, *, fresh_frame, measured_width, feedback_stamp, now, arm_still):
        """Return a bounded target, or None. Caller owns every actual SDK write."""
        if self.phase == 'FOLLOW_RECEIVER' or self.terminal:
            return None
        if (not math.isfinite(measured_width) or not math.isfinite(feedback_stamp)
                or not 0. <= measured_width <= .085):
            self.fault('invalid gripper feedback; no new gripper target')
            return None
        new_feedback = feedback_stamp != self.last_feedback_stamp
        self.last_feedback_stamp = feedback_stamp
        # Full opening ends gesture control. Dropout never starts re-closing,
        # including before full opening has been confirmed.
        if self.release_started:
            opened = (self.full_open_requested and self.last_sent is not None
                      and self.last_sent >= self.maximum-1e-5
                      and measured_width >= self.maximum-.001)
            if not opened:
                self.open_count = 0
            elif new_feedback:
                self.open_count += 1
            if self.open_count >= 3:
                self.phase, self.reason = 'HANDOVER_DONE', 'fully open confirmed; arm holds; no automatic re-close'
                return None
        expired = self.last_control_at is not None and now-self.last_control_at > .25
        if expired:
            # Even when the next tick already has a fresh frame, a camera gap
            # requires three new consistent frames before resuming control.
            self.control_hand.lost()
            self.mapping.pause()
            self.closed_since = self.last_control_at = self.fraction = None
        ready = False
        if fresh_frame:
            if pinch is None:
                self.control_hand.lost()
                self.mapping.pause()
                self.closed_since = None
                self.last_control_at = None
                self.fraction = None
            else:
                self.last_control_at = now
                ready = self.control_hand.observe(pinch)
                if ready:
                    self.fraction = self.mapping.observe(pinch, now)
        elif self.last_control_at is not None and now-self.last_control_at <= .25:
            ready = self.control_hand.good_frames >= 3
        lost = self.last_control_at is None or now-self.last_control_at > .25
        if self.phase == 'GESTURE_LEARN':
            self.reason = self.mapping.instruction
            if self.mapping.phase == 'ready':
                self.phase, self.reason = 'GESTURE_ARM', 'Pinch CLOSED again to begin gripper control'
                self.closed_since = None
            return None  # Calibration opening NEVER releases the cup.
        if self.phase == 'GESTURE_ARM':
            closed = ready and not lost and self.fraction is not None and self.fraction <= .03
            if not closed or not arm_still:
                self.closed_since = None
                self.reason = 'Pinch CLOSED again and hold; arm must be stationary'
                return None
            if self.closed_since is None:
                self.closed_since = now
            if now-self.closed_since < .4:
                return None
            try:
                self.ramp.reset(measured_width, now)
            except ValueError as exc:
                self.fault(exc)
                return None
            self.phase, self.reason = 'GESTURE_READY', 'Confirmed; keep grip unchanged until a new OPEN gesture; receiver motion has priority'
            self.filtered = self.fraction
            self.watch.pause()
            return None  # Closed confirmation must not alter the loaded grip.
        if self.phase == 'GESTURE_READY':
            if lost or (fresh_frame and not ready):
                self.phase, self.reason = 'GESTURE_ARM', 'Control hand lost before release; pinch CLOSED again'
                self.closed_since = None
                return None
            if not arm_still or not fresh_frame or self.fraction is None or self.fraction <= .05:
                return None
            # A first command below measured width could loosen an existing
            # force-maintaining grasp target. Send NOTHING until explicit open.
            desired = self.minimum+self.fraction*(self.maximum-self.minimum)
            if desired <= measured_width+1e-5:
                return None
            self.ramp.reset(measured_width, now)
            self.filtered = self.fraction
            self.phase, self.reason = 'GESTURE_CONTROL', 'Release started; arm holds permanently for this handover'
            self.release_started = True  # Latch before any external SDK write.
        if self.phase == 'GESTURE_CONTROL' and not arm_still:
            self.fault('arm moved during gesture control; no new gripper target')
            return None
        if self.phase == 'GESTURE_CONTROL' and (expired or lost or (fresh_frame and not ready)):
            self.phase, self.reason = 'GESTURE_PAUSED', 'Control hand lost/ambiguous; no new gripper target'
            self.open_count = 0
            self.full_open_requested = False
            self.watch.pause()
            return None  # Keep the current opening; never command a return.
        if self.phase == 'GESTURE_PAUSED':
            if not ready or lost or self.fraction is None or not arm_still:
                return None
            try:
                self.ramp.reset(measured_width, now)
            except ValueError as exc:
                self.fault(exc)
                return None
            self.phase, self.reason = 'GESTURE_CONTROL', 'Same control hand recovered; resume from measured opening'
            self.filtered = self.fraction
            self.last_sent = self.last_sent_at = None
            self.watch.pause()
        if self.fraction is None:
            return None
        self.filtered = .35*self.fraction + .65*self.filtered
        fraction = self.filtered
        self.full_open_requested = self.fraction >= .98
        try:
            if self.last_sent is not None:
                self.watch.observe(self.last_sent, measured_width, now)
            command = self.ramp.advance(fraction, now, measured_width)
        except (ValueError, RuntimeError) as exc:
            self.fault(exc)
            return None
        if self.last_sent is None or abs(command-self.last_sent) >= 1e-5 or now-self.last_sent_at >= .1:
            return command
        return None

    def snapshot(self):
        return dict(phase=self.phase, reason=self.reason, fraction=self.fraction,
                    handover_ready=self.ready_at is not None,
                    release_started=self.release_started,
                    ready_distance_mm=self.ready_distance*1000.,
                    receiver_still_seconds=self.still_elapsed,
                    receiver_still_required_seconds=self.stable_seconds,
                    receiver_motion_span_mm=self.motion_span*1000.,
                    receiver_still_tolerance_mm=self.still_tolerance*1000.,
                    control_hand_loss_policy='pause_keep_current_opening',
                    grip_width_mm=self.grip_width*1000., minimum_width_mm=self.minimum*1000.,
                    last_sent_mm=None if self.last_sent is None else self.last_sent*1000.,
                    mapping_phase=self.mapping.phase,
                    mapping_closed=None if self.mapping.mapping is None else self.mapping.mapping.closed,
                    mapping_open=None if self.mapping.mapping is None else self.mapping.mapping.opened)
