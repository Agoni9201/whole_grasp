"""Causal visual/light-pull handover policy. No cameras, SDK or actuator writes.

Torque is current-derived feedback, not external force or verified support.
No shipped model permits opening: shadow mode can collect without a model.
"""
from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

import numpy as np

FEATURE_VERSION = 'signed_median_peak_change_slope_v1'


def torque_features(times, delta):
    """28 features, feature-major, J1..J7 within each of four groups."""
    t, x = np.asarray(times, float), np.asarray(delta, float)
    if (t.ndim != 1 or len(t) < 4 or x.shape != (len(t), 7)
            or not np.isfinite(t).all() or not np.isfinite(x).all()
            or np.any(np.diff(t) <= 0)):
        raise ValueError('features require increasing times and finite Nx7 torque')
    middle = (t[0] + t[-1]) / 2
    early, late = x[t <= middle], x[t > middle]
    centered = t - t.mean()
    slope = centered @ x / (centered @ centered)
    return np.r_[np.median(x, axis=0), np.max(np.abs(x), axis=0),
                 np.median(late, axis=0)-np.median(early, axis=0), slope]


class LinearPullModel:
    """Portable binary action model; scores are not support probabilities."""

    def __init__(self, data):
        if data.get('feature_version') != FEATURE_VERSION or data.get('version') != 1:
            raise ValueError('unsupported fusion pull model version')
        self.data = data
        self.mean, self.scale, self.weights = [
            np.asarray(data[key], float) for key in ('mean', 'scale', 'weights')]
        self.intercept = float(data['intercept'])
        if (any(x.shape != (28,) or not np.isfinite(x).all()
                for x in (self.mean, self.scale, self.weights))
                or np.any(self.scale <= 0) or not np.isfinite(self.intercept)):
            raise ValueError('invalid fusion model coefficients')

    @classmethod
    def load(cls, path):
        return cls(json.loads(Path(path).expanduser().read_text()))

    def score(self, features):
        value = np.asarray(features, float)
        if value.shape != (28,) or not np.isfinite(value).all():
            raise ValueError('invalid fusion model input')
        z = float(((value-self.mean)/self.scale) @ self.weights + self.intercept)
        return float(1. / (1.+np.exp(-np.clip(z, -60., 60.))))


@dataclass(frozen=True)
class FusionConfig:
    ready_distance: float = .15
    ready_seconds: float = .15
    ready_max_step_m: float = .04
    ready_image_radius_fraction: float = .2
    settle_seconds: float = .5
    still_q_rad: float = .003
    still_dq_rad_s: float = .02
    hold_drift_rad: float = .009
    baseline_seconds: float = 3.
    baseline_max_std_nm: float = .1
    baseline_max_change_nm: float = .1
    window_seconds: float = 1.
    max_gap_seconds: float = .25
    min_window_samples: int = 6
    visual_max_age: float = .25
    visual_stable_seconds: float = .2
    confirm_seconds: float = .3
    candidate_max_seconds: float = 1.
    pull_on: float = .8
    pull_off: float = .3
    open_seconds: float = 5.
    maximum_width: float = .085

    def __post_init__(self):
        if not all(np.isfinite(v) and v > 0 for v in asdict(self).values()):
            raise ValueError('fusion timings/limits must be finite and positive')
        if (not 0 < self.pull_off < self.pull_on < 1 or self.maximum_width > .085
                or self.min_window_samples < 4
                or self.ready_image_radius_fraction > .5
                or self.candidate_max_seconds < self.confirm_seconds
                or self.window_seconds <= self.max_gap_seconds):
            raise ValueError('invalid fusion score/window/width limits')


class FusionHandover:
    """FOLLOW -> settle -> operator-confirmed baseline -> new pull -> opening.

    A fresh low action score AFTER current visual eligibility arms an edge;
    an old high score cannot be inherited through withdrawal/vision loss.
    """

    def __init__(self, config=None, *, mode='shadow', model=None):
        if mode not in ('shadow', 'release') or (mode == 'release' and model is None):
            raise ValueError('fusion release requires an explicitly supplied pull model')
        self.config, self.mode, self.model = config or FusionConfig(), mode, model
        self.phase, self.reason = 'FOLLOW_RECEIVER', 'waiting for fresh 3D delivery target'
        self._near = deque()
        self._motion = deque()
        self._baseline_samples = deque()
        self._window = deque()
        self._last_stamps = self._last_position_stamps = None
        self._last_sample_at = self._last_tick = None
        self.hold_q = self.baseline = self.baseline_std = None
        self._roi_revision = None
        self._clear_candidate()
        self.features = self.pull_score = None
        self._candidate_at = None
        self.parking_source = None
        self.delivery_evidence = dict(reason='waiting for fresh 3D preparation observations', frames=0, elapsed=0.)
        self.baseline_readiness = None

    @property
    def terminal(self):
        return self.phase in ('HANDOVER_DONE', 'FUSION_FAULT')

    def _clear_candidate(self):
        if self.phase == 'FUSION_CONFIRM':
            self.phase = 'FUSION_WAIT_TAKE'
        self._visual_since = self._low_since = self._candidate_since = None
        self._armed = False
        self._confirm_count = 0
        self._visual_sequences = set()

    def fault(self, reason):
        self.phase, self.reason = 'FUSION_FAULT', str(reason)
        self._clear_candidate()

    def observe_delivery(self, center, distance, centered, now):
        """Request an early hold, not proof of human readiness/contact.

        A few fresh, geometrically plausible observations suffice before
        occlusion. Human stillness and exact image centering are not required.
        """
        if self.phase != 'FOLLOW_RECEIVER':
            return False
        c = self.config
        point = np.asarray(center, float)
        good = (point.shape == (3,) and np.isfinite(point).all()
                and np.isfinite([distance, now]).all() and 0 < distance <= c.ready_distance
                and centered)
        if not good or (self._near and not 0 < now-self._near[-1][0] <= c.max_gap_seconds):
            self._near.clear()
        if not good:
            self.delivery_evidence = dict(reason='need fresh in-range 3D and coarse image region',
                frames=0, elapsed=0., distance_m=float(distance), coarse_centered=bool(centered))
            self.reason = self.delivery_evidence['reason']
            return False
        if self._near and np.linalg.norm(point-self._near[-1][1]) > c.ready_max_step_m:
            self._near.clear()  # Reject jumps, without requiring a stationary person.
        self._near.append((now, point.copy()))
        while len(self._near) > 1 and now-self._near[1][0] >= c.ready_seconds:
            self._near.popleft()
        elapsed = now-self._near[0][0]
        ready = len(self._near) >= 3 and elapsed+1e-9 >= c.ready_seconds
        self.delivery_evidence = dict(reason='fresh preparation observations allow early parking' if ready
            else 'collecting short 3D preparation segment; shadow S can park manually',
            frames=len(self._near), elapsed=elapsed, distance_m=float(distance), coarse_centered=True)
        self.reason = self.delivery_evidence['reason']
        return ready

    def delivery_lost(self):
        self._near.clear()
        self.delivery_evidence = dict(reason='wrist target lost; hold motion, no automatic arrival', frames=0, elapsed=0.)
        if self.phase == 'FOLLOW_RECEIVER':
            self.reason = self.delivery_evidence['reason']

    def begin_hold(self, q, *, source='visual_pre_occlusion'):
        if (self.phase != 'FOLLOW_RECEIVER' or source not in ('visual_pre_occlusion', 'manual_shadow')
                or (source == 'manual_shadow' and self.mode != 'shadow')):
            raise ValueError('invalid fusion parking transition/source')
        value = np.asarray(q, float)
        if value.shape != (7,) or not np.isfinite(value).all():
            raise ValueError('invalid hold position')
        self.hold_q = value.copy()
        self.parking_source = source
        self.phase, self.reason = 'FUSION_SETTLING', 'arm holding; receiver must not touch yet'
        self._motion.clear()

    def reset_baseline(self, reason='re-confirm ROI and untouched object before B'):
        if self.phase in ('FUSION_OPENING', 'HANDOVER_DONE', 'FUSION_FAULT'):
            self.fault('reset during/after opening is not allowed')
            return
        self.phase, self.reason = 'FUSION_WAIT_BASELINE', reason
        self.baseline = self.baseline_std = None
        self.features = self.pull_score = None
        self.baseline_readiness = None
        self._baseline_samples.clear()
        self._window.clear()
        self._clear_candidate()

    def tick(self, sample, visual, now, *, confirm_untouched=False, cancel=False):
        """Process validated parser snapshots; duplicates never advance evidence."""
        if cancel and self.phase == 'WOULD_RELEASE':
            self.reset_baseline('shadow trial reset; confirm untouched again')
            return
        if self.terminal or self.phase == 'FOLLOW_RECEIVER':
            return
        c = self.config
        if not np.isfinite(now) or (self._last_tick is not None and now <= self._last_tick):
            self.fault('fusion monotonic clock did not advance')
            return
        gap = self._last_tick is not None and now-self._last_tick > c.max_gap_seconds
        self._last_tick = now
        if gap:
            if self.phase == 'FUSION_OPENING':
                self.fault('fusion loop stalled during opening')
                return
            self._clear_candidate()
            self._motion.clear()
            self._window.clear()
            if self.phase == 'FUSION_BASELINE':
                self.reset_baseline('baseline interrupted; confirm untouched again')
        if cancel:
            self.reset_baseline('operator cancelled; confirm untouched again')
            return
        if sample is None:
            if self.phase == 'FUSION_OPENING':
                self.fault('feedback unavailable during opening')
            else:
                self.reset_baseline('feedback unavailable; baseline invalidated')
            return
        try:
            q, dq, tau, stamps, position_stamps = [np.asarray(sample[k], float) for k in
                ('q', 'dq_raw', 'tau', 'motor_timestamps', 'position_timestamps')]
            if any(x.shape != (7,) or not np.isfinite(x).all()
                   for x in (q, dq, tau, stamps, position_stamps)):
                raise ValueError('invalid vectors')
        except (KeyError, ValueError, TypeError):
            self.fault('invalid fusion feedback snapshot')
            return
        if (self._last_stamps is not None and (np.any(stamps < self._last_stamps)
                or np.any(position_stamps < self._last_position_stamps))):
            self.fault('fusion feedback timestamps went backwards')
            return
        fresh = self._last_stamps is None or np.all(stamps > self._last_stamps)
        if fresh:
            if self._last_sample_at is not None and now-self._last_sample_at > c.max_gap_seconds:
                self._window.clear()
                self._motion.clear()
                self._clear_candidate()
                if self.phase == 'FUSION_BASELINE':
                    self.reset_baseline('baseline feedback gap; confirm again')
            self._last_stamps, self._last_position_stamps = stamps.copy(), position_stamps.copy()
            self._last_sample_at = now
            self._motion.append((now, q.copy()))
            while len(self._motion) > 1 and now-self._motion[1][0] >= c.settle_seconds:
                self._motion.popleft()
        if self._last_sample_at is None or now-self._last_sample_at > c.max_gap_seconds:
            if self.phase == 'FUSION_OPENING':
                self.fault('motor stream stopped during opening')
            else:
                self._clear_candidate()
                self.reason = 'waiting for new motor packets'
            return
        still = (len(self._motion) >= 3
                 and now-self._motion[0][0] >= c.settle_seconds
                 and np.max(np.ptp([x for _, x in self._motion], axis=0)) <= c.still_q_rad
                 and np.max(np.abs(dq)) <= c.still_dq_rad_s)
        if np.max(np.abs(q-self.hold_q)) > c.hold_drift_rad:
            self.fault('arm moved outside frozen handover pose')
            return
        if self.phase == 'FUSION_SETTLING':
            if still:
                self.phase, self.reason = 'FUSION_WAIT_BASELINE', 'select ROI (R), hands away, then B'
            return
        v = visual or {}
        age = now-float(v.get('received_monotonic', -1e20))
        camera_ok = (0 <= age <= c.visual_max_age and bool(v.get('roi_confirmed'))
                     and not v.get('error'))
        revision = v.get('roi_revision')
        if self._roi_revision is not None and revision != self._roi_revision:
            if self.phase == 'FUSION_OPENING':
                self.fault('ROI changed during opening')
                return
            self.reset_baseline('ROI changed; hands away and re-confirm baseline')
        self._roi_revision = revision
        blockers = []
        if not still:
            blockers.append('arm_not_settled')
        if not v.get('roi_confirmed'):
            blockers.append('roi_not_confirmed')
        if not 0 <= age <= c.visual_max_age:
            blockers.append('camera_frame_not_fresh')
        if v.get('error'):
            blockers.append('camera_error')
        if v.get('roi_confirmed') and not v.get('clear'):
            blockers.append('hand_detection_uncertain' if v.get('rejected') else 'hand_in_roi')
        self.baseline_readiness = dict(ready=not blockers, blockers=blockers,
            arm_settled=bool(still), roi_confirmed=bool(v.get('roi_confirmed')),
            image_age_seconds=age, roi_clear=bool(v.get('clear')))
        if self.phase == 'FUSION_WAIT_BASELINE':
            self.reason = ('ready for untouched baseline; hands OFF object, press B' if not blockers
                           else 'baseline blocked: '+', '.join(blockers))
        if confirm_untouched and self.phase == 'FUSION_WAIT_BASELINE':
            if still and camera_ok and v.get('clear'):
                self.phase, self.reason = 'FUSION_BASELINE', 'recording untouched held-object baseline'
                self._baseline_samples.clear()
            else:
                self.reason = 'B rejected: '+', '.join(blockers)
        if self.phase == 'FUSION_BASELINE':
            if not still or not camera_ok or not v.get('clear'):
                self.reset_baseline('baseline interrupted by motion/hand/vision; confirm again')
                return
            if fresh:
                self._baseline_samples.append((now, tau.copy()))
            if (fresh and len(self._baseline_samples) >= c.min_window_samples
                    and self._baseline_samples[-1][0]-self._baseline_samples[0][0] >= c.baseline_seconds):
                values = np.array([x for _, x in self._baseline_samples])
                self.baseline_std = np.std(values, axis=0)
                half = len(values)//2
                change = np.median(values[half:], axis=0)-np.median(values[:half], axis=0)
                if (np.max(self.baseline_std) > c.baseline_max_std_nm
                        or np.max(np.abs(change)) > c.baseline_max_change_nm):
                    self.reset_baseline('baseline torque unstable; do not touch, inspect and retry B')
                    return
                self.baseline = np.median(values, axis=0)
                self.phase, self.reason = 'FUSION_WAIT_TAKE', 'baseline ready; grasp then gently pull'
                self._window.clear()
                self._clear_candidate()
            return
        if self.baseline is None:
            return
        visual_ok = camera_ok and bool(v.get('in_roi')) and bool(v.get('visible'))
        if self.phase == 'WOULD_RELEASE':
            if not visual_ok:
                self._clear_candidate()
                self._window.clear()
                self.features = self.pull_score = None
                self.phase, self.reason = 'FUSION_WAIT_TAKE', 'shadow candidate revoked; need a new visible pull'
            return
        if self.phase == 'FUSION_OPENING':
            if not visual_ok:
                self.fault('receiver lost/withdrawn during opening; no further opening targets')
            return  # Do not apply pre-opening torque classifier after load redistribution.
        if fresh:
            self._window.append((now, tau-self.baseline))
            while len(self._window) > 1 and now-self._window[0][0] > c.window_seconds+1e-9:
                self._window.popleft()
        self.features = self.pull_score = None
        ready = (len(self._window) >= c.min_window_samples
                 and self._window[-1][0]-self._window[0][0] >= .8*c.window_seconds)
        if ready:
            self.features = torque_features(*zip(*self._window))
            if self.model is not None:
                self.pull_score = self.model.score(self.features)
        if not visual_ok:
            self._clear_candidate()
            self.phase, self.reason = 'FUSION_WAIT_TAKE', 'no current visual receiver candidate; holding'
            return
        if self._visual_since is None:
            self._visual_since = now
        if self.model is None:
            self.reason = 'recording only: no pull model supplied, opening disabled'
            return
        if self.pull_score is None or not fresh:
            return
        if not self._armed:
            if self.pull_score <= c.pull_off:
                if self._low_since is None:
                    self._low_since = now
                if (now-self._low_since >= c.visual_stable_seconds
                        and now-self._visual_since >= c.visual_stable_seconds):
                    self._armed = True
                    self.reason = 'receiver visible; waiting for a NEW light pull'
            else:
                self._low_since = None
                self.reason = 'wait for quiet action score before new pull; old offset cannot arm'
            return
        if self.pull_score < c.pull_on:
            if self._candidate_since is not None:
                self._clear_candidate()
            self.phase, self.reason = 'FUSION_WAIT_TAKE', 'waiting for new confirmed light pull'
            return
        if self._candidate_since is None:
            self._candidate_since = now
            self._candidate_at = now
        self.phase, self.reason = 'FUSION_CONFIRM', 'current visual + new pull; confirming'
        self._confirm_count += 1
        self._visual_sequences.add(v.get('sequence'))
        elapsed = now-self._candidate_since
        if elapsed > c.candidate_max_seconds:
            self._clear_candidate()
            self.phase, self.reason = 'FUSION_WAIT_TAKE', 'candidate expired; need new pull'
        elif (elapsed >= c.confirm_seconds and self._confirm_count >= 3
              and len(self._visual_sequences) >= 2):
            self.phase = 'WOULD_RELEASE' if self.mode == 'shadow' else 'FUSION_OPENING'
            self.reason = ('shadow candidate only; no gripper writes' if self.mode == 'shadow'
                           else 'confirmed; keep arm fixed, bounded opening permitted')

    def snapshot(self):
        return dict(phase=self.phase, reason=self.reason, mode=self.mode,
                    hold_q_rad=None if self.hold_q is None else self.hold_q.tolist(),
                    hold_drift_limit_rad=self.config.hold_drift_rad,
                    parking_source=self.parking_source, delivery_evidence=self.delivery_evidence.copy(),
                    baseline_readiness=self.baseline_readiness,
                    baseline_nm=None if self.baseline is None else self.baseline.tolist(),
                    baseline_std_nm=None if self.baseline_std is None else self.baseline_std.tolist(),
                    features=None if self.features is None else self.features.tolist(),
                    pull_score=self.pull_score, armed=self._armed,
                    candidate_at=self._candidate_at, model_configured=self.model is not None)


def model_metadata(path):
    if not path:
        return None
    p = Path(path).expanduser()
    return dict(path=str(p.resolve()), sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
                model=json.loads(p.read_text()))
