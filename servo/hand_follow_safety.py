"""Bounded hand approach with measured progress, independent of final range.

No robot I/O: progress comes from actual joint samples. The URDF reach sphere
is diagnostic only; a distant destination must not prevent a bounded approach.
The original execution monitor still owns joint-limit and target-validity checks.
"""
from dataclasses import dataclass

import numpy as np

from servo.execution_safety import ExecutionSafetyMonitor, SafetyDecision


@dataclass(frozen=True)
class HandFollowLimits:
    progress_window_s: float = 3.0
    minimum_progress_m: float = 0.002
    resume_target_motion_m: float = 0.02
    maximum_sample_gap_s: float = 0.25
    settled_range_m: float = 0.01
    settled_image_px: float = 12.0
    reference_position_tolerance_m: float = 0.002
    reference_joint_tolerance_rad: float = 0.001


class HandFollowSafetyMonitor(ExecutionSafetyMonitor):
    """Use the installed reference for tracking error; stop stalled approaches.

    A stalled target remains held until the hand moves at least 2 cm. Merely
    waiting, losing/reacquiring the same hand, or repeating frames cannot cause
    repeated attempts to push farther into a joint boundary.
    """

    def __init__(self, model, *, position_leash, joint_leash, params=None):
        super().__init__(params=params)
        self.model = model
        self.reach_center, self.reach_radius = model.tcp_reach_bound()
        self.position_leash = float(position_leash)
        self.joint_leash = float(joint_leash)
        self.limits = HandFollowLimits()
        self._reference_error_since = None
        self._last_evaluation_time = None
        self._stalled_hand = None
        self._progress_since = None
        self._progress_m = 0.0
        self._required_progress_m = self.limits.minimum_progress_m
        self._previous_q = None
        self._previous_time = None

    def _clear_progress(self):
        self._progress_since = None
        self._progress_m = 0.0
        self._previous_q = None
        self._previous_time = None

    def evaluate(self, q_now, servo_out, now, *, reference_tcp=None, reference_q=None):
        detail = servo_out.get('camera_center')
        # Fixed-pose retreat and ordinary PBVS keep their existing policy.
        if not servo_out.get('center_control') or detail is None:
            self._clear_progress()
            self._reference_error_since = None
            self._last_evaluation_time = None
            return super().evaluate(q_now, servo_out, now)

        # Final hand range is a destination, not the reference already sent to
        # MIT. Preserve all base checks without applying its 20 cm destination
        # timeout to a normal, rate-limited approach.
        decision = super().evaluate(q_now, dict(servo_out, error=np.zeros(3)), now)
        if decision.hold:
            self._clear_progress()
            self._reference_error_since = None
            self._last_evaluation_time = None
            return decision

        try:
            q = np.asarray(q_now, dtype=float).reshape(7)
            hand = np.asarray(servo_out['marker_center'], dtype=float).reshape(3)
            target = np.asarray(servo_out['target_tcp'], dtype=float).reshape(3)
            error = np.asarray(servo_out['error'], dtype=float).reshape(3)
            A = np.asarray(servo_out['feature_jacobian'], dtype=float).reshape(3, 7)
            tcp_reference = np.asarray(reference_tcp, dtype=float).reshape(3)
            q_reference = np.asarray(reference_q, dtype=float).reshape(7)
            tcp = np.asarray(self.model.fk(q)[0], dtype=float).reshape(3)
            pixel_error = np.asarray(detail['pixel_error'], dtype=float).reshape(2)
            range_error = float(detail['distance_error_mm']) / 1000.
            if not all(np.isfinite(x).all() for x in (
                    q, hand, target, error, A, tcp_reference, q_reference, tcp, pixel_error)):
                raise ValueError('nonfinite hand/reference geometry')
            if not np.isfinite(range_error) or not np.isfinite(now):
                raise ValueError('nonfinite hand range or time')
        except (KeyError, TypeError, ValueError) as exc:
            self._clear_progress()
            self._reference_error_since = self._last_evaluation_time = None
            return SafetyDecision('hand_geometry_hold', hold=True, reason=str(exc))

        desired_distance = float(np.linalg.norm(target-hand))
        position_error = float(np.linalg.norm(tcp_reference-tcp))
        joint_error = float(np.max(np.abs(q_reference-q)))
        detail['execution_safety'] = dict(
            reference_position_error_mm=position_error*1000,
            reference_joint_error_rad=joint_error,
            reach_radius_m=self.reach_radius,
            reach_bound_policy='diagnostic_only',
            outside_reach_bound=bool(np.linalg.norm(hand-self.reach_center)
                                    > self.reach_radius+desired_distance+1e-9),
            progress_window_s=self.limits.progress_window_s,
            minimum_progress_mm=self.limits.minimum_progress_m*1000)

        reference_gap = None if self._last_evaluation_time is None else now-self._last_evaluation_time
        if reference_gap is None or not 0 < reference_gap <= self.limits.maximum_sample_gap_s:
            self._reference_error_since = None
        self._last_evaluation_time = now
        outside_reference = (
            position_error > self.position_leash + self.limits.reference_position_tolerance_m
            or joint_error > self.joint_leash + self.limits.reference_joint_tolerance_rad)
        if outside_reference:
            if self._reference_error_since is None or now < self._reference_error_since:
                self._reference_error_since = now
            if now-self._reference_error_since >= self.params.max_tracking_error_s:
                self._clear_progress()
                return SafetyDecision('reference_tracking_hold', hold=True,
                    reason=f'installed reference lag: TCP={position_error*1000:.1f}mm, joint={joint_error:.4f}rad')
        else:
            self._reference_error_since = None

        if self._stalled_hand is not None:
            if np.linalg.norm(hand-self._stalled_hand) < self.limits.resume_target_motion_m:
                return SafetyDecision('hand_approach_stalled_hold', hold=True,
                    reason='no measured progress; move the hand at least 20mm to retry')
            self._stalled_hand = None
            self._clear_progress()

        needs_motion = (abs(range_error) > self.limits.settled_range_m
                        or np.linalg.norm(pixel_error) > self.limits.settled_image_px)
        if not needs_motion:
            self._clear_progress()
            return SafetyDecision('tracking', hold=False)
        if detail.get('solver_status') == 'motion_disabled':
            self._clear_progress()
            return SafetyDecision('hand_motion_disabled_hold', hold=True)

        gap = None if self._previous_time is None else now-self._previous_time
        if gap is None or not 0 < gap <= self.limits.maximum_sample_gap_s:
            self._clear_progress()
            self._progress_since = now
            self._required_progress_m = min(self.limits.minimum_progress_m,
                                            .2*float(np.linalg.norm(error)))
        else:
            # Project MEASURED joint displacement onto the current visual-task
            # gradient. A hand moving toward an unmoving robot, camera noise,
            # or a moving host reference cannot masquerade as robot progress.
            norm = float(np.linalg.norm(error))
            if norm > 1e-9:
                self._progress_m -= float(error @ (A @ (q-self._previous_q))) / norm
        self._previous_q, self._previous_time = q.copy(), now
        elapsed = now-self._progress_since
        detail['execution_safety'].update(measured_progress_mm=self._progress_m*1000,
                                          required_progress_mm=self._required_progress_m*1000,
                                          progress_elapsed_s=elapsed)
        if elapsed >= self.limits.progress_window_s:
            if self._progress_m < self._required_progress_m:
                required_mm = self._required_progress_m*1000
                self._stalled_hand = hand.copy()
                self._clear_progress()
                return SafetyDecision('hand_approach_stalled_hold', hold=True,
                    reason=f'less than {required_mm:.2f}mm measured task progress in 3s; holding until hand moves 20mm')
            self._progress_since, self._progress_m = now, 0.0
            self._required_progress_m = min(self.limits.minimum_progress_m,
                                            .2*float(np.linalg.norm(error)))
        return SafetyDecision('tracking', hold=False)
