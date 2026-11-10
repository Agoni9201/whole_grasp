"""Pure, hardware-free validation of Nero's grouped CAN feedback."""

import numpy as np

from .real_velocity import DEFAULT_VELOCITY_SIGNS


class ControlDeadlineMissed(RuntimeError):
    pass


class FeedbackStale(RuntimeError):
    pass


class FeedbackHealthMonitor:
    def __init__(self, max_age_s=0.1, max_skew_s=0.05, max_control_gap_s=0.1):
        values = (max_age_s, max_skew_s, max_control_gap_s)
        if not all(np.isfinite(value) and value > 0.0 for value in values):
            raise ValueError("MIT feedback/gap budgets must be finite and positive")
        self.max_age_s, self.max_skew_s, self.max_control_gap_s = map(float, values)
        self._previous = None
        self._clock_offset = None
        self.last_read_sample = None

    @staticmethod
    def timestamp_snapshot(robot):
        """Diagnostic-only cache read: no SDK calls or validation-state changes."""
        parser = getattr(robot, "_parser", None)
        def stamp(name):
            try:
                return float(getattr(parser, name).timestamp)
            except (AttributeError, TypeError, ValueError):
                return float("nan")
        return {
            "position_timestamps": np.repeat(
                [stamp(name) for name in ("joint_12", "joint_34", "joint_56", "joint_7")],
                [2, 2, 2, 1]),
            "motor_timestamps": np.array([stamp(f"motor_state_{i}") for i in range(1, 8)]),
        }

    @staticmethod
    def describe_snapshot(sample, wall_time):
        if sample is None:
            return None
        result = {"age_reference_wall_time": float(wall_time)}
        for kind in ("position", "motor"):
            values = np.asarray(sample[f"{kind}_timestamps"], dtype=float).copy()
            result[f"{kind}_timestamps"] = values
            result[f"{kind}_age_s"] = float(wall_time) - values
        for key in ("sampled_wall_time", "sampled_monotonic_time"):
            if key in sample:
                result[key] = sample[key]
        return result

    def check_cycle(self, elapsed):
        if not np.isfinite(elapsed) or elapsed < 0 or elapsed > self.max_control_gap_s:
            raise ControlDeadlineMissed(
                f"MIT control deadline missed: {elapsed * 1000:.1f}ms "
                f"> {self.max_control_gap_s * 1000:.1f}ms"
            )

    def validate_age(self, sample, wall_time, monotonic_time):
        if not np.isfinite([wall_time, monotonic_time]).all():
            raise RuntimeError("MIT host clock is invalid")
        offset = float(wall_time) - float(monotonic_time)
        if self._clock_offset is None:
            self._clock_offset = offset
        clock_jump = offset - self._clock_offset
        if abs(clock_jump) > self.max_skew_s:
            raise RuntimeError(
                f"MIT wall clock changed by {clock_jump * 1000:+.1f}ms; "
                "CAN feedback age is untrusted. Stabilize host time synchronization "
                "before restarting the arm session"
            )
        self._clock_offset = offset  # allow gradual clock slew, reject a discontinuous jump
        stamps = np.r_[sample["position_timestamps"], sample["motor_timestamps"]]
        age = float(wall_time) - stamps
        if not np.isfinite(stamps).all() or np.any(stamps <= 0):
            raise RuntimeError("MIT feedback has missing/invalid per-axis timestamps")
        if np.any(age < -self.max_skew_s):
            raise RuntimeError("MIT feedback has future per-axis timestamps")
        if np.any(age > self.max_age_s):
            index = int(np.argmax(age))
            raise FeedbackStale(
                f"MIT feedback stale/future: {sample['labels'][index]} "
                f"age={age[index] * 1000:.1f}ms; limit={self.max_age_s * 1000:.1f}ms"
            )
        # Position groups and motor frames are two independently scheduled
        # streams. Validate coherence within each stream, not equal timestamps.
        for kind in ("position_timestamps", "motor_timestamps"):
            if np.ptp(sample[kind]) > self.max_skew_s:
                raise RuntimeError(f"MIT incoherent {kind}: inter-axis skew exceeded")
        return float(np.max(age))

    def read(self, robot, wall_time, monotonic_time):
        self.last_read_sample = None
        parser = getattr(robot, "_parser", None)
        if parser is None:
            raise RuntimeError("MIT per-axis feedback requires the SDK parser cache")
        q, position_stamps, motor_stamps, velocities, torques = [], [], [], [], []
        for name, fields in (
            ("joint_12", ("joint_1", "joint_2")),
            ("joint_34", ("joint_3", "joint_4")),
            ("joint_56", ("joint_5", "joint_6")),
            ("joint_7", ("joint_7",)),
        ):
            wrapped = getattr(parser, name, None)
            try:
                stamp = float(wrapped.timestamp)
                values = [float(getattr(wrapped.msg, field)) for field in fields]
                if stamp != float(wrapped.timestamp):
                    raise ValueError("frame changed during read")
            except (AttributeError, TypeError, ValueError) as exc:
                raise RuntimeError(f"MIT position feedback unavailable: {name}") from exc
            q.extend(values)
            position_stamps.extend([stamp] * len(fields))
        for index in range(1, 8):
            wrapped = getattr(parser, f"motor_state_{index}", None)
            try:
                stamp = float(wrapped.timestamp)
                values = [float(getattr(wrapped.msg, key)) for key in ("velocity", "torque")]
                if not np.isfinite(values).all() or stamp != float(wrapped.timestamp):
                    raise ValueError("invalid/changing motor frame")
            except (AttributeError, TypeError, ValueError) as exc:
                raise RuntimeError(f"MIT motor feedback unavailable: joint{index}") from exc
            motor_stamps.append(stamp)
            velocities.append(values[0])
            torques.append(values[1])
        q = np.asarray(q, dtype=np.float64)
        if q.shape != (7,) or not np.isfinite(q).all():
            raise RuntimeError("MIT position feedback is non-finite")
        sample = {
            "q": q,
            "dq": np.asarray(velocities) * DEFAULT_VELOCITY_SIGNS,
            "dq_raw": np.asarray(velocities),
            "tau": np.asarray(torques),
            "sampled_wall_time": float(wall_time),
            "sampled_monotonic_time": float(monotonic_time),
            "position_timestamps": np.asarray(position_stamps),
            "motor_timestamps": np.asarray(motor_stamps),
            "labels": [f"position_joint{i}" for i in range(1, 8)]
            + [f"motor_joint{i}" for i in range(1, 8)],
        }
        # Retain the attempted frame even when age/coherence validation fails;
        # logging the previous successful frame would misdiagnose the fault.
        self.last_read_sample = sample
        self.validate_age(sample, wall_time, monotonic_time)
        stamps = np.r_[sample["position_timestamps"], sample["motor_timestamps"]]
        if self._previous is not None and np.any(stamps < self._previous):
            raise RuntimeError("MIT per-axis feedback timestamp went backwards")
        self._previous = stamps.copy()
        return sample
