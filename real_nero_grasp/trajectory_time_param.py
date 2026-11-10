"""Path-preserving quintic time parameterization for joint trajectories.

A geometric joint path ``q(s)`` (here: straight joint-space edges between the
NBV planner's collision-validated waypoints) answers "which configurations does
the arm pass through".  Time parameterization answers "at which time is it at
each configuration, and with what velocity / acceleration".

Each straight joint-space edge is scaled by a quintic ``s(t/T)`` so the exact
geometric path is preserved (the collision validation performed on the joint
line therefore stays valid), while motion starts and stops with zero velocity
and zero acceleration.

Pure functions with no hardware or project dependencies: unit-testable offline.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "N_JOINTS",
    "quintic_s",
    "quintic_s_dot",
    "quintic_s_ddot",
    "JointSegment",
    "JointTrajectory",
    "time_parameterize_segment",
    "sample_segment",
    "build_joint_trajectory",
]

N_JOINTS = 7

# Peaks of s(tau) = 6*tau^5 - 15*tau^4 + 10*tau^3 on tau in [0, 1]:
#   max |ds/dtau|    = 15/8          at tau = 0.5
#   max |d2s/dtau2|  = 10/sqrt(3)    at tau = (1 -/+ 1/sqrt(3))/2
_PEAK_S_DOT = 15.0 / 8.0
_PEAK_S_DDOT = 10.0 / np.sqrt(3.0)


def quintic_s(tau):
    tau = min(max(float(tau), 0.0), 1.0)
    return 10.0 * tau ** 3 - 15.0 * tau ** 4 + 6.0 * tau ** 5


def quintic_s_dot(tau):
    tau = min(max(float(tau), 0.0), 1.0)
    return 30.0 * tau ** 2 - 60.0 * tau ** 3 + 30.0 * tau ** 4


def quintic_s_ddot(tau):
    tau = min(max(float(tau), 0.0), 1.0)
    return 60.0 * tau - 180.0 * tau ** 2 + 120.0 * tau ** 3


def _as_q(value):
    q = np.asarray(value, dtype=np.float64).reshape(N_JOINTS)
    if not np.all(np.isfinite(q)):
        raise ValueError("joint vector must contain only finite values")
    return q


@dataclass(frozen=True)
class JointSegment:
    """One time-parameterized straight joint-space edge."""

    q_start: np.ndarray
    q_goal: np.ndarray
    duration_s: float
    max_v_rad_s: float
    max_a_rad_s2: float

    def __post_init__(self):
        object.__setattr__(self, "q_start", _as_q(self.q_start))
        object.__setattr__(self, "q_goal", _as_q(self.q_goal))
        duration = float(self.duration_s)
        if not np.isfinite(duration) or duration < 0.0:
            raise ValueError("duration_s must be finite and non-negative")
        object.__setattr__(self, "duration_s", duration)


def time_parameterize_segment(q_start, q_goal, v_max_rad_s, a_max_rad_s2):
    """Synchronized quintic timing for one joint-space edge.

    All seven axes are synchronized by choosing the duration required by the
    slowest axis::

        max |dq/dt|   = (15/8)     * |dq| / T
        max |d2q/dt2| = (10/sqrt3) * |dq| / T^2
    """
    q_start = _as_q(q_start)
    q_goal = _as_q(q_goal)
    v_max = float(v_max_rad_s)
    a_max = float(a_max_rad_s2)
    if not np.isfinite(v_max) or v_max <= 0.0:
        raise ValueError("v_max_rad_s must be finite and positive")
    if not np.isfinite(a_max) or a_max <= 0.0:
        raise ValueError("a_max_rad_s2 must be finite and positive")

    delta = np.abs(q_goal - q_start)
    if float(np.max(delta)) <= 1e-12:
        return JointSegment(
            q_start, q_goal, duration_s=0.0, max_v_rad_s=0.0, max_a_rad_s2=0.0
        )

    t_v = _PEAK_S_DOT * delta / v_max
    t_a = np.sqrt(_PEAK_S_DDOT * delta / a_max)
    duration = float(np.max(np.maximum(t_v, t_a)))
    if duration < 1e-3:
        duration = 1e-3

    max_v = float(np.max(_PEAK_S_DOT * delta / duration))
    max_a = float(np.max(_PEAK_S_DDOT * delta / (duration ** 2)))
    return JointSegment(
        q_start, q_goal, duration_s=duration, max_v_rad_s=max_v, max_a_rad_s2=max_a
    )


def sample_segment(seg, t):
    """Return ``(q, dq, ddq)`` at time ``t`` within one segment.

    ``t`` outside ``[0, T]`` is clamped, so the terminal velocity/acceleration
    are naturally zero.
    """
    if seg.duration_s <= 0.0:
        return (
            seg.q_goal.copy(),
            np.zeros(N_JOINTS, dtype=np.float64),
            np.zeros(N_JOINTS, dtype=np.float64),
        )
    tau = min(max(float(t) / seg.duration_s, 0.0), 1.0)
    s = quintic_s(tau)
    ds_dt = quintic_s_dot(tau) / seg.duration_s
    dds_dt = quintic_s_ddot(tau) / (seg.duration_s ** 2)
    delta = seg.q_goal - seg.q_start
    return seg.q_start + s * delta, ds_dt * delta, dds_dt * delta


class JointTrajectory:
    """Concatenated time-parameterized joint-space segments with global sampling."""

    def __init__(self, segments):
        self.segments = list(segments)
        starts = [0.0]
        for seg in self.segments:
            starts.append(starts[-1] + float(seg.duration_s))
        self._starts = starts
        self.duration_s = float(starts[-1])

    def sample(self, t):
        t = min(max(float(t), 0.0), self.duration_s)
        for index, seg in enumerate(self.segments):
            end = self._starts[index + 1]
            if t <= end + 1e-12 or index == len(self.segments) - 1:
                return sample_segment(seg, t - self._starts[index])
        last = self.segments[-1]
        return sample_segment(last, self.duration_s - self._starts[-1])


def build_joint_trajectory(waypoints, v_max_rad_s, a_max_rad_s2):
    """Time-parameterize a sequence of joint waypoints into consecutive segments.

    ``waypoints`` is a sequence of (7,) arrays; the first entry is the start and
    each subsequent entry is a validated goal.  Returns a ``JointTrajectory``.
    """
    waypoints = [_as_q(w) for w in waypoints]
    if len(waypoints) < 2:
        raise ValueError("need at least two waypoints to build a trajectory")
    segments = [
        time_parameterize_segment(
            waypoints[index], waypoints[index + 1], v_max_rad_s, a_max_rad_s2
        )
        for index in range(len(waypoints) - 1)
    ]
    return JointTrajectory(segments)
