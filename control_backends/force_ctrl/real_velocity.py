"""Shared REAL motor-velocity reader for the Nero arm (the 2026-06 "gold key").

The pyAgxArm SDK zeros ``get_motor_states().velocity`` (a
``# TODO: remove this after the bug is fixed`` line in
``protocols/can_protocol/drivers/nero/versions/v111/driver.py``), but the parser
object ``robot._parser.motor_state_<i>.msg.velocity`` holds the REAL value
decoded from the 0x5 high-speed CAN frame. Reading the parser DIRECTLY (never
calling ``get_motor_states``, which would zero the field in place) yields a
clean velocity that finite-difference can never provide without lag — the
missing piece for a proper Cartesian damper ``-B*J*qdot``.

Single source of truth shared by:
  * ``control_backends.force_ctrl.mit_loop.NeroMITLoop._read_real_joint_vel``
    (the 200 Hz in-loop impedance damper)
  * ``scripts/test_official_cartesian_impedance.py`` (standalone tuning harness)

Probe-confirmed (``scripts/probe_motor_velocity_raw.py``): LIVE on all 7 joints,
scale ~1.0 (joint-side rad/s), sign [-1,-1,-1,-1,-1,+1,-1] (joint6 correct, the
rest negated — the SAME bug family as ``move_mit``'s
``if joint_index != 6: v_des *= -1``).

This function returns the RAW sign-corrected velocity (NO filtering). Low-pass
filtering is the caller's concern — filter state is inherently per-instance (the
200 Hz loop keeps its own ``_real_vel_filtered``; the standalone harness uses
the value unfiltered).
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

# Real motor-velocity sign correction (joint6 correct, joints 1-5,7 negated).
# Same bug family as move_mit's `if joint_index != 6: v_des *= -1`.
DEFAULT_VELOCITY_SIGNS = np.array([-1.0, -1.0, -1.0, -1.0, -1.0, 1.0, -1.0], dtype=np.float64)


def read_real_qdot_sample(
    robot,
    signs: Optional[np.ndarray] = None,
    n_joints: int = 7,
) -> Optional[Tuple[np.ndarray, Optional[np.ndarray]]]:
    """Read velocity plus per-joint CAN feedback timestamps from the parser.

    The timestamps are frame-identity tokens only: callers should compare them
    for exact change, not use them as a control-loop ``dt``.  Older/fake SDK
    objects without timestamps still return a valid velocity and ``None`` for
    the timestamp array.

    Parameters
    ----------
    robot : Driver
        The Nero pyAgxArm driver instance (already connected).
    signs : (n_joints,) array, optional
        Per-joint sign correction. Defaults to :data:`DEFAULT_VELOCITY_SIGNS`.
    n_joints : int
        Number of joints (Nero = 7).
    """
    parser = getattr(robot, "_parser", None)
    if parser is None:
        return None
    if signs is None:
        signs = DEFAULT_VELOCITY_SIGNS
    signs = np.asarray(signs, dtype=np.float64).ravel()
    if signs.shape[0] != n_joints:
        return None
    raw = np.full(n_joints, np.nan, dtype=np.float64)
    timestamps = np.full(n_joints, np.nan, dtype=np.float64)
    for i in range(1, n_joints + 1):
        ms = getattr(parser, f"motor_state_{i}", None)
        if ms is None or getattr(ms, "msg", None) is None:
            continue
        v = getattr(ms.msg, "velocity", None)
        if v is not None:
            raw[i - 1] = float(v)
        stamp = getattr(ms, "timestamp", None)
        if stamp is not None:
            try:
                timestamps[i - 1] = float(stamp)
            except (TypeError, ValueError):
                pass
    if not np.all(np.isfinite(raw)):
        return None
    timestamp_result = timestamps if np.all(np.isfinite(timestamps)) else None
    return raw * signs, timestamp_result


def read_real_qdot(
    robot,
    signs: Optional[np.ndarray] = None,
    n_joints: int = 7,
) -> Optional[np.ndarray]:
    """Read sign-corrected REAL joint velocity from the parser.

    This compatibility wrapper keeps the original array-only API used by the
    standalone tuning harness.  The production loop uses
    :func:`read_real_qdot_sample` so its sustained-speed guard only counts new
    CAN feedback frames.
    """
    sample = read_real_qdot_sample(robot, signs=signs, n_joints=n_joints)
    return None if sample is None else sample[0]
