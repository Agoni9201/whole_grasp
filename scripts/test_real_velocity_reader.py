#!/usr/bin/env python3
"""Offline unit test for NeroMITLoop._read_real_joint_vel (no hardware).

Validates the sign-correction + fallback logic of the real-motor-velocity
reader that powers the Cartesian damper -B*J*qdot:

  1. SIGN     : raw parser velocity * velocity_signs -> correct joint velocity.
                Default signs [-1,-1,-1,-1,-1,+1,-1] (joint6 correct, rest
                negated), matching the probe (probe_motor_velocity_raw.py:
                sign_agree joint6=100%, rest=0%) and the same bug family as
                move_mit's `if joint_index != 6: v_des *= -1`.
  2. FALLBACK : returns None (-> loop falls back to finite-diff) when the
                parser is absent, a motor_state field is missing, or a value
                is NaN, so a bad read can NEVER inject garbage into the damper.
  3. SHAPE    : wrong-length signs -> None.

The method is bound to a lightweight stub (no robot/URDF needed). Run:

    python3 scripts/test_real_velocity_reader.py
"""
import os
import sys
import types

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(SCRIPT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from control_backends.force_ctrl.mit_loop import NeroMITLoop  # noqa: E402
from control_backends.force_ctrl.data_types import ImpedanceConfig  # noqa: E402

DEFAULT_SIGNS = np.array([-1.0, -1.0, -1.0, -1.0, -1.0, 1.0, -1.0])


class _FakeMsg:
    def __init__(self, velocity, current=0.0, position=0.0):
        self.velocity = velocity
        self.current = current
        self.position = position


class _FakeMotorState:
    def __init__(self, velocity):
        self.msg = _FakeMsg(velocity)


class _FakeParser:
    def __init__(self, velocities, missing=None):
        missing = missing or set()
        for i in range(1, 8):
            if i in missing:
                setattr(self, f"motor_state_{i}", None)
            else:
                setattr(self, f"motor_state_{i}", _FakeMotorState(velocities[i - 1]))


class _FakeRobot:
    def __init__(self, velocities, parser_missing=False, missing_motor=None):
        self._parser = None if parser_missing else _FakeParser(velocities, missing=missing_motor)


class _StubLoop:
    """Minimal stand-in exposing exactly what _read_real_joint_vel touches."""
    N_JOINTS = 7

    def __init__(self, velocities, parser_missing=False, missing_motor=None):
        self._robot = _FakeRobot(velocities, parser_missing, missing_motor)
        self._real_vel_filtered = np.zeros(self.N_JOINTS)
        self._real_vel_last_time = None


def _make_reader(stub):
    """Bind the REAL method to the stub (tests the actual loop code)."""
    return types.MethodType(NeroMITLoop._read_real_joint_vel, stub)


_DEFAULT = object()  # sentinel for "use the method's default sign vector"


def _cfg(signs=_DEFAULT, tau=0.0):
    return ImpedanceConfig(
        velocity_signs=(None if signs is _DEFAULT else np.asarray(signs, dtype=np.float64)),
        real_vel_filter_tau=tau,
    )


def main() -> int:
    failures = []

    def check(name, cond):
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
        if not cond:
            failures.append(name)

    print("=== _read_real_joint_vel: sign correction + fallback ===")

    # 1. SIGN: raw * signs, explicit signs, tau=0 (no low-pass lag).
    raw = np.array([0.5, -0.3, 0.2, -0.1, 0.4, 0.6, -0.2])
    stub = _StubLoop(raw.tolist())
    out = _make_reader(stub)(_cfg(signs=DEFAULT_SIGNS))
    expected = raw * DEFAULT_SIGNS
    check("explicit signs: out == raw*signs",
          out is not None and np.allclose(out, expected))
    check("explicit signs: joint6 preserved (raw 0.6 -> +0.6)",
          out is not None and abs(out[5] - 0.6) < 1e-9)
    check("explicit signs: joint1 flipped (raw 0.5 -> -0.5)",
          out is not None and abs(out[0] + 0.5) < 1e-9)

    # 2. DEFAULT signs (cfg.velocity_signs=None) -> method uses DEFAULT_SIGNS.
    raw_pos = [1.0] * 7
    stub2 = _StubLoop(raw_pos)
    out2 = _make_reader(stub2)(_cfg())
    check("default signs: joint6 stays +1", out2 is not None and abs(out2[5] - 1.0) < 1e-9)
    check("default signs: joints 1-5,7 -> -1",
          out2 is not None and np.allclose(out2[[0, 1, 2, 3, 4, 6]], -1.0))

    # 3. FALLBACK: parser missing -> None.
    stub3 = _StubLoop(raw.tolist(), parser_missing=True)
    out3 = _make_reader(stub3)(_cfg())
    check("fallback: parser missing -> None", out3 is None)

    # 4. FALLBACK: one motor_state field None -> None.
    stub4 = _StubLoop(raw.tolist(), missing_motor={3})
    out4 = _make_reader(stub4)(_cfg())
    check("fallback: missing motor_state_3 -> None", out4 is None)

    # 5. FALLBACK: NaN velocity -> None.
    raw_nan = raw.tolist(); raw_nan[6] = float("nan")
    stub5 = _StubLoop(raw_nan)
    out5 = _make_reader(stub5)(_cfg())
    check("fallback: NaN velocity -> None", out5 is None)

    # 6. SHAPE: wrong-length signs -> None.
    stub6 = _StubLoop(raw.tolist())
    out6 = _make_reader(stub6)(_cfg(signs=np.array([-1.0, 1.0, -1.0])))  # len 3
    check("shape: 3-length signs -> None", out6 is None)

    # 7. CONSISTENCY: two consecutive reads (tau>0) track the same signed value.
    stub7 = _StubLoop([0.4] * 7)
    reader = _make_reader(stub7)
    a = reader(_cfg(signs=DEFAULT_SIGNS, tau=0.004))
    b = reader(_cfg(signs=DEFAULT_SIGNS, tau=0.004))
    check("consistency: steady input -> signed output stable sign",
          a is not None and b is not None
          and np.all(np.sign(a) == np.sign(DEFAULT_SIGNS * 0.4))
          and np.all(np.sign(b) == np.sign(DEFAULT_SIGNS * 0.4)))

    print("\n" + ("ALL PASS" if not failures else f"FAILURES: {failures}"))
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
