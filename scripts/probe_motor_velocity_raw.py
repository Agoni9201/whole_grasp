#!/usr/bin/env python3
"""Decisive probe: is joint velocity REAL on the CAN bus, just discarded by the SDK?

BACKGROUND (the finding):
  `get_motor_states(i).msg.velocity` always reads 0 — but NOT because the arm
  doesn't report velocity. The SDK v111 driver DELIBERATELY zeroes it:

      protocols/.../nero/versions/v111/driver.py:187-188
          # TODO: remove this after the bug is fixed
          motor_state.msg.velocity = 0.0

  That line runs INSIDE get_motor_states, AFTER the parser already decoded the
  velocity from the 0x5 high-speed CAN frame (Byte 0-1, int16, 0.001 rad/s).
  It's the same "TODO bug" family as the per-joint v_des sign flip
  (`if joint_index != 6: v_des *= -1`) and the current sign flip
  (`motor_state.msg.current *= -1`) — i.e. an unfinished sign/scale
  calibration Agilex ships disabled.

  The parser object `robot._parser.motor_state_<i>.msg.velocity` holds the
  value BEFORE the zeroing. This probe reads THAT directly — bypassing
  get_motor_states entirely — while you drag the arm in zero-gravity mode.

WHAT TO LOOK FOR (drag ONE joint slowly and steadily in the + direction):
  * If `raw_v` tracks `fdiff_v` (same sign, similar magnitude)  -> velocity is
    LIVE on the bus, recoverable. => proper Cartesian damping -B*J*qdot is on.
  * If `raw_v` stays 0 while `fdiff_v` shows motion -> parser does not surface
    velocity; stuck with finite-diff (still usable at 200 Hz, but confirmed).
  * `torque` / `current` nonzero under load -> confirms the 0x5 frame fields are
    decoded (makes velocity-decode likely).

    python3 scripts/probe_motor_velocity_raw.py --execute
    python3 scripts/probe_motor_velocity_raw.py --execute --duration 25 --hz 100
Drag each joint in turn. Emergency stop within reach.
"""
import argparse, os, sys, time
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(SCRIPT_DIR, ".."))
if PROJECT_ROOT in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
else:
    sys.path.insert(0, PROJECT_ROOT)


def _read_raw(parser, i):
    """Read motor_state_<i> straight off the parser — pre-zeroing values.

    Returns (velocity, current, torque, position, hz) or NaNs if no frame yet.
    This NEVER calls robot.get_motor_states(), so the SDK's `velocity = 0.0`
    discard line is never executed.
    """
    ms = getattr(parser, f"motor_state_{i}", None)
    if ms is None or getattr(ms, "msg", None) is None:
        return (float("nan"),) * 5
    m = ms.msg
    try:
        hz = getattr(ms, "hz", float("nan"))
    except Exception:
        hz = float("nan")
    return (
        float(getattr(m, "velocity", float("nan"))),
        float(getattr(m, "current", float("nan"))),
        float(getattr(m, "torque", float("nan"))),     # derived: slope*cur/gear
        float(getattr(m, "position", float("nan"))),
        float(hz),
    )


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--execute", action="store_true", required=True)
    ap.add_argument("--channel", default="can0")
    ap.add_argument("--firmware", default="v111")
    ap.add_argument("--duration", type=float, default=20.0, help="Drag window (s).")
    ap.add_argument("--hz", type=float, default=80.0, help="Sample rate (Hz).")
    args = ap.parse_args()

    from control_backends.force_ctrl.controller import NeroForceController

    # Same validated draggable zero-gravity backend as the execute preflight.
    ctrl = NeroForceController(
        channel=args.channel, firmware=args.firmware,
        zero_gravity=True, torque_limit=14.0,
    )
    ctrl.start()
    robot = ctrl._robot
    parser = getattr(robot, "_parser", None)
    if parser is None:
        print("FAIL: cannot reach robot._parser (SDK layout changed)."); ctrl.stop(); return 1

    # Wait for feedback + settle into zero-gravity.
    deadline = time.monotonic() + 5.0
    while ctrl.get_joint_pos() is None and time.monotonic() < deadline:
        time.sleep(0.05)
    q0 = ctrl.get_joint_pos()
    if q0 is None:
        print("FAIL: no joint feedback"); ctrl.stop(); return 1
    print(f"Zero-gravity active. start q(deg)={np.round(np.degrees(q0),1)}")

    # Sanity: confirm parser-direct position matches the controller's position
    # (proves we are reading the right object).
    raw = [_read_raw(parser, i) for i in range(1, 8)]
    p_parser = np.array([r[3] for r in raw])
    if np.all(np.isfinite(p_parser)):
        print(f"parser-direct position (deg) = {np.round(np.degrees(p_parser),1)}")
        print(f"controller    position (deg) = {np.round(np.degrees(q0),1)}")
        if np.max(np.abs(p_parser - q0)) > 0.05:
            print("WARN: parser vs controller position differ >0.05 rad — object mismatch!")

    print(">>> Grab the arm. You have 5s. <<<")
    time.sleep(5.0)
    print("=" * 78)
    print(f">>> DRAG NOW for {args.duration:.0f}s — move EACH joint slowly, one at a <<<")
    print(f">>> time, in its + direction. Watch raw_v vs fdiff_v below.          <<<")
    print("=" * 78)

    dt = 1.0 / args.hz
    samples = []  # list of (t, raw_v[7], cur[7], trq[7], pos[7])
    t0 = time.monotonic()
    last_print = 0.0
    while time.monotonic() - t0 < args.duration:
        now = time.monotonic() - t0
        row_v, row_c, row_t, row_p, row_hz = [], [], [], [], []
        for i in range(1, 8):
            v, c, tq, p, hz = _read_raw(parser, i)
            row_v.append(v); row_c.append(c); row_t.append(tq); row_p.append(p); row_hz.append(hz)
        samples.append((now, row_v, row_c, row_t, row_p))

        if now - last_print > 1.0 and len(samples) >= 2:
            # Finite-diff velocity from the parser position, last two samples.
            _, _, _, p_prev = samples[-2][0], None, None, None
            t_prev, _, _, _, p_prev_row = samples[-2]
            fdiff = np.array([
                (row_p[k] - p_prev_row[k]) / max(now - t_prev, 1e-4)
                for k in range(7)
            ])
            raw_v = np.array(row_v)
            print(f"  t={now:4.1f}s  raw_v={np.round(raw_v,3).tolist()}")
            print(f"           fdiff={np.round(fdiff,3).tolist()}")
            last_print = now
        time.sleep(dt)

    ctrl.stop()

    # ---- Analysis ----
    if len(samples) < 4:
        print("Not enough samples."); return 1
    V = np.array([s[1] for s in samples])      # (N,7) raw velocity
    C = np.array([s[2] for s in samples])      # current
    T = np.array([s[3] for s in samples])      # torque
    P = np.array([s[4] for s in samples])      # position
    tt = np.array([s[0] for s in samples])
    # Finite-diff velocity (central difference) from parser position.
    dpos = np.diff(P, axis=0); dtm = np.diff(tt)[:, None]
    FV = dpos / np.clip(dtm, 1e-4, None)       # (N-1,7)

    print("\n" + "=" * 78)
    print("RESULTS  (raw = parser-direct pre-zeroing; fdiff = from position)")
    print("=" * 78)
    any_live = False
    for j in range(7):
        rv = V[:, j]; rv = rv[np.isfinite(rv)]
        fv = FV[:, j]; fv = fv[np.isfinite(fv)]
        tq = T[:, j]; tq = tq[np.isfinite(tq)]
        raw_max = float(np.max(np.abs(rv))) if rv.size else 0.0
        fd_max = float(np.max(np.abs(fv))) if fv.size else 0.0
        tq_max = float(np.max(np.abs(tq))) if tq.size else 0.0
        # Sign agreement on samples where motion is unambiguous.
        agree = sign_agreement(rv, fv) if (rv.size and fv.size) else float("nan")
        live = raw_max > 0.05
        if live:
            any_live = True
        scale = (raw_max / fd_max) if fd_max > 0.05 else float("nan")
        flag = "LIVE" if live else "zero"
        print(f"  joint{j+1}: raw|v|max={raw_max:6.3f}  fdiff|v|max={fd_max:6.3f}  "
              f"scale(raw/fd)={scale if scale==scale else 0:5.2f}  "
              f"sign_agree={agree*100 if agree==agree else 0:4.0f}%  "
              f"|torque|max={tq_max:5.2f}  [{flag}]")

    print("\nVERDICT:")
    if any_live:
        print("  -> Raw velocity is LIVE on the CAN bus (the SDK just discards it).")
        print("     Cartesian damping -B*J*qdot is achievable with real velocity.")
        print("     Next: calibrate per-joint sign (+ scale if raw/fd != ~1), then")
        print("           wire parser-direct velocity into the 200 Hz impedance loop.")
    else:
        print("  -> Raw velocity is NOT surfaced by the parser either (all ~0).")
        print("     Velocity source = finite-diff position only (already used).")
        print("     Impedance damping stays on firmware kd (current approach).")
    return 0


def sign_agreement(a, b):
    """Fraction of samples where sign(a)==sign(b) (only where |b|>0.1)."""
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    mask = np.abs(b) > 0.1
    if not np.any(mask):
        return float("nan")
    return float(np.mean(np.sign(a[mask]) == np.sign(b[mask])))


if __name__ == "__main__":
    sys.exit(main())
