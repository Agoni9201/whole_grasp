#!/usr/bin/env python3
"""Decisive probe: does get_motor_states().velocity actually report motion?

Uses the SAME validated zero-gravity backend as the execute script's preflight
(NeroForceController zero_gravity=True) — the path you already confirmed is
hand-draggable. While you drag, it reads the RAW driver velocity field
(get_motor_states().msg.velocity, NOT the finite-diff estimate).

If velocity stays 0 while you clearly move the arm -> firmware/driver does NOT
decode velocity -> impedance damping must use firmware kd only.
If it tracks your motion -> real velocity available -> use it for software damping.

    python3 scripts/probe_motor_velocity.py --execute
Drag the arm around for the whole sample window.
"""
import argparse, os, sys, time
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(SCRIPT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--execute", action="store_true", required=True)
    ap.add_argument("--channel", default="can0")
    ap.add_argument("--firmware", default="v111")
    ap.add_argument("--duration", type=float, default=20.0, help="Drag-sample window (s).")
    args = ap.parse_args()

    from control_backends.force_ctrl.controller import NeroForceController
    # Same as execute preflight: validated draggable zero-gravity.
    ctrl = NeroForceController(channel=args.channel, firmware=args.firmware,
                              zero_gravity=True, torque_limit=14.0)
    ctrl.start()
    robot = ctrl._robot

    # Wait for feedback + settle into zero-gravity.
    deadline = time.monotonic() + 5.0
    while ctrl.get_joint_pos() is None and time.monotonic() < deadline:
        time.sleep(0.05)
    q0 = ctrl.get_joint_pos()
    if q0 is None:
        print("FAIL: no joint feedback"); ctrl.stop(); return 1
    print(f"Zero-gravity active (validated backend). start q(deg)={np.round(np.degrees(q0),1)}")
    print(">>> Confirm you can DRAG the arm now. You have 5s to grab it. <<<")
    time.sleep(5.0)

    print("=" * 60)
    print(f">>> DRAG THE ARM NOW for {args.duration:.0f}s (move EVERY joint) <<<")
    print("=" * 60)

    samples = []
    t0 = time.monotonic()
    last = 0.0
    while time.monotonic() - t0 < args.duration:
        # Read RAW driver velocity field (not the loop's finite-diff).
        row = []
        for i in range(1, 8):
            try:
                ms = robot.get_motor_states(i)
                v = ms.msg.velocity if (ms is not None and ms.msg is not None) else float("nan")
            except Exception:
                v = float("nan")
            row.append(v)
        samples.append(row)
        now = time.monotonic()
        if now - last > 1.0:
            arr = np.array(row)
            print(f"  t={now-t0:4.1f}s  raw velocity (rad/s): {np.round(arr,3).tolist()}")
            last = now
        time.sleep(0.02)

    ctrl.stop()
    V = np.array(samples)
    print("\n" + "=" * 60)
    print("RESULTS (raw driver velocity field while you dragged):")
    for i in range(7):
        col = V[:, i]
        col = col[np.isfinite(col)]
        if len(col) == 0:
            print(f"  joint{i+1}: no data"); continue
        print(f"  joint{i+1}: min={col.min():+.4f}  max={col.max():+.4f}  "
              f"|max|={np.abs(col).max():.4f}  std={col.std():.4f} rad/s")
    overall = float(np.nanmax(np.abs(V))) if len(samples) else 0.0
    print(f"\n  overall max|velocity| during drag: {overall:.4f} rad/s")
    if overall > 0.1:
        print("  -> VELOCITY IS REPORTED. Real velocity available -> use it for damping.")
    else:
        print("  -> VELOCITY NOT REPORTED (stub). Use firmware kd for damping.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
