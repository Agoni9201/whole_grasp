#!/usr/bin/env python3
"""Phase A1-A3: bench gates for Cartesian impedance (no tracking, arm powered).

Three cheap, decisive checks that MUST pass before any powered impedance
tracking (IMPEDANCE_CONTROL_PLAN.md §7). None of them move the arm
significantly; they probe firmware behavior and measure noise.

    python3 scripts/test_impedance_bench_gates.py --gate A --execute   # joint-7 probe
    python3 scripts/test_impedance_bench_gates.py --gate B --execute   # velocity noise
    python3 scripts/test_impedance_bench_gates.py --gate C             # gravity budget (offline, no --execute)

Gate A (HEAD RISK): does move_mit actually control joint 7 on this firmware?
    Sends a tiny t_ff on joint 7 (kp=0) and reads back whether joint 7
    torque/velocity responds. If joint 7 is silently dropped (v111 ambiguity),
    7-DOF impedance is unsafe -> restrict to joints 1-6 or switch firmware.

Gate B: how noisy is the raw joint velocity used for impedance damping?
    Samples _read_joint_vel with the arm physically still for ~10s and reports
    per-joint RMS. If > ~0.05 rad/s RMS, damping must be filtered (mit_loop)
    before powered tracking.

Gate C (offline): does |gravity_torque| + impedance_torque_limit fit under
    torque_limit on every joint across the tracking workspace? Uses the MuJoCo
    gravity model on a grid of poses. No arm needed.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(SCRIPT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from control_backends.force_ctrl.controller import (  # noqa: E402
    N_JOINTS,
    ZERO_GRAV_KD,
    NeroForceController,
)


def _wait_joint_state(ctrl, timeout: float = 5.0):
    """Wait for the MIT loop to populate joint state; returns JointState or None."""
    deadline = time.monotonic() + timeout
    s = ctrl.get_joint_state()
    while s is None and time.monotonic() < deadline:
        time.sleep(0.05)
        s = ctrl.get_joint_state()
    return s


def gate_a_joint7_probe(args) -> int:
    """Probe whether move_mit controls joint 7 on the installed firmware."""
    print("=" * 60)
    print("GATE A: joint-7 MIT control probe")
    print("=" * 60)
    print("Arm must be ENABLED but free to hold position. A tiny t_ff (0.2 Nm)")
    print("is sent on joint 7 for ~1s. We check whether joint 7 torque/velocity")
    print("responds. Keep hands clear.\n")

    ctrl = NeroForceController(
        channel=args.channel,
        firmware=args.firmware,
        zero_gravity=False,
        torque_limit=14.0,
    )
    ctrl.start()
    try:
        # Hold current pose with weak gains, then pulse t_ff on joint 7.
        q0 = ctrl.get_joint_pos()
        if q0 is None:
            print("FAIL: no joint feedback")
            return 1
        print(f"start q (deg) = {np.degrees(q0)}")

        # Baseline: hold with zero t_ff, read joint7 state.
        ctrl.command_joint_state(pos=q0, vel=np.zeros(N_JOINTS),
                                 kp=np.zeros(N_JOINTS), kd=ZERO_GRAV_KD,
                                 torques=np.zeros(N_JOINTS), gravity_enabled=True)
        time.sleep(0.5)
        s0 = _wait_joint_state(ctrl)
        if s0 is None:
            print("FAIL: no joint state for baseline")
            return 1

        # Pulse: add t_ff only on joint 7.
        tff = np.zeros(N_JOINTS)
        tff[6] = args.probe_torque
        ctrl.command_joint_state(pos=q0, vel=np.zeros(N_JOINTS),
                                 kp=np.zeros(N_JOINTS), kd=ZERO_GRAV_KD,
                                 torques=tff, gravity_enabled=True)
        time.sleep(1.0)
        s1 = _wait_joint_state(ctrl)
        if s1 is None:
            s1 = s0  # fallback so the release+print still work

        # Release.
        ctrl.command_joint_state(pos=q0, vel=np.zeros(N_JOINTS),
                                 kp=np.zeros(N_JOINTS), kd=ZERO_GRAV_KD,
                                 torques=np.zeros(N_JOINTS), gravity_enabled=True)

        tau7_delta = float(s1.torque[6] - s0.torque[6]) if s1.torque is not None and s0.torque is not None else 0.0
        q7_delta_deg = float(np.degrees(s1.pos[6] - s0.pos[6]))
        print(f"\nbaseline joint7 torque = {s0.torque[6] if s0.torque is not None else float('nan')}")
        print(f"pulsed  joint7 torque = {s1.torque[6] if s1.torque is not None else float('nan')}")
        print(f"joint7 torque delta = {tau7_delta:.3f} Nm  (probe t_ff = {args.probe_torque} Nm)")
        print(f"joint7 pos delta     = {q7_delta_deg:.3f} deg")

        responds = abs(tau7_delta) > 0.02 or abs(q7_delta_deg) > 0.5
        print("\n" + "=" * 60)
        if responds:
            print("PASS: joint 7 responds to move_mit t_ff -> 7-DOF impedance OK.")
            return 0
        print("INCONCLUSIVE/FAIL: joint 7 did not respond measurably.")
        print("Either t_ff is too small to overcome friction, or this firmware")
        print("silently drops joint 7. Re-run with larger --probe-torque, or")
        print("RESTRICT impedance to joints 1-6 (hold joint 7 stiff).")
        return 2
    finally:
        ctrl.stop()


def gate_b_velocity_noise(args) -> int:
    """Measure raw joint-velocity noise with the arm still."""
    print("=" * 60)
    print("GATE B: joint-velocity noise measurement")
    print("=" * 60)
    print("Keep the arm physically still (zero-gravity hold). Sampling raw")
    print(f"velocity for {args.sample_s:.0f}s.\n")

    ctrl = NeroForceController(
        channel=args.channel,
        firmware=args.firmware,
        zero_gravity=True,
        torque_limit=14.0,
    )
    ctrl.start()
    try:
        # Wait for the MIT loop to populate joint state (it returns None until
        # the first feedback frame arrives).
        warmup_deadline = time.monotonic() + 5.0
        s = ctrl.get_joint_state()
        while s is None and time.monotonic() < warmup_deadline:
            time.sleep(0.05)
            s = ctrl.get_joint_state()
        if s is None:
            print("FAIL: no joint state after 5s warmup (check arm connection).")
            return 1
        print(f"warmup OK: first state received, q(deg)={np.degrees(s.pos)}")
        # Settle a moment so transients from loop startup decay.
        time.sleep(1.0)

        samples = []
        t0 = time.monotonic()
        while time.monotonic() - t0 < args.sample_s:
            s = ctrl.get_joint_state()
            if s is not None and s.vel is not None:
                samples.append(np.asarray(s.vel, dtype=np.float64).ravel().copy())
            time.sleep(0.02)
        if len(samples) < 10:
            print("FAIL: too few samples")
            return 1
        V = np.vstack(samples)
        rms = np.sqrt(np.mean(V ** 2, axis=0))
        print(f"collected {len(samples)} samples")
        print(f"per-joint velocity RMS (rad/s):")
        for i, r in enumerate(rms):
            flag = "  <-- NOISY" if r > 0.05 else ""
            print(f"  joint{i+1}: {r:.4f}{flag}")
        max_rms = float(np.max(rms))
        print("\n" + "=" * 60)
        if max_rms > 0.05:
            print(f"ATTENTION: max RMS {max_rms:.4f} > 0.05 -> damping MUST be")
            print("low-pass filtered (or qdot synthesized from get_joint_angles")
            print("finite differences) before powered impedance tracking.")
            return 2
        print(f"PASS: max RMS {max_rms:.4f} <= 0.05 -> raw velocity usable.")
        return 0
    finally:
        ctrl.stop()


def gate_c_gravity_budget(args) -> int:
    """Offline: check |grav| + impedance_limit < torque_limit across workspace."""
    print("=" * 60)
    print("GATE C: gravity-compensation torque budget (offline, MuJoCo)")
    print("=" * 60)
    from control_backends.force_ctrl.gravity_comp import NeroGravityCompensator
    from models.nero_model import NeroPinocchioModel

    comp = NeroGravityCompensator()
    model = NeroPinocchioModel()
    lim = args.torque_limit
    imp_lim = args.impedance_torque_limit

    # Grid spanning a reasonable tracking workspace (radians).
    grid_q = []
    base = np.radians(np.array([4.0, 52.0, 3.0, 90.0, 0.0, 9.0, -46.0]))
    for d1 in (-0.3, 0.0, 0.3):
        for d4 in (-0.3, 0.0, 0.3):
            for d6 in (-0.3, 0.0, 0.3):
                q = base + np.array([d1, 0.0, 0.0, d4, 0.0, d6, 0.0])
                grid_q.append(model.clamp_to_limits(q))

    worst = np.zeros(N_JOINTS)
    worst_poses = [None] * N_JOINTS
    for q in grid_q:
        try:
            grav = comp.compute(q)
        except Exception as exc:
            print(f"  (skip pose, grav compute failed: {exc})")
            continue
        headroom_needed = np.abs(grav) + imp_lim
        for i in range(N_JOINTS):
            if headroom_needed[i] > worst[i]:
                worst[i] = headroom_needed[i]
                worst_poses[i] = q.copy()

    print(f"torque_limit = {lim} Nm, impedance_torque_limit = {imp_lim} Nm")
    print(f"checked {len(grid_q)} poses\n")
    print(f"{'joint':>6} {'max|grav|+imp':>14} {'headroom':>10} {'ok':>4}")
    all_ok = True
    for i in range(N_JOINTS):
        ok = worst[i] <= lim
        all_ok = all_ok and ok
        print(f"{i+1:>6} {worst[i]:>14.2f} {lim - worst[i]:>10.2f} {'YES' if ok else 'NO':>4}")
    print("\n" + "=" * 60)
    if all_ok:
        print(f"PASS: every joint has headroom for gravity + {imp_lim} Nm impedance.")
        return 0
    print("ATTENTION: some joint(s) would saturate. Lower impedance_torque_limit")
    print("(per-joint if needed) so that max|grav|+imp <= torque_limit everywhere.")
    return 2


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gate", choices=("A", "B", "C"), required=True)
    ap.add_argument("--execute", action="store_true", help="Required for gates A/B (real arm).")
    ap.add_argument("--channel", default="can0")
    ap.add_argument("--firmware", default="v111")
    ap.add_argument("--probe-torque", type=float, default=0.2, help="Gate A: t_ff pulse on joint 7 (Nm).")
    ap.add_argument("--sample-s", type=float, default=10.0, help="Gate B: sampling duration (s).")
    ap.add_argument("--torque-limit", type=float, default=14.0, help="Gate C: MIT loop torque limit (Nm).")
    ap.add_argument("--impedance-torque-limit", type=float, default=6.0, help="Gate C: impedance clip (Nm).")
    args = ap.parse_args()

    if args.gate in ("A", "B") and not args.execute:
        print(f"Gate {args.gate} requires the real arm: add --execute. Aborting.")
        return 1

    if args.gate == "A":
        return gate_a_joint7_probe(args)
    if args.gate == "B":
        return gate_b_velocity_noise(args)
    return gate_c_gravity_budget(args)


if __name__ == "__main__":
    sys.exit(main())
