#!/usr/bin/env python3
"""Phase A5: dry-run math validation for Cartesian impedance (no hardware).

Verifies the pure-function control law in ``control_backends/cartesian_impedance.py``
against the real Nero Pinocchio model:

1. e=0 -> tau=0 (no spurious torque at equilibrium)
2. 5cm offset -> tau points from tcp_now toward target_tcp
3. kp_joint in [0, cap=3]
4. orientation re-expression: e_ori_world == R_tcp @ e_ori_body produces
   a tau that rotates TCP toward R_d (not away)
5. wrench magnitude scales with stiffness

Run (in the agilexrobot conda env):

    python3 scripts/test_impedance_math_dryrun.py

This is a HARD gate before any powered impedance test (IMPEDANCE_CONTROL_PLAN.md
validation step 1 & 2). No CAN, no arm motion.
"""

from __future__ import annotations

import os
import sys

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(SCRIPT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.nero_model import NeroPinocchioModel  # noqa: E402
from control_backends.cartesian_impedance import (  # noqa: E402
    DEFAULT_CARTESIAN_KP,
    default_cartesian_kd,
    impedance_wrench,
    task_torque,
    project_joint_kp,
    clip_torque,
)


def _rotation_error_axis_angle(R_desired: np.ndarray, R_current: np.ndarray) -> np.ndarray:
    """Body-frame orientation error, matching pbvs_pose_servo (R_current @ rotvec(R_current^T R_desired))."""
    from scipy.spatial.transform import Rotation as Rot
    R_err = R_current.T @ R_desired
    return R_current @ Rot.from_matrix(R_err).as_rotvec()


def main() -> int:
    np.set_printoptions(precision=4, suppress=True)
    model = NeroPinocchioModel()

    # A representative nominal configuration (near the validated ArUco start pose).
    q = np.radians(np.array([4.0, 52.0, 3.0, 90.0, 0.0, 9.0, -46.0]))
    q = model.clamp_to_limits(q)

    tcp_now, R_tcp = model.fk(q)
    J = model.jacobian(q)
    qdot_m = np.zeros(7)  # arm stationary for the math test

    Kx = DEFAULT_CARTESIAN_KP.copy()
    Dx = default_cartesian_kd(Kx)
    print(f"Kx = {Kx}")
    print(f"Dx = {Dx}  (2*sqrt(Kx))")
    print(f"tcp_now = {tcp_now}")
    print(f"J shape = {J.shape}")

    failures = []

    # --- Test 1: zero error -> zero torque ---
    F0 = impedance_wrench(np.zeros(3), np.zeros(3), J, qdot_m, Kx, Dx)
    tau0 = task_torque(J, F0)
    n0 = float(np.max(np.abs(tau0)))
    print(f"\n[Test 1] zero-error tau max|.| = {n0:.2e} (expect ~0)")
    if n0 > 1e-9:
        failures.append(f"zero-error tau not zero: {n0}")

    # --- Test 2: 5cm offset along +x -> tau produces motion toward target ---
    offset = np.array([0.05, 0.0, 0.0])
    e_pos = offset  # target - current, world frame
    F = impedance_wrench(e_pos, np.zeros(3), J, qdot_m, Kx, Dx)
    tau = task_torque(J, F)
    # Resulting Cartesian motion direction from this joint torque: x_dot = J @ qdot,
    # with qdot = M^-1 tau approx -> check J @ tau sign matches e_pos direction.
    induced = J[:3, :] @ tau  # proxy: linear part of J times torque
    cos = float(np.dot(induced, e_pos) / (np.linalg.norm(induced) * np.linalg.norm(e_pos) + 1e-12))
    print(f"\n[Test 2] 5cm +x offset")
    print(f"  tau = {tau}")
    print(f"  J[:3]@tau (induced dir) = {induced}")
    print(f"  cos(induced, e_pos) = {cos:.3f} (expect > 0, i.e. toward target)")
    if cos <= 0.0:
        failures.append(f"tau does not pull toward target: cos={cos}")

    # --- Test 3: projected kp in [0, cap] ---
    kp_joint = project_joint_kp(J, Kx, cap=3.0)
    print(f"\n[Test 3] kp_joint = {kp_joint}")
    if np.any(kp_joint < 0) or np.any(kp_joint > 3.0 + 1e-9):
        failures.append(f"kp_joint out of [0,3]: {kp_joint}")

    # --- Test 4: orientation re-expression produces corrective rotation ---
    # Pure rotation target: rotate R_tcp by +20deg about world z.
    from scipy.spatial.transform import Rotation as Rot
    R_delta = Rot.from_rotvec(np.deg2rad(20.0) * np.array([0.0, 0.0, 1.0])).as_matrix()
    R_des = R_delta @ R_tcp
    e_ori_body = _rotation_error_axis_angle(R_des, R_tcp)
    e_ori_world = R_tcp @ e_ori_body  # THE critical re-expression
    F_rot = impedance_wrench(np.zeros(3), e_ori_world, J, qdot_m, Kx, Dx)
    tau_rot = task_torque(J, F_rot)
    induced_rot = J[3:, :] @ tau_rot  # angular part proxy
    cos_rot = float(np.dot(induced_rot, e_ori_world) /
                    (np.linalg.norm(induced_rot) * np.linalg.norm(e_ori_world) + 1e-12))
    print(f"\n[Test 4] pure +20deg z rotation")
    print(f"  e_ori_body   = {e_ori_body}")
    print(f"  e_ori_world  = {e_ori_world}  (= R_tcp @ e_ori_body)")
    print(f"  tau_rot      = {tau_rot}")
    print(f"  J[3:]@tau_rot induced = {induced_rot}")
    print(f"  cos(induced_rot, e_ori_world) = {cos_rot:.3f} (expect > 0, rotates toward R_d)")
    if cos_rot <= 0.0:
        failures.append(f"rotation torque wrong direction: cos={cos_rot}")
    # Sanity: body-frame error fed WITHOUT re-expression should generally differ
    F_wrong = impedance_wrench(np.zeros(3), e_ori_body, J, qdot_m, Kx, Dx)
    differs = not np.allclose(F_rot, F_wrong)
    print(f"  re-expression changes wrench? {differs} (True confirms body!=world)")

    # --- Test 5: torque scales with stiffness ---
    Kx_high = Kx * 4.0
    F_high = impedance_wrench(e_pos, np.zeros(3), J, qdot_m, Kx_high, default_cartesian_kd(Kx_high))
    ratio = float(np.linalg.norm(task_torque(J, F_high)) / (np.linalg.norm(tau) + 1e-12))
    print(f"\n[Test 5] stiffness x4 -> torque ratio = {ratio:.2f} (expect ~4)")
    if not (3.5 < ratio < 4.5):
        failures.append(f"torque does not scale linearly with stiffness: {ratio}")

    # --- Test 6: clip respects limit ---
    big = np.full(7, 100.0)
    clipped = clip_torque(big, limit=6.0)
    print(f"\n[Test 6] clip(100,...,6) max|.| = {float(np.max(np.abs(clipped))):.2f} (expect 6.0)")
    if float(np.max(np.abs(clipped))) > 6.0 + 1e-9:
        failures.append("clip did not enforce limit")

    print("\n" + "=" * 50)
    if failures:
        print(f"FAILED ({len(failures)}):")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("ALL MATH TESTS PASSED — impedance law is sound.")
    print("Next: run the bench gates (joint-7 probe, velocity noise) before powered tracking.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
