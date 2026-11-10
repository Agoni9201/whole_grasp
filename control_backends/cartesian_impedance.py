"""Cartesian impedance helpers for the Nero arm (MIT mode + gravity comp).

Pure-function module (NOT a backend). Implements the torque-based Cartesian
impedance control law, mapped onto the Nero MIT ``move_mit`` interface:

    Cartesian wrench   F = K_x · e  -  D_x · (J · qdot_m)        (6D)
    Task torque        tau_task = J^T · F                         (7D)
    Projected joint kp kp_joint  = cap(diag(J^T · diag(K_x) · J), 0, cap)
    MIT command        move_mit(p_des, 0, kp_joint, kd_joint, t_ff)

Gravity compensation is layered automatically by the 200 Hz MIT loop
(``mit_loop.py``), so ``t_ff`` carries ONLY the Cartesian error force, never
gravity.

Reference (ported math, not code): matthias-mayr/Cartesian-Impedance-Controller,
arXiv:2212.11215 (JOSS 2024). Phase A uses the libfranka heuristic damping
``D_x = 2·sqrt(K_x)`` which does not require a joint-space inertia matrix;
Phase B (optional) adds the double-diagonalization critically-damped damping
that needs ``M(q)`` from ``pinocchio.crba``.

See ``IMPEDANCE_CONTROL_PLAN.md`` §3-§6 for the Nero-specific rationale
(why kp_joint is small-nonzero, why heuristic damping is primary, why
``tau_ext`` is absent — no wrist F/T sensor).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import numpy.typing as npt

# Phase A default Cartesian stiffness: [translational (N/m), rotational (Nm/rad)].
# Conservative start for a ~0.6 m reach arm; tune up after push-compliance check.
DEFAULT_CARTESIAN_KP = np.array([200.0, 200.0, 200.0, 15.0, 15.0, 15.0], dtype=np.float64)


def default_cartesian_kd(kp: npt.ArrayLike = DEFAULT_CARTESIAN_KP) -> npt.NDArray[np.float64]:
    """libfranka heuristic damping: D_x = 2·sqrt(K_x).

    Does not require a joint-space inertia matrix; stable and slightly
    conservative. Switch to ``critically_damped_damping`` (Phase B) only if
    oscillation is observed AND ``M(q)`` is validated.
    """
    kp = np.asarray(kp, dtype=np.float64).ravel()
    return 2.0 * np.sqrt(np.maximum(kp, 0.0))


def impedance_wrench(
    e_pos_world: npt.ArrayLike,
    e_ori_world: npt.ArrayLike,
    jacobian: npt.NDArray[np.float64],
    qdot_measured: npt.ArrayLike,
    kx: npt.ArrayLike = DEFAULT_CARTESIAN_KP,
    dx: Optional[npt.ArrayLike] = None,
    target_twist: Optional[npt.ArrayLike] = None,
) -> npt.NDArray[np.float64]:
    """6D Cartesian wrench with optional desired TCP twist feed-forward.

    With ``target_twist=None`` this is the legacy law
    ``F = K_x·e - D_x·(J·qdot_m)``. When a desired twist is supplied, the
    damping term becomes ``D_x·(target_twist - J·qdot_m)``, preserving the
    same damper while providing smooth motion feed-forward for a moving
    visual target.

    Parameters
    ----------
    e_pos_world : (3,) translational error in WORLD frame (target - current).
    e_ori_world : (3,) orientation error re-expressed in WORLD frame.
                  Caller MUST supply ``R_tcp @ servo_out["orientation_error"]``
                  because the PBVS orientation error is body-frame while the
                  Jacobian is LOCAL_WORLD_ALIGNED.
    jacobian : (6, 7) TCP Jacobian, LOCAL_WORLD_ALIGNED.
    qdot_measured : (7,) filtered measured joint velocities (rad/s).
    kx : (6,) diagonal Cartesian stiffness.
    dx : (6,) diagonal Cartesian damping. Defaults to ``2·sqrt(kx)``.
    target_twist : optional (6,) desired TCP twist in the same world-aligned
                   frame as ``jacobian``. ``None`` preserves the legacy
                   zero-velocity target.
    """
    e_pos = np.asarray(e_pos_world, dtype=np.float64).ravel()
    e_ori = np.asarray(e_ori_world, dtype=np.float64).ravel()
    J = np.asarray(jacobian, dtype=np.float64)
    qdot_m = np.asarray(qdot_measured, dtype=np.float64).ravel()
    Kx = np.diag(np.asarray(kx, dtype=np.float64).ravel())
    if dx is None:
        Dx = np.diag(default_cartesian_kd(kx))
    else:
        Dx = np.diag(np.asarray(dx, dtype=np.float64).ravel())
    e = np.concatenate([e_pos, e_ori])
    x_dot = J @ qdot_m
    if target_twist is None:
        # Keep the historical expression exactly for callers that do not opt
        # into moving-target feed-forward.
        return Kx @ e - Dx @ x_dot
    desired_x_dot = np.asarray(target_twist, dtype=np.float64).ravel()
    if desired_x_dot.shape != (6,):
        raise ValueError(
            f"target_twist must have shape (6,), got {desired_x_dot.shape}"
        )
    if not np.all(np.isfinite(desired_x_dot)):
        raise ValueError("target_twist must contain only finite values")
    return Kx @ e + Dx @ (desired_x_dot - x_dot)


def task_torque(
    jacobian: npt.NDArray[np.float64],
    wrench: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Map Cartesian wrench to joint torques via Jacobian TRANSPOSE (not inverse).

    ``tau_task = J^T · F``. This is the dual of the velocity mapping
    ``x_dot = J · qdot`` and is the correct force->torque transform.

    NOTE: physically exact for a static Cartesian spring, but near singularities
    J^T can drive large joint torques from small Cartesian forces. Prefer
    :func:`dls_task_torque` (DLS map) for the author-validated Panthera law, which
    avoids the J^T@B@J velocity-noise amplification that causes wrist jitter.
    """
    J = np.asarray(jacobian, dtype=np.float64)
    F = np.asarray(wrench, dtype=np.float64).ravel()
    return J.T @ F


def dls_task_torque(
    jacobian: npt.NDArray[np.float64],
    wrench: npt.NDArray[np.float64],
    dls_lambda: float = 0.05,
) -> npt.NDArray[np.float64]:
    """Map Cartesian wrench to joint torques via the DLS (damped-least-squares) map.

    ``tau = J^T (J J^T + lambda^2 I)^{-1} F``.

    This is the force map used by the author-validated Panthera Python port
    (7_keyboard_ee_control_cart_impedance.py). It down-weights the wrench near
    singularities and, critically, avoids the noise amplification that plain
    ``J^T @ F`` suffers when F contains a Cartesian-damping term ``-B @ x_dot``
    built from noisy finite-difference velocity (J^T @ B @ J amplification).
    """
    J = np.asarray(jacobian, dtype=np.float64)
    F = np.asarray(wrench, dtype=np.float64).ravel()
    n = J.shape[0]
    JJt = J @ J.T
    damped = JJt + (float(dls_lambda) ** 2) * np.eye(n)
    alpha = np.linalg.solve(damped, F)
    return J.T @ alpha


def joint_damping_torque(
    qdot_measured: npt.ArrayLike,
    joint_damping: npt.ArrayLike,
) -> npt.NDArray[np.float64]:
    """Joint-space (primary) damping: ``tau_damp = -diag(joint_damping) @ qdot``.

    Applied DIRECTLY in joint space, so it does NOT pass through J^T and therefore
    does not amplify finite-difference velocity noise. This is the PRIMARY damping
    on Nero (where Cartesian velocity is only available via noisy finite-diff);
    keep the Cartesian ``B`` tiny (~0.01) per the Panthera author's guidance.
    """
    qd = np.asarray(qdot_measured, dtype=np.float64).ravel()
    jd = np.asarray(joint_damping, dtype=np.float64).ravel()
    return -jd * qd


# Static-friction (Coulomb) deadband: below this joint error the compensation
# turns off, matching the standalone tuning harness (avoids limit-cycle chatter
# at the rest pose). 0.5 deg.
COULOMB_DEADBAND_RAD = float(np.deg2rad(0.5))


def coulomb_compensation(
    q_rest: npt.ArrayLike,
    q_now: npt.ArrayLike,
    coulomb: npt.ArrayLike,
    deadband_rad: float = COULOMB_DEADBAND_RAD,
    smooth_width_rad: float = 0.0,
) -> npt.NDArray[np.float64]:
    """Per-joint Coulomb friction feed-forward.

    With ``smooth_width_rad <= 0`` this retains the standalone hard-sign law
    ``tau_c = coulomb * sign(q_rest - q)``.  A positive width replaces the
    discontinuous sign with ``tanh(error / width)``.  The smooth form is useful
    for visual tracking: the same friction assist remains valid while the
    reference moves, without injecting a kick whenever the moving reference
    crosses the measured joint position.

    Breaks static friction so the Cartesian spring can actually settle the last
    few mm/deg (Nero's joint2/joint4 have noticeable stiction). A small deadband
    (default 0.5 deg) disables the term once the joint is essentially at rest,
    avoiding limit-cycle chatter at the target. Joints with ``coulomb <= 0`` are
    skipped (zero compensation).

    Ported verbatim (semantics) from
    ``scripts/test_official_cartesian_impedance.py:coulomb_compensation`` so the
    in-loop law and the standalone tuning harness share ONE definition.

    ``smooth_width_rad`` is deliberately opt-in so the validated standalone
    fixed-target behavior remains unchanged for existing callers.
    """
    err = np.asarray(q_rest, dtype=np.float64).ravel() - np.asarray(q_now, dtype=np.float64).ravel()
    c = np.asarray(coulomb, dtype=np.float64).ravel()
    tau = np.zeros_like(c)
    width = float(smooth_width_rad)
    if width > 0.0:
        # Keep a true zero at the target while making the transition continuous.
        active = c > 0.0
        tau[active] = c[active] * np.tanh(err[active] / width)
    else:
        active = (c > 0.0) & (np.abs(err) >= float(deadband_rad))
        tau[active] = c[active] * np.sign(err[active])
    return tau


def project_joint_kp(
    jacobian: npt.NDArray[np.float64],
    kx: npt.ArrayLike = DEFAULT_CARTESIAN_KP,
    cap: float = 3.0,
) -> npt.NDArray[np.float64]:
    """Project Cartesian stiffness onto joint space: diag(J^T · diag(K_x) · J).

    Capped per-joint to ``cap`` (default 3.0) so the resulting joint spring is a
    WEAK anti-drift safety spring that does not compete with the Cartesian law
    (which lives in ``t_ff``). Keeping it small-but-NONZERO is deliberate for the
    Nero arm: pure-torque (kp=0) impedance stalls in joint friction deadbands,
    so a weak spring helps break static friction on small errors.
    """
    J = np.asarray(jacobian, dtype=np.float64)
    Kx_diag = np.asarray(kx, dtype=np.float64).ravel()
    Kq = J.T @ np.diag(Kx_diag) @ J
    kp_joint = np.clip(np.diag(Kq), 0.0, cap)
    return kp_joint


def clip_torque(
    tau: npt.ArrayLike,
    limit: float,
    per_joint_limit: Optional[npt.ArrayLike] = None,
) -> npt.NDArray[np.float64]:
    """Clip joint torque feed-forward.

    A symmetric global clip ``±limit`` protects the gravity-compensation budget
    (gravity is added AFTER this clip inside the MIT loop). Optionally a tighter
    per-joint ceiling (e.g. wrist joints 5-7 on Nero have lower torque ceilings).
    """
    tau = np.asarray(tau, dtype=np.float64).ravel().copy()
    tau = np.clip(tau, -float(limit), float(limit))
    if per_joint_limit is not None:
        pj = np.asarray(per_joint_limit, dtype=np.float64).ravel()
        tau = np.clip(tau, -pj, pj)
    return tau


def null_space_torque(
    q_now: npt.ArrayLike,
    qdot_measured: npt.ArrayLike,
    jacobian: npt.NDArray[np.float64],
    q_rest: npt.ArrayLike,
    kq: float = 10.0,
    dq: float = 0.3,
    jacobian_pinv: Optional[npt.NDArray[np.float64]] = None,
    damping: float = 1e-3,
) -> npt.NDArray[np.float64]:
    """Null-space joint impedance for the 7-DOF redundant elbow (Phase B).

    ``tau_null = N^T · (K_q·(q_rest - q) - D_q·qdot)``, with the simple
    (kinematic) null-space projector ``N = I - J_pinv·J``. Projects to zero
    wrench at the TCP, so it postures the elbow without disturbing compliant
    tracking. Keep ``kq`` WEAK relative to the Cartesian task.

    For a dynamically-consistent projector use ``J_bar = M^-1 J^T (J M^-1 J^T)^-1``
    instead (needs ``M(q)``, Phase B).
    """
    q = np.asarray(q_now, dtype=np.float64).ravel()
    qd = np.asarray(qdot_measured, dtype=np.float64).ravel()
    J = np.asarray(jacobian, dtype=np.float64)
    n = J.shape[1]
    if jacobian_pinv is None:
        J_pinv = _damped_pinv(J, damping)
    else:
        J_pinv = np.asarray(jacobian_pinv, dtype=np.float64)
    N = np.eye(n) - J_pinv @ J
    posture = float(kq) * (np.asarray(q_rest, dtype=np.float64).ravel() - q) - float(dq) * qd
    return N.T @ posture


def _damped_pinv(J: npt.NDArray[np.float64], damping: float) -> npt.NDArray[np.float64]:
    """Damped (right) pseudo-inverse J^T(JJ^T + λ²I)^-1."""
    JJt = J @ J.T
    damped = JJt + (damping ** 2) * np.eye(J.shape[0])
    return J.T @ np.linalg.inv(damped)


# ---------------------------------------------------------------------------
# Phase B (optional) — rigorous damping via joint-space inertia matrix M(q).
# Importing scipy lazily so Phase A users don't need scipy installed.
# ---------------------------------------------------------------------------

def cartesian_inertia(
    jacobian: npt.NDArray[np.float64],
    mass_matrix: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Cartesian (apparent) inertia Lambda_x = (J · M^-1 · J^T)^-1  (6x6).

    Requires the 7x7 joint-space inertia matrix ``M(q)`` from
    ``nero_model.NeroPinocchioModel.mass_matrix`` (``pinocchio.crba``).
    Configuration-dependent; recompute each cycle.
    """
    J = np.asarray(jacobian, dtype=np.float64)
    M = np.asarray(mass_matrix, dtype=np.float64)
    M_inv = np.linalg.inv(M)
    Lambda_inv = J @ M_inv @ J.T
    return np.linalg.inv(Lambda_inv)


def critically_damped_damping(
    kx: npt.ArrayLike,
    lambda_x: npt.NDArray[np.float64],
    xi: float = 0.9,
) -> npt.NDArray[np.float64]:
    """Double-diagonalization damping giving damping ratio ``xi`` in all 6 modes.

    Solves the generalized eigenproblem ``K_x v = w Lambda_x v`` (so
    ``V^T Lambda_x V = I`` and ``V^T K_x V = diag(w)``), then forms
    ``D = Lambda V diag(2·xi·sqrt(w)) V^T Lambda``.

    Requires ``scipy``. Ported from Cartesian-Impedance-Controller
    ``damping_methods``. Only use after validating ``M(q)`` against the real
    arm; if URDF inertial params are off, fall back to ``default_cartesian_kd``.
    """
    from scipy.linalg import eigh  # lazy import

    Kx = np.diag(np.asarray(kx, dtype=np.float64).ravel())
    Lambda = np.asarray(lambda_x, dtype=np.float64)
    w, V = eigh(Kx, Lambda)  # V^T Lambda V = I,  V^T Kx V = diag(w)
    w = np.clip(w, 0.0, None)
    D = Lambda @ V @ np.diag(2.0 * xi * np.sqrt(w)) @ V.T @ Lambda
    return 0.5 * (D + D.T)  # enforce symmetry against numerical drift
