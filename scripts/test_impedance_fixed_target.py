#!/usr/bin/env python3
"""Fixed-target Cartesian impedance regression — NO camera, NO vision, NO ROS.

Runs a passivity-preserving production baseline derived from
``test_official_cartesian_impedance.py`` inside the 200 Hz MIT background loop:

    F        = K @ e - B @ (J @ dq_real)
    tau_task = J.T @ F
    tau      = gravity + clip(tau_task, +/-10 Nm) + coulomb

The script sets the target and gains once, then only monitors state and safety.
Firmware joint damping defaults ON so the seventh (task-null-space) direction
is damped. Non-uniform legacy joint weights remain available only as an
explicit diagnostic override.

    # Stage 1: verify mode/velocity/null-space damping with no Cartesian wrench
    python3 scripts/test_impedance_fixed_target.py --execute --duration 10 \
        --cartesian-kp 0 0 0 0 0 0 --cartesian-kd 0 0 0 0 0 0 \
        --task-torque-limit 5 --max-excursion 0.15 --safe-margin 0.30

Do not run the full validated K/B profile until this zero-wrench stage passes.
Offset and legacy DLS runs are later diagnostic stages, not first-run commands.
Keep the hardware e-stop in hand.
"""

from __future__ import annotations

import argparse
import atexit
import os
import sys
import time

import numpy as np
from scipy.spatial.transform import Rotation as Rot

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(SCRIPT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from control_backends.base import BackendCommand  # noqa: E402
from control_backends.force_ctrl.controller import ZERO_GRAV_KD  # noqa: E402
from control_backends.force_ctrl.data_types import ImpedanceConfig  # noqa: E402
from control_backends.force_ctrl.impedance_profile import (  # noqa: E402
    VALIDATED_CARTESIAN_KD,
    VALIDATED_CARTESIAN_KP,
    VALIDATED_GRAVITY_FACTORS,
    VALIDATED_TASK_TORQUE_LIMIT,
    VALIDATED_TOTAL_TORQUE_LIMIT,
    VALIDATED_VELOCITY_SIGNS,
)
from control_backends.sdk_force_ctrl_backend import SdkForceCtrlBackend  # noqa: E402
from models.nero_model import NeroPinocchioModel  # noqa: E402


_ACTIVE_BACKEND_CLEANUP = None
_CLEANUP_HAD_ERROR = False


def _joint_vector(value, context: str) -> np.ndarray:
    """Return one finite seven-joint vector or fail before using it."""
    q = np.asarray(value, dtype=np.float64).ravel()
    if q.shape != (7,) or not np.all(np.isfinite(q)):
        raise RuntimeError(
            f"{context}: expected finite joint vector shape (7,), "
            f"got shape={q.shape}, values={q.tolist()}")
    return q


def _orientation_error_world(R_desired: np.ndarray, R_current: np.ndarray) -> np.ndarray:
    """World-frame orientation error via the log-map (validated C++ reference):

        e_ori_world = R_current @ rotvec(R_current^T @ R_desired)

    Single rotation to world frame. (The old script double-rotated; that was
    masked by tiny K_ori. The validated Panthera law uses ONE rotation.)
    """
    return R_current @ Rot.from_matrix(R_current.T @ R_desired).as_rotvec()


def _wait_joint_angles(backend: SdkForceCtrlBackend, timeout: float = 5.0):
    deadline = time.monotonic() + timeout
    q = backend.get_joint_angles()
    while q is None and time.monotonic() < deadline:
        time.sleep(0.05)
        q = backend.get_joint_angles()
    return None if q is None else _joint_vector(q, "joint feedback")


def _resolve_engagement_target(
    target_xyz,
    offset_xyz,
    rotz_deg,
    tcp_engage,
    R_engage,
    target_rpy_deg=None,
    target_quaternion=None,
):
    """Resolve a base-frame TCP target from the post-settle pose.

    ``target_xyz`` is an absolute base-frame position.  Orientation is
    absolute when ``target_rpy_deg`` or ``target_quaternion`` is supplied;
    otherwise ``rotz_deg`` preserves the legacy relative-yaw behavior.
    RPY uses fixed base axes in roll/pitch/yaw order.  Quaternions use the
    SciPy/PyAgxArm order ``x, y, z, w`` and are normalized before use.
    """
    if target_xyz is not None:
        target_tcp = np.asarray(target_xyz, dtype=np.float64)
    else:
        target_tcp = (
            np.asarray(tcp_engage, dtype=np.float64)
            + np.asarray(offset_xyz, dtype=np.float64)
        )
    if target_tcp.shape != (3,) or not np.all(np.isfinite(target_tcp)):
        raise ValueError(
            "TCP target position must contain three finite base-frame values")

    if target_rpy_deg is not None and target_quaternion is not None:
        raise ValueError(
            "--target-rpy-deg and --target-quaternion are mutually exclusive")
    if target_rpy_deg is not None:
        rpy = np.asarray(target_rpy_deg, dtype=np.float64).ravel()
        if rpy.shape != (3,) or not np.all(np.isfinite(rpy)):
            raise ValueError(
                "target RPY must contain three finite values in degrees")
        # Lower-case xyz denotes extrinsic rotations about fixed base axes.
        target_rotation = Rot.from_euler("xyz", rpy, degrees=True).as_matrix()
    elif target_quaternion is not None:
        quat = np.asarray(target_quaternion, dtype=np.float64).ravel()
        if quat.shape != (4,) or not np.all(np.isfinite(quat)):
            raise ValueError(
                "target quaternion must contain four finite x y z w values")
        quat_norm = float(np.linalg.norm(quat))
        if quat_norm <= 1e-12:
            raise ValueError("target quaternion must have non-zero norm")
        target_rotation = Rot.from_quat(quat / quat_norm).as_matrix()
    else:
        # ``None`` is useful for the CLI so a full absolute orientation can be
        # distinguished from the legacy relative-yaw mode.
        relative_yaw_deg = 0.0 if rotz_deg is None else float(rotz_deg)
        if not np.isfinite(relative_yaw_deg):
            raise ValueError("relative target yaw must be finite")
        R_offset = Rot.from_rotvec(
            np.deg2rad(relative_yaw_deg) * np.array([0.0, 0.0, 1.0])
        ).as_matrix()
        target_rotation = R_offset @ np.asarray(R_engage, dtype=np.float64)

    return target_tcp, target_rotation


def _build_impedance_config(args, target_tcp, R_target, q_rest):
    """Build the ImpedanceConfig handed to the 200 Hz loop (set ONCE)."""
    K = np.asarray(args.cartesian_kp, dtype=np.float64)
    B = np.asarray(args.cartesian_kd, dtype=np.float64)
    joint_damping = np.asarray(args.joint_damping, dtype=np.float64)
    tau_limit = np.asarray(args.tau_limit_per_joint, dtype=np.float64)
    kd_joint = np.asarray(ZERO_GRAV_KD, dtype=np.float64) * float(args.kd_scale)
    velocity_signs = np.asarray(
        [float(x) for x in args.velocity_signs.replace(",", " ").split()],
        dtype=np.float64)
    if velocity_signs.shape[0] != 7:
        raise ValueError(f"--velocity-signs needs 7 values, got {velocity_signs.shape[0]}")
    return ImpedanceConfig(
        active=True,
        target_tcp=np.asarray(target_tcp, dtype=np.float64),
        R_target=np.asarray(R_target, dtype=np.float64),
        K=K,
        B=B,
        joint_damping=joint_damping,
        tau_limit_per_joint=tau_limit,
        q_rest=np.asarray(q_rest, dtype=np.float64),
        null_kq=float(args.nullspace_kq),
        dls_lambda=float(args.dls_lambda),
        use_dls_mapping=(args.law == "panthera-real"),
        kp_joint=np.zeros(7),
        kd_joint=kd_joint,
        joint_hold_kp=np.asarray(args.joint_hold_kp, dtype=np.float64),
        vel_filter_tau=float(args.vel_filter_tau),
        qdot_trip_fast=float(args.qdot_trip_fast),
        use_real_velocity=bool(args.use_real_velocity),
        velocity_signs=velocity_signs,
        real_vel_filter_tau=float(args.real_vel_filter_tau),
        joint_weights=np.asarray(args.joint_weights, dtype=np.float64),
        task_torque_limit=float(args.task_torque_limit),
        coulomb=np.asarray(args.coulomb, dtype=np.float64),
    )


def _print_inner_trip_debug(debug: dict) -> None:
    """Print the frozen 200 Hz pre-trip window without writing from the RT loop."""
    trip = (debug or {}).get("impedance_trip")
    trace = (
        (trip or {}).get("trace")
        or (debug or {}).get("impedance_trace")
        or []
    )
    if not trip and not trace:
        return
    print("\n--- IMPEDANCE DIAGNOSTIC WINDOW ---")
    print(
        f"kind={(trip or {}).get('trigger_kind', 'slow-monitor')} "
        f"joint={(trip or {}).get('trigger_joint', 'outer')} "
        f"samples={len(trace)}")
    if not trace:
        return
    last = trace[-1]
    raw = np.stack([row["qdot_real_raw"] for row in trace])
    selected = np.stack([row["qdot_selected"] for row in trace])
    fdiff = np.stack([row["qdot_fdiff"] for row in trace])
    total = np.stack([row["tau_total"] for row in trace])
    dts = np.asarray([row["loop_dt"] for row in trace], dtype=np.float64)
    q_delta = np.asarray(trace[-1]["q"]) - np.asarray(trace[0]["q"])
    print(f"source={last['source']} fresh={last['fresh_mask'].astype(int).tolist()}")
    print(f"last selected qdot={np.round(last['qdot_selected'], 3).tolist()}")
    print(f"last raw qdot     ={np.round(last['qdot_real_raw'], 3).tolist()}")
    print(f"last fdiff qdot   ={np.round(last['qdot_fdiff'], 3).tolist()}")
    print(f"fresh-frame counts={last['over_count'].astype(int).tolist()}")
    print(f"window max |raw|  ={np.round(np.max(np.abs(raw), axis=0), 3).tolist()}")
    print(f"window max |sel|  ={np.round(np.max(np.abs(selected), axis=0), 3).tolist()}")
    print(f"window max |fdiff|={np.round(np.max(np.abs(fdiff), axis=0), 3).tolist()}")
    print(f"window q delta deg={np.round(np.degrees(q_delta), 2).tolist()}")
    print(f"window max |tau|  ={np.round(np.max(np.abs(total), axis=0), 2).tolist()}")
    print(
        f"loop dt min/max={1000.0*np.min(dts):.2f}/"
        f"{1000.0*np.max(dts):.2f}ms")


def _main_impl() -> int:
    global _ACTIVE_BACKEND_CLEANUP, _CLEANUP_HAD_ERROR
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--execute", action="store_true", help="Required: enables the real arm.")
    ap.add_argument("--channel", default="can0")
    ap.add_argument("--firmware", default="v111")
    ap.add_argument("--duration", type=float, default=30.0, help="Seconds of impedance tracking; 0 = forever.")
    ap.add_argument("--servo-hz", type=float, default=30.0, help="MONITOR rate (impedance runs at the 200 Hz loop).")
    ap.add_argument(
        "--law", choices=["panthera-real", "plain-jtf"], default="plain-jtf",
        help="plain-jtf = validated plain J^T@F mapping (default). "
             "panthera-real = legacy DLS force map for diagnostic A/B.")
    ap.add_argument(
        "--target-xyz", nargs=3, type=float, default=None, metavar=("X", "Y", "Z"),
        help="Absolute base-frame TCP target (m). Default: current TCP at start.")
    ap.add_argument(
        "--offset-xyz", nargs=3, type=float, default=[0.0, 0.0, 0.0], metavar=("DX", "DY", "DZ"),
        help="TCP target offset from the start pose (m). Ignored if --target-xyz given.")
    orientation = ap.add_mutually_exclusive_group()
    orientation.add_argument(
        "--target-rpy-deg", nargs=3, type=float, metavar=("ROLL", "PITCH", "YAW"),
        help="Absolute base-frame TCP orientation as fixed-axis roll, pitch, yaw (deg).")
    orientation.add_argument(
        "--target-quaternion", "--target-quat", dest="target_quaternion",
        nargs=4, type=float, metavar=("X", "Y", "Z", "W"),
        help="Absolute base-frame TCP orientation as quaternion x y z w.")
    orientation.add_argument(
        "--rotz-deg", type=float, default=None,
        help="Legacy target yaw about base z relative to start orientation (deg). "
             "Omit it to hold current orientation.")
    # ---- impedance gains ----
    ap.add_argument("--cartesian-kp", type=float, nargs=6,
                    default=[0.0] * 6,
                    metavar=("TX", "TY", "TZ", "RX", "RY", "RZ"),
                    help="Cartesian stiffness [transl N/m, rot Nm/rad]. "
                         "Safe first-run default is zero; validated later-stage "
                         f"profile is {VALIDATED_CARTESIAN_KP.tolist()}.")
    ap.add_argument("--cartesian-kd", type=float, nargs=6,
                    default=[0.0] * 6,
                    metavar=("TX", "TY", "TZ", "RX", "RY", "RZ"),
                    help="Cartesian damping B [transl N·s/m, rot N·m·s/rad]. "
                         "Safe first-run default is zero; validated later-stage "
                         f"profile is {VALIDATED_CARTESIAN_KD.tolist()}.")
    ap.add_argument("--joint-weights", type=float, nargs=7,
                    default=[1.0] * 7,
                    help="Per-joint scaling applied to J^T@F before task clipping. "
                         "Safe production default is all ones; non-uniform values "
                         "break the standard Cartesian passivity argument.")
    ap.add_argument("--task-torque-limit", type=float,
                    default=VALIDATED_TASK_TORQUE_LIMIT,
                    help="Symmetric task-torque clip before gravity/Coulomb (Nm).")
    ap.add_argument("--coulomb", type=float, nargs=7,
                    default=[0.0] * 7,
                    help="Per-joint Coulomb feed-forward (Nm), based on q_rest-q. "
                         "Safe baseline is off; legacy standalone values can be "
                         "passed explicitly after the spring/damping stages pass.")
    ap.add_argument("--joint-damping", type=float, nargs=7, default=[0.0] * 7,
                    help="Software joint-space damping (N·m·s/rad), finite-diff velocity. DEFAULT 0: "
                         "on Nero the finite-diff velocity is too laggy/noisy for the wrist and any "
                         "nonzero value can turn anti-damping (the joint7 runaway). Primary damping "
                         "is --kd-scale (firmware kd, clean velocity). Re-enable only if clean velocity "
                         "becomes available.")
    ap.add_argument("--kd-scale", type=float, default=1.0,
                    help="Multiplier on firmware joint kd (ZERO_GRAV_KD) -> joint-space damping "
                         "(damps task + null-space). Default 1 is the safe production baseline; "
                         "0 reproduces the legacy pure-torque experiment.")
    ap.add_argument("--dls-lambda", type=float, default=0.05, help="DLS regularization (panthera-real law).")
    ap.add_argument("--nullspace-kq", type=float, default=0.0,
                    help="7-DOF null-space posture stiffness (Nm/rad). 0 = pure (default, matches reference).")
    ap.add_argument("--joint-hold-kp", type=float, nargs=7,
                    default=[0.0] * 7,
                    help="Firmware position-hold kp on selected joints (Nm/rad), 7 values. "
                         "Default all-zero reproduces the validated pure-torque law.")
    # ---- limits / trips ----
    ap.add_argument("--tau-limit-per-joint", type=float, nargs=7,
                    default=[VALIDATED_TOTAL_TORQUE_LIMIT] * 7,
                    help="Per-joint total torque clip (also clipped to driver +-16).")
    ap.add_argument("--torque-limit", type=float,
                    default=VALIDATED_TOTAL_TORQUE_LIMIT,
                    help="Total torque clip (after gravity).")
    ap.add_argument("--joint-factors", type=float, nargs=7,
                    default=VALIDATED_GRAVITY_FACTORS.tolist(),
                    help="Per-joint gravity-comp factors (joint7 1.5x compensates wrist camera).")
    ap.add_argument("--vel-filter-tau", type=float, default=0.008,
                    help="Finite-diff velocity low-pass tau (s); used for "
                         "diagnostics/safety and as the impedance velocity "
                         "only when --no-use-real-velocity is selected.")
    ap.add_argument(
        "--velocity-signs",
        default=",".join(str(v) for v in VALIDATED_VELOCITY_SIGNS),
        help="Per-joint sign (7 comma-sep) for the REAL motor velocity. The SDK zeros "
             "get_motor_states().velocity (a TODO bug); the loop reads the parser value "
             "directly, which is LIVE but sign-flipped on joints 1-5,7 (joint6 correct) "
             "-- same bug family as move_mit's `if j!=6: v_des*=-1`. Calibrated by "
             "scripts/probe_motor_velocity_raw.py (sign_agree joint6=100%%, rest=0%%).")
    ap.add_argument("--use-real-velocity", dest="use_real_velocity",
                    action=argparse.BooleanOptionalAction, default=True,
                    help="Use the REAL motor velocity (parser-direct) for the Cartesian damper "
                         "-B*J*qdot (default ON, the gold key). --no-use-real-velocity falls back to "
                         "finite-diff (B must then stay tiny, divergence risk at the wrist). "
                         "If the real source disappears while selected, Cartesian B is zeroed "
                         "briefly and the source watchdog stops the run; it does not silently switch.")
    ap.add_argument("--real-vel-filter-tau", type=float, default=0.0,
                    help="Low-pass tau (s) on the real velocity. Default 0 matches "
                         "the standalone path and avoids extra damping phase lag.")
    ap.add_argument("--qdot-trip-fast", type=float, default=2.0,
                    help="In-loop divergence trip (rad/s); requires 5 fresh feedback "
                         "frames over threshold before firing, "
                         "so a compliant push-bounce (joint7 can spike 2-3 rad/s briefly) "
                         "or a single finite-diff noise spike does NOT kill the loop. "
                         "Only sustained runaway trips. Real guards: joint-limit e-stop + monitor excursion.")
    ap.add_argument("--settle-s", type=float, default=2.0, help="Zero-gravity settle before engaging impedance.")
    ap.add_argument("--max-excursion", type=float, default=0.20, help="Monitor trip if any joint drifts > this (rad) from track start.")
    ap.add_argument("--safe-margin", type=float, default=0.30, help="Monitor trip if any joint within this (rad) of a hard limit.")
    ap.add_argument("--instant-qdot-cap", type=float, default=6.0,
                    help="Monitor (belt-and-suspenders) qdot trip on the published velocity (rad/s). "
                         "Raised to 6.0 (catastrophic-only): a compliant arm moves fast when pushed.")
    ap.add_argument("--log-period", type=float, default=0.5)
    args = ap.parse_args()

    if not args.execute:
        print("This script drives the real arm. Add --execute."); return 1

    joint_factors = np.asarray(args.joint_factors, dtype=np.float64)
    dt = 1.0 / args.servo_hz

    print("=" * 64)
    print("In-loop Cartesian impedance test (200 Hz law, NO camera / NO vision)")
    print("=" * 64)
    print(f"  law            = {args.law}")
    print(f"  K (cart kp)    = {np.asarray(args.cartesian_kp)}  [transl N/m, rot Nm/rad]")
    print(f"  B (cart kd)    = {np.asarray(args.cartesian_kd)}  [transl N·s/m, rot N·m·s/rad]")
    vel_src = "REAL motor vel (parser-direct)" if args.use_real_velocity else "finite-diff (B must stay tiny!)"
    print(f"  velocity src   = {vel_src}")
    print(f"  velocity_signs = {np.asarray([float(x) for x in args.velocity_signs.replace(',',' ').split()])}")
    print(f"  kd_scale       = {args.kd_scale}  -> firmware kd = {np.round(np.asarray(ZERO_GRAV_KD)*args.kd_scale,2)}  (joint-space damping, clean firmware velocity)")
    print(f"  joint_damping  = {np.asarray(args.joint_damping)}  (software finite-diff; 0 on Nero)")
    print(f"  dls_lambda     = {args.dls_lambda}   use_dls = {args.law == 'panthera-real'}")
    print(f"  null_kq        = {args.nullspace_kq}   vel_filter_tau = {args.vel_filter_tau}")
    print(f"  joint_weights  = {np.asarray(args.joint_weights)}")
    print(f"  task torque lim= {args.task_torque_limit}   coulomb = {np.asarray(args.coulomb)}")
    print(f"  tau_limit/joint= {np.asarray(args.tau_limit_per_joint)}   torque_limit = {args.torque_limit}")
    print(f"  joint_factors  = {joint_factors}")
    print(f"  trips: excursion>{args.max_excursion} rad, limit margin<{args.safe_margin} rad, "
          f"monitor qdot>{args.instant_qdot_cap} rad/s, IN-LOOP qdot>{args.qdot_trip_fast} rad/s")
    print(f"  duration = {args.duration}s   settle = {args.settle_s}s")
    print("  Ctrl+C / slow monitor trip -> gravity+damping safe hold + disconnect.")
    print("  MIT-loop hard-limit/sustained-qdot trips remain electronic e-stop.\n")

    # Pinocchio model (shared by the script for FK display AND by the 200 Hz loop
    # for in-loop impedance via the backend's pin_model).
    # Pinocchio Data is mutable.  The 200 Hz control thread must not share an
    # instance with this 30 Hz monitor, or their FK/Jacobian calls can interleave
    # and return a pose/J pair assembled from different joint samples.
    model = NeroPinocchioModel()
    control_model = model.clone()
    backend = SdkForceCtrlBackend(
        dry_run=False, channel=args.channel, firmware=args.firmware,
        control_hz=200.0, torque_limit=args.torque_limit,
        gravity_factor=1.0, gravity_factors=joint_factors,
        zero_gravity=True, gravity_compensation=True,
        startup_joint_margin=args.safe_margin,
        pin_model=control_model,
    )

    print("Connecting MIT backend (zero-gravity)...")
    try:
        backend.connect()
    except Exception as exc:
        # connect() may have partially initialized the controller.
        cleanup_errors = []
        try:
            backend.emergency_stop()
        except Exception as estop_exc:
            cleanup_errors.append(f"emergency stop failed: {estop_exc}")
        try:
            backend.disconnect()
        except Exception as cleanup_exc:
            cleanup_errors.append(f"disconnect failed: {cleanup_exc}")
        if cleanup_errors:
            print(
                "CRITICAL: partial-start cleanup was not confirmed: "
                + "; ".join(cleanup_errors)
            )
        print(f"FAIL: MIT backend did not start safely: {exc}")
        return 2

    cleanup_done = False

    def _cleanup_backend() -> bool:
        nonlocal cleanup_done
        global _ACTIVE_BACKEND_CLEANUP, _CLEANUP_HAD_ERROR
        if cleanup_done:
            return not _CLEANUP_HAD_ERROR
        errors = []
        try:
            backend.disable_cartesian_impedance()
        except Exception as exc:
            errors.append(f"disable impedance failed: {exc}")
            try:
                backend.emergency_stop()
            except Exception as estop_exc:
                errors.append(f"electronic emergency stop failed: {estop_exc}")
        disconnected = False
        try:
            backend.disconnect()
            disconnected = True
        except Exception as exc:
            errors.append(f"backend disconnect failed: {exc}")
            try:
                backend.emergency_stop()
            except Exception as estop_exc:
                errors.append(
                    f"electronic emergency stop after disconnect failure "
                    f"failed: {estop_exc}")
        if errors:
            _CLEANUP_HAD_ERROR = True
            print("CRITICAL: cleanup was not fully confirmed:")
            for error in errors:
                print(f"  - {error}")
        if disconnected:
            cleanup_done = True
            _ACTIVE_BACKEND_CLEANUP = None
            try:
                atexit.unregister(_cleanup_backend)
            except Exception:
                pass
        return disconnected and not errors

    # This covers unexpected exceptions in FK, configuration, monitoring, and
    # diagnostics without requiring every call site to duplicate cleanup.
    _ACTIVE_BACKEND_CLEANUP = _cleanup_backend
    atexit.register(_cleanup_backend)
    print("MIT backend active; arm is draggable.\n")

    q = _wait_joint_angles(backend)
    if q is None:
        print("FAIL: no joint feedback."); _cleanup_backend(); return 1
    q = _joint_vector(q, "initial joint feedback")
    tcp_start, _ = model.fk(q)
    print(f"start q (deg)   = {np.round(np.degrees(q),2)}")
    print(f"start tcp (m)   = {np.round(tcp_start,4)}")

    # ---- Settle: zero-gravity hold, verify stability ----
    print(f"SETTLE {args.settle_s:.1f}s (zero-gravity, draggable)...")
    settle_q_ref = q.copy()
    t_settle = time.monotonic()
    last_log = 0.0
    try:
        while time.monotonic() - t_settle < args.settle_s:
            qn = _wait_joint_angles(backend, timeout=0.5)
            if qn is None:
                continue
            inner_error = backend.get_debug_state().get("last_error")
            if inner_error:
                raise RuntimeError(
                    f"MIT loop fault during settle: {inner_error}")
            backend.send(BackendCommand(
                q_cmd=qn.copy(), qdot_cmd=np.zeros(7), kp=np.zeros(7),
                kd=ZERO_GRAV_KD.copy(), torque_ff=np.zeros(7), mode="mit", stamp=time.time(),
            ))
            drift = float(np.max(np.abs(qn - settle_q_ref)))
            if time.monotonic() - last_log > 0.5:
                print(f"  settle drift = {np.degrees(drift):.3f} deg")
                last_log = time.monotonic()
            time.sleep(dt)
    except KeyboardInterrupt:
        print("\nAborted during settle."); _cleanup_backend(); return 0

    q = _wait_joint_angles(backend)
    if q is None:
        print("FAIL: lost feedback after settle."); _cleanup_backend(); return 1
    q = _joint_vector(q, "post-settle joint feedback")
    track_start_q = q.copy()
    limits = NeroPinocchioModel.JOINT_LIMITS
    engage_margins = np.minimum(q - limits[:, 0], limits[:, 1] - q)
    engage_limit_joint = int(np.argmin(engage_margins))
    if float(engage_margins[engage_limit_joint]) <= float(args.safe_margin):
        lower, upper = limits[engage_limit_joint]
        print(
            "\nREFUSING TO ENGAGE: post-settle posture is too close to a "
            "hard joint limit.\n"
            f"  joint{engage_limit_joint + 1}={q[engage_limit_joint]:.4f} rad, "
            f"limit=[{lower:.4f}, {upper:.4f}], "
            f"margin={engage_margins[engage_limit_joint]:.4f} "
            f"<= required {args.safe_margin:.4f} rad"
        )
        _cleanup_backend()
        return 2

    tcp_engage, R_engage = model.fk(q)
    target_tcp, R_target = _resolve_engagement_target(
        args.target_xyz,
        args.offset_xyz,
        args.rotz_deg,
        tcp_engage,
        R_engage,
        target_rpy_deg=args.target_rpy_deg,
        target_quaternion=args.target_quaternion,
    )
    print(f"engage q (deg)  = {np.round(np.degrees(q),2)}")
    print(f"engage tcp (m)  = {np.round(tcp_engage,4)}")
    print(f"target tcp (m)  = {np.round(target_tcp,4)}")
    print(f"target R       = {np.round(R_target,4).tolist()}")
    print(
        "engage initial err (m) = "
        f"{np.linalg.norm(target_tcp - tcp_engage):.6f}\n")

    # ---- Engage in-loop impedance (set target+gains ONCE) ----
    pre_engage_debug = backend.get_debug_state(include_trace=True)
    if pre_engage_debug.get("last_error"):
        print(
            "REFUSING TO ENGAGE: MIT loop faulted during settle: "
            f"{pre_engage_debug['last_error']}"
        )
        _print_inner_trip_debug(pre_engage_debug)
        _cleanup_backend()
        return 2
    cfg = _build_impedance_config(args, target_tcp, R_target, track_start_q)
    backend.set_cartesian_impedance(cfg)
    print("\nSETTLE OK. In-loop Cartesian impedance ENGAGED (200 Hz law). Push the arm to feel compliance.\n")

    # ---- Monitor loop (no impedance command sent here; the 200 Hz loop runs it) ----
    t0 = time.monotonic()
    last_log = 0.0
    tripped = False
    hard_tripped = False
    trip_reason = ""
    final_debug = {}
    count = 0
    try:
        while True:
            now = time.monotonic()
            loop_debug = backend.get_debug_state()
            inner_error = loop_debug.get("last_error")
            if inner_error:
                hard_tripped = True
                trip_reason = str(inner_error)
                final_debug = loop_debug
                break
            if args.duration > 0 and now - t0 >= args.duration:
                print(f"\nDuration {args.duration}s reached.")
                break

            q = backend.get_joint_angles()
            if q is None:
                time.sleep(dt); continue
            q = _joint_vector(q, "monitor joint feedback")

            tcp_now, R_tcp = model.fk(q)
            e_pos = target_tcp - tcp_now
            e_ori_world = _orientation_error_world(R_target, R_tcp)

            qdot_m = backend.get_joint_velocity()
            if qdot_m is None:
                qdot_m = np.zeros(7)
            else:
                qdot_m = _joint_vector(
                    qdot_m, "monitor joint-velocity feedback")

            # Slow monitor trips (the in-loop fast qdot trip is inside the MIT loop).
            excursion_per_joint = np.abs(q - track_start_q)
            excursion = float(np.max(excursion_per_joint))
            excursion_joint = int(np.argmax(excursion_per_joint))
            limits = NeroPinocchioModel.JOINT_LIMITS
            limit_margins = np.minimum(q - limits[:, 0], limits[:, 1] - q)
            limit_joint = int(np.argmin(limit_margins))
            near_limit = float(limit_margins[limit_joint]) < float(args.safe_margin)
            qd_max = float(np.max(np.abs(qdot_m)))
            qdot_joint = int(np.argmax(np.abs(qdot_m)))
            if excursion > args.max_excursion:
                tripped, trip_reason = True, (
                    f"joint{excursion_joint + 1} excursion {excursion:.3f}>{args.max_excursion} "
                    f"rad ({np.degrees(excursion):.1f}>{np.degrees(args.max_excursion):.1f}deg)")
            elif near_limit:
                lower, upper = limits[limit_joint, 0], limits[limit_joint, 1]
                tripped, trip_reason = True, (
                    f"joint{limit_joint + 1} within {args.safe_margin} rad of hard limit: "
                    f"q={q[limit_joint]:.3f} rad, limit=[{lower:.3f}, {upper:.3f}], "
                    f"margin={limit_margins[limit_joint]:.3f} rad")
            elif qd_max > args.instant_qdot_cap:
                tripped, trip_reason = True, (
                    f"monitor qdot joint{qdot_joint + 1} {qd_max:.2f}>{args.instant_qdot_cap} rad/s")
            if tripped:
                break

            count += 1
            if now - last_log >= args.log_period:
                err_mm = float(np.linalg.norm(e_pos)) * 1000.0
                rot_deg = float(np.degrees(np.linalg.norm(e_ori_world)))
                print(f"t={now-t0:5.1f}s err={err_mm:5.1f}mm rot={rot_deg:4.1f}deg "
                      f"|qdot_m|max={qd_max:.3f}rad/s exc={np.degrees(excursion):5.1f}deg")
                last_log = now
            time.sleep(dt)

    except KeyboardInterrupt:
        print("\nCtrl+C: stopping.")

    # Close the small race where the inner loop trips after the last monitor
    # poll but before the outer excursion check exits.
    post_debug = backend.get_debug_state(include_trace=True)
    if post_debug.get("last_error"):
        hard_tripped = True
        trip_reason = str(post_debug["last_error"])
        final_debug = post_debug

    # A hard inner-loop fault has already stopped the MIT thread and sent an
    # electronic e-stop.  Do not claim/send a gravity+damping hold afterward:
    # writes would only enter a dead command buffer.
    if hard_tripped:
        print(f"\n!!! HARD INNER TRIP: {trip_reason}")
        _print_inner_trip_debug(final_debug)
    elif tripped:
        print(f"\n!!! SLOW MONITOR TRIP: {trip_reason}")
        final_debug = post_debug
        _print_inner_trip_debug(final_debug)
        print("Disabling Cartesian impedance; entering gravity+damping safe hold (no e-stop).")
    if not hard_tripped:
        try:
            backend.disable_cartesian_impedance()
        except Exception as exc:
            hard_tripped = True
            trip_reason = f"failed to disable Cartesian impedance: {exc}"
            print(
                f"\n!!! FAIL-CLOSED: {trip_reason}; issuing electronic "
                "emergency stop."
            )
            backend.emergency_stop()
        # Brief gravity+damping hold so the arm does not get a sudden cut.
        if not hard_tripped:
            try:
                for _ in range(10):
                    qn = backend.get_joint_angles()
                    if qn is not None:
                        backend.send(BackendCommand(
                            q_cmd=np.asarray(qn), qdot_cmd=np.zeros(7), kp=np.zeros(7),
                            kd=ZERO_GRAV_KD.copy(), torque_ff=np.zeros(7), mode="mit", stamp=time.time(),
                        ))
                    time.sleep(0.02)
            except Exception as exc:
                hard_tripped = True
                trip_reason = f"failed to send gravity+damping safe hold: {exc}"
                print(
                    f"\n!!! FAIL-CLOSED: {trip_reason}; issuing electronic "
                    "emergency stop."
                )
                backend.emergency_stop()

    print(f"ran {count} monitor cycles, {time.monotonic()-t0:.1f}s")
    cleanup_ok = _cleanup_backend()
    if cleanup_ok:
        print("Disconnected. Arm back to default (gravity will hold it).")
    else:
        print(
            "CRITICAL: disconnection was not fully confirmed; keep the "
            "hardware e-stop engaged and inspect the CAN/controller state."
        )
    result = 0 if not (tripped or hard_tripped) else 2
    return result if cleanup_ok else 3


def main() -> int:
    """Run with a process-independent finally cleanup boundary."""
    global _ACTIVE_BACKEND_CLEANUP, _CLEANUP_HAD_ERROR
    _ACTIVE_BACKEND_CLEANUP = None
    _CLEANUP_HAD_ERROR = False
    try:
        result = _main_impl()
    except BaseException:
        cleanup = _ACTIVE_BACKEND_CLEANUP
        if cleanup is not None:
            cleanup()
        raise
    cleanup = _ACTIVE_BACKEND_CLEANUP
    if cleanup is not None:
        cleanup()
    return 3 if _CLEANUP_HAD_ERROR and result == 0 else result


if __name__ == "__main__":
    sys.exit(main())
