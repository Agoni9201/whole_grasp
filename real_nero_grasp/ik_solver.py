"""TRAC-IK solver construction, seeding, and deterministic Top-K selection.

The optional hooks on the composite solve functions let the legacy runner
inject its current module attributes at call time.  This preserves existing
``mock.patch.object(runner, ...)`` behavior while the default standalone path
uses the implementations defined here.
"""

import os

import numpy as np
from scipy.spatial.transform import Rotation as R

from .types import PlanNotExecutable, SafetyStopTriggered, ControlledMotionStopTriggered
from .planning_process import IsolatedKinematic, PlanningProcessError


__all__ = [
    "JOINT_LIMIT_NUMERICAL_TOL_RAD",
    "_wrap_joint_delta",
    "_pose_to_wxyz",
    "_get_ik_solver",
    "_joint_seed_candidates",
    "_joint_limit_margin",
    "_clip_numerical_joint_limit_overshoot",
    "_ranked_ik_solution_records_for_flange_pose",
    "_solve_ik_solutions_for_flange_pose",
    "_solve_ik_for_flange_pose",
]


_IK_SOLVER = None
_IK_SOLVER_KEY = None
_IK_SOLVER_ERROR = None

# URDF limits are stored as float32-like values by the deployed TRAC-IK
# binding.  For example, Nero J7 reports an upper limit of
# 1.570796251296997 rad while the configured 90 degree home target converts to
# 1.570796326794897 rad.  Treat only sub-microradian differences as numerical
# representation error; larger violations remain hard failures.
JOINT_LIMIT_NUMERICAL_TOL_RAD = 1e-6


def _wrap_joint_delta(q, q_ref):
    return (np.asarray(q, dtype=float) - np.asarray(q_ref, dtype=float) + np.pi) % (2.0 * np.pi) - np.pi


def _pose_to_wxyz(pose):
    pose = np.asarray(pose, dtype=float).reshape(6)
    quat_xyzw = R.from_euler("xyz", pose[3:]).as_quat()
    return np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]], dtype=float)


def _get_ik_solver(args):
    global _IK_SOLVER, _IK_SOLVER_ERROR, _IK_SOLVER_KEY
    isolated = (
        str(getattr(args, "trajectory_executor", "move_j")) == "mit"
        and not bool(getattr(args, "_planning_worker", False))
    )
    key = (os.path.abspath(str(args.ik_urdf)), str(args.ik_base_link),
           str(args.ik_tip_link), float(args.ik_timeout), float(args.ik_epsilon),
           str(args.ik_solver_type), isolated)
    if _IK_SOLVER is not None and _IK_SOLVER_KEY == key:
        return _IK_SOLVER
    if isolated:
        _IK_SOLVER = IsolatedKinematic(args)
        _IK_SOLVER_KEY = key
        return _IK_SOLVER
    try:
        from kinematic_trac_ik import Kinematic
    except Exception as exc:
        _IK_SOLVER_ERROR = exc
        raise RuntimeError(
            "TRAC-IK is not available. Install/source the TRAC-IK Python environment, "
            "or run without --motion-mode ik_joint."
        ) from exc

    urdf_path = os.path.abspath(str(args.ik_urdf))
    if not os.path.exists(urdf_path):
        raise FileNotFoundError(f"IK URDF not found: {urdf_path}")
    try:
        _IK_SOLVER = Kinematic(
            urdf_path=urdf_path,
            base_link_name=str(args.ik_base_link),
            target_link_name=str(args.ik_tip_link),
            timeout=float(args.ik_timeout),
            epsilon=float(args.ik_epsilon),
            solver_type=str(args.ik_solver_type),
        )
        _IK_SOLVER_KEY = key
        print(
            f"[ik] enabled urdf={urdf_path} "
            f"base={args.ik_base_link} tip={args.ik_tip_link} solver={args.ik_solver_type}"
        )
        return _IK_SOLVER
    except Exception as exc:
        _IK_SOLVER_ERROR = exc
        raise RuntimeError(f"Failed to initialize TRAC-IK solver: {type(exc).__name__}: {exc}") from exc


def _van_der_corput(index, base):
    """Return one deterministic low-discrepancy coordinate in ``[0, 1)``."""
    value = 0.0
    denominator = 1.0
    index = int(index)
    base = int(base)
    while index > 0:
        index, remainder = divmod(index, base)
        denominator *= base
        value += float(remainder) / denominator
    return float(value)


def _all_joint_halton_seeds(count, low, high, margin_rad):
    """Generate repeatable full-chain seeds without a random search explosion."""
    count = max(int(count), 0)
    low = np.asarray(low, dtype=float).reshape(7)
    high = np.asarray(high, dtype=float).reshape(7)
    margin = np.minimum(
        np.full(7, max(float(margin_rad), 0.0), dtype=float),
        np.maximum((high - low) * 0.1, 0.0),
    )
    inner_low = low + margin
    inner_high = high - margin
    primes = (2, 3, 5, 7, 11, 13, 17)
    seeds = []
    # Start at one: Halton index zero would put every joint at its lower
    # bound, which is both redundant and a poor numerical IK seed.
    for index in range(1, count + 1):
        unit = np.array(
            [_van_der_corput(index, base) for base in primes], dtype=float
        )
        seeds.append(inner_low + unit * (inner_high - inner_low))
    return seeds


def _joint_seed_candidates(current_joints, args, solver=None):
    """Return local and globally distributed seeds spanning all seven joints.

    TRAC-IK is a local numerical solver.  The historical seed set changed
    mostly J1/J7, so nominally free J2--J6 could repeatedly converge to the
    same shoulder/elbow/wrist basin.  Interleave deterministic full-chain
    Halton seeds with current/neutral interpolation seeds so even bounded
    fast-path prefixes contain more than one seven-DOF posture.
    """
    current = np.asarray(current_joints, dtype=float).reshape(7)
    initial = np.deg2rad(np.asarray(args.initial_joints_deg, dtype=float).reshape(7))

    interpolation_seeds = [
        (1.0 - alpha) * current + alpha * initial
        for alpha in (0.25, 0.50, 0.75)
    ]
    global_seeds = []
    if solver is not None:
        low, high = solver.get_joint_limits()
        global_seeds = _all_joint_halton_seeds(
            int(getattr(args, "ik_all_joint_seed_count", 24)),
            np.asarray(low, dtype=float).reshape(-1)[:7],
            np.asarray(high, dtype=float).reshape(-1)[:7],
            np.deg2rad(
                max(float(getattr(args, "ik_seed_limit_margin_deg", 2.0)), 0.0)
            ),
        )

    # Interleave local continuity seeds and global seven-joint seeds.  This is
    # important for side-view quick searches that deliberately use only a
    # bounded prefix of the complete seed set.
    seeds = [current, initial]
    for index in range(max(len(interpolation_seeds), len(global_seeds))):
        if index < len(global_seeds):
            seeds.append(global_seeds[index])
        if index < len(interpolation_seeds):
            seeds.append(interpolation_seeds[index])

    # Retain the old J7/J1 perturbations only as inexpensive supplementary
    # local branches; they are no longer the main source of IK diversity.
    for j7_deg in (90.0, 60.0, 30.0, 0.0, -30.0, -60.0):
        seed = current.copy()
        seed[6] = np.deg2rad(j7_deg)
        seeds.append(seed)

    for j1_delta_deg in (-20.0, 20.0):
        seed = current.copy()
        seed[0] += np.deg2rad(j1_delta_deg)
        seeds.append(seed)

    unique = []
    for seed in seeds:
        seed = np.asarray(seed, dtype=float).reshape(7)
        if not any(np.linalg.norm(_wrap_joint_delta(seed, old)) < 1e-6 for old in unique):
            unique.append(seed)
    return unique


def _joint_limit_margin(q, solver):
    q = np.asarray(q, dtype=float).reshape(-1)
    low, high = solver.get_joint_limits()
    low = np.asarray(low, dtype=float).reshape(-1)[: q.size]
    high = np.asarray(high, dtype=float).reshape(-1)[: q.size]
    return float(np.min(np.minimum(q - low, high - q)))


def _clip_numerical_joint_limit_overshoot(q, solver):
    """Return a limit-safe target, or ``None`` for a real/invalid violation.

    Planning and execution must use the same numerical boundary policy.
    Otherwise a fixed 90 degree home target can pass path preflight and then
    be rejected immediately before ``move_j`` because the URDF's float limit
    is a few ten-millionths of a radian below pi/2.
    """
    q = np.asarray(q, dtype=float).reshape(-1)
    low, high = solver.get_joint_limits()
    low = np.asarray(low, dtype=float).reshape(-1)[: q.size]
    high = np.asarray(high, dtype=float).reshape(-1)[: q.size]
    if (
        q.size == 0
        or low.size != q.size
        or high.size != q.size
        or not np.isfinite(q).all()
        or not np.isfinite(low).all()
        or not np.isfinite(high).all()
    ):
        return None
    tolerance = float(JOINT_LIMIT_NUMERICAL_TOL_RAD)
    if np.any(q < low - tolerance) or np.any(q > high + tolerance):
        return None
    return np.clip(q, low, high)


def _ranked_ik_solution_records_for_flange_pose(
    arm,
    flange_pose,
    args,
    label,
    current_joints_override=None,
    quiet=False,
    seed_limit=None,
    top_k=None,
    dedup_tolerance_rad=None,
    early_accept_margin_rad=None,
    *,
    get_solver=None,
    joint_seed_candidates=None,
    clip_limit=None,
    joint_limit_margin=None,
):
    """Return distinct IK solution records ranked by the legacy joint score.

    TRAC-IK can converge to the same numerical solution from several seeds.
    Counting those duplicates as separate alternatives gives downstream route
    search no additional choices, so solutions are sorted first and then
    deduplicated in bounded joint space.  Nero joints are not treated as
    continuously wrapping because the controller must remain inside the URDF
    limits during execution.

    ``early_accept_margin_rad`` is reserved for endpoint-existence checks.
    When set, seed traversal stops at the first valid solution whose hard
    joint-limit margin reaches the threshold.  Normal Top-K route planning
    leaves it unset and therefore preserves exhaustive ranking behavior.
    """
    get_solver = _get_ik_solver if get_solver is None else get_solver
    joint_seed_candidates = (
        _joint_seed_candidates
        if joint_seed_candidates is None
        else joint_seed_candidates
    )
    clip_limit = (
        _clip_numerical_joint_limit_overshoot
        if clip_limit is None
        else clip_limit
    )
    joint_limit_margin = (
        _joint_limit_margin
        if joint_limit_margin is None
        else joint_limit_margin
    )

    planning_stats = getattr(args, "_planning_stats", None)
    planning_mode = str(getattr(args, "_planning_ik_mode", "full"))
    if isinstance(planning_stats, dict):
        planning_stats["ik_calls"] = int(planning_stats.get("ik_calls", 0)) + 1
        mode_key = "fast_ik_calls" if planning_mode == "fast" else "full_ik_calls"
        planning_stats[mode_key] = int(planning_stats.get(mode_key, 0)) + 1

    solver = get_solver(args)
    flange_pose = np.asarray(flange_pose, dtype=float).reshape(6)
    target_pos = flange_pose[:3]
    target_wxyz = _pose_to_wxyz(flange_pose)
    current_joints = (
        np.asarray(arm.get_joint_angles(), dtype=float).reshape(7)
        if current_joints_override is None
        else np.asarray(current_joints_override, dtype=float).reshape(7)
    )

    try:
        seed_candidates = joint_seed_candidates(
            current_joints, args, solver=solver
        )
    except TypeError as exc:
        # Preserve compatibility with external/test hooks that implement the
        # historical two-argument seed callback.
        if "solver" not in str(exc):
            raise
        seed_candidates = joint_seed_candidates(current_joints, args)
    if seed_limit is not None:
        seed_candidates = seed_candidates[: max(int(seed_limit), 1)]
    solutions = []
    early_margin = (
        None
        if early_accept_margin_rad is None
        else max(float(early_accept_margin_rad), 0.0)
    )
    for seed_index, seed in enumerate(seed_candidates):
        if isinstance(planning_stats, dict):
            planning_stats["ik_seed_trials"] = int(
                planning_stats.get("ik_seed_trials", 0)
            ) + 1
        try:
            q = solver.solve_ik(
                target_position=target_pos,
                target_wxyz=target_wxyz,
                initial_guess=np.asarray(seed, dtype=float).reshape(-1)[: solver.num_joints],
                use_previous_solution=True,
            )
        except (PlanningProcessError, SafetyStopTriggered, ControlledMotionStopTriggered):
            raise
        except Exception as exc:
            if not quiet:
                print(f"[{label}][ik][warn] solve failed for seed: {type(exc).__name__}: {exc}")
            continue
        if q is None:
            continue
        q = np.asarray(q, dtype=float).reshape(-1)
        if q.size != 7:
            continue
        q = clip_limit(q, solver)
        if q is None:
            continue
        # Nero joints are bounded revolute joints.  Use the absolute numerical
        # command delta; wrapping would score/check a path the controller cannot
        # take across its joint limits.
        delta = q - current_joints
        max_delta = float(np.max(np.abs(delta)))
        if args.ik_max_joint_delta is not None and max_delta > float(args.ik_max_joint_delta):
            continue
        margin = joint_limit_margin(q, solver)
        joint7_delta = abs(float(delta[6]))
        # After the fingers have closed, preserving a small J7 change is much
        # less important than finding a reachable lift.  Keep the historical
        # preference for observation/pregrasp/grasp, but let lift endpoints
        # use their own (normally zero) J7 weight.  Hard URDF limits and path
        # collision validation are unchanged.
        joint7_weight = float(args.ik_joint7_weight)
        if "lift" in str(label).lower():
            joint7_weight = float(
                getattr(args, "ik_lift_joint7_weight", joint7_weight)
            )
        score = (
            float(np.linalg.norm(delta))
            + joint7_weight * joint7_delta
            - 0.02 * margin
        )
        record = {
            "score": float(score),
            "joints": np.asarray(q, dtype=float).reshape(7),
            "delta": np.asarray(delta, dtype=float).reshape(7),
            "limit_margin": float(margin),
            "seed_index": int(seed_index),
        }
        solutions.append(record)
        if early_margin is not None and float(margin) >= early_margin:
            return [record]

    if not solutions:
        raise PlanNotExecutable(f"{label} IK failed: no valid joint solution for target flange pose")

    solutions.sort(key=lambda item: item["score"])
    if dedup_tolerance_rad is None:
        dedup_tolerance_rad = np.deg2rad(
            max(float(getattr(args, "ik_solution_dedup_deg", 0.5)), 0.0)
        )
    dedup_tolerance_rad = max(float(dedup_tolerance_rad), 0.0)
    if top_k is None:
        top_k = getattr(args, "ik_solution_topk", 6)
    top_k = max(int(top_k), 1)

    distinct = []
    for record in solutions:
        q = record["joints"]
        if any(
            float(np.max(np.abs(q - old["joints"]))) <= dedup_tolerance_rad
            for old in distinct
        ):
            continue
        distinct.append(record)
        if len(distinct) >= top_k:
            break
    return distinct


def _solve_ik_solutions_for_flange_pose(
    arm,
    flange_pose,
    args,
    label,
    current_joints_override=None,
    quiet=False,
    seed_limit=None,
    top_k=None,
    dedup_tolerance_rad=None,
    *,
    get_solver=None,
    joint_seed_candidates=None,
    clip_limit=None,
    joint_limit_margin=None,
):
    """Return up to Top-K distinct, score-ordered joint solutions.

    This is the multi-solution API used by complete-route search.  Callers that
    need the historical single best solution should continue to use
    :func:`_solve_ik_for_flange_pose`.
    """
    records = _ranked_ik_solution_records_for_flange_pose(
        arm,
        flange_pose,
        args,
        label,
        current_joints_override=current_joints_override,
        quiet=quiet,
        seed_limit=seed_limit,
        top_k=top_k,
        dedup_tolerance_rad=dedup_tolerance_rad,
        get_solver=get_solver,
        joint_seed_candidates=joint_seed_candidates,
        clip_limit=clip_limit,
        joint_limit_margin=joint_limit_margin,
    )
    if not quiet:
        print(
            f"[{label}][ik] retained {len(records)} distinct joint "
            f"solution(s) for route search"
        )
        for rank, record in enumerate(records, start=1):
            print(
                f"[{label}][ik] option {rank}: "
                f"max_delta={np.rad2deg(np.max(np.abs(record['delta']))):.2f}deg "
                f"limit_margin={np.rad2deg(record['limit_margin']):.2f}deg "
                f"score={record['score']:.4f}"
            )
    return [record["joints"].copy() for record in records]


def _solve_ik_for_flange_pose(
    arm,
    flange_pose,
    args,
    label,
    current_joints_override=None,
    quiet=False,
    seed_limit=None,
    *,
    get_solver=None,
    joint_seed_candidates=None,
    clip_limit=None,
    joint_limit_margin=None,
):
    """Return the best IK solution while preserving the legacy single API."""
    solutions = _ranked_ik_solution_records_for_flange_pose(
        arm,
        flange_pose,
        args,
        label,
        current_joints_override=current_joints_override,
        quiet=quiet,
        seed_limit=seed_limit,
        top_k=1,
        get_solver=get_solver,
        joint_seed_candidates=joint_seed_candidates,
        clip_limit=clip_limit,
        joint_limit_margin=joint_limit_margin,
    )
    best = solutions[0]
    score = best["score"]
    q_best = best["joints"]
    delta = best["delta"]
    margin = best["limit_margin"]
    if not quiet:
        print(f"[{label}][ik] selected joint target deg =", np.round(np.rad2deg(q_best), 3).tolist())
        print(
            f"[{label}][ik] joint_delta_deg={np.round(np.rad2deg(delta), 3).tolist()} "
            f"max_delta={np.rad2deg(np.max(np.abs(delta))):.2f}deg "
            f"limit_margin={np.rad2deg(margin):.2f}deg score={score:.4f}"
        )
    return q_best
