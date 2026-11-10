"""Opt-in lightweight post-grasp RRT-Connect planning.

The legacy planner deliberately remains the default.  This module is used only
when ``--post-grasp-free-route`` is supplied.  It keeps the already validated
support-release endpoint, finds one collision-free high joint configuration,
and then connects to an IK solution at the requested place position.

Unlike the legacy lift planner, the high waypoint is a *joint-space region*:
only TCP height, hard joint limits and collision safety are mandatory.  TCP XY,
orientation and J7 may change naturally after the strict support release.
"""

from copy import copy
from dataclasses import replace
import contextlib
import io
import time

import numpy as np

from . import ik_solver
from .types import PlanNotExecutable, PlannedJointStage, UnsafeMotionTarget


_HALTON_PRIMES = (2, 3, 5, 7, 11, 13, 17)


def _van_der_corput(index, base):
    value = 0.0
    denominator = 1.0
    while index:
        index, remainder = divmod(index, base)
        denominator *= base
        value += remainder / denominator
    return value


def _halton(index, dimensions=7):
    return np.asarray(
        [_van_der_corput(index, _HALTON_PRIMES[i]) for i in range(dimensions)],
        dtype=float,
    )


def _quiet_edge_validator(validate_motion_joint_path, context, args, label):
    """Build a cached, quiet edge predicate for the exploratory search."""
    search_args = copy(args)
    search_args.motion_path_sample_joint_deg = max(
        float(getattr(args, "post_grasp_rrt_coarse_sample_deg", 4.0)), 1.0
    )
    cache = {}

    def valid(q0, q1):
        a = tuple(np.round(np.asarray(q0, dtype=float), 5))
        b = tuple(np.round(np.asarray(q1, dtype=float), 5))
        key = (a, b) if a <= b else (b, a)
        if key in cache:
            return cache[key]
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                validate_motion_joint_path(q0, q1, context, search_args, label)
        except (PlanNotExecutable, UnsafeMotionTarget, ValueError):
            cache[key] = False
        else:
            cache[key] = True
        return cache[key]

    return valid


def _steer(q_from, q_to, step):
    delta = np.asarray(q_to, dtype=float) - np.asarray(q_from, dtype=float)
    distance = float(np.linalg.norm(delta))
    if distance <= step:
        return np.asarray(q_to, dtype=float).copy(), True
    return np.asarray(q_from, dtype=float) + delta * (step / distance), False


def _nearest(nodes, query):
    points = np.asarray(nodes, dtype=float)
    return int(np.argmin(np.linalg.norm(points - query, axis=1)))


def _trace(nodes, parents, index):
    path = []
    while index >= 0:
        path.append(np.asarray(nodes[index], dtype=float).copy())
        index = int(parents[index])
    return list(reversed(path))


def rrt_connect(
    q_start,
    q_goal,
    lower,
    upper,
    edge_valid,
    *,
    step_rad,
    timeout_s,
    max_nodes,
    goal_bias,
    seed,
):
    """Return a collision-free joint path or ``None``.

    Sampling is deterministic for a given seed, which makes failed real-robot
    sessions reproducible.  Collision checking itself is supplied by the
    existing dynamic point-cloud/capsule validator.
    """
    q_start = np.asarray(q_start, dtype=float).reshape(7)
    q_goal = np.asarray(q_goal, dtype=float).reshape(7)
    if edge_valid(q_start, q_goal):
        return [q_start.copy(), q_goal.copy()]

    rng = np.random.default_rng(int(seed))
    trees = [
        {"nodes": [q_start.copy()], "parents": [-1], "root": "start"},
        {"nodes": [q_goal.copy()], "parents": [-1], "root": "goal"},
    ]
    deadline = time.monotonic() + max(float(timeout_s), 0.05)
    iteration = 0
    while time.monotonic() < deadline:
        if len(trees[0]["nodes"]) + len(trees[1]["nodes"]) >= int(max_nodes):
            break
        tree_a, tree_b = trees
        if rng.random() < float(goal_bias):
            sample = np.asarray(tree_b["nodes"][0], dtype=float)
        else:
            # Mix pseudorandom and low-discrepancy samples.  The latter makes
            # the small node budget cover all seven joints more evenly.
            if iteration % 2:
                unit = _halton(iteration + 1)
            else:
                unit = rng.random(7)
            sample = lower + unit * (upper - lower)
        iteration += 1

        index_a = _nearest(tree_a["nodes"], sample)
        q_new, _ = _steer(tree_a["nodes"][index_a], sample, step_rad)
        if not edge_valid(tree_a["nodes"][index_a], q_new):
            trees.reverse()
            continue
        tree_a["nodes"].append(q_new)
        tree_a["parents"].append(index_a)
        index_new_a = len(tree_a["nodes"]) - 1

        index_b = _nearest(tree_b["nodes"], q_new)
        connected = False
        while time.monotonic() < deadline:
            q_next, reached = _steer(tree_b["nodes"][index_b], q_new, step_rad)
            if not edge_valid(tree_b["nodes"][index_b], q_next):
                break
            tree_b["nodes"].append(q_next)
            tree_b["parents"].append(index_b)
            index_b = len(tree_b["nodes"]) - 1
            if reached:
                connected = True
                break
        if connected:
            path_a = _trace(tree_a["nodes"], tree_a["parents"], index_new_a)
            path_b = _trace(tree_b["nodes"], tree_b["parents"], index_b)
            if tree_a["root"] == "start":
                return path_a + list(reversed(path_b))[1:]
            return path_b + list(reversed(path_a))[1:]
        trees.reverse()
    return None


def _shortcut(path, edge_valid, attempts=80):
    path = [np.asarray(q, dtype=float).copy() for q in path]
    if len(path) <= 2:
        return path
    # Deterministic farthest-first shortcutting avoids a second random source.
    for _ in range(max(int(attempts), 0)):
        changed = False
        for i in range(len(path) - 2):
            for j in range(len(path) - 1, i + 1, -1):
                if j <= i + 1:
                    continue
                if edge_valid(path[i], path[j]):
                    path = path[: i + 1] + path[j:]
                    changed = True
                    break
            if changed:
                break
        if not changed:
            break
    return path


def _densify(path, maximum_joint_step_rad):
    dense = [np.asarray(path[0], dtype=float).copy()]
    for target in path[1:]:
        target = np.asarray(target, dtype=float)
        delta = target - dense[-1]
        count = max(
            int(np.ceil(np.max(np.abs(delta)) / maximum_joint_step_rad)), 1
        )
        start = dense[-1].copy()
        for index in range(1, count + 1):
            dense.append(start + delta * (index / count))
    return dense


def _stage_from_joints(arm, joints, args, label, modeled_tcp_pose_for_joints):
    tcp = np.asarray(modeled_tcp_pose_for_joints(joints, args), dtype=float).reshape(6)
    flange = np.asarray(arm.get_tcp2flange_pose(tcp), dtype=float).reshape(6)
    return PlannedJointStage(
        label=str(label),
        tcp_pose=tcp,
        flange_pose=flange,
        joints=np.asarray(joints, dtype=float).reshape(7),
        phase="carrying",
    )


def plan_post_grasp_joint_hold(q_start, carrying_context, args, label):
    """Reach the operator's exact joint goal with a checked payload path."""
    from .planning_process import config_snapshot, get_planning_process, planning_health_check
    if str(getattr(args, "trajectory_executor", "move_j")) == "mit" and not bool(
        getattr(args, "_planning_worker", False)
    ):
        scene = carrying_context.scene
        result = get_planning_process().request("post_grasp_joint_hold", dict(
            config=config_snapshot(args), q_start=np.asarray(q_start), label=label,
            scene=dict(full_scene_base=scene.full_scene_base,
                       environment_base=scene.environment_base, target_base=scene.target_base,
                       map_id=scene.map_id),
            attached_target_tcp=carrying_context.attached_target_tcp,
            closed_gripper_width=carrying_context.closed_gripper_width,
        ), timeout_s=max(float(getattr(args, "post_grasp_rrt_timeout", 3.0)) + 20., 30.),
            health_check=lambda: planning_health_check(args, label), args=args)
        if not result["ok"]:
            raise PlanNotExecutable(result["reason"])
        return tuple(PlannedJointStage(**stage) for stage in result["stages"])

    from .collision import validate_motion_joint_path, _urdf_chain_geometry
    from .motion_planning import _required_joint_limit_margin_deg, _modeled_tcp_pose_for_joints
    from .transforms import _as_pose_list

    q_start = np.asarray(q_start, dtype=float).reshape(7)
    goal = np.deg2rad(np.asarray(args.post_grasp_hold_joints_deg, dtype=float).reshape(7))
    lower, upper = ik_solver._get_ik_solver(args).get_joint_limits()
    reserve = np.deg2rad(_required_joint_limit_margin_deg(args, "motion_min_joint_limit_margin_deg"))
    lower, upper = np.asarray(lower) + reserve, np.asarray(upper) - reserve
    if np.any(goal < lower) or np.any(goal > upper):
        raise PlanNotExecutable(f"{label} requested hold joints violate the executor's joint-limit margin")
    start_tcp = _modeled_tcp_pose_for_joints(q_start, args)
    goal_tcp = _modeled_tcp_pose_for_joints(goal, args)
    context = replace(carrying_context, min_tcp_z=min(start_tcp[2], goal_tcp[2]) - .006)
    edge = _quiet_edge_validator(validate_motion_joint_path, context, args, f"{label}_search")
    if not edge(goal, goal):
        raise PlanNotExecutable(f"{label} requested hold joint pose fails robot/payload collision checks")
    path = rrt_connect(q_start, goal, lower, upper, edge,
        step_rad=np.deg2rad(float(getattr(args, "post_grasp_rrt_step_deg", 8.0))),
        timeout_s=float(getattr(args, "post_grasp_rrt_timeout", 3.0)),
        max_nodes=int(getattr(args, "post_grasp_rrt_max_nodes", 400)),
        goal_bias=float(getattr(args, "post_grasp_rrt_goal_bias", .20)),
        seed=int(getattr(args, "post_grasp_rrt_seed", 0)))
    if path is None:
        raise PlanNotExecutable(f"{label} no collision-free route to requested hold joints")
    path = _shortcut(path, edge, int(getattr(args, "post_grasp_rrt_shortcut_attempts", 80)))
    path = _densify(path, np.deg2rad(float(getattr(args, "post_grasp_execution_waypoint_deg", 20.0))))
    stages = []
    for index, (before, after) in enumerate(zip(path, path[1:]), start=1):
        stage_label = f"{label}_waypoint{index}"
        metrics = validate_motion_joint_path(before, after, context, args, stage_label)
        stages.append(PlannedJointStage(
            label=stage_label, tcp_pose=_modeled_tcp_pose_for_joints(after, args),
            flange_pose=_as_pose_list(_urdf_chain_geometry(after, args)[1]),
            joints=after.copy(), phase="carrying", min_tcp_z=context.min_tcp_z,
            path_clearance_margin=float(metrics["margin"])))
    print(f"[{label}] exact hold joints(deg)={np.rad2deg(goal).round(4).tolist()} "
          f"TCP_z={goal_tcp[2]:.4f}m carrying_waypoints={len(stages)}")
    return tuple(stages)


def _high_joint_candidates(q_support, grasp_tcp, args, modeled_tcp_pose_for_joints):
    solver = ik_solver._get_ik_solver(args)
    lower, upper = solver.get_joint_limits()
    lower = np.asarray(lower, dtype=float).reshape(7)
    upper = np.asarray(upper, dtype=float).reshape(7)
    target_z = float(grasp_tcp[2]) + max(
        float(getattr(args, "post_grasp_total_rise", 0.150)), 0.020
    )
    raw = []
    initial = getattr(args, "initial_joints_deg", None)
    if initial is not None:
        q_home = np.clip(np.deg2rad(np.asarray(initial, dtype=float).reshape(7)), lower, upper)
        raw.append(q_home)
        for alpha in (0.75, 0.50, 0.25):
            raw.append(q_support + alpha * (q_home - q_support))

    sample_count = max(int(getattr(args, "post_grasp_high_region_samples", 64)), 0)
    # Sample mostly around the support/home corridor, with a small global tail.
    for index in range(1, sample_count + 1):
        unit = _halton(index)
        global_q = lower + unit * (upper - lower)
        raw.append(global_q)

    scored = []
    for q in raw:
        tcp = np.asarray(modeled_tcp_pose_for_joints(q, args), dtype=float).reshape(6)
        if not np.isfinite(tcp).all() or float(tcp[2]) < target_z:
            continue
        xy_drift = float(np.linalg.norm(tcp[:2] - np.asarray(grasp_tcp)[:2]))
        max_xy = max(float(getattr(args, "post_grasp_high_region_max_xy", 0.25)), 0.02)
        if xy_drift > max_xy:
            continue
        score = (
            float(np.linalg.norm(q - q_support))
            + 0.5 * xy_drift
            + 0.2 * max(float(tcp[2]) - target_z, 0.0)
        )
        scored.append((score, q.copy(), tcp.copy()))
    scored.sort(key=lambda item: item[0])
    return scored, lower, upper, target_z


def _place_joint_candidates(arm, place_tcp, seed_joints, args):
    poses = [np.asarray(place_tcp, dtype=float).reshape(6).copy()]
    # One bounded fallback preserves the requested XYZ while allowing the
    # post-lift wrist orientation to continue into place.
    seed_tcp = None
    try:
        from .motion_planning import _modeled_tcp_pose_for_joints

        seed_tcp = _modeled_tcp_pose_for_joints(seed_joints, args)
    except Exception:
        seed_tcp = None
    if seed_tcp is not None:
        relaxed = poses[0].copy()
        relaxed[3:] = np.asarray(seed_tcp, dtype=float).reshape(6)[3:]
        poses.append(relaxed)

    results = []
    for pose_index, pose in enumerate(poses):
        flange = np.asarray(arm.get_tcp2flange_pose(pose), dtype=float).reshape(6)
        try:
            solutions = ik_solver._solve_ik_solutions_for_flange_pose(
                arm,
                flange,
                args,
                f"post_grasp_free_place_option{pose_index + 1}",
                current_joints_override=seed_joints,
                quiet=True,
                seed_limit=max(int(getattr(args, "post_grasp_place_ik_seed_limit", 5)), 1),
                top_k=max(int(getattr(args, "post_grasp_place_ik_topk", 2)), 1),
            )
        except PlanNotExecutable:
            continue
        results.extend((np.asarray(q, dtype=float), pose.copy()) for q in solutions)
    return results


def plan_post_grasp_free_route(
    arm,
    q_support,
    grasp_tcp,
    place_tcp,
    carrying_context,
    args,
    label,
    *,
    validate_motion_joint_path,
    modeled_tcp_pose_for_joints,
):
    """Plan support endpoint -> free high region -> exact place position."""
    if place_tcp is None:
        raise PlanNotExecutable("post-grasp free route requires a place pose")
    q_support = np.asarray(q_support, dtype=float).reshape(7)
    candidates, lower, upper, target_z = _high_joint_candidates(
        q_support, grasp_tcp, args, modeled_tcp_pose_for_joints
    )
    if not candidates:
        raise PlanNotExecutable(
            f"{label} found no joint sample with TCP z >= {target_z:.3f}m"
        )

    lift_context = replace(
        carrying_context,
        min_tcp_z=float(np.asarray(grasp_tcp, dtype=float)[2])
        + max(float(getattr(args, "motion_support_release_height", 0.020)), 0.020)
        - 0.006,
    )
    lift_edge = _quiet_edge_validator(
        validate_motion_joint_path, lift_context, args, f"{label}_rrt_lift_edge"
    )
    step = np.deg2rad(max(float(getattr(args, "post_grasp_rrt_step_deg", 8.0)), 1.0))
    high_attempts = max(int(getattr(args, "post_grasp_high_goal_attempts", 3)), 1)
    high_path = None
    q_high = None
    high_tcp = None
    started = time.monotonic()
    for attempt, (_, candidate, candidate_tcp) in enumerate(candidates[:high_attempts], start=1):
        if not lift_edge(candidate, candidate):
            continue
        high_path = rrt_connect(
            q_support,
            candidate,
            lower,
            upper,
            lift_edge,
            step_rad=step,
            timeout_s=float(getattr(args, "post_grasp_rrt_timeout", 3.0)),
            max_nodes=int(getattr(args, "post_grasp_rrt_max_nodes", 400)),
            goal_bias=float(getattr(args, "post_grasp_rrt_goal_bias", 0.20)),
            seed=int(getattr(args, "post_grasp_rrt_seed", 0)) + attempt,
        )
        if high_path is not None:
            q_high = candidate
            high_tcp = candidate_tcp
            break
    if high_path is None:
        raise PlanNotExecutable(
            f"{label} RRT-Connect could not reach the z>={target_z:.3f}m "
            f"high region using {min(len(candidates), high_attempts)} bounded goal(s)"
        )

    place_options = _place_joint_candidates(arm, place_tcp, q_high, args)
    if not place_options:
        raise PlanNotExecutable(
            f"{label} place XYZ has no IK solution in the two bounded orientation modes"
        )
    transport_path = None
    selected_place_tcp = None
    transport_context = replace(
        carrying_context,
        min_tcp_z=min(float(high_tcp[2]), float(np.asarray(place_tcp)[2])) - 0.010,
    )
    transport_edge = _quiet_edge_validator(
        validate_motion_joint_path,
        transport_context,
        args,
        f"{label}_rrt_transport_edge",
    )
    for option_index, (q_place, option_tcp) in enumerate(place_options, start=1):
        if not transport_edge(q_place, q_place):
            continue
        transport_path = rrt_connect(
            q_high,
            q_place,
            lower,
            upper,
            transport_edge,
            step_rad=step,
            timeout_s=float(getattr(args, "post_grasp_rrt_timeout", 3.0)),
            max_nodes=int(getattr(args, "post_grasp_rrt_max_nodes", 400)),
            goal_bias=float(getattr(args, "post_grasp_rrt_goal_bias", 0.20)),
            seed=int(getattr(args, "post_grasp_rrt_seed", 0)) + 100 + option_index,
        )
        if transport_path is not None:
            selected_place_tcp = option_tcp
            break
    if transport_path is None:
        raise PlanNotExecutable(
            f"{label} RRT-Connect could not connect the high region to any "
            f"of {len(place_options)} bounded place IK solution(s)"
        )

    high_path = _shortcut(
        high_path,
        lift_edge,
        attempts=int(getattr(args, "post_grasp_rrt_shortcut_attempts", 80)),
    )
    transport_path = _shortcut(
        transport_path,
        transport_edge,
        attempts=int(getattr(args, "post_grasp_rrt_shortcut_attempts", 80)),
    )
    execution_step = np.deg2rad(
        max(float(getattr(args, "post_grasp_execution_waypoint_deg", 20.0)), 2.0)
    )
    high_path = _densify(high_path, execution_step)
    transport_path = _densify(transport_path, execution_step)

    stages = []
    q_previous = q_support
    combined = [
        (q, lift_context, "lift") for q in high_path[1:]
    ] + [
        (q, transport_context, "transport") for q in transport_path[1:]
    ]
    for index, (q, context, phase_name) in enumerate(combined, start=1):
        # Mandatory final validation uses the original fine sampling settings.
        validate_motion_joint_path(
            q_previous,
            q,
            context,
            args,
            f"{label}_{phase_name}_waypoint{index}_final",
        )
        stages.append(
            _stage_from_joints(
                arm,
                q,
                args,
                f"{label}_{phase_name}_waypoint{index}",
                modeled_tcp_pose_for_joints,
            )
        )
        q_previous = np.asarray(q, dtype=float)

    elapsed = time.monotonic() - started
    print(
        f"[{label}][rrt] selected support->high->place route "
        f"high_z={float(high_tcp[2]):.3f}m required={target_z:.3f}m "
        f"waypoints={len(stages)} elapsed={elapsed:.2f}s"
    )
    return tuple(stages), np.asarray(selected_place_tcp, dtype=float).reshape(6)


__all__ = ["rrt_connect", "plan_post_grasp_free_route", "plan_post_grasp_joint_hold"]
