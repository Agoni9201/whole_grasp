"""Collision-checked IK stage and guarded-route planning primitives."""

from copy import copy
from dataclasses import replace
import re

import numpy as np
from scipy.spatial.transform import Rotation as R

from real_nero_point_test import pose_error

from . import ik_solver
from .transforms import (
    _as_pose_list,
    _make_T_from_pose,
    _pose_rpy_error_norm,
    interpolate_pose,
)
from .types import (
    MotionCollisionContext,
    PlanNotExecutable,
    PlannedJointStage,
    UnsafeMotionTarget,
)


__all__ = [
    "_motion_context",
    "_check_motion_joint_margin",
    "_plan_tcp_motion_stage_options",
    "_plan_tcp_motion_stage",
    "_plan_tcp_stage_sequence_beam",
    "_motion_route_height_candidates",
    "_plan_up_and_over_route",
    "_plan_guarded_route",
    "_modeled_tcp_pose_for_joints",
]


def _invariant_attached_target_table_start_failure(exc):
    """Return true when no IK branch can change the failed path sample.

    The attached target pose at sample zero is fixed by the grasp TCP.  If its
    robust payload floor already violates the table plane, trying more IK
    solutions or more beam parents repeats exactly the same collision test.
    """
    text = str(exc).lower()
    return (
        "attached_target" in text
        and "table_plane" in text
        and ("sample 0" in text or "sample=0" in text)
    )


def _invariant_support_release_start_failure(exc):
    """Detect a shared closed-tool collision that IK branching cannot fix."""
    text = str(exc).lower()
    if "support_release" not in text:
        return False
    matches = re.findall(
        r"(?:gripper_tool|gripper_closed_finger_[+-]1) gets within\s+"
        r"(-?\d+(?:\.\d+)?)mm of "
        r"environment at sample(?:=|\s+)(\d+)",
        text,
    )
    if len(matches) < 2:
        return False
    clearances = [float(value) for value, _sample in matches]
    samples = [int(sample) for _value, sample in matches]
    return (
        max(clearances) < 0.0
        and max(clearances) - min(clearances) <= 1.0
        and max(samples) <= 10
    )


def _collision_function(name):
    """Resolve collision helpers lazily to keep the dependency one-way."""
    from . import collision

    return getattr(collision, name)


def _motion_context(
    scene,
    phase,
    attached_target_tcp=None,
    placed_target_base=None,
    min_tcp_z=None,
    runtime_min_tcp_z_slack=0.0,
    initial_support_release=None,
    closed_gripper_width=None,
    require_support_release_latch_at_endpoint=True,
):
    return MotionCollisionContext(
        scene=scene,
        phase=str(phase),
        allow_target_tool_contact=str(phase) in {
            "target_approach",
            "released_retreat",
            "released_free",
        },
        attached_target_tcp=attached_target_tcp,
        placed_target_base=placed_target_base,
        min_tcp_z=None if min_tcp_z is None else float(min_tcp_z),
        runtime_min_tcp_z_slack=max(float(runtime_min_tcp_z_slack), 0.0),
        initial_support_release=initial_support_release,
        closed_gripper_width=(
            None
            if closed_gripper_width is None
            else max(float(closed_gripper_width), 0.0)
        ),
        require_support_release_latch_at_endpoint=bool(
            require_support_release_latch_at_endpoint
        ),
    )

def _check_motion_joint_margin(
    joints,
    args,
    label,
    *,
    get_solver=None,
    joint_limit_margin=None,
):
    get_solver = ik_solver._get_ik_solver if get_solver is None else get_solver
    joint_limit_margin = (
        ik_solver._joint_limit_margin
        if joint_limit_margin is None
        else joint_limit_margin
    )
    minimum_deg = max(
        float(getattr(args, "motion_min_joint_limit_margin_deg", 0.0)), 0.0
    )
    if minimum_deg <= 0.0:
        return
    solver = get_solver(args)
    margin_deg = float(np.rad2deg(joint_limit_margin(joints, solver)))
    if margin_deg < minimum_deg:
        raise PlanNotExecutable(
            f"{label} joint-limit margin={margin_deg:.2f}deg < "
            f"required {minimum_deg:.2f}deg"
        )

def _plan_tcp_motion_stage_options(
    arm,
    tcp_pose,
    q_previous,
    context,
    args,
    label,
    fixed_joints=None,
    max_valid_options=None,
    *,
    assert_table_guard=None,
    solve_ik_solutions=None,
    check_motion_joint_margin=None,
    validate_joint_path=None,
    clip_limit=None,
    get_solver=None,
    urdf_chain_geometry=None,
    as_pose_list=None,
    make_T_from_pose=None,
    pose_error_fn=None,
):
    """Return collision-validated joint alternatives for one TCP stage.

    A 7-DoF arm can reach one Cartesian pose through several distinct joint
    branches.  Keeping all locally valid branches is essential for the
    cross-stage beam search: a low-cost pregrasp branch may be a dead end for
    the later grasp or lift even though another pregrasp branch succeeds.

    ``max_valid_options`` is used by the legacy single-stage wrapper to retain
    its historical first-valid behaviour.  ``None`` keeps every distinct
    Top-K solution returned by the IK solver.
    """
    assert_table_guard = (
        _collision_function("assert_table_guard_target_tcp")
        if assert_table_guard is None
        else assert_table_guard
    )
    solve_ik_solutions = (
        ik_solver._solve_ik_solutions_for_flange_pose
        if solve_ik_solutions is None
        else solve_ik_solutions
    )
    check_motion_joint_margin = (
        _check_motion_joint_margin
        if check_motion_joint_margin is None
        else check_motion_joint_margin
    )
    validate_joint_path = (
        _collision_function("validate_motion_joint_path")
        if validate_joint_path is None
        else validate_joint_path
    )
    clip_limit = (
        ik_solver._clip_numerical_joint_limit_overshoot
        if clip_limit is None
        else clip_limit
    )
    get_solver = ik_solver._get_ik_solver if get_solver is None else get_solver
    urdf_chain_geometry = (
        _collision_function("_urdf_chain_geometry")
        if urdf_chain_geometry is None
        else urdf_chain_geometry
    )
    as_pose_list = _as_pose_list if as_pose_list is None else as_pose_list
    make_T_from_pose = (
        _make_T_from_pose if make_T_from_pose is None else make_T_from_pose
    )
    pose_error_fn = pose_error if pose_error_fn is None else pose_error_fn

    tcp_pose = np.asarray(tcp_pose, dtype=float).reshape(6)
    assert_table_guard(tcp_pose, args, label)
    if fixed_joints is None:
        flange_pose = np.asarray(
            arm.get_tcp2flange_pose(tcp_pose), dtype=float
        ).reshape(6)
        if args.min_flange_z is not None and float(flange_pose[2]) < float(
            args.min_flange_z
        ):
            raise UnsafeMotionTarget(
                f"{label} flange z={float(flange_pose[2]):.4f}m below "
                f"--min-flange-z={float(args.min_flange_z):.4f}m"
            )
        solve_kwargs = {"current_joints_override": q_previous}
        planning_seed_limit = getattr(args, "_planning_ik_seed_limit", None)
        if planning_seed_limit is not None:
            solve_kwargs["seed_limit"] = max(int(planning_seed_limit), 1)
        joint_options = solve_ik_solutions(
            arm,
            flange_pose,
            args,
            f"{label}_plan",
            **solve_kwargs,
        )
        rejected_options = []
        stages = []
        for option_index, option_joints in enumerate(joint_options, start=1):
            option_label = (
                label
                if option_index == 1
                else f"{label}_ik_option{option_index}"
            )
            try:
                check_motion_joint_margin(
                    option_joints, args, option_label
                )
                if (
                    bool(
                        getattr(
                            args, "grasp_coarse_collision_screen", False
                        )
                    )
                    and "pregrasp" in str(option_label).lower()
                ):
                    coarse_args = copy(args)
                    coarse_args.motion_path_sample_joint_deg = max(
                        float(
                            getattr(
                                args,
                                "grasp_coarse_collision_sample_joint_deg",
                                3.0,
                            )
                        ),
                        float(
                            getattr(args, "motion_path_sample_joint_deg", 1.0)
                        ),
                    )
                    validate_joint_path(
                        q_previous,
                        option_joints,
                        context,
                        coarse_args,
                        f"{option_label}_coarse_screen",
                    )
                path_metrics = validate_joint_path(
                    q_previous,
                    option_joints,
                    context,
                    args,
                    option_label,
                )
            except (PlanNotExecutable, UnsafeMotionTarget) as exc:
                rejected_options.append(
                    f"rank={option_index}:{type(exc).__name__}:{exc}"
                )
                print(
                    f"[{label}][ik][route_reject] option "
                    f"{option_index}/{len(joint_options)}: "
                    f"{type(exc).__name__}: {exc}"
                )
                if _invariant_attached_target_table_start_failure(exc):
                    print(
                        f"[{label}][ik][route_abort] attached-target table "
                        "failure is already present at sample 0; remaining "
                        "IK options cannot change the payload start pose"
                    )
                    raise
                continue
            joints = np.asarray(option_joints, dtype=float).reshape(7)
            path_margin = None
            if isinstance(path_metrics, dict) and "margin" in path_metrics:
                value = float(path_metrics["margin"])
                if np.isfinite(value):
                    path_margin = value
            if option_index > 1 and len(stages) == 0:
                print(
                    f"[{label}][ik][route_fallback] selected collision-checked "
                    f"option {option_index}/{len(joint_options)} after earlier "
                    "IK branch rejection"
                )
            stages.append(
                PlannedJointStage(
                    label=str(label),
                    tcp_pose=tcp_pose.copy(),
                    flange_pose=np.asarray(flange_pose, dtype=float).reshape(6),
                    joints=joints.copy(),
                    phase=str(context.phase),
                    min_tcp_z=context.min_tcp_z,
                    runtime_min_tcp_z_slack=context.runtime_min_tcp_z_slack,
                    require_support_release_latch_at_endpoint=bool(
                        getattr(
                            context,
                            "require_support_release_latch_at_endpoint",
                            True,
                        )
                    ),
                    path_clearance_margin=path_margin,
                )
            )
            if (
                max_valid_options is not None
                and len(stages) >= max(int(max_valid_options), 1)
            ):
                break
        if len(stages) == 0:
            raise PlanNotExecutable(
                f"{label} none of {len(joint_options)} retained IK solution(s) "
                "passed joint-margin and motion-path validation: "
                + " | ".join(rejected_options)
            )
        if len(stages) > 1:
            print(
                f"[{label}][ik][beam] retained {len(stages)} collision-checked "
                "joint branch(es) for downstream stages"
            )
        return tuple(stages)
    else:
        requested_joints = np.asarray(
            fixed_joints, dtype=float
        ).reshape(7)
        joints = clip_limit(
            requested_joints, get_solver(args)
        )
        if joints is None:
            raise PlanNotExecutable(
                f"{label} fixed joint target is invalid or outside joint limits"
            )
        _, T_base_flange = urdf_chain_geometry(joints, args)
        flange_pose = as_pose_list(T_base_flange)
        if args.min_flange_z is not None and float(flange_pose[2]) < float(
            args.min_flange_z
        ):
            raise UnsafeMotionTarget(
                f"{label} flange z={float(flange_pose[2]):.4f}m below "
                f"--min-flange-z={float(args.min_flange_z):.4f}m"
            )
        modeled_tcp = as_pose_list(
            T_base_flange
            @ make_T_from_pose(np.asarray(args.tcp_offset, dtype=float).reshape(6))
        )
        pos_err, rpy_err = pose_error_fn(modeled_tcp, tcp_pose)
        if pos_err > 0.003 or rpy_err > np.deg2rad(1.0):
            raise PlanNotExecutable(
                f"{label} fixed joint target does not match requested TCP: "
                f"pos_err={pos_err:.4f}m rpy_err={np.rad2deg(rpy_err):.2f}deg"
            )
        check_motion_joint_margin(joints, args, label)
        path_metrics = validate_joint_path(
            q_previous, joints, context, args, label
        )
        path_margin = None
        if isinstance(path_metrics, dict) and "margin" in path_metrics:
            value = float(path_metrics["margin"])
            if np.isfinite(value):
                path_margin = value
    return (
        PlannedJointStage(
            label=str(label),
            tcp_pose=tcp_pose.copy(),
            flange_pose=np.asarray(flange_pose, dtype=float).reshape(6),
            joints=np.asarray(joints, dtype=float).reshape(7),
            phase=str(context.phase),
            min_tcp_z=context.min_tcp_z,
            runtime_min_tcp_z_slack=context.runtime_min_tcp_z_slack,
            require_support_release_latch_at_endpoint=bool(
                getattr(
                    context,
                    "require_support_release_latch_at_endpoint",
                    True,
                )
            ),
            path_clearance_margin=path_margin,
        ),
    )


def _plan_tcp_motion_stage(
    arm,
    tcp_pose,
    q_previous,
    context,
    args,
    label,
    fixed_joints=None,
    **hooks,
):
    """Return the first locally valid stage for legacy/direct callers."""
    options = _plan_tcp_motion_stage_options(
        arm,
        tcp_pose,
        q_previous,
        context,
        args,
        label,
        fixed_joints=fixed_joints,
        max_valid_options=1,
        **hooks,
    )
    return options[0]


def _plan_tcp_stage_sequence_beam(
    arm,
    stage_specs,
    q_start,
    args,
    label,
    beam_width=None,
    stage_option_cache=None,
    *,
    plan_tcp_motion_stage_options=None,
):
    """Search collision-valid IK branches across an ordered TCP sequence.

    ``stage_specs`` contains dictionaries with ``tcp_pose``, ``context`` and
    ``label``.  A spec may additionally provide ``fixed_joints`` and a
    ``post_validate(q_previous, stage)`` callback for endpoint-only phase
    policies.  The result is ordered best-first as ``(stages, score)`` pairs;
    callers may try a later complete branch if release/return validation makes
    the first core branch unusable.
    """
    plan_options = (
        _plan_tcp_motion_stage_options
        if plan_tcp_motion_stage_options is None
        else plan_tcp_motion_stage_options
    )
    specs = tuple(stage_specs)
    if len(specs) == 0:
        return ((tuple(), 0.0),)
    configured_width = getattr(
        args,
        "grasp_stage_beam_width",
        getattr(args, "grasp_beam_width", 6),
    )
    width = max(
        int(configured_width if beam_width is None else beam_width),
        1,
    )
    dedup_rad = np.deg2rad(
        max(float(getattr(args, "ik_solution_dedup_deg", 0.5)), 0.0)
    )
    joint7_weight = max(float(getattr(args, "ik_joint7_weight", 0.35)), 0.0)
    beam = [
        {
            "stages": tuple(),
            "joints": np.asarray(q_start, dtype=float).reshape(7).copy(),
            "score": 0.0,
        }
    ]

    for stage_index, spec in enumerate(specs, start=1):
        if not isinstance(spec, dict):
            raise TypeError("beam stage specifications must be dictionaries")
        tcp_pose = np.asarray(spec["tcp_pose"], dtype=float).reshape(6)
        context = spec["context"]
        stage_label = str(spec.get("label", f"{label}_stage{stage_index}"))
        fixed_joints = spec.get("fixed_joints", None)
        post_validate = spec.get("post_validate", None)
        expanded = []
        failures = []
        for parent_index, parent in enumerate(beam, start=1):
            parent_q = np.asarray(parent["joints"], dtype=float).reshape(7)
            branch_label = (
                stage_label
                if len(beam) == 1
                else f"{stage_label}_beam_parent{parent_index}"
            )
            try:
                cache_key = None
                if stage_option_cache is not None:
                    fixed_key = (
                        None
                        if fixed_joints is None
                        else tuple(
                            np.round(
                                np.asarray(fixed_joints, dtype=float).reshape(7),
                                9,
                            ).tolist()
                        )
                    )
                    cache_key = (
                        str(context.phase),
                        tuple(np.round(tcp_pose, 9).tolist()),
                        tuple(np.round(parent_q, 9).tolist()),
                        fixed_key,
                        None
                        if context.min_tcp_z is None
                        else round(float(context.min_tcp_z), 9),
                    )
                cached_options = (
                    None
                    if cache_key is None
                    else stage_option_cache.get(cache_key)
                )
                failure_key = (
                    None
                    if cache_key is None
                    else ("stage_failure",) + cache_key
                )
                cached_failure = (
                    None
                    if failure_key is None
                    else stage_option_cache.get(failure_key)
                )
                if cached_failure is not None:
                    print(
                        f"[{branch_label}][ik][cache] reused collision/IK "
                        "stage failure"
                    )
                    raise PlanNotExecutable(str(cached_failure))
                if cached_options is None:
                    try:
                        options = plan_options(
                            arm,
                            tcp_pose,
                            parent_q,
                            context,
                            args,
                            branch_label,
                            fixed_joints=fixed_joints,
                        )
                    except (PlanNotExecutable, UnsafeMotionTarget) as exc:
                        if failure_key is not None:
                            stage_option_cache[failure_key] = str(exc)
                        raise
                    if cache_key is not None:
                        stage_option_cache[cache_key] = tuple(options)
                else:
                    options = tuple(
                        replace(stage, label=branch_label)
                        for stage in cached_options
                    )
                    print(
                        f"[{branch_label}][ik][cache] reused "
                        f"{len(options)} collision-checked stage option(s)"
                    )
            except (PlanNotExecutable, UnsafeMotionTarget) as exc:
                failures.append(
                    f"parent={parent_index}:{type(exc).__name__}:{exc}"
                )
                if _invariant_attached_target_table_start_failure(exc):
                    print(
                        f"[{stage_label}][beam][abort] attached-target table "
                        "failure is invariant at sample 0; skip remaining "
                        "beam parents"
                    )
                    raise
                if _invariant_support_release_start_failure(exc):
                    print(
                        f"[{stage_label}][beam][abort] closed-tool support "
                        "release collision is invariant near the shared "
                        "grasp start; skip remaining beam parents"
                    )
                    raise
                continue
            for option_index, stage in enumerate(options, start=1):
                try:
                    if callable(post_validate):
                        post_validate(parent_q, stage)
                except (PlanNotExecutable, UnsafeMotionTarget) as exc:
                    failures.append(
                        f"parent={parent_index}/option={option_index}:"
                        f"{type(exc).__name__}:{exc}"
                    )
                    continue
                q_next = np.asarray(stage.joints, dtype=float).reshape(7)
                delta = q_next - parent_q
                stage_joint7_weight = joint7_weight
                if "lift" in stage_label.lower():
                    stage_joint7_weight = max(
                        float(
                            getattr(
                                args,
                                "ik_lift_joint7_weight",
                                joint7_weight,
                            )
                        ),
                        0.0,
                    )
                transition_cost = float(np.linalg.norm(delta)) + (
                    stage_joint7_weight * abs(float(delta[6]))
                )
                expanded.append(
                    {
                        "stages": parent["stages"] + (stage,),
                        "joints": q_next.copy(),
                        "score": float(parent["score"]) + transition_cost,
                    }
                )
        if len(expanded) == 0:
            sample = " | ".join(failures[-6:]) or "no stage options"
            raise PlanNotExecutable(
                f"{label} beam exhausted at stage {stage_index}/{len(specs)} "
                f"({stage_label}): {sample}"
            )

        expanded.sort(key=lambda item: float(item["score"]))
        distinct = []
        for state in expanded:
            q = np.asarray(state["joints"], dtype=float).reshape(7)
            if any(
                float(np.max(np.abs(q - old["joints"]))) <= dedup_rad
                for old in distinct
            ):
                continue
            distinct.append(state)
            if len(distinct) >= width:
                break
        beam = distinct
        print(
            f"[{label}][beam] stage={stage_index}/{len(specs)} "
            f"name={stage_label} expanded={len(expanded)} kept={len(beam)}"
        )

    print(
        f"[{label}][beam] complete branches={len(beam)} "
        f"best_score={float(beam[0]['score']):.4f}"
    )
    return tuple(
        (tuple(state["stages"]), float(state["score"])) for state in beam
    )

def _motion_route_height_candidates(start_tcp, goal_tcp, scene, args):
    start_tcp = np.asarray(start_tcp, dtype=float).reshape(6)
    goal_tcp = np.asarray(goal_tcp, dtype=float).reshape(6)
    environment = np.asarray(scene.environment_base, dtype=float).reshape(-1, 3)
    scene_top = (
        float(np.percentile(environment[:, 2], 97.5))
        if environment.shape[0] > 0
        else max(float(start_tcp[2]), float(goal_tcp[2]))
    )
    vertical_clearance = max(
        float(getattr(args, "motion_transit_vertical_clearance", 0.080)), 0.0
    )
    desired = max(
        float(start_tcp[2]),
        float(goal_tcp[2]) + vertical_clearance,
        scene_top + vertical_clearance,
    )
    configured_max = float(getattr(args, "motion_transit_max_tcp_z", 0.50))
    hard_max = max(configured_max, float(start_tcp[2]), float(goal_tcp[2]))
    desired = min(desired, hard_max)
    minimum = min(
        desired,
        max(float(start_tcp[2]), float(goal_tcp[2]) + 0.020),
    )
    step = max(float(getattr(args, "motion_transit_height_step", 0.020)), 0.005)
    heights = []
    value = desired
    while value >= minimum - 1e-9:
        heights.append(float(value))
        value -= step
    if not heights or abs(heights[-1] - minimum) > 1e-6:
        heights.append(float(minimum))
    return heights, scene_top

def _plan_up_and_over_route(
    arm,
    start_tcp,
    goal_tcp,
    q_start,
    scene,
    args,
    label,
    transit_phase,
    goal_phase=None,
    attached_target_tcp=None,
    placed_target_base=None,
    first_phase=None,
    goal_joints=None,
    *,
    motion_route_height_candidates=None,
    motion_context=None,
    plan_tcp_motion_stage=None,
    interpolate_pose_fn=None,
    pose_rpy_error_norm=None,
):
    """Search highest feasible up/orient/horizontal/down joint route."""
    motion_route_height_candidates = (
        _motion_route_height_candidates
        if motion_route_height_candidates is None
        else motion_route_height_candidates
    )
    motion_context = _motion_context if motion_context is None else motion_context
    plan_tcp_motion_stage = (
        _plan_tcp_motion_stage
        if plan_tcp_motion_stage is None
        else plan_tcp_motion_stage
    )
    interpolate_pose_fn = (
        interpolate_pose if interpolate_pose_fn is None else interpolate_pose_fn
    )
    pose_rpy_error_norm = (
        _pose_rpy_error_norm
        if pose_rpy_error_norm is None
        else pose_rpy_error_norm
    )

    start_tcp = np.asarray(start_tcp, dtype=float).reshape(6)
    goal_tcp = np.asarray(goal_tcp, dtype=float).reshape(6)
    q_start = np.asarray(q_start, dtype=float).reshape(7)
    goal_phase = str(goal_phase or transit_phase)
    first_phase = str(first_phase or transit_phase)
    heights, scene_top = motion_route_height_candidates(
        start_tcp, goal_tcp, scene, args
    )
    failures = []
    for high_z in heights:
        try:
            stages = []
            q_previous = q_start

            def add_stage(tcp, suffix, phase, fixed=None, min_path_z=None):
                nonlocal q_previous
                context = motion_context(
                    scene,
                    str(phase),
                    attached_target_tcp=attached_target_tcp,
                    placed_target_base=placed_target_base,
                    min_tcp_z=min_path_z,
                    closed_gripper_width=getattr(
                        args, "_planning_closed_gripper_width", None
                    ),
                )
                stage = plan_tcp_motion_stage(
                    arm,
                    tcp,
                    q_previous,
                    context,
                    args,
                    f"{label}_{suffix}",
                    fixed_joints=fixed,
                )
                stages.append(stage)
                q_previous = np.asarray(stage.joints, dtype=float).reshape(7)

            def add_adaptive_transition(
                pose_from,
                pose_to,
                suffix,
                phase,
                fixed_final=None,
                min_path_z=None,
            ):
                """Slerp/subdivide until every seeded TRAC-IK edge is safe."""
                nonlocal q_previous
                stage_count_before = len(stages)
                q_before = q_previous.copy()
                max_segments = max(
                    int(getattr(args, "motion_transition_max_segments", 8)),
                    1,
                )
                segment_counts = []
                count = 1
                while count <= max_segments:
                    segment_counts.append(count)
                    count *= 2
                if segment_counts[-1] != max_segments and max_segments > 1:
                    segment_counts.append(max_segments)
                transition_failures = []
                for segment_count in segment_counts:
                    del stages[stage_count_before:]
                    q_previous = q_before.copy()
                    try:
                        for segment_index in range(1, segment_count + 1):
                            fraction = float(segment_index) / float(segment_count)
                            waypoint = (
                                np.asarray(pose_to, dtype=float).reshape(6)
                                if segment_index == segment_count
                                else interpolate_pose_fn(pose_from, pose_to, fraction)
                            )
                            waypoint_suffix = (
                                str(suffix)
                                if segment_count == 1
                                else f"{suffix}_{segment_index}of{segment_count}"
                            )
                            add_stage(
                                waypoint,
                                waypoint_suffix,
                                phase,
                                fixed=(
                                    fixed_final
                                    if segment_index == segment_count
                                    else None
                                ),
                                min_path_z=min_path_z,
                            )
                        if segment_count > 1:
                            print(
                                f"[{label}_{suffix}][route] adaptive transition "
                                f"selected {segment_count} Slerp segments"
                            )
                        return
                    except (PlanNotExecutable, UnsafeMotionTarget) as exc:
                        transition_failures.append(
                            f"segments={segment_count}:{type(exc).__name__}:{exc}"
                        )
                del stages[stage_count_before:]
                q_previous = q_before.copy()
                raise PlanNotExecutable(
                    f"{label}_{suffix} adaptive transition failed: "
                    + " | ".join(transition_failures[-3:])
                )

            current_high = start_tcp.copy()
            current_high[2] = high_z
            source_high_added = False
            if float(np.linalg.norm(current_high[:3] - start_tcp[:3])) > 0.010:
                add_adaptive_transition(
                    start_tcp,
                    current_high,
                    "source_high",
                    first_phase,
                )
                source_high_added = True

            # The joint state must always match the pose used as the start of
            # the next interpolation.  When the raise is below the 10 mm
            # waypoint threshold, q_previous still represents start_tcp, not
            # current_high.
            source_high_pose = (
                current_high.copy() if source_high_added else start_tcp.copy()
            )
            high_path_floor = high_z - max(
                float(getattr(args, "motion_high_path_z_tolerance", 0.020)),
                0.0,
            )
            goal_high = goal_tcp.copy()
            goal_high[2] = high_z
            source_stage_count = len(stages)
            q_at_source_high = q_previous.copy()
            topology_failures = []

            def transition_needed(pose_from, pose_to):
                return bool(
                    float(
                        np.linalg.norm(
                            np.asarray(pose_to, dtype=float)[:3]
                            - np.asarray(pose_from, dtype=float)[:3]
                        )
                    )
                    > 0.010
                    or pose_rpy_error_norm(pose_from, pose_to)
                    > np.deg2rad(3.0)
                )

            # A fixed-XYZ wrist rotation can be outside the reachable
            # orientation fibre even when both endpoint poses are reachable.
            # Search multiple safe high-level topologies instead of weakening
            # IK or collision limits:
            #   1) rotate above the source, then traverse (legacy preference),
            #   2) traverse with the source orientation, then rotate above goal,
            #   3) translate and Slerp together along the high corridor.
            for topology in (
                "orient_then_translate",
                "translate_then_orient",
                "combined_high",
            ):
                del stages[source_stage_count:]
                q_previous = q_at_source_high.copy()
                active_high = source_high_pose.copy()
                try:
                    if topology == "orient_then_translate":
                        oriented_source = source_high_pose.copy()
                        oriented_source[3:] = goal_tcp[3:]
                        if transition_needed(active_high, oriented_source):
                            add_adaptive_transition(
                                active_high,
                                oriented_source,
                                "orient_high",
                                transit_phase,
                                min_path_z=high_path_floor,
                            )
                            active_high = oriented_source
                        if transition_needed(active_high, goal_high):
                            add_adaptive_transition(
                                active_high,
                                goal_high,
                                "destination_high",
                                transit_phase,
                                min_path_z=high_path_floor,
                            )
                            active_high = goal_high
                    elif topology == "translate_then_orient":
                        translated_high = goal_high.copy()
                        translated_high[3:] = source_high_pose[3:]
                        if transition_needed(active_high, translated_high):
                            add_adaptive_transition(
                                active_high,
                                translated_high,
                                "destination_high_source_orient",
                                transit_phase,
                                min_path_z=high_path_floor,
                            )
                            active_high = translated_high
                        if transition_needed(active_high, goal_high):
                            add_adaptive_transition(
                                active_high,
                                goal_high,
                                "orient_destination_high",
                                transit_phase,
                                min_path_z=high_path_floor,
                            )
                            active_high = goal_high
                    else:
                        if transition_needed(active_high, goal_high):
                            add_adaptive_transition(
                                active_high,
                                goal_high,
                                "combined_high",
                                transit_phase,
                                min_path_z=high_path_floor,
                            )
                            active_high = goal_high

                    final_pos_error = float(
                        np.linalg.norm(goal_tcp[:3] - active_high[:3])
                    )
                    final_rpy_error = pose_rpy_error_norm(
                        active_high, goal_tcp
                    )
                    if (
                        final_pos_error > 1e-6
                        or final_rpy_error > 1e-6
                        or goal_joints is not None
                    ):
                        add_adaptive_transition(
                            active_high,
                            goal_tcp,
                            "goal",
                            goal_phase,
                            fixed_final=goal_joints,
                        )
                    elif not stages:
                        add_stage(
                            goal_tcp,
                            "goal",
                            goal_phase,
                            fixed=goal_joints,
                        )

                    print(
                        f"[{label}][route] selected up-and-over height={high_z:.3f}m "
                        f"topology={topology} scene_top_p97.5={scene_top:.3f}m "
                        f"stages={len(stages)}"
                    )
                    return tuple(stages)
                except (PlanNotExecutable, UnsafeMotionTarget) as exc:
                    failure = f"{topology}:{type(exc).__name__}:{exc}"
                    topology_failures.append(failure)
                    print(f"[{label}][route][topology_reject] {failure}")

            del stages[source_stage_count:]
            q_previous = q_at_source_high.copy()
            raise PlanNotExecutable(
                f"all high-level route topologies failed at z={high_z:.3f}: "
                + " | ".join(topology_failures[-3:])
            )
        except (PlanNotExecutable, UnsafeMotionTarget) as exc:
            failures.append(f"z={high_z:.3f}:{type(exc).__name__}:{exc}")
            print(f"[{label}][route][reject] {failures[-1]}")
    raise PlanNotExecutable(
        f"{label} has no collision-free up-and-over route: "
        + " | ".join(failures[-4:])
    )

def _plan_guarded_route(
    arm,
    start_tcp,
    goal_tcp,
    q_start,
    scene,
    args,
    label,
    transit_phase,
    goal_phase=None,
    attached_target_tcp=None,
    placed_target_base=None,
    first_phase=None,
    goal_joints=None,
    *,
    plan_up_and_over_route=None,
    motion_context=None,
    plan_tcp_motion_stage=None,
    validate_joint_path=None,
):
    """Choose either the full high route or a collision-checked direct route.

    Direct mode removes every scene-height/source-high/destination-high
    waypoint.  It still validates the exact move_j sweep against the complete
    scene, payload, table and joint limits.  A short local release retreat is
    retained only when leaving a newly placed object; it is not a high transit
    route.
    """
    plan_up_and_over_route = (
        _plan_up_and_over_route
        if plan_up_and_over_route is None
        else plan_up_and_over_route
    )
    motion_context = _motion_context if motion_context is None else motion_context
    plan_tcp_motion_stage = (
        _plan_tcp_motion_stage
        if plan_tcp_motion_stage is None
        else plan_tcp_motion_stage
    )
    validate_joint_path = (
        _collision_function("validate_motion_joint_path")
        if validate_joint_path is None
        else validate_joint_path
    )

    # A released target no longer constrains the arm, and the observation/home
    # joint target is already known.  Validate that exact joint sweep first;
    # do not spend Cartesian IK on an endpoint whose joints are supplied.  A
    # configured guarded/high route remains the fail-closed fallback.
    if bool(getattr(args, "motion_up_and_over", False)) and goal_joints is not None:
        direct_args = copy(args)
        direct_args.motion_up_and_over = False
        try:
            return _plan_guarded_route(
                arm,
                start_tcp,
                goal_tcp,
                q_start,
                scene,
                direct_args,
                label,
                transit_phase,
                goal_phase=goal_phase,
                attached_target_tcp=attached_target_tcp,
                placed_target_base=placed_target_base,
                first_phase=first_phase,
                goal_joints=goal_joints,
                plan_up_and_over_route=plan_up_and_over_route,
                motion_context=motion_context,
                plan_tcp_motion_stage=plan_tcp_motion_stage,
                validate_joint_path=validate_joint_path,
            )
        except (PlanNotExecutable, UnsafeMotionTarget) as exc:
            print(
                f"[{label}][route] direct fixed-joint return rejected: "
                f"{type(exc).__name__}: {exc}; try guarded route"
            )

    if bool(getattr(args, "motion_up_and_over", False)):
        return plan_up_and_over_route(
            arm,
            start_tcp,
            goal_tcp,
            q_start,
            scene,
            args,
            label,
            transit_phase=transit_phase,
            goal_phase=goal_phase,
            attached_target_tcp=attached_target_tcp,
            placed_target_base=placed_target_base,
            first_phase=first_phase,
            goal_joints=goal_joints,
        )

    start_tcp = np.asarray(start_tcp, dtype=float).reshape(6)
    goal_tcp = np.asarray(goal_tcp, dtype=float).reshape(6)
    q_previous = np.asarray(q_start, dtype=float).reshape(7)
    stages = []
    transit_phase = str(transit_phase)
    goal_phase = str(goal_phase or transit_phase)

    if first_phase is not None and str(first_phase) != transit_phase:
        retreat_context = motion_context(
            scene,
            str(first_phase),
            attached_target_tcp=attached_target_tcp,
            placed_target_base=placed_target_base,
            closed_gripper_width=getattr(
                args, "_planning_closed_gripper_width", None
            ),
        )
        # Contact permission is local to the release.  Before the long direct
        # edge begins, the retreat endpoint must satisfy the strict free phase.
        strict_context = motion_context(
            scene,
            transit_phase,
            attached_target_tcp=attached_target_tcp,
            placed_target_base=placed_target_base,
            closed_gripper_width=getattr(
                args, "_planning_closed_gripper_width", None
            ),
        )
        rise = max(
            float(getattr(args, "motion_direct_release_retreat", 0.020)),
            0.010,
        )
        xy_relax = max(
            float(getattr(args, "motion_release_retreat_xy_relax", 0.010)),
            0.0,
        )
        tilt_deg = max(
            float(
                getattr(args, "motion_release_retreat_tilt_relax_deg", 5.0)
            ),
            0.0,
        )
        max_attempts = max(
            int(getattr(args, "motion_release_retreat_max_attempts", 9)), 1
        )

        retreat_variants = []

        def add_retreat_variant(name, pose):
            pose = np.asarray(pose, dtype=float).reshape(6)
            key = tuple(np.round(pose, 9).tolist())
            if not any(existing[2] == key for existing in retreat_variants):
                retreat_variants.append((str(name), pose.copy(), key))

        exact_retreat = start_tcp.copy()
        exact_retreat[2] += rise
        add_retreat_variant("exact", exact_retreat)
        if xy_relax > 1e-9:
            for name, dx, dy in (
                ("x_plus", xy_relax, 0.0),
                ("x_minus", -xy_relax, 0.0),
                ("y_plus", 0.0, xy_relax),
                ("y_minus", 0.0, -xy_relax),
            ):
                shifted = exact_retreat.copy()
                shifted[0] += dx
                shifted[1] += dy
                add_retreat_variant(name, shifted)
        if tilt_deg > 1e-9:
            approach_name = str(getattr(args, "tool_approach_axis", "x"))
            approach_axis = approach_name[-1]
            tilt_axes = [axis for axis in ("x", "y", "z") if axis != approach_axis]
            base_rotation = R.from_euler("xyz", exact_retreat[3:]).as_matrix()
            axis_vectors = {
                "x": np.array([1.0, 0.0, 0.0]),
                "y": np.array([0.0, 1.0, 0.0]),
                "z": np.array([0.0, 0.0, 1.0]),
            }
            for axis in tilt_axes:
                for sign in (1.0, -1.0):
                    tilted = exact_retreat.copy()
                    delta = R.from_rotvec(
                        axis_vectors[axis] * np.deg2rad(sign * tilt_deg)
                    ).as_matrix()
                    tilted[3:] = R.from_matrix(base_rotation @ delta).as_euler(
                        "xyz"
                    )
                    add_retreat_variant(
                        f"tilt_{axis}_{'plus' if sign > 0.0 else 'minus'}",
                        tilted,
                    )

        retreat_stage = None
        retreat_failures = []
        for variant_index, (variant_name, retreat_tcp, _key) in enumerate(
            retreat_variants[:max_attempts], start=1
        ):
            variant_label = f"{label}_direct_release_retreat"
            if variant_index > 1:
                variant_label += f"_{variant_name}"
            try:
                candidate_stage = plan_tcp_motion_stage(
                    arm,
                    retreat_tcp,
                    q_previous,
                    retreat_context,
                    args,
                    variant_label,
                )
                validate_joint_path(
                    candidate_stage.joints,
                    candidate_stage.joints,
                    strict_context,
                    args,
                    f"{variant_label}_strict_endpoint",
                )
                retreat_stage = candidate_stage
                if variant_index > 1:
                    delta_xy = retreat_tcp[:2] - start_tcp[:2]
                    print(
                        f"[{label}][release-retreat] selected bounded fallback "
                        f"variant={variant_name} rise={rise * 1000.0:.0f}mm "
                        f"xy=[{delta_xy[0] * 1000.0:.0f},"
                        f"{delta_xy[1] * 1000.0:.0f}]mm "
                        f"tilt_limit={tilt_deg:.1f}deg"
                    )
                break
            except (PlanNotExecutable, UnsafeMotionTarget) as exc:
                retreat_failures.append(
                    f"{variant_name}:{type(exc).__name__}:{exc}"
                )
        if retreat_stage is None:
            raise PlanNotExecutable(
                f"{label} no bounded +{rise * 1000.0:.0f}mm release-retreat "
                f"variant passed IK/collision validation: "
                + " | ".join(retreat_failures[-4:])
            )
        stages.append(retreat_stage)
        q_previous = np.asarray(retreat_stage.joints, dtype=float).reshape(7)

    direct_context = motion_context(
        scene,
        transit_phase,
        attached_target_tcp=attached_target_tcp,
        placed_target_base=placed_target_base,
        closed_gripper_width=getattr(
            args, "_planning_closed_gripper_width", None
        ),
    )
    direct_stage = plan_tcp_motion_stage(
        arm,
        goal_tcp,
        q_previous,
        direct_context,
        args,
        f"{label}_direct",
        fixed_joints=goal_joints,
    )
    stages.append(direct_stage)

    if goal_phase != transit_phase:
        goal_context = motion_context(
            scene,
            goal_phase,
            attached_target_tcp=attached_target_tcp,
            placed_target_base=placed_target_base,
            closed_gripper_width=getattr(
                args, "_planning_closed_gripper_width", None
            ),
        )
        validate_joint_path(
            direct_stage.joints,
            direct_stage.joints,
            goal_context,
            args,
            f"{label}_direct_goal_policy",
        )
    print(
        f"[{label}][route] guarded-direct selected; high waypoints disabled "
        f"stages={len(stages)}"
    )
    return tuple(stages)

def _modeled_tcp_pose_for_joints(
    joints,
    args,
    *,
    urdf_chain_geometry=None,
    as_pose_list=None,
    make_T_from_pose=None,
):
    urdf_chain_geometry = (
        _collision_function("_urdf_chain_geometry")
        if urdf_chain_geometry is None
        else urdf_chain_geometry
    )
    as_pose_list = _as_pose_list if as_pose_list is None else as_pose_list
    make_T_from_pose = (
        _make_T_from_pose if make_T_from_pose is None else make_T_from_pose
    )
    _, T_base_flange = urdf_chain_geometry(joints, args)
    return as_pose_list(
        T_base_flange
        @ make_T_from_pose(np.asarray(args.tcp_offset, dtype=float).reshape(6))
    )
