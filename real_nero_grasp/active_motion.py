"""Collision-guarded active-observation motion planning.

Dependencies with mutable caches or higher-level planning ownership are
resolved lazily, while keyword-only hooks keep the module testable and avoid
cycles with the legacy runtime facade.
"""

import copy

import numpy as np

from .transforms import _as_pose_list, _make_T_from_pose
from .types import (
    GraspMotionPlan,
    MotionSafetyScene,
    PlanNotExecutable,
    PlannedJointStage,
    UnsafeMotionTarget,
)


__all__ = [
    "_tcp_pose_for_flange_pose",
    "_prepare_active_route_safety_scene",
    "_subdivide_active_direct_stage",
    "plan_active_direct_route",
    "plan_active_up_and_over_route",
]


def _resolve_collision_hook(name, hook):
    if hook is not None:
        return hook
    from . import collision

    return getattr(collision, name)


def _resolve_motion_planning_hook(name, hook):
    if hook is not None:
        return hook
    from . import motion_planning

    return getattr(motion_planning, name)


def _resolve_ik_hook(name, hook):
    if hook is not None:
        return hook
    from . import ik_solver

    return getattr(ik_solver, name)


def _tcp_pose_for_flange_pose(flange_pose, args):
    """Apply the configured flange->TCP transform without querying hardware."""
    return _as_pose_list(
        _make_T_from_pose(np.asarray(flange_pose, dtype=float).reshape(6))
        @ _make_T_from_pose(np.asarray(args.tcp_offset, dtype=float).reshape(6))
    )


def _prepare_active_route_safety_scene(
    q_start,
    scene_points_base,
    target_points_base,
    args,
    label,
    scene_capture_joints=None,
    *,
    bounded_motion_points_hook=None,
    filter_motion_robot_self_points_hook=None,
):
    """Build the frozen full-scene map shared by all active route modes."""
    bounded_motion_points_fn = _resolve_collision_hook(
        "_bounded_motion_points", bounded_motion_points_hook
    )
    filter_self_points_fn = _resolve_collision_hook(
        "filter_motion_robot_self_points",
        filter_motion_robot_self_points_hook,
    )
    target_points = bounded_motion_points_fn(target_points_base, args)
    full_scene = bounded_motion_points_fn(scene_points_base, args)
    self_filter_joints = (
        np.asarray(q_start, dtype=float).reshape(7)
        if scene_capture_joints is None
        else np.asarray(scene_capture_joints, dtype=float).reshape(7)
    )
    # The measured 2-D calibration covers one visible finger. Preserve the
    # existing geometric filter for other visible robot parts as well.
    if (not bool(getattr(args, "camera_self_mask", True))
            or getattr(args, "camera_self_mask_calibration", None)):
        full_scene = filter_self_points_fn(
            full_scene,
            self_filter_joints,
            args,
            f"{label}_scene",
        )
    # A geometric self-filter is allowed to remove camera-visible robot points,
    # but never the explicitly segmented object that active observation must
    # avoid.  Re-add it after filtering, as the grasp-workflow scene does.
    full_scene = bounded_motion_points_fn(
        np.vstack([full_scene, target_points]), args
    )
    if full_scene.shape[0] == 0 and bool(
        getattr(args, "active_path_require_scene", True)
    ):
        raise PlanNotExecutable(
            f"{label} has no complete base-frame scene for active motion"
        )
    safety_scene = MotionSafetyScene(
        full_scene_base=full_scene,
        # free_open uses full_scene_base.  Keeping the same cloud here also
        # makes the structure safe if a future active stage changes phase.
        environment_base=full_scene.copy(),
        target_base=target_points,
        map_id=f"{label}_active_preflight",
    )
    return target_points, full_scene, safety_scene


def _subdivide_active_direct_stage(
    q_start,
    stage,
    args,
    *,
    urdf_chain_geometry_hook=None,
    tcp_pose_for_flange_pose_hook=None,
):
    """Optionally turn one validated return edge into small joint commands.

    The collision validator samples a linear joint edge, but Nero ``move_j``
    receives only one endpoint and may progress the seven axes at different
    rates.  Active-view returns enable these micro-waypoints by default so the
    measured multi-axis motion stays closer to the collision-checked linear
    joint edge.
    """
    urdf_chain_geometry_fn = _resolve_collision_hook(
        "_urdf_chain_geometry", urdf_chain_geometry_hook
    )
    tcp_pose_for_flange_pose_fn = (
        _tcp_pose_for_flange_pose
        if tcp_pose_for_flange_pose_hook is None
        else tcp_pose_for_flange_pose_hook
    )
    q_start = np.asarray(q_start, dtype=float).reshape(7)
    q_goal = np.asarray(stage.joints, dtype=float).reshape(7)
    delta = q_goal - q_start
    waypoint_step_deg = float(
        getattr(args, "active_return_execution_waypoint_deg", 0.0)
    )
    if waypoint_step_deg <= 0.0:
        return (stage,)
    waypoint_step_deg = max(waypoint_step_deg, 1.0)
    count = max(
        1,
        int(
            np.ceil(
                float(np.rad2deg(np.max(np.abs(delta)))) / waypoint_step_deg
            )
        ),
    )
    if count == 1:
        return (stage,)
    stages = []
    for index in range(1, count + 1):
        if index == count:
            stages.append(stage)
            continue
        alpha = float(index) / float(count)
        joints = q_start + alpha * delta
        _, T_base_flange = urdf_chain_geometry_fn(joints, args)
        flange_pose = _as_pose_list(T_base_flange)
        if args.min_flange_z is not None and float(flange_pose[2]) < float(
            args.min_flange_z
        ):
            raise UnsafeMotionTarget(
                f"{stage.label} intermediate flange z={float(flange_pose[2]):.4f}m "
                f"below --min-flange-z={float(args.min_flange_z):.4f}m"
            )
        tcp_pose = tcp_pose_for_flange_pose_fn(flange_pose, args)
        stages.append(
            PlannedJointStage(
                label=f"{stage.label}_waypoint_{index}of{count}",
                tcp_pose=np.asarray(tcp_pose, dtype=float).reshape(6),
                flange_pose=np.asarray(flange_pose, dtype=float).reshape(6),
                joints=np.asarray(joints, dtype=float).reshape(7),
                phase=str(stage.phase),
                min_tcp_z=stage.min_tcp_z,
                runtime_min_tcp_z_slack=stage.runtime_min_tcp_z_slack,
            )
        )
    return tuple(stages)


def plan_active_direct_route(
    arm,
    goal_flange,
    q_start,
    scene_points_base,
    target_points_base,
    args,
    label,
    max_joint_delta_deg,
    max_joint_l2_deg,
    goal_joints=None,
    scene_capture_joints=None,
    initial_retreat_height=0.0,
    enforce_single_stage_return=False,
    *,
    prepare_active_route_safety_scene_hook=None,
    tcp_pose_for_flange_pose_hook=None,
    modeled_tcp_pose_for_joints_hook=None,
    motion_context_hook=None,
    plan_tcp_motion_stage_hook=None,
    subdivide_active_direct_stage_hook=None,
    validate_motion_joint_path_hook=None,
    get_ik_solver_hook=None,
    joint_limit_margin_hook=None,
):
    """Plan one guarded IK-joint edge directly to an observation pose.

    This deliberately omits the extra source-high/orient-high/destination-high
    waypoints.  It is not a legacy unguarded move: the complete joint sweep is
    checked with the canonical segmented arm/camera/gripper collision model,
    then executed with the same frozen runtime map.  The obsolete 90 mm active
    gripper sphere is intentionally not applied a second time here.
    """
    prepare_scene_fn = (
        _prepare_active_route_safety_scene
        if prepare_active_route_safety_scene_hook is None
        else prepare_active_route_safety_scene_hook
    )
    tcp_pose_fn = (
        _tcp_pose_for_flange_pose
        if tcp_pose_for_flange_pose_hook is None
        else tcp_pose_for_flange_pose_hook
    )
    modeled_tcp_fn = _resolve_motion_planning_hook(
        "_modeled_tcp_pose_for_joints", modeled_tcp_pose_for_joints_hook
    )
    motion_context_fn = _resolve_motion_planning_hook(
        "_motion_context", motion_context_hook
    )
    plan_stage_fn = _resolve_motion_planning_hook(
        "_plan_tcp_motion_stage", plan_tcp_motion_stage_hook
    )
    subdivide_fn = (
        _subdivide_active_direct_stage
        if subdivide_active_direct_stage_hook is None
        else subdivide_active_direct_stage_hook
    )
    validate_path_fn = _resolve_collision_hook(
        "validate_motion_joint_path", validate_motion_joint_path_hook
    )
    get_solver_fn = _resolve_ik_hook("_get_ik_solver", get_ik_solver_hook)
    joint_margin_fn = _resolve_ik_hook(
        "_joint_limit_margin", joint_limit_margin_hook
    )
    required_margin_fn = _resolve_motion_planning_hook(
        "_required_joint_limit_margin_deg", None
    )
    q_start = np.asarray(q_start, dtype=float).reshape(7)
    _, full_scene, safety_scene = prepare_scene_fn(
        q_start,
        scene_points_base,
        target_points_base,
        args,
        label,
        scene_capture_joints=scene_capture_joints,
    )
    goal_tcp = tcp_pose_fn(goal_flange, args)
    retreat_height = max(float(initial_retreat_height), 0.0)
    start_tcp = modeled_tcp_fn(q_start, args)
    downward_tolerance = max(
        float(
            getattr(
                args,
                "active_direct_return_downward_tolerance",
                0.002,
            )
        ),
        0.0,
    )
    runtime_downward_tolerance = max(
        float(
            getattr(
                args,
                "active_direct_return_runtime_downward_tolerance",
                0.010,
            )
        ),
        0.0,
    )
    runtime_floor_slack = (
        max(runtime_downward_tolerance - downward_tolerance, 0.0)
        if bool(enforce_single_stage_return) and retreat_height <= 0.0
        else 0.0
    )
    context = motion_context_fn(
        safety_scene,
        "free_open",
        min_tcp_z=(
            None
            if retreat_height <= 0.0 and not bool(enforce_single_stage_return)
            else float(start_tcp[2]) - downward_tolerance
        ),
        runtime_min_tcp_z_slack=runtime_floor_slack,
    )
    route_stages = []
    q_previous = q_start.copy()
    if retreat_height > 0.0:
        retreat_tcp = np.asarray(start_tcp, dtype=float).reshape(6).copy()
        retreat_tcp[2] += retreat_height
        retreat_stage = plan_stage_fn(
            arm,
            retreat_tcp,
            q_previous,
            context,
            args,
            f"{label}_retreat_up",
        )
        route_stages.extend(
            subdivide_fn(q_previous, retreat_stage, args)
        )
        q_previous = np.asarray(retreat_stage.joints, dtype=float).reshape(7)
    stage = plan_stage_fn(
        arm,
        goal_tcp,
        q_previous,
        context,
        args,
        f"{label}_direct",
        fixed_joints=goal_joints,
    )
    if bool(enforce_single_stage_return) and retreat_height <= 0.0:
        robust_clearance = max(
            float(
                getattr(
                    args,
                    "active_direct_return_min_clearance",
                    0.030,
                )
            ),
            float(getattr(args, "motion_path_min_clearance", 0.015)),
        )
        robust_args = copy.copy(args)
        robust_args.motion_path_min_clearance = robust_clearance
        validate_path_fn(
            q_start,
            np.asarray(stage.joints, dtype=float).reshape(7),
            motion_context_fn(safety_scene, "free_open"),
            robust_args,
            f"{label}_single_stage_robust_clearance",
        )
    if retreat_height > 0.0 or bool(enforce_single_stage_return):
        route_stages.extend(subdivide_fn(q_previous, stage, args))
    else:
        # Center/outbound observation moves keep their existing one-command
        # behavior.  The measured-path deviation that motivated this change
        # was specific to the long side-view return edge.
        route_stages.append(stage)
    minimum_margin_deg = required_margin_fn(
        args,
        "active_direct_min_joint_limit_margin_deg",
    )
    active_margin_deg = None
    if minimum_margin_deg > 0.0:
        active_margin_deg = min(
            float(
                np.rad2deg(
                    joint_margin_fn(
                        np.asarray(route_stage.joints, dtype=float).reshape(7),
                        get_solver_fn(args),
                    )
                )
            )
            for route_stage in route_stages
        )
        if active_margin_deg < minimum_margin_deg:
            raise PlanNotExecutable(
                f"{label} direct target joint-limit margin="
                f"{active_margin_deg:.2f}deg < required "
                f"{minimum_margin_deg:.2f}deg"
            )
    route_nodes = [q_start] + [
        np.asarray(route_stage.joints, dtype=float).reshape(7)
        for route_stage in route_stages
    ]
    route_deltas = [
        q_after - q_before
        for q_before, q_after in zip(route_nodes[:-1], route_nodes[1:])
    ]
    direct_max_deg = max(
        float(np.rad2deg(np.max(np.abs(delta)))) for delta in route_deltas
    )
    direct_l2_deg = sum(
        float(np.rad2deg(np.linalg.norm(delta))) for delta in route_deltas
    )
    if direct_max_deg > float(max_joint_delta_deg):
        raise UnsafeMotionTarget(
            f"{label} direct largest joint sweep {direct_max_deg:.2f}deg "
            f"exceeds active limit {float(max_joint_delta_deg):.2f}deg"
        )
    if direct_l2_deg > float(max_joint_l2_deg):
        raise UnsafeMotionTarget(
            f"{label} direct total joint sweep {direct_l2_deg:.2f}deg "
            f"exceeds active L2 limit {float(max_joint_l2_deg):.2f}deg"
        )
    plan = GraspMotionPlan(
        start_joints=q_start.copy(),
        before_close=tuple(route_stages),
        after_close=tuple(),
        after_release=tuple(),
        empty_recovery=tuple(),
        safety_scene=safety_scene,
    )
    print(
        f"[{label}][route] active guarded-direct edge fully validated: "
        f"joint_max={direct_max_deg:.2f}deg joint_l2={direct_l2_deg:.2f}deg"
        f" execution_waypoints={len(route_stages)}"
        + (
            (
                " single_stage_return=true"
                if len(route_stages) == 1
                else " segmented_direct_return=true"
            )
            if bool(enforce_single_stage_return) and retreat_height <= 0.0
            else ""
        )
        + (
            ""
            if retreat_height <= 0.0
            else f" retreat_up={retreat_height * 1000.0:.1f}mm"
        )
        + (
            ""
            if active_margin_deg is None
            else f" limit_margin={active_margin_deg:.2f}deg"
        )
    )
    return plan


def plan_active_up_and_over_route(
    arm,
    start_tcp,
    goal_flange,
    q_start,
    scene_points_base,
    target_points_base,
    args,
    label,
    max_joint_delta_deg,
    max_joint_l2_deg,
    goal_joints=None,
    scene_capture_joints=None,
    *,
    prepare_active_route_safety_scene_hook=None,
    tcp_pose_for_flange_pose_hook=None,
    plan_up_and_over_route_hook=None,
    validate_active_joint_path_hook=None,
):
    """Plan an active-view move as verified up/orient/across/down stages.

    Active observation has no intentional scene contact, so every robot,
    camera and open-gripper primitive is checked against the complete frozen
    RGB-D scene.  The legacy active-path capsules are also run on every edge;
    this preserves their deliberately conservative open-gripper envelope.
    """
    prepare_scene_fn = (
        _prepare_active_route_safety_scene
        if prepare_active_route_safety_scene_hook is None
        else prepare_active_route_safety_scene_hook
    )
    tcp_pose_fn = (
        _tcp_pose_for_flange_pose
        if tcp_pose_for_flange_pose_hook is None
        else tcp_pose_for_flange_pose_hook
    )
    plan_up_and_over_fn = _resolve_motion_planning_hook(
        "_plan_up_and_over_route", plan_up_and_over_route_hook
    )
    validate_active_path_fn = _resolve_collision_hook(
        "_validate_active_joint_path", validate_active_joint_path_hook
    )
    q_start = np.asarray(q_start, dtype=float).reshape(7)
    _, full_scene, safety_scene = prepare_scene_fn(
        q_start,
        scene_points_base,
        target_points_base,
        args,
        label,
        scene_capture_joints=scene_capture_joints,
    )
    goal_tcp = tcp_pose_fn(goal_flange, args)
    route = plan_up_and_over_fn(
        arm,
        np.asarray(start_tcp, dtype=float).reshape(6),
        goal_tcp,
        q_start,
        safety_scene,
        args,
        label,
        transit_phase="free_open",
        goal_phase="free_open",
        goal_joints=goal_joints,
    )

    q_previous = q_start.copy()
    for stage in route:
        validate_active_path_fn(
            q_previous,
            stage.joints,
            full_scene,
            args,
            f"{stage.label}_active_envelope",
            float(max_joint_delta_deg),
            float(max_joint_l2_deg),
        )
        q_previous = np.asarray(stage.joints, dtype=float).reshape(7)

    plan = GraspMotionPlan(
        start_joints=q_start.copy(),
        before_close=tuple(route),
        after_close=tuple(),
        after_release=tuple(),
        empty_recovery=tuple(),
        safety_scene=safety_scene,
    )
    print(
        f"[{label}][route] active up-and-over sequence fully validated: "
        f"stages={len(route)}"
    )
    return plan
