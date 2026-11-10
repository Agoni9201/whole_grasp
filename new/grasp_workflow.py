"""High-level grasp planning, confirmation, execution, and recovery workflow.

Collision models, candidate generation/filtering, IK, frozen-plan execution,
and hardware access remain in focused lower-level modules. They are injected
through GraspWorkflowHooks so the legacy facade can preserve dynamic patches
without this module importing real_nero_grasp.runtime.
"""

import copy
import os
import time
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from .grasp_planner import (
    _apply_transport_pose_fallback,
    _place_tcp_from_lift,
    _plan_grasp_pose_fallbacks,
)
from .transforms import interpolate_pose
from .stop_policy import request_ordinary_trajectory_stop
from .types import (
    ControlledMotionStopTriggered,
    GraspAttemptResult,
    GraspAttemptStatus,
    GraspNotDetected,
    GraspSelectionDeclined,
    InvalidGraspSelection,
    MotionNotReached,
    PlanNotExecutable,
    SafetyStopTriggered,
    TorqueCollisionStopTriggered,
    UnsafeMotionTarget,
)


@dataclass(frozen=True)
class GraspWorkflowHooks:
    """Explicit legacy-facade dependencies used by grasp orchestration."""

    assert_table_guard_target_tcp: Optional[Callable] = None
    move_flange_pose: Optional[Callable] = None
    move_tcp_pose: Optional[Callable] = None
    intermediate_tcp_pose: Optional[Callable] = None
    current_flange_pose_error: Optional[Callable] = None
    print_current_flange_pose_error: Optional[Callable] = None
    print_status: Optional[Callable] = None
    wait_for_joint_settle: Optional[Callable] = None
    arm_motion_status_snapshot: Optional[Callable] = None

    axis_vector: Optional[Callable] = None
    candidate_preflight_width_signs: Optional[Callable] = None
    defer_candidate_endpoint_reachability: Optional[Callable] = None
    recoverable_grasp_generation_failure: Optional[Callable] = None
    annotate_grasp_candidate_physical_aperture: Optional[Callable] = None
    annotate_grasp_candidate_reachability: Optional[Callable] = None
    annotate_grasp_candidate_workflow_preflight: Optional[Callable] = None
    assert_grasp_detected: Optional[Callable] = None
    choose_default_grasp_candidate: Optional[Callable] = None
    confirm_motion: Optional[Callable] = None
    execute_prevalidated_stages: Optional[Callable] = None
    execution_gripper_open_width: Optional[Callable] = None
    get_graspnet_helpers: Optional[Callable] = None
    grasp_camera_summary: Optional[Callable] = None
    grasp_candidate_width_shift: Optional[Callable] = None
    grasp_tcp_axis_diagnostics: Optional[Callable] = None
    grasp_to_tcp_poses: Optional[Callable] = None
    handle_motion_not_reached: Optional[Callable] = None
    move_initial: Optional[Callable] = None
    move_tcp_pose_with_intermediate: Optional[Callable] = None
    plan_up_and_over_grasp_sequence: Optional[Callable] = None
    preview_graspnet_pointcloud: Optional[Callable] = None
    recover_after_empty_grasp: Optional[Callable] = None
    recover_after_torque_stop: Optional[Callable] = None
    release_vision_model_caches: Optional[Callable] = None
    safe_stop: Optional[Callable] = None
    save_grasp_motion_plan: Optional[Callable] = None
    save_selected_grasp_pose: Optional[Callable] = None
    select_grasp_candidate: Optional[Callable] = None
    verify_empty_grasp_recovery_at_initial_observation: Optional[Callable] = None
    wait_until_grasp_close_pose_ready: Optional[Callable] = None
    wait_until_gripper_open: Optional[Callable] = None


def _require_hook(hooks, name):
    if hooks is None:
        raise RuntimeError(f"grasp-workflow dependency hook '{name}' was not supplied")
    value = getattr(hooks, name, None)
    if not callable(value):
        raise RuntimeError(f"grasp-workflow dependency hook '{name}' is unavailable")
    return value


def _prepare_graspnet_target_cloud(
    target_cloud,
    args,
    *,
    T_base_camera=None,
):
    """Remove tiny disconnected islands from GraspNet-only target input.

    The original target cloud remains available to physical-aperture and full
    motion-safety checks.  If no component clearly dominates, keep everything
    because partial visibility can legitimately split one object surface.
    """
    prepared_cloud = target_cloud
    points = np.asarray(getattr(prepared_cloud, "points", []), dtype=float)
    if points.ndim != 2 or points.shape[0] < 2:
        return prepared_cloud

    if bool(getattr(args, "graspnet_main_component_filter", True)):
        cluster_fn = getattr(prepared_cloud, "cluster_dbscan", None)
        select_fn = getattr(prepared_cloud, "select_by_index", None)
        if callable(cluster_fn) and callable(select_fn):
            eps = max(
                float(getattr(args, "graspnet_component_eps", 0.004)),
                0.001,
            )
            min_points = max(
                int(getattr(args, "graspnet_component_min_points", 8)), 2
            )
            min_dominance = float(
                np.clip(
                    getattr(args, "graspnet_component_min_dominance", 0.85),
                    0.5,
                    1.0,
                )
            )
            try:
                labels = np.asarray(
                    cluster_fn(
                        eps=eps,
                        min_points=min_points,
                        print_progress=False,
                    ),
                    dtype=np.int64,
                )
                valid = labels >= 0
                if labels.shape[0] == points.shape[0] and np.any(valid):
                    cluster_sizes = np.bincount(labels[valid])
                    largest_label = int(np.argmax(cluster_sizes))
                    keep_mask = labels == largest_label
                    largest_size = int(np.count_nonzero(keep_mask))
                    dominance = float(largest_size) / float(points.shape[0])
                    if dominance >= min_dominance:
                        prepared_cloud = select_fn(
                            np.flatnonzero(keep_mask).astype(int).tolist()
                        )
                        print(
                            "[grasp][input] dominant-component cleanup: "
                            f"kept={largest_size}/{points.shape[0]} "
                            f"({dominance:.3f}) removed_island_points="
                            f"{points.shape[0] - largest_size} "
                            f"eps={eps * 1000.0:.1f}mm "
                            f"min_points={min_points}"
                        )
                    else:
                        print(
                            "[grasp][input] target components have no "
                            f"dominant body: largest={largest_size}/"
                            f"{points.shape[0]} ({dominance:.3f}) < "
                            f"{min_dominance:.3f}; keep all points"
                        )
            except Exception as exc:
                print(
                    "[grasp][input][warn] target component cleanup "
                    f"unavailable: {type(exc).__name__}: {exc}; keep the "
                    "complete target cloud"
                )

    focus_info = {
        "enabled": bool(getattr(args, "graspnet_upper_focus", True)),
        "applied": False,
        "fallback": "missing_transform",
    }
    focus_select = getattr(prepared_cloud, "select_by_index", None)
    if T_base_camera is not None and callable(focus_select):
        from .grasp_candidates import focus_graspnet_target_indices

        prepared_points = np.asarray(prepared_cloud.points, dtype=float)
        focus_indices, focus_info = focus_graspnet_target_indices(
            prepared_points,
            T_base_camera=T_base_camera,
            enabled=bool(getattr(args, "graspnet_upper_focus", True)),
            lower_quantile=float(
                getattr(args, "graspnet_lower_target_quantile", 0.25)
            ),
            upright_lower_quantile=float(
                getattr(
                    args,
                    "graspnet_upright_lower_target_quantile",
                    0.15,
                )
            ),
            elongated_longitudinal_range=getattr(
                args, "graspnet_elongated_long_axis_range", (0.15, 0.85)
            ),
            elongated_threshold=float(
                getattr(args, "graspnet_focus_elongation_threshold", 1.5)
            ),
            upright_axis_cos=float(
                getattr(args, "graspnet_focus_upright_axis_cos", 0.75)
            ),
            min_retained_ratio=float(
                getattr(args, "graspnet_focus_min_retained_ratio", 0.50)
            ),
            min_retained_points=int(
                getattr(args, "graspnet_focus_min_retained_points", 1000)
            ),
        )
        if bool(focus_info.get("applied", False)):
            prepared_cloud = focus_select(focus_indices.astype(int).tolist())
    args._graspnet_target_focus_info = dict(focus_info)
    print(
        "[grasp][input][focus] "
        f"applied={bool(focus_info.get('applied', False))} "
        f"shape={'upright' if focus_info.get('upright', False) else ('elongated' if focus_info.get('elongated', False) else 'round/general')} "
        f"elongation={float(focus_info.get('elongation', 1.0)):.3f} "
        f"lower_excluded={float(focus_info.get('lower_quantile', 0.0)) * 100.0:.1f}% "
        f"kept={int(focus_info.get('retained_points', len(getattr(prepared_cloud, 'points', []))))}/"
        f"{int(focus_info.get('input_points', len(getattr(prepared_cloud, 'points', []))))} "
        f"fallback={focus_info.get('fallback', 'none')}"
    )
    return prepared_cloud


def _empty_recovery_execution_stages(plan, args):
    """Subdivide failure recovery so hardware follows its validated edge.

    A large Nero ``move_j`` command can advance the seven axes at different
    rates.  The collision checker validates a straight joint-space edge, so a
    one-command return may depart from that edge even though both endpoints
    are safe.  Each generated sub-edge remains part of the already validated
    frozen recovery route and still receives live preflight/runtime checks.
    """
    stages = tuple(getattr(plan, "empty_recovery", tuple()))
    if not stages:
        return tuple()
    step_deg = float(
        getattr(args, "empty_recovery_execution_waypoint_deg", 15.0)
    )
    if not np.isfinite(step_deg) or step_deg <= 0.0:
        return stages

    from .active_motion import _subdivide_active_direct_stage

    subdivision_args = copy.copy(args)
    subdivision_args.active_return_execution_waypoint_deg = step_deg
    subdivision_args.active_execution_waypoint_deg = step_deg
    # Subdivide the first grasp->pregrasp backout too.  It is commonly the
    # largest recovery edge, and one large Nero move_j can advance its seven
    # axes asynchronously enough to leave the collision-checked joint line.
    # The caller keeps the gripper closed until every piece belonging to this
    # first original stage has completed.
    expanded = []
    q_previous = np.asarray(plan.before_close[-1].joints, dtype=float).reshape(7)
    for stage in stages:
        pieces = _subdivide_active_direct_stage(
            q_previous,
            stage,
            subdivision_args,
        )
        expanded.extend(pieces)
        q_previous = np.asarray(stage.joints, dtype=float).reshape(7)
    if len(expanded) != len(stages):
        print(
            f"[return] empty-grasp recovery execution subdivided "
            f"{len(stages)} planned stage(s) into {len(expanded)} guarded "
            f"command(s), max_joint_step={step_deg:.1f}deg"
        )
    return tuple(expanded)


def _before_close_execution_stages(plan, args):
    """Keep large observation-to-grasp moves close to their validated edge.

    Nero can advance short-axis joints much faster than long-axis joints for a
    single large ``move_j`` command.  Planning validates a linear joint edge,
    so split only the frozen pre-close route into bounded joint waypoints at
    execution time.  Every sub-edge is a subset of the already validated edge
    and is live-preflighted again before being sent.
    """
    stages = tuple(getattr(plan, "before_close", tuple()))
    if not stages:
        return tuple()
    step_deg = float(
        getattr(args, "grasp_execution_waypoint_deg", 20.0)
    )
    if not np.isfinite(step_deg) or step_deg <= 0.0:
        return stages

    from .active_motion import _subdivide_active_direct_stage

    subdivision_args = copy.copy(args)
    expanded = []
    q_previous = np.asarray(plan.start_joints, dtype=float).reshape(7)
    for stage in stages:
        stage_step = _adaptive_execution_waypoint_deg(
            plan,
            stage,
            q_previous,
            args,
            base_step_deg=step_deg,
            smooth_step_deg=float(
                getattr(args, "grasp_smooth_open_waypoint_deg", 30.0)
            ),
            smooth_phases={"free_open"},
        )
        subdivision_args.active_execution_waypoint_deg = stage_step
        subdivision_args.active_return_execution_waypoint_deg = stage_step
        pieces = _subdivide_active_direct_stage(
            q_previous,
            stage,
            subdivision_args,
        )
        expanded.extend(pieces)
        q_previous = np.asarray(stage.joints, dtype=float).reshape(7)
    if len(expanded) != len(stages):
        print(
            f"[motion][execution] before-close route subdivided "
            f"{len(stages)} planned stage(s) into {len(expanded)} guarded "
            f"command(s), max_joint_step={step_deg:.1f}deg"
        )
    return tuple(expanded)


def _after_close_execution_stages(plan, args):
    """Bound very large carrying commands without changing the frozen path.

    The public Nero driver exposes one target per ``move_j`` command and no
    supported online blend/trajectory API.  A single 50--70 degree carrying
    edge can therefore finish different axes at visibly different times.  Use
    only a few joint-linear waypoints so the measured motion stays near the
    validated edge while avoiding the many tiny stop-and-go segments that made
    earlier executions look jerky.
    """
    stages = tuple(getattr(plan, "after_close", tuple()))
    if not stages:
        return tuple()
    step_deg = float(
        getattr(args, "carrying_execution_waypoint_deg", 30.0)
    )
    if not np.isfinite(step_deg) or step_deg <= 0.0:
        return stages

    from .active_motion import _subdivide_active_direct_stage

    subdivision_args = copy.copy(args)
    expanded = []
    before_close = tuple(getattr(plan, "before_close", tuple()))
    q_previous = np.asarray(
        before_close[-1].joints if before_close else plan.start_joints,
        dtype=float,
    ).reshape(7)
    for stage in stages:
        stage_step = _adaptive_execution_waypoint_deg(
            plan,
            stage,
            q_previous,
            args,
            base_step_deg=step_deg,
            smooth_step_deg=float(
                getattr(args, "carrying_smooth_waypoint_deg", 35.0)
            ),
            smooth_phases={"pickup_lift", "carrying"},
        )
        subdivision_args.active_execution_waypoint_deg = stage_step
        subdivision_args.active_return_execution_waypoint_deg = stage_step
        pieces = _subdivide_active_direct_stage(
            q_previous,
            stage,
            subdivision_args,
        )
        expanded.extend(pieces)
        q_previous = np.asarray(stage.joints, dtype=float).reshape(7)
    if len(expanded) != len(stages):
        print(
            "[motion][execution] carrying route smoothed from "
            f"{len(stages)} planned stage(s) to {len(expanded)} guarded "
            f"command(s), max_joint_step={step_deg:.1f}deg"
        )
    return tuple(expanded)


def _after_release_execution_stages(plan, args, start_joints):
    """Subdivide a successful released return into bounded joint commands.

    Planning validates a linear joint-space edge, but one large Nero ``move_j``
    can advance wrist axes at visibly different rates.  Keep the exact frozen
    edge and its collision policy while adding intermediate execution targets.
    """
    stages = tuple(getattr(plan, "after_release", tuple()))
    if not stages:
        return tuple()
    step_deg = float(
        getattr(args, "released_return_execution_waypoint_deg", 20.0)
    )
    if not np.isfinite(step_deg) or step_deg <= 0.0:
        return stages

    from .active_motion import _subdivide_active_direct_stage

    subdivision_args = copy.copy(args)
    expanded = []
    q_previous = np.asarray(start_joints, dtype=float).reshape(7)
    for stage in stages:
        stage_step = _adaptive_execution_waypoint_deg(
            plan,
            stage,
            q_previous,
            args,
            base_step_deg=step_deg,
            smooth_step_deg=float(
                getattr(args, "released_return_smooth_waypoint_deg", 30.0)
            ),
            smooth_phases={"released_free", "released_retreat", "free_open"},
        )
        subdivision_args.active_execution_waypoint_deg = stage_step
        subdivision_args.active_return_execution_waypoint_deg = stage_step
        pieces = _subdivide_active_direct_stage(
            q_previous,
            stage,
            subdivision_args,
        )
        expanded.extend(pieces)
        q_previous = np.asarray(stage.joints, dtype=float).reshape(7)
    if len(expanded) != len(stages):
        print(
            "[return] successful post-release route subdivided "
            f"{len(stages)} planned stage(s) into {len(expanded)} guarded "
            f"command(s), max_joint_step={step_deg:.1f}deg"
        )
    return tuple(expanded)


def _adaptive_execution_waypoint_deg(
    plan,
    stage,
    q_start,
    args,
    *,
    base_step_deg,
    smooth_step_deg,
    smooth_phases,
):
    """Use fewer stop-and-go commands only on demonstrably open path edges."""
    base = float(base_step_deg)
    smooth = max(float(smooth_step_deg), base)
    if (
        not bool(getattr(args, "adaptive_execution_smoothing", True))
        or smooth <= base
        or str(getattr(stage, "phase", "")) not in set(smooth_phases)
    ):
        return base
    extra = getattr(stage, "path_clearance_margin", None)
    if extra is None:
        return base
    extra = float(extra)
    tight_threshold = max(
        float(
            getattr(
                args,
                "adaptive_execution_tight_extra_clearance",
                0.015,
            )
        ),
        0.0,
    )
    tight_step = max(
        float(
            getattr(
                args,
                "adaptive_execution_tight_waypoint_deg",
                12.0,
            )
        ),
        1.0,
    )
    if np.isfinite(extra) and extra < tight_threshold:
        selected = min(base, tight_step)
        print(
            f"[motion][tight] {stage.label} extra_clearance="
            f"{extra * 1000.0:.1f}mm < "
            f"{tight_threshold * 1000.0:.1f}mm; "
            f"execution_step={selected:.1f}deg (base={base:.1f}deg)"
        )
        return selected
    threshold = max(
        float(getattr(args, "adaptive_execution_min_extra_clearance", 0.020)),
        0.0,
    )
    if not np.isfinite(extra) or extra < threshold:
        return base
    print(
        f"[motion][smooth] {stage.label} open-space edge extra_clearance="
        f"{extra * 1000.0:.1f}mm >= {threshold * 1000.0:.1f}mm; "
        f"execution_step={smooth:.1f}deg (base={base:.1f}deg)"
    )
    return smooth


def _release_gripper_after_place(
    arm,
    args,
    *,
    wait_until_gripper_open_hook,
):
    """Release at place with one bounded staged retry before giving up."""
    wait_fn = wait_until_gripper_open_hook
    target = min(
        float(args.gripper_open_width),
        float(getattr(args, "gripper_hardware_max_width", 0.085)),
    )
    force = float(args.gripper_force)
    arm.open_gripper(target, force, args.gripper_settle)
    if bool(wait_fn(arm, args)):
        return True

    retries = max(int(getattr(args, "place_release_retry_count", 1)), 0)
    step_count = max(int(getattr(args, "place_release_step_count", 3)), 1)
    step_settle = max(
        float(getattr(args, "place_release_step_settle", 0.25)), 0.0
    )
    for retry_index in range(1, retries + 1):
        status = arm.print_gripper_status()
        try:
            current = float(getattr(status, "value"))
        except (TypeError, ValueError):
            current = float(getattr(args, "gripper_close_width", 0.0))
        if not np.isfinite(current):
            current = float(getattr(args, "gripper_close_width", 0.0))
        current = float(np.clip(current, 0.0, target))
        widths = np.linspace(current, target, step_count + 1)[1:]
        print(
            "[return][release-retry] place release was not confirmed; "
            f"retry={retry_index}/{retries} staged_widths_mm="
            f"{np.round(widths * 1000.0, 1).tolist()}"
        )
        for width in widths:
            arm.open_gripper(float(width), force, step_settle)
        if bool(wait_fn(arm, args)):
            print(
                "[return][release-retry] gripper release confirmed; "
                "continue with the frozen retreat and return-home plan"
            )
            return True
    return False


__all__ = [
    "GraspWorkflowHooks",
    "_after_close_execution_stages",
    "_after_release_execution_stages",
    "_before_close_execution_stages",
    "_empty_recovery_execution_stages",
    "_release_gripper_after_place",
    "move_tcp_pose",
    "intermediate_tcp_pose",
    "move_tcp_pose_with_intermediate",
    "current_flange_pose_error",
    "print_current_flange_pose_error",
    "wait_until_grasp_close_pose_ready",
    "handle_motion_not_reached",
    "_recoverable_grasp_generation_failure",
    "plan_and_maybe_execute_grasp",
]


def move_tcp_pose(
    arm,
    tcp_pose,
    args,
    label,
    planned_joint_target=None,
    *,
    hooks,
):
    assert_table_guard_target_tcp = _require_hook(
        hooks, "assert_table_guard_target_tcp"
    )
    move_flange_pose = _require_hook(hooks, "move_flange_pose")
    assert_table_guard_target_tcp(tcp_pose, args, label)
    flange_pose = arm.get_tcp2flange_pose(tcp_pose)
    print(f"[{label}] target_tcp_pose    =", np.round(tcp_pose, 6).tolist())
    return move_flange_pose(
        arm,
        flange_pose,
        args,
        label,
        planned_joint_target=planned_joint_target,
    )


def intermediate_tcp_pose(current_tcp, target_tcp, args):
    current = np.asarray(current_tcp, dtype=float).reshape(6)
    target = np.asarray(target_tcp, dtype=float).reshape(6)
    mid = interpolate_pose(current, target, 0.5)
    lift = float(args.intermediate_lift)
    mid[2] = max(float(current[2]), float(target[2])) + lift
    if args.min_grasp_z is not None:
        mid[2] = max(mid[2], float(args.min_grasp_z) + lift)
    return mid


def move_tcp_pose_with_intermediate(
    arm,
    tcp_pose,
    args,
    label,
    planned_joint_target=None,
    *,
    hooks,
):
    move_tcp_pose = _require_hook(hooks, "move_tcp_pose")
    intermediate_tcp_pose = _require_hook(hooks, "intermediate_tcp_pose")
    try:
        return move_tcp_pose(
            arm,
            tcp_pose,
            args,
            label,
            planned_joint_target=planned_joint_target,
        )
    except MotionNotReached as first_exc:
        if planned_joint_target is not None:
            raise MotionNotReached(
                f"{label} prevalidated joint target was not reached; "
                "do not switch to an unvalidated intermediate path"
            ) from first_exc
        if not bool(args.use_intermediate_motion):
            raise
        current_tcp = np.asarray(arm.get_tcp_pose(), dtype=float).reshape(6)
        mid_tcp = intermediate_tcp_pose(current_tcp, tcp_pose, args)
        print(f"[{label}][fallback] direct target not reached: {first_exc}")
        print(f"[{label}][fallback] trying intermediate_tcp_pose =", np.round(mid_tcp, 6).tolist())
        try:
            move_tcp_pose(arm, mid_tcp, args, f"{label}_intermediate")
        except MotionNotReached as mid_exc:
            raise MotionNotReached(f"{label} intermediate not reached; original={first_exc}; intermediate={mid_exc}") from mid_exc
        print(f"[{label}][fallback] intermediate reached; retrying final target")
        return move_tcp_pose(arm, tcp_pose, args, label)


def current_flange_pose_error(arm, theoretical_flange_pose):
    theoretical = np.asarray(theoretical_flange_pose, dtype=float).reshape(6)
    current = np.asarray(arm.get_flange_pose(), dtype=float).reshape(6)
    delta = current - theoretical
    delta[3:] = (delta[3:] + np.pi) % (2.0 * np.pi) - np.pi
    pos_err = float(np.linalg.norm(delta[:3]))
    rpy_err = float(np.linalg.norm(delta[3:]))
    return theoretical, current, delta, pos_err, rpy_err


def print_current_flange_pose_error(
    arm,
    theoretical_flange_pose,
    label,
    *,
    hooks,
):
    current_flange_pose_error = _require_hook(
        hooks, "current_flange_pose_error"
    )
    theoretical, current, delta, pos_err, rpy_err = current_flange_pose_error(arm, theoretical_flange_pose)
    print(f"[{label}] theoretical_flange_pose =", np.round(theoretical, 6).tolist())
    print(f"[{label}] actual_flange_pose      =", np.round(current, 6).tolist())
    print(
        f"[{label}] actual_error: xyz={np.round(delta[:3], 6).tolist()}m "
        f"rpy={np.round(np.rad2deg(delta[3:]), 3).tolist()}deg "
        f"norm_pos={pos_err:.4f}m norm_rpy={np.rad2deg(rpy_err):.2f}deg"
    )
    return current, pos_err, rpy_err


def wait_until_grasp_close_pose_ready(
    arm,
    theoretical_flange_pose,
    args,
    *,
    hooks,
):
    current_flange_pose_error = _require_hook(
        hooks, "current_flange_pose_error"
    )
    print_current_flange_pose_error = _require_hook(
        hooks, "print_current_flange_pose_error"
    )
    wait_for_joint_settle = _require_hook(
        hooks, "wait_for_joint_settle"
    )
    arm_motion_status_snapshot = _require_hook(
        hooks, "arm_motion_status_snapshot"
    )
    pos_tol = float(args.grasp_close_pos_tol)
    rpy_tol = float(args.grasp_close_rpy_tol)
    timeout = float(args.grasp_close_wait_timeout)
    poll = max(float(args.move_poll_interval), 0.02)
    settle_args = copy.copy(args)
    settle_args.move_precommand_settle_timeout = timeout
    wait_for_joint_settle(
        arm,
        settle_args,
        "grasp_close_action_boundary",
    )
    start = time.time()
    printed_wait = False
    last_status = {"available": False, "normal": False, "reason": "not_sampled"}
    while True:
        _, _, _, pos_err, rpy_err = current_flange_pose_error(arm, theoretical_flange_pose)
        last_status = arm_motion_status_snapshot(arm, args)
        pose_ready = bool(pos_err <= pos_tol and rpy_err <= rpy_tol)
        controller_idle = bool(last_status.get("normal", False))
        if pose_ready and controller_idle:
            print(
                f"[grasp_close_pose_ready] close allowed: "
                f"pos_err={pos_err:.4f}m <= {pos_tol:.4f}m, "
                f"rpy_err={np.rad2deg(rpy_err):.2f}deg <= {np.rad2deg(rpy_tol):.2f}deg, "
                "encoders settled and controller reports idle/normal"
            )
            return
        if time.time() - start >= timeout:
            print_current_flange_pose_error(arm, theoretical_flange_pose, "grasp_close_pose_not_ready")
            raise MotionNotReached(
                f"grasp close pose not ready: pos_err={pos_err:.4f}m "
                f"(tol={pos_tol:.4f}m), rpy_err={np.rad2deg(rpy_err):.2f}deg "
                f"(tol={np.rad2deg(rpy_tol):.2f}deg), "
                f"controller_status={last_status}"
            )
        if not printed_wait:
            print(
                f"[grasp_close_pose_ready] waiting for tight close pose: "
                f"pos_err={pos_err:.4f}m tol={pos_tol:.4f}m, "
                f"rpy_err={np.rad2deg(rpy_err):.2f}deg tol={np.rad2deg(rpy_tol):.2f}deg, "
                f"controller={last_status.get('reason', 'unknown')}"
            )
            printed_wait = True
        time.sleep(poll)


def handle_motion_not_reached(
    arm,
    args,
    exc,
    context="grasp",
    *,
    hooks,
):
    print_status = _require_hook(hooks, "print_status")
    is_active = str(context).startswith("active")
    tag = "active" if is_active else "abort"
    print(f"[{tag}][warn] {exc}" if is_active else f"[abort] {exc}")
    print_status(arm)
    print(
        f"[{tag}] Motion command was sent but the target was not reached. "
        "This usually means Nero rejected the Cartesian pose, hit a workspace/joint limit, "
        "or the requested RPY jump is too large."
    )
    if is_active:
        print("[active] Active-view capture for this pose was skipped. No emergency stop was sent.")
    else:
        print("[abort] Subsequent grasp steps were skipped. No emergency stop was sent.")
    return "skip" if is_active else "quit"


def _recoverable_grasp_generation_failure(exc):
    message = str(exc)
    return bool(
        message.startswith("[generate_grasps]")
        or message.startswith("IK precheck rejected all grasp candidates")
    )


def plan_and_maybe_execute_grasp(
    arm,
    args,
    target_cloud,
    scene_cloud,
    T_base_camera,
    bgr,
    intr,
    out_dir,
    inst,
    plan_label="target",
    place_xyz_override=None,
    confirmation_word="GRASP",
    motion_scene_cloud=None,
    motion_environment_cloud=None,
    motion_environment_map=None,
    *,
    hooks,
):
    _axis_vector = _require_hook(hooks, "axis_vector")
    _candidate_preflight_width_signs = _require_hook(
        hooks, "candidate_preflight_width_signs"
    )
    _defer_candidate_endpoint_reachability = _require_hook(
        hooks, "defer_candidate_endpoint_reachability"
    )
    _recoverable_grasp_generation_failure = _require_hook(
        hooks, "recoverable_grasp_generation_failure"
    )
    annotate_grasp_candidate_physical_aperture = _require_hook(
        hooks, "annotate_grasp_candidate_physical_aperture"
    )
    annotate_grasp_candidate_reachability = _require_hook(
        hooks, "annotate_grasp_candidate_reachability"
    )
    annotate_grasp_candidate_workflow_preflight = _require_hook(
        hooks, "annotate_grasp_candidate_workflow_preflight"
    )
    assert_grasp_detected = _require_hook(hooks, "assert_grasp_detected")
    assert_table_guard_target_tcp = _require_hook(
        hooks, "assert_table_guard_target_tcp"
    )
    choose_default_grasp_candidate = _require_hook(
        hooks, "choose_default_grasp_candidate"
    )
    confirm_motion = _require_hook(hooks, "confirm_motion")
    execute_prevalidated_stages = _require_hook(
        hooks, "execute_prevalidated_stages"
    )
    recover_after_torque_stop = _require_hook(
        hooks, "recover_after_torque_stop"
    )
    execution_gripper_open_width = _require_hook(
        hooks, "execution_gripper_open_width"
    )
    get_graspnet_helpers = _require_hook(hooks, "get_graspnet_helpers")
    grasp_camera_summary = _require_hook(hooks, "grasp_camera_summary")
    grasp_candidate_width_shift = _require_hook(
        hooks, "grasp_candidate_width_shift"
    )
    grasp_tcp_axis_diagnostics = _require_hook(
        hooks, "grasp_tcp_axis_diagnostics"
    )
    grasp_to_tcp_poses = _require_hook(hooks, "grasp_to_tcp_poses")
    handle_motion_not_reached = _require_hook(
        hooks, "handle_motion_not_reached"
    )
    move_initial = _require_hook(hooks, "move_initial")
    move_tcp_pose_with_intermediate = _require_hook(
        hooks, "move_tcp_pose_with_intermediate"
    )
    plan_up_and_over_grasp_sequence = _require_hook(
        hooks, "plan_up_and_over_grasp_sequence"
    )
    preview_graspnet_pointcloud = _require_hook(
        hooks, "preview_graspnet_pointcloud"
    )
    print_current_flange_pose_error = _require_hook(
        hooks, "print_current_flange_pose_error"
    )
    print_status = _require_hook(hooks, "print_status")
    recover_after_empty_grasp = _require_hook(
        hooks, "recover_after_empty_grasp"
    )
    release_vision_model_caches = _require_hook(
        hooks, "release_vision_model_caches"
    )
    safe_stop = _require_hook(hooks, "safe_stop")
    save_grasp_motion_plan = _require_hook(hooks, "save_grasp_motion_plan")
    save_selected_grasp_pose = _require_hook(
        hooks, "save_selected_grasp_pose"
    )
    select_grasp_candidate = _require_hook(hooks, "select_grasp_candidate")
    verify_empty_grasp_recovery_at_initial_observation = _require_hook(
        hooks, "verify_empty_grasp_recovery_at_initial_observation"
    )
    wait_until_grasp_close_pose_ready = _require_hook(
        hooks, "wait_until_grasp_close_pose_ready"
    )
    wait_until_gripper_open = _require_hook(
        hooks, "wait_until_gripper_open"
    )
    if bool(getattr(args, "direct_pickup_contact_override", False)):
        print(
            "[protect][override][warn] DIRECT PICKUP CONTACT OVERRIDE is enabled: "
            "no short support-release stage will be inserted; target_approach "
            "open-finger/environment checks, pickup_lift gripper_tool/"
            "attached-target environment checks, and carrying/transport "
            "gripper_tool/environment checks are disabled in both preflight "
            "and runtime monitoring; released open fingers may pass the "
            "just-placed target, and the first empty-grasp backout uses the "
            "reverse target-approach contact policy. Gripper body, arm links, "
            "camera, carried payload/environment checks, payload table guard, "
            "and hard joint limits remain enabled."
        )
    if not bool(
        getattr(
            args,
            "motion_target_approach_finger_environment_guard",
            True,
        )
    ):
        print(
            "[protect][override][warn] target_approach open-finger environment "
            "collision checks are DISABLED for both workflow preflight and "
            "measured-joint runtime monitoring. Arm links, gripper body, camera, "
            "table, and all non-target-approach collision checks remain enabled."
        )
    if bool(getattr(args, "use_current_rpy", False)):
        detail = (
            "--use-current-rpy is incompatible with verified GraspNet axis mapping: "
            "it would keep the generated approach positions while replacing the gripper "
            "orientation. Remove this option and let the planner test both equivalent "
            "parallel-jaw orientations."
        )
        print(f"[plan][abort] {detail}")
        return GraspAttemptResult(
            GraspAttemptStatus.CONFIGURATION_FAILED, False, detail
        )
    if args.execute and bool(getattr(args, "require_gripper_contact", True)):
        stop_force = getattr(args, "gripper_stop_force", None)
        command_force = float(getattr(args, "gripper_force", 0.0))
        hardware_max_force = float(
            getattr(args, "gripper_hardware_max_force", 0.50)
        )
        effective_command_force = min(command_force, hardware_max_force)
        contact_config_ok = bool(
            args.slow_close_gripper
            and stop_force is not None
            and np.isfinite(float(stop_force))
            and float(stop_force) > 0.0
            and np.isfinite(command_force)
            and command_force > 0.0
            and np.isfinite(hardware_max_force)
            and hardware_max_force > 0.0
            and float(stop_force) <= effective_command_force
        )
        if not contact_config_ok:
            detail = (
                "confirmed-contact execution requires --slow-close-gripper, finite positive "
                "--gripper-stop-force, and stop_force <= the effective hardware-capped "
                "gripper force"
            )
            print(f"[plan][abort] {detail}")
            return GraspAttemptResult(GraspAttemptStatus.CONFIGURATION_FAILED, False, detail)
    if bool(getattr(args, "execute", False)) and (
        not bool(getattr(args, "grasp_motion_path_guard", True))
        or not bool(getattr(args, "motion_runtime_collision_guard", True))
        or str(getattr(args, "motion_mode", "cartesian")) != "ik_joint"
    ):
        detail = (
            "real grasp execution requires --motion-mode ik_joint, "
            "--grasp-motion-path-guard, and --motion-runtime-collision-guard; "
            "legacy/unguarded modes are dry-run diagnostics only"
        )
        print(f"[plan][abort] {detail}")
        return GraspAttemptResult(
            GraspAttemptStatus.CONFIGURATION_FAILED, False, detail
        )
    make_end_points_from_o3d_cloud, generate_grasps = get_graspnet_helpers()
    graspnet_target_cloud = _prepare_graspnet_target_cloud(
        target_cloud,
        args,
        T_base_camera=T_base_camera,
    )
    from .pointcloud_fusion import save_cloud

    save_cloud(os.path.join(out_dir, "graspnet_target_full.ply"), target_cloud)
    save_cloud(
        os.path.join(out_dir, "graspnet_target_focused.ply"),
        graspnet_target_cloud,
    )
    focus_applied = bool(
        getattr(args, "_graspnet_target_focus_info", {}).get("applied", False)
    )
    end_points, cloud_vis = make_end_points_from_o3d_cloud(
        graspnet_target_cloud,
        num_point=args.grasp_num_points,
        T_base_camera=T_base_camera,
        shape_aware_sampling=bool(
            getattr(args, "graspnet_elongated_top_sampling", True)
        ) and not focus_applied,
    )
    graspnet_approach_axis = _axis_vector(args.graspnet_approach_axis)
    graspnet_approach_col = int(np.argmax(np.abs(graspnet_approach_axis)))
    graspnet_approach_sign = 1 if graspnet_approach_axis[graspnet_approach_col] >= 0.0 else -1
    graspnet_width_axis = _axis_vector(args.graspnet_width_axis)
    graspnet_width_col = int(np.argmax(np.abs(graspnet_width_axis)))
    R_base_camera = np.asarray(T_base_camera, dtype=float).reshape(4, 4)[:3, :3]
    world_down_camera = R_base_camera.T @ np.array([0.0, 0.0, -1.0], dtype=float)
    graspnet_collision_approach_dist = max(
        float(
            getattr(
                args,
                "graspnet_collision_approach_dist",
                min(float(args.pregrasp_offset), 0.04),
            )
        ),
        0.01,
    )
    print(
        "[grasp][collision] early GraspNet corridor="
        f"{graspnet_collision_approach_dist * 1000.0:.1f}mm; "
        f"actual pregrasp offset remains {float(args.pregrasp_offset) * 1000.0:.1f}mm "
        "and is checked by full motion planning"
    )
    release_vision_model_caches()
    try:
        gg, rows = generate_grasps(
            end_points,
            graspnet_target_cloud,
            T_wc_cap=np.asarray(T_base_camera, dtype=float).reshape(4, 4),
            collision_cloud=scene_cloud,
            allow_collision_fallback=bool(args.allow_grasp_collision_fallback),
            allow_angle_fallback=bool(args.allow_grasp_angle_fallback),
            visual=bool(args.visualize_grasps),
            manual_select=bool(args.manual_grasp_select) and not bool(args.select_grasp_candidate),
            manual_topk=int(args.manual_topk),
            angle_deg=float(args.grasp_angle_deg),
            vertical_cam=world_down_camera,
            approach_col=graspnet_approach_col,
            approach_sign=graspnet_approach_sign,
            width_col=graspnet_width_col,
            collision_approach_dist=graspnet_collision_approach_dist,
            collision_isolated_filter=bool(
                getattr(args, "motion_collision_isolated_filter", True)
            ),
            collision_isolated_radius=float(
                getattr(args, "motion_collision_isolated_radius", 0.020)
            ),
            collision_isolated_min_support=int(
                getattr(args, "motion_collision_isolated_min_support", 8)
            ),
            target_empty_thresh=float(args.grasp_target_empty_thresh),
            target_empty_voxel_size=float(args.grasp_target_empty_voxel_size),
            support_plane_z=float(args.table_z),
        )
    except RuntimeError as exc:
        diagnostic_rows = getattr(exc, "diagnostic_candidate_rows", None)
        if diagnostic_rows:
            diagnostic_stage = str(
                getattr(exc, "diagnostic_filter_stage", "unknown")
            )
            if diagnostic_stage == "collision":
                print(
                    "[grasp][diagnostic] strict and relaxed GraspNet "
                    "collision filtering rejected every decoded candidate; "
                    "show pre-filter candidates for review only"
                )
            elif diagnostic_stage == "world_down_angle":
                print(
                    "[grasp][diagnostic] all decoded candidates failed the "
                    f"{float(args.grasp_angle_deg):.1f}deg world-down approach "
                    "constraint; show rejected candidates for review only"
                )
            else:
                print(
                    "[grasp][diagnostic] candidate generation failed after "
                    "decoding; show attached diagnostic candidates for "
                    "review only"
                )
            print(
                "[grasp][diagnostic] close the Open3D window to finish the "
                "failed planning attempt; these candidates will not enter IK "
                "or robot execution"
            )
            preview_graspnet_pointcloud(
                target_cloud,
                diagnostic_rows,
                args,
                out_dir,
            )
        recoverable = _recoverable_grasp_generation_failure(exc)
        level = "abort" if recoverable else "error"
        print(f"[plan][{level}] GraspNet/collision filtering failed for {plan_label}: {exc}")
        print(f"[plan][abort] No robot motion command was sent for this {plan_label}.")
        status = GraspAttemptStatus.PLANNING_FAILED if recoverable else GraspAttemptStatus.INTERNAL_ERROR
        return GraspAttemptResult(status, False, str(exc))
    current_tcp_pose = arm.get_tcp_pose()
    rows = annotate_grasp_candidate_physical_aperture(
        rows,
        target_cloud,
        T_base_camera,
        current_tcp_pose,
        args,
    )
    rows = annotate_grasp_candidate_reachability(rows, T_base_camera, current_tcp_pose, args, arm=arm)
    place_xyz = (
        place_xyz_override
        if place_xyz_override is not None
        else args.place_xyz
    )
    workflow_preflight_kwargs = {
        "place_xyz": place_xyz,
        "motion_environment_cloud": motion_environment_cloud,
    }
    if motion_environment_map is not None:
        workflow_preflight_kwargs["motion_environment_map"] = (
            motion_environment_map
        )
    rows = annotate_grasp_candidate_workflow_preflight(
        rows,
        arm,
        args,
        target_cloud,
        scene_cloud if motion_scene_cloud is None else motion_scene_cloud,
        T_base_camera,
        current_tcp_pose,
        plan_label,
        **workflow_preflight_kwargs,
    )
    preview_graspnet_pointcloud(target_cloud, rows, args, out_dir)
    try:
        if args.select_grasp_candidate or args.grasp_candidate_index is not None:
            grasp, selected_grasp_row = select_grasp_candidate(rows, bgr, intr, out_dir, args)
        else:
            grasp, selected_grasp_row = choose_default_grasp_candidate(rows, gg, args)
        if (
            not _defer_candidate_endpoint_reachability(args)
            and (selected_grasp_row is not None)
            and (not bool(selected_grasp_row.get("reachability_ok", True)))
            and not bool(args.allow_hard_grasp_candidate)
        ):
            raise PlanNotExecutable(
                f"default grasp candidate {int(selected_grasp_row.get('idx', 0))} is hard to reach "
                f"({selected_grasp_row.get('reachability_reason', 'unknown')}); execution remains fail-closed."
            )
        selected_width_sign = float(
            1.0
            if selected_grasp_row is None
            else selected_grasp_row.get("tcp_width_sign", 1.0)
        )
        selected_width_shift = grasp_candidate_width_shift(
            selected_grasp_row, selected_width_sign
        )
        pre_tcp, grasp_tcp, lift_tcp = grasp_to_tcp_poses(
            grasp,
            T_base_camera,
            args,
            current_tcp_pose,
            width_sign=selected_width_sign,
            width_shift=selected_width_shift,
        )
        if (
            selected_grasp_row is not None
            and selected_grasp_row.get("_workflow_motion_plan", None)
            is not None
        ):
            selected_width_sign = float(
                selected_grasp_row.get(
                    "workflow_preflight_width_sign",
                    selected_width_sign,
                )
            )
            selected_width_shift = grasp_candidate_width_shift(
                selected_grasp_row, selected_width_sign
            )
            pre_tcp = np.asarray(
                selected_grasp_row["_workflow_pre_tcp"], dtype=float
            ).reshape(6)
            grasp_tcp = np.asarray(
                selected_grasp_row["_workflow_grasp_tcp"], dtype=float
            ).reshape(6)
            lift_tcp = np.asarray(
                selected_grasp_row["_workflow_lift_tcp"], dtype=float
            ).reshape(6)
        planned_start_joints = None
        planned_pre_joints = None
        planned_grasp_joints = None
        planned_lift_joints = None
        if (
            str(getattr(args, "motion_mode", "cartesian")) == "ik_joint"
            and not bool(getattr(args, "grasp_motion_path_guard", True))
        ):
            if selected_grasp_row is None:
                raise PlanNotExecutable(
                    "selected grasp has no annotated full-sequence joint plan"
                )
            planned_start_joints = selected_grasp_row.get(
                "planned_start_joints", None
            )
            planned_pre_joints = selected_grasp_row.get(
                "planned_pre_joints", None
            )
            planned_grasp_joints = selected_grasp_row.get(
                "planned_grasp_joints", None
            )
            planned_lift_joints = selected_grasp_row.get(
                "planned_lift_joints", None
            )
            if any(
                value is None
                for value in (
                    planned_start_joints,
                    planned_pre_joints,
                    planned_grasp_joints,
                    planned_lift_joints,
                )
            ):
                raise PlanNotExecutable(
                    "selected grasp is missing the prevalidated pregrasp/grasp/lift "
                    "joint sequence"
                )
            current_joints_now = np.asarray(
                arm.get_joint_angles(), dtype=float
            ).reshape(7)
            start_error_deg = float(
                np.rad2deg(
                    np.max(
                        np.abs(
                            current_joints_now
                            - np.asarray(planned_start_joints, dtype=float).reshape(7)
                        )
                    )
                )
            )
            if start_error_deg > float(
                getattr(args, "grasp_plan_start_joint_match_deg", 2.0)
            ):
                raise PlanNotExecutable(
                    f"robot joints changed after grasp precheck by {start_error_deg:.2f}deg; "
                    "replanning is required"
                )
    except GraspSelectionDeclined as exc:
        print(f"[plan] {exc}; no robot motion command was sent for this {plan_label}.")
        return GraspAttemptResult(GraspAttemptStatus.DECLINED, False, str(exc))
    except InvalidGraspSelection as exc:
        print(f"[plan][abort] {exc}; no robot motion command was sent for this {plan_label}.")
        return GraspAttemptResult(GraspAttemptStatus.CONFIGURATION_FAILED, False, str(exc))
    except PlanNotExecutable as exc:
        print(f"[plan][abort] {exc}")
        print(f"[plan][abort] No robot motion command was sent for this {plan_label}.")
        print_status(arm)
        return GraspAttemptResult(GraspAttemptStatus.PLANNING_FAILED, False, str(exc))

    selected_camera_summary = grasp_camera_summary(grasp)
    gripper_open_width_cmd = execution_gripper_open_width(grasp, args)
    save_selected_grasp_pose(out_dir, grasp, T_base_camera, pre_tcp, grasp_tcp, lift_tcp, selected_grasp_row)

    print(f"\n[plan:{plan_label}]")
    print("selected_label =", inst.get("label", "unknown"))
    if selected_grasp_row is not None:
        print("selected_grasp_idx =", int(selected_grasp_row.get("idx", 0)))
    selected_width = selected_camera_summary.get("graspnet_width", None)
    if selected_width is not None:
        print(f"graspnet_width = {float(selected_width):.6f} m ({float(selected_width) * 1000.0:.2f} mm)")
    selected_depth = selected_camera_summary.get("graspnet_depth", None)
    if selected_depth is not None:
        print(f"graspnet_depth = {float(selected_depth):.6f} m ({float(selected_depth) * 1000.0:.2f} mm)")
    print("graspnet_t_camera =", np.round(np.asarray(selected_camera_summary["graspnet_translation_camera"], dtype=float), 6).tolist())
    print("graspnet_rpy_camera_deg =", np.round(np.asarray(selected_camera_summary["graspnet_rpy_camera_deg"], dtype=float), 3).tolist())
    print("graspnet_R_camera =")
    print(np.round(np.asarray(selected_camera_summary["graspnet_rotation_matrix_camera"], dtype=float), 6))
    print("topdown_grasp =", args.topdown_grasp)
    print("tool_approach_axis =", args.tool_approach_axis)
    print("graspnet_approach_axis =", args.graspnet_approach_axis)
    print("graspnet_width_axis =", args.graspnet_width_axis)
    print("tcp_gripper_width_axis =", args.tcp_gripper_width_axis)
    print("tcp_width_equivalent_sign =", int(selected_width_sign))
    print(
        "tcp_width_recenter = "
        f"{float(selected_width_shift):.6f} m "
        f"({float(selected_width_shift) * 1000.0:.2f} mm)"
    )
    try:
        axis_diag = grasp_tcp_axis_diagnostics(
            grasp,
            T_base_camera,
            args,
            grasp_tcp,
            width_sign=selected_width_sign,
        )
        print("graspnet_approach_base =", np.round(axis_diag["grasp_approach_base"], 4).tolist())
        print("graspnet_width_base    =", np.round(axis_diag["grasp_width_base"], 4).tolist())
        print("tcp_approach_base      =", np.round(axis_diag["tcp_approach_base"], 4).tolist())
        print("tcp_width_base         =", np.round(axis_diag["tcp_width_base"], 4).tolist())
        print(
            "axis_mapping_error_deg = "
            f"approach:{axis_diag['approach_angle_deg']:.2f} "
            f"width:{axis_diag['width_angle_deg']:.2f}"
        )
    except Exception as exc:
        print(f"[plan][warn] axis mapping diagnostics unavailable: {type(exc).__name__}: {exc}")
    print("approach_tilt_deg =", args.approach_tilt_deg)
    print("use_graspnet_depth_advance =", args.use_graspnet_depth_advance)
    print("grasp_advance =", args.grasp_advance)
    print(f"gripper_open_command = {gripper_open_width_cmd:.6f} m ({gripper_open_width_cmd * 1000.0:.2f} mm)")
    print("current_tcp   =", np.round(current_tcp_pose, 6).tolist())
    if args.use_current_rpy:
        print("rpy_mode      = current TCP RPY")
    elif args.fixed_rpy is not None:
        print("rpy_mode      = fixed RPY")
    else:
        print("rpy_mode      = generated grasp RPY")
    print("pre_tcp       =", np.round(pre_tcp, 6).tolist())
    print("grasp_tcp     =", np.round(grasp_tcp, 6).tolist())
    print("lift_tcp      =", np.round(lift_tcp, 6).tolist())
    if selected_grasp_row is not None:
        selected_pregrasp_offset = selected_grasp_row.get(
            "workflow_preflight_pregrasp_offset", None
        )
        selected_lift_relax = selected_grasp_row.get(
            "workflow_preflight_lift_relax_deg", None
        )
        if selected_pregrasp_offset is not None:
            print(
                "planning_pregrasp_offset = "
                f"{float(selected_pregrasp_offset):.3f} m"
            )
        if selected_lift_relax is not None:
            print(
                "planning_lift_relax = "
                f"{float(selected_lift_relax):+.1f} deg axis="
                f"{selected_grasp_row.get('workflow_preflight_lift_relax_axis', 'none')}"
            )
        selected_transport_route = selected_grasp_row.get(
            "workflow_preflight_transport_route", None
        )
        if selected_transport_route is not None:
            print(f"planning_transport_route = {selected_transport_route}")
    cached_motion_plan = (
        None
        if selected_grasp_row is None
        else selected_grasp_row.get("_workflow_motion_plan", None)
    )
    validation_only_plan = bool(
        cached_motion_plan is not None
        and getattr(cached_motion_plan, "validation_only", False)
    )
    if place_xyz is not None and not validation_only_plan:
        cached_place_tcp = (
            None
            if selected_grasp_row is None
            else selected_grasp_row.get("_workflow_place_tcp", None)
        )
        if cached_place_tcp is None:
            place_tcp = _place_tcp_from_lift(lift_tcp, place_xyz, args)
        else:
            place_tcp = np.asarray(cached_place_tcp, dtype=float).reshape(6)
        print("place_tcp     =", np.round(place_tcp, 6).tolist())
    else:
        place_tcp = None
        if validation_only_plan:
            print(
                "place_tcp     = None (grasp-validation plan stops after "
                "safe support release; gripper remains closed)"
            )
        else:
            print("place_tcp     = None (will keep gripper closed after lift)")

    # Defense in depth: validate the entire selected TCP sequence before the
    # gripper opens or the arm moves to pregrasp.  Candidate annotation already
    # filters unsafe grasps, but this also covers manual/fallback plans and the
    # optional place pose.
    planned_tcp_sequence = [
        (f"{plan_label}_pregrasp", pre_tcp),
        (f"{plan_label}_grasp", grasp_tcp),
        (f"{plan_label}_lift", lift_tcp),
    ]
    if place_tcp is not None:
        planned_tcp_sequence.append((f"{plan_label}_place", place_tcp))
    if not bool(getattr(args, "grasp_motion_path_guard", True)):
        try:
            for stage_label, tcp_pose in planned_tcp_sequence:
                assert_table_guard_target_tcp(tcp_pose, args, stage_label)
        except UnsafeMotionTarget as exc:
            detail = f"full grasp-sequence table preflight failed: {exc}"
            print(f"[plan][abort] {detail}")
            print(f"[plan][abort] No robot motion command was sent for this {plan_label}.")
            return GraspAttemptResult(GraspAttemptStatus.PLANNING_FAILED, False, detail)

    full_motion_plan = (
        None
        if selected_grasp_row is None
        else selected_grasp_row.get("_workflow_motion_plan", None)
    )
    reused_workflow_plan = full_motion_plan is not None
    if (
        bool(getattr(args, "grasp_motion_path_guard", True))
        and full_motion_plan is None
    ):
        raw_width_signs, width_signs, orientation_angles = (
            _candidate_preflight_width_signs(
                selected_grasp_row,
                grasp,
                T_base_camera,
                current_tcp_pose,
                args,
            )
        )
        primary_sign = float(raw_width_signs[0])
        plan_failures = (
            []
            if len(width_signs) > 0
            else [
                "no endpoint- and aperture-validated jaw-axis orientation remains"
            ]
        )
        original_width_sign = float(selected_width_sign)
        pose_fallback_used = False
        if (
            len(width_signs) > 0
            and abs(float(width_signs[0]) - primary_sign) >= 0.5
        ):
            print(
                f"[plan][jaw-fallback] primary width_sign={int(primary_sign):+d} "
                "was removed by aperture/endpoint checks; start with the "
                f"180deg fallback width_sign={int(width_signs[0]):+d}"
            )
        route_modes = ["direct"]
        if (
            bool(getattr(args, "motion_neutral_pregrasp_fallback", False))
            and not bool(getattr(args, "motion_up_and_over", False))
        ):
            route_modes.append("neutral")
        for route_mode in route_modes:
            if route_mode == "neutral":
                print(
                    "[plan][neutral] every direct jaw-axis route failed; "
                    "retrying through one neutral joint configuration"
                )
            route_failures = []
            for width_sign in width_signs:
                is_fallback = abs(float(width_sign) - primary_sign) >= 0.5
                if is_fallback and route_failures:
                    print(
                        f"[plan][jaw-fallback] primary width_sign="
                        f"{int(primary_sign):+d} failed the complete "
                        f"{route_mode} route; trying 180deg fallback "
                        f"width_sign={int(width_sign):+d}"
                    )
                width_shift = grasp_candidate_width_shift(
                    selected_grasp_row, width_sign
                )
                base_pre, base_grasp, base_lift = grasp_to_tcp_poses(
                        grasp,
                        T_base_camera,
                        args,
                        current_tcp_pose,
                        width_sign=width_sign,
                        width_shift=width_shift,
                    )
                def plan_pose_attempt(
                    pose_variant, transport_route_mode, attempt_index
                ):
                    variant_pre = np.asarray(
                        pose_variant["pre_tcp"], dtype=float
                    ).reshape(6)
                    variant_grasp = np.asarray(
                        pose_variant["grasp_tcp"], dtype=float
                    ).reshape(6)
                    variant_lift = np.asarray(
                        pose_variant["lift_tcp"], dtype=float
                    ).reshape(6)
                    validation_only = bool(
                        pose_variant.get("validation_only", False)
                    )
                    variant_place = (
                        None
                        if validation_only
                        else _place_tcp_from_lift(variant_lift, place_xyz, args)
                    )
                    for stage_label, tcp_pose in (
                        (f"{plan_label}_pregrasp", variant_pre),
                        (f"{plan_label}_grasp", variant_grasp),
                        (f"{plan_label}_lift", variant_lift),
                        (f"{plan_label}_place", variant_place),
                    ):
                        if tcp_pose is not None:
                            assert_table_guard_target_tcp(
                                tcp_pose, args, stage_label
                            )
                    variant_plan_kwargs = {
                        "motion_environment_cloud": motion_environment_cloud,
                        "pregrasp_route_mode": route_mode,
                        "transport_route_mode": transport_route_mode,
                        "validation_only": validation_only,
                    }
                    if motion_environment_map is not None:
                        variant_plan_kwargs["motion_environment_map"] = (
                            motion_environment_map
                        )
                    variant_plan_label = (
                        plan_label
                        if attempt_index == 0
                        else f"{plan_label}_fallback{attempt_index}"
                    )
                    return plan_up_and_over_grasp_sequence(
                        arm,
                        variant_pre,
                        variant_grasp,
                        variant_lift,
                        variant_place,
                        target_cloud,
                        scene_cloud
                        if motion_scene_cloud is None
                        else motion_scene_cloud,
                        T_base_camera,
                        args,
                        variant_plan_label,
                        **variant_plan_kwargs,
                    )

                selected, width_failure_records = _plan_grasp_pose_fallbacks(
                    base_pre,
                    base_grasp,
                    base_lift,
                    args,
                    plan_pose_attempt,
                )
                for record in width_failure_records:
                    pose_variant = record["variant"]
                    print(
                        f"[plan][variant][reject] route={route_mode}/"
                        f"width_sign={int(width_sign):+d}/"
                        f"stage={record['stage']}/"
                        f"transport={record['transport_route_mode']}/"
                        f"pre={float(pose_variant['pregrasp_offset']) * 1000.0:.0f}mm/"
                        f"lift={float(pose_variant['lift_relax_deg']):+.0f}deg:"
                        f"{type(record['exception']).__name__}:"
                        f"{record['exception']}"
                    )
                if selected is None:
                    stage_counts = {}
                    for record in width_failure_records:
                        stage = str(record["stage"])
                        stage_counts[stage] = stage_counts.get(stage, 0) + 1
                    samples = [
                        f"{record['stage']}:{type(record['exception']).__name__}:"
                        f"{record['exception']}"
                        for record in width_failure_records[:2]
                    ]
                    failure = (
                        f"route={route_mode}/width_sign={int(width_sign):+d}:"
                        f"stage-aware attempts={len(width_failure_records)} "
                        f"failures={stage_counts}: {' | '.join(samples)}"
                    )
                    route_failures.append(failure)
                    plan_failures.append(failure)
                    continue
                variant_plan = selected["plan"]
                selected_pose_variant = selected["variant"]
                selected_transport_route = selected["transport_route_mode"]
                variant_pre = np.asarray(
                    selected_pose_variant["pre_tcp"], dtype=float
                ).reshape(6)
                variant_grasp = np.asarray(
                    selected_pose_variant["grasp_tcp"], dtype=float
                ).reshape(6)
                variant_lift = np.asarray(
                    selected_pose_variant["lift_tcp"], dtype=float
                ).reshape(6)
                validation_only = bool(
                    selected_pose_variant.get("validation_only", False)
                )
                variant_place = (
                    None
                    if validation_only
                    else _place_tcp_from_lift(variant_lift, place_xyz, args)
                )
                variant_place = _apply_transport_pose_fallback(
                    variant_place, selected_pose_variant
                )
                if int(selected["attempt_index"]) > 0:
                    pose_fallback_used = True
                    print(
                        f"[plan][stage-fallback] selected "
                        f"pre={float(selected_pose_variant['pregrasp_offset']) * 1000.0:.0f}mm "
                        f"lift={float(selected_pose_variant['lift_relax_deg']):+.0f}deg "
                        f"transport={selected_transport_route} after "
                        f"{len(width_failure_records)} relevant failure(s)"
                    )
                selected_width_sign = float(width_sign)
                selected_width_shift = float(width_shift)
                pre_tcp = np.asarray(variant_pre, dtype=float)
                grasp_tcp = np.asarray(variant_grasp, dtype=float)
                lift_tcp = np.asarray(variant_lift, dtype=float)
                place_tcp = (
                    None
                    if variant_place is None
                    else np.asarray(variant_place, dtype=float)
                )
                full_motion_plan = variant_plan
                if selected_grasp_row is not None:
                    selected_grasp_row["workflow_preflight_route_kind"] = (
                        route_mode
                    )
                    selected_grasp_row["jaw_primary_width_sign"] = primary_sign
                    selected_grasp_row["jaw_selected_as_fallback"] = bool(
                        is_fallback
                    )
                    selected_grasp_row["jaw_selected_role"] = (
                        "fallback_180" if is_fallback else "primary"
                    )
                    selected_grasp_row[
                        "jaw_selected_orientation_delta_deg"
                    ] = float(
                        np.rad2deg(orientation_angles[float(width_sign)])
                    )
                    selected_grasp_row[
                        "workflow_preflight_pregrasp_offset"
                    ] = float(selected_pose_variant["pregrasp_offset"])
                    selected_grasp_row[
                        "workflow_preflight_lift_relax_deg"
                    ] = float(selected_pose_variant["lift_relax_deg"])
                    selected_grasp_row[
                        "workflow_preflight_lift_relax_axis"
                    ] = str(selected_pose_variant["lift_relax_axis"])
                    selected_grasp_row[
                        "workflow_preflight_transport_route"
                    ] = str(selected_transport_route)
                    selected_grasp_row[
                        "workflow_preflight_validation_only"
                    ] = validation_only
                break
            if full_motion_plan is not None:
                break
        if full_motion_plan is None:
            detail = (
                "whole-workflow motion preflight failed for every allowed "
                "jaw-axis orientation: "
                + " | ".join(plan_failures)
            )
            print(f"[plan][abort] {detail}")
            print(
                f"[plan][abort] Gripper was not opened and no robot motion command "
                f"was sent for this {plan_label}."
            )
            return GraspAttemptResult(
                GraspAttemptStatus.PLANNING_FAILED, False, detail
            )
        if selected_width_sign != original_width_sign or pose_fallback_used:
            if selected_width_sign != original_width_sign:
                print(
                    f"[plan][variant] switched equivalent TCP width sign "
                    f"{int(original_width_sign):+d} -> {int(selected_width_sign):+d} "
                    "because only the latter passed the complete route"
                )
            if selected_grasp_row is not None:
                selected_grasp_row["tcp_width_sign"] = selected_width_sign
            save_selected_grasp_pose(
                out_dir,
                grasp,
                T_base_camera,
                pre_tcp,
                grasp_tcp,
                lift_tcp,
                selected_grasp_row,
            )
            print("[plan][variant] corrected_pre_tcp  =", np.round(pre_tcp, 6).tolist())
            print("[plan][variant] corrected_grasp_tcp=", np.round(grasp_tcp, 6).tolist())
            print("[plan][variant] corrected_lift_tcp =", np.round(lift_tcp, 6).tolist())
        if abs(float(selected_width_sign) - primary_sign) >= 0.5:
            print(
                f"[plan][jaw-fallback] using the 180deg fallback width_sign="
                f"{int(selected_width_sign):+d}; primary width_sign="
                f"{int(primary_sign):+d} did not pass all hard checks"
            )
    if (
        bool(getattr(args, "grasp_motion_path_guard", True))
        and full_motion_plan is not None
    ):
        if reused_workflow_plan:
            print(
                "[plan][workflow] reusing the complete collision-checked plan "
                "that qualified this candidate for selection"
            )
        save_grasp_motion_plan(out_dir, full_motion_plan)

    if not args.execute:
        print("[dry-run] Not executing. Re-run with --execute after checking the plan.")
        return GraspAttemptResult(GraspAttemptStatus.PLANNED_ONLY, False, "execute_disabled")
    execution_validation_only = bool(
        full_motion_plan is not None
        and getattr(full_motion_plan, "validation_only", False)
    )
    if not args.yes:
        confirmation = (
            f"execute {confirmation_word}: open/pregrasp/grasp/close/"
            "support-release only; keep gripper closed"
            if execution_validation_only
            else f"execute {confirmation_word}: open/pregrasp/grasp/close/lift/place"
        )
        if not confirm_motion(
            confirmation
        ):
            print("[abort] execution cancelled by no")
            return GraspAttemptResult(GraspAttemptStatus.DECLINED, False, "cancelled_by_no")

    motion_sent = False
    grasp_pose_reached = False
    try:
        motion_sent = True
        arm.open_gripper(gripper_open_width_cmd, args.gripper_force, args.gripper_settle)
        if not wait_until_gripper_open(arm, args):
            raise GraspNotDetected("gripper did not reach the required open width before pregrasp")
        if full_motion_plan is not None:
            before_close_execution = _before_close_execution_stages(
                full_motion_plan, args
            )
            grasp_flange_pose, _ = execute_prevalidated_stages(
                arm,
                before_close_execution,
                full_motion_plan.start_joints,
                full_motion_plan,
                args,
            )
        else:
            move_tcp_pose_with_intermediate(
                arm,
                pre_tcp,
                args,
                f"{plan_label}_pregrasp",
                planned_joint_target=planned_pre_joints,
            )
            boundary_marker = object()
            previous_boundary = getattr(
                args, "_grasp_close_action_boundary_active", boundary_marker
            )
            args._grasp_close_action_boundary_active = True
            try:
                grasp_flange_pose = move_tcp_pose_with_intermediate(
                    arm,
                    grasp_tcp,
                    args,
                    f"{plan_label}_grasp",
                    planned_joint_target=planned_grasp_joints,
                )
            finally:
                if previous_boundary is boundary_marker:
                    if hasattr(args, "_grasp_close_action_boundary_active"):
                        delattr(args, "_grasp_close_action_boundary_active")
                else:
                    args._grasp_close_action_boundary_active = previous_boundary
        wait_until_grasp_close_pose_ready(arm, grasp_flange_pose, args)
        grasp_pose_reached = True
        print_current_flange_pose_error(arm, grasp_flange_pose, f"{plan_label}_grasp_close_pose_before")
        if args.slow_close_gripper:
            close_status = arm.close_gripper_slow(
                gripper_open_width_cmd,
                args.gripper_close_width,
                args.gripper_force,
                args.slow_close_step,
                args.slow_close_interval,
                args.gripper_settle,
                verbose=bool(args.verbose_gripper_close),
                stop_force=args.gripper_stop_force,
                contact_confirm_samples=args.gripper_contact_confirm_samples,
                contact_confirm_interval_s=args.gripper_contact_confirm_interval,
                empty_width_threshold=args.empty_grasp_width_threshold,
                closed_width_tolerance=args.gripper_closed_width_tol,
                close_completion_timeout_s=args.gripper_close_completion_timeout,
                close_completion_poll_s=args.gripper_close_completion_poll,
            )
        else:
            close_status = arm.close_gripper(args.gripper_close_width, args.gripper_force, args.gripper_settle)
        print_current_flange_pose_error(arm, grasp_flange_pose, f"{plan_label}_grasp_close_pose_after")
        assert_grasp_detected(close_status, args)
        if full_motion_plan is not None:
            after_close_execution = _after_close_execution_stages(
                full_motion_plan, args
            )
            _, carried_end_joints = execute_prevalidated_stages(
                arm,
                after_close_execution,
                full_motion_plan.before_close[-1].joints,
                full_motion_plan,
                args,
            )
        else:
            move_tcp_pose_with_intermediate(
                arm,
                lift_tcp,
                args,
                f"{plan_label}_lift",
                planned_joint_target=planned_lift_joints,
            )
            carried_end_joints = np.asarray(
                arm.get_joint_angles(), dtype=float
            ).reshape(7)
        if place_tcp is not None and not execution_validation_only:
            if full_motion_plan is None:
                move_tcp_pose_with_intermediate(
                    arm, place_tcp, args, f"{plan_label}_place"
                )
                carried_end_joints = np.asarray(
                    arm.get_joint_angles(), dtype=float
                ).reshape(7)
            print(
                "[return] opening gripper and confirming release before the "
                "prevalidated upward retreat"
            )
            if not _release_gripper_after_place(
                arm,
                args,
                wait_until_gripper_open_hook=wait_until_gripper_open,
            ):
                raise MotionNotReached(
                    "gripper release remained unconfirmed after the bounded "
                    "place retry; stay at place because the payload state is unknown"
                )
            if full_motion_plan is not None and full_motion_plan.after_release:
                released_return_stages = _after_release_execution_stages(
                    full_motion_plan,
                    args,
                    carried_end_joints,
                )
                print(
                    "[return] release confirmed; execute prevalidated "
                    f"retreat + return-home route stages="
                    f"{len(released_return_stages)}"
                )
                _, returned_joints = execute_prevalidated_stages(
                    arm,
                    released_return_stages,
                    carried_end_joints,
                    full_motion_plan,
                    args,
                )
                return_error_deg = float(
                    np.rad2deg(
                        np.max(
                            np.abs(
                                np.asarray(arm.get_joint_angles(), dtype=float).reshape(7)
                                - np.asarray(returned_joints, dtype=float).reshape(7)
                            )
                        )
                    )
                )
                print(
                    "[return] initial observation joints reached after place: "
                    f"joint_err={return_error_deg:.2f}deg"
                )
            elif full_motion_plan is not None and not args.skip_initial_move:
                raise MotionNotReached(
                    "release succeeded but the frozen workflow contains no "
                    "post-release return-home route"
                )
            elif full_motion_plan is None and not args.skip_initial_move:
                if not bool(move_initial(arm, args, require_confirm=False)):
                    raise MotionNotReached("return to initial pose failed")
        else:
            if execution_validation_only:
                print(
                    f"[done] grasp validation succeeded for {plan_label}: "
                    "safe support release completed; full lift/place were "
                    "not executable and the gripper remains closed."
                )
            else:
                print(f"[done] lifted {plan_label}; no place target provided, gripper remains closed.")
        return GraspAttemptResult(GraspAttemptStatus.SUCCESS, True, "executed")
    except KeyboardInterrupt:
        stopped = False
        stop_interrupt = getattr(arm, "stop_for_keyboard_interrupt", None)
        if callable(stop_interrupt):
            stopped = bool(stop_interrupt())
        else:
            print("[interrupt][warn] arm adapter has no controlled interrupt stop")
        detail = (
            "keyboard_interrupt_motion_terminated"
            if stopped
            else "keyboard_interrupt_stop_unconfirmed"
        )
        return GraspAttemptResult(
            GraspAttemptStatus.EXECUTION_FAILED, motion_sent, detail
        )
    except TorqueCollisionStopTriggered as exc:
        print(f"[torque] {exc}")
        try:
            recovered = bool(recover_after_torque_stop(arm, exc, args))
            detail = "verified reverse retreat completed" if recovered else "verified reverse retreat unavailable"
            print(f"[torque] current workflow is terminal after contact stop: {detail}")
            return GraspAttemptResult(
                GraspAttemptStatus.EXECUTION_FAILED,
                motion_sent,
                f"{exc}; {detail}",
            )
        except SafetyStopTriggered as recovery_exc:
            print(f"[protect][motion] torque retreat stop escalation: {recovery_exc}")
            return GraspAttemptResult(
                GraspAttemptStatus.EXECUTION_FAILED,
                motion_sent,
                f"{exc}; retreat failed safely: {recovery_exc}",
            )
        except Exception as recovery_exc:
            print(
                f"[torque][warn] verified retreat was not executed: "
                f"{type(recovery_exc).__name__}: {recovery_exc}"
            )
            return GraspAttemptResult(
                GraspAttemptStatus.EXECUTION_FAILED,
                motion_sent,
                f"{exc}; retreat unavailable: {type(recovery_exc).__name__}: {recovery_exc}",
            )
    except ControlledMotionStopTriggered as exc:
        print(f"[motion_stop] {exc}")
        print(
            "[motion_stop] Ordinary trajectory termination was confirmed; "
            "subsequent grasp/lift/place commands were skipped without "
            "electronic emergency stop."
        )
        return GraspAttemptResult(
            GraspAttemptStatus.EXECUTION_FAILED,
            motion_sent,
            str(exc),
        )
    except SafetyStopTriggered as exc:
        print(f"[protect][motion] {exc}")
        print("[protect][motion] Subsequent grasp/lift/place commands were skipped.")
        return GraspAttemptResult(GraspAttemptStatus.EXECUTION_FAILED, motion_sent, str(exc))
    except UnsafeMotionTarget as exc:
        print(f"[abort] unsafe target: {exc}")
        print("[abort] Subsequent motion commands were skipped.")
        return GraspAttemptResult(GraspAttemptStatus.EXECUTION_FAILED, motion_sent, str(exc))
    except PlanNotExecutable as exc:
        # Stored-plan/FK validation runs before the next move command is sent.
        # The previous stage has already reached and stopped, so an electronic
        # emergency stop is neither necessary nor helpful here.
        print(f"[abort] prevalidated execution stage rejected: {exc}")
        print(
            "[abort] No new motion command was sent for the rejected stage; "
            "the workflow is terminal without electronic emergency stop."
        )
        return GraspAttemptResult(
            GraspAttemptStatus.EXECUTION_FAILED, motion_sent, str(exc)
        )
    except GraspNotDetected as exc:
        print(f"[abort] grasp not detected: {exc}")
        print("[abort] Lift/place were skipped. Retracting and recovering to initial pose.")
        if grasp_pose_reached and full_motion_plan is not None:
            try:
                recovery_stages = list(
                    _empty_recovery_execution_stages(full_motion_plan, args)
                )
                expected = np.asarray(
                    full_motion_plan.before_close[-1].joints, dtype=float
                ).reshape(7)
                retract_count = 0
                if recovery_stages:
                    # The final piece retains the original first-stage label;
                    # all preceding ``_waypoint_*`` entries belong to the same
                    # grasp->pregrasp backout and must finish before opening.
                    first_original_label = str(
                        full_motion_plan.empty_recovery[0].label
                    )
                    retract_count = next(
                        (
                            index + 1
                            for index, stage in enumerate(recovery_stages)
                            if str(stage.label) == first_original_label
                        ),
                        1,
                    )
                    _, expected = execute_prevalidated_stages(
                        arm,
                        recovery_stages[:retract_count],
                        expected,
                        full_motion_plan,
                        args,
                    )
                delay = max(float(args.empty_grasp_return_delay), 0.0)
                if delay > 0.0:
                    print(
                        f"[return] empty grasp recovery delay {delay:.2f}s "
                        "at the prevalidated retract pose"
                    )
                    time.sleep(delay)
                print(
                    "[return] opening gripper before the remaining "
                    "prevalidated up-and-over recovery"
                )
                arm.open_gripper(
                    args.gripper_open_width,
                    args.gripper_force,
                    args.gripper_settle,
                )
                if not wait_until_gripper_open(arm, args):
                    raise MotionNotReached(
                        "gripper did not reopen during empty-grasp recovery"
                    )
                if len(recovery_stages) > retract_count:
                    execute_prevalidated_stages(
                        arm,
                        recovery_stages[retract_count:],
                        expected,
                        full_motion_plan,
                        args,
                    )
                recovered = True
            except ControlledMotionStopTriggered as retract_exc:
                print(f"[motion_stop] {retract_exc}")
                return GraspAttemptResult(
                    GraspAttemptStatus.EXECUTION_FAILED,
                    motion_sent,
                    f"{exc}; recovery_controlled_stop={retract_exc}",
                )
            except SafetyStopTriggered as retract_exc:
                print(f"[protect][motion] {retract_exc}")
                return GraspAttemptResult(
                    GraspAttemptStatus.EXECUTION_FAILED,
                    motion_sent,
                    f"{exc}; recovery_safety_stop={retract_exc}",
                )
            except Exception as retract_exc:
                stop_reason = (
                    "empty-grasp prevalidated recovery failed: "
                    f"{type(retract_exc).__name__}: {retract_exc}"
                )
                try:
                    request_ordinary_trajectory_stop(
                        arm,
                        stop_reason,
                        safe_stop_hook=safe_stop,
                    )
                except SafetyStopTriggered as stop_exc:
                    return GraspAttemptResult(
                        GraspAttemptStatus.EXECUTION_FAILED,
                        motion_sent,
                        f"{exc}; recovery_safety_stop={stop_exc}",
                    )
                return GraspAttemptResult(
                    GraspAttemptStatus.EXECUTION_FAILED,
                    motion_sent,
                    f"{exc}; recovery_controlled_stop={stop_reason}",
                )
        else:
            if grasp_pose_reached:
                try:
                    move_tcp_pose_with_intermediate(
                        arm,
                        pre_tcp,
                        args,
                        f"{plan_label}_empty_retract",
                        planned_joint_target=planned_pre_joints,
                    )
                except Exception as retract_exc:
                    stop_reason = (
                        "empty-grasp retract failed: "
                        f"{type(retract_exc).__name__}: {retract_exc}"
                    )
                    try:
                        request_ordinary_trajectory_stop(
                            arm,
                            stop_reason,
                            safe_stop_hook=safe_stop,
                        )
                    except SafetyStopTriggered as stop_exc:
                        return GraspAttemptResult(
                            GraspAttemptStatus.EXECUTION_FAILED,
                            motion_sent,
                            f"{exc}; recovery_safety_stop={stop_exc}",
                        )
                    return GraspAttemptResult(
                        GraspAttemptStatus.EXECUTION_FAILED,
                        motion_sent,
                        f"{exc}; recovery_controlled_stop={stop_reason}",
                    )
            recovered = recover_after_empty_grasp(arm, args)
        recovered_to_initial = bool(
            getattr(exc, "retryable_empty", False)
            and verify_empty_grasp_recovery_at_initial_observation(
                arm,
                args,
                recovered,
            )
        )
        if recovered_to_initial:
            detail = f"{exc}; empty_grasp_recovered_to_initial=true"
        elif recovered:
            detail = str(exc)
            if bool(getattr(exc, "retryable_empty", False)):
                detail += "; recovery_to_initial_not_verified"
        else:
            detail = f"{exc}; recovery_to_initial_failed"
        return GraspAttemptResult(
            GraspAttemptStatus.EXECUTION_FAILED,
            motion_sent,
            detail,
            empty_grasp_recovered_to_initial=recovered_to_initial,
        )
    except MotionNotReached as exc:
        handle_motion_not_reached(arm, args, exc)
        return GraspAttemptResult(GraspAttemptStatus.EXECUTION_FAILED, motion_sent, str(exc))
    except Exception as exc:
        safe_stop(arm, f"execution failed: {type(exc).__name__}: {exc}")
        raise
