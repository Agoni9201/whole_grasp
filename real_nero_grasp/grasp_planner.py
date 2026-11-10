"""Complete grasp-workflow planning and candidate preflight filtering."""

from copy import copy
from dataclasses import replace
import re
import time

import numpy as np
from scipy.spatial.transform import Rotation as R

from .grasp_geometry import grasp_planning_pose_variants
from . import ik_solver
from .types import (
    GraspMotionPlan,
    PlanNotExecutable,
    PlannedJointStage,
    UnsafeMotionTarget,
)


__all__ = [
    "plan_up_and_over_grasp_sequence",
    "plan_post_support_lift",
    "annotate_grasp_candidate_workflow_preflight",
]


def _place_tcp_from_lift(lift_tcp, place_xyz, args):
    """Build a place pose while preserving legacy orientation by default."""
    if place_xyz is None:
        return None
    place_tcp = np.asarray(lift_tcp, dtype=float).reshape(6).copy()
    place_tcp[:3] = np.asarray(place_xyz, dtype=float).reshape(3)
    configured_rpy = getattr(args, "place_rpy_deg", None)
    if configured_rpy is not None:
        place_tcp[3:] = np.deg2rad(
            np.asarray(configured_rpy, dtype=float).reshape(3)
        )
    return place_tcp


def _transport_pose_fallback_variants(variant, args):
    """Return a bounded local place region after exact transport fails.

    These variants change only the transport/place endpoint.  The selected
    pregrasp, grasp, support release and lift poses remain frozen, so a late
    transport failure does not restart global grasp planning.
    """
    if not bool(getattr(args, "transport_place_relax", False)):
        return ()
    xy = max(float(getattr(args, "transport_place_xy_relax", 0.020)), 0.0)
    z = max(float(getattr(args, "transport_place_z_relax", 0.020)), 0.0)
    tilt = max(
        float(getattr(args, "transport_place_tilt_relax_deg", 8.0)), 0.0
    )
    max_attempts = max(
        int(getattr(args, "transport_place_relax_max_attempts", 9)), 0
    )
    if max_attempts == 0:
        return ()

    variants = []

    def add(kind, *, xyz=(0.0, 0.0, 0.0), axis=None, degrees=0.0):
        relaxed = dict(variant)
        relaxed.update(
            {
                "transport_place_relax": True,
                "transport_place_xyz_offset": tuple(float(v) for v in xyz),
                "transport_place_tilt_axis": axis,
                "transport_place_tilt_deg": float(degrees),
                "transport_place_relax_kind": str(kind),
            }
        )
        variants.append(relaxed)

    # Raising the release location is normally the least intrusive fallback;
    # cardinal XY alternatives then cover basket-edge/workspace boundaries.
    if z > 1e-9:
        add("z_plus", xyz=(0.0, 0.0, z))
    if xy > 1e-9:
        for name, dx, dy in (
            ("x_plus", xy, 0.0),
            ("x_minus", -xy, 0.0),
            ("y_plus", 0.0, xy),
            ("y_minus", 0.0, -xy),
        ):
            add(name, xyz=(dx, dy, 0.0))
    if tilt > 1e-9:
        approach_axis = str(getattr(args, "tool_approach_axis", "x"))[-1]
        for axis in ("x", "y", "z"):
            if axis == approach_axis:
                continue
            for sign in (1.0, -1.0):
                add(
                    f"tilt_{axis}_{'plus' if sign > 0.0 else 'minus'}",
                    axis=axis,
                    degrees=sign * tilt,
                )
    return tuple(variants[:max_attempts])


def _apply_transport_pose_fallback(place_tcp, variant):
    """Apply one bounded transport endpoint fallback to a place TCP pose."""
    if place_tcp is None or not bool(variant.get("transport_place_relax", False)):
        return place_tcp
    relaxed = np.asarray(place_tcp, dtype=float).reshape(6).copy()
    relaxed[:3] += np.asarray(
        variant.get("transport_place_xyz_offset", (0.0, 0.0, 0.0)),
        dtype=float,
    ).reshape(3)
    axis = variant.get("transport_place_tilt_axis", None)
    degrees = float(variant.get("transport_place_tilt_deg", 0.0))
    if axis is not None and abs(degrees) > 1e-9:
        axis_vector = {
            "x": np.array([1.0, 0.0, 0.0]),
            "y": np.array([0.0, 1.0, 0.0]),
            "z": np.array([0.0, 0.0, 1.0]),
        }[str(axis)]
        base_rotation = R.from_euler("xyz", relaxed[3:]).as_matrix()
        delta = R.from_rotvec(axis_vector * np.deg2rad(degrees)).as_matrix()
        relaxed[3:] = R.from_matrix(base_rotation @ delta).as_euler("xyz")
    return relaxed


def _micro_lift_feasibility(arm, q_grasp, grasp_tcp, args, label):
    """Cheap strict-orientation world-Z IK screen after a grasp endpoint."""
    q_grasp = np.asarray(q_grasp, dtype=float).reshape(7)
    micro_tcp = np.asarray(grasp_tcp, dtype=float).reshape(6).copy()
    micro_tcp[2] += max(
        float(getattr(args, "grasp_micro_lift_height", 0.008)), 0.001
    )
    flange_pose = np.asarray(
        arm.get_tcp2flange_pose(micro_tcp), dtype=float
    ).reshape(6)
    seed_limit = max(
        int(getattr(args, "grasp_micro_lift_ik_seed_limit", 5)), 1
    )
    records = ik_solver._ranked_ik_solution_records_for_flange_pose(
        arm,
        flange_pose,
        args,
        f"{label}_micro_lift",
        current_joints_override=q_grasp,
        quiet=True,
        seed_limit=seed_limit,
        top_k=seed_limit,
    )
    best = records[0]
    return {
        "micro_lift_ik_ok": True,
        "micro_lift_joint_margin": float(
            np.rad2deg(best["limit_margin"])
        ),
        "micro_lift_max_joint_delta": float(
            np.rad2deg(np.max(np.abs(best["delta"])))
        ),
        "micro_lift_tcp": micro_tcp,
    }


def _support_release_rises(release_height):
    """Return unique monotonic fallback rises ending at the strict target."""
    release_height = max(float(release_height), 0.0)
    rises = []
    for requested in (0.005, 0.010, 0.015, release_height):
        rise = min(float(requested), release_height)
        if rise > 0.0 and not any(abs(rise - old) <= 1e-9 for old in rises):
            rises.append(rise)
    return tuple(rises)


def plan_post_support_lift(
    lift_tcp,
    lift_context,
    label,
    *,
    planner=None,
    arm=None,
    q_previous=None,
    args=None,
):
    """Plan legacy strict Lift Stage B or return its beam specification.

    Keeping this boundary separate lets a future planner replace Stage B
    without changing grasp workflow or frozen-plan execution.
    """
    if planner is not None:
        return planner(
            arm,
            np.asarray(lift_tcp, dtype=float).reshape(6),
            np.asarray(q_previous, dtype=float).reshape(7),
            lift_context,
            args,
            str(label),
        )
    return {
        "tcp_pose": np.asarray(lift_tcp, dtype=float).reshape(6).copy(),
        "context": lift_context,
        "label": str(label),
    }


def _grasp_plan_failure_stage(exc):
    """Infer the earliest failed workflow stage from planner diagnostics."""
    text = str(exc).lower()
    # This invariant is detected before any pregrasp route is expanded.  The
    # explanatory suffix contains the word "pregrasp", so classify it before
    # generic token matching; otherwise the neutral/pregrasp fallback repeats
    # a payload failure that no approach route can change.
    if _lift_failure_is_pose_invariant(exc):
        return "lift"
    # Test later stages first because their diagnostics can include the full
    # plan label and earlier-stage words such as ``pregrasp``.
    if any(
        token in text
        for token in (
            "_transport",
            "transport",
            "_place",
            "released_return",
            "released_retreat",
            "after_release",
        )
    ):
        return "transport"
    if "support_release" in text:
        return "support_release"
    if any(token in text for token in ("_lift", "lift")):
        return "lift"
    if "grasp" in text and "pregrasp" not in text:
        return "grasp"
    if any(token in text for token in ("_to_pregrasp", "pregrasp")):
        return "pregrasp"
    return "unknown"


def _lift_failure_is_pose_invariant(exc):
    """Return true when changing the lift endpoint cannot fix the failure."""
    text = str(exc).lower()
    table_start_failure = (
        "attached_target" in text
        and "table_plane" in text
        and ("sample 0" in text or "sample=0" in text)
    )
    if table_start_failure:
        return True

    # A fast beam may reach the same physical grasp through several IK
    # branches.  If every branch then reports the same closed-tool collision
    # in the first two lift samples, increasing Top-K/beam cannot change that
    # shared start geometry.  Expanding the whole workflow used to repeat the
    # identical pregrasp/grasp work dozens of times before failing again.
    matches = re.findall(
        r"(?:gripper_tool|gripper_closed_finger_[+-]1|attached_target) gets within\s+"
        r"(-?\d+(?:\.\d+)?)mm of environment at sample(?:=|\s+)"
        r"(\d+)",
        text,
    )
    unsafe_count = text.count("unsafemotiontarget:")
    if len(matches) < 2 or unsafe_count != len(matches):
        return False
    clearances = [float(clearance) for clearance, _sample in matches]
    samples = [int(sample) for _clearance, sample in matches]
    return (
        "lift" in text
        and max(samples) <= 2
        and max(clearances) < 0.0
        and max(clearances) - min(clearances) <= 3.0
    )


def _plan_grasp_pose_fallbacks(
    base_pre_tcp,
    base_grasp_tcp,
    base_lift_tcp,
    args,
    plan_attempt,
    lift_endpoint_precheck=None,
):
    """Run only fallbacks capable of changing the stage that failed.

    ``plan_attempt(variant, transport_route_mode, attempt_index)`` performs a
    complete workflow preflight and returns its frozen plan.  Failures are
    returned as structured records so callers can preserve detailed logs.
    """
    variants = grasp_planning_pose_variants(
        base_pre_tcp, base_grasp_tcp, base_lift_tcp, args
    )
    requested_lift = [
        item
        for item in variants
        if abs(float(item["lift_relax_deg"])) <= 1e-9
        and not bool(item.get("lift_height_fallback", False))
        and not bool(item.get("lift_region_fallback", False))
    ]
    lift_fallbacks = [
        item
        for item in variants
        if bool(item.get("lift_height_fallback", False))
        or bool(item.get("lift_region_fallback", False))
        or abs(float(item["lift_relax_deg"])) > 1e-9
    ]
    failures = []
    attempt_index = 0

    def run(variant, transport_route_mode):
        nonlocal attempt_index
        index = attempt_index
        attempt_index += 1
        try:
            if callable(lift_endpoint_precheck) and not bool(
                variant.get("validation_only", False)
            ):
                lift_endpoint_precheck(variant)
            plan = plan_attempt(variant, transport_route_mode, index)
        except (PlanNotExecutable, UnsafeMotionTarget) as exc:
            record = {
                "variant": variant,
                "transport_route_mode": str(transport_route_mode),
                "attempt_index": index,
                "stage": _grasp_plan_failure_stage(exc),
                "exception": exc,
            }
            failures.append(record)
            return None, record
        return plan, None

    def try_transport_fallback(variant):
        for relaxed in _transport_pose_fallback_variants(variant, args):
            plan, _record = run(relaxed, "direct")
            if plan is not None:
                print(
                    "[grasp][workflow][transport-relax] selected bounded "
                    f"place fallback={relaxed['transport_place_relax_kind']} "
                    f"xyz_offset={relaxed['transport_place_xyz_offset']} "
                    f"tilt={relaxed['transport_place_tilt_deg']:+.1f}deg"
                )
                return {
                    "plan": plan,
                    "variant": relaxed,
                    "transport_route_mode": "direct_relaxed_place",
                    "attempt_index": attempt_index - 1,
                }
        if not bool(
            getattr(args, "motion_neutral_transport_fallback", False)
        ):
            return None
        plan, _record = run(variant, "neutral")
        if plan is not None:
            return {
                "plan": plan,
                "variant": variant,
                "transport_route_mode": "neutral",
                "attempt_index": attempt_index - 1,
            }
        return None

    def try_validation_only(variant, *, validation_first=False):
        if not bool(getattr(args, "grasp_validation_mode", False)):
            return None
        validation = dict(variant)
        validation_lift = np.asarray(
            validation["grasp_tcp"], dtype=float
        ).reshape(6).copy()
        validation_height = max(
            float(getattr(args, "grasp_validation_lift_height", 0.045)),
            0.020,
        )
        validation_lift[2] += validation_height
        validation.update(
            {
                "lift_tcp": validation_lift,
                "lift_height": validation_height,
                "lift_height_fallback": True,
                "lift_region_fallback": True,
                "lift_xy_offset": (0.0, 0.0),
                "lift_relax_deg": 0.0,
                "lift_relax_axis": "none",
                "validation_only": True,
                "kind": "grasp_validation_only",
            }
        )
        if validation_first:
            print(
                "[grasp][workflow][validation-first] plan only through "
                "close + verified support release; full lift/transport is "
                "outside this execution plan"
            )
        else:
            print(
                "[grasp][workflow][validation] full lift/transport did not "
                "produce an executable route; reuse the validated grasp core "
                "and plan only through safe support release"
            )
        plan, _record = run(validation, "direct")
        if plan is None:
            return None
        after_close = tuple(getattr(plan, "after_close", ()))
        if after_close:
            actual_release_tcp = np.asarray(
                after_close[-1].tcp_pose, dtype=float
            ).reshape(6)
            validation["lift_tcp"] = actual_release_tcp.copy()
            validation["lift_height"] = float(
                actual_release_tcp[2]
                - np.asarray(validation["grasp_tcp"], dtype=float).reshape(6)[2]
            )
        return {
            "plan": plan,
            "variant": validation,
            "transport_route_mode": "validation_only",
            "attempt_index": attempt_index - 1,
        }

    for pre_variant in requested_lift:
        if bool(getattr(args, "grasp_validation_first", False)) and bool(
            getattr(args, "grasp_validation_mode", False)
        ):
            selected = try_validation_only(
                pre_variant, validation_first=True
            )
            if selected is not None:
                return selected, failures
            # The validation route contains every stage required to establish
            # grasp success.  If that core fails, full lift/transport cannot
            # repair it and must not trigger a larger search.
            validation_stage = failures[-1]["stage"] if failures else "unknown"
            if validation_stage == "pregrasp":
                continue
            return None, failures

        plan, failure = run(pre_variant, "direct")
        if plan is not None:
            return {
                "plan": plan,
                "variant": pre_variant,
                "transport_route_mode": "direct",
                "attempt_index": attempt_index - 1,
            }, failures

        stage = failure["stage"]
        if stage == "pregrasp":
            # Only a pregrasp failure can benefit from another approach
            # distance.  Keep the requested lift orientation unchanged.
            continue
        if bool(getattr(args, "post_grasp_free_route", False)):
            # The free-route planner already searches a high joint region and
            # two bounded place-orientation modes.  Re-entering legacy exact
            # lift-height/orientation/place fallbacks would repeat the costly
            # global work this opt-in mode is specifically meant to replace.
            print(
                "[grasp][workflow][free-route] bounded RRT attempt failed at "
                f"{stage}; skip legacy lift/transport fallback expansion"
            )
            return None, failures
        if stage == "grasp" or stage == "unknown":
            # Grasp TCP and collision geometry are identical for every pose
            # fallback; Top-K/beam IK was already exhausted by this attempt.
            return None, failures
        if stage == "transport":
            selected = try_transport_fallback(pre_variant)
            if selected is None:
                selected = try_validation_only(pre_variant)
            return selected, failures
        if stage == "lift":
            if _lift_failure_is_pose_invariant(failure["exception"]):
                print(
                    "[grasp][workflow][stage-fallback] skip all lift height/"
                    "orientation variants: every retained branch has the "
                    "same invariant lift-start contact; changing the final "
                    "lift endpoint cannot alter the shared start geometry"
                )
                return try_validation_only(pre_variant), failures
            same_pre_lift_fallbacks = [
                item
                for item in lift_fallbacks
                if abs(
                    float(item["pregrasp_offset"])
                    - float(pre_variant["pregrasp_offset"])
                )
                <= 1e-9
            ]
            max_region_attempts = max(
                int(
                    getattr(
                        args, "grasp_lift_region_max_attempts", 18
                    )
                ),
                1,
            )
            same_pre_lift_fallbacks = same_pre_lift_fallbacks[
                : max(max_region_attempts - 1, 0)
            ]
            for lift_variant in same_pre_lift_fallbacks:
                plan, lift_failure = run(lift_variant, "direct")
                if plan is not None:
                    return {
                        "plan": plan,
                        "variant": lift_variant,
                        "transport_route_mode": "direct",
                        "attempt_index": attempt_index - 1,
                    }, failures
                if lift_failure["stage"] == "transport":
                    selected = try_transport_fallback(lift_variant)
                    if selected is None:
                        selected = try_validation_only(lift_variant)
                    return selected, failures
                if _lift_failure_is_pose_invariant(
                    lift_failure["exception"]
                ):
                    print(
                        "[grasp][workflow][stage-fallback] stop remaining "
                        "lift variants: invariant lift-start contact was "
                        "already confirmed across retained IK branches"
                    )
                    return try_validation_only(pre_variant), failures
                if lift_failure["stage"] != "lift":
                    return None, failures
            return try_validation_only(pre_variant), failures
        if stage == "support_release":
            # Pregrasp distance and relaxed final-lift variants cannot change
            # the strict support-release endpoint/path.
            return None, failures
    return None, failures


def _resolve_hook(module_name, name, hook):
    if hook is not None:
        return hook
    if module_name == "collision":
        from . import collision as owner
    elif module_name == "motion_planning":
        from . import motion_planning as owner
    elif module_name == "grasp_geometry":
        from . import grasp_geometry as owner
    elif module_name == "grasp_candidates":
        from . import grasp_candidates as owner
    elif module_name == "transforms":
        from . import transforms as owner
    else:
        raise ValueError(f"unknown grasp-planner dependency owner: {module_name}")
    return getattr(owner, name)


def _candidate_preflight_width_signs_from_hooks(
    row,
    grasp,
    T_base_camera,
    current_tcp_pose,
    args,
    ordered_equivalent_jaw_width_signs,
    filter_equivalent_jaw_width_signs,
):
    ordered_signs, angle_by_sign = ordered_equivalent_jaw_width_signs(
        grasp,
        T_base_camera,
        current_tcp_pose,
        args,
    )
    eligible_signs = list(ordered_signs)
    if row is not None and "physical_aperture_ok_signs" in row:
        eligible_signs = filter_equivalent_jaw_width_signs(
            eligible_signs,
            row.get("physical_aperture_ok_signs", []),
        )
    if (
        row is not None
        and "reachability_ok_signs" in row
        and not bool(getattr(args, "allow_hard_grasp_candidate", False))
    ):
        eligible_signs = filter_equivalent_jaw_width_signs(
            eligible_signs,
            row.get("reachability_ok_signs", []),
        )
    return ordered_signs, eligible_signs, angle_by_sign


def _neutral_joints_for_pregrasp(args, override=None):
    """Resolve one operator-owned neutral joint configuration in radians."""
    value = override
    if value is None:
        value = getattr(args, "motion_neutral_joints_deg", None)
    if value is None:
        value = getattr(args, "initial_joints_deg", None)
    if value is None:
        raise PlanNotExecutable(
            "neutral pregrasp fallback requires --motion-neutral-joints-deg "
            "or --initial-joints-deg"
        )
    joints_deg = np.asarray(value, dtype=float).reshape(-1)
    if joints_deg.size != 7 or not np.isfinite(joints_deg).all():
        raise PlanNotExecutable(
            "neutral pregrasp joint configuration must contain seven finite "
            "joint angles in degrees"
        )
    return np.deg2rad(joints_deg.reshape(7))


def _plan_direct_grasp_sequence_beam(
    arm,
    start_tcp,
    q_start,
    pre_tcp,
    grasp_tcp,
    lift_tcp,
    place_tcp,
    scene,
    args,
    plan_label,
    pregrasp_route_mode,
    transport_route_mode,
    neutral_joints,
    planning_stage_cache,
    planning_metrics,
    validation_only=False,
    *,
    plan_guarded_route,
    motion_context,
    plan_tcp_motion_stage,
    plan_tcp_stage_sequence_beam,
    target_points_in_tcp,
    build_initial_support_release,
    validate_motion_joint_path,
    pose_rpy_error_norm,
    transform_points,
    tcp_transform_from_joints,
    modeled_tcp_pose_for_joints,
):
    """Plan the complete direct workflow with cross-stage IK backtracking."""
    route_mode = str(pregrasp_route_mode or "direct")
    if route_mode not in {"direct", "neutral"}:
        raise ValueError(f"unknown pregrasp route mode: {route_mode}")
    closed_gripper_width = max(
        float(
            getattr(
                args,
                "_planning_closed_gripper_width",
                getattr(args, "motion_closed_gripper_empty_width", 0.000),
            )
        ),
        0.0,
    )

    attached_target_tcp = target_points_in_tcp(
        scene.target_base, grasp_tcp, args
    )
    initial_support_release = build_initial_support_release(
        scene, grasp_tcp, args
    )
    free_context = motion_context(scene, "free_open")
    approach_context = motion_context(scene, "target_approach")
    carrying_context = motion_context(
        scene,
        "carrying",
        attached_target_tcp=attached_target_tcp,
        closed_gripper_width=closed_gripper_width,
    )
    pickup_lift_context = motion_context(
        scene,
        "pickup_lift",
        attached_target_tcp=attached_target_tcp,
        closed_gripper_width=closed_gripper_width,
    )
    free_post_route = bool(
        getattr(args, "post_grasp_free_route", False)
        and place_tcp is not None
        and not bool(validation_only)
    )

    specs = []
    roles = []
    if route_mode == "neutral":
        q_neutral = _neutral_joints_for_pregrasp(args, neutral_joints)
        if float(np.max(np.abs(q_neutral - q_start))) <= np.deg2rad(0.2):
            raise PlanNotExecutable(
                "neutral pregrasp fallback adds no route because the saved "
                "neutral joints already match the current joints"
            )
        neutral_tcp = modeled_tcp_pose_for_joints(q_neutral, args)
        specs.append(
            {
                "tcp_pose": neutral_tcp,
                "context": free_context,
                "label": f"{plan_label}_to_pregrasp_via_neutral_entry",
                "fixed_joints": q_neutral,
            }
        )
        roles.append("neutral")

    def validate_pregrasp_endpoint(_q_previous, stage):
        validate_motion_joint_path(
            stage.joints,
            stage.joints,
            approach_context,
            args,
            f"{plan_label}_pregrasp_target_policy",
        )

    specs.append(
        {
            "tcp_pose": pre_tcp,
            "context": free_context,
            "label": (
                f"{plan_label}_via_neutral_to_pregrasp"
                if route_mode == "neutral"
                else f"{plan_label}_to_pregrasp"
            ),
            "post_validate": validate_pregrasp_endpoint,
        }
    )
    roles.append("pregrasp")
    micro_lift_cache = {}

    def validate_grasp_micro_lift(_q_previous, stage):
        q_grasp = np.asarray(stage.joints, dtype=float).reshape(7)
        key = tuple(np.round(q_grasp, 9).tolist())
        cached = micro_lift_cache.get(key)
        if cached is None:
            try:
                cached = _micro_lift_feasibility(
                    arm,
                    q_grasp,
                    grasp_tcp,
                    args,
                    plan_label,
                )
            except (AttributeError, TypeError, ValueError) as exc:
                # Preserve compatibility with light-weight planner test doubles
                # that do not expose numerical TCP/flange conversion.
                print(
                    f"[{plan_label}][micro-lift][warn] numerical screen "
                    f"unavailable ({type(exc).__name__}: {exc}); defer to "
                    "legacy strict lift planner"
                )
                return
            micro_lift_cache[key] = cached
        if isinstance(planning_metrics, dict):
            previous = planning_metrics.get("micro_lift_joint_margin", None)
            if previous is None or float(cached["micro_lift_joint_margin"]) > float(previous):
                planning_metrics.update(cached)
        print(
            f"[{plan_label}][micro-lift] strict_world_z="
            f"{float(getattr(args, 'grasp_micro_lift_height', 0.008)) * 1000.0:.1f}mm "
            f"margin={float(cached['micro_lift_joint_margin']):.2f}deg "
            f"max_delta={float(cached['micro_lift_max_joint_delta']):.2f}deg"
        )

    specs.append(
        {
            "tcp_pose": grasp_tcp,
            "context": approach_context,
            "label": f"{plan_label}_grasp",
            "post_validate": validate_grasp_micro_lift,
        }
    )
    roles.append("grasp")

    skip_separate_lift = False
    support_spec_index = None
    support_tcp = None
    support_context = None
    validate_support_endpoint = None
    if initial_support_release is not None:
        release_height = max(
            float(getattr(args, "motion_support_release_height", 0.020)),
            float(initial_support_release.strict_after_rise) + 0.001,
        )
        requested_rise = float(lift_tcp[2] - grasp_tcp[2])
        if requested_rise < release_height - 1e-6:
            raise PlanNotExecutable(
                f"requested lift rise={requested_rise:.3f}m is below the "
                f"verified support-release height={release_height:.3f}m"
            )
        support_tcp = np.asarray(grasp_tcp, dtype=float).reshape(6).copy()
        support_tcp[2] += release_height
        support_context = motion_context(
            scene,
            "support_release",
            attached_target_tcp=attached_target_tcp,
            initial_support_release=initial_support_release,
            closed_gripper_width=closed_gripper_width,
        )

        def validate_support_endpoint(_q_previous, stage):
            validate_motion_joint_path(
                stage.joints,
                stage.joints,
                carrying_context,
                args,
                f"{plan_label}_support_release_strict_endpoint",
            )

        support_spec_index = len(specs)
        specs.append(
            {
                "tcp_pose": support_tcp,
                "context": support_context,
                "label": f"{plan_label}_support_release",
                "post_validate": validate_support_endpoint,
            }
        )
        support_is_lift = bool(validation_only) or (
            float(
                np.linalg.norm(
                    np.asarray(lift_tcp, dtype=float)[:3] - support_tcp[:3]
                )
            )
            <= 0.001
            and pose_rpy_error_norm(support_tcp, lift_tcp) <= np.deg2rad(1.0)
        )
        roles.append("lift" if support_is_lift else "support_release")
        skip_separate_lift = bool(support_is_lift)

    # The opt-in free planner always starts after one exact world-Z release.
    # When no explicit table/mat support patch was inferred, retain the same
    # 20 mm geometric separation but use the normal pickup policy.
    if free_post_route and support_spec_index is None:
        release_height = max(
            float(getattr(args, "motion_support_release_height", 0.020)),
            0.020,
        )
        support_tcp = np.asarray(grasp_tcp, dtype=float).reshape(6).copy()
        support_tcp[2] += release_height
        support_context = pickup_lift_context
        support_spec_index = len(specs)
        specs.append(
            {
                "tcp_pose": support_tcp,
                "context": support_context,
                "label": f"{plan_label}_strict_20mm_release",
            }
        )
        roles.append("support_release")

    lift_floor = float(
        support_tcp[2] if support_tcp is not None else grasp_tcp[2]
    ) - 0.005
    lift_context = motion_context(
        scene,
        "carrying" if initial_support_release is not None else "pickup_lift",
        attached_target_tcp=attached_target_tcp,
        min_tcp_z=lift_floor,
        closed_gripper_width=closed_gripper_width,
    )
    if not skip_separate_lift and not free_post_route:
        specs.append(
            plan_post_support_lift(
                lift_tcp,
                lift_context,
                f"{plan_label}_lift",
            )
        )
        roles.append("lift")
    if place_tcp is not None and not bool(validation_only) and not free_post_route:
        transport_mode = str(transport_route_mode or "direct")
        if transport_mode not in {"direct", "neutral"}:
            raise ValueError(
                f"unknown transport route mode: {transport_mode}"
            )
        if transport_mode == "neutral":
            q_transport_neutral = _neutral_joints_for_pregrasp(
                args, neutral_joints
            )
            neutral_tcp = modeled_tcp_pose_for_joints(
                q_transport_neutral, args
            )
            specs.append(
                {
                    "tcp_pose": neutral_tcp,
                    "context": carrying_context,
                    "label": f"{plan_label}_transport_via_neutral",
                    "fixed_joints": q_transport_neutral,
                }
            )
            roles.append("transport_neutral")
        specs.append(
            {
                "tcp_pose": place_tcp,
                "context": carrying_context,
                "label": f"{plan_label}_transport",
            }
        )
        roles.append("place")

    def run_core(stage_specs):
        return plan_tcp_stage_sequence_beam(
            arm,
            stage_specs,
            q_start,
            args,
            f"{plan_label}_{route_mode}",
            beam_width=max(
                int(
                    getattr(
                        args,
                        "grasp_stage_beam_width",
                        getattr(args, "grasp_beam_width", 6),
                    )
                ),
                1,
            ),
            stage_option_cache=planning_stage_cache,
        )

    try:
        core_branches = run_core(specs)
    except (PlanNotExecutable, UnsafeMotionTarget) as direct_exc:
        direct_text = str(direct_exc).lower()
        if (
            support_spec_index is None
            or "support_release" not in direct_text
            or "strict_endpoint" in direct_text
        ):
            raise
        release_height = float(support_tcp[2] - grasp_tcp[2])
        rises = _support_release_rises(release_height)
        segmented_specs = list(specs[:support_spec_index])
        segmented_roles = list(roles[:support_spec_index])
        for segment_index, rise in enumerate(rises, start=1):
            segment_tcp = np.asarray(grasp_tcp, dtype=float).reshape(6).copy()
            segment_tcp[2] += rise
            is_final = abs(rise - release_height) <= 1e-9
            segment_context = replace(
                support_context,
                require_support_release_latch_at_endpoint=is_final,
            )
            segment_spec = {
                "tcp_pose": segment_tcp,
                "context": segment_context,
                "label": (
                    f"{plan_label}_support_release_segment"
                    f"{segment_index}_{rise * 1000.0:.0f}mm"
                ),
            }
            if is_final:
                segment_spec["post_validate"] = validate_support_endpoint
            segmented_specs.append(segment_spec)
            segmented_roles.append(
                "lift" if is_final and skip_separate_lift else "support_release"
            )
        segmented_specs.extend(specs[support_spec_index + 1 :])
        segmented_roles.extend(roles[support_spec_index + 1 :])
        print(
            f"[{plan_label}][support-release] direct strict +"
            f"{release_height * 1000.0:.0f}mm endpoint failed; retry "
            f"segmented world-Z rises={[round(v * 1000.0) for v in rises]}mm"
        )
        core_branches = run_core(segmented_specs)
        specs = segmented_specs
        roles = segmented_roles
    completion_failures = []
    for branch_index, (stages, beam_score) in enumerate(
        core_branches, start=1
    ):
        stages = tuple(stages)
        try:
            grasp_index = roles.index("grasp")
            before_close = list(stages[: grasp_index + 1])
            after_close = list(stages[grasp_index + 1 :])
            grasp_stage = stages[grasp_index]
            placed_target_base = None
            after_release = []

            actual_place_tcp = place_tcp
            if free_post_route:
                if not after_close:
                    raise PlanNotExecutable(
                        f"{plan_label} free post-grasp route has no validated "
                        "20 mm support-release endpoint"
                    )
                from .post_grasp_rrt import plan_post_grasp_free_route

                free_stages, actual_place_tcp = plan_post_grasp_free_route(
                    arm,
                    after_close[-1].joints,
                    grasp_tcp,
                    place_tcp,
                    carrying_context,
                    args,
                    f"{plan_label}_post_grasp_free",
                    validate_motion_joint_path=validate_motion_joint_path,
                    modeled_tcp_pose_for_joints=modeled_tcp_pose_for_joints,
                )
                after_close.extend(free_stages)
                q_place = np.asarray(after_close[-1].joints, dtype=float).reshape(7)
                placed_target_base = transform_points(
                    tcp_transform_from_joints(q_place, args),
                    attached_target_tcp,
                )
            else:
                lift_index = roles.index("lift")
                lift_stage = stages[lift_index]

            if place_tcp is not None and not bool(validation_only):
                if not free_post_route:
                    place_index = roles.index("place")
                    q_place = np.asarray(
                        stages[place_index].joints, dtype=float
                    ).reshape(7)
                    placed_target_base = transform_points(
                        tcp_transform_from_joints(q_place, args),
                        attached_target_tcp,
                    )
                if not bool(args.skip_initial_move):
                    home_joints = np.deg2rad(
                        np.asarray(
                            args.initial_joints_deg, dtype=float
                        ).reshape(7)
                    )
                    home_tcp = modeled_tcp_pose_for_joints(home_joints, args)
                    after_release.extend(
                        plan_guarded_route(
                            arm,
                            actual_place_tcp,
                            home_tcp,
                            q_place,
                            scene,
                            args,
                            f"{plan_label}_released_return",
                            transit_phase="released_free",
                            goal_phase="released_free",
                            first_phase="released_retreat",
                            placed_target_base=placed_target_base,
                            goal_joints=home_joints,
                        )
                    )
                else:
                    retreat_tcp = np.asarray(actual_place_tcp, dtype=float).copy()
                    retreat_tcp[2] += max(
                        float(
                            getattr(
                                args,
                                "motion_direct_release_retreat",
                                0.020,
                            )
                        ),
                        0.010,
                    )
                    retreat_context = motion_context(
                        scene,
                        "released_retreat",
                        placed_target_base=placed_target_base,
                    )
                    after_release.append(
                        plan_tcp_motion_stage(
                            arm,
                            retreat_tcp,
                            q_place,
                            retreat_context,
                            args,
                            f"{plan_label}_released_retreat",
                        )
                    )

            empty_recovery = []
            q_previous = np.asarray(grasp_stage.joints, dtype=float).reshape(7)
            for reverse_index, original in enumerate(
                reversed(before_close[:-1])
            ):
                phase = (
                    "target_approach"
                    if reverse_index == 0
                    or str(original.phase) == "target_approach"
                    else "free_open"
                )
                context = motion_context(
                    scene, phase, min_tcp_z=original.min_tcp_z
                )
                validate_motion_joint_path(
                    q_previous,
                    original.joints,
                    context,
                    args,
                    f"{plan_label}_empty_reverse_{original.label}",
                )
                empty_recovery.append(
                    PlannedJointStage(
                        label=f"{plan_label}_empty_reverse_{original.label}",
                        tcp_pose=np.asarray(original.tcp_pose, dtype=float).copy(),
                        flange_pose=np.asarray(
                            original.flange_pose, dtype=float
                        ).copy(),
                        joints=np.asarray(original.joints, dtype=float).copy(),
                        phase=phase,
                        min_tcp_z=original.min_tcp_z,
                        runtime_min_tcp_z_slack=(
                            original.runtime_min_tcp_z_slack
                        ),
                    )
                )
                q_previous = np.asarray(
                    original.joints, dtype=float
                ).reshape(7)
                if bool(
                    getattr(args, "skip_empty_grasp_return", False)
                ) and reverse_index == 0:
                    break

            if not bool(
                getattr(args, "skip_empty_grasp_return", False)
            ) and not bool(args.skip_initial_move):
                home_joints = np.deg2rad(
                    np.asarray(args.initial_joints_deg, dtype=float).reshape(7)
                )
                if float(np.max(np.abs(q_previous - home_joints))) > np.deg2rad(
                    0.2
                ):
                    recovery_start_tcp = (
                        np.asarray(
                            empty_recovery[-1].tcp_pose, dtype=float
                        )
                        if empty_recovery
                        else start_tcp
                    )
                    home_tcp = modeled_tcp_pose_for_joints(home_joints, args)
                    empty_recovery.extend(
                        plan_guarded_route(
                            arm,
                            recovery_start_tcp,
                            home_tcp,
                            q_previous,
                            scene,
                            args,
                            f"{plan_label}_empty_return_home",
                            transit_phase="free_open",
                            goal_phase="free_open",
                            goal_joints=home_joints,
                        )
                    )

            print(
                f"[motion][beam] selected complete branch "
                f"{branch_index}/{len(core_branches)} score={beam_score:.4f} "
                f"route={route_mode}"
            )
            print(
                f"[motion][plan] full sequence validated before execution: "
                f"before_close={len(before_close)} "
                f"after_close={len(after_close)} "
                f"after_release={len(after_release)} "
                f"empty_recovery={len(empty_recovery)}"
            )
            return GraspMotionPlan(
                start_joints=q_start,
                before_close=tuple(before_close),
                after_close=tuple(after_close),
                after_release=tuple(after_release),
                empty_recovery=tuple(empty_recovery),
                safety_scene=scene,
                attached_target_tcp=attached_target_tcp,
                placed_target_base=placed_target_base,
                initial_support_release=initial_support_release,
                closed_gripper_width=closed_gripper_width,
                validation_only=bool(validation_only),
            )
        except (PlanNotExecutable, UnsafeMotionTarget) as exc:
            completion_failures.append(
                f"branch={branch_index}:{type(exc).__name__}:{exc}"
            )
            print(
                f"[{plan_label}][beam][completion_reject] "
                f"{completion_failures[-1]}"
            )
    raise PlanNotExecutable(
        f"{plan_label} all {len(core_branches)} core beam branch(es) failed "
        "release/return/recovery validation: "
        + " | ".join(completion_failures[-6:])
    )


def plan_up_and_over_grasp_sequence(
    arm,
    pre_tcp,
    grasp_tcp,
    lift_tcp,
    place_tcp,
    target_cloud,
    motion_scene_cloud,
    T_base_camera,
    args,
    plan_label,
    motion_environment_cloud=None,
    motion_environment_map=None,
    pregrasp_route_mode="direct",
    transport_route_mode="direct",
    neutral_joints=None,
    planning_stage_cache=None,
    planning_metrics=None,
    validation_only=False,
    *,
    prepare_motion_safety_scene_hook=None,
    assert_pickup_payload_table_start_hook=None,
    plan_guarded_route_hook=None,
    motion_context_hook=None,
    plan_tcp_motion_stage_hook=None,
    plan_tcp_stage_sequence_beam_hook=None,
    target_points_in_tcp_hook=None,
    build_initial_support_release_hook=None,
    validate_motion_joint_path_hook=None,
    pose_rpy_error_norm_hook=None,
    transform_points_hook=None,
    tcp_transform_from_joints_hook=None,
    modeled_tcp_pose_for_joints_hook=None,
):
    """TRAC-IK plan and validate every success/empty-grasp motion before motion."""
    if str(getattr(args, "motion_mode", "cartesian")) != "ik_joint":
        raise PlanNotExecutable(
            "whole-workflow collision guard requires --motion-mode ik_joint"
        )
    closed_gripper_width = max(
        float(
            getattr(
                args,
                "_planning_closed_gripper_width",
                getattr(args, "motion_closed_gripper_empty_width", 0.000),
            )
        ),
        0.0,
    )

    prepare_motion_safety_scene_fn = _resolve_hook(
        "collision", "prepare_motion_safety_scene", prepare_motion_safety_scene_hook
    )
    assert_pickup_payload_table_start_fn = _resolve_hook(
        "collision",
        "assert_pickup_payload_table_start",
        assert_pickup_payload_table_start_hook,
    )
    plan_guarded_route_fn = _resolve_hook(
        "motion_planning", "_plan_guarded_route", plan_guarded_route_hook
    )
    motion_context_fn = _resolve_hook(
        "motion_planning", "_motion_context", motion_context_hook
    )
    plan_tcp_motion_stage_fn = _resolve_hook(
        "motion_planning", "_plan_tcp_motion_stage", plan_tcp_motion_stage_hook
    )
    plan_tcp_stage_sequence_beam_fn = _resolve_hook(
        "motion_planning",
        "_plan_tcp_stage_sequence_beam",
        plan_tcp_stage_sequence_beam_hook,
    )
    target_points_in_tcp_fn = _resolve_hook(
        "collision", "_target_points_in_tcp", target_points_in_tcp_hook
    )
    build_initial_support_release_fn = _resolve_hook(
        "collision", "_build_initial_support_release", build_initial_support_release_hook
    )
    validate_motion_joint_path_fn = _resolve_hook(
        "collision", "validate_motion_joint_path", validate_motion_joint_path_hook
    )
    pose_rpy_error_norm_fn = _resolve_hook(
        "transforms", "_pose_rpy_error_norm", pose_rpy_error_norm_hook
    )
    transform_points_fn = _resolve_hook(
        "transforms", "_transform_points", transform_points_hook
    )
    tcp_transform_from_joints_fn = _resolve_hook(
        "collision", "_tcp_transform_from_joints", tcp_transform_from_joints_hook
    )
    modeled_tcp_pose_for_joints_fn = _resolve_hook(
        "motion_planning", "_modeled_tcp_pose_for_joints", modeled_tcp_pose_for_joints_hook
    )

    q_start = np.asarray(arm.get_joint_angles(), dtype=float).reshape(7)
    scene_kwargs = {
        "map_id": f"{plan_label}_preflight",
        "motion_environment_cloud": motion_environment_cloud,
        "current_joints": q_start,
    }
    # Keep injected legacy/test hooks compatible until a canonical environment
    # map is actually available for this capture.
    if motion_environment_map is not None:
        scene_kwargs["environment_map"] = motion_environment_map
    scene = prepare_motion_safety_scene_fn(
        target_cloud,
        motion_scene_cloud,
        T_base_camera,
        args,
        **scene_kwargs,
    )
    assert_pickup_payload_table_start_fn(
        scene,
        args,
        f"{plan_label}_pickup_start",
    )
    start_tcp = np.asarray(arm.get_tcp_pose(), dtype=float).reshape(6)

    if bool(getattr(args, "grasp_stage_beam_search", False)) and not bool(
        getattr(args, "motion_up_and_over", False)
    ):
        return _plan_direct_grasp_sequence_beam(
            arm,
            start_tcp,
            q_start,
            np.asarray(pre_tcp, dtype=float).reshape(6),
            np.asarray(grasp_tcp, dtype=float).reshape(6),
            np.asarray(lift_tcp, dtype=float).reshape(6),
            (
                None
                if place_tcp is None
                else np.asarray(place_tcp, dtype=float).reshape(6)
            ),
            scene,
            args,
            plan_label,
            pregrasp_route_mode,
            transport_route_mode,
            neutral_joints,
            planning_stage_cache,
            planning_metrics,
            bool(validation_only),
            plan_guarded_route=plan_guarded_route_fn,
            motion_context=motion_context_fn,
            plan_tcp_motion_stage=plan_tcp_motion_stage_fn,
            plan_tcp_stage_sequence_beam=plan_tcp_stage_sequence_beam_fn,
            target_points_in_tcp=target_points_in_tcp_fn,
            build_initial_support_release=build_initial_support_release_fn,
            validate_motion_joint_path=validate_motion_joint_path_fn,
            pose_rpy_error_norm=pose_rpy_error_norm_fn,
            transform_points=transform_points_fn,
            tcp_transform_from_joints=tcp_transform_from_joints_fn,
            modeled_tcp_pose_for_joints=modeled_tcp_pose_for_joints_fn,
        )

    if str(pregrasp_route_mode or "direct") == "neutral":
        q_neutral = _neutral_joints_for_pregrasp(args, neutral_joints)
        neutral_tcp = modeled_tcp_pose_for_joints_fn(q_neutral, args)
        before_close = list(
            plan_guarded_route_fn(
                arm,
                start_tcp,
                neutral_tcp,
                q_start,
                scene,
                args,
                f"{plan_label}_to_pregrasp_via_neutral_entry",
                transit_phase="free_open",
                goal_phase="free_open",
                goal_joints=q_neutral,
            )
        )
        before_close.extend(
            plan_guarded_route_fn(
                arm,
                neutral_tcp,
                pre_tcp,
                np.asarray(before_close[-1].joints, dtype=float).reshape(7),
                scene,
                args,
                f"{plan_label}_via_neutral_to_pregrasp",
                transit_phase="free_open",
                goal_phase="target_approach",
            )
        )
    elif str(pregrasp_route_mode or "direct") == "direct":
        before_close = list(
            plan_guarded_route_fn(
                arm,
                start_tcp,
                pre_tcp,
                q_start,
                scene,
                args,
                f"{plan_label}_to_pregrasp",
                transit_phase="free_open",
                goal_phase="target_approach",
            )
        )
    else:
        raise ValueError(
            f"unknown pregrasp route mode: {pregrasp_route_mode}"
        )
    q_previous = np.asarray(before_close[-1].joints, dtype=float).reshape(7)
    approach_context = motion_context_fn(
        scene, "target_approach"
    )
    grasp_stage = plan_tcp_motion_stage_fn(
        arm,
        grasp_tcp,
        q_previous,
        approach_context,
        args,
        f"{plan_label}_grasp",
    )
    before_close.append(grasp_stage)
    try:
        micro_metrics = _micro_lift_feasibility(
            arm,
            grasp_stage.joints,
            grasp_tcp,
            args,
            plan_label,
        )
        if isinstance(planning_metrics, dict):
            planning_metrics.update(micro_metrics)
    except (AttributeError, TypeError, ValueError) as exc:
        print(
            f"[{plan_label}][micro-lift][warn] numerical screen unavailable "
            f"({type(exc).__name__}: {exc}); defer to legacy strict lift "
            "planner"
        )

    attached_target_tcp = target_points_in_tcp_fn(
        scene.target_base, grasp_tcp, args
    )
    initial_support_release = build_initial_support_release_fn(
        scene, grasp_tcp, args
    )
    carrying_context = motion_context_fn(
        scene,
        "carrying",
        attached_target_tcp=attached_target_tcp,
        closed_gripper_width=closed_gripper_width,
    )
    pickup_lift_context = motion_context_fn(
        scene,
        "pickup_lift",
        attached_target_tcp=attached_target_tcp,
        closed_gripper_width=closed_gripper_width,
    )
    after_close = []
    q_after_release = np.asarray(grasp_stage.joints, dtype=float).reshape(7)
    lift_start_tcp = np.asarray(grasp_tcp, dtype=float).reshape(6)
    if initial_support_release is not None:
        release_height = max(
            float(getattr(args, "motion_support_release_height", 0.020)),
            float(initial_support_release.strict_after_rise) + 0.001,
        )
        requested_rise = float(lift_tcp[2] - grasp_tcp[2])
        if requested_rise < release_height - 1e-6:
            raise PlanNotExecutable(
                f"requested lift rise={requested_rise:.3f}m is below the "
                f"verified support-release height={release_height:.3f}m"
            )
        support_tcp = np.asarray(grasp_tcp, dtype=float).reshape(6).copy()
        support_tcp[2] += release_height
        support_context = motion_context_fn(
            scene,
            "support_release",
            attached_target_tcp=attached_target_tcp,
            initial_support_release=initial_support_release,
            closed_gripper_width=closed_gripper_width,
        )
        try:
            support_stage = plan_tcp_motion_stage_fn(
                arm,
                support_tcp,
                q_after_release,
                support_context,
                args,
                f"{plan_label}_support_release",
            )
            support_stages = [support_stage]
        except (PlanNotExecutable, UnsafeMotionTarget) as direct_exc:
            support_stages = []
            q_segment = q_after_release.copy()
            rises = _support_release_rises(release_height)
            print(
                f"[{plan_label}][support-release] direct strict endpoint "
                f"failed ({type(direct_exc).__name__}: {direct_exc}); "
                f"retry segmented rises="
                f"{[round(v * 1000.0) for v in rises]}mm"
            )
            for segment_index, rise in enumerate(rises, start=1):
                segment_tcp = np.asarray(grasp_tcp, dtype=float).reshape(6).copy()
                segment_tcp[2] += rise
                is_final = abs(rise - release_height) <= 1e-9
                segment_context = replace(
                    support_context,
                    require_support_release_latch_at_endpoint=is_final,
                )
                segment = plan_tcp_motion_stage_fn(
                    arm,
                    segment_tcp,
                    q_segment,
                    segment_context,
                    args,
                    f"{plan_label}_support_release_segment{segment_index}_"
                    f"{rise * 1000.0:.0f}mm",
                )
                support_stages.append(segment)
                q_segment = np.asarray(segment.joints, dtype=float).reshape(7)
            support_stage = support_stages[-1]
        # The endpoint must independently pass the normal carrying policy;
        # otherwise a release that merely postpones a table/neighbor collision
        # is not accepted.
        validate_motion_joint_path_fn(
            support_stage.joints,
            support_stage.joints,
            carrying_context,
            args,
            f"{plan_label}_support_release_strict_endpoint",
        )
        after_close.extend(support_stages)
        q_after_release = np.asarray(support_stage.joints, dtype=float).reshape(7)
        lift_start_tcp = support_tcp

    if bool(validation_only) and after_close:
        lift_stage = after_close[-1]
    elif (
        not after_close
        or float(np.linalg.norm(np.asarray(lift_tcp)[:3] - lift_start_tcp[:3]))
        > 0.001
        or pose_rpy_error_norm_fn(lift_start_tcp, lift_tcp) > np.deg2rad(1.0)
    ):
        lift_context = motion_context_fn(
            scene,
            "carrying" if after_close else "pickup_lift",
            attached_target_tcp=attached_target_tcp,
            min_tcp_z=float(lift_start_tcp[2]) - 0.005,
            closed_gripper_width=closed_gripper_width,
        )
        lift_stage = plan_post_support_lift(
            lift_tcp,
            lift_context,
            f"{plan_label}_lift",
            planner=plan_tcp_motion_stage_fn,
            arm=arm,
            q_previous=q_after_release,
            args=args,
        )
        after_close.append(lift_stage)
    else:
        lift_stage = after_close[-1]
    placed_target_base = None
    after_release = []

    if place_tcp is not None and not bool(validation_only):
        transport_mode = str(transport_route_mode or "direct")
        if transport_mode == "neutral":
            q_transport_neutral = _neutral_joints_for_pregrasp(
                args, neutral_joints
            )
            neutral_tcp = modeled_tcp_pose_for_joints_fn(
                q_transport_neutral, args
            )
            place_route = list(
                plan_guarded_route_fn(
                    arm,
                    lift_tcp,
                    neutral_tcp,
                    lift_stage.joints,
                    scene,
                    args,
                    f"{plan_label}_transport_via_neutral",
                    transit_phase="carrying",
                    goal_phase="carrying",
                    attached_target_tcp=attached_target_tcp,
                    goal_joints=q_transport_neutral,
                )
            )
            place_route.extend(
                plan_guarded_route_fn(
                    arm,
                    neutral_tcp,
                    place_tcp,
                    np.asarray(place_route[-1].joints, dtype=float).reshape(7),
                    scene,
                    args,
                    f"{plan_label}_transport",
                    transit_phase="carrying",
                    goal_phase="carrying",
                    attached_target_tcp=attached_target_tcp,
                )
            )
        elif transport_mode == "direct":
            place_route = plan_guarded_route_fn(
                arm,
                lift_tcp,
                place_tcp,
                lift_stage.joints,
                scene,
                args,
                f"{plan_label}_transport",
                transit_phase="carrying",
                goal_phase="carrying",
                attached_target_tcp=attached_target_tcp,
            )
        else:
            raise ValueError(
                f"unknown transport route mode: {transport_mode}"
            )
        after_close.extend(place_route)
        q_place = np.asarray(after_close[-1].joints, dtype=float).reshape(7)
        placed_target_base = transform_points_fn(
            tcp_transform_from_joints_fn(q_place, args), attached_target_tcp
        )

        if not bool(args.skip_initial_move):
            home_joints = np.deg2rad(
                np.asarray(args.initial_joints_deg, dtype=float).reshape(7)
            )
            home_tcp = modeled_tcp_pose_for_joints_fn(home_joints, args)
            after_release.extend(
                plan_guarded_route_fn(
                    arm,
                    place_tcp,
                    home_tcp,
                    q_place,
                    scene,
                    args,
                    f"{plan_label}_released_return",
                    transit_phase="released_free",
                    goal_phase="released_free",
                    first_phase="released_retreat",
                    placed_target_base=placed_target_base,
                    goal_joints=home_joints,
                )
            )
        else:
            retreat_tcp = np.asarray(place_tcp, dtype=float).copy()
            if bool(getattr(args, "motion_up_and_over", False)):
                retreat_tcp[2] = max(
                    float(retreat_tcp[2])
                    + float(
                        getattr(args, "motion_transit_vertical_clearance", 0.080)
                    ),
                    float(lift_tcp[2]),
                )
            else:
                # With high routing disabled, keep only the local release
                # separation needed to get the fingers clear of the object.
                retreat_tcp[2] += max(
                    float(
                            getattr(args, "motion_direct_release_retreat", 0.020)
                    ),
                    0.010,
                )
            retreat_context = motion_context_fn(
                scene,
                "released_retreat",
                placed_target_base=placed_target_base,
            )
            after_release.append(
                plan_tcp_motion_stage_fn(
                    arm,
                    retreat_tcp,
                    q_place,
                    retreat_context,
                    args,
                    f"{plan_label}_released_retreat",
                )
            )

    # Empty-grasp recovery is prevalidated too: reverse the exact outbound
    # route, then use the configured guarded route to the saved initial joints.
    empty_recovery = []
    q_previous = np.asarray(grasp_stage.joints, dtype=float).reshape(7)
    for reverse_index, original in enumerate(reversed(before_close[:-1])):
        # The first empty-grasp recovery edge always starts at the grasp pose
        # and backs out to the pregrasp route.  It is the exact reverse of the
        # final target approach even when the stored outbound route stage was
        # labelled free_open, so apply the same narrow contact policy here.
        phase = (
            "target_approach"
            if reverse_index == 0
            or str(original.phase) == "target_approach"
            else "free_open"
        )
        context = motion_context_fn(
            scene, phase, min_tcp_z=original.min_tcp_z
        )
        validate_motion_joint_path_fn(
            q_previous,
            original.joints,
            context,
            args,
            f"{plan_label}_empty_reverse_{original.label}",
        )
        empty_recovery.append(
            PlannedJointStage(
                label=f"{plan_label}_empty_reverse_{original.label}",
                tcp_pose=np.asarray(original.tcp_pose, dtype=float).copy(),
                flange_pose=np.asarray(original.flange_pose, dtype=float).copy(),
                joints=np.asarray(original.joints, dtype=float).copy(),
                phase=phase,
                min_tcp_z=original.min_tcp_z,
                runtime_min_tcp_z_slack=original.runtime_min_tcp_z_slack,
            )
        )
        q_previous = np.asarray(original.joints, dtype=float).reshape(7)
        if bool(getattr(args, "skip_empty_grasp_return", False)) and reverse_index == 0:
            break

    if not bool(getattr(args, "skip_empty_grasp_return", False)) and not bool(
        args.skip_initial_move
    ):
        home_joints = np.deg2rad(
            np.asarray(args.initial_joints_deg, dtype=float).reshape(7)
        )
        if float(np.max(np.abs(q_previous - home_joints))) > np.deg2rad(0.2):
            recovery_start_tcp = (
                np.asarray(empty_recovery[-1].tcp_pose, dtype=float)
                if empty_recovery
                else start_tcp
            )
            home_tcp = modeled_tcp_pose_for_joints_fn(home_joints, args)
            empty_recovery.extend(
                plan_guarded_route_fn(
                    arm,
                    recovery_start_tcp,
                    home_tcp,
                    q_previous,
                    scene,
                    args,
                    f"{plan_label}_empty_return_home",
                    transit_phase="free_open",
                    goal_phase="free_open",
                    goal_joints=home_joints,
                )
            )

    print(
        f"[motion][plan] full sequence validated before execution: "
        f"before_close={len(before_close)} after_close={len(after_close)} "
        f"after_release={len(after_release)} empty_recovery={len(empty_recovery)}"
    )
    return GraspMotionPlan(
        start_joints=q_start,
        before_close=tuple(before_close),
        after_close=tuple(after_close),
        after_release=tuple(after_release),
        empty_recovery=tuple(empty_recovery),
        safety_scene=scene,
        attached_target_tcp=attached_target_tcp,
        placed_target_base=placed_target_base,
        initial_support_release=initial_support_release,
        closed_gripper_width=closed_gripper_width,
        validation_only=bool(validation_only),
    )


def _grasp_plan_joint_route_cost(plan):
    required = ("start_joints", "before_close", "after_close", "after_release")
    if not all(hasattr(plan, name) for name in required):
        # Lightweight planner hooks in unit tests may return an opaque sentinel.
        # A real GraspMotionPlan always carries the fields above.
        return 0.0, 0.0, 0.0
    stages = (
        tuple(plan.before_close)
        + tuple(plan.after_close)
        + tuple(plan.after_release)
    )
    previous = np.asarray(plan.start_joints, dtype=float).reshape(7)
    total_l2_deg = 0.0
    max_step_deg = 0.0
    wrist_sweep_deg = 0.0

    for stage in stages:
        joints = np.asarray(stage.joints, dtype=float).reshape(7)
        delta_deg = np.rad2deg(joints - previous)
        total_l2_deg += float(np.linalg.norm(delta_deg))
        max_step_deg = max(
            max_step_deg,
            float(np.max(np.abs(delta_deg))),
        )
        wrist_sweep_deg += float(
            abs(delta_deg[4]) + abs(delta_deg[6])
        )
        previous = joints

    return total_l2_deg, max_step_deg, wrist_sweep_deg


def _ordered_grasp_roll_offsets(
    grasp,
    T_base_camera,
    current_tcp_pose,
    width_sign,
    width_shift,
    arm,
    args,
    candidate_idx,
    grasp_to_tcp_poses_fn,
):
    """Order roll offsets cheaply; complete-route cost makes the final choice."""
    roll_angles = [
        float(value)
        for value in getattr(args, "grasp_roll_candidates_deg", [0.0])
        if np.isfinite(float(value))
    ]
    if len(roll_angles) <= 1:
        return roll_angles if roll_angles else [0.0]
    try:
        current_joints = np.asarray(arm.get_joint_angles(), dtype=float).reshape(7)
    except Exception:
        return roll_angles

    roll_args = copy(args)
    roll_args.ik_max_joint_delta = None  # measure swing, do not gate it here
    ranked = []
    for offset in roll_angles:
        try:
            _, grasp_tcp, _ = grasp_to_tcp_poses_fn(
                grasp,
                T_base_camera,
                roll_args,
                current_tcp_pose,
                width_sign=width_sign,
                width_shift=width_shift,
                roll_offset_deg=offset,
            )
            flange_pose = np.asarray(
                arm.get_tcp2flange_pose(grasp_tcp), dtype=float
            ).reshape(6)
            records = ik_solver._ranked_ik_solution_records_for_flange_pose(
                arm,
                flange_pose,
                roll_args,
                f"{candidate_idx}_grasp_roll_select",
                current_joints_override=current_joints,
                quiet=True,
                top_k=1,
            )
            if not records:
                continue
            q = np.asarray(records[0]["joints"], dtype=float).reshape(7)
            swing = float(np.max(np.abs(q - current_joints)))
        except Exception:
            continue
        ranked.append((swing, abs(offset), offset))

    if not ranked:
        return roll_angles

    ranked.sort()
    ordered = [float(item[2]) for item in ranked]
    print(
        f"[grasp][roll-order] candidate={candidate_idx} "
        f"width_sign={int(width_sign):+d} order={ordered}"
    )
    return ordered


def annotate_grasp_candidate_workflow_preflight(
    rows,
    arm,
    args,
    target_cloud,
    motion_scene_cloud,
    T_base_camera,
    current_tcp_pose,
    plan_label,
    place_xyz=None,
    motion_environment_cloud=None,
    motion_environment_map=None,
    *,
    filter_grasp_candidates_by_width_hook=None,
    filter_grasp_candidates_by_table_guard_hook=None,
    defer_candidate_endpoint_reachability_hook=None,
    filter_grasp_candidates_by_reachability_hook=None,
    candidate_preflight_width_signs_hook=None,
    ordered_equivalent_jaw_width_signs_hook=None,
    filter_equivalent_jaw_width_signs_hook=None,
    grasp_candidate_width_shift_hook=None,
    grasp_width_m_hook=None,
    grasp_to_tcp_poses_hook=None,
    assert_table_guard_target_tcp_hook=None,
    plan_up_and_over_grasp_sequence_hook=None,
):
    """Validate every selectable candidate before asking the user to choose.

    The earlier reachability annotation checks only the pregrasp/grasp/lift
    endpoint IK sequence.  This second pass runs the same complete
    collision-checked workflow that execution will use and caches the valid
    plan on the row.  Consequently, a manually selected candidate cannot be
    advertised as executable and then immediately rejected by a later filter.
    """
    planning_started = time.perf_counter()
    rows = [dict(row) for row in rows]
    planning_stats = {
        "input_candidate_count": len(rows),
        "cheap_eligible_count": 0,
        "full_preflight_candidate_count": 0,
        "valid_plan_count": 0,
        "ik_calls": 0,
        "ik_seed_trials": 0,
        "fast_ik_calls": 0,
        "full_ik_calls": 0,
        "collision_configuration_evaluations": 0,
        "failure_stages": {
            key: 0
            for key in (
                "pregrasp",
                "grasp",
                "support_release",
                "lift",
                "transport",
                "unknown",
            )
        },
        "candidate_planning_seconds": {},
    }
    args._planning_stats = planning_stats
    args._planning_ik_mode = "full"
    if not bool(getattr(args, "grasp_motion_path_guard", True)):
        return rows

    filter_grasp_candidates_by_width_fn = _resolve_hook(
        "grasp_candidates",
        "filter_grasp_candidates_by_width",
        filter_grasp_candidates_by_width_hook,
    )
    filter_grasp_candidates_by_table_guard_fn = _resolve_hook(
        "grasp_candidates",
        "filter_grasp_candidates_by_table_guard",
        filter_grasp_candidates_by_table_guard_hook,
    )
    defer_candidate_endpoint_reachability_fn = _resolve_hook(
        "grasp_candidates",
        "_defer_candidate_endpoint_reachability",
        defer_candidate_endpoint_reachability_hook,
    )
    filter_grasp_candidates_by_reachability_fn = _resolve_hook(
        "grasp_candidates",
        "filter_grasp_candidates_by_reachability",
        filter_grasp_candidates_by_reachability_hook,
    )
    if candidate_preflight_width_signs_hook is not None:
        candidate_preflight_width_signs_fn = candidate_preflight_width_signs_hook
    elif (
        ordered_equivalent_jaw_width_signs_hook is None
        and filter_equivalent_jaw_width_signs_hook is None
    ):
        candidate_preflight_width_signs_fn = _resolve_hook(
            "grasp_candidates",
            "_candidate_preflight_width_signs",
            None,
        )
    else:
        ordered_equivalent_jaw_width_signs_fn = _resolve_hook(
            "grasp_candidates",
            "_ordered_equivalent_jaw_width_signs",
            ordered_equivalent_jaw_width_signs_hook,
        )
        filter_equivalent_jaw_width_signs_fn = _resolve_hook(
            "grasp_candidates",
            "_filter_equivalent_jaw_width_signs",
            filter_equivalent_jaw_width_signs_hook,
        )

        def candidate_preflight_width_signs_fn(
            row,
            grasp,
            T_base_camera,
            current_tcp_pose,
            args,
        ):
            return _candidate_preflight_width_signs_from_hooks(
                row,
                grasp,
                T_base_camera,
                current_tcp_pose,
                args,
                ordered_equivalent_jaw_width_signs_fn,
                filter_equivalent_jaw_width_signs_fn,
            )

    grasp_candidate_width_shift_fn = _resolve_hook(
        "grasp_candidates",
        "grasp_candidate_width_shift",
        grasp_candidate_width_shift_hook,
    )
    grasp_width_m_fn = _resolve_hook(
        "grasp_candidates", "grasp_width_m", grasp_width_m_hook
    )
    grasp_to_tcp_poses_fn = _resolve_hook(
        "grasp_geometry", "grasp_to_tcp_poses", grasp_to_tcp_poses_hook
    )
    assert_table_guard_target_tcp_fn = _resolve_hook(
        "collision",
        "assert_table_guard_target_tcp",
        assert_table_guard_target_tcp_hook,
    )
    plan_up_and_over_grasp_sequence_fn = (
        plan_up_and_over_grasp_sequence
        if plan_up_and_over_grasp_sequence_hook is None
        else plan_up_and_over_grasp_sequence_hook
    )

    # A row can be re-annotated after a new view/scene update.  Never allow a
    # plan or success flag produced for an older point cloud or robot state to
    # qualify the candidate in the current scene.
    for row in rows:
        for key in list(row):
            if key.startswith("_workflow_") or key.startswith(
                "workflow_preflight_"
            ):
                row.pop(key, None)
        row["workflow_preflight_ok"] = False
        row["workflow_preflight_reason"] = (
            "native GenX mesh/table rejected; retained for pose refinement only"
            if row.get("genx_refinement_only", False)
            else "not eligible for complete preflight"
        )

    # Deferred native GenX poses are repair inputs, never executable proposals.
    # A corrected pose clears this marker only after measured contact and CAD
    # geometry have passed in the numerical worker, then runs normal preflight.
    eligible = filter_grasp_candidates_by_width_fn(
        [row for row in rows if not row.get("genx_refinement_only", False)], args)
    eligible = filter_grasp_candidates_by_table_guard_fn(eligible, args)
    if (
        not defer_candidate_endpoint_reachability_fn(args)
        and not bool(getattr(args, "allow_hard_grasp_candidate", False))
    ):
        eligible = filter_grasp_candidates_by_reachability_fn(
            eligible, strict=True
        )
    planning_stats["cheap_eligible_count"] = len(eligible)
    preflight_budget = max(
        int(getattr(args, "grasp_preflight_max_candidates", 12)), 1
    )
    budgeted_eligible = list(eligible[:preflight_budget])
    for row in eligible[preflight_budget:]:
        row["workflow_preflight_ok"] = False
        row["workflow_preflight_reason"] = (
            "not preflighted due to planning budget"
        )
    eligible = budgeted_eligible
    eligible_ids = {id(row) for row in eligible}
    preflighted_ids = set()
    valid_plan_ids = set()
    candidate_started = {}
    target_valid_plans = max(
        int(getattr(args, "grasp_target_valid_plans", 3)), 1
    )
    print(
        "[grasp][workflow] complete route preflight before selection: "
        f"eligible={planning_stats['cheap_eligible_count']} "
        f"budget={len(eligible)} target_valid={target_valid_plans}"
    )

    def finish_candidate(row):
        row_id = id(row)
        started = candidate_started.pop(row_id, None)
        if started is None:
            return
        elapsed = time.perf_counter() - started
        key = str(int(row.get("idx", len(preflighted_ids))))
        planning_stats["candidate_planning_seconds"][key] = (
            float(planning_stats["candidate_planning_seconds"].get(key, 0.0))
            + float(elapsed)
        )
        print(
            f"[grasp][planning-stats][candidate] idx={key} "
            f"seconds={elapsed:.3f} ok="
            f"{bool(row.get('workflow_preflight_ok', False))}"
        )

    def run_route_pass(route_mode):
        route_passed = 0
        for row in rows:
            if id(row) not in eligible_ids or bool(
                row.get("workflow_preflight_ok", False)
            ):
                continue
            if len(valid_plan_ids) >= target_valid_plans:
                break
            row_id = id(row)
            candidate_started.setdefault(row_id, time.perf_counter())
            if route_mode == "direct" and row_id not in preflighted_ids:
                preflighted_ids.add(row_id)
                planning_stats["full_preflight_candidate_count"] = len(
                    preflighted_ids
                )
            if (
                route_mode == "neutral"
                and not set(
                    row.get(
                        "_workflow_preflight_direct_failure_stages", ()
                    )
                ).intersection({"pregrasp", "unknown"})
            ):
                # A neutral waypoint changes only observation->pregrasp.  It
                # cannot rescue grasp, lift, payload, or transport failures.
                continue
            grasp = row.get("grasp", None)
            candidate_idx = int(row.get("idx", 0))
            reason_key = f"workflow_preflight_{route_mode}_reason"
            if grasp is None:
                row[reason_key] = "candidate has no grasp pose"
                row["workflow_preflight_reason"] = row[reason_key]
                finish_candidate(row)
                continue

            # Freeze a candidate-specific *post-close collision envelope*.
            # It deliberately does not alter the hardware opening command,
            # candidate aperture validation, or GraspNet pose/width itself.
            try:
                solved_width = grasp_width_m_fn(row)
            except (AttributeError, TypeError, ValueError):
                # Lightweight planner tests and legacy injected candidates may
                # omit GraspNet metadata.  They use the explicit empty-close
                # fallback without weakening real candidate handling.
                solved_width = None
            empty_width = max(
                float(
                    getattr(args, "motion_closed_gripper_empty_width", 0.000)
                ),
                0.0,
            )
            width_padding = max(
                float(
                    getattr(args, "motion_closed_gripper_width_padding", 0.010)
                ),
                0.0,
            )
            closed_collision_width = (
                empty_width
                if solved_width is None
                else max(
                    empty_width,
                    max(float(solved_width), 0.0) + width_padding,
                )
            )
            row["post_close_collision_width"] = float(
                closed_collision_width
            )
            print(
                f"[grasp][closed-model] candidate={candidate_idx} "
                f"graspnet_width="
                f"{('unknown' if solved_width is None else f'{float(solved_width) * 1000.0:.1f}mm')} "
                f"post_close_padding={width_padding * 1000.0:.1f}mm "
                f"collision_gap={closed_collision_width * 1000.0:.1f}mm "
                "hardware_command_unchanged=true"
            )

            raw_width_signs, width_signs, orientation_angles = (
                candidate_preflight_width_signs_fn(
                    row,
                    grasp,
                    T_base_camera,
                    current_tcp_pose,
                    args,
                )
            )
            primary_sign = float(raw_width_signs[0])
            row["jaw_primary_width_sign"] = primary_sign
            row["jaw_fallback_width_sign"] = (
                None
                if len(raw_width_signs) < 2
                else float(raw_width_signs[1])
            )
            if len(width_signs) == 0:
                row[reason_key] = (
                    "no endpoint- and aperture-validated jaw-axis orientation "
                    "remains"
                )
                row["workflow_preflight_reason"] = row[reason_key]
                finish_candidate(row)
                continue
            if abs(float(width_signs[0]) - primary_sign) >= 0.5:
                print(
                    f"[grasp][workflow][jaw-fallback] candidate={candidate_idx} "
                    f"primary width_sign={int(primary_sign):+d} was removed by "
                    "aperture/endpoint checks; start with the 180deg fallback "
                    f"width_sign={int(width_signs[0]):+d}"
                )

            failures = []
            for width_sign in width_signs:
                is_fallback = abs(float(width_sign) - primary_sign) >= 0.5
                if is_fallback and failures:
                    print(
                        f"[grasp][workflow][jaw-fallback] "
                        f"candidate={candidate_idx} primary "
                        f"width_sign={int(primary_sign):+d} failed the complete "
                        "route; trying 180deg fallback "
                        f"width_sign={int(width_sign):+d}"
                    )
                width_shift = grasp_candidate_width_shift_fn(row, width_sign)
                roll_offsets = _ordered_grasp_roll_offsets(
                    grasp,
                    T_base_camera,
                    current_tcp_pose,
                    width_sign,
                    width_shift,
                    arm,
                    args,
                    candidate_idx,
                    grasp_to_tcp_poses_fn,
                )
                planning_stage_cache = {}
                fast_planning_stage_cache = {}
                lift_endpoint_cache = {}
                candidate_planning_metrics = {}

                def precheck_lift_endpoint(pose_variant):
                    if not bool(
                        getattr(args, "grasp_lift_endpoint_precheck", False)
                    ):
                        return
                    lift_pose = np.asarray(
                        pose_variant["lift_tcp"], dtype=float
                    ).reshape(6)
                    key = tuple(np.round(lift_pose, 9).tolist())
                    cached = lift_endpoint_cache.get(key)
                    if cached is not None:
                        if cached[0]:
                            return
                        raise PlanNotExecutable(cached[1])
                    label = (
                        f"{plan_label}_candidate{candidate_idx}_lift_"
                        "endpoint_global_precheck"
                    )
                    try:
                        try:
                            flange_pose = np.asarray(
                                arm.get_tcp2flange_pose(lift_pose),
                                dtype=float,
                            ).reshape(6)
                            precheck_joints = np.asarray(
                                arm.get_joint_angles(), dtype=float
                            ).reshape(7)
                        except (TypeError, ValueError):
                            # Compatibility/test doubles may expose these
                            # methods without numerical feedback.  The normal
                            # complete planner remains authoritative.
                            print(
                                f"[grasp][lift-precheck][warn] {label} "
                                "numerical arm conversion unavailable; defer "
                                "to complete route planning"
                            )
                            return
                        # This is an endpoint-existence test, not a route from
                        # the observation joints.  Disable the per-edge delta
                        # gate here; the later beam still checks the actual
                        # grasp-to-lift edge against every motion constraint.
                        endpoint_args = copy(args)
                        endpoint_args.ik_max_joint_delta = None
                        minimum_margin = max(
                            float(
                                getattr(
                                    args,
                                    "grasp_lift_min_joint_margin_deg",
                                    0.0,
                                )
                            ),
                            0.0,
                        )
                        records = ik_solver._ranked_ik_solution_records_for_flange_pose(
                            arm,
                            flange_pose,
                            endpoint_args,
                            label,
                            current_joints_override=precheck_joints,
                            quiet=True,
                            top_k=max(
                                int(getattr(args, "ik_solution_topk", 6)),
                                int(
                                    getattr(
                                        args, "ik_all_joint_seed_count", 24
                                    )
                                )
                                + 16,
                            ),
                            early_accept_margin_rad=np.deg2rad(
                                minimum_margin
                            ),
                        )
                        viable = [
                            item
                            for item in records
                            if float(np.rad2deg(item["limit_margin"]))
                            >= minimum_margin
                        ]
                        if not viable:
                            best_margin = max(
                                float(np.rad2deg(item["limit_margin"]))
                                for item in records
                            )
                            raise PlanNotExecutable(
                                f"{label} lift endpoint has IK only within "
                                f"{best_margin:.2f}deg of a hard joint limit; "
                                f"required >= {minimum_margin:.2f}deg"
                            )
                    except (PlanNotExecutable, UnsafeMotionTarget) as exc:
                        reason = str(exc)
                        lift_endpoint_cache[key] = (False, reason)
                        raise
                    lift_endpoint_cache[key] = (True, len(viable))
                    seed_trials = max(
                        int(item.get("seed_index", 0)) for item in viable
                    ) + 1
                    print(
                        f"[grasp][lift-precheck] candidate={candidate_idx} "
                        f"width_sign={int(width_sign):+d} "
                        f"height={float(pose_variant.get('lift_height', lift_pose[2] - base_grasp_tcp[2])):.3f}m "
                        f"viable_ik={len(viable)} seed_trials={seed_trials} "
                        "early_stop=true cached=true"
                    )

                def plan_pose_attempt(
                    pose_variant, transport_route_mode, attempt_index
                ):
                    pre_tcp = np.asarray(
                        pose_variant["pre_tcp"], dtype=float
                    ).reshape(6)
                    grasp_tcp = np.asarray(
                        pose_variant["grasp_tcp"], dtype=float
                    ).reshape(6)
                    lift_tcp = np.asarray(
                        pose_variant["lift_tcp"], dtype=float
                    ).reshape(6)
                    if (
                        str(getattr(args, "grasp_generator", "graspnet")) == "graspgenx"
                        or getattr(args, "grasp_table_reference_z", None) is not None
                    ):
                        from .graspgenx_integration import selected_mesh_table_clearance, grasp_mesh_table_margin
                        mesh_args = copy(args)
                        mesh_args.graspgenx_table_margin = grasp_mesh_table_margin(args)
                        try:
                            clearance = selected_mesh_table_clearance(pre_tcp, grasp_tcp, mesh_args)
                        except (RuntimeError, ValueError, OSError) as exc:
                            raise UnsafeMotionTarget(
                                f"candidate{candidate_idx}_grasp full open mesh rejected: {exc}"
                            ) from exc
                        pose_variant["mesh_table_check"] = dict(
                            pre_tcp=pre_tcp.copy(), grasp_tcp=grasp_tcp.copy(),
                            required_margin=mesh_args.graspgenx_table_margin,
                            clearance=float(clearance),
                        )
                    validation_only = bool(
                        pose_variant.get("validation_only", False)
                    )
                    candidate_place_tcp = (
                        None
                        if validation_only
                        else _place_tcp_from_lift(lift_tcp, place_xyz, args)
                    )
                    candidate_place_tcp = _apply_transport_pose_fallback(
                        candidate_place_tcp, pose_variant
                    )
                    for stage_label, tcp_pose in (
                        ("pregrasp", pre_tcp),
                        ("grasp", grasp_tcp),
                        ("lift", lift_tcp),
                        ("place", candidate_place_tcp),
                    ):
                        if tcp_pose is not None:
                            assert_table_guard_target_tcp_fn(
                                tcp_pose,
                                args,
                                (
                                    f"{plan_label}_candidate{candidate_idx}_"
                                    f"{stage_label}"
                                ),
                            )
                    variant_plan_label = (
                        f"{plan_label}_candidate{candidate_idx}"
                        if attempt_index == 0
                        else (
                            f"{plan_label}_candidate{candidate_idx}"
                            f"_fallback{attempt_index}"
                        )
                    )

                    def run_planner(planning_args, stage_cache, label):
                        planning_args._planning_closed_gripper_width = float(
                            closed_collision_width
                        )
                        candidate_plan_kwargs = {
                            "motion_environment_cloud": (
                                motion_environment_cloud
                            ),
                            "pregrasp_route_mode": route_mode,
                            "transport_route_mode": transport_route_mode,
                            "planning_stage_cache": stage_cache,
                            "validation_only": validation_only,
                        }
                        if motion_environment_map is not None:
                            candidate_plan_kwargs[
                                "motion_environment_map"
                            ] = motion_environment_map
                        if plan_up_and_over_grasp_sequence_hook is None:
                            candidate_plan_kwargs["planning_metrics"] = (
                                candidate_planning_metrics
                            )
                        return plan_up_and_over_grasp_sequence_fn(
                            arm,
                            pre_tcp,
                            grasp_tcp,
                            lift_tcp,
                            candidate_place_tcp,
                            target_cloud,
                            motion_scene_cloud,
                            T_base_camera,
                            planning_args,
                            label,
                            **candidate_plan_kwargs,
                        )

                    progressive = bool(
                        getattr(args, "grasp_progressive_beam", False)
                    ) and bool(
                        getattr(args, "grasp_stage_beam_search", False)
                    )
                    if progressive:
                        fast_args = copy(args)
                        fast_args.grasp_stage_beam_width = max(
                            min(
                                int(
                                    getattr(
                                        args,
                                        "grasp_fast_stage_beam_width",
                                        2,
                                    )
                                ),
                                int(
                                    getattr(
                                        args, "grasp_stage_beam_width", 6
                                    )
                                ),
                            ),
                            1,
                        )
                        fast_args.ik_solution_topk = max(
                            min(
                                int(
                                    getattr(
                                        args,
                                        "grasp_fast_ik_solution_topk",
                                        2,
                                    )
                                ),
                                int(getattr(args, "ik_solution_topk", 6)),
                            ),
                            1,
                        )
                        fast_args._planning_ik_seed_limit = max(
                            int(
                                getattr(
                                    args,
                                    "grasp_fast_ik_seed_limit",
                                    5,
                                )
                            ),
                            1,
                        )
                        fast_args._planning_ik_mode = "fast"
                        try:
                            plan = run_planner(
                                fast_args,
                                fast_planning_stage_cache,
                                f"{variant_plan_label}_fast",
                            )
                            print(
                                f"[grasp][workflow][progressive] candidate="
                                f"{candidate_idx} width_sign="
                                f"{int(width_sign):+d} fast complete route "
                                f"passed with beam="
                                f"{fast_args.grasp_stage_beam_width} topk="
                                f"{fast_args.ik_solution_topk}; skip full "
                                "beam expansion"
                            )
                            return plan
                        except (PlanNotExecutable, UnsafeMotionTarget) as exc:
                            if _lift_failure_is_pose_invariant(exc):
                                print(
                                    f"[grasp][workflow][progressive] candidate="
                                    f"{candidate_idx} width_sign="
                                    f"{int(width_sign):+d} fast route found "
                                    "an invariant lift-start failure shared "
                                    "by all retained IK branches; skip full "
                                    "beam expansion"
                                )
                                raise
                            failure_stage = _grasp_plan_failure_stage(exc)
                            if not bool(
                                getattr(
                                    args,
                                    "grasp_progressive_full_fallback",
                                    True,
                                )
                            ):
                                print(
                                    f"[grasp][workflow][progressive] candidate="
                                    f"{candidate_idx} width_sign="
                                    f"{int(width_sign):+d} local fast route "
                                    f"failed at {failure_stage}; full Top-K/"
                                    "global expansion is disabled"
                                )
                                raise
                            if failure_stage in {
                                "support_release",
                                "lift",
                                "transport",
                            }:
                                # The fast attempt has already produced and
                                # collision-checked all upstream stage options.
                                # Promote those frozen options into the full
                                # cache so Top-K expansion starts at the failed
                                # stage instead of recomputing pregrasp/grasp.
                                planning_stage_cache.update(
                                    fast_planning_stage_cache
                                )
                                print(
                                    f"[grasp][workflow][progressive] reuse "
                                    f"{len(fast_planning_stage_cache)} "
                                    "validated upstream stage-cache entries; "
                                    f"expand only from {failure_stage}"
                                )
                            print(
                                f"[grasp][workflow][progressive] candidate="
                                f"{candidate_idx} width_sign="
                                f"{int(width_sign):+d} fast route failed: "
                                f"{type(exc).__name__}: {exc}; expand to "
                                f"beam={int(getattr(args, 'grasp_stage_beam_width', 6))} "
                                f"topk={int(getattr(args, 'ik_solution_topk', 6))}"
                            )
                    full_args = copy(args)
                    full_args._planning_ik_seed_limit = None
                    full_args._planning_ik_mode = "full"
                    return run_planner(
                        full_args,
                        planning_stage_cache,
                        variant_plan_label,
                    )

                roll_choices = []
                width_failure_records = []
                for roll_offset in roll_offsets:
                    base_pre_tcp, base_grasp_tcp, base_lift_tcp = (
                        grasp_to_tcp_poses_fn(
                            grasp,
                            T_base_camera,
                            args,
                            current_tcp_pose,
                            width_sign=width_sign,
                            width_shift=width_shift,
                            roll_offset_deg=roll_offset,
                        )
                    )
                    planning_stage_cache.clear()
                    fast_planning_stage_cache.clear()
                    lift_endpoint_cache.clear()
                    candidate_planning_metrics.clear()

                    selected_for_roll, roll_failures = (
                        _plan_grasp_pose_fallbacks(
                            base_pre_tcp,
                            base_grasp_tcp,
                            base_lift_tcp,
                            args,
                            plan_pose_attempt,
                            lift_endpoint_precheck=(
                                None
                                if (
                                    bool(
                                        getattr(
                                            args,
                                            "post_grasp_free_route",
                                            False,
                                        )
                                    )
                                    or bool(
                                        getattr(
                                            args,
                                            "grasp_lift_feasible_region",
                                            False,
                                        )
                                    )
                                )
                                else precheck_lift_endpoint
                            ),
                        )
                    )
                    for record in roll_failures:
                        record["roll_offset_deg"] = float(roll_offset)
                    width_failure_records.extend(roll_failures)

                    if selected_for_roll is None:
                        continue

                    route_cost = _grasp_plan_joint_route_cost(
                        selected_for_roll["plan"]
                    )
                    roll_choices.append(
                        {
                            "selected": selected_for_roll,
                            "roll_offset_deg": float(roll_offset),
                            "route_cost": route_cost,
                            "planning_metrics": dict(
                                candidate_planning_metrics
                            ),
                        }
                    )

                if roll_choices:
                    best_roll = min(
                        roll_choices,
                        key=lambda item: item["route_cost"],
                    )
                    selected = best_roll["selected"]
                    roll_offset = best_roll["roll_offset_deg"]
                    candidate_planning_metrics = best_roll[
                        "planning_metrics"
                    ]
                    route_cost = best_roll["route_cost"]
                    print(
                        f"[grasp][roll-select] candidate={candidate_idx} "
                        f"width_sign={int(width_sign):+d} "
                        f"roll={roll_offset:+.1f}deg "
                        f"route_l2={route_cost[0]:.1f}deg "
                        f"max_step={route_cost[1]:.1f}deg "
                        f"wrist_sweep={route_cost[2]:.1f}deg"
                    )
                else:
                    selected = None

                for record in width_failure_records:
                    pose_variant = record["variant"]
                    failure = (
                        f"roll={float(record.get('roll_offset_deg', 0.0)):+.1f}deg/"
                        f"stage={record['stage']}/"
                        f"transport={record['transport_route_mode']}/"
                        f"pre={float(pose_variant['pregrasp_offset']) * 1000.0:.0f}mm/"
                        f"lift_h={float(pose_variant.get('lift_height', base_lift_tcp[2] - base_grasp_tcp[2])) * 1000.0:.0f}mm/"
                        f"lift={float(pose_variant['lift_relax_deg']):+.0f}deg:"
                        f"{type(record['exception']).__name__}:"
                        f"{record['exception']}"
                    )
                    print(
                        f"[grasp][workflow][reject] route={route_mode} "
                        f"candidate={candidate_idx} "
                        f"width_sign={int(width_sign):+d} {failure}"
                    )
                    stage_name = str(record["stage"])
                    stage_counts = planning_stats["failure_stages"]
                    stage_counts[stage_name] = int(
                        stage_counts.get(stage_name, 0)
                    ) + 1
                    if "micro_lift" in str(record["exception"]).lower():
                        row["micro_lift_ik_ok"] = False

                failure_stage_key = (
                    f"_workflow_preflight_{route_mode}_failure_stages"
                )
                prior_stages = set(row.get(failure_stage_key, ()))
                prior_stages.update(
                    str(record["stage"])
                    for record in width_failure_records
                )
                row[failure_stage_key] = tuple(sorted(prior_stages))

                if selected is None:
                    stage_counts = {}
                    for record in width_failure_records:
                        stage = str(record["stage"])
                        stage_counts[stage] = stage_counts.get(stage, 0) + 1
                    samples = []
                    for record in width_failure_records[:2]:
                        samples.append(
                            f"{record['stage']}:{type(record['exception']).__name__}:"
                            f"{record['exception']}"
                        )
                    sample = " | ".join(samples)
                    failures.append(
                        f"width_sign={int(width_sign):+d}:stage-aware "
                        f"attempts={len(width_failure_records)} "
                        f"failures={stage_counts}: {sample}"
                    )
                    continue

                candidate_plan = selected["plan"]
                selected_variant = selected["variant"]
                selected_transport_route = selected["transport_route_mode"]
                pre_tcp = np.asarray(
                    selected_variant["pre_tcp"], dtype=float
                ).reshape(6)
                grasp_tcp = np.asarray(
                    selected_variant["grasp_tcp"], dtype=float
                ).reshape(6)
                lift_tcp = np.asarray(
                    selected_variant["lift_tcp"], dtype=float
                ).reshape(6)
                validation_only = bool(
                    selected_variant.get("validation_only", False)
                )
                candidate_place_tcp = (
                    None
                    if validation_only
                    else _place_tcp_from_lift(lift_tcp, place_xyz, args)
                )
                candidate_place_tcp = _apply_transport_pose_fallback(
                    candidate_place_tcp, selected_variant
                )
                if int(selected["attempt_index"]) > 0:
                    print(
                        f"[grasp][workflow][stage-fallback] "
                        f"candidate={candidate_idx} "
                        f"width_sign={int(width_sign):+d} selected "
                        f"pre={float(selected_variant['pregrasp_offset']) * 1000.0:.0f}mm "
                        f"lift_height={float(selected_variant.get('lift_height', lift_tcp[2] - grasp_tcp[2])) * 1000.0:.0f}mm "
                        f"lift={float(selected_variant['lift_relax_deg']):+.0f}deg "
                        f"transport={selected_transport_route} after "
                        f"{len(width_failure_records)} relevant failure(s)"
                    )

                row["workflow_preflight_ok"] = True
                valid_plan_ids.add(id(row))
                row["workflow_preflight_reason"] = "ok"
                row["workflow_preflight_route_kind"] = route_mode
                row["workflow_preflight_roll_offset_deg"] = float(
                    roll_offset
                )
                row["workflow_preflight_route_l2_deg"] = float(
                    route_cost[0]
                )
                row["workflow_preflight_route_max_step_deg"] = float(
                    route_cost[1]
                )
                row["workflow_preflight_wrist_sweep_deg"] = float(
                    route_cost[2]
                )
                row["workflow_preflight_width_sign"] = float(width_sign)
                row["workflow_preflight_width_shift"] = float(width_shift)
                row["workflow_preflight_pregrasp_offset"] = float(
                    selected_variant["pregrasp_offset"]
                )
                row["workflow_preflight_lift_relax_deg"] = float(
                    selected_variant["lift_relax_deg"]
                )
                row["workflow_preflight_lift_height"] = float(
                    selected_variant.get(
                        "lift_height", lift_tcp[2] - grasp_tcp[2]
                    )
                )
                row["workflow_preflight_lift_relax_axis"] = str(
                    selected_variant["lift_relax_axis"]
                )
                row["workflow_preflight_transport_route"] = str(
                    selected_transport_route
                )
                row["workflow_preflight_validation_only"] = validation_only
                row["workflow_preflight_jaw_role"] = (
                    "fallback_180" if is_fallback else "primary"
                )
                row["tcp_width_sign"] = float(width_sign)
                row["jaw_selected_as_fallback"] = bool(is_fallback)
                row["jaw_selected_role"] = (
                    "fallback_180" if is_fallback else "primary"
                )
                row["jaw_selected_orientation_delta_deg"] = float(
                    np.rad2deg(orientation_angles[float(width_sign)])
                )
                row["pre_tcp_preview"] = np.asarray(pre_tcp, dtype=float)
                row["grasp_tcp_preview"] = np.asarray(grasp_tcp, dtype=float)
                row["lift_tcp_preview"] = np.asarray(lift_tcp, dtype=float)
                row["_workflow_motion_plan"] = candidate_plan
                if "mesh_table_check" in selected_variant:
                    row["_workflow_mesh_table_check"] = selected_variant["mesh_table_check"]
                    row["workflow_preflight_mesh_table_clearance_m"] = float(
                        selected_variant["mesh_table_check"]["clearance"]
                    )
                row["_workflow_pre_tcp"] = np.asarray(pre_tcp, dtype=float)
                row["_workflow_grasp_tcp"] = np.asarray(grasp_tcp, dtype=float)
                row["_workflow_lift_tcp"] = np.asarray(lift_tcp, dtype=float)
                row["_workflow_place_tcp"] = (
                    None
                    if candidate_place_tcp is None
                    else np.asarray(candidate_place_tcp, dtype=float)
                )
                row.update(candidate_planning_metrics)
                row.setdefault("micro_lift_ik_ok", True)
                route_passed += 1
                print(
                    f"[grasp][workflow][keep] route={route_mode} "
                    f"candidate={candidate_idx} "
                    f"width_sign={int(width_sign):+d} "
                    f"roll={roll_offset:+.1f}deg "
                    f"role={'fallback_180' if is_fallback else 'primary'} "
                    f"pregrasp={float(selected_variant['pregrasp_offset']) * 1000.0:.0f}mm "
                    f"lift_height={float(selected_variant.get('lift_height', lift_tcp[2] - grasp_tcp[2])) * 1000.0:.0f}mm "
                    f"lift_relax={float(selected_variant['lift_relax_deg']):+.0f}deg "
                    f"transport={selected_transport_route} "
                    "complete route validated"
                )
                break

            if not bool(row["workflow_preflight_ok"]):
                row[reason_key] = " | ".join(failures)
                if route_mode == "neutral":
                    direct_reason = str(
                        row.get(
                            "workflow_preflight_direct_reason",
                            "direct route failed",
                        )
                    )
                    row["workflow_preflight_reason"] = (
                        f"direct:{direct_reason} | neutral:{row[reason_key]}"
                    )
                else:
                    row["workflow_preflight_reason"] = row[reason_key]
            finish_candidate(row)
        return route_passed

    passed = run_route_pass("direct")
    used_neutral_pass = False
    if (
        passed == 0
        and len(eligible) > 0
        and bool(getattr(args, "motion_neutral_pregrasp_fallback", False))
        and not bool(getattr(args, "motion_up_and_over", False))
        and any(
            bool(
                set(
                    row.get(
                        "_workflow_preflight_direct_failure_stages", ()
                    )
                ).intersection({"pregrasp", "unknown"})
            )
            for row in eligible
        )
    ):
        used_neutral_pass = True
        print(
            "[grasp][workflow][neutral] all candidate/jaw direct beam routes "
            "failed; retrying the same eligible set through one neutral joint "
            "configuration"
        )
        passed = run_route_pass("neutral")

    print(
        f"[grasp][workflow] complete route filter kept "
        f"{passed}/{len(eligible)} candidates "
        f"route_pass={'neutral' if used_neutral_pass else 'direct'}"
    )
    for row in eligible:
        if id(row) not in preflighted_ids:
            row["workflow_preflight_ok"] = False
            row["workflow_preflight_reason"] = (
                "not preflighted due to planning budget"
            )
    planning_stats["valid_plan_count"] = len(valid_plan_ids)
    planning_stats["total_planning_seconds"] = (
        time.perf_counter() - planning_started
    )
    stage_summary = ",".join(
        f"{name}={count}"
        for name, count in planning_stats["failure_stages"].items()
    )
    print(
        "[grasp][planning-stats] "
        f"input={planning_stats['input_candidate_count']} "
        f"cheap_eligible={planning_stats['cheap_eligible_count']} "
        f"full_preflight={planning_stats['full_preflight_candidate_count']} "
        f"valid={planning_stats['valid_plan_count']} "
        f"ik_calls={planning_stats['ik_calls']} "
        f"ik_seed_trials={planning_stats['ik_seed_trials']} "
        f"fast_ik={planning_stats['fast_ik_calls']} "
        f"full_ik={planning_stats['full_ik_calls']} "
        f"collision_configs="
        f"{planning_stats['collision_configuration_evaluations']} "
        f"failures=[{stage_summary}] "
        f"total_seconds={planning_stats['total_planning_seconds']:.3f}"
    )
    return rows
