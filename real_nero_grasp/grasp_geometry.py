"""Pure grasp-frame to Nero TCP pose geometry.

This module contains the orientation/pose conversion helpers used by the real
Nero grasp workflow.  It intentionally has no robot-I/O, collision-planning,
or grasp-selection dependencies so it can be imported by both planning and
physical-aperture modules without creating an import cycle.
"""

import numpy as np
from scipy.spatial.transform import Rotation as R

from tools.real_nero_point_test import make_T
from .transforms import _as_pose_list


__all__ = [
    "_horizontal_unit",
    "_axis_vector",
    "_axis_index_sign",
    "_rotation_from_axis_constraints",
    "graspnet_rotation_to_tcp_rotation",
    "grasp_tcp_axis_diagnostics",
    "topdown_rotation_from_hint",
    "_approach_axis_local",
    "grasp_to_tcp_poses",
    "grasp_planning_pose_variants",
]

def _horizontal_unit(vec, fallback):
    out = np.array([float(vec[0]), float(vec[1]), 0.0], dtype=float)
    if np.linalg.norm(out) < 1e-6:
        out = np.asarray(fallback, dtype=float)
    return out / (np.linalg.norm(out) + 1e-9)


def _axis_vector(axis_name):
    axis = str(axis_name).strip().lower()
    sign = -1.0 if axis.startswith("-") else 1.0
    axis = axis[1:] if axis.startswith("-") else axis
    if axis == "x":
        return np.array([sign, 0.0, 0.0], dtype=float)
    if axis == "y":
        return np.array([0.0, sign, 0.0], dtype=float)
    if axis == "z":
        return np.array([0.0, 0.0, sign], dtype=float)
    raise ValueError(f"Unsupported tool axis {axis_name!r}")


def _axis_index_sign(axis_name):
    axis = str(axis_name).strip().lower()
    sign = -1.0 if axis.startswith("-") else 1.0
    axis = axis[1:] if axis.startswith("-") else axis
    if axis == "x":
        return 0, sign
    if axis == "y":
        return 1, sign
    if axis == "z":
        return 2, sign
    raise ValueError(f"Unsupported axis {axis_name!r}")


def _rotation_from_axis_constraints(axis_a, dir_a, axis_b, dir_b):
    idx_a, sign_a = _axis_index_sign(axis_a)
    idx_b, sign_b = _axis_index_sign(axis_b)
    if idx_a == idx_b:
        raise ValueError(
            f"Axis constraints must use different TCP axes: {axis_a}, {axis_b}"
        )
    dir_a = np.asarray(dir_a, dtype=float).reshape(3)
    dir_b = np.asarray(dir_b, dtype=float).reshape(3)
    dir_a = dir_a / max(float(np.linalg.norm(dir_a)), 1e-9)
    dir_b = dir_b - dir_a * float(np.dot(dir_a, dir_b))
    dir_b = dir_b / max(float(np.linalg.norm(dir_b)), 1e-9)

    cols = [None, None, None]
    cols[idx_a] = sign_a * dir_a
    cols[idx_b] = sign_b * dir_b
    if cols[0] is None:
        cols[0] = np.cross(cols[1], cols[2])
    elif cols[1] is None:
        cols[1] = np.cross(cols[2], cols[0])
    elif cols[2] is None:
        cols[2] = np.cross(cols[0], cols[1])
    R_out = np.column_stack(cols)
    U, _, Vt = np.linalg.svd(R_out)
    R_out = U @ Vt
    if np.linalg.det(R_out) < 0:
        U[:, -1] *= -1
        R_out = U @ Vt
    return R_out


def graspnet_rotation_to_tcp_rotation(R_base_grasp, args, width_sign=1.0, roll_offset_deg=0.0):
    R_base_grasp = np.asarray(R_base_grasp, dtype=float).reshape(3, 3)
    grasp_approach = R_base_grasp @ _axis_vector(args.graspnet_approach_axis)
    grasp_width = (
        R_base_grasp
        @ _axis_vector(args.graspnet_width_axis)
        * float(width_sign)
    )
    roll_deg = float(roll_offset_deg)
    if abs(roll_deg) > 1e-9:
        # Rotate the (arbitrary for a rotationally-symmetric target) width axis
        # around the approach axis.  This sweeps the grasp roll so the planner
        # can choose the flange orientation that minimizes joint swing.
        axis = grasp_approach / (np.linalg.norm(grasp_approach) + 1e-9)
        R_roll = R.from_rotvec(axis * np.deg2rad(roll_deg)).as_matrix()
        grasp_width = R_roll @ grasp_width
    return _rotation_from_axis_constraints(
        args.tool_approach_axis,
        grasp_approach,
        args.tcp_gripper_width_axis,
        grasp_width,
    )


def grasp_tcp_axis_diagnostics(
    grasp,
    T_base_camera,
    args,
    tcp_pose,
    width_sign=1.0,
):
    R_cg = np.asarray(grasp.rotation_matrix, dtype=float).reshape(3, 3)
    R_bg = (
        np.asarray(T_base_camera, dtype=float).reshape(4, 4)[:3, :3]
        @ R_cg
    )
    R_bt = R.from_euler(
        "xyz", np.asarray(tcp_pose, dtype=float).reshape(6)[3:]
    ).as_matrix()
    grasp_approach = R_bg @ _axis_vector(args.graspnet_approach_axis)
    grasp_width = (
        R_bg @ _axis_vector(args.graspnet_width_axis) * float(width_sign)
    )
    tcp_approach = R_bt @ _axis_vector(args.tool_approach_axis)
    tcp_width = R_bt @ _axis_vector(args.tcp_gripper_width_axis)
    approach_angle = np.rad2deg(
        np.arccos(
            np.clip(float(np.dot(grasp_approach, tcp_approach)), -1.0, 1.0)
        )
    )
    width_angle = np.rad2deg(
        np.arccos(np.clip(float(np.dot(grasp_width, tcp_width)), -1.0, 1.0))
    )
    return {
        "grasp_approach_base": grasp_approach,
        "grasp_width_base": grasp_width,
        "tcp_approach_base": tcp_approach,
        "tcp_width_base": tcp_width,
        "approach_angle_deg": float(approach_angle),
        "width_angle_deg": float(width_angle),
    }


def topdown_rotation_from_hint(R_hint, approach_axis, tilt_deg=5.0):
    axis = str(approach_axis).strip().lower()
    sign = -1.0 if axis.startswith("-") else 1.0
    axis = axis[1:] if axis.startswith("-") else axis
    down = np.array([0.0, 0.0, -1.0], dtype=float)
    tilt_rad = np.deg2rad(float(tilt_deg))
    tilt_hint = _horizontal_unit(
        R_hint[:, 1] if axis != "y" else R_hint[:, 0],
        [0.0, -1.0, 0.0],
    )
    approach_dir = down
    if abs(float(tilt_deg)) > 1e-9:
        approach_dir = (
            np.cos(tilt_rad) * down + np.sin(tilt_rad) * tilt_hint
        )
        approach_dir = approach_dir / (np.linalg.norm(approach_dir) + 1e-9)

    if axis == "x":
        x_axis = sign * approach_dir
        y_axis = _horizontal_unit(R_hint[:, 1], [0.0, -1.0, 0.0])
        z_axis = np.cross(x_axis, y_axis)
    elif axis == "y":
        y_axis = sign * approach_dir
        x_axis = _horizontal_unit(R_hint[:, 0], [1.0, 0.0, 0.0])
        z_axis = np.cross(x_axis, y_axis)
    elif axis == "z":
        z_axis = sign * approach_dir
        y_axis = _horizontal_unit(R_hint[:, 1], [0.0, -1.0, 0.0])
        x_axis = np.cross(y_axis, z_axis)
    else:
        raise ValueError(f"Unsupported --tool-approach-axis {approach_axis!r}")

    x_axis = x_axis / (np.linalg.norm(x_axis) + 1e-9)
    y_axis = y_axis / (np.linalg.norm(y_axis) + 1e-9)
    z_axis = z_axis / (np.linalg.norm(z_axis) + 1e-9)
    R_td = np.column_stack([x_axis, y_axis, z_axis])
    U, _, Vt = np.linalg.svd(R_td)
    R_td = U @ Vt
    if np.linalg.det(R_td) < 0:
        U[:, -1] *= -1
        R_td = U @ Vt
    return R_td


def _approach_axis_local(args):
    return _axis_vector(args.tool_approach_axis)


def grasp_to_tcp_poses(
    grasp,
    T_base_camera,
    args,
    current_tcp_pose=None,
    width_sign=1.0,
    width_shift=0.0,
    roll_offset_deg=0.0,
):
    R_cg = np.asarray(grasp.rotation_matrix, dtype=float).reshape(3, 3)
    t_cg = np.asarray(grasp.translation, dtype=float).reshape(3)
    U, _, Vt = np.linalg.svd(R_cg)
    R_cg = U @ Vt
    if np.linalg.det(R_cg) < 0:
        U[:, -1] *= -1
        R_cg = U @ Vt

    T_camera_grasp = np.eye(4)
    T_camera_grasp[:3, :3] = R_cg
    T_camera_grasp[:3, 3] = t_cg
    T_base_grasp = T_base_camera @ T_camera_grasp

    genx_tcp = getattr(grasp, "nero_tcp_pose_camera", None)
    if genx_tcp is not None:
        # GraspGenX predicts a physical Nero asset pose. Its root is mapped
        # through link7 and the configured TCP offset by the adapter.
        T_base_grasp = T_base_camera @ np.asarray(genx_tcp, dtype=float).reshape(4, 4)
    elif args.topdown_grasp:
        T_base_grasp[:3, :3] = topdown_rotation_from_hint(
            T_base_grasp[:3, :3],
            args.tool_approach_axis,
            args.approach_tilt_deg,
        )
    else:
        T_base_grasp[:3, :3] = graspnet_rotation_to_tcp_rotation(
            T_base_grasp[:3, :3],
            args,
            width_sign=width_sign,
            roll_offset_deg=roll_offset_deg,
        )

    approach_axis = _approach_axis_local(args)
    depth_advance = 0.0
    if genx_tcp is None and args.use_graspnet_depth_advance and hasattr(grasp, "depth"):
        try:
            depth_advance = float(grasp.depth)
        except Exception:
            depth_advance = 0.0
    native_advance = getattr(args, "graspgenx_contact_advance", 0.0)
    if native_advance == "auto":
        native_advance = getattr(grasp, "nero_tcp_contact_advance", 0.0)
    total_advance = (float(args.grasp_advance) + depth_advance
                     if genx_tcp is None else float(native_advance))
    if hasattr(grasp, "nero_execution_advance_m"):
        total_advance = float(grasp.nero_execution_advance_m)
    if abs(total_advance) > 1e-9:
        T_base_grasp = T_base_grasp @ make_T(
            np.eye(3), approach_axis * total_advance
        )

    if (
        args.min_grasp_z is not None
        and T_base_grasp[2, 3] < float(args.min_grasp_z)
    ):
        raw_z = float(T_base_grasp[2, 3])
        clamped_z = float(args.min_grasp_z)
        print(
            f"[grasp][warn] --min-grasp-z raised grasp z from {raw_z:.4f}m "
            f"to {clamped_z:.4f}m by "
            f"{(clamped_z - raw_z) * 1000.0:.1f}mm"
        )
        T_base_grasp[2, 3] = clamped_z

    T_base_pre = T_base_grasp @ make_T(
        np.eye(3), -approach_axis * float(args.pregrasp_offset)
    )
    T_base_lift = T_base_grasp.copy()
    T_base_lift[:3, 3] = T_base_lift[:3, 3] + np.array(
        [0.0, 0.0, float(args.lift_height)]
    )
    pre_pose = _as_pose_list(T_base_pre)
    grasp_pose = _as_pose_list(T_base_grasp)
    lift_pose = _as_pose_list(T_base_lift)

    if args.fixed_rpy is not None:
        fixed_rpy = np.asarray(args.fixed_rpy, dtype=float).reshape(3)
        pre_pose[3:] = fixed_rpy
        grasp_pose[3:] = fixed_rpy
        lift_pose[3:] = fixed_rpy
    elif args.use_current_rpy:
        if current_tcp_pose is None:
            raise ValueError("--use-current-rpy requires current_tcp_pose")
        current_rpy = np.asarray(current_tcp_pose, dtype=float).reshape(6)[3:]
        pre_pose[3:] = current_rpy
        grasp_pose[3:] = current_rpy
        lift_pose[3:] = current_rpy

    width_shift = float(width_shift)
    if not np.isfinite(width_shift):
        raise ValueError("width_shift must be finite")
    if abs(width_shift) > 1e-12:
        command_rotation = R.from_euler(
            "xyz", np.asarray(grasp_pose[3:], dtype=float)
        ).as_matrix()
        width_axis_base = command_rotation @ _axis_vector(
            args.tcp_gripper_width_axis
        )
        width_axis_base /= np.linalg.norm(width_axis_base) + 1e-12
        correction = width_axis_base * width_shift
        pre_pose[:3] += correction
        grasp_pose[:3] += correction
        lift_pose[:3] += correction
        if (
            args.min_grasp_z is not None
            and float(grasp_pose[2]) < float(args.min_grasp_z)
        ):
            # Preserve the configured grasp floor after a tilted width-axis
            # correction.  Apply the same vertical adjustment to all three
            # poses so the approach geometry remains internally consistent;
            # the corrected target fit and complete route are checked again.
            floor_raise = float(args.min_grasp_z) - float(grasp_pose[2])
            pre_pose[2] += floor_raise
            grasp_pose[2] += floor_raise
            lift_pose[2] += floor_raise

    return pre_pose, grasp_pose, lift_pose


def _adaptive_pregrasp_offsets(args, measured_offset=None):
    """Return ordered planning-only pregrasp distances.

    The commanded grasp pose is unchanged.  Only the collision-checked pose
    from which the final approach starts is varied, with the configured value
    always tried first.
    """
    fallback = 0.080 if measured_offset is None else float(measured_offset)
    base = float(getattr(args, "pregrasp_offset", fallback))
    offsets = [base]
    if not bool(getattr(args, "adaptive_pregrasp_offsets", False)):
        return offsets

    step = max(float(getattr(args, "adaptive_pregrasp_step", 0.020)), 0.0)
    attempts = max(int(getattr(args, "adaptive_pregrasp_attempts", 4)), 1)
    minimum = max(
        float(getattr(args, "adaptive_pregrasp_min", 0.040)), 0.001
    )
    maximum = max(
        float(getattr(args, "adaptive_pregrasp_max", 0.160)), minimum
    )
    # Prefer one slightly shorter approach before progressively backing away.
    raw = [base, base - step]
    raw.extend(base + step * index for index in range(1, attempts))
    for value in raw:
        value = float(np.clip(value, minimum, maximum))
        if not any(abs(value - old) <= 1e-9 for old in offsets):
            offsets.append(value)
        if len(offsets) >= attempts:
            break
    return offsets


def _lift_relax_axis_local(args):
    axis_name = str(
        getattr(args, "lift_orientation_relax_axis", "tool_approach")
    )
    if axis_name == "tool_approach":
        return _approach_axis_local(args), str(args.tool_approach_axis)
    return _axis_vector(axis_name), axis_name


def _adaptive_lift_heights(args, requested_height):
    """Return the preferred lift followed by bounded lower fallbacks."""
    requested = max(float(requested_height), 0.0)
    heights = [requested]
    if not bool(getattr(args, "adaptive_lift_heights", False)):
        return heights
    step = max(float(getattr(args, "adaptive_lift_step", 0.020)), 0.001)
    minimum = max(
        min(
            float(getattr(args, "adaptive_lift_min_height", 0.080)),
            requested,
        ),
        0.001,
    )
    attempts = max(int(getattr(args, "adaptive_lift_attempts", 7)), 1)
    for index in range(1, attempts):
        height = max(requested - step * index, minimum)
        if not any(abs(height - old) <= 1e-9 for old in heights):
            heights.append(float(height))
        if height <= minimum + 1e-9:
            break
    return heights


def grasp_planning_pose_variants(pre_tcp, grasp_tcp, lift_tcp, args):
    """Build ordered planning fallbacks without adding execution waypoints.

    Order is deliberately fail-soft: the exact requested poses are first,
    followed by alternate pregrasp distances with the original lift
    orientation.  Small lift-orientation changes are considered only after
    every original-orientation pregrasp option has failed.
    """
    pre_tcp = np.asarray(pre_tcp, dtype=float).reshape(6)
    grasp_tcp = np.asarray(grasp_tcp, dtype=float).reshape(6)
    lift_tcp = np.asarray(lift_tcp, dtype=float).reshape(6)

    approach = grasp_tcp[:3] - pre_tcp[:3]
    norm = float(np.linalg.norm(approach))
    if norm <= 1e-9:
        # Geometry normally makes this identical to the configured local tool
        # approach.  Retain a deterministic fallback for injected/test poses.
        R_base_tcp = R.from_euler("xyz", grasp_tcp[3:]).as_matrix()
        approach = R_base_tcp @ _approach_axis_local(args)
        norm = float(np.linalg.norm(approach))
    approach = approach / max(norm, 1e-9)

    pre_variants = []
    for offset in _adaptive_pregrasp_offsets(args, measured_offset=norm):
        pose = grasp_tcp.copy()
        pose[:3] = grasp_tcp[:3] - approach * float(offset)
        # Preserve the original pose bit-for-bit for the primary attempt.
        if len(pre_variants) == 0:
            pose = pre_tcp.copy()
        pre_variants.append((pose, float(offset)))

    requested_lift_height = max(float(lift_tcp[2] - grasp_tcp[2]), 0.0)
    lift_heights = _adaptive_lift_heights(args, requested_lift_height)
    variants = []
    for rank, (pre_pose, offset) in enumerate(pre_variants):
        variants.append(
            {
                "pre_tcp": pre_pose.copy(),
                "grasp_tcp": grasp_tcp.copy(),
                "lift_tcp": lift_tcp.copy(),
                "pregrasp_offset": offset,
                "lift_height": requested_lift_height,
                "lift_height_fallback": False,
                "lift_relax_deg": 0.0,
                "lift_relax_axis": "none",
                "lift_region_fallback": False,
                "lift_xy_offset": (0.0, 0.0),
                "validation_only": False,
                "kind": "requested" if rank == 0 else "pregrasp_fallback",
            }
        )

    # Lowering the endpoint changes only the post-contact lift.  Keep all
    # requested-height pregrasp alternatives first so a pregrasp failure can
    # still be handled independently, then append lower lift endpoints for
    # the same already-validated grasp geometry.
    for height in lift_heights[1:]:
        lowered_lift = lift_tcp.copy()
        lowered_lift[2] = float(grasp_tcp[2]) + float(height)
        for pre_pose, offset in pre_variants:
            variants.append(
                {
                    "pre_tcp": pre_pose.copy(),
                    "grasp_tcp": grasp_tcp.copy(),
                    "lift_tcp": lowered_lift.copy(),
                    "pregrasp_offset": offset,
                    "lift_height": float(height),
                    "lift_height_fallback": True,
                    "lift_relax_deg": 0.0,
                    "lift_relax_axis": "none",
                    "lift_region_fallback": False,
                    "lift_xy_offset": (0.0, 0.0),
                    "validation_only": False,
                    "kind": "lift_height_fallback",
                }
            )

    region_enabled = bool(
        getattr(args, "grasp_lift_feasible_region", False)
    )
    region_height = float(lift_heights[-1])
    if region_enabled:
        # Try exact world-Z endpoints first.  A small deterministic cardinal
        # XY set is added only as a later workspace-boundary fallback.
        region_height = min(
            requested_lift_height,
            max(
                region_height,
                float(getattr(args, "grasp_validation_lift_height", 0.045)),
            ),
        )
        radius = max(float(getattr(args, "grasp_lift_region_xy", 0.020)), 0.0)
        if radius > 1e-9:
            for dx, dy in (
                (radius, 0.0),
                (-radius, 0.0),
                (0.0, radius),
                (0.0, -radius),
            ):
                shifted_lift = lift_tcp.copy()
                shifted_lift[0] = float(grasp_tcp[0]) + dx
                shifted_lift[1] = float(grasp_tcp[1]) + dy
                shifted_lift[2] = float(grasp_tcp[2]) + region_height
                for pre_pose, offset in pre_variants:
                    variants.append(
                        {
                            "pre_tcp": pre_pose.copy(),
                            "grasp_tcp": grasp_tcp.copy(),
                            "lift_tcp": shifted_lift.copy(),
                            "pregrasp_offset": offset,
                            "lift_height": region_height,
                            "lift_height_fallback": True,
                            "lift_relax_deg": 0.0,
                            "lift_relax_axis": "none",
                            "lift_region_fallback": True,
                            "lift_xy_offset": (dx, dy),
                            "validation_only": False,
                            "kind": "lift_region_xy_fallback",
                        }
                    )

    relax_enabled = bool(
        getattr(args, "lift_orientation_relax", False)
    )
    # Explicit RPY modes are operator constraints, not suggestions.
    if (
        not relax_enabled
        or getattr(args, "fixed_rpy", None) is not None
        or bool(getattr(args, "use_current_rpy", False))
    ):
        return variants

    degrees = getattr(args, "lift_orientation_relax_degrees", (3.0, 6.0))
    degrees = tuple(
        abs(float(value))
        for value in degrees
        if np.isfinite(float(value)) and abs(float(value)) > 1e-9
    )
    axis_local, axis_name = _lift_relax_axis_local(args)
    # Orientation relaxation is the last resort.  When height fallback is
    # enabled, apply it only to the lowest lift endpoint instead of multiplying
    # every height by every roll perturbation.
    relaxed_base_lift = lift_tcp.copy()
    relaxed_height = requested_lift_height
    if len(lift_heights) > 1:
        relaxed_height = region_height
        relaxed_base_lift[2] = float(grasp_tcp[2]) + relaxed_height
    base_rotation = R.from_euler("xyz", relaxed_base_lift[3:]).as_matrix()
    for magnitude in degrees:
        for sign in (1.0, -1.0):
            signed_deg = sign * magnitude
            delta = R.from_rotvec(
                axis_local * np.deg2rad(signed_deg)
            ).as_matrix()
            relaxed_lift = relaxed_base_lift.copy()
            relaxed_lift[3:] = R.from_matrix(
                base_rotation @ delta
            ).as_euler("xyz")
            for pre_pose, offset in pre_variants:
                variants.append(
                    {
                        "pre_tcp": pre_pose.copy(),
                        "grasp_tcp": grasp_tcp.copy(),
                        "lift_tcp": relaxed_lift.copy(),
                        "pregrasp_offset": offset,
                        "lift_height": relaxed_height,
                        "lift_height_fallback": bool(
                            relaxed_height < requested_lift_height - 1e-9
                        ),
                        "lift_relax_deg": float(signed_deg),
                        "lift_relax_axis": axis_name,
                        "lift_region_fallback": bool(region_enabled),
                        "lift_xy_offset": (0.0, 0.0),
                        "validation_only": False,
                        "kind": "lift_orientation_fallback",
                    }
                )
    return variants
