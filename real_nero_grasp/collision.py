"""Robot geometry and point-cloud collision validation for real Nero motion."""

import os
import xml.etree.ElementTree as ET

import cv2
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as R

from tools.real_nero_point_test import handeye_transform
from real_nero_grasp.grasp_geometry import _axis_vector
from real_nero_grasp.pointcloud_fusion import _points_from_cloud, transform_cloud
from real_nero_grasp.transforms import (
    _as_pose_list,
    _make_T_from_pose,
    _point_segment_distances,
    _transform_points,
)
from real_nero_grasp.types import (
    InitialSupportRelease,
    MotionSafetyScene,
    PlanNotExecutable,
    UnsafeMotionTarget,
)

_URDF_CHAIN_CACHE = {}

__all__ = [
    "table_guard_enabled",
    "table_guard_min_tcp_z",
    "assert_table_guard_target_tcp",
    "_urdf_vector",
    "_urdf_origin_transform",
    "_load_urdf_joint_chain",
    "_urdf_chain_geometry",
    "_joint_path_tcp_z_profile",
    "_camera_collision_capsules",
    "_active_robot_capsules",
    "_tcp_transform_from_joints",
    "_motion_open_gripper_finger_capsules",
    "_motion_robot_capsules",
    "_motion_self_filter_capsules",
    "filter_motion_robot_self_points",
    "_bounded_motion_points",
    "filter_motion_isolated_environment_points",
    "prepare_motion_safety_scene",
    "assert_pickup_payload_table_start",
    "_target_points_in_tcp",
    "_build_initial_support_release",
    "_motion_configuration_metrics",
    "_motion_joint_path_metrics",
    "validate_motion_joint_path",
    "_active_robot_path_metrics",
    "_active_motion_scene_points",
    "_validate_active_joint_path",
]


def filter_motion_isolated_environment_points(points, args, label="motion_environment"):
    """Remove collision-only occupancy that has no local geometric support.

    The input is already a base-frame, usually voxelized environment cloud.
    A real basket rim can be thin, but remains locally continuous; isolated
    depth ghosts instead have very few neighbours in a 20--30 mm ball. This
    filter is applied only to the collision scene, never to the target cloud.
    """
    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    if not bool(getattr(args, "motion_collision_isolated_filter", False)):
        return points
    radius = max(
        float(getattr(args, "motion_collision_isolated_radius", 0.020)), 0.0
    )
    min_support = max(
        int(getattr(args, "motion_collision_isolated_min_support", 8)), 1
    )
    if points.shape[0] < min_support or radius <= 0.0 or min_support <= 1:
        return points
    tree = cKDTree(points)
    try:
        support = np.asarray(
            tree.query_ball_point(points, radius, workers=-1, return_length=True),
            dtype=np.int32,
        )
    except TypeError:
        support = np.asarray(
            [len(indices) for indices in tree.query_ball_point(points, radius)],
            dtype=np.int32,
        )
    keep = support >= min_support
    removed = int(np.count_nonzero(~keep))
    if removed == 0:
        return points
    if not np.any(keep):
        print(
            f"[motion][isolated][warn] map={label} filter would remove all "
            f"{points.shape[0]} environment points; retain the original cloud"
        )
        return points
    filtered = points[keep]
    print(
        f"[motion][isolated] map={label} kept={filtered.shape[0]}/{points.shape[0]} "
        f"removed={removed} radius={radius * 1000.0:.1f}mm "
        f"min_support={min_support}"
    )
    return filtered.astype(np.float32, copy=False)

def table_guard_enabled(args):
    return bool(getattr(args, "table_guard_enable", False))

def table_guard_min_tcp_z(args):
    return (
        float(getattr(args, "table_z", 0.0))
        + float(getattr(args, "table_tcp_clearance", 0.0))
        + float(getattr(args, "table_tool_bottom_offset", 0.0))
    )

def assert_table_guard_target_tcp(tcp_pose, args, label):
    if not table_guard_enabled(args):
        return
    tcp_pose = np.asarray(tcp_pose, dtype=float).reshape(6)
    min_z = table_guard_min_tcp_z(args)
    if float(tcp_pose[2]) < min_z:
        raise UnsafeMotionTarget(
            f"{label} target TCP z={float(tcp_pose[2]):.4f}m below table guard "
            f"min_tcp_z={min_z:.4f}m "
            f"(table_z={float(args.table_z):.4f}, clearance={float(args.table_tcp_clearance):.4f}, "
            f"tool_bottom_offset={float(args.table_tool_bottom_offset):.4f})."
        )

def _urdf_vector(text, default):
    if text is None or str(text).strip() == "":
        return np.asarray(default, dtype=float).reshape(3)
    values = np.fromstring(str(text), sep=" ", dtype=float)
    if values.size != 3 or not np.isfinite(values).all():
        raise ValueError(f"invalid URDF vector: {text!r}")
    return values.reshape(3)

def _urdf_origin_transform(origin_element):
    xyz = _urdf_vector(
        None if origin_element is None else origin_element.attrib.get("xyz"),
        [0.0, 0.0, 0.0],
    )
    rpy = _urdf_vector(
        None if origin_element is None else origin_element.attrib.get("rpy"),
        [0.0, 0.0, 0.0],
    )
    T = np.eye(4, dtype=float)
    T[:3, :3] = R.from_euler("xyz", rpy, degrees=False).as_matrix()
    T[:3, 3] = xyz
    return T

def _load_urdf_joint_chain(args):
    """Load the base-to-tip chain used by IK for conservative link-sweep checks."""
    key = (
        os.path.abspath(str(args.ik_urdf)),
        str(args.ik_base_link),
        str(args.ik_tip_link),
    )
    cached = _URDF_CHAIN_CACHE.get(key)
    if cached is not None:
        return cached

    urdf_path, base_link, tip_link = key
    root = ET.parse(urdf_path).getroot()
    by_child = {}
    for joint in root.findall("joint"):
        parent_element = joint.find("parent")
        child_element = joint.find("child")
        if parent_element is None or child_element is None:
            continue
        child = str(child_element.attrib.get("link", ""))
        parent = str(parent_element.attrib.get("link", ""))
        axis_element = joint.find("axis")
        axis = _urdf_vector(
            None if axis_element is None else axis_element.attrib.get("xyz"),
            [0.0, 0.0, 1.0],
        )
        axis_norm = float(np.linalg.norm(axis))
        if axis_norm > 1e-9:
            axis = axis / axis_norm
        by_child[child] = {
            "name": str(joint.attrib.get("name", "unnamed_joint")),
            "type": str(joint.attrib.get("type", "fixed")),
            "parent": parent,
            "child": child,
            "origin": _urdf_origin_transform(joint.find("origin")),
            "axis": axis,
        }

    reversed_chain = []
    link = tip_link
    visited = set()
    while link != base_link:
        if link in visited:
            raise RuntimeError(f"URDF joint chain contains a cycle at {link!r}")
        visited.add(link)
        joint = by_child.get(link)
        if joint is None:
            raise RuntimeError(
                f"URDF has no joint chain from {base_link!r} to {tip_link!r}; "
                f"stopped at {link!r}"
            )
        reversed_chain.append(joint)
        link = joint["parent"]
    chain = tuple(reversed(reversed_chain))
    movable_count = sum(
        joint["type"] in {"revolute", "continuous", "prismatic"}
        for joint in chain
    )
    if movable_count != 7:
        raise RuntimeError(
            f"active path guard expects the 7-DOF Nero chain, got {movable_count} movable joints"
        )
    _URDF_CHAIN_CACHE[key] = chain
    return chain

def _urdf_chain_geometry(
    joint_angles,
    args,
    *,
    load_urdf_joint_chain_hook=None,
):
    """Return movable-joint centers and the IK-tip transform in robot base."""
    centers, tip, _ = _urdf_chain_frames(
        joint_angles, args,
        load_urdf_joint_chain_hook=load_urdf_joint_chain_hook,
    )
    return centers, tip


def _urdf_chain_frames(joint_angles, args, *, load_urdf_joint_chain_hook=None):
    """Also retain post-motion child-link frames for URDF collision geometry."""
    load_urdf_joint_chain = (
        _load_urdf_joint_chain
        if load_urdf_joint_chain_hook is None
        else load_urdf_joint_chain_hook
    )
    q = np.asarray(joint_angles, dtype=float).reshape(7)
    T = np.eye(4, dtype=float)
    centers = []
    frames = {str(args.ik_base_link): T.copy()}
    q_index = 0
    for joint in load_urdf_joint_chain(args):
        T = T @ np.asarray(joint["origin"], dtype=float).reshape(4, 4)
        joint_type = str(joint["type"])
        if joint_type not in {"revolute", "continuous", "prismatic"}:
            frames[joint["child"]] = T.copy()
            continue
        centers.append(T[:3, 3].copy())
        value = float(q[q_index])
        q_index += 1
        motion = np.eye(4, dtype=float)
        axis = np.asarray(joint["axis"], dtype=float).reshape(3)
        if joint_type in {"revolute", "continuous"}:
            motion[:3, :3] = R.from_rotvec(axis * value).as_matrix()
        else:
            motion[:3, 3] = axis * value
        T = T @ motion
        frames[joint["child"]] = T.copy()
    return np.asarray(centers, dtype=float).reshape(-1, 3), T, frames


def _arm_collision_capsules(joint_angles, args, centers, legacy_radius):
    mode = str(getattr(args, "arm_collision_model", "legacy_capsules"))
    if mode == "urdf_capsules":
        from .arm_collision_geometry import arm_capsules
        _, _, frames = _urdf_chain_frames(joint_angles, args)
        return arm_capsules(args, _load_urdf_joint_chain(args), frames)
    if mode != "legacy_capsules":
        raise UnsafeMotionTarget(f"Unknown arm collision model: {mode}")
    capsules = [
        (center.copy(), center.copy(), legacy_radius, f"joint{index + 1}")
        for index, center in enumerate(centers)
    ]
    for index in range(max(centers.shape[0] - 1, 0)):
        a, b = centers[index:index + 2]
        if float(np.linalg.norm(b - a)) > 1e-5:
            capsules.append((a.copy(), b.copy(), legacy_radius, f"link{index + 1}_{index + 2}"))
    return capsules


def initialize_arm_collision_geometry(args):
    """Fit and verify CAD coverage before opening a hardware/control session."""
    if str(getattr(args, "arm_collision_model", "legacy_capsules")) != "urdf_capsules":
        return
    from .arm_collision_geometry import load_arm_geometry
    geometry = load_arm_geometry(args, _load_urdf_joint_chain(args))
    radii = [radius for item in geometry["report"] for radius in item["radii_mm"]]
    print(
        f"[collision][arm] model=urdf_capsules links={len(geometry['models'])} "
        f"capsules={len(radii)} radius_range={min(radii):.1f}..{max(radii):.1f}mm "
        f"mesh_padding={geometry['padding_m'] * 1000:.1f}mm "
        "all_collision_triangles_covered=true"
    )

def _joint_path_tcp_z_profile(
    q_start,
    q_target,
    args,
    *,
    urdf_chain_geometry_hook=None,
):
    """Sample absolute joint interpolation and return its modeled TCP heights."""
    urdf_chain_geometry = (
        _urdf_chain_geometry
        if urdf_chain_geometry_hook is None
        else urdf_chain_geometry_hook
    )
    q_start = np.asarray(q_start, dtype=float).reshape(7)
    q_target = np.asarray(q_target, dtype=float).reshape(7)
    delta = q_target - q_start
    sample_step = np.deg2rad(
        max(float(getattr(args, "initial_table_path_sample_joint_deg", 0.5)), 0.1)
    )
    sample_count = max(
        2,
        int(np.ceil(float(np.max(np.abs(delta))) / sample_step)) + 1,
    )
    tcp_offset = np.asarray(args.tcp_offset, dtype=float).reshape(6)
    z_values = []
    for alpha in np.linspace(0.0, 1.0, sample_count):
        q_sample = q_start + float(alpha) * delta
        _, T_base_flange = urdf_chain_geometry(q_sample, args)
        tcp_xyz = (T_base_flange @ np.r_[tcp_offset[:3], 1.0])[:3]
        z_values.append(float(tcp_xyz[2]))
    z_values = np.asarray(z_values, dtype=float)
    return {
        "samples": int(sample_count),
        "start_z": float(z_values[0]),
        "target_z": float(z_values[-1]),
        "min_z": float(np.min(z_values)),
        "min_sample": int(np.argmin(z_values)),
        "z_values": z_values,
        "max_joint_delta_deg": float(np.rad2deg(np.max(np.abs(delta)))),
    }

def _camera_collision_capsules(
    T_base_flange,
    args,
    body_radius,
    *,
    handeye_transform_hook=None,
):
    """Shared optical-frame-aware camera body and mount collision model.

    The hand-eye translation is the camera optical centre, not the centre of a
    solid ball.  The old flange-to-optical-centre capsule placed its spherical
    55 mm end cap in front of the lens and consequently classified visible
    scene points as camera collisions.  Keep the body entirely behind the
    optical plane and cover the physical bracket separately with a thinner
    capsule.
    """
    handeye = (
        handeye_transform
        if handeye_transform_hook is None
        else handeye_transform_hook
    )
    T_base_flange = np.asarray(T_base_flange, dtype=float).reshape(4, 4)
    T_flange_camera = handeye(
        args.calib_json, args.handeye_convention
    )
    T_base_camera = T_base_flange @ np.asarray(
        T_flange_camera, dtype=float
    ).reshape(4, 4)
    optical_center = np.asarray(T_base_camera[:3, 3], dtype=float).reshape(3)
    optical_axis = np.asarray(T_base_camera[:3, 2], dtype=float).reshape(3)
    axis_norm = float(np.linalg.norm(optical_axis))
    if (
        not np.isfinite(T_base_camera).all()
        or not np.isfinite(axis_norm)
        or axis_norm < 1e-9
    ):
        raise ValueError("camera hand-eye transform is non-finite or singular")
    optical_axis /= axis_norm

    radius = max(float(body_radius), 0.005)
    if not np.isfinite(radius):
        raise ValueError("camera collision body radius is non-finite")
    # A capsule end cap extends by one radius.  Starting its centreline one
    # radius behind the optical centre makes the front surface tangent to the
    # optical plane rather than occupying the camera field of view.
    rear_extent = max(
        float(
            getattr(
                args,
                "camera_collision_body_rear_extent",
                max(0.120, 2.0 * radius),
            )
        ),
        2.0 * radius,
    )
    if not np.isfinite(rear_extent):
        raise ValueError("camera collision body rear extent is non-finite")
    body_front = optical_center - radius * optical_axis
    body_back = optical_center - (rear_extent - radius) * optical_axis

    mount_radius = min(
        max(
            float(getattr(args, "camera_collision_mount_radius", 0.025)),
            0.003,
        ),
        radius,
    )
    if not np.isfinite(mount_radius):
        raise ValueError("camera collision mount radius is non-finite")
    mount_depth = float(
        getattr(args, "camera_collision_mount_attachment_depth", 0.060)
    )
    if not np.isfinite(mount_depth):
        raise ValueError("camera collision mount attachment depth is non-finite")
    # Keep the bracket end cap behind the optical plane as well, and attach it
    # within the modeled body depth.
    mount_depth = float(
        np.clip(mount_depth, radius, rear_extent - radius)
    )
    mount_attachment = optical_center - mount_depth * optical_axis
    flange = np.asarray(T_base_flange[:3, 3], dtype=float).reshape(3)
    return [
        (
            body_front.copy(),
            body_back.copy(),
            radius,
            "camera_body",
        ),
        (
            flange.copy(),
            mount_attachment.copy(),
            mount_radius,
            "camera_mount",
        ),
    ]

def _active_robot_capsules(
    joint_angles,
    args,
    *,
    urdf_chain_geometry_hook=None,
    camera_collision_capsules_hook=None,
):
    """Approximate arm, open gripper and camera mount with conservative capsules."""
    urdf_chain_geometry = (
        _urdf_chain_geometry
        if urdf_chain_geometry_hook is None
        else urdf_chain_geometry_hook
    )
    camera_collision_capsules = (
        _camera_collision_capsules
        if camera_collision_capsules_hook is None
        else camera_collision_capsules_hook
    )
    centers, T_base_flange = urdf_chain_geometry(joint_angles, args)
    link_radius = float(getattr(args, "active_path_link_radius", 0.060))
    tool_radius = float(getattr(args, "active_path_tool_radius", 0.090))
    camera_radius = float(getattr(args, "active_path_camera_radius", 0.055))
    capsules = _arm_collision_capsules(joint_angles, args, centers, link_radius)

    flange = np.asarray(T_base_flange[:3, 3], dtype=float).reshape(3)
    tcp_offset = np.asarray(args.tcp_offset, dtype=float).reshape(6)
    tcp = (T_base_flange @ np.r_[tcp_offset[:3], 1.0])[:3]
    capsules.append((flange.copy(), tcp.copy(), tool_radius, "open_gripper"))

    try:
        capsules.extend(
            camera_collision_capsules(T_base_flange, args, camera_radius)
        )
    except Exception as exc:
        if bool(getattr(args, "active_path_require_camera_model", True)):
            raise UnsafeMotionTarget(
                f"active path guard cannot build the camera-mount capsule: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
    return capsules

def _tcp_transform_from_joints(
    joint_angles,
    args,
    *,
    urdf_chain_geometry_hook=None,
    make_T_from_pose_hook=None,
):
    """FK transform for the configured TCP, including offset orientation."""
    urdf_chain_geometry = (
        _urdf_chain_geometry
        if urdf_chain_geometry_hook is None
        else urdf_chain_geometry_hook
    )
    make_T_from_pose = (
        _make_T_from_pose
        if make_T_from_pose_hook is None
        else make_T_from_pose_hook
    )
    _, T_base_flange = urdf_chain_geometry(joint_angles, args)
    return T_base_flange @ make_T_from_pose(
        np.asarray(args.tcp_offset, dtype=float).reshape(6)
    )

def _motion_open_gripper_finger_capsules(
    T_base_tcp,
    args,
    *,
    opening=None,
    primitive_prefix="gripper_finger",
    clamp_to_hardware=True,
    overlap_centerlines=False,
    axis_vector_hook=None,
):
    """Collision capsules for the two physical fingers at a requested gap."""
    axis_vector = _axis_vector if axis_vector_hook is None else axis_vector_hook
    T_base_tcp = np.asarray(T_base_tcp, dtype=float).reshape(4, 4)
    tcp = np.asarray(T_base_tcp[:3, 3], dtype=float).reshape(3)
    tcp_rotation = np.asarray(T_base_tcp[:3, :3], dtype=float).reshape(3, 3)
    width_axis = tcp_rotation @ axis_vector(
        getattr(args, "tcp_gripper_width_axis", "z")
    )
    approach_axis = tcp_rotation @ axis_vector(
        getattr(args, "tool_approach_axis", "x")
    )
    width_axis /= np.linalg.norm(width_axis) + 1e-12
    approach_axis /= np.linalg.norm(approach_axis) + 1e-12
    finger_radius = max(
        float(getattr(args, "motion_path_open_finger_radius", 0.007)), 0.003
    )
    hardware_max = max(
        float(getattr(args, "gripper_hardware_max_width", 0.100)), 0.0
    )
    if opening is None:
        opening = float(getattr(args, "gripper_open_width", 0.100))
    opening = max(float(opening), 0.0)
    if bool(clamp_to_hardware):
        opening = min(opening, hardware_max)
    # Gripper width feedback/commands describe the clear gap between the two
    # inner jaw faces.  A capsule center therefore sits one full capsule radius
    # outside half that gap.  Using only half a radius here modeled an 85 mm
    # opening as a 71 mm free gap and systematically created false finger
    # collisions for otherwise simple grasps.
    half_finger_spacing = (
        0.0
        if bool(overlap_centerlines)
        else 0.5 * max(opening, 0.0) + finger_radius
    )
    finger_back = max(
        float(getattr(args, "motion_path_open_finger_back", 0.065)), 0.010
    )
    finger_front = max(
        float(getattr(args, "motion_path_open_finger_front", 0.020)), 0.0
    )
    capsules = []
    for sign in (-1.0, 1.0):
        offset = sign * half_finger_spacing * width_axis
        a = tcp + offset - finger_back * approach_axis
        b = tcp + offset + finger_front * approach_axis
        capsules.append(
            (
                a.copy(),
                b.copy(),
                finger_radius,
                f"{primitive_prefix}_{int(sign):+d}",
            )
        )
    return capsules

def _motion_robot_capsules(
    joint_angles,
    args,
    open_gripper=False,
    *,
    closed_gripper_width=None,
    urdf_chain_geometry_hook=None,
    make_T_from_pose_hook=None,
    open_gripper_finger_capsules_hook=None,
    camera_collision_capsules_hook=None,
):
    """Conservative robot model for grasp/transport execution paths."""
    urdf_chain_geometry = (
        _urdf_chain_geometry
        if urdf_chain_geometry_hook is None
        else urdf_chain_geometry_hook
    )
    make_T_from_pose = (
        _make_T_from_pose
        if make_T_from_pose_hook is None
        else make_T_from_pose_hook
    )
    open_gripper_finger_capsules = (
        _motion_open_gripper_finger_capsules
        if open_gripper_finger_capsules_hook is None
        else open_gripper_finger_capsules_hook
    )
    camera_collision_capsules = (
        _camera_collision_capsules
        if camera_collision_capsules_hook is None
        else camera_collision_capsules_hook
    )
    centers, T_base_flange = urdf_chain_geometry(joint_angles, args)
    link_radius = float(getattr(args, "motion_path_link_radius", 0.060))
    # Keep the legacy option name for CLI compatibility.  The reduced 35 mm
    # radius applies only to the proximal palm/body; it is never used as a
    # solid post-close distal tool model.
    palm_radius = float(getattr(args, "motion_path_tool_radius", 0.035))
    camera_radius = float(getattr(args, "motion_path_camera_radius", 0.055))
    capsules = _arm_collision_capsules(joint_angles, args, centers, link_radius)

    flange = np.asarray(T_base_flange[:3, 3], dtype=float).reshape(3)
    T_base_tcp = T_base_flange @ make_T_from_pose(
        np.asarray(args.tcp_offset, dtype=float).reshape(6)
    )
    tcp = np.asarray(T_base_tcp[:3, 3], dtype=float).reshape(3)
    tool_axis = tcp - flange
    tool_length = float(np.linalg.norm(tool_axis))
    contact_length = max(
        float(getattr(args, "motion_target_tool_contact_length", 0.045)),
        0.005,
    )
    if tool_length > 1e-6:
        tool_direction = tool_axis / tool_length
        # A capsule includes spherical end caps.  End the palm/body centerline
        # one radius farther back so its distal sphere stops at the contact-zone
        # boundary instead of protruding through the TCP and rejecting every
        # legitimate held object.
        body_end = tcp - tool_direction * min(
            contact_length + palm_radius, tool_length
        )
    else:
        body_end = flange.copy()
    if float(np.linalg.norm(body_end - flange)) > 1e-6:
        capsules.append(
            (flange.copy(), body_end.copy(), palm_radius, "gripper_body")
        )
    if bool(open_gripper):
        # During free/open observation motion the 85 mm space between the jaws
        # is empty.  A filled 55 mm terminal capsule incorrectly turns basket
        # edges in that gap into robot collisions.  Keep the conservative palm
        # above and model only the two actual distal fingers here.
        capsules.extend(open_gripper_finger_capsules(T_base_tcp, args))
    else:
        # A closed gripper is not a solid 55 mm cylinder.  Keep the palm/body
        # conservative, but represent the distal geometry as the two physical
        # fingers.  An empty close overlaps both finger centerlines into one
        # compact envelope; a held-object plan supplies the candidate GraspNet
        # width plus its post-close collision allowance.
        if closed_gripper_width is None:
            closed_gripper_width = float(
                getattr(args, "motion_closed_gripper_empty_width", 0.000)
            )
        capsules.extend(
            open_gripper_finger_capsules(
                T_base_tcp,
                args,
                opening=closed_gripper_width,
                primitive_prefix="gripper_closed_finger",
                # This is a conservative post-grasp collision envelope, not a
                # hardware command.  GraspNet width + allowance may therefore
                # exceed the nominal 85 mm opening without changing actuation.
                clamp_to_hardware=False,
                overlap_centerlines=float(closed_gripper_width) <= 0.0,
            )
        )
    try:
        capsules.extend(
            camera_collision_capsules(T_base_flange, args, camera_radius)
        )
    except Exception as exc:
        if bool(getattr(args, "motion_path_require_camera_model", True)):
            raise UnsafeMotionTarget(
                "motion path guard cannot build the camera-mount capsule: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
    return capsules

def _motion_self_filter_capsules(
    joint_angles,
    args,
    *,
    urdf_chain_geometry_hook=None,
    make_T_from_pose_hook=None,
    axis_vector_hook=None,
    handeye_transform_hook=None,
):
    """Approximate occupied robot geometry for removing camera self-points.

    Unlike collision capsules, the open gripper is represented as two narrow
    fingers, not one filled cylinder.  This removes the fingers visible at the
    bottom of the RGB-D frame without erasing a real object in the free space
    between them.
    """
    urdf_chain_geometry = (
        _urdf_chain_geometry
        if urdf_chain_geometry_hook is None
        else urdf_chain_geometry_hook
    )
    make_T_from_pose = (
        _make_T_from_pose
        if make_T_from_pose_hook is None
        else make_T_from_pose_hook
    )
    axis_vector = _axis_vector if axis_vector_hook is None else axis_vector_hook
    handeye = (
        handeye_transform
        if handeye_transform_hook is None
        else handeye_transform_hook
    )
    centers, T_base_flange = urdf_chain_geometry(joint_angles, args)
    link_radius = max(
        float(getattr(args, "motion_self_filter_link_radius", 0.045)), 0.005
    )
    palm_radius = max(
        float(getattr(args, "motion_self_filter_palm_radius", 0.035)), 0.005
    )
    camera_radius = max(
        float(getattr(args, "motion_self_filter_camera_radius", 0.040)), 0.005
    )
    finger_radius = max(
        float(getattr(args, "motion_self_filter_finger_radius", 0.007)), 0.003
    )
    capsules = []
    for index, center in enumerate(centers):
        capsules.append(
            (center.copy(), center.copy(), link_radius, f"self_joint{index + 1}")
        )
    for index in range(max(centers.shape[0] - 1, 0)):
        a = centers[index]
        b = centers[index + 1]
        if float(np.linalg.norm(b - a)) > 1e-5:
            capsules.append(
                (a.copy(), b.copy(), link_radius, f"self_link{index + 1}_{index + 2}")
            )

    flange = np.asarray(T_base_flange[:3, 3], dtype=float).reshape(3)
    T_base_tcp = T_base_flange @ make_T_from_pose(
        np.asarray(args.tcp_offset, dtype=float).reshape(6)
    )
    tcp = np.asarray(T_base_tcp[:3, 3], dtype=float).reshape(3)
    tool_axis = tcp - flange
    tool_length = float(np.linalg.norm(tool_axis))
    if tool_length > 1e-6:
        palm_end = tcp - tool_axis / tool_length * min(0.055, tool_length)
        if float(np.linalg.norm(palm_end - flange)) > 1e-6:
            capsules.append(
                (flange.copy(), palm_end.copy(), palm_radius, "self_gripper_palm")
            )

    tcp_rotation = np.asarray(T_base_tcp[:3, :3], dtype=float).reshape(3, 3)
    width_axis = tcp_rotation @ axis_vector(
        getattr(args, "tcp_gripper_width_axis", "z")
    )
    approach_axis = tcp_rotation @ axis_vector(
        getattr(args, "tool_approach_axis", "x")
    )
    width_axis /= np.linalg.norm(width_axis) + 1e-12
    approach_axis /= np.linalg.norm(approach_axis) + 1e-12
    opening = min(
        float(getattr(args, "gripper_open_width", 0.100)),
        float(getattr(args, "gripper_hardware_max_width", 0.100)),
    )
    # Match the collision model: ``opening`` is the inner-face gap, while the
    # capsule centerline is another full finger radius outward.
    half_finger_spacing = 0.5 * max(opening, 0.0) + finger_radius
    finger_back = max(
        float(getattr(args, "motion_self_filter_finger_back", 0.065)), 0.010
    )
    finger_front = max(
        float(getattr(args, "motion_self_filter_finger_front", 0.020)), 0.0
    )
    for sign in (-1.0, 1.0):
        offset = sign * half_finger_spacing * width_axis
        a = tcp + offset - finger_back * approach_axis
        b = tcp + offset + finger_front * approach_axis
        capsules.append(
            (a.copy(), b.copy(), finger_radius, f"self_finger_{int(sign):+d}")
        )

    try:
        T_flange_camera = handeye(
            args.calib_json, args.handeye_convention
        )
        camera = (T_base_flange @ T_flange_camera)[:3, 3]
        capsules.append(
            (flange.copy(), camera.copy(), camera_radius, "self_camera_mount")
        )
    except Exception:
        # Collision planning separately fails closed if the required camera
        # model is unavailable.  Self filtering can conservatively do less.
        pass
    return capsules

def filter_motion_robot_self_points(
    points,
    joint_angles,
    args,
    label,
    *,
    motion_self_filter_capsules_hook=None,
    point_segment_distances_hook=None,
):
    motion_self_filter_capsules = (
        _motion_self_filter_capsules
        if motion_self_filter_capsules_hook is None
        else motion_self_filter_capsules_hook
    )
    point_segment_distances = (
        _point_segment_distances
        if point_segment_distances_hook is None
        else point_segment_distances_hook
    )
    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    if points.shape[0] == 0 or not bool(
        getattr(args, "motion_self_filter", True)
    ):
        return points.copy()
    keep = np.ones(points.shape[0], dtype=bool)
    margin = max(
        float(getattr(args, "motion_self_filter_margin", 0.005)), 0.0
    )
    for a, b, radius, _primitive in motion_self_filter_capsules(
        joint_angles, args
    ):
        distances = point_segment_distances(points, a, b)
        keep &= distances > float(radius) + margin
    filtered = points[keep]
    print(
        f"[{label}][self_filter] removed={int(points.shape[0] - filtered.shape[0])}/"
        f"{int(points.shape[0])} robot-visible points"
    )
    return filtered.astype(np.float32, copy=False)

def _bounded_motion_points(points, args):
    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    points = points[np.isfinite(points).all(axis=1)]
    voxel = float(getattr(args, "motion_path_scene_voxel", 0.005))
    if voxel > 0.0 and points.shape[0] > 0:
        keys = np.floor(points / voxel).astype(np.int64)
        _, unique_indices = np.unique(keys, axis=0, return_index=True)
        points = points[np.sort(unique_indices)]
    cap = int(max(getattr(args, "motion_path_obstacle_cap", 50000), 1))
    if points.shape[0] > cap:
        if bool(getattr(args, "motion_path_fail_on_obstacle_cap", True)):
            raise PlanNotExecutable(
                f"motion safety cloud has {points.shape[0]} occupied voxels > "
                f"cap={cap}; refusing to discard sparse obstacles"
            )
        indices = np.linspace(0, points.shape[0] - 1, cap, dtype=np.int64)
        points = points[indices]
    return points.astype(np.float32, copy=False)

def prepare_motion_safety_scene(
    target_cloud,
    motion_scene_cloud,
    T_base_camera,
    args,
    map_id="capture",
    motion_environment_cloud=None,
    current_joints=None,
    environment_map=None,
    *,
    fuse_dynamic_environment_clouds_hook=None,
    bounded_motion_points_hook=None,
    points_from_cloud_hook=None,
    transform_cloud_hook=None,
    filter_motion_robot_self_points_hook=None,
):
    """Build immutable full/non-target clouds in ``base_link`` coordinates."""
    bounded_motion_points = (
        _bounded_motion_points
        if bounded_motion_points_hook is None
        else bounded_motion_points_hook
    )
    points_from_cloud = (
        _points_from_cloud
        if points_from_cloud_hook is None
        else points_from_cloud_hook
    )
    transform_cloud_impl = (
        transform_cloud if transform_cloud_hook is None else transform_cloud_hook
    )
    filter_self_points = (
        filter_motion_robot_self_points
        if filter_motion_robot_self_points_hook is None
        else filter_motion_robot_self_points_hook
    )
    if environment_map is not None and bool(
        getattr(args, "motion_environment_map_enabled", True)
    ):
        required_fields = (
            "frame_id",
            "hard_environment_base",
            "uncertain_environment_base",
            "target_base",
            "generation",
        )
        missing_fields = [
            name for name in required_fields if not hasattr(environment_map, name)
        ]
        if missing_fields:
            raise PlanNotExecutable(
                "dynamic motion environment map is missing required field(s): "
                + ", ".join(missing_fields)
            )
        frame_id = str(getattr(environment_map, "frame_id"))
        if frame_id != "base_link":
            raise PlanNotExecutable(
                "dynamic motion environment map frame_id must be 'base_link', "
                f"got {frame_id!r}"
            )
        unavailable_fields = [
            name
            for name in (
                "hard_environment_base",
                "uncertain_environment_base",
                "target_base",
            )
            if getattr(environment_map, name) is None
        ]
        if unavailable_fields:
            raise PlanNotExecutable(
                "dynamic motion environment map field(s) are unavailable: "
                + ", ".join(unavailable_fields)
            )
        try:
            generation_value = float(getattr(environment_map, "generation"))
        except Exception as exc:
            raise PlanNotExecutable(
                "dynamic motion environment map generation is invalid: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        if not np.isfinite(generation_value) or generation_value < 0.0:
            raise PlanNotExecutable(
                "dynamic motion environment map generation must be a finite "
                f"non-negative integer, got {generation_value!r}"
            )
        generation = int(generation_value)
        if float(generation) != generation_value:
            raise PlanNotExecutable(
                "dynamic motion environment map generation must be an integer, "
                f"got {generation_value!r}"
            )
        try:
            hard_environment_base = bounded_motion_points(
                getattr(environment_map, "hard_environment_base"), args
            )
            uncertain_environment_base = bounded_motion_points(
                getattr(environment_map, "uncertain_environment_base"), args
            )
            map_target_base = bounded_motion_points(
                getattr(environment_map, "target_base"), args
            )
        except (PlanNotExecutable, UnsafeMotionTarget):
            raise
        except Exception as exc:
            raise PlanNotExecutable(
                "dynamic motion environment map contains invalid base_link point "
                f"arrays: {type(exc).__name__}: {exc}"
            ) from exc
        # The dynamic map owns obstacle occupancy, but its target snapshot may
        # have been assembled before target-view registration and quality
        # rejection.  When the caller supplies the final planning target,
        # always use that cloud for aperture/payload geometry.  Falling back
        # to the map target keeps old/offline map-only callers compatible.
        target_source = "dynamic_environment_map_snapshot"
        target_base = map_target_base
        if target_cloud is not None:
            if T_base_camera is None:
                raise PlanNotExecutable(
                    "final planning target requires T_base_camera when a "
                    "dynamic environment map is supplied"
                )
            target_base = bounded_motion_points(
                points_from_cloud(
                    transform_cloud_impl(
                        target_cloud,
                        np.asarray(T_base_camera, dtype=float).reshape(4, 4),
                    )
                ),
                args,
            )
            target_source = "final_planning_target_cloud"

        environment_parts = [
            points
            for points in (
                hard_environment_base,
                uncertain_environment_base,
            )
            if points.shape[0] > 0
        ]
        environment_base = bounded_motion_points(
            np.vstack(environment_parts)
            if environment_parts
            else np.zeros((0, 3), dtype=np.float32),
            args,
        )
        environment_base = filter_motion_isolated_environment_points(
            environment_base,
            args,
            f"{map_id}_dynamic",
        )
        full_parts = [environment_base]
        if target_base.shape[0] > 0:
            full_parts.append(target_base)
        full_scene_base = bounded_motion_points(
            np.vstack(full_parts),
            args,
        )
        if target_base.shape[0] == 0:
            raise PlanNotExecutable(
                "dynamic motion environment map target cloud is empty"
            )
        if environment_base.shape[0] == 0 and bool(
            getattr(args, "motion_path_require_scene", True)
        ):
            raise PlanNotExecutable(
                "dynamic motion environment map has no hard or uncertain "
                "environment occupancy"
            )
        source_map_id = str(getattr(environment_map, "map_id", "dynamic"))
        print(
            f"[motion][scene] map={map_id} full={full_scene_base.shape[0]} "
            f"target={target_base.shape[0]} environment={environment_base.shape[0]} "
            f"environment_source=dynamic_environment_map "
            f"hard={hard_environment_base.shape[0]} "
            f"uncertain={uncertain_environment_base.shape[0]} "
            f"source_map={source_map_id} generation={generation} "
            f"target_source={target_source} frame=base_link"
        )
        return MotionSafetyScene(
            full_scene_base=full_scene_base,
            environment_base=environment_base,
            target_base=target_base,
            map_id=str(map_id),
            environment_map=environment_map,
            generation=generation,
        )
    T_base_camera = np.asarray(T_base_camera, dtype=float).reshape(4, 4)
    target_base = bounded_motion_points(
        points_from_cloud(transform_cloud_impl(target_cloud, T_base_camera)), args
    )
    full_scene_base = bounded_motion_points(
        points_from_cloud(
            transform_cloud_impl(motion_scene_cloud, T_base_camera)
        ),
        args,
    )
    if current_joints is not None:
        full_scene_base = filter_self_points(
            full_scene_base,
            current_joints,
            args,
            f"motion_scene_{map_id}",
        )
        # Self filtering must never erase the explicitly segmented object.
        full_scene_base = bounded_motion_points(
            np.vstack([full_scene_base, target_base]), args
        )
    if target_base.shape[0] == 0:
        raise PlanNotExecutable("motion safety target cloud is empty")
    if full_scene_base.shape[0] == 0:
        raise PlanNotExecutable("motion safety scene cloud is empty")

    # Prefer the mask-owned non-target cloud.  NN subtraction is retained only
    # for old/offline captures that did not save this exact partition.
    exclusion = max(float(getattr(args, "motion_target_exclusion_dist", 0.002)), 0.0)
    environment_source = "mask_non_target"
    if motion_environment_cloud is not None:
        environment_base = bounded_motion_points(
            points_from_cloud(
                transform_cloud_impl(motion_environment_cloud, T_base_camera)
            ),
            args,
        )
        if current_joints is not None:
            environment_base = filter_self_points(
                environment_base,
                current_joints,
                args,
                f"motion_environment_{map_id}",
            )
    elif exclusion > 0.0:
        environment_source = "tight_target_nn_fallback"
        distances, _ = cKDTree(target_base).query(
            full_scene_base, k=1, workers=-1
        )
        environment_base = full_scene_base[np.asarray(distances) > exclusion]
    else:
        environment_source = "full_scene_no_exclusion"
        environment_base = full_scene_base.copy()
    environment_base = bounded_motion_points(environment_base, args)
    if environment_base.shape[0] == 0 and bool(
        getattr(args, "motion_path_require_scene", True)
    ):
        raise PlanNotExecutable(
            "motion safety environment became empty after selected-target removal"
        )

    # Clear-first single-view planning happens before the active-view state
    # machine can construct a canonical map.  In production parsed arguments
    # explicitly carry this flag, so promote the already transformed and
    # self-filtered base-frame partition through the same fusion owner here.
    # Lightweight legacy/test Namespaces without the new option retain the
    # historical arrays unchanged.
    generated_environment_map = None
    if bool(getattr(args, "motion_environment_map_enabled", False)):
        try:
            if fuse_dynamic_environment_clouds_hook is None:
                from .pointcloud_fusion import (
                    fuse_dynamic_environment_clouds as fuse_environment,
                    next_dynamic_environment_generation,
                )
            else:
                fuse_environment = fuse_dynamic_environment_clouds_hook
                from .pointcloud_fusion import (
                    next_dynamic_environment_generation,
                )
            generation = next_dynamic_environment_generation(args)
            generated_environment_map = fuse_environment(
                [f"{map_id}_single"],
                [environment_base],
                args,
                target_base=target_base,
                map_id=f"{map_id}_single_environment",
                generation=generation,
                target_center_base=np.median(target_base, axis=0),
            )
            hard = np.asarray(
                generated_environment_map.hard_environment_base,
                dtype=np.float32,
            ).reshape(-1, 3)
            uncertain = np.asarray(
                generated_environment_map.uncertain_environment_base,
                dtype=np.float32,
            ).reshape(-1, 3)
            environment_parts = [
                points for points in (hard, uncertain) if points.shape[0] > 0
            ]
            environment_base = bounded_motion_points(
                np.vstack(environment_parts)
                if environment_parts
                else np.zeros((0, 3), dtype=np.float32),
                args,
            )
            if environment_base.shape[0] == 0 and bool(
                getattr(args, "motion_path_require_scene", True)
            ):
                raise PlanNotExecutable(
                    "table-level dynamic environment map is empty after ROI "
                    "and confidence fusion"
                )
            full_scene_base = bounded_motion_points(
                np.vstack([environment_base, target_base]), args
            )
            environment_source = "generated_dynamic_environment_map"
        except (PlanNotExecutable, UnsafeMotionTarget):
            raise
        except Exception as exc:
            # The old exact environment remains a conservative safe fallback;
            # a map-construction software error must be visible in the log.
            generated_environment_map = None
            print(
                "[motion][scene][warn] table-level environment map generation "
                f"failed; use legacy base-frame occupancy: "
                f"{type(exc).__name__}: {exc}"
            )
    environment_base = filter_motion_isolated_environment_points(
        environment_base,
        args,
        f"{map_id}_{environment_source}",
    )
    full_scene_base = bounded_motion_points(
        np.vstack([environment_base, target_base]), args
    )
    print(
        f"[motion][scene] map={map_id} full={full_scene_base.shape[0]} "
        f"target={target_base.shape[0]} environment={environment_base.shape[0]} "
        f"environment_source={environment_source} "
        f"target_exclusion={exclusion * 1000.0:.1f}mm frame=base_link"
    )
    return MotionSafetyScene(
        full_scene_base=full_scene_base,
        environment_base=environment_base,
        target_base=target_base,
        map_id=str(map_id),
        environment_map=generated_environment_map,
        generation=(
            0
            if generated_environment_map is None
            else int(generated_environment_map.generation)
        ),
    )

def _target_points_in_tcp(
    target_base,
    grasp_tcp,
    args,
    *,
    bounded_motion_points_hook=None,
    make_T_from_pose_hook=None,
    transform_points_hook=None,
):
    bounded_motion_points = (
        _bounded_motion_points
        if bounded_motion_points_hook is None
        else bounded_motion_points_hook
    )
    make_T_from_pose = (
        _make_T_from_pose
        if make_T_from_pose_hook is None
        else make_T_from_pose_hook
    )
    transform_points = (
        _transform_points
        if transform_points_hook is None
        else transform_points_hook
    )
    target_base = bounded_motion_points(target_base, args)
    T_tcp_base = np.linalg.inv(make_T_from_pose(grasp_tcp))
    return bounded_motion_points(transform_points(T_tcp_base, target_base), args)


def assert_pickup_payload_table_start(scene, args, label):
    """Validate or establish the payload/table contact baseline before IK.

    At the first lift sample the payload is still at the grasp TCP, so its
    base-frame geometry is exactly ``scene.target_base`` and is independent
    of the robot IK branch.  A segmented object can legitimately rest on the
    calibrated table/mat and RGB-D/calibration error can put its robust floor
    slightly below that plane.  Such bounded start contact is accepted as a
    baseline; the pickup path may only reduce it, never move farther down.
    """
    if not table_guard_enabled(args):
        return
    target_base = np.asarray(scene.target_base, dtype=float).reshape(-1, 3)
    if target_base.shape[0] == 0:
        return
    floor_percentile = float(
        np.clip(
            getattr(args, "motion_payload_table_percentile", 10.0),
            0.0,
            25.0,
        )
    )
    payload_bottom = float(
        np.percentile(target_base[:, 2], floor_percentile)
    )
    table_plane = _payload_support_plane_z(args)
    support_tolerance = max(
        float(
            getattr(
                args,
                "motion_support_target_bottom_tolerance",
                0.020,
            )
        ),
        0.0,
    )
    raw_clearance = payload_bottom - table_plane
    clearance = raw_clearance + support_tolerance
    print(
        f"[{label}][payload-precheck] bottom_p{floor_percentile:g}="
        f"{payload_bottom:.4f}m table={table_plane:.4f}m support_tol="
        f"{support_tolerance * 1000.0:.1f}mm raw="
        f"{raw_clearance * 1000.0:.1f}mm margin="
        f"{clearance * 1000.0:.1f}mm"
    )
    if clearance < 0.0:
        raise UnsafeMotionTarget(
            f"{label} attached_target gets within "
            f"{clearance * 1000.0:.1f}mm of table_plane at sample 0; "
            "the robust target floor exceeds the bounded pickup-support "
            "baseline and is independent of pregrasp and IK branch"
        )
    if raw_clearance < 0.0:
        print(
            f"[{label}][payload-precheck][support-baseline] accept initial "
            f"table/mat contact={-raw_clearance * 1000.0:.1f}mm; pickup "
            "monitoring requires this overlap to decrease and forbids "
            "additional downward penetration"
        )
    return raw_clearance


def _payload_support_plane_z(args):
    """Return the physical payload support plane, not the TCP guard plane.

    ``table_tcp_clearance`` and ``table_tool_bottom_offset`` protect robot/tool
    geometry.  An object resting on the calibrated table must instead be
    measured against the table itself plus only the optional support-plane
    calibration offset.
    """
    return float(getattr(args, "table_z", 0.0)) + float(
        getattr(args, "motion_payload_support_plane_clearance", 0.0)
    )

def _build_initial_support_release(
    scene,
    grasp_tcp,
    args,
    *,
    table_guard_enabled_hook=None,
    bounded_motion_points_hook=None,
):
    """Identify a verified local support patch immediately below the target.

    Prefer the calibrated table/mat plane.  Objects inside a basket can rest
    well above ``table_z``; in that case conservatively infer a small local
    horizontal support patch only inside the target's robust XY footprint.
    Everything outside that patch remains a strict obstacle.
    """
    table_guard_is_enabled = (
        table_guard_enabled
        if table_guard_enabled_hook is None
        else table_guard_enabled_hook
    )
    bounded_motion_points = (
        _bounded_motion_points
        if bounded_motion_points_hook is None
        else bounded_motion_points_hook
    )
    if bool(getattr(args, "direct_pickup_contact_override", False)):
        return None
    if not table_guard_is_enabled(args):
        return None
    target = np.asarray(scene.target_base, dtype=np.float32).reshape(-1, 3)
    environment = np.asarray(
        scene.environment_base, dtype=np.float32
    ).reshape(-1, 3)
    if target.shape[0] == 0 or environment.shape[0] == 0:
        return None

    table_z = _payload_support_plane_z(args)
    plane_band = max(
        float(getattr(args, "motion_support_plane_band", 0.004)), 0.001
    )
    bottom_tolerance = max(
        float(getattr(args, "motion_support_target_bottom_tolerance", 0.020)),
        plane_band,
    )
    target_bottom = float(np.percentile(target[:, 2], 5.0))
    xy_margin = min(
        max(float(getattr(args, "motion_support_xy_margin", 0.003)), 0.0),
        0.003,
    )
    q_low, q_high = np.percentile(target[:, :2], [1.0, 99.0], axis=0)
    robust_target_xy = target[
        np.all(target[:, :2] >= q_low.reshape(1, 2), axis=1)
        & np.all(target[:, :2] <= q_high.reshape(1, 2), axis=1),
        :2,
    ]
    if robust_target_xy.shape[0] < 3:
        return None
    hull = cv2.convexHull(
        robust_target_xy.astype(np.float32), returnPoints=True
    ).reshape(-1, 2)
    if hull.shape[0] < 3:
        return None
    edges = np.roll(hull, -1, axis=0) - hull
    edge_lengths = np.linalg.norm(edges, axis=1)
    valid_edges = edge_lengths > 1e-6
    hull = hull[valid_edges]
    edges = edges[valid_edges]
    edge_lengths = edge_lengths[valid_edges]
    if hull.shape[0] < 3:
        return None
    centroid = np.mean(hull, axis=0)
    centroid_cross = (
        edges[:, 0] * (centroid[1] - hull[:, 1])
        - edges[:, 1] * (centroid[0] - hull[:, 0])
    )
    orientation = 1.0 if float(np.median(centroid_cross)) >= 0.0 else -1.0
    candidate_cross = (
        edges[None, :, 0]
        * (environment[:, None, 1] - hull[None, :, 1])
        - edges[None, :, 1]
        * (environment[:, None, 0] - hull[None, :, 0])
    )
    signed_edge_distance = (
        orientation
        * candidate_cross
        / edge_lengths.reshape(1, -1)
    )
    within_dilated_hull = np.all(
        signed_edge_distance >= -xy_margin, axis=1
    )
    calibrated_support = abs(target_bottom - table_z) <= bottom_tolerance
    local_support_enabled = bool(
        getattr(args, "motion_local_support_enable", True)
    )
    if not calibrated_support and not local_support_enabled:
        print(
            f"[motion][support] no table support exception: target z5="
            f"{target_bottom:.4f}m differs from table_z={table_z:.4f}m"
        )
        return None

    max_support_distance = (
        float(getattr(args, "motion_attached_object_clearance", 0.012))
        + plane_band
        if calibrated_support
        else max(
            float(
                getattr(
                    args,
                    "motion_local_support_contact_distance",
                    0.008,
                )
            ),
            plane_band,
        )
    )
    nearest_target_distance, _ = cKDTree(target).query(
        environment, k=1, workers=-1
    )
    nearest_target_distance = np.asarray(nearest_target_distance)
    support_plane_z = table_z
    support_source = "calibrated_table"
    support_plane_band = plane_band
    if not calibrated_support:
        max_gap = max(
            float(getattr(args, "motion_local_support_max_gap", 0.025)),
            plane_band,
        )
        local_candidates = environment[
            within_dilated_hull
            & (nearest_target_distance <= max_support_distance)
            & (environment[:, 2] >= target_bottom - max_gap)
            & (environment[:, 2] <= target_bottom + 0.002)
        ]
        min_points = max(
            int(getattr(args, "motion_support_min_points", 8)), 3
        )
        if local_candidates.shape[0] < min_points:
            print(
                f"[motion][support] no verified local support: only "
                f"{local_candidates.shape[0]}/{min_points} contact points; "
                f"target z5={target_bottom:.4f}m table_z={table_z:.4f}m"
            )
            return None
        support_plane_z = float(np.median(local_candidates[:, 2]))
        local_mad = float(
            np.median(np.abs(local_candidates[:, 2] - support_plane_z))
        )
        local_max_mad = max(
            float(
                getattr(args, "motion_local_support_plane_max_mad", 0.004)
            ),
            0.0005,
        )
        vertical_gap = target_bottom - support_plane_z
        if (
            local_mad > local_max_mad
            or vertical_gap < -0.002
            or vertical_gap > max_gap
        ):
            print(
                f"[motion][support] no verified local support: plane="
                f"{support_plane_z:.4f}m MAD={local_mad * 1000.0:.1f}mm "
                f"gap={vertical_gap * 1000.0:.1f}mm"
            )
            return None
        support_source = "observed_local_plane"
        support_plane_band = max(
            plane_band,
            min(2.5 * local_mad, local_max_mad),
        )

    support_mask = (
        (np.abs(environment[:, 2] - support_plane_z) <= support_plane_band)
        & within_dilated_hull
        & (nearest_target_distance <= max_support_distance)
        & (environment[:, 2] <= target_bottom + 0.002)
    )
    support = environment[support_mask]
    min_points = max(
        int(getattr(args, "motion_support_min_points", 8)), 3
    )
    if support.shape[0] < min_points:
        print(
            f"[motion][support] no table support exception: only "
            f"{support.shape[0]}/{min_points} local plane points"
        )
        return None
    median_z = float(np.median(support[:, 2]))
    mad_z = float(np.median(np.abs(support[:, 2] - median_z)))
    max_mad = max(
        float(getattr(args, "motion_support_plane_max_mad", 0.002)), 0.0005
    )
    allowed_plane_error = 0.010 if calibrated_support else support_plane_band
    allowed_mad = max_mad if calibrated_support else max(
        float(getattr(args, "motion_local_support_plane_max_mad", 0.004)),
        max_mad,
    )
    if abs(median_z - support_plane_z) > allowed_plane_error or mad_z > allowed_mad:
        print(
            f"[motion][support] no table support exception: plane median="
            f"{median_z:.4f}m MAD={mad_z * 1000.0:.1f}mm"
        )
        return None

    # The mask-owned environment can still contain a thin shell of selected
    # target points after RGB/depth edge registration and voxelisation.  A
    # closed gripper necessarily intersects the selected target, so treating
    # that shell as an external obstacle creates a large, pose-invariant
    # collision at support-release sample zero.  Remove only points that are
    # both very close to the selected-target cloud and inside its robust
    # footprint/height envelope.  Nearby basket walls outside that envelope
    # remain strict obstacles.
    target_contact_distance = max(
        float(
            getattr(
                args,
                "motion_support_target_contact_distance",
                0.008,
            )
        ),
        0.0,
    )
    target_contact_xy_margin = min(
        max(
            float(
                getattr(
                    args,
                    "motion_support_target_contact_xy_margin",
                    0.005,
                )
            ),
            0.0,
        ),
        0.010,
    )
    target_z_low, target_z_high = np.percentile(target[:, 2], [1.0, 99.0])
    target_contact_z_margin = min(
        max(
            float(
                getattr(
                    args,
                    "motion_support_target_contact_z_margin",
                    0.006,
                )
            ),
            0.0,
        ),
        0.012,
    )
    within_target_contact_hull = np.all(
        signed_edge_distance >= -target_contact_xy_margin, axis=1
    )
    target_contact_mask = (
        (~support_mask)
        & within_target_contact_hull
        & (nearest_target_distance <= target_contact_distance)
        & (environment[:, 2] >= target_z_low - target_contact_z_margin)
        & (environment[:, 2] <= target_z_high + target_contact_z_margin)
    )

    strict_environment = environment[~support_mask]
    tool_contact_environment = environment[
        ~(support_mask | target_contact_mask)
    ]
    target_contact_points = environment[target_contact_mask]
    if strict_environment.shape[0] == 0:
        raise PlanNotExecutable(
            "initial support classification removed the entire environment"
        )
    release = InitialSupportRelease(
        strict_environment_base=bounded_motion_points(
            strict_environment, args
        ),
        support_points_base=bounded_motion_points(support, args),
        start_tcp_pose=np.asarray(grasp_tcp, dtype=float).reshape(6).copy(),
        tool_contact_environment_base=bounded_motion_points(
            tool_contact_environment, args
        ),
        target_contact_points_base=bounded_motion_points(
            target_contact_points, args
        ),
        strict_after_rise=max(
            float(
                getattr(
                    args,
                    "motion_support_strict_rise",
                    0.018,
                )
                if calibrated_support
                else getattr(
                    args,
                    "motion_local_support_strict_rise",
                    0.040,
                )
            ),
            0.001,
        ),
        max_xy_drift=max(
            float(getattr(args, "motion_support_max_xy_drift", 0.005)), 0.001
        ),
        max_tilt_rad=np.deg2rad(
            max(float(getattr(args, "motion_support_max_tilt_deg", 3.0)), 0.1)
        ),
        downward_tolerance=max(
            float(getattr(args, "motion_support_downward_tolerance", 0.001)),
            0.0,
        ),
    )
    print(
        f"[motion][support] {support_source} support points="
        f"{release.support_points_base.shape[0]} strict_environment="
        f"{release.strict_environment_base.shape[0]} target_contact="
        f"{release.target_contact_points_base.shape[0]} tool_environment="
        f"{release.tool_contact_environment_base.shape[0]} "
        f"z_median={median_z:.4f}m "
        f"MAD={mad_z * 1000.0:.1f}mm strict_after_rise="
        f"{release.strict_after_rise * 1000.0:.1f}mm"
    )
    return release

def _motion_configuration_metrics(
    joint_angles,
    context,
    args,
    runtime=False,
    support_strict_latched=False,
    *,
    runtime_open_finger_start=False,
    motion_robot_capsules_hook=None,
    point_segment_distances_hook=None,
    tcp_transform_from_joints_hook=None,
    transform_points_hook=None,
    as_pose_list_hook=None,
    table_guard_enabled_hook=None,
):
    """Clearance at one measured/planned joint configuration."""
    motion_robot_capsules = (
        _motion_robot_capsules
        if motion_robot_capsules_hook is None
        else motion_robot_capsules_hook
    )
    point_segment_distances = (
        _point_segment_distances
        if point_segment_distances_hook is None
        else point_segment_distances_hook
    )
    tcp_transform_from_joints = (
        _tcp_transform_from_joints
        if tcp_transform_from_joints_hook is None
        else tcp_transform_from_joints_hook
    )
    transform_points = (
        _transform_points
        if transform_points_hook is None
        else transform_points_hook
    )
    as_pose_list = _as_pose_list if as_pose_list_hook is None else as_pose_list_hook
    table_guard_is_enabled = (
        table_guard_enabled
        if table_guard_enabled_hook is None
        else table_guard_enabled_hook
    )
    q = np.asarray(joint_angles, dtype=float).reshape(7)
    if not np.isfinite(q).all():
        raise UnsafeMotionTarget("motion collision guard received non-finite joints")
    scene = context.scene
    phase = str(context.phase)
    environment = np.asarray(scene.environment_base, dtype=np.float32).reshape(-1, 3)
    target = np.asarray(scene.target_base, dtype=np.float32).reshape(-1, 3)

    if phase == "free_open":
        robot_environment = np.asarray(scene.full_scene_base, dtype=np.float32).reshape(-1, 3)
        target_extra = np.zeros((0, 3), dtype=np.float32)
    else:
        robot_environment = environment
        target_extra = target if phase == "target_approach" else np.zeros((0, 3), dtype=np.float32)
    # During the verified near-vertical support-release only, omit the small
    # local patch that physically carries the target from both payload and
    # closed-tool environment checks.  The path validator latches strict mode
    # after ``strict_after_rise`` and all support points immediately become
    # ordinary obstacles again.
    if (
        phase == "support_release"
        and context.initial_support_release is not None
        and not bool(support_strict_latched)
    ):
        robot_environment = np.asarray(
            context.initial_support_release.strict_environment_base,
            dtype=np.float32,
        ).reshape(-1, 3)
    placed = (
        np.zeros((0, 3), dtype=np.float32)
        if context.placed_target_base is None
        else np.asarray(context.placed_target_base, dtype=np.float32).reshape(-1, 3)
    )

    plan_clearance = float(getattr(args, "motion_path_min_clearance", 0.015))
    runtime_clearance = float(
        getattr(args, "motion_runtime_stop_clearance", 0.008)
    )
    required_environment = runtime_clearance if runtime else plan_clearance
    required_target = float(
        getattr(args, "motion_target_non_tool_clearance", 0.005)
    )
    if runtime:
        required_target = min(required_target, runtime_clearance)

    best = {
        "margin": float("inf"),
        "nearest_clearance": float("inf"),
        "required_clearance": required_environment,
        "primitive": "none",
        "source": "none",
    }
    geometric_best = {
        "nearest_clearance": float("inf"),
        "required_clearance": None,
        "primitive": "none",
        "source": "none",
    }

    def consider_geometric(clearance, required, primitive, source):
        nonlocal geometric_best
        clearance = float(clearance)
        if clearance < float(geometric_best["nearest_clearance"]):
            geometric_best = {
                "nearest_clearance": clearance,
                "required_clearance": float(required),
                "primitive": str(primitive),
                "source": str(source),
            }

    def consider(points, a, b, radius, primitive, source, required):
        nonlocal best
        pts = np.asarray(points, dtype=np.float32).reshape(-1, 3)
        if pts.shape[0] == 0:
            return
        surface_distances = point_segment_distances(pts, a, b) - float(radius)
        nearest_index = int(np.argmin(surface_distances))
        clearance = float(surface_distances[nearest_index])
        consider_geometric(clearance, required, primitive, source)
        margin = clearance - float(required)
        if margin < float(best["margin"]):
            best = {
                "margin": float(margin),
                "nearest_clearance": clearance,
                "required_clearance": float(required),
                "primitive": str(primitive),
                "source": str(source),
                "nearest_point_base": np.asarray(
                    pts[nearest_index], dtype=float
                ).tolist(),
                "capsule_a_base": np.asarray(a, dtype=float).reshape(3).tolist(),
                "capsule_b_base": np.asarray(b, dtype=float).reshape(3).tolist(),
                "capsule_radius": float(radius),
            }

    def consider_constraint(
        margin, primitive, source="initial_support_constraint"
    ):
        nonlocal best
        if float(margin) < float(best["margin"]):
            best = {
                "margin": float(margin),
                "nearest_clearance": float(margin),
                "required_clearance": 0.0,
                "primitive": str(primitive),
                "source": str(source),
            }

    # The jaws remain physically open while moving from pregrasp to grasp;
    # closing is commanded only after the target_approach stage has reached its
    # endpoint.  Model that stage with the two real fingers instead of a filled
    # terminal capsule, otherwise valid target/neighbor points in the empty jaw
    # gap are incorrectly reported as gripper collisions.
    trial_clearance = getattr(args, "grasp_contact_check_clearance", None)
    cad_table_check = (
        phase == "target_approach"
        and getattr(args, "grasp_table_reference_z", None) is not None
    )
    if cad_table_check:
        from .graspgenx_integration import open_gripper_vertices_tcp, grasp_mesh_table_margin
        vertices = open_gripper_vertices_tcp(args)
        T_base_tcp = tcp_transform_from_joints(q, args)
        heights = vertices @ T_base_tcp[2, :3] + T_base_tcp[2, 3]
        lowest = int(np.argmin(heights))
        clearance = float(heights[lowest] - float(args.table_z))
        required = grasp_mesh_table_margin(args)
        if trial_clearance is None and (runtime or runtime_open_finger_start):
            slack = max(float(getattr(args, "motion_runtime_target_approach_finger_clearance_slack", .002)), 0.)
            required = max(min(required, runtime_clearance) - slack, 0.)
        consider_geometric(clearance, required, "gripper_cad_mesh", "table_plane")
        if clearance - required < float(best["margin"]):
            point = T_base_tcp[:3, :3] @ vertices[lowest] + T_base_tcp[:3, 3]
            best = dict(margin=clearance-required, nearest_clearance=clearance,
                        required_clearance=required, primitive="gripper_cad_mesh", source="table_plane",
                        nearest_point_base=[float(point[0]), float(point[1]), float(args.table_z)])

    robot_capsule_kwargs = {
        "open_gripper": phase
        in {
            "free_open",
            "target_approach",
            "released_retreat",
            "released_free",
        }
    }
    if getattr(context, "closed_gripper_width", None) is not None:
        robot_capsule_kwargs["closed_gripper_width"] = float(
            context.closed_gripper_width
        )
    robot_capsules = motion_robot_capsules(q, args, **robot_capsule_kwargs)
    for a, b, radius, primitive in robot_capsules:
        primitive_name = str(primitive)
        closed_finger = primitive_name.startswith("gripper_closed_finger_")
        ignore_target_approach_finger_environment = (
            phase == "target_approach"
            and str(primitive).startswith("gripper_finger_")
            and (
                bool(getattr(args, "direct_pickup_contact_override", False))
                or not bool(
                    getattr(
                        args,
                        "motion_target_approach_finger_environment_guard",
                        True,
                    )
                )
            )
        )
        ignore_closed_tool_environment = (
            phase in {"pickup_lift", "carrying"}
            and (primitive_name == "gripper_tool" or closed_finger)
            and bool(getattr(args, "direct_pickup_contact_override", False))
        )
        primitive_environment = robot_environment
        if (
            phase == "support_release"
            and context.initial_support_release is not None
            and not bool(support_strict_latched)
            and (primitive_name == "gripper_tool" or closed_finger)
            and context.initial_support_release.tool_contact_environment_base
            is not None
        ):
            primitive_environment = np.asarray(
                context.initial_support_release.tool_contact_environment_base,
                dtype=np.float32,
            ).reshape(-1, 3)
        if not (
            ignore_target_approach_finger_environment
            or ignore_closed_tool_environment
        ):
            primitive_required_environment = required_environment
            if (
                (runtime or runtime_open_finger_start)
                and primitive_name.startswith("gripper_finger_")
            ):
                if phase == "free_open":
                    finger_slack = float(getattr(
                        args, "motion_runtime_open_finger_clearance_slack", 0.001
                    ))
                elif phase == "target_approach":
                    finger_slack = float(getattr(
                        args, "motion_runtime_target_approach_finger_clearance_slack", 0.002
                    ))
                else:
                    finger_slack = 0.0
                primitive_required_environment = max(
                    min(required_environment, runtime_clearance)
                    - max(finger_slack, 0.0),
                    0.0,
                )
            trial_open_finger = (phase == "target_approach"
                                 and primitive_name.startswith("gripper_finger_")
                                 and trial_clearance is not None)
            if trial_open_finger:
                primitive_required_environment = max(float(trial_clearance), 0.0)
            if (
                closed_finger
                and phase in {"support_release", "pickup_lift", "carrying"}
            ):
                # A few millimetres of contact against a neighbouring object is
                # expected in clutter.  Express the allowance as a negative
                # required clearance only for the two closed fingers; palm,
                # camera, arm links and table guards remain unchanged.
                primitive_required_environment = -max(
                    float(
                        getattr(
                            args,
                            "motion_closed_finger_contact_tolerance",
                            0.010,
                        )
                    ),
                    0.0,
                )
            if cad_table_check and primitive_name.startswith("gripper_finger_"):
                # CAD vs the confirmed plane above owns the table check at
                # every sampled configuration. Do not also reject the same
                # surface using a padded capsule and noisy depth points.
                table_surface = np.abs(primitive_environment[:, 2] - float(args.table_z)) <= .005
                consider(primitive_environment[~table_surface], a, b, radius,
                         primitive, "environment", primitive_required_environment)
            else:
                consider(
                    primitive_environment,
                    a,
                    b,
                    radius,
                    primitive,
                    "environment",
                    primitive_required_environment,
                )
        target_contact_primitive = (
            primitive_name == "gripper_tool" or closed_finger
        ) or (
            phase == "target_approach"
            and str(primitive).startswith("gripper_finger_")
        )
        if target_extra.shape[0] > 0 and not (
            bool(context.allow_target_tool_contact) and target_contact_primitive
        ):
            target_required = required_target
            if phase == "target_approach" and str(primitive) == "gripper_body":
                # Only the final approach receives this narrow selected-target
                # allowance.  The same body still uses the normal environment
                # clearance above, and arm/camera primitives retain the normal
                # selected-target clearance.
                target_required = min(
                    target_required,
                    max(
                        float(
                            getattr(
                                args,
                                "motion_target_gripper_body_clearance",
                                0.002,
                            )
                        ),
                        0.0,
                    ),
                )
            consider(
                target_extra,
                a,
                b,
                radius,
                primitive,
                "selected_target",
                target_required,
            )
        placed_contact_primitive = (
            primitive_name == "gripper_tool" or closed_finger
        ) or (
            phase in {"released_retreat", "released_free"}
            and str(primitive).startswith("gripper_finger_")
        )
        # Release motion starts only after the hardware has opened the jaws.
        # Permit those two real fingers to remain adjacent to the just-placed
        # object at sample zero while still checking them against the frozen
        # non-target environment and checking every later released_free stage.
        if placed.shape[0] > 0 and not (
            bool(context.allow_target_tool_contact) and placed_contact_primitive
        ):
            consider(
                placed,
                a,
                b,
                radius,
                primitive,
                "placed_target",
                required_target,
            )

    if context.min_tcp_z is not None:
        current_tcp_z = float(tcp_transform_from_joints(q, args)[2, 3])
        consider_constraint(
            current_tcp_z - float(context.min_tcp_z),
            "stage_min_tcp_z",
            source="path_constraint",
        )

    support_rise = None
    if context.attached_target_tcp is not None:
        payload_tcp = np.asarray(context.attached_target_tcp, dtype=np.float32).reshape(-1, 3)
        T_base_tcp = tcp_transform_from_joints(q, args)
        payload_base = transform_points(T_base_tcp, payload_tcp)
        payload_environment = environment
        release = context.initial_support_release
        if release is not None:
            start_tcp = np.asarray(release.start_tcp_pose, dtype=float).reshape(6)
            current_tcp = np.asarray(as_pose_list(T_base_tcp), dtype=float).reshape(6)
            rise = float(current_tcp[2] - start_tcp[2])
            support_rise = rise
            xy_drift = float(
                np.linalg.norm(current_tcp[:2] - start_tcp[:2])
            )
            allowed_xy_drift = float(release.max_xy_drift)
            if runtime:
                allowed_xy_drift += max(
                    float(
                        getattr(
                            args,
                            "motion_runtime_support_xy_drift_slack",
                            0.002,
                        )
                    ),
                    0.0,
                )
            tilt = float(
                (
                    R.from_euler("xyz", current_tcp[3:])
                    * R.from_euler("xyz", start_tcp[3:]).inv()
                ).magnitude()
            )
            allowed_tilt = float(release.max_tilt_rad)
            if runtime:
                allowed_tilt += np.deg2rad(
                    max(
                        float(
                            getattr(
                                args,
                                "motion_runtime_support_tilt_slack_deg",
                                2.0,
                            )
                        ),
                        0.0,
                    )
                )
            consider_constraint(
                rise + float(release.downward_tolerance),
                "support_release_downward",
            )
            consider_constraint(
                allowed_xy_drift - xy_drift,
                "support_release_xy_drift",
            )
            # Convert the angular margin to an approximate 10 cm tool-tip
            # displacement so it shares the metric's metre units.
            consider_constraint(
                (allowed_tilt - tilt) * 0.10,
                "support_release_tilt",
            )
            if (
                not bool(support_strict_latched)
                and rise < float(release.strict_after_rise)
            ):
                payload_environment = np.asarray(
                    release.strict_environment_base, dtype=np.float32
                ).reshape(-1, 3)
        payload_robot_clearance = float(
            getattr(args, "motion_payload_robot_clearance", 0.005)
        )
        if runtime:
            payload_robot_clearance = min(
                payload_robot_clearance, runtime_clearance
            )
        for a, b, radius, primitive in robot_capsules:
            if primitive == "gripper_tool" or str(primitive).startswith(
                "gripper_closed_finger_"
            ):
                continue
            consider(
                payload_base,
                a,
                b,
                radius,
                primitive,
                "attached_target_vs_robot",
                payload_robot_clearance,
            )
        ignore_pickup_lift_payload_environment = (
            phase == "pickup_lift"
            and bool(getattr(args, "direct_pickup_contact_override", False))
        )
        if (
            payload_base.shape[0] > 0
            and payload_environment.shape[0] > 0
            and not ignore_pickup_lift_payload_environment
        ):
            distances, _ = cKDTree(payload_environment).query(
                payload_base, k=1, workers=-1
            )
            nearest = float(np.min(distances))
            required_payload = float(
                getattr(
                    args,
                    "motion_attached_object_clearance",
                    0.012 if not runtime else 0.006,
                )
            )
            if runtime:
                required_payload = min(required_payload, runtime_clearance)
            consider_geometric(
                nearest,
                required_payload,
                "attached_target",
                "environment",
            )
            margin = nearest - required_payload
            if margin < float(best["margin"]):
                best = {
                    "margin": float(margin),
                    "nearest_clearance": nearest,
                    "required_clearance": required_payload,
                    "primitive": "attached_target",
                    "source": "environment",
                }
        if table_guard_is_enabled(args) and payload_base.shape[0] > 0:
            # A fused RGB-D target can contain a small tail of flying points
            # below the physical support surface.  Using the absolute minimum
            # made one such point define the entire carried-object floor and
            # rejected every lift branch at sample zero.  A lower robust
            # percentile still rejects a genuinely low object body while
            # ignoring a sparse depth-noise tail.
            floor_percentile = float(
                np.clip(
                    getattr(args, "motion_payload_table_percentile", 10.0),
                    0.0,
                    25.0,
                )
            )
            payload_bottom = float(
                np.percentile(payload_base[:, 2], floor_percentile)
            )
            table_plane = _payload_support_plane_z(args)
            raw_payload_table_clearance = float(payload_bottom - table_plane)
            pickup_baseline = 0.0
            initial_support_tolerance = 0.0
            if phase in {"pickup_lift", "support_release"}:
                scene_target = np.asarray(
                    context.scene.target_base, dtype=np.float32
                ).reshape(-1, 3)
                if scene_target.shape[0] > 0:
                    start_bottom = float(
                        np.percentile(scene_target[:, 2], floor_percentile)
                    )
                    # Preserve only contact already present before lifting.
                    # A small downward tolerance covers controller/model
                    # quantisation, but any larger increase is still unsafe.
                    pickup_baseline = min(start_bottom - table_plane, 0.0)
                    initial_support_tolerance = max(
                        float(
                            getattr(
                                args,
                                "motion_support_downward_tolerance",
                                0.001,
                            )
                        ),
                        0.0,
                    )
            payload_table_clearance = float(
                raw_payload_table_clearance
                - pickup_baseline
                + initial_support_tolerance
            )
            if abs(payload_table_clearance) <= 1e-7:
                payload_table_clearance = 0.0
            consider_geometric(
                payload_table_clearance,
                0.0,
                "attached_target",
                "table_plane",
            )
            if payload_table_clearance < float(best["margin"]):
                best = {
                    "margin": payload_table_clearance,
                    "nearest_clearance": payload_table_clearance,
                    "required_clearance": 0.0,
                    "primitive": "attached_target",
                    "source": "table_plane",
                    "payload_bottom_percentile": floor_percentile,
                    "payload_bottom_z": payload_bottom,
                    "payload_raw_table_clearance": raw_payload_table_clearance,
                    "pickup_table_start_baseline": pickup_baseline,
                    "pickup_support_tolerance": initial_support_tolerance,
                }
    if support_rise is not None:
        best["support_rise"] = float(support_rise)
    best.update(
        {
            "geometric_nearest_clearance": float(
                geometric_best["nearest_clearance"]
            ),
            "geometric_required_clearance": geometric_best[
                "required_clearance"
            ],
            "geometric_primitive": str(geometric_best["primitive"]),
            "geometric_source": str(geometric_best["source"]),
        }
    )
    return best

def _motion_joint_path_metrics(
    q_start,
    q_target,
    context,
    args,
    *,
    allow_runtime_start=False,
    motion_configuration_metrics_hook=None,
):
    motion_configuration_metrics = (
        _motion_configuration_metrics
        if motion_configuration_metrics_hook is None
        else motion_configuration_metrics_hook
    )
    q0 = np.asarray(q_start, dtype=float).reshape(7)
    q1 = np.asarray(q_target, dtype=float).reshape(7)
    delta = q1 - q0
    sample_step = np.deg2rad(
        max(float(getattr(args, "motion_path_sample_joint_deg", 1.0)), 0.25)
    )
    sample_count = max(
        2, int(np.ceil(float(np.max(np.abs(delta))) / sample_step)) + 1
    )
    best = None
    best_index = -1
    support_strict_latched = bool(
        getattr(context, "support_release_strict_latched", False)
    )
    support_direction = str(getattr(context, "support_release_direction", "up"))
    if support_direction not in {"up", "return"}:
        raise UnsafeMotionTarget(f"invalid support-release direction: {support_direction!r}")
    previous_support_rise = None
    max_support_rise = -float("inf")
    min_support_rise = float("inf")
    final_support_rise = None
    for sample_index, alpha in enumerate(np.linspace(0.0, 1.0, sample_count)):
        planning_stats = getattr(args, "_planning_stats", None)
        if isinstance(planning_stats, dict):
            planning_stats["collision_configuration_evaluations"] = int(
                planning_stats.get("collision_configuration_evaluations", 0)
            ) + 1
        # Only the measured start may retain the finger tracking allowance.
        # Every future sample still has to pass the nominal planning margin.
        start_options = (
            {"runtime_open_finger_start": True}
            if allow_runtime_start and sample_index == 0
            and str(context.phase) == "target_approach"
            else {}
        )
        metrics = motion_configuration_metrics(
            q0 + float(alpha) * delta,
            context,
            args,
            runtime=False,
            support_strict_latched=support_strict_latched,
            **start_options,
        )
        if context.initial_support_release is not None:
            support_rise = float(metrics.get("support_rise", float("nan")))
            if not np.isfinite(support_rise):
                raise UnsafeMotionTarget(
                    "support-release path has no finite modeled TCP rise"
                )
            downward_tolerance = float(
                context.initial_support_release.downward_tolerance
            )
            if (
                previous_support_rise is not None
                and support_direction == "up"
                and support_rise
                < float(max_support_rise) - downward_tolerance
            ):
                drop = float(max_support_rise) - support_rise
                metrics = {
                    "margin": -drop,
                    "nearest_clearance": -drop,
                    "required_clearance": 0.0,
                    "primitive": "support_release_non_monotonic",
                    "source": "initial_support_constraint",
                    "support_rise": support_rise,
                }
            elif (
                previous_support_rise is not None
                and support_direction == "return"
                and support_rise > min_support_rise + downward_tolerance
            ):
                rise = support_rise - min_support_rise
                metrics = {
                    "margin": -rise,
                    "nearest_clearance": -rise,
                    "required_clearance": 0.0,
                    "primitive": "support_release_return_upward",
                    "source": "initial_support_constraint",
                    "support_rise": support_rise,
                }
            previous_support_rise = support_rise
            max_support_rise = max(max_support_rise, support_rise)
            min_support_rise = min(min_support_rise, support_rise)
            final_support_rise = support_rise
            if support_rise >= float(
                context.initial_support_release.strict_after_rise
            ):
                support_strict_latched = True
        if best is None or float(metrics["margin"]) < float(best["margin"]):
            best = metrics
            best_index = int(sample_index)
    out = dict(best)
    out.update(
        {
            "samples": int(sample_count),
            "sample_index": int(best_index),
            "max_joint_delta_deg": float(np.rad2deg(np.max(np.abs(delta)))),
            "joint_l2_deg": float(np.rad2deg(np.linalg.norm(delta))),
            "final_support_rise": final_support_rise,
            "support_strict_latched": bool(support_strict_latched),
        }
    )
    return out

def validate_motion_joint_path(
    q_start,
    q_target,
    context,
    args,
    label,
    *,
    allow_runtime_start=False,
    table_guard_enabled_hook=None,
    table_guard_min_tcp_z_hook=None,
    joint_path_tcp_z_profile_hook=None,
    motion_joint_path_metrics_hook=None,
):
    """Fail closed on joint sweep, table dip, robot, camera or payload contact."""
    table_guard_is_enabled = (
        table_guard_enabled
        if table_guard_enabled_hook is None
        else table_guard_enabled_hook
    )
    table_guard_min_z = (
        table_guard_min_tcp_z
        if table_guard_min_tcp_z_hook is None
        else table_guard_min_tcp_z_hook
    )
    joint_path_tcp_z_profile = (
        _joint_path_tcp_z_profile
        if joint_path_tcp_z_profile_hook is None
        else joint_path_tcp_z_profile_hook
    )
    motion_joint_path_metrics = (
        _motion_joint_path_metrics
        if motion_joint_path_metrics_hook is None
        else motion_joint_path_metrics_hook
    )
    q_start = np.asarray(q_start, dtype=float).reshape(7)
    q_target = np.asarray(q_target, dtype=float).reshape(7)
    delta = q_target - q_start
    max_delta_deg = float(np.rad2deg(np.max(np.abs(delta))))
    max_allowed = float(getattr(args, "motion_path_max_joint_delta_deg", 60.0))
    if max_delta_deg > max_allowed:
        raise UnsafeMotionTarget(
            f"{label} largest joint sweep {max_delta_deg:.2f}deg exceeds "
            f"motion limit {max_allowed:.2f}deg"
        )

    if table_guard_is_enabled(args) or context.min_tcp_z is not None:
        z_profile = joint_path_tcp_z_profile(q_start, q_target, args)
        min_z = (
            table_guard_min_z(args)
            if table_guard_is_enabled(args)
            else -float("inf")
        )
        if context.min_tcp_z is not None:
            min_z = max(min_z, float(context.min_tcp_z))
        if float(z_profile["min_z"]) < min_z:
            raise UnsafeMotionTarget(
                f"{label} sampled joint path TCP z={float(z_profile['min_z']):.4f}m "
                f"below required path min_tcp_z={min_z:.4f}m"
            )

    path_options = {"allow_runtime_start": True} if allow_runtime_start else {}
    metrics = motion_joint_path_metrics(q_start, q_target, context, args, **path_options)
    if (
        context.initial_support_release is not None
        and bool(
            getattr(
                context,
                "require_support_release_latch_at_endpoint",
                True,
            )
        )
    ):
        final_rise = metrics.get("final_support_rise", None)
        required_rise = float(
            context.initial_support_release.strict_after_rise
        )
        if (
            final_rise is None
            or not np.isfinite(float(final_rise))
            or float(final_rise) < required_rise
            or not bool(metrics.get("support_strict_latched", False))
        ):
            raise UnsafeMotionTarget(
                f"{label} support release ends at rise={final_rise}m before "
                f"strict payload checking latches at {required_rise:.4f}m"
            )
    print(
        f"[{label}][motion_path] phase={context.phase} "
        f"max_joint_delta={max_delta_deg:.2f}deg samples={metrics['samples']} "
        f"clearance={float(metrics['nearest_clearance']) * 1000.0:.1f}mm "
        f"required={float(metrics['required_clearance']) * 1000.0:.1f}mm "
        f"source={metrics['source']} primitive={metrics['primitive']} "
        f"sample={metrics['sample_index']}"
    )
    if not np.isfinite(float(metrics["margin"])):
        if bool(getattr(args, "motion_path_require_scene", True)):
            raise UnsafeMotionTarget(f"{label} motion-path clearance is unavailable")
    elif float(metrics["margin"]) < 0.0:
        if metrics["source"] == "initial_support_constraint":
            raise UnsafeMotionTarget(
                f"{label} violates {metrics['primitive']} at sample "
                f"{metrics['sample_index']}"
            )
        raise UnsafeMotionTarget(
            f"{label} {metrics['primitive']} gets within "
            f"{float(metrics['nearest_clearance']) * 1000.0:.1f}mm of "
            f"{metrics['source']} at sample {metrics['sample_index']}; required >= "
            f"{float(metrics['required_clearance']) * 1000.0:.1f}mm"
        )
    return metrics

def _active_robot_path_metrics(
    q_start,
    q_target,
    scene_points_base,
    args,
    *,
    active_robot_capsules_hook=None,
    point_segment_distances_hook=None,
):
    """Sample the commanded joint interpolation and measure full-arm clearance."""
    active_robot_capsules = (
        _active_robot_capsules
        if active_robot_capsules_hook is None
        else active_robot_capsules_hook
    )
    point_segment_distances = (
        _point_segment_distances
        if point_segment_distances_hook is None
        else point_segment_distances_hook
    )
    points = np.asarray(scene_points_base, dtype=np.float32).reshape(-1, 3)
    points = points[np.isfinite(points).all(axis=1)]
    voxel = float(getattr(args, "active_path_scene_voxel", 0.005))
    if voxel > 0.0 and points.shape[0] > 0:
        keys = np.floor(points / voxel).astype(np.int64)
        _, unique_indices = np.unique(keys, axis=0, return_index=True)
        points = points[np.sort(unique_indices)]
    cap = int(max(getattr(args, "active_path_obstacle_cap", 50000), 1))
    if points.shape[0] > cap:
        if bool(getattr(args, "motion_path_fail_on_obstacle_cap", True)):
            raise UnsafeMotionTarget(
                f"active safety cloud has {points.shape[0]} occupied voxels > "
                f"cap={cap}; refusing to discard sparse obstacles"
            )
        # Diagnostic override only.  Deterministic sampling keeps decisions
        # reproducible when explicitly requested.
        indices = np.linspace(0, points.shape[0] - 1, cap, dtype=np.int64)
        points = points[indices]
    if points.shape[0] == 0:
        return {
            "nearest_clearance": float("inf"),
            "near_count": 0,
            "samples": 0,
            "primitive": "none",
            "sample_index": -1,
        }

    q0 = np.asarray(q_start, dtype=float).reshape(7)
    # Nero's seven URDF joints are bounded revolute joints, not continuous.
    # move_j follows the absolute numerical target; wrapping across +/-pi would
    # check a different (and sometimes much shorter) path than the robot takes.
    delta = np.asarray(q_target, dtype=float).reshape(7) - q0
    sample_step = np.deg2rad(
        max(float(getattr(args, "active_path_sample_joint_deg", 2.0)), 0.5)
    )
    sample_count = max(2, int(np.ceil(float(np.max(np.abs(delta))) / sample_step)) + 1)
    safety_clearance = float(getattr(args, "active_path_min_clearance", 0.015))
    best_clearance = float("inf")
    best_count = 0
    best_primitive = "none"
    best_sample = -1

    for sample_index, alpha in enumerate(np.linspace(0.0, 1.0, sample_count)):
        q = q0 + float(alpha) * delta
        for a, b, radius, primitive_name in active_robot_capsules(q, args):
            clearance = point_segment_distances(points, a, b) - float(radius)
            local_min = float(np.min(clearance))
            if local_min < best_clearance:
                best_clearance = local_min
                best_count = int(np.count_nonzero(clearance < safety_clearance))
                best_primitive = str(primitive_name)
                best_sample = int(sample_index)
    return {
        "nearest_clearance": float(best_clearance),
        "near_count": int(best_count),
        "samples": int(sample_count),
        "primitive": best_primitive,
        "sample_index": int(best_sample),
    }

from .nbv_geometry import _active_motion_scene_points

def _validate_active_joint_path(
    q_start,
    q_target,
    scene_points_base,
    args,
    label,
    max_joint_delta_deg,
    max_joint_l2_deg,
    *,
    active_robot_path_metrics_hook=None,
):
    active_robot_path_metrics = (
        _active_robot_path_metrics
        if active_robot_path_metrics_hook is None
        else active_robot_path_metrics_hook
    )
    delta = (
        np.asarray(q_target, dtype=float).reshape(7)
        - np.asarray(q_start, dtype=float).reshape(7)
    )
    max_delta_deg = float(np.rad2deg(np.max(np.abs(delta))))
    l2_delta_deg = float(np.rad2deg(np.linalg.norm(delta)))
    if max_delta_deg > float(max_joint_delta_deg):
        raise UnsafeMotionTarget(
            f"{label} largest joint sweep {max_delta_deg:.2f}deg exceeds "
            f"active limit {float(max_joint_delta_deg):.2f}deg"
        )
    if l2_delta_deg > float(max_joint_l2_deg):
        raise UnsafeMotionTarget(
            f"{label} total joint sweep {l2_delta_deg:.2f}deg exceeds "
            f"active L2 limit {float(max_joint_l2_deg):.2f}deg"
        )
    metrics = active_robot_path_metrics(
        q_start,
        q_target,
        scene_points_base,
        args,
    )
    min_clearance = float(getattr(args, "active_path_min_clearance", 0.015))
    print(
        f"[{label}][path] max_joint_delta={max_delta_deg:.2f}deg "
        f"joint_l2={l2_delta_deg:.2f}deg "
        f"samples={int(metrics['samples'])} nearest_surface_clearance="
        f"{float(metrics['nearest_clearance']) * 1000.0:.1f}mm "
        f"primitive={metrics['primitive']} near_points={int(metrics['near_count'])}"
    )
    if not np.isfinite(float(metrics["nearest_clearance"])):
        if bool(getattr(args, "active_path_require_scene", True)):
            raise UnsafeMotionTarget(f"{label} full-arm path clearance could not be evaluated")
    elif float(metrics["nearest_clearance"]) < min_clearance:
        raise UnsafeMotionTarget(
            f"{label} full-arm path gets within "
            f"{float(metrics['nearest_clearance']) * 1000.0:.1f}mm of scene at "
            f"{metrics['primitive']}; required >= {min_clearance * 1000.0:.1f}mm"
        )
    return metrics
