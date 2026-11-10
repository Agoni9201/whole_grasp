"""Register a centered grasp scene to a caller-confirmed physical table plane.

This is opt-in observation registration. Robot FK, hand-eye files and collision
clearances stay independent of the correction. No points are removed as table.
"""

import json
from dataclasses import fields
from pathlib import Path

import numpy as np


def fit_table_reference(points, center_xy, reference_z, *, min_points=500,
                        max_tilt_deg=6., max_height_shift=.030):
    import open3d as o3d
    points = np.asarray(points, float).reshape(-1, 3)
    center_xy = np.asarray(center_xy, float).reshape(2)
    reference_z = float(reference_z)
    if not np.isfinite(reference_z) or not np.isfinite(center_xy).all():
        raise ValueError("Physical table reference must be finite")
    roi = (np.isfinite(points).all(axis=1)
           & (np.linalg.norm(points[:, :2] - center_xy, axis=1) < .30)
           & (np.abs(points[:, 2] - reference_z) < .10))
    local = points[roi]
    if len(local) < min_points:
        raise ValueError("Insufficient measured table points for physical reference")
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(local))
    o3d.utility.random.seed(0)
    plane, indices = cloud.segment_plane(.0025, 3, 500)
    n, d = np.array(plane[:3]), float(plane[3])
    if n[2] < 0:
        n, d = -n, -d
    tilt = float(np.rad2deg(np.arccos(np.clip(n[2], -1, 1))))
    if tilt > max_tilt_deg or n[2] <= 0:
        raise ValueError(f"Table normal tilt {tilt:.2f}deg exceeds {max_tilt_deg:.1f}deg")
    if len(indices) < min_points or len(indices) / len(local) < .60:
        raise ValueError("Dominant plane has insufficient table support")
    inliers = local[indices]
    if np.linalg.eigvalsh(np.cov(inliers[:, :2], rowvar=False)).min() < .001:
        raise ValueError("Table support is too narrow to determine a stable plane")
    height = float(-(n[:2] @ center_xy + d) / n[2])
    if abs(height - reference_z) > max_height_shift:
        raise ValueError(f"Table height correction {(reference_z-height)*1000:.1f}mm exceeds envelope")
    axis = np.cross(n, [0., 0., 1.])
    skew = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]],
                     [-axis[1], axis[0], 0]])
    rotation = np.eye(3) + skew + skew @ skew / (1. + n[2])
    source_pivot = np.r_[center_xy, height]
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = np.r_[center_xy, reference_z] - rotation @ source_pivot
    report = dict(plane=[*n.tolist(), d], roi_points=len(local), inliers=len(indices),
                  inlier_ratio=len(indices) / len(local),
                  plane_fit_rms_m=float(np.sqrt(np.mean((inliers @ n + d) ** 2))),
                  max_correction_tilt_deg=float(max_tilt_deg),
                  max_correction_height_shift_m=float(max_height_shift),
                  tilt_deg=tilt, height_at_target_m=height, physical_reference_z_m=reference_z,
                  T_registration=transform.tolist(), points_deleted_by_registration=0)
    return transform, report


def build_table_reference(args, capture_dir, T_base_camera, output_dir):
    """Numerical worker entry: preserve raw archives and register ALL safety views."""
    import open3d as o3d
    from .pointcloud_fusion import fuse_dynamic_environment_clouds
    if not bool(getattr(args, "active_view", False)):
        raise ValueError("Physical table reference requires target-centered active observations")
    reference = float(args.grasp_table_reference_z)
    if not np.isclose(float(args.table_z), reference, atol=1e-6):
        raise ValueError("Confirmed physical reference must match the configured table guard height")
    capture_dir, output_dir = Path(capture_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    anchors, sides = {}, []
    for path in sorted((capture_dir / "multiview_raw").glob("*/metadata.json")):
        meta = json.loads(path.read_text())
        name = meta["view_label"]
        if name in {"initial", "center"}:
            if name in anchors:
                raise ValueError(f"Ambiguous archived view: {name}")
            anchors[name] = path.parent
        elif name.startswith("side"):
            # A rejected side capture may be retried with the same label.
            # Keep every capture's obstacles, including rejected target views.
            sides.append((name, path.parent))
        else:
            raise ValueError(f"Unsupported centered observation label: {name}")
    if set(anchors) != {"initial", "center"}:
        raise ValueError("Physical table reference needs initial and centered raw archives")
    if int(args.active_side_view_count) > 0 and not sides:
        raise ValueError("Physical multiview table reference needs archived side observations")
    views = [(name, anchors[name]) for name in ("initial", "center")]
    views.extend((f"{name}_capture_{index}", folder)
                 for index, (name, folder) in enumerate(sides, start=1))
    scenes, obstacles, targets, corrections, reports = [], [], [], {}, []
    for name, folder in views:
        clouds = [o3d.io.read_point_cloud(str(folder / f"{kind}_base.ply"))
                  for kind in ("scene", "obstacle", "target")]
        if any(not len(cloud.points) for cloud in clouds):
            raise ValueError(f"Missing/nonempty raw clouds required for view {name}")
        target = np.asarray(clouds[2].points).copy()
        before = float(np.median(target[:, 2]))
        side_view = name.startswith("side")
        anchor = corrections["center"] if side_view else np.eye(4)
        for cloud in clouds:
            cloud.transform(anchor)
        fit_options = (
            dict(max_tilt_deg=float(getattr(args, "active_table_reg_max_tilt_deg", 5.)),
                 max_height_shift=float(getattr(args, "active_table_reg_max_height_shift", .015)))
            if side_view else {}
        )
        correction, report = fit_table_reference(
            clouds[1].points, np.median(np.asarray(clouds[2].points), axis=0)[:2],
            reference, **fit_options)
        for cloud in clouds:
            cloud.transform(correction)
        # The center establishes the absolute table frame. Side observations
        # are bounded by their remaining error in that frame, not by applying
        # the absolute center budget to the sum of both corrections.
        residual_plane = np.asarray(report["plane"], dtype=float)
        raw_normal = anchor[:3, :3].T @ residual_plane[:3]
        raw_d = float(residual_plane[3] + residual_plane[:3] @ anchor[:3, 3])
        report.update(
            center_anchor_applied=side_view,
            center_anchor_transform=anchor.tolist(),
            residual_tilt_before_deg=report["tilt_deg"],
            residual_height_before_m=report["height_at_target_m"],
            plane=[*raw_normal.tolist(), raw_d],
            tilt_deg=float(np.rad2deg(np.arccos(np.clip(raw_normal[2], -1, 1)))),
            height_at_target_m=float(-(raw_normal[:2] @ np.median(target, axis=0)[:2] + raw_d) / raw_normal[2]),
            T_registration=(correction @ anchor).tolist(),
        )
        scenes.append(clouds[0]); obstacles.append(clouds[1]); targets.append(np.asarray(clouds[2].points))
        corrections[name] = correction @ anchor
        report.update(view=name, source=str(folder), plane_fit_source="non_target_obstacle_cloud",
                      target_median_z_before_m=before,
                      target_median_z_after_m=float(np.median(targets[-1][:, 2])))
        reports.append(report)
        print(f"[table-reference] {name}: measured_table={report['height_at_target_m']*1000:.1f}mm "
              f"reference={reference*1000:.1f}mm tilt_before={report['tilt_deg']:.2f}deg correction=applied; "
              f"center_anchor={side_view} residual_height={report['residual_height_before_m']*1000:.1f}mm "
              f"residual_tilt={report['residual_tilt_before_deg']:.2f}deg; "
              f"fit_support={report['inlier_ratio']:.3f} rms={report['plane_fit_rms_m']*1000:.2f}mm; "
              "target and full obstacle cloud registered together")
    registered_camera = corrections["center"] @ np.asarray(T_base_camera, float)
    environment = fuse_dynamic_environment_clouds(
        [name for name, _ in views], obstacles, args, target_base=np.vstack(targets),
        target_center_base=np.median(targets[0], axis=0),
        map_id="physical_table_reference", generation=2)
    world_scene = o3d.geometry.PointCloud()
    world_obstacles = o3d.geometry.PointCloud()
    for scene, obstacle in zip(scenes, obstacles):
        world_scene += scene
        world_obstacles += obstacle
    world_scene.transform(np.linalg.inv(registered_camera))
    world_obstacles.transform(np.linalg.inv(registered_camera))
    arrays = dict(T_base_camera=registered_camera,
                  scene_points=np.asarray(world_scene.points), scene_colors=np.asarray(world_scene.colors),
                  environment_points=np.asarray(world_obstacles.points),
                  environment_colors=np.asarray(world_obstacles.colors))
    map_metadata = {}
    for field in fields(environment):
        value = getattr(environment, field.name)
        if isinstance(value, np.ndarray):
            arrays["map_" + field.name] = value
        else:
            map_metadata[field.name] = value
    np.savez_compressed(output_dir / "registered_scene.npz", **arrays)
    report = dict(reference_is_caller_confirmed=True, physical_table_z_m=reference,
                  scope="center-aligned single/multiview target; all captured obstacle views registered",
                  target_fusion_preserved=True,
                  handeye_file_changed=False, collision_clearance_changed=False,
                  original_camera_transform=np.asarray(T_base_camera).tolist(),
                  registered_virtual_camera_transform=registered_camera.tolist(),
                  views=reports, environment_map_metadata=map_metadata)
    report_path = output_dir / "table_reference_report.json"
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return str(report_path)


def prepare_table_referenced_scene(args, target, scene, T_base_camera, out_dir):
    from .planning_process import checked_request, config_snapshot
    from .types import DynamicEnvironmentMap, PlanNotExecutable
    from .visualization import get_o3d
    output = Path(out_dir) / "table_reference"
    output.mkdir(parents=True, exist_ok=True)
    result = checked_request("physical_table_reference", dict(
        config=config_snapshot(args), capture_dir=str(out_dir),
        T_base_camera=np.asarray(T_base_camera, float), output_dir=str(output)), args)
    if not result["ok"]:
        raise PlanNotExecutable(result["reason"])
    report = json.loads(Path(result["report"]).read_text())
    o3d = get_o3d()
    with np.load(output / "registered_scene.npz", allow_pickle=False) as data:
        def cloud(key):
            value = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(data[key + "_points"]))
            if len(data[key + "_colors"]):
                value.colors = o3d.utility.Vector3dVector(data[key + "_colors"])
            return value
        motion_scene, motion_environment = cloud("scene"), cloud("environment")
        metadata = report["environment_map_metadata"]
        metadata.update({key[4:]: data[key] for key in data.files if key.startswith("map_")})
        environment_map = DynamicEnvironmentMap(**metadata)
        registered_camera = data["T_base_camera"].copy()
    args._grasp_table_reference_report = result["report"]
    state = getattr(args, "_experiment_log_state", None)
    if state is not None:
        state["artifacts"]["table_reference_report"] = result["report"]
    # Preserve the already center-aligned target fusion (including its view
    # rejection and filtering). Apply the center's absolute correction once via
    # its virtual camera mapping; every raw safety view is corrected separately.
    return target, scene, registered_camera, motion_scene, motion_environment, environment_map
