"""Portable per-observation inputs, saved before any multiview registration.

No camera/robot/Open3D imports: the caller supplies the measured transform and
the single-view clouds.  Archive failures must never alter motion control.
"""

from datetime import datetime
import json
from pathlib import Path
import re
import shutil
import time
from uuid import uuid4

import numpy as np


_CAPTURE_FILES = (
    "color.png", "depth_m.npy", "capture_meta.json", "selected_target_mask.png",
)
_OPTIONAL_CAPTURE_FILES = (
    "depth_raw_m.npy", "depth_raw_frames.npz", "camera_self_mask.png",
    "camera_self_mask_overlay.png", "camera_self_mask_report.json",
    "depth_sdk_m.npy", "depth_raw_valid_count.npy",
    "depth_support_rejected.png", "depth_support_report.json",
)
_INSTANCE_SCALAR_FIELDS = (
    "id", "class_id", "label", "class_label", "score", "confidence",
    "observed_label", "requested_target_class", "is_box_only", "source",
    "visible_area_px", "clear_fallback_largest_visible", "clear_fallback_reason",
)
_INSTANCE_GEOMETRY_FIELDS = ("xyxy", "bbox", "box", "center", "center_camera", "center_base")
_PROCESSING_FIELDS = (
    "min_depth", "max_depth", "voxel_size", "scene_voxel_size",
    "target_outlier_filter", "outlier_neighbors", "outlier_std_ratio",
    "orbbec_depth_median_frames", "orbbec_depth_median_min_valid",
    "orbbec_depth_median_max_mad", "camera_self_mask", "camera_self_mask_expand_px",
    "camera_self_mask_calibration",
    "camera_self_edge_island_filter", "camera_near_field_ghost_filter",
    "camera_near_field_ghost_seed_depth", "camera_near_field_ghost_grow_depth",
)


def _array_description(value):
    return {"shape": list(value.shape), "dtype": str(value.dtype), "data_omitted": True}


def _json_value(value):
    if isinstance(value, np.ndarray):
        if value.size > 64:
            return _array_description(value)
        return _json_value(value.tolist())
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        if len(value) > 64:
            return {"length": len(value), "data_omitted": True}
        return [_json_value(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "__dict__"):
        return _json_value(vars(value))
    return str(value)


def _instance_metadata(instance):
    """Keep only bounded identification/geometry values, never image pixels."""
    if instance is None:
        return None
    if not isinstance(instance, dict):
        return {"type": type(instance).__name__}
    result = {}
    for name in _INSTANCE_SCALAR_FIELDS:
        value = instance.get(name)
        if value is None or isinstance(value, (str, int, float, bool, np.generic)):
            if name in instance:
                result[name] = _json_value(value)
    for name in _INSTANCE_GEOMETRY_FIELDS:
        if name not in instance:
            continue
        value = instance[name]
        if isinstance(value, np.ndarray) and value.size <= 16 and value.ndim <= 2:
            result[name] = _json_value(value)
        elif (isinstance(value, (tuple, list)) and len(value) <= 16
              and all(isinstance(item, (int, float, np.generic)) for item in value)):
            result[name] = _json_value(value)
    omitted = {}
    for name, value in instance.items():
        if isinstance(value, np.ndarray) and (value.ndim >= 2 or value.size > 16):
            omitted[str(name)] = _array_description(value)
            if "mask" in str(name).lower():
                omitted[str(name)]["mask_source_file"] = "selected_target_mask.png"
    if omitted:
        result["omitted_arrays"] = omitted
    return result


def _processing_metadata(args):
    result = {}
    for name in _PROCESSING_FIELDS:
        value = getattr(args, name, None)
        if value is not None and isinstance(value, (str, int, float, bool, np.generic)):
            result[name] = _json_value(value)
    return result


def _write_json(path, value):
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(_json_value(value), ensure_ascii=False, indent=2,
                       allow_nan=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _cloud_arrays(cloud, name):
    points = np.asarray(getattr(cloud, "points", cloud))
    colors = np.asarray(getattr(cloud, "colors", np.empty((0, 3))))
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"{name} points must be N x 3, got {points.shape}")
    if (not np.issubdtype(points.dtype, np.number)
            or np.issubdtype(points.dtype, np.complexfloating)):
        raise ValueError(f"{name} points must be numeric")
    if (colors.ndim != 2 or colors.shape[1] != 3
            or colors.shape[0] not in (0, points.shape[0])):
        raise ValueError(f"{name} colors must be N x 3 or empty, got {colors.shape}")
    if (not np.issubdtype(colors.dtype, np.number)
            or np.issubdtype(colors.dtype, np.complexfloating)):
        raise ValueError(f"{name} colors must be numeric")
    return points, colors


def _write_binary_ply(path, points, colors):
    has_colors = len(colors) == len(points) and len(colors) > 0
    fields = [(axis, "<f8") for axis in ("x", "y", "z")]
    if has_colors:
        fields += [(color, "u1") for color in ("red", "green", "blue")]
    vertices = np.empty(len(points), dtype=np.dtype(fields))
    for axis, values in zip(("x", "y", "z"), points.T):
        vertices[axis] = values
    if has_colors:
        # PLY uses byte RGB; clouds.npz preserves the exact input color values.
        rgb = np.clip(np.rint(np.nan_to_num(colors) * 255.0), 0, 255).astype(np.uint8)
        for color, values in zip(("red", "green", "blue"), rgb.T):
            vertices[color] = values
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        "comment unfused single-view input; points in metres\n"
        f"element vertex {len(points)}\n"
        "property double x\nproperty double y\nproperty double z\n"
    )
    if has_colors:
        header += "property uchar red\nproperty uchar green\nproperty uchar blue\n"
    with path.open("wb") as stream:
        stream.write((header + "end_header\n").encode("ascii"))
        vertices.tofile(stream)


def save_unfused_view(
    out_dir, view_label, target_cloud, scene_cloud, T_base_camera, *,
    obstacle_cloud=None, capture_dir=None, capture_joints=None,
    intrinsics=None, instance=None, args=None,
    pose_timing="measured_pre_capture", pose_read_monotonic=None,
):
    """Save a view independently of fusion gates; return a handle for status updates.

    ``*_camera`` files retain the supplied observation frame.  ``*_base`` files
    apply only ``T_base_camera``.  These clouds have already undergone the
    single-view perception pipeline; they are not untouched sensor samples.
    """
    if not bool(getattr(args, "save_active_raw_views", True)):
        return None
    view_dir = None
    metadata = None
    try:
        created = datetime.now().astimezone()
        safe_label = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(view_label)).strip("._")
        safe_label = safe_label[:80] or "view"
        view_id = f"{safe_label}_{created:%Y-%m-%d_%H-%M-%S_%f}_{uuid4().hex[:8]}"
        archive_root = Path(out_dir).resolve() / "multiview_raw"
        view_dir = archive_root / view_id
        view_dir.mkdir(parents=True, exist_ok=False)
        metadata = {
            "schema_version": 1,
            "view_id": view_id,
            "view_label": str(view_label),
            "created_at": created.isoformat(timespec="microseconds"),
            "archived_monotonic": time.monotonic(),
            "archive_status": "writing",
            "units": {"points": "metre", "capture_joints": "radian"},
            "frames": {"camera": "native_color_camera", "base": "base_link"},
            "transform_convention": "P_base = R_base_camera @ P_camera + t_base_camera",
            "registration_applied": False,
            "fusion_applied": False,
            "pointcloud_processing": (
                "Supplied single-view perception clouds, before multiview "
                "registration/fusion; single-view depth refinement, filtering "
                "and voxel sampling may already have been applied."
            ),
            "capture_depth_processing": (
                "depth_m.npy, when available, is SDK depth aligned to color "
                "and may be a temporal median checked against the same raw "
                "window; see capture_meta.json and depth_support_report.json."
            ),
            "pose_timing": str(pose_timing),
            "pose_synchronized_to_rgbd": False,
            "pose_read_monotonic": pose_read_monotonic,
            "capture_joints": capture_joints,
            "intrinsics": intrinsics,
            "instance": _instance_metadata(instance),
            "single_view_processing_settings": _processing_metadata(args),
            "capture_source_dir": None if capture_dir is None else str(capture_dir),
            "clouds": {},
            "arrays_file": "clouds.npz",
            "capture_files": {},
            "missing_capture_files": [],
            "status": {},
        }
        transform = np.asarray(T_base_camera, dtype=float)
        if transform.shape != (4, 4) or not np.isfinite(transform).all():
            raise ValueError("T_base_camera must be a finite 4 x 4 matrix")
        if not np.allclose(transform[3], [0, 0, 0, 1], atol=1e-8):
            raise ValueError("T_base_camera must be a homogeneous transform")
        rotation = transform[:3, :3]
        if (not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5)
                or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-5)):
            raise ValueError("T_base_camera rotation must be a proper rigid rotation")
        metadata["T_base_camera"] = transform.tolist()
        clouds = {"target": target_cloud, "scene": scene_cloud}
        if obstacle_cloud is not None:
            clouds["obstacle"] = obstacle_cloud
        # Validate all inputs before producing any apparently complete clouds.
        validated = {name: _cloud_arrays(cloud, name) for name, cloud in clouds.items()}
        arrays = {}
        for name, (points, colors) in validated.items():
            base = points @ rotation.T + transform[:3, 3]
            camera_name, base_name = f"{name}_camera.ply", f"{name}_base.ply"
            _write_binary_ply(view_dir / camera_name, points, colors)
            _write_binary_ply(view_dir / base_name, base, colors)
            arrays[f"{name}_camera_points"] = points
            arrays[f"{name}_base_points"] = base
            arrays[f"{name}_camera_colors"] = colors
            arrays[f"{name}_base_colors"] = colors
            metadata["clouds"][name] = {
                "points": int(len(points)), "colors": int(len(colors)),
                "camera_ply": camera_name, "base_ply": base_name,
            }
        np.savez(view_dir / "clouds.npz", **arrays)
        for name in (*_CAPTURE_FILES, *_OPTIONAL_CAPTURE_FILES):
            source = None if capture_dir is None else Path(capture_dir) / name
            if source is not None and source.is_file():
                shutil.copyfile(source, view_dir / name)
                metadata["capture_files"][name] = name
            elif name in _CAPTURE_FILES:
                metadata["missing_capture_files"].append(name)
        calibration_path = getattr(args, "calib_json", None)
        metadata["handeye"] = {
            "source_file": None if calibration_path is None else str(calibration_path),
            "convention": getattr(args, "handeye_convention", None),
        }
        if calibration_path is not None:
            try:
                snapshot = json.loads(Path(calibration_path).read_text(encoding="utf-8"))
                metadata["handeye"]["snapshot"] = snapshot
            except (OSError, ValueError) as exc:
                metadata["handeye"]["snapshot_error"] = f"{type(exc).__name__}: {exc}"
        metadata["archive_status"] = "complete"
        metadata_path = view_dir / "metadata.json"
        _write_json(metadata_path, metadata)
        entry = {
            "schema_version": 1, "view_id": view_id, "view_label": str(view_label),
            "created_at": metadata["created_at"], "view_dir": view_id,
            "metadata_path": f"{view_id}/metadata.json",
            "arrays_path": f"{view_id}/clouds.npz",
        }
        with (archive_root / "manifest.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(entry, ensure_ascii=False, allow_nan=False) + "\n")
        print(f"[active][archive] {view_label} unfused inputs saved: {view_dir}")
        return {"archive_root": str(archive_root), **entry}
    except Exception as exc:
        if view_dir is not None and metadata is not None:
            metadata["archive_status"] = "failed"
            metadata["archive_error"] = f"{type(exc).__name__}: {exc}"
            try:
                _write_json(view_dir / "metadata.json", metadata)
            except Exception:
                pass
        print(f"[active][archive][warn] {view_label} was not archived: {type(exc).__name__}: {exc}")
        return None


def update_unfused_view(record, **status_fields):
    """Record gate outcomes without deleting or changing archived input arrays."""
    if record is None:
        return
    try:
        path = Path(record["archive_root"]) / record["metadata_path"]
        metadata = json.loads(path.read_text(encoding="utf-8"))
        metadata.setdefault("status", {}).update(_json_value(status_fields))
        metadata["status_updated_at"] = datetime.now().astimezone().isoformat(timespec="microseconds")
        _write_json(path, metadata)
    except Exception as exc:
        print(f"[active][archive][warn] view status could not be saved: {type(exc).__name__}: {exc}")
