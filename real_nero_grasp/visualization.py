"""Lazy visualization and GraspNet helper loading for the real Nero runner."""

import os
import sys
import types

import numpy as np


__all__ = [
    "get_o3d",
    "draw_open3d_preview",
    "get_graspnet_helpers",
    "release_vision_model_caches",
]


def get_o3d():
    # Open3D 0.18 imports its optional ML package unconditionally.  This grasp
    # runner only uses geometry/io, and the deployed environment currently has
    # an sklearn/NumPy ABI mismatch inside open3d.ml.  Pre-registering the
    # unused optional namespace keeps core point-cloud functionality available
    # without mutating the conda environment.
    if (
        "open3d" not in sys.modules
        and os.environ.get("NERO_ENABLE_OPEN3D_ML", "0") != "1"
    ):
        sys.modules.setdefault("open3d.ml", types.ModuleType("open3d.ml"))
    try:
        import open3d as o3d
    except ImportError as exc:
        raise ImportError(
            "Open3D is required for point-cloud filtering and GraspNet input. "
            "Activate the project environment that has open3d installed."
        ) from exc
    return o3d


def draw_open3d_preview(
    geometries,
    window_name,
    args,
    width=1280,
    height=720,
    *,
    get_o3d_hook=None,
):
    """Show geometry in a separate interpreter while monitoring the MIT owner."""
    from .preview_process import show_preview
    from .planning_process import planning_health_check
    return show_preview(geometries, window_name, args.preview_point_size, width, height,
                        health_check=lambda: planning_health_check(args, "Open3D preview"))


def get_graspnet_helpers():
    from grasp_active_three_views_sceneaware_asymmetric_clear_nbv_obstacle_yolo26seg import (
        make_end_points_from_o3d_cloud,
        generate_grasps,
    )
    return make_end_points_from_o3d_cloud, generate_grasps


def release_vision_model_caches(*, collect_garbage=True):
    try:
        from grasp_active_three_views_sceneaware_asymmetric_clear_nbv_obstacle_yolo26seg import (
            release_vision_model_caches as release_impl,
        )
        release_impl(collect_garbage=collect_garbage)
        return
    except Exception as exc:
        print(
            f"[cache][warn] project cache release unavailable: "
            f"{type(exc).__name__}: {exc}"
        )
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception as exc:
        print(
            f"[cache][warn] torch cuda cache release failed: "
            f"{type(exc).__name__}: {exc}"
        )
