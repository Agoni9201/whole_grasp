#!/usr/bin/env python3
"""Interactively view fused point clouds saved by the real grasp runner."""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    import open3d as o3d
except ImportError as exc:
    raise SystemExit(
        "Open3D is unavailable. Run this through ./view_fused_pointcloud.sh "
        "or activate the grasp_yolo26 conda environment."
    ) from exc


PROJECT_DIR = Path(__file__).resolve().parent.parent
LOG_ROOT = PROJECT_DIR / "logs"
TARGET_NAME = "active_fused_target_camera0.ply"
SCENE_NAME = "active_fused_scene_camera0.ply"
SCENE_FALLBACK_NAMES = ("motion_safety_environment_base.ply",)


def newest_fusion_dir() -> Path:
    matches = list(LOG_ROOT.rglob(TARGET_NAME))
    if not matches:
        raise SystemExit(f"No {TARGET_NAME} found below {LOG_ROOT}")
    return max(matches, key=lambda path: path.stat().st_mtime).parent


def resolve_input(value: str | None) -> tuple[Path | None, Path | None, Path]:
    path = newest_fusion_dir() if value is None else Path(value).expanduser().resolve()
    if path.is_file():
        if path.suffix.lower() != ".ply":
            raise SystemExit(f"Expected a .ply file, got: {path}")
        # An explicitly supplied PLY is authoritative even when its filename
        # does not follow the active_fused_{target,scene}_camera0 convention.
        # Put arbitrary clouds in the scene slot so the default "both" mode
        # loads the file exactly once.
        if "target" in path.name.lower():
            return path, None, path.parent
        return None, path, path.parent
    if not path.is_dir():
        raise SystemExit(f"Path does not exist: {path}")
    target_matches = list(path.rglob(TARGET_NAME))
    scene_matches = list(path.rglob(SCENE_NAME))
    target = max(target_matches, key=lambda p: p.stat().st_mtime) if target_matches else None
    scene = max(scene_matches, key=lambda p: p.stat().st_mtime) if scene_matches else None
    if scene is None:
        fallback_matches = [
            candidate
            for name in SCENE_FALLBACK_NAMES
            for candidate in path.rglob(name)
        ]
        if fallback_matches:
            scene = max(fallback_matches, key=lambda p: p.stat().st_mtime)
            print(
                "WARNING: active_fused_scene_camera0.ply was not saved in this run.\n"
                f"         Falling back to motion-safety environment cloud: {scene}\n"
                "         This cloud is in base coordinates and has no original RGB colors."
            )
    return target, scene, path


def load_cloud(path: Path) -> o3d.geometry.PointCloud:
    cloud = o3d.io.read_point_cloud(str(path))
    if cloud.is_empty():
        raise SystemExit(f"Point cloud is empty or unreadable: {path}")
    print(f"Loaded {len(cloud.points):>7} points: {path}")
    return cloud


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Show fused target/scene point clouds in an interactive Open3D window."
    )
    parser.add_argument(
        "path", nargs="?",
        help="fusion directory or a .ply file; default: newest fusion result",
    )
    parser.add_argument(
        "--mode", choices=("both", "target", "scene"), default="both",
        help="clouds to display (default: both)",
    )
    parser.add_argument("--point-size", type=float, default=2.0)
    parser.add_argument(
        "--highlight-target", action="store_true",
        help="in both mode paint scene gray and target red; default: preserve original RGB",
    )
    parser.add_argument(
        "--keep-colors", action="store_true", help=argparse.SUPPRESS,
    )
    args = parser.parse_args()

    target_path, scene_path, folder = resolve_input(args.path)
    selected: list[tuple[str, Path]] = []
    if args.mode in ("both", "scene") and scene_path:
        selected.append(("scene", scene_path))
    if args.mode in ("both", "target") and target_path:
        selected.append(("target", target_path))
    if not selected:
        raise SystemExit(f"No point cloud for mode={args.mode!r} in {folder}")

    geometries = []
    for kind, path in selected:
        cloud = load_cloud(path)
        if args.mode == "both" and args.highlight_target and not args.keep_colors:
            cloud.paint_uniform_color([0.65, 0.65, 0.65] if kind == "scene" else [1.0, 0.1, 0.1])
        geometries.append(cloud)

    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.10)
    geometries.append(axes)
    print("Controls: left-drag rotate, wheel zoom, Shift+left-drag pan, Q/Esc close")

    visualizer = o3d.visualization.Visualizer()
    if not visualizer.create_window(
        window_name=f"Fused Point Cloud - {folder.name}", width=1280, height=800
    ):
        raise SystemExit("Could not create an Open3D window. Check DISPLAY/desktop access.")
    for geometry in geometries:
        visualizer.add_geometry(geometry)
    options = visualizer.get_render_option()
    options.point_size = max(1.0, args.point_size)
    options.background_color = [1.0, 1.0, 1.0]
    options.show_coordinate_frame = True
    visualizer.run()
    visualizer.destroy_window()


if __name__ == "__main__":
    main()
