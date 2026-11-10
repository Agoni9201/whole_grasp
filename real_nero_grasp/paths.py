"""Filesystem locations shared by the real Nero grasp package.

All defaults remain anchored at the repository root, even though the runtime
implementation now lives one directory deeper.
"""

from pathlib import Path


PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parent

DEFAULT_YOLO_WEIGHTS = str(PROJECT_ROOT / "models" / "yolo26m-seg.pt")
DEFAULT_IK_URDF = str(
    PROJECT_ROOT
    / "manipulator_grasp"
    / "nero_urdf"
    / "agx_arm_urdf"
    / "nero"
    / "urdf"
    / "nero_description.urdf"
)

GRASPNET_IMPORT_DIRS = tuple(
    str(PROJECT_ROOT / "graspnet-baseline" / name)
    for name in ("models", "dataset", "utils")
)


__all__ = [
    "PACKAGE_DIR",
    "PROJECT_ROOT",
    "DEFAULT_YOLO_WEIGHTS",
    "DEFAULT_IK_URDF",
    "GRASPNET_IMPORT_DIRS",
]
