"""Modular implementation of the real Nero perception-to-grasp workflow.

Importing this package does not initialize ROS, Open3D, GraspNet, TRAC-IK, or
robot hardware.  The legacy runner remains the compatibility facade.
"""

from .types import DynamicEnvironmentMap, GraspAttemptResult, GraspAttemptStatus


__all__ = [
    "DynamicEnvironmentMap",
    "GraspAttemptResult",
    "GraspAttemptStatus",
]
