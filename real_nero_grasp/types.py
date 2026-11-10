"""Shared exceptions, result enums, and immutable workflow data models.

The definitions in this module intentionally preserve the legacy runner's
public field layout and defaults so they can be re-exported without changing
runtime behavior.
"""

from dataclasses import dataclass
from enum import Enum

import numpy as np


__all__ = [
    "MotionNotReached",
    "UnsafeMotionTarget",
    "MotionPreflightRejected",
    "SafetyStopTriggered",
    "ControlledMotionStopTriggered",
    "TorqueCollisionStopTriggered",
    "GraspNotDetected",
    "TargetIdentityNotVerified",
    "PlanNotExecutable",
    "GraspSelectionDeclined",
    "InvalidGraspSelection",
    "GraspAttemptStatus",
    "GraspAttemptResult",
    "DynamicEnvironmentMap",
    "ActiveViewResult",
    "GripperCloseResult",
    "MotionSafetyScene",
    "InitialSupportRelease",
    "MotionCollisionContext",
    "PlannedJointStage",
    "GraspMotionPlan",
]


class MotionNotReached(RuntimeError):
    pass


class UnsafeMotionTarget(RuntimeError):
    pass


class MotionPreflightRejected(UnsafeMotionTarget):
    """Live path validation failed before sending this stage's target."""

    def __init__(self, message, *, motion_sent=False):
        super().__init__(message)
        self.motion_sent = bool(motion_sent)


class SafetyStopTriggered(RuntimeError):
    pass


class ControlledMotionStopTriggered(SafetyStopTriggered):
    """The current trajectory was terminated without electronic e-stop."""


class TorqueCollisionStopTriggered(ControlledMotionStopTriggered):
    """A sustained measured joint-torque excursion stopped one live stage.

    The exception carries only the reverse endpoint of the already frozen
    stage.  Recovery must still validate the measured-current-to-start edge
    before commanding the low-speed retreat.
    """

    def __init__(self, message, *, retreat_joints, context, stage_label):
        super().__init__(message)
        self.retreat_joints = _readonly_array(retreat_joints, np.float64, (7,))
        self.context = context
        self.stage_label = str(stage_label)


class GraspNotDetected(RuntimeError):
    def __init__(
        self,
        message,
        *,
        retryable_empty=False,
        reason_code=None,
    ):
        super().__init__(message)
        self.retryable_empty = bool(retryable_empty)
        self.reason_code = None if reason_code is None else str(reason_code)


class TargetIdentityNotVerified(RuntimeError):
    """The originally selected target could not be tracked unambiguously."""


class PlanNotExecutable(RuntimeError):
    pass


class GraspSelectionDeclined(RuntimeError):
    pass


class InvalidGraspSelection(ValueError):
    pass


class GraspAttemptStatus(str, Enum):
    SUCCESS = "success"
    CONFIGURATION_FAILED = "configuration_failed"
    CAPTURE_FAILED = "capture_failed"
    CLEAR_FAILED = "clear_failed"
    INTERNAL_ERROR = "internal_error"
    PLANNING_FAILED = "planning_failed"
    PLANNED_ONLY = "planned_only"
    CONTACT_CHECKED = "contact_checked"
    DECLINED = "declined"
    EXECUTION_FAILED = "execution_failed"


@dataclass(frozen=True)
class GraspAttemptResult:
    status: GraspAttemptStatus
    motion_sent: bool = False
    detail: str = ""
    empty_grasp_recovered_to_initial: bool = False

    @property
    def succeeded(self):
        return self.status is GraspAttemptStatus.SUCCESS


def _readonly_array(value, dtype, shape):
    """Return an owned, immutable NumPy array with a validated shape."""
    array = np.array(value, dtype=dtype, copy=True).reshape(shape)
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class DynamicEnvironmentMap:
    """Canonical dynamic obstacle occupancy expressed in ``base_link``.

    ``hard_environment_base`` is safe to consume as normal collision
    occupancy. ``uncertain_environment_base`` contains isolated observations
    that must remain visible to diagnostics and conservative policies without
    being silently promoted to a confirmed basket surface.  The voxel metadata
    arrays share one row order: ``voxel_keys[i]``, ``view_support[i]`` and
    ``confidence[i]`` describe the same occupied voxel.

    Frozen dataclasses alone do not make NumPy buffers immutable, so every
    array is copied and marked read-only here.
    """

    hard_environment_base: object
    uncertain_environment_base: object
    target_base: object = None
    voxel_keys: object = None
    view_support: object = None
    confidence: object = None
    source_view_names: tuple = ()
    roi_bounds_base: object = None
    frame_id: str = "base_link"
    map_id: str = "dynamic_environment"
    generation: int = 0
    voxel_size: float = 0.005

    def __post_init__(self):
        hard = _readonly_array(
            self.hard_environment_base,
            np.float32,
            (-1, 3),
        )
        uncertain = _readonly_array(
            self.uncertain_environment_base,
            np.float32,
            (-1, 3),
        )
        target = (
            None
            if self.target_base is None
            else _readonly_array(self.target_base, np.float32, (-1, 3))
        )
        keys = (
            _readonly_array([], np.int64, (0, 3))
            if self.voxel_keys is None
            else _readonly_array(self.voxel_keys, np.int64, (-1, 3))
        )
        support = (
            _readonly_array([], np.int32, (0,))
            if self.view_support is None
            else _readonly_array(self.view_support, np.int32, (-1,))
        )
        confidence = (
            _readonly_array([], np.float32, (0,))
            if self.confidence is None
            else _readonly_array(self.confidence, np.float32, (-1,))
        )
        if not (keys.shape[0] == support.shape[0] == confidence.shape[0]):
            raise ValueError(
                "voxel_keys, view_support, and confidence must have equal length"
            )
        bounds = (
            None
            if self.roi_bounds_base is None
            else _readonly_array(self.roi_bounds_base, np.float32, (2, 3))
        )
        voxel_size = float(self.voxel_size)
        if not np.isfinite(voxel_size) or voxel_size <= 0.0:
            raise ValueError("DynamicEnvironmentMap voxel_size must be finite and positive")
        try:
            generation_value = float(self.generation)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "DynamicEnvironmentMap generation must be a finite "
                "non-negative integer"
            ) from exc
        if not np.isfinite(generation_value) or generation_value < 0.0:
            raise ValueError(
                "DynamicEnvironmentMap generation must be a finite "
                "non-negative integer"
            )
        generation = int(generation_value)
        if float(generation) != generation_value:
            raise ValueError(
                "DynamicEnvironmentMap generation must be a finite "
                "non-negative integer"
            )

        object.__setattr__(self, "hard_environment_base", hard)
        object.__setattr__(self, "uncertain_environment_base", uncertain)
        object.__setattr__(self, "target_base", target)
        object.__setattr__(self, "voxel_keys", keys)
        object.__setattr__(self, "view_support", support)
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(
            self,
            "source_view_names",
            tuple(str(name) for name in self.source_view_names),
        )
        object.__setattr__(self, "roi_bounds_base", bounds)
        object.__setattr__(self, "frame_id", str(self.frame_id))
        object.__setattr__(self, "map_id", str(self.map_id))
        object.__setattr__(self, "generation", generation)
        object.__setattr__(self, "voxel_size", voxel_size)


@dataclass(frozen=True)
class ActiveViewResult:
    target_cloud: object
    scene_cloud: object
    T_base_camera: object
    used_multiview: bool = False
    kept_view_count: int = 1
    reason: str = "single_view"
    restore_ok: bool = True
    motion_sent: bool = False
    # GraspNet may use a target-near cropped scene, but whole-workflow motion
    # validation must retain the uncropped base-visible environment.
    motion_scene_cloud: object = None
    # Exact non-target pixels when available.  This avoids deleting a blanket
    # NN band around the target and accidentally hiding a touching neighbor.
    motion_environment_cloud: object = None
    # Canonical base-frame occupancy shared by planning and execution.  Kept at
    # the end to preserve every historical positional constructor.
    motion_environment_map: object = None


@dataclass(frozen=True)
class GripperCloseResult:
    status: object
    contact_detected: bool
    stop_reason: str
    contact_width: float = None
    contact_force: float = None


@dataclass(frozen=True)
class MotionSafetyScene:
    """Point-cloud world model expressed once in ``base_link`` coordinates."""

    full_scene_base: object
    environment_base: object
    target_base: object
    map_id: str = "capture"
    # Optional provenance for the dynamic map that produced environment_base.
    # Appended fields keep historical positional construction compatible.
    environment_map: object = None
    generation: int = 0


@dataclass(frozen=True)
class InitialSupportRelease:
    """Narrow exception for a payload initially resting on the table/mat.

    Only locally verified planar support points are removed from the payload's
    environment query during a short, nearly vertical release.  Robot geometry
    still checks the complete environment, and all payload obstacles become
    strict again before the release stage ends.
    """

    strict_environment_base: object
    support_points_base: object
    start_tcp_pose: object
    # Environment used only by the closed gripper tool during the short
    # support-release.  It excludes verified support points and non-target
    # points that are demonstrably segmentation leakage inside the selected
    # target envelope.  Arm links and the camera continue to use
    # ``strict_environment_base``.
    tool_contact_environment_base: object = None
    target_contact_points_base: object = None
    strict_after_rise: float = 0.018
    max_xy_drift: float = 0.005
    max_tilt_rad: float = np.deg2rad(3.0)
    downward_tolerance: float = 0.001


@dataclass(frozen=True)
class MotionCollisionContext:
    """Stage-specific collision policy used by preflight and live monitoring."""

    scene: MotionSafetyScene
    phase: str
    allow_target_tool_contact: bool = False
    attached_target_tcp: object = None
    placed_target_base: object = None
    min_tcp_z: float = None
    runtime_min_tcp_z_slack: float = 0.0
    initial_support_release: object = None
    # Clear gap between the two closed finger inner faces.  This is zero for
    # an empty close (the two 7 mm-radius finger centerlines overlap into one
    # compact envelope), or the candidate GraspNet width plus the configured
    # holding allowance after a successful grasp.  ``None`` keeps legacy
    # callers compatible and falls back to the empty-close width.
    closed_gripper_width: float = None
    gripper_mesh_opening: float = None
    # A segmented support release may legitimately stop below the height at
    # which strict payload/environment checking latches.  Intermediate
    # segments still obey the support-release path constraints; only the final
    # segment must end at or above ``strict_after_rise``.
    require_support_release_latch_at_endpoint: bool = True
    # A verified return along an interrupted release edge descends toward its
    # saved start instead of completing an upward release. Geometry, initial
    # support penetration limits and any already-latched strict checks remain.
    support_release_direction: str = "up"
    support_release_strict_latched: bool = False


@dataclass(frozen=True)
class PlannedJointStage:
    label: str
    tcp_pose: object
    flange_pose: object
    joints: object
    phase: str
    min_tcp_z: float = None
    runtime_min_tcp_z_slack: float = 0.0
    require_support_release_latch_at_endpoint: bool = True
    path_clearance_margin: float = None


@dataclass(frozen=True)
class GraspMotionPlan:
    start_joints: object
    before_close: tuple
    after_close: tuple
    after_release: tuple
    empty_recovery: tuple
    safety_scene: MotionSafetyScene
    attached_target_tcp: object = None
    placed_target_base: object = None
    initial_support_release: object = None
    # Frozen candidate-specific width used by both planning and measured-joint
    # runtime collision monitoring after the gripper closes.
    closed_gripper_width: float = None
    # A validation-only plan intentionally stops after the closed gripper has
    # completed its collision-checked support release.  No place/release is
    # implied even when the session has a configured place pose.
    validation_only: bool = False
