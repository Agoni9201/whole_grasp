"""Offline checks for interrupted support releases and terminal MIT stops."""

from dataclasses import replace
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest

from real_nero_grasp import collision
from real_nero_grasp.execution import recover_after_torque_stop
from real_nero_grasp.runtime_safety import (
    RuntimeSafetyHooks,
    _check_runtime_joint_torque,
    check_runtime_collision_current,
)
from real_nero_grasp.types import (
    ControlledMotionStopTriggered,
    InitialSupportRelease,
    MotionCollisionContext,
    MotionSafetyScene,
    TorqueCollisionStopTriggered,
    UnsafeMotionTarget,
)


def support_context(*, direction="return", environment=None):
    points = np.array([[2.0, 2.0, 2.0]] if environment is None else environment)
    scene = MotionSafetyScene(points, points, np.array([[0.0, 0.0, 0.0]]))
    release = InitialSupportRelease(
        strict_environment_base=points,
        support_points_base=np.zeros((0, 3)),
        start_tcp_pose=np.zeros(6),
    )
    return MotionCollisionContext(
        scene=scene,
        phase="support_release",
        attached_target_tcp=np.zeros((1, 3)),
        initial_support_release=release,
        require_support_release_latch_at_endpoint=direction == "up",
        support_release_direction=direction,
    )


def args():
    return SimpleNamespace(
        table_guard_enable=False,
        motion_runtime_collision_guard=True,
        motion_runtime_torque_guard=False,
        motion_path_require_scene=True,
        motion_path_sample_joint_deg=0.25,
        motion_runtime_support_drop_slack=0.0,
        save_dir="",
        _last_capture_out_dir="",
    )


def q_at(rise):
    q = np.zeros(7)
    q[0] = rise
    return q


def modeled_metrics(q, context, options, **kwargs):
    def transform(joints, _options):
        value = np.eye(4)
        value[2, 3] = joints[0]
        return value

    return collision._motion_configuration_metrics(
        q, context, options,
        motion_robot_capsules_hook=lambda *_a, **_k: [],
        tcp_transform_from_joints_hook=transform,
        **kwargs,
    )


def validate(start, end, context, options):
    return collision.validate_motion_joint_path(
        q_at(start), q_at(end), context, options, "reverse",
        motion_joint_path_metrics_hook=lambda q0, q1, ctx, a: (
            collision._motion_joint_path_metrics(
                q0, q1, ctx, a,
                motion_configuration_metrics_hook=modeled_metrics,
            )
        ),
    )


def test_partial_return_does_not_require_finishing_an_upward_18mm_release():
    result = validate(0.0004, 0.0, support_context(), args())
    assert result["final_support_rise"] == 0.0
    assert not result["support_strict_latched"]
    assert result["margin"] >= 0.0


def test_forward_release_still_requires_strict_endpoint_and_no_downward_motion():
    with pytest.raises(UnsafeMotionTarget, match="before strict payload"):
        validate(0.0, 0.0004, support_context(direction="up"), args())
    context = replace(support_context(direction="up"),
                      require_support_release_latch_at_endpoint=False)
    with pytest.raises(UnsafeMotionTarget, match="non_monotonic"):
        validate(0.010, 0.0, context, args())


def test_return_rejects_upward_backtracking_and_new_support_penetration():
    with pytest.raises(UnsafeMotionTarget, match="return_upward"):
        validate(0.0, 0.010, support_context(), args())
    with pytest.raises(UnsafeMotionTarget, match="support_release_downward"):
        validate(0.0004, -0.002, support_context(), args())


def test_return_keeps_neighbor_environment_obstacles_and_stage_floor():
    with pytest.raises(UnsafeMotionTarget, match="attached_target"):
        validate(0.0004, 0.0,
                 support_context(environment=[[0.003, 0.0, 0.0]]), args())
    with pytest.raises(UnsafeMotionTarget, match="below required path"):
        collision.validate_motion_joint_path(
            q_at(0.0004), q_at(0.0),
            replace(support_context(), min_tcp_z=0.0002), args(), "return",
            joint_path_tcp_z_profile_hook=lambda *_: {"min_z": 0.0},
        )


def test_return_keeps_payload_table_penetration_guard():
    options = args()
    options.table_guard_enable = True
    options.table_z = 0.0
    options.table_tcp_clearance = 0.0
    options.table_tool_bottom_offset = 0.0
    context = replace(support_context(), attached_target_tcp=np.array([[0., 0., -.005]]))
    with pytest.raises(UnsafeMotionTarget, match="table_plane"):
        collision.validate_motion_joint_path(
            q_at(0.0004), q_at(0.0), context, options, "return",
            joint_path_tcp_z_profile_hook=lambda *_: {"min_z": 0.0},
            motion_joint_path_metrics_hook=lambda q0, q1, ctx, a: (
                collision._motion_joint_path_metrics(
                    q0, q1, ctx, a,
                    motion_configuration_metrics_hook=modeled_metrics,
                )
            ),
        )


def test_return_keeps_robot_collision_xy_and_tilt_constraints():
    context = support_context(environment=[[0.0, 0.0, 0.0]])
    metrics = collision._motion_configuration_metrics(
        np.zeros(7), context, args(),
        motion_robot_capsules_hook=lambda *_a, **_k: [
            (np.zeros(3), np.zeros(3), 0.01, "camera_mount")
        ],
        tcp_transform_from_joints_hook=lambda *_: np.eye(4),
    )
    assert metrics["margin"] < 0.0
    assert metrics["primitive"] == "camera_mount"
    for transform, primitive in [
        (np.array([[1., 0., 0., .006], [0., 1., 0., 0.],
                   [0., 0., 1., 0.], [0., 0., 0., 1.]]),
         "support_release_xy_drift"),
        (np.array([[1., 0., 0., 0.], [0., 0., -1., 0.],
                   [0., 1., 0., 0.], [0., 0., 0., 1.]]),
         "support_release_tilt"),
    ]:
        metrics = collision._motion_configuration_metrics(
            np.zeros(7), support_context(), args(),
            motion_robot_capsules_hook=lambda *_a, **_k: [],
            tcp_transform_from_joints_hook=lambda *_: transform,
        )
        assert metrics["margin"] < 0.0
        assert metrics["primitive"] == primitive


def test_return_keeps_strict_latch_after_observing_airborne_height():
    latched = []

    def metrics(q, *_a, **kwargs):
        latched.append(kwargs["support_strict_latched"])
        return dict(margin=1.0, nearest_clearance=1.0,
                    required_clearance=0.0, primitive="none", source="none",
                    support_rise=float(q[0]))

    collision._motion_joint_path_metrics(
        q_at(0.020), q_at(0.0), support_context(), args(),
        motion_configuration_metrics_hook=metrics,
    )
    assert not latched[0]
    assert all(latched[1:])


def test_previously_latched_return_cannot_restore_support_contact_exception():
    context = support_context(environment=[[0., 0., 0.], [2., 2., 2.]])
    context = replace(
        context,
        initial_support_release=replace(
            context.initial_support_release,
            strict_environment_base=np.array([[2., 2., 2.]]),
        ),
        support_release_strict_latched=True,
    )
    with pytest.raises(UnsafeMotionTarget, match="attached_target"):
        validate(0.0004, 0.0, context, args())


def test_torque_stop_snapshots_strict_history_before_forward_context_unwinds():
    context = support_context(direction="up")
    options = args()
    options.motion_runtime_torque_guard = True
    options._runtime_motion_stage_label = "support"
    options._runtime_torque_guard_state = dict(
        stage_label="support", baseline_nm=np.zeros(7), consecutive=1,
    )
    options._runtime_support_release_state = dict(
        context_id=id(context), strict_latched=True,
    )
    arm = mock.Mock()
    arm.get_joint_torques.return_value = np.full(7, 9.0)
    with pytest.raises(TorqueCollisionStopTriggered) as caught:
        _check_runtime_joint_torque(
            arm, options, "support", np.zeros(7), context,
            hooks=RuntimeSafetyHooks(persist_runtime_collision_diagnostic=mock.Mock()),
        )
    assert caught.value.context.support_release_strict_latched
    assert not context.support_release_strict_latched


def test_runtime_partial_return_descends_but_cannot_reverse_upward():
    options = args()
    context = support_context()
    options._runtime_motion_collision_context = context
    arm = mock.Mock()
    hooks = RuntimeSafetyHooks(
        motion_configuration_metrics=modeled_metrics,
        persist_runtime_collision_diagnostic=mock.Mock(),
        safe_stop=mock.Mock(),
    )
    for rise in (0.010, 0.005, 0.0):
        metrics = check_runtime_collision_current(
            arm, options, "return", q_at(rise), hooks=hooks,
        )
        assert metrics["margin"] >= 0.0
    with pytest.raises(ControlledMotionStopTriggered, match="return_runtime_upward"):
        check_runtime_collision_current(
            arm, options, "return", q_at(0.002), hooks=hooks,
        )
    arm.terminate_motion.assert_called_once()
    hooks.safe_stop.assert_not_called()


@pytest.mark.parametrize("mit_args,owner", [(True, None), (False, SimpleNamespace(owns_control=True))])
def test_mit_stop_never_unlocks_recovery_or_sends_move_j(mit_args, owner):
    arm = mock.Mock()
    arm._mit_session_owner = owner
    stop = TorqueCollisionStopTriggered(
        "torque", retreat_joints=np.zeros(7), context=support_context(), stage_label="support",
    )
    validate_hook = mock.Mock()
    assert not recover_after_torque_stop(
        arm, stop, SimpleNamespace(trajectory_executor="mit" if mit_args else "move_j"),
        validate_motion_joint_path_hook=validate_hook,
    )
    validate_hook.assert_not_called()
    arm.get_joint_angles.assert_not_called()
    arm.allow_controlled_recovery.assert_not_called()
    arm.move_j.assert_not_called()


def test_move_j_return_uses_new_reverse_context_without_mutating_frozen_plan():
    context = support_context(direction="up")
    stop = TorqueCollisionStopTriggered(
        "torque", retreat_joints=np.zeros(7), context=context, stage_label="support",
    )
    arm = mock.Mock()
    arm.get_joint_angles.return_value = q_at(0.0004)
    options = args()
    seen = []

    def check(q0, q1, reverse, a, label):
        seen.append(reverse)
        return validate(float(q0[0]), float(q1[0]), reverse, a)

    assert recover_after_torque_stop(
        arm, stop, options,
        validate_motion_joint_path_hook=check,
        wait_until_joints_reached_hook=lambda *_: (True, np.zeros(7), "settled"),
    )
    reverse = seen[0]
    assert reverse.support_release_direction == "return"
    assert not reverse.require_support_release_latch_at_endpoint
    assert reverse.scene is context.scene
    assert reverse.attached_target_tcp is context.attached_target_tcp
    assert reverse.initial_support_release is context.initial_support_release
    assert context.support_release_direction == "up"
    assert context.require_support_release_latch_at_endpoint
    assert not hasattr(options, "_runtime_motion_collision_context")
    arm.move_j.assert_called_once()
