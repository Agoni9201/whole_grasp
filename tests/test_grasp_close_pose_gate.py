"""Close permission must agree with physical rotation error near vertical pitch."""

from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from real_nero_grasp.grasp_workflow import (
    GraspWorkflowHooks, current_flange_pose_error, wait_until_grasp_close_pose_ready,
)
from real_nero_grasp.types import MotionNotReached


# Saved endpoint/feedback from the 2026-10-29 10:14 real trial, not generated
# from the function under test. SDK values are rounded to six decimal places.
GOAL = np.array([-.514941, -.076374, .192372, .271671, 1.403480, -1.653929])
ACTUAL = np.array([-.515626, -.076191, .195342, .319238, 1.405810, -1.610712])


def close_gate(actual, *, pos_tol=.004):
    arm = mock.Mock()
    arm.get_flange_pose.return_value = actual
    hooks = GraspWorkflowHooks(current_flange_pose_error=current_flange_pose_error,
        print_current_flange_pose_error=mock.Mock(), wait_for_joint_settle=mock.Mock(),
        arm_motion_status_snapshot=mock.Mock())
    args = SimpleNamespace(trajectory_executor='mit', grasp_close_pos_tol=pos_tol,
        grasp_close_rpy_tol=np.deg2rad(3), grasp_close_wait_timeout=0, move_poll_interval=.02)
    wait_until_grasp_close_pose_ready(arm, GOAL, args, hooks=hooks)
    return arm, hooks


def test_saved_high_pitch_feedback_has_small_rotation_error_despite_large_euler_components():
    arm = mock.Mock()
    arm.get_flange_pose.return_value = ACTUAL
    _, _, delta, pos_err, angle = current_flange_pose_error(arm, GOAL)
    assert np.rad2deg(np.linalg.norm(delta[3:])) > 3.6
    assert .4 < np.rad2deg(angle) < .6
    assert .003 < pos_err < .0031
    _, hooks = close_gate(ACTUAL)
    hooks.wait_for_joint_settle.assert_not_called()


def test_original_three_mm_position_requirement_still_refuses_saved_feedback():
    with pytest.raises(MotionNotReached, match='pos_err=0.0031m'):
        close_gate(ACTUAL, pos_tol=.003)


def test_four_mm_position_requirement_does_not_accept_larger_translation():
    actual = GOAL.copy()
    actual[2] += .0045
    with pytest.raises(MotionNotReached, match='pos_err=0.0045m'):
        close_gate(actual)


def test_true_rotation_over_three_degrees_is_still_refused():
    actual = GOAL.copy()
    rotation = Rotation.from_rotvec([0, 0, np.deg2rad(4)]) * Rotation.from_euler('xyz', GOAL[3:])
    actual[3:] = rotation.as_euler('xyz')
    with pytest.raises(MotionNotReached, match='rotation_err=4.00deg'):
        close_gate(actual)
