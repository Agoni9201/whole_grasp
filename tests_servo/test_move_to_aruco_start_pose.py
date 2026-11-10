"""Pure helper tests for move_to_aruco_start_pose.py."""

import os
import sys

import numpy as np
import numpy.testing as npt

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts"))

from move_to_aruco_start_pose import check_hard_limits, joint_space_waypoints, step_towards


def test_step_towards_limits_each_joint_without_overshoot():
    current = np.array([0.0, 1.0, -1.0])
    target = np.array([0.5, 0.7, -1.01])
    out = step_towards(current, target, max_step=0.1)
    npt.assert_allclose(out, [0.1, 0.9, -1.01])


def test_step_towards_reaches_target_when_inside_step():
    current = np.array([0.0, 0.0, 0.0])
    target = np.array([0.01, -0.02, 0.03])
    out = step_towards(current, target, max_step=0.1)
    npt.assert_allclose(out, target)


def test_check_hard_limits_accepts_inside_limits():
    limits = np.array([[-1.0, 1.0]] * 7)
    ok, reason = check_hard_limits(np.zeros(7), limits=limits)
    assert ok
    assert reason == ""


def test_check_hard_limits_rejects_outside_limits():
    limits = np.array([[-1.0, 1.0]] * 7)
    q = np.zeros(7)
    q[2] = 1.2
    ok, reason = check_hard_limits(q, limits=limits)
    assert not ok
    assert "joint3" in reason


def test_joint_space_waypoints_split_by_max_joint_step():
    start = np.array([0.0, 0.0])
    target = np.array([0.0, 1.0])
    waypoints = joint_space_waypoints(start, target, max_step=0.3)
    assert len(waypoints) == 4
    npt.assert_allclose(waypoints[-1], target)
    for previous, current in zip([start] + waypoints[:-1], waypoints):
        assert np.max(np.abs(current - previous)) <= 0.3 + 1e-12
