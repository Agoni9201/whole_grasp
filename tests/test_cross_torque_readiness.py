"""Regression: the position-only point test has no extra torque prerequisite."""
import inspect
import time
from types import SimpleNamespace

import numpy as np
import pytest

from tools import cross_point_check as cross


class PositionArm:
    def __init__(self, motor_sample):
        self.q = np.array([-0.4, 0, 0.25, 0, 0, 0, 0], dtype=float)
        self.commands, self.motor_reads = [], []
        self.arm = SimpleNamespace(get_flange_pose=self._pose,
                                   get_motor_states=lambda index: self._motor(index, motor_sample))

    def _pose(self):
        return SimpleNamespace(timestamp=time.time(), msg=self.q[:6].copy())

    def _motor(self, index, sample):
        self.motor_reads.append(index)
        return sample

    def get_joint_angles(self):
        return self.q.copy()

    def _controller_motion_active(self):
        return False

    def get_joint_torques(self):
        pytest.fail("Position point test must not require or read torque feedback")

    def move_j(self, joints):
        self.q = np.asarray(joints, float).copy()
        self.commands.append(self.q.copy())


def segment(arm):
    source = arm.q.copy(), cross.pose_transform(arm.q[:6])
    target_q = arm.q.copy()
    target_q[0] += 0.001
    return source, [(target_q, cross.pose_transform(target_q[:6]))]


def execution_args():
    return SimpleNamespace(min_x=-0.8, max_x=0.8, min_y=-0.6, max_y=0.6,
                           min_z=0.02, max_z=1.2)


def fk(joints):
    return cross.pose_transform(np.asarray(joints)[:6])


@pytest.mark.parametrize("kind", ["absent", "no_stamp", "stale", "nan", "large"])
def test_position_segment_never_consults_motor_or_torque_feedback(kind, monkeypatch):
    samples = {
        "absent": None,
        "no_stamp": SimpleNamespace(msg=SimpleNamespace(torque=0)),
        "stale": SimpleNamespace(timestamp=0, msg=SimpleNamespace(torque=0)),
        "nan": SimpleNamespace(timestamp=float("nan"), msg=SimpleNamespace(torque=float("nan"))),
        "large": SimpleNamespace(timestamp=time.time(), msg=SimpleNamespace(torque=1000)),
    }
    arm = PositionArm(samples[kind])
    source, nodes = segment(arm)
    result = dict(motion_sent=False, samples=[])
    monkeypatch.setattr(cross.time, "sleep", lambda _: None)
    cross._execute_cross_steps(arm, nodes, np.eye(4), result, "to_hover", source=source,
                               fk=fk, args=execution_args())
    assert len(arm.commands) == 1 and not arm.motor_reads
    assert result["motion_sent"] is True and len(result["samples"]) == 1
    np.testing.assert_array_equal(arm.q, nodes[-1][0])


@pytest.mark.parametrize("stream", ["joint", "flange"])
def test_position_feedback_freshness_still_blocks_stale_data(stream):
    arm = PositionArm(None)
    source, nodes = segment(arm)
    if stream == "joint":
        arm.arm.get_joint_angles = lambda: SimpleNamespace(timestamp=time.time()-2, msg=arm.q.copy())
    else:
        arm.arm.get_flange_pose = lambda: SimpleNamespace(timestamp=time.time()-2, msg=arm.q[:6].copy())
    with pytest.raises(RuntimeError):
        cross._execute_cross_steps(arm, nodes, np.eye(4), dict(samples=[]), "to_hover", source=source,
                                   fk=fk, args=execution_args())
    assert not arm.commands


def test_position_executor_preserves_existing_stop_latch():
    arm = PositionArm(None)
    source, nodes = segment(arm)
    arm._safety_stop_latched = True
    with pytest.raises(RuntimeError, match="锁存"):
        cross._execute_cross_steps(arm, nodes, np.eye(4), dict(samples=[]), "return", source=source,
                                   fk=fk, args=execution_args())
    assert not arm.commands


def test_no_hidden_torque_dependency_remains_in_point_test_module():
    source = inspect.getsource(cross)
    assert "get_joint_torques" not in source
    assert "get_motor_states" not in source
    assert not hasattr(cross, "_wait_torque_feedback")
