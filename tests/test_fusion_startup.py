"""Startup mode transitions and their motion gate; no devices or SDK writes."""
from types import SimpleNamespace as NS
from unittest.mock import Mock

import numpy as np
import pytest

from handoff.fusion_handover import FusionHandover
from handoff.fusion_servo import FusionServoMixin, MIT_STARTUP_TIMEOUT_SECONDS


class Adapter(FusionServoMixin):
    pass


@pytest.fixture
def startup(monkeypatch):
    from handoff import fusion_servo
    clock = [10.]
    monkeypatch.setattr(fusion_servo.time, 'monotonic', lambda: clock[0])
    monkeypatch.setattr(fusion_servo.time, 'time', lambda: 100.+clock[0])
    msg = NS(ctrl_mode=1, mode_feedback=1, arm_status=0, err_code=0)
    parser = NS(**{f'motor_state_{i}': NS(timestamp=110., msg=NS(current=.1))
                   for i in range(1, 8)})
    a = Adapter()
    a._fusion = FusionHandover()
    a._fusion_simulated = False
    a._fusion_mit_ready = a._fusion_startup_waiting = False
    a._fusion_startup_at = a._fusion_hold_control_mode = None
    a.state, a._imp_engaged = 'HOLD_LIFTED', False
    a.backend = NS(get_robot=lambda: NS(_parser=parser,
        get_arm_status=lambda: NS(timestamp=100.+clock[0], msg=msg)))
    a._fusion_health = NS(read=lambda *_: dict(motor_timestamps=np.full(7, 100.+clock[0])))
    a._fusion_observer = NS(poll=lambda: ({}, []))
    a._fusion_recorder = NS(error=None)
    a._pause_target_motion = Mock()
    a.gripper = Mock()
    return a, clock, msg


def poll(a, clock):
    return a._poll_fusion(np.zeros(7), 100.+clock[0])


def test_initial_position_status_waits_for_actual_mit_without_latching_fault(startup):
    a, clock, msg = startup
    assert poll(a, clock) is False
    assert a._fusion.phase == 'FOLLOW_RECEIVER'
    assert a._fusion_startup_waiting and not a._fusion_mit_ready
    assert 'waiting for MIT feedback' in a._fusion.reason
    assert a._fusion.baseline is None
    clock[0] += .105  # Observed held-object startup delay in the failed run.
    msg.mode_feedback = 6
    assert poll(a, clock) is True
    assert a._fusion_mit_ready and not a._fusion_startup_waiting
    assert a._fusion.phase == 'FOLLOW_RECEIVER'
    assert 'MIT feedback ready' in a._fusion.reason
    a._pause_target_motion.assert_not_called()
    assert a.gripper.mock_calls == []


def test_startup_that_never_reaches_mit_times_out_and_stays_faulted(startup):
    a, clock, msg = startup
    assert poll(a, clock) is False
    clock[0] += MIT_STARTUP_TIMEOUT_SECONDS
    assert poll(a, clock) is False
    assert a._fusion.phase == 'FUSION_FAULT'
    assert 'MIT startup timed out' in a._fusion.reason
    assert not a._fusion_startup_waiting
    a._pause_target_motion.assert_called_once()
    msg.mode_feedback = 6
    poll(a, clock)
    assert a._fusion.phase == 'FUSION_FAULT'
    assert a.gripper.mock_calls == []


def test_position_mode_after_initial_mit_is_fault_even_without_baseline(startup):
    a, clock, msg = startup
    msg.mode_feedback = 6
    assert poll(a, clock) is True
    msg.mode_feedback = 1
    clock[0] += .01
    assert poll(a, clock) is False
    assert a._fusion.phase == 'FUSION_FAULT'
    assert a._fusion_startup_at is None  # No second startup grace period.


@pytest.mark.parametrize('changes', [
    dict(ctrl_mode=0), dict(arm_status=1), dict(err_code=1), dict(mode_feedback=2),
])
def test_startup_wait_never_exempts_invalid_control_or_error_status(startup, changes):
    a, clock, msg = startup
    for key, value in changes.items():
        setattr(msg, key, value)
    assert poll(a, clock) is False
    assert a._fusion.phase == 'FUSION_FAULT'
    assert not a._fusion_startup_waiting


@pytest.mark.parametrize('state,engaged,locked,phase', [
    ('TRACK_HAND', False, None, 'FOLLOW_RECEIVER'),
    ('HOLD_LIFTED', True, None, 'FOLLOW_RECEIVER'),
    ('HOLD_LIFTED', False, (1, 6), 'FOLLOW_RECEIVER'),
    ('HOLD_LIFTED', False, None, 'FUSION_SETTLING'),
])
def test_startup_exemption_is_only_before_following_or_handover(startup, state, engaged, locked, phase):
    a, clock, _ = startup
    a.state, a._imp_engaged = state, engaged
    a._fusion_hold_control_mode, a._fusion.phase = locked, phase
    assert poll(a, clock) is False
    assert a._fusion.phase == 'FUSION_FAULT'


def test_stale_status_is_rejected_during_startup(startup):
    a, clock, msg = startup
    a.backend = NS(get_robot=lambda: NS(get_arm_status=lambda: NS(timestamp=1., msg=msg)))
    with pytest.raises(ValueError, match='fresh arm status'):
        a._fusion_arm_status()


@pytest.mark.parametrize('state', ['HOLD_LIFTED', 'TRACK_HAND'])
def test_servo_cannot_engage_follow_or_accept_manual_parking_while_waiting_for_mit(state):
    from test_servo_follow_only import _tracking_node
    node = _tracking_node(state, False)
    node._poll_fusion = Mock(return_value=False)
    node._fusion_report = Mock()
    node._fusion_manual_stop = Mock()
    node._engage = Mock()
    node._on_timer()
    node._fusion_report.assert_called_once()
    node._fusion_manual_stop.assert_not_called()
    node._engage.assert_not_called()
    node._refresh.assert_not_called()
    assert node.gripper.mock_calls == []
