"""Device observations never transmit mode, enable, or motion commands."""
from types import SimpleNamespace as NS
from unittest.mock import Mock

from control_backends.force_ctrl.device_feedback import describe_device_feedback, read_device_feedback


def test_cached_read_uses_only_status_getters_and_copies_enable_state(monkeypatch):
    import control_backends.force_ctrl.device_feedback as module
    monkeypatch.setattr(module.time, 'time', lambda: 10.)
    status = NS(timestamp=9.9, msg=NS(ctrl_mode=1, arm_status=0, mode_feedback=6,
                                    teach_status=0, motion_status=0, err_code=0))
    flags = NS(driver_enable_status=True, driver_error_status=False, stall_status=False)
    drivers = [NS(timestamp=9.9, msg=NS(foc_status=flags)) for _ in range(7)]
    robot = Mock(spec=['get_arm_status', 'get_driver_states'])
    robot.get_arm_status.return_value = status
    robot.get_driver_states.side_effect = drivers
    result = read_device_feedback(robot)
    assert result['status']['mode_feedback'] == 6
    assert result['status']['fresh']
    assert [call.args[0] for call in robot.get_driver_states.call_args_list] == list(range(1, 8))
    assert all(item['driver_enable_status'] is True for item in result['drivers'])
    assert result['read_errors'] == []
    flags.driver_enable_status = False
    assert all(item['driver_enable_status'] is True for item in result['drivers'])


def test_stale_missing_and_future_feedback_do_not_count_as_current_enable():
    driver = NS(timestamp=8., msg=NS(foc_status=NS(driver_enable_status=True)))
    result = describe_device_feedback(None, [driver, None], 10.)
    assert result['status']['mode_feedback'] is None
    assert not result['status']['fresh']
    assert result['drivers'][0]['driver_enable_status'] is True
    assert not result['drivers'][0]['fresh']
    assert result['drivers'][1]['driver_enable_status'] is None
    driver.timestamp = 11.
    assert not describe_device_feedback(None, [driver], 10.)['drivers'][0]['fresh']


def test_getter_failure_preserves_other_joints_and_returns_an_observation():
    robot = Mock(spec=['get_arm_status', 'get_driver_states'])
    robot.get_arm_status.side_effect = RuntimeError('status unavailable')
    flags = NS(driver_enable_status=False, driver_error_status=True, stall_status=True)
    robot.get_driver_states.side_effect = [None, RuntimeError('joint2 unavailable')] + [
        NS(timestamp=10., msg=NS(foc_status=flags)) for _ in range(5)]
    result = read_device_feedback(robot)
    assert len(result['read_errors']) == 2
    assert result['drivers'][1]['driver_enable_status'] is None
    assert result['drivers'][2]['driver_enable_status'] is False
    assert result['drivers'][2]['driver_error_status'] is True
    assert result['drivers'][2]['stall_status'] is True
