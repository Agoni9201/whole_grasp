"""Normal-exit ordering and failure injection; never opens CAN or a camera."""

from types import SimpleNamespace as NS

import numpy as np
import pytest

from control_backends.force_ctrl.controller import NeroForceController, DEFAULT_KP, DEFAULT_KD
from control_backends.force_ctrl.feedback_health import FeedbackStale
from control_backends.sdk_force_ctrl_backend import SdkForceCtrlBackend


@pytest.fixture
def exit_controller(monkeypatch):
    import control_backends.force_ctrl.controller as module
    clock = NS(now=100.0)
    calls = []
    q = np.array([.1, .2, .1, .5, .1, .1, .1])

    class Monitor:
        def __init__(self, **kwargs):
            pass

        def read(self, *args):
            return {'q': q.copy(), 'position_timestamps': np.full(7, clock.now),
                    'motor_timestamps': np.full(7, clock.now)}

    class Loop:
        running = True
        fault = None
        command = None

        def get_debug_state(self):
            return {'last_error': self.fault}

        def hold_for_controlled_stop(self, command):
            calls.append('fixed_mit_hold')
            self.command = command

        def get_controlled_stop_status(self):
            return {'target_q': None if self.command is None else self.command.pos.copy(),
                    'transition_complete': self.command is not None,
                    'transition_duration_s': 0.0,
                    'transition_completed_monotonic': clock.now,
                    'phase': 'hold'}

        def stop(self):
            calls.append('mit_joined')
            self.running = False

    class Robot:
        def set_speed_percent(self, percent):
            assert not ctrl._loop.running
            assert percent == 0
            calls.append('speed_zero')

        def _send_msg(self, msg):
            assert not ctrl._loop.running
            assert msg.emergency_stop == 0
            calls.append('terminate')

        def set_motion_mode(self, mode):
            assert mode == 'j' and not ctrl._loop.running
            calls.append('mode_j')

        def move_j(self, target):
            assert not ctrl._loop.running
            np.testing.assert_array_equal(target, q)
            calls.append('device_current_q_hold')

        def get_arm_status(self):
            return NS(timestamp=clock.now, msg=NS(ctrl_mode=1, arm_status=0, mode_feedback=1, motion_status=0))

        def get_driver_states(self, index):
            return NS(timestamp=clock.now, msg=NS(foc_status=NS(driver_enable_status=True)))

        def disconnect(self):
            calls.append('disconnect')

        def disable(self):
            pytest.fail('Normal servo exit must not disable')

        def enable(self):
            pytest.fail('Exit must not re-enable after a fault')

        def electronic_emergency_stop(self):
            pytest.fail('No automatic emergency stop for a healthy normal exit')

    monkeypatch.setattr(module, 'time', NS(
        time=lambda: clock.now, monotonic=lambda: clock.now,
        sleep=lambda seconds: setattr(clock, 'now', clock.now + seconds)))
    monkeypatch.setattr(module, 'FeedbackHealthMonitor', Monitor)
    import control_backends.force_ctrl.normal_shutdown as shared
    monkeypatch.setattr(shared, 'time', module.time)
    monkeypatch.setattr(shared, 'FeedbackHealthMonitor', Monitor)
    ctrl = NeroForceController.__new__(NeroForceController)
    ctrl._robot, ctrl._loop = Robot(), Loop()
    ctrl._started = ctrl._robot_connected = ctrl._robot_enabled = True
    ctrl._emergency_stopped = False
    ctrl._startup_joint_margin = .08
    ctrl._exit_hold_q = None
    ctrl._kp, ctrl._kd = DEFAULT_KP.copy(), DEFAULT_KD.copy()
    return ctrl, calls, clock, q


def test_normal_exit_freezes_joins_then_confirms_enabled_device_hold(exit_controller):
    ctrl, calls, _, q = exit_controller
    ctrl.prepare_shutdown()
    ctrl.prepare_shutdown()
    ctrl.stop_keep_enabled()
    ctrl.stop_keep_enabled()
    assert calls == ['fixed_mit_hold', 'mit_joined', 'speed_zero',
                     'terminate', 'terminate', 'terminate', 'mode_j',
                     'device_current_q_hold', 'disconnect']
    np.testing.assert_array_equal(ctrl._loop.command.pos, q)
    np.testing.assert_array_equal(ctrl._loop.command.vel, 0)
    assert np.all(ctrl._loop.command.kp > 0)
    assert ctrl._robot_enabled
    assert not ctrl._robot_connected


def test_device_hold_is_observed_beyond_first_ack_and_records_switch_feedback(exit_controller):
    from control_backends.force_ctrl.normal_shutdown import handoff_to_position_hold
    ctrl, calls, clock, _ = exit_controller
    events = []
    started = clock.now
    result = handoff_to_position_hold(
        ctrl._robot, ctrl._loop.stop, check_fault=lambda: None,
        record_event=lambda event, **values: events.append((event, values)))
    assert clock.now - started >= .5
    assert result['observation_s'] >= .5 and result['confirmed']
    assert result['peak_joint_drift_deg'] == 0
    assert result['peak_joint_speed_deg_s'] == 0
    samples = [values for event, values in events if event == 'device_hold_sample']
    assert len(samples) >= 20
    assert samples[-1]['elapsed_s'] >= .5
    assert events[0][1]['stage'] == 'before_sender_join'
    assert events[1][1]['stage'] == 'before_mode_switch'
    assert calls.count('device_current_q_hold') == 1


def test_late_device_hold_drift_is_caught_after_initial_mode_ack(exit_controller):
    import control_backends.force_ctrl.normal_shutdown as shared
    ctrl, calls, clock, q = exit_controller
    started = clock.now
    class DelayedDrift:
        def read(self, *args):
            measured = q.copy()
            if clock.now - started > .2:
                measured[0] += np.deg2rad(1.1)
            return {'q': measured, 'position_timestamps': np.full(7, clock.now),
                    'motor_timestamps': np.full(7, clock.now)}
    from unittest.mock import patch
    with patch.object(shared, 'FeedbackHealthMonitor', DelayedDrift):
        with pytest.raises(RuntimeError, match='hold drifted'):
            shared.handoff_to_position_hold(ctrl._robot, ctrl._loop.stop, check_fault=lambda: None)
    assert clock.now - started >= .2
    assert 'disconnect' not in calls


@pytest.mark.parametrize('observation', [float('nan'), -1, 2.0])
def test_invalid_hold_observation_sends_no_commands(exit_controller, observation):
    from control_backends.force_ctrl.normal_shutdown import handoff_to_position_hold
    ctrl, calls, _, _ = exit_controller
    with pytest.raises(ValueError, match='observation'):
        handoff_to_position_hold(ctrl._robot, ctrl._loop.stop,
                                 check_fault=lambda: None, observation_s=observation)
    assert calls == []


def test_optional_hold_diagnostic_failure_does_not_prevent_enabled_handoff(exit_controller):
    from control_backends.force_ctrl.normal_shutdown import handoff_to_position_hold
    ctrl, calls, _, _ = exit_controller
    def fail(*args, **kwargs):
        raise OSError('diagnostics unavailable')
    result = handoff_to_position_hold(
        ctrl._robot, ctrl._loop.stop, check_fault=lambda: None, record_event=fail)
    assert result['confirmed'] and calls.count('device_current_q_hold') == 1


def test_unconfirmed_handoff_records_actual_device_mode_and_enable(exit_controller):
    from control_backends.force_ctrl.normal_shutdown import handoff_to_position_hold
    ctrl, calls, clock, _ = exit_controller
    ctrl._robot.get_arm_status = lambda: NS(
        timestamp=clock.now, msg=NS(ctrl_mode=1, arm_status=0, mode_feedback=6,
                                    motion_status=0, err_code=0))
    events = []
    with pytest.raises(RuntimeError, match='last_device_feedback=.*mode_feedback.*6'):
        handoff_to_position_hold(ctrl._robot, ctrl._loop.stop, check_fault=lambda: None,
                                 record_event=lambda event, **values: events.append((event, values)))
    assert events[-1][0] == 'device_hold_unconfirmed'
    feedback = events[-1][1]['device_feedback']
    assert feedback['status']['mode_feedback'] == 6
    assert all(item['driver_enable_status'] is True for item in feedback['drivers'])
    assert 'disconnect' not in calls


@pytest.mark.parametrize('fault_kind', ['emergency', 'inner_fault'])
def test_fault_exit_never_restores_enable_or_sends_normal_hold(exit_controller, fault_kind):
    ctrl, calls, _, _ = exit_controller
    if fault_kind == 'emergency':
        ctrl._emergency_stopped = True
    else:
        ctrl._loop.fault = 'feedback stale'
    ctrl._loop._electronic_stop_sent = True
    ctrl.prepare_shutdown()
    ctrl.stop_keep_enabled()
    assert calls == ['mit_joined', 'disconnect']


def test_unjoined_mit_sender_cannot_switch_mode_or_disconnect(exit_controller):
    ctrl, calls, _, _ = exit_controller
    def fail():
        raise RuntimeError('worker still running')
    ctrl._loop.stop = fail
    with pytest.raises(RuntimeError, match='worker still running'):
        ctrl.stop_keep_enabled()
    assert calls == ['fixed_mit_hold']
    assert ctrl._robot_connected


def test_stale_feedback_cannot_be_used_for_device_handoff(exit_controller):
    ctrl, calls, _, _ = exit_controller
    ctrl.prepare_shutdown()
    def stale(*args):
        raise FeedbackStale('feedback stale')
    ctrl._exit_stop.monitor.read = stale
    with pytest.raises(FeedbackStale):
        ctrl.stop_keep_enabled()
    assert calls == ['fixed_mit_hold']
    assert ctrl._robot_connected and ctrl._loop.running


def test_cached_identical_frames_never_confirm_stop(exit_controller):
    ctrl, calls, clock, q = exit_controller
    ctrl.prepare_shutdown()
    frozen = clock.now
    ctrl._exit_stop.monitor.read = lambda *args: {
        'q': q.copy(), 'position_timestamps': np.full(7, frozen),
        'motor_timestamps': np.full(7, frozen)}
    with pytest.raises(RuntimeError, match='settling was not confirmed'):
        ctrl.stop_keep_enabled()
    assert calls == ['fixed_mit_hold']


def test_no_enabled_acknowledgement_is_reported_not_hidden(exit_controller):
    ctrl, calls, clock, _ = exit_controller
    ctrl._robot.get_driver_states = lambda index: NS(
        timestamp=clock.now, msg=NS(foc_status=NS(driver_enable_status=False)))
    with pytest.raises(RuntimeError, match='not acknowledged'):
        ctrl.stop_keep_enabled()
    assert 'device_current_q_hold' in calls and 'disconnect' not in calls


def test_old_mode_acknowledgement_never_confirms_enabled_exit(exit_controller):
    ctrl, calls, _, _ = exit_controller
    ctrl._robot.get_arm_status = lambda: NS(
        timestamp=99.0, msg=NS(ctrl_mode=1, arm_status=0, mode_feedback=1, motion_status=0))
    with pytest.raises(RuntimeError, match='not acknowledged'):
        ctrl.stop_keep_enabled()
    assert 'disconnect' not in calls


def test_position_drift_cannot_confirm_stationary_hold(exit_controller):
    ctrl, calls, _, q = exit_controller
    ctrl.prepare_shutdown()
    q[0] += np.deg2rad(2.0)
    with pytest.raises(RuntimeError, match='drift exceeded'):
        ctrl.stop_keep_enabled()
    assert calls == ['fixed_mit_hold']


def test_backend_enabled_exit_never_falls_back_to_disabling_stop():
    calls = []
    def fail():
        calls.append('keep_enabled')
        raise RuntimeError('hold not confirmed')
    controller = NS(stop_keep_enabled=fail, stop=lambda: pytest.fail('disable fallback'))
    backend = SdkForceCtrlBackend(dry_run=False, keep_enabled_on_disconnect=True)
    backend._force_ctrl, backend._connected = controller, True
    with pytest.raises(RuntimeError, match='hold not confirmed'):
        backend.disconnect()
    assert calls == ['keep_enabled']
    assert backend._force_ctrl is controller and backend.connected


def test_sdk_capture_cleanup_prepares_hold_before_join(monkeypatch):
    import cameras.sdk_loop as host
    calls = []
    camera = NS(timeout=1.0, read=lambda: None, close=lambda: calls.append('camera_close'))
    worker = NS(start=lambda: None, join=lambda **kw: calls.append('capture_join'), is_alive=lambda: False)
    def interrupt():
        raise KeyboardInterrupt()
    node = NS(args=NS(), _callbacks=[], _timers=[[1.0, interrupt, 0.0]],
              prepare_shutdown=lambda: calls.append('prepare_hold'))
    monkeypatch.setattr(host, 'camera_for', lambda args: camera)
    monkeypatch.setattr(host.threading, 'Thread', lambda **kwargs: worker)
    host.init()
    try:
        with pytest.raises(KeyboardInterrupt):
            host.spin(node)
    finally:
        host.shutdown()
    assert calls == ['prepare_hold', 'capture_join', 'camera_close']


def test_standalone_factory_explicitly_selects_enabled_exit(monkeypatch):
    import time
    import handoff.grasp_to_servo_handoff as module
    captured = {}
    class Backend:
        def __init__(self, **kwargs):
            captured.update(kwargs)
        def connect(self):
            pass
        def get_joint_angles(self):
            return np.zeros(7)
        def get_robot(self):
            return NS(get_joint_angles=lambda: NS(timestamp=time.time()))
        def disconnect(self):
            pytest.fail('Healthy factory must return its backend')
    monkeypatch.setattr(module, 'NeroPinocchioModel', lambda: NS(clone=lambda: object()))
    monkeypatch.setattr(module, 'SdkForceCtrlBackend', Backend)
    module.build_servo_backend()
    assert captured['keep_enabled_on_disconnect'] is True
    assert captured['zero_gravity'] is False


def test_standalone_interrupt_closes_backend_once_before_detector_and_camera(monkeypatch):
    import servo_delivery as module
    import cameras.realsense as camera_module
    calls = []
    backend = NS(disconnect=lambda: calls.append('backend_exit'))
    target = NS(close=lambda: calls.append('detector_close'))
    class Node:
        def __init__(self, *args, **kwargs):
            pass
        def prepare_shutdown(self):
            calls.append('prepare_hold')
    def interrupt(*args):
        raise KeyboardInterrupt()
    monkeypatch.setattr(module, 'validate_servo_args', lambda args: None)
    monkeypatch.setattr(module, 'NeroPinocchioModel', lambda: NS(clone=lambda: object()))
    monkeypatch.setattr(module, 'build_target', lambda *args: target)
    monkeypatch.setattr(module, 'ServoDeliveryNode', Node)
    monkeypatch.setattr(module.sdk_loop, 'spin', interrupt)
    monkeypatch.setattr(camera_module, 'camera_for', lambda args: NS(read=lambda: None))
    monkeypatch.setattr(camera_module, 'close_all', lambda: calls.append('camera_close'))
    assert module.run_servo_delivery(NS(), backend=backend) == 130
    assert calls == ['prepare_hold', 'backend_exit', 'detector_close', 'camera_close']


def _unified_session(ctrl, clock, q):
    from real_nero_grasp.mit_trajectory_executor import MITTrajectorySession
    session = MITTrajectorySession.__new__(MITTrajectorySession)
    session._arm = NS(arm=ctrl._robot, _safety_stop_latched=False,
                      _safety_stop_request_confirmed=False,
                      _controlled_stop_request_confirmed=False, _stop_confirm_timeout=2.0)
    session._driver, session._loop = ctrl._robot, ctrl._loop
    session._owns_control, session._started = True, True
    session._shutdown_complete = False
    session._kp, session._kd = DEFAULT_KP.copy(), DEFAULT_KD.copy()
    session._feedback_timeout_s = .5
    ctrl._loop.get_joint_state = lambda: NS(
        pos=q.copy(), timestamp=clock.now, vel=np.zeros(7), torque=np.zeros(7))
    ctrl._loop.set_diagnostics_phase = lambda phase: None
    return session


@pytest.mark.parametrize('stage', ['active_side1', 'target_grasp', 'gripper_close',
                                  'lift', 'visual_servo', 'idle_planning'])
def test_every_unified_stage_exits_through_same_enabled_hold(exit_controller, monkeypatch, stage):
    import control_backends.force_ctrl.controller as controller_module
    import real_nero_grasp.mit_trajectory_executor as module
    ctrl, calls, clock, q = exit_controller
    monkeypatch.setattr(module, 'time', controller_module.time)
    session = _unified_session(ctrl, clock, q)
    session._phase = stage
    session.prepare_shutdown()
    session.shutdown()
    session.disconnect()
    assert calls == ['fixed_mit_hold', 'mit_joined', 'speed_zero',
                     'terminate', 'terminate', 'terminate', 'mode_j', 'device_current_q_hold']
    assert session._device_hold_confirmed and session._shutdown_complete
    assert session._arm._controlled_stop_request_confirmed
    # Only the grasp arm context may disconnect its existing CAN connection.
    assert 'disconnect' not in calls


def test_unified_exit_fault_never_switches_to_normal_mode(exit_controller):
    ctrl, calls, clock, q = exit_controller
    session = _unified_session(ctrl, clock, q)
    session._emergency_stop_requested = True
    session.shutdown()
    assert calls == ['mit_joined']
    assert not getattr(session, '_device_hold_confirmed', False)


def test_unified_handoff_requires_new_ack_and_retains_retryable_owner(exit_controller, monkeypatch):
    import control_backends.force_ctrl.controller as controller_module
    import real_nero_grasp.mit_trajectory_executor as module
    ctrl, calls, clock, q = exit_controller
    monkeypatch.setattr(module, 'time', controller_module.time)
    ctrl._robot.get_arm_status = lambda: NS(
        timestamp=99., msg=NS(ctrl_mode=1, arm_status=0, mode_feedback=1, motion_status=0))
    session = _unified_session(ctrl, clock, q)
    with pytest.raises(RuntimeError, match='not acknowledged'):
        session.shutdown()
    assert not session._shutdown_complete
    assert not getattr(session, '_device_hold_confirmed', False)
    assert 'disconnect' not in calls


def test_fault_during_final_mit_join_forbids_any_normal_target(exit_controller):
    ctrl, calls, _, _ = exit_controller
    def fault_stop():
        calls.append('mit_joined')
        ctrl._loop.running = False
        ctrl._loop.fault = 'terminal CAN failure'
    ctrl._loop.stop = fault_stop
    with pytest.raises(RuntimeError, match='forbids'):
        ctrl.stop_keep_enabled()
    assert calls == ['fixed_mit_hold', 'mit_joined']


def test_ctrl_c_in_servo_is_not_reported_as_successful_whole_workflow(monkeypatch):
    import handoff.grasp_to_servo_handoff as module
    import servo_delivery
    monkeypatch.setattr(servo_delivery, 'validate_servo_args', lambda args: None)
    monkeypatch.setattr(servo_delivery, 'run_servo_delivery', lambda *args, **kw: 130)
    backend = NS()
    with pytest.raises(KeyboardInterrupt):
        module.delivery_after_grasp(NS(), NS(execute=True, _servo_backend=backend))


def test_backend_cleanup_never_zeros_impedance_before_fixed_hold():
    from control_backends.force_ctrl.normal_shutdown import close_backend_normally
    calls = []
    backend = NS(prepare_shutdown=lambda: calls.append('fixed_hold'),
                 disconnect=lambda: calls.append('disconnect'),
                 disable_cartesian_impedance=lambda: pytest.fail('unsafe zero-Kp gap'))
    close_backend_normally(backend)
    assert calls == ['fixed_hold', 'disconnect']


def test_repeated_ctrl_c_cannot_interrupt_cleanup_and_handler_is_restored():
    import signal
    from control_backends.force_ctrl.normal_shutdown import defer_sigint
    previous = signal.getsignal(signal.SIGINT)
    with defer_sigint():
        signal.getsignal(signal.SIGINT)(signal.SIGINT, None)
        with defer_sigint():
            signal.getsignal(signal.SIGINT)(signal.SIGINT, None)
        signal.getsignal(signal.SIGINT)(signal.SIGINT, None)
    assert signal.getsignal(signal.SIGINT) is previous


def test_ctrl_c_during_servo_startup_enable_does_not_call_disable(exit_controller):
    ctrl, calls, _, _ = exit_controller
    ctrl._started = ctrl._robot_connected = ctrl._robot_enabled = False
    ctrl._loop.running = False
    ctrl._channel, ctrl._skip_reset = 'offline', True
    ctrl._robot.connect = lambda: None
    ctrl._robot.set_normal_mode = lambda: None
    def interrupted_enable():
        calls.append('enable_attempt')
        raise KeyboardInterrupt()
    ctrl._robot.enable = interrupted_enable
    with pytest.raises(KeyboardInterrupt):
        ctrl.start()
    assert calls == ['enable_attempt', 'mit_joined', 'speed_zero',
                     'terminate', 'terminate', 'terminate', 'mode_j',
                     'device_current_q_hold', 'disconnect']


def test_ctrl_c_after_worker_start_before_started_flag_still_holds(exit_controller):
    ctrl, calls, clock, q = exit_controller
    ctrl._started = ctrl._robot_connected = ctrl._robot_enabled = False
    ctrl._loop.running = False
    ctrl._channel, ctrl._skip_reset = 'offline', True
    ctrl._startup_joint_margin = 0.
    ctrl._zero_gravity_init = False
    ctrl._robot.connect = lambda: None
    ctrl._robot.set_normal_mode = lambda: None
    ctrl._robot.enable = lambda: True
    ctrl._robot.get_joint_angles = lambda: NS(msg=q, timestamp=clock.now)
    ctrl._loop.update_commands = lambda cmd: None
    def interrupted_start():
        ctrl._loop.running = True
        raise KeyboardInterrupt()
    ctrl._loop.start = interrupted_start
    with pytest.raises(KeyboardInterrupt):
        ctrl.start()
    assert calls[0:2] == ['fixed_mit_hold', 'mit_joined']
    assert calls[-1] == 'disconnect'


def test_unified_emergency_shutdown_never_restores_normal_mode(exit_controller):
    ctrl, calls, clock, q = exit_controller
    session = _unified_session(ctrl, clock, q)
    ctrl._loop.request_emergency_stop = lambda reason: calls.append('emergency')
    session.emergency_stop()
    assert calls == ['emergency', 'mit_joined']
    assert not getattr(session, '_device_hold_confirmed', False)


def test_unified_log_failure_after_hold_does_not_fail_shutdown(exit_controller, monkeypatch):
    import control_backends.force_ctrl.controller as controller_module
    import real_nero_grasp.mit_trajectory_executor as module
    monkeypatch.setattr(module, 'time', controller_module.time)
    ctrl, calls, clock, q = exit_controller
    session = _unified_session(ctrl, clock, q)
    def broken_log():
        raise OSError("log status unavailable")
    session._finish_shutdown_diagnostics = broken_log
    session.shutdown()
    assert session._shutdown_complete and session._device_hold_confirmed
    assert 'device_current_q_hold' in calls
    assert 'disable' not in calls and 'emergency' not in calls


def test_unified_device_handoff_precedes_optional_log_cleanup(exit_controller, monkeypatch):
    import control_backends.force_ctrl.controller as controller_module
    import real_nero_grasp.mit_trajectory_executor as module
    monkeypatch.setattr(module, 'time', controller_module.time)
    ctrl, calls, clock, q = exit_controller
    session = _unified_session(ctrl, clock, q)
    ctrl._loop.stop_diagnostics = lambda: calls.append('log_cleanup')
    session.shutdown()
    assert calls.index('device_current_q_hold') < calls.index('log_cleanup')


def test_backend_ctrl_c_during_connect_uses_non_disabling_cleanup():
    calls = []
    class Controller:
        def __init__(self, **kwargs):
            pass
        def start(self):
            raise KeyboardInterrupt()
        def stop_keep_enabled(self):
            calls.append('enabled_exit')
        def stop(self):
            pytest.fail('Ctrl+C startup must not call disabling stop')
    backend = SdkForceCtrlBackend(dry_run=False, keep_enabled_on_disconnect=True,
                                  controller_cls=Controller)
    with pytest.raises(KeyboardInterrupt):
        backend.connect()
    assert calls == ['enabled_exit']


def test_grasp_cleanup_retains_backend_when_hold_is_not_confirmed():
    from real_nero_grasp.grasp_workflow import _stop_impedance_backend
    calls = []
    def fail():
        calls.append('disconnect_attempt')
        raise RuntimeError('hold not confirmed')
    backend = NS(prepare_shutdown=lambda: calls.append('fixed_hold'), disconnect=fail,
                 disable_cartesian_impedance=lambda: pytest.fail('must not zero Kp'))
    args = NS(_servo_backend=backend)
    with pytest.raises(RuntimeError, match='hold not confirmed'):
        _stop_impedance_backend(args)
    assert args._servo_backend is backend
    assert calls == ['fixed_hold', 'disconnect_attempt']


def test_device_handoff_never_sends_out_of_physical_range_target(exit_controller):
    from control_backends.force_ctrl.normal_shutdown import handoff_to_position_hold
    ctrl, calls, _, q = exit_controller
    q[6] = np.pi
    with pytest.raises(RuntimeError, match='outside physical limits'):
        handoff_to_position_hold(ctrl._robot, ctrl._loop.stop, check_fault=lambda: None)
    assert calls == []
