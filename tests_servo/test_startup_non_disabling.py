"""Startup refusal/failure must not disable motors; all devices are fakes."""

from types import SimpleNamespace as NS

import numpy as np
import pytest

import control_backends.force_ctrl.controller as module
from control_backends.force_ctrl.feedback_health import FeedbackStale
from control_backends.sdk_force_ctrl_backend import SdkForceCtrlBackend


@pytest.fixture
def startup(monkeypatch):
    state = NS(
        calls=[], now=100.0, q=np.array([.1, .2, .1, .5, .1, .1, .1]),
        read_error=None, connect_error=None, enable_error=None, loop_error=None,
        loop_starts_before_error=False, reads=0, read_hook=None,
        handoff_error=None, handoff_calls=0, position_stamps=None, motor_stamps=None,
        disconnect_error=None, fault_on_loop_error=False, loop_debug_error=None,
        emergency_send_error=None,
        handoff_result={'confirmed': True, 'method': 'fake_enabled_position_hold'},
    )

    class Monitor:
        def __init__(self, *args, **kwargs):
            pass

        def read(self, *args):
            state.calls.append('feedback')
            state.reads += 1
            if state.read_hook is not None:
                state.read_hook(state.reads)
            if state.read_error is not None:
                raise state.read_error
            return {
                'q': state.q.copy(),
                'position_timestamps': (np.full(7, state.now) if state.position_stamps is None
                                        else state.position_stamps.copy()),
                'motor_timestamps': (np.full(7, state.now) if state.motor_stamps is None
                                     else state.motor_stamps.copy()),
            }

    class Robot:
        def connect(self):
            state.calls.append('connect')
            if state.connect_error is not None:
                raise state.connect_error

        def disconnect(self):
            state.calls.append('disconnect')
            if state.disconnect_error is not None:
                raise state.disconnect_error

        def reset(self):
            state.calls.append('reset')

        def set_normal_mode(self):
            state.calls.append('normal_mode')

        def enable(self):
            state.calls.append('enable')
            if state.enable_error is not None:
                raise state.enable_error
            return True

        def disable(self):
            pytest.fail('Startup cleanup must never send disable()')

        def get_joint_angles(self):
            pytest.fail('Startup must use per-axis fresh feedback, not SDK cached angles')

        def electronic_emergency_stop(self):
            state.calls.append('electronic_stop')

    class Loop:
        running = False
        terminal_fault = False
        _command_stop_latched = False
        _electronic_stop_sent = False
        stop_target = None

        def get_debug_state(self):
            return {'last_error': state.loop_debug_error}

        def update_commands(self, command):
            state.calls.append('initial_command')
            self.command = command

        def start(self):
            state.calls.append('loop_start')
            if state.loop_error is not None:
                self.running = state.loop_starts_before_error
                if state.fault_on_loop_error:
                    self.terminal_fault = True
                    state.loop_debug_error = 'feedback lost before initial send'
                raise state.loop_error
            self.running = True

        def hold_for_controlled_stop(self, command):
            state.calls.append('fixed_mit_hold')
            self.command = command
            self.stop_target = command.pos.copy()
            self._command_stop_latched = True

        def get_controlled_stop_status(self):
            return {'target_q': self.stop_target,
                    'transition_complete': self._command_stop_latched,
                    'transition_duration_s': 0.0,
                    'transition_completed_monotonic': state.now,
                    'phase': 'hold' if self._command_stop_latched else 'idle'}

        def stop(self):
            state.calls.append('sender_joined')
            self.running = False

        stop_sender = stop

        def stop_diagnostics(self):
            state.calls.append('diagnostics_closed')

        def request_emergency_stop(self, reason):
            state.calls.append('electronic_stop')
            if state.emergency_send_error is not None:
                raise state.emergency_send_error
            self._electronic_stop_sent = True

    def handoff(robot, stop_sender, *, check_fault, **kwargs):
        state.handoff_calls += 1
        state.calls.append('handoff_begin')
        check_fault()
        if state.handoff_error is not None:
            raise state.handoff_error
        stop_sender()
        assert not ctrl._loop.running
        if isinstance(state.handoff_result, dict) and state.handoff_result.get('confirmed') is True:
            state.calls.append('device_hold_confirmed')
        return state.handoff_result

    clock = NS(
        time=lambda: state.now,
        monotonic=lambda: state.now,
        sleep=lambda seconds: setattr(state, 'now', state.now + seconds),
    )
    monkeypatch.setattr(module, 'time', clock)
    monkeypatch.setattr(module, 'FeedbackHealthMonitor', Monitor)
    import control_backends.force_ctrl.normal_shutdown as shared
    monkeypatch.setattr(shared, 'time', clock)
    monkeypatch.setattr(shared, 'FeedbackHealthMonitor', Monitor)
    monkeypatch.setattr(module, 'handoff_to_position_hold', handoff)
    ctrl = module.NeroForceController.__new__(module.NeroForceController)
    ctrl._robot, ctrl._loop = Robot(), Loop()
    ctrl._channel = 'offline-fake-only'
    ctrl._started = ctrl._robot_connected = ctrl._robot_enabled = False
    ctrl._emergency_stopped = False
    ctrl._startup_control_touched = False
    ctrl._skip_reset = False
    ctrl._startup_joint_margin = .08
    ctrl._zero_gravity_init = False
    ctrl._exit_hold_q = None
    ctrl._kp, ctrl._kd = module.DEFAULT_KP.copy(), module.DEFAULT_KD.copy()
    state.controller = ctrl
    return state


def assert_no_takeover(state):
    assert not {'reset', 'normal_mode', 'enable', 'initial_command',
                'loop_start', 'handoff_begin', 'electronic_stop'} & set(state.calls)
    assert state.calls[-1] == 'disconnect'
    assert not state.controller._robot_connected


def test_measured_j4_125_9_degrees_is_refused_before_any_control_write(startup):
    startup.q[3] = 2.1975
    with pytest.raises(RuntimeError):
        startup.controller.start()
    assert startup.calls[0:2] == ['connect', 'feedback']
    assert_no_takeover(startup)
    startup.controller.cleanup_failed_start()
    assert startup.calls.count('disconnect') == 1


@pytest.mark.parametrize('problem', [
    FeedbackStale('feedback stale'),
    RuntimeError('MIT per-axis feedback requires the SDK parser cache'),
    RuntimeError('MIT position feedback unavailable: joint_34'),
])
def test_unavailable_or_stale_preflight_never_takes_control(startup, problem):
    startup.read_error = problem
    with pytest.raises(RuntimeError):
        startup.controller.start()
    assert_no_takeover(startup)
    assert startup.now < 110.0  # Any feedback-wait policy must remain bounded.


def test_zero_margin_does_not_turn_off_physical_limit_check(startup):
    startup.controller._startup_joint_margin = 0.0
    startup.q[3] = module.NERO_JOINT_LIMITS[3, 1] + .001
    with pytest.raises(RuntimeError):
        startup.controller.start()
    assert_no_takeover(startup)


def test_ctrl_c_before_takeover_only_disconnects(startup):
    startup.read_error = KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt):
        startup.controller.start()
    assert_no_takeover(startup)


@pytest.mark.parametrize('problem', [RuntimeError('partial connection failed'), KeyboardInterrupt()])
def test_partial_connection_failure_releases_receiver_without_mutation(startup, problem):
    startup.connect_error = problem
    with pytest.raises(type(problem)):
        startup.controller.start()
    assert startup.calls == ['connect', 'disconnect']
    startup.controller.cleanup_failed_start()
    assert startup.calls == ['connect', 'disconnect']


def test_transient_missing_frame_recovers_before_takeover(startup):
    def feedback_hook(count):
        startup.read_error = RuntimeError('joint_34 not received yet') if count == 1 else None

    startup.read_hook = feedback_hook
    startup.controller.start()
    first_write = startup.calls.index('reset')
    assert startup.calls[:first_write] == ['connect', 'feedback', 'feedback']
    assert startup.controller._started
    assert 'disconnect' not in startup.calls and 'electronic_stop' not in startup.calls


@pytest.mark.parametrize('bad_q', [np.array([np.nan] * 7), np.zeros(6)])
def test_invalid_initial_feedback_is_rejected_without_writes(startup, bad_q):
    startup.q = bad_q
    with pytest.raises(RuntimeError):
        startup.controller.start()
    assert_no_takeover(startup)


def test_successful_start_preflights_before_reset_and_uses_post_enable_feedback(startup):
    q_after_enable = startup.q + .01

    def feedback_hook(_count):
        if 'enable' in startup.calls:
            startup.q = q_after_enable.copy()

    startup.read_hook = feedback_hook
    startup.controller.start()
    assert startup.calls.index('feedback') < startup.calls.index('reset')
    assert startup.calls.index('reset') < startup.calls.index('normal_mode')
    enable_index = startup.calls.index('enable')
    command_index = startup.calls.index('initial_command')
    assert 'feedback' in startup.calls[enable_index + 1:command_index]
    np.testing.assert_array_equal(startup.controller._loop.command.pos, q_after_enable)
    assert startup.controller._started and startup.controller._loop.running
    assert 'disconnect' not in startup.calls and 'electronic_stop' not in startup.calls


def test_post_enable_margin_violation_does_not_start_mit_and_keeps_enabled(startup):
    def feedback_hook(_count):
        if 'enable' in startup.calls:
            startup.q[3] = module.NERO_JOINT_LIMITS[3, 1] - .04

    startup.read_hook = feedback_hook
    with pytest.raises(RuntimeError, match='Unsafe startup posture'):
        startup.controller.start()
    assert startup.calls.count('enable') == 1
    assert 'initial_command' not in startup.calls and 'loop_start' not in startup.calls
    assert startup.calls.index('device_hold_confirmed') < startup.calls.index('disconnect')
    assert startup.controller._robot_enabled and not startup.controller._robot_connected
    assert 'electronic_stop' not in startup.calls


@pytest.mark.parametrize('stream', ['position_stamps', 'motor_stamps'])
@pytest.mark.parametrize('axis', range(7))
def test_every_axis_must_have_new_feedback_after_enable(startup, stream, axis):
    def feedback_hook(_count):
        if 'enable' in startup.calls:
            stamps = np.full(7, startup.now)
            stamps[axis] = 100.0  # Still the pre-enable frame, unlike the other axes.
            setattr(startup, stream, stamps)

    startup.read_hook = feedback_hook
    with pytest.raises(RuntimeError, match='new per-axis feedback after enable'):
        startup.controller.start()
    assert startup.calls.count('enable') == 1
    assert 'initial_command' not in startup.calls and 'loop_start' not in startup.calls
    assert startup.calls.count('device_hold_confirmed') == 1
    assert 'electronic_stop' not in startup.calls
    assert startup.now < 104.0


@pytest.mark.parametrize('problem', [RuntimeError('enable transmit then fail'), KeyboardInterrupt()])
def test_enable_throws_after_transmit_handoffs_even_before_enabled_flag(startup, problem):
    startup.enable_error = problem
    with pytest.raises(type(problem)):
        startup.controller.start()
    assert startup.handoff_calls == 1
    assert 'device_hold_confirmed' in startup.calls
    assert startup.calls.index('device_hold_confirmed') < startup.calls.index('disconnect')
    assert startup.calls.count('enable') == 1
    assert 'initial_command' not in startup.calls
    assert 'electronic_stop' not in startup.calls
    startup.controller.cleanup_failed_start()
    assert startup.handoff_calls == 1 and startup.calls.count('disconnect') == 1


@pytest.mark.parametrize('sender_started', [False, True])
@pytest.mark.parametrize('problem', [RuntimeError('loop start interrupted'), KeyboardInterrupt()])
def test_partial_loop_start_cleans_up_with_verified_hold(startup, sender_started, problem):
    startup.loop_error = problem
    startup.loop_starts_before_error = sender_started
    with pytest.raises(type(problem)):
        startup.controller.start()
    assert startup.handoff_calls == 1
    if sender_started:
        assert 'fixed_mit_hold' in startup.calls
        assert startup.calls.index('fixed_mit_hold') < startup.calls.index('sender_joined')
    assert startup.calls.index('device_hold_confirmed') < startup.calls.index('disconnect')
    assert not startup.controller._loop.running
    assert 'electronic_stop' not in startup.calls


def test_handoff_untrusted_feedback_escalates_without_disable_or_reenable(startup):
    startup.enable_error = RuntimeError('enable transmit then fail')
    startup.handoff_error = FeedbackStale('handoff feedback is untrusted')
    with pytest.raises(RuntimeError):
        startup.controller.start()
    assert startup.calls.count('electronic_stop') == 1
    assert startup.calls.count('enable') == 1
    assert startup.calls.count('reset') == 1
    assert 'device_hold_confirmed' not in startup.calls
    assert startup.controller._emergency_stopped
    calls = startup.calls.copy()
    with pytest.raises(RuntimeError):
        startup.controller.cleanup_failed_start()
    assert startup.calls == calls  # No second handoff, e-stop, or hidden retry.


@pytest.mark.parametrize('problem', [RuntimeError('enable failed'), KeyboardInterrupt()])
def test_backend_double_cleanup_prefers_non_disabling_even_when_option_false(startup, problem):
    startup.enable_error = problem
    backend = SdkForceCtrlBackend(
        dry_run=False, keep_enabled_on_disconnect=False,
        controller_cls=lambda **kwargs: startup.controller,
    )
    with pytest.raises(type(problem)):
        backend.connect()
    assert startup.handoff_calls == 1
    assert startup.calls.count('disconnect') == 1
    assert startup.calls.count('enable') == 1
    assert not backend.connected and backend._force_ctrl is None
    assert 'electronic_stop' not in startup.calls


def test_backend_preflight_refusal_does_not_reenter_old_disabling_stop(startup):
    startup.q[3] = 2.1975
    backend = SdkForceCtrlBackend(
        dry_run=False, keep_enabled_on_disconnect=False,
        controller_cls=lambda **kwargs: startup.controller,
    )
    with pytest.raises(RuntimeError):
        backend.connect()
    assert_no_takeover(startup)
    assert startup.calls.count('disconnect') == 1
    assert backend._force_ctrl is None and not backend.connected


def test_backend_cleanup_failure_is_latched_across_disconnect(startup):
    startup.enable_error = RuntimeError('enable transmit then fail')
    startup.handoff_error = FeedbackStale('handoff feedback is untrusted')
    backend = SdkForceCtrlBackend(
        dry_run=False, keep_enabled_on_disconnect=False,
        controller_cls=lambda **kwargs: startup.controller,
    )
    with pytest.raises(RuntimeError):
        backend.connect()
    assert backend._force_ctrl is startup.controller
    assert startup.calls.count('electronic_stop') == 1
    calls = startup.calls.copy()
    with pytest.raises(RuntimeError):
        backend.disconnect()
    assert startup.calls == calls


@pytest.mark.parametrize('request_fails', [False, True])
def test_terminal_fault_during_start_requests_missing_fault_stop_once(startup, request_fails):
    startup.loop_error = RuntimeError('initial sender feedback lost')
    startup.fault_on_loop_error = True
    startup.loop_starts_before_error = True
    if request_fails:
        startup.emergency_send_error = RuntimeError('fault-stop transmit failed')
    with pytest.raises(RuntimeError):
        startup.controller.start()
    assert startup.calls.count('electronic_stop') == 1
    assert startup.calls.count('enable') == 1 and startup.calls.count('reset') == 1
    assert 'handoff_begin' not in startup.calls and 'fixed_mit_hold' not in startup.calls
    assert 'device_hold_confirmed' not in startup.calls
    assert startup.controller._emergency_stopped
    if request_fails:
        assert startup.controller._robot_connected
        assert startup.controller._startup_cleanup_error is not None
        calls = startup.calls.copy()
        with pytest.raises(RuntimeError):
            startup.controller.cleanup_failed_start()
        assert startup.calls == calls


def test_disconnect_failure_after_verified_hold_never_requests_estop(startup):
    startup.enable_error = RuntimeError('enable transmit then fail')
    startup.disconnect_error = RuntimeError('receiver teardown failed')
    backend = SdkForceCtrlBackend(
        dry_run=False, keep_enabled_on_disconnect=False,
        controller_cls=lambda **kwargs: startup.controller,
    )
    with pytest.raises(RuntimeError, match='cleanup'):
        backend.connect()
    assert startup.controller._device_hold_confirmed
    assert startup.controller._robot_enabled
    assert startup.calls.count('device_hold_confirmed') == 1
    assert startup.calls.count('disconnect') == 1
    assert 'electronic_stop' not in startup.calls
    assert backend._force_ctrl is startup.controller and backend.connected
    calls = startup.calls.copy()
    with pytest.raises(RuntimeError):
        backend.disconnect()
    assert startup.calls == calls


@pytest.mark.parametrize('result', [None, {'confirmed': False}])
def test_unconfirmed_handoff_result_cannot_masquerade_as_enabled_hold(startup, result):
    startup.enable_error = RuntimeError('enable transmit then fail')
    startup.handoff_result = result
    with pytest.raises(RuntimeError):
        startup.controller.start()
    assert not getattr(startup.controller, '_device_hold_confirmed', False)
    assert startup.calls.count('electronic_stop') == 1
    assert 'device_hold_confirmed' not in startup.calls and 'disconnect' not in startup.calls
    assert startup.calls.count('enable') == 1 and startup.calls.count('reset') == 1
    calls = startup.calls.copy()
    with pytest.raises(RuntimeError):
        startup.controller.cleanup_failed_start()
    assert startup.calls == calls
