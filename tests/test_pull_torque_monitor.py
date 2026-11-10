"""Sensor recording tests with no CAN and no controller commands."""
import json
import signal
import sys
import time
from types import SimpleNamespace as NS

import numpy as np

from tools import monitor_grasp_pull_torque as monitor


def test_motor_feedback_requires_fresh_finite_seven_joint_data():
    joints = NS(msg=[0.] * 7, timestamp=10.)
    motors = [NS(msg=NS(torque=float(i)), timestamp=10.) for i in range(7)]
    robot = NS(get_joint_angles=lambda: joints, get_motor_states=lambda i: motors[i - 1])
    sample = monitor.read_feedback(robot, 10.1)
    np.testing.assert_array_equal(sample['torque'], np.arange(7))
    np.testing.assert_allclose(sample['ages_ms'], 100.)
    motors[2].timestamp = 9.
    assert monitor.read_feedback(robot, 10.1) is None
    motors[2].timestamp = 10.; motors[4].msg.torque = float('nan')
    assert monitor.read_feedback(robot, 10.1) is None


def test_fixed_payload_baseline_separates_pull_and_rest_without_calling_it_external_force():
    baseline = np.array([.2, -4., .1, -4.5, 0., -.1, -.5])
    samples = []
    for i in range(15):
        torque = baseline.copy()
        if i >= 10:
            torque[1] += .7; torque[3] -= .3
        samples.append(dict(t_s=i / 10., torque=torque, q_deg=np.zeros(7),
            label='BASELINE' if i < 5 else 'REST' if i < 10 else 'PULL',
            ages_ms=np.full(7, 10.), skew_ms=1.))
    report, delta = monitor.summarize_samples(samples, .5)
    np.testing.assert_allclose(report['labels']['REST']['delta_rms_nm'], 0.)
    np.testing.assert_allclose(report['labels']['PULL']['delta_rms_nm'], [0., .7, 0., .3, 0., 0., 0.])
    np.testing.assert_allclose(report['baseline_torque_nm'], baseline)
    assert 'not external force' in report['semantics']


def test_complete_recording_uses_only_feedback_connect_disconnect_and_exports(tmp_path, monkeypatch):
    class ReadOnlyRobot:
        def connect(self): self.connected = True
        def disconnect(self): self.connected = False
        def get_joint_angles(self): return NS(msg=[0.] * 7, timestamp=time.time() - .01)
        def get_motor_states(self, index): return NS(msg=NS(torque=-.2 * index), timestamp=time.time() - .01)
    robot = ReadOnlyRobot()
    sdk = NS(AgxArmFactory=NS(create_arm=lambda config: robot), ArmModel=NS(NERO='nero'),
             NeroFW=NS(V111='v111'), create_agx_arm_config=lambda **kwargs: kwargs)
    monkeypatch.setitem(sys.modules, 'pyAgxArm', sdk)
    monkeypatch.setattr(sys, 'argv', ['monitor', '--duration', '.4', '--baseline-seconds', '.15',
                                    '--hz', '100', '--save-dir', str(tmp_path)])
    assert monitor.main() == 0
    assert not robot.connected
    output = next(tmp_path.iterdir())
    report = json.loads((output / 'summary.json').read_text())
    np.testing.assert_allclose(report['peak_abs_delta_nm'], 0.)
    assert report['baseline_samples'] >= 5
    assert all((output / name).stat().st_size > 0 for name in ('samples.csv', 'torque.png', 'torque.pdf'))


def test_sigint_callback_requests_exit_without_raising_inside_gui_and_preserves_export(tmp_path, monkeypatch):
    previous = signal.getsignal(signal.SIGINT)
    class ReadOnlyRobot:
        signal_returned = False
        def connect(self): self.connected = True; self.start = time.monotonic()
        def disconnect(self): self.connected = False
        def get_joint_angles(self): return NS(msg=[0.] * 7, timestamp=time.time() - .01)
        def get_motor_states(self, index):
            if index == 1 and not self.signal_returned and time.monotonic() - self.start > .3:
                signal.getsignal(signal.SIGINT)(signal.SIGINT, None)
                self.signal_returned = True
            return NS(msg=NS(torque=-.2 * index), timestamp=time.time() - .01)
    robot = ReadOnlyRobot()
    sdk = NS(AgxArmFactory=NS(create_arm=lambda config: robot), ArmModel=NS(NERO='nero'),
             NeroFW=NS(V111='v111'), create_agx_arm_config=lambda **kwargs: kwargs)
    monkeypatch.setitem(sys.modules, 'pyAgxArm', sdk)
    monkeypatch.setattr(sys, 'argv', ['monitor', '--duration', '2', '--baseline-seconds', '.15',
                                    '--hz', '100', '--save-dir', str(tmp_path)])
    assert monitor.main() == 0
    assert robot.signal_returned and not robot.connected
    assert signal.getsignal(signal.SIGINT) is previous
    output = next(tmp_path.iterdir())
    assert (output / 'summary.json').is_file() and (output / 'torque.pdf').is_file()
    assert len((output / 'samples.csv').read_text().splitlines()) < 100
