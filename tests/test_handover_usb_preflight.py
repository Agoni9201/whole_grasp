"""USB link rejection happens before the launcher can run any arm commands."""
import os
from pathlib import Path
import subprocess

import pytest

from tools.check_handover_usb import require_wrist_superspeed, usb_cameras


def camera(root, route, *, serial='CP0T963000S4', speed='480', product='Orbbec Gemini 335'):
    path = root/route
    path.mkdir()
    (path/'product').write_text(product)
    (path/'speed').write_text(speed)
    if serial is not None:
        (path/'serial').write_text(serial)


def test_two_cameras_on_usb2_are_reported_without_inventing_global_serial(tmp_path):
    camera(tmp_path, '1-3.1')
    camera(tmp_path, '1-3.3', product='Intel(R) RealSense(TM) Depth Camera 455', serial=None)
    devices = usb_cameras(tmp_path)
    assert len(devices) == 2 and devices[1]['serial'] is None
    with pytest.raises(RuntimeError, match='480M'):
        require_wrist_superspeed(devices, 'CP0T963000S4')


@pytest.mark.parametrize('speed', ['5000', '10000', '20000'])
def test_superspeed_wrist_does_not_require_external_rgb_only_camera_to_use_superspeed(tmp_path, speed):
    camera(tmp_path, '2-1', speed=speed)
    camera(tmp_path, '1-3.3', product='Intel RealSense D457', serial=None)
    result = require_wrist_superspeed(usb_cameras(tmp_path), 'CP0T963000S4')
    assert result['speed_mbps'] == float(speed)


def test_missing_wrist_does_not_continue_using_another_camera(tmp_path):
    camera(tmp_path, '2-1', serial='different', speed='5000')
    with pytest.raises(RuntimeError, match='found 0'):
        require_wrist_superspeed(usb_cameras(tmp_path), 'CP0T963000S4')


def test_launcher_exits_on_usb_preflight_failure_before_safe_grasp_wrapper(tmp_path):
    prefix = tmp_path/'env'
    (prefix/'bin').mkdir(parents=True)
    probe_args = tmp_path/'probe_args'
    probe = prefix/'bin/python'
    probe.write_text('#!/bin/sh\nprintf \'%s\\n\' "$@" > "$PROBE_ARGS"\nexit 2\n')
    probe.chmod(0o755)
    forbidden = tmp_path/'bash'
    forbidden.write_text('#!/bin/sh\ntouch "$FORBIDDEN_MARKER"\nexit 99\n')
    forbidden.chmod(0o755)
    marker = tmp_path/'unsafe_entry'
    env = dict(os.environ, CONDA_PREFIX=str(prefix), PATH=str(tmp_path)+os.pathsep+os.environ['PATH'],
               PROBE_ARGS=str(probe_args), FORBIDDEN_MARKER=str(marker))
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(['/bin/bash', str(root/'scripts/collect_handover_shadow.sh'), 'Left'], env=env)
    assert result.returncode == 2
    assert probe_args.read_text().splitlines() == ['tools/check_handover_usb.py', '--wrist-serial', 'CP0T963000S4']
    assert not marker.exists()
