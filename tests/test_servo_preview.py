"""Read-only preview tests. No display, subprocess, camera, or CAN is opened."""
from types import SimpleNamespace as NS
from unittest.mock import Mock
from pathlib import Path
import os
import pickle
import struct
import sys
import threading

import numpy as np
import pytest

from cameras.sdk_loop import Image
from cameras.servo_preview import (
    ServoPreview, overlay_lines, draw_payload, _FrameReader, _LatestFrameReceiver,
)
from real_nero_grasp.servo_cli import servo_parser
from servo_delivery import ServoDeliveryNode


def payload():
    return {
        'width': 320, 'height': 240, 'encoding': 'bgr8', 'data': bytes(320 * 240 * 3),
        'received_monotonic': 10., 'updated_monotonic': 10.,
        'status': 'LOST', 'stage': 'TRACK_HAND', 'reason': 'invalid_palm_depth',
        'landmarks': None, 'palm_pixel': None, 'camera_xyz_mm': None,
        'depth_points': 2, 'required_depth_points': 3, 'base_xyz_mm': None,
        'distance_m': .20, 'error_mm': '--',
    }


def test_preview_is_opt_in_and_works_with_follow_only_parser():
    parser = servo_parser()
    assert not parser.parse_args([]).servo_preview
    args = parser.parse_args(['--servo-preview', '--servo-follow-only'])
    assert args.servo_preview and args.servo_follow_only


def test_full_pipeline_cli_keeps_hand_preview_and_follow_only(monkeypatch):
    from real_nero_grasp.cli import parse_args
    config = Path(__file__).resolve().parents[1] / 'config' / 'pipeline.yaml'
    monkeypatch.setattr(sys, 'argv', [
        'main.py', '--config', str(config), '--phase', 'all', '--execute',
        '--target-class', 'bottle', '--active-view', '--no-active-multiview-gate',
        '--active-side-view-count', '2', '--trajectory-executor', 'mit',
        '--enable-servo-delivery', '--target-source', 'mediapipe',
        '--servo-follow-only', '--servo-preview', '--delivery-distance', '.20',
        '--max-linear-vel', '.01', '--max-joint-vel', '.03',
        '--impedance-target-linear-vel', '.01',
    ])
    args = parse_args()
    assert args.phase == 'all' and args.execute and args.enable_servo_delivery
    assert args.servo_preview and args.servo_follow_only
    assert args.target_source == 'mediapipe' and args.trajectory_executor == 'mit'
    assert args.delivery_distance == .20 and args.max_linear_vel == .01


def test_handoff_passes_preview_flag_and_reuses_existing_backend(monkeypatch):
    import handoff.grasp_to_servo_handoff as handoff
    import servo_delivery as module
    backend = NS()
    args = NS(execute=True, servo_preview=True, servo_follow_only=True,
              _servo_backend=backend)
    monkeypatch.setattr(module, 'validate_servo_args', lambda args: None)
    run = Mock(return_value=0)
    monkeypatch.setattr(module, 'run_servo_delivery', run)
    monkeypatch.setattr(handoff, 'handoff_after_lift',
                        lambda *a: pytest.fail('must reuse the existing backend'))
    assert handoff.delivery_after_grasp(NS(), args) == 0
    run.assert_called_once_with(args, backend=backend)
    assert run.call_args.args[0].servo_preview


@pytest.mark.parametrize('enabled', [False, True])
def test_servo_entry_opens_preview_only_if_requested_without_new_backend(monkeypatch, enabled):
    import servo_delivery as module
    import cameras.realsense as camera
    import cameras.servo_preview as display
    events = []
    preview = NS(start=lambda: events.append('preview_start'),
                 close=lambda: events.append('preview_close'))
    backend = NS(disconnect=lambda: events.append('backend_exit'))
    args = NS(servo_preview=enabled)
    monkeypatch.setattr(module, 'validate_servo_args', lambda args: None)
    monkeypatch.setattr(module, 'NeroPinocchioModel', lambda: NS(clone=lambda: NS()))
    monkeypatch.setattr(module, 'build_target', lambda *a: NS(close=lambda: None))
    monkeypatch.setattr(camera, 'camera_for', lambda args: NS(read=lambda: events.append('camera_read')))
    monkeypatch.setattr(camera, 'close_all', lambda: None)
    monkeypatch.setattr(display, 'ServoPreview', lambda: preview)
    def make_node(existing, received, model, *, target, preview=None):
        assert existing is backend and received is args
        assert (preview is not None) is enabled
        events.append('servo_node')
        return NS(prepare_shutdown=lambda: events.append('hold'))
    monkeypatch.setattr(module, 'ServoDeliveryNode', make_node)
    monkeypatch.setattr(module.sdk_loop, 'init', lambda **kw: None)
    monkeypatch.setattr(module.sdk_loop, 'spin', lambda node: events.append('servo_spin'))
    monkeypatch.setattr(module.sdk_loop, 'shutdown', lambda: None)
    assert module.run_servo_delivery(args, backend=backend) == 0
    assert events[0] == 'camera_read'
    assert ('preview_start' in events) is enabled
    if enabled:
        assert events.index('preview_start') < events.index('servo_spin')
        assert events.index('hold') < events.index('preview_close')


def test_full_queue_replaces_old_pending_frame_without_waiting_or_accumulating():
    preview = ServoPreview()
    preview.publish({'sequence': 1})
    preview.publish({'sequence': 2})
    assert preview._pending.qsize() == 1
    assert preview._pending.get_nowait() == {'sequence': 2}


def test_overlay_explains_depth_loss_and_distinguishes_coordinate_frames():
    data = payload()
    lines = '\n'.join(overlay_lines(data, 10.1))
    assert 'LOST' in lines and 'invalid_palm_depth' in lines
    assert 'Depth points: 2/5 (need 3)' in lines
    assert 'camera XYZ mm: unavailable' in lines
    assert 'base XYZ mm: unavailable' in lines


@pytest.mark.parametrize('field', ['received_monotonic', 'updated_monotonic'])
def test_stale_image_or_status_is_never_presented_as_current_tracking(field):
    data = payload()
    data['status'] = 'TRACKING'
    data[field] = 9.0
    assert 'STALE DISPLAY' in overlay_lines(data, 10.1)[0]


def test_depth_invalid_hand_still_draws_landmarks_without_a_3d_point():
    data = payload()
    data['landmarks'] = [[20 + i * 3, 100] for i in range(21)]
    data['palm_pixel'] = (50, 100)
    visual = draw_payload(data, 10.1)  # Only drawing, no imshow/namedWindow.
    assert visual.shape == (720 + 224, 960, 3)
    assert visual[:720].any()
    assert data['camera_xyz_mm'] is None


def _preview_node(monkeypatch):
    import servo_delivery as module
    monkeypatch.setattr(module.time, 'monotonic', lambda: 10.0)
    node = ServoDeliveryNode.__new__(ServoDeliveryNode)
    node._preview = NS(publish=Mock(), poll=lambda: None)
    node._preview_last_sent = 0.
    node._latest_image = Image(320, 240, 'bgr8', 960, bytes(320 * 240 * 3))
    node._consumed_sequence = node._sdk_frame_sequence = 7
    node._sdk_received_monotonic = 9.95
    node._preview_status, node._preview_reason = 'HOLD', 'target_rejected_hold'
    node._preview_base, node._preview_error_mm = None, '--'
    node.args = NS(hand_min_depth_landmarks=3)
    node.servo = NS(params=NS(desired_distance=.2))
    node.state = 'TRACK_HAND'
    node.target = NS(last_observation=NS(
        palm_center_camera_m=None, landmarks_pixel=np.zeros((21, 2), dtype=int),
        palm_pixel=(2, 2), valid_depth_landmarks=2), detect=Mock())
    return node


def test_preview_uses_same_frame_observation_and_never_runs_detector(monkeypatch):
    node = _preview_node(monkeypatch)
    node._publish_preview()
    data = node._preview.publish.call_args.args[0]
    assert data['landmarks'] is not None and data['camera_xyz_mm'] is None
    assert data['data'] is node._latest_image.data
    assert data['depth_points'] == 2
    assert data['reason'] == 'target_rejected_hold'
    node.target.detect.assert_not_called()
    node._publish_preview()
    node._preview.publish.assert_called_once()  # Duplicate tick is rate limited.


def test_preview_can_publish_new_frame_before_old_100ms_interval(monkeypatch):
    import servo_delivery as module
    node = _preview_node(monkeypatch)
    node._publish_preview()
    monkeypatch.setattr(module.time, 'monotonic', lambda: 10.04)
    node._sdk_frame_sequence += 1
    node._publish_preview()
    assert node._preview.publish.call_count == 2


def test_observation_from_old_frame_is_not_drawn_on_new_image(monkeypatch):
    node = _preview_node(monkeypatch)
    node._sdk_frame_sequence += 1
    node._publish_preview()
    assert node._preview.publish.call_args.args[0]['landmarks'] is None


def test_quit_window_requests_normal_exit_before_next_servo_update():
    node = ServoDeliveryNode.__new__(ServoDeliveryNode)
    node._preview = NS(poll=lambda: 'quit')
    node._servo_step = Mock()
    with pytest.raises(SystemExit) as result:
        node._on_timer()
    assert result.value.code == 0
    node._servo_step.assert_not_called()


def test_crashed_window_does_not_silently_continue_blind_tracking():
    node = ServoDeliveryNode.__new__(ServoDeliveryNode)
    node._preview = NS(poll=lambda: 'error: preview process exited')
    node._servo_step = Mock()
    with pytest.raises(RuntimeError, match='exiting servo'):
        node._on_timer()
    node._servo_step.assert_not_called()


def test_no_desktop_fails_before_launching_preview_worker(monkeypatch):
    import cameras.servo_preview as module
    monkeypatch.setattr(module.sys, 'platform', 'linux')
    monkeypatch.delenv('DISPLAY', raising=False)
    monkeypatch.delenv('WAYLAND_DISPLAY', raising=False)
    monkeypatch.setattr(module.subprocess, 'Popen', lambda *a, **k: pytest.fail('must not launch'))
    with pytest.raises(RuntimeError, match='desktop display'):
        ServoPreview().start()


def test_display_worker_is_isolated_from_terminal_ctrl_c(monkeypatch):
    import cameras.servo_preview as module
    monkeypatch.setenv('DISPLAY', ':test')
    process = NS(poll=lambda: None)
    launch = Mock(return_value=process)
    monkeypatch.setattr(module.subprocess, 'Popen', launch)
    monkeypatch.setattr(module.threading, 'Thread', Mock())
    preview = ServoPreview()
    preview._ready = NS(wait=lambda timeout: True)
    preview.start()
    assert launch.call_args.kwargs['start_new_session'] is True


def _packet(value):
    data = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
    return struct.pack('!I', len(data)) + data


def test_preview_input_handles_partial_packets_without_waiting_for_stdin():
    read_fd, write_fd = os.pipe()
    try:
        reader = _FrameReader(read_fd)
        packet = _packet({'sequence': 1})
        assert reader.poll() == (None, False)
        os.write(write_fd, packet[:2])
        assert reader.poll() == (None, False)
        os.write(write_fd, packet[2:8])
        assert reader.poll() == (None, False)
        os.write(write_fd, packet[8:] + _packet({'sequence': 2}))
        assert reader.poll() == ({'sequence': 2}, False)
        os.close(write_fd)
        write_fd = None
        assert reader.poll() == (None, True)
    finally:
        os.close(read_fd)
        if write_fd is not None:
            os.close(write_fd)


@pytest.mark.parametrize('size', [0, 32 * 1024 * 1024 + 1])
def test_preview_input_rejects_unbounded_packets(size):
    read_fd, write_fd = os.pipe()
    try:
        reader = _FrameReader(read_fd)
        os.write(write_fd, struct.pack('!I', size))
        with pytest.raises(ValueError, match='frame size'):
            reader.poll()
    finally:
        os.close(read_fd)
        os.close(write_fd)


def test_preview_input_reports_interrupted_frame_instead_of_blocking():
    read_fd, write_fd = os.pipe()
    try:
        reader = _FrameReader(read_fd)
        os.write(write_fd, _packet({'sequence': 1})[:-1])
        os.close(write_fd)
        with pytest.raises(EOFError, match='incomplete'):
            reader.poll()
    finally:
        os.close(read_fd)


def test_window_quit_joins_raw_receiver_without_any_thread_holding_buffered_stdin(monkeypatch):
    import cameras.servo_preview as module
    read_fd, write_fd = os.pipe()
    gui = NS(
        setNumThreads=Mock(), namedWindow=Mock(), imshow=Mock(),
        waitKey=Mock(side_effect=[-1, ord('q')]),
        getWindowProperty=Mock(return_value=1), destroyAllWindows=Mock(),
        WINDOW_NORMAL=0, WND_PROP_VISIBLE=0,
    )
    monkeypatch.setitem(sys.modules, 'cv2', gui)
    monkeypatch.setattr(module.sys, 'stdin', NS(fileno=lambda: read_fd))
    receivers = []
    def make_receiver(fd):
        receiver = _LatestFrameReceiver(fd)
        receivers.append(receiver)
        return receiver
    monkeypatch.setattr(module, '_LatestFrameReceiver', make_receiver)
    try:
        module._worker()
        gui.destroyAllWindows.assert_called_once_with()
        assert len(receivers) == 1 and not receivers[0]._thread.is_alive()
    finally:
        os.close(read_fd)
        os.close(write_fd)


def test_full_resolution_frame_drains_even_when_gui_does_not_poll():
    read_fd, write_fd = os.pipe()
    receiver = _LatestFrameReceiver(read_fd)
    finished = threading.Event()
    errors = []
    data = _packet({'sequence': 1, 'data': bytes(1280 * 720 * 3)})
    def write():
        try:
            remaining = memoryview(data)
            while remaining:
                remaining = remaining[os.write(write_fd, remaining):]
        except OSError as exc:
            errors.append(exc)
        finally:
            os.close(write_fd)
            finished.set()
    writer = threading.Thread(target=write, daemon=True)
    receiver.start()
    writer.start()
    try:
        # A GUI-bound receiver deadlocks here after filling the small OS pipe.
        assert finished.wait(3.0)
        receiver._thread.join(timeout=1.0)
        latest, ended = receiver.poll()
        assert latest['sequence'] == 1 and len(latest['data']) == 1280 * 720 * 3
        assert ended and not errors
        assert not receiver._thread.is_alive()
    finally:
        receiver.close()
        os.close(read_fd)
        writer.join(timeout=1.0)


def test_raw_receiver_shutdown_does_not_wait_for_a_complete_frame_or_eof():
    read_fd, write_fd = os.pipe()
    receiver = _LatestFrameReceiver(read_fd)
    receiver.start()
    try:
        os.write(write_fd, _packet({'sequence': 1})[:6])
        receiver.close()
        assert not receiver._thread.is_alive()
    finally:
        receiver.close()
        os.close(read_fd)
        os.close(write_fd)


def test_raw_receiver_propagates_invalid_packet_without_touching_gui():
    read_fd, write_fd = os.pipe()
    receiver = _LatestFrameReceiver(read_fd)
    receiver.start()
    try:
        os.write(write_fd, struct.pack('!I', 0))
        receiver._thread.join(timeout=1.0)
        assert not receiver._thread.is_alive()
        with pytest.raises(ValueError, match='frame size'):
            receiver.poll()
    finally:
        receiver.close()
        os.close(read_fd)
        os.close(write_fd)


def test_raw_receiver_retains_only_newest_frame_when_gui_is_busy():
    read_fd, write_fd = os.pipe()
    receiver = _LatestFrameReceiver(read_fd)
    receiver.start()
    try:
        os.write(write_fd, b''.join(_packet({'sequence': i}) for i in range(20)))
        os.close(write_fd)
        write_fd = None
        receiver._thread.join(timeout=1.0)
        latest, ended = receiver.poll()
        assert latest == {'sequence': 19} and ended
        assert receiver.poll() == (None, True)
    finally:
        receiver.close()
        os.close(read_fd)
        if write_fd is not None:
            os.close(write_fd)


def test_parent_cleanup_allows_graceful_eof_exit_before_signalling():
    preview = ServoPreview()
    process = NS(wait=Mock(return_value=0), terminate=Mock(), kill=Mock(),
                 stdin=Mock(), stdout=Mock())
    preview._process = process
    preview.close()
    assert preview._stop.is_set() and preview._process is None
    process.wait.assert_called_once_with(timeout=1.0)
    process.terminate.assert_not_called()
    process.kill.assert_not_called()
    process.stdin.close.assert_called_once_with()
    process.stdout.close.assert_called_once_with()


def test_parent_cleanup_bounds_wait_for_unresponsive_gui():
    import cameras.servo_preview as module
    preview = ServoPreview()
    process = NS(wait=Mock(side_effect=[module.subprocess.TimeoutExpired('preview', 1), 0]),
                 terminate=Mock(), kill=Mock(), stdin=Mock(), stdout=Mock())
    preview._process = process
    preview.close()
    process.terminate.assert_called_once_with()
    process.kill.assert_not_called()
    assert preview._process is None
