"""Bounded memory regressions with synthetic RGB/joints; no camera or CAN."""
from types import SimpleNamespace as NS

import numpy as np
import pytest

from servo.hand_position_memory import HandPositionMemory


def stable_memory():
    memory = HandPositionMemory()
    for i in range(13):
        memory.observe(([0, 0, .3], [0, 0, -1], [0, 1, 0]), i/30.)
    return memory


def test_memory_age_is_last_real_measurement_and_cannot_be_renewed_by_use():
    m = stable_memory()
    assert m.available(.5)
    assert m.available(.69)
    assert not m.available(.71)
    assert not m.available(.5)  # No retry/reset even if a caller's clock goes backward.


def test_continuous_stable_measurements_required_not_single_or_moving_old_point():
    m = HandPositionMemory()
    m.observe(([0, 0, .3], [0, 0, -1], [0, 1, 0]), 0.)
    assert not m.available(.01)
    for i in range(13):
        m.observe(([i*.003, 0, .3], [0, 0, -1], [0, 1, 0]), i/30.)
    assert not m.available(.45)


@pytest.mark.parametrize('kind', ['command', 'measured'])
def test_three_millimeter_path_budget_cannot_be_reset_without_new_depth(kind):
    m = stable_memory()
    m.start(np.zeros(3), np.zeros(3))
    if kind == 'command':
        # Accumulate path length, not just distance from the start.
        m.reference_sent([.0015, 0, 0]); m.reference_sent([0, 0, 0])
        assert m.remaining(np.zeros(3)) == pytest.approx(0)
    else:
        m.remaining([.0015, 0, 0])
        assert m.remaining([0, 0, 0]) == pytest.approx(0)
    assert not m.available(.5)


def test_current_image_must_agree_with_projected_base_memory():
    m = stable_memory()
    K = np.array([[600., 0, 320], [0, 600., 240], [0, 0, 1]])
    assert m.consistent_image([0, 0, .3], [320, 240], K)
    assert not m.consistent_image([0, 0, .3], [380, 240], K)
    assert not m.available(.5)


def memory_node(monkeypatch):
    import servo_delivery as delivery
    from real_nero_grasp.servo_cli import servo_parser
    from models.nero_model import NeroPinocchioModel
    from targets.hand_selection import HandRoles
    from targets.aruco_eye_in_hand_target import T_TCP_LINK7, load_transform_file
    q = np.array([.1, .2, -.1, 1.2, .05, -.1, .2])
    model = NeroPinocchioModel()
    T = T_TCP_LINK7@load_transform_file('config/2026-10-22_charuco_handeye.json')
    tcp, R = model.fk(q); R = R.copy()
    hand = tcp+R@(T[:3, 3]+T[:3, :3]@np.array([0., 0., .4]))
    target_tuple = (hand, np.array([0., 0., -1.]), np.array([0., 1., 0.]))
    roles = HandRoles(); roles.receiver = 'Left'; roles.presence = 'visible'
    target = NS(tcp_camera_transform=T, hand_roles=roles,
                last_diagnostics={}, last_observation=NS(handedness='Left', palm_pixel=(319.5, 239.5)))
    args = servo_parser().parse_args(['--target-source', 'mediapipe', '--gesture-handover',
        '--hand-camera-center', '--delivery-distance', '.05', '--gesture-preview-grip-width-mm', '40'])
    clock = NS(now=100., valid=True)
    monkeypatch.setattr(delivery, 'time', NS(time=lambda: clock.now, monotonic=lambda: clock.now))
    backend = delivery.FakeBackend(); backend.get_joint_angles = lambda: q.copy()
    node = delivery.ServoDeliveryNode(backend, args, model, target=target)
    node._engage(q, clock.now)
    node.state, node._target_lost, node._last_target_time = 'TRACK_HAND', False, clock.now
    K = np.array([[600., 0, 319.5], [0, 600., 239.5], [0, 0, 1.]])
    node._latest_info = NS(k=K.ravel())
    node.servo.set_camera_intrinsics(K, 640, 480)
    sequence = NS(value=0)

    def detect(*args):
        sequence.value += 1
        node._consumed_sequence = node._sdk_frame_sequence = sequence.value
        node._sdk_received_monotonic = clock.now
        node._detection_reason = 'ok' if clock.valid else 'invalid_palm_depth'
        return target_tuple if clock.valid else None

    node._detect_latest = detect
    for i in range(16):
        clock.now = 100.+i/30.
        node._servo_step()
    return node, clock, q


def test_delivery_depth_only_loss_is_bounded_and_never_establishes_readiness(monkeypatch):
    node, clock, q = memory_node(monkeypatch)
    stamp = node._hand_memory.stamp
    clock.valid = False
    remembered_frames = 0
    for i in range(1, 20):
        clock.now = stamp+i/30.
        node._servo_step()
        if node._preview_status == 'MEMORY_APPROACH':
            remembered_frames += 1
            assert np.linalg.norm(node._imp_cfg.target_twist[:3]) <= .01+1e-9
        assert node._hand_memory.command_travel <= .003+1e-9
        assert node._hand_memory.stamp == stamp
        assert not node._gesture.snapshot()['handover_ready']
    assert remembered_frames > 0
    assert node._preview_status == 'LOST'
    np.testing.assert_array_equal(node._imp_cfg.target_twist, 0)
    np.testing.assert_array_equal(node._imp_cfg.follow_kp, 0)
    assert node._gesture.still_elapsed == 0
    assert node.gripper is None


@pytest.mark.parametrize('reason', ['receiver_absent_2d', 'camera_frame_stale', 'receiver_acquiring_or_ambiguous'])
def test_memory_never_runs_for_missing_or_ambiguous_rgb_receiver(monkeypatch, reason):
    node, clock, q = memory_node(monkeypatch)
    clock.now += .03
    node._detection_reason = reason
    node._sdk_received_monotonic = clock.now
    previous = node._imp_smoothed_tcp.copy()
    assert not node._try_memory_approach(q, clock.now)
    np.testing.assert_array_equal(node._imp_smoothed_tcp, previous)


def test_memory_is_disabled_after_release_or_when_current_image_disagrees(monkeypatch):
    node, clock, q = memory_node(monkeypatch)
    clock.now += .03
    node._sdk_received_monotonic = clock.now
    node._detection_reason = 'invalid_palm_depth'
    node.target.last_observation.palm_pixel = (400, 240)
    assert not node._try_memory_approach(q, clock.now)
    assert node._hand_memory.reason == 'receiver_image_disagrees_with_memory'
    node._gesture.release_started = True
    assert not node._try_memory_approach(q, clock.now)


def test_duplicate_camera_frames_do_not_extend_memory_time_or_motion_budget(monkeypatch):
    import servo_delivery as delivery
    node, clock, q = memory_node(monkeypatch)
    stamp = node._hand_memory.stamp
    clock.valid = False
    clock.now += .03
    node._servo_step()
    assert node._preview_status == 'MEMORY_APPROACH'
    node._detect_latest = lambda *args: delivery._NO_NEW_FRAME
    clock.now = stamp+.31
    node._servo_step()
    assert node._hand_memory.blocked
    np.testing.assert_array_equal(node._imp_cfg.target_twist, 0)
    assert node._preview_reason == 'memory_expired'
