"""Cached diagnostics; no camera, CAN, or controller. Only the log worker may run."""

from datetime import datetime
from types import SimpleNamespace as NS
from unittest.mock import Mock, call

import numpy as np
import pytest

from servo.diagnostics import ServoDiagnostics


class _Model:
    def fk(self, q):
        return np.asarray(q, dtype=float)[:3].copy(), np.eye(3)


def _sample(*, q=None, debug=None, servo_out=True):
    return {
        "q": np.zeros(7) if q is None else np.asarray(q, dtype=float),
        "dq": np.full(7, 0.002),
        "debug": {} if debug is None else debug,
        "servo_out": {
            "target_tcp": np.array([0.1, 0.0, 0.0]),
            "error": np.array([0.1, 0.0, 0.0]),
            "qdot": np.full(7, 0.003),
        } if servo_out else None,
        "detection": None,
        "reference_updated": bool(servo_out),
    }


def _cfg(target=None):
    return NS(
        target_tcp=np.array([0.015, 0.0, 0.0]) if target is None else np.asarray(target),
        target_twist=np.array([0.01, 0.0, 0.0, 0.0, 0.0, 0.0]),
    )


def _observe(diag, sample=None, *, now=10.0, cfg=None, status="TRACKING", **kwargs):
    return diag.observe(
        _sample() if sample is None else sample,
        model=_Model(), cfg=_cfg() if cfg is None else cfg,
        status=status, reason="test", stage="TRACK_HAND",
        now=now, wall_time=1792461600.0, **kwargs,
    )


def _diag(tmp_path):
    writer = Mock()
    writer.dropped, writer.error = 0, None
    diag = ServoDiagnostics(tmp_path / "servo_diagnostics.jsonl", writer=writer)
    writer.submit.reset_mock()  # Constructor metadata is not a sampled tick.
    return diag, writer


def test_first_sample_is_immediate_then_at_most_once_per_second(tmp_path):
    diag, writer = _diag(tmp_path)
    _observe(diag, now=0.0)
    assert writer.submit.call_count == 1
    for now in (0.01, 0.5, 0.99):
        _observe(diag, now=now)
    assert writer.submit.call_count == 1
    _observe(diag, now=1.0)
    assert writer.submit.call_count == 2


def test_device_mode_and_enable_are_reported_from_feedback_not_host_commands(tmp_path):
    from servo.diagnostics import _console_line
    diag, writer = _diag(tmp_path)
    sample = _sample(debug={'control_path': 'impedance'})
    sample['device_feedback'] = {
        'status': {'mode_feedback': 1, 'arm_status': 0, 'age_ms': 20.},
        'drivers': [{'fresh': True, 'driver_enable_status': True} for _ in range(6)] + [
            {'fresh': False, 'driver_enable_status': True}],
    }
    _observe(diag, sample)
    row = writer.submit.call_args.args[0]
    assert row['device_feedback']['status']['mode_feedback'] == 1
    assert row['control_path'] == 'impedance'
    line = _console_line(row)
    assert 'device_mode=1' in line
    assert 'enabled_at_sample=6/7' in line


def test_tcp_reference_and_visual_goal_have_distinct_meanings_and_units(tmp_path):
    diag, writer = _diag(tmp_path)
    _observe(diag, now=10.0)
    row = writer.submit.call_args.args[0]
    np.testing.assert_allclose(row["actual_tcp_mm"], [0, 0, 0])
    np.testing.assert_allclose(row["equilibrium_tcp_mm"], [15, 0, 0])
    np.testing.assert_allclose(row["visual_goal_tcp_mm"], [100, 0, 0])
    assert row["leash_mm"] == pytest.approx(15)
    assert row["tracking_error_mm"] == pytest.approx(100)
    assert row["actual_motion_mm"] is None
    assert row["equilibrium_motion_mm"] is None
    assert datetime.fromisoformat(row["timestamp"]).utcoffset() is not None
    q = np.zeros(7)
    q[0] = 0.001
    _observe(diag, _sample(q=q), now=11.0, cfg=_cfg([0.020, 0, 0]))
    row = writer.submit.call_args.args[0]
    assert row["actual_motion_mm"] == pytest.approx(1)
    assert row["equilibrium_motion_mm"] == pytest.approx(5)
    assert row["joint_motion_max_deg"] == pytest.approx(np.rad2deg(0.001))


def test_command_uses_installed_twist_not_unbounded_pbvs_request(tmp_path):
    diag, writer = _diag(tmp_path)
    sample = _sample()
    sample["servo_out"]["tcp_twist"] = np.array([0.9, 0, 0, 0, 0, 0])
    _observe(diag, sample)
    row = writer.submit.call_args.args[0]
    np.testing.assert_allclose(row["command_linear_mm_s"], [10, 0, 0])
    np.testing.assert_allclose(row["requested_qdot_rad_s"], np.full(7, 0.003))
    np.testing.assert_allclose(row["measured_qdot_rad_s"], np.full(7, 0.002))


def test_lost_frame_never_reuses_previous_requested_joint_command(tmp_path):
    diag, writer = _diag(tmp_path)
    _observe(diag, now=10)
    _observe(diag, _sample(servo_out=False), status="LOST", now=11)
    row = writer.submit.call_args.args[0]
    assert row["requested_qdot_rad_s"] is None
    assert row["visual_goal_tcp_mm"] is None
    assert row["tracking_error_mm"] is None


def test_lost_hold_reports_zero_installed_twist_but_keeps_equilibrium(tmp_path):
    diag, writer = _diag(tmp_path)
    cfg = _cfg()
    cfg.target_twist = np.zeros(6)
    _observe(diag, _sample(servo_out=False), cfg=cfg, status="LOST")
    row = writer.submit.call_args.args[0]
    np.testing.assert_allclose(row["command_linear_mm_s"], [0, 0, 0])
    np.testing.assert_allclose(row["equilibrium_tcp_mm"], [15, 0, 0])
    assert row["requested_qdot_rad_s"] is None


def test_missing_inner_loop_telemetry_is_null_not_fake_zero(tmp_path):
    diag, writer = _diag(tmp_path)
    _observe(diag)
    row = writer.submit.call_args.args[0]
    for field in (
        "running", "impedance_active", "control_path", "impedance_cycles",
        "impedance_cycles_delta", "impedance_hz", "tau_impedance_nm",
        "tau_gravity_nm", "tau_ff_sent_nm", "tau_feedback_nm",
        "q_command_rad", "q_command_error_rad", "q_measured_at_command_rad",
        "dq_command_rad_s", "kp_command", "kd_command", "kp_12bit_estimate",
        "non_gravity_torque_command_nm", "tau_ff_command_nm", "command_timestamp",
        "command_sent_age_ms", "command_control_path", "stop_transition",
    ):
        assert row[field] is None, field


def _sent_command():
    return dict(
        pos=np.full(7, .12), vel=np.full(7, .01), kp=np.full(7, .75),
        kd=np.full(7, .2), q_measured=np.full(7, .10), dq_measured=np.full(7, .004),
        non_gravity_torque=np.full(7, .3), tau_ff=np.full(7, 2.3),
        wall_time=1792461599.9, monotonic_time=9.99, control_path="impedance",
    )


def test_final_command_snapshot_overrides_neither_config_nor_outer_measurements(tmp_path):
    diag, writer = _diag(tmp_path)
    cfg = _cfg()
    cfg.q_follow, cfg.follow_kp = np.full(7, .8), np.full(7, 9)
    sent = _sent_command()
    _observe(diag, _sample(q=np.full(7, .4), debug={"last_sent_command": sent}), cfg=cfg)
    row = writer.submit.call_args.args[0]
    np.testing.assert_allclose(row["q_rad"], .4)
    np.testing.assert_allclose(row["q_command_rad"], .12)
    np.testing.assert_allclose(row["q_measured_at_command_rad"], .1)
    np.testing.assert_allclose(row["q_command_error_rad"], .02)
    np.testing.assert_allclose(row["dq_command_rad_s"], .01)
    np.testing.assert_allclose(row["dq_measured_at_command_rad_s"], .004)
    np.testing.assert_allclose(row["kp_command"], .75)
    np.testing.assert_allclose(row["kd_command"], .2)
    np.testing.assert_allclose(row["tau_ff_command_nm"], 2.3)
    np.testing.assert_allclose(row["non_gravity_torque_command_nm"], .3)
    np.testing.assert_allclose(row["kp_12bit_estimate"], np.floor(.75 * 4095 / 500) * 500 / 4095)
    assert row["command_sent_age_ms"] == pytest.approx(10)
    assert datetime.fromisoformat(row["command_timestamp"]).timestamp() == pytest.approx(1792461599.9)
    assert row["command_control_path"] == "impedance"
    sent["pos"][:] = 5  # Queued observations own their vectors.
    np.testing.assert_allclose(row["q_command_rad"], .12)


def test_config_does_not_fabricate_a_missing_sent_command(tmp_path):
    diag, writer = _diag(tmp_path)
    cfg = _cfg()
    cfg.q_follow, cfg.follow_kp, cfg.kd_joint = np.ones(7), np.ones(7), np.ones(7)
    _observe(diag, cfg=cfg)
    row = writer.submit.call_args.args[0]
    assert row["q_command_rad"] is None
    assert row["kp_command"] is None
    assert row["kd_command"] is None


@pytest.mark.parametrize('invalid', [None, 'invalid', [1, 2, 3], np.full(7, np.nan)])
def test_invalid_snapshot_measurement_leaves_command_error_unknown(tmp_path, invalid):
    diag, writer = _diag(tmp_path)
    sent = _sent_command()
    sent["q_measured"] = invalid
    _observe(diag, _sample(debug={"last_sent_command": sent}))
    row = writer.submit.call_args.args[0]
    assert row["q_command_error_rad"] is None
    assert row["q_measured_at_command_rad"] is None
    np.testing.assert_allclose(row["q_command_rad"], .12)


def test_invalid_command_fields_and_timestamps_stay_null(tmp_path):
    diag, writer = _diag(tmp_path)
    sent = _sent_command()
    sent.update(pos=np.full(7, np.inf), kp='invalid', kd=[1],
                monotonic_time=float('nan'), wall_time=float('inf'))
    _observe(diag, _sample(debug={"last_sent_command": sent}))
    row = writer.submit.call_args.args[0]
    for field in ("q_command_rad", "q_command_error_rad", "kp_command", "kd_command",
                  "kp_12bit_estimate", "command_sent_age_ms", "command_timestamp"):
        assert row[field] is None


def test_stop_transition_reports_completion_without_reusing_follow_config(tmp_path):
    diag, writer = _diag(tmp_path)
    sent = _sent_command()
    sent["control_path"] = "stop_transition"
    sample = _sample(debug={
        "last_sent_command": sent,
        "stop_transition_fraction": .5, "stop_transition_complete": False,
    })
    _observe(diag, sample)
    row = writer.submit.call_args.args[0]
    assert row["stop_transition"] == {"fraction": .5, "complete": False}
    assert row["command_control_path"] == "stop_transition"
    sample["debug"].update(stop_transition_fraction=1.0, stop_transition_complete=True)
    _observe(diag, sample, now=11)
    row = writer.submit.call_args.args[0]
    assert row["stop_transition"] == {"fraction": 1.0, "complete": True}
    assert row["command_sent_age_ms"] == pytest.approx(1010)


def test_cached_inner_telemetry_and_cycle_hz_are_reported(tmp_path):
    diag, writer = _diag(tmp_path)
    debug = {
        "running": True, "impedance_active": True, "control_path": "impedance",
        "impedance_cycle_count": 100,
        "impedance_torque": np.full(7, 1.0),
        "gravity_torque": np.full(7, 2.0),
        "total_torque": np.full(7, 3.0),
        "feedback_torque": np.full(7, 4.0),
    }
    _observe(diag, _sample(debug=debug), now=10)
    row = writer.submit.call_args.args[0]
    assert row["impedance_cycles_delta"] is None
    assert row["impedance_hz"] is None
    debug["impedance_cycle_count"] = 350
    _observe(diag, _sample(debug=debug), now=12)
    row = writer.submit.call_args.args[0]
    assert row["running"] is True and row["impedance_active"] is True
    assert row["control_path"] == "impedance"
    assert row["impedance_cycles"] == 350
    assert row["impedance_cycles_delta"] == 250
    assert row["impedance_hz"] == pytest.approx(125)
    for field, expected in (
        ("tau_impedance_nm", 1), ("tau_gravity_nm", 2),
        ("tau_ff_sent_nm", 3), ("tau_feedback_nm", 4),
    ):
        np.testing.assert_allclose(row[field], np.full(7, expected))


def test_counter_reset_never_produces_negative_inner_loop_rate(tmp_path):
    diag, writer = _diag(tmp_path)
    _observe(diag, _sample(debug={"impedance_cycle_count": 100}), now=10)
    _observe(diag, _sample(debug={"impedance_cycle_count": 3}), now=11)
    row = writer.submit.call_args.args[0]
    assert row["impedance_cycles"] == 3
    assert row["impedance_cycles_delta"] is None
    assert row["impedance_hz"] is None
    assert row["impedance_counter_reset"] is True
    _observe(diag, _sample(debug={"impedance_cycle_count": 103}), now=12)
    assert writer.submit.call_args.args[0]["impedance_hz"] == pytest.approx(100)


def test_diagnostics_cli_is_opt_in_and_preserves_existing_servo_defaults():
    from real_nero_grasp.servo_cli import servo_parser
    parser = servo_parser()
    baseline = parser.parse_args([])
    enabled = parser.parse_args(["--servo-diagnostics", "--servo-follow-only", "--servo-preview"])
    assert not baseline.servo_diagnostics
    assert enabled.servo_diagnostics and enabled.servo_follow_only and enabled.servo_preview
    assert enabled.max_linear_vel == baseline.max_linear_vel
    assert enabled.max_joint_vel == baseline.max_joint_vel
    assert enabled.cart_k_scale == baseline.cart_k_scale
    assert enabled.cart_b_scale == baseline.cart_b_scale


def test_full_pipeline_cli_accepts_diagnostics_without_changing_control(monkeypatch):
    import sys
    from pathlib import Path
    from real_nero_grasp.cli import parse_args
    config = Path(__file__).resolve().parents[1] / "config" / "pipeline.yaml"
    monkeypatch.setattr(sys, "argv", [
        "main.py", "--config", str(config), "--phase", "servo", "--execute",
        "--target-source", "mediapipe", "--servo-follow-only", "--servo-preview",
        "--servo-diagnostics", "--max-linear-vel", ".01", "--max-joint-vel", ".03",
    ])
    args = parse_args()
    assert args.servo_diagnostics and args.servo_preview and args.servo_follow_only
    assert args.phase == "servo" and args.target_source == "mediapipe"
    assert args.max_linear_vel == .01 and args.max_joint_vel == .03


def test_frame_loss_and_duplicate_counts_include_throttled_ticks(tmp_path):
    diag, writer = _diag(tmp_path)
    sample = _sample()
    sample["detection"] = "valid"
    _observe(diag, sample, now=10, frame_sequence=1)
    lost = _sample(servo_out=False)
    lost["detection"] = "lost"
    _observe(diag, lost, now=10.25, frame_sequence=2)
    duplicate = _sample(servo_out=False)
    duplicate["detection"] = "duplicate"
    _observe(diag, duplicate, now=10.5, frame_sequence=2)
    _observe(diag, sample, now=11, frame_sequence=3,
             frame_received_monotonic=10.8, target_time=1792461599.9)
    row = writer.submit.call_args.args[0]
    assert row["interval_counts"] == {
        "ticks": 3, "frames": 2, "detected": 1, "lost": 1,
        "duplicate": 1, "reference_updates": 1,
    }
    assert row["frame_age_ms"] == pytest.approx(200)
    assert row["target_age_ms"] == pytest.approx(100, abs=.001)


def test_nonfinite_and_missing_vectors_do_not_look_like_zero_motion(tmp_path):
    from servo.diagnostics import _json_value
    import json
    diag, writer = _diag(tmp_path)
    sample = _sample(q=np.full(7, np.nan))
    sample["dq"] = None
    sample["debug"]["impedance_torque"] = np.full(7, np.inf)
    _observe(diag, sample)
    row = writer.submit.call_args.args[0]
    for field in ("q_rad", "actual_tcp_mm", "actual_motion_mm",
                  "joint_motion_max_deg", "measured_qdot_rad_s", "tau_impedance_nm"):
        assert row[field] is None, field
    json.dumps(_json_value(row), allow_nan=False)


def test_false_inner_flags_are_kept_and_not_mistaken_for_missing(tmp_path):
    diag, writer = _diag(tmp_path)
    _observe(diag, _sample(debug={
        "running": False, "impedance_active": False,
        "impedance_cycle_count": 0, "control_path": "position",
    }))
    row = writer.submit.call_args.args[0]
    assert row["running"] is False and row["impedance_active"] is False
    assert row["impedance_cycles"] == 0
    assert row["control_path"] == "position"


def test_failed_refresh_does_not_report_an_unaccepted_reference(tmp_path):
    from servo_delivery import ServoDeliveryNode
    node = ServoDeliveryNode.__new__(ServoDeliveryNode)
    node.args = NS(servo_hz=30, max_joint_vel=.03, follow_min_qdot=.01,
                   follow_kp=np.zeros(7), impedance_target_linear_vel=.01,
                   impedance_max_position_error=.015, visual_orientation=False,
                   impedance_target_angular_vel=.5)
    node.model = _Model()
    node._imp_cfg = _cfg()
    node._imp_target_update_time = 10.0
    node._imp_follow_q = np.zeros(7)
    node._imp_smoothed_tcp = np.zeros(3)
    node._imp_smoothed_R = np.eye(3)
    node._diag_sample = _sample()
    node._diag_sample["reference_updated"] = False
    node.backend = NS(set_cartesian_impedance=Mock(side_effect=RuntimeError("install failed")))
    node._stop_task = Mock(side_effect=SystemExit(1))
    out = _sample()["servo_out"]
    out["tcp_twist"] = np.array([.01, 0, 0, 0, 0, 0])
    with pytest.raises(SystemExit):
        node._refresh(np.zeros(7), out, 10.04)
    assert not node._diag_sample["reference_updated"]
    assert node._diag_sample["reference_write_failed"]
    diag, writer = _diag(tmp_path)
    _observe(diag, node._diag_sample, cfg=node._imp_cfg)
    row = writer.submit.call_args.args[0]
    assert row["reference_write_failed"]
    assert row["command_linear_mm_s"] is None
    assert row["equilibrium_tcp_mm"] is None


@pytest.mark.parametrize("key,value,status", [
    ("last_error", "inner feedback fault", "FAULT"),
    ("controlled_stop_reason", "joint buffer hold", "STOPPING"),
])
def test_inner_stop_cannot_be_mislabelled_as_previous_tracking(key, value, status):
    from servo_delivery import ServoDeliveryNode
    node = ServoDeliveryNode.__new__(ServoDeliveryNode)
    node._diagnostics = Mock()
    node._diag_sample = _sample(debug={key: value})
    node.model = _Model()
    node.state = "TRACK_HAND"
    node._preview_status, node._preview_reason = "TRACKING", "target_accepted"
    node._publish_diagnostics()
    assert node._diagnostics.observe.call_args.kwargs["status"] == status
    assert node._diagnostics.observe.call_args.kwargs["reason"] == value


def test_full_writer_queue_drops_instead_of_blocking_control():
    import queue
    import threading
    from servo.diagnostics import _ServoDiagnosticWriter
    writer = _ServoDiagnosticWriter.__new__(_ServoDiagnosticWriter)
    writer.stop = threading.Event()
    writer.records = queue.Queue(maxsize=1)
    writer.dropped = 0
    writer.submit({"sequence": 1})
    writer.submit({"sequence": 2})
    assert writer.dropped == 1
    assert writer.records.get_nowait() == {"sequence": 1}


def test_writer_status_and_close_are_visible_without_threads(tmp_path):
    diag, writer = _diag(tmp_path)
    writer.dropped, writer.error = 3, "disk unavailable"
    _observe(diag)
    row = writer.submit.call_args.args[0]
    assert row["writer_dropped"] == 3
    assert row["writer_error"] == "disk unavailable"
    diag.close()
    writer.close.assert_called_once_with()


def test_real_async_writer_saves_timestamped_json_and_console_without_control(tmp_path, capfd):
    import json
    path = tmp_path / "diagnostic_run" / "samples.jsonl"
    diag = ServoDiagnostics(path)
    try:
        _observe(diag, _sample(debug={"last_sent_command": _sent_command()}), now=10)
        _observe(diag, now=11)
    finally:
        diag.close()
    assert not diag.writer.thread.is_alive()
    assert diag.writer.error is None
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert [row["type"] for row in rows] == ["metadata", "servo_sample", "servo_sample"]
    assert rows[1]["actual_motion_mm"] is None
    assert rows[1]["q_command_error_rad"] == pytest.approx([.02] * 7)
    assert rows[1]["kp_command"] == [.75] * 7
    assert "not a device acknowledgement" in rows[0]["command_semantics"]
    assert rows[2]["actual_motion_mm"] == 0.0
    assert "[servo][diag]" in capfd.readouterr().out


def test_node_diagnostic_hook_is_cached_only_and_failures_are_best_effort():
    from servo_delivery import ServoDeliveryNode
    node = ServoDeliveryNode.__new__(ServoDeliveryNode)
    node.backend = Mock()
    node._diagnostics = Mock()
    node._diagnostics.observe.side_effect = RuntimeError("writer unavailable")
    node._diag_sample = _sample()
    node.model = _Model()
    node.state = "TRACK_HAND"
    node._imp_cfg = _cfg()
    node._publish_diagnostics()
    node._diagnostics.observe.assert_called_once()
    assert node._diagnostics.observe.call_args.args[0] is node._diag_sample
    assert node.backend.mock_calls == []


def test_model_failure_is_swallowed_by_node_hook_not_control(tmp_path):
    from servo_delivery import ServoDeliveryNode
    node = ServoDeliveryNode.__new__(ServoDeliveryNode)
    node._diagnostics, writer = _diag(tmp_path)
    node._diag_sample = _sample()
    node.model = NS(fk=Mock(side_effect=ValueError("invalid model state")))
    node.state, node._imp_cfg = "TRACK_HAND", _cfg()
    node.backend = Mock()
    node._publish_diagnostics()
    assert writer.submit.call_count == 0
    assert node.backend.mock_calls == []


def test_timer_diagnostic_failure_never_masks_original_control_error():
    from servo_delivery import ServoDeliveryNode
    node = ServoDeliveryNode.__new__(ServoDeliveryNode)
    node._preview = None
    fault = RuntimeError("original control fault")
    node._servo_step = Mock(side_effect=fault)
    node._publish_preview = Mock()
    node._diagnostics = Mock()
    node._diagnostics.observe.side_effect = OSError("diagnostic failure")
    node._diag_sample = _sample()
    node.model, node.state = _Model(), "TRACK_HAND"
    node.backend = Mock()
    with pytest.raises(RuntimeError) as caught:
        node._on_timer()
    assert caught.value is fault
    node._diagnostics.observe.assert_called_once()
    assert node.backend.mock_calls == []


@pytest.mark.parametrize("enabled", [False, True])
def test_empty_cache_or_disabled_diagnostics_never_reads_backend(enabled):
    from servo_delivery import ServoDeliveryNode
    node = ServoDeliveryNode.__new__(ServoDeliveryNode)
    reporter = Mock()
    node._diagnostics = reporter if enabled else None
    node._diag_sample = None if enabled else _sample()
    node.backend = Mock()
    node._publish_diagnostics()
    reporter.observe.assert_not_called()
    assert node.backend.mock_calls == []


def _unstarted_writer(path):
    import queue
    import threading
    from servo.diagnostics import _ServoDiagnosticWriter
    writer = _ServoDiagnosticWriter.__new__(_ServoDiagnosticWriter)
    writer.path, writer.records = path, queue.Queue(maxsize=16)
    writer.stop, writer.dropped, writer.error = threading.Event(), 0, None
    return writer


def test_broken_stdout_does_not_prevent_json_file_output(tmp_path, monkeypatch):
    import json
    diag, captured = _diag(tmp_path)
    _observe(diag, now=10)
    first = captured.submit.call_args.args[0]
    _observe(diag, now=11)
    second = captured.submit.call_args.args[0]
    path = tmp_path / "written" / "samples.jsonl"
    writer = _unstarted_writer(path)
    writer.records.put_nowait(first)
    writer.records.put_nowait(second)
    writer.stop.set()
    broken = Mock(side_effect=BrokenPipeError("closed terminal"))
    monkeypatch.setattr("builtins.print", broken)
    writer._run()  # Synchronous worker body: no diagnostic/control thread starts.
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(rows) == 2
    assert rows[0]["monotonic_time"] == 10
    assert rows[1]["monotonic_time"] == 11
    assert writer.error is None
    assert broken.call_count == 1


def test_file_open_failure_is_recorded_without_raising_or_retrying_commands(tmp_path, monkeypatch):
    from pathlib import Path
    writer = _unstarted_writer(tmp_path / "samples.jsonl")
    writer.records.put_nowait({"type": "metadata"})
    writer.stop.set()
    monkeypatch.setattr(Path, "open", Mock(side_effect=PermissionError("read only")))
    monkeypatch.setattr("builtins.print", Mock())
    writer._run()
    assert writer.error == "PermissionError: read only"
    assert writer.records.empty()


def test_close_join_has_bounded_timeout_even_for_a_blocked_writer(tmp_path):
    writer = _unstarted_writer(tmp_path / "samples.jsonl")
    writer.thread = Mock()
    writer.close()
    assert writer.stop.is_set()
    writer.thread.join.assert_called_once_with(timeout=.25)


@pytest.mark.parametrize("failure", [None, SystemExit(0), RuntimeError("servo failure")])
def test_diagnostic_cleanup_follows_arm_hold_and_disconnect(tmp_path, monkeypatch, failure):
    import servo_delivery as module
    import cameras.realsense as camera
    import cameras.servo_preview as display
    import servo.diagnostics as diagnostics_module
    from real_nero_grasp.servo_cli import servo_parser
    args = servo_parser().parse_args(["--servo-diagnostics", "--servo-preview"])
    args.save_dir = str(tmp_path)
    events = []
    backend = NS(disconnect=lambda: events.append("disconnect"))
    def close_reporter():
        events.append("diagnostic_close")
        raise OSError("diagnostic close failed")
    reporter = NS(close=close_reporter)
    monkeypatch.setattr(diagnostics_module, "ServoDiagnostics", Mock(return_value=reporter))
    monkeypatch.setattr(module, "validate_servo_args", lambda args: None)
    monkeypatch.setattr(module, "NeroPinocchioModel", lambda: NS(clone=lambda: NS()))
    monkeypatch.setattr(module, "build_target", lambda *a: NS(close=lambda: events.append("target_close")))
    monkeypatch.setattr(camera, "camera_for", lambda args: NS(read=lambda: events.append("camera_read")))
    monkeypatch.setattr(camera, "close_all", lambda: events.append("camera_close"))
    monkeypatch.setattr(display, "ServoPreview", lambda: NS(
        start=lambda: events.append("preview_start"),
        close=lambda: events.append("preview_close")))
    def make_node(existing, received, model, *, target, preview, diagnostics):
        assert existing is backend and received is args and diagnostics is reporter
        return NS(prepare_shutdown=lambda: events.append("hold"))
    monkeypatch.setattr(module, "ServoDeliveryNode", make_node)
    monkeypatch.setattr(module.sdk_loop, "init", lambda **kw: None)
    monkeypatch.setattr(module.sdk_loop, "shutdown", lambda: events.append("host_shutdown"))
    def spin(node):
        if failure is not None:
            raise failure
    monkeypatch.setattr(module.sdk_loop, "spin", spin)
    if failure is None:
        assert module.run_servo_delivery(args, backend=backend) == 0
    else:
        with pytest.raises(type(failure)) as caught:
            module.run_servo_delivery(args, backend=backend)
        assert caught.value is failure
    assert events.index("hold") < events.index("disconnect") < events.index("diagnostic_close")
    assert events[-1] == "diagnostic_close"


def test_diagnostic_initialization_failure_preserves_primary_camera_failure(tmp_path, monkeypatch):
    import servo_delivery as module
    import cameras.realsense as camera
    import servo.diagnostics as diagnostics_module
    from real_nero_grasp.servo_cli import servo_parser
    args = servo_parser().parse_args(["--servo-diagnostics"])
    args.save_dir = str(tmp_path)
    reporter = Mock(side_effect=OSError("cannot start diagnostics"))
    monkeypatch.setattr(diagnostics_module, "ServoDiagnostics", reporter)
    monkeypatch.setattr(module, "validate_servo_args", lambda args: None)
    monkeypatch.setattr(module, "NeroPinocchioModel", lambda: NS(clone=lambda: NS()))
    monkeypatch.setattr(module, "build_target", lambda *a: NS(close=lambda: None))
    failure = RuntimeError("original camera failure")
    monkeypatch.setattr(camera, "camera_for", lambda args: NS(read=Mock(side_effect=failure)))
    monkeypatch.setattr(camera, "close_all", lambda: None)
    monkeypatch.setattr(module.sdk_loop, "shutdown", lambda: None)
    backend = Mock()
    with pytest.raises(RuntimeError) as caught:
        module.run_servo_delivery(args, backend=backend)
    assert caught.value is failure
    reporter.assert_called_once()
    assert backend.mock_calls == [call.disconnect()]

