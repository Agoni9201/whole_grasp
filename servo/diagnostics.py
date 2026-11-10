"""Opt-in, low-rate servo observations. No backend, camera, or control access."""
from datetime import datetime
import json
from pathlib import Path
import queue
import threading
import time

import numpy as np


def _vector(value, length):
    if value is None:
        return None
    try:
        result = np.asarray(value, dtype=float).ravel()
    except (TypeError, ValueError):
        return None
    return result.copy() if result.shape == (length,) and np.isfinite(result).all() else None


def _distance(a, b, scale=1.0):
    return None if a is None or b is None else float(np.linalg.norm(a - b) * scale)


def _json_value(value):
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist())
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def _number(value, digits=2):
    return "--" if value is None else f"{float(value):.{digits}f}"


def _finite_number(value):
    if value is None or isinstance(value, (bool, np.bool_)):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if np.isfinite(number) else None


def _command_observations(debug, now):
    """Copy the completed SDK-send snapshot; never infer commands from config."""
    sent = debug.get("last_sent_command")
    sent = sent if isinstance(sent, dict) else {}
    position = _vector(sent.get("pos"), 7)
    measured = _vector(sent.get("q_measured"), 7)
    kp = _vector(sent.get("kp"), 7)
    monotonic = _finite_number(sent.get("monotonic_time"))
    wall_time = _finite_number(sent.get("wall_time"))
    timestamp = None
    if wall_time is not None:
        try:
            timestamp = datetime.fromtimestamp(wall_time).astimezone().isoformat()
        except (ValueError, OverflowError, OSError):
            pass
    fraction = _finite_number(debug.get("stop_transition_fraction"))
    complete = debug.get("stop_transition_complete")
    complete = bool(complete) if isinstance(complete, (bool, np.bool_)) else None
    return {
        "q_command_rad": position,
        "q_measured_at_command_rad": measured,
        "q_command_error_rad": None if position is None or measured is None else position - measured,
        "dq_command_rad_s": _vector(sent.get("vel"), 7),
        "dq_measured_at_command_rad_s": _vector(sent.get("dq_measured"), 7),
        "kp_command": kp,
        "kd_command": _vector(sent.get("kd"), 7),
        "kp_12bit_estimate": None if kp is None else np.floor(np.clip(kp, 0, 500) * 4095 / 500) * 500 / 4095,
        "non_gravity_torque_command_nm": _vector(sent.get("non_gravity_torque"), 7),
        "tau_ff_command_nm": _vector(sent.get("tau_ff"), 7),
        "command_timestamp": timestamp,
        "command_sent_age_ms": None if monotonic is None or monotonic > now else (now - monotonic) * 1000,
        "command_control_path": sent.get("control_path") if isinstance(sent.get("control_path"), str) else None,
        "stop_transition": None if fraction is None and complete is None else {
            "fraction": fraction, "complete": complete,
        },
    }


def _console_line(record):
    def point(key):
        value = record.get(key)
        return "--" if value is None else str(np.round(value, 1).tolist())

    torque = record.get("tau_impedance_nm")
    torque_max = None if torque is None else float(np.max(np.abs(torque)))
    kp = record.get("kp_command")
    kp_max = None if kp is None else float(np.max(kp))
    health = record.get("depth_health")
    depth_info = (f"depth_valid_pct={_number(health['scene']['valid_percent'], 1)} "
                  if health else "")
    device = record.get('device_feedback') or {}
    device_status = device.get('status') or {}
    drivers = device.get('drivers') or []
    enabled = sum(item.get('fresh') is True and item.get('driver_enable_status') is True
                  for item in drivers)
    device_info = (f"device_mode={device_status.get('mode_feedback')} "
                   f"device_arm_status={device_status.get('arm_status')} "
                   f"enabled_at_sample={enabled}/7 "
                   f"status_age_ms={_number(device_status.get('age_ms'), 1)} "
                   f"device_sample_age_ms={_number(None if device.get('sampled_wall_time') is None else (record['wall_time'] - device['sampled_wall_time']) * 1000, 1)} "
                   if device else '')
    return (
        f"[servo][diag] {record['timestamp']} status={record['status']} reason={record['reason']} "
        f"actual_mm={point('actual_tcp_mm')} eq_mm={point('equilibrium_tcp_mm')} "
        f"goal_mm={point('visual_goal_tcp_mm')} "
        f"move_mm={_number(record['actual_motion_mm'])} "
        f"eq_move_mm={_number(record['equilibrium_motion_mm'])} "
        f"leash_mm={_number(record['leash_mm'])} "
        f"cmd_mm_s={point('command_linear_mm_s')} "
        + (f"center_error_px={record['camera_center'].get('pixel_error')} "
           f"center_solver={record['camera_center'].get('solver_status')} "
           if record.get('camera_center') else '') +
        f"inner={record['control_path']} cycles+={record['impedance_cycles_delta']} "
        f"inner_hz={_number(record['impedance_hz'], 1)} "
        f"imp_tau_max_Nm={_number(torque_max)} "
        f"sent_kp_max={_number(kp_max)} cmd_age_ms={_number(record.get('command_sent_age_ms'), 1)} "
        f"{depth_info}"
        f"{device_info}"
        f"frames={record['interval_counts']}"
    )


class _ServoDiagnosticWriter:
    """Only this daemon touches files/stdout; the producer never waits on I/O."""

    def __init__(self, path):
        self.path = Path(path)
        self.records = queue.Queue(maxsize=16)
        self.stop = threading.Event()
        self.dropped = 0
        self.error = None
        self.thread = threading.Thread(target=self._run, daemon=True, name="servo_diagnostics")
        self.thread.start()

    def submit(self, record):
        if self.stop.is_set():
            return
        try:
            self.records.put_nowait(record)
        except queue.Full:
            self.dropped += 1

    def close(self):
        self.stop.set()
        self.thread.join(timeout=0.25)

    def _run(self):
        stream = None
        console_ok = True
        try:
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                stream = self.path.open("x", encoding="utf-8")
            except Exception as exc:
                self.error = f"{type(exc).__name__}: {exc}"
            while not self.stop.is_set() or not self.records.empty():
                try:
                    record = self.records.get(timeout=0.05)
                except queue.Empty:
                    continue
                frame = record.pop("_rgbd_frame", None)
                if frame is not None:
                    try:
                        from servo.depth_diagnostics import analyze_frame, save_frame
                        record["depth_health"] = analyze_frame(frame)
                        # The same JSON travels inside the archive, so moving
                        # it away from samples.jsonl does not lose provenance.
                        frame["metadata_json"] = json.dumps(_json_value({
                            **record["depth_health"], "timestamp": record["timestamp"],
                            "perception": record.get("perception"),
                            "dry_run": True, "q_image_semantics": "Synthetic zero joints; not robot feedback",
                        }), allow_nan=False)
                        record["rgbd_snapshot"] = save_frame(
                            self.path.parent / "rgbd", frame, record.pop("_rgbd_index"))
                    except Exception as exc:
                        record.pop("_rgbd_index", None)
                        record["depth_diagnostic_error"] = f"{type(exc).__name__}: {exc}"
                try:
                    if stream is not None:
                        stream.write(json.dumps(_json_value(record), ensure_ascii=False,
                                                allow_nan=False) + "\n")
                        stream.flush()
                except Exception as exc:
                    self.error = f"{type(exc).__name__}: {exc}"
                    try:
                        stream.close()
                    except Exception:
                        pass
                    stream = None
                if console_ok:
                    try:
                        if record.get("type") == "servo_sample":
                            print(_console_line(record), flush=True)
                        else:
                            print(f"[servo][diag] log={self.path} file_error={self.error}", flush=True)
                    except Exception:
                        console_ok = False  # A closed terminal must not affect control or file output.
        finally:
            if stream is not None:
                try:
                    stream.close()
                except Exception:
                    pass


class ServoDiagnostics:
    """One-second snapshots built only from observations the outer loop owns."""

    def __init__(self, path, *, writer=None, metadata=None):
        self.writer = writer if writer is not None else _ServoDiagnosticWriter(path)
        self._previous = None
        self._last_frame = None
        self._counts = self._empty_counts()
        # Full images are collected only during camera-only dry runs. Existing
        # live-control logging stays small, with no added image writer load.
        self._capture_rgbd = bool((metadata or {}).get("dry_run", False))
        self._rgbd_count = 0
        self._last_rgbd_time = None
        self.writer.submit({
            "type": "metadata", "schema_version": 2,
            "timestamp": datetime.now().astimezone().isoformat(),
            "parameters": metadata or {},
            "rgbd_evidence": {"enabled": self._capture_rgbd, "minimum_interval_s": 5.0,
                              "maximum_snapshots": 12, "depth_unit": "metre",
                              "q_image_semantics": "Synthetic zero joints during dry-run; not robot feedback"},
            "torque_semantics": (
                "tau_ff_sent_nm is the clipped feedforward sent to SDK, excluding firmware PD; "
                "tau_feedback_nm is asynchronous motor feedback, not a synchronized response."),
            "motion_semantics": "Motion fields are net displacement since the previous report, not path length.",
            "frame_semantics": "Frame counts describe observations consumed by the outer servo, not camera hardware FPS.",
            "reference_semantics": "Equilibrium/twist are host-side MIT references, not device acknowledgements; failed writes are null.",
            "device_feedback_semantics": (
                "Received SDK status/driver feedback, sampled at most once per second without transmitting commands. "
                "fresh and age_ms describe sampled_wall_time; check sample age and individual timestamps before interpreting current device state. "
                "Missing fields are unknown, not disabled or fault-free."),
            "command_semantics": (
                "Command fields come from the last completed seven-joint SDK-send batch, not the current config. "
                "Command error uses that batch's measured q; age identifies an old batch. SDK completion is not a device acknowledgement. "
                "kp_12bit_estimate assumes V111 host encoding over [0,500], using floor(kp*4095/500)*500/4095; "
                "it is not measured firmware gain or total motor torque."),
        })

    @staticmethod
    def _empty_counts():
        return dict(ticks=0, frames=0, detected=0, lost=0, duplicate=0, reference_updates=0)

    def observe(self, sample, *, model, cfg, status, reason, stage,
                frame_sequence=None, frame_received_monotonic=None, target_time=None,
                now=None, wall_time=None):
        now = time.monotonic() if now is None else float(now)
        wall_time = time.time() if wall_time is None else float(wall_time)
        self._counts["ticks"] += 1
        if frame_sequence is not None and frame_sequence != self._last_frame:
            self._counts["frames"] += 1
            self._last_frame = frame_sequence
        event = sample.get("detection")
        if event in ("valid", "lost", "duplicate"):
            self._counts[{"valid": "detected", "lost": "lost", "duplicate": "duplicate"}[event]] += 1
        self._counts["reference_updates"] += int(bool(sample.get("reference_updated")))
        previous = self._previous
        if previous is not None and now - previous["time"] < 1.0:
            return

        q = _vector(sample.get("q"), 7)
        actual = None if q is None else _vector(model.fk(q)[0], 3)
        equilibrium = _vector(getattr(cfg, "target_tcp", None), 3)
        installed_twist = _vector(getattr(cfg, "target_twist", None), 6)
        write_failed = bool(sample.get("reference_write_failed"))
        if write_failed:
            equilibrium, installed_twist = None, None
        out = sample.get("servo_out") or {}
        goal = _vector(out.get("target_tcp"), 3)
        error = _vector(out.get("error"), 3)
        command_twist = _vector(out.get("tcp_twist"), 6)
        debug = sample.get("debug") or {}
        cycles = debug.get("impedance_cycle_count")
        cycles = int(cycles) if cycles is not None else None
        interval = None if previous is None else now - previous["time"]
        delta_cycles = None
        counter_reset = False
        if previous is not None and cycles is not None and previous["cycles"] is not None:
            counter_reset = cycles < previous["cycles"]
            if not counter_reset:
                delta_cycles = cycles - previous["cycles"]
        joint_delta = None if previous is None or q is None or previous["q"] is None else float(
            np.max(np.abs(np.rad2deg(q - previous["q"]))))
        record = {
            "type": "servo_sample", "timestamp": datetime.fromtimestamp(wall_time).astimezone().isoformat(),
            "wall_time": wall_time, "monotonic_time": now, "interval_s": interval,
            "stage": stage, "status": status, "reason": reason,
            "detection_event": event, "reference_updated_this_tick": bool(sample.get("reference_updated")),
            "reference_write_failed": write_failed,
            "perception": sample.get("perception"),
            "gesture_handover": sample.get('gesture_handover'),
            "hand_memory": sample.get('hand_memory'),
            "camera_center": out.get('camera_center'),
            "command_angular_rad_s": None if installed_twist is None else installed_twist[3:],
            "frame_sequence": frame_sequence,
            "sample_age_ms": None if sample.get("sampled_monotonic") is None else (now - sample["sampled_monotonic"]) * 1000,
            "frame_age_ms": None if frame_received_monotonic is None else (now - frame_received_monotonic) * 1000,
            "target_age_ms": None if not target_time else (wall_time - target_time) * 1000,
            "actual_tcp_mm": None if actual is None else actual * 1000,
            "equilibrium_tcp_mm": None if equilibrium is None else equilibrium * 1000,
            "visual_goal_tcp_mm": None if goal is None else goal * 1000,
            "tracking_error_mm": None if error is None else float(np.linalg.norm(error) * 1000),
            "leash_mm": _distance(equilibrium, actual, 1000),
            "actual_motion_mm": None if previous is None else _distance(actual, previous["actual"], 1000),
            "equilibrium_motion_mm": None if previous is None else _distance(equilibrium, previous["equilibrium"], 1000),
            "joint_motion_max_deg": joint_delta,
            "q_rad": q, "measured_qdot_rad_s": _vector(sample.get("dq"), 7),
            "requested_qdot_rad_s": _vector(out.get("qdot"), 7),
            "pbvs_linear_mm_s": None if command_twist is None else command_twist[:3] * 1000,
            "command_linear_mm_s": None if installed_twist is None else installed_twist[:3] * 1000,
            "running": debug.get("running"), "impedance_active": debug.get("impedance_active"),
            "control_path": debug.get("control_path"), "impedance_cycles": cycles,
            "impedance_cycles_delta": delta_cycles, "impedance_counter_reset": counter_reset,
            "impedance_hz": None if delta_cycles is None or not interval else delta_cycles / interval,
            "tau_impedance_nm": _vector(debug.get("impedance_torque"), 7),
            "tau_gravity_nm": _vector(debug.get("gravity_torque"), 7),
            "tau_ff_sent_nm": _vector(debug.get("total_torque"), 7),
            "tau_feedback_nm": _vector(debug.get("feedback_torque"), 7),
            "inner_error": debug.get("last_error"),
            "jump_reacquire": sample.get('jump_reacquire'),
            "jump_reacquire_error_m": sample.get('jump_reacquire_error_m'),
            "device_feedback": sample.get('device_feedback'),
            "stop_reason": sample.get("stop_reason") or debug.get("controlled_stop_reason"),
            "interval_counts": self._counts.copy(),
            "writer_dropped": getattr(self.writer, "dropped", 0),
            "writer_error": getattr(self.writer, "error", None),
        }
        record.update(_command_observations(debug, now))
        frame = sample.get("rgbd_frame")
        if (self._capture_rgbd and frame is not None and self._rgbd_count < 12
                and (self._last_rgbd_time is None or now - self._last_rgbd_time >= 5.0)):
            self._rgbd_count += 1
            self._last_rgbd_time = now
            record["_rgbd_frame"] = frame
            record["_rgbd_index"] = self._rgbd_count
        self.writer.submit(record)
        self._previous = dict(time=now, q=q, actual=actual, equilibrium=equilibrium, cycles=cycles)
        self._counts = self._empty_counts()

    def close(self):
        self.writer.close()
