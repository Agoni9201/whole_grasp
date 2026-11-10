import json
from pathlib import Path
from types import SimpleNamespace

from real_nero_grasp.experiment_log import (
    finalize_experiment_log,
    initialize_experiment_log,
    mark_experiment_stage,
    record_experiment_views,
    record_gripper_close,
    record_live_path_clearance,
    record_runtime_measured_clearance,
    record_selected_grasp,
    sample_grasp_stability,
    set_experiment_session,
)


class _Arm:
    def __init__(self, statuses):
        self._statuses = iter(statuses)

    def print_gripper_status(self):
        return next(self._statuses)


def test_post_lift_force_drop_rejects_width_only_false_positive(tmp_path):
    args = SimpleNamespace(
        save_dir=str(tmp_path),
        grasp_stability_samples=3,
        grasp_stability_interval=0.0,
        grasp_stability_max_width_drift=0.002,
        empty_grasp_width_threshold=0.001,
    )
    initialize_experiment_log(args)
    close = SimpleNamespace(contact_width=0.0335, contact_force=0.502)
    result = sample_grasp_stability(
        _Arm([SimpleNamespace(value=0.0323, force=0.162)] * 3), args, close
    )
    assert result["max_abs_width_drift_m"] < 0.002
    assert result["force_retention_ratio"] < 0.5
    assert result["stable"] is False


def test_complete_experiment_result_is_written_once(tmp_path):
    session = tmp_path / "capture"
    session.mkdir()
    args = SimpleNamespace(
        save_dir=str(tmp_path),
        target_class="bottle",
        phase="grasp",
        execute=True,
        active_view=True,
        active_side_view_count=2,
        yolo_weights="model.pt",
        calib_json="calibration.json",
        empty_grasp_width_threshold=0.001,
        grasp_stability_samples=3,
        grasp_stability_interval=0.0,
        grasp_stability_max_width_drift=0.002,
        _workflow_motion_sent=True,
    )
    initialize_experiment_log(args, ["main.py", "--execute"])
    run_dir = args.save_dir
    set_experiment_session(args, session, target_label="bottle")
    record_experiment_views(
        args,
        SimpleNamespace(
            kept_view_count=3,
            used_multiview=True,
            reason="fused",
        ),
    )
    record_selected_grasp(
        args,
        {
            "idx": 8,
            "raw_score": 0.264608,
            "center_dist": 0.0332,
            "graspnet_width": 0.0807,
            "jaw_selected_orientation_delta_deg": 2.5,
        },
        0.0318,
    )
    record_live_path_clearance(
        args,
        {
            "nearest_clearance": 0.012,
            "required_clearance": 0.005,
            "margin": 0.007,
            "primitive": "gripper_finger_+1",
            "source": "environment",
        },
        "grasp",
    )
    record_runtime_measured_clearance(
        args,
        {
            "geometric_nearest_clearance": 0.009,
            "geometric_required_clearance": 0.005,
            "geometric_primitive": "gripper_finger_+1",
            "geometric_source": "environment",
        },
        "grasp",
        "target_approach",
        [0.0] * 7,
    )
    close = SimpleNamespace(
        status=SimpleNamespace(value=0.0440, force=0.45),
        contact_detected=True,
        stop_reason="force_contact",
        contact_width=0.0440,
        contact_force=0.45,
    )
    record_gripper_close(args, close)
    sample_grasp_stability(
        _Arm(
            [
                SimpleNamespace(value=0.0440, force=0.44),
                SimpleNamespace(value=0.0438, force=0.43),
                SimpleNamespace(value=0.0439, force=0.42),
            ]
        ),
        args,
        close,
    )
    mark_experiment_stage(args, "complete")
    result = SimpleNamespace(
        status=SimpleNamespace(value="success"),
        motion_sent=True,
        detail="executed",
    )
    first = finalize_experiment_log(args, result)
    second = finalize_experiment_log(args, result)

    assert second is None
    assert first["outcome"]["success"] is True
    assert first["views"]["used_for_final_reconstruction"] == 3
    assert first["grasp_stability"]["stable"] is True
    assert first["path_clearance"]["minimum_live_preflight_clearance_m"] == 0.012
    assert first["path_clearance"][
        "minimum_runtime_modeled_clearance_from_measured_joints_m"
    ] == 0.009
    assert first["corrections"]["width_recenter_m"] == 0.0318
    with (Path(run_dir) / "experiment_result.json").open(encoding="utf-8") as stream:
        per_session = json.load(stream)
    assert per_session["run_id"] == first["run_id"]
    lines = (tmp_path / "experiment_results.jsonl").read_text(
        encoding="utf-8"
    ).splitlines()
    assert len(lines) == 1
