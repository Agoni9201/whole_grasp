#!/usr/bin/env python3
"""Audit saved T01 evidence on CPU; never import camera, robot or model runtimes.

Distances to measured points describe observation support, not completion truth
or calibrated uncertainty. Each trial is a separate capture, not a paired test.
"""

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


ROOT = Path(__file__).resolve().parents[1]


def read_json(path):
    return json.loads(path.read_text())


def fingerprint(path):
    return {"path": str(path.relative_to(ROOT)),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def transform(points, matrix):
    matrix = np.asarray(matrix, dtype=float)
    return np.asarray(points) @ matrix[:3, :3].T + matrix[:3, 3]


def completion_metrics(directory):
    info = read_json(directory / "completion.json")
    measured = np.load(directory / "measured_camera.npy", allow_pickle=False)
    prediction = np.load(directory / "prediction_camera.npy", allow_pickle=False)
    base = np.load(directory / "prediction_base.npy", allow_pickle=False)
    for points in (measured, prediction, base):
        if points.ndim != 2 or points.shape[1] != 3 or not len(points) or not np.isfinite(points).all():
            raise ValueError(f"Invalid saved XYZ array in {directory}")
    to_prediction = cKDTree(prediction).query(measured)[0]
    to_measured = cKDTree(measured).query(prediction)[0]
    measured_base = transform(measured, info["T_base_camera"])
    table = read_json(directory.parent / "table_reference/table_reference_report.json")
    table_z = float(table["physical_table_z_m"])
    mapped_base = transform(prediction, info["T_base_camera"])
    reference = transform(measured, np.linalg.inv(info["T_base_reference"]) @ np.asarray(info["T_base_camera"]))
    normalization = info["normalization"]
    restored = (np.load(directory / "input_normalized.npy", allow_pickle=False)
                * normalization["bbox_diagonal_m"] + normalization["centroid_camera_m"])
    indices = np.load(directory / "sample_indices.npy", allow_pickle=False)
    center = next(view for view in table["views"]
                  if view["view"] == "center" and (ROOT / view["source"]).resolve() == (ROOT / info["rgb_source"]).resolve())
    metadata = read_json(ROOT / info["rgb_source"] / "metadata.json")
    expected_reference = np.asarray(center["T_registration"]) @ np.asarray(metadata["T_base_camera"])
    return {
        "source": str(directory.relative_to(ROOT)),
        "measured_points": len(measured), "prediction_points": len(prediction),
        "measured_to_prediction_p95_mm": float(np.quantile(to_prediction, .95) * 1000),
        "recorded_measured_to_prediction_p95_mm": info["observed_to_prediction_p95_m"] * 1000,
        "prediction_to_measured_p95_mm": float(np.quantile(to_measured, .95) * 1000),
        "prediction_without_nearby_measurement_fraction": {
            f"{threshold:g}mm": float(np.mean(to_measured > threshold / 1000))
            for threshold in (3, 6, 10)},
        "measured_bbox_camera_mm": (np.ptp(measured, axis=0) * 1000).tolist(),
        "prediction_bbox_camera_mm": (np.ptp(prediction, axis=0) * 1000).tolist(),
        "confirmed_table_z_m": table_z,
        "below_table_minus5mm": {
            "measured": int(np.sum(measured_base[:, 2] < table_z - .005)),
            "prediction": int(np.sum(base[:, 2] < table_z - .005))},
        "saved_base_transform_max_error_m": float(np.max(np.abs(mapped_base - base))),
        "normalized_input_roundtrip_max_error_m": float(np.max(np.abs(restored - reference[indices]))),
        "registered_rgb_reference_max_matrix_error": float(np.max(np.abs(expected_reference - info["T_base_reference"]))),
        "sampled_input_points": len(indices), "sampled_unique_indices": len(np.unique(indices)),
        "independent_heldout_geometry_used": False,
        "completion_source_sha256": fingerprint(directory / "completion.json")["sha256"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=Path, default=ROOT / "artifacts/ablation_20261101/trials.csv")
    parser.add_argument("--runtime-result", type=Path, action="append", default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    opts = parser.parse_args()
    with opts.trials.open(encoding="utf-8-sig", newline="") as stream:
        source_rows = list(csv.DictReader(stream))
    rows = []
    completion = []
    for source in source_rows:
        run = ROOT / source["source_run"]
        experiment = read_json(run / "experiment_result.json")
        row = {key: source[key] for key in (
            "trial", "group", "generator", "mgpc", "retained_views", "physical_outcome",
            "software_status", "software_success", "software_failure_stage", "observer_note",
            "session_s", "grasp_stage_wall_s", "native_full_routes", "refined_full_routes")}
        row["source_run"] = str(run.relative_to(ROOT))
        row["recorded_views"] = experiment["views"]
        archives = sorted(run.glob("*/multiview_raw/*/metadata.json"))
        row["archives"] = [{"path": str(p.relative_to(ROOT)),
                            "label": read_json(p)["view_label"],
                            "status": read_json(p)["status"],
                            "handeye_source": read_json(p)["handeye"].get("source_file")}
                           for p in archives]
        rows.append(row)
        if source["mgpc"] == "True":
            paths = list(run.glob("*/mgpc/completion.json"))
            if len(paths) != 1:
                raise ValueError(f"Expected one saved completion for {run}; found {len(paths)}")
            completion.append({"trial": source["trial"], **completion_metrics(paths[0].parent)})
    groups = {}
    for group in sorted({row["group"] for row in rows}):
        selected = [row for row in rows if row["group"] == group]
        groups[group] = {
            "n": len(selected),
            "physical_outcomes": dict(Counter(row["physical_outcome"] for row in selected)),
            "software_success_count": sum(row["software_success"] == "True" for row in selected),
            "mean_grasp_stage_wall_s": float(np.mean([float(row["grasp_stage_wall_s"]) for row in selected]))}
    paired = []
    for name in ("mgpc_CD_same_capture", "mgpc_CD_no_focus"):
        path = ROOT / "artifacts/banana_planning_diagnosis/contact_20261028" / name / "comparison.json"
        paired.append({"source": fingerprint(path), "saved_comparison": read_json(path)})
    snapshot_paths = ["main.py", "scripts/run_grasp_safe.sh", "config/pipeline_orbbec.yaml",
                      "cameras/options.py", "real_nero_grasp/cli.py", "real_nero_grasp/workflow.py",
                      "real_nero_grasp/active_view.py", "real_nero_grasp/active_view_scoring.py",
                      "real_nero_grasp/nbv_geometry.py",
                      "real_nero_grasp/pointcloud_fusion.py", "real_nero_grasp/mgpc_completion.py",
                      "real_nero_grasp/grasp_workflow.py", "tools/mgpc_inference.py",
                      "tools/compare_mgpc_reconstruction.py", "tools/audit_t01_saved_evidence.py",
                      "third_party/MGPC/models/MGPC.py"]
    runtime = []
    for path in opts.runtime_result:
        data = read_json(path)
        runtime.append({"source": fingerprint(path.resolve()), **{key: data[key] for key in
                       ("run_id", "command", "method", "views", "outcome")}})
    result = {
        "audit_scope_date": "2026-11-05", "hardware_connected": False,
        "motion_commands_sent": False, "model_inference_run": False,
        "trials_source": fingerprint(opts.trials.resolve()),
        "source_snapshot": [fingerprint(ROOT / path) for path in snapshot_paths],
        "trials": rows, "groups": groups, "completion_support": completion,
        "historical_same_capture_comparisons": paired, "runtime_examples": runtime,
        "limitations": [
            "Different trials are not same-capture paired comparisons; each group has two trials.",
            "Nearest-neighbor support depends on sensor density, thresholds and registration.",
            "Unsupported predictions can be true hidden surfaces; support is not calibrated uncertainty.",
            "A low measured-to-prediction distance does not validate hidden geometry or contacts.",
            "Table violations are relative to the saved confirmed plane, not independent geometry truth.",
            "Saved same-capture comparisons predate current candidate refinement; no new IK/path replay.",
            "Historical D435 captures do not establish performance of the current Orbbec setup."]}
    opts.output_dir.mkdir(parents=True, exist_ok=True)
    output = opts.output_dir / "evidence.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(f"Saved {len(rows)} trials and {len(completion)} completion audits to {output}")
    for row in completion:
        print(f"{row['trial']}: measured->prediction p95={row['measured_to_prediction_p95_mm']:.2f} mm; "
              f"prediction without measurement within 6 mm="
              f"{row['prediction_without_nearby_measurement_fraction']['6mm']:.1%}")


if __name__ == "__main__":
    main()
