#!/usr/bin/env python3
"""Offline two-finger calibration from reviewed RGB-D references; no hardware.

Uses bounded prior reviewed ROIs, keeps original RGB guards, removes thin
branches before measuring fingers, and requires separate validation images.
It never edits the previous calibration or pipeline YAML.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from real_nero_grasp.camera_self_mask import calibrated_self_mask
from tools.rebuild_camera_self_mask import polygons


def measured_fingers(image, calibration):
    """Fixed offline segmentation, not a runtime mask search/growth rule."""
    regions = calibration['provenance']['roi_polygons_px']
    if len(regions) != 2:
        raise ValueError('this tool requires two previously reviewed finger ROIs')
    dark = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) <= calibration['rgb_guard']['max_gray']
    dark &= image.max(axis=2)-image.min(axis=2) <= calibration['mask_color_guard']['max_channel_spread']
    result = np.zeros(image.shape[:2], np.uint8)
    for polygon in regions:
        roi = np.zeros_like(result)
        cv2.fillPoly(roi, [np.asarray(polygon, np.int32)], 1)
        # A bounded 8px expansion of the reviewed ROI permits the new reference
        # boundary, without letting cables elsewhere become a finger.
        roi = cv2.dilate(roi, np.ones((17, 17), np.uint8))
        candidate = cv2.morphologyEx((dark & (roi > 0)).astype(np.uint8),
                                    cv2.MORPH_OPEN, np.ones((13, 13), np.uint8))
        count, labels, stats, _ = cv2.connectedComponentsWithStats(candidate)
        eligible = [i for i in range(1, count) if np.any(labels[-1] == i)]
        if not eligible:
            raise ValueError('no bottom-connected finger inside a reviewed ROI')
        index = max(eligible, key=lambda i: stats[i, cv2.CC_STAT_AREA])
        if stats[index, cv2.CC_STAT_AREA] < 500:
            raise ValueError('measured finger is too small')
        result[labels == index] = 1
    return result


def rebuild(previous, training, validation, output, report_dir):
    training, validation = [Path(p) for p in training], [Path(p) for p in validation]
    if len(training) < 3 or not validation:
        raise ValueError('need >=3 training references and separate validation references')
    if {p.resolve() for p in training} & {p.resolve() for p in validation}:
        raise ValueError('training and validation captures must be distinct')
    output, report_dir = Path(output), Path(report_dir)
    if output.exists() or report_dir.exists():
        raise FileExistsError('output calibration/report directory must not already exist')
    prior = json.loads(Path(previous).read_text())
    calibration = copy.deepcopy(prior)
    height, width = calibration['profile']['height'], calibration['profile']['width']
    samples = []
    for directory in training+validation:
        image = cv2.imread(str(directory/'color.png'))
        meta = json.loads((directory/'capture_meta.json').read_text())
        if image is None or image.shape != (height, width, 3):
            raise ValueError('reference image/profile mismatch')
        for expected, measured in [('serial', 'serial'), ('width', 'width'), ('height', 'height'),
                                  ('depth_width', 'native_depth_width'), ('depth_height', 'native_depth_height')]:
            if str(meta[measured]) != str(calibration['profile'][expected]):
                raise ValueError(f'profile mismatch {expected}: {directory}')
        for key, expected in calibration['intrinsics'].items():
            if abs(meta['intrinsics'][key]-expected) > .5:
                raise ValueError(f'intrinsic mismatch {key}: {directory}')
        samples.append((directory, image, meta))
    measurements = [measured_fingers(image, prior) for _, image, _ in samples[:len(training)]]
    union, intersection = np.maximum.reduce(measurements), np.minimum.reduce(measurements)
    overlap = float(np.count_nonzero(intersection)/np.count_nonzero(union))
    if overlap < .95:
        raise ValueError('references are not stable enough for a fixed two-finger mask')
    envelope = cv2.dilate(union, np.ones((17, 17), np.uint8))
    stable = cv2.erode(intersection, np.ones((3, 3), np.uint8))
    calibration.update(schema_version=2, polygons_px=polygons(envelope),
        validation_polygons_px=polygons(stable),
        sources=[str(p/'color.png') for p in training],
        description='Current 100mm empty-gripper two-finger reference; fixed offline envelope with 8px margin, original RGB guards.',
        status='current_reference_offline_replay_trial_not_global_default',
        provenance=dict(previous_calibration=str(previous), roi_polygons_px=prior['provenance']['roi_polygons_px'],
            method='reviewed bounded two-finger ROIs; neutral-black segmentation; 13px offline opening removes thin cable branches; fixed 8px envelope',
            capture_count=len(training), heldout_captures=[str(p) for p in validation],
            measured_envelope_pixels=int(union.sum()), stable_pixels=int(stable.sum()),
            stable_to_measured_envelope=overlap, fixed_boundary_margin_px=8,
            opening_evidence='operator-declared capture width; latest user hardware log 99.768mm; no fresh CAN read by capture tool'))
    report_dir.mkdir(parents=True)
    temporary = report_dir/'candidate.json'
    temporary.write_text(json.dumps(calibration, indent=2)+'\n')
    reports = []
    try:
        for index, (directory, image, meta) in enumerate(samples):
            mask, report = calibrated_self_mask(image, temporary, meta)
            row = {k: report[k] for k in ('core_dark_fraction', 'silhouette_dark_fraction',
                                         'outer_dark_fraction', 'masked_pixels')}
            row.update(capture=str(directory), split='build' if index < len(training) else 'heldout',
                       rgb_sha256=hashlib.sha256((directory/'color.png').read_bytes()).hexdigest())
            # Reviewed scene probes: banana is outside the bottom finger ROIs;
            # the external left cable segment must remain an obstacle.
            if (height, width) == (720, 1280):
                row['upper_scene_masked_pixels'] = int(np.count_nonzero(mask[:480]))
                row['left_cable_probe_masked_pixels'] = int(np.count_nonzero(mask[550:590, 300:340]))
                if row['upper_scene_masked_pixels'] or row['left_cable_probe_masked_pixels']:
                    raise ValueError('reference mask includes the reviewed non-finger probes')
            reports.append(row)
            overlay = image.copy()
            cv2.polylines(overlay, [np.asarray(p, np.int32) for p in calibration['polygons_px']], True, (0, 255, 0), 2)
            cv2.imwrite(str(report_dir/f'{index:02d}_{row["split"]}_overlay.png'), overlay)
            cv2.imwrite(str(report_dir/f'{index:02d}_{row["split"]}_mask.png'), mask)
        calibration['validation_summary'] = reports
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open('x') as stream:
            json.dump(calibration, stream, indent=2)
            stream.write('\n')
        (report_dir/'validation.json').write_text(json.dumps(dict(
            calibration=str(output), rgb_guard=calibration['rgb_guard'],
            previous_rgb_guard=prior['rgb_guard'], runtime_adaptation=False,
            opening_freshly_measured=False, rows=reports), indent=2)+'\n')
    finally:
        temporary.unlink(missing_ok=True)
    return reports


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--previous', required=True, type=Path)
    p.add_argument('--training', required=True, nargs='+', type=Path)
    p.add_argument('--validation', required=True, nargs='+', type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--report-dir', required=True, type=Path)
    a = p.parse_args()
    print(json.dumps(rebuild(a.previous, a.training, a.validation, a.output, a.report_dir), indent=2))
