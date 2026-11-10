#!/usr/bin/env python3
"""Prepare causal torque windows from operator-reviewed intervals, no training.

Unannotated history, pre-baseline motion, faults, stale feedback and transition
boundaries are excluded. HOLD_NO_PULL remains a separate action, not NO_CONTACT.
"""
from __future__ import annotations

import argparse
from bisect import bisect_right
from collections import Counter, deque
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from handoff.fusion_handover import FEATURE_VERSION, torque_features
from tools.check_handover_dataset import read_inner_faults
from tools.review_handover_labels import LABELS, read_jsonl, video_gaps
from tools.indexed_mjpeg_avi import IndexedMJPEGVideo, FRAME_READER_VERSION

TARGET = {'TAKE_PULL': 1, 'NO_CONTACT': 0, 'BUMP_PUSH': 0, 'POST_RELEASE': 0}


def prepare(annotation_file, *, margin=.2, stride=.1):
    if not np.isfinite([margin, stride]).all() or margin < 0 or stride <= 0:
        raise ValueError('Margin must be nonnegative and stride positive')
    annotation_file = Path(annotation_file)
    annotation_bytes = annotation_file.read_bytes()
    annotations = json.loads(annotation_bytes)
    source = Path(annotations['source'])
    if not source.is_absolute():
        source = PROJECT/source
    source_bytes = (source/'samples.jsonl').read_bytes()
    digest = hashlib.sha256(source_bytes).hexdigest()
    if digest != annotations.get('source_sha256'):
        raise ValueError('Annotation/source hash mismatch')
    meta = json.loads((source/'metadata.json').read_text())
    if meta.get('dry_run') is not False or meta.get('mode') != 'shadow' or meta.get('control_mode') != 'mit_cartesian_hold':
        raise ValueError('Require real same-mode shadow data')
    group = annotations.get('physical_grasp_group')
    split = annotations.get('split')
    if not group or split not in ('train', 'validation', 'test', 'test_candidate'):
        raise ValueError('Reviewed intervals need explicit physical grasp group and split')
    frames = read_jsonl(source/'global/observations.jsonl')
    gaps=video_gaps([float(frame['received_monotonic']) for frame in frames])
    gap_starts=[gap['start_monotonic'] for gap in gaps]
    intervals = annotations['intervals']
    video_path=source/'global/global_rgb.avi'
    if video_path.is_file():
        video=IndexedMJPEGVideo(video_path)
        if video.unfinalized and any(a.get('frame_reader_version') != FRAME_READER_VERSION for a in intervals):
            raise ValueError('Unfinalized AVI annotations need frame boundary recheck in the repaired reviewer')
        if any(a['end_frame']>=len(video) for a in intervals):
            raise ValueError('Annotation references a frame not completely saved in AVI')
    starts = []
    for i, a in enumerate(intervals):
        if (a.get('reviewed_by') != 'operator' or a.get('evidence') != 'operator_video_review'
                or a.get('label') not in LABELS or a.get('end_exclusive') is not True):
            raise ValueError('Require explicit operator action review')
        lo, hi = a['start_frame'], a['end_frame']
        if type(lo) is not int or type(hi) is not int or not 0 <= lo < hi < len(frames):
            raise ValueError('Invalid annotation frame interval')
        begin = float(frames[lo]['received_monotonic'])
        end = float(frames[hi]['received_monotonic'])
        if abs(begin-a['start_monotonic'])>1e-8 or abs(end-a['end_monotonic'])>1e-8:
            raise ValueError('Annotation time does not match authoritative frame time')
        if i and intervals[i-1]['end_monotonic'] > begin:
            raise ValueError('Reviewed intervals overlap or are unsorted')
        starts.append(begin)
    config = meta['config']
    duration = float(config['window_seconds'])
    rows = read_jsonl(source/'samples.jsonl')
    bad_streams, reasons, counts = Counter(), Counter(), Counter()
    window = deque()
    previous_stamps = None
    previous_position = None
    previous_now = None
    last_export = None
    window_baseline = None
    output = []
    per_interval = [dict(index=i+1, label=a['label'], note=a.get('note', ''),
        duration_seconds=a['end_monotonic']-a['start_monotonic'],
        sample_phases=Counter(), exclusions=Counter(), windows=0) for i,a in enumerate(intervals)]
    inner_faults = [f['monotonic_time'] for f in read_inner_faults(source)
                    if isinstance(f.get('monotonic_time'), (int, float))]
    inner_cutoff = min(inner_faults) if inner_faults else float('inf')
    for r in rows:
        now = float(r['monotonic'])
        if previous_now is not None and now <= previous_now:
            raise ValueError('Raw sample monotonic clock is not increasing')
        previous_now = now
        idx = bisect_right(starts, now)-1
        a = intervals[idx] if idx >= 0 and now < intervals[idx]['end_monotonic'] else None
        policy, f = r.get('policy') or {}, r.get('feedback') or {}
        if a is not None:
            per_interval[idx]['sample_phases'][policy.get('phase', 'unknown')] += 1

        def reject(reason, clear=False):
            reasons[reason] += 1
            if a is not None:
                per_interval[idx]['exclusions'][reason] += 1
            if clear:
                window.clear()

        try:
            q, tau, stamps, positions = [np.asarray(f[k], float) for k in
                ('q', 'tau', 'motor_timestamps', 'position_timestamps')]
            if any(v.shape!=(7,) or not np.isfinite(v).all() for v in (q,tau,stamps,positions)):
                raise ValueError('invalid vectors')
        except (KeyError, TypeError, ValueError):
            reject('invalid_feedback', True)
            continue
        if previous_stamps is not None and (np.any(stamps<previous_stamps) or np.any(positions<previous_position)):
            reject('feedback_timestamps_reversed', True)
            bad_streams['feedback_timestamps_reversed'] += 1
            previous_stamps, previous_position = stamps, positions
            continue
        if previous_stamps is not None and not np.all(stamps>previous_stamps):
            reject('duplicate_motor_snapshot')
            continue
        previous_stamps, previous_position = stamps, positions
        sampled_wall = float(f['sampled_wall_time'])
        ages = sampled_wall-np.r_[stamps,positions]
        if np.any(ages < -.05) or np.any(ages > .05):
            reject('feedback_age_invalid', True)
            continue
        if now >= inner_cutoff:
            reject('inner_control_fault', True)
            continue
        baseline = policy.get('baseline_nm')
        if baseline is None:
            reject('no_completed_baseline', True)
            continue
        if policy.get('phase') not in ('FUSION_WAIT_TAKE', 'FUSION_CONFIRM'):
            reject('policy_not_valid_for_primary_action', True)
            continue
        device = r.get('device_status') or {}
        if device != dict(ctrl_mode=1, mode_feedback=6, arm_status=0, err_code=0):
            reject('invalid_device_status', True)
            continue
        hold = policy.get('hold_q_rad')
        if hold is None or np.max(np.abs(q-np.asarray(hold))) > config['hold_drift_rad']:
            reject('outside_original_pose_limit', True)
            continue
        base = np.asarray(baseline, float)
        if base.shape != (7,) or not np.isfinite(base).all():
            reject('invalid_baseline', True)
            continue
        if window_baseline is None or not np.array_equal(base, window_baseline):
            window.clear()
            window_baseline = base.copy()
        if window and now-window[-1][0] > config['max_gap_seconds']:
            window.clear()
        window.append((now, tau-base))
        while len(window)>1 and now-window[0][0] > duration+1e-9:
            window.popleft()
        if a is None:
            reject('unannotated_history')
            continue
        if a['label'] in ('UNKNOWN', 'ARM_DISTURBANCE'):
            reject('unknown_or_arm_disturbance')
            continue
        # The whole theoretical causal window, not just its final frame, must
        # lie inside one reviewed state after conservative boundary trimming.
        if now-duration < a['start_monotonic']+margin or now >= a['end_monotonic']-margin:
            reject('label_boundary_or_crossing_history')
            continue
        gap_idx=bisect_right(gap_starts,now)-1
        if (gap_idx>=0 and now>gaps[gap_idx]['start_monotonic']
                and now-duration<gaps[gap_idx]['end_monotonic']):
            reject('video_gap_in_causal_history',True)
            continue
        if len(window)<config['min_window_samples'] or window[-1][0]-window[0][0] < .8*duration:
            reject('incomplete_causal_window')
            continue
        if last_export is not None and now-last_export < stride-1e-9:
            reject('export_stride')
            continue
        last_export = now
        features = torque_features(*zip(*window))
        item = dict(session=source.name, physical_grasp_group=group, split=split,
            annotation_interval=idx+1, label=a['label'], binary_target=TARGET.get(a['label'], ''),
            use='primary_action' if a['label'] in TARGET else 'hold_no_pull_separate',
            monotonic=now, window_start=window[0][0], window_end=window[-1][0],
            reviewed_interval_start=a['start_monotonic'], reviewed_interval_end=a['end_monotonic'],
            labels_verified_by_operator=True, baseline_origin='original_online_baseline',
            policy_phase=policy['phase'],
            visual_candidate=bool((r.get('visual') or {}).get('visible') and (r.get('visual') or {}).get('in_roi')))
        item.update({f'feature_{i}':float(v) for i,v in enumerate(features)})
        output.append(item)
        counts[a['label']] += 1
        per_interval[idx]['windows'] += 1
    primary = Counter(r['label'] for r in output if r['use']=='primary_action')
    report = dict(version=1, annotation=str(annotation_file.resolve()),
        annotation_sha256=hashlib.sha256(annotation_bytes).hexdigest(), source=str(source), source_sha256=digest,
        physical_grasp_group=group, split=split,
        reviewed_intervals=len(intervals), reviewed_interval_counts=dict(Counter(a['label'] for a in intervals)),
        window_seconds=duration, boundary_margin_seconds=margin, export_stride_seconds=stride,
        feature_version=FEATURE_VERSION,
        video_gaps=gaps,
        frame_index_sha256=hashlib.sha256((source/'global/observations.jsonl').read_bytes()).hexdigest(),
        windows_by_action=dict(counts), primary_windows_by_action=dict(primary),
        total_windows=len(output), primary_windows=sum(primary.values()),
        primary_positive=sum(r['binary_target']==1 for r in output),
        primary_negative=sum(r['binary_target']==0 for r in output),
        exclusions=dict(reasons), stream_issues=dict(bad_streams), per_interval=per_interval,
        model_trained=False, ready_for_generalization_claim=False,
        notes=['Exported windows overlap and are not independent trials.',
               'All history lies within one reviewed interval; unknown gaps remain unknown.',
               'HOLD_NO_PULL is retained separately, not relabeled NO_CONTACT.',
               'Causal windows overlapping long gaps in recorded video are excluded.',
               'Visual eligibility is recorded but does not filter the torque classifier classes.',
               'These files do not authorize gripper writes or prove human support.'])
    if hashlib.sha256((source/'samples.jsonl').read_bytes()).hexdigest()!=digest:
        raise ValueError('Source changed during preparation')
    if annotation_file.read_bytes()!=annotation_bytes:
        raise ValueError('Annotations changed during preparation; run again after saving')
    return report, output


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--annotation', type=Path, required=True)
    p.add_argument('--out', type=Path)
    p.add_argument('--margin-seconds', type=float, default=.2)
    p.add_argument('--stride-seconds', type=float, default=.1)
    args = p.parse_args(argv)
    report, windows = prepare(args.annotation, margin=args.margin_seconds, stride=args.stride_seconds)
    out = args.out or PROJECT/'artifacts/handover_label_review_20261109/prepared'/Path(report['source']).name
    out.mkdir(parents=True, exist_ok=True)
    (out/'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    with (out/'reviewed_windows.csv').open('w', newline='') as stream:
        if windows:
            writer = csv.DictWriter(stream, fieldnames=list(windows[0]));writer.writeheader();writer.writerows(windows)
    print(json.dumps({k:report[k] for k in ('reviewed_interval_counts','windows_by_action',
        'primary_positive','primary_negative','total_windows','primary_windows')}, ensure_ascii=False, indent=2))
    print(f'Saved reviewed windows and report to {out}; no model training or hardware access.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
