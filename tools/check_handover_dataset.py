#!/usr/bin/env python3
"""Read-only collection inventory. Does not import cameras, CAN or arm code."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
import re
from pathlib import Path


def inspect_session(directory):
    meta = json.loads((directory/'metadata.json').read_text())
    collection = meta.get('collection') or {}
    source = directory/'samples.jsonl'
    counts, labels, visual_frames = Counter(), Counter(), set()
    labels_before_fault, labels_after_fault = Counter(), Counter()
    raw, before_fault, after_fault, baseline, all_hands = 0, 0, 0, False, 0
    fault_seen = False
    errors = []
    previous = None
    if source.exists():
        with source.open() as stream:
            for index, line in enumerate(stream, 1):
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    errors.append(f'malformed sample line {index}')
                    continue
                policy, feedback, visual = (row.get(k) or {} for k in ('policy', 'feedback', 'visual'))
                phase = policy.get('phase', 'unknown')
                counts[phase] += 1
                fault_seen |= phase == 'FUSION_FAULT'
                baseline |= policy.get('baseline_nm') is not None
                for e in row.get('events', []):
                    if e.get('action') == 'label':
                        labels[e.get('label', 'unknown')] += 1
                        target = labels_after_fault if fault_seen else labels_before_fault
                        target[e.get('label', 'unknown')] += 1
                if visual.get('sequence') not in visual_frames:
                    visual_frames.add(visual.get('sequence'))
                    all_hands += 'raw_hand_observations' in visual
                vectors = [feedback.get(k) for k in ('q', 'dq_raw', 'tau', 'motor_timestamps', 'position_timestamps')]
                if all(isinstance(v, list) and len(v) == 7
                       and all(isinstance(x, (int, float)) and math.isfinite(x) for x in v)
                       for v in vectors):
                    stamps = vectors[3]
                    if previous is None or all(a > b for a, b in zip(stamps, previous)):
                        raw += 1
                        after_fault += int(fault_seen)
                        before_fault += int(not fault_seen)
                        previous = stamps
    else:
        errors.append('missing samples.jsonl')
    video = directory/'global/global_rgb.avi'
    has_video = video.is_file() and video.stat().st_size > 0
    if not raw:
        errors.append('no seven-axis raw feedback')
    if not baseline:
        errors.append('no completed baseline')
    if not has_video:
        errors.append('no global RGB video')
    inner_faults = read_inner_faults(directory)
    return dict(directory=str(directory), collection_group=collection.get('group'),
        split=collection.get('split'), mode=meta.get('mode'), dry_run=meta.get('dry_run'),
        sample_rows=sum(counts.values()), phases=dict(counts), unique_motor_snapshots=raw,
        raw_snapshots_before_fault=before_fault, raw_snapshots_at_or_after_fault=after_fault,
        baseline_completed=baseline, label_key_events=dict(labels),
        label_key_events_before_fault=dict(labels_before_fault),
        label_key_events_at_or_after_fault=dict(labels_after_fault),
        global_video_saved=has_video, unique_visual_frames=len(visual_frames),
        visual_frames_with_raw_hand_records=all_hands,
        labels_verified=collection.get('labels_verified') is True,
        hold_reference_saved=any_hold_reference(directory), capture_issues=errors,
        inner_control_faults=inner_faults,
        review_flags=['inner_control_fault_recorded'] if inner_faults else [],
        ready_for_training=False,
        note='Inventory only: label alignment, class coverage and replay validity require review.')


def read_inner_faults(directory):
    """Outer handover state can stay WAIT_TAKE even when the MIT loop faults."""
    name = directory.name
    if '_p' not in name:
        return []
    pid = name.rsplit('_p', 1)[1]
    if not pid.isdigit():
        return []
    found = []
    for log in sorted((directory.parent.parent/'servo_diagnostics').glob(f'*_p{pid}/control_cycles.jsonl')):
        with log.open() as stream:
            for line in stream:
                if not re.search(r'"type"\s*:\s*"fault"', line):
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if record.get('type') == 'fault':
                    found.append(dict(source=str(log), reason=record.get('reason'),
                        wall_time=record.get('wall_time'), monotonic_time=record.get('monotonic_time')))
    return found


def any_hold_reference(directory):
    path = directory/'samples.jsonl'
    if not path.exists():
        return False
    with path.open() as stream:
        for line in stream:
            try:
                reference = (json.loads(line).get('policy') or {}).get('hold_q_rad')
                if isinstance(reference, list) and len(reference) == 7:
                    return True
            except json.JSONDecodeError:
                continue
    return False


def inventory(root):
    sessions = []
    for meta in sorted(Path(root).rglob('metadata.json')):
        # Pipeline cloud captures also have metadata.json; only inspect fusion.
        data = json.loads(meta.read_text())
        if data.get('control_mode') == 'mit_cartesian_hold' and 'global_serial' in data:
            sessions.append(inspect_session(meta.parent))
    assignments = defaultdict(set)
    for s in sessions:
        if s['collection_group'] and s['split']:
            assignments[s['collection_group']].add(s['split'])
    conflicts = {k: sorted(v) for k, v in assignments.items() if len(v) > 1}
    return dict(sessions=sessions, physical_grasps_by_split={
        split: sorted(group for group, splits in assignments.items() if split in splits)
        for split in ('train', 'validation', 'test')},
        conflicting_group_assignments=conflicts,
        ready_for_training=False,
        note='Counts are keyboard events and capture groups, not verified actions or independent samples.')


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path('logs/handover_dataset_20261109'))
    p.add_argument('--out', type=Path, help='Optional new JSON manifest; refuses to overwrite')
    args = p.parse_args(argv)
    if not args.root.is_dir():
        p.error(f'collection root does not exist: {args.root}')
    report = inventory(args.root)
    encoded = json.dumps(report, ensure_ascii=False, indent=2)+'\n'
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open('x') as stream:
            stream.write(encoded)
    print(encoded, end='')
    return 2 if report['conflicting_group_assignments'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
