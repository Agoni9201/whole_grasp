"""Human labels and whole-window history must not leak across action/state boundaries."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from tools.prepare_handover_reviewed_windows import prepare
from tools.indexed_mjpeg_avi import FRAME_READER_VERSION


@pytest.fixture
def recording(tmp_path):
    source = tmp_path/'source/handover_fusion/session_p123'
    (source/'global').mkdir(parents=True)
    config = dict(window_seconds=1., max_gap_seconds=.25,
                  min_window_samples=6, hold_drift_rad=.009)
    (source/'metadata.json').write_text(json.dumps(dict(dry_run=False, mode='shadow',
        control_mode='mit_cartesian_hold', config=config)))
    times = np.arange(161)*.05+10.
    observations = [dict(video_index=i, received_monotonic=float(t)) for i,t in enumerate(times)]
    (source/'global/observations.jsonl').write_text(''.join(json.dumps(o)+'\n' for o in observations))
    rows = []
    for t in times:
        rows.append(dict(monotonic=float(t), policy=dict(
            phase='FUSION_WAIT_BASELINE' if t < 11 else 'FUSION_WAIT_TAKE',
            baseline_nm=None if t < 11 else [0.]*7, hold_q_rad=[0.]*7),
            feedback=dict(q=[0.]*7, tau=[0.,float(t-10),0.,0.,0.,0.,0.],
                motor_timestamps=[float(t+100)]*7, position_timestamps=[float(t+100)]*7,
                sampled_wall_time=float(t+100+.005)),
            device_status=dict(ctrl_mode=1, mode_feedback=6, arm_status=0, err_code=0),
            visual=dict(label='NO_CONTACT', visible=False, in_roi=False)))
    path = tmp_path/'annotations.json'
    intervals = []
    for a,b,label in [(0,60,'NO_CONTACT'),(60,100,'TAKE_PULL'),(100,160,'POST_RELEASE')]:
        intervals.append(dict(start_frame=a,end_frame=b,start_monotonic=float(times[a]),
            end_monotonic=float(times[b]),label=label,reviewed_by='operator',
            evidence='operator_video_review',end_exclusive=True))
    annotation = dict(source=str(source),physical_grasp_group='grasp01',split='train',intervals=intervals)

    def save():
        (source/'samples.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
        annotation['source_sha256'] = hashlib.sha256((source/'samples.jsonl').read_bytes()).hexdigest()
        path.write_text(json.dumps(annotation))
        return path
    save()
    return source, rows, annotation, save


def test_uses_operator_labels_and_excludes_unknown_boundaries_and_prebaseline_history(recording):
    source, rows, annotation, save = recording
    originals = {p:p.read_bytes() for p in source.rglob('*') if p.is_file()}
    report, windows = prepare(save())
    assert report['primary_positive'] > 0 and report['primary_negative'] > 0
    assert 'no_completed_baseline' in report['exclusions']
    for row in windows:
        assert row['window_start'] >= row['reviewed_interval_start']+.2-1e-9
        assert row['window_end'] < row['reviewed_interval_end']-.2
        assert row['window_start'] >= 11.
        assert row['labels_verified_by_operator'] is True
    assert any(row['label']=='TAKE_PULL' for row in windows)  # Raw keyboard hint was NO_CONTACT.
    assert all(p.read_bytes()==content for p,content in originals.items())


def test_fault_samples_never_reuse_retained_last_policy_features(recording):
    source, rows, annotation, save = recording
    for r in rows:
        r['policy']['features'] = [999.]*28
        if r['monotonic'] >= 14.5:
            r['policy']['phase'] = 'FUSION_FAULT'
    report, windows = prepare(save())
    assert 'policy_not_valid_for_primary_action' in report['exclusions']
    assert windows and all(r['monotonic'] < 14.5 for r in windows)
    assert all(r['feature_0'] != 999. for r in windows)


def test_stale_packet_breaks_history_instead_of_being_hidden_inside_a_window(recording):
    source, rows, annotation, save = recording
    rows[40]['feedback']['sampled_wall_time'] += .1
    report, windows = prepare(save())
    assert report['exclusions']['feedback_age_invalid'] == 1
    assert all(not (r['window_start'] <= 12. <= r['window_end']) for r in windows)


def test_unannotated_gap_does_not_become_a_negative_class(recording):
    source, rows, annotation, save = recording
    annotation['intervals'].pop(0)
    report, windows = prepare(save())
    assert report['primary_windows_by_action'].get('NO_CONTACT',0) == 0
    assert all(r['monotonic'] >= 13. for r in windows)


def test_hold_is_preserved_as_its_own_class_and_not_labeled_untouched(recording):
    source, rows, annotation, save = recording
    annotation['intervals'][1]['label']='HOLD_NO_PULL'
    report, windows = prepare(save())
    held = [r for r in windows if r['label']=='HOLD_NO_PULL']
    assert held and all(r['binary_target']=='' and r['use']=='hold_no_pull_separate' for r in held)
    assert report['primary_positive']==0


def test_annotated_frame_time_must_match_source_mapping(recording):
    _, _, annotation, save = recording
    annotation['intervals'][1]['start_monotonic'] += .1
    with pytest.raises(ValueError,match='frame time'):
        prepare(save())


def test_hash_mismatch_and_overlapping_annotations_are_rejected(recording):
    source, _, annotation, save = recording
    path=save();(source/'samples.jsonl').write_text('changed')
    with pytest.raises(ValueError,match='hash mismatch'):
        prepare(path)
    annotation['intervals'][1]['start_frame']=59
    annotation['intervals'][1]['start_monotonic']=12.95
    with pytest.raises(ValueError,match='overlap'):
        prepare(save())


def test_interrupted_video_old_boundaries_cannot_enter_training_before_recheck(recording):
    import cv2
    import struct
    source,_,annotation,save=recording
    video=source/'global/global_rgb.avi'
    writer=cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*'MJPG'),30.,(32,24))
    assert writer.isOpened()
    for _ in range(161):
        writer.write(np.zeros((24,32,3),np.uint8))
    writer.release()
    content=bytearray(video.read_bytes())
    struct.pack_into('<I',content,4,0xffffffff)
    video.write_bytes(content)
    with pytest.raises(ValueError,match='frame boundary recheck'):
        prepare(save())
    for interval in annotation['intervals']:
        interval['frame_reader_version']=FRAME_READER_VERSION
    report,windows=prepare(save())
    assert windows and report['primary_positive']>0


def test_missing_video_is_not_treated_as_continuous_human_label_evidence(recording):
    source,_,annotation,save=recording
    path=source/'global/observations.jsonl'
    frames=[json.loads(line) for line in path.read_text().splitlines()]
    for frame in frames[80:]:
        frame['received_monotonic']+=1.
    path.write_text(''.join(json.dumps(frame)+'\n' for frame in frames))
    for interval in annotation['intervals']:
        interval['start_monotonic']=frames[interval['start_frame']]['received_monotonic']
        interval['end_monotonic']=frames[interval['end_frame']]['received_monotonic']
    report,windows=prepare(save())
    gap=report['video_gaps'][0]
    assert (gap['before_frame'],gap['after_frame'])==(79,80)
    assert report['exclusions']['video_gap_in_causal_history']>0
    assert windows
    assert all(row['window_end']<=gap['start_monotonic'] or row['monotonic']-1.>=gap['end_monotonic'] for row in windows)
