"""Offline annotation timing, source integrity, ambiguity and fault provenance."""
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from tools.review_handover_labels import ReviewProject, locate_sources, handler_for, REVIEW_PLAYER_VERSION
from tools.indexed_mjpeg_avi import IndexedMJPEGVideo, FRAME_READER_VERSION


@pytest.fixture
def project(tmp_path):
    source = tmp_path/'train/grasp01/handover_fusion/record_p123'
    (source/'global').mkdir(parents=True)
    (source/'metadata.json').write_text(json.dumps(dict(control_mode='mit_cartesian_hold',
        collection=dict(group='grasp01', split='train'))))
    stamps = [10., 10.05, 10.3, 10.9, 11.2, 11.7]
    observations = [dict(video_index=i, received_monotonic=t) for i, t in enumerate(stamps)]
    (source/'global/observations.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in observations))
    writer = cv2.VideoWriter(str(source/'global/global_rgb.avi'), cv2.VideoWriter_fourcc(*'MJPG'), 30., (64, 48))
    assert writer.isOpened()
    for i in range(len(stamps)):
        writer.write(np.full((48, 64, 3), i*30, np.uint8))
    writer.release()
    rows = []
    for i, t in enumerate(stamps):
        label = 'TAKE_PULL' if i in (1, 2) else 'POST_RELEASE' if i == 3 else None
        rows.append(dict(monotonic=t, wall_time=t+100.,
            policy=dict(phase='FUSION_FAULT' if i >= 4 else 'FUSION_WAIT_TAKE', baseline_nm=[0.]*7),
            events=[] if label is None else [dict(action='label', label=label, at=t)]))
    (source/'samples.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in rows))
    review = ReviewProject([(source, 'train', 'grasp01')], tmp_path/'review')
    yield review, source
    review.close()


def test_frame_mapping_uses_recorded_host_times_not_nominal_avi_fps(project):
    review, _ = project
    details = review.details('record_p123')
    assert details['elapsed'] == pytest.approx([0., .05, .3, .9, 1.2, 1.7])
    assert [m['label'] for m in details['markers']] == ['TAKE_PULL', 'POST_RELEASE']
    assert details['first_fault_frame'] == 4
    first = cv2.imdecode(np.frombuffer(review.frame('record_p123', 0), np.uint8), cv2.IMREAD_COLOR)
    last = cv2.imdecode(np.frombuffer(review.frame('record_p123', 5), np.uint8), cv2.IMREAD_COLOR)
    assert first.shape == last.shape == (48, 64, 3)
    assert last.mean()-first.mean() > 100


def test_review_starts_empty_and_never_accepts_keyboard_labels_as_truth(project):
    review, _ = project
    assert review.details('record_p123')['intervals'] == []
    assert not list(review.output.rglob('annotations/*.json'))


def test_annotation_provenance_unknown_gaps_and_original_logs_are_preserved(project):
    review, source = project
    original = {p.relative_to(source):p.read_bytes() for p in source.rglob('*') if p.is_file()}
    saved = review.save('record_p123', [dict(start_frame=1, end_frame=3, label='HOLD_NO_PULL')])
    item = saved['intervals'][0]
    assert item['label'] == 'HOLD_NO_PULL'  # Reviewer may correct the original P hint.
    assert item['start_monotonic'] == 10.05 and item['end_monotonic'] == 10.9
    assert item['end_exclusive'] and not item['policy_fault_present']
    document = json.loads(Path(saved['saved_to']).read_text())
    assert document['physical_grasp_group'] == 'grasp01'
    assert document['split'] == 'train' and document['release_validation'] is False
    assert len(document['intervals']) == 1  # Other frames remain unannotated, not negative.
    assert 'unannotated frames remain UNKNOWN' in document['label_semantics']
    for relative, content in original.items():
        assert (source/relative).read_bytes() == content


def test_fault_is_separate_from_operator_action_label_and_cannot_be_erased(project):
    review, _ = project
    saved = review.save('record_p123', [dict(start_frame=3, end_frame=5, label='TAKE_PULL',
        policy_fault_present=False)])
    assert saved['intervals'][0]['label'] == 'TAKE_PULL'
    assert saved['intervals'][0]['policy_fault_present'] is True


@pytest.mark.parametrize('intervals', [
    [dict(start_frame=0, end_frame=6, label='NO_CONTACT')],
    [dict(start_frame=2, end_frame=2, label='NO_CONTACT')],
    [dict(start_frame=-1, end_frame=2, label='NO_CONTACT')],
    [dict(start_frame=0, end_frame=2, label='SUPPORT_VERIFIED')],
    [dict(start_frame=0, end_frame=3, label='NO_CONTACT'),
     dict(start_frame=2, end_frame=4, label='TAKE_PULL')],
    ['invalid'],
])
def test_invalid_or_overlapping_annotations_are_not_saved(project, intervals):
    review, _ = project
    with pytest.raises(ValueError):
        review.save('record_p123', intervals)
    assert not review.annotations_path('record_p123').exists()


def test_edits_are_persistent_and_keep_one_previous_revision(project):
    review, _ = project
    review.save('record_p123', [dict(start_frame=0, end_frame=2, label='UNKNOWN')])
    review.save('record_p123', [dict(start_frame=0, end_frame=2, label='ARM_DISTURBANCE')])
    path = review.annotations_path('record_p123')
    assert review.details('record_p123')['intervals'][0]['label'] == 'ARM_DISTURBANCE'
    assert json.loads(path.with_suffix('.json.bak').read_text())['intervals'][0]['label'] == 'UNKNOWN'
    assert not path.with_suffix('.json.tmp').exists()


def test_source_content_change_is_not_silently_attached_to_old_annotations(project):
    review, _ = project
    review.save('record_p123', [])
    path = review.annotations_path('record_p123')
    data = json.loads(path.read_text());data['source_sha256'] = 'different source'
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='different source'):
        review.details('record_p123')


def test_unknown_session_and_bad_frame_cannot_access_other_paths(project):
    review, _ = project
    with pytest.raises(ValueError):
        review.frame('../../some-file', 0)
    with pytest.raises(ValueError):
        review.frame('record_p123', 99)
    with pytest.raises(ValueError):
        review.save('../outside', [])


def test_locating_sources_keeps_reserved_new_grasp_separate_from_training(project, tmp_path):
    _, source = project
    heldout = tmp_path/'new_grasp/handover_fusion/new_p456'
    heldout.mkdir(parents=True)
    reservation = tmp_path/'reservation.json'
    reservation.write_text(json.dumps(dict(source=str(heldout), proposed_group='new_grasp_2140')))
    found = locate_sources(tmp_path/'train', reservation)
    assert found == [(source, 'train', 'grasp01'), (heldout, 'test_candidate', 'new_grasp_2140')]


def test_old_player_cannot_overwrite_annotations_after_sync_fix(project):
    import io
    review, _ = project
    review.save('record_p123', [dict(start_frame=0,end_frame=2,label='UNKNOWN')])
    path = review.annotations_path('record_p123')
    original = path.read_bytes()
    body = json.dumps(dict(id='record_p123',intervals=[])).encode()
    handler = handler_for(review).__new__(handler_for(review))
    handler.path='/api/save';handler.headers={'Content-Length':str(len(body))};handler.rfile=io.BytesIO(body)
    replies=[];handler.respond=lambda content, **kwargs: replies.append((content,kwargs))
    handler.do_POST()
    assert replies[0][1]['status']==409
    assert path.read_bytes()==original
    assert review.index()['player_version']==REVIEW_PLAYER_VERSION


def test_frame_endpoint_returns_explicit_identity_for_image_time_consistency(project):
    review, _ = project
    cls=handler_for(review);handler=cls.__new__(cls)
    handler.path='/api/frame?id=record_p123&index=3'
    replies=[];handler.respond=lambda body, content_type, **kwargs: replies.append((body,content_type,kwargs))
    handler.do_GET()
    assert replies[0][1]=='image/jpeg'
    assert replies[0][2]['headers']=={'X-Review-Session':'record_p123','X-Review-Frame':'3'}
    frame=cv2.imdecode(np.frombuffer(replies[0][0],np.uint8),cv2.IMREAD_COLOR)
    assert 80<frame.mean()<100


def interrupt_avi(path, *, truncate_last=False):
    """Mirror the real interrupted AVI: unknown sizes, zero counts, no idx1."""
    import struct
    index=IndexedMJPEGVideo(path)
    content=bytearray(path.read_bytes())
    struct.pack_into('<I',content,4,0xffffffff)
    movi=content.find(b'movi')
    struct.pack_into('<I',content,movi-4,0xffffffff)
    struct.pack_into('<I',content,content.find(b'avih')+8+16,0)
    struct.pack_into('<I',content,content.find(b'strh')+8+32,0)
    offset,size=index.frames[-1]
    stop=offset+(size//2 if truncate_last else size+size%2)
    path.write_bytes(content[:stop])


def test_unfinalized_avi_random_seek_matches_independent_sequential_pixels(project, tmp_path):
    review,source=project
    video=source/'global/global_rgb.avi'
    # The oracle is a fresh sequential decoder before header corruption, not
    # another call to the same frame endpoint being tested.
    cap=cv2.VideoCapture(str(video));expected=[]
    for _ in range(6):
        ok,frame=cap.read();assert ok
        expected.append(frame)
    cap.release()
    interrupt_avi(video)
    rebuilt=ReviewProject([(source,'train','grasp01')],tmp_path/'rebuilt')
    for i in [4,1,5,0,3,4,0]:
        actual=cv2.imdecode(np.frombuffer(rebuilt.frame('record_p123',i),np.uint8),cv2.IMREAD_COLOR)
        # FFmpeg AVI decoding and OpenCV JPEG decoding can round YUV by 1–2.
        assert np.max(np.abs(actual.astype(int)-expected[i].astype(int)))<=3
    assert rebuilt.details('record_p123')['unfinalized_avi'] is True
    assert rebuilt.details('record_p123')['readable_frames']==6


def test_incomplete_tail_is_not_exposed_as_a_fabricated_frame(project, tmp_path):
    _,source=project
    interrupt_avi(source/'global/global_rgb.avi',truncate_last=True)
    rebuilt=ReviewProject([(source,'train','grasp01')],tmp_path/'rebuilt')
    details=rebuilt.details('record_p123')
    assert details['recorded_frames']==6 and details['readable_frames']==5
    assert details['unavailable_tail_frames']==1
    assert details['elapsed']==pytest.approx([0.,.05,.3,.9,1.2])
    with pytest.raises(ValueError,match='outside'):
        rebuilt.frame('record_p123',5)
    with pytest.raises(ValueError,match='boundaries'):
        rebuilt.save('record_p123',[dict(start_frame=0,end_frame=5,label='NO_CONTACT')])


def test_old_labels_on_unfinalized_video_require_explicit_boundary_recheck(project, tmp_path):
    review,source=project
    review.save('record_p123',[dict(start_frame=0,end_frame=2,label='NO_CONTACT',
                                  player_version='20261110-sync2')])
    interrupt_avi(source/'global/global_rgb.avi')
    rebuilt=ReviewProject([(source,'train','grasp01')],review.output)
    assert rebuilt.details('record_p123')['annotation_frames_need_recheck']==[1]
    result=rebuilt.save('record_p123',[dict(start_frame=0,end_frame=2,label='NO_CONTACT',
                                          player_version=REVIEW_PLAYER_VERSION)])
    assert result['intervals'][0]['frame_reader_version']==FRAME_READER_VERSION
    assert rebuilt.details('record_p123')['annotation_frames_need_recheck']==[]


def test_damaged_middle_chunk_is_rejected_without_shifting_later_frame_ids(project):
    _,source=project
    path=source/'global/global_rgb.avi'
    index=IndexedMJPEGVideo(path)
    content=bytearray(path.read_bytes())
    offset,_=index.frames[2]
    content[offset:offset+2]=b'xx'
    path.write_bytes(content)
    with pytest.raises(ValueError,match='damaged AVI frame'):
        IndexedMJPEGVideo(path)


def test_recorded_video_gaps_are_exposed_and_persist_as_separate_provenance(project):
    review,_=project
    gaps=review.details('record_p123')['video_gaps']
    assert len(gaps)==1
    gap=gaps[0]
    assert (gap['before_frame'],gap['after_frame'])==(2,3)
    assert gap['start_elapsed']==pytest.approx(.3)
    assert gap['end_elapsed']==pytest.approx(.9)
    assert gap['duration_seconds']==pytest.approx(.6)
    result=review.save('record_p123',[dict(start_frame=0,end_frame=4,label='NO_CONTACT')])
    assert result['intervals'][0]['label']=='NO_CONTACT'
    assert result['intervals'][0]['video_gap_present'] is True


def test_sync3_client_can_save_valid_boundaries_without_losing_its_open_draft(project):
    import io
    review,_=project
    body=json.dumps(dict(id='record_p123',player_version='20261110-sync3',intervals=[
        dict(start_frame=0,end_frame=2,label='HOLD_NO_PULL',player_version='20261110-sync3')])).encode()
    cls=handler_for(review);handler=cls.__new__(cls)
    handler.path='/api/save';handler.headers={'Content-Length':str(len(body))}
    handler.rfile=io.BytesIO(body);replies=[]
    handler.respond=lambda content,**kwargs:replies.append((content,kwargs))
    handler.do_POST()
    assert replies[0][1].get('status',200)==200
    assert review.details('record_p123')['intervals'][0]['frame_reader_version']==FRAME_READER_VERSION
