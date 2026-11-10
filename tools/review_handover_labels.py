#!/usr/bin/env python3
"""Local video-only action review. No cameras, CAN, robot or model inference.

Serves stored MJPEG AVI frames without ffmpeg. Source timestamps, rather than
nominal AVI FPS, define playback and annotation times. Originals stay read-only.
"""
from __future__ import annotations

import argparse
from bisect import bisect_left, bisect_right
from datetime import datetime
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import sys
import threading
from urllib.parse import parse_qs, urlparse
from zoneinfo import ZoneInfo

import cv2
import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from tools.indexed_mjpeg_avi import IndexedMJPEGVideo, FRAME_READER_VERSION
LABELS = ('NO_CONTACT', 'HOLD_NO_PULL', 'TAKE_PULL', 'BUMP_PUSH',
          'POST_RELEASE', 'ARM_DISTURBANCE', 'UNKNOWN')
DEFAULT_OUTPUT = PROJECT/'artifacts/handover_label_review_20261109'
REVIEW_PLAYER_VERSION = '20261110-sync4'
COMPATIBLE_SAVE_VERSIONS = {'20261110-sync3', REVIEW_PLAYER_VERSION}
VIDEO_GAP_NOTICE_SECONDS = .5


def read_jsonl(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def video_gaps(monotonic):
    """Long intervals without recorded images, independent of robot validity."""
    return [dict(before_frame=i, after_frame=i+1, start_monotonic=a,
                 end_monotonic=b, duration_seconds=b-a)
            for i,(a,b) in enumerate(zip(monotonic, monotonic[1:]))
            if b-a > VIDEO_GAP_NOTICE_SECONDS]


def locate_sources(root, reservation):
    result = []
    for meta in sorted(Path(root).rglob('metadata.json')):
        if meta.parent.parent.name != 'handover_fusion':
            continue
        rows = read_jsonl(meta.parent/'samples.jsonl')
        if not any((r.get('policy') or {}).get('baseline_nm') is not None for r in rows):
            continue
        data = json.loads(meta.read_text())
        if data.get('control_mode') != 'mit_cartesian_hold':
            continue
        result.append((meta.parent, (data.get('collection') or {}).get('split') or 'unassigned',
                       (data.get('collection') or {}).get('group') or 'unassigned'))
    if reservation and Path(reservation).is_file():
        data = json.loads(Path(reservation).read_text())
        source = Path(data['source'])
        if not source.is_absolute():
            source = PROJECT/source
        if source not in [x[0] for x in result]:
            result.append((source, 'test_candidate', data.get('proposed_group', 'reserved_test')))
    return result


class ReviewProject:
    def __init__(self, sources, output):
        self.output = Path(output).resolve()
        self.output.mkdir(parents=True, exist_ok=True)
        self.sessions = {}
        self.lock = threading.Lock()
        for directory, split, group in sources:
            directory = Path(directory).resolve()
            meta = json.loads((directory/'metadata.json').read_text())
            observations = read_jsonl(directory/'global/observations.jsonl')
            samples = read_jsonl(directory/'samples.jsonl')
            if not observations or not samples:
                continue
            frame_ids = [int(o['video_index']) for o in observations]
            if frame_ids != list(range(len(observations))):
                raise ValueError(f'Non-contiguous AVI mapping: {directory}')
            video = IndexedMJPEGVideo(directory/'global/global_rgb.avi')
            recorded_frames = len(observations)
            if len(video) > recorded_frames:
                raise ValueError(f'AVI has more frames than the timestamp log: {directory}')
            # A killed writer can leave a partial JPEG and a few buffered frames
            # unwritten. Do not show a fabricated image for those logged times.
            observations = observations[:len(video)]
            monotonic = [float(o['received_monotonic']) for o in observations]
            if any(b <= a for a, b in zip(monotonic, monotonic[1:])):
                raise ValueError(f'Non-increasing frame timestamps: {directory}')
            sample_times = [float(r['monotonic']) for r in samples]
            if any(b <= a for a, b in zip(sample_times, sample_times[1:])):
                raise ValueError(f'Non-increasing fusion timestamps: {directory}')
            # Coalesce auto-repeat keys; retain a hint, never an accepted label.
            markers = []
            previous_label = None
            for r in samples:
                for e in r.get('events', []):
                    if e.get('action') == 'label' and e.get('label') != previous_label:
                        previous_label = e['label']
                        index = bisect_left(monotonic, float(e['at']))
                        if index < len(monotonic):
                            markers.append(dict(index=index, label=e['label'], at=e['at']))
            identifier = directory.name
            if identifier in self.sessions:
                raise ValueError('Duplicate review session identifiers')
            fault_times = [r['monotonic'] for r in samples if (r.get('policy') or {}).get('phase') == 'FUSION_FAULT']
            clock_fault_times = []
            from tools.check_handover_dataset import read_inner_faults
            for f in read_inner_faults(directory):
                if isinstance(f.get('monotonic_time'), (int, float)):
                    clock_fault_times.append(f['monotonic_time'])
            self.sessions[identifier] = dict(id=identifier, directory=directory,
                split=split, group=group, observations=observations,
                monotonic=monotonic, sample_times=sample_times, samples=samples,
                markers=markers, first_fault=min(fault_times) if fault_times else None,
                inner_fault=min(clock_fault_times) if clock_fault_times else None,
                sha256=hashlib.sha256((directory/'samples.jsonl').read_bytes()).hexdigest(),
                video=video, recorded_frames=recorded_frames)
            self.sessions[identifier]['video_gaps']=video_gaps(monotonic)
        if not self.sessions:
            raise ValueError('No reviewable sessions found')

    def index(self):
        return dict(labels=list(LABELS), player_version=REVIEW_PLAYER_VERSION, output=str(self.output), sessions=[dict(
            id=s['id'], group=s['group'], split=s['split'], frames=len(s['monotonic']),
            recorded_frames=s['recorded_frames'], unfinalized_avi=s['video'].unfinalized,
            duration_seconds=s['monotonic'][-1]-s['monotonic'][0], markers=len(s['markers']))
            for s in self.sessions.values()])

    def annotations_path(self, identifier):
        self.session(identifier)
        return self.output/'annotations'/f'{identifier}.json'

    def session(self, identifier):
        if identifier not in self.sessions:
            raise ValueError('Unknown review session')
        return self.sessions[identifier]

    def details(self, identifier):
        s = self.session(identifier)
        phases = []
        for now in s['monotonic']:
            i = max(0, bisect_right(s['sample_times'], now)-1)
            phases.append((s['samples'][i].get('policy') or {}).get('phase', 'unknown'))
        path = self.annotations_path(identifier)
        existing = json.loads(path.read_text()) if path.exists() else None
        if existing and existing.get('source_sha256') != s['sha256']:
            raise ValueError('Saved annotations refer to different source content')
        intervals=[] if existing is None else existing['intervals']
        unverified=[i+1 for i,a in enumerate(intervals) if s['video'].unfinalized
                    and a.get('frame_reader_version') != FRAME_READER_VERSION]
        return dict(id=identifier, split=s['split'], group=s['group'],
            frame_reader_version=FRAME_READER_VERSION,
            recorded_frames=s['recorded_frames'], readable_frames=len(s['video']),
            unavailable_tail_frames=s['recorded_frames']-len(s['video']),
            unfinalized_avi=s['video'].unfinalized,
            annotation_frames_need_recheck=unverified,
            video_gaps=[dict(gap, start_elapsed=gap['start_monotonic']-s['monotonic'][0],
                end_elapsed=gap['end_monotonic']-s['monotonic'][0]) for gap in s['video_gaps']],
            approximate_start_wall=s['samples'][0]['wall_time']-s['sample_times'][0]+s['monotonic'][0],
            elapsed=[t-s['monotonic'][0] for t in s['monotonic']],
            phases=phases, markers=s['markers'],
            first_fault_frame=None if s['first_fault'] is None else bisect_left(s['monotonic'], s['first_fault']),
            inner_fault_frame=None if s['inner_fault'] is None else bisect_left(s['monotonic'], s['inner_fault']),
            intervals=intervals)

    def frame(self, identifier, index):
        s = self.session(identifier)
        if type(index) is not int or not 0 <= index < len(s['monotonic']):
            raise ValueError('Frame index outside recorded video')
        content = s['video'].jpeg(index)
        frame = cv2.imdecode(np.frombuffer(content, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError('Cannot decode the requested stored frame')
        ok, encoded = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 88])
        if not ok:
            raise ValueError('JPEG encoding failed')
        return encoded.tobytes()

    def save(self, identifier, intervals):
        s = self.session(identifier)
        if not isinstance(intervals, list):
            raise ValueError('Intervals must be a list')
        result = []
        for item in intervals:
            if not isinstance(item, dict):
                raise ValueError('Each interval must be an object')
            start, end, label = item.get('start_frame'), item.get('end_frame'), item.get('label')
            if (type(start) is not int or type(end) is not int
                    or not 0 <= start < end < len(s['monotonic']) or label not in LABELS):
                raise ValueError('Invalid label or interval boundaries')
            begin, finish = s['monotonic'][start], s['monotonic'][end]
            begin_i = max(0, bisect_right(s['sample_times'], begin)-1)
            end_i = bisect_left(s['sample_times'], finish)
            source_rows = s['samples'][begin_i:end_i]
            result.append(dict(start_frame=start, end_frame=end, label=label,
                start_monotonic=begin, end_monotonic=finish, end_exclusive=True,
                reviewed_by='operator', evidence='operator_video_review',
                note=str(item.get('note', ''))[:1000],
                player_version=str(item.get('player_version', 'legacy_unspecified'))[:64],
                frame_reader_version=(FRAME_READER_VERSION if item.get('player_version') in COMPATIBLE_SAVE_VERSIONS
                    else str(item.get('frame_reader_version', 'legacy_unspecified'))[:64]),
                policy_fault_present=any((r.get('policy') or {}).get('phase')=='FUSION_FAULT' for r in source_rows),
                online_baseline_present=all((r.get('policy') or {}).get('baseline_nm') is not None for r in source_rows) and bool(source_rows),
                inner_fault_present=s['inner_fault'] is not None and finish>s['inner_fault']))
            result[-1]['video_gap_present']=any(begin<gap['end_monotonic']
                and finish>gap['start_monotonic'] for gap in s['video_gaps'])
        result.sort(key=lambda x: x['start_frame'])
        if any(b['start_frame'] < a['end_frame'] for a, b in zip(result, result[1:])):
            raise ValueError('Intervals overlap; delete or shorten the conflicting interval')
        document = dict(version=1, source=str(s['directory']), source_sha256=s['sha256'],
            physical_grasp_group=s['group'], split=s['split'],
            reviewed_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
            label_semantics='Manual action intervals, not load/support truth; unannotated frames remain UNKNOWN.',
            release_validation=False, intervals=result)
        path = self.annotations_path(identifier)
        path.parent.mkdir(parents=True, exist_ok=True)
        encoded = json.dumps(document, ensure_ascii=False, indent=2)+'\n'
        with self.lock:
            if path.exists():
                path.with_suffix('.json.bak').write_text(path.read_text())
            temporary = path.with_suffix('.json.tmp')
            temporary.write_text(encoded)
            temporary.replace(path)
        return dict(saved_to=str(path), intervals=result)

    def close(self):
        pass  # Frame reads use independent, short-lived read-only file handles.


def handler_for(project):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def respond(self, body, content_type='application/json; charset=utf-8', status=200, headers=None):
            if isinstance(body, (dict, list)):
                body = json.dumps(body, ensure_ascii=False).encode()
            elif isinstance(body, str):
                body = body.encode()
            self.send_response(status)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            for key, value in (headers or {}).items():
                self.send_header(key, value)
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass  # A newer seek intentionally cancelled this image request.

        def do_GET(self):
            parsed = urlparse(self.path)
            query = parse_qs(parsed.query)
            try:
                if parsed.path == '/':
                    self.respond((PROJECT/'tools/handover_label_review.html').read_bytes(), 'text/html; charset=utf-8')
                elif parsed.path in ('/review_player.js', '/review_ui.js'):
                    filename = 'handover_review_player.js' if parsed.path == '/review_player.js' else 'handover_review_ui.js'
                    self.respond((PROJECT/'tools'/filename).read_bytes(), 'application/javascript; charset=utf-8')
                elif parsed.path == '/api/index':
                    self.respond(project.index())
                elif parsed.path == '/api/session':
                    self.respond(project.details(query['id'][0]))
                elif parsed.path == '/api/frame':
                    identifier, index = query['id'][0], int(query['index'][0])
                    self.respond(project.frame(identifier, index), 'image/jpeg', headers={
                        'X-Review-Session': identifier, 'X-Review-Frame': str(index)})
                else:
                    self.respond(dict(error='Not found'), status=404)
            except (ValueError, KeyError, OSError) as exc:
                self.respond(dict(error=str(exc)), status=400)

        def do_POST(self):
            try:
                if self.path != '/api/save':
                    self.respond(dict(error='Not found'), status=404)
                    return
                length = int(self.headers.get('Content-Length', '0'))
                if not 0 < length <= 2_000_000:
                    raise ValueError('Invalid request length')
                data = json.loads(self.rfile.read(length))
                if data.get('player_version') not in COMPATIBLE_SAVE_VERSIONS:
                    self.respond(dict(error='播放器已更新，请按 Ctrl+Shift+R 强制刷新后再保存。已有标注保留。'), status=409)
                    return
                self.respond(project.save(data['id'], data['intervals']))
            except (ValueError, KeyError, TypeError, OSError) as exc:
                self.respond(dict(error=str(exc)), status=400)
    return Handler


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=PROJECT/'logs/handover_dataset_20261109')
    p.add_argument('--reservation', type=Path, default=PROJECT/'artifacts/handover_test_candidate_20261109_2140/reservation.json')
    p.add_argument('--out', type=Path, default=DEFAULT_OUTPUT)
    p.add_argument('--port', type=int, default=8765)
    p.add_argument('--check', action='store_true', help='Validate stored-frame decoding and write the review queue, without starting a server')
    args = p.parse_args(argv)
    project = ReviewProject(locate_sources(args.root, args.reservation), args.out)
    try:
        (project.output/'review_index.json').write_text(json.dumps(project.index(), ensure_ascii=False, indent=2)+'\n')
        if args.check:
            for identifier, s in project.sessions.items():
                project.frame(identifier, 0)
                project.frame(identifier, len(s['monotonic'])-1)
            print(json.dumps(project.index(), ensure_ascii=False, indent=2))
            return 0
        server = ThreadingHTTPServer(('127.0.0.1', args.port), handler_for(project))
        print(f'离线核对页面：http://127.0.0.1:{args.port}', flush=True)
        print(f'仅读取录像；没有相机/CAN/机械臂连接。标注保存到 {project.output}/annotations', flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            server.server_close()
    finally:
        project.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
