#!/usr/bin/env python3
"""Short read-only Orbbec/arm timing recording. No calibration or motion commands."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import json
from pathlib import Path
import sys
import threading
import time

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cameras.options import add_camera_arguments, apply_pipeline_defaults
from cameras.orbbec import OrbbecCamera
from tools.calibrate_d435i_charuco import NeroReadOnlyFeedback


class TimingCamera(OrbbecCamera):
    """Observe SDK metadata while retaining the production latest-frame queue."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.recording = threading.Event()
        self.metadata_lock = threading.Lock()
        self.callback_rows = []
        self.by_sequence = {}
        self.global_timestamp_supported = False

    def _on_frame(self, frames):
        received = time.monotonic()
        wall = time.time()
        color = frames.get_color_frame()
        if self.recording.is_set() and color is not None:
            row = dict(sequence=int(color.get_index()), sensor_us=int(color.get_timestamp_us()),
                       system_us=int(color.get_system_timestamp_us()),
                       global_us=(int(color.get_global_timestamp_us())
                                  if self.global_timestamp_supported else 0),
                       callback_monotonic_s=received, callback_wall_s=wall)
            with self.metadata_lock:
                self.callback_rows.append(row)
                self.by_sequence[row['sequence']] = row
        super()._on_frame(frames)


def fit_clock(sensor_us, host_s):
    """Fit a clock relation; intercept includes unknown constant transport latency."""
    sensor = np.asarray(sensor_us, float) * 1e-6
    host = np.asarray(host_s, float)
    x = sensor - sensor[0]
    y = host - host[0]
    slope, intercept = np.linalg.lstsq(np.c_[x, np.ones(len(x))], y, rcond=None)[0]
    residual = (y - (slope*x + intercept))*1000.
    return dict(slope=float(slope), rate_difference_ppm=float((slope-1.)*1e6),
                sensor_origin_s=float(sensor[0]), host_origin_s=float(host[0]),
                centered_intercept_ms=float(intercept*1000.),
                raw_host_minus_sensor_ms_median=float(np.median(host-sensor)*1000.),
                residual_median_ms=float(np.median(residual)),
                residual_p01_ms=float(np.percentile(residual, 1)),
                residual_p99_ms=float(np.percentile(residual, 99)),
                residual_peak_to_peak_ms=float(np.ptp(residual)),
                meaning='Clock relation plus delivery delay; not an identified camera-to-motion offset')


def summarize(callbacks, consumed, joints, fps):
    if min(len(callbacks), len(consumed), len(joints)) < 2:
        raise ValueError('Need at least two camera callbacks, consumed frames and joint records')
    device = np.array([r['sensor_us'] for r in callbacks], float)
    system = np.array([r['system_us'] for r in callbacks], float)
    global_time = np.array([r['global_us'] for r in callbacks], float)
    callback = np.array([r['callback_monotonic_s'] for r in callbacks])
    wall = np.array([r['callback_wall_s'] for r in callbacks])
    sequence = np.array([r['sequence'] for r in callbacks])
    taken = np.array([r['sequence'] for r in consumed])
    queue_ms = np.array([r['callback_to_read_ms'] for r in consumed])
    qt = np.array([r['request_monotonic_s'] for r in joints])
    stamps = np.array([r['joint_feedback_wall_s'] for r in joints])
    q = np.rad2deg([r['q_rad'] for r in joints])
    fresh = np.r_[True, np.diff(stamps) > 0]
    fit = fit_clock(device, callback)
    report = dict(
        frames_callback=len(callbacks), frames_consumed=len(consumed),
        camera_callback_hz=float((len(callbacks)-1)/(callback[-1]-callback[0])),
        consumed_sequence_nonincreasing=int(np.count_nonzero(np.diff(taken) <= 0)),
        callback_sequence_nonincreasing=int(np.count_nonzero(np.diff(sequence) <= 0)),
        callback_sequence_gap_frames=int(np.maximum(np.diff(sequence)-1, 0).sum()),
        consumed_sequence_gap_frames=int(np.maximum(np.diff(taken)-1, 0).sum()),
        device_timestamp_nonincreasing=int(np.count_nonzero(np.diff(device) <= 0)),
        sdk_system_timestamp_nonincreasing=int(np.count_nonzero(np.diff(system) <= 0)),
        consumed_device_timestamp_nonincreasing=int(np.count_nonzero(np.diff([r['sensor_us'] for r in consumed]) <= 0)),
        callback_to_read_ms=dict(median=float(np.median(queue_ms)), p99=float(np.percentile(queue_ms, 99)), max=float(queue_ms.max())),
        callback_to_read_over_two_frame_periods=int(np.count_nonzero(queue_ms > 2000./fps)),
        device_to_callback_monotonic_fit=fit,
        host_wall_minus_monotonic_step_max_ms=float(np.max(np.abs(np.diff(wall-callback)))*1000.),
        joint_polls=len(joints), joint_unique_feedback=int(fresh.sum()),
        joint_poll_hz=float((len(joints)-1)/(qt[-1]-qt[0])),
        joint_unique_feedback_observed_hz=float(max(int(fresh.sum())-1, 0)/(qt[-1]-qt[0])),
        joint_feedback_timestamp_backwards=int(np.count_nonzero(np.diff(stamps) < 0)),
        joint_feedback_age_max_ms=max(r['joint_feedback_age_ms'] for r in joints),
        joint_poll_interval_max_ms=float(np.diff(qt).max()*1000.),
        joint_range_deg=np.ptp(q, axis=0).tolist(),
        camera_to_motion_time_offset_s=None,
        old_dataset_repairable=None,
        interpretation='Static run cannot identify fixed camera-to-motion delay or certify historical frame/pose pairing',
    )
    if np.all(system > 0):
        report['device_to_sdk_system_fit'] = fit_clock(device, system*1e-6)
    valid = global_time > 0
    report['global_timestamp_valid_frames'] = int(valid.sum())
    if np.any(valid):
        age_ms = (system[valid]-global_time[valid])*.001
        report['sdk_system_minus_global_ms'] = dict(median=float(np.median(age_ms)),
                                                   min=float(age_ms.min()), max=float(age_ms.max()),
                                                   meaning='SDK global clock mapping dependent; not independent latency ground truth')
    return report


def record_joints(arm, stop, rows, hz):
    """Poll feedback in parallel with image processing; preserve duplicate stamps."""
    next_read = time.monotonic()
    while not stop.is_set():
        before = time.monotonic()
        wall = time.time()
        q, stamp = arm.read_joints()
        pose, _, flange_stamp = arm.read()
        after = time.monotonic()
        rows.append(dict(request_monotonic_s=before, request_wall_s=wall,
                         return_monotonic_s=after, joint_feedback_wall_s=stamp,
                         flange_feedback_wall_s=flange_stamp,
                         joint_feedback_age_ms=(time.time()-stamp)*1000.,
                         q_rad=q.tolist(), flange_pose_xyzrpy=pose.tolist()))
        next_read += 1./hz
        stop.wait(max(0., next_read-time.monotonic()))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    add_camera_arguments(parser)
    parser.add_argument('--duration', type=float, default=20.)
    parser.add_argument('--joint-hz', type=float, default=100.)
    parser.add_argument('--channel', default='can0')
    parser.add_argument('--firmware', choices=('V111', 'V112'), default='V112')
    parser.add_argument('--arm-state', choices=('enabled-hold', 'teaching'), required=True,
                        help='Operator label only; never changes the controller mode')
    parser.add_argument('--preview', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--save-dir', type=Path, default=Path('calibration/orbbec_sync_stream'))
    apply_pipeline_defaults(parser, argv)
    args = parser.parse_args(argv)
    if args.camera_backend != 'orbbec':
        parser.error('Use --config config/pipeline_orbbec.yaml or --camera-backend orbbec')
    if not np.isfinite([args.duration, args.joint_hz]).all() or not 2. <= args.duration <= 120. or not 1. <= args.joint_hz <= 200.:
        parser.error('Require duration 2..120 seconds and joint-hz 1..200')
    return args


def main(argv=None):
    args = parse_args(argv)
    out = args.save_dir/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    out.mkdir(parents=True)
    camera = TimingCamera(serial=args.camera_serial, width=args.camera_width,
        height=args.camera_height, fps=args.camera_fps, timeout=args.camera_timeout,
        warmup=args.camera_warmup, enable_depth=False)
    stop = threading.Event()
    consumed, joints = [], []
    status = 'incomplete'
    writer = None
    start = None
    print('READ ONLY: camera + arm feedback; no enable, mode change, motion, stop, or gripper command.')
    print(f'Operator mode label: {args.arm_state}. Keep the arm stationary. Duration: {args.duration:g}s.')
    print('Log:', out, flush=True)
    try:
        with camera, NeroReadOnlyFeedback(args.channel, args.firmware, 5.) as arm:
            device = camera.pipeline.get_device()
            camera.global_timestamp_supported = device.is_global_timestamp_supported()
            state = arm.arm.get_arm_status()
            actual_state = (dict(ctrl_mode=str(state.msg.ctrl_mode),
                                 mode_feedback=str(state.msg.mode_feedback),
                                 arm_status=str(state.msg.arm_status), timestamp=state.timestamp)
                            if state is not None else None)
            (out/'session.json').write_text(json.dumps(dict(
                operator_arm_state=args.arm_state, sdk_arm_state_at_start=actual_state,
                serial=camera.serial, width=camera.width, height=camera.height, fps=camera.fps,
                firmware_profile=args.firmware, requested_joint_poll_hz=args.joint_hz,
                global_timestamp_supported=camera.global_timestamp_supported,
                camera_clock_settings_changed=False, robot_commands_sent=False), indent=2)+'\n')
            writer = cv2.VideoWriter(str(out/'rgb.avi'), cv2.VideoWriter_fourcc(*'MJPG'),
                                     camera.fps, (camera.width, camera.height))
            if not writer.isOpened():
                raise RuntimeError('Cannot open MJPG video output')
            title = 'Static sync audit | Q/Esc exits | no robot commands'
            if args.preview:
                cv2.namedWindow(title, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(title, 1100, 650)
            with ThreadPoolExecutor(max_workers=1) as pool:
                try:
                    start = time.monotonic()
                    future = pool.submit(record_joints, arm, stop, joints, args.joint_hz)
                    camera.recording.set()
                    last_print = start
                    while time.monotonic()-start < args.duration:
                        if future.done():
                            future.result()
                        frame = camera.read()
                        returned = time.monotonic()
                        with camera.metadata_lock:
                            source = camera.by_sequence.get(frame.sequence)
                            row = source.copy() if source is not None else None
                        # One already-queued warmup frame may precede recording.
                        if row is None:
                            continue
                        row.update(video_frame_index=len(consumed), read_return_monotonic_s=returned,
                                   callback_to_read_ms=(returned-row['callback_monotonic_s'])*1000.)
                        consumed.append(row)
                        writer.write(frame.bgr)
                        elapsed = returned-start
                        if args.preview:
                            visual = frame.bgr.copy()
                            cv2.rectangle(visual, (0, 0), (camera.width, 62), (20, 20, 20), -1)
                            cv2.putText(visual, f'{args.arm_state} | {elapsed:.1f}/{args.duration:g}s | seq {frame.sequence} | joints {len(joints)}',
                                (12, 24), cv2.FONT_HERSHEY_SIMPLEX, .65, (255, 255, 255), 1)
                            cv2.putText(visual, f'callback-to-read {row["callback_to_read_ms"]:.1f}ms (NOT exposure delay) | Q exits',
                                (12, 50), cv2.FONT_HERSHEY_SIMPLEX, .58, (0, 230, 230), 1)
                            cv2.imshow(title, visual)
                            if cv2.waitKey(1)&255 in (27, ord('q'), ord('Q')) or cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1:
                                status = 'operator_stopped'
                                break
                        if returned-last_print >= 2.:
                            print(f'[record] {elapsed:.1f}s frames={len(consumed)} joint_polls={len(joints)} callback_to_read={row["callback_to_read_ms"]:.1f}ms', flush=True)
                            last_print = returned
                    else:
                        status = 'completed'
                finally:
                    camera.recording.clear()
                    stop.set()
                future.result()
    except KeyboardInterrupt:
        status = 'interrupted'
        print('Stopped; partial records retained.')
    finally:
        stop.set()
        camera.recording.clear()
        if writer is not None:
            writer.release()
        if args.preview:
            cv2.destroyAllWindows()
        for filename, rows in (('callbacks.jsonl', camera.callback_rows), ('frames.jsonl', consumed), ('joints.jsonl', joints)):
            with (out/filename).open('w') as stream:
                for row in rows:
                    stream.write(json.dumps(row)+'\n')
        report = dict(status=status, operator_arm_state=args.arm_state, robot_commands_sent=False,
                      raw_calibration_changed=False, camera_to_motion_time_offset_s=None,
                      limitations=['Static clock data cannot certify exposure-to-motion alignment.',
                                   'A fixed clock bias cannot repair old image/pose pairs without historical robot trajectories.',
                                   'One static mode recording does not establish historical exposure-to-motion timing.',
                                   'SDK feedback stamps are host-side feedback timestamps, not independently synchronized robot controller exposure stamps.',
                                   'Video playback uses nominal FPS; frames.jsonl holds actual per-frame times.'])
        if min(len(camera.callback_rows), len(consumed), len(joints)) >= 2:
            report.update(summarize(camera.callback_rows, consumed, joints, args.camera_fps))
        (out/'summary.json').write_text(json.dumps(report, indent=2)+'\n')
        print('[SAVED]', out/'summary.json', flush=True)
    return report


if __name__ == '__main__':
    main()
