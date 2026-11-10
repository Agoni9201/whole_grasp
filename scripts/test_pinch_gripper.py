#!/usr/bin/env python3
"""RGB pinch aperture -> EMPTY-gripper opening; optional explicit CAN hold preparation."""
from __future__ import annotations

import argparse
from collections import deque
from datetime import datetime
import json
import math
from pathlib import Path
import sys
import time

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cameras.options import add_camera_arguments, apply_pipeline_defaults
from cameras.realsense import create_camera
from handoff.pinch_gripper import (
    MAX_FORCE_N, AutoPinchHand, AutoPinchMapping, GripperOnlyLink, PinchMapping,
    GripperResponseWatch, GripperUnresponsive,
    WidthRamp, measure_pinch,
)
from targets.mediapipe_hand_landmarker import DEFAULT_MODEL_PATH, MediaPipeHandLandmarker
from targets.hand_selection import filter_hand_observations


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    add_camera_arguments(p)
    p.add_argument('--execute', action='store_true', help='Connect CAN and send gripper targets; otherwise preview only')
    p.add_argument('--prepare-can-hold', action='store_true',
                   help='Explicitly hand an existing Ethernet J hold to CAN at current joint positions; no enable/disable')
    p.add_argument('--channel', default='can0')
    p.add_argument('--firmware', choices=('v111', 'v112', 'v120'), default='v111')
    p.add_argument('--model', type=Path, default=DEFAULT_MODEL_PATH)
    p.add_argument('--pinch-closed-ratio', type=float, help='Override automatic learning; supply both ratio endpoints')
    p.add_argument('--pinch-open-ratio', type=float, help='Override automatic learning; supply both ratio endpoints')
    p.add_argument('--min-width-mm', type=float, default=0.)
    p.add_argument('--max-width-mm', type=float, default=85.)
    p.add_argument('--force', type=float, default=.50, help='SDK gripper force setting, capped at existing 0.50 limit')
    p.add_argument('--open-seconds', type=float, default=5., help='Full test stroke command ramp opening time')
    p.add_argument('--close-seconds', type=float, default=2., help='Full test stroke command ramp closing time')
    p.add_argument('--max-lead-mm', type=float, default=5., help='Maximum target lead over current measured opening')
    p.add_argument('--log-dir', type=Path, default=ROOT / 'logs' / 'pinch_gripper')
    apply_pipeline_defaults(p, argv)
    a = p.parse_args(argv)
    try:
        if (a.pinch_closed_ratio is None) != (a.pinch_open_ratio is None):
            raise ValueError('supply both pinch ratio endpoints, or omit both for automatic learning')
        if a.pinch_closed_ratio is not None:
            PinchMapping(a.pinch_closed_ratio, a.pinch_open_ratio)
        WidthRamp(a.min_width_mm / 1000., a.max_width_mm / 1000., a.open_seconds,
                  a.close_seconds, a.max_lead_mm / 1000.)
    except ValueError as exc:
        p.error(str(exc))
    if not math.isfinite(a.force) or not 0. < a.force <= MAX_FORCE_N:
        p.error(f'force must be finite and in (0, {MAX_FORCE_N}]')
    if not 0. < a.max_lead_mm <= 5.:
        p.error('max-lead-mm must be in (0, 5] for this empty-gripper test')
    if (min(a.camera_width, a.camera_height, a.camera_fps) <= 0
            or not math.isfinite(a.camera_timeout) or a.camera_timeout <= 0
            or a.camera_warmup < 0):
        p.error('invalid camera settings')
    return a


def joints_stationary(history, now):
    """Use measured joint motion, not the SDK's target-arrival flag."""
    recent = [(stamp, q) for stamp, q in history if 0. <= now - stamp <= .75]
    return (len(recent) >= 5 and now - recent[0][0] >= .2
            and np.max(np.ptp([q for _, q in recent], axis=0)) <= np.deg2rad(.5))


def draw_overlay(frame, observations, hand, fraction, target, command, feedback, active, reason, execute,
                 mapping, fault=None):
    image = frame.bgr.copy()
    accepted, _ = filter_hand_observations(observations, image.shape, fingertips=True)
    for obs in observations:
        trusted = any(obs is value for value in accepted)
        color = (40, 210, 40) if trusted else (100, 100, 100)
        points = obs.landmarks_pixel
        for chain in ((0, 1, 2, 3, 4), (0, 5, 6, 7, 8), (5, 9, 13, 17, 0)):
            for i, j in zip(chain, chain[1:]):
                cv2.line(image, tuple(points[i]), tuple(points[j]), color, 2)
        for point in points:
            cv2.circle(image, tuple(point), 3, (30, 100, 255) if trusted else color, -1)
        cv2.line(image, tuple(points[4]), tuple(points[8]), (0, 220, 255) if trusted else color, 3)
        if not trusted:
            cv2.putText(image, 'ignored', tuple(points[0]), cv2.FONT_HERSHEY_SIMPLEX, .4, color, 1)
    percent = '--' if fraction is None else f'{100 * fraction:.0f}%'
    actual = '--' if feedback is None else f'{feedback.width * 1000:.1f}'
    target_text = '--' if target is None else f'{target * 1000:.1f}'
    command_text = '--' if command is None else f'{command * 1000:.1f}'
    mode = 'EXECUTE' if execute else 'PREVIEW: NO CAN'
    state = 'FAULT' if fault else ('FOLLOWING' if active else ('LEARNING' if mapping.phase != 'ready' else 'PAUSED'))
    rows = [
        f'{mode} | {state} | hand={hand.side or "unlocked"}',
        f'Pinch: {percent}   target: {target_text} mm',
        f'Ramped target: {command_text} mm   measured: {actual} mm',
        mapping.instruction,
        reason[:95],
    ]
    cv2.rectangle(image, (0, 0), (image.shape[1], 138), (25, 25, 25), -1)
    for i, text in enumerate(rows):
        cv2.putText(image, text, (10, 22 + i * 25), cv2.FONT_HERSHEY_SIMPLEX,
                    .48, (100, 240, 140) if active else (230, 230, 230), 1, cv2.LINE_AA)
    return image


def main(argv=None):
    a = parse_args(argv)
    camera = create_camera(backend=a.camera_backend, serial=a.camera_serial, width=a.camera_width,
                            height=a.camera_height, fps=a.camera_fps,
                            warmup=a.camera_warmup, timeout=a.camera_timeout,
                            enable_depth=False)
    explicit_mapping = (None if a.pinch_closed_ratio is None else
                        PinchMapping(a.pinch_closed_ratio, a.pinch_open_ratio))
    mapping = AutoPinchMapping(explicit_mapping)
    hand = AutoPinchHand()
    ramp = WidthRamp(a.min_width_mm / 1000., a.max_width_mm / 1000.,
                     a.open_seconds, a.close_seconds, a.max_lead_mm / 1000.)
    joint_history = deque(maxlen=20)
    response = GripperResponseWatch()
    link = None
    active, filtered, last_sent = False, None, None
    last_sent_time, send_count, fault = None, 0, None
    prepare_attempted, can_control_seen = False, False
    last_report = None
    reason = mapping.instruction
    output = a.log_dir.expanduser() / datetime.now().strftime('%Y-%m-%d_%H-%M-%S_%f')
    output.mkdir(parents=True, exist_ok=False)
    print(f'[pinch] log={output / "samples.jsonl"}')
    print('空夹爪比例测试；只开彩色流，不发送机械臂使能/失能命令。')
    if a.execute and a.prepare_can_hold:
        print('已指定 --prepare-can-hold：必要时切换以太网→CAN，清旧轨迹，用当前关节角建立J保持。')
        print('接管先设速度0，确认当前目标到位后设10%；不移动到预设姿态，不自动切回旧控制来源。')
    else:
        print('不切换机械臂控制模式或发送关节目标；执行需事先处于CAN位置保持。')
        if a.execute:
            print('若提示 ctrl_mode=3，以太网控制尚未交接；需加 --prepare-can-hold 重新运行。')
    print('无需按键：先把拇指食指捏合停约1秒，再充分张开停约1秒；自动记住范围后连续跟随。')
    print('学习范围期间不发送夹爪目标；手掌尽量正对相机。Q/Esc 退出。')
    print('丢手/多手/反馈异常时暂停，同一只手恢复识别后自动继续；上一小步可能完成。')
    if explicit_mapping is not None:
        print(f'使用指定范围：{explicit_mapping.closed:g} 对应闭合，{explicit_mapping.opened:g} 对应全开。')
    print(f'开口范围 {a.min_width_mm:g}–{a.max_width_mm:g} mm；全行程张开 {a.open_seconds:g}s、闭合 {a.close_seconds:g}s。')
    print(f'力参数 {a.force:g}；目标最多领先实测 {a.max_lead_mm:g}mm；持续1秒无明显响应将锁定故障，需退出重启。')
    print('--execute 已开启。机械臂需事先处于静止的位置保持。' if a.execute else '当前仅预览，不连接 CAN。')
    window = 'Pinch -> EMPTY gripper'
    exit_code = 0
    try:
        if a.execute:
            link = GripperOnlyLink.connect(channel=a.channel, firmware=a.firmware,
                                           force=a.force, max_lead=a.max_lead_mm / 1000.)
        with camera, MediaPipeHandLandmarker(model_path=a.model, num_hands=2,
                min_detection_confidence=.35, min_presence_confidence=.35,
                min_tracking_confidence=.35) as detector, (output / 'samples.jsonl').open('w') as log:
            log.write(json.dumps({'type': 'parameters', 'execute': a.execute,
                'width_range_mm': [a.min_width_mm, a.max_width_mm], 'force_setting': a.force,
                'open_seconds': a.open_seconds, 'close_seconds': a.close_seconds,
                'max_lead_mm': a.max_lead_mm, 'depth_used': False,
                'pinch_closed_ratio': a.pinch_closed_ratio, 'pinch_open_ratio': a.pinch_open_ratio,
                'mapping_mode': 'automatic_holds' if explicit_mapping is None else 'explicit',
                'response_timeout_seconds': response.timeout, 'refresh_seconds': .1,
                'lost_policy': 'pause_and_auto_resume_same_hand',
                'prepare_can_hold': a.prepare_can_hold,
                'arm_commands': bool(a.execute and a.prepare_can_hold)}) + '\n')
            while True:
                frame = camera.read()
                observations = detector.detect(frame.bgr, timestamp_ms=round(frame.hardware_ms))
                accepted, rejected_candidates = filter_hand_observations(observations, frame.bgr.shape, fingertips=True)
                now = time.monotonic()
                pinch, fraction, target, feedback = None, None, None, None
                sent_this_frame = None
                hand_ready = False
                invalid = None
                try:
                    if now - frame.received_monotonic > .25:
                        raise ValueError('RGB frame expired during detection')
                    pinch = measure_pinch(observations, frame.bgr.shape, hand.side)
                    hand_ready = hand.observe(pinch)
                    if hand_ready:
                        old_phase = mapping.phase
                        fraction = mapping.observe(pinch, now)
                        if mapping.phase != old_phase:
                            print(f'[pinch][LEARN] {mapping.instruction}', flush=True)
                            log.write(json.dumps({'type': 'mapping', 'phase': mapping.phase,
                                'closed_ratio': mapping.closed if mapping.mapping is None else mapping.mapping.closed,
                                'open_ratio': None if mapping.mapping is None else mapping.mapping.opened,
                                'monotonic': now}) + '\n')
                        if fraction is not None:
                            target = ramp.minimum + fraction * (ramp.maximum - ramp.minimum)
                except ValueError as exc:
                    invalid = str(exc)
                    hand.lost()
                    mapping.pause()
                device_error = None
                if link is not None:
                    try:
                        feedback = link.read()
                        joint_history.append((now, feedback.joints))
                        if not joints_stationary(joint_history, now):
                            device_error = 'waiting for measured joints to settle'
                        mode = link.device_status.get('ctrl_mode')
                        if mode == 1:
                            can_control_seen = True
                        elif can_control_seen:
                            raise RuntimeError(f'CAN control lost: ctrl_mode={mode}; no automatic takeover retry')
                        elif a.prepare_can_hold and not prepare_attempted and device_error is None:
                            prepare_attempted = True
                            print('[pinch][CONTROL] 交接到CAN并建立当前位置保持；此时不发送夹爪目标。', flush=True)

                            def record(event, **values):
                                log.write(json.dumps({'type': event, 'monotonic': time.monotonic(),
                                    **values}, allow_nan=False) + '\n')
                                log.flush()

                            link.prepare_can_hold(record=record)
                            can_control_seen = True
                            joint_history.clear()
                            hand.lost()
                            mapping.pause()
                            device_error = 'CAN hold confirmed; reacquiring fresh camera and hand'
                            print('[pinch][CONTROL] 已确认CAN当前位置保持，开始等待手势。', flush=True)
                        else:
                            # A pending route handoff is not a bad position
                            # sample: retain the stationary window needed to
                            # enter preparation on the following frames.
                            try:
                                link.require_can_control()
                            except ValueError as exc:
                                device_error = str(exc)
                    except (ValueError, AttributeError, TypeError, RuntimeError) as exc:
                        device_error = str(exc)
                        joint_history.clear()
                        if ((prepare_attempted and not can_control_seen)
                                or (can_control_seen and link.device_status.get('ctrl_mode') not in (None, 1))):
                            # Preparation failure/source loss must never trigger
                            # another automatic control takeover.
                            fault, exit_code = device_error, 1
                if invalid or device_error or not hand_ready or fraction is None or fault:
                    active, filtered = False, None
                    response.pause()
                    reason = (fault or invalid or device_error or (mapping.instruction if hand_ready
                              else f'Acquiring same hand: {hand.good_frames}/3'))
                    # Retain the original arm anchor across hand loss. Automatic
                    # resumption must not hide that somebody moved the arm.
                    if device_error:
                        hand.lost()
                # Poll exit before issuing this frame's command. The first image
                # creates the window; subsequent frames show the resulting state.
                if last_report is None:
                    cv2.imshow(window, draw_overlay(frame, observations, hand, fraction, target,
                        ramp.width, feedback, active, reason, a.execute, mapping, fault))
                key = cv2.waitKey(1) & 0xff
                if key in (27, ord('q'), ord('Q')) or cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                    break
                if (not active and hand_ready and fraction is not None
                        and not invalid and not device_error and not fault):
                    try:
                        if link is not None:
                            if not joints_stationary(joint_history, now):
                                raise ValueError('waiting for measured joints to settle')
                            feedback = link.read()
                            if link.anchor is None:
                                link.anchor = feedback.joints.copy()
                        initial = feedback.width if feedback is not None else (
                            ramp.width if ramp.width is not None else ramp.minimum)
                        ramp.reset(initial, time.monotonic())
                        active, filtered, last_sent, last_sent_time = True, fraction, None, None
                        response.pause()
                        reason = 'Following finger aperture; actual opening monitored'
                    except (ValueError, AttributeError, TypeError, RuntimeError) as exc:
                        active = False
                        reason = str(exc)
                if active:
                    try:
                        filtered = .35 * fraction + .65 * filtered
                        command = ramp.advance(filtered, time.monotonic(),
                                               None if feedback is None else feedback.width)
                        if link is not None and (last_sent is None or abs(command - last_sent) >= 1e-5
                                                or now - last_sent_time >= .1):
                            link.send(command)
                            last_sent, last_sent_time = command, now
                            sent_this_frame = command
                            send_count += 1
                        if link is not None and last_sent is not None:
                            response.observe(last_sent, feedback.width, now)
                    except GripperUnresponsive as exc:
                        fault = str(exc)
                        active, filtered, reason = False, None, fault
                        exit_code = 1
                        print(f'[pinch][FAULT] {fault}', flush=True)
                    except (ValueError, AttributeError, TypeError, RuntimeError) as exc:
                        active, filtered = False, None
                        reason = str(exc)
                        hand.lost()
                        response.pause()
                cv2.imshow(window, draw_overlay(frame, observations, hand, fraction, target,
                    ramp.width, feedback, active, reason, a.execute, mapping, fault))
                device_status = None if link is None else dict(link.device_status)
                if last_report is None or now - last_report >= 1.:
                    last_report = now
                    percent = '--' if fraction is None else f'{fraction * 100:.0f}%'
                    actual = '--' if feedback is None else f'{feedback.width * 1000:.1f}mm'
                    state_text = '' if device_status is None else f' arm={device_status}'
                    if device_error and reason != device_error:
                        state_text += f' device_error={device_error}'
                    state = 'FAULT' if fault else ('FOLLOW' if active else 'WAIT')
                    sent = '--' if last_sent is None else f'{last_sent*1000:.1f}mm'
                    print(f'[pinch][{state}] hand={hand.side or "--"} '
                          f'pinch={percent} sent={sent} actual={actual} sends={send_count} '
                          f'reason={reason}{state_text}', flush=True)
                log.write(json.dumps({'type': 'sample', 'monotonic': now,
                    'sequence': int(frame.sequence), 'frame_age_ms': (now-frame.received_monotonic)*1000.,
                    'active': active, 'reason': reason, 'hand': None if pinch is None else pinch.side,
                    'pinch_ratio': None if pinch is None else pinch.ratio, 'fraction': fraction,
                    'target_mm': None if target is None else target * 1000.,
                    'ramped_mm': None if ramp.width is None else ramp.width * 1000.,
                    'measured_mm': None if feedback is None else feedback.width * 1000.,
                    'force_feedback': None if feedback is None else feedback.force,
                    'mapping_phase': mapping.phase,
                    'mapping_closed': None if mapping.mapping is None else mapping.mapping.closed,
                    'mapping_open': None if mapping.mapping is None else mapping.mapping.opened,
                    'sent_this_frame_mm': None if sent_this_frame is None else sent_this_frame*1000.,
                    'last_sent_mm': None if last_sent is None else last_sent*1000.,
                    'send_count': send_count, 'fault': fault,
                    'observed_control': None if link is None else link.observed_control,
                    'selected_hand': hand.side, 'device_status': device_status,
                    'candidate_count': len(observations), 'accepted_count': len(accepted),
                    'rejected_candidates': rejected_candidates,
                    'device_error': device_error}, allow_nan=False) + '\n')
    except KeyboardInterrupt:
        print('\n[pinch] Ctrl+C: stop sending new gripper targets.')
    except Exception as exc:
        print(f'\n[pinch][error] {type(exc).__name__}: {exc}', file=sys.stderr)
        exit_code = 1
    finally:
        try:
            camera.close()
        finally:
            try:
                cv2.destroyAllWindows()
            finally:
                if link is not None:
                    link.close()
    print('[pinch] 已退出；未额外开合夹爪或改变机械臂使能。最后一个小目标仍可能完成。')
    if prepare_attempted:
        print('[pinch] 本次尝试过CAN当前位置保持交接；退出不恢复旧控制来源或旧轨迹。')
    return exit_code


if __name__ == '__main__':
    raise SystemExit(main())
