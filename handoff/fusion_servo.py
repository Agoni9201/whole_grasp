"""Opt-in adapter: sole existing MIT/effector owner executes the pure policy."""
from __future__ import annotations

import time
from dataclasses import asdict
from datetime import datetime
import os
from pathlib import Path

import numpy as np

from .fusion_cli import fusion_config
from .fusion_handover import FusionHandover, LinearPullModel, model_metadata
from .gesture_handover import read_opening, wait_initial_opening
from .pinch_gripper import GripperResponseWatch, WidthRamp
from control_backends.force_ctrl.feedback_health import FeedbackHealthMonitor

MIT_STARTUP_TIMEOUT_SECONDS = 2.


def start_fusion_session(args):
    """Start/verify observation before acquiring a new control backend."""
    from .fusion_observer import FusionObserver
    from .fusion_recording import FusionRecorder
    run = datetime.now().strftime('%Y-%m-%d_%H-%M-%S_%f')+f'_p{os.getpid()}'
    directory = Path(getattr(args, 'save_dir', 'logs') or 'logs').resolve()/'handover_fusion'/run
    recorder = FusionRecorder(directory, dict(
        version=1, control_mode='mit_cartesian_hold', mode=args.fusion_handover,
        dry_run=not bool(args.execute), config=asdict(fusion_config(args)),
        pull_model=model_metadata(args.fusion_pull_model),
        wrist_serial=getattr(args, 'camera_serial', None), global_serial=args.fusion_global_serial,
        wrist_profile=[getattr(args, 'camera_width', 640), getattr(args, 'camera_height', 480),
                       getattr(args, 'camera_fps', 30)],
        global_color_profile=[640, 480, 30], global_depth_enabled=False,
        receiver_side=args.fusion_receiver_side, support_verified=None,
        collection=dict(group=getattr(args, 'fusion_collection_group', None),
                        split=getattr(args, 'fusion_collection_split', None),
                        labels_verified=False,
                        label_semantics='keyboard marks; verify action intervals offline'),
        feedback_kind='current_derived_joint_torque_not_external_force',
        parking_policy='early 3D preparation segment; shadow-only manual S fallback; no blind approach',
        instructions='S: shadow-only park; R: ROI after hold; B: confirm untouched; P/T/U/H/N: action labels; C: reset; Q: quit'))
    observer = FusionObserver(dict(serial=args.fusion_global_serial, hand_model=args.hand_model,
        receiver_side=args.fusion_receiver_side, roi=args.fusion_roi,
        allow_manual_stop=args.fusion_handover == 'shadow', save_dir=str(directory/'global')))
    try:
        if getattr(args, 'fusion_collection_group', None):
            from .fusion_capture_snapshot import save_capture_snapshot
            save_capture_snapshot(args, directory)
        observer.start()
        deadline = time.monotonic()+15.
        while time.monotonic() < deadline:
            visual, events = observer.poll()
            if any(e['action'] == 'quit' for e in events):
                raise RuntimeError('global observer closed before delivery')
            if visual and 0 <= time.monotonic()-visual['received_monotonic'] <= .25:
                args._fusion_observer, args._fusion_recorder = observer, recorder
                return observer, recorder
            time.sleep(.02)
        raise TimeoutError('no fresh global hand observer frame before delivery')
    except BaseException:
        try:
            observer.close()
        finally:
            recorder.close()
        raise


class FusionServoMixin:
    def _init_fusion(self, *, simulated):
        self._fusion = None
        if not getattr(self.args, 'fusion_handover', None):
            return
        self._fusion_simulated = simulated
        self._fusion_observer = self.args._fusion_observer
        self._fusion_recorder = self.args._fusion_recorder
        model = LinearPullModel.load(self.args.fusion_pull_model) if self.args.fusion_pull_model else None
        self._fusion = FusionHandover(fusion_config(self.args), mode=self.args.fusion_handover, model=model)
        self._fusion_health = FeedbackHealthMonitor()
        if simulated:
            width = self.args.fusion_preview_grip_width_mm/1000.
        else:
            width, _, _ = wait_initial_opening(self.gripper)
        if not .001 < width < self._fusion.config.maximum_width:
            raise ValueError('fusion requires a measured held-object opening below its maximum')
        self._fusion_sim_width = self._fusion_grip_width = width
        previous = getattr(self.gripper, 'last_command', {}) or {}
        self._fusion_force = min(float(getattr(self.args, 'gripper_force', .5)), .5,
            float(getattr(self.args, 'gripper_hardware_max_force', .5)),
            float(previous.get('force', .5)))
        if not np.isfinite(self._fusion_force) or self._fusion_force <= 0:
            raise ValueError('invalid fusion gripper force limit')
        self._fusion_ramp = None
        self._fusion_watch = GripperResponseWatch()
        self._fusion_last_command = self._fusion_last_send = None
        self._fusion_last_gripper_stamp = None
        self._fusion_open_count = 0
        self._fusion_open_started = None
        self._fusion_hold_control_mode = None
        self._fusion_mit_ready = simulated
        self._fusion_startup_at = None
        self._fusion_startup_waiting = False
        self._fusion_visual = self._fusion_sample = None
        self._fusion_events = []
        self._fusion_next_ui = self._fusion_last_wrist_at = 0.
        self._fusion_wrist_sequence = None
        self.get_logger().warning('[fusion] %s; shadow S parks; R selects ROI after hold; B confirms untouched baseline; data=%s',
                                  self._fusion.mode, self._fusion_recorder.directory)

    def _fusion_arm_status(self):
        if self._fusion_simulated:
            return dict(ctrl_mode=1, mode_feedback=6, arm_status=0, err_code=0)
        state = self.backend.get_robot().get_arm_status()
        if (state is None or not 0 <= time.time()-state.timestamp <= .5):
            raise ValueError('fusion requires fresh arm status')
        msg = state.msg
        result = {key: int(getattr(msg, key)) for key in ('ctrl_mode', 'mode_feedback', 'arm_status', 'err_code')}
        if (result['ctrl_mode'] != 1 or result['arm_status'] != 0 or result['err_code'] != 0):
            raise ValueError(f'fusion requires healthy unchanged MIT/CAN mode: {result}')
        locked = getattr(self, '_fusion_hold_control_mode', None)
        if locked is not None and (result['ctrl_mode'], result['mode_feedback']) != locked:
            raise ValueError('arm control mode changed after baseline confirmation')
        if result['mode_feedback'] not in (4, 6):
            # The existing backend starts its MIT sender asynchronously while
            # the arm can still report the preceding joint-position hold. Do
            # not follow, park or confirm a baseline during this bounded wait.
            starting = (result['mode_feedback'] == 1 and locked is None
                        and not getattr(self, '_fusion_mit_ready', False)
                        and not getattr(self, '_imp_engaged', False)
                        and getattr(self, 'state', None) == 'HOLD_LIFTED'
                        and self._fusion.phase == 'FOLLOW_RECEIVER')
            if not starting:
                raise ValueError(f'fusion requires healthy unchanged MIT/CAN mode: {result}')
            mono = time.monotonic()
            if self._fusion_startup_at is None:
                self._fusion_startup_at = mono
            if mono-self._fusion_startup_at >= MIT_STARTUP_TIMEOUT_SECONDS:
                raise ValueError('MIT startup timed out: arm still reports joint-position mode')
            self._fusion_startup_waiting = True
            return result
        self._fusion_mit_ready = True
        self._fusion_startup_waiting = False
        return result

    def _poll_fusion(self, q_now, wall_now):
        if getattr(self, '_fusion', None) is None:
            return
        try:
            visual, events = self._fusion_observer.poll()
            self._fusion_visual, self._fusion_events = visual, events
            if any(e['action'] == 'quit' for e in events):
                self._shutdown(0)
                return
            mono = time.monotonic()
            if self._fusion_simulated:
                self._fusion_sample = dict(q=q_now.copy(), dq_raw=np.zeros(7), dq=np.zeros(7), tau=np.zeros(7),
                    motor_timestamps=np.full(7, wall_now), position_timestamps=np.full(7, wall_now),
                    sampled_monotonic_time=mono, sampled_wall_time=wall_now, simulated=True)
            else:
                self._fusion_sample = self._fusion_health.read(self.backend.get_robot(), time.time(), mono)
                # Optional current, when exposed by this firmware. Preserve its
                # own stamp; never invoke get_motor_states (it mutates velocity).
                parser = self.backend.get_robot()._parser
                currents, current_stamps = [], []
                for index in range(1, 8):
                    wrapped = getattr(parser, f'motor_state_{index}')
                    stamp = float(wrapped.timestamp)
                    value = getattr(wrapped.msg, 'current', None)
                    coherent = stamp == float(wrapped.timestamp)
                    currents.append(float(value) if coherent and value is not None else None)
                    current_stamps.append(stamp if coherent and value is not None else None)
                self._fusion_sample.update(motor_current=currents, current_timestamps=current_stamps)
            was_waiting = getattr(self, '_fusion_startup_waiting', False)
            self._fusion_device_status = self._fusion_arm_status()
            if self._fusion_recorder.error:
                raise RuntimeError(self._fusion_recorder.error)
            if getattr(self, '_fusion_startup_waiting', False):
                self._fusion.reason = 'waiting for MIT feedback; current joint hold retained; ROI unavailable'
                return False
            if was_waiting and self._fusion.phase == 'FOLLOW_RECEIVER':
                self._fusion.reason = 'MIT feedback ready; waiting for fresh 3D delivery target'
            return True
        except (RuntimeError, ValueError, TypeError, AttributeError, OSError) as exc:
            self._fusion_sample = None
            self._fusion_startup_waiting = False
            self._fusion.fault(exc)
            self._pause_target_motion(wall_now, immediate=True)
            return False

    def _fusion_report(self):
        fusion = self._fusion
        self._preview_status, self._preview_reason = fusion.phase, fusion.reason
        if fusion.phase != 'FOLLOW_RECEIVER':
            self.state = fusion.phase
        if getattr(self, '_fusion_reported', None) != (fusion.phase, fusion.reason):
            self._fusion_reported = fusion.phase, fusion.reason
            self.get_logger().warning('[fusion] %s: %s', fusion.phase, fusion.reason)
        now = time.monotonic()
        if now >= self._fusion_next_ui:
            self._fusion_next_ui = now+.5
            try:
                self._fusion_observer.control('status', phase=fusion.phase, reason=fusion.reason[:120])
            except (OSError, EOFError, RuntimeError) as exc:
                fusion.fault(f'global observer unavailable: {exc}')
        self._diag_note(fusion_handover=fusion.snapshot())

    def _fusion_manual_stop(self, q_now, now):
        """Explicit data-collection parking, never a release/baseline shortcut."""
        fusion = getattr(self, '_fusion', None)
        if (fusion is None or getattr(fusion, 'mode', None) != 'shadow'
                or fusion.phase != 'FOLLOW_RECEIVER' or not getattr(self, '_imp_engaged', False)
                or self.state == 'HOLD_LIFTED' or getattr(self, '_fusion_sample', None) is None):
            return False
        mono = time.monotonic()
        requested = any(e['action'] == 'stop' and 0 <= mono-e['at'] <= fusion.config.visual_max_age
                        for e in getattr(self, '_fusion_events', []))
        if not requested:
            return False
        self._fusion_begin_hold(q_now, now, source='manual_shadow')
        return True

    def _fusion_begin_hold(self, q_now, now, *, source='visual_pre_occlusion'):
        self._pause_target_motion(now, immediate=True)
        self._engage(q_now, now)  # Freeze ACTUAL pose once, with the existing sender.
        self._fusion.begin_hold(q_now, source=source)
        self._fusion_observer.control('hold')
        self._fusion_report()

    def _fusion_hold_step(self, q_now, wall_now):
        fusion = self._fusion
        now = time.monotonic()
        visual = self._fusion_visual or {}
        events = self._fusion_events
        confirm = any(e['action'] == 'baseline' and 0 <= now-e['at'] <= fusion.config.visual_max_age
                      and e.get('roi_revision') == visual.get('roi_revision') for e in events)
        cancel = any(e['action'] == 'cancel' for e in events)
        try:
            if self._fusion_recorder.error or self._fusion_recorder.queue.full():
                raise RuntimeError('fusion recorder unavailable; no opening')
            if self._fusion_simulated:
                measured, feedback_force, stamp = self._fusion_sim_width, 0., time.time()
            else:
                measured, feedback_force, stamp = read_opening(self.gripper.status())
            self._fusion_gripper_feedback = dict(width_m=measured, force=feedback_force, wall_timestamp=stamp)
            # Before the opening boundary, physical width must remain the held
            # opening. No other mode may silently open/close the gripper.
            if fusion.phase != 'FUSION_OPENING' and not fusion.terminal and abs(measured-self._fusion_grip_width) > .003:
                raise ValueError('held gripper opening changed; baseline invalid')
            fusion.tick(self._fusion_sample, visual, now, confirm_untouched=confirm, cancel=cancel)
            if fusion.phase == 'FUSION_WAIT_BASELINE':
                self._fusion_hold_control_mode = None
            elif fusion.phase == 'FUSION_BASELINE' and self._fusion_hold_control_mode is None:
                status = self._fusion_device_status
                self._fusion_hold_control_mode = status['ctrl_mode'], status['mode_feedback']
            if fusion.phase == 'FUSION_OPENING':
                self._fusion_open_step(measured, stamp, now)
        except (ValueError, RuntimeError, TypeError, AttributeError) as exc:
            fusion.fault(exc)
        if fusion.phase == 'FUSION_FAULT':
            self._pause_target_motion(wall_now, immediate=True)
        self._fusion_report()

    def _fusion_open_step(self, measured, stamp, now):
        c = self._fusion.config
        if self._fusion_ramp is None:
            self._fusion_ramp = WidthRamp(self._fusion_grip_width, c.maximum_width,
                                          open_seconds=c.open_seconds)
            self._fusion_ramp.reset(max(self._fusion_grip_width, measured), now)
            self._fusion_open_started = now
        if now-self._fusion_open_started > c.open_seconds+5.:
            raise ValueError('fusion opening feedback timeout')
        if self._fusion_last_command is not None:
            self._fusion_watch.observe(self._fusion_last_command, measured, now)
        fresh = self._fusion_last_gripper_stamp is None or stamp > self._fusion_last_gripper_stamp
        if self._fusion_last_gripper_stamp is not None and stamp < self._fusion_last_gripper_stamp:
            raise ValueError('gripper feedback timestamp went backwards')
        if fresh:
            self._fusion_last_gripper_stamp = stamp
            self._fusion_open_count = self._fusion_open_count+1 if measured >= c.maximum_width-.001 else 0
        if self._fusion_open_count >= 3:
            self._fusion.phase, self._fusion.reason = 'HANDOVER_DONE', 'full opening feedback confirmed; arm remains fixed'
            return
        if not fresh or (self._fusion_last_send is not None and now-self._fusion_last_send < .05-1e-9):
            return
        command = self._fusion_ramp.advance(1., now, measured_width=measured)
        if self._fusion_last_command is not None and command <= self._fusion_last_command+1e-8:
            return
        if command <= self._fusion_grip_width+1e-8:
            return
        # Recheck status, visual age and latest physical opening immediately
        # before every actual write, rather than trusting a queued permission.
        self._fusion_arm_status()
        v = self._fusion_visual or {}
        if not 0 <= time.monotonic()-v.get('received_monotonic', -1e20) <= c.visual_max_age:
            raise ValueError('visual evidence expired before gripper write')
        if self._fusion_simulated:
            self._fusion_sim_width = command
        else:
            latest, _, _ = read_opening(self.gripper.status())
            if (abs(command-latest) > self._fusion_ramp.max_lead+1e-9
                    or not self._fusion_grip_width <= command <= c.maximum_width):
                raise ValueError('fusion gripper target exceeds measured lead/width bound')
            self.gripper.open(width=command, force=self._fusion_force)
        self._fusion_last_command, self._fusion_last_send = command, now

    def _record_fusion(self):
        if getattr(self, '_fusion', None) is None:
            return
        try:
            image = self._latest_image
            sequence = getattr(self, '_sdk_frame_sequence', None)
            mono = time.monotonic()
            wrist = None
            if (image is not None and sequence != self._fusion_wrist_sequence
                    and mono-self._fusion_last_wrist_at >= .1):
                wrist = image, sequence
                self._fusion_wrist_sequence, self._fusion_last_wrist_at = sequence, mono
            self._fusion_recorder.write(dict(kind='sample', wall_time=time.time(), monotonic=mono,
                feedback=self._fusion_sample, device_status=getattr(self, '_fusion_device_status', None),
                control_startup=dict(mit_ready=getattr(self, '_fusion_mit_ready', False),
                                     waiting=getattr(self, '_fusion_startup_waiting', False)),
                policy=self._fusion.snapshot(), visual=self._fusion_visual, events=self._fusion_events,
                gripper_command=self._fusion_last_command,
                gripper_command_monotonic=self._fusion_last_send,
                gripper_feedback=getattr(self, '_fusion_gripper_feedback', None),
                wrist=dict(sequence=sequence, received_monotonic=getattr(self, '_sdk_received_monotonic', None),
                           q_at_image=self._latest_q,
                           detection_sequence=self._consumed_sequence,
                           detection_matches_frame=self._consumed_sequence == sequence,
                           aligned_depth_available=self._latest_depth is not None,
                           detection=getattr(self.target, 'last_diagnostics', None)),
                support_verified=None), wrist=wrist)
        except (RuntimeError, ValueError, AttributeError, TypeError) as exc:
            self._fusion.fault(exc)
            self._pause_target_motion(time.time(), immediate=True)
