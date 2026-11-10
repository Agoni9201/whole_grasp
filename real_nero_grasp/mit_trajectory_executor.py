"""Persistent unified MIT session for the whole grasp -> servo pipeline.

One ``NeroMITLoop`` runs on the already-connected grasp driver (``arm.arm``)
for the entire task.  Joint routes (NBV, grasp approach, lift, return) are
tracked through the loop's buffered command path; the visual servo reuses the
same loop for in-loop Cartesian impedance.  There is no move_j<->MIT mode
switching during task execution and **no second CAN driver**. Normal process
exit alone transfers the confirmed stationary hold to the device's joint mode.

Disabled by default: the workflow only routes here when
``args.trajectory_executor == "mit"``; ``move_j`` remains the default executor.
"""

from __future__ import annotations

import json
import os
import time
from typing import Callable, Optional

import numpy as np
from scipy.spatial.transform import Rotation

from control_backends.force_ctrl.data_types import JointCommands
from control_backends.force_ctrl.gravity_comp import NeroGravityCompensator
from control_backends.force_ctrl.impedance_profile import (
    VALIDATED_GRAVITY_FACTORS,
    VALIDATED_TOTAL_TORQUE_LIMIT,
)
from control_backends.force_ctrl.mit_loop import NERO_JOINT_LIMITS, NeroMITLoop
from control_backends.force_ctrl.normal_shutdown import (
    MITControlledStop, defer_sigint, handoff_to_position_hold, report_shutdown,
)

from .trajectory_time_param import N_JOINTS, build_joint_trajectory
from .types import ControlledMotionStopTriggered, MotionNotReached, SafetyStopTriggered

__all__ = [
    "MITTrajectorySession",
    "MITTrajectoryExecutor",  # backward-compatible alias
    "get_mit_session",
    "reset_mit_session",
]


def _deg2rad(deg):
    return float(deg) * np.pi / 180.0


def _flange_pose_error(actual_pose, target_pose):
    """Position (m) and orientation (rad) error between two [x,y,z,r,p,y] poses."""
    actual = np.asarray(actual_pose, dtype=np.float64).reshape(6)
    target = np.asarray(target_pose, dtype=np.float64).reshape(6)
    pos_err = float(np.linalg.norm(actual[:3] - target[:3]))
    R_actual = Rotation.from_euler("xyz", actual[3:])
    R_target = Rotation.from_euler("xyz", target[3:])
    ori_err = float((R_target * R_actual.inv()).magnitude())
    return pos_err, ori_err


def _build_pin_model():
    """Pinocchio model for in-loop Cartesian impedance (servo).  Lazy + optional:
    the joint-trajectory path works without it; only the servo reuse needs it.
    """
    try:
        from models.nero_model import NeroPinocchioModel

        return NeroPinocchioModel()
    except Exception as exc:
        print(
            f"[mit-session][warn] Pinocchio model unavailable ({type(exc).__name__}: "
            f"{exc}); in-loop Cartesian impedance / servo reuse is disabled"
        )
        return None


class MITTrajectorySession:
    """One MIT loop throughout the task; device position hold only on exit.

    The route geometry comes from the validated joint waypoints (``start_joints``
    plus each ``stage.joints``).  Time parameterization never changes the
    joint-space line, so the planner's collision validation stays valid; the
    tracking-error limit and the arrival gate ensure the arm actually follows
    that line.  Between routes the loop holds position in MIT mode.
    """

    def __init__(self, arm, args):
        self._arm = arm
        self._driver = getattr(arm, "arm", None)
        if self._driver is None:
            raise RuntimeError(
                "MIT session requires the grasp arm adapter to expose .arm "
                "(the connected pyAgxArm driver)"
            )
        self._grav = NeroGravityCompensator()
        self._pin = _build_pin_model()
        self._joint_limit_buffer_rad = _deg2rad(
            float(getattr(args, "mit_joint_limit_buffer_deg", 5.0))
        )
        self._loop = NeroMITLoop(
            robot=self._driver,
            gravity_comp=self._grav,
            control_freq=200,
            gravity_factors=np.asarray(VALIDATED_GRAVITY_FACTORS, dtype=np.float64),
            torque_limit=float(
                getattr(args, "mit_torque_limit", VALIDATED_TOTAL_TORQUE_LIMIT)
            ),
            joint_limit_buffer=self._joint_limit_buffer_rad,
            pin_model=(self._pin.clone() if self._pin is not None else None),
            max_feedback_age_s=float(getattr(args, "mit_max_feedback_age_s", 0.1)),
            max_feedback_skew_s=float(getattr(args, "mit_max_feedback_skew_s", 0.05)),
            max_control_gap_s=float(getattr(args, "mit_max_control_gap_s", 0.1)),
            hold_recovery_timeout_s=float(getattr(args, "mit_hold_recovery_timeout_s", 0.5)),
            controlled_stop_kp=np.asarray(getattr(args, "mit_kp", [30, 30, 30, 15, 10, 5, 5]), dtype=float),
            controlled_stop_kd=np.asarray(getattr(args, "mit_kd", [2, 2, 2, 1, .5, .3, .3]), dtype=float),
        )
        self._started = False
        self._owns_control = False
        self._shutdown_complete = False
        self._exit_stop_confirmed = False
        self._device_hold_confirmed = False
        self._emergency_stop_requested = False
        self._args = args
        self._feedback_timeout_s = float(getattr(args, "mit_feedback_timeout_s", 0.5))
        self._diagnostics_path = None
        self._period = 1.0 / 200.0
        self._kp = np.asarray(
            getattr(args, "mit_kp", [30.0, 30.0, 30.0, 15.0, 10.0, 5.0, 5.0]),
            dtype=np.float64,
        ).reshape(N_JOINTS)
        self._kd = np.asarray(
            getattr(args, "mit_kd", [2.0, 2.0, 2.0, 1.0, 0.5, 0.3, 0.3]),
            dtype=np.float64,
        ).reshape(N_JOINTS)

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    @property
    def running(self):
        return self._loop.running

    @property
    def owns_control(self):
        # Sticky ownership: a dead/stopped MIT worker must never silently fall
        # through to ordinary move_j after a fault or disconnect.
        return bool(getattr(self, "_owns_control", False))

    def set_phase(self, label):
        self._loop.set_diagnostics_phase(str(label))

    def _raise_if_terminal_fault(self, label):
        """Propagate a background-loop hard stop into the workflow immediately."""
        # Do not start another route while a fixed-hold feedback retry is active.
        deadline = time.monotonic() + 1.1
        while bool(getattr(self._loop, "_hold_recovering", False)):
            if time.monotonic() >= deadline:
                raise MotionNotReached(f"{label}: MIT fixed-hold recovery did not finish")
            time.sleep(0.005)
        if not bool(getattr(self._loop, "terminal_fault", False)):
            return
        reason = str(
            getattr(self._loop, "last_error", None)
            or "MIT control loop entered a terminal safety fault"
        )
        message = f"{label} MIT terminal safety fault: {reason}"
        setattr(self._arm, "_safety_stop_latched", True)
        setattr(self._arm, "_safety_stop_reason", message)
        setattr(self._arm, "_controlled_stop_request_confirmed", False)
        if "electronic emergency stop call failed" not in reason:
            setattr(self._arm, "_safety_stop_request_confirmed", True)
        raise SafetyStopTriggered(message)

    @staticmethod
    def _joint_limit_violations(joints, reserve_rad):
        joints = np.asarray(joints, dtype=np.float64).reshape(N_JOINTS)
        lower = NERO_JOINT_LIMITS[:, 0] + float(reserve_rad)
        upper = NERO_JOINT_LIMITS[:, 1] - float(reserve_rad)
        if np.any(lower >= upper):
            raise ValueError(
                f"MIT joint-limit reserve {np.rad2deg(reserve_rad):.2f}deg "
                "leaves an empty permitted range"
            )
        return [
            (
                index + 1,
                float(np.rad2deg(joints[index])),
                float(np.rad2deg(lower[index])),
                float(np.rad2deg(upper[index])),
            )
            for index in range(N_JOINTS)
            if not np.isfinite(joints[index])
            or joints[index] < lower[index]
            or joints[index] > upper[index]
        ]

    def _validate_joint_limit_preflight(self, stages, start_joints, args, label):
        """Reject unsafe MIT endpoints before starting the control loop.

        Joint trajectories are straight line segments, so endpoints inside the
        coordinate-wise effective limits imply every interpolated sample is
        inside them too.
        """
        runtime_reserve = self._joint_limit_buffer_rad
        tracking_reserve = _deg2rad(
            max(float(getattr(args, "mit_tracking_error_deg", 2.0)), 0.0)
        )
        slack_reserve = _deg2rad(
            max(
                float(
                    getattr(args, "mit_joint_limit_planning_slack_deg", 1.0)
                ),
                0.0,
            )
        )
        planned_reserve = runtime_reserve + tracking_reserve + slack_reserve

        checks = [
            (
                "measured_current",
                np.asarray(
                    self._arm.get_joint_angles(), dtype=np.float64
                ).reshape(N_JOINTS),
                runtime_reserve,
            ),
            (
                "route_start",
                np.asarray(start_joints, dtype=np.float64).reshape(N_JOINTS),
                runtime_reserve,
            )
        ]
        checks.extend(
            (
                f"stage[{index}]={stage.label}",
                np.asarray(stage.joints, dtype=np.float64).reshape(N_JOINTS),
                planned_reserve,
            )
            for index, stage in enumerate(stages, start=1)
        )
        for point_name, joints, reserve in checks:
            violations = self._joint_limit_violations(joints, reserve)
            if not violations:
                continue
            detail = "; ".join(
                f"J{joint}={value:.2f}deg effective=[{lower:.2f}, {upper:.2f}]deg"
                for joint, value, lower, upper in violations
            )
            raise MotionNotReached(
                f"{label} MIT joint-limit preflight rejected {point_name}: "
                f"{detail}; no motion command was sent"
            )
        print(
            f"[mit-exec][{label}][joint-limit-preflight] "
            f"runtime_buffer={np.rad2deg(runtime_reserve):.2f}deg "
            f"tracking={np.rad2deg(tracking_reserve):.2f}deg "
            f"slack={np.rad2deg(slack_reserve):.2f}deg "
            f"planned_margin={np.rad2deg(planned_reserve):.2f}deg"
        )

    def _stop_move_j(self):
        """Clear the move_j queue once and confirm stationary WITHOUT latching.

        ``RealNeroGraspArm.terminate_motion()`` latches a terminal safety stop,
        so the raw terminate frame plus the settle helper are used directly.
        """
        terminate = getattr(self._arm, "_send_ordinary_trajectory_terminate", None)
        settle = getattr(self._arm, "_wait_for_stationary_joints", None)
        if not callable(terminate) or not callable(settle):
            raise RuntimeError(
                "grasp arm adapter does not expose the ordinary-terminate / "
                "stationary-settle helpers required for the MIT takeover"
            )
        print("[mit-session] clearing move_j trajectory and confirming stationary")
        terminate(frames=3)
        settle(timeout_s=float(getattr(self._arm, "_stop_confirm_timeout", 1.5)))

    def ensure_started(self):
        if bool(getattr(self, "_shutdown_complete", False)):
            raise SafetyStopTriggered("MIT session was shut down; refusing restart")
        assert_motion = getattr(self._arm, "_assert_motion_not_latched", None)
        if callable(assert_motion) and not self.owns_control:
            assert_motion("MIT takeover")
        if bool(getattr(self._arm, "_safety_stop_latched", False)):
            raise SafetyStopTriggered("MIT stop is latched; refusing restart")
        self._raise_if_terminal_fault("MIT startup")
        if self._started:
            if not self._loop.running:
                raise SafetyStopTriggered("MIT worker stopped; refusing restart")
            return
        if self.owns_control:
            raise SafetyStopTriggered("MIT session was shut down; refusing restart")
        self._stop_move_j()
        q = np.asarray(self._arm.get_joint_angles(), dtype=np.float64).reshape(N_JOINTS)
        if not np.all(np.isfinite(q)):
            raise RuntimeError("non-finite MIT takeover joint feedback")
        if bool(getattr(self._args, "mit_q_log", False)):
            out_dir = str(
                getattr(self._args, "mit_q_log_dir", "")
                or getattr(self._args, "save_dir", "")
                or "logs"
            )
            os.makedirs(out_dir, exist_ok=True)
            self._diagnostics_path = os.path.join(out_dir, "mit_control_cycles.jsonl")
            self._loop.start_diagnostics(self._diagnostics_path)
            print(f"[mit-log] continuous control-cycle log={self._diagnostics_path}")
            self.set_phase("startup_hold")
        # Seed before the first sender cycle, not after starting its thread.
        self._loop.update_commands(
            JointCommands(
                pos=q,
                vel=np.zeros(N_JOINTS),
                kp=self._kp.copy(),
                kd=self._kd.copy(),
                torques=np.zeros(N_JOINTS),
            )
        )
        self._owns_control = True
        self._loop.start()
        self._started = True
        print("[mit-session] MIT loop started on the grasp driver (no second CAN)")

    def hold(self):
        """Hold current measured position (used between routes and on error)."""
        if bool(getattr(self._arm, "_safety_stop_latched", False)):
            return
        self._raise_if_terminal_fault("MIT hold")
        self._raise_if_route_stopped("MIT hold")
        q = np.asarray(self._arm.get_joint_angles(), dtype=np.float64).reshape(N_JOINTS)
        self._loop.update_commands(
            JointCommands(
                pos=q,
                vel=np.zeros(N_JOINTS),
                kp=self._kp.copy(),
                kd=self._kd.copy(),
                torques=np.zeros(N_JOINTS),
            )
        )
        self.set_phase("hold")

    def _raise_if_route_stopped(self, label):
        reason = getattr(self._loop, "_controlled_stop_reason", None)
        if reason:
            self.terminate_motion()
            raise ControlledMotionStopTriggered(f"{label}: {reason}; position hold confirmed")

    def terminate_motion(self, timeout_s=2.0):
        """Latch a fixed MIT hold; require fresh feedback to confirm settling."""
        self._arm._safety_stop_latched = True
        self._raise_if_terminal_fault("MIT stop")
        if bool(getattr(self, "_device_hold_confirmed", False)):
            return {"confirmed": True, "method": "device_enabled_position_hold"}
        self.set_phase("controlled_stop")
        stop = getattr(self, "_exit_stop", None)
        if stop is None:
            stop = MITControlledStop(
                self._driver, self._loop, self._kp, self._kd,
                check_fault=lambda: self._raise_if_terminal_fault("MIT stop"),
                timeout_s=timeout_s)
            self._exit_stop = stop
        stop.request()
        result = stop.wait(timeout_s=timeout_s)
        self._arm._controlled_stop_request_confirmed = True
        self._exit_stop_confirmed = True
        self.set_phase("controlled_stop_confirmed")
        return result

    def _exit_faulted(self):
        return bool(
            getattr(self, "_emergency_stop_requested", False)
            or getattr(self._loop, "terminal_fault", False)
            or getattr(self._arm, "_safety_stop_request_confirmed", False)
        )

    def prepare_shutdown(self):
        """Freeze the only controller before tearing down visual resources."""
        if bool(getattr(self, "_shutdown_complete", False)):
            return
        if self._exit_faulted():
            return  # Never restore normal mode after an emergency stop.
        if not self.owns_control or bool(getattr(self, "_exit_stop_confirmed", False)):
            return
        self.terminate_motion(timeout_s=float(getattr(self._arm, "_stop_confirm_timeout", 2.0)))

    @defer_sigint()
    def shutdown(self):
        """Confirm stop, join MIT, confirm enabled device hold; driver stays owned by arm."""
        if bool(getattr(self, "_shutdown_complete", False)):
            return
        if self.owns_control and not self._exit_faulted():
            self.prepare_shutdown()
            def check_fault():
                self._raise_if_terminal_fault("MIT exit")
                if self._exit_faulted():
                    raise SafetyStopTriggered("Electronic stop forbids normal-mode exit handoff")
            stop_sender = getattr(self._loop, "stop_sender", self._loop.stop)
            record = getattr(self._loop, "record_control_event", None)
            if callable(record):
                record("device_handoff_begin")
            result = handoff_to_position_hold(
                self._driver, stop_sender, check_fault=check_fault, record_event=record)
            record = getattr(self._loop, "record_control_event", None)
            if callable(record):
                record("device_hold_confirmed", result=result)
            self._device_hold_confirmed = True
            report_shutdown("[shutdown] device position hold confirmed; motors left enabled; no disable/e-stop sent")
        else:
            self._loop.stop()
        self._started = False
        self._shutdown_complete = True
        # Optional log finalization happens only after physical hold handoff.
        # It must not masquerade as a failed motion stop in outer cleanup.
        try:
            self._finish_shutdown_diagnostics()
        except Exception as exc:
            report_shutdown(f"[mit-log][warn] shutdown diagnostics failed after sender stop: {type(exc).__name__}: {exc}")

    def _finish_shutdown_diagnostics(self):
        close = getattr(self._loop, "stop_diagnostics", None)
        if callable(close):
            close()
        status_fn = getattr(self._loop, "get_diagnostics_status", None)
        status = status_fn() if callable(status_fn) else {"enabled": False}
        if status.get("enabled", False):
            print(
                f"[mit-log] written={status.get('written_cycles', 0)} "
                f"dropped={status.get('dropped_cycles', 0)} "
                f"error={status.get('error')} closed={status.get('closed')}"
            )
            try:
                path = os.path.join(
                    os.path.dirname(status["path"]), "mit_control_log_status.json"
                )
                with open(path, "w", encoding="utf-8") as stream:
                    json.dump({"timestamp": time.time(), **status}, stream, indent=2)
            except Exception as exc:
                print(f"[mit-log][warn] cannot save logger status: {type(exc).__name__}: {exc}")

    # ------------------------------------------------------------------ #
    # Joint trajectory
    # ------------------------------------------------------------------ #

    def start_control_diagnostics(self, path):
        return self._loop.ensure_control_diagnostics(path)

    def execute_stages(
        self,
        stages,
        start_joints,
        args,
        label,
        runtime_guard: Optional[Callable] = None,
    ):
        """Track one validated joint route; hold at the goal when finished.

        ``runtime_guard`` is an optional per-cycle measured-joint point-cloud
        guard.  It is intentionally not required: the geometric path was already
        collision validated and is preserved exactly by time parameterization,
        while the tracking-error limit bounds how far actual can leave it.
        """
        stages = tuple(stages)
        if not stages:
            return np.asarray(start_joints, dtype=np.float64).reshape(N_JOINTS)

        self._validate_joint_limit_preflight(stages, start_joints, args, label)
        self.ensure_started()
        self.set_phase(f"trajectory:{label}")

        waypoints = [np.asarray(start_joints, dtype=np.float64).reshape(N_JOINTS)]
        waypoints += [
            np.asarray(stage.joints, dtype=np.float64).reshape(N_JOINTS)
            for stage in stages
        ]
        goal = waypoints[-1]

        v_max = _deg2rad(float(getattr(args, "mit_max_joint_vel_deg_s", 30.0)))
        a_max = _deg2rad(float(getattr(args, "mit_max_joint_accel_deg_s2", 120.0)))
        traj = build_joint_trajectory(waypoints, v_max, a_max)
        print(
            f"[mit-exec][{label}] {len(stages)} stage(s), "
            f"{len(traj.segments)} segment(s), duration={traj.duration_s:.2f}s"
        )

        tracking_err_limit = _deg2rad(float(getattr(args, "mit_tracking_error_deg", 2.0)))
        tracking_confirm = max(
            int(getattr(args, "mit_tracking_error_confirm_samples", 3)), 1
        )
        feedback_timeout = float(getattr(args, "mit_feedback_timeout_s", 0.5))

        arrival_joint_err = _deg2rad(float(getattr(args, "nbv_arrival_joint_error_deg", 1.0)))
        arrival_vel = _deg2rad(float(getattr(args, "nbv_arrival_vel_deg_s", 1.0)))
        arrival_pose_mm = float(getattr(args, "nbv_arrival_pose_error_mm", 8.0))
        arrival_ori = _deg2rad(
            float(getattr(args, "nbv_arrival_orientation_error_deg", 3.0))
        )
        arrival_stable = max(int(getattr(args, "nbv_arrival_stable_samples", 5)), 1)
        arrival_timeout = float(getattr(args, "nbv_arrival_timeout_s", 5.0))
        guard_hz = max(float(getattr(args, "mit_runtime_guard_hz", 20.0)), 0.0)
        guard_period = float("inf") if guard_hz <= 0.0 else 1.0 / guard_hz
        planned_flange = np.asarray(stages[-1].flange_pose, dtype=np.float64).reshape(6)
        nbv_task_space_gate = str(label).startswith("active_")
        if not nbv_task_space_gate and all(
            str(getattr(stage, "phase", "")) in {"free_open", "carrying"}
            for stage in stages
        ):
            # Transit and carrying use the same joint tolerance as the next
            # stage's measured-start preflight. That next edge is still checked
            # from actual joints; target approach and support release stay strict.
            arrival_joint_err = min(
                float(getattr(args, "move_joint_tol", _deg2rad(2.0))),
                _deg2rad(float(getattr(args, "motion_stage_start_joint_match_deg", 2.0))),
                tracking_err_limit,
            )
        print(
            f"[mit-exec][{label}][arrival-policy] "
            f"joint<={np.rad2deg(arrival_joint_err):.2f}deg "
            f"position<={arrival_pose_mm:.1f}mm "
            f"orientation<={np.rad2deg(arrival_ori):.2f}deg "
            f"velocity<={np.rad2deg(arrival_vel):.2f}deg/s "
            f"stable_samples={arrival_stable} "
            f"joint_required={not nbv_task_space_gate}"
        )

        log_path = None
        if bool(getattr(args, "mit_q_log", False)):
            out_dir = str(
                getattr(args, "mit_q_log_dir", "")
                or getattr(args, "save_dir", "")
                or "logs"
            )
            os.makedirs(out_dir, exist_ok=True)
            log_path = os.path.join(out_dir, f"mit_trajectory_{label}.jsonl")

        track_consecutive = 0
        last_feedback_stamp = None
        last_new_feedback_mono = time.monotonic()
        last_guard_mono = -float("inf")
        gate_stable = 0
        start = time.monotonic()
        log_rows = []

        def _write_log():
            if not log_path or not log_rows:
                return
            with open(log_path, "a", encoding="utf-8") as stream:
                for row in log_rows:
                    stream.write(json.dumps(row) + "\n")
            log_rows.clear()

        try:
            while True:
                iter_start = time.monotonic()
                self._raise_if_terminal_fault(label)
                self._raise_if_route_stopped(label)
                t = iter_start - start
                q_des, dq_des, _ = traj.sample(t)
                try:
                    self._loop.update_commands(
                        JointCommands(
                            pos=q_des,
                            vel=dq_des,
                            kp=self._kp.copy(),
                            kd=self._kd.copy(),
                            torques=np.zeros(N_JOINTS),
                        )
                    )
                except RuntimeError:
                    self._raise_if_terminal_fault(label)
                    self._raise_if_route_stopped(label)
                    raise

                state = self._loop.get_joint_state()
                self._raise_if_terminal_fault(label)
                stamp = None if state is None else float(getattr(state, "timestamp", 0.0))
                valid_stamp = (
                    stamp is not None
                    and np.isfinite(stamp)
                    and stamp > 0.0
                )
                new_feedback = bool(
                    valid_stamp
                    and (
                        last_feedback_stamp is None
                        or stamp != last_feedback_stamp
                    )
                )
                if new_feedback:
                    last_feedback_stamp = stamp
                    last_new_feedback_mono = iter_start
                elif iter_start - last_new_feedback_mono > feedback_timeout:
                    raise MotionNotReached(
                        f"{label} MIT feedback timeout: SDK joint timestamp "
                        f"did not change for >{feedback_timeout:.2f}s"
                    )

                if state is not None and new_feedback:
                    q_actual = np.asarray(state.pos, dtype=np.float64).reshape(N_JOINTS)
                    dq_actual = np.asarray(state.vel, dtype=np.float64).reshape(N_JOINTS)
                    torque = np.asarray(state.torque, dtype=np.float64).reshape(N_JOINTS)
                    err = float(np.max(np.abs(q_actual - q_des)))
                    if err > tracking_err_limit:
                        track_consecutive += 1
                        if track_consecutive >= tracking_confirm:
                            raise MotionNotReached(
                                f"{label} MIT tracking error {np.rad2deg(err):.2f}deg "
                                f"> {np.rad2deg(tracking_err_limit):.2f}deg for "
                                f"{track_consecutive} samples; aborting"
                            )
                    else:
                        track_consecutive = 0

                    if (
                        runtime_guard is not None
                        and iter_start - last_guard_mono >= guard_period
                    ):
                        runtime_guard(self._arm, args, label, joints=q_actual)
                        last_guard_mono = iter_start

                    if bool(getattr(args, "mit_q_log", False)):
                        log_rows.append(
                            {
                                "t": t,
                                "label": str(label),
                                "feedback_timestamp": stamp,
                                "q_des": q_des.tolist(),
                                "dq_des": dq_des.tolist(),
                                "q_actual": q_actual.tolist(),
                                "dq_actual": dq_actual.tolist(),
                                "tracking_error_deg": float(np.rad2deg(err)),
                                "torque": torque.tolist(),
                            }
                        )

                done = t >= traj.duration_s
                if done:
                    actual_flange = self._arm.get_flange_pose()
                    pos_err, ori_err = _flange_pose_error(actual_flange, planned_flange)
                    q_actual = np.asarray(
                        self._arm.get_joint_angles(), dtype=np.float64
                    ).reshape(N_JOINTS)
                    state = self._loop.get_joint_state()
                    dq_actual = (
                        np.zeros(N_JOINTS)
                        if state is None
                        else np.asarray(state.vel, dtype=np.float64).reshape(N_JOINTS)
                    )
                    joint_err = float(np.max(np.abs(q_actual - goal)))
                    vel_ok = float(np.max(np.abs(dq_actual))) <= arrival_vel
                    joint_ok = joint_err <= arrival_joint_err
                    pose_ok = pos_err * 1000.0 <= arrival_pose_mm
                    ori_ok = ori_err <= arrival_ori
                    task_space_ok = bool(vel_ok and pose_ok and ori_ok)
                    arrival_ok = bool(
                        task_space_ok
                        and (nbv_task_space_gate or joint_ok)
                    )
                    if new_feedback:
                        if arrival_ok:
                            gate_stable += 1
                        else:
                            gate_stable = 0
                    if gate_stable >= arrival_stable:
                        print(
                            f"[mit-exec][{label}] arrival accepted: "
                            f"gate={'nbv_task_space' if nbv_task_space_gate else 'joint_and_task_space'} "
                            f"joint_err={np.rad2deg(joint_err):.2f}deg "
                            f"vel={np.rad2deg(float(np.max(np.abs(dq_actual)))):.2f}deg/s "
                            f"flange_pos={pos_err * 1000.0:.2f}mm "
                            f"flange_ori={np.rad2deg(ori_err):.2f}deg "
                            f"stable={gate_stable}/{arrival_stable}"
                        )
                        _write_log()
                        self.hold()  # hold at the reached goal
                        return goal
                    if time.monotonic() - start - traj.duration_s > arrival_timeout:
                        raise MotionNotReached(
                            f"{label} MIT arrival gate failed: "
                            f"gate={'nbv_task_space' if nbv_task_space_gate else 'joint_and_task_space'} "
                            f"joint_err={np.rad2deg(joint_err):.2f}deg "
                            f"joint_limit={np.rad2deg(arrival_joint_err):.2f}deg "
                            f"vel={np.rad2deg(float(np.max(np.abs(dq_actual)))):.2f}deg/s "
                            f"flange_pos={pos_err * 1000.0:.2f}mm "
                            f"flange_ori={np.rad2deg(ori_err):.2f}deg"
                        )

                if len(log_rows) >= 200:
                    _write_log()
                elapsed = time.monotonic() - iter_start
                if elapsed < self._period:
                    time.sleep(self._period - elapsed)
        except BaseException as exc:
            with defer_sigint():
                self._raise_if_terminal_fault(label)
                if isinstance(exc, KeyboardInterrupt):
                    self.prepare_shutdown()
                elif not bool(getattr(self._arm, "_safety_stop_latched", False)):
                    self.hold()  # preserve a previously latched controlled hold
                _write_log()  # Freeze the target BEFORE slow disk cleanup.
            raise

    # ------------------------------------------------------------------ #
    # Backend-compatible facade (used by servo_delivery)
    # ------------------------------------------------------------------ #

    def get_joint_angles(self):
        state = self._loop.get_joint_state()
        if state is not None:
            return np.asarray(state.pos, dtype=np.float64).reshape(N_JOINTS).copy()
        return np.asarray(self._arm.get_joint_angles(), dtype=np.float64).reshape(N_JOINTS)

    def get_joint_velocity(self):
        state = self._loop.get_joint_state()
        if state is None or state.vel is None:
            return None
        return np.asarray(state.vel, dtype=np.float64).reshape(N_JOINTS).copy()

    def set_cartesian_impedance(self, cfg):
        self.ensure_started()
        return self._loop.set_cartesian_impedance(cfg)

    def disable_cartesian_impedance(self):
        return self._loop.disable_cartesian_impedance()

    def get_debug_state(self, include_trace: bool = False) -> dict:
        return self._loop.get_debug_state(include_trace=include_trace)

    def get_robot(self):
        return self._driver

    def emergency_stop(self):
        self._emergency_stop_requested = True
        try:
            self._loop.request_emergency_stop("MIT session emergency stop")
        finally:
            self.shutdown()

    def disconnect(self):
        self.shutdown()

    @property
    def loop(self):
        return self._loop


# Backward-compatible alias.
MITTrajectoryExecutor = MITTrajectorySession


def get_mit_session(arm, args):
    """Return the one persistent MIT session for ``args``, creating it lazily."""
    session = getattr(args, "_mit_session", None)
    driver = getattr(arm, "arm", None)
    if session is None or getattr(session, "_driver", None) is not driver:
        session = MITTrajectorySession(arm, args)
        args._mit_session = session
        arm._mit_session_owner = session
        # Ensure the 200 Hz loop is stopped before the driver disconnects, so it
        # never sends move_mit on a dead CAN bus (the Ctrl+C "comm is None" bug).
        register = getattr(arm, "_register_pre_disconnect", None)
        if callable(register):
            register(session.shutdown)
    return session


def reset_mit_session(args):
    """Shut down and clear the cached session (error recovery / program end)."""
    session = getattr(args, "_mit_session", None)
    if session is not None:
        try:
            session.shutdown()
        except Exception as exc:
            print(f"[mit-session][warn] shutdown failed: {type(exc).__name__}: {exc}")
    args._mit_session = None
