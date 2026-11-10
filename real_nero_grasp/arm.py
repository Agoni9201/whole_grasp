"""Nero arm adapter with guarded motion and gripper contact handling."""

import time

import numpy as np

from tools.real_nero_point_test import NeroArm
from control_backends.force_ctrl.normal_shutdown import defer_sigint, report_shutdown
from control_backends.force_ctrl.feedback_health import FeedbackHealthMonitor
from control_backends.force_ctrl.mit_loop import NERO_JOINT_LIMITS

from .types import GripperCloseResult, SafetyStopTriggered


__all__ = ["RealNeroGraspArm"]


class RealNeroGraspArm(NeroArm):
    def __init__(
        self,
        *args,
        gripper_enabled=True,
        gripper_max_width=0.100,
        gripper_max_force=0.50,
        **kwargs,
    ):
        ordinary_stop_settle_timeout = kwargs.pop(
            "ordinary_stop_settle_timeout", None
        )
        kwargs.pop("ordinary_stop_trend_extension", None)
        super().__init__(*args, **kwargs)
        if ordinary_stop_settle_timeout is not None:
            self._stop_confirm_timeout = float(ordinary_stop_settle_timeout)
        self.gripper_enabled = bool(gripper_enabled)
        self.gripper_max_width = float(gripper_max_width)
        self.gripper_max_force = float(gripper_max_force)
        # A software safety stop is terminal for this process.  Several recovery
        # paths intentionally catch ordinary motion failures, so relying only on
        # exception propagation could otherwise allow a later move/effector
        # command after an electronic emergency stop was requested.
        self._safety_stop_latched = False
        self._safety_stop_reason = None
        self._safety_stop_request_confirmed = False
        self._safety_stop_max_attempts = 3
        self._controlled_stop_request_confirmed = False
        self._suppress_program_exit_estop = False
        if not np.isfinite(self.gripper_max_width) or self.gripper_max_width <= 0.0:
            raise ValueError("gripper_max_width must be finite and positive")
        if not np.isfinite(self.gripper_max_force) or self.gripper_max_force <= 0.0:
            raise ValueError("gripper_max_force must be finite and positive")
        self.gripper = None
        if self.gripper_enabled:
            self.gripper = self.arm.init_effector(self.arm.OPTIONS.EFFECTOR.AGX_GRIPPER)

    def _send_ordinary_trajectory_terminate(self, frames=3):
        """Clear any queued controller trajectory without setting e-stop."""
        from pyAgxArm.protocols.can_protocol.msgs.nero.default.transmit import (
            ArmMsgMotionCtrl,
        )

        sender = getattr(self.arm, "_send_msg", None)
        if not callable(sender):
            raise RuntimeError(
                "pyAgxArm driver does not expose the CAN trajectory-control sender"
            )
        count = max(int(frames), 1)
        terminate_msg = ArmMsgMotionCtrl(
            emergency_stop=0,
            track_ctrl=0x06,
            grag_teach_ctrl=0,
        )
        for index in range(count):
            sender(terminate_msg)
            if index + 1 < count:
                time.sleep(0.02)
        return count

    def _wait_for_stationary_joints(self, timeout_s=1.5):
        """Return the latest measured joints after motion has settled."""
        stable_required = 3
        stable_count = 0
        previous = None
        current = None
        last_delta_deg = float("inf")
        deadline = time.time() + float(timeout_s)
        while time.time() < deadline:
            time.sleep(0.05)
            current = np.asarray(self.get_joint_angles(), dtype=float).reshape(7)
            if not np.all(np.isfinite(current)):
                raise RuntimeError("Nero returned non-finite startup joint feedback")
            if previous is not None:
                last_delta_deg = float(
                    np.rad2deg(np.max(np.abs(current - previous)))
                )
                if last_delta_deg <= 0.15:
                    stable_count += 1
                else:
                    stable_count = 0
                if stable_count >= stable_required:
                    return current.copy(), last_delta_deg
            previous = current
        raise RuntimeError(
            "joint settling was not confirmed after clearing the controller "
            f"trajectory; last_delta={last_delta_deg:.3f}deg"
        )

    def __enter__(self):
        """Connect without allowing a stale target to resume on enable.

        Some Nero controller/driver combinations retain the previous ``move_j``
        target across a process disconnect.  The base adapter enabled the arm
        and restored normal speed immediately, so that old target could start
        moving before the new process displayed its initial-pose prompt.

        Clear the trajectory once while disabled and once immediately after
        enabling at zero speed, then overwrite it with the measured stationary
        joint position.  Only after that handshake is the configured speed
        restored.  No configured initial-pose target is sent here.
        """
        self.arm.connect()
        try:
            # This pre-enable terminate is the critical anti-resume barrier: it
            # clears a target left by a previously interrupted Python process
            # before motor enable can make that target live again.
            self._send_ordinary_trajectory_terminate(frames=3)

            deadline = time.time() + self.enable_timeout_s
            while True:
                if bool(self.arm.enable()):
                    break
                if time.time() > deadline:
                    raise TimeoutError("Timed out waiting for Nero enable()")
                time.sleep(0.05)

            # Do not restore execution speed until the post-enable controller
            # queue is empty and measured feedback is stationary.
            self.arm.set_speed_percent(0)
            self._send_ordinary_trajectory_terminate(frames=3)
            hold_joints, last_delta_deg = self._wait_for_stationary_joints()
            self.arm.move_j([float(value) for value in hold_joints])
            self._startup_hold_joints = hold_joints.copy()

            if self.speed_percent is not None:
                self.arm.set_speed_percent(int(self.speed_percent))
            print(
                "[startup] stale trajectory cleared; current joint hold verified "
                f"(last_delta={last_delta_deg:.3f}deg). No initial-pose motion "
                "has been commanded."
            )
            return self
        except Exception:
            try:
                self.arm.disconnect()
            except Exception:
                pass
            raise
        except KeyboardInterrupt:
            with defer_sigint():
                self.stop_for_program_exit("Ctrl+C during arm initialization")
                self.arm.disconnect()
            raise

    def _assert_motion_not_latched(self, command):
        if bool(getattr(self, "_safety_stop_latched", False)):
            raise SafetyStopTriggered(
                f"safety stop is latched; refusing subsequent {command} command"
            )
        if command in ("move_p", "move_j") and self._mit_control_owner() is not None:
            raise SafetyStopTriggered(
                f"MIT owns joint control; refusing ordinary {command} command"
            )

    def _mit_control_owner(self):
        owner = self.__dict__.get("_mit_session_owner")
        return owner if owner is not None and owner.owns_control else None

    def move_p(self, pose):
        self._assert_motion_not_latched("move_p")
        return super().move_p(pose)

    def move_j(self, joints):
        self._assert_motion_not_latched("move_j")
        return super().move_j(joints)

    def stop(self):
        # Refuse motion immediately, but do not treat the hardware RPC as
        # confirmed until it actually returns.  If CAN communication failed on
        # the first request, a later safety handler is still allowed to retry.
        self._safety_stop_latched = True
        if bool(getattr(self, "_safety_stop_request_confirmed", False)):
            return None
        attempts = max(int(getattr(self, "_safety_stop_max_attempts", 3)), 1)
        last_error = None
        for attempt in range(1, attempts + 1):
            try:
                owner = self._mit_control_owner()
                response = (
                    owner.emergency_stop()
                    if owner is not None
                    else super().stop()
                )
                self._safety_stop_request_confirmed = True
                return response
            except Exception as exc:
                last_error = exc
                report_shutdown(
                    f"[stop][warn] electronic emergency stop request "
                    f"{attempt}/{attempts} failed: {type(exc).__name__}: {exc}"
                )
        raise last_error

    def get_joint_torques(self):
        """Read Nero's seven high-speed joint-side torque estimates in N·m.

        The driver derives these values from motor-current feedback and gear
        reduction.  A partial/stale feedback set is deliberately reported as
        unavailable instead of inventing zeros, because zero would hide an
        external-contact event.
        """
        getter = getattr(self.arm, "get_motor_states", None)
        parser = getattr(self.arm, "_parser", None)
        torques = []
        for joint_index in range(1, 8):
            try:
                # V111's SDK getter zeroes velocity on this shared parser
                # object. Read torque without corrupting MIT diagnostics or
                # Cartesian damping's real-velocity source.
                wrapped = getattr(parser, f"motor_state_{joint_index}", None)
                message = getattr(wrapped, "msg", wrapped)
                torque = float(getattr(message, "torque"))
            except Exception:
                if not callable(getter):
                    return None
                try:
                    wrapped = getter(joint_index)
                    message = getattr(wrapped, "msg", wrapped)
                    torque = float(getattr(message, "torque"))
                except Exception:
                    return None
            if not np.isfinite(torque):
                return None
            torques.append(torque)
        return np.asarray(torques, dtype=np.float64)

    def allow_controlled_recovery(self):
        """Permit exactly one validated retreat after a confirmed normal stop."""
        if not bool(getattr(self, "_controlled_stop_request_confirmed", False)):
            raise RuntimeError(
                "cannot recover because ordinary trajectory termination was not confirmed"
            )
        if bool(getattr(self, "_safety_stop_request_confirmed", False)):
            raise RuntimeError(
                "cannot recover after an electronic emergency stop request"
            )
        self._safety_stop_latched = False
        # A retreat is a new motion edge.  If it later needs stopping, it must
        # send a fresh normal-stop request rather than reusing old success.
        self._controlled_stop_request_confirmed = False

    def finish_controlled_recovery(self):
        """Make a post-stop workflow terminal after its one recovery motion."""
        self._safety_stop_latched = True

    def _controller_motion_active(self):
        """Return True/False from Nero motion feedback, or None if unavailable."""
        getter = getattr(self.arm, "get_arm_status", None)
        if not callable(getter):
            return None
        try:
            wrapped = getter()
            stamp = float(getattr(wrapped, "timestamp", 0.0))
            age = time.time() - stamp
            if not np.isfinite(stamp) or stamp <= 0 or not 0 <= age <= 0.1:
                return None
            status = getattr(wrapped, "msg", wrapped)
            value = getattr(status, "motion_status")
            value = int(getattr(value, "value", value))
        except Exception:
            return None
        if value == 0:
            return False
        if value == 1:
            return True
        return None

    def _confirm_idle_for_program_exit(self, first_joints=None):
        """Confirm an idle controller without sending any motion command.

        ``track_ctrl=0x06`` terminates a controller trajectory, but on the real
        Nero it can briefly release trajectory holding before the subsequent
        current-position ``move_j`` arrives.  During CPU-only planning there is
        no trajectory to terminate, so sending it can create the visible
        down-then-up twitch reported on Ctrl+C.
        """
        reported_active = self._controller_motion_active()
        if reported_active is not False:
            return None
        try:
            monitor = FeedbackHealthMonitor()
            first = self._read_stop_feedback(monitor)
            time.sleep(0.06)
            second = self._read_stop_feedback(monitor)
            dt = second["position_timestamps"] - first["position_timestamps"]
            if not np.all(dt > 0) or not np.all(second["motor_timestamps"] > first["motor_timestamps"]):
                return None
            delta = np.abs(second["q"] - first["q"])
            if np.any(delta / dt > np.deg2rad(1.0)):
                return None
        except Exception:
            return None
        delta_deg = float(np.rad2deg(np.max(delta)))
        if delta_deg > 0.15:
            return None

        self._safety_stop_latched = True
        self._controlled_stop_request_confirmed = True
        return {
            "confirmed": True,
            "method": "controller_idle_no_command",
            "motion_status": "idle" if reported_active is False else "unavailable",
            "last_joint_delta_deg": delta_deg,
        }

    def _read_stop_feedback(self, monitor):
        sample = monitor.read(self.arm, time.time(), time.monotonic())
        q = sample["q"]
        if np.any(q < NERO_JOINT_LIMITS[:, 0]) or np.any(q > NERO_JOINT_LIMITS[:, 1]):
            raise RuntimeError("Stop feedback is outside physical joint limits")
        return sample

    def _register_pre_disconnect(self, callback):
        """Register a callback run before any shutdown/disconnect touches the driver."""
        if not hasattr(self, "_pre_disconnect_callbacks"):
            self._pre_disconnect_callbacks = []
        self._pre_disconnect_callbacks.append(callback)

    def _run_pre_disconnect_callbacks(self):
        for callback in list(getattr(self, "_pre_disconnect_callbacks", ())):
            try:
                callback()
            except Exception as exc:
                report_shutdown(
                    f"[shutdown][warn] pre-disconnect callback failed: "
                    f"{type(exc).__name__}: {exc}"
                )

    @defer_sigint()
    def stop_for_program_exit(self, reason="program exit"):
        """Stop a live trajectory before process exit or CAN disconnect."""
        owner = self._mit_control_owner()
        if owner is None:
            self._run_pre_disconnect_callbacks()
        try:
            fallback = np.asarray(self.get_joint_angles(), dtype=float).reshape(7)
        except Exception:
            fallback = None
        report_shutdown(f"[shutdown] {reason}; checking whether a live trajectory exists")
        try:
            if owner is not None:
                # Keep the sole MIT sender alive until zero-velocity holding
                # and fresh feedback have confirmed the stop.  Never infer
                # MIT idleness from the ordinary trajectory status bit.
                response = self.terminate_motion(fallback_joints=fallback)
                owner.shutdown()
            else:
                response = self._confirm_idle_for_program_exit(first_joints=fallback)
            if response is None:
                report_shutdown("[shutdown] live/uncertain motion detected; requesting ordinary trajectory termination")
                response = self.terminate_motion(fallback_joints=fallback)
            elif response.get("method") == "controller_idle_no_command":
                report_shutdown(
                    "[shutdown] controller and measured joints are already idle; "
                    "no terminate frame, zero-speed command, or hold move was sent"
                )
            report_shutdown(
                "[shutdown] safe shutdown state confirmed; "
                "electronic emergency stop was not requested: "
                f"details={response}"
            )
            return True
        except Exception as exc:
            if bool(getattr(self, "_suppress_program_exit_estop", False)):
                report_shutdown(
                    f"[shutdown][warn] {reason} ordinary trajectory "
                    "termination could not be confirmed, but electronic "
                    "emergency stop is suppressed for this controlled "
                    f"initial-timeout exit: {type(exc).__name__}: {exc}"
                )
                return False
            # If motion cannot be proven stopped, preserving the robot is more
            # important than honoring the preference for a non-e-stop exit.
            detail = (
                f"{reason} ordinary trajectory termination could not be confirmed: "
                f"{type(exc).__name__}: {exc}; escalating to electronic emergency stop"
            )
            report_shutdown(f"[shutdown][critical] {detail}")
            self._controlled_stop_request_confirmed = False
            try:
                self.stop()
            except Exception as stop_exc:
                report_shutdown(
                    "[shutdown][critical] electronic emergency stop request also "
                    f"failed: {type(stop_exc).__name__}: {stop_exc}"
                )
            return False

    def stop_for_keyboard_interrupt(self):
        """Stop a live trajectory for Ctrl+C without normally entering e-stop."""
        return self.stop_for_program_exit("Ctrl+C received")

    @defer_sigint()
    def __exit__(self, exc_type, exc_value, traceback):
        # Always clear the controller trajectory before CAN disconnect.  This
        # also covers normal returns and exceptions converted into result
        # objects, not only KeyboardInterrupt.  Without it, the next process
        # could re-enable a target retained by the controller.
        already_stopped = bool(
            getattr(self, "_controlled_stop_request_confirmed", False)
            or getattr(self, "_safety_stop_request_confirmed", False)
        )
        if not already_stopped:
            if exc_type is not None and issubclass(exc_type, KeyboardInterrupt):
                reason = "Ctrl+C received"
            elif exc_type is not None:
                reason = f"program exiting after {exc_type.__name__}"
            else:
                reason = "program workflow finished"
            self.stop_for_program_exit(reason)
        # This owner is safety-critical: a swallowed callback exception must
        # never allow disconnect while its sender could still be alive.
        owner = self._mit_control_owner()
        if owner is not None:
            try:
                owner.shutdown()
            except Exception as exc:
                try:
                    self.stop()
                except Exception as stop_exc:
                    print(f"[shutdown][critical] MIT emergency stop/join failed: {stop_exc}")
                raise RuntimeError(
                    "MIT sender shutdown was not confirmed; refusing driver disconnect"
                ) from exc
        self._run_pre_disconnect_callbacks()
        return super().__exit__(exc_type, exc_value, traceback)

    def terminate_motion(self, fallback_joints=None):
        """Terminate the current trajectory without setting the e-stop bit.

        pyAgxArm does not expose a public normal-stop method for Nero.  Its
        CAN motion-control message does define ``track_ctrl=0x06`` as
        "terminate execution", so send that request redundantly, set normal
        trajectory speed to zero, and overwrite the old move_j target with a
        fresh measured hold position.  Continuous joint feedback is required
        before reporting the ordinary stop as confirmed.

        This method deliberately never calls ``electronic_emergency_stop``.
        The process latch is set first so no later grasp/recovery command can
        be sent even if ordinary-stop confirmation fails.
        """
        self._safety_stop_latched = True
        timeout_s = float(getattr(self, "_stop_confirm_timeout", 2.0))
        owner = self._mit_control_owner()
        if owner is not None:
            response = owner.terminate_motion(timeout_s=timeout_s)
            if not response.get("confirmed", False):
                raise RuntimeError("MIT stop was not confirmed")
            self._controlled_stop_request_confirmed = True
            return response

        if bool(getattr(self, "_controlled_stop_request_confirmed", False)):
            response = self._confirm_idle_for_program_exit()
            if response is None:
                self._controlled_stop_request_confirmed = False
                raise RuntimeError("Previously confirmed stop is no longer verified")
            return response

        driver = self.arm
        monitor = FeedbackHealthMonitor()
        self._read_stop_feedback(monitor)
        terminate_frames = self._send_ordinary_trajectory_terminate(frames=3)

        # The low-level terminate command has no acknowledgement in this SDK.
        # A zero-speed normal joint target overwrites the asynchronous move_j
        # destination without disabling motors or entering e-stop state.
        driver.set_speed_percent(0)
        time.sleep(0.05)
        hold_joints = self._read_stop_feedback(monitor)["q"]
        driver.move_j([float(value) for value in hold_joints])

        stable_required = 3
        stable_count = 0
        previous = None
        last_delta_deg = float("inf")
        # Set the deadline for the stop confirmation
        deadline = time.monotonic() + timeout_s
    
        while time.monotonic() < deadline:
            time.sleep(0.05)
            current = self._read_stop_feedback(monitor)
            if previous is not None:
                dt = current["position_timestamps"] - previous["position_timestamps"]
                if not np.all(dt > 0) or not np.all(current["motor_timestamps"] > previous["motor_timestamps"]):
                    continue
                delta = np.abs(current["q"] - previous["q"])
                last_delta_deg = float(
                    np.rad2deg(np.max(delta))
                )
                report_shutdown(
                    f"[stop][confirm] last_delta={last_delta_deg:.3f}deg "
                    f"stable={stable_count}/{stable_required}"
                )
                if last_delta_deg <= 0.15 and np.all(delta / dt <= np.deg2rad(1.0)):
                    stable_count += 1
                else:
                    stable_count = 0
                if stable_count >= stable_required:
                    self._controlled_stop_request_confirmed = True
                    return {
                        "confirmed": True,
                        "method": "track_terminate_speed_zero_hold",
                        "terminate_frames": terminate_frames,
                        "last_joint_delta_deg": last_delta_deg,
                    }
            previous = current

        raise RuntimeError(
            "ordinary trajectory termination was sent but joint settling was "
            f"not confirmed within {timeout_s:.1f}s; last_delta={last_delta_deg:.3f}deg"
        )

    def _limit_gripper_width(self, width):
        requested = float(width)
        if not np.isfinite(requested):
            raise ValueError(f"non-finite gripper width command: {requested}")
        limited = float(np.clip(requested, 0.0, self.gripper_max_width))
        if abs(limited - requested) > 1e-9:
            print(
                f"[gripper][limit] requested width={requested:.4f}m capped to "
                f"hardware max={self.gripper_max_width:.4f}m"
            )
        return limited

    def _limit_gripper_force(self, force):
        requested = float(force)
        if not np.isfinite(requested) or requested <= 0.0:
            raise ValueError(f"invalid gripper force command: {requested}")
        limited = min(requested, self.gripper_max_force)
        if limited < requested - 1e-9:
            print(
                f"[gripper][limit] requested force={requested:.3f}N capped to "
                f"configured safety max={self.gripper_max_force:.3f}N"
            )
        return float(limited)

    def _send_gripper_command(self, width, force):
        response = self.gripper.move_gripper_m(value=float(width), force=float(force))
        # Preserve the actual loaded command for grasp->servo ownership handoff.
        # Measured contact width can be wider and must not replace it silently.
        self._last_gripper_command = dict(width=float(width), force=float(force))
        return response

    def open_gripper(self, width, force, settle_s=0.8):
        self._assert_motion_not_latched("gripper-open")
        if self.gripper is None:
            print("[gripper] disabled; skip open")
            return
        width = self._limit_gripper_width(width)
        force = self._limit_gripper_force(force)
        print(f"[gripper] open width={float(width):.4f}m force={float(force):.3f}N")
        resp = self._send_gripper_command(width, force)
        print("[gripper] open response =", resp)
        time.sleep(float(settle_s))
        self.print_gripper_status()

    def open_gripper_nowait(self, width, force):
        self._assert_motion_not_latched("gripper-open")
        if self.gripper is None:
            print("[gripper] disabled; skip async open")
            return
        width = self._limit_gripper_width(width)
        force = self._limit_gripper_force(force)
        print(f"[gripper] async open width={float(width):.4f}m force={float(force):.3f}N")
        resp = self._send_gripper_command(width, force)
        print("[gripper] async open response =", resp)

    def close_gripper(self, width, force, settle_s=1.0):
        self._assert_motion_not_latched("gripper-close")
        if self.gripper is None:
            print("[gripper] disabled; skip close")
            return None
        width = self._limit_gripper_width(width)
        force = self._limit_gripper_force(force)
        print(f"[gripper] close width={float(width):.4f}m force={float(force):.3f}N")
        resp = self._send_gripper_command(width, force)
        print("[gripper] close response =", resp)
        time.sleep(float(settle_s))
        return self.print_gripper_status()

    def close_gripper_slow(
        self,
        open_width,
        close_width,
        force,
        step_m,
        interval_s,
        settle_s=1.0,
        verbose=False,
        stop_force=None,
        contact_confirm_samples=2,
        contact_confirm_interval_s=0.10,
        empty_width_threshold=0.005,
        closed_width_tolerance=0.001,
        close_completion_timeout_s=2.0,
        close_completion_poll_s=0.10,
    ):
        self._assert_motion_not_latched("gripper-slow-close")
        if self.gripper is None:
            print("[gripper] disabled; skip slow close")
            return GripperCloseResult(None, False, "gripper_disabled")
        open_width = self._limit_gripper_width(open_width)
        close_width = self._limit_gripper_width(close_width)
        step_m = float(step_m)
        interval_s = float(interval_s)
        settle_s = float(settle_s)
        contact_confirm_interval_s = float(contact_confirm_interval_s)
        force = float(force)
        empty_width_threshold = float(empty_width_threshold)
        closed_width_tolerance = float(closed_width_tolerance)
        close_completion_timeout_s = float(close_completion_timeout_s)
        close_completion_poll_s = float(close_completion_poll_s)
        try:
            contact_confirm_samples = int(contact_confirm_samples)
        except Exception:
            return GripperCloseResult(None, False, "invalid_close_configuration")
        if (
            not np.isfinite(step_m)
            or step_m <= 0.0
            or not np.isfinite(interval_s)
            or interval_s < 0.0
            or not np.isfinite(settle_s)
            or settle_s < 0.0
            or not np.isfinite(contact_confirm_interval_s)
            or contact_confirm_interval_s < 0.0
            or not np.isfinite(empty_width_threshold)
            or empty_width_threshold < 0.0
            or not np.isfinite(closed_width_tolerance)
            or closed_width_tolerance < 0.0
            or not np.isfinite(close_completion_timeout_s)
            or close_completion_timeout_s < 0.0
            or not np.isfinite(close_completion_poll_s)
            or close_completion_poll_s <= 0.0
            or contact_confirm_samples < 2
        ):
            return GripperCloseResult(None, False, "invalid_close_configuration")
        if not np.isfinite(force) or force <= 0.0:
            return GripperCloseResult(None, False, "invalid_command_force")
        force = self._limit_gripper_force(force)
        if stop_force is None:
            return GripperCloseResult(None, False, "missing_stop_force")
        stop_force = float(stop_force)
        if not np.isfinite(stop_force) or stop_force <= 0.0:
            return GripperCloseResult(None, False, "invalid_stop_force")
        if stop_force > abs(force):
            return GripperCloseResult(None, False, "stop_force_exceeds_command_force")
        if (
            open_width <= close_width
            or closed_width_tolerance >= (open_width - close_width)
        ):
            return GripperCloseResult(None, False, "invalid_close_range")

        # ``empty_width_threshold`` remains a compatibility input for callers and
        # post-close policy.  During motion, however, only the physical close
        # target plus a tight tolerance means "fully closed".  The old 5 mm
        # heuristic must not turn a transient force spike into an early abort.
        fully_closed_width = float(close_width + closed_width_tolerance)

        def _feedback_error(phase, kind):
            names = {
                ("close", "missing"): "missing_close_feedback",
                ("close", "mode"): "invalid_feedback_mode",
                ("close", "force"): "invalid_force_feedback",
                ("close", "width"): "invalid_width_feedback",
                ("contact_confirmation", "missing"): "missing_contact_confirmation",
                ("contact_confirmation", "mode"): "invalid_contact_confirmation_mode",
                ("contact_confirmation", "force"): "invalid_contact_confirmation",
                ("contact_confirmation", "width"): "invalid_contact_confirmation",
                ("settled_contact", "missing"): "missing_settled_contact_status",
                ("settled_contact", "mode"): "invalid_settled_contact_status",
                ("settled_contact", "force"): "invalid_settled_contact_status",
                ("settled_contact", "width"): "invalid_settled_contact_status",
                ("final_close", "missing"): "missing_final_close_status",
                ("final_close", "mode"): "invalid_final_close_status",
                ("final_close", "force"): "invalid_final_close_status",
                ("final_close", "width"): "invalid_final_close_status",
            }
            return names.get((phase, kind), f"invalid_{phase}_feedback")

        def _read_feedback(phase):
            status = self.print_gripper_status()
            if status is None:
                reason = _feedback_error(phase, "missing")
                print(f"[gripper][warn] missing feedback during {phase}; stop closing")
                return None, None, None, reason
            mode = str(getattr(status, "mode", "")).strip().lower()
            if mode != "width":
                reason = _feedback_error(phase, "mode")
                print(
                    f"[gripper][warn] unexpected feedback mode={mode!r} "
                    f"during {phase}; stop closing"
                )
                return status, None, None, reason
            try:
                measured_force = abs(float(status.force))
            except Exception:
                measured_force = float("nan")
            if not np.isfinite(measured_force):
                reason = _feedback_error(phase, "force")
                print(f"[gripper][warn] invalid force feedback during {phase}; stop closing")
                return status, None, None, reason
            try:
                measured_width = float(status.value)
            except Exception:
                measured_width = float("nan")
            if not np.isfinite(measured_width):
                reason = _feedback_error(phase, "width")
                print(f"[gripper][warn] invalid width feedback during {phase}; stop closing")
                return status, None, measured_force, reason
            return status, measured_width, measured_force, None

        def _is_fully_closed(measured_width):
            return bool(float(measured_width) <= fully_closed_width)

        def _closed_result(status, measured_width, measured_force, reason):
            print(
                f"[gripper][empty] fully closed at width={float(measured_width):.4f}m "
                f"(target={close_width:.4f}m tol={closed_width_tolerance:.4f}m); "
                "no object contact"
            )
            return GripperCloseResult(
                status,
                False,
                reason,
                contact_width=float(measured_width),
                contact_force=float(measured_force),
            )

        def _confirm_contact(status, measured_width, measured_force):
            if _is_fully_closed(measured_width):
                return "closed", _closed_result(
                    status,
                    measured_width,
                    measured_force,
                    "force_at_closed_limit",
                )
            confirmed = 1
            required = contact_confirm_samples
            last_status = status
            last_width = float(measured_width)
            last_force = float(measured_force)
            while confirmed < required:
                time.sleep(contact_confirm_interval_s)
                last_status, last_width, last_force, error = _read_feedback(
                    "contact_confirmation"
                )
                if error is not None:
                    return "error", GripperCloseResult(
                        last_status,
                        False,
                        error,
                        contact_width=last_width,
                        contact_force=last_force,
                    )
                if verbose:
                    print(
                        f"[gripper][contact][sample] {confirmed + 1}/{required} "
                        f"measured_width={last_width:.4f}m force={last_force:.3f}"
                    )
                if _is_fully_closed(last_width):
                    return "closed", _closed_result(
                        last_status,
                        last_width,
                        last_force,
                        "force_at_closed_limit",
                    )
                if last_force < stop_force:
                    print(
                        f"[gripper][contact][transient] force dropped to "
                        f"{last_force:.3f} < {float(stop_force):.3f}; "
                        "resume closing toward the physical close limit"
                    )
                    return "transient", None
                confirmed += 1

            if settle_s > 0.0:
                time.sleep(settle_s)
                last_status, last_width, last_force, error = _read_feedback(
                    "settled_contact"
                )
                if error is not None:
                    return "error", GripperCloseResult(
                        last_status,
                        False,
                        error,
                        contact_width=last_width,
                        contact_force=last_force,
                    )
                if _is_fully_closed(last_width):
                    return "closed", _closed_result(
                        last_status,
                        last_width,
                        last_force,
                        "force_at_closed_limit",
                    )
                if last_force < stop_force:
                    print(
                        f"[gripper][contact][transient] force dropped to "
                        f"{last_force:.3f} < {float(stop_force):.3f} after settle; "
                        "resume closing toward the physical close limit"
                    )
                    return "transient", None

            print(
                f"[gripper][contact] confirmed samples={confirmed}/{required} "
                f"width={last_width:.4f}m force={last_force:.3f}"
            )
            return "contact", GripperCloseResult(
                last_status,
                True,
                "force_contact",
                contact_width=last_width,
                contact_force=last_force,
            )

        widths = list(np.arange(open_width, close_width, -step_m, dtype=float))
        if len(widths) == 0 or abs(widths[-1] - close_width) > 1e-6:
            widths.append(close_width)
        print(
            f"[gripper] slow close from {open_width:.4f}m to {close_width:.4f}m "
            f"step={step_m:.4f}m interval={interval_s:.2f}s force={float(force):.3f}N"
        )
        print(f"[gripper] slow close force stop enabled: |force| >= {stop_force:.3f}")
        last_status = None
        saw_below_threshold = False
        for width in widths:
            width = max(close_width, min(open_width, float(width)))
            if verbose:
                print(f"[gripper] slow close command width={width:.4f}m")
            resp = self._send_gripper_command(width, force)
            if verbose:
                print("[gripper] slow close response =", resp)
            time.sleep(interval_s)
            last_status, measured_width, measured_force, error = _read_feedback("close")
            if error is not None:
                return GripperCloseResult(
                    last_status,
                    False,
                    error,
                    contact_width=measured_width,
                    contact_force=measured_force,
                )
            if measured_force < stop_force:
                saw_below_threshold = True
                continue
            if not saw_below_threshold:
                print(
                    "[gripper][warn] force was already above threshold at the first valid "
                    "sample; contact has no low-force baseline and is rejected"
                )
                return GripperCloseResult(
                    last_status,
                    False,
                    "contact_without_force_baseline",
                    contact_width=measured_width,
                    contact_force=measured_force,
                )
            print(
                f"[gripper][contact] force threshold reached: command_width={float(width):.4f}m "
                f"measured_width={measured_width:.4f}m |force|={measured_force:.3f} "
                f">= {float(stop_force):.3f}; keep the loaded command and confirm"
            )
            outcome, result = _confirm_contact(
                last_status,
                measured_width,
                measured_force,
            )
            if result is not None:
                return result
            if outcome == "transient":
                saw_below_threshold = True
                continue

        if settle_s > 0:
            time.sleep(float(settle_s))
        completion_deadline = time.monotonic() + close_completion_timeout_s
        while True:
            last_status, measured_width, measured_force, error = _read_feedback(
                "final_close"
            )
            if error is not None:
                return GripperCloseResult(
                    last_status,
                    False,
                    error,
                    contact_width=measured_width,
                    contact_force=measured_force,
                )
            if _is_fully_closed(measured_width):
                return _closed_result(
                    last_status,
                    measured_width,
                    measured_force,
                    "close_limit_without_contact",
                )
            if measured_force >= stop_force:
                if not saw_below_threshold:
                    return GripperCloseResult(
                        last_status,
                        False,
                        "contact_without_force_baseline",
                        contact_width=measured_width,
                        contact_force=measured_force,
                    )
                print(
                    f"[gripper][contact] delayed force contact after close-limit command: "
                    f"width={measured_width:.4f}m |force|={measured_force:.3f}; confirm"
                )
                outcome, result = _confirm_contact(
                    last_status,
                    measured_width,
                    measured_force,
                )
                if result is not None:
                    return result
                if outcome == "transient":
                    # The 0 mm command remains loaded.  A transient final spike
                    # is not a reason to retract; keep polling for actual closure
                    # or a later sustained contact until the bounded deadline.
                    saw_below_threshold = True

            if time.monotonic() >= completion_deadline:
                print(
                    f"[gripper][warn] close-limit command was sent, but measured width "
                    f"remains {measured_width:.4f}m > {fully_closed_width:.4f}m without "
                    "confirmed high-force contact before timeout; stop and recover"
                )
                return GripperCloseResult(
                    last_status,
                    False,
                    "close_target_not_reached_timeout",
                    contact_width=measured_width,
                    contact_force=measured_force,
                )
            time.sleep(close_completion_poll_s)

    def print_gripper_status(self):
        if self.gripper is None:
            return None
        try:
            status = self.gripper.get_gripper_status()
            msg = None if status is None else status.msg
            if msg is not None:
                print(f"[gripper] status value={msg.value} force={msg.force} mode={msg.mode}")
            return msg
        except Exception as exc:
            print(f"[gripper] failed to read status: {type(exc).__name__}: {exc}")
            return None
