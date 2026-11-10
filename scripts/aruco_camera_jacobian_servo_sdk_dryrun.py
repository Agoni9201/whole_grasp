#!/usr/bin/env python3
"""Camera ArUco + SDK-feedback Panthera PBVS dry-run.

This replaces the old two-node recognition path for dry-run:

    aruco_ros single
    real_visual_servo/aruco_pose_from_eye_in_hand.py

with one project-local node:

    camera image + camera_info
    -> OpenCV ArUco marker pose in camera frame
    -> SDK read-only joint feedback
    -> Pinocchio FK + verified hand-eye transform
    -> marker center / normal / tangent in base frame
    -> Panthera PBVS velocity-Jacobian servo
    -> unified backend command log

It does not enable the arm, does not switch motion mode, and does not send
CPV / MIT / move_j commands.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(SCRIPT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from cameras import sdk_loop  # noqa: E402
import cv2  # noqa: E402
from cameras.sdk_loop import Node  # noqa: E402
from cameras.sdk_loop import CameraInfo, Image  # noqa: E402

from aruco_jacobian_servo_sdk_dryrun import SdkJointFeedbackReader  # noqa: E402
from control_backends.base import BackendCommand  # noqa: E402
from control_backends.sdk_cpv_velocity_backend import SdkCpvVelocityBackend  # noqa: E402
from servo.jacobian_position_servo import JacobianPositionServo, ServoParams  # noqa: E402
from servo.pbvs_pose_servo import PBVSPoseServo, PBVSParams  # noqa: E402
from targets.aruco_eye_in_hand_target import ArUcoEyeInHandTarget  # noqa: E402


DEFAULT_HAND_EYE = (
    "/home/dpf/桌面/RealRobot_Test/real_test/configs/"
    "2026-05-19_14-45-49_calibration.json"
)


def _format(values, precision: int = 4) -> str:
    return "[" + ", ".join(f"{float(v):.{precision}f}" for v in values) + "]"


def _vec7(text: str) -> np.ndarray:
    values = [float(item) for item in text.split(",")]
    if len(values) != 7:
        raise argparse.ArgumentTypeError("expected seven comma-separated floats")
    return np.asarray(values, dtype=np.float64)


def image_msg_to_bgr(msg: Image) -> np.ndarray:
    if msg.width <= 0 or msg.height <= 0 or len(msg.data) == 0:
        raise ValueError(
            f"empty image msg: encoding={msg.encoding!r} "
            f"width={msg.width} height={msg.height} step={msg.step} data_len={len(msg.data)}"
        )

    encoding = msg.encoding.lower()
    if encoding in {"bgr8", "rgb8"}:
        channels = 3
        dtype = np.uint8
    elif encoding in {"bgra8", "rgba8"}:
        channels = 4
        dtype = np.uint8
    elif encoding in {"mono8", "8uc1"}:
        channels = 1
        dtype = np.uint8
    elif encoding == "8uc3":
        channels = 3
        dtype = np.uint8
    else:
        raise ValueError(f"unsupported image encoding: {msg.encoding!r}")

    expected_min_step = msg.width * channels * np.dtype(dtype).itemsize
    if msg.step < expected_min_step:
        raise ValueError(
            f"invalid image step: encoding={msg.encoding!r} width={msg.width} "
            f"height={msg.height} step={msg.step} expected_min_step={expected_min_step}"
        )

    raw = np.frombuffer(msg.data, dtype=dtype)
    expected_total = msg.step * msg.height
    if raw.nbytes < expected_total:
        raise ValueError(
            f"image data too short: encoding={msg.encoding!r} "
            f"data_bytes={raw.nbytes} expected_bytes={expected_total}"
        )

    rows = raw[: expected_total].reshape(msg.height, msg.step)
    packed = rows[:, :expected_min_step].reshape(msg.height, msg.width, channels)

    if encoding in {"bgr8", "8uc3"}:
        return packed.copy()
    if encoding == "rgb8":
        return packed[:, :, ::-1].copy()
    if encoding == "bgra8":
        return packed[:, :, :3].copy()
    if encoding == "rgba8":
        return packed[:, :, [2, 1, 0]].copy()
    return np.repeat(packed, 3, axis=2).copy()


def _put_overlay_text(
    image: np.ndarray,
    text: str,
    line: int,
    color: tuple[int, int, int] = (255, 255, 255),
) -> None:
    y = 28 + line * 24
    cv2.putText(
        image,
        text,
        (10, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (0, 0, 0),
        3,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        text,
        (10, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        color,
        1,
        cv2.LINE_AA,
    )


def draw_camera_overlay(
    image: np.ndarray,
    target,
    servo_out: Optional[dict],
    status: str,
    camera_matrix: Optional[np.ndarray],
    dist_coeffs: Optional[np.ndarray],
    marker_size: float,
) -> np.ndarray:
    """Draw ArUco detection and servo status on the camera image."""
    display = image.copy()
    status_color = (0, 220, 0) if target is not None else (0, 160, 255)
    _put_overlay_text(display, status, 0, status_color)

    if target is None:
        _put_overlay_text(display, "No target marker", 1, (0, 160, 255))
    else:
        corners = np.asarray(target.corners, dtype=np.float32).reshape(1, 4, 2)
        cv2.aruco.drawDetectedMarkers(
            display,
            [corners],
            np.array([[target.marker_id]], dtype=np.int32),
        )
        if camera_matrix is not None and dist_coeffs is not None:
            rvec, _ = cv2.Rodrigues(target.T_camera_marker[:3, :3])
            tvec = target.T_camera_marker[:3, 3].reshape(3, 1)
            cv2.drawFrameAxes(
                display,
                np.asarray(camera_matrix, dtype=np.float64).reshape(3, 3),
                np.asarray(dist_coeffs, dtype=np.float64).ravel(),
                rvec,
                tvec,
                float(marker_size) * 0.75,
            )

        center_px = corners.reshape(4, 2).mean(axis=0).astype(int)
        cv2.circle(display, tuple(center_px), 5, (0, 255, 255), -1)
        _put_overlay_text(
            display,
            f"{target.dictionary_name} id={target.marker_id}",
            1,
            (0, 255, 0),
        )
        _put_overlay_text(
            display,
            f"marker_cam={_format(target.T_camera_marker[:3, 3], 3)}",
            2,
            (255, 255, 0),
        )
        _put_overlay_text(
            display,
            f"marker_base={_format(target.marker_center_base, 3)}",
            3,
            (255, 255, 0),
        )

    if servo_out is not None:
        err = float(np.linalg.norm(servo_out["error"]))
        rot_err = float(np.degrees(np.linalg.norm(servo_out.get("orientation_error", np.zeros(3)))))
        qdot_max = float(np.max(np.abs(servo_out["qdot"])))
        _put_overlay_text(
            display,
            f"err={err:.4f}m rot={rot_err:.2f}deg qdot_max={qdot_max:.4f}rad/s",
            4,
            (255, 255, 255),
        )
        target_tcp = servo_out.get("target_tcp")
        if target_tcp is not None:
            _put_overlay_text(
                display,
                f"target_tcp={_format(target_tcp, 3)}",
                5,
                (255, 255, 255),
            )
    return display


class ArucoCameraServoDryRunNode(Node):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__("nero_aruco_camera_jacobian_servo_sdk_dryrun")
        self.args = args
        self.last_image = None
        self.last_image_time: Optional[float] = None
        self.camera_matrix: Optional[np.ndarray] = None
        self.dist_coeffs: Optional[np.ndarray] = None
        self.last_log_time = 0.0
        self.last_image_warn_time = 0.0
        self.last_detect_time: Optional[float] = None
        self.count = 0
        self.started = time.monotonic()
        self.display_enabled = bool(args.display)
        self.display_initialized = False
        self.display_window = "Nero ArUco Servo Dry-Run"

        self.reader = SdkJointFeedbackReader(
            channel=args.channel,
            interface=args.interface,
            firmware=args.firmware,
        )
        self.backend = SdkCpvVelocityBackend(
            dry_run=True,
            channel=args.channel,
            max_joint_vel=args.max_joint_vel,
            velocity_signs=args.cpv_signs,
        )
        self.backend.connect()
        self.reader.connect()

        self.target_source = ArUcoEyeInHandTarget(
            handeye_path=args.handeye,
            aruco_dict=args.aruco_dict,
            marker_id=args.marker_id,
            marker_size=args.marker_size,
            handeye_convention=args.handeye_convention,
            handeye_mount_frame=args.handeye_mount_frame,
            offset_base=args.offset_base,
            offset_marker=args.offset_marker,
        )

        if args.servo_mode == "pbvs":
            params = PBVSParams(
                servo_hz=args.servo_hz,
                desired_distance=args.desired_distance,
                linear_gain=args.linear_gain,
                linear_integral_gain=args.linear_integral_gain,
                linear_derivative_gain=args.linear_derivative_gain,
                angular_gain=args.angular_gain,
                max_linear_vel=args.max_linear_vel,
                max_angular_vel=args.max_angular_vel,
                damping=args.damping,
                max_joint_vel=args.max_joint_vel,
                max_joint_step=args.max_joint_step,
                marker_center_alpha=args.target_filter_alpha,
                marker_normal_alpha=args.normal_filter_alpha,
                marker_tangent_alpha=args.normal_filter_alpha,
                tcp_twist_smooth_alpha=args.twist_smooth_alpha,
                target_timeout_s=args.target_timeout_s,
                max_target_jump=args.max_target_jump,
                position_deadband=args.position_deadband,
                orientation_deadband_rad=np.deg2rad(args.orientation_deadband_deg),
            )
            self.servo = PBVSPoseServo(params=params)
        else:
            params = ServoParams(
                servo_hz=args.servo_hz,
                desired_distance=args.desired_distance,
                linear_gain=args.linear_gain,
                max_linear_vel=args.max_linear_vel,
                damping=args.damping,
                max_joint_vel=args.max_joint_vel,
                max_joint_step=args.max_joint_step,
                target_filter_alpha=args.target_filter_alpha,
                normal_filter_alpha=args.normal_filter_alpha,
                qdot_filter_alpha=args.qdot_filter_alpha,
                target_timeout_s=args.target_timeout_s,
                max_target_jump=args.max_target_jump,
                position_deadband=args.position_deadband,
            )
            self.servo = JacobianPositionServo(params=params)
        self.params = params
        self.create_subscription(Image, args.image_topic, self.on_image, 5)
        self.create_subscription(CameraInfo, args.camera_info_topic, self.on_camera_info, 5)
        self.create_timer(1.0 / float(args.servo_hz), self.on_timer)

        self.get_logger().info("DRY-RUN only: no real robot commands will be sent.")
        self.get_logger().info(f"image_topic={args.image_topic}")
        self.get_logger().info(f"camera_info_topic={args.camera_info_topic}")
        self.get_logger().info(
            f"marker_id={args.marker_id} marker_size={args.marker_size:.4f}m "
            f"aruco_dict={args.aruco_dict}"
        )
        self.get_logger().info(f"handeye={args.handeye}")
        self.get_logger().info(
            f"handeye_mount_frame={args.handeye_mount_frame} "
            f"offset_base={_format(args.offset_base, 4)}"
        )

    def on_image(self, msg: Image) -> None:
        try:
            self.last_image = image_msg_to_bgr(msg)
            self.last_image_time = time.monotonic()
        except Exception as exc:
            now = time.monotonic()
            if now - self.last_image_warn_time >= 1.0:
                self.get_logger().warning(f"image conversion failed: {exc}")
                self.last_image_warn_time = now

    def on_camera_info(self, msg: CameraInfo) -> None:
        self.camera_matrix = np.asarray(msg.k, dtype=np.float64).reshape(3, 3)
        self.dist_coeffs = np.asarray(msg.d, dtype=np.float64)
        if len(self.dist_coeffs) == 0:
            self.dist_coeffs = np.zeros(5, dtype=np.float64)

    def on_timer(self) -> None:
        now = time.monotonic()
        if self.args.duration > 0.0 and now - self.started >= self.args.duration:
            self.get_logger().info(f"duration reached: {self.args.duration:.1f}s")
            sdk_loop.shutdown()
            return

        q_now = self.reader.read_joint_angles()
        if q_now is None:
            self._update_display("waiting for SDK joint feedback", None, None)
            self._log_waiting(now, "waiting for SDK joint feedback")
            return
        if self.last_image is None:
            self._log_waiting(now, "waiting for camera image")
            return
        if self.camera_matrix is None or self.dist_coeffs is None:
            self._update_display("waiting for camera_info", None, None)
            self._log_waiting(now, "waiting for camera_info")
            return

        target = self.target_source.detect(
            image=self.last_image,
            camera_matrix=self.camera_matrix,
            dist_coeffs=self.dist_coeffs,
            q_arm=q_now,
        )
        if target is None:
            servo_out = self.servo.update(q_now=q_now, target_position=None, target_normal=None, stamp=now)
            status = "marker_lost_hold"
        else:
            self.last_detect_time = now
            if self.args.servo_mode == "pbvs":
                servo_out = self.servo.update(
                    q_now=q_now,
                    target_position=target.marker_center_base,
                    target_normal=target.marker_normal_base,
                    target_tangent=target.marker_tangent_base,
                    stamp=now,
                )
            else:
                servo_out = self.servo.update(
                    q_now=q_now,
                    target_position=target.marker_center_base,
                    target_normal=target.marker_normal_base,
                    stamp=now,
                )
            if servo_out["target_rejected"]:
                status = "target_rejected_hold"
            elif servo_out["target_lost"]:
                status = "marker_lost_hold"
            else:
                status = "tracking"

        cmd = BackendCommand(
            q_cmd=q_now.copy(),
            qdot_cmd=servo_out["qdot"],
            mode="velocity",
            stamp=now,
        )
        self.backend.send(cmd)
        self.count += 1
        self._update_display(status, target, servo_out)

        if now - self.last_log_time >= self.args.log_period:
            self._log_status(status, q_now, target, servo_out, now)
            self.last_log_time = now

    def _log_waiting(self, now: float, text: str) -> None:
        if now - self.last_log_time >= self.args.log_period:
            self.get_logger().info(text)
            self.last_log_time = now

    def _log_status(self, status: str, q_now: np.ndarray, target, servo_out: dict, now: float) -> None:
        q_step = servo_out["q_cmd"] - q_now
        detected_age = None if self.last_detect_time is None else now - self.last_detect_time
        age_text = "none" if detected_age is None else f"{detected_age:.3f}s"
        target_text = "None"
        normal_text = "None"
        camera_text = "None"
        dict_text = "None"
        if target is not None:
            target_text = _format(target.marker_center_base, 4)
            normal_text = _format(target.marker_normal_base, 4)
            camera_text = _format(target.T_camera_marker[:3, 3], 4)
            dict_text = target.dictionary_name

        self.get_logger().info(
            f"{status} count={self.count} detect_age={age_text} "
            f"err={np.linalg.norm(servo_out['error']):.4f}m "
            f"rot_err={np.degrees(np.linalg.norm(servo_out.get('orientation_error', np.zeros(3)))):.2f}deg "
            f"qdot_max={np.max(np.abs(servo_out['qdot'])):.5f}rad/s "
            f"step_max={np.max(np.abs(q_step)):.5f}rad\n"
            f"  dict={dict_text} marker_camera={camera_text} marker_base={target_text} normal={normal_text}\n"
            f"  tcp_now={_format(servo_out['tcp_now'], 4)} target_tcp={_format_or_none(servo_out['target_tcp'])}\n"
            f"  q_now_deg={_format(np.degrees(q_now), 2)} "
            f"q_cmd_deg={_format(np.degrees(servo_out['q_cmd']), 2)}"
        )

    def _update_display(self, status: str, target, servo_out: Optional[dict]) -> None:
        if not self.display_enabled or self.last_image is None:
            return
        try:
            display = draw_camera_overlay(
                image=self.last_image,
                target=target,
                servo_out=servo_out,
                status=status,
                camera_matrix=self.camera_matrix,
                dist_coeffs=self.dist_coeffs,
                marker_size=self.args.marker_size,
            )
            if not self.display_initialized:
                cv2.namedWindow(self.display_window, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(self.display_window, 960, 540)
                self.display_initialized = True
            cv2.imshow(self.display_window, display)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                self.get_logger().info("display requested shutdown")
                sdk_loop.shutdown()
        except Exception as exc:
            self.display_enabled = False
            self.get_logger().warning(f"camera display disabled: {exc}")

    def close(self) -> None:
        self.reader.disconnect()
        self.backend.disconnect()
        if self.display_initialized:
            cv2.destroyWindow(self.display_window)


def _format_or_none(values) -> str:
    if values is None:
        return "None"
    return _format(values, 4)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-topic", default="/camera/camera/color/image_raw")
    parser.add_argument("--camera-info-topic", default="/camera/camera/color/camera_info")
    parser.add_argument("--handeye", default=DEFAULT_HAND_EYE)
    parser.add_argument(
        "--handeye-convention",
        choices=("link7_cam", "cam_link7", "tcp_cam", "cam_tcp", "mount_cam", "cam_mount"),
        default="link7_cam",
    )
    parser.add_argument(
        "--handeye-mount-frame",
        choices=("feedback_link7", "tcp_offset"),
        default="feedback_link7",
    )
    parser.add_argument("--offset-base", nargs=3, type=float, default=[0.02, -0.025, 0.0])
    parser.add_argument("--offset-marker", nargs=3, type=float, default=[0.0, 0.0, 0.0])
    parser.add_argument(
        "--aruco-dict",
        default="ARUCO_ROS_ALL",
        help="Use ARUCO_ROS_ALL to try common OpenCV dictionaries, then filter by marker_id.",
    )
    parser.add_argument("--marker-id", type=int, default=582)
    parser.add_argument("--marker-size", type=float, default=0.0677)

    parser.add_argument("--channel", default="can0")
    parser.add_argument("--interface", default="socketcan")
    parser.add_argument("--firmware", default="v111")
    parser.add_argument(
        "--cpv-signs",
        type=_vec7,
        default=np.ones(7, dtype=np.float64),
        help="Per-joint sign multiplier before SDK move_cpv_vel(), comma-separated 7 values.",
    )
    parser.add_argument("--duration", type=float, default=0.0, help="Seconds to run; 0 means forever.")
    parser.add_argument("--log-period", type=float, default=0.5)
    parser.add_argument("--verbose-backend", action="store_true")
    parser.add_argument("--no-display", dest="display", action="store_false")
    parser.set_defaults(display=True)

    parser.add_argument("--servo-mode", choices=("position", "pbvs"), default="pbvs")
    parser.add_argument("--servo-hz", type=float, default=30.0)
    parser.add_argument("--desired-distance", type=float, default=0.15)
    parser.add_argument("--linear-gain", type=float, default=5.0)
    parser.add_argument("--linear-integral-gain", type=float, default=1.2)
    parser.add_argument("--linear-derivative-gain", type=float, default=0.08)
    parser.add_argument("--angular-gain", type=float, default=2.6)
    parser.add_argument("--max-linear-vel", type=float, default=0.35)
    parser.add_argument("--max-angular-vel", type=float, default=2.20)
    parser.add_argument("--damping", type=float, default=0.02)
    parser.add_argument("--max-joint-vel", type=float, default=0.15)
    parser.add_argument("--max-joint-step", type=float, default=0.006)
    parser.add_argument("--target-filter-alpha", type=float, default=0.34)
    parser.add_argument("--normal-filter-alpha", type=float, default=0.24)
    parser.add_argument("--qdot-filter-alpha", type=float, default=0.35)
    parser.add_argument("--twist-smooth-alpha", type=float, default=0.72)
    parser.add_argument("--target-timeout-s", type=float, default=0.60)
    parser.add_argument("--max-target-jump", type=float, default=0.08)
    parser.add_argument("--position-deadband", type=float, default=0.004)
    parser.add_argument("--orientation-deadband-deg", type=float, default=1.5)
    parser.add_argument("--kp-scale", type=float, default=1.0)
    parser.add_argument("--kd-scale", type=float, default=1.0)
    from cameras.options import add_camera_arguments
    add_camera_arguments(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO if args.verbose_backend else logging.WARNING,
        format="%(levelname)s: %(message)s",
    )
    if not Path(args.handeye).expanduser().exists():
        raise FileNotFoundError(args.handeye)

    sdk_loop.init()
    node = ArucoCameraServoDryRunNode(args)
    try:
        sdk_loop.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("interrupted")
    finally:
        node.close()
        node.destroy_node()
        if sdk_loop.ok():
            sdk_loop.shutdown()


if __name__ == "__main__":
    main()
