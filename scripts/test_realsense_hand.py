#!/usr/bin/env python3
"""RealSense/Orbbec SDK + MediaPipe human-hand 3D preview (no ROS/CAN)."""
from __future__ import annotations

import argparse
from collections import deque
from datetime import datetime
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cameras.realsense import create_camera, depth_with_current_support  # noqa: E402
from targets.mediapipe_hand_landmarker import (  # noqa: E402
    DEFAULT_MODEL_PATH,
    MediaPipeHandLandmarker,
)


HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
)


def draw_observation(image: np.ndarray, observation, smoothed, jitter_mm: float) -> None:
    for start, end in HAND_CONNECTIONS:
        cv2.line(
            image,
            tuple(observation.landmarks_pixel[start]),
            tuple(observation.landmarks_pixel[end]),
            (40, 220, 40), 2, cv2.LINE_AA,
        )
    for pixel in observation.landmarks_pixel:
        cv2.circle(image, tuple(pixel), 3, (20, 80, 255), -1, cv2.LINE_AA)
    cv2.circle(image, observation.palm_pixel, 7, (255, 180, 0), 2, cv2.LINE_AA)

    score = observation.handedness_score
    score_text = f" {score:.2f}" if np.isfinite(score) else ""
    cv2.putText(
        image, f"{observation.handedness}{score_text}",
        (observation.palm_pixel[0] + 10, observation.palm_pixel[1] - 12),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 180, 0), 2, cv2.LINE_AA,
    )
    if smoothed is not None:
        x, y, z = np.asarray(smoothed) * 1000.0
        cv2.putText(
            image, f"camera XYZ: {x:+.0f} {y:+.0f} {z:.0f} mm",
            (14, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 255, 255), 2, cv2.LINE_AA,
        )
        cv2.putText(
            image,
            f"depth points: {observation.valid_depth_landmarks}/5  jitter(30): {jitter_mm:.1f} mm",
            (14, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA,
        )
    else:
        cv2.putText(
            image, f"RGB hand found; depth invalid: {observation.valid_depth_landmarks}/5",
            (14, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 120, 255), 2, cv2.LINE_AA,
        )


def observation_status(observation, min_depth_landmarks=3) -> str:
    if observation is None:
        return "no_hand_2d"
    if (observation.palm_center_camera_m is None
            or observation.valid_depth_landmarks < min_depth_landmarks):
        return "invalid_palm_depth"
    return "ok"


def save_snapshot(root: Path, frame, overlay: np.ndarray, observation, detection_depth=None,
                  *, min_depth_landmarks=3, capture_settings=None) -> Path:
    output = root / datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")
    output.mkdir(parents=True, exist_ok=False)
    if not cv2.imwrite(str(output / "color.png"), frame.bgr):
        raise OSError("failed to save color.png")
    if not cv2.imwrite(str(output / "overlay.png"), overlay):
        raise OSError("failed to save overlay.png")
    np.save(output / "depth_m.npy", frame.depth_m if detection_depth is None else detection_depth)
    np.save(output / "depth_filtered_m.npy", frame.depth_m)
    if frame.raw_depth_m is not None:
        np.save(output / "depth_raw_m.npy", frame.raw_depth_m)
    metadata = {
        "coordinate_frame": "aligned_color_optical_frame",
        "length_unit": "metre",
        "camera_serial": frame.serial,
        "camera_backend": getattr(frame, "backend", "unknown"),
        "hardware_ms": getattr(frame, "hardware_ms", None),
        "received_monotonic": getattr(frame, "received_monotonic", None),
        "color_shape": list(frame.bgr.shape),
        "capture_settings": capture_settings,
        "reason": observation_status(observation, min_depth_landmarks),
        "hand_detected_2d": observation is not None,
        "metric_position_usable": observation_status(observation, min_depth_landmarks) == "ok",
        "required_depth_landmarks": min_depth_landmarks,
        "K": frame.K.tolist(),
        "D": frame.D.tolist(),
        "sequence": int(frame.sequence),
        "palm_pixel": None if observation is None else list(observation.palm_pixel),
        "landmarks_normalized": (
            None if observation is None or not hasattr(observation, "landmarks_normalized")
            else observation.landmarks_normalized.tolist()
        ),
        "landmarks_pixel": (
            None if observation is None or not hasattr(observation, "landmarks_pixel")
            else observation.landmarks_pixel.tolist()
        ),
        "palm_center_camera_m": (
            None if observation is None or observation.palm_center_camera_m is None
            else observation.palm_center_camera_m.tolist()
        ),
        "palm_normal_camera": (
            None if observation is None or observation.palm_normal_camera is None
            else observation.palm_normal_camera.tolist()
        ),
        "palm_tangent_camera": (
            None if observation is None or observation.palm_tangent_camera is None
            else observation.palm_tangent_camera.tolist()
        ),
        "handedness": None if observation is None else observation.handedness,
        "handedness_score": (None if observation is None or not np.isfinite(observation.handedness_score)
                             else observation.handedness_score),
        "valid_depth_landmarks": None if observation is None else observation.valid_depth_landmarks,
        "palm_depth_m": (None if observation is None or not np.isfinite(observation.palm_depth_m)
                         else observation.palm_depth_m),
        "palm_depth_mad_m": (None if observation is None or not np.isfinite(observation.palm_depth_mad_m)
                             else observation.palm_depth_mad_m),
        "raw_current_support": frame.raw_depth_m is not None,
    }
    (output / "hand_pose.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8"
    )
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--serial")
    parser.add_argument("--backend", choices=("realsense", "orbbec"), default="realsense")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--depth-width", type=int)
    parser.add_argument("--depth-height", type=int)
    parser.add_argument("--depth-fps", type=int)
    parser.add_argument("--depth-filters", choices=("off", "all"), default="off")
    parser.add_argument("--show-depth", action="store_true", help="Show raw-supported aligned depth beside RGB; black means missing/out of range")
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--num-hands", type=int, default=1)
    parser.add_argument("--min-detection-confidence", type=float, default=0.5)
    parser.add_argument("--min-presence-confidence", type=float, default=0.5)
    parser.add_argument("--min-tracking-confidence", type=float, default=0.5)
    parser.add_argument("--depth-radius", type=int, default=4)
    parser.add_argument("--min-depth-landmarks", type=int, choices=range(1, 6), default=3,
                        help="Required valid palm landmarks, matching the servo default of 3/5")
    parser.add_argument("--min-depth", type=float, default=0.10)
    parser.add_argument("--max-depth", type=float, default=2.0)
    parser.add_argument("--smoothing-alpha", type=float, default=0.25)
    parser.add_argument("--save-dir", type=Path, default=ROOT / "captures" / "hand_landmarker")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--max-frames", type=int, default=0, help="0 means run until Q/Esc")
    args = parser.parse_args()
    confidences = (
        args.min_detection_confidence,
        args.min_presence_confidence,
        args.min_tracking_confidence,
    )
    if min(args.width, args.height, args.fps) <= 0 or args.warmup < 0 or args.timeout <= 0:
        parser.error("invalid camera settings")
    if any(value is not None and value <= 0 for value in (args.depth_width, args.depth_height, args.depth_fps)):
        parser.error("native depth dimensions and FPS must be positive")
    if args.num_hands < 1 or args.depth_radius < 0 or not 0.0 < args.min_depth < args.max_depth:
        parser.error("invalid hand/depth settings")
    if not all(0.0 <= value <= 1.0 for value in confidences):
        parser.error("confidence thresholds must be in [0, 1]")
    if not 0.0 < args.smoothing_alpha <= 1.0 or args.max_frames < 0:
        parser.error("invalid smoothing-alpha or max-frames")
    return args


def main() -> int:
    args = parse_args()
    camera = create_camera(backend=args.backend,
        serial=args.serial, width=args.width, height=args.height, fps=args.fps,
        timeout=args.timeout, warmup=args.warmup,
        depth_width=args.depth_width, depth_height=args.depth_height, depth_fps=args.depth_fps,
        depth_filters=args.depth_filters, emitter_enabled=True if args.backend == 'realsense' else None,
    )
    positions: deque[np.ndarray] = deque(maxlen=30)
    hits: deque[bool] = deque(maxlen=90)
    smoothed = None
    frame_count = 0
    last_report = 0.0
    print(f"{args.backend} SDK + MediaPipe human hand test; no ROS and no robot commands.")
    print("Show an open palm to the camera. Q/Esc exits; S saves RGB-D even if no hand/depth is detected.")
    capture_settings = {key: str(value) if isinstance(value, Path) else value
                        for key, value in vars(args).items()}
    try:
        with camera, MediaPipeHandLandmarker(
            model_path=args.model,
            num_hands=args.num_hands,
            min_detection_confidence=args.min_detection_confidence,
            min_presence_confidence=args.min_presence_confidence,
            min_tracking_confidence=args.min_tracking_confidence,
            depth_radius=args.depth_radius,
            min_depth_m=args.min_depth,
            max_depth_m=args.max_depth,
        ) as detector:
            print(f"camera serial={camera.serial}, profile={args.width}x{args.height}@{args.fps}")
            print(f"model={Path(args.model).expanduser().resolve()}")
            if not args.headless:
                cv2.namedWindow("RGB-D + MediaPipe hand (S save, Q/Esc exit)", cv2.WINDOW_NORMAL)
                cv2.resizeWindow("RGB-D + MediaPipe hand (S save, Q/Esc exit)", 1280, 480)
            while True:
                frame = camera.read()
                frame_count += 1
                depth = depth_with_current_support(frame.depth_m, frame.raw_depth_m, args.min_depth, args.max_depth)
                observations = detector.detect(
                    frame.bgr, depth, frame.K, int(round(frame.hardware_ms))
                )
                observation = observations[0] if observations else None
                reason = observation_status(observation, args.min_depth_landmarks)
                valid_position = reason == "ok"
                hits.append(bool(valid_position))
                if valid_position:
                    raw = np.asarray(observation.palm_center_camera_m, dtype=np.float64)
                    smoothed = raw.copy() if smoothed is None else (
                        args.smoothing_alpha * raw + (1.0 - args.smoothing_alpha) * smoothed
                    )
                    positions.append(raw)
                else:
                    positions.clear()
                    smoothed = None

                jitter_mm = 0.0
                if len(positions) >= 5:
                    centered = np.asarray(positions) - np.mean(positions, axis=0)
                    jitter_mm = float(np.sqrt(np.mean(np.sum(centered * centered, axis=1))) * 1000.0)
                overlay = frame.bgr.copy()
                hit_rate = 100.0 * sum(hits) / max(1, len(hits))
                cv2.putText(
                    overlay, f"3D detection rate: {hit_rate:.0f}% ({len(hits)} frames)",
                    (14, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA,
                )
                if observation is not None:
                    draw_observation(overlay, observation, smoothed if valid_position else None, jitter_mm)
                else:
                    cv2.putText(overlay, "RGB hand not detected; S saves this frame too",
                                (14, 62), cv2.FONT_HERSHEY_SIMPLEX, .62, (0, 120, 255), 2)
                if args.show_depth:
                    valid = np.isfinite(depth) & (depth >= args.min_depth) & (depth <= args.max_depth)
                    scaled = np.zeros(depth.shape, dtype=np.uint8)
                    scaled[valid] = np.clip(depth[valid] / args.max_depth * 255, 0, 255).astype(np.uint8)
                    depth_image = cv2.applyColorMap(scaled, cv2.COLORMAP_TURBO)
                    depth_image[~valid] = 0
                    if observation is not None:
                        for index in (0, 5, 9, 13, 17):
                            cv2.circle(depth_image, tuple(observation.landmarks_pixel[index]), args.depth_radius, (0, 255, 0), 2)
                    cv2.putText(depth_image, "Current depth: black=missing/out of range; S saves invalid-depth hands too",
                                (14, 30), cv2.FONT_HERSHEY_SIMPLEX, .65, (255, 255, 255), 2)
                    overlay = np.hstack((overlay, depth_image))

                now = time.monotonic()
                if now - last_report >= 0.5:
                    last_report = now
                    if valid_position:
                        xyz_mm = np.asarray(smoothed) * 1000.0
                        print(
                            f"\rhand={observation.handedness:<7} "
                            f"XYZ_mm=[{xyz_mm[0]:+7.1f}, {xyz_mm[1]:+7.1f}, {xyz_mm[2]:7.1f}] "
                            f"depth={observation.valid_depth_landmarks}/5 "
                            f"hit={hit_rate:5.1f}% jitter={jitter_mm:5.1f}mm",
                            end="", flush=True,
                        )
                    else:
                        print(f"\rreason={reason}; hit={hit_rate:5.1f}%", end="", flush=True)

                key = -1
                if not args.headless:
                    cv2.imshow("RGB-D + MediaPipe hand (S save, Q/Esc exit)", overlay)
                    key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q"), ord("Q")):
                    break
                if key in (ord("s"), ord("S")):
                    saved = save_snapshot(args.save_dir.expanduser(), frame, overlay, observation, depth,
                                          min_depth_landmarks=args.min_depth_landmarks,
                                          capture_settings=capture_settings)
                    print(f"\nSaved ({reason}): {saved}")
                if args.max_frames and frame_count >= args.max_frames:
                    break
    except KeyboardInterrupt:
        pass
    finally:
        camera.close()
        cv2.destroyAllWindows()
    print(f"\nFinished after {frame_count} frames.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

