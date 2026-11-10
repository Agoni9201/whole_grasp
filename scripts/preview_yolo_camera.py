#!/usr/bin/env python3
"""Compare target segmentation and optional scene detections on SDK RGB."""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
from ultralytics import YOLO

from cameras.realsense import create_camera


def detections(result, target_class=None):
    if result.boxes is None:
        return []
    return [
        (str(result.names[int(box.cls.item())]), round(float(box.conf.item()), 3))
        for box in result.boxes
        if target_class is None or str(result.names[int(box.cls.item())]) == target_class
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--target-class", required=True)
    parser.add_argument("--scene-weights", type=Path,
                        help="Optional general YOLO model for labels and scores of other objects")
    parser.add_argument("--serial", default=None)
    parser.add_argument("--backend", choices=("realsense", "orbbec"), default="realsense")
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    args = parser.parse_args()
    if not args.weights.is_file():
        parser.error(f"YOLO weights not found: {args.weights}")
    if args.scene_weights is not None and not args.scene_weights.is_file():
        parser.error(f"scene YOLO weights not found: {args.scene_weights}")
    if min(args.width, args.height, args.fps, args.imgsz) <= 0 or not 0 < args.conf < 1:
        parser.error("invalid camera or YOLO settings")

    model = YOLO(str(args.weights))
    names = [str(name) for name in model.names.values()]
    if model.task != "segment":
        parser.error(f"weights are for {model.task}, not instance segmentation")
    if args.target_class not in names:
        parser.error(f"class {args.target_class!r} is absent; available classes: {names}")
    scene_model = YOLO(str(args.scene_weights)) if args.scene_weights else None
    print(f"Target-model classes: {names}")
    if scene_model:
        print(f"Scene-model classes: {list(scene_model.names.values())}")
    print("Only RGB camera and YOLO are active. Q/Esc exits; no arm or CAN connection.")

    target_window = "TARGET: segmentation used by grasp (Q/Esc exit)"
    scene_window = "SCENE: general YOLO labels + confidence (Q/Esc exit)"
    cv2.namedWindow(target_window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(target_window, 960, 540)
    if scene_model:
        cv2.namedWindow(scene_window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(scene_window, 960, 540)

    last_log = 0.0
    try:
        with create_camera(backend=args.backend,
            serial=args.serial, width=args.width, height=args.height,
            fps=args.fps, enable_depth=False,
        ) as camera:
            while True:
                frame = camera.read()
                target_result = model.predict(
                    frame.bgr, imgsz=args.imgsz, conf=args.conf, verbose=False
                )[0]
                target_hits = detections(target_result, args.target_class)
                scene_result = (
                    scene_model.predict(frame.bgr, imgsz=args.imgsz,
                                        conf=args.conf, verbose=False)[0]
                    if scene_model else None
                )
                if time.monotonic() - last_log >= 1.0:
                    print(f"target detections: {target_hits}", flush=True)
                    print(f"all detections: {detections(target_result)}", flush=True)
                    if scene_result is not None:
                        print(f"scene detections: {detections(scene_result)}", flush=True)
                    last_log = time.monotonic()
                target_image = target_result.plot(conf=True, labels=True)
                status = (
                    f"TARGET {args.target_class}: {len(target_hits)}"
                    if target_hits else f"NO TARGET {args.target_class} >= {args.conf:.2f}"
                )
                cv2.putText(target_image, status, (16, 38), cv2.FONT_HERSHEY_SIMPLEX,
                            0.9, (0, 180, 0) if target_hits else (0, 0, 255), 2)
                cv2.imshow(target_window, target_image)
                if scene_result is not None:
                    cv2.imshow(scene_window,
                               scene_result.plot(conf=True, labels=True, masks=False))
                if (cv2.waitKey(1) & 255) in (27, ord("q"), ord("Q")):
                    break
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
