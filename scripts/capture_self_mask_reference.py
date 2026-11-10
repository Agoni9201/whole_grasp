#!/usr/bin/env python3
"""Capture independent RGB-D references for a manually prepared empty gripper.

Camera only. Does not connect CAN, measure/change opening, move/enable the arm,
build a mask, or overwrite any calibration/configuration. Opening is explicitly
an operator declaration until separately verified from gripper feedback.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cameras.options import add_camera_arguments, apply_pipeline_defaults
from cameras.realsense import camera_for, close_all


def save_reference(directory, frame, args, opening_mm):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    if not cv2.imwrite(str(directory/'color.png'), frame.bgr):
        raise OSError('cannot save self-mask reference RGB')
    np.save(directory/'depth_m.npy', frame.depth_m)
    if frame.raw_depth_m is not None:
        np.save(directory/'depth_raw_m.npy', frame.raw_depth_m)
    metadata = dict(backend=frame.backend, serial=frame.serial,
        width=frame.bgr.shape[1], height=frame.bgr.shape[0],
        native_depth_width=args.camera_depth_width or args.camera_width,
        native_depth_height=args.camera_depth_height or args.camera_height,
        intrinsics={key: float(getattr(frame.intr, key)) for key in ('fx', 'fy', 'cx', 'cy')},
        K=frame.K.tolist(), D=frame.D.tolist(), sequence=frame.sequence,
        hardware_ms=frame.hardware_ms, received_monotonic=frame.received_monotonic,
        captured_wall_time=time.time(), depth_unit='metre',
        opening_declared_mm=opening_mm, opening_verified_by_this_tool=False,
        arm_or_gripper_commands_sent=False)
    (directory/'capture_meta.json').write_text(json.dumps(metadata, indent=2)+'\n')
    return metadata


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    add_camera_arguments(p)
    p.add_argument('--opening-mm', type=float, required=True,
                   help='Operator-declared existing opening; this tool neither measures nor sets it')
    p.add_argument('--samples', type=int, default=4)
    p.add_argument('--interval-seconds', type=float, default=1.)
    p.add_argument('--save-dir', type=Path, default=Path('logs/orbbec_self_mask_20261109'))
    apply_pipeline_defaults(p, argv)
    args = p.parse_args(argv)
    if (not np.isfinite(args.opening_mm) or not 0 < args.opening_mm <= 100
            or args.samples < 4 or not np.isfinite(args.interval_seconds) or args.interval_seconds < .5):
        p.error('require opening in (0,100]mm, >=4 samples and interval >=0.5s')
    print('Camera only: no CAN, arm motion, enable or gripper commands.')
    print(f'Keep the EMPTY gripper already at {args.opening_mm:g}mm; do not move the camera/mount.')
    output = args.save_dir/datetime.now().strftime('%Y-%m-%d_%H-%M-%S_%f')
    records, previous_sequence, due = [], None, 0.
    try:
        camera = camera_for(args)
        while len(records) < args.samples:
            frame = camera.read()
            # Include capture age, never call repeated polling independent data.
            if frame.sequence == previous_sequence:
                continue
            previous_sequence = frame.sequence
            if not 0 <= time.monotonic()-frame.received_monotonic <= .45:
                raise RuntimeError('self-mask reference frame is stale')
            if frame.received_monotonic < due:
                continue
            directory = output/f'{len(records):02d}'
            record = save_reference(directory, frame, args, args.opening_mm)
            records.append(dict(directory=str(directory), sequence=record['sequence'],
                                received_monotonic=record['received_monotonic']))
            due = frame.received_monotonic+args.interval_seconds
            print(f'Saved {directory}', flush=True)
        (output/'manifest.json').write_text(json.dumps(dict(
            opening_declared_mm=args.opening_mm, opening_verified_by_this_tool=False,
            samples=records, purpose='three build references plus one held-out image; manual review required'), indent=2)+'\n')
        print(f'Reference directory: {output}', flush=True)
    finally:
        close_all()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
