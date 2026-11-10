#!/usr/bin/env python3
"""SDK RGB-D/registration preview and device listing; never imports the arm SDK."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import sys

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cameras.options import add_camera_arguments, apply_pipeline_defaults
from cameras.realsense import camera_for, close_all


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    add_camera_arguments(parser)
    parser.add_argument('--list-devices', action='store_true', help='Orbbec device/profile enumeration only')
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--max-frames', type=int, default=0)
    parser.add_argument('--save-dir', type=Path, default=Path('captures/orbbec_preview'))
    apply_pipeline_defaults(parser, argv)
    args = parser.parse_args(argv)
    if args.max_frames < 0:
        parser.error('--max-frames must be nonnegative')
    print('Camera only: no CAN connection, arm motion, or gripper commands.')
    if args.list_devices:
        if args.camera_backend != 'orbbec':
            parser.error('--list-devices currently enumerates the Orbbec SDK')
        import pyorbbecsdk as ob
        context = ob.Context()
        devices = context.query_devices()
        print(f'Orbbec devices: {devices.get_count()}')
        for index in range(devices.get_count()):
            print(f'{devices.get_device_name_by_index(index)} '
                  f'serial={devices.get_device_serial_number_by_index(index)} '
                  f'connection={devices.get_device_connection_type_by_index(index)}')
            device = devices.get_device_by_index(index)
            info = device.get_device_info()
            print(f'{info.get_name()} serial={info.get_serial_number()}')
            pipe = ob.Pipeline(device)
            for sensor in (ob.OBSensorType.COLOR_SENSOR, ob.OBSensorType.DEPTH_SENSOR):
                profiles = pipe.get_stream_profile_list(sensor)
                modes = set()
                for i in range(profiles.get_count()):
                    p = profiles.get_stream_profile_by_index(i).as_video_stream_profile()
                    modes.add(f'{p.get_width()}x{p.get_height()}@{p.get_fps()} {p.get_format()}')
                print(f'{sensor}:\n' + '\n'.join(modes))
        return
    camera = camera_for(args)
    title = 'RGB | aligned depth | registration overlay (S save, Q/Esc exit)'
    count = 0
    try:
        if not args.headless:
            cv2.namedWindow(title, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(title, 1440, 450)
        while not args.max_frames or count < args.max_frames:
            frame = camera.read()
            count += 1
            valid = np.isfinite(frame.depth_m) & (frame.depth_m > 0)
            scaled = np.clip(frame.depth_m / 2. * 255, 0, 255).astype(np.uint8)
            depth = cv2.applyColorMap(scaled, cv2.COLORMAP_TURBO)
            depth[~valid] = 0
            overlay = frame.bgr.copy()
            overlay[valid] = cv2.addWeighted(frame.bgr, .6, depth, .4, 0)[valid]
            if count == 1 or count % 30 == 0:
                print(f'{frame.backend} serial={frame.serial} RGB={frame.bgr.shape} '
                      f'aligned_depth={frame.depth_m.shape} valid={valid.mean():.1%} '
                      f'median={np.median(frame.depth_m[valid]) if valid.any() else 0.:.3f} m')
            if args.headless:
                continue
            cv2.imshow(title, np.hstack((frame.bgr, depth, overlay)))
            key = cv2.waitKey(1) & 255
            if key in (27, ord('q'), ord('Q')) or cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1:
                break
            if key in (ord('s'), ord('S')):
                output = args.save_dir / datetime.now().strftime('%Y-%m-%d_%H-%M-%S_%f')
                output.mkdir(parents=True)
                cv2.imwrite(str(output/'color.png'), frame.bgr)
                cv2.imwrite(str(output/'overlay.png'), overlay)
                np.save(output/'depth_m.npy', frame.depth_m)
                if frame.raw_depth_m is not None:
                    np.save(output/'depth_raw_m.npy', frame.raw_depth_m)
                (output/'intrinsics.json').write_text(json.dumps(dict(
                    backend=frame.backend, serial=frame.serial, K=frame.K.tolist(), D=frame.D.tolist(),
                    depth_unit='metre', rectified=True, sequence=frame.sequence,
                    hardware_ms=frame.hardware_ms), indent=2))
                print(f'Saved {output}')
    except KeyboardInterrupt:
        pass
    finally:
        close_all()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
