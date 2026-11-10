#!/usr/bin/env python3
"""D435i SDK RGB-D -> YOLO segmentation -> target cloud (no ROS/CAN)."""
import argparse
import json
import sys
from contextlib import closing
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
import open3d as o3d
from cameras.realsense import RealSenseCamera
from real_nero_grasp.depth_processing import depth_to_xyz_image
from real_nero_grasp.perception import cloud_from_mask, infer_yolo_instances, mask_for_instance

def _make_camera(serial, width, height, fps, warmup, depth_preset,
                 emitter_enabled, laser_power):
    return RealSenseCamera(
        serial=serial,
        width=width,
        height=height,
        fps=fps,
        warmup=warmup,
        depth_preset=depth_preset,
        emitter_enabled=emitter_enabled,
        laser_power=laser_power,
    )

def capture(width, height, fps, warmup, serial=None, depth_preset='high_density',
            emitter_enabled=None, laser_power=None):
    # This helper is single-shot; the live loop below uses one persistent camera.
    with _make_camera(
        serial, width, height, fps, warmup, depth_preset,
        emitter_enabled, laser_power,
    ) as camera:
        frame = camera.read()
        return frame.bgr, frame.depth_m, frame.intr


def capture_loop(width, height, fps, warmup, serial=None,
                 depth_preset='high_density', emitter_enabled=None,
                 laser_power=None):
    camera = _make_camera(
        serial, width, height, fps, warmup, depth_preset,
        emitter_enabled, laser_power,
    )
    try:
        camera.start()
        while True:
            frame = camera.read()
            intr = SimpleNamespace(
                fx=frame.intr.fx, fy=frame.intr.fy,
                cx=frame.intr.cx, cy=frame.intr.cy,
                distortion_coeffs=frame.D.tolist(),
                serial=frame.serial,
                color_frame_number=frame.sequence,
                depth_frame_number=frame.sequence,
                color_timestamp_ms=frame.hardware_ms,
                depth_timestamp_ms=frame.hardware_ms,
            )
            yield frame.bgr, frame.depth_m, intr
    except RuntimeError as exc:
        raise RuntimeError('D435i frame stream lost/disconnected; check USB 3.x cable, port power, and avoid moving the cable') from exc
    finally:
        camera.close()

def save_snapshot(args, bgr, depth, intr, overlay, mask, instance, raw_cloud, target_cloud):
    """Save one matching RGB-D/mask/cloud set; publish directory only after success."""
    stamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S_%f')
    root = Path(args.save_dir).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    pending = root / (stamp + '.partial')
    pending.mkdir(exist_ok=False)
    for name, array in [('color.png', bgr), ('overlay.png', overlay),
                        ('target_mask.png', (np.asarray(mask) > 0).astype(np.uint8) * 255)]:
        if not cv2.imwrite(str(pending / name), array):
            raise OSError(f'Failed to save {name}')
    np.save(pending / 'depth_m.npy', depth)
    for name, points in [('target_raw.ply', raw_cloud), ('target.ply', target_cloud)]:
        if not o3d.io.write_point_cloud(str(pending / name), points):
            raise OSError(f'Failed to save {name}')
    metadata = dict(
        coordinate_frame='color_optical_frame', length_unit='meter',
        depth_aligned_to='color', color_order='RGB in PLY; BGR in OpenCV arrays',
        width=int(bgr.shape[1]), height=int(bgr.shape[0]), intrinsics=vars(intr),
        K=[[intr.fx, 0, intr.cx], [0, intr.fy, intr.cy], [0, 0, 1]],
        label=instance['label'], score=float(instance['score']),
        bbox_xyxy=np.asarray(instance['xyxy']).tolist(),
        raw_points=len(raw_cloud.points), target_points=len(target_cloud.points),
        weights=str(Path(args.weights).resolve()), min_depth=args.min_depth,
        max_depth=args.max_depth, voxel_size=args.voxel_size,
        outlier_filter=args.target_outlier_filter,
    )
    (pending / 'capture_meta.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    destination = root / stamp
    pending.rename(destination)
    return destination.resolve()

def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--weights',default='models/yolo26m-seg.pt'); p.add_argument('--target-class',required=True); p.add_argument('--target-index',type=int,default=0); p.add_argument('--serial',default=None); p.add_argument('--width',type=int,default=640); p.add_argument('--height',type=int,default=480); p.add_argument('--fps',type=int,default=30); p.add_argument('--warmup',type=int,default=30); p.add_argument('--depth-preset',choices=('high_density','unchanged'),default='high_density'); p.add_argument('--emitter-enabled',action='store_true',default=None); p.add_argument('--laser-power',type=float,default=None); p.add_argument('--yolo-imgsz',type=int,default=640); p.add_argument('--yolo-conf',type=float,default=.25); p.add_argument('--min-depth',type=float,default=.05); p.add_argument('--max-depth',type=float,default=1.5); p.add_argument('--voxel-size',type=float,default=.0015); p.add_argument('--min-target-points',type=int,default=250); p.add_argument('--no-outlier-filter',action='store_true'); p.add_argument('--save-dir', default='captures/d435i_graspnet', help='Press S in RGB window to save a valid target snapshot here'); a=p.parse_args()
    if not all(x>0 for x in (a.width,a.height,a.fps)) or not a.min_depth<a.max_depth: p.error('invalid camera/depth parameters')
    a.yolo_weights=a.weights; a.yolo_classes=None; a.target_class=a.target_class; a.outlier_neighbors=20; a.outlier_std_ratio=2.; a.target_outlier_filter=not a.no_outlier_filter
    print(f'实时推理已启动；将 {a.target_class} 放入画面，在 RGB 窗口按 S 保存，Q/Esc 退出。')
    print('保存目录:', Path(a.save_dir).expanduser().resolve())
    v=o3d.visualization.Visualizer(); v.create_window('D435i target cloud'); cloud=o3d.geometry.PointCloud(); added=False
    try:
        with closing(capture_loop(
            a.width, a.height, a.fps, a.warmup,
            serial=a.serial,
            depth_preset=a.depth_preset,
            emitter_enabled=a.emitter_enabled,
            laser_power=a.laser_power,
        )) as frames:
            for bgr,z,intr in frames:
                # Existing projection is pinhole-only; never save misleading XYZ from distorted pixels.
                if np.any(np.abs(intr.distortion_coeffs) > 1e-9):
                    raise RuntimeError('Nonzero color distortion: rectify before pinhole back-projection')
                xyz=depth_to_xyz_image(z,intr); instances,overlay=infer_yolo_instances(bgr,a)
                snapshot = None  # Never save a target from an earlier frame.
                print('\rRGB',bgr.shape,'detections',[(x['label'],round(float(x['score']),2)) for x in instances],end=' ',flush=True)
                if 0<=a.target_index<len(instances) and not instances[a.target_index].get('is_box_only',False):
                    instance = instances[a.target_index]
                    mask=mask_for_instance(instance,xyz,a)
                    raw=SimpleNamespace(**vars(a)); raw.voxel_size=0.; raw.target_outlier_filter=False
                    try:
                        c0=cloud_from_mask(xyz,bgr,mask,raw)
                        c1=cloud_from_mask(xyz,bgr,mask,a)
                        if len(c1.points) < a.min_target_points:
                            raise RuntimeError(f'filtered target cloud too small: {len(c1.points)}')
                    except RuntimeError as exc:
                        print('\n当前目标不可保存:', exc)
                    else:
                        snapshot=(mask,instance,c0,c1)
                        print('cloud',len(c0.points),'->',len(c1.points),end='',flush=True)
                        cloud.points=o3d.utility.Vector3dVector(np.asarray(c1.points)); cloud.colors=o3d.utility.Vector3dVector(np.asarray(c1.colors))
                        if not added: v.add_geometry(cloud); added=True
                        else: v.update_geometry(cloud)
                if snapshot is None and added:
                    cloud.clear()
                    v.update_geometry(cloud)
                cv2.imshow('D435i RGB + YOLO mask (S save, Q/Esc)',overlay)
                if not v.poll_events():
                    break
                v.update_renderer()
                key=cv2.waitKey(1)&255
                if key in (27,ord('q'),ord('Q')):
                    break
                if key in (ord('s'),ord('S')):
                    if snapshot is None:
                        print('\n未保存：当前帧没有有效分割目标点云，请调整物体后重试。')
                    else:
                        try:
                            path=save_snapshot(a,bgr,z,intr,overlay,*snapshot)
                            print('\n已保存:', path)
                        except OSError as exc:
                            print('\n保存失败（不完整输出保留在 .partial 目录）:', exc)
    except KeyboardInterrupt:
        print('\n退出预览。')
    finally:
        v.destroy_window(); cv2.destroyAllWindows()
if __name__=='__main__': main()
