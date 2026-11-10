#!/usr/bin/env python3
"""Compare gyro, fixed-board vision, and robot rotation angles; read-only.

The rotation magnitude is invariant to a fixed camera-to-flange rotation.
No hand-eye transform or depth is used. The operator moves the arm through
the existing teaching interface; this program never enables or moves it.
"""
import argparse
from datetime import datetime
from pathlib import Path
import select
import sys
import time

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from cameras.options import add_camera_arguments,apply_pipeline_defaults
from cameras.realsense import create_camera
from tools.calibrate_d435i_charuco import (
    NeroReadOnlyFeedback,atomic_json,detect_charuco,draw_detection,
    load_board_config,quality_reason,
)


def integrate_gyro(samples,start_s,end_s,bias,end_bias=None):
    """Integrate body angular velocity using device timestamps and trapezoids."""
    data=np.asarray(samples,float)
    bias=np.asarray(bias,float).reshape(3)
    end_bias=bias if end_bias is None else np.asarray(end_bias,float).reshape(3)
    if (data.ndim!=2 or data.shape[1]!=4 or len(data)<2
            or not np.isfinite(data).all() or not np.isfinite(np.r_[bias,end_bias]).all()
            or np.any(np.diff(data[:,0])<=0) or not data[0,0]<=start_s<end_s<=data[-1,0]):
        raise ValueError('Gyro timestamps must increase and cover both RGB endpoint times')
    inside=data[(data[:,0]>start_s)&(data[:,0]<end_s)]
    endpoints=np.array([[t,*[np.interp(t,data[:,0],data[:,k]) for k in (1,2,3)]]
                        for t in (start_s,end_s)])
    segment=np.vstack((endpoints[0],inside,endpoints[1]))
    rotation=Rotation.identity()
    for a,b in zip(segment[:-1],segment[1:]):
        fraction=((a[0]+b[0])/2.-start_s)/(end_s-start_s)
        velocity=(a[1:]+b[1:])/2.-((1.-fraction)*bias+fraction*end_bias)
        rotation=rotation*Rotation.from_rotvec(velocity*(b[0]-a[0]))
    return rotation


def parse_args(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    add_camera_arguments(parser)
    parser.add_argument('--board',default='config/charuco_board.yaml')
    parser.add_argument('--channel',default='can0')
    parser.add_argument('--firmware',choices=('V111','V112'),default='V111')
    parser.add_argument('--save-dir',type=Path,default=Path('calibration/orbbec_rotation_audit'))
    apply_pipeline_defaults(parser,argv)
    args=parser.parse_args(argv)
    if args.camera_backend!='orbbec':
        parser.error('This independent gyro check requires the Orbbec backend')
    return args


def main(argv=None):
    args=parse_args(argv)
    import pyorbbecsdk as ob
    board,detector,_,quality=load_board_config(args.board)
    camera=create_camera(backend=args.camera_backend,serial=args.camera_serial,
        width=args.camera_width,height=args.camera_height,fps=args.camera_fps,
        timeout=args.camera_timeout,warmup=args.camera_warmup,enable_depth=False)
    gyro_samples=[]
    def on_gyro(frame):
        f=frame.as_gyro_frame();v=f.get_value()
        gyro_samples.append([f.get_timestamp_us()*1e-6,v.x,v.y,v.z])
    title='Rotation audit: S/Enter start/end | R reset bias | Q/Esc exit'
    print('READ ONLY: camera/gyro/robot feedback. No enable, motion, stop or gripper commands.')
    print('Keep still for 2 seconds until READY. Focus this window; S/Enter records START.')
    print('Use your existing low-speed teaching controls to change one joint, stop, then S/Enter records END.')
    print('R starts a fresh still baseline. The board must remain fixed and visible at both endpoints.')
    start=end=None
    bias=None
    message='Keep still: collecting gyro bias'
    reset_time=time.monotonic()
    try:
        with camera,NeroReadOnlyFeedback(args.channel,args.firmware,5.) as arm:
            sensor=camera.pipeline.get_device().get_sensor(ob.OBSensorType.GYRO_SENSOR)
            profile=sensor.get_stream_profile_list().get_stream_profile_by_index(0)
            sensor.start(profile,on_gyro)
            try:
                reset_time=time.monotonic()
                cv2.namedWindow(title,cv2.WINDOW_NORMAL);cv2.resizeWindow(title,1100,700)
                while True:
                    frame=camera.read()
                    detection=detect_charuco(frame.bgr,board,detector,frame.K,frame.D)
                    reason=quality_reason(detection,quality)
                    if bias is None and time.monotonic()-reset_time>=2. and len(gyro_samples)>=20:
                        baseline=np.asarray(gyro_samples.copy(),float)
                        bias=np.mean(baseline[:,1:],axis=0)
                        baseline_std=np.std(baseline[:,1:],axis=0)
                        message='READY: keep still, press S/Enter for START'
                        print('[READY] gyro bias rad/s=',bias.tolist(),'std=',baseline_std.tolist())
                    visual=draw_detection(frame.bgr,detection,frame.K,frame.D,quality['axis_length_m'])
                    state='BASELINE' if bias is None else ('READY' if start is None else 'RECORDING' if end is None else 'FINISHING')
                    cv2.rectangle(visual,(0,0),(visual.shape[1],100),(20,20,20),-1)
                    cv2.putText(visual,f'{state} | corners={detection["valid_corners"]} | S/Enter start/end, R reset, Q exit',
                                (12,26),cv2.FONT_HERSHEY_SIMPLEX,.57,(255,255,255),2)
                    cv2.putText(visual,message[:120],(12,57),cv2.FONT_HERSHEY_SIMPLEX,.52,(0,220,255),1)
                    cv2.putText(visual,'Board: OK' if reason is None else f'Board: {reason}',
                                (12,84),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,220,0) if reason is None else (0,0,255),1)
                    cv2.imshow(title,visual);key=cv2.waitKey(1)&255
                    if key in (27,ord('q'),ord('Q')) or cv2.getWindowProperty(title,cv2.WND_PROP_VISIBLE)<1:
                        break
                    record=key in (10,13,ord('s'),ord('S'))
                    if select.select([sys.stdin],[],[],0)[0]:
                        line=sys.stdin.readline()
                        if not line or line.strip().lower()=='q':break
                        record=record or not line.strip()
                    if key in (ord('r'),ord('R')):
                        gyro_samples.clear();reset_time=time.monotonic();bias=None;start=end=None
                        message='Keep still: collecting new gyro bias'
                        continue
                    if record and end is None:
                        if bias is None:
                            message='Wait for READY while keeping still'
                        elif reason is not None:
                            message=f'Not recorded: {reason}'
                        else:
                            pose,T,stamp=arm.read();q,qstamp=arm.read_joints()
                            snapshot=dict(camera_hw_s=frame.hardware_ms*.001,camera_sequence=frame.sequence,
                                image_received_monotonic=frame.received_monotonic,
                                joint_rad=q.tolist(),joint_stamp=qstamp,flange_stamp=stamp,
                                flange_pose=pose.tolist(),T_base_flange=T.tolist(),
                                T_camera_board=detection['T_camera_target'].tolist(),
                                corner_reprojection_px=detection['reprojection_mean_px'])
                            if start is None:
                                ready_bias=bias.copy()
                                baseline=np.asarray(gyro_samples.copy(),float)
                                baseline=baseline[(baseline[:,0]>=snapshot['camera_hw_s']-2.)
                                                  &(baseline[:,0]<=snapshot['camera_hw_s'])]
                                bias=np.mean(baseline[:,1:],axis=0)
                                baseline_std=np.std(baseline[:,1:],axis=0)
                                start=snapshot;start_image=frame.bgr.copy()
                                message='START recorded. Manually move one joint, stop, then S/Enter for END'
                                print('[START] q_deg=',np.degrees(q).round(3).tolist())
                                print('[START] refreshed gyro bias from the preceding 2 seconds:',bias.tolist())
                            else:
                                end=snapshot;end_image=frame.bgr.copy()
                                message='END recorded: stay still 2 seconds for final gyro bias'
                                print('[END] q_deg=',np.degrees(q).round(3).tolist())
                    if end is not None and gyro_samples[-1][0]>=end['camera_hw_s']+2.:
                        raw=np.asarray(gyro_samples.copy(),float)
                        if raw[0,0]>start['camera_hw_s']:
                            raise RuntimeError('Gyro and color device clocks do not overlap; raw timestamps need checking')
                        resting=raw[(raw[:,0]>=end['camera_hw_s']+.2)&(raw[:,0]<=end['camera_hw_s']+2.),1:]
                        final_bias=np.mean(resting,axis=0)
                        gyro=integrate_gyro(raw,start['camera_hw_s'],end['camera_hw_s'],bias,final_bias)
                        fixed_bias_gyro=integrate_gyro(raw,start['camera_hw_s'],end['camera_hw_s'],bias)
                        b0=np.array(start['T_base_flange']);b1=np.array(end['T_base_flange'])
                        c0=np.array(start['T_camera_board']);c1=np.array(end['T_camera_board'])
                        robot=Rotation.from_matrix(b0[:3,:3].T@b1[:3,:3])
                        vision=Rotation.from_matrix(c0[:3,:3]@c1[:3,:3].T)
                        angles=dict(gyro_deg=float(np.degrees(gyro.magnitude())),
                                    robot_deg=float(np.degrees(robot.magnitude())),
                                    vision_deg=float(np.degrees(vision.magnitude())))
                        out=args.save_dir/datetime.now().strftime('%Y%m%d_%H%M%S_%f');out.mkdir(parents=True)
                        report=dict(status='diagnostic_not_calibration',robot_commands_sent=False,
                            board=str(Path(args.board).resolve()),serial=camera.serial,
                            start=start,end=end,gyro_bias_rad_s=bias.tolist(),
                            gyro_ready_bias_rad_s=ready_bias.tolist(),gyro_bias_window_s=2.,
                            gyro_baseline_std_rad_s=baseline_std.tolist(),angles=angles,
                            gyro_end_bias_rad_s=final_bias.tolist(),
                            gyro_end_std_rad_s=np.std(resting,axis=0).tolist(),
                            gyro_fixed_start_bias_angle_deg=float(np.degrees(fixed_bias_gyro.magnitude())),
                            duration_s=end['camera_hw_s']-start['camera_hw_s'],
                            gyro_max_sample_gap_ms=float(np.max(np.diff(raw[:,0]))*1000.),
                            gyro_minus_robot_deg=angles['gyro_deg']-angles['robot_deg'],
                            vision_minus_robot_deg=angles['vision_deg']-angles['robot_deg'],
                            joint_change_deg=np.degrees(np.array(end['joint_rad'])-start['joint_rad']).tolist(),
                            limitations='One relative rotation test. Bias interpolates between still start/end baselines; device clock alignment and a rigid mount are assumed. IMU errors are not an absolute metrology certificate.')
                        atomic_json(out/'result.json',report);np.save(out/'gyro_samples.npy',raw)
                        cv2.imwrite(str(out/'start.png'),start_image);cv2.imwrite(str(out/'end.png'),end_image)
                        print(f'[ANGLES] robot={angles["robot_deg"]:.3f} deg | '
                              f'vision={angles["vision_deg"]:.3f} deg | gyro={angles["gyro_deg"]:.3f} deg')
                        print(f'[DIFFERENCE] vision-robot={report["vision_minus_robot_deg"]:+.3f} deg | '
                              f'gyro-robot={report["gyro_minus_robot_deg"]:+.3f} deg')
                        print('[SAVED]',out/'result.json')
                        print('One trial finished. Exit, or keep still and press R for another baseline.')
                        start=end=None;bias=None;reset_time=time.monotonic();gyro_samples.clear()
                        message=f'SAVED {out.name}; keep still for next bias baseline'
            finally:sensor.stop()
    except KeyboardInterrupt:
        print('\nStopped. Completed trial files are retained.')
    finally:cv2.destroyAllWindows()


if __name__=='__main__':
    main()
