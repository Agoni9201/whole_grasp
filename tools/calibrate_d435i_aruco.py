"""Stationary ArUco board, moving wrist D435: SDK capture and eye-in-hand solve.
No enable, speed, or motion commands. Robot positioning is external/manual.
"""
import argparse
import json
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
from scipy.spatial.transform import Rotation


def transform(rotation, translation):
    t = np.eye(4)
    t[:3, :3] = rotation
    t[:3, 3] = translation
    return t


def solve(samples):
    if len(samples) < 12:
        raise ValueError('Need at least 12 distinct stationary views; reserve additional views for validation')
    b = [np.asarray(s['T_base_flange'], dtype=float) for s in samples]
    c = [np.asarray(s['T_camera_marker'], dtype=float) for s in samples]
    for t in b + c:
        if t.shape != (4, 4) or not np.isfinite(t).all():
            raise ValueError('Invalid transform')
        if not np.allclose(t[3], [0, 0, 0, 1]) or not np.allclose(t[:3,:3].T @ t[:3,:3], np.eye(3), atol=1e-5) or np.linalg.det(t[:3,:3]) < .99:
            raise ValueError('Not a rigid transform')
    rotations = np.array([Rotation.from_matrix(b[0][:3,:3].T @ t[:3,:3]).as_rotvec() for t in b[1:]])
    if np.linalg.svd(rotations, compute_uv=False)[1] < .25:
        raise ValueError('Insufficient rotation about independent axes')
    # AX=XB: pairwise relative motions; rotation-vector alignment (Park), then translation LS.
    pairs = [(np.linalg.inv(b[j]) @ b[i], c[j] @ np.linalg.inv(c[i]))
             for i in range(len(b)) for j in range(i)]
    av = np.array([Rotation.from_matrix(a[:3,:3]).as_rotvec() for a, _ in pairs])
    bv = np.array([Rotation.from_matrix(d[:3,:3]).as_rotvec() for _, d in pairs])
    r = Rotation.align_vectors(av, bv)[0].as_matrix()
    lhs = np.vstack([a[:3,:3]-np.eye(3) for a, _ in pairs])
    rhs = np.concatenate([r @ d[:3,3]-a[:3,3] for a, d in pairs])
    p, _, rank, _ = np.linalg.lstsq(lhs, rhs, rcond=None)
    if rank < 3:
        raise ValueError('Degenerate translation system')
    x = transform(r, p)
    if not np.isfinite(x).all():
        raise ValueError('Hand-eye solver failed')
    boards = [bi @ x @ ci for bi, ci in zip(b, c)]
    reference = transform(Rotation.from_matrix(np.array([t[:3,:3] for t in boards])).mean().as_matrix(),
                          np.mean([t[:3,3] for t in boards], axis=0))
    return x, reference


def residuals(samples, x, reference):
    errors = [np.linalg.inv(reference) @ np.array(s['T_base_flange']) @ x @ np.array(s['T_camera_marker']) for s in samples]
    return {'translation_mm': [float(np.linalg.norm(t[:3,3])*1000) for t in errors],
            'rotation_deg': [float(Rotation.from_matrix(t[:3,:3]).magnitude()*180/np.pi) for t in errors]}


def read_pose(arm):
    msg = arm.get_flange_pose()
    if msg is None:
        raise RuntimeError('No flange feedback')
    stamp = float(msg.timestamp)
    pose = np.array(msg.msg, dtype=float).copy()
    if pose.shape != (6,) or not np.isfinite(pose).all() or not 0 <= time.time()-stamp <= .3:
        raise RuntimeError('Invalid/stale flange feedback')
    return transform(Rotation.from_euler('xyz', pose[3:]).as_matrix(), pose[:3]), stamp


def capture(a):
    from cameras.realsense import RealSenseCamera
    from pyAgxArm import AgxArmFactory, ArmModel, NeroFW, create_agx_arm_config
    if a.marker_mm <= 0:
        raise ValueError('marker-mm must be positive: measure black square, excluding white margin')
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, a.dictionary))
    if not 0 <= a.marker_id < len(dictionary.bytesList):
        raise ValueError('Marker ID outside dictionary')
    detector = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())
    half = a.marker_mm / 2000
    obj = np.array([[-half,half,0],[half,half,0],[half,-half,0],[-half,-half,0]], np.float32)
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=False)
    arm = AgxArmFactory.create_arm(create_agx_arm_config(robot=ArmModel.NERO,
        firmeware_version=getattr(NeroFW, a.firmware), interface='socketcan', channel=a.channel, bitrate=1000000))
    try:
        arm.connect()
        with RealSenseCamera(serial=a.serial) as camera, (out/'samples.jsonl').open('x') as log:
            print('Fixed board; position wrist externally, stop and settle. S: capture; Q/Esc: quit. No robot motion commands.')
            while True:
                frame = camera.read()
                corners, ids, _ = detector.detectMarkers(frame.bgr)
                preview = frame.bgr.copy()
                if ids is not None:
                    cv2.aruco.drawDetectedMarkers(preview, corners, ids)
                cv2.imshow('D435 wrist calibration: S save / Q quit', preview)
                key = cv2.waitKey(1) & 255
                if key in (27, ord('q')):
                    break
                if key != ord('s'):
                    continue
                try:
                    # Stationary capture: monitor before/after fresh frames, not exposure synchronization.
                    b0, ts0 = read_pose(arm)
                    time.sleep(.25)
                    camera.read()
                    frame = camera.read()
                    b1, ts1 = read_pose(arm)
                    delta = np.linalg.inv(b0) @ b1
                    if ts1 <= ts0 or np.linalg.norm(delta[:3,3]) > .001 or Rotation.from_matrix(delta[:3,:3]).magnitude() > np.deg2rad(.3):
                        raise RuntimeError('Wrist moving or feedback not advancing; settle and retry')
                    corners, ids, _ = detector.detectMarkers(frame.bgr)
                    matches = [] if ids is None else np.flatnonzero(ids.flatten() == a.marker_id)
                    if len(matches) != 1:
                        raise RuntimeError('Expected exactly one requested marker')
                    points = corners[matches[0]].reshape(4,2)
                    ok, rv, tv = cv2.solvePnP(obj, points, frame.K, frame.D, flags=cv2.SOLVEPNP_ITERATIVE)
                    if not ok or tv[2,0] <= 0:
                        raise RuntimeError('Invalid marker pose')
                    proj, _ = cv2.projectPoints(obj, rv, tv, frame.K, frame.D)
                    error = float(np.sqrt(np.mean(np.sum((proj.reshape(4,2)-points)**2, axis=1))))
                    if error > 1.5:
                        raise RuntimeError(f'Reprojection error {error:.2f}px exceeds 1.5px')
                    image_name = f'{time.time_ns()}.png'
                    if not cv2.imwrite(str(out/image_name), frame.bgr):
                        raise OSError('Image save failed')
                    sample = dict(T_base_flange=b1.tolist(), T_camera_marker=transform(cv2.Rodrigues(rv)[0],tv.flatten()).tolist(),
                        serial=frame.serial, dictionary=a.dictionary, marker_id=a.marker_id, marker_mm=a.marker_mm,
                        K=frame.K.tolist(), D=frame.D.tolist(), corners=points.tolist(), image=image_name,
                        reprojection_px=error, feedback_stamps=[ts0,ts1], frame_hardware_ms=frame.hardware_ms)
                    log.write(json.dumps(sample)+'\n')
                    log.flush()
                    print('Saved', image_name, 'reprojection px', error)
                except RuntimeError as exc:
                    print('Rejected:', exc)
    finally:
        try:
            arm.disconnect()
        finally:
            cv2.destroyAllWindows()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    c = sub.add_parser('capture')
    c.add_argument('--output', required=True, help='New session directory')
    c.add_argument('--dictionary', required=True, help='e.g. DICT_4X4_50')
    c.add_argument('--marker-id', required=True, type=int)
    c.add_argument('--marker-mm', required=True, type=float)
    c.add_argument('--serial', required=True)
    c.add_argument('--channel', default='can0')
    c.add_argument('--firmware', choices=['V111','V112'], default='V111')
    for name in ['solve', 'validate']:
        s = sub.add_parser(name)
        s.add_argument('--samples', required=True)
        if name == 'solve':
            s.add_argument('--output', required=True)
        else:
            s.add_argument('--calibration', required=True)
    a = p.parse_args(argv)
    if a.command == 'capture':
        return capture(a)
    samples = [json.loads(line) for line in Path(a.samples).read_text().splitlines() if line.strip()]
    if not samples:
        raise ValueError('Empty samples')
    fields = ['serial','dictionary','marker_id','marker_mm']
    metadata = {k:samples[0][k] for k in fields}
    if any(any(s[k] != metadata[k] for k in fields) for s in samples):
        raise ValueError('Mixed camera/board data')
    if a.command == 'solve':
        x, reference = solve(samples)
        result = dict(position=x[:3,3].tolist(), orientation=Rotation.from_matrix(x[:3,:3]).as_quat().tolist(),
            convention='flange_camera_xyzw', parent_frame='SDK_feedback_flange', child_frame='color_optical_frame',
            T_flange_camera=x.tolist(), T_base_marker=reference.tolist(), metadata=metadata,
            training_images=[s['image'] for s in samples], status='unvalidated', residuals=residuals(samples,x,reference))
        with Path(a.output).open('x') as f:
            json.dump(result,f,indent=2)
    else:
        result = json.loads(Path(a.calibration).read_text())
        if result['metadata'] != metadata:
            raise ValueError('Calibration camera/board mismatch')
        if set(result['training_images']) & {s['image'] for s in samples}:
            raise ValueError('Validation requires independent held-out captures')
        if len(samples) < 3:
            raise ValueError('Need at least 3 held-out views')
        report = residuals(samples,np.array(result['T_flange_camera']),np.array(result['T_base_marker']))
        print(json.dumps(report,indent=2))
        print('Residuals only: verify geometry and application tolerances before robot execution.')


if __name__ == '__main__':
    main()
