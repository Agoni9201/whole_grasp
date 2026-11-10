#!/usr/bin/env python3
"""Offline hand-eye initialization, loss, coverage and corner audit. No hardware."""
import argparse
import csv
import hashlib
import html
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.calibrate_d435i_charuco import (
    consistency_center_transform, detect_charuco, load_board_config,
    load_dataset, make_transform, residual_values, solve_opencv_methods,
    summarize_residuals,
)
from targets.aruco_eye_in_hand_target import load_transform_file


def refine(samples, seed, board_points, *, model='pose', loss='linear', method='lm',
           max_nfev=500, f_scale=1.):
    """Fit X/Y in B_i X C_i=Y; keep K/D and all observed poses unchanged."""
    bases = np.asarray([s['T_base_flange'] for s in samples])
    cameras = np.asarray([s['T_camera_target'] for s in samples])
    board_seed = consistency_center_transform(bases @ seed @ cameras)
    if model == 'pixels':
        # This audit uses the saved Orbbec rectified pinhole images only.
        if any(np.any(np.asarray(s['D']) != 0) for s in samples):
            raise ValueError('Pixel audit requires rectified images with D=0')
        frame_index = np.concatenate([np.full(len(s['charuco_ids']), i)
                                      for i, s in enumerate(samples)])
        objects = np.concatenate([board_points[np.asarray(s['charuco_ids'])]
                                  for s in samples])
        observed = np.concatenate([s['image_points'] for s in samples])
        intrinsics = np.asarray([s['K'] for s in samples])[frame_index]
    base_rt = bases[:, :3, :3].transpose(0, 2, 1)

    def transforms(p):
        x = make_transform(Rotation.from_rotvec(p[:3]).as_matrix() @ seed[:3, :3],
                           seed[:3, 3] + p[3:6])
        y = make_transform(Rotation.from_rotvec(p[6:9]).as_matrix() @ board_seed[:3, :3],
                           board_seed[:3, 3] + p[9:12])
        return x, y

    def residual(p):
        x, y = transforms(p)
        if model == 'pose':
            estimates = bases @ x @ cameras
            position = (estimates[:, :3, 3] - y[:3, 3]) / .005
            rotation = (Rotation.from_matrix(y[:3, :3]).inv()
                        * Rotation.from_matrix(estimates[:, :3, :3])).as_rotvec()
            return np.r_[position.ravel(), (rotation / np.deg2rad(1.)).ravel()]
        rc = x[:3, :3].T @ base_rt @ y[:3, :3]
        tc = np.einsum('ij,nj->ni', x[:3, :3].T,
                       np.einsum('nij,nj->ni', base_rt, y[:3, 3] - bases[:, :3, 3])
                       - x[:3, 3])
        points = np.einsum('nij,nj->ni', rc[frame_index], objects) + tc[frame_index]
        projected = points[:, :2] / points[:, 2:]
        projected = projected * intrinsics[:, [0, 1], [0, 1]] + intrinsics[:, :2, 2]
        return (projected - observed).ravel()

    started = time.perf_counter()
    result = least_squares(residual, np.zeros(12), method=method, loss=loss,
                           f_scale=f_scale, max_nfev=max_nfev,
                           ftol=1e-9, xtol=1e-9, gtol=1e-9)
    x, y = transforms(result.x)
    detail = dict(success=bool(result.success), status=int(result.status),
                  message=result.message, nfev=int(result.nfev),
                  njev=None if result.njev is None else int(result.njev),
                  elapsed_s=time.perf_counter()-started, cost=float(result.cost),
                  residual_model=model, loss=loss, method=method,
                  max_function_evaluations=max_nfev,
                  scaling='5 mm / 1 deg for pose; pixels for image coordinates',
                  huber_scale=f_scale, optimized_board_pose=y.tolist())
    return x, detail


def evaluate(samples, x, y, points):
    stats = summarize_residuals(residual_values(samples, x, y))
    pixel_errors = []
    positive = True
    for s in samples:
        camera = np.linalg.inv(np.asarray(s['T_base_flange']) @ x) @ y
        objects = points[np.asarray(s['charuco_ids'])]
        positive = positive and bool(np.all((objects @ camera[:3, :3].T
                                            + camera[:3, 3])[:, 2] > 0))
        projected, _ = cv2.projectPoints(objects, cv2.Rodrigues(camera[:3, :3])[0],
                                         camera[:3, 3], np.asarray(s['K']),
                                         np.asarray(s['D']))
        pixel_errors.extend(np.linalg.norm(projected.reshape(-1, 2)
                                            - s['image_points'], axis=1).tolist())
    stats['handeye_constrained_reprojection_px'] = dict(
        mean=float(np.mean(pixel_errors)), rms=float(np.sqrt(np.mean(np.square(pixel_errors)))),
        max=float(np.max(pixel_errors)))
    stats['all_board_points_in_front_of_camera'] = positive
    return stats


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train', default='calibration/orbbec_oldboard_check_20261102_172656')
    parser.add_argument('--validation', default='calibration/orbbec_sync_check_20261102_175351')
    parser.add_argument('--old-calibration', default='config/2026-09-01_19-08-35_calibration.json')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    train_root, validation_root = Path(args.train), Path(args.validation)
    train_data = load_dataset(train_root/'samples.json')
    validation_data = load_dataset(validation_root/'samples.json')
    train, validation = train_data['samples'], validation_data['samples']
    if train_data['board'] != validation_data['board'] or train_data['camera'] != validation_data['camera']:
        raise ValueError('Train/validation camera and board must match')
    if {s['image_sha256'] for s in train} & {s['image_sha256'] for s in validation}:
        raise ValueError('Validation images overlap training images')
    board, detector, _, _ = load_board_config(train_root/'board.yaml')
    points = board.getChessboardCorners()
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    (output/'corners').mkdir(exist_ok=True)
    coverage_rows, tiles, all_pixels = [], [], []
    width, height = train_data['camera']['width'], train_data['camera']['height']
    for s in train:
        path = train_root/s['image']
        if hashlib.sha256(path.read_bytes()).hexdigest() != s['image_sha256']:
            raise ValueError(f'Image changed: {path}')
        image = cv2.imread(str(path))
        d = detect_charuco(image, board, detector, np.asarray(s['K']), np.asarray(s['D']))
        ids, pixels = d['charuco_ids_flat'], d['image_points']
        if len(set(ids)) != len(ids) or np.any(ids < 0) or np.any(ids >= len(points)):
            raise ValueError(f'Invalid corner IDs: {path}')
        projected, _ = cv2.projectPoints(points[ids], d['rvec'], d['tvec'],
                                         np.asarray(s['K']), np.asarray(s['D']))
        projected = projected.reshape(-1, 2)
        ids_to_saved = dict(zip(s['charuco_ids'], s['image_points']))
        mismatch = float(np.max(np.linalg.norm(pixels - [ids_to_saved[int(i)] for i in ids], axis=1)))
        for index, observed, predicted in zip(ids, pixels, projected):
            a, b = tuple(np.rint(observed).astype(int)), tuple(np.rint(predicted).astype(int))
            cv2.circle(image, a, 4, (0, 255, 0), 1)
            cv2.drawMarker(image, b, (0, 0, 255), cv2.MARKER_CROSS, 5, 1)
            cv2.putText(image, str(index), (a[0]+4, a[1]-4), cv2.FONT_HERSHEY_SIMPLEX, .3, (255, 100, 0), 1)
        cv2.rectangle(image, (0, 0), (width, 35), (20, 20, 20), -1)
        cv2.putText(image, f'ID {s["sample_id"]:02d} | corners {len(ids)} | mean {d["reprojection_mean_px"]:.3f}px | green=detected red=PnP',
                    (10, 24), cv2.FONT_HERSHEY_SIMPLEX, .58, (255, 255, 255), 1)
        cv2.imwrite(str(output/'corners'/f'{s["sample_id"]:03d}.jpg'), image)
        tiles.append(cv2.resize(image, (384, 216)))
        all_pixels.extend(pixels.tolist())
        normal = np.asarray(s['T_camera_target'])[:3, 2]
        center = pixels.mean(axis=0)
        coverage_rows.append(dict(sample_id=int(s['sample_id']), corners=len(ids),
            reprojection_mean_px=float(d['reprojection_mean_px']),
            reprojection_rms_px=float(np.sqrt(np.mean(np.sum((pixels-projected)**2,axis=1)))),
            reprojection_max_px=float(np.linalg.norm(pixels-projected,axis=1).max()),
            saved_vs_redetected_max_px=mismatch,
            centroid_u_px=float(center[0]), centroid_v_px=float(center[1]),
            convex_hull_image_fraction=float(cv2.contourArea(cv2.convexHull(pixels.astype(np.float32)))/(width*height)),
            board_z_m=float(np.asarray(s['T_camera_target'])[2, 3]),
            board_tilt_from_optical_axis_deg=float(np.degrees(np.arccos(np.clip(abs(normal[2]), 0, 1))))))
    sheet = np.full((((len(tiles)+2)//3)*216, 3*384, 3), 255, np.uint8)
    for i, tile in enumerate(tiles): sheet[(i//3)*216:(i//3+1)*216, (i%3)*384:(i%3+1)*384] = tile
    cv2.imwrite(str(output/'corner_contact_sheet.jpg'), sheet)
    with (output/'frame_quality.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(coverage_rows[0])); writer.writeheader(); writer.writerows(coverage_rows)

    bases = np.asarray([s['T_base_flange'] for s in train])
    cameras = np.asarray([s['T_camera_target'] for s in train])
    rotation_distance = np.zeros((len(train), len(train)))
    duplicate_pairs = []
    # Keep the lower-error member of a near-identical robot AND visual pose pair.
    unique = []
    by_quality = sorted(range(len(train)), key=lambda i: train[i]['reprojection_mean_px'])
    for i in by_quality:
        duplicate_of = None
        for j in unique:
            rb = float(np.degrees(Rotation.from_matrix(bases[i, :3, :3].T @ bases[j, :3, :3]).magnitude()))
            rc = float(np.degrees(Rotation.from_matrix(cameras[i, :3, :3].T @ cameras[j, :3, :3]).magnitude()))
            if (np.linalg.norm(bases[i, :3, 3]-bases[j, :3, 3]) <= .001 and rb <= .3
                    and np.linalg.norm(cameras[i, :3, 3]-cameras[j, :3, 3]) <= .002 and rc <= .3):
                duplicate_of = j; break
        if duplicate_of is None: unique.append(i)
        else: duplicate_pairs.append(dict(excluded_id=train[i]['sample_id'], retained_id=train[duplicate_of]['sample_id']))
    for i in range(len(train)):
        for j in range(i):
            rotation_distance[i, j] = rotation_distance[j, i] = np.degrees(
                Rotation.from_matrix(bases[i, :3, :3].T @ bases[j, :3, :3]).magnitude())
    pixel_centers = np.asarray([[r['centroid_u_px'], r['centroid_v_px']] for r in coverage_rows])
    depth = cameras[:, 2, 3]
    distance = np.sqrt((rotation_distance/15.)**2
        + (np.linalg.norm(bases[:, None, :3, 3]-bases[None, :, :3, 3], axis=2)/.05)**2
        + ((depth[:, None]-depth[None, :])/.05)**2
        + (np.linalg.norm(pixel_centers[:, None]-pixel_centers[None, :], axis=2)/(width*.2))**2)
    chosen = [unique[0]]
    while len(chosen) < min(15, len(unique)):
        remaining = [i for i in unique if i not in chosen]
        chosen.append(max(remaining, key=lambda i: float(distance[i, chosen].min())))
    subsets = {'deduplicated': sorted(unique), 'diverse15': sorted(chosen[:15]), 'diverse12': sorted(chosen[:12])}
    _, closed, singular, failures = solve_opencv_methods(train)
    seeds = {name: row['T_flange_camera'] for name, row in closed.items()}
    seeds['September'] = load_transform_file(args.old_calibration)
    seeds['Identity'] = np.eye(4)
    results = []

    def record(label, samples, x, detail):
        y = consistency_center_transform([np.asarray(s['T_base_flange']) @ x
                                          @ np.asarray(s['T_camera_target']) for s in samples])
        row = dict(label=label, samples=len(samples), sample_ids=[s['sample_id'] for s in samples],
                   training=evaluate(samples, x, y, points),
                   heldout_fixed_board=evaluate(validation, x, y, points),
                   optimization=detail, T_flange_camera=x.tolist())
        if 'optimized_board_pose' in detail:
            fitted_board=np.asarray(detail['optimized_board_pose'])
            row['training_optimized_board']=evaluate(samples,x,fitted_board,points)
            row['heldout_optimized_training_board']=evaluate(validation,x,fitted_board,points)
        results.append(row)
        score = row['heldout_fixed_board']
        print(f'{label}: heldout mean={score["translation_mm"]["mean"]:.2f}mm '
              f'max={score["translation_mm"]["max"]:.2f}mm rotation={score["rotation_deg"]["mean"]:.2f}deg', flush=True)

    for name in ('TSAI', 'DANIILIDIS'):
        record(f'{name}_closed', train, seeds[name], dict(method='closed form', success=True))
    for name, seed in seeds.items():
        x, detail = refine(train, seed, points, model='pose', method='lm')
        record(f'{name}_LM_pose_L2', train, x, detail)
    for model in ('pose', 'pixels'):
        for loss in ('linear', 'huber'):
            x, detail = refine(train, seeds['TSAI'], points, model=model, loss=loss, method='trf')
            record(f'TSAI_TRF_{model}_{loss}', train, x, detail)
        scales=(.5,2.,5.) if model=='pose' else (3.,10.)
        for scale in scales:
            x,detail=refine(train,seeds['TSAI'],points,model=model,loss='huber',method='trf',f_scale=scale)
            record(f'TSAI_TRF_{model}_huber_scale{scale:g}',train,x,detail)
    for name in ('TSAI', 'September'):
        x, detail = refine(train, seeds[name], points, model='pixels', method='lm')
        record(f'{name}_LM_pixels_L2', train, x, detail)
    for name, indices in subsets.items():
        samples = [train[i] for i in indices]
        best, methods, _, _ = solve_opencv_methods(samples)
        record(f'{name}_{best}_closed', samples, methods[best]['T_flange_camera'],
               dict(method='closed form', success=True, selection='training diversity only'))
    # Show held-out image agreement using the fixed board pose fitted on training.
    validation_views=[]
    for s in validation:
        base=np.asarray(s['T_base_flange'])
        if all(np.linalg.norm(base[:3,3]-np.asarray(t['T_base_flange'])[:3,3])>.001
               or np.degrees(Rotation.from_matrix(base[:3,:3].T@np.asarray(t['T_base_flange'])[:3,:3]).magnitude())>.3
               for t in validation_views):
            validation_views.append(s)
    (output/'validation_projections').mkdir(exist_ok=True)
    validation_tiles=[]
    for label in ('DANIILIDIS_closed','TSAI_LM_pose_L2','TSAI_TRF_pose_huber'):
        row=next(r for r in results if r['label']==label)
        x=np.asarray(row['T_flange_camera'])
        y=consistency_center_transform(bases@x@cameras)
        for s in validation_views:
            picture=cv2.imread(str(validation_root/s['image']))
            camera=np.linalg.inv(np.asarray(s['T_base_flange'])@x)@y
            projected,_=cv2.projectPoints(points[np.asarray(s['charuco_ids'])],
                cv2.Rodrigues(camera[:3,:3])[0],camera[:3,3],np.asarray(s['K']),np.asarray(s['D']))
            projected=projected.reshape(-1,2)
            observed=np.asarray(s['image_points'])
            error=float(np.mean(np.linalg.norm(projected-observed,axis=1)))
            for a,b in zip(observed,projected):
                a,b=tuple(np.rint(a).astype(int)),tuple(np.rint(b).astype(int))
                cv2.line(picture,a,b,(0,180,255),1)
                cv2.circle(picture,a,4,(0,255,0),1)
                cv2.drawMarker(picture,b,(0,0,255),cv2.MARKER_CROSS,7,1)
            cv2.rectangle(picture,(0,0),(width,35),(20,20,20),-1)
            cv2.putText(picture,f'{label} | heldout {s["sample_id"]:02d} | mean {error:.1f}px',
                (10,24),cv2.FONT_HERSHEY_SIMPLEX,.65,(255,255,255),1)
            cv2.imwrite(str(output/'validation_projections'/f'{label}_{s["sample_id"]:03d}.jpg'),picture)
            validation_tiles.append(cv2.resize(picture,(384,216)))
    validation_sheet=np.vstack([np.hstack(validation_tiles[i:i+len(validation_views)])
                                for i in range(0,len(validation_tiles),len(validation_views))])
    cv2.imwrite(str(output/'validation_projection_sheet.jpg'),validation_sheet)
    pixels = np.asarray(all_pixels)
    starts=[r for r in results if r['label'].endswith('_LM_pose_L2')]
    reference=np.asarray(starts[0]['T_flange_camera'])
    multistart=dict(seeds=[r['label'] for r in starts],
        all_converged=all(r['optimization']['success'] for r in starts),
        cost_range=[min(r['optimization']['cost'] for r in starts),max(r['optimization']['cost'] for r in starts)],
        max_translation_from_first_mm=max(float(np.linalg.norm(np.asarray(r['T_flange_camera'])[:3,3]-reference[:3,3])*1000) for r in starts),
        max_rotation_from_first_deg=max(float(np.degrees(Rotation.from_matrix(reference[:3,:3].T@np.asarray(r['T_flange_camera'])[:3,:3]).magnitude())) for r in starts))
    coverage = dict(frame_quality=coverage_rows, robot_relative_rotation_singular_values=singular.tolist(),
                    robot_rotation_condition_ratio=float(singular[0]/singular[-1]),
                    pair_robot_rotation_range_deg=[float(rotation_distance[np.triu_indices(len(train), 1)].min()),
                                                   float(rotation_distance.max())],
                    board_depth_range_m=[float(depth.min()), float(depth.max())],
                    corner_pixel_bounds=[pixels.min(axis=0).tolist(), pixels.max(axis=0).tolist()],
                    frame_center_pixel_bounds=[pixel_centers.min(axis=0).tolist(), pixel_centers.max(axis=0).tolist()],
                    duplicate_pairs=duplicate_pairs, unique_pose_count=len(unique),
                    exact_duplicate_image_hash_count=len(train)-len({s['image_sha256'] for s in train}),
                    diversity_subsets={k:[train[i]['sample_id'] for i in v] for k,v in subsets.items()},
                    selection_distance='sqrt((robot rotation/15deg)^2+(position/50mm)^2+(board Z/50mm)^2+(image center/(.2W))^2)')
    report = dict(status='offline_diagnostic_not_promoted', training_dataset=str(train_root),
        validation_dataset=str(validation_root), validation_frames=len(validation),
        validation_independent_pose_count=len(validation_views), coverage=coverage, results=results,
        multistart=multistart,
        failures=failures, robot_commands_sent=False, runtime_calibration_changed=False,
        limitations=['Prior diagnostic reuse of the9 holdout frames precludes treating them as a fresh final acceptance set.',
                     'Huber acts on scaled residual components, not an invented corner-confidence output.',
                     'Low per-frame PnP reprojection is separate from handeye-constrained reprojection.',
                     'Repeated images or poses can bias weighting but do not by themselves imply numerical divergence.'])
    (output/'results.json').write_text(json.dumps(report, indent=2)+'\n')
    fields = ['label', 'samples', 'train_mean_mm', 'heldout_mean_mm', 'heldout_max_mm', 'heldout_mean_deg', 'heldout_max_deg', 'handeye_pixel_rms', 'converged', 'nfev', 'seconds']
    summary = []
    for row in results:
        v, detail = row['heldout_fixed_board'], row['optimization']
        summary.append(dict(label=row['label'], samples=row['samples'], train_mean_mm=row['training']['translation_mm']['mean'],
            heldout_mean_mm=v['translation_mm']['mean'], heldout_max_mm=v['translation_mm']['max'],
            heldout_mean_deg=v['rotation_deg']['mean'], heldout_max_deg=v['rotation_deg']['max'],
            handeye_pixel_rms=v['handeye_constrained_reprojection_px']['rms'],
            converged=detail['success'], nfev=detail.get('nfev'), seconds=detail.get('elapsed_s')))
    with (output/'summary.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f, fieldnames=fields); writer.writeheader(); writer.writerows(summary)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes=plt.subplots(1, 2, figsize=(11, 4.5), layout='constrained')
    for s,r in zip(train, coverage_rows):
        uv=np.asarray(s['image_points']); hull=cv2.convexHull(uv.astype(np.float32)).reshape(-1,2)
        axes[0].plot(*np.vstack((hull,hull[0])).T, alpha=.25)
        axes[0].text(r['centroid_u_px'],r['centroid_v_px'],str(r['sample_id']),fontsize=7)
    axes[0].set(xlim=(0,width),ylim=(height,0),xlabel='u (px)',ylabel='v (px)',title='21 detected corner footprints / sample IDs')
    for r in coverage_rows:
        axes[1].scatter(r['board_z_m']*1000,r['board_tilt_from_optical_axis_deg'],s=25)
        axes[1].annotate(str(r['sample_id']),(r['board_z_m']*1000,r['board_tilt_from_optical_axis_deg']),fontsize=7)
    axes[1].set(xlabel='Camera-board Z (mm)',ylabel='Board tilt from optical axis (deg)',title='Distance and tilt diversity')
    fig.savefig(output/'coverage.png',dpi=160);plt.close(fig)
    columns=''.join(f'<th>{html.escape(x)}</th>' for x in fields)
    body=''.join('<tr>'+''.join(f'<td>{html.escape(f"{r[k]:.3f}" if isinstance(r[k],float) else str(r[k]))}</td>' for k in fields)+'</tr>' for r in summary)
    links=''.join(f'<a href="corners/{s["sample_id"]:03d}.jpg">样本{s["sample_id"]:02d}</a> ' for s in train)
    page=f'''<!doctype html><meta charset="utf-8"><title>手眼标定方法核查</title><style>body{{font:15px sans-serif;margin:24px}}table{{border-collapse:collapse}}td,th{{border:1px solid #ddd;padding:7px}}img{{max-width:100%}}</style><h1>标定初值、Huber、数据分布和角点核查</h1><p>全部离线；原图、采集记录和运行外参未修改。验证9帧来自3个独立姿态，且此前已用于诊断，不是新的最终验收集。</p><p>平移单位mm，旋转单位度。handeye_pixel_rms为固定外参及训练帧板位置中位数、旋转均值约束下的像素误差，不同于单帧PnP误差；JSON另列优化器自身拟合板位置对应的结果。LM测试使用L2，Huber使用TRF，并提供TRF-L2对照。Huber还测试了多个尺度，未依据留出结果选择生产参数。</p><h2>方法对照</h2><table><tr>{columns}</tr>{body}</table><h2>留出姿态投影核对</h2><p>每列一个独立留出姿态，每行一种方法。绿色是实际角点，红色是固定训练板位置与外参预测，黄色连线标出偏差。留出帧没有参与该行外参及板位置拟合。</p><img src="validation_projection_sheet.jpg"><h2>分布</h2><img src="coverage.png"><h2>逐帧角点</h2><p>绿色检测角点、红色PnP投影点、蓝色ID；点击查看原尺寸。</p><p>{links}</p><img src="corner_contact_sheet.jpg">'''
    (output/'report.html').write_text(page, encoding='utf-8')
    print('Saved audit:',output,flush=True)


if __name__ == '__main__':
    main()
