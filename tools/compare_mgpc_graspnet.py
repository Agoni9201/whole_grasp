#!/usr/bin/env python3
"""Four-way offline GraspNet replay; no device objects or execution entrypoints.

Network inference, filter audit, and HTML rendering are separate stages. The
common measured collision scene never depends on MGPC output. Numerical plans
are diagnostics, not authorization to execute saved robot trajectories.
"""
import argparse
import contextlib
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.compare_mgpc_reconstruction import transform, save_cloud

CASES = [('A', 'single', False, '单视角实测'), ('B', 'multi', False, '多视角融合实测'),
         ('C', 'single', True, '单视角 → MGPC'), ('D', 'multi', True, '多视角融合 → MGPC')]
STAGES = ['network_collision', 'target_empty', 'nms', 'world_down_angle', 'center',
          'selection_budget', 'physical_aperture', 'table_guard', 'endpoint_ik',
          'workflow_failed', 'workflow_not_checked', 'workflow_pass']


def plain(value):
    """Strict JSON; distinguish unavailable data from numeric infinity."""
    if isinstance(value, dict):
        return {str(k): plain(v) for k,v in value.items() if not str(k).startswith('_')
                and (k != 'grasp' or isinstance(v,(list,tuple,np.ndarray,dict)))}
    if isinstance(value, (tuple,list)): return [plain(v) for v in value]
    if isinstance(value, np.ndarray): return plain(value.tolist())
    if isinstance(value, np.generic): return plain(value.item())
    if isinstance(value, float) and not np.isfinite(value): return None
    return value


def write_json(path, data):
    path.write_text(json.dumps(plain(data), ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def cloud(points, colors=None):
    import open3d as o3d
    result = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.asarray(points,dtype=float)))
    if colors is not None: result.colors = o3d.utility.Vector3dVector(np.asarray(colors,dtype=float))
    return result


def baseline_imports():
    for folder in ('pointnet2','models','dataset','utils','graspnetAPI'):
        path = str(ROOT/'graspnet-baseline'/folder)
        if path not in sys.path: sys.path.insert(0,path)
    # Compatibility required by the deployed graspnetAPI, matching runtime.
    for name,value in [('float',float),('int',int),('bool',bool)]:
        if name not in np.__dict__: setattr(np,name,value)


def load_runtime_args(manifest):
    from real_nero_grasp import runtime as r
    saved = sys.argv
    try:
        sys.argv = json.loads((Path(manifest['source_run'])/'experiment_result.json').read_text())['command']
        args = r.parse_args()
    finally: sys.argv = saved
    args.execute = False
    return r,args


def input_indices(count, num_points, mode='fixed'):
    """Offline ablation only: all preserves order and never pads or drops points."""
    if count <= 0 or num_points <= 0:
        raise ValueError('positive point counts required')
    if mode == 'all':
        return np.arange(count)
    if mode != 'fixed':
        raise ValueError(f'unknown sampling mode: {mode}')
    if count >= num_points:
        return np.random.choice(count, num_points, replace=False)
    return np.r_[np.arange(count), np.random.choice(count, num_points-count)]


def infer(opts):
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable in this process; offline inference requires GPU access')
    baseline_imports()
    from graspnet import GraspNet, pred_decode
    from real_nero_grasp.grasp_workflow import _prepare_graspnet_target_cloud
    manifest = json.loads((opts.input_dir/'manifest.json').read_text())
    _, args = load_runtime_args(manifest)
    T = np.asarray(manifest['T_base_camera'])
    torch.set_num_threads(4)
    net = GraspNet(input_feature_dim=0, num_view=300, num_angle=12, num_depth=4,
        cylinder_radius=.05, hmin=-.02, hmax_list=[.01,.02,.03,.04], is_training=False)
    checkpoint = ROOT/'logs/log_rs/checkpoint-rs.tar'
    state = torch.load(checkpoint,map_location='cpu',weights_only=True)
    net.load_state_dict(state['model_state_dict'],strict=True)
    del state
    net = net.cuda().eval()
    for letter,name,predicted,title in CASES:
        directory=opts.output_dir/letter
        directory.mkdir(exist_ok=True)
        points=np.load(opts.input_dir/name/('prediction_base.npy' if predicted else 'observed_base.npy'))
        native=transform(points,np.linalg.inv(T))
        prepared=_prepare_graspnet_target_cloud(cloud(native,np.zeros_like(native)),args,T_base_camera=T)
        prepared=np.asarray(prepared.points)
        np.save(directory/'network_target_camera.npy',prepared)
        for seed in opts.seeds:
            np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
            mode=getattr(opts,'sampling_mode','fixed')
            ids=input_indices(len(prepared),args.grasp_num_points,mode)
            tensor=torch.from_numpy(prepared[ids][None].astype(np.float32)).cuda()
            # Optional warmup/repeats for the input-count ablation. Existing replays
            # retain their default single inference and sampling sequence.
            repeats=getattr(opts,'timing_repeats',1)
            if repeats > 1:
                with torch.inference_mode(): net({'point_clouds':tensor})
            torch.cuda.synchronize()
            started=time.monotonic()
            with torch.inference_mode():
                output=net({'point_clouds':tensor})
                objectness=output['objectness_score'][0].argmax(0).cpu().numpy()
                predictions=pred_decode(output)[0].cpu().numpy()
                sa1_indices=output['sa1_inds'][0].cpu().numpy()
            if not np.isfinite(predictions).all():
                raise RuntimeError('Nonfinite network prediction')
            if sa1_indices.min() < 0 or sa1_indices.max() >= len(ids):
                raise RuntimeError('Backbone sampled an out-of-range input index')
            torch.cuda.synchronize()
            first_seconds=time.monotonic()-started
            timings=[first_seconds]
            for _ in range(repeats-1):
                started=time.monotonic()
                with torch.inference_mode():
                    repeat_output=net({'point_clouds':tensor})
                    repeat_predictions=pred_decode(repeat_output)[0].cpu().numpy()
                torch.cuda.synchronize()
                timings.append(time.monotonic()-started)
                if not np.allclose(predictions,repeat_predictions,rtol=1e-5,atol=1e-6):
                    raise RuntimeError('Repeated identical-input inference changed candidates')
            np.save(directory/f'raw_seed{seed}.npy',predictions)
            np.save(directory/f'sampled_indices_seed{seed}.npy',ids)
            np.save(directory/f'objectness_seed{seed}.npy',objectness)
            write_json(directory/f'input_seed{seed}.json',dict(seed=seed,title=title,
                input_frame='center aligned RGB camera', original_points=len(points),
                prepared_points=len(prepared),sampled_points=len(ids),decoded_candidates=len(predictions),
                network_seed_points=len(objectness),objectness_foreground=int((objectness==1).sum()),
                sampling_mode=mode,unique_input_indices=len(np.unique(ids)),
                sa1_points=len(sa1_indices),sa1_unique_indices=len(np.unique(sa1_indices)),
                input_focus=vars(args).get('_graspnet_target_focus_info'),seconds=first_seconds,
                timing_seconds=timings,median_seconds=float(np.median(timings))))
            print(f'{letter} seed={seed}: decoded {len(predictions)} candidates',flush=True)
    write_json(opts.output_dir/'inference.json',dict(seeds=opts.seeds,checkpoint=str(checkpoint),
        checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        input_feature_dim=0,colors_used_by_network=False,network_points=args.grasp_num_points,
        sampling_mode=getattr(opts,'sampling_mode','fixed'),
        camera_transform=T,source=str(opts.input_dir.resolve()),device=torch.cuda.get_device_name(),
        torch=torch.__version__,note='Same production architecture, checkpoint, component/focus filters and point count. Seed controls sampling; CUDA operations are not claimed bitwise deterministic.'))


def detector_witness(detector, grasp, approach_dist):
    """Exact point masks used by official ModelFreeCollisionDetector.detect."""
    xyz=(detector.scene_points-np.asarray(grasp.translation)) @ np.asarray(grasp.rotation_matrix)
    x,y,z=xyz.T; width=float(grasp.width);height=float(grasp.height);depth=float(grasp.depth)
    fw,fl=detector.finger_width,detector.finger_length
    band=(z > -height/2)&(z < height/2)
    length=(x > depth-fl)&(x < depth)
    outer=(y > -(width/2+fw))&(y < width/2+fw)
    masks={'left_finger':band&length&(y > -(width/2+fw))&(y < -width/2),
           'right_finger':band&length&(y < width/2+fw)&(y > width/2),
           'bottom':band&outer&(x <= depth-fl)&(x > depth-fl-fw),
           'approach_corridor':band&outer&(x <= depth-fl-fw)&(x > depth-fl-fw-max(approach_dist,fw))}
    return {name:detector.scene_points[mask] for name,mask in masks.items()}


def matched_ids(array, raw):
    """NMS preserves rows; map them back to stable raw IDs without rounding."""
    lookup={row.tobytes():i for i,row in enumerate(np.ascontiguousarray(raw,dtype=np.float64))}
    return [lookup[row.tobytes()] for row in np.ascontiguousarray(array,dtype=np.float64)]


def prepare_environment(opts,manifest,r,args):
    """Common measured map with calibrated self mask; no prediction geometry."""
    import cv2
    from real_nero_grasp.camera_self_mask import calibrated_self_mask
    capture_root=next(Path(manifest['source_run']).glob('*/active_fusion_registration.json')).parent
    registration=json.loads((capture_root/'active_fusion_registration.json').read_text())
    corrections={item['name']:item['background_transform'] for item in registration['views']}
    views={}
    for path in (capture_root/'multiview_raw').glob('*/metadata.json'):
        meta=json.loads(path.read_text());folder=path.parent
        rgb=cv2.imread(str(folder/'color.png'))
        sensor=json.loads((folder/'capture_meta.json').read_text())
        mask,mask_report=calibrated_self_mask(rgb,ROOT/'config/2026-10-22_camera_self_mask.json',sensor)
        with np.load(folder/'clouds.npz') as arrays:
            points=arrays['obstacle_camera_points'].copy()
        intr=meta['intrinsics'];u=np.rint(intr['fx']*points[:,0]/points[:,2]+intr['cx']).astype(int)
        v=np.rint(intr['fy']*points[:,1]/points[:,2]+intr['cy']).astype(int)
        inside=(u>=0)&(u<mask.shape[1])&(v>=0)&(v<mask.shape[0])
        keep=np.ones(len(points),bool);keep[inside]=mask[v[inside],u[inside]]==0
        base=transform(points[keep],meta['T_base_camera'])
        if meta['view_label'] in corrections: base=transform(base,corrections[meta['view_label']])
        views[meta['view_label']]=dict(meta=meta,cloud=cloud(base),mask=mask_report)
    measured=np.load(opts.input_dir/'multi/observed_base.npy')
    center=np.median(measured,axis=0)
    order=[n for n in ('initial','center','side1','side2') if n in views]
    labels=[n if not n.startswith('side') else 'side_capture_'+n[4:] for n in order]
    environment=r.fuse_dynamic_environment_clouds(labels,[views[n]['cloud'] for n in order],args,
        target_base=measured,target_center_base=center,map_id='mgpc_common_measured',generation=1)
    T=np.asarray(manifest['T_base_camera'])
    target_camera=cloud(transform(measured,np.linalg.inv(T)))
    scene=r.prepare_motion_safety_scene(target_camera,None,T,args,map_id='mgpc_common_measured',environment_map=environment)
    save_cloud(opts.output_dir/'environment_base.ply',scene.environment_base)
    np.savez_compressed(opts.output_dir/'safety_scene.npz',environment=scene.environment_base,
                        target=scene.target_base,full=scene.full_scene_base)
    write_json(opts.output_dir/'environment.json',dict(points=len(scene.environment_base),
        target_points=len(scene.target_base),source_views=order,self_mask='config/2026-10-22_camera_self_mask.json',
        target_source='same measured multi/observed_base.npy for all four groups',
        side_environment_transform='saved background_transform',
        prediction_used_for_safety=False))
    q=np.asarray(views['center']['meta']['capture_joints'])
    return scene,environment,target_camera,q


class NumericalArm:
    """Only pure FK/pose methods; no SDK instance, CAN/socket, or motion methods."""
    def __init__(self,r,args,q): self.r,self.args,self.q=r,args,np.asarray(q)
    def get_joint_angles(self): return self.q.copy()
    def get_tcp_pose(self): return self.r._modeled_tcp_pose_for_joints(self.q,self.args)
    def get_tcp2flange_pose(self,pose):
        return self.r._as_pose_list(self.r._make_T_from_pose(pose) @ np.linalg.inv(self.r._make_T_from_pose(self.args.tcp_offset)))


def evaluate(opts):
    baseline_imports()
    from graspnetAPI import GraspGroup
    from collision_detector import ModelFreeCollisionDetector
    from scipy.spatial import cKDTree
    from scipy.spatial.transform import Rotation
    manifest=json.loads((opts.input_dir/'manifest.json').read_text())
    r,args=load_runtime_args(manifest)
    args.grasp_preflight_max_candidates=opts.preflight_topk
    # Keep production target-valid-plan stopping, but label all untested rows.
    T=np.asarray(manifest['T_base_camera'])
    scene,environment,measured_target,q=prepare_environment(opts,manifest,r,args)
    arm=NumericalArm(r,args,q);current=arm.get_tcp_pose()
    full_camera=transform(scene.full_scene_base,np.linalg.inv(T))
    isolated_radius=float(getattr(args,'motion_collision_isolated_radius',.02))
    isolated_support=int(getattr(args,'motion_collision_isolated_min_support',8))
    support=cKDTree(full_camera).query_ball_point(full_camera,isolated_radius,return_length=True)
    supported=np.asarray(support)>=isolated_support
    filtered=full_camera[supported] if np.any(supported) else full_camera
    detector=ModelFreeCollisionDetector(filtered,voxel_size=.01)
    target_detector=ModelFreeCollisionDetector(np.asarray(measured_target.points),voxel_size=.005)
    np.save(opts.output_dir/'network_collision_voxels_base.npy',transform(detector.scene_points,T))
    strict_dist=max(args.graspnet_collision_approach_dist,.01);relaxed_dist=max(min(strict_dist,.03),.01)
    overview=[]
    for letter,name,predicted,title in CASES:
        directory=opts.output_dir/letter
        for seed in opts.seeds:
            result_path=directory/f'audit_seed{seed}.json'
            if result_path.exists() and not opts.force and not opts.rebuild_audits:
                result=json.loads(result_path.read_text());overview.append(result['summary'])
                print(f'{letter} seed={seed}: reuse completed audit',flush=True);continue
            raw=np.load(directory/f'raw_seed{seed}.npy');gg=GraspGroup(raw.copy())
            records=[dict(raw_id=i,score=float(g.score),width=float(g.width),height=float(g.height),
                depth=float(g.depth),translation_camera=g.translation,rotation_camera=g.rotation_matrix,
                translation_base=transform(np.asarray(g.translation)[None],T)[0],status='pending',checks={}) for i,g in enumerate(gg)]
            alive=np.ones(len(raw),bool)
            strict=[];relaxed=[];ious=[]
            for start in range(0,len(raw),32):
                group=gg[start:start+32]
                mask,parts=detector.detect(group,approach_dist=strict_dist,collision_thresh=.01,return_ious=True)
                strict.extend(mask.tolist());ious.extend(np.asarray(parts).T.tolist())
                relaxed.extend(detector.detect(group,approach_dist=relaxed_dist,collision_thresh=.02).tolist())
            strict=np.asarray(strict,bool);relaxed=np.asarray(relaxed,bool)
            use_relaxed=bool(len(raw) and strict.all())
            selected_collision=relaxed if use_relaxed else strict
            active_dist=relaxed_dist if use_relaxed else strict_dist
            for i,rec in enumerate(records):
                rec['checks']['network_collision']=dict(strict_ok=not strict[i],relaxed_ok=not relaxed[i],
                    selected_policy='relaxed_all_strict_failed' if use_relaxed else 'strict',
                    strict_iou_parts=dict(zip(['global','left_finger','right_finger','bottom','approach_corridor'],ious[i])))
                if selected_collision[i]: rec['status']='network_collision'
            alive &= ~selected_collision
            counts=[dict(stage='decoded',remaining=len(raw)),dict(stage='network_collision',remaining=int(alive.sum()))]
            # Physical evidence is shared across groups, independent of predicted surfaces.
            ids=np.flatnonzero(alive)
            if len(ids):
                _,empty=target_detector.detect(gg[ids],approach_dist=.01,collision_thresh=float('inf'),
                    return_empty_grasp=True,empty_thresh=.01)
                for i,failed in zip(ids,empty):
                    records[i]['checks']['target_empty']=dict(ok=not failed,reference='common measured target')
                    if failed: alive[i]=False;records[i]['status']='target_empty'
            counts.append(dict(stage='target_empty',remaining=int(alive.sum())))
            ids=np.flatnonzero(alive)
            if len(ids):
                reduced=gg[ids].nms();reduced.sort_by_score()
                retained=set(matched_ids(reduced.grasp_group_array,raw))
                for i in ids:
                    if i not in retained: alive[i]=False;records[i]['status']='nms'
            counts.append(dict(stage='nms',remaining=int(alive.sum())))
            target_camera=np.load(directory/'network_target_camera.npy')
            center=np.median(target_camera,axis=0)
            down_camera=T[:3,:3].T @ np.array([0.,0.,-1.])
            axis=r._axis_vector(args.graspnet_approach_axis)
            ranked=[]
            for i in np.flatnonzero(alive):
                i=int(i)
                g=gg[i];align=float(np.clip(np.dot(np.asarray(g.rotation_matrix)@axis,down_camera),-1,1))
                records[i]['checks']['world_down_angle']=dict(angle_deg=float(np.degrees(np.arccos(align))),limit_deg=args.grasp_angle_deg)
                if align < np.cos(np.deg2rad(args.grasp_angle_deg)):
                    alive[i]=False;records[i]['status']='world_down_angle';continue
                distance=float(np.linalg.norm(np.asarray(g.translation)-center))
                # No obstacle-direction preferences: same explicit neutral value across groups.
                priority=.28*max(align,0)+.30*np.clip(1-distance/.12,0,1)+.27+.05*float(g.score)
                records[i]['priority']=float(priority);records[i]['center_dist']=distance
                ranked.append(i)
            counts.append(dict(stage='world_down_angle',remaining=len(ranked)))
            close=[i for i in ranked if records[i]['center_dist']<.12]
            if close:
                for i in set(ranked)-set(close): records[i]['status']='center';alive[i]=False
                ranked=close
            counts.append(dict(stage='center',remaining=len(ranked)))
            ranked.sort(key=lambda i:(-records[i]['priority'],-records[i]['score'],i))
            chosen=ranked[:args.manual_topk]
            for i in ranked[args.manual_topk:]: records[i]['status']='selection_budget'
            rows=[dict(idx=i,grasp=gg[i],translation=gg[i].translation,raw_score=gg[i].score,
                       priority=records[i]['priority']) for i in chosen]
            rows=r.annotate_grasp_candidate_physical_aperture(rows,measured_target,T,current,args)
            rows=r.annotate_grasp_candidate_reachability(rows,T,current,args,arm=arm)
            # Preserve collision witnesses from every route the planner actually tries.
            preflighted=0
            for row in rows:
                raw_id=row['idx'];witnesses=[]
                cache_path=directory/f'plan_seed{seed}_raw{raw_id}.json'
                cheap_ok=(row.get('physical_aperture_ok') and row.get('table_guard_ok') is True
                          and row.get('reachability_ok',False))
                if cheap_ok and preflighted >= opts.preflight_topk:
                    row['workflow_preflight_ok']=False
                    row['workflow_preflight_reason']='not preflighted due to offline per-case planning budget'
                    records[raw_id]['production']=plain(row)
                    records[raw_id]['status']='workflow_not_checked'
                    continue
                if cheap_ok: preflighted+=1
                if cache_path.exists() and not opts.force:
                    cached=json.loads(cache_path.read_text())
                    row.update(cached['production'])
                    records[raw_id].update(cached)
                    print(f'{letter} seed={seed} raw={raw_id}: reuse numerical plan audit',flush=True)
                    continue
                original_metrics=r._motion_joint_path_metrics
                def record_metrics(q0,q1,context,settings):
                    metrics=original_metrics(q0,q1,context,settings)
                    if metrics.get('margin',0)<0 or not witnesses:
                        fraction=metrics.get('sample_index',0)/max(metrics.get('samples',2)-1,1)
                        q_hit=np.asarray(q0)+(np.asarray(q1)-np.asarray(q0))*fraction
                        witness=dict(phase=context.phase,metrics=metrics,q0=q0,q1=q1,q_hit=q_hit,
                            robot_capsules=r._motion_robot_capsules(q_hit,settings,
                                open_gripper=context.phase in ('free_open','target_approach','released_retreat','released_free'),
                                closed_gripper_width=getattr(context,'closed_gripper_width',None)))
                        witnesses.append(witness)
                    return metrics
                r._motion_joint_path_metrics=record_metrics
                try:
                    checked=r.annotate_grasp_candidate_workflow_preflight([row],arm,args,measured_target,None,T,current,
                        f'offline_{letter}_s{seed}_r{raw_id}',motion_environment_map=environment)
                finally: r._motion_joint_path_metrics=original_metrics
                checked_row=checked[0]
                row.update(checked_row)
                records[raw_id]['production']=plain(row)
                records[raw_id]['path_witnesses']=plain(witnesses)
                if not row.get('physical_aperture_ok'): status='physical_aperture'
                elif row.get('table_guard_ok') is not True: status='table_guard'
                elif not row.get('reachability_ok',False): status='endpoint_ik'
                elif row.get('workflow_preflight_ok'): status='workflow_pass'
                elif 'not preflighted' in row.get('workflow_preflight_reason',''): status='workflow_not_checked'
                else: status='workflow_failed'
                records[raw_id]['status']=status
                write_json(cache_path,{key:records[raw_id][key] for key in ('production','path_witnesses','status')})
                print(f'{letter} seed={seed} raw={raw_id} status={status}',flush=True)
            # A consistent per-case planning budget; others remain visibly unchecked.
            for rec in records:
                if rec['status']=='pending': rec['status']='workflow_not_checked'
            for rec,g in zip(records,gg):
                # Base TCP preview is the production command pose, not raw camera axes.
                row=next((row for row in rows if row['idx']==rec['raw_id']),None)
                sign=float(row.get('tcp_width_sign',1.)) if row else 1.
                shift=r.grasp_candidate_width_shift(row,sign) if row and row.get('physical_aperture_ok') else 0.
                pre,grasp,lift=r.grasp_to_tcp_poses(g,T,args,current_tcp_pose=current,width_sign=sign,width_shift=shift)
                if row and row.get('workflow_preflight_ok'):
                    pre,grasp,lift=[row[key] for key in ('pre_tcp_preview','grasp_tcp_preview','lift_tcp_preview')]
                rec['tcp_preview']=dict(pre=pre,grasp=grasp,lift=lift)
                if rec['status']=='network_collision':
                    parts=detector_witness(detector,g,active_dist)
                    rec['network_witness']={part:transform(points,T).tolist() for part,points in parts.items() if len(points)}
            summary=dict(case=letter,title=title,seed=seed,decoded=len(records),counts=counts,
                status_counts={stage:sum(rec['status']==stage for rec in records) for stage in STAGES},
                workflow_pass_ids=[rec['raw_id'] for rec in records if rec['status']=='workflow_pass'])
            result=dict(summary=summary,candidates=records,settings=dict(strict_approach_m=strict_dist,
                relaxed_approach_m=relaxed_dist,world_down_limit_deg=args.grasp_angle_deg,
                manual_topk=args.manual_topk,preflight_topk=opts.preflight_topk))
            write_json(result_path,result);overview.append(summary)
            write_json(opts.output_dir/'summary.json',overview)
    write_json(opts.output_dir/'summary.json',overview)
    write_json(opts.output_dir/'evaluation_settings.json',dict(args=vars(args),T_base_camera=T,
        start_joints=q,start_tcp=current,seeds=opts.seeds,hardware_connected=False,
        comparison='All groups use the same measured target/environment for occupancy, aperture and full-path checks. Completion changes only proposal geometry.',
        limitation='Saved scene and numerical IK, not live robot state or grasp success. Selection budget is explicitly recorded.'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir',type=Path,default=ROOT/'artifacts/mgpc_compare')
    parser.add_argument('--output-dir',type=Path,default=ROOT/'artifacts/mgpc_graspnet_compare')
    parser.add_argument('--seeds',type=int,nargs='+',default=[0,1,2])
    parser.add_argument('--preflight-topk',type=int,default=12)
    parser.add_argument('--sampling-mode',choices=['fixed','all'],default='fixed',help='Offline input-count ablation; all keeps every prepared point without padding')
    parser.add_argument('--timing-repeats',type=int,default=1)
    parser.add_argument('--stage',choices=['infer','evaluate','render','all'],default='all')
    parser.add_argument('--force',action='store_true')
    parser.add_argument('--rebuild-audits',action='store_true',help='Rebuild filter records while reusing completed numerical plan audits')
    opts=parser.parse_args();opts.output_dir.mkdir(parents=True,exist_ok=True)
    if opts.stage in ('infer','all'): infer(opts)
    if opts.stage in ('evaluate','all'): evaluate(opts)
    if opts.stage in ('render','all'):
        from tools.render_mgpc_graspnet import render
        render(opts)


if __name__=='__main__': main()
