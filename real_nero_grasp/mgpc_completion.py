"""Optional MGPC proposal geometry; measured safety clouds are never replaced."""
import json
import os
from pathlib import Path
import subprocess

import cv2
import numpy as np

from .paths import PROJECT_ROOT


def validate_mgpc_configuration(args):
    if not bool(getattr(args,'mgpc',False)): return
    for name in ('mgpc_python','mgpc_checkpoint'):
        path=Path(getattr(args,name))
        if not path.is_file(): raise FileNotFoundError(f'[mgpc] missing {name}: {path}')
    if not np.isfinite(args.mgpc_timeout) or args.mgpc_timeout<=0:
        raise ValueError('[mgpc] timeout must be finite and positive')


def crop_target_rgb(bgr,mask):
    mask=np.asarray(mask,dtype=bool)
    if mask.shape!=bgr.shape[:2] or not mask.any():
        raise ValueError('MGPC requires a nonempty target mask aligned to its RGB frame')
    y,x=np.where(mask)
    image=np.asarray(bgr).copy();image[~mask]=71
    crop=image[y.min():y.max()+1,x.min():x.max()+1]
    size=int(np.ceil(max(crop.shape[:2])/.85))
    square=np.full((size,size,3),71,np.uint8)
    top,left=(size-crop.shape[0])//2,(size-crop.shape[1])//2
    square[top:top+crop.shape[0],left:left+crop.shape[1]]=crop
    return cv2.resize(square,(224,224))


def _reference_rgb(args,bgr,out_dir,T_base_camera):
    """Use a paired RGB/mask/pose, preferably the archived center reference."""
    for path in sorted((Path(out_dir)/'multiview_raw').glob('center*/metadata.json'),reverse=True):
        metadata=json.loads(path.read_text())
        rgb=cv2.imread(str(path.parent/'color.png'))
        mask=cv2.imread(str(path.parent/'selected_target_mask.png'),0)
        if rgb is not None and mask is not None:
            reference_pose=np.asarray(metadata['T_base_camera'])
            table_report=getattr(args,'_grasp_table_reference_report',None)
            if table_report is not None:
                report=json.loads(Path(table_report).read_text())
                center=next(view for view in report['views']
                    if view['view']=='center' and Path(view['source']).resolve()==path.parent.resolve())
                # The fused target's world mapping already includes this same
                # correction. Both camera poses must share that world frame
                # before converting the points into the paired RGB camera.
                reference_pose=np.asarray(center['T_registration']) @ reference_pose
            return rgb,mask,reference_pose,str(path.parent)
    mask=cv2.imread(str(Path(out_dir)/'selected_target_mask.png'),0)
    if mask is None: raise RuntimeError('[mgpc] original capture target mask unavailable; completion aborted')
    return bgr,mask,np.asarray(T_base_camera),str(out_dir)


def complete_graspnet_target(target_cloud,T_base_camera,bgr,args,out_dir,*,runner=None):
    """Only return a new proposal cloud. Fail closed instead of a silent fallback."""
    if not bool(getattr(args,'mgpc',False)): return target_cloud
    from tools.compare_mgpc_reconstruction import fps,normalize,denormalize,transform,save_cloud
    import open3d as o3d
    directory=Path(out_dir)/'mgpc'
    directory.mkdir(parents=True,exist_ok=True)
    # Preserve the venv launcher path: resolving its symlink would start the
    # base Conda interpreter and lose the isolated CLIP/Transformers packages.
    python=Path(args.mgpc_python).expanduser().absolute()
    checkpoint=Path(args.mgpc_checkpoint).resolve()
    if not python.is_file() or not checkpoint.is_file():
        raise RuntimeError('[mgpc] dedicated Python or checkpoint missing; no grasp proposals generated')
    rgb,mask,T_base_reference,source=_reference_rgb(args,bgr,out_dir,T_base_camera)
    T_base_camera=np.asarray(T_base_camera,dtype=float).reshape(4,4)
    points=np.asarray(target_cloud.points,dtype=float).copy()
    reference=transform(points,np.linalg.inv(T_base_reference) @ T_base_camera)
    sampled,indices=fps(reference)
    normalized,normalization=normalize(sampled)
    np.save(directory/'input_normalized.npy',normalized)
    np.save(directory/'sample_indices.npy',indices)
    np.save(directory/'measured_camera.npy',points)
    if not cv2.imwrite(str(directory/'target_rgb.png'),crop_target_rgb(rgb,mask)):
        raise OSError('Could not save MGPC RGB crop')
    category=str(args.mgpc_category or getattr(args,'_active_target_label',None) or args.target_class or '').strip()
    if category.lower() in ('','manual','target','unknown','click','point'):
        raise RuntimeError('[mgpc] supply an explicit --mgpc-category for the selected object')
    request=dict(repository=str(PROJECT_ROOT/'third_party/MGPC'),checkpoint=str(checkpoint),
        category=category,seed=int(args.mgpc_seed),normalization=normalization,
        T_base_camera=T_base_camera.tolist(),T_base_reference=T_base_reference.tolist(),
        rgb_source=source,measured_points=len(points),role='proposal_only_measured_safety_unchanged')
    request_path=directory/'request.json'
    request_path.write_text(json.dumps(request,indent=2)+'\n')
    # Remove only this attempt's old worker outputs, so an interrupted/failed
    # retry cannot accidentally consume a stale prediction.
    prediction_path=directory/'prediction_normalized.npy'
    prediction_path.unlink(missing_ok=True)
    (directory/'worker_result.json').unlink(missing_ok=True)
    environment=os.environ.copy()
    environment.update(PYTHONNOUSERSITE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
        OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
    environment.pop('PYTHONPATH',None)
    environment.pop('PYTHONHOME',None)
    command=[str(python),str(PROJECT_ROOT/'tools/run_mgpc_completion.py'),'--request',str(request_path.resolve())]
    print(f'[mgpc] completing {len(points)} measured points; category={category}; RGB={source}',flush=True)
    try:
        with (directory/'worker.log').open('w') as log:
            (subprocess.run if runner is None else runner)(command,check=True,timeout=float(args.mgpc_timeout),
                stdout=log,stderr=subprocess.STDOUT,env=environment)
    except (subprocess.SubprocessError,OSError) as exc:
        raise RuntimeError(f'[mgpc] completion failed; grasp aborted. See {directory / "worker.log"}: {exc}') from exc
    prediction=np.load(prediction_path,allow_pickle=False)
    if prediction.shape!=(8192,3) or not np.isfinite(prediction).all():
        raise RuntimeError('[mgpc] invalid completion output; grasp aborted')
    predicted_reference=denormalize(prediction,normalization)
    result=transform(predicted_reference,np.linalg.inv(T_base_camera) @ T_base_reference)
    # Reject a grossly displaced/expanded output before it reaches GraspNet.
    from scipy.spatial import cKDTree
    observed_distance=cKDTree(result).query(points)[0]
    if float(np.quantile(observed_distance,.95))>.025:
        raise RuntimeError('[mgpc] prediction does not cover measured surface within 25 mm; grasp aborted')
    np.save(directory/'prediction_camera.npy',result)
    np.save(directory/'prediction_base.npy',transform(result,T_base_camera))
    save_cloud(directory/'prediction_camera.ply',result,np.tile([1.,.54,.20],(len(result),1)))
    proposal = result
    feedback = getattr(args, '_active_feedback_session', None)
    mixture_report = None
    if bool(getattr(args, 'active_grasp_feedback', False)):
        if feedback is None:
            raise RuntimeError('[mgpc] feedback requires an active evidence session')
        from .perception_evidence import measured_first_proposal
        feedback.prediction_base = transform(result, T_base_camera)
        combined, source, states, selected, mixture_report = measured_first_proposal(
            transform(points, T_base_camera), feedback.prediction_base, feedback.observations,
            tolerance=args.feedback_support_distance, duplicate_distance=args.feedback_duplicate_distance,
            prediction_ratio=args.feedback_prediction_ratio)
        proposal = transform(combined, np.linalg.inv(T_base_camera))
        # Preserve measured camera XYZ bit-for-bit instead of roundtripping it.
        proposal[:len(points)] = points
        np.savez_compressed(directory/'proposal_evidence.npz', points_camera=proposal, source=source,
                            prediction_states=states, selected_prediction_indices=selected)
        (directory/'proposal_evidence.json').write_text(json.dumps(mixture_report, indent=2)+'\n')
    output=o3d.geometry.PointCloud(o3d.utility.Vector3dVector(proposal))
    colors = np.tile([1.,.54,.20],(len(proposal),1))
    if mixture_report is not None and len(target_cloud.colors) == len(points):
        colors[:len(points)] = np.asarray(target_cloud.colors)
    output.colors=o3d.utility.Vector3dVector(colors)
    report=dict(request,output_points=len(result),observed_to_prediction_p95_m=float(np.quantile(observed_distance,.95)),
        safety_target='original measured target',safety_environment='original measured environment')
    (directory/'completion.json').write_text(json.dumps(report,indent=2)+'\n')
    if mixture_report is not None:
        report['proposal_evidence'] = mixture_report
        (directory/'completion.json').write_text(json.dumps(report, indent=2)+'\n')
    args._mgpc_completion_info=report
    generator=str(getattr(args,'grasp_generator','graspnet'))
    print(f'[mgpc] {len(output.points)} proposal points -> {generator}; '
          f'measured_first={mixture_report is not None}; occupancy/aperture/path use measured data',flush=True)
    return output
