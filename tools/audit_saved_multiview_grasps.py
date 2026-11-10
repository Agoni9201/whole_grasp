"""Audit archived multiview captures and saved candidate lines without hardware.

Usage: python tools/audit_saved_multiview_grasps.py RUN_DIR --output-json PATH
The ablation reuses current geometry/defaults plus the saved command. It tests
endpoint body distances, not full IK, path feasibility, or contact retention.
"""
import argparse
import contextlib
import copy
import io
import json
import pathlib
import sys
from types import SimpleNamespace

import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from real_nero_grasp.cli import parse_args
from real_nero_grasp.grasp_geometry import grasp_to_tcp_poses
from real_nero_grasp.grasp_candidates import annotate_grasp_candidate_physical_aperture, grasp_candidate_width_shift
from real_nero_grasp.transforms import _make_T_from_pose, _point_segment_distances
from real_nero_grasp import pointcloud_fusion as fusion

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('run_dir', type=pathlib.Path)
parser.add_argument('--output-json', type=pathlib.Path, required=True)
request = parser.parse_args()
run = request.run_dir.resolve()
experiment = json.loads((run/'experiment_result.json').read_text())
sys.argv = list(experiment['command'])
if '--config' in sys.argv:
    config_index = sys.argv.index('--config') + 1
    sys.argv[config_index] = str(ROOT / sys.argv[config_index])
args = parse_args()
args.execute = False
out = next(run.glob('*/active_fusion_registration.json')).parent
archive = {}
for path in (out/'multiview_raw').glob('*/metadata.json'):
    meta = json.loads(path.read_text())
    archive[meta['view_label']] = (path.parent, meta)
T_base_camera = np.array(archive['initial'][1]['T_base_camera'])

def read(name):
    return o3d.io.read_point_cloud(str(out/(name+'.ply')))

target_camera = read('graspnet_target_full')
fused_base = fusion._apply_T_points(np.asarray(target_camera.points), T_base_camera)
sources = {'fused_target': fused_base}
for name in ['center', 'side1']:
    sources[name+'_target_registered'] = np.asarray(read('active_'+name+'_target_base_registered').points)
for name, (path, _) in archive.items():
    sources[name+'_obstacle_raw'] = np.asarray(o3d.io.read_point_cloud(str(path/'obstacle_base.ply')).points)

lines = o3d.io.read_line_set(str(out/'graspnet_candidate_grippers.ply'))
points = np.asarray(lines.points).reshape(-1, 6, 3)
rows = []
for idx, pts in enumerate(points):
    approach = pts[1]-pts[0]
    approach /= np.linalg.norm(approach)
    width = pts[3]-pts[2]
    gap = np.linalg.norm(width)
    width /= gap
    width -= approach * np.dot(width, approach)
    width /= np.linalg.norm(width)
    grasp = SimpleNamespace(translation=pts[0], rotation_matrix=np.column_stack([approach, width, np.cross(approach,width)]), width=gap, depth=0.0, score=1.0)
    rows.append({'idx':idx,'grasp':grasp})

def body(pose, config):
    T = _make_T_from_pose(pose)
    flange = (T @ np.linalg.inv(_make_T_from_pose(config.tcp_offset)))[:3,3]
    tcp = T[:3,3]
    length = np.linalg.norm(tcp-flange)
    direction = (tcp-flange)/length
    radius = config.motion_path_tool_radius
    end = tcp-direction*min(config.motion_target_tool_contact_length+radius,length)
    return flange,end,radius

def clearance(xyz, cap):
    a,b,r = cap
    dist = _point_segment_distances(xyz,a,b)-r
    i = int(np.argmin(dist))
    return {'mm':float(dist[i]*1000),'point':xyz[i].tolist(), 'penetrating_points':int(np.sum(dist<0))}

results = {'source_run':str(run),'method':'Recovered candidate orientation/width from saved LineSet; endpoint-only body distances, no IK or robot commands. Aperture rerun with current code and saved command. A passing body distance is not a full-plan pass.', 'advance':[]}
for advance in [0.0,0.015,0.030]:
    config = copy.copy(args)
    config.grasp_advance = advance
    with contextlib.redirect_stdout(io.StringIO()):
        annotated = annotate_grasp_candidate_physical_aperture(rows,target_camera,T_base_camera,np.zeros(6),config)
    item = {'advance_mm':advance*1000,'candidates':[]}
    for row in annotated:
        rec = {'idx':row['idx'],'aperture_ok':row['physical_aperture_ok'],'aperture_reason':row['physical_aperture_reason']}
        shift = grasp_candidate_width_shift(row,1.0) if row['physical_aperture_ok'] else 0.0
        _,pose,_ = grasp_to_tcp_poses(row['grasp'],T_base_camera,config,np.zeros(6),width_shift=shift)
        rec['shift_mm'] = shift*1000
        rec['pose'] = np.asarray(pose).tolist()
        rec['distances'] = {name:clearance(xyz,body(pose,config)) for name,xyz in sources.items() if len(xyz)}
        item['candidates'].append(rec)
    results['advance'].append(item)
    eligible = [r for r in item['candidates'] if r['aperture_ok']]
    print('advance_mm',advance*1000,'aperture',len(eligible),'body_target>=2mm',sum(r['distances']['fused_target']['mm']>=2 for r in eligible))
    for r in eligible:
        print(r['idx'],'shift',round(r['shift_mm'],2),'body_mm',{k:round(v['mm'],2) for k,v in r['distances'].items()})

results['icp'] = []
src = fusion._downsample_points(sources['side1_target_registered'],args.active_reg_voxel)
tgt = fusion._downsample_points(sources['center_target_registered'],args.active_reg_voxel)
for voxel in [0,0.0025]:
    config=copy.copy(args)
    config.active_reg_icp_voxel=voxel
    T,info=fusion._rigid_icp_registration(src,tgt,config)
    full=fusion._apply_T_points(sources['side1_target_registered'],T)
    report={'voxel':voxel,'info':info,'T':T.tolist(),'mutual_overlap':fusion._mutual_overlap_stats(fusion._apply_T_points(src,T),tgt,args.active_reg_icp_overlap_thr)}
    # Hold the actual 30 mm-advance endpoint fixed: does target ICP alone
    # remove a body collision, even before recomputing the network grasp?
    report['fixed_endpoint_distances']={}
    for row in results['advance'][-1]['candidates']:
        if row['aperture_ok']:
            cap=body(row['pose'],args)
            report['fixed_endpoint_distances'][row['idx']]={'side_after':clearance(full,cap),'center':clearance(sources['center_target_registered'],cap)}
    results['icp'].append(report)
    print('ICP',voxel,info,'mutual',report['mutual_overlap'])

path=request.output_json.resolve()
path.parent.mkdir(parents=True, exist_ok=True)
results['effective_geometry']={key:getattr(args,key) for key in (
    'topdown_grasp','tool_approach_axis','graspnet_approach_axis',
    'tcp_gripper_width_axis','grasp_advance','pregrasp_offset',
    'motion_path_min_clearance','motion_target_non_tool_clearance',
    'motion_target_gripper_body_clearance','active_fusion_rigid_icp',
    'active_reg_icp_voxel','motion_path_tool_radius')}
results['limitations']=[
    'Uses current parser/defaults and current YAML with the saved argv; historical settings absent from argv cannot all be recovered.',
    'Candidate indices/orientations/widths recovered from 6-point LineSet groups; scores and network depths are not recoverable. This run did not enable depth advance.',
    'Only body endpoint distances are checked. Aperture count changes with advance. No full-plan success or reliable physical grip is implied.',
    'Raw obstacle distances precede robot self filtering and dynamic-map processing; do not equate every raw point with a final runtime collision.',
    'Different views observe different surfaces. Global nearest-neighbor residuals and Z percentiles alone do not measure true registration error or prove ghost thickness.',
]
results['view_quality']=[]
import cv2
from real_nero_grasp.collision import _urdf_chain_geometry
calibration=json.loads(pathlib.Path(args.calib_json).read_text())
for name,(folder,meta) in archive.items():
    scene=np.asarray(o3d.io.read_point_cloud(str(folder/'scene_base.ply')).points)
    target=np.asarray(o3d.io.read_point_cloud(str(folder/'target_base.ply')).points)
    plane,_=fusion._fit_table_plane(scene,args)
    mask=cv2.imread(str(folder/'selected_target_mask.png'),cv2.IMREAD_GRAYSCALE)>0
    depth=np.load(folder/'depth_m.npy')
    valid=mask & np.isfinite(depth) & (depth>args.min_depth) & (depth<args.max_depth)
    quality={'view':name,'mask_pixels':int(mask.sum()),'valid_depth_pixels':int(valid.sum()),'valid_depth_ratio':float(valid.sum()/max(mask.sum(),1)),'target_points':len(target)}
    if plane is not None:
        height=target@plane['normal']+plane['d']
        quality['height_above_own_table_q5_50_95_mm']=(np.percentile(height,[5,50,95])*1000).tolist()
        quality['target_points_below_5mm_above_table_ratio']=float(np.mean(height<.005))
        quality['table_normal']=plane['normal'].tolist()
    _,T_fk=_urdf_chain_geometry(meta['capture_joints'],args)
    T_sdk=np.asarray(meta['T_base_camera'])@np.linalg.inv(np.asarray(calibration['matrix_4x4']))
    delta=np.linalg.inv(T_fk)@T_sdk
    quality['sdk_vs_urdf_translation_mm']=float(np.linalg.norm(delta[:3,3])*1000)
    quality['sdk_vs_urdf_rotation_deg']=float(Rotation.from_matrix(delta[:3,:3]).magnitude()*180/np.pi)
    results['view_quality'].append(quality)
results['calibration_residual_summary']=calibration.get('residual_summary')
results['outcomes_same_date']=[]
for experiment_path in sorted((ROOT/'logs').glob('2026-10-22*/experiment_result.json')):
    record=json.loads(experiment_path.read_text())
    results['outcomes_same_date'].append({'run':experiment_path.parent.name,'views':record['views'],'outcome':record['outcome']})
path.write_text(json.dumps(results,indent=2))
print('saved',path)
