import ast
from contextlib import nullcontext
from dataclasses import fields
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace
from unittest import mock

import cv2
import numpy as np
import pytest

from real_nero_grasp.mgpc_completion import complete_graspnet_target, crop_target_rgb, validate_mgpc_configuration


def fixture_cloud(tmp_path):
    import open3d as o3d
    rng=np.random.default_rng(4)
    points=rng.uniform([-.02,-.02,.4],[.02,.02,.44],(200,3))
    cloud=o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    checkpoint=tmp_path/'weights.pt';checkpoint.write_bytes(b'test')
    args=SimpleNamespace(mgpc=True,mgpc_python='/usr/bin/python3',mgpc_checkpoint=str(checkpoint),
        mgpc_category='cube',mgpc_seed=0,mgpc_timeout=20,target_class='detected_label')
    image=np.full((30,40,3),[20,40,180],np.uint8)
    mask=np.zeros((30,40),np.uint8);mask[10:20,15:25]=255
    cv2.imwrite(str(tmp_path/'selected_target_mask.png'),mask)
    return cloud,points,args,image


def test_completion_disabled_does_not_start_worker_or_touch_files(tmp_path):
    original=object()
    runner=mock.Mock(side_effect=AssertionError('worker must not run'))
    assert complete_graspnet_target(original,None,None,SimpleNamespace(mgpc=False),tmp_path,runner=runner) is original
    runner.assert_not_called()
    assert not (tmp_path/'mgpc').exists()


@pytest.mark.parametrize('registered',[False,True])
def test_completion_reference_pose_roundtrip_preserves_measured_points(tmp_path,registered):
    original,points,args,image=fixture_cloud(tmp_path)
    launcher=tmp_path/'venv_python';launcher.symlink_to('/usr/bin/python3')
    args.mgpc_python=str(launcher)
    reference=tmp_path/'multiview_raw/center_test';reference.mkdir(parents=True)
    cv2.imwrite(str(reference/'color.png'),image)
    cv2.imwrite(str(reference/'selected_target_mask.png'),cv2.imread(str(tmp_path/'selected_target_mask.png'),0))
    T_center=np.eye(4);T_center[:3,3]=[.1,-.2,.05]
    (reference/'metadata.json').write_text(json.dumps(dict(T_base_camera=T_center.tolist())))
    T_input=np.array([[0,-1,0,.3],[1,0,0,.1],[0,0,1,.2],[0,0,0,1.]])
    expected_reference=T_center.copy()
    if registered:
        from scipy.spatial.transform import Rotation
        correction=np.eye(4)
        correction[:3,:3]=Rotation.from_euler('x',4.3,degrees=True).as_matrix()
        correction[:3,3]=[.001,-.002,-.0207]
        report_path=tmp_path/'table_reference_report.json'
        report_path.write_text(json.dumps(dict(views=[dict(view='center',source=str(reference),
                                                          T_registration=correction.tolist())])))
        args._grasp_table_reference_report=str(report_path)
        expected_reference=correction @ T_center
        T_input=correction @ T_input
    def worker(command,**kwargs):
        assert command[0]==str(launcher)
        request_path=Path(command[-1]);request=json.loads(request_path.read_text())
        assert request['category']=='cube'
        assert request['rgb_source']==str(reference)
        np.testing.assert_allclose(request['T_base_reference'],expected_reference)
        ids=np.load(request_path.parent/'sample_indices.npy')
        from tools.compare_mgpc_reconstruction import transform
        expected_points=transform(points,np.linalg.inv(expected_reference) @ T_input)[ids]
        np.testing.assert_allclose(request['normalization']['centroid_camera_m'],
                                   expected_points.mean(axis=0),atol=1e-7)
        assert kwargs['check'] is True
        assert 'PYTHONPATH' not in kwargs['env']
        sample=np.load(request_path.parent/'input_normalized.npy')
        np.save(request_path.parent/'prediction_normalized.npy',np.resize(sample,(8192,3)))
    completed=complete_graspnet_target(original,T_input,image,args,tmp_path,runner=worker)
    np.testing.assert_array_equal(original.points,points)
    from scipy.spatial import cKDTree
    assert cKDTree(np.asarray(completed.points)).query(points)[0].max()<1e-7
    assert completed is not original
    assert args._mgpc_completion_info['safety_target']=='original measured target'


@pytest.mark.parametrize('failure',['timeout','invalid','missing'])
def test_completion_failures_never_fall_back_to_stale_prediction(tmp_path,failure):
    original,points,args,image=fixture_cloud(tmp_path)
    directory=tmp_path/'mgpc';directory.mkdir()
    np.save(directory/'prediction_normalized.npy',np.zeros((8192,3)))
    def worker(*command,**kwargs):
        if failure=='timeout':raise subprocess.TimeoutExpired('worker',1)
        if failure=='invalid':np.save(directory/'prediction_normalized.npy',np.full((8192,3),np.nan))
    with pytest.raises((RuntimeError,FileNotFoundError)):
        complete_graspnet_target(original,np.eye(4),image,args,tmp_path,runner=worker)
    np.testing.assert_array_equal(original.points,points)


def test_rgb_crop_preserves_channels_and_rejects_invalid_mask():
    image=np.full((10,20,3),[10,20,200],np.uint8)
    mask=np.zeros((10,20),bool);mask[3:7,8:12]=True
    result=crop_target_rgb(image,mask)
    np.testing.assert_array_equal(result[112,112],[10,20,200])
    with pytest.raises(ValueError):crop_target_rgb(image,np.zeros((10,20),bool))


def test_missing_resources_rejected_before_arm_preflight(tmp_path):
    with pytest.raises(FileNotFoundError):
        validate_mgpc_configuration(SimpleNamespace(mgpc=True,mgpc_python=str(tmp_path/'absent'),
            mgpc_checkpoint='/tmp/absent',mgpc_timeout=1))


def test_workflow_supplies_measured_occupancy_and_scene_with_completed_proposals(tmp_path):
    from real_nero_grasp import grasp_workflow as workflow
    from real_nero_grasp.cli import parse_args
    from real_nero_grasp.grasp_geometry import _axis_vector
    import sys
    with mock.patch.object(sys,'argv',['test','--calib-json','/tmp/test.json','--phase','grasp','--mgpc']):
        args=parse_args()
    original,_,_,image=fixture_cloud(tmp_path)
    predicted=SimpleNamespace(points=np.ones((8192,3)))
    scene=object();arm=mock.Mock()
    make_inputs=mock.Mock(return_value=({},object()))
    generate=mock.Mock(side_effect=RuntimeError('[generate_grasps] test stop'))
    hooks={field.name:mock.Mock() for field in fields(workflow.GraspWorkflowHooks)}
    hooks.update(axis_vector=_axis_vector,get_graspnet_helpers=lambda:(make_inputs,generate),
        recoverable_grasp_generation_failure=lambda exc:True)
    with mock.patch('real_nero_grasp.mgpc_completion.complete_graspnet_target',return_value=predicted), \
         mock.patch.object(workflow,'_prepare_graspnet_target_cloud',side_effect=lambda value,*a,**k:value), \
         mock.patch('real_nero_grasp.pointcloud_fusion.save_cloud'):
        workflow.plan_and_maybe_execute_grasp(arm,args,original,scene,np.eye(4),image,None,str(tmp_path),
            {'label':'cube'},hooks=workflow.GraspWorkflowHooks(**hooks))
    assert make_inputs.call_args.args[0] is predicted
    assert generate.call_args.kwargs['target_support_cloud'] is original
    assert generate.call_args.kwargs['collision_cloud'] is scene
    hooks['annotate_grasp_candidate_physical_aperture'].assert_not_called()
    arm.move_j.assert_not_called()


def test_early_occupancy_detector_uses_measured_cloud_in_production_generator():
    from tools.compare_mgpc_graspnet import ROOT,baseline_imports
    baseline_imports()
    from graspnetAPI import GraspGroup
    from collision_detector import ModelFreeCollisionDetector
    path=ROOT/'grasp_active_three_views_sceneaware_asymmetric_clear_nbv_obstacle_yolo26seg.py'
    function=next(node for node in ast.parse(path.read_text()).body if isinstance(node,ast.FunctionDef) and node.name=='generate_grasps')
    raw=np.r_[1.,.04,.02,.03,np.eye(3).reshape(-1),[0,0,0],-1.][None]
    tensor=SimpleNamespace(detach=lambda:SimpleNamespace(cpu=lambda:SimpleNamespace(numpy=lambda:raw.copy())))
    namespace=dict(np=np,torch=SimpleNamespace(no_grad=nullcontext),GraspGroup=GraspGroup,
        ModelFreeCollisionDetector=ModelFreeCollisionDetector,
        _prepare_graspnet_for_inference=lambda:lambda value:value,_clear_cuda_cache=lambda:None,
        _offload_graspnet=lambda:None,pred_decode=lambda _: [tensor])
    exec(compile(ast.Module(body=[function],type_ignores=[]),str(path),'exec'),namespace)
    predicted=SimpleNamespace(points=np.array([[0,0,0.]]))
    measured=SimpleNamespace(points=np.array([[1,1,1.]]))
    with pytest.raises(RuntimeError,match='empty target closing volume'):
        namespace['generate_grasps']({},predicted,collision_cloud=measured,target_support_cloud=measured,
            collision_isolated_filter=False,allow_collision_fallback=False)
