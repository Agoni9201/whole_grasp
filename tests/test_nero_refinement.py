"""Nero refinement must preserve native plans and execute the checked TCP."""

import copy
from contextlib import closing
import json
import sys
from types import SimpleNamespace as NS
from unittest import mock

import numpy as np
import pytest

from real_nero_grasp.cli import parse_args
from real_nero_grasp import nero_refinement as refinement
from real_nero_grasp.grasp_geometry import grasp_to_tcp_poses
from real_nero_grasp.grasp_geometry import _rotation_from_axis_constraints as rotation_from_axes
from real_nero_grasp.transforms import _make_T_from_pose
from real_nero_grasp.types import PlanNotExecutable


def args_for(generator='graspnet', *flags):
    argv = ['test', '--config', 'config/pipeline.yaml', '--phase', 'grasp',
            '--grasp-table-reference-z', '0', '--nero-grasp-refine',
            '--grasp-generator', generator, *flags]
    with mock.patch.object(sys, 'argv', argv):
        args = parse_args()
    args.table_z = 0.
    args.motion_path_min_clearance = .005
    return args


def grasp_for(generator, args):
    tcp = np.eye(4)
    tcp[:3, :3] = rotation_from_axes('x', [0, 0, -1], 'z', [0, 1, 0])
    tcp[:3, 3] = [.1, .2, .03]
    mapping = refinement.graspnet_rotation_to_tcp_rotation(np.eye(3), args)
    grasp = NS(width=.1, depth=.02, height=.02, score=.8,
               rotation_matrix=tcp[:3, :3] @ mapping.T, translation=tcp[:3, 3].copy())
    if generator == 'graspgenx':
        grasp.nero_tcp_pose_camera = tcp.copy()
        grasp.nero_tcp_contact_advance = .004
    return grasp


@pytest.mark.parametrize('generator', ['graspnet', 'graspgenx'])
@pytest.mark.parametrize('native_advance', [0., .009, 'auto'])
def test_checked_advance_survives_pose_conversion_without_global_override(generator, native_advance):
    args = args_for(generator)
    args.graspgenx_contact_advance = native_advance
    args.grasp_advance = .015
    grasp = grasp_for(generator, args)
    original = copy.deepcopy(grasp)
    grasp.nero_execution_advance_m = .007
    pre, contact, lift = map(_make_T_from_pose, grasp_to_tcp_poses(grasp, np.eye(4), args))
    expected = grasp.translation + contact[:3, 0] * .007
    np.testing.assert_allclose(contact[:3, 3], expected, atol=1e-10)
    np.testing.assert_allclose(pre[:3, 3], expected - contact[:3, 0]*args.pregrasp_offset)
    np.testing.assert_allclose(lift[:3, 3], expected + [0, 0, args.lift_height])
    np.testing.assert_array_equal(grasp.translation, original.translation)
    if generator == 'graspgenx':
        np.testing.assert_array_equal(grasp.nero_tcp_pose_camera, original.nero_tcp_pose_camera)
    assert args.graspgenx_contact_advance == native_advance
    assert args.grasp_advance == .015


def test_orientation_families_keep_rotation_constraints_and_include_native():
    original = rotation_from_axes('x', [.5, 0, -np.sqrt(.75)], 'z', [.3, 1, 0])
    variants = dict((name, (matrix, angle)) for name, matrix, angle in refinement.orientation_variants(original))
    np.testing.assert_array_equal(variants['native'][0], original)
    np.testing.assert_allclose(variants['level_jaws'][0][:, 0], original[:, 0])
    assert variants['level_jaws'][0][2, 2] == pytest.approx(0, abs=1e-10)
    np.testing.assert_allclose(variants['vertical'][0][:, 0], [0, 0, -1], atol=1e-10)
    for matrix, angle in variants.values():
        np.testing.assert_allclose(matrix.T @ matrix, np.eye(3), atol=1e-10)
        assert np.linalg.det(matrix) == pytest.approx(1)
        assert 0 <= angle <= 180


def test_native_full_route_skips_search_and_keeps_original(tmp_path):
    args = args_for()
    args.post_grasp_hold_joints_deg = None
    grasp = grasp_for('graspnet', args)
    rows = [dict(idx=2, grasp=grasp, workflow_preflight_ok=True)]
    with mock.patch('real_nero_grasp.planning_process.checked_request', side_effect=AssertionError('unneeded search')):
        result = refinement.refine_failed_candidates(rows, None, args, None, None, np.eye(4),
            None, 'test', tmp_path, preflight=None, aperture=None, reachability=None, preflight_options={})
    assert result[0]['grasp'] is grasp
    assert result[0]['nero_full_route_ok']
    assert json.loads((tmp_path/'nero_refinement.json').read_text())['route_attempts'] == 0


def test_requested_hold_failure_disqualifies_pickup_route():
    args = args_for()
    plan = NS(safety_scene=None, attached_target_tcp=np.zeros((1, 3)),
              closed_gripper_width=.035, after_close=[NS(joints=np.zeros(7))])
    row = dict(workflow_preflight_ok=True, _workflow_motion_plan=plan)
    with mock.patch('real_nero_grasp.motion_planning._motion_context', return_value=object()), \
         mock.patch('real_nero_grasp.post_grasp_rrt.plan_post_grasp_joint_hold', side_effect=PlanNotExecutable('blocked hold')):
        assert not refinement.qualify_hold(row, args, 'test')
    assert not row['workflow_preflight_ok']
    assert 'blocked hold' in row['workflow_preflight_reason']


@pytest.mark.parametrize('generator', ['graspnet', 'graspgenx'])
def test_repaired_candidate_keeps_generator_and_exact_native_tcp(tmp_path, generator):
    args = args_for(generator)
    args.post_grasp_hold_joints_deg = None
    grasp = grasp_for(generator, args)
    altered = copy.deepcopy(grasp)
    new_rotation = rotation_from_axes('x', [.25, 0, -np.sqrt(.9375)], 'z', [0, 1, 0])
    altered.rotation_matrix = new_rotation @ refinement.graspnet_rotation_to_tcp_rotation(np.eye(3), args).T
    altered.nero_execution_advance_m = .011
    if generator == 'graspgenx':
        altered.nero_tcp_pose_camera[:3, :3] = new_rotation
        altered.nero_tcp_contact_advance = .011
    proposal = dict(grasp=refinement.pack_grasp(altered), family='tilt15',
                    angle_change_deg=15, advance_m=.011, cad_table_clearance_m=.012)
    rows = [dict(idx=7, generator=generator, grasp=grasp, workflow_preflight_ok=False,
                 genx_refinement_only=generator=='graspgenx')]
    def preflight(items, *unused, **kwargs):
        items[0].update(workflow_preflight_ok=True, workflow_preflight_reason='passed', _workflow_motion_plan=object())
        return items
    with mock.patch('real_nero_grasp.planning_process.checked_request', return_value=[dict(proposals=[proposal], samples=20)]), \
         mock.patch('real_nero_grasp.planning_process.planning_health_check'):
        refinement.refine_failed_candidates(rows, None, args, NS(points=np.zeros((1, 3))), None,
            np.eye(4), None, 'test', tmp_path, preflight=preflight,
            aperture=lambda rows, *a: rows, reachability=lambda rows, *a, **k: rows, preflight_options={})
    row = rows[0]
    assert row['generator'] == generator and row['nero_full_route_ok']
    assert row['nero_refinement_advance_m'] == .011
    _, contact, _ = grasp_to_tcp_poses(row['grasp'], np.eye(4), args)
    np.testing.assert_allclose(_make_T_from_pose(contact)[:3, :3], new_rotation, atol=1e-10)
    np.testing.assert_allclose(contact[:3], grasp.translation + new_rotation[:, 0] * .011)
    if generator == 'graspgenx':
        np.testing.assert_allclose(row['genx_tcp_pose_camera'], altered.nero_tcp_pose_camera)
        assert row['genx_contact_advance_m'] == .011
        assert not row['genx_refinement_only']
        assert row['genx_refined_table_clearance_m'] == .012
        report=json.loads((tmp_path/'nero_refinement.json').read_text())
        assert report['attempts'][0]['from_table_rejected_pool']


def test_repair_candidate_stays_rejected_when_complete_path_fails(tmp_path):
    args = args_for('graspgenx')
    args.post_grasp_hold_joints_deg = None
    grasp = grasp_for('graspgenx',args)
    proposal = dict(grasp=refinement.pack_grasp(grasp),family='vertical',
                    angle_change_deg=15,advance_m=.007,cad_table_clearance_m=.012)
    rows=[dict(idx=0,generator='graspgenx',grasp=grasp,genx_refinement_only=True,
               workflow_preflight_ok=False,workflow_preflight_reason='native rejected')]
    def preflight(items,*a,**kw):
        assert not items[0]['genx_refinement_only']
        items[0].update(workflow_preflight_ok=False,workflow_preflight_reason='IK failed')
        return items
    with mock.patch('real_nero_grasp.planning_process.checked_request',return_value=[dict(proposals=[proposal],samples=10)]), \
         mock.patch('real_nero_grasp.planning_process.planning_health_check'):
        refinement.refine_failed_candidates(rows,None,args,NS(points=np.zeros((1,3))),None,
            np.eye(4),None,'test',tmp_path,preflight=preflight,
            aperture=lambda rows,*a: rows,reachability=lambda rows,*a,**kw: rows,preflight_options={})
    assert rows[0]['genx_refinement_only']
    assert not rows[0]['workflow_preflight_ok']
    assert rows[0]['grasp'] is grasp


def test_real_genx_geometry_search_uses_measured_cloud_and_native_tcp(tmp_path):
    from real_nero_grasp.planning_process import PlanningProcess, config_snapshot
    args = args_for('graspgenx')
    args.motion_path_open_finger_front = .005
    args.grasp_physical_min_contact_inset = .010
    grasp = grasp_for('graspgenx', args)
    # Both measured contact sides are below the current fingertip, forcing a
    # nonzero advance. No completed/hallucinated object points are supplied.
    x, y = np.meshgrid(np.linspace(-.003, .003, 8), np.linspace(-.025, .025, 50))
    local = np.column_stack([x.ravel()+.002, np.zeros(x.size), y.ravel()])
    points = local @ grasp.nero_tcp_pose_camera[:3, :3].T + grasp.translation
    with closing(PlanningProcess()) as worker:
        result = worker.request('nero_grasp_geometry', dict(config=config_snapshot(args),
            rows=[dict(idx=0, generator='graspgenx', grasp=refinement.pack_grasp(grasp))],
            target_points=points, T_base_camera=np.eye(4)), timeout_s=60)
        telemetry = worker.request('ping', {})
    assert telemetry['can_forbidden']
    assert not telemetry['runtime_loaded'] and not telemetry['camera_loaded']
    assert result[0]['proposals']
    assert all(0 < p['advance_m'] <= .015 for p in result[0]['proposals'])
    assert all(p['cad_table_clearance_m'] >= .005 for p in result[0]['proposals'])
    for p in result[0]['proposals']:
        np.testing.assert_array_equal(p['grasp']['nero_tcp_pose_camera'][:3, 3], grasp.translation)


@pytest.mark.parametrize('flags', [
    ['--gripper-open-width', '.085'], ['--use-graspnet-depth-advance'],
    ['--nero-refine-max-route-variants', '9'], ['--nero-refine-timeout-s', 'nan'],
    ['--grasp-contact-check-only', '--grasp-contact-check-clearance', '-.005'],
])
def test_refinement_entry_rejects_incompatible_overrides(flags):
    with pytest.raises(SystemExit):
        args_for('graspnet', *flags)


def test_graspnet_final_preview_uses_same_complete_nero_cad_as_genx(tmp_path):
    from real_nero_grasp import runtime
    args = args_for()
    grasp = grasp_for('graspnet', args)
    grasp.nero_execution_advance_m = .007
    pre, contact, _ = grasp_to_tcp_poses(grasp, np.eye(4), args)
    with mock.patch.object(runtime, 'draw_open3d_preview') as drawn:
        assert runtime.preview_final_grasp_candidate(runtime.get_o3d().geometry.PointCloud(),
            grasp, dict(idx=0, generator='graspnet'), np.eye(4), pre, contact,
            1., 0., args, str(tmp_path))
    geometries = drawn.call_args.args[0]
    assert len(geometries) == 8
    assert len(geometries[-1].vertices) > 100
    assert (tmp_path/'final_nero_open_gripper.obj').is_file()
