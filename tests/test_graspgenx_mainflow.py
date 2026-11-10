"""Verify the GraspGenX flange/TCP bridge and fail-closed worker boundary."""

import json
import sys
from dataclasses import fields
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest

from real_nero_grasp.cli import parse_args
from real_nero_grasp.grasp_geometry import grasp_to_tcp_poses
from real_nero_grasp.graspgenx_integration import canonical_to_tcp, generate_graspgenx_candidates, selected_mesh_table_clearance, gripper_asset
from real_nero_grasp.transforms import _as_pose_list


def _args():
    with mock.patch.object(sys, "argv", ["main", "--phase", "grasp", "--calib-json", "/tmp/calibration.json", "--grasp-generator", "graspgenx"]):
        args = parse_args()
    args.table_z = 0.0
    return args


def test_asset_link7_tcp_chain_places_tcp_at_fingertips():
    mapped = canonical_to_tcp(_args())
    np.testing.assert_allclose(mapped[:3, 3], [.0015, 0, .133], atol=1e-4)
    np.testing.assert_allclose(mapped[:3, :3] @ [1, 0, 0], [0, 0, 1], atol=1e-4)


@pytest.mark.parametrize("opening", [.085, .100])
def test_asset_opening_tracks_the_commanded_working_gap(opening):
    from real_nero_grasp.paths import PROJECT_ROOT
    args = _args()
    args.gripper_open_width = opening
    args.gripper_hardware_max_width = .100
    name, gap = gripper_asset(args)
    provenance = json.loads((PROJECT_ROOT / "third_party/GraspGenX/assets/x_grippers" /
                             name / "nero_asset_provenance.json").read_text())
    assert gap == opening
    assert provenance["open_gap_m"] == pytest.approx(opening, abs=.001)


def test_final_selected_mesh_table_guard_uses_recentered_pose():
    args = _args()
    canonical_tcp = canonical_to_tcp(args)
    def tcp_at_asset_height(height):
        asset = np.eye(4)
        asset[2, 3] = height
        return _as_pose_list(asset @ canonical_tcp)
    clearance = selected_mesh_table_clearance(
        tcp_at_asset_height(.25), tcp_at_asset_height(.20), args
    )
    assert clearance > .005
    with pytest.raises(RuntimeError, match="table clearance"):
        selected_mesh_table_clearance(
            tcp_at_asset_height(.04), tcp_at_asset_height(-.02), args
        )


def test_genx_fixed_opening_is_not_deinflated_or_treated_as_saturated_prediction(capsys):
    from real_nero_grasp.grasp_candidates import annotate_grasp_candidate_physical_aperture
    args = _args()
    args.graspnet_decode_max_width = .100
    points = np.column_stack([np.full(200, -.020), np.zeros(200), np.linspace(-.03, .03, 200)])
    grasp = SimpleNamespace(width=.100, rotation_matrix=np.eye(3), translation=np.zeros(3),
                            nero_tcp_pose_camera=np.eye(4))
    rows = annotate_grasp_candidate_physical_aperture(
        [dict(idx=0, grasp=grasp, generator='graspgenx')],
        SimpleNamespace(points=points), np.eye(4), np.zeros(6), args)
    assert rows[0]['physical_aperture_ok']
    assert rows[0]['graspnet_nominal_width'] is None
    assert not rows[0]['graspnet_width_saturated']
    text = capsys.readouterr().out
    assert 'fixed_asset_opening=100.0mm' in text
    assert 'nominal=83.3mm' not in text
    grasp.width = .101
    rows = annotate_grasp_candidate_physical_aperture(
        [dict(idx=0, grasp=grasp, generator='graspgenx')],
        SimpleNamespace(points=points), np.eye(4), np.zeros(6), args)
    assert not rows[0]['physical_aperture_ok']
    assert 'fixed GenX asset width' in rows[0]['physical_aperture_reason']


def test_final_preview_draws_actual_nero_mesh_for_genx(tmp_path):
    from real_nero_grasp import runtime
    from real_nero_grasp.grasp_geometry import _rotation_from_axis_constraints
    args = _args()
    asset = np.eye(4)
    asset[2, 3] = .20
    T_camera_tcp = asset @ canonical_to_tcp(args)
    rotation = _rotation_from_axis_constraints(
        args.graspnet_approach_axis, T_camera_tcp[:3, :3] @ [1, 0, 0],
        args.graspnet_width_axis, T_camera_tcp[:3, :3] @ [0, 0, 1],
    )
    grasp = SimpleNamespace(rotation_matrix=rotation, translation=T_camera_tcp[:3, 3], width=.085)
    grasp.nero_tcp_pose_camera = T_camera_tcp
    pre, command, _ = grasp_to_tcp_poses(grasp, np.eye(4), args)
    cloud = runtime.get_o3d().geometry.PointCloud()
    with mock.patch.object(runtime, "draw_open3d_preview") as drawn:
        assert runtime.preview_final_grasp_candidate(
            cloud, grasp, {"idx": 0, "generator": "graspgenx"},
            np.eye(4), pre, command, 1., 0., args, str(tmp_path)
        )
    geometries = drawn.call_args.args[0]
    assert len(geometries) == 8
    assert len(geometries[-1].vertices) > 100
    assert (tmp_path / "final_graspgenx_open_gripper.obj").is_file()


def test_live_adapter_keeps_exact_genx_tcp_without_graspnet_advance(tmp_path):
    args = _args()
    args.grasp_advance = .03
    points = np.tile([.02, .01, .4], (200, 1))
    cloud = SimpleNamespace(points=points)
    pose = np.eye(4)
    pose[:3, 3] = [.02, .01, .25]

    def worker(command, **kwargs):
        request = json.loads((tmp_path / "graspgenx/request.json").read_text())
        assert request["seed"] == 42
        assert kwargs["check"] is True
        result = {
            "source": f"GraspGenX diffusion, {request['gripper_name']}",
            "candidates": [{
                "index": 0, "pose_camera": pose.tolist(), "score": .8,
                "table_safe": True, "minimum_table_clearance_m": .02,
            }],
        }
        (tmp_path / "graspgenx/candidates.json").write_text(json.dumps(result))

    _, rows = generate_graspgenx_candidates(cloud, np.eye(4), args, tmp_path, runner=worker)
    grasp = rows[0]["grasp"]
    assert grasp.width == pytest.approx(.100)
    _, tcp, _ = grasp_to_tcp_poses(grasp, np.eye(4), args)
    expected = pose @ canonical_to_tcp(args)
    np.testing.assert_allclose(tcp[:3], expected[:3, 3], atol=1e-6)
    assert rows[0]["generator"] == "graspgenx"
    assert rows[0]["genx_table_clearance_m"] == .02


@pytest.mark.parametrize('refine', [False, True])
def test_table_rejected_pool_is_separate_and_never_displaces_native_pose(tmp_path, refine):
    args = _args()
    args.nero_grasp_refine = refine
    args.manual_topk = 1
    pose = np.eye(4)
    pose[2, 3] = .25
    # A higher-scoring rejected pose must not consume the native Top-K slot.
    def worker(command, **kwargs):
        request = json.loads((tmp_path/'graspgenx/request.json').read_text())
        candidates = [dict(index=i, pose_camera=pose.tolist(), score=score,
            table_safe=safe, minimum_table_clearance_m=clearance) for i,score,safe,clearance in (
                (0,.99,False,-.10), (1,.98,False,-.003), (2,.97,False,.004),
                (3,.80,True,.020), (4,.70,True,.015))]
        (tmp_path/'graspgenx/candidates.json').write_text(json.dumps(
            dict(source=f"GraspGenX diffusion, {request['gripper_name']}", candidates=candidates)))
    _, rows = generate_graspgenx_candidates(SimpleNamespace(points=np.ones((200,3))),
        np.eye(4),args,tmp_path,runner=worker)
    assert rows[0]['generator_index'] == 3
    assert rows[0]['genx_native_table_safe']
    assert not rows[0]['genx_refinement_only']
    if refine:
        assert [row['generator_index'] for row in rows] == [3,2,1,0]
        assert all(row['genx_refinement_only'] for row in rows[1:])
    else:
        assert len(rows) == 1


def test_all_table_rejected_poses_survive_only_as_repair_material(tmp_path):
    args = _args()
    args.nero_grasp_refine = True
    args.graspgenx_contact_advance = 'auto'
    target = SimpleNamespace(points=np.ones((200,3)))
    pose = np.eye(4)
    pose[2,3] = .25
    def worker(command, **kwargs):
        request=json.loads((tmp_path/'graspgenx/request.json').read_text())
        (tmp_path/'graspgenx/candidates.json').write_text(json.dumps(dict(
            source=f"GraspGenX diffusion, {request['gripper_name']}", candidates=[dict(
                index=0,pose_camera=pose.tolist(),score=.9,table_safe=False,minimum_table_clearance_m=-.003)])))
    with mock.patch('real_nero_grasp.grasp_candidates.annotate_grasp_candidate_physical_aperture',
                    side_effect=AssertionError('deferred raw pose must skip native advance search')):
        _, rows = generate_graspgenx_candidates(target,np.eye(4),args,tmp_path,
            measured_target=target,runner=worker)
    assert len(rows)==1 and rows[0]['genx_refinement_only']
    assert not rows[0]['genx_contact_refine_ok']


def test_deferred_pose_cannot_enter_complete_route_planning():
    from real_nero_grasp.grasp_planner import annotate_grasp_candidate_workflow_preflight
    args = _args()
    args.grasp_motion_path_guard = True
    row = dict(idx=0,genx_refinement_only=True,physical_aperture_ok=True,
               table_guard_ok=True,grasp=SimpleNamespace(width=.1))
    forbidden = mock.Mock(side_effect=AssertionError('raw table-rejected pose reached IK/planner'))
    rows = annotate_grasp_candidate_workflow_preflight([row],None,args,None,None,
        np.eye(4),np.zeros(6),'deferred',plan_up_and_over_grasp_sequence_hook=forbidden)
    assert not rows[0]['workflow_preflight_ok']
    assert 'refinement only' in rows[0]['workflow_preflight_reason']
    assert args._planning_stats['ik_calls']==0
    forbidden.assert_not_called()


def test_contact_trial_advance_changes_native_tcp_along_approach_only():
    args = _args()
    args.grasp_advance = .030
    args.graspgenx_contact_advance = .011
    pose = np.eye(4)
    pose[:3, 3] = [.1, .2, .3]
    grasp = SimpleNamespace(width=.100, rotation_matrix=np.eye(3), translation=pose[:3, 3],
                            nero_tcp_pose_camera=pose)
    _, command, _ = grasp_to_tcp_poses(grasp, np.eye(4), args)
    np.testing.assert_allclose(command[:3], [.111, .2, .3])
    np.testing.assert_allclose(grasp.nero_tcp_pose_camera, pose)


def test_genx_contact_advance_can_use_normal_guarded_grasp_without_trial_clearance():
    with mock.patch.object(sys, 'argv', ['main', '--phase', 'grasp', '--calib-json', '/tmp/calibration.json',
                                       '--grasp-generator', 'graspgenx', '--graspgenx-contact-advance', '.009']):
        args = parse_args()
    assert not args.grasp_contact_check_only
    assert args.grasp_contact_check_clearance is None
    assert args.graspgenx_contact_advance == .009


def test_auto_advance_preserves_a_deep_native_grasp_and_only_advances_the_shallow_one(tmp_path):
    args = _args()
    args.graspgenx_contact_advance = 'auto'
    args.grasp_physical_min_contact_inset = .010
    args.motion_path_open_finger_front = .005
    pose = np.eye(4)
    pose[:3, :3] = np.diag([1., -1., -1.])
    pose[2, 3] = .25
    tcp = pose @ canonical_to_tcp(args)
    target_tcp = np.column_stack([np.full(400, -.0055), np.zeros(400), np.linspace(-.03, .03, 400)])
    points = target_tcp @ tcp[:3, :3].T + tcp[:3, 3]
    target = SimpleNamespace(points=points)
    shallow = pose.copy()
    shallow[2, 3] += .009
    def worker(command, **kwargs):
        request = json.loads((tmp_path/'graspgenx/request.json').read_text())
        data = dict(source=f"GraspGenX diffusion, {request['gripper_name']}", candidates=[
            dict(index=i,pose_camera=value.tolist(),score=.9-i*.1,table_safe=True,minimum_table_clearance_m=.10)
            for i,value in enumerate([pose,shallow])])
        (tmp_path/'graspgenx/candidates.json').write_text(json.dumps(data))
    _, rows = generate_graspgenx_candidates(target, np.eye(4), args, tmp_path,
                                           measured_target=target, runner=worker)
    assert [row['genx_contact_refine_ok'] for row in rows] == [True, True]
    assert rows[0]['genx_contact_advance_m'] == 0
    assert rows[1]['genx_contact_advance_m'] == pytest.approx(.009)
    _, first, _ = grasp_to_tcp_poses(rows[0]['grasp'], np.eye(4), args)
    _, second, _ = grasp_to_tcp_poses(rows[1]['grasp'], np.eye(4), args)
    np.testing.assert_allclose(first, second, atol=1e-7)
    np.testing.assert_allclose(rows[1]['grasp'].nero_tcp_pose_camera, shallow @ canonical_to_tcp(args))
    assert args.graspgenx_contact_advance == 'auto'


def test_auto_advance_does_not_trade_table_clearance_for_contact_depth(tmp_path):
    from real_nero_grasp.graspgenx_integration import open_gripper_mesh
    args = _args()
    args.graspgenx_contact_advance = 'auto'
    args.grasp_physical_min_contact_inset = .010
    args.motion_path_open_finger_front = .005
    pose = np.eye(4)
    pose[:3, :3] = np.diag([1., -1., -1.])
    pose[2, 3] = np.asarray(open_gripper_mesh(args).vertices)[:, 2].max() + .006
    tcp = pose @ canonical_to_tcp(args)
    local = np.column_stack([np.full(400, .0035), np.zeros(400), np.linspace(-.03, .03, 400)])
    target = SimpleNamespace(points=local @ tcp[:3, :3].T + tcp[:3, 3])
    def worker(command, **kwargs):
        request = json.loads((tmp_path/'graspgenx/request.json').read_text())
        data = dict(source=f"GraspGenX diffusion, {request['gripper_name']}", candidates=[
            dict(index=0,pose_camera=pose.tolist(),score=.9,table_safe=True,minimum_table_clearance_m=.006)])
        (tmp_path/'graspgenx/candidates.json').write_text(json.dumps(data))
    _, rows = generate_graspgenx_candidates(target, np.eye(4), args, tmp_path,
                                           measured_target=target, runner=worker)
    assert not rows[0]['genx_contact_refine_ok']
    assert rows[0]['grasp'].nero_tcp_contact_advance == 0


@pytest.mark.parametrize('flags', [
    ['--grasp-contact-check-clearance', '.002'],
    ['--grasp-contact-check-only', '--grasp-contact-check-clearance', '.006'],
    ['--grasp-contact-check-only', '--grasp-contact-check-clearance', '-.001'],
    ['--grasp-contact-check-only', '--grasp-table-reference-z', '0', '--grasp-contact-check-clearance', '-.006'],
    ['--grasp-contact-check-only', '--grasp-contact-check-clearance', 'nan'],
    ['--grasp-generator', 'graspnet', '--graspgenx-contact-advance', '.011'],
    ['--grasp-generator', 'graspnet', '--graspgenx-contact-advance', 'auto'],
    ['--grasp-contact-check-only', '--graspgenx-contact-advance', '.020'],
])
def test_trial_overrides_require_open_finger_diagnostic_mode_and_bounded_values(flags):
    with mock.patch.object(sys, 'argv', ['main', '--phase', 'grasp', '--calib-json', '/tmp/calibration.json',
                                       '--grasp-generator', 'graspgenx'] + flags):
        with pytest.raises(SystemExit):
            parse_args()


def test_contact_trial_can_reduce_mesh_margin_without_changing_normal_defaults():
    from real_nero_grasp.graspgenx_integration import grasp_mesh_table_margin
    args = _args()
    args.motion_path_min_clearance = .005
    assert grasp_mesh_table_margin(args) == .005
    args.grasp_contact_check_clearance = .002
    assert grasp_mesh_table_margin(args) == .002
    assert args.motion_path_min_clearance == .005
    assert args.graspgenx_table_margin == .005


def test_zero_margin_is_explicitly_available_only_for_contact_check():
    with mock.patch.object(sys, 'argv', ['main', '--phase', 'grasp', '--calib-json', '/tmp/calibration.json',
                                       '--grasp-contact-check-only', '--grasp-contact-check-clearance', '0']):
        args = parse_args()
    from real_nero_grasp.graspgenx_integration import grasp_mesh_table_margin
    assert args.grasp_contact_check_clearance == 0.0
    assert grasp_mesh_table_margin(args) == 0.0
    with mock.patch.object(sys, 'argv', ['main', '--phase', 'grasp', '--calib-json', '/tmp/calibration.json',
                                       '--grasp-contact-check-clearance', '0']):
        with pytest.raises(SystemExit):
            parse_args()


def test_minus_five_mm_trial_requires_contact_mode_and_confirmed_table_reference():
    with mock.patch.object(sys, 'argv', ['main', '--phase', 'grasp', '--calib-json', '/tmp/calibration.json',
                                       '--grasp-contact-check-only', '--grasp-table-reference-z', '0',
                                       '--grasp-contact-check-clearance', '-.005']):
        args = parse_args()
    from real_nero_grasp.graspgenx_integration import grasp_mesh_table_margin
    assert args.grasp_contact_check_clearance == -.005
    assert grasp_mesh_table_margin(args) == -.005


def test_trial_mesh_check_accepts_positive_reduced_gap_but_rejects_penetration():
    from real_nero_grasp.graspgenx_integration import open_gripper_mesh, grasp_mesh_table_margin
    args = _args()
    args.motion_path_min_clearance = .005
    vertices = np.asarray(open_gripper_mesh(args).vertices)
    def at_clearance(clearance):
        asset = np.eye(4)
        asset[2, 3] = clearance - vertices[:, 2].min()
        return _as_pose_list(asset @ canonical_to_tcp(args))
    pose = at_clearance(.003)
    with pytest.raises(RuntimeError, match='table clearance'):
        selected_mesh_table_clearance(pose, pose, args)
    args.grasp_contact_check_clearance = .002
    args.graspgenx_table_margin = grasp_mesh_table_margin(args)
    assert selected_mesh_table_clearance(pose, pose, args) == pytest.approx(.003)
    penetrating = at_clearance(-.001)
    with pytest.raises(RuntimeError, match='table clearance'):
        selected_mesh_table_clearance(penetrating, penetrating, args)
    args.grasp_contact_check_clearance = -.005
    args.graspgenx_table_margin = grasp_mesh_table_margin(args)
    allowed = at_clearance(-.0035)
    assert selected_mesh_table_clearance(allowed, allowed, args) == pytest.approx(-.0035)
    too_deep = at_clearance(-.0051)
    with pytest.raises(RuntimeError, match='table clearance'):
        selected_mesh_table_clearance(too_deep, too_deep, args)
    args.grasp_contact_check_clearance = 0.0
    args.graspgenx_table_margin = grasp_mesh_table_margin(args)
    assert selected_mesh_table_clearance(pose, pose, args) > 0
    with pytest.raises(RuntimeError, match='table clearance'):
        selected_mesh_table_clearance(penetrating, penetrating, args)


def test_genx_worker_failure_does_not_consume_stale_candidates(tmp_path):
    args = _args()
    cloud = SimpleNamespace(points=np.ones((200, 3)))
    folder = tmp_path / "graspgenx"
    folder.mkdir()
    (folder / "candidates.json").write_text("stale")

    def failed(*args, **kwargs):
        raise RuntimeError("worker failed")

    with pytest.raises(RuntimeError):
        generate_graspgenx_candidates(cloud, np.eye(4), args, tmp_path, runner=failed)
    assert not (folder / "candidates.json").exists()


def test_main_workflow_uses_genx_without_invoking_graspnet(tmp_path):
    from real_nero_grasp import grasp_workflow as workflow
    from real_nero_grasp.grasp_geometry import _axis_vector
    args = _args()
    target = SimpleNamespace(points=np.ones((200, 3)))
    arm = mock.Mock()
    hooks = {field.name: mock.Mock() for field in fields(workflow.GraspWorkflowHooks)}
    hooks.update(axis_vector=_axis_vector,
                 get_graspnet_helpers=mock.Mock(side_effect=AssertionError("GraspNet must not load")),
                 recoverable_grasp_generation_failure=lambda exc: True)
    with mock.patch("real_nero_grasp.graspgenx_integration.generate_graspgenx_candidates",
                    side_effect=RuntimeError("[graspgenx] no table-safe poses")) as generator, \
         mock.patch.object(workflow, "_prepare_graspnet_target_cloud", side_effect=lambda value, *a, **k: value), \
         mock.patch("real_nero_grasp.pointcloud_fusion.save_cloud"):
        result = workflow.plan_and_maybe_execute_grasp(
            arm, args, target, target, np.eye(4), None, None, str(tmp_path),
            {"label": "cube"}, hooks=workflow.GraspWorkflowHooks(**hooks)
        )
    generator.assert_called_once()
    assert generator.call_args.args[0] is target
    hooks["get_graspnet_helpers"].assert_not_called()
    arm.move_j.assert_not_called()
    assert not result.motion_sent
