import copy
import importlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest

from real_nero_grasp.active_grasp_feedback import ActiveGraspFeedback, FeedbackPlanningSnapshot
from real_nero_grasp.perception_evidence import (
    UNKNOWN, SUPPORTED, CONFLICT, INCONSISTENT, DepthEvidence,
    classify_prediction, contact_evidence, measured_first_proposal, visible_validation_gain,
)


def config():
    from real_nero_grasp.cli import parse_args
    with mock.patch('sys.argv', ['test', '--phase', 'grasp', '--calib-json', '/tmp/test.json',
            '--active-view', '--mgpc', '--grasp-generator', 'graspgenx',
            '--grasp-table-reference-z', '0', '--active-grasp-feedback',
            '--gripper-open-width', '.100', '--gripper-hardware-max-width', '.100']):
        return parse_args()


def plane(depth=1., mask=True):
    return DepthEvidence('center', np.eye(4), np.full((64, 64), depth),
                         np.full((64, 64), mask), dict(fx=50., fy=50., cx=32., cy=32.))


def row(offset=0., passed=True):
    pose = np.eye(4)
    pose[0, 3] = offset
    grasp = SimpleNamespace(translation=pose[:3, 3].copy(), rotation_matrix=np.eye(3),
                            nero_tcp_pose_camera=pose, width=.1, depth=0., score=.8)
    return dict(idx=0, grasp=grasp, generator='graspgenx', raw_score=.8,
                workflow_preflight_ok=passed, workflow_preflight_reason='synthetic_gate',
                _workflow_motion_plan=object(), physical_aperture_variants=[])


def contacts():
    return np.array([[x, y, z] for x in np.linspace(-.05, -.015, 15)
                     for y in [-.003, 0., .003] for z in [-.02, .02]])


def test_depth_support_free_space_occlusion_and_invalid_are_distinct():
    points = np.array([[0., 0., 1.], [0., 0., .9], [0., 0., 1.1], [10., 0., 1.]])
    status, *_ = classify_prediction(points, np.empty((0, 3)), [plane()])
    np.testing.assert_array_equal(status, [SUPPORTED, CONFLICT, UNKNOWN, UNKNOWN])
    unknown, *_ = classify_prediction(points, np.empty((0, 3)), [plane(0.)])
    assert np.all(unknown == UNKNOWN)


def test_disagreeing_observations_do_not_become_confident_measurement():
    status, *_ = classify_prediction(np.array([[0., 0., 1.]]), np.empty((0, 3)), [plane(), plane(1.2)])
    assert status[0] == INCONSISTENT


def test_measured_first_preserves_xyz_and_drops_only_prediction():
    measured = np.array([[x, 0., 1.] for x in np.linspace(-.1, .1, 40)])
    prediction = np.array([[0., 0., 1.], [0., 0., .9], [0., 0., 1.1], [.05, 0., 1.1]])
    result, source, states, selected, report = measured_first_proposal(measured, prediction, [plane()])
    np.testing.assert_array_equal(result[:len(measured)], measured)
    assert len(selected) == 2 and 1 not in selected
    assert states[1] == CONFLICT
    assert np.all(source[:len(measured)] == 0) and np.all(source[len(measured):] == 1)
    assert report['measured_coordinates_preserved']


def test_dense_prediction_has_bounded_sampling_and_does_not_promote_itself():
    rng = np.random.default_rng(2)
    measured = rng.normal(size=(40, 3)) * .01
    prediction = rng.normal(size=(8192, 3)) * .01 + [0., 0., 1.]
    result, source, *_ = measured_first_proposal(measured, prediction, prediction_ratio=.5)
    assert len(result) <= 60 and np.sum(source == 1) <= 20
    status, *_ = classify_prediction(prediction[:10], measured)
    assert np.all(status == UNKNOWN)


def test_rejected_prediction_cannot_thicken_contact_bands_again():
    measured = contacts() + [0., 0., 1.]
    wrong = np.array([[x, 0., .95] for x in np.linspace(-.05, -.015, 15)])
    tcp = np.eye(4)
    tcp[2, 3] = 1.
    result = contact_evidence(tcp, measured, wrong, [plane(.98)])
    assert result['sufficient']
    assert all(r['conflict_points'] == 0 for r in result['regions'])


def test_expected_visibility_respects_measured_occluder():
    points = np.array([[0., 0., 1.]])
    normals = np.array([[0., 0., -1.]])
    assert visible_validation_gain(np.eye(4), points, normals, [1.], np.empty((0, 3))) == 1.
    assert visible_validation_gain(np.eye(4), points, normals, [1.], np.array([[0., 0., .5]])) == 0.


def test_view_ranking_changes_with_the_region_to_validate():
    from real_nero_grasp.active_view import _look_at_camera_pose
    session = ActiveGraspFeedback(config())
    session.targets = np.array([[0., 0., 1.]])
    session.weights = np.ones(1)
    choices = [dict(name='front', score=0., T_base_camera=_look_at_camera_pose([0., 0., 0.], [0., 0., 1.])),
               dict(name='side', score=0., T_base_camera=_look_at_camera_pose([1., 0., 1.], [0., 0., 1.]))]
    session.normals = np.array([[0., 0., -1.]])
    assert session.rank_views(choices, np.empty((0, 3)))[0]['name'] == 'front'
    session.normals = np.array([[1., 0., 0.]])
    assert session.rank_views(choices, np.empty((0, 3)))[0]['name'] == 'side'


def test_protocol_requires_mandatory_side_and_clears_old_paths(tmp_path):
    session = ActiveGraspFeedback(config())
    session.measured_base = contacts()
    session.assess([row()], np.eye(4), tmp_path)
    assert not session.ready  # even an entirely supported, gate-passing center cannot stop
    session.valid_sides = 1
    session.assess([row()], np.eye(4), tmp_path)
    assert session.ready
    camera = np.eye(4)
    camera[0, 3] = 1.
    reused = session.reuse_rows(camera)[0]
    assert 'workflow_preflight_ok' not in reused and '_workflow_motion_plan' not in reused
    np.testing.assert_allclose(camera @ reused['grasp'].nero_tcp_pose_camera, np.eye(4))


def test_new_candidates_get_space_even_when_old_memory_is_full():
    session = ActiveGraspFeedback(config())
    old = [row(i*.02, False) for i in range(session.args.feedback_candidate_limit)]
    for i, item in enumerate(old):
        item['feedback_memory_id'] = f'old{i}'
    merged = session.merge_rows(old, [row(1., False), row(1.1, False)], np.eye(4))
    assert any(r['feedback_memory_id'].startswith('old') for r in merged)
    assert any(not r['feedback_memory_id'].startswith('old') for r in merged)
    assert len(merged) == session.args.feedback_candidate_limit


def test_changed_raw_index_matches_pose_without_replacing_identity():
    session = ActiveGraspFeedback(config())
    old = row()
    old['feedback_memory_id'] = 'oldA'
    new = row()
    new['idx'] = 99
    merged = session.merge_rows([old], [new], np.eye(4))
    assert len(merged) == 1 and merged[0]['feedback_memory_id'] == 'oldA'


def test_probe_reuses_old_before_any_network_regeneration(tmp_path):
    session = ActiveGraspFeedback(config())
    session.valid_sides = 1
    session.measured_base = contacts()
    session.assess([row()], np.eye(4), tmp_path)
    (tmp_path/'multiview_raw').mkdir()
    calls = []
    def probe(arm, args, *a, **kw):
        calls.append((args._feedback_generate, args.execute, kw['planning_only']))
        return FeedbackPlanningSnapshot([row()], np.eye(4), SimpleNamespace(points=contacts()), str(tmp_path))
    session.probe_runner = probe
    session.update(None, None, None, np.eye(4), None, None, {}, tmp_path, tmp_path/'round')
    assert calls == [(False, False, True)]
    assert session.ready


@pytest.mark.parametrize('preflight, refine_s, expected', [(12, 60., (6, 15.)), (2, 5., (2, 5.))])
def test_observation_probe_caps_do_not_change_final_grasp_budget(tmp_path, preflight, refine_s, expected):
    args = config()
    args.execute = True
    args.grasp_preflight_max_candidates = preflight
    args.grasp_target_valid_plans = 3
    args.nero_refine_timeout_s = refine_s
    calls = []
    def probe(arm, probe_args, *a, **kw):
        calls.append((probe_args.grasp_preflight_max_candidates,
                      probe_args.nero_refine_timeout_s, probe_args.grasp_target_valid_plans,
                      probe_args.execute, kw['planning_only']))
        return FeedbackPlanningSnapshot([row()], np.eye(4), SimpleNamespace(points=contacts()), str(tmp_path))
    session = ActiveGraspFeedback(args, probe)
    session.measured_base = contacts()
    (tmp_path/'multiview_raw').mkdir()
    session.update(None, None, None, np.eye(4), None, None, {}, tmp_path, tmp_path/'round')
    assert calls == [(*expected, 1, False, True)]
    assert (args.grasp_preflight_max_candidates, args.nero_refine_timeout_s,
            args.grasp_target_valid_plans, args.execute) == (preflight, refine_s, 3, True)
    assert not session.ready  # Smaller probes cannot replace the mandatory side.


@pytest.mark.parametrize('side_registration_keep', [True, False])
def test_failed_capture_skips_probes_but_new_scene_rechecks_even_if_target_rejected(tmp_path, side_registration_keep):
    from real_nero_grasp.active_view import _update_grasp_feedback
    args = config()
    session = ActiveGraspFeedback(args)
    args._active_feedback_session = session
    center = SimpleNamespace(points=contacts())
    scene = SimpleNamespace(points=contacts() + 1.)
    captures = []
    def update(*a, **kw):
        session.version += 1
        session.generation_calls += 1
    session.update = mock.Mock(side_effect=update)
    def register(names, targets, scenes, *a):
        return targets, scenes, [True] + [side_registration_keep] * (len(names)-1)
    hooks = SimpleNamespace(register_active_fusion_views=mock.Mock(side_effect=register),
        diagnose_active_fusion=lambda names, *a:[True]*len(names),
        merge_clouds=mock.Mock(side_effect=lambda clouds, *a, **kw:clouds[0]))
    def run():
        _update_grasp_feedback(None, args, center, scene, np.eye(4), captures,
                               np.eye(4), None, None, {}, tmp_path, hooks)
    with mock.patch('real_nero_grasp.active_view.transform_cloud', side_effect=lambda cloud, *_:cloud), \
         mock.patch('real_nero_grasp.active_view._fuse_active_target_clouds', return_value=(None, center)):
        run()
        session.attempts = 1  # Rejected capture still spends an attempt.
        run()
        assert session.update.call_count == 1 and session.version == 1
        assert session.valid_sides == 0 and not session.ready
        assert session.unchanged_updates[0]['attempt'] == 1
        assert hooks.register_active_fusion_views.call_count == 1
        captures.append(('side1', SimpleNamespace(points=contacts()+.1), None, None,
                         np.eye(4), SimpleNamespace(points=contacts()+2.)))
        session.attempts = 2
        run()
        assert session.update.call_count == 2 and session.version == 2
        assert hooks.register_active_fusion_views.call_count == 2
        assert session.valid_sides == int(side_registration_keep)
        assert len(hooks.merge_clouds.call_args.args[0]) == 2
        assert session.attempts == 2
        # Target-quality rejection can archive obstacle geometry without
        # appending a target tuple. It must invalidate the no-new-input cache.
        rejected = tmp_path/'multiview_raw/rejected_side'
        rejected.mkdir(parents=True)
        (rejected/'metadata.json').write_text(json.dumps(dict(
            view_label='side2', status=dict(capture_quality_keep=False))))
        run()
        assert session.update.call_count == 3 and session.version == 3
        assert session.valid_sides == int(side_registration_keep)


def test_timeout_never_qualifies_a_candidate(tmp_path):
    now = [0.]
    session = ActiveGraspFeedback(config(), clock=lambda: now[0])
    session.valid_sides = 1
    session.measured_base = contacts()
    now[0] = session.args.feedback_timeout + 1.
    session.assess([row()], np.eye(4), tmp_path)
    assert session.expired and not session.ready


@pytest.mark.parametrize('ready_after_one, expected_attempts', [(True, 1), (False, 2)])
def test_live_side_loop_early_stop_and_attempt_budget(ready_after_one, expected_attempts):
    # Reuse the production active-view harness, including exact return paths,
    # accepted/rejected captures and a late preflight rejection candidate.
    control = importlib.import_module('test_active_view_control_flow')
    args = control.active_args()
    session = SimpleNamespace(ready=False, expired=False, valid_sides=0,
                              attempts=0, targets=np.ones((1, 3)), write_summary=lambda *_: None,
                              record_view_attempt=lambda *_: None,
                              rank_views=lambda candidates, *_: candidates)
    args._active_feedback_session = session
    args.active_grasp_feedback = True
    def update(*a, **kw):
        args.active_side_max_capture_attempts = 2
        session.valid_sides = len(a[5])
        session.ready = bool(ready_after_one and session.valid_sides)
    with mock.patch.object(control, 'active_args', return_value=args), \
         mock.patch('real_nero_grasp.active_view._update_grasp_feedback', side_effect=update):
        result, arm, center_q, executed, attempted, _ = control.ActiveViewControlFlowTest()._run_late_side_rejection()
    assert session.attempts == expected_attempts
    assert sum('outbound' in label for label in attempted) == expected_attempts
    np.testing.assert_array_equal(arm.q, center_q)
    assert result.kept_view_count >= 2


def test_flag_is_opt_in_and_enforces_two_view_requirement():
    args = config()
    assert args.grasp_required_view_count == 2
    assert args.active_side_max_capture_attempts == 2 and not args.active_multiview_gate


def test_missing_feedback_session_cannot_bypass_observation_workflow():
    from real_nero_grasp.grasp_workflow import plan_and_maybe_execute_grasp
    result = plan_and_maybe_execute_grasp(None, config(), None, None, None, None, None, '/tmp', {}, hooks=None)
    assert result.status.value == 'configuration_failed' and not result.motion_sent


def test_workflow_refuses_center_only_feedback_result(tmp_path):
    from dataclasses import fields
    from real_nero_grasp.workflow import WorkflowHooks, run_single_grasp, record_workflow_motion, with_workflow_motion
    from real_nero_grasp.types import ActiveViewResult
    args = config()
    arm = mock.MagicMock()
    arm.get_joint_angles.return_value = np.zeros(7)
    manager = mock.MagicMock()
    manager.__enter__.return_value = arm
    captured = (np.zeros((8, 8, 3), np.uint8), object(), object(), str(tmp_path), {'label': 'banana'}, {})
    dependencies = {field.name: mock.Mock() for field in fields(WorkflowHooks)}
    dependencies.update(arm_factory=mock.Mock(return_value=manager),
                        capture_target_cloud_with_retries=mock.Mock(return_value=captured),
                        maybe_clear_obstacles_before_active=mock.Mock(return_value=captured),
                        T_base_camera_from_flange=mock.Mock(return_value=np.eye(4)),
                        run_optional_active_view=mock.Mock(return_value=ActiveViewResult(
                            captured[1], captured[2], np.eye(4), kept_view_count=1, reason='center_view_only')),
                        record_workflow_motion=record_workflow_motion, with_workflow_motion=with_workflow_motion)
    result = run_single_grasp(args, hooks=WorkflowHooks(**dependencies))
    assert result.status.value == 'capture_failed'
    dependencies['plan_and_maybe_execute_grasp'].assert_not_called()


def test_planning_only_returns_before_ui_selection_and_execution(tmp_path):
    from dataclasses import fields
    from real_nero_grasp.grasp_workflow import GraspWorkflowHooks, plan_and_maybe_execute_grasp
    from real_nero_grasp.grasp_geometry import _axis_vector
    from real_nero_grasp.cli import parse_args
    import open3d as o3d
    with mock.patch('sys.argv', ['test', '--phase', 'grasp', '--calib-json', '/tmp/test.json']):
        args = parse_args()
    target = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(contacts()))
    arm = mock.Mock(spec=['get_tcp_pose'])
    arm.get_tcp_pose.return_value = np.zeros(6)
    candidate = row()
    dependencies = {field.name: mock.Mock() for field in fields(GraspWorkflowHooks)}
    dependencies.update(axis_vector=_axis_vector,
        get_graspnet_helpers=lambda: (lambda *a, **kw: ({}, None), lambda *a, **kw: ([candidate['grasp']], [candidate])),
        annotate_grasp_candidate_physical_aperture=lambda rows, *a, **kw: rows,
        annotate_grasp_candidate_reachability=lambda rows, *a, **kw: rows,
        annotate_grasp_candidate_workflow_preflight=lambda rows, *a, **kw: rows)
    with mock.patch('real_nero_grasp.pointcloud_fusion.save_cloud'):
        result = plan_and_maybe_execute_grasp(arm, args, target, target, np.eye(4), None, None,
            str(tmp_path), {'label': 'banana'}, planning_only=True, hooks=GraspWorkflowHooks(**dependencies))
    assert isinstance(result, FeedbackPlanningSnapshot)
    for key in ('choose_default_grasp_candidate', 'confirm_motion', 'execute_prevalidated_stages',
                'preview_graspnet_pointcloud', 'preview_final_grasp_candidate'):
        dependencies[key].assert_not_called()
