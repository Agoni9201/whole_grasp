import tempfile
import unittest
import json
from pathlib import Path
from contextlib import ExitStack
from types import SimpleNamespace
from unittest import mock

import numpy as np

import real_nero_single_grasp_runner as runner
from real_nero_grasp.types import MotionPreflightRejected


class FakeArm:
    def __init__(self):
        self.q = np.zeros(7, dtype=float)

    def get_joint_angles(self):
        return self.q.copy()

    def get_flange_pose(self):
        return np.zeros(6, dtype=float)

    def get_tcp_pose(self):
        return np.zeros(6, dtype=float)


def active_args():
    return SimpleNamespace(
        active_view=True,
        _force_active_multiview=True,
        active_center_first=True,
        active_center_height=0.34,
        calib_json="/tmp/test_calibration.json",
        handeye_convention="flange_camera_xyzw",
        active_fixed_target_center=True,
        execute=True,
        yes=True,
        yes_active_view=False,
        active_path_guard=True,
        active_up_and_over=False,
        motion_runtime_collision_guard=True,
        motion_self_filter=False,
        active_confirm_each_view=True,
        motion_mode="ik_joint",
        active_side_view_count=1,
        active_min_side_disparity_deg=45.0,
        active_restore_center=True,
        active_fusion_reject_bad_views=True,
        active_fusion_voxel=0.001,
        scene_voxel_size=0.003,
        fusion_target_outlier_filter=False,
        fusion_scene_outlier_filter=False,
        outlier_neighbors=20,
        outlier_std_ratio=2.0,
        active_center_max_joint_delta_deg=45.0,
        active_center_max_joint_l2_deg=90.0,
        active_side_max_joint_delta_deg=45.0,
        active_side_max_joint_l2_deg=75.0,
        active_center_joint_match_deg=2.0,
        tcp_offset=[0.0] * 6,
    )


class ActiveViewControlFlowTest(unittest.TestCase):
    def _run_late_side_rejection(self, *, motion_sent=False, scene_failure=False,
                                 safety_fault=False, preview_hook=None, archive_enabled=True):
        arm = FakeArm()
        args = active_args()
        args.save_active_raw_views = archive_enabled
        args.active_side_view_count = 2
        args.active_side_max_capture_attempts = 3
        center_q = np.deg2rad([10, 5, 0, 10, 0, 0, 0])
        side_q = center_q + np.deg2rad([5, 5, 5, 5, 0, 0, 0])
        center_pose = np.array([0.0, 0.0, 0.5, 0.0, 0.0, 0.0])
        cloud = SimpleNamespace(points=np.array([[0.10, 0.0, 0.20]]))
        safety_scene = runner.MotionSafetyScene(
            full_scene_base=np.array([[10.0, 10.0, 10.0]]),
            environment_base=np.array([[10.0, 10.0, 10.0]]),
            target_base=np.zeros((0, 3)),
            map_id="late_side_rejection",
        )
        candidates = [
            {
                "name": f"side_candidate_{index}",
                "reason": "ok", "score": 1.0,
                "command_type": "flange", "flange": center_pose,
                "planned_joint_target": side_q,
                "view_angle_rad": angle,
                "angle_deg": float(np.rad2deg(angle)),
            }
            for index, angle in enumerate((0.5, 2.3, 4.1), start=1)
        ]
        executed = []
        attempted = []
        outbound_count = 0

        def fake_plan(_arm, goal_flange, q_start, _scene, _target, _args,
                      label, *_limits, goal_joints=None, **_kwargs):
            q_goal = center_q if goal_joints is None else goal_joints
            stage = runner.PlannedJointStage(
                label=f"{label}_direct", tcp_pose=np.zeros(6),
                flange_pose=np.asarray(goal_flange), joints=q_goal,
                phase="free_open",
            )
            return runner.GraspMotionPlan(
                start_joints=np.asarray(q_start), before_close=(stage,),
                after_close=(), after_release=(), empty_recovery=(),
                safety_scene=safety_scene,
            )

        def fake_execute(_arm, stages, _start, _plan, _args):
            nonlocal outbound_count
            label = stages[-1].label
            attempted.append(label)
            if "outbound" in label:
                outbound_count += 1
                if outbound_count == 3:
                    if safety_fault:
                        raise runner.SafetyStopTriggered("runtime safety stop")
                    raise MotionPreflightRejected(
                        "finger clearance 4.99mm < required 5.0mm",
                        motion_sent=motion_sent,
                    )
            executed.append(label)
            _arm.q = np.asarray(stages[-1].joints).copy()
            return np.zeros(6), _arm.q.copy()

        def fake_motion_capture(cloud_, _T, _q, _args, label):
            if scene_failure and label == "active_motion_scene_2":
                raise runner.UnsafeMotionTarget("new safety scene overflow")
            return cloud_

        capture_result = (
            np.zeros((2, 2, 3), dtype=np.uint8), cloud, cloud, cloud,
            "/tmp", {"label": "bottle"}, object(),
        )
        with tempfile.TemporaryDirectory() as out_dir, ExitStack() as stack:
            archive_gate_counts = []
            def fake_overlap(*_a, **_kw):
                manifest = Path(out_dir) / "multiview_raw" / "manifest.jsonl"
                archive_gate_counts.append(len(manifest.read_text().splitlines()) if manifest.exists() else 0)
                return len(archive_gate_counts) == 1
            patches = {
                "_target_center_base_from_cloud": {"return_value": np.zeros(3)},
                "_target_center_overhead_flange_pose_candidates": {
                    "return_value": [{"flange_pose": center_pose,
                                      "T_base_camera": np.eye(4),
                                      "roll_rad": 0.0, "rotation_error_deg": 0.0}],
                },
                "plan_active_direct_route": {"side_effect": fake_plan},
                "execute_prevalidated_stages": {"side_effect": fake_execute},
                "_urdf_chain_geometry": {
                    "return_value": (np.zeros((8, 3)), np.eye(4)),
                },
                "T_base_camera_from_flange": {"return_value": np.eye(4)},
                "capture_active_projected_target_cloud": {
                    "return_value": capture_result,
                },
                "_build_active_view_candidates": {"return_value": candidates},
                "_active_side_selection_order": {
                    "side_effect": lambda items, _args: items,
                },
                "_active_side_capture_overlap_ok": {"side_effect": fake_overlap},
                "_active_motion_scene_points": {
                    "return_value": np.array([[10.0, 10.0, 10.0]]),
                },
                "_active_motion_capture_base": {"side_effect": fake_motion_capture},
                "transform_cloud": {"side_effect": lambda item, _T: item},
                "register_active_fusion_views": {
                    "side_effect": lambda names, targets, scenes, *_: (
                        targets, scenes, [True] * len(targets)),
                },
                "diagnose_active_fusion": {
                    "side_effect": lambda names, targets, *_: [True] * len(targets),
                },
                "merge_clouds": {"return_value": cloud},
                "crop_scene_cloud_near_target": {"return_value": cloud},
                "save_cloud": {}, "preview_fused_cloud": {"side_effect": preview_hook},
            }
            mocks = {name: stack.enter_context(mock.patch.object(runner, name, **kwargs))
                     for name, kwargs in patches.items()}
            result = runner.run_optional_active_view(
                arm, args, cloud, cloud, np.eye(4),
                {"class_label": "bottle"}, out_dir,
            )
            archive = Path(out_dir) / "multiview_raw"
            manifest = archive / "manifest.jsonl"
            entries = [json.loads(line) for line in manifest.read_text().splitlines()] if manifest.exists() else []
            mocks["archive_records"] = [
                json.loads((archive / entry["metadata_path"]).read_text()) for entry in entries]
            mocks["archive_files_exist"] = all(
                (archive / entry["view_dir"] / "target_camera.ply").is_file()
                and (archive / entry["view_dir"] / "scene_base.ply").is_file()
                for entry in entries)
            mocks["archive_gate_counts"] = archive_gate_counts
        return result, arm, center_q, executed, attempted, mocks

    def test_late_preflight_rejection_preserves_accepted_views_and_raw_scenes(self):
        result, arm, center_q, executed, attempted, mocks = self._run_late_side_rejection()
        self.assertTrue(result.used_multiview)
        self.assertEqual(result.kept_view_count, 2)
        self.assertEqual(result.reason, "fused")
        self.assertTrue(result.restore_ok)
        np.testing.assert_allclose(arm.q, center_q)
        self.assertEqual(len(executed), 5)
        self.assertEqual(len(attempted), 6)
        self.assertEqual(mocks["capture_active_projected_target_cloud"].call_count, 3)
        self.assertEqual(mocks["register_active_fusion_views"].call_args.args[0],
                         ["center", "side1"])
        self.assertEqual(result.motion_environment_map.source_view_names,
                         ("initial", "center", "side_capture_1", "side_capture_2"))

    def test_partial_route_preflight_failure_does_not_resume_fusion(self):
        result, _arm, _q, _executed, _attempted, mocks = self._run_late_side_rejection(
            motion_sent=True,
        )
        self.assertEqual(result.reason, "active_view_unsafe")
        mocks["register_active_fusion_views"].assert_not_called()

    def test_every_captured_view_is_archived_before_gate_including_rejected_side(self):
        _result, _arm, _q, _executed, _attempted, mocks = self._run_late_side_rejection()
        records = mocks["archive_records"]
        self.assertEqual([r["view_label"] for r in records], ["initial", "center", "side1", "side2"])
        self.assertEqual(mocks["archive_gate_counts"], [3, 4])
        self.assertTrue(mocks["archive_files_exist"])
        self.assertTrue(records[2]["status"]["fusion_used"])
        self.assertFalse(records[3]["status"]["capture_quality_keep"])
        self.assertFalse(records[3]["status"]["fusion_used"])
        self.assertTrue(all(not r["registration_applied"] and not r["fusion_applied"] for r in records))

    def test_archive_disable_does_not_change_active_fusion(self):
        result, _arm, _q, _executed, _attempted, mocks = self._run_late_side_rejection(archive_enabled=False)
        self.assertTrue(result.used_multiview)
        self.assertEqual(result.kept_view_count, 2)
        self.assertEqual(mocks["archive_records"], [])

    def test_new_scene_failure_does_not_resume_fusion(self):
        result, _arm, _q, _executed, attempted, mocks = self._run_late_side_rejection(
            scene_failure=True,
        )
        self.assertEqual(result.reason, "active_view_unsafe")
        self.assertEqual(len(attempted), 5)
        mocks["register_active_fusion_views"].assert_not_called()

    def test_runtime_safety_stop_is_not_treated_as_candidate_rejection(self):
        with self.assertRaises(runner.SafetyStopTriggered):
            self._run_late_side_rejection(safety_fault=True)

    def test_terminal_map_failures_do_not_fall_back_or_open_preview(self):
        from real_nero_grasp.planning_process import PlanningProcessError
        for error in (runner.SafetyStopTriggered, runner.ControlledMotionStopTriggered,
                      PlanningProcessError):
            with self.subTest(error=error.__name__):
                preview = mock.Mock()
                with mock.patch.object(runner, 'fuse_dynamic_environment_clouds',
                                       side_effect=error('terminal map failure')):
                    with self.assertRaisesRegex(error, 'terminal map failure'):
                        self._run_late_side_rejection(preview_hook=preview)
                preview.assert_not_called()

    def test_fault_raised_during_preview_propagates_before_restore(self):
        from real_nero_grasp import active_view
        state = {'fault': False}
        def preview(*a):
            state['fault'] = True
        def check(_args):
            if state['fault']:
                raise runner.SafetyStopTriggered('preview-time MIT fault')
        with mock.patch.object(active_view, 'planning_health_check', side_effect=check):
            with self.assertRaisesRegex(runner.SafetyStopTriggered, 'preview-time MIT fault'):
                self._run_late_side_rejection(preview_hook=preview)

    def test_center_verification_refuses_nonfinite_feedback(self):
        arm = FakeArm()
        arm.q[0] = np.nan
        self.assertFalse(runner._active_center_joint_state_is_verified(
            arm, np.zeros(7), active_args(), "rejected candidate",
        ))

    def test_unreachable_overhead_center_uses_verified_tilted_anchor(self):
        arm = FakeArm()
        args = active_args()
        # Exercise the larger moved-camera fallback specifically; same-position
        # optical relaxation has its own focused geometry/IK regression test.
        args.active_center_optical_relaxation = False
        # Keep this test small and deterministic; the production defaults use
        # a denser grid after the overhead candidate has failed.
        args.active_center_fallback_tilts = "20"
        args.active_center_fallback_azimuth_step_deg = 180.0
        args.active_center_fallback_max_positions = 2
        args.active_center_fallback_roll_plan_candidates = 1
        args.active_orbit_min_radius = 0.30
        center_pose = np.array([0.0, 0.0, 0.5, 0.0, 0.0, 0.0])
        fallback_q = np.deg2rad([4.0, 3.0, 2.0, 1.0, 0.0, 0.0, -1.0])
        cloud = SimpleNamespace(
            points=np.array([[0.10, 0.0, 0.20]], dtype=float)
        )
        safety_scene = runner.MotionSafetyScene(
            full_scene_base=np.array([[10.0, 10.0, 10.0]]),
            environment_base=np.array([[10.0, 10.0, 10.0]]),
            target_base=np.zeros((0, 3)),
            map_id="tilted_center_anchor",
        )
        planned_labels = []

        def fake_plan(
            _arm,
            goal_flange,
            _q_start,
            _scene,
            _target,
            _args,
            label,
            *_limits,
            **_kwargs,
        ):
            planned_labels.append(str(label))
            if label == "active_center_plan":
                raise runner.PlanNotExecutable("overhead IK has no solution")
            stage = runner.PlannedJointStage(
                label=str(label),
                tcp_pose=np.zeros(6),
                flange_pose=np.asarray(goal_flange, dtype=float),
                joints=fallback_q,
                phase="free_open",
            )
            return runner.GraspMotionPlan(
                start_joints=np.zeros(7),
                before_close=(stage,),
                after_close=tuple(),
                after_release=tuple(),
                empty_recovery=tuple(),
                safety_scene=safety_scene,
            )

        def fake_execute(_arm, stages, _start, _plan, _args):
            _arm.q = np.asarray(stages[-1].joints, dtype=float).copy()
            return np.zeros(6), _arm.q.copy()

        def fake_lookat(cam_pos, _target, _T_camera_flange, _current, **_kwargs):
            T = np.eye(4)
            T[:3, 3] = np.asarray(cam_pos, dtype=float)
            return [{
                "flange_pose": np.r_[np.asarray(cam_pos, dtype=float), [0.0, 0.0, 0.0]],
                "T_base_camera": T,
                "roll_rad": 0.0,
                "rotation_error_deg": 0.0,
            }]

        with mock.patch.object(
            runner, "_target_center_base_from_cloud", return_value=np.zeros(3)
        ), mock.patch.object(
            runner,
            "_target_center_overhead_flange_pose_candidates",
            return_value=[{
                "flange_pose": center_pose,
                "T_base_camera": np.eye(4),
                "roll_rad": 0.0,
                "rotation_error_deg": 0.0,
            }],
        ), mock.patch.object(
            runner, "handeye_transform", return_value=np.eye(4)
        ), mock.patch.object(
            runner, "_look_at_camera_pose_flange_roll_candidates", side_effect=fake_lookat
        ), mock.patch.object(
            runner, "plan_active_direct_route", side_effect=fake_plan
        ), mock.patch.object(
            runner, "execute_prevalidated_stages", side_effect=fake_execute
        ), mock.patch.object(
            runner, "T_base_camera_from_flange", return_value=np.eye(4)
        ), mock.patch.object(
            runner,
            "capture_active_projected_target_cloud",
            side_effect=RuntimeError("capture is intentionally stopped after fallback anchor"),
        ), mock.patch.object(
            runner,
            "_active_motion_scene_points",
            return_value=np.array([[10.0, 10.0, 10.0]]),
        ), mock.patch.object(
            runner, "transform_cloud", side_effect=lambda item, _T: item
        ):
            result = runner.run_optional_active_view(
                arm,
                args,
                cloud,
                cloud,
                np.eye(4),
                {"class_label": "banana"},
                "/tmp",
            )

        self.assertEqual(result.reason, "active_capture_failed")
        self.assertTrue(result.motion_sent)
        self.assertEqual(planned_labels[0], "active_center_plan")
        self.assertTrue(planned_labels[1].startswith("active_center_fallback_t20_"))
        np.testing.assert_allclose(arm.q, fallback_q)

    def test_center_capture_failure_reports_verified_center_state(self):
        arm = FakeArm()
        args = active_args()
        center_q = np.deg2rad([5.0, 4.0, 3.0, 2.0, 1.0, 0.0, -1.0])
        center_pose = np.array([0.0, 0.0, 0.5, 0.0, 0.0, 0.0])
        cloud = SimpleNamespace(
            points=np.array([[0.10, 0.0, 0.20]], dtype=float)
        )
        safety_scene = runner.MotionSafetyScene(
            full_scene_base=np.array([[10.0, 10.0, 10.0]]),
            environment_base=np.array([[10.0, 10.0, 10.0]]),
            target_base=np.zeros((0, 3)),
            map_id="capture_failure_center_state",
        )
        stage = runner.PlannedJointStage(
            label="active_center_plan_direct",
            tcp_pose=np.zeros(6),
            flange_pose=center_pose,
            joints=center_q,
            phase="free_open",
        )
        center_plan = runner.GraspMotionPlan(
            start_joints=np.zeros(7),
            before_close=(stage,),
            after_close=tuple(),
            after_release=tuple(),
            empty_recovery=tuple(),
            safety_scene=safety_scene,
        )

        motion_capture_filters = []

        def fake_execute(_arm, stages, _start, _plan, _args):
            _arm.q = np.asarray(stages[-1].joints, dtype=float).copy()
            return np.zeros(6), _arm.q.copy()

        def fake_motion_capture_filter(
            capture_cloud,
            _T_base_camera,
            capture_joints,
            _args,
            label,
        ):
            motion_capture_filters.append(
                (str(label), np.asarray(capture_joints, dtype=float).copy())
            )
            return capture_cloud

        with mock.patch.object(
            runner, "_target_center_base_from_cloud", return_value=np.zeros(3)
        ), mock.patch.object(
            runner,
            "_target_center_overhead_flange_pose_candidates",
            return_value=[
                {
                    "flange_pose": center_pose,
                    "T_base_camera": np.eye(4),
                    "roll_rad": 0.0,
                    "rotation_error_deg": 0.0,
                }
            ],
        ), mock.patch.object(
            runner, "plan_active_direct_route", return_value=center_plan
        ), mock.patch.object(
            runner, "execute_prevalidated_stages", side_effect=fake_execute
        ), mock.patch.object(
            runner, "T_base_camera_from_flange", return_value=np.eye(4)
        ), mock.patch.object(
            runner,
            "capture_active_projected_target_cloud",
            side_effect=RuntimeError("duplicate target identity"),
        ), mock.patch.object(
            runner,
            "_active_motion_scene_points",
            return_value=np.array([[10.0, 10.0, 10.0]]),
        ), mock.patch.object(
            runner,
            "_active_motion_capture_base",
            side_effect=fake_motion_capture_filter,
        ), mock.patch.object(
            runner, "transform_cloud", side_effect=lambda item, _T: item
        ):
            result = runner.run_optional_active_view(
                arm,
                args,
                cloud,
                cloud,
                np.eye(4),
                {"class_label": "carrot"},
                "/tmp",
            )

        self.assertEqual(result.reason, "active_capture_failed")
        self.assertTrue(result.motion_sent)
        self.assertTrue(result.restore_ok)
        np.testing.assert_allclose(arm.q, center_q)
        self.assertEqual(
            [label for label, _joints in motion_capture_filters],
            [
                "active_motion_scene_initial",
                "active_motion_environment_single",
            ],
        )
        # The returned cloud is the original observation, so its self-filter
        # must use the joints at function entry, not the later center pose.
        np.testing.assert_allclose(
            motion_capture_filters[0][1],
            np.zeros(7),
        )
        np.testing.assert_allclose(
            motion_capture_filters[1][1],
            np.zeros(7),
        )

    def test_side_map_overflow_after_center_capture_keeps_center_fallback(self):
        arm = FakeArm()
        args = active_args()
        center_q = np.deg2rad([5.0, 4.0, 3.0, 2.0, 1.0, 0.0, -1.0])
        center_pose = np.array([0.0, 0.0, 0.5, 0.0, 0.0, 0.0])
        target_cloud = SimpleNamespace(
            points=np.array([[0.10, 0.0, 0.20]], dtype=float)
        )
        scene_cloud = SimpleNamespace(
            points=np.array([[0.20, 0.0, 0.20]], dtype=float)
        )
        obstacle_cloud = SimpleNamespace(
            points=np.array([[0.30, 0.0, 0.20]], dtype=float)
        )
        safety_scene = runner.MotionSafetyScene(
            full_scene_base=np.array([[10.0, 10.0, 10.0]]),
            environment_base=np.array([[10.0, 10.0, 10.0]]),
            target_base=np.zeros((0, 3)),
            map_id="center_overflow_fallback",
        )
        stage = runner.PlannedJointStage(
            label="active_center_plan_direct",
            tcp_pose=np.zeros(6),
            flange_pose=center_pose,
            joints=center_q,
            phase="free_open",
        )
        center_plan = runner.GraspMotionPlan(
            start_joints=np.zeros(7),
            before_close=(stage,),
            after_close=tuple(),
            after_release=tuple(),
            empty_recovery=tuple(),
            safety_scene=safety_scene,
        )
        execute_labels = []

        def fake_execute(_arm, stages, _start, _plan, _args):
            execute_labels.append(stages[-1].label)
            _arm.q = np.asarray(stages[-1].joints, dtype=float).copy()
            return np.zeros(6), _arm.q.copy()

        center_capture_result = (
            np.zeros((2, 2, 3), dtype=np.uint8),
            target_cloud,
            scene_cloud,
            obstacle_cloud,
            "/tmp",
            {"label": "radish"},
            object(),
        )
        overflow = runner.UnsafeMotionTarget(
            "active safety cloud has 55116 occupied voxels > cap=50000"
        )

        with tempfile.TemporaryDirectory() as out_dir, mock.patch.object(
            runner, "_target_center_base_from_cloud", return_value=np.zeros(3)
        ), mock.patch.object(
            runner,
            "_target_center_overhead_flange_pose_candidates",
            return_value=[
                {
                    "flange_pose": center_pose,
                    "T_base_camera": np.eye(4),
                    "roll_rad": 0.0,
                    "rotation_error_deg": 0.0,
                }
            ],
        ), mock.patch.object(
            runner, "plan_active_direct_route", return_value=center_plan
        ), mock.patch.object(
            runner, "execute_prevalidated_stages", side_effect=fake_execute
        ), mock.patch.object(
            runner, "T_base_camera_from_flange", return_value=np.eye(4)
        ), mock.patch.object(
            runner,
            "capture_active_projected_target_cloud",
            return_value=center_capture_result,
        ), mock.patch.object(
            runner,
            "_active_motion_scene_points",
            side_effect=[np.array([[10.0, 10.0, 10.0]]), overflow],
        ), mock.patch.object(
            runner, "_build_active_view_candidates"
        ) as build_candidates, mock.patch.object(
            runner, "transform_cloud", side_effect=lambda item, _T: item
        ), mock.patch.object(
            runner,
            "register_active_fusion_views",
            side_effect=lambda names, targets, scenes, *_: (
                targets,
                scenes,
                [True] * len(targets),
            ),
        ), mock.patch.object(
            runner,
            "diagnose_active_fusion",
            side_effect=lambda names, targets, *_: [True] * len(targets),
        ), mock.patch.object(
            runner,
            "merge_clouds",
            side_effect=lambda clouds, *_, **__: clouds[0],
        ), mock.patch.object(
            runner,
            "crop_scene_cloud_near_target",
            side_effect=lambda cloud, *_args, **_kwargs: cloud,
        ), mock.patch.object(
            runner, "save_cloud"
        ), mock.patch.object(
            runner, "preview_fused_cloud"
        ):
            result = runner.run_optional_active_view(
                arm,
                args,
                target_cloud,
                scene_cloud,
                np.eye(4),
                {"class_label": "radish"},
                out_dir,
            )

        self.assertFalse(result.used_multiview)
        self.assertEqual(result.kept_view_count, 1)
        self.assertEqual(result.reason, "center_view_only")
        self.assertTrue(result.restore_ok)
        self.assertTrue(result.motion_sent)
        self.assertEqual(execute_labels, ["active_center_plan_direct"])
        build_candidates.assert_not_called()
        np.testing.assert_allclose(arm.q, center_q)

    def test_each_side_returns_to_exact_center_joints(self):
        arm = FakeArm()
        args = active_args()
        center_q = np.deg2rad([10, 5, 0, 10, 0, 0, 0])
        side_q = np.deg2rad([15, 10, 5, 15, 0, 0, 0])
        center_pose = np.zeros(6, dtype=float)
        center_pose[2] = 0.5
        rejected_center_pose = center_pose.copy()
        rejected_center_pose[2] = 0.6
        side_pose = np.zeros(6, dtype=float)
        side_pose[:3] = [0.1, 0.0, 0.45]
        candidate = {
            "name": "safe_side",
            "reason": "ok",
            "score": 1.0,
            "command_type": "flange",
            "flange": side_pose,
            "planned_joint_target": side_q,
            "view_angle_rad": 0.5,
            "angle_deg": 30.0,
        }
        cloud = SimpleNamespace(points=np.zeros((4, 3), dtype=float))
        center_target_cloud = SimpleNamespace(
            points=np.array([[0.10, 0.0, 0.20]], dtype=float)
        )
        center_scene_cloud = SimpleNamespace(
            points=np.array([[0.20, 0.0, 0.20]], dtype=float)
        )
        side_target_cloud = SimpleNamespace(
            points=np.array([[0.30, 0.0, 0.20]], dtype=float)
        )
        side_scene_sentinel = np.array([0.91, 0.82, 0.73], dtype=float)
        side_scene_cloud = SimpleNamespace(
            points=side_scene_sentinel.reshape(1, 3)
        )
        calls = []
        plan_calls = []
        plan_scenes = {}
        plan_retreat_heights = {}
        plan_single_stage_returns = {}
        motion_capture_filters = []
        safety_scene = runner.MotionSafetyScene(
            full_scene_base=np.array([[10.0, 10.0, 10.0]]),
            environment_base=np.array([[10.0, 10.0, 10.0]]),
            target_base=np.zeros((0, 3)),
            map_id="active_control_flow",
        )

        def fake_direct_plan(
            _arm,
            _goal_flange,
            q_start,
            _scene,
            _target,
            _args,
            label,
            _max_delta,
            _max_l2,
            goal_joints=None,
            scene_capture_joints=None,
            initial_retreat_height=0.0,
            enforce_single_stage_return=False,
        ):
            del scene_capture_joints
            plan_calls.append(label)
            plan_scenes[label] = np.asarray(_scene, dtype=float).copy()
            plan_retreat_heights[label] = float(initial_retreat_height)
            plan_single_stage_returns[label] = bool(
                enforce_single_stage_return
            )
            if (
                label == "active_center_plan"
                and np.isclose(float(np.asarray(_goal_flange)[2]), 0.6)
            ):
                raise runner.PlanNotExecutable("first equivalent roll has no IK")
            if goal_joints is not None:
                q_goal = np.asarray(goal_joints, dtype=float).copy()
            elif label == "active_center_plan":
                q_goal = center_q.copy()
            else:
                q_goal = side_q.copy()
            stage = runner.PlannedJointStage(
                label=f"{label}_direct",
                tcp_pose=np.zeros(6),
                flange_pose=np.zeros(6),
                joints=q_goal,
                phase="free_open",
            )
            return runner.GraspMotionPlan(
                start_joints=np.asarray(q_start, dtype=float).copy(),
                before_close=(stage,),
                after_close=tuple(),
                after_release=tuple(),
                empty_recovery=tuple(),
                safety_scene=safety_scene,
            )

        def fake_execute(_arm, stages, _start, _plan, _args):
            calls.append(stages[-1].label)
            _arm.q = np.asarray(stages[-1].joints, dtype=float).copy()
            return np.zeros(6), _arm.q.copy()

        def fake_motion_capture_filter(
            capture_cloud,
            _T_base_camera,
            capture_joints,
            _args,
            label,
        ):
            motion_capture_filters.append(
                (
                    str(label),
                    np.asarray(capture_joints, dtype=float).copy(),
                )
            )
            return capture_cloud

        center_capture_result = (
            np.zeros((2, 2, 3), dtype=np.uint8),
            center_target_cloud,
            center_scene_cloud,
            cloud,
            "/tmp",
            {"label": "radish"},
            object(),
        )
        side_capture_result = (
            np.zeros((2, 2, 3), dtype=np.uint8),
            side_target_cloud,
            side_scene_cloud,
            cloud,
            "/tmp",
            {"label": "radish"},
            object(),
        )

        with tempfile.TemporaryDirectory() as out_dir, mock.patch.object(
            runner, "_target_center_base_from_cloud", return_value=np.zeros(3)
        ), mock.patch.object(
            runner,
            "_target_center_overhead_flange_pose_candidates",
            return_value=[
                {
                    "flange_pose": rejected_center_pose,
                    "T_base_camera": np.eye(4),
                    "roll_rad": -0.25,
                    "rotation_error_deg": 1.0,
                },
                {
                    "flange_pose": center_pose,
                    "T_base_camera": np.eye(4),
                    "roll_rad": 0.0,
                    "rotation_error_deg": 0.0,
                }
            ],
        ), mock.patch.object(
            runner, "plan_active_direct_route", side_effect=fake_direct_plan
        ), mock.patch.object(
            runner, "execute_prevalidated_stages", side_effect=fake_execute
        ), mock.patch.object(
            runner,
            "_urdf_chain_geometry",
            return_value=(np.zeros((8, 3)), np.eye(4)),
        ), mock.patch.object(
            runner, "T_base_camera_from_flange", return_value=np.eye(4)
        ), mock.patch.object(
            runner,
            "capture_active_projected_target_cloud",
            side_effect=[center_capture_result, side_capture_result],
        ), mock.patch.object(
            runner, "_build_active_view_candidates", return_value=[candidate]
        ), mock.patch.object(
            runner, "_active_side_capture_overlap_ok", return_value=True
        ), mock.patch.object(
            runner, "transform_cloud", side_effect=lambda item, _T: item
        ), mock.patch.object(
            runner,
            "_active_motion_capture_base",
            side_effect=fake_motion_capture_filter,
        ), mock.patch.object(
            runner,
            "register_active_fusion_views",
            side_effect=lambda names, targets, scenes, *_: (
                targets,
                scenes,
                [True] * len(targets),
            ),
        ), mock.patch.object(
            runner,
            "diagnose_active_fusion",
            side_effect=lambda names, targets, *_: [True] * len(targets),
        ), mock.patch.object(
            runner, "merge_clouds", return_value=cloud
        ), mock.patch.object(
            runner, "crop_scene_cloud_near_target", return_value=cloud
        ), mock.patch.object(
            runner, "save_cloud"
        ), mock.patch.object(
            runner, "preview_fused_cloud"
        ):
            result = runner.run_optional_active_view(
                arm,
                args,
                cloud,
                cloud,
                np.eye(4),
                {"class_label": "radish"},
                out_dir,
            )

        self.assertTrue(result.used_multiview)
        self.assertIsNotNone(result.motion_environment_map)
        self.assertEqual(
            result.motion_environment_map.source_view_names,
            ("initial", "center", "side_capture_1"),
        )
        self.assertEqual(result.motion_environment_map.generation, 1)
        self.assertEqual(args._motion_environment_generation, 1)
        self.assertEqual(
            calls,
            [
                "active_center_plan_direct",
                "active_side1_outbound_direct",
                "active_side1_return_center_reverse_edge1",
            ],
        )
        self.assertEqual(
            plan_calls,
            [
                "active_center_plan",
                "active_center_plan",
                "active_side1_outbound",
            ],
        )
        self.assertEqual(plan_retreat_heights["active_center_plan"], 0.0)
        self.assertEqual(plan_retreat_heights["active_side1_outbound"], 0.0)
        self.assertFalse(plan_single_stage_returns["active_center_plan"])
        self.assertFalse(plan_single_stage_returns["active_side1_outbound"])
        self.assertEqual(
            [label for label, _joints in motion_capture_filters],
            [
                "active_motion_scene_initial",
                "active_motion_scene_center",
                "active_motion_scene_1",
                "active_motion_environment_center",
                "active_motion_environment_1",
                "active_motion_environment_initial",
            ],
        )
        np.testing.assert_allclose(
            motion_capture_filters[0][1], np.zeros(7)
        )
        np.testing.assert_allclose(
            motion_capture_filters[1][1], center_q
        )
        np.testing.assert_allclose(
            motion_capture_filters[2][1], side_q
        )
        np.testing.assert_allclose(
            motion_capture_filters[3][1], center_q
        )
        np.testing.assert_allclose(
            motion_capture_filters[4][1], side_q
        )
        np.testing.assert_allclose(arm.q, center_q)


if __name__ == "__main__":
    unittest.main()
