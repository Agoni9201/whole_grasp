import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

import real_nero_single_grasp_runner as runner


def clear_candidate(mask):
    return {
        "idx": 1,
        "label": "radish",
        "clear_score": 0.75,
        "geometry_score": 0.58,
        "amodal_block_score": 1.0,
        "largest_visible_fallback": False,
        "mask": mask,
        "inst": {"label": "radish", "score": 0.96, "mask": mask},
    }


def flow_args(mask):
    return SimpleNamespace(
        enable_clear_obstacle=True,
        clear_max_rounds=1,
        clear_empty_grasp_retries=1,
        clear_score_threshold=0.35,
        clear_place_xyz=[-0.35, -0.15, 0.20],
        execute=True,
        yes=False,
        yes_clear=False,
        yes_active_view=False,
        active_view=True,
        active_multiview_gate=True,
        active_target_reselect="yolo",
        target_prompt=False,
        target_class="carrot",
        target_index=None,
        manual_click_only=False,
        select_target=False,
        post_clear_reidentify_min_coverage=0.30,
        post_clear_reidentify_min_iou=0.10,
        post_clear_reidentify_ambiguity_margin=0.15,
        _last_capture_xyz_img=np.zeros((4, 4, 3), dtype=np.float32),
        _last_target_mask=mask,
    )


class ClearFailClosedFlowTest(unittest.TestCase):
    def make_capture(self, out_dir, mask):
        return (
            np.zeros((4, 4, 3), dtype=np.uint8),
            object(),
            object(),
            out_dir,
            {"label": "carrot", "score": 0.98},
            object(),
        )

    def common_patches(self, candidate_rows):
        return mock.patch.multiple(
            runner,
            detect_clear_obstacle_candidates=mock.Mock(side_effect=candidate_rows),
            clear_candidate_is_actionable=mock.Mock(return_value=True),
            refine_xyz_inside_visible_mask=mock.Mock(side_effect=lambda xyz, *a, **k: xyz),
            cloud_from_mask=mock.Mock(return_value=object()),
            scene_cloud_from_xyz=mock.Mock(return_value=object()),
            obstacle_cloud_from_non_target_mask=mock.Mock(return_value=object()),
            save_selected_target_visuals=mock.Mock(),
            T_base_camera_from_flange=mock.Mock(return_value=np.eye(4)),
        )

    def test_single_plan_failure_uses_same_clear_target_multiview(self):
        mask = np.ones((4, 4), dtype=np.uint8)
        args = flow_args(mask)
        candidate = clear_candidate(mask)
        with tempfile.TemporaryDirectory() as out_dir:
            captured = self.make_capture(out_dir, mask)
            recaptured = self.make_capture(out_dir, mask)
            plan_results = [
                runner.GraspAttemptResult(runner.GraspAttemptStatus.PLANNING_FAILED),
                runner.GraspAttemptResult(runner.GraspAttemptStatus.SUCCESS, True),
            ]
            active = runner.ActiveViewResult(
                object(), object(), np.eye(4), True, 3, "fused", True
            )
            with self.common_patches([[candidate], []]), mock.patch.object(
                runner, "plan_and_maybe_execute_grasp", side_effect=plan_results
            ) as plan, mock.patch.object(
                runner, "run_optional_active_view", return_value=active
            ) as active_view, mock.patch.object(
                runner, "capture_target_cloud_with_retries", return_value=recaptured
            ):
                result = runner.maybe_clear_obstacles_before_active(object(), args, captured)

        self.assertIs(result, recaptured)
        self.assertEqual(plan.call_count, 2)
        clear_inst = active_view.call_args.args[5]
        self.assertEqual(clear_inst["label"], "clear_radish")
        self.assertEqual(clear_inst["class_label"], "radish")
        self.assertFalse(active_view.call_args.args[1].active_multiview_gate)
        self.assertTrue(active_view.call_args.args[1].active_strict_target_identity)

    def test_single_clear_candidates_share_one_frozen_environment_map(self):
        mask = np.ones((4, 4), dtype=np.uint8)
        args = flow_args(mask)
        args.motion_environment_map_enabled = True
        candidate = clear_candidate(mask)
        environment_map = runner.DynamicEnvironmentMap(
            hard_environment_base=np.array([[0.1, 0.0, 0.1]]),
            uncertain_environment_base=np.zeros((0, 3)),
            target_base=np.array([[0.0, 0.0, 0.1]]),
            map_id="clear_round_shared",
            generation=3,
        )
        prepared_scene = runner.MotionSafetyScene(
            full_scene_base=np.array(
                [[0.1, 0.0, 0.1], [0.0, 0.0, 0.1]]
            ),
            environment_base=np.array([[0.1, 0.0, 0.1]]),
            target_base=np.array([[0.0, 0.0, 0.1]]),
            map_id="clear_round_shared",
            environment_map=environment_map,
            generation=3,
        )
        arm = mock.Mock()
        arm.get_joint_angles.return_value = np.zeros(7)

        with tempfile.TemporaryDirectory() as out_dir:
            captured = self.make_capture(out_dir, mask)
            recaptured = self.make_capture(out_dir, mask)
            with self.common_patches([[candidate], []]), mock.patch.object(
                runner,
                "prepare_motion_safety_scene",
                return_value=prepared_scene,
            ) as prepare, mock.patch.object(
                runner,
                "plan_and_maybe_execute_grasp",
                return_value=runner.GraspAttemptResult(
                    runner.GraspAttemptStatus.SUCCESS,
                    True,
                ),
            ) as plan, mock.patch.object(
                runner,
                "capture_target_cloud_with_retries",
                return_value=recaptured,
            ):
                result = runner.maybe_clear_obstacles_before_active(
                    arm,
                    args,
                    captured,
                )

        self.assertIs(result, recaptured)
        prepare.assert_called_once()
        self.assertIs(
            plan.call_args.kwargs["motion_environment_map"],
            environment_map,
        )
        self.assertEqual(args._motion_environment_generation, 3)

    def test_unavailable_multiview_aborts_instead_of_returning_carrot(self):
        mask = np.ones((4, 4), dtype=np.uint8)
        args = flow_args(mask)
        with tempfile.TemporaryDirectory() as out_dir:
            captured = self.make_capture(out_dir, mask)
            with self.common_patches([[clear_candidate(mask)]]), mock.patch.object(
                runner,
                "plan_and_maybe_execute_grasp",
                return_value=runner.GraspAttemptResult(runner.GraspAttemptStatus.PLANNING_FAILED),
            ), mock.patch.object(
                runner,
                "run_optional_active_view",
                return_value=runner.ActiveViewResult(
                    object(), object(), np.eye(4), False, 1, "active_capture_failed", True
                ),
            ):
                result = runner.maybe_clear_obstacles_before_active(object(), args, captured)

        self.assertIsNone(result)

    def test_center_view_only_replans_same_clear_target(self):
        mask = np.ones((4, 4), dtype=np.uint8)
        args = flow_args(mask)
        candidate = clear_candidate(mask)
        with tempfile.TemporaryDirectory() as out_dir:
            captured = self.make_capture(out_dir, mask)
            recaptured = self.make_capture(out_dir, mask)
            active = runner.ActiveViewResult(
                object(),
                object(),
                np.eye(4),
                False,
                1,
                "center_view_only",
                True,
                True,
            )
            with self.common_patches([[candidate], []]), mock.patch.object(
                runner,
                "plan_and_maybe_execute_grasp",
                side_effect=[
                    runner.GraspAttemptResult(runner.GraspAttemptStatus.PLANNING_FAILED),
                    runner.GraspAttemptResult(runner.GraspAttemptStatus.SUCCESS, True),
                ],
            ) as plan, mock.patch.object(
                runner, "run_optional_active_view", return_value=active
            ), mock.patch.object(
                runner, "capture_target_cloud_with_retries", return_value=recaptured
            ):
                result = runner.maybe_clear_obstacles_before_active(
                    object(), args, captured
                )

        self.assertIs(result, recaptured)
        self.assertEqual(plan.call_count, 2)
        self.assertEqual(
            plan.call_args_list[1].kwargs["plan_label"],
            "clear1_center_fallback",
        )

    def test_center_view_only_restore_failure_does_not_replan(self):
        mask = np.ones((4, 4), dtype=np.uint8)
        args = flow_args(mask)
        with tempfile.TemporaryDirectory() as out_dir:
            captured = self.make_capture(out_dir, mask)
            with self.common_patches([[clear_candidate(mask)]]), mock.patch.object(
                runner,
                "plan_and_maybe_execute_grasp",
                return_value=runner.GraspAttemptResult(
                    runner.GraspAttemptStatus.PLANNING_FAILED
                ),
            ) as plan, mock.patch.object(
                runner,
                "run_optional_active_view",
                return_value=runner.ActiveViewResult(
                    object(),
                    object(),
                    np.eye(4),
                    False,
                    1,
                    "center_view_only",
                    False,
                    True,
                ),
            ):
                result = runner.maybe_clear_obstacles_before_active(
                    object(), args, captured
                )

        self.assertIsNone(result)
        self.assertEqual(plan.call_count, 1)

    def test_center_view_replan_failure_still_forbids_original_target(self):
        mask = np.ones((4, 4), dtype=np.uint8)
        args = flow_args(mask)
        with tempfile.TemporaryDirectory() as out_dir:
            captured = self.make_capture(out_dir, mask)
            with self.common_patches([[clear_candidate(mask)]]), mock.patch.object(
                runner,
                "plan_and_maybe_execute_grasp",
                side_effect=[
                    runner.GraspAttemptResult(runner.GraspAttemptStatus.PLANNING_FAILED),
                    runner.GraspAttemptResult(runner.GraspAttemptStatus.PLANNING_FAILED),
                ],
            ) as plan, mock.patch.object(
                runner,
                "run_optional_active_view",
                return_value=runner.ActiveViewResult(
                    object(),
                    object(),
                    np.eye(4),
                    False,
                    1,
                    "center_view_only",
                    True,
                    True,
                ),
            ):
                result = runner.maybe_clear_obstacles_before_active(
                    object(), args, captured
                )

        self.assertIsNone(result)
        self.assertEqual(plan.call_count, 2)

    def test_empty_or_failed_execution_does_not_trigger_multiview_or_target_fallback(self):
        mask = np.ones((4, 4), dtype=np.uint8)
        args = flow_args(mask)
        with tempfile.TemporaryDirectory() as out_dir:
            captured = self.make_capture(out_dir, mask)
            with self.common_patches([[clear_candidate(mask)]]), mock.patch.object(
                runner,
                "plan_and_maybe_execute_grasp",
                return_value=runner.GraspAttemptResult(
                    runner.GraspAttemptStatus.EXECUTION_FAILED,
                    True,
                    "empty_grasp",
                ),
            ), mock.patch.object(runner, "run_optional_active_view") as active_view:
                result = runner.maybe_clear_obstacles_before_active(object(), args, captured)

        self.assertIsNone(result)
        active_view.assert_not_called()

    def test_verified_empty_grasp_recaptures_and_reassesses_original_target(self):
        mask = np.ones((4, 4), dtype=np.uint8)
        args = flow_args(mask)
        candidate = clear_candidate(mask)
        with tempfile.TemporaryDirectory() as out_dir:
            captured = self.make_capture(out_dir, mask)
            recaptured = self.make_capture(out_dir, mask)
            empty_result = runner.GraspAttemptResult(
                runner.GraspAttemptStatus.EXECUTION_FAILED,
                True,
                "slow close ended without contact",
                empty_grasp_recovered_to_initial=True,
            )
            with self.common_patches([[candidate], []]), mock.patch.object(
                runner,
                "plan_and_maybe_execute_grasp",
                return_value=empty_result,
            ) as plan, mock.patch.object(
                runner,
                "recapture_original_target_after_clear",
                return_value=recaptured,
            ) as recapture, mock.patch.object(
                runner, "run_optional_active_view"
            ) as active_view:
                result = runner.maybe_clear_obstacles_before_active(
                    object(), args, captured
                )

        self.assertIs(result, recaptured)
        self.assertEqual(plan.call_count, 1)
        recapture.assert_called_once()
        np.testing.assert_array_equal(recapture.call_args.args[1], mask)
        self.assertEqual(recapture.call_args.args[2]["label"], "carrot")
        active_view.assert_not_called()

    def test_verified_empty_grasp_retry_is_bounded(self):
        mask = np.ones((4, 4), dtype=np.uint8)
        args = flow_args(mask)
        candidate = clear_candidate(mask)
        with tempfile.TemporaryDirectory() as out_dir:
            captured = self.make_capture(out_dir, mask)
            recaptured = self.make_capture(out_dir, mask)
            empty_result = runner.GraspAttemptResult(
                runner.GraspAttemptStatus.EXECUTION_FAILED,
                True,
                "slow close ended without contact",
                empty_grasp_recovered_to_initial=True,
            )
            with self.common_patches([[candidate], [candidate]]), mock.patch.object(
                runner,
                "plan_and_maybe_execute_grasp",
                return_value=empty_result,
            ) as plan, mock.patch.object(
                runner,
                "recapture_original_target_after_clear",
                return_value=recaptured,
            ) as recapture:
                result = runner.maybe_clear_obstacles_before_active(
                    object(), args, captured
                )

        self.assertIsNone(result)
        self.assertEqual(plan.call_count, 2)
        self.assertEqual(
            plan.call_args_list[1].kwargs["plan_label"],
            "clear1_retry1_single",
        )
        recapture.assert_called_once()

    def test_last_clear_round_is_reassessed_and_still_blocked_scene_aborts(self):
        mask = np.ones((4, 4), dtype=np.uint8)
        args = flow_args(mask)
        candidate = clear_candidate(mask)
        with tempfile.TemporaryDirectory() as out_dir:
            captured = self.make_capture(out_dir, mask)
            recaptured = self.make_capture(out_dir, mask)
            with self.common_patches([[candidate], [candidate]]), mock.patch.object(
                runner,
                "plan_and_maybe_execute_grasp",
                return_value=runner.GraspAttemptResult(runner.GraspAttemptStatus.SUCCESS, True),
            ) as plan, mock.patch.object(
                runner, "capture_target_cloud_with_retries", return_value=recaptured
            ):
                result = runner.maybe_clear_obstacles_before_active(object(), args, captured)

        self.assertIsNone(result)
        self.assertEqual(plan.call_count, 1)

    def test_high_amodal_occlusion_without_clear_object_forces_multiview(self):
        mask = np.ones((4, 4), dtype=np.uint8)
        args = flow_args(mask)

        def no_occluder(*call_args):
            call_args[4]._last_clear_recommended_action = (
                "active_view_no_actionable_occluder"
            )
            return []

        with tempfile.TemporaryDirectory() as out_dir:
            captured = self.make_capture(out_dir, mask)
            with self.common_patches([[]]):
                runner.detect_clear_obstacle_candidates.side_effect = no_occluder
                result = runner.maybe_clear_obstacles_before_active(
                    object(), args, captured
                )

        self.assertIs(result, captured)
        self.assertTrue(args._force_active_multiview)

    def test_forced_multiview_bypasses_normal_proximity_gate(self):
        args = SimpleNamespace(
            active_view=True,
            active_center_first=False,
            active_path_guard=False,
            _force_active_multiview=True,
        )
        with mock.patch.object(
            runner, "_should_use_active_multiview", return_value=False
        ) as gate, mock.patch.object(
            runner, "_build_active_view_candidates", return_value=[]
        ) as build:
            result = runner.run_optional_active_view(
                object(), args, object(), object(), np.eye(4), {}, "/tmp"
            )

        gate.assert_not_called()
        build.assert_called_once()
        self.assertEqual(result.reason, "no_valid_active_candidate")

    def test_strict_yolo_reselection_tracks_nearest_same_class_3d_instance(self):
        args = SimpleNamespace(
            active_target_reselect="yolo",
            active_strict_target_identity=True,
            active_project_radius=0.08,
            active_strict_identity_margin=0.015,
            active_yolo_fallback="click",
        )
        bgr = np.zeros((4, 4, 3), dtype=np.uint8)
        xyz = np.full((4, 4, 3), np.nan, dtype=float)
        near_mask = np.zeros((4, 4), dtype=np.uint8)
        near_mask[:2, :2] = 1
        far_mask = np.zeros((4, 4), dtype=np.uint8)
        far_mask[2:, 2:] = 1
        xyz[near_mask > 0] = [0.10, 0.00, 0.50]
        xyz[far_mask > 0] = [0.25, 0.00, 0.50]
        near = {"label": "radish", "mask": near_mask, "id": "tracked"}
        far = {"label": "radish", "mask": far_mask, "id": "other"}
        fake_cloud = SimpleNamespace(points=np.zeros((1, 3), dtype=float))

        with mock.patch.object(
            runner,
            "capture_rgbd_xyz_frame",
            return_value=(bgr, xyz, object(), "/tmp"),
        ), mock.patch.object(
            runner, "infer_yolo_instances", return_value=([far, near], None)
        ), mock.patch.object(
            runner,
            "mask_for_instance",
            side_effect=lambda inst, *_: inst["mask"],
        ), mock.patch.object(
            runner, "save_selected_target_visuals"
        ), mock.patch.object(
            runner, "refine_xyz_inside_visible_mask", return_value=xyz
        ), mock.patch.object(
            runner, "cloud_from_mask", return_value=fake_cloud
        ), mock.patch.object(
            runner, "scene_cloud_from_xyz", return_value=fake_cloud
        ), mock.patch.object(
            runner, "obstacle_cloud_from_non_target_mask", return_value=fake_cloud
        ):
            result = runner.capture_active_projected_target_cloud(
                args,
                {"class_label": "radish"},
                np.eye(4),
                np.array([0.10, 0.00, 0.50]),
                "strict_side",
            )

        self.assertEqual(result[5]["id"], "tracked")

    def test_active_view_reselection_uses_dedicated_lower_confidence(self):
        args = SimpleNamespace(
            target_prompt=True,
            target_index=3,
            select_target=True,
            yolo_conf=0.25,
            active_yolo_conf=0.05,
        )

        active_args = runner._active_view_second_capture_args(
            args,
            {"label": "cup"},
        )

        self.assertEqual(active_args.target_class, "cup")
        self.assertAlmostEqual(active_args.yolo_conf, 0.05)
        self.assertFalse(active_args.target_prompt)
        self.assertFalse(active_args.select_target)

    def test_active_yolo_abort_never_uses_projected_radius_fallback(self):
        args = SimpleNamespace(
            active_target_reselect="yolo",
            active_strict_target_identity=False,
            active_yolo_fallback="abort",
            yolo_conf=0.25,
            active_yolo_conf=0.05,
        )
        bgr = np.zeros((4, 4, 3), dtype=np.uint8)
        xyz = np.zeros((4, 4, 3), dtype=float)
        xyz[:, :, 2] = 0.50

        with mock.patch.object(
            runner,
            "capture_rgbd_xyz_frame",
            return_value=(bgr, xyz, object(), "/tmp"),
        ), mock.patch.object(
            runner, "infer_yolo_instances", return_value=([], None)
        ), mock.patch.object(
            runner, "mask_from_projected_target_center"
        ) as projected:
            with self.assertRaisesRegex(
                RuntimeError, "refusing to fuse an unverified"
            ):
                runner.capture_active_projected_target_cloud(
                    args,
                    {"class_label": "cup"},
                    np.eye(4),
                    np.array([0.10, 0.00, 0.50]),
                    "active_center",
                )

        projected.assert_not_called()

    def test_post_clear_same_view_reidentification_uses_reference_mask(self):
        args = SimpleNamespace(
            min_depth=0.05,
            max_depth=1.50,
            min_target_points=1,
            box_depth_band=0.045,
            post_clear_reidentify_min_coverage=0.30,
            post_clear_reidentify_min_iou=0.10,
            post_clear_reidentify_ambiguity_margin=0.15,
        )
        xyz = np.zeros((12, 12, 3), dtype=float)
        xyz[:, :, 2] = 0.40
        reference = np.zeros((12, 12), dtype=np.uint8)
        reference[7:11, 2:6] = 1
        tracked_mask = np.zeros_like(reference)
        tracked_mask[6:12, 1:7] = 1
        other_mask = np.zeros_like(reference)
        other_mask[1:5, 8:12] = 1
        tracked = {
            "label": "carrot",
            "score": 0.90,
            "mask": tracked_mask,
            "id": "tracked",
        }
        other = {
            "label": "carrot",
            "score": 0.99,
            "mask": other_mask,
            "id": "other",
        }

        mask, inst, detail = runner.reidentify_instance_from_reference_mask(
            [other, tracked],
            xyz,
            reference,
            "carrot",
            args,
        )

        self.assertEqual(inst["id"], "tracked")
        np.testing.assert_array_equal(mask, tracked_mask > 0)
        self.assertIn("reference_coverage=1.000", detail)

    def test_post_clear_reidentification_fails_closed_when_ambiguous(self):
        args = SimpleNamespace(
            min_depth=0.05,
            max_depth=1.50,
            min_target_points=1,
            box_depth_band=0.045,
            post_clear_reidentify_min_coverage=0.30,
            post_clear_reidentify_min_iou=0.10,
            post_clear_reidentify_ambiguity_margin=0.15,
        )
        xyz = np.zeros((8, 8, 3), dtype=float)
        xyz[:, :, 2] = 0.40
        reference = np.zeros((8, 8), dtype=np.uint8)
        reference[2:6, 2:6] = 1
        mask_a = np.zeros_like(reference)
        mask_a[2:6, 1:5] = 1
        mask_b = np.zeros_like(reference)
        mask_b[2:6, 3:7] = 1

        mask, inst, detail = runner.reidentify_instance_from_reference_mask(
            [
                {"label": "carrot", "score": 0.90, "mask": mask_a},
                {"label": "carrot", "score": 0.89, "mask": mask_b},
            ],
            xyz,
            reference,
            "carrot",
            args,
        )

        self.assertIsNone(mask)
        self.assertIsNone(inst)
        self.assertIn("ambiguous", detail)

    def test_post_clear_recapture_temporarily_enables_reference_tracking(self):
        mask = np.ones((4, 4), dtype=np.uint8)
        args = flow_args(mask)
        args.target_prompt = False
        args.target_class = None
        args.select_target = True
        recaptured = object()

        def capture(call_args):
            self.assertEqual(call_args.target_class, "carrot")
            self.assertFalse(call_args.select_target)
            np.testing.assert_array_equal(
                call_args._post_clear_reference_mask,
                mask,
            )
            return recaptured

        with mock.patch.object(
            runner,
            "capture_target_cloud_with_retries",
            side_effect=capture,
        ):
            result = runner.recapture_original_target_after_clear(
                args,
                mask,
                {"label": "carrot"},
            )

        self.assertIs(result, recaptured)
        self.assertIsNone(args.target_class)
        self.assertTrue(args.select_target)
        self.assertFalse(hasattr(args, "_post_clear_reference_mask"))

    def test_reference_reidentification_failure_never_falls_back_to_click(self):
        reference = np.ones((4, 4), dtype=np.uint8)
        args = SimpleNamespace(
            manual_click_only=False,
            target_index=None,
            target_class="carrot",
            select_target=False,
            _post_clear_reference_mask=reference,
            _post_clear_reference_label="carrot",
        )
        bgr = np.zeros((4, 4, 3), dtype=np.uint8)
        xyz = np.zeros((4, 4, 3), dtype=float)
        instance = {"label": "carrot", "mask": reference}

        with tempfile.TemporaryDirectory() as out_dir, mock.patch.object(
            runner,
            "infer_yolo_instances",
            return_value=([instance], None),
        ), mock.patch.object(
            runner,
            "reidentify_instance_from_reference_mask",
            return_value=(None, None, "same-class match is ambiguous"),
        ), mock.patch.object(runner, "click_point") as click:
            with self.assertRaises(runner.TargetIdentityNotVerified):
                runner.select_target_mask(bgr, xyz, args, out_dir)

        click.assert_not_called()

    def test_strict_identity_deduplicates_only_same_mask_and_same_3d_center(self):
        args = SimpleNamespace(
            active_identity_duplicate_mask_iou=0.95,
            active_identity_duplicate_center_dist=0.005,
            active_identity_duplicate_nested_min_containment=0.95,
            active_identity_duplicate_nested_min_area_ratio=0.50,
            active_identity_duplicate_nested_max_area_ratio=0.90,
            active_identity_duplicate_nested_center_dist=0.015,
        )
        mask_a = np.ones((8, 8), dtype=np.uint8)
        mask_b = mask_a.copy()
        low = {"score": 0.62, "id": "duplicate_low"}
        high = {"score": 0.81, "id": "duplicate_high"}
        duplicate_rows = [
            (0.0087, low, mask_a, np.array([0.0, 0.0, 0.20])),
            (0.0088, high, mask_b, np.array([0.0001, 0.0, 0.20])),
        ]

        kept = runner._deduplicate_strict_identity_candidates(
            duplicate_rows, args, "side"
        )

        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0][1]["id"], "duplicate_high")

        distinct_mask = np.zeros((8, 8), dtype=np.uint8)
        distinct_mask[:2, :2] = 1
        true_neighbor_rows = [
            duplicate_rows[1],
            (0.0090, {"score": 0.75}, distinct_mask, np.array([0.002, 0.0, 0.20])),
        ]
        self.assertEqual(
            len(
                runner._deduplicate_strict_identity_candidates(
                    true_neighbor_rows, args, "side"
                )
            ),
            2,
        )

        same_mask_different_depth = [
            duplicate_rows[1],
            (0.0090, {"score": 0.75}, mask_a, np.array([0.010, 0.0, 0.20])),
        ]
        self.assertEqual(
            len(
                runner._deduplicate_strict_identity_candidates(
                    same_mask_different_depth, args, "side"
                )
            ),
            2,
        )

    def test_strict_identity_collapses_nested_whole_object_and_body_masks(self):
        args = SimpleNamespace(
            active_identity_duplicate_mask_iou=0.95,
            active_identity_duplicate_center_dist=0.005,
            active_identity_duplicate_nested_min_containment=0.95,
            active_identity_duplicate_nested_min_area_ratio=0.50,
            active_identity_duplicate_nested_max_area_ratio=0.90,
            active_identity_duplicate_nested_center_dist=0.015,
        )
        whole_mask = np.ones((20, 20), dtype=np.uint8)
        body_mask = np.zeros_like(whole_mask)
        body_mask[2:20, 2:20] = 1
        rows = [
            (
                0.0152,
                {"score": 0.68, "id": "whole"},
                whole_mask,
                np.array([0.0091, -0.0001, 0.3010]),
            ),
            (
                0.0158,
                {"score": 0.47, "id": "body"},
                body_mask,
                np.array([0.0029, -0.0019, 0.2980]),
            ),
        ]

        kept = runner._deduplicate_strict_identity_candidates(
            rows, args, "active_center"
        )

        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0][1]["id"], "whole")

        far_depth_rows = [
            rows[0],
            (
                rows[1][0],
                rows[1][1],
                rows[1][2],
                np.array([0.0300, -0.0019, 0.2980]),
            ),
        ]
        self.assertEqual(
            len(
                runner._deduplicate_strict_identity_candidates(
                    far_depth_rows, args, "active_center"
                )
            ),
            2,
        )

    def test_unguarded_active_motion_cannot_execute(self):
        args = SimpleNamespace(
            active_view=True,
            active_center_first=False,
            active_path_guard=False,
            active_strict_target_identity=True,
            _force_active_multiview=True,
            execute=True,
            yes=True,
            yes_active_view=False,
            active_nbv_debug=False,
        )
        candidate = {
            "name": "side",
            "reason": "ok",
            "score": 1.0,
            "command_type": "tcp",
            "tcp": np.zeros(6),
        }
        with mock.patch.object(
            runner, "_build_active_view_candidates", return_value=[candidate]
        ), mock.patch.object(
            runner, "_target_center_base_from_cloud", return_value=np.zeros(3)
        ), mock.patch.object(
            runner, "move_tcp_pose"
        ), mock.patch.object(
            runner, "T_base_camera_from_flange", return_value=np.eye(4)
        ), mock.patch.object(
            runner,
            "capture_active_projected_target_cloud",
            side_effect=RuntimeError("strict identity not found"),
        ) as strict_capture, mock.patch.object(
            runner, "capture_target_cloud"
        ) as generic_capture:
            result = runner.run_optional_active_view(
                object(), args, object(), object(), np.eye(4), {"class_label": "radish"}, "/tmp"
            )

        strict_capture.assert_not_called()
        generic_capture.assert_not_called()
        self.assertEqual(result.reason, "active_guard_required")
        self.assertFalse(result.motion_sent)


if __name__ == "__main__":
    unittest.main()
