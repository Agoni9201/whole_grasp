import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

import real_nero_single_grasp_runner as runner
from real_nero_grasp.grasp_workflow import (
    _after_release_execution_stages,
    _before_close_execution_stages,
    _empty_recovery_execution_stages,
)


class FakeGrasp:
    def __init__(self, width):
        self.width = float(width)
        self.rotation_matrix = np.eye(3, dtype=float)
        self.translation = np.zeros(3, dtype=float)


class EmptyRecoveryExecutionPolicyTest(unittest.TestCase):
    def test_large_successful_return_edges_are_subdivided_from_release(self):
        retreat = SimpleNamespace(joints=np.ones(7, dtype=float))
        home = SimpleNamespace(joints=np.full(7, 2.0, dtype=float))
        plan = SimpleNamespace(after_release=(retreat, home))
        starts = []

        def subdivide(start, stage, _args):
            starts.append(np.asarray(start, dtype=float).copy())
            return (stage, stage)

        with mock.patch(
            "real_nero_grasp.active_motion._subdivide_active_direct_stage",
            side_effect=subdivide,
        ):
            stages = _after_release_execution_stages(
                plan,
                SimpleNamespace(
                    released_return_execution_waypoint_deg=20.0
                ),
                np.zeros(7, dtype=float),
            )

        self.assertEqual(stages, (retreat, retreat, home, home))
        np.testing.assert_allclose(starts[0], np.zeros(7))
        np.testing.assert_allclose(starts[1], np.ones(7))

    def test_large_before_close_edges_are_subdivided_from_plan_start(self):
        first = SimpleNamespace(joints=np.ones(7, dtype=float))
        second = SimpleNamespace(joints=np.full(7, 2.0, dtype=float))
        plan = SimpleNamespace(
            start_joints=np.zeros(7, dtype=float),
            before_close=(first, second),
        )
        starts = []

        def subdivide(start, stage, _args):
            starts.append(np.asarray(start, dtype=float).copy())
            return (stage, stage)

        with mock.patch(
            "real_nero_grasp.active_motion._subdivide_active_direct_stage",
            side_effect=subdivide,
        ):
            stages = _before_close_execution_stages(
                plan,
                SimpleNamespace(grasp_execution_waypoint_deg=20.0),
            )

        self.assertEqual(stages, (first, first, second, second))
        np.testing.assert_allclose(starts[0], np.zeros(7))
        np.testing.assert_allclose(starts[1], np.ones(7))

    def test_large_empty_recovery_edges_are_subdivided_in_order(self):
        first = SimpleNamespace(joints=np.ones(7, dtype=float))
        second = SimpleNamespace(joints=np.full(7, 2.0, dtype=float))
        plan = SimpleNamespace(
            before_close=(SimpleNamespace(joints=np.zeros(7, dtype=float)),),
            empty_recovery=(first, second),
        )
        starts = []

        def subdivide(start, stage, _args):
            starts.append(np.asarray(start, dtype=float).copy())
            return (stage, stage)

        with mock.patch(
            "real_nero_grasp.active_motion._subdivide_active_direct_stage",
            side_effect=subdivide,
        ):
            stages = _empty_recovery_execution_stages(
                plan,
                SimpleNamespace(empty_recovery_execution_waypoint_deg=15.0),
            )

        # The first backout edge is subdivided too, before reopening the jaws.
        self.assertEqual(stages, (first, first, second, second))
        self.assertEqual(len(starts), 2)
        np.testing.assert_allclose(starts[0], np.zeros(7))
        np.testing.assert_allclose(starts[1], np.ones(7))

    def test_empty_recovery_subdivision_can_be_disabled(self):
        stage = SimpleNamespace(joints=np.ones(7, dtype=float))
        plan = SimpleNamespace(
            before_close=(SimpleNamespace(joints=np.zeros(7, dtype=float)),),
            empty_recovery=(stage,),
        )
        stages = _empty_recovery_execution_stages(
            plan,
            SimpleNamespace(empty_recovery_execution_waypoint_deg=0.0),
        )
        self.assertEqual(stages, (stage,))


class FakeGripper:
    def __init__(self):
        self.commands = []

    def move_gripper_m(self, value, force):
        self.commands.append((float(value), float(force)))
        return None


def make_arm(statuses):
    arm = object.__new__(runner.RealNeroGraspArm)
    arm.gripper_enabled = True
    arm.gripper_max_width = 0.085
    arm.gripper_max_force = 0.30
    arm.gripper = FakeGripper()
    status_iter = iter(statuses)
    arm.print_gripper_status = lambda: next(status_iter, None)
    return arm


class GraspWidthContactPolicyTest(unittest.TestCase):
    def test_default_force_profile_uses_documented_minimum(self):
        with mock.patch(
            "sys.argv", ["runner", "--calib-json", "/tmp/calibration.json"]
        ):
            args = runner.parse_args()
        self.assertEqual(args.gripper_force, 0.50)
        self.assertEqual(args.gripper_hardware_max_force, 0.50)
        self.assertEqual(args.gripper_stop_force, 0.40)
        self.assertEqual(args.empty_grasp_width_threshold, 0.001)
        self.assertEqual(args.gripper_closed_width_tol, 0.001)
        self.assertEqual(args.gripper_close_completion_timeout, 2.0)
        self.assertEqual(args.graspnet_width_inflation, 1.2)
        self.assertEqual(args.graspnet_decode_max_width, 0.110)
        self.assertEqual(args.manual_topk, 24)
        self.assertEqual(args.max_grasp_width, 0.110)
        self.assertTrue(args.graspnet_main_component_filter)
        self.assertEqual(args.graspnet_component_eps, 0.004)
        self.assertEqual(args.graspnet_component_min_points, 8)
        self.assertEqual(args.graspnet_component_min_dominance, 0.85)
        self.assertEqual(args.grasp_physical_aperture_margin, 0.0005)
        self.assertEqual(args.grasp_physical_recenter_max, 0.020)
        self.assertEqual(args.clear_empty_grasp_retries, 1)
        self.assertEqual(
            args.grasp_physical_max_finger_penetration,
            0.001,
        )

    def test_motion_confirmation_enter_executes_and_no_cancels(self):
        with mock.patch("builtins.input", return_value="") as entered:
            self.assertTrue(runner.confirm_motion("test motion"))
        entered.assert_called_once()

        with mock.patch("builtins.input", return_value="  NO  ") as cancelled:
            self.assertFalse(runner.confirm_motion("test motion"))
        cancelled.assert_called_once()

    def test_motion_confirmation_reprompts_on_old_command_words(self):
        for old_word in ("INIT", "VIEW", "CLEAR", "GRASP"):
            with self.subTest(old_word=old_word), mock.patch(
                "builtins.input", side_effect=[old_word, ""]
            ) as prompt:
                self.assertTrue(runner.confirm_motion("test motion"))
                self.assertEqual(prompt.call_count, 2)

    def test_real_gripper_method_hard_caps_open_command(self):
        arm = object.__new__(runner.RealNeroGraspArm)
        arm.gripper_enabled = True
        arm.gripper_max_width = 0.085
        arm.gripper_max_force = 0.30
        arm.gripper = FakeGripper()
        arm.print_gripper_status = lambda: None
        with mock.patch.object(runner.time, "sleep"):
            arm.open_gripper(0.090, 0.30, 0.0)
        self.assertEqual(arm.gripper.commands, [(0.085, 0.30)])
        with mock.patch.object(runner.time, "sleep"):
            arm.open_gripper(0.085, 0.50, 0.0)
        self.assertEqual(arm.gripper.commands[-1], (0.085, 0.30))
        with self.assertRaises(ValueError):
            arm.open_gripper(float("nan"), 0.30, 0.0)

    def test_100mm_is_candidate_tolerance_but_execution_stays_at_85mm(self):
        args = SimpleNamespace(
            max_grasp_width=0.100,
            allow_oversize_grasp=False,
            gripper_open_width=0.085,
            gripper_hardware_max_width=0.085,
        )
        rows = [
            {"grasp": FakeGrasp(0.099)},
            {"grasp": FakeGrasp(0.101)},
        ]

        kept = runner.filter_grasp_candidates_by_width(rows, args)

        self.assertEqual(kept, [rows[0]])
        self.assertAlmostEqual(
            runner.execution_gripper_open_width(rows[0]["grasp"], args),
            0.085,
        )

    def test_inflated_width_is_retained_only_after_actual_aperture_fit(self):
        args = SimpleNamespace(
            max_grasp_width=0.090,
            allow_oversize_grasp=False,
            gripper_open_width=0.085,
            gripper_hardware_max_width=0.085,
            graspnet_width_inflation=1.2,
            graspnet_decode_max_width=0.100,
            grasp_physical_aperture_margin=0.002,
            grasp_physical_aperture_quantile=0.01,
            grasp_physical_aperture_min_points=50,
            grasp_physical_recenter_max=0.010,
            topdown_grasp=False,
            use_current_rpy=False,
            fixed_rpy=None,
            graspnet_approach_axis="x",
            graspnet_width_axis="y",
            tool_approach_axis="x",
            tcp_gripper_width_axis="y",
            approach_tilt_deg=5.0,
            use_graspnet_depth_advance=False,
            grasp_advance=0.0,
            min_grasp_z=None,
            pregrasp_offset=0.10,
            lift_height=0.10,
        )

        def grasp(width):
            return SimpleNamespace(
                width=float(width),
                height=0.020,
                depth=0.030,
                rotation_matrix=np.eye(3, dtype=float),
                translation=np.zeros(3, dtype=float),
            )

        fitting_points = np.column_stack(
            [
                np.zeros(200),
                np.linspace(-0.038, 0.038, 200),
                np.zeros(200),
            ]
        )
        rows = runner.annotate_grasp_candidate_physical_aperture(
            [
                {"idx": 0, "grasp": grasp(0.096)},
                {"idx": 1, "grasp": grasp(0.100)},
            ],
            SimpleNamespace(points=fitting_points),
            np.eye(4),
            np.zeros(6),
            args,
        )
        kept = runner.filter_grasp_candidates_by_width(rows, args)

        self.assertTrue(rows[0]["physical_aperture_ok"])
        self.assertAlmostEqual(rows[0]["graspnet_nominal_width"], 0.096 / 1.2)
        self.assertFalse(rows[1]["physical_aperture_ok"])
        self.assertTrue(rows[1]["graspnet_width_saturated"])
        self.assertEqual([row["idx"] for row in kept], [0])
        self.assertAlmostEqual(
            runner.execution_gripper_open_width(kept[0]["grasp"], args),
            0.085,
        )

        # The fit check must follow the final commanded TCP, not the raw
        # GraspNet center.  Advancing beyond the target removes these same
        # points from the real finger slab and must invalidate the candidate.
        args.grasp_advance = 0.10
        advanced = runner.annotate_grasp_candidate_physical_aperture(
            [{"idx": 0, "grasp": grasp(0.096)}],
            SimpleNamespace(points=fitting_points),
            np.eye(4),
            np.zeros(6),
            args,
        )
        self.assertFalse(advanced[0]["physical_aperture_ok"])
        self.assertIn(
            "actual finger slab",
            advanced[0]["physical_aperture_reason"],
        )

    def test_actual_aperture_rejects_miscentered_target_slab(self):
        args = SimpleNamespace(
            max_grasp_width=0.090,
            allow_oversize_grasp=False,
            gripper_open_width=0.085,
            gripper_hardware_max_width=0.085,
            graspnet_width_inflation=1.2,
            graspnet_decode_max_width=0.100,
            grasp_physical_aperture_margin=0.002,
            grasp_physical_aperture_quantile=0.01,
            grasp_physical_aperture_min_points=50,
            grasp_physical_recenter_max=0.010,
            topdown_grasp=False,
            use_current_rpy=False,
            fixed_rpy=None,
            graspnet_approach_axis="x",
            graspnet_width_axis="y",
            tool_approach_axis="x",
            tcp_gripper_width_axis="y",
            approach_tilt_deg=5.0,
            use_graspnet_depth_advance=False,
            grasp_advance=0.0,
            min_grasp_z=None,
            pregrasp_offset=0.10,
            lift_height=0.10,
        )
        candidate = SimpleNamespace(
            width=0.095,
            height=0.020,
            depth=0.030,
            rotation_matrix=np.eye(3, dtype=float),
            translation=np.zeros(3, dtype=float),
        )
        points = np.column_stack(
            [
                np.zeros(200),
                np.linspace(-0.050, 0.035, 200),
                np.zeros(200),
            ]
        )
        rows = runner.annotate_grasp_candidate_physical_aperture(
            [{"idx": 0, "grasp": candidate}],
            SimpleNamespace(points=points),
            np.eye(4),
            np.zeros(6),
            args,
        )

        self.assertFalse(rows[0]["physical_aperture_ok"])
        self.assertIn("target span", rows[0]["physical_aperture_reason"])
        self.assertEqual(
            runner.filter_grasp_candidates_by_width(rows, args), []
        )

    def test_actual_aperture_recenters_small_width_axis_offset(self):
        args = SimpleNamespace(
            max_grasp_width=0.090,
            allow_oversize_grasp=False,
            gripper_open_width=0.085,
            gripper_hardware_max_width=0.085,
            graspnet_width_inflation=1.2,
            graspnet_decode_max_width=0.100,
            grasp_physical_aperture_margin=0.002,
            grasp_physical_aperture_quantile=0.01,
            grasp_physical_aperture_min_points=50,
            grasp_physical_recenter_max=0.010,
            motion_path_open_finger_radius=0.014,
            motion_path_open_finger_back=0.065,
            motion_path_open_finger_front=0.020,
            topdown_grasp=False,
            use_current_rpy=False,
            fixed_rpy=None,
            graspnet_approach_axis="x",
            graspnet_width_axis="y",
            tool_approach_axis="x",
            tcp_gripper_width_axis="y",
            approach_tilt_deg=5.0,
            use_graspnet_depth_advance=False,
            grasp_advance=0.0,
            min_grasp_z=None,
            pregrasp_offset=0.10,
            lift_height=0.10,
        )
        candidate = SimpleNamespace(
            width=0.095,
            height=0.020,
            depth=0.030,
            rotation_matrix=np.eye(3, dtype=float),
            translation=np.zeros(3, dtype=float),
        )
        points = np.column_stack(
            [
                np.zeros(240),
                np.linspace(-0.043, 0.032, 240),
                np.zeros(240),
            ]
        )

        rows = runner.annotate_grasp_candidate_physical_aperture(
            [{"idx": 0, "grasp": candidate}],
            SimpleNamespace(points=points),
            np.eye(4),
            np.zeros(6),
            args,
        )

        self.assertTrue(rows[0]["physical_aperture_ok"])
        shift = runner.grasp_candidate_width_shift(rows[0], 1.0)
        self.assertLess(shift, -0.004)
        self.assertGreater(shift, -0.007)
        variant = next(
            item
            for item in rows[0]["physical_aperture_variants"]
            if item["width_sign"] == 1.0
        )
        self.assertAlmostEqual(
            variant["corrected_lower"],
            -variant["corrected_upper"],
            places=6,
        )

    def test_actual_aperture_checks_finger_capsule_endcaps(self):
        args = SimpleNamespace(
            max_grasp_width=0.090,
            allow_oversize_grasp=False,
            gripper_open_width=0.085,
            gripper_hardware_max_width=0.085,
            graspnet_width_inflation=1.2,
            graspnet_decode_max_width=0.100,
            grasp_physical_aperture_margin=0.002,
            grasp_physical_aperture_quantile=0.01,
            grasp_physical_aperture_min_points=50,
            grasp_physical_recenter_max=0.010,
            motion_path_open_finger_radius=0.014,
            motion_path_open_finger_back=0.065,
            motion_path_open_finger_front=0.020,
            topdown_grasp=False,
            use_current_rpy=False,
            fixed_rpy=None,
            graspnet_approach_axis="x",
            graspnet_width_axis="y",
            tool_approach_axis="x",
            tcp_gripper_width_axis="y",
            approach_tilt_deg=5.0,
            use_graspnet_depth_advance=False,
            grasp_advance=0.0,
            min_grasp_z=None,
            pregrasp_offset=0.10,
            lift_height=0.10,
        )
        candidate = SimpleNamespace(
            width=0.095,
            height=0.020,
            depth=0.030,
            rotation_matrix=np.eye(3, dtype=float),
            translation=np.zeros(3, dtype=float),
        )
        inside = np.column_stack(
            [
                np.zeros(240),
                np.linspace(-0.030, 0.030, 240),
                np.zeros(240),
            ]
        )
        endcap_contact = np.tile(
            np.array([[0.025, 0.042, 0.0]], dtype=float),
            (30, 1),
        )

        rows = runner.annotate_grasp_candidate_physical_aperture(
            [{"idx": 0, "grasp": candidate}],
            SimpleNamespace(points=np.vstack([inside, endcap_contact])),
            np.eye(4),
            np.zeros(6),
            args,
        )

        self.assertFalse(rows[0]["physical_aperture_ok"])
        self.assertIn("finger capsule", rows[0]["physical_aperture_reason"])
        with self.assertRaises(runner.PlanNotExecutable):
            runner.grasp_candidate_width_shift(rows[0], 1.0)

    def test_actual_aperture_rejects_subpercent_deep_finger_penetration(self):
        args = SimpleNamespace(
            max_grasp_width=0.090,
            allow_oversize_grasp=False,
            gripper_open_width=0.085,
            gripper_hardware_max_width=0.085,
            graspnet_width_inflation=1.2,
            graspnet_decode_max_width=0.100,
            grasp_physical_aperture_margin=0.002,
            grasp_physical_aperture_quantile=0.01,
            grasp_physical_aperture_min_points=50,
            grasp_physical_recenter_max=0.010,
            grasp_physical_max_finger_penetration=0.001,
            motion_path_open_finger_radius=0.014,
            motion_path_open_finger_back=0.065,
            motion_path_open_finger_front=0.020,
            topdown_grasp=False,
            use_current_rpy=False,
            fixed_rpy=None,
            graspnet_approach_axis="x",
            graspnet_width_axis="y",
            tool_approach_axis="x",
            tcp_gripper_width_axis="y",
            approach_tilt_deg=5.0,
            use_graspnet_depth_advance=False,
            grasp_advance=0.0,
            min_grasp_z=None,
            pregrasp_offset=0.10,
            lift_height=0.10,
        )
        candidate = SimpleNamespace(
            width=0.095,
            height=0.020,
            depth=0.030,
            rotation_matrix=np.eye(3, dtype=float),
            translation=np.zeros(3, dtype=float),
        )
        inside = np.column_stack(
            [
                np.zeros(1000),
                np.linspace(-0.030, 0.030, 1000),
                np.zeros(1000),
            ]
        )
        # One point is only 0.1% of the cloud, so the robust q01 clearance
        # alone would ignore it.  It lies at the positive finger centerline
        # and must still trip the absolute penetration limit.
        deep_penetration = np.array([[0.0, 0.0565, 0.0]])

        rows = runner.annotate_grasp_candidate_physical_aperture(
            [{"idx": 0, "grasp": candidate}],
            SimpleNamespace(
                points=np.vstack([inside, deep_penetration])
            ),
            np.eye(4),
            np.zeros(6),
            args,
        )

        self.assertFalse(rows[0]["physical_aperture_ok"])
        self.assertIn(
            "penetrates an open finger capsule",
            rows[0]["physical_aperture_reason"],
        )

    def test_missing_or_nonfinite_candidate_width_is_rejected(self):
        args = SimpleNamespace(max_grasp_width=0.090, allow_oversize_grasp=False)
        missing = {
            "grasp": SimpleNamespace(
                rotation_matrix=np.eye(3, dtype=float),
                translation=np.zeros(3, dtype=float),
            )
        }
        nonfinite = {"grasp": FakeGrasp(float("nan"))}
        valid = {"grasp": FakeGrasp(0.085)}

        kept = runner.filter_grasp_candidates_by_width(
            [missing, nonfinite, valid],
            args,
        )

        self.assertEqual(kept, [valid])

    def test_slow_close_requires_confirmed_force_contact(self):
        statuses = [
            SimpleNamespace(value=0.085, force=0.05, mode="width"),
            SimpleNamespace(value=0.083, force=0.29, mode="width"),
            SimpleNamespace(value=0.083, force=0.30, mode="width"),
        ]
        arm = make_arm(statuses)

        result = arm.close_gripper_slow(
            0.085,
            0.0,
            0.30,
            0.002,
            0.0,
            settle_s=0.0,
            stop_force=0.28,
            contact_confirm_samples=2,
            contact_confirm_interval_s=0.0,
        )

        self.assertTrue(result.contact_detected)
        self.assertEqual(result.stop_reason, "force_contact")
        self.assertAlmostEqual(result.contact_width, 0.083)
        self.assertTrue(all(width <= 0.085 for width, _ in arm.gripper.commands))
        self.assertAlmostEqual(arm.gripper.commands[-1][0], 0.083)

    def test_full_close_without_contact_is_empty_grasp(self):
        no_contact = SimpleNamespace(value=0.0, force=0.02, mode="width")
        arm = make_arm([no_contact, no_contact, no_contact, no_contact])
        result = arm.close_gripper_slow(
            0.010,
            0.0,
            0.30,
            0.005,
            0.0,
            settle_s=0.0,
            stop_force=0.28,
            contact_confirm_samples=2,
            contact_confirm_interval_s=0.0,
        )
        args = SimpleNamespace(
            allow_empty_grasp_lift=False,
            require_gripper_contact=True,
            require_gripper_status=True,
            empty_grasp_width_threshold=0.005,
        )

        self.assertFalse(result.contact_detected)
        self.assertEqual(result.stop_reason, "close_limit_without_contact")
        with self.assertRaises(runner.GraspNotDetected) as raised:
            runner.assert_grasp_detected(result, args)
        self.assertTrue(raised.exception.retryable_empty)
        self.assertEqual(
            raised.exception.reason_code,
            "close_limit_without_contact",
        )

    def test_force_at_fully_closed_limit_is_not_object_contact(self):
        statuses = [
            SimpleNamespace(value=0.005, force=0.05, mode="width"),
            SimpleNamespace(value=0.000, force=0.30, mode="width"),
        ]
        arm = make_arm(statuses)
        result = arm.close_gripper_slow(
            0.005,
            0.0,
            0.30,
            0.005,
            0.0,
            settle_s=0.0,
            stop_force=0.28,
            empty_width_threshold=0.005,
        )
        self.assertFalse(result.contact_detected)
        self.assertEqual(result.stop_reason, "force_at_closed_limit")
        args = SimpleNamespace(
            allow_empty_grasp_lift=False,
            require_gripper_contact=True,
            require_gripper_status=True,
            empty_grasp_width_threshold=0.001,
        )
        with self.assertRaises(runner.GraspNotDetected) as raised:
            runner.assert_grasp_detected(result, args)
        self.assertTrue(raised.exception.retryable_empty)
        self.assertEqual(raised.exception.reason_code, "force_at_closed_limit")

    def test_invalid_or_missing_force_feedback_fails_closed(self):
        invalid = SimpleNamespace(value=0.08, force=float("nan"), mode="width")
        arm = make_arm([invalid])
        result = arm.close_gripper_slow(
            0.085,
            0.0,
            0.30,
            0.002,
            0.0,
            settle_s=0.0,
            stop_force=0.28,
        )
        self.assertFalse(result.contact_detected)
        self.assertEqual(result.stop_reason, "invalid_force_feedback")

        arm = make_arm([])
        arm.print_gripper_status = lambda: None
        result = arm.close_gripper_slow(
            0.085,
            0.0,
            0.30,
            0.002,
            0.0,
            settle_s=0.0,
            stop_force=0.28,
        )
        self.assertFalse(result.contact_detected)
        self.assertEqual(result.stop_reason, "missing_close_feedback")

        wrong_mode = SimpleNamespace(value=0.08, force=0.05, mode="force")
        arm = make_arm([wrong_mode])
        result = arm.close_gripper_slow(
            0.085,
            0.0,
            0.30,
            0.002,
            0.0,
            settle_s=0.0,
            stop_force=0.28,
        )
        self.assertFalse(result.contact_detected)
        self.assertEqual(result.stop_reason, "invalid_feedback_mode")

    def test_5mm_force_spike_resumes_closing_to_physical_empty_limit(self):
        statuses = [
            SimpleNamespace(value=0.010, force=0.05, mode="width"),
            SimpleNamespace(value=0.005, force=0.29, mode="width"),
            SimpleNamespace(value=0.005, force=0.10, mode="width"),
            SimpleNamespace(value=0.000, force=0.05, mode="width"),
            SimpleNamespace(value=0.000, force=0.05, mode="width"),
        ]
        arm = make_arm(statuses)
        result = arm.close_gripper_slow(
            0.010,
            0.0,
            0.30,
            0.005,
            0.0,
            settle_s=0.0,
            stop_force=0.28,
            contact_confirm_samples=2,
            contact_confirm_interval_s=0.0,
            # The old empty-width heuristic was 5 mm.  It must not make this
            # transient spike count as a physically closed gripper.
            empty_width_threshold=0.005,
        )
        self.assertFalse(result.contact_detected)
        self.assertEqual(result.stop_reason, "close_limit_without_contact")
        self.assertIn((0.0, 0.30), arm.gripper.commands)

    def test_delayed_force_contact_after_zero_command_is_confirmed(self):
        statuses = [
            SimpleNamespace(value=0.005, force=0.05, mode="width"),
            SimpleNamespace(value=0.003, force=0.08, mode="width"),
            SimpleNamespace(value=0.003, force=0.30, mode="width"),
            SimpleNamespace(value=0.003, force=0.29, mode="width"),
            SimpleNamespace(value=0.003, force=0.29, mode="width"),
        ]
        arm = make_arm(statuses)
        result = arm.close_gripper_slow(
            0.005,
            0.0,
            0.30,
            0.005,
            0.0,
            settle_s=0.1,
            stop_force=0.28,
            contact_confirm_samples=2,
            contact_confirm_interval_s=0.0,
            closed_width_tolerance=0.001,
        )
        self.assertTrue(result.contact_detected)
        self.assertEqual(result.stop_reason, "force_contact")
        self.assertEqual([round(x[0], 3) for x in arm.gripper.commands], [0.005, 0.0])

    def test_zero_command_without_contact_must_reach_physical_limit(self):
        statuses = [
            SimpleNamespace(value=0.005, force=0.05, mode="width"),
            SimpleNamespace(value=0.003, force=0.05, mode="width"),
            SimpleNamespace(value=0.003, force=0.05, mode="width"),
        ]
        arm = make_arm(statuses)
        result = arm.close_gripper_slow(
            0.005,
            0.0,
            0.30,
            0.005,
            0.0,
            settle_s=0.0,
            stop_force=0.28,
            closed_width_tolerance=0.001,
            close_completion_timeout_s=0.0,
        )
        self.assertFalse(result.contact_detected)
        self.assertEqual(result.stop_reason, "close_target_not_reached_timeout")

    def test_final_transient_force_keeps_waiting_until_physically_closed(self):
        statuses = [
            SimpleNamespace(value=0.005, force=0.05, mode="width"),
            SimpleNamespace(value=0.003, force=0.05, mode="width"),
            SimpleNamespace(value=0.003, force=0.30, mode="width"),
            SimpleNamespace(value=0.003, force=0.10, mode="width"),
            SimpleNamespace(value=0.0008, force=0.08, mode="width"),
        ]
        arm = make_arm(statuses)
        result = arm.close_gripper_slow(
            0.005,
            0.0,
            0.30,
            0.005,
            0.0,
            settle_s=0.0,
            stop_force=0.28,
            contact_confirm_samples=2,
            contact_confirm_interval_s=0.0,
            closed_width_tolerance=0.001,
            close_completion_timeout_s=1.0,
            close_completion_poll_s=0.001,
        )
        self.assertFalse(result.contact_detected)
        self.assertEqual(result.stop_reason, "close_limit_without_contact")
        self.assertAlmostEqual(result.contact_width, 0.0008)

    def test_final_low_force_width_is_polled_until_physically_closed(self):
        statuses = [
            SimpleNamespace(value=0.005, force=0.05, mode="width"),
            SimpleNamespace(value=0.003, force=0.05, mode="width"),
            SimpleNamespace(value=0.0025, force=0.05, mode="width"),
            SimpleNamespace(value=0.0015, force=0.05, mode="width"),
            SimpleNamespace(value=0.0008, force=0.05, mode="width"),
        ]
        arm = make_arm(statuses)
        result = arm.close_gripper_slow(
            0.005,
            0.0,
            0.30,
            0.005,
            0.0,
            settle_s=0.0,
            stop_force=0.28,
            closed_width_tolerance=0.001,
            close_completion_timeout_s=1.0,
            close_completion_poll_s=0.001,
        )
        self.assertFalse(result.contact_detected)
        self.assertEqual(result.stop_reason, "close_limit_without_contact")

    def test_transient_force_spike_can_continue_to_later_confirmed_contact(self):
        statuses = [
            SimpleNamespace(value=0.015, force=0.05, mode="width"),
            SimpleNamespace(value=0.010, force=0.29, mode="width"),
            SimpleNamespace(value=0.010, force=0.10, mode="width"),
            SimpleNamespace(value=0.005, force=0.30, mode="width"),
            SimpleNamespace(value=0.005, force=0.29, mode="width"),
        ]
        arm = make_arm(statuses)
        result = arm.close_gripper_slow(
            0.015,
            0.0,
            0.30,
            0.005,
            0.0,
            settle_s=0.0,
            stop_force=0.28,
            contact_confirm_samples=2,
            contact_confirm_interval_s=0.0,
            empty_width_threshold=0.001,
        )
        self.assertTrue(result.contact_detected)
        self.assertEqual(result.stop_reason, "force_contact")
        self.assertEqual(
            [round(width, 3) for width, _ in arm.gripper.commands],
            [0.015, 0.010, 0.005],
        )

    def test_nonpositive_stop_force_fails_before_closing(self):
        arm = make_arm([])
        result = arm.close_gripper_slow(
            0.085,
            0.0,
            0.30,
            0.002,
            0.0,
            settle_s=0.0,
            stop_force=0.0,
        )
        self.assertFalse(result.contact_detected)
        self.assertEqual(result.stop_reason, "invalid_stop_force")
        self.assertEqual(arm.gripper.commands, [])

    def test_missing_width_feedback_fails_closed(self):
        arm = make_arm([SimpleNamespace(value=None, force=0.05, mode="width")])
        result = arm.close_gripper_slow(
            0.005,
            0.0,
            0.30,
            0.005,
            0.0,
            settle_s=0.0,
            stop_force=0.28,
        )
        self.assertFalse(result.contact_detected)
        self.assertEqual(result.stop_reason, "invalid_width_feedback")

    def test_invalid_slow_close_timing_fails_before_closing(self):
        arm = make_arm([])
        result = arm.close_gripper_slow(
            0.085,
            0.0,
            0.30,
            float("nan"),
            0.0,
            settle_s=0.0,
            stop_force=0.28,
        )
        self.assertFalse(result.contact_detected)
        self.assertEqual(result.stop_reason, "invalid_close_configuration")
        self.assertEqual(arm.gripper.commands, [])

    def test_contact_confirmation_requires_at_least_two_samples(self):
        arm = make_arm([])
        result = arm.close_gripper_slow(
            0.085,
            0.0,
            0.30,
            0.002,
            0.0,
            settle_s=0.0,
            stop_force=0.28,
            contact_confirm_samples=1,
        )
        self.assertEqual(result.stop_reason, "invalid_close_configuration")
        self.assertEqual(arm.gripper.commands, [])

    def test_invalid_close_range_sends_no_gripper_command(self):
        arm = make_arm([])
        result = arm.close_gripper_slow(
            0.0,
            0.0,
            0.30,
            0.002,
            0.0,
            settle_s=0.0,
            stop_force=0.28,
        )
        self.assertEqual(result.stop_reason, "invalid_close_range")
        self.assertEqual(arm.gripper.commands, [])

    def test_direct_close_is_rejected_when_contact_is_required(self):
        args = SimpleNamespace(
            execute=True,
            require_gripper_contact=True,
            slow_close_gripper=False,
            gripper_stop_force=0.28,
            gripper_force=0.30,
        )
        result = runner.plan_and_maybe_execute_grasp(
            object(), args, None, None, None, None, None, None, {},
        )
        self.assertEqual(result.status, runner.GraspAttemptStatus.CONFIGURATION_FAILED)
        self.assertFalse(result.motion_sent)

    def test_stop_force_above_hardware_force_cap_is_rejected_before_planning(self):
        args = SimpleNamespace(
            execute=True,
            require_gripper_contact=True,
            slow_close_gripper=True,
            gripper_stop_force=0.40,
            gripper_force=0.50,
            gripper_hardware_max_force=0.30,
        )
        result = runner.plan_and_maybe_execute_grasp(
            object(), args, None, None, None, None, None, None, {},
        )
        self.assertEqual(result.status, runner.GraspAttemptStatus.CONFIGURATION_FAILED)
        self.assertFalse(result.motion_sent)

    def test_empty_grasp_recovery_returns_to_initial_pose(self):
        args = SimpleNamespace(
            empty_grasp_return_delay=0.0,
            skip_empty_grasp_return=False,
            skip_initial_move=False,
            gripper_open_width=0.085,
            gripper_force=0.30,
            gripper_settle=0.0,
            _empty_grasp_recovery_joints=np.zeros(7, dtype=float),
        )
        arm = mock.Mock()
        with mock.patch.object(runner, "wait_until_gripper_open", return_value=True) as opened, mock.patch.object(
            runner, "wait_until_joints_reached", return_value=(True, np.zeros(7), 0.0)
        ) as wait_joints:
            recovered = runner.recover_after_empty_grasp(arm, args)
        self.assertTrue(recovered)
        arm.open_gripper.assert_called_once_with(0.085, 0.30, 0.0)
        opened.assert_called_once_with(arm, args)
        arm.move_j.assert_called_once()
        wait_joints.assert_called_once()

    def test_open_confirmation_requires_two_valid_consecutive_width_samples(self):
        statuses = iter(
            [
                SimpleNamespace(value=0.085, force=0.0, mode="force"),
                SimpleNamespace(value=float("nan"), force=0.0, mode="width"),
                SimpleNamespace(value=0.085, force=0.0, mode="width"),
                SimpleNamespace(value=0.070, force=0.0, mode="width"),
                SimpleNamespace(value=0.1005, force=0.0, mode="width"),
                SimpleNamespace(value=0.1005, force=0.0, mode="width"),
                SimpleNamespace(value=0.085, force=0.0, mode="width"),
                SimpleNamespace(value=0.085, force=0.0, mode="width"),
            ]
        )
        arm = mock.Mock()
        arm.gripper = object()
        arm.print_gripper_status.side_effect = lambda: next(statuses)
        args = SimpleNamespace(
            gripper_open_width=0.085,
            gripper_hardware_max_width=0.085,
            gripper_open_tol=0.003,
            gripper_open_timeout=1.0,
            gripper_open_confirm_samples=2,
            move_poll_interval=0.0,
        )

        self.assertTrue(runner.wait_until_gripper_open(arm, args))
        self.assertEqual(arm.print_gripper_status.call_count, 8)

    def test_workflow_motion_history_is_preserved_on_later_planning_failure(self):
        args = SimpleNamespace(_workflow_motion_sent=True)
        result = runner.with_workflow_motion(
            args,
            runner.GraspAttemptResult(
                runner.GraspAttemptStatus.PLANNING_FAILED,
                False,
                "no candidate",
            ),
        )
        self.assertTrue(result.motion_sent)
        self.assertEqual(result.status, runner.GraspAttemptStatus.PLANNING_FAILED)


if __name__ == "__main__":
    unittest.main()
