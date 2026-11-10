import importlib
import unittest
from pathlib import Path
from unittest import mock

import real_nero_grasp.arm as arm_module
import real_nero_grasp.cli as cli_module
import real_nero_grasp.grasp_geometry as geometry_module
import real_nero_grasp.ik_solver as ik_module
import real_nero_grasp.paths as paths
import real_nero_grasp.runtime as runtime
import real_nero_grasp.transforms as transforms_module
import real_nero_grasp.types as model_types


class RealNeroPackageCompatibilityTest(unittest.TestCase):
    def setUp(self):
        # These tests exercise facade/exit-code compatibility. Logging and
        # camera preflight have their own tests and require real argparse data,
        # whereas the exit-code stubs below deliberately return opaque objects.
        for patcher in (mock.patch.object(runtime, 'initialize_experiment_log'),
                        mock.patch('cameras.preflight.grasp_preflight')):
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_legacy_import_is_the_canonical_runtime_facade(self):
        legacy = importlib.import_module("real_nero_single_grasp_runner")

        self.assertIs(legacy, runtime)
        self.assertIs(legacy.GraspAttemptResult, model_types.GraspAttemptResult)
        self.assertIs(legacy.RealNeroGraspArm, arm_module.RealNeroGraspArm)
        self.assertIs(legacy.parse_args, cli_module.parse_args)
        self.assertIs(
            legacy.grasp_to_tcp_poses,
            geometry_module.grasp_to_tcp_poses,
        )
        self.assertIs(legacy._as_pose_list, transforms_module._as_pose_list)
        self.assertIs(legacy._get_ik_solver, ik_module._get_ik_solver)

    def test_default_paths_remain_anchored_at_the_project_root(self):
        project_root = Path(__file__).resolve().parents[1]

        self.assertEqual(paths.PROJECT_ROOT, project_root)
        self.assertEqual(Path(runtime.ROOT_DIR), project_root)
        self.assertEqual(
            Path(runtime.DEFAULT_IK_URDF),
            project_root
            / "manipulator_grasp"
            / "nero_urdf"
            / "agx_arm_urdf"
            / "nero"
            / "urdf"
            / "nero_description.urdf",
        )

    def test_main_preserves_success_and_general_failure_exit_codes(self):
        success = model_types.GraspAttemptResult(
            model_types.GraspAttemptStatus.SUCCESS
        )
        failure = model_types.GraspAttemptResult(
            model_types.GraspAttemptStatus.PLANNING_FAILED,
            detail="unit-test",
        )

        with mock.patch.object(runtime, "parse_args", return_value=object()), mock.patch.object(
            runtime, "run_single_grasp", return_value=success
        ):
            self.assertEqual(runtime.main(), 0)
        with mock.patch.object(runtime, "parse_args", return_value=object()), mock.patch.object(
            runtime, "run_single_grasp", return_value=failure
        ):
            self.assertEqual(runtime.main(), 2)

    def test_main_preserves_controlled_stop_and_estop_exit_codes(self):
        with mock.patch.object(runtime, "parse_args", return_value=object()), mock.patch.object(
            runtime,
            "run_single_grasp",
            side_effect=model_types.ControlledMotionStopTriggered("controlled"),
        ):
            self.assertEqual(runtime.main(), 2)
        with mock.patch.object(runtime, "parse_args", return_value=object()), mock.patch.object(
            runtime,
            "run_single_grasp",
            side_effect=model_types.SafetyStopTriggered("estop"),
        ):
            self.assertEqual(runtime.main(), 3)

    def test_grasp_plan_facade_forwards_stage_cache(self):
        stage_cache = {}
        planning_metrics = {}
        with mock.patch.object(
            runtime._grasp_planner_impl,
            "plan_up_and_over_grasp_sequence",
            return_value="plan",
        ) as planner:
            result = runtime.plan_up_and_over_grasp_sequence(
                mock.Mock(),
                [0.0] * 6,
                [0.0] * 6,
                [0.0] * 6,
                None,
                mock.Mock(),
                mock.Mock(),
                mock.Mock(),
                mock.Mock(),
                "unit",
                planning_stage_cache=stage_cache,
                planning_metrics=planning_metrics,
                validation_only=True,
            )

        self.assertEqual(result, "plan")
        self.assertIs(
            planner.call_args.kwargs["planning_stage_cache"], stage_cache
        )
        self.assertIs(
            planner.call_args.kwargs["planning_metrics"], planning_metrics
        )
        self.assertTrue(planner.call_args.kwargs["validation_only"])


if __name__ == "__main__":
    unittest.main()
