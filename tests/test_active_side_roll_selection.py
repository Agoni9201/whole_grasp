import unittest
import sys
from types import SimpleNamespace
from unittest import mock

import numpy as np

import real_nero_single_grasp_runner as runner


def roll_args(**overrides):
    values = {
        "active_side_roll_samples": 8,
        "active_side_roll_plan_candidates": 8,
        "active_side_max_joint_delta_deg": 45.0,
        "active_side_max_joint_l2_deg": 75.0,
        "active_direct_min_joint_limit_margin_deg": 0.0,
        "motion_mode": "ik_joint",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def pose_record(roll_deg, rotation_error_deg, marker):
    T = np.eye(4, dtype=float)
    pose = np.zeros(6, dtype=float)
    pose[0] = float(marker)
    return {
        "T_base_camera": T,
        "flange_pose": pose,
        "roll_rad": np.deg2rad(float(roll_deg)),
        "rotation_error_deg": float(rotation_error_deg),
    }


class ActiveSideRollSelectionTest(unittest.TestCase):
    def test_spherical_candidate_builder_uses_bound_rank_hook_contract(self):
        with mock.patch.object(
            sys,
            "argv",
            ["test", "--calib-json", "unused.json"],
        ):
            args = runner.parse_args()
        args.active_nbv_mode = "fixed"
        args.active_path_guard = False
        args.active_obstacle_aware = False
        args.active_anchor_min_disparity_deg = 0.0
        args.active_nbv_min_valid_candidates = 1
        args.active_side_view_count = 1

        target_center = np.array([0.0, 0.0, 0.30], dtype=float)
        target_points = target_center + np.array(
            [
                [-0.02, -0.02, -0.02],
                [-0.02, 0.02, 0.02],
                [0.02, -0.02, 0.02],
                [0.02, 0.02, -0.02],
                [0.0, 0.0, 0.03],
            ],
            dtype=float,
        )
        target_cloud = SimpleNamespace(points=target_points)
        scene_cloud = SimpleNamespace(points=target_points.copy())

        class Arm:
            def get_tcp_pose(self):
                return np.zeros(6, dtype=float)

            def get_flange_pose(self):
                return np.zeros(6, dtype=float)

            def get_joint_angles(self):
                return np.zeros(7, dtype=float)

        def rank(
            _arm,
            _args,
            cam_pos,
            center,
            _T_camera_flange,
            _current_flange,
            _current_joints,
            _label,
            roll_plan_limit=None,
            quick_seed_limit=None,
            allow_full_retry=True,
        ):
            T_base_camera = runner._look_at_camera_pose(cam_pos, center)
            return [
                {
                    "T_base_camera": T_base_camera,
                    "flange_pose": np.r_[cam_pos, 0.0, 0.0, 0.0],
                    "roll_rad": 0.0,
                    "rotation_error_deg": 0.0,
                    "planned_joint_target": np.zeros(7, dtype=float),
                    "joint_max_delta_deg": 0.0,
                    "joint_l2_delta_deg": 0.0,
                    "joint_limit_margin_deg": 30.0,
                }
            ]

        empty_metrics = {
            "count": 0,
            "ratio": 0.0,
            "nearest": float("inf"),
            "penalty": 0.0,
            "weighted_count": 0.0,
        }
        with mock.patch.object(
            runner, "transform_cloud", side_effect=lambda cloud, _T: cloud
        ), mock.patch.object(
            runner,
            "_points_from_cloud",
            side_effect=lambda cloud: np.asarray(cloud.points, dtype=float),
        ), mock.patch.object(
            runner, "handeye_transform", return_value=np.eye(4)
        ), mock.patch.object(
            runner,
            "_estimate_normals_toward_camera",
            return_value=np.tile([0.0, 0.0, 1.0], (target_points.shape[0], 1)),
        ), mock.patch.object(
            runner,
            "_prepare_active_obstacle_cloud",
            return_value=(
                np.zeros((0, 3), dtype=float),
                np.zeros(0, dtype=float),
                {"raw": 0, "kept": 0, "target_removed": 0, "table_removed": 0},
            ),
        ), mock.patch.object(
            runner,
            "_build_active_information_proxy",
            return_value={},
        ), mock.patch.object(
            runner,
            "_score_active_candidate_information",
            return_value={"score": 0.5, "visibility_vector": np.zeros(1)},
        ), mock.patch.object(
            runner,
            "_line_segment_obstacle_metrics",
            return_value=empty_metrics,
        ), mock.patch.object(
            runner, "_nearest_obstacle_distance", return_value=float("inf")
        ), mock.patch.object(
            runner, "_direction_context_penalty", return_value=(0.0, 1.0)
        ), mock.patch.object(
            runner, "_target_visibility_score", return_value=1.0
        ), mock.patch.object(
            runner, "_rank_active_lookat_roll_solutions", side_effect=rank
        ) as rank_hook:
            candidates = runner._build_active_view_candidates(
                Arm(),
                args,
                target_cloud,
                scene_cloud,
                np.eye(4),
                orbit_geometry="spherical_from_top",
                target_center_override=target_center,
            )

        self.assertGreater(rank_hook.call_count, 0)
        self.assertTrue(any(row.get("reason") == "ok" for row in candidates))

    def test_unreachable_minimum_rotation_roll_falls_back(self):
        records = [
            pose_record(0.0, 1.0, 0),
            pose_record(90.0, 30.0, 1),
            pose_record(180.0, 60.0, 2),
        ]

        def solve(_arm, flange_pose, _args, _label, **_kwargs):
            marker = int(round(float(flange_pose[0])))
            if marker == 0:
                raise runner.PlanNotExecutable("closest roll has no IK")
            joints = np.zeros(7, dtype=float)
            joints[0] = np.deg2rad(12.0 if marker == 1 else 20.0)
            return joints

        with mock.patch.object(
            runner,
            "_look_at_camera_pose_flange_roll_candidates",
            return_value=records,
        ), mock.patch.object(
            runner,
            "_solve_ik_for_flange_pose",
            side_effect=solve,
        ):
            ranked = runner._rank_active_lookat_roll_solutions(
                object(),
                roll_args(),
                np.array([0.0, 0.0, 0.3]),
                np.zeros(3),
                np.eye(4),
                np.zeros(6),
                np.zeros(7),
                "side",
            )

        self.assertEqual(len(ranked), 2)
        self.assertAlmostEqual(ranked[0]["flange_pose"][0], 1.0)
        self.assertAlmostEqual(ranked[0]["joint_max_delta_deg"], 12.0)

    def test_rolls_are_ranked_by_joint_motion_before_flange_rotation(self):
        records = [
            pose_record(0.0, 1.0, 0),
            pose_record(90.0, 40.0, 1),
        ]

        def solve(_arm, flange_pose, _args, _label, **_kwargs):
            marker = int(round(float(flange_pose[0])))
            joints = np.zeros(7, dtype=float)
            joints[0] = np.deg2rad(25.0 if marker == 0 else 8.0)
            return joints

        with mock.patch.object(
            runner,
            "_look_at_camera_pose_flange_roll_candidates",
            return_value=records,
        ), mock.patch.object(
            runner,
            "_solve_ik_for_flange_pose",
            side_effect=solve,
        ):
            ranked = runner._rank_active_lookat_roll_solutions(
                object(),
                roll_args(),
                np.array([0.0, 0.0, 0.3]),
                np.zeros(3),
                np.eye(4),
                np.zeros(6),
                np.zeros(7),
                "side",
            )

        self.assertAlmostEqual(ranked[0]["flange_pose"][0], 1.0)
        self.assertAlmostEqual(ranked[0]["joint_max_delta_deg"], 8.0)


if __name__ == "__main__":
    unittest.main()
