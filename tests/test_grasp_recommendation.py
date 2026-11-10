import unittest
from types import SimpleNamespace

import numpy as np

from real_nero_grasp.grasp_candidates import recommend_grasp_candidate


def recommendation_args():
    return SimpleNamespace(
        table_z=0.080,
        recommended_grasp_finger_table_margin=0.003,
        gripper_open_width=0.085,
        gripper_hardware_max_width=0.085,
        motion_path_open_finger_radius=0.007,
        motion_path_open_finger_back=0.065,
        motion_path_open_finger_front=0.020,
        tcp_gripper_width_axis="z",
        tool_approach_axis="x",
    )


class GraspRecommendationTest(unittest.TestCase):
    def test_prefers_table_safe_primary_over_risky_or_180_fallback(self):
        rows = [
            {
                "idx": 0,
                "priority": 0.70,
                "vertical_align": 0.8,
                "raw_score": 0.5,
                "jaw_selected_role": "primary",
                "grasp_tcp_preview": np.array([0, 0, 0.160, 0, 0, 0]),
            },
            {
                "idx": 1,
                "priority": 0.90,
                "vertical_align": 0.9,
                "raw_score": 0.8,
                "jaw_selected_role": "fallback_180",
                "grasp_tcp_preview": np.array([0, 0, 0.160, 0, 0, 0]),
            },
            {
                "idx": 2,
                "priority": 0.99,
                "vertical_align": 0.9,
                "raw_score": 0.9,
                "jaw_selected_role": "primary",
                "grasp_tcp_preview": np.array([0, 0, 0.120, 0, 0, 0]),
            },
        ]

        chosen = recommend_grasp_candidate(rows, recommendation_args())

        self.assertEqual(chosen["idx"], 0)
        self.assertTrue(chosen["recommended_grasp"])
        self.assertTrue(chosen["recommended_table_safe"])
        self.assertFalse(rows[2]["recommended_table_safe"])
        self.assertIn("open_finger_table_margin", chosen["recommended_reason"])
        self.assertEqual(
            [row["idx"] for row in rows if row["recommended_grasp"]],
            [0],
        )

    def test_legacy_rows_without_pose_keep_priority_ordering(self):
        rows = [
            {"idx": 4, "priority": 0.8},
            {"idx": 7, "priority": 0.6},
        ]

        chosen = recommend_grasp_candidate(rows, recommendation_args())

        self.assertEqual(chosen["idx"], 4)
        self.assertIsNone(chosen["recommended_table_margin"])

    def test_elongated_target_prefers_cross_axis_over_high_legacy_priority(self):
        rows = [
            {
                "idx": 0,
                "priority": 0.95,
                "vertical_align": 0.90,
                "raw_score": 0.60,
                "target_direction_applicable": True,
                "target_cross_axis_score": 0.08,
                "graspnet_nominal_width": 0.060,
                "physical_aperture_variants": [
                    {"ok": True, "width_sign": 1.0, "points": 180, "robust_span": 0.040}
                ],
                "tcp_width_sign": 1.0,
                "grasp_tcp_preview": np.array([0, 0, 0.160, 0, 0, 0]),
            },
            {
                "idx": 1,
                "priority": 0.65,
                "vertical_align": 0.82,
                "raw_score": 0.35,
                "target_direction_applicable": True,
                "target_cross_axis_score": 0.72,
                "graspnet_nominal_width": 0.060,
                "physical_aperture_variants": [
                    {"ok": True, "width_sign": 1.0, "points": 150, "robust_span": 0.045}
                ],
                "tcp_width_sign": 1.0,
                "grasp_tcp_preview": np.array([0, 0, 0.160, 0, 0, 0]),
            },
        ]

        chosen = recommend_grasp_candidate(rows, recommendation_args())

        self.assertEqual(chosen["idx"], 1)
        self.assertTrue(chosen["recommended_quality_ok"])
        self.assertFalse(rows[0]["recommended_direction_ok"])

    def test_only_low_quality_executable_row_is_labeled_fallback(self):
        row = {
            "idx": 3,
            "priority": 0.90,
            "vertical_align": 0.90,
            "raw_score": 0.05,
            "target_direction_applicable": True,
            "target_cross_axis_score": 0.10,
            "physical_aperture_variants": [
                {"ok": True, "width_sign": 1.0, "points": 20, "robust_span": 0.020}
            ],
            "tcp_width_sign": 1.0,
            "grasp_tcp_preview": np.array([0, 0, 0.160, 0, 0, 0]),
        }

        chosen = recommend_grasp_candidate([row], recommendation_args())

        self.assertTrue(chosen["recommended_grasp"])
        self.assertFalse(chosen["recommended_quality_ok"])
        self.assertIn("low_confidence_fallback", chosen["recommended_reason"])


if __name__ == "__main__":
    unittest.main()
