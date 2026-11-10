import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

import real_nero_single_grasp_runner as runner


def instance(label, score, x0, y0, x1, y1, shape=(24, 32)):
    mask = np.zeros(shape, dtype=np.uint8)
    mask[y0:y1, x0:x1] = 1
    return {
        "label": label,
        "score": float(score),
        "mask": mask,
        "xyxy": np.array([x0, y0, x1 - 1, y1 - 1], dtype=np.float32),
        "center": runner._mask_center(mask),
        "is_box_only": False,
    }


def args():
    return SimpleNamespace(
        manual_click_only=False,
        target_index=None,
        target_class="carrot",
        select_target=False,
        enable_clear_obstacle=True,
        clear_largest_visible_fallback=True,
        clear_min_points=4,
        min_target_points=4,
        min_depth=0.10,
        max_depth=2.00,
        clear_skip_target_iou=0.35,
        clear_score_threshold=0.35,
        clear_target_bbox_expand=8.0,
        clear_corridor_radius=0.045,
        clear_depth_margin=0.02,
        clear_front_depth_full_score=0.08,
        clear_overlap_weight=0.35,
        clear_front_weight=0.30,
        clear_center_weight=0.15,
        clear_corridor_weight=0.20,
        clear_amodal_weight=0.40,
        clear_amodal_occlusion_threshold=0.25,
        clear_amodal_min_hidden_overlap=0.05,
        clear_largest_fallback_occlusion=0.70,
    )


class LargestVisibleClearFallbackTest(unittest.TestCase):
    def setUp(self):
        self.bgr = np.zeros((24, 32, 3), dtype=np.uint8)
        self.xyz = np.zeros((24, 32, 3), dtype=np.float32)
        self.xyz[:, :, 2] = 0.50
        self.small = instance("radish", 0.99, 2, 2, 6, 6)
        self.large = instance("banana", 0.80, 12, 4, 24, 16)

    def test_selects_largest_visible_mask_and_can_exclude_target(self):
        selected = runner.largest_visible_clear_instance(
            self.bgr,
            self.xyz,
            args(),
            "/tmp",
            instances=[self.small, self.large],
        )
        self.assertEqual(selected["inst"]["label"], "banana")

        selected = runner.largest_visible_clear_instance(
            self.bgr,
            self.xyz,
            args(),
            "/tmp",
            instances=[self.small, self.large],
            exclude_mask=self.large["mask"],
        )
        self.assertEqual(selected["inst"]["label"], "radish")

    def test_rejects_an_area_with_no_valid_depth(self):
        self.xyz[self.large["mask"] > 0, 2] = np.nan
        selected = runner.largest_visible_clear_instance(
            self.bgr,
            self.xyz,
            args(),
            "/tmp",
            instances=[self.small, self.large],
        )
        self.assertEqual(selected["inst"]["label"], "radish")

    def test_missing_requested_target_becomes_clear_only_fallback(self):
        with tempfile.TemporaryDirectory() as out_dir:
            with mock.patch.object(
                runner,
                "infer_yolo_instances",
                side_effect=[([], self.bgr), ([self.small, self.large], self.bgr)],
            ):
                mask, selected = runner.select_target_mask(self.bgr, self.xyz, args(), out_dir)

        self.assertEqual(selected["label"], "banana")
        self.assertTrue(selected["clear_fallback_largest_visible"])
        self.assertEqual(selected["requested_target_class"], "carrot")
        self.assertEqual(int(np.count_nonzero(mask)), 144)

    def test_largest_fallback_bypasses_normal_occluder_score_only(self):
        row = {"largest_visible_fallback": True, "is_front": False, "clear_score": 0.0}
        self.assertTrue(runner.clear_candidate_is_actionable(row, args()))

    def test_high_target_occlusion_keeps_evidence_backed_occluder(self):
        cfg = args()
        target = instance("carrot", 0.98, 2, 2, 6, 6)
        self.xyz[target["mask"] > 0, 2] = 0.70
        occlusion = {
            "enabled": True,
            "valid": True,
            "reason": "matched",
            "label": "carrot",
            "occlusion_ratio": 0.80,
            "hidden_mask": np.zeros(target["mask"].shape, dtype=np.uint8),
            "amodal_mask": target["mask"].copy(),
        }
        with tempfile.TemporaryDirectory() as out_dir:
            with mock.patch.object(
                runner,
                "infer_yolo_instances",
                return_value=([target, self.small, self.large], self.bgr),
            ), mock.patch.object(
                runner,
                "infer_target_amodal_occlusion",
                return_value=occlusion,
            ):
                rows = runner.detect_clear_obstacle_candidates(
                    self.bgr,
                    self.xyz,
                    target["mask"],
                    target,
                    cfg,
                    out_dir,
                )

        self.assertEqual(rows[0]["label"], "banana")
        self.assertFalse(rows[0].get("largest_visible_fallback", False))
        self.assertTrue(runner.clear_candidate_is_actionable(rows[0], cfg))
        self.assertEqual(cfg._last_clear_recommended_action, "clear_first")

    def test_high_target_occlusion_uses_largest_only_without_actionable_evidence(self):
        cfg = args()
        target = instance("carrot", 0.98, 2, 2, 6, 6)
        occlusion = {
            "enabled": True,
            "valid": True,
            "reason": "matched",
            "label": "carrot",
            "occlusion_ratio": 0.80,
            "hidden_mask": np.zeros(target["mask"].shape, dtype=np.uint8),
            "amodal_mask": target["mask"].copy(),
        }
        with tempfile.TemporaryDirectory() as out_dir:
            with mock.patch.object(
                runner,
                "infer_yolo_instances",
                return_value=([target, self.small, self.large], self.bgr),
            ), mock.patch.object(
                runner,
                "infer_target_amodal_occlusion",
                return_value=occlusion,
            ):
                rows = runner.detect_clear_obstacle_candidates(
                    self.bgr,
                    self.xyz,
                    target["mask"],
                    target,
                    cfg,
                    out_dir,
                )

        self.assertEqual(rows[0]["label"], "banana")
        self.assertTrue(rows[0]["largest_visible_fallback"])
        self.assertEqual(cfg._last_clear_recommended_action, "clear_largest_visible")


if __name__ == "__main__":
    unittest.main()
