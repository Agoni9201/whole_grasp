import unittest

import numpy as np

from cv_process_updated_yolo26seg import match_visible_amodal_instances


def instance(label, score, x0, y0, x1, y1, shape=(100, 160)):
    mask = np.zeros(shape, dtype=np.uint8)
    mask[y0:y1, x0:x1] = 255
    return {
        "label": label,
        "score": float(score),
        "mask": mask,
        "xyxy": np.array([x0, y0, x1 - 1, y1 - 1], dtype=np.float32),
    }


class VisibleAmodalMatchingTest(unittest.TestCase):
    def test_merges_disconnected_visible_fragments(self):
        visible = [
            instance("banana", 0.92, 20, 30, 50, 60),
            instance("banana", 0.24, 90, 30, 120, 60),
        ]
        amodal = [instance("banana", 0.88, 20, 30, 120, 60)]

        matches = match_visible_amodal_instances(visible, amodal)

        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0]["status"], "matched")
        self.assertEqual(matches[0]["visible_indices"], [0, 1])
        self.assertEqual(matches[0]["visible_instance"]["fragment_count"], 2)
        self.assertAlmostEqual(matches[0]["visible_ratio"], 0.60, places=4)

    def test_keeps_multiple_same_class_objects_separate(self):
        visible = [
            instance("carrot", 0.95, 10, 20, 30, 40),
            instance("carrot", 0.94, 100, 20, 140, 40),
            instance("carrot", 0.25, 40, 20, 60, 40),
        ]
        amodal = [
            instance("carrot", 0.91, 10, 20, 60, 40),
            instance("carrot", 0.90, 100, 20, 140, 40),
        ]

        matches = match_visible_amodal_instances(visible, amodal)
        valid = [record for record in matches if record["occlusion_valid"]]

        self.assertEqual(len(valid), 2)
        self.assertEqual(sorted(len(record["visible_indices"]) for record in valid), [1, 2])
        self.assertFalse(any(record["status"] == "amodal_missing" for record in matches))

    def test_does_not_merge_off_object_same_class_detection(self):
        visible = [
            instance("banana", 0.95, 10, 20, 40, 50),
            instance("banana", 0.30, 110, 20, 140, 50),
        ]
        amodal = [instance("banana", 0.90, 10, 20, 50, 50)]

        matches = match_visible_amodal_instances(visible, amodal)

        valid = [record for record in matches if record["occlusion_valid"]]
        missing = [record for record in matches if record["status"] == "amodal_missing"]
        self.assertEqual(valid[0]["visible_indices"], [0])
        self.assertEqual(len(missing), 1)
        self.assertEqual(missing[0]["visible_indices"], [1])

    def test_rejects_occlusion_when_visible_mask_is_not_contained(self):
        visible = [instance("radish", 0.95, 10, 20, 60, 50)]
        amodal = [instance("radish", 0.90, 30, 20, 70, 50)]

        matches = match_visible_amodal_instances(visible, amodal)

        self.assertEqual(matches[0]["status"], "containment_failed")
        self.assertFalse(matches[0]["occlusion_valid"])
        self.assertLess(matches[0]["visible_inside_amodal_ratio"], 0.85)


if __name__ == "__main__":
    unittest.main()
