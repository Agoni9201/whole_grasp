from datetime import datetime
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

from real_nero_grasp.multiview_archive import save_unfused_view, update_unfused_view


def cloud(points, colors=None):
    return SimpleNamespace(
        points=np.asarray(points),
        colors=np.empty((0, 3)) if colors is None else np.asarray(colors),
    )


def read_ply(path, colored=True):
    with Path(path).open("rb") as stream:
        header = []
        while True:
            line = stream.readline().decode("ascii").strip()
            header.append(line)
            if line == "end_header":
                break
        fields = [(axis, "<f8") for axis in ("x", "y", "z")]
        if colored:
            fields += [(color, "u1") for color in ("red", "green", "blue")]
        values = np.fromfile(stream, dtype=np.dtype(fields))
    return header, values


class MultiviewArchiveTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.points = np.array([[0.1, 0.2, 0.3], [-0.2, 0.3, 0.4]], dtype=np.float32)
        self.colors = np.array([[0.12, 0.5, 1.0], [1.0, 0.0, 0.25]], dtype=np.float64)
        self.target = cloud(self.points, self.colors)
        self.transform = np.array([
            [0.0, -1.0, 0.0, 0.4], [1.0, 0.0, 0.0, -0.1],
            [0.0, 0.0, 1.0, 0.2], [0.0, 0.0, 0.0, 1.0],
        ])

    def save(self, **kwargs):
        return save_unfused_view(
            self.root, "active_side1", self.target, self.target,
            self.transform, **kwargs,
        )

    @staticmethod
    def directory(record):
        return Path(record["archive_root"]) / record["view_dir"]

    def test_exact_arrays_native_colors_and_handeye_only_transform(self):
        before_points, before_colors = self.points.copy(), self.colors.copy()
        record = self.save(
            obstacle_cloud=self.target, capture_joints=np.arange(7) / 10,
            intrinsics=SimpleNamespace(fx=np.float32(900), fy=901, cx=640, cy=360),
            instance={"label": "bottle", "score": np.float32(0.9)},
            pose_read_monotonic=123.4,
        )
        self.assertIsNotNone(record)
        directory = self.directory(record)
        expected = self.points @ self.transform[:3, :3].T + self.transform[:3, 3]
        with np.load(directory / "clouds.npz") as arrays:
            for prefix in ("target", "scene", "obstacle"):
                np.testing.assert_array_equal(arrays[f"{prefix}_camera_points"], before_points)
                np.testing.assert_array_equal(arrays[f"{prefix}_base_points"], expected)
                np.testing.assert_array_equal(arrays[f"{prefix}_camera_colors"], before_colors)
                np.testing.assert_array_equal(arrays[f"{prefix}_base_colors"], before_colors)
            self.assertEqual(arrays["target_camera_points"].dtype, before_points.dtype)
        _, vertices = read_ply(directory / "target_base.ply")
        np.testing.assert_array_equal(
            np.column_stack([vertices[axis] for axis in ("x", "y", "z")]), expected,
        )
        np.testing.assert_array_equal(
            np.column_stack([vertices[color] for color in ("red", "green", "blue")]),
            np.rint(self.colors * 255).astype(np.uint8),
        )
        metadata = json.loads((directory / "metadata.json").read_text())
        self.assertFalse(metadata["registration_applied"])
        self.assertFalse(metadata["fusion_applied"])
        self.assertFalse(metadata["pose_synchronized_to_rgbd"])
        self.assertEqual(metadata["pose_read_monotonic"], 123.4)
        self.assertEqual(metadata["intrinsics"]["fx"], 900.0)
        self.assertIsNotNone(datetime.fromisoformat(metadata["created_at"]).utcoffset())
        np.testing.assert_array_equal(self.points, before_points)
        np.testing.assert_array_equal(self.colors, before_colors)

    def test_repeated_labels_have_unique_portable_manifest_entries(self):
        one, two = self.save(), self.save()
        self.assertNotEqual(one["view_dir"], two["view_dir"])
        manifest = Path(one["archive_root"]) / "manifest.jsonl"
        entries = [json.loads(line) for line in manifest.read_text().splitlines()]
        self.assertEqual(len(entries), 2)
        for entry in entries:
            self.assertFalse(Path(entry["metadata_path"]).is_absolute())
            self.assertTrue((manifest.parent / entry["metadata_path"]).is_file())
            self.assertNotIn("archive_root", entry)

    def test_large_instance_mask_is_not_expanded_into_json(self):
        class NoPixelList(np.ndarray):
            def tolist(self):
                raise AssertionError("image arrays must never be expanded into JSON")

        mask = np.ones((720, 1280), dtype=np.uint8).view(NoPixelList)
        record = self.save(
            instance={
                "label": "bottle", "score": np.float32(0.9), "mask": mask,
                "xyxy": np.array([100, 200, 300, 400]),
                "center": [200, 300], "unrelated_pixels": mask,
            },
            args=SimpleNamespace(
                min_depth=0.05, voxel_size=0.002, target_outlier_filter=True,
                unrelated_camera_buffer=mask,
            ),
        )
        self.assertIsNotNone(record)
        encoded = (self.directory(record) / "metadata.json").read_text()
        self.assertLess(len(encoded), 12000)
        metadata = json.loads(encoded)
        instance = metadata["instance"]
        self.assertEqual(instance["xyxy"], [100, 200, 300, 400])
        self.assertEqual(instance["center"], [200, 300])
        self.assertNotIn("mask", instance)
        self.assertEqual(instance["omitted_arrays"]["mask"], {
            "shape": [720, 1280], "dtype": "uint8", "data_omitted": True,
            "mask_source_file": "selected_target_mask.png",
        })
        self.assertEqual(metadata["single_view_processing_settings"], {
            "min_depth": 0.05, "voxel_size": 0.002, "target_outlier_filter": True,
        })
        self.assertEqual(mask.shape, (720, 1280))
        self.assertEqual(int(mask[0, 0]), 1)

    def test_disabled_creates_no_archive(self):
        self.assertIsNone(self.save(args=SimpleNamespace(save_active_raw_views=False)))
        self.assertEqual(list(self.root.iterdir()), [])

    def test_rejected_view_is_preserved_with_status(self):
        record = self.save()
        directory = self.directory(record)
        original = (directory / "clouds.npz").read_bytes()
        update_unfused_view(record, capture_quality_keep=False, reject_reason="poor_overlap")
        update_unfused_view(record, fusion_keep=False)
        metadata = json.loads((directory / "metadata.json").read_text())
        self.assertEqual(metadata["status"], {
            "capture_quality_keep": False, "reject_reason": "poor_overlap", "fusion_keep": False,
        })
        self.assertEqual((directory / "clouds.npz").read_bytes(), original)
        self.assertTrue((directory / "target_base.ply").is_file())

    def test_capture_files_are_copied_and_calibration_is_snapshotted(self):
        source = self.root / "capture"
        source.mkdir()
        expected = {}
        for name in ("color.png", "depth_m.npy", "capture_meta.json", "selected_target_mask.png",
                     "depth_raw_m.npy", "depth_raw_frames.npz", "camera_self_mask.png",
                     "camera_self_mask_overlay.png", "camera_self_mask_report.json",
                     "depth_sdk_m.npy", "depth_raw_valid_count.npy",
                     "depth_support_rejected.png", "depth_support_report.json"):
            expected[name] = f"sample {name}".encode("ascii")
            (source / name).write_bytes(expected[name])
        calibration = self.root / "calibration.json"
        snapshot = {"matrix_4x4": np.eye(4).tolist()}
        calibration.write_text(json.dumps(snapshot))
        record = self.save(capture_dir=source, args=SimpleNamespace(
            calib_json=calibration, handeye_convention="flange_camera_xyzw",
        ))
        directory = self.directory(record)
        metadata = json.loads((directory / "metadata.json").read_text())
        for name, content in expected.items():
            self.assertEqual((directory / name).read_bytes(), content)
            self.assertEqual((source / name).read_bytes(), content)
        self.assertEqual(metadata["missing_capture_files"], [])
        self.assertEqual(metadata["handeye"]["snapshot"], snapshot)
        self.assertEqual(json.loads(calibration.read_text()), snapshot)

    def test_empty_cloud_and_absent_colors_are_supported(self):
        record = save_unfused_view(
            self.root, "empty", cloud(np.empty((0, 3))),
            cloud(self.points), np.eye(4),
        )
        directory = self.directory(record)
        header, vertices = read_ply(directory / "target_camera.ply", colored=False)
        self.assertIn("element vertex 0", header)
        self.assertEqual(len(vertices), 0)
        with np.load(directory / "clouds.npz") as arrays:
            self.assertEqual(arrays["target_camera_points"].shape, (0, 3))
            self.assertEqual(arrays["scene_camera_colors"].shape, (0, 3))

    def test_invalid_shape_has_error_record_without_success_manifest(self):
        bad = cloud(np.zeros((3, 4)))
        record = save_unfused_view(self.root, "bad", bad, self.target, np.eye(4))
        self.assertIsNone(record)
        archive = self.root / "multiview_raw"
        self.assertFalse((archive / "manifest.jsonl").exists())
        metadata = json.loads(next(archive.glob("*/metadata.json")).read_text())
        self.assertEqual(metadata["archive_status"], "failed")
        self.assertIn("N x 3", metadata["archive_error"])
        np.testing.assert_array_equal(bad.points, np.zeros((3, 4)))

    def test_io_failure_does_not_escape_or_create_success_manifest(self):
        with mock.patch("real_nero_grasp.multiview_archive.np.savez", side_effect=OSError("disk full")):
            record = self.save()
        self.assertIsNone(record)
        archive = self.root / "multiview_raw"
        self.assertFalse((archive / "manifest.jsonl").exists())
        metadata = json.loads(next(archive.glob("*/metadata.json")).read_text())
        self.assertEqual(metadata["archive_status"], "failed")


if __name__ == "__main__":
    unittest.main()
