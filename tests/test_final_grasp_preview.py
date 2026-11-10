import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
from scipy.spatial.transform import Rotation

from real_nero_grasp import runtime
from real_nero_grasp.collision import _motion_open_gripper_finger_capsules
from real_nero_grasp.grasp_geometry import grasp_to_tcp_poses
from real_nero_grasp.transforms import _make_T_from_pose


def preview_args():
    return SimpleNamespace(
        preview_final_grasp=True,
        tool_approach_axis="x",
        tcp_gripper_width_axis="z",
        graspnet_approach_axis="x",
        graspnet_width_axis="y",
        motion_path_open_finger_radius=0.007,
        motion_path_open_finger_back=0.065,
        motion_path_open_finger_front=0.005,
        gripper_hardware_max_width=0.085,
        gripper_open_width=0.085,
        topdown_grasp=False,
        use_graspnet_depth_advance=False,
        grasp_advance=0.030,
        min_grasp_z=None,
        pregrasp_offset=0.100,
        lift_height=0.070,
        fixed_rpy=None,
        use_current_rpy=False,
    )


class FinalGraspPreviewTest(unittest.TestCase):
    def test_finger_preview_matches_collision_and_bridge_is_behind_tcp(self):
        args = preview_args()
        T = _make_T_from_pose([0.13, -0.21, 0.44, 0.4, -0.3, 0.7])
        geometry = runtime._make_execution_gripper_lineset(T, 0.085, args)
        points = np.asarray(geometry.points)
        fingers = _motion_open_gripper_finger_capsules(T, args, opening=0.085)
        np.testing.assert_allclose(points[[3, 4]], fingers[0][:2])
        np.testing.assert_allclose(points[[5, 6]], fingers[1][:2])
        bridge = np.asarray(geometry.lines)[0]
        np.testing.assert_array_equal(bridge, [3, 5])
        approach = T[:3, 0]
        np.testing.assert_allclose(
            (points[bridge] - T[:3, 3]) @ approach, [-0.065, -0.065]
        )
        self.assertAlmostEqual(
            np.linalg.norm(points[2] - points[1]) - 2 * 0.007, 0.085
        )

    def test_final_preview_uses_commanded_tcp_in_cloud_camera_frame(self):
        args = preview_args()
        T_base_camera = _make_T_from_pose([0.2, -0.1, 0.6, 0.3, 2.2, -0.4])
        grasp = SimpleNamespace(
            rotation_matrix=Rotation.from_euler("xyz", [0.2, 0.1, -0.6]).as_matrix(),
            translation=np.array([0.01, -0.02, 0.4]),
            width=0.075,
        )
        pre, command, _ = grasp_to_tcp_poses(
            grasp, T_base_camera, args, width_shift=0.004
        )
        cloud = runtime.get_o3d().geometry.PointCloud()
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.object(runtime, "draw_open3d_preview") as draw:
                self.assertTrue(runtime.preview_final_grasp_candidate(
                    cloud, grasp, {"idx": 3}, T_base_camera, pre, command,
                    1, 0.004, args, directory,
                ))
            metadata = json.loads((Path(directory) / "final_grasp_preview.json").read_text())
            expected = np.linalg.inv(T_base_camera) @ _make_T_from_pose(command)
            np.testing.assert_allclose(metadata["T_camera0_tcp"], expected)
            drawn = np.asarray(draw.call_args.args[0][2].points)
            np.testing.assert_allclose(drawn[0], expected[:3, 3])
            self.assertLess(metadata["approach_angle_deg"], 0.001)
            self.assertAlmostEqual(
                np.linalg.norm(
                    np.array(metadata["T_camera0_pregrasp"])[:3, 3] - drawn[0]
                ),
                0.100,
            )


if __name__ == "__main__":
    unittest.main()
