from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock
import xml.etree.ElementTree as ET

import numpy as np

from real_nero_grasp import collision
from real_nero_grasp import arm_collision_geometry as geometry
from real_nero_grasp.paths import DEFAULT_IK_URDF


def arm_args():
    return SimpleNamespace(
        ik_urdf=DEFAULT_IK_URDF, ik_base_link="base_link", ik_tip_link="link7",
        arm_collision_model="urdf_capsules", tcp_offset=[0.165, 0, -0.025, 0, 0, 0],
        calib_json="unused", handeye_convention="flange_camera_xyzw",
    )


def triangle_coverage(triangles, capsules):
    """Independent check: one convex capsule must contain each whole triangle."""
    worst_per_capsule = []
    for a, b, radius in capsules:
        axis = b - a
        length2 = float(axis @ axis)
        projection = np.zeros(triangles.shape[:2]) if length2 < 1e-18 else np.clip(
            np.sum((triangles - a) * axis, axis=2) / length2, 0, 1
        )
        distance = np.linalg.norm(triangles - a - projection[..., None] * axis, axis=2)
        worst_per_capsule.append(np.max(distance - radius, axis=1))
    return np.min(worst_per_capsule, axis=0)


class UrdfArmCollisionGeometryTest(unittest.TestCase):
    def test_all_real_collision_triangles_have_three_mm_coverage(self):
        args = arm_args()
        chain = collision._load_urdf_joint_chain(args)
        fitted = geometry.load_arm_geometry(args, chain)
        root = ET.parse(args.ik_urdf).getroot()
        for link in root.findall("link"):
            name = link.get("name")
            if name not in fitted["models"]:
                continue
            mesh = link.find("collision/geometry/mesh")
            triangles = geometry.read_stl_triangles(geometry.resolve_mesh(args.ik_urdf, mesh.get("filename")))
            with self.subTest(link=name):
                self.assertLessEqual(float(triangle_coverage(triangles, fitted["models"][name]).max()), -0.003 + 1e-8)
        self.assertEqual(set(fitted["models"]), {f"link{i}" for i in range(1, 8)})

    def test_child_frame_includes_joint_rotation_and_fixed_offsets(self):
        joint = {"type": "revolute", "child": "one", "origin": np.eye(4), "axis": np.array([0, 0, 1.0])}
        fixed_origin = np.eye(4)
        fixed_origin[0, 3] = 0.2
        fixed = {"type": "fixed", "child": "fixed", "origin": fixed_origin}
        _, tip, frames = collision._urdf_chain_frames(
            np.r_[np.pi / 2, np.zeros(6)], arm_args(),
            load_urdf_joint_chain_hook=lambda _: [joint, fixed],
        )
        np.testing.assert_allclose(frames["one"][:3, :3] @ [1, 0, 0], [0, 1, 0], atol=1e-12)
        np.testing.assert_allclose(tip[:3, 3], [0, 0.2, 0], atol=1e-12)

    def test_transformed_mesh_stays_covered_at_several_joint_configurations(self):
        args = arm_args()
        chain = collision._load_urdf_joint_chain(args)
        fitted = geometry.load_arm_geometry(args, chain)
        for q in (np.zeros(7), np.array([0.7, -0.4, 0.5, 1.4, -0.6, 0.2, 0.8])):
            _, _, frames = collision._urdf_chain_frames(q, args)
            world = geometry.arm_capsules(args, chain, frames)
            for name, local in fitted["models"].items():
                chosen = [item for item in world if item[3].startswith(f"urdf_{name}_")]
                for (a, b, radius), actual in zip(local, chosen):
                    np.testing.assert_allclose(actual[0], frames[name][:3, :3] @ a + frames[name][:3, 3])
                    np.testing.assert_allclose(actual[1], frames[name][:3, :3] @ b + frames[name][:3, 3])
                    self.assertEqual(actual[2], radius)

    def test_nbv_and_grasp_use_identical_arm_geometry(self):
        args = arm_args()
        with mock.patch.object(collision, "_camera_collision_capsules", return_value=[]):
            active = collision._active_robot_capsules(np.zeros(7), args)
            grasp = collision._motion_robot_capsules(np.zeros(7), args)
        active = [item for item in active if item[3].startswith("urdf_")]
        grasp = [item for item in grasp if item[3].startswith("urdf_")]
        self.assertEqual(len(active), 7)
        for left, right in zip(active, grasp):
            np.testing.assert_allclose(left[0], right[0])
            np.testing.assert_allclose(left[1], right[1])
            self.assertEqual(left[2:], right[2:])

    def test_missing_mesh_fails_without_legacy_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "robot.urdf"
            path.write_text('<robot><link name="part"><collision><geometry><mesh filename="missing.stl"/></geometry></collision></link></robot>')
            with self.assertRaisesRegex(ValueError, "Missing"):
                geometry.load_arm_geometry(SimpleNamespace(ik_urdf=str(path)), [{"child": "part"}])

    def test_collision_origin_rotation_and_mesh_scale_are_applied(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "robot.urdf"
            mesh_path = path.parent / "part.stl"
            triangles = np.array([
                [[0, 0, 0], [0.02, 0, 0], [0, 0.01, 0]],
                [[0, 0, 0], [0.02, 0, 0], [0, 0, 0.01]],
            ])
            mesh_path.write_text("solid mesh\n" + "\n".join(
                "vertex " + " ".join(str(value) for value in vertex)
                for vertex in triangles.reshape(-1, 3)
            ) + "\nendsolid mesh\n")
            path.write_text('<robot><link name="part"><collision><origin xyz="0.4 0.1 0.2" rpy="0 0 1.5707963267948966"/><geometry><mesh filename="part.stl" scale="2 2 2"/></geometry></collision></link></robot>')
            fitted = geometry.load_arm_geometry(SimpleNamespace(ik_urdf=str(path)), [{"child": "part"}])
            rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
            expected = (triangles * 2) @ rotation.T + [0.4, 0.1, 0.2]
            self.assertLessEqual(float(triangle_coverage(expected, fitted["models"]["part"]).max()), -0.003 + 1e-8)

    def test_unknown_model_rejected(self):
        args = arm_args()
        args.arm_collision_model = "typo"
        with self.assertRaisesRegex(collision.UnsafeMotionTarget, "Unknown arm collision"):
            collision._arm_collision_capsules(np.zeros(7), args, np.zeros((7, 3)), 0.06)

    def test_thin_wrist_does_not_inherit_sixty_mm_sphere(self):
        args = arm_args()
        fitted = geometry.load_arm_geometry(args, collision._load_urdf_joint_chain(args))
        self.assertLess(max(item[2] for item in fitted["models"]["link7"]), 0.020)
        # A point at a capsule's centerline is still a collision.
        a, b, radius = fitted["models"]["link7"][0]
        point = 0.5 * (a + b)
        self.assertLess(geometry._distances(point[None, :], a, b)[0] - radius, 0)

    def test_geometry_is_warmed_before_hardware_workflow(self):
        from real_nero_grasp import runtime
        order = []
        args = SimpleNamespace(trajectory_executor="move_j")
        with mock.patch.object(collision, "initialize_arm_collision_geometry", side_effect=lambda _: order.append("geometry")), mock.patch.object(
            runtime._workflow_impl, "run_single_grasp", side_effect=lambda *a, **kw: order.append("hardware_workflow")
        ), mock.patch("cameras.realsense.close_all"):
            runtime.run_single_grasp(args)
        self.assertEqual(order, ["geometry", "hardware_workflow"])

    def test_geometry_failure_prevents_hardware_workflow(self):
        from real_nero_grasp import runtime
        with mock.patch.object(collision, "initialize_arm_collision_geometry", side_effect=ValueError("invalid CAD")), mock.patch.object(
            runtime._workflow_impl, "run_single_grasp"
        ) as workflow, mock.patch("cameras.realsense.close_all"):
            with self.assertRaisesRegex(ValueError, "invalid CAD"):
                runtime.run_single_grasp(SimpleNamespace(trajectory_executor="move_j"))
        workflow.assert_not_called()


if __name__ == "__main__":
    unittest.main()
