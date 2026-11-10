import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

import real_nero_single_grasp_runner as runner


class FakeGrasp:
    rotation_matrix = np.eye(3)
    translation = np.zeros(3)


class FakeArm:
    def get_joint_angles(self):
        return np.zeros(7, dtype=float)

    def get_tcp2flange_pose(self, pose):
        return np.asarray(pose, dtype=float)


def variant_args():
    return SimpleNamespace(
        graspnet_approach_axis="x",
        graspnet_width_axis="y",
        tool_approach_axis="x",
        tcp_gripper_width_axis="z",
        topdown_grasp=False,
        use_current_rpy=False,
        fixed_rpy=None,
        motion_mode="ik_joint",
        grasp_motion_path_guard=False,
        max_candidate_ik_joint_delta=np.deg2rad(95.0),
        max_candidate_pre_pos_delta=0.25,
        max_candidate_pre_rpy_delta_deg=60.0,
    )


class GraspOrientationVariantTest(unittest.TestCase):
    def test_parallel_jaw_sign_flip_preserves_approach_and_width_line(self):
        args = variant_args()
        R_pos = runner.graspnet_rotation_to_tcp_rotation(
            np.eye(3), args, width_sign=1.0
        )
        R_neg = runner.graspnet_rotation_to_tcp_rotation(
            np.eye(3), args, width_sign=-1.0
        )
        tcp_approach = runner._axis_vector(args.tool_approach_axis)
        tcp_width = runner._axis_vector(args.tcp_gripper_width_axis)

        np.testing.assert_allclose(R_pos @ tcp_approach, R_neg @ tcp_approach)
        np.testing.assert_allclose(R_pos @ tcp_width, -(R_neg @ tcp_width))

    def test_native_width_sign_is_kept_when_mirrored_is_closer_to_current_tcp(self):
        args = variant_args()
        R_neg = runner.graspnet_rotation_to_tcp_rotation(
            np.eye(3), args, width_sign=-1.0
        )
        current = np.zeros(6, dtype=float)
        current[3:] = runner.R.from_matrix(R_neg).as_euler("xyz")

        ordered, angles = runner._ordered_equivalent_jaw_width_signs(
            FakeGrasp(),
            np.eye(4),
            current,
            args,
        )

        self.assertEqual(ordered, [1.0])
        self.assertEqual(list(angles), [1.0])
        self.assertAlmostEqual(angles[1.0], np.pi)

    def test_reachability_rejects_unreachable_native_pose_without_mirroring(self):
        args = variant_args()
        arm = FakeArm()
        row = {"idx": 2, "grasp": FakeGrasp()}

        def fake_poses(
            _grasp,
            _T,
            _args,
            _current,
            width_sign=1.0,
            width_shift=0.0,
        ):
            base = np.zeros(6, dtype=float)
            base[0] = 0.10
            if width_sign < 0:
                base[3] = 0.20
            base[2] += float(width_shift)
            return base.copy(), base.copy(), base.copy()

        def fake_solve(_arm, _pose, _args, label, **_kwargs):
            negative = "_neg_" in label
            if label.endswith("_pre"):
                value = 0.5 if negative else 2.0
            elif label.endswith("_grasp"):
                value = 0.6 if negative else 2.1
            else:
                value = 0.7 if negative else 2.2
            q = np.zeros(7, dtype=float)
            q[0] = value
            return q

        with mock.patch.object(
            runner, "grasp_to_tcp_poses", side_effect=fake_poses
        ), mock.patch.object(
            runner, "_solve_ik_for_flange_pose", side_effect=fake_solve
        ):
            annotated = runner.annotate_grasp_candidate_reachability(
                [row], np.eye(4), np.zeros(6), args, arm=arm
            )[0]

        self.assertFalse(annotated["reachability_ok"])
        self.assertNotIn("tcp_width_sign", annotated)
        self.assertNotIn("planned_pre_joints", annotated)

    def test_reachability_keeps_primary_when_fallback_has_lower_joint_cost(self):
        args = variant_args()
        arm = FakeArm()
        row = {"idx": 5, "grasp": FakeGrasp()}
        R_primary = runner.graspnet_rotation_to_tcp_rotation(
            np.eye(3), args, width_sign=1.0
        )
        current = np.zeros(6, dtype=float)
        current[3:] = runner.R.from_matrix(R_primary).as_euler("xyz")

        def fake_poses(
            _grasp,
            _T,
            _args,
            _current,
            width_sign=1.0,
            width_shift=0.0,
        ):
            pose = np.zeros(6, dtype=float)
            pose[0] = 0.10
            pose[3:] = runner.R.from_matrix(
                runner.graspnet_rotation_to_tcp_rotation(
                    np.eye(3), args, width_sign=width_sign
                )
            ).as_euler("xyz")
            pose[2] += float(width_shift)
            return pose.copy(), pose.copy(), pose.copy()

        def fake_solve(_arm, _pose, _args, label, **_kwargs):
            q = np.zeros(7, dtype=float)
            q[0] = 0.40 if "_pos_" in label else 0.10
            return q

        with mock.patch.object(
            runner, "grasp_to_tcp_poses", side_effect=fake_poses
        ), mock.patch.object(
            runner, "_solve_ik_for_flange_pose", side_effect=fake_solve
        ):
            annotated = runner.annotate_grasp_candidate_reachability(
                [row], np.eye(4), current, args, arm=arm
            )[0]

        self.assertTrue(annotated["reachability_ok"])
        self.assertEqual(annotated["jaw_primary_width_sign"], 1.0)
        self.assertEqual(annotated["tcp_width_sign"], 1.0)
        self.assertFalse(annotated["jaw_selected_as_fallback"])
        self.assertEqual(annotated["jaw_selected_role"], "primary")
        self.assertAlmostEqual(annotated["planned_pre_joints"][0], 0.40)

    def test_reachability_rejects_native_pregrasp_that_is_too_far(self):
        args = variant_args()
        arm = FakeArm()
        row = {"idx": 6, "grasp": FakeGrasp()}
        R_primary = runner.graspnet_rotation_to_tcp_rotation(
            np.eye(3), args, width_sign=1.0
        )
        current = np.zeros(6, dtype=float)
        current[3:] = runner.R.from_matrix(R_primary).as_euler("xyz")

        def fake_poses(
            _grasp,
            _T,
            _args,
            _current,
            width_sign=1.0,
            width_shift=0.0,
        ):
            pose = np.zeros(6, dtype=float)
            pose[0] = 0.30 if width_sign > 0.0 else 0.10
            pose[3:] = runner.R.from_matrix(
                runner.graspnet_rotation_to_tcp_rotation(
                    np.eye(3), args, width_sign=width_sign
                )
            ).as_euler("xyz")
            pose[2] += float(width_shift)
            return pose.copy(), pose.copy(), pose.copy()

        def fake_solve(_arm, _pose, _args, label, **_kwargs):
            q = np.zeros(7, dtype=float)
            q[0] = 0.20 if "_pos_" in label else 0.30
            return q

        with mock.patch.object(
            runner, "grasp_to_tcp_poses", side_effect=fake_poses
        ), mock.patch.object(
            runner, "_solve_ik_for_flange_pose", side_effect=fake_solve
        ):
            annotated = runner.annotate_grasp_candidate_reachability(
                [row], np.eye(4), current, args, arm=arm
            )[0]

        self.assertFalse(annotated["reachability_ok"])
        self.assertEqual(annotated["jaw_primary_width_sign"], 1.0)
        self.assertIsNone(annotated["jaw_fallback_width_sign"])
        self.assertEqual(annotated["tcp_width_sign"], 1.0)
        self.assertFalse(annotated["jaw_selected_as_fallback"])
        self.assertEqual(annotated["reachability_ok_signs"], [])

    def test_whole_workflow_guard_defers_legacy_direct_ik(self):
        args = variant_args()
        args.grasp_motion_path_guard = True
        args.motion_up_and_over = True
        arm = FakeArm()
        row = {"idx": 3, "grasp": FakeGrasp()}

        def fake_poses(
            _grasp,
            _T,
            _args,
            _current,
            width_sign=1.0,
            width_shift=0.0,
        ):
            pose = np.zeros(6, dtype=float)
            pose[0] = 0.30  # exceeds the legacy direct-position threshold
            pose[3] = 0.10 if width_sign > 0 else 0.20
            pose[2] += float(width_shift)
            return pose.copy(), pose.copy(), pose.copy()

        with mock.patch.object(
            runner, "grasp_to_tcp_poses", side_effect=fake_poses
        ), mock.patch.object(
            runner, "_solve_ik_for_flange_pose"
        ) as legacy_solver:
            annotated = runner.annotate_grasp_candidate_reachability(
                [row], np.eye(4), np.zeros(6), args, arm=arm
            )[0]

        legacy_solver.assert_not_called()
        self.assertTrue(annotated["table_guard_ok"])
        self.assertTrue(annotated["reachability_ok"])
        self.assertTrue(annotated["reachability_deferred"])
        self.assertEqual(
            annotated["reachability_reason"],
            "deferred_to_whole_workflow_guard",
        )
        self.assertNotIn("planned_pre_joints", annotated)

    def test_guarded_direct_route_rejects_failed_native_endpoint_ik(self):
        args = variant_args()
        args.grasp_motion_path_guard = True
        args.motion_up_and_over = False
        arm = FakeArm()
        row = {"idx": 4, "grasp": FakeGrasp()}

        def fake_poses(
            _grasp,
            _T,
            _args,
            _current,
            width_sign=1.0,
            width_shift=0.0,
        ):
            pose = np.zeros(6, dtype=float)
            pose[0] = 0.10
            pose[3] = 0.10 if width_sign > 0 else 0.20
            pose[2] += float(width_shift)
            return pose.copy(), pose.copy(), pose.copy()

        def fake_solve(_arm, _pose, _args, label, **_kwargs):
            if "_pos_" in label:
                raise runner.PlanNotExecutable("positive endpoint unreachable")
            q = np.zeros(7, dtype=float)
            q[0] = {"pre": 0.2, "grasp": 0.3, "lift": 0.4}[label.rsplit("_", 1)[-1]]
            return q

        with mock.patch.object(
            runner, "grasp_to_tcp_poses", side_effect=fake_poses
        ), mock.patch.object(
            runner, "_solve_ik_for_flange_pose", side_effect=fake_solve
        ) as endpoint_solver:
            annotated = runner.annotate_grasp_candidate_reachability(
                [row], np.eye(4), np.zeros(6), args, arm=arm
            )[0]

        self.assertGreater(endpoint_solver.call_count, 0)
        self.assertFalse(annotated["reachability_ok"])
        self.assertNotIn("tcp_width_sign", annotated)
        self.assertNotIn("planned_pre_joints", annotated)


if __name__ == "__main__":
    unittest.main()
