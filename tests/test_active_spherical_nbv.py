import unittest
from types import SimpleNamespace

import numpy as np

import real_nero_single_grasp_runner as runner


def nbv_args(**overrides):
    values = {
        "active_nbv_orbit_degs": "20,30,35,45",
        "active_orbit_max_angle_deg": 45.0,
        "active_nbv_surface_weight": 0.35,
        "active_nbv_grasp_contact_weight": 0.30,
        "active_nbv_sweep_weight": 0.15,
        "active_nbv_occlusion_weight": 0.10,
        "active_nbv_image_weight": 0.10,
        "active_nbv_redundancy_weight": 0.10,
        "active_nbv_motion_weight": 0.10,
        "active_view_max_center_norm": 0.75,
        "active_orbit_min_radius": 0.30,
        "active_side_max_joint_delta_deg": 45.0,
        "active_side_max_joint_l2_deg": 75.0,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def information_result(proxy, cam_pos, args, **overrides):
    inputs = {
        "los_penalty": 0.0,
        "direction_collision": 0.0,
        "center_norm": 0.0,
        "depth": 0.30,
        "joint_max_delta_deg": 0.0,
        "joint_l2_delta_deg": 0.0,
    }
    inputs.update(overrides)
    result = runner._score_active_candidate_information(
        proxy,
        np.asarray(cam_pos, dtype=float),
        args=args,
        **inputs,
    )
    if not isinstance(result, dict):
        raise AssertionError(
            "_score_active_candidate_information must return its score "
            "components as a dict so active-view decisions remain auditable"
        )
    return result


class ActiveSphericalNBVTest(unittest.TestCase):
    def test_22_5_degree_step_generates_full_sixteen_azimuth_grid(self):
        azimuths = np.asarray(
            runner._active_spherical_azimuths(22.5), dtype=float
        ).reshape(-1)

        self.assertEqual(azimuths.shape[0], 16)
        azimuths_deg = np.mod(np.rad2deg(azimuths), 360.0)
        np.testing.assert_allclose(
            np.sort(azimuths_deg),
            np.arange(0.0, 360.0, 22.5),
            atol=1e-9,
        )
        self.assertEqual(
            np.unique(np.round(azimuths_deg, decimals=9)).shape[0], 16
        )

    def test_orbit_tilts_default_to_requested_spherical_levels(self):
        args = SimpleNamespace(active_orbit_max_angle_deg=45.0)

        tilts = runner._parse_active_orbit_tilts(args)

        self.assertEqual(
            {round(float(value), 6) for value in tilts},
            {20.0, 30.0, 35.0, 45.0},
        )
        self.assertEqual(len(tilts), 4)

    def test_orbit_tilts_are_deduplicated_and_above_max_are_dropped(self):
        args = nbv_args(
            active_nbv_orbit_degs="20, 30 30, 35, 45, 60",
            active_orbit_max_angle_deg=34.0,
        )

        tilts = runner._parse_active_orbit_tilts(args)

        self.assertEqual(
            {round(float(value), 6) for value in tilts}, {20.0, 30.0}
        )
        self.assertEqual(len(tilts), 2)
        self.assertNotIn(34.0, [round(float(value), 6) for value in tilts])

    def test_spherical_camera_position_keeps_radius_and_polar_height(self):
        center = np.array([0.20, -0.10, 0.15], dtype=float)
        radius = 0.40
        azimuth = np.deg2rad(90.0)
        tilt_deg = 30.0

        camera = runner._active_spherical_camera_position(
            center, radius, azimuth, tilt_deg
        )

        offset = np.asarray(camera, dtype=float) - center
        self.assertAlmostEqual(float(np.linalg.norm(offset)), radius, places=9)
        self.assertAlmostEqual(
            float(offset[2]),
            radius * np.cos(np.deg2rad(tilt_deg)),
            places=9,
        )
        np.testing.assert_allclose(
            offset[:2],
            [0.0, radius * np.sin(np.deg2rad(tilt_deg))],
            atol=1e-9,
        )

    def test_side_view_adds_surface_and_contact_information(self):
        center = np.zeros(3, dtype=float)
        target_points = np.array(
            [
                [0.050, 0.000, 0.000],
                [-0.050, 0.000, 0.000],
                [0.000, 0.025, 0.000],
                [0.000, -0.025, 0.000],
                [0.020, 0.010, 0.015],
                [-0.020, -0.010, 0.015],
            ],
            dtype=float,
        )
        target_normals = np.array(
            [
                [1.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, -1.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=float,
        )
        current_camera = np.array([0.0, 0.0, 0.30], dtype=float)
        args = nbv_args()
        proxy = runner._build_active_information_proxy(
            target_points,
            target_normals,
            center,
            current_camera,
            args,
        )

        same = information_result(proxy, current_camera, args)
        side_camera = runner._active_spherical_camera_position(
            center, 0.30, 0.0, 35.0
        )
        side = information_result(proxy, side_camera, args)

        self.assertGreater(side["surface_gain"], same["surface_gain"])
        self.assertGreater(side["contact_gain"], same["contact_gain"])
        self.assertLess(side["redundancy"], same["redundancy"])

    def test_information_score_uses_requested_weights_and_penalties(self):
        center = np.zeros(3, dtype=float)
        target_points = np.array(
            [
                [0.04, 0.00, 0.00],
                [-0.04, 0.00, 0.00],
                [0.00, 0.02, 0.00],
                [0.00, -0.02, 0.00],
                [0.00, 0.00, 0.02],
            ],
            dtype=float,
        )
        target_normals = np.array(
            [
                [1.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, -1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=float,
        )
        current_camera = np.array([0.0, 0.0, 0.30], dtype=float)
        args = nbv_args()
        proxy = runner._build_active_information_proxy(
            target_points,
            target_normals,
            center,
            current_camera,
            args,
        )
        candidate_camera = runner._active_spherical_camera_position(
            center, 0.30, np.deg2rad(45.0), 35.0
        )

        result = information_result(proxy, candidate_camera, args)

        expected = (
            0.35 * result["surface_gain"]
            + 0.30 * result["contact_gain"]
            + 0.15 * result["sweep_gain"]
            + 0.10 * result["occlusion_gain"]
            + 0.10 * result["image_quality"]
            - 0.10 * result["redundancy"]
            - 0.10 * result["motion_cost"]
        )
        self.assertAlmostEqual(result["score"], expected, places=10)

        blocked = information_result(
            proxy,
            candidate_camera,
            args,
            los_penalty=0.8,
            direction_collision=0.8,
        )
        costly = information_result(
            proxy,
            candidate_camera,
            args,
            joint_max_delta_deg=45.0,
            joint_l2_delta_deg=75.0,
        )
        self.assertGreater(result["occlusion_gain"], blocked["occlusion_gain"])
        self.assertGreater(result["score"], blocked["score"])
        self.assertGreater(costly["motion_cost"], result["motion_cost"])
        self.assertLess(costly["score"], result["score"])


if __name__ == "__main__":
    unittest.main()
