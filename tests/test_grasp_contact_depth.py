"""Reject apparent bilateral support that only grazes the finger tips."""

from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from real_nero_grasp.grasp_candidates import annotate_grasp_candidate_physical_aperture


def evaluate(left_along, right_along, *, advance=0.0, transform=None, extra=None):
    args = SimpleNamespace(
        gripper_open_width=0.100, gripper_hardware_max_width=0.100,
        grasp_physical_min_contact_inset=0.010,
        motion_path_open_finger_front=0.005,
        motion_path_open_finger_back=0.065,
        motion_path_open_finger_radius=0.007,
        topdown_grasp=False, use_current_rpy=False, fixed_rpy=None,
        graspnet_approach_axis="x", graspnet_width_axis="y",
        tool_approach_axis="x", tcp_gripper_width_axis="y",
        use_graspnet_depth_advance=False, grasp_advance=advance,
        min_grasp_z=None, pregrasp_offset=0.10, lift_height=0.07,
    )
    grasp = SimpleNamespace(
        width=0.080, height=0.020, depth=0.030,
        rotation_matrix=np.eye(3), translation=np.zeros(3),
    )
    across = np.linspace(-0.030, 0.030, 200)
    points = np.column_stack([
        np.where(across < 0, left_along, right_along), across, np.zeros(200)
    ])
    if extra is not None:
        points = np.vstack([points, extra])
    return annotate_grasp_candidate_physical_aperture(
        [{"idx": 0, "grasp": grasp}], SimpleNamespace(points=points),
        np.eye(4) if transform is None else transform, np.zeros(6), args,
    )[0]


@pytest.mark.parametrize("left,right", [(0.002, 0.002), (-0.015, 0.002)])
def test_tip_only_support_on_either_side_is_rejected(left, right):
    row = evaluate(left, right)
    assert not row["physical_aperture_ok"]
    assert "shallow bilateral contact" in row["physical_aperture_reason"]
    assert row["physical_aperture_variants"][0]["right_contact_inset"] == pytest.approx(.003)


def test_sufficient_depth_on_both_sides_is_accepted():
    assert evaluate(-0.015, -0.015)["physical_aperture_ok"]


def test_advancing_requires_contact_and_open_finger_checks_again():
    # The same surface becomes usable after advancing the execution TCP.
    assert not evaluate(.002, .002)["physical_aperture_ok"]
    assert evaluate(.002, .002, advance=.015)["physical_aperture_ok"]
    row = evaluate(-.015, -.015, extra=[[-.015, .0565, 0.]])
    assert not row["physical_aperture_ok"]
    assert "penetrates an open finger capsule" in row["physical_aperture_reason"]


def test_contact_depth_follows_final_tcp_frame():
    transform = np.eye(4)
    transform[:3, :3] = Rotation.from_euler("xyz", [.4, -.3, .8]).as_matrix()
    transform[:3, 3] = [.2, -.5, .4]
    for left, right in [(.002, .002), (-.015, -.015)]:
        local = evaluate(left, right)
        moved = evaluate(left, right, transform=transform)
        assert moved["physical_aperture_ok"] == local["physical_aperture_ok"]
        assert moved["physical_aperture_variants"][0]["left_contact_inset"] == pytest.approx(
            local["physical_aperture_variants"][0]["left_contact_inset"]
        )
