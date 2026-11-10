"""Offline checks for the supervised image-point out-and-back workflow."""

import time
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from tools import cross_point_check as cross


def transform(xyz=(-0.4, 0.0, 0.3), rpy=(0.0, 0.0, 0.0)):
    return cross.pose_transform(np.r_[xyz, rpy])


class CartesianSolver:
    """Synthetic, hardware-free kinematics with exactly inspectable paths."""

    def __init__(self):
        self.calls = []

    def get_joint_limits(self):
        return np.full(7, -3.0), np.full(7, 3.0)

    def solve_ik(self, target_position, target_wxyz, initial_guess,
                 use_previous_solution):
        assert use_previous_solution is False
        self.calls.append((np.asarray(target_position).copy(),
                           np.asarray(initial_guess).copy()))
        q = np.asarray(initial_guess, dtype=float).copy()
        q[:3] = target_position
        q[3:6] = Rotation.from_quat(
            np.asarray(target_wxyz)[[1, 2, 3, 0]]
        ).as_euler("xyz")
        return q


def cartesian_fk(q):
    q = np.asarray(q, dtype=float)
    return transform(q[:3], q[3:6])


def args_for_test():
    return SimpleNamespace(
        execute=True, tcp_offset=[0.0] * 6, hover=0.03,
        cross_step_mm=1.0, nero_speed_percent=1,
        min_x=-0.8, max_x=0.8, min_y=-0.6, max_y=0.6,
        min_z=0.02, max_z=1.2,
    )


def test_roundtrip_solves_only_two_endpoints_and_uses_exact_reverse():
    args = args_for_test()
    q_start = np.array([-0.4, 0.0, 0.25, 0.1, -0.2, 0.3, 0.4])
    offset = transform((0.02, 0.0, -0.01))
    start = cartesian_fk(q_start) @ offset
    point = start[:3, 3] + [0.025, 0.015, -0.04]
    solver = CartesianSolver()
    plan = cross.plan_cross_roundtrip(
        start, point, q_start, offset, solver, cartesian_fk, args,
    )

    np.testing.assert_allclose(plan["start_tcp"], start)
    np.testing.assert_array_equal(plan["start_joints"], q_start)
    hover = point + [0.0, 0.0, args.hover]
    np.testing.assert_allclose(plan["to_hover"][-1][1][:3, 3], hover)
    np.testing.assert_allclose(plan["approach"][-1][1][:3, 3], point)
    assert len(solver.calls) == 2
    assert len(plan["to_hover"]) == len(plan["approach"]) == 1
    assert len(plan["return"]) == 2
    assert len(plan["return_from_hover"]) == 1
    for _, tcp in plan["to_hover"] + plan["approach"] + plan["return"]:
        np.testing.assert_allclose(tcp[:3, :3], start[:3, :3], atol=1e-12)
    outbound = plan["to_hover"] + plan["approach"]
    expected_reverse = list(reversed([(q_start, start)] + outbound[:-1]))
    assert len(plan["return"]) == len(expected_reverse)
    for (q, tcp), (expected_q, expected_tcp) in zip(plan["return"], expected_reverse):
        np.testing.assert_array_equal(q, expected_q)
        np.testing.assert_array_equal(tcp, expected_tcp)
    np.testing.assert_array_equal(plan["return"][-1][0], q_start)
    np.testing.assert_allclose(plan["return_from_hover"][-1][1], start)


def test_roundtrip_densely_checks_joint_interpolation_not_cartesian_ik_nodes():
    args = args_for_test()
    q_start = np.array([-0.4, 0.0, 0.25, 0, 0, 0, 0])
    start = cartesian_fk(q_start)
    point = np.array([-0.36, 0.02, 0.27])
    checked = []

    def checked_fk(q):
        checked.append(np.asarray(q).copy())
        return cartesian_fk(q)

    solver = CartesianSolver()
    plan = cross.plan_cross_roundtrip(
        start, point, q_start, np.eye(4), solver, checked_fk, args,
    )
    assert len(solver.calls) == 2
    assert len(checked) > 10
    assert any(q_start[0] < q[0] < plan["to_hover"][0][0][0]
               for q in checked)
    np.testing.assert_allclose(solver.calls[1][1], plan["to_hover"][0][0])


def test_roundtrip_distance_budget_includes_every_outbound_leg():
    args = args_for_test()
    q_start = np.array([-0.4, 0.0, 0.25, 0, 0, 0, 0])
    start, point = cartesian_fk(q_start), np.array([0.2, 0.0, 0.25])
    with pytest.raises(ValueError):
        cross.plan_cross_roundtrip(
            start, point, q_start, np.eye(4), CartesianSolver(), cartesian_fk, args,
        )


def test_smaller_sampling_step_does_not_add_execution_commands():
    args = args_for_test()
    q_start = np.array([-0.4, 0.0, 0.25, 0, 0, 0, 0])
    start, point = cartesian_fk(q_start), np.array([-0.39, 0.0, 0.2])
    counts = []
    for step in (1.0, 0.5):
        args.cross_step_mm = step
        checked = []

        def checked_fk(q):
            checked.append(np.asarray(q).copy())
            return cartesian_fk(q)

        solver = CartesianSolver()
        plan = cross.plan_cross_roundtrip(
            start, point, q_start, np.eye(4), solver, checked_fk, args,
        )
        assert len(solver.calls) == 2
        assert len(plan["to_hover"]) == len(plan["approach"]) == 1
        assert len(plan["return"]) == 2
        counts.append(len(checked))
    assert counts[1] > counts[0]


def test_roundtrip_rejects_out_of_workspace_joint_interpolation_interior():
    args = args_for_test()
    q_start = np.array([-0.4, 0.0, 0.25, 0, 0, 0, 0])
    point = np.array([-0.36, 0.0, 0.2])

    def curved_fk(q):
        result = cartesian_fk(q)
        fraction = (q[0] + 0.4) / 0.04
        if 0 < fraction < 1:
            result[2, 3] -= 0.4 * np.sin(np.pi * fraction)
        return result

    with pytest.raises(ValueError):
        cross.plan_cross_roundtrip(
            cartesian_fk(q_start), point, q_start, np.eye(4),
            CartesianSolver(), curved_fk, args,
        )


@pytest.mark.parametrize("case", ["distance", "workspace", "ik", "fk", "joint_jump"])
def test_roundtrip_preflight_rejects_invalid_full_route(case):
    args = args_for_test()
    q_start = np.array([-0.4, 0.0, 0.25, 0, 0, 0, 0])
    start, point = cartesian_fk(q_start), np.array([-0.39, 0.0, 0.2])
    solver, fk = CartesianSolver(), cartesian_fk
    if case == "distance":
        point = np.array([0.2, 0.0, 0.2])
    elif case == "workspace":
        point[2] = -0.01
    elif case == "ik":
        solver.solve_ik = lambda **_: None
    elif case == "fk":
        fk = lambda _: transform((0, 0, 0))
    else:
        solver.solve_ik = lambda **_: q_start + 1.0
    with pytest.raises(ValueError):
        cross.plan_cross_roundtrip(start, point, q_start, np.eye(4), solver, fk, args)


class FakeRoundtripArm:
    def __init__(self):
        self.q = np.array([-0.4, 0.0, 0.25, 0, 0, 0, 0], dtype=float)
        self.commands, self.stops = [], []
        self.arm = SimpleNamespace(get_flange_pose=self._pose)

    def _pose(self):
        return SimpleNamespace(msg=self.q[:6].copy(), timestamp=time.time())

    def get_joint_angles(self):
        return self.q.copy()

    def _controller_motion_active(self):
        return False

    def get_joint_torques(self):
        raise AssertionError("Point moves must not request joint torque feedback")

    def move_j(self, q):
        self.commands.append(np.asarray(q).copy())
        self.q = np.asarray(q).copy()

    def stop_for_program_exit(self, reason):
        self.stops.append(reason)


@pytest.fixture
def roundtrip_case(tmp_path, monkeypatch):
    arm, args = FakeRoundtripArm(), args_for_test()
    point = np.array([-0.39, 0.0, 0.2])
    state = dict(
        observation_joints_rad=arm.q.copy(), flange_pose=arm.q[:6].copy(),
        P_base=point.copy(), target_pose=np.r_[point, [0, 0, 0]],
        target_tcp_pose=np.r_[point + [0, 0, args.hover], [0, 0, 0]],
        out_dir=str(tmp_path), cross_kinematics=(CartesianSolver(), cartesian_fk),
    )
    monkeypatch.setattr(cross.time, "sleep", lambda _: None)
    return arm, args, state


def test_roundtrip_success_records_point_before_exact_return(roundtrip_case, monkeypatch):
    arm, args, state = roundtrip_case
    origin = arm.q.copy()
    replies = iter(["MOVE", "APPROACH"])
    monkeypatch.setattr("builtins.input", lambda _: next(replies))
    assert cross.execute_cross_roundtrip(arm, args, state) is True
    assert arm.commands and not arm.stops
    assert len(arm.commands) == 4
    np.testing.assert_array_equal(arm.commands[0], arm.commands[2])
    np.testing.assert_array_equal(arm.commands[3], origin)
    np.testing.assert_array_equal(arm.q, origin)
    record = state["cross_last_execution"]
    assert record["purpose"] == "visual_point"
    np.testing.assert_allclose(record["actual_tcp_xyz_m"], state["P_base"])
    assert state["cross_roundtrip"]["status"] == "completed"
    assert state["cross_roundtrip"]["control_mode"] == "move_j"
    assert state["cross_roundtrip"]["software_torque_guard"] is False
    assert [sample["phase"] for sample in state["cross_roundtrip"]["samples"]] == [
        "to_hover", "approach", "return", "return",
    ]
    np.testing.assert_allclose(
        state["cross_roundtrip"]["point_execution"]["actual_tcp_xyz_m"],
        state["P_base"],
    )
    assert any(np.allclose(command[:3], state["P_base"]) for command in arm.commands)


def test_declining_roundtrip_never_moves(roundtrip_case, monkeypatch):
    arm, args, state = roundtrip_case
    monkeypatch.setattr("builtins.input", lambda _: "")
    assert cross.execute_cross_roundtrip(arm, args, state) is False
    assert not arm.commands and not arm.stops


def test_declining_approach_returns_from_hover_without_descending(roundtrip_case, monkeypatch):
    arm, args, state = roundtrip_case
    origin = arm.q.copy()
    replies = iter(["MOVE", ""])
    monkeypatch.setattr("builtins.input", lambda _: next(replies))
    assert cross.execute_cross_roundtrip(arm, args, state) is False
    assert arm.commands and not arm.stops
    assert len(arm.commands) == 2
    np.testing.assert_array_equal(arm.q, origin)
    hover_height = state["P_base"][2] + args.hover
    assert min(command[2] for command in arm.commands) >= hover_height - 1e-10
    assert not any(np.allclose(command[:3], state["P_base"]) for command in arm.commands)
    result = state["cross_roundtrip"]
    assert result["status"] == "cancelled_at_hover_returned"
    assert 0 <= result["return_joint_error_deg"] <= 0.2
    assert 0 <= result["return_tcp_error_mm"] <= 1.0


@pytest.mark.parametrize("reply", ["APPROACH", ""])
@pytest.mark.parametrize("joint_index", [0, 6])
def test_operator_motion_during_hover_prompt_forbids_all_further_motion(
    roundtrip_case, monkeypatch, reply, joint_index,
):
    arm, args, state = roundtrip_case
    commands_at_hover = []
    prompts = []

    def answer(prompt):
        prompts.append(prompt)
        if len(prompts) == 1:
            return "MOVE"
        assert len(prompts) == 2
        commands_at_hover.append(len(arm.commands))
        assert commands_at_hover[-1] > 0
        arm.q[joint_index] += np.deg2rad(0.3)
        return reply

    monkeypatch.setattr("builtins.input", answer)
    with pytest.raises(RuntimeError):
        cross.execute_cross_roundtrip(arm, args, state)
    assert len(arm.commands) == commands_at_hover[0]
    assert len(arm.stops) == 1
    assert state["cross_roundtrip"]["status"] == "failed"
    assert not state.get("cross_last_execution")


def test_operator_motion_during_first_confirmation_sends_no_commands(
    roundtrip_case, monkeypatch,
):
    arm, args, state = roundtrip_case

    def answer(_):
        arm.q[6] += np.deg2rad(0.3)
        return "MOVE"

    monkeypatch.setattr("builtins.input", answer)
    with pytest.raises(RuntimeError):
        cross.execute_cross_roundtrip(arm, args, state)
    assert not arm.commands and not arm.stops


def test_stale_raw_joint_wrapper_refuses_motion(roundtrip_case, monkeypatch):
    arm, args, state = roundtrip_case
    arm.arm.get_joint_angles = lambda: SimpleNamespace(
        msg=arm.q.copy(), timestamp=time.time() - 2.0,
    )
    monkeypatch.setattr("builtins.input", lambda _: "MOVE")
    with pytest.raises(RuntimeError):
        cross.execute_cross_roundtrip(arm, args, state)
    assert not arm.commands and not arm.stops


def test_roundtrip_does_not_request_sdk_motor_feedback(roundtrip_case, monkeypatch):
    arm, args, state = roundtrip_case

    def forbidden_motor_read(_):
        raise AssertionError("Point moves must not request SDK motor feedback")

    arm.arm.get_motor_states = forbidden_motor_read
    replies = iter(["MOVE", "APPROACH"])
    monkeypatch.setattr("builtins.input", lambda _: next(replies))
    assert cross.execute_cross_roundtrip(arm, args, state) is True
    assert arm.commands and not arm.stops
    assert state["cross_roundtrip"]["status"] == "completed"


def test_changed_observation_joints_refuse_roundtrip(roundtrip_case, monkeypatch):
    arm, args, state = roundtrip_case
    arm.q[6] += np.deg2rad(0.3)
    monkeypatch.setattr("builtins.input", lambda _: "MOVE")
    with pytest.raises((ValueError, RuntimeError)):
        cross.execute_cross_roundtrip(arm, args, state)
    assert not arm.commands


def test_teaching_mode_refuses_roundtrip(roundtrip_case, monkeypatch):
    arm, args, state = roundtrip_case
    arm.arm.get_arm_status = lambda: SimpleNamespace(
        msg=SimpleNamespace(ctrl_mode="TEACHING_MODE")
    )
    monkeypatch.setattr("builtins.input", lambda _: "MOVE")
    with pytest.raises((ValueError, RuntimeError)):
        cross.execute_cross_roundtrip(arm, args, state)
    assert not arm.commands


def test_roundtrip_stale_pose_stops_and_never_blindly_returns(roundtrip_case, monkeypatch):
    arm, args, state = roundtrip_case

    def pose():
        age = 2.0 if arm.commands else 0.0
        return SimpleNamespace(msg=arm.q[:6].copy(), timestamp=time.time()-age)

    arm.arm.get_flange_pose = pose
    monkeypatch.setattr("builtins.input", lambda _: "MOVE")
    with pytest.raises(RuntimeError, match="fresh"):
        cross.execute_cross_roundtrip(arm, args, state)
    assert len(arm.commands) == 1
    assert len(arm.stops) == 1
    assert state["cross_roundtrip"]["status"] == "failed"
    assert not state.get("cross_last_execution")


def test_latched_stop_refuses_roundtrip_before_motion(roundtrip_case):
    arm, args, state = roundtrip_case
    arm._safety_stop_latched = True
    with pytest.raises(RuntimeError, match="锁存"):
        cross.execute_cross_roundtrip(arm, args, state)
    assert not arm.commands and not arm.stops


def test_new_stop_latch_stops_roundtrip_without_return(roundtrip_case, monkeypatch):
    arm, args, state = roundtrip_case
    original_move = arm.move_j

    def move_and_latch(q):
        original_move(q)
        arm._safety_stop_latched = True

    arm.move_j = move_and_latch
    monkeypatch.setattr("builtins.input", lambda _: "MOVE")
    with pytest.raises(RuntimeError, match="锁存"):
        cross.execute_cross_roundtrip(arm, args, state)
    assert len(arm.commands) == 1
    assert len(arm.stops) == 1
    assert state["cross_roundtrip"]["status"] == "failed"


def test_invalid_full_route_sends_no_commands(roundtrip_case, monkeypatch):
    arm, args, state = roundtrip_case
    state["P_base"] = np.array([0.2, 0.0, 0.2])
    monkeypatch.setattr("builtins.input", lambda _: "MOVE")
    with pytest.raises(ValueError):
        cross.execute_cross_roundtrip(arm, args, state)
    assert not arm.commands and not arm.stops
