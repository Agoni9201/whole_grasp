"""Logic tests for the mit-impedance engage / target-refresh / freeze behavior
in the visual-servo execute node (Step 1 of the integration).

These bypass the heavy ROS Node.__init__ (which connects to real hardware) by
building a bare instance via __new__ and setting only the attributes the
impedance methods touch. No hardware, no ROS spin.
"""

import os
import sys
import logging
import time
from types import SimpleNamespace

import numpy as np
from scipy.spatial.transform import Rotation as Rot

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

import aruco_camera_jacobian_servo_sdk_execute as execute_module  # noqa: E402
from control_backends.force_ctrl.controller import ZERO_GRAV_KD  # noqa: E402
from control_backends.force_ctrl.impedance_profile import (  # noqa: E402
    VALIDATED_CARTESIAN_KD,
    VALIDATED_CARTESIAN_KP,
    VALIDATED_GRAVITY_FACTORS,
    VALIDATED_TASK_TORQUE_LIMIT,
    VALIDATED_TOTAL_TORQUE_LIMIT,
)
from aruco_camera_jacobian_servo_sdk_execute import (  # noqa: E402
    ArucoCameraServoExecuteNode,
    apply_orientation_target_anti_windup,
    apply_position_target_anti_windup,
    constrain_impedance_target_twist,
    effective_startup_limit_margin,
    effective_task_torque_limit,
    image_is_fresh,
    integrate_orientation_target_from_twist,
    parse_args,
)
from servo.execution_safety import ExecutionSafetyMonitor, SafetyParams  # noqa: E402


class _FakeForceCtrl:
    """Records emergency_stop so the regression test can assert it is NOT used."""
    def __init__(self):
        self.estop_called = False

    def emergency_stop(self):
        self.estop_called = True


class _FakeBackend:
    """Records set_cartesian_impedance calls so we can assert engage/refresh."""
    def __init__(self):
        self.cfgs = []
        self.sent = []
        self.disabled = False
        self.debug = {}
        self.raise_set = False
        self.raise_disable = False
        self.raise_send = False
        self.disconnected = False
        self._force_ctrl = _FakeForceCtrl()

    def set_cartesian_impedance(self, cfg):
        if self.raise_set:
            raise RuntimeError("set impedance failed")
        self.cfgs.append(cfg)

    def disable_cartesian_impedance(self):
        if self.raise_disable:
            raise RuntimeError("disable failed")
        self.disabled = True

    def send(self, cmd):
        if self.raise_send:
            raise RuntimeError("send failed")
        self.sent.append(cmd)

    def get_debug_state(self, include_trace=False):
        return self.debug

    def emergency_stop(self):
        self._force_ctrl.emergency_stop()

    def disconnect(self):
        self.disconnected = True


class _FakePin:
    """Stand-in pin model: fk returns a fixed TCP + rotation (deterministic)."""
    def __init__(self, tcp, R):
        self._tcp = np.asarray(tcp, dtype=np.float64).ravel()
        self._R = np.asarray(R, dtype=np.float64)

    def fk(self, q):
        return self._tcp.copy(), self._R.copy()


_FAKE_TCP = np.array([0.40, 0.0, 0.30])
_FAKE_R = np.eye(3)


def _bare_node():
    n = ArucoCameraServoExecuteNode.__new__(ArucoCameraServoExecuteNode)
    # Node.__init__ is bypassed (it connects hardware), so stub the logger the
    # engage path uses (self.get_logger().info(...)).
    n.get_logger = lambda: logging.getLogger("test_execute_impedance")
    n.backend = _FakeBackend()
    n.reader = None
    n._imp_engaged = False
    n._imp_cfg = None
    n._imp_smoothed_tcp = None
    n._imp_smoothed_R = None
    n._imp_fixed_R = None
    n._imp_target_update_time = None
    n._imp_engage_monotonic = None
    n._imp_engage_cycle_baseline = None
    n._imp_track_start_q = None
    n._impedance_fallback_triggered = False
    n._impedance_fallback_reason = ""
    n._impedance_terminal_fault = False
    n._imp_model = _FakePin(_FAKE_TCP, _FAKE_R)
    n._hold_command = lambda q, t, prefer_last=True: ("HOLD", q, t)
    n.params = SimpleNamespace(dt=1.0 / 30.0)
    n.args = SimpleNamespace(
        control_mode="mit-impedance",
        kd_scale=1.0,
        impedance_dls_lambda=0.05,
        impedance_vel_filter_tau=0.008,
        impedance_target_linear_vel=0.02,
        impedance_max_position_error=0.015,
        max_joint_vel=0.10,
        impedance_target_angular_vel=0.30,
        impedance_visual_orientation=False,
        impedance_max_orientation_error_deg=5.0,
        impedance_safe_margin=0.30,
        settle_max_drift_deg=2.0,
        execution_backend="mit-position",
        settle_s=2.0,
        log_period=999.0,
    )
    n.imp_use_dls = False
    n.imp_coulomb = np.zeros(7)
    n.imp_joint_weight = np.ones(7)
    n.imp_task_torque_limit = 10.0
    n.imp_tau_limit_per_joint = np.full(7, 14.0)
    n.imp_velocity_signs = np.array([-1.0, -1.0, -1.0, -1.0, -1.0, 1.0, -1.0])
    n.imp_use_real_velocity = True
    n.imp_real_vel_filter_tau = 0.004
    n.imp_qdot_trip_fast = 4.0
    n.imp_joint_hold_kp = np.zeros(7)
    n.imp_joint_hold_kd = None
    n.imp_follow_kp = np.array([8.0, 10.0, 8.0, 12.0, 4.0, 2.0, 2.0])
    n.imp_follow_min_qdot = 0.04
    n.imp_follow_release_error = 0.03
    n.cart_kp = np.array([200.0, 200.0, 200.0, 5.0, 5.0, 5.0])
    n.cart_kd = np.array([5.0, 5.0, 5.0, 0.2, 0.2, 0.2])
    n.zero_grav_kd = ZERO_GRAV_KD.copy()
    n.settle_started = None
    n.settle_q_ref = None
    n.phase = "preflight"
    n.last_log_time = 0.0
    n.count = 0
    n.display_initialized = False
    n.last_image = None
    n.last_image_q = None
    n.last_image_time = None
    n.last_image_warn_time = 0.0
    n.last_detect_time = None
    n._image_generation = 0
    n._detected_image_generation = -1
    n._cached_image_target = None
    n.safety = ExecutionSafetyMonitor(
        SafetyParams(startup_limit_margin=0.30))
    return n


def _servo_out(tcp, R=None, tcp_twist=None):
    if R is None:
        R = np.eye(3)
    out = {
        "target_tcp": np.asarray(tcp, dtype=np.float64),
        "target_rotation": np.asarray(R, dtype=np.float64),
        "error": np.zeros(3),
        "orientation_error": np.zeros(3),
        "qdot": np.zeros(7),
    }
    if tcp_twist is not None:
        out["tcp_twist"] = np.asarray(tcp_twist, dtype=np.float64)
    return out


def test_engage_installs_cfg_at_current_pose_zero_error():
    # Engage target = CURRENT TCP (from fk), NOT the visual target -> zero
    # initial error so the stiff spring does not yank the arm on cycle 1.
    n = _bare_node()
    q = np.zeros(7)
    cmd = n._build_impedance_command(q, _servo_out([0.50, 0.0, 0.40]), now=1.0)
    assert n._imp_engaged is True
    assert len(n.backend.cfgs) == 1
    assert np.allclose(n._imp_cfg.target_tcp, _FAKE_TCP)   # current pose, not visual
    assert np.allclose(n._imp_cfg.R_target, _FAKE_R)
    assert np.allclose(n._imp_track_start_q, q)
    assert np.allclose(cmd.torque_ff, 0.0)                 # benign buffered cmd
    assert cmd.mode == "mit"


def test_target_twist_constraint_caps_linear_and_angular_norms():
    twist = np.array([0.06, 0.08, 0.0, 0.30, 0.40, 0.0])
    constrained = constrain_impedance_target_twist(
        twist, max_linear_vel=0.05, max_angular_vel=0.20,
    )
    assert np.isclose(np.linalg.norm(constrained[:3]), 0.05)
    assert np.isclose(np.linalg.norm(constrained[3:]), 0.20)


def test_target_twist_constraint_removes_angular_feedforward_when_disabled():
    twist = np.array([0.02, 0.0, 0.0, 0.20, 0.0, 0.0])
    constrained = constrain_impedance_target_twist(
        twist, max_linear_vel=0.05, max_angular_vel=0.20,
        include_orientation=False,
    )
    assert np.allclose(constrained[:3], twist[:3])
    assert np.allclose(constrained[3:], 0.0)


def test_subsequent_cycle_rate_limits_target_toward_visual():
    n = _bare_node()
    q = np.zeros(7)
    n._build_impedance_command(q, _servo_out([0.50, 0.0, 0.40]), now=1.0)  # engage at _FAKE_TCP
    cfg_first = n._imp_cfg
    # Visual target is ~0.117 m away; the impedance equilibrium has its own
    # conservative cap: 0.02/30 m per cycle.
    n._build_impedance_command(q, _servo_out([0.50, 0.0, 0.40]), now=1.0 + 1.0 / 30)
    assert cfg_first is n._imp_cfg                          # same cfg, target mutated
    advanced = float(np.linalg.norm(n._imp_cfg.target_tcp - _FAKE_TCP))
    assert advanced <= 0.02 / 30.0 + 1e-9                   # rate-limited
    assert advanced > 0.0                                   # moving toward visual
    assert len(n.backend.cfgs) == 2


def test_target_rate_limit_uses_elapsed_time_without_reacquisition_catchup():
    n = _bare_node()
    q = np.zeros(7)
    visual = _FAKE_TCP + np.array([0.10, 0.0, 0.0])
    n._build_impedance_command(q, _servo_out(visual), now=1.0)

    # A callback arriving faster than nominal may advance only by v * real dt.
    n._build_impedance_command(q, _servo_out(visual), now=1.005)
    first_advance = float(np.linalg.norm(n._imp_cfg.target_tcp - _FAKE_TCP))
    assert first_advance <= 0.02 * 0.005 + 1e-9

    # A long pause (for example marker loss) must not produce a catch-up jump.
    before = n._imp_cfg.target_tcp.copy()
    n._build_impedance_command(q, _servo_out(visual), now=2.005)
    late_advance = float(np.linalg.norm(n._imp_cfg.target_tcp - before))
    assert late_advance <= 0.02 / 30.0 + 1e-9


def test_pbvs_linear_twist_drives_impedance_target_direction():
    n = _bare_node()
    q = np.zeros(7)
    visual = _FAKE_TCP + np.array([0.10, 0.0, 0.10])
    pbvs_twist = np.array([0.02, 0.0, 0.0, 0.0, 0.0, 0.0])
    servo_out = _servo_out(visual, tcp_twist=pbvs_twist)

    n._build_impedance_command(q, servo_out, now=1.0)
    n._build_impedance_command(q, servo_out, now=1.0 + 1.0 / 30.0)

    advance = n._imp_cfg.target_tcp - _FAKE_TCP
    assert np.isclose(advance[0], 0.02 / 30.0)
    assert np.allclose(advance[1:], 0.0, atol=1e-12)


def test_hybrid_follow_reference_integrates_pbvs_qdot():
    n = _bare_node()
    q = np.zeros(7)
    servo_out = _servo_out(
        _FAKE_TCP,
        tcp_twist=np.zeros(6),
    )
    servo_out["qdot"] = np.array([0.06, -0.03, 0.0, 0.02, 0.0, 0.0, -0.01])

    n._build_impedance_command(q, servo_out, now=1.0)
    n._build_impedance_command(q, servo_out, now=1.0 + 1.0 / 30.0)

    expected = servo_out["qdot"] / 30.0
    assert np.allclose(n._imp_cfg.q_follow, expected)
    expected_follow_kp = n.imp_follow_kp * np.clip(
        np.abs(servo_out["qdot"]) / n.imp_follow_min_qdot, 0.0, 1.0
    )
    assert np.allclose(n._imp_cfg.follow_kp, expected_follow_kp)
    assert n._imp_cfg.follow_max_error == n.args.max_joint_vel


def test_visual_impedance_passes_pbvs_twist_to_cartesian_damper():
    """Tracking uses velocity feed-forward without enabling a joint spring."""
    n = _bare_node()
    q = np.zeros(7)
    tcp_twist = np.array([0.03, -0.01, 0.02, 0.20, -0.10, 0.05])
    servo_out = _servo_out(_FAKE_TCP, tcp_twist=tcp_twist)

    n._build_impedance_command(q, servo_out, now=1.0)
    # The engage handoff is still a fixed-target hold.
    assert np.allclose(n._imp_cfg.target_twist, 0.0)
    n._build_impedance_command(q, servo_out, now=1.0 + 1.0 / 30.0)

    # This fixture disables visual orientation, so only the translational PBVS
    # velocity is fed to the Cartesian damper.
    assert np.allclose(n._imp_cfg.target_twist[:3], tcp_twist[:3])
    assert np.allclose(n._imp_cfg.target_twist[3:], 0.0)


def test_marker_hold_clears_visual_target_twist_with_follow_kp_zero():
    """A lost marker must not leave the last moving damper command active."""
    n = _bare_node()
    q = np.zeros(7)
    n._build_impedance_command(q, _servo_out(_FAKE_TCP), now=1.0)
    n._imp_cfg.follow_kp = np.zeros(7)
    n._imp_cfg.target_twist = np.array([0.02, 0.0, 0.0, 0.0, 0.0, 0.0])

    n._release_impedance_follow_spring()

    assert np.allclose(n._imp_cfg.target_twist, 0.0)
    assert np.allclose(n.backend.cfgs[-1].target_twist, 0.0)


def test_hybrid_follow_kp_releases_at_zero_visual_speed():
    n = _bare_node()
    q = np.zeros(7)
    servo_out = _servo_out(_FAKE_TCP, tcp_twist=np.zeros(6))

    n._build_impedance_command(q, servo_out, now=1.0)
    n._build_impedance_command(q, servo_out, now=1.0 + 1.0 / 30.0)

    assert np.allclose(n._imp_cfg.follow_kp, 0.0)


def test_hybrid_follow_kp_releases_when_operator_runs_ahead_of_reference():
    n = _bare_node()
    q = np.zeros(7)
    servo_out = _servo_out(_FAKE_TCP, tcp_twist=np.zeros(6))
    servo_out["qdot"] = np.array([0.08, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    n._build_impedance_command(q, servo_out, now=1.0)
    n._build_impedance_command(q, servo_out, now=1.0 + 1.0 / 30.0)
    assert n._imp_cfg.follow_kp[0] == n.imp_follow_kp[0]

    # q is now ahead of q_follow while PBVS still points in +joint1.  The
    # position spring must yield so the Cartesian task remains draggable.
    n._build_impedance_command(
        np.array([0.10, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        servo_out,
        now=1.0 + 2.0 / 30.0,
    )
    assert n._imp_cfg.follow_kp[0] == 0.0


def test_marker_hold_releases_stale_follow_spring_but_keeps_impedance_target():
    n = _bare_node()
    q = np.zeros(7)
    servo_out = _servo_out(_FAKE_TCP, tcp_twist=np.zeros(6))
    servo_out["qdot"] = np.full(7, 0.08)

    n._build_impedance_command(q, servo_out, now=1.0)
    n._build_impedance_command(q, servo_out, now=1.0 + 1.0 / 30.0)
    target_before = n._imp_cfg.target_tcp.copy()
    assert np.any(n._imp_cfg.follow_kp > 0.0)

    n._release_impedance_follow_spring()

    assert np.allclose(n._imp_cfg.follow_kp, 0.0)
    assert np.allclose(n._imp_cfg.target_tcp, target_before)


def test_pbvs_angular_twist_drives_impedance_target_axis():
    n = _bare_node()
    n.args.impedance_visual_orientation = True
    q = np.zeros(7)
    visual_R = Rot.from_euler("z", 30.0, degrees=True).as_matrix()
    pbvs_twist = np.array([0.0, 0.0, 0.0, 0.12, 0.0, 0.0])
    servo_out = _servo_out(_FAKE_TCP, visual_R, tcp_twist=pbvs_twist)

    n._build_impedance_command(q, servo_out, now=1.0)
    n._build_impedance_command(q, servo_out, now=1.0 + 1.0 / 30.0)

    rotvec = Rot.from_matrix(n._imp_cfg.R_target).as_rotvec()
    assert np.isclose(rotvec[0], 0.12 / 30.0)
    assert np.allclose(rotvec[1:], 0.0, atol=1e-12)


def test_world_angular_twist_left_multiplies_nonidentity_rotation():
    current = Rot.from_euler("x", 90.0, degrees=True).as_matrix()
    omega = np.array([0.0, 0.0, 0.12])
    dt = 1.0 / 30.0

    actual = integrate_orientation_target_from_twist(
        current, omega, dt, max_angular_vel=1.0
    )
    delta = Rot.from_rotvec(omega * dt).as_matrix()

    assert np.allclose(actual, delta @ current)
    assert not np.allclose(actual, current @ delta)


def test_target_converges_to_visual_after_many_cycles():
    # The rate-limited target must actually reach the visual target (not just creep).
    n = _bare_node()
    q = np.zeros(7)
    visual = np.array([_FAKE_TCP[0] + 0.01, 0.0, _FAKE_TCP[2]])  # 1 cm offset
    n._build_impedance_command(q, _servo_out(visual), now=1.0)   # engage at _FAKE_TCP
    for k in range(1, 40):
        n._build_impedance_command(q, _servo_out(visual), now=1.0 + k / 30)
    assert np.allclose(n._imp_cfg.target_tcp, visual, atol=1e-4)


def test_missing_target_falls_back_to_hold_without_engaging():
    n = _bare_node()
    q = np.zeros(7)
    so = {"target_tcp": None, "target_rotation": None}
    cmd = n._build_impedance_command(q, so, now=1.0)
    assert np.allclose(cmd.kp, 0.0)
    assert np.allclose(cmd.kd, ZERO_GRAV_KD)
    assert n._imp_engaged is False   # never engage on a missing target


def test_cfg_carries_tuned_extras():
    # The ported pieces (coulomb / joint_weights / task_torque_limit) reach the
    # 200 Hz loop via the cfg built on engage.
    n = _bare_node()
    n.imp_coulomb = np.array([0.0, 0.4, 0.0, 0.4, 0.0, 0.0, 0.0])
    n.imp_joint_weight = np.array([1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5])
    n.imp_task_torque_limit = 10.0
    n._build_impedance_command(np.zeros(7), _servo_out([0.5, 0.0, 0.4]), now=1.0)
    cfg = n._imp_cfg
    assert np.allclose(cfg.coulomb, [0.0, 0.4, 0.0, 0.4, 0.0, 0.0, 0.0])
    assert np.allclose(cfg.joint_weights, [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5])
    assert cfg.task_torque_limit == 10.0
    assert cfg.use_real_velocity is True
    assert cfg.use_dls_mapping is False  # plain-jtf default
    assert np.allclose(cfg.kd_joint, ZERO_GRAV_KD)
    assert np.allclose(cfg.joint_hold_kp, 0.0)
    assert cfg.joint_hold_kd is None


def test_joint_hold_without_explicit_kd_inherits_firmware_damping():
    n = _bare_node()
    n.imp_joint_hold_kp[-1] = 5.0
    n._build_impedance_command(
        np.zeros(7), _servo_out([0.5, 0.0, 0.4]), now=1.0
    )

    cfg = n._imp_cfg
    assert cfg.joint_hold_kd is None
    # The visual configuration contract is the important regression here:
    # mit_loop interprets None as "leave cfg.kd_joint unchanged".
    assert n.backend.cfgs[-1] is cfg
    assert np.allclose(cfg.kd_joint, ZERO_GRAV_KD)


def test_instability_trip_latches_gravity_damping_safe_hold():
    # A trip keeps the 200 Hz loop and gravity compensation alive, but must not
    # transition automatically to stiff visual position tracking.
    n = _bare_node()
    n._imp_engaged = True
    n._impedance_fallback_triggered = False
    n._imp_track_start_q = np.zeros(7)
    n._qdot_ringbuf = []
    n.args.impedance_fallback_window = 1.0
    n.args.impedance_instant_qdot_cap = 6.0
    n.args.impedance_fallback_oscillation = 0.3
    n.args.impedance_max_excursion = 0.30
    n.args.impedance_safe_margin = 0.30
    n.args.control_mode = "mit-impedance"
    # excursion trip: joint1 drifted 0.5 rad from track-start (> 0.30 max)
    q_now = np.array([0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    n._check_impedance_instability(q_now, now=1.0)
    assert n._impedance_fallback_triggered is True
    assert n.backend.disabled is True                    # graceful disable
    assert n.backend._force_ctrl.estop_called is False   # NO e-stop (no collapse)
    assert n.args.control_mode == "mit-impedance"        # no automatic mode switch
    assert n._imp_engaged is False
    safe_cmd = n.backend.sent[-1]
    assert np.allclose(safe_cmd.kp, 0.0)
    assert np.allclose(safe_cmd.kd, ZERO_GRAV_KD)
    assert np.allclose(safe_cmd.torque_ff, 0.0)


def test_visual_start_relative_excursion_can_be_disabled():
    n = _bare_node()
    n._imp_track_start_q = np.zeros(7)
    n._qdot_ringbuf = []
    n.args.impedance_fallback_window = 1.0
    n.args.impedance_instant_qdot_cap = 6.0
    n.args.impedance_fallback_oscillation = 6.0
    n.args.impedance_max_excursion = 0.0
    n.args.impedance_safe_margin = 0.10

    n._check_impedance_instability(
        np.array([0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), now=1.0
    )

    assert n._impedance_fallback_triggered is False
    assert n.backend.disabled is False


def test_hard_inner_trip_is_terminal_for_visual_run():
    n = _bare_node()
    n._imp_engaged = True
    n.backend.debug = {
        "last_error": "in-loop impedance qdot joint5 selected=4.2",
        "impedance_trip": {
            "trigger_kind": "qdot",
            "trigger_joint": 5,
        },
    }

    assert n._check_inner_impedance_fault() is True
    assert n._impedance_fallback_triggered is True
    assert n._impedance_terminal_fault is True
    assert "hard inner trip" in n._impedance_fallback_reason
    assert n._imp_engaged is False
    assert n.backend.sent == []


def test_unconfirmed_inner_impedance_handoff_is_terminal(monkeypatch):
    n = _bare_node()
    n._imp_engaged = True
    n._imp_engage_monotonic = 10.0
    n._imp_engage_cycle_baseline = 12
    n.backend.debug = {
        "running": True,
        "last_error": None,
        "impedance_active": True,
        "control_path": "buffered",
        "impedance_cycle_count": 12,
    }
    monkeypatch.setattr(execute_module.time, "monotonic", lambda: 10.2)

    assert n._check_inner_impedance_fault() is True
    assert n._impedance_terminal_fault is True
    assert "handoff not confirmed" in n._impedance_fallback_reason
    assert n.backend._force_ctrl.estop_called is True


def test_confirmed_inner_impedance_handoff_remains_active(monkeypatch):
    n = _bare_node()
    n._imp_engaged = True
    n._imp_engage_monotonic = 10.0
    n._imp_engage_cycle_baseline = 12
    n.backend.debug = {
        "running": True,
        "last_error": None,
        "impedance_active": True,
        "control_path": "impedance",
        "impedance_cycle_count": 13,
    }
    monkeypatch.setattr(execute_module.time, "monotonic", lambda: 10.2)

    assert n._check_inner_impedance_fault() is False
    assert n._impedance_terminal_fault is False
    assert n._imp_engaged is True
    assert n.backend._force_ctrl.estop_called is False


def test_disable_failure_escalates_to_electronic_estop():
    n = _bare_node()
    n._imp_engaged = True
    n._imp_track_start_q = np.zeros(7)
    n._qdot_ringbuf = []
    n.args.impedance_fallback_window = 1.0
    n.args.impedance_instant_qdot_cap = 6.0
    n.args.impedance_fallback_oscillation = 0.3
    n.args.impedance_max_excursion = 0.30
    n.args.impedance_safe_margin = 0.30
    n.backend.raise_disable = True

    n._check_impedance_instability(
        np.array([0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), now=1.0)

    assert n._impedance_terminal_fault is True
    assert n.backend._force_ctrl.estop_called is True
    assert n.backend.sent == []


def test_safe_hold_send_failure_escalates_to_electronic_estop():
    n = _bare_node()
    n._imp_engaged = True
    n._imp_track_start_q = np.zeros(7)
    n._qdot_ringbuf = []
    n.args.impedance_fallback_window = 1.0
    n.args.impedance_instant_qdot_cap = 6.0
    n.args.impedance_fallback_oscillation = 0.3
    n.args.impedance_max_excursion = 0.30
    n.args.impedance_safe_margin = 0.30
    n.backend.raise_send = True

    n._check_impedance_instability(
        np.array([0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), now=1.0)

    assert n.backend.disabled is True
    assert n._impedance_terminal_fault is True
    assert n.backend._force_ctrl.estop_called is True


def test_initial_impedance_set_failure_is_fail_closed():
    n = _bare_node()
    n.backend.raise_set = True

    cmd = n._build_impedance_command(
        np.zeros(7), _servo_out(_FAKE_TCP), now=1.0)

    assert n._impedance_terminal_fault is True
    assert n.backend._force_ctrl.estop_called is True
    assert n._imp_engaged is False
    assert np.allclose(cmd.kp, 0.0)
    assert np.allclose(cmd.kd, ZERO_GRAV_KD)


def test_impedance_target_refresh_failure_is_fail_closed():
    n = _bare_node()
    q = np.zeros(7)
    n._build_impedance_command(q, _servo_out(_FAKE_TCP), now=1.0)
    n.backend.raise_set = True

    cmd = n._build_impedance_command(
        q, _servo_out(_FAKE_TCP + np.array([0.01, 0.0, 0.0])),
        now=1.0 + 1.0 / 30.0,
    )

    assert n._impedance_terminal_fault is True
    assert n.backend._force_ctrl.estop_called is True
    assert n._imp_engaged is False
    assert np.allclose(cmd.kd, ZERO_GRAV_KD)


def test_impedance_settle_uses_damping_only_not_position_stiffness():
    n = _bare_node()
    now = 10.0
    n.settle_started = time.monotonic()
    n.settle_q_ref = np.zeros(7)
    n.phase = "settle"

    n._settle_step(np.zeros(7), target=None, now=now)

    cmd = n.backend.sent[-1]
    assert np.allclose(cmd.kp, 0.0)
    assert np.allclose(cmd.kd, ZERO_GRAV_KD)


def test_unexpected_impedance_timer_exception_is_fail_closed():
    n = _bare_node()

    def fail():
        raise ValueError("bad FK")

    n._on_timer_impl = fail
    n.on_timer()

    assert n._impedance_terminal_fault is True
    assert n.backend._force_ctrl.estop_called is True
    assert "unexpected impedance timer exception" in n._impedance_fallback_reason


def test_pre_engagement_marker_loss_uses_damping_only_hold():
    n = _bare_node()
    n._imp_engaged = False
    q = np.linspace(0.0, 0.06, 7)

    cmd = n._tracking_hold_command(q, now=1.0, prefer_last=True)

    assert np.allclose(cmd.q_cmd, q)
    assert np.allclose(cmd.kp, 0.0)
    assert np.allclose(cmd.kd, ZERO_GRAV_KD)


def test_fatal_impedance_safety_stop_immediately_estops():
    n = _bare_node()

    n._handle_fatal_safety_stop(
        "hard_joint_limit", "joint6 crossed hard limit"
    )

    assert n._impedance_terminal_fault is True
    assert n.backend._force_ctrl.estop_called is True
    assert "fatal safety stop" in n._impedance_fallback_reason


def test_close_reports_disable_failure_after_estop_and_disconnect():
    n = _bare_node()
    backend = n.backend
    n._imp_engaged = True
    backend.raise_disable = True

    with np.testing.assert_raises_regex(
        RuntimeError, "impedance disable failed during close"
    ):
        n.close()

    assert backend._force_ctrl.estop_called is True
    assert backend.disconnected is True


def test_first_impedance_engagement_rechecks_limit_margin():
    n = _bare_node()
    q = np.zeros(7)
    # joint6 lower limit is about -0.733; this leaves only 0.10 rad.
    q[5] = -0.633

    cmd = n._build_impedance_command(q, _servo_out(_FAKE_TCP), now=1.0)

    assert n.backend.cfgs == []
    assert n._impedance_terminal_fault is True
    assert "refusing impedance engagement" in n._impedance_fallback_reason
    assert np.allclose(cmd.kp, 0.0)
    assert np.allclose(cmd.kd, ZERO_GRAV_KD)


def test_position_reference_stops_at_force_leash_without_windup():
    n = _bare_node()
    q = np.zeros(7)
    far_visual = _FAKE_TCP + np.array([0.20, 0.0, 0.0])
    n._build_impedance_command(q, _servo_out(far_visual), now=1.0)
    for k in range(1, 100):
        n._build_impedance_command(q, _servo_out(far_visual), now=1.0 + k / 30)

    error = np.linalg.norm(n._imp_cfg.target_tcp - _FAKE_TCP)
    assert np.isclose(error, n.args.impedance_max_position_error)
    assert np.isclose(
        n._imp_cfg.max_position_error,
        n.args.impedance_max_position_error,
    )


def test_tcp_push_outside_leash_does_not_ratchet_pbvs_reference():
    n = _bare_node()
    q = np.zeros(7)
    visual = _FAKE_TCP + np.array([0.10, 0.0, 0.0])
    servo_out = _servo_out(
        visual,
        tcp_twist=np.array([0.02, 0.0, 0.0, 0.0, 0.0, 0.0]),
    )
    n._build_impedance_command(q, servo_out, now=1.0)
    n._build_impedance_command(q, servo_out, now=1.0 + 1.0 / 30.0)
    before_push = n._imp_cfg.target_tcp.copy()

    n._imp_model._tcp = _FAKE_TCP + np.array([0.0, 0.02, 0.0])
    n._build_impedance_command(q, servo_out, now=1.0 + 2.0 / 30.0)

    assert np.allclose(n._imp_cfg.target_tcp, before_push)


def test_saturated_position_reference_accepts_recovery_step():
    current = np.zeros(3)
    previous = np.array([0.025, 0.0, 0.0])

    outward = apply_position_target_anti_windup(
        previous,
        previous + np.array([0.001, 0.0, 0.0]),
        current,
        max_error=0.025,
    )
    recovering = apply_position_target_anti_windup(
        previous,
        previous - np.array([0.001, 0.0, 0.0]),
        current,
        max_error=0.025,
    )

    assert np.allclose(outward, previous)
    assert np.allclose(recovering, [0.024, 0.0, 0.0])


def test_marker_loss_freezes_last_impedance_target():
    n = _bare_node()
    q = np.zeros(7)
    visual = _FAKE_TCP + np.array([0.01, 0.0, 0.0])
    n._build_impedance_command(q, _servo_out(visual), now=1.0)
    n._build_impedance_command(q, _servo_out(visual), now=1.0 + 1.0 / 30)
    frozen_tcp = n._imp_cfg.target_tcp.copy()
    frozen_R = n._imp_cfg.R_target.copy()
    refresh_count = len(n.backend.cfgs)

    cmd = n._build_impedance_command(
        q,
        {"target_tcp": None, "target_rotation": None},
        now=1.0 + 2.0 / 30,
    )

    assert len(n.backend.cfgs) == refresh_count
    assert np.allclose(n._imp_cfg.target_tcp, frozen_tcp)
    assert np.allclose(n._imp_cfg.R_target, frozen_R)
    assert np.allclose(cmd.kp, 0.0)
    assert np.allclose(cmd.kd, ZERO_GRAV_KD)


def test_stale_image_real_timer_path_freezes_engaged_target(monkeypatch):
    n = _bare_node()
    q = np.zeros(7)
    n._imp_engaged = True
    n._imp_cfg = SimpleNamespace(
        target_tcp=np.array([0.41, 0.0, 0.30]),
        R_target=np.eye(3),
    )
    frozen_tcp = n._imp_cfg.target_tcp.copy()
    frozen_R = n._imp_cfg.R_target.copy()
    n.phase = "tracking"
    n.started = 0.0
    n.count = 0
    n.last_log_time = 100.0
    n.last_image_warn_time = 100.0
    n.last_image = np.zeros((2, 2, 3))
    n.last_image_time = 99.0
    n.camera_matrix = np.eye(3)
    n.dist_coeffs = np.zeros(5)
    n._mit_target_q = None
    n.last_mit_q_cmd = None
    n.args.duration = 0.0
    n.args.image_timeout_s = 0.20
    n.args.enable_progress_guard = False
    n._read_joint_angles = lambda: q.copy()
    n._check_inner_impedance_fault = lambda: False
    n._check_impedance_instability = lambda *_args: None
    n._update_display = lambda *_args: None

    def update(**kwargs):
        assert kwargs["target_position"] is None
        return {
            "q_cmd": q.copy(),
            "qdot": np.zeros(7),
            "error": np.zeros(3),
            "orientation_error": np.zeros(3),
            "target_tcp": None,
            "target_rotation": None,
            "tcp_now": _FAKE_TCP.copy(),
            "target_lost": True,
            "target_rejected": False,
        }

    n.servo = SimpleNamespace(update=update)
    n.target_source = SimpleNamespace(
        detect=lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("stale image must not be detected")
        )
    )
    monkeypatch.setattr(execute_module.time, "monotonic", lambda: 100.0)

    n._on_timer_impl()

    assert np.allclose(n._imp_cfg.target_tcp, frozen_tcp)
    assert np.allclose(n._imp_cfg.R_target, frozen_R)
    assert n._imp_engaged is True
    assert n.backend.disabled is False
    assert n.backend.cfgs == []
    assert np.allclose(n.backend.sent[-1].kp, 0.0)
    assert np.allclose(n.backend.sent[-1].kd, ZERO_GRAV_KD)


def test_visual_orientation_is_fixed_at_engagement_by_default():
    n = _bare_node()
    q = np.zeros(7)
    visual_R = Rot.from_euler("z", 30.0, degrees=True).as_matrix()
    n._build_impedance_command(q, _servo_out(_FAKE_TCP, visual_R), now=1.0)
    for k in range(1, 100):
        n._build_impedance_command(
            q, _servo_out(_FAKE_TCP, visual_R), now=1.0 + k / 30
        )
    assert np.allclose(n._imp_cfg.R_target, _FAKE_R)


def test_fixed_orientation_reference_does_not_ratchet_when_pushed():
    n = _bare_node()
    q = np.zeros(7)
    n._build_impedance_command(q, _servo_out(_FAKE_TCP), now=1.0)

    n._imp_model._R = Rot.from_euler("z", 20.0, degrees=True).as_matrix()
    n._build_impedance_command(q, _servo_out(_FAKE_TCP), now=1.0 + 1.0 / 30)
    assert np.allclose(n._imp_cfg.R_target, _FAKE_R)
    assert np.isclose(
        n._imp_cfg.max_orientation_error_rad,
        np.deg2rad(n.args.impedance_max_orientation_error_deg),
    )

    n._imp_model._R = Rot.from_euler("z", 2.0, degrees=True).as_matrix()
    n._build_impedance_command(q, _servo_out(_FAKE_TCP), now=1.0 + 2.0 / 30)
    assert np.allclose(n._imp_cfg.R_target, _FAKE_R)


def test_visual_orientation_reference_is_rate_limited_with_inner_error_leash():
    n = _bare_node()
    n.args.impedance_visual_orientation = True
    q = np.zeros(7)
    visual_R = Rot.from_euler("z", 30.0, degrees=True).as_matrix()
    n._build_impedance_command(q, _servo_out(_FAKE_TCP, visual_R), now=1.0)
    n._build_impedance_command(
        q, _servo_out(_FAKE_TCP, visual_R), now=1.0 + 1.0 / 30
    )
    first_angle = Rot.from_matrix(n._imp_cfg.R_target).magnitude()
    assert first_angle <= n.args.impedance_target_angular_vel / 30.0 + 1e-9

    for k in range(2, 100):
        n._build_impedance_command(
            q, _servo_out(_FAKE_TCP, visual_R), now=1.0 + k / 30
        )
    final_angle_deg = np.degrees(Rot.from_matrix(n._imp_cfg.R_target).magnitude())
    assert np.isclose(
        final_angle_deg,
        n.args.impedance_max_orientation_error_deg,
    )
    assert np.isclose(
        n._imp_cfg.max_orientation_error_rad,
        np.deg2rad(n.args.impedance_max_orientation_error_deg),
    )


def test_saturated_orientation_reference_accepts_recovery_step():
    current = np.eye(3)
    previous = Rot.from_euler("z", 5.0, degrees=True).as_matrix()
    outward_candidate = Rot.from_euler("z", 6.0, degrees=True).as_matrix()
    recovering_candidate = Rot.from_euler("z", 4.0, degrees=True).as_matrix()

    outward = apply_orientation_target_anti_windup(
        previous,
        outward_candidate,
        current,
        max_error_rad=np.deg2rad(5.0),
    )
    recovering = apply_orientation_target_anti_windup(
        previous,
        recovering_candidate,
        current,
        max_error_rad=np.deg2rad(5.0),
    )

    assert np.allclose(outward, previous)
    assert np.isclose(
        np.degrees(Rot.from_matrix(recovering).magnitude()),
        4.0,
    )


def test_impedance_status_distinguishes_reference_and_effective_errors(caplog):
    n = _bare_node()
    q = np.zeros(7)
    n._build_impedance_command(q, _servo_out(_FAKE_TCP), now=1.0)
    n._imp_smoothed_tcp = _FAKE_TCP + np.array([0.04, 0.0, 0.0])
    n._imp_smoothed_R = Rot.from_euler("z", 20.0, degrees=True).as_matrix()
    n.backend.debug = {
        "impedance_active": True,
        "control_path": "impedance",
        "impedance_cycle_count": 42,
        "impedance_torque": np.arange(7, dtype=np.float64),
        "gravity_torque": np.ones(7),
        "total_torque": np.arange(7, dtype=np.float64) + 1.0,
        "feedback_torque": np.full(7, 0.5),
    }

    servo_out = _servo_out(_FAKE_TCP)
    servo_out.update({
        "q_cmd": q.copy(),
        "tcp_now": _FAKE_TCP.copy(),
        "tcp_rotation": _FAKE_R.copy(),
    })
    cmd = SimpleNamespace(
        qdot_cmd=np.zeros(7),
        mode="mit-impedance",
    )

    with caplog.at_level(logging.INFO, logger="test_execute_impedance"):
        n._log_status(
            "tracking", q, None, servo_out, now=1.1, reason="", cmd=cmd
        )

    assert "imp_ref_pos_err=0.0400m" in caplog.text
    assert "imp_eff_pos_err=0.0150m" in caplog.text
    assert "imp_ref_rot_err=20.00deg" in caplog.text
    assert "imp_eff_rot_err=5.00deg" in caplog.text
    assert "inner_imp_active=True" in caplog.text
    assert "inner_path=impedance" in caplog.text
    assert "inner_cycles=42" in caplog.text
    assert "tau_imp_max=6.000Nm" in caplog.text
    assert "tau_imp=[0.000, 1.000, 2.000" in caplog.text


def test_each_image_is_detected_with_its_paired_joint_snapshot(monkeypatch):
    n = _bare_node()
    n.args.image_timeout_s = 0.20
    n.camera_matrix = np.eye(3)
    n.dist_coeffs = np.zeros(5)
    q_feedback = np.arange(7, dtype=np.float64) * 0.01
    detected_q = []
    targets = []

    def detect(**kwargs):
        detected_q.append(np.asarray(kwargs["q_arm"]).copy())
        target = object()
        targets.append(target)
        return target

    n.target_source = SimpleNamespace(detect=detect)
    n._read_joint_angles = lambda: q_feedback
    monkeypatch.setattr(
        execute_module, "image_msg_to_bgr", lambda _msg: np.zeros((2, 2, 3))
    )

    n.on_image(object())
    frame_time = n.last_image_time
    q_at_frame = q_feedback.copy()
    q_feedback[:] += 1.0

    target1, is_new1 = n._detect_latest_image_target(frame_time + 0.01)
    target2, is_new2 = n._detect_latest_image_target(frame_time + 0.02)

    assert target1 is targets[0]
    assert target2 is target1
    assert is_new1 is True
    assert is_new2 is False
    assert len(detected_q) == 1
    assert np.allclose(detected_q[0], q_at_frame)

    n.on_image(object())
    second_frame_time = n.last_image_time
    n._detect_latest_image_target(second_frame_time + 0.01)
    assert len(detected_q) == 2
    assert np.allclose(detected_q[1], q_feedback)


def test_short_aruco_miss_reuses_last_target_during_timeout():
    n = _bare_node()
    n.args.image_timeout_s = 0.20
    n.args.target_timeout_s = 0.60
    n.camera_matrix = np.eye(3)
    n.dist_coeffs = np.zeros(5)
    n.last_image = np.zeros((2, 2, 3), dtype=np.uint8)
    n.last_image_q = np.zeros(7)
    n.last_image_time = 10.0
    target = object()
    detections = iter([target, None, None])
    n.target_source = SimpleNamespace(detect=lambda **_: next(detections))

    n._image_generation = 1
    first, is_new = n._detect_latest_image_target(10.01)
    assert first is target
    assert is_new is True

    n._image_generation = 2
    reused, is_new = n._detect_latest_image_target(10.20)
    assert reused is target
    assert is_new is False

    n._image_generation = 3
    n.last_image_time = 10.69
    expired, is_new = n._detect_latest_image_target(10.70)
    assert expired is None
    assert is_new is False


def test_image_freshness_helper_rejects_missing_old_and_future_frames():
    assert image_is_fresh(9.80, now=10.0, timeout_s=0.20)
    assert not image_is_fresh(9.79, now=10.0, timeout_s=0.20)
    assert not image_is_fresh(None, now=10.0, timeout_s=0.20)
    assert not image_is_fresh(10.01, now=10.0, timeout_s=0.20)
    assert not image_is_fresh(10.0, now=10.0, timeout_s=0.0)


def test_execute_defaults_use_safe_impedance_profile():
    args = parse_args([])
    assert np.allclose(args.cartesian_kp, VALIDATED_CARTESIAN_KP)
    assert np.allclose(args.cartesian_kd, VALIDATED_CARTESIAN_KD)
    assert np.allclose(args.impedance_joint_weights, np.ones(7))
    assert np.allclose(args.joint_factors, VALIDATED_GRAVITY_FACTORS)
    assert args.impedance_torque_limit == VALIDATED_TASK_TORQUE_LIMIT
    assert args.impedance_task_torque_limit == VALIDATED_TASK_TORQUE_LIMIT
    assert args.torque_limit == VALIDATED_TOTAL_TORQUE_LIMIT
    assert np.allclose(args.impedance_coulomb, 0.0)
    assert np.allclose(args.impedance_joint_hold_kp, 0.0)
    assert args.impedance_joint_hold_kd is None
    assert np.allclose(
        args.impedance_follow_kp,
        [2.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0],
    )
    assert args.impedance_follow_min_qdot == 0.08
    assert args.impedance_follow_release_error == 0.02
    assert args.impedance_target_linear_vel == 0.02
    assert args.impedance_target_twist_feedforward is True
    assert args.impedance_max_position_error == 0.015
    assert args.impedance_visual_orientation is False
    assert args.impedance_max_orientation_error_deg == 5.0
    assert args.impedance_real_vel_filter_tau == 0.0
    assert args.impedance_qdot_trip_fast == 2.0
    assert args.impedance_max_excursion == 0.0
    assert args.impedance_safe_margin == 0.30
    assert args.image_timeout_s == 0.20


def test_impedance_startup_uses_stricter_limit_margin():
    assert effective_startup_limit_margin(
        "mit-impedance", 0.08, 0.30) == 0.30
    assert effective_startup_limit_margin(
        "mit-position", 0.08, 0.30) == 0.08


def test_effective_task_torque_limit_uses_stricter_positive_bound():
    assert effective_task_torque_limit(10.0, 6.0) == 6.0
    assert effective_task_torque_limit(4.0, 10.0) == 4.0
    assert effective_task_torque_limit(0.0, 10.0) == 10.0
    assert effective_task_torque_limit(0.0, -1.0) is None
