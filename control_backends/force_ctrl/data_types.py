from dataclasses import dataclass, field
import numpy as np


@dataclass
class JointCommands:
    """Thread-safe command buffer for MIT control."""
    pos: np.ndarray
    vel: np.ndarray
    kp: np.ndarray
    kd: np.ndarray
    torques: np.ndarray

    @classmethod
    def zeros(cls, n_joints: int = 7) -> "JointCommands":
        return cls(
            pos=np.zeros(n_joints),
            vel=np.zeros(n_joints),
            kp=np.zeros(n_joints),
            kd=np.zeros(n_joints),
            torques=np.zeros(n_joints),
        )


@dataclass
class JointState:
    """Current joint state feedback."""
    pos: np.ndarray
    vel: np.ndarray
    torque: np.ndarray
    timestamp: float = 0.0


@dataclass
class ImpedanceConfig:
    """Configuration for the in-loop Cartesian impedance mode (NeroMITLoop).

    Set ONCE from the (slow) outer thread via NeroMITLoop.set_cartesian_impedance;
    read every 200 Hz cycle inside _control_loop under _impedance_lock.

    Supports the validated Nero law used by
    ``test_official_cartesian_impedance.py``:

        dq       = parser real velocity (finite-difference is a hot fallback)
        F        = K @ e - B @ (J @ dq)
        tau_task = J.T @ F
        tau      = tau_task + optional joint/null/Coulomb terms

    Non-uniform ``joint_weights`` are retained for legacy experiments only:
    post-multiplying a Jacobian-transpose torque this way does not preserve the
    standard Cartesian passivity argument. Production callers default to ones.

    ``use_dls_mapping=True`` keeps the older regularized force mapping available
    as an explicit diagnostic alternative.

    Gravity is added by the existing MIT-loop path (NOT here), then the total is
    re-clipped. The default pure-torque path uses ``p_des=q``; visual hybrid
    tracking may provide ``q_follow`` with a low ``follow_kp`` instead.
    """
    active: bool = False
    target_tcp: np.ndarray = None      # (3,) world-frame target TCP position
    R_target: np.ndarray = None        # (3,3) world-frame target TCP rotation
    max_position_error: float = None   # optional norm clamp on target_tcp-current (m)
    max_orientation_error_rad: float = None  # optional norm clamp on SO(3) error (rad)
    K: np.ndarray = None               # (6,) Cartesian stiffness [transl N/m, rot Nm/rad]
    B: np.ndarray = None               # (6,) Cartesian damping. With real velocity
                                      # (use_real_velocity=True) this is a PROPER damper
                                      # (-B*J*qdot); set per-axis (transl N·s/m, rot N·m·s/rad).
                                      # Only keep TINY (~0.01) when forced onto finite-diff.
    target_twist: np.ndarray = None    # (6,) optional desired TCP twist in the same
                                      # world-aligned frame as J. When set, the
                                      # Cartesian damper is B*(target_twist - J*dq),
                                      # allowing visual motion feed-forward without
                                      # adding a joint position spring.
    joint_damping: np.ndarray = None   # (7,) optional software joint damping (N·m·s/rad)
    tau_limit_per_joint: np.ndarray = None  # (7,) per-joint torque clip (also clipped to driver +-16)
    q_rest: np.ndarray = None          # (7,) static null-space/hold reference
    null_kq: float = 0.0              # null-space posture stiffness (0 = off, matches "pure")
    dls_lambda: float = 0.05          # DLS regularization lambda
    use_dls_mapping: bool = True      # True = optional DLS; False = validated plain J^T@F
    kp_joint: np.ndarray = None       # (7,) small friction-break joint spring (0 for pure)
    kd_joint: np.ndarray = None       # (7,) firmware kd (0 in pure-torque mode)
    q_follow: np.ndarray = None       # (7,) moving visual joint reference
    follow_kp: np.ndarray = None      # (7,) low firmware Kp for visual feed-forward
    follow_max_error: float = None    # max |q_follow-q| used for firmware command
    joint_hold_kp: np.ndarray = None  # (7,) DIRECT joint position-hold spring kp·(q_rest-q)
                                      # added as torque. Stops drift on joints weakly coupled
                                      # to the TCP task (e.g. joint7 tool-roll, which the pure
                                      # Cartesian law cannot hold). Proven: zero-G/position-hold
                                      # don't drift; pure-torque impedance does. Default joint7=5.
    joint_hold_kd: np.ndarray = None  # (7,) firmware damping for the joints selected
                                      # by joint_hold_kp; None keeps kd_joint unchanged.
    vel_filter_tau: float = 0.008     # finite-diff velocity low-pass tau for THIS mode
    qdot_trip_fast: float = 2.0       # in-loop fast divergence trip (rad/s)
    # ---- Real motor velocity (2026-06 "gold key") ----
    # The SDK zeros get_motor_states().velocity (a `# TODO: remove this after the
    # bug is fixed` line), but the parser object robot._parser.motor_state_<i>
    # .msg.velocity holds the REAL value decoded from the 0x5 high-speed CAN
    # frame. Probe (scripts/probe_motor_velocity_raw.py) confirmed it is LIVE on
    # all 7 joints, scale ~1.0 (joint-side rad/s), with sign [-1,-1,-1,-1,-1,
    # +1,-1] (joint6 correct, the rest negated — the SAME bug family as
    # move_mit's `if joint_index != 6: v_des *= -1`). When use_real_velocity is
    # True the loop reads the parser DIRECTLY (never calling get_motor_states,
    # which would zero it in place) and applies velocity_signs, yielding a clean
    # velocity for the Cartesian damping -B*J*qdot that finite-difference could
    # never provide. Falls back to finite-diff if the parser read is unavailable.
    use_real_velocity: bool = True
    velocity_signs: np.ndarray = None  # (7,) per-joint sign; default [-1,-1,-1,-1,-1,+1,-1]
    real_vel_filter_tau: float = 0.0    # optional low-pass on real velocity (s).
                                       # Default 0 avoids extra phase lag because
                                       # the firmware value is already filtered.
    # ---- Optional pieces ported from the standalone tuning harness
    # (scripts/test_official_cartesian_impedance.py) so the in-loop law can
    # reproduce that exact validated feel. ALL default None => feature OFF,
    # leaving the existing panthera-real law bit-identical (backward compatible
    # with test_impedance_fixed_target.py and the mit-position path).
    joint_weights: np.ndarray = None   # (7,) legacy per-joint post-J^T scaling.
                                      # None/all-ones = passivity-preserving baseline.
    task_torque_limit: float = None   # Spring (task) torque clip applied to the weighted
                                      # tau_cart BEFORE gravity/coulomb are added. None =
                                      # no separate spring clip (only the total
                                      # tau_limit_per_joint applied by the caller). The
                                      # harness uses +-10 (official default 3.0 is too low
                                      # -> the spring saturates and the arm won't rebound).
    coulomb: np.ndarray = None        # (7,) per-joint Coulomb friction feed-forward (Nm).
    # Optional smooth transition width for Coulomb compensation (rad).  A
    # positive value uses tanh(error / width) instead of a hard sign, so the
    # same friction assist can be used while the visual target is moving and
    # while it is stationary without a mode-switch kick.
    coulomb_smooth_width: float = 0.0
                                      # Adds coulomb*sign(q_rest - q) with a 0.5 deg
                                      # deadband (see cartesian_impedance.coulomb_compensation).
                                      # None = no friction comp. The fixed-target
                                      # harness uses [0,0.4,0,0.4,0,0,0].
