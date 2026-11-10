#!/usr/bin/env python3
"""官方示例风格的 Nero 关节空间阻抗控制测试。

参考：
  /home/dpf/桌面/agilexrobot/agilex-arm-gravity-compensation/nero/main_jnt_imp.py
  /home/dpf/桌面/agilexrobot/agilex-arm-gravity-compensation/controller/jnt_imp_controller.py

默认重力补偿路径参考已验证脚本：
  python scripts/test_zero_gravity.py --execute --kd-scale 0.5 --joint-factors "1,1,1,1,1,1,1.5"

控制律：
  tau = G(q) + K * (q_target - q) + B * (0 - qdot)

执行方式参考官方示例：
  move_mit(joint, p_des=0, v_des=0, kp=0, kd=mit_kd, t_ff=tau)

这和 test_force_compliant_hold.py 不同：本脚本不使用固件侧 kp/kd，
而是把阻抗弹簧/阻尼全部算成 t_ff 力矩。

用法：
  conda activate agilexrobot
  cd "/home/dpf/桌面/RealRobot_Test/Visual Servo_柔顺控制"
  python3 scripts/test_official_joint_impedance.py
  python3 scripts/test_official_joint_impedance.py --execute --duration 30
"""

import argparse
import os
import sys
import time

import numpy as np


TORQUE_LIMIT = 14.0
DEFAULT_K = np.array([10.0, 10.0, 10.0, 10.0, 1.0, 1.0, 1.0], dtype=np.float64)
DEFAULT_B = np.array([0.5, 0.8, 0.8, 0.8, 0.2, 0.2, 0.2], dtype=np.float64)
ZERO_GRAV_KD = np.array([1.0, 1.0, 1.0, 0.5, 0.3, 0.2, 0.2], dtype=np.float64)
DEFAULT_JOINT_FACTORS = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.5], dtype=np.float64)
DEFAULT_COULOMB = np.zeros(7, dtype=np.float64)
N_JOINTS = 7
NERO_JOINT_LIMITS = np.array([
    [-2.705261, 2.705261],
    [-1.745330, 1.745330],
    [-2.757621, 2.757621],
    [-1.012291, 2.146755],
    [-2.757621, 2.757621],
    [-0.733039, 0.959932],
    [-1.570797, 1.570797],
], dtype=np.float64)


def parse_vec7(text: str, default: np.ndarray) -> np.ndarray:
    if not text:
        return default.copy()
    values = [float(item) for item in text.replace(",", " ").split()]
    if len(values) != N_JOINTS:
        raise argparse.ArgumentTypeError(
            f"expected 7 values, got {len(values)} from {text!r}"
        )
    return np.asarray(values, dtype=np.float64)


def rad2deg(v: float) -> float:
    return round(float(v) * 57.2958, 2)


def deg_list(q: np.ndarray) -> list[float]:
    return [rad2deg(v) for v in np.asarray(q, dtype=np.float64).ravel()]


class VelocityEstimator:
    """Finite-difference qdot with a small low-pass filter.

    Nero motor velocity feedback has not been reliable in our previous tests,
    so this follows the local MIT loop approach instead of reading SDK velocity.
    """

    def __init__(self, tau: float):
        self.tau = float(tau)
        self.last_q = None
        self.last_t = None
        self.qdot = np.zeros(N_JOINTS, dtype=np.float64)

    def update(self, q: np.ndarray) -> np.ndarray:
        now = time.time()
        q = np.asarray(q, dtype=np.float64).ravel()
        if self.last_q is None or self.last_t is None:
            self.last_q = q.copy()
            self.last_t = now
            return self.qdot.copy()
        dt = now - self.last_t
        if dt <= 1e-4:
            return self.qdot.copy()
        raw = (q - self.last_q) / dt
        alpha = dt / (self.tau + dt)
        self.qdot = alpha * raw + (1.0 - alpha) * self.qdot
        self.last_q = q.copy()
        self.last_t = now
        return self.qdot.copy()


class PinocchioNLE:
    """Pinocchio nonLinearEffects(q, qdot) for the modified Nero URDF."""

    def __init__(self, urdf_path: str):
        import pinocchio as pin

        self.pin = pin
        self.model = pin.buildModelFromUrdf(urdf_path)
        self.data = self.model.createData()
        self.arm_idx_q = []
        self.arm_idx_v = []
        for i in range(1, N_JOINTS + 1):
            jid = self.model.getJointId(f"joint{i}")
            if jid >= self.model.njoints:
                raise RuntimeError(f"joint{i} not found in {urdf_path}")
            self.arm_idx_q.append(self.model.joints[jid].idx_q)
            self.arm_idx_v.append(self.model.joints[jid].idx_v)

    def compute(self, q: np.ndarray, qdot: np.ndarray) -> np.ndarray:
        q_full = np.zeros(self.model.nq, dtype=np.float64)
        v_full = np.zeros(self.model.nv, dtype=np.float64)
        for i, idx in enumerate(self.arm_idx_q):
            q_full[idx] = q[i]
        for i, idx in enumerate(self.arm_idx_v):
            v_full[idx] = qdot[i]
        nle = self.pin.nonLinearEffects(self.model, self.data, q_full, v_full)
        return np.asarray(nle[self.arm_idx_v], dtype=np.float64)


def joint_limit_warning(q: np.ndarray, margin_rad: float) -> str:
    lower = NERO_JOINT_LIMITS[:, 0] + margin_rad
    upper = NERO_JOINT_LIMITS[:, 1] - margin_rad
    hits = []
    for i in range(N_JOINTS):
        if q[i] < lower[i] or q[i] > upper[i]:
            hits.append(
                f"J{i+1}={rad2deg(q[i]):.1f}deg "
                f"limit=[{rad2deg(NERO_JOINT_LIMITS[i,0]):.1f},"
                f"{rad2deg(NERO_JOINT_LIMITS[i,1]):.1f}]deg"
            )
    return ", ".join(hits)


def coulomb_compensation(err: np.ndarray, qdot: np.ndarray, coulomb: np.ndarray) -> np.ndarray:
    """Small static-friction/deadband assist toward the target.

    Direction is based on the spring error, not velocity, so the extra torque
    always helps return toward q_target.  Disabled by default.
    """
    tau = np.zeros(N_JOINTS, dtype=np.float64)
    for i in range(N_JOINTS):
        if coulomb[i] <= 0.0:
            continue
        # Near zero error we do not add Coulomb torque, otherwise it can buzz.
        if abs(err[i]) < np.deg2rad(0.5):
            continue
        tau[i] = coulomb[i] * np.sign(err[i])
    return tau


def main() -> None:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.normpath(os.path.join(script_dir, ".."))
    default_urdf = os.path.join(project_root, "models", "nero_with_gripper.urdf")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--channel", default="can0")
    parser.add_argument("--firmware", default="v111")
    parser.add_argument("--interface", default="socketcan")
    parser.add_argument("--freq", type=int, default=200)
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--urdf", default=default_urdf)
    parser.add_argument(
        "--gravity-source",
        choices=["mujoco", "pinocchio"],
        default="mujoco",
        help=(
            "mujoco matches the validated test_zero_gravity.py path. "
            "pinocchio matches the official example's nle(q,v)."
        ),
    )
    parser.add_argument("--torque-limit", type=float, default=TORQUE_LIMIT)
    parser.add_argument(
        "--pd-torque-limit",
        type=float,
        default=3.0,
        help="Clip only the impedance spring/damping torque before adding gravity.",
    )
    parser.add_argument("--gravity-factor", type=float, default=1.0)
    parser.add_argument(
        "--joint-factors",
        default="1,1,1,1,1,1,1.5",
        help="Default matches the validated zero-gravity test.",
    )
    parser.add_argument("--k", default="", help="Joint stiffness K, 7 values, N*m/rad.")
    parser.add_argument("--b", default="", help="Joint damping B, 7 values, N*m*s/rad.")
    parser.add_argument(
        "--coulomb",
        default="",
        help=(
            "Optional static-friction assist torque, 7 values in Nm. "
            "Example for J7 only: '0,0,0,0,0,0,0.15'. Default disabled."
        ),
    )
    parser.add_argument("--k-scale", type=float, default=1.0)
    parser.add_argument("--b-scale", type=float, default=1.0)
    parser.add_argument(
        "--mit-kd-scale",
        type=float,
        default=0.0,
        help=(
            "Firmware-side MIT kd scale. Default 0 keeps the official pure "
            "t_ff impedance behavior. Use 0.5 only for damping diagnostics."
        ),
    )
    parser.add_argument(
        "--mit-kd",
        default="",
        help="Override firmware-side MIT kd, 7 values. Empty = ZERO_GRAV_KD * mit-kd-scale.",
    )
    parser.add_argument(
        "--settle-s",
        type=float,
        default=0.0,
        help=(
            "Optional gravity-only settle before impedance. Default 0.0 matches "
            "the official example better and avoids sagging before q_target is held."
        ),
    )
    parser.add_argument("--vel-filter-tau", type=float, default=0.02)
    parser.add_argument("--max-excursion-deg", type=float, default=35.0)
    parser.add_argument("--safe-margin-deg", type=float, default=5.0)
    args = parser.parse_args()

    k = parse_vec7(args.k, DEFAULT_K * args.k_scale)
    b = parse_vec7(args.b, DEFAULT_B * args.b_scale)
    mit_kd = parse_vec7(args.mit_kd, ZERO_GRAV_KD * args.mit_kd_scale)
    coulomb = parse_vec7(args.coulomb, DEFAULT_COULOMB)
    joint_factors = parse_vec7(args.joint_factors, DEFAULT_JOINT_FACTORS) * args.gravity_factor
    max_excursion = np.deg2rad(args.max_excursion_deg)
    safe_margin = np.deg2rad(args.safe_margin_deg)

    print("=== Official-Style Nero Joint Impedance ===")
    print("law: tau = G(q) + K*(q_target-q) + B*(0-qdot)")
    print("execution: move_mit(..., kp=0, kd=mit_kd, t_ff=tau)")
    print(f"gravity_source={args.gravity_source}")
    print(f"urdf={args.urdf}")
    print(f"channel={args.channel} firmware={args.firmware} freq={args.freq}Hz")
    print(f"duration={args.duration}s gravity_only_settle={args.settle_s}s")
    print(f"torque_limit=±{args.torque_limit}Nm pd_torque_limit=±{args.pd_torque_limit}Nm")
    print(f"joint_factors={np.round(joint_factors, 3).tolist()}")
    print(f"K={np.round(k, 3).tolist()}")
    print(f"B={np.round(b, 3).tolist()}")
    print(f"mit_kd={np.round(mit_kd, 3).tolist()}")
    print(f"coulomb={np.round(coulomb, 3).tolist()}")

    if not args.execute:
        print("\n加 --execute 连接实机。")
        print("默认 K/B 已对齐官方 Nero 示例；太硬再降到 --k-scale 0.5 或 0.3。")
        return

    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    if args.gravity_source == "mujoco":
        from control_backends.force_ctrl.gravity_comp import NeroGravityCompensator

        gravity_model = NeroGravityCompensator(urdf_path=args.urdf)

        def compute_dynamics(q, qdot):
            return gravity_model.compute(q) * joint_factors

        print(f"MuJoCo gravity model loaded (nq={gravity_model.nq})")
    else:
        gravity_model = PinocchioNLE(args.urdf)

        def compute_dynamics(q, qdot):
            return gravity_model.compute(q, qdot) * joint_factors

        print("Pinocchio nle model loaded")

    from pyAgxArm import AgxArmFactory, create_agx_arm_config

    print("连接机械臂...")
    cfg = create_agx_arm_config(
        robot="nero",
        comm="can",
        channel=args.channel,
        interface=args.interface,
        firmeware_version=args.firmware,
    )
    robot = AgxArmFactory.create_arm(cfg)
    robot.connect()

    robot.set_normal_mode()
    for _ in range(100):
        if robot.enable():
            break
        time.sleep(0.01)
    else:
        raise RuntimeError("enable 失败")
    print("已使能")
    time.sleep(0.5)

    st = robot.get_arm_status()
    if st and "EMERGENCY" in str(getattr(st.msg, "arm_status", "")):
        print("检测到急停状态，复位中...")
        robot.disable()
        time.sleep(0.3)
        robot.reset()
        time.sleep(0.5)
        robot.clear_joint_error(255)
        time.sleep(0.3)
        robot.set_normal_mode()
        time.sleep(0.3)
        for _ in range(100):
            if robot.enable():
                break
            time.sleep(0.01)
        time.sleep(0.5)

    ja = robot.get_joint_angles()
    if ja is None:
        raise RuntimeError("无关节反馈")
    q0 = np.asarray(ja.msg, dtype=np.float64)
    print(f"启动角(deg): {deg_list(q0)}")
    q_target = q0.copy()
    print(f"锁定 q_target(deg): {deg_list(q_target)}")

    robot.set_auto_set_motion_mode_enabled(False)
    robot.set_motion_mode("mit")
    time.sleep(0.2)
    print("已切换 MIT 模式")

    vel_est = VelocityEstimator(args.vel_filter_tau)
    dt = 1.0 / args.freq
    t_start = time.time()
    count = 0

    if args.settle_s > 0.0:
        print(f"先按重力补偿 settle {args.settle_s:.2f}s，然后回到已锁定 q_target。")
    else:
        print("立即进入关节阻抗，不做 gravity-only settle。")
    print("Ctrl+C 停止并 disable。\n")

    try:
        while time.time() - t_start < args.duration:
            loop_start = time.time()
            ja = robot.get_joint_angles()
            if ja is None:
                time.sleep(dt)
                continue
            q = np.asarray(ja.msg, dtype=np.float64)
            qdot = vel_est.update(q)
            elapsed = time.time() - t_start

            limit_hit = joint_limit_warning(q, safe_margin)
            if limit_hit:
                raise RuntimeError(f"near joint limit: {limit_hit}")

            tau_dyn = compute_dynamics(q, qdot)
            if elapsed < args.settle_s:
                tau_pd = np.zeros(N_JOINTS, dtype=np.float64)
                tau_coulomb = np.zeros(N_JOINTS, dtype=np.float64)
                mode = "gravity"
            else:
                err = q_target - q
                if np.max(np.abs(err)) > max_excursion:
                    idx = int(np.argmax(np.abs(err)))
                    raise RuntimeError(
                        f"max excursion exceeded: J{idx+1} "
                        f"{rad2deg(err[idx]):.1f}deg > {args.max_excursion_deg:.1f}deg"
                    )
                tau_pd = k * err - b * qdot
                tau_pd = np.clip(tau_pd, -args.pd_torque_limit, args.pd_torque_limit)
                tau_coulomb = coulomb_compensation(err, qdot, coulomb)
                mode = "impedance"

            tau = np.clip(tau_dyn + tau_pd + tau_coulomb, -args.torque_limit, args.torque_limit)
            if not np.all(np.isfinite(tau)):
                raise RuntimeError(f"non-finite torque: {tau}")

            for i in range(N_JOINTS):
                robot.move_mit(
                    joint_index=i + 1,
                    p_des=0.0,
                    v_des=0.0,
                    kp=0.0,
                    kd=float(mit_kd[i]),
                    t_ff=float(tau[i]),
                )

            count += 1
            if count % max(1, int(args.freq * 0.5)) == 0:
                err_deg = np.zeros(N_JOINTS) if q_target is None else np.degrees(q - q_target)
                print(
                    f"t={elapsed:5.1f}s mode={mode:9s} "
                    f"max_err={np.max(np.abs(err_deg)):6.2f}deg "
                    f"qdot_max={np.max(np.abs(qdot)):.3f}rad/s "
                    f"j7_err={err_deg[6]:+.2f}deg "
                    f"j7_tau(g,pd,c,total)="
                    f"({tau_dyn[6]:+.2f},{tau_pd[6]:+.2f},{tau_coulomb[6]:+.2f},{tau[6]:+.2f}) "
                    f"tau_pd={np.round(tau_pd, 2).tolist()} "
                    f"tau={np.round(tau, 2).tolist()}"
                )

            sleep_s = dt - (time.time() - loop_start)
            if sleep_s > 0:
                time.sleep(sleep_s)

    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        elapsed = time.time() - t_start
        print(f"\n运行 {elapsed:.1f}s, 发了 {count} 轮")
        ja_final = robot.get_joint_angles()
        if ja_final is not None:
            qf = np.asarray(ja_final.msg, dtype=np.float64)
            print(f"最终角(deg): {deg_list(qf)}")
            print(f"相对目标变化(deg): {deg_list(qf - q_target)}")

        try:
            robot.electronic_emergency_stop()
        except Exception:
            pass
        time.sleep(0.2)
        try:
            robot.disable()
        except Exception:
            pass
        print("已停止")


if __name__ == "__main__":
    main()
