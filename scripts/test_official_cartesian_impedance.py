#!/usr/bin/env python3
"""官方示例风格的 Nero 笛卡尔空间阻抗控制测试。

参考：
  /home/dpf/桌面/agilexrobot/agilex-arm-gravity-compensation/nero/main_tast_imp.py
  /home/dpf/桌面/agilexrobot/agilex-arm-gravity-compensation/controller/task_imp_controller.py

在已经调通的关节空间阻抗执行链基础上，把关节弹簧换成 TCP 弹簧：

  F   = Kx * x_err + Bx * v_err
  tau = G(q) + J(q).T @ F + tau_coulomb

其中：
  x_err = [p_target - p_now, orientation_error(R_target, R_now)]
  v_err = -J(q) @ qdot

启动后锁定当前 TCP 位姿为目标。推动末端后，控制器尝试让 TCP
回到启动时的位置/姿态。该脚本不接 ArUco，只验证笛卡尔阻抗本体。

用法：
  conda activate agilexrobot
  cd "/home/dpf/桌面/RealRobot_Test/Visual Servo_柔顺控制"
  python3 scripts/test_official_cartesian_impedance.py
  python3 scripts/test_official_cartesian_impedance.py --execute --duration 30
"""

import argparse
import os
import sys
import time

import numpy as np


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(SCRIPT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from control_backends.force_ctrl.impedance_profile import (  # noqa: E402
    VALIDATED_CARTESIAN_KD,
    VALIDATED_CARTESIAN_KP,
    VALIDATED_COULOMB,
    VALIDATED_GRAVITY_FACTORS,
    VALIDATED_JOINT_WEIGHTS,
    VALIDATED_TASK_TORQUE_LIMIT,
    VALIDATED_TOTAL_TORQUE_LIMIT,
    VALIDATED_VELOCITY_SIGNS,
)
from control_backends.force_ctrl.real_velocity import read_real_qdot  # noqa: E402


TORQUE_LIMIT = VALIDATED_TOTAL_TORQUE_LIMIT
N_JOINTS = 7
DEFAULT_KX = VALIDATED_CARTESIAN_KP
DEFAULT_BX = VALIDATED_CARTESIAN_KD
DEFAULT_JOINT_FACTORS = VALIDATED_GRAVITY_FACTORS
DEFAULT_JOINT_WEIGHTS = VALIDATED_JOINT_WEIGHTS
DEFAULT_COULOMB = VALIDATED_COULOMB
REAL_VEL_SIGNS = VALIDATED_VELOCITY_SIGNS
NERO_JOINT_LIMITS = np.array([
    [-2.705261, 2.705261],
    [-1.745330, 1.745330],
    [-2.757621, 2.757621],
    [-1.012291, 2.146755],
    [-2.757621, 2.757621],
    [-0.733039, 0.959932],
    [-1.570797, 1.570797],
], dtype=np.float64)


def parse_vec(text: str, default: np.ndarray, n: int, name: str) -> np.ndarray:
    if not text:
        return default.copy()
    values = [float(item) for item in text.replace(",", " ").split()]
    if len(values) != n:
        raise argparse.ArgumentTypeError(f"{name} expected {n} values, got {len(values)}")
    return np.asarray(values, dtype=np.float64)


def rad2deg(v: float) -> float:
    return round(float(v) * 57.2958, 2)


def deg_list(q: np.ndarray) -> list[float]:
    return [rad2deg(v) for v in np.asarray(q, dtype=np.float64).ravel()]


def _skew(v: np.ndarray) -> np.ndarray:
    return np.array([
        [0.0, -v[2], v[1]],
        [v[2], 0.0, -v[0]],
        [-v[1], v[0], 0.0],
    ], dtype=np.float64)


def orientation_error_world(r_target: np.ndarray, r_now: np.ndarray) -> np.ndarray:
    """Rotation-vector orientation error, expressed in world/base frame."""
    import pinocchio as pin

    r_err = np.asarray(r_target, dtype=np.float64) @ np.asarray(r_now, dtype=np.float64).T
    return np.asarray(pin.log3(r_err), dtype=np.float64).reshape(3)


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


def coulomb_compensation(q_rest: np.ndarray, q: np.ndarray, coulomb: np.ndarray) -> np.ndarray:
    err = np.asarray(q_rest, dtype=np.float64) - np.asarray(q, dtype=np.float64)
    tau = np.zeros(N_JOINTS, dtype=np.float64)
    for i in range(N_JOINTS):
        if coulomb[i] <= 0.0 or abs(err[i]) < np.deg2rad(0.5):
            continue
        tau[i] = coulomb[i] * np.sign(err[i])
    return tau


class VelocityEstimator:
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


def main() -> None:
    default_urdf = os.path.join(PROJECT_ROOT, "models", "nero_with_gripper.urdf")

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
    )
    parser.add_argument("--torque-limit", type=float, default=TORQUE_LIMIT)
    parser.add_argument(
        "--task-torque-limit",
        type=float,
        default=VALIDATED_TASK_TORQUE_LIMIT,
    )
    parser.add_argument("--gravity-factor", type=float, default=1.0)
    parser.add_argument(
        "--joint-factors",
        default=",".join(str(v) for v in VALIDATED_GRAVITY_FACTORS),
    )
    parser.add_argument("--kx", default="", help="Cartesian stiffness [N/m,N/m,N/m,Nm/rad*3].")
    parser.add_argument("--bx", default="", help="Cartesian damping [Ns/m*3,Nms/rad*3].")
    parser.add_argument("--kx-scale", type=float, default=1.0)
    parser.add_argument("--bx-scale", type=float, default=1.0)
    parser.add_argument(
        "--joint-weights",
        default=",".join(str(v) for v in VALIDATED_JOINT_WEIGHTS),
    )
    parser.add_argument(
        "--coulomb",
        default=",".join(str(v) for v in VALIDATED_COULOMB),
    )
    parser.add_argument("--vel-filter-tau", type=float, default=0.02)
    parser.add_argument(
        "--use-real-velocity",
        dest="use_real_velocity",
        action="store_true",
        default=True,
        help="Read real motor velocity from the parser (gold key). Default ON.",
    )
    parser.add_argument(
        "--no-real-velocity",
        dest="use_real_velocity",
        action="store_false",
        help="Force finite-difference velocity (the old laggy path; for A/B comparison).",
    )
    parser.add_argument(
        "--velocity-signs",
        default=",".join(str(v) for v in VALIDATED_VELOCITY_SIGNS),
        help="Per-joint sign for real velocity (joint6 correct, rest negated).",
    )
    parser.add_argument("--safe-margin-deg", type=float, default=5.0)
    parser.add_argument("--max-pos-error", type=float, default=0.25, help="Safety trip in meters.")
    parser.add_argument("--max-rot-error-deg", type=float, default=45.0)
    parser.add_argument(
        "--position-only",
        action="store_true",
        help="Set rotational stiffness/damping to zero for first tests.",
    )
    args = parser.parse_args()

    kx = parse_vec(args.kx, DEFAULT_KX * args.kx_scale, 6, "kx")
    bx = parse_vec(args.bx, DEFAULT_BX * args.bx_scale, 6, "bx")
    if args.position_only:
        kx[3:] = 0.0
        bx[3:] = 0.0
    joint_factors = parse_vec(args.joint_factors, DEFAULT_JOINT_FACTORS, 7, "joint-factors")
    joint_factors = joint_factors * args.gravity_factor
    joint_weights = parse_vec(
        args.joint_weights, DEFAULT_JOINT_WEIGHTS, 7, "joint-weights"
    )
    coulomb = parse_vec(args.coulomb, DEFAULT_COULOMB, 7, "coulomb")
    velocity_signs = parse_vec(args.velocity_signs, REAL_VEL_SIGNS, 7, "velocity-signs")
    vel_source = "real(parser)" if args.use_real_velocity else "finite-diff"
    safe_margin = np.deg2rad(args.safe_margin_deg)
    max_rot_error = np.deg2rad(args.max_rot_error_deg)

    print("=== Official-Style Nero Cartesian Impedance ===")
    print("law: tau = G(q) + weights*(J.T @ F) + coulomb")
    print("F = Kx*x_err + Bx*(-J*qdot)")
    print(f"gravity_source={args.gravity_source}")
    print(f"urdf={args.urdf}")
    print(f"channel={args.channel} firmware={args.firmware} freq={args.freq}Hz")
    print(f"duration={args.duration}s torque_limit=±{args.torque_limit}Nm")
    print(f"task_torque_limit=±{args.task_torque_limit}Nm")
    print(f"joint_factors={np.round(joint_factors, 3).tolist()}")
    print(f"joint_weights={np.round(joint_weights, 3).tolist()}")
    print(f"Kx={np.round(kx, 3).tolist()}")
    print(f"Bx={np.round(bx, 3).tolist()}")
    print(f"coulomb={np.round(coulomb, 3).tolist()}")
    print(f"velocity_source={vel_source}  signs={velocity_signs.tolist()}  "
          f"(real=clean Bx damping; finite-diff=laggy, Bx must stay low)")

    if not args.execute:
        print("\n加 --execute 连接实机。首次建议加 --position-only 做位置保持测试。")
        return

    from models.nero_model import NeroPinocchioModel

    pin_model = NeroPinocchioModel(urdf_path=args.urdf)

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
    p_target, r_target, _ = pin_model.fk_and_jacobian(q0)
    q_rest = q0.copy()
    print(f"启动角(deg): {deg_list(q0)}")
    print(f"锁定 target_tcp={np.round(p_target, 4).tolist()}")

    robot.set_auto_set_motion_mode_enabled(False)
    robot.set_motion_mode("mit")
    time.sleep(0.2)
    print("已切换 MIT 模式")
    print("立即进入笛卡尔阻抗。Ctrl+C 停止并 disable。\n")

    vel_est = VelocityEstimator(args.vel_filter_tau)
    dt = 1.0 / args.freq
    t_start = time.time()
    count = 0

    try:
        while time.time() - t_start < args.duration:
            loop_start = time.time()
            ja = robot.get_joint_angles()
            if ja is None:
                time.sleep(dt)
                continue
            q = np.asarray(ja.msg, dtype=np.float64)
            # Always refresh finite-diff so the fallback stays valid if real
            # velocity ever drops out (avoids a stale-dt spike). Use the real
            # parser velocity when available — it is clean (no lag), so Bx can
            # finally be set to proper critical damping instead of staying low.
            fdiff_qdot = vel_est.update(q)
            real_qdot = read_real_qdot(robot, velocity_signs) if args.use_real_velocity else None
            qdot = real_qdot if real_qdot is not None else fdiff_qdot
            elapsed = time.time() - t_start

            limit_hit = joint_limit_warning(q, safe_margin)
            if limit_hit:
                raise RuntimeError(f"near joint limit: {limit_hit}")

            p_now, r_now, jac = pin_model.fk_and_jacobian(q)
            e_pos = p_target - p_now
            e_ori = orientation_error_world(r_target, r_now)
            if np.linalg.norm(e_pos) > args.max_pos_error:
                raise RuntimeError(f"position error too large: {np.linalg.norm(e_pos):.3f}m")
            if np.linalg.norm(e_ori) > max_rot_error:
                raise RuntimeError(f"rotation error too large: {rad2deg(np.linalg.norm(e_ori)):.1f}deg")

            xdot = jac @ qdot
            wrench = kx * np.concatenate([e_pos, e_ori]) - bx * xdot
            tau_task = joint_weights * (jac.T @ wrench)
            tau_task = np.clip(tau_task, -args.task_torque_limit, args.task_torque_limit)
            tau_dyn = compute_dynamics(q, qdot)
            tau_c = coulomb_compensation(q_rest, q, coulomb)
            tau = np.clip(tau_dyn + tau_task + tau_c, -args.torque_limit, args.torque_limit)

            if not np.all(np.isfinite(tau)):
                raise RuntimeError(f"non-finite torque: {tau}")

            for i in range(N_JOINTS):
                robot.move_mit(
                    joint_index=i + 1,
                    p_des=0.0,
                    v_des=0.0,
                    kp=0.0,
                    kd=0.0,
                    t_ff=float(tau[i]),
                )

            count += 1
            if count % max(1, int(args.freq * 0.5)) == 0:
                print(
                    f"t={elapsed:5.1f}s "
                    f"pos_err={1000.0*np.linalg.norm(e_pos):6.1f}mm "
                    f"rot_err={rad2deg(np.linalg.norm(e_ori)):5.1f}deg "
                    f"|xdot|={np.linalg.norm(xdot):.3f} "
                    f"tau_task={np.round(tau_task, 2).tolist()} "
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
            pf, rf, _ = pin_model.fk_and_jacobian(qf)
            ef = p_target - pf
            eof = orientation_error_world(r_target, rf)
            print(f"最终角(deg): {deg_list(qf)}")
            print(f"最终 TCP 误差: {1000.0*np.linalg.norm(ef):.1f}mm, 姿态 {rad2deg(np.linalg.norm(eof)):.1f}deg")

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
