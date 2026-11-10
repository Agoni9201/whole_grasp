#!/usr/bin/env python3
"""关节空间柔顺保持测试 — 复现 pyAgxArm compliant，重力补偿对齐 test_zero_gravity。

和 scripts/test_zero_gravity.py 使用同一条底层路径：
  200Hz 循环：
    1. 读关节角 q (7)
    2. NeroGravityCompensator.compute(q) -> tau_gravity
    3. tau_gravity *= joint_factors，clamp 到 torque_limit
    4. move_mit(i, p_des=q_start[i], kp=soft_kp[i], kd=soft_kd[i], t_ff=tau_gravity[i])

区别：
  - test_zero_gravity: p_des=q_now, kp=0，只补重力，手拖不会回弹。
  - 本脚本: p_des=q_start, kp 为低刚度，手推后会软回弹到启动姿态。

用法：
  conda activate agilexrobot
  cd "/home/dpf/桌面/RealRobot_Test/Visual Servo_柔顺控制"
  python3 scripts/test_force_compliant_hold.py
  python3 scripts/test_force_compliant_hold.py --execute --duration 30
"""

import argparse
import os
import sys
import time

import numpy as np


SOFT_KP = np.array([5.0, 5.0, 5.0, 3.0, 2.0, 1.0, 1.0])
SOFT_KD = np.array([2.0, 2.0, 2.0, 1.0, 0.5, 0.3, 0.3])
TORQUE_LIMIT = 14.0


def parse_vec7(text: str, default: np.ndarray) -> np.ndarray:
    if not text:
        return default.copy()
    values = [float(item) for item in text.replace(",", " ").split()]
    if len(values) != 7:
        raise argparse.ArgumentTypeError(
            f"expected 7 values, got {len(values)} from {text!r}"
        )
    return np.asarray(values, dtype=np.float64)


def rad2deg(v: float) -> float:
    return round(float(v) * 57.2958, 2)


def deg_list(q: np.ndarray) -> list[float]:
    return [rad2deg(v) for v in np.asarray(q, dtype=np.float64)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--channel", default="can0")
    parser.add_argument("--firmware", default="v111")
    parser.add_argument("--interface", default="socketcan")
    parser.add_argument("--freq", type=int, default=200)
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument(
        "--torque-limit",
        type=float,
        default=TORQUE_LIMIT,
        help=f"力矩安全上限 N*m，默认对齐 test_zero_gravity.py: {TORQUE_LIMIT}",
    )
    parser.add_argument("--gravity-factor", type=float, default=1.0)
    parser.add_argument(
        "--joint-factors",
        default="1,1,1,1,1,1,1.5",
        help="逐关节重力补偿系数，默认对齐 test_zero_gravity.py。",
    )
    parser.add_argument("--kp-scale", type=float, default=1.0)
    parser.add_argument("--kd-scale", type=float, default=1.0)
    parser.add_argument("--kp", default="", help="覆盖 soft_kp，7 个逗号或空格分隔数。")
    parser.add_argument("--kd", default="", help="覆盖 soft_kd，7 个逗号或空格分隔数。")
    parser.add_argument(
        "--zero-g-settle-s",
        type=float,
        default=0.5,
        help="启动后先按零重力命令保持的时间，再锁定 q_start 进入柔顺保持。",
    )
    args = parser.parse_args()

    kp = parse_vec7(args.kp, SOFT_KP * args.kp_scale)
    kd = parse_vec7(args.kd, SOFT_KD * args.kd_scale)
    joint_factors = parse_vec7(args.joint_factors, np.ones(7)) * args.gravity_factor

    print("=== Nero Compliant Joint Hold ===")
    print("重力补偿路径: scripts/test_zero_gravity.py")
    print("柔顺参数来源: pyAgxArm force_ctrl_demo --mode compliant")
    print(f"channel={args.channel} firmware={args.firmware} freq={args.freq}Hz")
    print(f"duration={args.duration}s torque_limit=±{args.torque_limit}Nm")
    print(f"joint_factors={np.round(joint_factors, 3).tolist()}")
    print(f"kp={np.round(kp, 3).tolist()}")
    print(f"kd={np.round(kd, 3).tolist()}")

    if not args.execute:
        print("\n加 --execute 连接实机。")
        print("预期手感: 比零重力硬一点，轻推可偏离，松手会软回到启动姿态。")
        return

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.normpath(os.path.join(script_dir, ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    from control_backends.force_ctrl.gravity_comp import NeroGravityCompensator
    from pyAgxArm import AgxArmFactory, create_agx_arm_config

    grav_comp = NeroGravityCompensator()
    print(f"重力补偿模型已加载 (nq={grav_comp.nq})")

    print(f"连接机械臂 ({args.firmware} driver)...")
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
    q0 = np.array(ja.msg, dtype=np.float64)
    print(f"启动角(deg): {deg_list(q0)}")

    robot.set_auto_set_motion_mode_enabled(False)
    robot.set_motion_mode("mit")
    time.sleep(0.2)
    print("已切换 MIT 模式")

    dt = 1.0 / args.freq
    count = 0
    t_start = time.time()
    q_start = None

    print(f"先零重力 settle {args.zero_g_settle_s:.2f}s，然后进入柔顺保持。")
    print("Ctrl+C 停止并 disable。\n")

    try:
        while time.time() - t_start < args.duration:
            ja = robot.get_joint_angles()
            if ja is None:
                time.sleep(dt)
                continue
            q = np.array(ja.msg, dtype=np.float64)

            tau_g = grav_comp.compute(q[:7]) * joint_factors
            tau_g = np.clip(tau_g, -args.torque_limit, args.torque_limit)

            elapsed = time.time() - t_start
            if elapsed < args.zero_g_settle_s:
                p_des = q
                kp_cmd = np.zeros(7)
                mode = "zero_g"
            else:
                if q_start is None:
                    q_start = q.copy()
                    print(f"锁定柔顺目标 q_start(deg): {deg_list(q_start)}")
                p_des = q_start
                kp_cmd = kp
                mode = "compliant"

            for i in range(7):
                robot.move_mit(
                    joint_index=i + 1,
                    p_des=float(p_des[i]),
                    v_des=0.0,
                    kp=float(kp_cmd[i]),
                    kd=float(kd[i]),
                    t_ff=float(tau_g[i]),
                )

            count += 1
            if count % max(1, int(args.freq * 0.5)) == 0:
                err = np.zeros(7) if q_start is None else np.degrees(q - q_start)
                print(
                    f"t={elapsed:5.1f}s mode={mode:9s} "
                    f"max_err={np.max(np.abs(err)):6.2f}deg "
                    f"q(deg)={deg_list(q)} tau_g={np.round(tau_g, 2).tolist()}"
                )

            time.sleep(dt)

    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        elapsed = time.time() - t_start
        print(f"\n运行 {elapsed:.1f}s, 发了 {count} 轮")
        ja_final = robot.get_joint_angles()
        if ja_final is not None:
            qf = np.array(ja_final.msg, dtype=np.float64)
            ref = q_start if q_start is not None else q0
            print(f"最终角(deg): {deg_list(qf)}")
            print(f"相对目标变化(deg): {deg_list(qf - ref)}")

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
