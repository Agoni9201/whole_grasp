#!/usr/bin/env python3
"""零重力拖动测试 — MIT + 带夹爪 MuJoCo 重力补偿。

核心逻辑：
  200Hz 循环：
    1. 读关节角 q (7)
    2. MuJoCo mj_inverse(q) → tau_gravity (7)
    3. clamp(tau_gravity, ±torque_limit)
    4. move_mit(i, p_des=q[i], kp=0, kd=kd[i], t_ff=tau_gravity[i])

参数对齐 fork force_ctrl_demo:
  kp = 0 (零重力不锁位置)
  kd = ZERO_GRAV_KD = [1.0, 1.0, 1.0, 0.5, 0.3, 0.2, 0.2]
  torque_limit = 8.0 N·m

用法：
  conda activate agilexrobot
  python scripts/test_zero_gravity.py --execute
"""
import argparse
import os
import sys
import time
import numpy as np

# === 对齐 fork force_ctrl 参数 ===
ZERO_GRAV_KD = np.array([1.0, 1.0, 1.0, 0.5, 0.3, 0.2, 0.2])
TORQUE_LIMIT = 14.0  # N·m，带夹爪需要更大（fork 8.0 是不带夹爪的）


def rad2deg(v):
    return round(v * 57.2958, 2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--channel", default="can0")
    parser.add_argument("--freq", type=int, default=200)
    parser.add_argument("--kd-scale", type=float, default=1.0,
                        help="kd 缩放因子 (default 1.0，范围 0.0~2.0)")
    parser.add_argument("--torque-limit", type=float, default=TORQUE_LIMIT,
                        help=f"力矩安全上限 N·m (default {TORQUE_LIMIT})")
    parser.add_argument("--duration", type=float, default=30.0,
                        help="测试时长秒 (default 30)")
    parser.add_argument("--no-gravity", action="store_true",
                        help="不加重力补偿（对比用）")
    parser.add_argument("--gravity-factor", type=float, default=1.0,
                        help="全局重力补偿因子 (default 1.0)")
    parser.add_argument("--joint-factors", type=str, default="",
                        help="逐关节补偿系数, 逗号分隔7个数, 如 '1,1,1,1,1,1,1.2'")
    args = parser.parse_args()

    if not args.execute:
        print("加 --execute 连接实机")
        print("可选: --no-gravity (不加重力补偿对比)")
        print(f"      --kd-scale 1.0 (kd 缩放)")
        print(f"      --torque-limit {TORQUE_LIMIT} (力矩安全上限)")
        print("      --gravity-factor 1.0 (重力补偿系数)")
        print("      --duration 30 (秒)")
        return

    # 加载重力补偿
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.normpath(os.path.join(script_dir, ".."))
    sys.path.insert(0, project_root)

    from control_backends.force_ctrl.gravity_comp import NeroGravityCompensator

    grav_comp = NeroGravityCompensator()
    print(f"✅ 重力补偿模型已加载 (nq={grav_comp.nq})")

    # 解析逐关节补偿系数
    joint_factors = np.ones(7)
    if args.joint_factors:
        vals = [float(x) for x in args.joint_factors.split(",")]
        assert len(vals) == 7, f"joint_factors 需要7个数, 收到{len(vals)}"
        joint_factors = np.array(vals)
    joint_factors *= args.gravity_factor
    if not np.allclose(joint_factors, args.gravity_factor):
        print(f"逐关节系数: {np.round(joint_factors, 3).tolist()}")

    # 连接臂
    from pyAgxArm import create_agx_arm_config, AgxArmFactory

    print("连接机械臂 (v111 driver)...")
    cfg = create_agx_arm_config(
        robot="nero", comm="can", channel=args.channel,
        interface="socketcan", firmeware_version="v111",
    )
    robot = AgxArmFactory.create_arm(cfg)
    robot.connect()

    # 使能
    robot.set_normal_mode()
    for _ in range(100):
        if robot.enable():
            break
        time.sleep(0.01)
    else:
        print("❌ enable 失败")
        return
    print("✅ 已使能")
    time.sleep(0.5)

    # 检查并清除急停状态
    st = robot.get_arm_status()
    if st and 'EMERGENCY' in str(getattr(st.msg, 'arm_status', '')):
        print("⚠️ 检测到急停状态，复位中...")
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
        st = robot.get_arm_status()
        if st:
            print(f"  复位后: arm_status={getattr(st.msg, 'arm_status', '?')}")
    else:
        print(f"  arm_status=NORMAL ✅")

    # 读初始位置
    ja = robot.get_joint_angles()
    if ja is None:
        print("❌ 无反馈")
        return
    q0 = np.array(ja.msg, dtype=np.float64)
    print(f"起始角(deg): {[rad2deg(v) for v in q0]}")

    # 进入 MIT 模式
    robot.set_auto_set_motion_mode_enabled(False)
    robot.set_motion_mode('mit')
    time.sleep(0.2)
    print("✅ 已切换 MIT 模式")

    # 零重力循环
    dt = 1.0 / args.freq
    kd = ZERO_GRAV_KD * args.kd_scale  # 逐关节 kd
    if args.no_gravity:
        print(f"\n⚠️  无重力补偿模式 (对比用)")
    else:
        print(f"\n=== 零重力模式 ===")
    print(f"  频率: {args.freq}Hz")
    print(f"  kd = ZERO_GRAV_KD × {args.kd_scale} = {np.round(kd, 3).tolist()}")
    print(f"  torque_limit = ±{args.torque_limit} N·m")
    print(f"  gravity_factor = {args.gravity_factor}")
    print(f"  时长: {args.duration}s")
    print(f"  用手拖动臂 → 应该感觉非常轻")
    print(f"  Ctrl+C 停止\n")

    t_start = time.time()
    count = 0

    try:
        while time.time() - t_start < args.duration:
            # 读当前关节角
            ja = robot.get_joint_angles()
            if ja is None:
                continue
            q = np.array(ja.msg, dtype=np.float64)

            # 计算重力补偿力矩（逐关节系数）
            if args.no_gravity:
                tau = np.zeros(7)
            else:
                tau = grav_comp.compute(q[:7]) * joint_factors

            # 力矩安全限幅
            tau = np.clip(tau, -args.torque_limit, args.torque_limit)

            # 发送 MIT 命令 (joint 1-7)
            for i in range(7):
                robot.move_mit(
                    joint_index=i + 1,
                    p_des=float(q[i]),
                    v_des=0.0,
                    kp=0.0,
                    kd=float(kd[i]),
                    t_ff=float(tau[i]),
                )

            count += 1
            # 每 2 秒打印一次
            if count % (args.freq * 2) == 0:
                elapsed = time.time() - t_start
                q_deg = [rad2deg(v) for v in q]
                tau_list = [round(t, 2) for t in tau]
                print(f"  t={elapsed:.0f}s  q(deg)={q_deg}  tau(Nm)={tau_list}")

            time.sleep(dt)

    except KeyboardInterrupt:
        pass

    elapsed = time.time() - t_start
    print(f"\n运行 {elapsed:.1f}s, 发了 {count} 轮")

    # 读最终位置
    ja_final = robot.get_joint_angles()
    if ja_final:
        qf = np.array(ja_final.msg, dtype=np.float64)
        diff = [rad2deg(qf[i]-q0[i]) for i in range(7)]
        print(f"最终角(deg): {[rad2deg(v) for v in qf]}")
        print(f"总变化(deg): {diff}")

    # 停止
    robot.electronic_emergency_stop()
    time.sleep(0.2)
    robot.disable()
    print("已停止")


if __name__ == "__main__":
    main()
