#!/usr/bin/env python3
"""测试 MIT 模式 — v111 驱动 (firmware 1.11)。

先用 move_j 验证 v111 驱动通信正常，再测 MIT。

用法：
  conda activate agilexrobot
  python scripts/test_mit_v111.py --execute
"""
import argparse
import time
import numpy as np


def rad2deg(v):
    return round(v * 57.2958, 2)


def print_status(robot, label=""):
    status = robot.get_arm_status()
    if status is not None:
        s = status.msg
        print(f"  [{label}] ctrl_mode={getattr(s,'ctrl_mode','?')}, "
              f"arm_status={getattr(s,'arm_status','?')}, "
              f"mode_feedback={getattr(s,'mode_feedback','?')}, "
              f"motion_status={getattr(s,'motion_status','?')}")
    else:
        print(f"  [{label}] 无状态反馈")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    if not args.execute:
        print("加 --execute 连接实机")
        return

    from pyAgxArm import create_agx_arm_config, AgxArmFactory

    print("=== 连接 (v111 driver, firmware 1.11) ===")
    cfg = create_agx_arm_config(
        robot="nero", comm="can", channel="can0", interface="socketcan",
        firmeware_version="v111",
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
    print("✅ enable 成功")
    time.sleep(0.5)
    print_status(robot, "使能后")

    # 读位置
    ja = robot.get_joint_angles()
    if ja is None:
        print("❌ 无反馈")
        return
    q0 = np.array(ja.msg, dtype=np.float64)
    print(f"当前角(deg): {[rad2deg(v) for v in q0]}")

    print(f"  当前角(deg): {[rad2deg(v) for v in q0]}")

    # === MIT 测试 ===
    print("\n" + "="*50)
    print("=== MIT 模式测试 (v111 driver) ===")
    print("="*50)

    # 关闭自动模式设置，手动控制
    robot.set_auto_set_motion_mode_enabled(False)
    print("已关闭 auto_set_motion_mode")

    # 手动设 MIT 模式
    print("\n--- 1. set_motion_mode('mit') ---")
    robot.set_motion_mode('mit')
    time.sleep(0.3)
    print_status(robot, "设MIT后")

    # MIT Test 1: kp=10, kd=0.8 锁住当前位置
    print("\n--- 2. MIT: kp=10, kd=0.8 锁住 (5s, 200Hz) ---")
    print("    试着拖动臂 → 应该有弹性阻力")
    t_start = time.time()
    count = 0
    while time.time() - t_start < 5.0:
        for i in range(6):
            robot.move_mit(
                joint_index=i + 1,
                p_des=float(q0[i]),
                v_des=0.0,
                kp=10.0,
                kd=0.8,
                t_ff=0.0,
            )
        count += 1
        time.sleep(0.005)
    print(f"  发了 {count} 轮 MIT 命令")

    ja2 = robot.get_joint_angles()
    if ja2:
        q2 = np.array(ja2.msg, dtype=np.float64)
        diff = [rad2deg(q2[i]-q0[i]) for i in range(7)]
        print(f"  变化(deg): {diff}")
        print_status(robot, "MIT锁住后")

    # MIT Test 2: 零力矩
    q_base = q2.copy() if ja2 else q0.copy()
    print("\n--- 3. MIT: kp=0, kd=0, t_ff=0 零力矩 (5s) ---")
    print("    试着拖动臂 → 应该自由浮动")
    robot.set_motion_mode('mit')
    t_start = time.time()
    count = 0
    while time.time() - t_start < 5.0:
        for i in range(6):
            robot.move_mit(
                joint_index=i + 1,
                p_des=float(q_base[i]),
                v_des=0.0,
                kp=0.0,
                kd=0.0,
                t_ff=0.0,
            )
        count += 1
        time.sleep(0.005)
    print(f"  发了 {count} 轮")

    ja3 = robot.get_joint_angles()
    if ja3:
        q3 = np.array(ja3.msg, dtype=np.float64)
        diff3 = [rad2deg(q3[i]-q_base[i]) for i in range(7)]
        print(f"  变化(deg): {diff3}")
        print_status(robot, "零力矩后")

    # MIT Test 3: 加 t_ff 看有没有力矩输出
    q_base2 = q3.copy() if ja3 else q_base.copy()
    print("\n--- 4. MIT: kp=0, kd=0, t_ff=0.5 N·m (5s) ---")
    print("    臂应该自己动")
    robot.set_motion_mode('mit')
    t_start = time.time()
    count = 0
    while time.time() - t_start < 5.0:
        for i in range(6):
            robot.move_mit(
                joint_index=i + 1,
                p_des=float(q_base2[i]),
                v_des=0.0,
                kp=0.0,
                kd=0.0,
                t_ff=0.5,
            )
        count += 1
        time.sleep(0.005)
    print(f"  发了 {count} 轮")

    ja4 = robot.get_joint_angles()
    if ja4:
        q4 = np.array(ja4.msg, dtype=np.float64)
        diff4 = [rad2deg(q4[i]-q_base2[i]) for i in range(7)]
        print(f"  变化(deg): {diff4}")
        if max(abs(d) for d in diff4[:6]) > 1.0:
            print("  ✅ 有力矩输出，MIT 生效！")
        else:
            print("  ❌ 没有力矩输出")
        print_status(robot, "t_ff后")

    # 停止
    print("\n=== 停止 ===")
    robot.electronic_emergency_stop()
    time.sleep(0.2)
    robot.disable()
    print("已停止")


if __name__ == "__main__":
    main()
