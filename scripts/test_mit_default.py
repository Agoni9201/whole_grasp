#!/usr/bin/env python3
"""测试 MIT 模式 — 用 default driver（和 move_j 一样的驱动）。

先 move_j 到安全位置，再切 MIT 模式测试。

用法：
  conda activate agilexrobot
  python scripts/test_mit_default.py --execute
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

    print("=== 连接 (default driver, 不指定 firmware version) ===")
    cfg = create_agx_arm_config(
        robot="nero", comm="can", channel="can0", interface="socketcan",
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

    # 读位置
    ja = robot.get_joint_angles()
    if ja is None:
        print("❌ 无反馈")
        return
    q0 = np.array(ja.msg, dtype=np.float64)
    print(f"当前角(deg): {[rad2deg(v) for v in q0]}")
    print_status(robot, "初始")

    # 先用 move_j 到一个安全位置（joint2 远离限幅）
    if abs(np.degrees(q0[1])) > 80:
        print("\njoint2 靠近限幅，先 move_j 拉回...")
        safe = q0.copy()
        safe[1] = 0.5
        robot.move_j(safe.tolist())
        time.sleep(4)
        ja = robot.get_joint_angles()
        if ja:
            q0 = np.array(ja.msg, dtype=np.float64)
            print(f"新位置(deg): {[rad2deg(v) for v in q0]}")

    # === MIT 测试 ===
    print("\n" + "="*50)
    print("=== MIT 模式测试 ===")
    print("="*50)

    # Test 1: 手动切 MIT 模式
    print("\n--- 1. set_motion_mode('mit') ---")
    robot.set_motion_mode('mit')
    time.sleep(0.2)
    print_status(robot, "设MIT后")

    # Test 2: 发 MIT 命令，kp=10, kd=0.8 (默认值) — 应该锁住当前位置
    print("\n--- 2. MIT: kp=10, kd=0.8 锁住当前位置 5s ---")
    print("    试着拖动臂，应该有弹性阻力")
    t_start = time.time()
    count = 0
    while time.time() - t_start < 5.0:
        for i in range(6):  # default driver 支持 1-6
            robot.move_mit(
                joint_index=i + 1,
                p_des=float(q0[i]),
                v_des=0.0,
                kp=10.0,
                kd=0.8,
                t_ff=0.0,
            )
        count += 1
        time.sleep(0.005)  # ~200Hz
    print(f"  发了 {count} 轮")

    ja2 = robot.get_joint_angles()
    if ja2:
        q2 = np.array(ja2.msg, dtype=np.float64)
        diff = [rad2deg(q2[i]-q0[i]) for i in range(7)]
        print(f"  变化(deg): {diff}")
        print_status(robot, "MIT后")
        if max(abs(d) for d in diff) > 1.0:
            print("  ✅ 关节有变化（可能被拖动了）")
        else:
            print("  ❌ 完全没动")

    # Test 3: MIT 全零 — 应该能自由拖动
    q_base = q2.copy() if ja2 else q0.copy()
    print("\n--- 3. MIT: kp=0, kd=0, t_ff=0 (零力矩) 5s ---")
    print("    试着拖动臂，应该自由浮动")
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

    # 停止
    print("\n=== 停止 ===")
    robot.electronic_emergency_stop()
    time.sleep(0.2)
    robot.disable()
    print("已停止")


if __name__ == "__main__":
    main()
