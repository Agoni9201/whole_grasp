#!/usr/bin/env python3
"""测试 move_j 关节位置控制 — 验证 SDK CAN 通信是否正常。

用法：
  conda activate agilexrobot
  python scripts/test_move_j.py --execute
"""
import argparse
import time
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--channel", default="can0")
    args = parser.parse_args()

    if not args.execute:
        print("加 --execute 连接实机")
        return

    from pyAgxArm import create_agx_arm_config, AgxArmFactory

    print("连接机械臂...")
    cfg = create_agx_arm_config(
        robot="nero", comm="can",
        channel=args.channel, interface="socketcan",
    )
    robot = AgxArmFactory.create_arm(cfg)
    robot.connect()

    # Enable
    robot.set_normal_mode()
    for _ in range(50):
        if robot.enable():
            break
        time.sleep(0.05)
    print("✅ 已使能")

    time.sleep(0.3)

    # 读当前关节角
    ja = robot.get_joint_angles()
    if ja is None:
        print("❌ 无反馈")
        return
    q = np.array(ja.msg, dtype=np.float64)
    print(f"当前关节角(deg): {np.round(np.degrees(q), 2).tolist()}")

    # 设置速度
    robot.set_speed_percent(20)
    print("速度设为 20%")

    # Test 1: move_j 小幅度移动 — 选一个远离限幅的关节
    # joint2 当前可能接近限幅，改为 joint1 ±0.2 rad
    target = q.copy()
    delta = 0.2  # rad
    # 如果 joint1 接近限幅就反向
    if target[0] + delta > 2.5:
        delta = -0.2
    target[0] += delta

    print(f"\n--- 测试: move_j joint1 {delta:+.1f}rad ---")
    print(f"目标(deg): {np.round(np.degrees(target), 2).tolist()}")
    robot.move_j(target.tolist())

    # 等待运动完成
    time.sleep(0.5)
    t0 = time.monotonic()
    arrived = False
    while time.monotonic() - t0 < 5.0:
        status = robot.get_arm_status()
        if status is not None:
            ms = getattr(status.msg, 'motion_status', None)
            if ms == 0:
                arrived = True
                break
        time.sleep(0.1)

    if arrived:
        print("✅ 运动完成 (motion_status=0)")
    else:
        print("⏰ 5秒超时，可能没到达")

    time.sleep(0.3)
    ja2 = robot.get_joint_angles()
    if ja2 is not None:
        q2 = np.array(ja2.msg, dtype=np.float64)
        diff = np.degrees(q2 - q)
        print(f"到达关节角(deg): {np.round(np.degrees(q2), 2).tolist()}")
        print(f"变化量(deg):       {np.round(diff, 2).tolist()}")
        if abs(diff[1]) > 1.0:
            print("✅ joint2 有明显移动，move_j 正常工作！")
        else:
            print("❌ joint2 没有移动，move_j 可能没生效")
    else:
        print("❌ 无反馈")

    # Test 2: 移回原位
    print(f"\n--- 测试: move_j 回原位 ---")
    robot.move_j(q.tolist())
    time.sleep(0.5)
    t0 = time.monotonic()
    while time.monotonic() - t0 < 5.0:
        status = robot.get_arm_status()
        if status is not None and getattr(status.msg, 'motion_status', None) == 0:
            print("✅ 已回到原位")
            break
        time.sleep(0.1)

    # 停止
    robot.electronic_emergency_stop()
    time.sleep(0.1)
    robot.disable()
    print("\n已停止")


if __name__ == "__main__":
    main()
