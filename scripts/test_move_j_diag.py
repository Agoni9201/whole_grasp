#!/usr/bin/env python3
"""最小 move_j 诊断 — 先清错误再测试。

用法：
  conda activate agilexrobot
  python scripts/test_move_j_diag.py --execute
"""
import argparse
import time


def rad2deg(v):
    return round(v * 57.2958, 2)


def print_status(robot, label=""):
    status = robot.get_arm_status()
    if status is not None:
        s = status.msg
        print(f"  [{label}]")
        print(f"    ctrl_mode={getattr(s, 'ctrl_mode', '?')}")
        print(f"    arm_status={getattr(s, 'arm_status', '?')}")
        print(f"    mode_feedback={getattr(s, 'mode_feedback', '?')}")
        print(f"    motion_status={getattr(s, 'motion_status', '?')}")
        print(f"    teach_status={getattr(s, 'teach_status', '?')}")
        print(f"    trajectory_num={getattr(s, 'trajectory_num', '?')}")
        print(f"    err_status={getattr(s, 'err_status', '?')}")
        # 打印原始字节看看
        raw = getattr(s, '_err_code', getattr(s, 'err_code', '?'))
        print(f"    err_code(raw)={raw}")
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

    print("=== 连接 ===")
    cfg = create_agx_arm_config(
        robot="nero", comm="can", channel="can0", interface="socketcan",
    )
    robot = AgxArmFactory.create_arm(cfg)
    robot.connect()

    # 先看看初始状态
    print_status(robot, "连接后")

    # 清错误
    print("\n=== 清错误 ===")
    robot.electronic_emergency_stop()
    time.sleep(0.3)
    robot.reset()
    time.sleep(0.3)
    robot.clear_joint_error(255)
    time.sleep(0.3)
    print_status(robot, "清错后")

    # 使能
    print("\n=== 使能 ===")
    robot.set_normal_mode()
    for i in range(100):
        if robot.enable():
            break
        time.sleep(0.01)
    else:
        print("❌ enable 失败")
        return
    print("✅ enable 成功")
    time.sleep(0.5)
    print_status(robot, "使能后")

    # 读初始位置
    ja = robot.get_joint_angles()
    if ja is None:
        print("❌ 无反馈")
        return
    q0 = list(ja.msg)
    print(f"初始角(deg): {[rad2deg(v) for v in q0]}")

    # Step 1: 如果 joint2 靠近限幅 (±100°)，先往回拉
    j2_deg = rad2deg(q0[1])
    if abs(j2_deg) > 80:
        print(f"\n⚠️  joint2={j2_deg}° 靠近限幅(±100°)，先往回拉")
        safe = list(q0)
        safe[1] = 0.5  # 拉到 ~28.6°
        robot.move_j(safe)
        time.sleep(0.5)
        t0 = time.monotonic()
        while time.monotonic() - t0 < 8.0:
            status = robot.get_arm_status()
            if status and getattr(status.msg, 'motion_status', None) == 0:
                print("✅ joint2 已拉回安全位置")
                break
            time.sleep(0.1)
        else:
            print("⏰ 拉回超时")
        time.sleep(0.3)
        # 更新 q0
        ja = robot.get_joint_angles()
        if ja:
            q0 = list(ja.msg)
            print(f"新位置(deg): {[rad2deg(v) for v in q0]}")
        print_status(robot, "拉回后")

    # Step 2: move_j 测试: joint1 +0.15 rad
    print("\n=== move_j: joint1 +0.15 rad ===")
    target = list(q0)
    target[0] += 0.15
    print(f"目标(deg): {[rad2deg(v) for v in target]}")
    robot.move_j(target)

    # 等待
    time.sleep(3.0)
    ja1 = robot.get_joint_angles()
    if ja1:
        q1 = list(ja1.msg)
        diff = [rad2deg(q1[i]-q0[i]) for i in range(7)]
        print(f"到达(deg): {[rad2deg(v) for v in q1]}")
        print(f"变化(deg): {diff}")
        if abs(diff[0]) > 0.5:
            print("✅ joint1 移动了！")
        else:
            print("❌ joint1 没动")
    print_status(robot, "move_j后")

    # 回原位
    print("\n=== move_j: 回原位 ===")
    robot.move_j(q0)
    time.sleep(3.0)
    ja2 = robot.get_joint_angles()
    if ja2:
        q2 = list(ja2.msg)
        diff2 = [rad2deg(q2[i]-q1[i]) for i in range(7)]
        print(f"到达(deg): {[rad2deg(v) for v in q2]}")
        print(f"变化(deg): {diff2}")
    print_status(robot, "回原位后")

    # 再来一次
    print("\n=== move_j: joint1 +0.15 rad (第2次) ===")
    robot.move_j(target)
    time.sleep(3.0)
    ja3 = robot.get_joint_angles()
    if ja3:
        q3 = list(ja3.msg)
        diff3 = [rad2deg(q3[i]-q2[i]) for i in range(7)]
        print(f"到达(deg): {[rad2deg(v) for v in q3]}")
        print(f"变化(deg): {diff3}")
        if abs(diff3[0]) > 0.5:
            print("✅ 第2次也动了！")
        else:
            print("❌ 第2次没动")
    print_status(robot, "第2次后")

    # 停止
    print("\n=== 停止 ===")
    robot.disable()
    print("已 disable")


if __name__ == "__main__":
    main()
