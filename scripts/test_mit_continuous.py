#!/usr/bin/env python3
"""诊断测试：连续发 MIT 命令，同时打印 CAN 数据。

用法：
  python scripts/test_mit_continuous.py --execute
  python scripts/test_mit_continuous.py --execute --firmware default
"""
import argparse, inspect, sys, time, numpy as np


def infer_mit_joint_count(robot):
    try:
        param = inspect.signature(robot.move_mit).parameters.get("joint_index")
        args = getattr(param.annotation, "__args__", ()) if param else ()
        joint_ids = [value for value in args if isinstance(value, int)]
        if joint_ids:
            return max(joint_ids)
    except Exception:
        pass
    return 7


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--channel", default="can0")
    parser.add_argument("--firmware", default="v111", choices=["default", "v111"])
    args = parser.parse_args()

    if not args.execute:
        print("加 --execute 连接实机")
        return

    from pyAgxArm import create_agx_arm_config, AgxArmFactory

    print(f"使用驱动: {args.firmware}")
    cfg = create_agx_arm_config(
        robot="nero", comm="can", channel=args.channel,
        interface="socketcan", firmeware_version=args.firmware,
    )
    robot = AgxArmFactory.create_arm(cfg)
    n_joints = infer_mit_joint_count(robot)
    print(f"当前 SDK move_mit 支持 {n_joints} 个关节")
    robot.connect()
    robot.set_normal_mode()
    for _ in range(50):
        if robot.enable():
            break
        time.sleep(0.05)
    print("✅ 已使能")

    time.sleep(0.3)
    ja = robot.get_joint_angles()
    if ja is None:
        print("❌ 无反馈")
        return
    q = np.array(ja.msg, dtype=np.float64)
    print(f"起始角度(deg): {np.round(np.degrees(q), 2).tolist()}")

    # 1. 先看看 enable 后不设模式，能不能拖
    print("\n--- 测试1: enable 后不设模式，尝试拖动 3s ---")
    time.sleep(3.0)
    ja2 = robot.get_joint_angles()
    q2 = np.array(ja2.msg, dtype=np.float64)
    diff1 = np.degrees(q2 - q)
    print(f"变化(deg): {np.round(diff1, 2).tolist()}")
    moved1 = np.max(np.abs(diff1)) > 0.5
    print(f"{'✅ 能拖' if moved1 else '❌ 不能拖'}")

    q = q2.copy()  # 更新基准

    # 2. 设置 MIT 模式，连续发命令
    print(f"\n--- 测试2: set_motion_mode('mit') + 连续 move_mit (200Hz, 5s) ---")
    robot.set_auto_set_motion_mode_enabled(False)
    robot.set_motion_mode('mit')
    time.sleep(0.1)

    t_start = time.time()
    count = 0
    while time.time() - t_start < 5.0:
        for i in range(n_joints):
            robot.move_mit(
                joint_index=i + 1,
                p_des=float(q[i]),
                v_des=0.0,
                kp=0.0,
                kd=0.0,
                t_ff=0.0,
            )
        count += 1
        time.sleep(0.005)  # ~200Hz
    print(f"  发了 {count} 轮 MIT 命令")

    ja3 = robot.get_joint_angles()
    q3 = np.array(ja3.msg, dtype=np.float64)
    diff2 = np.degrees(q3 - q)
    print(f"变化(deg): {np.round(diff2, 2).tolist()}")
    moved2 = np.max(np.abs(diff2)) > 0.5
    print(f"{'✅ 能拖' if moved2 else '❌ 不能拖'}")

    # 3. 加一点 t_ff 看有没有反应
    q = q3.copy()
    print(f"\n--- 测试3: 连续 move_mit, t_ff=0.5 N·m (5s) ---")
    robot.set_motion_mode('mit')
    t_start = time.time()
    while time.time() - t_start < 5.0:
        for i in range(n_joints):
            robot.move_mit(
                joint_index=i + 1,
                p_des=float(q[i]),
                v_des=0.0,
                kp=0.0,
                kd=0.0,
                t_ff=0.5,
            )
        time.sleep(0.005)

    ja4 = robot.get_joint_angles()
    q4 = np.array(ja4.msg, dtype=np.float64)
    diff3 = np.degrees(q4 - q)
    print(f"变化(deg): {np.round(diff3, 2).tolist()}")
    moved3 = np.max(np.abs(diff3)) > 0.5
    print(f"{'✅ 有力矩输出' if moved3 else '❌ 无力矩输出'}")

    # 停止
    robot.electronic_emergency_stop()
    time.sleep(0.1)
    robot.disable()
    print("\n已停止")


if __name__ == "__main__":
    main()
