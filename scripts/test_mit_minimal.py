#!/usr/bin/env python3
"""最小 MIT 模式测试 — 直接用官方 SDK，不经过我们的 force_ctrl。

用法：
  python scripts/test_mit_minimal.py --execute
"""

import argparse
import inspect
import sys
import time
import signal
import numpy as np


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
    args = parser.parse_args()

    if not args.execute:
        print("加 --execute 连接实机")
        return

    from pyAgxArm import create_agx_arm_config, AgxArmFactory

    print("连接机械臂...")
    cfg = create_agx_arm_config(robot="nero", comm="can",
                                channel=args.channel, interface="socketcan",
                                firmeware_version="v111")
    robot = AgxArmFactory.create_arm(cfg)
    n_joints = infer_mit_joint_count(robot)
    print(f"当前 SDK move_mit 支持 {n_joints} 个关节")
    robot.connect()

    # Enable
    robot.set_normal_mode()
    for _ in range(50):
        if robot.enable():
            break
        time.sleep(0.05)
    print("✅ 已使能")

    # 等反馈
    time.sleep(0.3)
    ja = robot.get_joint_angles()
    if ja is None:
        print("❌ 无反馈")
        return
    q = np.array(ja.msg, dtype=np.float64)
    print(f"当前关节角(deg): {np.round(np.degrees(q), 2).tolist()}")

    # 关掉自动模式设置
    robot.set_auto_set_motion_mode_enabled(False)
    print("已关闭 auto_set_motion_mode")

    # Step 1: 手动设 MIT 模式
    print("\n设置 MIT 模式...")
    robot.set_motion_mode('mit')
    time.sleep(0.2)

    # Step 2: 发 MIT 命令 — 全零（纯重力补偿都没有，臂应该能动）
    print("发送 MIT 命令: kp=0, kd=0, t_ff=0 (完全无力矩)")
    print("尝试拖动机械臂... 5秒后读角度")
    for i in range(n_joints):
        robot.move_mit(
            joint_index=i + 1,
            p_des=float(q[i]),
            v_des=0.0,
            kp=0.0,
            kd=0.0,
            t_ff=0.0,
        )

    time.sleep(5.0)
    ja2 = robot.get_joint_angles()
    if ja2 is not None:
        q2 = np.array(ja2.msg, dtype=np.float64)
        diff = np.degrees(q2 - q)
        print(f"5秒后关节角(deg): {np.round(np.degrees(q2), 2).tolist()}")
        print(f"变化量(deg):       {np.round(diff, 2).tolist()}")
        if np.max(np.abs(diff)) > 0.5:
            print("✅ 关节有变化，MIT 模式生效")
        else:
            print("❌ 关节没变化，MIT 模式可能没生效")
    else:
        print("❌ 无反馈")

    # 停止
    robot.electronic_emergency_stop()
    time.sleep(0.1)
    robot.disable()
    print("已停止")


if __name__ == "__main__":
    main()
