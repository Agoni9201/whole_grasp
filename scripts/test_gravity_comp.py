#!/usr/bin/env python3
"""Nero 带夹爪重力补偿实机测试。

三个阶段：
  1. 零重力模式 — 人手拖动，验证补偿方向正确
  2. 位置保持 — 10s 内臂不下掉
  3. 单关节微动 — 每次只动一个关节，验证方向

用法：
  # Dry-run（只打印，不连接实机）
  python scripts/test_gravity_comp.py

  # 实机执行（必须加 --execute）
  python scripts/test_gravity_comp.py --execute --channel can0 --firmware v111

  # 只跑零重力测试
  python scripts/test_gravity_comp.py --execute --mode zero_gravity

  # 自定义重力因子
  python scripts/test_gravity_comp.py --execute --gravity-factor 0.9

安全注意：
  - 测试前确认急停可用
  - 默认 dry-run，必须加 --execute 才会真机执行
  - Ctrl+C 随时急停
"""

import argparse
import inspect
import sys
import os
import time
import signal
import numpy as np

# Add project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from control_backends.force_ctrl.controller import NeroForceController
from control_backends.force_ctrl.data_types import JointState


# 默认参数
DEFAULT_KP = np.array([30.0, 30.0, 30.0, 15.0, 10.0, 5.0, 5.0])
DEFAULT_KD = np.array([2.0, 2.0, 2.0, 1.0, 0.5, 0.3, 0.3])


def print_joint(label, q, unit="rad"):
    """Pretty print joint angles."""
    deg = np.degrees(q)
    text = "  ".join(f"J{i+1}:{d:7.2f}°" for i, d in enumerate(deg))
    print(f"  [{label}] {text}")


def print_separator(char="="):
    print(char * 60)


def _enum_text(value):
    if value is None:
        return "None"
    name = getattr(value, "name", None)
    try:
        raw = int(value)
    except Exception:
        raw = value
    return f"{name}({raw})" if name else str(raw)


def _array_text(values):
    if values is None:
        return "None"
    return "[" + ", ".join(f"{float(v): .3f}" for v in values) + "]"


def _robot_arm_status_summary(robot):
    try:
        st = robot.get_arm_status()
    except Exception as exc:
        return f"读取失败 ({exc})"
    if st is None:
        return "无反馈"
    msg = st.msg
    return (
        f"ctrl={_enum_text(getattr(msg, 'ctrl_mode', None))}, "
        f"arm={_enum_text(getattr(msg, 'arm_status', None))}, "
        f"mode={_enum_text(getattr(msg, 'mode_feedback', None))}, "
        f"teach={_enum_text(getattr(msg, 'teach_status', None))}, "
        f"motion={_enum_text(getattr(msg, 'motion_status', None))}"
    )


def _arm_status_summary(ctrl):
    return _robot_arm_status_summary(ctrl.robot)


def print_robot_diagnostics(ctrl, label):
    """Print SDK and robot status fields needed to debug locked joints."""
    robot = ctrl.robot
    print_separator("-")
    print(f"诊断: {label}")

    try:
        import pyAgxArm
        print(f"  pyAgxArm: {getattr(pyAgxArm, '__file__', 'unknown')}")
        print(f"  SDK版本:  {getattr(pyAgxArm, '__version__', 'unknown')}")
    except Exception as exc:
        print(f"  pyAgxArm: 无法读取 ({exc})")

    try:
        print(f"  Driver:   {robot.__class__.__module__}.{robot.__class__.__name__}")
        print(f"  move_mit: {inspect.signature(robot.move_mit)}")
    except Exception as exc:
        print(f"  Driver:   无法读取 ({exc})")

    print(f"  MIT关节数: {getattr(ctrl, 'mit_joint_count', 'unknown')}")
    print(f"  MIT循环运行: {getattr(ctrl._loop, 'running', 'unknown')}")
    print(f"  MIT循环错误: {getattr(ctrl._loop, 'last_error', None)}")
    try:
        dbg = ctrl.get_debug_state()
        print(f"  Gravity enabled: {dbg['gravity_enabled']}")
        print(f"  Gravity torque:  {_array_text(dbg['gravity_torque'])}")
        print(f"  Command torque:  {_array_text(dbg['total_torque'])}")
        print(f"  Feedback torque: {_array_text(dbg['feedback_torque'])}")
    except Exception as exc:
        print(f"  Torque debug: 读取失败 ({exc})")

    try:
        fw = robot.get_firmware(timeout=0.5, min_interval=0.0)
        print(f"  Firmware反馈: {fw}")
    except Exception as exc:
        print(f"  Firmware反馈: 读取失败 ({exc})")

    try:
        print(f"  Enable列表: {robot.get_joints_enable_status_list()}")
    except Exception as exc:
        print(f"  Enable列表: 读取失败 ({exc})")

    try:
        st = robot.get_arm_status()
        if st is None:
            print("  ArmStatus: 无反馈")
        else:
            msg = st.msg
            print(
                "  ArmStatus: "
                f"ctrl={_enum_text(getattr(msg, 'ctrl_mode', None))}, "
                f"arm={_enum_text(getattr(msg, 'arm_status', None))}, "
                f"mode={_enum_text(getattr(msg, 'mode_feedback', None))}, "
                f"teach={_enum_text(getattr(msg, 'teach_status', None))}, "
                f"motion={_enum_text(getattr(msg, 'motion_status', None))}"
            )
            err = getattr(msg, "err_status", None)
            if err is not None:
                print(f"  ErrStatus: {err}")
    except Exception as exc:
        print(f"  ArmStatus: 读取失败 ({exc})")
    print_separator("-")


def test_zero_gravity(ctrl, duration=15.0):
    """阶段1: 零重力模式测试。

    通过标准：
    - 使能后无冲击
    - 不主动运动
    - 人手轻拖方向正常
    """
    print_separator()
    print("阶段1: 零重力模式测试")
    print(f"  持续 {duration:.0f}s，请用手拖动机械臂验证")
    print("  通过标准: 无冲击、可自由拖动、松手后缓慢下落")
    print("  Ctrl+C 提前结束")
    print_separator()

    ctrl.zero_gravity_mode()
    start = time.time()
    try:
        while time.time() - start < duration:
            q = ctrl.get_joint_pos()
            if q is not None:
                elapsed = time.time() - start
                # Check if MIT loop is alive
                if not ctrl._loop.running:
                    print(f"\n  ❌ MIT 控制循环已停止: {ctrl._loop.last_error}")
                    break
                print(f"\r  [{elapsed:5.1f}s] ", end="")
                print_joint("q", q)
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n  用户中断")
    print("  零重力测试结束")


def test_zero_torque(ctrl, duration=5.0):
    """Pure MIT passive test: kp=kd=t_ff=0 and no gravity overlay."""
    print_separator()
    print("诊断: 纯零力矩 MIT 测试")
    print(f"  持续 {duration:.0f}s，请扶住机械臂并尝试轻拖")
    print("  通过标准: 关节不再主动抱住；若仍完全拖不动，问题不在重力补偿")
    print("  注意: 无重力补偿，机械臂可能下坠")
    print_separator()

    ctrl.zero_torque_mode()
    start = time.time()
    try:
        while time.time() - start < duration:
            q = ctrl.get_joint_pos()
            if q is not None:
                elapsed = time.time() - start
                if not ctrl._loop.running:
                    print(f"\n  ❌ MIT 控制循环已停止: {ctrl._loop.last_error}")
                    break
                print(f"\r  [{elapsed:5.1f}s] ", end="")
                print_joint("q", q)
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n  用户中断")
    print("  纯零力矩测试结束")


def test_torque_probe(ctrl, joint_index=6, torque=0.8, duration=3.0):
    """Probe whether the MIT t_ff torque channel changes joint motion."""
    print_separator()
    print("诊断: MIT t_ff 力矩通道探测")
    print(f"  关节: J{joint_index}")
    print(f"  t_ff: {torque:.3f} N·m")
    print(f"  持续: {duration:.1f}s")
    print("  命令: kp=0, kd=0, 重力补偿关闭")
    print("  请扶住机械臂；若关节明显运动，随时 Ctrl+C 急停")
    print_separator()

    q_start = ctrl.get_joint_pos()
    if q_start is None:
        print("  ❌ 无法读取关节角度")
        return

    torques = np.zeros(7)
    torques[joint_index - 1] = torque
    zeros = np.zeros(7)

    print_joint("起始", q_start)
    ctrl.command_joint_state(
        pos=q_start.copy(),
        vel=zeros,
        kp=zeros,
        kd=zeros,
        torques=torques,
        gravity_enabled=False,
    )

    q_last = q_start.copy()
    probe_debug = None
    probe_status = None
    start = time.time()
    try:
        while time.time() - start < duration:
            q_now = ctrl.get_joint_pos()
            if q_now is not None:
                q_last = q_now.copy()
                elapsed = time.time() - start
                delta_deg = np.degrees(q_now - q_start)
                probe_delta = delta_deg[joint_index - 1]
                max_delta = np.max(np.abs(delta_deg))
                if not ctrl._loop.running:
                    print(f"\n  ❌ MIT 控制循环已停止: {ctrl._loop.last_error}")
                    break
                print(
                    f"\r  [{elapsed:5.1f}s] "
                    f"J{joint_index} Δ={probe_delta: .3f}°  "
                    f"max|Δ|={max_delta: .3f}°  ",
                    end="",
                )
                sys.stdout.flush()
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n  用户中断")
    finally:
        try:
            probe_debug = ctrl.get_debug_state()
        except Exception:
            probe_debug = None
        probe_status = _arm_status_summary(ctrl)
        ctrl.zero_torque_mode()
        time.sleep(0.2)

    print()
    print_joint("结束", q_last)
    delta = q_last - q_start
    print(f"  J{joint_index} 角度变化: {np.degrees(delta[joint_index - 1]):.3f}°")
    print(f"  最大关节变化: {np.max(np.abs(np.degrees(delta))):.3f}°")
    print(f"  探测期间 ArmStatus: {probe_status}")
    if probe_debug is not None:
        print(f"  探测期间 Gravity enabled: {probe_debug['gravity_enabled']}")
        print(f"  探测期间 Command torque:  {_array_text(probe_debug['total_torque'])}")
        print(f"  探测期间 Feedback torque: {_array_text(probe_debug['feedback_torque'])}")
    print("  t_ff 力矩通道探测结束")


def test_leader_drag(ctrl, duration=10.0):
    """Probe firmware built-in zero-force drag (leader) mode."""
    print_separator()
    print("诊断: 固件 leader 零力拖动模式")
    print(f"  持续 {duration:.0f}s，请轻轻拖动机械臂验证")
    print("  命令: 停止 MIT 循环，调用 robot.set_leader_mode()")
    print("  结束后会切回 normal mode")
    print_separator()

    ctrl._loop.stop()
    time.sleep(0.1)
    ctrl.robot.set_leader_mode()
    time.sleep(0.5)
    print(f"  Leader ArmStatus: {_arm_status_summary(ctrl)}")

    start = time.time()
    q_start = ctrl.get_joint_pos()
    q_last = q_start.copy() if q_start is not None else None
    if q_start is not None:
        print_joint("起始", q_start)

    try:
        while time.time() - start < duration:
            q_now = ctrl.get_joint_pos()
            if q_now is not None:
                q_last = q_now.copy()
                elapsed = time.time() - start
                if q_start is None:
                    print(f"\r  [{elapsed:5.1f}s] ", end="")
                    print_joint("q", q_now)
                else:
                    delta_deg = np.degrees(q_now - q_start)
                    print(
                        f"\r  [{elapsed:5.1f}s] max|Δ|={np.max(np.abs(delta_deg)): .3f}°  ",
                        end="",
                    )
                    sys.stdout.flush()
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n  用户中断")
    finally:
        ctrl.robot.set_normal_mode()
        time.sleep(0.2)

    print()
    if q_last is not None:
        print_joint("结束", q_last)
        if q_start is not None:
            delta = q_last - q_start
            print(f"  最大关节变化: {np.max(np.abs(np.degrees(delta))):.3f}°")
    print(f"  Normal ArmStatus: {_arm_status_summary(ctrl)}")
    print("  leader 零力拖动测试结束")


def test_sdk_torque_probe(args):
    """Raw SDK MIT torque probe without NeroForceController or MIT loop."""
    from pyAgxArm import create_agx_arm_config, AgxArmFactory

    print_separator()
    print("诊断: 官方 SDK 原生 MIT t_ff 探测")
    print(f"  Channel: {args.channel}")
    print(f"  Firmware: {args.firmware}")
    print(f"  关节: J{args.probe_joint}")
    print(f"  t_ff: {args.probe_torque:.3f} N·m")
    print(f"  持续: {min(args.duration, 10.0):.1f}s")
    print("  不启动项目控制器/重力补偿/后台 MIT 循环")
    print_separator()

    cfg = create_agx_arm_config(
        robot="nero",
        comm="can",
        channel=args.channel,
        interface="socketcan",
        firmeware_version=args.firmware,
    )
    robot = AgxArmFactory.create_arm(cfg)
    robot.connect()

    try:
        robot.reset()
        time.sleep(0.2)
        robot.set_normal_mode()
        timeout = time.time() + 5.0
        while not robot.enable():
            if time.time() > timeout:
                raise RuntimeError("Failed to enable Nero arm within 5 seconds")
            time.sleep(0.01)
        time.sleep(0.5)
        print(f"  enable后 ArmStatus: {_robot_arm_status_summary(robot)}")
        print("  enable后 reset + normal...")
        robot.reset()
        time.sleep(0.5)
        robot.set_normal_mode()
        time.sleep(0.5)
        print(f"  enable后复位 ArmStatus: {_robot_arm_status_summary(robot)}")

        q_start = None
        timeout = time.time() + 3.0
        while q_start is None:
            ja = robot.get_joint_angles()
            if ja is not None:
                q_start = np.array(ja.msg, dtype=np.float64)
            if time.time() > timeout:
                raise RuntimeError("No joint feedback received within 3 seconds")
            time.sleep(0.02)

        robot.set_motion_mode(robot.OPTIONS.MOTION_MODE.MIT)
        time.sleep(0.2)
        print(f"  启动 ArmStatus: {_robot_arm_status_summary(robot)}")
        print_joint("起始", q_start)

        q_last = q_start.copy()
        torques = np.zeros(7)
        duration = min(args.duration, 10.0)
        start = time.time()
        while time.time() - start < duration:
            ja = robot.get_joint_angles()
            if ja is not None:
                q_last = np.array(ja.msg, dtype=np.float64)

            for i in range(7):
                t_ff = args.probe_torque if i == args.probe_joint - 1 else 0.0
                robot.move_mit(
                    joint_index=i + 1,
                    p_des=float(q_start[i]),
                    v_des=0.0,
                    kp=0.0,
                    kd=0.0,
                    t_ff=float(t_ff),
                )

            for i in range(7):
                ms = robot.get_motor_states(i + 1)
                if ms is not None:
                    torques[i] = ms.msg.torque

            elapsed = time.time() - start
            delta_deg = np.degrees(q_last - q_start)
            print(
                f"\r  [{elapsed:5.1f}s] "
                f"J{args.probe_joint} Δ={delta_deg[args.probe_joint - 1]: .3f}°  "
                f"max|Δ|={np.max(np.abs(delta_deg)): .3f}°  "
                f"tau={_array_text(torques)}  ",
                end="",
            )
            sys.stdout.flush()
            time.sleep(0.02)

        print()
        print_joint("结束", q_last)
        delta = q_last - q_start
        print(f"  J{args.probe_joint} 角度变化: {np.degrees(delta[args.probe_joint - 1]):.3f}°")
        print(f"  最大关节变化: {np.max(np.abs(np.degrees(delta))):.3f}°")
        print(f"  探测 ArmStatus: {_robot_arm_status_summary(robot)}")
        print(f"  探测 Feedback torque: {_array_text(torques)}")

    finally:
        try:
            for i in range(7):
                robot.move_mit(
                    joint_index=i + 1,
                    p_des=0.0,
                    v_des=0.0,
                    kp=0.0,
                    kd=0.0,
                    t_ff=0.0,
                )
            time.sleep(0.1)
        except Exception:
            pass
        try:
            timeout = time.time() + 2.0
            while not robot.disable():
                if time.time() > timeout:
                    break
                time.sleep(0.01)
        except Exception:
            pass
        print("  SDK 原生探测结束，已 disable")


def test_sdk_position_probe(args):
    """Raw SDK MIT position probe without NeroForceController or MIT loop."""
    from pyAgxArm import create_agx_arm_config, AgxArmFactory

    delta = float(args.probe_delta)
    print_separator()
    print("诊断: 官方 SDK 原生 MIT 位置小步探测")
    print(f"  Channel: {args.channel}")
    print(f"  Firmware: {args.firmware}")
    print(f"  关节: J{args.probe_joint}")
    print(f"  目标增量: {delta:.4f} rad ({np.degrees(delta):.2f}°)")
    print(f"  kp/kd: {args.probe_kp:.2f}/{args.probe_kd:.2f}")
    print(f"  持续: {min(args.duration, 10.0):.1f}s")
    print("  不启动项目控制器/重力补偿/后台 MIT 循环")
    print_separator()

    cfg = create_agx_arm_config(
        robot="nero",
        comm="can",
        channel=args.channel,
        interface="socketcan",
        firmeware_version=args.firmware,
    )
    robot = AgxArmFactory.create_arm(cfg)
    robot.connect()

    try:
        robot.reset()
        time.sleep(0.2)
        robot.set_normal_mode()
        timeout = time.time() + 5.0
        while not robot.enable():
            if time.time() > timeout:
                raise RuntimeError("Failed to enable Nero arm within 5 seconds")
            time.sleep(0.01)

        q_start = None
        timeout = time.time() + 3.0
        while q_start is None:
            ja = robot.get_joint_angles()
            if ja is not None:
                q_start = np.array(ja.msg, dtype=np.float64)
            if time.time() > timeout:
                raise RuntimeError("No joint feedback received within 3 seconds")
            time.sleep(0.02)

        q_target = q_start.copy()
        q_target[args.probe_joint - 1] += delta
        robot.set_motion_mode(robot.OPTIONS.MOTION_MODE.MIT)
        time.sleep(0.2)
        print(f"  启动 ArmStatus: {_robot_arm_status_summary(robot)}")
        print_joint("起始", q_start)
        print_joint("目标", q_target)

        q_last = q_start.copy()
        torques = np.zeros(7)
        duration = min(args.duration, 10.0)
        start = time.time()
        while time.time() - start < duration:
            ja = robot.get_joint_angles()
            if ja is not None:
                q_last = np.array(ja.msg, dtype=np.float64)

            for i in range(7):
                if i == args.probe_joint - 1:
                    p_des = q_target[i]
                    kp = args.probe_kp
                    kd = args.probe_kd
                else:
                    p_des = q_start[i]
                    kp = 0.0
                    kd = 0.0
                robot.move_mit(
                    joint_index=i + 1,
                    p_des=float(p_des),
                    v_des=0.0,
                    kp=float(kp),
                    kd=float(kd),
                    t_ff=0.0,
                )

            for i in range(7):
                ms = robot.get_motor_states(i + 1)
                if ms is not None:
                    torques[i] = ms.msg.torque

            elapsed = time.time() - start
            delta_deg = np.degrees(q_last - q_start)
            print(
                f"\r  [{elapsed:5.1f}s] "
                f"J{args.probe_joint} Δ={delta_deg[args.probe_joint - 1]: .3f}°  "
                f"max|Δ|={np.max(np.abs(delta_deg)): .3f}°  "
                f"tau={_array_text(torques)}  ",
                end="",
            )
            sys.stdout.flush()
            time.sleep(0.02)

        print()
        print_joint("结束", q_last)
        actual_delta = q_last - q_start
        print(f"  J{args.probe_joint} 角度变化: {np.degrees(actual_delta[args.probe_joint - 1]):.3f}°")
        print(f"  最大关节变化: {np.max(np.abs(np.degrees(actual_delta))):.3f}°")
        print(f"  探测 ArmStatus: {_robot_arm_status_summary(robot)}")
        print(f"  探测 Feedback torque: {_array_text(torques)}")

    finally:
        try:
            for i in range(7):
                robot.move_mit(
                    joint_index=i + 1,
                    p_des=0.0,
                    v_des=0.0,
                    kp=0.0,
                    kd=0.0,
                    t_ff=0.0,
                )
            time.sleep(0.1)
        except Exception:
            pass
        try:
            timeout = time.time() + 2.0
            while not robot.disable():
                if time.time() > timeout:
                    break
                time.sleep(0.01)
        except Exception:
            pass
        print("  SDK 原生位置探测结束，已 disable")


def test_sdk_move_j_probe(args):
    """Raw SDK move_j probe to verify ordinary joint position control."""
    from pyAgxArm import create_agx_arm_config, AgxArmFactory

    delta = float(args.probe_delta)
    print_separator()
    print("诊断: 官方 SDK 普通 move_j 小步探测")
    print(f"  Channel: {args.channel}")
    print(f"  Firmware: {args.firmware}")
    print(f"  关节: J{args.probe_joint}")
    print(f"  目标增量: {delta:.4f} rad ({np.degrees(delta):.2f}°)")
    print(f"  持续: {min(args.duration, 10.0):.1f}s")
    print("  不使用 MIT；只验证普通关节位置控制是否能动")
    print_separator()

    cfg = create_agx_arm_config(
        robot="nero",
        comm="can",
        channel=args.channel,
        interface="socketcan",
        firmeware_version=args.firmware,
    )
    robot = AgxArmFactory.create_arm(cfg)
    robot.connect()

    try:
        robot.reset()
        time.sleep(0.2)
        robot.set_normal_mode()
        robot.set_speed_percent(10)
        robot.set_auto_set_motion_mode_enabled(True)
        timeout = time.time() + 5.0
        while not robot.enable():
            if time.time() > timeout:
                raise RuntimeError("Failed to enable Nero arm within 5 seconds")
            time.sleep(0.01)

        q_start = None
        timeout = time.time() + 3.0
        while q_start is None:
            ja = robot.get_joint_angles()
            if ja is not None:
                q_start = np.array(ja.msg, dtype=np.float64)
            if time.time() > timeout:
                raise RuntimeError("No joint feedback received within 3 seconds")
            time.sleep(0.02)

        q_target = q_start.copy()
        q_target[args.probe_joint - 1] += delta
        print(f"  Auto motion mode: {robot.get_auto_set_motion_mode_enabled()}")
        print(f"  启动 ArmStatus: {_robot_arm_status_summary(robot)}")
        robot.set_motion_mode(robot.OPTIONS.MOTION_MODE.J)
        time.sleep(0.5)
        print(f"  显式 set_motion_mode(J) 后: {_robot_arm_status_summary(robot)}")
        print_joint("起始", q_start)
        print_joint("目标", q_target)

        robot.move_j(q_target.tolist())
        time.sleep(0.2)
        print(f"\n  move_j() 后 ArmStatus: {_robot_arm_status_summary(robot)}")

        q_last = q_start.copy()
        torques = np.zeros(7)
        duration = min(args.duration, 10.0)
        start = time.time()
        while time.time() - start < duration:
            ja = robot.get_joint_angles()
            if ja is not None:
                q_last = np.array(ja.msg, dtype=np.float64)
            for i in range(7):
                ms = robot.get_motor_states(i + 1)
                if ms is not None:
                    torques[i] = ms.msg.torque
            elapsed = time.time() - start
            delta_deg = np.degrees(q_last - q_start)
            print(
                f"\r  [{elapsed:5.1f}s] "
                f"J{args.probe_joint} Δ={delta_deg[args.probe_joint - 1]: .3f}°  "
                f"max|Δ|={np.max(np.abs(delta_deg)): .3f}°  "
                f"tau={_array_text(torques)}  ",
                end="",
            )
            sys.stdout.flush()
            time.sleep(0.05)

        print()
        print_joint("结束", q_last)
        actual_delta = q_last - q_start
        print(f"  J{args.probe_joint} 角度变化: {np.degrees(actual_delta[args.probe_joint - 1]):.3f}°")
        print(f"  最大关节变化: {np.max(np.abs(np.degrees(actual_delta))):.3f}°")
        print(f"  探测 ArmStatus: {_robot_arm_status_summary(robot)}")
        print(f"  探测 Feedback torque: {_array_text(torques)}")

    finally:
        try:
            timeout = time.time() + 2.0
            while not robot.disable():
                if time.time() > timeout:
                    break
                time.sleep(0.01)
        except Exception:
            pass
        print("  SDK 普通 move_j 探测结束，已 disable")


def test_sdk_enable_probe(args):
    """Raw SDK enable/brake state probe."""
    from pyAgxArm import create_agx_arm_config, AgxArmFactory

    print_separator()
    print("诊断: 官方 SDK 使能/抱闸状态探测")
    print(f"  Channel: {args.channel}")
    print(f"  Firmware: {args.firmware}")
    print("  只发 reset/normal/enable，不发运动命令")
    print_separator()

    cfg = create_agx_arm_config(
        robot="nero",
        comm="can",
        channel=args.channel,
        interface="socketcan",
        firmeware_version=args.firmware,
    )
    robot = AgxArmFactory.create_arm(cfg)
    robot.connect()

    def print_enable_snapshot(label):
        try:
            enable_list = robot.get_joints_enable_status_list()
        except Exception as exc:
            enable_list = f"读取失败 ({exc})"
        print(f"  {label} ArmStatus: {_robot_arm_status_summary(robot)}")
        print(f"  {label} Enable列表: {enable_list}")

    try:
        print_enable_snapshot("连接后")
        robot.reset()
        time.sleep(0.3)
        print_enable_snapshot("reset后")
        robot.set_normal_mode()
        time.sleep(0.5)
        print_enable_snapshot("normal后")

        print("  全关节 enable(255)...")
        for _ in range(50):
            if robot.enable():
                break
            time.sleep(0.05)
        time.sleep(1.0)
        print_enable_snapshot("全关节enable后")
        print("  enable后 reset + normal...")
        robot.reset()
        time.sleep(0.5)
        robot.set_normal_mode()
        time.sleep(0.5)
        print_enable_snapshot("enable后复位")

        print("  逐关节 enable(1..7)...")
        for joint_index in range(1, 8):
            ok = robot.enable(joint_index)
            time.sleep(0.2)
            print(f"    J{joint_index} enable返回: {ok}")
            print_enable_snapshot(f"J{joint_index}后")

        print("  观察 3s...")
        start = time.time()
        while time.time() - start < 3.0:
            print_enable_snapshot(f"{time.time() - start:4.1f}s")
            time.sleep(0.5)

    finally:
        try:
            timeout = time.time() + 2.0
            while not robot.disable():
                if time.time() > timeout:
                    break
                time.sleep(0.01)
        except Exception:
            pass
        print("  SDK 使能/抱闸探测结束，已 disable")


def test_position_hold(ctrl, duration=10.0):
    """阶段2: 位置保持测试。

    通过标准：
    - 10s 内 TCP 不下掉
    - q2/q4 不漂移（< 0.5°）
    """
    print_separator()
    print("阶段2: 位置保持测试")
    print(f"  保持当前位置 {duration:.0f}s")
    print("  通过标准: 各关节漂移 < 0.5°")
    print_separator()

    # 切换到位置保持
    q = ctrl.get_joint_pos()
    if q is None:
        print("  ❌ 无法读取关节角度")
        return

    q_start = q.copy()
    print_joint("起始", q_start)
    ctrl.command_joint_pos(q_start)

    start = time.time()
    max_drift = np.zeros(7)
    try:
        while time.time() - start < duration:
            q_now = ctrl.get_joint_pos()
            if q_now is not None:
                elapsed = time.time() - start
                drift_deg = np.abs(np.degrees(q_now - q_start))
                max_drift = np.maximum(max_drift, drift_deg)
                print(f"\r  [{elapsed:5.1f}s] max_drift={np.max(drift_deg):.3f}°  ", end="")
                sys.stdout.flush()
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n  用户中断")

    print()
    print_joint("最大漂移(°)", max_drift)
    passed = np.max(max_drift) < 0.5
    status = "✅ 通过" if passed else "❌ 未通过"
    print(f"  {status}: 最大漂移 {np.max(max_drift):.3f}°")


def test_joint_nudge(ctrl):
    """阶段3: 单关节微动测试。

    通过标准：
    - 每次只动一个关节
    - 方向正确
    - 无明显振荡
    """
    print_separator()
    print("阶段3: 单关节微动测试")
    print("  每个关节依次微动 +0.02 rad (~1.1°)")
    print_separator()

    # 先用较小的增益
    kp = DEFAULT_KP * 0.5
    kd = DEFAULT_KD.copy()

    q = ctrl.get_joint_pos()
    if q is None:
        print("  ❌ 无法读取关节角度")
        return

    for joint_idx in range(7):
        q_target = q.copy()
        delta = 0.02  # radians
        q_target[joint_idx] += delta

        print(f"\n  关节{joint_idx+1}: +{np.degrees(delta):.1f}°")
        print_joint("  当前", q)
        print_joint("  目标", q_target)

        ctrl.command_joint_pos(q_target, kp=kp, kd=kd)
        time.sleep(1.5)

        q_new = ctrl.get_joint_pos()
        if q_new is not None:
            actual_delta = q_new[joint_idx] - q[joint_idx]
            correct = actual_delta > 0.005  # 至少动了 0.005 rad
            print(f"  实际变化: {np.degrees(actual_delta):.2f}°  "
                  f"{'✅ 方向正确' if correct else '❌ 方向异常'}")
            q = q_new.copy()
        else:
            print("  ❌ 无法读取反馈")

        # 回到原位
        ctrl.command_joint_pos(q, kp=kp, kd=kd)
        time.sleep(1.0)
        q = ctrl.get_joint_pos() or q


def main():
    parser = argparse.ArgumentParser(
        description="Nero 带夹爪重力补偿实机测试"
    )
    parser.add_argument(
        "--execute", action="store_true",
        help="实机执行（默认 dry-run 只打印参数）"
    )
    parser.add_argument("--channel", default="can0", help="CAN 通道")
    parser.add_argument(
        "--firmware",
        choices=["default", "v111"],
        default="v111",
        help="Nero 固件驱动版本；固件 >= 1.11 用 v111 (默认 v111)"
    )
    parser.add_argument(
        "--mode",
        choices=["zero_torque", "torque_probe", "sdk_torque_probe", "sdk_position_probe", "sdk_move_j_probe", "sdk_enable_probe", "leader_drag", "zero_gravity", "position_hold", "joint_nudge", "all"],
        default="all",
        help="测试模式 (默认 all)"
    )
    parser.add_argument(
        "--gravity-factor", type=float, default=1.0,
        help="重力补偿因子 (默认 1.0，可调 0.8~1.2)"
    )
    parser.add_argument(
        "--torque-limit", type=float, default=8.0,
        help="最大力矩限制 N·m (默认 8.0)"
    )
    parser.add_argument(
        "--mit-joint-count",
        type=int,
        choices=[6, 7],
        default=None,
        help="覆盖 MIT 发送关节数；默认根据当前 SDK 自动判断"
    )
    parser.add_argument(
        "--duration", type=float, default=10.0,
        help="位置保持测试时长 (秒，默认 10)"
    )
    parser.add_argument(
        "--probe-joint",
        type=int,
        choices=range(1, 8),
        default=6,
        help="torque_probe 模式下探测的关节编号 (默认 6)"
    )
    parser.add_argument(
        "--probe-torque",
        type=float,
        default=0.8,
        help="torque_probe 模式下发送的 t_ff N·m (默认 0.8)"
    )
    parser.add_argument(
        "--probe-delta",
        type=float,
        default=0.03,
        help="sdk_position_probe 模式下 Jn 目标增量 rad (默认 0.03)"
    )
    parser.add_argument(
        "--probe-kp",
        type=float,
        default=8.0,
        help="sdk_position_probe 模式下 MIT kp (默认 8.0)"
    )
    parser.add_argument(
        "--probe-kd",
        type=float,
        default=0.8,
        help="sdk_position_probe 模式下 MIT kd (默认 0.8)"
    )
    args = parser.parse_args()

    # 项目根目录下的 URDF
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    urdf_path = os.path.join(project_root, "models", "nero_with_gripper.urdf")

    print_separator()
    print("Nero 带夹爪重力补偿测试")
    print_separator()
    print(f"  URDF:     {urdf_path}")
    print(f"  Channel:  {args.channel}")
    print(f"  Firmware: {args.firmware}")
    print(f"  Grav因子: {args.gravity_factor}")
    print(f"  力矩限制: {args.torque_limit} N·m")
    print(f"  MIT关节数: {args.mit_joint_count or 'auto'}")
    print(f"  测试模式: {args.mode}")
    if args.mode in ("torque_probe", "sdk_torque_probe", "sdk_position_probe", "sdk_move_j_probe"):
        print(f"  探测关节: J{args.probe_joint}")
    if args.mode in ("torque_probe", "sdk_torque_probe"):
        print(f"  探测力矩: {args.probe_torque} N·m")
    if args.mode in ("sdk_position_probe", "sdk_move_j_probe"):
        print(f"  探测增量: {args.probe_delta} rad")
    if args.mode == "sdk_position_probe":
        print(f"  探测增益: kp={args.probe_kp}, kd={args.probe_kd}")
    print(f"  实机执行: {'是 ⚠️' if args.execute else '否 (dry-run)'}")

    if not os.path.isfile(urdf_path):
        print(f"\n❌ URDF 不存在: {urdf_path}")
        return 1

    if not args.execute:
        print("\n  [dry-run] 只验证参数，不连接实机")
        # 验证重力补偿可以计算
        from control_backends.force_ctrl.gravity_comp import NeroGravityCompensator
        grav = NeroGravityCompensator(urdf_path=urdf_path)
        q_test = np.array([0.0, 0.5, 0.0, 0.5, 0.0, 0.0, 0.0])
        tau = grav.compute(q_test)
        print(f"  重力补偿测试 (q=[0,0.5,0,0.5,0,0,0]):")
        print(f"    nq={grav.nq}")
        print(f"    tau={[f'{t:.3f}' for t in tau[:7]]}")
        print(f"\n  加 --execute 参数连接实机执行")
        return 0

    # ===== 实机执行 =====
    print("\n⚠️  即将连接实机！确认：")
    print("  1. 急停按钮可用")
    print("  2. 机械臂周围无障碍物")
    print("  3. CAN 接口已启动 (sudo ip link set can0 up type can bitrate 1000000)")
    input("\n  按 Enter 继续，Ctrl+C 取消...")

    if args.mode == "sdk_torque_probe":
        test_sdk_torque_probe(args)
        return 0

    if args.mode == "sdk_position_probe":
        test_sdk_position_probe(args)
        return 0

    if args.mode == "sdk_move_j_probe":
        test_sdk_move_j_probe(args)
        return 0

    if args.mode == "sdk_enable_probe":
        test_sdk_enable_probe(args)
        return 0

    ctrl = NeroForceController(
        channel=args.channel,
        urdf_path=urdf_path,
        zero_gravity=True,
        gravity_factor=args.gravity_factor,
        torque_limit=args.torque_limit,
        firmware=args.firmware,
        mit_joint_count=args.mit_joint_count,
    )

    def signal_handler(sig, frame):
        print("\n\n⚠️  Ctrl+C 急停！")
        ctrl.emergency_stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    try:
        print("\n连接机械臂...")
        ctrl.start()
        print("✅ 机械臂已连接并启动零重力模式")
        print_robot_diagnostics(ctrl, "启动后")

        if args.mode == "zero_torque":
            test_zero_torque(ctrl, duration=min(args.duration, 10.0))
            print_robot_diagnostics(ctrl, "纯零力矩测试后")

        if args.mode == "torque_probe":
            test_torque_probe(
                ctrl,
                joint_index=args.probe_joint,
                torque=args.probe_torque,
                duration=min(args.duration, 10.0),
            )
            print_robot_diagnostics(ctrl, "MIT t_ff 力矩探测后")

        if args.mode == "leader_drag":
            test_leader_drag(ctrl, duration=min(args.duration, 15.0))
            print_robot_diagnostics(ctrl, "leader 零力拖动测试后")

        if args.mode in ("zero_gravity", "all"):
            test_zero_gravity(ctrl, duration=min(args.duration, 15.0))
            print_robot_diagnostics(ctrl, "零重力测试后")

        if args.mode in ("position_hold", "all"):
            test_position_hold(ctrl, duration=args.duration)

        if args.mode in ("joint_nudge", "all"):
            test_joint_nudge(ctrl)

    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\n停止控制...")
        ctrl.stop()
        print("✅ 已安全停止")

    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
