#!/usr/bin/env python3
"""零重力诊断 — 分步隔离 kd 和 t_ff 的影响。

测试序列 (每个 5s):
  A. kp=0, kd=0, t_ff=0         → 基线：应该完全自由
  B. kp=0, kd=KD, t_ff=0        → 只加阻尼
  C. kp=0, kd=0, t_ff=tau_grav  → 只加重力补偿
  D. kp=0, kd=KD, t_ff=tau_grav → 完整零重力

用法：
  conda activate agilexrobot
  python scripts/test_zero_gravity_diag.py --execute
"""
import argparse, os, sys, time, numpy as np

ZERO_GRAV_KD = np.array([1.0, 1.0, 1.0, 0.5, 0.3, 0.2, 0.2])
TORQUE_LIMIT = 8.0

def rad2deg(v):
    return round(float(v) * 57.2958, 2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        print("加 --execute 连接实机")
        return

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.normpath(os.path.join(script_dir, ".."))
    sys.path.insert(0, project_root)

    from control_backends.force_ctrl.gravity_comp import NeroGravityCompensator
    from pyAgxArm import create_agx_arm_config, AgxArmFactory

    gc = NeroGravityCompensator()
    print(f"✅ 重力补偿模型 (nq={gc.nq})")

    # 先算一下当前配置的重力力矩
    print("\n--- 先算一个测试位置的重力力矩 ---")
    q_test = np.array([0.0, 1.0, 0.0, 0.5, 0.0, 0.0, 0.0])
    tau_test = gc.compute(q_test)
    print(f"  q=[0,57°,0,29°,0,0,0] → tau={[round(t,3) for t in tau_test]}")

    # 连接
    cfg = create_agx_arm_config(
        robot="nero", comm="can", channel="can0",
        interface="socketcan", firmeware_version="v111",
    )
    robot = AgxArmFactory.create_arm(cfg)
    robot.connect()
    robot.set_normal_mode()
    for _ in range(100):
        if robot.enable():
            break
        time.sleep(0.01)
    print("✅ 已使能")
    time.sleep(0.5)

    # 检查并清除急停状态
    st0 = robot.get_arm_status()
    if st0:
        s = st0.msg
        arm_st = getattr(s, 'arm_status', None)
        print(f"  [enable后] arm_status={arm_st}, "
              f"mode_feedback={getattr(s,'mode_feedback','?')}")
        if 'EMERGENCY' in str(arm_st):
            print("  ⚠️ 检测到急停状态，尝试复位...")
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
            st0 = robot.get_arm_status()
            if st0:
                s = st0.msg
                arm_st = getattr(s, 'arm_status', None)
                print(f"  [复位后] arm_status={arm_st}, "
                      f"mode_feedback={getattr(s,'mode_feedback','?')}")
                if 'NORMAL' in str(arm_st):
                    print("  ✅ 复位成功，arm_status=NORMAL")
                else:
                    print("  ❌ 复位失败，仍然异常")
                    return

    ja = robot.get_joint_angles()
    if ja is None:
        print("❌ 无反馈")
        return
    q0 = np.array(ja.msg, dtype=np.float64)
    print(f"当前角(deg): {[rad2deg(v) for v in q0]}")

    # 算当前重力力矩（未截断）
    tau_raw = gc.compute(q0[:7])
    print(f"重力力矩(未截断): {[round(t,3) for t in tau_raw]}")
    tau_clipped = np.clip(tau_raw, -TORQUE_LIMIT, TORQUE_LIMIT)
    print(f"重力力矩(截断后):  {[round(t,3) for t in tau_clipped]}")

    # 进 MIT
    st_pre = robot.get_arm_status()
    if st_pre:
        print(f"  [切MIT前] arm_status={getattr(st_pre.msg,'arm_status','?')}")

    robot.set_auto_set_motion_mode_enabled(False)
    robot.set_motion_mode('mit')
    time.sleep(0.2)

    st_post = robot.get_arm_status()
    if st_post:
        print(f"  [切MIT后] arm_status={getattr(st_post.msg,'arm_status','?')}, "
              f"mode_feedback={getattr(st_post.msg,'mode_feedback','?')}")

    # 关键诊断：检查模式是否真的切过去了
    st = robot.get_arm_status()
    if st:
        s = st.msg
        print(f"\n⚠️  MIT 模式状态检查:")
        print(f"  ctrl_mode={getattr(s,'ctrl_mode','?')}")
        print(f"  arm_status={getattr(s,'arm_status','?')}")
        print(f"  mode_feedback={getattr(s,'mode_feedback','?')}")
        print(f"  motion_status={getattr(s,'motion_status','?')}")
        mf = getattr(s, 'mode_feedback', None)
        if mf is not None and 'MIT' not in str(mf):
            print(f"  ❌ mode_feedback 不是 MIT！重新尝试切换...")
            robot.set_motion_mode('mit')
            time.sleep(0.5)
            st2 = robot.get_arm_status()
            if st2:
                print(f"  重试后 mode_feedback={getattr(st2.msg,'mode_feedback','?')}")
        else:
            print(f"  ✅ mode_feedback 确认 MIT")
    else:
        print("  ❌ 无法读取 arm_status")

    def send_mit_loop(q, kp, kd, t_ff, duration_s, label):
        """发 MIT 命令 duration_s 秒，返回角度变化"""
        q_before = q.copy()
        t_start = time.time()
        count = 0
        while time.time() - t_start < duration_s:
            # 每次读最新角度
            ja = robot.get_joint_angles()
            if ja is not None:
                q = np.array(ja.msg, dtype=np.float64)
            raise RuntimeError('Legacy MIT diagnostic had a missing command call; use validated SDK hold tools instead')
            count += 1
            time.sleep(0.005)

        ja_after = robot.get_joint_angles()
        q_after = np.array(ja_after.msg, dtype=np.float64) if ja_after else q
        diff = [rad2deg(q_after[i] - q_before[i]) for i in range(7)]
        print(f"  变化(deg): {diff}")
        # 打印模式确认
        st = robot.get_arm_status()
        if st:
            print(f"  mode_feedback={getattr(st.msg,'mode_feedback','?')}")
        return q_after

    # === 诊断测试 ===
    tests = [
        {
            "label": "A. 基线: kp=0, kd=0, t_ff=0",
            "desc": "应该完全自由（无阻尼、无力矩）",
            "kp": np.zeros(7), "kd": np.zeros(7), "t_ff": np.zeros(7),
        },
        {
            "label": "B. 只加阻尼: kp=0, kd=KD, t_ff=0",
            "desc": "应该有阻尼但能拖动，臂会下垂",
            "kp": np.zeros(7), "kd": ZERO_GRAV_KD, "t_ff": np.zeros(7),
        },
        {
            "label": "C. 只加重力: kp=0, kd=0, t_ff=tau_grav",
            "desc": "应该悬浮（无阻尼），推一下会飘",
            "kp": np.zeros(7), "kd": np.zeros(7), "t_ff": tau_clipped,
        },
        {
            "label": "D. 完整零重力: kp=0, kd=KD, t_ff=tau_grav",
            "desc": "应该悬浮+有阻尼，推哪停哪",
            "kp": np.zeros(7), "kd": ZERO_GRAV_KD, "t_ff": tau_clipped,
        },
    ]

    q_cur = q0.copy()
    for t in tests:
        print(f"\n--- {t['label']} ---")
        print(f"  {t['desc']}")
        print(f"  请用手拖动臂！5s 后自动下一项")
        # 更新 t_ff 为当前位置的重力力矩
        tau_now = gc.compute(q_cur[:7])
        tau_now = np.clip(tau_now, -TORQUE_LIMIT, TORQUE_LIMIT)
        if np.any(t["t_ff"] != 0):
            t["t_ff"] = tau_now  # 用当前位置的实时重力力矩
        q_cur = send_mit_loop(q_cur, t["kp"], t["kd"], t["t_ff"], 5.0, t["label"])

    # 停止
    robot.electronic_emergency_stop()
    time.sleep(0.2)
    robot.disable()
    print("\n已停止")


if __name__ == "__main__":
    raise SystemExit("This incomplete legacy diagnostic is disabled; use test_force_position_hold.py")
