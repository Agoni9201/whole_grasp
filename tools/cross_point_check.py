"""Supervised cross-point round trip and translation compensation.

This is not an autonomous contact controller or a global collision planner.
Only the point-test --cross-check mode uses it.
"""
import hashlib
import json
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.spatial.transform import Rotation


def pose_transform(pose):
    pose = np.asarray(pose, dtype=float).reshape(6)
    if not np.isfinite(pose).all():
        raise ValueError("Non-finite pose")
    transform = np.eye(4)
    transform[:3, :3] = Rotation.from_euler("xyz", pose[3:]).as_matrix()
    transform[:3, 3] = pose[:3]
    return transform


def tcp_transform(flange_pose, offset):
    return pose_transform(flange_pose) @ pose_transform(offset)


def orientation_error_deg(first, second):
    return float(np.rad2deg(Rotation.from_matrix(
        first[:3, :3].T @ second[:3, :3]
    ).magnitude()))


def corrected_translation(flange_camera, observation_flange, predicted, taught):
    from tools.calibrate_realsense_touch import check_transform, point3
    result = check_transform(flange_camera).copy()
    observation_flange = check_transform(observation_flange)
    delta_base = point3(taught) - point3(predicted)
    result[:3, 3] += observation_flange[:3, :3].T @ delta_base
    return result


def write_new(path, data):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def fresh_pose(arm):
    wrapped = arm.arm.get_flange_pose()
    if wrapped is None:
        raise RuntimeError("No flange feedback")
    stamp = float(getattr(wrapped, "timestamp", float("nan")))
    if not np.isfinite(stamp) or not 0 <= time.time() - stamp <= 0.5:
        raise RuntimeError("No fresh timestamped flange feedback")
    pose = np.asarray(getattr(wrapped, "msg", wrapped), dtype=float).reshape(6)
    return pose, pose_transform(pose), stamp


def record_taught_point(arm, args, state):
    from tools.calibrate_realsense_touch import stationary_feedback
    if state.get("target_pose") is None:
        raise ValueError("先用 c 采集并点击叉叉")
    path = Path(state["out_dir"]) / "cross_comparison.json"
    if path.exists():
        raise FileExistsError("本次画面已记录过示教点；重新 c 采集，不覆盖旧结果")
    print("自行示教，让同一已定义TCP接触部位轻触同一叉叉；夹爪开度不变，示教朝向可以不同。")
    if input("实际已对准并轻触、停稳，输入 TOUCH 记录；Enter 取消：").strip() != "TOUCH":
        return
    transforms, stamps = stationary_feedback(SimpleNamespace(read=lambda: fresh_pose(arm)))
    offset = pose_transform(args.tcp_offset)
    tcps = np.array([(transform @ offset)[:3, 3] for transform in transforms])
    taught = np.median(tcps, axis=0)
    angle = orientation_error_deg(pose_transform(state["flange_pose"]), transforms[-1])
    if angle > 3.0:
        print(f"[cross][warn] 示教姿态变化 {angle:.2f}deg，仅记录、不拒绝；请确认仍是同一TCP接触点。")
    delta = taught - np.asarray(state["P_base"], dtype=float)
    comparison = dict(
        created_at=datetime.now().astimezone().isoformat(),
        calibration_path=str(Path(args.calib_json).resolve()),
        calibration_sha256=state["calibration_sha256"],
        T_flange_camera=state["T_flange_camera"].tolist(),
        T_base_flange_observation=pose_transform(state["flange_pose"]).tolist(),
        T_base_flange_teach_samples=[transform.tolist() for transform in transforms],
        teach_flange_timestamps=stamps, tcp_offset_m_rad=list(args.tcp_offset),
        pixel=list(state["result"]["pixel"]), P_camera_m=state["P_camera"].tolist(),
        P_base_visual_m=state["P_base"].tolist(), P_base_taught_tcp_m=taught.tolist(),
        correction_base_mm=(delta * 1000).tolist(),
        visual_minus_taught_mm=(-delta * 1000).tolist(),
        error_norm_mm=float(np.linalg.norm(delta) * 1000),
        teach_attitude_difference_deg=angle,
        teach_sample_spread_mm=float(np.linalg.norm(tcps-taught, axis=1).max()*1000),
        correction_scope="single_point_translation_only",
        tolerance_mm=5.0, within_5mm=bool(np.linalg.norm(delta) <= 0.005),
    )
    last_execution = state.get("cross_last_execution")
    if last_execution is not None:
        comparison["last_execution"] = last_execution
        comparison["executed_minus_taught_mm"] = (
            (np.asarray(last_execution["actual_tcp_xyz_m"])-taught)*1000
        ).tolist()
        comparison["executed_minus_visual_mm"] = (
            (np.asarray(last_execution["actual_tcp_xyz_m"])-state["P_base"])*1000
        ).tolist()
    if state.get("cross_roundtrip") is not None:
        comparison["roundtrip_status"] = state["cross_roundtrip"]["status"]
    write_new(path, comparison)
    state["cross_comparison"] = comparison
    print("视觉 - 示教 TCP XYZ_mm =", np.round(-delta*1000, 2).tolist())
    print(f"空间误差 = {np.linalg.norm(delta)*1000:.2f} mm")
    print("需要补偿的基座 XYZ_mm =", np.round(delta*1000, 2).tolist())
    if last_execution is not None:
        print("上次实际执行 - 示教 TCP XYZ_mm =",
              np.round(comparison["executed_minus_taught_mm"], 2).tolist(),
              "；动作类型 =", last_execution["purpose"])
        print("执行跟踪偏差（实际执行 - 视觉）XYZ_mm =",
              np.round(comparison["executed_minus_visual_mm"], 2).tolist())
    print("结果：", path, "；输入 k 可导出平移补偿候选，原文件不改。")


def export_compensation(args, state):
    comparison = state.get("cross_comparison")
    if comparison is None:
        raise ValueError("先用 r 记录同一叉叉的示教轻触 TCP")
    if hashlib.sha256(Path(args.calib_json).read_bytes()).hexdigest() != comparison["calibration_sha256"]:
        raise ValueError("原始标定文件已改变，请重新采集比较，不叠加到另一套参数上")
    old = np.array(comparison["T_flange_camera"])
    corrected = corrected_translation(
        old, comparison["T_base_flange_observation"],
        comparison["P_base_visual_m"], comparison["P_base_taught_tcp_m"],
    )
    pos = corrected[:3, 3].tolist()
    quat = Rotation.from_matrix(corrected[:3, :3]).as_quat().tolist()
    result = dict(
        created_at=datetime.now().astimezone().isoformat(),
        status="single_point_candidate_requires_new_capture_validation",
        transform="camera_to_flange", parent_frame="flange", child_frame="d435_color_optical_frame",
        position=pos, orientation=quat, translation_m=pos, quaternion_xyzw=quat,
        matrix_4x4=corrected.tolist(), flange_camera_xyzw=pos+quat,
        source_calibration=comparison["calibration_path"],
        source_calibration_sha256=comparison["calibration_sha256"],
        translation_correction_flange_mm=((corrected[:3, 3]-old[:3, 3])*1000).tolist(),
        rotation_changed=False, comparison=comparison,
        note="SDK intrinsics unchanged. One cross constrains translation only; "
             "this fit does not establish accuracy at other poses or points.",
    )
    path = Path(state["out_dir"]) / "calibration_cross_compensated.json"
    write_new(path, result)
    print("补偿后相机到法兰平移_mm =", np.round(corrected[:3, 3]*1000, 3).tolist())
    print("候选文件：", path)
    print("旋转不变，未修改 pipeline.yaml 或原始标定。请用新画面验证，不以拟合点零残差宣告精度。")
    return path


def _jog_kinematics():
    from kinematic_trac_ik import Kinematic
    from real_nero_grasp.paths import DEFAULT_IK_URDF
    from real_nero_grasp.collision import _urdf_chain_geometry
    params = SimpleNamespace(ik_urdf=DEFAULT_IK_URDF, ik_base_link="base_link", ik_tip_link="link7")
    solver = Kinematic(urdf_path=DEFAULT_IK_URDF, base_link_name="base_link",
                       target_link_name="link7", timeout=0.1, epsilon=1e-5, solver_type="Distance")
    return solver, lambda q: _urdf_chain_geometry(q, params)[1]


def _bounded_jog_plan(start, target, q_start, offset, solver, fk, *, max_distance=0.050, step_m=0.001):
    distance = float(np.linalg.norm(target[:3, 3]-start[:3, 3]))
    if not np.isfinite(distance) or not 0 < step_m <= 0.001:
        raise ValueError("Invalid point-test distance or step")
    if distance > max_distance:
        raise ValueError("当前TCP离目标超过50 mm；先自行示教到叉叉上方附近，再执行局部点测")
    if orientation_error_deg(start, target) > 3.0:
        raise ValueError("当前姿态与采集姿态不同；自行恢复相同工具姿态，不自动翻转")
    if orientation_error_deg(start, target) > 1e-5:
        # Never rotate the tool for a point check, even within the readout tolerance.
        target = target.copy()
        target[:3, :3] = start[:3, :3]
    q = np.asarray(q_start, float).reshape(7)
    low, high = [np.asarray(value).reshape(7) for value in solver.get_joint_limits()]
    if (not np.isfinite(q).all() or np.any(q < low-1e-6) or np.any(q > high+1e-6)):
        raise ValueError("当前关节不在URDF范围内；拒绝点动")
    initial_flange = fk(q)
    modeled_start = initial_flange @ offset
    if (np.linalg.norm(modeled_start[:3, 3]-start[:3, 3]) > 0.002
            or orientation_error_deg(modeled_start, start) > 0.5):
        raise ValueError("FK与当前反馈不一致，拒绝点动")
    plan = []
    count = max(1, int(np.ceil(distance/step_m-1e-9)))
    for index in range(1, count+1):
        fraction = index/count
        tcp = start.copy()
        tcp[:3, 3] = start[:3, 3] + fraction*(target[:3, 3]-start[:3, 3])
        flange = tcp @ np.linalg.inv(offset)
        xyzw = Rotation.from_matrix(flange[:3, :3]).as_quat()
        solved = solver.solve_ik(target_position=flange[:3, 3], target_wxyz=xyzw[[3, 0, 1, 2]],
                                 initial_guess=q.copy(), use_previous_solution=False)
        if solved is None:
            raise ValueError("局部IK无解；没有发送新运动")
        next_q = np.asarray(solved, float).reshape(7)
        if (not np.isfinite(next_q).all() or np.any(next_q < low-1e-6) or np.any(next_q > high+1e-6)
                or np.rad2deg(np.max(np.abs(next_q-q))) > 2.0):
            raise ValueError("局部IK越界或单步关节变化超过2deg，拒绝翻转／大幅运动")
        for alpha in (0.0, 0.5, 1.0):
            modeled = fk(q + alpha*(next_q-q)) @ offset
            if (np.linalg.norm(modeled[:3, 3]-tcp[:3, 3]) > 0.002
                    or orientation_error_deg(modeled, tcp) > 0.5):
                raise ValueError("局部关节插值偏离预期毫米级TCP路径，拒绝点动")
        plan.append((next_q.copy(), tcp.copy()))
        q = next_q
        if index % 8 == 0:
            # TRAC-IK/FK work shares this process with the SDK CAN receiver.
            time.sleep(0)
    return plan


def _preflight_joint_segment(q_start, q_target, start, target, offset, solver, fk, args, phase):
    """Sample the joint interpolation for checking only, never as commands."""
    from tools.real_nero_point_test import assert_workspace
    q_start, q_target = [np.asarray(q, float).reshape(7) for q in (q_start, q_target)]
    low, high = [np.asarray(q, float).reshape(7) for q in solver.get_joint_limits()]
    if (not np.isfinite([q_start, q_target]).all()
            or np.any(np.minimum(q_start, q_target) < low-1e-6)
            or np.any(np.maximum(q_start, q_target) > high+1e-6)):
        raise ValueError(f"{phase}: 当前关节或目标IK超出URDF范围；没有发送运动")
    sweep = float(np.rad2deg(np.max(np.abs(q_target-q_start))))
    if sweep > 90.0:
        raise ValueError(f"{phase}: 单轴关节变化 {sweep:.2f}deg > 90deg，拒绝大幅翻转")
    for q, expected in ((q_start, start), (q_target, target)):
        modeled = fk(q) @ offset
        if (np.linalg.norm(modeled[:3, 3]-expected[:3, 3]) > 0.002
                or orientation_error_deg(modeled, expected) > 0.5):
            raise ValueError(f"{phase}: FK与阶段端点不一致；没有发送运动")
    step_m = float(getattr(args, "cross_step_mm", 1.0))/1000
    if not np.isfinite(step_m) or not 0 < step_m <= 0.001:
        raise ValueError("插值预检采样间距必须在0至1 mm之间")
    distance = float(np.linalg.norm(target[:3, 3]-start[:3, 3]))
    count = max(2, int(np.ceil(sweep/0.5)), int(np.ceil(distance/step_m)))
    while count <= 8192:
        poses = []
        for index, alpha in enumerate(np.linspace(0, 1, count+1)):
            tcp = fk(q_start+alpha*(q_target-q_start)) @ offset
            if not np.isfinite(tcp).all():
                raise ValueError(f"{phase}: 插值FK无效；没有发送运动")
            assert_workspace(tcp[:3, 3], args)
            if tcp[2, 3] < min(start[2, 3], target[2, 3])-0.002:
                raise ValueError(f"{phase}: 关节插值低于两个端点高度，拒绝未经确认的下探")
            if orientation_error_deg(tcp, start) > 3.0:
                raise ValueError(f"{phase}: 关节插值工具朝向变化超过3deg")
            poses.append(tcp)
            if index % 32 == 0:
                time.sleep(0)
        positions = np.array([tcp[:3, 3] for tcp in poses])
        if np.linalg.norm(np.diff(positions, axis=0), axis=1).max() <= step_m*1.001:
            return dict(phase=phase, sample_count=count+1, max_joint_delta_deg=sweep,
                        min_tcp_z_m=float(positions[:, 2].min()),
                        sampled_path_length_m=float(np.linalg.norm(np.diff(positions, axis=0), axis=1).sum()))
        count *= 2
    raise ValueError(f"{phase}: 插值预检未收敛；没有发送运动")


def plan_cross_roundtrip(start, point, q_start, offset, solver, fk, args):
    """Solve exactly two endpoint IKs; move_j handles each joint interpolation."""
    from tools.real_nero_point_test import assert_workspace
    point = np.asarray(point, float).reshape(3)
    if not np.isfinite(point).all() or not np.isfinite(args.hover) or args.hover < 0.03:
        raise ValueError("叉叉点必须有限，悬停高度至少30 mm")
    q_start = np.asarray(q_start, float).reshape(7)
    start = np.asarray(start, float).reshape(4, 4)
    if not np.isfinite(start).all():
        raise ValueError("观察位无效")
    hover, goal = start.copy(), start.copy()
    hover[:3, 3] = point+[0, 0, args.hover]
    goal[:3, 3] = point
    length = float(np.linalg.norm(hover[:3, 3]-start[:3, 3])+args.hover)
    if not np.isfinite(length) or length > 0.5:
        raise ValueError(f"点测去程 {length*1000:.1f}mm 超过500mm；请换近一些的观察位重新c采图")
    for tcp in (start, hover, goal):
        assert_workspace(tcp[:3, 3], args)
    current, joints = start.copy(), q_start.copy()
    phases, checks = [], []
    for phase, destination in (("to_hover", hover), ("approach", goal)):
        flange = destination @ np.linalg.inv(offset)
        xyzw = Rotation.from_matrix(flange[:3, :3]).as_quat()
        solved = solver.solve_ik(target_position=flange[:3, 3], target_wxyz=xyzw[[3, 0, 1, 2]],
                                 initial_guess=joints.copy(), use_previous_solution=False)
        if solved is None:
            raise ValueError(f"{phase}: 目标IK无解；没有发送运动")
        target_q = np.asarray(solved, float).reshape(7)
        checks.append(_preflight_joint_segment(joints, target_q, current, destination,
                                               offset, solver, fk, args, phase))
        phases.append([(target_q.copy(), destination.copy())])
        joints, current = target_q, destination
    checked_length = sum(check["sampled_path_length_m"] for check in checks)
    if checked_length > 0.5:
        raise ValueError("关节插值实际TCP去程超过500mm；没有发送运动")
    outbound = phases[0]+phases[1]

    def reverse(nodes):
        originals = [(q_start, start)]+nodes
        return [(q.copy(), tcp.copy()) for q, tcp in reversed(originals[:-1])]

    return dict(to_hover=phases[0], approach=phases[1],
                return_from_hover=reverse(phases[0]),
                **{"return": reverse(outbound)}, start_tcp=start.copy(),
                start_joints=q_start.copy(), outbound_length_m=checked_length, checks=checks)


def _assert_motion_mode(arm):
    if getattr(arm, "_safety_stop_latched", False):
        raise RuntimeError("停机已锁存，不发送新的运动／返程")
    reader = getattr(arm.arm, "get_arm_status", None)
    if callable(reader):
        wrapped = reader()
        status = getattr(wrapped, "msg", wrapped)
        if "TEACH" in str(getattr(status, "ctrl_mode", "")).upper():
            raise ValueError("当前仍在示教模式；先结束示教再m，不自动切换控制模式")


def _joint_sample(arm):
    reader = getattr(arm.arm, "get_joint_angles", None)
    if callable(reader):
        wrapped = reader()
        stamp = float(getattr(wrapped, "timestamp", float("nan")))
        if not np.isfinite(stamp) or not 0 <= time.time()-stamp <= 0.5:
            raise RuntimeError("无新鲜带时间戳的关节反馈")
        joints = np.asarray(getattr(wrapped, "msg", wrapped), float).reshape(7)
    else:
        joints = np.asarray(arm.get_joint_angles(), float).reshape(7)
        stamp = time.time()
    if not np.isfinite(joints).all():
        raise RuntimeError("无有效关节反馈")
    return joints, stamp


def _finite_joints(arm):
    return _joint_sample(arm)[0]


def _arrival_tolerances(args):
    values = (float(getattr(args, "move_joint_tol", np.deg2rad(0.5))),
              float(getattr(args, "move_pos_tol", 0.002)),
              float(getattr(args, "move_rpy_tol", np.deg2rad(1.0))))
    if not np.isfinite(values).all() or min(values) <= 0:
        raise ValueError("阶段到位容差必须有限且大于0")
    return float(np.rad2deg(values[0])), values[1], float(np.rad2deg(values[2]))


def _assert_at_node(arm, joints, tcp, offset, *, args=None):
    actual_q = _finite_joints(arm)
    _, actual_flange, _ = fresh_pose(arm)
    actual_tcp = actual_flange @ offset
    joint_error = float(np.rad2deg(np.max(np.abs(actual_q-joints))))
    position_error = float(np.linalg.norm(actual_tcp[:3, 3]-tcp[:3, 3]))
    joint_tol, pos_tol, ori_tol = (0.2, 0.001, 0.5) if args is None else _arrival_tolerances(args)
    if (joint_error > joint_tol or position_error > pos_tol
            or orientation_error_deg(actual_tcp, tcp) > ori_tol):
        raise RuntimeError("实际关节／TCP已离开预检路点，禁止继续或自动返程")
    return joint_error, position_error


def _execute_cross_steps(arm, nodes, offset, result, phase, *, source, fk, args, solver=None):
    """Send each stage endpoint once; check progress along the joint segment."""
    from tools.real_nero_point_test import assert_workspace
    joint_tol, pos_tol, ori_tol = _arrival_tolerances(args)
    timeout = float(getattr(args, "move_wait_timeout", 60.0))
    if not np.isfinite(timeout) or timeout <= 0:
        raise ValueError("阶段运动超时必须有限且大于0")
    for stage_index, (q_target, expected_tcp) in enumerate(nodes, 1):
        sample = dict(phase=phase, stage_index=stage_index,
                      planned_joints_rad=q_target.tolist(), planned_tcp_xyz_m=expected_tcp[:3, 3].tolist())
        try:
            _assert_motion_mode(arm)
            _assert_at_node(arm, source[0], source[1], offset, args=args)
            before = _finite_joints(arm)
            _, before_flange, _ = fresh_pose(arm)
            if solver is not None:
                _preflight_joint_segment(before, q_target, before_flange @ offset, expected_tcp,
                                         offset, solver, fk, args, phase)
            delta = q_target-before
            norm_sq = float(delta @ delta)
            result["motion_sent"] = True
            arm.move_j(q_target)
            deadline = time.monotonic()+timeout
            stable, previous_stamp, previous_joint_stamp = 0, None, None
            previous_q = None
            settled_samples = []
            while time.monotonic() < deadline:
                _assert_motion_mode(arm)
                actual_pose, actual_flange, stamp = fresh_pose(arm)
                measured = actual_flange @ offset
                actual_q, joint_stamp = _joint_sample(arm)
                if stamp == previous_stamp or joint_stamp == previous_joint_stamp:
                    time.sleep(0.01)
                    continue
                previous_stamp, previous_joint_stamp = stamp, joint_stamp
                progress = float(np.clip((actual_q-before) @ delta/norm_sq, 0, 1)) if norm_sq > 1e-15 else 1.0
                projected_q = before+progress*delta
                projected_tcp = fk(projected_q) @ offset
                if not np.isfinite(projected_tcp).all():
                    raise RuntimeError("运行中插值FK无效，停止后续运动")
                path_joint_error = float(np.rad2deg(np.max(np.abs(actual_q-projected_q))))
                path_tcp_error = float(np.linalg.norm(measured[:3, 3]-projected_tcp[:3, 3]))
                joint_error = float(np.rad2deg(np.max(np.abs(actual_q-q_target))))
                position_error = float(np.linalg.norm(measured[:3, 3]-expected_tcp[:3, 3]))
                orientation_error = orientation_error_deg(measured, expected_tcp)
                sample.update(
                    feedback_timestamp=stamp, joint_feedback_timestamp=joint_stamp,
                    actual_joints_rad=actual_q.tolist(), actual_tcp_xyz_m=measured[:3, 3].tolist(),
                    actual_flange_pose_m_rad=actual_pose.tolist(), joint_error_deg=joint_error,
                    position_error_mm=position_error*1000, orientation_error_deg=orientation_error,
                    joint_progress=progress, path_joint_error_deg=path_joint_error,
                    path_tcp_error_mm=path_tcp_error*1000)
                result["last_feedback"] = sample.copy()
                assert_workspace(measured[:3, 3], args)
                if (path_joint_error > 3.0 or path_tcp_error > 0.005
                        or orientation_error_deg(measured, projected_tcp) > 3.0):
                    raise RuntimeError("实际位置／关节偏离已检查的关节插值路径；停止，不自动返程")
                stationary = previous_q is not None and np.rad2deg(np.max(np.abs(actual_q-previous_q))) <= 0.1
                previous_q = actual_q.copy()
                reached = joint_error <= joint_tol and position_error <= pos_tol and orientation_error <= ori_tol
                motion_reader = getattr(arm, "_controller_motion_active", None)
                reported_active = motion_reader() if callable(motion_reader) else None
                sample["controller_motion_active"] = reported_active
                if reported_active is True or not reached:
                    settled_samples.clear()
                    stationary = False
                elif reported_active is not False:
                    now = time.monotonic()
                    settled_samples.append((now, actual_q.copy(), measured[:3, 3].copy()))
                    while len(settled_samples) > 2 and settled_samples[1][0] <= now-0.25:
                        settled_samples.pop(0)
                    stationary = (now-settled_samples[0][0] >= 0.25
                                  and np.rad2deg(np.ptp([entry[1] for entry in settled_samples], axis=0).max()) <= 0.1
                                  and np.linalg.norm(np.ptp([entry[2] for entry in settled_samples], axis=0)) <= 0.0005)
                stable = stable+1 if reached and stationary else 0
                if stable >= 2:
                    result["samples"].append(sample.copy())
                    print(f"[cross][{phase}] 阶段到位：joint={joint_error:.3f}deg "
                          f"TCP={position_error*1000:.3f}mm ori={orientation_error:.3f}deg")
                    break
                time.sleep(0.01)
            else:
                raise RuntimeError(f"{phase}阶段{timeout:g}s未到位："
                                   f"joint={sample.get('joint_error_deg', float('nan')):.3f}deg，"
                                   f"TCP={sample.get('position_error_mm', float('nan')):.3f}mm；停止、不自动返程")
        except BaseException as exc:
            sample["error"] = f"{type(exc).__name__}: {exc}"
            reader = getattr(arm.arm, "get_arm_status", None)
            if callable(reader):
                try:
                    wrapped = reader()
                    message = getattr(wrapped, "msg", wrapped)
                    sample["controller_status"] = {key: str(getattr(message, key, None))
                                                   for key in ("ctrl_mode", "arm_status", "motion_status")}
                except Exception:
                    sample["controller_status"] = "unavailable"
            result["failure_sample"] = sample.copy()
            print(f"[cross][failure] {phase}：{sample['error']}")
            print("[cross][failure] 目标TCP_mm =", np.round(expected_tcp[:3, 3]*1000, 3).tolist(),
                  "实际TCP_mm =", None if "actual_tcp_xyz_m" not in sample
                  else np.round(np.asarray(sample["actual_tcp_xyz_m"])*1000, 3).tolist())
            raise
        source = q_target, expected_tcp


def execute_cross_roundtrip(arm, args, state):
    """m: observation -> hover -> visual point -> exact saved observation q."""
    if not args.execute or int(getattr(args, "nero_speed_percent", 1)) != 1:
        raise ValueError("叉叉往返要求显式--execute和速度1%")
    _assert_motion_mode(arm)
    if state.get("observation_joints_rad") is None:
        raise ValueError("本轮缺少拍照关节位置；先停稳后c重新采图")
    if state.get("cross_roundtrip") is not None:
        raise ValueError("本轮往返已执行或中止；重新测试请c采图，不重复旧运动")
    path = Path(state["out_dir"]) / "cross_roundtrip.json"
    if path.exists():
        raise FileExistsError("本轮往返记录已存在，请c重新采图")
    original_q = np.asarray(state["observation_joints_rad"], float).reshape(7)
    q_start = _finite_joints(arm)
    _, flange, _ = fresh_pose(arm)
    offset = pose_transform(args.tcp_offset)
    start = flange @ offset
    capture_flange = pose_transform(state["flange_pose"])
    if (np.rad2deg(np.max(np.abs(q_start-original_q))) > 0.2
            or np.linalg.norm(flange[:3, 3]-capture_flange[:3, 3]) > 0.001
            or orientation_error_deg(flange, capture_flange) > 0.3):
        raise ValueError("机械臂已离开拍照原位；请停在新的观察位后c重新采图，不补走未知路径")
    solver, fk = state.get("cross_kinematics") or _jog_kinematics()
    state["cross_kinematics"] = solver, fk
    plan = plan_cross_roundtrip(start, state["P_base"], original_q, offset, solver, fk, args)
    print("[cross] 已求解悬停／目标两次IK，并检查关节插值与返程；目标朝向保持原值，不开合夹爪。")
    print("[cross] 视觉点 XYZ_mm =", np.round(np.asarray(state["P_base"])*1000, 2).tolist())
    print(f"[cross] 去程 {plan['outbound_length_m']*1000:.1f}mm，速度1%；无全局避障、无可靠轻触检测。")
    print("[cross] 每阶段只发一次move_j；预检采样不发送、不逐毫米停稳。无额外力矩门限。")
    if input("确认全臂、工具和往返路径无遮挡，输入 MOVE 到叉叉上方；Enter取消：").strip() != "MOVE":
        return False
    result = dict(created_at=datetime.now().astimezone().isoformat(), status="started",
                  motion_sent=False, automatic_contact_detection=False,
                  control_mode="move_j", software_torque_guard=False,
                  execution_strategy="endpoint_ik_joint_interpolation", preflight=plan["checks"],
                  original_joints_rad=original_q.tolist(), original_tcp_pose_matrix=start.tolist(),
                  P_base_visual_m=np.asarray(state["P_base"]).tolist(), samples=[])
    state["cross_roundtrip"] = result
    try:
        _assert_motion_mode(arm)
        origin = original_q, start
        _assert_at_node(arm, *origin, offset)
        _execute_cross_steps(arm, plan["to_hover"], offset, result, "to_hover", source=origin,
                             fk=fk, args=args, solver=solver)
        hover_hold_q = _finite_joints(arm)
        _, hover_hold_flange, _ = fresh_pose(arm)
        hover_hold = hover_hold_q, hover_hold_flange @ offset
        print("[cross] 已到叉叉上方。下面按视觉预测高度接近；看到接触或异常立即Ctrl+C，不继续下压。")
        approach = input("输入 APPROACH 接近预测点并自动原路返回；Enter放弃接近、原路返回：").strip() == "APPROACH"
        _assert_motion_mode(arm)
        hover_node = plan["to_hover"][-1] if plan["to_hover"] else origin
        _assert_at_node(arm, *hover_hold, offset)
        _assert_at_node(arm, *hover_node, offset, args=args)
        return_source = hover_node
        if approach:
            _execute_cross_steps(arm, plan["approach"], offset, result, "approach",
                                 source=hover_node, fk=fk, args=args, solver=solver)
            actual_pose, actual_flange, stamp = fresh_pose(arm)
            actual_tcp = actual_flange @ offset
            state["cross_last_execution"] = dict(
                created_at=datetime.now().astimezone().isoformat(), purpose="visual_point",
                requested_tcp_xyz_m=np.asarray(state["P_base"]).tolist(), actual_tcp_xyz_m=actual_tcp[:3, 3].tolist(),
                actual_flange_pose_m_rad=actual_pose.tolist(), feedback_timestamp=stamp,
                tracking_error_mm=float(np.linalg.norm(actual_tcp[:3, 3]-state["P_base"])*1000),
                automatic_contact_detection=False)
            result["point_execution"] = state["cross_last_execution"]
            print("[cross] 实际到位 TCP XYZ_mm =", np.round(actual_tcp[:3, 3]*1000, 2).tolist())
            return_source = plan["approach"][-1]
        _execute_cross_steps(arm, plan["return"] if approach else plan["return_from_hover"],
                             offset, result, "return", source=return_source, fk=fk, args=args, solver=solver)
        joint_error, position_error = _assert_at_node(arm, *origin, offset, args=args)
        result["return_joint_error_deg"] = joint_error
        result["return_tcp_error_mm"] = position_error*1000
        result["status"] = "completed" if approach else "cancelled_at_hover_returned"
        print("[cross] 已返回拍照原位。现在切示教，轻触同一叉叉，停稳后r、TOUCH记录；无需保持拍照朝向。")
        return approach
    except BaseException as exc:
        result["status"] = "failed"
        result["error"] = f"{type(exc).__name__}: {exc}"
        if result["motion_sent"]:
            arm.stop_for_program_exit("cross-point round trip interrupted/failed; no automatic return")
        raise
    finally:
        result["finished_at"] = datetime.now().astimezone().isoformat()
        write_new(path, result)


def supervised_jog(arm, args, state, *, down=False):
    if not args.execute:
        raise ValueError("点动需要显式 --execute")
    if state.get("target_tcp_pose") is None:
        raise ValueError("先用 c 点击叉叉")
    if getattr(arm, "_safety_stop_latched", False):
        raise RuntimeError("停机已锁存，不再发送运动")
    status_reader = getattr(arm.arm, "get_arm_status", None)
    if callable(status_reader):
        wrapped = status_reader()
        status = getattr(wrapped, "msg", wrapped)
        if "TEACH" in str(getattr(status, "ctrl_mode", "")).upper():
            raise ValueError("当前仍在示教模式；先用你现有的操作方式结束示教，再m/d，不自动切模式")
    pose, flange, _ = fresh_pose(arm)
    offset = pose_transform(args.tcp_offset)
    start = flange @ offset
    captured = pose_transform(state["target_tcp_pose"])
    if orientation_error_deg(start, captured) > 3.0:
        raise ValueError("恢复到采集时的工具姿态后再点动，不自动翻转工具")
    target = start.copy()
    if down:
        if not state.get("cross_hover_reached"):
            raise ValueError("先用 m 完成悬停，再逐次 d 下降")
        if np.linalg.norm(start[:2, 3]-np.asarray(state["P_base"])[:2]) > 0.005:
            raise ValueError("横向未对准预测点5 mm以内，禁止下降")
        floor = float(state["P_base"][2])
        target[2, 3] = max(floor, start[2, 3]-args.cross_step_mm/1000)
        if target[2, 3] >= start[2, 3]-1e-6:
            print("已到视觉预测高度，不继续向下顶。若未接触，用示教对准，再 r 比较。")
            return False
    else:
        target[:3, 3] = captured[:3, 3]
        if target[2, 3] < start[2, 3]-0.002:
            raise ValueError("m仅用于水平／向上对准悬停点；若当前高于悬停点，先示教到附近，下降只能逐次d")
    from tools.real_nero_point_test import assert_workspace
    assert_workspace(target[:3, 3], args)
    solver, fk = state.get("cross_kinematics") or _jog_kinematics()
    state["cross_kinematics"] = (solver, fk)
    q_start = np.asarray(arm.get_joint_angles(), float).reshape(7)
    plan = _bounded_jog_plan(start, target, q_start, offset, solver, fk)
    print("目标TCP XYZ_mm =", np.round(target[:3, 3]*1000, 2).tolist())
    print("局部监督点动：速度1%，单次关节步长≤2deg，无全局避障、无可靠轻触检测。")
    if input("确认局部路径和全臂周围无障碍；一旦接触就不再d。输入 MOVE 执行，Enter取消：").strip() != "MOVE":
        return False
    moved = False
    try:
        # Recheck after the operator confirmation; the robot may have been taught meanwhile.
        actual = np.asarray(arm.get_joint_angles(), float).reshape(7)
        if np.rad2deg(np.max(np.abs(actual-q_start))) > 0.2:
            raise RuntimeError("确认期间机械臂已移动；本次不发送目标，请重新点动")
        for q_target, expected_tcp in plan:
            arm.move_j(q_target)
            moved = True
            deadline = time.monotonic()+5.0
            stable, previous_stamp = 0, None
            while time.monotonic() < deadline:
                _, measured_flange, stamp = fresh_pose(arm)
                measured = measured_flange @ offset
                q = np.asarray(arm.get_joint_angles(), float).reshape(7)
                if stamp == previous_stamp:
                    time.sleep(0.01)
                    continue
                previous_stamp = stamp
                if (np.rad2deg(np.max(np.abs(q-q_target))) > 3.0
                        or orientation_error_deg(measured, expected_tcp) > 3.0
                        or np.linalg.norm(measured[:3, 3]-expected_tcp[:3, 3]) > 0.005):
                    raise RuntimeError("实际位置／姿态偏离毫米级点动范围")
                arrived = (np.max(np.abs(q-q_target)) <= np.deg2rad(0.1)
                           and np.linalg.norm(measured[:3, 3]-expected_tcp[:3, 3]) <= 0.0005)
                stable = stable+1 if arrived else 0
                if stable >= 2:
                    break
                time.sleep(0.01)
            else:
                raise RuntimeError("局部点动未到位，不继续下一个目标")
        if not down:
            state["cross_hover_reached"] = True
        actual_pose, actual_flange, stamp = fresh_pose(arm)
        actual_tcp = actual_flange @ offset
        state["cross_last_execution"] = dict(
            created_at=datetime.now().astimezone().isoformat(),
            purpose="approach_step" if down else "hover",
            requested_tcp_xyz_m=target[:3, 3].tolist(), actual_tcp_xyz_m=actual_tcp[:3, 3].tolist(),
            actual_flange_pose_m_rad=actual_pose.tolist(), feedback_timestamp=stamp,
            tracking_error_mm=float(np.linalg.norm(actual_tcp[:3, 3]-target[:3, 3])*1000),
            automatic_contact_detection=False,
        )
        print("实际执行 TCP XYZ_mm =", np.round(actual_tcp[:3, 3]*1000, 2).tolist())
        return True
    except BaseException:
        if moved:
            arm.stop_for_program_exit("cross-point jog interrupted/failed")
        raise
