"""抓取 → 伺服 控制权交接层。

时序（铁律：任何时刻只有一个控制源在发 CAN 运动命令）：
  1. move_j 轨迹普通终止（terminate_motion，非急停），轮询确认关节不再运动
  2. 新建 MIT 后端：跳过 reset，第一帧就用验证重力，关节位置保持到静止
+  3. 夹爪保持当前宽度（旧 driver 仍连通）
+  4. 确认 7 关节反馈新鲜后，才断开抓取阶段的 SDK driver
"""
from __future__ import annotations

import logging
import time

import numpy as np

from models.nero_model import NeroPinocchioModel
from control_backends.sdk_force_ctrl_backend import SdkForceCtrlBackend
from control_backends.force_ctrl.impedance_profile import (
    VALIDATED_GRAVITY_FACTORS,
    VALIDATED_TOTAL_TORQUE_LIMIT,
)

logger = logging.getLogger(__name__)

GRIPPER_MAX_WIDTH = 0.085   # 与抓取阶段一致的硬件钳位
GRIPPER_MAX_FORCE = 0.50


# def _ramp_gravity_factors(backend, target_factors, duration_s=1.0, steps=20):
#     """Avoid the first MIT frame applying 1.25x gravity and jerking the TCP up."""
#     start = np.ones(7, dtype=float)
#     target = np.asarray(target_factors, dtype=float).ravel()
#     steps = max(int(steps), 1)
#     dt = max(float(duration_s), 0.0) / steps
#     logger.info("[handoff] ramping gravity factors from 1.0 to validated profile")
#     for i in range(1, steps + 1):
#         alpha = i / steps
#         backend.set_gravity_factors(start + alpha * (target - start))
#         if dt > 0.0:
#             time.sleep(dt)
def _wait_position_hold(backend, q_hold, confirm_samples=3, settle_delta_deg=0.5, timeout_s=1.5):
    """MIT 关节保持稳定后再把控制权交给笛卡尔阻抗。"""
    q_hold = np.asarray(q_hold, dtype=float).reshape(7)
    confirmed = 0
    deadline = time.time() + timeout_s
    last_delta = float("nan")
    while time.time() < deadline:
        q = backend.get_joint_angles()
        if q is None or len(q) != 7 or not np.all(np.isfinite(q)):
            time.sleep(0.02)
            continue
        last_delta = float(np.max(np.abs(np.rad2deg(np.asarray(q, dtype=float).reshape(7) - q_hold))))
        if last_delta <= settle_delta_deg:
            confirmed += 1
            if confirmed >= confirm_samples:
                logger.info(
                    "[handoff] MIT position hold settled delta=%.3fdeg", last_delta
                )
                return True
        else:
            confirmed = 0
        time.sleep(0.05)
    logger.warning(
        "[handoff] MIT position hold did not settle in %.1fs last_delta=%.3fdeg",
        timeout_s, last_delta,
    )
    return False


def _hold_gripper(gripper, width, force):
    if gripper is None:
        raise RuntimeError("no gripper handle while grasp driver is still connected")
    return gripper.move_gripper_m(
        value=min(float(width), GRIPPER_MAX_WIDTH),
        force=min(float(force), GRIPPER_MAX_FORCE),
    )

def stop_grasp_control(arm, confirm_samples=2, settle_delta_deg=0.5, timeout_s=3.0):
    """停止 move_j 轨迹（普通终止）并轮询确认关节不再运动。"""
    fallback = np.asarray(arm.get_joint_angles(), dtype=float).reshape(7)
    arm.terminate_motion(fallback_joints=fallback)
    # 阻止稍后的 __exit__ 再次触发程序退出急停（工程内 session_control 同款用法）

    confirmed, last, last_stamp = 0, None, None
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        feedback = arm.arm.get_joint_angles()
        stamp = float(getattr(feedback, 'timestamp', 0.0))
        if not np.isfinite(stamp) or not 0 <= time.time() - stamp <= .5:
            return False
        if stamp == last_stamp:
            time.sleep(.02)
            continue
        last_stamp = stamp
        q = np.asarray(feedback.msg, dtype=float).reshape(7)
        if not np.isfinite(q).all():
            return False
        if last is not None:
            delta = float(np.max(np.abs(np.rad2deg(q - last))))
            if delta <= settle_delta_deg:
                confirmed += 1
                if confirmed >= confirm_samples:
                    arm._suppress_program_exit_estop = True
                    arm._controlled_stop_request_confirmed = True
                    logger.info("[handoff] move_j trajectory confirmed stopped")
                    return True
            else:
                confirmed = 0
        last = q
        time.sleep(0.15)
    logger.warning("[handoff] trajectory stop NOT confirmed in %.1fs", timeout_s)
    return False


class BackendGripper:
    """MIT 后端上的 AGX 夹爪。抓取阶段已经 init 过，不能对第二套 robot 再 init。"""

    def __init__(self, robot, gripper=None):
        self._robot = robot
        if gripper is not None:
            self._gripper = gripper
            return
        try:
            self._gripper = robot.init_effector(robot.OPTIONS.EFFECTOR.AGX_GRIPPER)
        except Exception as exc:
            if "already initialized" not in str(exc).lower():
                raise
            for name in ("gripper", "end_effector", "effector"):
                got = getattr(robot, name, None)
                if got is not None:
                    self._gripper = got
                    logger.info("[handoff] reuse already-initialized AGX gripper")
                    return
            raise RuntimeError(
                "AGX gripper already initialized and could not be reused"
            ) from exc

    def open(self, width=GRIPPER_MAX_WIDTH, force=GRIPPER_MAX_FORCE):
        if not np.isfinite(width) or width < 0. or not np.isfinite(force) or force <= 0.:
            raise ValueError('invalid gripper width/force command')
        width = min(float(width), GRIPPER_MAX_WIDTH)
        force = min(float(force), GRIPPER_MAX_FORCE)
        result = self._gripper.move_gripper_m(value=width, force=force)
        self.last_command = dict(width=width, force=force)
        return result

    def status(self):
        return self._gripper.get_gripper_status()


def build_servo_backend(channel="can0", firmware="v111", dry_run=False):
    """创建并连接 MIT 后端。zero_gravity=False：接管瞬间是位置保持，物体不掉。"""
    model = NeroPinocchioModel()  # models/nero_with_gripper.urdf（相对工程根）
    backend = SdkForceCtrlBackend(
        dry_run=dry_run,
        channel=channel,
        control_hz=200.0,
        firmware=firmware,
        interface="socketcan",
        torque_limit=VALIDATED_TOTAL_TORQUE_LIMIT,  # 14 N·m
        impedance_torque_limit=4.0,
        # gravity_factors=np.ones(7, dtype=float),
        gravity_factors=VALIDATED_GRAVITY_FACTORS.copy(),
        zero_gravity=False,
        gravity_compensation=True,
        pin_model=model.clone(),  # 200 Hz 内环线程用独立 clone（Pinocchio Data 非线程安全）
        startup_joint_margin=0.08,
        skip_reset=True,
        keep_enabled_on_disconnect=True,
    )
    try:
        backend.connect()
        q = backend.get_joint_angles()
        feedback = backend.get_robot().get_joint_angles()
        stamp = float(getattr(feedback, 'timestamp', 0.0))
        if q is None or len(q) != 7 or not np.all(np.isfinite(q)) or not 0 <= time.time() - stamp <= .5:
            raise RuntimeError("[handoff] MIT joint feedback unavailable or stale")
        return backend
    except BaseException:
        backend.disconnect()
        raise


def handoff(arm, args=None):
    """在抓取 arm 对象仍连接时完成控制权交接，返回可用的伺服后端。"""
    logger.info("[handoff] 1/3 stopping move_j control ...")
    if not stop_grasp_control(arm):
        raise RuntimeError("[handoff] aborted: move_j trajectory did not stop")
    logger.info("[handoff] 2/3 taking over with MIT backend (position hold) ...")
    backend = build_servo_backend(channel=getattr(args, "nero_channel", "can0"),
                                  firmware=getattr(args, "nero_firmware", "V111").lower())
    # _ramp_gravity_factors(backend, VALIDATED_GRAVITY_FACTORS, duration_s=1.0)
    q_hold = backend.get_joint_angles()
    if q_hold is None or len(q_hold) != 7 or not np.all(np.isfinite(q_hold)):
        raise RuntimeError("[handoff] MIT hold pose unavailable after connect")
    _wait_position_hold(backend, q_hold)
    logger.info("[handoff] 3/3 releasing grasp-arm SDK driver ...")
    try:
        arm.arm.disconnect()  # 抓取驱动下线；MIT 已接管
    except Exception as exc:  # 非致命：驱动已无轨迹
        # backend.emergency_stop()
        # backend.disconnect()
        raise RuntimeError("Grasp driver disconnect failed during handoff") from exc
    return backend

def handoff_after_lift(arm, args, close_status=None):
    """After the lift, hand the arm to the servo delivery backend.

    Unified MIT mode (``trajectory_executor == "mit"``) reuses the already
    running MIT session — the grasp already ran on MIT, so there is no
    move_j->MIT switch and no second CAN driver.  The legacy ``move_j`` path
    keeps the original stop + fresh-MIT-backend handoff.
    """
    if close_status is None:
        close_status = getattr(args, "_closed_gripper_status", None)
    width = float(getattr(close_status, "contact_width", None) or 0.0)
    if not np.isfinite(width) or width <= 0.0:
        width = 0.069
    force = min(float(getattr(args, "gripper_force", GRIPPER_MAX_FORCE)), GRIPPER_MAX_FORCE)
    strict_loaded = bool(getattr(args, 'gesture_handover', False) or getattr(args, 'fusion_handover', None))
    if strict_loaded:
        # The original commanded target maintains contact force. The measured
        # opening is captured separately as the gesture 1mm closure baseline.
        command = getattr(arm, '_last_gripper_command', None)
        if not isinstance(command, dict):
            raise RuntimeError('loaded gripper command unavailable; refusing gesture handoff')
        width, force = float(command['width']), float(command['force'])
        if not np.isfinite(width) or not 0. <= width <= GRIPPER_MAX_WIDTH or not 0. < force <= GRIPPER_MAX_FORCE:
            raise RuntimeError('invalid loaded gripper command; refusing gesture handoff')
    existing_gripper = getattr(arm, "gripper", None)
    try:
        _hold_gripper(existing_gripper, width, force)
        logger.info("[handoff] gripper hold sent on grasp driver width=%.4fm", width)
    except Exception as exc:
        if strict_loaded:
            raise
        logger.warning("[handoff] gripper hold before handoff failed: %s", exc)

    if str(getattr(args, "trajectory_executor", "move_j")).lower() == "mit":
        from real_nero_grasp.mit_trajectory_executor import get_mit_session

        session = get_mit_session(arm, args)
        session.ensure_started()
        try:
            held = BackendGripper(session.get_robot(), gripper=existing_gripper)
            held.open(width=width, force=force)
            args._servo_gripper = held
        except Exception as exc:
            if strict_loaded:
                raise
            logger.warning("[handoff] gripper hold command failed: %s", exc)
        args._servo_backend = session
        logger.info("[handoff] after lift: reusing running MIT session (unified, no second CAN)")
        return session

    logger.info("[handoff] after lift: stop move_j, MIT hold, keep gripper width=%.4fm", width)
    backend = handoff(arm, args)
    try:
        held = BackendGripper(backend.get_robot())
        held.open(width=width, force=force)
        args._servo_gripper = held
    except Exception as exc:
        if strict_loaded:
            raise
        logger.warning("[handoff] gripper hold command failed: %s", exc)
    args._servo_backend = backend
    return backend


handoff_after_close = handoff_after_lift


def delivery_after_grasp(arm, args):
    """runtime hook 入口：抬升 SUCCESS 后交接 → 伺服递送。"""
    if not getattr(args, 'execute', False):
        raise RuntimeError('Hardware handoff requires --execute')
    from servo_delivery import validate_servo_args
    validate_servo_args(args)
    backend = getattr(args, "_servo_backend", None)
    try:
        if backend is None:
            backend = handoff_after_lift(
                arm, args, getattr(args, "_closed_gripper_status", None)
            )
        from servo_delivery import run_servo_delivery
        code = run_servo_delivery(args, backend=backend)
        if code == 130:
            raise KeyboardInterrupt()  # Propagate interruption to the whole workflow.
        return code
    except Exception as exc:
        logger.warning("[handoff] servo after lift failed; checking controlled cleanup: %s", exc)
        backend = getattr(args, "_servo_backend", None)
        if backend is not None:
            from control_backends.force_ctrl.normal_shutdown import close_backend_normally
            close_backend_normally(backend)
            args._servo_backend = None
        raise
