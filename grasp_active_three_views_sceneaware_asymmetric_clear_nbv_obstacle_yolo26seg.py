import os
import sys
import gc
import time
import itertools
import atexit
from contextlib import contextmanager
from collections import deque


os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("YOLO_CONFIG_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), ".ultralytics"))
os.environ.setdefault("MUJOCO_GL", "glfw")
os.environ.setdefault("MUJOCO_VIEWER", "1")

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
GRASPNET_BASELINE_DIR = os.path.join(ROOT_DIR, 'graspnet-baseline')
for _path in (
    os.path.join(GRASPNET_BASELINE_DIR, 'pointnet2'),
    os.path.join(GRASPNET_BASELINE_DIR, 'graspnetAPI'),
    os.path.join(GRASPNET_BASELINE_DIR, 'models'),
    os.path.join(GRASPNET_BASELINE_DIR, 'dataset'),
    os.path.join(GRASPNET_BASELINE_DIR, 'utils'),
    os.path.join(ROOT_DIR, 'manipulator_grasp'),
):
    if _path not in sys.path:
        sys.path.insert(0, _path)

# The baseline root contains a source-only `pointnet2` package. Keep that
# parent directory behind site-packages so `pointnet2._ext` resolves to the
# installed CUDA extension; sibling helper modules above still use local code.
if GRASPNET_BASELINE_DIR not in sys.path:
    sys.path.append(GRASPNET_BASELINE_DIR)


def _preload_conda_libstdcpp():
    conda_prefix = os.environ.get("CONDA_PREFIX", "").strip()
    candidates = []
    if conda_prefix:
        candidates.append(os.path.join(conda_prefix, "lib", "libstdc++.so.6"))
    candidates.append("/home/zgj/anaconda3/envs/grasp_yolo26/lib/libstdc++.so.6")
    for candidate in candidates:
        if not os.path.exists(candidate):
            continue
        try:
            import ctypes
            ctypes.CDLL(candidate, mode=ctypes.RTLD_GLOBAL)
            return candidate
        except Exception:
            continue
    return None


_preload_conda_libstdcpp()

import numpy as np

# Compatibility for older dependencies such as transforms3d/graspnetAPI on
# NumPy >= 1.24, where aliases like np.float were removed.
if not hasattr(np, "float"):
    np.float = float
if not hasattr(np, "int"):
    np.int = int
if "bool" not in np.__dict__:
    np.bool = bool

import open3d as o3d
import scipy.io as scio
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as SciRot
import torch
from PIL import Image
import spatialmath as sm


import inspect
import cv2
import mujoco

from graspnetAPI import GraspGroup

VERBOSE_LOG = os.environ.get("MG_VERBOSE_LOG", "").strip().lower() in {"1", "true", "yes", "on", "debug", "verbose"}

from graspnet import GraspNet, pred_decode
from graspnet_dataset import GraspNetDataset
from collision_detector import ModelFreeCollisionDetector
from data_utils import CameraInfo, create_point_cloud_from_depth_image

from manipulator_grasp.arm.motion_planning import *
from manipulator_grasp.env.ur5_grasp_env import UR5GraspEnv

from cv_process_updated_yolo26seg import (
    segment_image,
    detect_context_instances_center_view,
    detect_objects,
    release_vision_model_caches,
    get_last_selected_instance,
    infer_instances_yolo26_amodal,
)

try:
    from kinematic_trac_ik import Kinematic as TracIKKinematic
except Exception as _trac_ik_import_error:
    TracIKKinematic = None
    _TRAC_IK_IMPORT_ERROR = _trac_ik_import_error
else:
    _TRAC_IK_IMPORT_ERROR = None

_TRAC_IK_SOLVER = None
_TRAC_IK_INIT_ERROR = None
_TRAC_IK_CONFIG_PRINTED = False


def _print_runtime_library_report():
    if os.environ.get("PRINT_RUNTIME_LIBS", "").strip().lower() not in {"1", "true", "yes", "on"}:
        return
    print("\n========== Runtime Library Report ==========")
    print(f"python executable: {sys.executable}")
    print(f"python version: {sys.version.replace(os.linesep, ' ')}")
    print(f"CONDA_PREFIX: {os.environ.get('CONDA_PREFIX', '')}")
    print(f"PYTHONPATH: {os.environ.get('PYTHONPATH', '')}")
    print(f"LD_LIBRARY_PATH: {os.environ.get('LD_LIBRARY_PATH', '')}")
    print(f"MUJOCO_GL: {os.environ.get('MUJOCO_GL', '')}")
    print(f"MUJOCO_VIEWER: {os.environ.get('MUJOCO_VIEWER', '')}")

    for name in ("numpy", "scipy", "torch", "mujoco", "cv2", "open3d", "ultralytics", "pointnet2"):
        try:
            module = __import__(name)
            print(f"{name}: version={getattr(module, '__version__', '?')} file={getattr(module, '__file__', '?')}")
        except Exception as exc:
            print(f"{name}: import_error={type(exc).__name__}: {exc}")

    patterns = (
        "numpy.libs",
        "scipy.libs",
        "libopenblas",
        "liblapack",
        "libblas",
        "libgfortran",
        "libquadmath",
        "libstdc++",
        "libgcc_s",
        "mujoco",
        "libGL",
        "libEGL",
        "libglfw",
        "libcuda",
        "libcudart",
        "libcublas",
        "libcudnn",
        "libtorch",
        "libc10",
        "pointnet2",
        "grasp_nms",
    )
    loaded = set()
    try:
        with open("/proc/self/maps", "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                path = line.rsplit(maxsplit=1)[-1].strip()
                if not path.startswith("/"):
                    continue
                if any(pat in path for pat in patterns):
                    loaded.add(path)
    except Exception as exc:
        print(f"/proc/self/maps read failed: {type(exc).__name__}: {exc}")
    print("\n-- loaded shared libraries from this process --")
    for path in sorted(loaded):
        print(path)
    print("======== End Runtime Library Report ========\n")


_print_runtime_library_report()


@contextmanager
def _temporary_env(overrides):
    old_values = {}
    for key, value in dict(overrides).items():
        old_values[key] = os.environ.get(key, None)
        os.environ[key] = str(value)
    try:
        yield
    finally:
        for key, old in old_values.items():
            if old is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old

def _brief_instance(inst):
    if inst is None:
        return None
    center = inst.get("bbox_center", inst.get("center", None))
    extent = inst.get("bbox_extent", None)
    return {
        "id": int(inst.get("id", -1)),
        "label": inst.get("label", "unknown"),
        "role": inst.get("role", "unknown"),
        "score": round(float(inst.get("score", 0.0)), 3),
        "front_point_count": int(inst.get("front_point_count", 0)),
        "nearest_dist": round(float(inst.get("nearest_dist", np.inf)), 4),
        "front_ratio": round(float(inst.get("front_ratio", 0.0)), 4),
        "projected_overlap": round(float(inst.get("projected_overlap", 0.0)), 4),
        "center": None if center is None else np.round(np.asarray(center, dtype=float).reshape(-1)[:3], 4).tolist(),
        "extent": None if extent is None else np.round(np.asarray(extent, dtype=float).reshape(-1)[:3], 4).tolist(),
    }

# 抓取末端测试
def mat_from_xmat(xmat_9):
    # mujoco: row-major 3x3 flattened
    return np.array(xmat_9, dtype=float).reshape(3, 3)

def rot_angle(Ra, Rb):
    # angle between rotations
    R = Ra.T @ Rb
    tr = np.clip((np.trace(R) - 1) / 2.0, -1.0, 1.0)
    return float(np.arccos(tr))

def pose_from_robot_SE3(T):
    # 兼容 spatialmath SE3：T.R / T.t
    R = np.array(T.R, dtype=float)
    t = np.array(T.t, dtype=float).reshape(3,)
    return R, t

def find_best_mujoco_frame_match(model, data, T_robot, topk=10):
    Rr, tr = pose_from_robot_SE3(T_robot)

    candidates = []
    # Sites
    for sid in range(model.nsite):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, sid)
        ts = np.array(data.site_xpos[sid], dtype=float)
        Rs = mat_from_xmat(data.site_xmat[sid])
        dp = np.linalg.norm(ts - tr)
        da = rot_angle(Rs, Rr)
        candidates.append(("site", name, dp, da))

    # Bodies
    for bid in range(model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid)
        tb = np.array(data.xpos[bid], dtype=float)
        Rb = mat_from_xmat(data.xmat[bid])
        dp = np.linalg.norm(tb - tr)
        da = rot_angle(Rb, Rr)
        candidates.append(("body", name, dp, da))

    # 排序：先位置误差，再角度误差
    candidates.sort(key=lambda x: (x[2], x[3]))
    return candidates[:topk]

def get_T_wc_from_camera(env, cam_name: str):
    """Compute T_wc (camera->world) in a *CV-style* camera frame.

    Notes:
      - MuJoCo camera frame: +X right, +Y up, camera looks along -Z.
      - CV/OpenCV frame (used by GraspNet + pointcloud projection): +X right, +Y down, +Z forward.
      - Conversion between CV->MuJoCo camera axes: R_mj_cv = diag(1, -1, -1)

    Therefore, for a vector expressed in CV camera coordinates v_cv:
        v_w = R_w_mj @ (R_mj_cv @ v_cv)
    so:
        R_w_cv = R_w_mj @ R_mj_cv
    """
    model, data = _get_mj_model_data(env)
    cid = _cam_id(model, cam_name)

    t_wc = np.array(data.cam_xpos[cid], dtype=float).reshape(3,)
    R_w_mj = np.array(data.cam_xmat[cid], dtype=float).reshape(3, 3)

    # CV -> MuJoCo
    R_mj_cv = np.diag([1.0, -1.0, -1.0])

    # CV -> World
    R_w_cv = R_w_mj @ R_mj_cv

    # Orthonormalize (numerical safety)
    U, _, Vt = np.linalg.svd(R_w_cv)
    R_w_cv = U @ Vt
    if np.linalg.det(R_w_cv) < 0:
        U[:, -1] *= -1
        R_w_cv = U @ Vt

    return sm.SE3.Rt(R_w_cv, t_wc)


def _orthonormalize_rotation_matrix(R, fallback=None):
    """Project a near-rotation matrix onto SO(3); return fallback if unusable."""
    try:
        R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    except Exception:
        return fallback
    if not np.isfinite(R).all():
        return fallback
    try:
        U, _, Vt = np.linalg.svd(R)
        R_fix = U @ Vt
        if np.linalg.det(R_fix) < 0:
            U[:, -1] *= -1
            R_fix = U @ Vt
        if not np.isfinite(R_fix).all() or np.linalg.det(R_fix) <= 0:
            return fallback
        return R_fix.astype(np.float64)
    except Exception:
        return fallback

def goto_observation_pose(env,
                          pos_w=(1.35, 0.6, 1.38),
                          rpy_w=(np.pi, 0.0, 0.0),
                          duration: float = 1.5,
                          fallback_joint_candidates=None):
    """Move EE to a high observation pose (top-down-ish) for YOLO26-seg.

    - First tries Cartesian motion (may fail due to IK).
    - If Cartesian IK fails, fall back to several joint-space target poses and
      execute a smooth joint trajectory (so you can *see* the arm moving).
    """
    if fallback_joint_candidates is None:
        fallback_joint_candidates = [
            np.array([0.000000, 0.150000, 0.0, 0.975000, -1.600000, -0.350000, -0.120000]),
            np.array([0.000000, 0.150000, 0.0, 0.975000, -1.600000, -0.350000, 0.000000]),
            np.array([0.000000, 0.349066, 0.0, 1.151917, -1.493363, 0.139626, -0.120000]),
            np.array([0.30, -1.30, 1.70, -1.90, -1.57, 0.0]),
            np.array([-0.30, -1.30, 1.70, -1.90, -1.57, 0.0]),
            np.array([0.0, -1.57, 1.57, -1.57, -1.57, 0.0]),
        ]
        obs_q_env = os.environ.get("NERO_OBS_Q", "").strip()
        if obs_q_env:
            try:
                q_env = np.fromstring(obs_q_env.replace(",", " "), sep=" ", dtype=float)
                if q_env.size >= 7 and np.isfinite(q_env[:7]).all():
                    fallback_joint_candidates.insert(0, q_env[:7])
            except Exception:
                print(f"[ObsPose] ignore invalid NERO_OBS_Q='{obs_q_env}'")

    robot = env.robot
    action = np.zeros(env.mj_model.nu)
    if env.mj_model.nu > robot.dof:
        action[robot.dof] = float(getattr(env, "_last_gripper_cmd", 0.0))

    def _apply_observation_z_offset():
        z_offset = float(os.environ.get("OBS_POSE_Z_OFFSET", "0.0"))
        if abs(z_offset) < 1e-6 or "_move_camera_pose_closed_loop" not in globals():
            return
        try:
            T_wc_now = get_T_wc_from_camera(env, EE_CAM_NAME)
            t_des = np.asarray(T_wc_now.t, dtype=float).reshape(3,).copy()
            t_des[2] += z_offset
            T_wc_des = sm.SE3.Rt(np.asarray(T_wc_now.R, dtype=float), t_des)
            print(
                f"[ObsPose] apply z offset {z_offset:+.3f}m: "
                f"camera_z {float(T_wc_now.t[2]):.4f}->{float(t_des[2]):.4f}"
            )
            _move_camera_pose_closed_loop(
                env,
                cam_name=EE_CAM_NAME,
                T_wc_des=T_wc_des,
                duration=float(os.environ.get("OBS_POSE_Z_OFFSET_DURATION", "0.45")),
                settle_steps=int(os.environ.get("OBS_POSE_Z_OFFSET_SETTLE_STEPS", "20")),
                pos_tol=float(os.environ.get("OBS_POSE_Z_OFFSET_POS_TOL", "0.008")),
                ang_tol_deg=float(os.environ.get("OBS_POSE_Z_OFFSET_ANG_TOL_DEG", "2.5")),
                max_iters=int(os.environ.get("OBS_POSE_Z_OFFSET_MAX_ITERS", "2")),
                third_iter_pos_trigger=0.010,
                third_iter_ang_trigger=2.0,
                debug=False,
            )
            T_wc_after = get_T_wc_from_camera(env, EE_CAM_NAME)
            print(f"[ObsPose] camera pos after z offset={np.round(np.asarray(T_wc_after.t, dtype=float), 4)}")
        except Exception as exc:
            print(f"[ObsPose] z offset correction skipped: {type(exc).__name__}: {exc}")

    q_cur = np.array(robot.get_joint(), dtype=float)
    print(f"[ObsPose] current q = {np.round(q_cur, 3)}")

    # --- Try Cartesian trajectory (keeps your intended world pose) ---
    try:
        if robot.dof != 6:
            print("[ObsPose] Nero7: skip slow Cartesian IK for observation pose; using joint-space fallback.")
            raise AssertionError

        T_start = robot.get_cartesian()
        T_target = sm.SE3.Trans(*pos_w) * sm.SE3(sm.SO3.RPY(list(rpy_w), order="xyz"))

        position_parameter = LinePositionParameter(T_start.t, T_target.t)
        attitude_parameter = OneAttitudeParameter(sm.SO3(T_start.R), sm.SO3(T_target.R))
        cartesian_parameter = CartesianParameter(position_parameter, attitude_parameter)
        velocity_parameter = QuinticVelocityParameter(duration)
        trajectory_parameter = TrajectoryParameter(cartesian_parameter, velocity_parameter)
        planner = TrajectoryPlanner(trajectory_parameter)

        total_time = duration
        time_step_num = round(total_time / 0.002) + 1
        times = np.linspace(0.0, total_time, time_step_num)

        for timei in times:
            if timei == 0.0:
                continue
            cart = planner.interpolate(timei)
            robot.move_cartesian(cart)  # may assert on IK failure
            joint = robot.get_joint()
            action[:len(joint)] = joint
            env.step(action)

        q_after = np.array(robot.get_joint(), dtype=float)
        print(f"[ObsPose] reached cart pose. q = {np.round(q_after, 3)}")
        _apply_observation_z_offset()
        return True

    except AssertionError:
        print("[ObsPose] Cartesian IK failed. Trying joint-space fallbacks...")

    # --- Joint-space fallbacks (guaranteed smooth & visible motion) ---
    q_start = np.array(robot.get_joint(), dtype=float)

    for q_tgt in fallback_joint_candidates:
        q_tgt = np.asarray(q_tgt, dtype=float).reshape(-1)
        q_tgt = np.pad(q_tgt, (0, max(0, robot.dof - q_tgt.size)))[:robot.dof]
        if np.linalg.norm(q_tgt - q_start) < 0.05:
            print(f"[ObsPose] already at fallback observation pose: {np.round(q_tgt, 3)}")
            try:
                T_wc_obs = get_T_wc_from_camera(env, EE_CAM_NAME)
                print(f"[ObsPose] camera pos={np.round(np.asarray(T_wc_obs.t, dtype=float), 4)}")
            except Exception:
                pass
            _apply_observation_z_offset()
            return True
        try:
            # Smooth joint trajectory to q_tgt
            parameter = JointParameter(q_start, q_tgt)
            velocity_parameter = QuinticVelocityParameter(duration)
            traj_param = TrajectoryParameter(parameter, velocity_parameter)
            planner = TrajectoryPlanner(traj_param)

            total_time = duration
            time_step_num = round(total_time / 0.002) + 1
            times = np.linspace(0.0, total_time, time_step_num)

            for timei in times:
                if timei == 0.0:
                    continue
                q = planner.interpolate(timei)
                robot.move_joint(q)
                action[:len(q)] = q
                env.step(action)

            # settle a bit
            action[:len(q)] = q_tgt
            for _ in range(150):
                env.step(action)

            q_after = np.array(robot.get_joint(), dtype=float)
            print(f"[ObsPose] fallback joint pose succeeded: {np.round(q_tgt, 3)} ; now q={np.round(q_after,3)}")
            try:
                T_wc_obs = get_T_wc_from_camera(env, EE_CAM_NAME)
                print(f"[ObsPose] camera pos={np.round(np.asarray(T_wc_obs.t, dtype=float), 4)}")
            except Exception:
                pass
            _apply_observation_z_offset()
            return True

        except Exception as e:
            print(f"[ObsPose] fallback failed: {np.round(q_tgt, 3)} ; err={type(e).__name__}")
            continue

    print("[ObsPose] All fallbacks failed. Using current pose for detection.")
    return False


def _prompt_target_class(prev_target_class=None):
    env_target_class = os.environ.get("YOLO26_TARGET_CLASS", "").strip()
    if env_target_class:
        return env_target_class
    if prev_target_class and _env_flag("AUTO_REUSE_TARGET_CLASS_ON_RETRY", "1"):
        print(f"[TargetSelect] reuse previous target class='{prev_target_class}' after failed attempt.")
        return prev_target_class

    prompt = "\n===============\nEnter class name (Enter/click=detect all YOLO26-seg instances, then click target)"
    if prev_target_class:
        prompt += f" [{prev_target_class}]"
    prompt += ": "
    value = input(prompt).strip()
    if value:
        if value.lower() in {"click", "manual", "select"}:
            return "__manual_click__"
        return value
    print("[TargetSelect] Empty class: detect all YOLO26-seg instances first, then click the target.")
    return "__manual_click__"


def _default_context_classes():
    return {
        "apple", "banana", "box", "bowl", "cup", "mug", "bottle",
        "table", "dining table", "orange", "t_block", "t-block", "tblock",
    }


EE_CAM_NAME = "ee_side_depth_cam"
def _cam_id(model, cam_name: str) -> int:
    cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
    if cid < 0:
        raise ValueError(f"[render] camera '{cam_name}' not found in model.")
    return cid


def _get_mj_model_data(env):
    """Best-effort fetch mujoco model/data from env."""
    if hasattr(env, "model") and hasattr(env, "data"):
        return env.model, env.data
    if hasattr(env, "mj_model") and hasattr(env, "mj_data"):
        return env.mj_model, env.mj_data
    if hasattr(env, "sim") and hasattr(env.sim, "model") and hasattr(env.sim, "data"):
        return env.sim.model, env.sim.data
    raise AttributeError("Cannot find mujoco model/data on env. Please expose env.model/env.data in UR5GraspEnv.")


def _renderer_get_ctx(renderer):
    """Try to fetch internal MjrContext from mujoco.Renderer across versions."""
    for name in ("ctx", "_ctx", "context", "_context", "_mjr_context", "mjr_context", "con", "_con"):
        if hasattr(renderer, name):
            return getattr(renderer, name)
    return None

def get_cam_fovy_rad(model, cam_name: str) -> float:
    """Return camera fovy in radians (robust to deg/rad conventions)."""
    cid = _cam_id(model, cam_name)
    fovy_raw = float(model.cam_fovy[cid])
    # Heuristic: if it looks like degrees, convert to rad.
    # (Typical rad fovy is < ~3.0; typical deg is 30~120)
    return np.deg2rad(fovy_raw) if fovy_raw > 3.2 else fovy_raw


def _depthbuf_to_meters(model, depth01: np.ndarray) -> np.ndarray:
    """OpenGL z-buffer (0..1 非线性) -> meters"""
    d = np.asarray(depth01, dtype=np.float32)
    d = np.clip(d, 0.0, 1.0)  # 关键：防止 >1 导致分母接近0炸飞
    znear = float(model.stat.extent) * float(model.vis.map.znear)
    zfar  = float(model.stat.extent) * float(model.vis.map.zfar)

    z_ndc = 2.0 * d - 1.0
    denom = (zfar + znear) - z_ndc * (zfar - znear)
    denom = np.maximum(denom, 1e-6)  # 放大一点，避免天文数字
    z = (2.0 * znear * zfar) / denom
    return z.astype(np.float32)

def render_from_camera(env, cam_name: str,
                       width: int = 640, height: int = 480,
                       depth_mode: str = "auto",
                       debug: bool = False):
    """
    Return:
      {"img": rgb_uint8(H,W,3), "depth": depth_meters(H,W)}
    """
    model, data = _get_mj_model_data(env)

    # cache renderer
    if not hasattr(env, "_mj_renderer") or env._mj_renderer is None:
        env._mj_renderer = mujoco.Renderer(model, height=height, width=width)
    r = env._mj_renderer

    cam = _cam_id(model, cam_name)

    # --- RGB ---
    if hasattr(r, "disable_depth_rendering"):
        r.disable_depth_rendering()
    if hasattr(r, "disable_segmentation_rendering"):
        r.disable_segmentation_rendering()

    r.update_scene(data, camera=cam)
    rgb = r.render()  # (H,W,3) uint8

    # --- DEPTH (MuJoCo official) ---
    r.enable_depth_rendering()
    r.update_scene(data, camera=cam)
    depth_raw = r.render().astype(np.float32)  # 官方 Renderer 的 depth 通常就是 meters
    r.disable_depth_rendering()

    mode = (depth_mode or "auto").lower()

    # ✅ 关键策略：用 Renderer 的 depth 时，auto 就当 meters（别再乱猜）
    if mode in ("auto", "meters", "m", "metric"):
        depth_m = depth_raw
    elif mode in ("zbuffer", "zbuf", "opengl"):
        depth_m = _depthbuf_to_meters(model, depth_raw)
    elif mode in ("zbuffer_rev", "zbuf_rev", "reversed"):
        depth_m = _depthbuf_to_meters(model, 1.0 - depth_raw)
    else:
        # 兜底
        depth_m = depth_raw

    # 简单的 sanity clamp（避免后面 percentile 反转）
    depth_m = np.where(np.isfinite(depth_m), depth_m, 0.0).astype(np.float32)
    depth_m = np.clip(depth_m, 0.0, 20.0)  # 20m 够用了

    if debug:
        print(f"[render] rgb={rgb.shape} "
              f"depth_raw(min/max)={float(depth_raw.min()):.6f}/{float(depth_raw.max()):.6f} "
              f"depth_m(min/max)={float(depth_m.min()):.6f}/{float(depth_m.max()):.6f} mode={depth_mode}")

    if cam_name == EE_CAM_NAME and hasattr(env, "_update_ee_side_camera_window"):
        try:
            env._update_ee_side_camera_window(force=True)
        except Exception:
            pass

    return {"img": rgb, "depth": depth_m}

def get_and_process_data(color_path, depth_path, mask_path, fovy_rad: float = None):
    """
    根据给定的 RGB 图、深度图、掩码图（可以是 文件路径 或 NumPy 数组），生成输入点云及其它必要数据

    修复点：
    - fovy 不能写死（以前是 pi/4），必须用真实相机 fovy（你 XML 里是 58°）
    - depth 必须是米单位（render_from_camera 已负责把 depth buffer 转成米）
    - 强制 GPU（pointnet2 不支持 CPU）
    """
#---------------------------------------
    # 1. 加载 color（可能是路径，也可能是数组）
    if isinstance(color_path, str):
        color = np.array(Image.open(color_path), dtype=np.float32) / 255.0
    elif isinstance(color_path, np.ndarray):
        color = color_path.astype(np.float32) / 255.0
    else:
        raise TypeError("color_path 既不是字符串路径也不是 NumPy 数组！")

    # 2. 加载 depth（可能是路径，也可能是数组）
    if isinstance(depth_path, str):
        depth = np.array(Image.open(depth_path), dtype=np.float32)
    elif isinstance(depth_path, np.ndarray):
        depth = depth_path.astype(np.float32)
    else:
        raise TypeError("depth_path 既不是字符串路径也不是 NumPy 数组！")

    # 3. 加载 mask（可能是路径，也可能是数组）
    if isinstance(mask_path, str):
        workspace_mask = np.array(Image.open(mask_path))
    elif isinstance(mask_path, np.ndarray):
        workspace_mask = mask_path
    else:
        raise TypeError("mask_path 既不是字符串路径也不是 NumPy 数组！")

    # 二值化 mask（0/255 或 0/1）
    if workspace_mask.ndim == 3:
        workspace_mask = workspace_mask[..., 0]
    workspace_mask = (workspace_mask > 0).astype(np.uint8)

    # --- 关键：确保 RGB / depth / mask 分辨率一致（不一致会直接导致点云“整体/局部偏移”） ---
    height, width = color.shape[0], color.shape[1]

    if depth.ndim == 3:
        depth = depth[..., 0]
    # 如果来自真实相机的 uint16 深度图（单位通常是 mm），这里统一转成 meters
    if np.issubdtype(depth.dtype, np.integer):
        depth = depth.astype(np.float32) / 1000.0
    if depth.shape != (height, width):
        depth = cv2.resize(depth, (width, height), interpolation=cv2.INTER_NEAREST)

    if workspace_mask.shape != (height, width):
        workspace_mask = cv2.resize(workspace_mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST)

    # 可选：mask 轻微腐蚀，减少边缘处的“飞点/混合像素”进入点云
    # 注意：对小目标（像苹果）盲目腐蚀会把有效像素吃掉，导致点云稀疏甚至“看起来像平面”。
    mask_area = int(np.count_nonzero(workspace_mask))
    kernel = np.ones((3, 3), np.uint8)
    if mask_area > 800:
        workspace_mask = cv2.erode(workspace_mask, kernel, iterations=1)

    # depth sanity check（如果还是 0..1 的 buffer，会导致尺度塌）
    dmin, dmax = float(np.nanmin(depth)), float(np.nanmax(depth))
    if (dmax <= 1.0 + 1e-3) and (dmin >= -1e-3) and (dmax < 0.05):
        raise RuntimeError(
            f"[get_and_process_data] depth 看起来仍是 depth buffer（未转米）。depth(min/max)={dmin:.4f}/{dmax:.4f}。"
            f"请检查 render_from_camera 是否在 env.render 分支也执行了 buffer->meters。"
        )
# 构造相机内参矩阵
    # 用真实 fovy（来自 model.cam_fovy / XML）
    if fovy_rad is None:
        fovy = np.pi / 4
    else:
        fovy = float(fovy_rad)

    # --- Intrinsics from MuJoCo camera fovy ---
    # MuJoCo's cam_fovy is *vertical* field of view.
    # So: fy from image height; fx adjusted by aspect ratio.
    fy = height / (2.0 * np.tan(fovy / 2.0))
    fx = fy * (width / float(height))
    c_x = width / 2.0
    c_y = height / 2.0
    intrinsic = np.array([
        [fx, 0.0, c_x],
        [0.0, fy, c_y],
        [0.0, 0.0, 1.0]
    ], dtype=np.float32)

    factor_depth = 1.0  # depth 已是米
    camera = CameraInfo(width, height, intrinsic[0][0], intrinsic[1][1], intrinsic[0][2], intrinsic[1][2], factor_depth)

    # 利用深度图生成点云 (H,W,3) 并保留组织结构
    cloud = create_point_cloud_from_depth_image(depth, camera, organized=True)

        # 过滤：mask + 合理深度 + 深度不连续(飞点)剔除
    valid = (workspace_mask > 0) & np.isfinite(depth) & (depth > 0)

    # 依据 mask 内部的深度分布做自适应截断（比固定 <2.0 更稳）
    if int(np.count_nonzero(valid)) > 0:
        dvals = depth[valid]
        lo, hi = [float(x) for x in np.percentile(dvals, [1, 99])]
        lo = max(0.05, lo - 0.02)
        hi = min(2.0, hi + 0.02)
    else:
        lo, hi = 0.05, 2.0

    # Depth discontinuity filter (edge / flying pixels)
    # 只在 mask 边界附近做“深度不连续”过滤，避免把曲面目标内部误删导致“空洞”。
    edge = np.zeros_like(depth, dtype=bool)
    dzdx = np.abs(depth[:, 1:] - depth[:, :-1])
    edge[:, 1:] |= (dzdx > 0.03)
    edge[:, :-1] |= (dzdx > 0.03)
    dzdy = np.abs(depth[1:, :] - depth[:-1, :])
    edge[1:, :] |= (dzdy > 0.03)
    edge[:-1, :] |= (dzdy > 0.03)

    boundary = cv2.dilate(workspace_mask.astype(np.uint8), kernel, iterations=1) ^                cv2.erode(workspace_mask.astype(np.uint8), kernel, iterations=1)
    boundary = boundary.astype(bool)

    mask = valid & (depth >= lo) & (depth <= hi) & (~(edge & boundary))
    # Debug：检查 mask 内深度是否“几乎是常数”（常见于 depth 语义/尺度错误 或 mask 与 depth 分辨率不一致）
    d_in = depth[mask]
    if d_in.size > 0:
        d_std = float(np.std(d_in))
        d_rng = float(np.max(d_in) - np.min(d_in))
        if d_std < 1e-4 and d_rng < 5e-4:
            print(
                f"[Warn] mask 内深度几乎不变：std={d_std:.6f} range={d_rng:.6f}。"
                "如果重建像平面/透视拉伸，请优先检查 render_from_camera 的 depth_mode。"
            )

    if int(np.count_nonzero(mask)) == 0:
        raise RuntimeError(
            f"[get_and_process_data] 有效点为0：mask可能为空 / 深度异常 / 深度边缘(飞点)过滤过严。"
            f" mask_sum={int(np.count_nonzero(workspace_mask))}, valid_sum={int(np.count_nonzero(valid))}, "
            f"depth(min/max)={dmin:.4f}/{dmax:.4f}, depth_clip=[{lo:.3f},{hi:.3f}]"
        )

    cloud_masked = cloud[mask]
    color_masked = color[mask]

    # Open3D 统计离群点剔除：能有效去掉那种“偏移很远的孤立点簇”
    # (这些点往往来自深度不连续/边缘混合像素/少量错误深度)
    if len(cloud_masked) >= 50:
        _pcd_tmp = o3d.geometry.PointCloud()
        _pcd_tmp.points = o3d.utility.Vector3dVector(cloud_masked.astype(np.float32))
        _pcd_tmp.colors = o3d.utility.Vector3dVector(color_masked.astype(np.float32))
        _, ind = _pcd_tmp.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        ind = np.asarray(ind, dtype=np.int64)
        if ind.size > 0:
            cloud_masked = cloud_masked[ind]
            color_masked = color_masked[ind]

    NUM_POINT = 3000
    if len(cloud_masked) >= NUM_POINT:
        idxs = np.random.choice(len(cloud_masked), NUM_POINT, replace=False)
    else:
        idxs1 = np.arange(len(cloud_masked))
        idxs2 = np.random.choice(len(cloud_masked), NUM_POINT - len(cloud_masked), replace=True)
        idxs = np.concatenate([idxs1, idxs2], axis=0)

    cloud_sampled = cloud_masked[idxs]
    color_sampled = color_masked[idxs]

    cloud_o3d = o3d.geometry.PointCloud()
    cloud_o3d.points = o3d.utility.Vector3dVector(cloud_masked.astype(np.float32))
    cloud_o3d.colors = o3d.utility.Vector3dVector(color_masked.astype(np.float32))

    # 强制 GPU（pointnet2 不支持 CPU）
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA不可用：GraspNet(pointnet2)算子不支持CPU。请确认 torch.cuda.is_available()==True。")
    device = torch.device("cuda:0")

    cloud_sampled = torch.from_numpy(cloud_sampled[np.newaxis].astype(np.float32)).contiguous().to(device, non_blocking=True)
    color_sampled = torch.from_numpy(color_sampled[np.newaxis].astype(np.float32)).contiguous().to(device, non_blocking=True)

    end_points = dict()
    end_points['point_clouds'] = cloud_sampled
    end_points['cloud_colors'] = color_sampled

    return end_points, cloud_o3d


# ================= Active top-view NBV helpers =================
def _clamp01(x):
    return np.clip(x, 0.0, 1.0)


def _estimate_normals_and_curvature(points_w: np.ndarray, radius: float = 0.02, max_nn: int = 30):
    if len(points_w) < 20:
        normals = np.zeros((len(points_w), 3), dtype=np.float32)
        curv = np.zeros((len(points_w),), dtype=np.float32)
        return normals, curv
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_w.astype(np.float64))
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=float(radius), max_nn=int(max_nn)))
    normals = np.asarray(pcd.normals, dtype=np.float32)
    kdt = o3d.geometry.KDTreeFlann(pcd)
    pts = np.asarray(pcd.points)
    curv = np.zeros((len(pts),), dtype=np.float32)
    for i in range(len(pts)):
        _, idx, _ = kdt.search_hybrid_vector_3d(pts[i], radius=float(radius), max_nn=int(max_nn))
        if len(idx) < 5:
            continue
        nb = pts[idx]
        mu = nb.mean(axis=0, keepdims=True)
        X = nb - mu
        C = (X.T @ X) / max(len(nb) - 1, 1)
        w, _ = np.linalg.eigh(C)
        w = np.maximum(w, 1e-12)
        curv[i] = float(w.min() / w.sum())
    return normals, curv


def _local_density_score(points_w: np.ndarray, k: int = 20):
    if len(points_w) < max(5, k + 1):
        return np.ones((len(points_w),), dtype=np.float32)
    tree = cKDTree(points_w)
    d, _ = tree.query(points_w, k=int(k))
    dk = d[:, -1]
    dk = (dk - dk.min()) / max(dk.max() - dk.min(), 1e-9)
    return dk.astype(np.float32)


def _mask_edge_weight(mask: np.ndarray, ksize: int = 5):
    m = (mask > 0).astype(np.uint8) * 255
    kernel = np.ones((ksize, ksize), np.uint8)
    er = cv2.erode(m, kernel, iterations=1)
    di = cv2.dilate(m, kernel, iterations=1)
    edge = ((di - er) > 0).astype(np.float32)
    return edge


def _depth_discontinuity_weight(depth: np.ndarray, mask: np.ndarray, ksize: int = 3):
    d = depth.astype(np.float32).copy()
    d[mask <= 0] = 0.0
    gx = cv2.Sobel(d, cv2.CV_32F, 1, 0, ksize=ksize)
    gy = cv2.Sobel(d, cv2.CV_32F, 0, 1, ksize=ksize)
    g = np.sqrt(gx * gx + gy * gy)
    if float(np.max(g)) > 1e-8:
        g = g / float(np.max(g))
    return g.astype(np.float32)


def _build_cam_intrinsics_from_fovy(width: int, height: int, fovy_rad: float) -> np.ndarray:
    fy = height / (2.0 * np.tan(float(fovy_rad) / 2.0))
    fx = fy * (width / float(height))
    cx = width / 2.0
    cy = height / 2.0
    return np.array([[fx, 0.0, cx],
                     [0.0, fy, cy],
                     [0.0, 0.0, 1.0]], dtype=np.float32)


def _sample_image_weight_for_points(weight_img: np.ndarray, mask: np.ndarray, depth_m: np.ndarray, fovy_rad: float):
    # Use the same organized point creation as get_and_process_data, then sample with the same valid mask.
    h, w = mask.shape[:2]
    K = _build_cam_intrinsics_from_fovy(w, h, float(fovy_rad))
    camera = CameraInfo(w, h, K[0, 0], K[1, 1], K[0, 2], K[1, 2], 1.0)
    cloud = create_point_cloud_from_depth_image(depth_m.astype(np.float32), camera, organized=True)
    valid = (mask > 0) & np.isfinite(depth_m) & (depth_m > 0)
    pts = cloud[valid].astype(np.float32)
    vals = weight_img[valid].astype(np.float32)
    return pts, vals


def build_grasp_sensitive_heatmap(points_w: np.ndarray,
                                  normals_w: np.ndarray,
                                  curvature: np.ndarray,
                                  edge_w_pts: np.ndarray = None,
                                  sparse_score: np.ndarray = None,
                                  coarse_grasps=None):
    curv = curvature.astype(np.float32).copy()
    if len(curv) > 0 and float(np.max(curv)) > 1e-8:
        curv = curv / float(np.max(curv))
    edge = np.zeros((len(points_w),), dtype=np.float32) if edge_w_pts is None else _clamp01(edge_w_pts.astype(np.float32))
    sparse = np.zeros((len(points_w),), dtype=np.float32) if sparse_score is None else _clamp01(sparse_score.astype(np.float32))
    heat = 0.55 * curv + 0.25 * edge + 0.20 * sparse
    return _clamp01(heat).astype(np.float32)


def _pca_main_axes_xy(points_w: np.ndarray):
    if len(points_w) < 10:
        return np.array([1.0, 0.0], dtype=np.float32), np.array([0.0, 1.0], dtype=np.float32)
    pts_xy = points_w[:, :2]
    mu = pts_xy.mean(axis=0, keepdims=True)
    X = pts_xy - mu
    C = (X.T @ X) / max(len(pts_xy) - 1, 1)
    w, V = np.linalg.eigh(C)
    idx = np.argsort(w)[::-1]
    major = V[:, idx[0]]
    minor = V[:, idx[1]]
    major = major / (np.linalg.norm(major) + 1e-9)
    minor = minor / (np.linalg.norm(minor) + 1e-9)
    return major.astype(np.float32), minor.astype(np.float32)


def _predict_visibility_from_viewdir(normals_w: np.ndarray, view_dir_w: np.ndarray):
    v = np.asarray(view_dir_w, dtype=np.float32)
    v = v / (np.linalg.norm(v) + 1e-9)
    vis = np.sum(normals_w * v[None, :], axis=1)
    return _clamp01(vis).astype(np.float32)


def predict_best_view_pair_from_topview(points_obj_w: np.ndarray,
                                        normals_obj_w: np.ndarray,
                                        curvature: np.ndarray,
                                        uncertainty: np.ndarray,
                                        grasp_heat: np.ndarray,
                                        prefer_surface_ratio: float = 0.4,
                                        occlusion_ratio: float = 0.3,
                                        grasp_ratio: float = 0.25,
                                        diversity_ratio: float = 0.05,
                                        debug: bool = False):
    major_xy, minor_xy = _pca_main_axes_xy(points_obj_w)
    front_dir = np.array([major_xy[0], major_xy[1], 0.0], dtype=np.float32)
    back_dir  = -front_dir
    left_dir  = np.array([minor_xy[0], minor_xy[1], 0.0], dtype=np.float32)
    right_dir = -left_dir

    vf = _predict_visibility_from_viewdir(normals_obj_w, front_dir)
    vb = _predict_visibility_from_viewdir(normals_obj_w, back_dir)
    vl = _predict_visibility_from_viewdir(normals_obj_w, left_dir)
    vr = _predict_visibility_from_viewdir(normals_obj_w, right_dir)

    U = _clamp01(uncertainty.astype(np.float32))
    G = _clamp01(grasp_heat.astype(np.float32))

    def _score_pair(va, vb):
        occ = float(np.mean(U * va) + np.mean(U * vb))
        gra = float(np.mean(G * U * va) + np.mean(G * U * vb))
        surf = float(np.mean(va) + np.mean(vb))
        div = 1.0
        score = occlusion_ratio * occ + grasp_ratio * gra + prefer_surface_ratio * surf + diversity_ratio * div
        return score, {'occ': occ, 'grasp': gra, 'surf': surf, 'div': div}

    s_fb, d_fb = _score_pair(vf, vb)
    s_lr, d_lr = _score_pair(vl, vr)

    best = 'front_back' if s_fb >= s_lr else 'left_right'
    labels = ('front', 'back') if best == 'front_back' else ('left', 'right')

    if debug:
        print(f"[ActiveNBV] major_xy={np.round(major_xy,4)} minor_xy={np.round(minor_xy,4)}")
        print(f"[ActiveNBV] score_front_back={s_fb:.4f} detail={d_fb}")
        print(f"[ActiveNBV] score_left_right={s_lr:.4f} detail={d_lr}")
        print(f"[ActiveNBV] best_pair={best}")

    return {
        'best_pair_name': best,
        'selected_labels': labels,
        'pair_scores': {'front_back': float(s_fb), 'left_right': float(s_lr)},
        'pair_details': {'front_back': d_fb, 'left_right': d_lr},
        'major_dir_xy': major_xy,
        'minor_dir_xy': minor_xy,
    }


def predict_best_view_pair_scene_aware(points_obj_w,
                                       normals_obj_w,
                                       curvature,
                                       uncertainty,
                                       grasp_heat,
                                       scene_context,
                                       candidate_num=8,
                                       debug=False):
    U = _clamp01(uncertainty.astype(np.float32))
    G = _clamp01(grasp_heat.astype(np.float32))

    center_w = scene_context["target_center_w"]
    ctx_pts = scene_context["context_points_w"]
    ctx_wts = scene_context["context_weights"]

    candidate_angles = np.linspace(0.0, 2.0 * np.pi, candidate_num, endpoint=False)

    if debug:
        print(f"[ActiveNBV][scene-aware] ctx_pts={ctx_pts.shape}, ctx_wts={ctx_wts.shape}")
        if ctx_pts.shape[0] == 0:
            print("[ActiveNBV][scene-aware] WARNING: scene context is empty; occlusion term will not affect view selection.")

    best_score = -1e9
    best_dir_xy = None
    records = []

    for ang in candidate_angles:
        d = np.array([np.cos(ang), np.sin(ang), 0.0], dtype=np.float32)
        d2 = -d

        v1 = _predict_visibility_from_viewdir(normals_obj_w, d)
        v2 = _predict_visibility_from_viewdir(normals_obj_w, d2)

        surf = float(np.mean(v1) + np.mean(v2))
        unc_gain = float(np.mean(U * v1) + np.mean(U * v2))
        grasp_gain = float(np.mean(G * U * v1) + np.mean(G * U * v2))

        occ1, clr1 = direction_context_penalty(center_w, ctx_pts, ctx_wts, d[:2])
        occ2, clr2 = direction_context_penalty(center_w, ctx_pts, ctx_wts, d2[:2])

        occ_pen = 0.5 * (occ1 + occ2)
        clearance = 0.5 * (clr1 + clr2)

        score = (
            0.30 * surf +
            0.22 * unc_gain +
            0.20 * grasp_gain +
            0.15 * clearance -
            0.30 * occ_pen
        )

        records.append({
            "name": f"pair_{int(np.rad2deg(ang)) % 360:03d}",
            "score": float(score),
            "dir_xy": d[:2].copy(),
            "detail": {
                "surf": float(surf),
                "unc_gain": float(unc_gain),
                "grasp_gain": float(grasp_gain),
                "clearance": float(clearance),
                "occ_penalty": float(occ_pen),
            }
        })

        if score > best_score:
            best_score = score
            best_dir_xy = d[:2].copy()

    if debug:
        recs = sorted(records, key=lambda x: x["score"], reverse=True)
        print("[ActiveNBV][scene-aware] top candidates:")
        for r in recs[:4]:
            print(r["name"], r["score"], r["detail"])
        max_occ = max((r["detail"]["occ_penalty"] for r in records), default=0.0)
        if max_occ <= 1e-6:
            print("[ActiveNBV][scene-aware] WARNING: all occ_penalty values are 0.0; surrounding-object influence is likely absent in this frame.")

    return {
        "best_pair_name": "scene_aware_pair",
        "orbit_horizontal_dir_xy": best_dir_xy,
        "candidate_records": records,
    }
# ================= Multi-view fusion helpers =================

# ---- Verification / alignment helpers (added) ----
def _nn_stats(src_pts: np.ndarray, tgt_pts: np.ndarray):
    """Nearest-neighbor distance stats from src -> tgt (meters)."""
    src = np.asarray(src_pts, dtype=np.float32).reshape(-1, 3)
    tgt = np.asarray(tgt_pts, dtype=np.float32).reshape(-1, 3)
    if src.size == 0 or tgt.size == 0:
        return {"median": float("nan"), "p95": float("nan"), "max": float("nan"), "inlier@1cm": 0.0}
    tree = cKDTree(tgt)
    d, _ = tree.query(src, k=1, workers=-1)
    d = np.asarray(d, dtype=np.float32)
    return {
        "median": float(np.median(d)),
        "p95": float(np.quantile(d, 0.95)),
        "max": float(np.max(d)),
        "inlier@1cm": float(np.mean(d < 0.01)),
    }


def _bbox_stats(pts: np.ndarray):
    pts = np.asarray(pts, dtype=np.float32).reshape(-1, 3)
    if pts.shape[0] == 0:
        z = np.zeros((3,), dtype=np.float32)
        return {"center": z, "extent": z}
    bmin = np.min(pts, axis=0)
    bmax = np.max(pts, axis=0)
    return {
        "center": ((bmin + bmax) * 0.5).astype(np.float32),
        "extent": (bmax - bmin).astype(np.float32),
    }

def _apply_T(pts: np.ndarray, T4: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float32).reshape(-1, 3)
    T4 = np.asarray(T4, dtype=np.float64).reshape(4, 4)
    ones = np.ones((pts.shape[0], 1), dtype=np.float64)
    ph = np.concatenate([pts.astype(np.float64), ones], axis=1)
    out = (ph @ T4.T)[:, :3]
    return out.astype(np.float32)

def _icp_multiscale(src_pts: np.ndarray,
                    tgt_pts: np.ndarray,
                    voxel: float = 0.003,
                    thr_list=(0.06, 0.04, 0.025, 0.015),
                    max_iter: int = 40):
    """Multi-scale point-to-point ICP.

    Important: Open3D's `fitness` is computed w.r.t the *current threshold*.
    If you end with a very small threshold, fitness can drop to ~0 even though
    earlier coarse stages aligned well. We therefore keep the best stage result.
    Returns (T_best, fitness_best, rmse_best).
    """
    src_pts = np.asarray(src_pts, dtype=np.float32).reshape(-1, 3)
    tgt_pts = np.asarray(tgt_pts, dtype=np.float32).reshape(-1, 3)
    if src_pts.shape[0] < 30 or tgt_pts.shape[0] < 30:
        return np.eye(4), 0.0, 0.0

    src = o3d.geometry.PointCloud()
    src.points = o3d.utility.Vector3dVector(src_pts.astype(np.float64))
    tgt = o3d.geometry.PointCloud()
    tgt.points = o3d.utility.Vector3dVector(tgt_pts.astype(np.float64))

    if voxel is not None and voxel > 0:
        src_d = src.voxel_down_sample(voxel_size=float(voxel))
        tgt_d = tgt.voxel_down_sample(voxel_size=float(voxel))
    else:
        src_d, tgt_d = src, tgt

    T = np.eye(4, dtype=np.float64)

    best_T = T.copy()
    best_fit = -1.0
    best_rmse = 0.0

    for thr in thr_list:
        reg = o3d.pipelines.registration.registration_icp(
            src_d, tgt_d,
            max_correspondence_distance=float(thr),
            init=T,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=int(max_iter)),
        )
        T = reg.transformation

        # track the best stage by fitness (coarse-to-fine)
        if float(reg.fitness) > best_fit:
            best_fit = float(reg.fitness)
            best_rmse = float(reg.inlier_rmse)
            best_T = T.copy()

    if best_fit < 0:
        return np.eye(4), 0.0, 0.0
    return best_T.astype(np.float64), float(best_fit), float(best_rmse)

def _nn_gate(src_pts: np.ndarray, tgt_pts: np.ndarray, thr: float):
    """Keep src points whose NN distance to tgt is <= thr."""
    src = np.asarray(src_pts, dtype=np.float32).reshape(-1, 3)
    tgt = np.asarray(tgt_pts, dtype=np.float32).reshape(-1, 3)
    if src.size == 0 or tgt.size == 0:
        return src, np.zeros((src.shape[0],), dtype=bool)
    tree = cKDTree(tgt)
    d, _ = tree.query(src, k=1, workers=-1)
    keep = (d <= float(thr))
    return src[keep], keep


def _target_translation_calibration(src_pts: np.ndarray,
                                    tgt_pts: np.ndarray,
                                    coarse_thr: float = 0.04,
                                    max_shift: float = 0.030,
                                    min_pairs: int = 60,
                                    min_improve: float = 0.08):
    """Estimate a small translation that reduces side/center target-cloud mismatch."""
    src = np.asarray(src_pts, dtype=np.float32).reshape(-1, 3)
    tgt = np.asarray(tgt_pts, dtype=np.float32).reshape(-1, 3)
    if src.shape[0] < int(min_pairs) or tgt.shape[0] < int(min_pairs):
        return src, {
            "applied": False,
            "reason": "too_few_points",
            "pairs": 0,
            "shift_norm": 0.0,
            "before": _nn_stats(src, tgt),
            "after": _nn_stats(src, tgt),
        }
    tree = cKDTree(tgt)
    d, nn = tree.query(src, k=1, workers=-1)
    d = np.asarray(d, dtype=np.float32)
    keep = np.isfinite(d) & (d <= float(coarse_thr))
    pair_count = int(np.count_nonzero(keep))
    before = _nn_stats(src, tgt)
    if pair_count < int(min_pairs):
        return src, {
            "applied": False,
            "reason": "too_few_pairs",
            "pairs": pair_count,
            "shift_norm": 0.0,
            "before": before,
            "after": before,
        }
    matched_tgt = tgt[np.asarray(nn[keep], dtype=np.int64)]
    matched_src = src[keep]
    delta = np.median(matched_tgt.astype(np.float64) - matched_src.astype(np.float64), axis=0)
    shift_norm = float(np.linalg.norm(delta))
    if (not np.isfinite(shift_norm)) or shift_norm <= 1e-6:
        return src, {
            "applied": False,
            "reason": "zero_shift",
            "pairs": pair_count,
            "shift_norm": shift_norm,
            "before": before,
            "after": before,
        }
    if shift_norm > float(max_shift):
        delta = delta * (float(max_shift) / max(shift_norm, 1e-9))
        shift_norm = float(max_shift)
    calibrated = (src.astype(np.float64) + delta.reshape(1, 3)).astype(np.float32)
    after = _nn_stats(calibrated, tgt)
    before_med = float(before.get("median", np.inf))
    after_med = float(after.get("median", np.inf))
    before_in = float(before.get("inlier@1cm", 0.0))
    after_in = float(after.get("inlier@1cm", 0.0))
    med_improve = (before_med - after_med) / max(before_med, 1e-6) if np.isfinite(before_med) else 0.0
    inlier_drop_tol = float(os.environ.get("MV_TARGET_CALIB_MAX_INLIER_DROP", "0.03"))
    keeps_fine_overlap = bool(after_in + inlier_drop_tol >= before_in)
    accept = bool(
        np.isfinite(after_med)
        and keeps_fine_overlap
        and (
            med_improve >= float(min_improve)
            or after_in >= before_in + 0.08
            or (after_med <= 0.010 and after_in >= before_in)
        )
    )
    return (calibrated if accept else src), {
        "applied": accept,
        "reason": "accepted" if accept else ("fine_overlap_worse" if not keeps_fine_overlap else "no_improvement"),
        "pairs": pair_count,
        "shift": delta.astype(np.float32),
        "shift_norm": shift_norm,
        "before": before,
        "after": after,
    }


def _target_rigid_calibration(src_pts: np.ndarray,
                              tgt_pts: np.ndarray,
                              voxel: float = 0.0025,
                              thr_list=(0.018, 0.012, 0.008),
                              max_iter: int = 25,
                              max_trans: float = 0.010,
                              max_rot_deg: float = 3.0,
                              min_improve: float = 0.04,
                              min_pairs: int = 80):
    """Estimate a conservative SE(3) correction for side/center target-cloud mismatch."""
    src = np.asarray(src_pts, dtype=np.float32).reshape(-1, 3)
    tgt = np.asarray(tgt_pts, dtype=np.float32).reshape(-1, 3)
    before = _nn_stats(src, tgt)
    if src.shape[0] < int(min_pairs) or tgt.shape[0] < int(min_pairs):
        return src, {
            "applied": False,
            "reason": "too_few_points",
            "T": np.eye(4, dtype=np.float32),
            "fit": 0.0,
            "rmse": 0.0,
            "trans": 0.0,
            "rot_deg": 0.0,
            "overlap_pairs": 0,
            "overlap_ratio": 0.0,
            "src_bbox": _bbox_stats(src),
            "tgt_bbox": _bbox_stats(tgt),
            "before": before,
            "after": before,
        }

    overlap_thr = float(os.environ.get("MV_TARGET_RIGID_OVERLAP_THR", "0.018"))
    src_overlap, overlap_keep = _nn_gate(src, tgt, thr=overlap_thr)
    overlap_pairs = int(src_overlap.shape[0])
    overlap_ratio = float(overlap_pairs / max(int(src.shape[0]), 1))
    min_overlap_ratio = float(os.environ.get("MV_TARGET_RIGID_MIN_OVERLAP_RATIO", "0.08"))
    if overlap_pairs < int(min_pairs) or overlap_ratio < min_overlap_ratio:
        return src, {
            "applied": False,
            "reason": "too_few_overlap_pairs",
            "T": np.eye(4, dtype=np.float32),
            "fit": 0.0,
            "rmse": 0.0,
            "trans": 0.0,
            "rot_deg": 0.0,
            "overlap_pairs": overlap_pairs,
            "overlap_ratio": overlap_ratio,
            "src_bbox": _bbox_stats(src),
            "tgt_bbox": _bbox_stats(tgt),
            "before": before,
            "after": before,
        }

    T_icp, fit, rmse = _icp_multiscale(
        src_overlap,
        tgt,
        voxel=float(voxel),
        thr_list=tuple(float(x) for x in thr_list),
        max_iter=int(max_iter),
    )
    R_icp = np.asarray(T_icp[:3, :3], dtype=np.float64)
    t_icp = np.asarray(T_icp[:3, 3], dtype=np.float64).reshape(3,)
    tr = np.clip((np.trace(R_icp) - 1.0) * 0.5, -1.0, 1.0)
    rot_deg = float(np.degrees(np.arccos(tr)))
    trans = float(np.linalg.norm(t_icp))
    calibrated = _apply_T(src, T_icp)
    after = _nn_stats(calibrated, tgt)

    before_med = float(before.get("median", np.inf))
    after_med = float(after.get("median", np.inf))
    before_p95 = float(before.get("p95", np.inf))
    after_p95 = float(after.get("p95", np.inf))
    before_in = float(before.get("inlier@1cm", 0.0))
    after_in = float(after.get("inlier@1cm", 0.0))
    med_improve = (before_med - after_med) / max(before_med, 1e-6) if np.isfinite(before_med) else 0.0
    p95_improve = (before_p95 - after_p95) / max(before_p95, 1e-6) if np.isfinite(before_p95) else 0.0
    min_fit = float(os.environ.get("MV_TARGET_RIGID_MIN_FIT", "0.10"))
    inlier_drop = float(os.environ.get("MV_TARGET_RIGID_MAX_INLIER_DROP", "0.03"))
    accept = bool(
        np.isfinite(after_med)
        and float(fit) >= min_fit
        and trans <= float(max_trans)
        and rot_deg <= float(max_rot_deg)
        and after_in + inlier_drop >= before_in
        and (
            med_improve >= float(min_improve)
            or p95_improve >= float(min_improve)
            or after_in >= before_in + 0.08
        )
    )
    if trans > float(max_trans):
        reason = "diagnostic_large_translation"
    elif rot_deg > float(max_rot_deg):
        reason = "diagnostic_large_rotation"
    elif float(fit) < min_fit:
        reason = "low_fit"
    elif after_in + inlier_drop < before_in:
        reason = "fine_overlap_worse"
    elif not accept:
        reason = "no_improvement"
    else:
        reason = "accepted"
    return (calibrated if accept else src), {
        "applied": accept,
        "reason": reason,
        "T": T_icp.astype(np.float32),
        "fit": float(fit),
        "rmse": float(rmse),
        "trans": float(trans),
        "rot_deg": float(rot_deg),
        "overlap_pairs": overlap_pairs,
        "overlap_ratio": overlap_ratio,
        "src_bbox": _bbox_stats(src),
        "tgt_bbox": _bbox_stats(tgt),
        "before": before,
        "after": after,
    }


def _se3_Rt(T: sm.SE3):
    R = np.array(T.R, dtype=np.float32)
    t = np.array(T.t, dtype=np.float32).reshape(3,)
    return R, t

def _cam_to_world_points(T_wc: sm.SE3, pts_c: np.ndarray) -> np.ndarray:
    """Row-vector convention: p_w = p_c @ R^T + t"""
    R, t = _se3_Rt(T_wc)
    pts_c = np.asarray(pts_c, dtype=np.float32).reshape(-1, 3)
    return pts_c @ R.T + t[None, :]

def _world_to_cam_points(T_wc: sm.SE3, pts_w: np.ndarray) -> np.ndarray:
    """Row-vector convention: p_c = (p_w - t) @ R"""
    R, t = _se3_Rt(T_wc)
    pts_w = np.asarray(pts_w, dtype=np.float32).reshape(-1, 3)
    return (pts_w - t[None, :]) @ R


def _detect_cartesian_is_world(env) -> bool:
    """Best-effort check whether robot.get_cartesian()/move_cartesian use world or base frame."""
    robot = env.robot
    T_wb = robot.base
    model, data = _get_mj_model_data(env)
    try:
        T_now = robot.get_cartesian()
        best_a = find_best_mujoco_frame_match(model, data, T_now, topk=1)[0]
        best_b = find_best_mujoco_frame_match(model, data, T_wb * T_now, topk=1)[0]
        return bool(best_a[2] <= best_b[2])
    except Exception:
        return True


def _build_frame_converters(env):
    robot = env.robot
    T_wb = robot.base
    cartesian_is_world = _detect_cartesian_is_world(env)

    def to_cmd(T_world: sm.SE3) -> sm.SE3:
        return T_world if cartesian_is_world else (T_wb.inv() * T_world)

    def to_world(T_cmd: sm.SE3) -> sm.SE3:
        return T_cmd if cartesian_is_world else (T_wb * T_cmd)

    return cartesian_is_world, to_cmd, to_world


def _get_trac_ik_solver(env):
    """Lazily create the TRAC-IK solver used in place of robot.ikine for Nero7."""
    global _TRAC_IK_SOLVER, _TRAC_IK_INIT_ERROR, _TRAC_IK_CONFIG_PRINTED
    if os.environ.get("NERO_USE_TRAC_IK", "1").strip().lower() in {"0", "false", "no", "off"}:
        return None
    if getattr(env.robot, "dof", 0) != 7:
        return None
    if _TRAC_IK_SOLVER is not None:
        return _TRAC_IK_SOLVER
    if TracIKKinematic is None:
        if _TRAC_IK_INIT_ERROR is None:
            _TRAC_IK_INIT_ERROR = _TRAC_IK_IMPORT_ERROR
            print(f"[TracIK] unavailable; fallback to robot.ikine. import_error={type(_TRAC_IK_IMPORT_ERROR).__name__}")
        return None

    urdf_path = os.environ.get(
        "NERO_TRAC_IK_URDF",
        os.path.join(ROOT_DIR, "manipulator_grasp", "nero_urdf", "agx_arm_urdf", "nero", "urdf", "nero_description.urdf"),
    )
    base_link = os.environ.get("NERO_TRAC_IK_BASE_LINK", "base_link")
    tip_link = os.environ.get("NERO_TRAC_IK_TIP_LINK", "link7")
    try:
        _TRAC_IK_SOLVER = TracIKKinematic(
            urdf_path=urdf_path,
            base_link_name=base_link,
            target_link_name=tip_link,
            timeout=float(os.environ.get("NERO_TRAC_IK_TIMEOUT", "0.02")),
            epsilon=float(os.environ.get("NERO_TRAC_IK_EPSILON", "0.00001")),
            solver_type=os.environ.get("NERO_TRAC_IK_SOLVER_TYPE", "Distance"),
        )
        if not _TRAC_IK_CONFIG_PRINTED:
            print(f"[TracIK] enabled urdf={urdf_path} base={base_link} tip={tip_link}")
            _TRAC_IK_CONFIG_PRINTED = True
    except Exception as e:
        _TRAC_IK_INIT_ERROR = e
        print(f"[TracIK] init failed; fallback to robot.ikine. err={type(e).__name__}: {e}")
        return None
    return _TRAC_IK_SOLVER


def _solve_ik_for_pose(env, T_target_cmd: sm.SE3, q_seed=None, allow_fallback: bool = True) -> np.ndarray:
    """Solve IK for an existing command-frame target pose, preferring TRAC-IK for Nero7."""
    robot = env.robot
    solver = _get_trac_ik_solver(env)
    if solver is not None:
        try:
            T_target_cmd = T_target_cmd if isinstance(T_target_cmd, sm.SE3) else sm.SE3(T_target_cmd)
            cartesian_is_world = _detect_cartesian_is_world(env)
            T_target_base = (robot.base.inv() * T_target_cmd) if cartesian_is_world else T_target_cmd
            T_tool = getattr(robot, "_tool", sm.SE3())
            T_tip_base = T_target_base * T_tool.inv()
            target_pos = np.asarray(T_tip_base.t, dtype=float).reshape(3,)
            quat_xyzw = SciRot.from_matrix(np.asarray(T_tip_base.R, dtype=float)).as_quat()
            target_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]], dtype=float)
            if q_seed is None:
                q_seed = np.asarray(robot.get_joint(), dtype=float).reshape(-1)[:solver.num_joints]
            q = solver.solve_ik(
                target_position=target_pos,
                target_wxyz=target_wxyz,
                initial_guess=np.asarray(q_seed, dtype=float).reshape(-1)[:solver.num_joints],
                use_previous_solution=True,
            )
            if q is not None:
                q = np.asarray(q, dtype=float).reshape(-1)
                if q.size == getattr(robot, "dof", q.size) and np.isfinite(q).all():
                    q_low = np.asarray(getattr(robot, "q_lim_low", np.full_like(q, -np.inf)), dtype=float).reshape(-1)
                    q_up = np.asarray(getattr(robot, "q_lim_up", np.full_like(q, np.inf)), dtype=float).reshape(-1)
                    n = min(q.size, q_low.size, q_up.size)
                    q[:n] = np.clip(q[:n], q_low[:n], q_up[:n])
                    return q
        except Exception as e:
            if VERBOSE_LOG:
                print(f"[TracIK] solve failed; fallback={allow_fallback}. err={type(e).__name__}: {e}")

    if allow_fallback:
        return np.asarray(robot.ikine(T_target_cmd), dtype=float).reshape(-1)
    return np.array([])


def _move_cartesian_smooth(env, T_target_cmd: sm.SE3, duration: float = 1.0, settle_steps: int = 80):
    """Smooth Cartesian motion in the frame expected by robot.move_cartesian()."""
    robot = env.robot
    action = np.zeros(env.mj_model.nu)
    action[env.robot.dof] = float(getattr(env, "_last_gripper_cmd", 0.0))
    T_start = robot.get_cartesian()

    position_parameter = LinePositionParameter(T_start.t, T_target_cmd.t)
    attitude_parameter = OneAttitudeParameter(sm.SO3(T_start.R), sm.SO3(T_target_cmd.R))
    cartesian_parameter = CartesianParameter(position_parameter, attitude_parameter)
    velocity_parameter = QuinticVelocityParameter(duration)
    trajectory_parameter = TrajectoryParameter(cartesian_parameter, velocity_parameter)
    planner = TrajectoryPlanner(trajectory_parameter)

    time_step_num = round(duration / 0.002) + 1
    times = np.linspace(0.0, duration, time_step_num)
    for timei in times:
        if timei == 0.0:
            continue
        cart = planner.interpolate(timei)
        robot.move_cartesian(cart)
        joint = robot.get_joint()
        action[:len(joint)] = joint
        env.step(action)

    action[:env.robot.dof] = robot.get_joint()
    for _ in range(int(settle_steps)):
        env.step(action)


def _move_pose_smooth(env, T_target_cmd: sm.SE3, duration: float = 1.0, settle_steps: int = 80):
    """Move to a pose; Nero7 uses endpoint IK + joint interpolation for robustness."""
    robot = env.robot
    if getattr(robot, "dof", 0) == 7:
        q_saved = np.array(robot.get_joint(), dtype=float)
        q_tgt = _solve_ik_for_pose(env, T_target_cmd, q_seed=q_saved)
        robot.set_joint(q_saved)
        if q_tgt.size == robot.dof and np.isfinite(q_tgt).all():
            move_joint_smooth(env, q_tgt, duration=duration, settle_steps=settle_steps)
            return
        raise AssertionError("[MovePose] Nero7 endpoint IK unreachable.")
    _move_cartesian_smooth(env, T_target_cmd, duration=duration, settle_steps=settle_steps)


def _save_mujoco_robot_state(env):
    return {
        "robot_q": np.asarray(env.robot.get_joint(), dtype=float).copy(),
        "qpos": np.asarray(env.mj_data.qpos, dtype=float).copy(),
        "qvel": np.asarray(env.mj_data.qvel, dtype=float).copy(),
        "ctrl": np.asarray(env.mj_data.ctrl, dtype=float).copy(),
    }


def _restore_mujoco_robot_state(env, state):
    try:
        env.robot.set_joint(state["robot_q"])
        env.mj_data.qpos[:] = state["qpos"]
        env.mj_data.qvel[:] = state["qvel"]
        env.mj_data.ctrl[:] = state["ctrl"]
        mujoco.mj_forward(env.mj_model, env.mj_data)
    except Exception:
        pass


def _sync_robot_joint_to_mujoco(env, q, update_ctrl: bool = False):
    q = np.asarray(q, dtype=float).reshape(-1)
    env.robot.set_joint(q)
    q_robot = np.asarray(env.robot.get_joint(), dtype=float).reshape(-1)
    for i, joint_name in enumerate(getattr(env.robot, "joint_names", [])):
        if i >= q_robot.size:
            break
        jid = mujoco.mj_name2id(env.mj_model, mujoco.mjtObj.mjOBJ_JOINT, str(joint_name))
        if jid < 0:
            continue
        qadr = int(env.mj_model.jnt_qposadr[jid])
        dadr = int(env.mj_model.jnt_dofadr[jid])
        env.mj_data.qpos[qadr] = float(q_robot[i])
        if 0 <= dadr < env.mj_data.qvel.size:
            env.mj_data.qvel[dadr] = 0.0
    if update_ctrl and env.mj_data.ctrl.size >= q_robot.size:
        env.mj_data.ctrl[:q_robot.size] = q_robot
    mujoco.mj_forward(env.mj_model, env.mj_data)
    return q_robot


def _pose_error_metrics(T_ref: sm.SE3, T_act: sm.SE3):
    t_ref = np.asarray(T_ref.t, dtype=np.float32).reshape(3,)
    t_act = np.asarray(T_act.t, dtype=np.float32).reshape(3,)
    pos_err = float(np.linalg.norm(t_ref - t_act))
    ang_err_deg = float(np.rad2deg(rot_angle(np.asarray(T_ref.R, dtype=float), np.asarray(T_act.R, dtype=float))))
    return pos_err, ang_err_deg


def _move_camera_pose_closed_loop(env,
                                  cam_name: str,
                                  T_wc_des: sm.SE3,
                                  duration: float = 1.0,
                                  settle_steps: int = 120,
                                  pos_tol: float = 0.004,
                                  ang_tol_deg: float = 1.5,
                                  max_iters: int = 3,
                                  third_iter_pos_trigger: float = 0.006,
                                  third_iter_ang_trigger: float = 1.5,
                                  debug: bool = True):
    """Closed-loop camera-pose execution using measured MuJoCo camera pose."""
    global _CAM_SERVO_CONFIG_PRINTED
    robot = env.robot
    _, to_cmd, to_world = _build_frame_converters(env)

    if debug and not _CAM_SERVO_CONFIG_PRINTED:
        print(
            f"[CamServoConfig] max_iters={int(max_iters)} pos_tol={float(pos_tol):.4f}m "
            f"ang_tol={float(ang_tol_deg):.2f}deg third_iter_if_pos_gt={float(third_iter_pos_trigger):.4f}m "
            f"or_ang_gt={float(third_iter_ang_trigger):.2f}deg"
        )
        _CAM_SERVO_CONFIG_PRINTED = True

    final_T_wc = get_T_wc_from_camera(env, cam_name)
    final_pos_err, final_ang_err = _pose_error_metrics(T_wc_des, final_T_wc)
    converged = False

    for it in range(int(max(max_iters, 1))):
        T_wc_now = get_T_wc_from_camera(env, cam_name)
        T_we_now_world = to_world(robot.get_cartesian())
        T_ec = T_we_now_world.inv() * T_wc_now
        T_we_des_world = T_wc_des * T_ec.inv()
        T_we_cmd = to_cmd(T_we_des_world)

        move_dur = float(duration if it == 0 else max(0.25, 0.5 * duration))
        if getattr(robot, "dof", 0) == 7:
            q_saved = np.asarray(robot.get_joint(), dtype=float).reshape(-1)
            q_tgt = _solve_ik_for_pose(env, T_we_cmd, q_seed=q_saved)
            robot.set_joint(q_saved)
            if q_tgt.size != robot.dof or not np.isfinite(q_tgt).all():
                raise AssertionError("[CamServo] Nero7 endpoint IK unreachable.")
            move_joint_smooth(env, q_tgt, duration=move_dur, settle_steps=settle_steps)
        else:
            _move_cartesian_smooth(env, T_we_cmd, duration=move_dur, settle_steps=settle_steps)

        final_T_wc = get_T_wc_from_camera(env, cam_name)
        final_pos_err, final_ang_err = _pose_error_metrics(T_wc_des, final_T_wc)
        if debug:
            print(
                f"[CamServo][{cam_name}] iter={it + 1}/{int(max(max_iters, 1))} "
                f"pos_err={final_pos_err:.4f}m ang_err={final_ang_err:.2f}deg"
            )
        if final_pos_err <= float(pos_tol) and final_ang_err <= float(ang_tol_deg):
            converged = True
            break
        if (
            int(max_iters) >= 3 and
            it == 1 and
            final_pos_err <= float(third_iter_pos_trigger) and
            final_ang_err <= float(third_iter_ang_trigger)
        ):
            if debug:
                print(
                    f"[CamServo][{cam_name}] skip iter=3 "
                    f"(pos_err={final_pos_err:.4f}m, ang_err={final_ang_err:.2f}deg)"
                )
            break

    return {
        "T_wc_actual": final_T_wc,
        "pos_err": float(final_pos_err),
        "ang_err_deg": float(final_ang_err),
        "iters": int(it + 1),
        "converged": bool(converged),
    }


def _look_at_T_wc(cam_pos_w: np.ndarray, target_w: np.ndarray, x_ref_w=(1.0, 0.0, 0.0), up_w=(0.0, 0.0, 1.0)) -> sm.SE3:
    """Build a CV-style camera pose T_wc that looks at target_w from cam_pos_w."""
    cam_pos_w = np.asarray(cam_pos_w, dtype=float).reshape(3,)
    target_w = np.asarray(target_w, dtype=float).reshape(3,)
    x_ref_w = np.asarray(x_ref_w, dtype=float).reshape(3,)
    up_w = np.asarray(up_w, dtype=float).reshape(3,)

    z = target_w - cam_pos_w
    nz = np.linalg.norm(z)
    if nz < 1e-9:
        raise ValueError('[look_at] cam_pos_w and target_w are too close.')
    z /= nz

    x = x_ref_w - z * float(np.dot(x_ref_w, z))
    nx = np.linalg.norm(x)
    if nx < 1e-6:
        if abs(float(np.dot(z, up_w))) > 0.98:
            up_w = np.array([0.0, 1.0, 0.0], dtype=float)
        x = np.cross(z, up_w)
        x /= (np.linalg.norm(x) + 1e-9)
    else:
        x /= nx

    y = np.cross(z, x)
    R = np.stack([x, y, z], axis=1)
    U, _, Vt = np.linalg.svd(R)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    return sm.SE3.Rt(R, cam_pos_w)


def _target_camera_metrics(T_wc: sm.SE3, target_w: np.ndarray):
    target_w = np.asarray(target_w, dtype=float).reshape(3,)
    p_c = _world_to_cam_points(T_wc, target_w[None, :])[0].astype(float)
    depth = float(p_c[2])
    center_norm = float(np.linalg.norm(p_c[:2]) / max(abs(depth), 1e-6))
    cam_pos = np.asarray(T_wc.t, dtype=float).reshape(3,)
    dist = float(np.linalg.norm(cam_pos - target_w))
    height = float(cam_pos[2] - target_w[2])
    visible = bool(depth > 0.05 and center_norm < float(os.environ.get("ACTIVE_NBV_SCAN_MAX_CENTER_NORM", "0.65")))
    return {
        "target_cam": p_c,
        "depth": depth,
        "center_norm": center_norm,
        "distance": dist,
        "height": height,
        "visible": visible,
    }


def _find_feasible_object_view_joint_scan(env,
                                          cam_name: str,
                                          target_w: np.ndarray,
                                          desired_dir_xy=None,
                                          q_seed=None,
                                          allow_far_fallback: bool = False):
    """Fast Nero7 fallback: scan valid joint poses instead of requiring exact IK."""
    robot = env.robot
    if getattr(robot, "dof", 0) != 7:
        return None
    target_w = np.asarray(target_w, dtype=float).reshape(3,)
    state = _save_mujoco_robot_state(env)
    q_cur = state["robot_q"].reshape(-1)
    q_lim_low = np.asarray(getattr(robot, "q_lim_low", np.full_like(q_cur, -np.inf)), dtype=float)
    q_lim_up = np.asarray(getattr(robot, "q_lim_up", np.full_like(q_cur, np.inf)), dtype=float)
    q7_pref = float(os.environ.get("NERO_IK_Q7_PREF", "-0.12"))
    base_seeds = [
        q_cur,
        np.array([0.0, 0.15, 0.0, 0.975, -1.6, -0.35, q7_pref], dtype=float),
        np.array([0.0, 0.349066, 0.0, 1.151917, -1.493363, 0.139626, q7_pref], dtype=float),
    ]
    if q_seed is not None:
        base_seeds.insert(0, np.asarray(q_seed, dtype=float).reshape(-1))

    base_offsets = [float(x) for x in os.environ.get(
        "ACTIVE_NBV_SCAN_BASE_OFFSETS",
        "0 -0.18 0.18 -0.38 0.38 -0.60 0.60 -0.85 0.85 -1.10 1.10",
    ).replace(",", " ").split()]
    shoulder_offsets = [float(x) for x in os.environ.get(
        "ACTIVE_NBV_SCAN_SHOULDER_OFFSETS",
        "0 -0.22 0.22 -0.45 0.45 -0.70 0.70",
    ).replace(",", " ").split()]
    joint3_offsets = [float(x) for x in os.environ.get(
        "ACTIVE_NBV_SCAN_JOINT3_OFFSETS",
        "0 -0.30 0.30 -0.60 0.60",
    ).replace(",", " ").split()]
    wrist_offsets = [float(x) for x in os.environ.get(
        "ACTIVE_NBV_SCAN_WRIST_OFFSETS",
        "0 -0.35 0.35 -0.60 0.60",
    ).replace(",", " ").split()]
    elbow_offsets = [float(x) for x in os.environ.get(
        "ACTIVE_NBV_SCAN_ELBOW_OFFSETS",
        "0 -0.18 0.18 -0.36 0.36",
    ).replace(",", " ").split()]
    wrist_pitch_offsets = [float(x) for x in os.environ.get(
        "ACTIVE_NBV_SCAN_WRIST_PITCH_OFFSETS",
        "0 -0.25 0.25 -0.50 0.50",
    ).replace(",", " ").split()]
    max_candidates = int(os.environ.get("ACTIVE_NBV_SCAN_MAX_CANDIDATES", "720"))

    desired_dist = float(os.environ.get("ACTIVE_NBV_SCAN_DESIRED_DIST", "0.20"))
    desired_height = float(os.environ.get("ACTIVE_NBV_SCAN_DESIRED_HEIGHT", "0.150"))
    desired_lateral = float(os.environ.get("ACTIVE_NBV_SCAN_DESIRED_LATERAL", "0.120"))
    min_lateral = float(os.environ.get("ACTIVE_NBV_SCAN_MIN_LATERAL", "0.100"))
    min_side_angle_deg = float(os.environ.get("ACTIVE_NBV_SCAN_MIN_SIDE_ANGLE_DEG", "25.0"))
    max_side_angle_deg = float(os.environ.get("ACTIVE_NBV_SCAN_MAX_SIDE_ANGLE_DEG", "45.0"))
    min_height = float(os.environ.get("ACTIVE_NBV_SCAN_MIN_HEIGHT", "0.110"))
    max_height = float(os.environ.get("ACTIVE_NBV_SCAN_MAX_HEIGHT", "0.280"))
    min_accept_dist = float(os.environ.get("ACTIVE_NBV_SCAN_MIN_ACCEPT_DIST", "0.160"))
    max_accept_dist = float(os.environ.get("ACTIVE_NBV_SCAN_MAX_ACCEPT_DIST", "0.280"))
    max_accept_height = float(os.environ.get("ACTIVE_NBV_SCAN_MAX_ACCEPT_HEIGHT", "0.300"))
    offset_combos = list(itertools.product(
        base_offsets,
        shoulder_offsets,
        joint3_offsets,
        elbow_offsets,
        wrist_offsets,
        wrist_pitch_offsets,
    ))
    offset_combos.sort(key=lambda x: (
        abs(x[0]) / 1.10 +
        abs(x[1]) / 0.70 +
        abs(x[2]) / 0.60 +
        abs(x[3]) / 0.36 +
        abs(x[4]) / 0.60 +
        abs(x[5]) / 0.50,
        max(abs(v) for v in x),
    ))
    desired_dir = None
    if desired_dir_xy is not None:
        desired_dir = np.asarray(desired_dir_xy, dtype=float).reshape(2,)
        nd = float(np.linalg.norm(desired_dir))
        desired_dir = desired_dir / nd if nd > 1e-6 else None

    cache_key = (
        tuple(np.round(target_w, 3).tolist()),
        tuple(np.round(q_cur[:robot.dof], 2).tolist()),
        int(max_candidates),
        float(round(desired_dist, 3)),
        float(round(max_accept_dist, 3)),
    )
    cache = getattr(env, "_active_nbv_joint_scan_cache", None)
    records = None
    checked = 0
    if isinstance(cache, dict) and cache.get("key") == cache_key:
        records = cache.get("records", None)
        checked = int(cache.get("checked", 0))
    if records is None:
        records = []
        near_hits = 0
        early_near_hits = int(os.environ.get("ACTIVE_NBV_SCAN_EARLY_NEAR_HITS", "10"))
        min_candidates_before_early = int(os.environ.get("ACTIVE_NBV_SCAN_MIN_CANDIDATES", "180"))
        try:
            for seed in base_seeds:
                q0 = np.asarray(seed, dtype=float).reshape(-1)
                if q0.size < getattr(robot, "dof", 7):
                    continue
                q0 = np.clip(q0[:robot.dof], q_lim_low[:robot.dof], q_lim_up[:robot.dof])
                for b, s, j3, e, w, wp in offset_combos:
                    q = q0.copy()
                    q[0] = q0[0] + b
                    q[1] = q0[1] + s
                    q[2] = q0[2] + j3
                    q[3] = q0[3] + e
                    q[4] = q0[4] - 0.45 * b + w
                    q[5] = q0[5] + wp
                    q[6] = q7_pref
                    q = np.clip(q, q_lim_low[:robot.dof], q_lim_up[:robot.dof])
                    _sync_robot_joint_to_mujoco(env, q, update_ctrl=False)
                    T_wc = get_T_wc_from_camera(env, cam_name)
                    m = _target_camera_metrics(T_wc, target_w)
                    cam_pos = np.asarray(T_wc.t, dtype=float).reshape(3,)
                    v_xy = cam_pos[:2] - target_w[:2]
                    nv = float(np.linalg.norm(v_xy))
                    dir_score0 = 1.0
                    if desired_dir is not None and nv > 1e-6:
                        dir_score0 = float(np.clip((np.dot(v_xy / nv, desired_dir) + 1.0) * 0.5, 0.0, 1.0))
                    if (
                        bool(m["visible"])
                        and float(m["distance"]) >= min_accept_dist
                        and float(m["distance"]) <= max_accept_dist
                        and min_height <= float(m["height"]) <= max_accept_height
                        and (desired_dir is None or dir_score0 >= float(os.environ.get("ACTIVE_NBV_SCAN_MIN_DIR_SCORE", "0.62")))
                    ):
                        near_hits += 1
                    jump = float(np.max(np.abs(q - q_cur[:robot.dof])))
                    rec = {
                        "q": q.copy(),
                        "T_wc": T_wc,
                        "q_jump": jump,
                        **m,
                    }
                    records.append(rec)
                    checked += 1
                    if (
                        checked >= max_candidates
                        or (near_hits >= early_near_hits and checked >= min_candidates_before_early)
                    ):
                        raise StopIteration
        except StopIteration:
            pass
        finally:
            _restore_mujoco_robot_state(env, state)
        env._active_nbv_joint_scan_cache = {"key": cache_key, "records": records, "checked": checked}
    else:
        _restore_mujoco_robot_state(env, state)

    scored = []
    best = None
    dir_weight = float(os.environ.get("ACTIVE_NBV_SCAN_DIR_WEIGHT", "7.0"))
    min_dir_score = float(os.environ.get("ACTIVE_NBV_SCAN_MIN_DIR_SCORE", "0.62"))
    for rec0 in records:
        rec = dict(rec0)
        cam_pos = np.asarray(rec["T_wc"].t, dtype=float).reshape(3,)
        v_xy = cam_pos[:2] - target_w[:2]
        nv = float(np.linalg.norm(v_xy))
        rec["lateral"] = nv
        dir_score = 0.5
        if desired_dir is not None and nv > 1e-6:
            dir_score = float(np.clip((np.dot(v_xy / nv, desired_dir) + 1.0) * 0.5, 0.0, 1.0))
        dist_err = abs(float(rec["distance"]) - desired_dist)
        height_err = abs(float(rec["height"]) - desired_height)
        lateral_err = abs(float(nv) - desired_lateral)
        side_angle_deg = float(np.rad2deg(np.arctan2(nv, max(float(rec["height"]), 1e-6))))
        height_penalty = 0.0 if min_height <= float(rec["height"]) <= max_height else 3.0
        lateral_penalty = 0.0 if nv >= min_lateral else 4.0 * (min_lateral - nv) / max(min_lateral, 1e-6)
        side_angle_penalty = 0.0 if side_angle_deg >= min_side_angle_deg else 4.0 * (min_side_angle_deg - side_angle_deg) / max(min_side_angle_deg, 1e-6)
        if side_angle_deg > max_side_angle_deg:
            side_angle_penalty += 5.0 * (side_angle_deg - max_side_angle_deg) / max(max_side_angle_deg, 1e-6)
        dir_penalty = 0.0 if desired_dir is None or dir_score >= min_dir_score else 3.0 * (min_dir_score - dir_score) / max(min_dir_score, 1e-6)
        far_penalty = 0.0
        if float(rec["distance"]) < min_accept_dist:
            far_penalty += 22.0 * (min_accept_dist - float(rec["distance"])) / max(min_accept_dist, 1e-6)
        if float(rec["distance"]) > max_accept_dist:
            far_penalty += 18.0 * (float(rec["distance"]) - max_accept_dist) / max(max_accept_dist, 1e-6)
        if float(rec["height"]) > max_accept_height:
            far_penalty += 10.0 * (float(rec["height"]) - max_accept_height) / max(max_accept_height, 1e-6)
        score = (
            2.2 * float(rec["center_norm"]) +
            7.0 * dist_err +
            3.0 * height_err +
            1.8 * lateral_err +
            height_penalty +
            lateral_penalty +
            side_angle_penalty +
            dir_penalty +
            far_penalty +
            0.18 * float(rec["q_jump"]) -
            dir_weight * dir_score
        )
        if not bool(rec["visible"]):
            score += 5.0
        rec["score"] = float(score)
        rec["dir_score"] = float(dir_score)
        rec["side_angle_deg"] = float(side_angle_deg)
        rec["near_enough"] = bool(
            min_accept_dist <= float(rec["distance"]) <= max_accept_dist
            and float(rec["height"]) <= max_accept_height
            and min_height <= float(rec["height"])
            and nv >= min_lateral
            and min_side_angle_deg <= side_angle_deg <= max_side_angle_deg
            and (desired_dir is None or dir_score >= min_dir_score)
            and bool(rec["visible"])
        )
        scored.append(rec)
        if best is None or rec["score"] < best["score"]:
            best = rec

    if best is None:
        return None
    near_pool = [rec for rec in scored if bool(rec.get("near_enough", False))]
    if len(near_pool) > 0:
        best = min(near_pool, key=lambda rec: rec["score"])
        best["near_fallback"] = False
    else:
        best["near_fallback"] = True
    if not bool(best["visible"]) and _env_flag("ACTIVE_NBV_SCAN_REQUIRE_VISIBLE", "1"):
        return None
    if bool(best.get("near_fallback", False)) and not allow_far_fallback and _env_flag("ACTIVE_NBV_SCAN_REQUIRE_NEAR", "1"):
        return None
    best["checked"] = int(checked)
    return best


def _rodrigues_rotate(v: np.ndarray, axis: np.ndarray, angle_rad: float) -> np.ndarray:
    """Rotate vector v around unit axis by angle_rad."""
    v = np.asarray(v, dtype=np.float64).reshape(3,)
    k = np.asarray(axis, dtype=np.float64).reshape(3,)
    k = k / (np.linalg.norm(k) + 1e-12)
    c = np.cos(float(angle_rad))
    s = np.sin(float(angle_rad))
    return (v * c + np.cross(k, v) * s + k * np.dot(k, v) * (1.0 - c)).astype(np.float64)


def _make_circle_orbit_pose(object_center_w: np.ndarray,
                            T_wc_center: sm.SE3,
                            orbit_horizontal_dir_xy: np.ndarray,
                            side_sign: float,
                            orbit_angle_rad: float,
                            min_radius: float = 0.10,
                            target_radius: float = None) -> sm.SE3:
    """
    Strict spherical-orbit pose generation around object center.

    Compared with the previous version:
      - Do NOT keep camera height fixed
      - Rotate the full 3D vector (camera - object_center)
      - Keep the full 3D radius constant
      - This makes the camera truly move from top-view toward side-view

    Args:
        object_center_w:
            object center in world
        T_wc_center:
            center-view camera pose
        orbit_horizontal_dir_xy:
            selected horizontal direction from NBV (2D)
        side_sign:
            +1 / -1 for the two symmetric side views
        orbit_angle_rad:
            orbit angle magnitude
        min_radius:
            lower bound for 3D radius
    """
    c = np.asarray(object_center_w, dtype=np.float64).reshape(3,)
    p0 = np.asarray(T_wc_center.t, dtype=np.float64).reshape(3,)

    # full 3D radius vector from object center to current camera
    v0 = p0 - c
    r = float(np.linalg.norm(v0))
    if r < 1e-6:
        v0 = np.array([0.0, 0.0, float(min_radius)], dtype=np.float64)
        r = float(min_radius)
    if r < min_radius:
        v0 = v0 / (np.linalg.norm(v0) + 1e-12) * float(min_radius)
        r = float(min_radius)
    if target_radius is not None and np.isfinite(float(target_radius)) and float(target_radius) > 1e-4:
        r = max(float(target_radius), float(min_radius))
        v0 = v0 / (np.linalg.norm(v0) + 1e-12) * r

    # desired horizontal exploration direction
    h = np.array([orbit_horizontal_dir_xy[0], orbit_horizontal_dir_xy[1], 0.0], dtype=np.float64)
    nh = np.linalg.norm(h)
    if nh < 1e-9:
        h = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    else:
        h = h / nh

    # build a rotation axis so that the top-view ray tilts toward +/- h
    # for a top-down center view, this will create a true side exploration
    axis = np.cross(v0, h)
    na = np.linalg.norm(axis)
    if na < 1e-9:
        # fallback: choose an axis orthogonal to v0
        z_axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        axis = np.cross(z_axis, v0)
        na = np.linalg.norm(axis)
        if na < 1e-9:
            axis = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        else:
            axis = axis / na
    else:
        axis = axis / na

    theta = float(side_sign) * float(orbit_angle_rad)

    # rotate the full 3D vector -> keep full 3D radius constant
    v_new = _rodrigues_rotate(v0, axis, theta)
    p_new = c + v_new

    T_new = _look_at_T_wc(p_new, c, x_ref_w=(1.0, 0.0, 0.0))
    return T_new


def _make_front_occluder_confirm_pose(occluder_center_w: np.ndarray,
                                      target_center_w: np.ndarray,
                                      front_dir_w: np.ndarray,
                                      T_wc_ref: sm.SE3,
                                      forward_offset: float = 0.11,
                                      lateral_offset: float = 0.10,
                                      up_offset: float = 0.16) -> sm.SE3:
    """
    Build a side-above confirmation pose for a front occluder.

    The target-view occluder center is usually biased toward the visible portion
    only. A side-above confirmation view separates obstacle/target before we
    trust that center for obstacle multiview.
    """
    occ = np.asarray(occluder_center_w, dtype=np.float64).reshape(3,)
    tgt = np.asarray(target_center_w, dtype=np.float64).reshape(3,)
    front = np.asarray(front_dir_w, dtype=np.float64).reshape(3,)
    cam_now = np.asarray(T_wc_ref.t, dtype=np.float64).reshape(3,)

    front[2] = 0.0
    n_front = np.linalg.norm(front)
    if n_front < 1e-8:
        front = occ - tgt
        front[2] = 0.0
        n_front = np.linalg.norm(front)
    if n_front < 1e-8:
        front = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    else:
        front = front / n_front

    lateral = np.cross(np.array([0.0, 0.0, 1.0], dtype=np.float64), front)
    n_lat = np.linalg.norm(lateral)
    if n_lat < 1e-8:
        lateral = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    else:
        lateral = lateral / n_lat

    to_target_xy = tgt - occ
    to_target_xy[2] = 0.0
    target_span_xy = float(np.linalg.norm(to_target_xy))
    forward = max(float(forward_offset), 0.55 * target_span_xy)
    lateral_mag = max(float(lateral_offset), 0.45 * target_span_xy)
    up = max(float(up_offset), 0.12)

    cand_offsets = [
        front * forward + lateral * lateral_mag,
        front * forward - lateral * lateral_mag,
    ]
    cand_offsets = sorted(cand_offsets, key=lambda off: np.linalg.norm((occ + off)[:2] - cam_now[:2]))

    cam_pos_w = occ + cand_offsets[0]
    cam_pos_w[2] = max(occ[2] + up, cam_now[2] - 0.02)
    return _look_at_T_wc(cam_pos_w, occ, x_ref_w=(1.0, 0.0, 0.0))


def _confirm_front_occluder_from_side_view(env,
                                           occ_center_seed_w,
                                           target_center_w,
                                           front_dir_w,
                                           fovy_rad,
                                           cam_name: str,
                                           debug: bool = True,
                                           debug_dir: str = None,
                                           iter_tag: str = "clear_confirm"):
    occ_center_seed_w = np.asarray(occ_center_seed_w, dtype=np.float32).reshape(3,)
    target_center_w = np.asarray(target_center_w, dtype=np.float32).reshape(3,)
    front_dir_w = np.asarray(front_dir_w, dtype=np.float32).reshape(3,)

    T_wc_ref = get_T_wc_from_camera(env, cam_name)
    T_wc_confirm = _make_front_occluder_confirm_pose(
        occ_center_seed_w,
        target_center_w,
        front_dir_w,
        T_wc_ref=T_wc_ref,
    )
    if debug:
        print(
            f"[FrontOcc] side-confirm seed_center={np.round(occ_center_seed_w, 4)} "
            f"target_center={np.round(target_center_w, 4)} "
            f"cam_pos={np.round(np.asarray(T_wc_confirm.t, dtype=float), 4)}"
        )
    _move_camera_pose_closed_loop(
        env,
        cam_name=cam_name,
        T_wc_des=T_wc_confirm,
        duration=1.0,
        settle_steps=60,
        pos_tol=0.004,
        ang_tol_deg=1.8,
        max_iters=3,
        third_iter_pos_trigger=0.006,
        third_iter_ang_trigger=1.5,
        debug=debug,
    )

    imgs_occ = render_from_camera(env, cam_name, depth_mode='auto', debug=debug)
    rgb_occ = imgs_occ["img"]
    depth_occ = imgs_occ["depth"]
    T_wc_occ = get_T_wc_from_camera(env, cam_name)
    bgr_occ = cv2.cvtColor(rgb_occ, cv2.COLOR_RGB2BGR)
    if debug_dir is not None:
        save_debug_image(
            bgr_occ,
            _debug_iter_path(debug_dir, iter_tag, "rgb"),
            title=f"{iter_tag}: rgb",
        )

    h_occ, w_occ = bgr_occ.shape[:2]
    K_occ = _build_cam_intrinsics_from_fovy(w_occ, h_occ, fovy_rad)
    camera_occ = CameraInfo(w_occ, h_occ, K_occ[0, 0], K_occ[1, 1], K_occ[0, 2], K_occ[1, 2], 1.0)
    uv_occ, valid_occ = _project_world_points_to_image(occ_center_seed_w[None, :], T_wc_occ, camera_occ)
    if not np.any(valid_occ):
        return None

    click_uv = tuple(int(round(v)) for v in uv_occ[0])
    if debug:
        print(f"[FrontOcc] side-confirm click_uv={click_uv}")
    try:
        occ_inst = create_instance_from_click(bgr_occ, depth_occ, click_uv, camera_occ, T_wc_occ, role="target")
    except Exception:
        return None

    occ_mask = occ_inst.get("mask", None)
    pts_w = np.asarray(occ_inst.get("points_w", np.zeros((0, 3), dtype=np.float32)), dtype=np.float32).reshape(-1, 3)
    if occ_mask is None or int(np.count_nonzero(occ_mask)) == 0 or pts_w.shape[0] < 80:
        return None

    occ_center_refined_w = occ_inst.get("bbox_center", None)
    if occ_center_refined_w is None:
        occ_center_refined_w = np.median(pts_w, axis=0).astype(np.float32)
    occ_center_refined_w = np.asarray(occ_center_refined_w, dtype=np.float32).reshape(3,)
    if debug:
        print(
            f"[FrontOcc] side-confirm refined_center={np.round(occ_center_refined_w, 4)} "
            f"points={pts_w.shape[0]}"
        )

    return {
        "bgr": bgr_occ,
        "depth": depth_occ,
        "T_wc": T_wc_occ,
        "camera": camera_occ,
        "click_uv": click_uv,
        "instance": occ_inst,
        "center_w": occ_center_refined_w,
    }


def estimate_object_center_from_mask(bgr: np.ndarray,
                                     depth_m: np.ndarray,
                                     mask: np.ndarray,
                                     T_wc: sm.SE3,
                                     fovy_rad: float):
    """Estimate object center from the segmented object.

    Returns a dict with:
      - center_uv: mask centroid in image pixels (u, v)
      - center_c : robust 3D center in camera frame
      - center_w : robust 3D center in world frame
      - cloud_center: masked object cloud (Open3D, camera frame)
    """
    if mask is None:
        raise RuntimeError('[object-center] segmentation mask is None.')

    if mask.ndim == 3:
        mask_bin = (mask[..., 0] > 0).astype(np.uint8)
    else:
        mask_bin = (mask > 0).astype(np.uint8)

    ys, xs = np.nonzero(mask_bin)
    if len(xs) == 0:
        raise RuntimeError('[object-center] empty segmentation mask.')

    u = float(np.mean(xs))
    v = float(np.mean(ys))

    _, cloud_center = get_and_process_data(bgr, depth_m, mask_bin, fovy_rad=fovy_rad)
    pts_c = np.asarray(cloud_center.points, dtype=np.float32)
    if pts_c.shape[0] == 0:
        raise RuntimeError('[object-center] masked point cloud is empty.')

    center_c = np.median(pts_c, axis=0).astype(np.float32)
    center_w = _cam_to_world_points(T_wc, center_c[None, :])[0].astype(np.float32)

    return {
        'center_uv': np.array([u, v], dtype=np.float32),
        'center_c': center_c,
        'center_w': center_w,
        'cloud_center': cloud_center,
    }


def _object_center_sanity_check(obj_info, image_shape):
    center_w = np.asarray(obj_info.get("center_w", np.zeros(3)), dtype=float).reshape(3,)
    center_uv = np.asarray(obj_info.get("center_uv", np.zeros(2)), dtype=float).reshape(2,)
    h, w = int(image_shape[0]), int(image_shape[1])
    edge_margin = float(os.environ.get("OBJ_CENTER_EDGE_MARGIN_PX", "35"))
    workspace = (
        float(os.environ.get("OBJ_CENTER_MIN_X", "1.05")),
        float(os.environ.get("OBJ_CENTER_MAX_X", "1.55")),
        float(os.environ.get("OBJ_CENTER_MIN_Y", "0.25")),
        float(os.environ.get("OBJ_CENTER_MAX_Y", "0.95")),
        float(os.environ.get("OBJ_CENTER_MIN_Z", "0.70")),
        float(os.environ.get("OBJ_CENTER_MAX_Z", "1.15")),
    )
    reasons = []
    if center_uv[0] < edge_margin or center_uv[0] > (w - edge_margin) or center_uv[1] < edge_margin or center_uv[1] > (h - edge_margin):
        reasons.append(f"uv_near_edge={np.round(center_uv, 2)}")
    if not np.isfinite(center_w).all():
        reasons.append("center_nonfinite")
    else:
        xmin, xmax, ymin, ymax, zmin, zmax = workspace
        if not (xmin <= center_w[0] <= xmax and ymin <= center_w[1] <= ymax and zmin <= center_w[2] <= zmax):
            reasons.append(f"center_out_of_workspace={np.round(center_w, 4)}")
    return (len(reasons) == 0), "; ".join(reasons)


def goto_object_centered_observation_pose(env,
                                          object_center_w: np.ndarray,
                                          cam_name: str = EE_CAM_NAME,
                                          hover_height: float = 0.3,
                                          duration: float = 1.2,
                                          settle_steps: int = 80,
                                          debug: bool = True):
    """Move the EE-camera to a top-down observation pose above the segmented object center.

    Note:
      - This project uses MuJoCo world Z-up. So "directly above the object" means
        aligning camera x/y with the object center and placing camera z above it by
        `hover_height`.
      - If you truly want the camera to stay at the same z as the object center, set
        hover_height=0.0, but that is usually not a valid "top" observation pose.
    """
    object_center_w = np.asarray(object_center_w, dtype=float).reshape(3,)
    robot = env.robot

    cam_pos_w = object_center_w.copy()
    cam_pos_w[2] = float(object_center_w[2] + hover_height)
    if np.linalg.norm(cam_pos_w - object_center_w) < 1e-6:
        cam_pos_w[2] += 0.15

    preserve_roll = os.environ.get("NERO_OBS_PRESERVE_CAMERA_ROLL", "1").strip().lower() not in {"0", "false", "no", "off"}
    if preserve_roll:
        try:
            T_wc_now = get_T_wc_from_camera(env, cam_name)
            x_ref_w = np.asarray(T_wc_now.R, dtype=float)[:, 0]
        except Exception:
            x_ref_w = np.array([1.0, 0.0, 0.0], dtype=float)
    else:
        x_ref_w = np.array([1.0, 0.0, 0.0], dtype=float)
    T_wc_des = _look_at_T_wc(cam_pos_w, object_center_w, x_ref_w=x_ref_w)

    if debug:
        print(f"[ObjCenter] object_center_w={np.round(object_center_w, 4)}")
        print(f"[ObjCenter] desired_cam_pos_w={np.round(cam_pos_w, 4)} hover_height={hover_height:.3f}")
    else:
        print(f"[ObjCenter] desired observation distance={float(hover_height):.3f}m")

    try:
        result = _move_camera_pose_closed_loop(
            env,
            cam_name=cam_name,
            T_wc_des=T_wc_des,
            duration=duration,
            settle_steps=max(int(settle_steps), 30),
            pos_tol=0.006,
            ang_tol_deg=2.5,
            max_iters=int(os.environ.get("OBJ_CENTER_MAX_ITERS", "1")),
            debug=debug,
        )
        T_wc_actual = result["T_wc_actual"]
        if (
            getattr(robot, "dof", 0) == 7
            and float(result.get("pos_err", 0.0)) > float(os.environ.get("OBJ_CENTER_FALLBACK_POS_ERR", "0.080"))
            and _env_flag("OBJ_CENTER_USE_JOINT_SCAN_FALLBACK", "1")
        ):
            raise AssertionError(f"object-center camera pose error too large: {float(result.get('pos_err', 0.0)):.4f}m")
    except Exception as e:
        if getattr(robot, "dof", 0) != 7 or not _env_flag("OBJ_CENTER_USE_JOINT_SCAN_FALLBACK", "1"):
            raise
        scan = _find_feasible_object_view_joint_scan(
            env,
            cam_name=cam_name,
            target_w=object_center_w,
            desired_dir_xy=None,
            allow_far_fallback=_env_flag("OBJ_CENTER_ALLOW_FAR_JOINT_SCAN_FALLBACK", "0"),
        )
        if scan is None:
            print(f"[ObjCenter] joint-scan fallback failed after camera IK error: {type(e).__name__}: {e}")
            raise
        print(
            f"[ObjCenter] camera IK fallback to joint-scan: "
            f"checked={scan.get('checked', 0)} center_norm={scan['center_norm']:.3f} "
            f"dist={scan['distance']:.3f}m height={scan['height']:.3f}m "
            f"near={not bool(scan.get('near_fallback', False))}"
        )
        move_joint_smooth(
            env,
            np.asarray(scan["q"], dtype=float).reshape(-1),
            duration=float(os.environ.get("OBJ_CENTER_JOINT_SCAN_DURATION", "0.45")),
            settle_steps=int(os.environ.get("OBJ_CENTER_JOINT_SCAN_SETTLE_STEPS", "8")),
        )
        T_wc_actual = get_T_wc_from_camera(env, cam_name)
    actual_cam_pos_w = np.asarray(T_wc_actual.t, dtype=float).reshape(3,)
    actual_radius = float(np.linalg.norm(actual_cam_pos_w - object_center_w))
    print(
        f"[ObjCenter] actual_cam_pos_w={np.round(actual_cam_pos_w, 4)} "
        f"center_radius={actual_radius:.4f}m"
    )
    return T_wc_actual


def refine_object_centered_observation_pose(env,
                                            cam_name: str = EE_CAM_NAME,
                                            hover_height: float = 0.22,
                                            duration: float = 0.8,
                                            settle_steps: int = 60,
                                            xy_tol: float = 0.01,
                                            z_tol: float = 0.01,
                                            debug: bool = True,
                                            target_class=None):
    """Closed-loop refinement after the coarse 'move above object' stage.

    Procedure:
      1) Capture once at the current 'top' pose.
      2) Re-segment the object and re-estimate the object center in world coordinates.
      3) If the current camera position is not centered above the refined object center
         (XY) or hover height is off (Z), perform one more Cartesian correction.

    Returns:
        dict with refined object center and the current/final desired camera pose.
    """
    model, _ = _get_mj_model_data(env)
    fovy_cam = get_cam_fovy_rad(model, cam_name)

    imgs = render_from_camera(env, cam_name, depth_mode='auto', debug=False)
    rgb = imgs['img']
    depth = imgs['depth']
    T_wc_now = get_T_wc_from_camera(env, cam_name)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    try:
        mask = segment_image(
            bgr,
            output_mask='mask_refine.png',
            target_class=target_class,
            interactive=False,
            allow_manual_click=False,
            conf_thresh=0.10,
        )
    except Exception as e:
        if debug:
            print(f"[ObjCenter][refine] skip refinement: YOLO26-seg did not find target from refined view ({type(e).__name__}: {e})")
        return {
            'object_center_w': None,
            'center_uv': None,
            'T_wc_before_refine': T_wc_now,
            'T_wc_des': T_wc_now,
            'desired_cam_pos_w': None,
            'xy_err': float("inf"),
            'z_err': float("inf"),
            'moved': False,
            'skipped': True,
            'reason': str(e),
        }

    obj_info = estimate_object_center_from_mask(
        bgr=bgr,
        depth_m=depth,
        mask=mask,
        T_wc=T_wc_now,
        fovy_rad=fovy_cam,
    )
    object_center_w = np.asarray(obj_info['center_w'], dtype=float).reshape(3,)
    cam_pos_now = np.asarray(T_wc_now.t, dtype=float).reshape(3,)

    desired_cam_pos_w = object_center_w.copy()
    desired_cam_pos_w[2] = float(object_center_w[2] + hover_height)

    xy_err = float(np.linalg.norm(cam_pos_now[:2] - desired_cam_pos_w[:2]))
    z_err = float(abs(cam_pos_now[2] - desired_cam_pos_w[2]))

    if debug:
        print(f"[ObjCenter][refine] uv={np.round(obj_info['center_uv'], 2)} center_w={np.round(object_center_w, 4)}")
        print(f"[ObjCenter][refine] cam_pos_now={np.round(cam_pos_now, 4)} desired_cam_pos_w={np.round(desired_cam_pos_w, 4)} "
              f"xy_err={xy_err:.4f} z_err={z_err:.4f}")

    moved = False
    T_wc_des = T_wc_now
    if (xy_err > float(xy_tol)) or (z_err > float(z_tol)):
        T_wc_des = goto_object_centered_observation_pose(
            env,
            object_center_w=object_center_w,
            cam_name=cam_name,
            hover_height=hover_height,
            duration=duration,
            settle_steps=settle_steps,
            debug=debug,
        )
        moved = True

    return {
        'object_center_w': object_center_w.astype(np.float32),
        'center_uv': np.asarray(obj_info['center_uv'], dtype=np.float32),
        'T_wc_before_refine': T_wc_now,
        'T_wc_des': T_wc_des,
        'desired_cam_pos_w': desired_cam_pos_w.astype(np.float32),
        'xy_err': xy_err,
        'z_err': z_err,
        'moved': moved,
    }

def cloud_from_rgbd_no_mask(color_bgr: np.ndarray, depth_m: np.ndarray, fovy_rad: float,
                           max_depth: float = 2.0):
    """Create an *unmasked* point cloud in the CV camera frame from RGB-D (MuJoCo rendered).
    Returns:
        pts_c (N,3), cols (N,3 in [0,1])
    """
    if color_bgr.dtype != np.float32:
        color = color_bgr.astype(np.float32) / 255.0
    else:
        color = np.clip(color_bgr, 0.0, 1.0).astype(np.float32)

    depth = depth_m.astype(np.float32)
    if depth.ndim == 3:
        depth = depth[..., 0]

    h, w = color.shape[:2]
    if depth.shape != (h, w):
        depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_NEAREST)

    K = _build_cam_intrinsics_from_fovy(w, h, fovy_rad)
    camera = CameraInfo(w, h, K[0, 0], K[1, 1], K[0, 2], K[1, 2], 1.0)

    cloud = create_point_cloud_from_depth_image(depth, camera, organized=True)  # (H,W,3)
    valid = np.isfinite(depth) & (depth > 0.0) & (depth < float(max_depth))
    pts_c = cloud[valid].astype(np.float32)
    cols = color[valid].astype(np.float32)
    return pts_c, cols


def _voxel_downsample_xyzrgb(pts: np.ndarray, cols: np.ndarray = None, voxel_size: float = 0.0):
    pts = np.asarray(pts, dtype=np.float32).reshape(-1, 3)
    if pts.shape[0] == 0 or voxel_size is None or voxel_size <= 0:
        return pts, None if cols is None else np.asarray(cols, dtype=np.float32).reshape(-1, 3)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float32))
    if cols is not None:
        cols = np.asarray(cols, dtype=np.float32).reshape(-1, 3)
        if cols.shape[0] == pts.shape[0]:
            pcd.colors = o3d.utility.Vector3dVector(cols.astype(np.float32))
    pcd = pcd.voxel_down_sample(voxel_size=float(voxel_size))
    pts_ds = np.asarray(pcd.points, dtype=np.float32)
    cols_ds = np.asarray(pcd.colors, dtype=np.float32) if pcd.has_colors() else None
    return pts_ds, cols_ds


def _points_inside_instance_obb(points_w: np.ndarray, instance, margin: float = 0.02):
    pts_w = np.asarray(points_w, dtype=np.float32).reshape(-1, 3)
    if pts_w.shape[0] == 0 or instance is None:
        return np.zeros((pts_w.shape[0],), dtype=bool)
    center = np.asarray(instance.get("bbox_center", np.zeros((3,), dtype=np.float32)), dtype=np.float32).reshape(3,)
    R = np.asarray(instance.get("bbox_R", np.eye(3, dtype=np.float32)), dtype=np.float32).reshape(3, 3)
    extent = np.asarray(instance.get("bbox_extent", np.full((3,), 0.02, dtype=np.float32)), dtype=np.float32).reshape(3,)
    rel = (pts_w - center[None, :]) @ R
    half = 0.5 * extent + float(margin)
    return np.all(np.abs(rel) <= half[None, :], axis=1)


def _extract_side_target_point_sets(color_bgr: np.ndarray,
                                    depth_m: np.ndarray,
                                    T_wc: sm.SE3,
                                    fovy_rad: float,
                                    target_instance,
                                    target_class: str = None,
                                    bbox_margin: float = 0.03,
                                    max_dist_to_centroid: float = 0.20,
                                    debug_mask_path: str = None):
    color = np.asarray(color_bgr, dtype=np.uint8)
    depth = np.asarray(depth_m, dtype=np.float32)
    if depth.ndim == 3:
        depth = depth[..., 0]
    h, w = depth.shape[:2]
    K = _build_cam_intrinsics_from_fovy(w, h, fovy_rad)
    camera = CameraInfo(w, h, K[0, 0], K[1, 1], K[0, 2], K[1, 2], 1.0)
    cloud_org = create_point_cloud_from_depth_image(depth.astype(np.float32), camera, organized=True)
    valid = np.isfinite(depth) & (depth > 0.03) & (depth < 2.0)
    pts_c_valid = cloud_org[valid].astype(np.float32)
    pts_w_valid = _cam_to_world_points(T_wc, pts_c_valid)
    cols_valid = (color.astype(np.float32) / 255.0)[valid].astype(np.float32)

    if pts_w_valid.shape[0] == 0:
        mask = np.zeros((h, w), dtype=np.uint8)
        if debug_mask_path is not None:
            cv2.imwrite(debug_mask_path, mask)
        return {
            "mask": mask,
            "pts_c": np.zeros((0, 3), dtype=np.float32),
            "pts_w": np.zeros((0, 3), dtype=np.float32),
            "cols": np.zeros((0, 3), dtype=np.float32),
        }

    if target_instance is not None:
        center = np.asarray(target_instance.get("bbox_center", np.median(pts_w_valid, axis=0)), dtype=np.float32).reshape(3,)
        obb_keep = _points_inside_instance_obb(pts_w_valid, target_instance, margin=float(bbox_margin))
        proj_box = _project_instance_bbox_xyxy(target_instance, T_wc, camera, image_shape=(h, w))
    else:
        center = np.median(pts_w_valid, axis=0).astype(np.float32)
        obb_keep = np.ones((pts_w_valid.shape[0],), dtype=bool)
        proj_box = None

    keep_ctr = np.linalg.norm(pts_w_valid - center[None, :], axis=1) <= float(max_dist_to_centroid)

    box_keep = np.ones((pts_w_valid.shape[0],), dtype=bool)
    if proj_box is not None:
        pad = 10.0
        x0 = max(int(np.floor(proj_box[0] - pad)), 0)
        y0 = max(int(np.floor(proj_box[1] - pad)), 0)
        x1 = min(int(np.ceil(proj_box[2] + pad)), w - 1)
        y1 = min(int(np.ceil(proj_box[3] + pad)), h - 1)
        ys, xs = np.nonzero(valid)
        box_keep = (xs >= x0) & (xs <= x1) & (ys >= y0) & (ys <= y1)

    keep = obb_keep & keep_ctr & box_keep
    seg_mask = None
    if target_class is not None and _env_flag("MV_SIDE_USE_YOLO_MASK", "0"):
        try:
            seg_mask = segment_image(
                color,
                output_mask='mask_side_tmp.png',
                target_class=str(target_class),
                interactive=False,
                allow_manual_click=False,
                conf_thresh=0.15,
            )
        except Exception:
            seg_mask = None
    if seg_mask is not None:
        seg_mask = (np.asarray(seg_mask) > 0).astype(np.uint8)
        ys_valid, xs_valid = np.nonzero(valid)
        seg_keep = seg_mask[ys_valid, xs_valid] > 0
        seg_combined = seg_keep & keep_ctr
        if target_instance is not None:
            seg_combined = seg_combined & obb_keep
        if int(np.count_nonzero(seg_combined)) >= 60:
            keep = seg_combined

    candidate_idx = np.flatnonzero(keep)
    if target_instance is not None and candidate_idx.size >= 80:
        pts_candidate = pts_w_valid[candidate_idx].astype(np.float32)
        pts_refined = extract_instance_points_from_candidates(
            target_instance,
            pts_candidate,
            bbox_margin=min(float(bbox_margin), 0.022),
            max_seed_radius=min(float(max_dist_to_centroid), 0.14),
            cluster_eps=0.012,
            min_cluster_points=20,
        )
        if pts_refined.shape[0] >= 40:
            tree = cKDTree(pts_candidate.astype(np.float64))
            dists, nn_idx = tree.query(pts_refined.astype(np.float64), k=1)
            nn_idx = np.asarray(nn_idx[np.asarray(dists) < 1e-5], dtype=np.int64)
            if nn_idx.size >= 40:
                refined_keep = np.zeros((candidate_idx.shape[0],), dtype=bool)
                refined_keep[np.unique(nn_idx)] = True
                keep = np.zeros_like(keep)
                keep[candidate_idx[refined_keep]] = True
                candidate_idx = np.flatnonzero(keep)

    candidate_mask = np.zeros((h, w), dtype=np.uint8)
    ys, xs = np.nonzero(valid)
    candidate_mask[ys[keep], xs[keep]] = 255
    if debug_mask_path is not None:
        cv2.imwrite(debug_mask_path, candidate_mask)

    return {
        "mask": candidate_mask,
        "pts_c": pts_c_valid[keep].astype(np.float32),
        "pts_w": pts_w_valid[keep].astype(np.float32),
        "cols": cols_valid[keep].astype(np.float32),
    }


def mask_to_world_points(mask, depth_m, T_wc, fovy_rad, depth_min=0.03, depth_max=2.0):
    if mask.ndim == 3:
        mask = mask[..., 0]
    mask = (mask > 0).astype(np.uint8)

    h, w = mask.shape[:2]
    K = _build_cam_intrinsics_from_fovy(w, h, fovy_rad)
    camera = CameraInfo(w, h, K[0, 0], K[1, 1], K[0, 2], K[1, 2], 1.0)

    cloud = create_point_cloud_from_depth_image(depth_m.astype(np.float32), camera, organized=True)
    valid = (mask > 0) & np.isfinite(depth_m) & (depth_m > depth_min) & (depth_m < depth_max)

    pts_c = cloud[valid].astype(np.float32)
    pts_w = _cam_to_world_points(T_wc, pts_c)
    return pts_w


def _fallback_mask_from_center_hint(depth_m: np.ndarray,
                                    T_wc: sm.SE3,
                                    fovy_rad: float,
                                    target_center_w: np.ndarray,
                                    xy_radius: float = None,
                                    z_radius: float = None,
                                    min_pixels: int = 80):
    """Build a target mask from a known 3D center when YOLO loses the object."""
    if target_center_w is None:
        return None
    depth = np.asarray(depth_m, dtype=np.float32)
    if depth.ndim == 3:
        depth = depth[..., 0]
    h, w = depth.shape[:2]
    target_center_w = np.asarray(target_center_w, dtype=np.float32).reshape(3,)
    xy_radius = float(os.environ.get("CENTER_FALLBACK_MASK_XY_RADIUS", "0.070")) if xy_radius is None else float(xy_radius)
    z_radius = float(os.environ.get("CENTER_FALLBACK_MASK_Z_RADIUS", "0.050")) if z_radius is None else float(z_radius)
    max_pixels = int(os.environ.get("CENTER_FALLBACK_MASK_MAX_PIXELS", "18000"))
    image_radius = float(os.environ.get("CENTER_FALLBACK_MASK_IMAGE_RADIUS", "115"))

    K = _build_cam_intrinsics_from_fovy(w, h, fovy_rad)
    camera = CameraInfo(w, h, K[0, 0], K[1, 1], K[0, 2], K[1, 2], 1.0)
    cloud = create_point_cloud_from_depth_image(depth.astype(np.float32), camera, organized=True)
    valid = np.isfinite(depth) & (depth > 0.03) & (depth < 2.0)
    if not np.any(valid):
        return None

    pts_c = cloud[valid].astype(np.float32)
    pts_w = _cam_to_world_points(T_wc, pts_c)
    ys, xs = np.nonzero(valid)
    uv_center, uv_valid = _project_world_points_to_image(target_center_w[None, :], T_wc, camera)
    image_keep = np.ones((pts_w.shape[0],), dtype=bool)
    if np.any(uv_valid):
        uv0 = uv_center[0]
        pix_dist = np.linalg.norm(
            np.stack([xs.astype(np.float32), ys.astype(np.float32)], axis=1) - uv0[None, :],
            axis=1,
        )
        image_keep = pix_dist <= image_radius

    rel = pts_w - target_center_w[None, :]
    keep = (
        (np.linalg.norm(rel[:, :2], axis=1) <= xy_radius)
        & (rel[:, 2] >= -0.020)
        & (rel[:, 2] <= z_radius)
        & image_keep
    )
    if int(np.count_nonzero(keep)) < int(min_pixels):
        keep = (
            (np.linalg.norm(rel[:, :2], axis=1) <= xy_radius * 1.25)
            & (rel[:, 2] >= -0.030)
            & (rel[:, 2] <= z_radius * 1.15)
            & image_keep
        )
    if int(np.count_nonzero(keep)) < int(min_pixels):
        return None

    mask = np.zeros((h, w), dtype=np.uint8)
    mask[ys[keep], xs[keep]] = 255

    if np.any(uv_valid):
        num_cc, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if num_cc > 1:
            uv0 = uv_center[0].astype(np.float32)
            best_label = 0
            best_score = float("inf")
            for lab in range(1, num_cc):
                area = int(stats[lab, cv2.CC_STAT_AREA])
                if area < int(min_pixels):
                    continue
                c = np.asarray(centroids[lab], dtype=np.float32)
                score = float(np.linalg.norm(c - uv0) - 0.002 * min(area, max_pixels))
                if score < best_score:
                    best_score = score
                    best_label = lab
            if best_label > 0:
                mask = (labels == best_label).astype(np.uint8) * 255

    pix = int(np.count_nonzero(mask))
    if pix > max_pixels:
        ys_m, xs_m = np.nonzero(mask)
        if np.any(uv_valid):
            uv0 = uv_center[0].astype(np.float32)
            d = np.linalg.norm(
                np.stack([xs_m.astype(np.float32), ys_m.astype(np.float32)], axis=1) - uv0[None, :],
                axis=1,
            )
        else:
            d = np.arange(xs_m.shape[0], dtype=np.float32)
        keep_idx = np.argsort(d)[:max_pixels]
        trimmed = np.zeros_like(mask)
        trimmed[ys_m[keep_idx], xs_m[keep_idx]] = 255
        mask = trimmed

    k = int(os.environ.get("CENTER_FALLBACK_MASK_CLOSE_K", "5"))
    if k > 1:
        kernel = np.ones((k, k), dtype=np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    if int(np.count_nonzero(mask)) > max_pixels:
        ys_m, xs_m = np.nonzero(mask)
        if np.any(uv_valid):
            uv0 = uv_center[0].astype(np.float32)
            d = np.linalg.norm(
                np.stack([xs_m.astype(np.float32), ys_m.astype(np.float32)], axis=1) - uv0[None, :],
                axis=1,
            )
        else:
            d = np.arange(xs_m.shape[0], dtype=np.float32)
        keep_idx = np.argsort(d)[:max_pixels]
        trimmed = np.zeros_like(mask)
        trimmed[ys_m[keep_idx], xs_m[keep_idx]] = 255
        mask = trimmed
    return mask


scene_instances = []
_SCENE_INSTANCE_ID = 0
_CAM_SERVO_CONFIG_PRINTED = False


def _next_scene_instance_id():
    global _SCENE_INSTANCE_ID
    _SCENE_INSTANCE_ID += 1
    return int(_SCENE_INSTANCE_ID)


def reset_scene_instances():
    global scene_instances, _SCENE_INSTANCE_ID
    scene_instances = []
    _SCENE_INSTANCE_ID = 0


def _query_click_depth(depth_m: np.ndarray, u: int, v: int, search_radius: int = 6):
    h, w = depth_m.shape[:2]
    u = int(np.clip(u, 0, w - 1))
    v = int(np.clip(v, 0, h - 1))
    d0 = float(depth_m[v, u])
    if np.isfinite(d0) and d0 > 0.03:
        return d0, u, v
    for r in range(1, int(max(search_radius, 1)) + 1):
        u0 = max(0, u - r)
        u1 = min(w, u + r + 1)
        v0 = max(0, v - r)
        v1 = min(h, v + r + 1)
        patch = depth_m[v0:v1, u0:u1]
        valid = np.isfinite(patch) & (patch > 0.03)
        if np.any(valid):
            idx = np.argwhere(valid)[0]
            vv = int(v0 + idx[0])
            uu = int(u0 + idx[1])
            return float(depth_m[vv, uu]), uu, vv
    raise RuntimeError(f"[click-init] invalid depth around click ({u}, {v}).")


def _camera_point_from_uvd(u, v, depth_m, camera):
    fx = float(camera.fx)
    fy = float(camera.fy)
    cx = float(camera.cx)
    cy = float(camera.cy)
    x = (float(u) - cx) * float(depth_m) / max(fx, 1e-6)
    y = (float(v) - cy) * float(depth_m) / max(fy, 1e-6)
    return np.array([x, y, float(depth_m)], dtype=np.float32)


def _region_grow_depth_mask(depth_m: np.ndarray,
                            seed_u: int,
                            seed_v: int,
                            seed_depth: float,
                            max_search_radius: int = 120,
                            abs_depth_thresh: float = 0.012,
                            rel_depth_thresh: float = 0.06,
                            local_smooth_thresh: float = 0.008,
                            min_pixels: int = 40,
                            max_pixels: int = 25000):
    h, w = depth_m.shape[:2]
    seed_u = int(np.clip(seed_u, 0, w - 1))
    seed_v = int(np.clip(seed_v, 0, h - 1))
    seed_depth = float(seed_depth)
    grow_thresh = max(float(abs_depth_thresh), float(rel_depth_thresh) * max(seed_depth, 1e-3))
    smooth_thresh = max(float(local_smooth_thresh), 0.5 * grow_thresh)

    mask = np.zeros((h, w), dtype=np.uint8)
    visited = np.zeros((h, w), dtype=np.uint8)
    q = deque([(seed_u, seed_v)])
    visited[seed_v, seed_u] = 1
    accepted = 0

    while q and accepted < int(max_pixels):
        u, v = q.popleft()
        d = float(depth_m[v, u])
        if (not np.isfinite(d)) or d <= 0.03:
            continue
        if abs(d - seed_depth) > grow_thresh:
            continue
        if max(abs(u - seed_u), abs(v - seed_v)) > int(max_search_radius):
            continue

        mask[v, u] = 1
        accepted += 1
        for dv in (-1, 0, 1):
            for du in (-1, 0, 1):
                if du == 0 and dv == 0:
                    continue
                uu = u + du
                vv = v + dv
                if uu < 0 or uu >= w or vv < 0 or vv >= h or visited[vv, uu]:
                    continue
                visited[vv, uu] = 1
                dn = float(depth_m[vv, uu])
                if (not np.isfinite(dn)) or dn <= 0.03:
                    continue
                if abs(dn - seed_depth) <= grow_thresh and abs(dn - d) <= smooth_thresh:
                    q.append((uu, vv))

    if accepted < int(min_pixels):
        return np.zeros((h, w), dtype=np.uint8)

    num_cc, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_cc <= 1:
        return mask
    seed_label = int(labels[seed_v, seed_u])
    if seed_label > 0:
        return (labels == seed_label).astype(np.uint8)
    best_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return (labels == best_label).astype(np.uint8)


def _bbox_corners_world(instance):
    center = np.asarray(instance.get("bbox_center", np.zeros((3,), dtype=np.float32)), dtype=np.float32).reshape(3,)
    R = np.asarray(instance.get("bbox_R", np.eye(3, dtype=np.float32)), dtype=np.float32).reshape(3, 3)
    extent = np.asarray(instance.get("bbox_extent", np.full((3,), 0.02, dtype=np.float32)), dtype=np.float32).reshape(3,)
    half = 0.5 * extent
    signs = np.array([
        [-1, -1, -1], [-1, -1, 1], [-1, 1, -1], [-1, 1, 1],
        [1, -1, -1], [1, -1, 1], [1, 1, -1], [1, 1, 1],
    ], dtype=np.float32)
    return center[None, :] + (signs * half[None, :]) @ R.T


def _project_world_points_to_image(pts_w, T_wc, camera):
    pts_w = np.asarray(pts_w, dtype=np.float32).reshape(-1, 3)
    if pts_w.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0,), dtype=bool)
    pts_c = _world_to_cam_points(T_wc, pts_w)
    z = pts_c[:, 2]
    valid = z > 1e-5
    uv = np.zeros((pts_c.shape[0], 2), dtype=np.float32)
    uv[:, 0] = float(camera.fx) * (pts_c[:, 0] / np.maximum(z, 1e-6)) + float(camera.cx)
    uv[:, 1] = float(camera.fy) * (pts_c[:, 1] / np.maximum(z, 1e-6)) + float(camera.cy)
    return uv, valid


def _project_instance_bbox_xyxy(instance, T_wc, camera, image_shape=None):
    corners_w = _bbox_corners_world(instance)
    uv, valid = _project_world_points_to_image(corners_w, T_wc, camera)
    if not np.any(valid):
        return None
    uv = uv[valid]
    x0, y0 = np.min(uv, axis=0)
    x1, y1 = np.max(uv, axis=0)
    if image_shape is not None:
        h, w = image_shape[:2]
        x0 = float(np.clip(x0, 0, w - 1))
        x1 = float(np.clip(x1, 0, w - 1))
        y0 = float(np.clip(y0, 0, h - 1))
        y1 = float(np.clip(y1, 0, h - 1))
    if x1 <= x0 or y1 <= y0:
        return None
    return np.array([x0, y0, x1, y1], dtype=np.float32)


def _bbox_overlap_ratio_xyxy(box_a, box_b):
    if box_a is None or box_b is None:
        return 0.0
    ax0, ay0, ax1, ay1 = [float(v) for v in box_a]
    bx0, by0, bx1, by1 = [float(v) for v in box_b]
    ix0 = max(ax0, bx0)
    iy0 = max(ay0, by0)
    ix1 = min(ax1, bx1)
    iy1 = min(ay1, by1)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    area_a = max((ax1 - ax0) * (ay1 - ay0), 1e-6)
    return float(inter / area_a)


def _bbox_area_ratio_xyxy(box, image_shape):
    if box is None or image_shape is None:
        return 0.0
    h, w = image_shape[:2]
    x0, y0, x1, y1 = [float(v) for v in box]
    area = max(x1 - x0, 0.0) * max(y1 - y0, 0.0)
    return float(area / max(float(h * w), 1.0))


def _compute_projected_bbox_view_metrics(target_instance,
                                         scene_instances,
                                         T_wc_view,
                                         camera,
                                         image_shape,
                                         T_wc_current=None):
    if target_instance is None or camera is None or image_shape is None:
        return {
            "target_bbox_area": 0.0,
            "target_bbox_gain": 0.5,
            "bbox_overlap_penalty": 0.0,
            "bbox_separation": 1.0,
            "bbox_visibility": 0.0,
        }
    target_xyxy = _project_instance_bbox_xyxy(target_instance, T_wc_view, camera, image_shape=image_shape)
    if target_xyxy is None:
        return {
            "target_bbox_area": 0.0,
            "target_bbox_gain": 0.0,
            "bbox_overlap_penalty": 1.0,
            "bbox_separation": 0.0,
            "bbox_visibility": 0.0,
        }
    target_area = _bbox_area_ratio_xyxy(target_xyxy, image_shape)
    current_area = target_area
    if T_wc_current is not None:
        current_xyxy = _project_instance_bbox_xyxy(target_instance, T_wc_current, camera, image_shape=image_shape)
        if current_xyxy is not None:
            current_area = _bbox_area_ratio_xyxy(current_xyxy, image_shape)
    area_gain = float(np.clip((target_area - current_area) / max(current_area, 1e-6), -1.0, 1.0))
    area_gain_01 = float(np.clip(0.5 + 0.5 * area_gain, 0.0, 1.0))
    bbox_visibility = float(np.clip(target_area / max(current_area, 1e-6), 0.0, 1.0))
    cam_pos = np.asarray(T_wc_view.t, dtype=np.float32).reshape(3,)
    target_center = np.asarray(target_instance.get("bbox_center", np.zeros((3,), dtype=np.float32)), dtype=np.float32).reshape(3,)
    target_depth = float(np.linalg.norm(target_center - cam_pos))
    occ_penalties = []
    for inst in scene_instances or []:
        if int(inst.get("id", -1)) == int(target_instance.get("id", -999)):
            continue
        inst_xyxy = _project_instance_bbox_xyxy(inst, T_wc_view, camera, image_shape=image_shape)
        overlap = _bbox_overlap_ratio_xyxy(target_xyxy, inst_xyxy)
        if overlap <= 1e-6:
            continue
        inst_center = np.asarray(inst.get("bbox_center", np.zeros((3,), dtype=np.float32)), dtype=np.float32).reshape(3,)
        inst_depth = float(np.linalg.norm(inst_center - cam_pos))
        depth_factor = float(np.clip((target_depth - inst_depth) / max(target_depth, 1e-6), 0.0, 1.0))
        role_weight = 1.0 if str(inst.get("role", "unknown")) == "obstacle" else 0.8
        occ_penalties.append(float(overlap * max(depth_factor, 0.35) * role_weight))
    overlap_pen = float(np.clip(max(occ_penalties) if len(occ_penalties) > 0 else 0.0, 0.0, 1.0))
    return {
        "target_bbox_area": float(target_area),
        "target_bbox_gain": float(area_gain_01),
        "bbox_overlap_penalty": float(overlap_pen),
        "bbox_separation": float(1.0 - overlap_pen),
        "bbox_visibility": float(bbox_visibility),
    }


def _mask_xyxy(mask):
    if mask is None:
        return None
    m = np.asarray(mask)
    if m.ndim == 3:
        m = m[..., 0]
    ys, xs = np.nonzero(m > 0)
    if xs.size == 0:
        return None
    return np.array([float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())], dtype=np.float32)


def _merge_context_detections(base_dets, extra_dets, iou_thresh: float = 0.60):
    merged = [dict(x) for x in (base_dets or [])]
    for det in extra_dets or []:
        det = dict(det)
        label = str(det.get("label", "unknown"))
        mask = det.get("mask", None)
        duplicate = False
        for old in merged:
            if not _same_label(old.get("label", ""), label):
                continue
            old_mask = old.get("mask", None)
            if mask is not None and old_mask is not None and _mask_iou(mask, old_mask) >= float(iou_thresh):
                duplicate = True
                break
        if not duplicate:
            merged.append(det)
    return merged


def _draw_xyxy(img, box, color, thickness=2, label=None):
    if box is None:
        return
    x0, y0, x1, y1 = [int(round(v)) for v in box]
    cv2.rectangle(img, (x0, y0), (x1, y1), color, int(thickness))
    if label:
        cv2.putText(img, str(label), (x0, max(18, y0 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)


def save_debug_image(image_bgr, output_path, title=None):
    if image_bgr is None:
        return
    canvas = np.asarray(image_bgr, dtype=np.uint8).copy()
    if title:
        cv2.putText(canvas, str(title), (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 20), 3, cv2.LINE_AA)
        cv2.putText(canvas, str(title), (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (240, 240, 240), 1, cv2.LINE_AA)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, canvas)


def save_yolo_detection_debug_overlay(image_bgr,
                                      output_path,
                                      target_class=None,
                                      conf_thresh=0.15,
                                      title=None):
    if image_bgr is None:
        return
    try:
        _, vis_img = detect_objects(
            image_or_path=image_bgr,
            target_class=target_class,
            conf_thresh=conf_thresh,
        )
        canvas = np.asarray(vis_img, dtype=np.uint8) if vis_img is not None else np.asarray(image_bgr, dtype=np.uint8).copy()
    except Exception as e:
        canvas = np.asarray(image_bgr, dtype=np.uint8).copy()
        cv2.putText(canvas, f"YOLO debug failed: {type(e).__name__}", (12, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
    if title:
        cv2.putText(canvas, str(title), (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 20), 3, cv2.LINE_AA)
        cv2.putText(canvas, str(title), (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (240, 240, 240), 1, cv2.LINE_AA)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, canvas)


def save_scene_instance_debug_overlay(image_bgr,
                                      instances,
                                      T_wc,
                                      camera,
                                      output_path,
                                      title=None,
                                      image_shape=None):
    if image_bgr is None:
        return
    canvas = np.asarray(image_bgr, dtype=np.uint8).copy()
    img_shape = canvas.shape[:2] if image_shape is None else image_shape
    colors = {
        "target": (40, 60, 230),
        "obstacle": (30, 200, 240),
        "unknown": (180, 220, 60),
    }
    for inst in instances:
        role = str(inst.get("role", "unknown"))
        color = colors.get(role, colors["unknown"])
        label = str(inst.get("label", role))
        mask = inst.get("mask", None)
        if mask is not None:
            m = np.asarray(mask, dtype=np.uint8)
            if m.ndim == 3:
                m = m[..., 0]
            m = (m > 0).astype(np.uint8)
            contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(canvas, contours, -1, color, 1)
            _draw_xyxy(canvas, _mask_xyxy(m), color, thickness=1, label=f"{label}:mask")
        proj_box = None if (T_wc is None or camera is None) else _project_instance_bbox_xyxy(inst, T_wc, camera, image_shape=img_shape)
        _draw_xyxy(canvas, proj_box, color, thickness=2, label=f"{label}:{role}")
    if title:
        cv2.putText(canvas, str(title), (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 20), 3, cv2.LINE_AA)
        cv2.putText(canvas, str(title), (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (240, 240, 240), 1, cv2.LINE_AA)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, canvas)


def _debug_iter_path(debug_dir, iter_tag, stem, ext=".png"):
    os.makedirs(debug_dir, exist_ok=True)
    return os.path.join(debug_dir, f"{iter_tag}_{stem}{ext}")


def _instance_from_points(points_w: np.ndarray,
                          seed_w: np.ndarray,
                          role: str = "unknown",
                          instance_id: int = None,
                          full_points_w: np.ndarray = None):
    pts = np.asarray(points_w, dtype=np.float32).reshape(-1, 3)
    if pts.shape[0] == 0:
        raise RuntimeError("[instance] empty point set.")
    mu = pts.mean(axis=0, keepdims=True)
    X = pts - mu
    if pts.shape[0] >= 3:
        _, _, Vt = np.linalg.svd(X.astype(np.float64), full_matrices=False)
        R = Vt.T.astype(np.float32)
    else:
        R = np.eye(3, dtype=np.float32)
    proj = X @ R
    pmin = proj.min(axis=0)
    pmax = proj.max(axis=0)
    extent = np.maximum((pmax - pmin).astype(np.float32), 1e-3)
    center_local = 0.5 * (pmin + pmax)
    bbox_center = (mu.reshape(3,) + R @ center_local.astype(np.float32)).astype(np.float32)

    visible_ratio = 1.0
    if full_points_w is not None and np.asarray(full_points_w).reshape(-1, 3).shape[0] > 0:
        full_pts = np.asarray(full_points_w, dtype=np.float32).reshape(-1, 3)
        rel = (full_pts - bbox_center[None, :]) @ R
        inside = np.all(np.abs(rel) <= (0.5 * extent[None, :] + 0.01), axis=1)
        denom = int(np.count_nonzero(inside))
        if denom > 0:
            visible_ratio = float(np.clip(pts.shape[0] / float(denom), 0.0, 1.0))

    return {
        "id": int(_next_scene_instance_id() if instance_id is None else instance_id),
        "seed": np.asarray(seed_w, dtype=np.float32).reshape(3,),
        "points": pts.astype(np.float32),
        "bbox_center": bbox_center.astype(np.float32),
        "bbox_R": R.astype(np.float32),
        "bbox_extent": extent.astype(np.float32),
        "visible_ratio": float(visible_ratio),
        "role": str(role),
    }


def create_instance_from_click(rgb, depth, click_uv, camera, T_wc, role="unknown"):
    if depth.ndim == 3:
        depth = depth[..., 0]
    depth = depth.astype(np.float32)
    u, v = int(click_uv[0]), int(click_uv[1])
    depth_seed, uu, vv = _query_click_depth(depth, u, v)
    seed_c = _camera_point_from_uvd(uu, vv, depth_seed, camera)
    seed_w = _cam_to_world_points(T_wc, seed_c[None, :])[0].astype(np.float32)
    cloud = create_point_cloud_from_depth_image(depth.astype(np.float32), camera, organized=True)
    valid = np.isfinite(depth) & (depth > 0.03) & (depth < 2.0)
    pts_c = cloud[valid].astype(np.float32)
    pts_w = _cam_to_world_points(T_wc, pts_c)
    if pts_w.shape[0] == 0:
        raise RuntimeError("[click-init] no valid points in depth image.")

    click_mask = _region_grow_depth_mask(depth, uu, vv, depth_seed)
    inst_pts = np.zeros((0, 3), dtype=np.float32)
    if np.count_nonzero(click_mask) > 0:
        inst_pts = _cam_to_world_points(T_wc, cloud[click_mask > 0].astype(np.float32))

    if inst_pts.shape[0] < 40:
        local_radius = max(0.04, 0.05 * float(depth_seed))
        d = np.linalg.norm(pts_w - seed_w[None, :], axis=1)
        local_pts = pts_w[d <= float(local_radius)]
        if local_pts.shape[0] < 20:
            local_pts = pts_w[d <= float(local_radius * 1.6)]
        if local_pts.shape[0] == 0:
            raise RuntimeError("[click-init] local neighborhood around click is empty.")
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(local_pts.astype(np.float64))
        labels = np.asarray(pcd.cluster_dbscan(eps=max(0.010, 0.016 * float(depth_seed)), min_points=20, print_progress=False), dtype=np.int32)
        if labels.size == 0 or np.all(labels < 0):
            inst_pts = local_pts
        else:
            best_label, best_dist = -1, float("inf")
            for lb in np.unique(labels[labels >= 0]):
                cluster_pts = local_pts[labels == lb]
                dist_seed = float(np.linalg.norm(cluster_pts.mean(axis=0) - seed_w))
                if dist_seed < best_dist:
                    best_label, best_dist = int(lb), dist_seed
            inst_pts = local_pts[labels == best_label] if best_label >= 0 else local_pts

    inst = _instance_from_points(inst_pts, seed_w=seed_w, role=role, full_points_w=pts_w)
    inst["mask"] = click_mask.astype(np.uint8) if np.count_nonzero(click_mask) > 0 else None
    inst["click_uv"] = np.array([uu, vv], dtype=np.int32)
    inst["xyxy"] = _mask_xyxy(inst["mask"]) if inst["mask"] is not None else None
    inst["label"] = role
    return inst


def _recognized_instance_at_click(click_uv, recognized_instances):
    if recognized_instances is None:
        return None
    u, v = int(click_uv[0]), int(click_uv[1])
    candidates = []
    for inst in recognized_instances:
        role = str(inst.get("role", "unknown")).lower()
        if role == "target":
            continue
        hit = False
        mask = inst.get("mask", None)
        if mask is not None:
            mask_arr = np.asarray(mask)
            if 0 <= v < mask_arr.shape[0] and 0 <= u < mask_arr.shape[1] and int(mask_arr[v, u]) > 0:
                hit = True
        xyxy = inst.get("xyxy", None)
        if not hit and xyxy is not None:
            x1, y1, x2, y2 = np.asarray(xyxy, dtype=float).reshape(4,)
            hit = bool((x1 <= u <= x2) and (y1 <= v <= y2))
        if hit:
            pts = np.asarray(inst.get("points", inst.get("points_w", np.zeros((0, 3), dtype=np.float32))), dtype=np.float32).reshape(-1, 3)
            candidates.append((pts.shape[0], inst))
    if len(candidates) == 0:
        return None
    candidates.sort(key=lambda x: x[0], reverse=True)
    return dict(candidates[0][1])


def _clicked_instance_from_recognized(recognized_inst, click_uv, role: str):
    inst = dict(recognized_inst)
    pts = np.asarray(inst.get("points", inst.get("points_w", np.zeros((0, 3), dtype=np.float32))), dtype=np.float32).reshape(-1, 3)
    if pts.shape[0] > 0:
        inst["points"] = pts.astype(np.float32)
        inst["points_w"] = pts.astype(np.float32)
    inst["role"] = str(role)
    inst["click_uv"] = np.asarray(click_uv, dtype=np.int32).reshape(2,)
    inst["is_user_clicked"] = True
    inst["selected_from_recognition"] = True
    inst["label"] = str(inst.get("label", role))
    return inst


def extract_instance_points_from_candidates(instance,
                                            candidate_points_w,
                                            bbox_margin: float = 0.018,
                                            max_seed_radius: float = 0.14,
                                            cluster_eps: float = 0.014,
                                            min_cluster_points: int = 25):
    pts = np.asarray(candidate_points_w, dtype=np.float32).reshape(-1, 3)
    if pts.shape[0] == 0:
        return pts
    center = np.asarray(instance.get("bbox_center", np.median(pts, axis=0)), dtype=np.float32).reshape(3,)
    R = np.asarray(instance.get("bbox_R", np.eye(3, dtype=np.float32)), dtype=np.float32).reshape(3, 3)
    extent = np.asarray(instance.get("bbox_extent", np.full((3,), 0.04, dtype=np.float32)), dtype=np.float32).reshape(3,)
    seed = np.asarray(instance.get("seed", center), dtype=np.float32).reshape(3,)
    rel_local = (pts - center[None, :]) @ R
    inside_bbox = np.all(np.abs(rel_local) <= (0.5 * extent[None, :] + float(bbox_margin)), axis=1)
    near_seed = np.linalg.norm(pts - seed[None, :], axis=1) <= float(max_seed_radius)
    keep = inside_bbox & near_seed
    if np.count_nonzero(keep) < max(int(min_cluster_points), 20):
        keep = near_seed
    pts_keep = pts[keep]
    if pts_keep.shape[0] == 0:
        return pts_keep
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts_keep.astype(np.float64))
    labels = np.asarray(pcd.cluster_dbscan(eps=float(max(cluster_eps, 1e-3)), min_points=int(max(min_cluster_points, 5)), print_progress=False), dtype=np.int32)
    if labels.size == 0 or np.all(labels < 0):
        return pts_keep
    seed_idx = int(np.argmin(np.linalg.norm(pts_keep - seed[None, :], axis=1)))
    seed_label = int(labels[seed_idx])
    if seed_label >= 0:
        return pts_keep[labels == seed_label]
    best_label, best_dist = -1, float("inf")
    for lb in np.unique(labels[labels >= 0]):
        cluster_pts = pts_keep[labels == lb]
        dist_seed = float(np.linalg.norm(cluster_pts.mean(axis=0) - seed))
        if dist_seed < best_dist:
            best_label, best_dist = int(lb), dist_seed
    return pts_keep[labels == best_label] if best_label >= 0 else pts_keep


def _extract_instance_points_adaptive(instance,
                                      candidate_points_w,
                                      strict_params=None,
                                      loose_params=None,
                                      min_keep_ratio: float = 0.20,
                                      min_keep_points: int = 300,
                                      debug_label: str = None):
    pts_raw = np.asarray(candidate_points_w, dtype=np.float32).reshape(-1, 3)
    if pts_raw.shape[0] == 0:
        return pts_raw

    strict_defaults = dict(bbox_margin=0.024, max_seed_radius=0.16, cluster_eps=0.015, min_cluster_points=20)
    loose_defaults = dict(bbox_margin=0.045, max_seed_radius=0.24, cluster_eps=0.020, min_cluster_points=12)
    if strict_params is not None:
        strict_defaults.update(strict_params)
    if loose_params is not None:
        loose_defaults.update(loose_params)

    pts_strict = extract_instance_points_from_candidates(instance, pts_raw, **strict_defaults)
    keep_thr = max(int(min_keep_points), int(np.ceil(float(min_keep_ratio) * pts_raw.shape[0])))
    if pts_strict.shape[0] >= keep_thr:
        return pts_strict

    pts_loose = extract_instance_points_from_candidates(instance, pts_raw, **loose_defaults)
    if pts_loose.shape[0] > pts_strict.shape[0]:
        if debug_label is not None:
            print(
                f"[{debug_label}] relaxed instance extraction "
                f"{pts_strict.shape[0]} -> {pts_loose.shape[0]} from raw={pts_raw.shape[0]}"
            )
        return pts_loose
    return pts_strict


def update_instance_with_new_view(instance, new_points, voxel_size: float = 0.0020):
    pts_old = np.asarray(instance.get("points", np.zeros((0, 3), dtype=np.float32)), dtype=np.float32).reshape(-1, 3)
    pts_new_raw = np.asarray(new_points, dtype=np.float32).reshape(-1, 3)
    pts_new = _extract_instance_points_adaptive(
        instance,
        pts_new_raw,
        strict_params=dict(bbox_margin=0.028, max_seed_radius=0.18, cluster_eps=0.016, min_cluster_points=16),
        loose_params=dict(bbox_margin=0.055, max_seed_radius=0.28, cluster_eps=0.022, min_cluster_points=8),
        min_keep_ratio=0.12,
        min_keep_points=180,
        debug_label="InstanceUpdate",
    )
    if pts_new.shape[0] == 0:
        return instance
    pts = np.concatenate([pts_old, pts_new], axis=0) if pts_old.shape[0] > 0 else pts_new
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
    if voxel_size is not None and voxel_size > 0:
        pcd = pcd.voxel_down_sample(voxel_size=float(voxel_size))
    merged = np.asarray(pcd.points, dtype=np.float32)
    updated = _instance_from_points(merged, seed_w=np.asarray(instance.get("seed", merged.mean(axis=0)), dtype=np.float32), role=str(instance.get("role", "unknown")), instance_id=int(instance.get("id", _next_scene_instance_id())), full_points_w=merged)
    old_vis = float(instance.get("visible_ratio", 0.0))
    old_extent = np.asarray(instance.get("bbox_extent", np.zeros((3,), dtype=np.float32)), dtype=np.float32).reshape(3,)
    new_extent = np.asarray(updated.get("bbox_extent", np.zeros((3,), dtype=np.float32)), dtype=np.float32).reshape(3,)
    point_gain = float(np.clip((merged.shape[0] - max(pts_old.shape[0], 1)) / max(float(max(pts_old.shape[0], 1)), 1.0), 0.0, 1.0))
    span_gain = float(np.clip((np.linalg.norm(new_extent) - np.linalg.norm(old_extent)) / max(np.linalg.norm(old_extent), 1e-6), 0.0, 1.0))
    updated["visible_ratio"] = float(np.clip(max(old_vis, old_vis + (1.0 - old_vis) * (0.45 * point_gain + 0.55 * span_gain)), 0.0, 1.0))
    for key in ("label", "mask", "xyxy", "center"):
        if key in instance and key not in updated:
            updated[key] = instance[key]
    return updated


def _replace_scene_instance(instances, updated_instance):
    out, replaced = [], False
    updated_id = int(updated_instance.get("id", -1))
    for inst in instances:
        if int(inst.get("id", -999)) == updated_id:
            out.append(updated_instance)
            replaced = True
        else:
            out.append(inst)
    if not replaced:
        out.append(updated_instance)
    return out


def assess_instance_observation(instance, min_visible_ratio: float = 0.68, min_points: int = 1200):
    pts = np.asarray(instance.get("points", np.zeros((0, 3), dtype=np.float32)), dtype=np.float32).reshape(-1, 3)
    visible_ratio = float(instance.get("visible_ratio", 0.0))
    extent = np.asarray(instance.get("bbox_extent", np.zeros((3,), dtype=np.float32)), dtype=np.float32).reshape(3,)
    span = float(np.linalg.norm(extent))
    return {
        "visible_ratio": visible_ratio,
        "num_points": int(pts.shape[0]),
        "bbox_extent": extent,
        "bbox_span": span,
        "well_observed": bool((visible_ratio >= float(min_visible_ratio)) and (pts.shape[0] >= int(min_points))),
    }


def compute_occluders(target, scene_instances, camera_pos, T_wc=None, camera=None, image_shape=None, min_projected_overlap: float = 0.08):
    target_center = np.asarray(target["bbox_center"], dtype=np.float32).reshape(3,)
    cam = np.asarray(camera_pos, dtype=np.float32).reshape(3,)
    rt = target_center - cam
    dt = float(np.linalg.norm(rt))
    if dt < 1e-6:
        return []
    rt_dir = rt / dt
    target_radius = 0.5 * float(np.linalg.norm(np.asarray(target["bbox_extent"], dtype=np.float32)))
    target_xyxy = None if (T_wc is None or camera is None) else _project_instance_bbox_xyxy(target, T_wc, camera, image_shape=image_shape)
    out = []
    for inst in scene_instances:
        if int(inst.get("id", -1)) == int(target.get("id", -999)):
            continue
        ci = np.asarray(inst["bbox_center"], dtype=np.float32).reshape(3,)
        ri = ci - cam
        di = float(np.linalg.norm(ri))
        if di < 1e-6 or di >= dt or float(np.dot(ri, rt)) <= 0.0:
            continue
        ri_dir = ri / di
        occ_radius = 0.5 * float(np.linalg.norm(np.asarray(inst["bbox_extent"], dtype=np.float32)))
        ang = float(np.arccos(np.clip(np.dot(ri_dir, rt_dir), -1.0, 1.0)))
        overlap_ang = np.arctan2(target_radius + occ_radius, max(dt, 1e-6))
        if ang > overlap_ang:
            continue
        inst_copy = dict(inst)
        proj_overlap = 0.0
        if target_xyxy is not None and T_wc is not None and camera is not None:
            occ_xyxy = _project_instance_bbox_xyxy(inst, T_wc, camera, image_shape=image_shape)
            proj_overlap = _bbox_overlap_ratio_xyxy(target_xyxy, occ_xyxy)
            if proj_overlap < float(min_projected_overlap):
                continue
            inst_copy["projected_xyxy"] = occ_xyxy
        inst_copy["projected_overlap"] = float(proj_overlap)
        inst_copy["target_depth_gap"] = float(dt - di)
        out.append(inst_copy)
    out.sort(key=lambda x: (-float(x.get("projected_overlap", 0.0)), x["target_depth_gap"], -float(x.get("visible_ratio", 0.0))))
    return out


def _dir_xy_from_world_vector(vec_w):
    v = np.asarray(vec_w, dtype=np.float32).reshape(3,)
    d = v[:2]
    n = float(np.linalg.norm(d))
    if n < 1e-8:
        return np.array([1.0, 0.0], dtype=np.float32)
    return (d / n).astype(np.float32)


def generate_occlusion_aware_nbv(target, occluders, camera_pos=None):
    center = np.asarray(target["bbox_center"], dtype=np.float32).reshape(3,)
    R = np.asarray(target["bbox_R"], dtype=np.float32).reshape(3, 3)
    major_xy = _dir_xy_from_world_vector(R[:, 0])
    minor_xy = _dir_xy_from_world_vector(R[:, 1])
    candidates, seen = [], []

    def _add(dir_xy, reason):
        d = _normalize_dir_xy(dir_xy)
        key = tuple(np.round(d, 3))
        if key in seen:
            return
        seen.append(key)
        candidates.append({"dir_xy": d.astype(np.float32), "reason": str(reason)})

    if len(occluders) == 0:
        _add(major_xy, "default_major")
        _add(-major_xy, "default_major_opposite")
        _add(minor_xy, "default_minor")
        return candidates

    for occ in occluders:
        rel = np.asarray(occ["bbox_center"], dtype=np.float32).reshape(3,) - center
        rel_local = R.T @ rel
        if rel[2] > 0.03 and abs(rel[2]) >= max(abs(rel[0]), abs(rel[1])):
            _add(minor_xy, "occluder_above_side")
            _add(-minor_xy, "occluder_above_side_opposite")
        elif abs(rel_local[1]) >= abs(rel_local[0]):
            sign = 1.0 if rel_local[1] >= 0.0 else -1.0
            _add(-sign * minor_xy, "occluder_lateral_opposite")
            _add(sign * major_xy, "occluder_lateral_secondary")
        else:
            occ_dir = _dir_xy_from_world_vector(rel)
            side = np.array([-occ_dir[1], occ_dir[0]], dtype=np.float32)
            _add(side, "occluder_front_side")
            _add(-side, "occluder_front_side_opposite")
    return candidates[:3]


def score_occlusion_aware_views(candidate_views, target, occluders, camera_pos):
    cam = np.asarray(camera_pos, dtype=np.float32).reshape(3,)
    target_center = np.asarray(target["bbox_center"], dtype=np.float32).reshape(3,)
    cur_dir = _dir_xy_from_world_vector(target_center - cam)
    scored = []
    for cand in candidate_views:
        d = _normalize_dir_xy(cand["dir_xy"])
        sep = 0.0
        for occ in occluders:
            occ_dir = _dir_xy_from_world_vector(np.asarray(occ["bbox_center"], dtype=np.float32) - target_center)
            sep += float(np.clip(0.5 * (1.0 - np.dot(d, occ_dir)), 0.0, 1.0))
        sep = sep / max(len(occluders), 1)
        visible_gain = float(np.clip(0.4 + 0.6 * sep, 0.0, 1.0))
        bbox_gain = float(np.clip(0.3 + 0.7 * sep, 0.0, 1.0))
        motion_penalty = float(np.clip(0.5 * (1.0 - np.dot(d, cur_dir)), 0.0, 1.0))
        item = dict(cand)
        item.update({
            "target_visible_area": visible_gain,
            "target_bbox_gain": bbox_gain,
            "occluder_separation": sep,
            "large_motion_penalty": motion_penalty,
            "score": float(0.38 * visible_gain + 0.28 * bbox_gain + 0.24 * sep - 0.10 * motion_penalty),
        })
        scored.append(item)
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored


def select_occlusion_aware_view_dirs(target, scene_instances, camera_pos, top_k: int = 2, debug: bool = False, T_wc=None, camera=None, image_shape=None):
    occluders = compute_occluders(target, scene_instances, camera_pos, T_wc=T_wc, camera=camera, image_shape=image_shape)
    if len(occluders) == 0:
        return None, occluders, []
    cand_views = generate_occlusion_aware_nbv(target, occluders, camera_pos=camera_pos)
    scored_views = score_occlusion_aware_views(cand_views, target, occluders, camera_pos)
    if debug and len(scored_views) > 0:
        print("[ClickNBV] top candidates:")
        for item in scored_views[: min(3, len(scored_views))]:
            print(
                f"  dir={np.round(item['dir_xy'],4)} score={item['score']:.4f} "
                f"vis={item['target_visible_area']:.4f} bbox={item['target_bbox_gain']:.4f} "
                f"sep={item['occluder_separation']:.4f} motion_pen={item['large_motion_penalty']:.4f} reason={item['reason']}"
            )
    picked = [np.asarray(item["dir_xy"], dtype=np.float32) for item in scored_views[: max(int(top_k), 1)]]
    if len(picked) == 1:
        picked.append(-picked[0])
    return picked[:2], occluders, scored_views


def build_o3d_cloud_from_world_points(points_w, T_wc_center, color_rgb=(0.7, 0.7, 0.7), voxel_size: float = 0.0015):
    pts_w = np.asarray(points_w, dtype=np.float32).reshape(-1, 3)
    if pts_w.shape[0] == 0:
        raise RuntimeError("[instance-cloud] empty world points.")
    pts_c = _world_to_cam_points(T_wc_center, pts_w)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts_c.astype(np.float32))
    cols = np.tile(np.asarray(color_rgb, dtype=np.float32).reshape(1, 3), (pts_c.shape[0], 1))
    pcd.colors = o3d.utility.Vector3dVector(cols)
    if voxel_size is not None and voxel_size > 0:
        pcd = pcd.voxel_down_sample(voxel_size=float(voxel_size))
    try:
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    except Exception:
        pass
    return pcd


def build_scene_instances_from_view(target_mask, context_dets, depth0, T_wc0, fovy_rad):
    h, w = target_mask.shape[:2]
    K = _build_cam_intrinsics_from_fovy(w, h, fovy_rad)
    camera = CameraInfo(w, h, K[0, 0], K[1, 1], K[0, 2], K[1, 2], 1.0)
    full_pts_w = _cam_to_world_points(T_wc0, cloud_from_rgbd_no_mask(np.zeros((h, w, 3), dtype=np.uint8), depth0, fovy_rad)[0])
    instances = []
    target_pts_w = mask_to_world_points(target_mask, depth0, T_wc0, fovy_rad)
    if target_pts_w.shape[0] > 0:
        target_seed = np.median(target_pts_w, axis=0).astype(np.float32)
        target_inst = _instance_from_points(target_pts_w, seed_w=target_seed, role="target", full_points_w=full_pts_w)
        target_inst["label"] = "target"
        target_inst["mask"] = (target_mask > 0).astype(np.uint8)
        target_inst["xyxy"] = _mask_xyxy(target_mask)
        instances.append(target_inst)
    for det in context_dets:
        mask = det.get("mask", None)
        if mask is None:
            if VERBOSE_LOG:
                print(f"[SceneInstances][drop] label={det.get('label', 'unknown')} reason=no_mask")
            continue
        pts_w = mask_to_world_points(mask, depth0, T_wc0, fovy_rad)
        if pts_w.shape[0] == 0:
            if VERBOSE_LOG:
                print(
                    f"[SceneInstances][drop] label={det.get('label', 'unknown')} "
                    f"score={float(det.get('score', 0.0)):.3f} reason=no_depth_points "
                    f"mask_area={int(np.count_nonzero(mask))}"
                )
            continue
        seed = np.median(pts_w, axis=0).astype(np.float32)
        inst = _instance_from_points(pts_w, seed_w=seed, role="obstacle", full_points_w=full_pts_w)
        inst["label"] = str(det.get("label", "unknown"))
        inst["mask"] = mask.astype(np.uint8)
        inst["xyxy"] = _mask_xyxy(mask)
        instances.append(inst)
        if VERBOSE_LOG:
            print(
                f"[SceneInstances][keep] label={inst['label']} score={float(det.get('score', 0.0)):.3f} "
                f"mask_area={int(np.count_nonzero(mask))} points={int(pts_w.shape[0])} "
                f"center={np.round(seed, 4)}"
            )
    return instances, camera


def _collect_single_click_uv(image_bgr, window_name="Click Instance"):
    show_img = image_bgr.copy()
    clicked = {"pt": None}
    cv2.imshow(window_name, show_img)

    def _cb(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            clicked["pt"] = (int(x), int(y))

    cv2.setMouseCallback(window_name, _cb)
    while clicked["pt"] is None:
        key = cv2.waitKey(10)
        if key == 27:
            cv2.destroyWindow(window_name)
            return None
    cv2.destroyWindow(window_name)
    return clicked["pt"]


def maybe_collect_click_instances(center_bgr, depth0, camera, T_wc0, recognized_instances=None):
    created = []
    while True:
        role = input("[ClickInit] add instance role [target/obstacle/unknown/done]: ").strip().lower()
        if role in ("", "done", "finish", "no", "n"):
            break
        if role not in ("target", "obstacle", "unknown"):
            print("[ClickInit] invalid role.")
            continue
        click_uv = _collect_single_click_uv(center_bgr, window_name=f"Click {role}")
        if click_uv is None:
            break
        rec_inst = _recognized_instance_at_click(click_uv, recognized_instances)
        if rec_inst is not None:
            inst = _clicked_instance_from_recognized(rec_inst, click_uv, role=role)
            print(
                f"[ClickInit] matched recognized instance label={inst.get('label', 'unknown')} "
                f"id={int(inst.get('id', -1))}; use its 3D box/points for clicked {role}."
            )
        else:
            inst = create_instance_from_click(center_bgr, depth0, click_uv, camera, T_wc0, role=role)
            inst["is_user_clicked"] = True
        created.append(inst)
        print(f"[ClickInit] created instance id={inst['id']} role={inst['role']} center={np.round(inst['bbox_center'],4)} extent={np.round(inst['bbox_extent'],4)}")
    return created


def inject_scene_instances_into_context(scene_context, extra_instances, default_weight: float = 1.1):
    extra_instances = list(extra_instances)
    if "_base_context_points_w" not in scene_context:
        scene_context["_base_context_points_w"] = np.asarray(scene_context.get("context_points_w", np.zeros((0, 3), dtype=np.float32)), dtype=np.float32).reshape(-1, 3).copy()
        scene_context["_base_context_weights"] = np.asarray(scene_context.get("context_weights", np.zeros((0,), dtype=np.float32)), dtype=np.float32).reshape(-1,).copy()
        scene_context["_base_context_instance_clouds"] = list(scene_context.get("context_instance_clouds", []))
        scene_context["_base_context_instances"] = list(scene_context.get("context_instances", []))
        scene_context["_base_context_labels"] = list(scene_context.get("context_labels", []))
    target_instances = [inst for inst in extra_instances if str(inst.get("role", "unknown")) == "target"]
    if len(target_instances) > 0:
        target_inst = target_instances[-1]
        scene_context["target_center_w"] = np.asarray(target_inst.get("bbox_center", scene_context.get("target_center_w", np.zeros((3,), dtype=np.float32))), dtype=np.float32).reshape(3,)
        scene_context["target_points_w"] = np.asarray(target_inst.get("points", scene_context.get("target_points_w", np.zeros((0, 3), dtype=np.float32))), dtype=np.float32).reshape(-1, 3)
        scene_context["target_instance"] = target_inst

    ctx_pts = np.asarray(scene_context.get("_base_context_points_w", np.zeros((0, 3), dtype=np.float32)), dtype=np.float32).reshape(-1, 3).copy()
    ctx_wts = np.asarray(scene_context.get("_base_context_weights", np.zeros((0,), dtype=np.float32)), dtype=np.float32).reshape(-1,).copy()
    clouds = list(scene_context.get("_base_context_instance_clouds", []))
    infos = list(scene_context.get("_base_context_instances", []))
    labels = list(scene_context.get("_base_context_labels", []))
    seen_ids = set()
    for inst in extra_instances:
        inst_id = int(inst.get("id", -1))
        if inst_id in seen_ids:
            continue
        seen_ids.add(inst_id)
        if str(inst.get("role", "unknown")) == "target":
            continue
        pts = np.asarray(inst.get("points", np.zeros((0, 3), dtype=np.float32)), dtype=np.float32).reshape(-1, 3)
        if pts.shape[0] == 0:
            continue
        role_i = str(inst.get("role", "unknown")).lower()
        if bool(inst.get("is_user_clicked", False)) and role_i == "obstacle":
            w = float(os.environ.get("SELECTED_OBSTACLE_CONTEXT_WEIGHT", "2.2"))
        else:
            w = float(default_weight)
        ctx_pts = np.concatenate([ctx_pts, pts], axis=0) if ctx_pts.shape[0] > 0 else pts
        ctx_wts = np.concatenate([ctx_wts, np.full((pts.shape[0],), w, dtype=np.float32)], axis=0) if ctx_wts.shape[0] > 0 else np.full((pts.shape[0],), w, dtype=np.float32)
        lbl = f"click_{inst.get('role', 'unknown')}"
        labels.extend([lbl] * pts.shape[0])
        infos.append({"label": lbl, "score": 1.0, "num_points": int(pts.shape[0]), "weight": w})
        clouds.append({
            "label": lbl,
            "score": 1.0,
            "weight": w,
            "id": int(inst.get("id", -1)),
            "role": str(inst.get("role", "unknown")),
            "is_user_clicked": bool(inst.get("is_user_clicked", False)),
            "selected_from_recognition": bool(inst.get("selected_from_recognition", False)),
            "seed": np.asarray(inst.get("seed", np.median(pts, axis=0)), dtype=np.float32).reshape(3,),
            "bbox_center": np.asarray(inst.get("bbox_center", np.median(pts, axis=0)), dtype=np.float32).reshape(3,),
            "bbox_R": np.asarray(inst.get("bbox_R", np.eye(3, dtype=np.float32)), dtype=np.float32).reshape(3, 3),
            "bbox_extent": np.asarray(inst.get("bbox_extent", np.full((3,), 0.04, dtype=np.float32)), dtype=np.float32).reshape(3,),
            "points_w": pts.astype(np.float32),
            "mask": inst.get("mask", None),
            "xyxy": inst.get("xyxy", None),
            "click_uv": inst.get("click_uv", None),
            "center": None,
        })
    scene_context["context_points_w"] = ctx_pts.astype(np.float32)
    scene_context["context_weights"] = ctx_wts.astype(np.float32)
    scene_context["context_instance_clouds"] = clouds
    scene_context["context_instances"] = infos
    scene_context["context_labels"] = labels
    scene_context["obstacle_points_w"] = ctx_pts.astype(np.float32)
    scene_context["obstacle_weights"] = ctx_wts.astype(np.float32)
    scene_context["scene_instances"] = list(extra_instances)
    return scene_context


def build_scene_context_from_center_view(bgr0, depth0, target_mask, context_dets, T_wc0, fovy_rad,
                                         local_radius=0.35,
                                         debug=False,
                                         use_geometric_fallback=True):
    h_img, w_img = bgr0.shape[:2]
    K_ctx = _build_cam_intrinsics_from_fovy(w_img, h_img, fovy_rad)
    camera_ctx = CameraInfo(w_img, h_img, K_ctx[0, 0], K_ctx[1, 1], K_ctx[0, 2], K_ctx[1, 2], 1.0)
    target_pts_w = mask_to_world_points(target_mask, depth0, T_wc0, fovy_rad)
    if target_pts_w.shape[0] == 0:
        raise RuntimeError("[scene-context] target points empty.")

    center_w = np.median(target_pts_w, axis=0).astype(np.float32)

    ctx_pts, ctx_wts, ctx_lbls, inst_infos = [], [], [], []
    instance_clouds = []

    class_weight_map = {
        "wall": 1.6, "robot": 1.6, "arm": 1.6, "gripper": 1.6,
        "box": 1.2, "bottle": 1.1, "cup": 1.1, "mug": 1.1, "bowl": 1.0,
        "table": 0.15, "desk": 0.15, "unknown": 1.0,
    }

    for det in context_dets:
        label = str(det.get("label", "unknown")).lower()
        score = float(det.get("score", 1.0))
        mask = det.get("mask", None)
        if mask is None:
            if debug:
                print(f"[SceneContext][drop] label={label} reason=no_mask")
            continue

        pts_w = mask_to_world_points(mask, depth0, T_wc0, fovy_rad)
        if pts_w.shape[0] == 0:
            if debug:
                print(
                    f"[SceneContext][drop] label={label} score={score:.3f} reason=no_depth_points "
                    f"mask_area={int(np.count_nonzero(mask))}"
                )
            continue

        dist = np.linalg.norm(pts_w - center_w[None, :], axis=1)
        raw_pts = int(pts_w.shape[0])
        pts_w = pts_w[dist <= float(local_radius)]
        if pts_w.shape[0] == 0:
            if debug:
                print(
                    f"[SceneContext][drop] label={label} score={score:.3f} reason=outside_local_radius "
                    f"raw_points={raw_pts} local_radius={local_radius:.3f}"
                )
            continue

        w = float(class_weight_map.get(label, 1.0)) * score

        ctx_pts.append(pts_w)
        ctx_wts.append(np.full((pts_w.shape[0],), w, dtype=np.float32))
        ctx_lbls.extend([label] * pts_w.shape[0])
        inst_infos.append({
            "label": label,
            "score": score,
            "num_points": int(pts_w.shape[0]),
            "weight": w,
        })
        instance_clouds.append({
            "label": label,
            "score": score,
            "weight": w,
            "id": int(_next_scene_instance_id()),
            "role": "obstacle",
            "seed": np.median(pts_w, axis=0).astype(np.float32),
            "bbox_center": np.median(pts_w, axis=0).astype(np.float32),
            "bbox_R": np.eye(3, dtype=np.float32),
            "bbox_extent": np.maximum(np.max(pts_w, axis=0) - np.min(pts_w, axis=0), 1e-3).astype(np.float32),
            "points_w": pts_w.astype(np.float32),
            "mask": mask.astype(np.uint8),
            "xyxy": None if det.get("xyxy", None) is None else np.asarray(det.get("xyxy"), dtype=np.float32),
            "center": det.get("center", None),
        })
        if debug:
            print(
                f"[SceneContext][keep] label={label} score={score:.3f} "
                f"raw_points={raw_pts} kept_points={int(pts_w.shape[0])} "
                f"center={np.round(np.median(pts_w, axis=0), 4)}"
            )

    if len(ctx_pts) == 0 and use_geometric_fallback:
        target_mask_u8 = (target_mask > 0).astype(np.uint8)
        h, w = target_mask_u8.shape[:2]
        K = _build_cam_intrinsics_from_fovy(w, h, fovy_rad)
        camera = CameraInfo(w, h, K[0, 0], K[1, 1], K[0, 2], K[1, 2], 1.0)
        organized = create_point_cloud_from_depth_image(depth0.astype(np.float32), camera, organized=True)
        valid = np.isfinite(depth0) & (depth0 > 0.03) & (depth0 < 2.0) & (target_mask_u8 == 0)
        non_target_pts_c = organized[valid].astype(np.float32)
        non_target_pts_w = _cam_to_world_points(T_wc0, non_target_pts_c)
        if non_target_pts_w.shape[0] > 0:
            dist = np.linalg.norm(non_target_pts_w - center_w[None, :], axis=1)
            keep = dist <= float(local_radius)
            geo_pts = non_target_pts_w[keep]
            if geo_pts.shape[0] > 0:
                ctx_pts.append(geo_pts.astype(np.float32))
                ctx_wts.append(np.full((geo_pts.shape[0],), 0.6, dtype=np.float32))
                ctx_lbls.extend(["geometry"] * geo_pts.shape[0])
                inst_infos.append({
                    "label": "geometry_fallback",
                    "score": 1.0,
                    "num_points": int(geo_pts.shape[0]),
                    "weight": 0.6,
                })
                instance_clouds.append({
                    "label": "geometry_fallback",
                    "score": 1.0,
                    "weight": 0.6,
                    "id": -1,
                    "role": "unknown",
                    "seed": np.median(geo_pts, axis=0).astype(np.float32),
                    "bbox_center": np.median(geo_pts, axis=0).astype(np.float32),
                    "bbox_R": np.eye(3, dtype=np.float32),
                    "bbox_extent": np.maximum(np.max(geo_pts, axis=0) - np.min(geo_pts, axis=0), 1e-3).astype(np.float32),
                    "points_w": geo_pts.astype(np.float32),
                    "mask": None,
                    "xyxy": None,
                    "center": None,
                })

    if len(ctx_pts) > 0:
        ctx_pts = np.concatenate(ctx_pts, axis=0).astype(np.float32)
        ctx_wts = np.concatenate(ctx_wts, axis=0).astype(np.float32)
    else:
        ctx_pts = np.zeros((0, 3), dtype=np.float32)
        ctx_wts = np.zeros((0,), dtype=np.float32)

    if debug:
        print(f"[SceneContext][build] local_radius={local_radius:.3f} target_pts={target_pts_w.shape[0]} context_pts={ctx_pts.shape[0]}")
        if len(inst_infos) == 0:
            print("[SceneContext][build] no valid context instances survived 3D projection / filtering.")
        else:
            print(f"[SceneContext][build] instance infos = {inst_infos}")

    # Pre-build a height-filtered obstacle cloud for view selection / avoidance.
    if target_pts_w.shape[0] > 0 and ctx_pts.shape[0] > 0:
        target_z_lo = float(np.percentile(target_pts_w[:, 2], 10))
        target_z_hi = float(np.percentile(target_pts_w[:, 2], 90))
        obstacle_keep = ctx_pts[:, 2] >= (target_z_lo - 0.005)
        obstacle_pts = ctx_pts[obstacle_keep].astype(np.float32)
        obstacle_wts = ctx_wts[obstacle_keep].astype(np.float32)
    else:
        target_z_lo = float(center_w[2] - 0.03)
        target_z_hi = float(center_w[2] + 0.03)
        obstacle_pts = ctx_pts.astype(np.float32)
        obstacle_wts = ctx_wts.astype(np.float32)

    return {
        "target_center_w": center_w,
        "target_points_w": target_pts_w.astype(np.float32),
        "target_instance": None,
        "context_points_w": ctx_pts,
        "context_weights": ctx_wts,
        "context_labels": ctx_lbls,
        "context_instances": inst_infos,
        "context_instance_clouds": instance_clouds,
        "obstacle_points_w": obstacle_pts,
        "obstacle_weights": obstacle_wts,
        "target_z_lo": float(target_z_lo),
        "target_z_hi": float(target_z_hi),
        "image_shape": bgr0.shape[:2],
        "camera_info": camera_ctx,
        "T_wc": T_wc0,
    }

def direction_context_penalty(center_w, context_points_w, context_weights, dir_xy, sigma_deg=28.0, near_radius=0.30):
    if context_points_w.shape[0] == 0:
        return 0.0, 1.0

    rel = context_points_w[:, :2] - center_w[None, :2]
    rr = np.linalg.norm(rel, axis=1)
    keep = rr > 1e-6
    rel = rel[keep]
    rr = rr[keep]
    ww = context_weights[keep]
    if rel.shape[0] == 0:
        return 0.0, 1.0

    theta = np.arctan2(dir_xy[1], dir_xy[0])
    phi = np.arctan2(rel[:, 1], rel[:, 0])
    dang = (phi - theta + np.pi) % (2 * np.pi) - np.pi

    sigma = np.deg2rad(float(sigma_deg))
    ang_w = np.exp(-0.5 * (dang / max(sigma, 1e-6)) ** 2)
    dist_w = np.clip(1.0 - rr / float(near_radius), 0.0, 1.0)

    raw = ang_w * dist_w * ww
    occ = float(np.sum(raw) / max(np.sum(ww), 1e-6))
    occ = float(np.clip(occ, 0.0, 1.0))
    clearance = 1.0 - occ
    return occ, clearance


def _prepare_obstacle_cloud(scene_context,
                            min_height_margin: float = -0.005,
                            max_xy_radius: float = 0.40):
    """Prepare a height-aware obstacle cloud from real depth points.

    This is the point-cloud based complement to the semantic/YOLO context term.
    Low table-plane points are suppressed, while points overlapping the target's
    vertical extent receive larger weights.
    """
    center_w = np.asarray(scene_context.get("target_center_w", np.zeros(3, dtype=np.float32)), dtype=np.float32).reshape(3,)
    target_pts = np.asarray(scene_context.get("target_points_w", np.zeros((0, 3), dtype=np.float32)), dtype=np.float32).reshape(-1, 3)
    obstacle_pts = np.asarray(scene_context.get("obstacle_points_w", scene_context.get("context_points_w", np.zeros((0, 3), dtype=np.float32))), dtype=np.float32).reshape(-1, 3)
    obstacle_wts = np.asarray(scene_context.get("obstacle_weights", scene_context.get("context_weights", np.zeros((0,), dtype=np.float32))), dtype=np.float32).reshape(-1,)

    if obstacle_pts.shape[0] == 0:
        return obstacle_pts, obstacle_wts, {
            "target_z_lo": float(center_w[2] - 0.03),
            "target_z_hi": float(center_w[2] + 0.03),
            "height_keep_z": float(center_w[2] - 0.035),
        }

    if obstacle_wts.shape[0] != obstacle_pts.shape[0]:
        obstacle_wts = np.ones((obstacle_pts.shape[0],), dtype=np.float32)

    if target_pts.shape[0] > 0:
        target_z_lo = float(np.percentile(target_pts[:, 2], 10))
        target_z_hi = float(np.percentile(target_pts[:, 2], 90))
    else:
        target_z_lo = float(center_w[2] - 0.03)
        target_z_hi = float(center_w[2] + 0.03)

    xy_dist = np.linalg.norm(obstacle_pts[:, :2] - center_w[None, :2], axis=1)
    z_keep_thresh = float(target_z_lo + min_height_margin)
    keep = (xy_dist <= float(max_xy_radius)) & (obstacle_pts[:, 2] >= z_keep_thresh)

    pts = obstacle_pts[keep]
    wts = obstacle_wts[keep]
    if pts.shape[0] == 0:
        pts = obstacle_pts
        wts = obstacle_wts

    denom = max((target_z_hi - target_z_lo) + 0.02, 0.02)
    height_norm = np.clip((pts[:, 2] - z_keep_thresh) / denom, 0.0, 2.0)
    height_w = (1.0 + 0.75 * height_norm).astype(np.float32)
    wts = (wts.astype(np.float32) * height_w).astype(np.float32)

    return pts.astype(np.float32), wts.astype(np.float32), {
        "target_z_lo": float(target_z_lo),
        "target_z_hi": float(target_z_hi),
        "height_keep_z": float(z_keep_thresh),
    }


def _candidate_camera_pose_from_dir(center_w, T_wc_center, dir_xy, orbit_angle_rad):
    target_radius = float(os.environ.get("ACTIVE_NBV_VIEW_RADIUS", "0.18"))
    T_wc_cand = _make_circle_orbit_pose(
        object_center_w=np.asarray(center_w, dtype=np.float64).reshape(3,),
        T_wc_center=T_wc_center,
        orbit_horizontal_dir_xy=np.asarray(dir_xy, dtype=np.float64).reshape(2,),
        side_sign=1.0,
        orbit_angle_rad=float(orbit_angle_rad),
        min_radius=float(os.environ.get("ACTIVE_NBV_MIN_VIEW_RADIUS", "0.14")),
        target_radius=target_radius,
    )
    return T_wc_cand


def _make_camera_reachability_fn(env, cam_name: str):
    robot = getattr(env, "robot", None)
    if robot is None or not hasattr(robot, "ikine"):
        return None
    max_joint_jump = float(os.environ.get("ACTIVE_NBV_MAX_JOINT_JUMP", "0.85"))
    min_joint_margin = float(os.environ.get("ACTIVE_NBV_MIN_JOINT_MARGIN", "0.030"))
    q_saved_base = np.asarray(robot.get_joint(), dtype=float).reshape(-1)
    try:
        _, to_cmd, to_world = _build_frame_converters(env)
        T_wc_now = get_T_wc_from_camera(env, cam_name)
        T_we_now_world = to_world(robot.get_cartesian())
        T_ec_inv = (T_we_now_world.inv() * T_wc_now).inv()
    except Exception:
        return None

    def _check(T_wc_cand):
        state = _save_mujoco_robot_state(env)
        q_saved = state["robot_q"].reshape(-1)
        try:
            T_we_des_world = T_wc_cand * T_ec_inv
            T_we_cmd = to_cmd(T_we_des_world)
            q_try = _solve_ik_for_pose(env, T_we_cmd, q_seed=q_saved)
            if q_try.size != getattr(robot, "dof", q_saved.size) or not np.isfinite(q_try).all():
                return {"reachable": False, "reason": "ik_failed", "q_jump": float("inf"), "joint_margin": -float("inf"), "pose_err": float("inf"), "ang_err": float("inf"), "q": None}
            _sync_robot_joint_to_mujoco(env, q_try, update_ctrl=False)
            T_wc_pred = get_T_wc_from_camera(env, cam_name)
            pred_pos_err, pred_ang_err = _pose_error_metrics(T_wc_cand, T_wc_pred)
        except Exception as e:
            return {"reachable": False, "reason": type(e).__name__, "q_jump": float("inf"), "joint_margin": -float("inf"), "pose_err": float("inf"), "ang_err": float("inf"), "q": None}
        finally:
            _restore_mujoco_robot_state(env, state)

        q_lim_low = np.asarray(getattr(robot, "q_lim_low", np.full_like(q_try, -np.inf)), dtype=float).reshape(-1)
        q_lim_up = np.asarray(getattr(robot, "q_lim_up", np.full_like(q_try, np.inf)), dtype=float).reshape(-1)
        n = min(q_try.size, q_saved.size, q_lim_low.size, q_lim_up.size)
        q_jump = float(np.max(np.abs(q_try[:n] - q_saved[:n]))) if n > 0 else float("inf")
        joint_margin = float(np.min(np.minimum(q_try[:n] - q_lim_low[:n], q_lim_up[:n] - q_try[:n]))) if n > 0 else -float("inf")
        max_pose_err = float(os.environ.get("ACTIVE_NBV_MAX_PRED_POSE_ERR", "0.035"))
        max_ang_err = float(os.environ.get("ACTIVE_NBV_MAX_PRED_ANG_ERR", "8.0"))
        reachable = bool(q_jump <= max_joint_jump and joint_margin >= min_joint_margin and pred_pos_err <= max_pose_err and pred_ang_err <= max_ang_err)
        if reachable:
            reason = "ok"
        elif q_jump > max_joint_jump:
            reason = "large_joint_jump"
        elif joint_margin < min_joint_margin:
            reason = "near_joint_limit"
        elif pred_pos_err > max_pose_err:
            reason = "pose_error_too_large"
        else:
            reason = "angle_error_too_large"
        return {"reachable": reachable, "reason": reason, "q_jump": q_jump, "joint_margin": joint_margin, "pose_err": float(pred_pos_err), "ang_err": float(pred_ang_err), "q": q_try.copy()}

    return _check


def _line_of_sight_occlusion_penalty(cam_pos_w,
                                     target_center_w,
                                     obstacle_points_w,
                                     obstacle_weights,
                                     corridor_radius: float = 0.045,
                                     min_axial_margin: float = 0.015):
    pts = np.asarray(obstacle_points_w, dtype=np.float32).reshape(-1, 3)
    wts = np.asarray(obstacle_weights, dtype=np.float32).reshape(-1,)
    if pts.shape[0] == 0:
        return 0.0, 1.0, {"los_count": 0, "los_ratio": 0.0, "nearest": float("inf"), "weighted_count": 0.0}
    if wts.shape[0] != pts.shape[0]:
        wts = np.ones((pts.shape[0],), dtype=np.float32)

    cam = np.asarray(cam_pos_w, dtype=np.float32).reshape(3,)
    target = np.asarray(target_center_w, dtype=np.float32).reshape(3,)
    ray = target - cam
    dist = float(np.linalg.norm(ray))
    if dist < 1e-8:
        return 0.0, 1.0, {"los_count": 0, "los_ratio": 0.0, "nearest": float("inf"), "weighted_count": 0.0}
    ray = ray / dist

    rel = pts - cam[None, :]
    axial = rel @ ray
    perp = np.linalg.norm(rel - axial[:, None] * ray[None, :], axis=1)
    keep = (axial > float(min_axial_margin)) & (axial < float(max(dist - 0.01, min_axial_margin))) & (perp < float(corridor_radius))

    if not np.any(keep):
        return 0.0, 1.0, {"los_count": 0, "los_ratio": 0.0, "nearest": float("inf"), "weighted_count": 0.0}

    axial_k = axial[keep]
    wts_k = wts[keep]
    nearest = float(np.min(axial_k))
    weighted_count = float(np.sum(wts_k))
    ratio = float(np.mean(keep))

    near_norm = min(0.20, dist * 0.70)
    near_risk = float(np.clip((near_norm - nearest) / max(near_norm, 1e-6), 0.0, 1.0))
    density_risk = float(np.clip(weighted_count / 350.0, 0.0, 1.0))
    ratio_risk = float(np.clip(ratio / 0.05, 0.0, 1.0))
    occ = float(np.clip(0.45 * near_risk + 0.35 * density_risk + 0.20 * ratio_risk, 0.0, 1.0))
    return occ, (1.0 - occ), {
        "los_count": int(np.count_nonzero(keep)),
        "los_ratio": float(ratio),
        "nearest": float(nearest),
        "weighted_count": float(weighted_count),
    }


def _candidate_camera_clearance_penalty(cam_pos_w,
                                        obstacle_points_w,
                                        obstacle_weights=None,
                                        safe_clearance: float = 0.10,
                                        obstacle_tree=None):
    pts = np.asarray(obstacle_points_w, dtype=np.float32).reshape(-1, 3)
    if pts.shape[0] == 0:
        return 0.0, 1.0, {"nearest": float("inf")}

    tree = obstacle_tree if obstacle_tree is not None else cKDTree(pts)
    nearest, _ = tree.query(np.asarray(cam_pos_w, dtype=np.float32).reshape(1, 3), k=1, workers=-1)
    nearest = float(np.asarray(nearest).reshape(-1)[0])
    occ = float(np.clip((safe_clearance - nearest) / max(safe_clearance, 1e-6), 0.0, 1.0))
    return occ, (1.0 - occ), {"nearest": float(nearest)}


def _candidate_trajectory_penalty(cam_pos_from_w,
                                  cam_pos_to_w,
                                  obstacle_points_w,
                                  safe_clearance: float = 0.08,
                                  sample_num: int = 7,
                                  obstacle_tree=None):
    pts = np.asarray(obstacle_points_w, dtype=np.float32).reshape(-1, 3)
    if pts.shape[0] == 0:
        return 0.0, 1.0, {"nearest": float("inf")}

    p0 = np.asarray(cam_pos_from_w, dtype=np.float32).reshape(3,)
    p1 = np.asarray(cam_pos_to_w, dtype=np.float32).reshape(3,)
    ts = np.linspace(0.0, 1.0, int(max(sample_num, 2)), dtype=np.float32)
    path_pts = p0[None, :] * (1.0 - ts[:, None]) + p1[None, :] * ts[:, None]
    tree = obstacle_tree if obstacle_tree is not None else cKDTree(pts)
    d, _ = tree.query(path_pts, k=1, workers=-1)
    nearest = float(np.min(np.asarray(d, dtype=np.float32)))
    occ = float(np.clip((safe_clearance - nearest) / max(safe_clearance, 1e-6), 0.0, 1.0))
    return occ, (1.0 - occ), {"nearest": float(nearest)}



def _normalize_dir_xy(dir_xy):
    d = np.asarray(dir_xy, dtype=np.float32).reshape(2,)
    n = float(np.linalg.norm(d))
    if n < 1e-8:
        return np.array([1.0, 0.0], dtype=np.float32)
    return d / n


def _dir_from_angle_rad(ang):
    return np.array([np.cos(float(ang)), np.sin(float(ang))], dtype=np.float32)


def _angle_from_dir_xy(dir_xy):
    d = _normalize_dir_xy(dir_xy)
    return float(np.arctan2(float(d[1]), float(d[0])))


def _smallest_angle_diff_deg(a_rad, b_rad):
    d = (float(a_rad) - float(b_rad) + np.pi) % (2.0 * np.pi) - np.pi
    return abs(np.rad2deg(d))


def _rotate_dir_xy(dir_xy, angle_rad: float):
    d = _normalize_dir_xy(dir_xy)
    c = float(np.cos(angle_rad))
    s = float(np.sin(angle_rad))
    return np.array([c * d[0] - s * d[1], s * d[0] + c * d[1]], dtype=np.float32)


def _obstacle_direction_penalty(scene_context, center_w, dir_xy):
    if scene_context is None:
        return 0.0, {"label": "", "angle_deg": 180.0, "selected": False}
    center = np.asarray(center_w, dtype=np.float32).reshape(3,)
    d = _normalize_dir_xy(dir_xy)
    best_penalty = 0.0
    best_detail = {"label": "", "angle_deg": 180.0, "selected": False}
    candidates = list(scene_context.get("scene_instances", [])) + list(scene_context.get("context_instance_clouds", []))
    seen_ids = set()
    for inst in candidates:
        inst_id = int(inst.get("id", -999999))
        if inst_id in seen_ids:
            continue
        seen_ids.add(inst_id)
        role = str(inst.get("role", "unknown")).lower()
        if role == "target":
            continue
        bbox_center = inst.get("bbox_center", None)
        if bbox_center is None:
            pts = np.asarray(inst.get("points", inst.get("points_w", np.zeros((0, 3), dtype=np.float32))), dtype=np.float32).reshape(-1, 3)
            if pts.shape[0] == 0:
                continue
            bbox_center = np.median(pts, axis=0)
        rel = np.asarray(bbox_center, dtype=np.float32).reshape(3,) - center
        dist_xy = float(np.linalg.norm(rel[:2]))
        if dist_xy < 1e-4:
            continue
        obs_dir = (rel[:2] / dist_xy).astype(np.float32)
        align = float(np.clip(np.dot(d, obs_dir), -1.0, 1.0))
        angle_deg = float(np.rad2deg(np.arccos(align)))
        sigma = float(os.environ.get("ACTIVE_NBV_OBSTACLE_DIR_SIGMA_DEG", "35.0"))
        angular_pen = float(np.exp(-0.5 * (angle_deg / max(sigma, 1e-6)) ** 2))
        dist_pen = float(np.clip(1.0 - dist_xy / float(os.environ.get("ACTIVE_NBV_OBSTACLE_DIR_RADIUS", "0.35")), 0.0, 1.0))
        selected_boost = float(os.environ.get("ACTIVE_NBV_SELECTED_OBSTACLE_DIR_BOOST", "1.6")) if bool(inst.get("is_user_clicked", False)) else 1.0
        role_boost = 1.15 if role == "obstacle" else 1.0
        penalty = float(np.clip(angular_pen * max(dist_pen, 0.25) * selected_boost * role_boost, 0.0, 1.0))
        if penalty > best_penalty:
            best_penalty = penalty
            best_detail = {
                "label": str(inst.get("label", role)),
                "angle_deg": float(angle_deg),
                "selected": bool(inst.get("is_user_clicked", False)),
            }
    return float(best_penalty), best_detail


def score_single_view_scene_aware(normals_obj_w,
                                  uncertainty,
                                  grasp_heat,
                                  center_w,
                                  ctx_pts,
                                  ctx_wts,
                                  dir_xy,
                                  scene_context=None,
                                  T_wc_center=None,
                                  orbit_angle_rad=np.deg2rad(30.0),
                                  target_instance=None,
                                  scene_instances_local=None,
                                  camera=None,
                                  image_shape=None,
                                  preferred_dirs_xy=None,
                                  reachability_fn=None):
    d3 = np.array([float(dir_xy[0]), float(dir_xy[1]), 0.0], dtype=np.float32)
    v = _predict_visibility_from_viewdir(normals_obj_w, d3)
    surf = float(np.mean(v))
    unc_gain = float(np.mean(uncertainty * v))
    grasp_gain = float(np.mean(grasp_heat * uncertainty * v))

    occ_sem, clr_sem = direction_context_penalty(center_w, ctx_pts, ctx_wts, dir_xy)

    occ_los, clr_los = 0.0, 1.0
    occ_cam, clr_cam = 0.0, 1.0
    occ_traj, clr_traj = 0.0, 1.0
    los_detail = {"los_count": 0, "los_ratio": 0.0, "nearest": float("inf"), "weighted_count": 0.0}
    cam_detail = {"nearest": float("inf")}
    traj_detail = {"nearest": float("inf")}
    candidate_cam_pos_w = None
    obstacle_meta = None
    bbox_metrics = {
        "target_bbox_area": 0.0,
        "target_bbox_gain": 0.5,
        "bbox_overlap_penalty": 0.0,
        "bbox_separation": 1.0,
        "bbox_visibility": 0.0,
    }
    obstacle_dir_penalty = 0.0
    obstacle_dir_detail = {"label": "", "angle_deg": 180.0, "selected": False}
    preferred_dir_bonus = 0.0
    nbv_reachable = True
    nbv_reachability_reason = "not_checked"
    nbv_q_jump = 0.0
    nbv_joint_margin = float("inf")
    nbv_pose_err = 0.0
    nbv_ang_err = 0.0
    nbv_q = None

    if scene_context is not None and T_wc_center is not None:
        obstacle_dir_penalty, obstacle_dir_detail = _obstacle_direction_penalty(scene_context, center_w, dir_xy)
        obstacle_cache = scene_context.get("_active_nbv_obstacle_cache", None)
        if obstacle_cache is None:
            obstacle_pts, obstacle_wts, obstacle_meta = _prepare_obstacle_cloud(scene_context)
            obstacle_tree = cKDTree(obstacle_pts) if obstacle_pts.shape[0] > 0 else None
            scene_context["_active_nbv_obstacle_cache"] = {
                "pts": obstacle_pts,
                "wts": obstacle_wts,
                "meta": obstacle_meta,
                "tree": obstacle_tree,
            }
        else:
            obstacle_pts = obstacle_cache["pts"]
            obstacle_wts = obstacle_cache["wts"]
            obstacle_meta = obstacle_cache["meta"]
            obstacle_tree = obstacle_cache.get("tree", None)
        T_wc_cand = _candidate_camera_pose_from_dir(center_w, T_wc_center, dir_xy, orbit_angle_rad)
        cam_now = np.asarray(T_wc_center.t, dtype=np.float32).reshape(3,)
        candidate_cam_pos_w = np.asarray(T_wc_cand.t, dtype=np.float32).reshape(3,)
        if reachability_fn is not None:
            reach = reachability_fn(T_wc_cand)
            nbv_reachable = bool(reach.get("reachable", False))
            nbv_reachability_reason = str(reach.get("reason", "unknown"))
            nbv_q_jump = float(reach.get("q_jump", float("inf")))
            nbv_joint_margin = float(reach.get("joint_margin", -float("inf")))
            nbv_pose_err = float(reach.get("pose_err", float("inf")))
            nbv_ang_err = float(reach.get("ang_err", float("inf")))
            q_reach = reach.get("q", None)
            if q_reach is not None:
                q_reach = np.asarray(q_reach, dtype=np.float32).reshape(-1)
                if q_reach.size > 0 and np.isfinite(q_reach).all():
                    nbv_q = q_reach

        occ_los, clr_los, los_detail = _line_of_sight_occlusion_penalty(
            cam_pos_w=candidate_cam_pos_w,
            target_center_w=center_w,
            obstacle_points_w=obstacle_pts,
            obstacle_weights=obstacle_wts,
            corridor_radius=0.045,
            min_axial_margin=0.015,
        )
        occ_cam, clr_cam, cam_detail = _candidate_camera_clearance_penalty(
            cam_pos_w=candidate_cam_pos_w,
            obstacle_points_w=obstacle_pts,
            obstacle_weights=obstacle_wts,
            safe_clearance=float(os.environ.get("ACTIVE_NBV_CAMERA_SAFE_CLEARANCE", "0.14")),
            obstacle_tree=obstacle_tree,
        )
        occ_traj, clr_traj, traj_detail = _candidate_trajectory_penalty(
            cam_pos_from_w=cam_now,
            cam_pos_to_w=candidate_cam_pos_w,
            obstacle_points_w=obstacle_pts,
            safe_clearance=float(os.environ.get("ACTIVE_NBV_TRAJ_SAFE_CLEARANCE", "0.12")),
            sample_num=7,
            obstacle_tree=obstacle_tree,
        )
        bbox_metrics = _compute_projected_bbox_view_metrics(
            target_instance,
            scene_instances_local if scene_instances_local is not None else scene_context.get("scene_instances", []),
            T_wc_cand,
            camera,
            image_shape,
            T_wc_current=T_wc_center,
        )
        if preferred_dirs_xy is not None and len(preferred_dirs_xy) > 0:
            align_scores = []
            for pref in preferred_dirs_xy:
                pd = _normalize_dir_xy(pref)
                align_scores.append(float(np.clip(np.dot(pd, _normalize_dir_xy(dir_xy)), -1.0, 1.0)))
            if len(align_scores) > 0:
                preferred_dir_bonus = float(np.clip(0.5 + 0.5 * max(align_scores), 0.0, 1.0))

    occ_total = float(np.clip(0.14 * occ_sem + 0.38 * occ_los + 0.26 * occ_cam + 0.22 * occ_traj, 0.0, 1.0))
    clearance_total = 1.0 - occ_total

    score = (
        0.16 * surf +
        0.15 * unc_gain +
        0.23 * grasp_gain +
        0.10 * clr_sem +
        0.08 * clr_los +
        0.10 * clr_cam -
        0.08 * clr_traj -
        0.14 * occ_sem -
        0.22 * occ_los -
        0.20 * occ_cam -
        0.18 * occ_traj +
        0.08 * bbox_metrics["bbox_visibility"] +
        0.08 * bbox_metrics["target_bbox_gain"] +
        0.08 * bbox_metrics["bbox_separation"] -
        0.16 * bbox_metrics["bbox_overlap_penalty"] +
        0.08 * preferred_dir_bonus -
        float(os.environ.get("ACTIVE_NBV_OBSTACLE_DIR_WEIGHT", "0.28")) * obstacle_dir_penalty
    )
    if reachability_fn is not None:
        if nbv_reachable:
            jump_norm = float(np.clip(nbv_q_jump / max(float(os.environ.get("ACTIVE_NBV_MAX_JOINT_JUMP", "0.85")), 1e-6), 0.0, 1.0))
            margin_norm = float(np.clip(nbv_joint_margin / 0.20, 0.0, 1.0))
            pose_norm = float(np.clip(nbv_pose_err / max(float(os.environ.get("ACTIVE_NBV_MAX_PRED_POSE_ERR", "0.035")), 1e-6), 0.0, 1.0))
            score += 0.35 - 0.18 * jump_norm - 0.20 * pose_norm + 0.05 * margin_norm
        else:
            score -= float(os.environ.get("ACTIVE_NBV_UNREACHABLE_PENALTY", "8.0"))
    return {
        "score": float(score),
        "surf": float(surf),
        "unc_gain": float(unc_gain),
        "grasp_gain": float(grasp_gain),
        "clearance": float(clearance_total),
        "clearance_sem": float(clr_sem),
        "clearance_los": float(clr_los),
        "clearance_cam": float(clr_cam),
        "clearance_traj": float(clr_traj),
        "occ_penalty": float(occ_total),
        "occ_sem": float(occ_sem),
        "occ_los": float(occ_los),
        "occ_cam": float(occ_cam),
        "occ_traj": float(occ_traj),
        "obstacle_dir_penalty": float(obstacle_dir_penalty),
        "obstacle_dir_label": str(obstacle_dir_detail.get("label", "")),
        "obstacle_dir_angle_deg": float(obstacle_dir_detail.get("angle_deg", 180.0)),
        "obstacle_dir_selected": bool(obstacle_dir_detail.get("selected", False)),
        "los_count": int(los_detail.get("los_count", 0)),
        "los_ratio": float(los_detail.get("los_ratio", 0.0)),
        "los_nearest": float(los_detail.get("nearest", float("inf"))),
        "cam_clearance_nearest": float(cam_detail.get("nearest", float("inf"))),
        "traj_clearance_nearest": float(traj_detail.get("nearest", float("inf"))),
        "candidate_cam_pos_w": None if candidate_cam_pos_w is None else candidate_cam_pos_w.astype(np.float32),
        "obstacle_meta": obstacle_meta,
        "bbox_visibility": float(bbox_metrics["bbox_visibility"]),
        "bbox_area": float(bbox_metrics["target_bbox_area"]),
        "bbox_gain": float(bbox_metrics["target_bbox_gain"]),
        "bbox_overlap_penalty": float(bbox_metrics["bbox_overlap_penalty"]),
        "bbox_separation": float(bbox_metrics["bbox_separation"]),
        "preferred_dir_bonus": float(preferred_dir_bonus),
        "nbv_reachable": bool(nbv_reachable),
        "nbv_reachability_reason": nbv_reachability_reason,
        "nbv_q_jump": float(nbv_q_jump),
        "nbv_joint_margin": float(nbv_joint_margin),
        "nbv_pose_err": float(nbv_pose_err),
        "nbv_ang_err": float(nbv_ang_err),
        "nbv_q": None if nbv_q is None else nbv_q.astype(np.float32),
        "visibility": v,
    }

def predict_best_view_sequence_scene_aware(points_obj_w,
                                            normals_obj_w,
                                            curvature,
                                            uncertainty,
                                            grasp_heat,
                                            scene_context,
                                            T_wc_center,
                                            orbit_angle_rad,
                                            candidate_num=16,
                                            min_disparity_deg=60.0,
                                            max_disparity_deg=270.0,
                                            debug=False,
                                            target_instance=None,
                                            scene_instances_local=None,
                                            camera=None,
                                            image_shape=None,
                                            preferred_dirs_xy=None,
                                            reachability_fn=None):
    U = _clamp01(uncertainty.astype(np.float32))
    G = _clamp01(grasp_heat.astype(np.float32))

    center_w = scene_context["target_center_w"]
    ctx_pts = scene_context["context_points_w"]
    ctx_wts = scene_context["context_weights"]
    t_select0 = time.perf_counter()

    candidate_num = int(os.environ.get("ACTIVE_NBV_CANDIDATE_NUM", "16"))
    min_disparity_deg = float(os.environ.get("ACTIVE_NBV_MIN_DISPARITY_DEG", str(float(min_disparity_deg))))
    max_disparity_deg = float(os.environ.get("ACTIVE_NBV_MAX_DISPARITY_DEG", str(float(max_disparity_deg))))
    preferred_disparity_deg = float(os.environ.get("ACTIVE_NBV_PREFERRED_DISPARITY_DEG", "75.0"))
    candidate_angles = np.linspace(0.0, 2.0 * np.pi, int(max(candidate_num, 4)), endpoint=False)
    orbit_deg_text = os.environ.get("ACTIVE_NBV_ORBIT_DEG_LIST", "").strip()
    if orbit_deg_text:
        try:
            orbit_angle_list = [
                np.deg2rad(float(x))
                for x in orbit_deg_text.replace(",", " ").split()
                if x.strip()
            ]
        except Exception:
            orbit_angle_list = [float(orbit_angle_rad)]
    else:
        orbit_angle_list = [np.deg2rad(v) for v in (30.0, 40.0)]
    orbit_angle_list = [float(a) for a in orbit_angle_list if np.isfinite(a) and abs(float(a)) > 1e-6]
    if len(orbit_angle_list) == 0:
        orbit_angle_list = [float(orbit_angle_rad)]

    if debug:
        print(f"[ActiveNBV][scene-aware-seq] ctx_pts={ctx_pts.shape}, ctx_wts={ctx_wts.shape}")
        if ctx_pts.shape[0] == 0:
            print("[ActiveNBV][scene-aware-seq] WARNING: scene context is empty; occlusion term will not affect view selection.")

    single_records = []
    for orbit_i, orbit_a in enumerate(orbit_angle_list):
        for ang in candidate_angles:
            dir_xy = _dir_from_angle_rad(ang)
            s = score_single_view_scene_aware(
                normals_obj_w,
                U,
                G,
                center_w,
                ctx_pts,
                ctx_wts,
                dir_xy,
                scene_context=scene_context,
                T_wc_center=T_wc_center,
                orbit_angle_rad=float(orbit_a),
                target_instance=target_instance,
                scene_instances_local=scene_instances_local,
                camera=camera,
                image_shape=image_shape,
                preferred_dirs_xy=preferred_dirs_xy,
                reachability_fn=None,
            )
            def _dbg_cast(v):
                if isinstance(v, np.ndarray):
                    return v
                if isinstance(v, dict):
                    return {kk: _dbg_cast(vv) for kk, vv in v.items()}
                if isinstance(v, (list, tuple)):
                    return type(v)(_dbg_cast(vv) for vv in v)
                if isinstance(v, (np.generic, float, int, bool)):
                    return float(v) if isinstance(v, (np.floating, float, np.integer, int)) else bool(v)
                return v

            detail = {k: _dbg_cast(v) for k, v in s.items()}
            if reachability_fn is not None:
                detail["nbv_reachable"] = False
                detail["nbv_reachability_reason"] = "not_checked_fast_skip"
                detail["nbv_q_jump"] = float("inf")
                detail["nbv_joint_margin"] = -float("inf")
                detail["nbv_pose_err"] = float("inf")
                detail["nbv_ang_err"] = float("inf")
                detail["nbv_q"] = None
            single_records.append({
                "name": f"view_{int(np.rad2deg(ang)) % 360:03d}_orb{int(round(np.rad2deg(float(orbit_a)))):02d}",
                "angle_rad": float(ang),
                "orbit_angle_rad": float(orbit_a),
                "dir_xy": dir_xy.copy(),
                "detail": detail,
                "score": float(s["score"]),
            })
    print(
        f"[ActiveNBV][sphere] orbit_deg_list="
        f"{[round(float(np.rad2deg(a)), 1) for a in orbit_angle_list]} "
        f"candidates={len(single_records)}"
    )

    checked_count = 0
    reach_reason_counts = {}

    def _apply_reachability(rec):
        nonlocal checked_count
        if reachability_fn is None or bool(rec.get("_reach_checked", False)):
            return
        checked_count += 1
        T_wc_cand = _candidate_camera_pose_from_dir(
            center_w,
            T_wc_center,
            rec["dir_xy"],
            rec.get("orbit_angle_rad", float(orbit_angle_rad)),
        )
        reach = reachability_fn(T_wc_cand)
        if (
            not bool(reach.get("reachable", False))
            and _env_flag("ACTIVE_NBV_USE_JOINT_SCAN_FALLBACK", "1")
            and getattr(env.robot, "dof", 0) == 7
        ):
            scan = _find_feasible_object_view_joint_scan(
                env,
                cam_name=EE_CAM_NAME,
                target_w=center_w,
                desired_dir_xy=rec["dir_xy"],
            )
            if scan is not None:
                scan_cam_pos = np.asarray(scan["T_wc"].t, dtype=np.float32).reshape(3,)
                scan_dir_xy = _normalize_dir_xy(scan_cam_pos[:2] - np.asarray(center_w, dtype=np.float32).reshape(3,)[:2])
                if _env_flag("ACTIVE_NBV_REWRITE_JOINT_SCAN_DIR", "1"):
                    s_scan = score_single_view_scene_aware(
                        normals_obj_w,
                        U,
                        G,
                        center_w,
                        ctx_pts,
                        ctx_wts,
                        scan_dir_xy,
                        scene_context=scene_context,
                        T_wc_center=T_wc_center,
                        orbit_angle_rad=rec.get("orbit_angle_rad", float(orbit_angle_rad)),
                        target_instance=target_instance,
                        scene_instances_local=scene_instances_local,
                        camera=camera,
                        image_shape=image_shape,
                        preferred_dirs_xy=preferred_dirs_xy,
                        reachability_fn=None,
                    )
                    old_detail = rec["detail"]
                    rec["detail"] = {k: _dbg_cast(v) for k, v in s_scan.items()}
                    rec["detail"]["requested_angle_rad"] = float(rec["angle_rad"])
                    rec["detail"]["requested_dir_xy"] = np.asarray(rec["dir_xy"], dtype=np.float32).copy()
                    rec["detail"]["requested_score"] = float(rec["score"])
                    rec["score"] = float(s_scan["score"])
                    rec["dir_xy"] = scan_dir_xy.astype(np.float32)
                    rec["angle_rad"] = float(_angle_from_dir_xy(scan_dir_xy))
                    rec["name"] = f"{rec['name']}_scan{int(round(np.rad2deg(rec['angle_rad']))) % 360:03d}"
                    for keep_key in ("nbv_reachable", "nbv_reachability_reason", "nbv_q_jump", "nbv_joint_margin", "nbv_pose_err", "nbv_ang_err", "nbv_q"):
                        if keep_key in old_detail:
                            rec["detail"][keep_key] = old_detail[keep_key]
                reach = {
                    "reachable": True,
                    "reason": "joint_scan",
                    "q_jump": float(scan.get("q_jump", 0.0)),
                    "joint_margin": 0.05,
                    "pose_err": float(scan.get("center_norm", 0.0)),
                    "ang_err": 0.0,
                    "q": np.asarray(scan["q"], dtype=np.float32).reshape(-1),
                    "scan": scan,
                }
        d = rec["detail"]
        d["nbv_reachable"] = bool(reach.get("reachable", False))
        d["nbv_reachability_reason"] = str(reach.get("reason", "unknown"))
        d["nbv_q_jump"] = float(reach.get("q_jump", float("inf")))
        d["nbv_joint_margin"] = float(reach.get("joint_margin", -float("inf")))
        d["nbv_pose_err"] = float(reach.get("pose_err", float("inf")))
        d["nbv_ang_err"] = float(reach.get("ang_err", float("inf")))
        q_reach = reach.get("q", None)
        if q_reach is not None:
            q_reach = np.asarray(q_reach, dtype=np.float32).reshape(-1)
            d["nbv_q"] = q_reach if q_reach.size > 0 and np.isfinite(q_reach).all() else None
        else:
            d["nbv_q"] = None
        if "scan" in reach:
            scan = reach["scan"]
            d["nbv_joint_scan"] = True
            T_scan = scan.get("T_wc", None)
            if T_scan is not None:
                d["nbv_scan_cam_pos_w"] = np.asarray(T_scan.t, dtype=np.float32).reshape(3,)
                scan_R = _orthonormalize_rotation_matrix(T_scan.R, fallback=None)
                d["nbv_scan_R_wc"] = None if scan_R is None else scan_R.astype(np.float32).reshape(3, 3)
            d["nbv_scan_center_norm"] = float(scan.get("center_norm", np.inf))
            d["nbv_scan_distance"] = float(scan.get("distance", np.inf))
            d["nbv_scan_height"] = float(scan.get("height", np.inf))
            d["nbv_scan_lateral"] = float(scan.get("lateral", np.inf))
            d["nbv_scan_dir_score"] = float(scan.get("dir_score", 0.0))
            d["nbv_scan_checked"] = int(scan.get("checked", 0))
            d["nbv_scan_near_fallback"] = bool(scan.get("near_fallback", False))
        else:
            d["nbv_joint_scan"] = False
        reason = str(d["nbv_reachability_reason"])
        reach_reason_counts[reason] = reach_reason_counts.get(reason, 0) + 1
        if bool(d["nbv_reachable"]):
            jump_norm = float(np.clip(d["nbv_q_jump"] / max(float(os.environ.get("ACTIVE_NBV_MAX_JOINT_JUMP", "0.85")), 1e-6), 0.0, 1.0))
            margin_norm = float(np.clip(d["nbv_joint_margin"] / 0.20, 0.0, 1.0))
            pose_norm = float(np.clip(d["nbv_pose_err"] / max(float(os.environ.get("ACTIVE_NBV_MAX_PRED_POSE_ERR", "0.035")), 1e-6), 0.0, 1.0))
            rec["score"] = float(rec["score"]) + 0.35 - 0.18 * jump_norm - 0.20 * pose_norm + 0.05 * margin_norm
        else:
            rec["score"] = float(rec["score"]) - float(os.environ.get("ACTIVE_NBV_UNREACHABLE_PENALTY", "8.0"))
        rec["_reach_checked"] = True

    if reachability_fn is not None:
        visual_sorted = sorted(single_records, key=lambda x: x["score"], reverse=True)
        target_success = int(os.environ.get("ACTIVE_NBV_TARGET_SUCCESS_VIEWS", "2"))
        if _env_flag("ACTIVE_NBV_REACH_CHECK_ALL", "1"):
            for rec in visual_sorted:
                _apply_reachability(rec)
        else:
            topk = min(len(visual_sorted), int(os.environ.get("ACTIVE_NBV_REACH_CHECK_TOPK", "6")))
            max_checks = min(len(visual_sorted), int(os.environ.get("ACTIVE_NBV_REACH_CHECK_MAX", "8")))
            for rec in visual_sorted[:topk]:
                _apply_reachability(rec)
            idx = topk
            while sum(1 for rec in single_records if bool(rec["detail"].get("nbv_reachable", False))) < max(target_success, 1) and idx < max_checks:
                _apply_reachability(visual_sorted[idx])
                idx += 1

    reachable_records = [rec for rec in single_records if bool(rec["detail"].get("nbv_reachable", True))]
    sort_pool = reachable_records if len(reachable_records) > 0 else single_records
    if len(reachable_records) > 0 and len(reachable_records) < len(single_records):
        print(f"[ActiveNBV][reach] reachable_candidates={len(reachable_records)}/{len(single_records)}; filtering unreachable views before selection.")
    elif len(reachable_records) == 0 and reachability_fn is not None:
        print("[ActiveNBV][reach] WARNING: no reachable active-view candidates; falling back to best scored candidate.")
    if reachability_fn is not None:
        print(f"[ActiveNBV][reach] checked={checked_count}/{len(single_records)} reason_counts={reach_reason_counts}")
    single_records_sorted = sorted(sort_pool, key=lambda x: x["score"], reverse=True)
    best1 = single_records_sorted[0]
    v1 = best1["detail"]["visibility"]

    second_candidates = []
    for rec in sort_pool:
        disp = _smallest_angle_diff_deg(rec["angle_rad"], best1["angle_rad"])
        if disp < float(min_disparity_deg) or disp > float(max_disparity_deg):
            continue
        v2 = rec["detail"]["visibility"]
        novelty = float(np.mean((1.0 - v1) * v2))
        unc_new = float(np.mean(U * (1.0 - v1) * v2))
        grasp_new = float(np.mean(G * U * (1.0 - v1) * v2))
        redundancy = float(np.mean(v1 * v2))
        if preferred_disparity_deg > min_disparity_deg:
            diversity_bonus = float(np.clip((disp - min_disparity_deg) / (preferred_disparity_deg - min_disparity_deg), 0.0, 1.0))
        else:
            diversity_bonus = float(np.clip(disp / 180.0, 0.0, 1.0))
        second_score = (
            0.18 * novelty +
            0.18 * unc_new +
            0.30 * grasp_new +
            0.16 * diversity_bonus +
            0.18 * rec["detail"]["clearance"] -
            0.28 * rec["detail"]["occ_penalty"] -
            0.08 * redundancy +
            0.12 * float(rec["detail"].get("bbox_separation", 1.0)) +
            0.08 * float(rec["detail"].get("bbox_gain", 0.5)) +
            0.06 * float(rec["detail"].get("preferred_dir_bonus", 0.0))
        )
        second_candidates.append({
            "name": rec["name"],
            "angle_rad": rec["angle_rad"],
            "orbit_angle_rad": rec.get("orbit_angle_rad", float(orbit_angle_rad)),
            "dir_xy": rec["dir_xy"].copy(),
            "score": float(second_score),
            "detail": {
                "novelty": float(novelty),
                "unc_new": float(unc_new),
                "grasp_new": float(grasp_new),
                "redundancy": float(redundancy),
                "diversity_bonus": float(diversity_bonus),
                "clearance": float(rec["detail"]["clearance"]),
                "occ_penalty": float(rec["detail"]["occ_penalty"]),
                "nbv_q": rec["detail"].get("nbv_q", None),
                "nbv_reachable": bool(rec["detail"].get("nbv_reachable", True)),
                "nbv_reachability_reason": str(rec["detail"].get("nbv_reachability_reason", "not_checked")),
                "occ_sem": float(rec["detail"].get("occ_sem", 0.0)),
                "occ_los": float(rec["detail"].get("occ_los", 0.0)),
                "occ_cam": float(rec["detail"].get("occ_cam", 0.0)),
                "occ_traj": float(rec["detail"].get("occ_traj", 0.0)),
                "bbox_gain": float(rec["detail"].get("bbox_gain", 0.5)),
                "bbox_separation": float(rec["detail"].get("bbox_separation", 1.0)),
                "disparity_deg": float(disp),
                "orbit_angle_deg": float(np.rad2deg(rec.get("orbit_angle_rad", float(orbit_angle_rad)))),
            }
        })

    if len(second_candidates) == 0:
        if len(sort_pool) == 1 and bool(best1["detail"].get("nbv_reachable", True)):
            second_candidates = [{
                "name": best1["name"],
                "angle_rad": best1["angle_rad"],
                "orbit_angle_rad": best1.get("orbit_angle_rad", float(orbit_angle_rad)),
                "dir_xy": best1["dir_xy"].copy(),
                "score": float(best1["score"]),
                "detail": {
                    "disparity_deg": 0.0,
                    "duplicate_primary": True,
                    "nbv_q": best1["detail"].get("nbv_q", None),
                    "nbv_reachable": bool(best1["detail"].get("nbv_reachable", True)),
                    "nbv_reachability_reason": str(best1["detail"].get("nbv_reachability_reason", "not_checked")),
                }
            }]
        else:
            # fallback: choose the farthest non-identical candidate from reachable views first
            fallback_pool = sort_pool if len(sort_pool) > 1 else single_records
            second_candidates = [{
                "name": rec["name"],
                "angle_rad": rec["angle_rad"],
                "orbit_angle_rad": rec.get("orbit_angle_rad", float(orbit_angle_rad)),
                "dir_xy": rec["dir_xy"].copy(),
                "score": _smallest_angle_diff_deg(rec["angle_rad"], best1["angle_rad"]),
                "detail": {
                    "disparity_deg": _smallest_angle_diff_deg(rec["angle_rad"], best1["angle_rad"]),
                    "orbit_angle_deg": float(np.rad2deg(rec.get("orbit_angle_rad", float(orbit_angle_rad)))),
                    "nbv_q": rec["detail"].get("nbv_q", None),
                    "nbv_reachable": bool(rec["detail"].get("nbv_reachable", True)),
                    "nbv_reachability_reason": str(rec["detail"].get("nbv_reachability_reason", "not_checked")),
                }
            } for rec in fallback_pool if rec["name"] != best1["name"]]
            if len(second_candidates) == 0:
                second_candidates = [{
                    "name": rec["name"],
                    "angle_rad": rec["angle_rad"],
                    "orbit_angle_rad": rec.get("orbit_angle_rad", float(orbit_angle_rad)),
                    "dir_xy": rec["dir_xy"].copy(),
                    "score": _smallest_angle_diff_deg(rec["angle_rad"], best1["angle_rad"]),
                    "detail": {
                        "disparity_deg": _smallest_angle_diff_deg(rec["angle_rad"], best1["angle_rad"]),
                        "orbit_angle_deg": float(np.rad2deg(rec.get("orbit_angle_rad", float(orbit_angle_rad)))),
                        "nbv_q": rec["detail"].get("nbv_q", None),
                        "nbv_reachable": bool(rec["detail"].get("nbv_reachable", True)),
                        "nbv_reachability_reason": str(rec["detail"].get("nbv_reachability_reason", "not_checked")),
                    }
                } for rec in single_records if rec["name"] != best1["name"]]

    best2 = sorted(second_candidates, key=lambda x: x["score"], reverse=True)[0]

    if debug:
        print("[ActiveNBV][scene-aware-seq] top single views:")
        for rec in single_records_sorted[:4]:
            d = rec["detail"]
            print(
                f"  {rec['name']} score={rec['score']:.4f} surf={d['surf']:.4f} unc={d['unc_gain']:.4f} "
                f"grasp={d['grasp_gain']:.4f} clr={d['clearance']:.4f} occ={d['occ_penalty']:.4f} "
                f"bbox_gain={d.get('bbox_gain', 0.5):.4f} bbox_sep={d.get('bbox_separation', 1.0):.4f} "
                f"reach={d.get('nbv_reachable', True)} qjump={d.get('nbv_q_jump', 0.0):.4f} "
                f"margin={d.get('nbv_joint_margin', float('inf')):.4f} pose_err={d.get('nbv_pose_err', 0.0):.4f} "
                f"reason={d.get('nbv_reachability_reason', 'not_checked')} "
                f"obs_dir={d.get('obstacle_dir_penalty', 0.0):.4f}/{d.get('obstacle_dir_label', '')}"
                f"@{d.get('obstacle_dir_angle_deg', 180.0):.1f}deg "
                f"(sem={d.get('occ_sem', 0.0):.4f}, los={d.get('occ_los', 0.0):.4f}, cam={d.get('occ_cam', 0.0):.4f}, traj={d.get('occ_traj', 0.0):.4f})"
            )
        print("[ActiveNBV][scene-aware-seq] selected sequence:")
        print(f"  primary={best1['name']} angle={np.rad2deg(best1['angle_rad']):.1f} orbit={np.rad2deg(best1.get('orbit_angle_rad', float(orbit_angle_rad))):.1f}")
        print(f"  secondary={best2['name']} angle={np.rad2deg(best2['angle_rad']):.1f} orbit={np.rad2deg(best2.get('orbit_angle_rad', float(orbit_angle_rad))):.1f} detail={best2['detail']}")

    selected_records = [
        rec for rec in (best1, best2)
        if reachability_fn is None or (
            bool(rec["detail"].get("nbv_reachable", False)) and rec["detail"].get("nbv_q", None) is not None
        )
    ]
    seen_names = {best1["name"], best2["name"]}
    exec_min_disp = float(os.environ.get("ACTIVE_NBV_EXEC_QUEUE_MIN_DISPARITY_DEG", "25.0"))

    def _can_execute_rec(rec):
        if rec["name"] in seen_names:
            return False
        if reachability_fn is not None and (
            not bool(rec["detail"].get("nbv_reachable", False)) or rec["detail"].get("nbv_q", None) is None
        ):
            return False
        return True

    def _min_disp_to_selected(rec):
        if len(selected_records) == 0:
            return 180.0
        return min(_smallest_angle_diff_deg(rec["angle_rad"], prev["angle_rad"]) for prev in selected_records)

    for strict_pass in (True, False):
        for rec in single_records_sorted:
            if not _can_execute_rec(rec):
                continue
            if strict_pass and _min_disp_to_selected(rec) < exec_min_disp:
                continue
            if strict_pass and float(rec.get("detail", {}).get("obstacle_dir_penalty", 0.0)) > float(os.environ.get("ACTIVE_NBV_EXEC_MAX_OBSTACLE_DIR_PENALTY", "0.72")):
                continue
            selected_records.append(rec)
            seen_names.add(rec["name"])
    selected_reachable = sum(1 for rec in selected_records if bool(rec["detail"].get("nbv_reachable", True)))
    selected_with_q = sum(1 for rec in selected_records if rec["detail"].get("nbv_q", None) is not None)
    print(
        f"[ActiveNBV][reach] exec_queue={len(selected_records)} "
        f"reachable={selected_reachable} with_ik_q={selected_with_q}"
    )
    print(
        f"[ActiveNBV][timing] selection_sec={time.perf_counter() - t_select0:.3f} "
        f"candidates={len(single_records)} reachable={len(reachable_records)}"
    )

    return {
        "best_pair_name": "scene_aware_sequence",
        "primary_view": best1,
        "secondary_view": best2,
        "selected_view_dirs_xy": [best1["dir_xy"].astype(np.float32), best2["dir_xy"].astype(np.float32)],
        "selected_view_records": selected_records,
        "reachable_candidate_count": int(len(reachable_records)),
        "candidate_records": single_records,
        "secondary_candidates": second_candidates,
    }


def detect_front_occluders(scene_context,
                           target_center_w,
                           front_dir_w,
                           corridor_len=0.14,
                           corridor_radius=0.05,
                           min_points=200,
                           debug=False):
    front = np.asarray(front_dir_w, dtype=np.float32).reshape(3,)
    n = float(np.linalg.norm(front))
    if n < 1e-8:
        return {"has_front_occluder": False, "front_point_count": 0, "occlusion_ratio": 0.0, "nearest_dist": np.inf, "front_instances": []}
    front = front / n
    center = np.asarray(target_center_w, dtype=np.float32).reshape(3,)

    all_pts = np.asarray(scene_context.get("obstacle_points_w", scene_context.get("context_points_w", np.zeros((0, 3), dtype=np.float32))), dtype=np.float32).reshape(-1, 3)
    if all_pts.shape[0] == 0:
        return {"has_front_occluder": False, "front_point_count": 0, "occlusion_ratio": 0.0, "nearest_dist": np.inf, "front_instances": []}

    rel = all_pts - center[None, :]
    axial = rel @ front
    perp = np.linalg.norm(rel - axial[:, None] * front[None, :], axis=1)
    keep = (axial > 0.0) & (axial < float(corridor_len)) & (perp < float(corridor_radius))
    front_pts = all_pts[keep]
    nearest_dist = float(np.min(axial[keep])) if np.any(keep) else np.inf
    occlusion_ratio = float(np.count_nonzero(keep) / max(all_pts.shape[0], 1))

    front_instances = []
    target_instance = scene_context.get("target_instance", None)
    target_pts = np.asarray(
        target_instance.get("points_w", np.zeros((0, 3), dtype=np.float32)) if target_instance is not None else np.zeros((0, 3), dtype=np.float32),
        dtype=np.float32,
    ).reshape(-1, 3)
    target_tree = cKDTree(target_pts) if target_pts.shape[0] >= 20 else None
    camera = scene_context.get("camera_info", None)
    image_shape = scene_context.get("image_shape", None)
    T_wc = scene_context.get("T_wc", None)
    target_xyxy = None
    if target_instance is not None and camera is not None and image_shape is not None and T_wc is not None:
        target_xyxy = _project_instance_bbox_xyxy(target_instance, T_wc, camera, image_shape=image_shape)
    for inst in scene_context.get("context_instance_clouds", []):
        pts = np.asarray(inst.get("points_w", np.zeros((0, 3), dtype=np.float32)), dtype=np.float32).reshape(-1, 3)
        if pts.shape[0] == 0:
            continue
        rel_i = pts - center[None, :]
        axial_i = rel_i @ front
        perp_i = np.linalg.norm(rel_i - axial_i[:, None] * front[None, :], axis=1)
        keep_i = (axial_i > 0.0) & (axial_i < float(corridor_len)) & (perp_i < float(corridor_radius))
        count_i = int(np.count_nonzero(keep_i))
        if count_i < int(min_points):
            continue
        nearest_i = float(np.min(axial_i[keep_i])) if count_i > 0 else np.inf
        nearest_perp_i = float(np.min(perp_i[keep_i])) if count_i > 0 else np.inf
        front_ratio_i = float(count_i / max(pts.shape[0], 1))
        target_near_ratio = 0.0
        target_near_count = 0
        if target_tree is not None and count_i > 0:
            d_nn, _ = target_tree.query(pts[keep_i], k=1, workers=-1)
            near_gate = float(os.environ.get("FRONT_OCC_TARGET_POINT_NEAR_GATE", "0.035"))
            target_near_count = int(np.count_nonzero(d_nn <= near_gate))
            target_near_ratio = float(target_near_count / max(count_i, 1))
        inst_xyxy = inst.get("xyxy", None)
        if inst_xyxy is None and camera is not None and image_shape is not None and T_wc is not None:
            inst_xyxy = _project_instance_bbox_xyxy(inst, T_wc, camera, image_shape=image_shape)
        projected_overlap = _bbox_overlap_ratio_xyxy(target_xyxy, inst_xyxy) if target_xyxy is not None else 0.0
        is_user_clicked = bool(inst.get("click_uv", None) is not None or str(inst.get("label", "")).lower().startswith("click_"))
        front_instances.append({
            "id": int(inst.get("id", -1)),
            "label": inst.get("label", "unknown"),
            "role": str(inst.get("role", "unknown")),
            "score": float(inst.get("score", 1.0)),
            "weight": float(inst.get("weight", 1.0)),
            "front_point_count": count_i,
            "nearest_dist": nearest_i,
            "nearest_perp": nearest_perp_i,
            "front_ratio": front_ratio_i,
            "target_near_ratio": target_near_ratio,
            "target_near_count": target_near_count,
            "seed": inst.get("seed", None),
            "bbox_center": inst.get("bbox_center", None),
            "bbox_R": inst.get("bbox_R", None),
            "bbox_extent": inst.get("bbox_extent", None),
            "points_w": pts.astype(np.float32),
            "mask": inst.get("mask", None),
            "xyxy": inst_xyxy,
            "projected_overlap": float(projected_overlap),
            "is_user_clicked": bool(is_user_clicked),
            "center": inst.get("center", None),
        })

    front_instances = sorted(front_instances, key=lambda x: (x["nearest_dist"], -x["front_point_count"]))
    has_front_occluder = (front_pts.shape[0] >= int(min_points)) or (len(front_instances) > 0)
    front_center_w = None
    front_nearest_center_w = None
    front_nearest_pts = np.zeros((0, 3), dtype=np.float32)
    if front_pts.shape[0] > 0:
        front_center_w = np.median(front_pts, axis=0).astype(np.float32)
        near_band = float(os.environ.get("FRONT_OCC_NEAREST_BAND_M", "0.025"))
        nearest_gate = nearest_dist + max(near_band, 1e-6)
        near_keep = axial[keep] <= nearest_gate
        front_nearest_pts = front_pts[near_keep]
        if front_nearest_pts.shape[0] > 0:
            front_nearest_center_w = np.median(front_nearest_pts, axis=0).astype(np.float32)

    if debug:
        print(f"[FrontOcc] total_front_points={front_pts.shape[0]} nearest_dist={nearest_dist:.4f} occlusion_ratio={occlusion_ratio:.4f}")
        if front_center_w is not None or front_nearest_center_w is not None:
            print(
                f"[FrontOcc] geometry_center={np.round(front_center_w, 4) if front_center_w is not None else None} "
                f"nearest_center={np.round(front_nearest_center_w, 4) if front_nearest_center_w is not None else None} "
                f"nearest_pts={int(front_nearest_pts.shape[0])}"
            )
        if len(front_instances) > 0:
            if VERBOSE_LOG:
                print(f"[FrontOcc] front_instances={front_instances[:4]}")
            else:
                print(f"[FrontOcc] front_instances_summary={[_brief_instance(x) for x in front_instances[:4]]}")

    return {
        "has_front_occluder": bool(has_front_occluder),
        "front_point_count": int(front_pts.shape[0]),
        "occlusion_ratio": float(occlusion_ratio),
        "nearest_dist": float(nearest_dist),
        "target_center_w": center.astype(np.float32),
        "front_dir_w": front.astype(np.float32),
        "front_points_w": front_pts.astype(np.float32),
        "front_center_w": front_center_w,
        "front_nearest_points_w": front_nearest_pts.astype(np.float32),
        "front_nearest_center_w": front_nearest_center_w,
        "front_instances": front_instances,
    }


def _mask_coverage(mask_a, mask_b):
    a = np.asarray(mask_a) > 0
    b = np.asarray(mask_b) > 0
    denom = int(np.count_nonzero(a))
    if denom <= 0:
        return 0.0
    return float(np.count_nonzero(a & b) / max(denom, 1))


def _mask_intersection_over_denominator(mask_a, mask_b, denom_mask):
    a = np.asarray(mask_a) > 0
    b = np.asarray(mask_b) > 0
    d = np.asarray(denom_mask) > 0
    denom = int(np.count_nonzero(d))
    if denom <= 0:
        return 0.0
    return float(np.count_nonzero(a & b & d) / max(denom, 1))


def _mask_iou(mask_a, mask_b):
    a = np.asarray(mask_a) > 0
    b = np.asarray(mask_b) > 0
    union = int(np.count_nonzero(a | b))
    if union <= 0:
        return 0.0
    return float(np.count_nonzero(a & b) / max(union, 1))


def _median_depth_in_mask(depth, mask):
    depth = np.asarray(depth, dtype=np.float32)
    m = np.asarray(mask) > 0
    vals = depth[m & np.isfinite(depth) & (depth > 0.03)]
    if vals.size == 0:
        return float("inf")
    return float(np.median(vals))


def _hidden_mask_occluder_points_world(depth,
                                       hidden_mask,
                                       visible_depth: float,
                                       T_wc,
                                       fovy_rad: float,
                                       min_points: int = 30):
    if T_wc is None:
        return np.zeros((0, 3), dtype=np.float32), None
    depth_m = np.asarray(depth, dtype=np.float32)
    if depth_m.ndim == 3:
        depth_m = depth_m[..., 0]
    if np.issubdtype(np.asarray(depth).dtype, np.integer):
        depth_m = depth_m / 1000.0
    mask = np.asarray(hidden_mask, dtype=np.uint8)
    if mask.shape != depth_m.shape:
        mask = cv2.resize(mask, (depth_m.shape[1], depth_m.shape[0]), interpolation=cv2.INTER_NEAREST)
    h, w = depth_m.shape[:2]
    K = _build_cam_intrinsics_from_fovy(w, h, float(fovy_rad))
    camera = CameraInfo(w, h, K[0, 0], K[1, 1], K[0, 2], K[1, 2], 1.0)
    cloud_c = create_point_cloud_from_depth_image(depth_m.astype(np.float32), camera, organized=True)
    valid = (mask > 0) & np.isfinite(depth_m) & (depth_m > 0.03)
    if np.isfinite(float(visible_depth)):
        depth_margin = float(os.environ.get("AMODAL_HIDDEN_OCCLUDER_DEPTH_MARGIN", "0.004"))
        front_valid = valid & (depth_m < (float(visible_depth) - depth_margin))
        if int(np.count_nonzero(front_valid)) >= int(min_points):
            valid = front_valid
    pts_c = cloud_c[valid].astype(np.float32)
    if pts_c.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.float32), None
    max_points = int(os.environ.get("AMODAL_HIDDEN_OCCLUDER_MAX_POINTS", "6000"))
    if pts_c.shape[0] > max_points:
        idx = np.linspace(0, pts_c.shape[0] - 1, max_points).astype(np.int64)
        pts_c = pts_c[idx]
    pts_w = _cam_to_world_points(T_wc, pts_c).astype(np.float32)
    center_w = np.median(pts_w, axis=0).astype(np.float32)
    return pts_w, center_w


def _same_label(label_a, label_b):
    a = str(label_a or "").strip().lower().replace("_", " ")
    b = str(label_b or "").strip().lower().replace("_", " ")
    ac = a.replace(" ", "").replace("-", "")
    bc = b.replace(" ", "").replace("-", "")
    return bool(a and b and (a == b or a in b or b in a or ac == bc or ac in bc or bc in ac))


def _select_target_amodal_instance(amodal_instances, target_instance, target_class=None):
    if len(amodal_instances) == 0:
        return None
    target_mask = None if target_instance is None else target_instance.get("mask", None)
    target_label = target_class or (None if target_instance is None else target_instance.get("label", None))
    best = None
    best_score = -np.inf
    for inst in amodal_instances:
        if target_label is not None and not _same_label(inst.get("label", ""), target_label):
            continue
        cov = _mask_coverage(target_mask, inst["mask"]) if target_mask is not None else 0.0
        visible_over_amodal = _mask_coverage(inst["mask"], target_mask) if target_mask is not None else 0.0
        iou = _mask_iou(target_mask, inst["mask"]) if target_mask is not None else 0.0
        area_norm = float(np.count_nonzero(inst["mask"]) / max(np.asarray(inst["mask"]).size, 1))
        score = 2.0 * cov + 0.6 * iou + 0.2 * float(inst.get("score", 0.0)) + 0.05 * area_norm
        if score > best_score:
            best_score = score
            best = dict(inst)
            best["_target_visible_alignment"] = float(cov)
            best["_target_visible_over_amodal"] = float(visible_over_amodal)
            best["_target_amodal_occlusion_ratio"] = float(np.clip(1.0 - visible_over_amodal, 0.0, 1.0))
            best["_target_visible_iou"] = float(iou)
    if best is None and target_mask is not None:
        for inst in amodal_instances:
            cov = _mask_coverage(target_mask, inst["mask"])
            visible_over_amodal = _mask_coverage(inst["mask"], target_mask)
            iou = _mask_iou(target_mask, inst["mask"])
            score = 2.0 * cov + 0.6 * iou + 0.2 * float(inst.get("score", 0.0))
            if score > best_score:
                best_score = score
                best = dict(inst)
                best["_target_visible_alignment"] = float(cov)
                best["_target_visible_over_amodal"] = float(visible_over_amodal)
                best["_target_amodal_occlusion_ratio"] = float(np.clip(1.0 - visible_over_amodal, 0.0, 1.0))
                best["_target_visible_iou"] = float(iou)
    return best


def augment_front_occluders_with_amodal(scene_context,
                                        front_occ_info,
                                        bgr,
                                        depth,
                                        fovy_rad=None,
                                        target_class=None,
                                        debug_dir=None,
                                        iter_tag="clear",
                                        debug=True):
    if not _env_flag("ENABLE_AMODAL_CLEAR_REASONING", "1"):
        return front_occ_info
    target_instance = scene_context.get("target_instance", None)
    if target_instance is None or target_instance.get("mask", None) is None:
        return front_occ_info

    try:
        amodal_instances, _ = infer_instances_yolo26_amodal(
            bgr,
            target_class=None,
            output_vis_path=None if debug_dir is None else _debug_iter_path(debug_dir, iter_tag, "amodal_yolo_overlay"),
            conf_thresh=float(os.environ.get("YOLO26_AMODAL_CLEAR_CONF", "0.15")),
        )
    except Exception as exc:
        if debug:
            print(f"[AmodalClear] skip amodal reasoning: {type(exc).__name__}: {exc}")
        return front_occ_info

    target_amodal = _select_target_amodal_instance(amodal_instances, target_instance, target_class=target_class)
    if target_amodal is None:
        if debug:
            print("[AmodalClear] no target amodal instance matched current target.")
        return front_occ_info

    target_visible_mask = np.asarray(target_instance["mask"], dtype=np.uint8)
    target_amodal_mask = np.asarray(target_amodal["mask"], dtype=np.uint8)
    target_hidden_mask = ((target_amodal_mask > 0) & ~(target_visible_mask > 0)).astype(np.uint8)
    target_visible_over_amodal = _mask_coverage(target_amodal_mask, target_visible_mask)
    target_amodal_occlusion_ratio = float(np.clip(1.0 - target_visible_over_amodal, 0.0, 1.0))
    target_visible_depth = _median_depth_in_mask(depth, target_visible_mask)
    T_wc_ctx = scene_context.get("T_wc", None)
    if fovy_rad is None:
        fovy_rad = float(os.environ.get("DEFAULT_CAMERA_FOVY_RAD", str(np.pi / 4.0)))
    hidden_min_points = int(os.environ.get("AMODAL_HIDDEN_OCCLUDER_MIN_POINTS", "30"))
    target_hidden_points_w, target_hidden_center_w = _hidden_mask_occluder_points_world(
        depth,
        target_hidden_mask,
        target_visible_depth,
        T_wc_ctx,
        float(fovy_rad),
        min_points=hidden_min_points,
    )
    front_occ_info["target_label"] = target_class or target_instance.get("label", "")
    front_occ_info["target_visible_instance_id"] = target_instance.get("id", None)
    same_object_ids = set()
    occluder_metrics = {}
    same_object_cov_thresh = float(os.environ.get("AMODAL_SAME_OBJECT_COVERAGE", "0.68"))
    occ_cov_thresh = float(os.environ.get("AMODAL_OCCLUDER_COVERAGE", "0.18"))
    depth_margin = float(os.environ.get("AMODAL_OCCLUDER_DEPTH_MARGIN", "0.008"))

    for inst in scene_context.get("scene_instances", []):
        inst_id = int(inst.get("id", -1))
        mask = inst.get("mask", None)
        if mask is None or inst_id < 0:
            continue
        role = str(inst.get("role", "unknown")).lower()
        if role == "target":
            continue
        same_cls = _same_label(inst.get("label", ""), target_class or target_instance.get("label", ""))
        inst_mask = np.asarray(mask, dtype=np.uint8)
        cov_inst_inside_target_amodal = _mask_coverage(inst_mask, target_amodal_mask)
        cov_target_amodal_by_inst = _mask_intersection_over_denominator(inst_mask, target_amodal_mask, target_amodal_mask)
        cov_target_hidden_by_inst = _mask_intersection_over_denominator(inst_mask, target_hidden_mask, target_hidden_mask)
        amodal_blocking_ratio = float(max(cov_target_amodal_by_inst, cov_target_hidden_by_inst * target_amodal_occlusion_ratio))
        iou_with_target_amodal = _mask_iou(inst_mask, target_amodal_mask)
        inst_depth = _median_depth_in_mask(depth, inst_mask)
        in_front_depth = bool(np.isfinite(inst_depth) and np.isfinite(target_visible_depth) and inst_depth < (target_visible_depth - depth_margin))
        if same_cls and cov_inst_inside_target_amodal >= same_object_cov_thresh:
            same_object_ids.add(inst_id)
            continue
        if amodal_blocking_ratio >= occ_cov_thresh and in_front_depth:
            occluder_metrics[inst_id] = {
                "amodal_overlap_coverage": float(amodal_blocking_ratio),
                "amodal_inst_inside_target": float(cov_inst_inside_target_amodal),
                "amodal_target_covered_by_inst": float(cov_target_amodal_by_inst),
                "amodal_hidden_covered_by_inst": float(cov_target_hidden_by_inst),
                "amodal_iou": float(iou_with_target_amodal),
                "amodal_depth_delta": float(target_visible_depth - inst_depth),
                "amodal_front_depth": True,
            }

    front_instances = []
    for inst in front_occ_info.get("front_instances", []):
        inst_id = int(inst.get("id", -1))
        if inst_id in same_object_ids:
            continue
        out = dict(inst)
        if inst_id in occluder_metrics:
            met = occluder_metrics[inst_id]
            out.update(met)
            out["target_near_ratio"] = float(max(float(out.get("target_near_ratio", 0.0)), met["amodal_overlap_coverage"]))
            out["front_ratio"] = float(max(float(out.get("front_ratio", 0.0)), min(1.0, met["amodal_overlap_coverage"] + 0.10)))
            out["projected_overlap"] = float(max(float(out.get("projected_overlap", 0.0)), met["amodal_iou"]))
        front_instances.append(out)

    existing_ids = {int(inst.get("id", -1)) for inst in front_instances}
    for inst in scene_context.get("scene_instances", []):
        inst_id = int(inst.get("id", -1))
        if inst_id in existing_ids or inst_id in same_object_ids or inst_id not in occluder_metrics:
            continue
        pts = np.asarray(inst.get("points", inst.get("points_w", np.zeros((0, 3), dtype=np.float32))), dtype=np.float32).reshape(-1, 3)
        if pts.shape[0] == 0:
            continue
        met = occluder_metrics[inst_id]
        front_instances.append({
            "id": inst_id,
            "label": inst.get("label", "unknown"),
            "role": str(inst.get("role", "obstacle")),
            "score": float(inst.get("score", 1.0)),
            "weight": float(inst.get("weight", 1.0)),
            "front_point_count": int(pts.shape[0]),
            "nearest_dist": float(os.environ.get("AMODAL_SYNTHETIC_NEAREST_DIST", "0.035")),
            "nearest_perp": 0.0,
            "front_ratio": float(min(1.0, met["amodal_overlap_coverage"] + 0.10)),
            "target_near_ratio": float(met["amodal_overlap_coverage"]),
            "target_near_count": int(round(met["amodal_overlap_coverage"] * max(pts.shape[0], 1))),
            "seed": inst.get("seed", None),
            "bbox_center": inst.get("bbox_center", None),
            "bbox_R": inst.get("bbox_R", None),
            "bbox_extent": inst.get("bbox_extent", None),
            "points_w": pts.astype(np.float32),
            "mask": inst.get("mask", None),
            "xyxy": inst.get("xyxy", None),
            "projected_overlap": float(met["amodal_iou"]),
            "is_user_clicked": bool(inst.get("is_user_clicked", False)),
            "center": inst.get("center", None),
            **met,
        })

    out_info = dict(front_occ_info)
    out_info["front_instances"] = sorted(front_instances, key=lambda x: (float(x.get("nearest_dist", np.inf)), -int(x.get("front_point_count", 0))))
    out_info["has_front_occluder"] = bool(out_info.get("has_front_occluder", False) or len(out_info["front_instances"]) > 0)
    out_info["amodal_enabled"] = True
    out_info["same_object_instance_ids"] = sorted(same_object_ids)
    out_info["amodal_occluder_ids"] = sorted(occluder_metrics.keys())
    out_info["target_hidden_points_w"] = target_hidden_points_w.astype(np.float32)
    out_info["target_hidden_center_w"] = target_hidden_center_w
    out_info["target_hidden_point_count"] = int(target_hidden_points_w.shape[0])
    out_info["target_amodal_occlusion_ratio"] = float(target_amodal_occlusion_ratio)
    scene_context["target_visible_mask"] = target_visible_mask
    scene_context["target_amodal_mask"] = target_amodal_mask
    scene_context["target_hidden_mask"] = target_hidden_mask
    scene_context["target_hidden_points_w"] = target_hidden_points_w.astype(np.float32)
    scene_context["target_hidden_center_w"] = target_hidden_center_w
    scene_context["target_visible_over_amodal"] = float(target_visible_over_amodal)
    scene_context["target_amodal_occlusion_ratio"] = float(target_amodal_occlusion_ratio)
    scene_context["target_amodal_instance"] = target_amodal
    scene_context["same_object_instance_ids"] = sorted(same_object_ids)
    if debug_dir is not None:
        cv2.imwrite(_debug_iter_path(debug_dir, iter_tag, "target_amodal_mask"), target_amodal_mask)
        cv2.imwrite(_debug_iter_path(debug_dir, iter_tag, "target_visible_mask"), target_visible_mask)
        cv2.imwrite(_debug_iter_path(debug_dir, iter_tag, "target_hidden_mask"), target_hidden_mask * 255)
        vis_cmp = np.zeros((*target_visible_mask.shape[:2], 3), dtype=np.uint8)
        visible_bool = target_visible_mask > 0
        amodal_bool = target_amodal_mask > 0
        hidden_bool = target_hidden_mask > 0
        vis_cmp[amodal_bool] = (0, 0, 180)
        vis_cmp[visible_bool] = (0, 180, 0)
        vis_cmp[visible_bool & amodal_bool] = (0, 220, 220)
        vis_cmp[hidden_bool] = (0, 0, 255)
        cv2.imwrite(_debug_iter_path(debug_dir, iter_tag, "target_visible_vs_amodal_mask"), vis_cmp)
    if debug:
        print(
            f"[AmodalClear] amodal_target label={target_amodal.get('label', 'unknown')} "
            f"visible_align={float(target_amodal.get('_target_visible_alignment', 0.0)):.3f} "
            f"visible_over_amodal={target_visible_over_amodal:.3f} "
            f"amodal_occ_ratio={target_amodal_occlusion_ratio:.3f} "
            f"hidden_pts={int(target_hidden_points_w.shape[0])} "
            f"hidden_center={np.round(target_hidden_center_w, 4) if target_hidden_center_w is not None else None} "
            f"same_object_ids={sorted(same_object_ids)} occluder_ids={sorted(occluder_metrics.keys())}"
        )
    return out_info


def should_clear_occluder_first(front_occ_info,
                                severe_ratio_thresh=0.10,
                                near_dist_thresh=0.08,
                                min_front_points=1500,
                                min_cloud_front_ratio=0.18,
                                min_target_near_ratio=0.10,
                                min_clicked_front_ratio=0.12,
                                debug=True):
    if not front_occ_info.get("has_front_occluder", False):
        return {"should_clear_first": False, "score": 0.0, "reason": "no_front_occluder"}

    ratio = float(front_occ_info.get("occlusion_ratio", 0.0))
    nearest = float(front_occ_info.get("nearest_dist", np.inf))
    front_pts = int(front_occ_info.get("front_point_count", 0))

    front_instances = front_occ_info.get("front_instances", [])
    only_geometry_fallback = (len(front_instances) > 0) and all(str(x.get("label", "")).lower() == "geometry_fallback" for x in front_instances)
    actionable_instances = []
    strong_actionable_instances = []
    blocking_instances = []
    for inst in front_instances:
        label = str(inst.get("label", "")).lower()
        mask = inst.get("mask", None)
        if label == "geometry_fallback":
            continue
        if (mask is not None) or (inst.get("bbox_center", None) is not None) or (inst.get("points_w", None) is not None):
            actionable_instances.append(inst)
            is_clicked = bool(inst.get("is_user_clicked", False))
            score_sem = float(inst.get("score", 0.0))
            front_ratio_i = float(inst.get("front_ratio", 0.0))
            target_near_ratio_i = float(inst.get("target_near_ratio", 0.0))
            nearest_i = float(inst.get("nearest_dist", np.inf))
            point_count_i = int(inst.get("front_point_count", 0))
            cloud_blocks = (
                nearest_i <= near_dist_thresh
                and front_ratio_i >= float(min_cloud_front_ratio)
                and target_near_ratio_i >= float(min_target_near_ratio)
                and point_count_i >= int(min_front_points * 0.10)
            )
            clicked_blocks = (
                is_clicked
                and nearest_i <= near_dist_thresh
                and front_ratio_i >= min_clicked_front_ratio
                and target_near_ratio_i >= max(0.04, 0.5 * float(min_target_near_ratio))
            )
            if clicked_blocks:
                strong_actionable_instances.append(inst)
                blocking_instances.append(inst)
                continue
            if cloud_blocks and score_sem >= 0.30:
                strong_actionable_instances.append(inst)
                blocking_instances.append(inst)

    score = 0.0
    score += 0.45 * np.clip(ratio / max(severe_ratio_thresh, 1e-6), 0.0, 1.0)
    score += 0.35 * np.clip((near_dist_thresh - nearest) / max(near_dist_thresh, 1e-6), 0.0, 1.0)
    score += 0.20 * np.clip(front_pts / max(min_front_points, 1), 0.0, 1.0)
    max_proj_overlap = max([float(inst.get("projected_overlap", 0.0)) for inst in blocking_instances], default=0.0)
    max_front_ratio = max([float(inst.get("front_ratio", 0.0)) for inst in blocking_instances], default=0.0)
    max_target_near_ratio = max([float(inst.get("target_near_ratio", 0.0)) for inst in blocking_instances], default=0.0)
    hidden_ratio = float(front_occ_info.get("target_amodal_occlusion_ratio", 0.0))
    hidden_pts = int(np.asarray(front_occ_info.get("target_hidden_points_w", np.zeros((0, 3), dtype=np.float32)), dtype=np.float32).reshape(-1, 3).shape[0])
    hidden_blocks = (
        hidden_ratio >= float(os.environ.get("FRONT_OCC_HIDDEN_CLEAR_RATIO_THRESH", "0.25"))
        and hidden_pts >= int(os.environ.get("FRONT_OCC_HIDDEN_CANDIDATE_MIN_POINTS", "30"))
        and front_occ_info.get("target_hidden_center_w", None) is not None
    )
    allow_raw_geometry_clear = _env_flag("FRONT_OCC_ALLOW_RAW_GEOMETRY_CLEAR", "0")

    geometry_blocks = (
        ratio >= float(os.environ.get("FRONT_OCC_GEOMETRY_RATIO_THRESH", "0.20"))
        and nearest <= near_dist_thresh
        and front_pts >= min_front_points
        and (hidden_blocks or allow_raw_geometry_clear)
    )

    if len(actionable_instances) == 0:
        should_clear = bool(geometry_blocks and _env_flag("FRONT_OCC_ALLOW_GEOMETRY_CLEAR", "1"))
        reason = "geometry_front_occluder_severe" if should_clear else "no_actionable_front_instance"
    elif len(strong_actionable_instances) == 0:
        should_clear = bool(geometry_blocks and _env_flag("FRONT_OCC_ALLOW_GEOMETRY_CLEAR", "1"))
        reason = "geometry_front_occluder_severe" if should_clear else "front_occluder_evidence_weak"
    elif len(blocking_instances) == 0:
        should_clear = False
        reason = "pointcloud_blocking_evidence_weak"
    elif only_geometry_fallback:
        should_clear = False
        reason = "geometry_only_front_occluder"
    else:
        should_clear = (
            (
                ratio >= severe_ratio_thresh
                and nearest <= near_dist_thresh
                and front_pts >= min_front_points
                and max_front_ratio >= float(min_cloud_front_ratio)
                and max_target_near_ratio >= float(min_target_near_ratio)
            )
            or (
                nearest <= 0.5 * near_dist_thresh
                and max_front_ratio >= max(0.12, 0.75 * float(min_cloud_front_ratio))
                and max_target_near_ratio >= max(0.06, 0.60 * float(min_target_near_ratio))
            )
        )
        reason = "front_occluder_severe" if should_clear else "front_occluder_mild"

    if debug:
        print(
            f"[FrontOcc] decision ratio={ratio:.4f} nearest={nearest:.4f} front_pts={front_pts} "
            f"actionable={len(actionable_instances)} strong={len(strong_actionable_instances)} "
            f"blocking={len(blocking_instances)} max_proj_overlap={max_proj_overlap:.4f} "
            f"max_front_ratio={max_front_ratio:.4f} max_target_near_ratio={max_target_near_ratio:.4f} "
            f"hidden_ratio={hidden_ratio:.4f} hidden_pts={hidden_pts} hidden_blocks={int(hidden_blocks)} "
            f"score={score:.4f} should_clear={should_clear}"
        )
        if len(front_instances) > 0:
            print("[FrontOcc] per-instance occlusion rates:")
            for inst in front_instances[:6]:
                print(
                    f"  id={int(inst.get('id', -1))} label={inst.get('label', 'unknown')} "
                    f"role={inst.get('role', 'unknown')} nearest={float(inst.get('nearest_dist', np.inf)):.4f} "
                    f"front_pts={int(inst.get('front_point_count', 0))} "
                    f"front_ratio={float(inst.get('front_ratio', 0.0)):.4f} "
                    f"target_near_ratio={float(inst.get('target_near_ratio', 0.0)):.4f} "
                    f"amodal_cov={float(inst.get('amodal_overlap_coverage', 0.0)):.4f} "
                    f"amodal_target_by_inst={float(inst.get('amodal_target_covered_by_inst', 0.0)):.4f} "
                    f"amodal_hidden_by_inst={float(inst.get('amodal_hidden_covered_by_inst', 0.0)):.4f} "
                    f"proj_overlap={float(inst.get('projected_overlap', 0.0)):.4f}"
                )

    return {
        "should_clear_first": bool(should_clear),
        "score": float(score),
        "reason": reason,
    }


def _select_front_occluder_candidate(front_occ_info):
    amodal_occluder_ids = {
        int(x) for x in front_occ_info.get("amodal_occluder_ids", [])
        if str(x).strip().lstrip("-").isdigit()
    }
    if len(amodal_occluder_ids) > 0:
        id_candidates = []
        for inst in front_occ_info.get("front_instances", []):
            inst_id = int(inst.get("id", -1))
            if inst_id not in amodal_occluder_ids:
                continue
            if inst.get("bbox_center", None) is None and inst.get("points_w", None) is None and inst.get("mask", None) is None:
                continue
            out = dict(inst)
            out["source"] = "amodal_occluder_id"
            out["selected_from_amodal_occluder_ids"] = True
            id_candidates.append(out)
        if len(id_candidates) > 0:
            def _amodal_id_priority(inst):
                return (
                    -float(inst.get("amodal_hidden_covered_by_inst", 0.0)),
                    -float(inst.get("amodal_overlap_coverage", 0.0)),
                    -float(inst.get("target_near_ratio", 0.0)),
                    -float(inst.get("score", 0.0)),
                    float(inst.get("nearest_dist", np.inf)),
                )
            chosen = sorted(id_candidates, key=_amodal_id_priority)[0]
            print(
                f"[FrontOcc][preview] use amodal occluder id binding "
                f"id={int(chosen.get('id', -1))} label={chosen.get('label', 'unknown')} "
                f"score={float(chosen.get('score', 0.0)):.3f} "
                f"amodal_hidden_by_inst={float(chosen.get('amodal_hidden_covered_by_inst', 0.0)):.4f}"
            )
            return chosen
    if _env_flag("FRONT_OCC_PREVIEW_USE_AMODAL_HIDDEN_DIRECTION", "1"):
        hidden_candidate = _make_hidden_front_occluder_candidate(front_occ_info)
        if hidden_candidate is not None:
            return hidden_candidate
    if _env_flag("FRONT_OCC_PREVIEW_USE_NEAREST_GEOMETRY", "1"):
        nearest_candidate = _make_unknown_front_occluder_candidate(front_occ_info, prefer_nearest=True)
        if nearest_candidate is not None:
            return nearest_candidate
    candidates = []
    for inst in front_occ_info.get("front_instances", []):
        label = str(inst.get("label", "unknown")).lower()
        if label == "geometry_fallback":
            continue
        if inst.get("bbox_center", None) is None and inst.get("points_w", None) is None and inst.get("mask", None) is None:
            continue
        candidates.append(inst)
    if len(candidates) == 0:
        return None

    def _priority(inst):
        role = str(inst.get("role", "unknown")).lower()
        clicked = 1 if inst.get("is_user_clicked", False) and role == "obstacle" else 0
        amodal_cov = float(inst.get("amodal_overlap_coverage", 0.0))
        target_near = float(inst.get("target_near_ratio", 0.0))
        front_ratio = float(inst.get("front_ratio", 0.0))
        score_sem = float(inst.get("score", 0.0))
        return (-clicked, -amodal_cov, -target_near, -front_ratio, -score_sem, float(inst.get("nearest_dist", np.inf)))

    return sorted(candidates, key=_priority)[0]


def _make_hidden_front_occluder_candidate(front_occ_info):
    hidden_center_w = front_occ_info.get("target_hidden_center_w", None)
    hidden_pts = np.asarray(front_occ_info.get("target_hidden_points_w", np.zeros((0, 3), dtype=np.float32)), dtype=np.float32).reshape(-1, 3)
    min_hidden_pts = int(os.environ.get("FRONT_OCC_HIDDEN_CANDIDATE_MIN_POINTS", "30"))
    if hidden_center_w is None and hidden_pts.shape[0] >= min_hidden_pts:
        hidden_center_w = np.median(hidden_pts, axis=0).astype(np.float32)
    if hidden_center_w is None or hidden_pts.shape[0] < min_hidden_pts:
        return None
    return {
        "id": -110,
        "label": "unknown_hidden_target_occluder",
        "role": "obstacle",
        "score": 0.0,
        "front_point_count": int(hidden_pts.shape[0]),
        "nearest_dist": float(front_occ_info.get("nearest_dist", np.inf)),
        "front_ratio": float(front_occ_info.get("target_amodal_occlusion_ratio", front_occ_info.get("occlusion_ratio", 0.0))),
        "target_near_ratio": float(front_occ_info.get("target_amodal_occlusion_ratio", 0.0)),
        "bbox_center": np.asarray(hidden_center_w, dtype=np.float32).reshape(3,),
        "points_w": hidden_pts.astype(np.float32),
        "mask": None,
        "is_user_clicked": False,
        "geometry_fallback": True,
        "source": "target_hidden_mask",
        "amodal_overlap_coverage": float(front_occ_info.get("target_amodal_occlusion_ratio", 0.0)),
    }


def _make_unknown_front_occluder_candidate(front_occ_info, prefer_nearest: bool = False):
    if prefer_nearest:
        front_center_w = front_occ_info.get("front_nearest_center_w", None)
        front_pts = np.asarray(front_occ_info.get("front_nearest_points_w", np.zeros((0, 3), dtype=np.float32)), dtype=np.float32).reshape(-1, 3)
    else:
        front_center_w = front_occ_info.get("front_center_w", None)
        front_pts = np.asarray(front_occ_info.get("front_points_w", np.zeros((0, 3), dtype=np.float32)), dtype=np.float32).reshape(-1, 3)
    if front_center_w is None and front_pts.shape[0] > 0:
        front_center_w = np.median(front_pts, axis=0).astype(np.float32)
    if front_center_w is None:
        return None
    return {
        "id": -100,
        "label": "unknown_nearest_front_occluder" if prefer_nearest else "unknown_front_occluder",
        "role": "obstacle",
        "score": 0.0,
        "front_point_count": int(front_occ_info.get("front_point_count", front_pts.shape[0])),
        "nearest_dist": float(front_occ_info.get("nearest_dist", np.inf)),
        "front_ratio": float(front_occ_info.get("occlusion_ratio", 0.0)),
        "target_near_ratio": 0.0,
        "bbox_center": np.asarray(front_center_w, dtype=np.float32).reshape(3,),
        "points_w": front_pts.astype(np.float32),
        "mask": None,
        "is_user_clicked": False,
        "geometry_fallback": True,
    }


def _select_detection_near_focus(context_dets, focus_uv, max_center_px: float = None, bbox_margin_px: float = None):
    if context_dets is None or len(context_dets) == 0 or focus_uv is None:
        return None
    u, v = int(focus_uv[0]), int(focus_uv[1])
    max_center_px = float(os.environ.get("FRONT_OCC_FOCUS_MAX_CENTER_PX", "140")) if max_center_px is None else float(max_center_px)
    bbox_margin_px = float(os.environ.get("FRONT_OCC_FOCUS_BBOX_MARGIN_PX", "25")) if bbox_margin_px is None else float(bbox_margin_px)
    ranked = []
    for det in context_dets:
        mask = det.get("mask", None)
        if mask is None:
            continue
        mask_arr = np.asarray(mask)
        hit = bool(0 <= v < mask_arr.shape[0] and 0 <= u < mask_arr.shape[1] and int(mask_arr[v, u]) > 0)
        xyxy = det.get("xyxy", None)
        bbox_dist = float("inf")
        inside_bbox = False
        if xyxy is not None:
            x1, y1, x2, y2 = np.asarray(xyxy, dtype=float).reshape(4,)
            inside_bbox = bool((x1 - bbox_margin_px) <= u <= (x2 + bbox_margin_px) and (y1 - bbox_margin_px) <= v <= (y2 + bbox_margin_px))
            dx = max(x1 - float(u), 0.0, float(u) - x2)
            dy = max(y1 - float(v), 0.0, float(v) - y2)
            bbox_dist = float(np.hypot(dx, dy))
        center = det.get("center", None)
        if center is None and xyxy is not None:
            x1, y1, x2, y2 = np.asarray(xyxy, dtype=float).reshape(4,)
            center = (0.5 * (x1 + x2), 0.5 * (y1 + y2))
        center_dist = float("inf")
        if center is not None:
            center_dist = float(np.hypot(float(u) - float(center[0]), float(v) - float(center[1])))
        area = int(np.count_nonzero(mask_arr))
        usable = bool(hit or inside_bbox or center_dist <= max_center_px or bbox_dist <= bbox_margin_px)
        if not usable:
            continue
        ranked.append((
            0 if hit else (1 if inside_bbox else 2),
            bbox_dist,
            center_dist,
            -float(det.get("score", 0.0)),
            -area,
            det,
        ))
    if len(ranked) == 0:
        return None
    ranked.sort(key=lambda x: x[:5])
    return ranked[0][-1]


def _compute_front_occluder_preview_focus(front_occ_info, occ_center_w):
    """Bias the re-observation focus beyond the hidden region toward the occluder body."""
    occ_center_w = np.asarray(occ_center_w, dtype=np.float32).reshape(3,)
    focus = occ_center_w.astype(np.float64).copy()

    front_dir = front_occ_info.get("front_dir_w", None)
    target_center = front_occ_info.get("target_center_w", None)
    if target_center is not None:
        target_center = np.asarray(target_center, dtype=np.float64).reshape(3,)
        dir_xy = focus[:2] - target_center[:2]
    elif front_dir is not None:
        front_dir = np.asarray(front_dir, dtype=np.float64).reshape(3,)
        dir_xy = front_dir[:2]
    else:
        dir_xy = np.zeros((2,), dtype=np.float64)
    nxy = float(np.linalg.norm(dir_xy))
    if nxy > 1e-6:
        dir_xy = dir_xy / nxy
        forward_offset = float(os.environ.get("FRONT_OCC_PREVIEW_FOCUS_FORWARD_OFFSET", "0.045"))
        focus[:2] += dir_xy * max(forward_offset, 0.0)
    else:
        forward_offset = 0.0

    up_offset = float(os.environ.get("FRONT_OCC_PREVIEW_FOCUS_UP_OFFSET", "0.020"))
    focus[2] += max(up_offset, 0.0)

    max_shift = float(os.environ.get("FRONT_OCC_PREVIEW_FOCUS_MAX_SHIFT", "0.080"))
    delta = focus - occ_center_w.astype(np.float64)
    shift = float(np.linalg.norm(delta))
    if shift > max_shift > 1e-6:
        focus = occ_center_w.astype(np.float64) + delta * (max_shift / shift)
        shift = max_shift
    return focus.astype(np.float32), {
        "forward_offset": float(forward_offset),
        "up_offset": float(up_offset),
        "shift": float(shift),
        "dir_xy": dir_xy.astype(np.float32),
    }


def preview_front_occluder_observation_for_clear_decision(env,
                                                         front_occ_info,
                                                         fovy_rad,
                                                         debug=False,
                                                         cam_name: str = EE_CAM_NAME,
                                                         debug_dir: str = None,
                                                         iter_tag: str = "clear_decision",
                                                         orbit_angle_deg: float = 30.0,
                                                         target_class: str = None):
    if not _env_flag("FRONT_OCC_PREVIEW_BEFORE_DECISION", "1"):
        return None
    chosen = _select_front_occluder_candidate(front_occ_info)
    if chosen is None:
        chosen = _make_unknown_front_occluder_candidate(front_occ_info)
    if chosen is None:
        print("[FrontOcc][preview] no front geometry for obstacle-centered re-observation.")
        return None

    occ_center_w = chosen.get("bbox_center", None)
    if occ_center_w is None:
        pts_w = np.asarray(chosen.get("points_w", np.zeros((0, 3), dtype=np.float32)), dtype=np.float32).reshape(-1, 3)
        if pts_w.shape[0] == 0:
            print("[FrontOcc][preview] selected obstacle has no usable geometry.")
            return None
        occ_center_w = np.median(pts_w, axis=0).astype(np.float32)
    occ_center_w = np.asarray(occ_center_w, dtype=np.float32).reshape(3,)
    print(
        f"[FrontOcc][preview] enter clear decision: re-observe obstacle-centered label={chosen.get('label', 'unknown')} "
        f"source={chosen.get('source', 'front_geometry')} "
        f"center_w={np.round(occ_center_w, 4)} front_ratio={float(chosen.get('front_ratio', 0.0)):.4f} "
        f"target_near_ratio={float(chosen.get('target_near_ratio', 0.0)):.4f} "
        f"amodal_cov={float(chosen.get('amodal_overlap_coverage', 0.0)):.4f}"
    )
    hidden_geometry_source = str(chosen.get("source", "")).lower() == "target_hidden_mask"
    allow_hidden_semantic_switch = _env_flag("FRONT_OCC_PREVIEW_ALLOW_HIDDEN_LABEL_SWITCH", "0")
    preview_focus_w, preview_focus_info = _compute_front_occluder_preview_focus(front_occ_info, occ_center_w)
    print(
        f"[FrontOcc][preview] shifted recognition focus_w={np.round(preview_focus_w, 4)} "
        f"from_center={np.round(occ_center_w, 4)} "
        f"dir_xy={np.round(preview_focus_info.get('dir_xy', np.zeros(2)), 4)} "
        f"forward={float(preview_focus_info.get('forward_offset', 0.0)):.3f}m "
        f"up={float(preview_focus_info.get('up_offset', 0.0)):.3f}m "
        f"shift={float(preview_focus_info.get('shift', 0.0)):.3f}m"
    )

    try:
        goto_object_centered_observation_pose(
            env,
            object_center_w=preview_focus_w,
            cam_name=cam_name,
            hover_height=float(os.environ.get("FRONT_OCC_PREVIEW_HOVER_HEIGHT", "0.25")),
            duration=0.8,
            settle_steps=50,
            debug=debug,
        )
        imgs_occ = render_from_camera(env, cam_name, depth_mode='auto', debug=debug)
        rgb_occ = imgs_occ["img"]
        depth_occ = imgs_occ["depth"]
        T_wc_occ = get_T_wc_from_camera(env, cam_name)
        bgr_occ = cv2.cvtColor(rgb_occ, cv2.COLOR_RGB2BGR)
        h_occ, w_occ = bgr_occ.shape[:2]
        K_occ = _build_cam_intrinsics_from_fovy(w_occ, h_occ, fovy_rad)
        camera_occ = CameraInfo(w_occ, h_occ, K_occ[0, 0], K_occ[1, 1], K_occ[0, 2], K_occ[1, 2], 1.0)
        uv_occ, valid_occ = _project_world_points_to_image(preview_focus_w[None, :], T_wc_occ, camera_occ)
        if not np.any(valid_occ):
            print("[FrontOcc][preview] shifted obstacle focus is not visible from obstacle-centered pose.")
            return None
        click_uv = tuple(int(round(v)) for v in uv_occ[0])
        print(f"[FrontOcc][preview] obstacle-centered focus recognition click_uv={click_uv}")

        def _geometry_mask_from_chosen_points():
            chosen_pts = np.asarray(chosen.get("points_w", np.zeros((0, 3), dtype=np.float32)), dtype=np.float32).reshape(-1, 3)
            min_geo_pts = int(os.environ.get("FRONT_OCC_PREVIEW_GEOMETRY_MIN_POINTS", "30"))
            if chosen_pts.shape[0] < min_geo_pts:
                return None, None
            uv_pts, valid_pts = _project_world_points_to_image(chosen_pts, T_wc_occ, camera_occ)
            if not np.any(valid_pts):
                return None, None
            uv_valid = np.round(uv_pts[valid_pts]).astype(np.int32)
            uv_valid[:, 0] = np.clip(uv_valid[:, 0], 0, w_occ - 1)
            uv_valid[:, 1] = np.clip(uv_valid[:, 1], 0, h_occ - 1)
            mask_geo = np.zeros((h_occ, w_occ), dtype=np.uint8)
            mask_geo[uv_valid[:, 1], uv_valid[:, 0]] = 255
            dilate_px = int(os.environ.get("FRONT_OCC_PREVIEW_GEOMETRY_MASK_DILATE", "7"))
            if dilate_px > 0:
                k = max(3, 2 * dilate_px + 1)
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
                mask_geo = cv2.dilate(mask_geo, kernel, iterations=1)
            seed = np.asarray(chosen.get("bbox_center", np.median(chosen_pts, axis=0)), dtype=np.float32).reshape(3,)
            inst_geo = _instance_from_points(chosen_pts, seed_w=seed, role="target")
            inst_geo["label"] = str(chosen.get("label", "unknown_front_occluder"))
            inst_geo["score"] = float(chosen.get("score", 0.0))
            inst_geo["mask"] = mask_geo
            inst_geo["xyxy"] = _mask_xyxy(mask_geo)
            inst_geo["selected_from_hidden_geometry"] = True
            return inst_geo, mask_geo

        preview_classes = _default_context_classes()
        target_label_for_preview = target_class or front_occ_info.get("target_label", None)
        if target_label_for_preview is not None and _env_flag("FRONT_OCC_PREVIEW_EXCLUDE_TARGET_LABEL", "1"):
            preview_classes = {c for c in preview_classes if not _same_label(c, target_label_for_preview)}
        chosen_label_for_preview = str(chosen.get("label", "") or "").strip()
        chosen_label_l = chosen_label_for_preview.lower()
        chosen_has_locked_semantic = bool(chosen_label_for_preview) and chosen_label_l not in {
            "unknown",
            "front_occluder",
            "unknown_front_occluder",
            "unknown_hidden_target_occluder",
            "target_hidden_mask",
            "geometry_fallback",
        }
        empty_target_mask = np.zeros((h_occ, w_occ), dtype=np.uint8)
        focus_dets_occ = detect_context_instances_center_view(
            image_or_path=bgr_occ,
            target_mask=empty_target_mask,
            conf_thresh=float(os.environ.get("FRONT_OCC_PREVIEW_RECOGNITION_CONF", "0.10")),
            context_classes=preview_classes,
            output_vis_path=None if debug_dir is None else _debug_iter_path(debug_dir, iter_tag, "preview_occluder_global_yolo_overlay"),
            exclude_target_overlap=1.01,
        )
        focus_det = _select_detection_near_focus(focus_dets_occ, click_uv)
        focus_recognized = False
        force_geometry_fallback = False
        if focus_det is not None:
            focus_label = str(focus_det.get("label", "unknown"))
            if target_label_for_preview is not None and _same_label(focus_label, target_label_for_preview):
                print(
                    f"[FrontOcc][preview] ignore focus recognition because it is current target "
                    f"label={focus_label}; use hidden/front geometry instead."
                )
                focus_det = None
                force_geometry_fallback = True
            elif chosen_has_locked_semantic and not _same_label(focus_label, chosen_label_for_preview):
                print(
                    f"[FrontOcc][preview] reject focus recognition label={focus_label} "
                    f"because clear decision is locked to label={chosen_label_for_preview}; "
                    f"use selected front occluder geometry instead."
                )
                focus_det = None
                force_geometry_fallback = True
        if focus_det is not None:
            occ_mask = np.asarray(focus_det.get("mask", None), dtype=np.uint8)
            occ_pts_w = mask_to_world_points(occ_mask, depth_occ, T_wc_occ, fovy_rad)
            min_focus_pts = int(os.environ.get("FRONT_OCC_PREVIEW_MIN_RECOGNIZED_POINTS", "80"))
            if occ_pts_w.shape[0] >= min_focus_pts:
                occ_center_from_mask = np.median(occ_pts_w, axis=0).astype(np.float32)
                focus_center_dist = float(np.linalg.norm(occ_center_from_mask.astype(np.float64) - preview_focus_w.astype(np.float64)))
                max_focus_center_dist = float(os.environ.get("FRONT_OCC_PREVIEW_MAX_FOCUS_CENTER_DIST", "0.045"))
                if focus_center_dist > max_focus_center_dist:
                    print(
                        f"[FrontOcc][preview] reject focus recognition label={focus_det.get('label', 'unknown')} "
                        f"because mask center is far from hidden/front focus dist={focus_center_dist:.4f}m "
                        f"max={max_focus_center_dist:.4f}m; use hidden/front geometry instead."
                    )
                    occ_inst = None
                    occ_mask = None
                    force_geometry_fallback = True
                    focus_det = None
                else:
                    occ_seed = occ_center_from_mask
                    focus_label = str(focus_det.get("label", "front_occluder"))
                    stable_label = focus_label
                    if hidden_geometry_source and not allow_hidden_semantic_switch:
                        stable_label = str(chosen.get("label", "unknown_hidden_target_occluder"))
                    occ_inst = _instance_from_points(occ_pts_w, seed_w=occ_seed, role="target")
                    occ_inst["label"] = stable_label
                    occ_inst["observed_label"] = focus_label
                    occ_inst["score"] = float(focus_det.get("score", 0.0))
                    occ_inst["mask"] = occ_mask.astype(np.uint8)
                    occ_inst["xyxy"] = focus_det.get("xyxy", _mask_xyxy(occ_mask))
                    occ_inst["selected_from_focus_recognition"] = True
                    occ_inst["focus_center_dist"] = focus_center_dist
                    chosen = dict(chosen)
                    chosen["observed_label"] = focus_label
                    if (not hidden_geometry_source) or allow_hidden_semantic_switch:
                        chosen["label"] = occ_inst["label"]
                    chosen["score"] = float(occ_inst.get("score", 0.0))
                    chosen["focus_center_dist"] = focus_center_dist
                    focus_recognized = True
                    print(
                        f"[FrontOcc][preview] focus recognized clearing target "
                        f"label={occ_inst['label']} observed_label={focus_label} "
                        f"score={float(occ_inst.get('score', 0.0)):.3f} "
                        f"points={int(occ_pts_w.shape[0])} center_dist={focus_center_dist:.4f}m"
                    )
            else:
                print(
                    f"[FrontOcc][preview] focus recognition label={focus_det.get('label', 'unknown')} "
                    f"has too few depth points {int(occ_pts_w.shape[0])} < {min_focus_pts}; fallback to focus click."
                )
                occ_inst = None
                occ_mask = None
        else:
            occ_inst = None
            occ_mask = None
            print("[FrontOcc][preview] no YOLO instance near hidden focus; fallback to focus click segmentation.")
        if occ_inst is None or occ_mask is None or int(np.count_nonzero(occ_mask)) == 0:
            geo_inst, geo_mask = _geometry_mask_from_chosen_points()
            if geo_inst is not None and geo_mask is not None and int(np.count_nonzero(geo_mask)) > 0:
                occ_inst = geo_inst
                occ_mask = geo_mask
                focus_recognized = False
                chosen_pts_count = int(np.asarray(chosen.get("points_w", np.zeros((0, 3), dtype=np.float32)), dtype=np.float32).reshape(-1, 3).shape[0])
                print(
                    f"[FrontOcc][preview] use hidden/front geometry as clearing target "
                    f"label={occ_inst.get('label', 'unknown')} points={chosen_pts_count}"
                )
            elif force_geometry_fallback:
                print("[FrontOcc][preview] hidden/front geometry is insufficient; no safe clearing target selected.")
                return None
            else:
                occ_inst = create_instance_from_click(bgr_occ, depth_occ, click_uv, camera_occ, T_wc_occ, role="target")
                occ_inst["label"] = str(chosen.get("label", "front_occluder"))
                occ_mask = occ_inst.get("mask", None)
                focus_recognized = False
                if occ_mask is None or int(np.count_nonzero(occ_mask)) == 0:
                    print("[FrontOcc][preview] obstacle-centered focus click produced empty mask.")
                    return None
        if debug_dir is not None:
            save_debug_image(bgr_occ, _debug_iter_path(debug_dir, iter_tag, "preview_occluder_center_rgb"), title=f"{iter_tag}: preview_occluder_rgb")
            cv2.imwrite(_debug_iter_path(debug_dir, iter_tag, "preview_occluder_target_mask"), np.asarray(occ_mask, dtype=np.uint8))

        context_dets_occ = detect_context_instances_center_view(
            image_or_path=bgr_occ,
            target_mask=occ_mask,
            conf_thresh=0.15,
            context_classes=_default_context_classes(),
            output_vis_path=None if debug_dir is None else _debug_iter_path(debug_dir, iter_tag, "preview_occluder_context_yolo_overlay"),
        )
        scene_context_occ = build_scene_context_from_center_view(
            bgr0=bgr_occ,
            depth0=depth_occ,
            target_mask=occ_mask,
            context_dets=context_dets_occ,
            T_wc0=T_wc_occ,
            fovy_rad=fovy_rad,
            debug=debug,
        )
        built_occ_instances, _ = build_scene_instances_from_view(occ_mask, context_dets_occ, depth_occ, T_wc_occ, fovy_rad)
        recognized_occ = _recognized_instance_at_click(click_uv, built_occ_instances)
        if recognized_occ is not None and not focus_recognized:
            rec_score = float(recognized_occ.get("score", 0.0))
            rec_min_score = float(os.environ.get("FRONT_OCC_PREVIEW_RECOGNIZED_MIN_SCORE", "0.25"))
            rec_label = str(recognized_occ.get("label", "unknown"))
            if target_label_for_preview is not None and _same_label(rec_label, target_label_for_preview):
                print(
                    f"[FrontOcc][preview] ignore recognized obstacle at click because it is current target "
                    f"label={rec_label}; keep geometry/focus target label={occ_inst.get('label', 'front_occluder')}"
                )
            elif chosen_has_locked_semantic and not _same_label(rec_label, chosen_label_for_preview):
                print(
                    f"[FrontOcc][preview] ignore recognized obstacle at click label={rec_label} "
                    f"because clear decision is locked to label={chosen_label_for_preview}; "
                    f"keep geometry/focus target label={occ_inst.get('label', 'front_occluder')}"
                )
            elif rec_score >= rec_min_score:
                occ_inst = dict(recognized_occ)
                occ_inst["role"] = "target"
                chosen = dict(chosen)
                if hidden_geometry_source and not allow_hidden_semantic_switch:
                    occ_inst["observed_label"] = rec_label
                    occ_inst["label"] = str(chosen.get("label", "unknown_hidden_target_occluder"))
                    chosen["observed_label"] = rec_label
                else:
                    chosen["label"] = occ_inst.get("label", chosen.get("label", "front_occluder"))
                chosen["score"] = rec_score
                print(
                    f"[FrontOcc][preview] recognized obstacle at click "
                    f"label={occ_inst.get('label', rec_label)} observed_label={rec_label} score={rec_score:.3f}"
                )
            elif chosen_has_locked_semantic and _same_label(rec_label, chosen_label_for_preview):
                occ_inst["observed_label"] = rec_label
                occ_inst["preview_recognition_score"] = rec_score
                print(
                    f"[FrontOcc][preview] keep FrontOcc/Amodal selected occluder despite low preview score "
                    f"label={chosen_label_for_preview} preview_score={rec_score:.3f} < {rec_min_score:.3f}; "
                    f"front_score={float(chosen.get('score', 0.0)):.3f} "
                    f"amodal_hidden_by_inst={float(chosen.get('amodal_hidden_covered_by_inst', 0.0)):.4f}"
                )
            else:
                print(
                    f"[FrontOcc][preview] ignore low-confidence recognized obstacle at click "
                    f"label={rec_label} score={rec_score:.3f} < {rec_min_score:.3f}; "
                    f"keep geometry target label={occ_inst.get('label', 'front_occluder')}"
                )
        clear_target_inst = _as_explicit_clear_target_instance(occ_inst, chosen=chosen, fallback_center_w=occ_center_w)
        occ_instances = _merge_scene_instances(built_occ_instances, [clear_target_inst])
        scene_context_occ = inject_scene_instances_into_context(scene_context_occ, [clear_target_inst])
        scene_context_occ["scene_instances"] = list(occ_instances)
        scene_context_occ["target_instance"] = clear_target_inst
        scene_context_occ["target_center_w"] = np.asarray(
            clear_target_inst.get("bbox_center", occ_center_w),
            dtype=np.float32,
        ).reshape(3,)
        print(
            f"[FrontOcc][preview] bind clear execution target "
            f"label={clear_target_inst.get('label', 'unknown')} "
            f"observed_label={clear_target_inst.get('observed_label', clear_target_inst.get('label', 'unknown'))} "
            f"center_w={np.round(scene_context_occ['target_center_w'], 4)} "
            f"source={clear_target_inst.get('front_occluder_source', chosen.get('source', 'front_geometry'))}"
        )
        _, cloud_occ_center = get_and_process_data(bgr_occ, depth_occ, occ_mask, fovy_rad=fovy_rad)
        q_occ = np.array(env.robot.get_joint(), dtype=float)
        occ_click_dirs, occ_occluders, _ = select_occlusion_aware_view_dirs(
            clear_target_inst,
            occ_instances,
            np.asarray(T_wc_occ.t, dtype=np.float32).reshape(3,),
            top_k=2,
            debug=debug,
            T_wc=T_wc_occ,
            camera=camera_occ,
            image_shape=bgr_occ.shape[:2],
        )
        fused_occ, occ_view_info = fuse_active_views_by_bbox(
            env,
            cam_name=cam_name,
            q_center=q_occ,
            T_wc_center=T_wc_occ,
            cloud_center_cam=cloud_occ_center,
            fovy_rad=fovy_rad,
            target_class=str(chosen.get("label", "front_occluder")),
            joint0_delta_rad=np.deg2rad(orbit_angle_deg),
            bbox_margin=0.03,
            max_dist_to_centroid=0.20,
            voxel_size=0.0010,
            icp_refine=False,
            nn_gate_dist=0.02,
            verify=True,
            save_debug_dir=(None if debug_dir is None else os.path.join(debug_dir, f"{iter_tag}_preview_occluder_multiview")),
            debug=debug,
            top_mask=occ_mask,
            top_depth=depth_occ,
            scene_context=scene_context_occ,
            manual_view_dirs_xy=occ_click_dirs,
            fusion_mode="balanced",
        )
        print(
            f"[FrontOcc][preview] obstacle multiview selected={occ_view_info.get('best_pair_name', 'unknown')} "
            f"occluders={len(occ_occluders)} center_pts={np.asarray(cloud_occ_center.points).shape[0]} "
            f"fused_pts={np.asarray(fused_occ.points).shape[0]}"
        )
        occ_pts_w = _cam_to_world_points(T_wc_occ, np.asarray(fused_occ.points, dtype=np.float32))
        occ_target = update_instance_with_new_view(clear_target_inst, occ_pts_w)
        occ_target = _as_explicit_clear_target_instance(occ_target, chosen=chosen, fallback_center_w=occ_center_w)
        scene_context_occ["target_instance"] = occ_target
        scene_context_occ["target_center_w"] = np.asarray(
            occ_target.get("bbox_center", scene_context_occ["target_center_w"]),
            dtype=np.float32,
        ).reshape(3,)
        scene_context_occ["scene_instances"] = _replace_scene_instance(list(occ_instances), occ_target)
        occ_grasp_cloud = _build_grasp_cloud_for_instance(occ_target, T_wc_occ, fallback_cloud=fused_occ)
        if occ_grasp_cloud is None:
            occ_grasp_cloud = cloud_occ_center
        occ_collision_cloud = _build_collision_cloud_for_grasp(scene_context_occ, T_wc_occ, target_cloud=occ_grasp_cloud)
        return {
            "chosen": chosen,
            "center_w": occ_center_w,
            "click_uv": click_uv,
            "bgr": bgr_occ,
            "depth": depth_occ,
            "mask": occ_mask,
            "T_wc": T_wc_occ,
            "camera": camera_occ,
            "scene_context": scene_context_occ,
            "target_instance": occ_target,
            "cloud_center": cloud_occ_center,
            "fused_cloud": fused_occ,
            "grasp_cloud": occ_grasp_cloud,
            "collision_cloud": occ_collision_cloud,
            "active_view_info": occ_view_info,
        }
    except Exception as exc:
        print(f"[FrontOcc][preview] obstacle-centered observation failed: {type(exc).__name__}: {exc}")
        return None


def execute_preview_front_occluder_grasp(env,
                                         preview_info,
                                         debug=False,
                                         visual=True,
                                         manual_select=True):
    if not isinstance(preview_info, dict):
        return {"executed": False, "reason": "missing_preview_info"}
    grasp_cloud = preview_info.get("grasp_cloud", None)
    if grasp_cloud is None:
        return {"executed": False, "reason": "empty_preview_grasp_cloud"}
    T_wc_occ = preview_info["T_wc"]
    scene_context_occ = preview_info["scene_context"]
    chosen = preview_info.get("chosen", {})
    target_inst = preview_info.get("target_instance", scene_context_occ.get("target_instance", {}))
    label = str(target_inst.get("label", chosen.get("label", "front_occluder")))
    center_for_log = np.asarray(
        target_inst.get("bbox_center", scene_context_occ.get("target_center_w", preview_info.get("center_w", np.zeros(3)))),
        dtype=float,
    ).reshape(3,)
    print(
        f"[FrontOcc][target-shift] clear target switched to observed obstacle "
        f"label={label} observed_label={target_inst.get('observed_label', label)} "
        f"center_w={np.round(center_for_log, 4)}"
    )
    try:
        gg_occ, gg_occ_src, _, _ = _choose_grasp_from_cloud_candidates(
            [
                ("preview_occluder_fused_cloud", preview_info.get("fused_cloud", None)),
                ("preview_occluder_grasp_cloud", grasp_cloud),
                ("preview_occluder_center_cloud", preview_info.get("cloud_center", None)),
            ],
            env=env,
            T_wc_cap=T_wc_occ,
            vertical_cam=np.asarray(T_wc_occ.R, dtype=float).T @ np.array([0.0, 0.0, 1.0], dtype=float),
            scene_context=scene_context_occ,
            target_center_w=scene_context_occ.get("target_center_w", preview_info.get("center_w", np.zeros(3))),
            display_bgr=preview_info["bgr"],
            display_camera=preview_info["camera"],
            display_cloud=preview_info.get("fused_cloud", grasp_cloud),
            collision_cloud=preview_info.get("collision_cloud", grasp_cloud),
            angle_deg=65.0,
            visual=visual,
            debug_label="FrontOccPreviewGrasp",
            manual_select=manual_select,
            manual_topk=8,
        )
        print(f"[FrontOcc][target-shift] grasp source={gg_occ_src}")
    except RuntimeError as exc:
        return {"executed": False, "reason": f"preview_occluder_grasp_generation_failed: {exc}"}
    try:
        with _temporary_env({
            "EXEC_ALLOW_SINGLE_SIDED_OBJECT_CONTACT": os.environ.get("CLEAR_EXEC_ALLOW_SINGLE_SIDED_CONTACT", "1"),
            "EXEC_SINGLE_SIDED_MIN_CONTACTS": os.environ.get("CLEAR_EXEC_SINGLE_SIDED_MIN_CONTACTS", "1"),
            "EXEC_SINGLE_SIDED_MIN_PENETRATION": os.environ.get("CLEAR_EXEC_SINGLE_SIDED_MIN_PENETRATION", "0.0"),
            "EXEC_CLEAR_GRASP_MODE": "1",
        }):
            ok = execute_grasp(env, gg_occ, T_wc_occ)
    except Exception as exc:
        return {"executed": False, "reason": f"preview_occluder_grasp_execution_failed: {type(exc).__name__}"}
    if not ok:
        return {"executed": False, "reason": "preview_occluder_grasp_execution_failed: ik_abort"}
    return {"executed": True, "reason": "preview_front_occluder_removed", "label": label}


def _estimate_preclear_direction_from_scene(scene_context):
    if scene_context is None:
        return None
    target_center = np.asarray(scene_context.get("target_center_w", np.zeros((3,), dtype=np.float32)), dtype=np.float32).reshape(3,)
    best_dir = None
    best_score = -np.inf
    for inst in scene_context.get("scene_instances", []):
        label = str(inst.get("label", "unknown")).lower()
        role = str(inst.get("role", "unknown")).lower()
        if label == "geometry_fallback":
            continue
        if role not in {"obstacle", "unknown"}:
            continue
        bbox_center = inst.get("bbox_center", None)
        if bbox_center is None:
            continue
        vec = np.asarray(bbox_center, dtype=np.float32).reshape(3,) - target_center
        vec[2] = 0.0
        nv = float(np.linalg.norm(vec))
        if nv < 1e-6:
            continue
        score = (
            1.2 * float(inst.get("projected_overlap", 0.0)) +
            0.8 * float(inst.get("is_user_clicked", False)) +
            0.6 * float(inst.get("front_ratio", 0.0)) +
            0.3 * float(inst.get("score", 0.0))
        )
        if score > best_score:
            best_score = score
            best_dir = (vec / nv).astype(np.float32)
    return best_dir


def attempt_clear_front_occluder(env,
                                 front_occ_info,
                                 center_bgr,
                                 center_depth,
                                 T_wc_center,
                                 fovy_rad,
                                 vertical_cam,
                                 target_center_w=None,
                                 front_dir_w=None,
                                 debug=False,
                                 cam_name: str = EE_CAM_NAME,
                                 debug_dir: str = None,
                                 iter_tag: str = "clear",
                                 orbit_angle_deg: float = 30.0):
    front_instances = list(front_occ_info.get("front_instances", []))
    if len(front_instances) == 0:
        return {"executed": False, "reason": "no_front_instance"}

    def _front_priority(inst):
        label = str(inst.get("label", "unknown")).lower()
        role = str(inst.get("role", "unknown")).lower()
        has_click = 1 if inst.get("is_user_clicked", False) and role == "obstacle" else 0
        non_geometry = 1 if label != "geometry_fallback" else 0
        has_bbox = 1 if inst.get("bbox_center", None) is not None else 0
        has_mask = 1 if inst.get("mask", None) is not None else 0
        target_near = float(inst.get("target_near_ratio", 0.0))
        front_ratio = float(inst.get("front_ratio", 0.0))
        score_sem = float(inst.get("score", 0.0))
        return (-has_click, -target_near, -front_ratio, -score_sem, -non_geometry, -has_bbox, -has_mask, float(inst.get("nearest_dist", np.inf)))

    actionable_fronts = []
    for inst in front_instances:
        label = str(inst.get("label", "unknown")).lower()
        if label == "geometry_fallback":
            continue
        if inst.get("bbox_center", None) is None and inst.get("points_w", None) is None and inst.get("mask", None) is None:
            continue
        is_clicked = bool(inst.get("is_user_clicked", False))
        score_sem = float(inst.get("score", 0.0))
        front_ratio_i = float(inst.get("front_ratio", 0.0))
        target_near_ratio_i = float(inst.get("target_near_ratio", 0.0))
        nearest_i = float(inst.get("nearest_dist", np.inf))
        if is_clicked:
            if not (nearest_i <= 0.08 and front_ratio_i >= 0.12 and target_near_ratio_i >= 0.05):
                continue
        else:
            if not (nearest_i <= 0.08 and score_sem >= 0.30 and front_ratio_i >= 0.18 and target_near_ratio_i >= 0.10):
                continue
        actionable_fronts.append(inst)
    actionable_fronts = sorted(actionable_fronts, key=_front_priority)
    chosen = actionable_fronts[0] if len(actionable_fronts) > 0 else None
    if chosen is None:
        return {"executed": False, "reason": "no_actionable_front_occluder"}

    occ_center_w = chosen.get("bbox_center", None)
    if occ_center_w is None:
        pts_w = np.asarray(chosen.get("points_w", np.zeros((0, 3), dtype=np.float32)), dtype=np.float32).reshape(-1, 3)
        if pts_w.shape[0] == 0:
            return {"executed": False, "reason": "front_occluder_missing_geometry"}
        occ_center_w = np.median(pts_w, axis=0).astype(np.float32)
    occ_center_w = np.asarray(occ_center_w, dtype=np.float32).reshape(3,)

    if debug:
        print(
            f"[FrontOcc] removing front occluder label={chosen['label']} "
            f"nearest={float(chosen.get('nearest_dist', np.inf)):.4f} "
            f"pts={int(chosen.get('front_point_count', 0))} center_w={np.round(occ_center_w, 4)}"
        )

    goto_object_centered_observation_pose(
        env,
        object_center_w=occ_center_w,
        cam_name=cam_name,
        hover_height=float(os.environ.get("FRONT_OCC_PREVIEW_HOVER_HEIGHT", "0.25")),
        duration=1.0,
        settle_steps=60,
        debug=debug,
    )
    imgs_occ = render_from_camera(env, cam_name, depth_mode='auto', debug=debug)
    rgb_occ = imgs_occ["img"]
    depth_occ = imgs_occ["depth"]
    T_wc_occ = get_T_wc_from_camera(env, cam_name)
    bgr_occ = cv2.cvtColor(rgb_occ, cv2.COLOR_RGB2BGR)
    h_occ, w_occ = bgr_occ.shape[:2]
    K_occ = _build_cam_intrinsics_from_fovy(w_occ, h_occ, fovy_rad)
    camera_occ = CameraInfo(w_occ, h_occ, K_occ[0, 0], K_occ[1, 1], K_occ[0, 2], K_occ[1, 2], 1.0)
    uv_occ, valid_occ = _project_world_points_to_image(occ_center_w[None, :], T_wc_occ, camera_occ)
    if not np.any(valid_occ):
        return {"executed": False, "reason": "occluder_center_not_visible"}
    click_uv = tuple(int(round(v)) for v in uv_occ[0])
    print(f"[FrontOcc] obstacle-centered click_uv={click_uv}")
    occ_inst = create_instance_from_click(bgr_occ, depth_occ, click_uv, camera_occ, T_wc_occ, role="target")

    if debug_dir is not None:
        save_debug_image(
            bgr_occ,
            _debug_iter_path(debug_dir, iter_tag, "occluder_center_rgb"),
            title=f"{iter_tag}: occluder_rgb",
        )
        save_yolo_detection_debug_overlay(
            bgr_occ,
            _debug_iter_path(debug_dir, iter_tag, "occluder_target_yolo_overlay"),
            target_class=str(chosen.get("label", "front_occluder")),
            conf_thresh=0.15,
            title=f"{iter_tag}: occluder_target_yolo",
        )

    occ_inst["label"] = str(chosen.get("label", "front_occluder"))
    occ_mask = occ_inst.get("mask", None)
    if occ_mask is None or int(np.count_nonzero(occ_mask)) == 0:
        return {"executed": False, "reason": "empty_occluder_mask"}
    if debug_dir is not None:
        cv2.imwrite(_debug_iter_path(debug_dir, iter_tag, "occluder_target_mask"), np.asarray(occ_mask, dtype=np.uint8))

    context_dets_occ = detect_context_instances_center_view(
        image_or_path=bgr_occ,
        target_mask=occ_mask,
        conf_thresh=0.15,
        context_classes=_default_context_classes(),
        output_vis_path=None if debug_dir is None else _debug_iter_path(debug_dir, iter_tag, "occluder_context_yolo_overlay"),
    )
    scene_context_occ = build_scene_context_from_center_view(
        bgr0=bgr_occ,
        depth0=depth_occ,
        target_mask=occ_mask,
        context_dets=context_dets_occ,
        T_wc0=T_wc_occ,
        fovy_rad=fovy_rad,
        debug=debug,
    )
    built_occ_instances, _ = build_scene_instances_from_view(occ_mask, context_dets_occ, depth_occ, T_wc_occ, fovy_rad)
    clear_target_inst = _as_explicit_clear_target_instance(occ_inst, chosen=chosen, fallback_center_w=occ_center_w)
    occ_instances = _merge_scene_instances(built_occ_instances, [clear_target_inst])
    scene_context_occ = inject_scene_instances_into_context(scene_context_occ, [clear_target_inst])
    scene_context_occ["scene_instances"] = list(occ_instances)
    scene_context_occ["target_instance"] = clear_target_inst
    scene_context_occ["target_center_w"] = np.asarray(
        clear_target_inst.get("bbox_center", occ_center_w),
        dtype=np.float32,
    ).reshape(3,)
    print(
        f"[FrontOcc] bind clear execution target label={clear_target_inst.get('label', 'unknown')} "
        f"center_w={np.round(scene_context_occ['target_center_w'], 4)} "
        f"source={clear_target_inst.get('front_occluder_source', chosen.get('source', 'front_geometry'))}"
    )
    if debug_dir is not None:
        save_scene_instance_debug_overlay(
            bgr_occ,
            occ_instances,
            T_wc_occ,
            camera_occ,
            _debug_iter_path(debug_dir, iter_tag, "occluder_scene_overlay"),
            title=f"{iter_tag}: occluder",
            image_shape=bgr_occ.shape[:2],
        )

    _, cloud_occ_center = get_and_process_data(bgr_occ, depth_occ, occ_mask, fovy_rad=fovy_rad)
    q_occ = np.array(env.robot.get_joint(), dtype=float)
    occ_click_dirs, occ_occluders, _ = select_occlusion_aware_view_dirs(
        clear_target_inst,
        occ_instances,
        np.asarray(T_wc_occ.t, dtype=np.float32).reshape(3,),
        top_k=2,
        debug=debug,
        T_wc=T_wc_occ,
        camera=camera_occ,
        image_shape=bgr_occ.shape[:2],
    )
    fused_occ, occ_view_info = fuse_active_views_by_bbox(
        env,
        cam_name=cam_name,
        q_center=q_occ,
        T_wc_center=T_wc_occ,
        cloud_center_cam=cloud_occ_center,
        fovy_rad=fovy_rad,
        target_class=str(chosen.get("label", "front_occluder")),
        joint0_delta_rad=np.deg2rad(orbit_angle_deg),
        bbox_margin=0.03,
        max_dist_to_centroid=0.20,
        voxel_size=0.0010,
        icp_refine=False,
        nn_gate_dist=0.02,
        verify=True,
        save_debug_dir=(None if debug_dir is None else os.path.join(debug_dir, f"{iter_tag}_occluder_multiview")),
        debug=debug,
        top_mask=occ_mask,
        top_depth=depth_occ,
        scene_context=scene_context_occ,
        manual_view_dirs_xy=occ_click_dirs,
        fusion_mode="balanced",
    )
    if debug:
        print(f"[FrontOcc] obstacle multiview selected={occ_view_info.get('best_pair_name', 'unknown')} occluders={len(occ_occluders)}")

    occ_pts_w = _cam_to_world_points(T_wc_occ, np.asarray(fused_occ.points, dtype=np.float32))
    occ_target = update_instance_with_new_view(clear_target_inst, occ_pts_w)
    occ_target = _as_explicit_clear_target_instance(occ_target, chosen=chosen, fallback_center_w=occ_center_w)
    scene_context_occ["target_instance"] = occ_target
    scene_context_occ["target_center_w"] = np.asarray(
        occ_target.get("bbox_center", scene_context_occ["target_center_w"]),
        dtype=np.float32,
    ).reshape(3,)
    scene_context_occ["scene_instances"] = _replace_scene_instance(list(occ_instances), occ_target)
    occ_cloud_grasp = _build_grasp_cloud_for_instance(occ_target, T_wc_occ, fallback_cloud=fused_occ)
    if occ_cloud_grasp is None:
        return {"executed": False, "reason": "empty_occluder_grasp_cloud"}
    vertical_occ = np.asarray(T_wc_occ.R, dtype=float).T @ np.array([0.0, 0.0, 1.0], dtype=float)
    try:
        gg_occ, gg_occ_src, _, _ = _choose_grasp_from_cloud_candidates(
            [
                ("occluder_fused_cloud", fused_occ),
                ("occluder_grasp_cloud", occ_cloud_grasp),
                ("occluder_center_cloud", cloud_occ_center),
            ],
            env=env,
            T_wc_cap=T_wc_occ,
            vertical_cam=vertical_occ,
            scene_context=scene_context_occ,
            target_center_w=scene_context_occ.get("target_center_w", occ_center_w),
            display_bgr=bgr_occ,
            display_camera=camera_occ,
            display_cloud=fused_occ,
            collision_cloud=_build_collision_cloud_for_grasp(scene_context_occ, T_wc_occ, target_cloud=occ_cloud_grasp),
            angle_deg=65.0,
            visual=os.environ.get("GRASP_VISUAL", "1").strip().lower() in {"1", "true", "yes", "on"},
            debug_label="FrontOccGrasp",
            manual_select=os.environ.get("GRASP_MANUAL_SELECT", "1").strip().lower() in {"1", "true", "yes", "on"},
            manual_topk=8,
        )
        if debug:
            print(f"[FrontOcc] grasp source={gg_occ_src}")
    except RuntimeError as e:
        return {"executed": False, "reason": f"occluder_grasp_generation_failed: {e}"}
    try:
        with _temporary_env({
            "EXEC_ALLOW_SINGLE_SIDED_OBJECT_CONTACT": os.environ.get("CLEAR_EXEC_ALLOW_SINGLE_SIDED_CONTACT", "1"),
            "EXEC_SINGLE_SIDED_MIN_CONTACTS": os.environ.get("CLEAR_EXEC_SINGLE_SIDED_MIN_CONTACTS", "1"),
            "EXEC_SINGLE_SIDED_MIN_PENETRATION": os.environ.get("CLEAR_EXEC_SINGLE_SIDED_MIN_PENETRATION", "0.0"),
            "EXEC_CLEAR_GRASP_MODE": "1",
        }):
            ok = execute_grasp(env, gg_occ, T_wc_occ)
    except Exception as e:
        return {"executed": False, "reason": f"occluder_grasp_execution_failed: {type(e).__name__}"}
    if not ok:
        return {"executed": False, "reason": "occluder_grasp_execution_failed: ik_abort"}
    return {
        "executed": True,
        "reason": "front_occluder_removed",
        "label": str(chosen.get("label", "unknown")),
        "click_uv": click_uv,
    }

def move_joint_smooth(env, q_tgt: np.ndarray, duration: float = 1.0, settle_steps: int = 120):
    """Smooth joint motion using existing Quintic planner; steps env for visibility."""
    robot = env.robot
    q0 = np.array(robot.get_joint(), dtype=float)
    q1 = np.array(q_tgt, dtype=float).copy()
    action = np.zeros(env.mj_model.nu)
    if env.mj_model.nu > robot.dof:
        action[robot.dof] = float(getattr(env, "_last_gripper_cmd", 0.0))
    if q0.shape == q1.shape and float(np.max(np.abs(q1 - q0))) < 1e-4:
        action[:len(q1)] = q1
        for _ in range(min(int(settle_steps), 2)):
            env.step(action)
        return np.array(robot.get_joint(), dtype=float)

    parameter = JointParameter(q0, q1)
    velocity_parameter = QuinticVelocityParameter(duration)
    traj_param = TrajectoryParameter(parameter, velocity_parameter)
    planner = TrajectoryPlanner(traj_param)

    time_step_num = round(duration / 0.002) + 1
    times = np.linspace(0.0, duration, time_step_num)
    for ti in times:
        if ti == 0.0:
            continue
        q = planner.interpolate(ti)
        robot.move_joint(q)
        action[:len(q)] = q
        env.step(action)

    action[:len(q)] = q1
    for _ in range(int(settle_steps)):
        env.step(action)

    return np.array(robot.get_joint(), dtype=float)


def _find_nero_joint_lift_candidate(env,
                                    min_lift: float = 0.04,
                                    desired_lift: float = 0.12,
                                    max_xy_drift: float = 0.16):
    """Find a Nero7 joint-space lift that keeps the current grasp closed."""
    robot = env.robot
    if getattr(robot, "dof", 0) != 7:
        return None
    state = _save_mujoco_robot_state(env)
    q0 = np.asarray(state["robot_q"], dtype=float).reshape(-1)[:robot.dof]
    q_lim_low = np.asarray(getattr(robot, "q_lim_low", np.full_like(q0, -np.inf)), dtype=float)[:robot.dof]
    q_lim_up = np.asarray(getattr(robot, "q_lim_up", np.full_like(q0, np.inf)), dtype=float)[:robot.dof]
    try:
        T0 = robot.get_cartesian()
        p0 = np.asarray(T0.t, dtype=float).reshape(3,)
        shoulder_offsets = [float(x) for x in os.environ.get(
            "NERO_LIFT_SHOULDER_OFFSETS",
            "0.12 0.22 0.34 -0.12 -0.22",
        ).replace(",", " ").split()]
        joint3_offsets = [float(x) for x in os.environ.get(
            "NERO_LIFT_JOINT3_OFFSETS",
            "0 -0.18 0.18 -0.35 0.35",
        ).replace(",", " ").split()]
        elbow_offsets = [float(x) for x in os.environ.get(
            "NERO_LIFT_ELBOW_OFFSETS",
            "-0.18 -0.32 0.18 0.32 0",
        ).replace(",", " ").split()]
        wrist_pitch_offsets = [float(x) for x in os.environ.get(
            "NERO_LIFT_WRIST_PITCH_OFFSETS",
            "0 0.18 -0.18 0.35 -0.35",
        ).replace(",", " ").split()]
        best = None
        checked = 0
        max_candidates = int(os.environ.get("NERO_LIFT_SCAN_MAX_CANDIDATES", "360"))
        for s, j3, e, wp in itertools.product(shoulder_offsets, joint3_offsets, elbow_offsets, wrist_pitch_offsets):
            q = q0.copy()
            q[1] = q0[1] + s
            q[2] = q0[2] + j3
            q[3] = q0[3] + e
            q[5] = q0[5] + wp
            q[6] = q0[6]
            q = np.clip(q, q_lim_low, q_lim_up)
            _sync_robot_joint_to_mujoco(env, q, update_ctrl=False)
            p = np.asarray(robot.get_cartesian().t, dtype=float).reshape(3,)
            lift = float(p[2] - p0[2])
            xy = float(np.linalg.norm(p[:2] - p0[:2]))
            jump = float(np.max(np.abs(q - q0)))
            checked += 1
            if lift >= float(min_lift) and xy <= float(max_xy_drift):
                score = abs(lift - float(desired_lift)) + 0.9 * xy + 0.10 * jump
                rec = {"q": q.copy(), "lift": lift, "xy_drift": xy, "q_jump": jump, "score": score, "checked": checked}
                if best is None or rec["score"] < best["score"]:
                    best = rec
            if checked >= max_candidates:
                break
        return best
    finally:
        _restore_mujoco_robot_state(env, state)


def _move_gripper_smooth(env, action: np.ndarray, target: float, step_delta: float = 2.0, max_steps: int = 1000):
    """Move gripper command channel to target smoothly while preserving arm command."""
    action = np.asarray(action, dtype=float)
    target = float(np.clip(target, 0.0, 255.0))
    min_step_delta = float(os.environ.get("EXEC_GRIPPER_MIN_STEP_DELTA", "2.0"))
    step_delta = max(abs(float(step_delta)), max(min_step_delta, 1e-6))
    settle_steps = max(0, int(os.environ.get("EXEC_GRIPPER_SETTLE_STEPS", "40")))
    if step_delta < 1e-6:
        action[env.robot.dof] = target
        env._last_gripper_cmd = float(target)
        env.step(action)
        return action

    current = float(action[env.robot.dof])
    for _ in range(int(max_steps)):
        err = target - current
        if abs(err) <= step_delta:
            current = target
        else:
            current += step_delta if err > 0.0 else -step_delta
        action[env.robot.dof] = current
        env._last_gripper_cmd = float(current)
        env.step(action)
        if current == target:
            break
    action[env.robot.dof] = target
    env._last_gripper_cmd = float(target)
    for _ in range(settle_steps):
        env.step(action)
    return action


def _env_flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() not in {"0", "false", "no", "off", ""}


def _geom_name(model, geom_id: int) -> str:
    if geom_id < 0:
        return ""
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(geom_id))
    return "" if name is None else str(name)


def _gripper_finger_side(name: str):
    low = str(name).lower()
    if "nero_gripper_link1" in low:
        return "left"
    if "nero_gripper_link2" in low:
        return "right"
    return None


def _is_gripper_contact_geom(name: str) -> bool:
    if _env_flag("EXEC_GRASP_CONTACT_TIP_ONLY", "0"):
        return _gripper_finger_side(name) is not None and "tip_collision" in str(name).lower()
    return _gripper_finger_side(name) is not None


def _is_scene_object_contact_geom(name: str) -> bool:
    low = str(name).lower()
    if not low:
        return False
    if low.startswith("nero_") or "gripper" in low or "camera" in low:
        return False
    if low in {"floor", "mocap"} or "table" in low or "zone_" in low:
        return False
    return True


def _contact_object_key(name: str) -> str:
    low = str(name or "").strip().lower()
    if not low:
        return ""
    for sep in ("_pt", "-pt"):
        if sep in low:
            head, tail = low.rsplit(sep, 1)
            if tail.isdigit() and head:
                return head
    return low


def _summarize_contacts(env, max_rows: int = 12):
    rows = []
    model, data = env.mj_model, env.mj_data
    max_contact_dist = float(os.environ.get("NERO_GRASP_CONTACT_MAX_DIST", "0.0008"))
    for i in range(int(data.ncon)):
        c = data.contact[i]
        g1 = _geom_name(model, c.geom1)
        g2 = _geom_name(model, c.geom2)
        side1 = _gripper_finger_side(g1) if _is_gripper_contact_geom(g1) else None
        side2 = _gripper_finger_side(g2) if _is_gripper_contact_geom(g2) else None
        valid_dist = float(c.dist) <= max_contact_dist
        rows.append({
            "g1": g1,
            "g2": g2,
            "dist": float(c.dist),
            "side": side1 or side2,
            "object": g2 if side1 is not None else (g1 if side2 is not None else ""),
            "object_key": _contact_object_key(g2 if side1 is not None else (g1 if side2 is not None else "")),
            "gripper_object": (
                valid_dist
                and (
                    (side1 is not None and _is_scene_object_contact_geom(g2))
                    or (side2 is not None and _is_scene_object_contact_geom(g1))
                )
            ),
        })
    rows.sort(key=lambda r: (not r["gripper_object"], abs(float(r["dist"]))))
    return rows[:int(max_rows)]


def _has_gripper_object_contact(env):
    rows = _summarize_contacts(env, max_rows=24)
    left_objects = {str(r.get("object_key", r["object"])) for r in rows if r["gripper_object"] and r["side"] == "left"}
    right_objects = {str(r.get("object_key", r["object"])) for r in rows if r["gripper_object"] and r["side"] == "right"}
    shared_objects = left_objects & right_objects
    require_two_sided = _env_flag("EXEC_REQUIRE_TWO_SIDED_GRASP_CONTACT", "1")
    if require_two_sided:
        if _env_flag("EXEC_ALLOW_SINGLE_SIDED_OBJECT_CONTACT", "0"):
            valid_rows = [r for r in rows if r["gripper_object"] and r["side"] in {"left", "right"}]
            min_contacts = int(os.environ.get("EXEC_SINGLE_SIDED_MIN_CONTACTS", "1"))
            min_penetration = float(os.environ.get("EXEC_SINGLE_SIDED_MIN_PENETRATION", "0.0"))
            has_stable_single_side = (
                len(valid_rows) >= min_contacts
                and any(float(r.get("dist", 0.0)) <= -abs(min_penetration) for r in valid_rows)
            )
            if has_stable_single_side:
                return True, rows
        return bool(shared_objects), rows
    return bool(left_objects or right_objects), rows


def _close_gripper_until_grasp_contact(env,
                                       action: np.ndarray,
                                       step_delta: float = 2.0,
                                       max_steps: int = 220):
    action = np.asarray(action, dtype=float)
    current = float(action[env.robot.dof])
    min_step_delta = float(os.environ.get("EXEC_GRIPPER_MIN_STEP_DELTA", "2.0"))
    step_delta = max(abs(float(step_delta)), max(min_step_delta, 1e-6))
    check_every = max(1, int(os.environ.get("EXEC_GRASP_CONTACT_CHECK_EVERY", "3")))
    stop_on_contact = _env_flag("EXEC_STOP_CLOSE_ON_CONTACT", "0")
    settle_steps = max(0, int(os.environ.get("EXEC_GRIPPER_CLOSE_SETTLE_STEPS", os.environ.get("EXEC_GRIPPER_SETTLE_STEPS", "100"))))
    seen_contact_ok = False
    seen_rows = []
    for step_i in range(int(max_steps)):
        current = min(255.0, current + step_delta)
        action[env.robot.dof] = current
        env._last_gripper_cmd = float(current)
        env.step(action)
        if (step_i % check_every) != 0 and current < 255.0:
            continue
        contact_ok, rows = _has_gripper_object_contact(env)
        if contact_ok:
            seen_contact_ok = True
            seen_rows = rows
            if stop_on_contact:
                return action, True, rows, "contact_seen_stop_on_contact"
        if current >= 255.0:
            break
    action[env.robot.dof] = 255.0
    env._last_gripper_cmd = 255.0
    for _ in range(settle_steps):
        env.step(action)
    contact_ok, rows = _has_gripper_object_contact(env)
    if contact_ok:
        return action, True, rows, "fully_closed_with_contact"
    if seen_contact_ok:
        return action, True, seen_rows, "contact_seen_during_close"
    return action, False, rows, "fully_closed_no_contact"


def _gripper_joint_positions(env):
    vals = []
    for joint_name in ("gripper_joint1", "gripper_joint2"):
        try:
            jid = mujoco.mj_name2id(env.mj_model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if jid < 0:
                vals.append(float("nan"))
                continue
            qadr = int(env.mj_model.jnt_qposadr[jid])
            vals.append(float(env.mj_data.qpos[qadr]))
        except Exception:
            vals.append(float("nan"))
    return vals


def fuse_three_views_by_bbox(env,
                            cam_name: str,
                            q_center: np.ndarray,
                            T_wc_center: sm.SE3,
                            cloud_center_cam: o3d.geometry.PointCloud,
                            fovy_rad: float,
                            target_class: str = None,
                            joint0_delta_rad: float = np.deg2rad(12.0),
                            base_yaw_idx: int = 0,
                            wrist_yaw_idx: int = 4,
                            enable_wrist_comp: bool = True,
                            bbox_margin: float = 0.02,
                            max_dist_to_centroid: float = 0.12,
                            voxel_size: float = 0.0010,
                            icp_refine: bool = False,
                            icp_voxel: float = 0.003,
                            icp_thr_list=(0.06, 0.04, 0.025, 0.015),
                            icp_max_iter: int = 40,
                            nn_gate_dist: float = 0.012,
                            verify: bool = True,
                            save_debug_dir: str = None,
                            debug: bool = True,
                            top_mask=None,
                            top_depth=None,
                            orbit_horizontal_dir_xy: np.ndarray = None,
                            custom_view_dirs_xy: list = None,
                            custom_view_records: list = None,
                            target_instance=None,
                            scene_context=None,
                            fusion_mode: str = "balanced"):
    """Fuse center view with two strict circular-orbit side views.

    Key change:
      - Side views are generated on a circle centered at the object center.
      - Side-view target extraction is instance-aware (projected bbox + 3D OBB),
        not just a loose global crop.
      - Overlap points are used for alignment / verification.
      - Novel points are used to extend the fused surface in balanced mode.
      - Conservative mode only appends overlap-consistent side points.
    """
    robot = env.robot
    fusion_mode = str(fusion_mode or "balanced").strip().lower()
    allow_joint_fallback = os.environ.get("MV_ALLOW_JOINT_FALLBACK", "").strip().lower() in {"1", "true", "yes", "on"}
    use_joint_fallback = bool(robot.dof == 6 or allow_joint_fallback)

    if save_debug_dir is not None:
        os.makedirs(save_debug_dir, exist_ok=True)

    # -------- Center cloud in camera/world --------
    pts0_c = np.asarray(cloud_center_cam.points, dtype=np.float32)
    col0_c = np.asarray(cloud_center_cam.colors, dtype=np.float32) if cloud_center_cam.has_colors() else None
    if pts0_c.shape[0] == 0:
        raise RuntimeError("[fuse_three_views_by_bbox] center cloud is empty.")

    pts0_w = _cam_to_world_points(T_wc_center, pts0_c)
    if target_instance is not None:
        center = np.asarray(target_instance.get("bbox_center", np.median(pts0_w, axis=0)), dtype=np.float32).reshape(3,)
        bbox_pts_w = _bbox_corners_world(target_instance)
        bmin = bbox_pts_w.min(axis=0) - float(bbox_margin)
        bmax = bbox_pts_w.max(axis=0) + float(bbox_margin)
    else:
        center = np.median(pts0_w, axis=0).astype(np.float32)
        bmin = pts0_w.min(axis=0) - float(bbox_margin)
        bmax = pts0_w.max(axis=0) + float(bbox_margin)

    # Optional downsample center
    if voxel_size is not None and voxel_size > 0:
        pcd0 = o3d.geometry.PointCloud()
        pcd0.points = o3d.utility.Vector3dVector(pts0_c.astype(np.float32))
        if col0_c is not None:
            pcd0.colors = o3d.utility.Vector3dVector(col0_c.astype(np.float32))
        pcd0 = pcd0.voxel_down_sample(voxel_size=float(voxel_size))
        pts0_c_ds = np.asarray(pcd0.points, dtype=np.float32)
        col0_c_ds = np.asarray(pcd0.colors, dtype=np.float32) if pcd0.has_colors() else None
    else:
        pts0_c_ds, col0_c_ds = pts0_c, col0_c

    cam_center_w = np.asarray(T_wc_center.t, dtype=np.float64).reshape(3,)
    orbit_radius = float(np.linalg.norm(cam_center_w - center.astype(np.float64)))

    if debug:
        print(f"[MV] center cloud: raw={pts0_c.shape[0]}, ds={pts0_c_ds.shape[0]}")
        print(f"[MV] world bbox min={np.round(bmin,3)} max={np.round(bmax,3)}")
        print(f"[MV] center(world)={np.round(center,3)}")
        print(f"[MV] circular orbit radius={orbit_radius:.4f}")
        print(f"[MV] bbox_margin={float(bbox_margin):.4f} max_dist={float(max_dist_to_centroid):.4f} nn_gate={float(nn_gate_dist):.4f}")
        print(f"[MV] fusion_mode={fusion_mode} icp_refine={bool(icp_refine)}")

    if fusion_mode in ("center_only", "no_side", "none"):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts0_c_ds.astype(np.float32))
        if col0_c_ds is not None:
            pcd.colors = o3d.utility.Vector3dVector(col0_c_ds.astype(np.float32))
        if save_debug_dir is not None:
            o3d.io.write_point_cloud(os.path.join(save_debug_dir, "fused_center_cam.ply"), pcd)
        if debug:
            print(f"[MV] center_only: skip side-view fusion, fused cloud points = {pts0_c_ds.shape[0]}")
        return pcd

    if orbit_horizontal_dir_xy is None:
        # fallback: use current camera ray horizontal projection
        v0 = cam_center_w - center.astype(np.float64)
        orbit_horizontal_dir_xy = np.array([v0[0], v0[1]], dtype=np.float64)
        n = np.linalg.norm(orbit_horizontal_dir_xy)
        if n < 1e-9:
            orbit_horizontal_dir_xy = np.array([1.0, 0.0], dtype=np.float64)
        else:
            orbit_horizontal_dir_xy = orbit_horizontal_dir_xy / n

    if debug:
        print(f"[MV] orbit_horizontal_dir_xy={np.round(np.asarray(orbit_horizontal_dir_xy),4)}")

    # -------- Helper to capture one side on strict circular orbit --------
    def _actual_view_dir_xy_from_pose(T_wc_actual):
        cam_pos = np.asarray(T_wc_actual.t, dtype=float).reshape(3,)
        return _normalize_dir_xy(cam_pos[:2] - center.astype(np.float64)[:2])

    def _empty_side_data():
        empty = np.zeros((0, 3), dtype=np.float32)
        return {
            "all_pts": empty,
            "all_cols": None,
            "overlap_pts": empty,
            "overlap_cols": None,
            "novel_pts": empty,
            "novel_cols": None,
        }

    def _capture_side_circle(side_name: str,
                             side_sign: float = None,
                             view_dir_xy: np.ndarray = None,
                             view_record: dict = None):
        q_before_side = np.array(robot.get_joint(), dtype=float)
        if view_dir_xy is None:
            dir_xy_local = np.asarray(orbit_horizontal_dir_xy, dtype=np.float64)
            pose_side_sign = float(side_sign)
        else:
            dir_xy_local = np.asarray(view_dir_xy, dtype=np.float64).reshape(2,)
            pose_side_sign = 1.0
        orbit_angle_exec = float(joint0_delta_rad)
        if view_record is not None:
            orbit_angle_exec = float(view_record.get("orbit_angle_rad", orbit_angle_exec))
        detail_record_pre = view_record.get("detail", {}) if isinstance(view_record, dict) else {}
        pre_joint_scan = bool(
            detail_record_pre.get("nbv_joint_scan", False)
            or str(detail_record_pre.get("nbv_reachability_reason", "")) == "joint_scan"
        )

        # 1) build target camera pose on circular orbit
        T_wc_side = _make_circle_orbit_pose(
            object_center_w=center.astype(np.float64),
            T_wc_center=T_wc_center,
            orbit_horizontal_dir_xy=dir_xy_local,
            side_sign=float(pose_side_sign),
            orbit_angle_rad=float(orbit_angle_exec),
            min_radius=float(os.environ.get("ACTIVE_NBV_MIN_VIEW_RADIUS", "0.14")),
            target_radius=float(os.environ.get("ACTIVE_NBV_VIEW_RADIUS", "0.18")),
        )
        if pre_joint_scan and _env_flag("ACTIVE_NBV_USE_REACHABLE_SCAN_POSE_AS_TARGET", "1"):
            scan_t = detail_record_pre.get("nbv_scan_cam_pos_w", None)
            scan_R = detail_record_pre.get("nbv_scan_R_wc", None)
            if scan_t is not None and scan_R is not None:
                scan_t = np.asarray(scan_t, dtype=float).reshape(3,)
                scan_R = _orthonormalize_rotation_matrix(scan_R, fallback=None)
                if np.isfinite(scan_t).all() and scan_R is not None:
                    try:
                        T_wc_side = sm.SE3.Rt(scan_R, scan_t)
                        dir_xy_local = _actual_view_dir_xy_from_pose(T_wc_side)
                        print(
                            f"[ActiveNBV][exec][{side_name}] use joint-scan reachable pose as calibrated target; "
                            f"scan_center_norm={float(detail_record_pre.get('nbv_scan_center_norm', np.inf)):.3f}"
                        )
                    except ValueError as exc:
                        print(
                            f"[ActiveNBV][exec][{side_name}] ignore invalid joint-scan pose rotation: {exc}; "
                            "fallback to planned circular pose."
                        )
        desired_cam_pos = np.asarray(T_wc_side.t, dtype=float).reshape(3,)
        view_angle_deg = float(np.rad2deg(np.arctan2(dir_xy_local[1], dir_xy_local[0])))
        print(
            f"[ActiveNBV][exec][{side_name}] target_cam_pos_w={np.round(desired_cam_pos, 4)} "
            f"dir_xy={np.round(dir_xy_local, 4)} dir_angle={view_angle_deg:.1f}deg "
            f"orbit_step={float(np.rad2deg(orbit_angle_exec)):.1f}deg"
        )

        moved_ok = True
        try:
            q_hint = None
            q_hint_joint_scan = False
            detail_record = {}
            if view_record is not None:
                detail_record = view_record.get("detail", {})
                q_hint = detail_record.get("nbv_q", None)
                q_hint_joint_scan = bool(detail_record.get("nbv_joint_scan", False) or str(detail_record.get("nbv_reachability_reason", "")) == "joint_scan")
            if q_hint is not None:
                q_hint = np.asarray(q_hint, dtype=float).reshape(-1)
            if q_hint is not None and q_hint.size == getattr(robot, "dof", 0) and np.isfinite(q_hint).all():
                print(
                    f"[ActiveNBV][exec][{side_name}] TRY: "
                    f"{'joint-scan feasible target' if q_hint_joint_scan else 'verified joint target from NBV reachability check'}."
                )
                move_joint_smooth(
                    env,
                    q_hint,
                    duration=float(os.environ.get("ACTIVE_NBV_DIRECT_DURATION", "0.32")),
                    settle_steps=int(os.environ.get("ACTIVE_NBV_DIRECT_SETTLE_STEPS", "1")),
                )
                T_wc_act = get_T_wc_from_camera(env, cam_name)
                pos_err, ang_err = _pose_error_metrics(T_wc_side, T_wc_act)
                move_info = {
                    "T_wc_actual": T_wc_act,
                    "pos_err": float(pos_err),
                    "ang_err_deg": float(ang_err),
                    "iters": 0,
                    "converged": bool(pos_err <= 0.035 or q_hint_joint_scan),
                    "joint_scan": bool(q_hint_joint_scan),
                }
            else:
                print(f"[ActiveNBV][exec][{side_name}] TRY: servo camera pose.")
                move_info = _move_camera_pose_closed_loop(
                    env,
                    cam_name=cam_name,
                    T_wc_des=T_wc_side,
                    duration=float(os.environ.get("ACTIVE_NBV_MOVE_DURATION", "0.35")),
                    settle_steps=int(os.environ.get("ACTIVE_NBV_SETTLE_STEPS", "1")),
                    pos_tol=0.004,
                    ang_tol_deg=2.0,
                    max_iters=int(os.environ.get("ACTIVE_NBV_MAX_ITERS", "1")),
                    debug=debug,
                )
            T_wc_act = move_info["T_wc_actual"]
            exec_pos_tol = float(os.environ.get("ACTIVE_NBV_EXEC_MAX_POS_ERR", "0.035"))
            if bool(move_info.get("joint_scan", False)):
                metrics = _target_camera_metrics(T_wc_act, center.astype(np.float64))
                min_exec_dist = float(os.environ.get("ACTIVE_NBV_EXEC_MIN_TARGET_DIST", "0.160"))
                min_exec_height = float(os.environ.get("ACTIVE_NBV_EXEC_MIN_TARGET_HEIGHT", "0.110"))
                if not bool(metrics["visible"]):
                    moved_ok = False
                    print(
                        f"[ActiveNBV][exec][{side_name}] FAILED: joint-scan pose does not see target; "
                        f"center_norm={metrics['center_norm']:.3f} depth={metrics['depth']:.3f}"
                    )
                elif float(metrics["distance"]) < min_exec_dist or float(metrics["height"]) < min_exec_height:
                    moved_ok = False
                    print(
                        f"[ActiveNBV][exec][{side_name}] FAILED: joint-scan pose too close to target; "
                        f"dist={metrics['distance']:.3f}m height={metrics['height']:.3f}m "
                        f"min_dist={min_exec_dist:.3f}m min_height={min_exec_height:.3f}m"
                    )
                else:
                    actual_lateral = float(np.linalg.norm(np.asarray(T_wc_act.t, dtype=float).reshape(3,)[:2] - center.astype(np.float64)[:2]))
                    actual_side_deg = float(np.rad2deg(np.arctan2(actual_lateral, max(float(metrics["height"]), 1e-6))))
                    max_side_deg = float(os.environ.get("ACTIVE_NBV_EXEC_MAX_SIDE_ANGLE_DEG", "45.0"))
                    actual_dir_xy = _actual_view_dir_xy_from_pose(T_wc_act)
                    actual_dir_ang = float(np.rad2deg(_angle_from_dir_xy(actual_dir_xy)))
                    desired_dir_err = _smallest_angle_diff_deg(_angle_from_dir_xy(actual_dir_xy), _angle_from_dir_xy(dir_xy_local))
                    print(
                        f"[ActiveNBV][exec][{side_name}] joint-scan view accepted: "
                        f"center_norm={metrics['center_norm']:.3f} dist={metrics['distance']:.3f}m "
                        f"height={metrics['height']:.3f}m "
                        f"lateral={actual_lateral:.3f}m side_angle={actual_side_deg:.1f}deg "
                        f"actual_dir={actual_dir_ang:.1f}deg dir_err={desired_dir_err:.1f}deg "
                        f"near={not bool(detail_record.get('nbv_scan_near_fallback', False))}"
                    )
                    max_dir_err = float(os.environ.get("ACTIVE_NBV_JOINT_SCAN_MAX_DIR_ERR_DEG", "45.0"))
                    if actual_side_deg > max_side_deg:
                        moved_ok = False
                        print(
                            f"[ActiveNBV][exec][{side_name}] FAILED: joint-scan side angle too large "
                            f"{actual_side_deg:.1f}deg > {max_side_deg:.1f}deg."
                        )
                    elif desired_dir_err > max_dir_err:
                        moved_ok = False
                        print(
                            f"[ActiveNBV][exec][{side_name}] FAILED: joint-scan actual direction differs from requested view "
                            f"by {desired_dir_err:.1f}deg > {max_dir_err:.1f}deg."
                        )
                    if moved_ok and _env_flag("ACTIVE_NBV_USE_REACHABLE_SCAN_POSE_AS_TARGET", "1"):
                        T_wc_side = T_wc_act
                        desired_cam_pos = np.asarray(T_wc_side.t, dtype=float).reshape(3,)
                        dir_xy_local = actual_dir_xy.astype(np.float64)
                        move_info["pos_err"] = 0.0
                        print(
                            f"[ActiveNBV][exec][{side_name}] calibrated planned pose to reachable joint-scan pose "
                            f"target_cam_pos_w={np.round(desired_cam_pos, 4)}"
                        )
            elif float(move_info.get("pos_err", np.inf)) > exec_pos_tol:
                recovered_by_scan = False
                if (
                    not bool(q_hint_joint_scan)
                    and _env_flag("ACTIVE_NBV_RECOVER_WITH_JOINT_SCAN_ON_EXEC_FAIL", "1")
                    and getattr(robot, "dof", 0) == 7
                ):
                    scan = _find_feasible_object_view_joint_scan(
                        env,
                        cam_name=cam_name,
                        target_w=center.astype(np.float64),
                        desired_dir_xy=dir_xy_local,
                    )
                    if scan is not None:
                        print(f"[ActiveNBV][exec][{side_name}] RECOVER: exact target missed; use joint-scan target for same view direction.")
                        move_joint_smooth(
                            env,
                            np.asarray(scan["q"], dtype=float).reshape(-1),
                            duration=float(os.environ.get("ACTIVE_NBV_DIRECT_DURATION", "0.32")),
                            settle_steps=int(os.environ.get("ACTIVE_NBV_DIRECT_SETTLE_STEPS", "1")),
                        )
                        T_wc_act = get_T_wc_from_camera(env, cam_name)
                        metrics = _target_camera_metrics(T_wc_act, center.astype(np.float64))
                        if bool(metrics["visible"]):
                            min_exec_dist = float(os.environ.get("ACTIVE_NBV_EXEC_MIN_TARGET_DIST", "0.160"))
                            min_exec_height = float(os.environ.get("ACTIVE_NBV_EXEC_MIN_TARGET_HEIGHT", "0.110"))
                            too_close_scan = float(metrics["distance"]) < min_exec_dist or float(metrics["height"]) < min_exec_height
                            if too_close_scan:
                                moved_ok = False
                                recovered_by_scan = False
                                print(
                                    f"[ActiveNBV][exec][{side_name}] FAILED: recovered joint-scan pose too close to target; "
                                    f"dist={metrics['distance']:.3f}m height={metrics['height']:.3f}m "
                                    f"min_dist={min_exec_dist:.3f}m min_height={min_exec_height:.3f}m"
                                )
                            else:
                                actual_lateral = float(np.linalg.norm(np.asarray(T_wc_act.t, dtype=float).reshape(3,)[:2] - center.astype(np.float64)[:2]))
                                actual_side_deg = float(np.rad2deg(np.arctan2(actual_lateral, max(float(metrics["height"]), 1e-6))))
                                max_side_deg = float(os.environ.get("ACTIVE_NBV_EXEC_MAX_SIDE_ANGLE_DEG", "45.0"))
                                actual_dir_xy = _actual_view_dir_xy_from_pose(T_wc_act)
                                actual_dir_ang = float(np.rad2deg(_angle_from_dir_xy(actual_dir_xy)))
                                desired_dir_err = _smallest_angle_diff_deg(_angle_from_dir_xy(actual_dir_xy), _angle_from_dir_xy(dir_xy_local))
                                move_info["T_wc_actual"] = T_wc_act
                                move_info["joint_scan"] = True
                                detail_record["nbv_joint_scan"] = True
                                detail_record["nbv_scan_near_fallback"] = bool(scan.get("near_fallback", False))
                                recovered_by_scan = True
                                print(
                                    f"[ActiveNBV][exec][{side_name}] joint-scan recovered view: "
                                    f"center_norm={metrics['center_norm']:.3f} dist={metrics['distance']:.3f}m "
                                    f"height={metrics['height']:.3f}m lateral={actual_lateral:.3f}m "
                                    f"side_angle={actual_side_deg:.1f}deg actual_dir={actual_dir_ang:.1f}deg "
                                    f"dir_err={desired_dir_err:.1f}deg near={not bool(scan.get('near_fallback', False))}"
                                )
                                max_dir_err = float(os.environ.get("ACTIVE_NBV_JOINT_SCAN_MAX_DIR_ERR_DEG", "45.0"))
                                if actual_side_deg > max_side_deg:
                                    moved_ok = False
                                    recovered_by_scan = False
                                    print(
                                        f"[ActiveNBV][exec][{side_name}] FAILED: recovered joint-scan side angle too large "
                                        f"{actual_side_deg:.1f}deg > {max_side_deg:.1f}deg."
                                    )
                                elif desired_dir_err > max_dir_err:
                                    moved_ok = False
                                    recovered_by_scan = False
                                    print(
                                        f"[ActiveNBV][exec][{side_name}] FAILED: recovered joint-scan actual direction differs from requested view "
                                        f"by {desired_dir_err:.1f}deg > {max_dir_err:.1f}deg."
                                    )
                                if recovered_by_scan and _env_flag("ACTIVE_NBV_USE_REACHABLE_SCAN_POSE_AS_TARGET", "1"):
                                    T_wc_side = T_wc_act
                                    desired_cam_pos = np.asarray(T_wc_side.t, dtype=float).reshape(3,)
                                    dir_xy_local = actual_dir_xy.astype(np.float64)
                                    move_info["pos_err"] = 0.0
                                    print(
                                        f"[ActiveNBV][exec][{side_name}] calibrated planned pose to recovered joint-scan pose "
                                        f"target_cam_pos_w={np.round(desired_cam_pos, 4)}"
                                    )
                if not recovered_by_scan:
                    moved_ok = False
                    actual_try_pos = np.asarray(T_wc_act.t, dtype=float).reshape(3,)
                    print(
                        f"[ActiveNBV][exec][{side_name}] FAILED: attempted camera pose but final error is too large; "
                        f"target_cam_pos_w={np.round(desired_cam_pos, 4)} "
                        f"attempted_cam_pos_w={np.round(actual_try_pos, 4)} "
                        f"pos_err={float(move_info.get('pos_err', np.inf)):.4f}m"
                    )
            else:
                # After the camera reaches the spherical-orbit position, do one more
                # orientation-only look-at correction using the measured camera position.
                T_wc_relook = _look_at_T_wc(np.asarray(T_wc_act.t, dtype=float).reshape(3,), center.astype(np.float64), x_ref_w=(1.0, 0.0, 0.0))
                pos_relook, ang_relook = _pose_error_metrics(T_wc_relook, T_wc_act)
                enable_relook = os.environ.get("ACTIVE_NBV_RELOOK", "0").strip().lower() in {"1", "true", "yes", "on"}
                if enable_relook and ang_relook > 1.2:
                    if debug:
                        print(f"[MV][{side_name}] relook correction ang_err={ang_relook:.2f}deg")
                    move_info = _move_camera_pose_closed_loop(
                        env,
                        cam_name=cam_name,
                        T_wc_des=T_wc_relook,
                        duration=float(os.environ.get("ACTIVE_NBV_RELOOK_DURATION", "0.15")),
                        settle_steps=int(os.environ.get("ACTIVE_NBV_RELOOK_SETTLE_STEPS", "2")),
                        pos_tol=0.004,
                        ang_tol_deg=1.4,
                        max_iters=1,
                        debug=debug,
                    )
        except Exception as e:
            moved_ok = False
            print(
                f"[ActiveNBV][exec][{side_name}] FAILED: camera pose IK unreachable "
                f"target_cam_pos_w={np.round(desired_cam_pos, 4)} err={type(e).__name__}: {e}"
            )

        if moved_ok:
            T_wc_final = get_T_wc_from_camera(env, cam_name)
            final_cam_pos = np.asarray(T_wc_final.t, dtype=float).reshape(3,)
            pos_err = float(np.linalg.norm(final_cam_pos - desired_cam_pos))
            actual_dir_xy_final = _actual_view_dir_xy_from_pose(T_wc_final)
            actual_angle_rad_final = float(_angle_from_dir_xy(actual_dir_xy_final))
            actual_angle_deg_final = float(np.rad2deg(actual_angle_rad_final))
            print(
                f"[ActiveNBV][exec][{side_name}] SUCCESS: executed_cam_pos_w={np.round(final_cam_pos, 4)} "
                f"pos_err={pos_err:.4f}m actual_dir={actual_angle_deg_final:.1f}deg"
            )
            desired_angle_rad_final = float(_angle_from_dir_xy(dir_xy_local))
            capture_dir_err_deg = _smallest_angle_diff_deg(actual_angle_rad_final, desired_angle_rad_final)
            max_capture_pos_err = float(os.environ.get("ACTIVE_NBV_CAPTURE_MAX_POS_ERR", "0.030"))
            max_capture_dir_err = float(os.environ.get("ACTIVE_NBV_CAPTURE_MAX_DIR_ERR_DEG", "15.0"))
            if _env_flag("ACTIVE_NBV_REFINE_POSE_BEFORE_CAPTURE", "1"):
                refine_pos_tol = float(os.environ.get("ACTIVE_NBV_REFINE_POS_TOL", "0.006"))
                refine_ang_tol = float(os.environ.get("ACTIVE_NBV_REFINE_ANG_TOL_DEG", "2.0"))
                refine_start_pos = float(os.environ.get("ACTIVE_NBV_REFINE_START_POS_ERR", "0.004"))
                refine_start_dir = float(os.environ.get("ACTIVE_NBV_REFINE_START_DIR_ERR_DEG", "3.0"))
                if pos_err > refine_start_pos or capture_dir_err_deg > refine_start_dir:
                    print(
                        f"[ActiveNBV][refine][{side_name}] start full-pose correction "
                        f"pos_err={pos_err:.4f}m dir_err={capture_dir_err_deg:.1f}deg"
                    )
                    try:
                        refine_info = _move_camera_pose_closed_loop(
                            env,
                            cam_name=cam_name,
                            T_wc_des=T_wc_side,
                            duration=float(os.environ.get("ACTIVE_NBV_REFINE_DURATION", "0.25")),
                            settle_steps=int(os.environ.get("ACTIVE_NBV_REFINE_SETTLE_STEPS", "6")),
                            pos_tol=refine_pos_tol,
                            ang_tol_deg=refine_ang_tol,
                            max_iters=int(os.environ.get("ACTIVE_NBV_REFINE_MAX_ITERS", "3")),
                            third_iter_pos_trigger=float(os.environ.get("ACTIVE_NBV_REFINE_THIRD_POS_TRIGGER", "0.008")),
                            third_iter_ang_trigger=float(os.environ.get("ACTIVE_NBV_REFINE_THIRD_ANG_TRIGGER", "2.5")),
                            debug=debug,
                        )
                        T_wc_refined = get_T_wc_from_camera(env, cam_name)
                        final_cam_pos = np.asarray(T_wc_refined.t, dtype=float).reshape(3,)
                        pos_err = float(np.linalg.norm(final_cam_pos - desired_cam_pos))
                        actual_dir_xy_final = _actual_view_dir_xy_from_pose(T_wc_refined)
                        actual_angle_rad_final = float(_angle_from_dir_xy(actual_dir_xy_final))
                        actual_angle_deg_final = float(np.rad2deg(actual_angle_rad_final))
                        capture_dir_err_deg = _smallest_angle_diff_deg(actual_angle_rad_final, desired_angle_rad_final)
                        print(
                            f"[ActiveNBV][refine][{side_name}] result "
                            f"converged={bool(refine_info.get('converged', False))} "
                            f"pos_err={pos_err:.4f}m dir_err={capture_dir_err_deg:.1f}deg "
                            f"actual_dir={actual_angle_deg_final:.1f}deg"
                        )
                    except Exception as e:
                        print(f"[ActiveNBV][refine][{side_name}] correction failed: {type(e).__name__}: {e}")
            if pos_err > max_capture_pos_err or capture_dir_err_deg > max_capture_dir_err:
                print(
                    f"[ActiveNBV][exec][{side_name}] SKIP: camera pose still too far from planned view "
                    f"pos_err={pos_err:.4f}m/{max_capture_pos_err:.4f}m "
                    f"dir_err={capture_dir_err_deg:.1f}/{max_capture_dir_err:.1f}deg; avoid ghosted fusion."
                )
                return _empty_side_data()
            if _env_flag("ACTIVE_NBV_ENFORCE_ACTUAL_VIEW_DIVERSITY", "1") and len(side_results) > 0:
                min_actual_disp = min(
                    _smallest_angle_diff_deg(actual_angle_rad_final, float(prev.get("capture_angle_rad", prev.get("angle_rad", 0.0))))
                    for prev in side_results
                    if int(np.asarray(prev.get("all_pts", np.zeros((0, 3), dtype=np.float32))).shape[0]) > 0
                ) if any(int(np.asarray(prev.get("all_pts", np.zeros((0, 3), dtype=np.float32))).shape[0]) > 0 for prev in side_results) else 180.0
                min_required = float(os.environ.get("ACTIVE_NBV_ACTUAL_MIN_SUCCESS_DISPARITY_DEG", "25.0"))
                if min_actual_disp < min_required:
                    print(
                        f"[ActiveNBV][exec][{side_name}] SKIP: actual view direction duplicates an already captured view "
                        f"min_disp={min_actual_disp:.1f}deg < {min_required:.1f}deg."
                    )
                    return _empty_side_data()

        if not moved_ok:
            if os.environ.get("MV_SKIP_UNREACHABLE_VIEW", "1").strip().lower() not in {"0", "false", "no", "off"}:
                print(f"[ActiveNBV][exec][{side_name}] SKIP: IK unreachable, skip this active view.")
                return _empty_side_data()
            if use_joint_fallback:
                fallback_sign = float(side_sign) if side_sign is not None else 1.0
                q_side = np.array(q_center, dtype=float).copy()
                q_side[base_yaw_idx] += fallback_sign * float(orbit_angle_exec)
                if enable_wrist_comp and wrist_yaw_idx is not None:
                    q_side[wrist_yaw_idx] -= fallback_sign * float(orbit_angle_exec)
                move_joint_smooth(env, q_side, duration=1.0, settle_steps=90)
            elif debug:
                print(
                    f"[MV][{side_name}] Nero7: skip legacy joint-space side-view fallback; "
                    "restore previous stable pose and capture there."
                )
            if not use_joint_fallback:
                if np.linalg.norm(np.asarray(robot.get_joint(), dtype=float) - q_before_side) > 1e-4:
                    move_joint_smooth(env, q_before_side, duration=0.25, settle_steps=15)

        if moved_ok and _env_flag("ACTIVE_NBV_CALIBRATE_BEFORE_CAPTURE", "1"):
            try:
                T_wc_now = get_T_wc_from_camera(env, cam_name)
                cam_pos_now = np.asarray(T_wc_now.t, dtype=float).reshape(3,)
                x_ref_now = np.asarray(T_wc_now.R, dtype=float)[:, 0]
                T_wc_centered = _look_at_T_wc(cam_pos_now, center.astype(np.float64), x_ref_w=x_ref_now)
                pre_pos_err, pre_ang_err = _pose_error_metrics(T_wc_centered, T_wc_now)
                print(
                    f"[ActiveNBV][calib][{side_name}] before_capture "
                    f"look_at_ang_err={pre_ang_err:.2f}deg pos_fixed={np.round(cam_pos_now, 4)}"
                )
                if pre_ang_err > float(os.environ.get("ACTIVE_NBV_CALIB_MIN_ANG_ERR", "0.8")):
                    _move_camera_pose_closed_loop(
                        env,
                        cam_name=cam_name,
                        T_wc_des=T_wc_centered,
                        duration=float(os.environ.get("ACTIVE_NBV_CALIB_DURATION", "0.20")),
                        settle_steps=int(os.environ.get("ACTIVE_NBV_CALIB_SETTLE_STEPS", "2")),
                        pos_tol=0.006,
                        ang_tol_deg=float(os.environ.get("ACTIVE_NBV_CALIB_ANG_TOL", "2.0")),
                        max_iters=int(os.environ.get("ACTIVE_NBV_CALIB_MAX_ITERS", "1")),
                        debug=debug,
                    )
            except Exception as e:
                print(f"[ActiveNBV][calib][{side_name}] calibration skipped: {type(e).__name__}: {e}")

        if moved_ok and _env_flag("ACTIVE_NBV_REQUIRE_TARGET_CENTERED_BEFORE_CAPTURE", "1"):
            T_wc_check = get_T_wc_from_camera(env, cam_name)
            cam_pos_check = np.asarray(T_wc_check.t, dtype=float).reshape(3,)
            metrics_check = _target_camera_metrics(T_wc_check, center.astype(np.float64))
            x_ref_check = np.asarray(T_wc_check.R, dtype=float)[:, 0]
            T_wc_look_check = _look_at_T_wc(cam_pos_check, center.astype(np.float64), x_ref_w=x_ref_check)
            _, look_ang_check = _pose_error_metrics(T_wc_look_check, T_wc_check)
            max_center_norm = float(os.environ.get("ACTIVE_NBV_CAPTURE_MAX_CENTER_NORM", "0.70"))
            max_look_ang = float(os.environ.get("ACTIVE_NBV_CAPTURE_MAX_LOOKAT_ANG_ERR", "28.0"))
            if (
                not bool(metrics_check["visible"])
                or float(metrics_check["center_norm"]) > max_center_norm
                or float(look_ang_check) > max_look_ang
            ):
                print(
                    f"[ActiveNBV][exec][{side_name}] SKIP: target not centered for capture "
                    f"center_norm={metrics_check['center_norm']:.3f} "
                    f"look_at_ang_err={look_ang_check:.2f}deg depth={metrics_check['depth']:.3f}"
                )
                return _empty_side_data()

        imgs = render_from_camera(env, cam_name, depth_mode='auto', debug=debug)
        rgb = imgs['img']
        depth = imgs['depth']
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        T_wc = get_T_wc_from_camera(env, cam_name)
        cam_info_side = CameraInfo(
            bgr.shape[1],
            bgr.shape[0],
            _build_cam_intrinsics_from_fovy(bgr.shape[1], bgr.shape[0], fovy_rad)[0, 0],
            _build_cam_intrinsics_from_fovy(bgr.shape[1], bgr.shape[0], fovy_rad)[1, 1],
            _build_cam_intrinsics_from_fovy(bgr.shape[1], bgr.shape[0], fovy_rad)[0, 2],
            _build_cam_intrinsics_from_fovy(bgr.shape[1], bgr.shape[0], fovy_rad)[1, 2],
            1.0,
        )
        uv_center, valid_center = _project_world_points_to_image(center[None, :], T_wc, cam_info_side)
        if np.any(valid_center):
            img_center = np.array([0.5 * bgr.shape[1], 0.5 * bgr.shape[0]], dtype=np.float32)
            uv_now = uv_center[0]
            uv_offset = uv_now - img_center
            if debug:
                print(
                    f"[MV][{side_name}] target_uv={np.round(uv_now, 2)} "
                    f"img_center={np.round(img_center, 2)} uv_offset={np.round(uv_offset, 2)}"
                )

        side_extract = _extract_side_target_point_sets(
            color_bgr=bgr,
            depth_m=depth,
            T_wc=T_wc,
            fovy_rad=fovy_rad,
            target_instance=target_instance,
            target_class=target_class,
            bbox_margin=float(bbox_margin),
            max_dist_to_centroid=float(max_dist_to_centroid),
            debug_mask_path=None if save_debug_dir is None else os.path.join(save_debug_dir, f"{side_name}_target_candidate_mask.png"),
        )
        pts_w = side_extract["pts_w"]
        cols = side_extract["cols"]

        if debug or _env_flag("MV_POSE_DIAG_LOG", "1"):
            cam_pos = np.asarray(T_wc.t, dtype=float).reshape(3,)
            center_c_side = _world_to_cam_points(T_wc, center[None, :])[0].astype(float)
            center_c_center = _world_to_cam_points(T_wc_center, center[None, :])[0].astype(float)
            R_delta = np.asarray(T_wc_center.R, dtype=float).T @ np.asarray(T_wc.R, dtype=float)
            tr_delta = np.clip((np.trace(R_delta) - 1.0) * 0.5, -1.0, 1.0)
            view_rot_delta_deg = float(np.degrees(np.arccos(tr_delta)))
            print(
                f"[MV][{side_name}][pose] cam_pos={np.round(cam_pos,4)} "
                f"dist_to_center={np.linalg.norm(cam_pos-center):.4f} "
                f"center_depth_side={center_c_side[2]:.4f} center_xy_side={np.round(center_c_side[:2], 4)} "
                f"center_depth_center={center_c_center[2]:.4f} "
                f"view_rot_delta={view_rot_delta_deg:.2f}deg target_candidate_pts={pts_w.shape[0]}"
            )

        pts_c_in_center = _world_to_cam_points(T_wc_center, pts_w)
        pts_c_in_center, cols = _voxel_downsample_xyzrgb(pts_c_in_center, cols, voxel_size=float(voxel_size))

        prior_clouds = [pts0_c_ds]
        for prev_side in side_results:
            prev_pts = np.asarray(prev_side.get("all_pts", np.zeros((0, 3), dtype=np.float32)), dtype=np.float32).reshape(-1, 3)
            if prev_pts.shape[0] > 0:
                prior_clouds.append(prev_pts)
        prior_pts = np.concatenate(prior_clouds, axis=0).astype(np.float32) if len(prior_clouds) > 0 else pts0_c_ds
        incremental_thr = float(os.environ.get("ACTIVE_NBV_INCREMENTAL_NN_THR", str(float(nn_gate_dist))))
        _, prior_overlap_keep = _nn_gate(pts_c_in_center, prior_pts, thr=incremental_thr)
        incremental_pts = pts_c_in_center[~prior_overlap_keep]
        incremental_count = int(incremental_pts.shape[0])
        incremental_ratio = float(incremental_count / max(int(pts_c_in_center.shape[0]), 1))
        if len(side_results) > 0:
            min_incremental_points = int(os.environ.get("ACTIVE_NBV_MIN_INCREMENTAL_POINTS", "120"))
            min_incremental_ratio = float(os.environ.get("ACTIVE_NBV_MIN_INCREMENTAL_RATIO", "0.08"))
            print(
                f"[ActiveNBV][exec][{side_name}] incremental_vs_existing "
                f"new={incremental_count}/{int(pts_c_in_center.shape[0])} ratio={incremental_ratio:.3f} "
                f"thr={incremental_thr:.4f}"
            )
            if _env_flag("ACTIVE_NBV_REQUIRE_INCREMENTAL_SECOND_VIEW", "1") and (
                incremental_count < min_incremental_points and incremental_ratio < min_incremental_ratio
            ):
                print(
                    f"[ActiveNBV][exec][{side_name}] SKIP: insufficient incremental target surface "
                    f"new={incremental_count} ratio={incremental_ratio:.3f} "
                    f"required_points={min_incremental_points} required_ratio={min_incremental_ratio:.3f}"
                )
                return _empty_side_data()

        if _env_flag("MV_TARGET_CENTER_CALIBRATION", "1") and pts_c_in_center.shape[0] > 0 and pts0_c_ds.shape[0] > 0:
            pts_c_calib, calib_info = _target_translation_calibration(
                pts_c_in_center,
                pts0_c_ds,
                coarse_thr=float(os.environ.get("MV_TARGET_CALIB_NN_THR", "0.045")),
                max_shift=float(os.environ.get("MV_TARGET_CALIB_MAX_SHIFT", "0.030")),
                min_pairs=int(os.environ.get("MV_TARGET_CALIB_MIN_PAIRS", "50")),
                min_improve=float(os.environ.get("MV_TARGET_CALIB_MIN_IMPROVE", "0.06")),
            )
            before = calib_info.get("before", {})
            after = calib_info.get("after", {})
            if debug or _env_flag("MV_TARGET_CALIB_LOG", "1"):
                shift = np.asarray(calib_info.get("shift", np.zeros(3, dtype=np.float32)), dtype=np.float32)
                if debug:
                    print(
                        f"[MV][{side_name}][calib] applied={bool(calib_info.get('applied', False))} "
                        f"reason={calib_info.get('reason', 'unknown')} pairs={int(calib_info.get('pairs', 0))} "
                        f"shift={np.round(shift, 4)} shift_norm={float(calib_info.get('shift_norm', 0.0)):.4f} "
                        f"before={before} after={after}"
                    )
                else:
                    print(
                        f"[MV][{side_name}][calib] applied={bool(calib_info.get('applied', False))} "
                        f"pairs={int(calib_info.get('pairs', 0))} "
                        f"shift_norm={float(calib_info.get('shift_norm', 0.0)):.4f} "
                        f"median {float(before.get('median', float('nan'))):.4f}->{float(after.get('median', float('nan'))):.4f} "
                        f"inlier1cm {float(before.get('inlier@1cm', 0.0)):.3f}->{float(after.get('inlier@1cm', 0.0)):.3f}"
                    )
            pts_c_in_center = pts_c_calib
            if _env_flag("MV_TARGET_RIGID_CALIBRATION", "1"):
                thr_env = os.environ.get("MV_TARGET_RIGID_THR_LIST", "0.018 0.012 0.008")
                rigid_thr = tuple(float(x) for x in thr_env.replace(",", " ").split() if x.strip())
                pts_c_rigid, rigid_info = _target_rigid_calibration(
                    pts_c_in_center,
                    pts0_c_ds,
                    voxel=float(os.environ.get("MV_TARGET_RIGID_VOXEL", "0.0025")),
                    thr_list=rigid_thr if len(rigid_thr) > 0 else (0.018, 0.012, 0.008),
                    max_iter=int(os.environ.get("MV_TARGET_RIGID_MAX_ITER", "25")),
                    max_trans=float(os.environ.get("MV_TARGET_RIGID_MAX_TRANS", "0.010")),
                    max_rot_deg=float(os.environ.get("MV_TARGET_RIGID_MAX_ROT_DEG", "3.0")),
                    min_improve=float(os.environ.get("MV_TARGET_RIGID_MIN_IMPROVE", "0.04")),
                    min_pairs=int(os.environ.get("MV_TARGET_RIGID_MIN_PAIRS", "80")),
                )
                if debug or _env_flag("MV_TARGET_CALIB_LOG", "1"):
                    rb = rigid_info.get("before", {})
                    ra = rigid_info.get("after", {})
                    print(
                        f"[MV][{side_name}][rigid] applied={bool(rigid_info.get('applied', False))} "
                        f"reason={rigid_info.get('reason', 'unknown')} fit={float(rigid_info.get('fit', 0.0)):.3f} "
                        f"rmse={float(rigid_info.get('rmse', 0.0)):.4f} "
                        f"overlap={int(rigid_info.get('overlap_pairs', 0))}/{float(rigid_info.get('overlap_ratio', 0.0)):.3f} "
                        f"dT={float(rigid_info.get('trans', 0.0)):.4f}m "
                        f"dR={float(rigid_info.get('rot_deg', 0.0)):.2f}deg "
                        f"median {float(rb.get('median', float('nan'))):.4f}->{float(ra.get('median', float('nan'))):.4f} "
                        f"inlier1cm {float(rb.get('inlier@1cm', 0.0)):.3f}->{float(ra.get('inlier@1cm', 0.0)):.3f}"
                    )
                    if str(rigid_info.get("reason", "")).startswith("diagnostic_large_"):
                        src_box = rigid_info.get("src_bbox", {})
                        tgt_box = rigid_info.get("tgt_bbox", {})
                        print(
                            f"[MV][{side_name}][rigid] DIAG: ICP requested an implausibly large correction; "
                            "treat this as mask/extrinsic/view-pose inconsistency, not as a valid calibration. "
                            f"src_center={np.round(np.asarray(src_box.get('center', np.zeros(3)), dtype=float), 4)} "
                            f"src_extent={np.round(np.asarray(src_box.get('extent', np.zeros(3)), dtype=float), 4)} "
                            f"tgt_center={np.round(np.asarray(tgt_box.get('center', np.zeros(3)), dtype=float), 4)} "
                            f"tgt_extent={np.round(np.asarray(tgt_box.get('extent', np.zeros(3)), dtype=float), 4)}"
                        )
                pts_c_in_center = pts_c_rigid
                if bool(rigid_info.get("applied", False)):
                    after = rigid_info.get("after", after)
            if _env_flag("MV_REJECT_POOR_TARGET_CALIB", "1"):
                after_median = float(after.get("median", float("inf")))
                after_inlier = float(after.get("inlier@1cm", 0.0))
                max_calib_median = float(os.environ.get("MV_TARGET_CALIB_MAX_MEDIAN", "0.030"))
                min_calib_inlier = float(os.environ.get("MV_TARGET_CALIB_MIN_INLIER_1CM", "0.15"))
                if (
                    (not np.isfinite(after_median))
                    or after_median > max_calib_median
                    or after_inlier < min_calib_inlier
                ):
                    print(
                        f"[MV][{side_name}][calib] SKIP: poor target alignment after calibration "
                        f"median={after_median:.4f} max={max_calib_median:.4f} "
                        f"inlier1cm={after_inlier:.3f} min={min_calib_inlier:.3f}; avoid ghosted fusion."
                    )
                    return _empty_side_data()

        overlap_pts, overlap_keep = _nn_gate(pts_c_in_center, pts0_c_ds, thr=float(nn_gate_dist))
        overlap_cols = cols[overlap_keep] if cols is not None and cols.shape[0] == pts_c_in_center.shape[0] else None
        novel_pts = pts_c_in_center[~overlap_keep]
        novel_cols = cols[~overlap_keep] if cols is not None and cols.shape[0] == pts_c_in_center.shape[0] else None

        if debug:
            print(
                f"[MV][{side_name}] split target pts: "
                f"overlap={overlap_pts.shape[0]} novel={novel_pts.shape[0]}"
            )

        if save_debug_dir is not None and pts_c_in_center.shape[0] > 0:
            pcd_dbg = o3d.geometry.PointCloud()
            pcd_dbg.points = o3d.utility.Vector3dVector(pts_c_in_center.astype(np.float32))
            if cols is not None:
                pcd_dbg.colors = o3d.utility.Vector3dVector(cols.astype(np.float32))
            o3d.io.write_point_cloud(os.path.join(save_debug_dir, f"{side_name}_target_all_in_center_cam.ply"), pcd_dbg)
            if novel_pts.shape[0] > 0:
                pcd_novel = o3d.geometry.PointCloud()
                pcd_novel.points = o3d.utility.Vector3dVector(novel_pts.astype(np.float32))
                if novel_cols is not None:
                    pcd_novel.colors = o3d.utility.Vector3dVector(novel_cols.astype(np.float32))
                o3d.io.write_point_cloud(os.path.join(save_debug_dir, f"{side_name}_novel_in_center_cam.ply"), pcd_novel)

        return {
            "all_pts": pts_c_in_center,
            "all_cols": cols,
            "overlap_pts": overlap_pts,
            "overlap_cols": overlap_cols,
            "novel_pts": novel_pts,
            "novel_cols": novel_cols,
            "incremental_pts": incremental_pts.astype(np.float32),
            "incremental_count": int(incremental_count),
            "incremental_ratio": float(incremental_ratio),
            "view_name": str(side_name),
            "dir_xy": dir_xy_local.astype(np.float32),
            "angle_rad": float(_angle_from_dir_xy(dir_xy_local)),
            "capture_dir_xy": _actual_view_dir_xy_from_pose(T_wc).astype(np.float32),
            "capture_angle_rad": float(_angle_from_dir_xy(_actual_view_dir_xy_from_pose(T_wc))),
            "orbit_angle_rad": float(orbit_angle_exec),
            "target_cam_pos_w": desired_cam_pos.astype(np.float32),
            "capture_cam_pos_w": np.asarray(T_wc.t, dtype=np.float32).reshape(3,),
        }

    custom_view_records = custom_view_records if custom_view_records is not None else []
    side_results = []
    side_names = []

    def _adaptive_dirs_after_first(first_side):
        base_dir = _normalize_dir_xy(first_side.get("capture_dir_xy", first_side.get("dir_xy", orbit_horizontal_dir_xy)))
        raw_dirs = [
            ("adaptive_opposite", -base_dir),
            ("adaptive_ortho_left", _rotate_dir_xy(base_dir, np.pi / 2.0)),
            ("adaptive_ortho_right", _rotate_dir_xy(base_dir, -np.pi / 2.0)),
            ("adaptive_plus60", _rotate_dir_xy(base_dir, np.deg2rad(60.0))),
            ("adaptive_minus60", _rotate_dir_xy(base_dir, -np.deg2rad(60.0))),
        ]
        scored = []
        for name, d in raw_dirs:
            obs_pen, obs_detail = _obstacle_direction_penalty(scene_context, center.astype(np.float32), d)
            disp = _smallest_angle_diff_deg(_angle_from_dir_xy(d), float(first_side.get("capture_angle_rad", first_side.get("angle_rad", 0.0))))
            if disp < float(os.environ.get("ACTIVE_NBV_ADAPTIVE_MIN_DISPARITY_DEG", "25.0")):
                continue
            score = float(disp / 180.0 - float(os.environ.get("ACTIVE_NBV_ADAPTIVE_OBSTACLE_WEIGHT", "1.2")) * obs_pen)
            scored.append((score, name, d, obs_pen, obs_detail))
        scored.sort(key=lambda x: x[0], reverse=True)
        if _env_flag("ACTIVE_NBV_ADAPTIVE_LOG", "1") and len(scored) > 0:
            print("[ActiveNBV][adaptive] second-view candidates after first actual view:")
            for score, name, d, obs_pen, obs_detail in scored:
                print(
                    f"  {name}: dir_angle={np.rad2deg(_angle_from_dir_xy(d)):.1f}deg "
                    f"score={score:.3f} obs_pen={obs_pen:.3f} "
                    f"obs={obs_detail.get('label', '')}@{obs_detail.get('angle_deg', 180.0):.1f}deg"
                )
        return [(name, d) for _, name, d, _, _ in scored]

    if custom_view_dirs_xy is not None and len(custom_view_dirs_xy) >= 2:
        target_success = int(os.environ.get("ACTIVE_NBV_TARGET_SUCCESS_VIEWS", "2"))
        max_attempts = int(os.environ.get("ACTIVE_NBV_MAX_EXEC_ATTEMPTS", "4"))
        attempt_num = min(len(custom_view_dirs_xy), max(max_attempts, target_success))
        for i in range(attempt_num):
            view_record_i = custom_view_records[i] if i < len(custom_view_records) else None
            side_name_i = f"view{i + 1}"
            side_data_i = _capture_side_circle(
                side_name_i,
                side_sign=(+1.0 if i % 2 == 0 else -1.0),
                view_dir_xy=custom_view_dirs_xy[i],
                view_record=view_record_i,
            )
            if int(np.asarray(side_data_i.get("all_pts", np.zeros((0, 3), dtype=np.float32))).shape[0]) > 0:
                side_results.append(side_data_i)
                side_names.append(side_name_i)
                if (
                    len(side_results) == 1
                    and target_success >= 2
                    and _env_flag("ACTIVE_NBV_ADAPTIVE_SECOND_AFTER_FIRST", "1")
                ):
                    for adaptive_name, adaptive_dir in _adaptive_dirs_after_first(side_results[0]):
                        if len(side_results) >= max(target_success, 1):
                            break
                        side_data_adapt = _capture_side_circle(adaptive_name, side_sign=+1.0, view_dir_xy=adaptive_dir)
                        if int(np.asarray(side_data_adapt.get("all_pts", np.zeros((0, 3), dtype=np.float32))).shape[0]) > 0:
                            side_results.append(side_data_adapt)
                            side_names.append(adaptive_name)
                if len(side_results) >= max(target_success, 1):
                    break
        if len(side_results) < 2 and _env_flag("ACTIVE_NBV_TRY_DEFAULT_SIDE_FALLBACK", "1"):
            if len(side_results) > 0 and "dir_xy" in side_results[0]:
                base_dir = _normalize_dir_xy(side_results[0]["dir_xy"])
            else:
                base_dir = _normalize_dir_xy(orbit_horizontal_dir_xy)
            fallback_dirs = [
                ("fallback_opposite", -base_dir),
                ("fallback_ortho_left", _rotate_dir_xy(base_dir, np.pi / 2.0)),
                ("fallback_ortho_right", _rotate_dir_xy(base_dir, -np.pi / 2.0)),
            ]
            if _env_flag("ACTIVE_NBV_ALLOW_LEGACY_SAME_DIR_FALLBACK", "0"):
                fallback_dirs.extend([
                    ("fallback_left", _normalize_dir_xy(orbit_horizontal_dir_xy)),
                    ("fallback_right", -_normalize_dir_xy(orbit_horizontal_dir_xy)),
                ])
            for fallback_name, fallback_dir in fallback_dirs:
                if len(side_results) >= 2:
                    break
                if any(
                    _smallest_angle_diff_deg(_angle_from_dir_xy(fallback_dir), float(sd.get("angle_rad", 0.0)))
                    < float(os.environ.get("ACTIVE_NBV_EXEC_MIN_SUCCESS_DISPARITY_DEG", "25.0"))
                    for sd in side_results
                ):
                    continue
                side_data_i = _capture_side_circle(fallback_name, side_sign=+1.0, view_dir_xy=fallback_dir)
                if int(np.asarray(side_data_i.get("all_pts", np.zeros((0, 3), dtype=np.float32))).shape[0]) > 0:
                    side_results.append(side_data_i)
                    side_names.append(fallback_name)
        if len(side_results) < 2:
            print(f"[ActiveNBV][exec] WARNING: only {len(side_results)}/2 side views produced target points; fusion will use available views.")
        while len(side_results) < 2:
            side_results.append(_empty_side_data())
            side_names.append(f"view{len(side_names) + 1}")
        if _env_flag("ACTIVE_NBV_PRINT_VIEW_SUMMARY", "1"):
            print("[ActiveNBV][views] successful observation views:")
            for name_i, data_i in zip(side_names, side_results):
                pts_i = int(np.asarray(data_i.get("all_pts", np.zeros((0, 3), dtype=np.float32))).shape[0])
                if pts_i <= 0:
                    print(f"  {name_i}: skipped/no target points")
                    continue
                target_pos = np.asarray(data_i.get("target_cam_pos_w", np.full(3, np.nan)), dtype=float).reshape(3,)
                capture_pos = np.asarray(data_i.get("capture_cam_pos_w", np.full(3, np.nan)), dtype=float).reshape(3,)
                dir_xy_i = _normalize_dir_xy(data_i.get("dir_xy", np.array([1.0, 0.0], dtype=np.float32)))
                angle_deg_i = float(np.rad2deg(_angle_from_dir_xy(dir_xy_i)))
                capture_angle_deg_i = float(np.rad2deg(float(data_i.get("capture_angle_rad", _angle_from_dir_xy(dir_xy_i)))))
                orbit_deg_i = float(np.rad2deg(float(data_i.get("orbit_angle_rad", float(joint0_delta_rad)))))
                print(
                    f"  {name_i}: target_cam_pos_w={np.round(target_pos, 4)} "
                    f"capture_cam_pos_w={np.round(capture_pos, 4)} "
                    f"planned_dir={angle_deg_i:.1f}deg actual_dir={capture_angle_deg_i:.1f}deg "
                    f"orbit={orbit_deg_i:.1f}deg points={pts_i} "
                    f"incremental={int(data_i.get('incremental_count', 0))} "
                    f"inc_ratio={float(data_i.get('incremental_ratio', 0.0)):.3f}"
                )
        sideL, sideR = side_results[0], side_results[1]
        left_name, right_name = side_names[0], side_names[1]
    else:
        sideL = _capture_side_circle("left", side_sign=+1.0)
        sideR = _capture_side_circle("right", side_sign=-1.0)
        left_name, right_name = "left", "right"

    # restore center pose
    side_points = int(np.asarray(sideL.get("all_pts", np.zeros((0, 3), dtype=np.float32))).shape[0]) + int(np.asarray(sideR.get("all_pts", np.zeros((0, 3), dtype=np.float32))).shape[0])
    restore_center = os.environ.get("ACTIVE_NBV_RESTORE_CENTER", "1").strip().lower() in {"1", "true", "yes", "on"}
    if side_points > 0 and restore_center:
        restore_duration = float(os.environ.get("ACTIVE_NBV_RESTORE_DURATION", "0.8"))
        restore_settle = int(os.environ.get("ACTIVE_NBV_RESTORE_SETTLE_STEPS", "45"))
        print("[ActiveNBV][exec] restore center observation pose before grasp execution.")
        move_joint_smooth(env, q_center, duration=restore_duration, settle_steps=restore_settle)
    elif side_points <= 0:
        print("[ActiveNBV][exec] all active views skipped; skip center-pose restore to avoid replan pause.")

    # -------- Alignment on overlap; fusion with novel surfaces --------
    def _align_and_select(side_data, side_name: str):
        if not isinstance(side_data, dict):
            if debug:
                print(f"[MV][{side_name}] skip malformed side-view data: {type(side_data).__name__}")
            return np.zeros((0, 3), dtype=np.float32), None
        overlap_pts = np.asarray(side_data["overlap_pts"], dtype=np.float32).reshape(-1, 3)
        overlap_cols = None if side_data["overlap_cols"] is None else np.asarray(side_data["overlap_cols"], dtype=np.float32).reshape(-1, 3)
        novel_pts = np.asarray(side_data["novel_pts"], dtype=np.float32).reshape(-1, 3)
        novel_cols = None if side_data["novel_cols"] is None else np.asarray(side_data["novel_cols"], dtype=np.float32).reshape(-1, 3)

        if overlap_pts.shape[0] < 80:
            if debug:
                print(f"[MV][{side_name}] reject side-view from fusion: overlap subset too small ({overlap_pts.shape[0]})")
            return np.zeros((0, 3), dtype=np.float32), None

        T_align = np.eye(4, dtype=np.float64)
        st_before = _nn_stats(overlap_pts, pts0_c_ds)
        if verify and debug:
            print(f"[MV][{side_name}] overlap NN stats vs center: {st_before}")

        cov = np.cov(overlap_pts.astype(np.float64).T) if overlap_pts.shape[0] >= 6 else np.eye(3, dtype=np.float64)
        eigvals = np.sort(np.linalg.eigvalsh(cov))[::-1]
        shape_ratio_21 = float(eigvals[1] / max(eigvals[0], 1e-9))
        shape_ratio_31 = float(eigvals[2] / max(eigvals[0], 1e-9))
        overlap_already_good = (
            st_before["median"] <= 0.0065 and
            st_before["p95"] <= 0.0195 and
            st_before["inlier@1cm"] >= 0.60
        )
        overlap_degenerate = (shape_ratio_21 < 0.06) or (shape_ratio_31 < 0.008)

        if icp_refine:
            if overlap_already_good or overlap_degenerate:
                if debug:
                    why = []
                    if overlap_already_good:
                        why.append("already_good")
                    if overlap_degenerate:
                        why.append(f"degenerate_shape(r21={shape_ratio_21:.4f},r31={shape_ratio_31:.4f})")
                    print(f"[MV][{side_name}] skip ICP: {', '.join(why)}")
            else:
                T_icp, fit, rmse = _icp_multiscale(
                    overlap_pts,
                    pts0_c_ds,
                    voxel=float(icp_voxel),
                    thr_list=tuple(icp_thr_list),
                    max_iter=int(icp_max_iter),
                )
                R_icp = np.asarray(T_icp[:3, :3], dtype=np.float64)
                t_icp = np.asarray(T_icp[:3, 3], dtype=np.float64).reshape(3,)
                tr_icp = np.clip((np.trace(R_icp) - 1.0) * 0.5, -1.0, 1.0)
                icp_rot_deg = float(np.degrees(np.arccos(tr_icp)))
                icp_trans = float(np.linalg.norm(t_icp))
                overlap_icp = _apply_T(overlap_pts, T_icp)
                st_after_icp = _nn_stats(overlap_icp, pts0_c_ds)
                accept = (
                    fit >= 0.12 and
                    np.isfinite(st_after_icp["median"]) and
                    icp_trans <= 0.015 and
                    icp_rot_deg <= 8.0 and
                    (
                        st_after_icp["median"] <= st_before["median"] * 0.97 or
                        st_after_icp["p95"] <= st_before["p95"] * 0.97
                    )
                )
                if debug:
                    print(
                        f"[MV][{side_name}] overlap-only ICP fit={fit:.4f} rmse={rmse:.4f} "
                        f"dT={icp_trans:.4f}m dR={icp_rot_deg:.2f}deg "
                        f"before={st_before} after={st_after_icp} accept={accept}"
                    )
                if accept:
                    T_align = T_icp

        overlap_aligned = _apply_T(overlap_pts, T_align)
        st_after = _nn_stats(overlap_aligned, pts0_c_ds)
        if verify and debug:
            print(f"[MV][{side_name}] verified overlap stats: {st_after}")

        reject_reasons = []
        if verify:
            max_median = float(os.environ.get("MV_FUSION_MAX_OVERLAP_MEDIAN", "0.025"))
            max_p95 = float(os.environ.get("MV_FUSION_MAX_OVERLAP_P95", "0.050"))
            min_inlier = float(os.environ.get("MV_FUSION_MIN_INLIER_1CM", "0.15"))
            if (not np.isfinite(st_after["median"])) or st_after["median"] > max_median:
                reject_reasons.append(f"median={st_after['median']:.4f}")
            if (not np.isfinite(st_after["p95"])) or st_after["p95"] > max_p95:
                reject_reasons.append(f"p95={st_after['p95']:.4f}")
            if st_after["inlier@1cm"] < min_inlier:
                reject_reasons.append(f"inlier@1cm={st_after['inlier@1cm']:.3f}")

        if len(reject_reasons) > 0:
            if debug:
                print(f"[MV][{side_name}] reject side-view from fusion: {', '.join(reject_reasons)}")
            return np.zeros((0, 3), dtype=np.float32), None

        if fusion_mode in ("strict", "strict_overlap"):
            strict_thr = float(os.environ.get("MV_STRICT_OVERLAP_THR", "0.006"))
            _, keep_strict = _nn_gate(overlap_aligned, pts0_c_ds, thr=strict_thr)
            overlap_emit = overlap_aligned[keep_strict]
            overlap_emit_cols = (
                overlap_cols[keep_strict]
                if overlap_cols is not None and overlap_cols.shape[0] == keep_strict.shape[0]
                else overlap_cols
            )
            overlap_emit, overlap_emit_cols = _voxel_downsample_xyzrgb(
                overlap_emit,
                overlap_emit_cols,
                voxel_size=float(voxel_size),
            )
            if debug:
                print(
                    f"[MV][{side_name}] strict_overlap fusion: "
                    f"thr={strict_thr:.4f} append={overlap_emit.shape[0]} "
                    f"from_overlap={overlap_aligned.shape[0]} novel_pts_ignored={novel_pts.shape[0]}"
                )
            return overlap_emit, overlap_emit_cols

        if fusion_mode in ("conservative", "overlap_only"):
            overlap_emit = overlap_aligned
            overlap_emit_cols = overlap_cols
            overlap_emit, overlap_emit_cols = _voxel_downsample_xyzrgb(
                overlap_emit,
                overlap_emit_cols,
                voxel_size=float(voxel_size),
            )
            if debug:
                print(
                    f"[MV][{side_name}] conservative fusion: "
                    f"append overlap subset only overlap_pts={overlap_emit.shape[0]} "
                    f"novel_pts_ignored={novel_pts.shape[0]}"
                )
            return overlap_emit, overlap_emit_cols

        if novel_pts.shape[0] == 0:
            if debug:
                print(f"[MV][{side_name}] no novel surface points after overlap split; skip fusion append.")
            overlap_dense = overlap_aligned
            overlap_dense_cols = overlap_cols
            if overlap_dense.shape[0] > 0:
                dense_thr = float(min(0.010, max(0.006, 0.5 * float(nn_gate_dist))))
                _, keep_dense = _nn_gate(overlap_dense, pts0_c_ds, thr=dense_thr)
                overlap_dense = overlap_dense[keep_dense]
                overlap_dense_cols = (
                    overlap_dense_cols[keep_dense]
                    if overlap_dense_cols is not None and overlap_dense_cols.shape[0] == keep_dense.shape[0]
                    else overlap_dense_cols
                )
                overlap_dense, overlap_dense_cols = _voxel_downsample_xyzrgb(
                    overlap_dense,
                    overlap_dense_cols,
                    voxel_size=float(max(0.5 * float(voxel_size), 0.0008)),
                )
            if debug:
                print(
                    f"[MV][{side_name}] no novel surface points; "
                    f"append dense overlap subset only overlap_dense={overlap_dense.shape[0]}"
                )
            return overlap_dense, overlap_dense_cols

        novel_aligned = _apply_T(novel_pts, T_align)
        novel_aligned, novel_cols = _voxel_downsample_xyzrgb(novel_aligned, novel_cols, voxel_size=float(voxel_size))
        overlap_dense = overlap_aligned
        overlap_dense_cols = overlap_cols
        dense_thr = float(min(0.010, max(0.006, 0.5 * float(nn_gate_dist))))
        _, keep_dense = _nn_gate(overlap_dense, pts0_c_ds, thr=dense_thr)
        overlap_dense = overlap_dense[keep_dense]
        overlap_dense_cols = (
            overlap_dense_cols[keep_dense]
            if overlap_dense_cols is not None and overlap_dense_cols.shape[0] == keep_dense.shape[0]
            else overlap_dense_cols
        )
        overlap_dense, overlap_dense_cols = _voxel_downsample_xyzrgb(
            overlap_dense,
            overlap_dense_cols,
            voxel_size=float(max(0.5 * float(voxel_size), 0.0008)),
        )
        if overlap_dense.shape[0] > 0:
            fused_side_pts = np.concatenate([overlap_dense, novel_aligned], axis=0).astype(np.float32)
            fused_side_cols = None
            if overlap_dense_cols is not None and novel_cols is not None:
                fused_side_cols = np.concatenate([overlap_dense_cols, novel_cols], axis=0).astype(np.float32)
        else:
            fused_side_pts = novel_aligned
            fused_side_cols = novel_cols
        if debug:
            print(
                f"[MV][{side_name}] balanced fusion append: "
                f"novel_pts={novel_aligned.shape[0]} overlap_dense={overlap_dense.shape[0]} "
                f"overlap_pts={overlap_pts.shape[0]}"
            )
        return fused_side_pts, fused_side_cols

    ptsL_g, colL_g = _align_and_select(sideL, left_name)
    ptsR_g, colR_g = _align_and_select(sideR, right_name)

    # -------- Merge in center camera frame --------
    all_pts = [pts0_c_ds]
    all_cols = [col0_c_ds] if col0_c_ds is not None else []

    if ptsL_g.shape[0] > 0:
        all_pts.append(ptsL_g)
        if colL_g is not None:
            all_cols.append(colL_g)
    if ptsR_g.shape[0] > 0:
        all_pts.append(ptsR_g)
        if colR_g is not None:
            all_cols.append(colR_g)

    pts_fused = np.concatenate(all_pts, axis=0).astype(np.float32)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts_fused)

    if len(all_cols) > 0:
        cols_fused = np.concatenate(all_cols, axis=0).astype(np.float32)
        if cols_fused.shape[0] == pts_fused.shape[0]:
            pcd.colors = o3d.utility.Vector3dVector(cols_fused)

    # Final light cleanup
    if voxel_size is not None and voxel_size > 0:
        pcd = pcd.voxel_down_sample(voxel_size=float(voxel_size))
    try:
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    except Exception:
        pass

    if save_debug_dir is not None:
        o3d.io.write_point_cloud(os.path.join(save_debug_dir, "fused_center_cam.ply"), pcd)

    if debug:
        print(f"[MV] fused cloud points = {np.asarray(pcd.points).shape[0]}")

    return pcd


def fuse_active_views_by_bbox(env,
                              cam_name: str,
                              q_center: np.ndarray,
                              T_wc_center: sm.SE3,
                              cloud_center_cam: o3d.geometry.PointCloud,
                              fovy_rad: float,
                              target_class: str = None,
                              joint0_delta_rad: float = np.deg2rad(30.0),
                              bbox_margin: float = 0.03,
                              max_dist_to_centroid: float = 0.20,
                              voxel_size: float = 0.0006,
                              icp_refine: bool = False,
                              icp_voxel: float = 0.003,
                              icp_thr_list=(0.06, 0.04, 0.025, 0.015),
                              icp_max_iter: int = 60,
                              nn_gate_dist: float = 0.02,
                              verify: bool = True,
                              save_debug_dir: str = None,
                              debug: bool = False,
                              top_mask=None,
                              top_depth=None,
                              scene_context=None,
                              manual_view_dirs_xy=None,
                              fusion_mode: str = "balanced"):
    """Predict best paired side views from the *top* frame, then only execute that pair.

    Pair choices:
      - front_back  -> use major_xy
      - left_right  -> use minor_xy

    And the side poses are generated on a strict circle around the object center.
    """
    pts0_c = np.asarray(cloud_center_cam.points, dtype=np.float32)
    if pts0_c.size == 0:
        raise RuntimeError('[ActiveNBV] center masked cloud is empty.')

    pts0_w = _cam_to_world_points(T_wc_center, pts0_c)
    normals_w, curv = _estimate_normals_and_curvature(pts0_w, radius=0.02, max_nn=30)
    sparse = _local_density_score(pts0_w, k=20)

    edge_pts = np.zeros((len(pts0_w),), dtype=np.float32)
    dd_pts = np.zeros((len(pts0_w),), dtype=np.float32)

    if top_mask is not None and top_depth is not None:
        edge_img = _mask_edge_weight(top_mask)
        dd_img = _depth_discontinuity_weight(top_depth, top_mask)
        _, edge_vals = _sample_image_weight_for_points(edge_img, (top_mask > 0).astype(np.uint8), top_depth, fovy_rad)
        _, dd_vals = _sample_image_weight_for_points(dd_img, (top_mask > 0).astype(np.uint8), top_depth, fovy_rad)
        n = min(len(edge_vals), len(pts0_c), len(pts0_w))
        edge_pts[:n] = edge_vals[:n]
        dd_pts[:n] = dd_vals[:n]

    curv_n = curv.copy()
    if len(curv_n) > 0 and float(np.max(curv_n)) > 1e-8:
        curv_n = curv_n / float(np.max(curv_n))

    uncertainty = _clamp01(0.35 * edge_pts + 0.35 * dd_pts + 0.20 * sparse + 0.10 * curv_n)
    grasp_heat = build_grasp_sensitive_heatmap(pts0_w, normals_w, curv, edge_pts, sparse, coarse_grasps=None)

    if scene_context is None:
        pred = predict_best_view_pair_from_topview(
            pts0_w,
            normals_w,
            curv,
            uncertainty,
            grasp_heat,
            debug=debug
        )

        best_pair = pred['best_pair_name']
        if best_pair == 'front_back':
            orbit_horizontal_dir_xy = pred['major_dir_xy']
        else:
            orbit_horizontal_dir_xy = pred['minor_dir_xy']
    else:
        target_instance_local = scene_context.get("target_instance", None)
        scene_instances_local = scene_context.get("scene_instances", [])
        camera_local = scene_context.get("camera_info", None)
        image_shape_local = scene_context.get("image_shape", None)
        pred = predict_best_view_sequence_scene_aware(
            pts0_w,
            normals_w,
            curv,
            uncertainty,
            grasp_heat,
            scene_context=scene_context,
            T_wc_center=T_wc_center,
            orbit_angle_rad=joint0_delta_rad,
            candidate_num=16,
            min_disparity_deg=float(os.environ.get("ACTIVE_NBV_MIN_DISPARITY_DEG", "60.0")),
            max_disparity_deg=270.0,
            debug=debug,
            target_instance=target_instance_local,
            scene_instances_local=scene_instances_local,
            camera=camera_local,
            image_shape=image_shape_local,
            preferred_dirs_xy=manual_view_dirs_xy,
            reachability_fn=_make_camera_reachability_fn(env, cam_name),
        )
        best_pair = pred["best_pair_name"]
        if int(pred.get("reachable_candidate_count", 1)) == 0 and os.environ.get("ACTIVE_NBV_SKIP_IF_NO_REACHABLE", "1").strip().lower() not in {"0", "false", "no", "off"}:
            print("[ActiveNBV][reach] SKIP: no IK-reachable active-view candidate; keep center cloud for grasp planning.")
            return cloud_center_cam, pred
        orbit_horizontal_dir_xy = pred["primary_view"]["dir_xy"]

    fused = fuse_three_views_by_bbox(
        env,
        cam_name=cam_name,
        q_center=q_center,
        T_wc_center=T_wc_center,
        cloud_center_cam=cloud_center_cam,
        fovy_rad=fovy_rad,
        target_class=target_class,
        joint0_delta_rad=joint0_delta_rad,
        base_yaw_idx=0,
        wrist_yaw_idx=4,
        enable_wrist_comp=True,
        bbox_margin=bbox_margin,
        max_dist_to_centroid=max_dist_to_centroid,
        voxel_size=voxel_size,
        icp_refine=icp_refine,
        icp_voxel=icp_voxel,
        icp_thr_list=icp_thr_list,
        icp_max_iter=icp_max_iter,
        nn_gate_dist=nn_gate_dist,
        verify=verify,
        save_debug_dir=save_debug_dir,
        debug=debug,
        top_mask=top_mask,
        top_depth=top_depth,
        orbit_horizontal_dir_xy=orbit_horizontal_dir_xy,
        custom_view_dirs_xy=(
            [rec["dir_xy"].astype(np.float32) for rec in pred.get("selected_view_records", [])]
            if scene_context is not None else manual_view_dirs_xy
        ),
        custom_view_records=(pred.get("selected_view_records", None) if scene_context is not None else None),
        target_instance=(scene_context.get("target_instance", None) if scene_context is not None else None),
        scene_context=scene_context,
        fusion_mode=fusion_mode,
    )
    return fused, pred


def _select_target_instance(instances):
    targets = [inst for inst in instances if str(inst.get("role", "unknown")) == "target"]
    if len(targets) > 0:
        return targets[-1]
    for inst in instances:
        if str(inst.get("label", "")).lower() == "target":
            return inst
    return None


def _merge_scene_instances(base_instances, extra_instances):
    merged = [dict(inst) for inst in base_instances]
    extra_instances = [dict(inst) for inst in extra_instances]
    if any(str(inst.get("role", "unknown")) == "target" for inst in extra_instances):
        merged = [inst for inst in merged if str(inst.get("role", "unknown")) != "target"]
    seen_ids = {int(inst.get("id", -1)) for inst in merged}
    for inst in extra_instances:
        inst_id = int(inst.get("id", -1))
        if inst_id in seen_ids:
            merged = [dict(inst) if int(cur.get("id", -1)) == inst_id else cur for cur in merged]
        else:
            merged.append(dict(inst))
            seen_ids.add(inst_id)
    return merged


def _as_explicit_clear_target_instance(instance, chosen=None, fallback_center_w=None):
    inst = dict(instance)
    if chosen is None:
        chosen = {}
    inst["role"] = "target"
    inst["clear_target"] = True
    inst["selected_for_clear_execution"] = True
    if "label" not in inst or str(inst.get("label", "")).strip() == "":
        inst["label"] = str(chosen.get("label", "front_occluder"))
    if "observed_label" not in inst and chosen.get("observed_label", None) is not None:
        inst["observed_label"] = str(chosen.get("observed_label"))
    chosen_score = float(chosen.get("score", 0.0))
    inst_score = float(inst.get("score", 0.0))
    if chosen_score > inst_score:
        inst["score"] = chosen_score
        inst["front_occ_score"] = chosen_score
    if chosen.get("id", None) is not None:
        inst["front_occluder_id"] = int(chosen.get("id"))
    if chosen.get("source", None) is not None:
        inst["front_occluder_source"] = str(chosen.get("source"))
    for key in (
        "front_ratio",
        "target_near_ratio",
        "nearest_dist",
        "amodal_overlap_coverage",
        "amodal_hidden_covered_by_inst",
        "front_point_count",
    ):
        if key in chosen and key not in inst:
            inst[key] = chosen[key]
    if inst.get("bbox_center", None) is None and fallback_center_w is not None:
        inst["bbox_center"] = np.asarray(fallback_center_w, dtype=np.float32).reshape(3,)
    return inst


def _build_grasp_cloud_for_instance(instance, T_wc_center, fallback_cloud=None):
    pts = np.asarray(instance.get("points", np.zeros((0, 3), dtype=np.float32)), dtype=np.float32).reshape(-1, 3)
    fused_extract = np.zeros((0, 3), dtype=np.float32)
    if pts.shape[0] > 0:
        pts_filtered = _extract_instance_points_adaptive(
            instance,
            pts,
            strict_params=dict(bbox_margin=0.024, max_seed_radius=0.20, cluster_eps=0.016, min_cluster_points=16),
            loose_params=dict(bbox_margin=0.050, max_seed_radius=0.30, cluster_eps=0.022, min_cluster_points=8),
            min_keep_ratio=0.15,
            min_keep_points=220,
            debug_label="GraspCloud",
        )
        if pts_filtered.shape[0] >= max(220, int(0.12 * pts.shape[0])):
            pts = pts_filtered
    if fallback_cloud is not None:
        pts_fb_c = np.asarray(fallback_cloud.points, dtype=np.float32).reshape(-1, 3)
        if pts_fb_c.shape[0] > 0:
            pts_fb_w = _cam_to_world_points(T_wc_center, pts_fb_c)
            fused_extract = _extract_instance_points_adaptive(
                instance,
                pts_fb_w,
                strict_params=dict(bbox_margin=0.026, max_seed_radius=0.22, cluster_eps=0.016, min_cluster_points=14),
                loose_params=dict(bbox_margin=0.045, max_seed_radius=0.28, cluster_eps=0.020, min_cluster_points=8),
                min_keep_ratio=0.06,
                min_keep_points=600,
                debug_label="GraspCloudFused",
            )
            if fused_extract.shape[0] > 0:
                if pts.shape[0] == 0:
                    pts = fused_extract
                else:
                    use_fused = (
                        pts.shape[0] < 1200 and fused_extract.shape[0] >= 800
                    ) or (
                        fused_extract.shape[0] > int(1.35 * pts.shape[0])
                    )
                    if use_fused:
                        pts_merge = np.concatenate([pts, fused_extract], axis=0)
                        pcd_merge = o3d.geometry.PointCloud()
                        pcd_merge.points = o3d.utility.Vector3dVector(pts_merge.astype(np.float64))
                        pcd_merge = pcd_merge.voxel_down_sample(voxel_size=0.0010)
                        pts = np.asarray(pcd_merge.points, dtype=np.float32)
                        print(
                            f"[GraspCloud] merge instance+fused "
                            f"instance_pts={pts_filtered.shape[0] if 'pts_filtered' in locals() else pts.shape[0]} "
                            f"fused_pts={fused_extract.shape[0]} merged={pts.shape[0]}"
                        )
    if pts.shape[0] >= 300:
        return build_o3d_cloud_from_world_points(pts, T_wc_center, voxel_size=0.0008)
    return fallback_cloud


def _build_collision_cloud_for_grasp(scene_context, T_wc_center, target_cloud=None):
    pts_world_parts = []
    if scene_context is not None:
        ctx_pts = np.asarray(
            scene_context.get("context_points_w", scene_context.get("obstacle_points_w", np.zeros((0, 3), dtype=np.float32))),
            dtype=np.float32,
        ).reshape(-1, 3)
        if ctx_pts.shape[0] > 0:
            pts_world_parts.append(ctx_pts)
    if target_cloud is not None:
        pts_t_c = np.asarray(target_cloud.points, dtype=np.float32).reshape(-1, 3)
        if pts_t_c.shape[0] > 0:
            pts_world_parts.append(_cam_to_world_points(T_wc_center, pts_t_c))
    if len(pts_world_parts) == 0:
        return target_cloud
    pts_world = np.concatenate(pts_world_parts, axis=0).astype(np.float32)
    return build_o3d_cloud_from_world_points(pts_world, T_wc_center, color_rgb=(0.7, 0.7, 0.7), voxel_size=0.0012)


def _clear_cuda_cache():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass
    gc.collect()


def _grasp_num_point_schedule(cloud_point_count: int, base_num_point: int):
    base = int(max(900, base_num_point))
    if cloud_point_count >= 16000:
        first = min(base, 2600)
    elif cloud_point_count >= 12000:
        first = min(base, 3000)
    elif cloud_point_count >= 8000:
        first = min(base, 3600)
    else:
        first = min(base, 4200)
    schedule = []
    for n in (first, min(first, 2200), 1600, 1200, 900):
        n = int(max(900, n))
        if n not in schedule:
            schedule.append(n)
    return schedule


def _choose_grasp_from_cloud_candidates(cloud_candidates,
                                        env,
                                        T_wc_cap,
                                        vertical_cam,
                                        scene_context=None,
                                        target_center_w=None,
                                        display_bgr=None,
                                        display_camera=None,
                                        display_cloud=None,
                                        collision_cloud=None,
                                        angle_deg: float = 65.0,
                                        num_point: int = 3000,
                                        visual: bool = True,
                                        debug_label: str = "TargetGrasp",
                                        manual_select: bool = True,
                                        manual_topk: int = 8):
    """Try multiple point-cloud candidates for grasp generation from most specific to broadest."""
    last_err = None
    for name, cloud in cloud_candidates:
        if cloud is None:
            continue
        pts = np.asarray(cloud.points, dtype=np.float32)
        if pts.shape[0] == 0:
            continue
        print(f"[{debug_label}] try cloud='{name}' points={pts.shape[0]}")
        try:
            release_vision_model_caches()
        except Exception:
            pass
        sample_schedule = _grasp_num_point_schedule(pts.shape[0], int(num_point))
        for sample_num in sample_schedule:
            try:
                print(f"[{debug_label}] cloud='{name}' sample_num={sample_num}")
                _clear_cuda_cache()
                end_points, cloud_for_collision = make_end_points_from_o3d_cloud(cloud, num_point=int(sample_num))
                gg, candidate_rows = generate_grasps(
                    end_points,
                    cloud_for_collision,
                    env=env,
                    T_wc_cap=T_wc_cap,
                    collision_cloud=collision_cloud,
                    scene_context=scene_context,
                    target_center_w=target_center_w,
                    display_bgr=display_bgr,
                    display_camera=display_camera,
                    display_cloud=display_cloud if display_cloud is not None else cloud_for_collision,
                    visual=visual,
                    vertical_cam=vertical_cam,
                    angle_deg=float(angle_deg),
                    manual_select=manual_select,
                    manual_topk=int(manual_topk),
                )
                return gg, name, cloud_for_collision, candidate_rows
            except RuntimeError as e:
                last_err = e
                print(f"[{debug_label}] cloud='{name}' sample_num={sample_num} failed: {e}")
                if "out of memory" in str(e).lower():
                    _clear_cuda_cache()
                    continue
                break
    if last_err is not None:
        raise last_err
    raise RuntimeError(f"[{debug_label}] no valid point cloud candidate for grasp generation.")


def _sanitize_grasp_rotation(gg):
    R_go = np.asarray(gg.rotation_matrices[0], dtype=float)
    if R_go.shape == (9,):
        R_go = R_go.reshape(3, 3)
    if R_go.shape != (3, 3):
        raise ValueError(f"[Grasp] invalid grasp rotation shape: {R_go.shape}")
    if not np.isfinite(R_go).all():
        raise ValueError("[Grasp] grasp rotation contains NaN/Inf")
    U, _, Vt = np.linalg.svd(R_go)
    R_go = U @ Vt
    if np.linalg.det(R_go) < 0:
        U[:, -1] *= -1
        R_go = U @ Vt
    return R_go


def _grasp_to_command_poses(env, grasp, T_wc_cap=None, pregrasp_offset: float = 0.08):
    robot = env.robot
    _, to_cmd, _ = _build_frame_converters(env)

    if T_wc_cap is None:
        T_wc = get_T_wc_from_camera(env, EE_CAM_NAME)
    else:
        T_wc = T_wc_cap

    if hasattr(grasp, "rotation_matrix"):
        R_co = np.asarray(grasp.rotation_matrix, dtype=float).reshape(3, 3)
        t_co = np.asarray(grasp.translation, dtype=float).reshape(3,)
    else:
        R_co = np.asarray(grasp.rotation_matrices[0], dtype=float)
        if R_co.shape == (9,):
            R_co = R_co.reshape(3, 3)
        t_co = np.asarray(grasp.translations[0], dtype=float).reshape(3,)

    if not np.isfinite(R_co).all() or not np.isfinite(t_co).all():
        raise ValueError("[grasp-ik] grasp pose contains NaN/Inf")

    U, _, Vt = np.linalg.svd(R_co)
    R_co = U @ Vt
    if np.linalg.det(R_co) < 0:
        U[:, -1] *= -1
        R_co = U @ Vt

    T_co = sm.SE3(sm.SO3(R_co))
    T_co.t = t_co
    T_wo_world = T_wc * T_co

    nero_topdown = getattr(robot, "dof", 0) == 7 and os.environ.get("NERO_GRASP_TOPDOWN", "1").strip().lower() not in {"0", "false", "no", "off"}
    if nero_topdown:
        # GraspNet poses were tuned around the old gripper/frame. Nero's TCP and
        # wrist reachability are more stable if TCP-Z points vertically down.
        # Nero's finger opening direction is TCP-Y, so preserve the horizontal
        # component of GraspNet's gripper-width axis as TCP-Y.
        z_axis = np.array([0.0, 0.0, -1.0], dtype=float)
        y_hint = np.asarray(T_wo_world.R, dtype=float)[:, 1]
        y_axis = np.array([y_hint[0], y_hint[1], 0.0], dtype=float)
        if np.linalg.norm(y_axis) < 1e-6 or os.environ.get("NERO_GRASP_PRESERVE_YAW", "1").strip().lower() in {"0", "false", "no", "off"}:
            y_axis = np.array([0.0, -1.0, 0.0], dtype=float)
        y_axis = y_axis / (np.linalg.norm(y_axis) + 1e-9)
        x_axis = np.cross(y_axis, z_axis)
        x_axis = x_axis / (np.linalg.norm(x_axis) + 1e-9)
        R_topdown = np.column_stack([x_axis, y_axis, z_axis]).astype(float)
        U, _, Vt = np.linalg.svd(R_topdown)
        R_topdown = U @ Vt
        if np.linalg.det(R_topdown) < 0:
            U[:, -1] *= -1
            R_topdown = U @ Vt
        T_wo_topdown = sm.SE3(sm.SO3(R_topdown))
        T_wo_topdown.t = np.asarray(T_wo_world.t, dtype=float).reshape(3,)
        T_wo_world = T_wo_topdown
        grasp_advance = float(os.environ.get("NERO_GRASP_ADVANCE", "0.040"))
        if abs(grasp_advance) > 1e-6:
            T_wo_world = T_wo_world * sm.SE3(0.0, 0.0, grasp_advance)
        if os.environ.get("NERO_APPLY_MIN_GRASP_Z_IN_POSE", "1").strip().lower() not in {"0", "false", "no", "off"}:
            min_grasp_z = float(os.environ.get("NERO_MIN_GRASP_Z", "0.755"))
            if float(T_wo_world.t[2]) < min_grasp_z:
                T_wo_world = sm.SE3.Trans(0.0, 0.0, min_grasp_z - float(T_wo_world.t[2])) * T_wo_world

    T_wo = to_cmd(T_wo_world)
    if nero_topdown:
        T_pre = T_wo * sm.SE3(0.0, 0.0, -float(pregrasp_offset))
    else:
        T_pre = T_wo * sm.SE3(-float(pregrasp_offset), 0.0, 0.0)
    return T_wo, T_pre, T_wo_world


def _grasp_is_ik_reachable(env, grasp, T_wc_cap=None, pregrasp_offset: float = 0.08):
    robot = env.robot
    q_saved = np.array(robot.get_joint(), dtype=float)
    try:
        offsets = [float(os.environ.get("NERO_PREGRASP_OFFSET", "0.06"))] if getattr(robot, "dof", 0) == 7 else [float(pregrasp_offset)]
        last_reason = "pregrasp_ik_failed"
        for offset in offsets:
            T_wo, T_pre, _ = _grasp_to_command_poses(env, grasp, T_wc_cap=T_wc_cap, pregrasp_offset=offset)
            robot.set_joint(q_saved)
            q_pre = _solve_ik_for_pose(env, T_pre, q_seed=q_saved)
            if q_pre.size == 0 or not np.isfinite(q_pre).all():
                last_reason = "pregrasp_ik_failed"
                continue
            robot.set_joint(q_pre)
            q_grasp = _solve_ik_for_pose(env, T_wo, q_seed=q_pre)
            if q_grasp.size == 0 or not np.isfinite(q_grasp).all():
                last_reason = "grasp_ik_failed"
                continue
            return True, {
                "reason": "ok",
                "q_pre": q_pre,
                "q_grasp": q_grasp,
                "pregrasp_offset": float(offset),
                "max_joint_jump": float(np.max(np.abs(q_grasp - q_pre))),
            }
        return False, {"reason": last_reason}
    except Exception as e:
        return False, {"reason": type(e).__name__}
    finally:
        robot.set_joint(q_saved)


def _front_dir_world_from_grasp(gg, T_wc):
    R_go = _sanitize_grasp_rotation(gg)
    front_dir_w = -np.asarray(np.asarray(T_wc.R, dtype=float) @ R_go[:, 2], dtype=float).reshape(3,)
    fdn = np.linalg.norm(front_dir_w)
    if fdn > 1e-9:
        front_dir_w = front_dir_w / fdn
    return front_dir_w.astype(np.float32)


def _front_dir_world_from_single_grasp(grasp, T_wc):
    if hasattr(grasp, "rotation_matrix"):
        R_go = np.asarray(grasp.rotation_matrix, dtype=float)
    else:
        R_go = np.asarray(grasp.rotation_matrices[0], dtype=float)
    if R_go.shape == (9,):
        R_go = R_go.reshape(3, 3)
    U, _, Vt = np.linalg.svd(R_go)
    R_go = U @ Vt
    if np.linalg.det(R_go) < 0:
        U[:, -1] *= -1
        R_go = U @ Vt
    front_dir_w = -np.asarray(np.asarray(T_wc.R, dtype=float) @ R_go[:, 2], dtype=float).reshape(3,)
    fdn = np.linalg.norm(front_dir_w)
    if fdn > 1e-9:
        front_dir_w = front_dir_w / fdn
    return front_dir_w.astype(np.float32)


def _collect_obstacle_dirs_for_target(scene_context, target_center_w):
    if scene_context is None or target_center_w is None:
        return []
    center = np.asarray(target_center_w, dtype=np.float32).reshape(3,)
    obstacle_dirs = []
    for inst in scene_context.get("scene_instances", []):
        role = str(inst.get("role", "unknown")).lower()
        label = str(inst.get("label", "unknown")).lower()
        if role not in {"obstacle", "unknown"}:
            continue
        if label == "geometry_fallback":
            continue
        bbox_center = inst.get("bbox_center", None)
        if bbox_center is None:
            continue
        v = np.asarray(bbox_center, dtype=np.float32).reshape(3,) - center
        v[2] = 0.0
        nv = float(np.linalg.norm(v))
        if nv < 1e-6:
            continue
        obstacle_dirs.append((v / nv).astype(np.float32))
    return obstacle_dirs


def _prompt_user_to_select_grasp(candidate_rows,
                                 debug_label: str,
                                 display_bgr=None,
                                 display_camera=None,
                                 T_wc_cap=None,
                                 display_cloud=None,
                                 show_pointcloud: bool = True):
    if len(candidate_rows) == 0:
        raise RuntimeError(f"[{debug_label}] no candidate grasp to select.")
    window_name = f"{debug_label} Candidates"
    overlay_img = None
    overlay_hits = []
    overlay_ok = False
    if show_pointcloud and display_cloud is not None:
        _show_grasp_candidate_pointcloud(display_cloud, candidate_rows, window_name=f"{debug_label} PointCloud")
    if display_bgr is not None and display_camera is not None and T_wc_cap is not None:
        overlay_img, overlay_hits = _show_grasp_candidate_overlay(
            display_bgr,
            display_camera,
            T_wc_cap,
            candidate_rows,
            window_name=window_name,
        )
        overlay_ok = overlay_img is not None
    print(f"[{debug_label}] ranked grasp candidates:")
    for row in candidate_rows:
        print(
            f"  [{row['idx']}] priority={row['priority']:.4f} "
            f"vertical_up={row['vertical_align']:.4f} "
            f"center_dist={row['center_dist']:.4f} "
            f"obs_avoid={row['obstacle_score']:.4f} "
            f"ik_jump={row['ik_jump']:.4f} "
            f"score={row['raw_score']:.4f} "
            f"t={np.round(row['translation'], 4)}"
        )
    default_idx = int(candidate_rows[0]["idx"])

    if overlay_ok:
        clicked = {"idx": None}

        def _cb(event, x, y, flags, param):
            if event != cv2.EVENT_LBUTTONDOWN:
                return
            for hit in overlay_hits:
                x0, y0, x1, y1 = hit["bbox"]
                if x0 <= x <= x1 and y0 <= y <= y1:
                    clicked["idx"] = int(hit["idx"])
                    return

        try:
            cv2.setMouseCallback(window_name, _cb)
        except cv2.error:
            overlay_ok = False

        print(
            f"[{debug_label}] click a numbered label / grasp point, or press digit 0-{len(candidate_rows)-1}; "
            f"Enter/Esc uses default {default_idx}."
        )
        while True:
            try:
                key = cv2.waitKey(30)
            except cv2.error:
                overlay_ok = False
                break
            if clicked["idx"] is not None:
                try:
                    cv2.destroyWindow(window_name)
                except cv2.error:
                    pass
                return int(clicked["idx"])
            if key in (13, 10):
                try:
                    cv2.destroyWindow(window_name)
                except cv2.error:
                    pass
                return default_idx
            if key == 27:
                try:
                    cv2.destroyWindow(window_name)
                except cv2.error:
                    pass
                return default_idx
            if ord('0') <= key <= ord('9'):
                idx = int(key - ord('0'))
                if 0 <= idx < len(candidate_rows):
                    try:
                        cv2.destroyWindow(window_name)
                    except cv2.error:
                        pass
                    return idx
                print(f"[{debug_label}] index out of range: {idx}")

    if overlay_img is not None and not overlay_ok:
        fallback_path = os.path.join(ROOT_DIR, "dbg_grasp_select.png")
        cv2.imwrite(fallback_path, overlay_img)
        print(f"[{debug_label}] OpenCV window unavailable; saved overlay to {fallback_path}")

    while True:
        raw = input(f"[{debug_label}] select grasp index [0-{len(candidate_rows)-1}] (Enter={default_idx}): ").strip()
        if raw == "":
            try:
                cv2.destroyWindow(window_name)
            except cv2.error:
                pass
            return default_idx
        try:
            idx = int(raw)
        except ValueError:
            print(f"[{debug_label}] invalid input: {raw}")
            continue
        if 0 <= idx < len(candidate_rows):
            try:
                cv2.destroyWindow(window_name)
            except cv2.error:
                pass
            return idx
        print(f"[{debug_label}] index out of range: {idx}")


def _show_grasp_candidate_overlay(display_bgr,
                                  camera,
                                  T_wc_cap,
                                  candidate_rows,
                                  window_name: str = "Grasp Candidates"):
    if display_bgr is None or camera is None or T_wc_cap is None or len(candidate_rows) == 0:
        return None, []
    img = np.asarray(display_bgr, dtype=np.uint8).copy()
    h, w = img.shape[:2]
    hit_regions = []
    colors = [
        (0, 255, 255), (0, 220, 0), (255, 180, 0), (255, 80, 80),
        (180, 80, 255), (0, 160, 255), (255, 0, 200), (80, 255, 120),
    ]
    pts_world = []
    dirs_world = []
    for row in candidate_rows:
        grasp = row["grasp"]
        if hasattr(grasp, "rotation_matrix"):
            R_co = np.asarray(grasp.rotation_matrix, dtype=float)
            t_co = np.asarray(grasp.translation, dtype=float).reshape(3,)
        else:
            R_co = np.asarray(grasp.rotation_matrices[0], dtype=float)
            t_co = np.asarray(grasp.translations[0], dtype=float).reshape(3,)
        if R_co.shape == (9,):
            R_co = R_co.reshape(3, 3)
        U, _, Vt = np.linalg.svd(R_co)
        R_co = U @ Vt
        if np.linalg.det(R_co) < 0:
            U[:, -1] *= -1
            R_co = U @ Vt
        T_co = sm.SE3.Rt(R_co, t_co)
        T_wo_world = T_wc_cap * T_co
        pts_world.append(np.asarray(T_wo_world.t, dtype=np.float32).reshape(3,))
        dirs_world.append(_front_dir_world_from_single_grasp(grasp, T_wc_cap))
    pts_world = np.asarray(pts_world, dtype=np.float32).reshape(-1, 3)
    uv, valid = _project_world_points_to_image(pts_world, T_wc_cap, camera)
    arrow_world = pts_world + 0.05 * np.asarray(dirs_world, dtype=np.float32).reshape(-1, 3)
    uv2, valid2 = _project_world_points_to_image(arrow_world, T_wc_cap, camera)

    for i, row in enumerate(candidate_rows):
        if not bool(valid[i]):
            continue
        x = int(np.clip(round(float(uv[i, 0])), 0, w - 1))
        y = int(np.clip(round(float(uv[i, 1])), 0, h - 1))
        color = colors[i % len(colors)]
        cv2.circle(img, (x, y), 7, color, 2, lineType=cv2.LINE_AA)
        if bool(valid2[i]):
            x2 = int(np.clip(round(float(uv2[i, 0])), 0, w - 1))
            y2 = int(np.clip(round(float(uv2[i, 1])), 0, h - 1))
            cv2.arrowedLine(img, (x, y), (x2, y2), color, 2, tipLength=0.28, line_type=cv2.LINE_AA)
        text = str(row["idx"])
        (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        tx0 = int(np.clip(x + 10, 0, max(w - tw - 6, 0)))
        ty0 = int(np.clip(y - 10, th + 6, h - 2))
        box0 = (tx0 - 3, ty0 - th - 3)
        box1 = (tx0 + tw + 3, ty0 + baseline + 3)
        cv2.rectangle(img, box0, box1, (0, 0, 0), -1)
        cv2.putText(img, text, (tx0, ty0), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, lineType=cv2.LINE_AA)
        hit_regions.append({
            "idx": int(row["idx"]),
            "bbox": (
                int(max(min(box0[0], x - 10), 0)),
                int(max(min(box0[1], y - 10), 0)),
                int(min(max(box1[0], x + 10), w - 1)),
                int(min(max(box1[1], y + 10), h - 1)),
            ),
        })
    cv2.putText(img, "Press 0-9 to select, Enter/Esc for default", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, lineType=cv2.LINE_AA)
    cv2.putText(img, "Press 0-9 to select, Enter/Esc for default", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 1, lineType=cv2.LINE_AA)
    cv2.putText(img, "Or click a label / grasp point", (12, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, lineType=cv2.LINE_AA)
    cv2.putText(img, "Or click a label / grasp point", (12, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 1, lineType=cv2.LINE_AA)
    try:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.imshow(window_name, img)
        cv2.waitKey(1)
    except cv2.error:
        return img, hit_regions
    return img, hit_regions


def _show_grasp_candidate_pointcloud(cloud, candidate_rows, window_name: str = "Grasp Candidates 3D"):
    if cloud is None or len(candidate_rows) == 0:
        return
    colors = [
        [1.0, 1.0, 0.0], [0.0, 0.86, 0.0], [1.0, 0.7, 0.0], [1.0, 0.3, 0.3],
        [0.7, 0.3, 1.0], [0.0, 0.65, 1.0], [1.0, 0.0, 0.8], [0.3, 1.0, 0.55],
    ]
    geoms = [cloud]
    for i, row in enumerate(candidate_rows):
        grasp = row["grasp"]
        color = colors[i % len(colors)]
        try:
            gripper = grasp.to_open3d_geometry()
            gripper.paint_uniform_color(color)
            geoms.append(gripper)
        except Exception:
            pass
        try:
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.006)
            sphere.translate(np.asarray(row["translation"], dtype=float).reshape(3,))
            sphere.paint_uniform_color(color)
            geoms.append(sphere)
        except Exception:
            pass
    print(f"[GraspSelect] point-cloud candidate colors: " + ", ".join([f"{row['idx']}->{i % len(colors)}" for i, row in enumerate(candidate_rows)]))
    try:
        o3d.visualization.draw_geometries(geoms, window_name=window_name)
    except Exception as e:
        print(f"[GraspSelect] point-cloud candidate visualization failed: {type(e).__name__}: {e}")


def _build_scene_state_from_center_view(env,
                                        target_class,
                                        fovy_cam,
                                        iter_tag,
                                        debug_dir,
                                        allow_click_init=True,
                                        context_classes=None,
                                        target_center_hint_w=None):
    if context_classes is None:
        context_classes = _default_context_classes()
    if VERBOSE_LOG:
        print(f"[Context] classes={sorted(str(x) for x in context_classes)}")

    imgs0 = render_from_camera(env, EE_CAM_NAME, depth_mode='auto', debug=VERBOSE_LOG)
    rgb0 = imgs0['img']
    depth0 = imgs0['depth']
    if VERBOSE_LOG:
        print("depth dtype:", depth0.dtype, "min/max:", float(np.min(depth0)), float(np.max(depth0)))
    T_wc0 = get_T_wc_from_camera(env, EE_CAM_NAME)
    bgr0 = cv2.cvtColor(rgb0, cv2.COLOR_RGB2BGR)
    if debug_dir is not None:
        save_debug_image(
            bgr0,
            _debug_iter_path(debug_dir, iter_tag, "center_rgb"),
            title=f"{iter_tag}: center_rgb",
        )
        if target_class:
            save_yolo_detection_debug_overlay(
                bgr0,
                _debug_iter_path(debug_dir, iter_tag, "target_yolo_overlay"),
                target_class=target_class,
                conf_thresh=0.15,
                title=f"{iter_tag}: target_yolo={target_class}",
            )
    try:
        mask0 = segment_image(
            bgr0,
            output_mask='mask_center.png',
            target_class=target_class,
            interactive=False,
            allow_manual_click=bool(target_center_hint_w is None or _env_flag("CENTER_YOLO_ALLOW_MANUAL_CLICK", "0")),
        )
    except Exception as e:
        if target_center_hint_w is None or not _env_flag("CENTER_MASK_FALLBACK_FROM_HINT", "1"):
            raise
        mask0 = _fallback_mask_from_center_hint(
            depth0,
            T_wc0,
            fovy_cam,
            target_center_hint_w,
            min_pixels=int(os.environ.get("CENTER_FALLBACK_MASK_MIN_PIXELS", "80")),
        )
        if mask0 is None:
            raise
        print(
            f"[CenterMaskFallback] YOLO center mask failed ({type(e).__name__}: {e}); "
            f"use 3D center hint mask pixels={int(np.count_nonzero(mask0))} "
            f"center_w={np.round(np.asarray(target_center_hint_w, dtype=float).reshape(3,), 4)}"
        )
    if debug_dir is not None and mask0 is not None:
        cv2.imwrite(_debug_iter_path(debug_dir, iter_tag, "target_mask"), np.asarray(mask0, dtype=np.uint8))
    _, cloud_center = get_and_process_data(bgr0, depth0, mask0, fovy_rad=fovy_cam)
    context_dets = detect_context_instances_center_view(
        image_or_path=bgr0,
        target_mask=mask0,
        conf_thresh=0.15,
        context_classes=context_classes,
        output_vis_path=None if debug_dir is None else _debug_iter_path(debug_dir, iter_tag, "context_yolo_overlay"),
    )
    if _env_flag("CONTEXT_FORCE_TBLOCK_DETECTION", "0"):
        try:
            tblock_dets = detect_context_instances_center_view(
                image_or_path=bgr0,
                target_mask=np.zeros_like(np.asarray(mask0, dtype=np.uint8)),
                conf_thresh=float(os.environ.get("CONTEXT_TBLOCK_CONF", "0.08")),
                context_classes={"t_block", "t-block", "tblock"},
                output_vis_path=None if debug_dir is None else _debug_iter_path(debug_dir, iter_tag, "context_tblock_yolo_overlay"),
                exclude_target_overlap=1.01,
            )
            if len(tblock_dets) > 0:
                before_n = len(context_dets)
                context_dets = _merge_context_detections(
                    context_dets,
                    tblock_dets,
                    iou_thresh=float(os.environ.get("CONTEXT_TBLOCK_MERGE_IOU", "0.60")),
                )
                print(f"[Context][tblock] supplemental detections={len(tblock_dets)} merged={len(context_dets) - before_n}")
        except Exception as exc:
            print(f"[Context][tblock] supplemental detection skipped: {type(exc).__name__}: {exc}")
    print(f"[Context] detections={len(context_dets)}")
    for i, det in enumerate(context_dets[:12]):
        label_i = str(det.get("label", "unknown"))
        if VERBOSE_LOG or label_i.lower().replace("-", "_") in {"t_block", "tblock"}:
            print(
                f"[Context] det[{i}] label={label_i} score={float(det.get('score', 0.0)):.3f} "
                f"area={int(np.count_nonzero(det.get('mask', np.zeros((0,), dtype=np.uint8))))} "
                f"xyxy={np.round(np.asarray(det.get('xyxy', []), dtype=float), 1).tolist() if det.get('xyxy', None) is not None else None}"
            )

    scene_context = build_scene_context_from_center_view(
        bgr0=bgr0,
        depth0=depth0,
        target_mask=mask0,
        context_dets=context_dets,
        T_wc0=T_wc0,
        fovy_rad=fovy_cam,
        debug=VERBOSE_LOG,
    )
    built_instances, camera0 = build_scene_instances_from_view(mask0, context_dets, depth0, T_wc0, fovy_cam)
    extra_instances = []
    if allow_click_init:
        if len(context_dets) == 0:
            print("[ClickInit] YOLO context detections empty. You can add instances by click.")
        extra_instances = maybe_collect_click_instances(bgr0, depth0, camera0, T_wc0, recognized_instances=built_instances)
    all_instances = _merge_scene_instances(built_instances, extra_instances)
    scene_context = inject_scene_instances_into_context(scene_context, extra_instances)

    save_scene_instance_debug_overlay(
        bgr0,
        [_select_target_instance(all_instances)] if _select_target_instance(all_instances) is not None else [],
        T_wc0,
        camera0,
        _debug_iter_path(debug_dir, iter_tag, "target_detect_overlay"),
        title=f"{iter_tag}: target",
        image_shape=bgr0.shape[:2],
    )
    save_scene_instance_debug_overlay(
        bgr0,
        all_instances,
        T_wc0,
        camera0,
        _debug_iter_path(debug_dir, iter_tag, "scene_instances_overlay"),
        title=f"{iter_tag}: scene",
        image_shape=bgr0.shape[:2],
    )
    target_instance = _select_target_instance(all_instances)
    if target_instance is not None:
        scene_context["target_instance"] = target_instance
    scene_context["scene_instances"] = list(all_instances)
    print(f"[SceneContext] context_points = {scene_context['context_points_w'].shape}")
    print(f"[SceneContext] num_instances = {len(all_instances)}")
    if VERBOSE_LOG:
        print(f"[SceneContext] instances = {scene_context['context_instances']}")
    else:
        print(f"[SceneContext] instance_summary = {[_brief_instance(x) for x in all_instances[:6]]}")
    return {
        "rgb": rgb0,
        "bgr": bgr0,
        "depth": depth0,
        "mask": mask0,
        "T_wc": T_wc0,
        "cloud_center": cloud_center,
        "scene_context": scene_context,
        "scene_instances": all_instances,
        "target_instance": target_instance,
        "camera": camera0,
    }


def _recenter_scene_state_to_target(env,
                                    scene_state,
                                    target_class,
                                    fovy_cam,
                                    iter_tag,
                                    debug_dir,
                                    hover_height):
    target_instance = scene_state.get("target_instance", None)
    if target_instance is None:
        return scene_state
    click_extras = [dict(inst) for inst in scene_state.get("scene_instances", []) if inst.get("click_uv", None) is not None]
    object_center_w = np.asarray(
        scene_state.get("scene_context", {}).get("target_center_w", target_instance["bbox_center"]),
        dtype=np.float32,
    ).reshape(3,)
    goto_object_centered_observation_pose(
        env,
        object_center_w=object_center_w,
        cam_name=EE_CAM_NAME,
        hover_height=hover_height,
        duration=0.8,
        settle_steps=60,
        debug=VERBOSE_LOG,
    )
    recentered = _build_scene_state_from_center_view(
        env,
        target_class=target_class,
        fovy_cam=fovy_cam,
        iter_tag=iter_tag,
        debug_dir=debug_dir,
        allow_click_init=False,
        target_center_hint_w=object_center_w,
    )
    if len(click_extras) > 0:
        recentered["scene_instances"] = _merge_scene_instances(recentered["scene_instances"], click_extras)
        recentered["scene_context"] = inject_scene_instances_into_context(recentered["scene_context"], click_extras)
        recentered["target_instance"] = _select_target_instance(recentered["scene_instances"])
        recentered["scene_context"]["scene_instances"] = list(recentered["scene_instances"])
    return recentered


def _run_target_multiview_cycle(env,
                                scene_state,
                                target_class,
                                fovy_cam,
                                orbit_angle_deg,
                                iter_tag,
                                multiview_debug_dir,
                                max_rounds: int = 2):
    target_instance = scene_state["target_instance"]
    scene_instances_local = list(scene_state["scene_instances"])
    scene_context_local = scene_state["scene_context"]
    T_wc0 = scene_state["T_wc"]
    cloud_center = scene_state["cloud_center"]
    mask0 = scene_state["mask"]
    depth0 = scene_state["depth"]
    q_center = np.array(env.robot.get_joint(), dtype=float)
    active_view_info = {"best_pair_name": "center_only"}
    fused_cloud = cloud_center

    if target_instance is None:
        raise RuntimeError("[TargetObs] target instance missing.")

    obs_info = assess_instance_observation(target_instance)
    print(
        f"[TargetObs] init visible_ratio={obs_info['visible_ratio']:.4f} "
        f"points={obs_info['num_points']} span={obs_info['bbox_span']:.4f} "
        f"well_observed={obs_info['well_observed']}"
    )
    round_limit = max(1, int(max_rounds))
    for nbv_round in range(round_limit):
        manual_dirs, occluders, _ = select_occlusion_aware_view_dirs(
            target_instance,
            scene_instances_local,
            np.asarray(T_wc0.t, dtype=np.float32).reshape(3,),
            top_k=2,
            debug=VERBOSE_LOG,
            T_wc=T_wc0,
            camera=scene_context_local.get("camera_info", None),
            image_shape=scene_context_local.get("image_shape", None),
        )
        print(f"[NBVLoop] round={nbv_round + 1}/{round_limit} manual_click_nbv={manual_dirs is not None} occluders={len(occluders)}")
        fused_cloud, active_view_info = fuse_active_views_by_bbox(
            env,
            cam_name=EE_CAM_NAME,
            q_center=q_center,
            T_wc_center=T_wc0,
            cloud_center_cam=cloud_center,
            fovy_rad=fovy_cam,
            target_class=target_class,
            joint0_delta_rad=np.deg2rad(orbit_angle_deg),
            bbox_margin=0.03,
            max_dist_to_centroid=0.20,
            voxel_size=0.0012,
            icp_refine=(os.environ.get("MV_ICP_REFINE", "1").strip().lower() not in {"0", "false", "no"}),
            icp_voxel=0.003,
            icp_thr_list=(0.06, 0.04, 0.025, 0.015),
            icp_max_iter=60,
            nn_gate_dist=0.02,
            verify=True,
            save_debug_dir=os.path.join(multiview_debug_dir, f"{iter_tag}_r{nbv_round+1}"),
            debug=VERBOSE_LOG,
            top_mask=mask0,
            top_depth=depth0,
            scene_context=scene_context_local,
            manual_view_dirs_xy=manual_dirs,
            fusion_mode=os.environ.get("MV_FUSION_MODE", "strict_overlap"),
        )
        fused_pts_w = _cam_to_world_points(T_wc0, np.asarray(fused_cloud.points, dtype=np.float32))
        target_instance = update_instance_with_new_view(target_instance, fused_pts_w)
        scene_instances_local = _replace_scene_instance(scene_instances_local, target_instance)
        click_extras = [dict(inst) for inst in scene_instances_local if inst.get("click_uv", None) is not None]
        scene_context_local = inject_scene_instances_into_context(scene_context_local, click_extras)
        scene_context_local["scene_instances"] = list(scene_instances_local)
        scene_context_local["target_instance"] = target_instance
        obs_info = assess_instance_observation(target_instance)
        print(
            f"[TargetObs] after round {nbv_round + 1}: visible_ratio={obs_info['visible_ratio']:.4f} "
            f"points={obs_info['num_points']} span={obs_info['bbox_span']:.4f} "
            f"well_observed={obs_info['well_observed']}"
        )
        if obs_info["well_observed"]:
            break

    grasp_cloud = _build_grasp_cloud_for_instance(target_instance, T_wc0, fallback_cloud=fused_cloud)
    if grasp_cloud is None:
        grasp_cloud = cloud_center
    collision_cloud = _build_collision_cloud_for_grasp(scene_context_local, T_wc0, target_cloud=grasp_cloud)
    scene_state["target_instance"] = target_instance
    scene_state["scene_instances"] = scene_instances_local
    scene_state["scene_context"] = scene_context_local
    scene_state["grasp_cloud"] = grasp_cloud
    scene_state["collision_cloud"] = collision_cloud
    scene_state["fused_cloud"] = fused_cloud
    scene_state["active_view_info"] = active_view_info
    return scene_state
# ================= 抓取推理网络 ====================
_GRASPNET_CACHE = None


def get_net():
    global _GRASPNET_CACHE
    if _GRASPNET_CACHE is not None:
        return _GRASPNET_CACHE
    # 初始化 GraspNet 网络结构
    net = GraspNet(
        input_feature_dim=0,  # 每个点输入的额外特征维度，这里为 0 表示只使用点云坐标信息（x, y, z），不附加颜色/法向等特征。
        num_view=300,  # 每个点云采样的候选视角数量，抓取点会从这些视角中选择最佳抓取方向。
        num_angle=12,  # 每个视角下候选抓取角度数量（绕视线旋转），例如将夹爪在不同角度放置。
        num_depth=4,  # 每个抓取点沿抓取方向的不同深度尝试数量，用于考虑抓取深浅。
        cylinder_radius=0.05,  # 用于局部点云分组的抓取圆柱半径，决定每个候选抓取点周围的邻域范围。
        hmin=-0.02,  # 抓取圆柱高度范围下界（负值表示可稍微延伸到点云下方），控制局部体素区域。
        hmax_list=[0.01, 0.02, 0.03, 0.04],  # 多个候选高度上界，对应不同抓取深度层，用于生成多尺度抓取候选。
        is_training=False  # 表示当前网络处于推理模式，而非训练模式，不会执行 dropout 等训练行为。
    )

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用，当前 GraspNet 推理必须在 GPU 上运行。")

    # 加载 checkpoint
    checkpoint_path = os.path.join(ROOT_DIR, 'logs/log_rs/checkpoint-rs.tar')
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    net.load_state_dict(checkpoint['model_state_dict'])

    # 切换到评估模式
    net.eval()
    _GRASPNET_CACHE = net
    return _GRASPNET_CACHE


def _prepare_graspnet_for_inference():
    net = get_net()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用，当前 GraspNet 推理必须在 GPU 上运行。")
    net.to(torch.device("cuda:0"))
    net.eval()
    return net


def _offload_graspnet():
    global _GRASPNET_CACHE
    if _GRASPNET_CACHE is None:
        return
    try:
        _GRASPNET_CACHE.to(torch.device("cpu"))
    except Exception:
        pass
    _clear_cuda_cache()

def _axis_arrows_at(origin, R, scale=0.06):
    """Create RGB axis arrows at origin with rotation R (3x3)."""
    origin = np.asarray(origin, dtype=float).reshape(3)
    R = np.asarray(R, dtype=float).reshape(3,3)

    def one_arrow(color_rgb, axis_vec):
        mesh = o3d.geometry.TriangleMesh.create_arrow(
            cylinder_radius=0.0035,
            cone_radius=0.006,
            cylinder_height=scale * 0.75,
            cone_height=scale * 0.25
        )
        mesh.compute_vertex_normals()
        # Arrow default points +Z; rotate +Z -> axis_vec
        z = np.array([0.0, 0.0, 1.0])
        a = axis_vec / (np.linalg.norm(axis_vec) + 1e-9)
        v = np.cross(z, a)
        c = np.clip(np.dot(z, a), -1.0, 1.0)
        if np.linalg.norm(v) < 1e-9:
            R_align = np.eye(3) if c > 0 else np.diag([1, -1, -1])
        else:
            s = np.linalg.norm(v)
            vx = np.array([[0, -v[2], v[1]],
                           [v[2], 0, -v[0]],
                           [-v[1], v[0], 0]])
            R_align = np.eye(3) + vx + vx @ vx * ((1 - c) / (s**2))
        mesh.rotate(R_align, center=(0,0,0))
        mesh.translate(origin)
        mesh.paint_uniform_color(color_rgb)
        return mesh

    x = R[:,0]; y = R[:,1]; z = R[:,2]
    return [
        one_arrow([1,0,0], x),
        one_arrow([0,1,0], y),
        one_arrow([0,0,1], z),
    ]

def make_end_points_from_o3d_cloud(cloud_o3d: o3d.geometry.PointCloud, num_point: int = 3000):
    """Build GraspNet inputs from an Open3D cloud (in *camera frame*)."""
    pts = np.asarray(cloud_o3d.points, dtype=np.float32)
    cols = np.asarray(cloud_o3d.colors, dtype=np.float32)
    if pts.shape[0] == 0:
        raise RuntimeError("[make_end_points] empty cloud.")

    # Cap for collision check speed (still keep geometry by random downsample)
    vis_cap = 20000
    if pts.shape[0] > vis_cap:
        idx = np.random.choice(pts.shape[0], vis_cap, replace=False)
        cloud_vis = o3d.geometry.PointCloud()
        cloud_vis.points = o3d.utility.Vector3dVector(pts[idx])
        cloud_vis.colors = o3d.utility.Vector3dVector(cols[idx])
    else:
        cloud_vis = cloud_o3d

    # Sample fixed num_point for network
    if pts.shape[0] >= num_point:
        idxs = np.random.choice(pts.shape[0], num_point, replace=False)
    else:
        idxs1 = np.arange(pts.shape[0])
        idxs2 = np.random.choice(pts.shape[0], num_point - pts.shape[0], replace=True)
        idxs = np.concatenate([idxs1, idxs2], axis=0)

    pts_s = pts[idxs]
    cols_s = cols[idxs]

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA不可用：GraspNet(pointnet2)算子不支持CPU。请确认 torch.cuda.is_available()==True。")
    device = torch.device("cuda:0")

    end_points = {
        "point_clouds": torch.from_numpy(pts_s[np.newaxis].astype(np.float32)).contiguous().to(device, non_blocking=True),
        "cloud_colors": torch.from_numpy(cols_s[np.newaxis].astype(np.float32)).contiguous().to(device, non_blocking=True),
    }
    return end_points, cloud_vis

def generate_grasps(end_points,
                    cloud,
                    env=None,
                    T_wc_cap=None,
                    collision_cloud=None,
                    scene_context=None,
                    target_center_w=None,
                    display_bgr=None,
                    display_camera=None,
                    display_cloud=None,
                    visual=False,
                    vertical_cam=None,
                    angle_deg=65.0,
                    approach_col=2,
                    approach_sign=1,
                    width_col=1,
                    auto_pick_col=False,
                    debug_axes=False,
                    ik_precheck_topk: int = 40,
                    manual_select: bool = True,
                    manual_topk: int = 8,
                    allow_collision_fallback: bool = True,
                    allow_angle_fallback: bool = True,
                    collision_approach_dist: float = 0.05,
                    collision_isolated_filter: bool = True,
                    collision_isolated_radius: float = 0.020,
                    collision_isolated_min_support: int = 8,
                    target_empty_thresh: float = 0.01,
                    target_empty_voxel_size: float = 0.005,
                    target_support_cloud=None):
    """
    主推理流程：
    0. 数据处理并生成输入
    1. 加载网络
    2. 前向推理（进行抓取预测解码）
    3. 碰撞检测
    4. NMS 去重 + 按置信度/得分排序（降序）
    5. 对抓取预测进行垂直角度筛选
    """

    # 1. 加载网络
    net = _prepare_graspnet_for_inference()

    # 2. 前向推理
    _clear_cuda_cache()
    try:
        with torch.no_grad():
            end_points = net(end_points)
            grasp_preds = pred_decode(end_points)
    finally:
        _offload_graspnet()

    gg_array = grasp_preds[0].detach().cpu().numpy()
    del grasp_preds
    gg = GraspGroup(gg_array)
    gg_before_collision = GraspGroup(gg_array.copy())
    del end_points
    _clear_cuda_cache()

    # 3. 碰撞检测
    if len(gg) > 0:
        cloud_collision = collision_cloud if collision_cloud is not None else cloud
        cloud_np = np.asarray(cloud_collision.points, dtype=np.float32)
        if bool(collision_isolated_filter):
            isolated_radius = max(float(collision_isolated_radius), 0.0)
            isolated_min_support = max(int(collision_isolated_min_support), 1)
            if (
                cloud_np.shape[0] >= isolated_min_support
                and isolated_radius > 0.0
                and isolated_min_support > 1
            ):
                collision_tree = cKDTree(cloud_np)
                try:
                    local_support = np.asarray(
                        collision_tree.query_ball_point(
                            cloud_np,
                            isolated_radius,
                            workers=1,
                            return_length=True,
                        ),
                        dtype=np.int32,
                    )
                except TypeError:
                    local_support = np.asarray(
                        [
                            len(indices)
                            for indices in collision_tree.query_ball_point(
                                cloud_np, isolated_radius
                            )
                        ],
                        dtype=np.int32,
                    )
                supported = local_support >= isolated_min_support
                if np.any(supported):
                    removed = int(np.count_nonzero(~supported))
                    if removed:
                        cloud_np = cloud_np[supported]
                        print(
                            "[grasp][collision][isolated] "
                            f"kept={cloud_np.shape[0]}/{supported.size} "
                            f"removed={removed} "
                            f"radius={isolated_radius * 1000.0:.1f}mm "
                            f"min_support={isolated_min_support}"
                        )
        mfcdetector = ModelFreeCollisionDetector(cloud_np, voxel_size=0.01)
        collision_approach_dist = max(float(collision_approach_dist), 0.01)
        relaxed_collision_approach_dist = max(
            min(collision_approach_dist, 0.03),
            0.01,
        )
        print(
            "[grasp][collision] voxel prefilter corridors: "
            f"strict={collision_approach_dist * 1000.0:.1f}mm "
            f"relaxed={relaxed_collision_approach_dist * 1000.0:.1f}mm"
        )
        collision_mask = mfcdetector.detect(
            gg,
            approach_dist=collision_approach_dist,
            collision_thresh=0.01,
        )
        gg_strict = gg[~collision_mask]
        if len(gg_strict) > 0:
            gg = gg_strict
        else:
            print("[Warning] Strict collision filtering removed all grasps; retry with relaxed collision settings.")
            collision_mask_relaxed = mfcdetector.detect(
                gg_before_collision,
                approach_dist=relaxed_collision_approach_dist,
                collision_thresh=0.02,
            )
            gg_relaxed = gg_before_collision[~collision_mask_relaxed]
            if len(gg_relaxed) > 0:
                gg = gg_relaxed
            else:
                if not bool(allow_collision_fallback):
                    gg_diagnostic = GraspGroup(gg_array.copy())
                    gg_diagnostic = gg_diagnostic.nms()
                    gg_diagnostic.sort_by_score()
                    diagnostic_rows = []
                    for diagnostic_idx, diagnostic_grasp in enumerate(
                        list(gg_diagnostic)[:max(int(manual_topk), 1)]
                    ):
                        diagnostic_rows.append(
                            {
                                "idx": int(diagnostic_idx),
                                "grasp": diagnostic_grasp,
                                "translation": np.asarray(
                                    diagnostic_grasp.translation,
                                    dtype=np.float32,
                                ).reshape(3,),
                                "collision_filter_ok": False,
                                "collision_filter_stage": "strict_and_relaxed",
                                "strict_approach_dist": float(
                                    collision_approach_dist
                                ),
                                "relaxed_approach_dist": float(
                                    relaxed_collision_approach_dist
                                ),
                            }
                        )
                    error = RuntimeError(
                        "[generate_grasps] strict and relaxed collision filtering removed all grasps; "
                        "refuse to restore colliding candidates"
                    )
                    error.diagnostic_candidate_rows = diagnostic_rows
                    error.diagnostic_filter_stage = "collision"
                    raise error
                print("[Warning] Relaxed collision filtering still removed all grasps; fallback to pre-collision grasps.")
                gg = gg_before_collision

    # Scene collision alone can accept a grasp whose closing volume contains a
    # neighbouring object or only empty space. Verify occupancy against the
    # segmented target cloud itself before ranking candidates.
    if len(gg) > 0 and float(target_empty_thresh) > 0.0:
        support_cloud = cloud if target_support_cloud is None else target_support_cloud
        target_np = np.asarray(support_cloud.points, dtype=np.float32)
        if target_np.shape[0] > 0:
            target_detector = ModelFreeCollisionDetector(
                target_np,
                voxel_size=max(float(target_empty_voxel_size), 0.001),
            )
            _, target_empty_mask = target_detector.detect(
                gg,
                approach_dist=0.01,
                collision_thresh=float("inf"),
                return_empty_grasp=True,
                empty_thresh=float(target_empty_thresh),
            )
            gg_target_supported = gg[~target_empty_mask]
            if len(gg_target_supported) == 0:
                raise RuntimeError(
                    "[generate_grasps] every collision-free candidate has an empty target closing volume"
                )
            if len(gg_target_supported) < len(gg):
                print(
                    "[generate_grasps] target occupancy filter: "
                    f"kept {len(gg_target_supported)}/{len(gg)} candidates"
                )
            gg = gg_target_supported

    # 4. NMS + 排序
    if len(gg) == 0:
        raise RuntimeError("[generate_grasps] no grasp candidates available even before collision fallback.")
    gg = gg.nms()
    gg.sort_by_score()

    # 5. 抓取角度筛选，并在保留候选中优先中心位置 + 更接近世界 +Z 的姿态
    if vertical_cam is None:
        vertical = np.array([0.0, 0.0, 1.0], dtype=float)
    else:
        vertical = np.asarray(vertical_cam, dtype=float).reshape(3,)
        vertical = vertical / (np.linalg.norm(vertical) + 1e-9)

    cos_th = np.cos(np.deg2rad(float(angle_deg)))
    filtered = []

    if auto_pick_col:
        # 自动选择与世界 +Z 正方向最接近的一列作为“approach axis”
        col_scores = []
        for col in (0, 1, 2):
            vals = []
            for g in gg:
                Rg = np.asarray(g.rotation_matrix, dtype=float).reshape(3, 3)
                a = Rg[:, col] * float(approach_sign)
                a = a / (np.linalg.norm(a) + 1e-9)
                vals.append(float(np.clip(np.dot(a, vertical), 0.0, 1.0)))
            col_scores.append(np.mean(vals) if len(vals) > 0 else -1)
        approach_col = int(np.argmax(col_scores))
        print(f"[generate_grasps] auto picked approach_col={approach_col}, scores={np.round(col_scores, 4)}")

    obstacle_dirs_w = _collect_obstacle_dirs_for_target(scene_context, target_center_w)
    ik_info_map = {}

    # For elongated objects (banana, cucumber, radish), a useful parallel-jaw
    # grasp normally closes across the long axis, not along it.  Preserve this
    # target-shape diagnostic with every candidate so the final executable
    # recommendation can distinguish path feasibility from grasp quality.
    target_main_axis_camera = None
    target_shape_elongation = 1.0
    target_direction_applicable = False
    target_shape_points = np.asarray(cloud.points, dtype=np.float64).reshape(-1, 3)
    if target_shape_points.shape[0] >= 16:
        centered_shape = target_shape_points - np.median(
            target_shape_points, axis=0, keepdims=True
        )
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(
                np.cov(centered_shape, rowvar=False)
            )
            order = np.argsort(eigenvalues)[::-1]
            eigenvalues = np.maximum(eigenvalues[order], 0.0)
            target_main_axis_camera = np.asarray(
                eigenvectors[:, order[0]], dtype=float
            ).reshape(3)
            target_main_axis_camera /= (
                np.linalg.norm(target_main_axis_camera) + 1e-12
            )
            target_shape_elongation = float(
                np.sqrt(eigenvalues[0] / max(eigenvalues[1], 1e-12))
            )
            target_direction_applicable = bool(target_shape_elongation >= 1.5)
            print(
                "[GraspRank][shape] target PCA "
                f"elongation={target_shape_elongation:.3f} "
                f"direction_gate={target_direction_applicable} "
                f"main_axis_camera={np.round(target_main_axis_camera, 4).tolist()}"
            )
        except (ValueError, np.linalg.LinAlgError):
            target_main_axis_camera = None

    def _vertical_align_of_grasp(g):
        Rg = np.asarray(g.rotation_matrix, dtype=float).reshape(3, 3)
        approach = Rg[:, int(approach_col)] * float(approach_sign)
        approach = approach / (np.linalg.norm(approach) + 1e-9)
        vertical_align = float(np.clip(np.dot(approach, vertical), -1.0, 1.0))
        return vertical_align
    # 抓取候选姿态打分函数
    def _build_grasp_meta(g, center_xyz, center_gate_dist):
        vertical_align = _vertical_align_of_grasp(g)
        vertical_score = float(np.clip(vertical_align, 0.0, 1.0)) # **竖直方向匹配度**（夹爪朝向是否接近垂直向下抓物体）
        center_dist = float(np.linalg.norm(np.asarray(g.translation, dtype=np.float32) - center_xyz.astype(np.float32)))
        center_score = float(np.clip(1.0 - center_dist / max(center_gate_dist, 1e-6), 0.0, 1.0)) # **中心位置匹配度**（抓取点是否接近物体中心）
        raw_score = float(g.score)
        obstacle_align = 0.0
        if len(obstacle_dirs_w) > 0 and T_wc_cap is not None:
            front_dir_w = _front_dir_world_from_single_grasp(g, T_wc_cap)
            front_xy = np.asarray(front_dir_w, dtype=np.float32).reshape(3,)
            front_xy[2] = 0.0
            n_front = float(np.linalg.norm(front_xy))
            if n_front > 1e-6:
                front_xy = front_xy / n_front
                obstacle_align = max(
                    float(np.clip(np.dot(front_xy[:2], np.asarray(obs_dir, dtype=np.float32).reshape(3,)[:2]), 0.0, 1.0))
                    for obs_dir in obstacle_dirs_w
                )
        obstacle_score = float(np.clip(1.0 - obstacle_align, 0.0, 1.0))
        ik_info = ik_info_map.get(id(g), None)
        ik_jump = float(ik_info.get("max_joint_jump", np.inf)) if ik_info is not None else np.inf
        ik_score = float(np.clip(1.0 - ik_jump / 1.8, 0.0, 1.0)) if np.isfinite(ik_jump) else 0.0
        width_long_axis_alignment = None
        target_cross_axis_score = None
        if target_main_axis_camera is not None:
            grasp_rotation = np.asarray(
                g.rotation_matrix, dtype=float
            ).reshape(3, 3)
            width_axis_camera = grasp_rotation[:, int(width_col)]
            width_axis_camera /= np.linalg.norm(width_axis_camera) + 1e-12
            width_long_axis_alignment = float(
                abs(np.dot(width_axis_camera, target_main_axis_camera))
            )
            target_cross_axis_score = float(
                np.clip(1.0 - width_long_axis_alignment, 0.0, 1.0)
            )
        priority = (
            0.28 * vertical_score +  # **竖直方向匹配度**
            0.30 * center_score +   # **中心位置匹配度** 
            0.27 * obstacle_score +   # **障碍物规避度**
            0.10 * ik_score +         # **关节运动代价**
            0.05 * raw_score          # **原始得分**
        )
        return {
            "vertical_align": float(vertical_align),
            "vertical_score": float(vertical_score),
            "center_dist": center_dist,
            "center_score": center_score,
            "obstacle_align": float(obstacle_align),
            "obstacle_score": float(obstacle_score),
            "ik_jump": float(ik_jump),
            "ik_score": float(ik_score),
            "raw_score": raw_score,
            "priority": float(priority),
            "target_shape_elongation": float(target_shape_elongation),
            "target_direction_applicable": bool(target_direction_applicable),
            "target_main_axis_camera": (
                None
                if target_main_axis_camera is None
                else target_main_axis_camera.astype(float).tolist()
            ),
            "width_long_axis_alignment": width_long_axis_alignment,
            "target_cross_axis_score": target_cross_axis_score,
        }

    def _filter_by_vertical(threshold_cos):
        keep = []
        for g in gg:
            if _vertical_align_of_grasp(g) >= threshold_cos:
                keep.append(g)
        return keep

    vertical_pass = _filter_by_vertical(cos_th)
    filtered = list(vertical_pass)

    if len(filtered) == 0:
        if not bool(allow_angle_fallback):
            pts_cloud = np.asarray(cloud.points, dtype=np.float32)
            diagnostic_center = (
                np.median(pts_cloud, axis=0)
                if pts_cloud.shape[0] > 0
                else np.zeros((3,), dtype=np.float32)
            )
            diagnostic_center_gate = 0.12
            diagnostic_rows = []
            for diagnostic_idx, diagnostic_grasp in enumerate(
                list(gg)[:max(int(manual_topk), 1)]
            ):
                diagnostic_meta = _build_grasp_meta(
                    diagnostic_grasp,
                    diagnostic_center,
                    diagnostic_center_gate,
                )
                vertical_align = float(
                    diagnostic_meta.get("vertical_align", -1.0)
                )
                diagnostic_rows.append(
                    {
                        "idx": int(diagnostic_idx),
                        "grasp": diagnostic_grasp,
                        "translation": np.asarray(
                            diagnostic_grasp.translation,
                            dtype=np.float32,
                        ).reshape(3,),
                        **diagnostic_meta,
                        "angle_constraint_ok": False,
                        "angle_to_world_down_deg": float(
                            np.rad2deg(
                                np.arccos(np.clip(vertical_align, -1.0, 1.0))
                            )
                        ),
                        "angle_constraint_deg": float(angle_deg),
                    }
                )
            error = RuntimeError(
                "[generate_grasps] no candidate satisfies the world-down approach-angle constraint "
                f"({float(angle_deg):.1f}deg)"
            )
            error.diagnostic_candidate_rows = diagnostic_rows
            error.diagnostic_filter_stage = "world_down_angle"
            raise error
        relaxed_angle_deg = min(float(angle_deg) + 15.0, 85.0)
        relaxed_cos_th = np.cos(np.deg2rad(relaxed_angle_deg))
        filtered = _filter_by_vertical(relaxed_cos_th)
        if len(filtered) > 0:
            print(
                f"[Warning] No grasp passes strict vertical-angle filter ({float(angle_deg):.1f}deg); "
                f"use relaxed filter ({relaxed_angle_deg:.1f}deg)."
            )
        else:
            print("[Warning] No grasp passes strict/relaxed vertical-angle filter; fallback to vertically ranked grasps.")
            filtered = list(gg[:min(len(gg), 20)])

    # 可选：再按“离中心不要太远”筛一下
    pts_cloud = np.asarray(cloud.points, dtype=np.float32)
    if pts_cloud.shape[0] > 0:
        center = np.median(pts_cloud, axis=0)
        max_center_dist = 0.12
        filtered2 = [g for g in filtered if np.linalg.norm(np.asarray(g.translation) - center) < max_center_dist]
        if len(filtered2) > 0:
            filtered = filtered2
        else:
            print("[Warning] All candidate grasps are far from cloud center; keep original candidates.")
    else:
        center = np.zeros((3,), dtype=np.float32)
        max_center_dist = 0.12

    candidate_meta = {id(g): _build_grasp_meta(g, center, max_center_dist) for g in filtered}

    # 对过滤后的抓取优先按：更接近世界 +Z、更靠近点云中心、网络分数更高
    filtered.sort(
        key=lambda g: (
            -candidate_meta[id(g)]["priority"],
            -candidate_meta[id(g)]["vertical_align"],
            candidate_meta[id(g)]["center_dist"],
            -candidate_meta[id(g)]["raw_score"],
        )
    )

    if env is not None and T_wc_cap is not None and len(filtered) > 0:
        reachable = []
        if getattr(env.robot, "dof", 0) == 7:
            ik_precheck_topk = int(os.environ.get("NERO_IK_PRECHECK_TOPK", "24"))
        checked = min(len(filtered), max(int(ik_precheck_topk), 1))
        last_reasons = []
        for g in filtered[:checked]:
            ok, info = _grasp_is_ik_reachable(env, g, T_wc_cap=T_wc_cap, pregrasp_offset=0.08)
            if ok:
                reachable.append(g)
                ik_info_map[id(g)] = info
            else:
                last_reasons.append(str(info.get("reason", "ik_failed")))
        retry_all_after_topk = os.environ.get("NERO_IK_PRECHECK_SCAN_ALL", "1").strip().lower() in {"1", "true", "yes", "on"}
        if len(reachable) == 0 and checked < len(filtered) and (getattr(env.robot, "dof", 0) != 7 or retry_all_after_topk):
            for g in filtered[checked:]:
                ok, info = _grasp_is_ik_reachable(env, g, T_wc_cap=T_wc_cap, pregrasp_offset=0.08)
                if ok:
                    reachable.append(g)
                    ik_info_map[id(g)] = info
                else:
                    last_reasons.append(str(info.get("reason", "ik_failed")))
        if len(reachable) > 0:
            print(f"[GraspIK] kept {len(reachable)}/{len(filtered)} IK-reachable grasps (checked={min(len(filtered), max(int(ik_precheck_topk), 1))} first).")
            filtered = reachable
            candidate_meta = {id(g): _build_grasp_meta(g, center, max_center_dist) for g in filtered}
        else:
            preview = ",".join(last_reasons[:3]) if len(last_reasons) > 0 else "unknown"
            if os.environ.get("GRASP_SKIP_UNREACHABLE_CLOUD", "1").strip().lower() not in {"0", "false", "no", "off"}:
                raise RuntimeError(f"IK precheck rejected all grasp candidates; skip this cloud. sample_reasons={preview}")
            print(f"[Warning] IK precheck rejected all grasp candidates; keep original ranking. sample_reasons={preview}")

    # 取前若干抓取供查看/手选
    top_grasps = filtered[:max(int(manual_topk), 1)]

    # 可视化过滤后的抓取，手动转换为 Open3D 物体
    grippers = [g.to_open3d_geometry() for g in top_grasps]
    # print(f"\nVisualizing top {len(top_grasps)} grasps after vertical filtering...")
    # o3d.visualization.draw_geometries([cloud, *grippers])
    # for gripper in grippers:
    #     o3d.visualization.draw_geometries([cloud, gripper])

    candidate_rows = []
    for idx, g in enumerate(top_grasps):
        meta = candidate_meta.get(id(g), _build_grasp_meta(g, center, max_center_dist))
        candidate_rows.append({
            "idx": int(idx),
            "grasp": g,
            "translation": np.asarray(g.translation, dtype=np.float32).reshape(3,),
            **meta,
        })

    chosen_idx = 0
    if manual_select and len(candidate_rows) > 1:
        chosen_idx = _prompt_user_to_select_grasp(
            candidate_rows,
            "GraspSelect",
            display_bgr=display_bgr,
            display_camera=display_camera,
            T_wc_cap=T_wc_cap,
            display_cloud=display_cloud if display_cloud is not None else cloud,
            show_pointcloud=bool(visual),
        )
    elif visual and _env_flag("GRASP_ALWAYS_SHOW_CANDIDATE_POINTCLOUD", "1"):
        _show_grasp_candidate_pointcloud(
            display_cloud if display_cloud is not None else cloud,
            candidate_rows,
            window_name="GraspSelect Fused PointCloud",
        )

    best_grasp = candidate_rows[int(chosen_idx)]["grasp"]
    best_meta = candidate_meta.get(id(best_grasp), None)
    if best_meta is not None:
        print(
            "[GraspRank] selected best grasp "
            f"priority={best_meta['priority']:.4f} "
            f"vertical_up={best_meta['vertical_align']:.4f} "
            f"center_dist={best_meta['center_dist']:.4f} "
            f"obs_avoid={best_meta['obstacle_score']:.4f} "
            f"ik_jump={best_meta['ik_jump']:.4f} "
            f"score={best_meta['raw_score']:.4f}"
        )
    if debug_axes:
        try:
            g = best_grasp
            Rg = np.asarray(g.rotation_matrix, dtype=float).reshape(3,3)
            tg = np.asarray(g.translation, dtype=float).reshape(3)
            # print angles of each axis to vertical
            for col in (0,1,2):
                a = Rg[:, col]
                a = a / (np.linalg.norm(a) + 1e-9)
                ang = np.degrees(np.arccos(np.clip(np.dot(a, vertical), -1.0, 1.0)))
                print(f"[dbg_axes] col={col} angle_to_vertical={ang:.1f} deg vec={np.round(a,3)}")
                arrows = _axis_arrows_at(tg, Rg, scale=0.06)
            o3d.visualization.draw_geometries([cloud, g.to_open3d_geometry(), *arrows])
        except Exception as e:
            print(f"[dbg_axes] failed: {type(e).__name__}: {e}")

    # 创建一个新的 GraspGroup 并添加最佳抓取
    new_gg = GraspGroup()            # 初始化空的 GraspGroup
    new_gg.add(best_grasp)           # 添加最佳抓取
    if visual and not manual_select:
        # This is a network-candidate preview, before real TCP/workflow
        # filtering. Both the cloud and Grasp geometry are in camera0.
        # The real robot's final execution preview lives in runtime.py and
        # maps the validated TCP back into this same camera frame.
        print("[GraspSelect] Raw GraspNet candidate in camera0; final TCP review follows workflow filtering.")
        grippers = new_gg.to_open3d_geometry_list()
        o3d.visualization.draw_geometries([cloud, *grippers])
    return new_gg, candidate_rows
# ================= 仿真执行抓取动作 ====================
def execute_grasp(env, gg, T_wc_cap=None):
    """
    执行抓取动作：将 GraspNet 输出（相机系）转换到世界系，并驱动机械臂执行。

    参数:
      env: 机器人环境对象
      gg : GraspGroup（GraspNet 输出）
      T_wc_cap: 可选，拍照时刻的相机位姿（camera->world），用于末端相机避免“拍照后相机动了”的误差
    """
    robot = env.robot
    cartesian_is_world, _, _ = _build_frame_converters(env)
    print(f"[FrameDetect] robot.get_cartesian seems in: {'world' if cartesian_is_world else 'base'}")

    PREGRASP_OFFSET = 0.08

    # ---- 0) 计算世界系抓取位姿 T_wo ----
    if gg is None or (hasattr(gg, "__len__") and len(gg) == 0):
        raise RuntimeError("[execute_grasp] gg 为空：没有可执行的抓取候选。")

    # 末端相机：必须用真实相机外参（不要手写/写死）
    if T_wc_cap is None:
        T_wc = get_T_wc_from_camera(env, EE_CAM_NAME)
    else:
        T_wc = T_wc_cap

    print("gg.translations[0] =", gg.translations[0], "norm=", np.linalg.norm(gg.translations[0]))
    action = np.zeros(env.mj_model.nu)

    # 1. 从当前中心观测位直接移动到预抓取，再推进到抓取位
    try:
        pre_offsets = [float(os.environ.get("NERO_PREGRASP_OFFSET", "0.06"))] if getattr(robot, "dof", 0) == 7 else [PREGRASP_OFFSET]
        last_err = None
        for pre_offset in pre_offsets:
            T_wo, T_pre, T_wo_world = _grasp_to_command_poses(env, gg, T_wc_cap=T_wc, pregrasp_offset=float(pre_offset))
            if getattr(robot, "dof", 0) == 7:
                min_grasp_z = float(os.environ.get("NERO_MIN_GRASP_Z", "0.755"))
                if float(T_wo.t[2]) < min_grasp_z:
                    dz = min_grasp_z - float(T_wo.t[2])
                    print(f"[ExecuteGrasp] raise grasp z by {dz:.3f}m to avoid table penetration.")
                    T_wo = sm.SE3.Trans(0.0, 0.0, dz) * T_wo
                    T_pre = sm.SE3.Trans(0.0, 0.0, dz) * T_pre
                    T_wo_world = sm.SE3.Trans(0.0, 0.0, dz) * T_wo_world
            print("T_wo_world.t =", np.array(T_wo_world.t).round(3))
            print("T_wo_cmd.t   =", np.array(T_wo.t).round(3))
            print(f"T_pre_cmd.t  = {np.array(T_pre.t).round(3)} pre_offset={float(pre_offset):.3f}")
            q_saved = np.array(robot.get_joint(), dtype=float)
            try:
                action[:env.robot.dof] = q_saved
                action = _move_gripper_smooth(
                    env,
                    action,
                    target=0.0,
                    step_delta=float(os.environ.get("EXEC_GRIPPER_OPEN_STEP_DELTA", "2.0")),
                    max_steps=int(os.environ.get("EXEC_GRIPPER_OPEN_MAX_STEPS", "180")),
                )
                _move_pose_smooth(env, T_pre, duration=1.0, settle_steps=60)
                _move_pose_smooth(env, T_wo, duration=0.8, settle_steps=40)
                break
            except AssertionError as e:
                last_err = e
                robot.set_joint(q_saved)
                action[:env.robot.dof] = q_saved
                print(f"[ExecuteGrasp] pregrasp offset {float(pre_offset):.3f} failed.")
        else:
            if os.environ.get("EXEC_SKIP_UNREACHABLE_GRASP", "1").strip().lower() not in {"0", "false", "no", "off"}:
                print("[ExecuteGrasp] all pregrasp offsets unreachable; skip this grasp.")
                return False
            raise last_err if last_err is not None else AssertionError()
        action[:env.robot.dof] = robot.get_joint()
        action, contact_ok, contact_rows, close_reason = _close_gripper_until_grasp_contact(
            env,
            action,
            step_delta=float(os.environ.get("EXEC_GRIPPER_CLOSE_STEP_DELTA", "2.0")),
            max_steps=int(os.environ.get("EXEC_GRIPPER_CLOSE_MAX_STEPS", "220")),
        )
        close_contact_ok = bool(contact_ok)
        close_contact_rows = contact_rows
        for _ in range(20):
            env.step(action)
        final_contact_ok, final_contact_rows = _has_gripper_object_contact(env)
        if final_contact_ok:
            contact_ok, contact_rows = True, final_contact_rows
        elif close_contact_ok:
            contact_ok, contact_rows = True, close_contact_rows
        else:
            contact_ok, contact_rows = False, final_contact_rows
        contact_preview = [
            f"{r['g1']}<->{r['g2']}:side={r['side']} obj={r['object']} key={r.get('object_key', r['object'])} ok={int(r['gripper_object'])} dist={r['dist']:.5f}"
            for r in contact_rows[:8]
        ]
        grip_q = _gripper_joint_positions(env)
        print(
            f"[ExecuteGrasp][contact] after_close gripper_object_contact={int(contact_ok)} "
            f"close_reason={close_reason} gripper_q={np.round(np.asarray(grip_q, dtype=float), 5).tolist()} "
            f"ncon={int(env.mj_data.ncon)} contacts={contact_preview}"
        )
        if _env_flag("EXEC_REQUIRE_GRASP_CONTACT", "1") and not contact_ok:
            print(f"[ExecuteGrasp] no gripper-object contact after close ({close_reason}); abort lift to avoid non-contact pickup.")
            action = _move_gripper_smooth(
                env,
                action,
                target=0.0,
                step_delta=float(os.environ.get("EXEC_GRIPPER_OPEN_STEP_DELTA", "2.0")),
                max_steps=int(os.environ.get("EXEC_GRIPPER_OPEN_MAX_STEPS", "180")),
            )
            return False

        # 4.提起物体
        T3 = T_wo
        if getattr(robot, "dof", 0) == 7:
            lift_offsets = [
                float(x)
                for x in os.environ.get("NERO_LIFT_OFFSETS", "0.12 0.16 0.20 0.24").replace(",", " ").split()
                if x.strip()
            ]
        else:
            lift_offsets = [0.3]
        lift_ok = False
        lift_err = None
        T4 = None
        for lift_offset in lift_offsets:
            T4_try = sm.SE3.Trans(0.0, 0.0, float(lift_offset)) * T3
            try:
                _move_pose_smooth(env, T4_try, duration=0.75, settle_steps=35)
                T4 = T4_try
                lift_ok = True
                print(f"[ExecuteGrasp] lift succeeded by Cartesian IK dz={float(lift_offset):.3f}m")
                break
            except AssertionError as e:
                lift_err = e
                print(f"[ExecuteGrasp] lift dz={float(lift_offset):.3f}m Cartesian IK failed.")
        if not lift_ok and getattr(robot, "dof", 0) == 7 and _env_flag("NERO_ENABLE_JOINT_LIFT_FALLBACK", "1"):
            lift_scan = _find_nero_joint_lift_candidate(
                env,
                min_lift=float(os.environ.get("NERO_JOINT_LIFT_MIN", "0.04")),
                desired_lift=float(os.environ.get("NERO_JOINT_LIFT_DESIRED", "0.12")),
                max_xy_drift=float(os.environ.get("NERO_JOINT_LIFT_MAX_XY_DRIFT", "0.16")),
            )
            if lift_scan is not None:
                print(
                    f"[ExecuteGrasp] joint-space lift fallback: "
                    f"lift={lift_scan['lift']:.3f}m xy_drift={lift_scan['xy_drift']:.3f}m "
                    f"checked={lift_scan.get('checked', 0)}"
                )
                move_joint_smooth(env, np.asarray(lift_scan["q"], dtype=float), duration=0.75, settle_steps=35)
                T4 = robot.get_cartesian()
                lift_ok = True
            else:
                print("[ExecuteGrasp] joint-space lift fallback failed: no upward joint candidate.")
        if not lift_ok:
            raise lift_err if lift_err is not None else AssertionError("[ExecuteGrasp] lift IK failed")

        if _env_flag("EXEC_VERIFY_CONTACT_AFTER_LIFT", "1"):
            lift_contact_ok, lift_contact_rows = _has_gripper_object_contact(env)
            lift_contact_preview = [
                f"{r['g1']}<->{r['g2']}:side={r['side']} obj={r['object']} key={r.get('object_key', r['object'])} ok={int(r['gripper_object'])} dist={r['dist']:.5f}"
                for r in lift_contact_rows[:8]
            ]
            print(
                f"[ExecuteGrasp][contact] after_lift gripper_object_contact={int(lift_contact_ok)} "
                f"contacts={lift_contact_preview}"
            )
            if not lift_contact_ok:
                if (
                    _env_flag("EXEC_CLEAR_GRASP_MODE", "0")
                    and close_contact_ok
                    and _env_flag("EXEC_TREAT_CLEAR_CONTACT_LOST_AFTER_LIFT_AS_SUCCESS", "1")
                ):
                    print("[ExecuteGrasp] clear grasp contact was present after close but lost after lift; treat as cleared and release.")
                    action[:env.robot.dof] = robot.get_joint()
                    action = _move_gripper_smooth(
                        env,
                        action,
                        target=0.0,
                        step_delta=float(os.environ.get("EXEC_GRIPPER_OPEN_STEP_DELTA", "2.0")),
                        max_steps=int(os.environ.get("EXEC_GRIPPER_OPEN_MAX_STEPS", "180")),
                    )
                    goto_observation_pose(env, duration=1.0)
                    return True
                print("[ExecuteGrasp] contact lost after lift; treat as failed grasp and release.")
                action[:env.robot.dof] = robot.get_joint()
                action = _move_gripper_smooth(
                    env,
                    action,
                    target=0.0,
                    step_delta=float(os.environ.get("EXEC_GRIPPER_OPEN_STEP_DELTA", "2.0")),
                    max_steps=int(os.environ.get("EXEC_GRIPPER_OPEN_MAX_STEPS", "180")),
                )
                goto_observation_pose(env, duration=1.0)
                return False

        # Move through an elevated, easy-to-reach air waypoint before placing.
        waypoint_text = os.environ.get("NERO_PLACE_AIR_WAYPOINTS_XYZ", "1.25 0.55 1.08; 1.30 0.60 1.10; 1.20 0.50 1.05")
        waypoint_candidates = []
        for chunk in waypoint_text.split(";"):
            vals = np.fromstring(chunk.replace(",", " "), sep=" ", dtype=float)
            if vals.size >= 3 and np.isfinite(vals[:3]).all():
                waypoint_candidates.append((float(vals[0]), float(vals[1]), float(vals[2])))
        if len(waypoint_candidates) > 0 and _env_flag("NERO_ENABLE_PLACE_AIR_WAYPOINT", "1"):
            waypoint_ok = False
            waypoint_err = None
            for wp_x, wp_y, wp_z in waypoint_candidates:
                T_air_try = sm.SE3.Trans(float(wp_x), float(wp_y), max(float(wp_z), float(T4.t[2]))) * sm.SE3(sm.SO3(T4.R))
                try:
                    _move_pose_smooth(env, T_air_try, duration=0.9, settle_steps=45)
                    T4 = T_air_try
                    waypoint_ok = True
                    print(f"[ExecuteGrasp] air waypoint succeeded at xyz=({wp_x:.3f}, {wp_y:.3f}, {max(float(wp_z), float(T4.t[2])):.3f})")
                    break
                except AssertionError as e:
                    waypoint_err = e
                    print(f"[ExecuteGrasp] air waypoint xyz=({wp_x:.3f}, {wp_y:.3f}, {max(float(wp_z), float(T4.t[2])):.3f}) Cartesian IK failed.")
            if not waypoint_ok and not _env_flag("NERO_PLACE_AIR_WAYPOINT_OPTIONAL", "1"):
                raise waypoint_err if waypoint_err is not None else AssertionError("[ExecuteGrasp] air waypoint IK failed")

        # 5.水平移动物体
        place_text = os.environ.get("NERO_PLACE_TARGETS_XY", "1.4 0.3; 1.30 0.45; 1.20 0.55; 1.35 0.60")
        place_xy_candidates = []
        for chunk in place_text.split(";"):
            vals = np.fromstring(chunk.replace(",", " "), sep=" ", dtype=float)
            if vals.size >= 2 and np.isfinite(vals[:2]).all():
                place_xy_candidates.append((float(vals[0]), float(vals[1])))
        if len(place_xy_candidates) == 0:
            place_xy_candidates = [(1.4, 0.3)]

        T5 = None
        place_err = None
        for place_x, place_y in place_xy_candidates:
            T5_try = sm.SE3.Trans(float(place_x), float(place_y), T4.t[2]) * sm.SE3(sm.SO3(T4.R))
            try:
                _move_pose_smooth(env, T5_try, duration=1.0, settle_steps=60)
                T5 = T5_try
                print(f"[ExecuteGrasp] place move succeeded at xy=({place_x:.3f}, {place_y:.3f})")
                break
            except AssertionError as e:
                place_err = e
                print(f"[ExecuteGrasp] place xy=({place_x:.3f}, {place_y:.3f}) Cartesian IK failed.")
        if T5 is None:
            if getattr(robot, "dof", 0) == 7 and _env_flag("NERO_ALLOW_RELEASE_AFTER_LIFT_IF_PLACE_IK_FAILS", "1"):
                print("[ExecuteGrasp] all place targets failed after successful lift; release at lifted pose and finish this grasp.")
                action[:env.robot.dof] = robot.get_joint()
                action = _move_gripper_smooth(env, action, target=0.0, step_delta=0.2, max_steps=1000)
                goto_observation_pose(env, duration=1.0)
                return True
            raise place_err if place_err is not None else AssertionError("[ExecuteGrasp] place IK failed")

        # 6.放置物体
        T6 = sm.SE3.Trans(0.0, 0.0, -0.1) * T5
        try:
            _move_pose_smooth(env, T6, duration=0.8, settle_steps=40)
        except AssertionError:
            if getattr(robot, "dof", 0) == 7 and _env_flag("NERO_ALLOW_RELEASE_AFTER_LIFT_IF_PLACE_IK_FAILS", "1"):
                print("[ExecuteGrasp] lower/place Cartesian IK failed after successful lift; release at current lifted pose and finish this grasp.")
                action[:env.robot.dof] = robot.get_joint()
                action = _move_gripper_smooth(env, action, target=0.0, step_delta=0.2, max_steps=1000)
                goto_observation_pose(env, duration=1.0)
                return True
            raise
        action[:env.robot.dof] = robot.get_joint()
        action = _move_gripper_smooth(env, action, target=0.0, step_delta=0.2, max_steps=1000)

        # 7.抬起夹爪
        T7 = sm.SE3.Trans(0.0, 0.0, 0.1) * T6
        _move_pose_smooth(env, T7, duration=0.7, settle_steps=40)
    except AssertionError:
        print("[ExecuteGrasp] Cartesian IK failed during grasp execution. Abort this grasp and return to observation pose.")
        try:
            action[:env.robot.dof] = robot.get_joint()
            action = _move_gripper_smooth(env, action, target=0.0, step_delta=0.2, max_steps=400)
        except Exception:
            env._last_gripper_cmd = 0.0
        goto_observation_pose(env, duration=1.0)
        return False

    # 8.回到全局观察位，而不是回到本轮目标的局部观察位
    goto_observation_pose(env, duration=1.0)
    return True
if __name__ == '__main__':

    env = UR5GraspEnv()
    env.reset()
    atexit.register(env.close)
    if os.environ.get("EE_SIDE_VIEW_WINDOW", "0").strip().lower() not in {"0", "false", "no", "off"}:
        env.enable_ee_side_camera_window(
            enabled=True,
            cam_name=os.environ.get("EE_SIDE_VIEW_CAM", EE_CAM_NAME),
            window_name=os.environ.get("EE_SIDE_VIEW_WINDOW_NAME", "EE Side Camera"),
            interval=int(os.environ.get("EE_SIDE_VIEW_INTERVAL", "10")),
        )
        if hasattr(env, "_update_ee_side_camera_window"):
            env._update_ee_side_camera_window(force=True)
    env._last_gripper_cmd = 0.0
    robot = env.robot
    model, data = _get_mj_model_data(env)

    # ===== object-centered multiview config =====
    N_GRASPS = 4
    OBJ_CENTER_HOVER_HEIGHT = float(os.environ.get("OBJ_CENTER_HOVER_HEIGHT", "0.25"))   # 相机“正上方”观察高度（MuJoCo world 是 Z-up）
    ORBIT_ANGLE_DEG = float(os.environ.get("MV_ORBIT_ANGLE_DEG", "30.0"))  # 默认保守一点，优先保证侧视与中心视角的重叠
    ENABLE_CLEAR_FIRST = os.environ.get("ENABLE_CLEAR_FIRST", "1").strip().lower() not in {"0", "false", "no", "off"}
    GRASP_VISUAL = os.environ.get("GRASP_VISUAL", "1").strip().lower() in {"1", "true", "yes", "on"}
    GRASP_MANUAL_SELECT = os.environ.get("GRASP_MANUAL_SELECT", "1").strip().lower() in {"1", "true", "yes", "on"}
    MAX_OCCLUSION_NBV_ROUNDS = 1
    target_class = None
    grasp_count = 0
    debug_overlay_dir = os.path.join(ROOT_DIR, "dbg_instance_vis")
    multiview_debug_dir = os.path.join(ROOT_DIR, "dbg_multiview")

    while grasp_count < N_GRASPS:
        iter_tag = f"g{grasp_count:02d}"

        for i in range(int(os.environ.get("INITIAL_SETTLE_STEPS", "80"))):
            env.step()

        # ------------------------------------------------------------------
        # Stage A: 在全局观察位做 YOLO26-seg，先记下目标中心坐标
        # ------------------------------------------------------------------
        goto_observation_pose(env)
        target_class = _prompt_target_class(target_class)
        force_target_click = (str(target_class).strip().lower() == "__manual_click__") if target_class is not None else False
        target_class_for_seg = None if force_target_click else target_class

        model, _ = _get_mj_model_data(env)
        fovy_cam = get_cam_fovy_rad(model, EE_CAM_NAME)

        imgs_det = render_from_camera(env, EE_CAM_NAME, depth_mode='auto', debug=VERBOSE_LOG)
        rgb_det = imgs_det['img']
        depth_det = imgs_det['depth']
        T_wc_det = get_T_wc_from_camera(env, EE_CAM_NAME)
        bgr_det = cv2.cvtColor(rgb_det, cv2.COLOR_RGB2BGR)
        mask_det = segment_image(
            bgr_det,
            output_mask='mask_det.png',
            target_class=target_class_for_seg,
            interactive=False,
            allow_manual_click=True,
            force_manual_click=force_target_click,
        )
        selected_target = get_last_selected_instance()
        if target_class_for_seg is None and selected_target is not None:
            target_class = str(selected_target.get("label", "")).strip() or None
            if target_class is not None:
                print(f"[TargetSelect] selected target label='{target_class}' score={float(selected_target.get('score', 0.0)):.3f}")
        else:
            target_class = target_class_for_seg

        obj_info = estimate_object_center_from_mask(
            bgr=bgr_det,
            depth_m=depth_det,
            mask=mask_det,
            T_wc=T_wc_det,
            fovy_rad=fovy_cam,
        )
        object_center_w = obj_info['center_w']
        print(f"[ObjCenter] uv={np.round(obj_info['center_uv'], 2)} center_w={np.round(object_center_w, 4)}")
        center_ok, center_reason = _object_center_sanity_check(obj_info, bgr_det.shape[:2])
        if not center_ok:
            print(f"[ObjCenter] unreliable center estimate ({center_reason}); retry from global observation.")
            for _ in range(int(os.environ.get("OBJ_CENTER_RETRY_SETTLE_STEPS", "40"))):
                env.step()
            continue

        # ------------------------------------------------------------------
        # Stage B: 先移动到“目标正上方”的中心观察位，再执行主动视角选择 + 圆轨迹采样
        # ------------------------------------------------------------------
        try:
            goto_object_centered_observation_pose(
                env,
                object_center_w=object_center_w,
                cam_name=EE_CAM_NAME,
                hover_height=OBJ_CENTER_HOVER_HEIGHT,
                duration=1.2,
                settle_steps=80,
                debug=VERBOSE_LOG,
            )
        except Exception as exc:
            print(
                f"[ObjCenter] observation pose failed for center_w={np.round(object_center_w, 4)} "
                f"({type(exc).__name__}: {exc}); retry from global observation."
            )
            goto_observation_pose(env, duration=1.0)
            continue

        # Closed-loop center refinement
        refine_info = refine_object_centered_observation_pose(
            env,
            cam_name=EE_CAM_NAME,
            hover_height=OBJ_CENTER_HOVER_HEIGHT,
            duration=0.8,
            settle_steps=60,
            xy_tol=0.008,
            z_tol=0.008,
            debug=VERBOSE_LOG,
            target_class=target_class,
        )
        if refine_info.get('moved', False):
            object_center_w = np.asarray(refine_info.get('object_center_w', object_center_w), dtype=np.float32).reshape(3,)
            print(f"[ObjCenter][refine] applied correction using center_w={np.round(refine_info['object_center_w'], 4)}")

        reset_scene_instances()
        scene_state = _build_scene_state_from_center_view(
            env,
            target_class=target_class,
            fovy_cam=fovy_cam,
            iter_tag=iter_tag,
            debug_dir=debug_overlay_dir,
            allow_click_init=True,
            target_center_hint_w=object_center_w,
        )
        scene_state = _run_target_multiview_cycle(
            env,
            scene_state,
            target_class=target_class,
            fovy_cam=fovy_cam,
            orbit_angle_deg=ORBIT_ANGLE_DEG,
            iter_tag=iter_tag,
            multiview_debug_dir=multiview_debug_dir,
            max_rounds=MAX_OCCLUSION_NBV_ROUNDS,
        )
        active_view_info = scene_state["active_view_info"]
        print(f"[ActiveNBV] selected={active_view_info.get('best_pair_name', 'unknown')}")
        if VERBOSE_LOG and "candidate_records" in active_view_info:
            topk = sorted(active_view_info["candidate_records"], key=lambda x: x["score"], reverse=True)[:4]
            for rec in topk:
                print(f"  {rec['name']} score={rec['score']:.4f} detail={rec['detail']}")

        T_wc0 = scene_state["T_wc"]
        vertical_c0 = np.asarray(T_wc0.R, dtype=float).T @ np.array([0.0, 0.0, 1.0], dtype=float)
        preclear_dir_w = _estimate_preclear_direction_from_scene(scene_state["scene_context"])
        front_occ = {"has_front_occluder": False, "front_point_count": 0, "occlusion_ratio": 0.0, "nearest_dist": np.inf, "front_instances": []}
        clear_decision = {"should_clear_first": False, "reason": "no_preclear_direction", "score": 0.0}
        preview_clear_target = None
        if preclear_dir_w is not None:
            front_occ = detect_front_occluders(
                scene_context=scene_state["scene_context"],
                target_center_w=scene_state["scene_context"]["target_center_w"],
                front_dir_w=preclear_dir_w,
                corridor_len=0.14,
                corridor_radius=0.05,
                min_points=200,
                debug=VERBOSE_LOG,
            )
            front_occ = augment_front_occluders_with_amodal(
                scene_context=scene_state["scene_context"],
                front_occ_info=front_occ,
                bgr=scene_state["bgr"],
                depth=scene_state["depth"],
                fovy_rad=fovy_cam,
                target_class=target_class,
                debug_dir=debug_overlay_dir,
                iter_tag=f"{iter_tag}_clear_decision",
                debug=True,
            )
            clear_decision = should_clear_occluder_first(front_occ, debug=True)
        if clear_decision["should_clear_first"] and not ENABLE_CLEAR_FIRST:
            print("[FrontOcc] CLEAR-FIRST disabled by ENABLE_CLEAR_FIRST=0; continue with target grasp.")
            clear_decision["should_clear_first"] = False

        if clear_decision["should_clear_first"]:
            print("[FrontOcc] CLEAR-FIRST recommended: front occluder is likely blocking the target approach.")
            if _env_flag("FRONT_OCC_PREVIEW_TARGET_SHIFT_CLEAR", "1"):
                preview_clear_target = preview_front_occluder_observation_for_clear_decision(
                    env,
                    front_occ_info=front_occ,
                    fovy_rad=fovy_cam,
                    debug=VERBOSE_LOG,
                    cam_name=EE_CAM_NAME,
                    debug_dir=debug_overlay_dir,
                    iter_tag=f"{iter_tag}_clear_decision",
                    orbit_angle_deg=ORBIT_ANGLE_DEG,
                    target_class=target_class,
                )
                goto_object_centered_observation_pose(
                    env,
                    object_center_w=np.asarray(scene_state["scene_context"]["target_center_w"], dtype=np.float32),
                    cam_name=EE_CAM_NAME,
                    hover_height=OBJ_CENTER_HOVER_HEIGHT,
                    duration=0.7,
                    settle_steps=30,
                    debug=VERBOSE_LOG,
                )
                T_wc0 = get_T_wc_from_camera(env, EE_CAM_NAME)
                vertical_c0 = np.asarray(T_wc0.R, dtype=float).T @ np.array([0.0, 0.0, 1.0], dtype=float)
                scene_state["T_wc"] = T_wc0
            if preview_clear_target is not None and _env_flag("FRONT_OCC_PREVIEW_TARGET_SHIFT_CLEAR", "1"):
                clear_result = execute_preview_front_occluder_grasp(
                    env,
                    preview_clear_target,
                    debug=VERBOSE_LOG,
                    visual=GRASP_VISUAL,
                    manual_select=GRASP_MANUAL_SELECT,
                )
            else:
                clear_result = attempt_clear_front_occluder(
                    env,
                    front_occ_info=front_occ,
                    center_bgr=scene_state["bgr"],
                    center_depth=scene_state["depth"],
                    T_wc_center=T_wc0,
                    fovy_rad=fovy_cam,
                    vertical_cam=vertical_c0,
                    target_center_w=scene_state["scene_context"]["target_center_w"],
                    front_dir_w=preclear_dir_w,
                    debug=VERBOSE_LOG,
                    cam_name=EE_CAM_NAME,
                    debug_dir=debug_overlay_dir,
                    iter_tag=f"{iter_tag}_clear",
                    orbit_angle_deg=ORBIT_ANGLE_DEG,
                )
            print(f"[FrontOcc] clear_result={clear_result}")
            clear_executed = bool(clear_result.get("executed", False))
            reobserve_after_clear_attempt = _env_flag("FRONT_OCC_REOBSERVE_AFTER_CLEAR_ATTEMPT", "1")
            if clear_executed or reobserve_after_clear_attempt:
                if clear_executed:
                    print("[FrontOcc] clear-first executed; reobserve original target before grasp planning.")
                else:
                    print(
                        "[FrontOcc] clear-first attempt did not execute successfully; "
                        "reobserve original target before any target grasp planning."
                    )
                retry_center_hint_w = np.asarray(scene_state["scene_context"]["target_center_w"], dtype=np.float32).reshape(3,)
                goto_object_centered_observation_pose(
                    env,
                    object_center_w=retry_center_hint_w,
                    cam_name=EE_CAM_NAME,
                    hover_height=OBJ_CENTER_HOVER_HEIGHT,
                    duration=1.0,
                    settle_steps=60,
                    debug=VERBOSE_LOG,
                )
                retry_refine_info = refine_object_centered_observation_pose(
                    env,
                    cam_name=EE_CAM_NAME,
                    hover_height=OBJ_CENTER_HOVER_HEIGHT,
                    duration=0.8,
                    settle_steps=60,
                    xy_tol=0.008,
                    z_tol=0.008,
                    debug=VERBOSE_LOG,
                    target_class=target_class,
                )
                if retry_refine_info.get("moved", False):
                    retry_center_hint_w = np.asarray(
                        retry_refine_info.get("object_center_w", retry_center_hint_w),
                        dtype=np.float32,
                    ).reshape(3,)
                    print(f"[ObjCenter][retry_refine] applied correction using center_w={np.round(retry_center_hint_w, 4)}")
                retry_state = _build_scene_state_from_center_view(
                    env,
                    target_class=target_class,
                    fovy_cam=fovy_cam,
                    iter_tag=f"{iter_tag}_retry",
                    debug_dir=debug_overlay_dir,
                    allow_click_init=False,
                    target_center_hint_w=retry_center_hint_w,
                )
                retry_state = _run_target_multiview_cycle(
                    env,
                    retry_state,
                    target_class=target_class,
                    fovy_cam=fovy_cam,
                    orbit_angle_deg=ORBIT_ANGLE_DEG,
                    iter_tag=f"{iter_tag}_retry",
                    multiview_debug_dir=multiview_debug_dir,
                    max_rounds=MAX_OCCLUSION_NBV_ROUNDS,
                )
                print(f"[FrontOcc] target retry multiview selected={retry_state['active_view_info'].get('best_pair_name', 'unknown')}")
                retry_preclear_dir_w = _estimate_preclear_direction_from_scene(retry_state["scene_context"])
                retry_front_occ = {"has_front_occluder": False, "front_point_count": 0, "occlusion_ratio": 0.0, "nearest_dist": np.inf, "front_instances": []}
                retry_clear_decision = {"should_clear_first": False}
                if retry_preclear_dir_w is not None:
                    retry_front_occ = detect_front_occluders(
                        scene_context=retry_state["scene_context"],
                        target_center_w=retry_state["scene_context"]["target_center_w"],
                        front_dir_w=retry_preclear_dir_w,
                        corridor_len=0.14,
                        corridor_radius=0.05,
                        min_points=200,
                        debug=VERBOSE_LOG,
                    )
                    retry_front_occ = augment_front_occluders_with_amodal(
                        scene_context=retry_state["scene_context"],
                        front_occ_info=retry_front_occ,
                        bgr=retry_state["bgr"],
                        depth=retry_state["depth"],
                        fovy_rad=fovy_cam,
                        target_class=target_class,
                        debug_dir=debug_overlay_dir,
                        iter_tag=f"{iter_tag}_retry_clear_decision",
                        debug=True,
                    )
                    retry_clear_decision = should_clear_occluder_first(retry_front_occ, debug=True)
                if retry_clear_decision.get("should_clear_first", False):
                    print("[FrontOcc] retry still requires clear-first; restart from global observation.")
                    continue
                retry_grasp_cloud = retry_state["grasp_cloud"]
                gg_retry, gg_retry_src, _, _ = _choose_grasp_from_cloud_candidates(
                    [
                        ("retry_fused_cloud", retry_state.get("fused_cloud", None)),
                        ("retry_grasp_cloud", retry_grasp_cloud),
                        ("retry_center_cloud", retry_state.get("cloud_center", None)),
                    ],
                    env=env,
                    T_wc_cap=retry_state["T_wc"],
                    vertical_cam=np.asarray(retry_state["T_wc"].R, dtype=float).T @ np.array([0.0, 0.0, 1.0], dtype=float),
                    scene_context=retry_state["scene_context"],
                    target_center_w=retry_state["scene_context"]["target_center_w"],
                    display_bgr=retry_state["bgr"],
                    display_camera=retry_state["camera"],
                    display_cloud=retry_state.get("fused_cloud", retry_grasp_cloud),
                    collision_cloud=retry_state.get("collision_cloud", retry_grasp_cloud),
                    angle_deg=65.0,
                    visual=GRASP_VISUAL,
                    debug_label="TargetRetryGrasp",
                    manual_select=GRASP_MANUAL_SELECT,
                    manual_topk=8,
                )
                print(f"[TargetGrasp] retry grasp source={gg_retry_src}")
                ok_retry = execute_grasp(env, gg_retry, retry_state["T_wc"])
                if ok_retry:
                    grasp_count += 1
                    target_class = None
                    continue
                print("[TargetGrasp] retry grasp execution failed; keep current target class and restart observation.")
                continue
            print(
                "[FrontOcc] clear-first attempt did not finish; "
                "FRONT_OCC_REOBSERVE_AFTER_CLEAR_ATTEMPT=0, return to original target center and continue target grasp."
            )
            goto_object_centered_observation_pose(
                env,
                object_center_w=np.asarray(scene_state["scene_context"]["target_center_w"], dtype=np.float32),
                cam_name=EE_CAM_NAME,
                hover_height=OBJ_CENTER_HOVER_HEIGHT,
                duration=0.7,
                settle_steps=30,
                debug=VERBOSE_LOG,
            )
            T_wc0 = get_T_wc_from_camera(env, EE_CAM_NAME)
            vertical_c0 = np.asarray(T_wc0.R, dtype=float).T @ np.array([0.0, 0.0, 1.0], dtype=float)
            scene_state["T_wc"] = T_wc0

        grasp_cloud = scene_state["grasp_cloud"]
        if grasp_cloud is None:
            raise RuntimeError("[TargetGrasp] no grasp cloud prepared.")
        print(f"[TargetGrasp] using filtered target cloud points={np.asarray(grasp_cloud.points).shape[0]}")
        try:
            gg, gg_src, _, _ = _choose_grasp_from_cloud_candidates(
                [
                    ("target_fused_cloud", scene_state.get("fused_cloud", None)),
                    ("target_grasp_cloud", grasp_cloud),
                    ("target_center_cloud", scene_state.get("cloud_center", None)),
                ],
                env=env,
                T_wc_cap=T_wc0,
                vertical_cam=vertical_c0,
                scene_context=scene_state["scene_context"],
                target_center_w=scene_state["scene_context"]["target_center_w"],
                display_bgr=scene_state["bgr"],
                display_camera=scene_state["camera"],
                display_cloud=scene_state.get("fused_cloud", grasp_cloud),
                collision_cloud=scene_state.get("collision_cloud", grasp_cloud),
                angle_deg=65.0,
                visual=GRASP_VISUAL,
                debug_label="TargetGrasp",
                manual_select=GRASP_MANUAL_SELECT,
                manual_topk=8,
            )
        except RuntimeError as e:
            print(f"[TargetGrasp] no executable grasp found: {e}")
            goto_observation_pose(env, duration=1.0)
            continue
        print(f"[TargetGrasp] selected grasp source={gg_src}")

        # 7) Execute grasp: gg is in the center-camera frame, so use T_wc0
        ok_target = execute_grasp(env, gg, T_wc0)
        if ok_target:
            grasp_count += 1
            target_class = None
        else:
            print("[TargetGrasp] grasp execution failed; keep current target class and restart observation.")

    env.close()
