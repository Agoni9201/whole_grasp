import os
import os.path
import sys

os.environ.setdefault("MUJOCO_GL", "glfw")
os.environ.setdefault("MUJOCO_VIEWER", "1")

sys.path.append('../../manipulator_grasp')

import time
import numpy as np
import spatialmath as sm
import mujoco
import mujoco.viewer
import cv2

from manipulator_grasp.arm.robot import Robot, Nero7
from manipulator_grasp.arm.motion_planning import *
from manipulator_grasp.utils import mj


def _make_renderer(model, height, width):
    renderer_cls = getattr(mujoco, "Renderer", None)
    if renderer_cls is None and hasattr(mujoco, "renderer"):
        renderer_cls = getattr(mujoco.renderer, "Renderer", None)
    if renderer_cls is None:
        raise AttributeError("No MuJoCo Renderer class found in this mujoco package.")
    return renderer_cls(model, height=height, width=width)


def _nero_gripper_public_to_ctrl(value):
    """Map existing 0(open)..255(closed) commands to Nero gripper travel."""
    value = float(np.clip(value, 0.0, 255.0))
    return (255.0 - value) / 255.0 * 0.05


class UR5GraspEnv:

    def __init__(self):
        self.sim_hz = 500

        self.mj_model: mujoco.MjModel = None
        self.mj_data: mujoco.MjData = None
        self.robot: Robot = None
        self.joint_names = []
        self.robot_q = np.zeros(7)
        self.robot_T = sm.SE3()
        self.T0 = sm.SE3()

        self.mj_renderer: mujoco.Renderer = None
        self.mj_depth_renderer: mujoco.Renderer = None
        self.ee_side_renderer: mujoco.Renderer = None
        self.mj_viewer: mujoco.viewer.Handle = None
        self._ee_side_window_enabled = False
        self._ee_side_window_name = "EE Side Camera"
        self._ee_side_window_cam = "ee_side_depth_cam"
        self._ee_side_window_interval = 10
        self._ee_side_window_step = 0
        self._ee_side_window_ready = False

        self.height = 640 # 256 640 720
        self.width = 640 # 256 640 1280
        self.fovy = np.pi / 4

    def enable_ee_side_camera_window(self,
                                     enabled=True,
                                     cam_name="ee_side_depth_cam",
                                     window_name="EE Side Camera",
                                     interval=10):
        self._ee_side_window_enabled = bool(enabled)
        self._ee_side_window_cam = str(cam_name)
        self._ee_side_window_name = str(window_name)
        self._ee_side_window_interval = max(1, int(interval))
        self._ee_side_window_step = 0
        if self._ee_side_window_enabled:
            print(
                f"[EESideView] enabled cam={self._ee_side_window_cam} "
                f"window='{self._ee_side_window_name}' interval={self._ee_side_window_interval}"
            )

    def _update_ee_side_camera_window(self, force=False):
        if not self._ee_side_window_enabled or self.mj_model is None or self.mj_data is None:
            return
        self._ee_side_window_step += 1
        if (not force) and (self._ee_side_window_step % self._ee_side_window_interval) != 0:
            return
        try:
            if self.ee_side_renderer is None:
                h = int(os.environ.get("EE_SIDE_VIEW_HEIGHT", "480"))
                w = int(os.environ.get("EE_SIDE_VIEW_WIDTH", "640"))
                self.ee_side_renderer = _make_renderer(self.mj_model, height=h, width=w)
            cam_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_CAMERA, self._ee_side_window_cam)
            if cam_id < 0:
                print(f"[EESideView] camera not found: {self._ee_side_window_cam}")
                self._ee_side_window_enabled = False
                return
            if hasattr(self.ee_side_renderer, "disable_depth_rendering"):
                self.ee_side_renderer.disable_depth_rendering()
            if hasattr(self.ee_side_renderer, "disable_segmentation_rendering"):
                self.ee_side_renderer.disable_segmentation_rendering()
            self.ee_side_renderer.update_scene(self.mj_data, camera=cam_id)
            rgb = self.ee_side_renderer.render()
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            if not self._ee_side_window_ready:
                cv2.namedWindow(self._ee_side_window_name, cv2.WINDOW_NORMAL)
                self._ee_side_window_ready = True
            cv2.imshow(self._ee_side_window_name, bgr)
            cv2.waitKey(1)
        except Exception as exc:
            print(f"[EESideView] disabled: {type(exc).__name__}: {exc}")
            self._ee_side_window_enabled = False

    def reset(self):

        # 初始化 MuJoCo 模型和数据
        filename = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'assets', 'scenes', 'scene.xml')
        self.mj_model = mujoco.MjModel.from_xml_path(filename)
        self.mj_data = mujoco.MjData(self.mj_model)
        mujoco.mj_forward(self.mj_model, self.mj_data)

        # 创建机械臂实例并设置其基座位置
        self.robot = Nero7()
        self.robot.set_base(mj.get_body_pose(self.mj_model, self.mj_data, "nero_base"))
        # 设置机械臂的初始关节角度，并同步到 MuJoCo 模型
        self.robot_q = self.robot.q0.copy()
        self.robot.set_joint(self.robot_q)
        self.joint_names = self.robot.joint_names
        [mj.set_joint_q(self.mj_model, self.mj_data, jn, self.robot_q[i]) for i, jn in enumerate(self.joint_names)]
        self.mj_data.ctrl[:self.robot.dof] = self.robot_q
        if self.mj_data.ctrl.size > self.robot.dof:
            self.mj_data.ctrl[self.robot.dof] = _nero_gripper_public_to_ctrl(0.0)
            for joint_name, joint_q in (("gripper_joint1", 0.05), ("gripper_joint2", -0.05)):
                try:
                    mj.set_joint_q(self.mj_model, self.mj_data, joint_name, joint_q)
                except Exception:
                    pass
        mujoco.mj_forward(self.mj_model, self.mj_data)
        # Nero gripper is mounted directly at the joint7/flange frame in nero7.xml.
        robot_tool = sm.SE3.Trans(0.0, 0.0, 0.1555)
        self.robot.set_tool(robot_tool)
        # 计算机械臂末端执行器的初始姿态。
        self.robot_T = self.robot.fkine(self.robot_q)
        self.T0 = self.robot_T.copy()

        # 创建两个渲染器实例，分别用于生成彩色图像和深度图
        self.mj_renderer = _make_renderer(self.mj_model, height=self.height, width=self.width)
        self.mj_depth_renderer = _make_renderer(self.mj_model, height=self.height, width=self.width)
        # 更新渲染器中的场景数据
        self.mj_renderer.update_scene(self.mj_data, 0)
        self.mj_depth_renderer.update_scene(self.mj_data, 0)
        # 启用深度渲染
        self.mj_depth_renderer.enable_depth_rendering()
        
        # 初始化被动查看器。新机器的 GLX/驱动环境不稳时可设置 MUJOCO_VIEWER=0。
        if os.environ.get("MUJOCO_VIEWER", "1").strip().lower() not in {"0", "false", "no", "off"}:
            self.mj_viewer = mujoco.viewer.launch_passive(self.mj_model, self.mj_data)
            # 为了方便观察
            self.mj_viewer.cam.lookat[:] = [1.8, 1.1, 1.7]  # 对应XML中的center
            self.mj_viewer.cam.azimuth = 210      # 对应XML中的azimuth
            self.mj_viewer.cam.elevation = -35    # 对应XML中的elevation
            self.mj_viewer.cam.distance = 1.2     # 根据场景调整的距离值
            self.mj_viewer.sync() # 立即同步更新

    def close(self):
        if self.mj_viewer is not None:
            try:
                self.mj_viewer.close()
            except Exception:
                pass
            self.mj_viewer = None
        if self.mj_renderer is not None:
            try:
                self.mj_renderer.close()
            except Exception:
                pass
            self.mj_renderer = None
        if self.mj_depth_renderer is not None:
            try:
                self.mj_depth_renderer.close()
            except Exception:
                pass
            self.mj_depth_renderer = None
        if self.ee_side_renderer is not None:
            try:
                self.ee_side_renderer.close()
            except Exception:
                pass
            self.ee_side_renderer = None
        if self._ee_side_window_ready:
            try:
                cv2.destroyWindow(self._ee_side_window_name)
            except Exception:
                pass
            self._ee_side_window_ready = False

    def step(self, action=None):
        qpos_prev = self.mj_data.qpos.copy()
        ctrl_prev = self.mj_data.ctrl.copy()
        if action is not None:
            n = min(len(action), self.mj_data.ctrl.size)
            self.mj_data.ctrl[:n] = action[:n]
            if n > self.robot.dof:
                self.mj_data.ctrl[self.robot.dof] = _nero_gripper_public_to_ctrl(action[self.robot.dof])
        elif self.robot is not None:
            joint = np.asarray(self.robot.get_joint(), dtype=float).reshape(-1)
            n = min(joint.size, self.mj_data.ctrl.size)
            self.mj_data.ctrl[:n] = joint[:n]
        mujoco.mj_step(self.mj_model, self.mj_data)
        qacc_limit = float(os.environ.get("MUJOCO_QACC_ROLLBACK_LIMIT", "1e12"))
        max_qacc = float(np.nanmax(np.abs(self.mj_data.qacc))) if self.mj_data.qacc.size else 0.0
        unstable = (
            (not np.all(np.isfinite(self.mj_data.qpos)))
            or (not np.all(np.isfinite(self.mj_data.qvel)))
            or (not np.all(np.isfinite(self.mj_data.qacc)))
            or max_qacc > qacc_limit
        )
        if unstable:
            self.mj_data.qpos[:] = qpos_prev
            self.mj_data.qvel[:] = 0.0
            self.mj_data.ctrl[:] = ctrl_prev
            self.mj_data.qacc[:] = 0.0
            mujoco.mj_forward(self.mj_model, self.mj_data)
            if os.environ.get("MUJOCO_WARN_UNSTABLE", "1").strip().lower() not in {"0", "false", "no", "off"}:
                print(f"[MuJoCo] unstable step rolled back; max |qacc|={max_qacc:.3g}.")
            return

        if self.mj_viewer is not None:
            self.mj_viewer.sync()
        self._update_ee_side_camera_window()

    def render(self):
        '''
        常用于强化学习或机器人控制任务中，提供环境的视觉观测数据。
        '''
        # 更新渲染器中的场景数据
        self.mj_renderer.update_scene(self.mj_data, 0)
        self.mj_depth_renderer.update_scene(self.mj_data, 0)
        # 渲染图像和深度图
        return {
            'img': self.mj_renderer.render(),
            'depth': self.mj_depth_renderer.render()
        }



if __name__ == '__main__':
    env = UR5GraspEnv()
    env.reset()
    for i in range(10000):
        env.step()
    imgs = env.render()
    env.close()
