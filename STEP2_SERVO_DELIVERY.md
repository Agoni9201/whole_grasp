# STEP2 — 视觉伺服递送集成（第二步）

> 目标：把 Visual Servo 工程的**伺服递送能力**接进 whole-grasp，实现：
> 抓取抬升保持 → 交接（停 move_j、MIT 阻抗接管、切腕部相机）→ 人手目标伺服 →
> 柔顺递送 → 人手交接 → 回程。
>
> 已确认的三个决策：
> 1. **单进程 + 开关分阶段**：`main.py --enable-servo-delivery` 开关默认关闭，
>    只有显式开启才执行伺服递送；抓取先单独验证。
> 2. **先保留 ArUco 目标源**验证伺服链路，人手位姿估计模块后接（接口相同，直接替换）。
> 3. **v1 不做 payload 补偿**，递送按低 K 高阻尼 + 低速方式，先跑通再加。
>
> 前置条件：STEP1 全部完成（权重真实文件到位、`tools` 包 import 修复、抓取验收通过）。

---

## 0. 第二步目标与流程

```
[第一步结束时：夹着物体停在抬升高度]
   │
   ├──(开关开启)--→ ① 交接 handoff.py
   │                  停 move_j → 确认轨迹终止 → 新建 MIT 后端（零重力+阻尼接管）
   │                  → 新 robot 实例初始化夹爪 → 相机源切到腕部相机
   │
   ├─→ ② 伺服感知：腕部相机 → 目标位姿（v1: ArUco；v2: 人手估计 hand_target.py）
   │                  → 手眼标定 → base 系目标（与第一步同坐标约定）
   │
   ├─→ ③ 闭环控制：PBVS 位姿误差 → DLS 阻尼伪逆 → 连续关节运动
   │                  → 200 Hz 笛卡尔阻抗内环（JᵀF + MuJoCo 重力补偿）
   │
   └─→ ④ 递送：柔顺接近人手 → 释放判据（手稳定 N 帧 + 距离阈值）→ 松爪 → 垂直回程 → DONE
```

---

## 1. 从 Visual Servo 工程复制（清单）

源：`Visual Servo_带阻抗的视觉伺服/`（下称 VS），目标：`whole-grasp/`

| 源（VS） | 目标（whole-grasp） | 说明 |
|---|---|---|
| `control_backends/` **整目录** | `control_backends/` | 阻抗后端。**注意 `control_backends/__init__.py` 导出了 `SdkCpvVelocityBackend`，所以 `sdk_cpv_velocity_backend.py`、`ros_move_j_backend.py` 也要一起复制**（不复制会 import 报错；这两个文件不用但保留） |
| `servo/` 整目录 | `servo/` | `__init__.py` 同时导出 jacobian_position_servo 和 pbvs，两个都要 |
| `targets/` 整目录 | `targets/` | 含 aruco_eye_in_hand_target.py（v1 目标源）和 hand_depth_target.py（占位，S5 参考） |
| `models/nero_model.py`、`nero_gravity_comp.py`、`nero_with_gripper.urdf`、`meshes/`、`__init__.py` | 合并进已有的 `models/` | 与 YOLO 权重共存；`meshes/` 是 URDF 引用的网格目录，必须带 |
| `scripts/*.py`（约 30 个） | `scripts/`（与已有 .sh 共存，无重名） | 参考实现 + 测试依赖；这些脚本内部自带 sys.path 引导，复制后原样可用 |
| `tests/` 整目录 | `tests_servo/` | VS 的离线回归（测试文件自带路径引导，放新目录不影响抓取工程的 tests/） |
| `pytest.ini` | 不复制 | 用命令行 `pytest tests_servo/` 指定路径 |

**不复制**：REFACTOR_PLAN.md / IMPEDANCE_CONTROL_PLAN.md / GRASP_DELIVERY_PLAN.md / GRASP_MERGE_PLAN.md（设计文档，留在原工程查）、`config/*.yaml`（VS 的 yaml 基本是文档性质，参数走 CLI，第二步统一进 `config/pipeline.yaml`）。

---

## 2. 环境补充（grasp_yolo26 上补装）

```bash
conda activate grasp_yolo26
conda install -c conda-forge pinocchio mujoco    # Windows / Linux 均可
# 验证：
python -c "import pinocchio; print(pinocchio.__version__)"
python -c "import mujoco; print(mujoco.__version__)"
```

其余依赖抓取环境已有（numpy/scipy/cv2/yaml/rclpy/sensor_msgs/pyAgxArm）。
注意：`pyAgxArm` 固件参数两工程一致用 V111。

---

## 3. 目录结构（第二步完成后）

```
whole-grasp/
├── main.py                        # 改造：--phase grasp|servo|all + --enable-servo-delivery
├── real_nero_grasp/               # 抓取阶段（第一步，不动）
├── control_backends/              # 【复制】VS 阻抗后端整包
│   ├── base.py / sdk_force_ctrl_backend.py / sdk_cpv_velocity_backend.py / ros_move_j_backend.py
│   └── force_ctrl/  (controller.py, mit_loop.py, gravity_comp.py, data_types.py,
│                     impedance_profile.py, real_velocity.py, cartesian_impedance.py)
├── servo/                         # 【复制】PBVS + 执行安全
│   └── pbvs_pose_servo.py / jacobian_position_servo.py / execution_safety.py
├── targets/                       # 【复制】目标源
│   ├── aruco_eye_in_hand_target.py    # v1 调试目标源
│   ├── hand_depth_target.py           # 占位（未来手跟随）
│   └── hand_target.py                 # 【新增】人手位姿目标（S5）
├── handoff/                       # 【新目录】交接层
│   └── grasp_to_servo_handoff.py  # 【新增】停 move_j → MIT 接管 → 切相机 → 初始化夹爪
├── servo_delivery.py              # 【新增】伺服递送主程序（根目录，与 main.py 平级）
├── config/
│   └── pipeline.yaml              # 【新增】两相机/标定/开关统一配置
├── models/                        # 合并：+nero_model.py / nero_gravity_comp.py / URDF / meshes/
├── scripts/                       # +VS 的 30 个 .py（参考/测试支持）
├── tools/                         # 第一步的（含 real_nero_point_test.py）
└── tests_servo/                   # 【复制】VS 测试
```

---

## 4. 新写模块

### 4.1 `handoff/grasp_to_servo_handoff.py` — 交接层（★新增）

| 函数 | 作用 |
|---|---|
| `stop_grasp_control(arm, args)` | 停 move_j：调 `arm.stop()` → `arm.terminate_motion(reason)` 发 CAN 控制帧清空轨迹队列 → 确认轨迹终止（复用 arm.py 的 `_send_ordinary_trajectory_terminate` 逻辑）；设置 `_suppress_program_exit_estop`，退出时不触发程序急停 |
| `build_servo_backend(cfg)` | 创建 `SdkForceCtrlBackend(zero_gravity=True, gravity_compensation=True, pin_model=NeroPinocchioModel(...).clone(), torque_limit=..., impedance_torque_limit=...)` 并 `connect()`；等待 7 关节反馈新鲜 |
| `init_gripper_on_backend(backend)` | 在新 robot 实例上 `init_effector(AGX_GRIPPER)` + 钳位（宽度 0.085 m / 力 0.50 N）——**递送结束松爪靠它**（MIT 后端的 robot 实例与抓取阶段的 arm 对象不是同一个，必须重新初始化） |
| `switch_camera(cfg, "global"→"wrist")` | 纯配置切换：换 ROS 订阅话题 + 换手眼标定 JSON + 换挂载帧约定；返回新 CameraConfig |
| `handoff(arm, args, cfg)` | 编排：①确认在抬升保持状态（TCP 高于桌面、夹爪闭合）→ ②`stop_grasp_control` → ③抓取 arm 对象退出（`__exit__` 断开）→ ④`build_servo_backend` + `init_gripper_on_backend` → ⑤`switch_camera` → 返回 `(backend, camera_cfg)` |

交接顺序**不可颠倒**：先停 move_j 并确认轨迹终止，再断开抓取 arm 对象，最后启动 MIT 内环。两个控制源共享 can0，任何时刻只有一个在发 CAN。

### 4.2 `servo_delivery.py` — 伺服递送主程序（★新增）

参考实现：VS 的 `scripts/aruco_camera_jacobian_servo_sdk_execute.py`（伺服主循环模板）。
本文件是它的「递送版」：目标从 ArUco marker 换成人手，状态机按递送阶段编排。

**整体结构**（rclpy Node，与 VS 执行脚本同构）：

```
订阅：腕部相机 Image + CameraInfo（话题来自 config/pipeline.yaml）
30 Hz 定时器：
  q_now ← backend 快照
  target ← 目标源.detect(最新图像, K, D, q_now)     # v1: ArUco；v2: hand_target
  servo_out ← PBVSPoseServo.update(q_now, target...)
  safety ← ExecutionSafetyMonitor.evaluate(q_now, servo_out)
  _build_impedance_command(q_now, servo_out)         # 移植自 VS 执行脚本:1532
      → 积分 tcp_twist 进名义目标（target_tcp/R_target，限速+抗饱和 leash）
      → 刷新 ImpedanceConfig → backend.set_cartesian_impedance(cfg)
  内环 200 Hz（mit_loop）：τ = JᵀF + 重力补偿 → move_mit
```

**状态机**：

| 状态 | 动作 | 完成条件 |
|---|---|---|
| `HOLD_LIFTED` | MIT 接管完成，保持抬升位姿（安装 ImpedanceConfig 目标=当前 TCP，零初始弹力误差） | 内环无故障、7 关节反馈新鲜、姿态误差 < 阈值 |
| `SWITCH_CAMERA` | 切腕部相机 + 手眼标定 | 收到有效图像 + 标定加载成功 |
| `TRACK_HAND` | PBVS 跟踪人手目标（标准参数档） | 位置误差 < 0.03 m 且姿态误差 < 5° 持续 N 帧 |
| `APPROACH` | 低速靠近（线速 ≤ 0.03 m/s，误差 leash 更小） | 到达递送距离（人手前 ~0.10 m） |
| `HANDOVER` | 保持位姿等待人手；**释放判据双条件**：手稳定 N 帧 **且** 距离阈值（手接近物体/掌心闭合信号可选） | 判据满足或超时回退 |
| `RELEASE` | `GripperAdapter.open(...)` 松爪（用 MIT 后端的 robot 实例） | 宽度反馈到位 |
| `RETREAT` | 垂直撤离到安全高度 | 高于递送点 0.15 m |
| `DONE` | 停内环、断开，打印全程日志 | — |

**安全机制（全部沿用 VS 已验证实现）**：
- 内环故障 → `_check_inner_impedance_fault`（移植自 VS:939）→ fail-closed 急停
- 阻抗不稳定（瞬时 qdot > 1.0 / RMS > 0.3）→ `_check_impedance_instability`（VS:1805）→ 撤阻抗 + 重力补偿安全保持
- 目标丢失 → ExecutionSafetyMonitor 保持位姿（不追旧目标超 0.45 s）
- 限位 → 硬限位急停；`--execute` 门槛保留

### 4.3 `targets/hand_target.py` — 人手位姿目标（★新增，S5 阶段实现）

**接口设计**：`detect(image, camera_matrix, dist_coeffs, q_arm)` 返回与
`ArucoEyeInHandTarget.detect` **同字段结构**（`hand_center_base`、`hand_normal_base`、
`hand_tangent_base`、`T_base_hand`），这样 PBVS 和伺服主程序**零改动**切换目标源。

| 函数 | 作用 |
|---|---|
| `detect(image, K, D, q_arm)` | 检测人手 → 求 base 系手心位姿 + 手面法向 |
| `_hand_keypoints(bgr)` | v1 用 **MediaPipe Hands**（`pip install mediapipe`，免训练、CPU 实时）输出 21 个手关键点 |
| `_palm_pose_from_keypoints(kpts_3d, K, depth)` | 由掌根+指尖关键点拟合手掌平面 → 掌心中心 + 法向（法向取手掌朝向相机的方向，与 ArUco normal 同约定） |
| `_to_base(pose_cam, q_arm)` | 复用 ArUco 目标的手眼链：`fk_to_4x4(q) @ T_mount_cam @ T_camera_hand` |
| `load_handeye(...)` | 直接复用 `aruco_eye_in_hand_target.py` 的 `load_transform_file` |

目标源切换开关：`servo_delivery.py --target-source aruco|mediapipe`（默认 aruco）。
**先用 ArUco 贴在人手上验证伺服递送全链**，再开 mediapipe。

### 4.4 `main.py` 改造（两阶段编排 + 开关）

```
main.py --phase grasp|servo|all（默认 all）--enable-servo-delivery（默认关）
```

```python
# 伪代码（实际写法参考 real_nero_grasp/runtime.main 的 CLI 风格）
def main():
    args = parse_args()
    if args.phase in ("grasp", "all"):
        result = real_nero_grasp.runtime.run_single_grasp(args)   # 抓取阶段（原样）
        # 抓取结束：夹着物体停在抬升保持（arm 上下文已退出、轨迹已停）
    if args.phase == "servo":
        # 仅伺服调试模式：需先手动把臂摆到安全位姿
        run_servo_delivery(args)
    elif args.phase == "all":
        if args.enable_servo_delivery and result 成功:
            run_servo_delivery(args)        # 内部先执行 handoff()
        else:
            print("servo delivery disabled; holding lifted pose.")
```

`run_servo_delivery(args)` = `handoff()` + 4.2 的状态机。开关关闭时流程在抬升保持处结束，
**抓取阶段行为和 STEP1 完全一致**，验证互不干扰。

---

## 5. 实施顺序（S1–S6，每步验收）

| 步 | 内容 | 验收（Windows 可做） |
|---|---|---|
| S1 | 按 §1 复制 VS 四个包 + models 合并 + scripts + tests_servo | `pytest tests_servo/ -q` 通过（VS 原 175 测试） |
| S2 | 装 pinocchio + mujoco（§2） | import 自检两条命令 |
| S3 | `config/pipeline.yaml`（global/wrist 两相机块 + 标定 JSON + 挂载约定）+ `handoff/grasp_to_servo_handoff.py` 骨架 | 离线单测：fake arm / fake backend 走通 `handoff()` 顺序 |
| S4 | `servo_delivery.py`（`--target-source aruco`；从 VS 执行脚本移植 `_build_impedance_command`/故障检查） | dry-run 模式 + 离线测试；`python servo_delivery.py --help` |
| S5 | `targets/hand_target.py`（MediaPipe v1） | 离线图片/视频：手心中心与法向输出合理 |
| S6 | `main.py` 改造（--phase/--enable-servo-delivery） | 离线：`--phase grasp` 路径不受影响；`--phase all`（开关关）在抬升保持处结束 |

> 注意 S4 移植时**不要照搬** VS 执行脚本的 preflight/settle/ArUco 专属逻辑——
> 只移植阻抗命令构建、故障检查、限幅、日志四类核心函数；状态机按 §4.2 递送阶段重写。

---

## 6. 切 Linux 后的实机验证清单（一次性执行，按顺序）

```bash
# 0. 环境自检
bash scripts/check_env.sh
python -c "import pinocchio, mujoco, mediapipe; print('servo deps OK')"

# 1. 抓取回归（确认 STEP2 改动没破坏第一步）
bash scripts/run_grasp_safe.sh --calib-json <标定.json> --target-class <物体>
#    验收：和 STEP1 一样，抓取抬升稳定保持

# 2. 空载伺服递送（ArUco 贴在手背上，不夹物体）
python servo_delivery.py --phase servo --target-source aruco --execute \
    --handeye <腕部相机标定.json> --handeye-convention link7_cam ...
#    验收：跟踪稳定、接近/递送/回程各阶段正常、故障急停路径正常

# 3. 全流程（开关开）
python main.py --phase all --enable-servo-delivery \
    --calib-json <全局相机标定.json> --handeye <腕部相机标定.json> ...
#    验收：抓取 → 交接（无 CAN 冲突告警）→ 递送到人手 → 松爪 → 回程

# 4. 人手估计切换
python main.py --phase all --enable-servo-delivery --target-source mediapipe ...
#    验收：无 ArUco 情况下完成递送
```

每个阶段先空载、再真物体；递送全程有人在旁、急停可用。

---

## 7. 铁律与风险

| # | 风险 | 对策 |
|---|---|---|
| 1 | 交接时 CAN 双控制源 | 严格按 §4.1 顺序：停 move_j → 确认轨迹终止 → 断开抓取 arm → 再启 MIT |
| 2 | 夹爪实例跟着 arm 对象断开 | 递送松爪必须在 MIT 后端的 robot 实例上重新 `init_effector`（§4.1） |
| 3 | 未做 payload 补偿，抬物递送可能下垂 | v1：接近/递送用低 K 高阻尼低速；发现下垂先降速，再考虑 payload 补偿（后续迭代） |
| 4 | 人手不稳定导致目标跳变 | PBVS 目标滤波 + 目标跳变拒绝（VS 已实现）+ 释放判据双条件 |
| 5 | ArUco 与 MediaPipe 坐标约定不一致 | hand_target 接口字段与 ArUco 目标严格同构；上线前先贴 marker 对比两种目标源输出 |
| 6 | 两相机标定混乱 | pipeline.yaml 单一配置来源；global 相机标定用 `flange_camera_xyzw` 约定、腕部相机用 VS 的 link7_cam 约定，各自独立成块 |

**通用铁律**（沿用）：`--execute` 门槛；急停可用、工作空间有人；所有阶段打印 q/目标/TCP/力矩/限幅状态；任何异常先停内环再排查。
