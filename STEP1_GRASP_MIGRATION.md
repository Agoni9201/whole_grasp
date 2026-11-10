# STEP1 — 抓取工程迁移到 whole-grasp（第一步）

> 目标：把 `real-nero-grasp-main` 的**抓取阶段**（全局相机 + YOLO + GraspNet + move_j 执行）
> 完整搬进新工程 `whole-grasp`，跑通「识别 → 规划 → 抓取 → 抬升保持」，并在抬升处
> 留好和第二步（腕部相机视觉伺服递送）的交接接口。
>
> 本文档按「照着做」的方式写：先建什么目录、再复制哪些文件、主程序是什么、
> 每个函数干什么、最后怎么验收。

---

## 0. 全流程回顾与第一步的定位

完整流程（你的图）：

```
┌─────────────── 抓取工程（离散 感知-规划-执行）──────────────┐   ← 第一步做这部分
│ 全局 RGB-D 相机                                              │
│  → YOLO 实例分割 → 深度反投影 → 目标点云                       │
│  → 多视角观察与清障决策                                       │
│  → GraspNet 抓取候选 → TRAC-IK 求逆解 → 碰撞约束 Beam Search   │
│  → [可选] 抓取后 RRT-Connect 搬运                              │
│  → move_j 执行 → 夹爪力/宽度接触确认 → 抓住并抬升               │
└───────────────────────────┬────────────────────────────────┘
                            ▼  ← 过渡：抓取→伺服交接、相机切换（第二步做）
┌────────────── 视觉伺服工程（连续 感知-纠偏-递送）──────────────┐   ← 第二步做这部分
│ 腕部相机 → 直接人手位姿估计（新，替代 ArUco）→ 手眼标定          │
│  → 人手目标位姿（基座系）→ PBVS 位姿误差 → DLS Jacobian         │
│  → 连续关节运动 → 笛卡尔阻抗控制 → 柔顺递送 → 人手交接           │
└──────────────────────────────────────────────────────────────┘
```

**第一步的交付物**：抓取阶段能独立、完整、安全地运行；结束时机械臂停在
「抓住物体 + 抬升保持」状态（这是第二步的起点）。

**第一步不做的**：视觉伺服、人手位姿估计、相机切换、阻抗控制（全部第二步，来自 Visual Servo 工程）。

---

## 1. 迁移三原则（为什么这么搬）

1. **整包复制 `real_nero_grasp/`，不改包名。**
   包内 31 个文件互相 `from real_nero_grasp import ...` / `from real_nero_grasp.xxx import ...`，
   tests 里 38 个测试也这么引用。改包名 = 上百处 import 全改，风险大收益小。
   新工程照样干净——它只包含实机抓取需要的东西。

2. **保持原工程的相对目录布局。**
   `paths.py` 用「包目录的上级 = 工程根」锚定所有路径：
   - IK 的 URDF：`PROJECT_ROOT/manipulator_grasp/nero_urdf/.../nero_description.urdf`
   - GraspNet import 目录：`PROJECT_ROOT/graspnet-baseline/{models,dataset,utils}`
   - YOLO 默认权重：`PROJECT_ROOT/yolov8s-seg.pt`
   所以 `graspnet-baseline/`、`manipulator_grasp/`、`models/` 必须原样放在新工程根目录。

3. **只复制实机链路需要的。** 仿真入口、训练代码、旧 RealSense demo、备份脚本不搬。

---

## 2. 目标目录结构（第一步完成后）

```
whole-grasp/
├── README.md                             # 【新建】项目说明（两阶段流程）
├── STEP1_GRASP_MIGRATION.md              # 本文档
├── main.py                               # 【新建】主程序入口（抓取阶段）
│
├── real_nero_grasp/                      # 【整包复制】抓取阶段核心包（31 个文件）
│   ├── runtime.py        # 入口门面：main() / run_single_grasp() / CLI
│   ├── workflow.py       # 抓取主流程编排
│   ├── grasp_workflow.py # 规划+执行抓取
│   ├── arm.py            # 臂+夹爪封装（SDK can0）
│   ├── arm_execution.py  # move_j 运动执行与到位等待
│   ├── perception.py     # 采图/掩码/点云
│   ├── grasp_geometry.py / grasp_candidates.py / grasp_planner.py
│   ├── ik_solver.py / collision.py / motion_planning.py / post_grasp_rrt.py
│   ├── active_view.py / active_view_scoring.py / active_motion.py
│   ├── clear_policy.py / pointcloud_fusion.py / depth_processing.py
│   ├── runtime_safety.py / stop_policy.py / execution.py / session_control.py
│   ├── cli.py / paths.py / types.py / transforms.py / visualization.py
│   └── __init__.py / __main__.py
│
├── grasp_active_three_views_sceneaware_asymmetric_clear_nbv_obstacle_yolo26seg.py
│                                        # 【复制】GraspNet 推理库（约 9900 行，暂整文件复制）
├── cv_process_updated_yolo26seg.py      # 【复制】YOLO26 分割模块
├── kinematic_trac_ik.py                 # 【复制】Trac-IK 封装
├── real_nero_single_grasp_runner.py     # 【复制】旧入口（兼容用，新入口是 main.py）
│
│  ↑ 以上 4 个文件必须留在根目录：它们是「运行时模块」，被包内模块和 23 个测试直接
│    import（visualization.py → 大脚本；clear_policy.py → cv_process；
│    ik_solver.py → kinematic_trac_ik；23 个 tests → runner），挪走要改代码。
│    彻底的整理方案（挪进包内子模块）放在第二步和 GraspNet 抽取一起做，见 §9。
│
├── graspnet-baseline/                   # 【整目录复制】GraspNet 模型/pointnet2/graspnetAPI/knn/utils/dataset
├── manipulator_grasp/                   # 【整目录复制】URDF + 大脚本 import 链依赖
├── models/                              # 【复制】YOLO 权重（food5_*.pt、yolo26l/m-seg.pt）
├── logs/log_rs/checkpoint-rs.tar        # 【复制】GraspNet 权重（必须！）
│
│  ↑ graspnet-baseline/ 与 manipulator_grasp/ 的位置被 paths.py 和大脚本的 sys.path
│    引导锚定（PROJECT_ROOT/graspnet-baseline/...），第一步不动，第二步可 vendor 化。
│
├── tests/                               # 【复制】38 个回归测试
│
├── tools/                               # 【新建】独立工具脚本收进来（互相 import 都在圈内）
│   ├── real_nero_point_test.py          # 手眼标定转换工具
│   ├── verify_handeye_point.py          # 点选手眼验证
│   ├── verify_handeye_aruco_consistency.py  # 手眼一致性验证
│   ├── nero_handeye_calib_pose_sweep.py # 手眼标定姿态采集（需加 3 行 sys.path 引导，见 §3.3）
│   ├── record_nero_observation_pose_seeds.py  # 观察位姿种子录制
│   ├── generate_view_rgbd_pointcloud.py # 离线 RGB-D → 点云
│   ├── view_fused_pointcloud.py         # 融合点云查看
│   └── teleop_nero_joints.py            # 手动关节调试
├── scripts/                             # 【新建】启动脚本
│   ├── run_grasp.sh                     # 基础启动（改自 run_real_nero_clean.sh）
│   ├── run_grasp_safe.sh                # 安全参数组合（改自 run_real_nero_safe.sh）
│   ├── launch_orbbec.sh                 # 相机驱动（改自 run_orbbec_gemini330_grasp.sh）
│   └── check_env.sh                     # 环境自检（改自 check_real_nero_env.sh）
│
└── config/                              # 【新建，预留】第二步统一配置（相机切换等）
```

---

## 3. 迁移操作（照着做）

### 3.1 建目录骨架

```bash
cd /path/to/whole-grasp
mkdir -p tools scripts config logs/log_rs models tests
```

### 3.2 复制核心包与顶层脚本

源目录记为 `SRC=real-nero-grasp-main/real-nero-grasp-main`（下同）：

```bash
# 核心包：整包复制，包名不变
cp -r "$SRC/real_nero_grasp" .

# 顶层脚本（实机链路直接依赖的）
cp "$SRC/grasp_active_three_views_sceneaware_asymmetric_clear_nbv_obstacle_yolo26seg.py" .
cp "$SRC/cv_process_updated_yolo26seg.py" .
cp "$SRC/kinematic_trac_ik.py" .
cp "$SRC/real_nero_single_grasp_runner.py" .

# 独立工具脚本 → tools/（它们互相 import，同目录可正常工作）
cp "$SRC/real_nero_point_test.py" "$SRC/verify_handeye_point.py" \
   "$SRC/verify_handeye_aruco_consistency.py" "$SRC/nero_handeye_calib_pose_sweep.py" \
   "$SRC/record_nero_observation_pose_seeds.py" "$SRC/generate_view_rgbd_pointcloud.py" \
   "$SRC/view_fused_pointcloud.py" "$SRC/teleop_nero_joints.py" tools/
```

### 3.3 tools/ 里唯一需要的一处代码改动

放进 `tools/` 后，`nero_handeye_calib_pose_sweep.py` 里有一行
`from real_nero_single_grasp_runner import (...)`，但 runner 在根目录、不在 `tools/` 里，
直接运行会 import 失败。修复：在这个文件的**最顶部**（所有 import 之前）加 3 行
sys.path 引导：

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # 把工程根目录加入 import 路径
```

其余 7 个工具脚本互相 import（`from real_nero_point_test import ...`）都在 `tools/`
圈内，不需要任何改动。以后如果有工具要 import `real_nero_grasp` 包，加同样的 3 行即可。

# 回归测试
cp -r "$SRC/tests" .
```

### 3.4 复制 GraspNet 依赖（大文件，耐心等）

```bash
# GraspNet 模型库（含 pointnet2 CUDA 扩展源码、graspnetAPI）
cp -r "$SRC/graspnet-baseline" .

# manipulator_grasp：大脚本 import 它（motion_planning、ur5_grasp_env），
# 且 IK 用的 URDF 在里面 → 整目录复制
cp -r "$SRC/manipulator_grasp" .

# 权重（GraspNet checkpoint + YOLO）
cp -r "$SRC/logs/log_rs/checkpoint-rs.tar" logs/log_rs/
cp "$SRC/models/"*.pt models/
```

> ⚠️ **Git LFS 指针陷阱（已踩过）**：如果复制过来的权重文件只有 ~133 字节，
> 开头是 `version https://git-lfs.github.com/spec/v1`，说明你拿到的是 Git LFS
> 指针而不是真实文件——**它们不能用**。本仓库没有配置 LFS remote、本地也没有
> LFS 缓存，`git lfs pull` 拉不下来。真实文件必须从**机器人（Linux 实机）**
> 上拷贝，路径在原工程部署目录下。真实大小对照：
>
> | 文件 | 真实大小 |
> |---|---|
> | `logs/log_rs/checkpoint-rs.tar` | 12.5 MB |
> | `models/yolo26l-seg.pt` | 63.7 MB |
> | `models/yolo26m-seg.pt` | 54.8 MB |
> | `models/food5_visible_v3_full_reviewed_corrected_best.pt` | 54.5 MB |
> | `models/food5_amodal_real_v2_best.pt` | 54.5 MB |
>
> 拷贝后用 `ls -la` 核对大小与上表一致才算迁移完成。注意 checkpoint 的目录层级
> 是 `logs/log_rs/`（大脚本按相对路径加载），不是根目录的 `log_rs/`。

> 注意：`graspnet-baseline/pointnet2/` 与 `knn/` 里的 CUDA 扩展（`grasp_nms` 等）
> 是编译产物，依赖运行环境。第一步沿用 conda 环境 `grasp_yolo26`（已编译好），
> 所以**不需要重编译**。如果以后换环境，才需要在 `pointnet2/` 里 `python setup.py install`。

### 3.5 新建主程序 `main.py`

内容如下（等价于原 `real_nero_single_grasp_runner.py`，只是换了名字）：

```python
#!/usr/bin/env python3
"""whole-grasp 主程序入口（第一步：抓取阶段）。

等价于原 real-nero-grasp 工程的 real_nero_single_grasp_runner.py：
所有 CLI 参数、主流程都在 real_nero_grasp.runtime 里。
"""

from real_nero_grasp import runtime as _runtime

if __name__ == "__main__":
    raise SystemExit(_runtime.main())
```

运行方式（两种等价）：

```bash
python main.py --calib-json <标定.json> [其他参数]
python -m real_nero_grasp --calib-json <标定.json> [其他参数]   # 包自带的 __main__.py
```

### 3.6 新建启动脚本 `scripts/run_grasp.sh`

改自原 `run_real_nero_clean.sh`，只改两处：去掉 `/home/zgj/...` 写死路径（换成你的机器
实际路径），入口改为 `main.py`：

```bash
#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

CONDA_SH="$HOME/anaconda3/etc/profile.d/conda.sh"          # ← 改成你的 conda 路径
ROS_SETUP="/opt/ros/humble/setup.bash"
NERO_SETUP="$HOME/Zgj/nero_handeye/install/setup.bash"     # ← 改成你的相机工作空间路径
USER_SITE="$HOME/.local/lib/python3.10/site-packages"

unset PYTHONPATH LD_LIBRARY_PATH
export PYTHONNOUSERSITE=1
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"
export YOLO_CONFIG_DIR="${YOLO_CONFIG_DIR:-$PROJECT_DIR/.ultralytics}"
mkdir -p "$MPLCONFIGDIR" "$YOLO_CONFIG_DIR/Ultralytics"

source "$CONDA_SH"
set +u; conda activate grasp_yolo26; set -u

CONDA_SITE="$(python - <<'PY'
import site
print(site.getsitepackages()[0])
PY
)"
CONDA_LIB="$CONDA_PREFIX/lib"

set +u
source "$ROS_SETUP"
source "$NERO_SETUP"
set -u

export PYTHONPATH="$CONDA_SITE:${PYTHONPATH-}:$USER_SITE"
export LD_LIBRARY_PATH="$CONDA_LIB:${LD_LIBRARY_PATH-}"

cd "$PROJECT_DIR"
exec python main.py "$@"
```

### 3.7 新建安全启动脚本 `scripts/run_grasp_safe.sh`

改自原 `run_real_nero_safe.sh`，保留它的全部安全参数（这些参数是实机安全的关键，
**第一次实机运行必须用这套**）：

```bash
#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

exec bash "$PROJECT_DIR/scripts/run_grasp.sh" \
  "$@" \
  --nero-speed-percent 5 \                          # 关节速度 5%
  --motion-runtime-torque-guard \
  --motion-runtime-torque-delta 8.0 \               # 力矩跳变护栏 Δ=8
  --motion-runtime-collision-guard \
  --motion-path-min-clearance 0.005 \               # 路径最小间隙 5mm
  --table-guard-enable --table-z 0.0 \              # 桌面护栏
  --slow-close-gripper --gripper-force 0.50 \       # 慢闭夹爪 + 力限 0.5N
  --gripper-contact-confirm-samples 2 \             # 接触确认
  --require-gripper-contact --require-gripper-status \
  --grasp-stage-beam-search \                       # Beam Search 抓取序列
  --adaptive-lift-heights --adaptive-lift-step 0.020 \  # 自适应抬升 2cm/步
  --pregrasp-offset 0.10 \
  --grasp-validation-mode --grasp-validation-first   # 先做验证性抓取
```

（完整安全参数见原 `run_real_nero_safe.sh`，迁移时逐条抄过来，一条都别删。）

### 3.8 复制相机与环境脚本

```bash
cp "$SRC/run_orbbec_gemini330_grasp.sh" scripts/launch_orbbec.sh
cp "$SRC/check_real_nero_env.sh" scripts/check_env.sh
cp "$SRC/check_grasp_env.sh" scripts/check_sim_env.sh   # 可选，仅仿真环境自检
```

### 3.9 不迁移的（明确排除）

| 文件/目录 | 原因 |
|---|---|
| `grasp_nbv_obstacle_yolo26seg_backup_6_6.py` | 旧备份 |
| `debug_mujoco_dofs.py` | 仿真调试 |
| `graspnet_All/` | 旧 RealSense demo（与实机链路无关，仅参考） |
| `run_grasp_clean.sh`（仿真启动） | 第一步不做仿真 |
| `run_nero_ros_clean.sh` | 与抓取无关的 ROS 启动 |
| `scripts/`（data_collection、dataset_tools、training） | 数据采集/训练工具 |
| `整体思路`、`README.md` | 文档，留在原工程 |

---

## 4. 主程序是怎么跑起来的（调用链）

```
main.py
 └─ real_nero_grasp.runtime.main()                 # runtime.py:3031
     ├─ cli.parse_args()                           # 解析 ~780 个 CLI 参数（cli.py）
     └─ run_single_grasp(args)                     # runtime.py:3024
         └─ workflow.run_single_grasp(args, hooks) # workflow.py:64，流程如下：
```

`workflow.run_single_grasp` 内部按顺序做（每一步对应你流程图里的一个环节）：

| # | 流程环节（对应你的图） | 调用的函数 |
|---|---|---|
| 1 | 连接机械臂（SDK can0/V111） | `RealNeroGraspArm(...)`（arm.py，with 上下文进入） |
| 2 | 设置 TCP offset | `arm.set_tcp_offset()`（workflow.py:95，arm 对象的**直接方法调用**，不经 session_control） |
| 3 | 运动到初始观察位姿并张爪 | workflow 调用**钩子** `prepare_initial_pose_for_capture(arm, args)` → runtime.py 装配层接线到 `session_control.prepare_initial_pose_for_capture()` → 内部调 `move_initial()`（人工确认→关节运动→到位等待）→ `arm.open_gripper()` → `wait_until_gripper_open()` |
| 3 | 全局相机采 RGB-D（多帧中值滤波） | `perception.capture_rgbd_xyz_frame()` |
| 4 | YOLO 实例分割 → 选目标掩码 | `cv_process_updated_yolo26seg.segment_image()` → `perception.select_target_mask()` |
| 5 | 深度反投影 → 目标/场景点云 | `perception.cloud_from_mask()` |
| 6 | [可选] 清障决策（amodal 遮挡估计） | `clear_policy.maybe_clear_obstacles_before_active()` |
| 7 | [可选] 多视角观察 + ICP 融合 | `active_view.run_optional_active_view()` → `pointcloud_fusion.register_active_fusion_views()` |
| 8 | GraspNet 推理出抓取候选 | `visualization.get_graspnet_helpers()` → 大脚本 `generate_grasps()` |
| 9 | 候选筛选（宽度/可达/桌面护栏/IK 预检） | `grasp_candidates.select_grasp_candidate()` |
| 10 | 抓取位姿 → base 系 TCP 位姿 | `grasp_geometry.grasp_to_tcp_poses()` |
| 11 | TRAC-IK 求关节解 + 碰撞约束规划 | `ik_solver._solve_ik_for_flange_pose()`、`grasp_planner.plan_up_and_over_grasp_sequence()` / `_plan_direct_grasp_sequence_beam()` |
| 12 | [可选] 抓取后 RRT-Connect 搬运 | `post_grasp_rrt.plan_post_grasp_free_route()` |
| 13 | move_j 分段执行（到位等待、启动重试） | `execution.execute_prevalidated_stages()` → `arm_execution.move_flange_pose_ik_joint()` |
| 14 | 夹爪慢闭 + 力/宽度接触确认 | `RealNeroGraspArm.close_gripper_slow()` |
| 15 | **抓住并抬升 → 停在此状态**（第二步的交接点） | 抬升阶段（`--lift-height 0.12` 自适应抬升） |
| 16 | [可选] 搬运到放置点、释放、回到观察位姿 | `session_control.return_initial_with_open_gripper()` |

> **钩子注入（hooks）模式**：`workflow.py` 是纯编排层，不 import 任何具体实现——所有
> 硬件/感知/规划动作都通过 `WorkflowHooks`（workflow.py:21 的 dataclass）注入，
> `runtime.py` 是装配层，把真实实现接进钩子（runtime.py:194 `_session_control_hooks()`）。
> 所以 workflow 里看到的 `prepare_initial_pose_for_capture(arm, args)` 不是 import 来的
> 函数，是钩子；真正的执行在 session_control.py 里。这样 38 个测试才能替换成假实现
> 离线跑单测。

**交接点（第二步在这里接）**：第 15 步结束后，机械臂「夹着物体停在抬升高度」。
第二步要做的三件事都从这个状态出发：① 停止 move_j 控制；② 相机源从全局切到腕部；
③ 启动视觉伺服的阻抗控制链，把物体递到人手。第一步**不需要**实现这些，
但验收时要确认「抬升后稳定保持」这个状态确实成立（见 §8 验收 6）。

---

## 5. 模块功能速查表（每个文件是干什么的）

### A. 入口与流程编排

**`main.py`**
| 名称 | 作用 |
|---|---|
| `main` 入口 | 调用 `runtime.main()`，无其他逻辑 |

**`real_nero_grasp/runtime.py`** — 入口门面
| 函数 | 作用 |
|---|---|
| `main(argv=None)` | 程序入口：解析参数 → 建 hooks → 跑 `run_single_grasp` |
| `run_single_grasp(args)` | 顶层流程：连接臂 → 抓取 → 结果统计 |
| `T_base_camera_from_flange(...)` | 实时法兰位姿 × 手眼标定 = 相机到基座的变换 |
| `set_tcp_offset()` / `move_initial()` | 设 TCP offset / 运动到初始位姿 |
| `check_runtime_collision_current()` | 运行期碰撞护栏入口 |

**`real_nero_grasp/workflow.py`** — 主流程编排
| 函数 | 作用 |
|---|---|
| `run_single_grasp(args, *, hooks)` | 编排 §4 表里的完整抓取流程（核心函数） |
| `WorkflowHooks` | 钩子容器：把碰撞/规划/IK/执行等模块注入流程（测试可替换） |

**`real_nero_grasp/cli.py`**
| 函数 | 作用 |
|---|---|
| `parse_args()` | 定义并解析 ~780 个 CLI 参数（相机话题、标定、权重、速度、护栏……） |

**`real_nero_grasp/session_control.py`** — 会话控制
| 函数 | 作用 |
|---|---|
| `confirm_motion(desc)` | 关键动作前的人工确认（安全门） |
| `set_tcp_offset()` / `move_initial()` | 初始化 TCP / 运动到初始观察位姿 |
| `prepare_initial_pose_for_capture(arm, args)` | 拍照前准备位姿 |
| `return_initial_with_open_gripper()` | 抓完后张爪回到初始位姿 |
| `recover_after_empty_grasp()` | 抓空后的恢复流程 |
| `assert_grasp_detected(close_status)` | 断言抓取成功判据 |

**`real_nero_grasp/types.py`** — 数据类与异常
| 类型 | 作用 |
|---|---|
| `GraspAttemptResult` / `GraspAttemptStatus` | 一次抓取尝试的结果/状态枚举 |
| `GripperCloseResult` / `ActiveViewResult` | 夹爪闭合/主动视角的结果 |
| `MotionSafetyScene` / `MotionCollisionContext` / `GraspMotionPlan` / `PlannedJointStage` | 运动安全场景/碰撞上下文/抓取运动计划 |
| `DynamicEnvironmentMap` / `InitialSupportRelease` | 动态环境图/初始支撑释放 |
| `MotionNotReached` / `UnsafeMotionTarget` / `SafetyStopTriggered` / `TorqueCollisionStopTriggered` / `GraspNotDetected` / `PlanNotExecutable` 等 | 各种失败信号（异常类型） |

### B. 臂与夹爪

**`real_nero_grasp/arm.py`**
| 函数/类 | 作用 |
|---|---|
| `RealNeroGraspArm(NeroArm)` | SDK 机械臂安全封装：连接（can0/V111）、e-stop 锁存、力矩护栏 |
| `move_p()` / `move_j()` | 笛卡尔/关节空间运动命令 |
| `stop()` / `terminate_motion()` | 停止运动（`terminate_motion` 直接发 CAN 控制帧） |
| `open_gripper()` / `close_gripper()` / `close_gripper_slow()` | 夹爪开/闭/慢闭（力限 0.50 N、宽度钳位 0.085 m、接触确认） |
| `get_joint_torques()` | 读关节力矩（护栏用） |

**`real_nero_grasp/arm_execution.py`** — move_j 执行
| 函数 | 作用 |
|---|---|
| `move_flange_pose(pose)` / `move_flange_pose_ik_joint(pose)` | 法兰位姿 → 关节解 → move_j 执行 |
| `wait_until_pose_reached(nero, target, args)` | 等待位姿到位（带超时） |
| `wait_until_joints_reached(nero, q_target, args)` | 等待关节到位 |
| `_send_joint_target_with_start_retry(q_target, ...)` | 发关节目标，未启动自动重试 |
| `_wait_for_joint_motion_start()` / `_wait_for_joint_settle()` | 等待运动启动/稳定 |
| `check_table_guard_current(arm, args)` | 桌面护栏检查（TCP 不撞桌） |

### C. 视觉感知

**`cv_process_updated_yolo26seg.py`**（顶层文件）— YOLO26 分割
| 函数 | 作用 |
|---|---|
| `get_yolo26seg_model()` / `get_yolo26_amodal_model()` | 加载可见/amodal 分割权重 |
| `segment_image(bgr)` | 整帧实例分割：掩码/类别/置信度 |
| `infer_visible_amodal_occlusion(...)` | 可见+amodal 联合推理 → 遮挡估计 |
| `match_visible_amodal_instances(...)` | 可见实例与 amodal 实例配对 |
| `detect_objects(bgr)` | 目标检测（bbox+类别） |
| `detect_context_instances_center_view(...)` | 中心视角的环境语义识别（清障决策用） |
| `release_vision_model_caches()` | 释放模型显存 |

**`real_nero_grasp/perception.py`** — 采图与点云
| 函数 | 作用 |
|---|---|
| `capture_rgbd_xyz_frame(...)` | ROS 话题采 RGB-D + 反投影 XYZ（多帧中值/MAD 校验） |
| `capture_target_cloud(...)` / `capture_target_cloud_with_retries(...)` | 采图+分割+掩码转点云（带重试） |
| `select_target_mask(...)` | 选目标掩码：YOLO 自动（`--target-class`）/ 人工点选（`click_point`） |
| `infer_yolo_instances(...)` | 跑 YOLO 得到实例列表 |
| `cloud_from_mask(xyz, bgr, mask, ...)` | 掩码内像素 → camera 系彩色点云 |
| `obstacle_cloud_from_non_target_mask(...)` | 非目标掩码 → 障碍物点云 |
| `camera_self_occlusion_mask(...)` | 相机自遮挡掩码（臂体挡镜头） |
| `camera_near_field_ghost_mask(...)` | 近场鬼影滤波（镜头上污渍/太近的伪点） |
| `mask_from_projected_target_center(...)` | 主动视角下重投影找目标掩码 |

**`real_nero_grasp/depth_processing.py`** — 深度处理
| 函数 | 作用 |
|---|---|
| `depth_to_xyz_image(depth_m, intr)` | 深度图 → XYZ 图 |
| `pointcloud2_xyz_image(cloud_msg)` | ROS PointCloud2 → XYZ 图 |
| `refine_xyz_inside_visible_mask(...)` | 掩码内深度补洞/平滑 |
| `_median_depth_for_mask(...)` | 掩码中值深度 |

**`real_nero_grasp/pointcloud_fusion.py`** — 多视角融合
| 函数 | 作用 |
|---|---|
| `fuse_target_clouds(...)` | 多视角目标点云融合 |
| `register_active_fusion_views(...)` | 主动视角点云的刚性 ICP 配准融合（核心） |
| `fuse_dynamic_environment_clouds(...)` | 动态环境图融合（避障用） |
| `merge_clouds(...)` / `transform_cloud(cloud, T)` | 点云合并/变换 |
| `save_cloud(...)` / `preview_fused_cloud(...)` | 保存/预览 |

### D. 抓取候选生成（你流程图里的 GraspNet 环节）

**`grasp_active_three_views_...yolo26seg.py`**（顶层大文件）— GraspNet 推理库
> 这个文件既是 MuJoCo 仿真主程序，又是实机的 GraspNet 推理库。第一步**整文件复制**
> 保证行为一致；以后可以抽成独立模块（见 §9 备注）。
| 函数 | 作用 |
|---|---|
| `get_net()` | 构建 GraspNet 网络（300 视角×12 角度×4 深度），加载 `logs/log_rs/checkpoint-rs.tar`（需 CUDA） |
| `make_end_points_from_o3d_cloud(cloud_o3d, num_point=3000)` | open3d 点云 → 采样成网络输入的 GPU 张量 |
| `generate_grasps(end_points, cloud, T_wc, collision_cloud, ...)` | 推理 → 解码 → GraspGroup → 碰撞过滤 → 孤立点过滤 → NMS → 按分排序 |
| `_prepare_graspnet_for_inference()` / `_offload_graspnet()` | 推理前准备/显存卸载 |

**`real_nero_grasp/grasp_geometry.py`** — 抓取 → TCP 位姿
| 函数 | 作用 |
|---|---|
| `graspnet_rotation_to_tcp_rotation(R, args)` | GraspNet 抓取旋转 → 夹爪 TCP 旋转 |
| `grasp_to_tcp_poses(grasp, ...)` | 单个抓取 → 预抓取/抓取/抬升三个 base 系 TCP 位姿 |
| `grasp_planning_pose_variants(pre, grasp, lift, args)` | 生成规划用位姿变体（朝向微调） |
| `topdown_rotation_from_hint(...)` | 由接近轴约束求自上而下的旋转 |

**`real_nero_grasp/grasp_candidates.py`** — 候选筛选
| 函数 | 作用 |
|---|---|
| `filter_grasp_candidates_by_width(rows, args)` | 宽度筛选（≤ 0.110 m） |
| `filter_grasp_candidates_by_reachability(rows)` | 可达性筛选 |
| `filter_grasp_candidates_by_table_guard(rows, args)` | 桌面护栏筛选（防撞桌） |
| `filter_grasp_candidates_by_workflow_preflight(rows)` | 全流程预检筛选（含 IK） |
| `annotate_grasp_candidate_reachability(...)` | 给候选标注 IK 可达性 |
| `select_grasp_candidate(...)` | 自动/人工选择最优候选（默认打分排序） |
| `choose_default_grasp_candidate(...)` | 默认候选选择（稳妥优先） |
| `draw_grasp_candidate_overlay(...)` | 把候选画到图上（人工选择界面） |

### E. IK 与运动规划

**`kinematic_trac_ik.py`**（顶层文件）
| 函数/类 | 作用 |
|---|---|
| `Kinematic(urdf_path, base_link, tip_link, ...)` | Trac-IK 求解器封装（Speed 模式，5 ms 超时） |
| `solve_fk(q)` | 正解 → `[x,y,z,qw,qx,qy,qz]` |
| `solve_ik(pose_wxyz, seed)` | 逆解（四元数 wxyz；解存为下次种子） |
| `get_joint_limits()` / `check_joint_limits(q)` | 限位查询/检查 |

**`real_nero_grasp/ik_solver.py`** — IK 求解
| 函数 | 作用 |
|---|---|
| `_get_ik_solver(args)` | 懒加载 Trac-IK 单例 |
| `_solve_ik_solutions_for_flange_pose(...)` | Halton 多种子批量求多解 |
| `_solve_ik_for_flange_pose(...)` | 限位裕量 + 排序取最优解 |
| `_joint_seed_candidates(current_joints, ...)` | 由当前关节生成种子 |

**`real_nero_grasp/grasp_planner.py`** — 抓取路径规划
| 函数 | 作用 |
|---|---|
| `plan_up_and_over_grasp_sequence(...)` | 「抬升-平移-下落」经典抓取序列规划 |
| `_plan_direct_grasp_sequence_beam(...)` | 碰撞约束 Beam Search 直接抓取序列（你流程图里的 Beam Search） |
| `plan_post_support_lift(...)` | 抓取后支撑抬升规划 |
| `_plan_grasp_pose_fallbacks(...)` | 位姿失败回退（换候选/换变体重试） |

**`real_nero_grasp/motion_planning.py`** — 通用运动规划
| 函数 | 作用 |
|---|---|
| `_plan_tcp_motion_stage(...)` | 单段 TCP 运动规划 |
| `_plan_tcp_stage_sequence_beam(...)` | 分段 Beam Search 序列规划 |
| `_plan_up_and_over_route(...)` / `_plan_guarded_route(...)` | 抬升平移路线 / 带护栏路线 |

**`real_nero_grasp/post_grasp_rrt.py`** — 抓取后自由路线
| 函数 | 作用 |
|---|---|
| `rrt_connect(q_start, q_goal, edge_valid, ...)` | RRT-Connect 关节空间搜索 |
| `plan_post_grasp_free_route(...)` | 抓取后 RRT 自由路线搬运规划 |

### F. 碰撞与安全

**`real_nero_grasp/collision.py`** — 碰撞模型
| 函数 | 作用 |
|---|---|
| `prepare_motion_safety_scene(...)` | 构建运动安全场景（臂/相机/夹爪分段胶囊模型 + 点云） |
| `validate_motion_joint_path(path, scene, args)` | 关节路径碰撞校验（核心） |
| `check_table_guard_current(...)` | 当前位姿桌面护栏 |
| `filter_motion_robot_self_points(...)` | 滤掉点云里机器人自身的点 |

**`real_nero_grasp/runtime_safety.py`** — 运行期护栏
| 函数 | 作用 |
|---|---|
| `check_runtime_collision_current(...)` | 运行期碰撞护栏（运动中实时检查） |
| `_check_runtime_joint_torque(...)` | 运行期力矩跳变护栏（Δ=8） |
| `RuntimeSafetyHooks` | 护栏钩子容器 |

**`real_nero_grasp/stop_policy.py`**
| 函数 | 作用 |
|---|---|
| `request_ordinary_trajectory_stop(...)` | 普通轨迹停止策略（分级停止） |

**`real_nero_grasp/execution.py`** — 执行与恢复
| 函数 | 作用 |
|---|---|
| `execute_prevalidated_stages(plan, ...)` | 把规划好的分段关节路径逐步 move_j 执行 |
| `execute_prevalidated_stage(...)` | 执行单段 |
| `recover_after_torque_stop(...)` | 力矩停止后的恢复（慢速回撤） |

### G. 主动观察与清障（你流程图里的「多视角观察与清障决策」）

**`real_nero_grasp/active_view.py`**
| 函数 | 作用 |
|---|---|
| `run_optional_active_view(...)` | 主动多视角主入口：候选生成→选视角→运动→补拍→融合 |
| `_build_active_view_candidates(...)` | 生成环绕/侧视观察候选 |
| `_target_center_overhead_flange_pose_candidates(...)` | 目标上方观察位姿候选 |
| `_look_at_camera_pose_minimize_flange_rotation(...)` | look-at 相机位姿（最小化法兰旋转） |
| `_rank_active_lookat_roll_solutions(...)` | 候选排序 |

**`real_nero_grasp/active_view_scoring.py`**
| 函数 | 作用 |
|---|---|
| `_build_active_information_proxy(...)` | 构建主动信息量代理模型（深度不确定思想） |
| `_score_active_candidate_information(...)` | 按信息增益给候选视角打分 |
| `_estimate_normals_toward_camera(...)` | 估计朝向相机的法向量 |

**`real_nero_grasp/active_motion.py`**
| 函数 | 作用 |
|---|---|
| `plan_active_direct_route(...)` | 主动视角直接路线规划 |
| `plan_active_up_and_over_route(...)` | 主动视角抬升-平移路线规划 |

**`real_nero_grasp/clear_policy.py`** — 清障策略
| 函数 | 作用 |
|---|---|
| `infer_target_amodal_occlusion(...)` | amodal 推理目标被遮挡程度 |
| `detect_clear_obstacle_candidates(...)` | 找需要先清走的遮挡物 |
| `maybe_clear_obstacles_before_active(...)` | 清障决策主入口：挡住就先抓走遮挡物 |

### H. 支持与工具

| 文件 | 作用 |
|---|---|
| `real_nero_grasp/paths.py` | 路径锚定（工程根、YOLO 默认权重、IK URDF、GraspNet import 目录） |
| `real_nero_grasp/transforms.py` | 位姿工具：`interpolate_pose`（位姿插值）、点变换、RPY 误差 |
| `real_nero_grasp/visualization.py` | 可视化 + `get_graspnet_helpers()`（懒加载大脚本的推理函数——实机链路靠它接上 GraspNet） |
| `tools/real_nero_point_test.py` | 手眼标定转换（`handeye_transform(calib, convention)`，默认 `flange_camera_xyzw`） |
| `tools/verify_handeye_point.py` | 点选一个像素 → 换到 base 系 → 对比真实位置，验证手眼标定 |
| `tools/nero_handeye_calib_pose_sweep.py` | 手眼标定姿态采集（绕 ArUco marker 扫 32 个姿态记录观测） |
| `tools/generate_view_rgbd_pointcloud.py` | 离线 RGB-D 文件 → 彩色 PLY 点云 |
| `tools/view_fused_pointcloud.py` | 查看融合点云 |
| `tools/teleop_nero_joints.py` | 手动关节控制调试 |

---

## 6. 常用 CLI 参数速查（最常用的一小部分）

| 参数 | 作用 |
|---|---|
| `--calib-json <json>` | **必填**。手眼标定文件（position/orientation xyzw） |
| `--target-class <名>` / `--select-target` / `--manual-click-only` | 按类别选目标 / 交互选择 / 只人工点选 |
| `--yolo-weights <pt>` / `--amodal-weights <pt>` | YOLO 可见/amodal 权重路径 |
| `--active-view` / `--enable-clear-obstacle` | 开主动多视角 / 开清障策略 |
| `--tcp-offset x y z` | TCP 偏移（**实机验证值 `[0.175, 0, -0.0235]`**，注意与 Visual Servo 工程统一） |
| `--initial-joints-deg j1 ... j7` | 初始观察位姿 |
| `--nero-interface socketcan --nero-channel can0 --nero-firmware V111` | SDK 连接参数 |
| `--nero-speed-percent 5` | 关节速度百分比（实机建议 5） |
| `--lift-height 0.12` / `--adaptive-lift-heights` | 抬升高度 / 自适应抬升 |
| `--place-xyz x y z` | 放置目标（递送环节，第二步会扩展成人手位姿） |

全部参数以 `python main.py --help` 输出为准（约 780 个，按功能分组）。

---

## 7. 环境与启动

第一步**不装任何新依赖**，完全沿用原抓取工程的环境：

```bash
# 1. 启动相机（Orbbec Gemini 330）
bash scripts/launch_orbbec.sh
#    等价于：ros2 launch orbbec_camera gemini_330_series.launch.py camera_name:=camera \
#            depth_registration:=true align_mode:=SW align_target_stream:=COLOR \
#            enable_frame_sync:=true ordered_pc:=true device_preset:="High Density"

# 2. 环境自检
bash scripts/check_env.sh
#    检查：numpy、scipy、cv2、can、pyAgxArm、rclpy、sensor_msgs、agx_arm_msgs

# 3. 跑抓取（先安全模式）
bash scripts/run_grasp_safe.sh --calib-json <标定.json> --target-class <物体类别>
```

环境依赖（`grasp_yolo26`）：torch+CUDA、ultralytics、open3d、trac-ik、spatialmath、
graspnetAPI、pointnet2（CUDA 扩展 `grasp_nms`）、mujoco、pyAgxArm、python-can、ROS2 Humble。

---

## 8. 第一步验收标准（按顺序全过才算完成）

1. **结构完整**：目录树与 §2 一致；`logs/log_rs/checkpoint-rs.tar` 和 `models/*.pt` 存在。
2. **入口可用**：`python main.py --help` 正常打印全部参数（不连臂、不开相机也能跑）。
3. **import 自检**（连不连臂都行）：
   ```bash
   python -c "from real_nero_grasp import runtime; print('runtime OK')"
   python -c "from visualization import get_graspnet_helpers; print('graspnet helpers OK')"
   #   ↑ 第二句会加载 9900 行大脚本，能过说明 graspnet-baseline/manipulator_grasp 依赖链完整
   python -c "from cv_process_updated_yolo26seg import get_yolo26seg_model; m=get_yolo26seg_model(); print('yolo OK')"
   python -c "from kinematic_trac_ik import Kinematic; k=Kinematic('manipulator_grasp/nero_urdf/agx_arm_urdf/nero/urdf/nero_description.urdf'); print('trac-ik OK')"
   ```
4. **离线感知 dry-run**：用一张离线 RGB-D（或 `tools/generate_view_rgbd_pointcloud.py` 生成的点云）
   跑 YOLO 分割 + GraspNet 推理，可视化候选抓取姿态（位置/方向合理）。
5. **测试回归**：`pytest tests/`（38 个测试，原工程同环境应全过）。
6. **实机单次抓取（安全模式）**：`scripts/run_grasp_safe.sh` 抓一个已知物体，
   确认：全程无碰撞护栏触发、夹爪接触确认通过、**抓取抬升后稳定保持 30 秒**（这是第二步的交接状态）。

---

## 9. 为第二步（视觉伺服递送）预留的接口

第二步会从 Visual Servo 工程引入伺服能力，第一步**不需要动代码**，但记住以下边界：

| 接口点 | 第一步状态 | 第二步要做的 |
|---|---|---|
| 流程出口 | 抓取抬升后保持（`--lift-height` 处） | 在此接「抓取→伺服交接」：停 move_j → 启动阻抗控制链 |
| 相机源 | 全局相机 Orbbec（`/camera/*` 话题） | 切换腕部相机（新的人手检测输入） |
| 控制方式 | SDK `move_j`（抓取阶段） | MIT 阻抗链（伺服阶段，来自 Visual Servo 工程的 `SdkForceCtrlBackend`/`NeroMITLoop`） |
| 递送目标 | 固定 `--place-xyz` | 人手实时位姿（PBVS 跟踪），替代固定放置点 |
| 人手检测 | 无 | **新模块**：直接人手位姿估计（替代 ArUco；Visual Servo 工程里 `hand_depth_target.py` 是预留占位） |
| 配置 | 各脚本参数分散 | 第二步统一到 `config/pipeline.yaml`（两相机话题、标定、切换逻辑） |

**CAN 单一控制源铁律**：抓取阶段的 move_j 和伺服阶段的 MIT 阻抗都直连 can0，
切换时必须先完全停止上一阶段控制（`arm.stop()` / backend `disconnect()`）再启动下一阶段，
任何时刻只有一个控制源。

**GraspNet 大脚本的后续清理**（可选，第二步时做）：把 `get_net`/`make_end_points_from_o3d_cloud`/
`generate_grasps`/`_prepare_graspnet_for_inference` 从 9900 行仿真脚本抽成独立模块
`real_nero_grasp/graspnet_inference.py`，删掉仿真部分。第一步先整文件复制保行为一致。

**根目录 4 个运行时模块的彻底整理**（第二步和抽取一起做，让根目录只剩 `main.py`）：

| 文件 | 第二步的归宿 |
|---|---|
| `grasp_active_three_views_...py` | 抽取后删除，只剩 `real_nero_grasp/graspnet_inference.py` |
| `cv_process_updated_yolo26seg.py` | 挪进 `real_nero_grasp/vision/yolo26seg.py`，改 `clear_policy.py` 和大脚本的 import |
| `kinematic_trac_ik.py` | 挪进 `real_nero_grasp/planning/trac_ik.py`，改 `ik_solver.py` 的 import |
| `real_nero_single_grasp_runner.py` | 挪走后需同步改 23 个测试的 import（机械替换即可）；或保留为兼容别名 |

现在不动它们，是因为每个挪动都要动代码和 import——等第二步本来就要重构时一起做，只改一遍。

---

## 10. 安全规则

- 实机运行**必须**用 `scripts/run_grasp_safe.sh`（速度 5%、力矩护栏 Δ=8、碰撞护栏、桌面护栏、慢闭夹爪）。
- 每次启动前确认：急停可用、工作空间无人、`can0` 上无其他控制源（旧的 ROS arm controller 必须停）。
- 第一次实机先跑「验证性抓取」（`--grasp-validation-first`）：先空走一遍再真抓。
- 原 `real-nero-grasp-main` 工程保持只读，新工程验收通过前不要删除。
