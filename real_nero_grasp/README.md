# Real Nero 抓取流程架构说明

本目录用于把原先集中在 `real_nero_single_grasp_runner.py` 中的实机抓取流程逐步拆分为职责明确、可独立测试的模块。

本次重构的首要约束是**保持现有行为不变**：不改变目标选择、清障、主动视角、GraspNet、IK、碰撞保护、夹爪闭合、失败恢复或实机停止策略。算法改进（例如跨阶段 beam search）必须在结构重构完成并通过回归测试之后单独实施。

## 1. 兼容契约

重构期间必须保持以下外部行为：

- 原有启动命令和脚本名称不变，包括：

  ```bash
  ./run_real_nero_clean.sh ...
  ./run_real_nero_safe.sh ...
  ./run_real_nero_food5_clear.sh ...
  ```

- `run_real_nero_clean.sh` 仍可执行：

  ```bash
  python real_nero_single_grasp_runner.py ...
  ```

- 原有 CLI 参数、默认值、参数覆盖顺序、交互提示和日志前缀保持不变。
- 按下回车表示确认执行，输入 `no` 表示取消当前待执行动作。
- 不带 `--execute` 时仍然只规划和展示，不发送抓取运动。
- 默认模型、标定文件和 URDF 路径必须以**项目根目录**为基准，而不是以 `real_nero_grasp/` 包目录为基准。包内代码计算项目根目录时应使用包目录的上一级。
- 日志与点云输出目录的命名和相对路径语义保持不变。

## 连续 `move_j` 路点的执行确认

分段后的预抓取、主动观察返回和自由抬升路线会连续发送多个关节目标。为避免上一段尚在收敛时覆盖下一条命令，执行层现在采用三道确认：

- 新路点发送前，编码器必须连续稳定 3 个采样；
- “指令已启动”必须表现为关节沿新目标方向取得至少 0.2° 的投影进展，其他方向的残余运动不再算启动；
- 控制器确认正常、但关节仍停留在发送起点时，提前结束长等待，并在重新执行实时路径预检后只重发一次同一冻结目标。

这些检查不改变 TRAC-IK 解、冻结碰撞地图或规划路线。失败时仍使用普通轨迹终止，不会因为路点指令未启动而直接触发电子急停。相关阈值可通过 `--move-precommand-settle-*`、`--move-stall-*` 和 `--move-late-retry-max-displacement-deg` 调整。
- 旧模块中的已有导入仍应可用，特别是手眼标定程序使用的：
  `DEFAULT_IK_URDF`、`MotionNotReached`、`PlanNotExecutable`、
  `_as_pose_list`、`_make_T_from_pose`、`_get_ik_solver` 和
  `_solve_ik_for_flange_pose`。
- 结果状态和进程退出码保持不变：

  | 退出码 | 含义 |
  |---:|---|
  | `0` | 抓取成功，或仅规划成功（`PLANNED_ONLY`） |
  | `2` | 一般流程失败，或普通轨迹终止后安全退出 |
  | `3` | 已请求电子急停后退出 |

异常类、枚举和 dataclass 必须只定义一次并由旧入口重新导出，不能在多个模块中复制定义，否则会破坏 `isinstance`、异常捕获和结果状态比较。

## 2. 当前目录与职责

本轮已经把可独立验证、依赖方向清晰的部分迁入新包。`runtime.py` 继续作为兼容门面与高层编排入口；算法实现、碰撞检查、路径规划和冻结计划执行分别由独立模块负责。对仍会被旧测试或外部脚本动态替换的依赖，门面在每次调用时显式注入，避免模块化后改变 monkeypatch、日志或安全分支语义。

```text
real_nero_grasp/
├── __init__.py             无硬件副作用的包入口
├── __main__.py             支持 python -m real_nero_grasp
├── README.md               本说明文档
├── paths.py                项目根目录、默认模型和 URDF 路径
├── types.py                异常、状态枚举、不可变计划与动态环境地图
├── transforms.py           位姿、齐次变换和基础距离计算
├── depth_processing.py     深度图、孔洞修复和二值 mask 几何
├── visualization.py        Open3D/GraspNet 的延迟加载和显示
├── arm.py                  Nero 实机类与夹爪反馈/停止控制
├── session_control.py      初始位、交互确认、夹爪确认与空抓恢复
├── arm_execution.py        法兰运动、关节指令重试和到位等待
├── grasp_geometry.py       GraspNet 坐标映射和 TCP 姿态生成
├── grasp_candidates.py     候选宽度、物理开口、可达性和人工选择
├── ik_solver.py            TRAC-IK、seed、Top-K 逆解和限位
├── collision.py            URDF/夹爪/相机几何与点云路径碰撞检查
├── motion_planning.py      碰撞约束下的阶段 IK 与受保护路线规划
├── active_motion.py        主动观察往返路线的规划原语
├── active_view.py          球面候选、视角排序和主动观察状态机
├── grasp_planner.py        完整抓取任务预检与候选工作流筛选
├── grasp_workflow.py       抓取确认、冻结计划执行与失败恢复编排
├── execution.py            只执行已冻结、已预检的关节阶段
├── runtime_safety.py       实测关节碰撞监控、普通停止与急停升级
├── active_view_scoring.py  等半径球面视角的信息收益评分
├── pointcloud_fusion.py    目标融合及桌面级动态环境占据地图
├── perception.py           RGB-D、YOLO、目标重识别和场景点云构建
├── clear_policy.py         Amodal 遮挡判断与 fail-closed 清障闭环
├── workflow.py             顶层会话顺序与 motion_sent 结果聚合
├── cli.py                  原样保留的 argparse 参数定义
└── runtime.py              向后兼容导出与顶层会话编排门面
```

模块依赖应保持从底层到上层的单向关系：

```text
types / paths / transforms / cli
       ↓
depth_processing / visualization / arm / session_control
grasp_geometry / ik_solver / active_view_scoring
       ↓
perception / grasp_candidates / pointcloud_fusion / collision
       ↓
motion_planning / active_motion / grasp_planner / arm_execution
execution / runtime_safety
       ↓
active_view / grasp_workflow / clear_policy
       ↓
workflow
       ↓
runtime（兼容导出与顶层 workflow）
       ↓
real_nero_single_grasp_runner.py / python -m real_nero_grasp
```

规划器不能发送实机运动；执行器只消费已经冻结和验证的计划，不能在执行途中重新选择抓取候选或悄悄更换 IK 分支。

## 3. 运行入口

用户入口继续是根目录中的 Shell 脚本。调用关系保持为：

```text
run_real_nero_food5_clear.sh
    → run_real_nero_safe.sh
        → run_real_nero_clean.sh
            → real_nero_single_grasp_runner.py
                → parse_args()
                → run_single_grasp(args)
```

根目录的 `real_nero_single_grasp_runner.py` 现在只有兼容转发代码；导入时它会映射到同一个 canonical `runtime` 模块，因此旧代码通过 `patch.object(real_nero_single_grasp_runner, ...)` 动态替换依赖的行为仍然有效。直接执行时则调用 `runtime.main()`，保留原退出码。

也可以使用新的等价包入口：

```bash
python -m real_nero_grasp --calib-json ...
```

实机日常运行仍建议继续使用原 Shell 脚本，避免遗漏环境初始化和安全参数。

## 4. 完整状态机

当前整套流程可概括为：

```text
解析参数与配置模型
    ↓
连接机械臂、设置 TCP、保存恢复关节位
    ↓
移动到初始观察位并打开夹爪
    ↓
采集 RGB-D、生成场景点云、选择并锁定原目标
    ↓
清障判断与清障闭环（按需）
    ↓
主动多视角观察与融合（按需或被 Amodal 强制）
    ↓
GraspNet 候选生成和逐级筛选
    ↓
完整抓取任务运动预检
    ↓
用户选择/确认
    ↓
仅在 --execute 下执行冻结计划
    ↓
成功、空抓恢复、规划失败、执行失败或安全停止
```

任意必要阶段失败后都必须返回明确状态，不能把失败静默解释为“不再需要清障”或继续抓取被禁止的原目标。

## 5. 清障流程

清障决策综合使用可见实例、Amodal 隐藏区域和深度关系：

1. 锁定第一次选择的原目标及其语义、mask 和三维位置参考。
2. 判断非目标物体是否处于目标前方、是否与目标/隐藏区域重叠，以及是否阻挡观察或接近走廊。
3. 找到证据充分的遮挡物时，先把该物体作为清障目标。
4. 先对同一个清障目标进行单视角抓取规划。
5. 若单视角仅发生规划失败且尚未发送运动，则对同一个清障目标启动主动多视角，再重新解算。
6. 清障成功后回到可验证观察位，重新拍摄并严格重识别最初的原目标，然后重新评估是否仍需清障。
7. 清障空抓只有在确认安全返回初始观察位后，才能重新拍摄并进行有限次数的语义重试。
8. 必要清障未完成、目标身份不明确或主动观察未安全恢复时，禁止继续抓原目标。

清障目标要求完整任务可执行：仅能夹住或抬起、但无法安全搬运到清障放置区，不能记为清障成功。

## 6. 主动视角流程

主动视角可能由目标附近障碍触发，也可能由 Amodal 遮挡证据强制触发：

1. 优先计算并验证目标正上方中心观察位；严格垂直光轴无解时先保持相机中心位置完全不变，依次尝试 `2°/4°/6°` 的小光轴倾角并确认目标仍在允许的成像中心范围内。只有同位置姿态仍无解或完整路径不安全，才在相同观察半径上尝试 `15°/20°/30°/35°` 的斜上方球面观察位。最终选中的位姿成为后续侧视往返的已验证观察锚点。
2. 在固定观察半径上生成球面角度候选，相机始终看向同一目标中心；候选与当前已验证观察锚点的观察方向默认必须相差至少 `30°`，避免两张图高度重合而没有新增表面或抓取通道信息。多个侧视角之间仍额外保持 `--active-min-side-disparity-deg` 的分离约束。
3. 使用新增表面、夹爪接触信息、接近通道信息、遮挡分离、图像质量、冗余和运动成本对候选排序。
4. 分数只负责排序；视野、深度、IK、关节余量、全臂碰撞和往返路径仍是硬门槛。
5. 每个侧视角出发前必须验证“中心到侧面”和“侧面回中心”。
6. 侧视采集后，新发现的环境点会使旧返回计划失效，必须用更新后的安全点云重新验证返回路线。
7. 目标融合可以拒绝质量差的侧视目标点，但该视角看到的环境障碍点仍应保留在保守安全地图中。
8. 抓取规划前必须确认机械臂已经恢复到预期中心观察关节状态。

强制多视角时，如果有效融合视角数没有达到要求，则不能跳过完整性门槛直接进入目标抓取。

IK 种子默认同时覆盖全部七个关节：当前/初始姿态、两者的插值姿态与 24 个关节限位内的确定性 Halton 低差异姿态交错排列，旧的 J1/J7 扰动仅作为补充。侧视角快速预检默认消费前 8 个交错种子，因此第一轮就包含多个七关节肩肘腕构型；完整路线仍保留并回溯 Top-6 不同 IK 分支。

主动侧视角现在采用渐进预算。第一轮只检查评分最高的 8 个球面位置，每个位置最多 2 个 roll、每个姿态 2 个快速 IK 种子；只要找到可执行观察位，就不再展开后续完整搜索。第一轮没有任何可达结果时，才对全部位置使用完整 roll/种子预算，并用完整预算复查首批 8 个高分位置，避免快速种子造成假阴性。每请求一个有效侧视角，最多实际移动并拍摄 3 次（`--active-side-max-capture-attempts`）；深度质量或配准质量不合格的已拍角度也会立即加入 `tried_angles`，避免再次拍摄相近、收益重复的方向。

## 7. GraspNet、IK 与抓取规划

GraspNet 候选只表示局部夹爪几何上可能形成抓取，不等于整台机械臂可以安全完成动作。候选还要依次接受：

```text
宽度和物理开口检查
    → 桌面高度检查
    → pregrasp / grasp / lift 端点 IK
    → 主夹爪方向完整路径预检
    → 主方向失败后才尝试 180°备用方向
    → 搬运、放置、释放、返回和空抓恢复预检
```

当前 Top-K IK 的含义是：同一个末端位姿可以保留多个不同关节构型供规划检查；最终实机只执行其中一条已经选定并缓存的关节路径，不会执行多次或在途中临时换解。

TRAC-IK 只负责生成满足末端位姿和硬关节限位的关节解。以下情况不应统称为“IK无解”：

- `IK failed`：没有得到有效端点逆解；
- `largest joint sweep`：存在端点解，但关节变化超过策略限制；
- `gets within ... environment`：存在端点解，但连接路径发生碰撞；
- 后续 `lift/place/return` 失败：当前局部姿态可达，但完整任务不可执行。

## 8. 碰撞与实机安全边界

重构不得放宽或绕过现有安全边界：

- IK 端点可达不能替代完整关节插值路径检查。
- 规划阶段继续检查机械臂连杆、夹爪、相机/安装座、目标载荷、环境点云、桌面高度和硬关节限位。
- 规划失败的候选不能发送夹爪打开或机械臂运动指令。
- 实机执行只能使用预先验证并冻结的 `GraspMotionPlan`/`PlannedJointStage`。
- 运行时继续根据实机关节反馈检查路径约束和点云净空。
- 观察位到 pregrasp/grasp 的冻结 `before_close` 路线若含大关节扫掠，执行时沿同一条已验证关节直线自动拆成最大 20° 的微过渡命令，降低 Nero 各轴异步到位造成的实测轨迹偏离；每条子边仍会重新执行实时点云和桌面预检。
- 规划阶段张开手指仍要求完整配置净空；运行时仅对 `free_open` 的两根张开手指提供 1 mm 跟踪/点云抖动容差。例如规划阈值 5 mm 时运行时手指阈值为 4 mm，掌部、相机、连杆、载荷和桌面阈值不变。
- 支撑脱离的 XY 规划漂移上限仍为 5 mm，运行时额外接受 2 mm 控制器跟踪偏差；超过 7 mm 时优先普通终止轨迹，普通停止无法确认时才升级电子急停。向下侵入、桌面和力矩保护不使用这项容差。
- 运行时环境/目标点云净空越界（包括有限的负净空模型穿入）先请求普通轨迹终止；只有普通停止无法确认，或非点云关键约束触发时才升级为电子急停。
- 侧视角返回中心默认按最大关节变化约 `10°` 拆成短关节段，使 Nero 的实测多轴进度更贴近预检的线性关节边；可用 `--active-return-execution-waypoint-deg` 调整，设为 `0` 可恢复单条 `move_j`。
- Food5 实机脚本使用成功率优先但仍有边界的主动视角门槛：配准后中位误差不超过 `8 mm`、p95 不超过 `30 mm`、1 cm 重合率不低于 `55%`；最终重合覆盖还要求 overlap median 不超过 `6 mm`。当这些重合指标通过时，目标点云中位中心偏移只做诊断，不再淘汰侧视角。这样香蕉等弯曲物体的新可见表面不会仅因中心移动被排除，保守 ICP 仍限制为最多 `10 mm/3°`。
- Food5 的侧视角返回专用规划净空默认 `20 mm`，运行时点云净空保护仍使用安全脚本的 `5 mm`。可分别通过 `ACTIVE_RETURN_MIN_CLEARANCE` 与 `MOTION_PATH_CLEARANCE` 覆盖。
- 运行时同时读取 Nero 七个关节的实测力矩。每个冻结阶段先记录力矩基线；任一关节相对基线连续 `--motion-runtime-torque-confirm-samples` 次超过 `--motion-runtime-torque-delta`（默认 8 N·m）时，按疑似外力接触处理。
- 疑似外力接触先普通终止，再把“当前实测关节位 → 本阶段起点”的反向边在同一冻结地图上重新校验；仅校验通过才以 `--motion-runtime-torque-retract-speed-percent`（默认 2%）低速退回。退回失败或普通停止无法确认时才升级电子急停，且当前任务不会继续。
- `Ctrl+C` 不再只是终止 Python：适配器会先发送普通轨迹终止并确认关节静止，然后干净退出；只有普通停止无法确认时才为保护设备升级电子急停。
- 目标接触、支撑释放等阶段性放宽必须由 `MotionCollisionContext` 明确限定，不能变成全局关闭碰撞检测。
- 高位绕行和中间路径是否启用仍由现有参数决定。food5 常用脚本当前关闭旧的多段高位 TCP 绕行，不应在重构中自动重新开启。

`types.py` 中的核心对象含义如下：

- `DynamicEnvironmentMap`：保存一代冻结的桌面级环境占据、来源视角、逐体素支持数和置信度；所有几何都在 `base_link`。
- `MotionSafetyScene`：引用对应代次的动态环境地图，并保存本次规划使用的完整场景、环境和目标点。
- `MotionCollisionContext`：描述当前阶段的接触、载荷和最低高度策略。
- `PlannedJointStage`：一个已经确定的 TCP、法兰和关节目标阶段。
- `GraspMotionPlan`：执行前冻结的抓取、抬升、放置、释放、返回和空抓恢复计划。
- `GraspAttemptResult`：顶层流程结果；`motion_sent` 必须准确反映整套流程中是否发送过运动。

### 8.1 桌面级动态环境地图

开启默认参数 `--motion-environment-map` 后，规划不再为每个候选分别从不同融合点云重新构造障碍物。环境地图遵循以下流程：

```text
每个视角的非目标场景点
    → 使用该视角实机关节做机械臂自体过滤
    → 直接变换到 base_link（不套用目标 ICP/配准修正）
    → 裁剪到桌面级 ROI
    → 各视角独立体素化并统计不同视角支持数
    → 冻结为同一代 DynamicEnvironmentMap
    → 所有候选的 pregrasp/grasp/lift/transport/return 共用该代地图
```

地图不会扩展成整个房间。默认只保留目标中心 XY 方向各 `0.45 m`，以及 `table_z - 0.02 m` 到 `table_z + 0.60 m`；放置点和清障放置点会自动以 `0.05 m` 余量纳入。也可用 `--motion-environment-map-{x,y}-{min,max}` 明确限定桌面边界，但显式边界若裁掉目标中心或搬运/清障放置端点，地图构建会直接拒绝，而不是在缺失障碍的区域继续规划。

同一视角内的重复深度点只算一次支持。至少两个不同视角支持的体素属于硬障碍；单视角连续篮沿等结构仍保留为硬障碍；孤立低支持点进入不确定层。碰撞规划同时消费硬障碍和不确定层，因此置信度分层不会默认删掉潜在障碍。输出碰撞点使用真实测量点作为体素代表，避免合成体素中心把篮沿表面平移半个体素。

目标点云与环境地图分开保存：目标可继续做主动视角配准和抓取融合，篮子、桌垫及其他环境点只做基座坐标变换。即使某个侧视角因目标质量或配准门槛未进入 GraspNet 目标融合，该视角观测到的环境障碍仍会进入安全地图。

### 8.2 High Density 深度与目标重建

抓取相机默认采用 `High Density`，但不使用 SDK 的时域/空间补点来制造表面；在机械臂静止后，由抓取程序采集 7 帧深度，要求一个像素至少 5 帧有效，并以中值融合。若该像素相对中值的时间 MAD 超过 `4 mm`，会从本帧深度中剔除。这能消除随帧跳变的悬空飞点，同时不会跨篮沿或目标边界填洞。

每次捕获目录会保存：

- `depth_temporal_mad_mm.png`：逐像素时序波动；
- `depth_temporal_unstable.png`：被时序稳定性规则剔除的飞点；
- `depth_valid_count.png`：每个像素在本次累计中有效的帧数。

主动多视角的目标融合使用更细的 `2.5 mm` 体素。每个视角在每个体素中最多贡献一个代表点，因此单一 High Density 视角不会凭借更多采样压过其他视角；同一体素的坐标与 RGB 按视角权重融合。只删除“单视角且空间孤立”的目标点，保留连贯但只能从一个侧面看见的物体表面。融合目录同时保存 `active_fused_target_raw_camera0.ply`、`active_fused_target_camera0.ply`、`active_fused_scene_raw_camera0.ply` 和 `active_fused_scene_camera0.ply`，前两个文件可直接对比物体重建的改善。

目标视角先用受限平移配准；若重合质量仍未达到融合门槛，再执行保守的鲁棒 ICP。默认采用 point-to-point、`15 mm → 8 mm` 两阶段对应距离和 `5 mm` Huber 权重，额外修正严格限制在 `10 mm / 3°` 内，并且只有中位误差、P95 或 1 cm 内点率确实改善且没有明显退化时才接受。该修正只用于目标点云重建，不会套用到篮子、桌面或其他环境障碍点云。可用下面的离线命令复测一个已有会话，不发送机械臂运动：

```bash
python scripts/test_conservative_icp_fusion.py \
  logs/<run_id>/<capture> \
  --estimation point_to_point
```

这套过滤不等价于降低安全要求：动态环境地图的 `hard` 和 `uncertain` 两层仍会被碰撞规划使用。时序不稳定点在进入环境地图之前已被剔除；稳定但只在一个视角出现的点会保留在不确定层，便于从日志判断是否需要补拍确认。

常用参数及默认值：

```text
--motion-environment-map-voxel 0.005
--motion-environment-map-radius-xy 0.45
--motion-environment-map-z-below-table 0.02
--motion-environment-map-z-above-table 0.60
--motion-environment-map-min-view-support 2
--motion-environment-map-single-view-local-radius 0.020
--motion-environment-map-single-view-neighbor-radius 0.009
--motion-environment-map-single-view-min-neighbors 2
```

`whole_workflow_motion_plan.json` 会记录地图 ID、会话内单调递增的代次、坐标系、体素尺寸、来源视角以及硬/不确定点数量，并额外保存两层 PLY，便于确认到底是哪一代环境导致候选被拒绝。参数对象的浅拷贝共享同一个代次计数器；同一轮候选筛选则直接复用同一个不可变地图对象。`--no-motion-environment-map` 仅用于兼容旧流程或离线对照，不建议作为绕过实机碰撞检查的手段。

## 9. 如何扩展规划器

后续新增功能应在相应模块完成，避免重新把逻辑堆回入口文件。

### 9.1 跨阶段 beam search

直达抓取路线已在 `grasp_planner.py` 与 `motion_planning.py` 中实现跨阶段搜索：

```text
current → pregrasp → grasp → lift → transport/place
```

每个阶段不再立即固定第一个 IK 解，而是展开当前保留分支的 Top-K IK 解，对每条新边检查关节限位、桌面高度和整臂/夹爪/相机对环境点云的连续轨迹净空，然后按累计关节运动代价去重并截断 beam。各阶段的所有分支复用同一代已冻结的 `DynamicEnvironmentMap`/`MotionSafetyScene`，不会因为分支不同而改换障碍地图。

beam 宽度由 `--grasp-stage-beam-width` 设置，`--grasp-beam-width` 是其等价别名，默认值为 `6`。裸 CLI 中 `--grasp-stage-beam-search` 默认关闭以保持兼容；`run_real_nero_safe.sh` 默认开启，可通过环境变量 `GRASP_STAGE_BEAM_WIDTH` 调整宽度。该搜索在旧高位路线关闭（`--no-motion-up-and-over`）时使用；开启高位路线时仍保留原有规划逻辑。

### 可回退的抓取后自由路径

`run_real_nero_food5_free_route.sh` 是独立的实验入口，原有
`run_real_nero_food5_clear.sh`、`run_real_nero_safe.sh` 和裸 CLI 默认行为均未改变。
新入口仍严格规划抓取后的世界 Z 方向 20 mm support release；随后不再要求
固定 XY、固定姿态或精确直线 full lift，而是在关节空间用轻量 RRT-Connect
寻找满足以下条件的路径：TCP 至少达到抓取点上方 150 mm、硬关节限位有效、
机械臂/相机/闭合夹爪/attached target 通过现有动态环境点云碰撞检查，最后抵达
配置的 place XYZ。搜索阶段使用 4° 粗采样，入选路径仍以原来的精细碰撞分辨率
重新验证；默认最多尝试 3 个高区目标、每个 3 秒/400 节点，并跳过旧的精确
lift/transport 全局回退扩展。

回退只需重新使用原入口：

```bash
./run_real_nero_food5_clear.sh ...
```

离线复测保存场景不会初始化 Nero 驱动，也不会发送运动：

```bash
PYTHONPATH=. ~/anaconda3/envs/grasp_yolo26/bin/python \
  tools/replay_post_grasp_rrt.py LOG_DIR --candidate INDEX
```

候选完整路径预检现在按“最早失败阶段”调度有限回退，避免把不同阶段的故障组合成大量无效重算：

- `pregrasp` 无解或路径不安全：只依次检查较短/较长的预抓距离。默认最多 4 个距离、间隔 20 mm，可由 `--adaptive-pregrasp-*` 参数调整。
- `grasp` 无解或碰撞：当前姿态的 Top-K IK 分支已经在 beam 内检查完毕，因此停止改变预抓/抬升位姿，交给下一个抓取候选；主夹爪方向的候选失败后，仍按既有策略把 180°方向作为备用。
- `lift` 无解：固定已经选中的 `pregrasp` 与 `grasp`，先保持姿态不变并把请求的抬升高度按默认 `20 mm` 逐级降低，最低可到 `30 mm`；例如请求 `200 mm` 时依次检查 `200/180/160/.../40/30 mm`。每个端点先做一次全局 IK 可达性与软关节余量检查，明显无解的高度不会再对每个 beam 父分支重复求解。只有最低允许高度仍失败时，才绕 TCP 局部接近轴尝试 `±3°、±6°、±10°、±15°`。抓取闭合姿态不改变，URDF 硬关节限位和完整碰撞检查始终生效。
- `transport/place` 不安全：固定抓取和抬升姿态，先在放置点附近进行有限可行域搜索。默认依次尝试向上 `20 mm`、XY 四个方向各 `20 mm`，以及绕非接近轴 `±8°` 的小姿态变化，最多 9 个端点；仍失败才尝试一次 `lift → neutral joints → place`。这些回退复用已缓存的 pregrasp/grasp/lift 结果，不会进入全局路径扩展，且每条边继续使用 carrying 碰撞策略与同一代冻结环境地图。

释放物体后的撤离不再要求唯一的严格笛卡尔位姿。默认先尝试世界 Z 方向上升 `20 mm`；该端点无 IK 解或不满足严格自由空间时，才在 XY `10 mm` 和非接近轴倾角 `5°` 的有界区域内尝试，最多 9 个端点。选中的撤离端点必须通过 released-retreat 接触策略及普通 released-free 严格端点检查，桌面、整臂、相机和环境碰撞保护没有关闭。

lift 阶段默认把 J7 运动代价权重设为 `0`（`--ik-lift-joint7-weight`），允许法兰为完成抬升更多地旋转 J7；安全脚本把最低可接受抬升设为 `30 mm`、XY 可行域放宽到 `30 mm`，末端姿态依次尝试 `±3°/±6°/±10°/±15°/±20°`。URDF 硬关节限位、桌面保护、碰撞检查和路径检查仍然生效。若指定了 `--fixed-rpy` 或 `--use-current-rpy`，姿态被视为操作者硬约束，抬升姿态松弛会自动禁用，但降低抬升高度仍可独立工作。也可用 `--no-adaptive-pregrasp-offsets`、`--no-adaptive-lift-heights`、`--no-lift-orientation-relax`、`--no-transport-place-relax` 或 `--no-motion-neutral-transport-fallback` 分别关闭对应回退。

动态环境地图只拥有篮子、桌面等障碍占据。候选规划传入最终配准/过滤后的目标点云时，载荷和夹爪开口检查始终使用该最终目标，不再使用地图构建前保存的历史目标快照。lift 起点若存在不超过 `--motion-support-target-bottom-tolerance` 的桌面/垫面基线接触会被接受，但后续采样只允许接触减小；比起点继续向下超过 `--motion-support-downward-tolerance` 仍会被规划和运行时保护拒绝。

beam search 只生成和选择计划，不发送运动。完整任务检查通过后，最终仍只会选择一条分支，冻结为唯一的 `GraspMotionPlan` 供 `execution.py` 执行；实机执行中不会临时更换 IK 解或执行多条分支。

### 9.2 单个中性关节过渡点

已实现的中性点是候选级全局回退，不是某个候选直达失败后立即绕行。规划器先对全部合格抓取候选及其主夹爪方向/180°备用方向执行第一轮 direct beam 预检。只有这一轮没有任何完整路线通过，且开启 `--motion-neutral-pregrasp-fallback` 时，才对同一批候选统一启动第二轮：

```text
current → neutral joints → pregrasp
```

中性点由七个关节角定义，不是强制保持抓取 RPY 的高位 TCP 点。默认使用 `--initial-joints-deg`，也可用 `--motion-neutral-joints-deg J1 J2 J3 J4 J5 J6 J7` 显式覆盖。路线最多只增加这一个关节中间点；`current → neutral` 和 `neutral → pregrasp` 两条边都复用第一轮的同一代冻结环境地图，并重新做完整关节路径碰撞、高度和限位检查。

裸 CLI 中该回退默认关闭；`run_real_nero_safe.sh` 默认开启。它与跨阶段 beam 一样，只在旧高位路线关闭时作为直达路线的有限回退，不会恢复多段高位 TCP 绕行。若第一轮已有任何 direct 候选通过，则不会计算第二轮 neutral 路线。

### 9.3 计算预算

当前已实现以下直接减少重复计算的预算：

- 主动观察先用 `8位置 × 2 roll × 2 seed` 的快速层，只有零可达结果才进入完整层；单个有效侧视角的实拍尝试上限默认为 3。
- 抓取 beam 对已经通过连续碰撞检查的阶段边进行会话内缓存。仅 lift 高度或姿态改变时，复用相同的 `current → pregrasp → grasp` 结果；相同 lift 端点先统一做一次全局可达性检查，找到第一个满足关节余量的 IK 后立即停止，只有尚未发现合格解时才遍历完整种子，不再由每个 Top-K 父分支重复运行完整种子搜索。
- 裸 CLI 仍可选择在快速层失败后扩展到完整搜索；当前 `run_real_nero_safe.sh` 显式使用 `--no-grasp-progressive-full-fallback`，只运行 `beam=2、IK Top-K=2` 的有界局部搜索，并把全关节 Halton 种子数设为 0，不再在末尾进入耗时且收益低的全局扩展。
- 每个候选在任何 IK 之前先以抓取 TCP 下的目标点云稳健底面执行一次载荷/桌面预检。若 lift 在 `sample 0` 已因同一附着目标与桌面相交而失败，候选会在 pregrasp 之前直接淘汰；后续 IK Top-K、beam 父分支、完整 Top-K 扩展以及 lift 高度、J7 和姿态变化也保留分层短路作为防线。附着目标桌面检查使用目标底部的稳健分位数而不是单个最低点，并仅在 `pickup_lift/support_release` 初始支撑阶段提供 5 mm 容差，避免少量深度离群点反复触发相同失败；搬运阶段和真实整体过低的载荷仍保留严格桌面保护。

候选数量、beam width 和更高层的可执行候选 quota 仍应属于规划器配置。达到 quota 后未检查的候选要标记为“未检查”，不能错误标记为“不可达”。

新增算法时必须把“能安全抓起”和“能完成清障/放置任务”区分为不同规划结果，不能因为不必要的后续阶段失败就掩盖前面真实的可达性。

### 9.4 抓取优先与支撑脱离

`run_real_nero_safe.sh` 默认启用 `--grasp-validation-first`。普通目标首先只规划 `pregrasp → grasp → 闭合 → 安全支撑脱离`；通过后即冻结为可执行抓取计划，夹爪保持闭合，不再因为 full lift、搬运或放置失败而删除该抓取。清障任务必须真正搬走遮挡物，因此 `clear_policy.py` 仍会关闭 validation-only，并用同一套有界局部规划完成搬运预检。

短距离支撑脱离把环境分为三部分：目标包络内、与目标点云距离足够近的分割泄漏点；局部验证支撑面；其余外部环境。抓取闭合后不再使用半径 55 mm 的实心 `gripper_tool` 代表整个指端，而是继续使用两根半径 7 mm 的真实手指胶囊：空抓时两指中心重合，形成约 14 mm 的保守总包络；正常候选按 `GraspNet width + 10 mm` 构造仅用于碰撞检查的两指内间距，而且不受 85 mm 硬件开口截断，也不会改变实际夹爪命令或候选宽度筛选。夹爪掌部保留独立保护，但半径由 55 mm 减为 35 mm。`support_release/pickup_lift/carrying` 中只有两根闭合手指允许最多 10 mm 的环境点云接触；机械臂连杆、夹爪掌部、相机、附着载荷与桌面仍保持严格保护。若多个 IK 解在共同起点报告相同闭合手指或载荷碰撞，规划器立即短路并缓存失败，不再对其余 beam 父分支重复计算。

### 9.5 自体边缘漏点与融合孤立点

每帧 RGB-D 在固定夹爪自体掩膜后还会执行一次保守的边缘小岛过滤：只检查自体掩膜外 3 pixels 内、深度不超过 220 mm、面积不超过 8 pixels 且直径不超过 6 pixels 的完整近场连通域。较大的连续近场表面不会被裁掉。过滤结果保存为 `camera_self_edge_island_mask.png`、叠加图和 JSON 报告。

动态环境融合在 5 mm 体素层再次删除“只来自一个视角且同视角邻域内没有任何支持”的孤立体素；具有相邻体素的细篮子边缘继续进入 uncertain/hard 分层，多视角重复观测的体素始终保留。该过滤同时作用于融合显示兼容点云和碰撞地图，避免单像素自体漏点在执行期成为硬障碍。

安全包装脚本对支撑脱离仍按 5 mm XY 漂移和 3°倾角生成严格规划，但实机运行时默认增加 15 mm XY 与 5°倾角跟踪容差，即总计允许 20 mm、8°的非严格竖直脱离。相对已经达到的最高点还允许 5 mm 控制器到位回弹；超过该回弹包络时优先普通终止，而真正低于抓取起点仍保持严格停止。以上容差只作用于抓取闭合后的短距离 support-release；桌面、环境点云、连杆、相机和力矩保护不随之放宽。

## 10. 重构结果与已实现的规划改进

最初一轮完成的是无行为变化的结构重构；其后在独立模块上依次加入了桌面级动态环境地图、跨阶段 beam search 和单个中性关节点回退。当前结果包括：

- 根 runner 收缩为 17 行稳定兼容入口，完整实现全部位于 `real_nero_grasp/`；
- `runtime.py` 从原来的约 15,229 行降至约 2,900 行，现以兼容导出、动态依赖注入和 CLI 退出码为主；
- 感知、候选筛选、Amodal 清障、主动观察、点云融合、碰撞、TRAC-IK、任务预检、实机执行、运行时安全和顶层 workflow 均有独立 owner；
- TRAC-IK 与 URDF 链缓存分别只由 `ik_solver.py` 和 `collision.py` 持有；默认路径始终以项目根目录为基准；
- `grasp_geometry.py` 复用 `transforms.py` 的位姿转换，不再复制 `_as_pose_list`；包内模块不反向导入 `runtime.py`；
- 对旧测试和外部扩展会动态替换的复合调用，`runtime.py` 每次调用都构造 hooks 并注入当前全局函数，而不是缓存导入时别名；
- 裸 CLI 为兼容保持 beam search 和中性点回退默认关闭；实机安全包装脚本则显式启用两者，并继续使用冻结的环境地图做所有边的连续碰撞检查；
- `motion_planning.py` 为每个 TCP 阶段保留多个已通过碰撞检查的 IK 分支，`grasp_planner.py` 跨阶段组合并筛选完整任务路径，避免过早固定某个局部 Top-1 解；
- 全部候选和两种夹爪方向的 direct 路线均失败后，候选筛选器可对原候选集进行一次经中性关节点的全局第二轮；
- 主动观察采用快速层/完整层渐进搜索并限制实际拍摄次数，失败质量角度也参与去重；
- 抓取完整路线也采用快速 `beam=2/Top-K=2`、失败后完整 `beam=6/Top-K=6` 的渐进搜索；
- lift 优先自适应降低抬升高度，允许 J7 低代价变化，并通过全局端点预检、阶段缓存和固定起点失败短路，避免对每个 beam 父分支及无效 lift 变体重复求解；
- 上述改进没有改变碰撞阈值或硬关节限位，也不会在执行期重规划；点云模型净空触发优先使用普通轨迹终止，无法确认停止时才升级电子急停。
- 尚未引入候选级全局计算 quota，也没有放宽碰撞阈值或硬安全限制。

`runtime.py` 仍保留少量兼容薄包装、诊断显示适配、手眼变换入口和 `main()`。这些代码是旧模块导入及 `patch.object(real_nero_single_grasp_runner, ...)` 契约的一部分，不应仅为减少行数而删除。后续算法改进应继续直接落在对应 owner 模块，并作为独立变更重新验证实机安全。

## 11. 测试与验证

必须使用项目的 Conda 环境；系统 Python 的 NumPy/SciPy ABI 与项目环境不匹配，并且可能缺少 Torch。

完整单元测试：

```bash
source /home/zgj/anaconda3/etc/profile.d/conda.sh
conda activate grasp_yolo26
python -m unittest discover -s tests -p 'test_*.py'
```

重构前的测试基线为：

```text
Ran 180 tests
OK
```

当前完整回归结果为：

```text
Ran 287 tests
OK
```

静态检查：

```bash
python -m compileall -q real_nero_grasp real_nero_single_grasp_runner.py
bash -n run_real_nero_clean.sh
bash -n run_real_nero_safe.sh
bash -n run_real_nero_food5_clear.sh
python real_nero_single_grasp_runner.py --help
python -m real_nero_grasp --help
```

每次模块迁移至少验证：

1. 当前完整单元测试继续通过；
2. 旧模块仍能导入兼容契约中的名称；
3. `--help`参数和默认值没有变化；
4. 根脚本的退出码仍为0/2/3；
5. 代表性规划失败、普通轨迹停止和电子急停日志前缀没有变化；
6. 不带 `--execute` 的完整离线规划不会发送运动；
7. 实机验证必须在人工确认、低速和现有安全保护全部启用的条件下单独进行。
