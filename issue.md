# Whole-Grasp 当前问题与解决方案

更新时间：2026-10-12

本文针对当前 `whole-grasp` 工程给出五个问题的定位、修改方案和验证步骤。涉及机械臂运动的项目必须先做离线/无负载验证，再做低速实机验证。

## 1. 桌面高度比以前低：桌面保护在哪里设置

### 当前实现

桌面保护的核心参数定义在：

- `real_nero_grasp/cli.py:901-905`
- `real_nero_grasp/collision.py:108-129`
- 实际安全启动脚本：`scripts/run_grasp_safe.sh:139-142`

最低允许 TCP 高度的计算公式是：

```text
min_tcp_z = table_z + table_tcp_clearance + table_tool_bottom_offset
```

三个参数的含义：

- `table_z`：桌面在机器人 `base` 坐标系中的真实 Z，高度单位为米。
- `table_tcp_clearance`：希望在真实桌面上方额外保留的安全距离。
- `table_tool_bottom_offset`：TCP 到夹爪/工具最低点的几何余量，不是桌面高度补偿。

当前实际使用的安全启动脚本默认值为：

```bash
TABLE_Z=0.0
TABLE_TCP_CLEARANCE=0.00
TABLE_TOOL_BOTTOM_OFFSET=0.005
```

注意：`new/run_real_nero_safe.sh` 是旧快照，里面的默认工具余量还是 `0.02`，并且引用了当前不存在的 `new/run_real_nero_clean.sh`。当前应使用 `scripts/run_grasp_safe.sh`，不要再以 `new/` 下的脚本作为运行入口。

### 解决方案

1. 重新测量桌面相对于机器人基坐标系原点的 Z。
2. 如果桌面比基坐标系原点低，`TABLE_Z` 应为负数；例如桌面低 35 mm，则使用 `-0.035`。
3. 只修改 `TABLE_Z` 来表达桌面真实高度，不要通过减小 `TABLE_TOOL_BOTTOM_OFFSET` 或盲目增加 `--grasp-advance` 来补偿桌高错误。
4. 保留工具余量，除非重新测量并确认 TCP 到夹爪最低点的几何关系发生了变化。

安全启动脚本把 `"$@"` 放在固定安全参数之前，因此命令行中直接传入另一个 `--table-z` 可能会被脚本末尾的值覆盖。应通过环境变量设置：

```bash
cd /home/yx/yx/panda_ws/src/whole-grasp
conda activate grasp_yolo26

TABLE_Z=-0.035 \
TABLE_TCP_CLEARANCE=0.000 \
TABLE_TOOL_BOTTOM_OFFSET=0.005 \
bash scripts/run_grasp_safe.sh <其余原有参数>
```

上面的 `-0.035` 只是示例，必须换成实测值。

如果直接运行 `main.py`，则可以使用：

```bash
python main.py \
  --table-guard-enable \
  --table-z -0.035 \
  --table-tcp-clearance 0.000 \
  --table-tool-bottom-offset 0.005 \
  <其余参数>
```

### 为什么必须改真实 `table_z`

`table_z` 不只用于最终 TCP 高度检查，还用于：

- 抓取候选的桌面过滤；
- 夹爪手指最低点检查；
- 抓取物体底面与支撑面的检查；
- lift/support-release 的接触基线；
- 动态环境地图的 Z 范围；
- 规划路径和运行时路径保护。

因此只修改抓取深度会造成各模块使用不同的“桌面”，存在安全风险。

### 验证标准

- 启动时打印最终的 `table_z`、`min_tcp_z`，确认不是旧值。
- 空载、低速检查 pregrasp、grasp、lift 路径，不能先拿真实物体试。
- 日志中的 `table_z`、payload support plane 和实际桌面一致。
- 桌面保护仍然启用，不能用 `--no-table-guard` 绕过。

---

## 2. 重新标定腕部相机

### 当前生效版本（2026-10-12）

当前 ChArUco 检测、五算法求解和独立验证已经切换到 `/home/yx/下载/charuco_handeye` 的新标定方法。执行本节的 `solve` 命令时，会自动运行 TSAI、PARK、HORAUD、ANDREFF、DANIILIDIS 五种算法并按原项目规则推荐最优结果，不再使用早期适配版的单一 PARK 求解。

重要：原项目的正式采集不是每张图都在示教模式下手动拖动机械臂，而是自动回放预先记录的关节姿态，检查关节和法兰到位稳定后再采图。当前 SDK 版 `capture` 仍是临时手动采集入口，尚未接入这段自动姿态回放，因此目前不能称为完整复现原项目的采集流程。

原项目随附的姿态文件为：

```text
/home/yx/下载/charuco_handeye/poses/nero_handeye_poses_2026-09-01_11-13-44.json  # 35 个训练姿态
/home/yx/下载/charuco_handeye/poses/nero_handeye_poses_2026-09-01_15-23-38.json  # 9 个验证姿态
```

完整 SDK-only 适配应由 `pyAgxArm` 安全分段回放这些关节姿态，默认每个姿态都等待人工确认，满足关节、法兰位置和法兰姿态稳定阈值后才允许保存 ChArUco 样本。该自动回放采集入口完成前，不再使用下面的临时手动采集数据作为最终标定结果。

文件使用规则：

- `config/2026-10-12_wrist_charuco.json` 是切换到完整五算法之前生成的结果，不使用；
- 新五算法结果统一保存为 `config/2026-10-12_wrist_charuco_5method.json`；
- 默认 `validate` 与原项目的独立验证定义一致；
- `--same-board-pose` 只是附加的严格检查，不是原方法的必需参数；
- 在训练残差和独立验证都通过之前，不修改 `config/pipeline.yaml`。

### 当前推荐工具与标定关系

新的推荐工具是：

```text
tools/calibrate_d435i_charuco.py
```

它完成 D435i 腕部相机的 SDK-only eye-in-hand 标定：

```text
固定 ChArUco 板
      ↑ 相机观测 T_camera_target
腕部 D435i
      ↑ 待求 T_flange_camera
Nero 法兰
      ↑ SDK 反馈 T_base_flange
机器人 base
```

工具直接使用 RealSense SDK 和 `pyAgxArm`，不依赖 ROS/ROS2。连接 Nero 后只读取 `get_flange_pose()`；不会调用 `arm.enable()`，不会发送关节、直线或笛卡尔运动命令。机械臂应处于正常示教模式，由人手动改变姿态。

本工具适配自 `/home/yx/下载/charuco_handeye`，保持原方法的核心步骤不变：

- 相同的 8×6 ChArUco 板定义和 `solvePnP` 位姿估计；
- 同时运行 TSAI、PARK、HORAUD、ANDREFF、DANIILIDIS 五种 OpenCV 手眼算法；
- 使用 `translation_std_mm + rotation_std_deg` 的固定板一致性评分推荐方法；
- 输出逐样本误差，允许在人工确认异常原因后重新求解；
- 使用没有参与求解的独立姿态验证。

当前已完成的接口替换是：原程序的 ROS2/Orbbec 图像改为 RealSense SDK，ROS2 机械臂反馈改为 `pyAgxArm` SDK 实际法兰反馈。标定数学、坐标方向、五算法评分和独立验证方法不变。ROS2 自动姿态回放不能简单等价成手动示教采集，仍需单独完成 SDK 自动回放适配。

旧的 `tools/calibrate_d435i_aruco.py` 和已有标定文件保留作为历史工具和对照，不要删除或覆盖。

### 第 0 步：确认标定板与设备参数

当前配置文件是 `config/charuco_board.yaml`：

```yaml
board:
  squares_x: 8
  squares_y: 6
  square_length_m: 0.028729
  marker_length_m: 0.020950
  dictionary: DICT_5X5_100
  legacy_pattern: false
```

这里的 OpenCV 顺序是“横向列数、纵向行数”。因此通常所说的 6 行 × 8 列标定板，在程序里就是 `squares_x: 8`、`squares_y: 6`。标定板在相机中横放、竖放或旋转，不需要交换这两个值。

尺寸说明：

- `square_length_m` 是一个完整棋盘格的边长；
- `marker_length_m` 是黑色 ArUco marker 的边长，不包括周围白边；
- 当前数值来自之前实测的同一块实体板；
- 如果重新打印、缩放打印或更换了标定板，必须重新测量并修改配置后再采集；
- 字典和 `legacy_pattern` 必须与打印这块板时的定义一致。

标定前还要满足：

1. 腕部相机和支架已经锁紧，采集后不得再拆装或碰动。
2. ChArUco 板固定在刚性平面上，单次采集过程中不能移动。
3. 为了进行最强的训练板位姿一致性验证，建议训练和验证完成前都不要移动标定板。
4. 训练集与验证集必须分别重新采集，不能复制图片或复用训练数据。
5. 相机序列号使用 `243622072771`，图像使用 640×480@30 Hz。
6. Nero 当前已经实机确认使用 `V111` 反馈协议。

准备运行环境和 CAN：

```bash
cd /home/yx/yx/panda_ws/src/whole-grasp
conda activate grasp_yolo26

sudo ip link set can0 up type can bitrate 1000000
ip -details link show can0

python -c "import cv2; print(cv2.__version__, hasattr(cv2, 'calibrateHandEye'))"
```

当前 `grasp_yolo26` 已将 `opencv-python` 和 `opencv-contrib-python` 同步固定到 `4.13.0.92`。上面最后一条命令应输出 `4.13.0 True`。OpenCV 5.0 当前 wheel 缺少 `calibrateHandEye`，不能执行原项目的五算法求解。

### 第 1 步：只测试 ChArUco 识别

这一步只打开 D435i，不连接机械臂：

```bash
python tools/calibrate_d435i_charuco.py test \
  --board config/charuco_board.yaml \
  --serial 243622072771 \
  --width 640 \
  --height 480 \
  --fps 30
```

ChArUco 位姿估计只使用彩色图和彩色相机内参，因此该工具以 color-only 模式启动 D435i，不打开深度流。这可以避免深度 UVC 故障阻断标定，但不代表深度相机已经恢复；抓取点云、重建和真人手三维定位仍必须单独确认深度流正常。

窗口操作：

- 小写 `s`：把当前带检测结果的调试图保存到 `calibration/charuco_debug/`；
- 小写 `q` 或 `Esc`：退出。

完整可见的 8 列 × 6 行标定板最多有 24 个 marker 和 35 个 ChArUco 内角点。开始正式采集前应确认：

- marker ID 和角点能持续显示；
- 坐标轴稳定地贴在板面上，没有明显翻转或跳动；
- 完整板正对相机时应接近 `markers=24`、`corners=35`；
- 窗口顶部显示 `quality=OK`。

如果完整、清晰的标定板仍不能稳定识别，不要进入后续步骤，应先检查板型、字典、尺寸配置、打印质量和曝光。

### 第 2 步：临时手动采集入口（不是原方法的正式自动回放）

下面命令只用于检测、调试或在自动回放入口完成前检查数据格式。若目标是严格执行 `/home/yx/下载/charuco_handeye` 的完整新方法，不应把这一步采集的数据直接作为最终标定训练集。

输出目录必须是一个尚不存在的新目录；如果重试，应更换末尾编号，不要覆盖旧数据：

```bash
python tools/calibrate_d435i_charuco.py capture \
  --output calibration/charuco_train_20261012_yyxx \
  --board config/charuco_board.yaml \
  --serial 243622072771 \
  --width 640 \
  --height 480 \
  --fps 30 \
  --channel can0 \
  --firmware V111
```

程序不会控制机械臂。正确操作顺序是：

1. 在示教模式下手动把腕部相机移动到一个新视角。
2. 停止触碰机械臂，等待机械臂和相机画面稳定约 1 秒。
3. 确认窗口显示 `quality=OK`。
4. 按小写 `s` 保存当前样本。
5. 看到终端打印 `Saved sample ...` 后，再移动到下一个视角。

窗口操作：

- 小写 `s`：保存；
- 小写 `d`：删除最后一个已保存样本；
- 小写 `q` 或 `Esc`：结束采集。

按下 `s` 后，程序会读取采样前后的法兰反馈，并从 5 张候选图中选择重投影误差最小的有效图。如果这段时间腕部移动超过 1 mm 或 0.3°，样本会显示 `Rejected: Wrist moved during capture`，不会写入数据集。

训练集最低需要 12 个有效样本，建议采集 25～35 个。视角必须包含：

- 左右、上下平移；
- 近、中、远距离；
- 绕至少两个独立旋转轴的明显旋转；
- 标定板分别出现在画面中心、左侧、右侧、上侧和下侧；
- 每张图中仍有足够角点且无明显运动模糊和强反光。

不要只平移而不旋转，也不要连续采集大量几乎相同的姿态。旋转激励不足时，求解会拒绝并报告：

```text
Insufficient flange rotation about independent axes
```

本步骤生成的主要内容是：

```text
calibration/charuco_train_20261012_01/
├── board.yaml
├── samples.json
├── images/
└── debug/
```

### 第 3 步：求解手眼矩阵

注意输入参数是整个数据集目录，不是旧流程中的 `samples.jsonl`：

```bash
python tools/calibrate_d435i_charuco.py solve \
  --dataset calibration/charuco_train_20261012_yx \
  --output config/2026-10-12_wrist_charuco_5method.json
```

`--output` 必须指向一个尚不存在的新文件。脚本使用只新建、不覆盖模式保护已有结果；如果出现 `FileExistsError`，先检查已有 JSON，不要直接删除。确认需要重新求解时应改用新文件名，例如在末尾增加 `_02`。

求解阶段与 `/home/yx/下载/charuco_handeye/scripts/03_solve_handeye.py` 一致，同时运行：

```text
TSAI
PARK
HORAUD
ANDREFF
DANIILIDIS
```

每种方法都计算固定板跨姿态一致性，并使用下面的分数从小到大排序：

```text
score = translation_mm.std + rotation_deg.std
```

终端会打印五算法表格、`Recommended method`、按平移误差从大到小排列的逐样本报告，以及推荐方法的完整残差统计。输出 JSON 的 `all_methods` 保留五种结果，顶层矩阵是推荐方法的结果。

输出矩阵的明确方向是：

```text
T_flange_camera
P_flange = T_flange_camera @ P_camera
convention = flange_camera_xyzw
parent_frame = SDK_feedback_flange
child_frame = d435_color_optical_frame
```

新结果初始状态为 `unvalidated`，生成后不要立刻修改 `config/pipeline.yaml`。先检查训练统计；如果训练误差本身已经超限，就不要继续用该结果验证或接入工程。

如果逐样本报告指出异常，必须先检查对应的 `images/`、`debug/`、机械臂稳定性和标定板状态。只有确认样本确实存在采集问题，才能在相同 `solve` 命令后添加 `--exclude` 以及对应的整数样本编号，并输出到一个新的 JSON。不能仅因为某个样本使统计变差就随意排除，也不能照抄文档中的示例编号。

### 第 4 步：采集独立验证集

训练集不能用于验证。保持标定板固定，重新运行采集并使用全新的输出目录：

```bash
python tools/calibrate_d435i_charuco.py capture \
  --output calibration/charuco_valid_20261012_06 \
  --board config/charuco_board.yaml \
  --serial 243622072771 \
  --width 640 \
  --height 480 \
  --fps 30 \
  --channel can0 \
  --firmware V111
```

验证集最少需要 5 个有效样本，建议采集 8～12 个与训练集不同的新姿态。程序会记录图片 SHA-256；如果复制或复用了训练图片，验证会直接拒绝。

先运行与原项目 `scripts/04_validate_handeye.py` 一致的独立验证：

```bash
python tools/calibrate_d435i_charuco.py validate \
  --dataset calibration/charuco_valid_20261012_06 \
  --calibration config/2026-10-12_wrist_charuco_5method.json
```

重点查看输出中的 `internal_consistency`。它先求验证集中固定板位姿的鲁棒中心，再报告每个新姿态相对于该中心的平移和旋转偏差；这与原项目的验证定义相同。

如果从训练开始到验证结束，标定板完全没有移动，还可以额外运行更严格的跨数据集检查：

```bash
python tools/calibrate_d435i_charuco.py validate \
  --dataset calibration/charuco_valid_20261012_06 \
  --calibration config/2026-10-12_wrist_charuco_5method.json \
  --same-board-pose
```

此时输出中同时包含：

- `internal_consistency`：验证集各视角变换到 base 后是否聚合稳定；
- `relative_to_training_board`：验证结果是否回到训练时同一个固定板位姿。

如果训练和验证之间移动过标定板，不能添加 `--same-board-pose`，但仍可使用默认的 `internal_consistency` 完成原方法验证。验证程序只打印报告，不会自动修改标定 JSON 的 `status`。

原项目历史独立验证基线是：平移 mean/median/std/max 为 `5.639/5.665/2.052/9.572 mm`，旋转为 `0.884/0.775/0.229/1.290 deg`。当前没有自动通过门槛，可先使用接近该历史基线的工程目标判断：

```text
translation_mm median <= 6 mm
translation_mm std    <= 3 mm
translation_mm max    <= 10 mm
rotation_deg median   <= 1 deg
rotation_deg std      <= 0.5 deg
rotation_deg max      <= 2 deg
```

如果误差超限，不要接入真实机械臂任务。按顺序检查：

1. 标定板方格和 marker 实际尺寸是否正确；
2. 标定板类型、字典和 `legacy_pattern` 是否与打印文件一致；
3. 腕部相机或支架是否松动；
4. 保存样本时机械臂是否完全静止；
5. 是否缺少两个独立旋转轴的姿态变化；
6. 图像是否模糊、反光、曝光异常或角点太少；
7. 训练或验证过程中标定板是否移动。

### 可选：尺寸配置写错后的重新处理

如果图像和法兰反馈已经正确采集，只是之后发现 ChArUco 尺寸配置写错，可以使用新配置重新处理原图，不必重新采集机器人位姿：

```bash
python tools/calibrate_d435i_charuco.py reprocess \
  --dataset calibration/charuco_train_20261012_01 \
  --board config/charuco_board.yaml \
  --output calibration/charuco_train_20261012_01_reprocessed
```

输出仍然必须使用一个不存在的新目录。重新处理后，应使用新目录重新求解，并重新采集独立验证集。若相机安装、机械臂反馈或原始图像本身有问题，`reprocess` 不能修复，必须重新采集。

### 第 5 步：验证通过后接入 Whole-Grasp

保留旧标定 JSON，不要覆盖，以便回退。确认独立验证满足要求后，再修改 `config/pipeline.yaml`：

```yaml
wrist_camera:
  grasp_calib_json: /home/yx/yx/panda_ws/src/whole-grasp/config/2026-10-12_wrist_charuco_5method.json
  grasp_convention: flange_camera_xyzw
  servo_calib_json: /home/yx/yx/panda_ws/src/whole-grasp/config/2026-10-12_wrist_charuco_5method.json
  servo_convention: link7_cam
  servo_mount_frame: feedback_link7
```

同一个 `T_flange_camera` 在抓取模块中使用 `flange_camera_xyzw` 命名；当前伺服坐标链把同方向矩阵称为 `link7_cam`，并挂载到 `feedback_link7`。接入后必须先进行无物体、低速、可随时急停的坐标方向检查，确认相机坐标中的前后、左右、上下变换到机器人 base 后方向正确，再运行抓取或真人手跟踪。

`tools/verify_handeye_aruco_consistency.py` 仍可作为额外的单 ArUco 交叉检查工具，但它不是新 ChArUco 训练/验证流程的一部分，也不能替代上述独立验证集。

---

## 3. 重新训练 YOLO 模型

### 必须训练分割模型

当前抓取代码在 `real_nero_grasp/perception.py:106-180` 中读取 `result.masks`，目标点云由实例 mask 截取。因此应训练 YOLO instance segmentation，而不只是目标检测框模型。

当前环境已经安装：

```text
ultralytics 8.4.150
```

当前基础权重是：

```text
models/yolo26m-seg.pt
```

### 数据目录

建议建立独立数据集目录：

```text
datasets/whole_grasp_v1/
├── images/
│   ├── train/
│   └── val/
├── labels/
│   ├── train/
│   └── val/
└── data.yaml
```

`data.yaml` 示例：

```yaml
path: /home/yx/yx/panda_ws/src/whole-grasp/datasets/whole_grasp_v1
train: images/train
val: images/val

names:
  0: cup
  1: bottle
  2: can
```

每张图对应一个同名 `.txt` 标签。分割标签格式是：

```text
class_id x1 y1 x2 y2 x3 y3 ...
```

坐标必须归一化到 `[0, 1]`，轮廓点应沿物体可见边缘排列。不能只提供 `xywh` 检测框标签。

### 数据采集与划分建议

- 使用最终要部署的腕部 D435 拍摄。
- 包含不同距离、角度、光照、桌面、遮挡和背景。
- 包含空桌面和容易误检的负样本。
- 同一段视频的相邻帧不要同时分到 train 和 val，否则验证结果会虚高。
- 验证集应包含真正没见过的场景和摆放方式。
- 如果训练 visible mask，只标可见轮廓。
- 如果训练 amodal 模型，必须单独建立数据集并标注被遮挡部分的完整轮廓，通过 `--amodal-weights` 接入，不能与 visible 标签混用。

### 微调训练

```bash
cd /home/yx/yx/panda_ws/src/whole-grasp
conda activate grasp_yolo26

yolo segment train \
  model=models/yolo26m-seg.pt \
  data=datasets/whole_grasp_v1/data.yaml \
  epochs=150 \
  imgsz=640 \
  batch=8 \
  device=0 \
  workers=4 \
  project=runs/segment \
  name=whole_grasp_v1
```

如果显存不足，先把 `batch=8` 改为 `4` 或 `2`，不要先降低图像尺寸，因为小物体的 mask 边缘会明显受影响。

### 验证

```bash
yolo segment val \
  model=runs/segment/whole_grasp_v1/weights/best.pt \
  data=datasets/whole_grasp_v1/data.yaml \
  imgsz=640 \
  device=0
```

除了 box/mask mAP，还应人工检查：

- mask 是否覆盖完整物体；
- 是否把桌面包含进 mask；
- 遮挡情况下 mask 是否跳到邻近物体；
- 相邻物体是否错误合并；
- 连续视频中的 mask 是否稳定。

### 不连接机械臂的 D435 测试

```bash
python scripts/test_D435i.py \
  --weights runs/segment/whole_grasp_v1/weights/best.pt \
  --target-class cup
```

也可以测试保存过的 RGB-D：

```bash
python tools/test_d435i_grasp_pointcloud.py \
  --capture-dir <某次包含color.png/depth_m.npy/capture_meta.json的目录> \
  --output /tmp/whole_grasp_yolo_probe \
  --yolo-weights runs/segment/whole_grasp_v1/weights/best.pt \
  --target-class cup \
  --view
```

### 接入抓取

先不要覆盖原来的权重，显式传入新权重：

```bash
bash scripts/run_grasp_safe.sh \
  --yolo-weights /home/yx/yx/panda_ws/src/whole-grasp/runs/segment/whole_grasp_v1/weights/best.pt \
  --target-class cup \
  <其余原有参数>
```

确认实机数据稳定后，再将最终权重复制到 `models/` 并修改默认路径。

---

## 4. Ctrl+C 后抓取保持使能，但伺服掉使能

### 根因

抓取部分和伺服部分使用了两套退出机制。

抓取部分：

- `real_nero_grasp/arm.py:290-359` 会先确认轨迹停止；
- 必要时发送普通轨迹终止和当前关节保持；
- 最终 SDK `disconnect()` 不主动调用电机 `disable()`；
- 因此 Ctrl+C 后通常仍保持使能。

伺服部分：

- `servo_delivery.py:648-662` 捕获 Ctrl+C 后进入 `finally`；
- `finally` 无条件调用 `backend.disconnect()`；
- `control_backends/sdk_force_ctrl_backend.py:291-304` 的 `disconnect()` 调用 `NeroForceController.stop()`；
- `control_backends/force_ctrl/controller.py:274-349` 的 `stop()` 明确执行：停止 MIT 线程、`robot.disable()`、断开连接。

所以伺服 Ctrl+C 掉使能是当前代码的确定行为，不是偶发现象。

### 不能采用的最小改法

不能简单删除 `backend.disconnect()`。伺服阻抗依赖 200 Hz MIT 线程持续发送指令；Python 进程退出后线程消失，如果没有先把控制权交回固件位置模式，机械臂可能失去稳定保持。

也不能在 Ctrl+C 时仅调用 `disable_cartesian_impedance()`，因为当前该函数会退回 `kp=0` 的零重力/阻尼命令，负载下可能下沉。

### 正确方案：增加“保持使能退出”通道

建议新增两个明确区分的方法：

```text
disconnect_disable()       # 现有行为：停止、掉使能、断开
hold_enabled_disconnect()  # Ctrl+C：交回固件位置保持、保持使能、断开
```

`hold_enabled_disconnect()` 应在 `NeroForceController` 内完成以下原子化流程：

1. 停止接受新的视觉目标。
2. 读取新鲜、有限的 7 轴关节角 `q_hold`。
3. 在 MIT 线程仍工作时，将目标切成当前关节位置保持，速度和目标 twist 清零，并确认连续若干周期稳定。
4. 停止 MIT 线程并确认线程已经退出；如果线程无法退出，沿用当前电子急停逻辑。
5. 使用底层 `pyAgxArm` 驱动切回 normal/joint position 模式。
6. 向固件发送 `move_j(q_hold)`，即“在当前位置保持”，不能发送旧目标。
7. 读取关节使能状态并确认仍使能；读取数个关节反馈确认没有继续运动。
8. 只断开 CAN/SDK 通信，不调用 `robot.disable()`。
9. 任一步骤无法确认时，回退到现有 `stop()`：宁可掉使能/急停，也不能假装保持成功。

建议修改位置：

- `control_backends/force_ctrl/controller.py`
  - 新增 `stop_keep_enabled()`。
- `control_backends/sdk_force_ctrl_backend.py`
  - 新增 `hold_enabled_disconnect()`，调用 controller 的新方法。
- `servo_delivery.py`
  - `KeyboardInterrupt` 路径调用 `hold_enabled_disconnect()`；
  - `finally` 需要知道清理已经完成，不能再次调用普通 `disconnect()`；
  - 正常结束、Ctrl+C、普通异常、fatal safety 分成明确策略。

推荐策略表：

| 退出原因 | 处理方式 |
|---|---|
| Ctrl+C，反馈正常 | 固件当前位置保持，保持使能，然后断开 |
| 正常完成递送 | 根据实验需求选择保持使能或正常 disable，必须显式配置 |
| 相机丢失/普通可恢复异常 | 停止目标更新并保持最后安全目标；退出时尝试当前位置保持 |
| 关节反馈失效、MIT 线程故障、越限、控制不稳定 | 保留现有 fail-closed：电子急停/disable |

同时应处理 `ServoDeliveryNode._shutdown()` 中直接调用 `backend.disconnect()` 的路径，避免它绕过新的退出策略。

### 必须增加的测试

单元测试：

- Ctrl+C 只调用 `hold_enabled_disconnect()`，不调用普通 `disconnect()`。
- 当前位置保持交接成功后，`disable()` 调用次数为 0。
- MIT 线程退出失败时必须调用电子急停。
- fatal safety 仍然调用电子急停/disable，不能被“保持使能”选项覆盖。
- 清理函数幂等，重复调用不会再次切控制模式。

实机测试顺序：

1. 无夹持物、低速、有人扶持/具备机械限位保护时测试。
2. 启动纯位置保持，不运行视觉跟踪，按 Ctrl+C。
3. 检查七轴使能状态、关节是否静止、是否存在旧目标恢复。
4. 再测试笛卡尔阻抗静止状态。
5. 最后才测试带负载和完整递送流程。

---

## 5. 重建效果不如原始版本：是否因为相机去噪没有开启

### 当前事实

目前不能直接得出“相机去噪没开”的结论，因为当前代码已经有三层处理：

1. `cameras/realsense.py:59-68` 已将 D435 深度预设设置为 `high_density`。
2. `real_nero_grasp/cli.py:30-32` 默认在每个静止视角采集 7 帧深度，要求至少 5 帧有效，并用 4 mm temporal MAD 去除不稳定像素。
3. `real_nero_grasp/perception.py` 和 `depth_processing.py` 还包含目标点云统计离群点过滤、体素降采样以及小孔洞修复。

当前确实没有启用 librealsense 的：

- spatial filter；
- temporal filter；
- hole filling filter；
- disparity-domain filter chain。

同时也没有显式设置并记录：

- `emitter_enabled`；
- `laser_power`；
- 彩色自动曝光状态；
- 实际生效的全部深度传感器选项。

原始 `real-nero-grasp-main/graspnet_All/grasp.py` 只显式开启了彩色自动曝光，并没有发现 SDK 空间/时间深度去噪。因此“原始效果好”不一定来自 RealSense 深度滤波。

还有一个重要差异：

- 旧 `new/run_real_nero_safe.sh` 启用了 `--camera-self-mask`；
- 当前 `scripts/run_grasp_safe.sh:35` 明确使用 `--no-camera-self-mask`。

如果你所说的“重建差”包含夹爪/相机自身点云、近场鬼影或环境地图污染，这个差异可能比 SDK 去噪更直接。但固定 self-mask 必须与当前相机安装和分辨率吻合，不能未经预览直接打开。

### 推荐诊断顺序

#### A. 先固定其他变量做同姿态对照

在完全相同的机器人姿态、物体位置、光照和分辨率下保存：

- 原始单帧深度；
- 当前 7 帧 median/MAD 深度；
- `high_density`；
- `high_accuracy`；
- emitter/laser 显式开启后的结果；
- 可选 spatial filter 结果。

每组至少记录：

- 有效深度像素数量；
- 桌面平面 Z 标准差/MAD；
- 目标 mask 内深度 MAD；
- 目标边缘向背景泄漏的像素数量；
- 单视角点数和融合后点数；
- 多视角配准前后的 median/P95 残差。

#### B. 先显式控制并记录硬件选项

在 `RealSenseCamera` 中增加可配置参数：

```text
camera_visual_preset = high_density | high_accuracy | default
camera_emitter_enabled = 0 | 1
camera_laser_power = 数值或 max
camera_enable_auto_exposure = 0 | 1
```

启动后读取实际值并写入每次 `capture_meta.json`。不能只调用 `set_option()` 而不记录最终值。

建议先比较 `high_density` 与 `high_accuracy`：

- `high_density` 通常保留更多点，但可能增加边缘飞点；
- `high_accuracy` 点更少，但物体表面和边界可能更干净。

抓取重建并不一定是点越多越好。

#### C. 再试 SDK 后处理，不要一次全开

推荐逐项 A/B：

1. raw；
2. raw + spatial；
3. raw + spatial + temporal；
4. raw + spatial + temporal + hole filling。

注意：

- temporal filter 在相机或物体运动时可能产生残影。当前腕部相机会移动，因此只能在机器人完全静止后的帧序列中使用并重置滤波历史。
- spatial filter 会平滑物体边缘，可能把桌面深度带进目标 mask。
- librealsense hole filling 可能跨越目标边界，不能替代当前只在目标 mask 内填小孔的保守逻辑。
- 不建议一开始使用 decimation filter，因为会降低小物体和边缘的空间分辨率。

#### D. 检查不是“相机去噪”的其他原因

优先排查：

1. 新旧手眼标定误差，尤其是旋转误差；
2. 当前 640×480 是否低于原实验使用的分辨率；
3. 多视角运动结束后是否真正静止再采集；
4. YOLO mask 是否比原模型更松或包含桌面；
5. `--camera-self-mask` 的差异；
6. target outlier filter、voxel size 是否与原版相同；
7. 多视角配准/ICP 是否拒绝了本应接受的视角，或错误接受了差视角；
8. D435 发射器是否被其他相机红外干扰。

### 建议实现

1. 给 `cameras/realsense.py` 增加相机控制参数和 raw/filtered 双路保存。
2. 给 `cameras/options.py` 和 `config/pipeline.yaml` 增加对应配置。
3. 在 `capture_meta.json` 保存全部实际相机设置和滤波链。
4. 新增相机-only A/B 工具，不连接机械臂，只比较固定场景的深度质量。
5. 验证指标通过后再把滤波接入正式抓取；不要直接修改正式流且只凭肉眼判断。

### 临时检查命令

可先使用现有相机-only 工具查看当前点云，不连接机械臂：

```bash
python tools/test_d435i_grasp_pointcloud.py \
  --output /tmp/d435_current_probe \
  --width 640 \
  --height 480 \
  --fps 30 \
  --warmup 30 \
  --yolo-weights models/yolo26m-seg.pt \
  --target-class cup \
  --view
```

如果目标类别不是 `cup`，替换为实际类别。每次 `--output` 必须使用新的不存在目录。

---

## 推荐处理顺序

1. 先实测并修正 `TABLE_Z`，因为它影响所有抓取安全检查。
2. 重新标定腕部相机，并用独立验证集验收。
3. 用相同标定与固定场景做深度/重建 A/B，对比后再决定是否增加 SDK 滤波。
4. 训练并离线验证新的 YOLO-seg 模型，再接入点云测试。
5. 最后实现伺服 Ctrl+C 的保持使能交接，并严格按无负载到有负载的顺序测试。

