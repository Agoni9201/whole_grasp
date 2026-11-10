# whole-grasp

whole-grasp 是 Nero 七轴机械臂上的整段抓取与递送工程。它用腕部 RGB-D 相机看物体，完成分割、补看、抓取规划和执行，再把物体递到人手并判断能否松手。

当前默认相机是腕部奥比 Gemini 330，配置在 [`config/pipeline_orbbec.yaml`](config/pipeline_orbbec.yaml)。早期香蕉对照使用过腕部 D435，那些记录不能直接代表现在的奥比结果。研究进度和哪些结论已经用真机验证，以 [`RESEARCH_STATUS.md`](RESEARCH_STATUS.md) 为准。

## 整段流程

程序有三个阶段，由 `--phase` 选择：

| 阶段 | 含义 |
| --- | --- |
| `grasp` | 只做抓取，停在抓住并抬升保持 |
| `servo` | 只做伺服递送，用于单独调试 |
| `all` | 先抓取；加上 `--enable-servo-delivery` 且抓取成功后，再进入递送 |

```text
腕部 RGB-D
  → YOLO 分割，反投影出目标和场景点云
  → 可选清障，再重新采集
  → 可选主动观察：选下一个视角、运动、静止采集、配准并融合
  → 可选 MGPC 补全，和实测点一起送给抓取网络
  → GraspNet 或 GraspGenX 产生候选
  → TRAC-IK、碰撞检查、路径规划
  → move_j 或 MIT 执行，慢闭夹爪，确认接触后抬升
  → 可选递送：人手目标、PBVS 外环、笛卡尔阻抗内环
  → 可选交接：视觉与七轴力矩一起判断是否开夹
```

抓取是“到观察位、静止采集、规划、再运动”。递送才按视觉持续修正。两条链路共用腕部相机，但分阶段开关采集。

主动补看、MGPC 和交接融合都是可选分支。不打开对应开关时，仍走原来的单视角 GraspNet 抓取和原有递送。这些新分支的离线测试和真机结果并不相同，不能把“代码里有”写成“已经提高成功率”。

## 代码怎么分工

| 路径 | 内容 |
| --- | --- |
| [`main.py`](main.py) | 启动入口，转入 `real_nero_grasp.runtime` |
| [`real_nero_grasp/`](real_nero_grasp/) | 感知、主动视角、融合、规划、执行和安全停止 |
| [`cameras/`](cameras/) | 奥比与 RealSense 采集、预检和流水线参数 |
| [`models/`](models/) | YOLO 权重、手部模型、Nero 模型 |
| [`manipulator_grasp/`](manipulator_grasp/) | 机械臂与夹爪模型、仿真资源 |
| [`graspnet-baseline/`](graspnet-baseline/) | GraspNet |
| [`third_party/MGPC/`](third_party/MGPC/) | 点云补全；本地改过 `models/MGPC.py` |
| [`third_party/GraspGenX/`](third_party/GraspGenX/) | 抓取候选生成；含本地夹爪资产和脚本改动 |
| [`servo/`](servo/)、[`servo_delivery.py`](servo_delivery.py) | 视觉伺服、阻抗与递送安全 |
| [`handoff/`](handoff/) | 抓取到递送的交接，以及视觉/力矩开夹 |
| [`control_backends/`](control_backends/) | `move_j`、SDK 速度和力控后端 |
| [`targets/`](targets/) | 人手、ArUco 和深度手部目标 |
| [`config/`](config/) | 相机流水线、手眼和外参 |
| [`calibration/`](calibration/) | 手眼与自遮罩的数值结果 |
| [`tools/`](tools/)、[`scripts/`](scripts/) | 标定、回放、诊断和启动脚本 |
| [`tests/`](tests/)、[`tests_servo/`](tests_servo/) | 抓取与伺服测试 |
| [`research/`](research/) | 主动感知、交接、GenX 三个专题 |

`real_nero_grasp/runtime.py` 负责按阶段把模块接起来。感知、视角、规划和执行的实现在同包的独立模块里。模块边界和依赖方向见 [`real_nero_grasp/README.md`](real_nero_grasp/README.md)。

## 启动

在工程根目录：

```bash
python main.py --config config/pipeline_orbbec.yaml
python -m real_nero_grasp --config config/pipeline_orbbec.yaml
bash scripts/run_grasp_safe.sh --config config/pipeline_orbbec.yaml
```

`main.py` 和 `python -m real_nero_grasp` 进入同一个 `runtime.main()`。`scripts/run_grasp_safe.sh` 会在参数后追加低速、路径保护和验证模式，后出现的同名参数会覆盖前面的值。

常用参数：

- `--phase grasp|servo|all`
- `--enable-servo-delivery`：抓取成功后进入递送
- `--grasp-generator graspnet|graspgenx`
- `--execute`：允许发送抓取运动

抓取会话在进入机械臂上下文时就会使能并保持当前位置。没有 `--execute` 也不等于全程离线。实机命令、历史试验和当前外参以 [`HRI_TEST_COMMANDS.md`](HRI_TEST_COMMANDS.md) 为准。执行顺序的逐函数说明在 [`STEP4_CODE_EXECUTION_GUIDE.md`](STEP4_CODE_EXECUTION_GUIDE.md)；其中相机描述写于 2026-09-16，当前腕部相机以奥比配置为准。

## 仓库里没有的数据

克隆这个仓库得不到运行日志，也得不到下面这些大文件：

- `logs/`、`Log/`、`real_nero_single_grasp_logs/` 和所有 `*.log`
- `artifacts/` 里的实验产物，`captures/` 里的原始采集
- `calibration/` 中的图片、视频和压缩包；yaml 与 json 结果仍在仓库里
- `.env`
- Python 虚拟环境和 `__pycache__`
- `third_party/MGPC/pretrained/`
- `third_party/GraspGenX/ext/graspgenx_checkpoints/`

仓库里保留了四个分割权重：`models/yolo26l-seg.pt`、`models/yolo26m-seg.pt`，以及两个 food5 权重。GraspGenX 的一部分夹爪网格通过 Git LFS 存放。MGPC 与 GraspGenX 的预训练检查点需要按各自上游说明另行准备。

## 文档

| 文档 | 用途 |
| --- | --- |
| [`RESEARCH_STATUS.md`](RESEARCH_STATUS.md) | 论文目标、三个专题的进度和已核实事实 |
| [`research/`](research/) | 主动感知、交接、GenX 的专题记录 |
| [`HRI_TEST_COMMANDS.md`](HRI_TEST_COMMANDS.md) | 实机命令和试验记录 |
| [`STEP1_GRASP_MIGRATION.md`](STEP1_GRASP_MIGRATION.md) 到 [`STEP4_CODE_EXECUTION_GUIDE.md`](STEP4_CODE_EXECUTION_GUIDE.md) | 迁移步骤和代码导读 |
| [`OPEN_ISSUES.md`](OPEN_ISSUES.md)、[`issue.md`](issue.md) | 历史问题；日期较早，不直接当作当前故障 |
