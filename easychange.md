# 常用修改与运行命令

人机交互的独立测试命令集中在 [HRI_TEST_COMMANDS.md](HRI_TEST_COMMANDS.md)，
包括直接真机交接、仅跟随、空夹爪比例开合、相机深度诊断及关闭记忆接近的对照命令。
本次跳过抓取直接测新交接逻辑，使用该文件第1项。
从多视角抓取到双手松夹的完整流程使用该文件第7项。

先进入环境：

```bash
cd /home/yx/yx/panda_ws/src/whole-grasp
conda activate grasp_yolo26
```

当前 `config/pipeline.yaml` 已启用当前腕部相机的夹爪自身遮罩，
safe 脚本不再强行关闭它。适用条件：相机 `243622072771`、彩色 1920×1080、
原生深度 1280×720、当前安装与本轮 100 mm 工作开口。遮罩 JSON 为
`config/2026-10-26_camera_self_mask_100mm_multiframe.json`，使用四次保存图像的稳定区域和边缘包络，并保留彩色目标边缘；每帧会核对内参和黑色夹指轮廓。原 85 mm 遮罩仍保留用于历史数据。
更换安装、开口或图像模式后，需重新标定遮罩；不匹配时会中止采集。
每次抓取采集目录会保存 `camera_self_mask_overlay.png` 和
`camera_self_mask_report.json`，用于检查实际删除区域。
点云与抓姿预览窗口由独立 Python 进程运行，避免窗口事件循环占用主控制进程的解释器锁；父进程等待时仍检查 MIT 状态。

机械臂碰撞几何默认使用 `urdf_capsules`：从 IK URDF 的七段 collision 网格生成独立包络，覆盖全部三角形并保留 3 mm 几何余量，替换原来统一 60 mm 的关节／连杆模型。
NBV、抓取规划和运行时保护共用此模型，连接机械臂前预加载；相机支架与夹爪沿用现有模型。原抓取命令无需增加参数。
新旧模型对比、原始深度支持修复和离线复查命令在 [HRI_TEST_COMMANDS.md](HRI_TEST_COMMANDS.md) 第9项。

新采集还会保存整个深度采样窗口的 `depth_raw_frames.npz`，与各帧序号、
时间戳一起归档到 `multiview_raw`。SDK 滤波中值现需通过同一原始窗口的有效帧数和 MAD 检查；
默认七帧至少五帧有效、MAD 不超过 4 mm，SDK／原始中值差不超过 8 mm。不会仅凭最后一帧原始无效就删除此前有充分支持的测量。
采集新增 `depth_support_report.json`、`depth_support_rejected.png`、`depth_raw_valid_count.npy` 和校验前的 `depth_sdk_m.npy`，
`depth_m.npy` 为校验后的结果。MIT 模式使用数值子进程和临时数组文件；缺测不填成桌面，碰撞间隙要求保持原值。

## MGPC 离线重建可视化（不运行 GraspNet）

已生成单视角、多视角、单视角＋MGPC、多视角＋MGPC 四窗对比，使用 22:19 保存的双视角短瓶数据。
默认显示 RGB 原色，可切换来源配色（青色实测、橙色 MGPC 预测）。预测表面只有深度观测支持的部分上色，未观测部分为灰色。

```bash
firefox artifacts/mgpc_compare/comparison.html
```

页面支持同步旋转、缩放、只看目标及叠加实测参考。
图片预览为 `artifacts/mgpc_compare/environment_comparison.png` 和 `target_comparison.png`，
四组完整场景 PLY 也在同目录。模型和权重位于 `third_party/MGPC/`。

四组 GraspNet 候选、完整筛选记录及碰撞位置可视化：

```bash
firefox artifacts/mgpc_graspnet_compare/comparison.html
```

在页面下方各组的候选下拉框选择姿态；可按淘汰原因筛选，查看红色碰撞见证点及对应部件。
四组共用实测碰撞环境。结果是离线数值预检，不代表真机成功率，详细限制见同目录 `README.md`。
只重建页面无需 GPU：`bash scripts/run_mgpc_graspnet_compare.sh --stage render`。

如需用已下载的权重重新计算（需要 GPU，不连接机械臂）：

```bash
bash scripts/run_mgpc_offline_compare.sh --stage all
```

## 当前高目标场景的多视角测试命令

23:17 数据中，瓶子位于纸盒上，正上方 35 cm 姿态未找到 IK；保持 35 cm 观察半径的
15° 倾斜回退已通过保存数据的离线路径检查。以下命令只先试一个正上方 roll，
跳过固定高位的小角度光轴搜索，尽快进入原有倾斜回退。适用于这次已排查的场景，
保留实际运行的完整 IK、碰撞检查和运动确认。

```bash
bash scripts/run_grasp_safe.sh \
  --config config/pipeline.yaml \
  --phase grasp --execute \
  --target-class bottle \
  --yolo-weights models/yolo26m-seg.pt \
  --active-view \
  --no-active-multiview-gate \
  --active-side-view-count 1 \
  --active-center-height 0.35 \
  --active-center-roll-plan-candidates 1 \
  --no-active-center-optical-relaxation \
  --active-capture-settle-seconds 1.0 \
  --active-fusion-rigid-icp \
  --active-reg-icp-estimation point_to_point \
  --active-reg-icp-voxel 0.0025 \
  --active-table-reg-max-height-shift 0.030 \
  --grasp-advance 0.015 \
  --save-active-raw-views \
  --preview-cloud
```

这会真实移动机械臂；上一轮退出请求过电子急停，测试前应恢复正常就绪。
预期出现 `[active][center-fallback] selected ... tilt=15.0deg ...`，
随后等待观察运动确认。此检查只覆盖初始位姿到倾斜观察锚点，后续侧视角和抓取仍需验证。

## 1. 修改YOLO分割模型

推荐运行时指定，不改源码：

```bash
--yolo-weights /绝对路径/best.pt
```

默认模型路径在：

```text
real_nero_grasp/paths.py
DEFAULT_YOLO_WEIGHTS = .../models/yolo26m-seg.pt
```

## 2. 更换腕部相机标定参数

将验证合格的JSON放入：

```text
/home/yx/yx/panda_ws/src/whole-grasp/config/
```

然后修改`config/pipeline.yaml`中的两项：

```yaml
wrist_camera:
  grasp_calib_json: /home/yx/yx/panda_ws/src/whole-grasp/config/新标定.json
  servo_calib_json: /home/yx/yx/panda_ws/src/whole-grasp/config/新标定.json
```

## 3. 完整联调：多视角抓取 + 人手跟踪（可选相机可视化）

下面以 `bottle` 和 `yolo26m-seg.pt` 为例；换目标时同步修改类别和模型路径。
启动时用 `--servo-preview` 选择打开窗口，省略该参数就关闭，不修改配置默认值。

```bash
sudo ip link set can0 up type can bitrate 1000000

GRASP_PREFLIGHT_MAX_CANDIDATES=18 bash scripts/run_grasp_safe.sh \
  --config config/pipeline.yaml \
  --phase all \
  --execute \
  --target-class bottle \
  --active-view \
  --no-active-multiview-gate \
  --active-side-view-count 2 \
  --save-active-raw-views \
  --preview-cloud \
  --yolo-weights models/yolo26m-seg.pt \
  --trajectory-executor mit \
  --mit-q-log \
  --mit-max-joint-vel-deg-s 15 \
  --mit-max-joint-accel-deg-s2 60 \
  --enable-servo-delivery \
  --target-source mediapipe \
  --servo-follow-only \
  --servo-preview \
  --servo-diagnostics \
  --delivery-distance 0.20 \
  --max-linear-vel 0.01 \
  --max-joint-vel 0.08 \
  --follow-max-error-rad 0.03 \
  --impedance-target-linear-vel 0.01
```

该命令会真实驱动机械臂。只有抓取流程成功后才进入人手伺服；抓取失败不会强行切换。
窗口仅在进入伺服阶段后打开，显示 RGB 骨架／掌心、相机和基坐标 XYZ、深度有效点数、
`TRACKING / LOST / HOLD / ACQUIRING` 及原因；不是另开一个相机测试程序。
`--servo-follow-only` 表示只跟踪、不进入自动递交／松爪／撤退；目标距离为 20 cm，
不代表真实手部与全臂始终具有 20 cm 安全间距。首次仍应空载、固定替代目标验收后再接真人手。
命令已开启 `--preview-cloud`，关闭点云窗口后才继续；无需预览时可去掉此参数，不影响数据保存。
可视化需要桌面环境，不要同时运行另一个相机预览程序。预览本身不证明位姿精度或近人安全。

### 保存每个视角，供离线更换融合方法

主动多视角默认保存融合前输入，可显式加 `--save-active-raw-views`。
每次成功采集后立即保存，早于重叠检查、配准和融合；被质量门拒绝的视角也保留。
归档路径在终端显示为 `[active][archive] ...`，位于本次初始采集目录下：

```text
logs/<会话>/<初始采集时间>/multiview_raw/
  manifest.jsonl
  initial_<时间>_<唯一ID>/
  center_<时间>_<唯一ID>/
  side1_<时间>_<唯一ID>/
  side2_<时间>_<唯一ID>/
```

每个视角目录包含：

- `target_camera.ply`、`scene_camera.ply`：该视角相机坐标系的目标和场景。
- `target_base.ply`、`scene_base.ply`：只应用手眼变换到 `base_link`，未做 ICP 或多视角修正。
- `obstacle_camera.ply`、`obstacle_base.ply`：有非目标障碍云时保存。
- `clouds.npz`：完整精度的相机／基坐标 XYZ 与 RGB 数组，避免 PLY 字节颜色量化。
- `metadata.json`：时间戳、`T_base_camera`、关节角、手眼快照、单位及视角接受／拒绝状态。
- `color.png`、`depth_m.npy`、`capture_meta.json`、`selected_target_mask.png`：已有 RGB-D、内参、掩码文件的副本；缺失会明确记录。

`manifest.jsonl` 用相对路径索引全部视角；重复侧视标签使用不同目录，不覆盖之前的尝试。
离线配准可直接使用 `*_camera.ply` 和 `T_base_camera`，或使用未修正的 `*_base.ply`。
这些是**经过单视角处理、未配准和未融合**的点云；重新建云可从保存的对齐深度和彩色图开始。
深度可能已做多帧中值，实际参数见 `capture_meta.json`；位姿来自当前流程反馈采样，不宣称 RGB-D 严格同步。
初始视角也会归档，但中心优先模式下它不参与目标融合。需要节省磁盘时用 `--no-save-active-raw-views`。

### 先单独测试人手伺服（跳过抓取／NBV）

先将机械臂放在周围无遮挡的测试位姿；首次去掉 `--execute` 检查相机／识别链路。
此时用的是模拟关节，不读取真实机械臂，不能验收基坐标定位或实际运动。
以下带 `--execute` 的命令会真实运动，以 20 cm 为跟踪目标距离，不自动递交、松爪或撤退：

```bash
bash scripts/run_grasp_safe.sh \
  --config config/pipeline.yaml \
  --phase servo --execute \
  --target-source mediapipe --servo-follow-only \
  --servo-preview --servo-diagnostics \
  --delivery-distance 0.20 \
  --max-linear-vel 0.01 --max-joint-vel 0.08 \
  --follow-max-error-rad 0.03 \
  --impedance-target-linear-vel 0.01
```

人手伺服已恢复原独立脚本的低增益关节跟随辅助：`Kp=[2,2,2,2,1,1,1]`，
随参考速度启用（`--follow-min-qdot 0.02`），助力增加时按
`--follow-kp-rise-time 0.20` 限制上升速度，被推到参考前方超过 `0.02 rad` 时释放。
启动时不加辅助；零参考速度和安全 HOLD 时关闭辅助。短时丢手会在
`0.30 s` 内将速度与辅助增益平滑衰减，超过目标超时后才清空旧参考。
加 `--follow-kp 0` 可关闭辅助做对照；也支持如 `--follow-kp 2,2,2,2,1,1,1`。
`--max-joint-vel 0.08` 限制参考关节速度（rad/s），`--follow-max-error-rad 0.03`
独立限制每轴参考与实测角度差（rad）；调速度不会同时放大参考领先角度。
辅助仍按每轴实际参考 `qdot` 衰减：`qdot=0.005 rad/s` 时目标为 25% 增益，
达到 `0.02 rad/s` 时目标为满额；从零升到设置上限约需 `0.20 s`。
每轴单独计算，零参考速度时仍撤去助力；提高速度上限不保证满额 Kp，也不保证解决跟踪问题。
上述命令使用末端参考限速 `0.01 m/s`；笛卡尔 K/B 和 15 mm 目标限幅沿用现有设置。
`--max-joint-vel` 不是实际关节速度硬上限。
MediaPipe VIDEO 跟踪的检测、存在和跟踪置信度默认均为 `0.35`，减少连续视频中的
`no_hand_2d` 短暂断档；需要更严格筛选时可显式传回 `0.5`。

#### 已能连续跟手后，对比 2 cm/s 参考速度

2026-10-20 的 `21-38-40` 实测日志中，连续跟踪阶段约为 26～28 Hz，阻抗内环约
180～190 Hz；原 PBVS 核心与本工程一致。命令中的 `0.01 m/s` 使参考移动 5 cm
至少需要 5 秒，15 mm 参考偏差上限也经常触发，实际机械运动可能更慢。
以下命令保留该次的 `Kp=4`、关节参考速度和偏差限制，只将两处末端速度上限同步
改为 `0.02 m/s`（原独立阻抗脚本默认的参考限速），用于下一轮实际响应对比：

```bash
bash scripts/run_grasp_safe.sh \
  --config config/pipeline.yaml \
  --phase servo --execute \
  --camera-width 640 --camera-height 480 --camera-fps 30 \
  --target-source mediapipe --servo-follow-only \
  --servo-preview --servo-diagnostics \
  --hand-depth-radius 8 \
  --delivery-distance 0.20 \
  --max-linear-vel 0.02 --max-joint-vel 0.08 \
  --follow-kp 4 --follow-max-error-rad 0.03 \
  --impedance-target-linear-vel 0.02
```

这会真实运动；参数上限不是实际速度保证。此次未增大跟随增益、笛卡尔刚度或参考偏差，
也未放宽反馈超时保护。预览改为最高 30 Hz，不代表实际机械响应已验证。
若继续单独对比滤波延迟，可用 `--target-filter-alpha`、`--normal-filter-alpha`、
`--target-blend-alpha`、`--twist-smooth-alpha`；默认分别为 `0.34 / 0.24 / 0.82 / 0.72`，
与原 PBVS 默认值一致。取值范围 `(0, 1]`，越大越快接受新观测，也越容易传递噪声；
上面命令保持这些默认值，不同时改变滤波。也支持 `servo:` 下同名下划线 YAML 配置，
命令行优先。诊断日志元数据 `pbvs_filters` 记录实际使用值。

#### 2026-10-21：力矩限幅、参考饱和恢复与跳变目标重确认

伺服配置现在分开使用原工程的两项上限：笛卡尔任务力矩在叠加重力补偿前限制为
`±10 N·m`，加上重力及其他前馈项后的主机前馈力矩限制为 `±14 N·m`。
后端若配置了更低总上限，仍会继续限制；重力补偿并非无限制，固件 Kp/Kd 力矩也不包含
在该主机前馈限幅中。诊断元数据记录 `task_torque_limit_nm` 与
`total_feedforward_torque_limit_nm`。这修正了旧配置将两项上限都设成 10 的问题；
当前姿态若仍需要超过总上限的力矩，仍会饱和，不能保证仅凭此修改解决实际运动问题。

位置参考仍采用 15 mm 偏差上限。实测运动将旧参考带出上限后，新参考按原有速度限制
缓慢朝限幅内恢复，同时允许改变方向，避免因微小反馈变化而长期冻结；不会一步跳回实际 TCP。
零请求速度、丢失目标时仍保持既有位置参考，不根据未知手部位置盲目后退。

MediaPipe 的目标跳变仍先拒绝并保持。只有连续有效的新帧在至少 `0.40 s` 内保持在
固定锚点的 `2 cm / 15°` 范围内、至少有三帧，且重新计算的 TCP 目标误差不超过现有
`20 cm` 范围，才清空旧 PBVS 滤波／积分历史并重新接受目标。
慢速漂移也会越过固定锚点而重新计时，不能靠逐帧小位移通过稳定检查。
丢手会清除该确认过程，重复帧不增加计数；ArUco 沿用原跳变拒绝逻辑。
诊断中 `jump_reacquire` 标记 `accepted_stable_target` 或 `outside_tracking_envelope`。
重新接受后仍按原来的速度、关节偏差和位置偏差限制推进，不直接跳到新目标。

离线验证覆盖手从 25 cm 靠近至 15 cm 后的后退方向、参考由前向后反转，以及稳定跳变到
近处时的重确认；这只验证已观测目标的位置跟踪，不是全臂近人避碰能力。原运行命令不变，
该版本尚需真实机械响应验证。

#### 2026-10-21：保持低速，增加低速跟随助力

满助力门槛由 `0.08` 改为 `0.02 rad/s`。例如仍使用 `--follow-kp 4` 时，某轴参考速度为
`0.02 rad/s`，目标辅助增益从旧设置的 1 提高到 4；运动速度和 Kp 设置上限是独立的。
增加的助力按 `--follow-kp-rise-time 0.20` 逐步加上：从零升到设置上限约需 0.2 秒，
较小的目标增益会按同一上升速度更早达到。降低助力不受该上升限速拖延。

零参考速度撤去辅助；被推超前和安全 HOLD 立即释放。短暂丢手沿用 0.30 秒衰减，
重新识别后从已经衰减的助力继续缓慢加起，不恢复到丢手前的满额值。
长时间丢手或重新进入阻抗控制时清零辅助状态。
该修改仅处理低速辅助；仍需真机验证是否减轻停顿，摩擦补偿本次未改。

`config/pipeline.yaml` 的 `servo.follow_min_qdot` 与 `servo.follow_kp_rise_time` 已记录新设置，
命令行可覆盖。诊断元数据记录 `follow_min_qdot_rad_s` 与 `follow_kp_rise_time_s`，
实际每轴 Kp 仍见 `kp_command` 和 `control_cycles.jsonl`。
原独立视觉伺服脚本保留原设置。主流程中 `--follow-kp-rise-time 0` 可显式关闭增加助力的限速；
`--follow-min-qdot 0` 仍为旧的固定增益模式，零参考速度也不自动撤去助力，常规跟手采用上面的正值。

下面保留末端参考速度 1 cm/s、Kp 上限 4、关节参考偏差 0.03 rad；任务／前馈总力矩上限仍为
10／14 N·m，位置参考偏差仍为 15 mm：

```bash
bash scripts/run_grasp_safe.sh \
  --config config/pipeline.yaml \
  --phase servo --execute \
  --camera-width 640 --camera-height 480 --camera-fps 30 \
  --target-source mediapipe --servo-follow-only \
  --servo-preview --servo-diagnostics \
  --hand-depth-radius 8 \
  --delivery-distance 0.20 \
  --max-linear-vel 0.01 --max-joint-vel 0.08 \
  --follow-kp 4 --follow-min-qdot 0.02 \
  --follow-kp-rise-time 0.20 \
  --follow-max-error-rad 0.03 \
  --impedance-target-linear-vel 0.01
```

#### 2026-10-21：掌心居中与距离跟随，七轴共同调整位置和朝向

新增显式开关 `--hand-camera-center`，仅用于 `--target-source mediapipe`。
不开启时沿用上面的固定朝向跟随。开启后，把掌心相对图像几何中心的偏差、TCP 到掌心的
距离偏差放进同一个七轴速度求解，允许平移和转动同时发生；不会指定必须由手腕或大臂完成。
相机雅可比包含手眼旋转、相机相对 TCP 的位置偏移；不把夹爪轴当成镜头轴，不跟随手掌翻面。
画面中心采用当前分辨率的 `((width-1)/2, (height-1)/2)`，内参主点不必等于这个位置。

采用后端实际的软件关节边界，内侧再留 `0.02 rad` 规划余量；接近边界时逐渐压低向外速度，
允许向内返回。关节参考也限制在规划边界内。任务同时约束末端线速度、角速度及关节速度，
正常跟随的关节参考加速度默认不超过 `0.5 rad/s²`；安全保持优先于加速度平滑。
若约束无解、求解不收敛或相机几何无效，则保持并记录原因，不发送未验证的中间解。
这些是参考约束，不保证实际关节绝不越界；原内环限位停机、反馈超时及正常退出保护继续生效。

姿态参考按角速度连续积分，并保留原来的 `5°` 相对实际姿态误差上限；该上限不是相对启动
朝向的总转角上限。位置参考仍限制在实际 TCP 的 `15 mm` 内。丢手时不再推进参考，不主动搜索。
`--hand-center-gain 2` 控制图像/距离误差收敛速度；
`--hand-center-max-joint-accel 0.5` 控制关节参考加速度。
不能与 `--visual-orientation` 混用；为了让检测到的掌心真正居中，`--hand-offset-base` 必须为零。

下面保留本次使用的末端线速度 `3 cm/s`、关节速度 `0.15 rad/s`，新增转动上限 `0.15 rad/s`：

```bash
bash scripts/run_grasp_safe.sh \
  --config config/pipeline.yaml \
  --phase servo --execute \
  --camera-width 640 --camera-height 480 --camera-fps 30 \
  --target-source mediapipe --servo-follow-only \
  --hand-camera-center \
  --servo-preview --servo-diagnostics \
  --hand-depth-radius 8 \
  --delivery-distance 0.20 \
  --max-linear-vel 0.03 --max-joint-vel 0.15 \
  --max-angular-vel 0.15 \
  --follow-kp 4 --follow-min-qdot 0.02 \
  --follow-kp-rise-time 0.20 \
  --follow-max-error-rad 0.03 \
  --impedance-target-linear-vel 0.03 \
  --impedance-target-angular-vel 0.15
```

该模式尚未经过真机验证，上面带 `--execute` 会真实运动。
预览中的黄色十字为居中目标，连线连接检测到的掌心。
日志新增 `camera_center`：`pixel_error` 为当前关节姿态下投影的滤波后掌心偏差，
不是未经滤波的 MediaPipe 像素；另含距离误差、各轴距软件边界的余量、受约束关节及求解状态。
`command_angular_rad_s` 记录实际安装的姿态参考角速度。终端同步显示居中误差与求解状态。

本轮新增专项 35 项通过，扩大回归 626 项通过、2 项既有失败（见下方）。
另对 9 组已记录的关节/掌心状态重新计算指令，参考速度和位置边界检查通过；使用同相机
另一次采集的内参做运动学核对，不是历史图像逐帧回放。5 段各 15 秒的移动目标运动学仿真，
共 2250 步未出现求解保持。本机完整控制更新耗时中位约 0.84 ms、99 分位约 1.6 ms，
不包含图像检测及设备交互。这些验证不模拟摩擦、动力学和真实相机延迟，未连接真机。

低速助力与平滑增加修改后的扩大离线回归：591 项通过，2 项此前已存在的失败仍保留：
`tests_servo/test_execute_impedance_engage.py::test_visual_impedance_passes_pbvs_twist_to_cartesian_damper`
（独立脚本将速度裁到参考上限，而测试期望原始速度）；
`tests_servo/test_impedance_law_units.py::test_moving_target_disables_discontinuous_coulomb_kick`
（移动目标仍存在原有摩擦补偿项，而测试期望零力矩）。两项涉及的实现本轮未修改，
该结果不代表全部回归通过。

`--servo-diagnostics` 是可选排查开关，不改运动参数。终端每秒输出 `[servo][diag]`，
带时间戳的完整记录保存在 `logs/servo_diagnostics/日期时间_p进程号/samples.jsonl`。
不带 `--execute` 的 MediaPipe 诊断会额外在同目录 `rgbd/` 留存同帧 RGB、对齐深度、
内参、关键点和失败原因：间隔至少 5 秒，最多 12 组。`.npz` 保存原精度数据，
`.png` 左侧为 RGB、右侧为深度；黑色表示无深度，品红色表示有深度但超出筛选范围，
绿圈对应五个关键点采样窗口。`samples.jsonl` 的 `depth_health` 区分全图与六个
采样区域的缺失、过近、过远像素及深度分位数，`perception` 记录当帧检测原因。
图像落盘与统计在后台诊断线程完成；带 `--execute` 时不启用该图像采样。
dry-run 的关节角为模拟零值，保存的基座坐标与关节角不能用于真实手眼标定验证。
同目录的 `control_cycles.jsonl` 逐周期记录最终发给 SDK 的关节目标、Kp/Kd、
参考误差、前馈力矩和退出过渡；若抓取已开启内环日志，则复用原日志，路径打印在终端。
`kp_12bit_estimate` 是主机对 V111 编码量化的估计，不是设备回执；前馈力矩不包含固件 PD。
`actual_mm` 是实际 TCP；`eq_mm` 是内部控制目标；`goal_mm` 是视觉期望位置。
`move_mm / eq_move_mm` 是相邻两次报告的净位移；`cycles+ / inner_hz` 表示阻抗内环推进情况。
第一条报告还没有前次样本，位移／周期增量显示 `--` 或 `None`；从第二条开始判断。
先让手保持稳定，窗口显示 `TRACKING` 后观察约 5 秒，再缓慢横向移动几厘米；
若出现明显异常立即用正常退出停止测试，不加大增益或放宽保护。
把连续 10～20 秒终端输出和日志路径发来，区分“识别暂停／目标未推进／目标推进但实际没动”。
日志异步写入，有界队列满时丢诊断；诊断失败不会调用控制接口，也不改变退出保持逻辑。

`--servo-diagnostics` 还会每秒读取一次 SDK 已接收的设备状态，不发送 CAN 查询或控制命令。
`device_feedback` 保存设备实际 `ctrl_mode / mode_feedback / arm_status / err_code`、
七轴使能、驱动错误、堵转／碰撞状态及各自时间戳；缺失字段为未知。
终端 `device_mode` 是设备反馈模式（当前 V111 SDK 中 `6=MIT`、`1=move_j`），
不能用主机的 `inner=impedance` 代替。
`enabled_at_sample` 只统计采样时反馈在 0.5 秒内且使能为真的轴，
需同时看 `status_age_ms`、`device_sample_age_ms` 和 JSON 中各轴时间戳；不是持续使能保证。
2026-10-20 `21-49-00` 日志中，视觉更新连续，实际 TCP 三轴变化均不到 0.1 mm，
并发生退出设备保持未确认。该旧日志未保存上述设备状态，不能据此确定掉使能、
控制模式未切换或机械阻力中的哪一项，也不能将提高参考速度视为解决办法。
新退出日志增加 `device_hold_ack_sample / device_hold_unconfirmed`，保存交接时的
实际模式、使能、反馈时效和停稳判断；失败时异常信息也附最后设备状态，不自动重使能。

正常 Ctrl+C、窗口退出共用抓取的停稳协调器：冻结目标，在原 MIT 内环中用约 `0.4 s`
从伺服跟随平滑过渡到固定保持，随后统一检查每轴新反馈、速度和漂移。
位置与增益采用力矩连续的过渡，任务力矩逐渐撤去，重力补偿持续；最终仍确认
`move_j` 设备保持后退出并保留使能。切换后保持同一连接观察至少 `0.5 s`，继续检查新反馈、
停稳及漂移；`control_cycles.jsonl` 中的 `device_hold_sample` 记录这段 q/dq/力矩和各轴时间戳，
`device_hold_confirmed` 汇总最大漂移／速度。它用于定位切换抖动，不代表已消除抖动，也不放宽保护。

`--servo-preview` 同时打开伺服相机窗口：骨架／掌心圈、相机与基坐标 XYZ、深度有效点数、
`TRACKING / LOST / HOLD / ACQUIRING` 和具体原因。窗口按 `Q`、`Esc` 或关闭窗口走正常退出保持，
不是急停。预览复用本次识别结果，不额外打开相机；单独显示进程只接收图像／状态，
不连接 CAN。显示队列只保留最新待发送帧，子进程持续读取图像、不等待 GUI 刷新；
预览发送上限为 30 Hz，不改变识别帧率。画面／状态过期时标红 `STALE DISPLAY`。
无桌面环境去掉此参数；运行时预览崩溃会退出伺服并走受控清理，不能靠旧画面判断机械臂状态。

独立预览掌心 XYZ（只相机，不连接 CAN）：

```bash
python scripts/test_realsense_hand.py --serial 243622072771 --width 1280 --height 720 --fps 30
```

### 全流程统一 Ctrl+C 退出

`main.py`／`run_grasp_safe.sh` 的抓取、NBV、多视角采集、保持、抬升、人手伺服共用以下规则，原命令不用改：

1. 终止任务，锁存停止；不自动返程、不额外松爪，不执行下一阶段。
2. MIT 正在运行时，先固定当前关节目标、速度目标置零，用新反馈确认停稳。
3. 停止并确认 MIT 发送线程退出后，用**同一 CAN 连接**交接设备零速当前位置保持，
   核对模式与七轴使能，再断开。该交接只在退出时发生，任务中不切换控制方式。
4. `move_j` 路径沿用轨迹普通终止／当前位置保持；已经静止时不会额外执行返程。

清理期间再次 Ctrl+C 不打断保持交接，清理结束时会提示已延后处理；窗口 `Q`／`Esc` 也走正常退出。
已有急停不会自动复位／重新使能；失联、限位、无法确认停稳／线程退出等故障仍可能触发保护。
强杀进程、断电以及设备自身保护不属于正常 Ctrl+C 的保证范围。
夹爪不会因退出额外打开，但已发出的夹爪指令可能完成；未做负载防坠认证。

成功交接日志：`[shutdown] device position hold confirmed; motors left enabled; no disable/e-stop sent`
（独立伺服的对应日志以 `Servo exit:` 开头）。若报保持未确认，不能当作已安全保持。

### 伺服启动失败时的处理（2026-10-20）

- 先只读检查七轴反馈、配置限位和启动余量，通过后才切模式／使能。
- 接管前检查不通过：只退出接收连接，保留设备原状态，不发使能、掉使能或急停。
  日志包含 `rejected before control takeover`；不是已进入伺服后又退出。
- 已开始接管后失败：先停稳并确认设备位置保持，再退出；不调用掉使能清理。
  若反馈不可信或保持无法确认，仍走故障保护，不假报保持成功；已确认保持后的断连错误不额外急停。
- 本次未调整限位或 Kp/Kd。J4 若仍超过配置上限，会继续拒绝启动，不能靠删参数绕过。

### 异常停机：第一批 R1/R2 分流（2026-10-19）

- 已确认静止的任务失败不额外发普通关节运动；执行异常、碰撞检查计算失败先请求受控保持，不能跳过检查继续走。
- MIT 进入软件限位缓冲区时，使用现有位置保持增益锁住旧目标，保留重力补偿；
  需连续新反馈确认停稳。没有可用保持配置、2 秒未确认、保持丢失仍进入故障保护。
- 人手伺服的阻抗配置／更新失败先确认保持；关节反馈失信、失控和硬限位保护仍保留。
- 停机与日志清理分离：确认设备位置保持之后再收尾日志；日志异常不作为急停理由。
- `move_j` 普通停机也检查各轴反馈时间戳，重复缓存和历史“已停稳”标志不能冒充新确认。
- 此批 R2 仍是**终止当前轮任务并保持后退出**，不会自动重试、返程、松爪、复位或续发旧轨迹。
  沿用现有固定位置保持停车，尚未新增减速度受限刹车轨迹或程序内人工授权重规划。
- 统一停止入口的分级、时间戳与确认结果记录在实验结果的 `fault_events`；
  MIT 软件缓冲停止原因也进入控制周期日志（启用 `--mit-q-log` 时）。

原命令无需新增参数。这里只改变异常处理，不放宽反馈、间隙、硬限位阈值，不更改正常运行 Kp/Kd。
仍需离线回归与空载隔离环境验收；不能把“急停次数减少”当作近人作业已经安全。

目标恢复修正：

- 人手持续丢失超过 0.45 秒后清理旧目标，连续 3 张稳定新 RGB-D 帧后恢复跟踪；
  保留原跳变保护、限速和跟踪距离。重复帧不算新检测，识别失败时停止目标速度前馈。
- 终端 `lost reason=` 区分 `no_hand_2d`（未检测到手）、`invalid_palm_depth`（深度不足）、
  `camera_frame_stale`（帧过期）；`[servo][hold]` 给出限位／跳变等拦截原因，
  `[servo][tracking]` 显示掌心基坐标和位置误差。

以上为离线验证，尚未验证真机退出后的持续保持。先空载、远离人手检查正常退出；
若退出报保持未确认，不代表已安全停止，勿徒手进入机械臂工作范围。

离线全量回归：858 项通过，仍有 6 个原有失败，其中 2 个涉及伺服前馈／摩擦补偿；预览窗口／真机尚未实测，
不能据此认定近人跟踪已安全。首次运动应空载、空旷，并以固定替代目标先核对方向／保持行为。

沿用原来的抓取／多视角命令接人手跟踪：把 `--phase grasp` 改成 `--phase all`，加上：

```bash
--enable-servo-delivery --target-source mediapipe --servo-follow-only --servo-preview
```

完整流程也可加 `--servo-preview`，仅进入伺服后打开；不要同时另跑相机预览程序占用设备。

只有抓取和抬升成功才进入伺服；`--servo-follow-only` 不会自动释放物体。

### “阻力大”与阻尼不是一回事

- 目标丢失时保留原位置目标，位置弹簧仍在，手推会有回拉感；笛卡尔平移刚度当前为 `200 N/m`。
- 当前笛卡尔平移阻尼 `B=5 N·s/m`（`--cart-b-scale` 默认 `1`）；固件关节 `kd` 另外为
  `[1, 1, 1, 0.5, 0.3, 0.2, 0.2]`，两层不是同一个参数。
- 命令限制为 `0.01 m/s`、`0.03 rad/s`（约 `1.72°/s`），所以跟随会慢，但限速不等于阻尼。

目前没有数据证明就是阻尼过大，本次没有修改刚度／阻尼／重力与摩擦补偿参数，也不建议直接降为零。

### MIT 静止保持的短暂超时等待

统一 MIT 抓取会话默认 `--mit-hold-recovery-timeout-s 0.5`，单位秒，可设 `0` 关闭，最大 `1`。
仅已停稳的固定 `hold` 可等待反馈恢复：预算包含已经发生的中断，
不发送过期结果、不改变保持目标；七轴反馈重新通过原年龄／一致性检查、
连续 3 次新帧确认低速且相对中断前漂移不超过 0.5°后，重算原位置保持。
运动轨迹／主动阻抗伺服不使用此宽限；持续失联、异常漂移和限位保护仍可能急停。
独立 `--phase servo` 创建自己的后端，不应用这个抓取会话参数。
新记录位于 `mit_control_cycles.jsonl` 的 `type=hold_recovery`。
这不是实时性修复或设备断流行为认证；未进行真机验收，不能保证中断期间持续保持使能。

## 4. 只执行抓取，不执行视觉伺服

```bash
sudo ip link set can0 up type can bitrate 1000000

bash scripts/run_grasp_safe.sh \
  --config config/pipeline.yaml \
  --phase grasp \
  --execute \
  --target-prompt \
  --yolo-weights models/yolo26m-seg.pt
```

不要添加`--enable-servo-delivery`。

固定抓取`cup`、自动分割并强制执行中心视角 + 2个NBV侧视角：

```bash
bash scripts/run_grasp_safe.sh \
  --config config/pipeline.yaml \
  --phase grasp \
  --execute \
  --target-class cup \
  --active-view \
  --no-active-multiview-gate \
  --active-side-view-count 2 \
  --preview-cloud \
  --yolo-weights models/yolo26m-seg.pt
```

这里不要再加`--target-prompt`，否则程序会询问`object name / CLICK / POINT`。初始画面使用`--yolo-conf`，换视角后会自动用较低的`--active-yolo-conf 0.05`重识别同一个`cup`；若重识别失败会放弃该次主动融合，不再把目标周围的球形区域误当成分割结果。

## 5. 打开相机可视化

查看RGB-D、YOLO分割和目标点云：

```bash
python scripts/test_D435i.py \
  --weights models/yolo26m-seg.pt \
  --target-class 你的类别名
```

查看真人手掌心识别：

```bash
python scripts/test_realsense_hand.py \
  --serial 243622072771 \
  --width 640 \
  --height 480 \
  --fps 30
```

按`Q`或`Esc`退出可视化。

### 空夹爪：两指间距连续控制开口

先退出原跟手／抓取程序，停止其他程序发运动指令，把机械臂置于已使能、静止的位置保持。
下面是独立的**空夹爪**测试，只开彩色流，不需要手眼标定或有效深度，也不启动 MIT 控制器。
默认只写夹爪目标；下方命令显式加了 `--prepare-can-hold`，必要时会切换主控来源、设置
主控速度、发送当前关节位置保持。它不使能／失能或复位，不移动到预设姿态。
机械臂未处于静止位置保持或反馈无效时，不发送夹爪指令。不要用于持杯交接。

2026-10-22 17:36这次记录中，手势学习完成并调用SDK发送89次，夹爪实测仍为84.959毫米；
主控始终 `ctrl_mode=3`。随后只读CAN监听也读到 `0x2A1=0300010001000000`，未发送任何报文。
[SDK控制来源定义](https://github.com/agilexrobotics/pyAgxArm/blob/master/pyAgxArm/protocols/can_protocol/drivers/nero/default/driver.py)
中3是以太网，1是CAN；原抓取程序有控制初始化，独立脚本此前遗漏了这一环。
这支持优先补齐控制权交接；尚未真机验证它是否完全解决不响应，不能仅凭主控模式推断夹爪硬件故障。

`--prepare-can-hold` 只允许从已使能、静止的以太网J保持交接：先清旧轨迹，第一条选择CAN的
模式报文速度设0，再清队列并发送当前关节角作为保持目标。收到发令后新鲜的CAN/J反馈、
新目标到位标志且位移检查通过后，才把主控速度设10%，继续观察保持，再允许夹爪控制。
准备失败不自动重试；进入切换后失败会尝试设回速度0，停发夹爪，保留使能。已经处于CAN时
跳过交接。运行中丢失CAN控制权会锁定故障，不反复争抢控制。
退出保留当前控制来源／保持，不恢复旧以太网目标。此选项涉及主控交接，未做真机切换验证。

普通跟随前的静止判断使用实际关节反馈稳定窗口，不要求SDK的 `motion_status` 到位标志为0；
显式交接中则需要这个标志确认刚发送的当前位置保持目标，之后才允许恢复主控速度。
已观察到正常、已使能、实际静止的机械臂仍报告 `motion_status=1`。故障码、位置模式、
七关节使能／反馈新鲜度和手臂位移检查继续有效。终端每秒打印 `FOLLOW/WAIT/FAULT`、具体原因、
夹爪实测开口和机械臂状态数值，同步写入日志，避免只在预览画面中提示拦截原因。

```bash
conda activate grasp_yolo26
python scripts/test_pinch_gripper.py \
  --config config/pipeline.yaml \
  --camera-width 640 --camera-height 480 --camera-fps 30 \
  --open-seconds 5 --close-seconds 2 \
  --force 0.50 \
  --max-lead-mm 5 \
  --prepare-can-hold \
  --execute
```

去掉 `--execute` 时只做相机和比例预览，不连接 CAN，也不执行保持交接。
不加 `--prepare-can-hold` 时不发送任何手臂命令；若主控仍是以太网控制，会明确等待并停发夹爪。

启动后只需**拇指食指捏合停约1秒，再充分张开停约1秒**，不需要 `C/O/A` 或其他
标定／启动按键。连续3帧识别同一只手后锁定左右身份，自动记录两个稳定端点；学习期间
不发送夹爪目标，两个端点记录完成后自动连续跟随。请让手掌尽量正对相机，并确实展示
你希望对应全开的最大张幅：仅观察某个中间开度，程序无法知道你还能再张开多少。

几何输入仍是“两指间距÷掌宽”，但不再错误地把0和1当作每个人的捏合、全开端点。
捏合有指尖关键点的中心偏移，也会被记录为0%；充分张开才记作100%。两个端点各保留
3%的抖动死区，中间线性映射到 `0–85 mm`；50%对应42.5毫米。学习完成后固定尺度，
不会因为这次只张开一点就重新把它当作100%，丢手恢复也不用重新学习。每次重启会重新学习。
如需复用日志中的端点，可同时传入 `--pinch-closed-ratio` 和 `--pinch-open-ratio` 跳过学习。

画面显示比例目标、经过限速的指令和实测开口。5秒／2秒是全测试行程的指令渐变时间，
部分行程按相同速度变化；不能当作已验证的机械响应时间。可用
`--min-width-mm`／`--max-width-mm` 缩小测试行程。

`Q`／`Esc`／Ctrl+C退出。丢手、多手、左右标签改变、画面过期或设备反馈异常都会暂停，
反馈有效且同一只手重新连续识别3帧后自动继续，从当前实测开口重新限速，避免跳变；
机械臂位置检查保留最初的参考，不会因丢手恢复而隐藏手臂位移。本测试为空载全行程，
不使用持杯方案的“最多额外闭合1毫米”限制；两者丢手都停发目标，不自动回夹。

此前实机日志中实测始终为84.959毫米，2毫米领先上限把目标卡在82.959毫米；因此旧画面
的 `ACTIVE` 不能证明夹爪正在跟随。现在空载测试默认最多领先5毫米（与现有慢速闭合的
默认5毫米步长一致，CLI不允许更大），仍受5秒／2秒限速。目标不变时约每0.1秒重发；
目标偏离实测超过1毫米，持续1秒没有至少0.2毫米的同方向进展，就显示 `FAULT / NO RESPONSE`
并停止继续发令。故障锁定后丢手重入不会自动重试，需退出检查并重启。

力参数改为0.50，维持项目现有0.50上限；[厂商ROS接口](https://github.com/agilexrobotics/agx_arm_ros/blob/ros2/README_EN.md)
列出的范围从0.5开始，但这不能证明直接调用Python SDK时旧0.30一定被固件拒绝。
5毫米领先量用于避免旧2毫米限制长期卡住，并非已经测得硬件死区；以上修改未操作真机验证。

暂停、故障、退出后，最后一个至多领先5毫米的小目标仍可能完成，不能保证机械瞬间刹停。
退出不额外开合夹爪或切换机械臂使能。记录保存在 `logs/pinch_gripper/<时间>/samples.jsonl`，
包括学习到的端点、每次成功调用SDK发送的目标、发送次数、实测开口／力、驱动使能和
反馈新鲜度；若收到 `0x159` 控制帧，也单独记录，其含义是观测到控制报文，不能当成
夹爪接受／完成动作的确认。

### 双手递杯：比例开合已接入主伺服

通过显式开关 `--gesture-handover` 启用，必须配合 `--hand-camera-center`，
不与 `--servo-follow-only` 同时使用。
按已确认的单人双手操作设计；只锁定左右手身份，不识别人员身份。
原来的跟随测试和旧递送入口保持原行为；这个开关使用新的双手交接流程：

1. 先只伸出接杯手。连续识别3帧后锁定其左右身份；跟随这只手，另一只手出现也不会抢走目标。
2. **跟随目标仍为5厘米**；准备交接的距离独立设为TCP到掌心**不超过15厘米**
   （`--gesture-ready-distance 0.15`）。掌心距画面中心小于12像素，并且掌心在机器人
   基座坐标里**连续停稳3秒**，就进入交接待命，不必等接杯手先消失。单纯追着移动的手
   保持居中不算停稳。停稳初始容差为1厘米：计时期间三维位置包围盒的对角线不超过10毫米，
   可用 `--gesture-still-tolerance-mm` 调整；这是待实测验证的初始值。计时用实际经过的秒数，
   通过 `--gesture-stable-seconds` 设置，不受旧 `--handover-stable-frames` 控制。
   **待命前**丢手、深度失效、不居中、距离不满足或有效帧间隔超过0.25秒，都重新计时。
   **松夹前**，A手新的有效三维位置偏离待命位置超过上述1厘米容差、不再居中或离开准备距离，
   都撤销松夹授权并恢复跟随。保留已学端点，但作废旧张开请求；A重新稳定3秒后需要B重新捏合确认。
   单纯遮挡或深度失效不等同于撤回；二维仍可见但明显偏离记忆投影时也会撤销授权，等待新深度。
   这里量的是TCP到掌心；停稳仅用于进入待命，不能证明杯子已经被人接稳。
3. 进入待命时，把阻抗目标固定到当前实际姿态，机械臂原地保持。另一只手先捏合约1秒、
   充分张开约1秒以学习范围；**学习期间不动夹爪**。
4. 出现 `GESTURE_ARM` 后，再捏合并稳定约0.4秒，进入 `GESTURE_READY`；**这时仍不发送夹爪目标**，
   保持原抓取夹持指令。B随后主动张开，且A仍满足交接条件、机械臂已停稳，才开始比例开合。
   同一帧A移动与B张开同时发生时，A优先，禁止松夹。之后不用按键。
   0%对应原夹杯开口附近，最多额外收紧1毫米；100%对应允许的全开位置，默认85毫米。
   **持杯时0%不是把夹爪闭到0毫米。** 中间开度按这个允许区间连续映射。
5. **第一条松夹目标发送前即锁定 `release_started`**，本轮不再恢复A手跟随；不等到全开才停止跟随。
   不存在“部分松开后继续追手”的路径。控制手丢失、出现两个同侧可信候选、身份不明或画面过期时，
   **停发夹爪目标，保持当前开口，
   不自动回夹**。无论开到多少都生效，不需要先确认全开。同一只控制手连续识别3帧且机械臂
   静止后，已开始松夹的流程从当前实测开口恢复比例控制；未开始松夹时则要求重新捏合确认。
   松夹后如A相对待命位置持续偏移超过5厘米、有效测量持续0.5秒，进入 `HANDOVER_CANCELLED`，
   夹爪保持当前开口、机械臂原地保持，不自动回夹或追赶；丢手本身不作为撤回证据。
6. 两指给出全开目标，且3份不同时间戳的夹爪反馈确认到达全开附近（1毫米容差），本次交接结束。
   此后停止手势控制，也不自动撤退；机械臂继续原地保持，直到用户退出。

2026-10-25增加有限位置记忆，仅作用于上述手势交接的**待命前、尚未松夹**阶段：

- A手此前连续至少0.3秒有可靠三维测量，位置包围盒变化不超过1厘米；现在只有深度失效，
  RGB中同一只A手仍可信且与记忆点投影相符，才允许一次有限接近。
- 记忆保存最后实测的基座坐标和采集时间。最长0.3秒、最多额外3毫米、线速度最高1厘米/秒，
  任一预算达到就停止；3毫米同时检查新参考累计路径和实测TCP累计路径，不能重复使用旧坐标刷新预算。
  开始时重设当前实际姿态的阻抗保持，避免把此前位置领先量也带进记忆接近。
- A手二维也消失、身份不明、图像与记忆不符、相机过期、预算耗尽或松夹已开始时，不使用记忆继续运动。
  必须取得新的可信三维测量才可能重新获得预算。安全保护和求解检查照常生效。
- 记忆不刷新“最后有效三维测量”时间、不累计稳定3秒、不产生松夹授权。预览显示 `MEMORY_APPROACH`，
  诊断独立记录 `hand_memory`；无效深度仍标为无效，不伪装成实测。可用 `--no-hand-memory-approach` 关闭。
- 上述限制是软件指令与反馈检查，不保证物理瞬间停止，也不能用来跨越任意大的相机盲区。

以上已接入主伺服及抓取成功后的同一入口；力反馈仍只记录，不用于判断接稳或触发松夹。

开合参考限速仍为全允许行程张开5秒、闭合2秒，目标最多领先实测5毫米；这些是指令约束，
不是实机响应保证。丢手后不再发送新目标，但最后已发送的小目标仍可能完成，不能保证瞬间刹停。
额外闭合下限从启动时新鲜的**实测夹杯开口**固定计算，不跟着后续开度漂移。
抓取切到伺服时保留抓取阶段最后的实际夹持指令，不用更宽的测量值悄悄替换承载指令。
夹爪力参数不超过原抓取设置和0.50上限。关节力矩继续通过 `--servo-diagnostics` 记录，
仅作为观察量，不触发松夹。

图中这类很小的假手，现已先检查候选关键点是否有效、掌宽是否至少15像素，再计数。
空夹爪脚本也复用这个筛选；两只几何可信的手不会按“谁大就选谁”处理。
空夹爪预览中被剔除的候选显示为灰色，并标记 `ignored`，不参与比例计算。
这能减少小候选误检造成的暂停，**不能保证剔除所有看似合理的假骨架或左右标签错误**，
没有重训练模型。诊断记录候选数、剔除原因、手的角色、交接阶段、比例和夹爪开口。
画面及日志同时显示停稳计时、待命是否已锁定和未满足条件的原因；尚无有效三维目标时也显示状态。

2026-10-23远距离接近保护已调整（仅 `--hand-camera-center` 人手跟随）：

- 远处但连续识别、深度有效的人手，允许按当前速度逐步接近；不会再因为距最终目标
  超过20厘米、持续1秒就同时停止接近和居中。最终交接距离仍是下面命令设置的5厘米。
- 跟踪误差改查**已经安装的阻抗位置／关节参考**与实测反馈，原来的15毫米位置领先、
  0.03弧度关节领先及速度、力矩、目标跳变、关节限位保护继续生效。
- 即使手在模型估算的最远范围以外，也允许在速度、关节与参考领先限制内逐步靠近；
  不会因为最终够不到手就提前停在原处。URDF最外侧边界球只记录诊断，
  `outside_reach_bound=true` 不触发暂停；稳定跳变目标的重新接收也遵循这一规则。
- 因姿态、关节限制或实际跟不上而无法继续取得进展时暂停。连续3秒，实测关节运动对
  “距离＋居中”任务的有效改善不足等效2毫米，就暂停。小幅居中时门槛降为剩余任务
  误差的20%（取两者较小值）；这里的等效毫米包含图像任务权重，不是TCP路径长度。
  手自己靠近、深度抖动或主机参考向前走，不算机械臂取得进展。
- 暂停后同一位置不会定时重试；手从暂停位置移动至少2厘米后才允许重新尝试，
  仍需通过已有的有效帧／目标跳变确认。日志及预览给出
  `hand_approach_stalled_hold` 或 `reference_tracking_hold`；触及关节保护范围也会暂停。
  这些暂停表示软件保护条件已触发，不证明机械臂已达到全部姿态中的最远位置。
  详细数值在诊断的
  `camera_center.execution_safety` 中。

以上停滞时间和进展门槛是首版软件参数，已做离线验证，未验证真机响应；本次没有调整
相机配置，持瓶遮挡和最大深度分辨率的近距离盲区仍需单独处理。

下面用于**已经夹住杯子后，单独运行跟随与手势交接**，沿用已有跟随速度；空爪对比跟随
也可沿用这条命令（空爪开口附近仅有很小的手势开合余量）：

```bash
bash scripts/run_grasp_safe.sh \
  --config config/pipeline.yaml \
  --phase servo --execute \
  --camera-width 1920 --camera-height 1080 --camera-fps 30 \
  --camera-depth-width 1280 --camera-depth-height 720 --camera-depth-fps 30 \
  --camera-depth-filters all \
  --target-source mediapipe --gesture-handover \
  --hand-camera-center \
  --servo-preview --servo-diagnostics \
  --hand-depth-radius 8 \
  --delivery-distance 0.05 \
  --gesture-ready-distance 0.15 --hand-memory-approach \
  --gesture-stable-seconds 3 --gesture-still-tolerance-mm 10 \
  --max-linear-vel 0.03 --max-joint-vel 0.15 \
  --max-angular-vel 0.15 \
  --follow-kp 4 --follow-min-qdot 0.02 \
  --follow-kp-rise-time 0.20 \
  --follow-max-error-rad 0.03 \
  --impedance-target-linear-vel 0.03 \
  --impedance-target-angular-vel 0.15 \
  --gesture-extra-close-mm 1 \
  --gesture-open-seconds 5 --gesture-close-seconds 2
```

主流程已经由MIT后端持有控制权，**这里不加空夹爪脚本的 `--prepare-can-hold`**，不会在MIT运行中
切回J模式或新建第二个夹爪CAN连接。带 `--execute` 会真实跟随和开合；本次合入只做了离线验证，
完整持杯交接尚未实测。

如需只看相机与状态流程，去掉 `--execute` 并显式加 `--gesture-preview-grip-width-mm 40`
（40只是例子，表示模拟起始开口，不是读到的硬件数据）。预览不连接CAN；真机执行时禁止
这个模拟参数，必须从新鲜反馈读取夹持开口。抓取成功后自动接入时，在原 `--phase all`
命令中加 `--enable-servo-delivery --gesture-handover --delivery-distance 0.05`，并移除
`--servo-follow-only`；其他抓取参数沿用原命令。

## 6. D435系列最大分辨率与深度去噪

2026-10-22只读枚举确认：序列号`243622072771`实际报告型号为 **D435IF**，支持彩色
`1920×1080@30`、原生深度`1280×720@30`。`config/pipeline.yaml`现已设为这两路最大模式，
不能把深度也设成1920×1080；输出深度对齐到1080p彩色图，不等于原生深度分辨率提升到1080p。

SDK主流程的空间去噪、时间去噪和独立补洞已全部开启，保留`High Density`、红外发射器和
激光功率360。滤波链为：深度对齐→转视差→空间滤波→时间滤波→转回深度→补洞。
空间滤波使用2次迭代、alpha=0.5、delta=20、4像素补洞；时间滤波alpha=0.4、delta=20，
只允许近期有效历史，不使用无限保留；独立补洞使用邻近较近值。
**降采样保持关闭**，因为它会降低深度分辨率；这些选项的含义见
[SDK滤波说明](https://github.com/realsenseai/librealsense/blob/master/doc/post-processing-filters.md)。

当前配置：

```yaml
wrist_camera:
  serial: "243622072771"
  width: 1920
  height: 1080
  fps: 30
  depth_width: 1280
  depth_height: 720
  depth_fps: 30
  depth_filters: all
  depth_preset: high_density
  emitter_enabled: true
  laser_power: 360
```

**命令行会覆盖YAML**。旧命令里的`--camera-width 640 --camera-height 480`要删除或改成下面的参数，
其余跟随/交接参数沿用本文件上一节的完整命令：

```bash
--camera-width 1920 --camera-height 1080 --camera-fps 30 \
--camera-depth-width 1280 --camera-depth-height 720 --camera-depth-fps 30 \
--camera-depth-filters all
```

`--camera-depth-filters off`可关闭后处理。未指定独立深度宽高/帧率时，SDK保持兼容，沿用彩色参数。
不读取本配置的独立旧测试脚本仍使用其自身参数。启动日志应包含：

```text
[camera] visual_preset=... (high_density)
[camera] emitter_enabled=1
[camera] laser_power=360/360
[camera] RGB=1920x1080@30; depth=1280x720@30
[camera] depth_filters=all; decimation=off; raw aligned depth retained when filtering
```

已只启动相机读取12帧，确认两路最大模式同时出图、滤波链可运行，整帧读取及处理约38毫秒；
这是相机测试，不包含MediaPipe、机械臂控制和预览负载，不能当作端到端30fps的保证。
滤波可能降低实时性；采集侧丢弃排队旧帧，避免积压后逐帧追赶，帧龄包含对齐及滤波耗时。

随后`2026-10-22_21-12-06`真实执行在启动阶段出现关节反馈103.2毫秒未更新，触发100毫秒
保护。之前的单相机验证不足以验证与控制并行运行；该次测试**没有证明最大画质组合可稳定上机**。
合成帧测试也发现SDK滤波会阻塞同进程Python接收线程。全滤波模式现自动使用独立相机进程，
通过两个有锁共享图像缓冲区传递最新RGB/深度，避免图像大包序列化和SDK滤波占用控制进程。
相机子进程不继承机械臂连接，并禁止打开CAN；异常会向父进程报告，不自动重启或复位机械臂。
独立伺服入口先完成首次人手推理预热，再创建机械臂后端。控制的100毫秒反馈保护保持原值。
以上为隔离与启动顺序修复，仍需区分不连接CAN的并行测试和真实控制验证。
修复后已进行50帧“真实相机＋MediaPipe＋预览数据序列化＋模拟接收/控制线程”并行检查：
模拟接收线程最大间隔4.99毫秒，模拟控制线程最大间隔8.71毫秒，模拟反馈最大年龄3.20毫秒；
从相机收帧到推理完成的帧龄中位数66.81毫秒、P95为76.13毫秒。该检查没有连接CAN，
没有发送机械臂/夹爪命令，不能替代真实CAN负载及设备反馈验证。

补洞和时间历史可能产生当前帧未实测的值。相机保留同帧未滤波的对齐深度，手部伺服只使用
当前原始深度有效且在允许距离内的位置上的滤波结果，**不会仅凭补洞把原先0/5变成可运动目标**。
抓取采集时额外保存`depth_raw_m.npy`；纯相机伺服诊断的NPZ同时保留原始、滤波和实际用于跟踪的
深度，便于检查“确实没读数”还是“被范围过滤”。相机验证没有证明此前的手部深度故障已解决。

激光功率设置后，设备预设名称可能显示为`Custom`，表示在`High Density`基础上覆盖了设置。

## 7. 夹爪接近阶段的距离阈值

手眼标定不包含默认桌面高度。桌面高度是抓取运行参数`TABLE_Z`，与手眼标定JSON无关。

本次日志是：

```text
source=environment
phase=target_approach
gripper_finger_+1 clearance=3.5mm, required >= 5.0mm
```

因此这次不是`TABLE_Z`触发，而是最终抓取接近阶段的“手指—环境点云”阈值触发。

### 已修复：接近阶段容差参数实际生效

`real_nero_grasp/collision.py` 已接入 `motion_runtime_target_approach_finger_clearance_slack`。按安全脚本原有配置，规划仍要求 5 mm；只有最终接近时打开夹指对环境的运行阈值为 `5 mm - 2 mm = 3 mm`。本体、相机、连杆和其他阶段不使用这项接近容差，原有自由运动手指容差也没有改变。

这不是关闭碰撞保护：小于 3 mm 的接近夹指间隙仍会按现有连续采样规则触发停止。执行前新增 `[motion][clearance-policy]` 日志，方便核对生效值。17:05 的触发记录由原 4.83/5 mm 改为 4.83/3 mm；只证明保存的记录在新规则下通过，不证明未来运动或实物间隙。

`--grasp-table-reference-z 0` 也已支持中心单视角和多视角，目标现有融合不变，各原始环境视角单独注册；完整命令和验证顺序见 [HRI_TEST_COMMANDS.md 第9节](HRI_TEST_COMMANDS.md)。运行时只对打开夹指应用接近容差，规划检查、桌面检查和接触确认仍保留。

### 桌面高度单独设置

`scripts/run_grasp_safe.sh`中的默认值是：

```bash
TABLE_Z=0.0
```

如果实际桌面在机械臂`base_link`坐标系中更低，用实测值覆盖，例如实测为`-0.05m`：

```bash
TABLE_Z=-0.05 \
bash scripts/run_grasp_safe.sh \
  --config config/pipeline.yaml \
  --phase grasp \
  --execute \
  --target-prompt \
  --yolo-weights models/yolo26m-seg.pt
```

必须使用`base_link`坐标系下的实际值，`-0.05m`只是示例。修改`TABLE_Z`只影响桌面高度保护，不会改变本次`source=environment`的手指—点云阈值。

### 临时方法的影响

在代码接入专用slack之前，把`MOTION_RUNTIME_STOP_CLEARANCE`改为`0.003`虽然可以让本次`3.5mm`通过，但它会同时将机械臂、相机和夹爪本体的全局运行阈值降为`3mm`，不如上面的定向修改安全。
