# 抓取、人手跟随与手势交接：测试命令

## 最新 T01：实际重跑 MGPC／GenX 与三路输入对照（2026-11-09）

新分支由 `--active-grasp-feedback` 显式启用，默认旧流程保留。中心后至少一次有效补看、最多两次尝试；先复核旧候选，再按需重新请求 MGPC/GenX。完整范围和限制见 [实现记录](research/ACTIVE_PERCEPTION_IMPLEMENTATION_20261109.md)，实际模型测试见 [新结果](artifacts/active_grasp_feedback_live_20261109/README.md)。当前小样本未显示混合升级收益，不能将软件测试通过等同于效果改善。

**以下两步实际运行模型，不连接相机/机械臂。** 沙箱外 RTX 4060 已确认可用；之前的 `CUDA=False` 是沙箱内设备不可见。第一步必须写 `--inference live`，否则工具默认 `saved` 仅复用历史模型输出。输出目录需尚不存在；这两步仍不能代替新 NBV 选中角度的实采。

```bash
PYTHONNOUSERSITE=1 MPLCONFIGDIR=/tmp/matplotlib \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
/home/yx/miniconda3/envs/grasp_yolo26/bin/python tools/replay_active_grasp_feedback.py \
  --run-dir \
    logs/banana_mgpc/ablation_v3_nero_refine/G_single_mgpc_genx/2026-11-01_14-04-41_913652_p551460 \
    logs/banana_mgpc/ablation_v3_nero_refine/G_single_mgpc_genx/2026-11-01_14-07-22_692308_p563137 \
  --inference live \
  --output-dir artifacts/active_grasp_feedback_live_repeat/replay_live

PYTHONNOUSERSITE=1 MPLCONFIGDIR=/tmp/matplotlib \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
/home/yx/miniconda3/envs/grasp_yolo26/bin/python tools/compare_genx_proposal_inputs.py \
  --live-replay artifacts/active_grasp_feedback_live_repeat/replay_live \
  --output-dir artifacts/active_grasp_feedback_live_repeat/input_comparison \
  --seeds 42 43 44
```

第二步在第一步的同一份新 P 上，对纯实测、原全预测＋focus、新混合逐一重跑 GenX，再做相同预算的实测几何/数值路线检查。两场景×三种子的联合资格组数为 1/6、3/6、1/6；是中心候选分析，不是执行许可或成功率。

以后显式试新运行分支时需 `--active-view --mgpc --mgpc-category banana --grasp-generator graspgenx --active-grasp-feedback --grasp-required-view-count 2`、中心/回中心/原始视图存档及已确认桌面参考；旧命令的 `--no-mgpc` 和 `--grasp-required-view-count 1` 需对应替换。新模式接管侧视数量及尝试预算，不按旧的 `--active-side-view-count 0` 跳过侧视。本轮没有改相机 YAML 或当前外参，也没有执行真机命令。[原缓存回放](artifacts/active_grasp_feedback_20261109/README.md)仅作流程验证记录。

### T01 真机试跑：新混合 MGPC＋NBV（用户要求，2026-11-09）

独立入口 [test_active_grasp_feedback.sh](scripts/test_active_grasp_feedback.sh)。先退出占用奥比或机械臂的程序，香蕉放在桌面；已经持物时不要重复运行抓取入口。

```bash
cd /home/yx/yx/panda_ws/src/whole-grasp
conda activate grasp_yolo26
sudo ip link set can0 up type can bitrate 1000000
bash scripts/test_active_grasp_feedback.sh
```

流程：中心采集→实际 MGPC/混合/GenX→至少一次有效侧视、最多两次尝试→每次回中心更新→符合接触/路线要求后真正闭爪、抬升并到配置持物关节位。不含 `--grasp-contact-check-only`，保留原预览/执行确认；若没有合格候选则结束，不保证一定抓。沿用 §7.3 的 `sept01_joint_refined_candidate`、V112、MIT 8°/s、100 mm 开口及支撑释放 25 mm，显式选择奥比 CP0T963000S4。此入口只测 T01 抓取，不进入 T02 递送/交接。

结果保存在 `logs/active_grasp_feedback_20261109/banana_orbbec_sept_joint/` 下本轮目录，重点为 `feedback/summary.json` 和各轮 `feedback_evaluation.json`。入口已做 shell 语法、完整转发及生产 CLI/遮罩模式与开口检查，见 [当前参数核对](artifacts/t01_self_mask_20261109_1710/entry_check.json)；该核对用假 Python 截获参数，没有启动硬件。只读 sysfs 已见奥比 USB 5000M。

**17:10 首次 T01 运行在自遮罩处退出，MGPC/NBV 尚未开始。** 本入口遗漏当天参考遮罩，仍继承 YAML 的 11 月 5 日旧版。现已显式使用 `config/orbbec_self_mask_20261109_100mm_reference_trial.json`；同一失败图的 core/silhouette/outer 从 0.901/0.842/0.060 变为 1.000/0.998/0.029，原门限通过，已检查的香蕉/外侧电线区域均未被遮。公共 YAML/外参/保护门限未改，仍逐帧校验；继续用上面的同一条命令重试。见 [诊断与叠图](artifacts/t01_self_mask_20261109_1710/README.md)。

**17:22轮耗时与保护更新：** 两次侧视采集outer=0.288/0.179拒绝，无有效补看；第二次回中心因多路MIT反馈约100ms老化触发电子急停，尚未抓取。已跳过无新捕获后约159秒重复更新，并在原入口显式加入观察小预算（6个昂贵预检/1条路线目标/15秒Nero）；最终抓取仍为原预算和门限。相同真实候选的离线试算61.93→18.99秒、169项＋5子测试通过，见[耗时/故障记录](artifacts/active_grasp_feedback_timing_20261109_1722/README.md)及[当前入口参数](artifacts/active_grasp_feedback_timing_20261109_1722/entry_check.json)。这不代表反馈故障已修复；电子急停后的恢复沿用现场既有流程，不自动清故障继续。下一轮前仍需处理侧视轮廓旁的外侧黑背景问题，不能靠放宽反馈或遮罩门限取得通过。

## 最新 T02：奥比腕部＋全局RealSense同步采集（2026-11-09）

**只补N不接触对照时：** 正常退出且仍稳定持物后，用`Right --held --group grasp01 --split train`原入口；启动阻抗ENGAGED并进入跟随阶段后，在全局窗口直接S人工停车，无需先伸手让腕部跟随。等WAIT_BASELINE再R/Enter、无人接触时B、WAIT_TAKE后N60秒/Q。20:45一轮在停稳阶段即触发姿态FAULT，尚无基线，反复R不能恢复；现有明确按键禁止提示。门限/故障锁存不变，见[最新R诊断](artifacts/handover_roi_locked_20261109_2045/README.md)。

**20:26新内环保护待排查：** 最新续采已正常记录T3/P2，但末尾主机墙钟跳变约417ms触发MIT保护/急停请求，断开后未发enable/reset/hold。不能按此前正常退出的假设直接重启--held；先核对时钟稳定与设备恢复条件，助手未修改时间服务或硬件。用户明确N专门对照尚未采：恢复采集、R/B完成并到WAIT_TAKE后，按N、双手完全离开物体/机械臂60秒、Q退出；这段无视觉候选是预期。见[续采与时钟证据](artifacts/handover_dataset_review_20261109_2022/README.md)。

**当前优先集中采数据：** 用户确认可用一小时以上，先6个真实抓取批次（grasp01–03/train、grasp04/validation、grasp05–06/test），再离线标注/训练/回放。当前已持物且旧程序退出后，从下面开始：

```bash
bash scripts/collect_handover_shadow.sh Right --held --group grasp01 --split train
```

每批：R/B建基线→N不碰60秒→H握住不拉/P轻拉/U完全松手等待，6次→T碰推后松手等待，6次→N靠近不碰撤回，3次。按键为粗标签，保留录像复核；完整步骤与后续批次命令见[集中采集单](research/HANDOVER_COLLECTION_20261109.md)。同次物理抓取重启仍用同group/split，不能通过换编号伪造独立测试。数据在`logs/handover_dataset_20261109/<split>/<group>/`，旧无group入口仍兼容。退出后用`python tools/check_handover_dataset.py --root logs/handover_dataset_20261109`只读检查。全局所有手关键点、精确停车参考和采集源码/配置快照已加入新记录；不加载模型、不自动开夹。

**18:04实采后下一步：** 已自动停车并采到基线和3次轻拉／松手；当前瓶颈是侧手/遮挡导致全局掌宽过滤，以及P/U动作时刻不一致。先调整固定全局机位，让握住香蕉时手背/手指更清楚；腕部相机不动，停车后R重新圈区。握住不拉时先看全局candidate=True是否能出现，再采轻拉。按键尽量由另一人操作或事后按视频标注；本轮不直接训练。见[曲线、3次现象与截图](artifacts/handover_shadow_20261109_180449/README.md)。

流程：抓取→腕部视觉递送→接触前停稳→无人接触时建基线→接收者握住轻拉→当前外部视觉与新轻拉共同确认→保持并开夹。

**近距遮挡停车修正：** 不再要求腕部在物体遮手时持续识别、精确居中或等人手静止1秒。当前脚本把跟随目标提前到25cm，28cm内用至少3个可信3D观测、跨度≥0.15秒建立停车候选，图像只需在中心附近的宽松区域（误差≤图像短边20%）；相邻3D跳变>4cm或丢检仍清空候选，不盲走。以上为试点参数，不能保证本机一定在遮挡前触发。停车后只凭真实七轴反馈判臂停稳，接取用全局RGB，不要求腕部继续看见手。

**采集时可在全局窗口按S人工停车。** 仅shadow有效：冻结当前实际姿态，不开夹、不建立基线，也不能用于release绕过许可。自动停车还没触发、腕部开始被遮挡时，就用S结束跟随；等待`FUSION_WAIT_BASELINE`后再R/B。如果上轮程序已退出、机器人仍稳定持物，可用`bash scripts/collect_handover_shadow.sh Left --held`直接进servo，跳过初始位/张爪/重新抓取；不能与另一个控制程序同时运行。`--held`仍会接入MIT保持/跟随，不是纯只读采集。

**持物入口已修复：** 原脚本遗漏`--held`解析导致“Expected one optional receiver side”；现支持下面的Right持物命令，启动输出应含`receiver=Right phase=servo handover=shadow`。Left同理，省略左右手默认Left。已有持物时保留`--held`。

```bash
bash scripts/collect_handover_shadow.sh Right --held
```

**17:49持物启动修复：** 若刚启动因`mode_feedback: 1`进入FUSION_FAULT，后面即使`device_mode=6`也不会自行恢复，R不能框选。已修复初始位置保持到MIT反馈之间的短暂等待（最多2秒）；运行中模式异常仍故障。退出旧程序后重跑上面命令；看到`cartesian impedance ENGAGED`、进入跟随后，在**全局相机窗口**按S停车，等`FUSION_WAIT_BASELINE`后按R，框选物体可接取部分并Enter确认；手完全离开物体与框后按B。不能靠按R绕过故障或停稳检查。详见[复核](artifacts/handover_held_startup_20261109/README.md)。

**当前设备：腕部为Orbbec Gemini 335 `CP0T963000S4`；用户确认全局实物为D457，SDK枚举名称显示RealSense D455、序列号 `234222300443`。** 两项分别记录，入口按序列号选设备，不因名称差异改用旧D435序列号。全局只启用RGB，兼容D455/D457名称，不扩展其深度/抓取支持；本轮只读枚举，未启动视频流或连接CAN验证。

**16:34首次运行受阻：USB链路降到2.0。** 日志在奥比选择`1280×720@30`时失败，尚未抓取或进入fusion。当前sysfs/`lsusb -t`确认奥比`1-3.1`与全局`1-3.3`都为480M，且同在`1-3`的480M Hub下；该Hub还连接CAN适配器。只移动相机连接，不要拔整只Hub或CAN适配器。优先把奥比用USB 3数据线直连主机USB 3口，全局相机也尽量分开接。线、Hub及其上行线任一环节为USB 2都会限制链路，不能仅凭接口外观判断。

重接后先做以下相机检查，均不连接机械臂。第一项腕部至少应为`5000M`，第二项确认原1280×720@30/1280×800@30能实际采到帧后再执行完整入口；不自动降分辨率/改帧率或换标定。

```bash
python tools/check_handover_usb.py
python scripts/preview_rgbd_camera.py --config config/pipeline_orbbec.yaml --headless --max-frames 30
```

入口脚本已加同一USB只读检查，链路不满足时在机械臂初始化/张爪之前退出。USB检查通过只表示链路速率满足要求，不等于双流、对齐或交接有效率已验证。

**同轮后续检查：** 两台相机已恢复到5000M（奥比2-1、全局2-8）。再次只读SDK枚举仍为`RealSense D455 / 234222300443`，因此全局采集参数不用换序列号；sysfs另显示`238143111084`，不能直接拿它替换SDK选流序列号。下一步执行上面的相机30帧检查，通过后重跑下方原入口。

**用户已完成30帧复测：** PID319826日志正常输出第1/30帧，RGB1280×720@30、原生深度1280×800@30，noise_only/holes_off、软件D2C及去畸变；整图有效深度86.8%、中位0.567m。重复粘贴不计作两次独立试验。此项确认腕部原模式恢复，不是人手深度或双流交接验证；接下来直接运行下方shadow入口。

**16:45自遮罩失败已处理为本次试验遮罩：** 用户重新启用can0后机械臂连接正常；本轮张爪99.768mm，奥比内参/模式与旧遮罩一致，但旧轮廓`core=0.915/silhouette=0.854`不匹配，未进行目标检测或抓取。用户确认相机/支架未调整。本轮经授权仅开启腕部相机补采4张参考图，没有连接CAN或改变夹爪；三张构建新固定轮廓，第四张与原失败图回放均通过原门限（core/silhouette=1.000、outer≈0.034）。新文件`config/orbbec_self_mask_20261109_100mm_reference_trial.json`已仅由下方T02入口显式指定，公共YAML/旧遮罩/手眼外参未改，不关闭遮罩或放宽门限。

已复核叠图和指定电线探测区，后者未被掩膜；这是当前场景/100mm开口的离线试验校验，不是跨姿态成功保证。第四张和原失败图未参与轮廓集合计算，但开发时已查看，不作独立盲测统计。42项相关测试及完整CLI检查通过。下一步仍运行同一个shadow入口；新姿态若轮廓变化，原有逐帧校验仍会退出。完整证据见[本轮遮罩记录](artifacts/orbbec_self_mask_20261109_reference/README.md)。

**16:59下一次失败不是原轮廓问题：** 新遮罩core=1.000/silhouette≈1.000已通过，outer=0.248失败。右夹指后方新出现竖直金属条，其所在区域贡献外侧黑像素86.48%；先在程序退出后把金属条与右夹指在画面中分开，保持相机支架不动，再原命令。不要扩大遮罩把金属条当自身删除，也不放宽0.150门限。见[外侧检查标注图](artifacts/orbbec_self_mask_20261109_outer_obstacle/outer_dark_diagnostic.png)与[数值](artifacts/orbbec_self_mask_20261109_outer_obstacle/README.md)。如果金属条不可移动，需另做外物保留机制；本轮未改运行代码/配置或操作硬件。

**以下默认完整入口适用于物体在桌面、机械臂可开始抓取时。** 先退出占用相机或机械臂的旧程序；如果已经持物，用上面的`--held`方式，避免默认流程的初始张爪。

```bash
cd /home/yx/yx/panda_ws/src/whole-grasp
conda activate grasp_yolo26
sudo ip link set can0 up type can bitrate 1000000

bash scripts/collect_handover_shadow.sh Left
```

`Left/Right`指**外部未镜像画面中MediaPipe标出的接收手**，窗口已补每只检测手的实际标签；默认Left，若接收手标Right，改成`bash scripts/collect_handover_shadow.sh Right`。第一轮固定单接收者，不能直接复制腕部身份。外部RGB无需基座外参，不修改YAML的`global_camera.enabled`。

入口脚本完整参数见[scripts/collect_handover_shadow.sh](scripts/collect_handover_shadow.sh)：沿用§7.3曾成功的单视角/no-MGPC/GenX抓取组合及`sept01_joint_refined_candidate`，同一试验外参显式传给抓取与腕部跟随，显式指定11月9日当前100mm固定自遮罩；不修改YAML，也不把该候选升级为已验证的全局伺服精度。此轮不混入T01新MGPC/NBV反馈分支，便于归因交接数据。脚本会实际抓取和递送，**交接阶段shadow不松夹**；不加载任何力矩模型。

操作顺序：

1. 接收者先只伸接收手，腕部有效三维观测引导递送；当前25cm目标/28cm准备范围，只用短段可信3D提前停车，不要求人手静止或精确居中。若仍因遮挡无法自动停车，在全局窗口按S冻结实际姿态。丢检本身不代表到位；S只是本次shadow采集的人工选择，不算自动递送成功。
2. 全局窗口显示`FUSION_WAIT_BASELINE`后，按`R`框选**物体露出部分的接取区域**，回车确认。若提供`--fusion-roi XMIN YMIN XMAX YMAX`，首次R确认该预设区域；之后R重新框选。
3. 双手离开接取区域，确认无人触碰物体，按`B`。该键只声明未接触并开始采基线，不能开夹。基线约3秒；当前试点波动/前后变化门限均为0.1 N·m，可通过`--fusion-baseline-max-std-nm`/`--fusion-baseline-max-change-nm`调整；它们不是验证过的释放阈值。
4. 显示`FUSION_WAIT_TAKE`后，先N完全不碰10秒，再P开始握住轻拉约2–3秒，完全松手按U并等15–30秒，重复3次；然后分别采T短碰/推和H握住不拉，结束动作完全松手都按U。按键在全局窗口生效，标签持续到下一标签；P不能提前代表靠近或只是握住，标签按键不参与开夹许可。动作真实时刻需结合录像复核，承重仍为unknown。
5. `C`取消/清掉本轮基线，重新B前先确认无人接触；shadow的`WOULD_RELEASE`也可C重置。`Q/Esc`或关闭窗口受控退出。开夹中的C/失联锁定中断，不自动夹回。

无`--fusion-pull-model`时仅采集，不产生释放候选；有模型的shadow只显示`WOULD_RELEASE`，始终不发新夹爪开合命令。全局相机看不到当前接收手、撤回或反馈失效时取消候选，不能沿用旧力矩偏差。旧手势/距离自动开夹分支在fusion模式中被绕开。

输出：`<save-dir>/handover_fusion/<时间_p进程号>/metadata.json`、`samples.jsonl`、`wrist_rgb/`、`global/observations.jsonl`及`global/global_rgb.avi`。记录七轴/q/原始dq/逐轴时标、可选原始current、控制状态、夹爪反馈和事件；两路用host monotonic对齐，AVI名义帧率不能代替真实时标。静止接取阶段腕部不再运行3D目标检测，日志用`detection_matches_frame`区分历史检测；RGB仍保存用于回放。

补齐同步正反例后，可离线拟合小模型（以下会话路径须替换）：

```bash
python tools/train_handover_pull.py \
  --train logs/实际路径/训练会话A logs/实际路径/训练会话B \
  --test logs/实际路径/独立留出会话 \
  --joints 1,2,3,4,5,6,7 \
  --out models/handover_pull_pilot.json
```

`--joints 2`为J2对照，`--joints 1,2,3,5,6,7`为去J4对照。训练/留出必须是不同完整持物会话，不能将相邻窗口随机分组；不要反复用留出结果挑模型。当前训练器二分类`TAKE_PULL`与已标记的`NO_CONTACT/BUMP_PUSH/POST_RELEASE`，`HOLD_NO_PULL`不冒充无接触负例。输出仅为动作分类统计，须继续回放状态机及受控闭环验证。

未来验证后将shadow改为`--fusion-handover release --fusion-pull-model models/handover_pull_pilot.json`才接开夹；无模型或模型窗口不一致/非同MIT模式会提前报错。默认短确认0.3秒、开夹5秒，沿用原伺服85 mm/0.5上限并服从更小的用户硬件限值；这些不是本轮新真机验证参数。本次仅代码修改与离线测试，没有启动相机、CAN或夹爪。

## 0. 更换腕部奥比相机后的 SDK 测试（2026-11-02）

旧仓库的奥比是 Gemini 330 系列 ROS 驱动；这里接入官方 Orbbec Python SDK v2，供抓取、NBV、视觉跟随共用。
新配置为 `config/pipeline_orbbec.yaml`；现场 USB 已识别为 **Gemini 335／USB 3.2**。
`config/pipeline.yaml` 和下文原 A–H 命令保留为 D435 历史配置。
**相机已更换；奥比配置现已指向11月3日离线验证后的试验外参。不要直接执行下文带旧 D435 配置的机械臂命令。**
奥比配置不使用 D435 的序列号、360 激光功率和旧夹指遮罩。相机预览无需手眼标定。
**11月3日的试验外参已填入奥比配置；实机到位偏向一侧，见0.25。**
**最新对照见0.15：已按重投影误差排序，并分别排除Top2／Top3重算；留出记录未体现稳定改善。**
**0.16已核查求解退出逻辑：当前五算法约0.023秒完成，没有误差不到阈值就永久迭代的逻辑；暂不因该猜测重采或改质量门限。**
**0.17已完成23组初值／LM／Huber／去重／多样性对照，附逐帧角点和留出姿态投影图。7种初值收敛到几乎同一解；目前没有同时改善留出平移与旋转、足以替换运行外参的结果。**
**0.18已核查能否按同步误差删帧：21张旧记录缺少所需时序元数据；后9张未见拍照前后关节变化或回调顺序异常。目前没有可确认的不同步坏帧，未据手眼残差冒充同步误差删图。**
**0.19已完成纯使能与示教两段20秒只读连续诊断：两组分别601／600帧，均无帧号／设备时标回退或跳帧。静止数据未识别图像相对运动的固定延迟，不能直接修正旧21张；暂停追加求解器对照，不猜测删帧。**
**0.20已按用户要求排除06／14重算：单排06、单排14、两张都排除均已与完整21张比较；排除14可略降留出平移误差，但旋转变差，其他帧仍有明显转角冲突。这是定向敏感性检验，未将任何候选设为运行外参。**
**0.22已复算11月3日纯使能保持的14张：训练内位置一致性均值4.79 mm，最差姿态对相对转角差2.94°；跨批次参考仍有偏差。下一步采5个新姿态独立验收，候选尚未投入运行。**
**0.23已用新采的8个独立姿态验证：位置一致性均值5.98 mm、最大8.85 mm，旋转均值1.49°、最大2.69°。已将这版作为试验外参填入奥比配置；抓取与跟随加载的矩阵已离线核对一致。**

### 0.1 只安装奥比 SDK，复用现有环境

无需 `uv`、新建 Conda 环境或重装 PyTorch。包名为 `pyorbbecsdk2`，导入名为 `pyorbbecsdk`。
下面直接安装预编译包到当前 `grasp_yolo26` 环境，不需要另存 `/tmp`；`--no-deps` 避免更新现有依赖。
当前 `grasp_yolo26` 已安装并成功导入 SDK **2.1.3**，PyTorch 保持 `2.7.1+cu128`；当前机器跳过安装，直接做 0.2。
下面保留安装命令供其他环境复用。官方 PyPI 源下载较慢时使用清华镜像安装同一版本。

```bash
cd /home/yx/yx/panda_ws/src/whole-grasp
conda activate grasp_yolo26
python -m pip install --no-deps --only-binary=:all: pyorbbecsdk2==2.1.3 --index-url https://pypi.tuna.tsinghua.edu.cn/simple
python -c "import pyorbbecsdk; print(pyorbbecsdk.__file__)"
```

参考：[官方 SDK v2 安装说明](https://github.com/orbbec/pyorbbecsdk/tree/v2-main)。

本机已由你完成奥比 USB 权限规则安装并拔插相机，SDK 已成功枚举 **Gemini 335，序列号 `CP0T963000S4`**；序列号已写入奥比配置。
当前机器跳过此步骤。下面保留新机器所需的规则安装命令；若 SDK 能枚举但报 `usbEnumerator openUsbDevice failed`，在自己的终端安装规则，然后拔插相机：

```bash
sudo bash /home/yx/miniconda3/envs/grasp_yolo26/lib/python3.10/site-packages/pyorbbecsdk/shared/install_udev_rules.sh
```

它只安装 SDK 自带的奥比 udev 规则并重新加载设备权限，不需要让整个抓取程序以 root 运行。

### 0.2 先看相机型号和支持的分辨率，再看 RGB-D 对齐

结束旧相机进程和 ROS 奥比驱动，避免占用。以下全部只开相机，不连接 CAN，不启动机械臂。

```bash
python scripts/preview_rgbd_camera.py --config config/pipeline_orbbec.yaml --list-devices
python scripts/preview_rgbd_camera.py --config config/pipeline_orbbec.yaml
```

窗口依次显示 RGB、对齐深度和叠加图；黑色深度为无效，S 保存 RGB／深度／内参，Q 或 Esc 退出。
本机已完成 30 帧无窗口实采：SDK 正常启动与退出，RGB `(720,1280,3)`、对齐深度 `(720,1280)`，相机子进程独立于 CAN。
默认 RGB `1280×720@30`、原生深度 `1280×800@30`；SDK 按实际深度比例换算为米，并做软件 D2C 对齐和共同去畸变。
不支持某个模式时会列出支持模式，不会偷偷缩放深度冒充对齐；按列出的模式改奥比 YAML。
`depth_filters: all` 在奥比上只启用噪声点剔除，不启用时域补洞、空间补洞或孔洞填充。
若设备没有 `High Density` 预设，将奥比 YAML 的 `depth_preset` 改为 `unchanged` 使用设备当前预设。

### 0.3 测 YOLO 和人手深度（保留可视化）

```bash
python scripts/preview_yolo_camera.py \
  --backend orbbec \
  --weights models/food5_visible_v3_full_reviewed_corrected_best.pt \
  --target-class banana \
  --width 1280 --height 720 --fps 30 \
  --imgsz 640 --conf 0.25
```

```bash
python scripts/test_realsense_hand.py \
  --backend orbbec \
  --serial CP0T963000S4 \
  --width 1280 --height 720 --fps 30 \
  --depth-width 1280 --depth-height 800 --depth-fps 30 \
  --depth-filters all --show-depth \
  --depth-radius 8 \
  --save-dir captures/orbbec_hand
```

手部脚本保留原文件名，后端已支持奥比。S 保存匹配的 RGB、有效深度和掌心诊断；出现“RGB hand found; depth invalid”也可以保存。
已完成此脚本 10 帧无窗口联调：奥比和 MediaPipe 正常启动、读取及退出。该次未取得有效 3D 手，仍需用上述可视化命令把手放入视野，检查掌心深度；这不代表手部识别效果已验证。
RGB-D 与人手预览通过后，给新安装位置填写对应的 `grasp_calib_json` 和 `servo_calib_json`，再将主流程命令的配置切换为 `config/pipeline_orbbec.yaml`。
若夹指进入视野，还需用奥比画面建立对应遮罩；不能套用原 D435 遮罩。

#### 0.3.1 T02 当前奥比近距离人手 RGB-D 测试（2026-11-05）

本次只测相机，不需要启动 CAN，也不连接、移动机械臂或控制夹爪。先退出其他占用相机的抓取/跟随/预览程序；现有持物状态不要因相机测试而改变。用户确认奥比人手近距深度尚未验证，本节不预设它与旧 D435 有相同问题。

```bash
cd /home/yx/yx/panda_ws/src/whole-grasp
conda activate grasp_yolo26
python scripts/test_realsense_hand.py \
  --backend orbbec --serial CP0T963000S4 \
  --width 1280 --height 720 --fps 30 \
  --depth-width 1280 --depth-height 800 --depth-fps 30 \
  --depth-filters all --show-depth \
  --num-hands 2 \
  --min-detection-confidence 0.35 \
  --min-presence-confidence 0.35 \
  --min-tracking-confidence 0.35 \
  --depth-radius 8 --min-depth-landmarks 3 \
  --min-depth 0.10 --max-depth 2.0 \
  --smoothing-alpha 1.0 \
  --save-dir logs/orbbec_hand_depth_debug
```

RGB/原生深度配置对应当前奥比 YAML；此独立脚本不自动加载 YAML。这里用 8 像素采样半径、至少 3/5 掌部关键点，显示不做时间平滑；没有修改主流程的默认参数。左侧 RGB/骨架，右侧与 RGB 对齐的深度；黑色表示无有效深度或超出本次筛选范围。相机 XYZ 为相机系毫米值，不是基座系。

操作：先只伸一只手，摊掌放在镜头前约 40–50 cm，稳定约 3–5 s 后按 S；再依次到约 30、20、15 cm，各停住并按 S。距离指镜头到手，用实际距离作粗参考；三维失效时不要用旧 Z 当作当前距离。能看到物体时再把手移到实际接物位置，正面/侧面各试一次。成功、骨架仍在但缺深度、骨架也消失都按 S 各存几帧。点击预览窗口后按键；Q/Esc 退出，Ctrl+C 也可退出。

结果区分：`reason=ok` 表示有掌心候选且至少 3 个掌部关键点深度有效，不保证测量准确；`invalid_palm_depth` 表示二维手已检出但深度条件不足；`no_hand_2d` 表示当前无二维手输出。图中 `jitter` 只是最近有效帧位置变化，移动手时不能当作静态精度。当前 `--min-depth 0.10` 会排除小于 10 cm 的深度，这是软件筛选值，不是相机硬件近距下限。

已补充完全无二维手时的 S 保存能力；每个 `logs/orbbec_hand_depth_debug/<时间>/` 含 `color.png`、`overlay.png`、实际检测/滤波/原始深度，以及带状态、关键点、相机时间和采集参数的 `hand_pose.json`。无手时相关手部字段为 null，不伪造 0/5 测量。记录终端打印的成功/失败保存目录，并注明大致距离及当时是正面、侧面还是被物体挡住，供后续核查。

本轮仅无硬件软件验证：`tests/test_hand_preview_snapshot.py` 与 `tests/test_mediapipe_hand_geometry.py` 共 5 项通过，命令参数已通过解析检查；奥比本次现场效果待用户执行后确认。

**2026-11-05 16:36–16:39 用户实测回看：** 程序正常处理 4882 帧并退出，保存的 15 帧中 4 个 `ok`、9 个 `invalid_palm_depth`、2 个 `no_hand_2d`；这些人工挑选的截图不用于计算整轮成功率。16:36:56/16:37:52 的掌心与五处关键点共六个采样区在未滤波、已对齐深度中均全为 0；16:38:15 的 5/5 掌部深度完整。16:38:44/47 的 RGB 中手明显被香蕉挡住，二维也丢失。用户确认没有量过实际镜头到手的距离，不能据此标定最近有效测距。详见 [同像素对照图、逐区域统计与边界](artifacts/orbbec_hand_depth_20261105_1636/README.md)。下一次沿用同命令，机器人/相机/持物状态保持不变，让手在同一前后位置和近似掌面方向下作图像左/中/右对照，每处停几秒存 S，再对照张掌/握持，注明位置和实测距离；暂不改深度门槛或启动抓取跟随命令。

#### 0.3.2 T02 奥比原生深度模式近距对照（2026-11-05）

第二轮测试正常处理 7484 帧，保存的 6 张中：左侧两张与右侧一张均 5/5，中部一张 0/5，实际握香蕉两张无二维手。中部帧的掌心与五点未滤波深度均全零；左右成功帧估计 Z 不同，无外部距离标签，不能当作严格同距离对照。[第二轮图片/统计](artifacts/orbbec_hand_depth_20261105_1703/README.md)。

新核实：[官方数据手册](https://orbbec.com/wp-content/uploads/2024/04/Gemini-330-series-Datasheet-V1.4_20250124.pdf) §4.7、PDF 第 37 页，Gemini 335 在原生深度 **1280×800** 模式标称 Min-Z 为 **0.26 m**，**848×480** 为 **0.18 m**。脚本 `--min-depth 0.10` 仅是软件筛选下限。本次只在独立相机脚本比较 848×480 原生模式，RGB、置信门槛、采样半径和噪点滤波保持相同；不改 YAML/外参，不连接 CAN，不移动或开夹。

```bash
cd /home/yx/yx/panda_ws/src/whole-grasp
conda activate grasp_yolo26
python scripts/test_realsense_hand.py \
  --backend orbbec --serial CP0T963000S4 \
  --width 1280 --height 720 --fps 30 \
  --depth-width 848 --depth-height 480 --depth-fps 30 \
  --depth-filters all --show-depth \
  --num-hands 2 \
  --min-detection-confidence 0.35 \
  --min-presence-confidence 0.35 \
  --min-tracking-confidence 0.35 \
  --depth-radius 8 --min-depth-landmarks 3 \
  --min-depth 0.10 --max-depth 2.0 \
  --smoothing-alpha 1.0 \
  --save-dir logs/orbbec_hand_depth_848x480_debug
```

先退出上一次相机程序。机器人、相机与当前持物状态保持不变，先在约 40–50 cm 处左右/中部张掌，尽量用尺作独立距离参考；每处停几秒存 S。再在约 20–25 cm 处重复，最后存实际握持帧。S/退出规则同 §0.3.1；启动日志应显示 `depth=848x480@30`，对齐到 RGB 后的检测数组仍为 720×1280，不是没有切换。模式改变还会影响视场和采样密度，不能只依据个别成功帧认定近距问题解决。手被香蕉挡住时，换深度模式不能保证恢复完整二维骨架。

命令参数及相机对象配置已离线核对；848×480 实际采集和效果尚待用户测试。若中部张掌仍持续缺深度，后续再检查原生深度/左右红外及物体/夹指光路，不继续重复同样张掌截图或降低门槛。

**2026-11-05 17:22 用户实测：** 启动日志确认 `depth=848x480@30`，3795 帧正常结束；选存 3 帧为 5/5 ok（Z=447 mm）、3/5 ok（Z=268.5 mm）、握持 no_hand_2d。3/5 帧缺失的手腕/小指根部在未滤波深度也全零；握持遮挡没有消失。没有同位置/独立距离配对，不能据这批快照认定 848×480 优于默认，更不能把终端 162–179 mm 数值当精度验证。详见 [本轮图片/数组核查](artifacts/orbbec_hand_depth_20261105_1722/README.md)。当前不改 YAML 或释放条件，相机初筛先收束，优先准备视觉状态、逐轴力矩和人工碰/拉/松事件的统一时钟记录；目前两个独立脚本不构成现成的同步采集器。

### 0.4 奥比手眼标定：先识别板，再采集（2026-11-02）

**16:18这轮使用了另一块板，粗测黑白棋盘本体约214×160 mm（8×6格），单个黑色标记边长约20 mm，均不是精确测量。之后已换回旧板，当前操作见0.7；本节保留新板采集记录。**
已单独记录在 `config/charuco_board_orbbec.yaml`：方格边长约26.7 mm、标记约20 mm。旧 `config/charuco_board.yaml` 留给旧板，奥比标定不要再选它。
下方命令已切换新板配置／标记尺寸；已有25张图片已离线重处理，当前不需要重新采集整套数据。

当前 `depth_filters: all` 开启的是 **SDK 软件噪点剔除**，不是机内硬件降噪。
实机复查：按主流程启动并加载 `High Density` 后，机内 `hardware_noise_enabled=False`、SDK 噪点剔除 `is_enabled=True`。机内阈值当前约 `0.2`，本次没有修改其开关或阈值。
ChArUco 手眼标定只开 RGB，不读取深度，因此深度降噪不参与角点／手眼求解，也不会因标定而自动打开硬件深度降噪。
标定与主流程均通过同一个奥比适配器，使用**共同去畸变后的 RGB 和匹配的 K、D=0**；不要传旧 D435 的 `--intrinsics`。

先只看标定板，不连接 CAN：

```bash
python tools/calibrate_d435i_charuco.py test \
  --backend orbbec --serial CP0T963000S4 \
  --width 1280 --height 720 --fps 30 \
  --board config/charuco_board_orbbec.yaml
```

文件名保留旧名，现已支持奥比。先确认画面角点正常、`quality=OK`。新板仍为8×6、`DICT_5X5_100`，使用上述单独配置；这不会使粗测尺寸变成精确标定值。

确认板配置对应实物后，用现有低速示教方式移动机械臂，标定板固定，每次停稳后按 S 保存；D 删除上一张，Q 退出。
以下脚本**只读法兰反馈，不使能、不发送机械臂运动指令**。先退出抓取／跟随程序，避免相机或控制连接冲突。

```bash
python tools/calibrate_d435i_charuco.py capture \
  --backend orbbec --serial CP0T963000S4 \
  --width 1280 --height 720 --fps 30 \
  --board config/charuco_board_orbbec.yaml \
  --channel can0 --firmware V111 \
  --output "calibration/orbbec_handeye_$(date +%Y%m%d_%H%M%S)"
```

采集多个位置和不同方向的倾斜姿态，避免只平移或只绕一个轴；建议先采约 20 个停稳姿态。数据保存后再离线求解、另采数据验证，再接入主流程；原 D435 标定文件保留。

#### 本次 16:18 奥比采集：样本已保留，手眼结果待排查

`calibration/orbbec_handeye_20261102_161824` 已保存 25 个姿态，无缺失图片；角点平均重投影误差约 `0.171 px`。
Ctrl+C 只中断了后续采图，没有丢失已保存样本。字体警告和 `drawFrameAxes` 越界警告不等同于标定失败。
RGB-only 标定入口的相机日志已修正为 `depth=disabled`；原来的 `filters=off` 不表示抓取流程的降噪配置被关闭。
以后采集时 Ctrl+C 会正常结束并打印数据目录，不再显示退出堆栈。

已离线运行五种手眼方法，候选保存在该目录的 `handeye_candidate.json`，未接入任何运行配置。
目前同一固定板在基座系中的训练一致性误差为：平移中位约 `22.8 mm`、最大 `57.0 mm`；旋转中位约 `2.19°`。
用户已确认板和相机支架全程固定。对这些图片做小幅内参重估后，平移中位仍约 `22.8 mm`，暂不能用“角点清楚”或“改一下内参”解释或消除偏差。
诊断保存在该目录的 `diagnosis.json`；当前结果不能当作已经验收的手眼外参。

下一步只读核对 RGB PnP 与深度，不重采整套标定。保持同一块板固定，手动选择三个能看清同一标记的机械臂姿态，每个姿态停稳后采三次，共九次。相机和夹爪开口保持不变。
**这条命令现在默认打开实时相机窗口，无需加参数。** 窗口显示标记边框和 ID、选中中心、深度毫米数、锁定 ID 和保存数量；红色 `NO MARKERS DETECTED` 表示未识别，橙色提示目标 ID 不在画面或深度缺失。
首次成功采样后锁定标记 ID。确认选中标记可见且有深度，点一下相机窗口让它获得焦点，再按 **S 或 Enter** 采样；终端按 Enter 也能采。窗口 Q／Esc／关闭按钮或终端 Ctrl+C 正常退出，已保存数据保留。
只有需要无窗口运行时才加 `--no-preview`；平时直接用下面的原命令。
程序只读反馈、采图，不发送使能、运动或夹爪命令；先结束其他相机／控制程序。

```bash
python tools/verify_handeye_aruco_consistency.py \
  --config config/pipeline_orbbec.yaml \
  --calib-json calibration/orbbec_handeye_20261102_161824_board214x160/handeye_candidate.json \
  --aruco-dict DICT_5X5_100 \
  --marker-length-mm 20 \
  --samples 9 \
  --save-dir logs/orbbec_handeye_diagnosis
```

重点看 `[rgb-vs-depth]`：它直接比较相机坐标系里的 RGB 估计与深度返回，不依赖候选手眼外参。
单帧偏差与同姿态重复采样，再结合跨姿态变化，用来继续区分相机几何、时序和机器人坐标链问题；先不按此候选执行抓取。

#### 16:55 九次核对已完成：下一步先核对实物尺寸

日志为 `logs/orbbec_handeye_diagnosis/handeye_aruco_consistency_2026-11-02_16-55-39.json`。已完成三个姿态、各三次，暂时不用重复跑这九次。

| 姿态 | 深度测距均值 | 单标记 RGB 估距 | 整板35角点 RGB 复算 | 深度减整板 RGB |
| --- | ---: | ---: | ---: | ---: |
| 1 | 434.3 mm | 461.7 mm | 455.5 mm | −21.2 mm |
| 2 | 394.0 mm | 426.3 mm | 411.2 mm | −17.2 mm |
| 3 | 496.0 mm | 547.1 mm | 518.2 mm | −22.2 mm |

同姿态重复测距标准差为 0–0.47 mm，因此本次主要表现为系统差异。只用一个小 ArUco 标记会放大部分误差；整板复算仍有约17–22 mm差异，不能据单标记的27–51 mm就断言深度相机偏了这么多。
整板与深度平面的法向差约0.2–0.8°，深度平面内部偏差中位量级约0.5–0.65 mm。出厂深度到RGB的前后位移约−1.77 mm，不能直接当成这约2 cm差异的解释。
**该次日志及上表使用的是后来确认不适用的旧板尺寸，不能直接拿来判定深度相机误差。** 最新确认是另一块板，粗测214×160 mm、标记20 mm；此前约219.7 mm只是两种测距一致时的假设值，不能用来替代实测214 mm。
整板逐帧复算保存在 `board_vs_depth_analysis.json`，三姿态汇总保存在同目录 `diagnosis_summary.json`。

#### 新板尺寸更正后的离线重算（当前状态）

原始25张图片和原始9次RGB-D记录均保留。使用新板配置重新检测25张图片并运行全部五种手眼方法，结果在：

- 数据：`calibration/orbbec_handeye_20261102_161824_board214x160/samples.json`
- 候选：同目录 `handeye_candidate.json`，仍为 `unvalidated`，已记录尺寸粗测来源。
- 对比：同目录 `comparison.json`。

重算后平移一致性中位约22.3 mm、最大55.0 mm，旋转中位约2.19°；更正尺寸没有消除跨姿态不一致，不能据此启用真抓取。
按新尺寸复算已有9次数据，三姿态整板RGB估距约423.3、382.1、481.6 mm；对应深度约434.3、394.0、496.0 mm，差约+11.0、+11.9、+14.4 mm。差值方向已反转，因此不能沿用旧尺寸下“深度偏近17–22 mm”的说法，也不能靠调深度比例强行对齐。
新复算保存在 `logs/orbbec_handeye_diagnosis/board214x160_reanalysis.json`。下一步应查剩余的位姿／图像一致性问题；不再重复要求核对同一轮已经提供的粗测尺寸。
`config/pipeline_orbbec.yaml` 的抓取／跟随外参仍未填入候选，桌面高度和SDK深度比例也未改动。

### 0.5 换回原实体标定板：先采12个姿态作对照

适用于换回原 D435 标定时使用的那块实体板；相机仍是当前奥比 Gemini 335。
旧板用 `config/charuco_board.yaml`，新214×160 mm板用 `config/charuco_board_orbbec.yaml`，两者不能混用。
旧板的记录尺寸本身也只是既有测量依据，不能因为“以前用过”就当作本次标定已经准确。

先退出其他相机／控制程序。将旧板固定平整，整个采集过程中不移动板和相机支架。
先采12个不同的机械臂姿态，每个姿态停稳约2秒后按S保存一张，保持足够角点可见。
姿态要包含绕不同方向的倾斜变化，避免只平移或只绕一个轴；保存到 `sample 11` 就是12张，按Q退出。
脚本只读法兰反馈，不使能、不发送机械臂运动或夹爪命令；机械臂仍由你通过现有示教方式移动。

```bash
python tools/calibrate_d435i_charuco.py capture \
  --backend orbbec --serial CP0T963000S4 \
  --width 1280 --height 720 --fps 30 \
  --board config/charuco_board.yaml \
  --channel can0 --firmware V111 \
  --output "calibration/orbbec_oldboard_check_$(date +%Y%m%d_%H%M%S)"
```

完成后先离线试算这12张。若仍有厘米级跨姿态不一致，继续增加相似照片不能保证解决问题，应转查位姿反馈与图像对应关系；若明显改善，也需另采验证数据后再决定能否接入抓取。
旧的25张、新板更正后的25张，以及本次旧板对照分别保留，不覆盖原始数据或自动更新运行外参。

### 0.6 17:26 旧板21张已完成：后续只采关节／图像时序诊断

`calibration/orbbec_oldboard_check_20261102_172656` 已保存21张，Ctrl+C正常退出，无缺失图片。
已运行五种手眼方法：当前排名DANIILIDIS，平移一致性中位18.95 mm、最大46.83 mm，旋转中位2.18°、最大5.22°。
与新板改尺寸后的中位22.30 mm相比仍是厘米级偏差；拍摄姿态也不同，不能把数值下降直接当成旧板已解决问题。
候选在该目录的 `handeye_candidate.json`，状态仍为 `unvalidated`，未接入抓取／跟随。

已核对：

- 两组保存位姿的矩阵换算与SDK的RPY算法一致，没有发现这里的直接转换错误。
- 当前静止姿态只读3次，法兰反馈与SDK根据关节角计算的FK差约0.015 mm、0.0024°；这只是软件自洽，不验证实物几何准确。
- 对保存图片重估少量内参／径向畸变，结果仍为厘米级偏差；没有据此修改运行相机模型。
- 平面PnP另一组解的图像误差明显更差，不支持改用另一解。
- 用户确认奥比会随第7轴和夹爪一起转，当前相机到法兰的安装模型假设成立。

上述核对和结果分别保存在同目录 `diagnosis.json`、`current_feedback_fk_audit.json`、`radial_camera_refit_audit.json`、`planar_pose_ambiguity_audit.json`。
**先不要再换板或重采整套。** 原来的21张没有关节角和图像接收时刻，无法补查这些字段；现已给采集程序增加采图前后关节角、关节／法兰时间戳、图像接收／读取时刻、帧序号和读取时图像年龄。没有增加运动指令或新拒绝阈值。

下面只作诊断：旧板保持固定，选择3个不同观察姿态，每个停稳约2秒后按S采3次，共9张，保存至 `sample 08` 后按Q退出。
`--candidate-frames 1` 让每次S只选一帧，去掉5帧择优的时序歧义；不是断言之前5帧择优就是故障原因。
此组9张用于核对关节与图像对应关系，不直接运行要求至少12张的常规手眼求解。

```bash
python tools/calibrate_d435i_charuco.py capture \
  --backend orbbec --serial CP0T963000S4 \
  --width 1280 --height 720 --fps 30 \
  --board config/charuco_board.yaml \
  --channel can0 --firmware V111 \
  --candidate-frames 1 \
  --output "calibration/orbbec_sync_check_$(date +%Y%m%d_%H%M%S)"
```

窗口继续显示角点识别状态，终端每次保存会打印 `image_age=...ms`、`joint_delta=...deg`。
脚本只读反馈，机械臂仍由你用现有示教方式移动；原有相机、板配置和运行外参没有因这次诊断被自动替换。

#### 17:53 时序诊断已完成：不用再重复采这9张

数据为 `calibration/orbbec_sync_check_20261102_175351`，共9张、3个独立机械臂姿态；完整结果在同目录 `diagnosis.json` 和 `timing_feedback_analysis.json`。

- 图像从主机接收回调到读取返回约0.93–2.30 ms，9张均在采样前后反馈请求的时间区间内。该数值不是曝光至主机的总延迟。
- 9次采样前后关节反馈无变化。同姿态3次视觉估计的最大位置变化约0.07–0.19 mm、旋转变化约0.04–0.08°。
- 关节FK与法兰反馈在三个姿态都一致，差约0.010–0.015 mm、0.0017–0.0025°；这是同一套名义几何下的软件一致性，不验证实物绝对精度。
- 第一个姿态与前21张的最后一个姿态基本相同，两轮图像估计相差仅约0.053 mm、0.0155°，没有看到该姿态下明显的跨轮漂移。
- 使用前21张候选对这组数据作留出核对，固定板位置一致性误差中位仍约17.86 mm、最大20.53 mm。9张只有3个独立姿态，不能当成9个不同姿态重新标定。

这9张的端点记录没有显示运动或回调读取积压，但未记录连续流及共同曝光时基，不能彻底排除采集窗口内变化、曝光前缓存或示教模式下的时序问题。用户最新说明旧采集为示教模式，当前为纯使能保持；此前“已使能保持”的前提由此更正。后续独立诊断见0.19，不能以当前静止模式验证通过直接证明旧21张同步正确。当前候选仍未验收，不改桌面或深度比例凑一致。
用户已确认：相机随第7轴和夹爪一起转；机械臂底座用螺栓牢固固定在标定板所在的同一张台面上。当前没有证据把偏差归因于底座松动。

### 0.7 当前下一步：只转一次，比较三个独立的转角读数

对已有21张做联合优化，再用后9张核对，位置误差仍约16–21 mm；只换求解器未解决问题。
当前静止姿态中，IMU重力方向与画面估出的板面法向平均相差约0.35°；这个单姿态结果还不能解释跨姿态偏差。
下一步比较同一次转动的**机械臂反馈转角、画面转角、陀螺仪积分转角**，不使用待验证的手眼外参或深度测距。
该核对只用于定位问题；IMU本身也有零偏和误差，三个数不能直接作为标定验收结论。

退出其他相机程序，旧标定板保持固定。程序只读相机、陀螺仪和关节／法兰反馈，不使能或发送运动、停止、夹爪指令。

```bash
cd /home/yx/yx/panda_ws/src/whole-grasp
conda activate grasp_yolo26
python tools/verify_handeye_rotation_imu.py \
  --config config/pipeline_orbbec.yaml
```

1. 保持机械臂静止约2秒，窗口显示 `READY`，确认板为 `Board: OK`。
2. 点击相机窗口，按 **S** 记录起点。
3. 用你已有的低速示教方式，只改变一个有余量的关节约10–15°，远离限位，其他关节不动，然后停稳。起点和终点均需能看清标定板。
4. 再按 **S** 记录终点，继续静止约2秒，等待输出 `[ANGLES]` 和 `[SAVED]`。
5. 按 **Q** 退出，把三个转角或 `result.json` 的路径发来。

窗口Enter或终端Enter也可记录，R重新采静止零偏。结果保存在 `calibration/orbbec_rotation_audit/<时间>/result.json`，同时保存两端照片、关节反馈与原始陀螺仪数据。
程序按RGB硬件时间截取陀螺仪区间，用起止静止段估计零偏，并记录采样间断与零偏变化，供后续判断读数是否可信。现在按起点S时重新使用刚过去2秒的静止数据估零偏，避免READY后长时间等待仍用启动值。
新工具与相关采集流程共24项测试通过，包含READY后长时间等待、零偏变化的回归测试；现场结果见下，运行手眼外参仍未更新。

#### 18:37 第一轮转角数据：两路相机读数接近，仍需隔离关节

数据在 `calibration/orbbec_rotation_audit/20261102_183716_661875/result.json`，复核在同目录 `analysis.json`。

| 来源 | 起止转角 |
| --- | ---: |
| 机械臂法兰反馈 | 15.421° |
| 固定板画面估计 | 15.873° |
| 陀螺仪积分 | 15.936° |

画面与陀螺仪的角度大小差0.063°，均比机械臂反馈大约0.45–0.52°。比较完整三维旋转时，两路相机估计相差约0.262°，不能把只差0.063°说成姿态完全一致。
陀螺仪连续采样，最大间隔约5.048 ms；起止相隔52.729秒，初始静止段有瞬态。改用起点前1／2／3／5秒的静止段估零偏，积分角度为15.866–15.906°，仍比机械臂反馈大约0.45–0.49°。
该差异不依赖手眼外参，不能单靠换一个固定手眼矩阵消除；但一轮数据也不能证明机械臂编码器或某个关节一定不准。

本次关节变化依次为 `[+0.005, -0.008, +0.714, -0.146, -13.934, -5.166, -2.854]°`，第5、6、7轴均有明显运动，无法据此定位某一轴。没有修改编码器比例、关节零位、URDF或运行外参。
该轮安排为单独转第7轴，现已完成，结果见下。保持板固定，起止均停稳；按S后完成动作并停稳就再按S，减少长时间空等造成的积分漂移。不要用笛卡尔姿态拖动代替单关节转动。
最后重新显示 `READY` 只是程序在为下一轮准备零偏，不表示刚保存的数据失败。

#### 18:46 第7轴单独转动完成：本轮未见明显转角不一致

数据在 `calibration/orbbec_rotation_audit/20261102_184614_965188/result.json`，原始结果保留，离线重算在同目录 `analysis.json`。
第7轴变化−14.799°，其余六轴变化均不超过0.002°；机械臂反馈转角14.799°，画面14.935°，两者差0.136°。两端角点平均重投影误差约0.190／0.213 px。
`drawFrameAxes ... out of frame` 是画出来的坐标轴端点越界，不等同于角点检测或位姿求解失败。

原始陀螺仪读数14.478°使用的是起点前约226秒估出的启动零偏。改用起点前2秒静止数据重算为14.737°，比机械臂反馈小0.062°；起点前1／2／3／5秒窗口的结果为14.757／14.737／14.704／14.692°。因此不能拿原始−0.321°就判定第7轴异常。
诊断脚本已在起点S处刷新静止零偏；没有修改机械臂关节比例、零位、URDF、手眼外参或深度比例。

该轮安排为单独转第6轴约10°，现已完成，结果见下。每个端点按S前静止至少2秒，并保持标定板可见。无需重测第7轴，也无需重采整套手眼数据。
这轮只支持“该姿态下第7轴转角基本相符”；约18 mm的跨姿态位置误差仍未解决，不能据此接入抓取。

#### 18:52 第6轴及两次手腕运动联合核对：暂停继续逐轴测试

数据为 `calibration/orbbec_rotation_audit/20261102_185244_191052/result.json`，单轮复核在同目录 `analysis.json`，两轮联合核对在 `wrist_pair_analysis.json`。
第6轴变化−10.802°，其他关节变化不超过0.006°；机械臂反馈10.807°、画面10.631°，差−0.176°。陀螺仪10.415°，但积分区间长达53.697秒、静止段也存在振动，不能据此直接修正编码器比例或零位。

联合结果更关键：用原21张求出的手眼候选换算，固定板在第7轴运动前后的位置会相差11.64 mm、方向相差1.43°；第6轴运动前后相差14.28 mm、1.93°。因此两次单轴转角接近，不等于旧手眼外参已经准确。
只用这两次相对运动拟合一个局部外参，板的位置变化可降至1.91／1.02 mm，但这是拟合数据上的结果。拿该局部外参检查原21张，位置一致性中位仍约20.7 mm、最大60.4 mm，未解决问题；后9张的中位仍约13.8 mm。此处统计均以各数据集自身平均板位姿为参考，不能与其他参考定义的数值直接作严格增益比较。
局部解仅存在诊断JSON中，没有覆盖候选或填入运行配置，也没有修改关节、URDF、深度比例、桌面或抓取参数。

**当前先不继续第5轴或其余各轴，不重采整套。** 下一步核对奥比相机的实际固定结构，再区分支架随负载形变、机器人名义几何与实物差异、视觉位姿估计误差。用户已确认底座螺栓牢固；当前不能把根因直接定为机器人或支架。

### 0.8 尺寸来源核对：已确认当前实体板与历史尺寸一致

用户已确认相机使用螺钉＋金属支架。当前没有独立测量证明支架形变，不能把它当作已定位的原因，也不继续逐轴试转。

| 数据 | 程序实际选用的板参数 | 物理尺寸依据 |
| --- | --- | --- |
| 16:18原25张 | 最初误用旧板；后来离线按26.7 mm格长、20 mm标记重算 | 当时另一块板，用户粗测棋盘本体214×160 mm、标记约20 mm |
| 17:26旧板21张、17:53后9张 | 28.729 mm格长、20.95 mm标记，模型棋盘本体229.832×172.374 mm | 用户最新确认与历史尺寸一致，约229×173 mm、标记20.95 mm；保留原测量精度 |
| 18:37／18:46／18:52三次转角 | 同上旧板参数 | 转角测试不能补上实体板的绝对尺寸依据 |

旧格长28.729 mm来自历史宽度229/8与高度173/6的平均，不能把小数位数当成测量精度；两方向格长本来约28.625与28.833 mm。`config/charuco_board_orbbec.yaml` 文件名只是历史命名，不代表奥比相机必须使用那块214×160 mm的板。

已复用最新21张和后9张，直接按不同宽／高重新计算PnP和手眼，不覆盖原始数据：

| 板尺寸假设（不含白边） | 原21张位置一致性中位 | 后9张位置一致性中位 |
| --- | ---: | ---: |
| 现模型229.832×172.374 mm | 18.95 mm | 17.86 mm |
| 历史宽高229×173 mm分别使用 | 17.78 mm | 18.19 mm |
| 假设最新仍是214×160 mm | 18.71 mm | 18.69 mm |

这些只是尺寸假设检验，不是对实体尺寸的测量，也不能据误差最小反推板宽高。说明两种已知尺寸替换均未单独解决问题，但仍必须核实当前板尺寸才能给出可信外参。角点重投影小并不能验证米制尺寸：整板等比例放大时，PnP可以把距离一起放大而保持同样的投影。

另已检查：放开焦距、主点和五项畸变后，后9张误差仍约18.3 mm；只改一个内部关节的固定零位也未能解释全部记录。多关节同时拟合会降低部分训练误差，但现有姿态不足以独立验收，未应用任何关节补偿。第6／7轴转角接近只能检查这次相对转动，不能验证整机零位、所有轴线方向或绝对位置准确。
旧RGB-D九次记录中，用原始深度平面代替纯PnP估距后也未得到整体一致；这只是3个独立姿态上的诊断拟合，不能拿它的残差代替当前手眼的独立验证。

证据文件：`calibration/orbbec_oldboard_check_20261102_172656/board_dimension_assumption_audit.json`、`full_intrinsics_pose_audit.json`、`calibration/orbbec_sync_check_20261102_175351/joint_zero_crosscheck.json`、`logs/orbbec_handeye_diagnosis/depth_plane_pose_audit.json`。
**尺寸问题现已回复：用户确认最新21张和后9张的实体板与历史尺寸一致。** 按约229×173 mm、黑色标记20.95 mm记录；保留历史测量精度，不再重复要求量板或重拍。运行外参仍未启用。

### 0.9 已确认的输入矛盾：不使用板尺寸也能复现

用户另确认：最新21张每次按S时，已松手，机械臂使能并保持位置。当前不支持把问题直接归因于手扶机械臂。

已从原始照片重新检测角点，与保存坐标逐点完全一致，未发现“保存照片和角点是不同帧”的证据。再遮去标记内部，用独立的棋盘角点算法重算，后9张的一致性误差仍约17.9 mm；该算法在原21张中成功19张，属于诊断对照，没有删除失败帧或覆盖原结果。

更直接的检查是：只使用两张图片中同名角点的位置和RGB内参，求平面图像变换并分解相对运动，不输入格长、标记边长或手眼外参。列出所有满足可见点条件的解，取与机器人转角最接近的解作为乐观下界：

| 样本ID（从0开始） | 机器人相对转角 | 图像中最接近的有效转角 | 差值 |
| --- | ---: | ---: | ---: |
| 06与14 | 7.98° | 14.97° | 6.99° |
| 05与14 | 23.55° | 30.58° | 7.03° |
| 04与13 | 10.00° | 16.13° | 6.13° |

图像平面变换的对应点误差约0.2–0.3 px。用板外背景特征估计，06／14的有效转角为约14.98°或38.97°，05／14背景本质矩阵估计约30.51°，与标定板图像结果接近，仍未匹配机器人记录。背景检查受平面退化与特征匹配影响，不是独立实物测量，但暂不支持“只是板动了”的解释。
21张共有210个配对，其中138对即使选最接近的可见解，也与机器人转角差超过1°。配对共享21张数据，不能当作210次独立实验。这种输入不相容在手眼求解之前就已存在；更换固定手眼矩阵或板的米制尺寸不能直接修复它。该判断仍依赖RGB内参、板的平面性与安装刚性，不能据此宣布某个机械部件损坏。

只读实机还查询到固件自报 `1.121`，官方SDK的 `resolve_firmware_profile` 映射为 `V112`，此前命令配置为 `V111`。这是待统一的配置差异；但源码确认V112继承完全相同的 `get_flange_pose` 和 `get_joint_angles`，切到V112只读复查时，名义FK与法兰反馈仍差约0.010 mm、0.00064°。因此**不能把切换V112宣称为18 mm问题的修复**；本次没有更改主流程运动配置、控制模式、关节零位或手眼外参。
参考：[官方固件差异表](https://github.com/agilexrobotics/pyAgxArm/blob/master/docs/nero/firmware_reference.md)。

证据：`calibration/orbbec_oldboard_check_20261102_172656/dimension_free_homography_audit.json`、`background_motion_audit.json`、`independent_corner_audit.json`；固件查询与只读反馈在 `calibration/orbbec_sync_check_20261102_175351/firmware_readonly_check.json`、`v112_readonly_feedback_audit.json`。
图表在 `artifacts/orbbec_handeye_diagnosis_20261102/motion_consistency.png`／`.pdf`，汇总在同目录 `diagnostic_summary.json`。
当前已定位到“图像运动与机器人姿态记录不相容”，尚未定位到产生差异的具体接口或物理环节。原21张没有关节记录，不能离线追溯当时每个关节的实物／编码器状态；不据此丢掉异常样本、强行给出已验收外参或继续盲目重采整套。

### 0.10 先推进流程联调：误差对照与奥比手势窗口预演

用户提供的旧标定截图中，7.34／7.67 mm与1.33／1.19°对应旧JSON的**均值**，截图列名写成中位数。下面统一使用日志均值，不把两种统计混比：

| 数据 | 平移均值 | 平移最大 | 旋转均值 | 旋转最大 |
| --- | ---: | ---: | ---: | ---: |
| 旧TSAI记录 | 7.34 mm | 15.25 mm | 1.33° | 2.48° |
| 旧HORAUD记录 | 7.67 mm | 15.61 mm | 1.19° | 2.11° |
| 新奥比21张训练一致性 | 21.04 mm | 46.83 mm | 2.44° | 5.22° |
| 新奥比后9张留出内部一致性（3个独立姿态） | 15.81 mm | 20.53 mm | 0.81° | 1.16° |

旧表与新21张的位置指标相差明显；后9张覆盖更小，也不能替代完整工作空间的精度验证。以上均为固定板一致性，不是实物TCP接触误差。暂停追求标定优化、先联调软件流程，不等于这份外参已满足闭爪／贴桌接近或真人跟随精度。

先运行下面的**真实奥比画面＋模拟机器人后端**，检查人手RGB-D识别、可视化和手势状态逻辑。临时候选仅通过命令行传给这次预演；没有填写到运行YAML，不连接CAN，不执行机械臂运动或夹爪开合：

```bash
cd /home/yx/yx/panda_ws/src/whole-grasp
conda activate grasp_yolo26
bash scripts/run_grasp_safe.sh \
  --config config/pipeline_orbbec.yaml --phase servo \
  --handeye calibration/orbbec_oldboard_check_20261102_172656/handeye_candidate.json \
  --target-source mediapipe --hand-camera-center \
  --servo-preview --servo-diagnostics \
  --hand-depth-radius 8 \
  --delivery-distance 0.05 \
  --gesture-handover --gesture-preview-grip-width-mm 40
```

窗口显示手部骨架、深度及交接状态，Q／Esc退出。40 mm是模拟夹持开口。模拟机器人使用零关节角，不能据基座坐标或交接距离状态验证实机精度；本项也不等于已经跑通真机抓取→跟随→交接。
真机串联仍需先确认新相机在实际抓取／跟随位置的定位偏差可接受；不要直接用第7项旧D435配置运行已换奥比的设备。

### 0.11 用户指定9月1日外参：离线应用对照已完成

对照文件为 `config/2026-09-01_19-08-35_calibration.json`，其定义明确是米制的相机到法兰变换 `T_flange_camera`，按原方向直接使用，没有重拟合外参、取逆试分数或删样本。下表固定使用同一批保存的图像位姿与机器人反馈：

| 数据与参考 | 当前11月2日候选：平移均值／最大 | 9月1日外参：平移均值／最大 |
| --- | ---: | ---: |
| 最新21张，各自内部板位姿一致性 | 21.04／46.83 mm | 17.80／37.99 mm |
| 后9张，以各外参在前21张估出的板位姿为固定参考 | 17.69／22.39 mm | 12.62／18.05 mm |

平移有改善，但方向精度变差：最新21张的旋转均值2.44→3.36°，后9张固定参考的旋转均值1.00→2.34°。
仅在后9张内部重新居中，平移均值15.81→10.96 mm；该口径比固定前21张参考宽松，不能把两个数混用。
两次单轴核对也复用原始记录：第7轴前后板位置变化11.64→3.55 mm，第6轴14.28→8.90 mm。

结论为“位置改善、旋转变差”，不能概括成整体已恢复到旧标定精度。该文件自身记录的平移均值4.19 mm来自9月旧数据，不能套给当前安装。
报告：`artifacts/orbbec_handeye_diagnosis_20261102/compare_20260901_extrinsic.json`。本次只做离线对照，未连接机械臂、未修改YAML里的抓取／伺服外参路径。

### 0.12 清理后下一步：9月外参的单视角局部实机检查

2026-11-02已清理旧参数调试图像／深度／点云4874个文件，逻辑大小约7.21 GiB；结果JSON、轨迹记录、报告明确引用的文件保留。`ablation_v3_nero_refine`、近期`banana_hri`、全部`calibration`、奥比诊断和力矩数据的清理前后文件清单一致。
删除清单在 `artifacts/storage_cleanup_20261102/plan.json`，执行结果在同目录 `result.json`。被删除的旧调参原始图像和深度不能完整重放，正式A–H原始数据未删除。

外参暂用用户指定的 `config/2026-09-01_19-08-35_calibration.json`，通过CLI仅作用于本次试验。先验证当前桌面工作区，不继续逐轴试转或重采整套。单视角、GraspGenX、不加MGPC；保留已有桌面参考修正、碰撞和力矩检查。V112来自实机固件1.121的SDK映射，不是外参补偿。
**下面会真实移动机械臂，到抓取接近终点拍照，但不闭爪，不进入人手跟随／交接。** 现场从侧面确认双指已到香蕉两侧的合适夹持高度、没有偏到一边，且与桌面有实际间隙。软件完成本身不代替这个观察。

```bash
cd /home/yx/yx/panda_ws/src/whole-grasp
conda activate grasp_yolo26
sudo ip link set can0 up type can bitrate 1000000

MOTION_SUPPORT_RELEASE_HEIGHT=0.025 bash scripts/run_grasp_safe.sh \
  --config config/pipeline_orbbec.yaml --phase grasp --execute \
  --calib-json config/sept01_joint_refined_candidate.json \
  --nero-firmware V112 \
  --target-class banana \
  --yolo-weights models/food5_visible_v3_full_reviewed_corrected_best.pt \
  --active-view --no-active-multiview-gate \
  --active-center-height 0.35 --active-center-roll-plan-candidates 1 \
  --no-active-center-optical-relaxation --active-capture-settle-seconds 1.0 \
  --active-side-view-count 0 --grasp-required-view-count 1 \
  --grasp-table-reference-z 0 \
  --preview-cloud \
  --trajectory-executor mit --mit-q-log \
  --mit-max-joint-vel-deg-s 8 --mit-max-joint-accel-deg-s2 8 \
  --grasp-generator graspgenx --nero-grasp-refine --graspgenx-contact-advance auto \
  --grasp-contact-check-only \
  --no-mgpc \
  --save-dir logs/orbbec_local_check/genx_contact
```

本命令的参数与标定文件格式已离线核对，尚未执行实机验证。正常完成时按现有接触检查流程拍照并退回；保护中断时可能停在当前位置，以终端状态为准。该轮现场位置确认通过后，再安排同场景闭爪试验，然后单独验证人手跟随，最后串联全流程；当前没有将任何局部检查称为完整精度验收。

### 0.13 在9月外参上修正旋转：离线拟合与留出对照

可以用旧外参作初值／先验，并用今天的数据更新旋转；已完成离线试算。使用最新21张拟合，9张／3个独立姿态及单轴运动不参与拟合。保留全部样本；这些诊断记录此前已用于其他比较，因此下表不是全新的最终验收集。
固定参考口径与0.11一致：对每个外参，使用21张估出的固定板位姿核对后9张。

| 方案 | 后9张平移均值 | 后9张平移最大 | 后9张旋转均值 |
| --- | ---: | ---: | ---: |
| 9月外参原样 | 12.62 mm | 18.05 mm | 2.34° |
| 固定相机位置，只拟合旋转 | 15.80 mm | 26.29 mm | 1.13° |
| 以旧外参为先验，小幅联合调整位置与旋转 | 13.55 mm | 16.42 mm | 2.14° |

第一种更新的旋转改动约7.87°，方向一致性改善，但位置恶化。第二种联合更新的旋转改动约2.10°、外参平移改动约0.45 mm，降低最大位置误差与旋转均值，但平均位置误差增加约0.93 mm。先验是软约束，2°／3 mm是先验标准差，不是硬截断阈值。
旋转与位置不能独立评价：相机坐标中的目标也会随旋转修正而移动；即使外参平移向量不变，基坐标目标位置也会改变。

单独保存的试验候选是 `artifacts/orbbec_handeye_diagnosis_20261102/sept01_joint_refined_candidate.json`，状态为 `unvalidated_trial_only_not_promoted`。抓取与伺服两个加载器均已核对其矩阵／四元数一致；没有覆盖9月原文件，没有改运行YAML，0.12的命令仍使用9月原文件。
完整拟合、不同先验和单轴对照结果在同目录 `sept01_rotation_refinement.json`。现有跨姿态输入不相容仍在，这次只是有取舍的候选更新，尚未证明实际抓取或跟随精度改善。

上述到位对照已完成，暂停重复执行。两次都包含桌面配准和垂直抓姿修正，且分别重新生成候选，因此不能用实机外观相近来判断原始外参是否更准。

### 0.14 当前只处理标定：保存输入的不相容下界

未修改抓取、推进量、IK、控制或碰撞参数。后续标定验收只使用原始相机观测、机器人位姿记录及未参与拟合的验证记录；不以试抓结果替代精度验证。
按当前21张保存的位姿，sample06／14的机器人相对转角7.981°、相机相对转角14.991°，差7.010°。固定手眼变换保持相对旋转角大小，因此对这批保存位姿，无论怎样拟合一个固定外参，至少有一帧的板方向一致性误差不小于3.505°。这是输入数据条件下的下界，不证明某个硬件损坏，也不等于实际TCP误差。
证据及推导在 `artifacts/orbbec_handeye_diagnosis_20261102/calibration_input_error_bound.json`。补充固定桌面平面约束的外参试算也未取得足以验收的改进，记录在同目录 `known_table_plane_calibration_audit.json`。当前没有接受任何新修正版或覆盖原外参；需要先解决标定输入／位姿对应的不相容，不能继续靠改JSON或调整抓取参数来声称修好。

### 0.15 按重投影误差排除Top2／Top3：已执行离线对照

按用户要求，从原始照片重新检测角点、求PnP、逐角点计算欧氏像素误差，导出各帧均值／RMS／中位／P95／最大值，并核对原图SHA-256。原图及原始`samples.json`保留，排除仅作用于本轮派生求解，未修改其他模块。

排序表位于 `artifacts/handeye_reprojection_trim_20261102/`：

- `current21_reprojection_rank.csv`：最新21张，按帧平均重投影误差降序，附原手眼位置／旋转残差。
- `validation9_reprojection_rank.csv`：后9张，仅导出质量，不参与删帧／拟合。
- `earlier25_corrected_board_reprojection_rank.csv`：最初25张，使用其更正后的另一块板参数单独计算，不混进最新21张。

最新21张帧平均重投影误差0.171–0.306 px，没有大于1.5 px的帧；最高三帧为sample06（0.306 px）、08（0.258 px）、05（0.257 px），ID从0计。每帧的最大单角点误差也均低于0.77 px。
每组重新运行全部五种手眼方法，推荐方法仍均为DANIILIDIS；结果以下列后9张作留出核对，板参考来自各组参与求解的训练帧，没有用留出分数挑选排除帧：

| 求解输入 | 排除sample ID | 训练平移均值 | 后9张平移均值 | 后9张平移最大 | 后9张旋转均值 |
| --- | --- | ---: | ---: | ---: | ---: |
| 全21张 | 无 | 21.04 mm | 17.69 mm | 22.39 mm | 1.00° |
| 按平均误差排Top2 | 06、08 | 20.32 mm | 17.97 mm | 23.10 mm | 1.02° |
| 按平均误差排Top3 | 06、08、05 | 19.81 mm | 17.41 mm | 23.47 mm | 1.06° |

补充按RMS排序的敏感性核对：排06／03后，留出平移均值17.97 mm、最大23.77 mm；再排05后，均值17.54 mm、最大24.37 mm，也未体现稳定改善。此对照只用于排除排序口径差异，没有扩大删除数量或自动覆盖外参。
sample14的平均重投影误差仅0.177 px，但全21张解中的手眼位置残差约44.7 mm；因此这组数据的手眼不一致并未主要表现为高重投影帧。不能把图像拟合误差排序直接当作机器人／手眼异常排序。

每种排除方案的外参JSON、完整五算法日志与验证JSON已分别保存；汇总为同目录 `comparison.csv`／`comparison.json`。后9张只有3个独立姿态，且此前已有诊断使用，不称为全新最终验收集。原始照片未删除，运行外参未替换；这次少量排帧试验尚未修复标定精度。

### 0.16 收敛阈值／最大迭代次数核查：未发现无限求解

已核对正式入口 `tools/calibrate_d435i_charuco.py`：`solve_opencv_methods`依次调用五种`cv2.calibrateHandEye`方法一次，随后比较残差、输出矩阵；没有“直到误差小于0.01才退出”的循环。实测最新21张、五方法全部完成约0.023秒。该API没有`max_iter`、`epsilon`或`timeout`输入参数，向板YAML添加这些字段不会改变它的行为。
两个板YAML里的 `quality.max_reprojection_error_px: 2.0` 是单帧采集／重处理的质量上限，不是优化器停止条件。最新21张平均重投影误差0.171–0.306 px，把该值改为0.1会使这些帧全部不通过质量检查。代码中的相机`while True`用于预览、等待S／Q按键；反馈等待另有5秒超时，不属于外参求解循环。
`tools/charuco_intrinsics.py`中可选的`cv2.calibrateCamera`用于内参诊断，是另一个函数；不能将其迭代条件当作当前外参求解的停止条件。本轮未修改任何收敛、筛选、抓取或控制参数。
当前手眼采集只读取RGB，深度不参与角点／手眼求解。光照可能影响成像质量，但当前记录角点完整、重投影误差较低，没有证据支持因白天光线直接重采整套。
实际问题仍是“矩阵已经求出，留出位姿一致性误差偏大”。运行审计在 `artifacts/handeye_solver_termination_audit_20261102.json`；防止程序挂起与提高外参精度需要分别判断，不能把正常退出当作精度验收。

### 0.17 初值、LM、Huber、姿态分布与角点：23组离线对照

入口为 `tools/audit_handeye_methods.py`，只读已有21张旧板照片及机器人位姿，用后9张作留出核对；不连接相机或CAN，不修改原图、采集记录、运行外参和抓取参数。后9张只有3个独立姿态，且此前已用于诊断，不能当作新的最终验收集。
结果在 [完整报告](artifacts/handeye_method_audit_20261102_complete/report.html)、[方法表](artifacts/handeye_method_audit_20261102_complete/summary.csv)、[逐帧质量](artifacts/handeye_method_audit_20261102_complete/frame_quality.csv)。原尺寸角点图在同目录 `corners/`，留出预测投影在 `validation_projections/`。

1. **初值／局部最优**：现有正式求解本来就运行Tsai等五种方法，不以随机矩阵启动LM。本轮新增Tsai／Park／Horaud／Andreff／Daniilidis／9月外参／单位矩阵7种初值的LM精修，全部正常收敛，目标函数值几乎一致；最终外参相对第一组的差异小于0.001 mm／0.0002°。没有证据支持这批数据主要被差初值卡住；多初值一致也不能数学上证明全局最优。
2. **Tsai直接解／LM／Huber**：均已实际运行。位姿残差按5 mm和1°缩放；Huber使用TRF，并同跑TRF+L2隔离求解器差异。位姿Huber尺度0.5／1／2／5、像素Huber尺度1／3／10都作对照，未据留出分数挑选生产参数。所有非线性求解均正常终止，单次小于0.03秒，评估预算上限500；它们没有无期限等待某个绝对误差阈值。SciPy的LM只支持L2，见[官方说明](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html)。
3. **重复／多样性**：没有重复图像哈希，只有sample19／20这一对近重复姿态，保留图像误差较低的20后重算。距离范围289–551 mm、板倾角6–33°、机器人姿态最大相对转角63.2°；角点覆盖横向112–1163、纵向134–533像素（图像1280×720），垂直覆盖仍不均匀。旋转激励奇异值1.258／1.162／0.607，不呈明显近零退化。另按训练集运动／距离／画面位置跨度选12／15张重算，没有获得稳定改善；“重复帧多会拖崩”不符合这批数据的现象。
4. **角点**：已重新检测原图、核对SHA、逐角点ID及保存坐标。每帧33–35角点，重检测与保存角点的最大差为0像素；帧平均重投影0.171–0.306像素，全部帧的最大单角点误差0.767像素。可视化未见明显错配；这不排除系统性的视觉位姿偏差，也没有虚构一个检测器未提供的“角点置信度”来删图。

下表统一使用训练帧确定的固定板参考，评价留出帧中板位置／方向的一致性；数值不是尺量TCP误差：

| 方法 | 留出平移均值 | 留出平移最大 | 留出旋转均值 |
| --- | ---: | ---: | ---: |
| 原全21张Daniilidis | 17.69 mm | 22.39 mm | 1.00° |
| Tsai直接解 | 18.27 mm | 24.76 mm | 1.24° |
| Tsai→LM，位姿L2 | 15.24 mm | 18.65 mm | 1.77° |
| Tsai→TRF，位姿Huber，尺度1 | 15.56 mm | 19.48 mm | 1.50° |
| Tsai→LM，直接拟合图像角点 | 13.85 mm | 16.98 mm | 2.61° |
| 去掉近重复姿态后20张 | 18.07 mm | 22.44 mm | 1.05° |
| 优先保留跨度大的15张 | 17.12 mm | 22.10 mm | 1.02° |
| 优先保留跨度大的12张 | 17.50 mm | 27.01 mm | 1.12° |

JSON另存优化器自身拟合板位置的评估，避免将“重新取训练板均值”与原始优化目标混淆。直接像素L2拟合在优化器自身的板位置下，训练角点RMS仍约22.50像素；不能将单帧PnP约0.2像素说成固定手眼外参的投影精度。留出投影图中绿色为实测角点、红色为训练板参考和外参的预测，黄色线显示偏差；验证照片没有参与该行外参或板参考拟合。
本轮没有修复外参，也没有证明某个硬件损坏。主要未解项仍是保存照片与机器人位姿的跨姿态不相容（0.14），继续换初值、删几张清晰照片或放宽停止阈值不能据此宣称解决。新离线优化用已知真值的模拟数据验证了位姿／像素两种模型、Huber异常样本和超预算退出，4项测试通过。

复跑同一离线审计（覆盖该报告目录下的派生报告，不改原始采集）：

```bash
cd /home/yx/yx/panda_ws/src/whole-grasp
conda activate grasp_yolo26
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/matplotlib python tools/audit_handeye_methods.py \
  --output artifacts/handeye_method_audit_20261102_complete
```

### 0.18 能否按时间同步误差排除坏帧：元数据核查

可以在有可靠曝光时间、机器人时间及运动记录的前提下排除不同步帧，但不能将手眼残差大直接称为不同步。当前采集直接使用Orbbec SDK和pyAgxArm，不通过ROS `joint_state`、`image_view snapshot`或快门软触发命令；按S之后读取视频流帧，并在前后读取机器人反馈。

已逐帧检查原始记录，结果保存在 `artifacts/handeye_sync_metadata_audit_20261102/audit.json` 和 `frames.csv`：

- **训练21张**：有相机设备时间和两次法兰反馈时间，但没有保存图像回调／读取时刻、拍照前后关节角及前法兰位姿。不能准确追溯曝光相对于机器人运动的时间误差；缺少元数据不等于这些帧一定不同步。
- **后9张**：前后关节角及法兰反馈位姿均一致；图像回调、读取时刻均在机器人读取请求区间内，帧号／设备时间正常前进。回调到读取完成耗时0.928–2.302 ms。没有发现可据这些字段标记的时序异常，但两端静止不证明中间及曝光前的整个延迟窗口都静止。
- **重要口径**：这0.928–2.302 ms不是曝光到机器人位姿的同步误差。相机设备时钟与主机时钟没有在记录中映射到共同曝光时基；所有帧的绝对同步误差均应标为未知。此前 `timing_rotation_audit.json` 的相对时钟残差也不是测得的绝对同步延迟。官方时间戳说明区分设备采集时刻、主机接收时刻与全局时间映射，见[Orbbec说明](https://orbbec.github.io/OrbbecSDK_v2/tools/timestamp_tracker/)。

目前没有可确认的不同步帧，因此排除列表为空，未伪造一次“删同步坏帧后的精度提升”，也未修改采集、抓取或运行外参。后9张仅3个独立姿态，不能单独替代完整训练集。时间同步仍未彻底排除，但现有数据不支持将其直接认定为主因。
固定法兰→TCP变换可以影响平移及旋转轴的表达，但不会改变相对旋转角的大小；仅凭“TCP原点不一样”不能解释8°与15°。停稳采集也并非必须依赖硬触发：关键是曝光、传输和反馈对应的整个窗口处在同一静止姿态。若继续验证同步，应先以少量同姿态连续记录查缓存／时间映射及关节流，不先删除原图或重采整套。

### 0.19 当前执行：20秒连续图像／关节同步诊断

**模式更正：旧标定采集时为示教模式，现在为纯使能保持。** 先在当前状态做下面的静止检查，程序不发送使能、模式切换、关节运动、停止或夹爪指令；`--arm-state`只是人工标签。相机使用与标定相同的Orbbec RGB采集／最新帧队列，关闭深度；不更改相机时钟设置、外参、求解器或抓取参数。

保持当前机械臂静止，先退出其他占用相机的程序，然后运行。默认打开RGB可视化，20秒自动结束；Q／Esc、关闭窗口或Ctrl+C可提前结束并保存已采数据。此检查不需要标定板识别，也不需要手动换姿态。

```bash
cd /home/yx/yx/panda_ws/src/whole-grasp
conda activate grasp_yolo26
python tools/diagnose_handeye_sync.py \
  --config config/pipeline_orbbec.yaml \
  --arm-state enabled-hold \
  --duration 20 --joint-hz 100
```

保存至 `calibration/orbbec_sync_stream/<时间>/`：

- `rgb.avi`：连续消费的RGB帧；`frames.jsonl`逐帧记录视频索引与真实时间，不用视频名义播放速率判断真实采样时间。
- `callbacks.jsonl`：所有记录期间的有效RGB回调帧号、设备微秒时间、SDK系统接收时间、全局时间（若SDK当前可用）、主机单调／墙上时钟。
- `joints.jsonl`：并行读取的七关节角、法兰位姿、反馈时标及读取前后时间。100 Hz是请求轮询频率，实际新反馈频率另算，不把重复反馈当新样本。
- `session.json`：人工模式标签及启动时可读的SDK模式状态。`summary.json`：帧号／设备时标回退、向前跳帧数、回调到读取积压、时钟拟合漂移／残差、主机时钟跳变、关节变化及新反馈频率。

向前跳帧可来自最新帧队列丢弃过期帧，不等于返回旧帧；时标回退和帧号回退分别统计。设备时间与主机时间的拟合截距包含不同起点与未知固定传输延迟，不能直接叫相机相对机器人运动的延迟。静止记录能检查时钟和帧输送异常，但缺乏运动激励，不能独立识别固定的图像—运动时间偏移；报告中的 `camera_to_motion_time_offset_s`因此保留null。
**结果不作简单A／B二分**：时钟拟合稳定不保证旧21张可修复，因为旧采集没有保存曝光附近的连续关节轨迹；发现随机积压也不自动证明整套都必须重采。先看能否定位受影响帧及是否有足够有效姿态，再决定保留、对齐或补采，不基于猜测删图。
`--arm-state`只记录操作者提供的模式标签。两种模式的20秒对照现在均已完成，不需要重复静止录制；程序从不自行切换模式或移动机械臂。

#### 22:11 当前纯使能状态的20秒连续诊断：已完成

已实际只读采集，数据目录 `calibration/orbbec_sync_stream/20261102_221153_171795`，占用约34 MB；相机Gemini335、RGB1280×720@30，启动SDK状态为CAN_CTRL／MOVE_J／NORMAL。程序没有发送使能、模式切换、运动、停止或夹爪命令。

| 指标 | 本次结果 |
| --- | ---: |
| 回调／消费／视频帧数 | 601／601／601 |
| 实际相机回调频率 | 29.985 Hz |
| 帧号回退／重复、设备时标不递增、向前跳帧 | 均0 |
| 回调到读取完成 | 中位3.11 ms，P99 6.61 ms，最大6.90 ms |
| 关节读取／新反馈 | 2003／1965次，实际新反馈约98.10 Hz |
| 最大反馈年龄 | 15.35 ms |
| 七关节全程最大角度范围 | 最大0.012° |
| 设备时钟→SDK系统时间的短段仿射拟合 | 相对斜率约−540 ppm，残差峰峰1.42 ms |
| SDK系统时间−SDK全局映射时间 | 中位15.97 ms，范围15.32–17.33 ms |

结果图：[timing.png](calibration/orbbec_sync_stream/20261102_221153_171795/timing.png)；完整字段：[summary.json](calibration/orbbec_sync_stream/20261102_221153_171795/summary.json)。图中第一项是相对首帧的两时钟差，不是曝光延迟。
这20秒没有发现帧回退或明显积压，且关节确实近乎静止。设备与系统的经过时间在这段相差约10.7 ms，能用仿射关系拟合；不应以一个固定常数跨采集会话修正。−540 ppm是这次短段的相对拟合量，不能据此断言长期硬件振荡器误差。
SDK全局时间有效，但其映射依赖SDK；15.97 ms不是独立测得的机器人—图像时间偏移。当前仍无法直接抢救旧21张，也没有证据宣告它们全部作废。旧采集为示教模式，与本次状态不同，不能直接外推。

#### 22:17 示教模式的20秒对照：已完成

SDK实际报告 `TEACHING_MODE／MOVE_J／NORMAL`，与操作者模式标签一致。数据目录为 `calibration/orbbec_sync_stream/20261102_221721_406265`，约35 MB；完整结果见 [summary.json](calibration/orbbec_sync_stream/20261102_221721_406265/summary.json)。终端的Qt字体目录警告只涉及窗口字体，不影响采图；本次录制正常结束。

| 指标 | 纯使能保持 | 示教模式 |
| --- | ---: | ---: |
| 回调／消费图像帧 | 601／601 | 600／600 |
| 帧号／设备时间回退、重复、向前跳帧 | 均0 | 均0 |
| 回调到读取完成中位／P99／最大 | 3.11／6.61／6.90 ms | 3.92／6.67／17.34 ms |
| 超过两帧周期的读取积压 | 0 | 0 |
| 关节读取／新反馈 | 2003／1965 | 2002／1940 |
| 七关节最大角度变化 | 0.012° | 0°（反馈分辨率内） |
| 设备时间→SDK系统时间拟合相对速率 | −540 ppm | −521 ppm |
| 两时钟线性拟合残差峰峰 | 1.42 ms | 1.27 ms |
| SDK系统时间−SDK全局映射时间中位 | 15.97 ms | 16.18 ms |

示教模式下唯一一帧回调到读取耗时超过10 ms，最大17.34 ms，仍小于30 Hz的一个33.3 ms帧周期；不能将它称为返回旧图像。两段静止记录都没有发现帧回退或明显缓存积压，也不能测出图像相对真实运动的固定延迟。SDK系统／全局时间相差约16 ms，含SDK映射与传输，不是已知的图像—机械臂动作offset。旧21张仍缺少曝光附近连续关节轨迹，不能凭这两段数据回填并宣称修好外参。

示教模式命令保留供复现，本轮已经运行完毕，无需重复：

```bash
python tools/diagnose_handeye_sync.py \
  --config config/pipeline_orbbec.yaml \
  --arm-state teaching \
  --duration 20 --joint-hz 100
```

### 0.20 排除8°／15°异常对后重算：已完成

旧板21张中sample06和14的相对转角：机器人7.981°，视觉PnP 14.991°，相差7.010°。这只能说明至少有一张保存位姿不相容，不能指出具体哪一张有错。按用户要求运行四组：完整21张、仅排除06、仅排除14、同时排除06和14。这里“排除”仅用于派生求解；原始`images/`和`samples.json`均保留。全部使用原有五方法求解器，独立后9张按训练集确定的固定板位姿验证。板全程固定是这一列比较的前提。

| 求解输入 | 训练姿态数 | 后9张平移均值／最大 | 后9张旋转均值／最大 | 剩余最差一对相对转角差 |
| --- | ---: | ---: | ---: | ---: |
| 原21张 | 21 | 17.69／22.39 mm | 1.000°／1.580° | 06／14：7.010° |
| 排除06 | 20 | 18.28／23.99 mm | 1.047°／1.502° | 05／14：6.842° |
| 排除14 | 20 | 16.16／19.66 mm | 1.056°／1.721° | 06／15：5.979° |
| 排除06和14 | 19 | 16.14／20.20 mm | 1.074°／1.646° | 04／13：5.686° |

四组的推荐方法均为Daniilidis。单排06不改善，单排14使留出平移均值降低约1.53 mm但旋转变差，两张同时排除的平移均值只比单排14再低约0.02 mm。两张同时排除后仍有04／13约5.69°的转角冲突，所以这不是一对坏帧导致的全部问题。后9张只有3个独立姿态、此前也用于诊断，因此不能据这1.5 mm的探索性变化宣布外参已改准；候选均未投入抓取主流程。

派生结果：[comparison.csv](artifacts/handeye_pair_exclusion_20261102/comparison.csv)、[comparison.json](artifacts/handeye_pair_exclusion_20261102/comparison.json)。每组另存候选外参JSON、五方法求解日志及后9张验证JSON，便于复核；本轮没有连接相机／机械臂，也没有修改原始图像、运行外参或抓取参数。

### 0.21 下一步：先重建可信的标定输入

复核已采的后9张中三个不同姿态：机器人／视觉相对转角差分别为0.672°、0.904°、0.499°，明显小于旧21张的最差7.010°；但三个姿态不足以求出并独立验收可靠外参。旧批次是否受示教模式或采样时序影响，现有记录无法确定。继续删旧照片或只改求解器不能修正其余姿态的转角矛盾。

使用**同一块旧板**并保持固定，用已有控制方式移动到至少12个不同的观察姿态；每次停稳后在**纯使能保持**状态等待约2秒，再在相机窗口按`S`保存。位置、距离和倾斜方向都要变化；`--candidate-frames 1`让一张照片对应一次采样窗口。采集脚本只读相机和机械臂反馈，不切换模式、不发运动指令。旧的21张和后9张保留原样。

```bash
cd /home/yx/yx/panda_ws/src/whole-grasp
conda activate grasp_yolo26
python tools/calibrate_d435i_charuco.py capture \
  --backend orbbec --serial CP0T963000S4 \
  --width 1280 --height 720 --fps 30 \
  --board config/charuco_board.yaml \
  --channel can0 --firmware V111 \
  --candidate-frames 1 \
  --output "calibration/orbbec_enabled_hold_train_$(date +%Y%m%d_%H%M%S)"
```

保存到`sample 11`即12张，按`Q`退出。先检查这12张的机器人／视觉相对转角是否仍有数度冲突；若有，暂停采集并查记录的法兰位姿与实物运动。若没有，再求手眼候选，并另采至少5个**新的独立姿态**验证板位姿一致性；验证通过后才更新`config/pipeline_orbbec.yaml`中的抓取和伺服外参路径。不要把旧9张的三种重复姿态当成5个独立验收姿态。

### 0.22 11月3日纯使能保持采集结果与下一步

`calibration/orbbec_enabled_hold_train_20261103_142639`保存了14张、每张35个角点；14张图像均存在，帧号严格递增。重投影均值0.136–0.266 px，图像读取年龄0.87–2.36 ms，采样前后最大关节变化0.007°。

14张之间共91对：机器人／视觉相对转角差中位0.604°、最大2.941°（样本07／12，机器人31.530°、视觉28.589°）；旧21张最差对06／14相差7.010°。新批次仍有数度偏差，不能仅凭角点重投影很小判断手眼准确。

原五方法求解器推荐PARK，新候选在 `calibration/orbbec_enabled_hold_train_20261103_142639/handeye_candidate.json`。训练集板位姿一致性：位置均值4.79 mm、最大11.75 mm；旋转均值0.879°、最大1.559°。这些是**参与求解的14张**上的残差，不是独立精度。用相同PARK算法比较，新候选与旧21张候选相差约59.4 mm／6.43°，不能直接覆盖运行外参。

对前一天保存的9张做只读、跨批次参考核对，内部一致性位置均值10.87 mm、旋转均值2.05°；9张仅对应3个不同姿态，且采集日期不同，不能作为最终验收。现在需要另采至少5个**与训练集不同**的姿态，尽量在左右位置、远近和两个倾斜方向上展开，采集过程保持板与相机支架固定。每次移动结束后，纯使能保持、静止约2秒再按`S`。这条命令只采图和读反馈，不控制机械臂。

```bash
cd /home/yx/yx/panda_ws/src/whole-grasp
conda activate grasp_yolo26
python tools/calibrate_d435i_charuco.py capture \
  --backend orbbec --serial CP0T963000S4 \
  --width 1280 --height 720 --fps 30 \
  --board config/charuco_board.yaml \
  --channel can0 --firmware V111 \
  --candidate-frames 1 \
  --output "calibration/orbbec_enabled_hold_validate_$(date +%Y%m%d_%H%M%S)"
```

保存5个不同姿态后按`Q`，记录输出的Dataset目录。验证时使用`validate`命令且不加`--same-board-pose`，先检查新验证集自身的板位姿一致性；若板从训练到验证全程固定，可另加该选项检查相对训练板位姿的绝对偏差。验证前不启用新候选。

### 0.23 11月3日独立验证已完成：奥比试验外参

新验证集为 `calibration/orbbec_enabled_hold_validate_20261103_143804`，8张照片均存在、帧号递增、每张35个角点；重投影均值0.193–0.408 px，采样前后最大关节变化0.001°。与14张训练集没有重复图片。

14张训练求出的PARK外参在这8张新姿态上的**板位姿内部一致性**：位置均值5.98 mm、最大8.85 mm；旋转均值1.49°、最大2.69°。若两次采集间标定板始终固定，相对训练时板位姿的均值为9.10 mm／1.73°、最大12.82 mm／2.25°。验证集28对相对转角差最大3.75°，说明角度误差并未完全消失。这些数值衡量相机／机器人推回的标定板是否落在同一位置，不能等同于夹指的实际到位误差。

按此前“达到旧标定量级先走通流程”的试验标准，已从原候选单独保存 `config/orbbec_handeye_20261103_trial.json`，并将 `config/pipeline_orbbec.yaml` 的抓取与跟随外参都指向该文件。原始候选及采集数据未覆盖；此文件状态标记为`offline_independent_validation_complete_trial_only`。已离线验证两处加载器读出的4×4矩阵相同，最大元素差小于1e-14。下一次奥比实机测试使用 `--config config/pipeline_orbbec.yaml`，先在可视化中核对目标与相机坐标，再判断实际夹指到位情况；不要把这组板一致性数值当作实物精度证书。

### 0.24 新奥比外参的香蕉实机到位检查

这条命令会真实移动机械臂、打开点云和最终抓取姿态可视化，到夹持位置保存 RGB-D／关节反馈，正常完成后沿预检路线撤回。**不闭爪、不抬升、不进入人手跟随。** 相比0.12的历史命令，已去掉9月外参的 `--calib-json` 覆盖项；抓取与跟随外参均从 `config/pipeline_orbbec.yaml` 加载11月3日的试验文件。若保护中断，以终端状态为准，不能假设已退回。现场看两根夹指是否位于香蕉两侧、进入合适深度，以及与桌面是否仍有真实间隙。

```bash
cd /home/yx/yx/panda_ws/src/whole-grasp
conda activate grasp_yolo26
sudo ip link set can0 up type can bitrate 1000000

bash scripts/run_grasp_safe.sh \
  --config config/pipeline_orbbec.yaml --phase grasp --execute \
  --nero-firmware V112 \
  --target-class banana \
  --yolo-weights models/food5_visible_v3_full_reviewed_corrected_best.pt \
  --active-view --no-active-multiview-gate \
  --active-center-height 0.35 --active-center-roll-plan-candidates 1 \
  --no-active-center-optical-relaxation --active-capture-settle-seconds 1.0 \
  --active-side-view-count 0 --grasp-required-view-count 1 \
  --grasp-table-reference-z 0 \
  --preview-cloud \
  --trajectory-executor mit --mit-q-log \
  --mit-max-joint-vel-deg-s 8 --mit-max-joint-accel-deg-s2 8 \
  --grasp-generator graspgenx --nero-grasp-refine --graspgenx-contact-advance auto \
  --grasp-contact-check-only \
  --no-mgpc \
  --save-dir logs/orbbec_handeye_trial/contact_check
```

### 0.25 11月3日实机复测：一侧贴近与新奥比夹指遮罩

0.24这次规划的香蕉点云在两指之间基本居中：算法内部的两侧接触点23/22、宽度方向重心修正0.2 mm；实物照片中，按现场机器人基坐标的方向约定，**右指更靠近香蕉**。日志里的左右侧编号不能直接对应照片左右。机械臂到位的横向误差约1 mm，主要是竖直方向比计划高3.2 mm，不能解释明显的横向偏差。现有记录还无法在手眼外参、目标深度位置和夹爪实际几何之间定唯一原因；本次先保留原外参与抓取推进量。旧命令的最终位置误差3.28 mm略超过`--grasp-close-pos-tol`默认3 mm，因而**在拍最终照片前退出，没有闭爪，也没有正常退回**，并非电子急停。

奥比旧配置未开启自身夹指遮罩。现用两张100 mm张开状态的奥比RGB-D记录建立`config/orbbec_self_mask_20261103_100mm_trial.json`，并在两张记录上离线复核：分别剔除13,397和12,058个有效夹指深度像素，与香蕉分割掩膜均无重叠。新遮罩已经在`config/pipeline_orbbec.yaml`启用，运行时会逐帧核对相机参数和夹指轮廓。它只剔除画面下缘已测得的两根夹指；画面上方零散点若来自电线或远处物体，不会被当作夹指一并删除。

若机械臂仍停在上次的香蕉附近，先用现场熟悉的受控操作恢复到观察位，再运行。以下仍然**不闭爪**；唯一新增的到位参数是4 mm，供最终到位拍照与关节反馈记录使用，不放宽桌面碰撞检查。请保留最终`contact_check/*/color.png`与`contact_pose.json`，以及前面观察帧的`camera_self_mask_overlay.png`、`camera_self_mask_report.json`和终端输出，以便把实物偏差与预测姿态直接对比。最终到位快照只采集RGB-D，不运行点云遮罩。

```bash
cd /home/yx/yx/panda_ws/src/whole-grasp
conda activate grasp_yolo26
sudo ip link set can0 up type can bitrate 1000000

bash scripts/run_grasp_safe.sh \
  --config config/pipeline_orbbec.yaml --phase grasp --execute \
  --nero-firmware V112 \
  --target-class banana \
  --yolo-weights models/food5_visible_v3_full_reviewed_corrected_best.pt \
  --active-view --no-active-multiview-gate \
  --active-center-height 0.35 --active-center-roll-plan-candidates 1 \
  --no-active-center-optical-relaxation --active-capture-settle-seconds 1.0 \
  --active-side-view-count 0 --grasp-required-view-count 1 \
  --grasp-table-reference-z 0 \
  --preview-cloud \
  --trajectory-executor mit --mit-q-log \
  --mit-max-joint-vel-deg-s 8 --mit-max-joint-accel-deg-s2 8 \
  --grasp-generator graspgenx --nero-grasp-refine --graspgenx-contact-advance auto \
  --grasp-contact-check-only --grasp-close-pos-tol 0.004 \
  --no-mgpc \
  --save-dir logs/orbbec_handeye_trial/contact_check_masked
```

这次0.25实测的终端状态为`contact_checked`：最终RGB-D已保存，未闭爪，按预检路线撤回。右指仍较靠近香蕉，深度没有明显变化。所选GenX候选原始分数0.9745；程序沿夹爪开合轴修正了10.9 mm，并用修正后的姿态完成整条路线检查。终点法兰横向误差小于1 mm、竖直方向高约2.2 mm，不能解释持续的右侧偏差。

离线对照已保存到`artifacts/orbbec_mask_residual_20261103/`：将规划目标点投影到最终RGB，点群在同等图像行上比实际香蕉偏左约58–122像素；将CAD夹指投影到同一照片，轮廓与真实夹指大体重合。观察位的规划点云回投到初始／中心RGB，各有80.3%点落在香蕉掩膜中。因此横向偏差主要发生在**观察位目标坐标传到最终夹持位**这段链路；仅凭这次数据仍不能把手眼外参、机器人跨姿态位姿误差与目标表面深度偏差再拆开。最终相机非常靠近香蕉，最终深度图读数与规划几何不一致；此次对位依据RGB，不能用最终深度直接求固定补偿量。不要据此直接给所有姿态加固定横移。

本次两个观察帧的夹指遮罩分别剔除13,755和10,491个有效深度像素。红框中保留下的稀疏高点投影到观察图像时位于灰色桌面区域、夹指遮罩之外；它们更像深度／融合残点，继续扩大夹指遮罩会误删场景。最终到位快照本身只采RGB-D，不生成供规划用的点云。

### 0.26 定位右指偏差：同一固定标记的两姿态只读对照

本次不抓取、不改外参。把**原来那块8×6、`DICT_5X5_100`标定板**固定在香蕉原位置附近；同一黑色标记在下面两个姿态中都要看得见，期间不要移动板。先退出抓取程序，保持相机、机械臂连接只由本脚本使用。脚本**只读相机和CAN反馈，不发送机械臂使能、移动、停止或夹爪命令**；姿态调整由现场熟悉的受控示教／点动完成。

1. 观察位：相机朝向标定板，待机械臂静止，确认窗口里有标记ID及有效深度；按S采3次。第一次成功后程序锁定该标记ID。
2. 接近位：保持标定板固定，用现有受控操作改变机械臂到接近抓取时的腕部朝向，但让相机离标记仍足够远；窗口里的中心深度建议在300–450 mm，不能显示`DEPTH MISSING`。停稳后对**同一ID**按S再采3次。若必须接近到深度不可用，停在较高位置即可，不要为了测试贴近桌面。
3. 采满6次自动退出。把终端输出或新生成的`logs/orbbec_grasp_pose_audit/handeye_aruco_consistency_*.json`发来；我会分别比较前3次与后3次标记在机器人基坐标的平均位置，并检查RGB PnP与深度是否一致。两组三次各自稳定而组间明显偏移，说明观察位到抓取位的坐标链随姿态变化；若组间一致，就继续查香蕉目标点云与候选接触区。

```bash
cd /home/yx/yx/panda_ws/src/whole-grasp
conda activate grasp_yolo26
sudo ip link set can0 up type can bitrate 1000000

python tools/verify_handeye_aruco_consistency.py \
  --config config/pipeline_orbbec.yaml \
  --calib-json config/orbbec_handeye_20261103_trial.json \
  --nero-firmware V112 \
  --aruco-dict DICT_5X5_100 \
  --marker-length-mm 20.95 \
  --samples 6 \
  --save-dir logs/orbbec_grasp_pose_audit
```

这个20.95 mm是旧板标记外框的历史估计值，影响RGB PnP测距；最终判断还要结合深度结果，不将其当成新的精密实测尺寸。

**11月3日16:05这组六次结果：**同一ID 23，前3次和后3次各自静止。深度反投影的固定标记基坐标组间差`[+0.66,+17.82,+9.35] mm`，合计20.13 mm，沿这次香蕉夹爪开合轴约−17.36 mm；各组内部最大两两差分别1.17和0.24 mm。整板35角点RGB PnP独立复算仍有`[-1.25,+18.74,+4.71] mm`组间差、姿态差约1.80°；每张平均重投影误差0.236–0.247 px。机器人报告转角20.598°，视觉整板转角20.528°，幅度只差0.070°。照片中的香蕉、红块、蓝块反投影后也分别沿基坐标Y变动约+16.4、+11.8、+13.8 mm，故不能把标记板意外移动或单个标记误检当成主要解释。

已离线把相同六帧代入当前11月3日、9月1日原版、9月1日修正版和11月2日候选外参：整板组间差分别19.4、17.5、21.6和24.7 mm；换一个旧JSON没有实质消除偏差，运行配置**没有切换外参**。这些姿态只移动约88 mm、转约20.6°，还没有到最终抓取位，已经测到显著的跨姿态坐标不一致。它能够解释香蕉持续靠右指的一部分，但不能仅凭六帧把误差唯一归到手眼旋转、机器人名义运动学或深度对齐。下一步要在标定板保持固定的前提下，采覆盖目标工作区的多个**不同**姿态重算外参，并保留此六帧和原8个验证姿态作独立验收；六次采样只有两个不同姿态，不能单独解完整手眼外参。

### 0.27 下一步：在抓取工作区补足不同姿态，离线验收新外参

板继续固定在0.26的位置，整个新采集期间不要挪动。相机支架、夹爪也不动。旧六帧作为独立对照，不混入训练集；旧14张板位置不同，也不能直接与新照片拼成单块固定板的数据集。此步骤只采集和离线计算，不改运行外参、不执行抓取。

使用现场已有的受控方式调整机械臂。选择约12–15个**不同**的静止姿态，包含原观察朝向、0.26的接近朝向，以及两者之间和两侧的腕部朝向；不要在同一姿态连续按S凑数量。每次停稳、预览显示`quality=OK`后按S保存。保持板完整可见；不需要靠近桌面或物体。采集程序仅连接相机和只读关节反馈，不发运动或夹爪指令。窗口按Q结束。

```bash
cd /home/yx/yx/panda_ws/src/whole-grasp
conda activate grasp_yolo26
python tools/calibrate_d435i_charuco.py capture \
  --backend orbbec --serial CP0T963000S4 \
  --width 1280 --height 720 --fps 30 \
  --board config/charuco_board.yaml \
  --channel can0 --firmware V112 \
  --candidate-frames 1 \
  --output calibration/orbbec_grasp_region_train_20261103
```

采完后在同一环境离线求候选，并用11月3日独立的8姿态数据验证。`validate`输出的是8姿态内部一致性；还要将新候选代入0.26保存的两姿态六帧，检查原来约20 mm的跨姿态漂移是否下降。**两项都通过后**再考虑切换`config/pipeline_orbbec.yaml`；只看训练误差或重投影误差不能决定是否切换。

```bash
python tools/calibrate_d435i_charuco.py solve \
  --dataset calibration/orbbec_grasp_region_train_20261103 \
  --output calibration/orbbec_grasp_region_train_20261103/handeye_candidate.json

python tools/calibrate_d435i_charuco.py validate \
  --dataset calibration/orbbec_enabled_hold_validate_20261103_143804 \
  --calibration calibration/orbbec_grasp_region_train_20261103/handeye_candidate.json
```

**11月3日这轮已完成，不需重跑上面的求解命令。**实际保存16张，均检出35角点；单帧重投影误差0.148–0.295 px，图像读取年龄0.92–2.21 ms，保存时关节变化接近0°。用户确认本轮标定板全程没有被碰到或移动。

新PARK外参在16张训练帧上的板位置残差均值9.81 mm、最大14.28 mm；在此前独立8姿态上均值10.34 mm、最大23.15 mm。当前运行外参在该独立8姿态上是均值约5.98 mm、最大8.85 mm。旧六帧固定标记的两姿态位置差从当前外参的20.13 mm降到新PARK的14.63 mm，仍明显存在。用新数据中独立测试较好的TSAI方法，独立8姿态均值8.1 mm、最大15.1 mm，旧六帧差16.0 mm；没有一种直接求解方法同时通过两项对照。

再以新PARK为初值做普通最小二乘和Huber非线性优化：Huber结果在训练集均值8.28 mm、独立8姿态均值5.84 mm／最大12.0 mm，但旧六帧差仍18.62 mm。新16帧中共有120个姿态配对，其中38对的机器人与RGB视觉相对转角大小差超过1°，最大2.44°；**固定手眼变换无法改变相对转角大小**。同一批照片的平面PnP第二解重投影误差约5.4–9.4 px，当前解约0.15–0.30 px，不能靠切换PnP分支解决。

结论：**新候选未通过独立验收，不接入`config/pipeline_orbbec.yaml`。**这轮结果也不支持继续盲目增加照片或只调优化器参数；先用现有数据核查视觉姿态与机器人位姿为何在不同姿态下不相容，再决定是否需要新的采集方式。以上比较均为离线计算，没有运动或夹爪命令。

如果继续定位，请复用0.7的`verify_handeye_rotation_imu.py`，但这次在保持板可见和安全间隙的前提下，记录**两个不同腕部朝向之间的完整运动**，而不是重复此前只动单关节的三轮。新数据中样本08和13的机器人相对转角14.70°、整板视觉相对转角12.26°，相差2.44°；同时采陀螺仪才有机会区分视觉PnP／板问题与机器人实际运动问题。此工具只读，不发送运动命令；不需要再采整套手眼数据。若陀螺仪接近视觉，优先查机器人实际运动与名义反馈；若接近机器人，优先查视觉姿态／标定板；若三者互不一致，则暂不做单一归因。

**17:05补充实测：**`calibration/orbbec_rotation_audit/20261103_170549_612662/result.json`记录的机器人／视觉／陀螺仪相对转角分别为33.518°／33.518°／33.641°，该次运动的转角大小一致。当前外参计算的固定板起终点位置差5.74 mm、完整板朝向差1.53°；新16张候选分别为7.60 mm、0.42°。因此不能只根据三个**转角大小**相等就宣称外参或完整三维运动已对齐。该次主要变化是关节1约−37.4°，关节5约−2.8°、关节6约−4.0°；出现2.44°不一致的训练帧08→13则同时改变关节1/2/3/4/5/6，**不是同一运动**。这轮说明误差并非每次运动都会出现，但尚未定位训练帧08→13的差异。Qt字体警告不影响采集。运行外参未更换。

若继续做一次针对性诊断，在保持标定板可见及现场安全间隙的条件下，用受控操作选择两个会同时改变多个腕部／前臂关节的安全姿态；不要求机械臂自动回放08→13，也不为对照强行到达那两个角度。停稳后按S记录起点，受控移动，停稳后按S记录终点。使用与新采集一致的V112反馈配置：

```bash
python tools/verify_handeye_rotation_imu.py \
  --config config/pipeline_orbbec.yaml --firmware V112
```

**17:13第二次实测：**`calibration/orbbec_rotation_audit/20261103_171357_575621/result.json`中机器人／视觉／陀螺仪转角为44.060°／44.575°／44.306°；主要关节变化是1号+7.6°、2号−8.2°、6号+52.4°、7号−2.4°。用户确认START到END期间标定板没有被碰到或移动。单帧板角点重投影为0.218／0.161 px，但按**当前运行外参**计算，同一固定板在基坐标的起终点位置差`[-1.03,+35.77,-5.73] mm`，合计36.24 mm，完整三维朝向差4.09°。新16张PARK候选对应32.21 mm／2.53°。所以三路**转角大小**接近，不代表相机到机器人基坐标的三维位置已对齐。

纯离线试算一项固定的相机安装平移修正：同时拟合17:05、17:13和旧六帧三组固定板对照，三组残留仍为21.7／17.0／10.9 mm，且原独立8姿态的位置一致性恶化至均值17.3 mm、最大42.8 mm。单纯按一个固定横移／安装位置补偿不能兼顾各姿态；把板尺寸当自由缩放量时，17:13这组需约0.50倍且仍残留20.7 mm，也不支持以修改历史板尺寸直接修复。以上都是模型估计，**不是说实体板真的移动了36 mm**；运行外参仍未修改。暂不再重复同类转角测试，下一步需要独立核对相机或法兰的实际空间位移，才能区分相机安装／视觉测距与机器人名义位姿。

### 0.28 按11月3日全部可用照片联合求一次外参：候选及实机对位验证

已用当天14张训练、8张独立验证、16张新训练、6张固定标记照片（重算整板35角点）和两次转角测试的4张照片，共48张。标定板可能在不同采集时段换过位置，因此每个时段单独估计板在基坐标的位置，统一估计同一套相机到法兰外参。比较文件为`artifacts/orbbec_all_today_20261103/comparison.json`；可复算脚本为同目录`recompute.py`。**位置误差口径为每一时段固定板估计位置相对该时段均值的RMS，再对各时段等权平均；不是实际夹爪对香蕉的误差。**

| 外参 | 六时段等权平均位置RMS | 旧8张RMS | 新16张RMS | 旧六帧两姿态最大位置差 | 17:13两姿态位置差 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 当前运行外参 | 8.67 mm | 6.0 mm | 9.8 mm | 19.4 mm | 36.2 mm |
| 全部数据联合拟合 | **6.92 mm** | **4.4 mm** | **9.3 mm** | **13.4 mm** | **23.0 mm** |

联合拟合是这批候选按上述口径的最低值，已保存为`config/orbbec_handeye_20261103_all_data_trial.json`，并通过现有标定JSON读取接口检查。相对当前外参，相机到法兰平移改变约22 mm，旋转约1.98°；旧6帧和17:13的偏差虽下降，却没有归零。因为这48张也用于联合拟合，表格不能冒充独立实机精度；留出17:13两张不参与拟合重算时仍有约26 mm的板位置差。所以先用下面**只到抓取接触位置、不开爪闭合**的命令比较香蕉在两指间是否比0.25居中，暂不让完整抓取命令自动切换外参。命令的其余参数沿用0.25，仅通过`--calib-json`切换外参并另设日志目录。

```bash
cd /home/yx/yx/panda_ws/src/whole-grasp
conda activate grasp_yolo26

bash scripts/run_grasp_safe.sh \
  --config config/pipeline_orbbec.yaml --phase grasp --execute \
  --calib-json config/orbbec_handeye_20261103_all_data_trial.json \
  --nero-firmware V112 \
  --target-class banana \
  --yolo-weights models/food5_visible_v3_full_reviewed_corrected_best.pt \
  --active-view --no-active-multiview-gate \
  --active-center-height 0.35 --active-center-roll-plan-candidates 1 \
  --no-active-center-optical-relaxation --active-capture-settle-seconds 1.0 \
  --active-side-view-count 0 --grasp-required-view-count 1 \
  --grasp-table-reference-z 0 \
  --preview-cloud \
  --trajectory-executor mit --mit-q-log \
  --mit-max-joint-vel-deg-s 8 --mit-max-joint-accel-deg-s2 8 \
  --grasp-generator graspgenx --nero-grasp-refine --graspgenx-contact-advance auto \
  --grasp-contact-check-only --grasp-close-pos-tol 0.004 \
  --no-mgpc \
  --save-dir logs/orbbec_handeye_trial/contact_check_all_data
```

此命令不闭爪。先看最终姿态预览；如果右指看起来会插入香蕉，就拒绝这次姿态。到位后对照0.25的同角度实物照片，检查左右夹指与香蕉的距离；把新日志目录和现场照片发来。新的`auto`修正深度取决于新一帧点云，不保证每次都是离线回放的8 mm。若仍偏一侧，不要依据离线RMS把这个候选写入`pipeline_orbbec.yaml`，也不要直接改用完整交接命令。

**11月5日19:41接触检查结论与下一步**：该次`contact_checked`已拍照并退回，夹指实测开口99.8 mm，用户确认全程未碰动香蕉。以这次保存的初始视角、中心视角和最终RGB做离线复核，`all_data`与`sept01_joint_refined_candidate`的夹指CAD／实拍遮罩IoU分别为0.688／0.780；两个视角算出的同一桌面高度差分别为8.93／1.72 mm；初始视角目标点投到终点RGB后，落在黄色香蕉内的比例分别为77.9%／89.3%。详见[逐项结果](artifacts/orbbec_20261105_contact_projection/comparison.json)与[叠图](artifacts/orbbec_20261105_contact_projection/cad_sept01_joint/cad_vs_observed_fingers.png)。这三项均支持旧的`sept01_joint_refined_candidate`在**这次接触区域**更合适；不能因为`all_data`在48张标定训练数据上的平均残差更低就认为它对这次抓取更准。也不能只凭这一帧把外参永久写入YAML。下一次仅换外参、保持其他参数原样做接触检查：

```bash
bash scripts/run_grasp_safe.sh \
  --config config/pipeline_orbbec.yaml --phase grasp --execute \
  --calib-json config/sept01_joint_refined_candidate.json \
  --nero-firmware V112 \
  --target-class banana \
  --yolo-weights models/food5_visible_v3_full_reviewed_corrected_best.pt \
  --active-view --no-active-multiview-gate \
  --active-center-height 0.35 --active-center-roll-plan-candidates 1 \
  --no-active-center-optical-relaxation --active-capture-settle-seconds 1.0 \
  --active-side-view-count 0 --grasp-required-view-count 1 \
  --grasp-table-reference-z 0 \
  --preview-cloud \
  --trajectory-executor mit --mit-q-log \
  --mit-max-joint-vel-deg-s 8 --mit-max-joint-accel-deg-s2 8 \
  --grasp-generator graspgenx --nero-grasp-refine --graspgenx-contact-advance auto \
  --grasp-contact-check-only --grasp-close-pos-tol 0.004 \
  --no-mgpc \
  --save-dir logs/orbbec_handeye_trial/contact_check_sept_joint
```

此命令到最终抓取位拍照，**不闭爪**。看两指与香蕉的实际距离；若右指仍贴近或明显偏位，先用这一帧的点云／RGB判断是否为抓取候选自身偏心，不再凭CAD遮罩单独调外参。现阶段不要用下方0.29的`all_data`完整交接命令判断外参已经解决了偏位。

#### 11月5日20:35后的对照试验：相机与夹指几何修正

20:35这一轮已正常到位、拍照并退回，用户确认香蕉没有被碰动，实物仍偏向右指。原始初始／中心视角的点云投到终点RGB时都已有偏差，桌面校正对该偏差的影响较小。只根据“标定板平均残差最低”选择外参，未能解决实物对位。

已用这次采集RGB中的100 mm开口黑色夹指轮廓，对`sept01_joint_refined_candidate`做局部几何拟合，保存为`config/orbbec_handeye_20261105_finger_fit_trial.json`。相对原候选，相机安装平移变化约6.7 mm、旋转约1.14°。独立检查两次未移动的香蕉：中心视角点云投到终点RGB后落在黄色香蕉内的比例，19:41为85.8%→96.7%，20:35为76.9%→89.3%。这只是图像一致性诊断，不是毫米级准确率。旧标定板验证残差有所变差，所以此文件是**抓取区域的试验外参**，尚不能替换全局标定用于伺服交接。

复用20:35真实点云、起始关节与保存的GenX候选，重新转换原始视角、校正桌面并做当前接触检查流程的离线路径预检，已通过；仍选原网络候选20，推进11 mm。最终TCP相对原计划变化约`[+2.1,+5.1,+0.03] mm`（机器人基坐标），支持先验证右偏是否减轻。拟合与复核结果见[夹指叠图](artifacts/orbbec_20261105_contact_projection/silhouette_fit_overlay.png)、[跨视角指标](artifacts/orbbec_20261105_contact_projection/cross_view_comparison.json)、[离线路径结果](artifacts/orbbec_20261105_contact_projection/finger_fit_replay/result.json)。

```bash
bash scripts/run_grasp_safe.sh \
  --config config/pipeline_orbbec.yaml --phase grasp --execute \
  --calib-json config/orbbec_handeye_20261105_finger_fit_trial.json \
  --nero-firmware V112 \
  --target-class banana \
  --yolo-weights models/food5_visible_v3_full_reviewed_corrected_best.pt \
  --active-view --no-active-multiview-gate \
  --active-center-height 0.35 --active-center-roll-plan-candidates 1 \
  --no-active-center-optical-relaxation --active-capture-settle-seconds 1.0 \
  --active-side-view-count 0 --grasp-required-view-count 1 \
  --grasp-table-reference-z 0 \
  --preview-cloud \
  --trajectory-executor mit --mit-q-log \
  --mit-max-joint-vel-deg-s 8 --mit-max-joint-accel-deg-s2 8 \
  --grasp-generator graspgenx --nero-grasp-refine --graspgenx-contact-advance auto \
  --grasp-contact-check-only --grasp-close-pos-tol 0.004 \
  --no-mgpc \
  --save-dir logs/orbbec_handeye_trial/contact_check_finger_fit
```

本次仍为张开夹指到位拍照、退回。观察香蕉相对两指的实际位置，并用保存的终点RGB再次核对；新生成候选和现场点云会变化，离线通过不保证下一次选择同一个候选。

**21:22实机反馈更新：当前右偏好转的成功基准见7.3。** 用户确认该轮右偏有所好转；`2026-11-05_21-22-39_197404_p375404`完整抓取、抬升及持物位均完成，使用的外参是`config/sept01_joint_refined_candidate.json`。选中GenX原网络候选21，保留原生朝向，自动推进8 mm，沿开合轴居中修正约11.85 mm；没有触发`nero_refinement`的姿态改造。闭夹接触宽度约33.9 mm，持物后约33.55 mm，力反馈绝对值约0.50持续保持。这一轮同时改变了抓姿、目标点云和实到位状态，不能把右偏好转单独归因于外参，更不能据此认定上面的`finger_fit_trial`已通过实机对位验证。当前优先沿用7.3这组已实测成功的组合，继续观察右偏的重复性。

### 0.29 新外参完整试跑：抓香蕉 → 持物位 → 可视化跟随A手 → B手交接

本次只用`config/orbbec_handeye_20261103_all_data_trial.json`作试验：`--calib-json`供抓取定位，`--handeye`供伺服跟手，**两项必须同时传**。单视角先走通完整链路；抓取成功后按现有规划去配置的持物关节位，再自动转入人手跟随和手势交接。`--gesture-handover`启用双手识别：先单独伸出A手，连续识别后锁定为跟随手；另一只B手用于交接手势。`--servo-preview`打开实时相机／手部／深度／保持原因窗口。保留现有运动、碰撞和力矩保护；若抓取未成功，伺服不会启动。新外参在离线留出姿态仍有约26 mm的板位置差，这次实物对位结果需要单独记录。

```bash
cd /home/yx/yx/panda_ws/src/whole-grasp
conda activate grasp_yolo26
sudo ip link set can0 up type can bitrate 1000000

MOTION_SUPPORT_RELEASE_HEIGHT=0.025 bash scripts/run_grasp_safe.sh \
  --config config/pipeline_orbbec.yaml --phase all --execute \
  --calib-json config/orbbec_handeye_20261103_all_data_trial.json \
  --handeye config/orbbec_handeye_20261103_all_data_trial.json \
  --nero-firmware V112 \
  --target-class banana \
  --yolo-weights models/food5_visible_v3_full_reviewed_corrected_best.pt \
  --active-view --no-active-multiview-gate \
  --active-center-height 0.35 --active-center-roll-plan-candidates 1 \
  --no-active-center-optical-relaxation --active-capture-settle-seconds 1.0 \
  --active-side-view-count 0 --grasp-required-view-count 1 \
  --grasp-table-reference-z 0 \
  --preview-cloud \
  --trajectory-executor mit --mit-q-log \
  --mit-max-joint-vel-deg-s 8 --mit-max-joint-accel-deg-s2 8 \
  --grasp-generator graspgenx --nero-grasp-refine --graspgenx-contact-advance auto \
  --grasp-close-pos-tol 0.004 \
  --no-mgpc \
  --enable-servo-delivery \
  --target-source mediapipe --gesture-handover --hand-camera-center \
  --servo-preview --servo-diagnostics \
  --hand-depth-radius 8 \
  --delivery-distance 0.05 \
  --gesture-ready-distance 0.15 \
  --gesture-stable-seconds 0.5 --gesture-still-tolerance-mm 10 \
  --hand-memory-approach \
  --max-linear-vel 0.03 --max-joint-vel 0.15 \
  --max-angular-vel 0.15 \
  --follow-kp 4 --follow-min-qdot 0.02 \
  --follow-kp-rise-time 0.20 --follow-max-error-rad 0.03 \
  --impedance-target-linear-vel 0.03 \
  --impedance-target-angular-vel 0.15 \
  --gesture-extra-close-mm 1 \
  --gesture-open-seconds 5 --gesture-close-seconds 2 \
  --save-dir logs/orbbec_handeye_trial/full_grasp_follow_all_data
```

终端到最终抓取姿态会要求人工确认；确认实物夹指和桌面间隙后再继续。进入跟手后先只伸出A手完成身份锁定，再伸出B手；窗口应显示两只手及`Receiver`／`control`的左右标签。按下方第1项的B手手势顺序完成交接；`Q`或`Esc`可受控结束。保留日志和现场照片，以便区分新外参对抓取对位与跟手定位各自的影响。**11月5日16:30那次命令用了`--servo-follow-only`，实际只检测一只手且不进入手势交接；不能用于验证双手分工。**

**11月5日16:47完整试跑排查**：香蕉已夹住并抬起；后续预览始终处于`FOLLOW_RECEIVER`，没有进入`GESTURE_LEARN`，因此B手动作不会松夹。A手偶尔进入15 cm准备范围，但图像居中误差仍约40–65 px，而进入学习要求小于12 px；多数时间A手丢失或掌心深度无效，稳定计时一直为0。预览现直接显示`Handover phase`和`A hand center error`。当时命令要求稳定3秒；**只有窗口明确显示`GESTURE_LEARN`后**B手捏合、张开才会被用于学习。若A手一靠近就丢失，不要反复做B手手势，先看预览里的识别、深度和居中误差。该次GraspGenX自动接近量为5 mm，闭夹处实测TCP比计划高约2.9 mm；仅凭这一次不能把浅抓归因于外参。

**11月5日17:10复测修正**：这次实物未被抬起，右指还过于贴近香蕉。需要把横向居中和沿接近轴下探当作两个独立修正量；**右指偏位不能靠下探解决，但偏位修好后可以继续下探**。选中候选3在点云上预计的两侧接触深入量仅约10.4/10.3 mm，刚超过10 mm最低门槛；实际闭爪位的TCP又比计划高2.5 mm。它的开放夹指CAD距模型桌面约16.7 mm，当前CAD门槛是5 mm，因此模型几何并未要求停在此深度；这不等于允许无预检地把11.7 mm余量全部下探。闭爪力反馈约0.50，抬升后约0.16，旧“稳定”记录只看宽度，误把疑似空抓带入跟手；现在若抬升后力反馈低于闭爪时的一半，就记为未确认持物并停止进入跟手，机械臂保持当前位置。该次A手曾在15 cm内、居中且位置稳定约0.66秒，随后被遮挡，达不到原3秒门槛。上面的奥比完整流程试跑命令因此将**进入手势学习的稳定时间**改为0.5秒；距离、居中、位置容差以及B手确认和松夹条件不变。A手短暂满足条件后才会显示`GESTURE_LEARN`，遮挡后现有状态机会保持已取得的准备状态。这个0.5秒是为当前相机遮挡场景做的试验值，不作为论文结论；先确保香蕉真正被抬起再验证交接。

离线核对：该次实际到位TCP横向只偏离**指令位置**约1 mm；这是机械臂的跟踪误差，**不是夹指相对真实香蕉的横向误差**，不能拿1 mm解释或否定照片中的明显右偏。程序曾依据点云沿开合轴主动平移约10.2 mm，平移后点云在模型两指间近似对称（约−18.5～+16.8 mm），实物却仍偏向右指；真实物体相对夹指的横向偏差尚无毫米级测量。因此需要检查目标点云映射、相机到夹爪变换和夹指模型。用保存RGB和100 mm开口投影夹指CAD时，当前联合外参的预测遮罩与实测遮罩IoU约0.68，旧试验外参约0.75；这支持继续检查几何链，但遮罩和CAD都非精确真值，不能仅凭IoU更换外参。离线叠图见`artifacts/orbbec_20261105_gripper_projection_audit/cad_vs_observed_fingers.png`。本帧原始中心视角桌面高度偏约12 mm，现有桌面配准已对该视角做了修正，不应再额外盲减12 mm。

**叠图颜色和下探复核（11月5日）**：叠图中的**青色**是夹指CAD经当前手眼外参投到采集RGB的预测轮廓，**紫红色**是之前三帧实拍RGB制作的固定夹指遮罩，并非本帧在线识别结果。核对本帧原始RGB，紫红色在检查的夹指边缘仍基本贴着实际黑色轮廓；青色与它的错位是真实诊断信号，不能说只是画图误差。例如图像第550行，左指青色约341–487 px、实物黑色约362–460 px，右指青色约989–1233 px、实物黑色约1075–1185 px。11月3日与11月5日共四张保存RGB在这一行的内侧实物边缘也基本稳定（左约459–463 px、右约1075–1077 px），因此不是一张照片偶然分割错位。手眼、安装几何和CAD指形共同决定青色轮廓，单帧不能单独判定是哪一个出错；原始分辨率下青色／保存遮罩IoU约0.687。相同CAD换旧试验外参约0.750、9月外参约0.817，说明外参对轮廓影响明显，但不能只凭这张图选择抓取外参。两者都是采图时张开100 mm的夹指，不能从该图直接读出最终抓取深度。旧修正器对同一姿态找到第一个通过接触检查的3 mm推进就停止；这不是桌面5 mm门槛拦截了继续下探。用本次保存的1498个目标点逐毫米复核：3 mm时两侧接触深入约10.4/10.3 mm、CAD距桌面16.7 mm；8 mm时约15.4/15.3 mm、CAD距桌面11.7 mm。现仅对**GenX触发Nero修正**的姿态，优先尝试接触再深入5 mm且桌面模型额外保留5 mm的位置；最浅合格姿态保留作路径备选。复用本次真实点云、网络候选与起始关节的完整**离线**路线回放仍选候选3的`vertical_flip`，推进从3变8 mm，预抓取、接近、抬升和持物路线均通过，见`artifacts/orbbec_20261105_deeper_contact_replay/2026-11-05_17-10-44_141826_p119935/`。这还不是实机验证；右指偏位也没有由下探修好，先用抓取接触检查观察两指相对香蕉的位置，确认居中后再执行闭爪和交接。

#### 11月5日首次试跑：夹指遮罩拒绝采集，已离线修正

`2026-11-05_15-55-29_709974_p35743`这次在目标识别前终止：100 mm开口已到位，旧遮罩的固定外轮廓没有覆盖本帧左夹指边缘，`outer=0.266`超过原阈值`0.150`；`silhouette=0.969`比`0.970`略低。诊断RGB显示越界区域仍是黑色夹指，不是香蕉；旧遮罩在本帧漏掉约1144个有效深度像素。未触发电子急停，机械臂保持空闲。

已用原来两帧与本次失败帧离线建立固定三帧夹指轮廓，保存为`config/orbbec_self_mask_20261105_100mm_threeframe_trial.json`，并在`config/pipeline_orbbec.yaml`切换到它。外轮廓仅固定增加8像素边界，运行时不追踪、不扩张；原夹指轮廓文件保留。三帧离线回放均通过原有`core/silhouette/outer`门槛，新遮罩未覆盖图像中香蕉区域。**继续用上面0.29的同一条命令重试**，不需要更改抓取或伺服外参。若再次拒绝采集，保留新的诊断目录，不要临时关闭遮罩。

更新：2026-11-02。**第7项已更新为 F 组多视角 GraspGenX 抓香蕉 → 指定关节持物位 → 带 RGB-D 可视化的人手跟随 → 双手手势交接。沿用现有 GenX 候选与评分，不加入本次讨论的翻转修改。A–H 抓取对比仍见 9.2–9.9，候选修正说明见 9.1.8。9.0.2 的 −5 mm 指令仅为历史诊断。**
第1～6项独立测试人机交互，主程序使用 `--phase servo`，跳过识别物体、规划抓取和抬升。
**只测新交接逻辑用第1项；抓取加人手交接用第7项；香蕉八组抓取对比用第9项。**

每个新终端先执行：

```bash
cd /home/yx/yx/panda_ws/src/whole-grasp
conda activate grasp_yolo26
sudo ip link set can0 up type can bitrate 1000000
```

每次只运行一个测试，先退出上一个相机／机械臂控制程序。独立真机测试从当前已使能、静止的姿态开始，
不会先去抓取物体，也不会自动移动到预设的起始姿态。第7项则会执行完整的观察、抓取与抬升运动。

| 项目 | 测什么 | 机械臂跟随 | 夹爪执行 |
| --- | --- | --- | --- |
| 1 | 本次完整交接，启用有限位置记忆 | 松夹前跟随 | B手确认后控制 |
| 2 | 单独人手跟随与居中 | 是 | 不发开合指令 |
| 3 | 空夹爪两指比例开合 | 不跟随，仅当前位置保持 | 是，空载全行程 |
| 4 | 主程序相机、手识别、深度诊断 | 不连接CAN | 模拟 |
| 5 | 完整交接，关闭记忆接近作对照 | 松夹前跟随 | B手确认后控制 |
| 6 | 空夹爪脚本的手指比例预览 | 不连接CAN | 不执行 |
| 7 | GenX多视角抓香蕉 → 指定持物位 → 可视化跟随A → B手松夹 | 抓取成功后接入 | 抓取夹持、B手确认后松夹 |

## 1. 本次真机测试：跟随A手 → 学习B手 → 松夹

启动前，夹爪应已夹住空杯或测试物体，机械臂处于静止状态。程序读取当前实测夹持开口，
不会替你完成抓取。不要把夹爪闭到零位后直接测这项；若只是空爪测试，使用第2或第3项。
空爪已接近85毫米全开时，持杯流程只剩约1毫米的允许开合范围，不能用来判断全行程比例效果。

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
  --gesture-ready-distance 0.15 \
  --gesture-stable-seconds 3 --gesture-still-tolerance-mm 10 \
  --hand-memory-approach \
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

操作与预期：

1. 先只伸出A手，锁定左右身份。机械臂跟随、转向，让掌心居中，跟随目标为TCP到掌心5厘米。
2. TCP到掌心不超过15厘米、掌心居中、基座坐标中的位置稳定3秒后，显示 `GESTURE_LEARN`。
   这时机械臂保持，不必要求先测到5厘米。
3. B手拇指与食指捏合约1秒，再充分张开约1秒；学习期间夹爪不动。
4. 显示 `GESTURE_ARM` 后，B手再次捏合约0.4秒，进入 `GESTURE_READY`；此时仍不发送开合目标。
5. **想测重新跟随，就在此时保持B手捏合，再移动A手超过原稳定容差。** 应恢复跟随，夹爪保持原夹持指令。
   A重新稳定3秒后，跳过已完成的学习，再次捏合确认；旧的张开请求不会补执行。
6. 想松夹时，让A保持交接状态、机械臂停稳，B再主动张开。进入 `GESTURE_CONTROL`，夹爪按比例张开。
   **从开始松夹起，本轮机械臂不再跟随A；不等待全开才锁住跟随。**
7. 全开到位且反馈确认后，显示 `HANDOVER_DONE`，机械臂原地保持，不自动撤退或回夹。

有限记忆接近只适用于：待命前、尚未松夹，A此前有稳定三维测量，现在二维仍可信但深度短暂失效。
最多0.3秒／额外3毫米／1厘米每秒，任一预算到达就暂停；它不计入稳定3秒，不产生松夹授权。
A二维也消失时不会继续靠旧位置运动。出现 `MEMORY_APPROACH` 才表示实际进入该分支，不是每次丢深度都会进入。

## 2. 真机只跟随人手，不进入手势交接

可空爪运行。只伸出一只手观察跟随、居中及有效深度下的近距离退让；本项不测试双手身份或手势接管。
有限位置记忆属于手势交接流程，本项不启用。

```bash
bash scripts/run_grasp_safe.sh \
  --config config/pipeline.yaml \
  --phase servo --execute \
  --camera-width 1920 --camera-height 1080 --camera-fps 30 \
  --camera-depth-width 1280 --camera-depth-height 720 --camera-depth-fps 30 \
  --camera-depth-filters all \
  --target-source mediapipe --servo-follow-only \
  --hand-camera-center \
  --servo-preview --servo-diagnostics \
  --hand-depth-radius 8 \
  --delivery-distance 0.05 \
  --no-hand-memory-approach \
  --max-linear-vel 0.03 --max-joint-vel 0.15 \
  --max-angular-vel 0.15 \
  --follow-kp 4 --follow-min-qdot 0.02 \
  --follow-kp-rise-time 0.20 \
  --follow-max-error-rad 0.03 \
  --impedance-target-linear-vel 0.03 \
  --impedance-target-angular-vel 0.15
```

## 3. 真机只测空夹爪：拇指食指比例开合

**本项必须空载，不用于持杯。** 只开彩色流，不需要有效深度。显式的 `--prepare-can-hold`
必要时把已有以太网J保持交给CAN，以当前实测关节角建立保持；不使能／失能，不去预设姿态。
不要与主伺服同时运行，也不要把这个参数加进第1项。

```bash
python scripts/test_pinch_gripper.py \
  --config config/pipeline.yaml \
  --camera-width 640 --camera-height 480 --camera-fps 30 \
  --min-width-mm 0 --max-width-mm 85 \
  --open-seconds 5 --close-seconds 2 \
  --force 0.50 --max-lead-mm 5 \
  --prepare-can-hold \
  --execute
```

只伸出一只手，先捏合约1秒、再充分张开约1秒；学习后直接比例跟随，不需要按C或A，也不需要再次捏合确认。
这与持杯交接的确认规则不同。丢手时停发目标，不自动回夹；`FAULT / NO RESPONSE` 表示已锁定故障，不能靠重新伸手自动恢复。

## 4. 只测主程序相机、识别与深度，不连接CAN

使用同样的最大分辨率与滤波参数，可在夹持物体前后比较手掌深度。`40` 是明确指定的模拟夹持开口，
不是硬件读数；不能把这个参数用于带 `--execute` 的命令。

```bash
bash scripts/run_grasp_safe.sh \
  --config config/pipeline.yaml \
  --phase servo \
  --camera-width 1920 --camera-height 1080 --camera-fps 30 \
  --camera-depth-width 1280 --camera-depth-height 720 --camera-depth-fps 30 \
  --camera-depth-filters all \
  --target-source mediapipe --gesture-handover \
  --hand-camera-center \
  --servo-preview --servo-diagnostics \
  --hand-depth-radius 8 \
  --delivery-distance 0.05 \
  --gesture-ready-distance 0.15 \
  --gesture-stable-seconds 3 --gesture-still-tolerance-mm 10 \
  --hand-memory-approach \
  --gesture-preview-grip-width-mm 40 \
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

骨架清楚不代表深度有效。看 `Depth points`、`Camera XYZ` 和 `invalid_palm_depth`。
模拟关节角不是实机角度，基座坐标和状态变化不能用于验证真实跟随或手眼精度。

## 5. 真机完整交接，但关闭有限记忆接近

与第1项仅记忆开关不同，用于比较“只丢深度时直接暂停”和“有限记忆接近”。其余操作沿用第1项。

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
  --gesture-ready-distance 0.15 \
  --gesture-stable-seconds 3 --gesture-still-tolerance-mm 10 \
  --no-hand-memory-approach \
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

## 6. 只看手指比例与自动学习，不动夹爪

不连接CAN，只开彩色流；学习操作与第3项一致。

```bash
python scripts/test_pinch_gripper.py \
  --config config/pipeline.yaml \
  --camera-width 640 --camera-height 480 --camera-fps 30 \
  --min-width-mm 0 --max-width-mm 85 \
  --open-seconds 5 --close-seconds 2 \
  --force 0.50 --max-lead-mm 5
```

## 7. 完整真机流程：GenX多视角抓香蕉 → 指定持物位 → 可视化跟随A → B手松夹

抓取部分沿用 11 月 1 日 F1/F2 实测成功的配置：**中心＋一个侧视角、刚体 ICP、GraspGenX＋Nero 候选修正、不加 MGPC**。
抓住后先短距离脱离桌面，再到配置中的指定持物关节位 `[-0.002, 0, 0.002, 100.6, 0, 0, 0]°`，然后自动进入人手跟随／交接。
`--phase all --enable-servo-delivery` 是串联入口；`--servo-preview --servo-diagnostics` 会打开伺服 RGB-D 窗口并记录诊断。
相机分辨率、深度滤波、抓取与伺服各自的手眼参数均读取 `config/pipeline.yaml`。在有桌面显示的终端运行；先退出其他相机或机械臂控制程序。
CAN 初始化已包含在下面的完整命令中。11 月 2 日 09:54 两次启动的 `[WARN] Device is DOWN` 和 `Timed out waiting for Nero enable()` 均发生于 `arm_setup`；现场只读检查显示 `can0` 为 `DOWN / STOPPED`、RX/TX 均为 0。此时先启动 CAN 接口，再运行抓取，不需要修改 GenX、IK 或伺服参数。

```bash
cd /home/yx/yx/panda_ws/src/whole-grasp
conda activate grasp_yolo26
sudo ip link set can0 up type can bitrate 1000000

MOTION_SUPPORT_RELEASE_HEIGHT=0.025 bash scripts/run_grasp_safe.sh \
  --config config/pipeline.yaml --phase all --execute \
  --target-class banana \
  --yolo-weights models/food5_visible_v3_full_reviewed_corrected_best.pt \
  --active-view --no-active-multiview-gate \
  --active-center-height 0.35 --active-center-roll-plan-candidates 1 \
  --no-active-center-optical-relaxation --active-capture-settle-seconds 1.0 \
  --active-fusion-rigid-icp \
  --active-table-reg-max-height-shift 0.030 \
  --preview-cloud \
  --trajectory-executor mit --mit-q-log \
  --mit-max-joint-vel-deg-s 12 --mit-max-joint-accel-deg-s2 8 \
  --active-side-view-count 1 --grasp-required-view-count 2 \
  --grasp-table-reference-z 0 \
  --grasp-generator graspgenx --nero-grasp-refine --graspgenx-contact-advance auto \
  --grasp-close-pos-tol 0.004 \
  --no-mgpc \
  --enable-servo-delivery \
  --target-source mediapipe --gesture-handover \
  --hand-camera-center \
  --servo-preview --servo-diagnostics \
  --hand-depth-radius 8 \
  --delivery-distance 0.05 \
  --gesture-ready-distance 0.15 \
  --gesture-stable-seconds 3 --gesture-still-tolerance-mm 10 \
  --hand-memory-approach \
  --max-linear-vel 0.03 --max-joint-vel 0.15 \
  --max-angular-vel 0.15 \
  --follow-kp 4 --follow-min-qdot 0.02 \
  --follow-kp-rise-time 0.20 \
  --follow-max-error-rad 0.03 \
  --impedance-target-linear-vel 0.03 \
  --impedance-target-angular-vel 0.15 \
  --gesture-extra-close-mm 1 \
  --gesture-open-seconds 5 --gesture-close-seconds 2 \
  --save-dir logs/banana_hri/genx_multi
```

操作顺序：

1. 按终端提示确认初始观察位和最终抓取。点云／抓姿的 Open3D 窗口关闭后才继续执行。
2. 确认香蕉实际被夹起并保持。抓取流程报告成功后会进入伺服，终端应出现 `[handoff] after lift: reusing running MIT session`；使用原 MIT 会话，不启动第二套运动控制。
3. 自动打开 **`Hand servo RGB-D | Q/Esc/close: controlled exit`** 窗口。先只伸出接收手 A，锁定身份后缓慢移动，观察机械臂跟随和掌心居中。窗口显示手部关键点、有效深度点数、坐标、跟踪误差、暂停原因以及 A/B 身份和交接状态；并非只显示抓取点云。
4. A 在 TCP 周围 15 cm 内、掌心居中且位置稳定 3 s 后，进入 `GESTURE_LEARN`。B 手捏合约 1 s，再充分张开约 1 s，学习控制范围；这时夹爪保持夹持。
5. 显示 `GESTURE_ARM` 后，B 再次捏合约 0.4 s 确认。进入 `GESTURE_READY` 后，A 接稳香蕉，B 主动张开以逐步松夹。开始松夹即停止跟随 A；松夹前 A 移动则恢复跟随。
6. 全开反馈确认后显示 `HANDOVER_DONE`，原地保持。完整交接以该状态与现场实际接物为准；抓取的 `status=success` 本身不能证明交接完成。

手势交接沿用当前代码的 **85 mm 松夹上限**；抓取采集前仍按已验证的 **100 mm** 张开，不会因这条命令改回 85 mm。11 月 1 日 F1/F2 的实测夹持开口约 36 mm，位于现有交接范围内。
上述两阶段单独测试已有基础，这条更新后的整段香蕉流程仍需本次实机验证；此次只更新命令，未修改抓取评分或手势控制。

排查与日志：

- 伺服阶段保持可视化窗口打开；按 Q/Esc、关闭窗口或 Ctrl+C 会受控退出，不是继续下一阶段。抓取点云窗口的关闭则用于继续抓取流程，两者含义不同。
- 看不到 A 时检查 `Reason`、`Depth points`、`Receiver`；未进入交接时检查 `Receiver still`、`handover ready`、`Handover`。短暂丢失深度时可显示 `MEMORY_APPROACH`，预算和条件沿用第1项。
- 抓取记录：`logs/banana_hri/genx_multi/<运行ID>/experiment_result.json`；伺服记录：`logs/banana_hri/genx_multi/servo_diagnostics/<运行ID>/samples.jsonl`，另有 `control_cycles.jsonl`（以终端实际输出路径为准）。
- 此命令执行完整手势交接，不添加 `--servo-follow-only`。全开后不自动撤退或重新闭合。

### 7.1 11 月 2 日手跟随诊断：RGB 看见手，但掌心深度经常无效

09:56 抓取已闭爪并到指定持物位；10:00:45–10:01:52 的 66 条定时伺服诊断中，46 条为 `invalid_palm_depth`，15 条为 `receiver_absent_2d`，不是全程没有识别到手。
失败时常见 `candidate_count=1, accepted_count=1, receiver_presence=visible`，但掌心五处采样仅有 0–2 处有效，而主流程要求至少 3 处；10:01:31 曾测到掌心相机 Z 约 296 mm 并进入跟随。
Qt 字体提示和 MediaPipe feedback tensor 提示未阻止检测。当前没有保存失败帧的 RGB-D，尚不能确定是近距离深度缺失、遮挡，还是采样落到无深度区域，不能直接降低有效点数门槛处理。

**夹爪可以继续夹着物体；先退出上一条伺服／抓取程序，使机械臂保持静止，再单独看相机。下面不连接 CAN，也不移动或松开夹爪。** 相机配置与第7项一致，并采用相同的当前原始深度支持规则。

```bash
cd /home/yx/yx/panda_ws/src/whole-grasp
conda activate grasp_yolo26
python scripts/test_realsense_hand.py \
  --serial 243622072771 \
  --width 1920 --height 1080 --fps 30 \
  --depth-width 1280 --depth-height 720 --depth-fps 30 \
  --depth-filters all --show-depth \
  --num-hands 2 --depth-radius 8 \
  --min-detection-confidence 0.35 \
  --min-presence-confidence 0.35 \
  --min-tracking-confidence 0.35 \
  --save-dir logs/hand_depth_debug
```

先将摊开的手掌放在相机前约 30–50 cm，再逐渐靠近。左侧看 RGB 骨架，右侧看深度；黑色表示没有当前有效深度，绿色圆圈是掌心五处采样区域。
看见骨架但深度黑／`depth invalid: 0/5` 时按 **S**，现在允许保存“RGB 检测成功但深度失败”的帧，不再只允许保存成功帧。
每份记录包含 `color.png`、`overlay.png`、`depth_m.npy`（实际检测深度）、`depth_filtered_m.npy`、`depth_raw_m.npy` 和 `hand_pose.json`。Q/Esc 退出。

### 7.2 持物拉拽：只读七轴力矩反馈与基线差值

上一轮 Ctrl+C 已确认保持当前位置且电机继续使能。保持现有持物状态，先退出抓取／跟随程序，再运行只读记录器。
它使用 SDK 反馈连接，不调用机械臂使能／掉使能、运动、重力模式切换或夹爪接口；不会重新抓取、松夹或移动到其他姿态，也不接管当前控制器。

```bash
cd /home/yx/yx/panda_ws/src/whole-grasp
conda activate grasp_yolo26
python tools/monitor_grasp_pull_torque.py --plot --duration 60
```

操作：

1. 前 **5 s** 保持持物状态、不触碰物体，建立含物体重量的固定姿态基线。终端和曲线标题持续显示 `BASELINE READY` 后开始测试。`label=REST` 是人工标注的静止段，不表示基线还在建立；按 P/R 才会改变该标签，程序不自动猜测是否在拉拽。
2. 曲线窗口上方为 J1–J7 的电机力矩反馈，下方为 `当前反馈 − 初始基线`，单位 N·m。不要把控制器发出的力矩命令当反馈。
3. 点击曲线窗口，按 **P** 标记拉拽段，轻拉物体约 2 s；松手后按 **R** 标记静止段并等约 3 s。重复数次，可换拉拽方向。P/R 只标记数据，不控制机械臂或夹爪。
4. Q/Esc、关闭曲线窗口、Ctrl+C 或满 60 s 结束记录。终端输出每轴峰值变化、基线噪声与关节姿态漂移。

输出：`logs/pull_torque/<时间>/samples.csv`、`summary.json`、`torque.png`、`torque.pdf`。CSV 包含各轴反馈时间年龄；缺失、过期或整组重复反馈不当作新测量。七轴反馈异步，不能当作同时刻精确外力测量。
基线差值还会受到姿态、重力和控制器响应影响；先比较同一姿态下多次 REST/PULL 的重复性，再考虑作为交接意图的辅助证据。此次只记录，不用力矩触发松夹或修改手势状态机。

**14:04 拉拽记录：** 操作者确认约 26 s 前完全未触碰物体，之后持续拉着；键盘 PULL 标签从 25.545 s 开始，一直未切回 REST。J2 在未触碰段的峰值基线差约 0.047 N·m，PULL 段约 2.993 N·m；J1/J3/J7 后段也有变化。J4 在未触碰段就有约 2.128 N·m 的周期变化，PULL 段峰值约 2.078 N·m，因此不能把 J4 的大幅变化直接作为拉拽证据。J2 的主要变化在约 49 s 后才出现，尚不能证明能及时检测到 26 s 的拉拽开始。
见 [分段曲线](artifacts/pull_torque_20261102_140415/rest_pull_comparison.png)、[七轴对照 CSV](artifacts/pull_torque_20261102_140415/joint_comparison.csv) 与 [解释边界](artifacts/pull_torque_20261102_140415/analysis.json)。原始数据未修改，未设置交接触发阈值。
下一次沿用同一只读命令，先完全不碰至少 15 s，再按 P 轻拉约 2 s、松手按 R 保持约 3 s，重复至少 3 次；需要确认变化是否跟随拉／松反复出现，而不是只比较一次持续拉拽前后的峰值。

**14:32 拉／松标记成功：** PULL 从 32.870 s 到 45.586 s，之后操作者确认完全松手、物体悬空。J4 拉前 REST 的峰值基线差约 0.029 N·m，PULL 段约 3.677 N·m；但松手后仍持续偏移约 +2.51 N·m，未恢复初始基线。J2 拉前也已有约 0.898 N·m 的背景变化，PULL 段峰值约 2.104 N·m。此轮说明反馈有响应，尚不能用固定基线差阈值直接判定持续拉拽／松手。已分开计算拉前 REST 和松手后 REST，避免混为同一噪声水平；见 [分段曲线](artifacts/pull_torque_20261102_143233/pull_release_comparison.png) 与 [分段统计](artifacts/pull_torque_20261102_143233/analysis.json)。
本轮 Ctrl+C 曾被 Tk 绘图回调捕获并继续运行到 60 s；记录器已改为 SIGINT 设置退出标志，循环退出后保存并关闭只读连接。P/R 也已解除与 Matplotlib 平移／复位快捷键的冲突，并支持大写。此次只修改记录器的交互与退出，不改变机械臂或夹爪控制。

### 7.3 当前奥比：完整抓取后退出，再采静止拉／松力矩（2026-11-05）

本轮按用户要求先完成抓取并停住，再启动只读采集。抓取使用当前选择的 `sept01_joint_refined_candidate` 试验外参；只通过本次CLI指定，不修改YAML。下方命令去掉了原接触检查中的 `--grasp-contact-check-only`，会真正闭爪、抬起并到YAML指定持物关节位；没有启用视觉递送、手势交接或自动释放。最终姿态预览与执行确认仍保留，实物明显偏位时拒绝该候选。

先退出其他抓取／相机程序；物体尚在桌面、机械臂已处于可重新开始抓取的位置时运行。若已经夹着物体并停稳，不要重跑抓取命令，直接在确认上一程序退出后执行第二步。

**第一步：抓取并到持物位。**

```bash
cd /home/yx/yx/panda_ws/src/whole-grasp
conda activate grasp_yolo26
sudo ip link set can0 up type can bitrate 1000000

MOTION_SUPPORT_RELEASE_HEIGHT=0.025 bash scripts/run_grasp_safe.sh \
  --config config/pipeline_orbbec.yaml --phase grasp --execute \
  --calib-json config/sept01_joint_refined_candidate.json \
  --nero-firmware V112 \
  --target-class banana \
  --yolo-weights models/food5_visible_v3_full_reviewed_corrected_best.pt \
  --active-view --no-active-multiview-gate \
  --active-center-height 0.35 --active-center-roll-plan-candidates 1 \
  --no-active-center-optical-relaxation --active-capture-settle-seconds 1.0 \
  --active-side-view-count 0 --grasp-required-view-count 1 \
  --grasp-table-reference-z 0 \
  --preview-cloud \
  --trajectory-executor mit --mit-q-log \
  --mit-max-joint-vel-deg-s 8 --mit-max-joint-accel-deg-s2 8 \
  --grasp-generator graspgenx --nero-grasp-refine --graspgenx-contact-advance auto \
  --grasp-close-pos-tol 0.004 \
  --no-mgpc \
  --save-dir logs/handover_torque_20261105/grasp_hold_sept_joint
```

正常完成应有 `[done] grasp and requested joint hold completed`、`gripper remains closed`、`[session] status=success`，退出过程应有 `[shutdown] safe shutdown state confirmed`。等待程序返回命令提示符、现场确认物体悬空夹住且机械臂稳定后再采；正常结束无需另外Ctrl+C。该退出路径将MIT受控交接到设备位置保持，基类退出仅断开SDK连接，不发送开夹或关节掉使能。若尚在运行而需中止，按一次Ctrl+C并等待受控退出；保护/急停或抓取失败不能当作成功持物，采集器也不会恢复使能或替代保持控制。退出后的控制状态与刚才MIT运动阶段不同，本次基线必须在退出后重新采。

**第二步：同一终端只读采集，显式使用V112。**

```bash
python tools/monitor_grasp_pull_torque.py \
  --channel can0 --firmware V112 \
  --plot --duration 120 --baseline-seconds 5 --hz 20 \
  --save-dir logs/pull_torque/20261105_static_pull_release
```

首轮只做一种动作“轻拉／完全松手”，不混入碰、推或握住不拉，以便现有P/R标签准确使用：

1. 开始后前15秒完全不碰物体；前5秒用于基线，等待终端显示 `BASELINE READY`，其后未触碰段用于观察噪声。
2. 点击曲线窗口，开始轻拉时按 **P**，保持约5秒；完全松手时按 **R**，至少等10秒。重复4次，尽量保持同一方向，不用持续加力追求大峰值。之后保持不碰至记录结束。
3. 120秒自动保存退出，也可在松手后的静止段按Q结束。P/R是人工动作标签；基线完成后终端一直显示 `BASELINE READY` 不表示要重新等基线。记录器只读，不会因拉拽自动松夹。

输出为 `logs/pull_torque/20261105_static_pull_release/<时间>/`，含 `samples.csv`、`summary.json`、`torque.png`、`torque.pdf`。保留终端打印的完整目录，并另记第一步抓取日志目录和现场是否确实完全松手。`--hz 20`是轮询设置，实际有效频率由记录重算；该脚本不是同步视觉／承重采集器。首轮先检查拉／松重复性与松手残余，之后再单独采碰／推、握住不拉等反例，不能由这轮直接选好自动释放参数。

本节为源码核查后的操作命令，尚未由助手执行硬件。

**21:02首轮用户实采已核查：** `logs/pull_torque/20261105_static_pull_release/2026-11-05_21-02-23_361135/`，1245条、约120秒、实际10.37 Hz，4段PULL。按键盘标签，首次P之前REST的J4最大偏差已约1.780 N·m，其中超过1 N·m连续约8.733秒；第一段PULL的J4峰值仅0.974 N·m。J2虽有较强峰值，但超过0.6 N·m的连续跨度在4段PULL中均不足2秒。当前不能据总峰值选自动释放阈值。操作者随后已确认0–25秒未触碰、P/R准确对应实际开始拉与完全松手，故首次P前的J4大偏移发生于无触碰阶段。见 [分段曲线与核查](artifacts/pull_torque_20261105_210223/README.md)。

当时为确认背景准备的60秒**全程不触碰**命令保留如下：同一夹持/静止保持，不重新抓取、不按P，不会开夹或重新使能。**随后21:26这轮已取得约68秒确认未触碰的拉前参照，不再要求为该目的重复执行。**

```bash
python tools/monitor_grasp_pull_torque.py \
  --channel can0 --firmware V112 \
  --plot --duration 60 --baseline-seconds 5 --hz 20 \
  --save-dir logs/pull_torque/20261105_static_untouched
```

**21:26第二轮实采与用户确认：** `logs/pull_torque/20261105_static_pull_release/2026-11-05_21-26-48_665149/`，1173条/约120秒/9.775 Hz，含唯一一次P约68.34–78.54秒。用户确认P/R对应实际拉/完全松手，之前和之后都不碰；不能将整轮归入全程未碰。J4拉前/拉时/松后最大偏差分别0.034/1.851/1.981 N·m，松后>1 N·m连续约8.495秒；J2拉时峰值2.577，但>0.6连续仅0.841秒。说明该次有交互响应，松后偏移却不能证明仍握持。先查拉后周期变化来源，再补同控制模式握住不拉/短碰负例，不启用自动开夹。见 [分阶段曲线与重算](artifacts/pull_torque_20261105_212648/README.md)。

## 8. 离线对比 GraspNet 与 GraspGenX：Nero 夹爪和桌面

本节只读取 `2026-10-22_23-11-33_150232_p234265` 保存的中心视角点云与候选，不连接或移动机器人。
GraspGenX 已适配 `models/nero_with_gripper.urdf` 的法兰、底座和两根手指 STL，夹爪资源位于
`third_party/GraspGenX/assets/x_grippers/nero_2f_85/`。按 URDF 关节位置计算的张开净间隙为 85 mm；
关节位置与实机宽度指令的对应关系仍需测量确认。

**直接看已保存的对比结果：**

```bash
cd /home/yx/yx/panda_ws/src/whole-grasp
RUN=logs/2026-10-22_23-11-33_150232_p234265
CAP=$RUN/2026-10-22_23-11-51_192823
third_party/GraspGenX/.venv/bin/python tools/compare_grasp_generators_viewer.py \
  --input-json "$RUN/graspgenx_input/center.json" \
  --graspnet-json "$RUN/generator_compare/graspnet/center__graspnet__seed42.json" \
  --graspgenx-json "$RUN/generator_compare/graspgenx_nero/center__nero_2f_85__seed42.json" \
  --table-preview-json "$CAP/final_grasp_preview.json" \
  --table-report-json "$RUN/generator_compare/graspgenx_nero/center__nero_2f_85__seed42_table.json" \
  --safe-only
```

保持终端运行，打开 `http://localhost:8081`。左侧是 GraspNet 原始候选的通用夹爪示意，
右侧是真实 Nero 张开夹爪网格和通过桌面检查的 GraspGenX 候选；红色网格表示标定桌面。
两个名次滑块分别切换候选。去掉 `--safe-only` 可查看全部 40 个 GenX 候选及桌面状态；
看完按 Ctrl+C 退出。

本次中心点云、随机种子 42 的 GenX 结果中，按完整张开夹爪网格、`table_z=0`、5 mm 最小余量和
100 mm 直线预抓取退让检查，**21/40 通过，19/40 不通过**。模型最高分候选通过，
但第 3 高分候选与桌面相交约 134 mm。GraspGenX 只输入目标点云，模型分数不能替代桌面碰撞检查，
也不能与 GraspNet 网络分数直接比较。此处只检查桌面平面；其他环境障碍、机械臂、自碰撞、
抓取接触、IK 与完整路径仍需共用现有工作流预检。缺失的 TSDF 与 PointNet VAE 表示会触发加载警告，
当前 `sweep_volume_v2` 检查点不使用这两项。

重新生成 Nero 候选时使用 `third_party/GraspGenX/scripts/demo_object_pc.py` 的
`--gripper_name nero_2f_85 --seed 42 --save_grasps_dir` 参数；桌面结果由
`tools/check_graspgenx_table.py` 生成。资源构建脚本为 `tools/build_nero_graspgenx_asset.py`，
现有资源已经生成，无需重复运行。

## 9. 香蕉真机八组对比：单／多视角 × GraspNet／GraspGenX × 原始／MGPC

**以下 9.2–9.9 是同一版 A–H 对比命令。** 八组统一启用 `--nero-grasp-refine`，使用 `--grasp-table-reference-z 0`、闭爪前位置容差 4 mm 和 25 mm 短抬升；GraspNet 先检查原姿态＋15 mm 推进，GraspGenX 先检查原生 TCP＋逐候选自动推进，均在没有完整可行候选时才启动姿态／深度修正。当前 A–H 的关节速度／加速度统一为 12°/s、8°/s²；本次只统一执行参数，抓取推进、候选修正和保护设置沿用现有配置。单视角与多视角的目标融合和桌面注册方式见下文。规划要求完整夹爪 CAD 离桌至少 5 mm，最终接近的运行阈值为 3 mm。截至 10 月 31 日，A–H 都已有至少一次功能成功记录，但此前轮次使用过不同速度／加速度及代码版本，不能直接拼成正式八组成功率。后续重复实验从 A–H 按统一参数采样；现有日志目录保留，使用各轮 run_id 区分记录，统计时核对保存命令并区分旧参数轮次。每次规划失败也如实记入该组，并单列为规划失败，不写作已经闭爪后的夹取失败。

MGPC 已作为可选候选输入接入主流程。GraspGenX 现通过独立 Python 进程生成 Nero 候选，先以真实张开夹爪网格排除桌面相交，
再经主流程的**实测目标夹持区、实际夹爪开口、IK、完整路径预检和运行时碰撞保护**。GenX 输出的夹爪资产法兰位姿经 URDF 与当前 TCP 偏置转换；
它不是 GraspNet 的姿态，也没有直接把模型分数当作执行许可。MGPC 只替换候选生成输入，安全几何仍取实测点云。
本轮按用户要求使用 100 mm 工作开口：safe 脚本、命令限幅、碰撞模型和 GenX 的 `nero_2f_100` 资产一致。
GenX 候选使用固定 100 mm 张开宽度作为安全检查输入，**该宽度不是 GenX 预测值**；两个网络的分数不能相互比较。
八组均使用 YOLO 的 `banana` 实例掩膜；中心或侧视角漏检时停止该组，不用手动点选来补一次自动检测失败。
八组测抓取、短距离脱离桌面，以及持物到指定关节位；成功后保持闭爪，不进入人手跟随／交接，不启用清障，不移动作为障碍的固定方块。固定持物位见 9.1.3，八组共用同一组角度。
每次都把目标香蕉放回同一位置和朝向，再运行下一组；一个终端只运行一组。

新版 A–H 日志统一写到 `logs/banana_mgpc/ablation_v3_nero_refine/`，`ablation_v2` 等旧目录保留排障记录。当前比较包含各生成器的适配和统一的失败候选修正：GraspNet 原生推进 15 mm、修正时搜索 0–30 mm；GraspGenX 原生自动推进和修正均在 0–15 mm 内。论文应说明这项差别；八组结果反映适配后整条抓取流程的表现，若要单独比较网络本身，还需统一或消融推进策略。先各组跑一轮确认流程，再固定参数做重复试验；规划失败计入总体任务失败，并单列失败阶段。

**首次 GenX 实机务必先核对候选预览与 `selected_grasp_pose.json` 的抓取 TCP 是否落在目标夹持区，法兰是否位于物体上方，夹爪开口方向是否正确；确认提示出现前机器人不会执行抓取。**
自制 GenX 夹爪资产的 URDF 开口为 100 mm，尚未通过实机量测验证手指网格与硬件的逐毫米对应；如果预览、TCP 或桌面高度不符，停止该组测试。

| 组别 | 目标输入 | 采集 | 日志目录 |
| --- | --- | --- | --- |
| A | 中心单视角实测 → GraspNet | 中心视角，无侧视角 | `logs/banana_mgpc/ablation_v3_nero_refine/A_single` |
| B | 多视角融合实测 → GraspNet | 中心＋1个侧视角 | `logs/banana_mgpc/ablation_v3_nero_refine/B_multi` |
| C | 中心单视角 → MGPC → GraspNet | 与 A 相同 | `logs/banana_mgpc/ablation_v3_nero_refine/C_single_mgpc` |
| D | 多视角融合 → MGPC → GraspNet | 与 B 相同 | `logs/banana_mgpc/ablation_v3_nero_refine/D_multi_mgpc` |
| E | 中心单视角实测 → GraspGenX | 与 A 相同 | `logs/banana_mgpc/ablation_v3_nero_refine/E_single_genx` |
| F | 多视角融合实测 → GraspGenX | 与 B 相同 | `logs/banana_mgpc/ablation_v3_nero_refine/F_multi_genx` |
| G | 中心单视角 → MGPC → GraspGenX | 与 A 相同 | `logs/banana_mgpc/ablation_v3_nero_refine/G_single_mgpc_genx` |
| H | 多视角融合 → MGPC → GraspGenX | 与 B 相同 | `logs/banana_mgpc/ablation_v3_nero_refine/H_multi_mgpc_genx` |

单视角组也先到与多视角组相同的中心观察位置。初始图像用于自动选目标和保留环境，不作为额外目标视角融合。
中心位置或要求的目标视角数未取得时，该组停止，不把失败的多视角测试悄悄计作单视角。

### 9.0 C／D／E 核对结果与下一步（历史记录；最新 E 真抓取命令见 9.0.3）

**本节保存历史排障证据。** E 已在 9.0.3 记录一次实机成功；当前 A–H 对比请复制 9.2–9.9，下面表格的试验安排属于旧版，不作为当前操作顺序。

| 组别 | 已进入主流程的修复 | 保存数据的最新结果 | 当前安排 |
| --- | --- | --- | --- |
| C | MGPC 中心参考相机应用桌面注册变换；完整张开夹爪网格检查提前到候选预检 | 同帧同种子：实测输入 18 个候选中 1 个通过接触＋网格对桌面检查，MGPC 的 10 个中 0 个通过 | 先做保存数据离线对照，暂停原命令反复试机 |
| D | 同 C；侧视角环境也独立注册，已确认采到并保留侧视角 | 同帧同种子：实测输入 13 个候选中 1 个通过两项检查，MGPC 的 14 个中 0 个通过 | 新增 2 mm 不闭爪试验；保存数据的候选 7 已通过路径预检，见 9.0.1 |
| E | Nero 100 mm 资产、原生 TCP 映射和共享安全预检已接入；固定开口不再按 GraspNet 预测宽度除以 1.2 或判断预测饱和 | 原始候选接触深入不足；前移 11 mm 后在默认规划 5 mm 设置下，接近路径间隙约 3.1 mm 不通过 | 新增 2 mm＋11 mm 前移的不闭爪试验，保存数据已通过路径预检，见 9.0.1 |

**“桌面网格”这个旧说法已更正。** C／D／E 共用的日志现在写明 `open gripper CAD mesh vs table plane`：是打开夹爪的 CAD 三角网格对桌面平面的模型检查；不是在桌面凭空增加一个实体障碍。GenX 开口日志改为 `fixed_asset_opening=100.0mm`，不再误写成网络预测的 `nominal=83.3mm`。规划 5 mm、最终接近打开夹指的运行阈值 3 mm、接触深入门槛 10 mm 均保留；改文字不代表这些候选已通过。

本轮已替你完成下列离线检查，均未连接硬件、未发送运动：

- [C／D 同帧同种子对照](artifacts/banana_planning_diagnosis/contact_20261028/mgpc_CD_same_capture/comparison.json)：复用各轮已有 MGPC 输出和实测场景，用该轮相同种子重新生成实测输入候选。上述 1 对 0 是接触＋网格对桌面的通过数，**尚不代表 IK、完整路径或实物成功**。
- [关闭上部输入裁剪的对照](artifacts/banana_planning_diagnosis/contact_20261028/mgpc_CD_no_focus/comparison.json)：MGPC 两轮仍为 0 个通过，不能靠简单加 `--no-graspnet-upper-focus` 宣布修好。
- [C／D 推进量探测](artifacts/banana_planning_diagnosis/contact_20261028/mgpc_CD_advance_probe.json)：尝试 0／5／10／15 mm，C 在 5 mm 时仅一个候选通过两项几何筛选，模型桌面间隙约 5.14 mm；D 均无通过。该 C 候选未做完整路径验证，**不要据此直接将 C 命令改为 5 mm 试机**。
- [E 前移 11 mm 的路径预检](artifacts/banana_planning_diagnosis/contact_20261028/genx_E_2116_route_probe.json)：接近段未通过环境点间隙检查；尚未进入持物关节位规划。它与桌面平面的 7.4 mm 几何间隙是不同检查，不能互相替代。

下一步先复核候选输入和位姿适配，不需要重新标定，也不需要为了这一步再次采图或运行 `--execute`。如需复现 C／D 对照，在现有环境运行这一条即可：

```bash
cd /home/yx/yx/panda_ws/src/whole-grasp
conda activate grasp_yolo26
python tools/audit_mgpc_saved_capture.py \
  --run-dir \
    logs/banana_mgpc/C_single_mgpc/2026-10-28_21-04-09_821276_p465766 \
    logs/banana_mgpc/D_multi_mgpc/2026-10-28_21-12-09_611263_p467921 \
  --output-dir artifacts/banana_planning_diagnosis/contact_20261028/mgpc_CD_same_capture
```

它只使用保存的点云和现有 GraspNet GPU 环境；不启动相机、CAN 或机械臂，不新建环境。最后每轮打印 `measured`／`mgpc` 的 `proposals`、`aperture`、`mesh` 数量。若要复现关闭输入裁剪的消融，给这条命令追加 `--no-focus`，并把输出目录改为 `artifacts/banana_planning_diagnosis/contact_20261028/mgpc_CD_no_focus`。本轮相关回归测试为 134 项通过，另 17 个子测试通过；这不是实机成功率。

#### 9.0.1 历史 2 mm 不闭爪诊断（当前 A–H 命令见 9.2–9.9）

模型间隙不能直接代替现场距离，所以现在提供单独的实机诊断设置：`--grasp-contact-check-clearance 0.002`。**仅在本次不闭爪试验中**，将最后接近的两根打开夹指对环境点、以及完整打开夹爪 CAD 对桌面平面的最低要求设为 2 mm；规划和运行都用 2 mm。手臂、肘部、相机和夹爪主体对环境点的检查、关节限位、力矩保护、NBV 运动沿用原设置，负间隙仍被拦截。常规正式抓取默认仍为规划 5 mm、最终接近打开夹指运行 3 mm。

到达后保存新的 RGB-D 和反馈位姿，然后按预检路径张开退回；不闭爪、不抬升、不去持物关节位。请在接近时从侧面录像，核对两根夹指与桌面的真实距离；不要把手伸进运动范围。若其他保护触发，程序按原逻辑停止，不能保证一定完成拍照和退回。

已用保存数据完成预检：[D 的 2 mm 试验](artifacts/banana_planning_diagnosis/contact_20261028/clearance_trials/D.json)、[E 的 2 mm 试验](artifacts/banana_planning_diagnosis/contact_20261028/clearance_trials/E.json) 均通过接近、支持脱离和张开回退路径计算；没有连接硬件。下面会重新采集、生成和规划，仍可能因新场景或新候选而拒绝。相关回归测试 155 项通过，另 27 个子测试通过。

**先跑 E：** 仅为此次诊断沿 GenX 原生接近轴额外前移 11 mm；这不是竖直向下固定移动 11 mm。该参数不能用于正常闭爪抓取，也不叠加 GraspNet 的 `--grasp-advance`。

```bash
cd /home/yx/yx/panda_ws/src/whole-grasp
conda activate grasp_yolo26
bash scripts/run_grasp_safe.sh \
  --config config/pipeline.yaml --phase grasp --execute \
  --target-class banana \
  --yolo-weights models/food5_visible_v3_full_reviewed_corrected_best.pt \
  --active-view --no-active-multiview-gate \
  --active-center-height 0.35 --active-center-roll-plan-candidates 1 \
  --no-active-center-optical-relaxation --active-capture-settle-seconds 1.0 \
  --preview-cloud \
  --trajectory-executor mit --mit-q-log \
  --mit-max-joint-vel-deg-s 5 --mit-max-joint-accel-deg-s2 20 \
  --active-side-view-count 0 --grasp-required-view-count 1 \
  --grasp-table-reference-z 0 \
  --grasp-generator graspgenx \
  --graspgenx-contact-advance 0.011 \
  --grasp-contact-check-only --grasp-contact-check-clearance 0.002 \
  --no-mgpc \
  --save-dir logs/banana_clearance_check/E_2mm
```

启动后的抓取规划日志应出现 `[motion][contact-check-trial] ... clearance=2.0mm ... NO CLOSE OR LIFT`，最终运动确认也会显示 `CONTACT CHECK` 和 `trial finger/table clearance=2.0mm`。正常完成结果为 `contact_checked`，终点数据保存在本次采集目录的 `contact_check/` 下，`contact_pose.json` 同时记录试验间隙与 GenX 推进量。

**再跑 D：** MGPC＋GraspNet 保留原来的 15 mm 推进，只调整诊断中的夹指/桌面间隙。

```bash
bash scripts/run_grasp_safe.sh \
  --config config/pipeline.yaml --phase grasp --execute \
  --target-class banana \
  --yolo-weights models/food5_visible_v3_full_reviewed_corrected_best.pt \
  --active-view --no-active-multiview-gate \
  --active-center-height 0.35 --active-center-roll-plan-candidates 1 \
  --no-active-center-optical-relaxation --active-capture-settle-seconds 1.0 \
  --active-fusion-rigid-icp --active-table-reg-max-height-shift 0.030 \
  --preview-cloud \
  --trajectory-executor mit --mit-q-log \
  --mit-max-joint-vel-deg-s 5 --mit-max-joint-accel-deg-s2 20 \
  --active-side-view-count 1 --grasp-required-view-count 2 \
  --grasp-table-reference-z 0 \
  --grasp-generator graspnet --grasp-advance 0.015 \
  --grasp-contact-check-only --grasp-contact-check-clearance 0.002 \
  --mgpc --mgpc-category banana \
  --save-dir logs/banana_clearance_check/D_2mm
```

**C 暂不安排实机命令：** 同样用 2 mm 做离线检查，并将推进改为 5 mm 后，候选 5 已通过接触和桌面几何筛选，但预抓取 IK 解贴住关节极限，余量约 0°，低于保留的 8° 要求。C 此时的障碍已经是关节可达性，继续降桌面阈值无效，见 [C 诊断](artifacts/banana_planning_diagnosis/contact_20261028/clearance_trials/C.json)。

#### 9.0.2 已完成的接触诊断：最低 −5 mm，不闭爪（下一次真抓取用 9.0.3）

09:16 这次 2 mm 指令已生效。新点云重新生成了两个 GenX 候选：候选 0 的左右接触深入约 15.1／11.2 mm，接近路径的最小模型间隙约 1.018 mm，未达到 2 mm；候选 1 的完整夹爪 CAD 算出对桌面间隙 −3.5 mm，因此两者均在抓取接近前被拒绝。这次没有电子急停；机械臂到过观察位，但没有执行抓取接近。末尾误写的 `no GraspNet candidate remains` 已改为生成器通用文字，实际用的仍是 GenX。

已定位候选 0 的最近环境点，基座坐标约 `[-0.4883, -0.0332, 0.00194] m`，即桌面上方约 1.9 mm；它可能含有桌面深度起伏，不能仅凭模型距离判断真实碰撞，见 [本次阻挡点诊断](artifacts/banana_planning_diagnosis/contact_20261029/E_0916_witness.json)。没有修改手眼标定、桌面高度或点云来制造通过结果。

此前把用户要求理解成只允许正间隙，范围过窄。用户已明确允许模型预测穿桌不超过 5 mm，因此本次使用 **`--grasp-contact-check-clearance -0.005`**：桌面这一项允许 −5～+5 mm，正间隙大于 5 mm 也照常允许；只有低于 −5 mm 才因该项拒绝。**−3.5 mm 不再单独构成桌面拒绝理由。** 该参数仍只允许用于不闭爪诊断，且要求已有的 `--grasp-table-reference-z 0`；正式抓取默认不变。

为避免两套模型重复给同一个桌面不同结论，负间隙诊断在**每个规划路径采样和实时反馈姿态**都计算完整张开夹爪 CAD 对桌面平面的距离，保留 −5 mm 边界。夹指对桌面高度 ±5 mm 测量带的检查由该 CAD／平面检查承担，不再让加粗胶囊叠加桌面点噪声后再次拒绝同一桌面。桌面测量带以外的障碍仍不允许夹指模型穿入；手臂、相机、夹爪主体对完整环境点的检查、关节限位、力矩和轨迹跟踪保护保留。没有修改标定或抬低桌面坐标。

已用**这次 09:16 的保存数据**回放，原先模型桌面间隙约 −3.54 mm 的候选 1，现已通过完整路径预检，见 [E 本次 −5 mm 试验回放](artifacts/banana_planning_diagnosis/contact_20261029/E_0916_minus5_trial_candidate1.json)。这是离线结果，没有连接或移动机械臂；下面仍会重新采图、生成候选和规划，不保证一定选择候选 1，其他检查仍可能拒绝。相关回归测试 155 项通过，另 37 个子测试通过。

**09:39 实机观察记录：** 用户观察到最低夹指轻轻接触桌面，随后 Ctrl+C 中断；程序确认位置保持，没有电子急停。本轮选中候选 1，计划终点完整 CAD 最低点为 −4.43 mm；按日志中停稳后的实际关节反馈重算，CAD 最低点为 −2.67 mm。该轮没有正常到位确认、终点 RGB-D 或退回记录，不能把计划终点和实际停下位置等同，也不能把模型负值说成实物压入桌面的深度。记录为“现场轻触桌面、停止反馈对应模型约 −2.7 mm”，不据这一帧全局平移标定或桌面，详见 [本次接触观察与停止反馈](artifacts/banana_planning_diagnosis/contact_20261029/E_0939_observed_touch.json)。计划模型最低点还比停下时低约 1.8 mm，尚未验证走完整个计划终点的实际接触情况。

**最终确认窗口的显示差别：** 当前 GenX 额外显示青色完整 Nero 夹爪 CAD；纯 GraspNet 的最终窗口主要显示简化的候选轮廓和执行夹指中心线，没有完整实体厚度。黄色线条也不是实际夹指外表面，不能直接用线条到点云的距离代表实体间隙。本节带 `--grasp-table-reference-z 0` 的 GraspNet／GenX 桌面检查都使用同一 Nero 完整 CAD，差异主要在预览显示，尚未统一。

**执行方式：** 关节速度上限为 2°/s，张开接近，到位拍照后按预检路径退回，不闭爪、不抬升。确认提示会明确显示 `CONTACT CHECK` 和 `trial finger/table clearance=-5.0mm`。请从侧面观察或录像实际间隙。Ctrl+C 会走受控停止并保持当前位置，**不保证瞬时停住，也不会自动退回**；现场准备好设备实体急停，手不要进入运动范围。

```bash
cd /home/yx/yx/panda_ws/src/whole-grasp
conda activate grasp_yolo26
bash scripts/run_grasp_safe.sh \
  --config config/pipeline.yaml --phase grasp --execute \
  --target-class banana \
  --yolo-weights models/food5_visible_v3_full_reviewed_corrected_best.pt \
  --active-view --no-active-multiview-gate \
  --active-center-height 0.35 --active-center-roll-plan-candidates 1 \
  --no-active-center-optical-relaxation --active-capture-settle-seconds 1.0 \
  --preview-cloud \
  --trajectory-executor mit --mit-q-log \
  --mit-max-joint-vel-deg-s 8 --mit-max-joint-accel-deg-s2 8 \
  --active-side-view-count 0 --grasp-required-view-count 1 \
  --grasp-table-reference-z 0 \
  --grasp-generator graspgenx \
  --graspgenx-contact-advance 0.011 \
  --grasp-contact-check-only --grasp-contact-check-clearance -0.005 \
  --no-mgpc \
  --save-dir logs/banana_clearance_check/E_minus5mm
```

若 D 也需要同样的桌面误差验证，将 9.0.1 的 D 指令改为 `--grasp-contact-check-clearance -0.005`、关节速度 2°/s、加速度 8°/s²，另存到 `logs/banana_clearance_check/D_minus5mm`。C 已找到的关节解贴近极限问题仍不会被这项间隙设置放行。

#### 9.0.3 E 已成功的配置与修复记录：自动推进，正常桌面保护（2026-10-29）

**11:03 本版本实机成功记录：** `2026-10-29_11-03-38_845411_p545323`，单中心视角、GenX、无 MGPC，用户现场确认香蕉被抓起；日志为 `status=success detail=executed`，完成短抬升和指定关节持物。自动选择推进 9 mm，两侧接触深入约 13.2／10.9 mm，计划 CAD 离桌约 7.5 mm；闭爪前实际位置误差约 2.9 mm、真实旋转误差约 0.47°，通过 4 mm／3° 检查。夹持在约 33.9 mm 开口处确认，力反馈绝对值约 0.50 N；持物后开口约 33.0 mm、力反馈约 0.45 N。该轮用正常 5 mm 规划／3 mm 实时桌面保护，没有使用 −5 mm 放行，也没有引入 −2.67 mm 全局高度补偿。记录见 [本轮结果](logs/banana_mgpc/E_single_genx/2026-10-29_11-03-38_845411_p545323/experiment_result.json)。这是一次实机成功，不是整体成功率；这帧到位误差也比前一轮小，不能从一次结果把成功全部归因于自动推进或 4 mm 容差。先保留这一版参数，下面旧回放记录属于此前的修复证据。

**不再继续放宽穿桌阈值。** 现场已经观察到轻触桌面，−5 mm 试验只能说明模型和实物存在差异，不能作为所有姿态安全的依据。此前多轮命令含 `--grasp-contact-check-only`，按设计不会闭爪或抬升，不能把它们统计成真正抓取失败。

本次修改的是模型使用方式和 GenX 的接触推进量：

- 带 `--grasp-table-reference-z 0` 时，正常抓取和诊断的最终接近均用**完整 Nero CAD 对已确认桌面平面**逐路径采样及逐实时反馈检查。桌面高度 ±5 mm 测量带由该平面检查承担，打开夹指不再用加粗胶囊叠加同一桌面点噪声重复拒绝；非桌面障碍、手臂、相机、主体、关节和力矩保护保留。
- 正常规划桌面最小间隙仍为 **5 mm**，最终接近的实时 CAD 桌面阈值为原来的 **3 mm**；没有使用 −5 mm 诊断放宽。有限的 CAD 桌面间隙越限先受控停止；未确认停止仍按原逻辑升级保护。
- GenX 现在用 **`--graspgenx-contact-advance auto`**：每个候选先试原生姿态，从 0 到 15 mm 按 1 mm 步长检查，选择同时满足实测双侧接触与完整 CAD 桌面余量的最小推进。已够深的候选保留 0 mm，浅候选才增加推进；没有可行深度就继续拒绝，随后仍必须过 IK 和完整路径。数值模式保持固定推进语义，默认仍为 0，GraspNet 的推进规则不变。

**10:31 这轮重新拒绝的原因：** 候选 0 在固定推进 9 mm 后 CAD 离桌仅 0.7 mm；候选 1 的右侧深入约 8.2 mm，低于保留的 10 mm；候选 2 在同样推进后 CAD 算出低于桌面 2.4 mm。这次没有进入抓取 IK，前一轮的闭爪角度修复没有机会运行。固定 9 mm 不能同时适合这三个不同的姿态，不能靠重复运行同一数值解决。

改为自动推进后，本帧候选 1 选择 **11 mm**，两侧深入约 **14.1／10.2 mm**、CAD 离桌约 **7.98 mm**，完整抓取和指定关节持物路线通过；候选 2 原生已足够深入，保留 **0 mm**，不再盲目加深，但仍须接受后续 IK／路径筛选，见 [10:31 本帧自动推进回放](artifacts/banana_planning_diagnosis/contact_20261029/E_1031_auto_advance/comparison.json)。另外两帧用同一算法自动选出的推进分别为 [09:39 的 8 mm](artifacts/banana_planning_diagnosis/contact_20261029/E_0939_auto_advance/comparison.json)、[10:14 的 9 mm](artifacts/banana_planning_diagnosis/contact_20261029/E_1014_auto_advance/comparison.json)，也通过完整路线与持物预检。三帧都复用了各自保存的网络候选和实测点云，没有重采集、重跑 GPU 模型或驱动机械臂；不是实机成功率。

自动调整只在 GenX 适配入口实现：入口确认实测目标上下文，扫描使用几何判断，没有用 try/catch 处理正常候选失败，也没有改变接触或桌面门槛。相关回归测试 239 项通过，另 66 个子测试通过。

用本次 09:39 **同一帧候选**回放：原来选中的候选 1 在 11 mm 推进时，模型桌面间隙 −4.43 mm；恢复正常保护后不再合格。候选 2 在 9 mm 推进时，左右接触深入量约 **13.0／11.4 mm**，完整 CAD 最低点距桌面约 **6.95 mm**；接近、短距离脱离桌面、张开回退与持物到指定七轴角的 4 段路径均通过，见 [9 mm 正常抓取完整预检](artifacts/banana_planning_diagnosis/contact_20261029/E_0939_9mm_full_grasp_preflight.json)。已成功的 A 组旧接近路径也通过新桌面检查回放，见 [A 组回归](artifacts/banana_planning_diagnosis/contact_20261029/A_success_normal_cad_regression.json)。相关软件测试 257 项通过，另 66 个子测试通过，均未连接机械臂。

**下面这条会闭爪。** 成功确认接触后短距离抬升，再去配置中的指定关节角 `[-0.002, 0, 0.002, 100.6, 0, 0, 0]°`，保持闭爪。已经删除 `--grasp-contact-check-only` 和负间隙参数。自动推进针对每个新候选计算，不再固定使用上一帧的 9 mm；新一轮仍会重新采集、生成和预检，不能预先保证实物抓取成功。

**10:14 本轮没有闭爪的原因与修复：** 这轮已经确认正常到达抓取位置，规划 CAD 离桌约 8.2 mm。之后的闭爪前检查仍要求位置误差不超过 3 mm，实际约 3.053 mm，略超；该检查还把欧拉角分量差的模长 3.68° 当作姿态误差，虽然真实相对旋转只有约 0.52°，与 MIT 到达检查的约 0.48° 一致。因此闭爪命令未发出，没有电子急停。现在闭爪角度判断复用 SO(3) 相对旋转角，门槛仍为 3°；**仅本节 E 命令增加 `--grasp-close-pos-tol 0.004`，默认 3 mm 不变**。这不是修改抓取深度或桌面阈值。见 [本轮闭爪检查与实际停止反馈回放](artifacts/banana_planning_diagnosis/contact_20261029/E_1014_close_gate_replay.json)。

同时，按本次实际停止反馈重新作为接触起点时，原来计划的 20 mm 支持脱离终点仅比实际起点高约 17.6 mm，达不到保留的 18 mm 确认要求。为避免下一步再次拒绝，本条命令用 `MOTION_SUPPORT_RELEASE_HEIGHT=0.025` 将计划短抬升设为 **25 mm**；safe 脚本通过这个变量读取高度，变量只作用于本条命令。已用本次点云重新检查：新接近路线、25 mm 支持脱离、持物到指定关节位均通过；从本次实际停止关节作为接触起点，到新脱离终点约上升 22.6 mm，也通过原保护，见 [4 mm 闭爪容差与 25 mm 短抬升预检](artifacts/banana_planning_diagnosis/contact_20261029/E_1014_close4mm_release25mm_preflight.json)。闭爪角度算法、超限仍拒绝闭爪及既有流程相关测试 161 项通过，另 58 个子测试通过。上述回放没有连接或驱动机械臂，实物夹持结果仍需验证。

```bash
cd /home/yx/yx/panda_ws/src/whole-grasp
conda activate grasp_yolo26
MOTION_SUPPORT_RELEASE_HEIGHT=0.025 bash scripts/run_grasp_safe.sh \
  --config config/pipeline.yaml --phase grasp --execute \
  --target-class banana \
  --yolo-weights models/food5_visible_v3_full_reviewed_corrected_best.pt \
  --active-view --no-active-multiview-gate \
  --active-center-height 0.35 --active-center-roll-plan-candidates 1 \
  --no-active-center-optical-relaxation --active-capture-settle-seconds 1.0 \
  --preview-cloud \
  --trajectory-executor mit --mit-q-log \
  --mit-max-joint-vel-deg-s 8 --mit-max-joint-accel-deg-s2 8 \
  --active-side-view-count 0 --grasp-required-view-count 1 \
  --grasp-table-reference-z 0 \
  --grasp-generator graspgenx --graspgenx-contact-advance auto \
  --grasp-close-pos-tol 0.004 \
  --no-mgpc \
  --save-dir logs/banana_mgpc/E_single_genx
```

规划日志应有 `[motion][table-policy] ... planning=5.0mm runtime=3.0mm`、`[graspgenx][contact-refine]` 和 `genx_native_tcp_advance = 实际值 mode = auto`；最终确认应包含 `close/support-release/joint-hold`，而不是 `CONTACT CHECK`。闭爪前通过时应打印 `close allowed`、位置容差 `0.0040m`，角度字段为 `rotation_err`。最终预览中的实体模型属于新候选，请确认后再执行；不要把旧的 −5 mm 诊断参数加回本命令。

### 9.1 先用相机实时检查 YOLO 掩膜（不连接机械臂）

`models/food5_visible_v3_full_reviewed_corrected_best.pt` 是仓库已有的五类食物实例分割权重，类别包括 `banana`、`radish`、`carrot`、`tomato` 和 `cucumber`。
把香蕉放在计划抓取的位置，用下面命令检查 `banana` 掩膜是否贴合目标。
同一个窗口还会显示该模型检出的其他类别及各自置信度；终端分别打印 `target detections` 和 `all detections`。
这一步只打开 D435i，不连接 CAN。按 Q/Esc 退出。

```bash
cd /home/yx/yx/panda_ws/src/whole-grasp
conda activate grasp_yolo26
python scripts/preview_yolo_camera.py \
  --weights models/food5_visible_v3_full_reviewed_corrected_best.pt \
  --target-class banana \
  --width 1920 --height 1080 --fps 30 \
  --imgsz 640 --conf 0.25
```

如本次先确认 100 mm 开口，退出相机预览后执行：

```bash
bash scripts/run_grasp_safe.sh \
  --config config/pipeline.yaml --phase grasp --execute \
  --gripper-test open
```

该命令启动时建立当前关节保持，随后只测试开爪并退出，不执行初始姿态、观察路径或抓取。终端应显示 `open width=0.1000m`；可在保持静止时核对两指内侧实际净距。

如需**单独让夹爪掉使能**，先放稳夹持物并结束抓取程序，再运行下面的 SDK 命令。它只连接 CAN 并发送夹爪掉使能指令，不调用机械臂关节使能／掉使能、运动或夹爪回零接口。`--disable-gripper` 只是让抓取流程跳过夹爪控制，不能代替硬件掉使能。

```bash
cd /home/yx/yx/panda_ws/src/whole-grasp
conda activate grasp_yolo26
python - <<'PY'
import time
from pyAgxArm import AgxArmFactory, ArmModel, NeroFW, create_agx_arm_config

config = create_agx_arm_config(
    robot=ArmModel.NERO, firmeware_version=NeroFW.V111,
    interface="socketcan", channel="can0", bitrate=1000000,
)
robot = AgxArmFactory.create_arm(config)
gripper = robot.init_effector(robot.OPTIONS.EFFECTOR.AGX_GRIPPER)
robot.connect()
try:
    deadline = time.monotonic() + 3.0
    while not gripper.disable_gripper():
        if time.monotonic() >= deadline:
            raise TimeoutError("未收到夹爪掉使能确认；请检查反馈和是否仍有其他程序控制夹爪")
        time.sleep(0.1)
    print("夹爪掉使能已由反馈确认；未发送机械臂关节控制指令。")
finally:
    robot.disconnect()
PY
```

`target detections: []` 表示当前帧和 0.25 阈值下没有检出 `banana`；
`all detections` 有其他类别也不代表香蕉被正确分割。预览中红色 `NO TARGET banana` 表示不能进入抓取测试。
Qt 的 `QFontDatabase: Cannot find font directory` 提示与 YOLO 检测为空无关。
**若没有准确的 `banana` 实例掩膜，就先不要执行 A–H。**
八组抓取命令使用与预览相同的权重、默认 640 推理分辨率和 0.25 初始识别阈值；MGPC 的形状提示词也改成 `banana`。

**MIT 真机测试前处理自动校时：** 10 月 27 日 14:40 的实测在显示融合点云期间触发 `MIT wall clock changed`。
连续控制日志显示墙上时间相对单调时钟回拨约 427.6 ms；系统日志在同一时刻 `14:41:10.196006` 记录 `systemd-timesyncd: Initial synchronization to time server`，确认是首次 NTP 同步引起。
这轮已经启用新机械臂模型和原始深度支持检查；GraspNet 尚未开始运行，不能归为抓姿、点数设置或碰撞过滤失败。HTTP 服务器的 `/favicon.ico` 404 只是浏览器找不到网站图标，与机械臂急停无关。

在抓取／伺服主程序已退出、机械臂静止时，测试前执行一次以下系统命令，再运行原 A–H 命令：

```bash
sudo timedatectl set-ntp false
timedatectl show -p NTP
```

应显示 `NTP=no`。这是暂停操作系统自动校时，不改变当前时间、不解除机械臂急停，也不发送机器人指令；机械臂若仍锁定急停，沿用已验证的设备恢复步骤，不要用放宽保护阈值代替恢复。
测试期间不要手动改系统时间或另外运行 `ntpdate` 等校时命令。所有机械臂程序退出后，再恢复自动校时：

```bash
sudo timedatectl set-ntp true
```

上述系统命令由你在终端执行，本次代码修改没有更改系统服务。控制反馈年龄、时钟跳变和急停保护保持开启；以后相同报错会直接显示 `changed by -427.6ms` 这样的跳变量。

本轮使用 100 mm 工作开口，到位检查要求反馈接近目标值；它不是对机械最大张度的实测结论。
配置现使用 `config/2026-10-26_camera_self_mask_100mm_multiframe.json`：由四次保存的 100 mm 图像建立共同稳定区域和已观测边缘包络，解决单张图边缘差几像素就被拒绝的问题。
核心、稳定轮廓和包络外区域仍使用原检查阈值；线上不平移或扩大遮罩。只有包络内符合实测黑色夹指颜色的像素被删除，带颜色的香蕉接触边缘保留。
四次图像及四次逐帧留出验证均通过，85 mm 图像仍会被拒绝。离线检查报告和边缘叠加图位于 `artifacts/camera_self_mask_100mm_validation/`；实机每个新视角仍须通过校验。
若再次失败，程序返回 `capture_failed`，不进入该帧的抓取规划；采集目录保存红线轮廓诊断图 `camera_self_mask_mismatch.png` 和检查报告 `camera_self_mask_report.json`。

下列八条抓取命令各自完整；每个新终端只需按本文件开头先 `cd` 并激活 `grasp_yolo26`，然后复制一组执行。
相机配置、YOLO 重新识别策略和随机种子沿用配置文件及程序默认值；`run_grasp_safe.sh` 默认开启抓取验证模式、路径和运行时碰撞保护，不需要预设环境变量。
原八组命令直接使用本轮 100 mm 默认开口，无需新增参数。终端应显示 `open width=0.1000m`，到位确认仍要求反馈接近目标值。
初始点云、融合点云、候选及最终抓姿的 Open3D 窗口均在独立进程打开，终端应出现 `[preview] isolated Open3D worker ...; no CAN ownership`。
父进程等待期间持续检查 MIT 状态；控制停止请求会关闭窗口并向主流程传递。关闭预览窗口后仍按原流程等待运动确认。
该改动消除了原来窗口与 MIT 发送/反馈线程共享解释器的阻塞路径，已通过模拟窗口持有解释器锁、窗口崩溃和控制故障的离线测试；实际 CAN/SDK 连续性仍需本轮实机日志确认。
23:10 日志进一步定位到关闭融合窗口后的 GraspNet 计算阶段：MIT 周期为 103.8 ms，超过原有 100 ms 限制。现在 MIT 持有控制权时，GraspNet 的输入采样、模型加载、推理和初筛也在独立进程执行。
关闭融合窗口后应出现 `[graspnet][worker] pid=...; isolated load/inference/filtering; no CAN ownership`，计算期间父进程持续检查控制状态，结果保存在本轮采集目录的 `graspnet_worker/`。加载或推理失败不会退回主进程计算。
MIT 保持期间的缓存释放不再执行显式全量 Python 垃圾回收；业务阶段日志也不再覆盖真实的 `hold/trajectory/stop` 控制状态。控制周期与反馈保护阈值沿用原值。
已用本次保存的香蕉输入做离线 GPU 回放，得到 17 个候选，模拟父进程心跳最大间隔约 5.5 ms；这不是实际 CAN 时序测量，复测仍需确认完整控制日志。

10 月 27 日 09:49 的 A 组实测运行 115 秒，本轮未复现控制急停：开口反馈为 99.996 mm，初始与中心采集通过遮罩检查，预览和 GraspNet 子进程正常返回，退出时以新鲜反馈确认位置保持。
本轮结果为 `planning_failed`：生成 15 个候选，12 个通过开口检查，完整路径通过数为 0；主要拒绝原因为肘部／相机碰撞、关节余量不足或 IK 无解。在线只进行快速分支搜索（`fast_ik=48, full_ik=0`），不能据此断言所有 IK 分支均不可行。
离线复查发现一处反复阻挡候选 0、9 的可疑深度：initial 图像像素约 `(571, 1029)` 位于白色桌面区域，深度为 305 mm，变换至基座后高度约 228 mm；按当前桌面模型，该像素深度应约 521 mm。原始七帧中有五帧提供此近距离返回，故仅去掉 SDK 补洞也不能保证消除它。
该点投影到 center 图像之外，另一视角无法证实或否定。额外增加到每个端点 16 个 IK 种子的离线检查中，候选 0 保留的八个抓取端点解仍与它相交；检查不包含整段路径，不证明全局无解。
诊断结果见 [summary.json](artifacts/banana_planning_diagnosis/summary.json)，位置见 [初始视角红圈图](artifacts/banana_planning_diagnosis/initial_witness_small.png)。本轮仅归档诊断，没有扩大夹指遮罩、删除可疑障碍点或降低碰撞／控制保护阈值；残留异常深度仍待修复。

10 月 27 日碰撞模型更新：机械臂部分默认改用 `urdf_capsules`，从当前 IK URDF 的七段 collision STL 拟合各段包络，使用各段实际坐标和关节旋转，不再对全部关节／连杆统一套 60 mm 半径。
当前拟合半径为 16.5–58.6 mm，所有碰撞网格三角形均被完整覆盖，并留 3 mm 几何余量；NBV、抓取规划和运行时碰撞检查共用此模型。模型在连接机械臂前预加载和检查，缺少或不支持的网格会报错，不会自动降级到旧模型。
**下面 A–H 命令无需增加参数。** 启动应出现 `[collision][arm] model=urdf_capsules ... all_collision_triangles_covered=true`。相机、支架、夹爪和点云占用策略沿用现有模型；3 mm 是 CAD 包络余量，不是实机测量精度。
相同保存关节解、相同环境点云的离线对比：候选 0 的预抓取间隙约 9.1 → 24.7 mm；抓取端点间隙约 −31.6 → −16.0 mm，仍被拒绝。模型更贴合 CAD 不代表残留假深度已消失，也不代表完整路径已通过。
对比图见 [机械臂 CAD 与新旧包络](artifacts/banana_planning_diagnosis/arm_collision_comparison.png)，完整数值和源网格哈希见 [arm_collision_audit.json](artifacts/banana_planning_diagnosis/arm_collision_audit.json)。离线单姿态计算中位耗时约 8.5 → 9.2 ms；这不是 CAN 实时性或实机安装精度验证。

复查同一组保存数据（不连接相机或机械臂）：

```bash
python tools/audit_arm_collision_geometry.py \
  --run-dir logs/banana_mgpc/A_single/2026-10-27_09-49-20_678020_p113011 \
  --endpoint-json artifacts/banana_planning_diagnosis/endpoint_probe.json \
  --output-dir artifacts/banana_planning_diagnosis
```

10 月 27 日深度伪点修复：采集现在对**同一次采样窗口的原始、未经过 SDK 滤波的深度**执行已有的有效帧数和 MAD 检查，默认仍为七帧中至少五帧、MAD 不超过 4 mm。SDK 连续补出的同一个数值不再算作七次真实观测；滤波中值与稳定原始中值相差超过 `2 × MAD阈值`（默认 8 mm）时，也置为无效。
这不是按白色、物体类别、高度或桌面假设删点；原始窗口支持充分的测量继续使用 SDK 的降噪结果。无效深度不会被替换成桌面点，也没有新增“已确认自由空间”的标记。原本依赖补洞的局部点会减少，不能保证每根细线的全部表面都有有效深度。
原始窗口支持不足的小片在已有的地图离群点规则下被剔除：同一保存数据的运动环境点数 22855 → 21235，原悬空点位置到最近环境点的距离约 206.5 mm；该位置不再形成肘部碰撞。香蕉目标点云 initial 仍有 6023 点、center 仍有 4226 点，红方块与左侧线缆的有效测量也保留。可看 [地图修复前后对比图](artifacts/banana_planning_diagnosis/raw_depth_fix/environment_comparison.png)；像素级保留比例、点云和排除区域见 [raw_depth_fix/replay.json](artifacts/banana_planning_diagnosis/raw_depth_fix/replay.json) 与其 `initial/`、`center/` 子目录。
相同关节解、新机械臂模型下，候选 0 的预抓取间隙约 24.7 → 55.6 mm；抓取端点改由 `gripper_finger_-1` 报告约 −0.7 mm 间隙，仍不满足 5 mm 要求。修复这一处深度伪点不代表所有候选或整段抓取路径都已通过。
MIT 模式下，原始深度校验在数值子进程执行，高清深度通过临时 `.npy` 文件交换，避免在父进程序列化超过 100 MB 的数组。1920×1080、七帧离线验证中，模拟父进程心跳最大间隔约 2.6 ms；这是软件并发验证，尚不是实际 CAN 运行结果。临时文件在完成或失败时清理，工作进程失败不会退回父进程计算。
**A–H 命令继续原样使用。** 每次采集应看到 `[camera][depth_support] raw_frames=7 min_valid=5 ...`。采集和 `multiview_raw` 归档新增 `depth_sdk_m.npy`（校验前 SDK 中值）、`depth_raw_valid_count.npy`、`depth_support_rejected.png`、`depth_support_report.json`；`depth_m.npy` 保存校验后的深度，完整原始窗口继续保留。

离线复查上述深度修复（不连接相机或机械臂）：

```bash
python tools/replay_raw_depth_support.py \
  --run-dir logs/banana_mgpc/A_single/2026-10-27_09-49-20_678020_p113011 \
  --endpoint-json artifacts/banana_planning_diagnosis/endpoint_probe.json \
  --output-dir artifacts/banana_planning_diagnosis/raw_depth_fix
```

10 月 27 日 15:08 的 A 组日志为 `planning_failed`，没有触发电子急停：19 个候选中 16 个通过开口检查，预算内检查了 12 个，均未通过；另外 4 个未检查，不能据此判定全部候选不可行。
保存点云与 RGB 投影核对显示，候选 0 的手指碰撞对应红方块，候选 1、12 对应桌面。本轮命令的 `--grasp-advance 0.015` 会在生成抓姿之外再沿接近方向推进 15 mm，压缩桌面间隙。
此前将推进改成 0 后，候选 12 虽然通过路径预检，但这只证明了避碰，未证明接触深度足够。16:43、16:51、16:54 三次实测均出现浅抓；16:54 夹爪闭至约 0.06 mm 仍未检测到接触，闭合前法兰位置误差约 1.4 mm。
**下面香蕉 A–D 命令改为 `--grasp-advance 0.015`，同时安全脚本默认开启双侧接触深度硬检查。** 每侧接触带沿接近方向的中位位置，必须距直线夹指前端至少 10 mm；只在指尖边缘擦过的点不再使候选通过。该值是本轮软件筛选阈值，不是实测接触精度。该检查同样用于 E–H 的实测点云，无需额外添加命令参数。
三次保存的实际选中候选在推进 0 时均被新检查拒绝。16:54 候选 2 推进 15 mm 后，左右接触深入量分别为 16.4/12.2 mm，并通过当前验证模式的预抓取、接近、验证抬升与空抓恢复路径预检，接近路径最小间隙为 8.7 mm。推进 30 mm 时该候选的手指已与环境相交，不能直接叠加 GraspNet 的 30 mm 深度。
碰撞间隙仍要求 5 mm，开口仍为 100 mm。未连接机械臂验证修改后的实际接触，也未验证搬运／交接；不同场景中 15 mm 仍可能无可行候选，届时应停止而不是降低间隙。
离线证据见 [三次接触深度对比](artifacts/banana_planning_diagnosis/contact_depth_1654/geometry.json) 和 [15 mm 完整预检](artifacts/banana_planning_diagnosis/contact_depth_1654/advance_15_preflight.json)。启动日志应包含 `min_contact_inset=10.0mm`，浅抓候选的拒绝原因会显示 `shallow bilateral contact`。

**18:49 实机复测修正：上述接触筛选仍未解决真实空抓，先核对几何／坐标链，不再通过增加推进量反复试抓。** 该次选中候选 9，推进确为 15 mm，软件计算的左右接触深入量为 20.7/14.1 mm，但实机仍闭至约 0.1 mm 无接触，法兰反馈到位误差约 1.4 mm。这说明现有模型中的接触支持不等于实物接触，不能再据此把原因直接归为 GraspNet 深度不足。
按当前 `models/nero_with_gripper.urdf`、100 mm 开口与手眼外参投影，两个夹指的预测区域都未覆盖保存的夹指遮罩。对比图中青色为 CAD 预测、紫红色为已标定遮罩：[夹指投影对比](artifacts/banana_planning_diagnosis/contact_1849/cad_vs_observed_fingers.png)。该诊断依赖“URDF link7 等于 SDK flange”“CAD／开口映射符合实物”“遮罩对应夹指”三个假设，不能单凭投影认定只有 TCP 或手眼外参有错。
同一静止桌面在 initial 与 center 的未配准点云中，目标附近平面高度相差约 10 mm。报告见 [cad_projection.json](artifacts/banana_planning_diagnosis/contact_1849/cad_projection.json)。这些现象不能单独证明已有手眼标定失效，也可能涉及深度、安装映射或诊断模型。实物模型一致、紫红色区域为活动夹指已由用户确认，当前保留已有标定，先核对抓姿到真实夹指接触区域的转换；不要求重新标定。下述固定标记核验仅为可选诊断。没有用此投影自动修改 TCP、手眼外参、碰撞模型或推进值。

复查上述几何诊断（只读保存数据，不连接机械臂或相机）：

```bash
python tools/audit_gripper_camera_geometry.py \
  --capture-dir logs/banana_mgpc/A_single/2026-10-27_18-49-53_441770_p277939/2026-10-27_18-50-12_218069 \
  --output-dir artifacts/banana_planning_diagnosis/contact_1849
```

### 9.1.1 可选诊断：固定标记的只读核验（不要求重新标定）

保存的两个姿态中，SDK 法兰反馈与 IK URDF `link7` 的位置差均小于 0.02 mm，旋转差小于 0.003°，暂未发现两者坐标系混用。这是软件坐标一致性，不是独立的实物精度测量。
现用标定的 20 帧训练角点，用单个手眼外参和固定标定板位姿统一解释时，重投影坐标分量 RMS 约 12 像素。仅对同批训练角点做联合优化后仍约 10.5 像素；该候选没有接入流程。这只是对旧样本的诊断，受固定板、角点及采集时序等假设影响，不能据此推翻已有实测验证，也不作为要求重新标定的依据。

先退出抓取程序，避免两个程序同时占用相机和控制资源。使用 **10 月 22 日同一块 `DICT_5X5_100` 标定板**，固定在桌面上。下列 `20.95` 是该板配置中的黑色 ArUco 标记边长，单位 mm，不含白边；若换了其他板或打印比例变过，必须填写实测边长，未确定时可删除 `--marker-length-mm 20.95`，此时只报告深度一致性。

```bash
python tools/verify_handeye_aruco_consistency.py \
  --config config/pipeline.yaml \
  --calib-json config/2026-10-22_charuco_handeye.json \
  --aruco-dict DICT_5X5_100 \
  --marker-length-mm 20.95 \
  --camera-depth-filters off \
  --samples 9 \
  --save-dir logs/handeye_contact_check
```

该脚本现在只连接机器人反馈，不使能、不设置速度、不发送运动／夹爪／停机指令。相机使用 pipeline 的分辨率，核验时关闭深度补洞，另用已知标记尺寸计算不依赖深度的 RGB PnP 位置。
通过现有示教方式选择三个有位置／朝向差异、且能看到同一标记的观察姿态，每个姿态停稳后按 Enter 采三次；标定板保持不动。程序自动锁定首次成功采到的标记 ID，后续看不到同一 ID 就跳过；采集前后法兰移动超过 1 mm 或 0.3°也会跳过。

- 同一姿态的三次结果先用来检查重复噪声，不能只在一个姿态按九次就判断手眼通过。
- 深度结果随视角明显漂移、RGB PnP 较稳定：优先检查深度／对齐；PnP 的判断依赖准确的标记尺寸和可用角点。
- 两路都随视角明显漂移：优先检查手眼外参、机器人位姿与采集时序；该现象本身不能唯一确定哪一项错误。
- 两路都稳定而实际指尖仍偏离：需进一步核对 TCP／夹爪安装变换和开口映射；跨视角稳定也不等于绝对接触精度已验证。

终端分别显示深度和 `[rgb-pnp consistency]` 统计，完整结果保存为 `logs/handeye_contact_check/handeye_aruco_consistency_*.json`。这一步用来确定下一处修正，不执行抓取，也不自动覆盖外参。

### 9.1.2 修复后的验证顺序：单视角 → 多视角 → 八组

**启动前的位置：** 新命令会重新采集、规划并移动到初始观察位，不会从上次停止位置继续。如果机械臂仍在香蕉附近，先由你按熟悉的低速示教／受控流程撤离至安全观察位置，再启动本节命令。

本次已修改：

- 接近容差现在实际生效。规划仍要求 5 mm；只有 `target_approach` 中打开夹指对环境点的运行阈值按 `5−2=3 mm` 判断。执行前重检也允许**实际起点**沿用这个手指容差，随后所有未来路径采样仍按规划阈值检查，避免到达后因 3／5 mm 衔接不一致而无法退回。连杆、本体、相机、桌面检查、力矩保护和接触确认沿用原设置；原有自由运动手指容差不变。
- 绝对桌面参考支持中心＋侧视角。目标的现有融合、视角筛选和滤波保留，通过中心的注册变换更新世界坐标；初始、中心和每次侧视角的完整环境分别注册到已确认的桌面 Z=0。原始文件和手眼文件保留。

你认可的上次接触位置已经保存在 [深度参考记录](artifacts/banana_planning_diagnosis/contact_20261028/depth_reference_comparison.json)：计划 TCP Z=23.77 mm，最后记录的保持位 TCP 约 20.84 mm。它只是该场景参考，尚未闭爪成功，不作为全局固定高度，也不直接重放。

**第一步：单视角张开检查。** 先退出 9.1 的相机预览，避免占用相机。下面仍使用 food5、15 mm 推进和 100 mm 开口，到达终点拍照后退回；不会闭爪或执行抓取抬升。中途保护停止时不保证自动退回。

```bash
bash scripts/run_grasp_safe.sh \
  --config config/pipeline.yaml --phase grasp --execute \
  --target-class banana \
  --yolo-weights models/food5_visible_v3_full_reviewed_corrected_best.pt \
  --active-view --no-active-multiview-gate \
  --active-center-height 0.35 --active-center-roll-plan-candidates 1 \
  --no-active-center-optical-relaxation --active-capture-settle-seconds 1.0 \
  --preview-cloud \
  --trajectory-executor mit --mit-q-log \
  --mit-max-joint-vel-deg-s 15 --mit-max-joint-accel-deg-s2 60 \
  --active-side-view-count 0 --grasp-required-view-count 1 \
  --grasp-table-reference-z 0 \
  --grasp-contact-check-only \
  --grasp-generator graspnet \
  --grasp-advance 0.015 \
  --no-mgpc \
  --save-dir logs/banana_mgpc/contact_check_v2
```

按提示确认初始位、中心观察及最终运动，关闭点云和候选预览后回终端。最终提示必须包含 `CONTACT CHECK` 和 `NO CLOSE OR LIFT`。候选位置异常时输入 `no` 取消。

终端应出现：

```text
[motion][clearance-policy] planning=5.0mm runtime=5.0mm target_approach_open_fingers=3.0mm configured_slack=2.0mm
[table-reference] initial: ... reference=0.0mm ...
[table-reference] center: ... reference=0.0mm ...
```

完成后应有 `[contact-check] RGB-D and final open-gripper pose saved: .../contact_pose.json` 和 `status=contact_checked`，并正常退回。确认侧面观察的位置合适后，用下方 **9.2 A 命令**做一次实际夹取验证；A 命令会闭爪并在接触确认后做验证抬升。

最终姿态预览的颜色：黄橙色线框是实际执行夹指的中心线（没有画出厚度）；绿色短线和青色宽度线是生成器的原始候选，尚未叠加执行推进／居中修正。黄色球是最终 TCP，蓝色球是预抓取 TCP，紫色线连接两者；红／绿／蓝粗箭头是 TCP 的 X／Y／Z 坐标轴。判断实际执行位置以黄橙色线框为准。

**10 月 28 日 19:27 这次验证：** 已到达并保存终点 RGB-D，拍照期间位姿稳定；用户现场确认当前位置适合夹取。照片在 [终点彩色图](logs/banana_mgpc/contact_check_v2/2026-10-28_19-27-00_693665_p383460/2026-10-28_19-27-21_067148/contact_check/2026-10-28_19-28-57_349751/color.png)，位姿在同目录 `contact_pose.json`。旧代码退回前把实际起点约 4.5 mm 的模型间隙与 5 mm 规划阈值比较，尚未发送退回指令就拒绝并保持，最终状态为 `execution_failed`；照片已保存不代表退回已完成。上述起点容差已补上，离线回放见 [退回修复验证](artifacts/banana_planning_diagnosis/contact_20261028/run_1927_return_fix.json)。仍未实际闭爪，不记为抓取成功；后续 A 命令重新采集规划，不是从此停止点继续闭爪。

**10 月 28 日 19:41 的 A 组停止已修复：** 这次停在预抓取，未执行最终接近、闭爪或抬升。到位时关节误差 1.12°，法兰位置误差 2.28 mm、姿态误差 1.39°；旧 MIT 自由运动到位门限为 1°，随后异常处理又因错误传入 `hooks` 抛出 `TypeError`。现在仅非 NBV 的 `free_open` 阶段沿用已有 `move_joint_tol`，并受下一阶段起点容差约束，当前为 2°；其位置／姿态／速度和稳定采样要求保留，最终接近及闭爪后阶段仍用原 1° 关节门限，闭爪前 3 mm／3° 检查保留。新增 `[arrival-policy]` 会打印实际门限。异常处理已修正，超限仍返回原始失败原因并跳过后续动作；61 项相关离线测试通过。**继续使用下方 9.2 A 原命令，不增加参数**，也不再加深抓取；启动前位置要求见本节开头。

**10 月 28 日 20:01 的 A 组已确认夹持接触：** 反馈开口约 35.5 mm，连续接触力约 0.50 N，程序已确认接触。随后在抬升前，旧代码把实际到位比名义抓取位置低约 2.47 mm 判断为 `support_release_downward`，起点即被拒绝，尚未发送抬升指令。现在只在接触确认后记录一次实际关节对应的 TCP 作为支撑释放起点，并同步更新物体相对 TCP 的表示，使起点的物体世界点云保持原位；原关节目标、1 mm 向下容差、18 mm 后恢复严格碰撞检查及其他保护保留。记录为本次采集目录的 `post_close_contact_reference.json`，终端显示 `[grasp][contact-reference]`。后续各段共享同一个起点，不能每段重新归零掩盖下落。相关 111 项离线测试通过，这次保存数据的原抬升路径回放通过，见 [抬升起点修复回放](artifacts/banana_planning_diagnosis/contact_20261028/run_2001_support_reference_fix.json)；实机提起未验证，本次结果仍保留 `execution_failed`。**夹着香蕉时先按熟悉的现场流程放开并恢复观察位，再重跑下方 A 原命令**，不会从旧停止点继续。A 的当前验证计划成功后只短距离抬离支撑面约 2 cm 并保持闭爪，不自动递送或回初始位。

**10 月 28 日 20:12 的 A 组短抬升流程通过：** [本次结果](logs/banana_mgpc/A_single/2026-10-28_20-12-31_090676_p400842/experiment_result.json) 为 `status=success`。接触开口 35.177 mm、反馈力约 0.494 N；闭爪后实际起点更新生效，支撑释放路径检查与实机到位均通过。抬升后的 5 次反馈均有效，开口约 34.94 mm、最大开口变化 0.246 mm，反馈力 0.421–0.429 N，夹持保持代理指标为稳定。计划只包含约 2 cm 的短抬升，结束后保持闭爪和电机使能，不自动返回；未发生电子急停。日志仅验证控制流程与夹持反馈，实验的物体离桌判定以现场观察为准。旧完成提示 `full lift/place were not executable` 已改为 `outside this validation-only plan`，明确当前验证计划没有安排完整抬升和放置，不据此认定完整路径 IK 失败。保持本次参数，现场确认离桌后继续下面的多视角检查与 B 组。

**第二步：多视角张开检查。** 单视角检查与 A 夹取通过后，再运行下面命令。必须取得中心＋1个侧视角，不能用漏检或被拒绝的视角凑数。

```bash
bash scripts/run_grasp_safe.sh \
  --config config/pipeline.yaml --phase grasp --execute \
  --target-class banana \
  --yolo-weights models/food5_visible_v3_full_reviewed_corrected_best.pt \
  --active-view --no-active-multiview-gate \
  --active-center-height 0.35 --active-center-roll-plan-candidates 1 \
  --no-active-center-optical-relaxation --active-capture-settle-seconds 1.0 \
  --active-fusion-rigid-icp \
  --active-table-reg-max-height-shift 0.030 \
  --preview-cloud \
  --trajectory-executor mit --mit-q-log \
  --mit-max-joint-vel-deg-s 15 --mit-max-joint-accel-deg-s2 60 \
  --active-side-view-count 1 --grasp-required-view-count 2 \
  --grasp-table-reference-z 0 \
  --grasp-contact-check-only \
  --grasp-generator graspnet \
  --grasp-advance 0.015 \
  --no-mgpc \
  --save-dir logs/banana_mgpc/contact_check_multi_v2
```

除了中心视角，应出现 `[table-reference] side1_capture_...` 等侧视角注册日志；报告在本次采集目录的 `table_reference/table_reference_report.json`，检查 `views` 包含初始、中心和侧视角。正常结束同样应为 `contact_checked` 并退回。随后用 **9.3 B 命令**实际夹取；A/B 基础验证通过后再按 C–H 逐组测试，GenX 首次执行仍需核对真实夹爪姿态预览。

这次离线验证使用上次实际反馈的 204 个记录样本：旧逻辑有4个低于5 mm的记录，新逻辑下这批已记录样本不触发保护；不代表未执行的后续运动必然通过。见 [运行时回放](artifacts/banana_planning_diagnosis/table_runtime_fix_validation/runtime_replay.json)。保存的真实多视角数据也完成了三视角桌面注册，见 [桌面回放](artifacts/banana_planning_diagnosis/table_runtime_fix_validation/table_replay.json)。实际抓取效果仍由这次验证决定，不降低其他保护阈值，不通过手动点选掩膜补自动识别失败。

### 9.1.3 抓住后持物抬到指定关节位（已默认启用）

按现场截图，七轴目标依次为 **`[-0.002, 0, 0.002, 100.6, 0, 0, 0]°`**，配置在 `config/pipeline.yaml` 的 `grasp.post_grasp_hold_joints_deg`。下面 A–H 原命令都有 `--config config/pipeline.yaml`，直接使用，不需要另加参数。

执行顺序：**接触确认 → 短距离抬离支撑面 → 持物运动到指定七轴角度 → 保持闭爪**。路径在闭爪前由独立数值进程检查；沿用现有 RRT-Connect、关节余量、全臂／相机／夹爪／物体碰撞检查与执行监控，目标角度不会被其他 IK 解替换。目标或路径不通过时，该轮在闭爪前拒绝执行。闭爪后仍按实际接触位姿更新物体相对 TCP 的位置，每段执行前重新检查。

终端确认应包含 `support-release/joint-hold` 和七轴目标；正常结束显示 `[done] grasp and requested joint hold completed`、`status=success`，随后保持闭爪和电机使能，不自动放开或返回。按当前模型，这个目标的 TCP Z 约 0.368 m，夹指和物体的实际最低点由各自模型检查。

`--grasp-contact-check-only` 仍只拍照、张开撤回，不执行此持物动作。如果只想复测此前的短抬升，可在相应命令中增加 `--no-post-grasp-hold`。下一轮为重新采集与规划；若当前仍持物，先按熟悉的现场流程放下并恢复观察位。

刚才 20:12 成功数据的离线检查得到三段持物路径，终点保持截图原角度；再用该次实测接触位姿更新物体模型后，三段细采样检查也通过。见 [指定关节位离线回放](artifacts/banana_planning_diagnosis/contact_20261028/joint_hold_replay.json)。未连接硬件执行这个新动作，实机到位与持物稳定性仍需下一轮验证。

**20:34 这次停止发生在抓取前：** 候选 1 的固定持物路线已通过，但完整打开夹爪网格的桌面间隙只有 3.2 mm，低于 5 mm，最终检查拒绝。此前简化模型给该候选约 5.5 mm，因此先被选中，整轮在末尾才取消。现在相同的完整网格检查提前到每个候选的实际推进／居中姿态，失败候选不进入 IK 路径规划和最终排序；通过的同一姿态检查结果可直接复用，姿态变化时重新检查。5 mm 阈值、15 mm 推进和指定七轴目标保留。共享检查的 `[graspgenx]` 旧报错标签已改为 `[grasp][mesh-table]`，不会把 GraspNet 误显示为 GenX。

本次保存数据复查：候选 1 拒绝，候选 2／4 的完整网格桌面间隙为 7.18／6.45 mm，按原推荐规则改选候选 4；这只是离线筛选结果，新一轮会重新生成与检查。相关 119 项测试通过，见 [候选网格筛选回放](artifacts/banana_planning_diagnosis/contact_20261028/run_2034_mesh_filter.json)。继续使用下面 A 原命令，不增加参数；本次日志 `motion_sent=True` 包含此前观察位运动，不表示已经闭爪或执行指定持物抬升。

### 9.1.4 11:39 A 与 11:42 B 的停止原因及修复（当前生效）

- **A 已确认夹持并完成短抬升，未到指定持物位。** `2026-10-29_11-39-46_357873_p567407` 在约 35.2 mm 开口处确认夹持，随后支撑脱离通过。搬运到 `target_joint_hold_waypoint2_waypoint_1of2` 时，末端位置误差约 1.84 mm、姿态误差约 1.02°，速度已为零，但最大关节误差约 1.0217°，略超旧 1° 到位门槛，等待 5 秒仍未收敛而停止，没有电子急停。现在 `free_open` 与 `carrying` 共用既有运动到位容差，并取下一段起点容差及跟踪限制的较小值；当前为 2°。下一段仍从实测关节重新检查路径，末端位置、姿态、速度及稳定采样门槛保留；`target_approach` 与 `support_release` 仍为原 1° 关节门槛，闭爪位置容差仍为命令中的 4 mm。此修复调整搬运的到位判定，没有提高控制器本身的跟踪精度。见 [A 停止反馈回放](artifacts/banana_planning_diagnosis/contact_20261029/A_1139_arrival_replay.json)。原轮次保留 `execution_failed`，不改写为已到指定持物位。
- **B 停在侧视角计算阶段，尚未生成抓取候选。** `2026-10-29_11-42-33_124995_p580947` 完成中心观察后，侧视角 NBV 搜索触发旧 30 秒数值进程期限。用同一份保存点云、相机位姿和关节状态回放，搜索约需 33 秒，主要耗在多姿态／多种子 IK。数值请求默认期限现为 60 秒；控制健康检查继续运行，超时仍终止并保持，不内联计算或自动执行。修复后在禁止 CAN 的独立规划进程中约 33.5 秒返回 7 个端点有效的侧视角记录；这不等于后续完整路径已验证或实机多视角成功。见 [B 独立进程回放](artifacts/banana_planning_diagnosis/contact_20261029/B_1142_nbv_worker60_replay.json)。

相关离线测试 **85 项通过，另 9 个子测试通过**。重新复制下面的 A、B 命令验证即可，无须添加参数；新一轮重新采集和规划，不从旧停止点续执行。两次旧日志保留原状态，新旧代码版本的结果分别注明，先确认这两组流程再继续其他组。

### 9.1.5 11:54 B 的中心桌面修正（当前生效）

本轮 `2026-10-29_11-54-59_353175_p593902` 已运行 ICP 并融合中心＋侧视角，但在 GraspNet 运行前，中心原始桌面的 5.12° 倾角超过绝对桌面修正函数的旧 5° 上限，修正尚未应用就退出。中心是相对融合的参考帧，其他视角对齐中心不会消除中心自身相对真实桌面的偏差。

现在**绝对桌面修正允许角度为 6°**，原有桌面点数、占比、覆盖范围与 30 mm 高度修正范围继续检查，通过后实际应用刚体变换。中心修正用于已有融合目标的世界坐标映射，所有原始观察的场景与障碍各自注册至确认的桌面 Z=0；已通过的目标融合和各原始文件保留。该范围属于观测坐标校正，不是允许夹指穿桌；规划／运行时桌面间隙仍为 5／3 mm。没有加入新的命令行参数。

本轮中心桌面拟合支持率约 94.5%、平面 RMS 约 0.88 mm。用原始三视角数据回放，中心桌面倾角从 **5.121° 降至 0.015°**，香蕉附近的桌面高度从 **+23.10 mm 降至约 −0.013 mm**；中心实测目标点的高度中位数从约 55.1 mm 修正为 31.3 mm。初始／侧视角校正后桌面高度误差约 0.42／0.035 mm，三视角障碍环境已重新构建，融合目标仍为 2560 点。见 [本轮保存数据回放](artifacts/banana_planning_diagnosis/contact_20261029/B_1154_table_reference/replay.json)。这是点云坐标中的一致性验证，尚未确认原始偏差究竟来自手眼、机械臂位姿还是深度测量。

**相关测试 41 项通过。继续使用下方 B 原命令复测即可。** 新日志的 `tilt_before=... correction=applied` 表示打印的是修正前倾角，场景变换已经应用；没有这条成功日志则未完成对应视角的校正。本次未连接或驱动机械臂，也未运行新的 GraspNet 推理，因此不预先断言后续抓取候选或完整路径一定通过。

**14:38 B 实机复测已越过桌面修正，停在抓姿筛选：** `2026-10-29_14-38-21_488210_p613459` 的初始、中心、侧视角均出现 `correction=applied`，中心修正前倾角 5.16°、高度约 22.5 mm；中心＋侧视角成功融合为 2486 点。GraspNet 生成 12 个候选，其中 10 个未通过双侧接触检查，剩余 1／8 号的完整 CAD 对桌面间隙分别约 −38.0／−7.8 mm，均未进入抓取 IK；该轮为 `planning_failed`，未执行抓取接近或闭爪，观察运动已执行，没有电子急停。

复用这轮相同网络输出、实测目标与桌面变换，对 0–30 mm 推进量按 1 mm 步长探测：9 号在 16–19 mm 时可同时通过双侧接触和 CAD 桌面检查，见 [同帧推进量探测](artifacts/banana_planning_diagnosis/contact_20261029/B_1438_advance_probe.json)。但进一步用 16 mm 检查完整流程时，虽然预抓取和抓取 IK 有解，最终接近中夹指对环境点的模型间隙仅约 0.7 mm，未通过 5 mm 路径检查，见 [16 mm 完整路径回放](artifacts/banana_planning_diagnosis/contact_20261029/B_1438_route16.json)。因此不能把这个几何候选当作可执行抓姿，也不据此把 B 命令改成固定 16 mm。该回放未连接机械臂；保留本轮失败结果，不将排障调参结果混入原参数统计。

### 9.1.6 为什么简单场景仍被拒绝：实体夹爪、姿态和具体障碍的定位

**最新 15:00 轮次并未运行 GraspNet。** `2026-10-29_15-00-45_039476_p622481` 的第一侧视角因遮罩外围暗像素占比 19.5% 超过 15% 被跳过；内部核心与稳定轮廓均为 100%，单凭这个外围判据不能断言夹爪安装改变。随后换第二侧视角，采集和融合均通过。最终停止在第二侧视角的绝对桌面校正：原始高度偏差约 34.2 mm，超过旧流程逐视角使用的 30 mm 范围。该轮没有生成抓姿，不属于网络抓取失败。

**已修正桌面校正的衔接。** 初始／中心仍按确认的桌面建立绝对参考；侧视角先应用同一个中心绝对变换，再拟合和校正它相对该参考的剩余误差。侧视角沿用现有 `--active-table-reg-max-height-shift` 与 `--active-table-reg-max-tilt-deg` 的相对校正范围，未放宽夹爪对桌面的碰撞间隙。报告的 `T_registration` 保存完整复合变换，`height_at_target_m` 仍记录原始总偏差，另保存 `residual_height_before_m`、`residual_tilt_before_deg`。每次原始观察的障碍继续保留，包括未用于目标融合的观察。

本轮原数据回放中，中心原始偏高约 22.1 mm；侧视角原始偏高约 34.3 mm，在中心参考下仅需再校正约 10.1 mm、1.93°。校正后重新拟合侧视角桌面：高度约 −0.004 mm、倾角约 0.039°；中心和侧视角的目标点高度中位数分别约 31.9／30.2 mm。见 [15:00 桌面校正回放](artifacts/banana_planning_diagnosis/contact_20261029/B_1500_table_reference/replay.json)。这些数值只验证保存点云的坐标一致性，不代表实机绝对测量达到该精度。

**前轮 14:38 的“穿桌”来源已分解到实体夹指。** 网络输出用简化平行夹爪表示，候选建议开口约 55–76 mm；主流程当时以 Nero 100 mm 张开姿态接近，再统一推进 15 mm，并按点云沿开合轴居中。这几步会改变真实夹指到桌面的距离：

- 1 号候选开合轴倾斜很大，100 mm 开口对应两侧内表面约 65 mm 的竖直高差，而香蕉点云主要高约 30 mm。15 mm 沿接近轴推进使 TCP 再下降约 11.4 mm，沿倾斜开合轴的居中修正又下降约 9.5 mm；最终低侧真实指尖的 CAD 点为 −38.0 mm。即使仅把开口缩到该候选约 59 mm，保持同一位姿仍为约 −24.7 mm，不能靠缩开口单独解决。
- 8 号候选两侧内表面的高差约 29 mm，15 mm 推进同样消耗约 11.4 mm 的竖直余量，低侧完整 CAD 最低点为 −7.8 mm。
- “双侧接触检查通过”只表示选定薄片中有满足深入量的目标点，不保证实体夹指其他部位不碰桌。当前接触薄片总厚度约 14 mm，真实 CAD 夹指最宽处约 56 mm、前端逐渐变窄；两者是不同精度的几何表示。接触筛选也没有直接测量实际接触，因此未通过该检查不等于实物一定空抓。

见 [实体夹指与桌面的投影图](artifacts/banana_planning_diagnosis/contact_20261029/B_geometry_audit/cad_table_comparison.png)、[逐候选几何分解](artifacts/banana_planning_diagnosis/contact_20261029/B_geometry_audit/candidate_geometry.json)。图中黄色为实测目标点，蓝色为真实 CAD 夹指投影，黑线为桌面，红虚线为 5 mm 规划间隙；这是未执行候选的模型预测。

**9 号 16 mm 回放的 0.7 mm 障碍不是桌面。** 最近障碍点为 `[-0.54478, -0.00256, 0.01721] m`，来自初始图像像素约 `(1246, 507)`，颜色及原图位置均对应红色方块边缘；距融合香蕉点云约 38 mm。详见 [障碍来源定位](artifacts/banana_planning_diagnosis/contact_20261029/B_1438_witness_source.json)、[原始图像](logs/banana_mgpc/ablation_v2/B_multi/2026-10-29_14-38-21_488210_p613459/2026-10-29_14-38-36_693082/multiview_raw/initial_2026-10-29_14-38-43_953866_793c5670/color.png)。该姿态打开的夹指贴近这个方块，不能把它归为桌面噪声删除。

**当前应改的是候选对真实 Nero 夹爪的适配。** 固定 15 mm 不能同时适合所有姿态；需要联合真实可用接触区、实体 CAD、开口和推进量生成或修正候选，并保留完整路径检查。深度搜索本身不能解决这轮所有问题：已有探测只证明 9 号存在接触＋桌面几何可行深度，完整路径仍被方块阻挡。更换 IK 或 RRT 也不能直接修正一个终点就穿桌的抓姿。

另已核对 SDK 法兰反馈与 URDF 前向计算：保存的各视角软件计算差约 0.02 mm／0.003°，未发现两套软件 FK 换算互相冲突；这不验证真实机械臂绝对精度。原始桌面的 10–34 mm／约 4–5° 偏差仍指向感知到机器人坐标映射链，需要继续区分手眼、RGB-D 与实机几何误差。对固定外参偏差的离线假设未能解释所有留出视角，未据此改写手眼文件，见 [只读外参假设审计](artifacts/banana_planning_diagnosis/contact_20261029/B_geometry_audit/plane_extrinsic_hypothesis.json)。

桌面参考衔接相关测试 **42 项通过**，包括“侧视角总偏差 34 mm、相对中心偏差约 12 mm 应通过”和“相对中心仍超配置范围应拒绝”。本次只连接了数值计算进程，未连接或驱动机械臂。B 命令无需增加参数；修复只说明保存数据的桌面参考步骤可以继续，不保证重新生成的抓姿一定通过实体几何和完整路径检查。

### 9.1.7 Nero 100 mm 候选修正：修改前后离线完整路径对照

本节记录接入前的离线原型对照，**主流程接入及 GraspGenX 回放结果见 9.1.8**。对每帧复用完全相同的已保存 GraspNet 输出、实测目标、障碍占用、桌面变换、开始关节和保护参数，不重新采图或推理。两种方案实际开口都为 100 mm；模型宽度字段保留原值，不将所有网络宽度直接改为 100 mm。闭爪执行逻辑未修改。

- 修改前：原始姿态＋保存命令中的固定 15 mm 推进，逐个候选检查，不因找到少数可行候选就提前结束统计。
- 修改后：保留原来已通过的候选；对原来失败的候选，有限搜索保持原方向、把开合轴放平、将接近方向调整到距竖直 30°／15°／0°，以及夹爪绕接近轴翻转 180° 的方向。每种方向在 0–30 mm 范围按 1 mm 步长寻找通过现有双侧接触＋真实 Nero CAD 桌面检查的最小推进，随后每个原始候选最多尝试 8 个几何可行方案的完整路径。所有原有接触与碰撞门槛保留，红色方块也保留。
- 完整路径通过：预抓取、接近、短抬升、空抓回退均通过当前工作流预检，并且从短抬升终点到配置中的指定关节持物位也规划通过。本节未测试搬运放置／交接或实际夹持稳定性。

| 保存帧 | 原始候选数 | 修改前通过完整路径 | 修改后找到可行修正版的原始候选数 | 新增路径尝试 | 全帧离线耗时 |
| --- | ---: | ---: | ---: | ---: | ---: |
| B 多视角，10 月 29 日 14:38，原轮次规划失败 | 12 | 0 | **4** | 42 | 79.2 s |
| A 单视角，10 月 29 日 11:39，原轮次已夹持并抬升 | 18 | 1 | **11** | 63 | 120.2 s |

B 通过的是原始 2／3／4／6 号的修正版；A 原来通过的 2 号保留，另外 10 个原始候选找到可行修正版。按原始候选分别计数，不能把同一候选的一串深度样本当作多个成功抓姿。举例：B 的 3 号将开合轴放平、推进 17 mm 后，CAD 对桌面间隙约 9.4 mm，预抓取、接近、短抬升、空抓回退和 3 段指定关节持物路径全部通过；2 号调整至约 30° 接近、推进仍为 15 mm，CAD 间隙约 8.5 mm，也通过完整路径。

这说明**真实夹爪约束下的姿态／深度搜索值得接入验证**，不说明仅把开口数字改为 100 mm 就能改善。新方案增加了计算预算；比较结果是两帧保存数据下的有限搜索结果，不是网络性能提升或实机成功率。修改后保留基线成功是算法设计，不应单独作为无回退的泛化证据。搜索失败也不证明其他预算或姿态下不存在解。

结果：[完整 JSON 与每次失败原因](artifacts/nero_candidate_refinement/comparison.json)、[汇总 CSV](artifacts/nero_candidate_refinement/comparison.csv)。各候选日志位于同目录的源轮次子目录。离线工具为 `tools/compare_nero_candidate_refinement.py`，不创建硬件实例或调用执行入口。

复现到独立目录（无需 GPU、相机、CAN 或安装新环境）：

```bash
cd /home/yx/yx/panda_ws/src/whole-grasp
conda activate grasp_yolo26
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python tools/compare_nero_candidate_refinement.py \
  --run-dir \
    logs/banana_mgpc/ablation_v2/B_multi/2026-10-29_14-38-21_488210_p613459 \
    logs/banana_mgpc/ablation_v2/A_single/2026-10-29_11-39-46_357873_p567407 \
  --output-dir artifacts/nero_candidate_refinement_repeat
```

### 9.1.8 已接入主流程：GraspNet 和 GraspGenX 共用 Nero 候选修正

**A–H 均已加 `--nero-grasp-refine`，直接复制 9.2–9.9 对应组命令。** 原始候选只要有一条路径连同指定关节持物位通过，就复用原候选；全部不通过时才搜索修正。修正使用实测目标和实测障碍，MGPC 补全只负责给网络提出抓姿，不能作为接触／碰撞通过的证据。100 mm 开口和慢闭爪反馈逻辑保持原样，规划 5 mm、运行 3 mm 门限保持原样。

- GraspNet：沿用离线验证的姿态族，按 1 mm 步长在总推进 0–30 mm 内搜索。
- GraspGenX：在**原生 Nero TCP**中修改姿态，推进范围保留 0–15 mm；不会混用 GraspNet 的额外 15 mm。每个修正版保存自己的推进量，后续检查、预览和执行使用同一个 TCP。
- 每种方向先找通过双侧接触和完整 CAD 对桌面检查的最浅推进；每个原始候选最多尝试 8 种几何可行方案的路径，默认额外搜索预算 60 s、找到 3 个完整可行候选就停止。预算在单次路径尝试之间检查，正在进行的检查可以超过预算少量时间。找不到时如实报规划失败。
- 预抓取、接近、短抬升、空抓回退和指定关节持物路径都通过后才能进入可执行列表。持物路径缓存复用；不会在选中后重复搜索。耗时的几何搜索放在禁止 CAN 的数值进程中。
- 最终确认窗口现在两种生成器都画**青色真实 Nero CAD**，位置对应最终执行 TCP。绿色线框仍表示网络候选参考；以青色 CAD 和黄色终点为最终检查对象。

保存数据的主流程回放结果（不重新推理、不连接相机或机械臂、不发送运动命令）：

| 保存轮次 | 原候选完整路径数 | 新增可行修正版 | 主流程结果 |
| --- | ---: | ---: | --- |
| B／GraspNet，14:38 | 0 | 3，原始 2／3／4 号 | 选中 3 号：开合轴放平、推进 17 mm，模型桌面间隙约 9.4 mm；完整规划通过 |
| A／GraspNet，11:39 | 1 | 0，跳过修正 | 保留原始 2 号，完整规划通过 |
| E／GraspGenX，11:03 已成功帧 | 1 | 0，跳过修正 | 保留原始 0 号，完整规划通过 |
| E／GraspGenX，10:31 失败帧，原命令固定推进 9 mm | 0 | 2，原始 1／2 号 | 选中 1 号：方向不变、推进 11 mm，完整规划通过 |

B 的在线回放找到 3 个即停，因此不同于 9.1.7 的全量离线计数 4 个；A 原来已有可行候选，因此跳过修正，不再穷举到 11 个。**以上是旧数据的规划结果，不是真机抓取成功率；C／D／F／G／H 尚需实测。**

E 的 1 号推进至 11 mm，已有自动推进也能找到，不能把它全部算作新增姿态搜索的收益；本次额外找到的 2 号使用竖直接近＋绕接近轴翻转，推进 1 mm，也通过完整路径。当前修正只处理生成器前置筛选后保留的候选，不能挽救所有已被网络／原生桌面预筛丢弃的姿态。

验证：115 项相关测试通过，覆盖两种生成器的实际推进量、原生 TCP、完整 CAD 预览、原可行路径复用、持物路径拒绝及数值进程隔离；A–H 命令均通过 shell 语法与 CLI 参数解析检查。本轮未连接或控制硬件。

各帧的 `nero_refinement.json` 保存新增搜索次数、选用的方向／推进量及失败原因；`selected_grasp_pose.json` 和 `whole_workflow_motion_plan.json` 保存实际选中 TCP 与含持物位的路径。回放文件在 [主流程回放目录](artifacts/nero_candidate_refinement/mainflow/)。

如需离线复查 GraspGenX（使用现有环境，无需重装）：

```bash
cd /home/yx/yx/panda_ws/src/whole-grasp
conda activate grasp_yolo26
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python tools/replay_nero_refinement.py \
  --run-dir \
    logs/banana_mgpc/E_single_genx/2026-10-29_11-03-38_845411_p545323 \
    logs/banana_mgpc/E_single_genx/2026-10-29_10-31-48_477903_p528171 \
  --output-dir artifacts/nero_candidate_refinement/genx_replay_repeat
```

### 9.2 A：中心单视角，GraspNet，不加 MGPC

```bash
MOTION_SUPPORT_RELEASE_HEIGHT=0.025 bash scripts/run_grasp_safe.sh \
  --config config/pipeline.yaml --phase grasp --execute \
  --target-class banana \
  --yolo-weights models/food5_visible_v3_full_reviewed_corrected_best.pt \
  --active-view --no-active-multiview-gate \
  --active-center-height 0.35 --active-center-roll-plan-candidates 1 \
  --no-active-center-optical-relaxation --active-capture-settle-seconds 1.0 \
  --preview-cloud \
  --trajectory-executor mit --mit-q-log \
  --mit-max-joint-vel-deg-s 12 --mit-max-joint-accel-deg-s2 8 \
  --active-side-view-count 0 --grasp-required-view-count 1 \
  --grasp-table-reference-z 0 \
  --grasp-generator graspnet --nero-grasp-refine \
  --grasp-advance 0.015 \
  --grasp-close-pos-tol 0.004 \
  --no-mgpc \
  --save-dir logs/banana_mgpc/ablation_v3_nero_refine/A_single
```

### 9.3 B：多视角融合，GraspNet，不加 MGPC

```bash
MOTION_SUPPORT_RELEASE_HEIGHT=0.025 bash scripts/run_grasp_safe.sh \
  --config config/pipeline.yaml --phase grasp --execute \
  --target-class banana \
  --yolo-weights models/food5_visible_v3_full_reviewed_corrected_best.pt \
  --active-view --no-active-multiview-gate \
  --active-center-height 0.35 --active-center-roll-plan-candidates 1 \
  --no-active-center-optical-relaxation --active-capture-settle-seconds 1.0 \
  --active-fusion-rigid-icp \
  --active-table-reg-max-height-shift 0.030 \
  --preview-cloud \
  --trajectory-executor mit --mit-q-log \
  --mit-max-joint-vel-deg-s 12 --mit-max-joint-accel-deg-s2 8 \
  --active-side-view-count 1 --grasp-required-view-count 2 \
  --grasp-table-reference-z 0 \
  --grasp-generator graspnet --nero-grasp-refine \
  --grasp-advance 0.015 \
  --grasp-close-pos-tol 0.004 \
  --no-mgpc \
  --save-dir logs/banana_mgpc/ablation_v3_nero_refine/B_multi
```

### 9.4 C：中心单视角，MGPC 后用 GraspNet

下方命令已统一为本节 A–H 的新参数。后文的 21:04 结果是历史排障记录；正式对比可重测并记录规划失败，勿把旧轮次当作新参数的结果。E 的成功未证明 C 的候选问题已经解决。

**20:54 这一轮的失败原因与修复：** MGPC 成功输出 8192 点，GraspNet 生成 13 个候选，其中 8 个通过实测开口检查，但完整打开夹爪网格的桌面间隙全部不足 5 mm（最好约 1.6 mm），因此 `ik_calls=0`，没有执行接近、闭爪或指定关节位动作。本轮是 `planning_failed`，没有电子急停。

同时查到桌面修正后的点云与 MGPC 使用的中心相机位姿混用了两个世界基准。现在中心参考相机也应用同一桌面注册变换，再将点云送入其对应的 RGB 相机坐标；该修复同样适用于 D／G／H 的 MGPC 输入。保存数据的转换误差中位数由约 20.96 mm 降至浮点数精度，29 项相关测试通过。这个数值是网络输入坐标转换的差异，不是机械臂抓取误差；只修复坐标一致性，不保证补全网络的新输出一定产生可执行候选。

20:54 后曾建议重跑 C；21:04 再次失败后，当时暂停了重复试机。诊断记录见 [MGPC 本轮诊断](artifacts/banana_planning_diagnosis/contact_20261028/mgpc_2054_diagnosis.json)。本节当前命令用于新一版对比，仍保留 5 mm 网格对桌面阈值、实测安全点云和指定持物关节位；若没有可执行候选，如实记录规划失败。

**21:04 复测记录：暂停反复运行 C 真机。** 已核对本轮 `mgpc/request.json`，中心参考相机等于该轮桌面注册变换乘原始中心位姿，坐标修复确实生效；观测点到补全点的 95% 最近距离约 1.90 mm。新的 10 个抓姿中 5 个通过开口检查，但完整网格对桌面间隙分别约 −5.0、−15.9、0.7、−41.5、−42.5 mm，全部低于 5 mm，因此没有进入 IK、接近、闭爪或高位持物动作。本轮记录为规划失败。同帧 GPU 推理对照现已完成，结果见 9.0；不能用其他场景的 A 成功代替本帧结论。见 [21:04 诊断](artifacts/banana_planning_diagnosis/contact_20261028/mgpc_2104_diagnosis.json)。

```bash
MOTION_SUPPORT_RELEASE_HEIGHT=0.025 bash scripts/run_grasp_safe.sh \
  --config config/pipeline.yaml --phase grasp --execute \
  --target-class banana \
  --yolo-weights models/food5_visible_v3_full_reviewed_corrected_best.pt \
  --active-view --no-active-multiview-gate \
  --active-center-height 0.35 --active-center-roll-plan-candidates 1 \
  --no-active-center-optical-relaxation --active-capture-settle-seconds 1.0 \
  --preview-cloud \
  --trajectory-executor mit --mit-q-log \
  --mit-max-joint-vel-deg-s 12 --mit-max-joint-accel-deg-s2 8 \
  --active-side-view-count 0 --grasp-required-view-count 1 \
  --grasp-table-reference-z 0 \
  --grasp-generator graspnet --nero-grasp-refine \
  --grasp-advance 0.015 \
  --grasp-close-pos-tol 0.004 \
  --mgpc --mgpc-category banana \
  --save-dir logs/banana_mgpc/ablation_v3_nero_refine/C_single_mgpc
```

### 9.5 D：多视角融合，MGPC 后用 GraspNet

下方命令已统一为本节 A–H 的新参数。后文的 21:12 结果是历史排障记录；正式对比可重测并记录规划失败，勿把旧轮次当作新参数的结果。E 的成功未证明 D 的候选问题已经解决。

**21:12 这一轮 D 组也为规划失败。** 中心与侧视角均采集并通过当前融合门限，实际保留 `['center', 'side1']`，融合目标 2417 点；MGPC 补全为 8192 点，GraspNet 得到 14 个候选，9 个通过开口检查，但 9 个均未通过 5 mm 完整网格桌面检查，最好为 4.3 mm。没有进入抓取 IK、接近、闭爪或指定持物位执行，没有电子急停。输入参考相机的桌面修正已核对生效，见 [D 组诊断](artifacts/banana_planning_diagnosis/contact_20261028/mgpc_D_2112_diagnosis.json)。本轮不能归因于“没采到侧视角”，也不能与 C 的另一帧、另一随机种子直接比较算法优劣；先保留 C／D 规划失败记录，暂停盲目重复真机，后续同帧离线对照后再决定输入处理或候选修正策略。

```bash
MOTION_SUPPORT_RELEASE_HEIGHT=0.025 bash scripts/run_grasp_safe.sh \
  --config config/pipeline.yaml --phase grasp --execute \
  --target-class banana \
  --yolo-weights models/food5_visible_v3_full_reviewed_corrected_best.pt \
  --active-view --no-active-multiview-gate \
  --active-center-height 0.35 --active-center-roll-plan-candidates 1 \
  --no-active-center-optical-relaxation --active-capture-settle-seconds 1.0 \
  --active-fusion-rigid-icp \
  --active-table-reg-max-height-shift 0.030 \
  --preview-cloud \
  --trajectory-executor mit --mit-q-log \
  --mit-max-joint-vel-deg-s 12 --mit-max-joint-accel-deg-s2 8 \
  --active-side-view-count 1 --grasp-required-view-count 2 \
  --grasp-table-reference-z 0 \
  --grasp-generator graspnet --nero-grasp-refine \
  --grasp-advance 0.015 \
  --grasp-close-pos-tol 0.004 \
  --mgpc --mgpc-category banana \
  --save-dir logs/banana_mgpc/ablation_v3_nero_refine/D_multi_mgpc
```

### 9.6 E：中心单视角，GraspGenX，不加 MGPC

下方执行参数与 11:03 已成功的 E 轮次一致，日志另存至新版对比目录；历史失败分析保留在后文。

**21:16 这轮 E 不是被桌面检查拒绝：** GenX 输出 40 个姿态，其中 1 个通过原始网格桌面预筛，模型间隙约 16.1 mm；其后实测点云开口检查估计左右接触深入量只有 2.9／2.5 mm，未达到当前每侧 10 mm 的筛选门槛，因此没有接近或闭爪。居中修正后的同一姿态，离线完整网格桌面间隙约 18.0 mm，桌面并非该候选的直接拒绝原因。

这里的“网格桌面检查”是**夹爪 CAD 三角网格与桌面 Z=0 平面的几何检查**：把模型放到计划姿态，计算最低模型点到桌面的竖直距离，要求至少 5 mm；这是模型预测，不是实时实测距离。A／B 现场看到最低点离桌约 10 mm，与“至少 5 mm”不矛盾。**接触深入量**则沿夹指长度方向衡量目标点进入指尖后方多远，与竖直桌面间隙是两个量；10 mm 深入门槛属于当前工程筛选规则，尚不等同于实际接触或论文验证的最优阈值。

GenX 当前按原生 Nero TCP 姿态映射，`grasp_to_tcp_poses` 不叠加 GraspNet 的 `--grasp-advance`，因此不能靠给 E 添加该参数直接修正。对这次唯一候选的**离线几何探测**显示：沿接近轴前移 11 mm 后，左右深入量约 13.9／10.9 mm，完整网格对桌面间隙约 7.4 mm，两项几何筛选均通过，见 [E 接触深度探测](artifacts/banana_planning_diagnosis/contact_20261028/genx_E_2116_contact_probe.json)。**默认 5 mm 的流程预检在接近段未通过：夹指与实测环境点的模型间隙约 3.1 mm**，见 [E 路径预检](artifacts/banana_planning_diagnosis/contact_20261028/genx_E_2116_route_probe.json)。以上是修正前的历史诊断。现在正常接近也已统一 CAD／桌面平面检查；本节 E 命令现按候选自动计算原生接近推进量，使用正常保护，完整说明和离线证据见 9.0.3。下面命令会闭爪、短距离脱离桌面，再去指定持物关节位。

```bash
MOTION_SUPPORT_RELEASE_HEIGHT=0.025 bash scripts/run_grasp_safe.sh \
  --config config/pipeline.yaml --phase grasp --execute \
  --target-class banana \
  --yolo-weights models/food5_visible_v3_full_reviewed_corrected_best.pt \
  --active-view --no-active-multiview-gate \
  --active-center-height 0.35 --active-center-roll-plan-candidates 1 \
  --no-active-center-optical-relaxation --active-capture-settle-seconds 1.0 \
  --preview-cloud \
  --trajectory-executor mit --mit-q-log \
  --mit-max-joint-vel-deg-s 12 --mit-max-joint-accel-deg-s2 8 \
  --active-side-view-count 0 --grasp-required-view-count 1 \
  --grasp-table-reference-z 0 \
  --grasp-generator graspgenx --nero-grasp-refine --graspgenx-contact-advance auto \
  --grasp-close-pos-tol 0.004 \
  --no-mgpc \
  --save-dir logs/banana_mgpc/ablation_v3_nero_refine/E_single_genx
```

### 9.7 F：多视角融合，GraspGenX，不加 MGPC

```bash
MOTION_SUPPORT_RELEASE_HEIGHT=0.025 bash scripts/run_grasp_safe.sh \
  --config config/pipeline.yaml --phase grasp --execute \
  --target-class banana \
  --yolo-weights models/food5_visible_v3_full_reviewed_corrected_best.pt \
  --active-view --no-active-multiview-gate \
  --active-center-height 0.35 --active-center-roll-plan-candidates 1 \
  --no-active-center-optical-relaxation --active-capture-settle-seconds 1.0 \
  --active-fusion-rigid-icp \
  --active-table-reg-max-height-shift 0.030 \
  --preview-cloud \
  --trajectory-executor mit --mit-q-log \
  --mit-max-joint-vel-deg-s 12 --mit-max-joint-accel-deg-s2 8 \
  --active-side-view-count 1 --grasp-required-view-count 2 \
  --grasp-table-reference-z 0 \
  --grasp-generator graspgenx --nero-grasp-refine --graspgenx-contact-advance auto \
  --grasp-close-pos-tol 0.004 \
  --no-mgpc \
  --save-dir logs/banana_mgpc/ablation_v3_nero_refine/F_multi_genx
```

### 9.8 G：中心单视角，MGPC 后用 GraspGenX

```bash
MOTION_SUPPORT_RELEASE_HEIGHT=0.025 bash scripts/run_grasp_safe.sh \
  --config config/pipeline.yaml --phase grasp --execute \
  --target-class banana \
  --yolo-weights models/food5_visible_v3_full_reviewed_corrected_best.pt \
  --active-view --no-active-multiview-gate \
  --active-center-height 0.35 --active-center-roll-plan-candidates 1 \
  --no-active-center-optical-relaxation --active-capture-settle-seconds 1.0 \
  --preview-cloud \
  --trajectory-executor mit --mit-q-log \
  --mit-max-joint-vel-deg-s 12 --mit-max-joint-accel-deg-s2 8 \
  --active-side-view-count 0 --grasp-required-view-count 1 \
  --grasp-table-reference-z 0 \
  --grasp-generator graspgenx --nero-grasp-refine --graspgenx-contact-advance auto \
  --grasp-close-pos-tol 0.004 \
  --mgpc --mgpc-category banana \
  --save-dir logs/banana_mgpc/ablation_v3_nero_refine/G_single_mgpc_genx
```

### 9.9 H：多视角融合，MGPC 后用 GraspGenX

```bash
MOTION_SUPPORT_RELEASE_HEIGHT=0.025 bash scripts/run_grasp_safe.sh \
  --config config/pipeline.yaml --phase grasp --execute \
  --target-class banana \
  --yolo-weights models/food5_visible_v3_full_reviewed_corrected_best.pt \
  --active-view --no-active-multiview-gate \
  --active-center-height 0.35 --active-center-roll-plan-candidates 1 \
  --no-active-center-optical-relaxation --active-capture-settle-seconds 1.0 \
  --active-fusion-rigid-icp \
  --active-table-reg-max-height-shift 0.030 \
  --preview-cloud \
  --trajectory-executor mit --mit-q-log \
  --mit-max-joint-vel-deg-s 12 --mit-max-joint-accel-deg-s2 8 \
  --active-side-view-count 1 --grasp-required-view-count 2 \
  --grasp-table-reference-z 0 \
  --grasp-generator graspgenx --nero-grasp-refine --graspgenx-contact-advance auto \
  --grasp-close-pos-tol 0.004 \
  --mgpc --mgpc-category banana \
  --save-dir logs/banana_mgpc/ablation_v3_nero_refine/H_multi_mgpc_genx
```

### 9.10 执行及记录

- 启动后先按原提示确认观察运动；自动分割到 `banana` 时不需要点选。漏检时记录失败并停止该组。
- `--preview-cloud` 的窗口关闭后继续。保留原有抓取运动确认；没有加入 `--yes` 或自动跳过确认。
- MGPC 由 `.venv_mgpc/bin/python` 独立进程加载 `third_party/MGPC/pretrained/ckpt_8192.pt`，结束后释放显存，再运行所选生成器。
  日志应出现 `[mgpc] 8192 predicted points -> graspnet/graspgenx proposals`。加载、推理或几何检查失败会停止该次抓取，不会偷偷换回无 MGPC。
- GraspGenX 由 `third_party/GraspGenX/.venv/bin/python` 独立进程加载模型；每轮保存 `graspgenx/request.json`、`proposal_camera.npy`、
  `worker.log`、`candidates.json`。日志中的 `table-safe` 只是第一道几何过滤；最终必须看到原主流程的完整工作流预检通过。
  GenX 抓取使用模型已定义的实际夹爪位姿，主流程不对其应用 `--grasp-advance`；E–H 无需添加该参数。
  主流程部分旧日志名仍含 `graspnet`，用 `grasp_candidates.json` 里的 `generator` 字段区分来源。
- MGPC 只改候选生成输入；目标闭合区占据、实际开口、桌面、完整路径和实机运行时碰撞仍用实测数据。
  夹爪尺寸、间隙、接触确认和空抓保护沿用安全脚本，不因 MGPC 放宽。
- 本轮使用原有抓取验证模式：夹取后做支撑脱离／验证抬升，不自动进入搬运放置或人手交接。
  成功后可能仍夹着香蕉；按原流程安全退出、取下并恢复同一位置后再测下一组。
- 每组保留完整日志。MGPC 组额外保存捕获目录下 `mgpc/target_rgb.png`、`request.json`、
  `prediction_camera.ply`、`prediction_base.npy`、`worker.log` 和 `completion.json`；不要复用上一轮静态预测。
- 每组运行目录的 `experiment_result.json` 记录 `method.grasp_generator`、`method.mgpc`、使用的随机种子，
  `selected_grasp.generator` 记录真正选中的候选来源；候选详细位姿在 `grasp_candidates.json`。
- 记录是否识别成功、是否找到通过预检的候选、是否实际夹起、是否滑落，以及失败阶段。
  固定障碍布局，比较时每组重复若干次，不能用网络分数或一次成功代替成功率。

MGPC 曾在保存点云上完成 GPU 推理，输出 8192 点。GenX 工作进程按本节默认 `64/20` 候选配置，在旧瓶子中心点云上生成
40 个候选，25 个通过桌面网格检查；这是旧瓶子点云的离线通路验证，不代表香蕉抓取效果。首次实机仍须按本节预览核对 TCP 与夹爪几何。
这八组的真机抓取成功率尚未测得，不能把离线可视化或模型分数当作真机结果。

## 10. 网络权重离线重排：10% / 40% / 70%

结果位于 `artifacts/grasp_weight_rerank/comparison.html`，三窗并排显示同一候选池的排名和保存 TCP 姿态。
默认优先列出首选发生变化的数据；可以切换每一名候选，查看八项得分贡献、实测 RGB 点云和 MGPC 预测参考。
重排固定候选、归一化和原过滤结果，其余评分权重按比例缩减。没有连接机械臂或修改主流程默认权重。

```bash
cd /home/yx/yx/panda_ws/src/whole-grasp
python3 -m http.server 8082 --bind 127.0.0.1 --directory artifacts/grasp_weight_rerank
```

浏览器打开 `http://127.0.0.1:8082/comparison.html`。结果说明见同目录 `SUMMARY.md`，完整排名为 `rankings.csv`。
重新生成离线结果：

```bash
cd /home/yx/yx/panda_ws/src/whole-grasp
PYTHONNOUSERSITE=1 MPLCONFIGDIR=/tmp/matplotlib .venv_mgpc/bin/python tools/rerank_grasp_weights.py
```

数据包含已有四组 GraspNet 离线审计和 8 次保存完整评分的历史采集；香蕉新日志尚无候选，GraspGenX 尚无完整重排输入。
只保留保存的完整流程检查通过候选，不将碰撞、检查失败或未检查候选重新加入排名。
历史检查不等于当前实机安全认证，离线首选改变也不代表成功率提高；历史候选 ID 不能直接用于新采集执行。

## 11. GraspNet 输入点数离线对比：3000 点 / 全量目标点

使用已保存的旧短瓶四组点云 A/B/C/D；香蕉新日志尚没有可用于这四组对比的完整目标数据。
两方案保持模型、目标清理裁剪、实测安全环境、过滤和默认候选评分相同，只改变网络输入点的整理方式。
`fixed` 使用原来的固定 3000 点；`all` 保留清理裁剪后的全部点，既不抽点也不补点，保持点顺序。
全量方案的三个 seed 输入相同，不作为三个独立样本；离线数值路径预检通过不等于实机抓取成功。

```bash
cd /home/yx/yx/panda_ws/src/whole-grasp
python3 -m http.server 8083 --bind 127.0.0.1 --directory artifacts/graspnet_input_points_compare
```

浏览器打开 `http://127.0.0.1:8083/comparison.html`，页面并排统计两方案，再分别提供四组候选浏览器。
可查看实测 RGB 点云、完整过滤原因、TCP 位姿和碰撞位置。数值表见 `SUMMARY.md`；首选姿态对照为 `top_candidates.png`。
主流程的 `--grasp-num-points` 默认值保持 3000。

重跑完整离线对比（需要 CUDA，不连接机械臂或相机）：

```bash
cd /home/yx/yx/panda_ws/src/whole-grasp
PYTHONNOUSERSITE=1 MPLCONFIGDIR=/tmp/matplotlib .venv_mgpc/bin/python tools/compare_graspnet_input_points.py
```

## 看状态、找日志与退出

| 提示 | 含义 |
| --- | --- |
| `TRACKING` | 正在接受跟随目标 |
| `MEMORY_APPROACH` | 有限使用最后可信位置，不是读到了新深度 |
| `GESTURE_LEARN` | A已稳定，B学习端点，夹爪不动 |
| `GESTURE_ARM` | 等待B捏合确认，夹爪不动 |
| `GESTURE_READY` | 已确认但尚未松夹，A移动仍优先恢复跟随 |
| `GESTURE_CONTROL` | 已进入松夹，机械臂本轮不再跟随 |
| `GESTURE_PAUSED` | 控制手无效，停发夹爪目标，不自动回夹 |
| `HANDOVER_DONE` | 已确认全开，本轮结束并原地保持 |
| `HANDOVER_CANCELLED` | 松夹后A持续移走，停止新开合目标并保持 |
| `invalid_palm_depth` | 有二维识别但没有足够可信的掌心深度 |
| `hand_approach_stalled_hold` | 有目标但连续缺乏有效运动进展，暂停 |
| `GESTURE_FAULT` | 故障锁定，停止新的夹爪目标 |

主伺服日志：`logs/servo_diagnostics/<时间_p进程号>/samples.jsonl`，真机另记录 `control_cycles.jsonl`。
第4项无执行诊断还会在同目录 `rgbd/` 保存有限数量的同帧图像、深度和关键点；真机命令默认不保存这些图像。
空夹爪脚本日志：`logs/pinch_gripper/<时间>/samples.jsonl`。

`Q`、`Esc`、关闭主伺服预览窗口或Ctrl+C属于受控退出，不是急停。退出后最后已发送的小夹爪目标仍可能完成。
本次逻辑修改未证明此前退出抖动已经解决；需要排查时保留整个会话的两份主伺服日志。
以上旧手势测试中的力矩只记录，不参与松夹。11月9日新增fusion模式另见本文件顶部；其代码可联合视觉/动作模型决策，但尚无本机真实模型或新交接效果验证。

本文件中的参数已核对CLI；代码离线验证不等于整套持杯交接已经通过真机验证。
