# 叉叉点测与标定记录

当前约定（2026-10-15）：

- **内参保持 SDK 自带参数，不替换。** 今天得到的新内参仅记录在 [INTRINSICS_2026-10-15.md](../config/INTRINSICS_2026-10-15.md)。
- 抓取、伺服外参均改用 `config/2026-05-19_14-45-49_calibration.json`，原始JSON内容不变。
- 接下来只做一个叉叉：视觉定位→机械臂接近预测点→原路返回拍照位→示教轻触→比较XYZ并生成平移补偿候选。
- 单点只能得到该工作姿态附近的平移补偿，不求解旋转，也不能单凭拟合点宣布全工作空间精度。

## 当前执行方法：一个叉叉，视觉／示教对照

这是**监督下的点测往返**，不是无人值守的力控探触，也没有全局避障。关闭其他机械臂控制程序，保持叉叉、相机安装和夹爪开度不变；手和身体不要进入运动范围。

### 1. 启动

先把相机摆到能看见叉叉、深度有效的位置，机械臂停稳。下面的 `--execute` 会使能机械臂并做现有抓取适配器的启动保持检查；不会自动去初始位或开合夹爪。

```bash
cd /home/yx/yx/panda_ws/src/whole-grasp
conda activate grasp_yolo26

python tools/real_nero_point_test.py \
  --calib-json config/2026-05-19_14-45-49_calibration.json \
  --handeye-convention flange_camera_xyzw \
  --camera-serial 243622072771 \
  --camera-width 1280 --camera-height 720 --camera-fps 30 \
  --camera-emitter-enabled --camera-laser-power 360 \
  --nero-channel can0 --nero-firmware V111 \
  --nero-speed-percent 1 \
  --skip-initial-move --target-frame tcp --hover 0.03 \
  --cross-check --cross-step-mm 1 --execute \
  --save-dir logs/cross_point
```

启动后点击叉叉中心、按Enter，同时保存拍照位的关节角。深度使用SDK原内参和15帧有效中值；采集期间有运动或叉叉深度不合格时拒绝该次记录。**拍照后先不要示教移动，直接输入m。**

### 2. 输入m，执行一次去程和返程

1. 在终端输入 `m`。程序先预检**完整去程和返程**；你确认无遮挡后，输入 `MOVE`。
2. **只求两个终点IK：叉叉上方30 mm的悬停位、叉叉视觉位置。** 观察位→悬停位只发一次 `move_j`，控制器执行关节插值；不再先横移、逐毫米点动或反复停稳。
3. 到悬停位后，输入 `APPROACH`，再发一次 `move_j` 到视觉目标。记录实际执行TCP后，按“目标→悬停→原关节角”两次 `move_j` 自动返回。返程不重新求IK、不去项目固定初始位。若按Enter放弃接近，则从悬停位一次 `move_j` 返回。
4. 看到提前接触、顶压或异常，立即 **Ctrl+C**。程序会请求停止，不在故障后强行继续接近或自动返程；保留现有停机确认与必要的急停升级。

目标朝向沿用观察位、不开合夹爪；关节插值并不保证TCP走直线或途中朝向完全不变。发送前对**实际要执行的关节插值**密集采样，检查限位、FK、工作空间和途中姿态；这些采样点**只检查、不发送**。去程总长限制500 mm，接近目标就是视觉高度，不额外压低；没有全局避障，须确认整条关节运动和工具路径无遮挡。

只在悬停、目标、返回这些**阶段终点**验收：默认关节最大误差≤0.5°、TCP位置误差≤2 mm、姿态误差≤1°，并确认新反馈连续停稳。每阶段默认超时60秒，可用 `--move-wait-timeout` 指定。它们是运动到位条件，不等于相机定位精度；实际TCP会单独保存。**到预测坐标不代表确认了真实轻触。**

已移除本点测脚本额外添加的 **2 N·m 软件判停**及其力矩时间戳／等待要求，不改机械臂自身保护、抓取或伺服保护配置。关节／法兰反馈新鲜度、路径和限位检查、到位验收、Ctrl+C停机与故障锁存仍保留。接近阶段必须人工观察，提前接触或顶压就立即Ctrl+C，不依靠软件力矩判据识别轻触。

原命令不变；`--cross-step-mm 1` 现在只表示插值FK的**预检采样密度**，不是执行步长，也不需要输入d。若已示教离开原位或刚更新代码，退出并重新启动、采图，不使用旧原位记录。超时或异常会保存最后目标／实际关节、TCP、误差和阶段，停机后不强行返程。

### 3. 示教记录实际点并比较

确认机械臂正常返回后，切到示教，让**同一已定义TCP的接触部位**轻触同一个叉叉；夹爪开度不变，停稳后输入 `r`，再输入 `TOUCH`。可以保持示教模式记录，朝向可以不同，**不再用3°姿态差阻止记录**；程序使用示教时完整法兰位姿换算TCP，并保存姿态差。

终端输出：

```text
视觉 - 示教 TCP XYZ_mm = [...]
空间误差 = ... mm
需要补偿的基座 XYZ_mm = [...]
```

比较的是**视觉点与示教TCP**，不把法兰位置直接当TCP，也不把30 mm悬停量算成误差。相机观察时的法兰变换与示教时的法兰变换分开保存。
完成过m往返时，还会显示“实际执行TCP－示教TCP”和“实际执行TCP－视觉点”。实际执行值保存在**返回前**，不会拿返回后的TCP与叉叉比较。外参补偿只用“示教－视觉”，不吸收悬停量或控制跟踪误差；若TCP实际接触点本身不准，该误差也会混入定位结果。

### 4. 生成补偿候选，再验证

输入 `k`，在本次 `logs/cross_point/时间戳/` 下生成：

- `cross_comparison.json`：两套XYZ、误差、实际姿态、时间戳及原标定哈希。
- `calibration_cross_compensated.json`：只修平移的相机到法兰候选，旋转保持原值。
- `cross_roundtrip.json`：m结束时已保存的往返状态、原位关节角、计划／实际路点、预测点处实际TCP与返回误差。故障也留记录，不因失败自动继续运动。

计算为：`Δ_base = P_示教TCP - P_视觉`；`t_新 = t_原 + R_观察法兰.T × Δ_base`。不是把基座误差直接加到法兰坐标下的外参平移上。

原JSON和 `pipeline.yaml` 不会被 `k` 覆盖。退出后，用同一命令将 `--calib-json` 改成候选的完整路径，重新拍图、点测、示教比较；以新采集的实际空间误差是否在**5 mm**内判断，不拿原拟合点的零残差当作成功。

## 旧触点流程留档（当前不再重复采集）

历史触点方案假设有固定实体尖点。当前不再采用此假设，以下仅留档；不要按下面内容重采8点。

**这个工具全程只读机械臂，不使能、不切换控制模式、不自动移动、不修改原项目参数。**
使用生产相机相同的 SDK 彩色对齐深度／针孔反投影，不使用实验 ChArUco K/D，不估计深度缩放。
它绕过平面 PnP，不代表所有内参、机器人实际运动学或安装误差都已解决。

## 1. 准备

- 在工作区域固定8个编号叉叉，分散摆放，建议两三种高度。用不透明、无反光、刚性的表面，叉叉不要靠近方块边缘。
- 1～6用于求解；7、8只用于验证，不能后来改成训练点。
- 不必测量各点间距。**从第一次触碰到最后验证，所有叉叉都不能移动。**
- 相机、尖点安装固定，TCP必须与已确认的实体尖点一致。触碰时不要顶动方块或压弯表面。
- 自行通过已熟悉的安全方式进入机械臂示教状态；工具不会代为进入示教。关闭其他占用相机的程序，不要同时运行其他机械臂控制程序。

在项目根目录的同一个终端运行；保留下面的目录变量：

```bash
conda activate grasp_yolo26
cd /home/yx/yx/panda_ws/src/whole-grasp
TOUCH_CAL_DIR="logs/touch_handeye_$(date +%Y-%m-%d_%H-%M-%S)"
```

## 2. 依次轻触，记录真实参考点

```bash
python tools/calibrate_realsense_touch.py touch \
  --output "$TOUCH_CAL_DIR/references" \
  --channel can0 --firmware V111
```

按终端提示，尖点实际轻触1～8号叉叉，停稳后按 Enter。记录的是
`P_base = T_base_flange × [0.165, 0, -0.025, 1]`，**不是相机观察时悬空 TCP 的位置**。
每点取10次新鲜法兰反馈；运动或反馈不更新时拒绝记录。`q`取消，未完成文件不能求解。

## 3. 相机观察相同的8个叉叉

触碰全部结束后，移开尖点，把腕部相机放到能看清这些叉叉的位置。采集和触碰是分开的阶段，不要求腕部相机在触碰时保持固定。

```bash
python tools/calibrate_realsense_touch.py observe \
  --references "$TOUCH_CAL_DIR/references/references.json" \
  --output "$TOUCH_CAL_DIR/observe_train" \
  --serial 243622072771 --width 1280 --height 720 --fps 30 \
  --channel can0 --firmware V111
```

Enter打开实时预览，自行调整腕部相机，停稳后在图像窗口按 **s** 冻结。之后按提示依次点击对应编号的叉叉中心，每点按 Enter确认；`q`退出。
标记和画面中的编号必须一一对应。冻结的是已记录法兰位姿对应的RGB-D，不会用点击时的新姿态代替。
深度取15帧有效中值，拒绝不稳定深度、孔洞过多或跨深度边界的像素块，不填洞。

如需匹配当前设备的显式激光设置，`observe` / `verify` 可另外加 `--emitter-enabled --laser-power 360`；功率必须在设备支持范围内。

## 4. 离线求解新候选，不覆盖旧标定

```bash
python tools/calibrate_realsense_touch.py solve \
  --references "$TOUCH_CAL_DIR/references/references.json" \
  --observations "$TOUCH_CAL_DIR/observe_train/observations.json" \
  --output "$TOUCH_CAL_DIR/handeye_candidate.json"
```

6个训练点做刚体三维配准，尺度固定为1，输出 `T_flange_camera`（相机点转换到法兰）。7、8号只统计误差。
输出同时包含 `matrix_4x4`、`position/orientation` 和 `translation_m/quaternion_xyzw`；四元数顺序是xyzw，位置单位米。
候选不会自动写入 `config/pipeline.yaml` 或覆盖旧JSON。求解画面中的留出点误差还不算独立新视角验收。

## 5. 换观察姿态，独立验证

只移动机械臂，不动叉叉。换一个明显不同且能看到叉叉的观察姿态，建议左右各做一次：

```bash
python tools/calibrate_realsense_touch.py observe \
  --references "$TOUCH_CAL_DIR/references/references.json" \
  --output "$TOUCH_CAL_DIR/observe_valid_left" \
  --serial 243622072771 --point-ids 7 8

python tools/calibrate_realsense_touch.py observe \
  --references "$TOUCH_CAL_DIR/references/references.json" \
  --output "$TOUCH_CAL_DIR/observe_valid_right" \
  --serial 243622072771 --point-ids 7 8

python tools/calibrate_realsense_touch.py validate \
  --references "$TOUCH_CAL_DIR/references/references.json" \
  --observations "$TOUCH_CAL_DIR/observe_valid_left/observations.json" \
                 "$TOUCH_CAL_DIR/observe_valid_right/observations.json" \
  --calibration "$TOUCH_CAL_DIR/handeye_candidate.json" \
  --output "$TOUCH_CAL_DIR/validation.json"
```

每批仍是预览按s、点击叉叉按Enter。报告包含位置误差均值、RMSE、最大值、XYZ误差以及同一点跨视角最大两两距离。
拒绝训练图片冒充新验证数据、相机序列号／分辨率／K/D不匹配、触点参考文件不一致。
**软件不会自动宣告“达到2 mm”或授权运动；看新点、新视角的实际误差。**

## 6. 画面保留一个叉叉，实际轻触核对

例如保留7号（其他标记可以不在画面中，但不能移动已测点）：

```bash
python tools/calibrate_realsense_touch.py verify \
  --references "$TOUCH_CAL_DIR/references/references.json" \
  --calibration "$TOUCH_CAL_DIR/handeye_candidate.json" \
  --point-id 7 --output "$TOUCH_CAL_DIR/verify_cross7" \
  --serial 243622072771 --hover-mm 30
```

1. 预览按s，点击7号叉叉按Enter，先看“视觉预测－原触点参考”的XYZ／空间误差。
2. 工具只计算同姿态、预测点上方30 mm的悬停位姿。**它没有IK／全臂碰撞预检，不能把输出直接粘贴成运动命令。**
3. 第一轮不自动走向叉叉：你在安全示教状态下让实体尖点真正轻触叉叉，停稳后在终端输入 `TOUCH`。不确定就 Enter跳过。
4. 对比“视觉－实际轻触”以及“实际轻触－原参考”，结果保存到 `verification.json`。前者包含整条视觉定位链的误差；后者用于看重复触碰、机器人姿态变化或标记移动的影响，不能单独归因于手眼。

本轮**没有自动悬停、自动下降或接触检测**。MIT阻抗不等于可靠的触碰检测。须先确认这轮误差，再另行把验证运动接入现有完整安全规划／控制流程。

所有输出目录／最终JSON都不覆盖已有文件。重复运行时换一个新名字；旧参数和原始数据都保留。
