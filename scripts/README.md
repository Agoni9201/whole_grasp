# Test Scripts

## force_ctrl 验证脚本（Step 4-5）

按照 REFACTOR_PLAN.md Section 5 的验证计划，依次执行：

### 5.1 零重力测试
```bash
# Dry-run（只打印，不执行）
python3 scripts/test_zero_gravity.py --channel can0

# 实机执行（需要 --execute）
python3 scripts/test_zero_gravity.py --channel can0 --execute --kd-scale 0.5 --joint-factors "1,1,1,1,1,1,1.5"
```
通过标准：使能后无冲击、不主动运动、人手轻拖方向正常

### 5.2 位置保持测试
```bash
python3 scripts/test_force_position_hold.py --channel can0
python3 scripts/test_force_position_hold.py --channel can0 --execute --duration 30
```
当前结果：30s 实机保持已通过，可以保持，不下掉、不抖。记录中最大误差约 1.202deg，主要来自 joint2。

通过标准：10-30s 内 TCP 不下掉、q2/q4 不继续漂移

### 5.3 单关节微动测试
```bash
# 待补；当前先用 position hold + ArUco dry-run 作为进入视觉伺服前的最小门槛
```
通过标准：每次只动一个关节、方向正确、无明显振荡

### 5.4 固定保持 30s
```bash
python3 scripts/test_force_position_hold.py --channel can0 --execute --duration 30
```
通过标准：30s 内 TCP 不下掉、q_err 不扩大

### 5.5 笛卡尔干跑测试
```bash
python3 scripts/test_aruco_jacobian_servo_dryrun.py --channel can0
```
通过标准：qdot 方向合理、q_cmd 变化小、CPV velocity dry-run 不发 CAN

### 5.6 SDK 反馈 ArUco dry-run（主线，不依赖 ROS）
```bash
python3 scripts/aruco_jacobian_servo_sdk_dryrun.py \
  --marker-center 0.25,0.0,0.45 \
  --marker-normal -1,0,0
```

说明：这个脚本只用 `pyAgxArm.connect(start_read_thread=True)` 读取当前关节角，不 enable、不切控制模式、不发送 `move_cpv_vel/move_mit/move_j`。

通过标准：target_tcp、qdot、q_cmd 方向合理；CPV velocity dry-run 不发 CAN；不会有任何实机运动。

### 5.7 相机 ArUco + SDK dry-run（当前集成版）
```bash
# Terminal A: 启动 RealSense 相机
ros2 launch realsense2_camera rs_launch.py \
  align_depth.enable:=true \
  depth_module.depth_profile:=1280x720x30 \
  rgb_camera.color_profile:=1280x720x30

# Terminal B: 只需要相机 image/camera_info topic；不需要再开 aruco_ros single
python3 scripts/aruco_camera_jacobian_servo_sdk_dryrun.py
```

等价于把之前两个程序合并到当前项目内：

```bash
ros2 run aruco_ros single ...
python3 /home/dpf/桌面/RealRobot_Test/real_test/real_visual_servo/aruco_pose_from_eye_in_hand.py ...
```

默认参数沿用已验证流程：

```text
marker_id=582
marker_size=0.0677
aruco_dict=ARUCO_ROS_ALL
handeye=/home/dpf/桌面/RealRobot_Test/real_test/configs/2026-05-19_14-45-49_calibration.json
handeye_mount_frame=feedback_link7
offset_base=[0.02, -0.025, 0.0]
```

说明：`marker_id` 只用于筛选目标标签；脚本默认尝试常见 OpenCV ArUco 字典，然后只接受 id 为 582 的检测结果。坐标尺度主要由 `marker_size`、相机内参、handeye 和当前关节角决定。

通过标准：能打印 `marker_camera/marker_base/normal`，`target_tcp` 正常，`qdot` 无跳变，`step_max <= 0.003rad`。

相机 topic 检查：

```bash
ros2 topic hz /camera/camera/color/image_raw
ros2 topic echo /camera/camera/color/image_raw --once
ros2 topic echo /camera/camera/color/camera_info --once
```

`image_raw` 需要有非零 `width/height/step/data`。如果脚本打印 `empty image msg` 或 `unsupported image encoding`，先看这三个字段。

### 5.8 ROS 桥接 ArUco dry-run（兼容旧目标发布，不是主线）
```bash
# Terminal A: 发布 base frame 下的腕部 ArUco pose
python3 /home/dpf/桌面/RealRobot_Test/real_test/real_visual_servo/aruco_pose_from_eye_in_hand.py \
  --handeye /home/dpf/桌面/RealRobot_Test/real_test/configs/2026-05-19_14-45-49_calibration.json \
  --handeye-mount-frame feedback_link7 \
  --tcp-feedback-frame tcp_offset \
  --output-topic /aruco_pose_base

# Terminal B: 只计算 servo 命令，不执行机械臂
python3 scripts/aruco_jacobian_servo_ros_dryrun.py
```
说明：这个脚本依赖 `/feedback/joint_states` 和 `/aruco_pose_base`，只用于复用旧 ROS ArUco publisher 做桥接验证。

通过标准：marker normal 方向正确、target_tcp 在 marker normal 前方、qdot 无跳变、q_cmd 每周期变化小。

### 5.9 相机 ArUco + Panthera PBVS + MIT/重补偿低速执行
```bash
# Terminal A: 启动 RealSense 相机
ros2 launch realsense2_camera rs_launch.py \
  align_depth.enable:=true \
  depth_module.depth_profile:=1280x720x30 \
  rgb_camera.color_profile:=1280x720x30

# Terminal B: 先复跑 dry-run，确认固定 marker 稳定
python3 scripts/aruco_camera_jacobian_servo_sdk_dryrun.py \
  --aruco-dict DICT_ARUCO_ORIGINAL \
  --duration 15

# Terminal C: 慢速移动到视觉伺服起始关节位；到位后脚本退出
python3 scripts/move_to_aruco_start_pose.py \
  --execute

# Terminal C: 再启动真机低速跟踪；必须显式加 --execute
python3 scripts/aruco_camera_jacobian_servo_sdk_execute.py \
  --execute \
  --duration 30
```

说明：当前 ArUco 控制律按 Panthera `19_aruco_marker_tracking_velocity_jacobian.py` 复现：marker pose -> PBVS TCP twist -> 6x7 Jacobian qdot。Panthera 最后发送 `Joint_Vel`；Nero 这里因为 CPV/follower 已实测出现 J2/J7 跳变，默认把 qdot 积分成小步 `q_cmd`，通过 `SdkForceCtrlBackend -> NeroForceController -> NeroMITLoop -> NeroGravityCompensator` 持续发送 `q_cmd + zero_vel + gravity + KP/KD`。

默认启动姿态为当前确认的安全起始位：`[3.73, -3.32, 7.29, 101.74, -3.76, 10.53, 29.42]deg`。脚本不会自动移动到该姿态，只会要求当前反馈在 `--start-tolerance-deg` 范围内；如需临时从其他姿态启动，可显式加 `--no-require-start-pose`。

默认控制为 Panthera PBVS ArUco 跟随：`--execution-backend mit-position`、`--firmware v111`、`--servo-mode pbvs`、`desired_distance=0.15m`、`max_linear_vel=0.35m/s`、`max_angular_vel=2.20rad/s`、`max_joint_vel=0.15rad/s`、`max_joint_step=0.006rad`。默认重力补偿逐关节系数为 `--joint-factors 1,1,1,1,1,1,1.5`，对齐已经通过的零重力测试。要让位置跟随更软，可以显式加 `--kd-scale 0.5` 或在命令行降低速度上限。

运行中 marker 丢失或目标跳变时保持上一次 MIT 目标关节位；靠近关节限位会保持当前反馈位置，硬限位会急停。默认不再启用“误差过大 hold”和“不收敛 hold”，因此 marker 放远时会持续朝目标走，直到限位保护或你手动停止。

CPV velocity 后端只保留为诊断路径。若确实要诊断，必须显式指定：

```bash
python3 scripts/aruco_camera_jacobian_servo_sdk_execute.py \
  --execute \
  --execution-backend cpv \
  --firmware v112 \
  --cpv-signs 1,1,1,1,1,1,1 \
  --duration 10
```

注意：当前 Nero CPV/follower 状态切换已经出现过 J2/J7 大跳变，所以它不再作为 Panthera 主线复现方式。

### 5.9.1 基座坐标系下的固定 6D TCP 位姿

`test_impedance_fixed_target.py` 可以把 TCP 移到一个绝对基座坐标系位姿。位置用米；
姿态可以用基座固定轴的 roll/pitch/yaw（度），或四元数 `x y z w`。脚本接入后
先零重力稳定，再在 200 Hz 阻抗环中连续移动到目标，不是关节位置硬跳转：

```bash
python3 scripts/test_impedance_fixed_target.py \
  --execute --duration 30 \
  --target-xyz 0.35 0.00 0.45 \
  --target-rpy-deg 180 0 0 \
  --cartesian-kp 200 200 200 5 5 5 \
  --cartesian-kd 5 5 5 0.2 0.2 0.2 \
  --task-torque-limit 10 \
  --joint-weights 1 1 1 1 0.5 0.5 0.5 \
  --coulomb 0 0.4 0 0.4 0 0 0 \
  --joint-factors 1 1.25 1 1.25 1 1 1.5 \
  --max-excursion 0.30 --safe-margin 0.30
```

也可以把姿态改成 `--target-quaternion X Y Z W`；两种绝对姿态参数不能同时给。
不传绝对姿态时，省略 `--rotz-deg` 表示保持接入时姿态；`--rotz-deg` 仍表示
相对起始姿态绕基座 Z 轴的旧行为。目标位置和姿态必须先在机器人基座坐标系中
确认，不能直接把相机坐标或物体坐标原样填入。第一次实机只用空夹爪、低速和较小
位移验证到位误差，确认无误后再接近物品。

### 5.10 静止 ArUco 的首轮笛卡尔阻抗视觉跟随

先通过上面的相机 dry-run，并保持 marker 静止、急停可用。首轮只跟位置，
接入后不要推机械臂；先验证目标缓慢收敛，以及短暂遮挡 marker 时平衡点冻结：

```bash
python3 scripts/aruco_camera_jacobian_servo_sdk_execute.py \
  --execute \
  --duration 30 \
  --execution-backend mit-position \
  --control-mode mit-impedance \
  --servo-mode pbvs \
  --no-require-start-pose \
  --settle-s 3 \
  --image-timeout-s 0.20 \
  --max-linear-vel 0.06 \
  --max-joint-vel 0.10 \
  --impedance-law plain-jtf \
  --cartesian-kp "200,200,200,5,5,5" \
  --cartesian-kd "5,5,5,0.2,0.2,0.2" \
  --impedance-task-torque-limit 10 \
  --impedance-torque-limit 10 \
  --torque-limit 14 \
  --impedance-joint-weights "1,1,1,1,0.5,0.5,0.5" \
  --impedance-coulomb "0,0,0,0,0,0,0" \
  --joint-factors "1,1.25,1,1.25,1,1,1.5" \
  --gravity-factor 1 \
  --kd-scale 0 \
  --impedance-joint-hold-kp "0,0,0,0,0,0,0" \
  --use-real-velocity \
  --impedance-real-vel-filter-tau 0 \
  --impedance-target-linear-vel 0.01 \
  --impedance-max-position-error 0.015 \
  --no-impedance-visual-orientation \
  --impedance-max-orientation-error-deg 5 \
  --impedance-max-excursion 1.0 \
  --impedance-safe-margin 0.15 \
  --impedance-fallback-oscillation 1.0 \
  --impedance-instant-qdot-cap 3.0 \
  --impedance-qdot-trip-fast 2.0
```

这里故意不复制固定目标测试中的 J2/J4 Coulomb `0.4`：视觉平衡点会移动，
混合跟踪路径现在把 PBVS `qdot` 积分为 `q_follow`，以低 firmware `Kp/Kd`
提供连续运动，同时保留 Cartesian `J^T F` 的推拉柔顺。若启用 Coulomb，
它按 `q_follow` 的运动方向补偿；不要再把固定接入姿态作为移动目标。
首轮保留已经通过固定目标测试的宽松 `1.0rad` 起点包络；视觉长期移动时
可用 `--impedance-max-excursion 0` 禁用这项不适合移动任务的包络。
无论该项是否启用，硬关节限位、0.15rad 限位裕量、反馈陈旧、力矩裁限
和速度保护都仍然有效。

---

## 运行形态

```text
Terminal A: force_ctrl test script (--execute)
Terminal B: joint/tcp monitor
```

**安全注意：**
- 测试前确认急停可用
- 必须先停止 ROS arm controller
- 所有脚本默认 dry-run，必须加 `--execute` 才会真机执行
- Ctrl+C 随时急停
