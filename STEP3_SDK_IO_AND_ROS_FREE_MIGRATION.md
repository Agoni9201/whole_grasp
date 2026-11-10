> **2026-09-15：SDK 改造已实施，但实机验收未通过。请先阅读文末“已实施 SDK 改造：实际结果”，当前仍缺 TRAC-IK、CUDA 不可用且有 5 项回归失败。历史 diff 不再作为可直接应用方案。**

# STEP3：双相机 SDK 化、输入输出契约与改造草案

审计日期：2026-09-14。本次只新增文档，未修改运行代码。
本文的硬件职责是用户目标架构；当前代码没有完成固定全局相机到腕部多视角的双相机切换。
以下大范围 diff 是设计草案，含省略号/待实现函数，不可直接 git apply；小范围精确 diff 见末尾。
行号以本次读取版本为准。基础环境 import 成功不等于 GraspNet CUDA 扩展、相机、实机已通过验收。

## 硬件职责

|设备|用途|标定|
|---|---|---|
|Orbbec Gemini 330|抓取初始视图（全局）|独立 `T_base_global_camera`|
|Intel RealSense D435i|抓取多视角 + Servo 人手/ArUco|独立 `T_feedback_link7_wrist_camera`|
|NERO/AGX|move_j、关节反馈、MIT、夹爪|pyAgxArm + socketcan `can0`|

## 当前抓取数据链（代码现状）

1. 相机采集：`real_nero_grasp/perception.py:capture_rgbd_xyz_frame`。输入是彩色 `H×W×3`、对齐深度 `H×W`、内参；输出组织化 XYZ `H×W×3`，float32，单位米。
2. 深度反投影：`real_nero_grasp/depth_processing.py:depth_to_xyz_image`，`x=(u-cx)z/fx`、`y=(v-cy)z/fy`，无效点为 NaN。H/W 由相机实际输出决定，不在代码中固定。
3. YOLO：彩色图 + XYZ 图，输出实例 mask（H×W）及实例信息。
4. 目标/场景点云：`cloud_from_mask` 与 `scene_cloud_from_xyz` 输出 Open3D 点云，形状为可变 `N×3`；随后按 `voxel_size` / `scene_voxel_size` 降采样。
5. GraspNet：`grasp_workflow.py:664-668` 调 `make_end_points_from_o3d_cloud`。默认 `--grasp-num-points=3000`，不足则有放回采样，超过则无放回采样，输入张量为 `point_clouds: (1,3000,3)`、`cloud_colors: (1,3000,3)`，CUDA float32。
6. 可视化/碰撞点云：最多保留 20000 点（大脚本 `make_end_points_from_o3d_cloud` 的 `vis_cap`），这不是网络输入上限。

因此替换相机时必须保证：彩色与深度已对齐、深度单位米、内参对应彩色分辨率；算法不要求固定的 H×W，但网络输入固定为 `1×3000×3`（可由 CLI 改 `--grasp-num-points`）。

### 不是固定“几乘几”的三个层次

|层次|实际维度|颜色/单位|证据|
|---|---|---|---|
|彩色帧|H×W×3|uint8 BGR，0–255|perception.py:1502|
|对齐深度|H×W|float32 米|depth_processing.py:56|
|组织化点云|H×W×3|float32 XYZ，米，相机光学系|depth_processing.py:64|
|筛选目标云|N_target×3|Open3D points；颜色 RGB 0–1|perception.py:850|
|环境云|N_scene×3|同上，点数独立于目标点数|perception.py:883|
|模型批输入|1×P×3，默认 P=3000|float32 CUDA；颜色是独立同形状张量|大脚本:8844|

例如相机选择 640×480 时，数组是 480×640×3，最多 307200 个像素位置；这只是示例，不能声称当前驱动正在输出此分辨率。`scripts/launch_orbbec.sh:26` 没有显式指定宽高/FPS，运行分辨率由驱动默认值/传入参数决定。YOLO 的 `--yolo-imgsz=640` 是推理预处理尺寸，也不是点云的宽高。

降采样后 N 不是 H×W，掩码、有效深度、近远阈值、体素、离群点过滤都改变 N。20000 只是在此函数中对 cloud_vis 的限额，不是所有环境碰撞云的统一限额。Open3D 内部 Vector3dVector 通常存 double；采样送网络时明确转换 float32。

### 建议替换接口的最小边界

优先保持 `capture_rgbd_xyz_frame -> (bgr, xyz_img, intr, out_dir)` 和 `capture_target_cloud -> (bgr, target_cloud, scene_cloud, out_dir, inst, intr)` 两个调用契约不变。
无 ROS 模式下明确禁用 ROS PointCloud2 分支，或新增 SDK 的组织化 XYZ 分支；只替换 `capture_rgbd` 而仍启用 `use_registered_points` 会继续走 ROS。

新增的 Frame 建议携带：`camera_id`、`frame_id`、递增 `sequence`、硬件时间戳、主机 monotonic 接收时间、color、depth_m、K(3×3)、畸变 D、光学坐标系名称、深度有效性掩码。时间戳必须明确时钟域；机械臂反馈应匹配曝光时刻，不能直接把处理时刻的 q 当曝光时刻的 q。

反投影用的是针孔模型，不能忽略畸变后直接套任意未校正图像。SDK 必须提供与图像匹配的校正内参，或使用 SDK 反投影；resize/crop 后也必须相应修改 K。相同宽高不等于深度已几何对齐。

## 各算法块输入输出总表

|模块/文件|输入|输出及下游|
|---|---|---|
|CLI/runtime|命令行、标定路径、执行开关|Namespace → workflow；当前 servo 参数未合并完整|
|采集 tools/real_nero_point_test.py:224|相机话题或离线 RGB/depth 文件|BGR、depth_m、intr、out_dir；实时默认仍是 ROS|
|YOLO/实例筛选 perception.py|BGR、XYZ、类别/置信度、用户选择|H×W mask + inst；可选 amodal 遮挡估计|
|depth_processing.py|深度、内参、可见 mask|XYZ 图、局部深度修复；不能用 amodal 猜出的区域伪造真实深度|
|perception.py:1556|XYZ、mask、滤波参数|目标云、场景云、实例、内参；非目标障碍云还写入 args._last_obstacle_cloud|
|transforms/runtime.py:1362|法兰位姿、手眼 JSON|4×4 T_base_camera；当前固定采用腕部乘法链|
|clear_policy/active_motion|实例、目标/障碍云、机器人状态|清障/观察运动决策、重新采图、更新场景|
|active_view/pointcloud_fusion|多次观测及各帧外参、目标身份|ActiveViewResult：target_cloud、scene_cloud、used_multiview、restore_ok、motion_sent 等；运动环境云标注 base，不能把所有返回云一概当 base|
|GraspNet 预处理 grasp_workflow.py:664|参考相机系目标 Open3D 云|point_clouds/cloud_colors 批张量、cloud_vis|
|GraspNet 候选生成|采样 XYZ、网络权重、碰撞/场景条件|抓取候选：translation(3)、rotation(3×3)、width、score；后续筛选/排序|
|grasp_geometry/grasp_candidates|候选、T_base_camera、TCP 偏置、工具轴约定|基座系 TCP/法兰预抓取、抓取、抬升目标；位置米、旋转须明确表示|
|ik_solver/kinematic_trac_ik|目标法兰位姿、关节种子/限位、URDF|7 关节候选（rad）或无解；SDK TCP 与 IK flange 必须分清|
|collision/grasp_planner/motion_planning|关节候选、机器人/夹爪/相机几何、环境点云、负载|可执行分阶段关节路径或拒绝原因；抓取后 RRT 是可选路径|
|arm_execution/session_control|关节目标、限速、到位容差、反馈|SDK move_j + 到位/停止/异常结果；非 200 Hz 视觉闭环|
|夹爪/抓取 workflow|夹爪宽度/力反馈、慢闭参数、路径|GripperCloseResult；GraspAttemptResult(status,motion_sent,detail)|
|handoff.py:95|仍在 with 上下文内的抓取 arm|停轨迹 → MIT backend → 断旧 driver；返回 backend|
|D435i/目标检测|彩色、K/D、q；MediaPipe 另需米制对齐深度|ArucoTargetResult 或 HandTargetResult：base 系中心(3)、法向(3)、切向(3)、位姿(4×4)|
|models/nero_model.py|q(7)|TCP 位置(3)、R(3×3)、雅可比；模型 clone 分离线程 Data|
|PBVS servo/pbvs_pose_servo.py:284|q、目标位置/法向/切向或旋转、时间|dict：qdot(7)、tcp_twist(6)、target_rotation、误差、状态等；角速度 rad/s，线速度 m/s|
|execution_safety.py:62|q、servo_out、时间|SafetyDecision(status,hold,fatal,reason)|
|servo_delivery 阻抗刷新|TCP twist、当前 TCP、dt、限幅|ImpedanceConfig：target_tcp(3)、R_target(3×3)、K/B、q_follow(7)、力矩限额等|
|force_ctrl MIT loop|关节/速度反馈、阻抗配置、模型、重力补偿|7 轴 MIT 指令；力矩 N·m；配置/反馈异常应停止或保持|
|递送状态机|目标误差、连续有效帧、夹爪状态、返回目标|HOLD_LIFTED→TRACK_HAND→APPROACH→HANDOVER→RELEASE→RETREAT；退出/断连|

### 两套外参与融合参考系

统一记号 `T_A_B`：将 B 系点变换到 A 系。外置 330：`p_base=T_base_global_camera*p_global`；腕部 D435i：`p_base=T_base_link7(q)*T_link7_wrist_camera*p_wrist`。网络仍要求参考相机系，则融合完用 `inverse(T_base_reference_camera)` 转回参考相机；不能直接把 base 云当网络原先的 camera 云，也不能再次重复变换。

旧 330 JSON 的“相机型号”不能证明它是外置标定：当前旧代码用 flange-camera 链，说明既有消费端假设眼在手上。必须核实旧采集安装关系。D435i 旧文件来源未验证，不能直接复用。SDK 自带 RGB-depth 外参是相机内部标定，不能替代机器人手眼标定。

## Servo 当前 I/O（尚未 SDK 化）

`servo_delivery.py` 当前是 ROS2 Node：订阅腕部 Image/CameraInfo，30 Hz 取 7 关节反馈，目标源输出 3D 中心/法向/切向（base 系），PBVS 输出 `qdot`、TCP twist 和误差，随后刷新阻抗配置；MIT 内环为 200 Hz。MediaPipe 分支的 `self._latest_depth` 当前没有深度订阅，不能直接切换。

## 无 ROS 改造 diff 草案（尚未执行）

```diff
--- a/real_nero_grasp/perception.py
+++ b/real_nero_grasp/perception.py
@@
-    capture_rgbd_fn = _default_capture_rgbd if capture_rgbd_hook is None else capture_rgbd_hook
+    capture_rgbd_fn = sdk_camera_capture if capture_rgbd_hook is None else capture_rgbd_hook
```

实际应新增统一 `cameras/` 适配器，分别实现 Gemini 330 和 D435i 的 `get_frame() -> {color, depth_m, intrinsics, timestamp}`；不能把两个相机共用一份外参。

```diff
--- a/servo_delivery.py
+++ b/servo_delivery.py
@@
-import rclpy
-from rclpy.node import Node
-from sensor_msgs.msg import CameraInfo, Image
+from cameras.realsense_d435i import RealSenseD435i
@@
-class ServoDeliveryNode(Node):
+class ServoDeliveryLoop:
@@
-        self.create_subscription(...)
-        self.create_timer(1.0 / float(args.servo_hz), self._on_timer)
+        self.camera = RealSenseD435i(...)
+        # monotonic loop；控制频率与采集线程解耦
```

```diff
--- a/real_nero_grasp/paths.py
+++ b/real_nero_grasp/paths.py
@@
-DEFAULT_YOLO_WEIGHTS = str(PROJECT_ROOT / "models" / "yolov8s-seg.pt")
+DEFAULT_YOLO_WEIGHTS = str(PROJECT_ROOT / "models" / "yolo26m-seg.pt")
```

```diff
--- a/config/pipeline.yaml
+++ b/config/pipeline.yaml
@@
-  calib_json: /path/to/你的全局相机标定.json
+  calib_json: /实际路径/global_camera.json
+  convention: base_global_camera
@@
-  calib_json: /home/dpf/桌面/RealRobot_Test/real_test/configs/...
+  calib_json: /实际路径/d435i_wrist_camera.json
```

## 必须先确认的契约

- Gemini 330 SDK Python 包名及彩色/深度分辨率；
- D435i 使用的分辨率、FPS、深度尺度（现有验证程序支持 width/height/fps 参数）；
- 全局外参 JSON 的字段和方向（建议明确 `T_base_global_camera`）；
- D435i hand-eye JSON 的字段和方向（`link7_cam` / `feedback_link7`）；
- pyAgxArm 的关节反馈、普通停止、MIT 接管和夹爪 API。

## 当前不能实机运行的原因

1. 全局 330 标定 JSON 仍是占位符。
2. 抓取默认 YOLO 路径指向不存在的 `models/yolov8s-seg.pt`。
3. 主伺服程序仍依赖 ROS2 Image/CameraInfo/timer。
4. MediaPipe 深度订阅未实现。
5. 部分启动/参考脚本含其他机器的绝对路径。

## 进一步发现：不是改路径就能解决的问题

|位置|已观察到的行为|需实现/验证|
|---|---|---|
|real_nero_grasp/cli.py:20、servo_delivery.py:429|主 CLI 强制 calib-json，没有 target-source 等 Servo 参数；独立 parse_servo_args 没有统一接到 main|按 phase 校验必需参数，主入口共享伺服参数定义；不能假设 `main.py --phase servo --target-source aruco` 当前可用|
|config/pipeline.yaml 与运行入口|本次搜索 main/runtime/handoff/servo_delivery 未找到 pipeline.yaml 加载|先建立 YAML→配置→CLI 覆盖链；当前改 YAML 不保证运行会读取。之前称占位 YAML 必然导致启动失败不准确，主抓取实际读 args.calib_json|
|servo_delivery.py:16、tools/real_nero_point_test.py:224|伺服顶层 rclpy；抓取实时采集 ROS|SDK 适配后也要去掉顶层/隐含导入依赖，FakeBackend dry-run 当前仍不能保证无 ROS|
|servo_delivery.py:204 起|保存最新图像，检测时传 q_now，未按图像时间匹配；可重复消费旧帧|有效序号去重、帧龄超时、曝光/反馈时间匹配；稳定 N 帧不能把同一张图重复 N 次|
|servo_delivery.py:222 附近|读取 _latest_depth，但未接入深度流|SDK Frame 提供 depth_m、验证对齐、空洞；D435i 深度比例从设备读取|
|handoff/grasp_to_servo_handoff.py:95|build_servo_backend 使用默认 can0/v111，未传抓取 args|透传已确认通道/固件，不得悄悄切到默认通道|
|handoff/grasp_to_servo_handoff.py:28|停止确认前就设置退出急停抑制标志|确认失败及所有异常路径必须有清晰控制权和停止责任；反馈需新鲜有效，不能只比较缓存值|
|handoff/grasp_to_servo_handoff.py:71|以 zero_gravity=False 注释声称“物体不掉”|此声明不是固件保证；离线与空载验证位置保持安装时序、接管延迟、反馈年龄|
|servo_delivery.py:468|连接 backend、建 Node 在 try/finally 外|节点/标定加载失败也应进入可靠资源回收，防止 MIT 后端遗留；预检应在执行前|
|servo_delivery.py:400 起|APPROACH 比较 PBVS error 与 handover_distance|PBVS error 是期望偏置后的误差，不等于 TCP 到手中心的实际距离；明确释放距离定义|
|servo_delivery.py:410 起|RELEASE 读取 status 但未检查开度到位就 RETREAT|需宽度阈值、连续有效反馈、超时与失败分支|
|servo_delivery.py:423|RETREAT 把固定 TCP 目标交给有 desired_distance 的 PBVS|核对固定目标是否再次被加入目标偏置，回程不应停在错误高度|

以上属于静态审计，尚未声称已重现实机故障。没有证明一次审计找出了所有隐藏问题。

## 可精确审阅的小范围 diff

以下基于当前文件行号，仍需用户确认后才修改代码。YOLO m 仅是候选默认值；若任务是 food5 类别，应确认是否改用自训练 visible 权重，不能仅按模型大小决定。

```diff
--- a/real_nero_grasp/paths.py
+++ b/real_nero_grasp/paths.py
@@ -13,1 +13,1 @@
-DEFAULT_YOLO_WEIGHTS = str(PROJECT_ROOT / "models" / "yolov8s-seg.pt")
+DEFAULT_YOLO_WEIGHTS = str(PROJECT_ROOT / "models" / "yolo26m-seg.pt")
--- a/real_nero_grasp/cli.py
+++ b/real_nero_grasp/cli.py
@@ -85,1 +85,1 @@
-    p.add_argument("--yolo-weights", default=DEFAULT_YOLO_WEIGHTS, help="YOLO segmentation weights path; default is ./yolov8s-seg.pt")
+    p.add_argument("--yolo-weights", default=DEFAULT_YOLO_WEIGHTS, help=f"YOLO segmentation weights path; default is {DEFAULT_YOLO_WEIGHTS}")
--- a/scripts/aruco_camera_jacobian_servo_sdk_dryrun.py
+++ b/scripts/aruco_camera_jacobian_servo_sdk_dryrun.py
@@ -64,4 +64,1 @@
-DEFAULT_HAND_EYE = (
-    "/home/dpf/桌面/RealRobot_Test/real_test/configs/"
-    "2026-05-19_14-45-49_calibration.json"
-)
+DEFAULT_HAND_EYE = None
@@ -496,1 +493,1 @@
-    parser.add_argument("--handeye", default=DEFAULT_HAND_EYE)
+    parser.add_argument("--handeye", required=True)
```

最后一项不能孤立应用：execute 参考脚本导入该常量（scripts/aruco_camera_jacobian_servo_sdk_execute.py:58、2212），也必须同步改为 `parser.add_argument("--handeye", required=True)`。主递送程序则应走共享配置加载和文件预检，不能让 None 进入 Path/JSON loader。旧脚本 docstring 中的历史路径仅是示例文字，与实际执行路径分开处理。

### 大范围接口差异草案：按实施顺序

```diff
--- /dev/null
+++ b/cameras/frame.py
@@ proposed contract (非可直接应用补丁)
+# RGBDFrame: bgr(H,W,3), depth_m(H,W), K(3,3), D,
+# sequence, camera_id, optical_frame, capture_stamp, received_monotonic
+# Camera: start(), get_frame(timeout), stop(); 严格验证形状/单位/对齐
--- /dev/null
+++ b/cameras/realsense_d435i.py
@@ proposed
+# pyrealsense2 按 serial 选择，持续 pipeline、align color、读取 depth_scale
+# 参考 tools/verify_handeye_point.py:146；不逐帧反复开关 pipeline
--- /dev/null
+++ b/cameras/orbbec_camera.py
@@ proposed
+# pyorbbecsdk 版本确认后实现；显式选择 profile、对齐和同步
+# SDK 原始数据格式与深度单位在边界归一化
--- a/real_nero_grasp/runtime.py
+++ b/real_nero_grasp/runtime.py
@@ -1362 onward (架构草案)
-# 单一路径：T_base_flange @ T_flange_camera
+# resolve_camera_pose(frame.camera_id, exposure_joint_state, calibration)
+# 固定全局返回固定外参；腕部返回与 mount_frame 匹配的 FK × 手眼
--- a/real_nero_grasp/active_view.py
+++ b/real_nero_grasp/active_view.py
@@ proposed
+# 初始 global 目标 → base 定位 → 规划 wrist 观察
+# wrist mask/内参/自遮挡/外参随帧更新，融合到共同参考系
--- a/real_nero_grasp/cli.py
+++ b/real_nero_grasp/cli.py
@@ proposed
-# 无条件 required calib-json + 孤立 parse_servo_args
+# 共享参数注册器，--phase 条件校验，--config 读取 YAML
+# 配置优先级：显式 CLI > YAML > 内置默认；两相机外参分别校验
--- a/servo_delivery.py
+++ b/servo_delivery.py
@@ proposed
-# ROS Node + latest Image/CameraInfo
+# DeliveryController.step(frame, feedback, now)，无 SDK/ROS 的可测状态机
+# 外层持有 Camera/backend，try/finally 包含全部初始化步骤
+# 仅新鲜新帧更新 stable_frames；空帧执行 hold/timeout
--- a/scripts/run_grasp.sh
+++ b/scripts/run_grasp.sh
@@ proposed
-# source ROS / home/zgj 硬编码
+# 使用已选解释器；SDK 相机启动由应用管理；不 source ROS
```

完整可执行补丁须在相机 profile、标定格式与目标类别确认后产生；不能把这些架构片段误当已完成实现。

## 标定程序与验证安排

- 已有 D435i SDK 单帧采集：tools/verify_handeye_point.py:146；它是验证工具的一部分，不是完整标定采集求解器。
- 已有 pose sweep 是运动辅助，需外部采集节点；无 ROS 版本需改为保存 SDK 反馈和图像观测。
- 建议分离“无运动自动命令的采样/记录”与离线 solver，保存原始图像、板尺寸、K/D、关节、末端位姿、时钟、相机序列号和帧名。
- 外置相机：固定相机、末端刚性安装板；腕部：固定板、移动腕部相机。具体 solver 的输入方向必须按方程验证，不因函数名随便选 OpenCV API。
- 使用未参与求解的观测检查重投影误差、base 下固定目标一致性和尺度；先离线，再低速空载验证。

## 实施与验收边界

1. 核对 SDK 版本/相机 profiles/CUDA/PyTorch/pointnet2/knn/TRAC-IK，不能用 import cv2 成功代替整套依赖验收。
2. 先通过 main --help、runtime/handoff 导入，再运行 tests 与 tests_servo。测试目录文件数不等于测试用例数。
3. SDK 仅采集无 CAN：记录两相机实际 H/W/FPS、depth_scale、有效点数和帧龄；保存原始数据供离线回放。
4. 离线回放验证 `H×W×3 → N×3 → 1×P×3`、颜色/单位/内参与坐标一致；禁止模型静默下载未知权重。
5. 纯 FakeBackend + 录制图像完成状态机；FakeBackend q 固定时不会自然模拟移动到位，需明确仿真反馈模型。
6. 在各级验收通过后才安排真实抓取、空载接管和递送；所有真实运动显式 --execute，保持用户指定的 CAN 单控制源时序。

尚待讨论：初始抓取是否必须依赖外置 330，或先以 D435i 单腕部相机完成抓取多视角和递送，再加入外置全局初始化；两者工作量不同。本文按用户双相机目标记录，不宣称源工程已经有此切换能力。


## 2026-09-14 补充：按用户确认，仅提供 diff，不实施运行代码

### 当前阶段重新定义（用户确认）

本阶段把 D435i 作为**唯一启用的相机**：它暂时同时承担抓取初始视图和抓取多视角观察；Gemini 330 已改作未来的全局外置相机，本阶段完全不启动、不订阅、不参与坐标变换。因而当前验收目标不是“相机单测”，而是项目第一阶段的真实链路：

```text
D435i SDK RGB-D
 → YOLO 实例分割
 → 深度反投影/目标点云
 → 腕部相机随臂多视角采集
 → 点云融合/清障决策
 → GraspNet 候选
 → Trac-IK/碰撞约束
 → move_j + AGX 接触确认
 → 抓住并抬升保持
```

这意味着当前抓取实验需要使用 D435i 的 `T_feedback_link7_wrist_camera` hand-eye，而不能把旧 330 标定 JSON 传给抓取 CLI。未来接入 330 后，才增加独立 `T_base_global_camera` 的初始视图，并把 D435i 保留为多视角相机；那是第二个阶段，不阻塞当前 D435i 单相机验收。

### 当前阶段代码边界 diff（由用户应用）

```diff
--- a/config/pipeline.yaml
+++ b/config/pipeline.yaml
@@
-global_camera:                    # 抓取阶段（第一步）
+global_camera:                    # 暂不启用；Gemini 330 留待全局初始视图阶段
+  enabled: false
@@
-wrist_camera:                     # 伺服递送阶段（第二步）
+wrist_camera:                     # 当前抓取多视角 + 未来 Servo
+  enabled: true
+  sdk: realsense_d435i
@@
-  image_topic: /camera/camera/color/image_raw
-  info_topic: /camera/camera/color/camera_info
-  depth_topic: /camera/camera/aligned_depth_to_color/image_raw   # mediapipe 人手目标用
-  calib_json: /home/dpf/桌面/RealRobot_Test/real_test/configs/2026-05-19_14-45-49_calibration.json
+  image_topic: null                 # SDK 模式，不读取 ROS topic
+  info_topic: null
+  depth_topic: null
+  serial: <D435I_SERIAL>
+  color_width: 640
+  color_height: 480
+  fps: 30
+  calib_json: <D435I腕部手眼标定.json>
   convention: link7_cam           # ← 伺服工程已验证的约定
   mount_frame: feedback_link7
```

`pipeline.yaml` 当前尚未被主入口自动加载；上面的字段只有在实现配置加载器后才生效。当前临时验证仍应显式传 CLI 参数，避免产生“改 YAML 已接线”的错觉。

```diff
--- a/real_nero_grasp/perception.py
+++ b/real_nero_grasp/perception.py
@@
-        capture_points_fn = _default_capture_ros_color_pointcloud_frame
+        capture_points_fn = _sdk_rgbd_to_xyz_frame
@@
-        bgr, depth_m, intr, out_dir = capture_rgbd_fn(args)
+        bgr, depth_m, intr, out_dir = capture_rgbd_fn(args)
@@
-    if args.use_registered_points:
+    if args.use_registered_points and not getattr(args, "sdk_camera", False):
        # ROS PointCloud2 仅保留兼容模式；当前 D435i SDK 验收不得进入此分支
```

上段是接口位置示意；真正实现应把 SDK 采集器作为 hook 注入 runtime，不能在 perception.py 内部偷偷创建全局 pipeline。SDK 采集器必须长生命周期、启动一次、明确 serial、读取深度 scale、align 到 color，并返回既有四元组。

```diff
--- a/real_nero_grasp/runtime.py
+++ b/real_nero_grasp/runtime.py
@@
 def run_single_grasp(args):
+    # 当前阶段：唯一相机是 wrist D435i；所有初始/active-view capture 均复用同一 SDK 对象
+    args.sdk_camera = True
+    args.camera_role = "wrist"
+    args.handeye_convention = "link7_cam"
+    args.handeye_mount_frame = "feedback_link7"
```

这段不应在未实现 SDK hook 前直接应用；`run_single_grasp` 的所有 active-view recapture 参数也必须继续指向同一个 D435i 采集器，不能每次重新启动设备。

### 当前第一阶段验收顺序

1. **静态/离线**：用 D435i 录制帧确认 `H×W×3 → N×3 → 1×3000×3`，查看 raw/filtered PLY；确认深度米制和 color-depth 对齐。
2. **单视图真实抓取前感知**：D435i 采集 → YOLO mask → 目标点云，使用真实目标类别和实际权重；不发 CAN。
3. **多视角空载感知**：机械臂接入 SDK，但先 `--execute` 关闭；在安全范围验证 active-view 规划、每个观察位姿采集新 D435i 帧、目标身份重识别和融合云。
4. **抓取实机回归**：速度/力矩/碰撞护栏保持脚本默认值，`--execute` 才允许 move_j；确认夹爪接触、抬升和保持。
5. **阶段完成定义**：D435i 单相机完成“识别→多视角融合→GraspNet→IK→move_j→接触确认→抬升保持”。此时才进入抓取到 Servo 的 handoff 验证。

多视角时每帧必须保存：D435i 图像序号和时间戳、采集时关节反馈、彩色内参、对齐深度、D435i hand-eye；融合前先将各帧云转换到 base 或共同参考相机系。相机随臂运动时不能复用初始帧的 `T_base_camera`。

### 当前实验命令模板（SDK 采集器实现后）

```bash
conda activate grasp_yolo26
# 先不接 CAN：只做 D435i → 分割 → 目标云
python tools/test_D435i.py --output captures/d435i_single --yolo-weights models/yolo26m-seg.pt --target-class <实际类别> --view

# 之后才运行抓取阶段；全局 330 不启动，使用 D435i wrist hand-eye
python main.py --phase grasp \
  --calib-json <D435i腕部手眼标定.json> \
  --handeye-convention flange_camera_xyzw \
  --execute
```

上面的 `tools/test_D435i.py` 只有在你把 STEP3 中的测试 diff 应用后才存在；`main.py` 当前仍是 ROS 采集路径，不能把命令模板当作 SDK 接线已经完成。尤其 `--handeye-convention` 的实际值必须和 JSON 文件字段方向一致：`link7_cam` 是 Servo 目标类的约定，而抓取 CLI 当前枚举是 `flange_camera_xyzw` 等旧约定，不能直接混填。应在 SDK 适配阶段增加明确的抓取 wrist hand-eye 参数和转换测试。

### 必须在连接机械臂前解决的阻塞项

- 为抓取阶段实现 D435i SDK capture hook，并让所有 active-view 再采集复用它；
- 确认抓取使用的 D435i hand-eye JSON 方向、挂载帧和 TCP/flange 偏置；
- 统一抓取 CLI 的相机角色与 hand-eye convention，避免 Servo 的 `link7_cam` 字符串落入旧抓取枚举；
- 去掉抓取实时路径对 `rclpy/sensor_msgs` 的强制导入，ROS 兼容代码只能延迟导入；
- 验证 active-view 的每帧位姿与 D435i 帧时间匹配；
- 先用 dry-run/录制回放确认融合，再启动任何 `--execute` 运动。

若这些项目未完成，D435i 的单帧点云可视化成功也不能推出多视角抓取链可用。330 暂不参与，所以全局相机标定和全局初始视图暂不属于本阶段阻塞项。

本节取代前文“质量报告”的建议：不生成综合质量评分或质量报告文件。用实际抓取成功率作最终任务指标；采集阶段只在终端输出基本有效深度与点数，保存 RGB-D/PLY。采集元数据 JSON 是回放所需的内参/尺度，不是质量报告。

### ROS2 依赖清单与替换边界

|文件/位置|现有依赖和间接调用|SDK 改法及完成判据|
|---|---|---|
|tools/real_nero_point_test.py:224、287|capture_rgbd→capture_ros_rgbd_frame；另一路订阅 PointCloud2|保留离线入口，实时 SDK 返回同一四元组；注册点云开关须改为 SDK XYZ/明确拒绝，不能静默回 ROS|
|real_nero_grasp/perception.py:1490、1502|通过默认 hook 间接走上述 ROS 函数|同时替换深度与注册点云两个 hook。runtime 注入的 hook 也要更新，不能只改默认函数|
|tools/verify_handeye_point.py:253、460、647 附近|rclpy/QoS/Image/CameraInfo；SDK ImportError 后回退 PoseStamped|实时采集换 SDK；SDK 失败显式报错，取消 ROS 自动回退；禁用 read-nero-tcp-ros 旧参数|
|tools/nero_handeye_calib_pose_sweep.py:55、1406|Pose 发布器用于外部标定节点|用本地采样记录器保存相机观测和同期 SDK 位姿；不要只删除 import 而继续实例化发布器|
|servo_delivery.py:16、88、143、178、345、476|顶层 rclpy、ROS Image 转换、Node、订阅、timer、shutdown/spin|独立 Frame+step 控制器；SDK 数组直接进入目标检测；所有退出分支统一回收资源|
|scripts/aruco_camera_jacobian_servo_sdk_dryrun.py:51、322、566|名字含 SDK 但仅机器人侧 SDK，图像仍 ROS|共享 SDK Frame 与采集器；保留数学函数导入接口；SDK dryrun 可能读机械臂，不等于纯离线|
|scripts/aruco_camera_jacobian_servo_sdk_execute.py:52、2605|同上且导入 dryrun 的常量/辅助函数|先抽出纯数学/图像函数，再迁移相机与主循环；不能破坏 tests_servo 的导入链|
|scripts/aruco_jacobian_servo_ros_dryrun.py:50|ROS joint/target 话题|作为可选历史参考保留，不在无 ROS 入口中导入/执行；不要求重写所有参考脚本|
|scripts/run_grasp.sh:4、30|工程目录指向 scripts，source ROS/旧用户目录|根目录改为脚本上一级，使用激活的 conda 解释器，去掉 ROS 路径注入；保留其他 SDK/CUDA 必要环境|
|scripts/check_env.sh:4、31、63|source ROS 与空 NERO_SETUP；检查 ROS 包|不 source ROS；检查 SDK/CUDA/算法依赖并给出失败退出码，不能仅打印异常后返回成功|
|scripts/launch_orbbec.sh:26|ros2 launch|SDK 由应用持有时不再调用它；历史 ROS 启动器可保留但标为不适用于 SDK 路径|
|tests_servo/test_image_conversion.py:6|sensor_msgs Image + ROS 脚本导入|SDK 新测试用 ndarray；旧消息转换测试归入可选 ROS 测试，不能用跳过掩盖主入口残留依赖|
|tests_servo/test_execute_impedance_engage.py:21|间接 import ROS execute 脚本|抽取 engage 数学/逻辑到无 ROS 模块并更新测试 import；保留现有行为断言|
|control_backends/ros_move_j_backend.py|未完成的可选 ROS 后端占位，实际没有 rclpy import|SDK 主入口不选择它即可；不能把名字含 ros 当成必须删文件|
|targets/、models/、servo/ 算法模块|本轮未发现必须导入 ROS 消息|保留 ndarray 接口，别无谓改算法。pyAgxArm CAN 反馈不需要 ROS topic|

交付检查必须覆盖直接和间接 import；纯离线测试不 import 相机 SDK、不枚举 USB、不连接 CAN。相机工具不调用 main.py 抓取 workflow，也不传虚构标定以绕过 CLI。

### D435i 相机替换小实验：完整新增文件 diff

此工具只验证“相机帧 → 原反投影 → 原 YOLO 实例/可选手工 mask → 原目标/场景云”。不代表完整抓取迁移完成：它刻意未应用 330 专属的自遮挡/近场鬼影过滤、时间融合、主动观察或 GraspNet 二次目标预处理。保存的数据可以继续离线比较这些步骤。

复用已有 capture_realsense_frame（SDK、align color、设备 depth_scale），先限制只有一台 D435i，避免误选相机；后续生产适配器需支持 serial、持续采集、时钟/畸变元数据。640×480/30 是测试请求而非硬件保证，SDK 不支持时明确报错，按设备 profile 选择。保存点云是相机系，无需手眼标定；不能直接据此控制机械臂。

```diff
--- /dev/null
+++ b/tools/test_d435i_grasp_pointcloud.py
@@ -0,0 +1,120 @@
+#!/usr/bin/env python3
+"""Camera-only D435i/recorded RGB-D probe; no ROS or arm connection."""
+import argparse
+import json
+import sys
+from pathlib import Path
+from types import SimpleNamespace
+
+sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
+import cv2
+import numpy as np
+from real_nero_grasp.depth_processing import depth_to_xyz_image
+from real_nero_grasp.perception import (
+    cloud_from_mask, scene_cloud_from_xyz, infer_yolo_instances, mask_for_instance,
+)
+from tools.verify_handeye_point import capture_realsense_frame
+
+
+def xyz_from_frame(bgr, depth_m, intr):
+    if bgr is None or bgr.ndim != 3 or bgr.shape[2] != 3 or bgr.dtype != np.uint8:
+        raise ValueError("color must be HxWx3 uint8 BGR")
+    if depth_m.ndim != 2 or depth_m.shape != bgr.shape[:2]:
+        raise ValueError("depth must be aligned to color with matching H/W")
+    if not np.issubdtype(depth_m.dtype, np.floating):
+        raise ValueError("depth must be floating-point metres, not raw z16")
+    values = np.array([intr.fx, intr.fy, intr.cx, intr.cy], dtype=float)
+    if not np.isfinite(values).all() or intr.fx <= 0 or intr.fy <= 0:
+        raise ValueError("invalid intrinsics")
+    return depth_to_xyz_image(depth_m, intr)
+
+
+def main():
+    p = argparse.ArgumentParser(description=__doc__)
+    p.add_argument("--capture-dir", type=Path, help="Replay color.png/depth_m.npy/capture_meta.json")
+    p.add_argument("--output", type=Path, required=True, help="New output directory; never overwrite")
+    p.add_argument("--width", type=int, default=640)
+    p.add_argument("--height", type=int, default=480)
+    p.add_argument("--fps", type=int, default=30)
+    p.add_argument("--warmup", type=int, default=30)
+    p.add_argument("--mask", type=Path, help="Optional HxW nonzero foreground PNG")
+    p.add_argument("--yolo-weights", help="Optional existing segmentation checkpoint")
+    p.add_argument("--target-class")
+    p.add_argument("--target-index", type=int, default=0)
+    p.add_argument("--view", action="store_true")
+    p.add_argument("--min-depth", type=float, default=0.05)
+    p.add_argument("--max-depth", type=float, default=1.50)
+    p.add_argument("--min-target-points", type=int, default=250)
+    p.add_argument("--voxel-size", type=float, default=0.0015)
+    p.add_argument("--scene-voxel-size", type=float, default=0.004)
+    p.add_argument("--no-target-outlier-filter", dest="target_outlier_filter", action="store_false")
+    args = p.parse_args()
+    if args.mask and args.yolo_weights:
+        p.error("choose --mask or --yolo-weights")
+    if args.yolo_weights and not Path(args.yolo_weights).is_file():
+        p.error("YOLO checkpoint must already exist locally")
+    if not 0 <= args.min_depth < args.max_depth or min(args.voxel_size, args.scene_voxel_size) < 0:
+        p.error("invalid depth bounds or voxel sizes")
+    args.yolo_classes, args.yolo_conf, args.yolo_imgsz = None, 0.25, 640
+    args.box_depth_band = 0.045
+    args.outlier_neighbors, args.outlier_std_ratio, args.scene_cloud_cap = 20, 2.0, 60000
+    args.output.mkdir(parents=True, exist_ok=False)
+    if args.capture_dir:
+        root = args.capture_dir
+        bgr = cv2.imread(str(root / "color.png"), cv2.IMREAD_COLOR)
+        depth = np.load(root / "depth_m.npy", allow_pickle=False)
+        meta = json.loads((root / "capture_meta.json").read_text())
+        intr = SimpleNamespace(**{k: float(meta["intrinsics"][k]) for k in ("fx", "fy", "cx", "cy")})
+    else:
+        import pyrealsense2 as rs
+        devices = list(rs.context().query_devices())
+        if len(devices) != 1 or "D435I" not in devices[0].get_info(rs.camera_info.name).upper():
+            raise RuntimeError("Connect exactly one RealSense D435i; no automatic device guessing")
+        print("camera:", devices[0].get_info(rs.camera_info.name), devices[0].get_info(rs.camera_info.serial_number))
+        bgr, depth, intr, root = capture_realsense_frame(
+            str(args.output / "capture"), args.width, args.height, args.fps, args.warmup)
+        print("RGB-D replay directory:", root)
+    xyz = xyz_from_frame(bgr, depth, intr)
+    mask = np.ones(depth.shape, dtype=bool)
+    if args.mask:
+        image = cv2.imread(str(args.mask), cv2.IMREAD_GRAYSCALE)
+        if image is None or image.shape != depth.shape:
+            raise ValueError("mask must match frame H/W")
+        mask = image > 0
+    elif args.yolo_weights:
+        instances, _ = infer_yolo_instances(bgr, args, str(args.output / "yolo.png"))
+        for i, item in enumerate(instances):
+            print("instance", i, item["label"], item["score"])
+        if not 0 <= args.target_index < len(instances):
+            raise RuntimeError("No instance at target-index; inspect YOLO overlay")
+        chosen = instances[args.target_index]
+        if chosen.get("is_box_only", False):
+            raise RuntimeError("Segmentation mask missing; refusing box-only target comparison")
+        mask = mask_for_instance(chosen, xyz, args)
+    else:
+        print("No target mask: showing entire scene; this is not an object-specific evaluation")
+    valid = np.isfinite(xyz).all(axis=2) & (depth > args.min_depth) & (depth < args.max_depth)
+    print("frame H/W:", depth.shape, "XYZ:", xyz.shape, "metres")
+    print("selected pixels:", int(mask.sum()), "valid:", int((valid & mask).sum()),
+          "valid fraction:", float((valid & mask).sum()) / max(1, int(mask.sum())))
+    raw_args = SimpleNamespace(**vars(args))
+    raw_args.voxel_size, raw_args.target_outlier_filter = 0.0, False
+    raw = cloud_from_mask(xyz, bgr, mask, raw_args)
+    filtered = cloud_from_mask(xyz, bgr, mask, args)
+    scene = scene_cloud_from_xyz(xyz, bgr, args)
+    print("target points raw -> filtered:", len(raw.points), "->", len(filtered.points))
+    if len(filtered.points) < 3000:
+        print("Fewer than 3000 points: default GraspNet sampling will duplicate points")
+    import open3d as o3d
+    for name, cloud in (("raw", raw), ("filtered", filtered), ("scene", scene)):
+        if not o3d.io.write_point_cloud(str(args.output / (name + ".ply")), cloud):
+            raise OSError("PLY write failed: " + name)
+    if not cv2.imwrite(str(args.output / "selected_mask.png"), mask.astype(np.uint8) * 255):
+        raise OSError("mask write failed")
+    if args.view:
+        o3d.visualization.draw_geometries([raw], window_name="Raw camera-frame cloud")
+        o3d.visualization.draw_geometries([filtered], window_name="Grasp filtered cloud")
+
+
+if __name__ == "__main__":
+    main()
```

以上工具对输入 npy 要求浮点米；不能从浮点数本身识别“误把毫米保存成浮点”的错误，因此必须核对采集 depth_scale 和已知物体距离。反投影沿用原抓取针孔模型；若 SDK 当前彩色 profile 未校正畸变，需另行使用 SDK 反投影或先校正，不能把形状检查当几何标定通过。

### 离线验证 diff（由用户创建后运行）

```diff
--- /dev/null
+++ b/tests/test_d435i_pointcloud_contract.py
@@ -0,0 +1,40 @@
+"""Offline contract checks; no camera, ROS, CAN, or GUI."""
+from types import SimpleNamespace
+import numpy as np
+import pytest
+from tools.test_d435i_grasp_pointcloud import xyz_from_frame
+
+
+def frame():
+    return np.zeros((2, 3, 3), dtype=np.uint8), SimpleNamespace(fx=100., fy=100., cx=1., cy=0.)
+
+
+def test_known_metric_point_and_invalid_depth():
+    bgr, intr = frame()
+    depth = np.array([[0., 0.5, np.nan], [np.inf, 0.5, -1.]], dtype=np.float32)
+    xyz = xyz_from_frame(bgr, depth, intr)
+    assert xyz.shape == (2, 3, 3)
+    assert xyz.dtype == np.float32
+    np.testing.assert_allclose(xyz[0, 1], [0., 0., 0.5])
+    np.testing.assert_allclose(xyz[1, 1], [0., 0.005, 0.5])
+    for row, col in [(0, 0), (0, 2), (1, 0), (1, 2)]:
+        assert np.isnan(xyz[row, col]).all()
+
+
+def test_raw_integer_depth_cannot_be_mistaken_for_metres():
+    bgr, intr = frame()
+    with pytest.raises(ValueError, match="metres"):
+        xyz_from_frame(bgr, np.full((2, 3), 500, dtype=np.uint16), intr)
+
+
+def test_mismatched_sizes_are_rejected():
+    bgr, intr = frame()
+    with pytest.raises(ValueError, match="matching"):
+        xyz_from_frame(bgr, np.ones((3, 2), dtype=np.float32), intr)
+
+
+def test_bad_focal_length_is_rejected():
+    bgr, intr = frame()
+    intr.fx = 0.
+    with pytest.raises(ValueError, match="intrinsics"):
+        xyz_from_frame(bgr, np.ones((2, 3), dtype=np.float32), intr)
```

验证范围：确定米制输入对应已知几何结果；空洞/非有限值不成为有效点；尺寸错误、raw z16、错误内参被拒绝。这些断言不证明硬件对齐正确，也不验证机械臂执行。本文仅检查 diff 内 Python 语法，未创建脚本、未运行测试、未采集设备。

### 用户应用 diff 后的操作

以下命令在创建上述两个文件后才可用；目录名必须是新目录，避免覆盖上一组实验。

```bash
conda activate grasp_yolo26
python -m pytest tests/test_d435i_pointcloud_contract.py -q
python tools/test_d435i_grasp_pointcloud.py --output captures/d435i_scene_01 --view
python tools/test_d435i_grasp_pointcloud.py --output captures/d435i_object_01 --yolo-weights models/yolo26m-seg.pt --target-class cup --view
```

`cup` 仅是通用模型示例，按模型实际类别替换；food5 模型应传相应 visible 权重和真实类别。YOLO 会打印目标索引；多实例用 --target-index，索引只对应本次检测，不应当作跨帧身份。

采集器会打印含 color.png/depth_m.npy/capture_meta.json 的目录。把实际路径填入下面命令，可以用同一帧改变滤波参数，不必重新采相机：

```bash
python tools/test_d435i_grasp_pointcloud.py --capture-dir <实际RGB-D目录> --mask captures/d435i_object_01/selected_mask.png --output captures/d435i_replay_01 --voxel-size 0 --no-target-outlier-filter --view
```

应分别看 raw.ply 与 filtered.ply；两个窗口依次出现。目标过滤前后都保存，点数少于 3000 只提示重复采样，不直接判定抓取失败。当前工具没有网络推理，不把点云预览称作 GraspNet 成功。

最终抓取实验使用相同目标集、姿态/距离范围、权重、采样点数、规划/执行参数和成功定义（例如夹持确认并抬升保持）。分别记录成功次数/总次数及失败阶段；检测失败、深度缺失、IK 不可达、执行/夹爪失败不能全部归因于相机。若两相机分辨率/视角不同，应记录差异。无需额外质量评分报告。

### 启动/检查脚本完整替换 diff（SDK 主采集接线完成后再应用）

下列检查器不连接 CAN、不枚举相机。pyorbbecsdk 名称须与实际安装的厂商 SDK 一致；MediaPipe 留到切人手分支时验证。该检查器不是完整 GraspNet CUDA 扩展/TRAC-IK 验收，仍需后续离线推理与 IK 回归。

启动器直接使用已激活环境，去掉旧 ROS source 和强行拼接 PYTHONPATH；不任意清空 SDK/CUDA 必要的 LD_LIBRARY_PATH。应从没有 source ROS 的新终端激活 conda 运行，避免继承 ROS 路径。修改启动器本身不会让仍依赖 ROS 的 Python 采集路径自动变成 SDK。

```diff
--- a/scripts/run_grasp.sh
+++ b/scripts/run_grasp.sh
@@ -1,41 +1,8 @@
 #!/usr/bin/env bash
 set -euo pipefail
-
-PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
-CONDA_SH="/home/zgj/anaconda3/etc/profile.d/conda.sh"
-ROS_SETUP="/opt/ros/humble/setup.bash"
-NERO_SETUP="/home/zgj/Zgj/nero_handeye/install/setup.bash"
-USER_SITE="/home/zgj/.local/lib/python3.10/site-packages"
-
-unset PYTHONPATH
-unset LD_LIBRARY_PATH
+PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
+: "${CONDA_PREFIX:?Activate grasp_yolo26 before running this script}"
+PYTHON_BIN="$CONDA_PREFIX/bin/python"
 export PYTHONNOUSERSITE=1
-export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"
-export YOLO_CONFIG_DIR="${YOLO_CONFIG_DIR:-$PROJECT_DIR/.ultralytics}"
-mkdir -p "$MPLCONFIGDIR" "$YOLO_CONFIG_DIR/Ultralytics"
-
-source "$CONDA_SH"
-set +u
-conda activate grasp_yolo26
-set -u
-
-CONDA_SITE="$(python - <<'PY'
-import site
-print(site.getsitepackages()[0])
-PY
-)"
-CONDA_LIB="$CONDA_PREFIX/lib"
-
-set +u
-source "$ROS_SETUP"
-source "$NERO_SETUP"
-set -u
-
-ROS_PYTHONPATH="${PYTHONPATH-}"
-ROS_LD_LIBRARY_PATH="${LD_LIBRARY_PATH-}"
-
-export PYTHONPATH="$CONDA_SITE:$ROS_PYTHONPATH:$USER_SITE"
-export LD_LIBRARY_PATH="$CONDA_LIB:$ROS_LD_LIBRARY_PATH"
-
 cd "$PROJECT_DIR"
-exec python main.py "$@"
+exec "$PYTHON_BIN" main.py "$@"
--- a/scripts/check_env.sh
+++ b/scripts/check_env.sh
@@ -1,70 +1,28 @@
 #!/usr/bin/env bash
 set -euo pipefail
-
-PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
-# CONDA_SH="/home/zgj/anaconda3/etc/profile.d/conda.sh"
-ROS_SETUP="/opt/ros/humble/setup.bash"
-# NERO_SETUP="/home/zgj/Zgj/nero_handeye/install/setup.bash"
-# USER_SITE="/home/zgj/.local/lib/python3.10/site-packages"
-
-CONDA_SH="${CONDA_SH:-$(conda info --base 2>/dev/null)/etc/profile.d/conda.sh}"
-NERO_SETUP="${NERO_SETUP:-}"
-USER_SITE="${USER_SITE:-$(python -c 'import site; print(site.getusersitepackages())')}"
-
-unset PYTHONPATH
-unset LD_LIBRARY_PATH
+PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
+: "${CONDA_PREFIX:?Activate grasp_yolo26 before running this script}"
 export PYTHONNOUSERSITE=1
-
-source "$CONDA_SH"
-set +u
-conda activate grasp_yolo26
-set -u
-
-CONDA_SITE="$(python - <<'PY'
-import site
-print(site.getsitepackages()[0])
-PY
-)"
-CONDA_LIB="$CONDA_PREFIX/lib"
-
-set +u
-source "$ROS_SETUP"
-source "$NERO_SETUP"
-set -u
-
-ROS_PYTHONPATH="${PYTHONPATH-}"
-ROS_LD_LIBRARY_PATH="${LD_LIBRARY_PATH-}"
-
-export PYTHONPATH="$CONDA_SITE:$ROS_PYTHONPATH:$USER_SITE"
-export LD_LIBRARY_PATH="$CONDA_LIB:$ROS_LD_LIBRARY_PATH"
-
 cd "$PROJECT_DIR"
-python - <<'PY'
-import importlib.util
-import os
-import site
+"$CONDA_PREFIX/bin/python" - <<'PYENV'
+import importlib
 import sys
-
-print("== Real Nero Python Environment ==")
 print("python:", sys.executable)
-print("CONDA_PREFIX:", os.environ.get("CONDA_PREFIX", ""))
-print("PYTHONNOUSERSITE:", os.environ.get("PYTHONNOUSERSITE", ""))
-print("ENABLE_USER_SITE:", site.ENABLE_USER_SITE)
-print("PYTHONPATH entries:")
-for item in os.environ.get("PYTHONPATH", "").split(":"):
-    if item:
-        print(" ", item)
-print("LD_LIBRARY_PATH entries:")
-for item in os.environ.get("LD_LIBRARY_PATH", "").split(":"):
-    if item:
-        print(" ", item)
-print()
-
-for name in ["numpy", "scipy", "cv2", "can", "pyAgxArm", "rclpy", "sensor_msgs", "agx_arm_msgs"]:
+failed = []
+for name in ["numpy", "scipy", "cv2", "can", "pyAgxArm", "pyrealsense2",
+             "pyorbbecsdk", "open3d", "ultralytics", "torch", "pinocchio", "mujoco", "pytest"]:
     try:
-        mod = __import__(name)
-        print(f"{name}: version={getattr(mod, '__version__', '?')} file={getattr(mod, '__file__', '?')}")
+        module = importlib.import_module(name)
+        print(name, getattr(module, "__file__", None))
     except Exception as exc:
-        spec = importlib.util.find_spec(name)
-        print(f"{name}: import_error={type(exc).__name__}: {exc}; spec={None if spec is None else spec.origin}")
-PY
+        print(name, type(exc).__name__, str(exc))
+        failed.append(name)
+try:
+    import torch
+    print("torch CUDA:", torch.version.cuda, "available:", torch.cuda.is_available())
+    if not torch.cuda.is_available():
+        failed.append("CUDA")
+except ImportError:
+    pass
+raise SystemExit(1 if failed else 0)
+PYENV
```

### 验证工具取消 ROS 自动回退的精确 diff

仅改变 ImportError 回退；设备通信错误仍向上报错，不伪造位姿。完整 SDK 化时另移除 read-nero-tcp-ros 参数及其分支。

```diff
--- a/tools/verify_handeye_point.py
+++ b/tools/verify_handeye_point.py
@@ -650,8 +650,7 @@
                 args.nero_speed_percent,
             )
         except ImportError as e:
-            print(f"[warn] {e}; falling back to ROS PoseStamped topic {args.nero_ros_pose_topic}")
-            flange_pose = read_ros_pose_stamped(args.nero_ros_pose_topic, args.nero_ros_pose_timeout, "tcp_pose")
+            raise RuntimeError("NERO SDK unavailable; ROS fallback disabled") from e
     elif args.read_nero_tcp_ros:
         flange_pose = read_ros_pose_stamped(args.nero_ros_pose_topic, args.nero_ros_pose_timeout, "tcp_pose")
     else:
```


## 2026-09-15：全代码 ROS 迁移操作清单（以本节为准）

本节仅为手动修改说明，不修改 Python/SH 文件。替代前文不完整的架构 diff；前文的 sdk_camera 强行赋值、把 handeye_convention 强行改成 link7_cam、tools/test_D435i.py 命令模板均不可直接采用。当前真实测试脚本是 scripts/test_D435i.py。抓取 CLI 和坐标变换需显式适配，不能给现有 Namespace 填几个新字段就声称迁移完成。

### 先回答：有插值，还是每一帧实时规划？

抓取主链是规划阶段生成/验证路径，执行阶段跟踪保存的关节目标，不是每帧图像重做末端 IK。

- motion_planning.py:757 的 add_adaptive_transition 明确进行路径细分；786–789 行按 fraction 调 interpolate_pose_fn 生成中间位姿，随后 add_stage 求解/校验 IK。先尝试较少段数，失败才增加细分，并不是每段固定使用最大数量。
- motion_planning.py:231 对计划中的法兰目标求 IK 候选。
- execution.py:104 的 execute_prevalidated_stage 从实测起点重新验证到 stage.joints 的路径，向 move_flange_pose 传 planned_joint_target=stage.joints，不按相机帧实时改目标。
- arm_execution.py:460 发送 move_j(q_target)。底层固件的时间插值曲线不能从 Python 代码确认。
- 相机帧率、规划中间路点数量、SDK 反馈轮询频率是三个不同概念。多视角重拍会更新感知和后续计划，但不代表最终接近目标时逐帧视觉伺服。
- 后续 Servo 是 PBVS/雅可比/DLS 连续纠偏，200 Hz 阻抗内环；也不能把它描述成每张图跑完整 GraspNet+TRAC-IK 规划。

### A. 共享 SDK 模块必须先实现

新增 cameras/realsense_camera.py（建议名称，尚未创建）：定义单实例 start/read/close 和 Frame。start 只调用一次 pipeline.start；预热也放 try/finally 保护范围；超时有限次失败退出，不能用 30×10 秒吞掉异常。read 返回同一次 align frameset 的 bgr、depth_m、intr/K/D、sequence、采集时间与时钟域。采集与 CPU 推理解耦，UI 不被推理阻塞。close 幂等，异常/退出都释放设备。

由最外层会话管理相机生命周期，通过 hook/对象引用供初始采集和 active-view 重采集使用。读取时检查新帧和帧龄；到位后丢弃运动中的旧帧，再采稳定帧；多帧深度融合只能在相机静止期间进行。当前单帧测试通过不代表这个生产接口已实现。

### B. 正式抓取：不能漏掉间接调用

|当前文件和行号/函数锚点|要改成什么|关联修改|
|---|---|---|
|tools/real_nero_point_test.py:24 import capture_ros_rgbd_frame|SDK-only 模式删除此导入，采集改为共享 SDK 会话|保留 load_offline_rgbd 和非 ROS 数学函数|
|同文件:224 capture_rgbd / :228 return capture_ros_rgbd_frame|离线分支保持；实时分支 read 共享相机，返回 bgr,depth_m,intr,out_dir|保留保存路径和原深度融合所需参数语义；不要每次开关 pipeline|
|同文件:285 capture_ros_color_pointcloud_frame / :514 调用|SDK-only 路径取消 ROS PointCloud2 消费，改从 depth_m 反投影；同步删除/改写调用处|不能返回 ndarray 却继续当 cloud_msg 解码|
|real_nero_grasp/runtime.py:172 import / :1118 hook 传递|移除旧 capture_ros_color_pointcloud_frame 注入，改成 SDK 采集契约|runtime 显式传 hook，单改 perception 的默认值不生效|
|real_nero_grasp/perception.py:1490–1517 默认采集及 capture_rgbd_xyz_frame|默认与注入路径统一读 SDK；保留 bgr,xyz_img,intr,out_dir 输出|迁移调用方和测试一起改签名，不能留半套 ROS hook|
|同文件 capture_rgbd_xyz_frame 中 use_registered_points 分支|当前 D435i 模式从对齐 depth 生成 XYZ；旧 --use-registered-points 应显式拒绝或映射到实现完整的 SDK XYZ 模式|不要默默进入 ROS，也不要把不兼容消息结构传下游|
|同文件:1664–1667 重试失败提示|换成相机型号/序列号、profile、帧龄、SDK 错误信息|不再指导用户运行 ros2 topic 命令|
|real_nero_grasp/cli.py 相机参数组|新增 --camera-backend realsense、--camera-serial、color/depth profile、timeout/warmup 参数；旧 orbbec topic 参数退出当前主路径|把实际仍使用的滤波/融合参数拆出来保留，不能按名字包含 orbbec 就全删|
|real_nero_grasp/active_view.py / active_motion.py 及 workflow hook|追踪所有 recapture，统一使用会话相机；位姿对应所采帧|args 复制时不能隐式复制 pipeline 或新建连接；先检查 deepcopy 等参数复制行为|
|real_nero_grasp/depth_processing.py pointcloud2_xyz_image|无 ROS SDK 主路径不再调用消息解析；保留 depth_to_xyz_image 等纯数组算法|兼容解析如保留，隔离到可选适配模块，不要求删除纯函数才能运行|

所有 topic 参数的具体行号可能随用户编辑改变，用参数名/函数名定位。替换这些地方不是删掉 import 就完成；入口、默认分支、hook、重试、测试和退出都要一起接通。

### C. 标定和辅助工具（之前遗漏的调用也纳入）

|文件/当前行号|修改要求|
|---|---|
|tools/verify_handeye_point.py:240–408 capture_ros_rgbd_frame|将 ROS Image/CameraInfo/QoS/init/spin/shutdown 整个采集实现替换为共享 SDK 采集，不保留主路径对 rclpy 的要求|
|同文件:146 capture_realsense_frame|现有 SDK 单帧功能可复用数据转换，但正式连续采集使用共享会话；保存 D、序列号、时间戳和 profile|
|同文件:458–497 read_ros_pose_stamped / :657 分支|SDK-only 版删除 ROS 位姿读取及 --read-nero-tcp-ros 选项，SDK 失败直接报错，保留手动 --flange-pose 路径|
|同文件:616 live-orbbec 分支|当前仅 D435i，改用实时 SDK 相机选择；330 分支先明确未启用，不偷换成 D435i|
|tools/record_nero_observation_pose_seeds.py:50–52|删除 capture_ros_rgbd_frame 的局部导入和调用，保存图像时读共享 D435i SDK 帧；此处是间接 ROS 依赖|
|tools/verify_handeye_aruco_consistency.py:25、181|同样替换采集函数导入和实际调用，保留 ArUco 一致性计算|
|tools/nero_handeye_calib_pose_sweep.py:53–104 _HandeyePosePublisher|移除 rclpy/geometry_msgs Pose 发布器；用本地样本记录器替代，记录 SDK 位姿+板观测|
|同文件:1183 实例化 / :1406 参数|取消发布器创建/关闭、pose topic/hz 参数，改成 sample-dir/采样触发；其他通过 capture_rgbd 间接取图的调用统一 SDK|

注意：旧 verify_handeye_point 存在内置 DEFAULT_CALIB_POS/QUAT，不应拿它当 D435i 标定。实机标定验证必须显式选择已核验的 D435i JSON。pose sweep 是运动程序，替换记录方式不能绕过 --execute 与原安全限制。

### D. 递送主程序的 ROS 全部出口

servo_delivery.py：

1. 第16–18行删除 rclpy、Node、CameraInfo/Image 导入；引入共享 SDK Frame 和采集会话。
2. 第88行 image_msg_to_bgr 不再作为 SDK 输入路径。SDK 明确返回 uint8 BGR，不能继续访问 msg.data/height/encoding。
3. ServoDeliveryNode 类声明和 super().__init__ 替换为普通控制器初始化，模型/后端/相机从外层传入。
4. 第178行附近两处 create_subscription 与 create_timer 整块替换为外层采集+单线程 step 调度。图像与深度来自同一帧，不再分别订阅。
5. _on_image/_on_info/_detect_latest 改成消费 Frame.bgr/depth_m/K/D/时间；移除 .k/.d 等 ROS 消息字段。MediaPipe 使用真实 depth_m，不再读取未赋值的 _latest_depth。
6. 所有 get_logger() 改为标准 logging；warn 改为 warning；不改故障触发条件和停止动作的含义。
7. 第345行 _shutdown 内 rclpy.shutdown 改为设置停止事件/退出状态；控制器不直接销毁外层持有的资源。
8. 第476–488行 init/spin/shutdown 替换为 try/finally 保护下的会话循环。backend 创建、相机启动、模型/标定初始化均要被失败清理覆盖。
9. 每张新帧最多计一次稳定计数；无新帧/超时保持或失败，不把旧目标标记为当前观测；关节反馈和曝光时间须匹配。
10. 调度重构后仍保持 MIT 独立内环，不在图像回调内跑 200 Hz 运动控制。

### E. Visual Servo 复制脚本、测试及路径注入

|文件/位置|要做的替换|
|---|---|
|scripts/aruco_camera_jacobian_servo_sdk_dryrun.py:44–45、51–54|删除 /opt/ros 手工 sys.path 注入和 ROS imports；类 Node、订阅、spin 同 D 节迁移；仅机器人侧原本是 SDK|
|同文件:359、474、566–576|把全部 rclpy 退出改为统一停止事件/清理，不能只改 main 最末的 shutdown|
|scripts/aruco_camera_jacobian_servo_sdk_execute.py:45–46、52–55|同样移除硬编码 ROS Python 搜索路径与 imports；采集改共享 SDK|
|同文件所有 rclpy.shutdown 分支（当前754–1297、2141、2605–2625等）|逐分支保留故障原因、急停/保持动作，再设退出状态；不是简单批量替换为空操作|
|scripts/aruco_jacobian_servo_ros_dryrun.py:43–53、158–159|若继续使用该诊断工具，JointState→SDK反馈、PoseStamped目标→检测/回放目标，Node/timer/spin 同步替换；若不使用则明确列为 legacy 非 SDK 入口|
|tests_servo/test_image_conversion.py:6、10|SDK 图像契约测试直接用 ndarray；如保留 ROS 消息兼容测试则隔离为可选测试，主 tests_servo 不因 import sensor_msgs 失败|
|tests_servo/test_execute_impedance_engage.py 的 execute_module 导入|被测数学/engage 逻辑抽到无 ROS 模块，更新 import；仍验证原逻辑，不能删除断言来让测试过|
|control_backends/ros_move_j_backend.py|可选占位后端不接主链；不需要改成第二套 SDK 后端，已有 SDK 实现。若目标是仓库完全无 ROS，可归档；若目标是运行无 ROS，可保留为 legacy|
|scripts/run_grasp.sh、check_env.sh、launch_orbbec.sh|按前述启动器修改说明；当前 330 不用，launch_orbbec 不从正式路径调用。注释中的旧 ROS 字符串与真实导入区分|

当前未发现 targets/、models/、servo/ 数学模块直接依赖 ROS；不改它们的数学计算。SDK 后端 pyAgxArm 本身不替换成 rospy。tests/、配置、README/STEP1/2 的旧命令也要在最终迁移时同步标注，不能让文档仍指向 ros2 topic/launch。

### 完成判据（不是只看 SH 启动成功）

- 无 ROS 环境能导入主入口、SDK 采集、handoff、递送及所有当前启用测试。
- 所有实时采集调用路径都能追溯到同一个 D435i SDK 会话，初始采集和多视角重拍没有 ROS 回退。
- 主程序遇到 SDK/标定错误明确退出并清理，不打印“source Humble”作为解决方法。
- 回归测试维持原规划/碰撞/控制语义；录制帧回放通过；再空载验证真实采集与位姿对应。
- 多视角真实拍摄必须先有正确 D435i 标定和受控观察运动；不能说“不连接机械臂就已验收真实多视角”。离线回放是另一层验收。
- 库名全局检索只是清单核对，还要检索 capture_ros/read_ros、Node/get_logger、topic 参数、runtime hooks 和 /opt/ros 的间接使用。

本节是逐位置实施要求，不是已完成的可执行实现。共享 SDK 模块、配置生命周期和所有调用签名必须一起设计和验证；不能把前文带省略号的 diff 直接粘贴到生产代码。

## 2026-09-15 已实施 SDK 改造：实际结果（优先阅读）

**本轮已直接修改代码。不能据此立即上实机：当前回归未全部通过、CUDA 不可用、TRAC-IK 缺失。**

### 当前实际架构

- Gemini 330 本阶段关闭。D435/D435IF SDK 是当前实时相机输入。
- cameras/realsense.py：懒加载 SDK、按序列号/profile 建立持久会话；初始采集、多视角重采集复用会话，浅拷贝 args 不复制相机；有界超时、预热异常释放、退出 close_all。对齐 color、米制深度、序号去重、非零畸变拒绝，保留深度多帧中位数/MAD 处理。
- 正式抓取保持原四元组/XYZ hook：capture_rgbd 返回 SDK 数据；历史 capture_ros_* 名称只作为兼容函数名存在，函数体不再使用 ROS。注册点云模式返回本地数组封装，原纯数组 PointCloud2 解码可继续使用，不需要 sensor_msgs。
- cameras/sdk_loop.py：直接 SDK 采集线程 + 本地定时回调；Image/CameraInfo 是普通 Python 数据对象，topic 字符串仅为历史标签，无 ROS 通信。保留 Node/create_subscription 形式是为了复用已经测试的图像转换与阻抗控制器，不能把这些方法名理解成仍运行 ROS。
- servo_delivery 与两个 ArUco SDK 参考脚本均已改用本地 SDK host。MediaPipe 深度来自同一 SDK 对齐帧；递送目标检测按新帧序号去重，过期目标清零前馈与跟随弹簧；修复 hand_target 中 ndarray 与 None 比较。
- 初始化、错误、退出路径加入设备回收；handoff 按用户顺序停 move_j→MIT接管→断旧驱动，停止确认后才抑制旧驱动退出急停，检查 SDK 反馈时间戳并传递通道/固件。此处验证的是 SDK 聚合反馈时间戳，不等于已验证每个关节 CAN 包独立新鲜度，必须空载验收。
- 递送增加开爪宽度反馈确认/超时；释放阶段不再依赖继续检测目标才能撤离；固定回程目标取消 PBVS 人手距离偏置并经过安全判断。保持接管、释放和回程尚未实机验证。
- 标定 pose publisher 改为本地 JSONL 位姿记录，不再依赖外部 ROS 节点。但这不是新建完整手眼求解器；标定采集/求解和挂载验证仍待完成。
- 可选 ROS joint/target 诊断脚本退役并给出 SDK 替代入口。原 test_zero_gravity_diag.py 有缺失 MIT 调用的语法残片，已显式禁用，未猜测补发任何电机命令。
- 修复 _workflow_hooks 未定义 args、run_grasp_safe 调用不存在的 run_real_nero_clean.sh、工具拷贝丢失加号等问题。
- 330 固定自遮挡图像 mask 默认关闭，安全启动器同步关闭旧 mask；不能未经校准把 330 遮罩套到 D435i。真实相机碰撞几何还需按实际安装核实。

### 新配置/参数

`--config config/pipeline.yaml` 显式加载 SDK 配置；显式 CLI 覆盖 YAML；标定相对路径按 YAML 所在目录解析。未传 --config 则使用 CLI 默认，不再假定修改 YAML 自动生效。

相机参数：--camera-serial、--camera-width、--camera-height、--camera-fps、--camera-timeout、--camera-warmup。

抓取：--calib-json 必需，仍使用 --handeye-convention 的 flange_camera_* 约定。不得把 link7_cam 字符串强行塞进抓取公式。

伺服：--handeye 必需；使用 --wrist-handeye-convention、--wrist-handeye-mount-frame，避免主 CLI 与抓取同名参数冲突。主入口已注册 Servo 参数，包括 --target-source。

当前 YAML 中两个 wrist 标定路径均为 null，必须填真实 D435 标定；不是要求重新启用 330。抓取 flange 和 Servo feedback_link7 的定义未核实前不要共用文件。

### 本轮验证与实机阻塞

- main.py --help：通过。
- runtime、handoff、servo_delivery 导入：通过，不需要 ROS。
- AST 检查当前运行模块和测试：无 rclpy/rospy/sensor_msgs/geometry_msgs/agx_arm_msgs/cv_bridge 直接导入；Python 语法通过，4 个启动 SH 的 bash -n 通过。
- 完整 tests + tests_servo：**514 passed，5 failed，10 subtests passed**。
- 新增 SDK 会话/米制点云/预热失败清理/本地回调/YAML覆盖/交接旧反馈测试：6 passed，不访问 USB/CAN。
- 3 个 grasp_stage_beam_search 失败触发缺失 trac_ik；2 个 Servo 失败分别是 target_twist 限幅与测试期望不一致、运动目标时库仑补偿仍产生力矩。
- 用改动前源码快照复测相关测试组，得到同样 5 项失败；为导入旧伺服脚本仅替换 ROS import 为本地 SDK host，未改原控制数学。没有删除或放宽这些断言，也没有根据断言擅自改变已有 MIT 力矩律。
- check_env.sh 失败项：trac_ik、CUDA。实际 torch CUDA 构建为13.0，cuda.is_available=False；检查器返回非零。真实 main 抓取 --execute 在连接机械臂前检查标定、权重、CUDA、IK和GraspNet导入。
- 未执行新的 USB 采图、真实多视角、MIT交接、夹爪或机械臂运动：按逐级验收要求，当前回归未通过，未进入硬件级测试。

### 仍需完成，不能宣称整机已经可运行

1. 安装并验证项目实际使用的 trac_ik Python 实现及其 native 依赖；不要换同名不兼容 API。
2. 匹配 PyTorch/CUDA 与 GPU 驱动，验证 pointnet2/knn/graspnet 扩展和实际推理。此轮没有安装/更换系统驱动或 conda 包。
3. 分析并解决两项已有阻抗回归差异，明确速度前馈和库仑项的正确控制要求。
4. 完成 D435 腕部手眼标定、flange/link7/TCP 对应关系与真实挂载几何验证；当前时间元数据区分硬件毫秒和主机接收 monotonic，但尚未实现曝光时刻与关节的硬件同步/插值。抓取仅应在确认静止后采集；伺服需低速验证时间偏差。
5. 实际载荷和固件下验证持位接管、每关节反馈新鲜度、停止与夹爪释放；SDK 后端参数不能保证“绝不会掉高”。
6. 再按既定顺序跑离线推理、空载感知/多视角、抓取、空载伺服和全流程。

### 修改文件与实际 unified diff

以下从本轮修改前快照自动生成，是已实施差异，不是让用户重新执行的终端命令。旧方案片段保留为历史，以本节实现和测试状态为准。

- `cameras/__init__.py`
- `cameras/options.py`
- `cameras/preflight.py`
- `cameras/realsense.py`
- `cameras/sdk_loop.py`
- `config/pipeline.yaml`
- `handoff/grasp_to_servo_handoff.py`
- `real_nero_grasp/cli.py`
- `real_nero_grasp/perception.py`
- `real_nero_grasp/runtime.py`
- `real_nero_grasp/servo_cli.py`
- `scripts/aruco_camera_jacobian_servo_sdk_dryrun.py`
- `scripts/aruco_camera_jacobian_servo_sdk_execute.py`
- `scripts/aruco_jacobian_servo_ros_dryrun.py`
- `scripts/check_env.sh`
- `scripts/launch_orbbec.sh`
- `scripts/run_grasp.sh`
- `scripts/run_grasp_safe.sh`
- `scripts/test_zero_gravity_diag.py`
- `servo_delivery.py`
- `targets/hand_target.py`
- `tests/test_sdk_camera_session.py`
- `tests_servo/test_image_conversion.py`
- `tools/nero_handeye_calib_pose_sweep.py`
- `tools/real_nero_point_test.py`
- `tools/record_nero_observation_pose_seeds.py`
- `tools/test_d435i_grasp_pointcloud.py`
- `tools/verify_handeye_aruco_consistency.py`
- `tools/verify_handeye_point.py`

#### cameras/__init__.py

```diff
--- /dev/null
+++ b/cameras/__init__.py
@@ -0,0 +1 @@
+"""Direct camera SDK adapters. Importing this package does not access USB."""

```

#### cameras/options.py

```diff
--- /dev/null
+++ b/cameras/options.py
@@ -0,0 +1,44 @@
+"""Shared SDK profile and YAML options; CLI overrides YAML defaults."""
+import argparse
+from pathlib import Path
+
+
+def add_camera_arguments(p):
+    p.add_argument('--config', default=None, help='Optional SDK pipeline YAML; CLI overrides YAML')
+    p.add_argument('--camera-serial', default=None)
+    p.add_argument('--camera-width', type=int, default=640)
+    p.add_argument('--camera-height', type=int, default=480)
+    p.add_argument('--camera-fps', type=int, default=30)
+    p.add_argument('--camera-timeout', type=float, default=10.)
+    p.add_argument('--camera-warmup', type=int, default=30)
+
+
+def apply_pipeline_defaults(p, argv=None):
+    probe=argparse.ArgumentParser(add_help=False)
+    probe.add_argument('--config')
+    opts,_=probe.parse_known_args(argv)
+    if not opts.config: return
+    import yaml
+    path=Path(opts.config).expanduser().resolve()
+    data=yaml.safe_load(path.read_text()) or {}
+    if data.get('global_camera',{}).get('enabled',False):
+        p.error('Gemini 330 is not supported in this D435-only SDK phase')
+    wrist=data.get('wrist_camera',{})
+    if not wrist.get('enabled',True) or wrist.get('backend','realsense')!='realsense':
+        p.error('D435 wrist SDK must be enabled')
+    defaults={}
+    for key in ('serial','width','height','fps','timeout','warmup'):
+        if key in wrist: defaults['camera_'+key]=wrist[key]
+    mapping={'grasp_calib_json':'calib_json','grasp_convention':'handeye_convention',
+             'servo_calib_json':'handeye','servo_convention':'wrist_handeye_convention',
+             'servo_mount_frame':'wrist_handeye_mount_frame'}
+    for key,dest in mapping.items():
+        if wrist.get(key) is not None:
+            value=wrist[key]
+            if key.endswith('calib_json'):
+                value=str((path.parent/Path(value).expanduser()).resolve())
+            defaults[dest]=value
+    for key in ('target_source','max_linear_vel','max_joint_vel'):
+        if key in data.get('servo',{}): defaults[key]=data['servo'][key]
+    known={a.dest for a in p._actions}
+    p.set_defaults(**{k:v for k,v in defaults.items() if k in known})

```

#### cameras/preflight.py

```diff
--- /dev/null
+++ b/cameras/preflight.py
@@ -0,0 +1,26 @@
+"""Read-only preflight performed before any SDK arm connection."""
+from pathlib import Path
+import numpy as np
+
+
+def grasp_preflight(args):
+    if not getattr(args, 'execute', False):
+        return
+    from tools.real_nero_point_test import handeye_transform
+    transform = handeye_transform(args.calib_json, args.handeye_convention)
+    if not np.isfinite(transform).all():
+        raise ValueError('Non-finite wrist calibration')
+    for path in (args.yolo_weights, args.ik_urdf):
+        if not Path(path).is_file():
+            raise FileNotFoundError(path)
+    root = Path(__file__).resolve().parents[1]
+    checkpoint = root/'logs/log_rs/checkpoint-rs.tar'
+    if not checkpoint.is_file() or checkpoint.stat().st_size < 1000:
+        raise FileNotFoundError(f'Missing real GraspNet checkpoint: {checkpoint}')
+    import torch
+    if not torch.cuda.is_available():
+        raise RuntimeError('GraspNet requires CUDA. Fix PyTorch/driver compatibility before --execute.')
+    from real_nero_grasp.ik_solver import _get_ik_solver
+    _get_ik_solver(args)
+    from real_nero_grasp.visualization import get_graspnet_helpers
+    get_graspnet_helpers()

```

#### cameras/realsense.py

```diff
--- /dev/null
+++ b/cameras/realsense.py
@@ -0,0 +1,176 @@
+"""Persistent D435-family RGB-D capture; SDK imported only on start."""
+import atexit
+import json
+import threading
+import time
+from dataclasses import dataclass
+from pathlib import Path
+from types import SimpleNamespace
+import numpy as np
+
+
+@dataclass
+class Frame:
+    bgr: np.ndarray
+    depth_m: np.ndarray
+    intr: object
+    K: np.ndarray
+    D: np.ndarray
+    sequence: int
+    hardware_ms: float
+    received_monotonic: float
+    serial: str
+
+
+class RealSenseCamera:
+    def __init__(self, serial=None, width=640, height=480, fps=30, timeout=10., warmup=30):
+        if min(width, height, fps) <= 0 or timeout <= 0 or warmup < 0:
+            raise ValueError('Invalid camera profile/timeout/warmup')
+        self.serial, self.width, self.height, self.fps = serial, width, height, fps
+        self.timeout, self.warmup = timeout, warmup
+        self.pipeline = None
+        self.last_sequence = None
+        self.lock = threading.RLock()
+
+    def start(self):
+        with self.lock:
+            if self.pipeline is not None:
+                return self
+            import pyrealsense2 as rs
+            ctx = rs.context()
+            devices = [d for d in ctx.query_devices()
+                       if '435' in d.get_info(rs.camera_info.name).upper()
+                       and (not self.serial or d.get_info(rs.camera_info.serial_number) == self.serial)]
+            if len(devices) != 1:
+                raise RuntimeError(f'Expected one selected D435-family camera; found {len(devices)}')
+            self.serial = devices[0].get_info(rs.camera_info.serial_number)
+            config = rs.config()
+            config.enable_device(self.serial)
+            config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
+            config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
+            pipe = rs.pipeline(ctx)
+            profile = pipe.start(config)
+            self.pipeline = pipe
+            try:
+                self.scale = profile.get_device().first_depth_sensor().get_depth_scale()
+                self.align = rs.align(rs.stream.color)
+                for _ in range(self.warmup):
+                    pipe.wait_for_frames(int(self.timeout * 1000))
+            except BaseException:
+                self.close()
+                raise
+            return self
+
+    def read(self):
+        with self.lock:
+            self.start()
+            deadline = time.monotonic() + self.timeout
+            while time.monotonic() < deadline:
+                frames = self.align.process(self.pipeline.wait_for_frames(
+                    max(1, int((deadline-time.monotonic())*1000))))
+                color, depth = frames.get_color_frame(), frames.get_depth_frame()
+                if not color or not depth:
+                    continue
+                seq = depth.get_frame_number()
+                if seq == self.last_sequence:
+                    continue
+                ci = color.profile.as_video_stream_profile().intrinsics
+                # Existing grasp code uses pinhole back-projection. Reject unsupported raw distortion.
+                if np.any(np.abs(ci.coeffs) > 1e-9):
+                    raise RuntimeError('Nonzero color distortion: rectify before pinhole back-projection')
+                bgr = np.asanyarray(color.get_data()).copy()
+                z = np.asanyarray(depth.get_data()).astype(np.float32) * float(self.scale)
+                if bgr.shape[:2] != z.shape:
+                    raise RuntimeError('SDK color/depth alignment shape mismatch')
+                self.last_sequence = seq
+                intr = SimpleNamespace(fx=ci.fx, fy=ci.fy, cx=ci.ppx, cy=ci.ppy)
+                return Frame(bgr, z, intr, np.array([[ci.fx,0,ci.ppx],[0,ci.fy,ci.ppy],[0,0,1.]]),
+                             np.asarray(ci.coeffs), seq, depth.get_timestamp(), time.monotonic(), self.serial)
+            raise TimeoutError('No new aligned SDK frame')
+
+    def close(self):
+        with self.lock:
+            pipe, self.pipeline = self.pipeline, None
+            self.last_sequence = None
+            if pipe is not None:
+                pipe.stop()
+
+    def __enter__(self): return self.start()
+    def __exit__(self, *exc): self.close()
+
+
+_sessions = {}
+
+def camera_for(args):
+    settings = (getattr(args, 'camera_serial', None), getattr(args, 'camera_width', 640),
+                getattr(args, 'camera_height', 480), getattr(args, 'camera_fps', 30),
+                getattr(args, 'camera_timeout', 10.), getattr(args, 'camera_warmup', 30))
+    if settings not in _sessions:
+        _sessions[settings] = RealSenseCamera(*settings)
+    return _sessions[settings]
+
+
+def close_all():
+    sessions = list(_sessions.values())
+    _sessions.clear()
+    errors = []
+    for camera in sessions:
+        try:
+            camera.close()
+        except Exception as exc:
+            errors.append(exc)
+    if errors:
+        raise RuntimeError('Failed to close camera session') from errors[0]
+
+
+atexit.register(close_all)
+
+
+def capture_rgbd(args):
+    from tools.verify_handeye_point import median_valid_depth_frames
+    camera = camera_for(args)
+    # Discard queued frame after robot motion; next read must have a new sequence.
+    camera.read()
+    count = max(1, int(getattr(args, 'orbbec_depth_median_frames', 1)))
+    frames = [camera.read() for _ in range(count)]
+    frame = frames[-1]
+    depth, _ = median_valid_depth_frames(
+        [f.depth_m for f in frames],
+        min_valid_frames=getattr(args, 'orbbec_depth_median_min_valid', 1),
+        max_mad=getattr(args, 'orbbec_depth_median_max_mad', None))
+    import cv2
+    from datetime import datetime
+    out = Path(getattr(args, 'save_dir', None) or 'captures') / datetime.now().strftime('%Y-%m-%d_%H-%M-%S_%f')
+    out.mkdir(parents=True, exist_ok=False)
+    if not cv2.imwrite(str(out/'color.png'), frame.bgr):
+        raise OSError('Cannot save RGB capture')
+    np.save(out/'depth_m.npy', depth)
+    (out/'capture_meta.json').write_text(json.dumps(dict(
+        intrinsics=vars(frame.intr), width=frame.bgr.shape[1], height=frame.bgr.shape[0],
+        serial=frame.serial, K=frame.K.tolist(), D=frame.D.tolist(),
+        sequence=frame.sequence, hardware_ms=frame.hardware_ms,
+        received_monotonic=frame.received_monotonic, depth_unit='metre',
+        temporal_frames=count), indent=2))
+    return frame.bgr, depth, frame.intr, str(out)
+
+
+def capture_legacy(save_dir, color_topic=None, depth_topic=None, camera_info_topic=None,
+                   depth_scale=None, timeout_s=10., warmup_frames=30,
+                   depth_median_frames=1, depth_median_min_valid=1, depth_median_max_mad=None):
+    """Old tools' positional signature; topic arguments do not select a transport."""
+    return capture_rgbd(SimpleNamespace(save_dir=save_dir, camera_timeout=timeout_s,
+        camera_warmup=warmup_frames, orbbec_depth_median_frames=depth_median_frames,
+        orbbec_depth_median_min_valid=depth_median_min_valid,
+        orbbec_depth_median_max_mad=depth_median_max_mad))
+
+
+def organized_cloud(args):
+    """Array-backed organized XYZ envelope for legacy pure-array consumers, not ROS."""
+    from real_nero_grasp.depth_processing import depth_to_xyz_image
+    bgr, depth, intr, out = capture_rgbd(args)
+    xyz = np.ascontiguousarray(depth_to_xyz_image(depth, intr), dtype='<f4')
+    h, w = depth.shape
+    msg = SimpleNamespace(height=h, width=w, data=xyz.tobytes(), is_bigendian=False,
+        point_step=12, row_step=12*w, fields=[SimpleNamespace(name=n, offset=i*4, datatype=7, count=1)
+        for i,n in enumerate(('x','y','z'))], header=SimpleNamespace(frame_id='d435_color_optical_frame'))
+    return bgr, msg, intr, out

```

#### cameras/sdk_loop.py

```diff
--- /dev/null
+++ b/cameras/sdk_loop.py
@@ -0,0 +1,101 @@
+"""Local SDK callback host for ported camera controllers; no ROS middleware.
+
+Image/CameraInfo are plain in-process envelopes preserving tested conversion code.
+Topic strings are descriptive labels only, not network endpoints.
+"""
+import logging
+import threading
+import time
+from dataclasses import dataclass, field
+from types import SimpleNamespace
+from .realsense import camera_for
+
+@dataclass
+class Image:
+    width: int = 0
+    height: int = 0
+    encoding: str = 'bgr8'
+    step: int = 0
+    data: bytes = b''
+    header: object = None
+
+@dataclass
+class CameraInfo:
+    k: list = field(default_factory=list)
+    d: list = field(default_factory=list)
+
+_running = False
+
+def init(args=None):
+    global _running
+    _running = True
+
+def shutdown():
+    global _running
+    _running = False
+
+def ok(): return _running
+
+
+class Node:
+    def __init__(self, name):
+        self._logger = logging.getLogger(name)
+        self._callbacks = []
+        self._timers = []
+    def get_logger(self): return self._logger
+    def create_subscription(self, kind, label, callback, *args, **kwargs):
+        if kind not in (Image, CameraInfo):
+            raise ValueError('SDK host accepts image/calibration callbacks only')
+        self._callbacks.append((kind, label, callback))
+    def create_timer(self, period, callback):
+        if period <= 0: raise ValueError('Timer period must be positive')
+        self._timers.append([period, callback, time.monotonic()])
+    def destroy_node(self): pass
+
+
+def spin(node):
+    camera = camera_for(node.args)
+    stop = threading.Event()
+    latest = [None]
+    failure = []
+    def produce():
+        try:
+            while not stop.is_set():
+                latest[0] = camera.read()
+        except Exception as exc:
+            failure.append(exc)
+    worker = threading.Thread(target=produce, name='d435-sdk-capture', daemon=True)
+    worker.start()
+    last = None
+    try:
+        while ok():
+            if failure: raise RuntimeError('SDK camera stream failed') from failure[0]
+            frame = latest[0]
+            if frame is not None and frame.sequence != last:
+                last = frame.sequence
+                if time.monotonic()-frame.received_monotonic > .45:
+                    raise TimeoutError('SDK frame stale')
+                # This stamp is host receipt time, NOT hardware exposure wall time.
+                now = time.time(); sec=int(now)
+                stamp=SimpleNamespace(sec=sec, nanosec=int((now-sec)*1e9))
+                image=Image(frame.bgr.shape[1], frame.bgr.shape[0], 'bgr8', frame.bgr.shape[1]*3,
+                            frame.bgr.tobytes(), SimpleNamespace(stamp=stamp))
+                info=CameraInfo(frame.K.reshape(-1).tolist(), frame.D.tolist())
+                node._latest_depth=frame.depth_m
+                node._sdk_frame_sequence=frame.sequence
+                node._sdk_received_monotonic=frame.received_monotonic
+                for kind, label, callback in node._callbacks:
+                    callback(info if kind is CameraInfo else image)
+            now = time.monotonic()
+            for timer in node._timers:
+                if now >= timer[2]:
+                    timer[2] = now+timer[0]
+                    timer[1]()
+                    if not ok(): break
+            stop.wait(.005)
+    finally:
+        stop.set()
+        worker.join(timeout=camera.timeout+2.)
+        if worker.is_alive():
+            raise RuntimeError('SDK capture worker did not exit')
+        camera.close()

```

#### config/pipeline.yaml

```diff
--- a/config/pipeline.yaml
+++ b/config/pipeline.yaml
@@ -1,22 +1,24 @@
-# 整个流程的相机配置。腕部相机沿用伺服工程已验证的手眼标定（link7_cam 约定）。
-global_camera:                    # 抓取阶段（第一步）
-  image_topic: /camera/color/image_raw
-  info_topic: /camera/color/camera_info
-  depth_topic: /camera/depth/image_raw
-  calib_json: /path/to/你的全局相机标定.json     # 抓取工程用的那套（flange_camera_xyzw）
-  convention: flange_camera_xyzw
-
-wrist_camera:                     # 伺服递送阶段（第二步）
-  image_topic: /camera/camera/color/image_raw
-  info_topic: /camera/camera/color/camera_info
-  depth_topic: /camera/camera/aligned_depth_to_color/image_raw   # mediapipe 人手目标用
-  calib_json: /home/dpf/桌面/RealRobot_Test/real_test/configs/2026-05-19_14-45-49_calibration.json
-  convention: link7_cam           # ← 伺服工程已验证的约定
-  mount_frame: feedback_link7
-  offset_base: [0.02, -0.025, 0]
-
+# SDK-only D435 wrist phase. CLI values override these defaults.
+# T_A_B maps B-coordinate points into A coordinates.
+global_camera:
+  enabled: false
+  model: Gemini330
+wrist_camera:
+  enabled: true
+  backend: realsense
+  serial: null
+  width: 640
+  height: 480
+  fps: 30
+  timeout: 10.0
+  warmup: 30
+  # Grasp uses SDK flange; servo FK uses feedback_link7. Supply independently verified transforms.
+  grasp_calib_json: null
+  grasp_convention: flange_camera_xyzw
+  servo_calib_json: null
+  servo_convention: link7_cam
+  servo_mount_frame: feedback_link7
 servo:
-  target_source: aruco            # v1: aruco；v2: mediapipe
+  target_source: aruco
   max_linear_vel: 0.03
   max_joint_vel: 0.06
-  handover_distance: 0.10

```

#### handoff/grasp_to_servo_handoff.py

```diff
--- a/handoff/grasp_to_servo_handoff.py
+++ b/handoff/grasp_to_servo_handoff.py
@@ -30,18 +30,28 @@
     fallback = np.asarray(arm.get_joint_angles(), dtype=float).reshape(7)
     arm.terminate_motion(fallback_joints=fallback)
     # 阻止稍后的 __exit__ 再次触发程序退出急停（工程内 session_control 同款用法）
-    arm._suppress_program_exit_estop = True
-    arm._controlled_stop_request_confirmed = True
 
-    confirmed, last = 0, None
+    confirmed, last, last_stamp = 0, None, None
     deadline = time.time() + timeout_s
     while time.time() < deadline:
-        q = np.asarray(arm.get_joint_angles(), dtype=float).reshape(7)
+        feedback = arm.arm.get_joint_angles()
+        stamp = float(getattr(feedback, 'timestamp', 0.0))
+        if not np.isfinite(stamp) or not 0 <= time.time() - stamp <= .5:
+            return False
+        if stamp == last_stamp:
+            time.sleep(.02)
+            continue
+        last_stamp = stamp
+        q = np.asarray(feedback.msg, dtype=float).reshape(7)
+        if not np.isfinite(q).all():
+            return False
         if last is not None:
             delta = float(np.max(np.abs(np.rad2deg(q - last))))
             if delta <= settle_delta_deg:
                 confirmed += 1
                 if confirmed >= confirm_samples:
+                    arm._suppress_program_exit_estop = True
+                    arm._controlled_stop_request_confirmed = True
                     logger.info("[handoff] move_j trajectory confirmed stopped")
                     return True
             else:
@@ -85,31 +95,45 @@
         pin_model=model.clone(),  # 200 Hz 内环线程用独立 clone（Pinocchio Data 非线程安全）
         startup_joint_margin=0.08,
     )
-    backend.connect()
-    q = backend.get_joint_angles()
-    if q is None or len(q) != 7 or not np.all(np.isfinite(q)):
+    try:
+        backend.connect()
+        q = backend.get_joint_angles()
+        feedback = backend.get_robot().get_joint_angles()
+        stamp = float(getattr(feedback, 'timestamp', 0.0))
+        if q is None or len(q) != 7 or not np.all(np.isfinite(q)) or not 0 <= time.time() - stamp <= .5:
+            raise RuntimeError("[handoff] MIT joint feedback unavailable or stale")
+        return backend
+    except BaseException:
         backend.disconnect()
-        raise RuntimeError("[handoff] MIT backend connected but joint feedback unavailable")
-    return backend
+        raise
 
 
-def handoff(arm):
+def handoff(arm, args=None):
     """在抓取 arm 对象仍连接时完成控制权交接，返回可用的伺服后端。"""
     logger.info("[handoff] 1/3 stopping move_j control ...")
     if not stop_grasp_control(arm):
         raise RuntimeError("[handoff] aborted: move_j trajectory did not stop")
     logger.info("[handoff] 2/3 taking over with MIT backend (position hold) ...")
-    backend = build_servo_backend()
+    backend = build_servo_backend(channel=getattr(args, "nero_channel", "can0"),
+                                  firmware=getattr(args, "nero_firmware", "V111").lower())
     logger.info("[handoff] 3/3 releasing grasp-arm SDK driver ...")
     try:
         arm.arm.disconnect()  # 抓取驱动下线；MIT 已接管
     except Exception as exc:  # 非致命：驱动已无轨迹
-        logger.warning("[handoff] grasp-arm disconnect failed (non-fatal): %s", exc)
+        backend.emergency_stop()
+        backend.disconnect()
+        raise RuntimeError("Grasp driver disconnect failed during handoff") from exc
     return backend
 
 
 def delivery_after_grasp(arm, args):
     """runtime hook 入口：交接 → 伺服递送。"""
-    backend = handoff(arm)
+    if not getattr(args, 'execute', False):
+        raise RuntimeError('Hardware handoff requires --execute')
+    from cameras.realsense import close_all
+    from servo_delivery import validate_servo_args
+    validate_servo_args(args)
+    close_all()
+    backend = handoff(arm, args)
     from servo_delivery import run_servo_delivery  # 延迟 import，避免循环依赖
     return run_servo_delivery(args, backend=backend)

```

#### real_nero_grasp/cli.py

```diff
--- a/real_nero_grasp/cli.py
+++ b/real_nero_grasp/cli.py
@@ -17,7 +17,7 @@
 
 def parse_args():
     p = argparse.ArgumentParser(description="Single-view real Nero GraspNet runner.")
-    p.add_argument("--calib-json", required=True)
+    p.add_argument("--calib-json", help="D435 wrist camera to flange calibration JSON")
     p.add_argument("--handeye-convention", default="flange_camera_xyzw", choices=["flange_camera_xyzw", "camera_flange_xyzw", "flange_camera_wxyz", "camera_flange_wxyz"])
     p.add_argument("--orbbec-color-topic", default="/camera/color/image_raw")
     p.add_argument("--orbbec-depth-topic", default="/camera/depth/image_raw")
@@ -32,7 +32,7 @@
     p.add_argument("--orbbec-depth-median-max-mad", type=float, default=0.004, help="Reject a valid median-depth pixel when its temporal MAD exceeds this many metres; <=0 disables the High-Density flying-point filter")
     p.add_argument("--camera-self-mask", dest="camera_self_mask", action="store_true", help="Mask the calibrated fixed gripper silhouette from every aligned RGB-D frame")
     p.add_argument("--no-camera-self-mask", dest="camera_self_mask", action="store_false")
-    p.set_defaults(camera_self_mask=True)
+    p.set_defaults(camera_self_mask=False)
     p.add_argument("--camera-self-mask-expand-px", type=int, default=12, help="Pixel dilation around the fixed camera/gripper self-occlusion polygons")
     p.add_argument("--camera-self-edge-island-filter", dest="camera_self_edge_island_filter", action="store_true", help="Remove tiny near-depth components immediately beside the accepted gripper self mask")
     p.add_argument("--no-camera-self-edge-island-filter", dest="camera_self_edge_island_filter", action="store_false")
@@ -940,4 +940,15 @@
         "--enable-servo-delivery", action="store_true", default=False,
         help="仅当 --phase all 且抓取成功后：交接 MIT 后端并执行伺服递送",
     )
-    return p.parse_args()
+    from cameras.options import add_camera_arguments
+    from real_nero_grasp.servo_cli import add_servo_arguments
+    add_camera_arguments(p)
+    add_servo_arguments(p)
+    from cameras.options import apply_pipeline_defaults
+    apply_pipeline_defaults(p)
+    args = p.parse_args()
+    if args.phase != 'servo' and not args.calib_json:
+        p.error('--calib-json is required for grasp/all')
+    if (args.phase == 'servo' or args.enable_servo_delivery) and not args.handeye:
+        p.error('--handeye is required for servo delivery')
+    return args

```

#### real_nero_grasp/perception.py

```diff
--- a/real_nero_grasp/perception.py
+++ b/real_nero_grasp/perception.py
@@ -1540,8 +1540,8 @@
             raise RuntimeError(
                 f"color size {bgr.shape[1]}x{bgr.shape[0]} != depth size "
                 f"{depth_m.shape[1]}x{depth_m.shape[0]}. Depth must be aligned "
-                "to color. Start Orbbec with depth registration/alignment "
-                "enabled, or use an aligned depth topic."
+                "to color. Configure SDK depth registration/alignment "
+                "enabled."
             )
         xyz_img = depth_xyz_fn(depth_m, intr)
     if (xyz_img.shape[1], xyz_img.shape[0]) != (bgr.shape[1], bgr.shape[0]):
@@ -1661,10 +1661,10 @@
                 "Robot motion was not commanded."
             )
             print("[capture][check] Verify Orbbec is publishing:")
-            print(f"  ros2 topic hz {args.orbbec_color_topic}")
-            print(f"  ros2 topic hz {args.orbbec_depth_topic}")
+            print("  Check D435 SDK serial, USB3 connection and color profile.")
+            print("  Check aligned depth profile and frame freshness.")
             print(
-                f"  ros2 topic echo --once {args.orbbec_camera_info_topic}"
+                "  Intrinsics must come from the active SDK color profile."
             )
             print(
                 "[capture][check] If all are 0, restart Orbbec launch or "

```

#### real_nero_grasp/runtime.py

```diff
--- a/real_nero_grasp/runtime.py
+++ b/real_nero_grasp/runtime.py
@@ -3008,7 +3008,7 @@
     )
 
 
-def _workflow_hooks():
+def _workflow_hooks(args=None):
     """Resolve all legacy-patchable session dependencies for one run."""
     return _workflow_impl.WorkflowHooks(
         arm_factory=RealNeroGraspArm,
@@ -3032,10 +3032,11 @@
 
 
 def run_single_grasp(args):
-    return _workflow_impl.run_single_grasp(
-        args,
-        hooks=_workflow_hooks(),
-    )
+    from cameras.realsense import close_all
+    try:
+        return _workflow_impl.run_single_grasp(args, hooks=_workflow_hooks(args))
+    finally:
+        close_all()
 
 
 def main(argv=None):
@@ -3054,6 +3055,8 @@
         if getattr(args, "phase", "all") == "servo":
             from servo_delivery import run_servo_delivery
             raise SystemExit(run_servo_delivery(args, backend=None))
+        from cameras.preflight import grasp_preflight
+        grasp_preflight(args)
         result = run_single_grasp(args)
         if isinstance(result, GraspAttemptResult):
             print(

```

#### real_nero_grasp/servo_cli.py

```diff
--- /dev/null
+++ b/real_nero_grasp/servo_cli.py
@@ -0,0 +1,45 @@
+import argparse
+
+def servo_parser():
+    p = argparse.ArgumentParser(add_help=False)
+    p.add_argument("--execute", action="store_true", help="真实执行门槛（缺省 dry-run）")
+    p.add_argument("--target-source", choices=["aruco", "mediapipe"], default="aruco")
+    p.add_argument("--wrist-image-topic", default="/camera/camera/color/image_raw")
+    p.add_argument("--wrist-info-topic", default="/camera/camera/color/camera_info")
+    p.add_argument("--handeye", default=None,
+                   help="腕部相机手眼标定（伺服工程已验证的那套）")
+    p.add_argument("--wrist-handeye-convention", default="link7_cam")
+    p.add_argument("--wrist-handeye-mount-frame", default="feedback_link7")
+    p.add_argument("--offset-base", default="0.02,-0.025,0")
+    p.add_argument("--aruco-dict", default="ARUCO_ROS_ALL")
+    p.add_argument("--marker-id", type=int, default=582)
+    p.add_argument("--marker-size", type=float, default=0.0677)
+    p.add_argument("--servo-hz", type=float, default=30.0)
+    p.add_argument("--delivery-distance", type=float, default=0.15)
+    p.add_argument("--max-linear-vel", type=float, default=0.03)
+    p.add_argument("--max-angular-vel", type=float, default=0.6)
+    p.add_argument("--max-joint-vel", type=float, default=0.06)
+    p.add_argument("--visual-orientation", action="store_true", default=False)
+    p.add_argument("--approach-error-m", type=float, default=0.03)
+    p.add_argument("--approach-error-deg", type=float, default=5.0)
+    p.add_argument("--handover-distance", type=float, default=0.10)
+    p.add_argument("--stable-frames", type=int, default=10)
+    p.add_argument("--handover-stable-frames", type=int, default=20)
+    p.add_argument("--impedance-max-position-error", type=float, default=0.015)
+    p.add_argument("--impedance-max-orientation-error-deg", type=float, default=5.0)
+    p.add_argument("--impedance-target-linear-vel", type=float, default=0.03)
+    p.add_argument("--impedance-target-angular-vel", type=float, default=0.5)
+    p.add_argument("--cart-k-scale", type=float, default=1.0)
+    p.add_argument("--cart-b-scale", type=float, default=1.0)
+    p.add_argument("--follow-kp", type=float, default=0.0,
+                   help="固件关节跟随弹簧（逗号分隔 7 个或单值）")
+    p.add_argument("--follow-min-qdot", type=float, default=0.01)
+    p.add_argument("--dls-lambda", type=float, default=0.05)
+    p.add_argument("--use-dls", action="store_true", default=False)
+    p.add_argument("--vel-filter-tau", type=float, default=0.008)
+    return p
+
+def add_servo_arguments(parser):
+    for action in servo_parser()._actions:
+        if action.option_strings and not any(x in parser._option_string_actions for x in action.option_strings):
+            parser._add_action(action)

```

#### scripts/aruco_camera_jacobian_servo_sdk_dryrun.py

```diff
--- a/scripts/aruco_camera_jacobian_servo_sdk_dryrun.py
+++ b/scripts/aruco_camera_jacobian_servo_sdk_dryrun.py
@@ -40,18 +40,10 @@
 if SCRIPT_DIR not in sys.path:
     sys.path.insert(0, SCRIPT_DIR)
 
-for ros_path in (
-    "/opt/ros/humble/lib/python3.10/site-packages",
-    "/opt/ros/humble/local/lib/python3.10/dist-packages",
-    "/usr/lib/python3/dist-packages",
-):
-    if os.path.isdir(ros_path) and ros_path not in sys.path:
-        sys.path.append(ros_path)
-
-import rclpy  # noqa: E402
+from cameras import sdk_loop  # noqa: E402
 import cv2  # noqa: E402
-from rclpy.node import Node  # noqa: E402
-from sensor_msgs.msg import CameraInfo, Image  # noqa: E402
+from cameras.sdk_loop import Node  # noqa: E402
+from cameras.sdk_loop import CameraInfo, Image  # noqa: E402
 
 from aruco_jacobian_servo_sdk_dryrun import SdkJointFeedbackReader  # noqa: E402
 from control_backends.base import BackendCommand  # noqa: E402
@@ -356,7 +348,7 @@
         now = time.monotonic()
         if self.args.duration > 0.0 and now - self.started >= self.args.duration:
             self.get_logger().info(f"duration reached: {self.args.duration:.1f}s")
-            rclpy.shutdown()
+            sdk_loop.shutdown()
             return
 
         q_now = self.reader.read_joint_angles()
@@ -471,7 +463,7 @@
             key = cv2.waitKey(1) & 0xFF
             if key in (27, ord("q")):
                 self.get_logger().info("display requested shutdown")
-                rclpy.shutdown()
+                sdk_loop.shutdown()
         except Exception as exc:
             self.display_enabled = False
             self.get_logger().warning(f"camera display disabled: {exc}")
@@ -551,6 +543,8 @@
     parser.add_argument("--orientation-deadband-deg", type=float, default=1.5)
     parser.add_argument("--kp-scale", type=float, default=1.0)
     parser.add_argument("--kd-scale", type=float, default=1.0)
+    from cameras.options import add_camera_arguments
+    add_camera_arguments(parser)
     return parser.parse_args()
 
 
@@ -563,17 +557,17 @@
     if not Path(args.handeye).expanduser().exists():
         raise FileNotFoundError(args.handeye)
 
-    rclpy.init()
+    sdk_loop.init()
     node = ArucoCameraServoDryRunNode(args)
     try:
-        rclpy.spin(node)
+        sdk_loop.spin(node)
     except KeyboardInterrupt:
         node.get_logger().info("interrupted")
     finally:
         node.close()
         node.destroy_node()
-        if rclpy.ok():
-            rclpy.shutdown()
+        if sdk_loop.ok():
+            sdk_loop.shutdown()
 
 
 if __name__ == "__main__":

```

#### scripts/aruco_camera_jacobian_servo_sdk_execute.py

```diff
--- a/scripts/aruco_camera_jacobian_servo_sdk_execute.py
+++ b/scripts/aruco_camera_jacobian_servo_sdk_execute.py
@@ -41,18 +41,10 @@
 if SCRIPT_DIR not in sys.path:
     sys.path.insert(0, SCRIPT_DIR)
 
-for ros_path in (
-    "/opt/ros/humble/lib/python3.10/site-packages",
-    "/opt/ros/humble/local/lib/python3.10/dist-packages",
-    "/usr/lib/python3/dist-packages",
-):
-    if os.path.isdir(ros_path) and ros_path not in sys.path:
-        sys.path.append(ros_path)
-
-import rclpy  # noqa: E402
+from cameras import sdk_loop  # noqa: E402
 import cv2  # noqa: E402
-from rclpy.node import Node  # noqa: E402
-from sensor_msgs.msg import CameraInfo, Image  # noqa: E402
+from cameras.sdk_loop import Node  # noqa: E402
+from cameras.sdk_loop import CameraInfo, Image  # noqa: E402
 
 from aruco_camera_jacobian_servo_sdk_dryrun import (  # noqa: E402
     DEFAULT_HAND_EYE,
@@ -751,22 +743,22 @@
                 self._fail_closed_impedance(
                     f"unexpected impedance timer exception: "
                     f"{type(exc).__name__}: {exc}")
-                if rclpy.ok():
-                    rclpy.shutdown()
+                if sdk_loop.ok():
+                    sdk_loop.shutdown()
                 return
             raise
 
     def _on_timer_impl(self) -> None:
         now = time.monotonic()
         if self._check_inner_impedance_fault():
-            rclpy.shutdown()
+            sdk_loop.shutdown()
             return
         if self._impedance_terminal_fault:
-            rclpy.shutdown()
+            sdk_loop.shutdown()
             return
         if self.args.duration > 0.0 and now - self.started >= self.args.duration:
             self.get_logger().info(f"duration reached: {self.args.duration:.1f}s")
-            rclpy.shutdown()
+            sdk_loop.shutdown()
             return
 
         q_now = self._read_joint_angles()
@@ -788,7 +780,7 @@
             except Exception as exc:
                 self._fail_closed_impedance(
                     f"latched safe-hold command failed: {exc}")
-                rclpy.shutdown()
+                sdk_loop.shutdown()
                 return
             self.last_mit_q_cmd = q_now.copy()
             self.count += 1
@@ -847,7 +839,7 @@
 
         if decision.fatal:
             self._handle_fatal_safety_stop(status, decision.reason)
-            rclpy.shutdown()
+            sdk_loop.shutdown()
             return
 
         if decision.hold:
@@ -900,7 +892,7 @@
                 )
 
         if self._impedance_terminal_fault:
-            rclpy.shutdown()
+            sdk_loop.shutdown()
             return
 
         reason = decision.reason
@@ -917,7 +909,7 @@
             if self.args.control_mode == "mit-impedance":
                 self._fail_closed_impedance(
                     f"impedance command send failed: {exc}")
-                rclpy.shutdown()
+                sdk_loop.shutdown()
                 return
             raise
         if self.args.execution_backend != "cpv":
@@ -927,7 +919,7 @@
         if self.args.control_mode == "mit-impedance":
             self._check_impedance_instability(q_now, now)
             if self._impedance_terminal_fault:
-                rclpy.shutdown()
+                sdk_loop.shutdown()
                 return
         self.count += 1
         self._update_display(status, target, servo_out)
@@ -1052,11 +1044,11 @@
         ok, reason = self.safety.check_start_posture(q_now)
         if not ok:
             self.get_logger().error(f"refusing to start: unsafe start posture: {reason}")
-            rclpy.shutdown()
+            sdk_loop.shutdown()
             return
 
         if not self._check_start_pose(q_now):
-            rclpy.shutdown()
+            sdk_loop.shutdown()
             return
 
         if target is None:
@@ -1126,16 +1118,16 @@
             q_now = self.backend.get_joint_angles() if self.backend is not None else None
             if q_now is None:
                 self.get_logger().error("MIT backend running but no joint feedback")
-                rclpy.shutdown()
+                sdk_loop.shutdown()
                 return
             q_now = np.asarray(q_now, dtype=np.float64).ravel()
             ok, reason = self.safety.check_start_posture(q_now)
             if not ok:
                 self.get_logger().error(f"unsafe posture: {reason}")
-                rclpy.shutdown()
+                sdk_loop.shutdown()
                 return
             if not self._check_start_pose(q_now):
-                rclpy.shutdown()
+                sdk_loop.shutdown()
                 return
             self.last_mit_q_cmd = q_now.copy()
             self.phase = "settle"
@@ -1160,7 +1152,7 @@
                     "refusing CPV execute: joints are not already enabled; "
                     f"enable_list={enable_list}. Run move_to_aruco_start_pose.py first."
                 )
-                rclpy.shutdown()
+                sdk_loop.shutdown()
                 return
             robot = self.reader.take_robot()
             self.backend = SdkCpvVelocityBackend(
@@ -1187,15 +1179,15 @@
             self.get_logger().error(
                 f"{backend_name} backend started but joint feedback did not match preflight feedback"
             )
-            rclpy.shutdown()
+            sdk_loop.shutdown()
             return
         ok, reason = self.safety.check_start_posture(q_now)
         if not ok:
             self.get_logger().error(f"unsafe posture after {backend_name} start: {reason}")
-            rclpy.shutdown()
+            sdk_loop.shutdown()
             return
         if not self._check_start_pose(q_now):
-            rclpy.shutdown()
+            sdk_loop.shutdown()
             return
         if backend_name != "cpv":
             self.last_mit_q_cmd = q_now.copy()
@@ -1265,7 +1257,7 @@
     def _settle_step(self, q_now: np.ndarray, target, now: float) -> None:
         if self.backend is None or self.settle_started is None or self.settle_q_ref is None:
             self.get_logger().error("settle phase entered without backend/reference")
-            rclpy.shutdown()
+            sdk_loop.shutdown()
             return
 
         if self.args.control_mode == "mit-impedance":
@@ -1274,7 +1266,7 @@
                 self.get_logger().error(
                     "refusing impedance engagement during settle: "
                     f"unsafe limit margin: {reason}")
-                rclpy.shutdown()
+                sdk_loop.shutdown()
                 return
 
         fake_servo_out = {
@@ -1285,7 +1277,7 @@
         decision = self.safety.evaluate(q_now, fake_servo_out, now)
         if decision.fatal or decision.hold:
             self.get_logger().error(f"settle safety stop: {decision.status} {decision.reason}")
-            rclpy.shutdown()
+            sdk_loop.shutdown()
             return
 
         drift = float(np.max(np.abs(q_now - self.settle_q_ref)))
@@ -1294,7 +1286,7 @@
                 f"{self.args.execution_backend} hold drift too large during settle: "
                 f"{np.degrees(drift):.2f}deg > {self.args.settle_max_drift_deg:.2f}deg"
             )
-            rclpy.shutdown()
+            sdk_loop.shutdown()
             return
 
         if self.args.control_mode == "mit-impedance":
@@ -2138,7 +2130,7 @@
             key = cv2.waitKey(1) & 0xFF
             if key in (27, ord("q")):
                 self.get_logger().info("display requested shutdown")
-                rclpy.shutdown()
+                sdk_loop.shutdown()
         except Exception as exc:
             self.display_enabled = False
             self.get_logger().warning(f"camera display disabled: {exc}")
@@ -2602,11 +2594,11 @@
     if not Path(args.handeye).expanduser().exists():
         raise FileNotFoundError(args.handeye)
 
-    rclpy.init()
+    sdk_loop.init()
     node = None
     try:
         node = ArucoCameraServoExecuteNode(args)
-        rclpy.spin(node)
+        sdk_loop.spin(node)
     except KeyboardInterrupt:
         if node is not None:
             node.get_logger().info("interrupted")
@@ -2621,8 +2613,8 @@
                 node.destroy_node()
             except Exception as exc:
                 cleanup_error = cleanup_error or exc
-        if rclpy.ok():
-            rclpy.shutdown()
+        if sdk_loop.ok():
+            sdk_loop.shutdown()
         if cleanup_error is not None:
             raise RuntimeError(
                 f"visual-servo cleanup was not confirmed: {cleanup_error}"

```

#### scripts/aruco_jacobian_servo_ros_dryrun.py

```diff
--- a/scripts/aruco_jacobian_servo_ros_dryrun.py
+++ b/scripts/aruco_jacobian_servo_ros_dryrun.py
@@ -1,332 +1,4 @@
 #!/usr/bin/env python3
-"""Continuous ROS dry-run for wrist ArUco Jacobian servo.
-
-This script subscribes to current joints and a base-frame ArUco pose, then
-runs the Panthera-style position-only Jacobian servo continuously.
-
-It never sends real robot commands.  Commands go through
-SdkForceCtrlBackend(dry_run=True) so the unified backend path is exercised
-without touching CAN / MIT control.
-
-Typical run:
-
-    # Terminal A: publish /aruco_pose_base from the wrist camera
-    python3 /home/dpf/桌面/RealRobot_Test/real_test/real_visual_servo/aruco_pose_from_eye_in_hand.py \
-      --handeye /home/dpf/桌面/RealRobot_Test/real_test/configs/2026-05-19_14-45-49_calibration.json \
-      --handeye-mount-frame feedback_link7 \
-      --tcp-feedback-frame tcp_offset \
-      --output-topic /aruco_pose_base
-
-    # Terminal B: compute servo commands only
-    python3 scripts/aruco_jacobian_servo_ros_dryrun.py
-"""
-
-from __future__ import annotations
-
-import argparse
-import logging
-import math
-import os
-import sys
-import time
-from typing import Iterable, Optional
-
-import numpy as np
-
-
-SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
-PROJECT_ROOT = os.path.normpath(os.path.join(SCRIPT_DIR, ".."))
-if PROJECT_ROOT not in sys.path:
-    sys.path.insert(0, PROJECT_ROOT)
-
-for ros_path in (
-    "/opt/ros/humble/lib/python3.10/site-packages",
-    "/opt/ros/humble/local/lib/python3.10/dist-packages",
-    "/usr/lib/python3/dist-packages",
-):
-    if os.path.isdir(ros_path) and ros_path not in sys.path:
-        sys.path.append(ros_path)
-
-import rclpy  # noqa: E402
-from geometry_msgs.msg import PoseStamped  # noqa: E402
-from rclpy.node import Node  # noqa: E402
-from sensor_msgs.msg import JointState  # noqa: E402
-
-from control_backends.base import BackendCommand  # noqa: E402
-from control_backends.sdk_force_ctrl_backend import (  # noqa: E402
-    DEFAULT_KD,
-    DEFAULT_KP,
-    SdkForceCtrlBackend,
-)
-from models.nero_model import ARM_JOINT_NAMES  # noqa: E402
-from servo.jacobian_position_servo import JacobianPositionServo, ServoParams  # noqa: E402
-
-
-def format_vec(values: Iterable[float], precision: int = 4) -> str:
-    return "[" + ", ".join(f"{float(v):.{precision}f}" for v in values) + "]"
-
-
-def quat_xyzw_to_matrix(quat_xyzw: Iterable[float]) -> np.ndarray:
-    x, y, z, w = [float(v) for v in quat_xyzw]
-    norm = math.sqrt(x * x + y * y + z * z + w * w)
-    if norm < 1e-12:
-        raise ValueError("Quaternion norm is zero")
-    x, y, z, w = x / norm, y / norm, z / norm, w / norm
-    return np.array(
-        [
-            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
-            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
-            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
-        ],
-        dtype=np.float64,
-    )
-
-
-def marker_pose_to_center_normal(msg: PoseStamped) -> tuple[np.ndarray, np.ndarray]:
-    pose = msg.pose
-    center = np.array(
-        [pose.position.x, pose.position.y, pose.position.z],
-        dtype=np.float64,
-    )
-    rotation = quat_xyzw_to_matrix(
-        [
-            pose.orientation.x,
-            pose.orientation.y,
-            pose.orientation.z,
-            pose.orientation.w,
-        ]
-    )
-    normal = rotation[:, 2].copy()
-    return center, normal
-
-
-def joint_state_to_q(msg: JointState, joint_names: list[str]) -> np.ndarray:
-    if msg.name:
-        joint_map = {
-            name: float(msg.position[index])
-            for index, name in enumerate(msg.name)
-            if index < len(msg.position)
-        }
-        missing = [name for name in joint_names if name not in joint_map]
-        if missing:
-            raise KeyError(f"Missing joints in feedback: {missing}")
-        return np.asarray([joint_map[name] for name in joint_names], dtype=np.float64)
-
-    if len(msg.position) < len(joint_names):
-        raise ValueError(
-            f"JointState has {len(msg.position)} positions but needs {len(joint_names)}"
-        )
-    return np.asarray(msg.position[: len(joint_names)], dtype=np.float64)
-
-
-class ArucoJacobianServoDryRunNode(Node):
-    def __init__(self, args: argparse.Namespace) -> None:
-        super().__init__("nero_aruco_jacobian_servo_ros_dryrun")
-        self.args = args
-        self.joint_names = args.joint_names.split(",")
-        self.params = ServoParams(
-            servo_hz=args.servo_hz,
-            desired_distance=args.desired_distance,
-            linear_gain=args.linear_gain,
-            max_linear_vel=args.max_linear_vel,
-            damping=args.damping,
-            max_joint_vel=args.max_joint_vel,
-            max_joint_step=args.max_joint_step,
-            target_filter_alpha=args.target_filter_alpha,
-            normal_filter_alpha=args.normal_filter_alpha,
-            qdot_filter_alpha=args.qdot_filter_alpha,
-            target_timeout_s=args.target_timeout_s,
-            max_target_jump=args.max_target_jump,
-            position_deadband=args.position_deadband,
-        )
-        self.servo = JacobianPositionServo(params=self.params)
-        self.backend = SdkForceCtrlBackend(dry_run=True, channel=args.channel)
-        self.backend.connect()
-
-        self.kp = DEFAULT_KP.copy() * float(args.kp_scale)
-        self.kd = DEFAULT_KD.copy() * float(args.kd_scale)
-        self.q_now: Optional[np.ndarray] = None
-        self.last_joint_time: Optional[float] = None
-        self.target_position: Optional[np.ndarray] = None
-        self.target_normal: Optional[np.ndarray] = None
-        self.last_target_time: Optional[float] = None
-        self.last_log_time = 0.0
-        self.last_joint_warn_time = 0.0
-        self.started = time.monotonic()
-        self.command_count = 0
-
-        self.create_subscription(JointState, args.joint_topic, self.on_joint, 10)
-        self.create_subscription(PoseStamped, args.target_topic, self.on_target, 10)
-        self.create_timer(1.0 / float(args.servo_hz), self.on_timer)
-
-        self.get_logger().info("DRY-RUN only: no real robot commands will be sent.")
-        self.get_logger().info(f"joint_topic={args.joint_topic}")
-        self.get_logger().info(f"target_topic={args.target_topic}")
-        self.get_logger().info(f"joint_names={self.joint_names}")
-        self.get_logger().info(
-            "servo: "
-            f"hz={self.params.servo_hz:.1f} desired_distance={self.params.desired_distance:.3f}m "
-            f"max_linear_vel={self.params.max_linear_vel:.3f}m/s "
-            f"max_joint_vel={self.params.max_joint_vel:.3f}rad/s "
-            f"max_joint_step={self.params.max_joint_step:.4f}rad"
-        )
-
-    def on_joint(self, msg: JointState) -> None:
-        try:
-            self.q_now = joint_state_to_q(msg, self.joint_names)
-            self.last_joint_time = time.monotonic()
-        except Exception as exc:
-            now = time.monotonic()
-            if now - self.last_joint_warn_time >= 1.0:
-                self.get_logger().warning(f"joint feedback parse failed: {exc}")
-                self.last_joint_warn_time = now
-
-    def on_target(self, msg: PoseStamped) -> None:
-        try:
-            self.target_position, self.target_normal = marker_pose_to_center_normal(msg)
-            self.last_target_time = time.monotonic()
-        except Exception as exc:
-            self.get_logger().warning(f"target pose parse failed: {exc}")
-
-    def on_timer(self) -> None:
-        now = time.monotonic()
-        if self.q_now is None:
-            self._log_waiting(now, "waiting for joint feedback")
-            self._maybe_shutdown(now)
-            return
-
-        target_age = None
-        target_position = self.target_position
-        target_normal = self.target_normal
-        if self.last_target_time is None:
-            target_position = None
-            target_normal = None
-        else:
-            target_age = now - self.last_target_time
-            if target_age > self.params.target_timeout_s:
-                target_position = None
-                target_normal = None
-
-        result = self.servo.update(
-            q_now=self.q_now,
-            target_position=target_position,
-            target_normal=target_normal,
-            stamp=now,
-        )
-        cmd = BackendCommand(
-            q_cmd=result["q_cmd"],
-            qdot_cmd=result["qdot"],
-            kp=self.kp,
-            kd=self.kd,
-            mode="mit",
-            stamp=now,
-        )
-        self.backend.send(cmd)
-        self.command_count += 1
-
-        if now - self.last_log_time >= self.args.log_period:
-            self._log_result(result, target_age)
-            self.last_log_time = now
-
-        self._maybe_shutdown(now)
-
-    def _log_waiting(self, now: float, text: str) -> None:
-        if now - self.last_log_time >= self.args.log_period:
-            self.get_logger().info(text)
-            self.last_log_time = now
-
-    def _log_result(self, result: dict, target_age: Optional[float]) -> None:
-        if result["target_rejected"]:
-            status = "target_rejected_hold"
-        elif result["target_lost"]:
-            status = "target_lost_hold"
-        else:
-            status = "tracking"
-
-        q_step = result["q_cmd"] - self.q_now
-        qdot_max = float(np.max(np.abs(result["qdot"])))
-        step_max = float(np.max(np.abs(q_step)))
-        error_norm = float(np.linalg.norm(result["error"]))
-        age_text = "none" if target_age is None else f"{target_age:.3f}s"
-        target_text = "None"
-        normal_text = "None"
-        normal_side_text = "n/a"
-        if result["target_tcp"] is not None:
-            target_text = format_vec(result["target_tcp"], 4)
-            normal_text = format_vec(result["marker_normal"], 4)
-            normal_side = float(
-                np.dot(
-                    result["tcp_now"] - result["marker_center"],
-                    result["marker_normal"],
-                )
-            )
-            normal_side_text = f"{normal_side:.4f}m"
-
-        self.get_logger().info(
-            f"{status} count={self.command_count} target_age={age_text} "
-            f"err={error_norm:.4f}m qdot_max={qdot_max:.5f}rad/s "
-            f"step_max={step_max:.5f}rad normal_side={normal_side_text}\n"
-            f"  tcp_now={format_vec(result['tcp_now'], 4)} "
-            f"target_tcp={target_text} normal={normal_text}\n"
-            f"  q_now_deg={format_vec(np.degrees(self.q_now), 2)} "
-            f"q_cmd_deg={format_vec(np.degrees(result['q_cmd']), 2)}"
-        )
-
-    def _maybe_shutdown(self, now: float) -> None:
-        if self.args.duration > 0.0 and now - self.started >= self.args.duration:
-            self.get_logger().info(f"duration reached: {self.args.duration:.1f}s")
-            rclpy.shutdown()
-
-    def close(self) -> None:
-        self.backend.disconnect()
-
-
-def parse_args() -> argparse.Namespace:
-    parser = argparse.ArgumentParser(description=__doc__)
-    parser.add_argument("--joint-topic", default="/feedback/joint_states")
-    parser.add_argument("--target-topic", default="/aruco_pose_base")
-    parser.add_argument("--joint-names", default=",".join(ARM_JOINT_NAMES))
-    parser.add_argument("--channel", default="can0")
-    parser.add_argument("--duration", type=float, default=0.0, help="Seconds to run; 0 means forever.")
-    parser.add_argument("--log-period", type=float, default=0.5)
-    parser.add_argument("--verbose-backend", action="store_true", help="Print every dry-run BackendCommand.")
-
-    parser.add_argument("--servo-hz", type=float, default=30.0)
-    parser.add_argument("--desired-distance", type=float, default=0.15)
-    parser.add_argument("--linear-gain", type=float, default=1.0)
-    parser.add_argument("--max-linear-vel", type=float, default=0.02)
-    parser.add_argument("--damping", type=float, default=0.04)
-    parser.add_argument("--max-joint-vel", type=float, default=0.05)
-    parser.add_argument("--max-joint-step", type=float, default=0.003)
-    parser.add_argument("--target-filter-alpha", type=float, default=0.20)
-    parser.add_argument("--normal-filter-alpha", type=float, default=0.20)
-    parser.add_argument("--qdot-filter-alpha", type=float, default=0.35)
-    parser.add_argument("--target-timeout-s", type=float, default=0.45)
-    parser.add_argument("--max-target-jump", type=float, default=0.06)
-    parser.add_argument("--position-deadband", type=float, default=0.002)
-    parser.add_argument("--kp-scale", type=float, default=1.0)
-    parser.add_argument("--kd-scale", type=float, default=1.0)
-    return parser.parse_args()
-
-
-def main() -> None:
-    args = parse_args()
-    logging.basicConfig(
-        level=logging.INFO if args.verbose_backend else logging.WARNING,
-        format="%(levelname)s: %(message)s",
-    )
-    rclpy.init()
-    node = ArucoJacobianServoDryRunNode(args)
-    try:
-        rclpy.spin(node)
-    except KeyboardInterrupt:
-        node.get_logger().info("interrupted")
-    finally:
-        node.close()
-        node.destroy_node()
-        if rclpy.ok():
-            rclpy.shutdown()
-
-
-if __name__ == "__main__":
-    main()
+"""The ROS joint/pose-topic diagnostic was retired during SDK migration."""
+if __name__ == '__main__':
+    raise SystemExit('Use scripts/aruco_camera_jacobian_servo_sdk_dryrun.py with D435 SDK instead.')

```

#### scripts/check_env.sh

```diff
--- a/scripts/check_env.sh
+++ b/scripts/check_env.sh
@@ -1,77 +1,24 @@
 #!/usr/bin/env bash
 set -euo pipefail
-
-PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
-: "${CONDA_PREFIX:?请先激活 grasp_yolo26 环境}"
-# # CONDA_SH="/home/zgj/anaconda3/etc/profile.d/conda.sh"
-# ROS_SETUP="/opt/ros/humble/setup.bash"
-# # NERO_SETUP="/home/zgj/Zgj/nero_handeye/install/setup.bash"
-# # USER_SITE="/home/zgj/.local/lib/python3.10/site-packages"
-
-# CONDA_SH="${CONDA_SH:-$(conda info --base 2>/dev/null)/etc/profile.d/conda.sh}"
-# NERO_SETUP="${NERO_SETUP:-}"
-# USER_SITE="${USER_SITE:-$(python -c 'import site; print(site.getusersitepackages())')}"
-
-# unset PYTHONPATH
-# unset LD_LIBRARY_PATH
+PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
+: "${CONDA_PREFIX:?Activate grasp_yolo26 before running this script}"
 export PYTHONNOUSERSITE=1
-
-# source "$CONDA_SH"
-# set +u
-# conda activate grasp_yolo26
-# set -u
-
-# CONDA_SITE="$(python - <<'PY'
-# import site
-# print(site.getsitepackages()[0])
-# PY
-# )"
-# CONDA_LIB="$CONDA_PREFIX/lib"
-
-# set +u
-# source "$ROS_SETUP"
-# source "$NERO_SETUP"
-# set -u
-
-# ROS_PYTHONPATH="${PYTHONPATH-}"
-# ROS_LD_LIBRARY_PATH="${LD_LIBRARY_PATH-}"
-
-# export PYTHONPATH="$CONDA_SITE:$ROS_PYTHONPATH:$USER_SITE"
-# export LD_LIBRARY_PATH="$CONDA_LIB:$ROS_LD_LIBRARY_PATH"
-
 cd "$PROJECT_DIR"
 "$CONDA_PREFIX/bin/python" - <<'PY'
-import importlib.util
-import os
-import site
+import importlib
 import sys
-
-print("== Real Nero Python Environment ==")
-print("python:", sys.executable)
-print("CONDA_PREFIX:", os.environ.get("CONDA_PREFIX", ""))
-print("PYTHONNOUSERSITE:", os.environ.get("PYTHONNOUSERSITE", ""))
-print("ENABLE_USER_SITE:", site.ENABLE_USER_SITE)
-print("PYTHONPATH entries:")
-for item in os.environ.get("PYTHONPATH", "").split(":"):
-    if item:
-        print(" ", item)
-print("LD_LIBRARY_PATH entries:")
-for item in os.environ.get("LD_LIBRARY_PATH", "").split(":"):
-    if item:
-        print(" ", item)
-print()
-
-failed = []
-for name in [
-    "numpy", "scipy", "cv2", "can", "pyAgxArm", "pyrealsense2",
-    "open3d", "ultralytics", "torch", "pinocchio", "mujoco", "pytest",
-]:
+print('Python:', sys.executable)
+failed=[]
+for name in ['numpy','scipy','cv2','can','pyAgxArm','pyrealsense2','open3d','ultralytics','torch','pinocchio','mujoco','pytest','trac_ik']:
     try:
-        mod = __import__(name)
-        print(f"{name}: version={getattr(mod, '__version__', '?')} file={getattr(mod, '__file__', '?')}")
+        module=importlib.import_module(name)
+        print(name, getattr(module,'__file__',None))
     except Exception as exc:
-        spec = importlib.util.find_spec(name)
-        print(f"{name}: import_error={type(exc).__name__}: {exc}; spec={None if spec is None else spec.origin}")
-        failed.append(name)
-raise SystemExit(1 if failed else 0)
+        print(name, type(exc).__name__, str(exc)); failed.append(name)
+if 'torch' not in failed:
+    import torch
+    print('CUDA build:',torch.version.cuda,'available:',torch.cuda.is_available())
+    if not torch.cuda.is_available(): failed.append('CUDA')
+print('Failed:',failed)
+raise SystemExit(bool(failed))
 PY

```

#### scripts/launch_orbbec.sh

```diff
--- a/scripts/launch_orbbec.sh
+++ b/scripts/launch_orbbec.sh
@@ -1,46 +1,4 @@
 #!/usr/bin/env bash
-set -euo pipefail
-
-ROS_SETUP="${ROS_SETUP:-/opt/ros/humble/setup.bash}"
-CAMERA_WORKSPACE_SETUP="${CAMERA_WORKSPACE_SETUP:-/home/zgj/Zgj/nero_handeye/install/setup.bash}"
-
-if [[ ! -f "$ROS_SETUP" ]]; then
-  echo "ROS setup not found: $ROS_SETUP" >&2
-  exit 2
-fi
-if [[ ! -f "$CAMERA_WORKSPACE_SETUP" ]]; then
-  echo "Camera workspace setup not found: $CAMERA_WORKSPACE_SETUP" >&2
-  exit 2
-fi
-
-set +u
-source "$ROS_SETUP"
-source "$CAMERA_WORKSPACE_SETUP"
-set -u
-
-echo "[camera] Gemini 330 grasp configuration"
-echo "[camera] preset=${GEMINI_DEVICE_PRESET:-High Density} registration=color frame_sync=true"
-echo "[camera] SDK temporal=${GEMINI_TEMPORAL_FILTER:-false} spatial=${GEMINI_SPATIAL_FILTER:-false} software_noise=${GEMINI_SOFTWARE_NOISE_FILTER:-true} hardware_noise=${GEMINI_HARDWARE_NOISE_FILTER:-true} SDK_hole_fill=false"
-echo "[camera] settled-pose temporal fusion is performed by the grasp runner; do not use SDK temporal/spatial filling to fabricate basket-edge depth"
-
-exec ros2 launch orbbec_camera gemini_330_series.launch.py \
-  camera_name:=camera \
-  enable_color:=true \
-  enable_depth:=true \
-  depth_registration:=true \
-  align_mode:=SW \
-  align_target_stream:=COLOR \
-  enable_frame_sync:=true \
-  enable_point_cloud:=true \
-  enable_colored_point_cloud:=true \
-  ordered_pc:=true \
-  device_preset:="${GEMINI_DEVICE_PRESET:-High Density}" \
-  enable_temporal_filter:="${GEMINI_TEMPORAL_FILTER:-false}" \
-  enable_spatial_filter:="${GEMINI_SPATIAL_FILTER:-false}" \
-  enable_noise_removal_filter:="${GEMINI_SOFTWARE_NOISE_FILTER:-true}" \
-  noise_removal_filter_min_diff:="${GEMINI_NOISE_REMOVAL_MIN_DIFF:-256}" \
-  noise_removal_filter_max_size:="${GEMINI_NOISE_REMOVAL_MAX_SIZE:-80}" \
-  enable_hardware_noise_removal_filter:="${GEMINI_HARDWARE_NOISE_FILTER:-true}" \
-  hardware_noise_removal_filter_threshold:="${GEMINI_HARDWARE_NOISE_THRESHOLD:--1.0}" \
-  enable_hole_filling_filter:=false \
-  "$@"
+# 330 is excluded from the current D435 SDK phase.
+echo 'Gemini 330 is disabled. D435 SDK camera is managed by main.py; no ROS launch is required.' >&2
+exit 2

```

#### scripts/run_grasp.sh

```diff
--- a/scripts/run_grasp.sh
+++ b/scripts/run_grasp.sh
@@ -1,43 +1,10 @@
 #!/usr/bin/env bash
 set -euo pipefail
-
 PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
-: "${CONDA_PREFIX:?请先激活 grasp_yolo26 环境}"
-PYTHON_BIN="$CONDA_PREFIX/bin/python"
-# CONDA_SH="/home/zgj/anaconda3/etc/profile.d/conda.sh"
-# ROS_SETUP="/opt/ros/humble/setup.bash"
-# NERO_SETUP="/home/zgj/Zgj/nero_handeye/install/setup.bash"
-# USER_SITE="/home/zgj/.local/lib/python3.10/site-packages"
-
-# unset PYTHONPATH
-# unset LD_LIBRARY_PATH
+: "${CONDA_PREFIX:?Activate grasp_yolo26 before running this script}"
 export PYTHONNOUSERSITE=1
 export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"
 export YOLO_CONFIG_DIR="${YOLO_CONFIG_DIR:-$PROJECT_DIR/.ultralytics}"
-mkdir -p "$MPLCONFIGDIR" "$YOLO_CONFIG_DIR/Ultralytics"
-
-# source "$CONDA_SH"
-# set +u
-conda activate grasp_yolo26
-set -u
-
-CONDA_SITE="$(python - <<'PY'
-import site
-print(site.getsitepackages()[0])
-PY
-)"
-CONDA_LIB="$CONDA_PREFIX/lib"
-
-set +u
-source "$ROS_SETUP"
-source "$NERO_SETUP"
-set -u
-
-ROS_PYTHONPATH="${PYTHONPATH-}"
-ROS_LD_LIBRARY_PATH="${LD_LIBRARY_PATH-}"
-
-export PYTHONPATH="$CONDA_SITE:$ROS_PYTHONPATH:$USER_SITE"
-export LD_LIBRARY_PATH="$CONDA_LIB:$ROS_LD_LIBRARY_PATH"
-
+mkdir -p "$MPLCONFIGDIR" "$YOLO_CONFIG_DIR"
 cd "$PROJECT_DIR"
-exec "$PYTHON_BIN" main.py "$@"
+exec "$CONDA_PREFIX/bin/python" main.py "$@"

```

#### scripts/run_grasp_safe.sh

```diff
--- a/scripts/run_grasp_safe.sh
+++ b/scripts/run_grasp_safe.sh
@@ -14,10 +14,10 @@
   GRASP_VALIDATION_FIRST_FLAG="--no-grasp-validation-first"
 fi
 
-exec bash "$PROJECT_DIR/run_real_nero_clean.sh" \
+exec bash "$PROJECT_DIR/run_grasp.sh" \
   "$@" \
   --nero-speed-percent 5 \
-  --camera-self-mask \
+  --no-camera-self-mask \
   --grasp-motion-path-guard \
   --motion-runtime-collision-guard \
   --motion-runtime-torque-guard \

```

#### scripts/test_zero_gravity_diag.py

```diff
--- a/scripts/test_zero_gravity_diag.py
+++ b/scripts/test_zero_gravity_diag.py
@@ -152,13 +152,7 @@
             ja = robot.get_joint_angles()
             if ja is not None:
                 q = np.array(ja.msg, dtype=np.float64)
-            for i in range(7):
-                    p_des=float(q[i]),
-                    v_des=0.0,
-                    kp=float(kp[i]) if hasattr(kp, '__len__') else float(kp),
-                    kd=float(kd[i]) if hasattr(kd, '__len__') else float(kd),
-                    t_ff=float(t_ff[i]) if hasattr(t_ff, '__len__') else float(t_ff),
-                )
+            raise RuntimeError('Legacy MIT diagnostic had a missing command call; use validated SDK hold tools instead')
             count += 1
             time.sleep(0.005)
 
@@ -216,4 +210,4 @@
 
 
 if __name__ == "__main__":
-    main()
+    raise SystemExit("This incomplete legacy diagnostic is disabled; use test_force_position_hold.py")

```

#### servo_delivery.py

```diff
--- a/servo_delivery.py
+++ b/servo_delivery.py
@@ -13,9 +13,9 @@
 from collections import deque
 
 import numpy as np
-import rclpy
-from rclpy.node import Node
-from sensor_msgs.msg import CameraInfo, Image
+from cameras import sdk_loop
+from cameras.sdk_loop import Node
+from cameras.sdk_loop import CameraInfo, Image
 
 from models.nero_model import NeroPinocchioModel
 from servo.pbvs_pose_servo import PBVSPoseServo, PBVSParams
@@ -123,8 +123,8 @@
             aruco_dict=args.aruco_dict,
             marker_id=args.marker_id,
             marker_size=args.marker_size,
-            handeye_convention=args.handeye_convention,
-            handeye_mount_frame=args.handeye_mount_frame,
+            handeye_convention=args.wrist_handeye_convention,
+            handeye_mount_frame=args.wrist_handeye_mount_frame,
             offset_base=np.fromstring(args.offset_base, sep=","),
             model=model,
         )
@@ -132,8 +132,8 @@
         from targets.hand_target import HandTarget
         return HandTarget(
             handeye_path=args.handeye,
-            handeye_convention=args.handeye_convention,
-            handeye_mount_frame=args.handeye_mount_frame,
+            handeye_convention=args.wrist_handeye_convention,
+            handeye_mount_frame=args.wrist_handeye_mount_frame,
             offset_base=np.fromstring(args.offset_base, sep=","),
             model=model,
         )
@@ -143,12 +143,12 @@
 class ServoDeliveryNode(Node):
     """递送状态机：HOLD_LIFTED→TRACK_HAND→APPROACH→HANDOVER→RELEASE→RETREAT→DONE。"""
 
-    def __init__(self, backend, args, model):
+    def __init__(self, backend, args, model, target=None):
         super().__init__("servo_delivery")
         self.backend = backend
         self.args = args
         self.model = model
-        self.target = build_target(args, model.clone())
+        self.target = target if target is not None else build_target(args, model.clone())
         self.servo = PBVSPoseServo(params=self._servo_params(), model=model.clone())
         self.safety = ExecutionSafetyMonitor(params=SafetyParams())
         self.gripper = None
@@ -161,6 +161,9 @@
         self.stable_frames = 0
         self.retreat_target = None
 
+        self._latest_depth = None
+        self._consumed_sequence = None
+        self._release_samples = 0
         self._latest_image = None
         self._latest_info = None
         self._latest_q = None
@@ -205,6 +208,10 @@
     def _detect_latest(self, q_now, now):
         if self._latest_image is None or self._latest_info is None:
             return None
+        sequence = getattr(self, '_sdk_frame_sequence', None)
+        if sequence == self._consumed_sequence or time.monotonic() - getattr(self, '_sdk_received_monotonic', 0.) > .45:
+            return None
+        self._consumed_sequence = sequence
         K = np.asarray(self._latest_info.k, dtype=np.float64).reshape(3, 3)
         D = np.asarray(self._latest_info.d, dtype=np.float64).reshape(-1)
         bgr = image_msg_to_bgr(self._latest_image)
@@ -332,6 +339,7 @@
     def _fail_closed(self, reason):
         self.get_logger().error(f"[FAIL-CLOSED] {reason}")
         try:
+            self.backend.emergency_stop()
             self.backend.disable_cartesian_impedance()
         except Exception:
             pass
@@ -342,16 +350,23 @@
             self.backend.disconnect()
         except Exception:
             pass
-        rclpy.shutdown()
+        sdk_loop.shutdown()
         raise SystemExit(exit_code)
 
     # ---------- 主循环 ----------
     def _on_timer(self):
         now = time.time()
         q_now = self.backend.get_joint_angles()
-        if q_now is None or len(q_now) != 7:
+        if q_now is None or len(q_now) != 7 or not np.isfinite(q_now).all():
+            self._fail_closed('Missing or invalid joint feedback')
             return
         q_now = np.asarray(q_now, dtype=np.float64).ravel()
+        if not isinstance(self.backend, FakeBackend):
+            feedback = self.backend.get_robot().get_joint_angles()
+            stamp = float(getattr(feedback, 'timestamp', 0.0))
+            if not 0 <= time.time() - stamp <= .5:
+                self._fail_closed('Stale SDK joint feedback')
+                return
 
         debug = self.backend.get_debug_state() or {}
         if debug.get("last_error"):
@@ -369,6 +384,22 @@
                 self._engage(q_now, now)
             else:
                 self.state, self.state_entered = "TRACK_HAND", now
+            return
+
+        if self.state == "RELEASE":
+            if now - self.state_entered > 5.0:
+                self._fail_closed('Gripper release feedback timeout')
+                return
+            opened = self.gripper is None
+            if self.gripper is not None:
+                status = self.gripper.status()
+                msg = getattr(status, 'msg', None)
+                width = float(getattr(msg, 'value', float('nan')))
+                opened = (getattr(msg, 'mode', '') == 'width' and np.isfinite(width) and width >= .080)
+            self._release_samples = self._release_samples + 1 if opened else 0
+            if self._release_samples >= 3:
+                self.state, self.state_entered = 'RETREAT', now
+                self.retreat_target = None
             return
 
         if self.state == "RETREAT":
@@ -383,7 +414,12 @@
         target = self._detect_latest(q_now, now)
         if target is None:
             if now - self._last_target_time > 0.45:
-                self.get_logger().warn("target lost; holding last impedance target")
+                self.stable_frames = 0
+                if self._imp_cfg is not None:
+                    self._imp_cfg.target_twist = np.zeros(6)
+                    self._imp_cfg.follow_kp = np.zeros(7)
+                    self.backend.set_cartesian_impedance(self._imp_cfg)
+                self.get_logger().warning("target lost; holding last impedance target")
                 # 不刷新名义目标：内环柔顺保持最后一个目标
             return
         self._last_target_time = now
@@ -404,11 +440,12 @@
 
         if self.state == "TRACK_HAND" and self.stable_frames >= int(self.args.stable_frames):
             self.state, self.state_entered, self.stable_frames = "APPROACH", now, 0
+            self.servo.params.desired_distance = float(self.args.handover_distance)
             self.get_logger().info("hand target acquired; approaching")
-        elif self.state == "APPROACH" and err_pos < float(self.args.handover_distance):
+        elif self.state == "APPROACH" and ok_err and np.linalg.norm(center - np.asarray(self.model.fk(q_now)[0])) <= float(self.args.handover_distance) + .01:
             self.state, self.state_entered, self.stable_frames = "HANDOVER", now, 0
             self.get_logger().info("at handover distance; waiting for stable hand")
-        elif self.state == "HANDOVER" and self.stable_frames >= int(self.args.handover_stable_frames):
+        elif self.state == "HANDOVER" and self.stable_frames >= int(self.args.handover_stable_frames) and np.linalg.norm(center - np.asarray(self.model.fk(q_now)[0])) <= float(self.args.handover_distance) + .01:
             self.state, self.state_entered = "RELEASE", now
             self.get_logger().info("hand stable; releasing gripper")
             if self.gripper is not None:
@@ -421,73 +458,67 @@
 
     def _fixed_step(self, q_now, target_tcp, now):
         """RETREAT：沿固定目标步进（复用 PBVS，目标固定）。"""
-        servo_out = self.servo.update(q_now, target_tcp, None, None, None, now)
-        self._refresh(q_now, servo_out, now)
+        distance = self.servo.params.desired_distance
+        try:
+            self.servo.params.desired_distance = 0.0
+            servo_out = self.servo.update(q_now, target_tcp, None, None, None, now)
+        finally:
+            self.servo.params.desired_distance = distance
+        decision = self.safety.evaluate(q_now, servo_out, now)
+        if decision.fatal:
+            self._fail_closed(decision.reason)
+        elif not decision.hold:
+            self._refresh(q_now, servo_out, now)
 
 
 def parse_servo_args(argv):
-    p = argparse.ArgumentParser(description="servo delivery (wrist camera + PBVS + impedance)")
-    p.add_argument("--execute", action="store_true", help="真实执行门槛（缺省 dry-run）")
-    p.add_argument("--target-source", choices=["aruco", "mediapipe"], default="aruco")
-    p.add_argument("--wrist-image-topic", default="/camera/camera/color/image_raw")
-    p.add_argument("--wrist-info-topic", default="/camera/camera/color/camera_info")
-    p.add_argument("--handeye", default="/home/dpf/桌面/RealRobot_Test/real_test/configs/2026-05-19_14-45-49_calibration.json",
-                   help="腕部相机手眼标定（伺服工程已验证的那套）")
-    p.add_argument("--handeye-convention", default="link7_cam")
-    p.add_argument("--handeye-mount-frame", default="feedback_link7")
-    p.add_argument("--offset-base", default="0.02,-0.025,0")
-    p.add_argument("--aruco-dict", default="ARUCO_ROS_ALL")
-    p.add_argument("--marker-id", type=int, default=582)
-    p.add_argument("--marker-size", type=float, default=0.0677)
-    p.add_argument("--servo-hz", type=float, default=30.0)
-    p.add_argument("--delivery-distance", type=float, default=0.15)
-    p.add_argument("--max-linear-vel", type=float, default=0.03)
-    p.add_argument("--max-angular-vel", type=float, default=0.6)
-    p.add_argument("--max-joint-vel", type=float, default=0.06)
-    p.add_argument("--visual-orientation", action="store_true", default=False)
-    p.add_argument("--approach-error-m", type=float, default=0.03)
-    p.add_argument("--approach-error-deg", type=float, default=5.0)
-    p.add_argument("--handover-distance", type=float, default=0.10)
-    p.add_argument("--stable-frames", type=int, default=10)
-    p.add_argument("--handover-stable-frames", type=int, default=20)
-    p.add_argument("--impedance-max-position-error", type=float, default=0.015)
-    p.add_argument("--impedance-max-orientation-error-deg", type=float, default=5.0)
-    p.add_argument("--impedance-target-linear-vel", type=float, default=0.03)
-    p.add_argument("--impedance-target-angular-vel", type=float, default=0.5)
-    p.add_argument("--cart-k-scale", type=float, default=1.0)
-    p.add_argument("--cart-b-scale", type=float, default=1.0)
-    p.add_argument("--follow-kp", type=float, default=0.0,
-                   help="固件关节跟随弹簧（逗号分隔 7 个或单值）")
-    p.add_argument("--follow-min-qdot", type=float, default=0.01)
-    p.add_argument("--dls-lambda", type=float, default=0.05)
-    p.add_argument("--use-dls", action="store_true", default=False)
-    p.add_argument("--vel-filter-tau", type=float, default=0.008)
+    from real_nero_grasp.servo_cli import servo_parser
+    from cameras.options import add_camera_arguments
+    p = argparse.ArgumentParser(parents=[servo_parser()])
+    add_camera_arguments(p)
+    from cameras.options import apply_pipeline_defaults
+    apply_pipeline_defaults(p, argv)
     return p.parse_args(argv)
 
 
+def validate_servo_args(args):
+    from pathlib import Path
+    if not getattr(args, 'handeye', None) or not Path(args.handeye).is_file():
+        raise ValueError('Provide --handeye pointing to verified D435 wrist calibration')
+    if args.servo_hz <= 0:
+        raise ValueError('servo-hz must be positive')
+
+
 def run_servo_delivery(args, backend=None):
-    if backend is None:
-        if getattr(args, "execute", False):
-            from handoff.grasp_to_servo_handoff import build_servo_backend
-            backend = build_servo_backend()
-        else:
-            backend = FakeBackend()
-    model = NeroPinocchioModel()
-    rclpy.init(args=[])
-    node = ServoDeliveryNode(backend, args, model)
+    from cameras.realsense import camera_for, close_all
+    node = None
     try:
-        rclpy.spin(node)
+        validate_servo_args(args)
+        model = NeroPinocchioModel()
+        # Parse calibration and initialize detection before touching the arm.
+        target = build_target(args, model.clone())
+        camera_for(args).read()
+        if backend is None:
+            if getattr(args, 'execute', False):
+                from handoff.grasp_to_servo_handoff import build_servo_backend
+                backend = build_servo_backend(channel=getattr(args, 'nero_channel', 'can0'),
+                    firmware=getattr(args, 'nero_firmware', 'V111').lower())
+            else:
+                backend = FakeBackend()
+        sdk_loop.init(args=[])
+        node = ServoDeliveryNode(backend, args, model, target=target)
+        sdk_loop.spin(node)
     except KeyboardInterrupt:
-        pass
+        return 130
+    except Exception:
+        if backend is not None and not isinstance(backend, FakeBackend):
+            backend.emergency_stop()
+        raise
     finally:
-        try:
+        sdk_loop.shutdown()
+        if backend is not None:
             backend.disconnect()
-        except Exception:
-            pass
-        try:
-            rclpy.shutdown()
-        except Exception:
-            pass
+        close_all()
     return 0
 
 

```

#### targets/hand_target.py

```diff
--- a/targets/hand_target.py
+++ b/targets/hand_target.py
@@ -84,7 +84,7 @@
         p_index = _to3d(5)
         p_middle = _to3d(9)
         p_pinky = _to3d(17)
-        if None in (p_wrist, p_index, p_middle, p_pinky):
+        if any(p is None for p in (p_wrist, p_index, p_middle, p_pinky)):
             return None
 
         # 掌心中心：手腕点（递送时最稳定）

```

#### tests/test_sdk_camera_session.py

```diff
--- /dev/null
+++ b/tests/test_sdk_camera_session.py
@@ -0,0 +1,100 @@
+"""SDK lifecycle/transport regression without USB or robot access."""
+import sys
+import time
+from types import SimpleNamespace
+import numpy as np
+import pytest
+from cameras import realsense, sdk_loop
+
+
+def frame(seq=1):
+    intr=SimpleNamespace(fx=100.,fy=100.,cx=0.,cy=0.)
+    return realsense.Frame(np.zeros((2,2,3),np.uint8),np.full((2,2),.5,np.float32),
+                          intr,np.eye(3),np.zeros(5),seq,10.,time.monotonic(),'test')
+
+
+def test_registry_reuses_camera_across_shallow_copied_args():
+    from copy import copy
+    args=SimpleNamespace(camera_serial='offline')
+    try:
+        assert realsense.camera_for(args) is realsense.camera_for(copy(args))
+        assert realsense.camera_for(args).pipeline is None
+    finally:
+        realsense.close_all()
+
+
+def test_median_capture_and_organized_xyz_use_same_metric_contract(monkeypatch,tmp_path):
+    class Fake:
+        def read(self): return frame()
+    monkeypatch.setattr(realsense,'camera_for',lambda args:Fake())
+    args=SimpleNamespace(save_dir=str(tmp_path),orbbec_depth_median_frames=3,
+                         orbbec_depth_median_min_valid=2)
+    bgr,cloud,intr,out=realsense.organized_cloud(args)
+    from real_nero_grasp.depth_processing import pointcloud2_xyz_image
+    xyz=pointcloud2_xyz_image(cloud)
+    np.testing.assert_allclose(xyz[1,1],[.005,.005,.5])
+    assert bgr.shape==(2,2,3)
+    assert (tmp_path / out).is_dir()
+
+
+def test_warmup_failure_stops_started_pipeline(monkeypatch):
+    class Device:
+        def get_info(self,key): return 'D435i' if key=='name' else 'serial'
+    calls=[]
+    class Pipe:
+        def start(self,cfg): return SimpleNamespace(get_device=lambda:SimpleNamespace(
+            first_depth_sensor=lambda:SimpleNamespace(get_depth_scale=lambda:.001)))
+        def wait_for_frames(self,ms): raise RuntimeError('timeout')
+        def stop(self): calls.append('stopped')
+    cfg=SimpleNamespace(enable_device=lambda *a:None,enable_stream=lambda *a:None)
+    fake=SimpleNamespace(context=lambda:SimpleNamespace(query_devices=lambda:[Device()]),
+        camera_info=SimpleNamespace(name='name',serial_number='serial'),
+        config=lambda:cfg,pipeline=lambda ctx:Pipe(),align=lambda x:None,
+        stream=SimpleNamespace(color='color',depth='depth'),format=SimpleNamespace(bgr8=1,z16=2))
+    monkeypatch.setitem(sys.modules,'pyrealsense2',fake)
+    camera=realsense.RealSenseCamera(warmup=1)
+    with pytest.raises(RuntimeError,match='timeout'): camera.start()
+    assert camera.pipeline is None and calls==['stopped']
+
+
+def test_sdk_loop_delivers_color_depth_and_info_without_ros(monkeypatch):
+    class Fake:
+        timeout=.1
+        closed=False
+        def read(self):
+            time.sleep(.005)
+            return frame()
+        def close(self): self.closed=True
+    camera=Fake()
+    monkeypatch.setattr(sdk_loop,'camera_for',lambda args:camera)
+    node=sdk_loop.Node('test');node.args=SimpleNamespace()
+    received=[]
+    node.create_subscription(sdk_loop.Image,'color',lambda img:received.append(img))
+    def tick():
+        if received: sdk_loop.shutdown()
+    node.create_timer(.01,tick)
+    sdk_loop.init()
+    sdk_loop.spin(node)
+    assert camera.closed and len(received)==1
+    assert received[0].width==2
+    np.testing.assert_allclose(node._latest_depth,.5)
+
+
+def test_config_cli_overrides_yaml(tmp_path):
+    import argparse
+    from cameras.options import add_camera_arguments,apply_pipeline_defaults
+    path=tmp_path/'pipeline.yaml'
+    path.write_text('wrist_camera:\n  width: 848\n')
+    args=['--config',str(path),'--camera-width','640']
+    p=argparse.ArgumentParser();add_camera_arguments(p);apply_pipeline_defaults(p,args)
+    assert p.parse_args(args).camera_width==640
+
+
+def test_handoff_stale_feedback_does_not_suppress_exit_stop():
+    from handoff.grasp_to_servo_handoff import stop_grasp_control
+    feedback=SimpleNamespace(timestamp=0.,msg=[0.]*7)
+    arm=SimpleNamespace(get_joint_angles=lambda:[0.]*7,
+        terminate_motion=lambda **kw:None,
+        arm=SimpleNamespace(get_joint_angles=lambda:feedback))
+    assert stop_grasp_control(arm) is False
+    assert not getattr(arm,'_suppress_program_exit_estop',False)

```

#### tests_servo/test_image_conversion.py

```diff
--- a/tests_servo/test_image_conversion.py
+++ b/tests_servo/test_image_conversion.py
@@ -3,7 +3,7 @@
 
 import numpy as np
 import numpy.testing as npt
-from sensor_msgs.msg import Image
+from cameras.sdk_loop import Image
 
 sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
 

```

#### tools/nero_handeye_calib_pose_sweep.py

```diff
--- a/tools/nero_handeye_calib_pose_sweep.py
+++ b/tools/nero_handeye_calib_pose_sweep.py
@@ -51,57 +51,19 @@
 
 
 class _HandeyePosePublisher:
+    """Local pose JSONL recorder; old topic argument is now an output file path."""
     def __init__(self, topic, hz):
-        import rclpy
-        from geometry_msgs.msg import Pose
-        from rclpy.node import Node
-
-        self.rclpy = rclpy
-        self.Pose = Pose
-        self._owns_rclpy = not rclpy.ok()
-        if self._owns_rclpy:
-            rclpy.init(args=None)
-        self.node = Node("nero_handeye_pose_sweep_pose_pub")
-        self.publisher = self.node.create_publisher(Pose, topic, 10)
-        self.topic = topic
-        self.period = 1.0 / max(float(hz), 0.1)
-        self._lock = threading.Lock()
-        self._pose6 = None
-        self._running = True
-        self._thread = threading.Thread(target=self._loop, daemon=True)
-        self._thread.start()
-        print(f"[handeye_pub] publishing actual flange Pose to {topic} at {float(hz):.1f}Hz")
-
+        path = Path(topic).expanduser()
+        if path.suffix != '.jsonl':
+            raise ValueError('Use --handeye-pose-log <path.jsonl>, not a ROS topic')
+        path.parent.mkdir(parents=True, exist_ok=True)
+        self.file = path.open('x', encoding='utf-8')
     def set_pose(self, pose6):
-        with self._lock:
-            self._pose6 = np.asarray(pose6, dtype=float).reshape(6).copy()
-
-    def _make_msg(self, pose6):
-        msg = self.Pose()
-        msg.position.x = float(pose6[0])
-        msg.position.y = float(pose6[1])
-        msg.position.z = float(pose6[2])
-        quat = R.from_euler("xyz", pose6[3:6]).as_quat()
-        msg.orientation.x = float(quat[0])
-        msg.orientation.y = float(quat[1])
-        msg.orientation.z = float(quat[2])
-        msg.orientation.w = float(quat[3])
-        return msg
-
-    def _loop(self):
-        while self._running:
-            with self._lock:
-                pose6 = None if self._pose6 is None else self._pose6.copy()
-            if pose6 is not None:
-                self.publisher.publish(self._make_msg(pose6))
-            time.sleep(self.period)
-
+        pose = np.asarray(pose6, dtype=float).reshape(6)
+        self.file.write(json.dumps({'wall_time': time.time(), 'flange_pose': pose.tolist()}) + '\n')
+        self.file.flush()
     def close(self):
-        self._running = False
-        self._thread.join(timeout=1.0)
-        self.node.destroy_node()
-        if self._owns_rclpy and self.rclpy.ok():
-            self.rclpy.shutdown()
+        self.file.close()
 
 
 def _camera_look_at_T(cam_pos_base, target_base, roll_deg=0.0):
@@ -1334,7 +1296,7 @@
                 if i == len(planned):
                     ans = input(
                         f"[{label}] Capture the FINAL calibration sample now. "
-                        "Wait until handeye_calibration prints piper [ok], then press Enter to finish; q to stop: "
+                        "Confirm the SDK sample and pose log are saved, then Enter to finish; q to stop: "
                     ).strip().lower()
                     if pose_publisher is not None:
                         _sleep_with_message(
@@ -1403,7 +1365,7 @@
     p.add_argument("--no-auto-base-azimuth", dest="auto_base_azimuth", action="store_false", help="Use only --base-azimuth-deg")
     p.set_defaults(auto_base_azimuth=True)
     p.add_argument("--save-dir", default="handeye_pose_sweep_logs")
-    p.add_argument("--handeye-pose-pub-topic", default=None, help="Publish actual Nero flange pose as geometry_msgs/Pose for handeye_calibration_ros piper_topic")
+    p.add_argument("--handeye-pose-log", dest="handeye_pose_pub_topic", default=None, help="Save actual flange poses to a new JSONL file; no ROS publisher")
     p.add_argument("--handeye-pose-pub-hz", type=float, default=10.0)
     p.add_argument("--pose-publish-settle-sec", type=float, default=0.3, help="Short delay after each move so hand-eye pose topic publishes a few samples before capture")
     p.add_argument("--final-publish-hold-sec", type=float, default=5.0, help="Keep publishing the final pose for a short time before shutting down")
@@ -1464,6 +1426,8 @@
     p.add_argument("--correction-attempts", type=int, default=1)
     p.add_argument("--correct-pos-tol", type=float, default=0.006)
     p.add_argument("--correct-rpy-tol-deg", type=float, default=3.0)
+    from cameras.options import add_camera_arguments
+    add_camera_arguments(p)
     return p.parse_args()
 
 

```

#### tools/real_nero_point_test.py

```diff
--- a/tools/real_nero_point_test.py
+++ b/tools/real_nero_point_test.py
@@ -225,18 +225,9 @@
         if args.rgb is None or args.depth is None:
             raise ValueError("Provide both --rgb and --depth for offline mode")
         return load_offline_rgbd(args)
-    return capture_ros_rgbd_frame(
-        args.save_dir,
-        args.orbbec_color_topic,
-        args.orbbec_depth_topic,
-        args.orbbec_camera_info_topic,
-        args.orbbec_depth_scale,
-        args.orbbec_timeout,
-        args.orbbec_warmup_frames,
-        getattr(args, "orbbec_depth_median_frames", 1),
-        getattr(args, "orbbec_depth_median_min_valid", 1),
-        getattr(args, "orbbec_depth_median_max_mad", None),
-    )
+    from cameras.realsense import capture_rgbd as capture_sdk
+    return capture_sdk(args)
+
 
 
 def _pointcloud2_field_dtype(datatype):
@@ -283,106 +274,9 @@
 
 
 def capture_ros_color_pointcloud_frame(args):
-    try:
-        import rclpy
-        from rclpy.node import Node
-        from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
-        from sensor_msgs.msg import CameraInfo, Image, PointCloud2
-    except ImportError as e:
-        raise ImportError(
-            "ROS2 Python packages/libraries are required for registered point cloud mode. "
-            "In the same terminal, run: source /opt/ros/humble/setup.bash, then activate/use "
-            "the Python environment for this project, and verify with: python3 -c 'import rclpy'. "
-            f"Current LD_LIBRARY_PATH={os.environ.get('LD_LIBRARY_PATH', '')!r}"
-        ) from e
-
-    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
-    out_dir = os.path.join(args.save_dir, timestamp)
-    os.makedirs(out_dir, exist_ok=True)
-
-    rclpy.init(args=None)
-
-    class ColorCloudNode(Node):
-        def __init__(self):
-            super().__init__("real_nero_point_cloud_capture")
-            qos = QoSProfile(
-                history=HistoryPolicy.KEEP_LAST,
-                depth=10,
-                reliability=ReliabilityPolicy.BEST_EFFORT,
-            )
-            self.color_msg = None
-            self.info_msg = None
-            self.cloud_msg = None
-            self.color_count = 0
-            self.cloud_count = 0
-            self.create_subscription(Image, args.orbbec_color_topic, self._on_color, qos)
-            self.create_subscription(CameraInfo, args.orbbec_camera_info_topic, self._on_info, qos)
-            self.create_subscription(PointCloud2, args.orbbec_points_topic, self._on_cloud, qos)
-
-        def _on_color(self, msg):
-            self.color_msg = msg
-            self.color_count += 1
-
-        def _on_info(self, msg):
-            self.info_msg = msg
-
-        def _on_cloud(self, msg):
-            self.cloud_msg = msg
-            self.cloud_count += 1
-
-        def ready(self):
-            warmup = max(int(args.orbbec_warmup_frames), 0)
-            return (
-                self.color_msg is not None
-                and self.info_msg is not None
-                and self.cloud_msg is not None
-                and self.color_count > warmup
-                and self.cloud_count > warmup
-            )
-
-    node = ColorCloudNode()
-    try:
-        deadline = time.time() + float(args.orbbec_timeout)
-        while rclpy.ok() and time.time() < deadline:
-            rclpy.spin_once(node, timeout_sec=0.05)
-            if node.ready():
-                break
-        if not node.ready():
-            raise TimeoutError(
-                "Timed out waiting for color/registered point cloud messages. "
-                f"color={node.color_count}, cloud={node.cloud_count}, info={int(node.info_msg is not None)}"
-            )
-
-        rgb = _ros_image_to_cv(node.color_msg)
-        k = node.info_msg.k
-        intr = Intrinsics(float(k[0]), float(k[4]), float(k[2]), float(k[5]))
-        rgb_path = os.path.join(out_dir, "color.png")
-        meta_path = os.path.join(out_dir, "capture_meta.json")
-        cv2.imwrite(rgb_path, rgb)
-        with open(meta_path, "w", encoding="utf-8") as f:
-            json.dump(
-                {
-                    "mode": "registered_pointcloud",
-                    "rgb": rgb_path,
-                    "intrinsics": {"fx": intr.fx, "fy": intr.fy, "cx": intr.cx, "cy": intr.cy},
-                    "color_topic": args.orbbec_color_topic,
-                    "camera_info_topic": args.orbbec_camera_info_topic,
-                    "points_topic": args.orbbec_points_topic,
-                    "cloud_width": int(node.cloud_msg.width),
-                    "cloud_height": int(node.cloud_msg.height),
-                    "cloud_frame_id": node.cloud_msg.header.frame_id,
-                    "color_width": int(node.color_msg.width),
-                    "color_height": int(node.color_msg.height),
-                    "timestamp": timestamp,
-                },
-                f,
-                indent=2,
-            )
-        print(f"[capture] saved color frame and registered point cloud metadata to {out_dir}")
-        return rgb, node.cloud_msg, intr, out_dir
-    finally:
-        node.destroy_node()
-        rclpy.shutdown()
+    """Compatibility name: returns SDK organized XYZ, with no ROS dependency."""
+    from cameras.realsense import organized_cloud
+    return organized_cloud(args)
 
 
 def print_status(nero, target_pose=None):
@@ -768,6 +662,8 @@
     p.add_argument("--max-y", type=float, default=0.60)
     p.add_argument("--min-z", type=float, default=0.02)
     p.add_argument("--max-z", type=float, default=1.20)
+    from cameras.options import add_camera_arguments
+    add_camera_arguments(p)
     return p.parse_args()
 
 

```

#### tools/record_nero_observation_pose_seeds.py

```diff
--- a/tools/record_nero_observation_pose_seeds.py
+++ b/tools/record_nero_observation_pose_seeds.py
@@ -18,6 +18,7 @@
 import cv2
 import numpy as np
 
+from cameras.realsense import capture_rgbd as capture_sdk_rgbd
 from tools.real_nero_point_test import NeroArm
 
 
@@ -49,15 +50,7 @@
 def _capture_aruco(args):
     from tools.verify_handeye_point import capture_ros_rgbd_frame
 
-    bgr, _, _, out_dir = capture_ros_rgbd_frame(
-        args.save_dir,
-        args.orbbec_color_topic,
-        args.orbbec_depth_topic,
-        args.orbbec_camera_info_topic,
-        args.orbbec_depth_scale,
-        args.orbbec_timeout,
-        args.orbbec_warmup_frames,
-    )
+    bgr, _, _, out_dir = capture_sdk_rgbd(args)
     visible, msg = _detect_aruco(bgr, args.aruco_dict, args.aruco_id)
     return visible, msg, out_dir
 
@@ -86,6 +79,8 @@
     p.add_argument("--orbbec-depth-scale", type=float, default=None)
     p.add_argument("--orbbec-timeout", type=float, default=5.0)
     p.add_argument("--orbbec-warmup-frames", type=int, default=5)
+    from cameras.options import add_camera_arguments
+    add_camera_arguments(p)
     return p.parse_args()
 
 

```

#### tools/test_d435i_grasp_pointcloud.py

```diff
--- a/tools/test_d435i_grasp_pointcloud.py
+++ b/tools/test_d435i_grasp_pointcloud.py
@@ -107,8 +107,8 @@
         print("Fewer than 3000 points: default GraspNet sampling will duplicate points")
     import open3d as o3d
     for name, cloud in (("raw", raw), ("filtered", filtered), ("scene", scene)):
-        if not o3d.io.write_point_cloud(str(args.output / (name  ".ply")), cloud):
-            raise OSError("PLY write failed: "  name)
+        if not o3d.io.write_point_cloud(str(args.output / (name + ".ply")), cloud):
+            raise OSError("PLY write failed: " + name)
     if not cv2.imwrite(str(args.output / "selected_mask.png"), mask.astype(np.uint8) * 255):
         raise OSError("mask write failed")
     if args.view:

```

#### tools/verify_handeye_aruco_consistency.py

```diff
--- a/tools/verify_handeye_aruco_consistency.py
+++ b/tools/verify_handeye_aruco_consistency.py
@@ -21,6 +21,7 @@
 import numpy as np
 from scipy.spatial.transform import Rotation as R
 
+from cameras.realsense import capture_rgbd as capture_sdk_rgbd
 from tools.verify_handeye_point import (
     capture_ros_rgbd_frame,
     load_calibration,
@@ -149,6 +150,8 @@
     p.add_argument("--nero-firmware", default="V111", choices=["DEFAULT", "V111", "V112"])
     p.add_argument("--nero-enable-timeout", type=float, default=5.0)
     p.add_argument("--nero-speed-percent", type=int, default=3)
+    from cameras.options import add_camera_arguments
+    add_camera_arguments(p)
     return p.parse_args()
 
 
@@ -178,15 +181,7 @@
             if ans == "q":
                 break
 
-            bgr, depth_m, intr, out_dir = capture_ros_rgbd_frame(
-                args.save_dir,
-                args.orbbec_color_topic,
-                args.orbbec_depth_topic,
-                args.orbbec_camera_info_topic,
-                args.orbbec_depth_scale,
-                args.orbbec_timeout,
-                args.orbbec_warmup_frames,
-            )
+            bgr, depth_m, intr, out_dir = capture_sdk_rgbd(args)
             try:
                 u, v, detected_id, corners, ids = _detect_aruco_center(bgr, args.aruco_dict, args.aruco_id)
                 depth = robust_depth_at(depth_m, u, v, args.depth_radius)

```

#### tools/verify_handeye_point.py

```diff
--- a/tools/verify_handeye_point.py
+++ b/tools/verify_handeye_point.py
@@ -10,6 +10,10 @@
 
 If --u/--v are omitted, click one point in the RGB image and press Enter.
 """
+
+import sys
+from pathlib import Path
+sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
 
 import argparse
 import json
@@ -237,175 +241,10 @@
     raise ValueError(f"Unsupported ROS image encoding: {msg.encoding}")
 
 
-def capture_ros_rgbd_frame(
-    save_dir,
-    color_topic,
-    depth_topic,
-    camera_info_topic,
-    depth_scale,
-    timeout_s,
-    warmup_frames,
-    depth_median_frames=1,
-    depth_median_min_valid=1,
-    depth_median_max_mad=None,
-):
-    try:
-        import rclpy
-        from rclpy.node import Node
-        from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
-        from sensor_msgs.msg import CameraInfo, Image
-    except ImportError as e:
-        raise ImportError("ROS2 Python packages are required for --live-orbbec. Source /opt/ros/humble/setup.bash first.") from e
-
-    os.makedirs(save_dir, exist_ok=True)
-    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
-    out_dir = os.path.join(save_dir, timestamp)
-    os.makedirs(out_dir, exist_ok=True)
-
-    owns_rclpy = not rclpy.ok()
-    if owns_rclpy:
-        rclpy.init(args=None)
-
-    class RgbdCaptureNode(Node):
-        def __init__(self):
-            super().__init__("verify_handeye_rgbd_capture")
-            qos = QoSProfile(
-                history=HistoryPolicy.KEEP_LAST,
-                depth=10,
-                reliability=ReliabilityPolicy.BEST_EFFORT,
-            )
-            self.color_msg = None
-            self.depth_msg = None
-            self.depth_msgs = []
-            self.info_msg = None
-            self.color_count = 0
-            self.depth_count = 0
-            self.info_count = 0
-            self.create_subscription(Image, color_topic, self._on_color, qos)
-            self.create_subscription(Image, depth_topic, self._on_depth, qos)
-            self.create_subscription(CameraInfo, camera_info_topic, self._on_info, qos)
-
-        def _on_color(self, msg):
-            self.color_msg = msg
-            self.color_count += 1
-
-        def _on_depth(self, msg):
-            self.depth_msg = msg
-            self.depth_count += 1
-            warmup = max(int(warmup_frames), 0)
-            wanted = max(int(depth_median_frames), 1)
-            if self.depth_count > warmup and len(self.depth_msgs) < wanted:
-                self.depth_msgs.append(msg)
-
-        def _on_info(self, msg):
-            self.info_msg = msg
-            self.info_count += 1
-
-        def ready(self):
-            warmup = max(int(warmup_frames), 0)
-            wanted = max(int(depth_median_frames), 1)
-            return (
-                self.color_msg is not None
-                and self.depth_msg is not None
-                and self.info_msg is not None
-                and self.color_count > warmup
-                and self.depth_count > warmup
-                and len(self.depth_msgs) >= wanted
-            )
-
-    node = RgbdCaptureNode()
-    try:
-        deadline = time.time() + float(timeout_s)
-        while rclpy.ok() and time.time() < deadline:
-            rclpy.spin_once(node, timeout_sec=0.05)
-            if node.ready():
-                break
-        if not node.ready():
-            raise TimeoutError(
-                "Timed out waiting for ROS RGB-D messages. "
-                f"color={node.color_count}, depth={node.depth_count}, camera_info={node.info_count}"
-            )
-
-        rgb = _ros_image_to_cv(node.color_msg)
-        depth_raw_frames = [_ros_image_to_cv(msg) for msg in node.depth_msgs]
-        depth_raw = depth_raw_frames[-1]
-        scale = float(depth_scale) if depth_scale is not None else (0.001 if np.issubdtype(depth_raw.dtype, np.integer) else 1.0)
-        depth_frames_m = [frame.astype(np.float32) * scale for frame in depth_raw_frames]
-        depth_m, depth_valid_count, temporal_mad, unstable_depth = median_valid_depth_frames(
-            depth_frames_m,
-            min_valid_frames=depth_median_min_valid,
-            max_mad=depth_median_max_mad,
-            return_stats=True,
-        )
-        per_frame_valid_ratio = [
-            float(np.count_nonzero(np.isfinite(frame) & (frame > 0.0))) / float(frame.size)
-            for frame in depth_frames_m
-        ]
-        median_valid_ratio = float(np.count_nonzero(depth_m > 0.0)) / float(depth_m.size)
-
-        k = node.info_msg.k
-        intr = Intrinsics(float(k[0]), float(k[4]), float(k[2]), float(k[5]))
-
-        rgb_path = os.path.join(out_dir, "color.png")
-        depth_npy_path = os.path.join(out_dir, "depth_m.npy")
-        depth_png_path = os.path.join(out_dir, "depth_mm.png")
-        depth_valid_count_path = os.path.join(out_dir, "depth_valid_count.png")
-        depth_temporal_mad_path = os.path.join(out_dir, "depth_temporal_mad_mm.png")
-        depth_temporal_unstable_path = os.path.join(out_dir, "depth_temporal_unstable.png")
-        meta_path = os.path.join(out_dir, "capture_meta.json")
-
-        cv2.imwrite(rgb_path, rgb)
-        np.save(depth_npy_path, depth_m)
-        cv2.imwrite(depth_png_path, np.clip(depth_m * 1000.0, 0, 65535).astype(np.uint16))
-        cv2.imwrite(depth_valid_count_path, depth_valid_count)
-        cv2.imwrite(
-            depth_temporal_mad_path,
-            np.clip(temporal_mad * 1000.0, 0, 65535).astype(np.uint16),
-        )
-        cv2.imwrite(depth_temporal_unstable_path, unstable_depth.astype(np.uint8) * 255)
-        with open(meta_path, "w", encoding="utf-8") as f:
-            json.dump(
-                {
-                    "rgb": rgb_path,
-                    "depth_m_npy": depth_npy_path,
-                    "depth_mm_png": depth_png_path,
-                    "depth_valid_count_png": depth_valid_count_path,
-                    "intrinsics": {"fx": intr.fx, "fy": intr.fy, "cx": intr.cx, "cy": intr.cy},
-                    "depth_scale": scale,
-                    "color_topic": color_topic,
-                    "depth_topic": depth_topic,
-                    "camera_info_topic": camera_info_topic,
-                    "color_encoding": node.color_msg.encoding,
-                    "depth_encoding": node.depth_msg.encoding,
-                    "depth_median_frames": int(len(depth_frames_m)),
-                    "depth_median_min_valid": int(np.clip(int(depth_median_min_valid), 1, len(depth_frames_m))),
-                    "depth_median_max_mad_m": (
-                        None
-                        if depth_median_max_mad is None
-                        else float(depth_median_max_mad)
-                    ),
-                    "depth_temporal_mad_mm_png": depth_temporal_mad_path,
-                    "depth_temporal_unstable_png": depth_temporal_unstable_path,
-                    "depth_temporal_unstable_pixels": int(np.count_nonzero(unstable_depth)),
-                    "depth_valid_ratio_per_frame": per_frame_valid_ratio,
-                    "depth_valid_ratio_median": median_valid_ratio,
-                    "timestamp": timestamp,
-                },
-                f,
-                indent=2,
-            )
-        print(
-            f"[capture] depth median frames={len(depth_frames_m)} "
-            f"min_valid={int(np.clip(int(depth_median_min_valid), 1, len(depth_frames_m)))} "
-            f"valid={median_valid_ratio:.1%} "
-            f"temporal_reject={int(np.count_nonzero(unstable_depth))}"
-        )
-        print(f"[capture] saved ROS RGB-D frame to {out_dir}")
-        return rgb, depth_m, intr, out_dir
-    finally:
-        node.destroy_node()
-        if owns_rclpy and rclpy.ok():
-            rclpy.shutdown()
+def capture_ros_rgbd_frame(*args, **kwargs):
+    """Compatibility signature used by older tools; transport is D435 SDK only."""
+    from cameras.realsense import capture_legacy
+    return capture_legacy(*args, **kwargs)
 
 
 def _response_msg(resp):
@@ -456,45 +295,7 @@
 
 
 def read_ros_pose_stamped(topic, timeout_s, label):
-    try:
-        import rclpy
-        from geometry_msgs.msg import PoseStamped
-        from rclpy.node import Node
-    except ImportError as e:
-        raise ImportError("ROS2 Python packages are required for reading /feedback/tcp_pose. Source /opt/ros/humble/setup.bash first.") from e
-
-    rclpy.init(args=None)
-
-    class PoseCaptureNode(Node):
-        def __init__(self):
-            super().__init__("verify_handeye_pose_capture")
-            self.msg = None
-            self.create_subscription(PoseStamped, topic, self._on_pose, 10)
-
-        def _on_pose(self, msg):
-            self.msg = msg
-
-    node = PoseCaptureNode()
-    try:
-        deadline = time.time() + float(timeout_s)
-        while rclpy.ok() and time.time() < deadline:
-            rclpy.spin_once(node, timeout_sec=0.05)
-            if node.msg is not None:
-                break
-        if node.msg is None:
-            raise TimeoutError(f"Timed out waiting for PoseStamped on {topic}")
-
-        p = node.msg.pose.position
-        q = node.msg.pose.orientation
-        xyz = np.array([p.x, p.y, p.z], dtype=float)
-        quat_xyzw = np.array([q.x, q.y, q.z, q.w], dtype=float)
-        rpy = R.from_quat(quat_xyzw).as_euler("xyz", degrees=False)
-        pose = np.r_[xyz, rpy]
-        print(f"[nero] {label} from {topic} =", np.round(pose, 6).tolist())
-        return pose
-    finally:
-        node.destroy_node()
-        rclpy.shutdown()
+    raise RuntimeError("ROS pose input removed. Use --read-nero-flange or --flange-pose.")
 
 
 def robust_depth_at(depth_m, u, v, radius):
@@ -557,7 +358,7 @@
 def parse_args():
     p = argparse.ArgumentParser(description="Verify Nero eye-in-hand calibration with one clicked RGB-D point.")
     p.add_argument("--live-realsense", action="store_true", help="Capture and save one aligned RGB-D frame from RealSense")
-    p.add_argument("--live-orbbec", action="store_true", help="Capture and save one RGB-D frame from Orbbec ROS2 topics")
+    p.add_argument("--live-orbbec", action="store_true", help="Deprecated alias for D435 SDK capture (330 disabled)")
     p.add_argument("--save-dir", default="handeye_verify_captures", help="Directory for live camera captures")
     p.add_argument("--rs-width", type=int, default=1280)
     p.add_argument("--rs-height", type=int, default=720)
@@ -579,8 +380,8 @@
     p.add_argument("--cx", type=float, default=None)
     p.add_argument("--cy", type=float, default=None)
     p.add_argument("--calib-json", default=None, help="Hand-eye calibration json with position and orientation")
-    p.add_argument("--read-nero-flange", action="store_true", help="Read current flange pose from Nero using pyAgxArm; falls back to ROS pose topic if pyAgxArm is unavailable")
-    p.add_argument("--read-nero-tcp-ros", action="store_true", help="Read current Nero TCP pose from a ROS PoseStamped topic")
+    p.add_argument("--read-nero-flange", action="store_true", help="Read current flange pose from Nero using pyAgxArm; SDK failure is fatal; no ROS fallback")
+    p.set_defaults(read_nero_tcp_ros=False)
     p.add_argument("--nero-ros-pose-topic", default="/feedback/tcp_pose", help="ROS PoseStamped topic for --read-nero-tcp-ros or pyAgxArm fallback")
     p.add_argument("--nero-ros-pose-timeout", type=float, default=5.0)
     p.add_argument("--nero-interface", default="socketcan")
@@ -613,17 +414,8 @@
             args.rs_warmup_frames,
         )
     elif args.live_orbbec:
-        rgb, depth_m, intr, out_dir = capture_ros_rgbd_frame(
-            args.save_dir,
-            args.orbbec_color_topic,
-            args.orbbec_depth_topic,
-            args.orbbec_camera_info_topic,
-            args.orbbec_depth_scale,
-            args.orbbec_timeout,
-            args.orbbec_warmup_frames,
-            args.orbbec_depth_median_frames,
-            args.orbbec_depth_median_min_valid,
-        )
+        from cameras.realsense import capture_rgbd
+        rgb, depth_m, intr, out_dir = capture_rgbd(args)
     else:
         if args.rgb is None or args.depth is None:
             raise ValueError("Provide --rgb and --depth, or use --live-realsense/--live-orbbec")

```



## 2026-09-15 补充：独立 D435i ArUco 腕部标定

新增 `tools/calibrate_d435i_aruco.py`，硬件调试工具放 tools；算法回归放 `tests/test_aruco_handeye_solver.py`。

- `capture --output <新目录> --serial <相机序列号> --dictionary <字典名称> --marker-id <ID> --marker-mm <黑色方块外边长毫米>`：SDK RGB-D预览，S采样，Q/Esc退出。默认 can0/V111，可指定 --channel/--firmware。不调用 enable/move/set_speed；不提供自动摆位功能。板固定，腕部在外部控制下换位并停稳；至少12个有不同位置、绕不同轴旋转的视角。不能只平移或只绕一个轴。输出图片、角点、内参、相机序列号和米/弧度法兰反馈。
- `solve --samples <采集目录/samples.jsonl> --output <新标定.json>`：以 B_i X C_i=固定板基座位姿求解，SciPy相对旋转向量配准与平移最小二乘，不依赖ROS。当前OpenCV 5.0缺少calibrateHandEye，因此使用上述求解。输出 position、orientation(xyzw)，X明确是 T_flange_camera；与抓取 flange_camera_xyzw 格式一致。结果保持 unvalidated，不自动写入pipeline配置。
- `validate --samples <另一组目录/samples.jsonl> --calibration <标定.json>`：至少3个独立新视角，相同板且板不能移动，报告固定板位姿误差(mm/deg)，拒绝直接复用训练图片。残差不是碰撞安全或绝对精度认证。单平面ArUco可能有姿态歧义，应检查图像、不同倾角及独立验证误差；需应用方确定精度容差。
- SDK反馈检查新鲜度、时间戳前进、采样前后位置/角度稳定性；仍是静止采集近似，没有实现曝光与各组CAN反馈硬同步。无法检测采样间移动后返回，采样期间必须保持静止。
- 法兰定义需与该固件和IK模型核对；不能直接把此JSON当作 link7_cam。腕部重新安装后必须重新标定。标定板字典、ID、真实边长尚待用户提供，程序不猜尺寸。
- 验证：独立合成数据恢复相机到法兰变换、米单位、独立视角残差、重复姿态退化拒绝，2 tests passed；--help通过。没有连接真实相机/机械臂进行本程序验收。

### 环境与挂载检查（本次没有安装或替换环境）

驱动570.133.07支持CUDA12.8，RTX4060 Laptop；当前torch2.14.0/torchvision0.29.0为CUDA13.0组合，导致GPU不可用。应选择官方配套torch/torchvision CUDA12.8或更低构建，在确认可下载版本和依赖后替换；GraspNet pointnet2/knn本地CUDA扩展需要针对选定torch及兼容CUDA工具链重新编译/验证。conda环境的CUDA运行库不替代宿主NVIDIA驱动，不能仅凭激活环境认为兼容。这里未给出未经验证的安装版本。

TRAC-IK仍缺失：代码要求 from trac_ik import TracIK，并要求 joint_limits/fk等接口；联网pip index versions trac-ik没有找到发行版。不能用随意同名包或ROS的trac_ik_python顶替。仍须找到原工程使用的绑定来源并验证接口，安装工作未完成。

Visual Servo/models/nero_with_gripper.urdf与本工程逐字一致；该URDF没有D435/330相机挂载定义。本次保留旧模型，没有把旧相机形状认定为D435i实测几何。两项阻抗测试仍保留失败记录与旧行为，空载交接未测试；这些均不代表允许直接整链执行。

### 本次新增文件的完整 diff

```diff
--- /dev/null
+++ b/tools/calibrate_d435i_aruco.py
@@ -0,0 +1,194 @@
+"""Stationary ArUco board, moving wrist D435: SDK capture and eye-in-hand solve.
+No enable, speed, or motion commands. Robot positioning is external/manual.
+"""
+import argparse
+import json
+import sys
+import time
+from pathlib import Path
+sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
+import cv2
+import numpy as np
+from scipy.spatial.transform import Rotation
+
+
+def transform(rotation, translation):
+    t = np.eye(4)
+    t[:3, :3] = rotation
+    t[:3, 3] = translation
+    return t
+
+
+def solve(samples):
+    if len(samples) < 12:
+        raise ValueError('Need at least 12 distinct stationary views; reserve additional views for validation')
+    b = [np.asarray(s['T_base_flange'], dtype=float) for s in samples]
+    c = [np.asarray(s['T_camera_marker'], dtype=float) for s in samples]
+    for t in b + c:
+        if t.shape != (4, 4) or not np.isfinite(t).all():
+            raise ValueError('Invalid transform')
+        if not np.allclose(t[3], [0, 0, 0, 1]) or not np.allclose(t[:3,:3].T @ t[:3,:3], np.eye(3), atol=1e-5) or np.linalg.det(t[:3,:3]) < .99:
+            raise ValueError('Not a rigid transform')
+    rotations = np.array([Rotation.from_matrix(b[0][:3,:3].T @ t[:3,:3]).as_rotvec() for t in b[1:]])
+    if np.linalg.svd(rotations, compute_uv=False)[1] < .25:
+        raise ValueError('Insufficient rotation about independent axes')
+    # AX=XB: pairwise relative motions; rotation-vector alignment (Park), then translation LS.
+    pairs = [(np.linalg.inv(b[j]) @ b[i], c[j] @ np.linalg.inv(c[i]))
+             for i in range(len(b)) for j in range(i)]
+    av = np.array([Rotation.from_matrix(a[:3,:3]).as_rotvec() for a, _ in pairs])
+    bv = np.array([Rotation.from_matrix(d[:3,:3]).as_rotvec() for _, d in pairs])
+    r = Rotation.align_vectors(av, bv)[0].as_matrix()
+    lhs = np.vstack([a[:3,:3]-np.eye(3) for a, _ in pairs])
+    rhs = np.concatenate([r @ d[:3,3]-a[:3,3] for a, d in pairs])
+    p, _, rank, _ = np.linalg.lstsq(lhs, rhs, rcond=None)
+    if rank < 3:
+        raise ValueError('Degenerate translation system')
+    x = transform(r, p)
+    if not np.isfinite(x).all():
+        raise ValueError('Hand-eye solver failed')
+    boards = [bi @ x @ ci for bi, ci in zip(b, c)]
+    reference = transform(Rotation.from_matrix(np.array([t[:3,:3] for t in boards])).mean().as_matrix(),
+                          np.mean([t[:3,3] for t in boards], axis=0))
+    return x, reference
+
+
+def residuals(samples, x, reference):
+    errors = [np.linalg.inv(reference) @ np.array(s['T_base_flange']) @ x @ np.array(s['T_camera_marker']) for s in samples]
+    return {'translation_mm': [float(np.linalg.norm(t[:3,3])*1000) for t in errors],
+            'rotation_deg': [float(Rotation.from_matrix(t[:3,:3]).magnitude()*180/np.pi) for t in errors]}
+
+
+def read_pose(arm):
+    msg = arm.get_flange_pose()
+    if msg is None:
+        raise RuntimeError('No flange feedback')
+    stamp = float(msg.timestamp)
+    pose = np.array(msg.msg, dtype=float).copy()
+    if pose.shape != (6,) or not np.isfinite(pose).all() or not 0 <= time.time()-stamp <= .3:
+        raise RuntimeError('Invalid/stale flange feedback')
+    return transform(Rotation.from_euler('xyz', pose[3:]).as_matrix(), pose[:3]), stamp
+
+
+def capture(a):
+    from cameras.realsense import RealSenseCamera
+    from pyAgxArm import AgxArmFactory, ArmModel, NeroFW, create_agx_arm_config
+    if a.marker_mm <= 0:
+        raise ValueError('marker-mm must be positive: measure black square, excluding white margin')
+    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, a.dictionary))
+    if not 0 <= a.marker_id < len(dictionary.bytesList):
+        raise ValueError('Marker ID outside dictionary')
+    detector = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())
+    half = a.marker_mm / 2000
+    obj = np.array([[-half,half,0],[half,half,0],[half,-half,0],[-half,-half,0]], np.float32)
+    out = Path(a.output)
+    out.mkdir(parents=True, exist_ok=False)
+    arm = AgxArmFactory.create_arm(create_agx_arm_config(robot=ArmModel.NERO,
+        firmeware_version=getattr(NeroFW, a.firmware), interface='socketcan', channel=a.channel, bitrate=1000000))
+    try:
+        arm.connect()
+        with RealSenseCamera(serial=a.serial) as camera, (out/'samples.jsonl').open('x') as log:
+            print('Fixed board; position wrist externally, stop and settle. S: capture; Q/Esc: quit. No robot motion commands.')
+            while True:
+                frame = camera.read()
+                corners, ids, _ = detector.detectMarkers(frame.bgr)
+                preview = frame.bgr.copy()
+                if ids is not None:
+                    cv2.aruco.drawDetectedMarkers(preview, corners, ids)
+                cv2.imshow('D435 wrist calibration: S save / Q quit', preview)
+                key = cv2.waitKey(1) & 255
+                if key in (27, ord('q')):
+                    break
+                if key != ord('s'):
+                    continue
+                try:
+                    # Stationary capture: monitor before/after fresh frames, not exposure synchronization.
+                    b0, ts0 = read_pose(arm)
+                    time.sleep(.25)
+                    camera.read()
+                    frame = camera.read()
+                    b1, ts1 = read_pose(arm)
+                    delta = np.linalg.inv(b0) @ b1
+                    if ts1 <= ts0 or np.linalg.norm(delta[:3,3]) > .001 or Rotation.from_matrix(delta[:3,:3]).magnitude() > np.deg2rad(.3):
+                        raise RuntimeError('Wrist moving or feedback not advancing; settle and retry')
+                    corners, ids, _ = detector.detectMarkers(frame.bgr)
+                    matches = [] if ids is None else np.flatnonzero(ids.flatten() == a.marker_id)
+                    if len(matches) != 1:
+                        raise RuntimeError('Expected exactly one requested marker')
+                    points = corners[matches[0]].reshape(4,2)
+                    ok, rv, tv = cv2.solvePnP(obj, points, frame.K, frame.D, flags=cv2.SOLVEPNP_ITERATIVE)
+                    if not ok or tv[2,0] <= 0:
+                        raise RuntimeError('Invalid marker pose')
+                    proj, _ = cv2.projectPoints(obj, rv, tv, frame.K, frame.D)
+                    error = float(np.sqrt(np.mean(np.sum((proj.reshape(4,2)-points)**2, axis=1))))
+                    if error > 1.5:
+                        raise RuntimeError(f'Reprojection error {error:.2f}px exceeds 1.5px')
+                    image_name = f'{time.time_ns()}.png'
+                    if not cv2.imwrite(str(out/image_name), frame.bgr):
+                        raise OSError('Image save failed')
+                    sample = dict(T_base_flange=b1.tolist(), T_camera_marker=transform(cv2.Rodrigues(rv)[0],tv.flatten()).tolist(),
+                        serial=frame.serial, dictionary=a.dictionary, marker_id=a.marker_id, marker_mm=a.marker_mm,
+                        K=frame.K.tolist(), D=frame.D.tolist(), corners=points.tolist(), image=image_name,
+                        reprojection_px=error, feedback_stamps=[ts0,ts1], frame_hardware_ms=frame.hardware_ms)
+                    log.write(json.dumps(sample)+'\n')
+                    log.flush()
+                    print('Saved', image_name, 'reprojection px', error)
+                except RuntimeError as exc:
+                    print('Rejected:', exc)
+    finally:
+        try:
+            arm.disconnect()
+        finally:
+            cv2.destroyAllWindows()
+
+
+def main(argv=None):
+    p = argparse.ArgumentParser(description=__doc__)
+    sub = p.add_subparsers(dest='command', required=True)
+    c = sub.add_parser('capture')
+    c.add_argument('--output', required=True, help='New session directory')
+    c.add_argument('--dictionary', required=True, help='e.g. DICT_4X4_50')
+    c.add_argument('--marker-id', required=True, type=int)
+    c.add_argument('--marker-mm', required=True, type=float)
+    c.add_argument('--serial', required=True)
+    c.add_argument('--channel', default='can0')
+    c.add_argument('--firmware', choices=['V111','V112'], default='V111')
+    for name in ['solve', 'validate']:
+        s = sub.add_parser(name)
+        s.add_argument('--samples', required=True)
+        if name == 'solve':
+            s.add_argument('--output', required=True)
+        else:
+            s.add_argument('--calibration', required=True)
+    a = p.parse_args(argv)
+    if a.command == 'capture':
+        return capture(a)
+    samples = [json.loads(line) for line in Path(a.samples).read_text().splitlines() if line.strip()]
+    if not samples:
+        raise ValueError('Empty samples')
+    fields = ['serial','dictionary','marker_id','marker_mm']
+    metadata = {k:samples[0][k] for k in fields}
+    if any(any(s[k] != metadata[k] for k in fields) for s in samples):
+        raise ValueError('Mixed camera/board data')
+    if a.command == 'solve':
+        x, reference = solve(samples)
+        result = dict(position=x[:3,3].tolist(), orientation=Rotation.from_matrix(x[:3,:3]).as_quat().tolist(),
+            convention='flange_camera_xyzw', parent_frame='SDK_feedback_flange', child_frame='color_optical_frame',
+            T_flange_camera=x.tolist(), T_base_marker=reference.tolist(), metadata=metadata,
+            training_images=[s['image'] for s in samples], status='unvalidated', residuals=residuals(samples,x,reference))
+        with Path(a.output).open('x') as f:
+            json.dump(result,f,indent=2)
+    else:
+        result = json.loads(Path(a.calibration).read_text())
+        if result['metadata'] != metadata:
+            raise ValueError('Calibration camera/board mismatch')
+        if set(result['training_images']) & {s['image'] for s in samples}:
+            raise ValueError('Validation requires independent held-out captures')
+        if len(samples) < 3:
+            raise ValueError('Need at least 3 held-out views')
+        report = residuals(samples,np.array(result['T_flange_camera']),np.array(result['T_base_marker']))
+        print(json.dumps(report,indent=2))
+        print('Residuals only: verify geometry and application tolerances before robot execution.')
+
+
+if __name__ == '__main__':
+    main()
```

```diff
--- /dev/null
+++ b/tests/test_aruco_handeye_solver.py
@@ -0,0 +1,26 @@
+import numpy as np
+import pytest
+from scipy.spatial.transform import Rotation
+from tools.calibrate_d435i_aruco import transform, solve, residuals
+
+
+def test_eye_in_hand_direction_and_units():
+    rng = np.random.default_rng(32)
+    x = transform(Rotation.from_euler('xyz',[.15,-.25,.1]).as_matrix(),[.03,-.04,.08])
+    board = transform(Rotation.from_euler('xyz',[.4,.1,-.2]).as_matrix(),[.5,.1,.4])
+    samples = []
+    for _ in range(16):
+        b = transform(Rotation.from_rotvec(rng.normal(0,.4,3)).as_matrix(),rng.uniform(-.2,.2,3))
+        c = np.linalg.inv(x) @ np.linalg.inv(b) @ board
+        samples.append(dict(T_base_flange=b.tolist(),T_camera_marker=c.tolist()))
+    actual, reference = solve(samples[:12])
+    np.testing.assert_allclose(actual,x,atol=1e-8)
+    report = residuals(samples[12:], actual, reference)
+    assert max(report['translation_mm']) < 1e-5
+    assert max(report['rotation_deg']) < 1e-5
+
+
+def test_repeated_views_rejected():
+    s = dict(T_base_flange=np.eye(4).tolist(),T_camera_marker=np.eye(4).tolist())
+    with pytest.raises(ValueError,match='independent axes'):
+        solve([s]*12)
```

## 2026-09-15 执行导读审查补充：未解决问题及延期边界

配套导读：[STEP4_CODE_EXECUTION_GUIDE.md](STEP4_CODE_EXECUTION_GUIDE.md)。本次仅修改文档，没有更改算法、保护逻辑或安装环境，没有执行任何机械臂动作。以下是待办记录，不能将“记录后延期”理解为实机验证已通过。

| 项目与代码位置 | 证据/影响 | 应在哪一步前解决 |
|---|---|---|
| **execute门覆盖不完整**：`real_nero_grasp/workflow.py:67`、`arm.py:98`、`session_control.py:239`；最终抓取门在`grasp_workflow.py:1354` | grasp/all即使不加execute仍进入机械臂上下文；使能、当前位置move_j保持、初始移动和张爪在最终执行门前。`cameras/preflight.py:6`无execute直接返回，也无法隔离这些动作。不能把main不加execute作为无硬件副作用dry-run | 主入口连接实臂前补统一执行门或真正只读/离线分支；不能靠“不加execute”规避。独立相机感知工具不走该臂上下文 |
| TRAC-IK绑定缺失（上次环境检查） | 原工程只有kinematic_trac_ik.py封装，未发现trac_ik实现；pip查询未找到相符发行包。默认ik_joint需要兼容TracIK构造、joint_limits、fk和IK接口 | 多视角/抓取规划前确定原绑定来源、安装并验证FK/IK；不能只复制封装 |
| GPU与扩展工具链（上次检查，用户后续变更需重验） | 驱动570.133.07支持CUDA12.8；torch2.14.0构建cu130。系统/usr/bin/nvcc为10.1。已建议torch2.7.1+cu128/torchvision0.22.1，尚未收到安装后结果。GraspNet明确要求CUDA | GraspNet推理前验证真实GPU张量运算、pointnet2/knn扩展兼容；编译时用匹配CUDA工具链。YOLO能CPU运行不代表GraspNet也能 |
| 手眼JSON待接入和核验：`config/pipeline.yaml:16`、`:18`，`runtime.py:1362` | 用户已完成标定，尚未核对文件内容；需要当前D435安装状态、米、四元数次序、变换方向、法兰定义。grasp为flange，servo为feedback_link7；不能直接共享未核对JSON。YAML只有传--config才加载 | 使用相机定位驱动机械臂、多视角融合前；纯相机坐标点云测试不依赖手眼外参 |
| 相机/工具挂载几何：`runtime.py:1868`等碰撞胶囊模型 | VS机械臂/夹爪URDF和本工程一致，但无D435相机模型。保留旧几何不能证明相机本体/线缆的碰撞间距；旧自遮挡mask已经默认关闭 | 有腕部相机的移动路径验收前核对实际安装外形、TCP、碰撞包络；空载不等于相机不会碰撞 |
| **交接触发条件不足**：`workflow.py:232`的after_grasp_lifted条件；`grasp_workflow.py:1487`附近成功返回 | hook只检查SUCCESS，没有限定结果是“物体仍夹持且完成所需抬升”。旧工作流验证模式可短支撑释放后成功，有place目标可放下后成功。安全脚本默认验证模式会影响结果 | all递送前增加明确的lifted/holding状态契约或限制不兼容模式；不能把所有SUCCESS都当作已抬升持物 |
| MIT空载交接未验收：`handoff/grasp_to_servo_handoff.py:111` | 当前顺序先停move_j确认，再MIT保持，最后断抓取driver；代码顺序不证明固件切换没有掉高。反馈聚合时间戳也不证明所有CAN分组同一时刻 | 先独立保持/停止验证，再受控空载交接，通过后才夹物体全流程 |
| 两项阻抗回归仍失败（此前记录） | target_twist限幅期望不一致；运动目标库仑补偿仍产生力矩。原工程基线也能复现，只说明不是本轮SDK迁移新引入，不能证明行为正确。用户要求暂留旧实现，本次未改测试或控制律 | 可继续离线定位和明确旧行为；主动递送前核对期望与控制律，不能用实机运动代替尚未解释的控制回归 |
| 相机曝光/机械臂反馈未硬同步 | 静止采集为近似；运动中帧和读取的关节不一定同一时刻，影响融合与伺服目标坐标 | 多视角先停稳再采集并验证重合误差；动态递送要检查延迟、帧新鲜度、反馈时间关联 |
| 递送回程尚不是完整回初始位：`servo_delivery.py:405`的RETREAT | 目前向基座+Z退让0.15m，未实现抓取初始位回程。保持最后阻抗目标的丢失目标策略、松爪80mm反馈门槛也需实机匹配 | 人手递送前独立验证退让空间、停止/目标丢失、夹爪反馈；不能宣称整套“递送后自动回初始位”已实现 |
| 单平面ArUco标定与验证边界 | 独立程序2项合成测试通过不等于用户实际JSON已验证；单平面PnP姿态歧义、板尺寸错误、手眼旋转激励不足会导致外参偏差 | 实臂定位前用独立静止视角/已知位置验证；不需要因更换文件路径重新采集，但需确认原标定有效 |

此前记录的514 passed / 5 failed是当时测试快照，不是本次重新运行结果。本次文档审查没有重跑完整回归，也没有接受硬件风险或绕过任何失败项。


## 2026-09-15 D435i保存功能已实施

新增 --save-dir 和RGB窗口S键保存；输出目标原始/过滤PLY、彩深、mask、overlay、内参及设备/帧元数据。无有效目标时拒绝保存且清除旧云显示；不足点数不再终止预览。生成器在退出时显式关闭，释放相机。未调用机械臂或自动运行GraspNet。

验证：--help、语法检查通过；合成数据保存/读回检查深度、PLY坐标与颜色、米单位及7项文件通过。未进行实机相机窗口验收。使用方式见STEP4末尾。

```diff
--- a/scripts/test_D435i.py
+++ b/scripts/test_D435i.py
@@ -1,7 +1,10 @@
 #!/usr/bin/env python3
 """D435i SDK RGB-D -> YOLO segmentation -> target cloud (no ROS/CAN)."""
 import argparse
+import json
 import sys
+from contextlib import closing
+from datetime import datetime
 from pathlib import Path
 from types import SimpleNamespace
 
@@ -45,27 +48,99 @@
             fs=align.process(pipe.wait_for_frames(10000)); color,depth=fs.get_color_frame(),fs.get_depth_frame()
             if not color or not depth: continue
             bgr=np.asanyarray(color.get_data()).copy(); z=np.asanyarray(depth.get_data()).astype(np.float32)*float(scale); ci=color.profile.as_video_stream_profile().intrinsics
-            yield bgr,z,SimpleNamespace(fx=float(ci.fx),fy=float(ci.fy),cx=float(ci.ppx),cy=float(ci.ppy))
+            yield bgr,z,SimpleNamespace(fx=float(ci.fx),fy=float(ci.fy),cx=float(ci.ppx),cy=float(ci.ppy),
+                distortion_model=str(ci.model), distortion_coeffs=list(ci.coeffs),
+                serial=d.get_info(rs.camera_info.serial_number),
+                color_frame_number=color.get_frame_number(), depth_frame_number=depth.get_frame_number(),
+                color_timestamp_ms=color.get_timestamp(), depth_timestamp_ms=depth.get_timestamp())
     finally: pipe.stop()
 
+def save_snapshot(args, bgr, depth, intr, overlay, mask, instance, raw_cloud, target_cloud):
+    """Save one matching RGB-D/mask/cloud set; publish directory only after success."""
+    stamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S_%f')
+    root = Path(args.save_dir).expanduser()
+    root.mkdir(parents=True, exist_ok=True)
+    pending = root / (stamp + '.partial')
+    pending.mkdir(exist_ok=False)
+    for name, array in [('color.png', bgr), ('overlay.png', overlay),
+                        ('target_mask.png', (np.asarray(mask) > 0).astype(np.uint8) * 255)]:
+        if not cv2.imwrite(str(pending / name), array):
+            raise OSError(f'Failed to save {name}')
+    np.save(pending / 'depth_m.npy', depth)
+    for name, points in [('target_raw.ply', raw_cloud), ('target.ply', target_cloud)]:
+        if not o3d.io.write_point_cloud(str(pending / name), points):
+            raise OSError(f'Failed to save {name}')
+    metadata = dict(
+        coordinate_frame='color_optical_frame', length_unit='meter',
+        depth_aligned_to='color', color_order='RGB in PLY; BGR in OpenCV arrays',
+        width=int(bgr.shape[1]), height=int(bgr.shape[0]), intrinsics=vars(intr),
+        K=[[intr.fx, 0, intr.cx], [0, intr.fy, intr.cy], [0, 0, 1]],
+        label=instance['label'], score=float(instance['score']),
+        bbox_xyxy=np.asarray(instance['xyxy']).tolist(),
+        raw_points=len(raw_cloud.points), target_points=len(target_cloud.points),
+        weights=str(Path(args.weights).resolve()), min_depth=args.min_depth,
+        max_depth=args.max_depth, voxel_size=args.voxel_size,
+        outlier_filter=args.target_outlier_filter,
+    )
+    (pending / 'capture_meta.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
+    destination = root / stamp
+    pending.rename(destination)
+    return destination.resolve()
+
 def main():
-    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--weights',default='models/yolo26m-seg.pt'); p.add_argument('--target-class',required=True); p.add_argument('--target-index',type=int,default=0); p.add_argument('--width',type=int,default=640); p.add_argument('--height',type=int,default=480); p.add_argument('--fps',type=int,default=30); p.add_argument('--warmup',type=int,default=30); p.add_argument('--yolo-imgsz',type=int,default=640); p.add_argument('--yolo-conf',type=float,default=.25); p.add_argument('--min-depth',type=float,default=.05); p.add_argument('--max-depth',type=float,default=1.5); p.add_argument('--voxel-size',type=float,default=.0015); p.add_argument('--min-target-points',type=int,default=250); p.add_argument('--no-outlier-filter',action='store_true'); a=p.parse_args()
+    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--weights',default='models/yolo26m-seg.pt'); p.add_argument('--target-class',required=True); p.add_argument('--target-index',type=int,default=0); p.add_argument('--width',type=int,default=640); p.add_argument('--height',type=int,default=480); p.add_argument('--fps',type=int,default=30); p.add_argument('--warmup',type=int,default=30); p.add_argument('--yolo-imgsz',type=int,default=640); p.add_argument('--yolo-conf',type=float,default=.25); p.add_argument('--min-depth',type=float,default=.05); p.add_argument('--max-depth',type=float,default=1.5); p.add_argument('--voxel-size',type=float,default=.0015); p.add_argument('--min-target-points',type=int,default=250); p.add_argument('--no-outlier-filter',action='store_true'); p.add_argument('--save-dir', default='captures/d435i_graspnet', help='Press S in RGB window to save a valid target snapshot here'); a=p.parse_args()
     if not all(x>0 for x in (a.width,a.height,a.fps)) or not a.min_depth<a.max_depth: p.error('invalid camera/depth parameters')
     a.yolo_weights=a.weights; a.yolo_classes=None; a.target_class=a.target_class; a.outlier_neighbors=20; a.outlier_std_ratio=2.; a.target_outlier_filter=not a.no_outlier_filter
-    print('实时推理已启动；慢慢把杯子放入画面，按 Q/Esc 退出。')
+    print(f'实时推理已启动；将 {a.target_class} 放入画面，在 RGB 窗口按 S 保存，Q/Esc 退出。')
+    print('保存目录:', Path(a.save_dir).expanduser().resolve())
     v=o3d.visualization.Visualizer(); v.create_window('D435i target cloud'); cloud=o3d.geometry.PointCloud(); added=False
     try:
-        for bgr,z,intr in capture_loop(a.width,a.height,a.fps,a.warmup):
-            xyz=depth_to_xyz_image(z,intr); instances,overlay=infer_yolo_instances(bgr,a)
-            print('\rRGB',bgr.shape,'detections',[(x['label'],round(float(x['score']),2)) for x in instances],end=' ',flush=True)
-            if 0<=a.target_index<len(instances) and not instances[a.target_index].get('is_box_only',False):
-                mask=mask_for_instance(instances[a.target_index],xyz,a); ca=SimpleNamespace(**vars(a)); raw=SimpleNamespace(**vars(ca)); raw.voxel_size=0.; raw.target_outlier_filter=False
-                c0=cloud_from_mask(xyz,bgr,mask,raw); c1=cloud_from_mask(xyz,bgr,mask,ca); print('cloud',len(c0.points),'->',len(c1.points),end='',flush=True)
-                cloud.points=o3d.utility.Vector3dVector(np.asarray(c1.points)); cloud.colors=o3d.utility.Vector3dVector(np.asarray(c1.colors))
-                if not added: v.add_geometry(cloud); added=True
-                else: v.update_geometry(cloud)
-            cv2.imshow('D435i RGB + YOLO mask (Q/Esc)',overlay); v.poll_events(); v.update_renderer()
-            if cv2.waitKey(1)&255 in (27,ord('q'),ord('Q')): break
+        with closing(capture_loop(a.width,a.height,a.fps,a.warmup)) as frames:
+            for bgr,z,intr in frames:
+                # Existing projection is pinhole-only; never save misleading XYZ from distorted pixels.
+                if np.any(np.abs(intr.distortion_coeffs) > 1e-9):
+                    raise RuntimeError('Nonzero color distortion: rectify before pinhole back-projection')
+                xyz=depth_to_xyz_image(z,intr); instances,overlay=infer_yolo_instances(bgr,a)
+                snapshot = None  # Never save a target from an earlier frame.
+                print('\rRGB',bgr.shape,'detections',[(x['label'],round(float(x['score']),2)) for x in instances],end=' ',flush=True)
+                if 0<=a.target_index<len(instances) and not instances[a.target_index].get('is_box_only',False):
+                    instance = instances[a.target_index]
+                    mask=mask_for_instance(instance,xyz,a)
+                    raw=SimpleNamespace(**vars(a)); raw.voxel_size=0.; raw.target_outlier_filter=False
+                    try:
+                        c0=cloud_from_mask(xyz,bgr,mask,raw)
+                        c1=cloud_from_mask(xyz,bgr,mask,a)
+                        if len(c1.points) < a.min_target_points:
+                            raise RuntimeError(f'filtered target cloud too small: {len(c1.points)}')
+                    except RuntimeError as exc:
+                        print('\n当前目标不可保存:', exc)
+                    else:
+                        snapshot=(mask,instance,c0,c1)
+                        print('cloud',len(c0.points),'->',len(c1.points),end='',flush=True)
+                        cloud.points=o3d.utility.Vector3dVector(np.asarray(c1.points)); cloud.colors=o3d.utility.Vector3dVector(np.asarray(c1.colors))
+                        if not added: v.add_geometry(cloud); added=True
+                        else: v.update_geometry(cloud)
+                if snapshot is None and added:
+                    cloud.clear()
+                    v.update_geometry(cloud)
+                cv2.imshow('D435i RGB + YOLO mask (S save, Q/Esc)',overlay)
+                if not v.poll_events():
+                    break
+                v.update_renderer()
+                key=cv2.waitKey(1)&255
+                if key in (27,ord('q'),ord('Q')):
+                    break
+                if key in (ord('s'),ord('S')):
+                    if snapshot is None:
+                        print('\n未保存：当前帧没有有效分割目标点云，请调整物体后重试。')
+                    else:
+                        try:
+                            path=save_snapshot(a,bgr,z,intr,overlay,*snapshot)
+                            print('\n已保存:', path)
+                        except OSError as exc:
+                            print('\n保存失败（不完整输出保留在 .partial 目录）:', exc)
+    except KeyboardInterrupt:
+        print('\n退出预览。')
     finally:
         v.destroy_window(); cv2.destroyAllWindows()
 if __name__=='__main__': main()
```

## 2026-09-15 D435i 采集断连记录

一次 `test_D435i.py` 运行中 RealSense 返回 `Device disconnected ... xioctl(VIDIOC_S_FMT), errno=5`，含义是USB视频设备在运行时断开或驱动重置失败，不是YOLO低置信度。此前同时出现的 `stop() cannot be called before start()` 是测试脚本无条件清理造成的次生异常，已在 `scripts/test_D435i.py:40-68` 修复：记录 `started`，只对成功启动的pipeline调用stop；原始断连错误改为提示检查USB3线缆、端口供电和接头稳定性。该修复不解决物理USB问题。

同次输出还显示检测置信度和目标有效点数随帧变化（242/146点会被拒绝，20634→3361点有效），说明按S保存必须等待稳定高质量分割。测试中尚未确认是否成功按S保存了正式目录；运行后应检查是否存在非`.partial`目录及 `target.ply`。

## 2026-09-15 标定JSON格式兼容

`config/2026-09-01_19-08-35_calibration.json`使用独立ArUco标定器格式：`translation_m` + `quaternion_xyzw`，并包含`matrix_4x4`；旧读取器只认`position` + `orientation`，导致执行前preflight的`KeyError: position`。已在`tools/verify_handeye_point.py:116`兼容两种字段，仍按CLI的`flange_camera_xyzw`解释，不自动改逆变换。该文件的`matrix_4x4`与`translation_m/quaternion_xyzw`一致；其`transform: camera_to_flange`命名必须与采集时的坐标定义继续核对，不能仅凭字段名宣称方向正确。

## 2026-09-15 GraspNet正式入口点云扩展路径修复

`main --execute` preflight暴露了与离线测试不同的导入顺序：`grasp_active...py`把源码目录`graspnet-baseline/pointnet2`放在sys.path首位，遮蔽了conda中已安装的`pointnet2._ext`。已修改该入口顶部路径接线：pointnet2源码目录仍用于`pointnet2_modules.py`等兄弟模块，但baseline根目录改为追加到sys.path末尾，使`pointnet2`解析到`/home/.../site-packages/pointnet2`及其CUDA `_ext`。正式helper导入和沙箱外、无机械臂连接的preflight均通过。再次执行前需保持`env -u PYTHONPATH`/`PYTHONNOUSERSITE=1`的SDK终端环境。
