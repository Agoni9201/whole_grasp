# 待解决问题记录(OPEN ISSUES)

> 建档:2026-09-21 / 最近校对:2026-09-22 晚
> 场景:真机单次抓取,目标 = 薯片筒(Pringles 罐),命令带 `--target-class bottle`
> **处理约定(用户 2026-09-22):先把最基本流程跑通,再按编号逐个处理。本文件只记账,不在本轮改代码。**
> 解决一个,把对应状态改成 ✅ 并补一行结论。

## 0. 实机 run 观察

| run | 首帧识别 | 目标选择 | 运动结果 |
| --- | --- | --- | --- |
| `2026-09-20_20-09-14_855909` | overlay `scissors 0.52`(罩住筒+茄子),无 bottle | 过滤后 `instances=0` → 点像素 → `manual` | 未进入抓取 |
| `2026-09-20_20-16-47_941799` | overlay `bottle 0.48` + `dining table 0.26` + `donut 0.44` | 按 bottle 自动选中 | 成功进入抓取 |
| `2026-09-21_22-43-50_647134` | overlay `bottle 0.49`(筒)+ `spoon 0.40`(茄子) | 按 bottle 自动选中 | 中心位停滞 5.12° → 普通停未确认 → 电子急停 |
| 2026-09-21 晚(侧视去程) | 同上场景 | bottle 自动选中 | 侧视去程停滞 8.77° → 普通停确认成功、无急停 |
| `2026-09-22_14-14-15_001581` | `instances=0` → 点选 `manual` | 中心观察到位;侧视 45° 去程到位(0.6mm/1.44°) | 侧视 `keep=False`;回程 28.7mm < 30mm → `active_view_unsafe`,臂停侧视 |
| `2026-09-22_14-28-43_462470` | `manual` | 已进入抓取 candidate6 | 预抓取到位;接近段运行时 `finger+1` 0.8mm < 5mm → 普通停,未闭爪 |
| 伺服 `--phase servo`(无 `--execute`) | — | FakeBackend 全零关节 | 检测通了;随后 `Rejecting target jump` 0.14→0.34 m(预期,非崩) |

补充事实:

- `instances=0` 是**过滤后**的计数(conf + 类名两道过滤);`yolo_instances.png` 画的是**过滤前**的全部检出,两者语义不同。
- `manual` 是点像素路径写死的标签,下游一律当作"无类别身份"。
- 侧视 overlay 里出现 `hot dog 0.39` / `spoon 0.40` 是**未过滤显示**,不是被选中的目标;但 `manual` 目标在侧视会**无条件**抛"匹配失败"(逻辑洞,见 P3)。
- 清障重识别(P4)当前**测不到**:命令未开 `--enable-clear-obstacle`。
- **停滞稳定复现,急停不是**:第三次同样停滞但普通停确认成功。

## 1. 问题清单(总表)

| 编号 | 问题 | 影响 | 状态 |
| --- | --- | --- | --- |
| P1 | `--target-class bottle` 过滤后 `instances=0`;模型对该筒的类名**不稳定** | 偶尔选中、偶尔选不到 | ✅ 现象确认(2026-09-22 校对) |
| P2 | 点选后实例 `label="manual"`,类别身份丢失 | 身份链断点:侧视 `target_class=None`、清障重识别丢 reference mask | ✅ 确认(降级:服务融合,不是当前 abort 原因) |
| P3 | 侧视在 manual 时不按类别过滤,overlay 混入 `hot dog 0.39`/`spoon 0.40` | 观察干扰;**另有逻辑洞:manual + 有检出 → 必抛匹配失败** | ✅ 确认 |
| P4 | `manual` 目标在清障后无法按类别重新检测 | 当前命令**未开** `--enable-clear-obstacle`(默认关)→ 测不到 | ⏸ 暂缓 |
| P5 | 观察位停滞停在目标外几度;**停滞与急停是两件事** | 停滞稳定复现;急停取决于停确认窗口是否撞上暂态 | ✅ 确认(两次停滞:5.12° / 8.77°) |
| P6 | 停滞分支只打范数不打 7 轴分量与 current_joints;停确认只打末值 | 事后无法判断是 J6 贴限位、整臂提前结束还是停机暂态 | ⬜ 待补日志(优先级 1) |
| P7 | NBV 快路径"找到 1 个 ok 就停",`min_valid_candidates=8` 走不到 | 只剩最差候选:单轴 30.2°、J6 限位裕度 1.97° | ✅ 确认(硬门之一) |
| P8 | 侧视去程是单条 30° move_j(返回按 10° 切) | 七轴不同步 + 贴限位 → 收尾差;两层重发都不覆盖 | ✅ 确认 |
| P9 | 观察运动失败被包成 `ControlledMotionStopTriggered`(继承 `SafetyStopTriggered`) | active_view 的"换候选/回中心"分支接不到 → 整场判死 | ✅ 确认(硬门之一) |
| P10 | 返回地图自滤用中心关节,侧视夹爪点变假障碍 | 侧视并图后 `self_filter removed=0`;12.2mm / 28.7mm 撞 30mm 门槛 | ✅ 确认;2026-09-22 再次挡住回中心 |
| P11 | `keep=False` 的侧视仍用新点云否掉已验证回程 | 质量不合格本不融合,却把 129.8mm 旧回程作废,臂停侧视 | ⬜ 待处理 |
| P12 | OpenCV≥4.7 无 `estimatePoseSingleMarkers`,放码后异常急停 | 伺服一见到码就崩;diff 已给用户自改 | ⬜ 用户自改中 |
| P13 | 检测异常冒泡到 `emergency_stop`;丢目标日志刷屏 | 检测失败应保持阻抗,不应急停 | ⬜ 与 P12 一起 |
| P14 | 桌面降到与基座齐平后抓取偏浅 | `min_tcp_z=table_z+0+tool_bottom_offset`;0.02→0.01 仍浅 | ⬜ 待处理(参数) |
| P15 | `run_grasp_safe.sh` 默认 validation-first,实际只抬 ~4.5cm | `--lift-height 0.12` 走不到;抬升用 `GRASP_VALIDATION_LIFT_HEIGHT` | ⬜ 待处理(参数) |
| P16 | 初始张爪卡在 65mm,目标 85mm | 采图前退出;硬件卡滞/指间有物 | ✅ 现象确认(现场处理) |
| P17 | 抓取接近段运行时 0.8mm 停,**因此从未发闭爪** | 闭爪只在接近到位且 `grasp_close_pose_ready` 之后;接近被停则后续全跳过 | ⬜ 待处理;2026-09-22 15:49 再发 |
| P18 | 离线伺服 `Rejecting target jump` | FakeBackend 关节全零 + 手持码晃动;检测链已通 | ✅ 预期行为,不挡真机 |
| P19 | 侧视回中心 TCP z 掉到规划地板以下,运行时停 | 规划按关节直线只许比侧视起点低 2mm;真机 `move_j` 七轴不同步,实际掉 10mm+ | ⬜ 待处理(active-view 回程硬门) |
| P20 | 交接后伺服能动但不灵敏,终端几乎无状态 | `pipeline.yaml`/`servo_cli` 线速度 0.03m/s、关节 0.06rad/s,比 Visual Servo execute 的 0.35/0.15 小一个数量级;无逐帧误差打印 | ⬜ 待处理(参数+日志) |
| P21 | 腕部 D435 `VIDIOC_S_FMT errno=5` 打不开 | 抬升后交接会急停;USB 卡死后**下一轮首帧采集**也会在张爪后同样报错退出 | ⬜ 待处理 |
| P22 | 交接/急停时夹爪松开、末端反复 | 抓取 SDK `disconnect` 未把闭爪保持交给 MIT;相机异常走 `emergency_stop`;接触力仅 0.5N | ⬜ 待处理 |

**当前挡住"基本流程跑通"的:** P19/P10/P11(侧视回中心 TCP 掉高度或间隙) → 若绕过 active-view 则 P17(接近段运行时碰撞)。P14/P15 影响抓稳和抬升高度。P12 挡住放码后的伺服。其余记账,流程跑通后再逐个处理。

---

## P1:首帧 `instances=0`

**现象**
日志 `[yolo] instances=0 overlay=<run_dir>/yolo_instances.png`。

**含义(已确认)**
`instances` 是经两道过滤后的列表,不是 YOLO 原始检出数:

- `conf >= --yolo-conf`(默认 0.25)
- `_class_matches(label, --target-class)`:归一化后相等,或互相包含

所以 `instances=0` 只说明"没有任何检测的类别名能匹配 bottle",不代表画面里没有物体。

**结论(2026-09-22 校对):不是"模型永远认不出筒",而是 COCO 类名不稳定**

三次同场景的 overlay 证据(全部来自 `yolo_instances.png`,即过滤前的原始检出):

| run | overlay 上的检出 | 结果 |
| --- | --- | --- |
| `2026-09-20_20-09-14_855909` | `scissors 0.52`(一个大框罩住筒+茄子),无 bottle | → `instances=0` → 点像素 → `manual` |
| `2026-09-20_20-16-47_941799` | `bottle 0.48` + `dining table 0.26` + `donut 0.44` | → 按 bottle 自动选中 |
| `2026-09-21_22-43-50_647134` | `bottle 0.49`(筒)+ `spoon 0.40`(茄子) | → 按 bottle 自动选中 |

所以 P1 的处理方式是"**提高一次命中的概率**",而不是"修一个必然失败的 bug":
不要指望固定一个类名永远稳定,身份判定要走 P2/B 方案的几何弱约束路线。

**代码位置**

- 过滤逻辑:`real_nero_grasp/perception.py:94`(`_class_matches`)、`perception.py:163-166`
- 原始 overlay:`real_nero_grasp/perception.py:142`
- 打印:`real_nero_grasp/perception.py:617`(首帧)、`:1908`(侧视)
- 相关参数:`real_nero_grasp/cli.py:79`(`--target-class`)、`:88`(`--yolo-conf`)

**候选方案(2026-09-22 修订)**

- [ ] 方案 A2:降低 `--yolo-conf`(如 0.15),让低分 `bottle` 存活(成本最低,可先做)
- [x] ~~方案 A1:用 overlay 上的真实类别当 `--target-class`~~ —— 类名本身就不稳定(scissors / bottle / spoon 换来换去),不解决根本问题
- [x] ~~方案 B:用 `--select-target` 点 mask~~ —— **救不了 `instances=0`**:可点的实例列表已经被 `--target-class` 过滤过了,列表空了只能点像素 → 还是 `manual`(见 P2)

**验证标准**

- [ ] 连续 3 次同场景首帧,至少 2 次能自动命中;否则走"未过滤 mask 点选"(见 P2 新方案)

---

## P2:点选后标签变成 `manual`

**现象**
点选薯片筒后,实例字典为 `{"label": "manual", "score": 1.0, ...}`。

**含义(已确认)**
`manual` 不是 YOLO 类别,而是"用户点出来的、没有语义标签"的标记。做法是:取点击像素的 3D 点为球心,`--manual-radius`(默认 0.045 m)为半径取局部点云;`score=1.0` 是写死的。

**触发路径**

- 显式:`--manual-click-only`,或 `--target-prompt` 里输入 `POINT`/`manual`
- 隐式(本次):过滤后实例为 0 → 点像素回退

**代码位置**

- 点选生成实例:`real_nero_grasp/perception.py:713-732`(label 写在 `:726`)
- 点击未命中任何 mask 时的回退:`perception.py:679-681`

**下游影响(这是 P3/P4 的根因)**

`{"", "manual", "unknown", "target"}` 被当作"无类别身份":

- `real_nero_grasp/perception.py:1754-1770` `_active_view_second_capture_args` → 侧视 `target_class=None`、`manual_click_only=True`
- `real_nero_grasp/perception.py:260` `label_is_specific` → 重识别不做类别过滤
- `real_nero_grasp/perception.py:1699-1701`、`:1718-1721` `recapture_original_target_after_clear` → 丢掉 reference mask、`target_class=None`

**候选方案(2026-09-22 修订:见第 6 节"身份链方案")**

- [x] ~~方案 1:改用 `--select-target` 点 YOLO mask~~ —— **救不了 `instances=0`**,列表已被过滤空(见 P1)
- [ ] 方案 2:点选时同时保留"请求的类名"(`requested_target_class`)与所选 mask 的真实标签:
      - 点到 YOLO mask → 用该 mask 的真实标签(可能是 scissors/cup),另存请求的 `bottle`
      - 点到空白 → 用局部球点云,但 `class_label` 仍写请求类名
- [ ] 方案 3:侧视继续沿用 3D 中心对齐(`active_strict_target_identity` 的距离判定),类名只做软约束,对不上再投影
- [ ] 方案 4:接受 `manual`,但明确记录"几何匹配"的行为边界与风险,并加强几何校验

**验证标准**

- [ ] 点选后实例带有可用的类别信息(真实标签或请求类名),不再只有 `manual`
- [ ] 侧视不再因 `manual` 无条件抛 `no YOLO instance matched`

---

## P3:侧视不过滤类别,出现 `hot dog 0.39`

**现象**
侧视日志/overlay 中出现茄子被检成 `hot dog 0.39`,而不是 bottle。

**含义(已确认)**
因为首帧目标标签是 `manual`,侧视的 `target_class` 被设为 `None`,该分支下不做类别过滤,于是 YOLO 返回全类别列表,茄子被误分类为 `hot dog` / `spoon` 只是"未过滤的候选",不代表它被选中;那次真正的目标是投影球回退的结果。

**逻辑洞(2026-09-22 新增,比 overlay 显示更重要)**
`perception.py:1913-1997`:当 label 是 `manual` 时,"按类名匹配"的整段被跳过,`chosen` 保持 `None`,而下面又无条件 `raise RuntimeError("no YOLO instance matched the first-view target label")`。
所以 **只要侧视 YOLO 检出任意实例,点选目标就必然抛这个"匹配失败"**,然后才回退到投影中心。
→ 也就是说:只要目标是点选来的,侧视"匹配失败"几乎 100% 发生,与 YOLO 检出了什么无关。

**代码位置**

- 侧视 yolo 分支:`real_nero_grasp/perception.py:1902-1913`
- 类别排除集合与无条件 raise:`perception.py:1913-1997`(raise 在 `:1994-1997`)
- 回退顺序:`perception.py:2007-2042`
- 参数:`real_nero_grasp/cli.py:162`(`--active-target-reselect`,默认 `yolo`)、`:163`(`--active-yolo-fallback`,默认 `projected`)

**候选方案(2026-09-22 修订)**

- [ ] 方案 1(必须):修掉逻辑洞 —— 无类别时**不要进"按类名匹配"再无条件 raise**,应直接走"3D 中心对齐 → 失败再投影"
- [ ] 方案 2:日志里的 `instances=` 保持"过滤后数量"语义,overlay 可以继续画全部检出(避免把未过滤候选误读成选中目标)
- [ ] 方案 3:修好 P2 后,侧视恢复按类别过滤,此问题自然消失

**验证标准**

- [ ] 点选目标时侧视日志不再出现 `no YOLO instance matched the first-view target label`
- [ ] 侧视能按 3D 中心对齐到同一目标(或明确走投影回退并打印原因)

---

## P4:`manual` 目标在清障后无法按类别重新检测

**状态:⏸ 暂缓** —— 当前运行命令**没有 `--enable-clear-obstacle`**(默认关闭,`cli.py:332` 的 `--clear-largest-visible-fallback` 也需显式开启),清障与清障后重识别这条链现在根本走不到,**测不到也就无法验证**。

**现象(代码层面成立)**
清障后重识别会退化为纯几何匹配;`manual` 情况下不会重新检测 bottle,而是再次要求人工点击。

**含义(已确认)**
`recapture_original_target_after_clear` 对 `manual` 标签会丢掉 reference mask 并把 `target_class` 置空;重识别函数也只按 mask IoU + 中心距离匹配,没有类别约束。

**代码位置**

- `real_nero_grasp/perception.py:1699-1701`、`:1718-1721`(`recapture_original_target_after_clear`)
- `real_nero_grasp/perception.py:259-266`(`label_is_specific`,重识别不做类别过滤)

**风险**

- [ ] 场景里有多个大小相近、位置相近的物体时,存在认成另一个物体的风险。
- [ ] 清障结束后需要重新人工点击,流程中断。

**候选方案(2026-09-22 修订)**

- [x] ~~方案 1:优先修 P1/P2~~ —— 身份链修好后 P4 会跟着改善,不需要单独做
- [ ] 方案 2:为重识别加入"几何 + 颜色/点云统计"的一致性约束(**等身份链修好、有现场数据后再评估**)
- [ ] 方案 3:对 `manual` 目标禁用清障后自动重新关联,改为显式要求人工确认(安全优先,可作兜底)

**结论**:先不做;等开启清障功能、并跑通 active-view 之后再回来。

**验证标准**

- [ ] 清障后能自动认出原目标,或明确、可控地要求人工确认。

---

## P5:观察位停滞(未到位)与电子急停

> **重要澄清(2026-09-22)**:停滞和急停是**两段独立的事**。
> - **停滞稳定复现**:两次都在目标外几度停住 —— 第一次中心位 `5.12°`(J6 为主),第三次侧视去程 `8.77°`。
> - **急停只是碰运气**:第三次同样停滞(8.77°),但普通停**确认成功**(`last_delta=0.071°`),没有急停。
> 所以"急停"取决于停确认窗口是否撞上停机暂态;"停滞"才是真正的硬问题。

**现象一:第一次中心位(2026-09-21,未进入抓取)**

| 量 | 值 | 含义 |
| --- | --- | --- |
| 预校验目标 | `[-7.71, 13.524, 0.146, 66.429, 6.0, -30.37, 81.289]` deg | 路径检查已通过 |
| 最大关节步 | 17.35°(几乎全部来自 J6) | 规划认为可执行 |
| 到位判定 | `joint_err=5.12°`,`last_delta=0.005°` | 已经几乎不动,但离目标超过 5° |
| 法兰误差 | 位置 21.6mm,姿态 13.34° | 不是阈值误报,是真的没到 |
| 普通停 | 末值 `last_delta=0.118°`,1.0s 内未凑满连续 3 次 | 普通终止未确认静止 |
| 结果 | `electronic_emergency_stop` | 程序按 fail-closed 退出 |

**现象二:第三次侧视去程(2026-09-21 晚)**

| 量 | 值 | 含义 |
| --- | --- | --- |
| 停滞残差 | 8.77° | 同样是"走到附近停住" |
| 普通停 | `last_delta=0.071°`,确认成功 | **没有急停** |
| 后果 | 臂停在侧视中途 | 后续流程无法继续(见 P9) |

**链路(代码位置)**

1. 中心位由预校验关节目标规划并下发:
   `active_view.py:2453-2488`(逐 roll 规划)→ `active_motion.py:321-329`(`{label}_direct` 阶段)
   → `execution.py:183-204`(`move_flange_pose(..., planned_joint_target=...)`,失败时先请求普通停)
2. 到位/停滞判定 `arm_execution.py:568-597`:
   - 到位:`‖wrap(q−q*)‖₂ ≤ --move-joint-tol`(默认 2.0°,`cli.py:914`)
   - 停滞:距下发 ≥ `--move-stall-grace`(1.0s)+ 连续 `--move-stall-confirm-samples`(5)次 `max|Δq| ≤ --move-stall-delta-deg`(0.02°),且 `‖err‖₂ ≥ --move-stall-far-error-deg`(5.0°)
   - 注意两个阈值都是 **7 关节范数**,所以 5.12° 可能主要来自 J6 一个轴
3. 抛 `MotionNotReached`:`arm_execution.py:733-739`(先判关节,再判法兰位姿)
4. **两层重发都没有触发**:
   - 起始重发 `arm_execution.py:462-499`(要求朝新目标方向走出 ≥0.2°,本次已走 17.35°)
   - 迟发重试 `arm_execution.py:689-721`(要求从下发瞬间起位移 ≤0.5°,本次 17.35°)
5. 普通终止 `stop_policy.py:11-63` → `arm.py:355-422`(3 帧 `track_ctrl=0x06` + 速度设 0 + 以当前实测关节为 hold 目标;1.0s 内要求连续 3 次 Δq≤0.15°)
6. 未确认 → `arm.py:419-422` 抛错 → `stop_policy.py:54-63` 升级 `safe_stop` → 电子急停
7. 急停后 `_safety_stop_latched` 使本进程后续所有 move 被拒(`arm.py:149-161`),退出码 3

**根因(已确认)**

- **不是碰撞**:台面/运行期碰撞/力矩保护在等待循环里每个采样都在跑(`arm_execution.py:570-571`),两次的异常都没有出现。
- **控制器停在目标外几度**:预校验只保证"可达 + 不超限位 + 路径不碰撞",不保证伺服收尾精度;残差集中在单轴(第一次 J6)。
- **阈值错位**:到位公差 2° / 停滞远误差 5° / 实际残差 5.12°~8.77° → 构成"既不算到位、也不算没动"的夹缝。
- **两层重发之间的空档**:重发只覆盖"命令未生效"(位移 ≤0.5°),不覆盖"走了 17°/30° 再停住"。
- **停确认窗口偏紧(但第三次已能通过)**:1.0s 内要求连续 3 次 Δq≤0.15°;第一次撞上停机暂态失败,第三次成功。

**待确认(需要日志/实验,对应 P6)**

- [ ] `terminate_motion` 只打印最后一个 `last_delta` → 需要窗口内每采样 Δq
- [ ] 停滞分支只打印范数 → 需要 7 个关节分量,确认是 J6 贴限位还是整臂提前结束
- [ ] 停滞时刻的控制器状态(`motion_status`、joint limit 错误位)未记录 → 复用 `_arm_motion_status_snapshot`(`arm_execution.py:189`)

**候选方案**:见第 5 节"方案设计"(残差重试 + P7 源头治理)。

**验证标准**

- [ ] 5~9° 残差能自动补一次到位(不再需要人工干预)
- [ ] 仍然不到位的场景,按现有普通停流程处理,不引入新的危险路径
- [ ] 真正停不下来时仍能升级急停(fail-closed 不被破坏)

---

## P6:判定分支的日志不足以定位原因

**现象**:停滞判定与停确认都只输出"最终数值",没有过程数据,事后无法区分故障形态。

**需要补的日志**

- [ ] `wait_until_joints_reached` 停滞分支(`arm_execution.py:581-587`):输出 7 个关节的误差分量 + `_arm_motion_status_snapshot`
- [ ] `terminate_motion`(`arm.py:391-422`):输出 1.0s 窗口内每个采样的 Δq(至少首 5 个与末 5 个)
- [ ] 到位成功时也输出一次各关节分量(便于对比正常收尾形态)

**验证标准**

- [ ] 复现同类故障时,日志能直接区分"控制器提前结束"与"停机暂态未收敛"

---

## P7:NBV 快路径过早收兵(硬门)

**现象**
`--active-nbv-fast-plan-candidates` 默认 8:快路径只检查信息分最高的前 8 个球面位置,只要其中**任意 1 个**可达就 `break`,跳过完整 IK 扩展;`--active-nbv-min-valid-candidates`(默认 8)因此永远走不到。

**证据(第三次 run)**
最终只剩 `nbv_090_orb30` 一个候选:单轴 **30.2°**、J6 限位裕度仅 **1.97°**(门槛 1.5°)。
→ 这个"最差姿势"就是 P8 那个 30° 大动作的来源。

**代码位置**

- 快路径预算:`real_nero_grasp/active_view.py:1671-1691`(`fast_plan_budget` / `desired_valid_viewpoints`)
- **提前退出**:`active_view.py:1812-1822`(`valid_viewpoints > 0` 就 break)
- "凑够数量才停"的分支:`active_view.py:1823-1828`(要求先处理完 `initial_plan_budget=32` 个,实际到不了)
- 参数:`real_nero_grasp/cli.py:175-180`

**候选方案(2026-09-22)**

- [ ] 方案 1(必须):快路径的退出条件改为"**凑不满 `desired_valid_viewpoints` 就不收兵**"(至少 >1,最好 ≥`min_valid_candidates`)
- [ ] 方案 2:候选准入加质量门槛 —— 优先**丢弃限位裕度 <8°、单轴步长 >25°** 的候选
- [ ] 方案 3:日志里打印"fast pass found N/M"与每个候选的 `jmax` / 限位裕度,便于事后判断

**验证标准**

- [ ] 日志中出现多个 `ok` 候选,且被选中的候选限位裕度明显大于 2°、单轴步长明显小于 30°

---

## P8:侧视去程是一条 30° 的 `move_j`(硬门)

**现象**
- **返回**路径会按 `--released-return-execution-waypoint-deg`(默认 10°)切成多个路点;
- **去程**(outbound)故意保持**单条指令**(`active_motion.py:350-356` 的注释:中心/外出观察动作维持"一条命令"语义)。

Nero 七轴不同步,贴限位时收尾差 → 直接落在 P5 的停滞判据里(1s 宽限 + 连续 5 次 Δq≤0.02° + 误差范数 ≥5° → 失败)。而两层重发只覆盖"命令没动"(位移 ≤0.5°),**走了 17°/30° 再停住不会重发**。

**代码位置**

- 去程不细分:`real_nero_grasp/active_motion.py:350-356`
- 停滞判据:`real_nero_grasp/arm_execution.py:568-597`
- 两层重发:`arm_execution.py:462-499`、`:689-721`

**候选方案**

- [ ] **只做对照实验**:给去程加一个可选的切路点参数(类似 `--active-center-subdivide-deg`),**不和残差重试/P7 绑在一起改**,否则分不清是哪条起作用
- [ ] 实验条件:同一摆放、同一目标,分别跑"单指令"和"切路点"各 2~3 次

**验证标准**

- [ ] 能给出对照数据:去程停滞次数、单轴最大步长、到位残差

---

## P9:观察运动失败会把整场判死(硬门)

**现象**
`execution.py:190-204` 把任何 `MotionNotReached` 包装成 `ControlledMotionStopTriggered`,而它**继承自 `SafetyStopTriggered`**(`types.py:41-46`)。

`active_view` 里"换下一个侧视候选 / 回中心"的分支只接 `MotionNotReached`(`active_view.py:3300`、`:3370`、`:3787`),**接不到这个包装后的异常** → 直接向上冒泡 → 臂停在侧视中途,抓取和伺服都不会开始。

**代码位置**

- 包装点:`real_nero_grasp/execution.py:190-204`
- 异常层级:`real_nero_grasp/types.py:38-46`
- 只接 `MotionNotReached` 的分支:`active_view.py:3300`、`:3370`、`:3787`

**候选方案**

- [ ] 方案 1:区分"可恢复的观察失败"与"终止性安全停止" —— 普通停**确认成功**后,允许 active_view 回中心或试下一个候选
- [ ] 方案 2:电子急停仍然只保留给"普通停确认失败 / 碰撞 / 力矩",语义不变
- [ ] 方案 3:回中心后要给出明确状态(用单视角点云继续,还是本场不抓)—— 见第 6 节待拍板问题 Q2

**验证标准**

- [ ] 侧视停滞 → 普通停确认 → **回中心**(或换候选)→ 流程继续,不再整场判死

---

## P10:返回地图自滤用错关节(暂非主因)

**现象**
融合地图按每帧关节调用 `_active_motion_capture_base`;但返回重规划时把侧视点云并进地图后,**用中心位姿的关节滤了一次** → 侧视下的夹爪点被当成假障碍。

**现状**:第三次因为换了摆放 + 30° 视角刚好避开(52.7mm > 30mm 门槛),暂时不是主因,**但代码缺陷仍在**。

**代码位置**:`real_nero_grasp/active_view.py` 中返回路径的地图构建与 `_active_motion_capture_base` 调用点(做运动修复时顺手确认)。

**候选方案**

- [ ] 把返回并图改成使用每帧自己的 `_active_motion_capture_base`(已有函数),单独验证

**验证标准**

- [ ] 返回重规划不再出现由侧视夹爪产生的假障碍点

---

## P11:`keep=False` 的侧视否掉已验证回程

**现象(`2026-09-22_14-14-15` → 侧视 `2026-09-22_14-16-13_069812`)**

- 去程 45° 到位(法兰 0.6mm / 1.44°)。
- 侧视质量 `med=0.0060m p95=0.0242m in1cm=0.621 keep=False`(投影球 80mm 深度杂)。
- 去之前回程已验证 **129.8mm ≥ 30mm**,切成 4 段。
- 采完仍用新点云重规划: `gripper_finger_+1` **28.7mm < 30.0mm**(差 1.3mm)。
- fail-closed:不准用旧回程,臂停侧视,`active_view_unsafe`,抓取禁止。

**含义**
质量不合格的帧本不进入融合,却改写了返回占用图。28.7 vs 30.0 也和 P10 自滤用错关节叠在一起。

**代码位置**
`real_nero_grasp/active_view.py` 侧视采集后 `keep=False` 仍走 `return_center_updated_scene`;失败则 `side_return_plan is None` 并 raise。

**处理方向(流程跑通后再做)**
`keep=False` 时执行去之前那条已验证回程,不要用新点云否决。不要把 30mm 降到 28mm。

---

## P12:ArUco `estimatePoseSingleMarkers` 在新 OpenCV 上不存在

**现象**
一放码(id=582)就:
`AttributeError: module 'cv2.aruco' has no attribute 'estimatePoseSingleMarkers'`
→ `run_servo_delivery` 的 `except Exception` → `backend.emergency_stop()`。

没码时只刷 `target lost`(检测返回 None)。有码才走进求位姿 API。

**代码位置**
- `targets/aruco_eye_in_hand_target.py:198-203`
- `servo_delivery.py:513-516`
- 标定脚本已用 `cv2.solvePnP`:`tools/calibrate_d435i_aruco.py:82,118`

**处理方向**
已给用户带行号的 diff(solvePnP 兼容 + 检测异常当丢目标)。用户自改,本仓库未动。

---

## P13:检测异常急停 + 丢目标刷屏

**现象**
检测路径未捕获的异常会急停整场。丢目标时 30Hz 打 `target lost; holding last impedance target`。

**代码位置**
`servo_delivery.py:414-422`(`_on_timer`);`_fail_closed` 仍留给内环/反馈/qdot。

**处理方向**
与 P12 同一 diff:try/except 当 `target=None`;lost 日志 1s 限频。

---

## P14:桌面与基座齐平后抓取偏浅

**现象**
旧桌高于基座时护栏碰不到。新桌与基座一样高后,闭合深度不够,物体抓不起来。

护栏(安全脚本):
```
min_tcp_z = table_z(0.0) + table_tcp_clearance(0.00) + table_tool_bottom_offset(0.02) = 0.02 m
```
TCP 不能低于桌面上 2cm。`--grasp-advance` 默认 0.015m。`TABLE_TOOL_BOTTOM_OFFSET=0.010` 试过仍浅。

**代码位置**
- `scripts/run_grasp_safe.sh:97` `--table-z`
- `collision.py:111-116` `table_guard_min_tcp_z`
- `cli.py:783` `--grasp-advance`

**处理方向(参数,不改默认语义)**
`TABLE_Z` 保持 0.0。继续降 `TABLE_TOOL_BOTTOM_OFFSET`(如 0.005)并加大 `--grasp-advance 0.030`。俯视时指尖离桌更近,需慢速看。

---

## P15:validation-first 实际只抬 ~4.5cm

**现象**
想多抬一点。`run_grasp_safe.sh` 默认 `GRASP_VALIDATION_MODE=1` 且 `GRASP_VALIDATION_FIRST=1`,规划一做出 4.5cm 验证抬升就返回,`--lift-height 0.12` 走不到。

`2026-09-22_14-28-43`: grasp_tcp z=0.135, lift_tcp z=0.176(**+4.1cm**)。

**代码位置**
- `scripts/run_grasp_safe.sh:6-15,119-124`
- `grasp_planner.py:359-424`

**处理方向**
抬升用环境变量 `GRASP_VALIDATION_LIFT_HEIGHT`(脚本把 `--lift-height` 写在用户参数后面会覆盖命令行)。例如 `0.070`。完整 12cm 需 `GRASP_VALIDATION_FIRST=0`。

---

## P16:初始张爪未到 85mm

**现象**
命令 open 0.085m,反馈卡在 **0.0653m**,力为负,超时 `samples=0/2` → `GraspNotDetected: gripper failed to open before capture`。无急停。

**代码位置**
`session_control.py:194-236,310-312`

**处理方向**
现场:指间异物、手动扳开、再跑。不是规划问题。下一轮已张到 0.0847m 确认。

---

## P17:抓取接近段运行时碰撞停(当前抓取硬门)

**现象(`2026-09-22_14-28-43_462470`)**

| 项 | 值 |
| --- | --- |
| 候选 | target_candidate6, `manual`, validation_only |
| pre / grasp / lift TCP z | 0.233 / **0.135** / 0.176 m |
| 预抓取 | 两段到位 |
| 接近段规划 | `target_approach` 最近 7.1mm,门槛 5.0mm |
| 运行时 | `gripper_finger_+1` **0.8mm**,要求 ≥5.0mm |
| 停法 | 普通停确认,无电子急停;未闭爪 |

规划按关节直线检查;Nero `move_j` 七轴不同步,7.1mm 裕度被吃到 0.8mm。闭合 z=0.135 已进入物体高度,这次主要是接近段擦环境,不是桌面 2cm 护栏。

**复现 2(`2026-09-22_15-49-25_396686`)**
candidate6 预抓取到位;接近段规划 **5.5mm**;运行时仍 **0.8mm** 停。用户问「为什么不闭合」:闭爪只在接近到位且 `grasp_close_pose_ready` 之后才 `slow close`。接近被停后日志写 `subsequent grasp/lift/place commands were skipped`,闭爪指令从未发出。

**代码位置**
运行时间隙监视(接近段 `phase=target_approach`);诊断 `runtime_collision_diagnostics.jsonl`。

**处理方向(流程跑通后再做)**
接近段规划裕度太小则换候选或切路点。不要把运行时 5mm 降到 0.8mm。

---

## P18:离线伺服 jump reject(预期)

**现象**
`--phase servo` 无 `--execute`:先 `target lost`,放码后 `Rejecting target jump: 0.14→0.34 m`。无 traceback、无急停。

FakeBackend 关节全零,基座系目标不可信;手持码超过 `max_target_jump=0.08m`(`pbvs_pose_servo.py:82,345-354`)就会拒帧。说明 **P12 求位姿已通**(若用户已打上 solvePnP)。真机 `--execute` 时码要停稳。

---

## P19:侧视回中心 TCP 掉到规划地板以下

**现象(多次,最近 `2026-09-22_14-41-11`)**

规划阶段两个候选直接被拒:

- 回程采样 TCP z=0.3102m < 要求 0.3140m
- 回程采样 TCP z=0.3182m < 要求 0.3210m

第三个候选规划擦线通过(`stage_min_tcp_z` 间隙 1.9mm,要求 0)。去程到位后回程第 1/4 路点运行时:

```
strict_floor_violation=10.4mm extra_runtime_slack=8.0mm runtime_envelope_excess=2.4mm
```

普通停确认,无电子急停,整场结束。臂停在回程中途。

**机制**
回程地板 = **侧视起点 TCP z − 2mm**(`--active-direct-return-downward-tolerance`,默认 0.002)。
运行时额外再允许到总共低 10mm(`--active-direct-return-runtime-downward-tolerance` 0.010,其中 8mm 是 runtime slack)。
规划用**关节空间直线**采样 TCP z;Nero `move_j` 七轴不同步,实际 TCP 会在中段掉得更低。侧视法兰约 0.43m、中心约 0.49m,大关节行程(34°~43°)回中心时几乎必然下凹。

**代码位置**
- 地板定义:`active_motion.py:269-302`
- 规划拒绝:`collision.py:2206-2218`
- 运行时 slack:`runtime_safety.py:392-499`
- 参数:`cli.py:147-149`

**处理方向(流程跑通后再做;不改代码时可试参数)**
先抬再横移:`--active-direct-return-retreat-height 0.05`(默认 0)。不要先把 2mm/10mm 地板放宽当长期方案。

---

## P21:抬升 SUCCESS 后腕部相机打不开,异常急停

**现象**
接触确认 width≈69mm、力 0.5N;抬升法兰 z +70mm 到位;`grasp validation succeeded`。随后:

```
Emergency stop triggered
RuntimeError: Device disconnected. Failed to reconnect: xioctl(VIDIOC_S_FMT) failed, errno=5
RuntimeError: stop() cannot be called before start()
```

栈:`after_grasp_lifted` → `delivery_after_grasp` 先 `close_all()` 再 `handoff`(MIT 接管、抓取 SDK 断开)→ `run_servo_delivery` 里 `camera_for(args).read()` 重启 D435 失败。`except Exception` 调 `backend.emergency_stop()`。

**后续 run(张爪之后、YOLO 之前)** 同一错误出现在首帧采集:`perception.py` → `cameras/realsense.py:52 pipe.start(config)` → `xioctl(VIDIOC_S_FMT) errno=5`。这次无急停(还没进伺服)。说明上一轮 USB 异常后设备卡在坏状态,新进程 `start()` 也会失败。

**代码位置**
- `handoff/grasp_to_servo_handoff.py:129-139`(`close_all` 在交接前)
- `servo_delivery.py:514`、`:525-528`(任意异常急停)
- `cameras/realsense.py:54-61,91-96`(warmup 失败仍 `stop()`,USB 已掉则二次异常)

**处理方向**
相机启动失败应保持 MIT 位置+夹爪,不要急停。`close()` 在 `pipeline` 未成功 start 时不要 `stop()`。USB 掉线单独记,和视觉控制律无关。

---

## P22:交接时夹爪松开、末端反复

**现象(用户现场)**
已经夹住并抬升成功,抬升后/交接时夹爪突然松开,末端开始反复晃。日志里软件仍写 `gripper remains closed`,没有 `slow close` 的反向张爪命令。

**可能叠加的原因**
1. 交接 `arm.arm.disconnect()` 切断抓取驱动,MIT 侧没有立刻下发「保持当前宽度」;夹爪失令会松。
2. P21 相机异常立刻 `emergency_stop()`,MIT 刚接管就被急停,末端会抽动/反复。
3. 闭爪接触阈值 `|force|>=0.400`,命令力 0.500N(也是 `GRIPPER_MAX_FORCE`)。夹持很轻,抬升惯性也可能滑脱。

**代码位置**
`grasp_to_servo_handoff.py:111-126,65-75`;`servo_delivery.py` 异常急停。

**处理方向**
断开抓取 SDK 前,MIT 先对夹爪下发保持当前宽度/力。相机/检测失败保持阻抗,不急停(与 P13 同类)。接触力是否过小等有现场数据再评。

---

## 2. 复现步骤(复现后补全)

> 目的:把本次问题稳定复现出来,再逐个解决。每次复现都记录 run 目录。

1. 环境:`conda activate grasp_yolo26`,确认 Orbbec 相机已发布,SDK 别名/标定文件路径正确。
2. 摆放:**孤立摆放优先** —— 薯片筒放桌面中央,减少旁边物体进入返回图(对应 P10 的规避条件):尽量不同时摆茄子。
3. 启动命令(补全与本次一致的其余参数)。**建议先加 `--no-active-multiview-gate`**,降低邻物干扰:

   ```bash
   bash scripts/run_grasp_safe.sh --calib-json <标定.json> --target-class bottle \
     --no-active-multiview-gate <其余参数与本次一致>
   ```

4. 记录首帧日志 `[yolo] instances=0` 或命中的类名、点选后的 `label=manual`、侧视 overlay 上的误检类别。
5. 记录 run 目录(已有三次现场数据):
   - `2026-09-20_20-09-14_855909` —— overlay `scissors 0.52` → instances=0 → manual
   - `2026-09-20_20-16-47_941799` —— overlay `bottle 0.48` → 自动选中(成功)
   - `2026-09-21_22-43-50_647134` —— overlay `bottle 0.49` + `spoon 0.40` → 自动选中
   - 第三次侧视去程停滞 8.77°(2026-09-21 晚,run 目录待补)
6. 若只想快速看模型的分类结果,不动机械臂:

   ```bash
   python3 scripts/test_D435i.py --weights models/yolo26m-seg.pt --target-class bottle \
     --yolo-conf 0.05 --max-depth 3.0 --save-dir captures/d435i_graspnet
   ```

7. 复现 P5(中心观察位 / 停机链)。目标是同一类**单关节大角度**观察动作(本次是 J6≈17.35° 的腕部 roll):

   - [ ] 先用 `--no-execute` 干跑一次,记录中心位预校验关节目标,尤其是 J6 的目标与步长
   - [ ] 再执行一次,记录:`[active_center_plan_direct][ik] joint_delta_deg`、到位判定、停滞判定数值
   - [ ] 记录 7 个关节的分量(不是范数),确认残差是否集中在 J6
   - [ ] 记录普通停窗口内的采样序列(需要先补 P6 的日志)
   - [ ] 记录是否出现 `[motion_stop]` / `electronic_emergency_stop` / 退出码 3

   命令模板:

   ```bash
   bash scripts/run_grasp_safe.sh --calib-json <标定.json> --target-class <实际类别> \
     --select-target <其余参数与本次一致>
   ```

8. 急停后的现场处理顺序(复现前先确认,避免二次风险):

   - [ ] 确认机械臂已静止、无悬空负载
   - [ ] 结束当前进程(已 latch,同进程无法再运动)
   - [ ] 清急停 / 重新 enable(必要时重新上电)
   - [ ] 重新运行前先 `--no-execute` 验证中心位目标仍然可达

## 3. 每次 run 的产物与检查点

| 文件 | 用途 |
| --- | --- |
| `yolo_instances.png` | **过滤前**的原始检测 overlay,看真实类别与分数 |
| `selected_target_mask.png` / `selected_target_overlay.png` | 最终选中的目标 mask |
| `capture_meta.json` | 内参、序列号、时间戳 |
| `selected_grasp_pose.json` / `whole_workflow_motion_plan.json` | 抓取候选与运动计划 |

检查点清单(每次复现后勾选):

- [ ] 记录首帧 `instances=` 的数值,以及 `yolo_instances.png` 上筒的真实标签
- [ ] 记录最终选中实例的 `label`(是否 `manual`)
- [ ] 记录侧视 `instances=` 数值与其中的误检类别
- [ ] 记录是否走了 projected / click 回退
- [ ] 记录是否有 Same-class identity ambiguous / distance 超限的报错
- [ ] 记录中心位预校验关节目标与最大关节步(尤其 J6)
- [ ] 记录到位/停滞判定的关节分量,不只范数
- [ ] 记录普通停窗口的采样序列与是否升级急停
- [ ] 记录 NBV:fast pass 找到几个可达点、是否跳过 full expansion、候选的 `jmax` 与限位裕度
- [ ] 记录侧视去程单轴最大步长与到位残差(对照 P8)

## 4. 落地顺序

**先跑通基本流程(抓起并短抬升),再按编号改代码。** 跑通期间只用参数/摆放绕过,不在本轮改源码。

现场绕过(不改代码):

- 孤立摆放;必要时暂时去掉 `--active-view` 以绕过 P10/P11,先验证闭爪+抬升。
- `TABLE_Z=0.0`,`TABLE_TOOL_BOTTOM_OFFSET=0.005`,`--grasp-advance 0.030`(P14)。
- `GRASP_VALIDATION_LIFT_HEIGHT=0.070`(P15)。
- 张爪前确认夹爪空且能到 ~85mm(P16)。
- 伺服先 `--phase servo` 无 `--execute` 验检测(P18);真机放码要停稳。P12 需用户已打上 solvePnP。

流程跑通后再逐个改(建议顺序):

| 顺序 | 编号 | 怎样算过 |
| --- | --- | --- |
| 1 | P6 日志 | 停滞能看到 7 轴与停确认序列 |
| 2 | P11 | `keep=False` 走旧回程,臂能回中心 |
| 3 | P10 | 返回并图按侧视关节自滤 |
| 4 | P5/P9 | 残差重试;观察失败回中心,不整场判死 |
| 5 | P7 | NBV 多个 ok,限位裕度明显大于 2° |
| 6 | P17 | 接近段规划裕度不足则换候选/切路点,不再 7mm 擦线执行 |
| 7 | P12/P13 | 放码不急停;lost 限频 |
| 8 | P1–P3 | 身份链;侧视不再因 `manual` 必抛匹配失败 |
| 9 | P8 | 去程切路点对照 |
| 10 | P4 | 仅当开启清障后 |
| 11 | P14/P15 | 若参数已够用则关闭;不够再评默认值 |

**原则上不变的是**:
- 不把停滞远误差 5° 改到 8~10°(那是放宽"没到也算走完",会让 8.77° 这种真没到被当成成功,后面的 NBV/手眼会歪);
- 先不给观察动作单独的 joint tolerance;
- P4 清障重识别先不做(P4 未开启,且身份链修好后会跟着好)。

## 5. 方案设计:残差重试与停确认窗口(2026-09-22 定稿,按此实施)

> 目标:让"已经走到目标附近、只是停在几度外"的情况能自动补救;让"几乎停住"不被误判成"停不下来";同时**不削弱** fail-closed 原则。

### 方案 A(核心):残差重试(residual retry)

**问题**:现有两层重发都只覆盖"命令未生效"(位移 ≤0.5°),对"走了 17° 停在 5° 外"没有处理。

**位置**:`arm_execution.py:689-721`(late retry 之后,同一个函数 `move_flange_pose_ik_joint` 内)。

**触发条件(全部满足才重试)**

1. 等待返回的是**停滞**(不是超时)—— 需要让 `wait_until_joints_reached` 记录失败原因;
2. 残差 `‖wrap(q_now − q_target)‖₂ ≤ --move-residual-retry-max-deg`(取 **12°**:5.12° 与 8.77° 都在内;30° 这种半路大偏差不重试);
3. 本次命令**尚未用过**残差重试(每个目标最多 1 次);
4. 机械臂已静止(复用 `_wait_for_joint_settle` 的 pre-command settle 栅栏);
5. 用冻结的运行期碰撞上下文复核"当前 → 目标"路径(照抄 late retry 的 `validate_path` 调用方式,`arm_execution.py:488-496`)。

**安全约束(重要)**

- 重试的是**同一个已预校验目标**,不新增目标、不改目标;
- 熔断:重试后仍不到位 → 直接按原逻辑抛 `MotionNotReached`(不再重试第三次);
- 任何保护异常(台面/碰撞/力矩)照常抛出,不进入重试;
- 超时(还在动)不重试 —— 还在动就继续等,不打断。

**伪代码**

```python
if not joint_reached and current_joints is not None:
    reason = getattr(arm, "_last_wait_failure_reason", None)   # 新增
    residual = float(np.linalg.norm(wrap_angle_error(current_joints, q_target)))
    limit = np.deg2rad(max(float(getattr(args, "move_residual_retry_max_deg", 12.0)), 0.0))
    if (
        reason == "stalled"
        and residual <= limit
        and bool(getattr(args, "move_residual_retry", True))
        and not bool(getattr(arm, "_residual_retry_used", False))
    ):
        print(f"[{label}][residual-retry] stalled at joint_err="
              f"{np.rad2deg(residual):.2f}deg <= {np.rad2deg(limit):.2f}deg; "
              "resend the same prevalidated target once")
        arm._residual_retry_used = True
        send_joint_target(arm, q_target, args, f"{label}_residual_retry")
        joint_reached, current_joints, joint_err = wait_joints(
            arm, q_target, args, f"{label}_joint_after_residual_retry"
        )
```

**新增参数**

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `--move-residual-retry` / `--no-move-residual-retry` | 开 | 总开关 |
| `--move-residual-retry-max-deg` | 12.0 | 允许重试的最大残差(7 关节范数) |

**测试**(加入 `tests/test_joint_command_retry.py`)

- [ ] 停滞 + 残差 5° → 重发一次并到位
- [ ] 停滞 + 残差 30° → 不重发
- [ ] 超时(非停滞)→ 不重发
- [ ] 已重试过一次 → 不再重发
- [ ] 重发前调用路径复核(冻结上下文)

### 方案 B:停确认窗口 —— 只参数化,默认先不动

**背景**:`arm.py:391-422` 硬编码"1.0s 内连续 3 次 Δq ≤ 0.15°"。
第一次(中心位)撞上停机暂态失败并升级急停;**第三次(侧视去程)同样停滞但确认成功**(`last_delta=0.071°`)。

**决定(2026-09-22)**:先**只把三个量提成参数**,默认值保持 1.0s / 0.15° / 3 次不变 —— 避免用"放宽"掩盖真正停不住的情况。等 P6 过程日志到位后再决定是否改默认值。

**改法**

1. 三个量提成参数,由 `workflow.py:87-97` 构造 `RealNeroGraspArm(...)` 时从 args 传入;
2. 默认保持 `1.0s / 0.15° / 3 次`;
3. 预置 `--stop-stationary-grace`(暂态宽限,默认 0 先不生效),以及窗口/阈值参数,便于后续对照实验;
4. 输出窗口内每次采样的 Δq(见方案 D)。

**后续(只有再次出现急停才做)**:把窗口调到 2.0s 或启用 0.3s 暂态宽限,并用参数做对照实验。

**测试**(加入 `tests/test_arm_startup_shutdown.py` 或新建)

- [ ] 采样序列 `[0.6, 0.4, 0.3, 0.12, 0.05, 0.04]` → 在 2.0s 窗口内确认(参数化后)
- [ ] 采样序列全程 ≥0.5° → 仍升级急停
- [ ] grace 内的暂态不打断连续计数(参数化后)

### 方案 C:分层阈值 —— 不做

- [x] ~~把 `--move-stall-far-error-deg` 从 5° 提到 8~10°~~ —— **否决**:那等于"没到也算走完",会把 8.77° 这种真没到当成成功,后面的 NBV/手眼都会歪。**残差重试比放宽到位阈值合适**。
- [x] ~~给观察动作单独的 joint tolerance~~ —— **暂缓**:`T_base_camera` 用实测法兰能自洽,但和规划的侧视几何会偏。等运动稳定后再评。

### 方案 D:诊断日志(对应 P6,优先级 1,和运动修复一起做)

- [ ] `wait_until_joints_reached`:失败时记录原因(`stalled` / `timeout`)+ 7 个关节误差分量 + `current_joints`
- [ ] 停滞分支打印 `_arm_motion_status_snapshot`(`arm_execution.py:189`),含 `motion_status` 与 joint limit 错误位
- [ ] `terminate_motion`:打印窗口内每次采样 Δq

### 方案 E:去程切路点 —— 只做对照实验,不和 1~4 绑在一起

中心/外出观察动作目前刻意保持"一条命令"(`active_motion.py:350-356`)。返回已经按 10° 切路点,去程是否也切,**等残差重试 + NBV 多候选之后再看**;一次改太多会分不清是哪条起作用。

**注意**:如果实验做,也必须单独开关、单独验证。

### 其他已讨论的决策

- [x] 残差上限取 **12°**(覆盖 5.12° / 8.77°)
- [x] 每个目标严格**只重试 1 次**
- [x] 停确认窗口**先只参数化、默认不动**
- [ ] "重试后仍失败 → 是否允许一次受控回退":现有 `allow_controlled_recovery` 只在普通停确认后可用,留待观察

## 6. 身份链方案(P1–P3)与待拍板问题

> 身份链的定位:**服务融合质量,不是当前 abort 的原因**。运动修好之前它不挡流程。

### 关键认识

- `--select-target` **救不了 `instances=0`**:可点的实例列表已经按 `--target-class` 过滤过了,列表空了就只能点像素 → `manual`(P1/P2)。
- 侧视在 `manual` 时"匹配失败"是**无条件**的(P3 逻辑洞),与 YOLO 检出了什么无关。

### 建议的首帧目标选择策略

1. **有 bottle 就用 bottle**(第三次 run 已发生过);
2. **没有 bottle 时**,让用户点**未过滤**的 YOLO mask(实例保留模型真实标签,scissors/cup 也可以),并同时记下"我请求的是 bottle";
3. **点到空白才用球**(局部点云),但 `class_label` 仍写请求类名,供侧视做**软约束**;
4. 侧视继续用 **3D 中心对齐**(已存在的 `active_strict_target_identity` 距离判定),对不上再投影;
5. 修掉 `manual` + 有检出就必抛 `no YOLO instance matched` 的洞;
6. 侧视 overlay 可以继续画全部检出,但**日志里的 `instances=` 保持"过滤后数量"语义**。

### 待拍板 Q1:这个筒的"身份"以什么为准?

| 选项 | 含义 | 后果 |
| --- | --- | --- |
| A | YOLO 类名必须是 `bottle` | 模型不稳(scissors / bottle / spoon 反复),会反复 `instances=0` |
| **B(倾向)** | **以点中的那团 3D 中心为准**,类名只做弱约束 | 与现有投影回退一致,最稳;纯几何,靠距离区分 |
| C | 点中哪个 mask 就用那个 mask 的类名往下传 | 保留语义,但会传"错的"类名下去 |

**倾向 B 为主 + C 做提示**:几何对齐当硬约束,类名当软参考。

### 待拍板 Q2:观察运动停滞、普通停已确认之后怎么办?

| 选项 | 含义 | 代价 |
| --- | --- | --- |
| 1 | 回中心,用**单视角点云**继续抓 | 能跑通全流程;单视角点云差,成功率和安全性下降 |
| **2(倾向)** | **必须拿到融合才允许抓**(回中心后可再试别的侧视;侧视全失败则本场不抓) | 安全、质量高;可能一场跑不完,卡在观察阶段 |

**倾向先选 2**:现在缺的不是"能抓一次",而是"稳定跑通";等 2~3 次融合成功、运动修复验证过之后,再放开选项 1 作为降级路径。

## 7. 变更记录

| 日期 | 变更 | 说明 |
| --- | --- | --- |
| 2026-09-21 | 建档 | 记录 P1–P4 现象、代码位置、候选方案与复现步骤 |
| 2026-09-21 | 增补 P5/P6 | 记录中心观察位未到位 → 普通停未确认 → 电子急停的链路、根因与待确认项 |
| 2026-09-21 | 增补第 5 节 | 残差重试 / 停确认窗口 / 分层阈值 / 诊断日志 的方案设计与待讨论点 |
| 2026-09-22 | 校对修订 | 用三次实机日志核对:P1 降级为"类名不稳定"(附三次 overlay 证据)、P3 增补逻辑洞、P4 标注未开清障暂缓、P5 拆分"停滞"与"急停" |
| 2026-09-22 | 新增 P7–P10 | NBV 快路径过早收兵、去程单条 30° move_j、观察失败判死整场、返回地图用错关节 |
| 2026-09-22 | 修订方案 | 残差重试上限定为 12°;停确认窗口"只参数化、默认不动";否决放宽停滞远误差;去程切路点改为对照实验;新增身份链方案与 Q1/Q2 待拍板 |
| 2026-09-22 晚 | 增补 P11–P18 | 侧视 keep=False 否决回程;ArUco API 急停;丢目标刷屏;桌面齐平抓浅;validation 只抬 4.5cm;张爪 65mm;接近段运行时 0.8mm;离线伺服 jump。约定先跑通基本流程再逐个改代码 |
| 2026-09-22 晚 | 增补 P19 | 侧视回中心 TCP z 掉破规划地板(2mm 规划 / 10mm 运行时),move_j 七轴不同步 |
| 2026-09-22 晚 | P17 补「不闭合」;增补 P21/P22 | 接近停则永不闭爪;抬升 SUCCESS 后 D435 重启失败急停;交接失令松爪+末端反复 |
