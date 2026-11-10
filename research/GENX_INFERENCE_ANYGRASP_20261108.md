# GenX 推理侧到底改哪里，以及 AnyGrasp 是否适合作为底座

日期：2026-11-08。[T04 专题](GENX_EXECUTION.md)、[研究总入口](../RESEARCH_STATUS.md)。

本轮核对官方仓库、AnyGrasp 论文及使用文档，并阅读本地 GenX 调用和去噪循环。只更新研究文档，没有安装 SDK、申请许可证、下载权重、训练网络、修改算法或运行硬件。此前 MGPC/NBV/跨视角、GenX、清障与递送目标继续保留；用户提出 AnyGrasp 是在比较可选底座，尚未决定切换。

## 1. 先区分训练、推理与网络扩展

- **训练原生成器**：用数据和损失更新 GenX 参数，例如让新增场景/关节编码器真正参与噪声预测。
- **改推理算法**：保持原参数不变，改变部署时怎样采样、评价、分配预算、重采样或修正结果。
- **训练新增小网络**：原 GenX 冻结，用另外的数据训练执行评价器。这包含新网络训练，但不等于重训 GenX 主干。
- **修改输入数据处理**：改变实测/预测的构造、采样或筛选，也是部署流程变化；不自动构成 GenX 网络结构创新。

“GenX 只能推理、没有训练代码”不准确。[官方训练脚本](https://github.com/NVlabs/GraspGenX/blob/main/scripts/train_xgrasp.py) 存在；[官方 README](https://github.com/NVlabs/GraspGenX) 仍将完整训练数据列为待发布，并说明训练只在 Docker 测试。原始全数据不现成，不排除自建数据微调，但新增 q/场景条件需要相应数据和监督，本地未验证训练可运行。

## 2. 当前工程中的入口

| 位置 | 当前作用 | 可以研究的改动及边界 |
| --- | --- | --- |
| [run_graspgenx_worker.py](../tools/run_graspgenx_worker.py) | 去中心化目标点云，调用 `run_planner_on_object(..., planner="diffusion")`，还原平移，做桌面检查 | 控制候选数、分批生成、失败后追加预算、多输入候选池等；增大数量或切换现成 planner 是基线/配置变化，不能单独宣称新方法 |
| [grasp_candidates.py](../real_nero_grasp/grasp_candidates.py) 的 `recommend_grasp_candidate` 与 [grasp_workflow.py](../real_nero_grasp/grasp_workflow.py) | 接触/开口/路线检查和手工推荐 | 添加执行评价器；若目标是减少完整规划次数，应在昂贵完整检查前安排候选，不能全部检查后才声称省下规划时间 |
| [nero_refinement.py](../real_nero_grasp/nero_refinement.py) | 原候选没有完整有效路线时搜索有限姿态/深度变体 | 已有输出后修正；接触推进、翻转或局部调整不等于改去噪网络 |
| [generator.py](../third_party/GraspGenX/graspgenx/models/generator.py) 的 `forward_inference` | `for k in timesteps` 内预测噪声，调用位置/旋转 scheduler 更新抓姿 | 真正的扩散内引导位置：在指定步骤评价预计最终抓姿，重加权/重采样后继续去噪；或使用正确构造的可微代价引导 |
| [grasp_server.py](../third_party/GraspGenX/graspgenx/grasp_server.py) 与 [samplers/planner.py](../third_party/GraspGenX/graspgenx/samplers/planner.py) | 采样器封装及 planner 调度 | 若场景、q、证据要进内部采样循环，需沿 worker→sampler→generator 传入，不能只给外层请求增加字段 |

当前 `obs_embedding` 拼接物体与夹爪特征；原有权重没有学过任意新增的七轴 q。直接拼接新维度会破坏输入兼容性，也不会自动产生新条件能力。外部评价器使用 q，或推理代价使用 q，不需要假装原噪声预测网络已经学会该条件。

## 3. 三类具体方案

### A. 最容易理解：生成后重排

```text
原 GenX 生成候选 g1...gN
→ 新评价器 C_phi(g, 场景, q, 观测支持) 排出检查顺序
→ 按顺序进行原有完整几何/路线验证
→ 选择通过且质量合格的抓姿
```

例如两个候选都能夹住香蕉：一个需要绕过障碍或贴近关节限位，另一个路线更容易完成；评价器希望更早检查后者。评价器可用解析几何规则，也可训练小 MLP/PointNet。离线规划结果可以监督指定预算下的规划通过率/路线代价，不等于真实抓牢概率，不强制使用强化学习。

边界：只调整候选顺序，不产生原池里不存在的方向。要与当前手工评分、原 GenX 分数和简单几何代价比较；按场景/物体分组测试。[GraRe（2026 预印本）](https://arxiv.org/abs/2608.00946) 已研究冻结抓取器的学习重排，“加一个排序网络”本身不能直接作为新颖性主张。

### B. 改生成过程：扩散中反馈

```text
多组随机初始抓姿
→ 原 GenX 去噪若干步
→ 估计各组最终可能形成的抓姿
→ 按碰撞/可达性/观测证据等评价重加权和重采样
→ 使用原 GenX 继续去噪
→ 最终候选进入完整验证
```

例如在后段发现多数组朝向障碍一侧，另一组接近方向有空间，就把后续采样资源更多分配给后者。原网络权重不变，输出分布可以改变。该例用于解释机制，不保证可以恢复任何原模型完全缺失的抓取模式。

实施注意：

- 本地 scheduler 有 `pred_original_sample`，可作为预计干净抓姿的构造来源；不能把任意早期噪声姿态当真实抓姿算接触，再用强硬阈值删除全部样本。
- 当前主要接口含 `torch.inference_mode()` 和 `torch.no_grad()`。无梯度重采样可以避开可微 IK，但若用梯度引导，需处理调用边界；冻结权重和禁止输入梯度是不同的事。
- 旋转、位置使用不同 scheduler 的情况下，样本及关联状态必须一致重采样，保持合法旋转和多样性。完整规划不能无预算地放在每个粒子每一步中。
- 内部坐标基于目标中心化；场景、桌面和机器人几何要转换到一致坐标。原 worker 在输出后加回中心平移；不能直接拿基坐标障碍与中心化抓姿比较。
- 评价器若只学过最终候选，对中间预测是否适用需单独验证。未知/预测区域不是已知自由空间。

[Steer2Grasp（2026 预印本）](https://arxiv.org/abs/2609.33546) 已有冻结扩散、部署奖励及粒子重采样路线。这里说明改法和可行入口，不把迁移到 GenX 自动认定创新；未来差异可能围绕观测证据、完整任务执行和 NBV 反馈，需要与相关方法比较。

### C. 新条件进入原噪声预测网络

若要求“GenX 本体直接读取场景、机械臂状态并生成抓姿”，需改条件编码/融合并训练或微调相应模块。这是网络/训练路线，数据成本另算；不能靠推理时多传一个 q 参数完成。训练新增评价器可作为有网络改进的另一种实现，但论文命名需要准确。

## 4. AnyGrasp 的定位与公开程度

[AnyGrasp 论文，T-RO 2023](https://arxiv.org/abs/2212.08333) 的几何处理模块明确基于 GSNet/Graspness，并增加稳定性相关预测；还设计时序关联模块以跟踪抓姿。因此可理解为同团队沿 GraspNet/Graspness 路线发展的后续方法，不应将它等同于完整开源、可直接改训的 `graspnet-baseline`。论文结构也不能直接证明 2026 更新 SDK 的全部内部细节。

[官方 SDK README](https://github.com/graspnet/anygrasp_sdk) 明确因知识产权原因只提供需许可的 SDK 库；[检测说明](https://github.com/graspnet/anygrasp_sdk/blob/main/grasp_detection/README.md) 使用 `gsnet.so`，[注册说明](https://github.com/graspnet/anygrasp_sdk/blob/main/license_registration/README.md) 需要机器特征 ID。公开仓库有示例和部分算子，不等于核心模型训练实现开放。当前公开包不适合作为直接添加网络层、修改损失并重训的底座。

按 2026-07 更新后的 [USAGE.md](https://github.com/graspnet/anygrasp_sdk/blob/main/grasp_detection/USAGE.md)：

- 检测接口为 `create_detector(config)` 和 `detector.get_grasp(points, optional_params)`；输入米单位相机坐标 Nx3 点云，输出 `graspnetAPI.GraspGroup`，含位姿、开口、插入深度、分数等。
- 现成支持 `region_steering`（与输入点一一对应的区域掩膜）、`approach_steering` 和 `approach_thresh`。调用这些控制属于使用已有能力；不能作为自研模块创新。
- 检测只传 XYZ；示例颜色用于显示。论文中的跟踪模块与检测接口分开，不应混为一个模型调用。
- 7-DoF 抓取指位姿六维加夹爪开口，不是返回七轴关节轨迹。SDK 的宽度/高度配置不是 GenX 式任意夹爪形态条件化，也不能替代本工程 Nero CAD/TCP 和完整路线检查。
- 输入融合点云可作为待测集成方式，但精度、遮挡鲁棒性和本机时延需实测；论文其他机器人结果不能直接外推到当前奥比/Nero。若输入只有目标点，不能依赖 SDK 对输入点云的碰撞过滤覆盖未输入的环境。

## 5. 选择建议

| 目标 | 更合适的选择 | 理由与限制 |
| --- | --- | --- |
| 增加有代表性的抓取对照 | 评估 AnyGrasp | 已有成熟方法与 SDK；需要许可证、依赖适配和本地验证，不承诺优于当前 GenX |
| 修改扩散采样、加入部署时执行约束 | 继续研究 GenX | 源码可读，内部采样可改；先区分候选覆盖不足与排序问题，不用不可解释地改参数来凑创新 |
| 直接修改网络层/损失并在公开基准训练 | 优先评估 [EconomicGrasp 官方实现](https://github.com/iSEE-Laboratory/EconomicGrasp)；已有 GraspNet-baseline 也可作起点 | 有公开模型、训练、评测与权重说明，仍有算子编译和数据处理成本；不能把作者报告训练时长当本机承诺 |

近期建议保留已经接入的 GenX，先盘点固定候选池能否给出可行抓法，再选重排或内部采样试点。AnyGrasp 可作为新增对照候选，尚不建议为了“更容易改网络”切换过去。若网络主干改造是明确要求，应单独比较开放训练底座，不能将闭源 SDK 包装成可训练架构。此前主动感知、MGPC、清障和交接工作均不因本轮比较而取消。
