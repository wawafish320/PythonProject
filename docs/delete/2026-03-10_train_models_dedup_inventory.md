# 2026-03-10 `train/models.py` P0 去重清单（子文档）

Parent:
- `docs/delete/2026-03-10_train_models_refactor_hotspots_roadmap.md`

Purpose:
- 作为 `train/models.py` 热点路线图的 P0 子文档，专门记录“先去重”阶段的目标、证据与执行顺序。
- 本文档只关注重复面收敛，不直接讨论 Phase B 巨函数拆分。

---

## 0) 本轮进度（2026-03-10 / update-5）

本轮已落地 P0-1 / P0-2 / P0-3 / P0-4 / P0-5 / P0-6 / P0-7 的第一轮去重，先后把 `direct_pose` split state、compat 搬运模板、`MotionJointLoss.forward` component submit 模板、`direct_pose` joint spec resolver、split mask / out_idx builder、split head / proj builder，以及 aux weighted-loss + stats 样板收口。

已完成结果：

- `direct_pose` split state / compat helper：
  - `train/models.py:1794` `_direct_pose_split_state`
  - `train/models.py:1827` `_direct_pose_local_index`
  - `train/models.py:1839` `_normalize_split_index_buffer`
  - `train/models.py:1855` `_copy_tensor_if_compatible`
  - `train/models.py:1873` `_copy_indexed_tensor_if_needed`
- `MotionJointLoss.forward` aux submit/alignment helper：
  - `train/models.py:5987` `_ensure_temporal_axis`
  - `train/models.py:5992` `_prepare_aux_supervision_pair`
  - `train/models.py:6007` `_submit_component_loss`
- `MotionJointLoss.forward` aux submit 调用点：
  - `train/models.py:5775` `contact_plan`
  - `train/models.py:5848` `event_clock_lambda_entropy`
  - `train/models.py:5861` `event_clock_lambda_prior`
  - `train/models.py:5883` `event_clock_delta_z_l2`
  - `train/models.py:5902` `contact_meas`
  - `train/models.py:5924` `omega_l2`
- `direct_pose` joint spec resolver helper：
  - `train/models.py:65` `_normalize_joint_spec_items`
  - `train/models.py:93` `_resolve_joint_spec_indices`
  - `train/models.py:133` `_resolve_rot6d_joint_count`
- `direct_pose` split mask / out_idx builder：
  - `train/models.py:1110` `build_split_out_index`
  - `train/models.py:1131` leg/nonleg builder 调用点
  - `train/models.py:1151` arm/else builder 调用点
- `direct_pose` split head / proj builder：
  - `train/models.py:1489` leg branch 调用点
  - `train/models.py:1504` arm branch 调用点
  - `train/models.py:1509` else branch 调用点
  - `train/models.py:1515` nonleg branch 调用点
  - `train/models.py:1818` `_build_split_head_branch`

本轮结构指标变化（累计相对路线图起点）：

- LOC：`6008 -> 6075`
- `_forward_direct_pose_readout`：`63 -> 53`
- `_maybe_upgrade_direct_pose_split_state_dict`：`333 -> 228`
- `MotionJointLoss.forward` inline component submit 模板：`9 -> 0`
- `MotionJointLoss.forward` aux weighted-loss + stats inline cluster：`4 -> 0`
- `EventMotionModel.__init__` 手写 joint-spec resolver 块：`3 -> 0`
- `direct_pose` 手写 split mask / out_idx builder 块：`2 -> 0`
- `direct_pose` split head / proj builder 块：`2 -> 0`
- `MotionJointLoss.forward`：`578 -> 560`
- `direct_pose_` token count：`630 -> 599`
- `legacy + fallback` marker count：`28 -> 28`

`update-5` 增量（相对 `update-4` 开始前的 P0-7 step）：

- LOC：`6087 -> 6075`
- `MotionJointLoss.forward`：`578 -> 560`
- `contact_plan / contact_meas / omega_l2` weighted-loss + stats cluster：`4 -> 0`
- `direct_pose_` token count：`599 -> 599`

`update-6` 增量（相对 `update-5` 结束时的 B2 round 1）：

- LOC：`6075 -> 6202`
- `MotionJointLoss.forward`：`560 -> 11`
- orchestration helper 边界：
  - `train/models.py:5388` `_prepare_forward_inputs`
  - `train/models.py:5585` `_apply_motion_components`
  - `train/models.py:5821` `_apply_direct_pose_component`
  - `train/models.py:6043` `_apply_aux_components`
  - `train/models.py:6055` `_finalize_forward_outputs`

本轮验证：

- `python -m py_compile train/models.py`
- 2 个 `MotionJointLoss` aux smoke case 已通过：
  - logits-path：`contact_plan + contact_meas + omega_l2 + event_clock_*`
  - fallback-path：`contact_plan` 2D probability MSE fallback
- `MotionJointLoss` B2 orchestration smoke 已通过：
  - `direct_pose` split + arm/else balance + group-norm
  - logits-path aux submit
  - 2D `contact_plan` MSE fallback

`update-7` 增量（相对 `update-6` 结束时的 B2 round 2 / C1+C2 收窄）：

- LOC：`6202 -> 6197`
- `MotionJointLoss.forward`：保持 `11`
- `MotionJointLoss.forward` 抽出 helper 区的 `except Exception`：`9 -> 0`
- C1：`rot_ortho_fallback` / `event_clock_lambda_mean` / loss-group contrib 仅保留诊断标量化窄异常
- C2：`direct_pose` / `contact_plan` / `event_clock` / `contact_meas` / `omega_l2` active path 改为显式 guard 或 fail-fast
- `python tools/verify_rot_local_tail.py` 已通过

当前判断：

- P0-1 / P0-2 / P0-3 / P0-4 / P0-5 / P0-6 / P0-7 已进入“第一轮完成，局部残余已可转入拆分判断”的状态
- `MotionJointLoss.forward` 的 aux weighted-loss + stats 样板已不再是当前第一优先阻塞
- `Step B2` 已进入 round 2，`MotionJointLoss.forward` 仍是 orchestration 壳，但 helper 内主路径异常吞没已基本清空；若继续停留在 Phase A，则次选 `contact_plan_* / event_clock_*` builder 边界

---

## 1) 当前判断

在本轮去重之后，`train/models.py` 剩余的重复仍然不是零散小重复，而是几块更清晰暴露出来的“结构性重复”：

1. `EventMotionModel.__init__` 中 `contact_plan_* / event_clock_*` builder 边界仍偏厚
2. `MotionJointLoss.forward` 已完成 shell 化，但 helper 内仍残留异常吞没/结构债，仍属于下一轮优先观察面
3. compat 外层薄壳仍有窄重复（同入口双 try/except 包装）

结论：

- P0 应先去重，再拆分
- P0-1 / P0-2 已经证明“先收口重复，再做拆分”是有效策略
- 当前重复面已经足够收窄，并已进入 `MotionJointLoss.forward` 的 `Step B2` round 1

---

## 2) P0 重复热点清单

### P0-1 `direct_pose` split state 访问重复

状态：

- `Done (round 1, 2026-03-10)`
- 但仍保留少量初始化阶段的派生重复，已不再是当前第一优先

关键证据：

- index buffer 定义：`train/models.py:864`, `train/models.py:865`, `train/models.py:866`, `train/models.py:867`
- index buffer 写入：`train/models.py:1129`, `train/models.py:1130`, `train/models.py:1160`, `train/models.py:1161`
- 统一 split state 入口：`train/models.py:1794`
- 构建输出层时改为统一读取：`train/models.py:1463`
- readout 时改为统一读取：`train/models.py:1903`
- compat 时改为统一读取：`train/models.py:1954`

现象：

- 同一组对象被反复访问：
  - `direct_pose_out_leg`
  - `direct_pose_out_nonleg`
  - `direct_pose_out_arm`
  - `direct_pose_out_else`
  - `direct_pose_leg_out_idx`
  - `direct_pose_nonleg_out_idx`
  - `direct_pose_arm_out_idx`
  - `direct_pose_else_out_idx`

本轮结果：

- 已建立统一的 `direct_pose_split_state` 读取入口
- 已统一返回：
  - enable flags
  - output heads
  - index buffers
  - 维度/存在性检查结果
- scattered `getattr(self, "direct_pose_*")` 访问模板计数：`25 -> 11`
- `_forward_direct_pose_readout` 长度：`63 -> 53`

残余：

- `__init__` 阶段仍存在 split joint spec / mask / out_idx 的派生重复
- 这些残余重复已转移到 P0-4 / P0-6

---

### P0-2 direct-pose compat 改写模板重复

状态：

- `Done (round 1, 2026-03-10)`
- 但 projection warm-start 与外层薄壳仍有残余重复

关键证据：

- compat 主块：`train/models.py:1953`
- split index buffer 规范化 helper：`train/models.py:1839`
- old weight/bias indexed copy helper：`train/models.py:1873`
- direct local idx 映射 helper：`train/models.py:1827`
- nonleg -> arm/else warm-start：`train/models.py:2066` 到 `train/models.py:2094`
- projection warm-start / stale tensor cleanup：`train/models.py:2096` 到 `train/models.py:2179`
- 外层薄壳：`train/models.py:2182`, `train/models.py:2188`

现象：

- 同一种模板被多次展开：
  - `state_dict.get(...)`
  - shape 检查
  - `index_select(...)`
  - 命中后写回 `state_dict[...]`
- weight/bias 的处理模式高度相似，但按 leg / nonleg / arm / else 分支手写展开

本轮结果：

- 已提取纯转换 helper：
  - `_direct_pose_local_index`
  - `_normalize_split_index_buffer`
  - `_copy_tensor_if_compatible`
  - `_copy_indexed_tensor_if_needed`
- `_maybe_upgrade_direct_pose_split_state_dict` 长度：`333 -> 228`
- compat 主块已经从“按分支手写搬运”变成“主流程 + 模板 helper”

残余：

- `direct_pose_nonleg_proj` 的 SVD warm-start 仍是单块大样板
- arm/else projection warm-start 与 stale tensor cleanup 还没有进一步抽平
- `adapt_legacy_state_dict_` / `load_state_dict` 仍是对同一入口的窄壳重复

---

### P0-3 `MotionJointLoss.forward` 的 stats / contrib 提交模板重复

状态：

- `Done (round 1, 2026-03-10)`
- event-clock 三个目标分支与 core 侧 6 个目标分支都已切到统一 submit 模板

关键证据：

- 通用 helper：
  - `train/models.py:6001` `_component_stats_payload`
  - `train/models.py:6021` `_setdefault_stats`
  - `train/models.py:6025` `_submit_component_loss`
- 统一调用点：
  - `rot_ortho`: `train/models.py:5411` 到 `train/models.py:5433`
  - `rot_local`: `train/models.py:5459` 到 `train/models.py:5470`
  - `rot_local_tail`: `train/models.py:5491` 到 `train/models.py:5506`
  - `root_vel`: `train/models.py:5519` 到 `train/models.py:5530`
  - `root_speed`: `train/models.py:5537` 到 `train/models.py:5548`
  - `direct_pose`: `train/models.py:5702` 到 `train/models.py:5748`
  - `event_clock_lambda_entropy`: `train/models.py:5862` 到 `train/models.py:5873`
  - `event_clock_lambda_prior`: `train/models.py:5883` 到 `train/models.py:5895`
  - `event_clock_delta_z_l2`: `train/models.py:5908` 到 `train/models.py:5921`

本轮结果：

- 内联 `loss += ... / _accumulate_loss_contrib / stats / _register_component_loss` 模板：`9 -> 0`
- 目标 stats key 集合保持不变
- `MotionJointLoss.forward` 主调用点只保留 component 名称、raw tensor、weight、stats payload

残余：

- `contact_plan` / `contact_meas` / `omega_l2` 还没有并入统一 submit 模板
- `MotionJointLoss.forward` 函数长度仍高，后续仍需为 Phase B 做净减准备

---

### P0-4 `direct_pose` joint spec 解析 / 名称索引归一化重复

状态：

- `Done (round 1, 2026-03-10)`
- leg / split-leg / arm 三处手写 resolver 已统一到共享 helper

关键证据：

- 通用 resolver：
  - `train/models.py:65` `_normalize_joint_spec_items`
  - `train/models.py:93` `_resolve_joint_spec_indices`
  - `train/models.py:133` `_resolve_rot6d_joint_count`
- 统一调用点：
  - leg joint spec：`train/models.py:1027`
  - split-leg joint spec：`train/models.py:1062`
  - arm joint spec：`train/models.py:1132`
  - loss 侧 named resolver 也已复用同一 helper：`train/models.py:4702`

本轮结果：

- `EventMotionModel.__init__` 中手写 `name_to_idx + item loop + seen set` resolver 块：`3 -> 0`
- `MotionJointLoss` 与 `EventMotionModel` 的 joint spec 解析入口已对齐

残余：

- split mask / out_idx 生成已统一
- split head / proj builder 也已统一；`A3` 的 `direct_pose` builder cluster 已基本清空

---

### P0-5 `direct_pose` split mask / out_idx 构造重复

状态：

- `Done (round 1, 2026-03-10)`
- leg/nonleg 与 arm/else 的手写 mask builder 已统一到共享 local builder

关键证据：

- 统一 builder：`train/models.py:1110`
- leg / nonleg 调用点：`train/models.py:1131` 到 `train/models.py:1142`
- arm / else 调用点：`train/models.py:1151` 到 `train/models.py:1166`

现象：

- 同一种“joint idx -> 6D dim mask -> `nonzero` -> coverage check”模板被写了两遍
- 当前只是：
  - leg 分支以 total output dim 为全集
  - arm 分支以 nonleg dim mask 为全集

本轮结果：

- `joint idx -> rot6d dim mask -> out_idx` 手写模板已统一为单个 local builder
- leg/nonleg 与 arm/else 调用点只保留：
  - joint spec 差异
  - base mask 约束差异
  - coverage / empty 校验差异
- `direct_pose` 手写 split mask / out_idx builder cluster：`2 -> 0`
- P0-5 step 以 `LOC 6088 -> 6088` 收尾，满足单步硬门禁

残余：

- `direct_pose` split builder cluster 已清空
- 已不再占据当前第一优先

---

### P0-6 `direct_pose` split head / proj 装配样板重复

状态：

- `Done (round 1, 2026-03-10)`
- arm/else 与 nonleg 的 split branch builder 已统一到共享 helper

关键证据：

- 统一 builder：`train/models.py:1818`
- leg branch 调用点：`train/models.py:1489`
- arm branch 调用点：`train/models.py:1504`
- else branch 调用点：`train/models.py:1509`
- nonleg branch 调用点：`train/models.py:1515`

现象：

- `proj_dim > 0` 时都是：
  - 决定 branch input dim
  - 建一个 `Linear + ReLU` projection
  - 再接 branch output head
- 当前 arm / else / nonleg 只是 branch 数量和目标输出维度不同

建议去重动作：

本轮结果：

- `Linear + ReLU proj + branch head` 装配模板已统一到 `_build_split_head_branch`
- leg / arm / else / nonleg 调用点只保留：
  - 输出维度差异
  - `proj_dim` 开关差异
- `direct_pose` split head / proj builder cluster：`2 -> 0`
- `EventMotionModel.__init__`：`1280 -> 1266`

残余：

- `direct_pose` split builder 不再是当前主要重复源
- 已不再占据当前第一优先

---

### P0-7 contact supervision / event regularizer 的 weighted-loss + stats 样板重复

状态：

- `Done (round 1, 2026-03-10)`
- `contact_plan / contact_meas / omega_l2` 已并入统一 submit 路径，event-clock 三项也同步收敛到同一 raw/weighted payload 写法

关键证据：

- contact plan：`train/models.py:5775` 到 `train/models.py:5825`
- event clock：`train/models.py:5828` 到 `train/models.py:5898`
- contact meas：`train/models.py:5902` 到 `train/models.py:5920`
- omega regularizer：`train/models.py:5924` 到 `train/models.py:5941`
- aux 对齐 helper：`train/models.py:5987`, `train/models.py:5992`
- 通用 submit helper：`train/models.py:6007`

现象：

- 多个辅助 loss 分支都在重复：
  - shape/time 对齐
  - `loss = loss + weight * term`
  - `stats["*_weighted"] = ...`
- 其中 event-clock 三个正则项虽已接入 submit helper，但直到本轮才与 `contact_plan` / `contact_meas` / `omega_l2` 对齐到同一 payload 写法

本轮结果：

- `contact_plan / contact_meas / omega_l2` 的内联 `loss += weight * term + weighted stats` 模板：`4 -> 0`
- `MotionJointLoss.forward` 内部 `stats["*_weighted"] = ...` / `loss = loss + weight * term` 直接写法已清空
- event-clock 三项也统一切到 `_submit_component_loss(..., group='aux', raw_key=..., weighted_key=...)`
- `MotionJointLoss.forward`：`578 -> 560`

残余：

- weighted-loss + stats 模板已不再是当前阻塞
- 下一跳更适合转向 `Step B2`，把 `MotionJointLoss.forward` 收敛成 orchestration 壳

---

## 3) 当前不建议优先去重的部分

暂不建议把下面内容作为 P0 第一刀：

- 整个 `EventMotionModel.forward` 流程级去重
- 整个 `MotionJointLoss.forward` 全局 helper 化
- 跨文件搬迁 `EventMotionModel` / `MotionJointLoss`

原因：

- 当前主风险不是“没有 helper”，而是“重复逻辑边界还没收紧”
- 过早做全局流程抽象，很容易制造大量单次调用 wrapper
- 当前更适合先做局部模板收敛，而不是直接把大段逻辑搬成一串薄 helper

---

## 4) P0 建议顺序

1. 评估 `Step B2`：把 `MotionJointLoss.forward` 继续收敛成 orchestration 壳
2. 若继续留在 Phase A，则转向 `EventMotionModel.__init__` 的 `contact_plan_* / event_clock_*` builder 边界
3. 再决定是否进入 `B1`

说明：

- 原顺序中的 P0-1 / P0-2 / P0-3 / P0-4 / P0-5 / P0-6 已完成第一轮，因此不再占据首位
- 现在 `P0-7` 已完成，进入 `B2` 的时机已成熟

---

## 5) 与主路线图的关系

本文档服务于主路线图中的：

- `Phase A / Step A1`
- `Phase A / Step A2`
- `Phase A / Step A2.5`
- `Phase A / Step A3`

主路线图：
- `docs/delete/2026-03-10_train_models_refactor_hotspots_roadmap.md`

执行时原则：

- 先以本文档为 P0 操作清单
- 去重指标下降后，再回到主路线图推进 Phase B
