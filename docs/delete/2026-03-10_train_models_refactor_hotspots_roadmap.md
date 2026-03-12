# 2026-03-10 `train/models.py` 热点重构路线图（v1）

Date: 2026-03-10  
Status: Draft / Active v1 / update-5（`train/posttrain.py` 已做过较细整理，本轮优先锁定 `train/models.py`；并明确采用“先去重，后拆分”作为第一优先策略；`train/training_MPL.py` 暂列下一顺位）  
Scope: `train/models.py`（本轮只做结构去耦与职责拆分，不改模型语义、不改默认超参、不改 rotvec 几何约定）  
Goal: 在**不改变语义/行为**前提下，优先降低“重复样板 + 巨函数 + 上下文耦合 + 静默异常 + 隐式副作用”维护风险。  
Non-goal: 不改核心算法/数学定义、不改默认训练入口、不引入新的模型架构。

关联子文档：
- P0 去重清单：`docs/delete/2026-03-10_train_models_dedup_inventory.md`

---

## 0) 当前策略（先去重是第一优先策略；先去重/收边界，再拆分，再收紧异常，最后剥离副作用）

本轮总原则先明确：

- **第一优先策略不是拆大函数，而是先去重。**
- 只有当重复 guard / 重复映射 / 重复 compat 样板 / 重复 stats 提交模板先收敛后，才进入 Phase B 的巨函数拆分。
- 如果 Phase A 尚未带来可量化重复下降，Phase B 默认不启动。
- P0 去重目标与证据清单，统一记录在子文档：`docs/delete/2026-03-10_train_models_dedup_inventory.md`

统一执行顺序：

1. **Phase A: 重复收敛与边界解耦（低/中风险）**
   - 先压缩 `direct_pose / contact_plan / phase / event_clock` 的重复 guard、映射和装配样板。
   - 先收口 `EventMotionModel` 内部边界，再进入超长 `forward` 拆分。
2. **Phase B: 巨函数按职责拆分（中/高风险）**
   - 优先拆 `EventMotionModel.forward`，其次拆 `MotionJointLoss.forward`。
   - 一次只拆一个职责块，先编排壳，再纯计算 helper。
3. **Phase C: 静默异常清理（中风险）**
   - 重点清理模型加载兼容、分支 warm-start、tail-loss 选择等热点路径中的广义异常吞没。
4. **Phase D: 隐式状态副作用外移（中风险）**
   - 将 loss 统计、compat 适配、副作用写回从主计算路径剥开，改为明确提交点。

核心原则：
- one step, one commit
- 每步必须有 before/after 结构指标
- 任何一步回归失败，立即停在当前 commit，不继续后续步骤
- 每步固定汇报 4 项：总行数（LOC）、`def`/函数数、最大函数行数、目标重复块数量
- 每步还必须汇报至少 **1 项本轮主题相关的结构债指标**
- **单步硬门禁**：以 Step 收尾为准，必须满足 `LOC_after <= LOC_before`

新增约束（本路线图强制）：
- 不允许继续把新功能堆进 `EventMotionModel.forward`
- 不允许新增新的“兼容壳 + 主实现”双轨共存
- 不允许“只抽函数不删旧逻辑”
- 每次新增 helper 必须带来至少 1 项可量化净收益

### 拆分约束（v2，Phase B 强制执行）

拆分前准入（满足任一即可进入候选）：
- 被调用 >=2 次（或下一步已明确会复用 >=2 次）
- 封装了可独立命名的领域概念
- 拆出后调用处更易读（调用点行数或嵌套层级下降）

硬禁止（命中任一则不允许拆）：
- 纯转发 wrapper / 无实质边界的间接层
- 函数名仅复述代码字面行为（无领域语义）
- 再次引入 `locals()` / `globals()` 一类黑盒上下文
- 只抽函数、不删除原地旧逻辑（双实现并存）
- **Step 收尾时** `LOC_after > LOC_before`
- 参数爆炸：helper 参数 > 8 且未收敛到结构化 context（TypedDict / dataclass）

单次调用 helper 的例外规则（替代绝对禁止）：
- 调用次数 = 1 允许，但必须同时满足：
  - 具备独立概念命名
  - 调用点可读性提升
  - 至少 1 项结构指标下降（最大函数长度 / 重复块 / 异常吞没）

拆分后验流程（强制）：
1. 先按“复用价值”拆分
2. Step 收尾前检查 `LOC_after <= LOC_before`
3. 若不满足：定位并回收不必要间接层（参数搬运层、单次无概念 helper、纯 wrapper）
4. 在**当前 step**完成净减回收后，才允许进入下一步

---

## 0.5) 本轮进展（2026-03-10 / update-5）

本轮已经完成 Phase A 的前四个低/中风险去重点中的前三轮，以及 `MotionJointLoss.forward` 的 aux payload 收口：

- `A1` 已完成 round 1：`direct_pose` split state 读取改为单入口
- `A2` 已完成 round 1：compat 升级路径改为“主流程 + 模板 helper”
- `A2.5` 已完成 round 1：`MotionJointLoss.forward` 的 component submit 模板已统一
- `A2.5 / P0-7` 已完成 round 1：`contact_plan / contact_meas / omega_l2` 的 weighted-loss + stats 样板已并入统一 submit 路径
- `A3` 前置（P0-4）已完成 round 1：`direct_pose` joint spec resolver 已统一
- `A3 / P0-5` 已完成 round 1：`direct_pose` split mask / out_idx builder 已统一
- `A3 / P0-6` 已完成 round 1：`direct_pose` split head / proj builder 已统一

本轮已落地内容：

- `direct_pose` 入口/compat helper：
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
- `def` / 函数数：`71 -> 85`
- 最大函数长度：`EventMotionModel.forward = 2130`（未动）
- 目标重复块数量 #1：`MotionJointLoss.forward` 内联 component submit 模板 `9 -> 0`
- 目标重复块数量 #2：`MotionJointLoss.forward` aux weighted-loss + stats inline cluster `4 -> 0`
- 本轮主题结构债指标 #1：`direct_pose_` token count `630 -> 599`
- 本轮主题结构债指标 #2：`legacy + fallback` marker count `28 -> 28`
- 局部热点长度：
  - `_forward_direct_pose_readout`: `63 -> 53`
  - `_maybe_upgrade_direct_pose_split_state_dict`: `333 -> 228`
  - `MotionJointLoss.forward`: `578 -> 560`
  - `EventMotionModel.__init__`: `1266`

`update-5` 增量（相对 `update-4` 开始前的 P0-7 step）：

- LOC：`6087 -> 6075`
- `MotionJointLoss.forward`：`578 -> 560`
- 目标重复块数量：`contact_plan / contact_meas / omega_l2` weighted-loss + stats cluster `4 -> 0`
- 本轮主题结构债指标：`direct_pose_` token count `599 -> 599`

`update-6` 增量（相对 `update-5` 结束时的 B2 round 1）：

- LOC：`6075 -> 6202`
- `MotionJointLoss.forward`：`560 -> 11`
- 新增 orchestration helper 边界：
  - `train/models.py:5388` `_prepare_forward_inputs`
  - `train/models.py:5585` `_apply_motion_components`
  - `train/models.py:5821` `_apply_direct_pose_component`
  - `train/models.py:6043` `_apply_aux_components`
  - `train/models.py:6055` `_finalize_forward_outputs`

验证状态：

- `python -m py_compile train/models.py` 已通过
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
- `MotionJointLoss.forward` 抽出 helper 区（`train/models.py:5389-6046`）内的广义异常吞没：`9 -> 0`
- 新增窄异常/显式 fallback 语义：
  - `_stats_float_or`：仅用于诊断标量化失败时回退到 `nan/0.0`
  - `rot_local_tail` / `contact_plan`：改为显式 index / shape guard，不再吞没编程错误
  - `direct_pose` / `event_clock` / `contact_meas` / `omega_l2`：active path 改为 fail-fast

验证状态补充：

- `python -m py_compile train/models.py` 已通过
- `python tools/verify_rot_local_tail.py` 已通过

当前结论：

- Phase A 已继续带来可量化重复下降，`P0-7` 已完成，`MotionJointLoss.forward` 辅助 payload 样板已收口
- `Step B2` 已进入 round 1，`MotionJointLoss.forward` 已收敛成 orchestration 壳
- 当前余量从“forward 过长”切换到“helper 内部仍有局部异常吞没与结构债”，Phase A 残余则仍落在 `EventMotionModel.__init__` 的 `contact_plan_* / event_clock_*` builder 边界

---

## 1) 基线现状（针对本轮热点问题）

当前代码快照核对（`train/models.py`）：

- **热点问题 1**：`train/models.py:513` 起的 `EventMotionModel` 类跨度达到 `3957` 行，模型定义、运行时路由、compat 适配、encoder attach 全部缠在一起。
- **热点问题 2**：`train/models.py:2252` 的 `EventMotionModel.forward` 长达 `2130` 行，是当前仓内最重的单函数热点。
- **热点问题 3**：`train/models.py:4471` 起的 `MotionJointLoss` 类跨度 `1613` 行，loss 配置解析、group weighting、统计上报、tail selection、direct pose supervision 全塞在一个类里。
- **热点问题 4**：`train/models.py:1953` 的 `_maybe_upgrade_direct_pose_split_state_dict` 仍有 `228` 行，compat 加载、索引映射、投影 warm-start、旧 ckpt 清理仍在一个 helper 内，但已从首轮去重中明显收缩。
- **热点问题 5**：`train/models.py:1899` 的 `_forward_direct_pose_readout` 目前 `53` 行，已从“分散读取 + 多套 guard”收敛为“统一 split state -> 输出组装”主流程。
- **热点问题 6**：`train/models.py:6064` 的 `MotionJointLoss.forward` 已在 `B2 round 1` 收敛到 `11` 行 orchestration 壳；当前残余转为 helper 内部的异常边界与进一步净减空间。

结构指标基线（本路线图起点）：
- LOC: `6008`
- `def` / 函数数: `71`
- 最大函数长度: `EventMotionModel.forward = 2130`
- 目标重复块计数: `direct split/head/index guard cluster = 6`
- 本轮主题结构债指标 #1: `direct_pose_` token count = `630`
- 本轮主题结构债指标 #2: `legacy + fallback` marker count = `28`

当前快照（update-2）：

- LOC: `6083`
- `def` / 函数数: `82`
- 最大函数长度: `EventMotionModel.forward = 2130`
- 目标重复块计数: `component submit inline cluster = 0`
- 本轮主题结构债指标 #1: `direct_pose_` token count = `599`
- 本轮主题结构债指标 #2: `legacy + fallback` marker count = `28`

`update-5` 说明：

- `P0-7` 以 `LOC_after < LOC_before` 收尾，满足单步硬门禁
- 当前累计快照已回填到 `update-5`

为什么本轮先做 `train/models.py`：

- `train/posttrain.py` 虽然也很大，但你已经做过一轮比较细的整理，现阶段更多是入口编排层的乱。
- `train/training_MPL.py` 的复杂度高，但主要还是 trainer / diagnostics 侧。
- `train/models.py` 目前是“模型语义 + 分支路由 + compat 加载 + loss 逻辑”四类职责最强耦合点，继续放着最容易产生隐性回归和局部修补式膨胀。

---

## 2) 具体改动流程

## Phase A — 重复收敛与边界解耦（A1 + A2 + A2.5 + A3）

### Step A1 — 收敛 direct-pose 分支 guard / index / head 访问（低风险）

目标：将 `train/models.py:1899` 附近 direct split 相关的重复 guard 与索引访问收敛成显式结构化入口。

实施：
- 新增内部结构化 helper（例如 `direct_pose_split_state` builder），集中提供：
  - leg/nonleg/arm/else 输出 head
  - `idx_leg / idx_nonleg / idx_arm / idx_else`
  - split / arm-split enable 标志
- `_forward_direct_pose_readout` 只保留“读结构 -> 写输出”主流程。

约束：
- 先保持字段 1:1 映射，不裁剪现有功能。
- 不改变任何输出 tensor shape / keyset。

验收门：
- `train/models.py` 可正常 import
- direct split guard 重复块计数下降
- `_forward_direct_pose_readout` 长度下降且异常信息集合保持一致

当前状态（update-1）：

- `Completed / round 1`
- 已满足首轮验收门
- 后续不再单独扩展 A1，而是把其残余重复下沉到 A3 的 `direct_pose` init builder/resolver 收敛

### Step A2 — 收敛 compat 升级路径的重复索引/投影搬运样板（中风险）

目标：将 `train/models.py:1953` 附近 ckpt 兼容逻辑拆成几段可命名的纯转换 helper。

建议对象：
- old head -> leg/nonleg weight/bias 分发
- arm/else 本地索引映射
- nonleg projection SVD warm-start
- stale branch tensor 清理

约束：
- 不改变兼容结果，只改组织方式
- 删除原地重复实现，禁止 helper 和旧块双轨并存

验收门：
- `_maybe_upgrade_direct_pose_split_state_dict` 长度显著下降
- compat 路径的 key set / shape set 保持一致

当前状态（update-1）：

- `Completed / round 1`
- 已满足首轮验收门
- 后续残余问题保留在：
  - projection warm-start 继续抽平
  - stale tensor cleanup 是否进一步 helper 化
  - `adapt_legacy_state_dict_` / `load_state_dict` 双壳重复留待 Phase C / D 处理

### Step A2.5 — 收敛 `MotionJointLoss.forward` 的 component-loss 提交模板（中风险）

目标：优先收敛 `MotionJointLoss.forward` 中反复出现的 weighted loss / contrib / stats / register 提交样板。

建议对象：

- `rot_ortho`
- `rot_local`
- `rot_local_tail`
- `root_vel`
- `root_speed`
- `direct_pose`
- `event_clock_lambda_entropy`
- `event_clock_lambda_prior`
- `event_clock_delta_z_l2`

实施：

- 先引入统一的 component submit helper / payload helper
- 主调用点尽量只保留：
  - component 名称
  - raw tensor
  - weight
  - stats payload

约束：

- 先做模板收敛，不立即把整个 `MotionJointLoss.forward` 拆成大量 helper
- 不改变 stats key set
- 不改变 loss 标量语义

验收门：

- `MotionJointLoss.forward` 中重复提交模板数量下降
- 相关 stats key 集合一致
- weighted/raw 指标与现有语义保持一致

当前状态（update-2）：

- `Completed / round 1`
- 已新增统一 helper（后续在 `update-5` 中继续收紧为单一 submit 路径）：
  - `train/models.py:6022` `_setdefault_stats`
  - `train/models.py:6007` `_submit_component_loss`
- 已切换到统一模板的目标分支：
  - `rot_ortho`
  - `rot_local`
  - `rot_local_tail`
  - `root_vel`
  - `root_speed`
  - `direct_pose`
  - `event_clock_lambda_entropy`
  - `event_clock_lambda_prior`
  - `event_clock_delta_z_l2`
- 首轮验收已满足：
  - inline contrib/register 模板 `9 -> 0`
  - 相关 stats key 与 weighted/raw 语义保持一致
- 当前残余：
  - `update-5` 前仍遗留的 `contact_plan` / `contact_meas` / `omega_l2` weighted payload 样板，已在 `P0-7` 收口
  - `MotionJointLoss.forward` 长度仍偏大，现已转为 `Step B2` 候选

### Step A3 — 建立“模型编排壳 + 子系统 helper”边界（中风险）

目标：将 `EventMotionModel.__init__` 的参数使用和子系统装配收敛为明确子模块边界。

实施：
- 先按子系统分块：
  - `contact_plan_*`
  - `contact_phase_state_*`
  - `event_clock_*`
  - `direct_pose_*`
- 将“参数读取 + module/buffer 创建 + 索引初始化”整理为各自 helper
- `update-5` 后，`direct_pose_*` 子系统内的 split builder 重复已基本清空，
  后续优先转向：
  - `contact_plan_* / event_clock_*` builder 边界
  - `MotionJointLoss.forward` 的 orchestration 边界（若进入 `B2`）

约束：
- 暂不改 `__init__` 对外签名
- 不跨文件拆分，先在 `train/models.py` 内收边界

验收门：
- `EventMotionModel.__init__` 长度下降
- 至少 1 项主题结构债指标下降

当前状态（update-5）：

- `In progress / direct-pose builder cluster cleared`
- 已完成 `direct_pose` 子系统中最先暴露出的 resolver 去重：
  - `train/models.py:65` `_normalize_joint_spec_items`
  - `train/models.py:93` `_resolve_joint_spec_indices`
  - `train/models.py:133` `_resolve_rot6d_joint_count`
- 已完成 split mask / out_idx builder 去重：
  - `train/models.py:1110` `build_split_out_index`
- 已完成 split head / proj builder 去重：
  - `train/models.py:1489` leg branch
  - `train/models.py:1504` arm branch
  - `train/models.py:1509` else branch
  - `train/models.py:1515` nonleg branch
  - `train/models.py:1818` `_build_split_head_branch`
- 已替换的初始化调用点：
  - leg joint spec：`train/models.py:1027`
  - split-leg joint spec：`train/models.py:1062`
  - arm joint spec：`train/models.py:1132`
  - leg/nonleg split index：`train/models.py:1131`
  - arm/else split index：`train/models.py:1151`
- 当前残余优先留在：
  - `contact_plan_* / event_clock_*` builder 边界
  - `MotionJointLoss.forward` 的 orchestration 壳拆分（见 `B2`）

### Phase A 当前进度（2026-03-10）

- Step A1（Completed / round 1，P0）：已完成 direct split 读写口收口
- Step A2（Completed / round 1，P0）：已完成 compat state_dict 升级块第一轮去重
- Step A2.5（Completed / round 1，P0）：已完成 `MotionJointLoss.forward` component submit 模板收敛
- Step A2.5 / P0-7（Completed / round 1，P0）：已完成 `contact_plan / contact_meas / omega_l2` weighted-loss + stats 样板去重
- Step A3（In progress / round 1 direct-pose-builder-cleared，P0）：已完成 joint spec resolver 去重 + split mask/out_idx builder 去重 + split head/proj builder 去重，当前残余转向 `contact_plan_* / event_clock_*` builder 边界
- Phase B 准入条件（强制）已满足；`P0-7` 净减已回收，当前可开始评估 `B2`
- P0 操作清单见：`docs/delete/2026-03-10_train_models_dedup_inventory.md`

---

## Phase B — 巨函数职责拆分（B1 + B2）

### Step B1 — 拆 `EventMotionModel.forward`（高风险）

建议边界：
- 输入预处理 / history 对齐
- contact plan / phase state / event clock 更新
- direct pose 路由
- 主输出组装
- debug / aux 输出组装

强制要求：
- 拆完后 `forward` 只保留编排壳
- 每拆一块都删除原地对应块，禁止双实现并存
- 目标长度：`2130 -> <= 500`

回归门：
- `python -m py_compile train/models.py`
- 依赖模型正向的最小 smoke 路径通过
- 关键输出 key 集合保持一致

### Step B2 — 拆 `MotionJointLoss.forward`（高风险）

建议边界：
- base loss / rot_local / root velocity
- direct pose supervision
- contact / event-clock / adaptive weighting
- stats 累加 / component register / final loss 汇总

强制要求：
- `MotionJointLoss.forward` 从“全量逻辑主体”收敛成 loss orchestration 壳
- 目标长度：`578 -> <= 180`

回归门：
- loss key 集合一致
- 核心 loss 标量名一致
- A/B 对照允许浮点微小扰动，不允许 keyset 漂移

### Phase B 当前进度（2026-03-11）

- Step B1（Deferred）：仍后置，先不与 `EventMotionModel.forward` 大拆分并行推进
- Step B2（In progress / round 2 exception-narrowed）：`MotionJointLoss.forward` 保持 `11` 行，rot/root、direct-pose、aux、finalize helper 内的广义异常吞没已清空，tail/direct/aux 现分流为显式 guard 或 fail-fast

---

## Phase C — 静默异常清理（C1 + C2）

### Step C1 — 建立异常点清单与级别（低风险）

目标：把 `train/models.py` 中当前“为兼容/诊断而保留”的宽泛异常先列表，不立即全部删除。

优先清单：
- `_maybe_upgrade_direct_pose_split_state_dict`
- `adapt_legacy_state_dict_`
- `MotionJointLoss.forward` 中 tail-loss / fallback 相关异常块

分类建议：
- 可降级为窄异常
- 应 fail-fast
- 可保留但必须记录 fallback 发生次数

验收门：
- 形成位置/原因/fallback 行为清单

当前清单（`MotionJointLoss.forward` 抽出 helper）：
- C1：`rot_ortho_fallback` / `event_clock_lambda_mean` / loss-group contrib 仅保留“诊断标量化失败 -> `nan/0.0`”的窄异常
- C2：`direct_pose`、`contact_plan`、`event_clock`、`contact_meas`、`omega_l2` active path 不再吞没 `Exception`
- C2：`rot_local_tail` candidate / top-k 选择改为显式边界检查，fallback 语义为“无有效候选则跳过 tail term”

### Step C2 — 清理热点路径中的广义异常吞没（中风险）

目标：优先把模型主路径里的 `except Exception` 缩窄。

约束：
- 先处理热点路径，再碰外围 debug 路径
- 每处替换必须明确 fallback 语义

验收门：
- 广义异常吞没计数下降
- compat / forward / loss 回归路径保持一致

当前进度：
- `MotionJointLoss.forward` 抽出 helper 区的 `except Exception` 已降为 `0`
- 余下 Phase C 热点回到 compat / debug / geometry 旁路，不再阻塞 `B2` 主路径

---

## Phase D — 副作用剥离（D1 + D2）

### Step D1 — 将 compat 改写从主加载路径剥离为显式转换层（中风险）

目标：避免 `EventMotionModel` 主体同时承担“当前实现”和“历史 checkpoint 修复器”的双职责。

方向：
- `adapt_legacy_state_dict_` 收敛为薄壳
- 兼容转换逻辑变成独立 helper 组
- 主加载路径只负责“是否需要转换 + 调用转换 + 返回结果”

### Step D2 — 将 loss 统计写入与主计算分离（中风险）

目标：减少 `MotionJointLoss.forward` 里“算 loss”与“填 stats/registry”交织。

方向：
- 主计算先得到 component payload
- 统一在 finalize 阶段写入 stats / registry

---

## 3) 本轮建议的实际起手顺序

建议顺序：

1. `A1`: 收敛 `_forward_direct_pose_readout`（已完成 round 1）
2. `A2`: 收敛 `_maybe_upgrade_direct_pose_split_state_dict`（已完成 round 1）
3. `A2.5`: 收敛 `MotionJointLoss.forward` 的 component submit 模板（已完成 round 1）
4. `A3`: 继续收敛 `direct_pose` 的 split-head init 样板，再扩展到 `EventMotionModel.__init__` 子系统边界
5. `B1`: 拆 `EventMotionModel.forward`
6. `B2`: 拆 `MotionJointLoss.forward`
7. `C1/C2`: 清理广义异常吞没
8. `D1/D2`: compat 与 stats 副作用外移

理由：

- 第一优先目标是先把重复面收窄，而不是立刻拆超长函数。
- `A1/A2` 已证明 direct-pose 局部边界先收口是有效路径
- `P0-7` 完成后，`MotionJointLoss.forward` 的辅助 payload 重复已不再是首要阻塞
- 再碰超长函数，能进一步降低拆 `forward` / `loss.forward` 时的上下文噪音
- 当前更自然的下一跳，是在 `B2` 中利用已收紧的 payload 路径把 `MotionJointLoss.forward` 收敛成 orchestration 壳
- 具体的 P0 去重对象与证据，放在子文档单独维护，避免主路线图被细节淹没

---

## 4) 后续优先级（文件级）

当前建议的仓内清理优先级：

1. `train/models.py`
2. `train/training_MPL.py`
3. `train/posttrain.py`

说明：

- `train/posttrain.py` 不是没问题，而是相对已经有过一轮整理，目前不是第一爆点
- `train/training_MPL.py` 仍然是第二顺位大热点，但建议等 `train/models.py` 的模型/损失边界先清出来，再继续拆 trainer

---

## 5) 建议验证命令（每步最小固定集）

```bash
python -m py_compile train/models.py train/posttrain.py train/training_MPL.py
python tools/check_standard_rotvec_semantics.py
python tools/check_lambda_fusion_blend_geometry.py
```

如果某一步已触及模型正向/损失行为，再补：

```bash
python train/posttrain.py --help
python train/training_MPL.py --help
```

---

## 6) 备注

- 这份路线图只记录“为什么现在先动 `train/models.py`”以及建议拆分顺序。
- 现阶段不主张同时大拆 `train/models.py` 和 `train/training_MPL.py`，否则回归面会过大。
