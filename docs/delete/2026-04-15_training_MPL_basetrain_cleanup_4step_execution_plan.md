# 2026-04-15 `train/training_MPL.py` basetrain 清理四步执行文档

Date: 2026-04-15  
Status: Draft / Execution-Ready  
Scope: `train/training_MPL.py`、fresh basetrain active config、与 train entry 直接耦合的少量 helper / docs  
Primary Goal: 按当前 fresh basetrain -> posttrain 主链的真实 contract，系统性收掉 `train/training_MPL.py` 里已经退役但仍残留的 parser / wrapper / archaeology / dead-arg 分支。  
Non-goal:

- 不改 `train/posttrain.py` 的主链逻辑
- 不改 `MotionJointLoss` 数学定义
- 不改 `ckpt_last_*` handoff 机制
- 不删 fresh 主链仍在使用的 `freerun_stage_schedule`
- 不动 frozen `pretrain_contact` 这条 basetrain rollout contract

关联文档：

- fresh 主链 runbook：`docs/basetrain_to_posttrain_top7_fresh_chain_runbook.md`
- legacy / compat 审计：`docs/delete/2026-04-14_train_legacy_compat_deletion_audit.md`

---

## 0. TL;DR

当前 `train/training_MPL.py` 的 basetrain 入口，已经和旧阶段相比明显收窄：

1. donor 只认 `ckpt_last_{run_name}.pth`
2. basetrain rollout contact 固定走 frozen `pretrain_contact`
3. fresh basetrain 仍依赖当前 `freerun_stage_schedule`
4. old-boundary / whitebox / parser-swallow / CLI wrapper 这类“历史包袱”不再是主链必需

因此推荐按下面 4 步执行：

1. **先清 config schema**
2. **再删 dead args 与旧 wrapper**
3. **再删除 parser swallow compat 层**
4. **最后把 whitebox contacts 从 train entry 拆走**

这 4 步的顺序不能反过来。  
原因是 Step 3 会把“静默兼容旧字段”改成 fail-fast；如果不先做 Step 1/2，活跃 basetrain config 会先炸。

---

## 1. 当前主链 contract

在开始删除前，先固定哪些东西是 **不能误删** 的：

### 1.1 fresh basetrain donor contract

- donor 固定使用 `ckpt_last_{run_name}.pth`
- 不再依赖 `ckpt_epoch_014.pth`

来源：

- `docs/basetrain_to_posttrain_top7_fresh_chain_runbook.md:15`
- `train/training_MPL.py:2932`

### 1.2 basetrain rollout contact contract

- fresh basetrain rollout 现在固定通过 frozen encoder + contact head 解析 `pretrain_contact`
- `contact_plan_enable=True` 时，如果这条 frozen path 不存在，入口直接 fatal

来源：

- `docs/basetrain_to_posttrain_top7_fresh_chain_runbook.md:33`
- `train/training_MPL.py:1214`

### 1.3 stage schedule contract

- `freerun_stage_schedule` 仍然是当前 basetrain 的 live contract
- 但它已经不再承载旧 freerun-loss / noise branches
- 当前口径只保留：TF / LR / history / direct-pose trainability 覆盖

来源：

- `train/training_MPL.py:352`
- `train/training_MPL.py:391`
- `train/training_MPL.py:4295`

### 1.4 当前 active top7 basetrain config 仍残留的 stale keys

本轮 active config：

- `config/exp_phase_DirectBranch_v1_d1_cp015_tailk7_rankmix_tw020_corridor_hold_tail15_phasea050_fixedsched_ep014center_control_denseckpt_seed2024_20260401.json`

其中仍残留 6 个 stale keys：

- `rot_local_tail_rank_mix`
- `rot_local_tail_reduce`
- `rot_local_tail_uniform_mix`
- `save_fit_ckpt_epochs`
- `seed`
- `trainbase_contacts_source`

来源：

- `config/exp_phase_DirectBranch_v1_d1_cp015_tailk7_rankmix_tw020_corridor_hold_tail15_phasea050_fixedsched_ep014center_control_denseckpt_seed2024_20260401.json:36`
- `config/exp_phase_DirectBranch_v1_d1_cp015_tailk7_rankmix_tw020_corridor_hold_tail15_phasea050_fixedsched_ep014center_control_denseckpt_seed2024_20260401.json:40`
- `config/exp_phase_DirectBranch_v1_d1_cp015_tailk7_rankmix_tw020_corridor_hold_tail15_phasea050_fixedsched_ep014center_control_denseckpt_seed2024_20260401.json:226`
- `config/exp_phase_DirectBranch_v1_d1_cp015_tailk7_rankmix_tw020_corridor_hold_tail15_phasea050_fixedsched_ep014center_control_denseckpt_seed2024_20260401.json:261`

注意：

- runbook 当前只显式删除了前 5 个键
- `trainbase_contacts_source` 也应该纳入退役范围，因为 fresh runbook 已经把 contact source 固定成 shell/runtime contract，而不是 `training_MPL` config key

来源：

- `docs/basetrain_to_posttrain_top7_fresh_chain_runbook.md:83`

---

## 2. 四步执行总览

| Step | 名称 | 目标 | 风险 | 建议优先级 |
|---|---|---|---|---|
| 1 | 清 config schema | 先把 active config 里的 stale keys 清掉 | 低 | P0 |
| 2 | 删 dead args + 旧 wrapper | 收掉已经悬空或只剩历史意义的 train entry 外壳 | 中低 | P0 |
| 3 | 删 parser swallow compat | 把“静默忽略旧字段”改成硬边界 | 中 | P1 |
| 4 | 拆出 whitebox contacts | 让 `training_MPL` 只保主训练路径，不再挂 archaeology plumbing | 中高 | P1 |

执行顺序：

- 必须 `1 -> 2 -> 3 -> 4`

---

## 3. Step 1 — 清 config schema

### 3.1 目标

先把 active basetrain config 从“依赖 sanitization 脚本才能跑”收回到“原始 config 自身就合法”。

这个阶段只动：

- active basetrain config
- runbook 中的 sanitized key list
- 必要时补一份 clean config

### 3.2 要删除的 config 键

从 active basetrain config 中移除：

- `rot_local_tail_rank_mix`
- `rot_local_tail_reduce`
- `rot_local_tail_uniform_mix`
- `save_fit_ckpt_epochs`
- `seed`
- `trainbase_contacts_source`

其中：

- 前 3 个已经和当前 `MotionJointLoss` 口径脱节
- `save_fit_ckpt_epochs` 已经不再驱动任何 checkpoint 保存逻辑，当前只保存 `ckpt_last`
- `seed` 在 `training_MPL` parser 中不是合法字段
- `trainbase_contacts_source` 已经被 fresh runbook 的 `CONTACT_SOURCE=pretrain_contact` runtime contract 取代

### 3.3 本步不删代码

本步不要求马上删除 `training_MPL.py` 里的 compat 代码。  
目标只是先把 active 配置树清干净，避免后续 Step 3 把主链直接打爆。

### 3.4 修改项

建议修改：

- active config：
  - `config/exp_phase_DirectBranch_v1_d1_cp015_tailk7_rankmix_tw020_corridor_hold_tail15_phasea050_fixedsched_ep014center_control_denseckpt_seed2024_20260401.json`
- runbook 文案：
  - `docs/basetrain_to_posttrain_top7_fresh_chain_runbook.md`

### 3.5 验收条件

- `python3 - <<'PY'` 方式检查 active config 对 `training_MPL` parser 不再产生 unknown keys
- unknown keys 从当前的 6 个降到 0
- runbook 的 sanitized 描述与实际 config 残留一致，不再漏掉 `trainbase_contacts_source`

### 3.6 完成后收益

- basetrain config 不再依赖“先 sanitize 才合法”
- 为 Step 3 的 parser fail-fast 铺平道路

---

## 4. Step 2 — 删 dead args 与旧 wrapper

### 4.1 目标

把 `training_MPL.py` 里已经悬空、无主链消费、或只剩旧壳意义的 train-entry 参数和 wrapper 收掉。

### 4.2 建议删除项

#### A. `main()` 外层 wrapper 旧壳

删除或改成极薄直通：

- `--arpg_export_only`
- `allow_val_on_train`
- `val_ratio`
- 手工构造 `GLOBAL_ARGS = SimpleNamespace(...)`
- `"[ARPG-PATCH]"` 包装日志

相关位置：

- `train/training_MPL.py:5139`

判断依据：

- 这些逻辑只在 `main()` 层出现
- 当前 fresh basetrain 主链不依赖它们
- 真正的训练解析已经由 `_parse_train_entry_args()` 完成

#### B. 明显悬空的训练参数

建议移除：

- `warmup_steps`
- `min_lr_ratio`
- `tf_warmup_steps`
- `tf_total_steps`
- `foot_contact_threshold`

相关位置：

- `train/training_MPL.py:3962`
- `train/training_MPL.py:4007`
- `train/training_MPL.py:4211`
- `train/training_MPL.py:4371`

判断依据：

- `warmup_steps` / `min_lr_ratio` 只在 parser 中出现，没有训练逻辑消费
- `tf_warmup_steps` / `tf_total_steps` 只是传进 `Trainer.__init__` 并保存字段，没有后续使用
- `foot_contact_threshold` 只有 dataclass / parser / runtime 透传，没有实际计算逻辑消费

### 4.3 本步建议保留

不要在这一步删除：

- `export_onnx_step_stateful_nophase(...)`
- `freerun_stage_schedule`
- `trainbase_contacts_pretrain_clamp`
- `trainbase_contacts_pretrain_affine_stats`

原因：

- ONNX 导出函数仍被 `train/export_onnx_from_ckpt.py` 直接导入
- 其余项仍属于 fresh basetrain 主链的一部分

### 4.4 验收条件

- `python3 -m train.training_MPL --help` 正常输出
- active config 仍能过 parser
- `train/export_onnx_from_ckpt.py` 不因 wrapper 变化而失效

### 4.5 完成后收益

- train entry 更薄
- 参数面更接近实际运行 contract
- 减少“看起来还能调，其实完全不生效”的假开关

---

## 5. Step 3 — 删除 parser swallow compat 层

### 5.1 目标

把 `training_MPL.py` 从“静默吞掉旧字段”改成“明确拒绝旧字段”。

### 5.2 目标对象

删除：

- `LEGACY_IGNORED_TRAIN_ENTRY_KEYS`
- `LEGACY_IGNORED_TRAIN_ENTRY_FLAGS`
- `_parse_train_entry_args()` 里对这组 legacy flags 的清洗逻辑
- `_load_train_entry_config_defaults()` 里对这组旧 key 的静默过滤逻辑

相关位置：

- `train/training_MPL.py:368`
- `train/training_MPL.py:379`
- `train/training_MPL.py:3846`
- `train/training_MPL.py:4233`

### 5.3 这一步的前提

必须先完成 Step 1。  
否则 active config 仍然会依赖 parser swallow layer，删除后会直接 parse error。

### 5.4 推荐策略

分两小步：

1. 先把这组 ignored keys 从 active configs / runbook / tools 里清空
2. 再删除 `training_MPL.py` 里的 swallow layer

### 5.5 本步建议保留

暂时保留下面这组“removed-key fail-fast guard”：

- `LEGACY_LOSS_KEYS`
- `LEGACY_LOSS_TOPLEVEL_KEYS`
- `_legacy_loss_keys_msg(...)`
- `_assert_no_legacy_loss_keys_in_schedule(...)`

原因：

- 它们现在不是 compat 行为，而是更早、更友好的 removed-key 报错层
- 在彻底清干净 archive/tooling 之前，保留 fail-fast 更稳

### 5.6 验收条件

- active basetrain config 直接 `--config_json` 可解析
- 对任一旧 key / 旧 flag，不再是“静默忽略”，而是明确报错
- runbook 不再需要“依赖 parser 帮忙吞旧字段”

### 5.7 完成后收益

- config schema 从“软边界”变成“硬边界”
- 后续维护时，不会再出现“为什么这个字段写了没效果”的隐性兼容问题

---

## 6. Step 4 — 把 whitebox contacts 从 train entry 拆走

### 6.1 目标

让 `train/training_MPL.py` 只保 basetrain 主训练路径，不再挂着 whitebox archaeology / diagnostics plumbing。

### 6.2 当前现状

`training_MPL.py` 里自己已经写明：

- basetrain rollout contact 现在固定走 frozen `pretrain_contact`
- whitebox contacts 这套 knob 只保留给 non-rollout diagnostics / archaeology helper

相关位置：

- `train/training_MPL.py:1214`
- `train/training_MPL.py:4173`

此外：

- `_contact_meas_whitebox(...)` 在 `training_MPL.py` 内部没有主训练调用
- 外部主要由 `train/validate/run_freerun_cycles.py` 使用

### 6.3 推荐拆法

#### A. 先从 train entry 去参数面

从 `training_MPL.py` parser / runtime config 中移除：

- `contact_meas_gate_by_hit`
- `contact_meas_ground_z_mode`
- `contact_meas_ground_z_beta`
- `contact_meas_ground_z_window`
- `contact_meas_ground_z_quantile`
- `contact_meas_ground_z_slew_up_cm`
- `contact_meas_ground_z_slew_down_cm`
- `foot_contact_threshold`

相关位置：

- `train/training_MPL.py:4177`
- `train/training_MPL.py:4370`
- `train/training_MPL.py:4421`

#### B. 再把 whitebox helper 下沉

将以下实现迁到更合理的位置：

- `train/validate/run_freerun_cycles.py`
- 或单独 diagnostics helper 模块

待迁移实现：

- `_resolve_contact_meas_cfg(...)`
- `_resolve_contact_meas_bone_names(...)`
- `_resolve_contact_meas_foot_indices(...)`
- `_compute_contact_meas_ground_z(...)`
- `_compute_contact_meas_whitebox_state(...)`
- `_build_contact_meas_whitebox_debug(...)`
- `_contact_meas_whitebox(...)`

相关位置：

- `train/training_MPL.py:3281`
- `train/training_MPL.py:3629`

### 6.4 为什么不建议直接硬删实现

因为 validation 侧仍有 live caller：

- `train/validate/run_freerun_cycles.py:4025`

所以这一步的正确动作是：

- **先把 train entry 不再暴露这套功能**
- **再把 helper 下沉到 validation / diagnostics**
- **最后再从 `training_MPL.py` 删除实现**

### 6.5 验收条件

- basetrain 主训练路径不再暴露 whitebox knobs
- `training_MPL.py` 不再承担 archaeology diagnostics 责任
- validation 侧仍能独立运行 whitebox 分析（如果还需要保留）

### 6.6 完成后收益

- `training_MPL.py` 职责更单一
- basetrain train entry 与 validation archaeology 解耦
- 训练主链不再背着大量“历史实验 plumbing”

---

## 7. 不建议本轮删除的内容

以下内容虽然也看起来“旧”，但不建议放进这 4 步：

### 7.1 `freerun_stage_schedule`

原因：

- 当前 basetrain 仍真实依赖它
- 只是已删掉旧 freerun-loss / noise branches，不代表整个 schedule 系统过时

相关位置：

- `train/training_MPL.py:391`
- `train/training_MPL.py:4295`

### 7.2 `ckpt_last` 保存逻辑

原因：

- 当前 fresh basetrain donor handoff 就依赖它

相关位置：

- `train/training_MPL.py:2926`
- `docs/basetrain_to_posttrain_top7_fresh_chain_runbook.md:15`

### 7.3 frozen `pretrain_contact` rollout contract

原因：

- 这是当前 fresh basetrain 的核心运行 contract，不是 legacy

相关位置：

- `train/training_MPL.py:1214`
- `docs/basetrain_to_posttrain_top7_fresh_chain_runbook.md:33`

### 7.4 ONNX 导出函数本体

原因：

- 虽然 fresh runbook 不直接依赖，但 `train/export_onnx_from_ckpt.py` 仍直接导入

相关位置：

- `train/training_MPL.py:4985`
- `train/export_onnx_from_ckpt.py:43`

如果要收边界，优先做法是：

- 把“训练后自动导出”改成 opt-in
- 不是直接删掉导出函数

---

## 8. 推荐执行顺序与提交切分

建议按 4 个独立 commit / patch 来做：

### Patch 1 — config 清理

- 清 active basetrain config stale keys
- 更新 fresh runbook 的 sanitized key 清单

### Patch 2 — train entry 瘦身

- 删除 dead args
- 删除 `main()` 外层旧 wrapper
- 保持 `_parse_train_entry_args()` 为唯一正式入口

### Patch 3 — schema 硬边界

- 删除 parser swallow compat 层
- 让旧 key / 旧 flag 直接 fail-fast

### Patch 4 — whitebox 解耦

- 先迁 helper
- 再从 `training_MPL.py` 移除 whitebox 参数与实现

---

## 9. 每步最小验收清单

### Step 1

- [ ] active basetrain config 不再含 6 个 stale keys
- [ ] runbook 的 sanitized 描述与 config 实际状态一致

### Step 2

- [ ] `python3 -m train.training_MPL --help`
- [ ] `python3 -m py_compile train/training_MPL.py`

### Step 3

- [ ] active config 直接 `--config_json` 可解析
- [ ] 旧 key / 旧 flag 不再静默吞掉

### Step 4

- [ ] basetrain train entry 不再暴露 whitebox knobs
- [ ] validation/diagnostics 若仍需要 whitebox，调用路径独立可用

---

## 10. 最终目标状态

完成这 4 步后，`train/training_MPL.py` 的 basetrain 入口应该收敛到下面这条口径：

- active config 自身合法，不依赖 sanitize 才能解析
- train entry 只暴露真实生效的参数
- 旧字段不再被静默兼容
- whitebox archaeology 不再挂在训练主入口
- 主链 contract 明确且稳定：
  - `ckpt_last`
  - frozen `pretrain_contact`
  - current `freerun_stage_schedule`

这时再往下做更细的函数级拆分或大规模删旧代码，风险会小很多。
