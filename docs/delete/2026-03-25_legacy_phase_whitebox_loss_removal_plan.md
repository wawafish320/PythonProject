# 2026-03-25 legacy `contact_phase_state` / `phase_z_in` / whitebox / old loss-key 删除计划

Date: 2026-03-25  
Status: Draft / Active  
Scope: `train/phase_state_compat.py`, `train/models.py`, `train/posttrain.py`, `train/training_MPL.py`, `train/validate/run_freerun_cycles.py`, related legacy tools  
Goal: 把已经退出 mainline 的历史路径从代码里真正删掉，而不是继续以 compat / fallback / guard 形式残留。  
Non-goal: 本文档不定义新的 phase 机制，也不重设计新的 contact pipeline。

---

## 0) TL;DR

当前仓库里还有 4 组“文档已退场、代码未退场”的残留：

1. `contact_phase_state`：
   - 已不属于 mainline，但 compat 映射与 token guard 仍在。
2. `phase_z_in`：
   - 已不是当前主线文档入口，但 direct head / posttrain / freerun 仍保留完整路径。
3. whitebox contact：
   - 文档侧已从主线移除，但 `training_MPL` 与部分 validate/runtime 分支仍保留 whitebox helper / source 选择。
4. old loss keys：
   - 当前状态是“拒绝接受”；后续口径应统一成“已移除”，避免继续给人一种“还能兼容，只是禁用”的感觉。

这 4 组应该按“先删入口与 compat，再删实现与工具，再删守卫”的顺序收口。

---

## 1) 删除目标与证据

### 1.1 `contact_phase_state` compat / guard 残留

当前证据：

- compat key 映射仍在：
  - `train/phase_state_compat.py`
- 运行时 guard 仍把该 token 当作“禁止出现但仍需防守”的历史物：
  - `tools/check_posttrain_legacy_code_guard.py`
- 一些旧工具还在扫描 / 传递该配置：
  - `tools/run_stage6_plantransplant_compare.py`
  - `tools/run_cp015_oldplan_component_ablation.py`
  - `tools/run_cp015_oldplan_downstream_chain.py`
  - `tools/run_stage6_nline_only_min_confirm.py`
  - `tools/diag7_knn_input_vs_delta.py`

目标状态：

- 删除 `train/phase_state_compat.py`
- 删除所有 `contact_phase_state_*` compat key / state key / prefix map
- 删除仅用于防回归该 token 的 guard / checklist / tool 扫描逻辑
- 删除所有只服务于旧 phase-state 路线的脚本参数透传

---

### 1.2 `phase_z_in` phase-hint 路径残留

当前证据：

- posttrain config 仍暴露：
  - `train/posttrain.py`
  - `direct_pose_use_phase_z`
  - `direct_pose_phase_z_mode`
- model forward 仍构造 / 路由：
  - `train/models.py`
  - `phase_z_in_direct`
  - `replace_contacts` 路由
- freerun 仍导出 / 记录：
  - `train/validate/run_freerun_cycles.py`
  - `phase_z_in`
- 多个旧实验工具仍显式打开该开关：
  - `tools/run_oldd1_newflow_chain.py`
  - `tools/run_oldd1_skip70b_replace_compare.py`
  - `tools/run_cp015_oldplan_component_ablation.py`
  - `tools/run_stage6_phasefrontload_compare.py`

目标状态：

- 从 `PostTrainConfig` 与相关 config IO 中删除 `direct_pose_use_phase_z` / `direct_pose_phase_z_mode`
- 从 `EventMotionModel` 中删除 `phase_z_in_direct` 输入装配与 `replace_contacts` 路由
- 从 `run_freerun_cycles` 中删除 `phase_z_in` 导出、summary、probe 字段
- 删除只围绕 `phase_z_in` 做对照实验的旧工具入口

---

### 1.3 whitebox contact 路径残留

当前证据：

- basetrain helper 仍在：
  - `train/training_MPL.py`
  - `_contact_meas_whitebox(...)`
- basetrain source 仍允许 whitebox：
  - `train/training_MPL.py`
  - `--trainbase_contacts_source {auto, whitebox, pretrain_contact}`
- freerun / validate 仍保留 whitebox runtime path 与调试日志：
  - `train/validate/run_freerun_cycles.py`
  - `--contacts_meas_source` 仍包含 `whitebox`
  - `--log_contacts_whitebox*`
- 仍有专门围绕 whitebox 的历史工具：
  - `tools/summarize_freerun_contact_whitebox_peaks.py`
  - `tools/diagnose_freerun_ttd_anchor.py`

目标状态：

- 从 `training_MPL` 删除 `_contact_meas_whitebox`
- 从 `training_MPL` 删除 basetrain `whitebox` source 分支与相关 CLI
- 从 `run_freerun_cycles` 删除 `whitebox` meas source、fallback、debug log 路径
- 删除 whitebox 专用分析脚本，或改为只读 archive 外部产物而不再依赖 runtime 开关

---

### 1.4 old loss keys：从“禁用”改成“已移除”

当前证据：

- `MotionJointLoss` 仍保留 `legacy_kwargs` 拦截：
  - `train/models.py`
- `training_MPL` 仍保留 `LEGACY_LOSS_KEYS` / `LEGACY_LOSS_TOPLEVEL_KEYS` / schedule 校验：
  - `train/training_MPL.py`

当前口径问题：

- 代码行为其实已经是“拒绝接受”
- 但文案还容易让人理解成“deprecated / 暂时禁用 / 也许还能兼容”

目标状态：

- 对外口径统一改成：**旧 loss key 已移除**
- 报错文案明确写“removed”
- 后续如果配置树完全清干净，再评估是否把多层 legacy guard 进一步收缩

---

## 2) 推荐删除顺序

### P0. 先删入口 / 文案 / checklist

- 文档与 checklist 不再提 retired 路线
- `legacy loss key` 统一改成 “removed”
- 工具注释 / help 文本不再把 whitebox / phase-state 说成现行路径

### P1. 删 config / CLI 暴露面

- `train/posttrain.py`：删除 `direct_pose_use_phase_z` / `direct_pose_phase_z_mode`
- `train/training_MPL.py`：删除 `whitebox` basetrain source 选项
- `train/validate/run_freerun_cycles.py`：删除 `whitebox` meas source 与相关 debug flags

### P2. 删实现分支

- `train/models.py`：删除 `phase_z_in_direct` 路由与 replace-contacts 逻辑
- `train/training_MPL.py`：删除 `_contact_meas_whitebox`
- `train/phase_state_compat.py`：整体删除

### P3. 删历史工具 / 守卫

- 删除只服务于 retired 路线的工具脚本
- 删除仅为防止旧 token 重新进入而存在、但已不再有主实现依赖的 guard

---

## 3) 验收条件

完成后应满足：

- `rg "contact_phase_state" train tools` 只剩 archive / 问题记录，runtime 主路径为 0
- `rg "phase_z_in" train tools` 不再出现在当前主训练 / 主验证路径
- `rg "_contact_meas_whitebox|log_contacts_whitebox|trainbase_contacts_source.*whitebox" train tools` 为 0
- `train/models.py` 与 `train/training_MPL.py` 对旧 loss key 的口径统一为 “removed”
- `docs/posttrain_pipeline.md` 与主代码的 runtime contract 一致，不再出现“文档删了，代码还留着”的双轨状态

---

## 4) 备注

这份文档的目的不是“马上重构所有历史实验资产”，而是明确：

- 哪些是当前 mainline 不应继续携带的历史包袱
- 哪些删除动作应优先做
- 删除完成后，哪些 guard / compat 才有资格继续一起删掉
