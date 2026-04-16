# 2026-04-13 `train/posttrain.py` 热点重构路线图（v1）

Date: 2026-04-13  
Status: Draft / Active v1 / update-6（P1 + P2 热点压缩已完成；P3 已做两轮保守净减，当前仍建议先 smoke/stabilize，再决定是否继续）  
Scope: `train/posttrain.py`（本轮只做结构净减、边界收口、职责重排；不改训练语义、不改 checkpoint 兼容策略、不改默认超参）  
Goal: 在**不改变语义/行为**前提下，优先降低 `train/posttrain.py` 的“薄 wrapper 过多 + 单文件巨函数过重 + rollout/build/config 三团职责缠绕”维护风险。  
Non-goal: 不改模型数学定义、不改 loss 公式、不改 CLI 默认行为、不做跨文件重构。

---

## 0) 当前策略（先净减，后拆分；优先回收无边界 helper，再处理真正的巨函数热点）

本轮总原则：

- **第一优先策略不是继续新增 helper，而是先做净减法回收。**
- 只有当“单点调用 + 纯转发 + 无真实 contract”的 helper 先回收后，才进入后续巨函数拆分。
- 对于 `build/compat`、`rollout kernel`、`soft-fail policy` 这类真实边界，默认不为了追求 `def` 数下降而回收。

统一执行顺序：

1. **Phase A: 薄 wrapper 净减（低风险）**
   - 回收 single-call、近邻、纯搬运/纯组包 helper。
   - 禁止把真实 contract helper 一并打平。
2. **Phase B: 巨函数按稳定语义块拆分（中风险）**
   - 优先处理 CLI/build/finalize 这类“局部超长但语义段落明确”的函数。
   - 只有在拆出后调用点更清楚时才新增 helper。
3. **Phase C: rollout 轨道收边界（中风险）**
   - 收紧 `prepare -> step -> finalize` 三段的 context contract。
   - 避免重新长出一批 dict-pack / wrapper helper。
4. **Phase D: config/parser 压缩（中风险）**
   - 不打散 schema contract；只压缩 parser 内部局部样板和后处理块。

核心原则：

- one step, one commit
- 每步必须有 before/after 结构指标
- 任何一步如果只“抽函数不删旧逻辑”，视为失败
- 任何新增 helper 必须带来至少 1 项净收益：`LOC`、最大函数长度、重复块数量、或 single-use helper 数下降

本文件专用约束：

- 不允许为了“拆分看起来更整齐”再引入新的 `dict-pack helper`
- 不允许新增只服务单一调用点、且仅做 `getattr / setattr / state_dict.get / default` 搬运的 wrapper
- 不允许把带独立 warning / fatal / compat policy 的 helper 误收回大函数

---

## 0.5) 本轮进展（2026-04-13 / update-6）

本轮已经完成两轮低风险净减、P1 的三个 giant hotspot 压缩、P2 的三个热点压缩，并对 P3 做了两轮保守净减：先压缩 `_lambda_fusion_finalize` 内部重复聚合，再把 `_lambda_rollout_apply_direct_leg_adjustments` 收成 early-return 主干，不新增 top-level helper，不改变 loss 数学或 rollout/direct-pose 语义。

已完成的回收：

- round 1：
  - `_finite`
  - `_rollout_cond_raw_idx`
  - `_resolve_rollout_step_tensor`
  - `_build_rollout_unroll_ctx`
  - `_lambda_rollout_unroll_steps`
  - `_train_mode_display_name`
  - `_unfreeze_lambda_fusion`
- round 2：
  - `_predict_pretrain_contacts_from_frozen`
  - `_resolve_rollout_time_index`
  - `_append_leg_align_group_term`
  - `_lambda_fusion_init_accum_ctx`
  - `_merge_grad_norm`
  - `_set_seed`
  - `_expected_trainable_prefixes`
- round 3（P1 giant hotspot 压缩）：
  - `_build_posttrain_arg_parser`
  - `_build_posttrain_model_from_ckpt`
  - `_lambda_fusion_finalize`
- round 4（P2 direct-pose/parser/rollout 热点压缩）：
  - `_resolve_direct_pose_build_cfg`
  - `_cfg_parse_direct_pose`
  - `_lambda_rollout_unroll_single_step`
- round 5（P3 保守净减 / finalize 聚合压缩）：
  - `_lambda_fusion_finalize`
- round 6（P3 保守净减 / leg-adjustment 主干压缩）：
  - `_lambda_rollout_apply_direct_leg_adjustments`

累计结果：

- 已净减 `14` 个 top-level helper
- `P1` 三个 giant hotspot 已完成压缩，且都保留了原 CLI / build / finalize 语义
- `P2` 三个热点已完成压缩：`direct_pose/build` infer/compat policy、direct-pose parser 后处理、rollout kernel 子概念
- `def` 数从 `84 -> 97`，但换来更低的 `LOC` / 最大函数长度 / `>=300 LOC` 函数数；P2 新增 helper 均对应稳定语义块，而非 `dict-pack` / `getattr` / `state_dict.get` 搬运层
- `checkpoint compat`、`soft-fail policy`、`config contract` 边界保持原语义

当前结构指标（worktree snapshot）：

- LOC：`5557`
- top-level `def` / 函数数：`97`
- 最大函数长度：`_lambda_fusion_finalize = 260`
- `>= 150 LOC` 的函数数：`7`
- `>= 300 LOC` 的函数数：`0`
- 小型 single-use helper 数（caller `<=1` 且函数长度 `<=20`，粗略统计）：`20`
- 主题结构债指标 #1：`_record_posttrain_soft_fail` 调用点数 = `42`
- 主题结构债指标 #2：`direct_pose_` token count = `1062`

当前最长函数（前 8）：

- `train/posttrain.py:2692` `_lambda_fusion_finalize` = `260`
- `train/posttrain.py:2272` `_lambda_rollout_unroll_single_step` = `213`
- `train/posttrain.py:1953` `_lambda_rollout_apply_direct_leg_adjustments` = `173`
- `train/posttrain.py:2954` `_lambda_fusion_loss_rollout` = `170`
- `train/posttrain.py:1548` `_lambda_rollout_prepare_context` = `167`
- `train/posttrain.py:3620` `_run_training_loop` = `166`
- `train/posttrain.py:1379` `_rollout_step_common` = `151`
- `train/posttrain.py:907` `_cfg_parse_direct_pose` = `124`

验证状态：

- `python3 -m py_compile train/posttrain.py` 已通过
- `_build_posttrain_arg_parser` 的 CLI contract 等价校验已通过（`131` 个 option 的 `dest/default/type/choices/help` 保持一致）
- P2 结束后 `python3 -m py_compile train/posttrain.py` 已再次通过
- P3 首轮保守净减后，`_lambda_fusion_finalize` 定向 smoke 已通过（synthetic `accum_ctx/finalize_ctx`）
- P3 第二轮保守净减后，`_lambda_rollout_apply_direct_leg_adjustments` 定向 smoke 已通过（early-return path + simple active path）

当前结论：

- `train/posttrain.py` **已经完成从“存在明显过细 helper + 三个 giant hotspot 压顶”到“`>=300 LOC` 函数清零、P2 热点完成压缩”的状态切换**。
- 当前主风险进一步收敛为：
  - rollout 轨道仍依赖大 `ctx` / `accum` / `weights` payload
  - `_lambda_fusion_finalize` 仍是最大函数，但已进一步收敛到 `260 LOC`
  - `direct_pose` leg-adjustment 主干已从 `235 -> 173 LOC`
  - 残余 small helper / stats payload 仍可做保守 P3，但边际收益低于 P1/P2

---

## 1) 当前现状（P2 完成后快照）

当前代码快照核对（`train/posttrain.py`）：

- **热点问题 1**：`train/posttrain.py:2692` 的 `_lambda_fusion_finalize` 为 `260` 行，是当前最大函数；stats payload 仍然偏重，但已从 `272` 行进一步下降。
- **热点问题 2**：`train/posttrain.py:2272` 的 `_lambda_rollout_unroll_single_step` 已从 `335 -> 213` 行，仍是 rollout kernel，但不再是 `>=300 LOC` 风险。
- **热点问题 3**：`train/posttrain.py:1953` 的 `_lambda_rollout_apply_direct_leg_adjustments` 已从 `235 -> 173` 行，主干层级已明显收紧。
- **热点问题 4**：`train/posttrain.py:4099` 的 `_resolve_direct_pose_build_cfg` 已从 `283 -> 114` 行，infer 与 compat/fatal policy 已拆成稳定语义块。
- **热点问题 5**：`train/posttrain.py:907` 的 `_cfg_parse_direct_pose` 已从 `263 -> 124` 行，parser 后处理尾巴已明显收紧。
- **热点问题 6**：`train/posttrain.py:2954` 的 `_lambda_fusion_loss_rollout` 为 `170` 行，仍是 rollout/finalize 之间的次级编排热点。
- **热点问题 7**：文件内仍有约 `20` 个“小型 single-use helper”，说明 residual helper 净减还没做完，但这已不是头号矛盾。

结构指标（当前 snapshot）：

- LOC：`5557`
- top-level `def` / 函数数：`97`
- 最大函数长度：`260`
- `>=100 LOC` 的函数数：`13`
- `>=150 LOC` 的函数数：`7`
- `>=200 LOC` 的函数数：`3`
- `>=300 LOC` 的函数数：`0`

为什么 `train/posttrain.py` 现在的优先级成立：

- 这份文件已经不是“helper 乱长”的早期状态，前两轮净减已把最薄的一层收掉。
- P1 的 CLI/build/finalize 三个大块已经压下来了，下一步如果继续只盯着它们做微调，边际收益会下降。
- 所以下一阶段不应默认继续拆分，而应先做 stabilization / smoke；若继续 P3，只做保守的净减或 stats payload 收口。

---

## 2) 具体改动流程

## Phase A — 薄 helper 净减（A1 + A2）

### Step A1 — 回收近邻单点 wrapper（已完成）

目标：收回 rollout/runtime 中“调用点很近 + 只做一步搬运/组包”的 helper。

已完成：

- `_finite`
- `_rollout_cond_raw_idx`
- `_resolve_rollout_step_tensor`
- `_build_rollout_unroll_ctx`
- `_lambda_rollout_unroll_steps`
- `_train_mode_display_name`
- `_unfreeze_lambda_fusion`

收益：

- 减少跳转层数
- `rollout` 主线少了两层“dict-pack + for-loop wrapper”
- `main()` 和 train-mode 切换不再为了单次打印/单次 enable 额外跳转

### Step A2 — 回收单点 init / infer / merge 小 helper（已完成）

目标：继续收掉不会形成真实语义边界的单点 helper。

已完成：

- `_predict_pretrain_contacts_from_frozen`
- `_resolve_rollout_time_index`
- `_append_leg_align_group_term`
- `_lambda_fusion_init_accum_ctx`
- `_merge_grad_norm`
- `_set_seed`
- `_expected_trainable_prefixes`

收益：

- 去掉了“名字大于内容”的中继层
- `rollout_step_common`、`_lambda_rollout_apply_direct_leg_adjustments`、`_lambda_fusion_loss_rollout`、`main()` 的局部控制流更直接
- 残余 helper 更接近“真实 contract / policy / domain 概念”

### Step A3 — 残余小 helper 二次筛查（待做）

目标：逐个复核剩余约 `20` 个 small single-use helper，只收以下类型：

- 单点调用
- 函数长度很短
- 不承担 warning / fatal / compat / shape contract
- 内联后主流程更清楚

优先候选：

- `train/posttrain.py:455` `_as_path`
- `train/posttrain.py:659` `_canon_phase_reset_source`
- `train/posttrain.py:1156` `_resolve_device`
- `train/posttrain.py:3198` `_iter_infinite`

注意：

- 这一组不是“必须全收”
- 像 `_merge_norm_spec`、`_resolve_train_mode`、`_module_grad_norm` 这类虽然也是 single-use，但仍有稳定语义，不应机械回收

---

## Phase B — 巨函数压缩（P1/P2 已完成）

### Step B1 — 压缩 `_build_posttrain_arg_parser`（已完成）

目标：把原 `train/posttrain.py:4831` 的 CLI 定义从 `495` 行压缩到可维护状态。

建议方向：

- 先按参数组分段：`path/data`、`rollout/runtime`、`lambda/direct`、`build/compat`
- 用局部表驱动生成重复样板参数
- 保留 help 文案与默认值，不改 CLI contract

已完成结果：

- `train/posttrain.py:5125` `_build_posttrain_arg_parser` 已压到 `12` 行
- 参数规格集中到 `train/posttrain.py:4819` `_POSTTRAIN_ARG_SPECS`
- CLI contract 等价校验已通过：`131` 个 option 的 `dest/default/type/choices/help` 保持一致
- 该步带来文件 `LOC` 净减，满足 `LOC_after <= LOC_before`

### Step B2 — 压缩 `_lambda_fusion_finalize`（已完成）

目标：把原 `train/posttrain.py:2753` 的 finalize 热点按稳定语义块分段。

建议拆分顺序：

1. 先抽 `accum_ctx` 读取与基础 totals 聚合
2. 再抽 direct-leg/group-norm 统计汇总
3. 最后抽 stats payload 组装

已完成结果：

- `train/posttrain.py:2692` `_lambda_fusion_finalize` 已从 `402 -> 260`（P1 首轮压缩到 `272`，P3 保守净减继续压到 `260`）
- 已拆出稳定语义块：
  - `train/posttrain.py:2753` `_finalize_direct_group_norm`
  - `train/posttrain.py:2822` `_finalize_leg_align_joint_stats`
  - `train/posttrain.py:2845` `_summarize_lambda_finalize_stats`
- 当前仍保留 boundary stats / aux payload 组装在主函数内，后续若继续压缩，应优先做 stats payload 收口而不是新增 dict-pack wrapper

### Step B3 — 压缩 `_build_posttrain_model_from_ckpt`（已完成）

目标：把原 `train/posttrain.py:5328` 的“全流程 builder”收成三个阶段壳：

1. ckpt / shape / build cfg infer
2. model instantiate / encoder attach
3. pre-load compat + post-load runtime guards

已完成结果：

- `train/posttrain.py:5410` `_build_posttrain_model_from_ckpt` 已从 `309 -> 18`
- 已收成三段壳：
  1. `train/posttrain.py:5260` `_resolve_posttrain_model_build_state`
  2. `train/posttrain.py:5380` `_instantiate_posttrain_model`
  3. `train/posttrain.py:5425` `_load_posttrain_checkpoint_into_model`
- `_drop_*` / `_infer_*` / `_resolve_direct_pose_build_cfg` 等 compat/fatal policy helper 保持原边界，不做薄 wrapper 串接

### Step B4 — 压缩 `_resolve_direct_pose_build_cfg`（已完成）

目标：把 `train/posttrain.py:4099` 内部的三类职责分开：

- weight-shape infer
- cfg override / canonicalization
- incompatibility / fatal policy

注意：

- 这里只有在“拆出后仍然是独立领域概念”时才允许新增 helper
- 不能拆出 `_infer_xxx_cfg`、`_normalize_xxx_cfg` 这类名字大于内容的薄层

已完成结果：

- `train/posttrain.py:4099` `_resolve_direct_pose_build_cfg` 已从 `283 -> 114`
- 已拆出稳定语义块：
  - `train/posttrain.py:3914` `_infer_direct_pose_head_shape`
  - `train/posttrain.py:3989` `_infer_direct_pose_ckpt_layout`
  - `train/posttrain.py:4084` `_resolve_direct_pose_ckpt_compat_policy`
- warning / fatal / compat policy 仍集中在清晰边界内，没有新增纯 `dict-pack` / `getattr` / `state_dict.get` 搬运 helper

---

## Phase C — rollout 轨道收边界（已处理首轮）

### Step C1 — 压缩 `_lambda_rollout_unroll_single_step`（已完成）

原判断：

- `train/posttrain.py:2272` 虽然仍是 rollout kernel，但已不再是 `>=300 LOC` 风险
- 在 `ctx/weights/accum/state_vars` schema 未进一步收稳前，不建议贸然把它切成多个“传参爆炸 helper”

建议动作：

- 先收紧输入 payload 的 schema
- 再决定是否只拆出真正稳定的子概念，例如：
  - lambda reliability apply
  - boundary stat emit
  - direct/inc/blend error aggregation

已完成结果：

- `train/posttrain.py:2272` `_lambda_rollout_unroll_single_step` 已从 `335 -> 213`
- 已拆出稳定语义块：
  - `train/posttrain.py:2190` `_lambda_rollout_decode_model_outputs`
  - `train/posttrain.py:2209` `_lambda_rollout_accumulate_plan_terms`
  - `train/posttrain.py:2246` `_lambda_rollout_accumulate_direct_objective`
  - `train/posttrain.py:2298` `_lambda_rollout_accumulate_gate_supervision`
- `>=300 LOC` 函数数已从 `1 -> 0`
- 仍保留大 `ctx/weights/accum/state_vars` contract，不在本轮引入 dataclass/TypedDict 迁移

### Step C2 — 压缩 `ctx` / `weights` / `accum` 大字典的隐式 contract

目标：

- 减少“必须跳进去看 keyset 才知道协议”的阅读成本
- 让 `prepare -> run_unroll -> finalize` 三段的 payload 约定更显式

优先方向：

- 先固定 key schema，再考虑 local TypedDict / dataclass
- 如果结构化 context 不能减少代码量，就不做

---

## Phase D — config/parser 压缩（已处理首轮）

### Step D1 — 压缩 `_cfg_parse_direct_pose`（已完成）

目标：保持 declarative schema 风格，但压缩其后处理尾巴。

优先处理：

- leg-align 子段
- group-norm 子段
- 统一 canonicalize / clamp / fallback 的后处理模板

不建议处理：

- 不要把 `_cfg_get_*` 系列回收
- 不要破坏 `_cfg_from_schema` 的 declarative 形态

已完成结果：

- `train/posttrain.py:907` `_cfg_parse_direct_pose` 已从 `263 -> 124`
- 保留 `_cfg_from_schema` declarative 主边界
- 主要压缩 parser 内的局部样板、optional override canonicalization、group-norm fallback normalize

### Step D2 — 审视 `_cfg_parse_lambda_rollout`

目标：清掉局部后处理样板，不破坏 parser 主边界。

优先处理：

- `rollout_include_boundary`
- `phase_reset_source`
- `posttrain_contacts_source`
- affine/clamp 这类末端 normalize 逻辑

---

## 3) 保留边界清单（不作为优先回收目标）

以下 helper 当前建议保留，不应为了追求 `def` 数继续打平：

- config contract：
  - `_cfg_pick`
  - `_cfg_get_bool`
  - `_cfg_get_int`
  - `_cfg_get_float`
  - `_cfg_from_schema`
  - `_cfg_reject_removed_targets`
  - `_cfg_reject_retired_shell_keys`
  - `_cfg_reject_retired_direct_pose_highorder`
- rollout/runtime boundary：
  - `_prepare_rollout_cond`
  - `_prepare_rollout_contacts_input`
  - `_update_rollout_recurrent_state`
  - `_apply_rollout_carry_state`
  - `_rollout_step_common`
  - `_lambda_rollout_prepare_context`
  - `_lambda_rollout_build_reg_params`
  - `_lambda_rollout_resolve_nonleg_focus`
  - `_record_posttrain_soft_fail`
- build & checkpoint compat：
  - `_resolve_direct_pose_build_cfg`
  - `_drop_direct_pose_ckpt_tensors`
  - `_adapt_direct_pose_phase_z_ckpt_inputs`
  - `_drop_retired_direct_pose_highorder_ckpt_tensors`
  - `_drop_incompatible_direct_pose_leg_ckpt_tensors`
  - `_infer_event_clock_build_cfg`
  - `_infer_lambda_fusion_build_cfg`

保留理由统一为：

- 带独立 warning / fatal / compat policy
- 带 shape infer / contract normalize
- 是稳定领域概念，而不是简单字面动作

---

## 4) 下一步优先级

### P1（已完成）

- `train/posttrain.py:5125` `_build_posttrain_arg_parser`
- `train/posttrain.py:2692` `_lambda_fusion_finalize`
- `train/posttrain.py:5410` `_build_posttrain_model_from_ckpt`

### P2（已完成）

- `train/posttrain.py:4099` `_resolve_direct_pose_build_cfg`
- `train/posttrain.py:907` `_cfg_parse_direct_pose`
- `train/posttrain.py:2272` `_lambda_rollout_unroll_single_step`

### P3（仅保守推进）

- 残余 small single-use helper 的零碎回收
- `_lambda_fusion_finalize` / stats payload 的进一步局部收口（首轮已做）
- `_lambda_rollout_apply_direct_leg_adjustments` 的稳定子概念复核（主干已做 early-return 压缩）
- 禁止 `build/compat` 子 helper 的进一步切碎
- 禁止任何会引入新的 `dict-pack` 中转层的拆分

---

## 5) 当前结论

当前对 `train/posttrain.py` 的结构判断：

- **不是已经明显过度拆分**
- **也不是当前完全拆分合理、无需再动**
- 当前最准确的结论是：
  - **少量过细 helper 已基本回收完成**
  - **P1 的 CLI/build/finalize 三个 giant hotspot 已完成压缩**
  - **P2 的 rollout kernel、`direct_pose/build` compat 壳、以及 parser 后处理尾巴已完成首轮压缩**
  - **P3 已完成两轮保守净减；当前最大函数为 `_lambda_fusion_finalize = 260`，`>=300 LOC` 函数已清零**

因此后续路线不应继续以“继续压 P1/P2 已完成热点”为主目标，而应切换为：

- 先做 stabilization / smoke，确认 P2 重构后的 parser / direct-pose build / rollout kernel 行为边界
- 若继续 P3，只做保守净减或 stats payload 收口，且必须避免薄 wrapper / `dict-pack` 中转层
