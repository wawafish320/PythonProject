# 2026-03-07 Trainbase / Posttrain 未有效进入训练目标的计算分支清单

> Historical note (`2026-04-15`): this inventory remains useful for branch archaeology, but several interfaces mentioned here have since been removed from mainline, including `contact_phase_state_*`, layout `fallback_to_bone_names`, and some posttrain compat aliases. Treat those mentions as historical only.

## 目标

为本轮完整复跑前，先把**这次未有效进入训练目标的计算分支**整理出来，便于后续在
`train/training_MPL.py` 与 `train/posttrain.py` 中做函数级清理。

这里的“未有效进入训练目标”有一个严格口径：

- 指本轮没有成为有效 supervision / 有效训练分支 / 主优化目标；
- **不等于**“完全没跑到相关代码”；
- 某些 runtime signal、state core、compat shell 仍可能被读取、建模或参与前向。

本文档只回答一个问题：

> 对于本轮指定训练命令 + `docs/posttrain_pipeline.md` 主链，哪些分支是 active，哪些分支是本轮未有效进入训练目标/仅保留兼容壳？

---

## 本轮运行口径

### A. trainbase 入口

使用命令：

```bash
python -m train.training_MPL \
  --config_json config/exp_phase_mpl.clean.json \
  --run_name exp_phase_DirectBranch_v1_d1 \
  --out ./models/MLPL2_DirectBranch_v1 \
  --depth 3 \
  --encoder_path ./models/motion_encoder_equiv_stageA.pt \
  --contact_plan_enable \
  --trainbase_contacts_source pretrain_contact \
  --trainbase_contacts_pretrain_clamp 1.0 \
  --trainbase_contacts_pretrain_affine_stats debug_output/_tmp_phaseb_affine_20260304/affine_fit_mix08/affine_stats.json \
  --contact_plan_init_mode learnable+obs --contact_plan_init_hidden 128 \
  --direct_pose_enable --w_direct_pose 0.2 \
  --contact_plan_time_pe_dim 16 \
  --direct_pose_meas_mode concat \
  --direct_pose_meas_drop_prob 0.1 --direct_pose_plan_drop_prob 0.1 --direct_pose_meas_noise_std 0.03 \
  --use_event_clock \
  --event_clock_max_delta 0.5 \
  --event_clock_hidden_dim 64 \
  --event_clock_gate_hidden_dim 32 \
  --event_clock_lambda_entropy_weight 0.01 \
  --event_clock_lambda_prior_weight 0.01 \
  --event_clock_delta_z_l2_weight 0.001
```

其中 trainbase contacts 主线现在应显式固定为：

- `trainbase_contacts_source=pretrain_contact`
- `trainbase_contacts_pretrain_clamp=1.0`
- `trainbase_contacts_pretrain_affine_stats=affine_mix08`

注意：`train/training_MPL.py` 现已把 `phase_reset_source` 从 trainbase CLI/config contract 中移除并强制固定为 `none`，
因此这里不能再传 `--phase_reset_source none`。

### B. posttrain 主链

使用 `docs/posttrain_pipeline.md` 中当前 main entry。

当前 accepted mainline 已更新为：

`Stage6 -> 70a -> 70b_concat -> 70c_replacecontacts (historical reference shell) -> promoted 70R (low-LR trunkfull s180) -> 71 -> 72 -> lambda final`

其中当前 downstream handoff 不再是 plain `new70R` recipe，而是：

- `models/__tmp_70R_new_lowlr_trunkfull_s180_20260308/ckpt_last_WalkF_stage7_70R_new_lowlr_trunkfull_s180_20260308.pth`

并按文档当前主线口径固定：

- `posttrain_contacts_source=pretrain_contact`
- `pretrain_contact + clamp1 + affine_mix08`
- `phase_reset_source=none`

当前接受链 / 进退清单参考：

- `debug_output/_tmp_70R_lowlr_trunkfull_s180_rounds5_20260308/s180_verdict.md`
- `debug_output/_tmp_chain_s180promote_20260308/chain_verdict.md`
- `docs/Problems/active/2026-03-08_posttrain_s180_promote_regression_progress_checklist.md`

### Update（2026-03-08 PM）: experimental lane status — retain `72micro_s70`, retire `hybridcarrytrain`

在补齐 `70R(s180) -> 71m -> 72micro_hybridcarry_s70 -> lambda final` 闭环后，当前 experimental 口径应进一步收敛为：

1. accepted mainline 仍保持：`70R(s180) -> 71 -> 72 -> lambda final`；
2. 如果还需要保留一条 experimental downstream lane，当前应保留的是：`70R(s180) -> 71m -> 72micro_s70 -> lambda final`；
3. `72micro_hybridcarrytrain` / `72micro_hybridcarry_s70` 不再建议作为长期并行分支：
   - 它的价值主要是 root-cause 确认：`accepted72 -> 72micro_hybridcarrytrain` 的 `arms_main` / `A_52_59 arms` / `B_76_80 arms` 明显比原始 `72micro` 更干净；
   - 但闭到 final `lambda` 后，相对 plain `72micro_s70` 只剩 marginal change，见：
     - `debug_output/_tmp_lambdamicro_vs_lambdahybridcarry_from_s180_s70_blend_20260308_Walk_F/gate_metrics.json`
     - `blend_mean: 0.485850 -> 0.485307`
     - `rollout_mean: 0.946175 -> 0.945303`
     - `foot_l_ball_l_blend_mean: 3.019336 -> 3.013556`
     - `calf_r_blend_mean: 1.472629 -> 1.470054`
   - 这些变化方向虽大多为正，但量级过小，不足以支撑新增一条 ship-target / active-handoff lane。
4. 因此当前文档与后续清理口径应记为：
   - **保留** `72micro_s70 -> lambda final` 作为 experimental candidate；
   - **归档并准备移除** `hybridcarrytrain` 线，仅保留 compare 包和必要结论文字；
   - 后续若继续追 `71m / 72_micro`，默认从 plain `72micro_s70` 开始，而不是从 hybridcarry 变体继续。

---

## 基准数据清单路径

后续做函数级清理时，baseline 锚点仍保持为：

- 文档：`docs/Problems/active/2026-03-07_trainbase_posttrain_baseline_data_checklist.md`
- accepted baseline compare 包：
  - `debug_output/_tmp_phaseD_direct_geolocal_compare_20260305/old_whitebox_vs_new_fullchain_pretrain/gate_metrics.json`
  - `debug_output/_tmp_phaseD_direct_geolocal_compare_20260305/old_whitebox_vs_new_fullchain_pretrain/global_signal_summary.txt`
  - `debug_output/_tmp_phaseD_direct_geolocal_compare_20260305/old_whitebox_vs_new_fullchain_pretrain/summary_metrics.txt`
- `2026-03-07 eval_on` compare 包：
  - `debug_output/_tmp_posttrain_from_exp_phase_DirectBranch_v1_d1_20260307_eval_on_compare_Walk_F/gate_metrics.json`
  - `debug_output/_tmp_posttrain_from_exp_phase_DirectBranch_v1_d1_20260307_eval_on_compare_Walk_F/global_signal_summary.txt`
  - `debug_output/_tmp_posttrain_from_exp_phase_DirectBranch_v1_d1_20260307_eval_on_compare_Walk_F/summary_metrics.txt`

当前 accepted successor line（`s180 promote -> 71 -> 72 -> lambda`）的对照包同步记录为：

- `debug_output/_tmp_70R_lowlr_trunkfull_s180_rounds5_20260308/s180_verdict.md`
- `debug_output/_tmp_chain_s180promote_20260308/chain_verdict.md`
- `debug_output/_tmp_chain_s180promote_20260308/compare_vs_accepted_r5_direct/global_signal_summary.txt`
- `debug_output/_tmp_chain_s180promote_20260308/compare_vs_evalon_20260307_direct/global_signal_summary.txt`
- `debug_output/_tmp_chain_s180promote_20260308/compare_vs_accepted_r5_blend/summary_metrics.txt`
- `debug_output/_tmp_chain_s180promote_20260308/compare_vs_evalon_20260307_blend/summary_metrics.txt`
- `docs/Problems/active/2026-03-08_posttrain_s180_promote_regression_progress_checklist.md`

---

## 先给结论

这次**不要动**的主链：

1. `train/training_MPL.py`
   - `contact_plan` 主链本身仍在用；
   - `contact_phase_state` 主体仍在用，但 reset/event 分支本轮关闭；
   - `direct_pose` 基础 `concat` 桥接仍在用；
   - `event_clock` 仍在用；
   - basetrain rollout contacts 主输入已收缩为 `trainbase_contacts_source=pretrain_contact`；
   - 当前主链口径固定为 `pretrain_contact`（可叠 `clamp1 + affine_mix08`）；`whitebox` 已从 runtime/fallback lane 退休，仅保留历史记录。
2. `train/posttrain.py`
   - newflow XOR 主链仍在用：`train_direct_pose` / `train_lambda_head`；
   - `pretrain_contact` 路由仍在用；
   - split direct head / arm split / phase_z concat / `replace_contacts` / leg-only / nonleg-only / lambda final 都仍在用；
   - 当前 accepted Stage7 downstream line 已更新为：`promoted 70R (s180 low-LR trunkfull) -> 71 -> 72 -> lambda final`；
   - `71m / 72_micro` 仍只保留为 experimental lanes，不升成当前 accepted mainline；
   - 在这些 experimental lanes 里，当前保留的 downstream candidate 是 `70R(s180) -> 71m -> 72micro_s70 -> lambda final`；
   - `72micro_hybridcarrytrain` / `72micro_hybridcarry_s70` 只保留为 root-cause diagnostic evidence：它确认了 cross-cycle `pose_hist` carry 是 arms 漂移主因，但 final-`lambda` 收益不足以覆盖新增复杂度，因此后续计划从 active experimentation 中移除。

这次优先清理的，是**没有命中**或**只剩兼容壳**的分支，不是上面这些 active 主链。

## 重要例外（最后处理）

下面 3 组**不能误解成“完全没跑到相关代码”**，当前只应标记为“未有效进入训练目标”或“仅兼容读取”，并且建议放到最后处理：

1. `contacts_meas`
   - `contacts_meas` 监督 loss 本轮没进；
   - 更新（2026-03-09）：whitebox `contacts_meas` runtime/fallback lane 已整段删除；
   - 当前剩余问题仅是 `w_contact_meas` 监督 loss 壳是否仍需单独退休；
   - 因此更新后的口径应记为：`supervision-off; whitebox runtime retired`。
2. `contact_phase_state`
   - reset / event 分支本轮没进；
   - 但 `contact_phase_state` 状态本体仍是 active。
   - 因此当前口径应记为：`reset/event-off but state-core-active`。
3. `lambda final` 的 direct/leg 兼容字段
   - 这批字段在 final config 中仍可能被读取、参与建模兼容或 checkpoint 兼容；
   - 但在 `train_lambda_head` 模式下，不会转成有效训练分支。
   - 因此当前口径应记为：`compat-read but not effective train branch in lambda-head mode`。

建议的处理时机：

- 先完成真正 retired / supervision-off 壳的清理；
- 再把 **basetrain rollout contacts source 固定到 `pretrain_contact` 主线**（如需对齐 posttrain，补 `clamp1 + affine_mix08`）；
- 完整复跑 `train.training_MPL` + `docs/posttrain_pipeline.md` 主链并确认 baseline 无回退；
- 上述前置条件满足后，再进入这 3 个重要例外的**整段移除**。

Update（2026-03-09）：

- `pretrain_contact + clamp1 + affine_mix08` 已成为当前 accepted mainline；
- `promoted 70R (s180 low-LR trunkfull) -> 71 -> 72 -> lambda final` 已完成主链复跑并被接受；
- 因此这里不再是“暂缓删除”的状态，而是可以进入执行阶段。
- 单独的整段移除清单见：`docs/Problems/active/2026-03-09_trainbase_posttrain_full_removal_checklist.md`

---

## 一、`train/training_MPL.py`：本轮 active / inactive

## 1.1 本轮 active 主链

本轮明确命中的功能：

- `contact_plan_enable=true`
- `contact_plan_inject=plan_z`
- `contact_plan_init_mode=learnable+obs`
- `contact_phase_state_enable=true`
- `direct_pose_enable=true`
- `direct_pose_meas_mode=concat`
- `use_event_clock=true`
- `w_contact_plan=1.0`
- `w_direct_pose=0.2`
- `freerun_stage_schedule` 仍在用，但只做 TF/LR schedule，不做 freerun loss 训练

因此，`contact_plan` / `phase_state` / `direct_pose` / `event_clock` 的**主路径不可按 dead code 处理**。

## 1.2 本轮未命中的 trainbase 分支

### A. freerun loss 训练分支：未命中

当前值：

- `freerun_horizon=0`
- `freerun_weight=0.0`

含义：

- freerun **评估**仍会写 `valfree_ep*.json`；
- 但 freerun **loss/反传**分支本轮关闭。

清理定位建议：

- `train/training_MPL.py` 中与 `_short_freerun_loss`、`_freerun_loss_window`、freerun gradient log/ratio 相关的训练分支，可先独立盘点。

### B. teacher/input noise 分支：未命中

当前值：

- `teacher_rot_noise_deg=0.0`
- `teacher_rot_noise_prob=0.0`
- `input_step_noise_prob=0.0`
- `input_noise_deg_mix=[]`

含义：

- 所有 teacher noise / step noise / mixed input noise 本轮都没有生效。

### C. `contacts_meas` supervision loss：未有效进入训练目标

当前值：

- `w_contact_meas=0.0`

注意：

- 这里只是 **loss term** 没开；
- 不是 `contacts_meas` 整条链没用；
- 更新（2026-03-09）：basetrain rollout source 已固定为 `pretrain_contact`；whitebox runtime/fallback 已删除。

在接入改造完成后，`contact_plan_enable=true` 时训练/rollout 的 `contacts_in_t` 将优先来自
`trainbase_contacts_source` 解析结果：

- `pretrain_contact`：走 frozen encoder + frozen contact head；

因此：

- **可清理候选**：`w_contact_meas` 对应的监督 loss 壳；
- **已完成**：`_contact_meas_whitebox` 本体与 `whitebox/auto` source 解析已退休；
- **后续重点**：聚焦 `w_contact_meas` 监督 loss 壳是否还需要独立删除。

### D. phase reset / event reset 分支：未有效进入训练目标

当前值：

- `phase_reset_source=none`
- `contact_phase_state_event_kind=none`

含义：

- `contact_phase_state` 本体仍在跑；
- 但本轮**没有**使用 `contacts_meas` crossing 去 reset phase；
- 也没有 touchdown/liftoff/both 事件重置；
- 当前应归类为 `reset/event-off but state-core-active`。

这是本轮最明确的“可拆 reset 壳 / 保留 state 本体”的分界线。

### E. direct-pose 只走 `concat`，其余 direct mode 分支未命中

当前值：

- `direct_pose_meas_mode=concat`

因此本轮未命中：

- `mode_select`

如果只围绕本轮 trainbase 主链清理，`mode_select` 是优先检查对象。

### F. direct-pose split / leg-specialized 训练分支：未命中

当前值（保持默认关闭）：

- `direct_pose_split_enable=false`
- `direct_pose_arm_split_enable=false`
- `direct_pose_loss_leg_split=false`
- `direct_pose_loss_group_norm_enable=false`
- `direct_pose_grad_monitor_enable=false`

因此本轮 trainbase 没有命中：

- leg/non-leg split
- arm/else split
- group norm reweight
- direct grad monitor

注意：这些分支在 **posttrain** 主链里会被使用；因此这里只能说“对 `training_MPL` 本轮没打到”，**不能据此删 `train/posttrain.py` 中对应实现**。

### G. contact TD hazard 分支：未命中

`config/exp_phase_mpl.clean.json` 已不再携带旧的 `contact_td_hazard_*` 训练口径；本轮不走这条线。

### H. lambda fusion postprocess：trainbase 本轮未命中

`train/training_MPL.py` 中仍保留 `lambda_fusion_apply` / reliability 后处理壳，但本轮 trainbase 命令没有启用这条线。

如果要做 trainbase 清理，这组分支值得单独 grep：

- `lambda_fusion_apply`
- `_lambda_fusion_apply_reliability`

---

## 二、`train/posttrain.py`：docs 主链下哪些分支没用

## 2.1 当前 docs 主链实际在用的分支

当前 `docs/posttrain_pipeline.md` 主链并不窄，下面这些都是 active 的：

- newflow XOR：`train_direct_pose` / `train_lambda_head`
- `posttrain_contacts_source=pretrain_contact`
- split direct head：`direct_pose_split_enable=true`
- 3-way arm split：`direct_pose_arm_split_enable=true`
- phase_z concat：historical `70b_concat`
- `phase_z -> replace_contacts`：historical `70c_replacecontacts` / generated `70b_replace` / promoted `70R` / `71` / `72` / `lambda`
- `direct_pose_nonleg_train_only=true`：70R
- `direct_pose_leg_train_only=true`：71/72（以及 final config 中存在但在 lambda mode 下不生效）
- `direct_pose_leg_mode=so3`
- `direct_pose_loss_group_norm_enable=true`
- `direct_pose_grad_monitor_enable=true`：Stage6、70R
- `lambda_fusion_use_rollout_step=true`：lambda final

所以 posttrain 里真正能删的不是这些 active 分支，而是下面这些未命中/只剩兼容壳的部分。

## 2.2 本轮 posttrain 全链未命中的分支

### A. 旧 target / legacy stage 壳：未命中

当前 runtime 已明确 newflow-only：

- 旧 Stage1-5 target 不再是 mainline
- `train_contact_td_hazard` 不是当前主链

这类壳已经属于“有 guard 但不在本轮 contract 中”的清理候选。

### B. 非 `pretrain_contact` contacts source：未命中

当前 `train.posttrain` 运行时直接要求：

- `posttrain_contacts_source=pretrain_contact`

因此本轮不会命中：

- `whitebox`
- `model`
- `gt`
- `zero`

这些来源现在更像 validate/reference lane，而不是 `train.posttrain` mainline。

### C. `phase_reset_source != none` 的所有 reset 分支：本轮未命中

本轮 8 个 stage config 都是：

- `phase_reset_source=none`

因此不会命中：

- `contacts_meas` reset
- `ttc_gt` reset
- 以及已退休但仍保留 guard 的 `ttc_pred` / `td_hazard`

这里建议把清理思路分两层：

1. 先删 retired alias / error shell；
2. 再评估 `contacts_meas` / `ttc_gt` 是否还有复现实验需要。

### D. hinge 全家桶：本轮未命中

当前 docs 主链和 active configs 都不再使用：

- `direct_pose_hinge_*`
- `direct_pose_hinge_train_only`
- `direct_pose_hinge_gate_train_only`
- hinge gate / hinge eps / hinge sup / hinge stance / hinge reg

这组是 posttrain 清理的高价值热点，因为：

- 配置字段多；
- checkpoint compat 壳多；
- 当前 mainline 完全不用。

### E. leg side routing / sign gate / rank1：本轮未命中

当前 8-stage config 统一为：

- `direct_pose_leg_side_routing=false`
- `direct_pose_leg_side_sign_gate=false`
- `direct_pose_leg_side_rank1=false`

因此以下整组本轮未命中：

- per-side routed shared leg head
- sign gate
- rank1 side factorization
- side extra cue / other-side phase / relative phase

### F. direct loss SICS focus：本轮未命中

当前值：

- `direct_pose_loss_sics=null`

因此以下没有生效：

- SIC hard mask
- SIC boost
- `direct_pose_loss_cycle_gte` 的阶段性启用逻辑

### G. lambda gate supervision：本轮未命中

当前值：

- `lambda_gate_sup_weight=0.0`

因此 lambda final 虽然训练 `lambda_fusion_head`，但没有额外 gate supervision 支线。

### H. direct reinit / shape-adaptation 兼容壳：本轮未命中

主链当前没有走：

- `direct_pose_reinit=true`
- 因输入维度变化触发的重建/迁移分支

这类逻辑主要是兼容“checkpoint shape 与新 head 配置不一致”的历史改模场景。

---

## 三、最值得后续清理的热点函数

## 3.1 `train/training_MPL.py`

建议按下面顺序做：

1. **先清 loss 壳，不清主干输入**
   - freerun loss
   - `w_contact_meas` loss term
   - teacher/input noise
2. **再清 reset/event 壳**
   - `phase_reset_source`
   - `contact_phase_state_event_kind`
3. **最后清 direct_pose 的训练期附加壳**
   - `mode_select`
   - split/group_norm/grad_monitor（仅限 trainbase 入口）

建议重点 grep：

- `_short_freerun_loss`
- `_freerun_loss_window`
- `_contact_meas_whitebox`
- `phase_reset_source`
- `contact_phase_state_event_kind`
- `direct_pose_meas_mode`
- `direct_pose_split_enable`
- `direct_pose_grad_monitor_enable`
- `lambda_fusion_apply`

## 3.2 `train/posttrain.py`

建议按下面顺序做：

1. **先清 runtime 已拒收/已退休壳**
   - legacy target
   - retired reset alias
   - non-mainline contacts source
2. **再清 hinge 家族**
   - 参数解析
   - config 读取
   - checkpoint compat
   - train-mode guard
3. **最后清未启用的高级 leg routing 支线**
   - side routing
   - sign gate
   - rank1
   - SIC focus

建议重点 grep：

- `_canon_phase_reset_source`
- `_resolve_train_mode`
- `_build_posttrain_model_from_ckpt`
- `_build_rollout_mode_kwargs`
- `_unfreeze_for_train_mode`
- `posttrain_contacts_source`
- `direct_pose_hinge`
- `direct_pose_leg_side_`
- `direct_pose_loss_sics`
- `lambda_gate_sup_`

---

## 四、这次清理时的边界提醒

### 可以当作“本轮未有效进入训练目标”看待的

- trainbase freerun loss 训练
- trainbase teacher/input noise
- trainbase `w_contact_meas` 监督 loss（仅限 loss 壳；whitebox runtime 已退休）
- trainbase phase reset/event reset（仅限 reset/event 壳，不含 `contact_phase_state` state core）
- posttrain hinge 家族
- posttrain non-`pretrain_contact` sources
- posttrain side routing / sign gate / rank1
- posttrain SIC focus
- posttrain lambda gate supervision

### 不能因为“看起来旧”或“本轮未有效训练”就直接删的

- `contact_plan` 主体
- `contact_phase_state` 主体
- whitebox `contacts_meas` 生成（已于 2026-03-09 退休；此条仅保留作历史记录）
- posttrain split/arm-split
- `phase_z concat` 与 `replace_contacts`
- 70R 的 `nonleg_train_only`
- 71/72 的 `leg_train_only`
- lambda final 的 `lambda_fusion_use_rollout_step`

---

## 五、推荐的实际清理顺序

1. 先在 `train/posttrain.py` 做 **hinge + retired reset/source 壳** 清理；
2. 再在 `train/training_MPL.py` 清 **freerun loss / noise / contact_meas loss**；
3. 最后再收 `phase_reset_source` / `contact_phase_state_event_kind` 等 phase reset 壳；
4. 对 `direct_pose` 高阶支线（side routing / SIC focus / sign gate / rank1）单开一轮，因为这组最容易牵连 checkpoint compat。
   - 本轮执行记录：`docs/Problems/active/2026-03-07_posttrain_direct_pose_highorder_cleanup_round.md`
5. 最后再处理 3 个重要例外：whitebox/fallback lane 下的 `contacts_meas` runtime 输入、`contact_phase_state` state core、`lambda final` compat-read 字段。

这个顺序的原因是：

- 第 1、2 步对当前主链最不敏感；
- 第 3 步会碰 phase state；
- 第 4 步虽然本轮没用，但在 `posttrain` 里 spread 很广，最好后收；
- 第 5 步涉及“代码仍在跑但未形成有效训练目标”的 3 个重要例外；截至 2026-03-09，主链改造与复跑前置条件已满足，可单开整段移除执行。
- 执行清单见：`docs/Problems/active/2026-03-09_trainbase_posttrain_full_removal_checklist.md`

---

## 六、2026-03-08 新增定位结论（已更新）：根因锁定在 `current70R -> new70R`，`s180 low-LR trunkfull` 已通过全链

> Update（2026-03-08 PM）: 本节 `6.2-6.7` 保留当日排查轨迹与实验支线记录；当前 active mainline 结论以新增 `6.8` 为准。旧的“当前卡在 `71` / `72_micro` 可作主线替代”已经不再是推荐流程。

### 6.1 本轮目标口径

本轮定位的目标已经明确为：

- **不再追求 strict old reproduce 作为主目标**；
- 主线口径保持 `best.pt + pretrain_contact + clamp1 + affine_mix08`；
- `stageA bundle` 只保留为隔离实验/历史对照，不再回切成 mainline。

对应参考：

- `docs/posttrain_pipeline.md`
- `models/motion_encoder_equiv.pt.best.pt`
- `debug_output/_tmp_phaseb_affine_20260304/affine_fit_mix08/affine_stats.json`

这意味着：

- historical `70b_concat` 的判断标准不再是“是否最像 old 70b concat”；
- 而是“在 current mainline 输入分布下，哪条 `70b` 路线对后续 `70R/71/72/lambda` 更稳定”。

### 6.2 `70b_replace` 最小隔离结论

#### 6.2.1 原因判断

本轮围绕 historical `70b_concat -> 70b_replace` 的最小隔离结论是：

- `phase_z` 本身不是主要漂移源；
- 真正变化更大的是 current basetrain / 70a 输出的 `plan/meas` 概率语义；
- 因此历史 `70b concat` recipe 在 current route 下更脆弱；
- 把 historical `70b_concat` 改成 `70b_replace`（`direct_pose_phase_z_mode=replace_contacts`）是对的，但不能直接拿旧 concat warm-start 生搬硬套。

为避免 equal-dim 情况把旧 `plan/meas` 列误当 phase 列，本轮新增了 warm-start 适配工具：

- `tools/adapt_phase_replace_from_concat_ckpt.py`

该工具用于：

- 从 concat ckpt 推断 `contact_dim`；
- 保留 first-layer base 列；
- phase tail 零初始化；
- 同时把 cfg 切到 `direct_pose_use_phase_z=true` + `direct_pose_phase_z_mode=replace_contacts`。

#### 6.2.2 训练与评估产物

- config：`debug_output/_tmp_70b_replace_train_20260307/posttrain_70b_replacecontacts_from70a_zerophase.json`
- ckpt：`models/__tmp_70b_replace_train_20260307/ckpt_last_WalkF_stage7_70b_replacecontacts_train_20260307_from70a_zerophase.pth`
- eval：`debug_output/_tmp_70b_replace_train_eval_20260307_Walk_F/Walk_F_freerun_cycles.json`

#### 6.2.3 对比结论

1. 对比 current rerun `70b`：
   - compare：`debug_output/_tmp_70b_rerun_vs_70b_replace_train_20260307_Walk_F/gate_metrics.json`
   - `global_mean_rel_delta_pct = -20.5896%`
   - `leg8_mean_delta = -0.2317`
   - `non_leg_mean_delta = -0.0184`
   - gate：`keep_lower_body=false, fix_non_leg=true, calf_main=true, calf_aux=true`
   - 解读：global / calf / non-leg 都明显改善；唯一未过的是 `keep_lower_body`，主要因为 `SIC12-15 foot_l/ball_l` 仍有轻微回退。

2. 对比 current rerun `70c`：
   - compare：`debug_output/_tmp_70c_rerun_vs_70b_replace_train_20260307_Walk_F/gate_metrics.json`
   - `global_mean_rel_delta_pct = -20.0356%`
   - `leg8_mean_delta = -0.1728`
   - `non_leg_mean_delta = -0.0288`
   - 4 个 gate 全过。

本轮结论：

- 在 current route 下，`70b_replace` 明显优于历史 `70b concat` rerun；
- 并且已经优于 current rerun `70c`；
- 因此 `70b concat -> 70c replace` 这条历史两段式，在 current mainline 下已经出现被 `70b_replace` 单段替代的明确信号。

### 6.3 `70R` 验证：新链仍成立

`70R` 从 `new70b_replace` 起链后的产物：

- config：`debug_output/_tmp_70R_from_new70b_replace_20260307/posttrain_70R_from_new70b_replace.json`
- ckpt：`models/__tmp_70R_from_new70b_replace_20260307/ckpt_last_WalkF_stage7_70R_from_new70b_replace_20260307.pth`
- eval：`debug_output/_tmp_70R_from_new70b_replace_eval_20260307_Walk_F/Walk_F_freerun_cycles.json`
- compare：`debug_output/_tmp_70R_current_vs_new70breplace_20260307_Walk_F/gate_metrics.json`

对比 current rerun `70R`：

- `global_mean_rel_delta_pct = -10.0206%`
- `leg8_mean_delta = -0.1709`
- `non_leg_mean_delta = +0.0127`
- 4 个 gate 全过。

本轮结论：

- `70b_replace -> 70R` 这条链是成立的；
- 回退问题并不在 `70R`；
- 因而后续主要矛盾不应再回退到 `70b_replace` 本身，而应继续往下游 `71` 看。

### 6.4 （historical, superseded）`71 -> 72 -> lambda` 新链结论：第一次翻负发生在 `71`

从 `new70R` 继续起链，本轮已完整跑通：

- `71` ckpt：`models/__tmp_71_from_new70R_20260308/ckpt_last_WalkF_stage7_71_from_new70R_20260308.pth`
- `72` ckpt：`models/__tmp_72_from_new71_20260308/ckpt_last_WalkF_stage7_72_from_new71_20260308.pth`
- `lambda` ckpt：`models/__tmp_lambda_from_new72_20260308/ckpt_last_WalkF_stage7_lambda_from_new72_20260308.pth`

对应评估/对比：

- `71` eval：`debug_output/_tmp_71_from_new70R_eval_20260308_Walk_F/Walk_F_freerun_cycles.json`
- `72` eval：`debug_output/_tmp_72_from_new71_eval_20260308_Walk_F/Walk_F_freerun_cycles.json`
- `lambda` eval：`debug_output/_tmp_lambda_from_new72_eval_20260308_Walk_F/Walk_F_freerun_cycles.json`
- `71` compare：`debug_output/_tmp_71_current_vs_new_chain_20260308_Walk_F/gate_metrics.json`
- `72` compare：`debug_output/_tmp_72_current_vs_new_chain_20260308_Walk_F/gate_metrics.json`
- `lambda` compare：`debug_output/_tmp_lambda_current_vs_new70breplace_chain_20260308_Walk_F/gate_metrics.json`

其中 `71` 是第一个翻负的 stage：

- `global_mean_rel_delta_pct = +8.7463%`
- `leg8_mean_delta = +0.0093`
- `non_leg_mean_delta = +0.0127`
- gate：`keep_lower_body=false, fix_non_leg=false, calf_main=true, calf_aux=false`

`72` 和 `lambda` 基本延续同样模式：

- `72`：`global_mean_rel_delta_pct = +8.1841%`，`keep_lower_body=false`，`fix_non_leg=false`
- `lambda`：与 `72` 基本一致，未出现额外新的 primary break

因此当前定位应更新为：

- `70b_replace` 不是新问题源；
- `70R` 也不是新问题源；
- **当前 first-negative stage 已明确前移并锁定到 `71`。**

### 6.5 （historical, superseded）`71` 的最小隔离：先降 `lr/steps`，但仍未追上 current `71`

#### 6.5.1 已完成的 3 个最小实验

1. `lr=3e-4, e1, s60`
   - config：`debug_output/_tmp_71_lowlrsteps_20260308/posttrain_71_from_new70R_lr3e4_e1_s60.json`
   - ckpt：`models/__tmp_71_lowlrsteps_20260308/ckpt_last_WalkF_stage7_71_from_new70R_lr3e4_e1_s60_20260308.pth`
   - eval：`debug_output/_tmp_71_lowlrsteps_eval_20260308_Walk_F/Walk_F_freerun_cycles.json`
   - 对比 current `71`：`debug_output/_tmp_71_current_vs_lowlrsteps_20260308_Walk_F/gate_metrics.json`
   - 对比 `70R`：`debug_output/_tmp_70R_vs_71_lowlrsteps_20260308_Walk_F/gate_metrics.json`

2. `lr=3e-4, e1, s30`
   - config：`debug_output/_tmp_71_lowlrsteps_20260308/posttrain_71_from_new70R_lr3e4_e1_s30.json`
   - ckpt：`models/__tmp_71_lowlrsteps_s30_20260308/ckpt_last_WalkF_stage7_71_from_new70R_lr3e4_e1_s30_20260308.pth`
   - eval：`debug_output/_tmp_71_lowlrsteps_s30_eval_20260308_Walk_F/Walk_F_freerun_cycles.json`
   - 对比 current `71`：`debug_output/_tmp_71_current_vs_lowlrsteps_s30_20260308_Walk_F/gate_metrics.json`
   - 对比 `70R`：`debug_output/_tmp_70R_vs_71_lowlrsteps_s30_20260308_Walk_F/gate_metrics.json`

3. `lr=1e-4, e1, s60`
   - config：`debug_output/_tmp_71_lowlrsteps_20260308/posttrain_71_from_new70R_lr1e4_e1_s60.json`
   - ckpt：`models/__tmp_71_lr1e4_s60_20260308/ckpt_last_WalkF_stage7_71_from_new70R_lr1e4_e1_s60_20260308.pth`
   - eval：`debug_output/_tmp_71_lr1e4_s60_eval_20260308_Walk_F/Walk_F_freerun_cycles.json`
   - 对比 current `71`：`debug_output/_tmp_71_current_vs_lr1e4s60_20260308_Walk_F/gate_metrics.json`
   - 对比 `70R`：`debug_output/_tmp_70R_vs_71_lr1e4s60_20260308_Walk_F/gate_metrics.json`

#### 6.5.2 当前结论

1. 三个低 lr/低步数变体，相对 `70R` 都仍是正收益：
   - `debug_output/_tmp_70R_vs_71_lowlrsteps_20260308_Walk_F/gate_metrics.json`
   - `debug_output/_tmp_70R_vs_71_lowlrsteps_s30_20260308_Walk_F/gate_metrics.json`
   - `debug_output/_tmp_70R_vs_71_lr1e4s60_20260308_Walk_F/gate_metrics.json`
   - 这 3 组 compare 都是 4 gate 全过。

2. 但三者都没有追上 current rerun `71`：
   - `lr=3e-4, e1, s60`：`global_mean_rel_delta_pct = +7.3753%`
   - `lr=3e-4, e1, s30`：`global_mean_rel_delta_pct = +10.9704%`
   - `lr=1e-4, e1, s60`：`global_mean_rel_delta_pct = +9.5138%`

3. 在当前已测候选中，最接近 current `71` 的仍是：
   - `lr=3e-4, e1, s60`
   - 即 `debug_output/_tmp_71_current_vs_lowlrsteps_20260308_Walk_F/gate_metrics.json`

4. `lr=1e-4, e1, s60` 并没有继续带来提升：
   - 相对 `lr=3e-4, e1, s60` 的 compare：`debug_output/_tmp_71_lr3e4s60_vs_lr1e4s60_20260308_Walk_F/gate_metrics.json`
   - 结果表现为：calf 有局部改善，但 global / lower-body 并没有更好。

#### 6.5.3 loss 曲线解释

loss 曲线与摘要见：

- 原摘要：`debug_output/_tmp_71_lowlrsteps_20260308/stage71_loss_curve_summary.md`
- 含 `lr=1e-4` 摘要：`debug_output/_tmp_71_lowlrsteps_20260308/stage71_loss_curve_summary_with_lr1e4.md`
- **公平口径（只看 epoch1 前 60 步）摘要**：`debug_output/_tmp_71_lowlrsteps_20260308/stage71_loss_curve_summary_epoch1_with_lr1e4.md`
- plot：`debug_output/_tmp_71_lowlrsteps_20260308/stage71_total_loss_curves_epoch1_compare_with_lr1e4.png`

当前解释应更新为：

- 原版 `new71 (lr=1e-3)` 明显有 over-update，`~s20-30` 后就出现 plateau / 回升；
- `lr=3e-4` 能压住一部分回升，但没有真正追平 current `71`；
- `lr=1e-4` 的 raw loss 到 `s60` 仍在下降，但平滑窗口最低点仍落在 `~s28`，后半段窗口均值重新抬升；
- 因此问题**不像**“只要继续降 lr 就能修好”，也**不像**“单纯把 steps 从 60 砍到 30 就对了”；
- 更像是 `71` 在 current new-chain 下存在额外的语义/优化错配，`lr/steps` 只是在缓和症状，还没有打到根因。

### 6.6 （historical, superseded）当前状态与建议动作

截至 `2026-03-08`，当前状态应记录为：

1. 主线目标仍是保留 current mainline：
   - `best.pt + pretrain_contact + clamp1 + affine_mix08`

2. `70b_replace` 已经是一个**有效且更优**的最小替代候选：
   - `70b_replace -> 70R` 成立；
   - 不建议再把主要精力放回“继续修历史 `70b concat` recipe”。

3. 当前真正卡点是 `71`：
   - 现有纯 `lr/steps` sweep 已经说明：问题不只是 optimizer aggressiveness；
   - `72/lambda` 的退化主要是承接 `71`，不是新的首发问题。

4. 下一步推荐：
   - 继续保持“最小隔离 debug”，**不要重跑整条 full chain**；
   - 直接做 `current 71` vs `new-chain best 71 candidate` 的同 batch white-box diff；
   - 优先对齐 `direct head IO / plan-meas-phase / per-bone DirectGeoLocalDeg`；
   - 在 `71` 根因进一步明确前，不要贸然把 `docs/posttrain_pipeline.md` 主链直接改写成 `70b_replace` 正式主线。

补充：repo 已新增一个 **experimental `71m` curriculum branch** 用于最小 A/B：

- config：`config/posttrain_WalkF_stage7_71m_legcurriculum_proj10_lr3e4_e1_s60_20260308_fromarmchain.json`
- 机制：保留 `leg_train_only=true`、`replace_contacts`、split/head 结构不动，把旧 `71` 的 plain leg loss 与旧 `72` 的 `proj` align loss 合到单 stage 内；
- schedule：前 `15` steps `align_weight=0`，随后 `25` steps 线性 ramp 到 `10`；
- 目的：验证当前卡点更像“`71 -> 72` 的硬切换问题”还是“`71` 本体语义错配”，暂不作为 mainline 替代。

补充：基于上面的 `71m`，最早先完成过一轮 **`72_micro` step sweep**（固定 `72` objective，改 `lr=1e-4, e1, s10/s20/s30/s40/s50`）。**在这第一轮只看到 `s50` 为止时**，阶段性 best-overall 候选是 `s50`；这一结论已在后文补跑 `s60/s70` 后更新，当前最终 best-overall 应以 [6.7](#67-72_micro-再补-s60s70以及-s30-s70---lambda-final-影响) 的 **`s70`** 为准。

- 历史 `s50` 产物：`models/__tmp_72micro_from_71m_20260308/ckpt_last_WalkF_stage7_72micro_from_71m_lr1e4_e1_s50_20260308.pth`
- 历史 `s50` eval：`debug_output/_tmp_72micro_from_71m_s50_eval_20260308_Walk_F/Walk_F_freerun_cycles.json`
- 历史 compare：
  - vs `71m`：`debug_output/_tmp_71m_vs_72micro_s50_20260308_Walk_F/gate_metrics.json`（`global_mean_rel_delta_pct = -4.1433%`，`leg8_mean_delta = -0.034536`）
  - vs current `72`：`debug_output/_tmp_72_vs_72micro_s50_20260308_Walk_F/gate_metrics.json`（`global_mean_rel_delta_pct = -0.0138%`，`leg8_mean_delta = -0.000110`，`foot_l/ball_l = 0.4416`，`calf_r_global = 0.3343`）
- 历史结论只需保留一句：`s50` 已经证明 `72` 的 **step budget** 是有效调节轴，也说明主要卡点已经从 `70b/70R` 前移到 `71 -> 72` 的衔接；但它只是首轮 sweep 的强中间候选，不是当前最终 best-overall 结论。

### 6.7 （historical exploratory branch）`72_micro` 再补 `s60/s70`，以及 `s30-s70 -> lambda final` 影响

基于上面的 `71m -> 72_micro` 实验分支，本轮继续补跑了更长步数的 `72` 微阶段，并把 `s30/s40/s50/s60/s70` 全部接入 `lambda final` 做终点检查。

#### 6.7.1 新增 `72_micro s60/s70` 产物

- `s60` ckpt：`models/__tmp_72micro_from_71m_20260308/ckpt_last_WalkF_stage7_72micro_from_71m_lr1e4_e1_s60_20260308.pth`
- `s70` ckpt：`models/__tmp_72micro_from_71m_20260308/ckpt_last_WalkF_stage7_72micro_from_71m_lr1e4_e1_s70_20260308.pth`
- `s60` eval：`debug_output/_tmp_72micro_from_71m_s60_eval_20260308_Walk_F/Walk_F_freerun_cycles.json`
- `s70` eval：`debug_output/_tmp_72micro_from_71m_s70_eval_20260308_Walk_F/Walk_F_freerun_cycles.json`
- 对比 `71m`：
  - `debug_output/_tmp_71m_vs_72micro_s60_20260308_Walk_F/gate_metrics.json`
  - `debug_output/_tmp_71m_vs_72micro_s70_20260308_Walk_F/gate_metrics.json`
- 对比 full `72`：
  - `debug_output/_tmp_72_vs_72micro_s60_20260308_Walk_F/gate_metrics.json`
  - `debug_output/_tmp_72_vs_72micro_s70_20260308_Walk_F/gate_metrics.json`

关键指标如下：

1. 相对 `71m`，增加 step 仍持续带来收益：
   - `s50`：`global_mean_rel_delta_pct = -4.1433%`，`leg8_mean_delta = -0.034536`
   - `s60`：`global_mean_rel_delta_pct = -4.2985%`，`leg8_mean_delta = -0.035829`
   - `s70`：`global_mean_rel_delta_pct = -4.5916%`，`leg8_mean_delta = -0.038272`

2. 相对 current `72`，`s50` 之后没有翻负，反而开始略微超过 full `72`：
   - `s50`：`global_mean_rel_delta_pct = -0.0138%`，`leg8_mean_delta = -0.000110`
   - `s60`：`global_mean_rel_delta_pct = -0.1756%`，`leg8_mean_delta = -0.001403`
   - `s70`：`global_mean_rel_delta_pct = -0.4813%`，`leg8_mean_delta = -0.003846`

3. hotspot 走势也继续朝好的方向收敛：
   - `SIC12-15 foot_l/ball_l`
     - `s50`: `0.4416`
     - `s60`: `0.4370`
     - `s70`: `0.4356`
   - `calf_r_global`
     - `s50`: `0.3343`
     - `s60`: `0.3305`
     - `s70`: `0.3270`

4. 当前解读应更新为：
   - 在 `71m -> 72_micro` 这条 experimental lane 上，`72` 的 step budget **还没有在 `s50` 附近见顶**；
   - `s60/s70` 都比 `s50` 更好，而且 improvement 方向一致；
   - 但即使到 `s70`，`calf_r_global=0.3270` 仍没有回到 full `72` 的 `0.2664`，说明 `72_micro` 的额外收益主要来自 overall / leg8 / foot hotspot，而不是把 `calf_r` 主热点完全修回去。

因此如果只看 `72_micro` 本体，当前 best-overall 候选应从**首轮 sweep 的 `s50`** 更新为 **`s70`**，`s60` 次之。

#### 6.7.2 `s30/s40/s50/s60/s70 -> lambda final` 产物

`lambda final` 全部统一接在对应的 `72_micro` ckpt 之后：

- ckpt 目录：`models/__tmp_lambda_from_72micro_20260308`
- `s30`：`models/__tmp_lambda_from_72micro_20260308/ckpt_last_WalkF_stage7_lambda_from_72micro_s30_20260308.pth`
- `s40`：`models/__tmp_lambda_from_72micro_20260308/ckpt_last_WalkF_stage7_lambda_from_72micro_s40_20260308.pth`
- `s50`：`models/__tmp_lambda_from_72micro_20260308/ckpt_last_WalkF_stage7_lambda_from_72micro_s50_20260308.pth`
- `s60`：`models/__tmp_lambda_from_72micro_20260308/ckpt_last_WalkF_stage7_lambda_from_72micro_s60_20260308.pth`
- `s70`：`models/__tmp_lambda_from_72micro_20260308/ckpt_last_WalkF_stage7_lambda_from_72micro_s70_20260308.pth`

对应 freerun eval：

- `debug_output/_tmp_lambda_from_72micro_s30_eval_20260308_Walk_F/Walk_F_freerun_cycles.json`
- `debug_output/_tmp_lambda_from_72micro_s40_eval_20260308_Walk_F/Walk_F_freerun_cycles.json`
- `debug_output/_tmp_lambda_from_72micro_s50_eval_20260308_Walk_F/Walk_F_freerun_cycles.json`
- `debug_output/_tmp_lambda_from_72micro_s60_eval_20260308_Walk_F/Walk_F_freerun_cycles.json`
- `debug_output/_tmp_lambda_from_72micro_s70_eval_20260308_Walk_F/Walk_F_freerun_cycles.json`

对比产物分两组：

1. 对比各自输入 `72_micro`：
   - `debug_output/_tmp_72micro_s30_vs_lambda_s30_20260308_Walk_F/gate_metrics.json`
   - `debug_output/_tmp_72micro_s40_vs_lambda_s40_20260308_Walk_F/gate_metrics.json`
   - `debug_output/_tmp_72micro_s50_vs_lambda_s50_20260308_Walk_F/gate_metrics.json`
   - `debug_output/_tmp_72micro_s60_vs_lambda_s60_20260308_Walk_F/gate_metrics.json`
   - `debug_output/_tmp_72micro_s70_vs_lambda_s70_20260308_Walk_F/gate_metrics.json`

2. 对比当前 full-chain `lambda_from_new72`：
   - `debug_output/_tmp_lambdafull_vs_lambda72micro_s30_20260308_Walk_F/gate_metrics.json`
   - `debug_output/_tmp_lambdafull_vs_lambda72micro_s40_20260308_Walk_F/gate_metrics.json`
   - `debug_output/_tmp_lambdafull_vs_lambda72micro_s50_20260308_Walk_F/gate_metrics.json`
   - `debug_output/_tmp_lambdafull_vs_lambda72micro_s60_20260308_Walk_F/gate_metrics.json`
   - `debug_output/_tmp_lambdafull_vs_lambda72micro_s70_20260308_Walk_F/gate_metrics.json`

#### 6.7.3 `lambda final` 的实际影响（修正口径后）

这里需要明确修正：前面基于 `debug_output/_tmp_72micro_s*_vs_lambda_s*_20260308_Walk_F/gate_metrics.json` 得出的“`lambda final` 是 noop”结论，**口径不对**。

根因是：

1. `tools/build_stage7_old_new_summary.py` 只读取 `per_step_direct_geolocal_deg.DirectGeoLocalDeg`；
2. 而 `per_step_direct_geolocal_deg` 在 `run_freerun_cycles` 中导出的就是 **direct 分支** 的逐步误差，不是 lambda blend 后实际用于 rollout 的输出；
3. `lambda final` 训练时只解冻 `lambda_fusion_head`，不会改 `direct_pose_*` 专家本体；
4. 因此用 `DirectGeoLocalDeg` 去比 `72_micro -> lambda final`，天然会看到“完全不变”。

也就是说，之前观测到的“0 变化”并不代表 `final` 没生效，而是代表：

- `direct branch` 没变（这是**设计如此**）；
- 不是 `blend / rollout output` 没变。

本轮重新对白盒 JSON 做了逐步核对后，可以确认：

1. `72_micro` ckpt 本身**没有** `lambda_fusion_head.*` 权重；
   - 因而在 `72_micro` eval 上，即使传了 `--lambda_fusion_apply`，也不会真的启用 learned lambda head；
   - 对应现象是：`LambdaMean/LambdaEffMean/LambdaRelMean = None`，且 `BlendGeoLocalDeg == GeoLocalDeg`。

2. `lambda final` 之后则相反：
   - ckpt 中已有 `lambda_fusion_head.*`；
   - freerun 中 `LambdaMean≈0.973`，`LambdaEffMean` 在前 `10` 步按 warmup 从 `0 -> 0.97`，之后稳定在接近 `0.97`；
   - 说明 `final` 的 lambda 头**确实被训练并在推理时被应用了**。

3. 在真正反映 rollout 输出的 blended 指标上，`lambda final` 不是 noop，而是**大幅生效**：
   - 以与 `build_stage7_old_new_summary` 相同的 mask（`cycle>=1` 且去掉 wrap boundary）统计：
   - `s30`：`BlendGeoLocalDeg 62.9227 -> 0.5007`
   - `s50`：`BlendGeoLocalDeg 62.9227 -> 0.5000`
   - `s70`：`BlendGeoLocalDeg 62.9227 -> 0.5000`
   - 同时 `GeoLocalDeg` 也从 `62.9227` 降到约 `0.958-0.959`
   - 但 `DirectGeoLocalDeg` 保持不变：
     - `s30`: `0.1438 -> 0.1438`
     - `s50`: `0.1420 -> 0.1420`
     - `s70`: `0.1414 -> 0.1414`

4. 因此当前正确解读应是：
   - `lambda final` **没有改变 direct expert 本身**；
   - 但它**显著改变了最终 blend / rollout 的实际使用输出**；
   - 前面的“noop”只是由于比较脚本只看了 direct branch 指标。

补充：`lambda final` 后的 blended rollout 已经与 current full-chain `lambda_from_new72` 非常接近：

- `s30/s40/s50/s60/s70` 的 masked `BlendGeoLocalDeg` 都在 `~0.500` 左右；
- 相对 current full `lambda_from_new72` 的差值约为 `-0.0027 ~ -0.0037`；
- 说明这条 `72micro -> final` 链路是**真实接上了 final 效果**的，不是空跑。

#### 6.7.4 当前更新后的结论

截至 `2026-03-08`，这部分结论应更新为：

1. `72_micro` 的增步数收益是真实存在的，而且 `s50 -> s60 -> s70` 仍在单调改善；
2. 当前 `72_micro` 最优点应从 `s50` 更新为 **`s70`**；
3. `lambda final` **不是 noop**，它对 blended rollout 有显著收益；
4. 但 `lambda final` 的收益不会体现在 `DirectGeoLocalDeg` 这类 direct-only 指标上，因此不能再用 `build_stage7_old_new_summary.py` 的当前口径来判断 `lambda` stage 是否生效；
5. 因而如果后续继续评估这条 branch：
   - `72` 阶段仍看 `DirectGeoLocalDeg` / hotspot，用来选 direct 专家；
   - `lambda final` 必须改看 `BlendGeoLocalDeg` / `GeoLocalDeg` / 实际 rollout 指标；
   - 最好补一个 **blend-aware** 的 compare 工具，避免再次把 `lambda` stage 误判成 noop。

换句话说，当前新增证据把问题重新拆清为两层：

- `72` 的 step budget 仍然是有效轴（影响 direct expert）；
- `final` 也是真正有效的，但它作用在 **lambda blend 后的 rollout**，不是 direct branch 本身。

### 6.7.5 按用户指定基线重算：`s70` 不是“全面更好”，而是“尾部更好、均值混合”

这里再补一个重要修正：如果基线不是 `current full lambda_from_new72`，而是用户指定的这份实际基线：

- `debug_output/_tmp_posttrain_from_exp_phase_DirectBranch_v1_d1_20260307_eval_on_Walk_F_series/Walk_F_freerun_cycles.json`

那么结论会发生明显变化。

本轮已按这个基线重新生成一套 **blend-aware** 对比：

- 总汇总：`debug_output/_tmp_blendaware_expbaseline_summary_20260308/summary.md`
- 结构化结果：`debug_output/_tmp_blendaware_expbaseline_summary_20260308/summary.json`
- 单点 compare：
  - `debug_output/_tmp_blendaware_expbaseline_vs_s30_20260308_Walk_F/gate_metrics.json`
  - `debug_output/_tmp_blendaware_expbaseline_vs_s40_20260308_Walk_F/gate_metrics.json`
  - `debug_output/_tmp_blendaware_expbaseline_vs_s50_20260308_Walk_F/gate_metrics.json`
  - `debug_output/_tmp_blendaware_expbaseline_vs_s60_20260308_Walk_F/gate_metrics.json`
  - `debug_output/_tmp_blendaware_expbaseline_vs_s70_20260308_Walk_F/gate_metrics.json`

#### 6.7.5.1 基线本身

这个用户指定基线在相同 mask（`cycle>=1 && drop_wrap`）下为：

- `blend(mean/p95/p99/max) = 0.497677 / 0.703717 / 0.795526 / 0.824611`
- `rollout(mean/p95/p99/max) = 0.961235 / 1.347935 / 1.547076 / 1.574776`
- `direct(mean/p95/p99/max) = 0.131316 / 0.179830 / 0.203642 / 0.221648`
- `foot_l_ball_l_blend(mean/p95/p99/max) = 2.811336 / 4.045138 / 4.068645 / 4.074522`
- `calf_r_blend(mean/p95/p99/max) = 1.469515 / 4.492343 / 5.219478 / 5.377793`

#### 6.7.5.2 `s70` 为什么看起来“应该更好”，但实际又不是

因为它对不同统计量的影响方向不一样：

1. 对 **blend mean**，`s70` 并没有超过这个基线：
   - `0.497677 -> 0.499989`，`Δ=+0.002312`
   - 也就是说，从最终 blend 的**均值**看，`s70` 反而略差。

2. 但对 **blend tail**，`s70` 是明显更好的：
   - `p95: 0.703717 -> 0.701387`
   - `p99: 0.795526 -> 0.775305`
   - `max: 0.824611 -> 0.784823`

3. 对 **rollout 指标**，`s70` 则是均值和尾部都更好：
   - `mean: 0.961235 -> 0.958694`
   - `p95: 1.347935 -> 1.336697`
   - `p99: 1.547076 -> 1.520229`
   - `max: 1.574776 -> 1.544497`

4. 对 **direct 指标**，`s70` 反而更差，只在 `max` 上略好：
   - `mean: 0.131316 -> 0.141379`
   - `p95: 0.179830 -> 0.197752`
   - `p99: 0.203642 -> 0.212048`
   - `max: 0.221648 -> 0.220986`

因此，`s70` 的真实画像不是“全面优于基线”，而是：

- **blend/rollout 的 tail 更强**；
- **rollout 均值也更强**；
- 但 **blend mean** 不一定更强；
- **direct mean / direct p95 / direct p99** 明显不如这个基线。

#### 6.7.5.3 hotspot 为什么也会给出相反信号

同样地，hotspot 也不是单向一致的：

1. `foot_l/ball_l blend`：`s70` 比用户基线更差
   - `mean: 2.811336 -> 2.951937`
   - `p95: 4.045138 -> 4.189278`
   - `p99: 4.068645 -> 4.189426`
   - `max: 4.074522 -> 4.189463`

2. `calf_r blend`：`s70` 比用户基线更好
   - `mean: 1.469515 -> 1.447080`
   - `p95: 4.492343 -> 3.974870`
   - `p99: 5.219478 -> 4.751584`
   - `max: 5.377793 -> 4.904610`

所以“新流程 tail 更好”的观察是对的，但它主要体现在：

- overall `blend/rollout` 的 tail；
- `calf_r` 这类热点；

而不是体现在 `foot_l/ball_l` 或所有 mean 指标上。

#### 6.7.5.4 当前应如何统一解读

按这个用户指定基线，当前更准确的结论应写成：

1. `s70` **不是**“全面优于基线”；
2. `s70` 的主要优势在于：
   - `blend p95/p99/max`
   - `rollout mean/p95/p99/max`
   - `direct max`
   - `calf_r blend mean/p95`
3. `s70` 的主要劣势在于：
   - `blend mean`
   - `direct mean/p95/p99`
   - `foot_l/ball_l blend mean/p95/p99/max`
4. 因而这条 branch 当前更像是一个 **tail-risk / worst-case 改善方案**，而不是 strict mean-optimal 替代。

#### 6.7.5.5 现在的排序应该怎么用

基于 `debug_output/_tmp_blendaware_expbaseline_summary_20260308/summary.md`：

- `best_blend_mean`: `s40`
- `best_blend_p95/p99/max`: `s70`
- `best_rollout_mean`: `s40`
- `best_rollout_p95/p99/max`: `s70`
- `best_direct_mean/p95/p99/max`: `s70`
- `best_foot_hotspot_*`: `s70`（这里是候选内部最优；但相对用户基线仍未追平）
- `best_calf_hotspot_mean/p95`: `s70`
- `best_calf_hotspot_p99/max`: `s50`

因此，如果后续目标是：

- **追 overall mean**：优先看 `s40`
- **追 tail / worst-case / rollout 稳定性**：优先看 `s70`
- **追 calf 极端尾部**：`s50` 也值得保留为备选

补充：repo 现已新增 `tools/build_stage7_lambda_blend_summary.py`，后续涉及 `lambda final` 的比较，统一应使用这版脚本，而不再使用 direct-only 的 `tools/build_stage7_old_new_summary.py`。

#### 6.7.6 从 `s60` anchor 做低学习率 continuation：`s75/s80/s90` 比 current `s70` 更有希望，但收益结构继续分化

基于上面的观察，本轮没有继续按原配方把 `72_micro` 从 `s70` 往上硬拉，而是改成：

- 从 `s60` ckpt 开始 continuation；
- 保持 `72` objective 不变；
- 把学习率降到 `lr=5e-5`；
- 只补短 continuation budget，取 `+5/+10/+15/+20/+30`，对应总步数 `s65/s70/s75/s80/s90`。

原因是：

- 原始 `72_micro s70` 训练 log 显示，优化 loss 在 `~s60` 附近更像局部低点，`61-70` 段存在回升；
- 因此更合理的检查方式不是“原口径继续堆步数”，而是“从 `s60` 出发做低 lr continuation”，看 tail budget 是否还能换到更好的 direct-stage 指标。

本轮 continuation 训练产物：

- out_dir：`models/__tmp_72micro_tail_from_s60_lr5e5_20260308`
- console logs：`debug_output/_tmp_72micro_tail_from_s60_lr5e5_logs_20260308`
- `s65` ckpt：`models/__tmp_72micro_tail_from_s60_lr5e5_20260308/ckpt_last_WalkF_stage7_72micro_from_71m_s65_cont_froms60_lr5e5_20260308.pth`
- `s70` ckpt：`models/__tmp_72micro_tail_from_s60_lr5e5_20260308/ckpt_last_WalkF_stage7_72micro_from_71m_s70_cont_froms60_lr5e5_20260308.pth`
- `s75` ckpt：`models/__tmp_72micro_tail_from_s60_lr5e5_20260308/ckpt_last_WalkF_stage7_72micro_from_71m_s75_cont_froms60_lr5e5_20260308.pth`
- `s80` ckpt：`models/__tmp_72micro_tail_from_s60_lr5e5_20260308/ckpt_last_WalkF_stage7_72micro_from_71m_s80_cont_froms60_lr5e5_20260308.pth`
- `s90` ckpt：`models/__tmp_72micro_tail_from_s60_lr5e5_20260308/ckpt_last_WalkF_stage7_72micro_from_71m_s90_cont_froms60_lr5e5_20260308.pth`

对应 `Walk_F` freerun eval：

- `debug_output/_tmp_72micro_tail_from_s60_lr5e5_eval_20260308_Walk_F/s65/Walk_F_freerun_cycles.json`
- `debug_output/_tmp_72micro_tail_from_s60_lr5e5_eval_20260308_Walk_F/s70/Walk_F_freerun_cycles.json`
- `debug_output/_tmp_72micro_tail_from_s60_lr5e5_eval_20260308_Walk_F/s75/Walk_F_freerun_cycles.json`
- `debug_output/_tmp_72micro_tail_from_s60_lr5e5_eval_20260308_Walk_F/s80/Walk_F_freerun_cycles.json`
- `debug_output/_tmp_72micro_tail_from_s60_lr5e5_eval_20260308_Walk_F/s90/Walk_F_freerun_cycles.json`

统一相对 current `72_micro s70` 的 direct-stage compare：

- `debug_output/_tmp_72micro_s70_vs_tail_s65_20260308_Walk_F/gate_metrics.json`
- `debug_output/_tmp_72micro_s70_vs_tail_s70_20260308_Walk_F/gate_metrics.json`
- `debug_output/_tmp_72micro_s70_vs_tail_s75_20260308_Walk_F/gate_metrics.json`
- `debug_output/_tmp_72micro_s70_vs_tail_s80_20260308_Walk_F/gate_metrics.json`
- `debug_output/_tmp_72micro_s70_vs_tail_s90_20260308_Walk_F/gate_metrics.json`

关键结果如下：

1. `s65` / continuation-`s70` 都没有追平 current `s70`：
   - `s65`：`global_mean_rel_delta_pct = +0.2868%`
   - continuation-`s70`：`global_mean_rel_delta_pct = +0.1210%`
   - 说明“从 `s60` 低 lr 续到 70”并不等价于原始一口气训练到 current `s70`。

2. `s75/s80/s90` 相对 current `s70` 开始重新翻成正收益：
   - `s75`：`global_mean_rel_delta_pct = -0.2365%`，`leg8_mean_delta = -0.001881`
   - `s80`：`global_mean_rel_delta_pct = -0.2282%`，`leg8_mean_delta = -0.001815`
   - `s90`：`global_mean_rel_delta_pct = -0.1082%`，`leg8_mean_delta = -0.000860`

3. hotspot tradeoff 继续分化：
   - `foot_l/ball_l` 最好的是 `s80`：`0.4356 -> 0.4076`
   - `calf_r_global` 最好的是 `s90`：`0.3270 -> 0.3202`
   - `calf_r_sic2_4` 最好的是 `s90`：`0.5369 -> 0.5336`
   - `s75` 虽然 overall 最好，但 `calf_r_sic2_4 = 0.5511`，没有跟着一起改善。

4. current 解读应更新为：
   - `s75` 更像 **overall mean-optimal continuation candidate**；
   - `s80` 更像 **overall + foot hotspot 平衡更好** 的候选；
   - `s90` 更像 **专门压 calf_r hotspot** 的候选；
   - 但这三者相对 current `s70` 仍然都没有把 `gate_keep_lower_body` 翻回 `true`，因此它们更适合记为“局部更优 continuation 候选”，还不是干净的 drop-in winner。

因此如果后续只保留 3 个 direct-stage continuation 候选，应优先保留：

- `s75`（overall）
- `s80`（overall + foot）
- `s90`（calf）

#### 6.7.7 `s75/s80/s90 -> lambda final`：direct-stage 的小幅收益大多没有在 final blend 上保留下来

在确认 `s75/s80/s90` 是 direct-stage continuation 的相对较优点后，本轮继续把这三条接到 `lambda final`，统一复用原始 fullcompat 配置：

- config：`config/posttrain_WalkF_stage7_lambda_final_calib_20260227_fromarmchain_fullcompat.json`
- runtime 口径保持不变：`lr=2e-4, e1, s200, train_lambda_head=true`

本轮 `lambda final` 训练产物：

- out_dir：`models/__tmp_lambda_from_72micro_tail_20260308`
- console logs：`debug_output/_tmp_lambda_from_72micro_tail_logs_20260308`
- `s75` ckpt：`models/__tmp_lambda_from_72micro_tail_20260308/ckpt_last_WalkF_stage7_lambda_from_72micro_tail_s75_20260308.pth`
- `s80` ckpt：`models/__tmp_lambda_from_72micro_tail_20260308/ckpt_last_WalkF_stage7_lambda_from_72micro_tail_s80_20260308.pth`
- `s90` ckpt：`models/__tmp_lambda_from_72micro_tail_20260308/ckpt_last_WalkF_stage7_lambda_from_72micro_tail_s90_20260308.pth`

对应 `Walk_F` freerun eval：

- `debug_output/_tmp_lambda_from_72micro_tail_eval_20260308_Walk_F/s75/Walk_F_freerun_cycles.json`
- `debug_output/_tmp_lambda_from_72micro_tail_eval_20260308_Walk_F/s80/Walk_F_freerun_cycles.json`
- `debug_output/_tmp_lambda_from_72micro_tail_eval_20260308_Walk_F/s90/Walk_F_freerun_cycles.json`

统一相对现有 `lambda_from_72micro_s70` 的 blend-aware compare：

- `debug_output/_tmp_lambda72micro_s70_vs_tail_s75_20260308_Walk_F/gate_metrics.json`
- `debug_output/_tmp_lambda72micro_s70_vs_tail_s80_20260308_Walk_F/gate_metrics.json`
- `debug_output/_tmp_lambda72micro_s70_vs_tail_s90_20260308_Walk_F/gate_metrics.json`

`lambda final` 训练本身是正常的：

- 三条 run 都成功进入 `train_lambda_head` 模式；
- 训练末段 `lambda_mean ≈ 0.973`、`train-log lambda_eff_mean ≈ 0.944`，eval 侧 masked `LambdaMean/LambdaEffMean` 约为 `0.97286-0.97289`，与前面已验证的 `lambda final` 行为一致；
- 因此这轮没有出现“lambda 没有真正训起来”的问题。

但从真正的 blend-aware 指标看，结论并不乐观：

1. `s75`：相对现有 `lambda_from_72micro_s70` 整体变差
   - `blend mean/p95/p99/max: 0.499989/0.701387/0.775305/0.784823 -> 0.500377/0.708124/0.780327/0.788577`
   - `rollout mean/p95/p99/max: 0.958694/1.336697/1.520229/1.544497 -> 0.958979/1.338475/1.522598/1.545366`
   - `foot_l/ball_l blend mean/p95: 2.9519/4.1893 -> 3.0253/4.2525`
   - `calf_r_blend_global mean/p95: 1.4471/3.9749 -> 1.4646/3.9616`
   - `gate_blend_improves=false`，`gate_rollout_improves=false`，`gate_foot_hotspot_improves=false`

2. `s80`：只保留了很弱的 rollout mean 收益，但 blend / hotspot 没跟上
   - `blend mean/p95/p99/max: 0.499989/0.701387/0.775305/0.784823 -> 0.500148/0.708823/0.779496/0.789105`
   - `rollout mean/p95/p99/max: 0.958694/1.336697/1.520229/1.544497 -> 0.958412/1.342312/1.522689/1.546124`
   - `foot_l/ball_l blend mean/p95: 2.9519/4.1893 -> 2.9925/4.2188`
   - `calf_r_blend_global mean/p95: 1.4471/3.9749 -> 1.4615/3.9748`
   - `gate_blend_improves=false`，`gate_rollout_improves=true`，但 `gate_foot_hotspot_improves=false`、`gate_calf_blend_improves=false`

3. `s90`：calf 方向的 direct-stage 改善也没有成功转成更好的 final blend
   - `blend mean/p95/p99/max: 0.499989/0.701387/0.775305/0.784823 -> 0.500907/0.707782/0.782681/0.791161`
   - `rollout mean/p95/p99/max: 0.958694/1.336697/1.520229/1.544497 -> 0.959386/1.341909/1.525657/1.548865`
   - `foot_l/ball_l blend mean/p95: 2.9519/4.1893 -> 3.0291/4.2589`
   - `calf_r_blend_global mean/p95: 1.4471/3.9749 -> 1.4581/3.9446`
   - `gate_blend_improves=false`，`gate_rollout_improves=false`，`gate_calf_blend_improves=false`

4. 当前解读应更新为：
   - direct-stage continuation 确实还能在 `s75/s80/s90` 这几个点上挖到一点局部收益；
   - 但这些收益 **大多没有穿透到 `lambda final` 后的 blended rollout**；
   - `s80` 是三者里唯一还保留一点 `rollout_mean` 微弱改善的候选，但 `rollout p95/p99/max` 仍更差，而且被更差的 `blend` 全口径 / `foot hotspot` 抵消；
   - `s90` 虽然把 `calf_r_blend_global p95` 从 `3.9749` 压到 `3.9446`，但 `blend_mean` / `rollout_mean` / `foot hotspot` 仍全面更差；
   - 因此截至当前，`lambda_from_72micro_s70` 仍然是更稳的 final-stage 基线，`s75/s80/s90 -> final` 还不能替代它。

换句话说，这一轮结果把问题进一步拆清为：

- `72_micro` 的 direct-stage 还存在可继续微调的局部空间；
- 但一旦接上 `lambda final`，这些局部 direct 改善并不会自动转成更好的最终 blend；
- 因而后续如果继续优化这条 experimental lane，重点不应只是“再找更长的 direct step budget”，而应该检查 **direct 改善在 lambda head 下为何没有保留下来**。

### 6.8 当前 accepted flow 更新：`s180 low-LR trunkfull` 已通过 `71 -> 72 -> lambda`

截至 `2026-03-08` 本轮收口，当前 active 结论应更新为：

1. 真正根因已经锁定在 `current70R -> new70R`，不是 `71` 首发；
2. 原始 `70R` recipe（`direct_pose_nonleg_train_only=true` + trunk freeze + 原始 LR）会把模型锁在离更优 trunk working point 很近、但不够好的位置；
3. 当前 accepted fix 是：
   - `low LR + trunkfull`
   - `180 step + rounds=5`
   - promoted handoff ckpt：`models/__tmp_70R_new_lowlr_trunkfull_s180_20260308/ckpt_last_WalkF_stage7_70R_new_lowlr_trunkfull_s180_20260308.pth`
4. 这条 promoted handoff 已继续跑通并通过 downstream chain：
   - `71`：`models/__tmp_71_from_s180_70R_20260308/ckpt_last_WalkF_stage7_71_from_s180_70R_20260308.pth`
   - `72`：`models/__tmp_72_from_s180_71_20260308/ckpt_last_WalkF_stage7_72_from_s180_71_20260308.pth`
   - `lambda`：`models/__tmp_lambda_from_s180_72_20260308/ckpt_last_WalkF_stage7_lambda_from_s180_72_20260308.pth`
5. 相对旧 `new` chain 的 rounds=5 对比结果：
   - `71`: overall `legs_main=-0.065894`, `arms_main=-0.103466`; A=`-0.236356`, B=`-0.115982`
   - `72`: overall `legs_main=-0.026292`, `arms_main=-0.103466`; A=`-0.236356`, B=`-0.115982`
   - `lambda`: blend-aware `BlendGeoLocalDeg=-0.020124`, `GeoLocalDeg=-0.014627`, `DirectGeoLocalDeg=-0.032109`
6. 因此当前 docs / runtime 主线应更新为：
   - `Stage6 -> 70a -> 70b_concat -> 70c_replacecontacts (historical reference shell) -> promoted 70R (s180 low-LR trunkfull) -> 71 -> 72 -> lambda final`
7. 需要保留的 caution：
   - 相对 `2026-03-07 eval_on` baseline，整体/global 已更好；
   - 但 `arms_main@A/B` 仍未完全压过该 baseline，因此这条线应记为 **pass with watchlist**，不是“严格支配所有 baseline”。

当前推荐参考：

- `docs/posttrain_pipeline.md`
- `debug_output/_tmp_70R_lowlr_trunkfull_s180_rounds5_20260308/s180_verdict.md`
- `debug_output/_tmp_chain_s180promote_20260308/chain_verdict.md`
- `docs/Problems/active/2026-03-08_posttrain_s180_promote_regression_progress_checklist.md`
