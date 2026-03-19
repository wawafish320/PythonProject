# 2026-03-16 Trainbase smoke / loss cleanup handoff

## 目标

把这轮针对 `pretrain_mpl_min`、`train.training_MPL`、`docs/posttrain_pipeline.md` 主线的 smoke 结果固定下来，
为后续分步执行下面三件事做 handoff：

1. 移除 `rot_geo` 这类“有计算/有统计、但不进入总 loss”的遗留项；
2. 修 `train/pretrain_mpl_min.py` 导出的 MotionEncoder bundle 格式，使其满足当前 trainbase 的 rotvec contract；
3. 系统清理 `contact_phase_state` / `direct_pose_hinge` / legacy guard 残留。

本文档只记录：

- 这轮 smoke 实际跑了什么；
- 哪些结果已经确认；
- 当前建议的执行顺序；
- 哪些地方先不要误判。

---

## 进展更新（2026-03-18，同步到当前代码状态）

### 当前结论

1. `rot_geo` 清理已完成
   - `train/models.py` 与 `train/training_MPL.py` 里原先的 runtime / debug / stats 残留已经移除。
   - 本文档不再把 `rot_geo` 视为未完成项。

2. MotionEncoder bundle contract 已闭环
   - `train/pretrain_mpl_min.py` 的 `_gather_state()` 现在会调用
     `stamp_standard_rotvec_spec(..., asset_kind="motion_encoder_bundle", source="pretrain_mpl_min")`。
   - 当前 smoke 导出物已验证可通过 `require_standard_rotvec_bundle(...)`。

3. validate lane retired 指标清理已完成
   - `train/validate/run_freerun_cycles.py` / `train/validate/run_teacher_rollout.py`
     不再包含 `direct_pose_hinge` / `direct_hinge_delta`。

4. 本轮补齐了最后一批静态残留
   - `train/posttrain.py` 的 retired shell token 表中已移除 `direct_pose_hinge_` / `direct_hinge_delta`。
   - active trainbase configs 已移除 `contact_phase_state_*`：
     - `config/exp_phase_mpl.clean.json`
     - `config/exp_phase_DirectBranch_v1_d1_noreset.json`
     - `config/exp_phase_DirectBranch_v1_d1_noreset_compat_20260226.json`
   - `tools/check_posttrain_newflow_active_configs.py` 已补齐，当前会静态检查：
     - 3 个 active trainbase configs
     - 6 个 canonical newflow posttrain configs

### 当前实际状态

- 已完成：
  - `rot_geo` cleanup
  - `pretrain_mpl_min` MotionEncoder bundle contract 修复与 smoke 验证
  - validate lane `direct_pose_hinge` / `direct_hinge_delta` 清理
  - `train/posttrain.py` retired shell token 收尾
  - active config `contact_phase_state_*` 清理
  - active config checker / docs 同步
- 未完成：
  - 本 handoff 范围内无剩余代码清理项。
  - 后续若需要新的 smoke artifact，只需按当前源码重跑导出/训练，不需要再放宽 contract 或恢复 legacy shell。

下面各节保留原始 smoke 记录作为背景；
若与历史快照冲突，以本节当前状态为准。

---

## 本轮 smoke 口径

### A. pretrain smoke

实际执行的是用户命令的轻量版，缩到单 clip / 单 epoch：

```bash
python3 -m train.pretrain_mpl_min \
  --in_glob './raw_data/processed_data/Walk_F.npz' \
  --out models/__tmp_loss_smoke_20260316/motion_encoder_equiv_stageA.smoke.pt \
  --out_best models/__tmp_loss_smoke_20260316/motion_encoder_equiv.pt.best.smoke.pt \
  --epochs 1 \
  --lr 3e-4 \
  --amp_scale_min 0.2 \
  --amp_scale_max 2.5 \
  --w_amp_equiv 1.0 \
  --w_amp_rank 0 \
  --w_amp_rel 0 \
  --T_w 50 \
  --batch_size 4 \
  --log_every 1
```

关键结果：

- 成功完成，无 NaN / 无 backward 报错；
- 输出：
  - `models/__tmp_loss_smoke_20260316/motion_encoder_equiv_stageA.smoke.pt`
  - `models/__tmp_loss_smoke_20260316/motion_encoder_equiv.pt.best.smoke.pt`
- 日志确认本轮只有 `amp_equiv` 还在有效权重里，`amp_rank` / `amp_rel` 仅计算统计，不参与本次总 loss。

### B. trainbase smoke

先尝试直接使用 smoke 导出的 bundle 跑：

```bash
python3 -m train.training_MPL ... --encoder_path ./models/__tmp_loss_smoke_20260316/motion_encoder_equiv_stageA.smoke.pt
```

结果：

- 失败；
- 失败原因不是训练逻辑本身，而是 bundle contract 不满足当前入口：
  - `train.models.require_standard_rotvec_bundle`
  - `rotvec_semantics=standard_axis_angle_v1` 顶层字段缺失
- 随后改用正式 bundle `models/motion_encoder_equiv_stageA.pt`，trainbase smoke 跑通。

实际成功执行的是：

```bash
python3 -m train.training_MPL \
  --config_json config/exp_phase_mpl.clean.json \
  --run_name exp_phase_DirectBranch_v1_d1_smoke_20260316 \
  --out ./models/__tmp_loss_smoke_20260316/MLPL2_DirectBranch_v1 \
  --depth 3 \
  --encoder_path ./models/motion_encoder_equiv_stageA.pt \
  --contact_plan_enable \
  --contact_plan_init_mode learnable+obs \
  --contact_plan_init_hidden 128 \
  --direct_pose_enable --w_direct_pose 0.2 \
  --contact_plan_time_pe_dim 16 \
  --direct_pose_meas_mode concat \
  --direct_pose_meas_drop_prob 0.1 \
  --direct_pose_plan_drop_prob 0.1 \
  --direct_pose_meas_noise_std 0.03 \
  --use_event_clock \
  --event_clock_max_delta 0.5 \
  --event_clock_hidden_dim 64 \
  --event_clock_gate_hidden_dim 32 \
  --event_clock_lambda_entropy_weight 0.01 \
  --event_clock_lambda_prior_weight 0.01 \
  --event_clock_delta_z_l2_weight 0.001 \
  --epochs 1 \
  --batch 2 \
  --num_workers 0 \
  --train_files ./raw_data/processed_data/Walk_F.npz \
  --monitor_batches 1 \
  --teacher_eval_max_batches 1 \
  --eval_horizon 20 \
  --eval_warmup 10 \
  --force_valfree_eval \
  --log_every 1
```

关键结果：

- `GradConn` 自检通过：
  - `"[GradConn] ok: window=8 grad_hits=115."`
- train / teacher / valfree metrics 都写出：
  - `models/__tmp_loss_smoke_20260316/MLPL2_DirectBranch_v1/exp_phase_DirectBranch_v1_d1_smoke_20260316/metrics/train_ep001.json`
  - `models/__tmp_loss_smoke_20260316/MLPL2_DirectBranch_v1/exp_phase_DirectBranch_v1_d1_smoke_20260316/metrics/teacher_ep001.json`
  - `models/__tmp_loss_smoke_20260316/MLPL2_DirectBranch_v1/exp_phase_DirectBranch_v1_d1_smoke_20260316/metrics/valfree_ep001.json`
- 训练本身成功，末尾只有 ONNX 导出失败：
  - `Module onnx is not installed!`
  - 该问题不影响本轮 loss / backward smoke 结论。

### C. posttrain pipeline smoke

按 `docs/posttrain_pipeline.md` 主链脚本做 smoke。

结果分两类：

- 成功：
  - `tools/run_oldd1_skip70b_replace_compare.py`
  - `tools/run_oldd1_skip70b_lowdrift_to71.py`
  - `tools/run_71_lowlr_sweep.py`
  - `tools/run_72_lowlr_sweep.py`
  - `tools/run_72_lowlr_to_lambda.py`
- 失败：
  - `tools/run_oldd1_newflow_chain.py`

失败原因：

- 不是训练/评估链路坏掉；
- 是它在汇总阶段依赖的 reference compare artifact 缺失：
  - `debug_output/_tmp_chain_s180promote_20260308/compare_vs_accepted_r5_direct/global_signal_summary.txt`

因此，本轮对 `docs/posttrain_pipeline.md` 的结论是：

- 当前主链 runner 大多可走快路径完成；
- 但 `run_oldd1_newflow_chain.py` 仍依赖一份本地未归档的 accepted compare 文本工件，后续若要把它当成完整 smoke 入口，需要先补 reference artifact 或降级该依赖。

---

## 这轮最重要的 loss 结论

## 1. `rot_geo` 现在是“计算 + 统计”，不是有效训练项

代码证据：

- `train/models.py:5083` 的 `_forward_base_inner()` 会计算 `l_geo`
- 但这里只把它写进：
  - `stats['rot_geo']`
- 实际初始化的 `loss` 是：
  - `loss = self.w_attn_reg * l_attn`
- 后续 `forward()` 中没有任何 `rot_geo` 对应的 `_submit_component_loss(...)`

所以当前口径非常明确：

- `rot_geo` 不是“没跑到代码”；
- 它是**有计算 / 有统计 / 不进入总 loss / 不参与反传**。

结合这轮 smoke，当前建议不是“把 `rot_geo` 接回去”，而是：

- **按清理目标直接移除 `rot_geo` 这条残留统计支路**。

这与当前状态更一致，因为：

- 现有训练结果已经可用；
- `rot_geo` 没有被当作优化目标；
- 后续保留它只会继续制造“像 loss 但其实不是 loss”的歧义。

## 2. trainbase 里还有几类“统计项看起来像 loss，但本质不是训练目标”

### A. `contact_plan_mse`

代码位置：

- `train/models.py:5854`

当前状态：

- 只作为 `extra_stats` 挂到 stats；
- 真正进总 loss 的是 `contact_plan_bce` / `contact_plan_weighted`。

### B. `direct_pose_geo` / `dir_*` / `dir_group_norm_*`

代码位置：

- `train/models.py:5737`
- `train/models.py:5824`

当前状态：

- `direct_pose_objective` / `direct_pose_weighted` 才是有效训练项；
- `direct_pose_geo`、`direct_pose_geo_deg`、`dir_leg_base`、`dir_nonleg_base`、`dir_arm_base`、`dir_group_norm_*` 等，都是为了诊断 direct pose 行为的统计展开，不单独形成新的 loss 项。

### C. `event_clock_lambda_mean`

代码位置：

- `train/models.py:5914`

当前状态：

- `event_clock_lambda_entropy`
- `event_clock_lambda_prior`
- `event_clock_delta_z_l2`

这三个是有效 regularization；

- `event_clock_lambda_mean` 只是观测统计。

## 3. pretrain 里本轮显式关闭的 loss 项

代码位置：

- `train/pretrain_mpl_min.py:2089`
- `train/pretrain_mpl_min.py:2136`

这轮 pretrain smoke 中：

- `loss_amp_rank` 仍计算；
- `loss_amp_rel` 仍计算；
- 但因为用户命令显式传了：
  - `--w_amp_rank 0`
  - `--w_amp_rel 0`

所以它们在这轮 run 中只留作日志统计，不进入总 loss。

这不属于 dead code，更像：

- “代码仍 active，但在本轮配置下权重为 0”。

---

## 这轮 smoke 的 trainbase 数值读法

来自：

- `models/__tmp_loss_smoke_20260316/MLPL2_DirectBranch_v1/exp_phase_DirectBranch_v1_d1_smoke_20260316/metrics/train_ep001.json`

关键信息：

- `loss = 0.707617`
- `loss_group/core = 0.060067`
- `loss_group/aux = 0.647550`

说明：

- 这轮 smoke 里主要贡献来自 aux 项；
- 其中最大头是 `contact_plan_weighted = 0.654257`；
- `direct_pose_weighted = 0.017596`；
- `rot_ortho_weighted = 0.000654`；
- event clock 三项加起来量级很小；
- `rot_geo` 只记了 `stats['rot_geo'] = 0.016421`，并未反映到 `loss_group/core` 中。

这再次印证：

- `rot_geo` 当前不是训练目标；
- 若继续保留，会让 train metrics 看起来像“loss inventory”，但实际上混着一批非反传统计。

---

## bundle contract 结论

## 1. 当前状态

当前 `train.training_MPL` 仍保持严格 MotionEncoder bundle contract：

- `train/models.py`
- `require_standard_rotvec_bundle(payload, context="MotionEncoder bundle")`

而 `train/pretrain_mpl_min.py` 已同步到同一口径：

- 保存 bundle 时补齐顶层 rotvec / angvel / geometry contract 字段；
- `_gather_state()` 会调用标准 stamp helper；
- smoke 导出物已验证可通过 `require_standard_rotvec_bundle(...)`。

## 2. 当前策略

这里不再建议放宽 trainbase 入口检查。
后续若要刷新 smoke artifact，应直接使用当前 `pretrain_mpl_min` 重新导出，
保持 smoke 产物与正式 bundle 同一 schema。

---

## legacy / retired 残留结论

静态检查结果：

```bash
python3 -m py_compile train/posttrain.py train/models.py train/training_MPL.py train/eval_utils.py train/validate/run_freerun_cycles.py train/validate/run_teacher_rollout.py
python3 tools/check_posttrain_newflow_active_configs.py
python3 tools/check_posttrain_legacy_code_guard.py
```

结果：

- `py_compile`: 通过
- `check_posttrain_newflow_active_configs.py`: 通过
- `check_posttrain_legacy_code_guard.py`: 通过

补充状态：

- runtime leftover 扫描里，`direct_pose_hinge` / `direct_hinge_delta` 已不再出现在 mainline runtime / validate lane；
- `train/posttrain.py` 里原先仅剩的 retired shell token 表残留也已删除；
- active trainbase configs 里的 `contact_phase_state_*` 已清空；
- 当前这条 handoff 不再有 legacy guard blocker。

---

## 当前建议结论

这份 handoff 对应的 cleanup 已经收口：

1. `rot_geo`：已清；
2. bundle schema：已修并完成 smoke contract 验证；
3. validate lane retired tokens：已清；
4. `train/posttrain.py` retired shell token：已清；
5. active configs / checker / docs：已同步。

后续若继续扩展 smoke 或 posttrain lane，建议直接以
`docs/posttrain_pipeline.md` 的 Validation Checklist 作为新的静态入口，
而不是再沿用这里早期“只剩 `contact_phase_state`”的旧判断。

---

## One-sentence handoff

这轮 handoff 已完成闭环：

- `rot_geo` 已删；
- MotionEncoder bundle contract 已闭环；
- retired shell / validate leftovers 已清；
- active configs、checker 与 pipeline docs 已同步；
- **当前没有挂起的 cleanup blocker。**
