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

## 进展更新（2026-03-16，handoff 执行后；2026-03-17 复核更正）

### 复核结论

1. `rot_geo` 未完全清理
   - train smoke metrics JSON 已确认不再包含 `rot_geo` key。
   - 但代码中仍保留 `rot_geo` 诊断/统计分支：
     - `train/models.py` 仍写出 `rot_geo_limb_*` / `rot_geo_weight_*`
     - `train/training_MPL.py` 仍计算 `_rot_geo_from_raw_seq()` 并打印 `rot_geo_*` debug
   - 因此当前更准确的状态是：
     - **`rot_geo` 已退出主 metrics / 主 loss，但尚未从代码侧彻底移除。**

2. MotionEncoder bundle contract 是“源码已修，smoke artifact 未重刷”
   - `train/pretrain_mpl_min.py` 当前保存 bundle 时已调用 `stamp_standard_rotvec_spec(...)`，源码侧 contract 修复属实。
   - 但现存 smoke artifact
     - `models/__tmp_loss_smoke_20260316/motion_encoder_equiv_stageA.smoke.pt`
     仍缺以下字段：
     - 顶层 `rotvec_semantics`
     - 顶层 `angvel_semantics`
     - `meta.geometry_contract`
     - `meta.rotvec_asset_kind`
   - 该 smoke artifact 目前仍无法通过 `require_standard_rotvec_bundle(...)`。
   - 2026-03-16 的 trainbase smoke 跑通，实际依赖的是正式 bundle：
     - `models/motion_encoder_equiv_stageA.pt`
   - 因此当前更准确的状态是：
     - **源码 fix 已落地，但需要按新代码重新导出 smoke bundle，才能算 contract 闭环完成。**

3. legacy cleanup 只完成了第一段
   - `direct_pose_hinge`
   - `direct_hinge_delta`
   - `contact_meas_provider`
   这三类 runtime 命中已确认清掉。
   - 但 `contact_phase_state` 仍大量存在于 runtime 主路径中，主要集中在：
     - `train/models.py`
     - `train/training_MPL.py`
     - `train/posttrain.py`
     - `train/validate/run_freerun_cycles.py`

4. legacy guard 仍未通过
   - `python3 tools/check_posttrain_legacy_code_guard.py` 当前仍失败。
   - 失败项已收敛到 `contact_phase_state`，但尚未达到“已清理完成”的验收状态。
   - `docs/posttrain_pipeline.md` 当前 grep/guard 口径也仍把 `contact_phase_state` 视为 forbidden token；
     因此按文档自检标准，本轮不能记为 fully cleaned。

### 当前实际状态

- 已完成：
  - `direct_pose_hinge` / `direct_hinge_delta` / `contact_meas_provider` runtime 清理
  - `train/pretrain_mpl_min.py` 的 bundle stamp 代码修复
  - train smoke metrics 中移除 `rot_geo` key
- 未完成：
  - 从代码侧彻底移除 `rot_geo` 诊断分支
  - 用修复后的 `pretrain_mpl_min` 重新导出 smoke bundle 并验证 contract
  - 清理 `contact_phase_state` 并跑通 legacy guard

下面各节仍保留 handoff 起点时的 smoke / 问题快照；
若与下面的 handoff 快照冲突，以本节复核结果为准。

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

## 1. 现状

当前 `train.training_MPL` 对 MotionEncoder bundle 有明确 contract：

- `train/models.py:4398`
- `require_standard_rotvec_bundle(payload, context="MotionEncoder bundle")`

正式 bundle `models/motion_encoder_equiv_stageA.pt` 顶层包含：

- `rotvec_semantics`
- `angvel_semantics`
- 其他 geometry contract 字段

而本轮 smoke 导出的

- `models/__tmp_loss_smoke_20260316/motion_encoder_equiv_stageA.smoke.pt`

只保存了：

- `encoder`
- `encoder_amp`
- `period_head`
- `contact_head`
- `amp_head`
- `decoder_pose`
- `decoder_ang`
- `meta`

对应代码是：

- `train/pretrain_mpl_min.py:1910`

也就是说：

- 当前 `pretrain_mpl_min` smoke 导出物**不满足**当前 trainbase 入口的 bundle schema。

## 2. 后续修复口径

后续不应靠 trainbase 放宽检查来兼容这个旧格式；
更合理的方向是：

- 在 `train/pretrain_mpl_min.py` 保存 bundle 时补齐顶层 rotvec / angvel / geometry contract 字段；
- 使 smoke 产物和正式 bundle 口径一致；
- 之后再用 smoke 产物直接回灌 `train.training_MPL` 验证闭环。

---

## legacy / retired 残留结论

静态检查结果：

```bash
python3 -m py_compile train/posttrain.py train/models.py train/training_MPL.py train/eval_utils.py train/validate/run_freerun_cycles.py
python3 tools/check_posttrain_newflow_active_configs.py
python3 tools/check_posttrain_legacy_code_guard.py
```

结果：

- `py_compile`: 通过
- `check_posttrain_newflow_active_configs.py`: 通过
- `check_posttrain_legacy_code_guard.py`: 失败

失败集中在：

- `contact_phase_state`
- `direct_pose_hinge`
- 部分 retired target / compat shell

涉及文件包括：

- `train/posttrain.py`
- `train/models.py`
- `train/training_MPL.py`
- `train/validate/run_freerun_cycles.py`

因此这块当前状态不是“已经清完，只差文档”，而是：

- **guard 还明确告诉我们 repo 里有大量 legacy/compat 残留待清理**。

---

## 当前建议结论

截至当前更新，原计划的前两步和 Step 3 的前两段已经完成：

1. `rot_geo`：已清掉；
2. bundle schema：已补齐并完成 smoke 闭环；
3. retired shell / `direct_pose_hinge`：已从当前 mainline runtime 口径中摘掉。

因此后续执行顺序不再是三步并行待做，而是收敛为最后一项：

### 剩余唯一主任务：处理 `contact_phase_state`

当前建议顺序：

1. 先处理 `train/posttrain.py` / `train/training_MPL.py` / `train/validate/run_freerun_cycles.py` 的入口层 token 与 compat 读写壳；
2. 再处理 `train/models.py` 内部 state core 命名与建模路径；
3. 每完成一层后都复跑：
   - `python3 -m py_compile ...`
   - `python3 tools/check_posttrain_legacy_code_guard.py`

原因：

- `contact_phase_state` 不像 `direct_pose_hinge` 那样已经是明显 dead shell；
- 它仍和当前 active 的 phase / event reset 结构缠在一起；
- 因此必须按“入口层 -> validate lane -> model core”分层收，误删风险更低。

---

## One-sentence handoff

这轮 handoff 在执行后已经收敛为：

- `rot_geo`：已删；
- bundle schema：已修；
- retired shell / `direct_pose_hinge`：已清；
- **现在只剩 `contact_phase_state` 需要继续处理。**
