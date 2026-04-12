# DSN Auxiliary Leg Supervision Step 2 Record

> Status: archived / retired aux-family mechanism record
> Reader note: this aux / shared-trunk family did **not** become current repo mainline; any `recommend`, `default`, `ship`, `mainline`, or `current` wording below is historical family-local language only.
> Current entry points: `docs/posttrain_pipeline.md`, `docs/train_design/2026-04-11_donor_contract_basin_universal_debug_synthesis.md`, `docs/retired_directions/aux_shared_trunk_family/2026-04-11_aux_detach_downstream_reevaluation_record.md`

日期：`2026-04-10`

## Touched Files

- `tests/train/test_posttrain_direct_pose_aux_leg.py`
- `docs/retired_directions/aux_shared_trunk_family/2026-04-10_dsn_auxiliary_leg_supervision_step2_record.md`

## Scope Kept

本轮只做 Step 2：

- 在现有 `train_direct_pose` rollout/loss 路径接入 `L_aux_leg`
- 增加 aux weight schedule helper
- 增加最小必需 logging
- 补强 direct-train trainable param selection sanity，确认 `direct_pose_aux_leg_head.*` 会进入 trainable params

本轮明确**没有做**：

- export strip
- 70a/70b replace/downstream 接线
- standalone aux train mode
- full training / full posttrain run
- KL / consistency / distillation / cross-head agreement
- 新 normalization / EMA-heavy aux tower

## Loss Wiring

实现位置：`train/posttrain.py`

主接线点：

1. `ret["direct_pose_aux_leg"]` 在 rollout unroll 单步内读取
2. 仅当以下条件同时满足时计算 raw aux loss：
   - `objective == "direct"`
   - `direct_pose_aux_leg_enable == true`
   - forward 返回了 `ret["direct_pose_aux_leg"]`
   - `model.direct_pose_leg_out_idx` 有效
3. 当前只支持 `direct_pose_aux_leg_loss_mode="geo"`
4. rollout 内对每步 aux loss 按现有 `step_weights` 聚合，形成 rollout-level `aux_leg_loss`
5. 总 loss 形式为：

```text
L_total = L_main + aux_leg_weight * aux_leg_loss
```

其中 `L_main` 保持原 direct rollout total 语义，aux 只做加法，不改写 main direct loss，也不改写 `out_direct`。

## Leg Slice 语义来源

aux 监督目标严格复用现有 `direct_pose_leg_out_idx`：

- 从 full-Y ground-truth 中直接按 `direct_pose_leg_out_idx` 取 leg rot6d slice
- 不新增 joint mapping
- 不把 aux output 混入 main output schema

因此 aux head 的监督维度与 Step 1 scaffold 的 attach point 完全一致。

## Weight Schedule 定义

helper：`_direct_pose_aux_leg_weight(cfg, global_step)`

使用字段：

- `direct_pose_aux_leg_weight`
- `direct_pose_aux_leg_warmup_steps`
- `direct_pose_aux_leg_hold_steps`
- `direct_pose_aux_leg_decay_steps`
- `direct_pose_aux_leg_min_weight`

语义：

- `target_weight <= 0`：weighted contribution 恒为 `0`
- warmup：从 `0` 线性升到 `target_weight`
- hold：保持 `target_weight`
- decay：线性降到 `min_weight`
- decay 结束后：保持 `min_weight`
- 若 `decay_steps <= 0`：视为不做 decay，保持 `target_weight`
- `min_weight` 会被夹到 `[0, target_weight]`

## Logging Keys

新增最小必需 stats：

- `aux_leg_weight`
- `aux_leg_loss`
- `aux_leg_loss_weighted`
- `aux_leg_over_main`

定义：

- `aux_leg_weight`：当前 step 的 schedule weight
- `aux_leg_loss`：rollout 聚合后的 raw aux geo loss
- `aux_leg_loss_weighted`：`aux_leg_weight * aux_leg_loss`
- `aux_leg_over_main`：`aux_leg_loss / max(dir_geo, 1e-6)`

说明：

- disabled 时这些值显式写 `0`
- sham (`enable=true, weight=0`) 时：
  - `aux_leg_loss` 仍正常计算
  - `aux_leg_loss_weighted == 0`

## Baseline / Sham / Aux Sanity

轻量验证方式：

- `python3 -m py_compile train/posttrain.py tests/train/test_posttrain_direct_pose_aux_leg.py`
- `python3 -m unittest tests.train.test_posttrain_direct_pose_aux_leg`

覆盖结论：

- baseline (`enable=false`)
  - total loss 保持 baseline 行为
  - aux 相关 weighted contribution 为 `0`
- sham (`enable=true, weight=0`)
  - raw `aux_leg_loss` 可计算
  - `aux_leg_loss_weighted == 0`
  - total 与 baseline 一致
- aux (`enable=true, weight>0`)
  - total 中包含 weighted aux term
  - `aux_leg_loss_weighted == aux_leg_weight * aux_leg_loss`

## Trainable Param Inclusion Sanity

本轮没有新增 train mode，也没有改 optimizer 结构；只对现有 `train_direct_pose` lane 做 focused static sanity。

覆盖点：

- `_freeze_all(model)` 后，当前测试模型的参数全部被冻结
- `_unfreeze_for_train_mode(model, cfg, "direct")` 后，仅恢复现有 direct-pose 训练语义对应的参数
- `_expected_trainable_prefixes("direct", cfg=cfg, model=model)` 显式包含 `direct_pose_aux_leg_head`
- `_select_trainable_params(model)` 返回的 `params/names` 与 `requires_grad=True` 的实际参数集合一致

本轮 sanity 结果：

- direct train mode 下，`direct_pose_aux_leg_head.weight` / `direct_pose_aux_leg_head.bias` 会进入 trainable params
- `direct_pose_head.*` 与 `direct_pose_leg_head.*` 仍保持可训练，说明现有 direct-pose train 语义未被收窄
- `shared_encoder.*` 仍保持冻结，不会误进入 `_select_trainable_params(...)`
- 所有被选中的 trainable param name 都落在 `_expected_trainable_prefixes(...)` 返回的 direct 前缀集合内

实现说明：

- 这次只补了 `tests/train/test_posttrain_direct_pose_aux_leg.py` 中的 focused unit-like test
- `train/posttrain.py` 现有实现已满足本轮要求，因此未改动

## Lightweight Validation

实际运行命令：

- `python3 -m py_compile train/posttrain.py tests/train/test_posttrain_direct_pose_aux_leg.py`
- `python3 -m unittest tests.train.test_posttrain_direct_pose_aux_leg`

结果：

- `py_compile` 通过
- `unittest` 通过（`Ran 3 tests in 0.067s`, `OK`）
- 命令输出中有 `tqdm not found` warning，但不影响本轮 static sanity / unit-like test 结论

## Compatibility

- 旧 ckpt 兼容逻辑继续沿用 Step 1 scaffold
- Step 2 没有把 aux 变成 downstream/inference 必需 contract
- inference / deploy main output schema 未改

## Remaining Work

仍未完成，且本轮明确不做：

- export strip
- downstream / 70a / 70b replace integration
- full training evidence
- posttrain CLI 全流程验证
