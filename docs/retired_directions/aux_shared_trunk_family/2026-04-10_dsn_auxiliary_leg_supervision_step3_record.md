# 2026-04-10 DSN auxiliary leg supervision step3 record

> Status: archived / retired aux-family mechanism record
> Reader note: this aux / shared-trunk family did **not** become current repo mainline; any `recommend`, `default`, `ship`, `mainline`, or `current` wording below is historical family-local language only.
> Current entry points: `docs/posttrain_pipeline.md`, `docs/train_design/2026-04-11_donor_contract_basin_universal_debug_synthesis.md`, `docs/retired_directions/aux_shared_trunk_family/2026-04-11_aux_detach_downstream_reevaluation_record.md`

> Status: step3 deploy/export strip landed
> Scope: final posttrain export strip + focused static sanity + checklist/doc backfill

## 1. Goal

- 让 aux-trained artifact 保留 train-time 版本
- 同时导出 baseline-compatible deploy/handoff artifact
- 不改 Step 1 attach point，不改 Step 2 loss/logging/train-mode 语义

## 2. Files touched

- `train/posttrain.py`
- `tests/train/test_posttrain_direct_pose_aux_leg.py`
- `docs/retired_directions/aux_shared_trunk_family/2026-04-10_dsn_auxiliary_leg_supervision_implementation_checklist.md`
- `docs/retired_directions/aux_shared_trunk_family/2026-04-10_dsn_auxiliary_leg_supervision_step3_record.md`

## 3. Strip semantics

实现位置：`train/posttrain.py`

本轮新增两个最小 helper：

- `_strip_direct_pose_aux_leg_state_dict_for_export(...)`
- `_strip_direct_pose_aux_leg_posttrain_cfg_for_export(...)`

语义如下：

1. final export 前检查 `model.state_dict()` 是否包含 `direct_pose_aux_leg_head.*`
2. 若包含：
   - 保存 train artifact：`ckpt_last_train_{run_name}.pth`
   - 保存 handoff artifact：`ckpt_last_{run_name}.pth`
3. handoff artifact 会：
   - 从 state-dict 中移除 `direct_pose_aux_leg_head.*`
   - 在 `posttrain_cfg` 中把 `direct_pose_aux_leg_enable=false`
   - 同时把 aux weight / warmup / hold / decay / min_weight / log_enable 归零或关闭
4. 若当前模型本来就没有 aux head tensor，则保持原行为，只写 `ckpt_last_{run_name}.pth`

说明：

- 这次没有扩成通用 state-dict 清洗框架
- `ckpt_step_*` snapshot 仍保持 train-time 语义，不做 strip
- main direct output schema 未改

## 4. Focused sanity

新增 focused unit-like test：`tests/train/test_posttrain_direct_pose_aux_leg.py`

覆盖点：

- aux-enabled model 的 raw state 经过 strip 后，不再包含 `direct_pose_aux_leg_head.*`
- strip 后的非 aux tensor 与原 state 保持一致
- stripped state 可被 baseline-compatible model `load_state_dict(strict=False)` 正常接受
- handoff `posttrain_cfg` strip 后，aux enable / weight / schedule / logging 开关被关闭

## 5. Lightweight validation

实际运行命令：

- `python3 -m py_compile train/posttrain.py tests/train/test_posttrain_direct_pose_aux_leg.py`
- `python3 -m unittest tests.train.test_posttrain_direct_pose_aux_leg`

结果：

- `py_compile` 通过
- `unittest` 通过：`Ran 4 tests in 0.057s`, `OK`
- 命令输出中有 `tqdm not found` warning，但不影响本轮 static sanity 结论

## 6. Result

Step 3 目标已补齐：

- aux-trained final export 现在明确区分 train artifact 与 handoff artifact
- `direct_pose_aux_leg_head.*` 不会成为 downstream required contract
- stripped artifact 维持 baseline-compatible direct-pose state contract

## 7. Still not done

本轮明确未做：

- 训练
- 完整 posttrain CLI 跑通
- `70a -> 70b replace` 真实链路验证
- downstream 训练逻辑本体改动

剩余项仅保留 Step 4：

- `70a -> 70b replace` baseline-contract 链路验证
