> TRIAGE: DOWNGRADE-SUPERSEDED. Historical design/probe/planning note only; do not treat as live action-handoff delivery plan. Superseded by `2026-06-07_action_handoff_inbetween_closeout_decision_record.md` §1/§7 under its stated read-only / zero-new-injection scope.

# Action Handoff z Injection Capability Design Task Note

Date: 2026-05-25
Status: Design-only, standalone task (not wired into P6 runner scaffold)

## 0. Scope

本 note 是独立任务设计，目标是定义 synthetic-boundary injection capability 的最小可实现闭环，不与当前 P6 scaffold 链路混合实现。

In scope:
- owner module 决策（`run_freerun_cycles` vs 独立 evaluator）。
- 注入 tensor schema（rot6d/rootvel/angvel + step 语义）。
- 注入执行 seam（放在 rollout 哪个阶段）。
- metrics window 对齐契约（entry/recovery）。
- unit/smoke 验证计划（不接 P6，不跑 full matrix）。

Out of scope:
- 不改 `train/training_MPL.py`。
- 不改 `train/posttrain.py`。
- 不重训。
- 不接入 `tools/run_action_handoff_p6_synthetic_boundary_eval.py`。

## 1. Current evidence snapshot

### 1.1 当前 HEAD 的 `run_freerun_cycles` 没有 injection CLI / 参数面

- `parse_args()` 无 `--inject_*`（`train/validate/run_freerun_cycles.py:10049`）。
- `_run_freerun_cycles(...)` 参数签名也无注入参数（`train/validate/run_freerun_cycles.py:2234`）。
- P6 runner tool 已显式标注“runner CLI 不支持 inject args”（`tools/run_action_handoff_p6_synthetic_boundary_eval.py:605`, `tools/run_action_handoff_p6_synthetic_boundary_eval.py:627`）。

### 1.2 历史 substrate artifact 显示旧链路曾经具备注入能力

- 历史 trial log 包含 `--inject_pose_npz --inject_at_step --inject_from_step --inject_fields --inject_pose_hist_full`（`debug_output/_tmp_turn_a_to_b_entry_probe_20260515/trials/trial_003_Walk_R_To_L_M0_N40/run.log`）。
- 产物 `Walk_F_freerun_cycles.json` 里存在 `inject_enabled/inject_pose_npz/inject_at_step/inject_fields/inject_metadata`（`debug_output/_tmp_turn_a_to_b_entry_probe_20260515/trials/trial_003_Walk_R_To_L_M0_N40/Walk_F_freerun_cycles.json`）。
- `paired_delta.json` 保存了 `inject_metadata.applied_log`，包含实际 slice：`RootVelocity [3,5)`, `BoneRotations6D [5,281)`, `BoneAngularVelocities [281,419)`（`debug_output/_tmp_turn_a_to_b_entry_probe_20260515/trials/trial_003_Walk_R_To_L_M0_N40/paired_delta.json`）。

结论：**当前分支能力状态是“历史存在、现状缺失”**。这不是简单 CLI 未暴露，而是 capability 不在当前代码面。

## 2. Owner module decision

### Option A: 直接在 `train/validate/run_freerun_cycles.py` 重建注入能力

优点：
- rollout 状态机 owner 本就在该文件（`_run_freerun_cycles` 主循环，`apply_free_carry_raw` 调用点见 `train/validate/run_freerun_cycles.py:6668`）。
- 直接产出现有 freerun JSON，便于复用现有 downstream reducer。

风险：
- `run_freerun_cycles.py` 已很重，继续叠加 orchestration 语义会提高维护负担。
- 若直接在函数体内扩展大量注入逻辑，容易违反“最小变更”。

### Option B: 新建独立 injection evaluator（推荐）

建议新增（示例命名）：
- `train/validate/run_freerun_injection_eval.py`（entry）
- `train/validate/injection_runtime.py`（纯注入 runtime helper；无 training/posttrain 依赖）

优点：
- 保持 `run_freerun_cycles.py` 的常规 free-run 语义稳定。
- 注入能力和实验语义解耦，单测更聚焦。
- 后续 P6/Px 复用更清晰：P6 只消费该 evaluator 产物。

风险：
- 初次落地需要明确和 `run_freerun_cycles` 的共享 seam。

### Recommendation

推荐 **Option B（独立 injection evaluator）**。
`run_freerun_cycles.py` 仅保留可复用的 runtime seam（通过 helper 共享），不承担 action-handoff 注入 orchestration owner。

## 3. Injection tensor schema contract (v1 draft)

数据源约束：target clip `.npz`（例如 `Walk_R_To_L.npz`）必须含：
- `x_in_features`: shape `[T_x, Dx]`, dtype `float32`
- `bone_rot6d`: shape `[T_raw, J, 6]`, dtype `float32`
- `bone_ang_vel`: shape `[T_raw, J, 3]`, dtype `float32`
- `root_vel`: shape `[T_raw, Vr]`, dtype `float32`
- `state_layout_json`: 必须声明 `RootVelocity/BoneRotations6D/BoneAngularVelocities` slice

当前观测（Walk_R_To_L）：
- `Dx=419`, `J=46`, `Vr=2`
- `RootVelocity [3,5)`, `BoneRotations6D [5,281)`, `BoneAngularVelocities [281,419)`（来自 `state_layout_json`）

### 3.1 Runtime injection payload

`inject_payload`（device 目标为 rollout device）：
- `rot6d_raw`: `torch.Tensor`, shape `[1, J*6]`, dtype=`float32`, device=`rollout_device`
- `rootvel_raw`: `torch.Tensor`, shape `[1, Vr]`, dtype=`float32`, device=`rollout_device`
- `angvel_raw`: `torch.Tensor`, shape `[1, J*3]`, dtype=`float32`, device=`rollout_device`
- `source_frame_index`: `int`（默认 `inject_from_step`）

### 3.2 Step semantics

- `inject_at_step`: 全局 free-run step（0-based，作用于 tiled 序列步）。
- `inject_from_step`: target clip 源帧索引（0-based，映射到 target npz）。
- `length`: v1 固定 `1`（单次冲击注入），后续可扩展 `>1`。
- `inject_fields`: 子集于 `{rootvel, rot6d, angvel}`。

fail-fast:
- `inject_at_step < 0` 或 `inject_at_step >= free_steps` -> fatal。
- `inject_from_step < 0` 或 `inject_from_step >= target_len` -> fatal。
- 字段缺失/shape 不匹配/非 finite -> fatal。
- `rot6d` 长度不等于 `rot6d_x_slice` 宽度、`angvel` 不等于 `angvel_x_slice` 宽度 -> fatal。

## 4. Execution seam contract

### 4.1 注入位置（推荐）

在 free-run 主循环中，`y_used_raw` 计算完成后、`apply_free_carry_raw` 之前进行单步 override。

理由：
- `apply_free_carry_raw` 是 X-state 更新标准入口（`train/rollout_kernel.py:274`）。
- 在 carry 之前替换 `y_used_raw` 的指定分量，可保证后续 `motion_raw`、`rootvel/angvel` 推进逻辑一致（`train/validate/run_freerun_cycles.py:6668`）。

### 4.2 Pose history 语义

v1 提供两种模式：
- `inject_pose_hist_full=false`：仅修改 `y_used_raw`（最小）。
- `inject_pose_hist_full=true`：同步重写 pose_hist buffer tail（参考历史 metadata 的 `full_buffer` 语义）。

建议默认 `true`，并记录 metadata：
- `pose_hist_tail_rewrite.requested/applied/mode/stride/pose_hist_len/frame_indices`。

## 5. Metrics window alignment contract

沿用历史 substrate 口径：
- `entry_window`: `[inject_at_step-entry_window_pre_k, inject_at_step+entry_window_post_k]`，闭区间；
- `post_inject_recovery`: `[inject_at_step, inject_at_step+recovery_window_k]`，闭区间；
- 所有窗口边界都 clamp 到有效 step 范围。

当前 canonical 参数（来自 `sweep_config.json`）：
- `entry_window_pre_k=8`
- `entry_window_post_k=8`
- `recovery_window_k=16`

窗口元数据必须写入产物：
- `rel_origin_step=inject_at_step`
- `rel_key`:
  - entry 用 `step_rel_entry`
  - recovery 用 `step_rel_inject`
- `t_start/t_end/window_steps`

## 6. Capability output contract (standalone evaluator)

每个 trial 输出：
- `Walk_F_freerun_cycles.json`（含 `inject_*` 元数据和 `metrics_per_step`）
- `paired_delta.json`（含 `entry_window/post_inject_recovery.metric_summary`）
- `run.log`（命令与 stdout/stderr）

`paired_delta.metric_summary` 至少包括：
- `ContactMismatchRate`
- `FootSlipBallL`
- `FootSlipBallR`
- `RootStepDispErr`
- `GeoLocalDeg`

并为每个 metric 记录：
- `n/mean/max_abs/end/peak_step_rel`
- 条件 skip reason（若 `n==0` 且指标可条件跳过）

## 7. Unit/smoke plan (不接 P6)

### 7.1 Unit tests

新增建议：
- `tests/train/test_injection_schema_contract.py`
  - 非法 field 名、shape mismatch、越界 step、非 finite。
- `tests/train/test_injection_runtime_apply.py`
  - 单步注入后，`motion_raw` 对应 slice 被改写；
  - 未选字段保持不变。
- `tests/train/test_injection_window_contract.py`
  - `entry/recovery` 窗口边界、`window_steps`、`rel_key` 正确。

### 7.2 Smoke tests

最小两条（不接 P6）：
- normal-like: `Walk_F -> Walk_R_To_L`, `inject_at_step=40`
- weak-like: `Walk_F -> Walk_L_To_R`, `inject_at_step=80`

每条仅 1 trial，产出完整 `freerun + paired_delta + run.log`。

### 7.3 Acceptance gate（capability 级）

- 两条 smoke 均成功执行。
- 注入 metadata 显示 `applied_steps_n == 1` 且 fields 覆盖预期。
- `paired_delta` 两窗口存在且 required metrics `n>0`（foot slip 允许条件 `n==0` 但必须有 skip reason）。
- 不对 P6 pass/fail 作任何结论。

## 8. Next-step implementation boundary

实现阶段必须遵守：
- 不改 `train/training_MPL.py` / `train/posttrain.py`。
- 不在 P6 runner tool 直接扩注入逻辑。
- 先完成 capability unit/smoke 验证，再讨论接 P6。
