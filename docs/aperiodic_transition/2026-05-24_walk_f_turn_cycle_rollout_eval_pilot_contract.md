# Walk_F Turn-Cycle Rollout-Eval Pilot Contract (2026-05-24)

## §0 Scope

- 本 pilot 名称固定为 `rollout-eval`，不是 `closed-loop`。
- `rollout-eval` 定义：`teacher rollout + free-run rollout` 配对的离线 evaluator。
- 本 pilot 明确 **不是** runtime arbiter，不是 handoff，不是 EventHead，不是 attractor membership，不是 phase_structured promote。
- 评估范围固定为 5-clip in-family 小世界：
  - `Walk_F`
  - `Walk_L_To_L`
  - `Walk_L_To_R`
  - `Walk_R_To_L`
  - `Walk_R_To_R`
- 本 pilot 不做 generalization claim。

## §1 Frozen Inputs / No-Retrain Rule

- checkpoint 必须 commit-pinned 与 path-pinned；`run_manifest.json` 必须记录两者。
- 明确规则：`do not retrain during evaluator pilot`。
- evaluator first-run baseline 只允许评估一个固定 checkpoint。
- 后续如需替换 checkpoint，必须作为新的 run manifest / run 目录记录，不得覆盖 baseline artifact。

## §2 Locked Eval Axes

- Clips 轴锁定：
  - `Walk_F`, `Walk_L_To_L`, `Walk_L_To_R`, `Walk_R_To_L`, `Walk_R_To_R`
- Walk_F phase start grid 锁定：`[0, 22, 44, 66]`
- Turn clip phase start grid 锁定：`[0, floor(T_clip/2)]`
- Free-run horizon 锁定：`max(T_turn_clip) + 120 = 214` frames post-start。
- Pairing 规则锁定：每个 `(clip, phase_start)` 必须同时产 `teacher_rollout` + `free_run`；任一缺失则该 row 作废并记录 `invalid_reason`。
- Eval band artifact source 锁定（只读，不重跑 probe）：
  - `debug_output/walk_f_causal_state_scaffold_v1_20260523_layerC1_query_boundary_check/summary.json`
  - `debug_output/walk_f_causal_state_scaffold_v1_20260524_layerC2_pose_phase_library_check/summary.json`
- Tool 输出目录必须显式命名为：
  - `debug_output/walk_f_turn_cycle_rollout_eval_pilot_YYYYMMDD_<run_name>`
  - 或等价的同语义显式路径（必须含日期 + run_name）。

## §3 Critical Primitive Rule

- 强约束：Evaluator **MUST consume raw primitives from C.1/C.2 artifacts, NOT verdict statuses**。

允许消费（白名单）：
- `band_quantile_value`
- `phase_loss_quantiles_on_query`
- `loss_curve`
- `out_of_band_frame_count`
- C.2 config 级 `phase_loss_quantiles` / config primitive summaries（适用时）

禁止消费（黑名单）：
- `return_to_reference_status`
- `return_to_reference_status_pre_neighbor_consistency`
- `neighbor_consistency_conflicts`
- `neighbor_consistency_conflict_pair_count`
- 任意 `*_pair_count` aggregate
- `attractor_membership_status` / `phase_structure_status` 作为 evaluator label

理由：
- C.1 verdict 已受 neighbor-consistency override 影响，常被压到 `INSUFFICIENT_EVIDENCE` / `never_left`；直接消费 verdict 会导致 evaluator 无法区分真实失败模式。

## §4 Failure Taxonomy

- evaluator 结论必须落到以下之一：
  - `PROMISING_IN_FAMILY`
  - `TRAINING_MECHANISM_FAIL.EXPOSURE_BIAS_DRIFT`
  - `TRAINING_MECHANISM_FAIL.STATE_CARRY_BUG`
  - `TRAINING_MECHANISM_FAIL.CAPACITY`
  - `TRAINING_MECHANISM_FAIL.OBJECTIVE_BLIND_TO_BAND`
  - `DATA_INSUFFICIENT_OR_AMBIGUOUS`

子假设判据：
- `EXPOSURE_BIAS_DRIFT`:
  - teacher_loss 平稳，但 free_run_loss 或 band_violation 随 horizon 单调/近单调发散。
- `STATE_CARRY_BUG`:
  - free_run 在 `k <= 10` 快速漂移，且对 `clip/start` 近乎无关。
- `CAPACITY`:
  - turn clips 的 teacher_loss 明显 saturate / 降不下去。
- `OBJECTIVE_BLIND_TO_BAND`:
  - free_run 不明显漂移，但 teacher/free 的 band-violation rate 接近，表明 objective 对 out-of-band 惩罚不足。
- `DATA_INSUFFICIENT_OR_AMBIGUOUS`（仅在同时满足时可用）：
  - per-(clip, group) return-rate 在 phase_start 网格上 `std/mean > 0.5`；且
  - band-violation curve 在 2 个 estimator-config 邻域上方向不一致。

## §4.1 Runner v1 Band-Primitive Deferral (Normative)

1. Runner v1 标签：
- 当前 runner v1 暂不做 same-scale band primitive recomputation。
- Runner v1 必须在 `summary.json` 和 `run_manifest.json` 顶层 emit：
  - `runner_version = "v1_neighborhood_proxy_only_no_band_blind"`
  - `contract_section_acknowledged = "docs/aperiodic_transition/2026-05-24_walk_f_turn_cycle_rollout_eval_pilot_contract.md §4.1"`
- 目的：未来 runner v2 恢复 full §4 taxonomy 后，旧 artifact 可被 grep 区分。

2. `OBJECTIVE_BLIND_TO_BAND` 暂停：
- `TRAINING_MECHANISM_FAIL.OBJECTIVE_BLIND_TO_BAND` 在 runner v1 中是 reserved / not emitted。
- findings memo 若使用 runner v1 artifact，禁止声明 objective-blind-to-band 结论。
- 只有 same-scale band primitive recomputation 落地后，才允许重新启用该 classifier path。

3. `AMBIGUOUS` 是更松的 proxy：
- Runner v1 的 `DATA_INSUFFICIENT_OR_AMBIGUOUS` 判据是完整 §4 conjunctive criterion 的 strict subset / looser proxy：
  - 当前只使用 neighborhood-direction conflict（part-2）。
  - `std/mean > 0.5` return-rate criterion（part-1）被短路为 `True`。
- 任何 findings memo 引用 runner v1 artifact 的 AMBIGUOUS 标签时，必须显式声明：`"v1 looser AMBIGUOUS criterion"`。
- 不得把 runner v1 的 AMBIGUOUS 当作完整 §4 判定。

4. Reopen criterion（normative, grep-able）：
- “Runner v1 的 `_classify_failure` 中 `part1 = True` 短路以及 `OBJECTIVE_BLIND_TO_BAND` emit path 的恢复，MUST 先由一份独立 memo 签下并 merge：`docs/aperiodic_transition/YYYY-MM-DD_walk_f_rollout_eval_same_scale_band_recompute_contract.md`。在该 memo 落地前，移除 `part1 = True`、重新引入 `band_violation_rate_by_group` 数值消费、或恢复 `_band_thresholds_from_c1` / `_band_metrics_from_curve` helper，均视为违反 removal_policy §3-§4 + scaffold v1 §6 的 contract-versioning 规则。”

5. Code locations under §4.1 deferral：

| item | location | status |
| --- | --- | --- |
| `_classify_failure` part1 short-circuit | `tools/run_walk_f_turn_cycle_rollout_eval.py:370` | reserved under §4.1; do not remove without same-scale recompute contract |
| `OBJECTIVE_BLIND_TO_BAND` emit path | removed; was in `_classify_failure` | reserved under §4.1; do not re-add without same-scale recompute contract |
| `_band_thresholds_from_c1` / `_band_metrics_from_curve` | removed by `23a57dd` | reserved under §4.1; only restorable via same-scale recompute contract |

## §5 Posttrain Gate

- Posttrain pilot 禁止在 evaluator 前启动。
- scheduled-sampling window 仅对应 `EXPOSURE_BIAS_DRIFT`。
- 若 evaluator 定位为 `STATE_CARRY_BUG` / `CAPACITY` / `OBJECTIVE_BLIND_TO_BAND`，禁止直接跑 scheduled sampling；先写对应 intervention memo。
- 单次 pilot 只允许改一个变量，不得混合多个改动（如 loss reweight + new head + longer teacher horizon）。

## §6 Outputs

工具必须输出：
- `run_manifest.json`
- `per_row_metrics.jsonl`
- `summary.json`
- `invalid_rows.jsonl`（如有）
- `summary.md`

每 row 最少字段：
- `clip`
- `phase_start`
- `clip_role`: `walk_f_reference | turn_query`
- `teacher_artifact_path`
- `free_run_artifact_path`
- `teacher_loss_summary`
- `free_run_loss_summary`
- `band_violation_rate_by_group`
- `out_of_band_frame_count_by_group`
- `return_like_rate_by_group`（基于 raw primitive 重算，不能读取 C.1 verdict）
- `invalid_reason` 或 `null`

## §7 Non-Outputs

显式禁止输出/改动：
- EventHead target
- handoff_ready
- transition_done
- attractor_membership
- phase_structured promote
- checkpoint writes
- training config mutation
- runtime arbiter/switch changes

## Implementation Notes (Non-Binding)

- 建议新增 orchestration tool：`tools/run_walk_f_turn_cycle_rollout_eval.py`。
- 不重写 rollout kernel；优先复用现有 teacher/free-run runner 或其 CLI。
- evaluator 允许新增 artifact reader / metric aggregator，但只能读取 frozen band artifacts；不得重跑 `tools/run_walk_f_causal_state_probe.py`。
- 若现有 teacher/free-run 路径不支持所需 `phase_start/horizon` 切片，工具层必须 fail-fast 并在输出中记录 TODO，不得硬改训练入口语义。
