> TRIAGE: KEEP-AS-PROVENANCE. Historical evidence / anti-relitigation note; preserve to explain why this path was rejected or bounded. Do not use as a live plan unless a new decision record explicitly reopens it.

# Action Handoff z P6 Threshold / Acceptance Contract Note (Provisional Smoke v0.1)

Date: 2026-05-26
Status: Provisional smoke acceptance contract (not production sign-off)

## 1. Scope

本文定义 **provisional P6 smoke acceptance contract**。

目标：评估当前 8-row full-matrix injected smoke artifact：
- `debug_output/_tmp_action_handoff_p6_synthetic_boundary_eval_20260525_runner_injected_full_matrix_v3/p6_synthetic_boundary_eval_summary.json`

非目标：
- 不是 production threshold。
- 不是最终 P6 sign-off。
- 不得将本合同解释为 “P6 production passed”。

## 2. Preconditions

以下前置条件必须全部满足，否则评估状态必须 blocked：

1. 8/8 rows executed。
2. canonical metric completeness complete。
3. no proxy metrics。
4. row binding audit passed。
5. stress differentiability observed。
6. injection rootvel/rot6d applied。
7. `angvel target_slice_missing` 在当前 runner output layout 下是 expected warning（非 blocker）。

## 3. Metric Units And Direction

- `ContactMismatchRate`: fraction, lower better
- `FootSlipBallL`: m/s, lower better
- `FootSlipBallR`: m/s, lower better
- `RootStepDispErr`: meters, lower better
- `GeoLocalDeg`: degrees, lower better

## 4. Lexicographic Safety Priority

- Tier 1: contact + foot safety (`ContactMismatchRate`, `FootSlipBallL`, `FootSlipBallR`)
- Tier 2: root continuity (`RootStepDispErr`)
- Tier 3: pose quality (`GeoLocalDeg`)
- Tier 4: confidence/fallback/known-risk diagnostics

Tier 顺序不可交换：只有上层 tier 满足，才讨论下层 tier 的 accept 语义。

## 5. Row Classes

- `normal` rows: 应满足 provisional accept band。
- `weak_stress` rows: 预期可暴露已知风险；若违反 accept band，应分类为 `weak_fallback_required_known_risk` / `known_weak_source_failure`，而非 global framework rejection。

## 6. Threshold Derivation Policy

策略：

1. 若已有已签署（signed）阈值合同，优先使用 signed threshold。
2. 若无 signed threshold，则从 **normal rows** 推导 provisional smoke thresholds，并标注 `calibration-on-current-smoke`。

当前状态：未发现 signed threshold contract，因此采用当前 smoke 推导。

推导规则：

- 非零指标：
  - `threshold(metric) = max(normal_rows(metric)) * 1.10`
- 若某指标 normal 行全为 0：
  - 使用 `small_eps` 或显式文档值（当前无该情形）。

有效性边界：

- 该推导仅用于 smoke 分类。
- 不可作为 production pass 阈值。

## 7. Acceptance Statuses

Row-level statuses:

- `normal_accept`
- `normal_fail`
- `weak_pass`
- `weak_fallback_required_known_risk`
- `blocked_metric_incomplete`
- `blocked_binding_mismatch`
- `blocked_injection_not_applied`
- `inconclusive_threshold_missing`

## 8. Overall Verdict Rules

1. 若任一 canonical metric 缺失：`blocked_metric_incomplete`。
2. 若任一 normal row 在 Tier 1 或 Tier 2 fail：`p6_smoke_failed_normal_safety`。
3. 若 normal rows 全通过、weak rows 按预期失败：`p6_smoke_accept_with_known_weak_fallback_required`。
4. 若 normal 与 weak 全通过：`p6_smoke_accept_all_rows_provisional`。
5. 若阈值不可辩护（缺失或不可追溯）：`inconclusive_threshold_missing`。

## 9. Explicit Wording Policy

- 不得写 “P6 passed” 而不带 “provisional smoke”。
- 合法表达示例：
  - “P6 provisional smoke acceptance passed for normal rows; weak rows require fallback.”
  - “P6 production pass not established.”

## 10. Provisional Threshold Values (From Current Smoke)

Source normal rows（4 rows）来自：
- `debug_output/_tmp_action_handoff_p6_synthetic_boundary_eval_20260525_runner_injected_full_matrix_v3/p6_synthetic_boundary_eval_summary.json`

Derived thresholds (`max(normal)*1.10`):

- `ContactMismatchRate` = `0.42795389048991356`
- `FootSlipBallL` = `2.4616703116893768` m/s
- `FootSlipBallR` = `1.629058708712586` m/s
- `RootStepDispErr` = `0.004635973210293552` m
- `GeoLocalDeg` = `0.5552212246170306` deg

标注：`provisional smoke thresholds / calibration-on-current-smoke`。
