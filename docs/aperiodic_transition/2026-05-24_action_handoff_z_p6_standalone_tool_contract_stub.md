> TRIAGE: DOWNGRADE-SUPERSEDED. Historical design/probe/planning note only; do not treat as live action-handoff delivery plan. Superseded by `2026-06-07_action_handoff_inbetween_closeout_decision_record.md` §5/§6/§7 under its stated read-only / zero-new-injection scope.

# Action Handoff z P6 Standalone Tool Contract Stub

Date: 2026-05-24
Status: Contract stub (planning-only). No evaluator wiring in this document.

## 1. Tool Identity

- Planned owner tool: `tools/run_action_handoff_p6_synthetic_boundary_eval.py`
- Role: P6 synthetic-boundary orchestration owner for action-handoff z route.
- Scope level: tools-layer orchestration only.

## 2. CLI Contract

Required args:
- `--substrate-sweep-config` (`str`, path)
  - expected default source family: `debug_output/_tmp_turn_a_to_b_entry_probe_20260515/sweep_config.json`
- `--p4-sweep-summary` (`str`, path)
  - e.g. `debug_output/_tmp_action_handoff_z_probe_v1_p4_alt_sweep_20260524/p4_alt_sweep_summary.json`
- `--p4-weak-source-analysis` (`str`, path)
  - e.g. `debug_output/_tmp_action_handoff_z_probe_v1_walk_l_to_r_failure_analysis_20260524/walk_l_to_r_failure_analysis.json`
- `--z-feature-summary` (`str`, path)
  - e.g. `debug_output/_tmp_action_handoff_z_probe_v1_20260524/p4_cross_clip_entry.json`
- `--trial-matrix-json` (`str`, path)
- `--out-dir` (`str`, path)

Optional args:
- `--configs` (`str list`): selected P4-alt config IDs (default contract: `n12_q0p10_topk5 n24_q0p10_topk5`).
- `--strict` (`flag`): enable strict fail-fast on any missing optional-derived field.
- `--dry-run-only` (`flag`): allowed mode for contract validation without rollout/evaluator execution.

CLI behavior contract:
- Missing required arg -> exit code `2` with explicit `[FATAL]` message.
- Invalid JSON schema -> exit code `2`.
- Existing `out-dir` is allowed; output files are overwritten atomically.

## 3. Required Artifacts Contract

Required artifact set (all must exist and parse):
- synthetic-boundary substrate config (`sweep_config.json`)
- P4-alt sweep summary
- Walk_L_To_R weak-source analysis
- z-feature summary contract source
- trial-matrix input JSON

Artifact path policy:
- Absolute or repo-relative paths accepted.
- Resolved absolute path must be recorded in output provenance.

## 4. Trial Matrix Schema

Top-level JSON object:

```json
{
  "version": "p6_trial_matrix_v1",
  "pairs": [
    {
      "pair_bucket": "strong | weak_stress",
      "source_clip": "Walk_F",
      "target_clip": "Walk_R_To_L",
      "horizon_N": 12,
      "config_id": "n12_q0p10_topk5",
      "phase_start": null,
      "notes": "optional"
    }
  ],
  "policy": {
    "require_weak_pairs": [
      "Walk_L_To_R->Walk_R_To_L",
      "Walk_L_To_R->Walk_R_To_R"
    ],
    "require_long_horizon_N_gte": 24,
    "require_short_horizon_set": [6, 12]
  }
}
```

Validation rules:
- `pairs` non-empty.
- `pair_bucket` in `{strong, weak_stress}`.
- Each weak stress pair appears at least once in short (`N in {6,12}`) and once in long (`N>=24`) bucket.
- At least one strong-pair short + one strong-pair long sample.

## 5. `p6_retrieval_metadata` Schema

Per trial required field:

```json
{
  "p6_retrieval_metadata": {
    "enabled": true,
    "source_clip": "Walk_L_To_R",
    "source_frame": null,
    "target_clip": "Walk_R_To_L",
    "selected_target_frame": null,
    "horizon_N": 24,
    "z_distance": 1.0669,
    "z_rank_topk": 1,
    "z_margin_top1_top2": 0.2333,
    "future_equiv_score": -0.1067,
    "future_equiv_score_available": true,
    "future_equiv_quantile_q": 0.1,
    "top_k": 5,
    "mean_spearman_zdist_vs_futuredist": 0.1127,
    "num_queries": 39,
    "z_feature_contract": {
      "repr_name": "z_bottleneck",
      "repr_dim": 32,
      "dtype": "float32",
      "device": "cpu"
    },
    "value_semantics": {
      "z_distance": "top1_future_distance_vs_random_ratio proxy",
      "future_equiv_score": "top1_equiv_hit_rate_vs_random_top1",
      "z_margin_top1_top2": "random_top1_expectation - top1_equiv_hit_rate",
      "frame_indices_unavailable": true
    }
  }
}
```

Schema constraints:
- `horizon_N` required integer.
- `z_feature_contract.repr_dim/dtype/device` must be present (nullable only if source artifact lacks contract).
- If frame-level retrieval does not exist in source artifacts, `source_frame/selected_target_frame` must be `null` and `frame_indices_unavailable=true`.

## 6. `p6_fallback` Schema

Per trial required field:

```json
{
  "p6_fallback": {
    "retrieval_status": "selected | fallback | no_good_candidate",
    "fallback_triggered": true,
    "fallback_reason": "z_distance_too_large",
    "fallback_reasons_all": [
      "z_distance_too_large",
      "future_equiv_below_floor"
    ],
    "no_good_candidate": false,
    "long_horizon_warning": true,
    "warning_reason": "long_horizon_degradation_risk",
    "thresholds": {
      "max_z_distance_ratio": 0.95,
      "min_hit_lift": 0.0,
      "min_spearman": 0.2,
      "min_margin_top1_top2": 0.05,
      "long_horizon_N_gte": 24
    }
  }
}
```

Consistency rules:
- `retrieval_status=selected` -> `fallback_triggered=false`, `fallback_reason=null`.
- `retrieval_status=no_good_candidate` -> `fallback_triggered=true`, `no_good_candidate=true`.
- `long_horizon_warning=true` requires `horizon_N>=24`.

## 7. Report Schema (`JSON` + `MD`)

### 7.1 `p6_report.json`

Top-level required keys:

```json
{
  "tool": "run_action_handoff_p6_synthetic_boundary_eval",
  "status": "planning_eval_only | ready_for_impl_review",
  "generated_at_utc": "...",
  "inputs": {},
  "trial_matrix": [],
  "retrieval_summary": {},
  "fallback_summary": {},
  "safety_summary": {},
  "decision_boundary": {},
  "residual_risks": [],
  "provenance": {}
}
```

Required summary semantics:
- `retrieval_summary`: by bucket/pair/horizon distribution of `z_distance`, `future_equiv_score`, `spearman`.
- `fallback_summary`: fallback rate, no-good-candidate rate, long-horizon-warning rate.
- `safety_summary`: contract placeholders allowed in dry-run; real values required only when evaluator wiring exists.
- `decision_boundary`: weak-pair concentration vs strong-pair spillover decision statement.

### 7.2 `p6_report.md`

Required sections:
1. Status line
2. Input artifact provenance
3. Trial matrix coverage
4. Retrieval summary
5. Fallback summary
6. Safety summary (or `schema_placeholder_only` marker)
7. Decision boundary and recommendation
8. Residual risks

## 8. Failure Modes and Fail-Fast Rules

Fail-fast conditions (`exit 2`):
- Required artifact missing/unreadable.
- Required schema key missing.
- Trial matrix violates weak/strong + short/long coverage minima.
- Pair in trial matrix not found in selected P4-alt config summary.
- Numeric fields are non-finite (`NaN`, `Inf`) after parsing.

Soft warnings (continue with warning block):
- Frame-level indices unavailable (`source_frame/selected_target_frame=null`).
- Safety metrics unavailable in dry-run mode.
- Selected config marked non-pass-like globally but included intentionally for stress coverage.

## 9. Explicit Non-Goals

- No evaluator wiring in this contract stub.
- No rollout execution in this contract stub.
- No edits to `train/validate/run_freerun_cycles.py` in this step.
- No edits to `train/training_MPL.py` or `train/posttrain.py`.
- No z retraining / beta sweep / Dz sweep.

## 10. Adoption Gate

This contract stub is considered accepted only when:
- CLI contract accepted.
- artifact contract accepted.
- trial matrix schema accepted.
- retrieval/fallback schema accepted.
- report schema accepted.
- fail-fast rules accepted.

Until acceptance: planning-only; do not start evaluator wiring.
