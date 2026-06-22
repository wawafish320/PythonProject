> TRIAGE: DOWNGRADE-SUPERSEDED. Historical design/probe/planning note only; do not treat as live action-handoff delivery plan. Superseded by `2026-06-07_action_handoff_inbetween_closeout_decision_record.md` §1/§3.4/§7 under its stated read-only / zero-new-injection scope.

# Action Handoff z P6 Implementation Readiness Checklist (Planning Only)

Date: 2026-05-24
Status: Planning-only checklist. No evaluator/runtime code changes in this document.

## 0. Scope and Guardrails

- 目标：为 P6 implementation 做 readiness 定义，不在此文中实现。
- 本文不包含：
  - `train/validate/run_freerun_cycles.py` 代码改动
  - `train/training_MPL.py` 代码改动
  - `train/posttrain.py` 代码改动
  - z retraining、beta/Dz sweep、P6 evaluator 接线实现
- 当前结论基线保持：
  - H3 partially supported under recalibrated P4-alt yardstick
  - P1 magnitude-regression risk unresolved
  - `Walk_L_To_R` known weak-source risk
  - P6 only in planning path (no implementation clearance yet)

---

## 1. P6 orchestration entry check（先决条件）

Current check result (2026-05-24):
- `tools/run_walk_f_turn_cycle_rollout_eval.py` is the canonical **rollout-eval pilot** orchestrator under `docs/aperiodic_transition/2026-05-24_walk_f_turn_cycle_rollout_eval_pilot_contract.md`, not yet the canonical P6 synthetic-boundary orchestrator.
- Current P6 definition in the action-handoff design is still anchored to `debug_output/_tmp_turn_a_to_b_entry_probe_20260515/sweep_config.json` substrate (`docs/aperiodic_transition/2026-05-24_action_handoff_predictive_contrastive_z_probe_design.md:156`).

Implication:
- Do not directly wire P6 evaluator logic into `tools/run_walk_f_turn_cycle_rollout_eval.py` before canonical-entry decision is explicitly signed off.
- First implementation step should remain schema dry-run + contract validation.

---

## 2. P6 需要改的具体文件和最小插入点（候选计划，待 canonical entry 决策后生效）

### 2.1 Candidate insertion owner (tools layer first; only if promoted to canonical P6 owner)

1. `tools/run_walk_f_turn_cycle_rollout_eval.py`
- CLI entry and run contract: `tools/run_walk_f_turn_cycle_rollout_eval.py:417`
- run manifest build: `tools/run_walk_f_turn_cycle_rollout_eval.py:522`
- per-row assembly loop: `tools/run_walk_f_turn_cycle_rollout_eval.py:661`
- summary export: `tools/run_walk_f_turn_cycle_rollout_eval.py:747`

Planned minimal insertions:
- Add optional P6 planning flags (read-only inputs, no behavior mutation by default).
- Add per-row `p6_retrieval_metadata` / `p6_fallback` payload slots.
- Add summary-level P6 stress-case counters and concentration stats.
- Keep failure taxonomy main path unchanged; P6 fields are additive diagnostics.

### 2.2 Existing metric source (read-only consumption)

2. `train/validate/run_freerun_cycles.py`
- per-step metrics entry: `train/validate/run_freerun_cycles.py:9313`
- round summary aggregation: `train/validate/run_freerun_cycles.py:9648`
- CLI parser entry: `train/validate/run_freerun_cycles.py:10049`

Current readiness statement:
- This file already exports required safety metrics for P6 planning consumption (e.g., `GeoLocalDeg`, `RootStepDispErr`, contact-related fields when enabled).
- First implementation pass should prefer consuming existing JSON outputs in tools layer, not editing this file immediately.

### 2.3 Optional helper module (if schema logic grows)

3. Optional new file (future): `tools/action_handoff/p6_schema.py`
- Purpose: centralize P6 retrieval metadata validation/normalization to avoid bloating orchestrator.
- Not required for first implementation spike if schema stays small.

---

## 3. z retrieval metadata schema（定义）

Per-row field proposal in `per_row_metrics.jsonl`:

```json
{
  "p6_retrieval_metadata": {
    "enabled": true,
    "source_clip": "Walk_L_To_R",
    "source_frame": 24,
    "target_clip": "Walk_R_To_L",
    "selected_target_frame": 31,
    "horizon_N": 24,
    "z_distance": 0.1832,
    "z_rank_topk": 1,
    "z_margin_top1_top2": 0.0127,
    "future_equiv_score": 0.4210,
    "future_equiv_score_available": true,
    "future_equiv_quantile_q": 0.10,
    "z_feature_contract": {
      "repr_name": "z_bottleneck",
      "repr_dim": 32,
      "dtype": "float32",
      "device": "cpu"
    }
  }
}
```

Schema notes:
- `horizon_N` must be explicit (`6/12/24/48` class) to support long-horizon warnings.
- `z_margin_top1_top2` is mandatory for fallback confidence checks.
- `future_equiv_score` can be null only when `future_equiv_score_available=false`.
- `z_feature_contract` is required for reproducibility (shape implied by `repr_dim`; dtype/device explicit).

---

## 4. fallback / no-good-candidate 字段定义

Per-row field proposal:

```json
{
  "p6_fallback": {
    "retrieval_status": "selected | fallback | no_good_candidate",
    "fallback_triggered": false,
    "fallback_reason": null,
    "no_good_candidate": false,
    "long_horizon_warning": false,
    "warning_reason": null,
    "thresholds": {
      "max_z_distance": 0.0,
      "min_margin_top1_top2": 0.0,
      "long_horizon_N_gte": 24
    }
  }
}
```

Recommended reason enums:
- `fallback_reason`:
  - `z_distance_too_large`
  - `z_margin_too_small`
  - `future_equiv_below_floor`
  - `stress_pair_policy`
  - `runtime_guardrail`
- `warning_reason`:
  - `long_horizon_degradation_risk`
  - `weak_source_pair_risk`
  - `insufficient_future_equiv_signal`

Contract behavior:
- `retrieval_status=no_good_candidate` implies `fallback_triggered=true` and `no_good_candidate=true`.
- `long_horizon_warning=true` required when `horizon_N>=24` and weak-source policy triggers.

---

## 5. 强 pair + weak stress pair trial matrix（定义）

Matrix objective: separate global route readiness from known weak-pair concentration.

### 4.1 Pair buckets

- Strong bucket (normal cases): choose from consistently strong per-source/per-pair P4-alt regions.
- Weak stress bucket (must include):
  - `Walk_L_To_R -> Walk_R_To_L`
  - `Walk_L_To_R -> Walk_R_To_R`

### 4.2 Horizon buckets

- Short: `N in {6, 12}`
- Long warning zone: `N in {24, 48}`

### 4.3 Minimal trial matrix shape

- Axes:
  - pair_bucket: `strong`, `weak_stress`
  - ordered_pair: concrete `source->target`
  - horizon_bucket: `short`, `long`
  - phase_start: locked pilot phase starts
- Minimum coverage rule (planning baseline):
  - each weak stress pair must have both short and long coverage
  - strong bucket must have both short and long coverage
  - all rows must emit `p6_retrieval_metadata` + `p6_fallback`

---

## 6. P6 pass/fail report template（定义）

Output files (planned):
- `p6_readiness_report.md`
- `p6_readiness_report.json`

### 5.1 Required report sections (markdown)

1. Status line
- partially supported / not fully passed / unresolved risks

2. Coverage
- trial counts by pair bucket and horizon bucket

3. Retrieval quality
- z distance/rank/margin distributions by bucket
- future-equivalence score availability and distribution

4. Fallback behavior
- fallback rate, no-good-candidate rate
- long-horizon warning rate (`N>=24`)

5. Safety metrics (existing P6 priority)
- contact/foot/root/pose metrics summary:
  - contact mismatch family
  - foot slip family
  - `RootStepDispErr`
  - `GeoLocalDeg`

6. Decision boundary evaluation
- weak-only fail concentration vs strong-pair fail spillover
- recommended next action: hold / limited proceed with fallback / block

### 5.2 Required JSON top-level keys

```json
{
  "status": "planning_eval_only",
  "trial_matrix": {},
  "retrieval_summary": {},
  "fallback_summary": {},
  "safety_summary": {},
  "decision_boundary": {},
  "residual_risks": []
}
```

---

## 7. Boundary Confirmation（不碰 training/posttrain 入口）

Confirmed no-touch for this readiness phase:
- `train/training_MPL.py`
- `train/posttrain.py`

Preferred implementation path once approved:
1. Phase-0: canonical P6 orchestration entry decision (promote a tool as P6 owner, or define a new P6-specific tool).
2. Phase-1: schema dry-run and contract validation with existing artifacts (no rollout rerun).
3. Phase-2: tools-level additive wiring in the selected canonical P6 owner.
4. Phase-3: only if required, evaluate minimal additive export hooks in `train/validate/run_freerun_cycles.py`.
5. Keep train entry semantics unchanged; no basetrain/posttrain behavior mutation.

This ordering is consistent with current module boundaries (`train/MODULE_BOUNDARIES.md`) and with current evaluator pilot contract (`docs/aperiodic_transition/2026-05-24_walk_f_turn_cycle_rollout_eval_pilot_contract.md`).

---

## 8. Ready-to-Implement Gate (after checklist signoff)

Implementation can start only when all are accepted:
- canonical P6 orchestration entry accepted
- file touch-map accepted
- retrieval metadata schema accepted
- fallback/no-good schema accepted
- trial matrix accepted
- pass/fail template accepted
- boundary confirmation accepted

Until then: planning-only, no P6 evaluator code changes.
