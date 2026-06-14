> TRIAGE: DOWNGRADE-SUPERSEDED. Historical design/probe/planning note only; do not treat as live action-handoff delivery plan. Superseded by `2026-06-07_action_handoff_inbetween_closeout_decision_record.md` §1/§7 under its stated read-only / zero-new-injection scope.

# Action Handoff z P6 Canonical Owner Decision Note

Date: 2026-05-24
Status: Decision note (planning-only, no code implementation)

## 1. Decision Scope

This note decides the canonical orchestration owner for P6 synthetic-boundary evaluation.

In scope:
- Compare three owner options.
- Evaluate boundary cleanliness, substrate reuse, metadata extensibility, and safety-metric compatibility.
- Provide a clear recommendation.

Out of scope:
- No evaluator wiring implementation.
- No rollout execution.
- No training/posttrain entry change.

## 2. Hard Context

- P6 in current design is defined on synthetic-boundary substrate and explicitly references `debug_output/_tmp_turn_a_to_b_entry_probe_20260515/sweep_config.json`.
- `tools/run_walk_f_turn_cycle_rollout_eval.py` is currently bound to rollout-eval pilot contract, not yet canonical P6 owner.
- Current readiness/checklist direction is tools-first and no-touch for `train/training_MPL.py` and `train/posttrain.py`.

## 3. Candidate Owners

1. New standalone tool: `tools/run_action_handoff_p6_synthetic_boundary_eval.py`
2. Extend pilot tool: `tools/run_walk_f_turn_cycle_rollout_eval.py`
3. Minimal extension in `train/validate/run_freerun_cycles.py`

## 4. Comparison Matrix

### 4.1 Option A: New standalone P6 tool

- Reuse `_tmp_turn_a_to_b_entry_probe_20260515/sweep_config.json`: Yes, naturally aligned.
- Carry z retrieval metadata: Yes, clean schema owner.
- Avoid training-entry contamination: Yes.
- Connect existing safety metrics: Yes, via consuming existing freerun/rollout outputs.
- Risk profile: Lowest boundary risk; moderate new-tool maintenance cost.
- Minimal change scope: new tool + doc/contract updates; no mutation of pilot owner semantics.

### 4.2 Option B: Extend `run_walk_f_turn_cycle_rollout_eval.py`

- Reuse `_tmp_turn_a_to_b_entry_probe_20260515/sweep_config.json`: Possible but conceptually mixed.
- Carry z retrieval metadata: Yes, technically feasible.
- Avoid training-entry contamination: Yes.
- Connect existing safety metrics: Yes.
- Risk profile: Medium-high semantic drift risk (pilot scope and P6 scope mixed).
- Minimal change scope: smallest LOC delta, but highest contract-coupling risk.

### 4.3 Option C: Minimal extension in `run_freerun_cycles.py`

- Reuse `_tmp_turn_a_to_b_entry_probe_20260515/sweep_config.json`: Indirect only; poor orchestration fit.
- Carry z retrieval metadata: Possible, but pushes orchestration concern into low-level runner.
- Avoid training-entry contamination: Mostly yes, but increases validate-runner semantic load.
- Connect existing safety metrics: Native yes.
- Risk profile: Medium boundary violation risk (owner mismatch: runner vs orchestrator).
- Minimal change scope: medium; simple now, expensive later due to ownership ambiguity.

## 5. Evaluation Against Required Criteria

### 5.1 Substrate alignment

Best: Option A
Reason: P6 definition is synthetic-boundary-centric, and substrate contract can be owned directly without pilot-contract coupling.

### 5.2 z metadata ownership clarity

Best: Option A
Reason: metadata/fallback schema belongs to handoff decision layer, not rollout pilot core or low-level freerun runner.

### 5.3 Boundary cleanliness

Best: Option A
Reason: keeps pilot contract stable and avoids turning `run_freerun_cycles.py` into a mixed orchestration endpoint.

### 5.4 Safety metric reuse

All options can reuse existing metrics, but Option A does so with least semantic coupling.

### 5.5 Change-risk vs maintainability

Best balance: Option A
Reason: one-time new entry cost, lower long-term contract risk.

## 6. Recommendation (Explicit)

Recommend **Option A: new standalone canonical P6 owner**:
- `tools/run_action_handoff_p6_synthetic_boundary_eval.py`

Rationale:
- Best fit to current fact pattern: P6 is not rollout-eval pilot.
- Maintains clean owner boundaries:
  - pilot tool remains pilot tool,
  - freerun runner remains low-level metric producer,
  - P6 tool owns synthetic-boundary orchestration + z retrieval decision metadata.
- Lowest risk of contract drift and historical confusion.

## 7. Minimal Implementation Envelope (for future execution, not done here)

Phase 0 (decision freeze):
- lock canonical owner as standalone P6 tool.

Phase 1 (dry-run to real contract bridge):
- keep existing dry-run schema outputs as compatibility target.
- formalize `p6_retrieval_metadata` / `p6_fallback` required keys.

Phase 2 (tool implementation):
- implement standalone P6 tool that:
  - consumes synthetic-boundary substrate config,
  - consumes existing safety-metric artifacts,
  - emits P6 report bundle.

Phase 3 (optional integration):
- only if needed, add light shared helpers; do not repurpose pilot tool into P6 owner.

## 8. Non-Goals

- No code changes in this note.
- No direct evaluator integration in this note.
- No retraining/ablation in this note.
