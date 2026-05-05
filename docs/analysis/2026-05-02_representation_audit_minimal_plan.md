# Representation Audit Minimal Plan

## Branch

- working branch: `plan/representation-audit-minimal-20260502`

## Purpose

Replace the previous attribution-heavy line with a smaller, pre-registered audit that first asks:

1. Is `free_carry` producing a lineage-relevant representation drift?
2. Is the observed lineage gap already present at `single-step`, or does it amplify within the first `1-3` rollout steps?
3. Only if the gap is static-dominated, where should representation/path ablation start?

This plan intentionally does **not** start from SIC12-16 attribution.

## Fixed Contract

Keep the same canonical contract as the earlier Walk_F 4.06 work unless a later document explicitly supersedes it:

- `encoder_bundle=/private/tmp/exp_motion_head_soft_walkf/models/motion_encoder_equiv.pt.best.pt`
- `time_index_mode=cycle`
- `phase_reset_source=none`
- `contact_plan_init_mode=learnable`
- `contacts_meas_source=pretrain_contact`
- `contacts_meas_pretrain_clamp=1.0`
- `contacts_meas_pretrain_affine_stats=debug_output/_tmp_phaseb_affine_20260304/affine_fit_mix08/affine_stats.json`
- `lambda_fusion_apply=false`

Do not change:

- teacher path behavior
- normalization
- preprocessing
- RNG
- dataloader order

Summary output must include a stable contract hash:

- `contract_hash = sha256(json.dumps(sorted_contract_dict, sort_keys=True))`

The runner must assert this hash matches the intended fixed contract before emitting any result classification.

## Scope

Primary clips:

- `easy=Walk_R_To_L`
- `boundary=Walk_F`
- `hard=Walk_L_To_L`

Clip selection is frozen for the entire run.
Do not change clip scope mid-run.

Primary lineages:

- `seed2024`
- `seed2025`

Primary horizons:

- `single-step`
- `1-step`
- `3-step`

## Step 0: Prerequisite Audits

Step 1 and Step 2 are not allowed to run until both prerequisite audits are recorded in the summary.

### Step 0.A: Swap-Path Parity Audit

The previous work showed a discrepancy between:

- Round 2 `combo_trunk_motion_head` SIC12-16 closure: `0.746117`
- Round 3 `combo_trunk_motion_head` SIC12-16 closure: `0.670449`

Audit requirement:

1. Round 2 and Round 3 style swap evaluation must use the same shared swap helper.
2. BN / eval-mode handling must be identical across both paths.
3. The prerequisite summary must record the old numbers, the re-audited numbers, and the residual delta.

If the re-audited delta remains materially non-zero, Step 2 cross-lineage interpretation is blocked until the method drift is understood.

### Step 0.B: Max-Combo Single-Frame Prior Audit

Before Step 1 and Step 2, record:

- `max_combo teacher single-frame closure`
- `max_combo 1-step closure`
- `max_combo SIC12-16 closure`

Interpretation:

- if `max_combo teacher single-frame closure = 1.0`, static-representation explanations already have a strong prior
- if it is materially below `1.0`, early-amplification explanations still have significant room

These values must be committed into `summary.json` and `summary.md` before any Step 1 / Step 2 result classification.

## Step 1: Representation Self-Consistency

### Chosen Semantics

Use **(b) re-encoding consistency** as the main probe.

For each lineage, clip, and horizon `h in {1,2,3}`:

1. Run the teacher-conditioned path and capture `state_{t+h}^{gt}`.
2. Run the free-carry path and capture `state_{t+h}^{fr}`.
3. Re-feed `state_{t+h}^{gt}` and `state_{t+h}^{fr}` into the same model under the same `cond / time_index / contacts / pose_history` contract at `t+h`.
4. Compare the resulting activations.

### Required Null Gate

Before claiming any re-encoding drift:

1. Re-feed `state_{t+h}^{gt}` back into the model under the same `t+h` contract.
2. Compare that result against the teacher-path activation captured from the same `state_{t+h}^{gt}`.

Expected result:

- `teacher_refeed_null ~= 0`

If this null gate is not approximately zero, the re-feed pipeline is invalid and Step 1 stops.

### Primary Readouts

- primary: `frozen_encoder_hidden`
- secondary: `h_final`
- tertiary: `motion_head_preact`

`motion_head_preact` means the activation immediately upstream of the motion-head output projection / readout, using one shared implementation across both lineages.

### Meaning

This does **not** directly explain the seed2024 vs seed2025 4.06deg gap.
It answers whether the carry-produced state remains on the model's own representation manifold.

### Exact Metric Definition

For a chosen readout layer `L`, define:

- `act_gt(t,h,L) = act_L( refeed(state_{t+h}^{gt}) )`
- `act_fr(t,h,L) = act_L( refeed(state_{t+h}^{fr}) )`

Then:

- `drift_L(h) = mean_{valid t} mean_{batch i} RMS_c( act_fr(t,h,L,i,c) - act_gt(t,h,L,i,c) )`
- `drift_cos_gap_L(h) = mean_{valid t} mean_{batch i} ( 1 - cos( vec(act_fr(t,h,L,i,:)), vec(act_gt(t,h,L,i,:)) ) )`

Reduction conventions are fixed:

- L2 uses per-sample root-mean-square over channels
- cosine uses whole-vector cosine over the full channel dimension
- horizon scores are averaged over all valid anchor times `t` and then over batch
- no end-of-sequence-only reduction is allowed

### Noise Floor

Before claiming that two lineage drift rates are materially different, estimate:

- `noise_floor_L(h)`

Recommended procedure:

1. Repeat the same lineage / same clip / same horizon audit twice under alternate eval RNG seeds that do not change preprocessing, dataloader order, or contract.
2. If the pipeline is deterministic and both runs are identical, set `noise_floor_L(h)=0`.

Material difference rule:

- claim `lineages materially different at (L,h)` only if `|drift_gap_L(h)| > 2 * noise_floor_L(h)`

### Required Output

For each clip, each `h in {1,2,3}`, and each readout layer:

- `seed2024 drift_L(h)`
- `seed2025 drift_L(h)`
- `drift_gap_L(h) = drift_2024_L(h) - drift_2025_L(h)`
- `noise_floor_L(h)`
- `teacher_refeed_null_L(h)`

Use the same metric family throughout:

- `per-channel L2`
- `cosine gap`

### Interpretation

- both small and close: carry channel healthy; lineage gap likely elsewhere
- both large and close: structural carry fragility, but not lineage-specific
- clearly different: carry is lineage-relevant

## Step 2: Early-Horizon Gap Split

### Main Principle

Do not judge by absolute values alone.
The main quantity is **delta-of-delta**.

### Required Tables

For each clip, emit:

| metric row | single-step | 1-step | 3-step |
|---|---:|---:|---:|
| seed2024 self diff |  |  |  |
| seed2025 self diff |  |  |  |
| cross-lineage diff |  |  |  |

### Definitions

- `self diff`: lineage prediction vs its own GT under the fixed contract
- `cross-lineage diff`: seed2024 prediction vs seed2025 prediction under matched horizon/mode

`cross-lineage diff` is fixed to interpretation **(a)**:

1. start both lineages from the same teacher initialization and same matched teacher-side contract
2. let each lineage roll its own model for horizon `h`
3. compare the two predictions at the same horizon

Do not use "each lineage from its own free-run history" as the Step 2 cross-lineage definition.

Derived quantities:

- `early_amp_A(h) = self_A(h) - self_A(single-step)`
- `early_amp_B(h) = self_B(h) - self_B(single-step)`
- `cross_amp(h) = cross(h) - cross(single-step)`

All Step 2 metrics use the same fixed reduction family as Step 1:

- per-sample channel RMS for L2
- whole-vector cosine gap
- average over batch and valid anchor times

### Why 3-Step Is Required

Do not jump from `1-step` to SIC12-16.
`3-step` is the minimal ramp anchor for distinguishing:

- static-dominated gap
- early amplification within frame `2-5`
- later accumulation only

### Thresholds

Use the following pre-registered thresholds:

- `static_representation_dominant`:
  `cross(single-step) >= 0.7 * cross(3-step)`
- `early_amplification_dominant`:
  `cross(3-step) >= 1.5 * cross(single-step)`
- `late_accumulation_dominant`:
  `cross(3-step) <= 1.2 * cross(single-step)` and prior canonical evidence shows SIC12-16 is materially larger

If none of the above are cleanly satisfied, classify as:

- `ambiguous_needs_step3`

## Step 3: Conditional Path / Representation Audit

Do **not** run by default.
Only trigger if Step 2 indicates the gap is static-dominated.

### Allowed Form

Inference-time path probes only.
No retraining.

Candidate forms:

- `P1`: full canonical path
- `P2`: audited `main-only` path
- `P3`: local-vs-global writeback consistency check

### Constraint

`main-only` must be operationally identified in code before any run.
Do not use conceptual names without a concrete inference-time implementation.

### Hypothesis Requirement

Each Step 3 check must declare a one-line hypothesis before execution.
Example form:

- "If local/global pose writeback is inconsistent, then matched-horizon state re-entry should diverge before large rollout depth."

## Pre-Registered Decision Tree

### Step 1

- both lineages close and small:
  carry channel healthy; gap not primarily on this chain
- both lineages close and large:
  carry fragility exists, but it does not explain lineage difference
- lineages materially different:
  carry is lineage-relevant

### Step 2

- `single-step` already explains most of the cross gap:
  `static_representation_dominant`
- gap ramps sharply by `1-3` steps:
  `early_amplification_dominant`
- `single-step`, `1-step`, and `3-step` remain flat while later rollout is known to explode:
  `late_accumulation_dominant`

### Step 3

- trigger only when Step 2 lands on `static_representation_dominant`

## Deliverables

Produce one runner and one summary.

### Runner

Suggested path:

- `tools/run_walkf_representation_audit_minimal.py`

Runner requirements:

- read-only
- no optimizer intervention
- no retraining
- fixed metric family: `per-channel L2 + cosine`
- fixed contract above
- shared helper reuse with the earlier Walk_F attribution loaders / swap utilities where applicable
- emit machine-readable JSON
- emit human-readable Markdown summary

### Summary

Suggested path:

- `debug_output/_tmp_walkf_representation_audit_minimal_YYYYMMDD/summary.md`

The summary must answer:

0. Did Step 0.A and Step 0.B pass, and what were the prerequisite numbers?
1. Is carry-state re-encoding drift small or large within each lineage?
2. Are the two lineage drift rates close or different?
3. Is the cross-lineage gap already static at `single-step`, or does it ramp by `1-3` steps?
4. Does the evidence justify a Step 3 path audit, or is that unnecessary?

## Non-Goals

Do not do any of the following in this phase:

- SIC12-16 attribution decomposition
- SAM / SWA / soup
- retraining
- new error families
- inverse-model round-trip training
- information-capacity estimation
- changing clip selection mid-run

## Suggested Implementation Order

1. Reuse the existing Walk_F attribution loader/runtime reconstruction helpers.
2. Implement Step 1 re-encoding capture first.
3. Implement Step 2 horizon table next.
4. Write summary with the decision tree above.
5. Only after Step 1+2 are stable, decide whether Step 3 is needed.

## Exit Criteria

This phase is complete when:

- Step 0 prerequisite numbers are emitted and accepted
- Step 1 and Step 2 are both runnable on `seed2024` and `seed2025`
- all three primary clips are covered
- `single-step / 1-step / 3-step` tables are emitted
- the summary can classify the result as one of:
  - `carry_healthy_static_dominant`
  - `carry_healthy_early_amplification`
  - `carry_lineage_relevant`
  - `ambiguous_needs_step3`

If the result is `ambiguous_needs_step3`, the summary must also state:

- which two decision branches were numerically too close to separate
- which threshold comparison failed to cleanly separate them
- the one-line Step 3 hypothesis to test next
