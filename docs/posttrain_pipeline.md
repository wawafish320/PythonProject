# Posttrain Pipeline (Global Canonical / StepC Boundary)

> Last updated: 2026-04-12
> Status: current global canonical posttrain flow
> Caveat: `N=5 / limited-N`

This document defines the current global canonical posttrain boundary contract.

The canonical claim is now:

- the active default boundary contract is the StepC unified-leg-terminal handoff
- the default downstream continuation remains:
  - `70a-StepC donor -> replace-StepC -> 70R-StepC -> 71-StepC(lr=3e-4) -> 72-StepC(lr=1e-4) -> lambda-StepC`

This is a boundary-contract promotion, not a donor-family-exclusive claim.
Legacy old-boundary records and donor-family-specific notes are maintained in separate docs.

---

## 1) Source of Truth

Primary promotion artifacts:

- canonical downstream handoff verification:
  - `debug_output/_tmp_stage6_stepc_canonical_chain_20260412/decision.md`
  - `debug_output/_tmp_stage6_stepc_canonical_chain_20260412/summary.md`
- full downstream continuation from canonical `70R-StepC`:
  - `debug_output/_tmp_stage6_stepc_70r_to_lambda_20260412/decision.md`
  - `debug_output/_tmp_stage6_stepc_70r_to_lambda_20260412/baseline_vs_stepc_full_chain_comparison.md`

Cross-family supporting evidence:

- clean top7 causality record:
  - `docs/train_design/2026-04-12_top7_clean_stage6_stepc_causality_record.md`
- clean top7 chain comparison:
  - `debug_output/_tmp_top7_clean_stage6_stepc_chain_20260412/decision.md`
- top7 old-cut vs StepC bridge/control:
  - `debug_output/_tmp_top7_posttrain_oldcut_vs_stepc_20260412/decision.md`

Related interpretation docs:

- legacy old-boundary control:
  - `docs/posttrain_pipeline_legacy_old_boundary.md`
- top3 anchor/control:
  - `docs/posttrain_pipeline_top3_anchor_control.md`
- top7 family default under clean StepC:
  - `docs/posttrain_pipeline_top7_clean_stepc.md`

If this document conflicts with a newer accepted problem/audit record, update this document to match the newer accepted record.

---

## 2) What Is Being Promoted

The promoted object is:

- the StepC unified-leg-terminal boundary / handoff contract

The promoted object is **not**:

- a claim that one donor family now replaces all others as the only meaningful narrative
- a claim that every hotspot became uniformly better
- a claim that `top3` was “wrong” rather than a valid old-boundary-compatible anchor/control

Most accurate causal read:

- legacy old-boundary handoff was creating a real downstream compatibility mismatch
- StepC unified-leg-terminal semantics remove that mismatch at the real downstream interface
- the gain survives through `70a -> replace -> 70R -> 71 -> 72 -> lambda`
- local mixed hotspot behavior remains possible without overturning the canonical ranking under the locked Step B' policy

---

## 3) Locked Runtime / Decision Contract

These settings remain part of the canonical runtime / reporting definition:

- contacts source:
  - `pretrain_contact`
- clamp:
  - `1.0`
- encoder bundle:
  - canonical bundle from the promoted StepC artifacts
- affine stats:
  - `debug_output/_tmp_phaseb_affine_20260304/affine_fit_mix08/affine_stats.json`

Selection / reporting policy:

- Step A gate remains necessary-but-not-sufficient
- Step B' primary remains `all_ex_root_mean`
- tie-break1 remains `all_ex_root_p95`
- tie-break2 remains `leg_mean`
- hard reject remains fixed incumbent `nonleg_p95` threshold
- eval contract remains `model-source`

Interpretation rule:

- canonical promotion is bound to this locked policy
- do not require every local metric to improve if the accepted ranking contract still gives a stable Step B' `win`

---

## 4) Canonical Chain

### 4.1 Stage list

| Stage | Status | Notes |
|---|---|---|
| `70a-StepC donor` | required | first canonical downstream handoff stage under StepC upgrade |
| `replace-StepC` | required | canonical low-drift replace handoff under StepC |
| `70R-StepC` | required | canonical nonleg recovery stage under StepC |
| `71-StepC(lr=3e-4)` | required | locked downstream continuation |
| `72-StepC(lr=1e-4)` | required | locked downstream continuation |
| `lambda-StepC` | required | chain closure; retains the StepC gain |

### 4.2 Canonical artifact map

| Stage | Config / artifact |
|---|---|
| `70a-StepC donor` | `debug_output/_tmp_stage6_stepc_canonical_chain_20260412/configs/posttrain_70a_fromfresh_stepc_20260412.json` |
| `replace-StepC` | `debug_output/_tmp_stage6_stepc_canonical_chain_20260412/configs/posttrain_70b_replace_lowdrift_fromfresh_stepc_20260412.json` |
| `70R-StepC` | `debug_output/_tmp_stage6_stepc_canonical_chain_20260412/configs/posttrain_70R_fromfresh_stepc_20260412.json` |
| `71-StepC(lr=3e-4)` | `debug_output/_tmp_stage6_stepc_70r_to_lambda_20260412/configs/posttrain_71_from_70R_stepc_lr3e4_20260412.json` |
| `72-StepC(lr=1e-4)` | `debug_output/_tmp_stage6_stepc_70r_to_lambda_20260412/configs/posttrain_72_from_71_stepc_lr1e4_20260412.json` |
| `lambda-StepC` | `debug_output/_tmp_stage6_stepc_70r_to_lambda_20260412/configs/posttrain_lambda_from_72_stepc_20260412.json` |

### 4.3 What changed relative to the former canonical

The old document framed the canonical line as old-boundary old-cut semantics.
The current canonical line changes that framing:

- `70a` is now interpreted through the StepC donor upgrade path
- `replace` is now the StepC-preserving replace handoff
- `70R -> 71 -> 72 -> lambda` are now understood as downstream continuation of a StepC-compatible handoff, not continuation of the old boundary contract

---

## 5) Why This Is the Global Canonical

Accepted canonical evidence now supports all of the following:

1. `70a-StepC donor` already beats canonical old-cut `70a`
2. `replace-StepC` preserves that gain as a real handoff improvement
3. `70R-StepC` remains better than canonical old-cut `70R`
4. the downstream continuation through `71/72/lambda` keeps the gain rather than washing it out
5. `lambda-StepC` remains better than canonical old-cut `lambda` on the locked Step B' policy

Practical read:

- the StepC gain is real at the true downstream interface
- the gain is retained but attenuated across the full chain
- this is enough to promote the StepC boundary contract to global canonical status

Most accurate one-line summary:

> StepC unified-leg-terminal is now the default global posttrain boundary contract because its downstream compatibility gain survives the real canonical chain through final `lambda`.

---

## 6) Relationship to Legacy / Top3 / Top7

### 6.1 Legacy old-boundary chain

The former old-boundary chain is retained only as:

- legacy control
- historical reproduction target
- comparison baseline

See:

- `docs/posttrain_pipeline_legacy_old_boundary.md`

### 6.2 Top3

`top3` should now be documented as:

- old-boundary-compatible anchor
- legacy control
- compatibility reference range

It should **not** be documented as a universal natural optimum.

See:

- `docs/posttrain_pipeline_top3_anchor_control.md`

### 6.3 Top7

`top7` should now be documented as:

- a StepC-compatible expansion family
- a family-level default under clean StepC

It provides strong corroborating evidence for the same boundary-causality story, but it is not the sole definition of global canonical by itself.

See:

- `docs/posttrain_pipeline_top7_clean_stepc.md`

---

## 7) Caveats

These caveats remain mandatory:

- `N=5 / limited-N`
- some local hotspots remain mixed
- some downstream stages show small `leg_p95` tradeoffs
- promotion is policy-bound to Step B', not to a requirement that every local metric improve simultaneously

Do not write:

- `top7 太 aggressive`
- `top3 天然最优`
- `everything was only boundary`

Prefer:

- legacy old-boundary mismatch was the dominant downstream compatibility bottleneck
- StepC fixes the real handoff contract
- donor-family residual burden can still exist without overturning the canonical boundary promotion

---

## 8) Preferred Use

Use this document when answering:

- what is the current global canonical posttrain chain?
- what boundary contract is currently default?
- what should new posttrain discussion treat as the default handoff interpretation?

Do **not** use the legacy old-boundary document as the current default source of truth.

