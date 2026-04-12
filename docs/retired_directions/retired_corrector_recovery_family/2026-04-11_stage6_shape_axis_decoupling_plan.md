# 2026-04-11 Stage6 shape axis-decoupling plan

> Retired from active execution on 2026-04-11.  
> Postmortem status: historical redesign-planning record only; do **not** use this document as the live Stage6 execution plan or as authority for Step A / Step B checkpoint ranking.  
> Why retired: the clean-workspace sealed-spec audit downgraded the shape-axis / universal-corrector recovery track and replaced it with the `direct_pose` stabilization track plus Step B' downstream-sensitive ranking cleanup.  
> Active repo references: `docs/posttrain_pipeline.md`, `docs/train_design/2026-04-09_top3_anchor_top7_expansion_framework.md`, and `docs/train_design/2026-04-12_top7_clean_stage6_stepc_causality_record.md`.  
> Sync note: this header is a main-repo retirement pointer added so future readers do not mistake this archived plan for the current decision record.

> Status: redesign planning after R0 static audit  
> Scope: `E1-top3` first; no `E2A-R`, no bad `top7`, no mixed donor expansion

## 1. Problem rewrite

The previous bundled R0 changed too many axes at once:

- frozen donor boundary
- fresh residual/corrector parameterization
- behavior observable interface
- residual signal/objective behavior

Attribution was therefore not clean. New R0 must answer only:

> Frozen donor + any minimal trainable tail: can it at least match the `top3` locked contract Stage6 shape?

It should not try to prove a universal corrector yet.

## 2. Axes

| axis | definition | R0 setting |
|---|---|---|
| Interface axis | what observable is exposed to the tail | hold minimal; use only the then-live behavior-space tensors already produced by the old-boundary replace path |
| Signal axis | what supervision signal trains the tail | hold the then-live downstream objective; no behavioral matching loss |
| Parameter axis | additive residual vs replacement readout / corrector family | hold closest-to-identity minimal tail; no richer replacement readout family |
| Freeze boundary axis | whether donor/shared trunk/head can move | this is the only R0 variable: donor frozen, tail trainable |

## 3. New layered proposal

### R0: freeze boundary only

- Donor: `E1-top3` only.
- Frozen set includes donor encoder, shared trunk, shared head, and all heads that the locked contract used to train (`direct_pose_*`, `contact_plan_*`, `event_clock_*`, and related locked-contract readout heads/buffers as instantiated).
- The only trainable parameters in R0 are `arm_residual_corrector.*`.
- Tail: minimal trainable tail, initialized as strict identity (`gate=0` plus bitwise-preserving no-op application).
- Tail capacity floor: do not under-size the tail relative to the locked-contract trainable budget; use at least the same order of magnitude of trainable parameters so R0 failure cannot be dismissed as a trivial tail-capacity miss.
- Corrector: gate starts at exactly `0.0`; no-op path must be bitwise guarded.
- Observable: no behavior-space richer observable; use only the then-live minimal existing tensors.
- Loss / data / compute budget / epoch count / rollout schedule / eval protocol: match the locked-contract comparison as tightly as possible; the intended axis change is freeze boundary, not training budget.
- Loss: no new behavioral matching loss.
- Question: can frozen donor + minimal trainable tail at least match the `top3` locked contract?

### R1: parameter axis

- Hold donor freeze and R0 observable/loss fixed.
- Compare additive residual against replacement readout / alternative minimal corrector parameterizations.
- Do not add richer behavior observables yet.

### R2: interface axis

- Hold freeze boundary and selected R1 parameterization fixed.
- Add behavior observable richer version only here.
- Keep signal/loss unchanged so interface attribution stays clean.

### R3: signal axis

- Hold freeze boundary, parameterization, and interface fixed from R2.
- Add behavioral matching loss or other behavior-level supervision only here.

## 4. Acceptance logic

R0 should not be read as “branch improves over sham”. It should be read as:

- If R0 trainable minimal tail cannot match `top3` locked contract, freeze boundary alone is insufficient.
- If R0 matches or exceeds `top3` locked contract, then the frozen-donor boundary is viable and R1 can test parameterization.
- `donor_raw` remains an identity/baseline anchor, not a promotion target.

## 5. Minimal next experiment

Now that the static audit and runtime guard are in place, the smallest next experiment worth running is:

- first, a fix-verification replay of the original `baseline_locked` and original `sham_lr0` configs on the fixed pipeline;
- only if that replay closes the historical identity-sanity ambiguity, restart the new decoupled R0.

Then the minimal new experiment is:

- `R0_freeze_boundary_only_top3`: frozen `E1-top3` donor + minimal identity-gated trainable tail.
- Keep interface, signal, and donor selection fixed.
- Compare against the existing `top3` locked contract and `donor_raw` anchor.
- This changes only the freeze boundary axis relative to locked-contract training.

Why it is now worth running:

- warmstart helper is copy-only and bitwise donor-preserving;
- dry-run artifact-level passthrough rewrite is fixed;
- train and eval now hard-fail on donor bitwise identity mismatch;
- no branch/full experiment is interpreted before identity guard passes;
- the remaining historical ambiguity is explicitly tested by the fix-verification replay, rather than carried into the relaunched R0.

## 5.1 Parity guard

Add a runtime/eval-level prereg guard in addition to tensor identity:

- `assert_eval_metric_parity`: when two compared runs are tensor-identical on the declared runtime-preserved prefixes, their eval metrics must also match within prereg tolerance.
- If tensor identity passes but eval parity fails, classify as `runtime/eval parity failure` and stop interpretation.

This closes the gap between:

- tensor-level identity as a necessary condition;
- eval-level parity as the sufficient condition for no-op sanity.

## 6. Kill criterion

Add a calendar gate:

- By **2026-05-15**, if R0–R2 still do not produce a Stage6 shape that is `>= top3 locked contract`, downgrade universal redesign to **deferred research**.
- Production fallback then uses `top3 anchor + freeze shared head`.

This keeps universal redesign from absorbing all production time when it fails the `top3` acceptance anchor.

## 7. Anchor-preserving parallel track

Anchor-preserving is not a serial fallback. It should run as a parallel insurance track.

Its value:

- regression baseline for every universal-redesign step;
- production insurance if R0–R2 fail the 2026-05-15 gate;
- acceptance gate anchor: universal work must at least match the top3 locked contract;
- debugging anchor: separates “new method is better” from “new method merely changed the comparison contract”.

Parallel recommendation:

- Keep universal redesign on R0–R2 axis-decoupled ladder.
- In parallel, maintain `top3 anchor + freeze shared head` as the production-safe lane.
- Do not wait for universal redesign to fail before preserving the anchor lane.
