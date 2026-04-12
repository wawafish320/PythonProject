# Posttrain Pipeline (Top7 Clean StepC Family Default)

> Last updated: 2026-04-12
> Status: top7-family default under clean StepC
> Caveat: `N=5 / limited-N`

This document records the current default posttrain chain for the `top7` donor family once the handoff contract is upgraded to clean StepC unified-leg-terminal semantics.

This is a family-level default.
It is not the sole definition of global canonical by itself.

---

## 1) Core Claim

Primary claim:

- `top7` is not simply “too aggressive”
- under the legacy old-boundary handoff, `top7` looked more compromised than it really was
- once the handoff is upgraded to clean StepC semantics, the early downstream drag clearly shrinks and the full chain becomes viable

Most accurate short description:

> `top7` exceeded what the legacy old-boundary contract could cleanly absorb; under clean StepC handoff it becomes a viable expansion-family default.

---

## 2) Family Default Chain

Default `top7` family chain:

- `top7 donor -> clean stage6-StepC handoff -> 70a -> new70b_replace_lowdrift -> 70R -> 71(lr=3e-4) -> 72(lr=1e-4) -> lambda`

Primary artifact roots:

- `debug_output/_tmp_top7_clean_stage6_stepc_chain_20260412/summary.json`
- `debug_output/_tmp_top7_clean_stage6_stepc_chain_20260412/decision.md`

---

## 3) Stage-Level Read

### 3.1 `70a`

Most accurate read:

- improved versus pseudo-StepC
- still mixed versus old-cut in the clean causality record
- therefore boundary mismatch is dominant, but not the sole source of early drag

### 3.2 `replace`

Most accurate read:

- `replace` already shows clean extra rescue
- this is the first stage where the legacy handoff burden clearly shrinks in a stable way

### 3.3 `70R`

Most accurate read:

- `70R` is the decisive strong-rescue stage
- clean StepC does not merely avoid collapse; it cleanly beats both `O` and `P`

### 3.4 `71 -> 72 -> lambda`

Most accurate read:

- the gain survives full downstream continuation
- `lambda` acts as chain closure rather than a new mechanism-specific win

---

## 4) Causal Interpretation

Preferred main explanation:

- legacy `stage6` handoff / fragmented boundary contract was the dominant early drag
- `top7` still carries some residual donor / early-recipe burden
- therefore:
  - boundary mismatch is the primary cause
  - donor burden is residual rather than zero

Recommended wording:

- `top7` exceeded what the legacy old-boundary contract could cleanly absorb
- clean StepC handoff makes `top7` downstream-compatible

Avoid:

- `top7 太 aggressive`
- `everything was only boundary`
- `70a residual means the StepC story failed`

---

## 5) Relationship to `top3`

Most accurate pairing:

- `top3` = anchor/control
- `top7` = expansion family

This pair should be documented as:

- anchor vs expansion
- not winner-take-all donor ideology

See:

- `docs/posttrain_pipeline_top3_anchor_control.md`

---

## 6) Relationship to the Global Canonical

The global canonical document promotes:

- the StepC boundary contract itself

This file explains how that promoted contract behaves inside the `top7` family.

Therefore:

- this doc is family-scoped
- `docs/posttrain_pipeline.md` remains the global default doc

---

## 7) Evidence Basis

Primary top7-family evidence:

- `docs/train_design/2026-04-12_top7_clean_stage6_stepc_causality_record.md`
- `debug_output/_tmp_top7_clean_stage6_stepc_chain_20260412/decision.md`
- `debug_output/_tmp_top7_posttrain_oldcut_vs_stepc_20260412/decision.md`

Accepted high-level read from those artifacts:

- `replace` already shows rescue
- `70R -> lambda` stays cleanly better
- raw `70a` still preserves a residual mixed signal

---

## 8) Caveats

- `N=5 / limited-N`
- raw `70a` is not fully clean in the `top7` family
- some hotspot behavior remains mixed
- this doc should not be cited as proof that `top7` alone defines the global canonical

---

## 9) Related Docs

- `docs/posttrain_pipeline.md`
- `docs/posttrain_pipeline_legacy_old_boundary.md`
- `docs/posttrain_pipeline_top3_anchor_control.md`
