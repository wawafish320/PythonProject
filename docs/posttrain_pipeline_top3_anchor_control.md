# Posttrain Pipeline (Top3 Anchor / Legacy Control)

> Last updated: 2026-04-12
> Status: legacy-compatible anchor / control
> Caveat: `N=5 / limited-N`

This document defines how `top3` should be interpreted in the current posttrain documentation system.

`top3` is not the preferred label for a universally optimal donor scope.
Its primary role is:

- old-boundary-compatible anchor
- compatibility control
- reference operating range for absorbability audits

---

## 1) Core Definition

### 1.1 What `top3` is

`top3` is:

- the donor range that the legacy old-boundary downstream contract could reliably absorb
- the current control anchor for compatibility comparisons
- the cleanest legacy-compatible reference when comparing expansion-family behavior

### 1.2 What `top3` is not

`top3` is not:

- proof of a universal natural optimum
- proof that broader donor scope is intrinsically wrong
- proof that `top7` cannot work after the boundary contract is repaired

---

## 2) Why `top3` Still Matters

`top3` remains necessary because it cleanly anchors:

- what old-boundary-compatible behavior looked like
- what downstream absorbability looked like before StepC promotion
- how to separate donor burden from boundary-induced mismatch

Most accurate short description:

> `top3` is the old-boundary-compatible anchor/control, not a metaphysical optimum.

---

## 3) Recommended Terminology

Prefer:

- `top3 = old-boundary-compatible anchor/control`
- `top3 = legacy-compatible operating range`
- `top3 = anchor reference for compatibility audits`

Avoid:

- `top3 天然最优`
- `top3 is the final semantic scope`
- `top3 proves top7 is too aggressive`

---

## 4) Relationship to the Global Canonical

The current global canonical is now:

- the StepC boundary / handoff contract

Within that system, `top3` remains useful as:

- legacy control
- compatibility anchor
- donor-range reference for old-boundary absorbability questions

This means:

- `top3` still matters
- `top3` no longer defines the global canonical by itself

See:

- `docs/posttrain_pipeline.md`

---

## 5) Relationship to `top7`

Most accurate contrast:

- `top3` = anchor/control
- `top7` = expansion family

Recommended interpretation rule:

- judge `top7` by whether the expansion stays absorbable under clean StepC semantics
- do not judge `top7` only by whether it matched `top3` under legacy old-boundary conditions

See:

- `docs/posttrain_pipeline_top7_clean_stepc.md`

---

## 6) Evidence Basis

Primary framework memo:

- `docs/train_design/2026-04-09_top3_anchor_top7_expansion_framework.md`

Supporting April 12 causal update:

- `docs/train_design/2026-04-12_top7_clean_stage6_stepc_causality_record.md`

Core read retained from those artifacts:

- `top3` fits the role of what the old boundary could still absorb
- `top7` is better understood as an expansion family whose fate depends heavily on the boundary contract

---

## 7) Caveats

- `N=5 / limited-N`
- this doc is about interpretation role, not about replacing the global canonical chain
- `top3` should not be over-generalized beyond the old-boundary/control role it actually earned

---

## 8) Related Docs

- `docs/posttrain_pipeline.md`
- `docs/posttrain_pipeline_legacy_old_boundary.md`
- `docs/posttrain_pipeline_top7_clean_stepc.md`

