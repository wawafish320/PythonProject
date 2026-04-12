# Posttrain Pipeline (Legacy Old-Boundary Chain)

> Last updated: 2026-04-12
> Status: legacy control / former canonical
> Caveat: `N=5 / limited-N`

This document preserves the former old-boundary canonical chain as a reproducible legacy-control reference.

It is no longer the current global default.
Use it only for:

- historical reproduction
- control comparisons
- old-boundary compatibility interpretation

For the current global canonical document, see:

- `docs/posttrain_pipeline.md`

---

## 1) Historical Source of Truth

The old-boundary chain was originally locked by these records:

- lane handoff and chain semantics:
  - `docs/Problems/active/2026-03-14_oldd1_newflow_leg_regression_handoff.md`
- low-drift replace decision:
  - `docs/Problems/active/2026-03-14_oldd1_skip70b_replace_lowdrift_experiment.md`
- `71` attribution and lower-LR fix:
  - `docs/Problems/active/2026-03-14_71_regression_attribution.md`
- `72` lower-LR sweep and winning choice:
  - `docs/Problems/active/2026-03-14_72_loss_curve_attribution.md`
  - `docs/Problems/active/2026-03-14_72_lowlr_sweep.md`
- `lambda` continuation from the winning `72`:
  - `docs/Problems/active/2026-03-15_72_lowlr_to_lambda.md`

The downgrade from canonical to legacy-control is supported by:

- `debug_output/_tmp_stage6_stepc_canonical_chain_20260412/decision.md`
- `debug_output/_tmp_stage6_stepc_70r_to_lambda_20260412/decision.md`

---

## 2) Legacy Runtime / Decision Contract

Historical runtime contract:

- contacts source:
  - `pretrain_contact`
- clamp:
  - `1.0`
- encoder bundle:
  - `models/motion_encoder_equiv.pt.best.pt`
- affine stats:
  - `debug_output/_tmp_phaseb_affine_20260304/affine_fit_mix08/affine_stats.json`

Historical selection / reporting policy:

- default decision contract: `model-source`
- optional archive contract: strict `pretrain_contact`
- chain selection was driven by aggregate leg-side behavior under the accepted Step B' policy of that time

---

## 3) Legacy Chain

### 3.1 Stage list

| Stage | Status | Notes |
|---|---|---|
| `Stage6` | legacy | old-boundary entry stage |
| `70a` | legacy | last plain upstream stage under old boundary |
| `new70b_replace_lowdrift` | legacy | historical operational replace handoff |
| raw `70b` | archive-only | diagnostic-only; never the accepted downstream handoff |
| `70R` | legacy | historical nonleg recovery stage |
| `71(lr=3e-4)` | legacy | historical lower-LR continuation |
| `72(lr=1e-4)` | legacy | historical lower-LR continuation |
| `lambda` | legacy | historical chain closure |

### 3.2 Stage semantics

`new70b_replace_lowdrift`

- semantic base:
  - `config/posttrain_WalkF_stage7_70b_phasezin_splitB2_pe32h512_20260227_fromarmchain.json`
- historical active overrides:
  - `lr=3e-4`
  - `epochs=1`
  - `steps_per_epoch=60`
- historical build rule:
  - build from the `70a` replace-zerophase warmstart, not from raw `70b`

`70R`

- semantic base:
  - `config/posttrain_WalkF_stage7_70R_nonleg_recovery_proj256_preleg_20260227_fromarmchain.json`
- historical promote recipe:
  - `tools/run_posttrain_nonleg_trunk_ablation.py`
  - `--trunk-mode full`
  - `--epochs 1`
  - `--steps-per-epoch 180`

`71(lr=3e-4)`

- semantic base:
  - `config/posttrain_WalkF_stage7_71_legonly_after_nonlegproj256_20260227_fromarmchain.json`
- historical override:
  - `lr=3e-4`

`72(lr=1e-4)`

- semantic base:
  - `config/posttrain_WalkF_stage7_72_legomega_after_nonlegproj256_20260227_fromarmchain.json`
- historical override:
  - `lr=1e-4`

`lambda`

- semantic base:
  - `config/posttrain_WalkF_stage7_lambda_final_calib_20260227_fromarmchain_fullcompat.json`
- historical read:
  - a direct-metric near-no-op relative to the winning old `72`

---

## 4) Why This Doc Is Still Kept

This document still matters because it is the cleanest reference for:

- historical reproduction
- old-boundary control comparisons
- interpreting what the legacy contract could absorb
- explaining why `top3` became the old-boundary-compatible anchor/control

Most accurate current status:

- this doc is no longer the default
- this doc is still the legacy-control baseline

---

## 5) Relationship to Top3

Within the current documentation system, `top3` should be read here as:

- the old-boundary-compatible anchor/control
- the donor range that this legacy contract could still absorb reliably

It should not be read here as proof of a universal natural optimum.

See:

- `docs/posttrain_pipeline_top3_anchor_control.md`

---

## 6) Caveats

- do not cite this file as the current global canonical
- do not reuse old-boundary language as the default interpretation for new posttrain decisions
- `N=5 / limited-N` still applies to the audit layer that downgraded this chain

---

## 7) Related Docs

- `docs/posttrain_pipeline.md`
- `docs/posttrain_pipeline_top3_anchor_control.md`
- `docs/posttrain_pipeline_top7_clean_stepc.md`

