# Posttrain Pipeline (Current Canonical Chain)

> Last updated: 2026-03-15
> Status: current default posttrain flow for the old d1 / newflow lane.

This document keeps only the active mainline:

`Stage6 -> 70a -> new70b_replace_lowdrift -> 70R -> 71(lr=3e-4) -> 72(lr=1e-4) -> lambda`

Everything else is archive-only unless a problem doc explicitly re-opens it.
In particular:

- raw `70b` stays diagnostic-only
- historical `70c_replacecontacts` is not an active handoff stage
- `71m`, `72_micro`, `hybridcarry`, `skip70c`, and legacy Stage1-5 lanes are not part of the current pipeline

---

## 1) Source of Truth

The current chain is locked by these records:

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

If this document conflicts with one of the files above, update this document to match the newer problem record.

### Reverse lookup index

Use this when you need to answer "why is this stage in the chain?" without re-reading every active note.

| Question | Open this doc | Locked answer |
|---|---|---|
| What is the real old d1 newflow handoff chain? | `docs/Problems/active/2026-03-14_oldd1_newflow_leg_regression_handoff.md` | operational chain is `Stage6 -> 70a -> new70b_replace -> 70R -> 71 -> 72 -> lambda`; raw `70b` is diagnostic-only |
| Why did we replace the old replace stage with `new70b_replace_lowdrift`? | `docs/Problems/active/2026-03-14_oldd1_skip70b_replace_lowdrift_experiment.md` | the low-drift replace stage is the cleaner upstream handoff and its calf hotspot regression is absorbed by downstream `70R/71` |
| Why is `71` now `lr=3e-4` instead of the old default recipe? | `docs/Problems/active/2026-03-14_71_regression_attribution.md` | unchanged `71` over-stepped the cleaner candidate `70R` start; same semantics + lower LR fixed it |
| Why was `72` identified as the next bottleneck after fixing `71`? | `docs/Problems/active/2026-03-14_72_loss_curve_attribution.md` | unchanged `72` flipped the aggregate lead almost immediately because of early leg-side overshoot |
| Why is the current `72` recipe `lr=1e-4`? | `docs/Problems/active/2026-03-14_72_lowlr_sweep.md` | lower LR directly fixed the `72` overshoot, and `lr=1e-4` was the best tested case |
| Why does the chain still continue to `lambda` if `72` already wins? | `docs/Problems/active/2026-03-15_72_lowlr_to_lambda.md` | `lambda` preserves the `72(lr=1e-4)` gains and is effectively a no-op on the tracked direct metrics |

---

## 2) Locked Runtime Contract

These settings are part of the pipeline definition, not optional tuning:

- base ckpt:
  - `models/MLPL2_DirectBranch_v1/exp_phase_DirectBranch_v1_d1/ckpt_best_free_exp_phase_DirectBranch_v1_d1.pth`
- contacts source:
  - `pretrain_contact`
- clamp:
  - `1.0`
- encoder bundle:
  - `models/motion_encoder_equiv.pt.best.pt`
- affine stats:
  - `debug_output/_tmp_phaseb_affine_20260304/affine_fit_mix08/affine_stats.json`

Main runtime policy:

- `train.posttrain` must keep the XOR contract: `train_direct_pose` or `train_lambda_head`
- legacy hinge/contact-hazard/contact-ttc shells are not part of mainline runtime
- `contact_phase_state*` is retired from mainline runtime and active configs
- `contact_meas_provider*` is not part of mainline posttrain runtime
- active mainline no longer treats `whitebox` as the default contact path

Selection / reporting policy for this chain:

- default decision contract: `model-source`
- optional archive contract: strict `pretrain_contact`
- current chain selection is driven by aggregate leg-side behavior, not by forcing every nonleg delta to be zero

---

## 3) Canonical Chain

### 3.1 Stage list

| Stage | Recipe | Status | Notes |
|---|---|---|---|
| `Stage6` | `config/posttrain_WalkF_stage6_direct_cond_anchor_splitfirst_3way_armchain_pe32h512_20260227.json` | required | current newflow entry stage |
| `70a` | `config/posttrain_WalkF_stage7_70a_splitB2_pe32h512_20260227_fromarmchain.json` | required | last plain upstream stage |
| `new70b_replace_lowdrift` | generated from the `70b_phasezin` base semantics with low-drift overrides | required | real operational replace handoff |
| raw `70b` | `config/posttrain_WalkF_stage7_70b_phasezin_splitB2_pe32h512_20260227_fromarmchain.json` | diagnostic-only | do not use as downstream handoff |
| `70R` | promoted nonleg-recovery stage from low-drift replace | required | current winning handoff into downstream |
| `71(lr=3e-4)` | base `71` semantics, lower LR only | required | fixes the old early overshoot in `71` |
| `72(lr=1e-4)` | base `72` semantics, lower LR only | required | fixes the old early overshoot in `72` |
| `lambda` | `config/posttrain_WalkF_stage7_lambda_final_calib_20260227_fromarmchain_fullcompat.json` | required | preserves the `72(lr=1e-4)` result |

### 3.2 Stage semantics

`new70b_replace_lowdrift`

- semantic base: `config/posttrain_WalkF_stage7_70b_phasezin_splitB2_pe32h512_20260227_fromarmchain.json`
- active overrides:
  - `direct_pose_use_phase_z=true`
  - `direct_pose_phase_z_mode=replace_contacts`
  - `lr=3e-4`
  - `epochs=1`
  - `steps_per_epoch=60`
- build it from the `70a` replace-zerophase warmstart, not from raw `70b`

`70R`

- semantic base: `config/posttrain_WalkF_stage7_70R_nonleg_recovery_proj256_preleg_20260227_fromarmchain.json`
- current promote recipe:
  - `tools/run_posttrain_nonleg_trunk_ablation.py`
  - `--trunk-mode full`
  - `--epochs 1`
  - `--steps-per-epoch 180`
- input is `new70b_replace_lowdrift`

`71(lr=3e-4)`

- semantic base: `config/posttrain_WalkF_stage7_71_legonly_after_nonlegproj256_20260227_fromarmchain.json`
- active override:
  - `lr=3e-4`
- keep the stage semantics unchanged; the fix is smaller step size, not redesign
- keep dense step checkpoints for attribution / replay:
  - `0,5,10,20,40,60,120,180`

`72(lr=1e-4)`

- semantic base: `config/posttrain_WalkF_stage7_72_legomega_after_nonlegproj256_20260227_fromarmchain.json`
- active override:
  - `lr=1e-4`
- keep the stage semantics unchanged; the fix is smaller step size, not redesign
- keep dense step checkpoints for attribution / replay:
  - `0,5,10,20,40,60,120,180`

`lambda`

- semantic base: `config/posttrain_WalkF_stage7_lambda_final_calib_20260227_fromarmchain_fullcompat.json`
- no new semantics are introduced here
- on the current winning lane, `lambda` is a direct-metric no-op relative to `72(lr=1e-4)`

---

## 4) Why This Is the Canonical Chain

The active verdict is now stable:

1. raw `70b` is not the real handoff
   - the operational handoff is `70a -> new70b_replace_lowdrift`
2. low-drift replace is the right upstream stage choice
   - it gives a much cleaner start into `70R`
3. unchanged `71` was over-stepping the cleaner `70R` start
   - `71(lr=3e-4)` fixes that without changing objective semantics
4. unchanged `72` had the same problem even more sharply
   - `72(lr=1e-4)` fixes that and becomes the best tested `72`
5. `lambda` does not give the win back
   - the `72(lr=1e-4)` gains are preserved to final `lambda`

Current winning endpoint (`model-source`):

| lane | all_ex_root | leg | nonleg | arm | foot_l/ball_l@SIC12-15 | calf_r@SIC2-4 |
|---|---:|---:|---:|---:|---:|---:|
| current `lambda` | 0.112074 | 0.296389 | 0.072222 | 0.082665 | 0.812663 | 0.288880 |
| candidate `lambda` | 0.101969 | 0.186385 | 0.083717 | 0.091849 | 0.385267 | 0.042300 |

Practical read:

- aggregate leg-side quality is clearly better on the current candidate chain
- `nonleg/arm` remain slightly higher, but this is not blocking the current flow decision
- the active default therefore moves to the new chain above

---

## 5) Preferred Reproduction Path

Use the recorded automation runners instead of reconstructing ad hoc one-off commands.
This keeps the generated configs and checkpoint handoffs aligned with the accepted experiments.

### 5.1 Upstream lane lock

1. establish / replay the old d1 newflow reference lane
   - `tools/run_oldd1_newflow_chain.py`
2. build and compare the low-drift replace candidate
   - `tools/run_oldd1_skip70b_replace_compare.py`
3. continue low-drift replace through `70R -> 71`
   - `tools/run_oldd1_skip70b_lowdrift_to71.py`

### 5.2 Downstream winning choices

4. run the `71` lower-LR sweep and take `lr=3e-4`
   - `tools/run_71_lowlr_sweep.py`
5. run the `72` lower-LR sweep and take `lr=1e-4`
   - `tools/run_72_lowlr_sweep.py`
6. continue the winning `72` to final `lambda`
   - `tools/run_72_lowlr_to_lambda.py`

### 5.3 Reference generated configs and checkpoints

These are the locked reference artifacts for the current canonical line:

- low-drift replace config:
  - `debug_output/_tmp_oldd1_skip70b_lowdrift_20260314/configs/posttrain_70b_replace_lowdrift_from_oldd1_20260314.json`
- low-drift replace ckpt:
  - `models/__tmp_oldd1_skip70b_lowdrift_20260314/70b_replace_lowdrift/ckpt_last_WalkF_stage7_70b_replace_lowdrift_from_oldd1_20260314.pth`
- promoted `70R` ckpt:
  - `models/__tmp_oldd1_skip70b_lowdrift_to71_20260314/70R/ckpt_last_WalkF_stage7_70R_from_oldd1_lowdrift_replace_20260314.pth`
- winning `71(lr=3e-4)` config:
  - `debug_output/_tmp_71_lowlr_sweep_20260314/configs/posttrain_71_lr3e4_20260314.json`
- winning `71(lr=3e-4)` ckpt:
  - `models/__tmp_71_lowlr_sweep_20260314/lr3e4/ckpt_last_WalkF_stage7_71_lr3e4_from_candidate70R_20260314.pth`
- winning `72(lr=1e-4)` config:
  - `debug_output/_tmp_72_lowlr_sweep_20260314/configs/posttrain_72_lr1e4_20260314.json`
- winning `72(lr=1e-4)` ckpt:
  - `models/__tmp_72_lowlr_sweep_20260314/lr1e4/ckpt_last_WalkF_stage7_72_lr1e4_from_lowlr71_20260314.pth`
- final `lambda` ckpt:
  - `models/__tmp_72_lowlr_to_lambda_20260315/lambda/ckpt_last_WalkF_stage7_lambda_from_lowlr72lr1e4_20260315.pth`

---

## 6) Validation Checklist

After finishing the chain:

1. run the static no-legacy checks

```bash
if rg -n "direct_pose_hinge|direct_hinge_delta|contact_phase_state|contact_meas_provider" \
  train/posttrain.py train/models.py train/training_MPL.py train/eval_utils.py train/validate/run_freerun_cycles.py; then
  echo "[FAIL] legacy references found"
  exit 1
fi
python3 -m py_compile \
  train/posttrain.py train/models.py train/training_MPL.py \
  train/eval_utils.py train/validate/run_freerun_cycles.py
python3 tools/check_posttrain_newflow_active_configs.py
python3 tools/check_posttrain_legacy_code_guard.py
```

2. evaluate the final ckpt with the locked contract

```bash
ENCODER_BUNDLE=models/motion_encoder_equiv.pt.best.pt
AFFINE_STATS=debug_output/_tmp_phaseb_affine_20260304/affine_fit_mix08/affine_stats.json

PYTHONPATH=. python -m train.validate.run_freerun_cycles \
  --teacher validate/teacher_batches/Walk_F_teacher.json \
  --model <ckpt> \
  --rounds 5 \
  --time-index-mode cycle \
  --depth 3 \
  --event_clock auto \
  --phase_reset_source none \
  --contacts_meas_source model \
  --lambda_fusion_apply \
  --log_contacts \
  --export_direct_arm_probe \
  --export_joint_direct_geolocal_series
```

3. if you want the archive strict read, run the same eval with `pretrain_contact`

```bash
PYTHONPATH=. python -m train.validate.run_freerun_cycles \
  --teacher validate/teacher_batches/Walk_F_teacher.json \
  --model <ckpt> \
  --rounds 5 \
  --time-index-mode cycle \
  --depth 3 \
  --event_clock auto \
  --phase_reset_source none \
  --contacts_meas_source pretrain_contact \
  --contacts_meas_pretrain_clamp 1.0 \
  --contacts_meas_pretrain_affine_stats "${AFFINE_STATS}" \
  --encoder-bundle "${ENCODER_BUNDLE}" \
  --lambda_fusion_apply \
  --log_contacts \
  --export_direct_arm_probe \
  --export_joint_direct_geolocal_series
```

Primary metrics to read first:

- `all_ex_root`
- `leg`
- `legs_main`
- `foot_l/ball_l@SIC12-15`
- `calf_r@SIC2-4`

---

## 7) Explicitly Not Mainline

Do not put these back into the pipeline document unless a newer problem record re-activates them:

- raw `70b` as a required handoff stage
- historical `70c_replacecontacts` as an active stage name
- plain old `70R -> 71 -> 72 -> lambda` without the lower-LR fixes
- `71m`
- `72_micro`
- `hybridcarry`
- `skip70c`
- `whitebox` contacts as default posttrain route
- legacy Stage1-5 guidance in this file

If any of those need to be revisited, document them in a problem note first, then decide whether they re-enter this pipeline doc.
