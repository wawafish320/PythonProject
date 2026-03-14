# 2026-03-14 old d1 -> new posttrain flow handoff

## Goal

Lock the current meaning of this lane so later A/B work can focus on the leg regression instead of re-discovering the chain shape.

This document is for the lane:

- old d1-style basetrain
- current new Stage6 / new downstream flow
- no Stage5.5
- no BCE
- no `rollback_planner_core` surgery
- no frontload / interface / phase_z side study
- accepted final stays the main baseline

The question of this round was:

- if we start from the re-trained old d1 basetrain and run the current new posttrain flow directly,
- what does the operational `Stage6 -> 70a -> new70b_replace -> 70R -> 71 -> 72 -> lambda final` chain look like,
- how much of raw `70b` is real handoff vs diagnostic-only artifact,
- and where exactly does leg improve or regress.

---

## Artifacts

- automation: `tools/run_oldd1_newflow_chain.py`
- machine summary: `debug_output/_tmp_oldd1_newflow_chain_20260314/summary.json`
- readable summary: `debug_output/_tmp_oldd1_newflow_chain_20260314/summary.md`

Base checkpoint:

- `models/MLPL2_DirectBranch_v1/exp_phase_DirectBranch_v1_d1/ckpt_best_free_exp_phase_DirectBranch_v1_d1.pth`

Locked posttrain contract:

- `--posttrain_contacts_source pretrain_contact`
- `--posttrain_contacts_pretrain_clamp 1.0`
- `--encoder_bundle models/motion_encoder_equiv.pt.best.pt`
- `--posttrain_contacts_pretrain_affine_stats debug_output/_tmp_phaseb_affine_20260304/affine_fit_mix08/affine_stats.json`

Locked final eval contracts:

- strict: `contacts_meas_source=pretrain_contact`
- accepted-aligned: `contacts_meas_source=model`

---

## Current canonical chain

Important wording lock:

- operational handoff chain:
  - `Stage6 -> 70a -> new70b_replace -> 70R -> 71 -> 72 -> lambda final`
- diagnostic-only side artifact:
  - raw `70b`
- for this lane, `new70b_replace` is built from `70a` warmstart, not from raw `70b`

Stage checkpoints:

- Stage6:
  - ckpt: `models/__tmp_stage6_basetrain_compare_20260313/old_bestfree/ckpt_last_old_bestfree_stage6_cmp_20260313.pth`
  - source config: `config/posttrain_WalkF_stage6_direct_cond_anchor_splitfirst_3way_armchain_pe32h512_20260227.json`
- 70a:
  - ckpt: `models/__tmp_oldd1_newflow_chain_20260314/70a/ckpt_last_WalkF_stage7_70a_from_oldd1_newflow_20260314.pth`
  - config: `config/posttrain_WalkF_stage7_70a_splitB2_pe32h512_20260227_fromarmchain.json`
- 70b diagnostic-only:
  - ckpt: `models/__tmp_oldd1_newflow_chain_20260314/70b/ckpt_last_WalkF_stage7_70b_from_oldd1_newflow_20260314.pth`
  - config: `config/posttrain_WalkF_stage7_70b_phasezin_splitB2_pe32h512_20260227_fromarmchain.json`
- 70a replace zerophase warmstart:
  - ckpt: `models/__tmp_oldd1_newflow_chain_20260314/warmstart/ckpt_last_oldd1_70a_replacecontacts_zerophase_20260314.pth`
- new70b_replace:
  - ckpt: `models/__tmp_oldd1_newflow_chain_20260314/70b_replace/ckpt_last_WalkF_stage7_70b_replace_from_oldd1_newflow_20260314.pth`
  - generated config: `debug_output/_tmp_oldd1_newflow_chain_20260314/configs/posttrain_70b_replacecontacts_from_oldd1_20260314.json`
- 70R:
  - ckpt: `models/__tmp_oldd1_newflow_chain_20260314/70R/ckpt_last_WalkF_stage7_70R_from_oldd1_newflow_s180_20260314.pth`
  - generated config: `debug_output/_tmp_oldd1_newflow_chain_20260314/configs/posttrain_70R_from_oldd1_replace_lr3e4_e1_s60_20260314.json`
  - promote mode: `tools/run_posttrain_nonleg_trunk_ablation.py --trunk-mode full --epochs 1 --steps-per-epoch 180`
- 71:
  - ckpt: `models/__tmp_oldd1_newflow_chain_20260314/71/ckpt_last_WalkF_stage7_71_from_oldd1_newflow_20260314.pth`
  - config: `config/posttrain_WalkF_stage7_71_legonly_after_nonlegproj256_20260227_fromarmchain.json`
- 72:
  - ckpt: `models/__tmp_oldd1_newflow_chain_20260314/72/ckpt_last_WalkF_stage7_72_from_oldd1_newflow_20260314.pth`
  - config: `config/posttrain_WalkF_stage7_72_legomega_after_nonlegproj256_20260227_fromarmchain.json`
- lambda final:
  - ckpt: `models/__tmp_oldd1_newflow_chain_20260314/lambda/ckpt_last_WalkF_stage7_lambda_from_oldd1_newflow_20260314.pth`
  - config: `config/posttrain_WalkF_stage7_lambda_final_calib_20260227_fromarmchain_fullcompat.json`

---

## Stage-by-stage readout

Important note:

- For `Stage6 -> 72`, the useful stage-progress signal is the direct-path group metrics.
- `BlendGeoLocalDeg` / `GeoLocalDeg` stay pinned near `60.282111` before lambda, so they are not useful for stage-to-stage root-cause reading until `lambda final`.

### Model-source stage progress

| stage | DirectGeoLocalDeg | all_ex_root | leg | nonleg | arm | else |
|---|---:|---:|---:|---:|---:|---:|
| Stage6 | 0.315735 | 0.315735 | 0.865450 | 0.196877 | 0.222863 | 0.135457 |
| 70a | 0.275083 | 0.275083 | 0.730911 | 0.176525 | 0.203549 | 0.112650 |
| 70b | 0.308443 | 0.308443 | 0.730643 | 0.217157 | 0.254408 | 0.129109 |
| new70b_replace | 0.280736 | 0.280736 | 0.662440 | 0.198205 | 0.226846 | 0.130508 |
| 70R | 0.158235 | 0.158235 | 0.556049 | 0.072222 | 0.082665 | 0.047537 |
| 71 | 0.111911 | 0.111911 | 0.295473 | 0.072222 | 0.082665 | 0.047537 |
| 72 | 0.112074 | 0.112074 | 0.296389 | 0.072222 | 0.082665 | 0.047537 |
| lambda | 0.112074 | 0.112074 | 0.296389 | 0.072222 | 0.082665 | 0.047537 |

### Transition deltas

Negative is better, positive is worse.

| transition | d_all_ex_root | d_leg | d_nonleg | d_arm | d_else |
|---|---:|---:|---:|---:|---:|
| Stage6 -> 70a | -0.040652 | -0.134539 | -0.020352 | -0.019313 | -0.022808 |
| 70a -> 70b | +0.033361 | -0.000268 | +0.040632 | +0.050859 | +0.016459 |
| 70b -> new70b_replace | -0.027708 | -0.068204 | -0.018952 | -0.027562 | +0.001399 |
| new70b_replace -> 70R | -0.122500 | -0.106391 | -0.125984 | -0.144181 | -0.082970 |
| 70R -> 71 | -0.046325 | -0.260576 | +0.000000 | +0.000000 | +0.000000 |
| 71 -> 72 | +0.000163 | +0.000916 | +0.000000 | +0.000000 | +0.000000 |
| 72 -> lambda | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |

### Per-stage interpretation

- Stage6:
  - clear weak point versus the control lanes
  - bad enough on leg that the lane starts behind before Stage7 even begins
- 70a:
  - good cleanup step
  - reduces both leg and nonleg
- 70b:
  - diagnostic-only stage for this lane
  - first clear regression point
  - leg is flat, but nonleg and arm get worse
- new70b_replace:
  - operational replace stage built directly from `70a` warmstart
  - partial recovery relative to raw `70b`
  - compared with `70a`, it improves leg but worsens `all_ex_root / nonleg / arm`
  - this is now the main stage to target if we want to explain the leg/nonleg tradeoff
- 70R:
  - biggest direct-path cleanup step in the whole chain
  - especially important for `all_ex_root` and `nonleg`
- 71:
  - biggest leg-specific cleanup step
  - this is where leg gets pulled down from `0.556049` to `0.295473`
- 72:
  - basically neutral for this lane
  - almost no useful movement, slight leg regression
- lambda:
  - does not change direct-path quality anymore
  - only restores the blend/global metrics into the final usable range

---

## Final result

### Final model-source

- `DirectGeoLocalDeg=0.112074`
- `BlendGeoLocalDeg=0.111467`
- `GeoLocalDeg=0.534542`
- direct group:
  - `all_ex_root=0.112074`
  - `leg=0.296389`
  - `nonleg=0.072222`
  - `arm=0.082665`
  - `else=0.047537`

### Final strict

- `DirectGeoLocalDeg=0.111971`
- `BlendGeoLocalDeg=0.111156`
- `GeoLocalDeg=0.530454`
- direct group:
  - `all_ex_root=0.111971`
  - `leg=0.293528`
  - `nonleg=0.072716`
  - `arm=0.083060`
  - `else=0.048265`

### Final direct windows / hotspots

Model-source:

- `legs_main=0.296389`
- `arms_main=0.082665`
- `A_52_59 legs_main=0.427246 arms_main=0.071218`
- `B_76_80 legs_main=0.225787 arms_main=0.090594`
- `foot_l/ball_l@SIC12-15=0.812663`
- `calf_r@SIC2-4=0.288880`

Strict:

- `legs_main=0.293528`
- `arms_main=0.083060`
- `A_52_59 legs_main=0.400387 arms_main=0.071439`
- `B_76_80 legs_main=0.217172 arms_main=0.101627`
- `foot_l/ball_l@SIC12-15=0.785841`
- `calf_r@SIC2-4=0.272596`

---

## Comparison against the three control lanes

### A. Current accepted final anchor

Accepted anchor model-source:

- `all_ex_root=0.112947`
- `leg=0.274360`
- `nonleg=0.078048`

This lane model-source delta vs A:

- `all_ex_root=-0.000873`
- `leg=+0.022029`
- `nonleg=-0.005826`

Interpretation:

- slightly better on overall direct and nonleg
- clearly worse on leg
- therefore not a clean win over the current accepted anchor

### B. Full oldplan downstream chain

Full oldplan final:

- model-source:
  - `all_ex_root=0.120145`
  - `leg=0.278087`
  - `nonleg=0.085995`
- strict:
  - `all_ex_root=0.117883`
  - `leg=0.280194`
  - `nonleg=0.082789`

This lane vs B:

- model-source:
  - `all_ex_root=-0.008071`
  - `leg=+0.018302`
  - `nonleg=-0.013774`
- strict:
  - `all_ex_root=-0.005912`
  - `leg=+0.013334`
  - `nonleg=-0.010073`

Interpretation:

- better on overall direct and nonleg
- worse on leg
- still mixed, not a clean replacement

### C. rollback_planner_core challenger

Rollback planner core:

- Stage6:
  - `all_ex_root=0.305250`
  - `leg=0.766829`
  - `nonleg=0.205449`
- final model-source:
  - `all_ex_root=0.114635`
  - `leg=0.296311`
  - `nonleg=0.075354`
- final strict:
  - `all_ex_root=0.114862`
  - `leg=0.286085`
  - `nonleg=0.077841`

This lane vs C:

- Stage6:
  - `all_ex_root=+0.008029`
  - `leg=+0.107401`
  - `nonleg=-0.013456`
- final model-source:
  - `all_ex_root=-0.002562`
  - `leg=+0.000078`
  - `nonleg=-0.003132`
- final strict:
  - `all_ex_root=-0.002891`
  - `leg=+0.007443`
  - `nonleg=-0.005125`

Interpretation:

- it recovers enough downstream to beat or match C on overall direct / nonleg
- but leg still does not produce a clean win

### strict vs model-source consistency

For both B and C:

- strict conclusion and model-source conclusion are aligned
- both stay "mixed"

For A:

- only model-source is same-contract
- the repo does not archive an accepted-final strict snapshot for same-contract comparison

---

## What is now clear

The full posttrain flow is now clear enough that future leg-focused A/B work should not start from scratch.

The main takeaways are:

1. The lane is real and reproducible.
2. The lane is worth keeping as a challenger lane.
3. It is not ready for baseline / promote discussion.
4. The main unresolved issue is leg, not nonleg.
5. The useful root-cause search space is now much narrower:
   - Stage6 starts behind on leg
   - 70b adds regression
   - 70R fixes overall/nonleg
   - 71 fixes most of the leg gap
   - 72 and lambda do almost nothing for direct leg quality

This means:

- if the goal is to explain leg regression, the highest-value comparisons are not `72` or `lambda`
- the highest-value comparisons are:
  - `Stage6`
  - `70a`
  - `70b`
  - `new70b_replace`
  - `70R`
  - `71`

---

## Recommended A/B plan for leg regression

### Priority order

1. Stage6 startline
   - because this lane is already behind the full oldplan / rollback challenger on leg at Stage6
   - first question: why does old d1 basetrain enter Stage6 with weaker leg carry under the new flow

2. 70a / 70b / new70b_replace
   - because raw `70b` is the first obvious regression but is only diagnostic
   - the real optimization handoff is `70a -> new70b_replace`
   - second question: what in the replace handoff is buying leg while leaking into nonleg / arm

Update (`2026-03-14 PM`):

- first conservative replace candidate has already been run:
  - doc: `docs/Problems/active/2026-03-14_oldd1_skip70b_replace_lowdrift_experiment.md`
  - result: stage-level direct metrics improve very strongly vs the current `new70b_replace`
  - but `calf_r@SIC2-4` worsens sharply
- so the current leg root-cause framing becomes:
  - current replace stage likely overshoots broadly
  - but the replacement candidate may still create a narrower calf-local instability that downstream must be checked

3. 70R -> 71 boundary
   - because `70R` is the largest overall cleanup and `71` is the largest leg cleanup
   - third question: why does 71 recover so much leg but still stop short of the accepted anchor

4. 72 / lambda only after the above
   - these are not the main root-cause stages for leg in this lane
   - use them only after the upstream leg signal is already better

### Suggested fixed-lane A/B protocol

When testing a leg hypothesis:

- keep the same base ckpt:
  - `models/MLPL2_DirectBranch_v1/exp_phase_DirectBranch_v1_d1/ckpt_best_free_exp_phase_DirectBranch_v1_d1.pth`
- keep the same contract:
  - `pretrain_contact + clamp1 + affine_mix08 + encoder_bundle`
- keep the same final eval pair:
  - strict `pretrain_contact`
  - accepted-aligned `model`
- always compare against:
  - A accepted final anchor
  - B full oldplan chain
  - C rollback planner core challenger

### Metrics to watch first for leg work

Primary:

- `all_ex_root`
- `leg`
- `nonleg`
- `DirectGeoLocalDeg`

Secondary:

- `legs_main`
- `A_52_59`
- `B_76_80`
- `foot_l/ball_l@SIC12-15`
- `calf_r@SIC2-4`

Interpretation rule:

- a leg-targeted A/B should not be accepted only because `all_ex_root` improves
- it should also improve or at least not worsen:
  - `leg`
  - `legs_main`
  - `foot_l/ball_l@SIC12-15`
  - `calf_r@SIC2-4`

---

## Current verdict

- keep this lane as a challenger lane
- do not discuss baseline switch or promote yet
- focus next work on the leg regression cause inside the current chain, not on re-opening retired directions
