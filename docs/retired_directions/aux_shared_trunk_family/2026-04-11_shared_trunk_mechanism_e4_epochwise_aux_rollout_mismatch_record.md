# 2026-04-11 shared trunk mechanism E4 epochwise aux-rollout mismatch record

> Status: archived / retired aux-family mechanism record
> Reader note: this aux / shared-trunk family did **not** become current repo mainline; any `recommend`, `default`, `ship`, `mainline`, or `current` wording below is historical family-local language only.
> Current entry points: `docs/posttrain_pipeline.md`, `docs/train_design/2026-04-11_donor_contract_basin_universal_debug_synthesis.md`, `docs/retired_directions/aux_shared_trunk_family/2026-04-11_aux_detach_downstream_reevaluation_record.md`

> Status: completed  
> Scope: `stage6 native` only; observational confirmation only; no objective redesign, no aux-weight sweep, no `70a/70b`  
> Result: **mixed / negative for the strong within-arm hypothesis**; clear late-phase mismatch only in `shared_attach_aux`, plus strong final cross-arm mismatch

## 1. Fixed question

This round only asked:

> when `aux_leg_loss(epoch)` keeps going down, does `freerun leg(epoch)` get worse, or at least fail to improve?

Required arms:

1. `shared_attach_aux`
2. `aux_detach`
3. `late_attach_aux`

Per user instruction, this round reuses prior readings and does **not** re-argue E1/E2/E3:

- `docs/retired_directions/aux_shared_trunk_family/2026-04-10_shared_trunk_mechanism_e1_aux_detach_record.md`
- `docs/retired_directions/aux_shared_trunk_family/2026-04-10_shared_trunk_mechanism_e2_frozen_trunk_aux_readability_record.md`
- `docs/retired_directions/aux_shared_trunk_family/2026-04-10_shared_trunk_mechanism_e3_late_attach_probe_record.md`

## 2. Pre-checks

## 2.1 Artifact check

Checked:

- `models/__tmp_dsn_aux_leg_matched_chain_20260410/stage6/aux`
- `models/__tmp_dsn_aux_leg_matched_chain_20260410/stage6/aux_detach`
- `models/__tmp_dsn_aux_leg_matched_chain_20260410/stage6/late_attach_aux`

Result:

- only final ckpts + train logs were present
- no reusable `ckpt_step_*` / `ckpt_epoch_*` snapshots existed

So a rerun was required.

## 2.2 Raw-train-ckpt freerun smoke

Smoke command:

```bash
PYTHONPATH=. debug_output/_tmp_phasecd_min_ablation_20260330/cpu_nomps_exec.py -m train.validate.run_freerun_cycles \
  --teacher validate/teacher_batches/Walk_F_teacher.json \
  --model models/__tmp_dsn_aux_leg_matched_chain_20260410/stage6/aux/ckpt_last_train_lr3e4_e8x60_wd1e4_reinit1_stage6_tailfix_dsn_auxw02_20260410.pth \
  --rounds 1 --depth 1 --time-index-mode cycle \
  --phase_reset_source none \
  --contacts_meas_source pretrain_contact \
  --contacts_meas_pretrain_clamp 1.0 \
  --contacts_meas_pretrain_affine_stats debug_output/_tmp_phaseb_affine_20260304/affine_fit_mix08/affine_stats.json \
  --encoder-bundle models/motion_encoder_equiv.pt.best.pt \
  --out debug_output/_tmp_dsn_aux_leg_matched_chain_20260411/smoke_raw_aux_train_ckpt \
  --force
```

Result:

- passed
- `run_freerun_cycles` can load raw posttrain train ckpts with `posttrain_cfg`
- no blocker; no new export feature was needed

Smoke artifact:

- `debug_output/_tmp_dsn_aux_leg_matched_chain_20260411/smoke_raw_aux_train_ckpt/Walk_F_freerun_cycles.json`

## 3. Minimal rerun with epoch snapshots

Because no snapshots existed, reran the 3 requested arms with the **same stage6 native recipe** and only added:

- `save_step_ckpts=0,60,120,180,240,300,360,420,480`

Generated matched configs:

- `debug_output/_tmp_dsn_aux_leg_matched_chain_20260411/configs/shared_attach_aux_epochsnap.json`
- `debug_output/_tmp_dsn_aux_leg_matched_chain_20260411/configs/aux_detach_epochsnap.json`
- `debug_output/_tmp_dsn_aux_leg_matched_chain_20260411/configs/late_attach_aux_epochsnap.json`

Rerun model roots:

- `models/__tmp_dsn_aux_leg_matched_chain_20260411/stage6/shared_attach_aux_epochsnap`
- `models/__tmp_dsn_aux_leg_matched_chain_20260411/stage6/aux_detach_epochsnap`
- `models/__tmp_dsn_aux_leg_matched_chain_20260411/stage6/late_attach_aux_epochsnap`

Historical-match sanity check:

- `debug_output/_tmp_dsn_aux_leg_matched_chain_20260411/e4_epochwise/e4_rerun_vs_historical_final.json`
- all 3 reruns reproduced the historical epoch-8 final summaries exactly (`delta = 0.0` for `leg mean`, `leg p95`, `all_ex_root mean`, `all_ex_root p95`)

So the reruns are matched, not a new recipe drift.

## 4. Actual commands run

## 4.1 Train reruns

### `shared_attach_aux`

```bash
PYTHONPATH=. debug_output/_tmp_phasecd_min_ablation_20260330/cpu_nomps_exec.py -m train.posttrain \
  --config debug_output/_tmp_dsn_aux_leg_matched_chain_20260411/configs/shared_attach_aux_epochsnap.json \
  --ckpt_in models/cp015_phasecd_tailk_probe_20260331/exp_phase_DirectBranch_v1_d1_cp015_tailk7_rankmix_tw020_corridor_hold_tail15_phasea050_fixedsched_ep014center_control_denseckpt_seed2024_20260401/ckpt_epoch_014.pth \
  --out_dir models/__tmp_dsn_aux_leg_matched_chain_20260411/stage6/shared_attach_aux_epochsnap \
  --run_name lr3e4_e8x60_wd1e4_reinit1_stage6_tailfix_dsn_sharedaux_epochsnap_20260411 \
  --posttrain_contacts_source pretrain_contact \
  --posttrain_contacts_pretrain_clamp 1.0 \
  --encoder_bundle models/motion_encoder_equiv.pt.best.pt \
  --posttrain_contacts_pretrain_affine_stats debug_output/_tmp_phaseb_affine_20260304/affine_fit_mix08/affine_stats.json
```

### `aux_detach`

```bash
PYTHONPATH=. debug_output/_tmp_phasecd_min_ablation_20260330/cpu_nomps_exec.py -m train.posttrain \
  --config debug_output/_tmp_dsn_aux_leg_matched_chain_20260411/configs/aux_detach_epochsnap.json \
  --ckpt_in models/cp015_phasecd_tailk_probe_20260331/exp_phase_DirectBranch_v1_d1_cp015_tailk7_rankmix_tw020_corridor_hold_tail15_phasea050_fixedsched_ep014center_control_denseckpt_seed2024_20260401/ckpt_epoch_014.pth \
  --out_dir models/__tmp_dsn_aux_leg_matched_chain_20260411/stage6/aux_detach_epochsnap \
  --run_name lr3e4_e8x60_wd1e4_reinit1_stage6_tailfix_dsn_auxdetach_epochsnap_20260411 \
  --posttrain_contacts_source pretrain_contact \
  --posttrain_contacts_pretrain_clamp 1.0 \
  --encoder_bundle models/motion_encoder_equiv.pt.best.pt \
  --posttrain_contacts_pretrain_affine_stats debug_output/_tmp_phaseb_affine_20260304/affine_fit_mix08/affine_stats.json
```

### `late_attach_aux`

```bash
PYTHONPATH=. debug_output/_tmp_phasecd_min_ablation_20260330/cpu_nomps_exec.py -m train.posttrain \
  --config debug_output/_tmp_dsn_aux_leg_matched_chain_20260411/configs/late_attach_aux_epochsnap.json \
  --ckpt_in models/cp015_phasecd_tailk_probe_20260331/exp_phase_DirectBranch_v1_d1_cp015_tailk7_rankmix_tw020_corridor_hold_tail15_phasea050_fixedsched_ep014center_control_denseckpt_seed2024_20260401/ckpt_epoch_014.pth \
  --out_dir models/__tmp_dsn_aux_leg_matched_chain_20260411/stage6/late_attach_aux_epochsnap \
  --run_name lr3e4_e8x60_wd1e4_reinit1_stage6_tailfix_dsn_lateattach_aux_epochsnap_20260411 \
  --posttrain_contacts_source pretrain_contact \
  --posttrain_contacts_pretrain_clamp 1.0 \
  --encoder_bundle models/motion_encoder_equiv.pt.best.pt \
  --posttrain_contacts_pretrain_affine_stats debug_output/_tmp_phaseb_affine_20260304/affine_fit_mix08/affine_stats.json
```

## 4.2 Batch epochwise freerun eval + aggregation

Analysis helper added under debug-only path:

- `debug_output/_tmp_dsn_aux_leg_matched_chain_20260411/e4_epochwise_aux_rollout_analysis.py`

Executed:

```bash
PYTHONPATH=. python3 debug_output/_tmp_dsn_aux_leg_matched_chain_20260411/e4_epochwise_aux_rollout_analysis.py --run-evals
```

This script ran the same fixed eval recipe for each epoch snapshot (`1..8`) of each arm:

- `teacher = validate/teacher_batches/Walk_F_teacher.json`
- `rounds = 5`
- `depth = 3`
- `time-index-mode = cycle`
- `phase_reset_source = none`
- `contacts_meas_source = pretrain_contact`
- `contacts_meas_pretrain_clamp = 1.0`
- `contacts_meas_pretrain_affine_stats = debug_output/_tmp_phaseb_affine_20260304/affine_fit_mix08/affine_stats.json`
- `encoder-bundle = models/motion_encoder_equiv.pt.best.pt`

and then ran `tools/phasea_group_summary.py --cycle_gte 1 --drop_wrap`.

## 5. Output artifacts

Primary outputs:

- table csv: `debug_output/_tmp_dsn_aux_leg_matched_chain_20260411/e4_epochwise/e4_epochwise_table.csv`
- combined json: `debug_output/_tmp_dsn_aux_leg_matched_chain_20260411/e4_epochwise/e4_epochwise_metrics.json`
- correlation json: `debug_output/_tmp_dsn_aux_leg_matched_chain_20260411/e4_epochwise/e4_correlations.json`
- rerun-vs-history check: `debug_output/_tmp_dsn_aux_leg_matched_chain_20260411/e4_epochwise/e4_rerun_vs_historical_final.json`
- plot: `debug_output/_tmp_dsn_aux_leg_matched_chain_20260411/e4_epochwise/e4_epochwise_aux_loss_vs_leg_p95.png`

Per-epoch group summaries live under:

- `debug_output/_tmp_dsn_aux_leg_matched_chain_20260411/e4_epochwise/shared_attach_aux`
- `debug_output/_tmp_dsn_aux_leg_matched_chain_20260411/e4_epochwise/aux_detach`
- `debug_output/_tmp_dsn_aux_leg_matched_chain_20260411/e4_epochwise/late_attach_aux`

## 6. Epochwise results

Current `phasea_group_summary` uses the same summary value for `DirectGeoLocalDeg` and `all_ex_root`, so only `all_ex_root` is reported below.

| arm | epoch | aux_leg_loss | aux_leg_over_main | leg mean | leg p95 | all_ex_root mean | all_ex_root p95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `shared_attach_aux` | `1` | `0.150498` | `0.103429` | `3.600986` | `15.093977` | `1.726197` | `6.706787` |
| `shared_attach_aux` | `2` | `0.110540` | `0.080070` | `1.604378` | `5.140162` | `0.731995` | `2.746501` |
| `shared_attach_aux` | `3` | `0.090446` | `0.055230` | `1.092974` | `3.082576` | `0.505677` | `1.938164` |
| `shared_attach_aux` | `4` | `0.078847` | `0.044641` | `0.767092` | `2.062629` | `0.352334` | `1.195815` |
| `shared_attach_aux` | `5` | `0.072849` | `0.039018` | `0.872817` | `1.896572` | `0.366850` | `1.352452` |
| `shared_attach_aux` | `6` | `0.065077` | `0.034288` | `0.709040` | `1.619963` | `0.314729` | `1.103781` |
| `shared_attach_aux` | `7` | `0.059068` | `0.030329` | `0.657603` | `1.488769` | `0.260299` | `0.938242` |
| `shared_attach_aux` | `8` | `0.054306` | `0.028285` | `0.708246` | `1.793149` | `0.268797` | `0.990646` |
| `aux_detach` | `1` | `0.150569` | `0.103462` | `3.596327` | `14.975139` | `1.726933` | `6.687765` |
| `aux_detach` | `2` | `0.111062` | `0.080457` | `1.651206` | `4.744741` | `0.757271` | `2.872584` |
| `aux_detach` | `3` | `0.090922` | `0.055654` | `1.261911` | `3.423264` | `0.523502` | `1.985348` |
| `aux_detach` | `4` | `0.079232` | `0.044932` | `0.931860` | `2.321500` | `0.399609` | `1.404936` |
| `aux_detach` | `5` | `0.073228` | `0.039770` | `0.831660` | `1.844991` | `0.345223` | `1.263865` |
| `aux_detach` | `6` | `0.065269` | `0.033941` | `0.659349` | `1.508423` | `0.290791` | `1.024933` |
| `aux_detach` | `7` | `0.059939` | `0.030789` | `0.788365` | `1.953524` | `0.289769` | `1.118872` |
| `aux_detach` | `8` | `0.054265` | `0.028465` | `0.597226` | `1.316565` | `0.262125` | `0.937262` |
| `late_attach_aux` | `1` | `0.152949` | `0.105151` | `3.598332` | `15.120994` | `1.730029` | `6.716115` |
| `late_attach_aux` | `2` | `0.123932` | `0.089976` | `1.649833` | `4.986226` | `0.760121` | `2.934097` |
| `late_attach_aux` | `3` | `0.115282` | `0.070670` | `1.189836` | `3.082560` | `0.516731` | `1.952092` |
| `late_attach_aux` | `4` | `0.107628` | `0.061394` | `0.916311` | `2.218812` | `0.412649` | `1.503497` |
| `late_attach_aux` | `5` | `0.105732` | `0.056322` | `0.718795` | `1.653992` | `0.322584` | `1.145269` |
| `late_attach_aux` | `6` | `0.101599` | `0.053600` | `0.755329` | `1.953848` | `0.342059` | `1.222086` |
| `late_attach_aux` | `7` | `0.096982` | `0.051130` | `0.627658` | `1.421675` | `0.263156` | `0.931515` |
| `late_attach_aux` | `8` | `0.094440` | `0.049972` | `0.660008` | `1.438880` | `0.251461` | `0.913264` |

## 7. Descriptive correlations

Required all-epoch descriptive correlation (`epoch 1..8`, `aux_leg_loss` vs `leg p95`):

| arm | Pearson | Spearman |
| --- | ---: | ---: |
| `shared_attach_aux` | `0.933228` | `0.928571` |
| `aux_detach` | `0.932341` | `0.928571` |
| `late_attach_aux` | `0.955546` | `0.952381` |

This is the most important raw readout of E4:

- the requested **full-epoch anti-alignment did not appear**
- all 3 arms show strong **positive** correlation over epochs `1..8`
- i.e. the naive “aux gets better while freerun gets worse” reading is **not** what the whole curve says

Interpretation:

- early `stage6` convergence dominates the full-epoch trajectory
- from epoch 1 to later epochs, both train-side aux readability and freerun error improve together
- therefore the simple all-epoch correlation is **not discriminative enough** to confirm the stronger mismatch claim by itself

## 8. Late-phase check

Because the full-epoch trend is dominated by early convergence, the only visible mismatch signal is late-stage:

epoch `7 -> 8` deltas:

| arm | `Δ aux_leg_loss` | `Δ leg p95` | `Δ all_ex_root p95` | `Δ leg mean` |
| --- | ---: | ---: | ---: | ---: |
| `shared_attach_aux` | `-0.004762` | `+0.304380` | `+0.052404` | `+0.050643` |
| `aux_detach` | `-0.005675` | `-0.636958` | `-0.181611` | `-0.191139` |
| `late_attach_aux` | `-0.002542` | `+0.017204` | `-0.018251` | `+0.032350` |

Equivalent percentage view for `epoch 7 -> 8`:

- `shared_attach_aux`
  - `aux_leg_loss`: `-8.06%`
  - `leg p95`: `+20.45%`
  - `all_ex_root p95`: `+5.59%`
- `aux_detach`
  - `aux_leg_loss`: `-9.47%`
  - `leg p95`: `-32.61%`
  - `all_ex_root p95`: `-16.23%`
- `late_attach_aux`
  - `aux_leg_loss`: `-2.62%`
  - `leg p95`: `+1.21%`
  - `all_ex_root p95`: `-1.96%`

Late-phase minima:

- `shared_attach_aux`
  - best `leg p95` at epoch `7`: `1.488769`
  - final epoch `8`: `1.793149` (worse)
- `aux_detach`
  - best `leg p95` at epoch `8`: `1.316565`
  - no late reversal
- `late_attach_aux`
  - best `leg p95` at epoch `7`: `1.421675`
  - final epoch `8`: `1.438880` (slightly worse)

So the only clean E4 mismatch signal is:

- **shared attach** keeps reducing aux loss late, but rollout degrades from epoch `7` to `8`
- **detach** removes that late reversal
- **late attach** still shows a tiny residual `leg p95` reversal, but much weaker than shared attach

## 9. Cross-arm final mismatch

Epoch `8` cross-arm comparison is still highly informative:

| arm | aux_leg_loss | aux_leg_over_main | leg mean | leg p95 | all_ex_root mean | all_ex_root p95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `shared_attach_aux` | `0.054306` | `0.028285` | `0.708246` | `1.793149` | `0.268797` | `0.990646` |
| `aux_detach` | `0.054265` | `0.028465` | `0.597226` | `1.316565` | `0.262125` | `0.937262` |
| `late_attach_aux` | `0.094440` | `0.049972` | `0.660008` | `1.438880` | `0.251461` | `0.913264` |

Key deltas:

- `shared_attach_aux` vs `aux_detach`
  - `aux_leg_loss`: `+0.000041` (effectively identical)
  - `leg p95`: `+0.476584` (shared attach much worse)
- `shared_attach_aux` vs `late_attach_aux`
  - `aux_leg_loss`: `-0.040134` (shared attach has much **better** aux readability)
  - `leg p95`: `+0.354269` (but still **worse** freerun)

This is still a real supervision/readability vs rollout mismatch at the **cross-arm endpoint**:

- better aux readability does **not** guarantee better freerun
- and nearly identical aux readability (`shared_attach_aux` vs `aux_detach`) can coexist with a large freerun gap

## 10. Answers to the required questions

### Q1. In `shared_attach_aux`, when `aux_leg_loss(epoch)` drops, does `leg p95(epoch)` worsen, or at least fail to improve?

**Not as a full-epoch systematic trend.**

What the data says:

- over epochs `1..8`, `leg p95` improves massively: `15.093977 -> 1.793149`
- full-epoch correlation is strongly positive, not negative:
  - Pearson `0.933228`
  - Spearman `0.928571`

So the strong version of the E4 hypothesis is **not confirmed**.

But there is a **late-phase local mismatch**:

- epoch `7 -> 8`: `aux_leg_loss` keeps dropping (`-8.06%`)
- while `leg p95` worsens (`+20.45%`)
- and `all_ex_root p95` also worsens (`+5.59%`)

So the answer is:

- **no clean full-epoch mismatch**
- **yes weak late-stage mismatch**

### Q2. In `aux_detach`, is that coupling significantly weakened or gone?

**Yes for the late-phase reversal; no for the naive full-epoch correlation.**

Full-epoch:

- still strongly positive (`Pearson 0.932341`, `Spearman 0.928571`)

Late-phase:

- unlike `shared_attach_aux`, `aux_detach` keeps improving on rollout at epoch `7 -> 8`
  - `aux_leg_loss`: `-9.47%`
  - `leg p95`: `-32.61%`
  - `all_ex_root p95`: `-16.23%`

So the specific late-stage “aux keeps improving while rollout stalls/reverses” coupling is **clearly weaker / absent** in `aux_detach`.

### Q3. In `late_attach_aux`, even with stronger readability than sham, does mismatch still remain?

**Only weakly.**

Full-epoch:

- again positive correlation, not anti-correlation

Late-phase:

- epoch `7 -> 8` still shows a tiny `leg p95` worsening (`+1.21%`) while `aux_leg_loss` drops (`-2.62%`)
- but `all_ex_root p95` still improves slightly (`-1.96%`)

So `late_attach_aux` does **not** show a strong epochwise mismatch, but it also does **not** produce a clean “better aux readability => better freerun” story.

### Q4. Do the 3 arms together support:

#### (a) `gradient conflict / redundancy` as the near mechanism?

**Yes, still supported.**

Main reason is not the full-epoch correlation.  
Main reason is the reproduced **endpoint cross-arm mismatch**:

- `shared_attach_aux` and `aux_detach` have essentially the same epoch-8 aux readability
- but `shared_attach_aux` freerun is much worse

That remains highly consistent with:

- aux-gradient-through-trainable-pipeline as the damaging near mechanism

which is exactly the E1 direction.

#### (b) `supervision–rollout objective mismatch` as the deeper direction?

**Retained, but weakened and no longer primary after E4.**

The most constraining E4 fact is:

- `shared_attach_aux` and `aux_detach` end with nearly identical epoch-8 aux readability
  - `aux_leg_loss = 0.054306` vs `0.054265`
- but freerun differs sharply
  - `leg p95` gap = `+0.476584`

If objective mismatch were already the dominant explanation, two arms with the same objective and nearly the same terminal aux loss should not separate this strongly on freerun.  
That separation points more directly to:

- **which gradient path was allowed to update the trainable direct/trunk pipeline**

rather than:

- the objective semantics alone

So E4 does **not** support keeping `supervision–rollout mismatch` as the primary root explanation.  
It survives only as a weaker secondary explanation for why:

- better readability still does not reliably translate into better freerun
- and shared attach shows a late-stage local reversal

### Q5. After E4, is it enough to justify scoping `E5 rollout-aware aux objective`?

**Not enough to launch `E5c rollout-aware aux objective` directly.**

Updated go / no-go:

- **GO first**: `E5a`
  - minimal `shared_attach_aux` `epoch 7` early-stop vs `epoch 8` final confirmation
  - purpose: test whether the late reversal is robust rather than a simple overtrain artifact
- **OPTIONAL next**: `E5b`
  - another small gradient-path probe extending E1-style evidence
  - purpose: further squeeze the `gradient-path` explanation before changing objectives
- **NO-GO for now**: `E5c rollout-aware aux objective`

Reason:

- E4 shifted explanatory weight toward `aux-gradient-through-trainable-pipeline interference`
- the strongest new fact is endpoint path-dependence, not objective-semantics failure
- so a direct jump to objective redesign is now too large a step

What E4 does justify is:

- a smaller follow-up that first checks whether the residual late-phase reversal is stable
- and whether gradient-path explanations can absorb the remaining behavior without changing the objective

## 11. Final concise readout

E4 did **not** deliver the strongest hoped-for confirmation.

What it actually shows is:

1. full-epoch `aux_leg_loss` vs `leg p95` is **positively** correlated in all 3 arms, because early stage6 convergence dominates
2. the only clear mismatch signal is **late-phase**, especially `shared_attach_aux` epoch `7 -> 8`
3. `aux_detach` removes that late-phase reversal
4. final cross-arm mismatch remains very strong:
   - same aux readability can yield very different freerun
   - better aux readability can still yield worse freerun

So the best updated E4 reading is:

> as a strict epochwise observational confirmation, the result is mixed / weaker than expected;  
> but combined with E1–E3, it strengthens aux-gradient path interference as the primary near mechanism, weakens `supervision–rollout mismatch` as the primary root explanation, and points to `E5a-first` rather than direct `E5c`.

## 12. Updated mechanism hierarchy (post-E4)

This section updates the earlier E1–E3 prior.  
The prior does **not** get discarded, but it does need to yield to the stronger E4 endpoint evidence.

### 1. Primary near mechanism

- `aux gradient, when routed through the trainable shared trunk, perturbs trunk parameters toward solutions that serve aux readability without commensurate benefit to freerun rollout`

Why this is now primary:

- E1 already showed the harm depends on aux gradient entering the trainable direct/trunk path
- E4 adds the strongest new discriminator:
  - `shared_attach_aux` and `aux_detach` finish with almost identical `aux_leg_loss`
  - but freerun still differs strongly
- E4 also adds a second, weaker support signal:
  - `shared_attach_aux` shows a late `epoch 7 -> 8` reversal
  - `aux_detach` removes that reversal

So the best current primary explanation is:

- not “the target itself is already wrong”
- but “allowing that target’s gradient to flow through the trainable direct/trunk pipeline is what causes the damage”

Important unresolved split inside this primary bucket:

- `sign conflict`
- vs `capacity / plasticity sink`

Current E4 evidence does **not** separate them.

What E4 says about this split:

- support for the strongest `sign conflict` reading is **weak**
  - the full-epoch curves do not show a clean “aux improves while rollout monotonically worsens” pattern
  - instead, all three arms show strong positive full-epoch correlation
- the data are more naturally compatible with a `capacity / plasticity sink` reading
  - the trainable trunk can still move toward states that improve aux readability
  - yet those states need not deliver comparable freerun benefit
  - and two arms can end at nearly identical aux readability with very different rollout quality

So after E4, the honest primary claim is:

- the **gradient path through the trainable trunk is causal**
- but the internal failure mode (`sign conflict` vs `capacity / plasticity sink`) is still unresolved
- and that unresolved split is exactly what `E5b` should target

### 2. Secondary, retained but weaker

- `supervision–rollout objective mismatch`

Why it is downgraded:

- E4 failed to produce the hoped-for strong within-arm monotone mismatch
- full-epoch correlation is strongly positive in all three arms
- the most decisive E4 fact is path-dependent endpoint separation, which points more to gradient path than to objective semantics

Why it is still retained:

- better aux readability still does not reliably imply better freerun
- `shared_attach_aux` still shows a late-phase local mismatch
- but this late-phase support is still only a **single-seed** observation at present

So this explanation survives, but only as:

- a weaker secondary layer
- not the primary root after E4

If `E5a-seed` fails to reproduce the late reversal, then this explanation should be downgraded again, potentially to:

- `not currently supported by direct within-arm evidence`

### 3. Harm modulator

- `attach mismatch`

Why:

- E3 showed attach location changes harm magnitude
- but attach changes did not produce a clean rescue

So attach still matters, but as a:

- **modulator**
- not the dominant mechanism

### 4. Secondary retained

- `structural fork / head-side competition`

Why it stays in the stack:

- attach changes do not fully erase harm
- some extra head-side / branch-side competition may still contribute

But after E1 + E4, this is still behind:

- trunk-directed gradient interference

### 5. Not primary

- `capacity saturation / no usable signal`

Why:

- E2 and E3 both show nonzero leg-readable signal
- the pipeline is not signal-empty

So this remains de-emphasized.

## 13. Recommended next step / E5 status

### Go / no-go

| item | status | reason |
| --- | --- | --- |
| `E5a-seed` | **GO first** | the late reversal is currently `n=1`; replicate it before spending downstream budget |
| `E5a-downstream` | **conditional GO** | only worth doing if `E5a-seed` reproduces the reversal and we want to know whether the early-stop advantage survives handoff |
| `E5b` | **GO after `E5a-seed`** | this is the right discriminator for `sign conflict` vs `capacity / plasticity sink`, regardless of whether the late reversal fully replicates |
| `E5c` | **NO-GO for now** | E4 weakens the case for jumping directly to rollout-aware objective redesign |

### Recommended sequence

1. `E5a-seed`
   - rerun one more `shared_attach_aux` seed with the same epochwise snapshot protocol
   - ask only whether the `epoch 7 -> 8` reversal replicates
2. Branch on that result:
   - if seed B **does** reproduce the reversal:
     - run `E5a-downstream`
     - run `E5b` in parallel if budget allows
   - if seed B **does not** reproduce the reversal:
     - drop the late-phase mismatch from the evidentiary core
     - skip `E5a-downstream`
     - still run `E5b`, because the endpoint cross-arm path dependence remains hard evidence
3. Only if `E5b` still leaves objective-level misalignment unexplained, open `E5c`

### `E5b` purpose

`E5b` should be framed explicitly as:

> distinguish `sign conflict` from `capacity / plasticity sink`

Minimal readout logic:

- if harm changes roughly monotonically with aux-gradient strength while aux readability stays similar,
  - that is more compatible with `capacity / plasticity sink`
- if there is a more non-monotone rescue pattern,
  - or gradient surgery / partial cancellation produces a disproportionate gain,
  - that is more compatible with `sign conflict`

### Bottom line

After E4, the cleanest recommendation is:

> do **not** jump directly into `rollout-aware aux objective`;  
> first replicate the late reversal at `n=2`, then use `E5b` to decide whether the primary path effect is closer to `sign conflict` or to `capacity / plasticity sink`.
