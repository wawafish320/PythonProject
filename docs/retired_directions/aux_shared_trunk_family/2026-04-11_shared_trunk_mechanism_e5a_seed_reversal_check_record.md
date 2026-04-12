# E5a-seed late-reversal replication (`shared_attach_aux`, seed-B)

> Status: archived / retired aux-family mechanism record
> Reader note: this aux / shared-trunk family did **not** become current repo mainline; any `recommend`, `default`, `ship`, `mainline`, or `current` wording below is historical family-local language only.
> Current entry points: `docs/posttrain_pipeline.md`, `docs/train_design/2026-04-11_donor_contract_basin_universal_debug_synthesis.md`, `docs/retired_directions/aux_shared_trunk_family/2026-04-11_aux_detach_downstream_reevaluation_record.md`

> Date: `2026-04-11`
> Scope: **only** `E5a-seed`
> Result: **seed B does not reproduce the E4 epoch `7 -> 8` late-phase reversal**

## 1. Minimal question

This record answers one question only:

- on a new seed B, inside `shared_attach_aux`, does the E4 late-phase reversal replicate?
- i.e. when `aux_leg_loss` keeps dropping from epoch `7 -> 8`, does freerun `leg p95` worsen again, or at least fail to improve?

Out of scope for this record:

- no `70a / 70b`
- no `E5a-downstream`
- no `E5b` sweep / surgery / clip / detach sweep
- no `E5c`
- no objective change
- no aux-weight sweep
- no multi-seed expansion beyond one new seed-B rerun

## 2. Reused prior conclusions

E1–E4 are reused as-is and are not re-judged here:

- `docs/retired_directions/aux_shared_trunk_family/2026-04-10_shared_trunk_mechanism_e1_aux_detach_record.md`
- `docs/retired_directions/aux_shared_trunk_family/2026-04-10_shared_trunk_mechanism_e2_frozen_trunk_aux_readability_record.md`
- `docs/retired_directions/aux_shared_trunk_family/2026-04-10_shared_trunk_mechanism_e3_late_attach_probe_record.md`
- `docs/retired_directions/aux_shared_trunk_family/2026-04-11_shared_trunk_mechanism_e4_epochwise_aux_rollout_mismatch_record.md`

Seed-A values are reused directly from E4.

## 3. Existing-artifact check

Before rerunning, I checked for an already-complete seed-B matched artifact under the expected locations and nearby obvious variants:

- `models/__tmp_dsn_aux_leg_matched_chain_20260411/stage6/shared_attach_aux_epochsnap_seedB`
- `debug_output/_tmp_dsn_aux_leg_matched_chain_20260411/e5a_seed`
- nearby names containing `seedB`, `seedb`, `2025`, or `e5a`

Result:

- **no complete seed-B matched artifact existed**
- so one minimal matched rerun was required

## 4. Seed knob check

I first verified that current runtime already supports a real seed knob, and did **not** add any new plumbing.

Observed effective seed knob:

- posttrain runtime field: `seed`
- seed-A existing run: `seed = 0`
- seed-B rerun: `seed = 2025`

Applied rule:

- copied E4 config
  - from `debug_output/_tmp_dsn_aux_leg_matched_chain_20260411/configs/shared_attach_aux_epochsnap.json`
  - to `debug_output/_tmp_dsn_aux_leg_matched_chain_20260411/configs/shared_attach_aux_epochsnap_seedB.json`
- changed only:
  - `"seed": 2025`

Notes:

- donor checkpoint stayed fixed:
  - `models/cp015_phasecd_tailk_probe_20260331/exp_phase_DirectBranch_v1_d1_cp015_tailk7_rankmix_tw020_corridor_hold_tail15_phasea050_fixedsched_ep014center_control_denseckpt_seed2024_20260401/ckpt_epoch_014.pth`
- `stage6 native`, contacts recipe, encoder bundle, epochs, steps, and snapshot schedule all remained matched to E4
- I did **not** use any new feature or extra seed field

## 5. Actual commands run

## 5.1 Artifact search

```bash
find models/__tmp_dsn_aux_leg_matched_chain_20260411 \
     debug_output/_tmp_dsn_aux_leg_matched_chain_20260411 \
     -maxdepth 5 \( -iname '*seedB*' -o -iname '*seedb*' -o -iname '*2025*' -o -iname '*e5a*' \) | sort
```

## 5.2 Seed-B matched posttrain rerun

```bash
PYTHONPATH=. debug_output/_tmp_phasecd_min_ablation_20260330/cpu_nomps_exec.py -m train.posttrain \
  --config debug_output/_tmp_dsn_aux_leg_matched_chain_20260411/configs/shared_attach_aux_epochsnap_seedB.json \
  --ckpt_in models/cp015_phasecd_tailk_probe_20260331/exp_phase_DirectBranch_v1_d1_cp015_tailk7_rankmix_tw020_corridor_hold_tail15_phasea050_fixedsched_ep014center_control_denseckpt_seed2024_20260401/ckpt_epoch_014.pth \
  --out_dir models/__tmp_dsn_aux_leg_matched_chain_20260411/stage6/shared_attach_aux_epochsnap_seedB \
  --run_name lr3e4_e8x60_wd1e4_reinit1_stage6_tailfix_dsn_sharedaux_epochsnap_seed2025_20260411 \
  --posttrain_contacts_source pretrain_contact \
  --posttrain_contacts_pretrain_clamp 1.0 \
  --encoder_bundle models/motion_encoder_equiv.pt.best.pt \
  --posttrain_contacts_pretrain_affine_stats debug_output/_tmp_phaseb_affine_20260304/affine_fit_mix08/affine_stats.json
```

Matched recipe:

- `stage6 native`
- `8 epochs × 60 steps`
- `save_step_ckpts = 0,60,120,180,240,300,360,420,480`
- only semantic change: posttrain `seed = 2025`

## 5.3 Seed-B epochwise freerun eval

Seed-A epochwise freerun/group summaries were reused from E4.

For seed B, I ran the same E4 eval recipe over all snapshots:

```bash
for step in 60 120 180 240 300 360 420 480; do
  epoch=$((step/60))
  run_dir=$(printf 'debug_output/_tmp_dsn_aux_leg_matched_chain_20260411/e5a_seed/shared_attach_aux_seedB/epoch_%02d_step_%06d_freerun' "$epoch" "$step")
  summary=$(printf 'debug_output/_tmp_dsn_aux_leg_matched_chain_20260411/e5a_seed/shared_attach_aux_seedB/epoch_%02d_step_%06d_group_summary.json' "$epoch" "$step")
  PYTHONPATH=. debug_output/_tmp_phasecd_min_ablation_20260330/cpu_nomps_exec.py -m train.validate.run_freerun_cycles \
    --teacher validate/teacher_batches/Walk_F_teacher.json \
    --model models/__tmp_dsn_aux_leg_matched_chain_20260411/stage6/shared_attach_aux_epochsnap_seedB/ckpt_step_$(printf '%06d' "$step")_lr3e4_e8x60_wd1e4_reinit1_stage6_tailfix_dsn_sharedaux_epochsnap_seed2025_20260411.pth \
    --rounds 5 --depth 3 --time-index-mode cycle \
    --phase_reset_source none \
    --contacts_meas_source pretrain_contact \
    --contacts_meas_pretrain_clamp 1.0 \
    --contacts_meas_pretrain_affine_stats debug_output/_tmp_phaseb_affine_20260304/affine_fit_mix08/affine_stats.json \
    --encoder-bundle models/motion_encoder_equiv.pt.best.pt \
    --export_joint_direct_geolocal_series \
    --out "$run_dir" --force && \
  PYTHONPATH=. debug_output/_tmp_phasecd_min_ablation_20260330/cpu_nomps_exec.py \
    tools/phasea_group_summary.py "$run_dir/Walk_F_freerun_cycles.json" \
    --cycle_gte 1 --drop_wrap --out "$summary" || exit 1
done
```

Same eval recipe as E4:

- `teacher = validate/teacher_batches/Walk_F_teacher.json`
- `rounds = 5`
- `depth = 3`
- `time-index-mode = cycle`
- `phase_reset_source = none`
- `contacts_meas_source = pretrain_contact`
- `contacts_meas_pretrain_clamp = 1.0`
- same affine stats / encoder bundle
- same `phasea_group_summary.py --cycle_gte 1 --drop_wrap`

## 5.4 Aggregation + plot generation

```bash
PYTHONPATH=. python3 debug_output/_tmp_dsn_aux_leg_matched_chain_20260411/e5a_seed/e5a_seed_reversal_check.py
```

## 6. Output artifacts

Primary outputs:

- config: `debug_output/_tmp_dsn_aux_leg_matched_chain_20260411/configs/shared_attach_aux_epochsnap_seedB.json`
- seed-B model dir: `models/__tmp_dsn_aux_leg_matched_chain_20260411/stage6/shared_attach_aux_epochsnap_seedB`
- seed-B eval dir: `debug_output/_tmp_dsn_aux_leg_matched_chain_20260411/e5a_seed/shared_attach_aux_seedB`
- aggregation script: `debug_output/_tmp_dsn_aux_leg_matched_chain_20260411/e5a_seed/e5a_seed_reversal_check.py`
- epochwise csv: `debug_output/_tmp_dsn_aux_leg_matched_chain_20260411/e5a_seed/e5a_seed_epochwise_table.csv`
- epoch7/8 csv: `debug_output/_tmp_dsn_aux_leg_matched_chain_20260411/e5a_seed/e5a_seed_epoch78_compare.csv`
- metrics json: `debug_output/_tmp_dsn_aux_leg_matched_chain_20260411/e5a_seed/e5a_seed_metrics.json`
- aux-vs-leg plot: `debug_output/_tmp_dsn_aux_leg_matched_chain_20260411/e5a_seed/e5a_seed_shared_attach_aux_loss_vs_leg_p95.png`
- root plot: `debug_output/_tmp_dsn_aux_leg_matched_chain_20260411/e5a_seed/e5a_seed_shared_attach_all_ex_root_p95.png`

## 7. Per-epoch results

| seed | epoch | aux_leg_loss | aux_leg_over_main | leg mean | leg p95 | all_ex_root mean | all_ex_root p95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `seed A` | `1` | `0.150498` | `0.103429` | `3.600986` | `15.093977` | `1.726197` | `6.706787` |
| `seed A` | `2` | `0.110540` | `0.080070` | `1.604378` | `5.140162` | `0.731995` | `2.746501` |
| `seed A` | `3` | `0.090446` | `0.055230` | `1.092974` | `3.082576` | `0.505677` | `1.938164` |
| `seed A` | `4` | `0.078847` | `0.044641` | `0.767092` | `2.062629` | `0.352334` | `1.195815` |
| `seed A` | `5` | `0.072849` | `0.039018` | `0.872817` | `1.896572` | `0.366850` | `1.352452` |
| `seed A` | `6` | `0.065077` | `0.034288` | `0.709040` | `1.619963` | `0.314729` | `1.103781` |
| `seed A` | `7` | `0.059068` | `0.030329` | `0.657603` | `1.488769` | `0.260299` | `0.938242` |
| `seed A` | `8` | `0.054306` | `0.028285` | `0.708246` | `1.793149` | `0.268797` | `0.990646` |
| `seed B` | `1` | `0.152515` | `0.101425` | `3.767184` | `14.646245` | `1.843079` | `7.079636` |
| `seed B` | `2` | `0.116947` | `0.083781` | `1.856094` | `5.746659` | `0.824308` | `3.188083` |
| `seed B` | `3` | `0.089102` | `0.056565` | `1.049732` | `2.766312` | `0.455442` | `1.591683` |
| `seed B` | `4` | `0.077689` | `0.045144` | `0.876393` | `2.249531` | `0.367189` | `1.345972` |
| `seed B` | `5` | `0.072212` | `0.039070` | `0.931779` | `2.288257` | `0.391031` | `1.506628` |
| `seed B` | `6` | `0.064074` | `0.032281` | `0.757389` | `1.774643` | `0.313573` | `1.100584` |
| `seed B` | `7` | `0.059712` | `0.031253` | `0.840133` | `1.988129` | `0.346959` | `1.255716` |
| `seed B` | `8` | `0.055731` | `0.029426` | `0.709255` | `1.669341` | `0.307925` | `1.215963` |

## 8. Seed-A vs seed-B epoch 7/8 comparison

| seed | epoch | aux_leg_loss | leg p95 | all_ex_root p95 |
| --- | ---: | ---: | ---: | ---: |
| `seed A` | `7` | `0.059068` | `1.488769` | `0.938242` |
| `seed A` | `8` | `0.054306` | `1.793149` | `0.990646` |
| `seed B` | `7` | `0.059712` | `1.988129` | `1.255716` |
| `seed B` | `8` | `0.055731` | `1.669341` | `1.215963` |

Epoch `7 -> 8` deltas:

| seed | Δ aux_leg_loss | aux % | Δ leg p95 | leg p95 % | Δ all_ex_root p95 | all_ex_root p95 % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `seed A` | `-0.004762` | `-8.06%` | `+0.304380` | `+20.45%` | `+0.052404` | `+5.59%` |
| `seed B` | `-0.003982` | `-6.67%` | `-0.318787` | `-16.03%` | `-0.039754` | `-3.17%` |

## 9. Direct answers to the required questions

### Q1. In seed B, from epoch `7 -> 8`, does `aux_leg_loss` continue to decrease?

**Yes.**

- epoch `7`: `0.059712`
- epoch `8`: `0.055731`
- delta: `-0.003982` (`-6.67%`)

### Q2. In seed B, does `leg p95` worsen again, or at least fail to improve?

**No.**

It improves materially:

- epoch `7`: `1.988129`
- epoch `8`: `1.669341`
- delta: `-0.318787` (`-16.03%`)

So the E4 late-phase `leg p95` reversal does **not** replicate on seed B.

### Q3. In seed B, does `all_ex_root p95` worsen again, or at least fail to improve?

**No.**

It also improves:

- epoch `7`: `1.255716`
- epoch `8`: `1.215963`
- delta: `-0.039754` (`-3.17%`)

So the E4 late-phase `all_ex_root p95` worsening also does **not** replicate on seed B.

### Q4. With seed A + seed B together, is the late-phase reversal robust enough to keep as evidence in the E4/E5 chain?

**No, not as a robust within-arm signal.**

Best current read:

- seed A shows the late reversal
- seed B shows the opposite direction: aux readability keeps improving **and** rollout also improves from epoch `7 -> 8`

So the direct within-arm late-reversal evidence is now:

- `1 / 2` seeds
- not stable enough to treat as a robust replication

What still remains valid:

- E4 cross-arm endpoint mismatch remains real
- but the specific **within-arm late reversal** is no longer strong enough to stand as a stable support pillar

### Q5. If seed B does not reproduce the reversal, should “secondary: supervision–rollout objective mismatch” be downgraded further?

**Yes.**

Recommended downgrade:

- from “supported by a direct within-arm late-phase reversal”  
- to “currently lacks stable direct within-arm evidence; retained only as a weaker secondary possibility, mainly supported by cross-arm endpoint mismatch rather than replicated within-arm reversal”

In short:

- the E4 secondary explanation should be **downgraded**
- the phrase “目前缺少直接 within-arm 证据” is appropriate

### Q6. Based on E5a-seed, should the next step be `E5a-downstream`, or skip to `E5b`? `E5c` remains paused.

Recommended next step:

- **skip `E5a-downstream`**
- **go directly to `E5b`**
- **keep `E5c` paused**

Reason:

- `E5a-downstream` was only attractive if the late reversal itself replicated
- after seed B, that late reversal is not stable enough to justify spending downstream budget
- `E5b` is still the right discriminator for aux-gradient interference vs other explanations

## 10. Bottom-line interpretation

Concise read:

- seed B confirms only the first half of the pattern:
  - `aux_leg_loss` still drops from epoch `7 -> 8`
- but it fails the actual reversal criterion:
  - `leg p95` improves instead of worsening
  - `all_ex_root p95` also improves instead of worsening

Therefore:

- **E5a-seed is negative for late-reversal replication**
- the E4 epoch `7 -> 8` reversal should now be treated as a **single-seed anecdote**, not a stable replicated within-arm effect
- the clean next move is **`E5b`, not `E5a-downstream`**
