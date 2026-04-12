# 2026-04-10 Shared Trunk Mechanism Disambiguation — E3 `late_attach_probe` Record

> Status: archived / retired aux-family mechanism record
> Reader note: this aux / shared-trunk family did **not** become current repo mainline; any `recommend`, `default`, `ship`, `mainline`, or `current` wording below is historical family-local language only.
> Current entry points: `docs/posttrain_pipeline.md`, `docs/train_design/2026-04-11_donor_contract_basin_universal_debug_synthesis.md`, `docs/retired_directions/aux_shared_trunk_family/2026-04-11_aux_detach_downstream_reevaluation_record.md`

## 1. Scope

This round only runs the minimal `E3 late_attach_probe`.

Goal:

> Distinguish whether the current failure is closer to `attach mismatch`, or to a broader `capacity saturation / no usable leg signal in current trunk pipeline`.

Per user constraint, this round:

- does **not** redesign trunk/head
- does **not** expand into attach-point grid search
- does **not** rerun `baseline / sham / aux / aux_detach / E2`
- only adds one diagnostic attach option and two new `stage6 native` arms:
  - `late_attach_sham`
  - `late_attach_aux`

## 2. Minimal implementation

### 2.1 Why `leg_boundary` was implemented on `direct_pose_leg_head` hidden

The plan priority was:

1. `direct_pose_out_leg` input
2. fallback to `direct_pose_leg_head` hidden

After checking the current `stage6` direct split-readout path, `direct_pose_out_leg` input is the same shared hidden that the existing aux head already reads:

- current shared aux tap = shared direct hidden
- current leg readout input = the same shared direct hidden

So using `direct_pose_out_leg` input would not create an actual E3 discrimination.  
To keep the probe meaningful while staying minimal, `leg_boundary` was implemented as:

- `direct_pose_leg_head[:-1]` hidden
- only for the non-routed `direct_pose_leg_head` path
- default behavior remains `shared_trunk`

### 2.2 Code changes

- `train/models.py`
  - added `direct_pose_aux_leg_attach` with:
    - `shared_trunk` (default, backward-compatible)
    - `leg_boundary`
  - `leg_boundary` resolves aux input from `direct_pose_leg_head` hidden
  - added guardrails so `leg_boundary` only works on the simple non-routed leg-head path
- `train/posttrain.py`
  - added config / CLI plumbing for `direct_pose_aux_leg_attach`
  - export-strip path resets attach to `shared_trunk`
- `tests/train/test_posttrain_direct_pose_aux_leg.py`
  - added focused unit test covering `leg_boundary` forward path

Validation run:

- `python3 -m py_compile train/models.py train/posttrain.py tests/train/test_posttrain_direct_pose_aux_leg.py`
- `python3 -m unittest tests.train.test_posttrain_direct_pose_aux_leg`

Result:

- `5` tests passed

## 3. New configs

- `debug_output/_tmp_dsn_aux_leg_matched_chain_20260410/configs/posttrain_WalkF_stage6_direct_cond_anchor_splitfirst_3way_armchain_pe32h512_tailfix_lr3e4_e8x60_wd1e4_reinit1_late_attach_sham_20260410.json`
- `debug_output/_tmp_dsn_aux_leg_matched_chain_20260410/configs/posttrain_WalkF_stage6_direct_cond_anchor_splitfirst_3way_armchain_pe32h512_tailfix_lr3e4_e8x60_wd1e4_reinit1_late_attach_aux_20260410.json`

Both keep the fixed stage6 recipe and only change:

- `direct_pose_aux_leg_enable=true`
- `direct_pose_aux_leg_attach=leg_boundary`
- `direct_pose_aux_leg_detach_feat=false`
- `direct_pose_aux_leg_log_enable=true`

Per-arm difference:

- `late_attach_sham`: `direct_pose_aux_leg_weight=0.0`
- `late_attach_aux`: `direct_pose_aux_leg_weight=0.2`

## 4. Actual commands

### 4.1 Train

```bash
PYTHONPATH=. debug_output/_tmp_phasecd_min_ablation_20260330/cpu_nomps_exec.py -m train.posttrain \
  --config debug_output/_tmp_dsn_aux_leg_matched_chain_20260410/configs/posttrain_WalkF_stage6_direct_cond_anchor_splitfirst_3way_armchain_pe32h512_tailfix_lr3e4_e8x60_wd1e4_reinit1_late_attach_sham_20260410.json \
  --ckpt_in models/cp015_phasecd_tailk_probe_20260331/exp_phase_DirectBranch_v1_d1_cp015_tailk7_rankmix_tw020_corridor_hold_tail15_phasea050_fixedsched_ep014center_control_denseckpt_seed2024_20260401/ckpt_epoch_014.pth \
  --out_dir models/__tmp_dsn_aux_leg_matched_chain_20260410/stage6/late_attach_sham \
  --run_name lr3e4_e8x60_wd1e4_reinit1_stage6_tailfix_dsn_lateattach_sham_20260410 \
  --posttrain_contacts_source pretrain_contact \
  --posttrain_contacts_pretrain_clamp 1.0 \
  --encoder_bundle models/motion_encoder_equiv.pt.best.pt \
  --posttrain_contacts_pretrain_affine_stats debug_output/_tmp_phaseb_affine_20260304/affine_fit_mix08/affine_stats.json
```

```bash
PYTHONPATH=. debug_output/_tmp_phasecd_min_ablation_20260330/cpu_nomps_exec.py -m train.posttrain \
  --config debug_output/_tmp_dsn_aux_leg_matched_chain_20260410/configs/posttrain_WalkF_stage6_direct_cond_anchor_splitfirst_3way_armchain_pe32h512_tailfix_lr3e4_e8x60_wd1e4_reinit1_late_attach_aux_20260410.json \
  --ckpt_in models/cp015_phasecd_tailk_probe_20260331/exp_phase_DirectBranch_v1_d1_cp015_tailk7_rankmix_tw020_corridor_hold_tail15_phasea050_fixedsched_ep014center_control_denseckpt_seed2024_20260401/ckpt_epoch_014.pth \
  --out_dir models/__tmp_dsn_aux_leg_matched_chain_20260410/stage6/late_attach_aux \
  --run_name lr3e4_e8x60_wd1e4_reinit1_stage6_tailfix_dsn_lateattach_auxw02_20260410 \
  --posttrain_contacts_source pretrain_contact \
  --posttrain_contacts_pretrain_clamp 1.0 \
  --encoder_bundle models/motion_encoder_equiv.pt.best.pt \
  --posttrain_contacts_pretrain_affine_stats debug_output/_tmp_phaseb_affine_20260304/affine_fit_mix08/affine_stats.json
```

### 4.2 Free-run eval + group summary

```bash
PYTHONPATH=. debug_output/_tmp_phasecd_min_ablation_20260330/cpu_nomps_exec.py -m train.validate.run_freerun_cycles \
  --teacher validate/teacher_batches/Walk_F_teacher.json \
  --model models/__tmp_dsn_aux_leg_matched_chain_20260410/stage6/late_attach_sham/ckpt_last_lr3e4_e8x60_wd1e4_reinit1_stage6_tailfix_dsn_lateattach_sham_20260410.pth \
  --rounds 5 --depth 3 --time-index-mode cycle --phase_reset_source none \
  --contacts_meas_source pretrain_contact --contacts_meas_pretrain_clamp 1.0 \
  --contacts_meas_pretrain_affine_stats debug_output/_tmp_phaseb_affine_20260304/affine_fit_mix08/affine_stats.json \
  --encoder-bundle models/motion_encoder_equiv.pt.best.pt \
  --export_joint_direct_geolocal_series \
  --out debug_output/_tmp_dsn_aux_leg_matched_chain_20260410/stage6/late_attach_sham/stage6_freerun --force

PYTHONPATH=. debug_output/_tmp_phasecd_min_ablation_20260330/cpu_nomps_exec.py tools/phasea_group_summary.py \
  debug_output/_tmp_dsn_aux_leg_matched_chain_20260410/stage6/late_attach_sham/stage6_freerun/Walk_F_freerun_cycles.json \
  --cycle_gte 1 --drop_wrap \
  --out debug_output/_tmp_dsn_aux_leg_matched_chain_20260410/stage6/late_attach_sham/stage6_group_summary.json
```

```bash
PYTHONPATH=. debug_output/_tmp_phasecd_min_ablation_20260330/cpu_nomps_exec.py -m train.validate.run_freerun_cycles \
  --teacher validate/teacher_batches/Walk_F_teacher.json \
  --model models/__tmp_dsn_aux_leg_matched_chain_20260410/stage6/late_attach_aux/ckpt_last_lr3e4_e8x60_wd1e4_reinit1_stage6_tailfix_dsn_lateattach_auxw02_20260410.pth \
  --rounds 5 --depth 3 --time-index-mode cycle --phase_reset_source none \
  --contacts_meas_source pretrain_contact --contacts_meas_pretrain_clamp 1.0 \
  --contacts_meas_pretrain_affine_stats debug_output/_tmp_phaseb_affine_20260304/affine_fit_mix08/affine_stats.json \
  --encoder-bundle models/motion_encoder_equiv.pt.best.pt \
  --export_joint_direct_geolocal_series \
  --out debug_output/_tmp_dsn_aux_leg_matched_chain_20260410/stage6/late_attach_aux/stage6_freerun --force

PYTHONPATH=. debug_output/_tmp_phasecd_min_ablation_20260330/cpu_nomps_exec.py tools/phasea_group_summary.py \
  debug_output/_tmp_dsn_aux_leg_matched_chain_20260410/stage6/late_attach_aux/stage6_freerun/Walk_F_freerun_cycles.json \
  --cycle_gte 1 --drop_wrap \
  --out debug_output/_tmp_dsn_aux_leg_matched_chain_20260410/stage6/late_attach_aux/stage6_group_summary.json
```

## 5. Artifacts

### 5.1 Train artifacts

- `models/__tmp_dsn_aux_leg_matched_chain_20260410/stage6/late_attach_sham/posttrain_log_lr3e4_e8x60_wd1e4_reinit1_stage6_tailfix_dsn_lateattach_sham_20260410.json`
- `models/__tmp_dsn_aux_leg_matched_chain_20260410/stage6/late_attach_sham/ckpt_last_train_lr3e4_e8x60_wd1e4_reinit1_stage6_tailfix_dsn_lateattach_sham_20260410.pth`
- `models/__tmp_dsn_aux_leg_matched_chain_20260410/stage6/late_attach_sham/ckpt_last_lr3e4_e8x60_wd1e4_reinit1_stage6_tailfix_dsn_lateattach_sham_20260410.pth`

- `models/__tmp_dsn_aux_leg_matched_chain_20260410/stage6/late_attach_aux/posttrain_log_lr3e4_e8x60_wd1e4_reinit1_stage6_tailfix_dsn_lateattach_auxw02_20260410.json`
- `models/__tmp_dsn_aux_leg_matched_chain_20260410/stage6/late_attach_aux/ckpt_last_train_lr3e4_e8x60_wd1e4_reinit1_stage6_tailfix_dsn_lateattach_auxw02_20260410.pth`
- `models/__tmp_dsn_aux_leg_matched_chain_20260410/stage6/late_attach_aux/ckpt_last_lr3e4_e8x60_wd1e4_reinit1_stage6_tailfix_dsn_lateattach_auxw02_20260410.pth`

### 5.2 Eval artifacts

- `debug_output/_tmp_dsn_aux_leg_matched_chain_20260410/stage6/late_attach_sham/stage6_freerun/Walk_F_freerun_cycles.json`
- `debug_output/_tmp_dsn_aux_leg_matched_chain_20260410/stage6/late_attach_sham/stage6_group_summary.json`
- `debug_output/_tmp_dsn_aux_leg_matched_chain_20260410/stage6/late_attach_aux/stage6_freerun/Walk_F_freerun_cycles.json`
- `debug_output/_tmp_dsn_aux_leg_matched_chain_20260410/stage6/late_attach_aux/stage6_group_summary.json`

## 6. Results

## 6.1 Train-time aux readability

### `aux_leg_loss`

| arm | first | last | mean | first60 mean | last60 mean | first60→last60 drop | rel drop |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `shared_attach_sham` | `0.168736` | `0.169967` | `0.173757` | `0.174277` | `0.173296` | `0.000981` | `0.56%` |
| `shared_attach_aux` | `0.168736` | `0.040644` | `0.085204` | `0.150498` | `0.054306` | `0.096192` | `63.92%` |
| `late_attach_sham` | `0.168736` | `0.169967` | `0.173757` | `0.174277` | `0.173296` | `0.000981` | `0.56%` |
| `late_attach_aux` | `0.168736` | `0.094807` | `0.112318` | `0.152949` | `0.094440` | `0.058509` | `38.25%` |
| `E1_aux_detach` | `0.168736` | `0.041369` | `0.085561` | `0.150569` | `0.054265` | `0.096305` | `63.96%` |
| `E2_frozen_trunk` | `0.168736` | `0.120766` | `0.140991` | `0.156868` | `0.133332` | `0.023536` | `15.00%` |

### `aux_leg_loss_weighted`

| arm | first | last | mean | first60 mean | last60 mean | first60→last60 drop | rel drop |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `shared_attach_sham` | `0.000000` | `0.000000` | `0.000000` | `0.000000` | `0.000000` | `0.000000` | `0.00%` |
| `shared_attach_aux` | `0.033747` | `0.008129` | `0.017041` | `0.030100` | `0.010861` | `0.019238` | `63.92%` |
| `late_attach_sham` | `0.000000` | `0.000000` | `0.000000` | `0.000000` | `0.000000` | `0.000000` | `0.00%` |
| `late_attach_aux` | `0.033747` | `0.018961` | `0.022464` | `0.030590` | `0.018888` | `0.011702` | `38.25%` |
| `E1_aux_detach` | `0.033747` | `0.008274` | `0.017112` | `0.030114` | `0.010853` | `0.019261` | `63.96%` |
| `E2_frozen_trunk` | `0.033747` | `0.024153` | `0.028198` | `0.031374` | `0.026666` | `0.004707` | `15.00%` |

### `aux_leg_over_main`

| arm | first | last | mean | first60 mean | last60 mean | first60→last60 drop | rel drop |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `shared_attach_sham` | `0.084368` | `0.094620` | `0.102208` | `0.120731` | `0.090809` | `0.029922` | `24.78%` |
| `shared_attach_aux` | `0.084368` | `0.021403` | `0.051911` | `0.103429` | `0.028285` | `0.075144` | `72.65%` |
| `late_attach_sham` | `0.084368` | `0.094620` | `0.102208` | `0.120731` | `0.090809` | `0.029922` | `24.78%` |
| `late_attach_aux` | `0.084368` | `0.047919` | `0.067277` | `0.105151` | `0.049972` | `0.055178` | `52.48%` |
| `E1_aux_detach` | `0.084368` | `0.022092` | `0.052184` | `0.103462` | `0.028465` | `0.074997` | `72.49%` |
| `E2_frozen_trunk` | `0.084368` | `0.070249` | `0.079971` | `0.100284` | `0.070091` | `0.030193` | `30.11%` |

### Readability takeaways

Key E3 comparison:

- `late_attach_aux` vs `late_attach_sham`
  - `aux_leg_loss last60 mean`: `0.173296 -> 0.094440` (`-0.078856`)
  - `aux_leg_loss mean`: `0.173757 -> 0.112318` (`-0.061439`)
  - `aux_leg_over_main last60 mean`: `0.090809 -> 0.049972` (`-0.040836`)

So **train-time readability does improve clearly** at `leg_boundary`.

But it is still weaker than shared-attach `aux`:

- `late_attach_aux` vs `shared_attach_aux`
  - `aux_leg_loss last60 mean`: `0.094440` vs `0.054306` (`+0.040134`)
  - `aux_leg_loss mean`: `0.112318` vs `0.085204` (`+0.027114`)
  - `aux_leg_over_main last60 mean`: `0.049972` vs `0.028285` (`+0.021688`)

So `leg_boundary` is **more readable than sham/E2**, but **not better than normal shared-attach aux training**.

## 6.2 Stage6 free-run eval / group summary

Current `phasea_group_summary` uses the same value for `DirectGeoLocalDeg` and `all_ex_root`, so the table reports them together.

### Mean

| arm | DirectGeoLocalDeg / all_ex_root | leg | nonleg | arm | else |
| --- | ---: | ---: | ---: | ---: | ---: |
| `shared_attach_sham` | `0.237614` | `0.598018` | `0.159688` | `0.175657` | `0.121944` |
| `shared_attach_aux` | `0.268797` | `0.708246` | `0.173780` | `0.194554` | `0.124680` |
| `late_attach_sham` | `0.237614` | `0.598018` | `0.159688` | `0.175657` | `0.121944` |
| `late_attach_aux` | `0.251461` | `0.660008` | `0.163126` | `0.183973` | `0.113853` |
| `E1_aux_detach` | `0.262125` | `0.597226` | `0.189670` | `0.220076` | `0.117801` |

### p95

| arm | DirectGeoLocalDeg / all_ex_root p95 | leg p95 | nonleg p95 | arm p95 | else p95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `shared_attach_sham` | `0.877658` | `1.385009` | `0.547003` | `0.627461` | `0.318261` |
| `shared_attach_aux` | `0.990646` | `1.793149` | `0.596016` | `0.723004` | `0.355787` |
| `late_attach_sham` | `0.877658` | `1.385009` | `0.547003` | `0.627461` | `0.318261` |
| `late_attach_aux` | `0.913264` | `1.438880` | `0.553547` | `0.631515` | `0.319612` |
| `E1_aux_detach` | `0.937262` | `1.316565` | `0.699526` | `0.801407` | `0.344549` |

### Key deltas

`shared_attach_aux - shared_attach_sham`:

- mean:
  - `all_ex_root`: `+0.031183`
  - `leg`: `+0.110228`
- p95:
  - `all_ex_root`: `+0.112988`
  - `leg`: `+0.408140`

`late_attach_aux - late_attach_sham`:

- mean:
  - `all_ex_root`: `+0.013847`
  - `leg`: `+0.061990`
- p95:
  - `all_ex_root`: `+0.035607`
  - `leg`: `+0.053871`

So `late_attach_aux` is still **worse than `late_attach_sham`** on the target eval readout.

However, relative to shared-attach `aux`, late attach does reduce the harm magnitude:

- `all_ex_root mean` harm reduced by `0.017336` (`55.6%`)
- `all_ex_root p95` harm reduced by `0.077381` (`68.5%`)
- `leg mean` harm reduced by `0.048238` (`43.8%`)
- `leg p95` harm reduced by `0.354269` (`86.8%`)

So attach location matters, but the late attach still does **not** flip the result into a net win.

## 7. Decision

### Q1. `late_attach_aux` relative to `late_attach_sham`: does aux readability show a clear net gain?

**Yes.**

The clearest readout is unweighted `aux_leg_loss`:

- `last60 mean`: `0.173296 -> 0.094440`
- `mean`: `0.173757 -> 0.112318`
- relative drop: `0.56% -> 38.25%`

So `leg_boundary` is not unreadable; it contains usable leg-predictive signal.

### Q2. `late_attach_aux` relative to `late_attach_sham`: does `leg` / `all_ex_root` show a clear net gain?

**No.**

On stage6 free-run, `late_attach_aux` is still worse than `late_attach_sham`:

- `all_ex_root mean`: `0.237614 -> 0.251461` (`+0.013847`)
- `all_ex_root p95`: `0.877658 -> 0.913264` (`+0.035607`)
- `leg mean`: `0.598018 -> 0.660008` (`+0.061990`)
- `leg p95`: `1.385009 -> 1.438880` (`+0.053871`)

So `late_attach` **does not work** in the decision-rule sense required to support a clean `attach mismatch` explanation.

### Q3. Is this enough to support `attach mismatch` or `capacity saturation / no usable signal`?

Primary answer: **No clean support for either as the main explanation.**

What E3 actually says is:

- `leg_boundary` does have usable leg-readable signal
  - so this is **not** a pure `no usable signal` story
- but moving the attach later still fails to produce a net stage6 gain
  - so this is **not** a clean `attach mismatch` rescue either

The more precise reading is:

- attach location **modulates the harm magnitude**
- but the deeper issue is that aux-gradient-driven updates in the current direct pipeline are still not aligned with better stage6 behavior

### Q4. Putting E1 + E2 + E3 together, what is the fuller problem direction?

#### Primary conclusion

- `gradient conflict / redundancy`

Reason:

- `E1`: `aux_detach ≈ sham` on `leg`  
  → the major harm depends on aux gradient entering the trainable shared/direct pipeline
- `E2`: frozen shared hidden is only partially readable  
  → there is some signal, but not enough to justify “just read what is already there”
- `E3`: later leg-boundary attach improves readability and reduces harm magnitude, but still remains worse than sham  
  → the issue is not merely “wrong tap”, but what the aux gradient does to the trainable pipeline

#### Secondary retained explanation

- `structural fork / head-side competition`

Reason:

- late attach reduces the shared-attach injury substantially, especially `leg p95`
- but still does not become beneficial
- this is consistent with a head-side / direct-branch competition effect: once aux is allowed to backprop through current direct-pose machinery, it still pushes updates that are only partly compatible with the main rollout objective

#### Tertiary / weaker retained explanation

- `attach mismatch`

Reason:

- attach location clearly changes injury magnitude
- but not enough to rescue the arm into a net win

So `attach mismatch` is better read as a **secondary modulator**, not the main cause.

#### De-emphasized explanation

- `capacity saturation / no usable signal`

Reason:

- `late_attach_aux` readability is clearly above sham
- `E2` frozen-trunk readability is also above sham

So the pipeline is **not** devoid of leg-readable signal. The problem is more about **how the gradient uses the pipeline**, not the total absence of signal.

### Q5. Is `E4 downstream confirmation` still worth doing?

**No, not on the current E3 arm.**

Reason:

- the plan explicitly recommends pushing only a stage6 arm that gives a **clear surviving mechanism signal**
- `late_attach_aux` does **not** beat `late_attach_sham` on `leg` / `all_ex_root`
- there is no positive E3 survivor worth promoting into `70a -> 70b`

So pushing this E3 arm downstream would mostly spend budget confirming a stage6-negative result.  
If there is a next round, it should target the now-better-supported direction:

- suppressing `gradient conflict / redundancy`
- or isolating `head-side competition`

rather than confirming this late-attach arm downstream.

## 8. Final concise summary

`E3 late_attach_probe` yields a mixed but still useful result:

- `late_attach_aux` **does** gain clear train-time readability over `late_attach_sham`
- but it **does not** gain on stage6 free-run `leg` / `all_ex_root`
- therefore the result is **not enough** to support a clean `attach mismatch` explanation
- and it also **does not** support `no usable signal`

Best current synthesis:

1. **Primary**: `gradient conflict / redundancy`
2. **Secondary**: `structural fork / head-side competition`
3. **Secondary but weaker**: `attach mismatch` as a harm modulator
4. **Not primary**: `capacity saturation / no usable signal`

So after `E1 + E2 + E3`, the more complete picture is:

> the pipeline contains some leg-readable signal, and moving the tap later reduces harm, but any aux gradient entering the current direct pipeline still fails to translate into better stage6 behavior; the dominant issue is gradient-side interference / competition, not a simple missing signal or a simple wrong tap.

Post-E4 note:

- `docs/retired_directions/aux_shared_trunk_family/2026-04-11_shared_trunk_mechanism_e4_epochwise_aux_rollout_mismatch_record.md`
- E4 does **not** justify upgrading `supervision–rollout mismatch` to the primary explanation
- the stronger new E4 fact is instead the cross-arm endpoint mismatch:
  - `shared_attach_aux` vs `aux_detach` has nearly identical epoch-8 `aux_leg_loss`
  - but `leg p95` still differs by `+0.476584`
- so the E1–E4 chain should continue to treat `gradient conflict / redundancy` as the primary interpretation, with E4 adding endpoint confirmation rather than overturning that ranking
