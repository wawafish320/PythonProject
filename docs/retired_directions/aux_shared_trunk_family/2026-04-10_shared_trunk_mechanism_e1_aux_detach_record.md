# 2026-04-10 shared trunk mechanism E1 aux_detach record

> Status: archived / retired aux-family mechanism record
> Reader note: this aux / shared-trunk family did **not** become current repo mainline; any `recommend`, `default`, `ship`, `mainline`, or `current` wording below is historical family-local language only.
> Current entry points: `docs/posttrain_pipeline.md`, `docs/train_design/2026-04-11_donor_contract_basin_universal_debug_synthesis.md`, `docs/retired_directions/aux_shared_trunk_family/2026-04-11_aux_detach_downstream_reevaluation_record.md`

> Status: completed  
> Scope: `stage6 native` only; reuse historical `baseline / sham / aux`; only add one new arm `aux_detach`  
> Goal: answer whether current `DSN aux-leg` damage requires aux gradient to enter the shared trunk

## 1. Fixed scope

- Reused existing `stage6` three-arm artifacts from `docs/retired_directions/aux_shared_trunk_family/2026-04-10_dsn_auxiliary_leg_supervision_three_arm_real_training_record.md`; did **not** rerun `baseline / sham / aux`.
- This round only added:
  - `aux_detach`
  - structure present
  - aux head still forward
  - aux loss still enabled
  - only semantic change vs `aux`: `direct_pose_aux_leg_detach_feat=true`
- No code change was needed.

Runtime note:

- Current mainline marks `direct_pose_use_phase_z` / `direct_pose_phase_z_mode` as parser-dead config IO (`docs/posttrain_pipeline.md`).
- Saved `posttrain_cfg` for the reused/new runs shows these fields as `None`, so this knob is **not** a confound for E1 in current runtime.

## 2. New arm definition

Config file:

- `debug_output/_tmp_dsn_aux_leg_matched_chain_20260410/configs/posttrain_WalkF_stage6_direct_cond_anchor_splitfirst_3way_armchain_pe32h512_tailfix_lr3e4_e8x60_wd1e4_reinit1_dsn_aux_detach_20260410.json`

Effective aux-specific knobs:

| arm | `direct_pose_aux_leg_enable` | `direct_pose_aux_leg_weight` | `direct_pose_aux_leg_detach_feat` | `direct_pose_aux_leg_log_enable` |
| --- | ---: | ---: | ---: | ---: |
| `aux` | `true` | `0.2` | `false` | `true` |
| `aux_detach` | `true` | `0.2` | `true` | `true` |

## 3. Actual commands run

### 3.1 Train

```bash
PYTHONPATH=. debug_output/_tmp_phasecd_min_ablation_20260330/cpu_nomps_exec.py -m train.posttrain \
  --config debug_output/_tmp_dsn_aux_leg_matched_chain_20260410/configs/posttrain_WalkF_stage6_direct_cond_anchor_splitfirst_3way_armchain_pe32h512_tailfix_lr3e4_e8x60_wd1e4_reinit1_dsn_aux_detach_20260410.json \
  --ckpt_in models/cp015_phasecd_tailk_probe_20260331/exp_phase_DirectBranch_v1_d1_cp015_tailk7_rankmix_tw020_corridor_hold_tail15_phasea050_fixedsched_ep014center_control_denseckpt_seed2024_20260401/ckpt_epoch_014.pth \
  --out_dir models/__tmp_dsn_aux_leg_matched_chain_20260410/stage6/aux_detach \
  --run_name lr3e4_e8x60_wd1e4_reinit1_stage6_tailfix_dsn_auxdetach_20260410 \
  --posttrain_contacts_source pretrain_contact \
  --posttrain_contacts_pretrain_clamp 1.0 \
  --encoder_bundle models/motion_encoder_equiv.pt.best.pt \
  --posttrain_contacts_pretrain_affine_stats debug_output/_tmp_phaseb_affine_20260304/affine_fit_mix08/affine_stats.json
```

### 3.2 Freerun eval

```bash
PYTHONPATH=. debug_output/_tmp_phasecd_min_ablation_20260330/cpu_nomps_exec.py -m train.validate.run_freerun_cycles \
  --teacher validate/teacher_batches/Walk_F_teacher.json \
  --model models/__tmp_dsn_aux_leg_matched_chain_20260410/stage6/aux_detach/ckpt_last_lr3e4_e8x60_wd1e4_reinit1_stage6_tailfix_dsn_auxdetach_20260410.pth \
  --rounds 5 --depth 3 --time-index-mode cycle \
  --phase_reset_source none \
  --contacts_meas_source pretrain_contact \
  --contacts_meas_pretrain_clamp 1.0 \
  --contacts_meas_pretrain_affine_stats debug_output/_tmp_phaseb_affine_20260304/affine_fit_mix08/affine_stats.json \
  --encoder-bundle models/motion_encoder_equiv.pt.best.pt \
  --export_joint_direct_geolocal_series \
  --out debug_output/_tmp_dsn_aux_leg_matched_chain_20260410/stage6/aux_detach/stage6_freerun \
  --force
```

### 3.3 Group summary

```bash
PYTHONPATH=. debug_output/_tmp_phasecd_min_ablation_20260330/cpu_nomps_exec.py \
  tools/phasea_group_summary.py \
  debug_output/_tmp_dsn_aux_leg_matched_chain_20260410/stage6/aux_detach/stage6_freerun/Walk_F_freerun_cycles.json \
  --cycle_gte 1 \
  --drop_wrap \
  --out debug_output/_tmp_dsn_aux_leg_matched_chain_20260410/stage6/aux_detach/stage6_group_summary.json
```

## 4. Run success

| item | status |
| --- | --- |
| `aux_detach train` | `ok` |
| `aux_detach eval` | `ok` |
| `aux_detach group summary` | `ok` |

Artifacts:

- train ckpt: `models/__tmp_dsn_aux_leg_matched_chain_20260410/stage6/aux_detach/ckpt_last_train_lr3e4_e8x60_wd1e4_reinit1_stage6_tailfix_dsn_auxdetach_20260410.pth`
- handoff ckpt: `models/__tmp_dsn_aux_leg_matched_chain_20260410/stage6/aux_detach/ckpt_last_lr3e4_e8x60_wd1e4_reinit1_stage6_tailfix_dsn_auxdetach_20260410.pth`
- train log: `models/__tmp_dsn_aux_leg_matched_chain_20260410/stage6/aux_detach/posttrain_log_lr3e4_e8x60_wd1e4_reinit1_stage6_tailfix_dsn_auxdetach_20260410.json`
- eval json: `debug_output/_tmp_dsn_aux_leg_matched_chain_20260410/stage6/aux_detach/stage6_freerun/Walk_F_freerun_cycles.json`
- summary json: `debug_output/_tmp_dsn_aux_leg_matched_chain_20260410/stage6/aux_detach/stage6_group_summary.json`

## 5. Four-arm stage6 metrics

### 5.1 Mean metrics

| arm | `DirectGeoLocalDeg` | `all_ex_root` | `leg` | `nonleg` | `arm` | `else` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `baseline` | `0.250873` | `0.250873` | `0.566078` | `0.182721` | `0.201131` | `0.139205` |
| `sham` | `0.237614` | `0.237614` | `0.598018` | `0.159688` | `0.175657` | `0.121944` |
| `aux` | `0.268797` | `0.268797` | `0.708246` | `0.173780` | `0.194554` | `0.124680` |
| `aux_detach` | `0.262125` | `0.262125` | `0.597226` | `0.189670` | `0.220076` | `0.117801` |

### 5.2 p95 readout

| arm | `all_ex_root p95` | `leg p95` | `nonleg p95` | `arm p95` | `else p95` |
| --- | ---: | ---: | ---: | ---: | ---: |
| `baseline` | `0.836182` | `1.324971` | `0.604531` | `0.687248` | `0.351424` |
| `sham` | `0.877658` | `1.385009` | `0.547003` | `0.627461` | `0.318261` |
| `aux` | `0.990646` | `1.793149` | `0.596016` | `0.723004` | `0.355787` |
| `aux_detach` | `0.937262` | `1.316565` | `0.699526` | `0.801407` | `0.344549` |

### 5.3 Train-time aux telemetry

Mean over full train log:

| arm | `aux_leg_weight` | `aux_leg_loss` | `aux_leg_loss_weighted` | `aux_leg_over_main` |
| --- | ---: | ---: | ---: | ---: |
| `baseline` | `0.000000` | `0.000000` | `0.000000` | `0.000000` |
| `sham` | `0.000000` | `0.173757` | `0.000000` | `0.102208` |
| `aux` | `0.200000` | `0.085204` | `0.017041` | `0.051911` |
| `aux_detach` | `0.200000` | `0.085561` | `0.017112` | `0.052184` |

Key observation:

- `aux` and `aux_detach` have nearly identical train-time aux telemetry.
- So `detach` did **not** disable the aux task itself; it mainly blocked aux gradient from entering the shared trunk.

## 6. Readout

### Q1. `aux_detach` 更接近 `sham` 还是 `aux`？

For the target injury metric `leg`, `aux_detach` is effectively **`sham`-like**:

- `leg mean`: `0.597226` vs `sham=0.598018` vs `aux=0.708246`
- `leg p95`: `1.316565` vs `sham=1.385009` vs `aux=1.793149`

Distance check:

- 6-metric mean L1 distance:
  - `aux_detach -> sham = 0.128358`
  - `aux_detach -> aux = 0.172655`
- p95 L1 distance:
  - `aux_detach -> sham = 0.480805`
  - `aux_detach -> aux = 0.723118`

So on the full readout it is also closer to `sham` than to `aux`, though not every scalar fully returns to sham.

### Q2. `aux_detach` 是否显著缓解了 `aux` 在 `leg` 上的伤害？

Yes.

- `leg mean`: `0.708246 -> 0.597226`, improvement `-0.111020`
- `leg p95`: `1.793149 -> 1.316565`, improvement `-0.476584`

Relative to `baseline`:

- `aux` leg mean harm: `+0.142168`
- `aux_detach` leg mean harm: `+0.031147`
- `detach` removes about **78.1%** of the mean leg-side extra harm

For `leg p95`:

- `aux` harm vs baseline: `+0.468178`
- `aux_detach` vs baseline: `-0.008405`
- `detach` removes **>100%** of the p95 leg-side extra harm

### Q3. 这是否说明主要伤害来自 aux gradient 进入 shared trunk？

Yes, for the **leg-side damage** this is the strongest reading.

Reason:

- `aux` and `aux_detach` keep the same aux structure, same aux head, same aux weight, same aux telemetry
- once aux gradient is detached from the shared trunk, the leg collapse largely disappears

So the harmful effect does **not** require turning the aux task off; it mainly requires preventing its gradient from flowing into the shared trunk.

### Q4. 更支持哪一类机制？

Primary support:

- **gradient conflict / redundancy / attach mismatch**

Not primarily supported:

- **structural fork / head-side competition**

Why:

- If pure extra-head structure or head-side competition were the main cause, `aux_detach` should have stayed near `aux`
- Instead, the main leg injury almost fully collapses back to `sham` once trunk-directed aux gradient is removed

Nuance:

- `aux_detach` does **not** fully recover all metrics:
  - `DirectGeoLocalDeg` remains above baseline
  - `nonleg / arm` mean and p95 even worsen vs `sham`
- So there may still be some secondary structure-side or optimizer-side perturbation
- But the **main leg-side damage signature** is much more consistent with trunk-directed gradient interference than with pure extra-head structure

## 7. Bottom line

E1 says the current `DSN aux-leg` failure mode is **not** just “having an extra leg head”.

The damaging part is primarily:

- letting aux supervision backprop into the shared trunk

Therefore the result supports:

- shared-trunk gradient conflict / redundancy / attach mismatch

and does **not** support taking “head exists, therefore it hurts” as the main explanation.
