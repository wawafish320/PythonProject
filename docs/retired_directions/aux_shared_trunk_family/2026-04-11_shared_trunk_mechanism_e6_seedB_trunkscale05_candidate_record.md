# E6 staged seed-B trunkscale05 candidate validation

> Status: archived / retired aux-family mechanism record
> Reader note: this aux / shared-trunk family did **not** become current repo mainline; any `recommend`, `default`, `ship`, `mainline`, or `current` wording below is historical family-local language only.
> Current entry points: `docs/posttrain_pipeline.md`, `docs/train_design/2026-04-11_donor_contract_basin_universal_debug_synthesis.md`, `docs/retired_directions/aux_shared_trunk_family/2026-04-11_aux_detach_downstream_reevaluation_record.md`

Date: 2026-04-11

## 1. Question and pre-registered rule

This round answers one practical question only:

> under `seed B`, can `shared_attach_aux` with shared-trunk-only aux-grad scale
> `0.5` become a more worthwhile Stage6 candidate than `aux_detach`?

Scope constraints followed:

- reuse E1 / E5a / E5b / E5d conclusions; no re-judging mechanism hierarchy
- do **Stage A first**:
  - `seed = 2025`
  - `direct_pose_aux_leg_attach = shared_trunk`
  - `direct_pose_aux_leg_trunk_grad_scale = 0.5`
- only run **Stage B (`seed-B aux_detach`)** if Stage A lands in the pre-registered gray zone
- no scale sweep, no probe rerun, no downstream, no new recipe changes

Primary metric:

- `all_ex_root p95`

Safety constraint:

- `leg p95`

Pre-registered decision rule:

- **promote** if `all_ex_root p95 <= 0.93` and `leg p95 <= 1.38`
- **reject** if `all_ex_root p95 > 0.95` or `leg p95 > 1.45`
- otherwise **gray zone**, and only then run Stage B (`seed-B aux_detach`)

## 2. Artifact check and staged execution

I first checked for pre-existing E6 artifacts.

Actual search command:

```bash
find models/__tmp_dsn_aux_leg_matched_chain_20260411/stage6 -maxdepth 2 \
  \( -iname '*trunkscale05*seedB*' -o -iname '*aux_detach*seedB*' \) | sort
```

Result:

- no existing `shared_attach_aux_trunkscale05_epochsnap_seedB` artifact
- no existing `aux_detach_epochsnap_seedB` artifact

Therefore Stage A required one new matched run.

## 3. Fixed recipe and actual knob

Matched chain kept fixed:

- `stage6 native`
- donor / base config / contacts recipe matched E4 / E5a / E5d chain
- `8 epochs × 60 steps`
- `save_step_ckpts = 0,60,120,180,240,300,360,420,480`
- `seed B = 2025`

Stage A-only change:

- `direct_pose_aux_leg_attach = shared_trunk`
- `direct_pose_aux_leg_trunk_grad_scale = 0.5`

No other objective / attach / donor / contacts / teacher changes were introduced.

## 4. Actual commands run

### 4.1 Stage A train

```bash
PYTHONPATH=. debug_output/_tmp_phasecd_min_ablation_20260330/cpu_nomps_exec.py -m train.posttrain \
  --config debug_output/_tmp_dsn_aux_leg_matched_chain_20260411/configs/shared_attach_aux_trunkscale05_epochsnap_seedB.json \
  --ckpt_in models/cp015_phasecd_tailk_probe_20260331/exp_phase_DirectBranch_v1_d1_cp015_tailk7_rankmix_tw020_corridor_hold_tail15_phasea050_fixedsched_ep014center_control_denseckpt_seed2024_20260401/ckpt_epoch_014.pth \
  --out_dir models/__tmp_dsn_aux_leg_matched_chain_20260411/stage6/shared_attach_aux_trunkscale05_epochsnap_seedB \
  --run_name lr3e4_e8x60_wd1e4_reinit1_stage6_tailfix_dsn_sharedaux_trunkscale05_epochsnap_seed2025_20260411 \
  --posttrain_contacts_source pretrain_contact \
  --posttrain_contacts_pretrain_clamp 1.0 \
  --encoder_bundle models/motion_encoder_equiv.pt.best.pt \
  --posttrain_contacts_pretrain_affine_stats debug_output/_tmp_phaseb_affine_20260304/affine_fit_mix08/affine_stats.json
```

### 4.2 Stage A final endpoint eval

```bash
mkdir -p debug_output/_tmp_dsn_aux_leg_matched_chain_20260411/e6_candidate/shared_attach_aux_trunkscale05_seedB && \
PYTHONPATH=. debug_output/_tmp_phasecd_min_ablation_20260330/cpu_nomps_exec.py -m train.validate.run_freerun_cycles \
  --teacher validate/teacher_batches/Walk_F_teacher.json \
  --model models/__tmp_dsn_aux_leg_matched_chain_20260411/stage6/shared_attach_aux_trunkscale05_epochsnap_seedB/ckpt_step_000480_lr3e4_e8x60_wd1e4_reinit1_stage6_tailfix_dsn_sharedaux_trunkscale05_epochsnap_seed2025_20260411.pth \
  --rounds 5 --depth 3 --time-index-mode cycle \
  --phase_reset_source none \
  --contacts_meas_source pretrain_contact \
  --contacts_meas_pretrain_clamp 1.0 \
  --contacts_meas_pretrain_affine_stats debug_output/_tmp_phaseb_affine_20260304/affine_fit_mix08/affine_stats.json \
  --encoder-bundle models/motion_encoder_equiv.pt.best.pt \
  --export_joint_direct_geolocal_series \
  --out debug_output/_tmp_dsn_aux_leg_matched_chain_20260411/e6_candidate/shared_attach_aux_trunkscale05_seedB \
  --force && \
PYTHONPATH=. debug_output/_tmp_phasecd_min_ablation_20260330/cpu_nomps_exec.py \
  tools/phasea_group_summary.py \
  debug_output/_tmp_dsn_aux_leg_matched_chain_20260411/e6_candidate/shared_attach_aux_trunkscale05_seedB/Walk_F_freerun_cycles.json \
  --cycle_gte 1 --drop_wrap \
  --out debug_output/_tmp_dsn_aux_leg_matched_chain_20260411/e6_candidate/shared_attach_aux_trunkscale05_seedB/group_summary.json
```

Stage B:

- **not triggered**
- no `seed-B aux_detach` train was run

## 5. Primary artifacts

New config:

- `debug_output/_tmp_dsn_aux_leg_matched_chain_20260411/configs/shared_attach_aux_trunkscale05_epochsnap_seedB.json`

New Stage A model root:

- `models/__tmp_dsn_aux_leg_matched_chain_20260411/stage6/shared_attach_aux_trunkscale05_epochsnap_seedB`

Stage A complete artifacts:

- train log: `models/__tmp_dsn_aux_leg_matched_chain_20260411/stage6/shared_attach_aux_trunkscale05_epochsnap_seedB/posttrain_log_lr3e4_e8x60_wd1e4_reinit1_stage6_tailfix_dsn_sharedaux_trunkscale05_epochsnap_seed2025_20260411.json`
- final ckpt: `models/__tmp_dsn_aux_leg_matched_chain_20260411/stage6/shared_attach_aux_trunkscale05_epochsnap_seedB/ckpt_last_lr3e4_e8x60_wd1e4_reinit1_stage6_tailfix_dsn_sharedaux_trunkscale05_epochsnap_seed2025_20260411.pth`
- final endpoint ckpt used for eval: `models/__tmp_dsn_aux_leg_matched_chain_20260411/stage6/shared_attach_aux_trunkscale05_epochsnap_seedB/ckpt_step_000480_lr3e4_e8x60_wd1e4_reinit1_stage6_tailfix_dsn_sharedaux_trunkscale05_epochsnap_seed2025_20260411.pth`
- final freerun json: `debug_output/_tmp_dsn_aux_leg_matched_chain_20260411/e6_candidate/shared_attach_aux_trunkscale05_seedB/Walk_F_freerun_cycles.json`
- final freerun summary: `debug_output/_tmp_dsn_aux_leg_matched_chain_20260411/e6_candidate/shared_attach_aux_trunkscale05_seedB/group_summary.json`

Reused references:

- seed-A `aux_detach` proxy reference from E1 / E4 endpoint
- seed-B `shared_attach_aux scale=1.0` within-seed bad reference from E5a endpoint

## 6. Final endpoint table

| arm | role | seed | attach / detach | trunk grad scale | aux_leg_loss (epoch 8 mean) | aux_leg_over_main (epoch 8 mean) | leg mean | leg p95 | all_ex_root mean | all_ex_root p95 |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `aux_detach` | proxy reference | `0` | `direct_pose_aux_leg_detach_feat=true` | `0.0` | `0.054265` | `0.028465` | `0.597226` | `1.316565` | `0.262125` | `0.937262` |
| `shared_attach_aux` | within-seed bad reference | `2025` | `shared_trunk` | `1.0` | `0.055731` | `0.029426` | `0.709255` | `1.669341` | `0.307925` | `1.215963` |
| `shared_attach_aux_trunkscale05` | Stage A candidate | `2025` | `shared_trunk` | `0.5` | `0.056572` | `0.028163` | `0.728391` | `1.688474` | `0.326261` | `1.167383` |
| `aux_detach` | Stage B | `2025` | `direct_pose_aux_leg_detach_feat=true` | `0.0` | not run | not run | not run | not run | not run | not run |

## 7. Readout

### 7.1 Relative to seed-B `shared_attach_aux scale=1.0`

Within seed B, `trunkscale05` shows a **weak mixed shift** relative to the bad reference:

- `all_ex_root p95`: `1.215963 -> 1.167383` (better by `0.048580`)
- `leg p95`: `1.669341 -> 1.688474` (worse by `0.019133`)
- `all_ex_root mean`: `0.307925 -> 0.326261` (worse)
- `leg mean`: `0.709255 -> 0.728391` (worse)

So within seed B, `0.5` is **not** a clear practical win over `1.0`: it helps the primary tail metric somewhat, but does not improve the leg tail and does not improve the freerun means.

### 7.2 Cross-seed baseline shift and within-seed consistency

E5a + E5d + E6 together show a large cross-seed baseline shift on `all_ex_root p95` for the shared-attach family:

| arm | scale | seed-A `all_ex_root p95` | seed-B `all_ex_root p95` | seed-B minus seed-A |
| --- | ---: | ---: | ---: | ---: |
| `shared_attach_aux` | `1.0` | `0.990646` | `1.215963` | `+0.225317` |
| `shared_attach_aux_trunkscale05` | `0.5` | `0.909538` | `1.167383` | `+0.257844` |

This cross-seed shift (`~0.23 .. 0.26`) is larger than the within-seed scale effect itself. That matters for interpretation:

- the pre-registered absolute promote gate (`all_ex_root p95 <= 0.93`) was calibrated from seed-A `aux_detach`
- that gate implicitly assumes the relevant baseline is roughly stable across seed
- current E6 data do **not** justify treating that assumption as self-evident

So the Stage-A absolute-threshold reject is still the correct **mechanical** rule outcome, but it should not be over-read as proving that `seed-B scale=0.5` is strictly worse than `detach`.

At the same time, the **within-seed direction** from E5d does replicate:

- seed A: `1.0 -> 0.5` gives `0.990646 -> 0.909538` (`-0.081108`)
- seed B: `1.0 -> 0.5` gives `1.215963 -> 1.167383` (`-0.048580`)

So the fairer read is:

- the local `0.5`-beats-`1.0` direction is replicated across both seeds
- but the effect size is modest and is dominated by cross-seed variance
- therefore E6 does **not** provide a strong cross-seed case that `0.5` should replace `detach`

### 7.3 Pre-registered decision

Stage A still hits **`reject`** under the pre-registered rule:

- `all_ex_root p95 = 1.167383 > 0.95`
- `leg p95 = 1.688474 > 1.45`

So the staged protocol outcome is unchanged:

- **decision bucket = `reject`**
- **Stage B not allowed**

This rule outcome should be read narrowly as:

- `seed-B scale=0.5` failed to clear the pre-registered absolute gate

not as:

- `seed-B scale=0.5` has been strictly falsified against `detach`

### 7.4 Stage B trigger status

Stage B was **not triggered**.

Reason:

- Stage A already landed in the pre-registered `reject` bucket
- therefore the staged rule forbids adding `seed-B aux_detach` in this round

This leaves one comparison intentionally unresolved:

- `seed-B shared_attach_aux trunkscale05` vs `seed-B aux_detach`

So E6 supports a ship decision, but not a strong claim that `0.5` is materially worse than `detach` within the same seed.

### 7.5 Practical recommendation

Final action remains:

- **ship `aux_detach` as the practical default**

The honest reason is:

- `seed-B scale=0.5` did not make a positive case for shifting away from `detach`
- best-case reading, it is only **comparable** to `detach`
- worst-case reading, it is worse
- either way, there is no shipping reason to replace the cleaner `detach` default

So the strongest supported conclusion is:

- **candidate is, at best, comparable to `detach`; no positive case for shifting away from `detach`**

## 8. Repo changes

Source-code change status:

- **no source code changes**

Files added for this round:

- config only: `debug_output/_tmp_dsn_aux_leg_matched_chain_20260411/configs/shared_attach_aux_trunkscale05_epochsnap_seedB.json`
- record only: `docs/retired_directions/aux_shared_trunk_family/2026-04-11_shared_trunk_mechanism_e6_seedB_trunkscale05_candidate_record.md`
