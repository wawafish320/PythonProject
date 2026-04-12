# 2026-04-10 shared trunk mechanism E2 frozen trunk aux readability record

> Status: archived / retired aux-family mechanism record
> Reader note: this aux / shared-trunk family did **not** become current repo mainline; any `recommend`, `default`, `ship`, `mainline`, or `current` wording below is historical family-local language only.
> Current entry points: `docs/posttrain_pipeline.md`, `docs/train_design/2026-04-11_donor_contract_basin_universal_debug_synthesis.md`, `docs/retired_directions/aux_shared_trunk_family/2026-04-11_aux_detach_downstream_reevaluation_record.md`

> Status: completed  
> Scope: `stage6 native` train-only readability probe; no downstream `70a / 70b`, no extra sweep  
> Goal: test whether current `shared trunk hidden` is already leg-readable when the shared trunk itself is frozen

## 1. Fixed scope

- This round only runs the minimal `E2 frozen_trunk_aux_readability`.
- No code change.
- One new config-only arm:
  - keep aux structure and aux loss on
  - keep `direct_pose_aux_leg_detach_feat=false`
  - freeze `direct_pose_head.*` by `optimizer_param_group_overrides`
- Readout priority follows the plan:
  - look at `aux_leg_loss` descent
  - compare against existing `aux` and `sham`
  - do **not** expand into downstream chain this round

Reference plan:

- `docs/retired_directions/aux_shared_trunk_family/2026-04-10_shared_trunk_mechanism_disambiguation_plan.md`

## 2. New arm definition

Config file:

- `debug_output/_tmp_dsn_aux_leg_matched_chain_20260410/configs/posttrain_WalkF_stage6_direct_cond_anchor_splitfirst_3way_armchain_pe32h512_tailfix_lr3e4_e8x60_wd1e4_reinit1_frozen_trunk_aux_readability_20260410.json`

Allowed diff vs normal `aux`:

| knob | `aux` | `E2 frozen_trunk_aux_readability` |
| --- | --- | --- |
| `direct_pose_aux_leg_enable` | `true` | `true` |
| `direct_pose_aux_leg_weight` | `0.2` | `0.2` |
| `direct_pose_aux_leg_detach_feat` | `false` | `false` |
| `direct_pose_aux_leg_log_enable` | `true` | `true` |
| `optimizer_param_group_overrides` | `none` | freeze `direct_pose_head` with `lr=0.0` |

## 3. Actual command run

```bash
PYTHONPATH=. debug_output/_tmp_phasecd_min_ablation_20260330/cpu_nomps_exec.py -m train.posttrain \
  --config debug_output/_tmp_dsn_aux_leg_matched_chain_20260410/configs/posttrain_WalkF_stage6_direct_cond_anchor_splitfirst_3way_armchain_pe32h512_tailfix_lr3e4_e8x60_wd1e4_reinit1_frozen_trunk_aux_readability_20260410.json \
  --ckpt_in models/cp015_phasecd_tailk_probe_20260331/exp_phase_DirectBranch_v1_d1_cp015_tailk7_rankmix_tw020_corridor_hold_tail15_phasea050_fixedsched_ep014center_control_denseckpt_seed2024_20260401/ckpt_epoch_014.pth \
  --out_dir models/__tmp_dsn_aux_leg_matched_chain_20260410/stage6/frozen_trunk_aux_readability \
  --run_name lr3e4_e8x60_wd1e4_reinit1_stage6_tailfix_dsn_e2_frozentrunkaux_20260410 \
  --posttrain_contacts_source pretrain_contact \
  --posttrain_contacts_pretrain_clamp 1.0 \
  --encoder_bundle models/motion_encoder_equiv.pt.best.pt \
  --posttrain_contacts_pretrain_affine_stats debug_output/_tmp_phaseb_affine_20260304/affine_fit_mix08/affine_stats.json
```

## 4. Runtime confirmation

Observed optimizer grouping from runtime stdout:

- `freeze_shared_trunk`:
  - `lr=0.00e+00`
  - `prefixes=direct_pose_head`
  - `tensors=4`
  - `params=285184`
- `default`:
  - `lr=3.00e-04`
  - `tensors=18`
  - `params=668510`

So this was not a detach trick; it was a true frozen-trunk probe.

Artifacts:

- train ckpt: `models/__tmp_dsn_aux_leg_matched_chain_20260410/stage6/frozen_trunk_aux_readability/ckpt_last_train_lr3e4_e8x60_wd1e4_reinit1_stage6_tailfix_dsn_e2_frozentrunkaux_20260410.pth`
- handoff ckpt: `models/__tmp_dsn_aux_leg_matched_chain_20260410/stage6/frozen_trunk_aux_readability/ckpt_last_lr3e4_e8x60_wd1e4_reinit1_stage6_tailfix_dsn_e2_frozentrunkaux_20260410.pth`
- train log: `models/__tmp_dsn_aux_leg_matched_chain_20260410/stage6/frozen_trunk_aux_readability/posttrain_log_lr3e4_e8x60_wd1e4_reinit1_stage6_tailfix_dsn_e2_frozentrunkaux_20260410.json`

## 5. Aux readability results

### 5.1 Core aux-loss table

| arm | `aux_leg_loss` first | `aux_leg_loss` last | mean | first60 mean | last60 mean | first60->last60 drop | relative drop |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `sham` | `0.168736` | `0.169967` | `0.173757` | `0.174277` | `0.173296` | `0.000981` | `0.56%` |
| `aux` | `0.168736` | `0.040644` | `0.085204` | `0.150498` | `0.054306` | `0.096192` | `63.92%` |
| `E2 frozen_trunk` | `0.168736` | `0.120766` | `0.140991` | `0.156868` | `0.133332` | `0.023536` | `15.00%` |

### 5.2 Epoch means

| arm | epoch means of `aux_leg_loss` |
| --- | --- |
| `sham` | `0.174277, 0.172826, 0.174359, 0.173325, 0.175293, 0.174074, 0.172603, 0.173296` |
| `aux` | `0.150498, 0.110540, 0.090446, 0.078847, 0.072849, 0.065077, 0.059068, 0.054306` |
| `E2 frozen_trunk` | `0.156868, 0.144588, 0.143045, 0.138826, 0.139752, 0.137505, 0.134010, 0.133332` |

### 5.3 Relative position

- `E2 last60 mean - aux last60 mean = +0.079026`
- `E2 last60 mean - sham last60 mean = -0.039964`
- `E2 mean - aux mean = +0.055787`
- `E2 mean - sham mean = -0.032766`

Interpretation:

- `E2` is **better than sham**, so frozen shared hidden is **not completely unreadable**
- but `E2` is **much weaker than normal aux**
- the descent speed and final level are nowhere near the normal `aux` arm

## 6. Decision

### Q1. 冻结 trunk 后，aux head 还能把 aux loss 明显降下来吗？

Yes, but only **partially**.

- vs `sham`, `E2` does learn:
  - `last60 mean: 0.173296 -> 0.133332`
  - this is real readability above sham/no-learning level
- but vs normal `aux`, it is much weaker:
  - `last60 mean: 0.133332` vs `0.054306`
  - relative drop: `15.00%` vs `63.92%`

So the answer is:

- there is **some** leg-readable signal in the current frozen shared trunk hidden
- but it is **not enough** to support the strong claim that the hidden is already richly leg-readable

### Q2. 这更支持哪边？

This result does **not** support a clean “already enough readable signal; only gradient conflict” interpretation.

Instead it supports the weaker, more nuanced reading:

- frozen shared hidden contains **partial** readable leg signal
- but normal `aux` learns much more once trunk is allowed to move

So E2 pushes the ambiguity toward:

- `attach mismatch`
- and/or `capacity saturation / not enough usable signal at current attach point`

rather than a pure:

- `shared hidden already fully readable, only gradient conflict`

### Q3. 和 E1 合起来怎么读？

`E1` said:

- the major leg-side damage depends on aux gradient entering the shared trunk

`E2` now adds:

- once the shared trunk is frozen, aux head can only recover a **limited** amount of readability

Combined reading:

- trunk-directed aux gradient is indeed the main **damage path**
- but that gradient is probably also the route by which aux tries to create more leg-readable trunk features
- and under the current attach point / objective coupling, that adaptation is harmful to the main path

So the combined story is most consistent with:

- **gradient conflict on an attach point that is not sufficiently leg-ready**

more than either extreme:

- pure structure-only competition
- or fully readable hidden with pure benign redundancy

## 7. Bottom line

I do **not** read E2 as a success for “shared trunk hidden is already enough”.

I read it as:

- `some readable signal exists`
- `but clearly weaker than normal aux`
- therefore current attach point still looks **under-leg-readable**
- and the remaining ambiguity is now more concentrated on:
  - `attach mismatch`
  - versus broader `usable-signal/capacity` limitation

That means:

- E2 does **not** invalidate E1
- but it does say the next disambiguation, if we continue, should be attach-oriented rather than another generic auxiliary-loss sweep
