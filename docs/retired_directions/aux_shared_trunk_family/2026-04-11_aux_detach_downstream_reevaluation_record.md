# 2026-04-11 aux_detach downstream reevaluation record

> Status: retired negative-evidence record
> Current role: records why aux-leg supervision was not retained as a justified mainline feature
> Do not read this file as keeping the aux-family live by default.

> Scope: independent feature re-evaluation for `aux-leg supervision`  
> Decision target: whether the feature itself remains justified in the mainline pipeline  
> Fixed chain: `stage6 -> 70a -> 70b replace`  
> Fixed recipes: historical matched `70a` / `70b replace` downstream configs, historical contacts / encoder / eval recipe, original `create_replace_zerophase_warmstart(...)` warmstart logic  
> Non-goals: no new mechanism probe, no E-chain continuation, no new sweep, no objective redesign, no sham/aux rerun

## 1. Prior conclusions reused directly

This record intentionally reuses already-established points and does **not** reopen them:

- `aux_detach` is the most stable low-risk winner **inside the aux-family**
- `aux_detach` at `stage6` does **not** already show a clean net gain over `baseline`
- the original aux-leg justification was:
  - training-only auxiliary supervision
  - preserve baseline downstream contract
  - hope for downstream payoff at `70a -> 70b replace`
- historical three-arm real chain already showed:
  - `aux` did not beat `baseline`
  - but `aux_detach` downstream head-to-head had not yet been completed

So the only unresolved question for this round is:

> does `aux_detach` produce a **clean downstream win over baseline** on the real matched chain?

## 2. Artifact audit: reused vs newly added

### 2.1 Reused existing complete artifacts

The following artifacts already existed and were reused directly without rerun:

| stage | arm | train ckpt | eval json | group summary | action |
| --- | --- | --- | --- | --- | --- |
| `stage6` | `baseline` | ok | ok | ok | reused |
| `stage6` | `aux_detach` | ok | ok | ok | reused |
| `70a` | `baseline` | ok | ok | ok | reused |
| `70b replace` | `baseline` | ok | ok | ok | reused |

Reused paths:

- `models/__tmp_dsn_aux_leg_matched_chain_20260410/stage6/baseline`
- `models/__tmp_dsn_aux_leg_matched_chain_20260410/stage6/aux_detach`
- `models/__tmp_dsn_aux_leg_matched_chain_20260410/70a/baseline`
- `debug_output/_tmp_dsn_aux_leg_matched_chain_20260410/70a/baseline/eval_model_source/Walk_F_freerun_cycles.json`
- `debug_output/_tmp_dsn_aux_leg_matched_chain_20260410/70a/baseline/eval_model_source_group_summary.json`
- `models/__tmp_dsn_aux_leg_matched_chain_20260410/70b_replace/baseline`
- `debug_output/_tmp_dsn_aux_leg_matched_chain_20260410/70b_replace/baseline/eval_model_source/Walk_F_freerun_cycles.json`
- `debug_output/_tmp_dsn_aux_leg_matched_chain_20260410/70b_replace/baseline/eval_model_source_group_summary.json`

### 2.2 Newly added artifacts in this round

Only the missing `aux_detach` downstream artifacts were added:

| stage | arm | train ckpt | eval json | group summary | action |
| --- | --- | --- | --- | --- | --- |
| `70a` | `aux_detach` | ok | ok | ok | newly run |
| `70b replace` | `aux_detach` | ok | ok | ok | newly run |

New paths:

- `models/__tmp_dsn_aux_leg_matched_chain_20260410/70a/aux_detach/ckpt_last_WalkF_stage7_70a_lr3e4_dsn_auxdetach_20260410.pth`
- `models/__tmp_dsn_aux_leg_matched_chain_20260410/70a/aux_detach/posttrain_log_WalkF_stage7_70a_lr3e4_dsn_auxdetach_20260410.json`
- `debug_output/_tmp_dsn_aux_leg_matched_chain_20260410/70a/aux_detach/eval_model_source/Walk_F_freerun_cycles.json`
- `debug_output/_tmp_dsn_aux_leg_matched_chain_20260410/70a/aux_detach/eval_model_source_group_summary.json`
- `models/__tmp_dsn_aux_leg_matched_chain_20260410/70b_replace/aux_detach/warmstart/ckpt_last_70a_replace_zerophase_aux_detach_20260410.pth`
- `debug_output/_tmp_dsn_aux_leg_matched_chain_20260410/70b_replace/aux_detach/warmstart/replace_zerophase_report.json`
- `models/__tmp_dsn_aux_leg_matched_chain_20260410/70b_replace/aux_detach/ckpt_last_WalkF_stage7_70b_replace_lowdrift_e3x60_lr5e5_dsn_auxdetach_20260410.pth`
- `models/__tmp_dsn_aux_leg_matched_chain_20260410/70b_replace/aux_detach/posttrain_log_WalkF_stage7_70b_replace_lowdrift_e3x60_lr5e5_dsn_auxdetach_20260410.json`
- `debug_output/_tmp_dsn_aux_leg_matched_chain_20260410/70b_replace/aux_detach/eval_model_source/Walk_F_freerun_cycles.json`
- `debug_output/_tmp_dsn_aux_leg_matched_chain_20260410/70b_replace/aux_detach/eval_model_source_group_summary.json`

## 3. Actual commands run

Artifact discovery / reuse audit:

```bash
find debug_output/_tmp_dsn_aux_leg_matched_chain_20260410 -maxdepth 4 \( -name '*group_summary.json' -o -name '*summary.json' -o -name '*eval*.json' \) | sort
find models -path '*aux_detach*' | rg '70a|70b_replace|stage6'
```

Only missing downstream `aux_detach` stages were executed.

### 3.1 `70a` aux_detach

```bash
PYTHONPATH=. debug_output/_tmp_phasecd_min_ablation_20260330/cpu_nomps_exec.py -m train.posttrain \
  --config debug_output/_tmp_ep014center_70a_lowlr_sweep_20260328/configs/posttrain_70a_lr3e4_from_ep014center_20260328.json \
  --ckpt_in models/__tmp_dsn_aux_leg_matched_chain_20260410/stage6/aux_detach/ckpt_last_lr3e4_e8x60_wd1e4_reinit1_stage6_tailfix_dsn_auxdetach_20260410.pth \
  --out_dir models/__tmp_dsn_aux_leg_matched_chain_20260410/70a/aux_detach \
  --run_name WalkF_stage7_70a_lr3e4_dsn_auxdetach_20260410 \
  --posttrain_contacts_source pretrain_contact \
  --posttrain_contacts_pretrain_clamp 1.0 \
  --encoder_bundle models/motion_encoder_equiv.pt.best.pt \
  --posttrain_contacts_pretrain_affine_stats debug_output/_tmp_phaseb_affine_20260304/affine_fit_mix08/affine_stats.json

PYTHONPATH=. debug_output/_tmp_phasecd_min_ablation_20260330/cpu_nomps_exec.py -m train.validate.run_freerun_cycles \
  --teacher validate/teacher_batches/Walk_F_teacher.json \
  --model models/__tmp_dsn_aux_leg_matched_chain_20260410/70a/aux_detach/ckpt_last_WalkF_stage7_70a_lr3e4_dsn_auxdetach_20260410.pth \
  --rounds 5 --depth 3 --time-index-mode cycle \
  --event_clock auto \
  --phase_reset_source none \
  --contacts_meas_source model \
  --lambda_fusion_apply \
  --log_contacts \
  --export_direct_arm_probe \
  --export_joint_direct_geolocal_series \
  --out debug_output/_tmp_dsn_aux_leg_matched_chain_20260410/70a/aux_detach/eval_model_source \
  --force

PYTHONPATH=. debug_output/_tmp_phasecd_min_ablation_20260330/cpu_nomps_exec.py \
  tools/phasea_group_summary.py \
  debug_output/_tmp_dsn_aux_leg_matched_chain_20260410/70a/aux_detach/eval_model_source/Walk_F_freerun_cycles.json \
  --cycle_gte 1 \
  --drop_wrap \
  --out debug_output/_tmp_dsn_aux_leg_matched_chain_20260410/70a/aux_detach/eval_model_source_group_summary.json
```

### 3.2 `70b replace` aux_detach

```bash
python3 - <<'PY'
from pathlib import Path
from tools.run_cp015_oldplan_downstream_chain import create_replace_zerophase_warmstart
create_replace_zerophase_warmstart(
    Path('models/__tmp_dsn_aux_leg_matched_chain_20260410/70a/aux_detach/ckpt_last_WalkF_stage7_70a_lr3e4_dsn_auxdetach_20260410.pth'),
    Path('models/__tmp_dsn_aux_leg_matched_chain_20260410/70b_replace/aux_detach/warmstart/ckpt_last_70a_replace_zerophase_aux_detach_20260410.pth'),
    Path('debug_output/_tmp_dsn_aux_leg_matched_chain_20260410/70b_replace/aux_detach/warmstart/replace_zerophase_report.json'),
)
PY

PYTHONPATH=. debug_output/_tmp_phasecd_min_ablation_20260330/cpu_nomps_exec.py -m train.posttrain \
  --config debug_output/_tmp_cp015_tailk7_replace_schedule_ablation_20260402/configs/posttrain_70b_replace_lowdrift_e3x60_lr5e5_from_cp015_tailk7_70a_20260402.json \
  --ckpt_in models/__tmp_dsn_aux_leg_matched_chain_20260410/70b_replace/aux_detach/warmstart/ckpt_last_70a_replace_zerophase_aux_detach_20260410.pth \
  --out_dir models/__tmp_dsn_aux_leg_matched_chain_20260410/70b_replace/aux_detach \
  --run_name WalkF_stage7_70b_replace_lowdrift_e3x60_lr5e5_dsn_auxdetach_20260410 \
  --posttrain_contacts_source pretrain_contact \
  --posttrain_contacts_pretrain_clamp 1.0 \
  --encoder_bundle models/motion_encoder_equiv.pt.best.pt \
  --posttrain_contacts_pretrain_affine_stats debug_output/_tmp_phaseb_affine_20260304/affine_fit_mix08/affine_stats.json

PYTHONPATH=. debug_output/_tmp_phasecd_min_ablation_20260330/cpu_nomps_exec.py -m train.validate.run_freerun_cycles \
  --teacher validate/teacher_batches/Walk_F_teacher.json \
  --model models/__tmp_dsn_aux_leg_matched_chain_20260410/70b_replace/aux_detach/ckpt_last_WalkF_stage7_70b_replace_lowdrift_e3x60_lr5e5_dsn_auxdetach_20260410.pth \
  --rounds 5 --depth 3 --time-index-mode cycle \
  --event_clock auto \
  --phase_reset_source none \
  --contacts_meas_source model \
  --lambda_fusion_apply \
  --log_contacts \
  --export_direct_arm_probe \
  --export_joint_direct_geolocal_series \
  --out debug_output/_tmp_dsn_aux_leg_matched_chain_20260410/70b_replace/aux_detach/eval_model_source \
  --force

PYTHONPATH=. debug_output/_tmp_phasecd_min_ablation_20260330/cpu_nomps_exec.py \
  tools/phasea_group_summary.py \
  debug_output/_tmp_dsn_aux_leg_matched_chain_20260410/70b_replace/aux_detach/eval_model_source/Walk_F_freerun_cycles.json \
  --cycle_gte 1 \
  --drop_wrap \
  --out debug_output/_tmp_dsn_aux_leg_matched_chain_20260410/70b_replace/aux_detach/eval_model_source_group_summary.json
```

No baseline rerun occurred. No sham/aux rerun occurred.

## 4. Result table

### 4.1 Absolute metrics

| stage | arm | `all_ex_root mean` | `all_ex_root p95` | `leg mean` | `leg p95` | `nonleg mean` | `nonleg p95` | `arm mean` | `arm p95` | `else mean` | `else p95` |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `stage6` | `baseline` | `0.250873` | `0.836182` | `0.566078` | `1.324971` | `0.182721` | `0.604531` | `0.201131` | `0.687248` | `0.139205` | `0.351424` |
| `stage6` | `aux_detach` | `0.262125` | `0.937262` | `0.597226` | `1.316565` | `0.189670` | `0.699526` | `0.220076` | `0.801407` | `0.117801` | `0.344549` |
| `70a` | `baseline` | `0.215662` | `0.722130` | `0.534848` | `1.329624` | `0.146649` | `0.511962` | `0.163004` | `0.584925` | `0.107992` | `0.285793` |
| `70a` | `aux_detach` | `0.246634` | `0.830595` | `0.582586` | `1.412086` | `0.173996` | `0.610583` | `0.198705` | `0.685155` | `0.115593` | `0.328243` |
| `70b replace` | `baseline` | `0.186194` | `0.644118` | `0.395349` | `0.911244` | `0.140971` | `0.483077` | `0.161008` | `0.552980` | `0.093612` | `0.278943` |
| `70b replace` | `aux_detach` | `0.182422` | `0.643131` | `0.407572` | `1.006993` | `0.133741` | `0.466280` | `0.152037` | `0.509387` | `0.090495` | `0.281773` |

### 4.2 Delta: `aux_detach - baseline`

Negative is better.

| stage | `all_ex_root mean` | `all_ex_root p95` | `leg mean` | `leg p95` | `nonleg mean` | `nonleg p95` | `arm mean` | `arm p95` | `else mean` | `else p95` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `stage6` | `+0.011251` | `+0.101080` | `+0.031148` | `-0.008405` | `+0.006949` | `+0.094995` | `+0.018945` | `+0.114158` | `-0.021404` | `-0.006874` |
| `70a` | `+0.030972` | `+0.108466` | `+0.047739` | `+0.082461` | `+0.027347` | `+0.098620` | `+0.035701` | `+0.100229` | `+0.007602` | `+0.042450` |
| `70b replace` | `-0.003772` | `-0.000987` | `+0.012223` | `+0.095749` | `-0.007231` | `-0.016797` | `-0.008972` | `-0.043593` | `-0.003117` | `+0.002830` |

## 5. Stage-by-stage readout

### 5.1 `stage6`

Already from reused artifact, `aux_detach` does not open a baseline-positive case:

- `all_ex_root mean/p95` both worsen
- `leg mean` worsens
- only `leg p95` is marginally lower, but this is not a net stage win

So the feature has no early clean gain window at `stage6`.

### 5.2 `70a`

`70a` is clearly negative for `aux_detach` relative to `baseline`:

- `all_ex_root`, `leg`, `nonleg`, `arm`, `else` all worsen on mean
- `all_ex_root p95` and `leg p95` both worsen materially
- there is no interpretation under which this stage is a downstream win

So the hoped-for “preserve contract now, pay off at downstream handoff” story is already not supported at `70a`.

### 5.3 `70b replace`

Final `70b replace` is **mixed**, not cleanly positive:

- slight gains:
  - `all_ex_root mean`: `-0.003772`
  - `all_ex_root p95`: `-0.000987`
  - `nonleg mean/p95`: improved
  - `arm mean/p95`: improved
- clear losses:
  - `leg mean`: `+0.012223`
  - `leg p95`: `+0.095749`
  - `else p95`: slight regression

The decisive point is that the final target is **not** a clean baseline beat. It is at best a mixed trade:

- aggregate headline barely moves
- tail improvement in `all_ex_root p95` is effectively negligible
- the feature’s intended leg-side value proposition is still not realized at the final stage

Under the stated decision rule, this does **not** qualify as “clean downstream positive evidence”.

## 6. Decision

### 6.1 Does `aux_detach` create a clean downstream positive case?

No.

Reason:

1. `stage6` does not beat `baseline`
2. `70a` is clearly worse than `baseline`
3. final `70b replace` is only mixed:
   - some nonleg / arm improvement
   - but `leg` regresses, especially `leg p95`
   - aggregate improvement is too small to count as a clean win

Therefore this round does **not** provide sufficient baseline-positive downstream evidence to justify keeping the aux-leg feature in the mainline pipeline.

### 6.2 Final two-layer conclusion

- `global default = baseline`
- inside the now-retired aux-family records, `aux_detach` remains the least-bad reference variant

In the requested binary framing, this round lands on:

> **feature not justified for mainline**

## 7. Final statement

`aux_detach` remains the lowest-risk reference arm **if** the aux-family is ever reopened for some future targeted need.

But on the actual matched downstream chain used for mainline decision-making, `aux_detach` still fails to produce a clean final beat over `baseline`.

So the correct retention decision for the feature itself is:

- keep mainline on `baseline`
- do **not** retain aux-leg supervision as a justified mainline feature on the current evidence
