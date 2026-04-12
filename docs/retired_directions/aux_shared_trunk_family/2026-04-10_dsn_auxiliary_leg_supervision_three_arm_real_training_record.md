# 2026-04-10 DSN auxiliary leg supervision three-arm real training record

> Status: archived / retired aux-family mechanism record
> Reader note: this aux / shared-trunk family did **not** become current repo mainline; any `recommend`, `default`, `ship`, `mainline`, or `current` wording below is historical family-local language only.
> Current entry points: `docs/posttrain_pipeline.md`, `docs/train_design/2026-04-11_donor_contract_basin_universal_debug_synthesis.md`, `docs/retired_directions/aux_shared_trunk_family/2026-04-11_aux_detach_downstream_reevaluation_record.md`

> Status: real matched three-arm run completed
> Scope: `baseline` / `sham` / `DSN aux-leg` on fixed `stage6 -> 70a -> new70b_replace_lowdrift`
> Goal: 只回答这条 DSN auxiliary leg supervision 线在真实 matched downstream chain 上是否值得继续

## 1. Fixed matched recipe

### 1.1 Fixed donor family

- donor family: `cp015 tailk7 rankmix tw020 corridor_hold tail15 phasea050 fixedsched ep014center control denseckpt seed2024`
- fixed basetrain donor ckpt:
  - `models/cp015_phasecd_tailk_probe_20260331/exp_phase_DirectBranch_v1_d1_cp015_tailk7_rankmix_tw020_corridor_hold_tail15_phasea050_fixedsched_ep014center_control_denseckpt_seed2024_20260401/ckpt_epoch_014.pth`

### 1.2 Fixed stage configs / semantics

| stage | reused config | epochs | steps_per_epoch | lr | encoder_bundle | direct_pose_use_phase_z | direct_pose_phase_z_mode |
| --- | --- | ---: | ---: | ---: | --- | --- | --- |
| `stage6` | `config/posttrain_WalkF_stage6_direct_cond_anchor_splitfirst_3way_armchain_pe32h512_tailfix_lr3e4_e8x60_wd1e4_reinit1_20260401.json` | `8` | `60` | `3e-4` | `models/motion_encoder_equiv.pt.best.pt` | `false` | `concat` |
| `70a` | `debug_output/_tmp_ep014center_70a_lowlr_sweep_20260328/configs/posttrain_70a_lr3e4_from_ep014center_20260328.json` | `5` | `60` | `3e-4` | `models/motion_encoder_equiv.pt.best.pt` | `false` | `concat` |
| `new70b_replace_lowdrift` | `debug_output/_tmp_cp015_tailk7_replace_schedule_ablation_20260402/configs/posttrain_70b_replace_lowdrift_e3x60_lr5e5_from_cp015_tailk7_70a_20260402.json` | `3` | `60` | `5e-5` | `models/motion_encoder_equiv.pt.best.pt` | `true` | `concat` |

Shared fixed contact args:

- `posttrain_contacts_source=pretrain_contact`
- `posttrain_contacts_pretrain_clamp=1.0`
- `posttrain_contacts_pretrain_affine_stats=debug_output/_tmp_phaseb_affine_20260304/affine_fit_mix08/affine_stats.json`

### 1.3 Fixed arm definitions

- `baseline`
  - `direct_pose_aux_leg_enable=false`
- `sham`
  - `direct_pose_aux_leg_enable=true`
  - `direct_pose_aux_leg_weight=0.0`
- `DSN aux-leg`
  - `direct_pose_aux_leg_enable=true`
  - `direct_pose_aux_leg_weight=0.2`

Aux-only fixed knobs for `sham` / `DSN aux-leg`:

- `direct_pose_aux_leg_variant=linear`
- `direct_pose_aux_leg_hidden=0`
- `direct_pose_aux_leg_detach_feat=false`
- `direct_pose_aux_leg_loss_mode=geo`
- `direct_pose_aux_leg_warmup_steps=0`
- `direct_pose_aux_leg_hold_steps=0`
- `direct_pose_aux_leg_decay_steps=0`
- `direct_pose_aux_leg_min_weight=0.0`
- `direct_pose_aux_leg_log_enable=true`

Output roots:

- models: `models/__tmp_dsn_aux_leg_matched_chain_20260410`
- eval/debug: `debug_output/_tmp_dsn_aux_leg_matched_chain_20260410`

## 2. Actual commands run

### 2.1 `stage6`

#### baseline

```bash
PYTHONPATH=. debug_output/_tmp_phasecd_min_ablation_20260330/cpu_nomps_exec.py -m train.posttrain --config config/posttrain_WalkF_stage6_direct_cond_anchor_splitfirst_3way_armchain_pe32h512_tailfix_lr3e4_e8x60_wd1e4_reinit1_20260401.json --ckpt_in models/cp015_phasecd_tailk_probe_20260331/exp_phase_DirectBranch_v1_d1_cp015_tailk7_rankmix_tw020_corridor_hold_tail15_phasea050_fixedsched_ep014center_control_denseckpt_seed2024_20260401/ckpt_epoch_014.pth --out_dir models/__tmp_dsn_aux_leg_matched_chain_20260410/stage6/baseline --run_name lr3e4_e8x60_wd1e4_reinit1_stage6_tailfix_dsn_base_20260410 --posttrain_contacts_source pretrain_contact --posttrain_contacts_pretrain_clamp 1.0 --encoder_bundle models/motion_encoder_equiv.pt.best.pt --posttrain_contacts_pretrain_affine_stats debug_output/_tmp_phaseb_affine_20260304/affine_fit_mix08/affine_stats.json --direct_pose_aux_leg_enable false --direct_pose_aux_leg_variant linear --direct_pose_aux_leg_hidden 0 --direct_pose_aux_leg_detach_feat false --direct_pose_aux_leg_weight 0.0 --direct_pose_aux_leg_loss_mode geo --direct_pose_aux_leg_warmup_steps 0 --direct_pose_aux_leg_hold_steps 0 --direct_pose_aux_leg_decay_steps 0 --direct_pose_aux_leg_min_weight 0.0 --direct_pose_aux_leg_log_enable false
PYTHONPATH=. debug_output/_tmp_phasecd_min_ablation_20260330/cpu_nomps_exec.py -m train.validate.run_freerun_cycles --teacher validate/teacher_batches/Walk_F_teacher.json --model models/__tmp_dsn_aux_leg_matched_chain_20260410/stage6/baseline/ckpt_last_lr3e4_e8x60_wd1e4_reinit1_stage6_tailfix_dsn_base_20260410.pth --rounds 5 --depth 3 --time-index-mode cycle --phase_reset_source none --contacts_meas_source pretrain_contact --contacts_meas_pretrain_clamp 1.0 --contacts_meas_pretrain_affine_stats debug_output/_tmp_phaseb_affine_20260304/affine_fit_mix08/affine_stats.json --encoder-bundle models/motion_encoder_equiv.pt.best.pt --export_joint_direct_geolocal_series --out debug_output/_tmp_dsn_aux_leg_matched_chain_20260410/stage6/baseline/stage6_freerun --force
PYTHONPATH=. debug_output/_tmp_phasecd_min_ablation_20260330/cpu_nomps_exec.py tools/phasea_group_summary.py debug_output/_tmp_dsn_aux_leg_matched_chain_20260410/stage6/baseline/stage6_freerun/Walk_F_freerun_cycles.json --cycle_gte 1 --drop_wrap --out debug_output/_tmp_dsn_aux_leg_matched_chain_20260410/stage6/baseline/stage6_group_summary.json
```

#### sham

```bash
PYTHONPATH=. debug_output/_tmp_phasecd_min_ablation_20260330/cpu_nomps_exec.py -m train.posttrain --config config/posttrain_WalkF_stage6_direct_cond_anchor_splitfirst_3way_armchain_pe32h512_tailfix_lr3e4_e8x60_wd1e4_reinit1_20260401.json --ckpt_in models/cp015_phasecd_tailk_probe_20260331/exp_phase_DirectBranch_v1_d1_cp015_tailk7_rankmix_tw020_corridor_hold_tail15_phasea050_fixedsched_ep014center_control_denseckpt_seed2024_20260401/ckpt_epoch_014.pth --out_dir models/__tmp_dsn_aux_leg_matched_chain_20260410/stage6/sham --run_name lr3e4_e8x60_wd1e4_reinit1_stage6_tailfix_dsn_sham_20260410 --posttrain_contacts_source pretrain_contact --posttrain_contacts_pretrain_clamp 1.0 --encoder_bundle models/motion_encoder_equiv.pt.best.pt --posttrain_contacts_pretrain_affine_stats debug_output/_tmp_phaseb_affine_20260304/affine_fit_mix08/affine_stats.json --direct_pose_aux_leg_enable true --direct_pose_aux_leg_variant linear --direct_pose_aux_leg_hidden 0 --direct_pose_aux_leg_detach_feat false --direct_pose_aux_leg_weight 0.0 --direct_pose_aux_leg_loss_mode geo --direct_pose_aux_leg_warmup_steps 0 --direct_pose_aux_leg_hold_steps 0 --direct_pose_aux_leg_decay_steps 0 --direct_pose_aux_leg_min_weight 0.0 --direct_pose_aux_leg_log_enable true
PYTHONPATH=. debug_output/_tmp_phasecd_min_ablation_20260330/cpu_nomps_exec.py -m train.validate.run_freerun_cycles --teacher validate/teacher_batches/Walk_F_teacher.json --model models/__tmp_dsn_aux_leg_matched_chain_20260410/stage6/sham/ckpt_last_lr3e4_e8x60_wd1e4_reinit1_stage6_tailfix_dsn_sham_20260410.pth --rounds 5 --depth 3 --time-index-mode cycle --phase_reset_source none --contacts_meas_source pretrain_contact --contacts_meas_pretrain_clamp 1.0 --contacts_meas_pretrain_affine_stats debug_output/_tmp_phaseb_affine_20260304/affine_fit_mix08/affine_stats.json --encoder-bundle models/motion_encoder_equiv.pt.best.pt --export_joint_direct_geolocal_series --out debug_output/_tmp_dsn_aux_leg_matched_chain_20260410/stage6/sham/stage6_freerun --force
PYTHONPATH=. debug_output/_tmp_phasecd_min_ablation_20260330/cpu_nomps_exec.py tools/phasea_group_summary.py debug_output/_tmp_dsn_aux_leg_matched_chain_20260410/stage6/sham/stage6_freerun/Walk_F_freerun_cycles.json --cycle_gte 1 --drop_wrap --out debug_output/_tmp_dsn_aux_leg_matched_chain_20260410/stage6/sham/stage6_group_summary.json
```

#### DSN aux-leg

```bash
PYTHONPATH=. debug_output/_tmp_phasecd_min_ablation_20260330/cpu_nomps_exec.py -m train.posttrain --config config/posttrain_WalkF_stage6_direct_cond_anchor_splitfirst_3way_armchain_pe32h512_tailfix_lr3e4_e8x60_wd1e4_reinit1_20260401.json --ckpt_in models/cp015_phasecd_tailk_probe_20260331/exp_phase_DirectBranch_v1_d1_cp015_tailk7_rankmix_tw020_corridor_hold_tail15_phasea050_fixedsched_ep014center_control_denseckpt_seed2024_20260401/ckpt_epoch_014.pth --out_dir models/__tmp_dsn_aux_leg_matched_chain_20260410/stage6/aux --run_name lr3e4_e8x60_wd1e4_reinit1_stage6_tailfix_dsn_auxw02_20260410 --posttrain_contacts_source pretrain_contact --posttrain_contacts_pretrain_clamp 1.0 --encoder_bundle models/motion_encoder_equiv.pt.best.pt --posttrain_contacts_pretrain_affine_stats debug_output/_tmp_phaseb_affine_20260304/affine_fit_mix08/affine_stats.json --direct_pose_aux_leg_enable true --direct_pose_aux_leg_variant linear --direct_pose_aux_leg_hidden 0 --direct_pose_aux_leg_detach_feat false --direct_pose_aux_leg_weight 0.2 --direct_pose_aux_leg_loss_mode geo --direct_pose_aux_leg_warmup_steps 0 --direct_pose_aux_leg_hold_steps 0 --direct_pose_aux_leg_decay_steps 0 --direct_pose_aux_leg_min_weight 0.0 --direct_pose_aux_leg_log_enable true
PYTHONPATH=. debug_output/_tmp_phasecd_min_ablation_20260330/cpu_nomps_exec.py -m train.validate.run_freerun_cycles --teacher validate/teacher_batches/Walk_F_teacher.json --model models/__tmp_dsn_aux_leg_matched_chain_20260410/stage6/aux/ckpt_last_lr3e4_e8x60_wd1e4_reinit1_stage6_tailfix_dsn_auxw02_20260410.pth --rounds 5 --depth 3 --time-index-mode cycle --phase_reset_source none --contacts_meas_source pretrain_contact --contacts_meas_pretrain_clamp 1.0 --contacts_meas_pretrain_affine_stats debug_output/_tmp_phaseb_affine_20260304/affine_fit_mix08/affine_stats.json --encoder-bundle models/motion_encoder_equiv.pt.best.pt --export_joint_direct_geolocal_series --out debug_output/_tmp_dsn_aux_leg_matched_chain_20260410/stage6/aux/stage6_freerun --force
PYTHONPATH=. debug_output/_tmp_phasecd_min_ablation_20260330/cpu_nomps_exec.py tools/phasea_group_summary.py debug_output/_tmp_dsn_aux_leg_matched_chain_20260410/stage6/aux/stage6_freerun/Walk_F_freerun_cycles.json --cycle_gte 1 --drop_wrap --out debug_output/_tmp_dsn_aux_leg_matched_chain_20260410/stage6/aux/stage6_group_summary.json
```

### 2.2 `70a`

#### baseline

```bash
PYTHONPATH=. debug_output/_tmp_phasecd_min_ablation_20260330/cpu_nomps_exec.py -m train.posttrain --config debug_output/_tmp_ep014center_70a_lowlr_sweep_20260328/configs/posttrain_70a_lr3e4_from_ep014center_20260328.json --ckpt_in models/__tmp_dsn_aux_leg_matched_chain_20260410/stage6/baseline/ckpt_last_lr3e4_e8x60_wd1e4_reinit1_stage6_tailfix_dsn_base_20260410.pth --out_dir models/__tmp_dsn_aux_leg_matched_chain_20260410/70a/baseline --run_name WalkF_stage7_70a_lr3e4_dsn_base_20260410 --posttrain_contacts_source pretrain_contact --posttrain_contacts_pretrain_clamp 1.0 --encoder_bundle models/motion_encoder_equiv.pt.best.pt --posttrain_contacts_pretrain_affine_stats debug_output/_tmp_phaseb_affine_20260304/affine_fit_mix08/affine_stats.json
PYTHONPATH=. debug_output/_tmp_phasecd_min_ablation_20260330/cpu_nomps_exec.py -m train.validate.run_freerun_cycles --teacher validate/teacher_batches/Walk_F_teacher.json --model models/__tmp_dsn_aux_leg_matched_chain_20260410/70a/baseline/ckpt_last_WalkF_stage7_70a_lr3e4_dsn_base_20260410.pth --rounds 5 --depth 3 --time-index-mode cycle --event_clock auto --phase_reset_source none --contacts_meas_source model --lambda_fusion_apply --log_contacts --export_direct_arm_probe --export_joint_direct_geolocal_series --out debug_output/_tmp_dsn_aux_leg_matched_chain_20260410/70a/baseline/eval_model_source --force
PYTHONPATH=. debug_output/_tmp_phasecd_min_ablation_20260330/cpu_nomps_exec.py tools/phasea_group_summary.py debug_output/_tmp_dsn_aux_leg_matched_chain_20260410/70a/baseline/eval_model_source/Walk_F_freerun_cycles.json --cycle_gte 1 --drop_wrap --out debug_output/_tmp_dsn_aux_leg_matched_chain_20260410/70a/baseline/eval_model_source_group_summary.json
```

#### sham

```bash
PYTHONPATH=. debug_output/_tmp_phasecd_min_ablation_20260330/cpu_nomps_exec.py -m train.posttrain --config debug_output/_tmp_ep014center_70a_lowlr_sweep_20260328/configs/posttrain_70a_lr3e4_from_ep014center_20260328.json --ckpt_in models/__tmp_dsn_aux_leg_matched_chain_20260410/stage6/sham/ckpt_last_lr3e4_e8x60_wd1e4_reinit1_stage6_tailfix_dsn_sham_20260410.pth --out_dir models/__tmp_dsn_aux_leg_matched_chain_20260410/70a/sham --run_name WalkF_stage7_70a_lr3e4_dsn_sham_20260410 --posttrain_contacts_source pretrain_contact --posttrain_contacts_pretrain_clamp 1.0 --encoder_bundle models/motion_encoder_equiv.pt.best.pt --posttrain_contacts_pretrain_affine_stats debug_output/_tmp_phaseb_affine_20260304/affine_fit_mix08/affine_stats.json
PYTHONPATH=. debug_output/_tmp_phasecd_min_ablation_20260330/cpu_nomps_exec.py -m train.validate.run_freerun_cycles --teacher validate/teacher_batches/Walk_F_teacher.json --model models/__tmp_dsn_aux_leg_matched_chain_20260410/70a/sham/ckpt_last_WalkF_stage7_70a_lr3e4_dsn_sham_20260410.pth --rounds 5 --depth 3 --time-index-mode cycle --event_clock auto --phase_reset_source none --contacts_meas_source model --lambda_fusion_apply --log_contacts --export_direct_arm_probe --export_joint_direct_geolocal_series --out debug_output/_tmp_dsn_aux_leg_matched_chain_20260410/70a/sham/eval_model_source --force
PYTHONPATH=. debug_output/_tmp_phasecd_min_ablation_20260330/cpu_nomps_exec.py tools/phasea_group_summary.py debug_output/_tmp_dsn_aux_leg_matched_chain_20260410/70a/sham/eval_model_source/Walk_F_freerun_cycles.json --cycle_gte 1 --drop_wrap --out debug_output/_tmp_dsn_aux_leg_matched_chain_20260410/70a/sham/eval_model_source_group_summary.json
```

#### DSN aux-leg

```bash
PYTHONPATH=. debug_output/_tmp_phasecd_min_ablation_20260330/cpu_nomps_exec.py -m train.posttrain --config debug_output/_tmp_ep014center_70a_lowlr_sweep_20260328/configs/posttrain_70a_lr3e4_from_ep014center_20260328.json --ckpt_in models/__tmp_dsn_aux_leg_matched_chain_20260410/stage6/aux/ckpt_last_lr3e4_e8x60_wd1e4_reinit1_stage6_tailfix_dsn_auxw02_20260410.pth --out_dir models/__tmp_dsn_aux_leg_matched_chain_20260410/70a/aux --run_name WalkF_stage7_70a_lr3e4_dsn_auxw02_20260410 --posttrain_contacts_source pretrain_contact --posttrain_contacts_pretrain_clamp 1.0 --encoder_bundle models/motion_encoder_equiv.pt.best.pt --posttrain_contacts_pretrain_affine_stats debug_output/_tmp_phaseb_affine_20260304/affine_fit_mix08/affine_stats.json
PYTHONPATH=. debug_output/_tmp_phasecd_min_ablation_20260330/cpu_nomps_exec.py -m train.validate.run_freerun_cycles --teacher validate/teacher_batches/Walk_F_teacher.json --model models/__tmp_dsn_aux_leg_matched_chain_20260410/70a/aux/ckpt_last_WalkF_stage7_70a_lr3e4_dsn_auxw02_20260410.pth --rounds 5 --depth 3 --time-index-mode cycle --event_clock auto --phase_reset_source none --contacts_meas_source model --lambda_fusion_apply --log_contacts --export_direct_arm_probe --export_joint_direct_geolocal_series --out debug_output/_tmp_dsn_aux_leg_matched_chain_20260410/70a/aux/eval_model_source --force
PYTHONPATH=. debug_output/_tmp_phasecd_min_ablation_20260330/cpu_nomps_exec.py tools/phasea_group_summary.py debug_output/_tmp_dsn_aux_leg_matched_chain_20260410/70a/aux/eval_model_source/Walk_F_freerun_cycles.json --cycle_gte 1 --drop_wrap --out debug_output/_tmp_dsn_aux_leg_matched_chain_20260410/70a/aux/eval_model_source_group_summary.json
```

### 2.3 `new70b_replace_lowdrift`

#### baseline

```bash
python3 - <<'PY'
from pathlib import Path
from tools.run_cp015_oldplan_downstream_chain import create_replace_zerophase_warmstart
create_replace_zerophase_warmstart(
    Path('models/__tmp_dsn_aux_leg_matched_chain_20260410/70a/baseline/ckpt_last_WalkF_stage7_70a_lr3e4_dsn_base_20260410.pth'),
    Path('models/__tmp_dsn_aux_leg_matched_chain_20260410/70b_replace/baseline/warmstart/ckpt_last_70a_replace_zerophase_baseline_20260410.pth'),
    Path('debug_output/_tmp_dsn_aux_leg_matched_chain_20260410/70b_replace/baseline/warmstart/replace_zerophase_report.json'),
)
PY
PYTHONPATH=. debug_output/_tmp_phasecd_min_ablation_20260330/cpu_nomps_exec.py -m train.posttrain --config debug_output/_tmp_cp015_tailk7_replace_schedule_ablation_20260402/configs/posttrain_70b_replace_lowdrift_e3x60_lr5e5_from_cp015_tailk7_70a_20260402.json --ckpt_in models/__tmp_dsn_aux_leg_matched_chain_20260410/70b_replace/baseline/warmstart/ckpt_last_70a_replace_zerophase_baseline_20260410.pth --out_dir models/__tmp_dsn_aux_leg_matched_chain_20260410/70b_replace/baseline --run_name WalkF_stage7_70b_replace_lowdrift_e3x60_lr5e5_dsn_base_20260410 --posttrain_contacts_source pretrain_contact --posttrain_contacts_pretrain_clamp 1.0 --encoder_bundle models/motion_encoder_equiv.pt.best.pt --posttrain_contacts_pretrain_affine_stats debug_output/_tmp_phaseb_affine_20260304/affine_fit_mix08/affine_stats.json
PYTHONPATH=. debug_output/_tmp_phasecd_min_ablation_20260330/cpu_nomps_exec.py -m train.validate.run_freerun_cycles --teacher validate/teacher_batches/Walk_F_teacher.json --model models/__tmp_dsn_aux_leg_matched_chain_20260410/70b_replace/baseline/ckpt_last_WalkF_stage7_70b_replace_lowdrift_e3x60_lr5e5_dsn_base_20260410.pth --rounds 5 --depth 3 --time-index-mode cycle --event_clock auto --phase_reset_source none --contacts_meas_source model --lambda_fusion_apply --log_contacts --export_direct_arm_probe --export_joint_direct_geolocal_series --out debug_output/_tmp_dsn_aux_leg_matched_chain_20260410/70b_replace/baseline/eval_model_source --force
PYTHONPATH=. debug_output/_tmp_phasecd_min_ablation_20260330/cpu_nomps_exec.py tools/phasea_group_summary.py debug_output/_tmp_dsn_aux_leg_matched_chain_20260410/70b_replace/baseline/eval_model_source/Walk_F_freerun_cycles.json --cycle_gte 1 --drop_wrap --out debug_output/_tmp_dsn_aux_leg_matched_chain_20260410/70b_replace/baseline/eval_model_source_group_summary.json
```

#### sham

```bash
python3 - <<'PY'
from pathlib import Path
from tools.run_cp015_oldplan_downstream_chain import create_replace_zerophase_warmstart
create_replace_zerophase_warmstart(
    Path('models/__tmp_dsn_aux_leg_matched_chain_20260410/70a/sham/ckpt_last_WalkF_stage7_70a_lr3e4_dsn_sham_20260410.pth'),
    Path('models/__tmp_dsn_aux_leg_matched_chain_20260410/70b_replace/sham/warmstart/ckpt_last_70a_replace_zerophase_sham_20260410.pth'),
    Path('debug_output/_tmp_dsn_aux_leg_matched_chain_20260410/70b_replace/sham/warmstart/replace_zerophase_report.json'),
)
PY
PYTHONPATH=. debug_output/_tmp_phasecd_min_ablation_20260330/cpu_nomps_exec.py -m train.posttrain --config debug_output/_tmp_cp015_tailk7_replace_schedule_ablation_20260402/configs/posttrain_70b_replace_lowdrift_e3x60_lr5e5_from_cp015_tailk7_70a_20260402.json --ckpt_in models/__tmp_dsn_aux_leg_matched_chain_20260410/70b_replace/sham/warmstart/ckpt_last_70a_replace_zerophase_sham_20260410.pth --out_dir models/__tmp_dsn_aux_leg_matched_chain_20260410/70b_replace/sham --run_name WalkF_stage7_70b_replace_lowdrift_e3x60_lr5e5_dsn_sham_20260410 --posttrain_contacts_source pretrain_contact --posttrain_contacts_pretrain_clamp 1.0 --encoder_bundle models/motion_encoder_equiv.pt.best.pt --posttrain_contacts_pretrain_affine_stats debug_output/_tmp_phaseb_affine_20260304/affine_fit_mix08/affine_stats.json
PYTHONPATH=. debug_output/_tmp_phasecd_min_ablation_20260330/cpu_nomps_exec.py -m train.validate.run_freerun_cycles --teacher validate/teacher_batches/Walk_F_teacher.json --model models/__tmp_dsn_aux_leg_matched_chain_20260410/70b_replace/sham/ckpt_last_WalkF_stage7_70b_replace_lowdrift_e3x60_lr5e5_dsn_sham_20260410.pth --rounds 5 --depth 3 --time-index-mode cycle --event_clock auto --phase_reset_source none --contacts_meas_source model --lambda_fusion_apply --log_contacts --export_direct_arm_probe --export_joint_direct_geolocal_series --out debug_output/_tmp_dsn_aux_leg_matched_chain_20260410/70b_replace/sham/eval_model_source --force
PYTHONPATH=. debug_output/_tmp_phasecd_min_ablation_20260330/cpu_nomps_exec.py tools/phasea_group_summary.py debug_output/_tmp_dsn_aux_leg_matched_chain_20260410/70b_replace/sham/eval_model_source/Walk_F_freerun_cycles.json --cycle_gte 1 --drop_wrap --out debug_output/_tmp_dsn_aux_leg_matched_chain_20260410/70b_replace/sham/eval_model_source_group_summary.json
```

#### DSN aux-leg

```bash
python3 - <<'PY'
from pathlib import Path
from tools.run_cp015_oldplan_downstream_chain import create_replace_zerophase_warmstart
create_replace_zerophase_warmstart(
    Path('models/__tmp_dsn_aux_leg_matched_chain_20260410/70a/aux/ckpt_last_WalkF_stage7_70a_lr3e4_dsn_auxw02_20260410.pth'),
    Path('models/__tmp_dsn_aux_leg_matched_chain_20260410/70b_replace/aux/warmstart/ckpt_last_70a_replace_zerophase_aux_20260410.pth'),
    Path('debug_output/_tmp_dsn_aux_leg_matched_chain_20260410/70b_replace/aux/warmstart/replace_zerophase_report.json'),
)
PY
PYTHONPATH=. debug_output/_tmp_phasecd_min_ablation_20260330/cpu_nomps_exec.py -m train.posttrain --config debug_output/_tmp_cp015_tailk7_replace_schedule_ablation_20260402/configs/posttrain_70b_replace_lowdrift_e3x60_lr5e5_from_cp015_tailk7_70a_20260402.json --ckpt_in models/__tmp_dsn_aux_leg_matched_chain_20260410/70b_replace/aux/warmstart/ckpt_last_70a_replace_zerophase_aux_20260410.pth --out_dir models/__tmp_dsn_aux_leg_matched_chain_20260410/70b_replace/aux --run_name WalkF_stage7_70b_replace_lowdrift_e3x60_lr5e5_dsn_auxw02_20260410 --posttrain_contacts_source pretrain_contact --posttrain_contacts_pretrain_clamp 1.0 --encoder_bundle models/motion_encoder_equiv.pt.best.pt --posttrain_contacts_pretrain_affine_stats debug_output/_tmp_phaseb_affine_20260304/affine_fit_mix08/affine_stats.json
PYTHONPATH=. debug_output/_tmp_phasecd_min_ablation_20260330/cpu_nomps_exec.py -m train.validate.run_freerun_cycles --teacher validate/teacher_batches/Walk_F_teacher.json --model models/__tmp_dsn_aux_leg_matched_chain_20260410/70b_replace/aux/ckpt_last_WalkF_stage7_70b_replace_lowdrift_e3x60_lr5e5_dsn_auxw02_20260410.pth --rounds 5 --depth 3 --time-index-mode cycle --event_clock auto --phase_reset_source none --contacts_meas_source model --lambda_fusion_apply --log_contacts --export_direct_arm_probe --export_joint_direct_geolocal_series --out debug_output/_tmp_dsn_aux_leg_matched_chain_20260410/70b_replace/aux/eval_model_source --force
PYTHONPATH=. debug_output/_tmp_phasecd_min_ablation_20260330/cpu_nomps_exec.py tools/phasea_group_summary.py debug_output/_tmp_dsn_aux_leg_matched_chain_20260410/70b_replace/aux/eval_model_source/Walk_F_freerun_cycles.json --cycle_gte 1 --drop_wrap --out debug_output/_tmp_dsn_aux_leg_matched_chain_20260410/70b_replace/aux/eval_model_source_group_summary.json
```

## 3. Run success

| item | baseline | sham | aux |
| --- | --- | --- | --- |
| `stage6 train` | `ok` | `ok` | `ok` |
| `stage6 eval` | `ok` | `ok` | `ok` |
| `stage6 group summary` | `ok` | `ok` | `ok` |
| `70a train` | `ok` | `ok` | `ok` |
| `70a eval` | `ok` | `ok` | `ok` |
| `70a group summary` | `ok` | `ok` | `ok` |
| `70b warmstart` | `ok` | `ok` | `ok` |
| `70b train` | `ok` | `ok` | `ok` |
| `70b eval` | `ok` | `ok` | `ok` |
| `70b group summary` | `ok` | `ok` | `ok` |

Observed notes:

- no real blocker exposed
- no strip / loader / `posttrain_cfg` residual issue surfaced in this round
- `sham` / `aux` `stage6` exports both wrote stripped handoff artifacts and successfully continued into `70a -> 70b`

## 4. Metric tables

### 4.1 `stage6 native`

| arm | `DirectGeoLocalDeg` | `all_ex_root` | `leg` | `nonleg` | `arm` | `else` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `baseline` | 0.250873 | 0.250873 | 0.566078 | 0.182721 | 0.201131 | 0.139205 |
| `sham` | 0.237614 | 0.237614 | 0.598018 | 0.159688 | 0.175657 | 0.121944 |
| `DSN aux-leg` | 0.268797 | 0.268797 | 0.708246 | 0.173780 | 0.194554 | 0.124680 |

Delta vs `baseline`:

- `sham`: `DirectGeoLocalDeg=-0.013260`, `all_ex_root=-0.013260`, `leg=+0.031940`, `nonleg=-0.023033`, `arm=-0.025474`, `else=-0.017261`
- `DSN aux-leg`: `DirectGeoLocalDeg=+0.017923`, `all_ex_root=+0.017923`, `leg=+0.142168`, `nonleg=-0.008940`, `arm=-0.006577`, `else=-0.014526`

### 4.2 `70a native`

| arm | `DirectGeoLocalDeg` | `all_ex_root` | `leg` | `nonleg` | `arm` | `else` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `baseline` | 0.215662 | 0.215662 | 0.534848 | 0.146649 | 0.163004 | 0.107992 |
| `sham` | 0.237464 | 0.237464 | 0.621698 | 0.154386 | 0.177818 | 0.099003 |
| `DSN aux-leg` | 0.234599 | 0.234599 | 0.570475 | 0.161977 | 0.186807 | 0.103290 |

Delta vs `baseline`:

- `sham`: `DirectGeoLocalDeg=+0.021802`, `all_ex_root=+0.021802`, `leg=+0.086850`, `nonleg=+0.007738`, `arm=+0.014814`, `else=-0.008989`
- `DSN aux-leg`: `DirectGeoLocalDeg=+0.018937`, `all_ex_root=+0.018937`, `leg=+0.035627`, `nonleg=+0.015329`, `arm=+0.023803`, `else=-0.004702`

### 4.3 `new70b_replace_lowdrift`

| arm | `DirectGeoLocalDeg` | `all_ex_root` | `leg` | `nonleg` | `arm` | `else` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `baseline` | 0.186194 | 0.186194 | 0.395349 | 0.140971 | 0.161008 | 0.093612 |
| `sham` | 0.183281 | 0.183281 | 0.429534 | 0.130038 | 0.147208 | 0.089452 |
| `DSN aux-leg` | 0.190083 | 0.190083 | 0.420563 | 0.140249 | 0.161796 | 0.089321 |

Delta vs `baseline`:

- `sham`: `DirectGeoLocalDeg=-0.002913`, `all_ex_root=-0.002913`, `leg=+0.034185`, `nonleg=-0.010934`, `arm=-0.013800`, `else=-0.004159`
- `DSN aux-leg`: `DirectGeoLocalDeg=+0.003889`, `all_ex_root=+0.003889`, `leg=+0.025213`, `nonleg=-0.000722`, `arm=+0.000788`, `else=-0.004291`

## 5. Interpretation

### 5.1 Sham relative to baseline: control or not?

结论：`sham` **不是 clean no-op control**，只能算“attachment control”，不能算“完全中性 control”。

原因很直接：

- `stage6` 上它已经改了真实轨迹：
  - `all_ex_root` 变好 `-0.013260`
  - 但核心目标 `leg` 变差 `+0.031940`
- `70a` 上偏离更明显：
  - `all_ex_root=+0.021802`
  - `leg=+0.086850`
- `70b replace` 上虽然 `all_ex_root` / `nonleg` / `arm` 比 baseline 低一点，
  - 但 `leg` 仍然更差 `+0.034185`

所以：

- `sham` 依然有价值，因为它说明“单纯挂 head / 多两个参数张量”本身就会改变训练轨迹；
- 但它不满足“完全只是 control、几乎不动结果”的预期。

### 5.2 DSN aux-leg relative to baseline: gain or no gain?

结论：**没有得到可接受的 baseline 增益。**

按目标优先级看：

- `stage6 native`
  - `DirectGeoLocalDeg=+0.017923`
  - `leg=+0.142168`
- `70a native`
  - `DirectGeoLocalDeg=+0.018937`
  - `leg=+0.035627`
- `new70b_replace_lowdrift`
  - `DirectGeoLocalDeg=+0.003889`
  - `leg=+0.025213`

也就是说：

- 没有任何一个 stage 出现 `DSN aux-leg` 对 `baseline` 的 clean win；
- 到最终 `70b replace` 仍然没赢 baseline。

### 5.3 Gain earliest stage?

结论：**没有。**

如果把目标锁定为本轮最关键的 `DirectGeoLocalDeg / all_ex_root / leg`，则：

- `DSN aux-leg` 没有在 `stage6`
- 没有在 `70a`
- 也没有在 `70b replace`

出现对 baseline 的首次正增益。

### 5.4 Is gain preserved after `70b replace`?

结论：**无可保留增益。**

最终 `70b replace` 上：

- `DSN aux-leg` 相对 baseline
  - `all_ex_root=+0.003889`
  - `leg=+0.025213`
- 因此不能说“upstream auxiliary gain survives handoff”

最多只能说：

- `DSN aux-leg` 比 `sham` 的 `leg` 回退稍小一些（`0.420563` vs `0.429534`）
- 但它仍然没有 beat baseline，也没有形成值得保留的 downstream advantage

## 6. Final verdict

- `sham`：**not just control**；它本身就改变了训练轨迹，尤其在 `leg` 上持续偏差
- `DSN aux-leg`：**not better than baseline**
- 推荐结论：**终止这条线，不扩 scope**

最关键的失败证据是：

1. `DSN aux-leg` 在 `stage6` 直接把 `leg` 拉差到 `+0.142168`
2. 即使经过 `70a -> 70b replace`，最终也仍然没有赢 baseline：
   - `all_ex_root=+0.003889`
   - `leg=+0.025213`
3. `sham` 也不是中性 control，说明这个 training-only side-head attach 本身就在显著扰动轨迹

一句话结论：

> 在当前 fixed matched recipe 下，DSN auxiliary leg supervision 没有给出 baseline-positive、downstream-preserved 的证据；不值得继续沿这条线扩展。

## 7. Higher-order takeaways

### 7.1 This is more than a null result

这轮结果最有信息量的地方，不是单纯“没增益”，而是它把问题更明确地定位到了 shared trunk。

- 当前证据更支持：瓶颈在 **shared trunk 的可用表示 / 优化容量**
- 不支持的方向是：把问题主要归因于 split 后 **branch-side 容量不足**
- 因而，历史上把精力放在 branch expansion / parallel branch transplant / replace 的解释框架，需要用这轮 matched sham + aux 结果重新审视

### 7.2 `stage6 leg +0.142168` falsifies the "add leg supervision on current trunk" route

最关键的观测仍然是：

- `stage6` 上 `DSN aux-leg` 相对 baseline
  - `DirectGeoLocalDeg=+0.017923`
  - `leg=+0.142168`

也就是说，最应该受益的 `leg` 反而受伤最重。

这并不能唯一证明根因一定是 capacity-bound。当前数据同样兼容若干不同机制，例如：

- **capacity saturation**：leg-relevant trunk capacity 已经被 main objective 占满
- **gradient conflict**：aux head 与 main head 在 trunk 上推动了互相覆盖的 leg representation
- **attach-point mismatch**：aux head 的梯度没有有效作用到真正决定 leg 的更深层表示
- **redundant supervision**：aux geo loss 与 main target 高度重合，没有提供新信息，只增加了竞争

这些机制目前在本轮结果里无法区分；但它们共享同一个实操结论：

> `leg` 在当前 trunk 下不是 supervision-bound 的。无论根本机制是 capacity 饱和、梯度冲突、还是 supervision 冗余，"在 trunk 上加 leg 辅助监督" 这条路已经被证伪。

因此，在**不改 trunk 结构**的前提下，继续沿“再换 attach point / aux weight / warmup / decay”去加 leg auxiliary head，预期都很差；大概率只会重复同一类失败。

### 7.3 `sham` is retrospectively damaging to earlier branch interpretations

`sham` 的价值不只是当 control，而是它揭示了：

- 只要在这条链上挂一个额外 head，优化轨迹本身就会分叉
- 这种分叉不是单调的，也不能简单解释成“参数更多带来 regularization”

本轮看到的就是：

- `stage6`：`sham` 的 `all_ex_root` 优于 baseline，但 `leg` 更差
- `70a`：`sham` 明显输 baseline
- `70b replace`：`sham` 的 `all_ex_root / nonleg / arm` 略优于 baseline，但 `leg` 仍更差

所以对历史实验更严格的说法应该是：

- 凡是没有 matched sham 的 branch / auxiliary 实验
- 其 observed gap 都不能再直接解释为纯 loss effect
- 至少有一部分差异，可能只是 **structural perturbation / optimizer trajectory fork**

这并不自动推翻所有历史结论；但它要求后续若要复盘 A1 expanded branch / transplant / replace 一类结果，必须把 sham-level perturbation 单独审计出来。

### 7.4 No early aux gain means schedule design had no window to save

这轮三 stage 的结果还说明另一件事：

- 不存在“aux 先帮 trunk 学到更好特征”
- 然后收益在 downstream handoff 后仍被保留

因为如果这种机制存在，至少应该在 `stage6` 或 `70a` 先观察到一个早期正增益窗口，再讨论 warmup / decay 是否能把它保住。

但当前观测是：

- `stage6` 没有
- `70a` 没有
- `70b replace` 也没有

所以结论不是“schedule 没调好”，而是：

- 根本没有出现一个值得被 schedule 保护的正信号窗口
- `lambda` warmup / hold / decay 在这条线上没有发挥空间

### 7.5 Updated interpretation of the earlier split findings

结合这次 matched 三臂结果，更合理的历史解释是：

- 问题主要在 split 之前的 shared trunk
- 不是 split 之后 branch capacity 不够
- branch 扩容没有解除 main head 与 auxiliary head 对 trunk features 的竞争
- 它只是把竞争之后的 readout 做大了

因此，这轮结果支持把 earlier split findings 重解释为：

> shared trunk capacity / competition 问题，而不是 branch capacity 问题。

## 8. Code changes

- 无训练逻辑改动
- 无 blocker fix
- 本轮仅新增本 record 文档
