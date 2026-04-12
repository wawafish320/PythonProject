# 2026-04-11 R0 minimal interface-decoupling record

> Archived on 2026-04-12.  
> Current role: historical stop-record for the retired R0 minimal interface-decoupling line.  
> Reader guidance: `E1-top3` here is the legacy old-boundary-compatible anchor/control; `70b_replace_lowdrift` is the then-locked comparison target, not the current global canonical chain.

> Status: stopped at no-op identity sanity  
> Scope: `R0` only / `E1-top3 donor` only / no `E2A-R` / no bad `top7` / no mixed donor

## 1. Scope

- Donor: `E1-top3` `70a` only.
- Baseline: then-locked-contract `70b_replace_lowdrift`.
- Sham: frozen-donor residual corrector present, `arm_residual_corrector` optimizer `lr=0`.
- Branch: same residual corrector, normal optimizer.
- Residual design: `y = y_donor + gate * residual(obs)`, learned `gate` init `0.0`, fresh-init residual body, observable detached, donor frozen.

## 2. Artifacts

- Prereg: `docs/retired_directions/retired_corrector_recovery_family/2026-04-11_r0_minimal_interface_decoupling_prereg.md`
- Top3 donor: `models/__tmp_cp015_tailk3_rankmix_tw020_stage70a_from_tailfix_e1_20260408/ckpt_last_WalkF_stage7_70a_lr3e4_from_cp015_tailk3_rankmix_tw020_stage6tailfix_e1_20260408.pth`
- R0 warmstart: `models/__tmp_r0_minimal_interface_decoupling_20260411/warmstart/ckpt_last_e1top3_70a_replace_zerophase_r0_20260411.pth`
- Locked baseline config source: `debug_output/_tmp_cp015_tailk7_replace_schedule_ablation_20260402/configs/posttrain_70b_replace_lowdrift_e3x60_lr5e5_from_cp015_tailk7_70a_20260402.json`

## 3. Configs

- `baseline_locked`: `debug_output/_tmp_r0_minimal_interface_decoupling_20260411/configs/posttrain_70b_replace_lowdrift_e3x60_r0baseline_top3_20260411.json`
- `sham_lr0`: `debug_output/_tmp_r0_minimal_interface_decoupling_20260411/configs/posttrain_70b_replace_lowdrift_e3x60_r0sham_lr0_top3_20260411.json`
- `branch`: `debug_output/_tmp_r0_minimal_interface_decoupling_20260411/configs/posttrain_70b_replace_lowdrift_e3x60_r0branch_top3_20260411.json`

## 4. Code changes

- `train/models.py`
  - Add learned scalar gate support for `ArmResidualCorrector`.
  - Add observable-detach / runtime-assert plumbing and runtime info export.
- `train/posttrain.py`
  - Add config parsing for R0 gate / detach / assert knobs.
  - Add hard asserts for trainable isolation, step0 identity, and no donor gradients.
  - Add exact-identity + straight-through gradient preservation in arm residual rollout application.
- `train/validate/run_freerun_cycles.py`
  - Rebuild eval-side model with arm-residual gate / detach / assert config from checkpoint.

## 5. Commands run

### 5.1 Prep

```bash
python3 -m py_compile train/models.py train/posttrain.py train/validate/run_freerun_cycles.py
```

```bash
python3 - <<'PY'
from pathlib import Path
from tools.run_cp015_oldplan_downstream_chain import create_replace_zerophase_warmstart
create_replace_zerophase_warmstart(
    Path('models/__tmp_cp015_tailk3_rankmix_tw020_stage70a_from_tailfix_e1_20260408/ckpt_last_WalkF_stage7_70a_lr3e4_from_cp015_tailk3_rankmix_tw020_stage6tailfix_e1_20260408.pth'),
    Path('models/__tmp_r0_minimal_interface_decoupling_20260411/warmstart/ckpt_last_e1top3_70a_replace_zerophase_r0_20260411.pth'),
    Path('debug_output/_tmp_r0_minimal_interface_decoupling_20260411/warmstart/replace_zerophase_report.json'),
)
PY
```

### 5.2 Dry-run

```bash
PYTHONPATH=. debug_output/_tmp_phasecd_min_ablation_20260330/cpu_nomps_exec.py -m train.posttrain \
  --config debug_output/_tmp_r0_minimal_interface_decoupling_20260411/configs/posttrain_70b_replace_lowdrift_e3x60_r0sham_lr0_top3_20260411.json \
  --out_dir models/__tmp_r0_minimal_interface_decoupling_20260411/dryrun_sham_lr0 \
  --run_name WalkF_stage7_70b_replace_lowdrift_e3x60_r0sham_lr0_dryrun_top3_20260411 \
  --epochs 1 --steps_per_epoch 1
```

```bash
PYTHONPATH=. debug_output/_tmp_phasecd_min_ablation_20260330/cpu_nomps_exec.py -m train.posttrain \
  --config debug_output/_tmp_r0_minimal_interface_decoupling_20260411/configs/posttrain_70b_replace_lowdrift_e3x60_r0branch_top3_20260411.json \
  --out_dir models/__tmp_r0_minimal_interface_decoupling_20260411/dryrun_branch \
  --run_name WalkF_stage7_70b_replace_lowdrift_e3x60_r0branch_dryrun_top3_20260411 \
  --epochs 1 --steps_per_epoch 1
```

### 5.3 Full runs

```bash
PYTHONPATH=. debug_output/_tmp_phasecd_min_ablation_20260330/cpu_nomps_exec.py -m train.posttrain \
  --config debug_output/_tmp_r0_minimal_interface_decoupling_20260411/configs/posttrain_70b_replace_lowdrift_e3x60_r0baseline_top3_20260411.json
```

```bash
PYTHONPATH=. debug_output/_tmp_phasecd_min_ablation_20260330/cpu_nomps_exec.py -m train.posttrain \
  --config debug_output/_tmp_r0_minimal_interface_decoupling_20260411/configs/posttrain_70b_replace_lowdrift_e3x60_r0sham_lr0_top3_20260411.json
```

`branch` full run was not launched, because the prereg no-op identity gate failed before branch interpretation was allowed.

### 5.4 Eval / summary

```bash
PYTHONPATH=. debug_output/_tmp_phasecd_min_ablation_20260330/cpu_nomps_exec.py -m train.validate.run_freerun_cycles \
  --teacher validate/teacher_batches/Walk_F_teacher.json \
  --model models/__tmp_r0_minimal_interface_decoupling_20260411/baseline_locked/ckpt_last_WalkF_stage7_70b_replace_lowdrift_e3x60_lr5e5_r0baseline_top3_20260411.pth \
  --rounds 5 --depth 3 --time-index-mode cycle \
  --event_clock auto --phase_reset_source none \
  --contacts_meas_source model --lambda_fusion_apply --log_contacts \
  --export_direct_arm_probe --export_joint_direct_geolocal_series \
  --out debug_output/_tmp_r0_minimal_interface_decoupling_20260411/baseline_locked/eval_model_source --force
```

```bash
PYTHONPATH=. debug_output/_tmp_phasecd_min_ablation_20260330/cpu_nomps_exec.py tools/phasea_group_summary.py \
  debug_output/_tmp_r0_minimal_interface_decoupling_20260411/baseline_locked/eval_model_source/Walk_F_freerun_cycles.json \
  --cycle_gte 1 --drop_wrap \
  --out debug_output/_tmp_r0_minimal_interface_decoupling_20260411/baseline_locked/eval_model_source_group_summary.json
```

```bash
PYTHONPATH=. debug_output/_tmp_phasecd_min_ablation_20260330/cpu_nomps_exec.py -m train.validate.run_freerun_cycles \
  --teacher validate/teacher_batches/Walk_F_teacher.json \
  --model models/__tmp_r0_minimal_interface_decoupling_20260411/sham_lr0/ckpt_last_WalkF_stage7_70b_replace_lowdrift_e3x60_r0sham_lr0_top3_20260411.pth \
  --rounds 5 --depth 3 --time-index-mode cycle \
  --event_clock auto --phase_reset_source none \
  --contacts_meas_source model --lambda_fusion_apply --log_contacts \
  --export_direct_arm_probe --export_joint_direct_geolocal_series \
  --out debug_output/_tmp_r0_minimal_interface_decoupling_20260411/sham_lr0/eval_model_source --force
```

```bash
PYTHONPATH=. debug_output/_tmp_phasecd_min_ablation_20260330/cpu_nomps_exec.py tools/phasea_group_summary.py \
  debug_output/_tmp_r0_minimal_interface_decoupling_20260411/sham_lr0/eval_model_source/Walk_F_freerun_cycles.json \
  --cycle_gte 1 --drop_wrap \
  --out debug_output/_tmp_r0_minimal_interface_decoupling_20260411/sham_lr0/eval_model_source_group_summary.json
```

## 6. Runtime sanity

- `zero gate step0 equality`: pass in `dryrun_sham_lr0`, `dryrun_branch`, and full `sham_lr0`; first logged `arm_residual_gate=0.0`, `arm_residual_all_omega_zero=1.0`.
- `observable detach`: pass; first / last `arm_residual_obs_requires_grad_pre=0.0`, `arm_residual_obs_requires_grad_post=0.0` in `sham_lr0`.
- `donor frozen`: pass; trainable isolation assert allowed only `arm_residual_corrector.*`.
- `no donor grad`: pass in dry-runs and full `sham_lr0`; backward finished without gradient-path fatal.
- Implementation note: the first dry-run exposed a gate=0 graph-disconnect issue; `train/posttrain.py` was patched to keep exact identity forward via `correction - correction.detach()` while preserving gate-gradient flow.

## 7. Metrics

| arm | all_ex_root mean | all_ex_root p95 | leg p95 | note |
|---|---:|---:|---:|---|
| `donor_noop` | `0.412148303` | `1.658376932` | `2.324671268` | reused existing `E1-top3 70a` eval; warmstart is copy-only |
| `baseline_locked` | `0.234223005` | `0.893541753` | `1.560555220` | locked-contract replace trained on top3 warmstart |
| `sham_lr0` | `0.412148303` | `1.658376932` | `2.324671268` | residual structure, lr=0 |
| `branch` | not run | not run | not run | blocked by no-op identity failure |

## 8. Acceptance-gate readout

- Identity sanity (`baseline_locked` vs `sham_lr0`): fail.
  - `all_ex_root mean`: `0.234223005` vs `0.412148303`; abs diff `0.177925298`, rel `75.964057%`, tolerance `0.002342230`.
  - `all_ex_root p95`: `0.893541753` vs `1.658376932`; abs diff `0.764835179`, rel `85.595908%`, tolerance `0.008935418`.
- Supplemental donor sanity (`donor_noop` vs `sham_lr0`): pass exactly on headline metrics.
  - `all_ex_root mean`: diff `0.0`, tolerance `0.004121483`.
  - `all_ex_root p95`: diff `0.0`, tolerance `0.016583769`.
- Gradient-path assert: pass for dry-run branch / sham and full sham.
- Branch vs baseline: not evaluated.
- Branch vs sham: not evaluated.
- Leg veto: not evaluated.
- Final classification: prereg strict stop = `pipeline/data-path leak` because `baseline_locked ≉ sham_lr0`; supplemental readout says the residual no-op path itself is identity-clean (`donor_noop == sham_lr0`), so the actionable blocker is the comparison contract between trained locked baseline and no-op frozen residual sham, not an observed donor-gradient leak.
