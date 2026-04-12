# 2026-04-10 DSN auxiliary leg supervision step4 record

> Status: archived / retired aux-family mechanism record
> Reader note: this aux / shared-trunk family did **not** become current repo mainline; any `recommend`, `default`, `ship`, `mainline`, or `current` wording below is historical family-local language only.
> Current entry points: `docs/posttrain_pipeline.md`, `docs/train_design/2026-04-11_donor_contract_basin_universal_debug_synthesis.md`, `docs/retired_directions/aux_shared_trunk_family/2026-04-11_aux_detach_downstream_reevaluation_record.md`

> Status: step4 contract verification landed
> Scope: stripped aux-trained handoff artifact -> real baseline `70a -> new70b_replace_lowdrift` short-chain validation

## 1. Goal

- 验证 Step 3 导出的 stripped handoff artifact 是否真的满足 baseline-compatible contract
- 验证范围只限真实 load / short-run chain：
  - stripped aux-trained `stage6` artifact -> baseline `70a`
  - resulting `70a` artifact -> baseline `new70b_replace_lowdrift`
- 不讨论指标优劣，不扩到 `70R/71/72/lambda`

## 2. Reused contract / assets

- 复用现成 Stage7 config 语义：
  - `config/posttrain_WalkF_stage7_70a_splitB2_pe32h512_20260227_fromarmchain.json`
  - `debug_output/_tmp_posttrain_pipeline_from_bestfree_20260317/configs/posttrain_70b_replace_lowdrift_fromfresh_20260317.json`
- 复用现成本地 baseline assets：
  - base donor: `models/MLPL2_DirectBranch_v1_20260317/exp_phase_DirectBranch_v1_d1_20260317/ckpt_best_free_exp_phase_DirectBranch_v1_d1_20260317.pth`
  - encoder bundle: `models/motion_encoder_equiv_20260317.pt.best.pt`
  - pretrain-contact affine stats: `debug_output/_tmp_phaseb_affine_20260304/affine_fit_mix08/affine_stats.json`
- replace warmstart 复用现有 helper：
  - `tools.run_cp015_oldplan_downstream_chain.create_replace_zerophase_warmstart(...)`

## 3. Commands run

- `python3 -m train.posttrain --config config/posttrain_WalkF_stage6_direct_cond_anchor_splitfirst_3way_armchain_pe32h512_20260227.json --ckpt_in models/MLPL2_DirectBranch_v1_20260317/exp_phase_DirectBranch_v1_d1_20260317/ckpt_best_free_exp_phase_DirectBranch_v1_d1_20260317.pth --out_dir models/__tmp_dsn_aux_leg_step4_20260410/stage6_aux --run_name WalkF_stage6_auxleg_contract_smoke_20260410 --epochs 1 --steps_per_epoch 1 --encoder_bundle models/motion_encoder_equiv_20260317.pt.best.pt --posttrain_contacts_source pretrain_contact --posttrain_contacts_pretrain_clamp 1.0 --posttrain_contacts_pretrain_affine_stats debug_output/_tmp_phaseb_affine_20260304/affine_fit_mix08/affine_stats.json --direct_pose_aux_leg_enable true --direct_pose_aux_leg_weight 0.2 --direct_pose_aux_leg_log_enable true`
- `python3 - <<'PY' ... torch.load(stage6_train/stage6_strip) ... PY`
- `python3 -m train.posttrain --config config/posttrain_WalkF_stage7_70a_splitB2_pe32h512_20260227_fromarmchain.json --ckpt_in models/__tmp_dsn_aux_leg_step4_20260410/stage6_aux/ckpt_last_WalkF_stage6_auxleg_contract_smoke_20260410.pth --out_dir models/__tmp_dsn_aux_leg_step4_20260410/70a --run_name WalkF_stage7_70a_from_stage6_auxleg_contract_smoke_20260410 --epochs 1 --steps_per_epoch 1 --encoder_bundle models/motion_encoder_equiv_20260317.pt.best.pt --posttrain_contacts_source pretrain_contact --posttrain_contacts_pretrain_clamp 1.0 --posttrain_contacts_pretrain_affine_stats debug_output/_tmp_phaseb_affine_20260304/affine_fit_mix08/affine_stats.json`
- `python3 - <<'PY' ... create_replace_zerophase_warmstart(...) ... PY`
- `python3 -m train.posttrain --config debug_output/_tmp_posttrain_pipeline_from_bestfree_20260317/configs/posttrain_70b_replace_lowdrift_fromfresh_20260317.json --ckpt_in models/__tmp_dsn_aux_leg_step4_20260410/warmstart/ckpt_last_70a_replace_zerophase_20260410.pth --out_dir models/__tmp_dsn_aux_leg_step4_20260410/70b_replace_lowdrift --run_name WalkF_stage7_70b_replace_lowdrift_from_stage6_auxleg_contract_smoke_20260410 --epochs 1 --steps_per_epoch 1`
- `python3 - <<'PY' ... torch.load(stage6_train/stage6_strip/stage70a/stage70b_replace) ... PY`

## 4. Actual results

### 4.1 Aux-trained `stage6` export / strip

- aux-enabled short-run completed successfully:
  - output handoff: `models/__tmp_dsn_aux_leg_step4_20260410/stage6_aux/ckpt_last_WalkF_stage6_auxleg_contract_smoke_20260410.pth`
  - output train artifact: `models/__tmp_dsn_aux_leg_step4_20260410/stage6_aux/ckpt_last_train_WalkF_stage6_auxleg_contract_smoke_20260410.pth`
- CLI log explicitly reported:
  - saved train artifact with aux tensors
  - stripped `2` `direct_pose_aux_leg_head.*` tensors for handoff artifact
- static inspection:
  - train artifact: `aux_key_count=2`, `direct_pose_aux_leg_enable=True`
  - stripped artifact: `aux_key_count=0`, `direct_pose_aux_leg_enable=False`, `direct_pose_aux_leg_weight=0.0`, `direct_pose_aux_leg_log_enable=False`

### 4.2 Stripped `stage6` artifact -> baseline `70a`

- baseline `70a` short-run completed successfully:
  - output: `models/__tmp_dsn_aux_leg_step4_20260410/70a/ckpt_last_WalkF_stage7_70a_from_stage6_auxleg_contract_smoke_20260410.pth`
- no aux-specific loader blocker appeared
- resulting `70a` artifact inspection:
  - `aux_key_count=0`
  - `direct_pose_aux_leg_enable=False`

### 4.3 `70a` artifact -> baseline `new70b_replace_lowdrift`

- warmstart helper completed successfully:
  - warmstart ckpt: `models/__tmp_dsn_aux_leg_step4_20260410/warmstart/ckpt_last_70a_replace_zerophase_20260410.pth`
  - report: `debug_output/_tmp_dsn_aux_leg_step4_20260410/warmstart/replace_zerophase_report.json`
  - helper report confirms current recipe semantics are a raw copy (`copied_without_phase_z_direct_adaptation=true`)
- baseline replace short-run completed successfully:
  - output: `models/__tmp_dsn_aux_leg_step4_20260410/70b_replace_lowdrift/ckpt_last_WalkF_stage7_70b_replace_lowdrift_from_stage6_auxleg_contract_smoke_20260410.pth`
- resulting replace artifact inspection:
  - `aux_key_count=0`
  - `direct_pose_aux_leg_enable=False`

## 5. Warnings observed

- `tqdm not found` warning appeared in each CLI run; non-blocking
- `checkpoint has contact_plan_init_head weights; overriding contact_plan_init_mode -> learnable+obs` appeared during loads; existing compat behavior, non-blocking
- `ckpt period_dim=32 but Event-Clock period_feat_dim=0` info appeared during loads; existing compat behavior, non-blocking

## 6. Verdict

- stripped aux-trained artifact **can** serve as a baseline-compatible handoff
- baseline `70a` path **can** load and continue from the stripped aux-trained `stage6` artifact
- baseline `70a -> new70b_replace_lowdrift` handoff **has been verified**
- Step 4 success criterion is met:
  - **yes**, `70a -> 70b replace` baseline-contract verification passed

## 7. Not claimed

- 本轮没有做 full-length training
- 本轮没有做 metric verdict / recipe comparison
- 本轮没有扩到 `70R/71/72/lambda`
- 本轮没有改 downstream training logic，本次验证未暴露需要修复的 contract blocker
