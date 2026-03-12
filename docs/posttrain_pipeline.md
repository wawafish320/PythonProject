# Posttrain Pipeline (Newflow Main Entry)

> Last updated: 2026-03-09  
> Status: **main entry** for posttrain runs in this repo（runtime/docs synced to current `pretrain_contact + affine_mix08` operational line）.

This document only keeps the Stage6→Stage7 newflow path:

`Stage6 (split-first/armchain) -> Stage7 70a -> 70b_concat -> 70c_replacecontacts (historical reference shell) -> promoted 70R (low-LR trunkfull s180) -> 71 -> 72 -> lambda final`

Legacy Stage1-5/hinge/contact-hazard training docs are no longer part of the main path, and
legacy posttrain targets are retired in current `train.posttrain`.  
They are kept for historical reproduction only (see [Section 6](#6-legacy-repro-only)).

---

## 1) Source of Truth

- Newflow handoff (experiment fact source):
  - `docs/Problems/active/2026-02-26_stage6_stage7_newflow_handoff.md`
- Pretrain-contact route handoff / acceptance record:
  - `docs/Problems/active/2026-03-04_pretrain_contact_route_debug_handoff.md`
- Pretrain-contact route cleanup readiness / current accepted line:
  - `docs/Problems/active/2026-03-05_pretrain_contact_route_cleanup_readiness.md`
- Current Stage7 promote-chain report / progress-regression checklist:
  - `docs/Problems/active/2026-03-08_posttrain_s180_promote_regression_progress_checklist.md`
- Base-flow simplification review record:
  - `docs/Problems/active/2026-03-02_trainbase_simplify_review.md`
- Trainbase v2 core/patch design (governance entry):
  - `docs/trainbase_design/2026-03-02_trainbase_v2_core_patch_flow.md`
- Active config runlist (authoritative active chain list):
  - `docs/delete/2026-03-01_posttrain_active_whitelist_runtime.txt`
- contact_phase_state 移除计划/执行记录：
  - `docs/delete/2026-03-03_trainbase_remove_contact_phase_state_refactor_plan.md`
  - `docs/delete/2026-03-03_trainbase_remove_contact_phase_state_phaseA_report.md`
- contact-meas provider 移除计划/执行记录：
  - `docs/delete/2026-03-03_posttrain_remove_contact_meas_provider_plan.md`
  - `docs/delete/2026-03-03_training_MPL_contact_signal_decouple_plan.md`

---

## 2) Newflow Policy (must follow)

- `train.posttrain` main target contract is still XOR: `train_direct_pose` or `train_lambda_head`.
- Legacy Stage1-5 target keys are no longer part of mainline config schema and are not used by runtime.
- Active configs must not contain any `direct_pose_hinge_*`, `contact_td_hazard_*`, or `contact_ttc_*` keys (retired shell; runtime reject/no compatibility fallback).
- `direct_pose` 高阶支线 `direct_pose_leg_side_*` 与 `direct_pose_loss_sic*` 已从当前 posttrain mainline 退休；runtime 只接受 inert defaults，不再允许 active 启用。
- `contact_phase_state` 已从 trainbase 主链移除：mainline runtime 代码与 active configs 均禁止出现 `contact_phase_state*` 键/语义。
- 当前 `train.posttrain` rollout contacts 路由固定为 `pretrain_contact`；不再把 `whitebox` 视为当前 posttrain runtime 默认线。
- 当前推荐的主线运行口径为：`pretrain_contact + clamp1 + affine_mix08`。
- `--encoder_bundle` 是 `pretrain_contact` 路由的必需资产；当缺少可用 `encoder/contact_head` 时，runtime fail-fast 直接报错退出（禁止 silent fallback）。
- `--posttrain_contacts_pretrain_affine_stats` 当前主线建议固定为 `debug_output/_tmp_phaseb_affine_20260304/affine_fit_mix08/affine_stats.json`。
- 历史 `whitebox` 路由已退休；仅作为 archive/reference 记录保留，不再作为 validate/control 执行建议。
- `posttrain` 入口不再定义 `contact_meas_provider*` 语义（source A/B 仅在 validate lane）。
- 如需来源 A/B，对比入口保留在 `train.validate.run_freerun_cycles --contacts_meas_source {pretrain_contact|model|gt|zero}`。
- Mainline rollout API no longer accepts `direct_hinge_delta` (`train/posttrain.py` / `train/training_MPL.py` / `train/eval_utils.py` are hinge-free).
- Active configs should pass XOR guard:

```bash
python3 tools/check_posttrain_newflow_active_configs.py
```

- Optional static anti-regression guard for `train/posttrain.py`:

```bash
python3 tools/check_posttrain_legacy_code_guard.py
```

---

## 3) Recommended Main Chain (s180-promote mainline)

Run in this exact order.

> 2026-03-08 update: the accepted Stage7 downstream line no longer continues from the plain `new70R` recipe.  
> The current passing route is:  
> `Stage6 -> 70a -> 70b_concat -> 70c_replacecontacts (historical reference shell) -> promoted 70R (low-LR trunkfull s180) -> 71 -> 72 -> lambda final`

Operational rule:

- Keep Stage6 / 70a / `70b_concat` / `70c_replacecontacts` as the upstream reference path, but do not read the historical file names as the current semantic stage names.
- Historical filename mapping:
  - `70b_concat` -> `config/posttrain_WalkF_stage7_70b_phasezin_splitB2_pe32h512_20260227_fromarmchain.json`
  - `70c_replacecontacts` -> `config/posttrain_WalkF_stage7_70c_replacecontacts_splitB2_pe32h512_20260227_fromarmchain.json`
- Current promote step no longer hands off from the plain historical `70c_replacecontacts` shell; it promotes from the generated `new70b_replace` config:
  - `debug_output/_tmp_70R_lowlr_from_new70b_20260308/posttrain_70R_from_new70b_replace_lr3e4_e1_s60_20260308.json`
- Treat plain `config/posttrain_WalkF_stage7_70R_nonleg_recovery_proj256_preleg_20260227_fromarmchain.json` as a historical/reference recipe, not the current accepted downstream handoff.
- For current mainline downstream runs, promote the accepted `70R` replacement checkpoint:
  - `models/__tmp_70R_new_lowlr_trunkfull_s180_20260308/ckpt_last_WalkF_stage7_70R_new_lowlr_trunkfull_s180_20260308.pth`
- Promotion evidence / downstream acceptance:
  - `debug_output/_tmp_70R_lowlr_trunkfull_s180_rounds5_20260308/s180_verdict.md`
  - `debug_output/_tmp_chain_s180promote_20260308/chain_verdict.md`
  - `docs/Problems/active/2026-03-08_posttrain_s180_promote_regression_progress_checklist.md`

Recommended artifact chain:

1. `config/posttrain_WalkF_stage6_direct_cond_anchor_splitfirst_3way_armchain_pe32h512_20260227.json`
2. `config/posttrain_WalkF_stage7_70a_splitB2_pe32h512_20260227_fromarmchain.json`
3. `70b_concat`（historical filename: `config/posttrain_WalkF_stage7_70b_phasezin_splitB2_pe32h512_20260227_fromarmchain.json`）
4. `70c_replacecontacts`（historical filename: `config/posttrain_WalkF_stage7_70c_replacecontacts_splitB2_pe32h512_20260227_fromarmchain.json`；historical reference shell only）
5. Promote `70R` with `tools/run_posttrain_nonleg_trunk_ablation.py` from generated `new70b_replace` config into:
   - `models/__tmp_70R_new_lowlr_trunkfull_s180_20260308/ckpt_last_WalkF_stage7_70R_new_lowlr_trunkfull_s180_20260308.pth`
6. Continue `71` from promoted `70R`:
   - base config: `config/posttrain_WalkF_stage7_71_legonly_after_nonlegproj256_20260227_fromarmchain.json`
   - accepted output: `models/__tmp_71_from_s180_70R_20260308/ckpt_last_WalkF_stage7_71_from_s180_70R_20260308.pth`
7. Continue `72` from promoted `71`:
   - base config: `config/posttrain_WalkF_stage7_72_legomega_after_nonlegproj256_20260227_fromarmchain.json`
   - accepted output: `models/__tmp_72_from_s180_71_20260308/ckpt_last_WalkF_stage7_72_from_s180_71_20260308.pth`
8. Continue `lambda final` from promoted `72`:
   - base config: `config/posttrain_WalkF_stage7_lambda_final_calib_20260227_fromarmchain_fullcompat.json`
   - accepted output: `models/__tmp_lambda_from_s180_72_20260308/ckpt_last_WalkF_stage7_lambda_from_s180_72_20260308.pth`

Commands (current accepted route):

```bash
ENCODER_BUNDLE=models/motion_encoder_equiv.pt.best.pt
AFFINE_STATS=debug_output/_tmp_phaseb_affine_20260304/affine_fit_mix08/affine_stats.json
PRETRAIN_CLAMP=1.0

# Stage6
PYTHONPATH=. python -m train.posttrain --config config/posttrain_WalkF_stage6_direct_cond_anchor_splitfirst_3way_armchain_pe32h512_20260227.json   --posttrain_contacts_source pretrain_contact   --posttrain_contacts_pretrain_clamp "${PRETRAIN_CLAMP}"   --encoder_bundle "${ENCODER_BUNDLE}"   --posttrain_contacts_pretrain_affine_stats "${AFFINE_STATS}"
# 70a
PYTHONPATH=. python -m train.posttrain --config config/posttrain_WalkF_stage7_70a_splitB2_pe32h512_20260227_fromarmchain.json   --posttrain_contacts_source pretrain_contact   --posttrain_contacts_pretrain_clamp "${PRETRAIN_CLAMP}"   --encoder_bundle "${ENCODER_BUNDLE}"   --posttrain_contacts_pretrain_affine_stats "${AFFINE_STATS}"
# historical 70b_concat (filename still carries old `70b_phasezin` naming)
PYTHONPATH=. python -m train.posttrain --config config/posttrain_WalkF_stage7_70b_phasezin_splitB2_pe32h512_20260227_fromarmchain.json   --posttrain_contacts_source pretrain_contact   --posttrain_contacts_pretrain_clamp "${PRETRAIN_CLAMP}"   --encoder_bundle "${ENCODER_BUNDLE}"   --posttrain_contacts_pretrain_affine_stats "${AFFINE_STATS}"
# historical 70c_replacecontacts shell (reference only; current promote step does not hand off from this shell)
PYTHONPATH=. python -m train.posttrain --config config/posttrain_WalkF_stage7_70c_replacecontacts_splitB2_pe32h512_20260227_fromarmchain.json   --posttrain_contacts_source pretrain_contact   --posttrain_contacts_pretrain_clamp "${PRETRAIN_CLAMP}"   --encoder_bundle "${ENCODER_BUNDLE}"   --posttrain_contacts_pretrain_affine_stats "${AFFINE_STATS}"

# current accepted promote step: generated `new70b_replace` config
PYTHONPATH=. python tools/run_posttrain_nonleg_trunk_ablation.py   --config debug_output/_tmp_70R_lowlr_from_new70b_20260308/posttrain_70R_from_new70b_replace_lr3e4_e1_s60_20260308.json   --trunk-mode full   --out-dir models/__tmp_70R_new_lowlr_trunkfull_s180_20260308   --run-name WalkF_stage7_70R_new_lowlr_trunkfull_s180_20260308   --epochs 1   --steps-per-epoch 180   --save-step-ckpts 0,1,5,20,60,180

PYTHONPATH=. python -m train.posttrain   --config config/posttrain_WalkF_stage7_71_legonly_after_nonlegproj256_20260227_fromarmchain.json   --ckpt_in models/__tmp_70R_new_lowlr_trunkfull_s180_20260308/ckpt_last_WalkF_stage7_70R_new_lowlr_trunkfull_s180_20260308.pth   --out_dir models/__tmp_71_from_s180_70R_20260308   --run_name WalkF_stage7_71_from_s180_70R_20260308   --posttrain_contacts_source pretrain_contact   --posttrain_contacts_pretrain_clamp "${PRETRAIN_CLAMP}"   --encoder_bundle "${ENCODER_BUNDLE}"   --posttrain_contacts_pretrain_affine_stats "${AFFINE_STATS}"
PYTHONPATH=. python -m train.posttrain   --config config/posttrain_WalkF_stage7_72_legomega_after_nonlegproj256_20260227_fromarmchain.json   --ckpt_in models/__tmp_71_from_s180_70R_20260308/ckpt_last_WalkF_stage7_71_from_s180_70R_20260308.pth   --out_dir models/__tmp_72_from_s180_71_20260308   --run_name WalkF_stage7_72_from_s180_71_20260308   --posttrain_contacts_source pretrain_contact   --posttrain_contacts_pretrain_clamp "${PRETRAIN_CLAMP}"   --encoder_bundle "${ENCODER_BUNDLE}"   --posttrain_contacts_pretrain_affine_stats "${AFFINE_STATS}"
PYTHONPATH=. python -m train.posttrain   --config config/posttrain_WalkF_stage7_lambda_final_calib_20260227_fromarmchain_fullcompat.json   --ckpt_in models/__tmp_72_from_s180_71_20260308/ckpt_last_WalkF_stage7_72_from_s180_71_20260308.pth   --out_dir models/__tmp_lambda_from_s180_72_20260308   --run_name WalkF_stage7_lambda_from_s180_72_20260308   --posttrain_contacts_source pretrain_contact   --posttrain_contacts_pretrain_clamp "${PRETRAIN_CLAMP}"   --encoder_bundle "${ENCODER_BUNDLE}"   --posttrain_contacts_pretrain_affine_stats "${AFFINE_STATS}"
```

## 4) Optional Chain (fromarmchain_skip70c)

Use this only when explicitly testing the skip70c branch:

- `config/posttrain_WalkF_stage7_70R_nonleg_recovery_proj256_preleg_20260228_fromarmchain_skip70c.json`
- `config/posttrain_WalkF_stage7_71_legonly_after_nonlegproj256_20260228_fromarmchain_skip70c.json`
- `config/posttrain_WalkF_stage7_72_legomega_after_nonlegproj256_20260228_fromarmchain_skip70c.json`
- `config/posttrain_WalkF_stage7_lambda_final_calib_20260228_fromarmchain_skip70c_fullcompat.json`

When running this branch, keep the same `ENCODER_BUNDLE` / `AFFINE_STATS` / `PRETRAIN_CLAMP` contract as Section 3.

---

## 4.5) Experimental Branch (`71m` curriculum; not mainline)

Use this only for isolated `70R -> 71m` A/B after the upstream root cause is already locked at `current70R -> new70R`.
As of `2026-03-08`, do **not** replace the accepted `s180-promote -> 71 -> 72 -> lambda` route with this branch.

Status update (`2026-03-08 PM`):

- keep the accepted mainline unchanged: `70R(s180) -> 71 -> 72 -> lambda final`;
- if an experimental downstream lane is still needed, the retained lane is `70R(s180) -> 71m -> 72micro_s70 -> lambda final`;
- `72micro_hybridcarrytrain` / `72micro_hybridcarry_s70` should be treated as **diagnostic-only** root-cause probes for cross-cycle `pose_hist` carry, not as a new active handoff lane;
- reason: the hybridcarry lane confirms the carry diagnosis, but after closing the chain to `lambda final` it only brings marginal changes over plain `72micro_s70`, while adding branch/runtime complexity and still not clearing the accepted foot/calf hotspot bar.

- `config/posttrain_WalkF_stage7_71m_legcurriculum_proj10_lr3e4_e1_s60_20260308_fromarmchain.json`

Design intent:

- keep current `leg_train_only=true` + `replace_contacts` + split/head structure unchanged;
- merge old `71` plain leg loss and old `72` align loss into one stage;
- use `align_mode=proj` with a conservative schedule: warmup `15` steps at weight `0`, ramp `25` steps, target `10`;
- keep the run short (`lr=3e-4`, `epochs=1`, `steps_per_epoch=60`) so it stays a minimal debug branch.

Command:

```bash
ENCODER_BUNDLE=models/motion_encoder_equiv.pt.best.pt
AFFINE_STATS=debug_output/_tmp_phaseb_affine_20260304/affine_fit_mix08/affine_stats.json
PRETRAIN_CLAMP=1.0

PYTHONPATH=. python -m train.posttrain --config config/posttrain_WalkF_stage7_71m_legcurriculum_proj10_lr3e4_e1_s60_20260308_fromarmchain.json \
  --posttrain_contacts_source pretrain_contact \
  --posttrain_contacts_pretrain_clamp "${PRETRAIN_CLAMP}" \
  --encoder_bundle "${ENCODER_BUNDLE}" \
  --posttrain_contacts_pretrain_affine_stats "${AFFINE_STATS}"
```

---

## 4.6) Experimental Branch (`72_micro s70`; best-overall candidate)

Use this only as the follow-up micro-stage after the experimental `71m` branch.
It is still **not** the recommended mainline replacement for the accepted `s180-promote -> 71 -> 72` route.

- `ckpt_in`: `models/__tmp_71m_from_new70R_20260308/ckpt_last_WalkF_stage7_71m_from_new70R_20260308.pth`
- best candidate: `models/__tmp_72micro_from_71m_20260308/ckpt_last_WalkF_stage7_72micro_from_71m_lr1e4_e1_s70_20260308.pth`
- eval: `debug_output/_tmp_72micro_from_71m_s70_eval_20260308_Walk_F/Walk_F_freerun_cycles.json`
- retained experimental final candidate from the accepted `s180` upstream: `models/__tmp_lambda_from_s180_72micro_20260308/ckpt_last_WalkF_stage7_lambda_from_s180_72micro_s70_20260308.pth`
- canonical blend-aware compare vs accepted final: `debug_output/_tmp_lambdafull_vs_lambdamicro_from_s180_s70_blend_recheck_20260308_Walk_F/gate_metrics.json`
- note: this `s70` is the original `lr=1e-4` sweep winner from the direct `71m -> 72_micro` lane; it does **not** refer to the later low-lr continuation `s70` (`models/__tmp_72micro_tail_from_s60_lr5e5_20260308/..._s70_cont_froms60_lr5e5_20260308.pth`).

Operational note (`2026-03-08 PM`):

- keep this lane as the **preferred preserved experimental branch** if future `71m/72_micro` follow-up work is needed;
- do **not** switch the accepted mainline handoff to this lane yet, because final-`lambda` overall improves but `foot_l/ball_l` and `calf_r` hotspots are still worse than the accepted final;
- do **not** continue the `hybridcarrytrain` variant as a parallel active branch; archive its evidence as root-cause confirmation only, then remove it from active experimentation/documentation when convenient.

Design intent:

- keep the current `72` objective (`align_mode=proj`, `align_weight=20`) unchanged;
- only reduce the optimization budget to make `72` a micro-stage from `71m`;
- use `lr=1e-4`, `epochs=1`, `steps_per_epoch=70` as the current best-overall point from the `s10/s20/s30/s40/s50/s60/s70` sweep.

Observed status (same masked DirectGeoLocalDeg summary protocol):

- vs `71m`: `global_mean_rel_delta_pct = -4.5916%`, `leg8_mean_delta = -0.038272`, `non_leg_mean_delta = 0.0`
- vs current `72`: `global_mean_rel_delta_pct = -0.4813%`, `leg8_mean_delta = -0.003846`, `non_leg_mean_delta = 0.0`
- tradeoff: `SIC12-15 foot_l/ball_l` still stays much better than full `72` (`0.8872 -> 0.4356`), while `calf_r` still does not fully recover to full-`72` level (`0.2664 -> 0.3270`).

Command:

```bash
ENCODER_BUNDLE=models/motion_encoder_equiv.pt.best.pt
AFFINE_STATS=debug_output/_tmp_phaseb_affine_20260304/affine_fit_mix08/affine_stats.json

PYTHONPATH=. python -m train.posttrain \
  --config config/posttrain_WalkF_stage7_72_legomega_after_nonlegproj256_20260227_fromarmchain.json \
  --ckpt_in models/__tmp_71m_from_new70R_20260308/ckpt_last_WalkF_stage7_71m_from_new70R_20260308.pth \
  --out_dir models/__tmp_72micro_from_71m_20260308 \
  --run_name WalkF_stage7_72micro_from_71m_lr1e4_e1_s70_20260308 \
  --lr 1e-4 \
  --epochs 1 \
  --steps_per_epoch 70 \
  --posttrain_contacts_source pretrain_contact \
  --posttrain_contacts_pretrain_clamp 1.0 \
  --encoder_bundle "${ENCODER_BUNDLE}" \
  --posttrain_contacts_pretrain_affine_stats "${AFFINE_STATS}"
```

---

## 5) Validation Checklist

After chain completion:

1. Run static no-hinge/no-phase-state/no-provider checks:

```bash
if rg -n "direct_pose_hinge|direct_hinge_delta|contact_phase_state|contact_meas_provider"   train/posttrain.py train/models.py train/training_MPL.py train/eval_utils.py train/validate/run_freerun_cycles.py; then
  echo "[FAIL] legacy hinge/phase-state/provider references found"
  exit 1
fi
python3 -m py_compile train/posttrain.py train/models.py train/training_MPL.py train/eval_utils.py train/validate/run_freerun_cycles.py
```

2. Run active-config XOR guard:

```bash
python3 tools/check_posttrain_newflow_active_configs.py
```

3. Run optional static anti-regression guard:

```bash
python3 tools/check_posttrain_legacy_code_guard.py
```

4. If reproducing the accepted `70R` promote step, run the promotion gate on `rounds=5` before continuing downstream:

- export freerun JSON with both `direct_arm_probe` and `per_step_direct_geolocal_deg` enabled;
- compare `current70R / new70R / promoted70R` with `tools/compare_direct_arm_probe.py`;
- keep the same summary keys throughout: `legs_main`, `arms_main`, `A_52_59`, `B_76_80`;
- accepted promotion evidence is recorded in:
  - `debug_output/_tmp_70R_lowlr_trunkfull_s180_rounds5_20260308/s180_verdict.md`

Minimal candidate eval command:

```bash
PYTHONPATH=. python -m train.validate.run_freerun_cycles   --teacher validate/teacher_batches/Walk_F_teacher.json   --model models/__tmp_70R_new_lowlr_trunkfull_s180_20260308/ckpt_last_WalkF_stage7_70R_new_lowlr_trunkfull_s180_20260308.pth   --rounds 5 --phase_reset_source none --lambda_fusion_apply   --export_direct_arm_probe --export_joint_direct_geolocal_series   --out debug_output/_tmp_70R_lowlr_trunkfull_s180_rounds5_20260308/eval_cand --force
```

5. Run freerun validation on final ckpt with explicit `pretrain_contact + clamp1 + affine_mix08`:

```bash
ENCODER_BUNDLE=models/motion_encoder_equiv.pt.best.pt
AFFINE_STATS=debug_output/_tmp_phaseb_affine_20260304/affine_fit_mix08/affine_stats.json

PYTHONPATH=. python -m train.validate.run_freerun_cycles   --teacher validate/teacher_batches/Walk_F_teacher.json   --model <ckpt>   --rounds 5 --time-index-mode cycle --depth 3   --event_clock auto --phase_reset_source none   --contacts_meas_source pretrain_contact   --contacts_meas_pretrain_clamp 1.0   --contacts_meas_pretrain_affine_stats "${AFFINE_STATS}"   --encoder-bundle "${ENCODER_BUNDLE}"   --lambda_fusion_apply --log_contacts   --export_direct_arm_probe --export_joint_direct_geolocal_series
```

6. Compare the promoted downstream chain against both reference anchors:

- previous `new` chain: `debug_output/_tmp_chain_s180promote_20260308/chain_verdict.md`
- baseline anchors from `2026-03-07`: `docs/Problems/active/2026-03-08_posttrain_s180_promote_regression_progress_checklist.md`
- acceptance wording: use `pass with watchlist`; do not claim strict domination over every baseline.

Optional extra A/B reference (`model` source):

```bash
PYTHONPATH=. python -m train.validate.run_freerun_cycles   --teacher validate/teacher_batches/Walk_F_teacher.json   --model <ckpt>   --rounds 5 --time-index-mode cycle --depth 3   --event_clock auto --phase_reset_source none   --contacts_meas_source model --lambda_fusion_apply --log_contacts   --export_direct_arm_probe --export_joint_direct_geolocal_series
```

## 6) Legacy Repro Only

Legacy documents are **not** mainline training guidance anymore. Use only for historical reproduction/auditing:

- `docs/phase_disambiguation_bridge.md`
- `docs/posttrain_legacy_repro_index.md`

Important:

- On current main branch (after 2026-03-01 C1), old legacy target training modes are retired in `train.posttrain`.
- Do **not** expect legacy commands in those docs to run here directly.
- If you must replay historical legacy chains, use an archived snapshot/branch and keep outputs isolated from newflow baselines.

---

## 7) Change Record

### 2026-03-09（Phase I）

- Retired the remaining runtime/validate `whitebox` contacts route from main-tree execution guidance.
- Removed the historical `whitebox` freerun command from this document and narrowed the documented validate source set to `pretrain_contact|model|gt|zero`.

### 2026-03-08（Phase H）

- Promoted the accepted Stage7 downstream route from plain `new70R -> 71 -> 72 -> lambda` to `s180 low-LR trunkfull 70R promote -> 71 -> 72 -> lambda`.
- Rewrote the mainline chain section to treat plain `70R` config replay as historical/reference only and documented the accepted `70R` promote command plus downstream `--ckpt_in` overrides.
- Updated validation guidance to require the `rounds=5` 70R promotion gate (`legs_main / arms_main / A_52_59 / B_76_80`) before continuing downstream.
- Kept `71m / 72_micro` marked as experimental lanes only; they are no longer the recommended root-cause fix path.
- Clarified the preserved experimental lane as `70R(s180) -> 71m -> 72micro_s70 -> lambda final`, while marking `72micro_hybridcarrytrain` as diagnostic-only and removable after archival.

### 2026-03-06（Phase G）

- Synced policy wording from historical `whitebox default` to current runtime fact: `train.posttrain` now runs on `pretrain_contact` route.
- Promoted `pretrain_contact + clamp1 + affine_mix08` from “explicit experiment” wording to the documented mainline operational recipe.
- Updated recommended chain commands and validation commands to use explicit `pretrain_contact` + affine settings.
- Demoted `whitebox` wording to historical reference only.

### 2026-03-04（Phase E）

- Synced `contact_meas_provider*` retirement status into policy/checklist sections.
- Added then-current validation commands around `whitebox` control / source A/B comparison.
- Added source-of-truth links for provider retirement and training contact-signal decouple plans.

### 2026-03-05（Phase F）

- Added explicit `pretrain_contact` full-chain command template with optional affine calibration stats.
- This wording was later superseded by the 2026-03-06 runtime/doc sync in Phase G.

### 2026-03-03（Phase D）

- Synced contact_phase_state removal status into policy/checklist sections.
- Added explicit no-phase-state static check line in validation checklist.
- Updated source-of-truth links to 2026-03-03 removal plan/report docs.
