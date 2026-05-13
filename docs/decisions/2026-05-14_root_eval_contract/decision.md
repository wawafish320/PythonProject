# Root Eval Contract Closeout

Date: 2026-05-14

## Problem

Root was reintroduced into free-run evaluation and produced two different symptoms:

- pose drift appeared severe when final lambda checkpoints were evaluated with `lambda_fusion_apply=false`;
- raw root/world translation error grew almost linearly across cycles, even when pose and velocity contracts were stable.

The branch question was whether to fix root carry / train a root-specific objective, or to correct the evaluator contract and promotion gates.

## Decision

1. Final lambda checkpoints must be evaluated with lambda fusion applied.

   A checkpoint is treated as final-lambda when `state_dict` contains `lambda_fusion_head.*` and `posttrain_cfg` has both `lambda_fusion_enable=true` and `train_lambda_head=true`. In that case, `run_freerun_cycles` fails fast unless `--lambda_fusion_apply` is set. Explicit ablations may use `--allow_lambda_apply_off_ablation`; those outputs are marked `valid_for_final_conclusion=false`.

2. Root/world translation metrics are layered instead of treated as one drift scalar.

   - `RootPosErrRaw*`: absolute world placement / continuity error.
   - `RootPosErrOffsetCorrected*`: root position error after adding `cycle_idx * cycle_net_disp` to tiled GT.
   - `RootDispErrStartToCurrent*`: displacement-from-first-aligned-step error between predicted walk and offset-corrected GT walk.
   - `RootStepDispErr*`: per-step displacement increment error.

3. Do not change root carry or add targeted root-trajectory training from this branch.

   Current evidence points to evaluator placement convention plus missing lambda application, not a root carry integration bug. Root carry/runtime placement should remain a monitoring contract, not a promotion blocker for this closeout.

## Evidence

Pose contract, final lambda checkpoint:

- ckpt: `debug_output/_tmp_71_lr1e4_lowlr_downstream_20260504/lambda/checkpoints/ckpt_last_WalkF_stage7_lambda_from_lowlr72_lr1e4_20260504.pth`
- artifact: `debug_output/_tmp_root_drift_123_verify_20260514_0057/lambda_closed_loop_summary.csv`
- baseline without apply, round4: `GeoLocalDeg=87.6344`
- natural/apply, round4: `GeoLocalDeg=0.4755`, `LambdaEffMean=0.957674`
- artifact: `debug_output/_tmp_root_include_root_check_lambda_apply_20260514_0106/pose_include_vs_exroot_per_round.csv`
- canonical carried blend, round4: `BlendGeoLocalDeg_ex_root=0.093436`, `DirectGeoLocalDeg_ex_root=0.093424`

Root/world translation contract:

- artifact: `debug_output/_tmp_root_world_translation_contract_20260514_0326/root_translation_renamed_per_round.csv`
- round4 raw vs offset-corrected: `RootPosErrRawMean=3.9915m`, `RootPosErrOffsetCorrectedMean=0.03958m`
- artifact: `debug_output/_tmp_root_world_translation_contract_20260514_0332/root_walk_displacement_error_summary.csv`
- walk displacement: `DispErrMean=0.003701m`, `DispErrP95=0.007532m`, `DispErrEnd=0.006794m`
- per-step displacement: `StepDispErrMean=0.000311m`, `StepDispErrP95=0.000814m`
- artifact: `debug_output/_tmp_root_world_translation_contract_20260514_0326/root_velocity_gate_summary.csv`
- velocity contract: `RootSpeedRatioMean=0.99927`, `RootSpeedBiasMps=-9.39e-4`, `RootVelDirErrDegP95=1.35e-11deg`

The tensor series used by the offline root contract were `[434,2]`, `float64`, `cpu` for `cond_raw_rootvel`, `carry_rootvel_used`, and `teacher_gt_rootvel`. The evaluator-side root position layering operates on denormalized root position tensors `[B, free_steps, Droot]`; its runtime shape / dtype / device are exported under `rootpos_round_offset_correction.tensor_meta`.

## Implementation Notes

- Guard helper: `train/validate/run_freerun_cycles.py:121`
- Fail-fast enforcement: `train/validate/run_freerun_cycles.py:1291`
- JSON contract fields: `train/validate/run_freerun_cycles.py:2073`
- Root offset correction and displacement metrics: `train/validate/run_freerun_cycles.py:6810`
- Per-step root metric exports: `train/validate/run_freerun_cycles.py:9354`
- Per-round root metric aliases/summaries: `train/validate/run_freerun_cycles.py:9680`
- The lambda guard is intentionally in the evaluator entry, not in `train/rollout_kernel.py` or model code.
- `RootPosErrMean/Start/End` are kept for backward compatibility and aliased as `RootPosErrRaw*`.
- Promotion should use pose metrics from the carried blend path when lambda is applied, plus offset-corrected/root-displacement metrics for root contract monitoring.
- Raw root position error may still be useful for world-continuity placement checks, but not as direct evidence of autoregressive rollout drift.

## Remaining Risk

- The root displacement evidence came from offline contract reconstruction plus evaluator-side denormalized state comparison, not a separate runtime reset-placement rollout.
- Hidden/contact/phase second-order coupling was not exhaustively revalidated after every possible ablation.
- If future runs show `RootStepDispErrP95` or `RootSpeedBiasMps` growing materially while pose remains stable, reopen root velocity magnitude / normalizer checks before changing rot6d or carry integration.
