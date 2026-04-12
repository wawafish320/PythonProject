# 2026-04-11 R0 recovery static audit

> Postmortem / archive status on 2026-04-11.  
> This document remains valid as a historical static-audit record, but it is **not** the live driver for Stage6 next actions and should not be read as permission to continue the R0 recovery / universal-corrector line as mainline work.  
> Why archived: the clean-workspace sealed-spec audit concluded that legacy corrector recovery is out of scope for the active track; current execution is governed by the `direct_pose` stabilization plan and Step B' downstream-sensitive ranking decision.  
> Active repo references: `docs/posttrain_pipeline.md`, `docs/train_design/2026-04-09_top3_anchor_top7_expansion_framework.md`, and `docs/train_design/2026-04-12_top7_clean_stage6_stepc_causality_record.md`.  
> Reader guidance: use this file for archaeology / root-cause context only, not for live sweep design, checkpoint selection, or promotion decisions.
> Paired archived specs:
> - `docs/retired_directions/retired_corrector_recovery_family/2026-04-11_r0_minimal_interface_decoupling_prereg.md`
> - `docs/retired_directions/retired_corrector_recovery_family/2026-04-11_r0_minimal_interface_decoupling_record.md`

> Status: static audit + minimal guard fix  
> Scope: `E1-top3` R0 recovery only; no `branch` full run; no `E2A-R` / `top7` / mixed donor expansion

## 1. Audited paths

- Warmstart helper: `tools/run_cp015_oldplan_downstream_chain.py:207`
- Train checkpoint load/rebuild/save: `train/posttrain.py:7063`, `train/posttrain.py:7927`, `train/posttrain.py:6217`
- Shared train/eval arm-residual application: `train/posttrain.py:3062`
- Eval checkpoint load/rebuild: `train/validate/run_freerun_cycles.py:638`, `train/validate/run_freerun_cycles.py:1749`
- Relevant R0 configs: `debug_output/_tmp_r0_minimal_interface_decoupling_20260411/configs/posttrain_70b_replace_lowdrift_e3x60_r0{baseline,sham_lr0,branch}_top3_20260411.json`

## 2. Warmstart conclusion

`create_replace_zerophase_warmstart(...)` is copy-only: it loads `src_ckpt`, shallow-copies the top-level checkpoint dict, and writes it to `dst_ckpt` without state-dict tensor surgery.

Static artifact check:

- `src_donor` vs R0 `warmstart`: `145` common model tensors, `0` missing, `0` extra, `0` changed.
- Therefore the warmstart helper itself bitwise preserves donor state, including registered buffers.

## 3. Parameter / buffer action table

| parameter/buffer name | source path / function | action | residual=0 bitwise preserve donor? | risk |
|---|---|---|---|---|
| all `model.*` tensors in warmstart (`145` tensors) | `tools/run_cp015_oldplan_downstream_chain.py:207` | copied by `torch.save(dict(obj))` | yes | none |
| checkpoint metadata / `posttrain_cfg` in warmstart | `tools/run_cp015_oldplan_downstream_chain.py:213` | copied top-level payload | yes | none |
| `direct_pose_head.*`, `direct_pose_head_{leg,arm,else}.*` | `train/posttrain.py:7212` / `train/posttrain.py:7756` | copied unless direct-head shape override/reinit requests drop | yes in R0 sham/branch; no in `baseline_locked` by design | possible |
| `direct_pose_out_{leg,nonleg,arm,else}.*` | `train/posttrain.py:7212` / `train/posttrain.py:7756` | copied unless direct split/factorized mismatch triggers drop | yes in R0 sham/branch; trained in `baseline_locked` | possible |
| `direct_pose_{arm,else,nonleg}_proj.*` | `train/posttrain.py:7756` | copied unless direct readout reinit/override | yes in R0 sham/branch | possible |
| `direct_pose_input_adapter*.*` | `train/posttrain.py:7756` | copied unless adapter shape/mode mismatch | yes if present and shape-compatible | possible |
| `direct_pose_leg_head*.*`, `direct_pose_leg_gate_head*.*`, `direct_pose_leg_side_*` | `train/posttrain.py:7798` / `train/posttrain.py:7824` | copied; high-order retired prefixes are dropped; leg bones/mode mismatch can drop | yes for current R0 tensors that remain | possible |
| `direct_pose_leg_out_idx`, `direct_pose_nonleg_out_idx`, `direct_pose_arm_out_idx`, `direct_pose_else_out_idx` | registered buffers in `train/models.py`; loaded via `train/posttrain.py:7928` | copied; can drop only when direct-pose reinit/override drops direct tensors | yes in R0 sham/branch | possible |
| `direct_pose_leg_joint_idx_tensor` | registered buffer in `train/models.py`; loaded via `train/posttrain.py:7928` | copied; can drop on leg bones/mode mismatch | yes in R0 sham/branch | possible |
| optional `direct_pose_leg_side_pos_{r,l}_tensor` | registered buffers in `train/models.py` | retired/dropped in current compat shell | absent in current R0 warmstart | possible |
| `contact_plan_cell.*`, `contact_plan_head.*`, `contact_plan_time_head.*`, `contact_plan_init_z` | `train/posttrain.py:7095` / `train/posttrain.py:7928` | copied | yes | none |
| `contact_plan_init_head.*` | `train/posttrain.py:7484` / `train/posttrain.py:7910` | copied if shape-compatible; dropped on shape mismatch | yes in current R0; guarded after load | possible |
| `event_clock_gate.*`, `event_clock_corrector.*` | `train/posttrain.py:7485` | `event_clock=auto` copies when weights exist; `event_clock=off` would drop on save | yes in current R0 (`auto`) | possible |
| `lambda_fusion_head.*` | `train/posttrain.py:7531` | copied after shape inference | yes if present | possible |
| `so3_delta_corrector.*`, `so3_corr_gate_logit` | model state_dict load | copied if present and shape-compatible | yes if present | possible |
| `arm_residual_corrector.*` | `train/posttrain.py:7876` / `train/models.py:604` | absent in warmstart; fresh-init for R0 unless `arm_residual_use_donor_weight_continuation=true`; donor tensors dropped if present | donor path yes; corrector is intentionally new | none for donor |
| `arm_residual_corrector.gate` | `train/models.py:621` | fresh scalar parameter, initialized `0.0`; branch train can update it | donor path yes when gate=0 | none |
| `frozen_encoder.*`, `frozen_period_head.*` | `train/posttrain.py:7075`, eval strip at `train/validate/run_freerun_cycles.py:652` | stripped from runtime load and supplied by `encoder_bundle`; before fix, save rewrote these checkpoint tensors | runtime eval yes; checkpoint artifact no before fix | confirmed, fixed |
| `contact_plan_input_proj.*` | `train/posttrain.py:7075`, eval strip at `train/validate/run_freerun_cycles.py:652` | stripped from runtime load; now passthrough-preserved on save if present | runtime eval yes | possible, fixed |
| BatchNorm `running_mean/running_var/num_batches_tracked` | `rg BatchNorm/running_*` | not present in audited model path | n/a | none |
| LayerNorm weights/biases | `train/models.py` LayerNorm modules | parameters only; no mutable running buffers | yes if copied | none |

## 4. Dry-run side effect audit

Pre-fix artifact comparison showed:

- `warmstart` vs `dryrun_sham`, excluding `arm_residual_corrector.*`: `8` changed tensors.
- All changed tensors were in `frozen_encoder.*` / `frozen_period_head.*`.
- Largest observed `max_abs_diff`: `0.026925843209028244` on `frozen_period_head.fc.weight`.
- Excluding `arm_residual_corrector.*`, `frozen_encoder.*`, `frozen_period_head.*`, and `contact_plan_input_proj.*`: `0` donor-runtime tensors changed.

Root cause:

- Train load strips `frozen_encoder.*`, `frozen_period_head.*`, `contact_plan_input_proj.*` from checkpoint runtime load.
- The model then attaches `encoder_bundle`.
- The old save path wrote `model.state_dict()`, so the stripped checkpoint passthrough keys were silently replaced by bundle tensors.

Does dry-run checkpoint pollute subsequent eval?

- Current `run_freerun_cycles` eval does **not** consume the polluted `frozen_encoder.*` / `frozen_period_head.*` tensors; it strips them before runtime load.
- So current freerun eval semantics were not polluted.
- However the dry-run checkpoint artifact itself was donor-side non-identity for those passthrough tensors, which is a confirmed artifact-level rewrite.

Fix:

- Added passthrough preservation for stripped checkpoint prefixes before step/full checkpoint save.
- Verification on the new guard dry-run checkpoint: `warmstart` vs `_tmp_guard_verify_sham`, excluding only `arm_residual_corrector.*`, has `0` changed donor tensors, `0` extra, `0` missing.

Critical non-closure:

- This fix does **not** by itself explain the original `baseline_locked` vs `sham_lr0` ~2x eval gap.
- Reason: current freerun eval strips `frozen_encoder.*` / `frozen_period_head.*` before runtime load, so the fixed artifact rewrite is not the obvious runtime path behind the observed metric gap.
- Additional config diff confirms the original two arms were not runtime-equivalent:
  - `baseline_locked`: `train_direct_pose=true`
  - `sham_lr0`: `train_direct_pose=false`, `train_arm_residual=true`
- Therefore the artifact-level bug is real and fixed, but it is **not yet a closed root-cause explanation** for the historical 2x gap.

## 4.1 Historical restart gate before any hypothetical R0 relaunch

At the time, before any hypothetical R0 relaunch, the minimum restart gate would have been:

1. rerun original `baseline_locked` on the fixed pipeline;
2. rerun original `sham_lr0` on the fixed pipeline;
3. compare freerun metrics under the same tolerance used by prereg identity sanity.

Expected readout:

- If the gap disappears, the fixed pipeline closed the practical failure even if the exact mechanism was partially indirect.
- If the gap remains, the artifact rewrite was non-closing and the remaining blocker is elsewhere.

Highest-priority remaining suspects if the gap remains:

- comparison-contract mismatch (`train_direct_pose` vs `train_arm_residual`);
- optimizer param-group injection / ordering side effects;
- schedule / warmup / epoch-budget mismatch;
- eval harness parity (`teacher`, rounds, checkpoint selection, deterministic eval state);
- EMA / shadow-weight asymmetry if any such path exists in the compared lane.

## 5. Runtime guard added

Implemented `assert_donor_bitwise_identity` in `train/posttrain.py:4943`.

Coverage:

- Tensor-level no-op path guard in `_lambda_rollout_apply_arm_residual_adjustments`: `train/posttrain.py:3116`
- Train load guard: `train/posttrain.py:7927`
- Train post-step guard: `train/posttrain.py:6112`
- Train save guard: `train/posttrain.py:6222`
- Eval load guard: `train/validate/run_freerun_cycles.py:1749`

Behavior:

- Hard-fails on first missing / unexpected / mismatched tensor.
- Prints first mismatch tensor and `max_abs_diff`.
- Runtime comparison ignores only intentionally non-runtime passthrough prefixes (`frozen_encoder.*`, `frozen_period_head.*`, `contact_plan_input_proj.*`) and the trainable tail (`arm_residual_corrector.*`).
- No-op SO(3 residual uses `base + (corrected - corrected.detach())`, so gate=0 / all-omega-zero has bitwise forward identity while preserving corrector-gradient flow.

## 6. Verification

Ran:

```bash
python3 -m py_compile train/posttrain.py train/validate/run_freerun_cycles.py train/models.py
```

Ran a minimal 1-step sham dry-run into `_tmp_guard_verify_sham`; it completed and saved.

Ran a checkpoint comparison:

```text
non_arm_changed_count 0
extra_non_arm_keys []
missing_non_arm_keys []
```

Ran a minimal freerun load smoke on the guard checkpoint:

```text
[Done] clips=1 ok / 0 failed
```

No new full experiment was run.
