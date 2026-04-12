# 2026-04-10 DSN auxiliary leg supervision step1 record

> Status: archived / retired aux-family mechanism record
> Reader note: this aux / shared-trunk family did **not** become current repo mainline; any `recommend`, `default`, `ship`, `mainline`, or `current` wording below is historical family-local language only.
> Current entry points: `docs/posttrain_pipeline.md`, `docs/train_design/2026-04-11_donor_contract_basin_universal_debug_synthesis.md`, `docs/retired_directions/aux_shared_trunk_family/2026-04-11_aux_detach_downstream_reevaluation_record.md`

> Status: step1 scaffold landed
> Scope: config schema + model member + optional forward output only / no-loss / no-train / no-export-strip

## 1. Goal
- 为 training-only `auxiliary leg head` 搭结构壳子
- 不改变 baseline inference / downstream contract
- 不接入 `L_aux_leg`

## 2. Files touched
- `train/posttrain.py`
- `train/models.py`
- `docs/retired_directions/aux_shared_trunk_family/2026-04-10_dsn_auxiliary_leg_supervision_step1_record.md`

## 3. Config fields added
- `direct_pose_aux_leg_enable`
- `direct_pose_aux_leg_variant`
- `direct_pose_aux_leg_hidden`
- `direct_pose_aux_leg_detach_feat`
- `direct_pose_aux_leg_weight`
- `direct_pose_aux_leg_loss_mode`
- `direct_pose_aux_leg_warmup_steps`
- `direct_pose_aux_leg_hold_steps`
- `direct_pose_aux_leg_decay_steps`
- `direct_pose_aux_leg_min_weight`
- `direct_pose_aux_leg_log_enable`

## 4. Model wiring
- Attach point: `direct_pose_head` shared trunk output
- Aux output dim source: `direct_pose_leg_out_idx`
- Forward return key: `ret["direct_pose_aux_leg"]`
- Supported aux variants in step1: `linear`, `mlp`
- Step1 note: aux head currently requires shared direct trunk; factorized direct readout path is not wired for aux attach

## 5. Compatibility sanity
- Old ckpt load behavior: old checkpoints without `direct_pose_aux_leg_head.*` can still load; missing aux tensors and leg index buffers are backfilled from current model defaults during `load_state_dict`
- Baseline config unchanged: default config keeps `direct_pose_aux_leg_enable=false`, does not instantiate aux head, and forward schema stays unchanged
- Aux disabled path unchanged: main output path is untouched and aux output is not mixed into main path

## 6. Intentionally not done
- `L_aux_leg`
- aux weight schedule
- optimizer/train-mode changes
- export strip
- `70a -> 70b replace` verification

## 7. Next step
- Step 2: wire `L_aux_leg` + logging, keep deploy contract unchanged
