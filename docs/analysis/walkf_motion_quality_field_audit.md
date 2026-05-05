# WalkF Motion Quality Field Audit

## Scope

Audited files:

- `train/validate/run_freerun_cycles.py`
- `train/rollout_kernel.py`
- `train/training_MPL.py`
- `tools/run_walkf_scheduled_sampling_pilot.py`
- `tools/run_walkf_carry_consistency_pilot.py`

This audit is only about the current eval/training pipeline semantics of `motion_quality`.

## Source Trace

### 1. `motion_quality` is assembled from X-state root slices, not from pose-output metrics

`train/validate/run_freerun_cycles.py` builds `motion_quality` in `_motion_quality_payload(...)` from:

- `pred_x_raw[..., rootvel_slice]` / `gt_x_raw[..., rootvel_slice]`
- `pred_x_raw[..., rootpos_slice]` / `gt_x_raw[..., rootpos_slice]`

See:

- `train/validate/run_freerun_cycles.py:152-225`
- `train/validate/run_freerun_cycles.py:6901-6907`

The payload does not read `predY`, `predY_blend`, `predY_direct`, per-joint geodesic errors, or any other direct pose-quality signal.

### 2. In free-run eval, `predX` is the carried next-state X, not direct model pose output

Eval tiles `cond_tgt_raw` into `cond_seq_raw` and uses it as the per-step raw condition sequence:

- `train/validate/run_freerun_cycles.py:2416-2438`

Inside the rollout loop, each step selects:

- `cond_raw_step = cond_seq_raw[:, t + 1]` when time-major

See:

- `train/validate/run_freerun_cycles.py:4484-4492`

After the model predicts `y_used_raw`, the eval path updates `motion_raw` by calling `apply_free_carry_raw(...)` with `cond_next_raw=cond_raw_step`:

- `train/validate/run_freerun_cycles.py:6719-6729`

Then the carried X-state is appended to `predsX`:

- `train/validate/run_freerun_cycles.py:6750`

Later, `predX` is formed from `predsX`, aligned against `gtX = state_seq[:, t+1]`, and both are denormalized into `predX_raw` / `gtX_raw`:

- `train/validate/run_freerun_cycles.py:6830-6854`

`motion_quality` is then computed from those denormalized X-states:

- `train/validate/run_freerun_cycles.py:6901-6907`

### 3. `apply_free_carry_raw(...)` makes root velocity/position contract-driven

`train/rollout_kernel.py` resolves the free-carry slices from trainer runtime:

- `rootvel_x_slice`
- `rootpos_x_slice`

See:

- `train/rollout_kernel.py:120-160`

The actual free-carry update does three separate things:

1. copy pose rotation from `y_next_raw` into X rot6d slice
2. derive `angvel_x_slice` from consecutive rotations
3. write root velocity and root position from `cond_next_raw`

Critical lines:

- pose copied from model pose output: `x_next[..., rot6d_x_slice] = y_next_raw[..., rot6d_y_slice]`
- root velocity overwritten from condition contract: `x_next[..., rootvel_x_slice] = vel_world`
- root position integrated from that velocity: `x_next[..., rootpos_x_slice] = pos`

See:

- `train/rollout_kernel.py:273-342`

More specifically:

- `cond_next_raw` is parsed as `[..., dir_x, dir_y, speed]` at `train/rollout_kernel.py:306-311`
- `vel_world = dir_unit_world * cond_speed` at `train/rollout_kernel.py:331-333`
- root position is updated by Euler integration from that velocity at `train/rollout_kernel.py:337-341`

Therefore, the root velocity and root position fields in the carried X-state are not model-decided pose outputs. They are contract integration artifacts.

### 4. Training uses the same free-carry kernel

`Trainer` seeds a dedicated scheduled-sampling RNG:

- `train/training_MPL.py:1175-1179`

During rollout training, the trainer stores `latest_cond_raw_for_env` from step inputs:

- `train/training_MPL.py:1913-1914`

Then scheduled-sampling update calls `_rollout_kernel.update_rollout_carry_state(...)`:

- `train/training_MPL.py:1917-1933`

`update_rollout_carry_state(...)` uses the same `apply_free_carry_raw(...)` path with `cond_next_raw=rollout.latest_cond_raw_for_env`:

- `train/rollout_kernel.py:986-997`

So this contract-driven root carry is shared between train-time rollout state updates and eval-time free-run state updates.

### 5. Downstream pilot usage

`tools/run_walkf_scheduled_sampling_pilot.py` explicitly requests `--export_motion_quality`:

- `tools/run_walkf_scheduled_sampling_pilot.py:258-269`

It then consumes `motion_quality` for:

- trajectory diversity based on `pred_root_pos_series`
- aggregated `velocity_ratio_mean`
- aggregated `velocity_ratio_std`
- aggregated `root_displacement_ratio`
- aggregated `freeze_lag_score`

See:

- `tools/run_walkf_scheduled_sampling_pilot.py:290-333`
- `tools/run_walkf_scheduled_sampling_pilot.py:360-375`
- `tools/run_walkf_scheduled_sampling_pilot.py:492-544`

That means the current pilot gate is partly driven by contract artifacts, not by direct model pose behavior.

`tools/run_walkf_carry_consistency_pilot.py` contains no `motion_quality` references and does not currently use these fields for gating.

## Field Classification

Legend:

- `model-driven`: directly reflects model pose/joint output behavior
- `contract-driven`: populated from rollout contract / carry integration, not directly from model pose output
- `composite`: derived from comparing or aggregating other fields

Current-state conclusion: this `motion_quality` payload contains zero `model-driven` fields.

| field | class | source note |
| --- | --- | --- |
| `available` | composite | Boolean availability flag based on presence/shape of `pred_x_raw`, `gt_x_raw`, and root slices |
| `pred_velocity_mean` | contract-driven | Mean of norm of `pred_x_raw[..., rootvel_slice]` |
| `gt_velocity_mean` | contract-driven | Mean of norm of `gt_x_raw[..., rootvel_slice]` |
| `velocity_ratio_mean` | composite | `pred_velocity_mean / gt_velocity_mean` |
| `pred_velocity_std` | contract-driven | Std of norm of `pred_x_raw[..., rootvel_slice]` |
| `gt_velocity_std` | contract-driven | Std of norm of `gt_x_raw[..., rootvel_slice]` |
| `velocity_ratio_std` | composite | `pred_velocity_std / gt_velocity_std` |
| `velocity_step_ratio_mean` | composite | Mean of per-step speed ratio sequence |
| `velocity_step_ratio_std` | composite | Std of per-step speed ratio sequence |
| `freeze_lag_score` | composite | Lag/correlation score derived from predicted vs GT root-speed series |
| `freeze_lag_best_lag_frames` | composite | Argmax lag from predicted vs GT root-speed correlation sweep |
| `freeze_lag_best_corr` | composite | Best correlation from predicted vs GT root-speed correlation sweep |
| `pred_root_speed_series` | contract-driven | Norm of contract-written predicted root velocity |
| `gt_root_speed_series` | contract-driven | Norm of GT root velocity state |
| `velocity_step_ratio_series` | composite | Elementwise ratio of predicted/GT root-speed series |
| `pred_root_path_length` | contract-driven | Sum of step lengths from contract-integrated predicted root positions |
| `gt_root_path_length` | contract-driven | Sum of step lengths from GT root positions |
| `root_displacement_ratio` | composite | `pred_root_path_length / gt_root_path_length` |
| `pred_root_net_displacement` | contract-driven | Net displacement from contract-integrated predicted root positions |
| `gt_root_net_displacement` | contract-driven | Net displacement from GT root positions |
| `root_net_displacement_ratio` | composite | `pred_root_net_displacement / gt_root_net_displacement` |
| `pred_root_pos_series` | contract-driven | Predicted root position series from free-carry integration |
| `gt_root_pos_series` | contract-driven | GT root position series from X-state |

## Required Verdicts For Current Pipeline

Under the current pipeline, the following fields are not valid model anti-cheating gates:

- `pred_root_pos_series`
- `pred_root_speed_series`
- `gt_root_pos_series`
- `gt_root_speed_series`
- `velocity_ratio_mean`
- `velocity_ratio_std`
- `root_displacement_ratio`
- `root_net_displacement_ratio`
- `freeze_lag_score`

Reason:

- `pred_root_pos_series` and `pred_root_speed_series` are contract artifacts produced by `apply_free_carry_raw(...)`, which uses `cond_next_raw` to write root velocity and integrate root position.
- `gt_root_pos_series` and `gt_root_speed_series` are the corresponding GT X-state channels.
- `velocity_ratio_*`, `root_*_ratio`, and `freeze_lag_score` are derived from those root-contract series, so they also do not isolate model pose behavior.

Operationally:

- `velocity_ratio_mean / velocity_ratio_std`: `NOT_EVALUABLE` as anti-cheating metrics
- `root_displacement_ratio / root_net_displacement_ratio`: sanity-only
- `freeze_lag_score`: sanity-only

## What Should Stay Primary

Keep `GeoLocalDeg` as the primary rollout metric.

Why:

- `GeoLocalDeg` is emitted in `metrics_per_step` from pose-geodesic comparison on predicted pose output vs GT pose output, not from carried root contract fields.
- See `train/validate/run_freerun_cycles.py:9292-9303` and round summaries at `train/validate/run_freerun_cycles.py:9573-9578`.

Recommended interpretation split:

- `GeoLocalDeg` and future pose/joint temporal metrics: model-behavior metrics
- root velocity / root position family under current `motion_quality`: sanity-only contract diagnostics

## Immediate Policy Change

For current WalkF auditing:

1. Do not use `motion_quality` root fields as anti-cheating gates.
2. Treat the existing root-family fields as `sanity_only`.
3. Use `GeoLocalDeg` as the main directional readout until pose/joint temporal quality payloads are added.

## Freeze Case Study: real C0 on Walk_F seed2024

Observed case:

- artifact root: `debug_output/_tmp_walkf_pathc_c0_real_20260504`
- verdict: `AMBIGUOUS`
- key pose-gate values:
  - `joint_angle_velocity_ratio_mean = 0.1694`
  - `joint_angle_velocity_ratio_std = 0.1431`
  - `per_joint_silence_rate = 0.6630`
  - `joint_angle_jitter_score = 0.0000`
  - `GeoLocalDeg_temporal_smoothness = 0.1407`
- rate delta was `24.91%`, but that improvement was a freeze byproduct, not a valid success.

Interpretation:

- The pose-based gate correctly catches this as a freeze failure mode: motion becomes too silent, velocity collapses, and jitter is near zero because the model barely moves.
- Root-family `motion_quality` fields cannot express this failure mode cleanly as a model-behavior gate, because those root channels are still dominated by carry-contract semantics rather than direct pose output behavior.
- This is exactly why the anti-cheating gate must stay pose-based (`joint_angle_velocity_ratio_*`, `joint_angle_jitter_score`, `per_joint_silence_rate`, `GeoLocalDeg_temporal_smoothness`) while root-family fields remain `sanity_only`.
