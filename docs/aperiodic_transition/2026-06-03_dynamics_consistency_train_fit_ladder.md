> TRIAGE: KEEP-AS-PROVENANCE. Historical evidence / anti-relitigation note; preserve to explain why this path was rejected or bounded. Do not use as a live plan unless a new decision record explicitly reopens it.

# GT-Residual-Anchored Dynamics-Consistency Ladder

Date: 2026-06-03

## Scope

本轮修正上一版 dynamics-consistency ladder 的两个 formulation 问题：

- `dynamics_consistency` 不再把 residual 往 0 压，而是锚到 GT residual：
  `MSE(r_pred, r_gt)`，其中 `r = (Y_next - f(X_t, cmd)) / eval_y_scale`。
- rate/support/pose continuity 的可微 surrogate 提回 dynamics arm，旧的 zero-residual
  objective 只作为 witness 记录，不回传梯度。

Artifacts:

- script: `tools/run_action_handoff_dynamics_consistency_train_fit_ladder.py`
- summary: `debug_output/_tmp_action_handoff_dynamics_consistency_gt_residual_ladder_20260603/summary.md`
- summary json: `debug_output/_tmp_action_handoff_dynamics_consistency_gt_residual_ladder_20260603/summary.json`
- rows: `debug_output/_tmp_action_handoff_dynamics_consistency_gt_residual_ladder_20260603/rows.csv`
- step log: `debug_output/_tmp_action_handoff_dynamics_consistency_gt_residual_ladder_20260603/step_log.csv`

Tensor contract:

- decoder input: `[B,input_dim]`, `float32`, `cpu`
- decoder state output: `state281 [B,16,281]`, `float32`, `cpu`
- decoder aux output: `bone_angvel [B,16,138]`, `float32`, `cpu`
- base operator input: `X_norm [B,15,419]`, `float32`, `cpu`
- base operator eval output: `Y_raw [B,15,278]`, `float32`, `cpu`
- dynamics residual target: `r_gt [B,15,278]`, `float32`, `cpu`
- channel gradient groups: pose `state281[...,0:276]`, rootvel `state281[...,276:278]`,
  contact `state281[...,279:281]`, bone_angvel aux `[...,138]`

## Guard And Base f

Path identity still passes before training:

- guard windows: `188`
- max abs reconstructed seq delta: `0.0`
- reconstructed GT acceptance: `1.0000`
- decoder-path-from-GT-raw acceptance: `1.0000`
- physical `X_raw -> X_norm` max abs error: `2.4318695e-05`

Checkpoint/load confound check:

- exact raw checkpoint/model name+shape overlap: `133 / 143` tensors
- matched model numel ratio: `0.91772294`
- shape mismatch count: `0`
- model keys missing raw ckpt names: the 10 `frozen_encoder.*` / `frozen_period_head.*` tensors
- raw ckpt keys not in model: 6 `shared_encoder.8.*` tensors

These are the same FreeRun schema/runtime-load warnings seen in the previous run. They do not
prove a strict full load contract, but the main `ret["out"]` operator is not behaving like random
weights on GT:

| set | n | GT self anchored loss | GT zero-residual loss | scaled RMS | frame RMS p95 | pose raw RMS | rootvel raw RMS |
|---|---:|---:|---:|---:|---:|---:|---:|
| one_window | 1 | 0.0 | 0.01098269 | 0.10479833 | 0.13747472 | 0.00870097 | 0.01483217 |
| eight_window | 8 | 0.0 | 0.00976357 | 0.09881076 | 0.13811165 | 0.00838107 | 0.01702234 |
| full_188 | 188 | 0.0 | 0.00742318 | 0.08615790 | 0.12618683 | 0.00788686 | 0.01332482 |

The previous smoking gun was real: zero-residual dynamics could be gamed below the one-window
GT floor. The repaired arm no longer does that; its zero-residual witness ends at `0.02268642`,
above the one-window GT zero-residual floor `0.01098269`.

## Loss Setup

Dynamics arm active weights:

- `dynamics_consistency`: `1.0`, now `MSE(r_pred, r_gt)`
- `command_compatibility`: `24.0`
- `endpoint_reaching`: `4.0`
- `regime_reaching`: `1.0`
- `contact_schedule`: `4.0`
- `pose_continuity_loss`: `4.0`
- `rootvel_rate_loss`: `4.0`
- `yaw_rate_loss`: `4.0`
- `bone_angvel_rate_loss`: `1.0`
- `fk_foot_slip_loss`: `0.35`

Symptom arm keeps `dynamics_consistency=0.0`; anchored dynamics and zero-residual are witness-only
there.

## One-Window Results

The ladder stopped at one window by failure-stop rule.

| arm | accept | failed families | dyn anchor | zero-resid witness | state MSE | contact MSE | aux MSE |
|---|---:|---|---:|---:|---:|---:|---:|
| dynamics_consistency | 0.0000 | rate_budget, support_honesty, support_side_correctness, command_response, pose_continuity | 0.01161698 | 0.02268642 | 0.00015706 | 6.6143e-10 | 0.00016331 |
| symptom_ablation | 0.0000 | rate_budget, support_side_correctness, command_response, pose_continuity | 0.03431828 | 0.04455883 | 0.00008852 | 5.9997e-11 | 7.5351e-11 |

Dynamics arm terms:

- `dynamics_consistency`: `0.71668708 -> 0.01161698`
- `dynamics_zero_residual_witness`: `0.72697896 -> 0.02268642`
- `command_compatibility`: `0.00258665 -> 1.1062e-09`
- `endpoint_reaching`: `0.00706867 -> 7.1288e-06`
- `regime_reaching`: `0.00153025 -> 9.5393e-08`
- `rootvel_rate_loss`: `0.00121305 -> 2.7491e-07`
- `yaw_rate_loss`: `0.00404183 -> 2.0178e-09`
- `bone_angvel_rate_loss`: `0.00276317 -> 1.5682e-05`
- `fk_foot_slip_loss`: `70.03194427 -> 1.2747e-06`

Hard-band misses in the dynamics arm:

| family | metric | value | threshold | delta |
|---|---|---:|---:|---:|
| rate_budget | `angvel_step_rms_p95` | 0.59937647 | 0.59933325 | +4.32e-05 |
| support_honesty | `contact_step_l2_p95` | 0.64827996 | 0.64826574 | +1.42e-05 |
| pose_continuity | `pose_step_l2_p95` | 0.01759324 | 0.01174183 | +0.00585141 |
| command_response | `heading_error_p95_rad` | 4.5539e-05 | 3.9418e-08 | +4.55e-05 |

Foot slip itself is within band in the dynamics arm: `foot_slip_p95_mps=1.24353966` vs
threshold `2.76610528`. The support failure comes from the contact-step side of
`support_honesty`, not FK slip.

Final channel gradient instrumentation:

| arm | pose norm | contact norm | rootvel norm | bone_angvel norm | min pose-rootvel cos | min pose-contact cos | min rootvel-bone cos |
|---|---:|---:|---:|---:|---:|---:|---:|
| dynamics_consistency | 0.14283905 | 0.00017273 | 0.00099800 | 0.00079503 | -0.12654210 | -0.11695782 | -0.19157706 |
| symptom_ablation | 0.00976362 | 0.00002212 | 0.00040700 | 0.00001044 | -0.12554862 | -0.22564131 | -0.13093049 |

## Answers

1. Mechanism vs symptom:

The previous zero-residual mechanism objective had a degenerate solution. Anchoring to GT residual
blocks the low-pass basin exploit: the zero-residual witness no longer goes below the GT floor.
However, even with anchored dynamics and rate/support/pose surrogate losses active, acceptance is
still `0.0000`. Therefore the six-family contract cannot collapse to dynamics-consistency alone,
and the current differentiable surrogates are not exact enough for the hard acceptance bands.

2. Entanglement:

Entanglement is not proven binding in this run. The cosines are weak/near-orthogonal and similar
across the dynamics and symptom arms. They remain useful instrumentation, but this run does not
justify pose/dynamics lifting or a representation change.

3. Base-operator manifold:

No off-manifold conclusion is justified. Endpoint and regime terms go near zero, anchored dynamics
is lowered substantially, and the remaining failures are hard-band/surrogate mismatches plus
support-side/command/pose constraints. This is not clean evidence that the target turning regime
is unreachable under the Walk_F base operator.

## Decision

Failure signature: `train_fit_fail_no_binding_signature_yet`.

The old `train_fit_fail_entanglement_signature_negative_channel_cosine` conclusion is withdrawn.
The next debug-only step should improve the differentiable losses to target the exact hard-band
metrics that still fail: margin-aware pose p95, heading p95, contact-step p95, and angvel-step p95.
Do not move to lifting/axis split, root/foot anchoring, diffusion, or production changes from this
evidence.
