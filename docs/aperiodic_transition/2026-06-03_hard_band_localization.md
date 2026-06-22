> TRIAGE: KEEP-AS-PROVENANCE. Historical evidence / anti-relitigation note; preserve to explain why this path was rejected or bounded. Do not use as a live plan unless a new decision record explicitly reopens it.

# Hard-Band Per-Frame Localization

Date: 2026-06-03

## Scope

本轮只加 debug-only read-only localization 仪器，定位
`dynamics_consistency` one-window dynamics arm 的 4 个 hard-band miss。没有改
production Trainer/runtime/gate/checkpoint，没有改 loss 权重或模型结构，没有重开
entanglement / lifting / diffusion / yaw 预测。

Artifact:

- script: `tools/run_action_handoff_dynamics_consistency_train_fit_ladder.py --localize-only`
- summary: `debug_output/_tmp_action_handoff_dynamics_consistency_localization_20260603/summary.md`
- json: `debug_output/_tmp_action_handoff_dynamics_consistency_localization_20260603/localization_summary.json`
- per-frame: `debug_output/_tmp_action_handoff_dynamics_consistency_localization_20260603/per_frame.csv`
- per-channel: `debug_output/_tmp_action_handoff_dynamics_consistency_localization_20260603/per_channel.csv`

Tensor contract:

- decoder input: `x [1,4957]`, `float32`, `cpu`
- decoder output state: `state281 [1,16,281]`, `float32`, `cpu`
- decoder aux: `bone_angvel [1,16,138]`, `float32`, `cpu`
- dynamics witness input: `X_norm [1,15,419]`, `float32`, `cpu`
- dynamics witness residual: `Y residual [1,15,278]`, `float32`, `cpu`
- `per_frame.csv`: `16` rows, frame-aligned hard metric values / band / margin / event mask
- `per_channel.csv`: `6240` rows = `15 * (276 pose + 138 angvel + 2 contact)` channel deltas

Guard remains valid: reconstructed GT acceptance `1.0000`, decoder-path-from-GT-raw
acceptance `1.0000`, max abs seq delta `0.0`.

## Result

Window: `Walk_L_To_L:0-15`. Oracle event/support switch frames are `[7, 8]`; boundary
window with switch frame +/-1 is `[6, 7, 8, 9]`.

| metric | p95 / band | p95 margin | over frames | boundary overlap | dynamics-high overlap | class | next |
|---|---:|---:|---|---:|---:|---|---|
| `pose_step_l2_p95` | `0.01759324 / 0.01174183` | `+0.00585141` | `[1,2,3,4,5,6,8,12]` | `0.2500` | `0.1250` | `c_pending` | 先判别 loss-balance vs 真表征冲突 |
| `contact_step_l2_p95` | `0.64827996 / 0.64826574` | `+1.42e-05` | `[7]` | `1.0000` | `0.0000` | `(b)` | 改 event-aware band |
| `angvel_step_rms_p95` | `0.59937647 / 0.59933325` | `+4.32e-05` | `[8]` | `1.0000` | `0.0000` | `(b)` | 改 event-aware band |
| `heading_error_p95_rad` | `4.5539e-05 / 3.9418e-08` | `+4.55e-05` | all `0..15` | `0.2500` | `0.0625` | `(a)` | 换 tail-aware surrogate / hard-gate aligned command p95 |

`dynamics_zero_resid_frame_rms_scaled` 的 p95-high frame 是 `[2]`。它不与
`angvel_step_rms_p95` miss frame `[8]` 共定位；`angvel` miss 不是 dynamics residual
没有贴够 GT 的同源现象。`pose_step` 与 dynamics-high 只有 `1/8` overlap，说明 pose
miss 也不是主要由 zero-residual witness 尾部驱动。

## Surrogate vs Gate

| metric | training surrogate | final loss | per-frame surrogate mean | gate p95 margin |
|---|---|---:|---:|---:|
| `pose_step_l2_p95` | `pose_continuity_loss` | `8.52595e-05` | `8.60445e-05` | `+5.85141e-03` |
| `contact_step_l2_p95` | `contact_schedule` | `7.78173e-10` | `6.61426e-10` | `+1.42162e-05` |
| `angvel_step_rms_p95` | `bone_angvel_rate_loss` | `1.56824e-05` | `1.56503e-05` | `+4.32165e-05` |
| `heading_error_p95_rad` | `command_compatibility` | `1.10621e-09` | `8.27117e-10` | `+4.54992e-05` |

For `contact` / `angvel` / `heading`, the mean surrogate is low while the p95 hard gate still has
tail miss. For `pose_step`, the margin is two orders larger and spread across 8 transitions, so it
does not fit the 1e-5 tail-only bucket. This only rules out `(a)` / `(b)` for the localized
one-window miss; it does not by itself prove a representation ceiling.

## Full-188 GT Control

Full reconstructed GT sample-level control is not the hard gate itself, but it tells whether
single-frame spikes exist in legal motion:

- `pose_step`: `39 / 2820` GT transition samples exceed their clip p95 band, event-boundary ratio
  `0.0000`, max sample margin `0.00409168`. Legal non-event pose spikes exist, but the localized
  prediction has `8 / 15` over-band transitions and p95 margin `0.00585141`, so the one-window
  pose failure remains a real blocker.
- `contact_step`: `17 / 2820` GT transition samples exceed band, event-boundary ratio `0.9412`.
  This supports `(b)` event-aware band rather than representation change.
- `angvel_step`: `38 / 2820` GT transition samples exceed band, event-boundary ratio `0.5789`.
  The localized one-window miss is exactly frame `8`, inside the switch boundary, so classify this
  run as `(b)`.
- `heading_error`: `32 / 3008` GT samples exceed band with max margin only `1.83e-08`; the predicted
  miss is all frames at `1e-5` scale, so classify as `(a)` surrogate/gate mismatch, not geometry.

## Decision

The four misses do not have one shared cause:

- `(a) surrogate tail`: `heading_error_p95_rad`.
- `(b) event-aware band`: `contact_step_l2_p95`, `angvel_step_rms_p95`.
- `c_pending`: `pose_step_l2_p95`, still split between `(c1)` loss-balance / surrogate-gate mismatch
  and `(c2)` true representation conflict.

Therefore the next step is not lifting/entanglement/diffusion. First apply the localization-authorized
debug fixes for `(a)` and `(b)`: event-aware bands for contact/angvel around support switches, and a
small heading tolerance or tail-aware command surrogate. Then run a minimal pose-step sweep:
if pose can be brought under band without increasing dynamics/endpoint/support failures, classify
as `(c1)` loss-balance; only if pose can drop below band only by breaking dynamics or endpoint does
it earn `(c2)` representation conflict.

Follow-up resolved this fork in
`docs/aperiodic_transition/2026-06-03_pose_step_c1c2_sweep.md`: gate-aligned pose surrogate
weight `4096` brings `pose_step_l2_p95` to `0.01163445` below band `0.01174183`, with adjusted
four-metric pass `1`, dynamics anchor improved to `0.00726000`, and endpoint at `3.03e-5`.
Therefore this one-window pose miss is `(c1)` loss-balance / surrogate-gate alignment, not earned
evidence for `(c2)` representation conflict.

Follow-up guard in `docs/aperiodic_transition/2026-06-03_adjusted_acceptance_guard.md` confirms the
adjusted acceptance view does not make shortcut / command-demotion negative controls pass, but it
also shows `gate_w4096` is not a full-family pass: `rate_budget` and `support_side_correctness`
still fail. Therefore the next step is to localize those full-family blockers, not to run 8-window
yet and not to reopen representation changes.
