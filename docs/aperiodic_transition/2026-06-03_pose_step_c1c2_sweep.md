> TRIAGE: KEEP-AS-PROVENANCE. Historical evidence / anti-relitigation note; preserve to explain why this path was rejected or bounded. Do not use as a live plan unless a new decision record explicitly reopens it.

# Pose-Step C1/C2 Sweep

Date: 2026-06-03

## Scope

Debug-only one-window discriminator for the remaining `pose_step_l2_p95` miss. This is not a
production Trainer/runtime/gate/checkpoint change. It does not change model structure or the
production acceptance path.

Artifacts:

- base sweep: `debug_output/_tmp_action_handoff_pose_step_c1c2_sweep_20260603/summary.md`
- base rows: `debug_output/_tmp_action_handoff_pose_step_c1c2_sweep_20260603/rows.csv`
- extended sweep: `debug_output/_tmp_action_handoff_pose_step_c1c2_sweep_ext_20260603/summary.md`
- extended rows: `debug_output/_tmp_action_handoff_pose_step_c1c2_sweep_ext_20260603/rows.csv`
- saved debug outputs: each row has `*_pred_raw.npz` and `*_decoder_state.pt`

Tensor contract:

- decoder input: `x [1,4957]`, `float32`, `cpu`
- decoder output state: `state281 [1,16,281]`, `float32`, `cpu`
- decoder aux: `bone_angvel [1,16,138]`, `float32`, `cpu`
- pose gate view: adjacent `rot6d [1,15,276]`, metric computed as L2/sqrt(276)
- debug dynamics witness: `X_norm [1,15,419]`, `float32`, `cpu`; residual `Y [1,15,278]`,
  `float32`, `cpu`

Guard remained valid: reconstructed GT acceptance `1.0000`, decoder-path-from-GT-raw acceptance
`1.0000`, max abs seq delta `0.0`.

## Surrogate Alignment

The existing training surrogate and hard gate are not identical:

- `pose_continuity_loss`: mean MSE between predicted and GT pose-step delta over
  `rot6d [1,15,276]`.
- `pose_step_l2_p95`: p95 of raw adjacent-frame `rot6d` L2/sqrt(276).

The sweep therefore includes a debug-only gate-aligned term:

`pose_gate_margin_loss = mean/top-k relu(pose_step_l2 - band)^2`

where the one-window band is `0.011741833388805387`.

## Sweep Results

The `(a)` / `(b)` fixes are applied only in the isolated decision view:

- contact/angvel over-band frames inside support-switch +/-1 are treated as event-aware pass.
- heading uses tolerance `1e-4 rad`, which covers the localized `4.55e-5` tail.

Base sweep:

| mode | weight | pose p95 | margin | over frames | dyn anchor | endpoint | adjusted pass |
|---|---:|---:|---:|---|---:|---:|---:|
| mean | 4 | 0.01759324 | +0.00585141 | `1,2,3,4,5,6,8,12` | 0.01161698 | 7.13e-06 | 0 |
| mean | 256 | 0.01313093 | +0.00138910 | `2,12` | 0.00694792 | 1.05e-05 | 0 |
| gate | 256 | 0.01216735 | +0.00042552 | `2,5,6,12` | 0.00921357 | 7.63e-06 | 0 |

Extended gate-only sweep:

| mode | weight | pose p95 | margin | over frames | dyn anchor | endpoint | adjusted pass |
|---|---:|---:|---:|---|---:|---:|---:|
| gate | 1024 | 0.01183161 | +0.00008978 | `1,2,4,6,12` | 0.00840900 | 1.37e-05 | 0 |
| gate | 4096 | 0.01163445 | -0.00010738 | `1` | 0.00726000 | 3.03e-05 | 1 |

The `gate_w4096` row saved:

- pred raw: `debug_output/_tmp_action_handoff_pose_step_c1c2_sweep_ext_20260603/gate_w4096_pred_raw.npz`
- tiny decoder state: `debug_output/_tmp_action_handoff_pose_step_c1c2_sweep_ext_20260603/gate_w4096_decoder_state.pt`

## Verdict

Classify `pose_step_l2_p95` as `(c1) loss-balance / surrogate-gate alignment`, not `(c2)`
representation conflict.

Reason:

- A gate-aligned pose surrogate can bring p95 below band: `0.01163445 < 0.01174183`.
- The adjusted four hard metrics pass after the already-localized `(a)` / `(b)` fixes.
- Dynamics anchor improves from the baseline `0.01161698` to `0.00726000`; this is the opposite of
  "pose only passes by breaking dynamics".
- Endpoint loss rises from `7.13e-6` to `3.03e-5`, but this run does not show an endpoint/dynamics
  tradeoff severe enough to justify a representation-conflict claim.

Follow-up guard: `docs/aperiodic_transition/2026-06-03_adjusted_acceptance_guard.md` replays the
saved `gate_w4096` row under the same adjusted view. It confirms the shortcut / command-demotion
negative controls still fail, but the row is not a full-family pass because `rate_budget` and
`support_side_correctness` still fail. Therefore c1 remains established, but 8-window is blocked
until those full-family blockers are localized and cleared under the same negative-control-safe
acceptance view.
