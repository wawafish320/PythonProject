> TRIAGE: KEEP-AS-PROVENANCE. Historical evidence / anti-relitigation note; preserve to explain why this path was rejected or bounded. Do not use as a live plan unless a new decision record explicitly reopens it.

# Adjusted Acceptance Guard

Date: 2026-06-03

## Scope

This is the required negative-control / full-family guard before any 8-window run. It replays the
saved one-window `gate_w4096` debug output under the adjusted acceptance view:

- event-aware contact / angvel step bands at oracle support-switch frames +/-1.
- heading tolerance `1e-4 rad`.
- original reconstructed full-family checks for regime, rate budget, support honesty,
  support-side correctness, command response, pose continuity, and endpoint bridgeability.

This is debug-only and read-only. It does not change production Trainer/runtime/gate/checkpoint and
does not retrain.

Artifacts:

- script: `tools/run_action_handoff_adjusted_acceptance_guard.py`
- summary: `debug_output/_tmp_action_handoff_adjusted_acceptance_guard_20260603/summary.md`
- json: `debug_output/_tmp_action_handoff_adjusted_acceptance_guard_20260603/adjusted_acceptance_guard_summary.json`
- rows: `debug_output/_tmp_action_handoff_adjusted_acceptance_guard_20260603/rows.csv`
- replay source: `debug_output/_tmp_action_handoff_pose_step_c1c2_sweep_ext_20260603/gate_w4096_pred_raw.npz`

Tensor contract:

- saved `pred_raw`: `[1,6704]`, `float32`, CPU NumPy replay source
- saved `true_raw`: `[1,6704]`, `float32`, CPU NumPy replay control
- reshaped decoder state: `state281 [1,16,281]`, `float32`, CPU NumPy view
- reshaped decoder aux: `bone_angvel [1,16,138]`, `float32`, CPU NumPy view
- sequence fields: `rot6d [16,276]`, `root_pos [16,3]`, `root_vel [16,2]`,
  `bone_angvel [16,138]`, `contact [16,2]`, `cond_dir [16,2]`, `yaw_rate [16]`

## Verdict

The guard fails. Do not run 8-window yet.

| check | result |
|---|---:|
| `gate_w4096` adjusted full-family pass | `false` |
| shortcut negative controls still fail | `true` |
| command demotion negative controls still fail | `true` |
| decision | `adjusted_acceptance_guard_failed_do_not_run_8window` |

This is not a rollback of the c1 verdict. The pose-step c1 result still stands: `pose_step_l2_p95`
passes after the gate-aligned surrogate, and dynamics/endpoint do not expose a representation
conflict. The blocker is narrower: adjusted four-metric pass is not the same as full-family pass.

## Gate W4096 Full Family

Window: `Walk_L_To_L:0-15`. Event switch frames are `[7, 8]`; event boundary frames are
`[6, 7, 8, 9]`.

| family | result | evidence |
|---|---:|---|
| `regime_reached` | pass | `bone_angvel_level_rms_to_target = 0.35147908`, band `0.90655224` |
| `rate_budget` | fail | component angvel and yaw-rate p95 are still over band |
| `support_honesty` | pass | contact over frame `[7]` is event-excused; foot slip `1.24329633 < 2.76610528` |
| `support_side_correctness` | fail | support-side failure count `11` |
| `command_response` | pass | heading p95 `2.257e-05 < 1e-4` tolerance |
| `pose_continuity` | pass | pose p95 `0.01163445 < 0.01174183` |
| `endpoint_bridgeability` | pass | replayed original row endpoint proxy remains true |

The `rate_budget` miss is small but real under the current reconstructed bands:

| metric | value | band | margin |
|---|---:|---:|---:|
| `angvel_step_rms_p95` | `0.59938842` | event-aware `0.59933325` | unexcused over frames `[]` |
| `angvel_component_p95_p95` | `0.78110996` | `0.78103643` | `+7.35e-05` |
| `rootvel_step_l2_p95` | `0.02975973` | `0.04241988` | pass |
| `yaw_rate_step_abs_p95` | `0.13712941` | `0.13711452` | `+1.49e-05` |
| `contact_step_l2_p95` | `0.64826824` | event-aware `0.64826574` | unexcused over frames `[]` |

The `support_side_correctness` miss is larger than a numerical tail. Example failures from the
same replay:

| feature | value | legal band |
|---|---:|---:|
| `single_support_claimed_minus_opposite_mean_mps` | `-1.46850691` | `[0.13790957, 1.48327184]` |
| `single_support_claimed_minus_opposite_p95_mps` | `-0.32594603` | `[0.21920434, 1.97579140]` |
| `single_support_claimed_speed_ratio_p95` | `0.79027531` | `[1.21437608, 3.54838089]` |

## Negative Controls

Shortcut negative controls still fail under the same adjusted acceptance view:

- shortcut / synthetic rows: `63 / 63` remain fail, adjusted pass rate `0.0` for every case.
- covered cases include one-frame switch, matched hard seam, linear pose/contact proxy,
  direct/lambda rows, and artifact bridge proxies.
- command demotion replay rows: `21 / 21` demoted negative rows remain fail, pass count `0`.

This means the adjusted event/heading/pose view did not open a shortcut leak in the available
negative controls.

## Decision

Classification remains:

- contact / angvel hard-step misses: `(b)` event-aware band, already negative-control-safe in this
  guard for the tested shortcuts.
- heading miss: `(a)` heading tolerance / tail-aware surrogate.
- pose miss: `(c1)` loss-balance / surrogate-gate alignment, not `(c2)` representation conflict.

However, `gate_w4096` is not a one-window full-family acceptance pass. The next step is not
8-window and not representation work. First localize the two full-family blockers under the same
debug replay:

1. rate-budget tail: `angvel_component_p95_p95` and `yaw_rate_step_abs_p95`.
2. support-side correctness: 11 support-side band failures despite support honesty passing.

Only after those pass under the same negative-control-safe acceptance view should the fixed global
pose weight be taken to an 8-window debug sweep.
