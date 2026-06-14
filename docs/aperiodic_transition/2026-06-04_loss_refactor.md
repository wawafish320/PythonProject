> TRIAGE: KEEP-AS-PROVENANCE. Historical evidence / anti-relitigation note; preserve to explain why this path was rejected or bounded. Do not use as a live plan unless a new decision record explicitly reopens it.

# Loss Refactor: Root-Support Axis

Date: 2026-06-04

## Scope

Debug-only one-window refactor in `tools/run_action_handoff_dynamics_consistency_train_fit_ladder.py`.
No production Trainer/runtime/gate/checkpoint path was changed.

Artifacts:

- canonical summary: `debug_output/_tmp_action_handoff_causal_loss_refactor_20260604/summary.md`
- canonical json: `debug_output/_tmp_action_handoff_causal_loss_refactor_20260604/loss_refactor_summary.json`
- canonical rows: `debug_output/_tmp_action_handoff_causal_loss_refactor_20260604/rows.csv`
- canonical step log: `debug_output/_tmp_action_handoff_causal_loss_refactor_20260604/step_log.csv`
- saved `pred_raw`: `debug_output/_tmp_action_handoff_causal_loss_refactor_20260604/causal3_one_window_pred_raw.npz`
- saved decoder state: `debug_output/_tmp_action_handoff_causal_loss_refactor_20260604/causal3_one_window_decoder_state.pt`
- adjusted guard: `debug_output/_tmp_action_handoff_causal_loss_refactor_20260604/adjusted_guard/summary.md`

Tensor contract:

- decoder input: `x [1,4957]`, `float32`, CPU tensor.
- decoder output state: `state281 [1,16,281]`, `float32`, CPU tensor.
- decoder aux: `bone_angvel [1,16,138]`, `float32`, CPU tensor.
- saved `pred_raw`: `[1,6704]`, `float32`, CPU NumPy.
- sequence replay: `rot6d [16,276]`, `root_vel [16,2]`, `root_pos [16,3]`,
  `bone_angvel [16,138]`, `contact [16,2]`, `yaw_rate [16]`.

## Metric Definition Check

Before re-cutting the loss, the hard metrics were re-read from the adjusted guard path:

- `rate_budget`: `angvel_step_rms_p95`, `angvel_component_p95_p95`,
  `rootvel_step_l2_p95`, `yaw_rate_step_abs_p95`.
- `support_honesty`: event-aware `contact_step_l2_p95` plus FK contacted-foot speed.
- `support_side_correctness`: support-side feature bands, including FK foot-speed asymmetry,
  foot-relative-to-root features, `root_speed_mean`, `root_lateral_mean`, and
  `support_lateral_product`.
- `command_response`: `heading_error_p95_rad`, evaluated with the authorized `1e-5 rad`
  tolerance in the adjusted guard.

This makes the old `L_dynamics/L_contact/L_goal` split invalid for root: `rootvel_step_l2_p95`
was inside `L_dynamics`, while root-relative support-side failures were inside `L_contact`.

## Current 3-Term Cut

| witness / old symptom | causal item | debug-only surrogate |
|---|---|---|
| `dynamics_consistency` rot6d residual | `L_articulation` | GT-residual anchor over `rot6d [B,15,276]` |
| pose continuity | `L_articulation` | band-normalized rot6d step margin |
| joint rate witnesses | `L_articulation` | band-normalized `bone_angvel` rate margins; RMS uses switch-frame +/-1 event mask |
| `dynamics_consistency` root residual | `L_root_support` | GT-residual anchor over `root_vel [B,15,2]` |
| root path | `L_root_support` | rootvel step hard-margin, GT rootvel-step anchor, GT rootvel-path anchor |
| support-side correctness | `L_root_support` | FK support-side feature band margins, including root speed/lateral features |
| contact / foot honesty | `L_root_support` | event-aware contact-step, oracle contact anchor, FK contacted-foot speed margin |
| endpoint / regime / command | `L_goal` | endpoint, final target-regime `bone_angvel` level, heading p95, yaw-rate-step margin |

The canonical one-window run used:

| item | weight |
|---|---:|
| `L_articulation` | `8.0` |
| `L_root_support` | `8.0` |
| `L_goal` | `1.0` |

Other run parameters: `lr=1e-3`, `epochs=800`, `heading_tolerance_rad=1e-5`,
`loss_refactor_support_feature_topk=0`.

## Guard Identity

The reconstructed guard path remained identical:

| check | value |
|---|---:|
| reconstructed GT acceptance | `1.0000` |
| decoder-path-from-GT acceptance | `1.0000` |
| `max_abs_seq_delta` | `0.00000000` |

## Canonical One-Window Result

Window: `Walk_L_To_L:0-15`.

| family / metric | value | band / tolerance | result |
|---|---:|---:|---:|
| `regime_reached` | `0.36050415` | `0.90655224` | pass |
| `pose_continuity.pose_step_l2_p95` | `0.01171762` | `0.01174183` | pass |
| `support_honesty.contact_step_l2_p95` | `0.64826576` | event-aware `0.64826574` | pass |
| `support_honesty.foot_slip_p95_mps` | `2.21873012` | `2.76610528` | pass |
| `command_response.heading_error_p95_rad` | `8.50e-08` | `1e-5` | pass |
| `rate_budget.angvel_step_rms_p95` | `0.81653877` | event-aware `0.59933325` | fail; unexcused frame `14` |
| `rate_budget.angvel_component_p95_p95` | `0.78129946` | `0.78103643` | fail, `+2.63e-04` |
| `rate_budget.rootvel_step_l2_p95` | `0.08872701` | `0.04241988` | fail |
| `support_side_correctness` | `4` failures | `0` | fail |

The remaining support-side failures are still root/support kinematics:

| feature | value | legal band |
|---|---:|---:|
| `right_rel_y_mean` | out of band | see adjusted guard artifact |
| `left_rel_y_mean` | out of band | see adjusted guard artifact |
| `root_speed_mean` | out of band | see adjusted guard artifact |
| `support_lateral_product` | out of band | see adjusted guard artifact |

## Convergence Evidence

The old split cleanly exposed root as the missing axis:

| run | factor cut | rootvel p95 | support-side failures | pose | heading | verdict |
|---|---|---:|---:|---:|---:|---|
| old `dynamics/contact/goal` | root split across two items | `0.15646994` | `7` | pass | pass | fail |
| root6 stable run | `articulation/root_support/goal` + root anchors | `0.07172529` | `4` | slight fail | pass | fail |
| canonical root8 | same cut, stronger articulation | `0.08872701` | `4` | pass | pass | fail |

The run is not an 8-window authorization. It is a narrower result:

- root support recut reduces the root failure substantially, so the previous collapse failure was
  a mis-factorization, not a representation ceiling.
- root-rate and support-side failures move together, confirming they are one physical axis.
- there is also a current `L_articulation` high-frequency tail: `angvel_step_rms_p95` fails at
  unexcused frame `14`, and `angvel_component_p95_p95` is `2.63e-04` above band.

## Minimax Feasibility Follow-Up

The debug objective was extended with a gate-aligned minimax mode:

- `loss_refactor_objective=minimax`
- soft max temperature `0.05`
- anchor tie-breaker weight `0.05`
- gate vector: band/interval violations only.
- anchor tie-breaker: GT residual, rootvel path/rate anchor, contact anchor, endpoint anchor.

Pure minimax from random initialization is not decision-eligible: it falls into a worse basin
than the canonical weighted run.

| run | warmup | epochs | hard max gate surrogate | rootvel p95 | angvel RMS p95 | angvel component p95 | pose p95 | support-side failures | adjusted failed family |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| weighted canonical | n/a | 800 | n/a | `0.08872701` | `0.81653877` | `0.78129946` | `0.01171762` | `4` | `rate_budget,support_side_correctness` |
| pure minimax | 0 | 800 | `6.03799105` | `0.10388507` | `1.24110788` | `2.26864855` | `0.04025987` | `12` | `rate_budget,support_honesty,support_side_correctness,command_response,pose_continuity` |
| warm-start minimax | 400 | 800 | `1.37367880` | `0.09766193` | `0.74581712` | `0.81688924` | `0.01173052` | `4` | `rate_budget,support_side_correctness` |
| warm-start minimax | 400 | 2000 | `0.64862227` | `0.07587766` | `0.75066227` | `0.83270992` | `0.01160417` | `4` | `rate_budget,support_side_correctness` |
| warm-start minimax | 400 | 4800 | `0.22338350` | `0.06226914` | `0.77274601` | `0.78107752` | `0.01167020` | `3` | `rate_budget,support_side_correctness` |

The e4800 run is the cleanest current feasibility artifact:

- summary: `debug_output/_tmp_action_handoff_causal_loss_refactor_minimax_warm400_e4800_20260604/summary.md`
- rows: `debug_output/_tmp_action_handoff_causal_loss_refactor_minimax_warm400_e4800_20260604/rows.csv`
- step log: `debug_output/_tmp_action_handoff_causal_loss_refactor_minimax_warm400_e4800_20260604/step_log.csv`
- saved `pred_raw`: `debug_output/_tmp_action_handoff_causal_loss_refactor_minimax_warm400_e4800_20260604/causal3_minimax_one_window_pred_raw.npz`
- saved decoder state: `debug_output/_tmp_action_handoff_causal_loss_refactor_minimax_warm400_e4800_20260604/causal3_minimax_one_window_decoder_state.pt`

At e4800, all minimax surrogate terms except rootvel are effectively slack:

| surrogate | final value |
|---|---:|
| `root_support_rootvel_rate_margin_loss` | `0.223383501` |
| `root_support_contact_support_side_root_speed_mean_margin_loss` | `0.028205762` |
| `root_support_contact_support_side_support_lateral_product_margin_loss` | `0.000176064` |
| `articulation_angvel_component_margin_loss` | `1.07e-08` |
| `articulation_pose_step_margin_loss` | `6.34e-09` |
| `articulation_angvel_rms_margin_loss` | `1.84e-09` |

The last sampled e4800 trend is still descending and concentrated in rootvel:

| epoch | hard max | rootvel surrogate | support-side aggregate | rootvel channel grad norm |
|---:|---:|---:|---:|---:|
| 4000 | `0.27464762` | `0.27464762` | `0.00142208` | `24.9863` |
| 4200 | `0.26118681` | `0.26118681` | `0.00140428` | `23.7339` |
| 4400 | `0.24781060` | `0.24781060` | `0.00138685` | `22.0736` |
| 4600 | `0.23491828` | `0.23491828` | `0.00137030` | `18.6788` |
| 4799 | `0.22338350` | `0.22338350` | `0.00135188` | `17.4102` |

An e6000 run is not final-decision eligible because it leaves the good basin after about
epoch 5000:

| epoch | hard max | rootvel surrogate | support-side aggregate |
|---:|---:|---:|---:|
| 4800 | `0.22332712` | `0.22332712` | `0.00135179` |
| 5000 | `2778068.75` | `10.50111771` | `111373.61718750` |
| 5999 | `6.55938196` | `6.55938196` | `0.60554916` |

Historical read at this stage: the root-support recut plus minimax was not c1 yet, and it
was still too early to call c2. The only robust binding surrogate was rootvel-rate, but it
was still descending before the optimizer instability. The lower-LR GT-basin probe below
supersedes this intermediate read.

The clean pass set at e4800 is contact honesty, pose continuity, heading, regime, guard
identity, and negative-control rejection.

## GT Warm-Start Feasibility Check

Because reconstructed GT and decoder-path-from-GT both pass at `1.0000`, the remaining
rootvel question was tested from a supervised GT basin instead of another random basin.

Implementation:

- warmup mode: `supervised_flat`, optimizing `MSE(pred_std, ytr_std)` over
  `pred_raw [1,6704] float32 CPU` through the decoder.
- warmup epochs: `2000`
- minimax tail lr: `1e-5`
- minimax tail epochs: `1000`

Artifacts:

- summary: `debug_output/_tmp_action_handoff_causal_loss_refactor_minimax_gtwarm_flat2000_tail1e5_e3000_20260604/summary.md`
- rows: `debug_output/_tmp_action_handoff_causal_loss_refactor_minimax_gtwarm_flat2000_tail1e5_e3000_20260604/rows.csv`
- step log: `debug_output/_tmp_action_handoff_causal_loss_refactor_minimax_gtwarm_flat2000_tail1e5_e3000_20260604/step_log.csv`
- saved `pred_raw`: `debug_output/_tmp_action_handoff_causal_loss_refactor_minimax_gtwarm_flat2000_tail1e5_e3000_20260604/causal3_minimax_one_window_pred_raw.npz`
- saved decoder state: `debug_output/_tmp_action_handoff_causal_loss_refactor_minimax_gtwarm_flat2000_tail1e5_e3000_20260604/causal3_minimax_one_window_decoder_state.pt`

The warmup reached GT precision before minimax:

| epoch | objective | `flat_standardized` | hard max gate surrogate | rootvel surrogate |
|---:|---:|---:|---:|---:|
| 1000 | `9.19e-18` | `9.19e-18` | `0.01379557` | `0.0` |
| 1900 | `8.98e-18` | `8.98e-18` | `0.01379557` | `0.0` |
| 2000 | `0.00073969` | `8.59e-18` | `0.01379557` | `0.0` |

With the low-LR minimax tail, rootvel stays inside the hard band:

| metric | value | band | pass |
|---|---:|---:|---:|
| `rootvel_step_l2_p95` | `0.03266504` | `0.04241988` | yes |
| `angvel_step_rms_p95` | `0.59922332` | `0.59933325` | yes |
| `angvel_component_p95_p95` | `0.75860095` | `0.78103643` | yes |
| `yaw_rate_step_abs_p95` | `0.13688877` | `0.13711452` | yes |
| `pose_step_l2_p95` | `0.01116749` | `0.01174183` | yes |
| `foot_slip_p95_mps` | `1.32591879` | `2.76610528` | yes |

Final minimax terms:

| surrogate | final value |
|---|---:|
| `root_support_rootvel_rate_margin_loss` | `0.0` |
| `root_support_side_margin_loss` | `4.14e-7` |
| `articulation_angvel_component_margin_loss` | `0.00280295` |
| `loss_refactor_hard_max_gate_violation` | `0.00280295` |

The only remaining hard failure is support-side:

| feature | value | band min | band max | margin |
|---|---:|---:|---:|---:|
| `support_lateral_product` | `-0.11965578` | `-0.11868681` | `0.33190343` | `-0.00096897` |

Interpretation:

- rootvel is not a representation ceiling and not a feasibility blocker.
- the cold/warm weighted rootvel holdout was an optimizer/tail-LR issue.
- from GT basin, minimax preserves rootvel inside band and leaves only a tiny support-side
  interval failure.
- the current one-window is still not an 8-window authorization, because support-side hard
  correctness is not fully cleared.

## Hard-Gate Support Closure

The final support-side holdout was `support_lateral_product`. The support surrogate was
made directly hard-gate aligned for that feature only:

- linear feature key: `support_lateral_product`
- excluded duplicated support feature: `heading_error_p95_rad`
- hard-gate feature key: `support_lateral_product`
- hard-gate tolerance: `1e-6 + 1e-5 * max(1, |lo|, |hi|)`
- safety margin: `1e-6`, to avoid stopping on the float32/float64 guard boundary
- support scale floor: `0.01`
- minimax temperature: `0.005`
- warmup: `supervised_flat`, `2000` epochs
- minimax tail: `5000` epochs at `1e-5`

Implementation is still debug-only in
`tools/run_action_handoff_dynamics_consistency_train_fit_ladder.py`; production
Trainer/runtime/gate/checkpoint were not touched.

Artifacts:

- summary: `debug_output/_tmp_action_handoff_causal_loss_refactor_minimax_gtwarm_flat2000_tail1e5_lat_hardtol0p01_safe1e6_tau0p005_e7000_20260604/summary.md`
- adjusted guard: `debug_output/_tmp_action_handoff_causal_loss_refactor_minimax_gtwarm_flat2000_tail1e5_lat_hardtol0p01_safe1e6_tau0p005_e7000_20260604/adjusted_guard/summary.md`
- rows: `debug_output/_tmp_action_handoff_causal_loss_refactor_minimax_gtwarm_flat2000_tail1e5_lat_hardtol0p01_safe1e6_tau0p005_e7000_20260604/rows.csv`
- step log: `debug_output/_tmp_action_handoff_causal_loss_refactor_minimax_gtwarm_flat2000_tail1e5_lat_hardtol0p01_safe1e6_tau0p005_e7000_20260604/step_log.csv`
- saved `pred_raw`: `debug_output/_tmp_action_handoff_causal_loss_refactor_minimax_gtwarm_flat2000_tail1e5_lat_hardtol0p01_safe1e6_tau0p005_e7000_20260604/causal3_minimax_one_window_pred_raw.npz`
- saved decoder state: `debug_output/_tmp_action_handoff_causal_loss_refactor_minimax_gtwarm_flat2000_tail1e5_lat_hardtol0p01_safe1e6_tau0p005_e7000_20260604/causal3_minimax_one_window_decoder_state.pt`

Tensor contract:

- decoder input: `x [1,4957] float32 CPU`
- decoder output state: `state281 [1,16,281] float32 CPU`
- decoder aux: `bone_angvel [1,16,138] float32 CPU`
- saved `pred_raw`: `[1,6704] float32 CPU NumPy`

Final one-window adjusted guard:

| family / metric | value | band / tolerance | result |
|---|---:|---:|---:|
| `regime_reached` | `0.35147774` | `0.90655224` | pass |
| `angvel_step_rms_p95` | `0.59888975` | event-aware `0.59933325` | pass |
| `angvel_component_p95_p95` | `0.75034146` | `0.78103643` | pass |
| `rootvel_step_l2_p95` | `0.03490959` | `0.04241988` | pass |
| `yaw_rate_step_abs_p95` | `0.13711222` | `0.13711452` | pass |
| `contact_step_l2_p95` | `0.64826574` | event-aware `0.64826574` | pass |
| `foot_slip_p95_mps` | `1.30187497` | `2.76610528` | pass |
| `heading_error_p95_rad` | `0.00000995` | `0.00001000` | pass |
| `pose_step_l2_p95` | `0.01128157` | `0.01174183` | pass |
| `support_side_correctness` | `0` failures | `0` | pass |

Final minimax terms:

| surrogate | final value |
|---|---:|
| `loss_refactor_hard_max_gate_violation` | `1.49011612e-06` |
| `root_support_contact_support_side_support_lateral_product_margin_loss` | `1.49011612e-06` |
| `root_support_rootvel_rate_margin_loss` | `0.0` |
| `goal_heading_margin_loss` | `0.0` |
| `goal_yaw_rate_margin_loss` | `0.0` |
| `articulation_angvel_component_margin_loss` | `5.90e-10` |

The support closure trend was monotonic after the hard-gate surrogate was visible:

| run | support hard margin | support-side failures | adjusted failed family |
|---|---:|---:|---|
| e3000, no safety | `0.02211630` | `1` | `support_side_correctness,command_response` |
| e3600, no safety | `0.00929683` | `1` | `support_side_correctness` |
| e4600, no safety | `0.00377968` | `1` | `support_side_correctness` |
| e6200, no safety | `0.00060350` | `1` | `support_side_correctness` |
| e7000, no safety | `0.00000224` | `1` | `support_side_correctness` |
| e7000, `1e-6` safety | `0.00000149` | `0` | none |

Conclusion: one-window feasibility is confirmed for `flat state281 + deterministic decoder +
3 causal items`. The remaining failures were optimizer/surrogate alignment issues, not a
representation ceiling and not an infeasible one-window hard contract.

## Negative Controls

Adjusted guard was run on the saved canonical prediction with the same `1e-5` heading tolerance.

| guard | result |
|---|---:|
| shortcut negative controls still fail | `true` |
| command demotion negative controls still fail | `true` |
| one-window full-family pass | `true` |
| script-local debug decision | `adjusted_acceptance_guard_passed_ready_for_8window_debug_sweep` |
| research 8-window authorization | `false`, pending rootvel contract slack |

## Verdict

One-window c1 is confirmed.

The correct 3-item physical cut is `L_articulation`, `L_root_support`, `L_goal`. GT warm-start
plus low-LR minimax and hard-gate-aligned `support_lateral_product` clear rootvel-rate,
support-side correctness, rate budget, support honesty, pose continuity, and command response
on the one-window train-fit.

Do not run 8-window yet. The one-window artifact proves representation and one-window
feasibility; it does not resolve the separate zero-slack `rootvel_step_l2_p95` band issue.
Before an 8-window sweep, rootvel must be moved to the baseline-normalized / percentile
tolerance contract described in the 2026-06-01 acceptance contract.
