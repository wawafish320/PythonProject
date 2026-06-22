> TRIAGE: KEEP-AS-PROVENANCE. Historical evidence / anti-relitigation note; preserve to explain why this path was rejected or bounded. Do not use as a live plan unless a new decision record explicitly reopens it.

# Action-Handoff Inbetween Train-Fit Independent Review

Date: 2026-06-04

Scope: independent read-only review. I read the requested status docs, train-fit saga docs,
the three debug scripts, and the cited artifacts. I did not train, launch a probe, edit
production `train/` code, mutate a checkpoint, or change a gate/runtime. This file is the only
write.

## Executive Verdict

Proceed to an 8-window debug train-fit, but only as a Layer-2 fixed-oracle-schedule
train-fit discriminator. Do not describe it as deployment, schedule learning, `Walk_L_To_R`
coverage, or recipe generalization.

The one-window result is real in the narrow sense: `flat state281 + deterministic decoder +
3 causal items + GT warm-start + low-LR minimax + hard-gate-aligned support_lateral_product`
can satisfy the full adjusted family for `Walk_L_To_L:0-15`. It is also a deliberately heavy
single-window overfit/memorization proof. The current evidence excludes a one-window
representation ceiling; it does not prove multi-window capacity, optimization stability,
sampling unnecessary in general, or deployment readiness.

## Carrying Conclusions

### 1. c2 excluded / one-window full-family pass

Verdict: **confirmed with strict scope**.

Evidence:

- The status snapshot already scopes the representation conclusion to the one-window question:
  `state281` remains flat and c2 is excluded only for that question
  (`docs/aperiodic_transition/2026-06-04_status_synthesis.md:16`,
  `docs/aperiodic_transition/2026-06-04_status_synthesis.md:17`).
- The final run used `x [1,4957] float32 CPU` to produce `state281 [1,16,281] float32 CPU`,
  `bone_angvel [1,16,138] float32 CPU`, and saved `pred_raw [1,6704] float32 CPU NumPy`
  (`docs/aperiodic_transition/2026-06-04_loss_refactor.md:283`,
  `docs/aperiodic_transition/2026-06-04_loss_refactor.md:284`,
  `docs/aperiodic_transition/2026-06-04_loss_refactor.md:285`,
  `docs/aperiodic_transition/2026-06-04_loss_refactor.md:286`).
- The final adjusted metrics pass all families: rootvel `0.03490959 < 0.04241988`,
  yaw-rate `0.13711222 < 0.13711452`, heading `9.95e-6 < 1e-5`, pose `0.01128157 <
  0.01174183`, and support-side failures `0`
  (`docs/aperiodic_transition/2026-06-04_loss_refactor.md:288`,
  `docs/aperiodic_transition/2026-06-04_loss_refactor.md:295`,
  `docs/aperiodic_transition/2026-06-04_loss_refactor.md:296`,
  `docs/aperiodic_transition/2026-06-04_loss_refactor.md:299`,
  `docs/aperiodic_transition/2026-06-04_loss_refactor.md:300`,
  `docs/aperiodic_transition/2026-06-04_loss_refactor.md:301`).
- The model is a small flattened MLP, not a structured rollout model
  (`tools/run_action_handoff_oracle_schedule_trajectory_decoder_smoke.py:123`,
  `tools/run_action_handoff_oracle_schedule_trajectory_decoder_smoke.py:127`,
  `tools/run_action_handoff_oracle_schedule_trajectory_decoder_smoke.py:131`), and the final
  one-window run explicitly used `supervised_flat` warm-start
  (`docs/aperiodic_transition/2026-06-04_loss_refactor.md:196`,
  `docs/aperiodic_transition/2026-06-04_loss_refactor.md:265`).

Interpretation: this proves one-window feasibility / expressivity under an overfit ladder. It
also proves the previous "flat state281 cannot possibly express this one window" claim was too
strong. It does **not** prove generalization. The status note says the same explicitly:
"Do not treat one-window c1 as recipe generalization"
(`docs/aperiodic_transition/2026-06-04_status_synthesis.md:86`).

### 2. Negative controls under relaxed bands

Verdict: **confirmed for available controls; p99 width remains a contract risk**.

Evidence:

- Band audit relabeled rootvel using continuous p99: `Walk_L_To_L` rootvel band
  `0.04241988 -> 0.08486664`, `Walk_R_To_L` `0.04802140 -> 0.07909000`, and
  `Walk_R_To_R` `0.07419514 -> 0.10134733`
  (`docs/aperiodic_transition/2026-06-04_band_audit.md:60`,
  `docs/aperiodic_transition/2026-06-04_band_audit.md:61`,
  `docs/aperiodic_transition/2026-06-04_band_audit.md:63`).
- The final combined guard under accepted relabels reports one-window pass, shortcut controls
  still fail, command demotion controls still fail, reconstructed GT acceptance `1.0000`, and
  decoder-path-from-GT acceptance `1.0000`
  (`docs/aperiodic_transition/2026-06-04_band_audit.md:79`,
  `docs/aperiodic_transition/2026-06-04_band_audit.md:80`,
  `docs/aperiodic_transition/2026-06-04_band_audit.md:81`,
  `docs/aperiodic_transition/2026-06-04_band_audit.md:82`,
  `docs/aperiodic_transition/2026-06-04_band_audit.md:83`).
- The band audit script tests every candidate relabel, then applies all accepted relabels and
  reruns the final combined guard (`tools/run_action_handoff_band_audit.py:731`,
  `tools/run_action_handoff_band_audit.py:786`). Shortcut fail is computed as all cases having
  adjusted pass rate `0.0`, not just as an aggregate pass count
  (`tools/run_action_handoff_band_audit.py:416`,
  `tools/run_action_handoff_band_audit.py:418`).
- In the final combined JSON, `negative_control:direct_full` still has `n=3`, adjusted pass
  count `0`, failed families `rate_budget`, `pose_continuity`, and `endpoint_bridgeability`
  (`debug_output/_tmp_action_handoff_band_audit_20260604/band_audit_summary.json:1434`).
  Command demotion has `n=21`, demoted negative pass count `0`
  (`debug_output/_tmp_action_handoff_band_audit_20260604/band_audit_summary.json:1507`).
- The command-demotion replay itself shows the direct/lambda family does not rely on a
  single command gate: direct and lambda still fail `rate_budget`, `command_compatibility`,
  `pose_continuity`, and `endpoint_bridgeability`
  (`docs/aperiodic_transition/2026-06-03_command_response_demotion_replay.md:67`,
  `docs/aperiodic_transition/2026-06-03_command_response_demotion_replay.md:68`).

Risk: p99 is permissive relative to the acceptance-contract language around p95/p95-style rate
budgeting (`docs/aperiodic_transition/2026-06-01_middle_generator_acceptance_contract.md:154`,
`docs/aperiodic_transition/2026-06-01_middle_generator_acceptance_contract.md:159`,
`docs/aperiodic_transition/2026-06-01_middle_generator_acceptance_contract.md:160`). It is
acceptable as an anti-zero-slack debug relabel, but 8-window must report a shadow p95-normalized
view. A sample that only passes because of p99 rootvel should be labeled "p99-only pass", not
clean motion quality.

### 3. Three causal terms + minimax + hinge formulation

Verdict: **confirmed**.

Evidence:

- The current cut maps articulation, root/support, and goal terms to the relevant hard metrics
  (`docs/aperiodic_transition/2026-06-04_loss_refactor.md:50`,
  `docs/aperiodic_transition/2026-06-04_loss_refactor.md:53`,
  `docs/aperiodic_transition/2026-06-04_loss_refactor.md:55`).
- `_band_margin_loss` is a true hinge on normalized band violation:
  `relu(vals / band - 1)^2`, so it has zero loss/gradient once inside band
  (`tools/run_action_handoff_dynamics_consistency_train_fit_ladder.py:612`,
  `tools/run_action_handoff_dynamics_consistency_train_fit_ladder.py:623`).
- `_interval_margin_loss` uses the same hard-gate tolerance form as support-side evaluation,
  with an optional safety margin for hard-gate keys
  (`tools/run_action_handoff_dynamics_consistency_train_fit_ladder.py:636`,
  `tools/run_action_handoff_dynamics_consistency_train_fit_ladder.py:653`,
  `tools/run_action_handoff_support_contract_tightening_probe.py:414`,
  `tools/run_action_handoff_support_contract_tightening_probe.py:415`).
- The train surrogate and support-side hard feature both define `support_lateral_product` as
  support balance times `root_lateral_mean`
  (`tools/run_action_handoff_dynamics_consistency_train_fit_ladder.py:914`,
  `tools/run_action_handoff_support_contract_tightening_probe.py:380`).
- Minimax is formed over hard gate terms plus individual support feature terms, with the anchor
  tie-breaker separated (`tools/run_action_handoff_dynamics_consistency_train_fit_ladder.py:1058`,
  `tools/run_action_handoff_dynamics_consistency_train_fit_ladder.py:1074`,
  `tools/run_action_handoff_dynamics_consistency_train_fit_ladder.py:1086`).

No hidden inconsistency found. The final `support_lateral_product` closure is hard-gate aligned,
not merely a same-name surrogate. It uses tolerance/safety, so the scalar value is not literally
the hard gate boolean, but the zero boundary is intentionally inside the pass tolerance.

### 4. Razor-thin margins

Verdict: **confirmed as expected minimax boundary behavior; high fragility for 8-window**.

Evidence:

- Final old-band one-window margins are thin for yaw-rate (`2.30e-6`), heading (`4.68e-8`),
  and contact-step event-aware equality; rootvel and pose have more slack
  (`debug_output/_tmp_action_handoff_causal_loss_refactor_minimax_gtwarm_flat2000_tail1e5_lat_hardtol0p01_safe1e6_tau0p005_e7000_20260604/summary.md:55`,
  `debug_output/_tmp_action_handoff_causal_loss_refactor_minimax_gtwarm_flat2000_tail1e5_lat_hardtol0p01_safe1e6_tau0p005_e7000_20260604/summary.md:58`,
  `debug_output/_tmp_action_handoff_causal_loss_refactor_minimax_gtwarm_flat2000_tail1e5_lat_hardtol0p01_safe1e6_tau0p005_e7000_20260604/summary.md:62`).
- Final hard max gate violation is `1.49e-6`, concentrated in
  `support_lateral_product`
  (`docs/aperiodic_transition/2026-06-04_loss_refactor.md:303`,
  `docs/aperiodic_transition/2026-06-04_loss_refactor.md:308`).
- Since the hinge terms go zero inside band, the objective does not create extra slack unless
  the anchor tie-breaker happens to move that way (`tools/run_action_handoff_dynamics_consistency_train_fit_ladder.py:623`,
  `tools/run_action_handoff_dynamics_consistency_train_fit_ladder.py:1086`).

Interpretation: this is not the same as the old zero-slack contract hold. The p99 relabel gives
rootvel contract slack, and the one-window rootvel already passed the old band. The remaining
near-boundary values are optimizer/objective geometry: minimax drives the worst term just under
the gate and then stops caring. For 8-window, a pass-rate-only report is insufficient; require
per-window normalized slack and near-boundary counts.

### 5. Prior correction chain

Verdict: **confirmed; no evidence of reverse over-correction**.

Audits:

- Entanglement: the dynamics ladder explicitly says acceptance stayed `0.0000` but gradient
  cosines did not justify pose/dynamics lifting; the old entanglement conclusion was withdrawn
  (`docs/aperiodic_transition/2026-06-03_dynamics_consistency_train_fit_ladder.md:130`,
  `docs/aperiodic_transition/2026-06-03_dynamics_consistency_train_fit_ladder.md:138`,
  `docs/aperiodic_transition/2026-06-03_dynamics_consistency_train_fit_ladder.md:151`).
- Pose c1/c2: localization first kept pose as `c_pending`, then the follow-up gate-aligned
  pose surrogate brought `pose_step_l2_p95` under band without earning c2
  (`docs/aperiodic_transition/2026-06-03_hard_band_localization.md:87`,
  `docs/aperiodic_transition/2026-06-03_hard_band_localization.md:98`,
  `docs/aperiodic_transition/2026-06-03_hard_band_localization.md:101`).
- Lifted/anchored representation: the full layer-2 harness reports no robust anchored-over-flat
  win over 22,560 rows and 188 windows, with native anchored losing valid pass, heading, rate,
  and support-side on many comparable groups
  (`debug_output/_tmp_action_handoff_layer2_harness_20260603/layer2_harness_full_verdict.md:7`,
  `debug_output/_tmp_action_handoff_layer2_harness_20260603/layer2_harness_full_verdict.md:14`,
  `debug_output/_tmp_action_handoff_layer2_harness_20260603/layer2_harness_full_verdict.md:28`,
  `debug_output/_tmp_action_handoff_layer2_harness_20260603/layer2_harness_full_verdict.md:31`,
  `debug_output/_tmp_action_handoff_layer2_harness_20260603/layer2_harness_full_verdict.md:32`,
  `debug_output/_tmp_action_handoff_layer2_harness_20260603/layer2_harness_full_verdict.md:33`).
- Root/support collapse: the old root split across dynamics/contact was corrected to
  `L_root_support`, and the final one-window pass supports the optimizer/surrogate-alignment
  interpretation (`docs/aperiodic_transition/2026-06-04_loss_refactor.md:42`,
  `docs/aperiodic_transition/2026-06-04_loss_refactor.md:116`,
  `docs/aperiodic_transition/2026-06-04_loss_refactor.md:325`).

## Next-Round Direction

Verdict: **confirm 8-window, with Stage 0 instrumentation before any run is interpreted**.

The proposed Stage1/Stage2/Stage3 structure is the right next step because all current
one-window blockers have been reduced to either confirmed train-fit feasibility or band-contract
scope. Another one-window round is likely lower value than testing whether the same machinery
survives multiple windows.

Required preflight/instrumentation:

1. Use the same fixed-oracle-schedule Layer-2 scope. Log decoder input as `[8,4957] float32 CPU`
   and outputs as `state281 [8,16,281] float32 CPU`, `bone_angvel [8,16,138] float32 CPU`
   unless the script explicitly moves device.
2. Select 8 windows stratified across `Walk_L_To_L`, `Walk_R_To_L`, and `Walk_R_To_R`, including
   support-switch windows and high rootvel/yaw-rate percentile windows. Do not use only easy
   neighbors of `Walk_L_To_L:0-15`.
3. Report per-window rows: target, start/end, support switch frames, every family boolean, raw
   metric value, band, normalized slack `(band - value) / band`, and a near-boundary flag for
   `abs(slack) < 1e-3` and `<1e-4`.
4. Keep two band views: accepted p99 bands for the decision gate, plus p95-shadow / old-band
   diagnostics. Any "p99-only pass" must be called out.
5. Run Stage1 as supervised-fit-8 first. This separates path/capacity from minimax optimization,
   but it is still memorization evidence because the MLP capacity is large relative to 8 samples.
6. Run Stage2 minimax-8 over the worst `(window x metric)` term, not averaged family loss.
   Report worst-window identity and per-window pass rate, not only aggregate pass.
7. Add at least two seeds or basin arms: random/weighted and supervised-flat warm-start. If only
   GT warm-start passes, classify as optimization-fragile, not generalization-ready.
8. Re-run shortcut and command-demotion guards under the accepted combined bands. For artifact
   rows that cannot be trajectory-rescored, explicitly list the non-rate families that still
   fail.

Systemic risks that should stay visible:

- Oracle schedule is still an oracle. A successful 8-window run proves only fixed-schedule
  Layer-2 train-fit; it does not solve support/event schedule prediction.
- The 188-window reconstructability universe excludes the ungroundable `Walk_L_To_R` transition:
  `Walk_L_To_R` is audit-only in the band audit and has `contact_d=0.7031`
  (`docs/aperiodic_transition/2026-06-04_band_audit.md:30`,
  `debug_output/_tmp_action_handoff_middle_acceptance_replay_probe_20260601/middle_acceptance_replay_summary.md:46`).
  Passing 8 of the compatible windows does not answer the strategic incompatible-side problem.
- If 8-window stalls, classify in this order: invalid guard/path, supervised capacity failure,
  minimax optimization failure, then conditioning conflict/multimodality. Do not jump directly
  to sampling/diffusion unless deterministic fixed-schedule train-fit is already clean and the
  residual evidence shows accepted multi-branch alternatives under identical conditioning.

No better prior step found, except the Stage 0 instrumentation above. The next action should not
be production integration, schedule learning, full 188-window fitting, or a representation swap.
