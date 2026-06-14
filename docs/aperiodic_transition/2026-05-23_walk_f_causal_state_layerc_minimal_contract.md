> TRIAGE: DOWNGRADE-SUPERSEDED. Historical design/probe/planning note only; do not treat as live action-handoff delivery plan. Superseded by `2026-06-07_action_handoff_inbetween_closeout_decision_record.md` §1/§3.4 under its stated read-only / zero-new-injection scope.

# Walk_F Causal-State Scaffold v1 Layer C Minimal Contract (2026-05-23)

> 本 memo 是 **CONTRACT + IMPL scope** for the first Layer C probe. It remains
> read-only. It does not train, does not emit EventHead targets, does not define
> `handoff_ready`, does not define `transition_done`, and does not consume or
> write checkpoints / fingerprints / train config.

## §0 Relation to Scaffold v1

Layer C minimal is a narrow implementation slice under
`docs/aperiodic_transition/2026-05-22_walk_f_causal_state_scaffold_v1.md`.

It inherits:

- `reference_family = ["Walk_F"]`;
- planar translation / global-yaw quotient semantics;
- single-trajectory caveat: `Walk_F` has one 88-frame clip, so class-level
  recurrence remains `INSUFFICIENT_EVIDENCE`;
- `turn_dyn` degeneracy guardrail;
- audit-only output.

This memo does not supersede the scaffold v1 definition. It only fixes the first
Layer C estimator grid and output boundary.

## §1 Mode and Input

Tool mode:

```text
phase_library_check
```

Allowed input clips:

```text
Walk_F
```

Layer C minimal MUST fail-fast if any non-reference query clip is supplied.
`Walk_L_To_L`, `Walk_L_To_R`, `Walk_R_To_L`, and `Walk_R_To_R` belong to Layer
C.1 query boundary / censoring, not this mode.

Reason: Layer C minimal only asks whether the single `Walk_F` trajectory has a
repeatable phase-library self-consistency signal under leave-one-neighborhood
matching. It does not estimate membership boundaries for query clips.

## §2 Expected Evidence Status

The expected current-data evidence outcome is:

```text
phase_structure_status = insufficient_evidence
evidence_status = INSUFFICIENT_EVIDENCE
layer_c_contract_status = pass
```

This is not an implementation failure. The parent scaffold already records that
the single-trajectory leave-one-phase baseline is ill-conditioned. On 88 frames
at 60fps, neighborhood radius choices trade off between too few independent
candidates and local phase leakage.

Layer C minimal may report a self-consistency signal, including a signal that is
consistent across the full estimator grid, but on the current single `Walk_F`
trajectory it MUST keep `phase_structure_status = insufficient_evidence`.
`phase_structured` requires a stronger data setting or a follow-up contract that
can separate true recurrent phase from same-clip leakage.

## §3 Estimator Grid

Layer C minimal uses a fixed 2 x 2 x 2 x 2 grid:

| Dimension | Values | Unit |
| --- | --- | --- |
| `history_window_frames` | `[6, 12]` | frames |
| `future_horizon_frames` | `[6, 12]` | frames |
| `neighborhood_radius_frames` | `[4, 8]` | frames |
| `distance_metric` | `["z_mse", "z_l1"]` | per-channel robust-normalized |

The tool must report every grid point. If one setting appears to beat the
phase-agnostic baseline but nearby settings do not, the result is
`phase_structure_status = insufficient_evidence`.

## §4 Feature Groups

Included groups:

- `root_body_vel_only`: canonical-yaw quotient root velocity only. This is the
  preferred root-body self-consistency view because it reduces clip-start
  displacement leakage.
- `root_body_pos_vel`: canonical-yaw quotient displacement + velocity. This is
  reported as an ablation only because clip-start-anchored displacement may carry
  elapsed-progress/template information.
- `contact`: `FootEvidence.{L,R}.soft_contact_score`.

Excluded groups:

- `turn_dyn`: excluded from Layer C membership / phase claims because Layer B
  showed it is degenerate on `Walk_F`.
- `pose_dyn`: not run; requires processed-data schema validation.
- `pose_rel`: not run; needs a separate non-template contract.
- raw absolute `RootYaw`: excluded from the quotient state.
- `yaw_activity_debug`: excluded; debug oracle only.

## §5 Algorithm Boundary

For each included feature group and each estimator setting:

1. Build valid `Walk_F` phase candidates from history window `H_F(t)` and future
   window `Y_F(t)` in the same quotient space.
2. For every valid query phase `t` from the same `Walk_F` clip, exclude candidate
   phases `p` where `abs(p - t) <= neighborhood_radius_frames`.
3. Match `H_F(t)` to the nearest remaining `H_F(p)` under the declared metric.
4. Predict `Y_F(t)` with `Y_F(p*)`.
5. Compare phase-aware future loss against a phase-agnostic future baseline built
   from the same nonlocal candidate set.

This is a self-consistency diagnostic, not attractor membership.

## §6 Output Fields

Required root fields:

- `tool_mode = "read_only"`
- `contract = "walk_f_causal_state_scaffold_v1"`
- `mode = "phase_library_check"`
- `track = "walk_f_phase_library_self_consistency"`
- `reference_family = ["Walk_F"]`
- `clips = ["Walk_F"]`
- `layer_c_contract_status`
- `expected_insufficient_evidence_is_contract_pass`
- `internal_se2_gauge_precondition`
- `estimation_grid`
- `included_feature_groups`
- `excluded_feature_groups`
- `phase_structure_status_per_group`
- `self_consistency_signal_per_group`
- `walk_f_leave_one_neighborhood_baseline`
- `insufficient_evidence`

Per group:

- `phase_structure_status` in
  `{phase_structured, phase_degenerate, insufficient_evidence}`
- `evidence_status` in `{PASS, FAIL, INSUFFICIENT_EVIDENCE}`
- `self_consistency_signal_status`
- `reference_degenerate`
- `active_channels`
- `excluded_channels`
- `config_results`
- `baseline_loss_quantiles`
- `loss_curve_summary`
- `phase_hat_curve_summary`
- `group_ablation_role`

Per config:

- `history_window_frames`
- `future_horizon_frames`
- `neighborhood_radius_frames`
- `distance_metric`
- `valid_phase_candidate_count`
- `valid_query_count`
- `valid_query_fraction`
- `phase_loss_quantiles`
- `phase_agnostic_loss_quantiles`
- `median_relative_improvement`
- `beats_phase_agnostic_baseline`
- `phase_hat_curve`
- `loss_curve`
- `loss_percentile_curve`
- `phase_confidence_gap`

## §7 Fail-Fast Conditions

Layer C minimal MUST fail-fast before writing artifacts when:

- `--clips` is not exactly `Walk_F`;
- `Walk_F` raw JSON is missing;
- required raw fields are missing or non-finite;
- raw units are not meters or `RootYaw` is not radians;
- `FootEvidence.{L,R}.soft_contact_score` is missing;
- the internal SE(2) sanity precondition does not pass;
- an unknown mode is supplied;
- implementation attempts to emit query leave/return fields, EventHead targets,
  `handoff_ready`, or `transition_done`.

Layer C minimal MUST NOT read a prior Layer A artifact as its precondition. It
reruns the lightweight SE(2) sanity internally to avoid stale artifact
dependencies.

## §8 Explicit Non-Outputs

Layer C minimal does not output:

- `leave_interval`;
- `return_interval`;
- `left_censored_leave`;
- `right_censored_return`;
- `return_to_reference_status`;
- attractor membership;
- transition truth;
- EventHead target;
- `handoff_ready`;
- `transition_done`;
- cross-attractor claims.

Those fields require query phase lookup and membership boundary/censoring logic
and are reserved for Layer C.1.
