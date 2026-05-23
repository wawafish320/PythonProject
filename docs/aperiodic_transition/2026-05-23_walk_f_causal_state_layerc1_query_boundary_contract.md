# Walk_F Causal-State Scaffold v1 Layer C.1 Query Boundary + Censoring Audit Contract (2026-05-23)

> 本 memo 是 **CONTRACT + IMPL scope** for the first Layer C.1 probe. It remains
> read-only. It does not train, does not emit EventHead targets, does not define
> `handoff_ready`, does not define `transition_done`, does not assert attractor
> membership, does not emit cross-attractor claims, and does not consume or
> write checkpoints / fingerprints / train config / freeze policy.

## §0 Relation to Scaffold v1 and Layer C Minimal

Layer C.1 is a narrow implementation slice under

- `docs/aperiodic_transition/2026-05-22_walk_f_causal_state_scaffold_v1.md`
  §1.2 (planar-translation / global-yaw quotient semantics), §3.3-§3.4
  (degenerate-on-reference handling + feature-conditional phase structure),
  §4 (single-trajectory caveat), §5.3 (query setup + boundary fields),
  §5.4 (censoring is not failure), §6 (per-clip leave/return fields).
- `docs/aperiodic_transition/2026-05-23_walk_f_causal_state_layerc_minimal_contract.md`
  (mode `phase_library_check`, included groups, leave-one-neighborhood
  estimator grid, `phase_structure_status` enum, §8 non-outputs).

Layer C.1 inherits Layer C minimal verbatim except for the explicit additions
spelled out in §3-§8 below. It does NOT supersede Layer C minimal; both modes
remain available on the same tool.

## §1 Three Hard-Locked Rules (verbatim, no rewording)

These three statements are normative. Any implementation that violates them
MUST fail-fast (no silent fallback, no INSUFFICIENT_EVIDENCE fig-leaf, no
deprecation warning) per `docs/removal_policy.md` §3-§4.

1. **Layer C.1 only outputs query boundary audit.** It does NOT output
   attractor membership, transition truth, EventHead targets,
   `handoff_ready`, `transition_done`, or any cross-attractor claim. The
   artifact MUST explicitly mark every one of those fields as
   `not_emitted_by_this_tool` / `forbidden_by_contract`.

2. **In the single-trajectory Walk_F baseline, `left_censored_leave`,
   `right_censored_return`, and `return_to_reference_status =
   INSUFFICIENT_EVIDENCE` are all contract PASS, not implementation
   failure.** Justification: the scaffold v1 §4 single-trajectory caveat
   explicitly records that the Walk_F leave-one-phase baseline is
   ill-conditioned on the current 88-frame trajectory; §5.4 explicitly
   states that censored leave/return outcomes on short turn clips are valid
   probe results, not failures. The artifact root MUST therefore include
   `expected_insufficient_evidence_is_contract_pass = true` and a
   `censoring_summary` block that catalogues these outcomes without
   flagging them as errors.

3. **Query `leave_interval` / `return_interval` MUST be reported
   feature-group-separated. `turn_dyn` is NEVER folded into combined
   membership.** `turn_dyn` MAY appear only as a debug / inspection
   reference (e.g. yaw activity), never as part of a combined attractor
   score. Any code path that hands `turn_dyn` to a combined-membership /
   boundary-aggregation helper MUST raise `FailFastError`. There is NO
   tolerance threshold under which `turn_dyn` is "almost allowed"; the
   guardrail is binary.

## §2 Mode and Input

Tool mode:

```text
query_boundary_check
```

Allowed input `--clips` (set-equal; order-independent; duplicates forbidden):

```text
[Walk_F, Walk_L_To_L, Walk_L_To_R, Walk_R_To_L, Walk_R_To_R]
```

Reference and query families:

```text
reference_family = ["Walk_F"]
query_family     = ["Walk_L_To_L", "Walk_L_To_R", "Walk_R_To_L", "Walk_R_To_R"]
```

Fail-fast conditions BEFORE `args.out_dir.mkdir`:

- `--clips` is not set-equal to the 5-clip list above (missing, extra, or
  duplicated entries all fail-fast).
- `--mode` is anything other than `query_boundary_check`.
- Any clip's raw JSON is missing.
- Raw schema / unit checks already enforced by `_load_clip` /
  `_load_clip_contact_signals` (radians, meters, scalar `RootYaw`, vector
  `RootPosition` and `RootVelocityXY`, `FootEvidence.{L,R}.soft_contact_score`
  scalar).
- Internal SE(2) gauge precondition (`gauge_check` rerun across the 5
  clips) does not pass.
- Walk_F phase library produces zero valid phase candidates on every
  estimator setting for a non-degenerate group.

Layer C.1 MUST NOT read any external Layer A / Layer B / Layer C
artifact as its precondition. It reruns the lightweight SE(2) sanity
internally and rebuilds the Walk_F phase library on-the-fly via the
`phase_library_check` (Layer C minimal) helper functions.

## §3 Included and Excluded Feature Groups

Included feature groups (root_body_vel_only, root_body_pos_vel,
contact) are inherited verbatim from Layer C minimal §4. Their channel
maps, gauge source, and group ablation roles are reused unchanged.

Excluded feature groups (and the reason for exclusion):

- `turn_dyn`: excluded by §1 rule 3 above. Layer B already marks
  `turn_dyn` reference-degenerate on Walk_F; combining a zero-variance
  reference with a non-zero query signal would silently produce an
  unbounded combined score. Any combined-membership entry point that
  reads `turn_dyn` MUST raise.
- `pose_dyn`: not run; requires `processed_data/*.npz` schema validation
  and is reserved for Layer B.1.
- `pose_rel`: not run; requires a separate non-templating contract and
  is reserved for Layer B.1.
- absolute `RootYaw`: excluded by the canonical SE(2) quotient (§1.2).
- `yaw_activity_debug`: excluded; debug oracle only, not phase / boundary
  evidence.

## §4 Walk_F Phase Library Source

Layer C.1 reuses the Layer C minimal Walk_F phase library construction
(`_build_layer_c_group_matrix` + `_build_phase_windows`) but with the
following discipline:

- Walk_F z-matrix is built from Walk_F's own median + MAD-derived robust
  scale per channel.
- Query z-matrices are built by applying Walk_F's median + robust scale
  to the same channel set (NOT the query clip's own median/scale).
- If a feature group is `reference_degenerate` on Walk_F (no active
  channel survives the MAD epsilon), the group is marked
  `phase_degenerate` for ALL query clips × ALL configs and boundary
  detection is skipped. No silent fallback to a non-Walk_F reference
  scale is allowed.

Walk_F self-baseline loss distribution:

- Identical to Layer C minimal: for each Walk_F phase `t`, exclude
  candidate phases `p` with `abs(p - t) <= neighborhood_radius_frames`,
  match `H_F(t)` to nearest `H_F(p)` under the declared metric, predict
  `Y_F(t)` with `Y_F(p*)`, accumulate the future loss.
- The self-baseline band is derived from this loss distribution by
  taking `band_quantile_value = np.quantile(walk_f_self_baseline_loss,
  band_quantile)`.

Query phase-aware future loss:

- For each query phase `t_q` (built with the SAME history / future window
  on the query clip in the SAME quotient z-space), match `H_q(t_q)` to
  the nearest Walk_F phase `p*` (no neighborhood exclusion; the query
  comes from a different clip), compute `phase_loss_q(t_q) =
  metric(Y_q(t_q), Y_F(p*))`.

## §5 Estimator Grid

Layer C.1 takes the Layer C minimal 2x2x2x2 grid verbatim and adds three
boundary-detection dimensions:

| Dimension                                | Values                       | Unit         | Source     |
| ---                                      | ---                          | ---          | ---        |
| `history_window_frames`                  | `[6, 12]`                    | frames       | Layer C    |
| `future_horizon_frames`                  | `[6, 12]`                    | frames       | Layer C    |
| `neighborhood_radius_frames`             | `[4, 8]`                     | frames       | Layer C    |
| `distance_metric`                        | `["z_mse", "z_l1"]`          | per-channel  | Layer C    |
| `band_quantile`                          | `["P75", "P90", "P95"]`      | quantile     | Layer C.1  |
| `min_consecutive_frames_for_leave`       | `[2, 4]`                     | frames       | Layer C.1  |
| `min_consecutive_frames_for_return`      | `[2, 4]`                     | frames       | Layer C.1  |

Total per (query_clip, included_feature_group): 2x2x2x2x3x2x2 = 192 configs.
With 4 query clips × 3 included groups, this gives 2304 config rows.

Neighbor-consistency rule (mandatory):

- Two configs are NEIGHBORS if they differ by exactly one grid step in
  exactly one of the 7 dimensions above.
- The neighbor-consistency tolerance is the SMALLER of
  `min_consecutive_frames_for_leave` in the grid, i.e. 2 frames.
- For each config `c`, scan its neighbors. If any neighbor `c'`
  - has a leave_interval-existence boolean that differs from `c`, OR
  - has a return_interval-existence boolean that differs from `c`, OR
  - has both leave_intervals defined but disagrees by more than 2 frames
    on `leave_start_frame` OR `leave_end_frame`, OR
  - has both return_intervals defined but disagrees by more than 2 frames
    on `return_start_frame` OR `return_end_frame`,
  then `c.return_to_reference_status` is overridden to
  `INSUFFICIENT_EVIDENCE` and the conflicting `(c, c')` pair is recorded
  in the per-group aggregate.

## §6 Output Fields

Root summary fields (all required):

- `tool`
- `tool_mode = "read_only"`
- `contract = "walk_f_causal_state_scaffold_v1"`
- `mode = "query_boundary_check"`
- `track = "walk_f_query_leave_return_boundary_audit"`
- `track_role = "query_boundary_audit_not_attractor_membership_not_transition_truth"`
- `reference_family`
- `query_family`
- `clips`
- `layer_c1_contract_status` in `{pass, fail}`
- `expected_insufficient_evidence_is_contract_pass` (must be `true`)
- `internal_se2_gauge_precondition` (full block, source =
  `internal_lightweight_layerA_rerun_not_external_artifact`)
- `walk_f_phase_library_source` (must be
  `internal_phase_library_check_rerun_not_external_artifact`)
- `estimation_grid`
- `included_feature_groups`
- `excluded_feature_groups`
- `per_query_clip`
- `per_group_aggregate`
- `censoring_summary`
- `insufficient_evidence`

Per `(query_clip, feature_group, config)` (required):

- `history_window_frames`
- `future_horizon_frames`
- `neighborhood_radius_frames`
- `distance_metric`
- `band_quantile` (e.g. `"P90"`)
- `band_quantile_fraction` (e.g. `0.90`)
- `min_consecutive_frames_for_leave`
- `min_consecutive_frames_for_return`
- `valid_query_count`
- `valid_query_fraction`
- `band_quantile_value` (the actual scalar threshold derived from
  Walk_F's self-baseline loss distribution at this estimator setting)
- `phase_loss_quantiles_on_query`
- `self_baseline_band_quantiles` (full quantile summary of the
  Walk_F self-baseline loss distribution at this estimator setting)
- `leave_interval` (`[start_frame, end_frame]` closed-interval integer
  pair OR `null`)
- `return_interval` (`[start_frame, end_frame]` closed-interval integer
  pair OR `null`)
- `left_censored_leave` (boolean)
- `right_censored_return` (boolean)
- `return_to_reference_status` in
  `{returned, never_left, never_returned, INSUFFICIENT_EVIDENCE}`
- `return_to_reference_status_pre_neighbor_consistency` (raw status before
  the §5 neighbor-consistency override is applied; reported for audit
  transparency)
- `neighbor_consistency_conflicts` (list of conflicting neighbor config IDs)
- `out_of_band_frame_count` (count of query phase frames with phase_loss >
  band_quantile_value)
- `loss_curve` (per-query-phase-frame phase_loss + matched Walk_F frame +
  out_of_band bool)
- `loss_percentile_curve` (per-query-phase-frame phase_loss percentile
  vs Walk_F self-baseline loss distribution)

Per `(query_clip, feature_group)` aggregate (required):

- `phase_structure_status` in
  `{phase_structured, phase_degenerate, insufficient_evidence}` (inherits
  Layer C minimal semantics; Layer C.1 cannot promote a group to
  `phase_structured` on the current data)
- `reference_degenerate` (boolean; from Walk_F)
- `active_channels`
- `excluded_channels`
- `group_ablation_role`
- `returned_config_count`
- `never_left_config_count`
- `never_returned_config_count`
- `insufficient_evidence_config_count`
- `left_censored_leave_config_count`
- `right_censored_return_config_count`
- `neighbor_consistency_conflict_pair_count`
- `neighbor_consistency_conflict_pairs` (truncated list with config IDs)

## §7 Shape and Dtype Conventions

- All numpy arrays are `float64` on CPU.
- Per-frame matrices: `(T, C_active)`.
- History windows: flattened `(H * C_active,)` per phase frame.
- Future windows: flattened `(F * C_active,)` per phase frame.
- `leave_interval` and `return_interval` are closed-interval integer
  pairs `[start_frame, end_frame]` mapped back to clip frame indices via
  the query clip's phase-frame array, or `null`.

## §8 Explicit Non-Outputs (forbidden by contract)

Layer C.1 does NOT emit any of the following. The artifact MUST mark each
one as `not_emitted_by_this_tool` or `forbidden_by_contract`:

- `attractor_membership_status` (always
  `not_emitted_by_this_tool_layer_c1_query_boundary_only`)
- `event_head_target_status` (always `not_emitted_by_this_tool`)
- `handoff_ready_status` (always `not_emitted_by_this_tool`)
- `transition_done_status` (always `not_emitted_by_this_tool`)
- `cross_attractor_claim_status` (always `forbidden_by_contract`)
- `transition_truth_promotion` (always `forbidden_by_contract`)
- combined-membership / combined-score over feature groups
- any promotion of `phase_structure_status` to `phase_structured` on the
  current single-trajectory Walk_F baseline
- any field that consumes `turn_dyn` as combined-membership evidence

Any future memo that wants to add even ONE of these fields MUST version
the contract (new memo with a new date stamp under
`docs/aperiodic_transition/`), per `docs/removal_policy.md` §6.

## §9 Validation

Required validation (manual, no CI hookup yet):

```bash
python3 -m py_compile tools/run_walk_f_causal_state_probe.py

PYTHONPATH=. python3 tools/run_walk_f_causal_state_probe.py \
  --mode query_boundary_check \
  --clips Walk_F,Walk_L_To_L,Walk_L_To_R,Walk_R_To_L,Walk_R_To_R \
  --raw_root raw_data \
  --out_dir debug_output/walk_f_causal_state_scaffold_v1_20260523_layerC1_query_boundary_check
```

Fail-fast smoke tests (each must exit 2, leave no output directory):

- `--clips Walk_F` (missing every query clip)
- `--clips Walk_F,Walk_L_To_L,Walk_L_To_R,Walk_R_To_L` (missing `Walk_R_To_R`)
- `--clips Walk_F,Walk_F,Walk_L_To_L,Walk_L_To_R,Walk_R_To_L,Walk_R_To_R`
  (duplicate `Walk_F`)

Regression: `yaw_debug`, `gauge_check`, `reference_scale_check`, and
`phase_library_check` numbers MUST remain bit-identical to the committed
versions on `feat/walk-f-causal-state-probe` HEAD.
