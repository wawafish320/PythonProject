# Walk_F Causal-State Scaffold v1 Layer B.1 Pose Reference-Scale + Degeneracy Audit Contract (2026-05-23)

> 本 memo 是 **CONTRACT + IMPL scope** for the first Layer B.1 probe. It is
> strictly read-only. It does NOT train, does NOT emit EventHead targets,
> does NOT define `handoff_ready`, does NOT define `transition_done`, does
> NOT assert attractor membership, does NOT emit query leave/return
> boundaries, does NOT emit cross-attractor claims, does NOT compute a
> combined-membership / combined-score, does NOT consume or write
> checkpoints / fingerprints / train config / freeze policy, does NOT
> write to `raw_data/`.

## §0 Relation to Scaffold v1 and Existing Layers

Layer B.1 is a narrow implementation slice under

- `docs/aperiodic_transition/2026-05-22_walk_f_causal_state_scaffold_v1.md`
  §1.2 (planar-translation / global-yaw quotient semantics),
  §3.3 (degenerate-on-reference handling),
  §3.4 (feature-conditional phase structure),
  §4 (single-trajectory caveat),
  §5.2 (pose_dyn / pose_rel feature groups),
  §6 (per-group `reference_scale`, `reference_degenerate`,
  `phase_structure_status` enum).
- `docs/aperiodic_transition/2026-05-23_walk_f_causal_state_layerc_minimal_contract.md`
  (`reference_family = ["Walk_F"]`, single-trajectory caveat).
- `docs/aperiodic_transition/2026-05-23_walk_f_causal_state_layerc1_query_boundary_contract.md`
  (the `turn_dyn`-never-folded-into-combined-membership guardrail
  inherits verbatim).

Layer B.1 does NOT supersede Layer B (mode `reference_scale_check`,
which already covers `root_body` / `turn_dyn` / `contact` reference
scale on the raw-JSON SE(2) quotient). It strictly adds the two
processed-data-only groups (`pose_dyn`, `pose_rel`) that Layer B
intentionally deferred (see Layer B `not_run_feature_groups_layer_b`
= `("pose_dyn", "pose_rel")` at `tools/run_walk_f_causal_state_probe.py:152`
and the Layer B summary `insufficient_evidence` entry "pose_dyn /
pose_rel ... deferred to Layer B.1" at
`tools/run_walk_f_causal_state_probe.py:3607`).

Both `reference_scale_check` (Layer B) and `pose_reference_scale_check`
(Layer B.1) remain available on the same tool.

## §1 Three Hard-Locked Rules (verbatim, no rewording)

These three statements are normative. Any implementation that violates
them MUST fail-fast (no silent fallback, no `INSUFFICIENT_EVIDENCE`
fig-leaf, no deprecation warning) per `docs/removal_policy.md` §3-§4.

1. **Layer B.1 only emits per-feature-group reference scale +
   degeneracy + per-clip schema/finite/frame-alignment audit on the
   `pose_dyn` and `pose_rel` groups.** It does NOT emit
   `phase_structured`, leave/return intervals, attractor membership,
   transition truth, EventHead targets, `handoff_ready`,
   `transition_done`, or any cross-attractor claim. The artifact MUST
   mark each of those fields as `not_emitted_by_this_tool` /
   `forbidden_by_contract` (see §9).

2. **Walk_F is the only reference. Query clips are NEVER allowed to
   re-derive their own per-channel median/MAD scale.** Walk_F's
   per-channel median + MAD is the authoritative robust scale; query
   clip per-channel summaries are reported on Walk_F-projected
   z-statistics (i.e. `(query_channel - walk_f_median) /
   max(walk_f_robust_std, epsilon)`). Enforcement is load-bearing,
   not advisory: every normalization helper on the Layer B.1 query
   path MUST accept a `scale_source: str` keyword and MUST call
   `_layer_b1_forbid_query_self_rescale(...)` whenever
   `scale_source != "walk_f"`. Currently the only such helper is
   `_layer_b1_query_channel_walk_f_projected`. Any future helper
   that performs query normalization MUST also wire the same
   sentinel; reviewers may grep for the helper name and verify the
   `scale_source` kwarg is required and validated.

3. **`turn_dyn` is NEVER folded into a combined-membership /
   combined-score over pose / root / contact groups.** The Layer B.1
   probe does NOT compute a combined score in any form — that is
   reserved for later layers and explicitly forbidden here. If any
   code path attempts to hand `turn_dyn` (or any future degenerate
   group) into a combined-membership helper, it MUST raise
   `FailFastError`. There is NO tolerance threshold under which
   `turn_dyn` is "almost allowed"; the guardrail is binary.

## §2 Mode and Input

Tool mode:

```text
pose_reference_scale_check
```

Allowed input `--clips` (set-equal; order-independent; duplicates
forbidden):

```text
[Walk_F, Walk_L_To_L, Walk_L_To_R, Walk_R_To_L, Walk_R_To_R]
```

Reference and query families:

```text
reference_family = ["Walk_F"]
query_family     = ["Walk_L_To_L", "Walk_L_To_R", "Walk_R_To_L", "Walk_R_To_R"]
```

Fail-fast conditions BEFORE `args.out_dir.mkdir`:

- `--clips` is not set-equal to the 5-clip list above (missing, extra,
  or duplicated entries all fail-fast).
- `--mode` is anything other than `pose_reference_scale_check` and the
  caller dispatched to this code path.
- Any clip's raw JSON is missing (loaded for frame-count cross-check
  via `_load_clip`; raw JSON pose feature fields are NOT consumed by
  Layer B.1 — see §4).
- Any clip's `raw_data/processed_data/<clip>.npz` is missing.
- Any required NPZ key is missing or has the wrong rank/shape/dtype
  (see §3).
- Any required NPZ value is non-finite.
- Processed-data NPZ frame count `T_npz` disagrees with raw-JSON
  `len(Frames)`.
- Processed-data NPZ FPS disagrees with raw-JSON `FPS`.
- Processed-data `meta_json.spaces.bone_rotations != "local_6d"` or
  `meta_json.spaces.bone_angular_velocities != "local_rad_per_sec"`
  (silent unit conversion is forbidden).

Layer B.1 MUST NOT read any external Layer A / Layer B / Layer C /
Layer C.1 artifact as its precondition.

## §3 Processed-Data Schema Validation (`raw_data/processed_data/*.npz`)

Each clip's `<clip>.npz` MUST satisfy:

| Key | Required rank / shape | Required dtype | Source field |
| --- | --- | --- | --- |
| `bone_ang_vel` | `(T, 46, 3)` | `float32` | per-bone local angular velocity (`spaces.bone_angular_velocities = "local_rad_per_sec"`) |
| `bone_rot6d` | `(T, 46, 6)` | `float32` | per-bone local 6D rotation (`spaces.bone_rotations = "local_6d"`) |
| `bone_names` | `(46,)` | object (Python str) | bone identifier list |
| `parents` | `(46,)` | `int32` | parent-bone index (for documentation only; not consumed by B.1) |
| `FPS` | scalar | `int32` | clip frame rate; MUST equal raw-JSON `FPS` |
| `meta_json` | scalar object | dict | MUST contain `units == "meters"`, `spaces.bone_rotations == "local_6d"`, `spaces.bone_angular_velocities == "local_rad_per_sec"` |

Layer B.1 reads the NPZ with `allow_pickle=True` because `meta_json`
and `bone_names` are object arrays. The implementation MUST limit
`allow_pickle=True` strictly to these two reads — all numeric arrays
are validated by explicit dtype / shape checks.

Frame alignment:

- `T_npz_bone_ang_vel == T_npz_bone_rot6d == len(raw_json.Frames)`.
- `T_npz` MUST be `>= 4` (matches Layer B
  `MIN_FRAMES_FOR_PHASE_ESTIMABILITY`; all current 5 clips satisfy this).
- `bone_names_npz_count == 46`. Bone count is also asserted against
  `bone_ang_vel.shape[1]` and `bone_rot6d.shape[1]`.

The implementation MUST NOT silently fall back to raw-JSON pose
fields when an NPZ key is missing. Raw JSON is consumed only for
clip presence / frame count / FPS cross-check (already enforced by
the shared `_load_clip` helper).

## §4 Feature Group Definitions

### §4.1 `pose_dyn`

- **Source array**: `bone_ang_vel` (shape `(T, 46, 3)`, units
  `local_rad_per_sec` per `meta_json.spaces.bone_angular_velocities`).
- **Extracted matrix shape**: `(T, 138)` after flattening
  `(T, 46, 3) -> (T, 46 * 3)` in row-major order
  `bone_index * 3 + axis_index`.
- **Channel naming**: 138 channels, `bone_ang_vel.<bone_name>.<axis>`
  where `<axis>` is one of `x`, `y`, `z` and `<bone_name>` is the
  exact entry from `bone_names`.
- **dtype**: `float64` (cast from source `float32`); device `cpu`.
- **Quotient discipline**: `bone_ang_vel` is in bone-LOCAL frame per
  the NPZ `spaces` metadata, therefore it is by construction
  invariant to global-yaw rotation and planar translation. Layer B.1
  does NOT re-apply the SE(2) quotient to this group; doing so would
  double-count the local-frame transform.
- **`group_ablation_role`**: `bone_local_angular_velocity_pose_dynamics`.

### §4.2 `pose_rel`

- **Source array**: forward-difference of `bone_rot6d` (shape
  `(T, 46, 6)`, units `local_6d` per
  `meta_json.spaces.bone_rotations`), scaled by FPS to a
  per-second rate.
- **Extracted matrix shape**: `(T, 276)`, computed as
  ```
  rot6d_flat = bone_rot6d.reshape(T, 46 * 6)  # float64
  pose_rel = np.zeros((T, 276), dtype=np.float64)
  pose_rel[1:] = (rot6d_flat[1:] - rot6d_flat[:-1]) * fps
  ```
  Frame `t=0` is zero by construction (same first-frame convention
  as `yaw_rate_rad_per_s` in `_canonical_quotient_features`). This
  is the "relative pose delta" view required by scaffold v1 §5.2
  and is NOT the raw `bone_rot6d` itself; raw-pose templating is
  explicitly avoided.
- **Channel naming**: 276 channels,
  `bone_rot6d_dot.<bone_name>.c<component_index>` where
  `<component_index>` is one of `0`, `1`, `2`, `3`, `4`, `5` and
  `<bone_name>` is the exact entry from `bone_names`.
- **dtype**: `float64`; device `cpu`.
- **Quotient discipline**: `bone_rot6d` is in bone-LOCAL frame per
  the NPZ `spaces` metadata, therefore the time-delta is also in
  bone-LOCAL frame and is invariant to global-yaw rotation and
  planar translation. Layer B.1 does NOT re-apply the SE(2)
  quotient.
- **`group_ablation_role`**:
  `bone_local_rot6d_first_difference_relative_pose_delta_velocity`.

### §4.3 Excluded Groups (and reason)

- `turn_dyn`: excluded by §1 rule 3 above. Already marked
  reference-degenerate on Walk_F by Layer B at
  `feature_groups_layer_b.turn_dyn`.
- `root_body`, `root_body_vel_only`, `root_body_pos_vel`, `contact`:
  not in scope; already covered by Layer B
  (`reference_scale_check`) and Layer C minimal
  (`phase_library_check`). Layer B.1 is the pose-only slice.
- absolute `RootYaw`: excluded by the canonical SE(2) quotient
  (scaffold v1 §1.2).
- `yaw_activity_debug`: excluded; debug oracle only.

## §5 Walk_F Reference Scale + Degeneracy Boundary

Walk_F per-channel statistics (computed exclusively from Walk_F):

- `median = float(np.median(finite))`
- `mad    = float(np.median(np.abs(finite - median)))`
- `robust_std_from_mad = float(mad * MAD_TO_GAUSSIAN_SIGMA)`
- `std`, `ptp`, `mean_abs`, `min`, `max`, `n_values`, `n_finite`,
  `all_finite`

Degeneracy boundary (estimator-level numeric):

| Group | `epsilon_mad_estimator_only` | Unit | Rationale |
| --- | --- | --- | --- |
| `pose_dyn` | `1.0e-4` | rad / sec | bone-local angular velocity; quiescent bones (fingers etc.) on Walk_F can sit near zero |
| `pose_rel` | `1.0e-4` | (local-6d-component) / sec | bone-local rot6d first-difference rate; quiescent bones likewise |

A channel is degenerate iff `mad <= epsilon` (upper-inclusive,
matching the existing Layer B convention at
`tools/run_walk_f_causal_state_probe.py:1367`). A group is
degenerate iff EVERY channel in the group is degenerate.

Note: epsilon values are estimator-level numeric thresholds and are
NOT contract definition. The artifact ALWAYS reports raw MAD / std /
ptp / mean_abs per channel so a reviewer can re-judge the threshold.

Per scaffold v1 §6 + Layer B `_classify_phase_structure_status`
enum at `tools/run_walk_f_causal_state_probe.py:1305`:

- `group_reference_degenerate == True` -> `phase_structure_status =
  "phase_degenerate"`, `layer_c_candidate = False`.
- `group_reference_degenerate == False` AND `group_all_finite == True`
  AND `T_walk_f >= MIN_FRAMES_FOR_PHASE_ESTIMABILITY` ->
  `phase_structure_status = "insufficient_evidence"`,
  `layer_c_candidate = True`.
- Any other case (non-finite, too few frames) ->
  `phase_structure_status = "insufficient_evidence"`,
  `layer_c_candidate = False`.

Layer B.1 NEVER emits `phase_structured`. That requires a
predictive-loss comparison against a phase-agnostic baseline
(scaffold v1 §3.4) and is reserved for a later layer.

## §6 Query Clip Inspection (Walk_F-Projected)

For each query clip and each included group, Layer B.1:

1. Extracts the query's `(T_q, C_group)` channel matrix using the
   same §4 definitions and the query clip's own FPS.
2. Computes per-channel `(query_value - walk_f_median) /
   max(walk_f_robust_std_from_mad, epsilon_mad)` z-projection.
3. Reports per-channel:
   - `walk_f_projected_z_median` (`float(np.median(z_finite))`)
   - `walk_f_projected_z_mad` (`float(np.median(np.abs(z - z_median)))`)
   - `walk_f_projected_z_p05`, `_p50`, `_p95`, `_max_abs`
   - `n_values`, `n_finite`, `all_finite`
4. ALSO reports the same per-channel raw scale (`median`, `mad`,
   `std`, etc.) for the query channel itself, clearly labelled
   `query_self_scale_inspection_only`. These query-self numbers
   MUST NOT be aggregated into the Walk_F reference scale.

Group-level reference_degenerate / phase_structure_status for the
query clip is inherited from Walk_F's group decision (a query clip
cannot promote Walk_F's degeneracy verdict — `pose_dyn` /
`pose_rel` are read-only Walk_F-scale audit only). If Walk_F has
`group_reference_degenerate = True`, the query block reports
`phase_structure_status = "phase_degenerate"` and skips Walk_F
z-projection (would divide by epsilon and produce meaningless huge
z-scores).

If `len(reference_clips) == 0` (defensive — the CLI guard already
rejects this case), the artifact reports
`phase_structure_status = "insufficient_evidence"`,
`layer_c_candidate = False`, and `walk_f_projected_z_*` fields are
`null`.

## §7 Output Fields

Required root summary fields (all required):

- `tool`
- `tool_mode = "read_only"`
- `contract = "walk_f_causal_state_scaffold_v1"`
- `mode = "pose_reference_scale_check"`
- `track = "walk_f_pose_reference_scale_and_degeneracy"`
- `track_role =
  "pose_reference_scale_and_degeneracy_prerequisite_not_membership"`
- `reference_family = ["Walk_F"]`
- `query_family = ["Walk_L_To_L", "Walk_L_To_R", "Walk_R_To_L",
  "Walk_R_To_R"]`
- `clips`
- `raw_root`
- `processed_data_root`
- `processed_data_schema_status` in `{pass, fail}`
- `processed_data_schema_status_reason`
- `definition_layer`
- `estimation_grid` (per-group `epsilon_mad_estimator_only` block;
  `note` stating epsilon is estimator-level)
- `included_feature_groups`
- `excluded_feature_groups`
- `feature_groups_meta` (per-group channel-count, source, unit_label,
  group_ablation_role)
- `reference_scale_source` in `{"reference_clips_only",
  "no_reference_clip_present"}` (the latter is only reachable as
  defense-in-depth; the CLI guard rejects it)
- `reference_clip_names_resolved`
- `query_clip_names`
- `reference_clip_status` and `reference_clip_status_reason`
- `per_group_reference_scale`
  - per group: `channels` (per-channel scale dict), `group_max_channel_mad`,
    `group_reference_degenerate`, `channel_degenerate_count`,
    `channel_total_count`, `phase_structure_status`,
    `phase_structure_status_enum_source`, `layer_c_candidate`,
    `phase_estimable_candidate`, `epsilon_mad_estimator_only`,
    `unit_label`, `source`.
- `per_query_clip_inspection`
  - per query clip × per group: `clip`, `frame_count`, `fps`,
    `npz_path`, `npz_schema_status`, `walk_f_projected_z_summary`
    (per-channel z-statistics), `query_self_scale_inspection_only`
    (per-channel raw scale), `phase_structure_status`,
    `reference_degenerate` (inherited from Walk_F),
    `boundary_detection_status =
    "not_emitted_by_this_tool_layer_b1_reference_scale_only"`.
- `phase_structure_status_per_group`
- `phase_structure_status_enum =
  ["phase_structured", "phase_degenerate", "insufficient_evidence"]`
- `phase_structure_status_enum_source =
  "docs/aperiodic_transition/2026-05-22_walk_f_causal_state_scaffold_v1.md:255"`
- `frame_alignment_per_clip` (per clip: `clip`, `json_frame_count`,
  `npz_bone_ang_vel_T`, `npz_bone_rot6d_T`, `aligned`)
- `insufficient_evidence` (catalogue; see §9)
- `notes`
- `sensitivity_summary` (per-group `epsilon_mad_grid_status =
  "single_point_per_group_layerB1"`; other grids
  `not_emitted_in_this_mode`)
- Explicit non-output markers (see §9).

Per clip artifact JSON (`<clip>__pose_reference_scale_check.json`):

- `clip`
- `raw_json_path`
- `npz_path`
- `frame_count`
- `fps`
- `clip_role` in `{"reference_clip", "query_clip_not_reference"}`
- `contributes_to_reference_scale` (boolean)
- `processed_data_schema`:
  - per array: `key`, `rank`, `shape`, `dtype`, `device = "cpu"`,
    `finite_coverage` (`n_total`, `n_finite`, `all_finite`)
  - meta block: `meta_json_units`, `meta_json_spaces_bone_rotations`,
    `meta_json_spaces_bone_angular_velocities`, `meta_json_fps`,
    `bone_names_count`, `parents_count`
- `frame_alignment`: `json_frame_count`, `npz_bone_ang_vel_T`,
  `npz_bone_rot6d_T`, `aligned`
- `feature_groups_layer_b1`: per group
  - `group`, `channels` (per-channel block), `extracted_matrix_shape`
    (`(T, C_group)`), `extracted_matrix_dtype = "float64"`,
    `extracted_matrix_device = "cpu"`, `group_ablation_role`,
    `source`, `unit_label`, `epsilon_mad_estimator_only`,
    `group_reference_degenerate`, `group_max_channel_mad`,
    `phase_structure_status`, `layer_c_candidate`.
  - For reference clips, `channels` carries Walk_F per-channel
    median / MAD / etc.
  - For query clips, `channels` carries both
    `walk_f_projected_z_summary` and
    `query_self_scale_inspection_only`.

## §8 Shape, Dtype, and Device Conventions

- All numpy arrays consumed by Layer B.1 are cast to `float64`
  before scale computation. Source NPZ arrays remain `float32` on
  disk; reading them at `float32` and casting to `float64` once is
  the only conversion path.
- All Layer B.1 arrays live on CPU. Device is reported explicitly
  in `extracted_matrix_device` and per-array `device = "cpu"`.
- `bone_ang_vel` raw NPZ array shape: `(T, 46, 3)`.
- `bone_rot6d` raw NPZ array shape: `(T, 46, 6)`.
- `pose_dyn` extracted matrix shape: `(T, 138)`.
- `pose_rel` extracted matrix shape: `(T, 276)`.
- `bone_names` shape: `(46,)`, dtype object (Python str).
- `parents` shape: `(46,)`, dtype `int32`.
- `FPS` scalar, dtype `int32`.

## §9 Explicit Non-Outputs (forbidden by contract)

Layer B.1 does NOT emit any of the following. The artifact MUST mark
each one as `not_emitted_by_this_tool` or `forbidden_by_contract`:

- `attractor_membership_status` (always `not_emitted_by_this_tool`)
- `event_head_target_status` (always `not_emitted_by_this_tool`)
- `handoff_ready_status` (always `not_emitted_by_this_tool`)
- `transition_done_status` (always `not_emitted_by_this_tool`)
- `cross_attractor_claim_status` (always `forbidden_by_contract`)
- `transition_truth_promotion` (always `forbidden_by_contract`)
- `combined_membership_score_status` (always
  `forbidden_by_contract`)
- `phase_library_status` (always
  `not_implemented_layerB1_see_phase_library_check`)
- `predictive_loss_status` (always
  `not_implemented_layerB1_see_phase_library_check`)
- `leave_interval`, `return_interval`, `left_censored_leave`,
  `right_censored_return`, `return_to_reference_status` (always
  `not_emitted_by_this_tool_layer_b1_reference_scale_only`)
- any promotion of `phase_structure_status` to `phase_structured`
- any field that consumes `turn_dyn` as combined-membership evidence

Any future memo that wants to add even ONE of these fields MUST
version the contract (new memo with a new date stamp under
`docs/aperiodic_transition/`), per `docs/removal_policy.md` §6.

## §10 PASS / FAIL / INSUFFICIENT_EVIDENCE Rules

- **Contract PASS** (Layer B.1 `layer_b1_contract_status = "pass"`)
  is achievable while every group reports
  `phase_structure_status = "insufficient_evidence"`, as long as:
  - processed-data schema validation passed for every clip;
  - frame alignment passed for every clip;
  - all NPZ arrays are finite;
  - Walk_F reference scale was computed exclusively from Walk_F;
  - no forbidden non-output field was emitted;
  - `expected_insufficient_evidence_is_contract_pass = true` is
    explicitly recorded at the root.
- **Group-level `phase_degenerate`** for `pose_dyn` or `pose_rel`
  is a legitimate Layer B.1 result and does NOT downgrade
  `layer_b1_contract_status`. It does mean the group cannot be
  promoted into a combined-membership claim later.
- **`INSUFFICIENT_EVIDENCE` must NOT be used to mask a schema failure.**
  Schema / frame-alignment / finite / required-clip violations MUST
  exit 2 with no out_dir, as required by §2 / §11.

## §11 Fail-Fast Conditions Summary (each must exit 2, no out_dir)

- `--clips` not set-equal to the 5-clip list;
- duplicate clip name in `--clips`;
- unknown `--mode`;
- raw JSON missing for any clip;
- processed_data NPZ missing for any clip;
- required NPZ key missing (`bone_ang_vel`, `bone_rot6d`,
  `bone_names`, `parents`, `FPS`, `meta_json`);
- NPZ rank / shape / dtype mismatch;
- non-finite NPZ value in any required array;
- `meta_json.units != "meters"`;
- `meta_json.spaces.bone_rotations != "local_6d"`;
- `meta_json.spaces.bone_angular_velocities != "local_rad_per_sec"`;
- NPZ FPS disagrees with raw JSON FPS;
- NPZ frame count disagrees with raw JSON `len(Frames)`;
- bone-count mismatch (`bone_names` len != 46, or
  `bone_ang_vel.shape[1] != 46`, or `bone_rot6d.shape[1] != 46`);
- any normalization helper called without `scale_source="walk_f"`
  (the only allowed value) — `_layer_b1_forbid_query_self_rescale`
  is wired into `_layer_b1_query_channel_walk_f_projected` as a
  load-bearing kwarg validator;
- any code path attempting to fold `turn_dyn` into a
  combined-membership helper.

## §12 Validation

Required validation (manual, no CI hookup yet):

```bash
python3 -m py_compile tools/run_walk_f_causal_state_probe.py

PYTHONPATH=. python3 tools/run_walk_f_causal_state_probe.py \
  --mode pose_reference_scale_check \
  --clips Walk_F,Walk_L_To_L,Walk_L_To_R,Walk_R_To_L,Walk_R_To_R \
  --raw_root raw_data \
  --out_dir debug_output/walk_f_causal_state_scaffold_v1_20260523_layerB1_pose_reference_scale_check
```

Fail-fast smoke tests (each MUST exit 2 and leave no output directory):

- `--clips Walk_F` (missing every query clip)
- `--clips Walk_F,Walk_L_To_L,Walk_L_To_R,Walk_R_To_L` (missing
  `Walk_R_To_R`)
- `--clips Walk_F,Walk_F,Walk_L_To_L,Walk_L_To_R,Walk_R_To_L,Walk_R_To_R`
  (duplicate `Walk_F`)
- `--mode this_mode_does_not_exist ...` (unknown mode)

Regression: `yaw_debug`, `gauge_check`, `reference_scale_check`,
`phase_library_check`, and `query_boundary_check` numbers MUST remain
bit-identical to their committed (or, for `query_boundary_check`,
in-tree HEAD) versions on `feat/walk-f-causal-state-probe`.
