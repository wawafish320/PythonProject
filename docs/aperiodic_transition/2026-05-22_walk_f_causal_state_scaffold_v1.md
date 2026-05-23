# Walk_F Causal-State Attractor Scaffold v1 (2026-05-22)

> 本 memo 是 **CONTRACT drafting / research-audit planning**。不写 Python 代码、不新增 probe、不跑训练、不新增 EventHead / `arbiter_head` 模块、不改 checkpoint slot / fingerprint / CLI / config / freeze policy、不 commit。本文只为后续 v1 probe 定义 semantic scaffold，避免把 yaw-rate / energy / Walk_F template 误写成通用 transition truth。

## §0 Status & Inputs

- **身份**：Walk_F causal-state attractor scaffold v1。它接在 step3 reopen memo 之后：`touchdown_any_foot` 已降级为 gait diagnostic / auxiliary tag，不能作为 transition EventHead 主 supervision；见 `docs/aperiodic_transition/2026-05-22_step3_reopen_and_v1_transition_pivot_memo.md:15` 到 `docs/aperiodic_transition/2026-05-22_step3_reopen_and_v1_transition_pivot_memo.md:18`。
- **当前 customer**：v1 research/audit only。本文不恢复 EventHead v0，不定义 `handoff_ready`，不授权 step7 slot reservation。
- **理论来源**：causal state / predictive equivalence principle。Two histories are equivalent only if they induce the same future distribution under the same observation/control context. We take the principle; we do **not** import CSSR / epsilon-machine implementation as a required algorithm.
- **Reference family in this scaffold**：`reference_family = {Walk_F}` only. It is a single-class Walk-family scaffold, not a general motion-transition claim.
- **Fail-fast policy**：any future schema / ckpt / CLI surface that consumes this scaffold must explicitly version the contract. Silent fallback or defaulting old fields into new semantics is forbidden by `docs/removal_policy.md:42` to `docs/removal_policy.md:45` and `docs/removal_policy.md:104` to `docs/removal_policy.md:113`.
- **Relation to prior docs**：This memo does not supersede any prior `docs/aperiodic_transition/` memo. It introduces an additional v1 research-audit scaffold layered on top of step3 reopen and Arbiter v0 lock. Future memos re-using the term "attractor" in a different sense must cite this scaffold's definition or explicitly version off it.

## §1 Theoretical Commitment

### §1.1 Attractor Definition

For this v1 scaffold, an attractor is not a numeric energy basin and not a raw-pose template.

**CONTRACT definition**:

> An attractor is a recurrent class of causal states. A causal state is an equivalence class of histories that, modulo global planar translation and global yaw, induce the same future-motion distribution under the same action/control context. The class is recurrent if the system, once it has entered the class, keeps returning to it within the same context.

For current data:

- `Walk_F` is treated as a candidate recurrent predictive class for `EAction::Walk`.
- The 4 turn clips are treated only as candidate excursions from / back to that class.
- A return-to-attractor claim means: a query history can be mapped to a `Walk_F` recurrent phase such that its future prediction error is within the `Walk_F` self-baseline band for the relevant feature groups and estimator settings.

This is **not** a statement that the motion should hand off to Walk_F, nor that Walk_F is the only downstream loop in the runtime system.

For v1 data sufficiency this definition is theoretical, not estimable from current data as a class-level attractor; see §4 for which sub-claims are probeable on the current single-trajectory `Walk_F`.

### §1.2 Comparison Space

Comparison is defined in a planar translation / global-yaw quotient space.

Locked semantic constraints:

- Absolute world position is not part of the attractor state.
- Absolute `RootYaw` is not part of the attractor state.
- Root velocity / displacement features must be represented in a canonical yaw quotient; the canonical gauge must be specified by the probe.
- Yaw-rate, yaw acceleration, relative yaw over a window, contact signals, and pose dynamics may be feature groups because they describe dynamics, not the absolute heading coordinate.
- Any future probe must include a yaw-invariance sanity check. A synthetic global-yaw rotation should not materially change the membership curve under the canonical quotient. The rotation grid must cover at least `{pi/6, pi/2, pi}` and the `+/-pi` wrap boundary. Failure at the wrap boundary is contract failure, not estimator noise. If this fails, mark the probe `INSUFFICIENT_EVIDENCE`.

The current raw JSON supports this distinction: `RootYaw` is exported and measured in radians in the raw metadata, see `raw_data/Walk_L_To_R.json:29` to `raw_data/Walk_L_To_R.json:31`; the coordinate axes are explicitly declared in `raw_data/Walk_L_To_R.json:10` to `raw_data/Walk_L_To_R.json:14`.

### §1.3 What Is Locked vs Not Locked

Locked by this memo:

- causal-state / predictive-equivalence framing;
- `Walk_F` as the only current reference family;
- SE(2)-style planar translation / global-yaw quotient semantics;
- audit labels only: no EventHead target, no training label, no slot;
- result form = intervals + stability + insufficiency markers, not single magic frames.

Not locked by this memo:

- history length;
- future horizon;
- feature weights;
- exact predictive distance metric;
- baseline percentile band;
- phase bin count;
- start/end threshold;
- neural vs non-neural estimator.

All non-locked numbers are estimator-level choices and must be reported as sensitivity axes.

## §2 Scope and Non-Goals

This scaffold explicitly does **not** claim:

- cross-attractor transition validity, e.g. Walk -> Run, Walk -> Idle, Walk -> HitReact, Walk -> Knockback, Walk -> Attack;
- control-conditioned interrupt policy;
- `handoff_ready`;
- `transition_done`;
- stride-cycle invariant learning;
- EventHead target schema;
- Arbiter input schema;
- checkpoint/fingerprint slot semantics;
- production runtime switching behavior.

The only claim future probes may attempt on current data is:

> The single `Walk_F` trajectory exhibits enough internally predictive phase structure for a candidate Walk-family causal-state reference, and the 4 turn clips can be evaluated as possible leave/return excursions relative to that reference.

Any stronger wording must be marked `INSUFFICIENT_EVIDENCE`.

## §3 Definition vs Estimation Split

### §3.1 Definition Layer

The definition layer is numeric-free:

- causal state = predictive equivalence class;
- attractor = recurrent causal-state class under a given action/control context;
- current reference = `Walk_F`;
- membership evidence = predictive return to the reference recurrent class;
- output = audit-only curves / intervals / stability, not labels for training.

### §3.2 Estimation Layer

The estimation layer necessarily has parameters. They are not contract truth.

Future probe artifacts must persist:

| Estimator choice | Required reporting |
| --- | --- |
| history window | grid values; sensitivity of leave/return intervals |
| future horizon | grid values; sensitivity of leave/return intervals |
| feature groups | per-group loss curves and ablations |
| distance metric | exact formula / implementation owner |
| phase lookup | nearest-neighbor / predictive-loss matching rule; excluded self-neighborhood for Walk_F baseline |
| baseline band | quantiles or robust interval; no fixed absolute threshold |
| smoothing / latching | if used, grid values and interval stability |

Rule: a probe setting can support a conclusion only if nearby settings give the same qualitative result. If the boundary moves substantially across reasonable estimator settings, the result is `INSUFFICIENT_EVIDENCE`.

### §3.3 Degenerate-on-Reference Feature Groups

Feature groups can be degenerate on `Walk_F`. This must be handled explicitly.

Example from current read-only inspection: Walk_F has `RootYaw` and yaw-rate essentially flat, while the 4 turn clips have a clean yaw-rate burst. A yaw-rate group may dominate a Walk_F-normalized loss if normalized by a zero-variance reference baseline.

Required handling:

- report each feature group separately before any combined score;
- mark reference variance / robust scale per group;
- if a group has near-zero reference scale, do **not** convert it into unbounded z-score evidence;
- either report it as a descriptive debug group, exclude it from combined attractor membership, or normalize with an explicitly declared non-reference scale;
- any combined score must include group ablation showing whether the conclusion is driven by a degenerate group.

This is the main guardrail against replacing "attractor" with a hidden yaw-rate threshold.

### §3.4 Phase-Structured Is Feature-Conditional

`Walk_F` being phase-structured is not a precondition that can be assumed globally. It is a probe output.

Definition:

> `Walk_F` is phase-structured for a feature group if that group exhibits non-degenerate recurrent variation and supports future prediction better than a phase-agnostic baseline.

Consequences:

- If yaw features are constant on `Walk_F`, yaw does not establish Walk_F phase.
- Contact, root-body velocity, or pose-dynamics groups may establish phase if they are non-degenerate.
- The probe must report which groups, if any, provide usable phase structure.

## §4 Data Sufficiency Caveats

Current data has strong limitations:

- `Walk_F` is one trajectory with 88 frames. At 60fps this is about 1.47 seconds. The existing feasibility inventory records the same clip count and FPS in `docs/aperiodic_transition/2026-05-21_teacher_tag_v0_feasibility_inventory.md:19`.
- One `Walk_F` trajectory is **not** class-level recurrence evidence. It can only test single-trajectory self-similarity.
- Any `Walk_F` leave-one-phase baseline built from this data is ill-conditioned: it estimates intra-trajectory consistency, not between-take recurrence.
- Current 4 turn clips can be evaluated as excursions relative to this single trajectory, but success would mean only "embeds into this self-consistent Walk_F trajectory", not "belongs to a learned walk attractor class".
- If future data adds multiple Walk_F takes / speeds / starts, this caveat can be relaxed without changing the theoretical definition in §1.

Therefore, on current data:

| Question | Current evidence status |
| --- | --- |
| Is `Walk_F` a class-level attractor? | `INSUFFICIENT_EVIDENCE` |
| Is `Walk_F` internally phase-self-consistent under selected feature groups? | probeable |
| Do turn clips leave and return to the same single-trajectory predictive structure? | probeable |
| Does this generalize to Run / Idle / Hit / Attack / Knockback? | `INSUFFICIENT_EVIDENCE` |
| Does this define handoff or interrupt policy? | `INSUFFICIENT_EVIDENCE` |
| Is action/control conditioning empirically meaningful in v1? | NO — only one context is present; conditioning becomes meaningful only when cross-action / control-trace data is added |

## §5 Probe Design Contract

No probe is implemented by this memo. A future read-only probe may be designed as follows.

### §5.1 Dual-Track Output

The future artifact should separate two tracks:

| Track | Role | Contract status |
| --- | --- | --- |
| `yaw_activity_debug` | Walk-turn descriptive oracle; reports yaw S-curve / yaw-rate burst, start/end candidates, left-censoring | debug only; not attractor definition |
| `walk_f_causal_state_scaffold` | Predictive-equivalence membership relative to `Walk_F` reference family | v1 research audit only |

This separation is mandatory. A clean yaw-rate curve must not be promoted to general transition truth.

### §5.2 Candidate Feature Groups

Feature groups should be reported separately:

- `root_body`: root velocity / displacement in the canonical yaw quotient, plus acceleration if used;
- `turn_dyn`: yaw-rate, yaw acceleration, relative yaw over the window;
- `contact`: `FootEvidence.{L,R}.soft_contact_score`, contact transitions, optional stance fractions;
- `pose_dyn`: bone angular velocity summaries or lower-body summaries;
- `pose_rel`: optional relative pose delta summaries, if raw-pose templating is explicitly avoided.

The probe must not hide these groups inside a single scalar without per-group curves and ablations.

### §5.3 Baseline and Query Logic

Reference setup:

- Build `Walk_F` phase candidates from reference windows.
- For each phase candidate, record a history window `H_F(p)` and future window `Y_F(p)` per feature group.
- Build a leave-one-phase baseline on `Walk_F` by excluding local neighboring frames before phase matching.

Query setup:

- For query clip frame `t`, construct `H_q(t)` in the same quotient space.
- Match to one or more candidate `Walk_F` phases using the declared estimator.
- Predict `Y_q(t)` with the selected `Y_F(p*)`.
- Report loss / percentile / phase confidence per feature group.

Boundary output:

- `leave_interval`: interval where query exits the `Walk_F` self-baseline band, if observable;
- `return_interval`: interval where query re-enters the `Walk_F` self-baseline band, if observable;
- `left_censored_leave`: true when the clip starts after the leave event has already occurred;
- `right_censored_return`: true when the clip ends before return can be established;
- `boundary_stability`: sensitivity across estimator grid.

### §5.4 Censoring Is Not Failure

Short turn clips may start inside the turn. In that case, `leave_interval = null` or touches frame 0 and `left_censored_leave = true` is a valid result, not a probe failure.

Similarly, if a clip ends before enough post-return frames exist, the probe must mark `right_censored_return = true` rather than invent a return frame.

## §6 Required Artifact Fields

Future artifact root fields:

- `tool`
- `tool_mode = "read_only"`
- `contract = "walk_f_causal_state_scaffold_v1"`
- `reference_family = ["Walk_F"]`
- `scope = "single_trajectory_walk_family_probe"`
- `raw_root`
- `clips`
- `definition_layer`
- `estimation_grid`
- `feature_groups`
- `quotient_definition`
- `yaw_invariance_sanity`
- `walk_f_baseline`
- `per_clip`
- `sensitivity_summary`
- `insufficient_evidence`

For each feature group:

- `reference_scale`
- `reference_degenerate`
- `baseline_loss_quantiles`
- `phase_structure_score`
- `phase_structure_status` in `{phase_structured, phase_degenerate, insufficient_evidence}`
- `loss_curve`
- `loss_percentile_curve`
- `phase_hat_curve`
- `phase_confidence_gap`
- `group_ablation_role`

For each clip:

- `leave_interval`
- `return_interval`
- `left_censored_leave`
- `right_censored_return`
- `return_to_reference_status`
- `debug_yaw_activity_window`
- `notes`

All frame-index outputs are artifact evidence, not training labels.

## §7 Relation to EventHead / Arbiter

- EventHead remains paused in v0. This scaffold does not re-enable `event_progress_hat`, `event_duration_hat`, or `event_class_hat`.
- This scaffold does not define `event_head` checkpoint slot semantics.
- Arbiter v0 remains motion-only and G-A2=NO for EventHead input, as locked in `docs/aperiodic_transition/2026-05-21_arbiter_v0_spec_lock.md:40` to `docs/aperiodic_transition/2026-05-21_arbiter_v0_spec_lock.md:49`.
- A future `PredictiveStateHead` or renamed EventHead would require a separate v1 implementation contract with explicit consumer, target/loss, data sufficiency, and checkpoint contract.

## §8 Validation

Required validation for this memo:

```bash
python3 tools/doc_lint_lambda_fusion_naming.py docs
```

No `py_compile` is required by this memo because no Python source is modified.

No training, probe implementation, teacher pipeline write, raw_data write, checkpoint write, or commit is part of this memo.
