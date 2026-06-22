# Middle-Generator Acceptance Contract

Date: 2026-06-01

Status: **SPEC-ONLY v0**. This document defines the acceptance contract for middle-state
generation in the action-handoff / aperiodic-transition branch. It does not implement a
model, trainer, runtime gate, data pipeline, or experiment.

Sources:
- `debug_output/_tmp_action_handoff_two_frame_dynamics_check_20260601/two_frame_dynamics_check_summary.md`
- `debug_output/_tmp_action_handoff_bone_angvel_bridge_probe_20260601_v1/bone_angvel_bridge_probe_summary.md`
- `debug_output/_tmp_action_handoff_regime_bridge_probe_20260601_v2/regime_bridge_probe_summary.md`
- `debug_output/_tmp_action_handoff_matched_seam_neuron_audit_20260531_default/head_spectrum.md`
- `docs/aperiodic_transition/2026-05-31_action_handoff_inbetween_process_retrospective.md`

## 1. Scope / Non-goals

This document only defines the **middle-generator acceptance contract**: what a generated
middle trajectory must prove before it can be treated as a feasible bridge from Walk_F into a
turn regime.

Non-goals:
- Do not train a model.
- Do not change the production gate, trainer, runtime, rollout path, or checkpoint contract.
- Do not treat this spec as the final implementation.
- Do not reopen the start/end formulation, commanded-yaw path, soft-endpoint reframe, or old
  endpoint search.
- Do not train a yaw predictor. Yaw remains a commanded-yaw / `cond_dir` path.
- Do not add a residual head.
- Do not continue endpoint, yaw, or discriminator instrumentation as the next lever.
- Do not make `bone_angvel` continuity itself the target. It is a regime witness plus a rate
  budget signal, not an objective to flatten.
- Do not package `hidden_pre`, `z`, `plan_z`, representation R2, or hidden collapse as motion
  success.

## 2. Current Position

The start/end formulation has already held up enough to stop searching for another endpoint
definition. The current gap is **middle generation**: how to produce a support-aware,
budget-respecting trajectory distribution between a Walk_F arbitrary phase and a target turn
regime.

The middle is not a Motion Matching hard cut. The matched-pose audit is a microscope: it shows
that pose/contact can be made close at a seam, while the local dynamics can still disagree. In
the two-frame dynamics check, matched pairs have low current-pose distance but large tangent
differences: `bone_angvel` delta RMS is `0.6133`, `0.7197`, and `0.6245` rad/s for the three
groundable turn targets, with root-velocity delta L2 `0.2798`, `0.3618`, and `0.2109`.

The regime bridge probes narrow the issue:
- Representation and hidden-space probes can identify regime-related drivers, but they are not
  success criteria. The regime mapping probe explicitly reports rep mapping as probe-only.
- Direct and lambda paths can look smoother in internal channels while still failing motion
  honesty: the direct-motion table reports `pop_safe = 0.0000` for `direct_full`,
  `lambda_force1`, `lambda_model`, and `main`.
- Ramp/mapping variants can reduce hidden discontinuity, but the reported `motion_safe_v2`
  pass rate remains `0.0000` for ramp and mapping variants. That is a failure of realized
  bridge honesty, not proof that the target regime is unreachable.

Therefore the objective is not to tune a better query/cost or find another endpoint. The
objective is to define and later train/evaluate a **middle trajectory distribution** that can
reach the target regime without exceeding transition-rate, support, root, yaw, and contact
budgets.

## 3. Key Reframe: `bone_angvel` level vs rate

`bone_angvel [138]` must be read in two layers.

**Level.** In a true turn regime, the per-joint angular-velocity level should differ from
Walk_F. That difference is a witness that the body tangent has entered the target regime. Pulling
the final `bone_angvel` level back toward Walk_F would hide the turn dynamics and reward
over-continuity.

**Rate.** The transition from walk-level to turn-level cannot happen in one frame without a
matching support/root/contact bridge. A one-frame jump in `bone_angvel`, root velocity, support
foot state, or contact plan creates the practical risks observed in probes: visible pop,
jump-step, wrong-foot landing, and foot skate.

Contract implication:
- Do not mark a sample as failed merely because `bone_angvel` changes from Walk_F.
- Do require a per-frame rate budget for the change.
- Evaluate target-regime level and transition rate as separate metrics.
- Treat existing seam differences around `0.61-0.72` rad/s as **total matched-seam deltas**,
  not as a per-frame continuous-baseline threshold.

## 4. Middle Generator Contract

This section defines the expected I/O surface for the contract, not a committed architecture.

### Input

- **Start context:** Walk_F arbitrary phase, `ctx` shape `[B,C,281]`, dtype `float32`, device
  determined by the caller/eval loop. The 281 schema is the current action-handoff state schema;
  `bone_angvel` is not assumed to be part of the primary 281 channels.
- **Commanded yaw / `cond_dir` path:** an explicit commanded-yaw or direction signal over the
  horizon, shape to follow the current command path, e.g. `[B,H,1]` yaw-rate command or
  `[B,H,2]` direction command. This is a command, not a predicted yaw target.
- **Soft endpoint / target regime cue:** a target-regime cue sufficient to define the desired
  landing region. It should not collapse into a single hard endpoint frame.
- **Optional support/contact schedule cue:** explicit support phase or left/right contact
  schedule, e.g. contact probabilities/logits `[B,H,2]` or a discrete support-state cue.
- **Optional dynamics witness features:** training/eval-only features such as derived or
  recorded `bone_angvel` shape `[B,H,138]`, dtype `float32`. These features may supervise or
  evaluate regime/rate behavior, but they are not the primary condition in v0.

### Output

- **Middle trajectory:** shape `[B,H,281]` or a future successor schema, dtype `float32`, on the
  caller-selected device.
- **Channels covered by the trajectory contract:** pose, contact, root motion, and yaw/heading
  response according to the active schema.
- **Optional dynamics witness:** derived or explicit `bone_angvel [B,H,138]` for evaluation or
  auxiliary supervision only.

### Conditioning stance

The first version should not use an opaque latent as the main condition. Prefer explicit:
- support phase,
- contact schedule,
- turn-rate or yaw-rate command,
- commanded yaw / `cond_dir`,
- target-regime cue.

`hidden_pre`, `z`, `plan_z`, or similar latent states may be logged for diagnosis. They do not
define acceptance and must not be used as the binding success criterion.

## 5. Acceptance Contract v0

The generated middle passes v0 only if it satisfies all required metric families below. Exact
numeric thresholds are not fixed here; they must be calibrated from continuous Walk_F and turn
baselines before being used as a gate.

### A. Regime reached

Question: does the middle trajectory end inside the target turn regime?

Required evidence:
- realized yaw/root direction follows the target turn regime;
- pose/contact/root state is compatible with the target regime;
- `bone_angvel` level can be used as a witness that the target tangent has been reached.

Non-evidence:
- `hidden_pre`/`z` R2,
- hidden collapse to Walk_F/turn,
- linear readout success,
- representation proximity alone.

These can diagnose why a sample failed, but they cannot certify motion success.

### B. Rate budget

Question: does the path move from walk-level to turn-level within a continuous-motion budget?

Required metrics:
- per-frame `Delta bone_angvel` RMS over `[B,H-1,138]`;
- per-frame `Delta bone_angvel` p95 over `[B,H-1,138]`;
- optional root-velocity and yaw-rate deltas over the same frames.

Acceptance rule:
- Compare RMS and p95 against continuous Walk_F and target-turn baseline bands.
- Use baseline-normalized or percentile-based thresholds, not the matched-seam total delta.
- Existing seam delta RMS values `0.6133`, `0.7197`, and `0.6245` rad/s are evidence that the
  hard seam is too large; they are not the per-frame budget.

### C. Support honesty

Question: does the support/contact story agree with the actual FK foot motion?

Required metrics:
- contact schedule vs FK foot velocity consistency;
- support-foot velocity under declared single-support or dual-contact frames;
- baseline-normalized foot slip or support violation, not raw absolute slip alone.

Reason: the current baseline already has non-trivial foot slip, so a raw absolute threshold can
mislabel both baseline and generated motion. The contract should compare to per-regime
continuous bands, e.g. a ratio or z-score against Walk_F and target-turn support phases.

A sample fails support honesty if it declares a planted foot while FK foot motion exceeds the
baseline-normalized band, or if it changes contact/support state without a plausible transition
phase.

### D. Command response

Question: does realized yaw/root direction follow the commanded-yaw path?

Required metrics:
- heading/yaw correlation or signed alignment with the command;
- heading MAE or integrated yaw deviation;
- root-direction consistency with commanded `cond_dir`.

Constraint:
- yaw remains a command path. Do not train or evaluate this as a yaw predictor success.
- A trajectory with smooth pose but wrong signed yaw response fails.

### E. Pose continuity

Question: does the sample avoid visible pose pop?

Required metrics:
- local pose continuity across adjacent frames, e.g. rot6d-derived geodesic change;
- root translation/velocity continuity;
- seam-local pop check at start and end of the generated middle.

Constraint:
- Pose-L2 is not sufficient. It may confirm that a frame is visually close, but it cannot prove
  tangent, support, or command honesty.

### F. Endpoint bridgeability

Question: does an endpoint candidate admit a budgeted, support-aware path from the current
Walk_F context?

Required evidence:
- the endpoint pose/contact is feasible;
- a support/contact schedule can bridge from the start phase to the endpoint phase;
- the required `bone_angvel`, root, and yaw-rate changes fit continuous-baseline budgets;
- FK foot motion is honest under the proposed support schedule.

This is a feasibility filter over endpoint candidates. It is not a request to redesign the
start/end formulation or retune the endpoint search.

## 6. Anti-patterns / Failure Modes

- **Pose-L2 averaging:** averaging mutually exclusive left/right support branches can create a
  half-step, wrong-foot landing, or backward support phase even when pose distance looks small.
- **MM cost/query regression:** moving complexity back into query weights, transition windows,
  and cost knobs recreates the hard-cut tuning problem instead of solving middle generation.
- **Opaque latent shortcut:** hidden/rep match, readout R2, or hidden collapse improves while
  realized motion still fails yaw, support, or pop checks.
- **Over-continuity:** smoothing `bone_angvel` or root tangent so strongly that the trajectory
  never reaches the turn regime; the turn witness is pulled back to Walk_F.
- **Over-discontinuity:** allowing a one-frame regime switch because the endpoint frame is close;
  this exceeds rate/support budgets and produces jump-step or foot skate.
- **Direct/lambda shortcut:** direct or lambda branches produce a smoother-looking vector while
  contact, support foot, root motion, or yaw response is not honest.
- **Residual-head patching:** adding another correction head to hide the symptom without
  satisfying support-aware rate and command metrics.

## 7. Architecture Implications

The binding design choice is not "which generator family wins"; it is whether the objective and
acceptance gate are support-aware.

If the target is one canonical transition per start/target pair, a masked transformer with
explicit commanded-yaw, support/contact, and rate-budget losses is likely more controllable than
sampling-heavy machinery. It can be evaluated deterministically against the contract above.

If the target requires multiple valid branches, especially left-support vs right-support
solutions from similar pose endpoints, then diffusion or another sampler becomes relevant. That
architecture choice is only justified if the data and contract require multi-solution sampling.

Diffusion does not relax the metric contract. A diffusion sample that wins pose-L2 but violates
support honesty, command response, or rate budget still fails.

## 8. Minimal Next Validation

Do not train a middle generator yet. The next minimal action should be metric-only:

1. Define an eval-only scaffold that computes the v0 metrics on existing trajectories and
   existing probe outputs. This scaffold should not modify production trainer/runtime/gate.
2. Use existing continuous Walk_F and turn artifacts to calibrate baseline bands for:
   `Delta bone_angvel` RMS/p95, root-velocity delta, yaw-rate delta, support-foot velocity, and
   baseline-normalized foot slip.
3. Re-score existing matched, mapping, and ramp probes under the v0 acceptance contract.
4. Report per-target and per-start-phase rows; do not average away `Walk_L_To_R` or support-side
   failures.
5. Decide whether to train a middle generator only after the metric-only replay shows which
   failure family is binding: regime not reached, rate over-budget, support dishonest, command
   mismatch, pose pop, or endpoint not bridgeable.

This validation is a probe/eval plan, not an implementation request in this document.

## 9. Open Questions

- Is the support schedule an input cue, a generated output, or both?
- Is `bone_angvel` an explicit output, an auxiliary loss target, or only derived from pose
  finite differences for evaluation?
- How should endpoint bridgeability be calibrated across Walk_F phases and turn targets?
- Does multimodal support-foot branching require diffusion, or can an explicit support cue make
  a deterministic generator sufficient?
- Should the 281 schema be upgraded, or should dynamics witnesses such as `bone_angvel [138]`
  remain eval/auxiliary-only features outside the primary trajectory schema?
