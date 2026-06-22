> TRIAGE: DOWNGRADE-SUPERSEDED. Historical design/probe/planning note only; do not treat as live action-handoff delivery plan. Superseded by `2026-06-07_action_handoff_inbetween_closeout_decision_record.md` §5/§6/§8 under its stated read-only / zero-new-injection scope.

# Inbetween Modeling Audit and Representation Spec

Date: 2026-06-02

Status: spec-only / read-only synthesis. No model training, no production runtime/trainer/gate
forward, no checkpoint mutation, no residual head, no endpoint/yaw/discriminator continuation.

Sources:
- `docs/aperiodic_transition/2026-05-31_action_handoff_inbetween_process_retrospective.md`
- `docs/aperiodic_transition/2026-06-01_middle_generator_acceptance_contract.md`
- `debug_output/_tmp_action_handoff_support_topology_granularity_coverage_audit_20260602/support_topology_granularity_coverage_summary.md`
- `debug_output/_tmp_action_handoff_oracle_schedule_trajectory_decoder_smoke_20260602/oracle_schedule_trajectory_decoder_smoke_summary.md`
- `debug_output/_tmp_action_handoff_support_topology_learner_condition_ablation_20260602/support_topology_learner_condition_ablation_summary.md`
- `debug_output/_tmp_action_handoff_support_contract_tightening_20260602/support_contract_tightening_summary.md`
- `debug_output/_tmp_action_handoff_middle_acceptance_replay_probe_20260601/middle_acceptance_replay_summary.md`
- `debug_output/_tmp_action_handoff_regime_bridge_probe_20260601_v2/regime_bridge_probe_summary.md`
- `debug_output/_tmp_action_handoff_bone_angvel_bridge_probe_20260601_v1/bone_angvel_bridge_probe_summary.md`

## 1. Scope / Non-goals

This document is a formulation and representation spec for action-handoff inbetween modeling.
It does not train a generator, attach a runtime path, edit a trainer, change a gate, or mutate a
checkpoint.

Non-goals:
- Do not change the start/end formulation or the F4 commanded-yaw checkpoint line.
- Do not reopen endpoint search, yaw plumbing, or discriminator instrumentation.
- Do not re-prove yaw. Yaw/`cond_dir` stays a commanded cue, not a predicted success target.
- Do not build a residual seam patch or residual head.
- Do not treat current flat decoder failure as an architecture conclusion.
- Do not assert support-foot-anchor, FK-aware loss, deterministic decoding, or diffusion as the
  correct answer before targeted validation.
- Do not package `hidden_pre`, `z`, `plan_z`, representation R2, or hidden collapse as motion
  success.

Tensor contract used in this spec:
- Start context: `ctx [B,C,281]`, `float32`, caller-selected device.
- Command cue: yaw-rate command `[B,H,1]` or `cond_dir/yaw [B,H,2..3]`, `float32`,
  caller-selected device. It is commanded input only.
- Optional support schedule: discrete support tokens `[B,H]`, `int64`, or contact/support
  probabilities `[B,H,2]`, `float32`, caller-selected device.
- Middle output: state trajectory `[B,H,281]`, `float32`, caller-selected device.
- Optional dynamics witness: `bone_angvel [B,H,138]`, `float32`, same device as the generated
  trajectory when evaluated or used as an auxiliary signal.

## 2. Problem Reframe

The inbetween problem is not a Motion Matching hard cut. A hard cut chooses a seam frame; the
current evidence shows that pose/contact proximity at a seam can still leave incompatible
tangent, root, support, and rate behavior.

The inbetween problem is not pose interpolation. Linear or proxy pose/contact interpolation can
be visually close in part of state space while failing command response, support honesty, and
rate budget.

The inbetween problem is: under geometry-compatible but dynamics/regime-inconsistent
conditions, generate a physically valid middle trajectory from a Walk_F start context into a
target turn regime.

Definitions:
- Consistent geometry: start pose, contact, endpoint region, and target-resume region are
  compatible enough that a bridge candidate is not rejected by pose/contact/endpoint geometry.
- Inconsistent dynamics: `bone_angvel`, root velocity, support-foot behavior, tangent direction,
  or regime-level motion statistics differ between the Walk_F start and target turn regime.
- Valid bridge: the generated middle satisfies support/FK/root/command/rate acceptance, not just
  pose distance. Required checks include support honesty, support side correctness,
  FK-derived foot slip, root/yaw command response, pose continuity, and rate budget.

The phrase "inconsistent but consistent" means: geometry can be close enough for a bridge to
exist, while dynamics are not close enough for a one-frame seam or flat interpolation to be
valid.

## 3. Evidence Ledger

| artifact path | key numbers | supports what | does not prove what |
|---|---|---|---|
| `debug_output/_tmp_action_handoff_middle_acceptance_replay_probe_20260601/middle_acceptance_replay_summary.md` | real continuous pass rate `0.9898`; matched hard seam `0.0000`; one-frame angvel/root switch `0.0000`; linear pose/contact proxy `0.0000`; direct/lambda/main negative controls `0.0000`; `Walk_L_To_R` `contact_d=0.7031` | The acceptance gate distinguishes continuous real motion from seam/proxy shortcuts; middle must satisfy motion families, not just endpoint pose | Does not prove a specific generator family; does not prove diffusion is needed |
| `debug_output/_tmp_action_handoff_regime_bridge_probe_20260601_v2/regime_bridge_probe_summary.md` | direct_full/lambda_force1/lambda_model/main `pop_safe=0.0000`; foot slip mean `0.7183..0.8083` m/s; velocity budget angvel deltas `0.6133`, `0.7197`, `0.6245` rad/s | Direct/lambda state replacement and hard seams do not create honest bridges; dynamics difference is real | Does not prove the target regime is unreachable; does not prove a residual patch should be added |
| `debug_output/_tmp_action_handoff_bone_angvel_bridge_probe_20260601_v1/bone_angvel_bridge_probe_summary.md` | input `state [1,40,419]`, `float32`, `cpu`; `bone_angvel [1,40,138]`, `float32`, `cpu`; canonical state281 contains bone_angvel `False`; k3/k4 ramp hidden collapse mean `0.8224`; best ramp motion_safe_v2 `0.0000`; best mapping motion_safe_v2 `0.0000` | `bone_angvel` is a regime witness and rate signal; changing it affects regime representation | Does not make `bone_angvel` a schema field; does not say smoothing it to Walk_F is desirable; does not prove bridge success |
| `debug_output/_tmp_action_handoff_support_contract_tightening_20260602/support_contract_tightening_summary.md` | wrong_side pass rate `0.0000`; wrong_side rejected by support_side_correctness `1.0000`; wrong_side rejected by endpoint_bridgeability `1.0000`; real continuous false positive rates `0.0000`; available_context normalized multi-signature fraction `0.2917` | Support side and endpoint bridgeability are binding; wrong-side support can be detected | The multimodality table only justifies retaining a sampling branch as deferred optionality, not requiring diffusion |
| `debug_output/_tmp_action_handoff_support_topology_granularity_coverage_audit_20260602/support_topology_granularity_coverage_summary.md` | `16` split-topology rows all `granularity_fragment`; `left_domain_coverage_gap=0`; `true_new_support_mode=0`; unique unseen topologies `12`; topology granularity change allowed now `False` | Current topology issue is granularity/coverage fragmentation; no evidence for a true new support mode | Does not license lowering granularity before layer-2 decoder validation |
| `debug_output/_tmp_action_handoff_support_topology_learner_condition_ablation_20260602/support_topology_learner_condition_ablation_summary.md` | true learner train top1 mostly `0.9915..1.0000`; leave-clip/block tests low; unseen topologies up to `22`; decision `data_coverage_insufficient_expand_clips_no_generator`; layer-1 conclusion not `diffusion required` | Learners can fit train but are coverage/granularity-bound under blocked/leave-clip splits | Does not prove current data is enough; does not prove sampling/diffusion is needed |
| `debug_output/_tmp_action_handoff_oracle_schedule_trajectory_decoder_smoke_20260602/oracle_schedule_trajectory_decoder_smoke_summary.md` | matched windows `188`; reconstructed GT train/test accept `1.0000`; oracle_copy_direct test accept `0.9744`; flat decoder split train/test accept `0.0000`; support token accuracy on `Walk_L_To_R` `1.0000`; layer-2 diffusion/sampling evidence none | The guard can reconstruct valid GT; flat state281 decoder fails train acceptance under oracle schedule | Does not prove deterministic decoding is impossible; does not prove diffusion required |
| `docs/aperiodic_transition/2026-05-31_action_handoff_inbetween_process_retrospective.md` | F4 AR wiring `yaw_overwrite_max_abs=0.0`; body sensitivity ordered; masked path has `cmd_yaw [B,H,1]`; W1d held-out grounded clips all fail; `data_or_formulation_license_granted=false` | Start/end/yaw plumbing is not the current blocker; data and formulation remain entangled | Does not prove adding data alone fixes it; does not license another endpoint/yaw/discriminator round |
| `docs/aperiodic_transition/2026-06-01_middle_generator_acceptance_contract.md` | middle output expected as `[B,H,281]` or successor schema, `float32`; optional `bone_angvel [B,H,138]`; acceptance families: regime, rate, support, command, pose continuity, endpoint bridgeability | Acceptance must be motion-level and support-aware from day 1 | Does not bind a concrete architecture |

## 4. Modeling Axes

| axis | observed evidence | required model role | failure mode if ignored | open question |
|---|---|---|---|---|
| A. Geometry compatibility axis | `Walk_L_To_R` pose can be close (`pose_d=0.0162`) while contact is not (`contact_d=0.7031`); groundable targets have pose/contact-compatible seams | Select or condition on endpoint regions that admit a bridge; keep pose/contact feasibility separate from dynamics | A pose-close but contact-invalid endpoint is treated as bridgeable, causing false positives | How much endpoint softness is enough without admitting wrong support phases? |
| B. Regime / dynamics witness axis | `bone_angvel` deltas `0.6133..0.7197` rad/s at matched seams; `bone_angvel` ramp affects hidden collapse but not motion_safe_v2 | Use dynamics witnesses to verify target-regime level and rate budget; avoid over-continuity | The model averages or smooths away turn dynamics, or jumps them in one frame | Should `bone_angvel` be aux supervision only, or derived-only evaluation? |
| C. Support event / timing axis | wrong-side support rejected at `1.0000`; topology rows are granularity fragments; oracle contact pass-through can guard GT | Model support events/timing explicitly enough to bind support side and transition phase | Wrong-foot landing, impossible support switch, or support side mismatch | Can support schedule be predicted reliably after coverage expansion, or should it remain an input layer? |
| D. Root trajectory / foot grounding axis | direct/lambda foot slip means `0.7183..0.8083` m/s; flat decoder foot ratio up to `14.7792`; support honesty fails | Represent root path and support-foot grounding jointly with FK checks | Pose looks acceptable but root translation causes planted-foot skate | Is support-foot-anchored coordinate structure sufficient, or does root need separate spline/event modeling? |
| E. Decoder representation axis | Oracle-schedule flat decoder train/test accept `0.0000` after reconstructed GT guard passes `1.0000` | Choose a representation that can train-fit the local bridge before generalization claims | More loss/capacity hides a representation mismatch without train-fit acceptance | Which lifted state gives the smallest one-window/8-window train-fit ladder? |
| F. Stochasticity axis | support contract reports available_context multi-signature fraction `0.2917`, but learner ablation and decoder smoke say no diffusion evidence | Keep sampling/diffusion as a deferred branch only if deterministic schedule-conditioned residual remains multi-modal after train-fit | Premature sampling masks schedule/representation failure; deterministic-only may miss true support branches | After fixed schedule and train-fit, do residuals remain multi-modal under motion acceptance? |

## 5. Candidate Formulations

| candidate | input/output contract | what it explains | what it fails to explain | generalization risk | relation to current evidence | minimal validation needed | status |
|---|---|---|---|---|---|---|---|
| A. Flat state281 decoder | Input `ctx [B,C,281] float32`, command `[B,H,*] float32`, endpoint `[B,279] float32`, optional schedule tokens; output `state281 [B,H,281] float32` | Simple baseline; directly matches current state schema | Does not encode support-foot grounding or root/local separation; current oracle-schedule train gate fails | High: may learn averaged pose/contact/root fields and foot skate | Flat decoder split train/test accept `0.0000` with GT guard `1.0000` | One-window and 8-window overfit ladder to separate objective bug from representation failure | weak |
| B. State281 + FK/support-aware loss decoder | Same output `state281 [B,H,281]`; add FK foot slip, support_side_correctness, command, rate losses in debug-only smoke | Tests whether the issue is missing loss terms around existing representation | Still asks flat state to express anchored/root-coupled structure implicitly | Medium/high: can improve metrics locally but may not fix representation conditioning | Current smoke has support/FK failures; FK-aware objective is plausible but not proven | Train-fit ladder under fixed oracle schedule; require train accept > GT-calibrated threshold before leave-clip claims | plausible |
| C. Support-foot-anchored representation | Input schedule tokens/contact plus command and endpoint; output local pose plus root trajectory expressed relative to declared support foot, then reconstruct `state281 [B,H,281]` | Explains no-slip structure and support side binding | Does not by itself solve timing prediction or regime dynamics | Medium: anchor errors can cause discontinuities at support switches | Support wrong-side rejection and foot slip evidence motivate testing it, but do not prove it | Toy smoke on matched windows with fixed schedule: compare foot slip/support honesty against flat state | plausible |
| D. Phase-conditioned deterministic decoder | Input start context, commanded cue, endpoint region, phase/support features; output `state281 [B,H,281]` or lifted local dynamics | Explains phase-dependent local dynamics and commanded turn response | May fail when support event sequence changes within horizon | Medium: phase features can leak or overfit small clips | Learner ablation shows context/endpoint/command tiers help seen top3 but blocked splits remain low | Event/phase toy overfit under held support schedule; check command_response and rate budget | plausible |
| E. Event-segment / piecewise-smooth decoder | Input support event schedule with segment boundaries and command; output per-segment local pose/root curve reconstructed to state281 | Explains abrupt contact events as transitions between smooth segments | Requires reliable event topology and segment boundary labels | Medium: over-fragmentation can memorize topology fragments | Topology audit says all unseen rows are `granularity_fragment`, so event granularity is central | Piecewise toy smoke: fixed GT events, evaluate if abruptness can be expressed without support violations | plausible |
| F. Decoupled latent transition representation | Input explicit schedule/command/endpoint plus compact latent for residual style; output lifted root/local pose then state281 | Can model residual style without making hidden proximity a success criterion | Latent can become another proxy unless acceptance is motion-level | Medium/high with 188 windows; latent may memorize clips | Retrospective warns hidden/z proxy was self-written and radius-gamed | One-window/8-window train-fit with latent disabled/enabled; acceptance must be independent of latent metric | deferred |
| G. Diffusion / sampling trajectory decoder | Input same explicit schedule/command/endpoint; sample trajectory distribution over lifted or state281 representation | Can represent true multiple valid trajectories when same conditions admit different support/root solutions | Does not fix support honesty, FK, command, or train-fit by itself | High until data coverage and deterministic baseline are established | Current artifacts explicitly say no layer-2 diffusion/sampling evidence; support contract only says retain branch | Only after fixed schedule deterministic train-fit passes, test residual multimodality under acceptance | deferred |
| H. Hybrid: schedule layer + deterministic lifted decoder | Layer 1 predicts or selects support/event topology and timing; Layer 2 consumes fixed schedule, command, endpoint, start ctx and decodes lifted root/local pose to `state281 [B,H,281]` | Separates support/timing from local trajectory representation; aligns with coverage/granularity-bound evidence | Still needs validation that lifted decoder can train-fit and generalize | Medium: schedule layer is coverage-bound; lifted layer may still fail | Best matches current evidence: learner train-fit but coverage-bound, oracle schedule flat train-fit fails, support contract binding | First validate Layer 2 on fixed oracle schedule via tiny overfit and lifted toy smokes; defer full schedule learner | preferred |

## 6. Abruptness vs Multimodality

Dynamics abruptness at contact events is not the same thing as multimodality. A support switch
can create a sharp but deterministic change in root tangent, local pose velocity, and
`bone_angvel` level. Treating every abrupt change as stochastic would hide a representation
problem.

When an oracle support schedule is given, the first modeling assumption should be
piecewise-smooth deterministic decoding: within each support/event segment, the local dynamics
should be smooth enough to satisfy rate/FK/root/command acceptance; at event boundaries, the
decoder should allow controlled discontinuity in the correct channels without violating support
or pose continuity.

Diffusion or another sampler becomes evidence-backed only after:
- fixed schedule is supplied or learned with acceptable accuracy;
- a deterministic decoder can train-fit the same schedule and pass acceptance on train windows;
- residual failures under the same fixed schedule show genuine multi-peak alternatives rather
  than support/FK/root/rate violations.

Current evidence does not show diffusion required. It shows a flat representation train-fit
failure under oracle schedule and a coverage/granularity-bound support layer.

## 7. Recommended Representation

Recommended direction: keep the formulation layered and validate lifted deterministic
representations before committing to sampling.

Layer 1: support/event topology + timing.
- Role: represent support side, support switches, flight/unknown fragments, and event timing.
- Current stance: coverage/granularity-bound. The topology audit reports `16`
  `granularity_fragment` rows, `true_new_support_mode=0`, and no permission to lower topology
  granularity now.
- Output contract: schedule tokens `[B,H] int64` or contact/support probabilities `[B,H,2]
  float32`, on caller-selected device.

Layer 2: schedule-conditioned lifted deterministic decoder.
- Input: `ctx [B,C,281] float32`, commanded yaw/`cond_dir` cue `[B,H,*] float32`, endpoint/region
  cue, and support/event schedule from Layer 1 or oracle.
- Output: reconstructable middle `state281 [B,H,281] float32`, plus optional aux/eval
  `bone_angvel [B,H,138] float32`.
- Binding rule: success is measured by motion-level acceptance, not hidden/latent proximity.

Lifted representation options to validate:
- Phase-conditioned local dynamics: represent local pose/root velocity by support phase or event
  segment, then reconstruct to state281.
- Support-foot-anchored coordinates: express root/local body motion relative to the declared
  support foot during support intervals, with explicit switch handling.
- Root trajectory separated from local pose: decode root path/heading separately from local
  pose, then check FK and support consistency.
- `bone_angvel` as aux witness or regime regularizer: evaluate or regularize regime/rate
  behavior, but do not add it as a mandatory primary schema field in v0.

Acceptance from day 1:
- FK foot slip under declared support.
- `support_side_correctness`.
- `command_response` for yaw/root direction.
- `rate_budget` for `bone_angvel`, root, and yaw-rate changes.
- `pose_continuity` at start, event boundaries, and landing.

This recommendation does not claim foot anchoring is correct. It claims the next question is
which lifted deterministic representation can train-fit the bridge under a known schedule.

## 8. Minimal Next Validation

Do not train a full generator in this step. Use tiny, debug-only smokes that do not attach to
production runtime/trainer/gate.

1. Instrumented one-window / 8-window overfit ladder for the current oracle-schedule smoke.
   - What it tests: whether the current objective/flat representation can train-fit at all under
     fixed support schedule and reconstructed-domain acceptance.
   - Required preflight: verify path identity before any overfit run. The decoder output must go
     through the exact same reconstruction + acceptance path that gives reconstructed GT
     acceptance `1.0000`. If this is not true, fix the guard path first; otherwise a failed
     one-window run is not interpretable. If it is true, `state281` MSE near zero must imply
     acceptance pass for that window.
   - Required logging per step: record each loss term value and gradient norm for `foot_vel`,
     `root_pos`, `command`, `pose_step`, flat-state reconstruction, and aux `bone_angvel`.
     The tensors under this ladder remain `state281 [B,H,281] float32` and optional
     `bone_angvel [B,H,138] float32` on the experiment device.
   - Required ablation arm: run an otherwise identical arm with aux `bone_angvel` removed from
     the loss. It consumes a large output budget and is the cheapest direct test of the
     gradient-swamping hypothesis.
   - Pass criteria: reconstructed GT guard remains `1.0000`; one-window train acceptance reaches
     `1.0000`; 8-window train acceptance is nonzero and support_honesty/support_side_correctness
     are not both collapsed; losses are finite for `state281 [B,H,281] float32` and optional
     `bone_angvel [B,H,138] float32`.
   - Fail criteria: train acceptance stays `0.0000` with finite loss and valid GT guard. If
     state MSE goes near zero while acceptance fails, classify as operator mismatch between
     train target and eval reconstruction path. If state MSE plateaus and `foot_vel` loss does
     not decrease, classify as gradient swamping or loss-balance failure.
   - Decision unlocked: objective bug vs flat representation failure vs loss-gradient
     swamping. If the instrumented one-window run cannot train-fit, do not scale capacity or add
     diffusion.

2. Support-foot-anchored representation toy smoke.
   - What it tests: whether anchoring local/root motion to the declared support foot gives a
     structural no-slip benefit under oracle schedule.
   - Pass criteria: on the same matched windows, FK foot slip/support_honesty improves over flat
     state281 while command_response and pose_continuity do not regress below reconstructed GT
     calibrated bands.
   - Fail criteria: foot slip remains comparable to flat decoder, or support-side correctness is
     lost at support switches.
   - Decision unlocked: whether foot anchoring deserves a real lifted decoder experiment.

3. Phase-conditioned / event-segment deterministic toy smoke.
   - What it tests: whether contact-event abruptness can be represented as piecewise-smooth
     deterministic dynamics rather than stochastic residual.
   - Pass criteria: fixed-event segments satisfy rate_budget and support_honesty on train
     windows, with controlled event-boundary changes and no command_response collapse.
   - Fail criteria: event boundaries still require over-budget `bone_angvel`/root/yaw jumps or
     produce FK support violations.
   - Decision unlocked: whether to keep deterministic Layer 2 as the main path before sampling.

## 9. Decision Boundary

Topology granularity can be lowered only when:
- candidate merges pass layer-2 oracle-schedule decoding under realized motion/FK metrics;
- support_side_correctness and endpoint_bridgeability do not regress on wrong-side/shuffled
  controls;
- held-out topology fragments are explained by the coarser class without raising false
  positives.

A deterministic decoder can be committed when:
- fixed-schedule train-fit passes the one-window/8-window ladder;
- reconstructed GT guards remain valid;
- train windows satisfy FK foot slip, support_side_correctness, command_response, rate_budget,
  and pose_continuity;
- deterministic residuals do not show accepted multi-branch alternatives under identical
  schedule/command/endpoints.

Sampling/diffusion can be retained as an active branch when:
- the support/event schedule is fixed or accurately predicted;
- deterministic lifted decoding can train-fit and pass acceptance;
- remaining accepted residual trajectories are multi-modal under the same conditioning rather
  than failures of support, FK, root, command, or rate.

Data must be expanded when:
- topology learner train-fit remains high but blocked/leave-clip tests stay low with high unseen
  topology count;
- `Walk_L_To_R` remains ungroundable (`contact_d > 0.30`) for the unchanged groundability gate;
- layer-1 coverage/granularity blocks schedule prediction before layer-2 can be evaluated fairly.

A route must be stopped when:
- one-window train-fit fails under valid reconstructed GT guard and finite losses;
- motion acceptance improves only through hidden/latent proxy metrics;
- it requires changing production runtime/trainer/gate or checkpoint before a debug-only proof;
- it depends on yaw as a learned target rather than a commanded cue;
- it claims diffusion required before fixed-schedule deterministic train-fit and residual
  multimodality have been tested.

## 10. Final Position

The formulation has moved from endpoint/yaw search to middle representation. Start/end and F4
commanded-yaw plumbing are no longer the main local levers.

The current flat decoder is not the right endpoint of the investigation. Its oracle-schedule
train gate failure is a representation/objective warning, not a final architecture verdict.

The main question is representation choice: how to express support events, root trajectory,
local pose dynamics, FK grounding, commanded response, and rate budget in a bridgeable middle.

Support topology/timing should be handled as a Layer 1 coverage/granularity problem. The Layer 2
decoder should first be tested under fixed schedule.

The recommended next step is to evaluate lifted deterministic representations with tiny
train-fit and structural smokes before discussing full generator training.

Sampling/diffusion remains deferred optionality, not a current requirement.
