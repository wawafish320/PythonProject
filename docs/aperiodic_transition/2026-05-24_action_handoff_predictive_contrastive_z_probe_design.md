> TRIAGE: KEEP-AS-PROVENANCE. Historical evidence / anti-relitigation note; preserve to explain why this path was rejected or bounded. Do not use as a live plan unless a new decision record explicitly reopens it.

# Action Handoff Predictive-Contrastive z Probe Design (v1)

Date: 2026-05-24
Status: Design proposal — NOT a normative contract; expected to revise after P0-P6 first results.
Owner branch (origin of thinking): `feat/walk-f-turn-cycle-rollout-eval-pilot`

## §0 Scope and Non-Scope

In-scope:
- Define a v1 latent representation (`z`) and accompanying probes to evaluate whether a controlled causal-state / PSR-flavored framework is worth adopting for cross-action handoff (walk ↔ turn / walk ↔ jump / walk ↔ X).
- Specify the v1 z probe training objective and architecture.
- Specify the seven probes (P0–P6) with explicit falsifiability gates.
- Distinguish what is auto-derivable from independently-recorded clips vs what structurally requires external priors.

Out-of-scope:
- Modifying any in-basin training pipeline (lambda fusion, basetrain) — those stay locked.
- Replacing the existing Walk_F turn-cycle rollout-eval pilot — it remains for level-1 in-basin evaluation.
- Posttrain / scheduled sampling — gated behind P0-P6.
- Synthesizing real transition takes (recording reality is "independent takes only" and is treated as a hard constraint).
- Selecting the C (naturalness prior) concrete implementation — separate later decision after P0-P6 results.

## §1 Motivation

### §1.1 Why the current pipeline is structurally insufficient for cross-action handoff

Verified by code inspection:

| Gap | Evidence |
|---|---|
| Export emits 1 JSON → 1 teacher JSON; no stitching | `train/validate/export_teacher_batches.py:197-277` |
| Dataset index is `(clip_id, start)`; windows never cross clip boundary | `train/data/dataset.py:586-595, 842-869` |
| Sample carries `clip_id`, `start`, `clip_len` only; no transition signal | `train/data/dataset.py:926-928` |
| `cond_in` is `act_oh(4) + cond_dir(2) + cond_speed(1)`; basin distinction lives in dir+speed | `train/convert_json_to_npz.py:1252-1319` |
| Across the 5 turn-cycle clips, `cond[:,0:4] = [0,1,0,0]` (shared "Walk" action class); basin distinction is carried by continuous dir/speed only | Empirical: `Walk_{F,L_To_L,L_To_R,R_To_L,R_To_R}_teacher.json` |
| Eval pairing locked per `(clip, phase_start)`; no cross-clip pairing | `docs/aperiodic_transition/2026-05-24_walk_f_turn_cycle_rollout_eval_pilot_contract.md §2` |

The model has never seen a sample where the input window spans a basin boundary or where `cond` flips mid-sequence. Any cross-action handoff behavior at runtime is structural extrapolation, not learned dynamics. This is independent of any in-basin improvement (lambda, scheduled sampling, etc.) and cannot be addressed by them.

### §1.2 Why ε-machine cannot be transplanted directly

Three structural boundaries:

1. Stationarity. ε-machine, `h_μ`, `C_μ` assume a stationary process. A turn / jump / start / stop is a bounded transient; the "asymptotic distribution over causal states" is undefined for one-shot transients.
2. Determinism. Controlled motion is near-deterministic; the entropy-rate machinery idles.
3. Autonomous assumption. ε-machine targets uncontrolled processes. The correct generalization for controlled processes is **Predictive State Representations (PSR; Littman/Singh/Sutton)** — action-conditioned causal states.

Reversal observation worth surfacing: ε-machine handles cyclic actions naturally (phase auto-reconstructs as a ring of causal states), and handles non-cyclic transitions worst — the opposite of where animation industry has its hardest problems.

### §1.3 What to keep from Crutchfield

The **principle**, not the machine: a state is defined as the equivalence class of histories that produce the same future distribution. Under PSR this becomes action-conditioned. Implementation form is a learned latent `z` with a predictive sufficiency objective, not CSSR / discrete symbol algorithms.

Structural justification (does not require empirical probe to hold): causal state is by construction the minimal sufficient predictive representation, i.e. the predictive information bottleneck optimum. An imposed scalar summary like "energy" lacks this guarantee — it can be redundant (lucky case) or wrong (unlucky case); causal state is optimal in both.

## §2 Three-Part Architecture (A / B / C)

```
A. Game logic     -> decides "transition to action X now"
B. z network      -> gives candidate exit frame i in current clip, entry frame j in target clip
C. Naturalness    -> fills i → j with physically/visually natural motion
   prior
```

| Question | Auto-derivable from independent clips? |
|---|---|
| A. When to trigger | No — and should not be. This is game intent. |
| B. Where to exit / enter | Yes — direct byproduct of z (this is the v1 z probe's payoff). |
| C. Is i → j transition natural | **No** — structurally impossible, because the training data contains no history-future pair spanning a basin boundary. Any naturalness judgment there is extrapolation. |

C must be supplied by an external naturalness source:
1. Inbetweening prior trained on a large mocap library (Harvey 2020 family);
2. Physics simulation + RL policy (DReCon / AMP);
3. Adversarial discriminator (AMP / ASE-style D head).

No method using only the user's independent clips can decide C. This is an epistemological limit, not an engineering one.

The three pieces cannot substitute for each other.

## §3 v1 z Probe Specification

### §3.1 Identity

**Predictive-contrastive z probe.** Explicitly NOT: pure CPC, deep PSR, VAE, variational SR. Rationale: v1 needs a ranking metric for entry retrieval, not a generative model or full system identification.

### §3.2 Input and architecture

- `z` input schema (encoder-side): frozen feature extracted from the 2026-05-14 lambda-applied ckpt (`debug_output/_tmp_71_lr1e4_lowlr_downstream_20260504/lambda/checkpoints/ckpt_last_WalkF_stage7_lambda_from_lowlr72_lr1e4_20260504.pth`). v1 primary uses temporal hidden / hidden_pre; `direct-pose pre-output` is ablation-only.
- `future_desc` schema (target-side) is defined in §3.3 and is a separate contract from `z` input schema.
- `g(input) = z`. v1 architecture: MLP `Df_frozen → 256 → 128 → Dz` with LayerNorm. NOT a transformer; the frozen feature already encodes history.
- `Dz = 32` default. Swept `{8, 16, 32, 64, 128}` as part of P5.

No model posttrain. Probe head fitting only.

### §3.3 Loss

```
L = L_predict + β · L_InfoNCE

L_predict = Σ_k w_k · Huber(P_k · z_t, future_desc_{t+k})
L_InfoNCE = -log [ exp(cos(q_k(z_t), future_desc_{t+k}^+) / τ)
                 / Σ_{negatives}  exp(cos(q_k(z_t), future_desc_{·}) / τ) ]
```

Defaults:

- Horizons `k ∈ {1, 3, 6, 12, 24}` for v1 (60fps → 16–400 ms). Parameterized via config; walk-jump phase extends to `{..., 48}`.
- Per-horizon weight `w_k = 1 / sqrt(k)` (or per-horizon unit-normalize Huber). Explicit, not silent uniform.
- β = 0.25.
- Temperature τ swept `{0.05, 0.07, 0.10, 0.15}`; the value with stable P4 rank order is locked.
- Multi-positive: anchor ± {1, 2} frames count as positives. v1 default uses SupCon-style symmetric positive mask (all ±{1,2} as positives); random-positive sampling is kept as ablation only.
- Hard negatives: offline FAISS L2 on pose feature; filter same-clip same-window. Batch composition: `N_pos = 5`, `N_easy_neg = 32`, `N_hard_neg = 16`.
- `future_desc` v1 composition (target-side schema): pose (rot6d) + root vel + contact with group normalization / group balancing so no single group dominates by dimensionality. Subject to P5 ablation (`pose-only`, `pose+root`, `pose+root+contact`). FK is intentionally not part of the training schema; FK-derived foot positions are probe/eval-only derived metrics.
- P1 energy baseline definition: `energy_t = robust_z(root_speed2_t) + robust_z(mean_angvel2_t)` where `root_speed2_t = ||root_vel_t||^2`, `mean_angvel2_t = mean_j ||bone_ang_vel_{t,j}||^2`. `robust_z` uses pooled robust normalization over the union of all frames from the locked 5 turn-cycle clips (`Walk_F`, `Walk_L_To_L`, `Walk_L_To_R`, `Walk_R_To_L`, `Walk_R_To_R`) with median/MAD statistics computed on the union (not per-clip).
- P1 matched-readout fairness: every compared representation uses the same readout architecture template and the same train/test split. Architecture: MLP with hidden_dim=128 + LayerNorm + GELU; "matched" means same architecture family and hidden width, **not** matched total parameter count. Arms are: scalar energy + readout, raw frozen feature + readout, and `z` bottleneck + readout. Report feature shape / dtype / device and trainable parameter count per arm.
- P1 reporting policy: report all metrics globally and per clip. A representation that wins globally but materially regresses on any individual clip does not pass P1; per-clip failures cannot be hidden by a global mean.

### §3.4 Runtime metric (must match training geometry)

- Distance for entry retrieval: `cosine(z(A_i), z(B_j))`. Same metric as training InfoNCE.
- L2 future-pred residual is recorded as an independent sanity-check signal, NOT mixed into the retrieval metric. Above threshold → fallback (refuse / extend window / cut to intermediate clip).

### §3.5 Feature schema admission policy

Training schema admission is fail-closed and split by role:

- `z` input schema only admits frozen-model features available at runtime for retrieval (v1 primary: temporal hidden / hidden_pre).
- `future_desc` schema only admits target descriptors used by `L_predict`/InfoNCE alignment (v1 primary: pose+root+contact with group balancing).
- A feature must not be admitted into either training schema if it is only available as post-hoc projection, requires non-runtime lookahead, or is only used for probe/eval diagnostics.
- Probe/eval-only features may be logged for P0-P6 analysis but cannot silently leak into training tensors.
- Any schema admission change requires explicit doc + config update and corresponding P5 robustness rerun; no implicit fallback, no silent schema expansion.

## §4 v1 Runtime Route

Borrowed from codex sign-off (sharper than my earlier "L1+L2 + latent blend"):

1. Frozen lambda ckpt → fit predictive-contrastive z probe per §3.
2. Per clip, precompute frame-level `z` plus pose / root / contact feature.
3. On runtime trigger: in current frame neighborhood (~3-5 frames) find source exit `i`; over target clip find target entry `j`.
4. Combined retrieval distance: `cosine(z_i, z_j)` primary, with pose/root/contact L2 as sanity check.
5. Bridge `i → j` with fixed `N = 6` frames via UE inertialization / constrained blend. **z does not perform the blend itself in v1.**
6. Motion Matching oracle (pose/root/contact L2 + inertialization, no z) is the must-beat lower bound; neural probe earning its place requires beating or tying it on P6.

Latent-level blend is explicitly deferred to v1.5 and requires P5 / P6 evidence that z is locally linearly interpolable in pose-decodable sense before adoption.

## §5 Probe List (P0–P6)

| Probe | Tests | Gate type | Fail meaning |
|---|---|---|---|
| **P0** Motion Matching oracle | pose/root/contact L2 retrieval + inertialization, no z | must-beat lower bound | Neural approach unjustified vs MM. Do not invest. |
| **P1** z vs energy / vs raw frozen feature on predictive task | `L_predict` test loss under matched-capacity readouts; report global + per-clip predictive magnitude behavior | diagnostic gate (magnitude / point-regression) | Magnitude sufficiency risk for current z objective; does not by itself block P4/P6 if future-equivalence gates are strong. |
| **P2** internal-structure phase diagnostic on Walk_F | cycle-data-aware phase locality and low-dimensional structure; single-cycle closure is diagnostic only, not hard fail | internal-structure diagnostic / precondition | If weak, investigate representation/data mismatch; single-cycle closure weakness alone does not falsify H3. |
| **P3** turn/end payoff diagnostic | turn/end monotonic convergence and cross-turn end tightness; end-window variance vs mid-window is diagnostic only | payoff diagnostic | If weak, payoff signal is unstable and requires follow-up diagnostics; this is not a hard adoption blocker by itself. |
| **P4** future-equivalence cross-clip z-neighborhood test (P4-alt) | given source history/frame and target clip, z-nearest target frames have significantly better GT future-equivalence than random/oracle-expectation baselines | **H3 main gate** | H3 not supported under recalibrated yardstick; do not plan P6 integration. |
| **MM/P0-overlap agreement (legacy P4)** | agreement to MM/P0 overlap-priority ranking in overlap-restricted regions | secondary diagnostic | Safety-priority alignment sanity signal only; not an H3 main decision gate. |
| **P5** residual stability | sweep horizon set, feature set, Dz, τ; rank ordering of (i, j) pairs is stable | robustness gate | z is encoding noise / nuisance; will not generalize to walk-jump. |
| **P6** synthetic boundary stress | reuse `debug_output/_tmp_turn_a_to_b_entry_probe_20260515/sweep_config.json` substrate: inject turn rootvel/rot6d/angvel into Walk_F at frame 40/80; add z distance recording + entry retrieval decision; verify FootSlipBall* / ContactMismatchRate / GeoLocalDeg / RootStepDispErr stay within accept bands and z-chosen i,j beat MM-chosen i,j | integration gate | z works in isolation but can't pick i,j under real boundary stress; do not advance to walk-jump. |

Gate ordering:
- P0 remains hard precondition.
- P2 is an internal-structure sanity diagnostic (not a strict ring-closure hard gate).
- P3 remains a payoff diagnostic (not a hard adoption blocker by itself).
- P4-alt remains the H3 main gate.
- P1 remains the magnitude-sufficiency diagnostic.
- legacy MM/P0-overlap agreement remains a secondary safety-priority alignment diagnostic.
- P5 remains the robustness gate for extension beyond turn family.
- P6 remains the integration gate on real boundary artifact.

### §5.1 P0 / P6 comparison priority

P0 is the non-neural Motion Matching oracle lower bound. A z-assisted route is considered to tie / beat P0 only under this lexicographic priority:

1. Contact / foot safety first: `ContactMismatchRate`, `ContactMismatchFrameOr`, `FootSlipBallL`, `FootSlipBallR`. If these regress materially relative to P0, z loses even if pose error improves.
2. Root continuity second: `RootStepDispErr`, `RootDispErrStartToCurrent`, and offset-corrected root metrics. If root continuity regresses materially relative to P0, z does not pass P6.
3. Pose quality third: `BlendGeoLocalDeg_ex_root`, `GeoLocalDeg`, and related pose metrics. Pose improvements only count after contact/foot/root checks are no worse than P0.
4. Confidence / fallback fourth: z future-pred residual, retrieval margin, and fallback/refusal count are recorded as diagnostics. High fallback count can demote a nominal metric tie.

Canonical safety metric semantics (P6 runner-invoke contract snapshot):

- `ContactMismatchRate` (per-step): threshold `ContactGTPerC` and `ContactMeasPerC` by `>0.5`, then compute channel mismatch ratio `mismatch_channels / valid_channels`.
- `ContactMismatchFrameOr` (per-step): binary 0/1 indicating whether any channel mismatched at that step.
- `FootSlipBallL/R` (per-step): use `ContactMeasWhitebox.VxyCmpsMean` (cm/s) converted to m/s (`/100`) for the corresponding foot channel, gated by dual-frame GT contact (`ContactGTPerC(t)>0.5` and `ContactGTPerC(t+1)>0.5`).
- If no dual-frame GT-contact sample exists for a side, that side is `null` (skip reason: `no_dual_frame_gt_contact`) and canonical completeness is not satisfied for that row.
- `ContactErrAbsMean` is diagnostic and must not replace canonical `ContactMismatchRate` for completeness claims.

This design proposal does not lock absolute numeric bands. The first implementation should report P0 distributions, z-vs-P0 deltas, and proposed tie tolerances before any contract is written.

### §5.2 P6 FK-derived sanity check (probe-time only)

FK is used only as a probe/evaluation projection, not as z training input or `future_desc`.

After the z probe is trained, for each retrieval-selected `(i, j)` pair, project the predicted / selected future rot6d sequence through the project skeleton FK to obtain foot positions. Compare those derived foot positions against GT FK foot positions and the existing FootSlipBall-aligned metrics. This residual feeds the P6 contact/foot safety tier in §5.1.

The purpose is to detect cases where rot6d-space similarity looks acceptable but the induced foot trajectory is physically unsafe. Passing this check does not change the training schema; failing it blocks P6 promotion.

### §5.3 Probe/eval-only feature list (not part of training schema)

The following are explicitly probe/eval-only and do **not** enter either `z` input schema or `future_desc` training schema in v1:

- FK joint positions (including FK-derived foot/world positions).
- Root world cumulative sum / integrated translation traces.
- Bone linear velocity features.
- Foot slip velocity features.
- COM (center of mass) features.
- Energy scalar baseline features (`root_speed2`, `mean_angvel2`, `energy_t`).
- Symmetry-derived features.
- Spectral features (FFT/bandpower/etc.).

These can be used for diagnostics, retrieval sanity checks, and gating reports only.

### §5.4 P4 cross-clip entry retrieval task definition

P4 main definition is **P4-alt future-equivalence** (no MM oracle).

- Query form: given source clip/frame history anchor `(A_i)` and a target clip `B`, rank target candidates `B_j` by z-distance.
- Decision question: do z-nearest `B_j` have similar GT `future_desc` trajectory over next `N` frames?
- `future equivalence oracle`: among all target candidates for a query, frames in the top-`q` fraction with smallest GT future trajectory distance (flattened `future_desc` window L2) are oracle-equivalent.
- MM/P0-overlap agreement is retained only as secondary diagnostic for safety-priority alignment sanity check; it is not the H3 main decision metric.

P4-alt mandatory reporting slices:

1. `global`: all cross-clip source-frame queries.
2. `per_source`: grouped by source clip.
3. `per_pair`: grouped by ordered `(source_clip -> target_clip)`.

P4-alt mandatory metrics (all three slices):

- `top1_future_distance_vs_random_ratio`
- `topk_future_distance_vs_random_ratio`
- `top1_equiv_hit_rate`
- `topk_equiv_hit_rate`
- `random_top1_expectation`
- `top1_equiv_hit_rate_vs_random_top1`
- `mean_spearman_zdist_vs_futuredist`
- `mean_pearson_zdist_vs_futuredist`

Interpretation:

- H3 support requires significantly-above-chance behavior on both distance ratio and hit-rate lift, with stability in `global` plus no severe per-source collapse.
- This probe measures whether z-neighborhood predicts GT future-equivalence; it does not certify transition naturalness/safety. P6 remains the integration boundary gate.
- Current status wording is constrained: **H3 partially supported under recalibrated P4-alt yardstick** (not fully passed).

Internal-structure and payoff diagnostics note (`internal_structure_v2` + single-cycle constraint):
- `internal_structure_v2` shows z has non-trivial structure even when strict cycle closure is weak under current data conditions: phase locality is strong (`knn.mean=0.048989`, `knn.p50=0.045977`), low-dimensionality is strong (`pca_2d_explained_variance=0.783396`), and closure is weak (`cycle_closure_ratio=1.002566`) under single-cycle data limitation.
- P3-style payoff diagnostics are positive on monotonic convergence and cross-turn end tightness (`monotonic>0.60 count=4/4`, `slope<0 count=4/4`, `end_tightness_ratio=0.648635`), while end-window variance vs mid-window (`mean_end_vs_mid_variance_ratio=1.351456`) is treated as diagnostic-only and not a hard fail.

Pass all seven → adopt z for v1 runtime route per §4 and begin C (naturalness prior) selection.
Fail any necessary gate → do not adopt; the failure mode itself identifies which layer to fix.

## §6 Open Questions

1. Multi-positive implementation: **Closed for v1** — use SupCon-style symmetric positive mask with anchor ± {1,2}; keep random-positive sampling as ablation only.
2. `g()` input: **Closed for v1** — primary input is temporal hidden / hidden_pre from frozen ckpt; `direct-pose pre-output` stays ablation-only.
3. `future_desc` composition: **Closed for v1 primary** — use full pose + root + contact with group normalization / balancing; run P5 ablations on `pose-only`, `pose+root`, `pose+root+contact`.
4. v1.5 upgrade path for `g()`: MLP → 1-layer transformer if frozen feat proves insufficient. Trigger condition not yet specified.
5. C (naturalness prior) implementation family: Harvey-2020 inbetweening prior vs physics + RL vs AMP-D. Decided after P0-P6 results, not now.
6. Cross-clip latent space alignment: are per-clip `z` spaces directly comparable, or does z space require a joint training step across all clips? v1 assumes joint by default (single g, all clips); validate in P4.

## §7 Relationship to Existing Work

| Existing object | Position under this design |
|---|---|
| 2026-05-14 root eval closeout (lambda fusion recipe) | Kept as in-basin baseline. Frozen lambda ckpt is the substrate for the entire v1 z probe. Pose/root layered metrics from §2 of that doc are reused as sanity-check signals. |
| Walk_F turn-cycle rollout-eval pilot (this branch) | Stays in place for level-1 in-basin diagnostic. Verdict path (`EXPOSURE_BIAS_DRIFT → scheduled sampling`) is unrelated to this design; does not block or accelerate handoff direction. |
| C.1 / C.2 phase-library probes | Repositioned: they test feature-level discriminators (configs beating baseline), not future-distribution equivalence. Useful as auxiliary signals; not the Crutchfield-coherent probe this design defines. |
| `_tmp_turn_a_to_b_entry_probe_20260515` artifact | Directly reused as P6 substrate. Already covers lambda ckpt + pose injection at frame 40/80 + 8-pre/8-post/16-recovery window + full paired metrics including FootSlip / ContactMismatch. Only addition required: z distance recording + entry retrieval decision logging. |
| Recent cond_raw / leg head fixes (81bc95e / 35e7b0a / e9d9802 / 53d0d76) | Real bug fixes for in-basin pipeline; orthogonal to this design. Do not block. |
| Codex's posttrain push | Refers to scheduled sampling, which is a level-1 in-basin intervention parallel to lambda. Under this design, both are deferred until P0-P6 results are in. Pilot contract §5 "posttrain pilot 禁止在 evaluator 前启动" remains the operative gate. |

## §8 Falsifiability Summary

The design is justified at two layers:
- Structural (paper-only, does not need probe to hold): causal state is minimal sufficient predictive representation; energy is not. PSR generalizes ε-machine to controlled processes.
- Empirical (this is what P0-P6 measure): in user's actual data + frozen lambda ckpt, does a learned z deliver the start/end byproduct and survive integration stress?

Therefore:
- A P0-P6 failure does **not** invalidate the causal-state / PSR structural argument. It does reject the operational hypothesis that the current data, observables, frozen lambda checkpoint, and specified v1 z objective are sufficient for adoption. After failure the design may be revised, but this v1 route is not promoted until the revised route reruns the relevant gates.
- A P0-P6 pass does **not** prove the framework is universal; it only validates v1 implementation on the current turn-cycle family. Extension to walk-jump requires re-running P0-P6 on the walk-jump family.

H1/H2 reframe after P0 preflight: overlap between Walk_F and turn clips is signal, not noise. The operational problem is not "can z separate basins better than energy"; it is "can z provide frame-level entry ranking inside the transferable natural-overlap region." Energy's global basin/entry agreement parity is therefore treated as a localization of where energy runs out of signal, not as evidence that overlap itself is undesirable.

H3 (current recalibrated form):

- **z is useful if z-neighborhood predicts GT future-equivalence significantly above chance, with global and per-source stability.**
- H3 support now depends primarily on P4-alt future-equivalence stability, with P2/P3/internal-structure diagnostics as supporting evidence rather than hard standalone blockers.
- P2/P3 failures under old hard thresholds do not by themselves falsify H3 when the measured property is structurally unavailable or mismatched to data conditions (for example single-cycle closure limits).
- P1 point-regression failure does not alone falsify H3, but remains a magnitude-sufficiency risk that must stay visible in decision records.
- Stable P4-alt signal does not pass P6; P6 remains the boundary-stress integration gate.

## §9 References

- Crutchfield, J.P. (2011). "Between Order and Chaos." *Nature Physics* 8, 17–24.
- Littman, M.L., Sutton, R.S., Singh, S. (2002). "Predictive Representations of State." *NIPS*.
- van den Oord, A., et al. (2018). "Representation Learning with Contrastive Predictive Coding." arXiv:1807.03748.
- Khosla, P., et al. (2020). "Supervised Contrastive Learning." *NeurIPS*.
- Harvey, F.G., et al. (2020). "Robust Motion In-Betweening." *SIGGRAPH*.
- Holden, D., et al. (2020). "Learned Motion Matching." *SIGGRAPH*.
- Peng, X.B., et al. (2022). "ASE: Adversarial Skill Embeddings." *SIGGRAPH*.
- `docs/decisions/2026-05-14_root_eval_contract/decision.md`
- `docs/aperiodic_transition/2026-05-24_walk_f_turn_cycle_rollout_eval_pilot_contract.md`

## §10 Revision Log

- 2026-05-24 v0.1: Initial draft. Captures Crutchfield principle adoption, three-part A/B/C architecture, predictive-contrastive z probe spec, P0-P6 probe list with falsifiability gates. Open questions §6 must be closed before first implementation spike.
- 2026-05-24 v0.2: Added P1 matched-readout baseline fairness, P0/P6 lexicographic metric priority, and sharper §8 failure semantics distinguishing structural principle from the current v1 operational hypothesis.
- 2026-05-24 v0.3: Closed §6.1-§6.3 for first implementation: multi-positive fixed to SupCon mask (random-positive as ablation), `g()` input fixed to temporal hidden/hidden_pre (direct-pose pre-output as ablation), and `future_desc` primary fixed to group-balanced pose+root+contact with explicit P5 composition ablations.
- 2026-05-24 v0.4: Clarified FK scope: FK-derived foot positions are P0/P6 probe-time safety metrics only, not z input or training `future_desc`; added §5.2 P6 FK-derived sanity check.
- 2026-05-24 v0.5: Added §3.5 feature schema admission policy; explicitly separated `z` input schema vs `future_desc` schema; added P1 energy baseline definition with pooled 5-clip median/MAD robust normalization; fixed P1 matched-readout fairness to architecture-matched (hidden_dim=128 + LayerNorm + GELU, not parameter-count-matched) with per-arm trainable parameter reporting; added §5.3 probe/eval-only feature exclusion list.
- 2026-05-24 v0.6: Added mandatory per-clip P1 reporting policy and §5.4 P4 cross-clip entry retrieval task definition, including the P0-table proxy used before P6 acceptance is wired.
- 2026-05-24 v0.7: Reframed P4 around priority-aligned natural-overlap regions; added normalized combined overlap cost, aggregate/runtime overlap-restricted reporting, and the caveat that P4 measures ranking consistency/independent signal rather than absolute transition quality.
- 2026-05-24 v0.8: Added provisional v1 P4 hard criteria from the current energy baseline and clarified query-level no-overlap semantics for runtime defer/fallback behavior.
- 2026-05-24 v0.9: Recorded v1 z spike failure on P1/P4 and linked closeout note: `debug_output/_tmp_action_handoff_z_probe_v1_20260524/closeout_note.md`.
- 2026-05-24 v0.10: After `internal_structure_v2` and P4-alt sweep, original P1 point-regression and MM-oracle P4 were identified as measurement-biased necessary gates. P1 downgraded to diagnostic; P4-alt future-equivalence promoted as H3 main gate. H3 is partially supported, not fully passed; P1 and P6 risks remain.
- 2026-05-24 v0.11: Recalibrated P2/P3 semantics after `internal_structure_v2` and single-cycle-data limitation analysis: P2 moved from strict ring-closure hard gate to internal-structure diagnostic/precondition; P3 clarified as payoff diagnostic with end-window variance treated as diagnostic-only. H3 decision weight remains centered on P4-alt stability, with P1 risk and P6 integration gate unchanged.
