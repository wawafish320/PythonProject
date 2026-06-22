> TRIAGE: KEEP-AS-PROVENANCE. Historical evidence / anti-relitigation note; preserve to explain why this path was rejected or bounded. Do not use as a live plan unless a new decision record explicitly reopens it.

# Action Handoff z Attractor-Support Audit Note (A1/A2/A3)

Date: 2026-05-29

Status: research/probe follow-up note (not production sign-off)

Scope: frozen-artifact audit; no retrain; in-basin pipeline locked (design §0).

## 0. Why this audit

P0–P6 closeout (`docs/aperiodic_transition/2026-05-26_action_handoff_z_probe_closeout_decision_record.md`)
left the v1 route as "not production-approved but justified for a next phase under
fallback-aware planning". Before touching any code, three questions that P2–P4 did
**not** answer needed checking, because they decide whether the next phase is
"source-coverage / fallback + transition attraction" vs "attractor / phase-matching
redesign":

- A1. Is contact phase actually encoded in `z` (or does the attractor only pull pose)?
- A2. For the weak rows, is the **source** `z` far from the target attractor (coverage /
  fallback problem), or is the **target anchor itself** diffuse (attractor-definition
  problem)?
- A3. Can turn-end phase be mapped back to a Walk_F re-entry phase reliably — the runtime
  "convergence → read target phase → map to Walk_F phase → authored blend" step?

Tool: `tools/run_action_handoff_z_attractor_support_audit.py`
Artifact: `debug_output/_tmp_action_handoff_z_attractor_support_audit_20260529/`
Input: `debug_output/_tmp_action_handoff_z_probe_v1_20260524/z_features_per_clip.npz`
(frozen z v1; `future_desc` layout = pose[0:276] + root_vel[276:278] + contact[278:280],
contact channel0=right / channel1=left).

## 1. A1 — Contact phase IS encoded in z

| metric | value | reading |
|---|---|---|
| contact decodability test R² (z, dim 32) | **0.8398** | strong |
| contact decodability test R² (hidden_pre, dim 512) | 0.7964 | z ≥ hidden_pre |
| contact decodability test R² (energy, dim 1) | −0.0482 | scalar can't |
| kNN contact-state purity in z | **0.8286** vs chance 0.3792 (lift +0.4494) | strong |
| z-step at contact transition vs non-transition | 0.0056 vs 0.0029 (lift 1.90×) | z moves at contact events |
| per-clip z→contact nRMSE | F=0.43, L_L=0.18, L_R=0.13, R_L=0.37, R_R=0.51 | all < 1, no per-clip hiding |

Interpretation: **`contact_phase_encoded`**. The 32-dim bottleneck retains contact/foot
regime at least as well as the 512-dim hidden feature, and `z` trajectory speed spikes at
contact transitions. This removes the conversation's worry that the attractor "only pulls
pose, not foot/contact regime". Notably Walk_L_To_R (the weak source) has the **lowest**
per-clip contact nRMSE (0.13) — its contact is well-encoded; its weakness is not a contact
representation failure.

## 2. A2 — Weak rows are source-off-support; target anchors are well-defined

Cosine geometry (matches runtime retrieval, design §3.4). Anchor = target end-window
centroid; diffuseness = anchor_radius / clip_spread.

| target anchor | radius_cos | clip_spread | diffuseness |
|---|---|---|---|
| Walk_L_To_L | 0.0145 | 0.0744 | 0.196 |
| Walk_L_To_R | 0.0355 | 0.1229 | 0.289 |
| Walk_R_To_L | 0.0205 | 0.3325 | 0.062 |
| Walk_R_To_R | 0.0099 | 0.2806 | 0.035 |

All four anchors are **well-defined** (diffuseness 0.04–0.29, all far below the 0.80
ill-defined threshold). So the **attractor definition is sound** — the problem is not a
messy target.

| group | mean d_min / anchor_radius | mean gap_in_source_steps |
|---|---|---|
| known weak pairs (L_R→R_L, L_R→R_R) | 25.39 | 175.12 |
| normal pairs | 22.65 | 137.57 |

Verdict: **`source_off_support_coverage_or_fallback`**.

Honest caveat on magnitude: `d_min / anchor_radius` is ~22–25 for *every* pair, not just
weak ones. That large value is expected and structural — these are independent takes, so no
clip's `z` naturally sits inside another clip's turn-end attractor (design §1.1: no
boundary-spanning support exists in the data). The decision-relevant signals are therefore
(a) **all anchors are well-defined**, and (b) weak pairs are only *marginally* further than
normal (~27% more z-steps to close the gap), **not** categorically unreachable.

Implication: this supports the fallback-aware route — a transition/bridge mechanism plus
source-specific (Walk_L_To_R) fallback — and argues **against** redesigning the attractor or
enumerating pairs in posttrain. It is consistent with closeout §10's weak-source-handling
blocker.

### 2.1 Implied bridge endpoints (exit_i → entry_j)

A2 now records, per pair, the closest-approach **exit frame** in the source (argmin of
cosine distance to the target anchor) and the **entry frame** in the target nearest that
exit-frame `z`. This is the `(exit_i, entry_j)` pair the B network is meant to produce, and
is directly reusable for bridge construction. Weak pairs:

| pair | exit_i (all / end-window) | entry_j | entry_cos_dist |
|---|---|---|---|
| Walk_L_To_R → Walk_R_To_L | 43 / 43 (of 50) | 85 (of 86) | 0.425 |
| Walk_L_To_R → Walk_R_To_R | 49 / 49 (of 50) | 82 (of 93) | 0.209 |

Observations:
- For both weak pairs the exit frame sits at the **source end-window** (43/50, 49/50) and the
  entry lands at the **target end-window** (85/86, 82/93). Closest approach is end-to-end
  (source convergence → target attractor), which is geometrically what a turn→turn handoff
  should look like, and the end-window exit agrees with the all-frame exit (no mid-clip
  shortcut). So the weak pairs are not geometrically pathological — their exit/entry frames
  are sensible.
- Weak-pair entry_cos_dist (0.21–0.42) is **not** an outlier; the worst entry gap in the
  whole matrix is a *normal*-classified pair (Walk_R_To_R → Walk_R_To_L, 0.773). This again
  says weak-pair difficulty is marginal, not structural.
- Caveat for downstream use: for several pairs the all-frame exit and the end-window exit
  diverge (e.g. Walk_F → Walk_R_To_R: 25 vs 81). When the closest approach is mid-clip,
  exit-frame selection must be **constrained to the convergence/end window**, or the bridge
  will exit from a non-convergent phase. Use the `exit_frame_source_end_window` field for
  bridge planning, not the unconstrained `exit_frame_all_frames`.

## 3. A3 — Re-entry phase must be contact-driven, not z-driven

Re-entry candidate = turn-end frame; map to a Walk_F frame by (i) contact distance and (ii)
`z` cosine, then compare the two chosen Walk_F frames.

| turn clip | contact-NN Walk_F frame | z-NN Walk_F frame | phase gap (/cycle) | contact sharpness | z agrees? |
|---|---|---|---|---|---|
| Walk_L_To_L | 60 | 85 | 0.287 | 0.008 | no |
| Walk_L_To_R | 53 | 86 | 0.379 | 0.019 | no |
| Walk_R_To_L | 50 | 84 | 0.391 | 0.276 | no |
| Walk_R_To_R | 3 | 25 | 0.253 | 0.015 | no |

Aggregate: mean phase gap 0.328 of cycle, mean contact sharpness 0.080, z-agrees **0/4**.

Interpretation: **`reentry_needs_contact_phase_not_z`**. Contact gives a mostly sharp,
well-defined re-entry phase (sharpness < 0.02 for three clips), but `z`'s nearest Walk_F
frame disagrees by ~1/3 of the gait cycle every time. So although `z` encodes contact (A1)
and has within-clip phase locality (P2), it is **not cross-clip phase-comparable** for
selecting a re-entry frame. The runtime "read target phase → map back to Walk_F phase" step
must be driven by the **contact / phase signal, not z**.

Sub-flag: Walk_R_To_L contact sharpness is 0.276 (aliased) — its re-entry phase is ambiguous
from contact alone and will need a velocity/direction tiebreak or fallback.

Caveat: Walk_F is a single cycle, so multi-cycle phase aliasing remains untestable with
current data (closeout §13). A3 is assessed within the one available cycle.

## 4. Net effect on the next-phase decision

- A1 (good news): contact/foot regime is in `z`, so attractor-membership / entry-retrieval
  can reason about contact, not just pose. Reduces what fallback must hand-craft.
- A2: target attractors are clean and weak rows are source-off-support → the route is
  **transition attraction + source-specific fallback**, not attractor redesign and not
  pair-wise posttrain. Matches closeout §10.
- A3: adds a concrete runtime-design constraint — **re-entry phase = contact-driven**, with
  a velocity/direction tiebreak where contact is aliased (Walk_R_To_L). `z` is for
  attractor membership / convergence, not for picking the Walk_F re-entry frame.

None of the three findings reopens the framework; together they sharpen the fallback-aware
plan and keep posttrain off the critical path.

## 5. Provisional-threshold posture

All gates here (R² > max(0.30, energy); kNN lift > 0.15; reach_norm 1.5; diffuse 0.80;
phase-agree 0.15) are smoke-only, calibrated on the current frozen artifact — same posture
as the P6 threshold contract note. They classify; they do not certify.
