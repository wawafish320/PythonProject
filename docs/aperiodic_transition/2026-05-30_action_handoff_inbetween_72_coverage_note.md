> TRIAGE: KEEP-AS-PROVENANCE. Historical evidence / anti-relitigation note; preserve to explain why this path was rejected or bounded. Do not use as a live plan unless a new decision record explicitly reopens it.

# Goal-Conditioned In-Betweening — §7.2 Data Pipeline + Coverage Findings

Date: 2026-05-30

Status: implementation closeout for spec §7.2 (data pipeline + unit tests) + real-clip
coverage findings — FOR REVIEW, input to §7.3. No training, no model, no loss, no
checkpoint dependency (spec §0 / staging lock).

Parent spec: `2026-05-29_goal_conditioned_inbetweening_spec.md` (§7.2 = staging step 2).
Direction: `2026-05-29_action_handoff_goal_conditioned_inbetweening_direction.md`.

## 0. What was built

Pure data/sampling pipeline + reproducible diagnostics + unit tests. Single source of
truth for the egocentric state and the grounded alignment lives in
`train/data/action_handoff_inbetween.py`; both diagnostic tools import it so the
diagnostic and the sampler can never drift.

- `train/data/action_handoff_inbetween.py` — egocentric state `s_t` (D_s=281 =
  pose276 + ego_vel2 + yaw_rate1 + contact2), full-state Walk_F alignment
  (pose-localize + contact-refine), groundability gate, `load_clip_states`, the
  3-type `InbetweenSampler`, goal-conditioning encoder, torch dataset wrapper.
- `tools/run_action_handoff_grounded_alignment_check.py` — productized §7.1 check.
- `tools/run_action_handoff_inbetween_sampler_coverage.py` — sampler behavior on the
  real 5 clips.
- `tests/train/test_action_handoff_inbetween_state_semantics.py`,
  `tests/train/test_action_handoff_inbetween_sampler_semantics.py` — 14 tests, all pass.

## 1. §7.1 alignment reproduced (locked acceptance)

Full-state φ = pose top-k neighborhood (cycle-phase localization) refined by min
contact distance. NB: a genuine standardized 281-d L2 is pose-dominated and degenerates
to pose-only (artifact-backed by the alignment tool `standardized_281d_comparator`: it
equals pose-only on 3/4 clips; R_L picks f0 contact_d 0.960 vs full-state f2 0.162); only
pose-localize + contact-refine reproduces the locked numbers — consistent with re-entry
resolver design D3 (ego_vel phase-flat, yaw_rate≈0 at onset, so neither picks the frame).

| clip | pose-only φ (contact_d) | full-state φ (cyc) | pose_d | contact_d | groundable |
|---|---|---|---|---|---|
| Walk_L_To_L | f41 (0.113) | f37 (0.43) | 0.009 | 0.074 | True |
| Walk_L_To_R | f40 (0.743) | f43 (0.49) | 0.016 | **0.703** | **False** |
| Walk_R_To_L | f0 (**0.960**) | **f2** (0.02) | 0.011 | **0.162** | True |
| Walk_R_To_R | f1 (0.029) | f82 (0.94) | 0.020 | 0.013 | True |

Walk_F egocentric sanity: yaw_rate min/med/max = −0.000/0.000/0.000 rad/s; ego
lateral |max| = 0.000 (straight walk is phase-flat; turn signal lives in heading
rotation — confirms the yaw_rate channel is load-bearing, spec §1.1).

Gate verdict: groundable = {L_L, R_L, R_R}; FAILS = {L_R}. Matches spec §2b exactly.

## 2. Sampler coverage on real clips (n=6000/progress, n_grounded=4000, seed=0)

Clip frames: Walk_F 87, L_L 54, L_R 50, R_L 86, R_R 93.

| progress | within | grounded | augmented | gap min/med/max | biased lift (med) |
|---|---|---|---|---|---|
| 0.00 | 0.510 | 0.348 | 0.142 | 12/12/12 | 1.32 |
| 0.50 | 0.493 | 0.357 | 0.150 | 21/21/21 | 1.17 |
| 1.00 | 0.506 | 0.343 | 0.151 | 28/30/30 | 1.02 |

Grounded fallback rate per turn clip: L_L/R_L/R_R = 100% grounded_ok, 0% fallback;
**L_R = 100% within-clip fallback** (later-onset never clears the gate within the
sampler's configured scan window onsets 1..8: best contact_d = 0.473 at onset 8 > 0.30.
Scanning past the window, contact_d does eventually clear 0.30 but pose_d crosses the
0.05 pose gate — onset drifts off the Walk_F loop pose — so the clip stays non-groundable
and the failure reason shifts from contact to pose. Artifact-backed by the alignment
tool's `later_onset_scan_failed_clips` table.).

## 3. Findings that feed §7.3 (the decisions these change)

1. **B1 risk is concentrated on Walk_L_To_R.** It receives ZERO grounded supervision —
   served entirely by within-clip + augmentation. So the §6 B1 gate **must report
   per-clip, not collapsed**: 3 groundable clips' high reach_rate will otherwise mask
   L_R. Recommend an explicit L_R row in the first B1 probe.
2. **`gap_max` is bounded by the shortest clip.** L_R (50 frames) cannot fit C+gap+K at
   gap=30 (needs 52); the sampler clamps its within-clip gap to 28. `gap_max=30` is not
   free — C=16/K=6/gap_max=30 interact with clip length. Setting these in §7.3 must
   account for the 50-frame floor.
3. **Biased-sampling lift washes out at long gaps** (1.32 → 1.02 as gap 12 → 30): a long
   masked middle spans most of the clip, so the onset/transition bias dilutes. If
   onset-focused supervision must persist at long gaps, bias the anchor position rather
   than the middle's mean interest.

## 4. Still PROVISIONAL / open (carry into §7.3)

- Groundability gate threshold = 0.30 contact_d (chosen between R_L 0.162 and L_R 0.703;
  wide margin, but the number itself awaits freerun calibration).
- K=6, C=16, gap 12→30, ratios 0.50/0.35/0.15 — all provisional (spec §1/§2).
- §8 model decisions (AR-with-goal vs masked; fine-tune base vs separate module; loss
  weights; B1 thresholds) are unresolved and gate the §7.3 build.

## 5. Out of scope (unchanged)

Model init/weights, losses, scheduled-sampling training loop, freerun runner wiring,
production thresholds, any >minute-scale training. These are §7.3+.
