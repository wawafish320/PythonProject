# Walk_F Path C Stage 1 Closeout

## Scope

This note closes the current Stage 1 training-side intervention loop for Walk_F Path C after the remaining cheap matrix gap (`B0`) was executed on seed2024.

Artifacts referenced here:

- wrong C0 / partial B0: `debug_output/_tmp_walkf_pathc_c0_stage1_20260503`
- real C0: `debug_output/_tmp_walkf_pathc_c0_real_20260504`
- D0: `debug_output/_tmp_walkf_pathc_d0_20260504_rerun1`
- B0: `debug_output/_tmp_walkf_pathc_b0_20260504`
- motion-quality audit: `docs/analysis/walkf_motion_quality_field_audit.md`
- zero-training deployment evidence: `debug_output/_tmp_walkf_method2_zero_training_20260502` and `debug_output/_tmp_walkf_method2_soup_characterization_20260502`

## Matrix Outcome

| arm | actual semantics | seed2024 result | key readout |
| --- | --- | --- | --- |
| A0 | baseline | reference only | baseline free-run rate=`0.1507` |
| wrong C0 (= partial B0) | raw SS, `K=5`, no stabilizer | `NOT_DIRECTIONAL` | TF-off worsened (`rate_delta_pct=-30.06%`), hist_tf only marginal (`2.82%`) |
| real C0 | stabilizer only, no SS | `AMBIGUOUS` | freeze: vel mean=`0.1694`, vel std=`0.1431`, silence=`0.6630`, rate delta=`24.91%` |
| D0 | SS `K=3` + paired teacher/free pose consistency | `AMBIGUOUS` | deeper freeze: vel mean=`0.1463`, vel std=`0.1986`, silence=`0.7719`, rate delta=`37.00%` |
| B0 | SS `K=3` + rollout pose loss only, no consistency | `NOT_DIRECTIONAL` | jitter/explode: vel mean=`0.3928`, vel std=`0.2786`, jitter=`0.0070`, rate delta=`-101.91%` |

Stage 1 conclusion is now closed under the intended semantics:

- pure SS endpoint is unstable
- pure stabilizer endpoint freezes
- combined D0 does not resolve the tradeoff; it strengthens the freeze attractor instead of breaking it
- B0 closes the last cheap matrix gap and does not justify further same-family pilot branching

## Main Findings

### 1. Freeze attractor is structural in the current training setup

The strongest finding is that D0 freezes more severely than real C0:

- real C0: `joint_angle_velocity_ratio_mean=0.1694`, `per_joint_silence_rate=0.6630`
- D0: `joint_angle_velocity_ratio_mean=0.1463`, `per_joint_silence_rate=0.7719`

Scheduled sampling at `K=3` did not break the low-motion solution basin. In this setup it coexisted with paired pose consistency and made the freeze signature worse. That is not evidence for a simple dosage miss; it is evidence that the current loss/horizon interaction still rewards short-horizon low-motion behavior.

### 2. Raw rate is not reliable once freeze appears

Both real C0 and D0 show positive `rate_delta_pct`, but the pose-based gate marks them as failures:

- real C0 rate delta=`24.91%`, verdict=`AMBIGUOUS`, failure mode=`freeze`
- D0 rate delta=`37.00%`, verdict=`AMBIGUOUS`, failure mode=`freeze`

Interpretation:

- lower rate can come from trivial motion collapse
- rate improvement without sufficient joint motion is not a valid closed-loop success
- primary conclusions must stay conditioned on the pose gate, not on rate alone

### 3. Pose-based anti-cheating gate paid off

The redesign in `docs/analysis/walkf_motion_quality_field_audit.md` was necessary and correct:

- root-family `motion_quality` fields remain `sanity_only`
- anti-cheating gating stays on pose output behavior:
  - `joint_angle_velocity_ratio_mean`
  - `joint_angle_velocity_ratio_std`
  - `joint_angle_jitter_score`
  - `per_joint_silence_rate`
  - `GeoLocalDeg_temporal_smoothness`

Without the pose gate, D0 would have looked better than C0 on rate alone even though it froze harder.

## Decision

Stage 1 should stop here rather than continue to lower-ROI same-family pilots.

Reason:

- `B0` was the last cheap missing matrix cell
- `B0` did not produce a viable non-freezing, non-exploding compromise
- `D0` strengthened freeze instead of weakening it
- remaining ideas like lower consistency weight, hidden consistency, or longer `K` move into a different scope because they need new assumptions or new control mechanisms

This closeout is about current Path C semantics only. It is not a claim that all future training-side interventions are exhausted. It is a claim that this current SS/consistency family does not show a clean Stage 1 win on seed2024.

## Deployment Recommendation

Use post-hoc soup as the current practical mitigation path.

Supporting evidence:

- `debug_output/_tmp_walkf_method2_zero_training_20260502/summary.md`
  - baseline seed2024 Walk_F mean=`13.070575`
  - subset soup (`anchor_seed2025`) Walk_F mean=`8.156637`
  - readout verdict=`weight_space_plausible`
- `debug_output/_tmp_walkf_method2_soup_characterization_20260502/summary.md`
  - post-hoc soup is described as the "leading zero-training mitigation"
  - carry drift and self-horizon amplification both improved in the subset soup characterization

So the best current recommendation is:

- deployment-side: prefer the validated post-hoc soup path
- research-side: preserve the new diagnostics and stop forcing more Path C pilot branches into Stage 1

## Future Work

If training-side work resumes, it should be framed as a new stage with new assumptions:

- architecture-level anti-freeze design, not just another weight sweep
- longer-horizon training only together with explicit explosion control
- data-diversity / motion-variance audit to test whether freeze is partly data-induced
- rate metric redesign so motion collapse is penalized directly rather than handled only downstream by the gate

Operationally, keep the current audit and resume infrastructure. The transferable output of this stage is the diagnostic chain:

- closed-loop compounding is the relevant failure axis
- freeze attractor is real under pose-space consistency
- pose-based anti-cheating gating is necessary
- post-hoc soup is already a usable zero-training patch
