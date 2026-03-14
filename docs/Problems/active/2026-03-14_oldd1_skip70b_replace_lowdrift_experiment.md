# 2026-03-14 old d1 skip-raw70b replace low-drift experiment

## Goal

Design the next leg-focused experiment on top of the clarified chain semantics:

- raw `70b` is diagnostic-only
- operational handoff is `70a -> new70b_replace`

So the next experiment should not be "skip 70b" again.

It should answer a narrower question:

- can we keep the leg gain that `new70b_replace` gets over `70a`,
- while reducing the nonleg / arm / all_ex_root drift introduced by the current replace stage?

---

## Hypothesis

Current `new70b_replace` is too aggressive for the old d1 lane.

Observed tradeoff (`70a -> new70b_replace`, model-source):

- `all_ex_root: 0.275083 -> 0.280736` (`+0.005653`, worse)
- `leg: 0.730911 -> 0.662440` (`-0.068472`, better)
- `nonleg: 0.176525 -> 0.198205` (`+0.021680`, worse)
- `arm: 0.203549 -> 0.226846` (`+0.023297`, worse)

Interpretation:

- the replace stage is doing something useful for leg,
- but it is over-updating the rest of the direct path.

So the next candidate should be:

- same `70a -> replace_zerophase_warmstart -> new70b_replace` semantics
- same downstream `70R -> 71 -> 72 -> lambda`
- but a more conservative replace stage

---

## Proposed experiment

### Control lane

Use the already completed old d1 lane:

- `70a`: `models/__tmp_oldd1_newflow_chain_20260314/70a/ckpt_last_WalkF_stage7_70a_from_oldd1_newflow_20260314.pth`
- `new70b_replace`: `models/__tmp_oldd1_newflow_chain_20260314/70b_replace/ckpt_last_WalkF_stage7_70b_replace_from_oldd1_newflow_20260314.pth`
- final: `models/__tmp_oldd1_newflow_chain_20260314/lambda/ckpt_last_WalkF_stage7_lambda_from_oldd1_newflow_20260314.pth`

### Candidate lane

Name:

- `new70b_replace_lowdrift`

Definition:

- source: same `70a` ckpt
- warmstart: same `replace_zerophase_warmstart`
- replace mode: still `direct_pose_phase_z_mode=replace_contacts`
- only change stage strength:
  - `lr=3e-4`
  - `epochs=1`
  - `steps_per_epoch=60`

Reason:

- keep semantics unchanged
- reduce update magnitude
- test whether the current replace stage is simply overshooting

### First run result (`2026-03-14 PM`)

This candidate has already been run once.

Artifacts:

- runner: `tools/run_oldd1_skip70b_replace_compare.py`
- result summary: `debug_output/_tmp_oldd1_skip70b_lowdrift_20260314/summary.json`
- readable result: `debug_output/_tmp_oldd1_skip70b_lowdrift_20260314/summary.md`

Stage-level model-source direct metrics:

| lane | DirectGeoLocalDeg | all_ex_root | leg | nonleg | arm | else |
|---|---:|---:|---:|---:|---:|---:|
| 70a | 0.275083 | 0.275083 | 0.730911 | 0.176525 | 0.203549 | 0.112650 |
| current `new70b_replace` | 0.280736 | 0.280736 | 0.662440 | 0.198205 | 0.226846 | 0.130508 |
| candidate `new70b_replace_lowdrift` | 0.156709 | 0.156709 | 0.375867 | 0.109324 | 0.126458 | 0.068824 |

Delta vs current `new70b_replace`:

- `DirectGeoLocalDeg=-0.124026`
- `all_ex_root=-0.124026`
- `leg=-0.286573`
- `nonleg=-0.088881`
- `arm=-0.100388`
- `else=-0.061684`

Interpretation:

- the low-drift candidate is not just a small cleanup
- it is a very large stage-level improvement over the current replace artifact
- this strongly supports the overshoot hypothesis for the current replace stage

But there is one important hotspot warning:

- `foot_l/ball_l@SIC12-15`: `1.049908 -> 0.551791` (better)
- `calf_r@SIC2-4`: `0.457252 -> 1.030131` (much worse)

So the current reading is:

- broad direct metrics: strong win
- local calf hotspot: clear regression

This means the candidate should not be accepted from stage metrics alone.

The next correct question is whether:

- `70R` and `71` can absorb this `calf_r@SIC2-4` regression,
- while preserving the large overall / leg gain already seen at the replace stage.

---

## Exact lane shape

Candidate chain:

1. Stage6
2. 70a
3. `new70b_replace_lowdrift` from `70a` warmstart
4. 70R
5. 71
6. 72
7. lambda final

Do not run raw `70b` as a required stage for this experiment.

If raw `70b` is run at all, it should be labeled diagnostic-only and kept out of the main verdict wording.

---

## Config recipe

Base config:

- `config/posttrain_WalkF_stage7_70b_phasezin_splitB2_pe32h512_20260227_fromarmchain.json`

Generate candidate config by overriding:

- `ckpt_in=<70a_warmstart_ckpt>`
- `out_dir=models/__tmp_oldd1_skip70b_lowdrift_20260314/70b_replace_lowdrift`
- `run_name=WalkF_stage7_70b_replace_lowdrift_from_oldd1_20260314`
- `direct_pose_use_phase_z=true`
- `direct_pose_phase_z_mode=replace_contacts`
- `lr=3e-4`
- `epochs=1`
- `steps_per_epoch=60`
- same `pretrain_contact + clamp1 + affine_mix08 + encoder_bundle`

Downstream:

- keep `70R` exactly the same promote recipe as current lane
- keep `71 / 72 / lambda` configs exactly the same

This keeps the variable isolated to the replace stage only.

---

## Readout plan

### Stage readout that matters most

Primary comparison:

- `70a`
- current `new70b_replace`
- candidate `new70b_replace_lowdrift`

Primary metrics:

- `all_ex_root`
- `leg`
- `nonleg`
- `arm`
- `DirectGeoLocalDeg`

### Acceptance rule at replace stage

The candidate is useful only if it does both:

1. improves drift relative to current `new70b_replace`
   - target:
     - `all_ex_root < 0.280736`
     - `nonleg < 0.198205`
     - `arm < 0.226846`
2. keeps a meaningful part of the leg gain vs `70a`
   - current leg gain vs `70a` is `0.068472`
   - minimum acceptable retained gain: at least half
   - therefore require:
     - `leg <= 0.696675`

Interpretation:

- if it cleans up nonleg/arm but leg bounces all the way back toward `70a`, it is not useful

### Downstream acceptance rule

At `70R` and `71`, compare candidate vs current old d1 lane.

The candidate is promising only if:

- `70R`:
  - `all_ex_root` is no worse than current lane by more than `+0.005`
  - `nonleg` is no worse than current lane by more than `+0.003`
- `71`:
  - `leg` improves vs current lane by at least `0.005`, or
  - `legs_main / hotspots` improve clearly without giving back nonleg

At final:

- compare against:
  - current old d1 lane
  - B full oldplan chain
  - C rollback planner core challenger
  - A accepted anchor

Primary final decision metric:

- not just `all_ex_root`
- must also improve:
  - `leg`
  - `legs_main`
  - `foot_l/ball_l@SIC12-15`
  - `calf_r@SIC2-4`

---

## Why this is the right next experiment

It is the narrowest experiment that matches what is now known:

- Stage6 starts behind on leg
- raw `70b` is not the operational handoff
- `new70b_replace` is the first stage where leg improves but nonleg/arm get traded away
- `70R` and `71` already do useful cleanup, so the next rational question is whether a cleaner replace stage gives them a better starting point

So this experiment is better than:

- re-running raw `70b`
- changing `70R` first
- changing `72` / `lambda` first

because it targets the earliest operational tradeoff stage in the lane.

---

## Continuation result (`70R -> 71`, 2026-03-14 PM)

This continuation has now been run.

Artifacts:

- runner: `tools/run_oldd1_skip70b_lowdrift_to71.py`
- result summary: `debug_output/_tmp_oldd1_skip70b_lowdrift_to71_20260314/summary.json`
- readable result: `debug_output/_tmp_oldd1_skip70b_lowdrift_to71_20260314/summary.md`
- candidate `70R`: `models/__tmp_oldd1_skip70b_lowdrift_to71_20260314/70R/ckpt_last_WalkF_stage7_70R_from_oldd1_lowdrift_replace_20260314.pth`
- candidate `71`: `models/__tmp_oldd1_skip70b_lowdrift_to71_20260314/71/ckpt_last_WalkF_stage7_71_from_oldd1_lowdrift_replace_20260314.pth`

Model-source direct metrics:

| lane | DirectGeoLocalDeg | all_ex_root | leg | nonleg | arm | else | calf_r@SIC2-4 | foot_l/ball_l@SIC12-15 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| candidate `new70b_replace_lowdrift` | 0.156709 | 0.156709 | 0.375867 | 0.109324 | 0.126458 | 0.068824 | 1.030131 | 0.551791 |
| current `70R` | 0.158235 | 0.158235 | 0.556049 | 0.072222 | 0.082665 | 0.047537 | 0.613849 | 1.118483 |
| candidate `70R` | 0.130926 | 0.130926 | 0.349263 | 0.083717 | 0.091849 | 0.064498 | 0.393019 | 0.860095 |
| current `71` | 0.111911 | 0.111911 | 0.295473 | 0.072222 | 0.082665 | 0.047537 | 0.440912 | 0.599272 |
| candidate `71` | 0.127787 | 0.127787 | 0.331611 | 0.083717 | 0.091849 | 0.064498 | 0.295644 | 0.540575 |

Key readout:

- The `calf_r@SIC2-4` hotspot does recover strongly once the lane reaches `70R`.
  - `1.030131 -> 0.393019` at candidate `70R`
  - `0.393019 -> 0.295644` at candidate `71`
- So the replace-stage calf regression is not persistent through the next two stages.

But the downstream tradeoff is mixed:

- candidate `70R` is much better than current `70R` on:
  - `all_ex_root: -0.027310`
  - `leg: -0.206786`
  - `calf_r@SIC2-4: -0.220830`
  - `foot_l/ball_l@SIC12-15: -0.258388`
- candidate `70R` is worse than current `70R` on:
  - `nonleg: +0.011496`
  - `arm: +0.009184`

At `71` the lane partially gives back the `70R` win:

- candidate `71` vs current `71`
  - `all_ex_root: +0.015877` (worse)
  - `leg: +0.036138` (worse)
  - `nonleg: +0.011496` (worse)
  - `arm: +0.009184` (worse)
  - `calf_r@SIC2-4: -0.145269` (better)
  - `foot_l/ball_l@SIC12-15: -0.058697` (better)

Interpretation:

- `70R` can absorb the calf hotspot and still preserve a very strong leg win.
- `71` improves the candidate's calf hotspot further, but it does not restore the lane to the current old d1 `71` level on the main direct metrics.
- So this low-drift replace variant is still interesting as a leg-sensitive challenger lane, but only if later stages can keep the `70R` gain instead of letting `71` wash it back out.

Operational verdict at this point:

- `70R`: encouraging
- `71`: not yet promotable over the current old d1 lane
- next useful A/B remains:
  - candidate `70R -> 71` behavior vs current `70R -> 71`
  - specifically why `71` gives back leg/all_ex_root while hotspot cleanup still improves

---

## Follow-up only if this candidate fails

If `new70b_replace_lowdrift` still has the same leg/nonleg tradeoff, the next fallback should be:

- same recipe but `steps_per_epoch=30`

If both fail, move the investigation downstream:

- `new70b_replace` vs `70R`
- `70R` vs `71`

instead of reopening raw `70b`.
