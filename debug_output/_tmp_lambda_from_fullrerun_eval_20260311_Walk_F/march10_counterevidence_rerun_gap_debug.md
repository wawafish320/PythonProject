# March 10 counter-evidence vs March 11 fresh rerun gap

## Short conclusion

- Most likely: the gap is mainly a `train.training_MPL` run-to-run basin problem on the current mainline, not a deterministic "current stack always runs bad" bug.
- More specifically, the fresh `2026-03-11` basetrain landed in a different contact-plan / event-clock basin, and that basin was then preserved through `Stage6 -> 70R` and only partially repaired by `71/72`.
- Less likely: a literal downstream handoff path mistake. The saved `ckpt_in` chain is internally consistent all the way to fresh lambda final.
- Also less likely: compare/eval bar mixing as the root cause. It explains confusion in acceptance wording, but it does not explain the same-command `70R -> 71 -> 72` hotspot divergence.
- Not supported as main cause: `train/history.py` shared pose-history helper or `contact_phase_state` removal alone. Existing surviving probes only support "possible contributors", not "proven primary cause".

## Metric bookkeeping: keep the bars separate

These numbers answer different questions and must not be mixed:

| Quantity | Value | Meaning |
|---|---:|---|
| `2026-03-07 eval_on anchor` | `0.131316` | trainbase anchor bar |
| `historical accepted final` | `0.112947` | replacement bar from historical accepted compare/doc snapshot |
| `2026-03-11 accepted final recheck` | `0.122122` | same-command localization bar only |
| `2026-03-10 better candidate` | `0.1180` | manual run summary only; original artifacts/logs were deleted |
| `2026-03-11 fresh final` | `0.121730` | fresh rerun final |

Immediate reads:

- Relative to the `eval_on` anchor, fresh final still improves: `0.131316 -> 0.121730` (`-7.30%`).
- Relative to the historical accepted final, fresh final still regresses: `0.112947 -> 0.121730` (`+7.78%`).
- Relative to the `2026-03-10` better candidate, fresh final is still worse: `0.1180 -> 0.121730` (`+3.16%` non-reproduction).
- The `2026-03-10` better candidate itself was a real counter-example to "current mainline cannot produce a strong chain": it improved over the `eval_on` anchor by about `-10.14%`, but still trailed the historical accepted final by about `+4.47%`.

Important note on the March 10 run:

- The `2026-03-10` `0.1180` result is from a manually preserved run summary.
- Its original logs/artifacts were later deleted.
- So it is strong counter-evidence, but not a fully auditable artifact pack.

## What the surviving stage artifacts say

### 1) The fresh chain divergence is already present upstream of `71`

From the surviving same-command artifacts:

- accepted trainbase best-free: `DirectGeoLocalDeg mean = 6.322915`, `ContactPlanPerC ~= [0.3164, 0.5656]`
- fresh trainbase best-free: `DirectGeoLocalDeg mean = 5.933194`, `ContactPlanPerC ~= [0.7440, 0.1771]`
- fresh `Stage6`, `70a`, `70b`, `70R`, `71`, `72`, and fresh final all keep essentially the same fresh-side plan signature

This means:

- the fresh-vs-accepted side flip is already present in fresh trainbase best-free;
- `70R` is not the source of the flip;
- `71` is the first stage where the bad basin becomes clearly visible in the target SIC windows.

### 2) `71` is the first exposure stage, but not necessarily the root cause

For the hotspot windows:

| Stage | Global mean | SIC13-15 | SIC36-37 |
|---|---:|---:|---:|
| accepted `70R` | `0.153849` | `0.208464` | `0.132891` |
| fresh `70R` | `0.152398` | `0.185592` | `0.136220` |
| accepted `71` | `0.121916` | `0.134648` | `0.100780` |
| fresh `71` | `0.127998` | `0.182665` | `0.144194` |
| accepted `72/final recheck` | `0.122122` | `0.120387` | `0.106731` |
| fresh `72/final` | `0.121730` | `0.153277` | `0.122215` |

Read:

- `71` is where the accepted chain repairs the target windows strongly.
- Fresh `71` does not perform the same repair.
- `72` partially repairs the fresh `71` spike.
- `lambda final` is basically unchanged from fresh `72` on those windows.

So the clean phrasing is:

- `71` is where the fresh problem is first exposed at full strength;
- it does **not** follow that `71` code itself is the unique root cause;
- it is equally consistent with "upstream basin mismatch, amplified at `71`".

### 3) This is not just a global-mean story

The same-command recheck bar makes that obvious:

- accepted final recheck: `0.122122`
- fresh final: `0.121730`

Global mean alone would say "fresh is slightly better".
But in the actual target windows:

- accepted final recheck `SIC13-15 = 0.120387`, fresh final `= 0.153277`
- accepted final recheck `SIC36-37 = 0.106731`, fresh final `= 0.122215`

So:

- compare/eval bar mixing can definitely mislead the narrative;
- but even after fixing the bookkeeping, the fresh hotspot regression is still real.

## What current code/history says

### 1) Current relevant code has not moved after March 10, but the March 10 better run predates the rotvec fix commit

`git log` on the relevant files shows the current repo head is still:

- `f6e117e` (`2026-03-10 21:33:31 +0800`)

for:

- `train/training_MPL.py`
- `train/posttrain.py`
- `train/models.py`
- `train/validate/run_freerun_cycles.py`
- `config/exp_phase_mpl.clean.json`

So I do **not** see evidence for a second code drift after `f6e117e` between `2026-03-10` and `2026-03-11` in the files you asked me to audit.

However, with the new timing clarification, the important boundary is now:

- the `2026-03-10` better candidate happened **before** `f6e117e`;
- the `2026-03-11` fresh rerun happened **after** `f6e117e`.

That means March 10 vs March 11 is not a pure same-code comparison. It crosses a real semantic boundary:

- `train/geometry.py` changed `so3_log_map` from legacy half-angle behavior to standard axis-angle / rotvec;
- `train/posttrain.py` and `train/validate/run_freerun_cycles.py` removed the old external `*2` compensation;
- the repo's norm/bundle/template assets were migrated and stamped to the new semantics at the same time.

So the following hypothesis is now stronger than I wrote before:

- March 10 vs March 11 may partly differ because they were trained/evaluated under different rotvec semantics contracts, especially in stages that use direct-leg omega alignment.

At the same time, this still does **not** by itself explain the fresh basetrain-side flipped `ContactPlanPerC`, because that flip is already present at fresh trainbase before `71/72/final`.

### 2) `train.training_MPL` is run-to-run nonreproducible by default

Current code evidence:

- `train/training_MPL.py:5615` builds the train loader with `shuffle=True`
- I do not find a trainbase `_set_seed(...)` path in `train/training_MPL.py`
- I do not find a trainbase `--seed` argument being carried into the saved config
- the fresh fullrerun `config_resolved.json` has no saved seed

By contrast, posttrain **does** seed itself:

- `train/posttrain.py:3126` defines `_set_seed`
- `train/posttrain.py:5107` calls `_set_seed(cfg.seed)`
- active posttrain configs use `seed=0`

So the asymmetry is:

- basetrain: inherently run-to-run variable
- downstream posttrain: mostly deterministic given the input checkpoint

This is the strongest current explanation for "March 10 could hit a better candidate while March 11 fresh rerun did not reproduce it".

### 3) Best-free selection is a possible secondary issue, but not the main one inside the March 11 run

Current selector behavior:

- `train/training_MPL.py:2189` computes best-free from `GeoDegCurve` drift slope
- `train/training_MPL.py:2733-2739` saves best-free by minimizing that drift proxy, not by downstream `DirectGeoLocalDeg`, not by contact-plan side semantics, and not by posttrain-end performance

However, within the `2026-03-11` fresh basetrain run itself:

- `best_free` lands on epoch `18`
- epoch `18` is also the best `GeoLocalDeg` among the saved `valfree` epochs of that same run

So I do **not** see evidence that this specific run threw away an obviously better freerun epoch.

The more accurate read is:

- the selector is not aligned with the eventual downstream objective;
- but the bigger issue still looks like the basin of the run, not a simple in-run epoch mis-pick.

## New minimal A/B I ran: fresh trainbase `best_teacher` vs existing fresh `best_free`

I ran one new trainbase-only eval today:

- new artifact: `debug_output/_tmp_basetrain_fullrerun_bestteacher_eval_20260312_Walk_F/Walk_F_freerun_cycles.json`

Compared with the existing fresh best-free trainbase eval:

| Fresh trainbase ckpt | Global mean | SIC13-15 | SIC36-37 | ContactPlanPerC |
|---|---:|---:|---:|---|
| `best_free` | `5.933194` | `5.365458` | `5.798755` | `[0.7440, 0.1771]` |
| `best_teacher` | `5.976292` | `5.391453` | `5.809125` | `[0.7226, 0.1917]` |

Interpretation:

- fresh `best_teacher` is still first-channel-high, i.e. still on the same flipped side family;
- it is only slightly less extreme than fresh `best_free`, not accepted-like;
- therefore the March 11 gap is **not** well explained by "the fresh run had a good trainbase ckpt, but best-free selected the wrong winner".

This pushes the ranking toward:

1. different basetrain basin across reruns
2. downstream `70R/72` sensitivity to the rotvec semantics boundary plus the inherited basin
3. downstream `71/72` sensitivity to that basin
4. selector-proxy mismatch as a secondary factor, not the main one

## Handoff mismatch audit

I checked the saved `posttrain_cfg.ckpt_in` chain in the fresh artifacts:

- fresh `70R` <- fresh `70b_replace`
- fresh `71` <- fresh `70R`
- fresh `72` <- fresh `71`
- fresh `lambda final` <- fresh `72`

So I do not see evidence for:

- accidentally mixing accepted and fresh downstream checkpoints;
- an obvious `ckpt_in` typo causing the March 11 final to come from the wrong upstream branch.

That does **not** rule out semantic sensitivity to the upstream start point.
It does rule out the simplest path-mix bug.

## Hypothesis ranking

### Most likely

`2026-03-11` failed to reproduce the `2026-03-10` better candidate because two things changed at once:

1. fresh basetrain landed in a different basin under an unseeded `train.training_MPL` run;
2. March 10 vs March 11 crossed the `f6e117e` rotvec-semantics boundary, which is especially relevant for `70R/72` direct-leg omega alignment behavior.

Short version:

- trainbase basin differs
- March 10 candidate and March 11 rerun are not on exactly the same SO(3) semantics contract
- `70R -> 71 -> 72` is sensitive to that basin
- fresh `71` is where the mismatch becomes visible
- fresh `72` only partially fixes it
- lambda final does almost nothing after that

### Plausible but secondary

- best-free proxy misalignment: yes, because the selector optimizes freerun drift slope rather than downstream hotspot behavior
- but no current evidence says the March 11 run already contained an accepted-like ckpt that was simply not chosen

### Currently least likely

- pure downstream handoff typo
- pure compare/eval bar mix
- a second post-`f6e117e` deterministic code regression in the audited files
- `train/history.py` shared pose-history helper as primary cause
- `contact_phase_state` removal alone as primary cause

## What the rotvec fix can and cannot explain

What it **can** explain:

- March 10 and March 11 are no longer apples-to-apples for stages that use direct-leg omega / oracle alignment.
- This is especially relevant for `70R` and `72`, both of which use `direct_pose_leg_align_mode=proj` with `direct_pose_leg_align_weight=20`.
- So part of the March 10 vs March 11 gap may be a real pre-fix vs post-fix training-objective change, not just noise.

What it **does not** explain well on its own:

- fresh trainbase best-free is already side-flipped before posttrain starts;
- fresh trainbase best-teacher is still in the same flipped family;
- `71` is where the hotspot gets exposed most strongly, but `71` itself has `direct_pose_leg_align_weight=0`.

So the rotvec fix now looks like a meaningful cross-date confounder, but not a full replacement for the basin / inherited-plan mismatch explanation.

## Recommended next minimal experiment

If you allow exactly one more small experiment, I would do:

### `71`-only current-code handoff A/B

Run current `71` config twice, with the same seed and same runtime:

1. `accepted 70R(s180) -> current 71`
2. `fresh 70R(fullrerun) -> current 71`

Then evaluate both with the same `Walk_F` freerun command and compare:

- global mean
- `SIC13-15`
- `SIC36-37`
- `ContactPlanPerC`

Why this is the highest-value next step:

- it is still much cheaper than any fullchain rerun;
- it directly tests whether current `71` code can still repair an accepted-like starting basin;
- it tells us whether the fresh failure is mostly "bad upstream basin" or "current `71` no longer repairs even a good basin";
- after the new trainbase `best_teacher` A/B, this is now the cleanest remaining branch in the hypothesis tree.

Expected decision value:

- if current `71` still repairs accepted `70R` but not fresh `70R`, the diagnosis becomes "basin/handoff sensitivity" with much higher confidence;
- if current `71` also fails on accepted `70R`, then current `71` code/runtime becomes much more suspicious than it looks today.
