# 72 loss curve attribution

Artifacts:

- runner: `tools/run_72_loss_curve_attribution.py`
- machine summary: `debug_output/_tmp_72_loss_curve_attribution_20260314/summary.json`
- readable summary: `debug_output/_tmp_72_loss_curve_attribution_20260314/summary.md`
- loss curve plot: `debug_output/_tmp_72_loss_curve_attribution_20260314/72_loss_curve_compare.png`
- loss curve numeric summary: `debug_output/_tmp_72_loss_curve_attribution_20260314/72_loss_curve_summary.md`

Scope guard:

- active comparison is `current 71 -> 72` vs `candidate 71 (lr=3e-4) -> 72`
- `72` semantics stayed unchanged
- first pass uses the existing `72` posttrain logs
- because log alone cannot answer when freerun aggregate first flips, this round also adds replay snapshots at `s000/s005/s010/s020/s040/s060/s120/s180`
- eval contract is model-source only; strict was not needed because the conclusion is already stable under replay snapshots

## Short conclusion

- `candidate 71 (lr=3e-4)` really does hand `72` a better start on aggregate leg/all_ex_root:
  - `all_ex_root: 0.111911 -> 0.107064`
  - `leg: 0.295473 -> 0.215044`
- but unchanged `72` gives that cleaner start back almost immediately:
  - at `s000`, candidate is still better than current on aggregate
    - `all_ex_root: -0.004846`
    - `leg: -0.080429`
  - by `s005`, candidate has already flipped to worse
    - `all_ex_root: +0.021607`
    - `leg: +0.068372`
- existing `72` logs show the same story from the train side: candidate `72` has a much larger early overshoot on the leg-side direct objective
  - `total start20: 2.739258 vs 2.263405`
  - `dir_geo start20: 2.734824 vs 2.261341`
  - `dir_group_norm_leg start20: 1.706216 vs 1.192757`
  - `leg_align_weighted start20: 0.004434 vs 0.002064`
- mid training is misleadingly calm:
  - candidate `72` is slightly lower on `total/dir_geo mid20`
  - but replay aggregate never recovers against current at any post-start snapshot
  - so this is **not** just late overfit, and **not** just snapshot selection
- the surviving wins are local, not global:
  - final candidate `72` still keeps `foot_l/ball_l@SIC12-15` and `calf_r@SIC2-4` better than current `72`
  - but broader regressions on `calf_l`, `ball_l`, `ball_r`, plus SIC35-48 / SIC21-22 / SIC03 windows, drag `leg` and then `all_ex_root` back above current
- best next minimal change: try **lower LR `72`** first, or a **gentler `72`**; plain early-stop only helps if you effectively make `72` near-zero, because every post-start candidate snapshot is already worse than current on aggregate

## End-state table

| lane | DirectGeoLocalDeg | all_ex_root | leg | nonleg | arm | legs_main | arms_main | foot_l/ball_l@SIC12-15 | calf_r@SIC2-4 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| current `71` | 0.111911 | 0.111911 | 0.295473 | 0.072222 | 0.082665 | 0.295473 | 0.082665 | 0.599272 | 0.440912 |
| candidate `71` (`lr=3e-4`) | 0.107064 | 0.107064 | 0.215044 | 0.083717 | 0.091849 | 0.215044 | 0.091849 | 0.429449 | 0.099602 |
| current `72` | 0.112074 | 0.112074 | 0.296389 | 0.072222 | 0.082665 | 0.296389 | 0.082665 | 0.812663 | 0.288880 |
| candidate `72` | 0.121936 | 0.121936 | 0.298698 | 0.083717 | 0.091849 | 0.298698 | 0.091849 | 0.586438 | 0.191232 |

Immediate read:

- candidate `71` wins aggregate at the `72` handoff
- unchanged `72` barely changes current lane aggregate, but hurts candidate lane a lot
- final candidate `72` still wins the two named hotspots, yet loses aggregate:
  - `all_ex_root: +0.009863`
  - `leg: +0.002308`
  - `foot_l/ball_l@SIC12-15: -0.226225`
  - `calf_r@SIC2-4: -0.097649`

## Replay snapshots

Replay finals exactly reproduced the known reference finals, so these snapshots are apples-to-apples with the already accepted `71/72` endpoints.

| lane_snapshot | DirectGeoLocalDeg | all_ex_root | leg | nonleg | arm | legs_main | arms_main | foot_l/ball_l@SIC12-15 | calf_r@SIC2-4 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| current `s000` | 0.111911 | 0.111911 | 0.295473 | 0.072222 | 0.082665 | 0.295473 | 0.082665 | 0.599272 | 0.440912 |
| current `s005` | 0.129621 | 0.129621 | 0.395093 | 0.072222 | 0.082665 | 0.395093 | 0.082665 | 1.397935 | 0.463666 |
| current `s010` | 0.128126 | 0.128126 | 0.386686 | 0.072222 | 0.082665 | 0.386686 | 0.082665 | 0.827815 | 0.357110 |
| current `s020` | 0.114371 | 0.114371 | 0.309314 | 0.072222 | 0.082665 | 0.309314 | 0.082665 | 0.709409 | 0.305969 |
| current `s040` | 0.116234 | 0.116234 | 0.319793 | 0.072222 | 0.082665 | 0.319793 | 0.082665 | 0.568103 | 0.294565 |
| current `s060` | 0.120596 | 0.120596 | 0.344328 | 0.072222 | 0.082665 | 0.344328 | 0.082665 | 0.902402 | 0.264104 |
| current `s120` | 0.106344 | 0.106344 | 0.264158 | 0.072222 | 0.082665 | 0.264158 | 0.082665 | 0.480181 | 0.290744 |
| current `s180` | 0.112074 | 0.112074 | 0.296389 | 0.072222 | 0.082665 | 0.296389 | 0.082665 | 0.812663 | 0.288880 |
| candidate `s000` | 0.107064 | 0.107064 | 0.215044 | 0.083717 | 0.091849 | 0.215044 | 0.091849 | 0.429449 | 0.099602 |
| candidate `s005` | 0.151228 | 0.151228 | 0.463466 | 0.083717 | 0.091849 | 0.463466 | 0.091849 | 1.155790 | 0.246557 |
| candidate `s010` | 0.152090 | 0.152090 | 0.468314 | 0.083717 | 0.091849 | 0.468314 | 0.091849 | 1.809244 | 0.313736 |
| candidate `s020` | 0.131960 | 0.131960 | 0.355084 | 0.083717 | 0.091849 | 0.355084 | 0.091849 | 1.407482 | 0.136625 |
| candidate `s040` | 0.144241 | 0.144241 | 0.424164 | 0.083717 | 0.091849 | 0.424164 | 0.091849 | 0.776220 | 0.113536 |
| candidate `s060` | 0.140788 | 0.140788 | 0.404737 | 0.083717 | 0.091849 | 0.404737 | 0.091849 | 1.128362 | 0.286177 |
| candidate `s120` | 0.121458 | 0.121458 | 0.296009 | 0.083717 | 0.091849 | 0.296009 | 0.091849 | 0.521412 | 0.199641 |
| candidate `s180` | 0.121936 | 0.121936 | 0.298698 | 0.083717 | 0.091849 | 0.298698 | 0.091849 | 0.586438 | 0.191232 |

### Candidate minus current by snapshot

This is the clearest answer to “when does the regression first appear?”

| snapshot | d_all_ex_root | d_leg | d_nonleg | d_arm | d_foot_l/ball_l@SIC12-15 | d_calf_r@SIC2-4 |
|---|---:|---:|---:|---:|---:|---:|
| `s000` | -0.004846 | -0.080429 | 0.011496 | 0.009184 | -0.169823 | -0.341310 |
| `s005` | 0.021607 | 0.068372 | 0.011496 | 0.009184 | -0.242144 | -0.217110 |
| `s010` | 0.023964 | 0.081627 | 0.011496 | 0.009184 | 0.981429 | -0.043374 |
| `s020` | 0.017589 | 0.045770 | 0.011496 | 0.009184 | 0.698073 | -0.169344 |
| `s040` | 0.028007 | 0.104371 | 0.011496 | 0.009184 | 0.208118 | -0.181029 |
| `s060` | 0.020191 | 0.060408 | 0.011496 | 0.009184 | 0.225960 | 0.022073 |
| `s120` | 0.015115 | 0.031851 | 0.011496 | 0.009184 | 0.041231 | -0.091103 |
| `s180` | 0.009863 | 0.002308 | 0.011496 | 0.009184 | -0.226225 | -0.097649 |

Key readout:

- the aggregate flip happens at `s005`, not in the tail
- candidate never regains the aggregate lead at any post-start snapshot
- current lane also has its own `72` instability (`s120` is its best snapshot, not `s180`), but candidate remains worse than current on aggregate at every post-start snapshot anyway
- so the candidate problem is not just “picked the wrong tail checkpoint”

## Loss curve read

Artifacts:

- plot: `debug_output/_tmp_72_loss_curve_attribution_20260314/72_loss_curve_compare.png`
- numeric summary: `debug_output/_tmp_72_loss_curve_attribution_20260314/72_loss_curve_summary.md`

Important note first:

- the stored `72` posttrain log has **no explicit `omega`-named loss key**
- the observable `72`-specific terms are the `leg_align_*` family:
  - `leg_align_loss`
  - `leg_align_weighted`
  - `leg_align_distal_loss`
  - `leg_align_proximal_loss`
  - plus per-joint `leg_align_joint_loss_*`

### Window means (`start20 / mid20 / late20`)

| key | current start20 | candidate start20 | delta | current mid20 | candidate mid20 | delta | current late20 | candidate late20 | delta |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `total` | 2.263405 | 2.739258 | 0.475853 | 1.924936 | 1.873930 | -0.051006 | 1.983918 | 2.003513 | 0.019595 |
| `dir_geo` | 2.261341 | 2.734824 | 0.473483 | 1.923878 | 1.872471 | -0.051406 | 1.982859 | 2.001883 | 0.019024 |
| `leg_align_weighted` | 0.002064 | 0.004434 | 0.002370 | 0.001059 | 0.001459 | 0.000401 | 0.001059 | 0.001630 | 0.000571 |
| `dir_group_norm_leg` | 1.192757 | 1.706216 | 0.513459 | 0.924986 | 0.887224 | -0.037762 | 0.995050 | 1.006654 | 0.011605 |
| `dir_leg_base` | 0.005205 | 0.007214 | 0.002009 | 0.003865 | 0.004584 | 0.000719 | 0.003868 | 0.004719 | 0.000852 |
| `dir_nonleg_base` | 0.001242 | 0.001272 | 0.000030 | 0.001233 | 0.001239 | 0.000006 | 0.001226 | 0.001234 | 0.000009 |
| `boundary_dir_geo` | 0.007636 | 0.008373 | 0.000737 | 0.007842 | 0.008251 | 0.000409 | 0.008131 | 0.008767 | 0.000636 |

### Epoch means

| key | current epoch1/2/3 | candidate epoch1/2/3 | read |
|---|---|---|---|
| `total` | 2.104447 / 1.923855 / 2.001807 | 2.276644 / 1.896688 / 2.021917 | candidate is much worse in epoch1, slightly lower in epoch2, then worse again in epoch3 |
| `dir_geo` | 2.102736 / 1.922738 / 2.000763 | 2.273151 / 1.895149 / 2.020420 | same shape as `total` |
| `leg_align_weighted` | 0.001711 / 0.001117 / 0.001044 | 0.003493 / 0.001538 / 0.001498 | candidate pays a higher `72`-specific leg-align cost all the way through |
| `dir_group_norm_leg` | 1.074735 / 0.914937 / 1.003803 | 1.265739 / 0.891172 / 1.023311 | candidate collapses after the big early spike, but not into better freerun aggregate |
| `dir_leg_base` | 0.004870 / 0.003949 / 0.003816 | 0.006565 / 0.004626 / 0.004547 | leg-side direct base stays consistently higher on candidate |
| `boundary_dir_geo` | 0.007662 / 0.007812 / 0.007833 | 0.008495 / 0.008339 / 0.008608 | candidate is boundary-worse the whole time |

Peak inside the first 20 steps:

- `total`: current `s005=3.113869`, candidate `s005=4.478547`
- `dir_geo`: current `s005=3.108846`, candidate `s005=4.466986`
- `dir_group_norm_leg`: current `s005=1.939771`, candidate `s005=3.393183`
- `leg_align_weighted`: current `s005=0.005022`, candidate `s005=0.011561`

Interpretation:

- this is a classic **early overshoot** pattern
- candidate mid20 train loss then looks a bit better, but replay aggregate is already worse from `s005` onward
- that means the logs support **early over-step + some objective mismatch / wrong tradeoff**, not simple late overfit
- snapshot selection matters on both lanes, but does **not** explain away the candidate regression

## 71 -> 72 gain decomposition

This separates inherited `71` start differences from `72`’s own contribution.

| metric | inherited (candidate71-current71) | current72-current71 | candidate72-candidate71 | stage72 gain gap | final gap (candidate72-current72) |
|---|---:|---:|---:|---:|---:|
| DirectGeoLocalDeg | -0.004846 | 0.000163 | 0.014872 | 0.014709 | 0.009863 |
| all_ex_root | -0.004846 | 0.000163 | 0.014872 | 0.014709 | 0.009863 |
| leg | -0.080429 | 0.000916 | 0.083653 | 0.082737 | 0.002308 |
| nonleg | 0.011496 | 0.000000 | 0.000000 | 0.000000 | 0.011496 |
| arm | 0.009184 | 0.000000 | 0.000000 | 0.000000 | 0.009184 |
| legs_main | -0.080429 | 0.000916 | 0.083653 | 0.082737 | 0.002308 |
| arms_main | 0.009184 | 0.000000 | 0.000000 | 0.000000 | 0.009184 |
| foot_l/ball_l@SIC12-15 | -0.169823 | 0.213391 | 0.156989 | -0.056403 | -0.226225 |
| calf_r@SIC2-4 | -0.341310 | -0.152032 | 0.091629 | 0.243661 | -0.097649 |

This is the core attribution:

- candidate inherits a better aggregate `71` start
- `72` barely hurts the current lane on aggregate
- `72` hurts the candidate lane a lot more:
  - `candidate72-candidate71 all_ex_root = +0.014872`
  - `candidate72-candidate71 leg = +0.083653`
- that stage-72 damage fully overwhelms the inherited candidate start advantage
- nonleg/arm tax is inherited and frozen, but the decisive aggregate give-back is the candidate lane’s extra `72` leg damage

## What `72` is hurting

### Final candidate `72` vs current `72`: biggest leg regressions

| leg_joint | delta(candidate72-current72) | current72 | candidate72 |
|---|---:|---:|---:|
| calf_l | 0.073529 | 0.281619 | 0.355148 |
| ball_l | 0.056393 | 0.248964 | 0.305358 |
| ball_r | 0.020095 | 0.242842 | 0.262936 |
| foot_l | 0.002722 | 0.409273 | 0.411996 |
| foot_r | -0.021797 | 0.305525 | 0.283728 |
| thigh_r | -0.027533 | 0.282815 | 0.255282 |
| calf_r | -0.032569 | 0.259140 | 0.226570 |
| thigh_l | -0.052374 | 0.340936 | 0.288562 |

### Final candidate `72` vs current `72`: worst leg windows

| leg_SIC | delta(candidate72-current72) | current72 | candidate72 |
|---|---:|---:|---:|
| SIC45 | 0.288106 | 0.208989 | 0.497094 |
| SIC48 | 0.273289 | 0.272011 | 0.545300 |
| SIC37 | 0.214321 | 0.292767 | 0.507089 |
| SIC47 | 0.205110 | 0.328796 | 0.533906 |
| SIC46 | 0.187589 | 0.380530 | 0.568118 |
| SIC36 | 0.186622 | 0.240836 | 0.427458 |
| SIC35 | 0.174651 | 0.157383 | 0.332034 |
| SIC21 | 0.170733 | 0.192716 | 0.363449 |
| SIC43 | 0.139228 | 0.243676 | 0.382905 |
| SIC03 | 0.136851 | 0.191233 | 0.328083 |
| SIC44 | 0.132862 | 0.220416 | 0.353278 |
| SIC22 | 0.102598 | 0.184627 | 0.287225 |

### Candidate lane: what `72` itself adds on top of candidate `71`

`candidate72 - candidate71` is broad damage, not just a tiny local tradeoff:

- worst joints:
  - `foot_l +0.154647`
  - `calf_l +0.126556`
  - `ball_r +0.103259`
  - `thigh_r +0.072250`
  - `thigh_l +0.064383`
  - `foot_r +0.059448`
- worst windows:
  - `SIC50 +0.414025`
  - `SIC48 +0.355679`
  - `SIC38 +0.352483`
  - `SIC37 +0.341303`
  - `SIC46 +0.289962`
  - `SIC47 +0.284915`
  - `SIC49 +0.280567`
  - `SIC39 +0.262300`

By contrast, `current72 - current71` is much smaller on aggregate and does not cause the lane to lose its lead against itself.

## Why the named hotspots stay better but aggregate still loses

This is the paradox the data resolves:

- `foot_l/ball_l@SIC12-15` stays better at final candidate `72` (`-0.226225` vs current `72`)
- `calf_r@SIC2-4` also stays better at final candidate `72` (`-0.097649`)
- but those are just two local windows inside the full leg average
- outside those windows, candidate `72` is much worse on:
  - `calf_l` overall
  - `ball_l` overall
  - `ball_r` overall
  - `foot_l` overall is actually slightly worse, despite the SIC12-15 hotspot win
  - large late mid-cycle windows (`SIC35-50`) dominate the give-back

Concrete example:

- `foot_l`
  - hotspot `SIC12-15`: candidate is much better (`-0.629414` average vs current)
  - but `SIC35-50`: candidate is much worse (`+0.286643` average)
  - so the hotspot win survives while the full-joint aggregate does not
- `calf_r`
  - hotspot `SIC2-4`: candidate is better (`-0.097649`)
  - but outside that window it has mixed behavior, and other joints regress enough to erase the leg-level advantage

So the right read is:

- `72` keeps a few local wins
- but it trades them for much broader leg damage
- aggregate `leg` and `all_ex_root` therefore move in the wrong direction

## Direct answers

### 1) `72` 的回退最早发生在什么时候？

- not inherited from candidate `71`
- it is introduced **inside `72`**, and the first visible aggregate flip is already at `s005`

### 2) `72` 到底伤了什么？

- biggest final direct-group damage is still leg-side
- the worst final aggregate drag comes from broader leg joints/windows, especially:
  - `calf_l`
  - `ball_l`
  - `ball_r`
  - `SIC35-48`, plus `SIC21-22` and `SIC03`
- the candidate lane’s own `72` stage also worsens almost every leg joint relative to candidate `71`

### 3) loss curve 支持哪种解释？

- primary: **early overshoot**
- secondary: **objective mismatch / wrong tradeoff**
- not supported as the main explanation:
  - pure late overfit
  - pure snapshot selection issue

Why not “just snapshot selection”:

- best post-start candidate snapshot is `s120`
  - `all_ex_root=0.121458`
  - `leg=0.296009`
- but current `s120` is still much better
  - `all_ex_root=0.106344`
  - `leg=0.264158`
- so even the best candidate post-start snapshot does not fix the aggregate comparison

### 4) 下一步最值得试的最小改动是什么？

Recommended order:

1. **lower LR `72`**
   - reason: the cross-lane aggregate flip happens by `s005`, so the cleanest lever is shrinking the early update magnitude
2. **gentler `72`**
   - if lowering LR alone is not enough, next try reducing `72`’s leg-side aggression without changing the whole downstream chain
3. **shorter `72` / early-stop**
   - only as a secondary lever
   - the current replay says a normal post-start early-stop still does not recover the aggregate lead
   - it helps only if `72` is made almost a no-op
4. **change `72` loss semantics / weights**
   - not the first move yet
   - current evidence already points to over-step before it points to a fundamentally wrong `72` objective definition

## One-sentence answer

为什么 `candidate 71 (lr=3e-4)` 明明赢了 current `71`，但到了 `72` 又把 aggregate 优势吐回去了？

- 因为 unchanged `72` 会在前 5 步就对这个更干净的 candidate `71` 起点发生明显 early overshoot：`total / dir_geo / dir_group_norm_leg / leg_align` 早期一起冲高，导致 broader leg windows（尤其 `calf_l`、`ball_l`、`ball_r` 和 `SIC35-48`）被拉坏；后面虽然 `foot_l/ball_l@SIC12-15` 和 `calf_r@SIC2-4` 这些局部 hotspot 还能保持更好，但它们已经不足以抵消更大范围的 leg aggregate 回退。
