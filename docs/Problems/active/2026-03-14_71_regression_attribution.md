# 71 regression attribution

Artifacts:

- runner: `tools/run_71_regression_attribution.py`
- summary JSON: `debug_output/_tmp_71_regression_attribution_20260314/summary.json`
- readable summary: `debug_output/_tmp_71_regression_attribution_20260314/summary.md`
- lower-LR follow-up runner: `tools/run_71_lowlr_sweep.py`
- lower-LR follow-up summary: `debug_output/_tmp_71_lowlr_sweep_20260314/summary.json`
- lower-LR follow-up readable summary: `debug_output/_tmp_71_lowlr_sweep_20260314/summary.md`

Scope guard:

- raw `70b` is not the main chain here; it is diagnostic-only
- the active comparison is `current 70R -> 71` vs `candidate lowdrift 70R -> 71`
- `71` semantics stayed unchanged; this round only added replay snapshots at `s000/s020/s060/s120/s180`
- eval contract is model-source only; strict was not needed because replay finals matched the existing reference finals exactly

## Short conclusion

- lowdrift replace + candidate `70R` is genuinely better at `71` start:
  - `all_ex_root: 0.158235 -> 0.130926`
  - `leg: 0.556049 -> 0.349263`
- but the unchanged `71` recipe spends that advantage almost immediately:
  - by `s020`, candidate is already worse than current on
    - `all_ex_root: 0.135698 vs 0.122295`
    - `leg: 0.376105 vs 0.353883`
- the final loss is mostly **not** inherited from candidate `70R`; it is mainly that candidate `71` fails to realize the large `71`-stage leg/all_ex_root gains that current `71` gets
- `calf_r@SIC2-4` and `foot_l/ball_l@SIC12-15` do stay better at final candidate `71`, but those are local wins; aggregate `leg` is dragged down by broader regressions on `calf_l`, `ball_r`, `foot_l`, `foot_r` and by worse leg windows around `SIC03`, `SIC08-13`, `SIC24-26`, `SIC34-37`
- best next minimal change: try **lower LR `71`** first, with **shorter `71` / early-stop** as the secondary lever; the replay damage appears in the first 20 steps, so shrinking the early update is more directly targeted than only trimming the tail

Update after the lower-LR sweep:

- that hypothesis is now validated
- keeping `71` semantics unchanged and only lowering LR is already enough to beat current `71`
- best tested case is `lr=3e-4`
  - final `all_ex_root: 0.107064` vs current `71` `0.111911`
  - final `leg: 0.215044` vs current `71` `0.295473`
  - final `foot_l/ball_l@SIC12-15: 0.429449` vs current `71` `0.599272`
  - final `calf_r@SIC2-4: 0.099602` vs current `71` `0.440912`
- so the current evidence no longer says “`71` must be redesigned first”; it now says “the unchanged `71` objective was mainly over-stepping on the cleaner candidate `70R` start, and lower LR fixes most of that”

## Reference end-state table

| lane | DirectGeoLocalDeg | all_ex_root | leg | nonleg | arm | legs_main | arms_main | foot_l/ball_l@SIC12-15 | calf_r@SIC2-4 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| current `70R` | 0.158235 | 0.158235 | 0.556049 | 0.072222 | 0.082665 | 0.556049 | 0.082665 | 1.118483 | 0.613849 |
| candidate `70R` | 0.130926 | 0.130926 | 0.349263 | 0.083717 | 0.091849 | 0.349263 | 0.091849 | 0.860095 | 0.393019 |
| current `71` | 0.111911 | 0.111911 | 0.295473 | 0.072222 | 0.082665 | 0.295473 | 0.082665 | 0.599272 | 0.440912 |
| candidate `71` | 0.127787 | 0.127787 | 0.331611 | 0.083717 | 0.091849 | 0.331611 | 0.091849 | 0.540575 | 0.295644 |

Immediate read:

- candidate starts `71` with a large leg/all_ex_root advantage
- candidate keeps a fixed inherited nonleg/arm tax:
  - `nonleg: +0.011496`
  - `arm: +0.009184`
- final candidate `71` still wins the two named hotspots:
  - `foot_l/ball_l@SIC12-15: -0.058697`
  - `calf_r@SIC2-4: -0.145269`
- but it loses aggregate:
  - `all_ex_root: +0.015877`
  - `leg: +0.036138`

## 71 replay snapshots

Replay finals exactly reproduced the existing reference finals, so these snapshots are apples-to-apples with the already known `70R/71` endpoints.

| lane_snapshot | DirectGeoLocalDeg | all_ex_root | leg | nonleg | arm | legs_main | arms_main | foot_l/ball_l@SIC12-15 | calf_r@SIC2-4 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| current `s000` | 0.158235 | 0.158235 | 0.556049 | 0.072222 | 0.082665 | 0.556049 | 0.082665 | 1.118483 | 0.613849 |
| current `s020` | 0.122295 | 0.122295 | 0.353883 | 0.072222 | 0.082665 | 0.353883 | 0.082665 | 0.473506 | 0.408613 |
| current `s060` | 0.124932 | 0.124932 | 0.368718 | 0.072222 | 0.082665 | 0.368718 | 0.082665 | 0.952248 | 0.320382 |
| current `s120` | 0.112269 | 0.112269 | 0.297488 | 0.072222 | 0.082665 | 0.297488 | 0.082665 | 0.433427 | 0.535532 |
| current `s180` | 0.111911 | 0.111911 | 0.295473 | 0.072222 | 0.082665 | 0.295473 | 0.082665 | 0.599272 | 0.440912 |
| candidate `s000` | 0.130926 | 0.130926 | 0.349263 | 0.083717 | 0.091849 | 0.349263 | 0.091849 | 0.860095 | 0.393019 |
| candidate `s020` | 0.135698 | 0.135698 | 0.376105 | 0.083717 | 0.091849 | 0.376105 | 0.091849 | 0.783757 | 0.428210 |
| candidate `s060` | 0.134537 | 0.134537 | 0.369575 | 0.083717 | 0.091849 | 0.369575 | 0.091849 | 0.722985 | 0.198369 |
| candidate `s120` | 0.126309 | 0.126309 | 0.323293 | 0.083717 | 0.091849 | 0.323293 | 0.091849 | 0.536233 | 0.136361 |
| candidate `s180` | 0.127787 | 0.127787 | 0.331611 | 0.083717 | 0.091849 | 0.331611 | 0.091849 | 0.540575 | 0.295644 |

Key readout:

- current lane gets a huge early `71` win already at `s020`
  - `all_ex_root: 0.158235 -> 0.122295`
  - `leg: 0.556049 -> 0.353883`
- candidate lane does the opposite at `s020`
  - `all_ex_root: 0.130926 -> 0.135698`
  - `leg: 0.349263 -> 0.376105`
- candidate later recovers part of that damage, with its best snapshot at `s120`
  - `all_ex_root: 0.126309`
  - `leg: 0.323293`
- but even that best candidate snapshot is still worse than current final
  - vs current `s180`: `all_ex_root +0.014398`, `leg +0.027820`

## Loss curve read

Artifacts:

- plot: `debug_output/_tmp_71_regression_attribution_20260314/71_loss_curve_compare.png`
- text summary: `debug_output/_tmp_71_regression_attribution_20260314/71_loss_curve_summary.md`

Main observations:

- candidate `71` starts with clearly higher training loss than current
  - `total start20: 2.103968 vs 1.873939`
  - `dir_group_norm_leg start20: 1.075360 vs 0.805355`
- in the middle of training, candidate loss is not obviously worse, and is even a bit lower on `total`
  - `total mid20(steps 80-99): 1.904910 vs 1.950976`
- but that lower mid-train loss does **not** translate into better freerun eval; candidate `s060/s120` are still worse than current
- by late training, both lanes drift upward again
  - current `late20 total: 2.017227`
  - candidate `late20 total: 2.057502`
- `dir_nonleg_base` stays almost flat in both lanes, which matches the eval story that inherited nonleg/arm gap is mostly untouched by `71`

Interpretation:

- the loss curve does not look like simple catastrophic divergence
- instead it looks like **objective mismatch / over-aggressive early leg update**
- candidate lane pays a larger early leg-side normalization cost, then can reduce training loss later without recovering freerun quality
- so “just train longer” is not supported by the curve; the more plausible fix is a **gentler early update**, with `lower LR` ahead of `shorter 71`

## Lower-LR follow-up (`2026-03-14`, candidate `70R` only)

This follow-up keeps `71` semantics unchanged and only changes LR.

Setup:

- start ckpt: candidate `70R`
- eval: model-source only
- dense snapshots: `s000/s005/s010/s020/s040/s060/s120/s180`
- cases:
  - `lr=5e-4`
  - `lr=3e-4`

### End-state table

| lane_snapshot | DirectGeoLocalDeg | all_ex_root | leg | nonleg | arm | legs_main | arms_main | foot_l/ball_l@SIC12-15 | calf_r@SIC2-4 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| current `71` | 0.111911 | 0.111911 | 0.295473 | 0.072222 | 0.082665 | 0.295473 | 0.082665 | 0.599272 | 0.440912 |
| baseline candidate `71` (`lr=1e-3`) | 0.127787 | 0.127787 | 0.331611 | 0.083717 | 0.091849 | 0.331611 | 0.091849 | 0.540575 | 0.295644 |
| candidate `71`, `lr=5e-4` | 0.112792 | 0.112792 | 0.247261 | 0.083717 | 0.091849 | 0.247261 | 0.091849 | 0.561664 | 0.054138 |
| candidate `71`, `lr=3e-4` | 0.107064 | 0.107064 | 0.215044 | 0.083717 | 0.091849 | 0.215044 | 0.091849 | 0.429449 | 0.099602 |

Direct read:

- `lr=5e-4`
  - already repairs most of the early-overshoot problem
  - final vs current `71`:
    - `all_ex_root: +0.000881` (almost tied)
    - `leg: -0.048212` (better)
- `lr=3e-4`
  - fully flips the verdict
  - final vs current `71`:
    - `all_ex_root: -0.004846`
    - `leg: -0.080429`
    - `foot_l/ball_l@SIC12-15: -0.169823`
    - `calf_r@SIC2-4: -0.341310`

### Early snapshot behavior

This was the real test: does lower LR fix the first-20-step failure?

| case_snapshot | all_ex_root | leg | vs current `71` all_ex_root | vs current `71` leg |
|---|---:|---:|---:|---:|
| `lr=5e-4 s005` | 0.129126 | 0.339143 | 0.017216 | 0.043670 |
| `lr=5e-4 s010` | 0.126407 | 0.323845 | 0.014496 | 0.028372 |
| `lr=5e-4 s020` | 0.123257 | 0.306128 | 0.011346 | 0.010655 |
| `lr=3e-4 s005` | 0.126253 | 0.322981 | 0.014342 | 0.027508 |
| `lr=3e-4 s010` | 0.122985 | 0.304598 | 0.011074 | 0.009125 |
| `lr=3e-4 s020` | 0.119240 | 0.283534 | 0.007330 | -0.011939 |

Interpretation:

- baseline `lr=1e-3` flipped negative by `s020`
- `lr=5e-4` slows that damage but does not fully eliminate it by `s020`
- `lr=3e-4` is the first tested case that already beats current `71` on `leg` by `s020`
- by `s120` and `s180`, `lr=3e-4` is better than current `71` on both `all_ex_root` and `leg`

### Loss-curve confirmation

Lower LR did exactly what the loss-curve hypothesis predicted.

Start-20 averages:

| lane | total | dir_group_norm_leg | dir_leg_base | boundary_dir_geo |
|---|---:|---:|---:|---:|
| current `71` (`1e-3`) | 1.873939 | 0.805355 | 0.006704 | 0.007558 |
| candidate `71` (`1e-3`) | 2.103968 | 1.075360 | 0.006883 | 0.008310 |
| candidate `71` (`5e-4`) | 1.850478 | 0.821871 | 0.004526 | 0.008084 |
| candidate `71` (`3e-4`) | 1.772125 | 0.743518 | 0.003915 | 0.008003 |

This is the strongest evidence from the sweep:

- lower LR directly collapses the early leg-side normalized update
- `lr=3e-4` not only fixes the candidate-vs-baseline gap, it even drops below current `71` on `dir_group_norm_leg start20`
- so the original diagnosis was right: the candidate lane was not “under-trained”, it was being over-stepped early

## Candidate minus current by snapshot

This is the clearest answer to “when does the regression first appear?”

| snapshot | d_all_ex_root | d_leg | d_nonleg | d_arm | d_legs_main | d_foot_l/ball_l@SIC12-15 | d_calf_r@SIC2-4 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `s000` | -0.027310 | -0.206786 | 0.011496 | 0.009184 | -0.206786 | -0.258388 | -0.220830 |
| `s020` | 0.013403 | 0.022222 | 0.011496 | 0.009184 | 0.022222 | 0.310251 | 0.019597 |
| `s060` | 0.009605 | 0.000857 | 0.011496 | 0.009184 | 0.000857 | -0.229263 | -0.122012 |
| `s120` | 0.014040 | 0.025805 | 0.011496 | 0.009184 | 0.025805 | 0.102806 | -0.399172 |
| `s180` | 0.015877 | 0.036138 | 0.011496 | 0.009184 | 0.036138 | -0.058697 | -0.145269 |

Interpretation:

- at `s000`, candidate is clearly better on aggregate `all_ex_root/leg`
- by `s020`, candidate has already crossed to worse on aggregate
- so the regression is **introduced inside `71`, within the first 20 update steps**
- after that, candidate never regains the overall lead, even though some hotspots recover and later become better again

## 70R -> 71 gain decomposition

This separates inherited start differences from `71`’s own contribution.

| metric | inherited (`candidate70R-current70R`) | `71` gain gap (`(cand71-cand70R)-(cur71-cur70R)`) | final gap (`candidate71-current71`) |
|---|---:|---:|---:|
| DirectGeoLocalDeg | -0.027310 | 0.043187 | 0.015877 |
| all_ex_root | -0.027310 | 0.043187 | 0.015877 |
| leg | -0.206786 | 0.242925 | 0.036138 |
| nonleg | 0.011496 | 0.000000 | 0.011496 |
| arm | 0.009184 | 0.000000 | 0.009184 |
| legs_main | -0.206786 | 0.242925 | 0.036138 |
| arms_main | 0.009184 | 0.000000 | 0.009184 |
| foot_l/ball_l@SIC12-15 | -0.258388 | 0.199691 | -0.058697 |
| calf_r@SIC2-4 | -0.220830 | 0.075562 | -0.145269 |

This table is the core attribution:

- for `nonleg/arm`, the candidate loss is inherited and then frozen:
  - `71` does not repair it
- for `all_ex_root/leg`, inherited state is actually favorable for candidate at `s000`
- the decisive failure is the `71` gain gap:
  - current `71` gets a very large extra win from its `70R` start
  - candidate `71` gets only a small extra win, and even regresses early

Numerically:

- `all_ex_root`
  - inherited: `-0.027310` better
  - `71` gain gap: `+0.043187` worse
  - final: `+0.015877` worse
- `leg`
  - inherited: `-0.206786` better
  - `71` gain gap: `+0.242925` worse
  - final: `+0.036138` worse

So the main cause is **`71` under-gain / self-induced rollback**, not start-state inheritance.

## What `71` is hurting

Final candidate `71` vs current `71`, biggest leg regressions:

| leg_joint | delta(candidate-current) | current `71` | candidate `71` |
|---|---:|---:|---:|
| `calf_l` | 0.176958 | 0.274648 | 0.451606 |
| `ball_r` | 0.056374 | 0.214300 | 0.270674 |
| `foot_l` | 0.055202 | 0.408709 | 0.463911 |
| `foot_r` | 0.053937 | 0.291310 | 0.345247 |

Worst aggregate leg windows:

| SIC | delta(candidate-current) | current `71` | candidate `71` |
|---|---:|---:|---:|
| `SIC26` | 0.239443 | 0.091224 | 0.330667 |
| `SIC03` | 0.229487 | 0.298561 | 0.528049 |
| `SIC12` | 0.222226 | 0.398464 | 0.620689 |
| `SIC24` | 0.208647 | 0.190522 | 0.399169 |
| `SIC25` | 0.192599 | 0.214588 | 0.407187 |
| `SIC13` | 0.178620 | 0.280130 | 0.458750 |

## Why hotspot wins survive while aggregate loses

The local hotspot story is real:

- `calf_r@SIC2-4` stays better at final candidate `71`
  - `0.440912 -> 0.295644`
- `foot_l/ball_l@SIC12-15` also stays better
  - `0.599272 -> 0.540575`

But those wins are narrow and do not represent the whole leg distribution.

Examples of local hotspot wins:

- `calf_r`
  - `SIC01: -0.724893`
  - `SIC52: -0.695004`
  - `SIC80: -0.563922`
- `foot_l`
  - `SIC14: -0.607126`
- `ball_l`
  - `SIC14: -0.421790`
  - `SIC13: -0.322658`

Examples of broader losses that outweigh them:

- `calf_l`
  - `SIC24: +1.550570`
  - `SIC25: +1.605473`
  - `SIC08: +1.057305`
  - `SIC09: +1.144947`
- `foot_l`
  - `SIC08: +0.890244`
  - `SIC07: +0.792984`
  - `SIC15: +0.666208`
- `ball_r`
  - `SIC49: +0.904418`
  - `SIC59: +0.476025`
  - `SIC50: +0.454157`
- `foot_r`
  - `SIC54: +0.716778`
  - `SIC48: +0.553791`
  - `SIC34: +0.439062`

So the hotspot paradox is:

- candidate `71` improves a few specific windows that we were already watching
- but it simultaneously broadens error elsewhere on the leg manifold
- aggregate `leg` and `all_ex_root` care about the whole surface, not just the named hotspots

## Direct answers

### 1. 回退最早发生在什么时候？

不是 inherited deficit first。

- `s000` 时 candidate 还明显更好
- 到 `s020` 就已经翻成更差
- 所以最早回退发生在 **`71` 训练前 20 steps 内**

### 2. `71` 到底伤了什么？

- direct group:
  - `all_ex_root`
  - `leg`
  - inherited `nonleg/arm` 也没被修复
- hotspot/window/joint:
  - 最主要拖累来自 `calf_l`, `ball_r`, `foot_l`, `foot_r`
  - 以及 `SIC03`, `SIC08-13`, `SIC24-26`, `SIC34-37`
- 为什么热点继续变好但 aggregate 变差：
  - 因为 `calf_r@SIC2-4`、`foot_l/ball_l@SIC12-15` 只是局部窗口
  - 更大面积的 leg windows 同时回退，净效应仍然更差

### 3. candidate `71` 比 current `71` 更差，主要归因是哪类？

主要是 **`71` 自身更新造成的回退 / under-gain**。

- `all_ex_root`:
  - inherited 是 `-0.027310` better
  - 但 `71` gain gap 是 `+0.043187` worse
- `leg`:
  - inherited 是 `-0.206786` better
  - 但 `71` gain gap 是 `+0.242925` worse

补充：

- `nonleg/arm` 的差距主要是 inherited deficit
- 但最终输掉总体 verdict 的主因仍是 `71` 没守住 leg/all_ex_root 的 `70R` 起点优势

### 4. 下一步最值得试的最小改动是什么？

这一步现在已经有结论了：

1. **先把 `71` 改成 lower-LR**
   - 当前最好 tested case 是 `lr=3e-4`
   - 它已经优于 current `71`
2. `shorter 71` / `early-stop` 退到 secondary lever
   - 如果之后想再压 tail risk，可以在 `lr=3e-4` 基础上继续测
   - 但它不再是 primary fix
3. 暂时不需要先做结构性 redesign
   - 现有证据更支持“same `71`, smaller step”
   - 只有当 lower-LR lane 在更完整下游验证里失守，才需要升级到 preserve-first / targeted redesign

所以当前推荐 recipe 是：

- `candidate 70R -> 71(lr=3e-4, same semantics)` 作为新的优先继续验证对象
- snapshot 仍建议保留密集口径，至少保留 `0/5/10/20/40/60/120/180`

## Downstream continuation: `71(lr=3e-4) -> 72 -> lambda`

This follow-up has now been run.

Artifacts:

- runner: `tools/run_71_lowlr_to72_lambda.py`
- summary JSON: `debug_output/_tmp_71_lowlr_to72lambda_20260314/summary.json`
- readable summary: `debug_output/_tmp_71_lowlr_to72lambda_20260314/summary.md`

End-state table:

| lane | DirectGeoLocalDeg | all_ex_root | leg | nonleg | arm | legs_main | arms_main | foot_l/ball_l@SIC12-15 | calf_r@SIC2-4 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| candidate `71` lowlr | 0.107064 | 0.107064 | 0.215044 | 0.083717 | 0.091849 | 0.215044 | 0.091849 | 0.429449 | 0.099602 |
| current `72` | 0.112074 | 0.112074 | 0.296389 | 0.072222 | 0.082665 | 0.296389 | 0.082665 | 0.812663 | 0.288880 |
| candidate `72` | 0.121936 | 0.121936 | 0.298698 | 0.083717 | 0.091849 | 0.298698 | 0.091849 | 0.586438 | 0.191232 |
| current `lambda` | 0.112074 | 0.112074 | 0.296389 | 0.072222 | 0.082665 | 0.296389 | 0.082665 | 0.812663 | 0.288880 |
| candidate `lambda` | 0.121936 | 0.121936 | 0.298698 | 0.083717 | 0.091849 | 0.298698 | 0.091849 | 0.586438 | 0.191232 |

Key readout:

- the low-LR `71` win is real, but unchanged `72` gives much of it back
- candidate `72` vs candidate `71` lowlr:
  - `all_ex_root: +0.014872`
  - `leg: +0.083653`
  - `foot_l/ball_l@SIC12-15: +0.156989`
  - `calf_r@SIC2-4: +0.091629`
- candidate `72` vs current `72`:
  - `all_ex_root: +0.009863`
  - `leg: +0.002308`
  - `nonleg: +0.011496`
  - `arm: +0.009184`
  - but still better on the two watched hotspots:
    - `foot_l/ball_l@SIC12-15: -0.226225`
    - `calf_r@SIC2-4: -0.097649`
- `lambda` is neutral here:
  - candidate `lambda` vs candidate `72` is exactly unchanged on all listed direct metrics

Updated bottleneck:

- fixing `71` with lower LR was necessary and successful
- the first new downstream give-back point is now `72`
- so the next rational experiment should target `72`, not re-open `71`

## One-line answer

为什么 lowdrift replace 在 replace/70R 看起来更好，但到了 `71` 反而整体输给 current `71`？

因为 candidate `70R` 虽然把 `71` 的起点抬高了，但 unchanged `71` recipe 在前 20 steps 就把这部分 leg/all_ex_root 优势花掉了，而且只保留了局部 `calf_r` / `foot_l/ball_l` 热点收益，最终更广泛的 `calf_l`、`ball_r`、`foot_l`、`foot_r` 窗口回退盖过了这些局部改进。 
