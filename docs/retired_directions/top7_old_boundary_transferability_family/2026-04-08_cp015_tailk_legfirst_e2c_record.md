# 2026-04-08 CP015 tailk leg-first E2-C record

> Archived on 2026-04-12.  
> Current role: historical old-boundary `top7` transferability record inside the archived `E0/E1/E2A/E2C/E3A/A1-S1..S5` family, not current design policy.  
> Reader guidance: any `主线` / `推荐` / `默认下一步` / `canonical` wording below is preserved as family-local historical language.

> Last updated: 2026-04-08  
> Scope: **E2-C only** / fixed top7 support / leg-first -> nonleg expansion / fixed replace context / deterministic first-forward  
> Machine summary: `debug_output/_tmp_cp015_tailk_legfirst_e2c_20260408/summary.json`

## 1. Scope / inherited conclusions

本轮只执行 **E2-C: leg-first -> nonleg expansion**，直接继承以下结论，不重复证明：

- root cause 不在 planner semantics 主线
- root cause 不在 replace entry 外部 rollout state
- root cause 不在 `contacts_in_t`
- earliest semantic split 在 `direct_pose_head` boundary
- `direct_pose_head` 是 earliest boundary / necessary anchor，但不是 standalone sufficient module
- first-step split 最像 whole direct-branch contract mismatch
- E0 已显示：`epoch014/015` 明显优于最终 checkpoint；最坏拐点在 `epoch015 -> stage6_tailfix`
- E1 与 E2-A 已显示：当前能拿到的 gain 主要集中在 `dir_base` / `dir_nonleg`
- E1 与 E2-A 已显示：`dir_leg` 基本不动，`out_direct` 也几乎不改善
- normality probe 在当前 transplant assay 下可能完全不区分；若再次不区分，必须标为 `normality_probe_non_discriminative`

本轮唯一要回答的是：

> 能否通过一个明确对准 leg contract formation 的 upstream path，使 produced checkpoint 在 fixed replace assay 下首先抬升 `dir_leg`，同时尽量保留 E1 / E2-A 已拿到的 `dir_base` / `dir_nonleg` gain？

---

## 2. 为什么 E2-C 现在优先于 generic E2-B

`E1-top3` 和 `E2A-R` 的改善分布高度同构：

- 都主要改善 `dir_base` / `dir_nonleg`
- 都几乎不改善 `dir_leg`
- 都几乎不改善 `out_direct`

因此，当前最高优先级不是再开一个 generic readout-first 变体，而是先做一次最干净的 **leg-targeted basetrain path-shaping** 检查：

- 若 leg-first path 能先把 `dir_leg` 抬起来，则说明缺的 lever 确实更接近 leg contract formation
- 若 leg-first path 仍然抬不动 `dir_leg`，则说明当前这类 basetrain path family 的控制力已经接近上限，更该转向 `E3`

---

## 3. E2-C design / controls / invariants

### 3.1 Single new arm

本轮只新增一个 arm：

- `E2C-L`: matched leg-first arm

### 3.2 Fixed assay

与 E0 / E1 / E2-A 完全对齐：

- host replace context: `coadapt_allrot_interface_bestlr_longer_4x_20260406`
- transplant-compatible target: 同 host + baseline replace 的 7-module direct-branch transplant
- mode: deterministic / single-step / first-forward
- offset: `45`
- contacts: baseline replace native same-entry `contacts_in_t`

### 3.3 Strict controls / invariants

保持不变：

- same basetrain pipeline / optimizer / lr / wd / data / seed / init
- same `save_fit_ckpt_epochs = 12-15`
- same `rot_local_tail_reduce = rank_linear_mix`
- same `rot_local_tail_uniform_mix = 0.4`
- same `rot_local_tail_rank_mix = 0.6`
- same stage6 tailfix config
- same final `70a` config
- same seed policy

### 3.4 No support-width drift / no degraded variant

本轮没有把变量偷换成 support-width：

- `rot_local_tail_k = 7`
- `phase_b.core.rot_local_tail_k = 7`
- `phase_c.core.rot_local_tail_k = 7`
- `phase_d.core.rot_local_tail_k = 7`

并且：

- `degraded_e2c_variant = false`
- 现有 config plumbing 已能干净表达 leg-first -> nonleg expansion
- 不需要新 scheduler machinery

### 3.5 Reused config surface

E2-C 只复用现有 direct-pose loss surface：

- `direct_pose_loss_leg_split`
- `direct_pose_loss_group_norm_enable`
- `direct_pose_loss_group_norm_w_leg`
- `direct_pose_loss_group_norm_w_nonleg`

---

## 4. Arm inventory

| arm | provenance | schedule | basetrain `epoch014` | final `70a` |
|---|---|---|---|---|
| `E1-top7` | reuse existing | `7 -> 7 -> 7` | `models/cp015_phasecd_tailk_probe_20260331/exp_phase_DirectBranch_v1_d1_cp015_tailk7_rankmix_tw020_corridor_hold_tail15_phasea050_fixedsched_ep014center_control_denseckpt_seed2024_20260401/ckpt_epoch_014.pth` | `models/__tmp_cp015_tailk7_stage70a_from_tailfix_20260402/ckpt_last_WalkF_stage7_70a_lr3e4_from_cp015_tailk7_stage6tailfix_20260402.pth` |
| `E1-top3` | reuse existing | `3 -> 3 -> 3` | `models/cp015_phasecd_tailk_probe_20260331/exp_phase_DirectBranch_v1_d1_cp015_tailk3_rankmix_tw020_corridor_hold_tail15_phasea050_fixedsched_ep014center_control_denseckpt_seed2024_20260408/ckpt_epoch_014.pth` | `models/__tmp_cp015_tailk3_rankmix_tw020_stage70a_from_tailfix_e1_20260408/ckpt_last_WalkF_stage7_70a_lr3e4_from_cp015_tailk3_rankmix_tw020_stage6tailfix_e1_20260408.pth` |
| `E2A-R` | reuse existing | `3 -> 5 -> 7` | `models/cp015_phasecd_tailk_probe_20260331/exp_phase_DirectBranch_v1_d1_cp015_tailk357ramp_rankmix_tw020_corridor_hold_tail15_phasea050_fixedsched_ep014center_control_denseckpt_seed2024_20260408/ckpt_epoch_014.pth` | `models/__tmp_cp015_tailk357ramp_stage70a_from_tailfix_e2a_20260408/ckpt_last_WalkF_stage7_70a_lr3e4_from_cp015_tailk357ramp_stage6tailfix_e2a_20260408.pth` |
| `E2C-L` | new leg-first arm | `7 -> 7 -> 7 (leg-first -> nonleg expansion)` | `models/cp015_phasecd_tailk_probe_20260331/exp_phase_DirectBranch_v1_d1_cp015_tailk7_legfirst_nonlegexp_rankmix_tw020_corridor_hold_tail15_phasea050_fixedsched_ep014center_control_denseckpt_seed2024_20260408/ckpt_epoch_014.pth` | `models/__tmp_cp015_tailk7_legfirst_stage70a_from_tailfix_e2c_20260408/ckpt_last_WalkF_stage7_70a_lr3e4_from_cp015_tailk7_legfirst_stage6tailfix_e2c_20260408.pth` |

E2C-L 新增中间点：

- stage6 tailfix final: `models/__tmp_cp015_tailk7_legfirst_stage6_tailfix_e2c_20260408/lr3e4_e8x60_wd1e4_reinit1/ckpt_last_lr3e4_e8x60_wd1e4_reinit1_stage6_tailfix_tailk7_legfirst_e2c_20260408.pth`
- stage6 eval: `debug_output/_tmp_cp015_tailk_legfirst_e2c_20260408/stage6_tailfix/stage6_freerun/Walk_F_freerun_cycles.json`
- final `70a` eval: `debug_output/_tmp_cp015_tailk_legfirst_e2c_20260408/stage70a/eval_model_source/Walk_F_freerun_cycles.json`

---

## 5. Config diff table

| field | `E1-top7` | `E2C-L` | note |
|---|---:|---:|---|
| `rot_local_tail_k` | `7` | `7` | kept fixed at top7 |
| `phase_b.core.rot_local_tail_k` | `7` | `7` | kept fixed at top7 |
| `phase_c.core.rot_local_tail_k` | `7` | `7` | kept fixed at top7 |
| `phase_d.core.rot_local_tail_k` | `7` | `7` | kept fixed at top7 |
| `phase_a.core.direct_pose_loss_leg_split` | `null` | `true` | early leg-first split objective |
| `phase_a.core.direct_pose_loss_group_norm_w_nonleg` | `null` | `0.0` | leg-only warmup |
| `phase_b.core.direct_pose_loss_group_norm_w_nonleg` | `null` | `0.25` | early nonleg expansion / leg-dominant |
| `phase_c.core.direct_pose_loss_group_norm_w_nonleg` | `null` | `1.0` | full nonleg restored inside split objective |
| `phase_d.core.direct_pose_loss_leg_split` | `null` | `false` | late return to full-branch baseline objective |
| `save_fit_ckpt_epochs` | `12-15` | `12-15` | fixed |

结论上，这轮唯一新增实质变量就是：

- direct-pose **leg vs nonleg formation path**

而不是：

- support width

---

## 6. Basetrain leg-first schedule table

| stage | epoch range | top7 support | objective | leg / nonleg weight |
|---|---|---:|---|---|
| top-level | global default | `7` | fixed top7 target | n/a |
| `phase_a` | `1-5` | `7` | `leg_only` | `1.0 / 0.0` |
| `phase_b` | `6-9` | `7` | `leg_dominant` | `1.0 / 0.25` |
| `phase_c` | `10-11` | `7` | `split_full_nonleg` | `1.0 / 1.0` |
| `phase_d` | `12-15` | `7` | `full_branch_target` | split off / baseline objective |

对应 config：

- `config/exp_phase_DirectBranch_v1_d1_cp015_tailk7_legfirst_nonlegexp_rankmix_tw020_corridor_hold_tail15_phasea050_fixedsched_ep014center_control_denseckpt_seed2024_20260408.json`

---

## 7. Stage6 tailfix / final `70a` result summary

### 7.1 E2C-L own-chain fixed transfer summary

| checkpoint | `out_direct` gap | `dir_base` gap | `dir_leg` gap | `dir_nonleg` gap | aggregate transfer score |
|---|---:|---:|---:|---:|---:|
| `E2C-L stage6 tailfix final` | `0.565272` | `1.035263` | `2.382889` | `0.743885` | `0.039344` |
| `E2C-L final 70a` | `0.469371` | `0.997315` | `2.120300` | `0.754508` | `0.128023` |

链内读法：

- `70a` 相比 E2C-L 自身 stage6 确实继续改善了 aggregate transferability
- 但最终 `70a` 仍低于 `E1-top3 = 0.232733`，也低于 `E2A-R = 0.156738`
- stage6 相比 `E2A-R stage6` 有轻微 leg gap 改善（`2.382889 < 2.500365`），但这点改善没有 survive 到 final `70a`

### 7.2 Native freerun group summary

| checkpoint | `all_ex_root` | `leg` | `nonleg` | `arm` | `else` |
|---|---:|---:|---:|---:|---:|
| `E2C-L stage6 tailfix final` | `0.469013` | `1.165215` | `0.318483` | `0.365064` | `0.208383` |
| `E2C-L final 70a` | `0.396219` | `1.052135` | `0.254399` | `0.299105` | `0.148731` |

这说明：

- E2C-L 自身链内 native freerun 仍是改善的
- 但主判据仍然是 fixed replace-transferability，而不是 native freerun

---

## 8. Fixed transfer assay table

### 8.1 Final `70a` raw gaps

| arm | `out_direct` gap | `dir_base` gap | `dir_leg` gap | `dir_nonleg` gap |
|---|---:|---:|---:|---:|
| `E1-top7` | `0.469847` | `1.293004` | `2.010747` | `1.137816` |
| `E1-top3` | `0.475508` | `0.794442` | `1.931073` | `0.548684` |
| `E2A-R` | `0.461871` | `0.960091` | `2.021634` | `0.730569` |
| `E2C-L` | `0.469371` | `0.997315` | `2.120300` | `0.754508` |

### 8.2 Final `70a` closure ratios vs transplant-compatible target

| arm | `out_direct` closure | `dir_base` closure | `dir_leg` closure | `dir_nonleg` closure | aggregate |
|---|---:|---:|---:|---:|---:|
| `E1-top7` | `0.000000` | `0.000000` | `0.000000` | `0.000000` | `0.000000` |
| `E1-top3` | `-0.012050` | `0.385584` | `0.039624` | `0.517774` | `0.232733` |
| `E2A-R` | `0.016976` | `0.257472` | `-0.005414` | `0.357920` | `0.156738` |
| `E2C-L` | `0.001012` | `0.228683` | `-0.054484` | `0.336881` | `0.128023` |

### 8.3 Delta summary

`E2C-L - E1-top7`:

- `aggregate_transfer_score`: `+0.128023`
- `dir_base_closure_ratio`: `+0.228683`
- `dir_leg_closure_ratio`: `-0.054484`
- `dir_nonleg_closure_ratio`: `+0.336881`
- `out_direct_closure_ratio`: `+0.001012`

`E2C-L - E1-top3`:

- `aggregate_transfer_score`: `-0.104710`
- `dir_base_closure_ratio`: `-0.156901`
- `dir_leg_closure_ratio`: `-0.094108`
- `dir_nonleg_closure_ratio`: `-0.180893`
- `out_direct_closure_ratio`: `+0.013061`

`E2C-L - E2A-R`:

- `aggregate_transfer_score`: `-0.028715`
- `dir_base_closure_ratio`: `-0.028789`
- `dir_leg_closure_ratio`: `-0.049070`
- `dir_nonleg_closure_ratio`: `-0.021039`
- `out_direct_closure_ratio`: `-0.015964`

---

## 9. `dir_leg`-focused interpretation

本轮最核心的问题是：`dir_leg` 有没有第一次被明确抬起来？

答案是否定的。

final `70a` 上：

- `E2C-L dir_leg gap = 2.120300`
- `E1-top7 dir_leg gap = 2.010747`
- `E1-top3 dir_leg gap = 1.931073`
- `E2A-R dir_leg gap = 2.021634`

对应 `dir_leg` closure 也同样更差：

- vs `E1-top7`: `-0.054484`
- vs `E1-top3`: `-0.094108`
- vs `E2A-R`: `-0.049070`

因此这轮不能判成 “首次出现实质性 leg closure improvement”。更准确的读法是：

- stage6 上 leg-first path 可能短暂给出了一点 leg-side lift
- 但 final `70a` 没有保住它
- 最终产物并没有比现有 arm 更 transfer-compatible 地进入 leg contract

---

## 10. Nonleg retention / giveback summary

相对 `E1-top3`，E2C-L 只保留了：

- `dir_base` closure retention: `0.593083`
- `dir_nonleg` closure retention: `0.650632`

并且发生了明确回吐：

- `dir_base` closure delta vs `E1-top3`: `-0.156901`
- `dir_nonleg` closure delta vs `E1-top3`: `-0.180893`

相对 `E2A-R`，E2C-L 也没有守住已有 nonleg gain：

- `dir_base` closure delta vs `E2A-R`: `-0.028789`
- `dir_nonleg` closure delta vs `E2A-R`: `-0.021039`

本轮 summary 判定：

- `unacceptable_nonleg_giveback = true`

也就是说，这次不是 “用一点 nonleg 给回去换来明显 leg gain”，而是：

- 没换到 leg gain
- 还丢了一部分已有的 `dir_base` / `dir_nonleg` gain

---

## 11. Replace-normality summary

### 11.1 Fixed readout result

| case | `plan/direct` sensitivity | `plan zero Δ` (deg) | `direct zero Δ` (deg) | `meas zero Δ` (deg) | label |
|---|---:|---:|---:|---:|---|
| host-native bad reference | `0.385554` | `0.145226` | `3.951890` | `0.085150` | `plan_compensatory` |
| baseline-transplant target | `0.385554` | `0.145226` | `3.951890` | `0.085150` | `plan_compensatory` |
| `E1-top7` | `0.385554` | `0.145226` | `3.951890` | `0.085150` | `plan_compensatory` |
| `E1-top3` | `0.385554` | `0.145226` | `3.951890` | `0.085150` | `plan_compensatory` |
| `E2A-R` | `0.385554` | `0.145226` | `3.951890` | `0.085150` | `plan_compensatory` |
| `E2C-L` | `0.385554` | `0.145226` | `3.951890` | `0.085150` | `plan_compensatory` |

### 11.2 Interpretation

这轮必须明确写成：

- `normality_probe_non_discriminative`

因为在当前口径下，这个 probe：

- 连 host-native bad reference / transplant-compatible target 都分不出来
- 对 `E1-top7` / `E1-top3` / `E2A-R` / `E2C-L` 也完全不区分
- spans 全为 `0`

所以本轮关于 “是否更正常进入 replace” 的判断，只能保守依赖：

- fixed transferability

不能把这条 probe 硬解读成支持或反对 E2C-L。

---

## 12. Proxy telemetry summary

### 12.1 `direct_pose_head.0` input-block statistics

| arm | `plan` /dim | `direct` /dim | `meas` /dim | `plan/direct` | `plan/meas` | `plan/(direct+meas)` |
|---|---:|---:|---:|---:|---:|---:|
| `E1-top7` | `2.011084` | `2.038038` | `1.948133` | `0.986775` | `1.032313` | `0.504515` |
| `E1-top3` | `2.009971` | `2.033001` | `2.028098` | `0.988672` | `0.991062` | `0.494933` |
| `E2A-R` | `2.009031` | `2.034409` | `2.024326` | `0.987525` | `0.992445` | `0.494989` |
| `E2C-L` | `2.010425` | `2.033732` | `2.023941` | `0.988540` | `0.993322` | `0.495463` |

### 12.2 Interpretation

proxy 仍然没有升级成主判据：

- E2C-L 的 proxy 与 `E1-top3` / `E2A-R` 都非常接近
- 但这种接近没有对应到更好的 `dir_leg` closure
- 因此 `direct_pose_head.0` 依然只是 supportive telemetry，不是 leading indicator，也不是 root-cause proof

---

## 13. Interpretation

本轮应明确判成：

- `Case 3`

理由是：

- `E2C-L` final `70a` **优于 `E1-top7`**，但没有优于 `E1-top3`，也没有优于 `E2A-R`
- `dir_leg` 没有出现明确改善，反而比三条比较臂都更差
- 为了这次 leg-first path，还发生了明确的 nonleg giveback
- normality probe 继续完全 non-discriminative，因此结论只能依赖 fixed transferability

因此，本轮不支持：

- **leg-targeted path-shaping is the missing lever**

更不支持：

- **top7 viable under leg-targeted transfer-compatible path**

更准确的说法是：

- 当前这类 basetrain path-shaping family 对 leg contract formation 的控制力仍然不足
- 仅靠这条 family，无法把 top7 导到一个比 `E1-top3` / `E2A-R` 更好的 replace-compatible basin

---

## 14. 下一步：为什么更应该转 `E3`

本轮的默认下一步应是：

- `E3`

而不是继续优先一个新的 basetrain-path `E2-B`，原因有三点：

- 已经做过 support-scope 变体（`E1-top3`）与 curriculum 变体（`E2A-R`），这次再加 leg-first 变体（`E2C-L`），三者都没有把 `dir_leg` 明确拉起来
- E2C-L 在 stage6 上的轻微 leg-side lift 没能 survive 到 final `70a`，说明问题更像是后续 co-adaptation / allocation 在把它重新拉回 non-transferable basin
- 既然本轮既没有拿到明确 leg gain，又伴随 nonleg giveback，那么再继续在同一 basetrain path family 内横向挪动，信息增益已经偏低

如果之后仍要回到 `E2-B`，也应该是：

- 一个 **leg-targeted** `E2-B`

但那更像是 `E3` 之后的回补分支，而不是当前最高优先级主线。

---

## 15. Direct answers

1. `E2C-L` 是否比 `E1-top7` 产生更 replace-transferable 的 final checkpoint？  
   是。`aggregate_transfer_score = 0.128023 > 0.000000`。

2. `E2C-L` 是否比 `E1-top3` 更好？  
   否。`0.128023 < 0.232733`。

3. `E2C-L` 是否比 `E2A-R` 更好？  
   否。`0.128023 < 0.156738`。

4. `E2C-L` 是否明确抬升了 `dir_leg`？  
   否。final `70a` 的 `dir_leg gap = 2.120300`，比 `E1-top7` / `E1-top3` / `E2A-R` 都更差。

5. 这个 leg gain 是否伴随了不可接受的 nonleg giveback？  
   是，而且更准确地说是：没有拿到 leg gain，同时发生了不可接受的 nonleg giveback。

6. 是否可以判断 `leg-targeted path-shaping is the missing lever`，或更强的 `top7 viable under leg-targeted transfer-compatible path`？  
   不能。本轮不支持这两个判断。

7. 下一步最该开的是 leg-targeted `E2-B` 还是 `E3`？  
   `E3`。
