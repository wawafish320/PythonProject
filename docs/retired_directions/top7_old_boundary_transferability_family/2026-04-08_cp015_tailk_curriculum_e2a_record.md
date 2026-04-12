# 2026-04-08 CP015 tailk curriculum E2-A record

> Archived on 2026-04-12.  
> Current role: historical old-boundary `top7` transferability record inside the archived `E0/E1/E2A/E2C/E3A/A1-S1..S5` family, not current design policy.  
> Reader guidance: any `主线` / `推荐` / `默认下一步` / `canonical` wording below is preserved as family-local historical language.

> Last updated: 2026-04-08  
> Scope: **E2-A only** / `top3 -> top5 -> top7` matched curriculum / fixed replace context / deterministic first-forward  
> Machine summary: `debug_output/_tmp_cp015_tailk_curriculum_e2a_20260408/summary.json`

## 1. Scope / inherited conclusions

本轮只执行 **E2-A: top3 warmup -> top7 ramp**，直接继承以下结论，不重复证明：

- root cause 不在 planner semantics 主线
- root cause 不在 replace entry 外部 rollout state
- root cause 不在 `contacts_in_t`
- earliest semantic split 在 `direct_pose_head` boundary
- first-step split 最像 whole direct-branch contract mismatch
- `direct_pose_head` 是 earliest boundary / necessary anchor，但不是 standalone sufficient module
- weight-space 高 closure 需要 7-module direct branch joint contract
- baseline 的 7-module direct branch transplant 到 coadapt context 后可以 work，所以问题不是 “top7 impossible”
- E0 已显示：当前 top7 path 的 replace-transferability 在最早可用 stage6 exact `epoch013` 就已经偏差
- E0 已显示：`epoch014/015` 明显优于最终 checkpoint；最坏拐点在 `epoch015 -> stage6_tailfix`
- E1 已显示：`top3` support 比 `top7` support 更 replace-transferable，但不足以单独恢复 replace-normal
- `direct_pose_head.0` proxy 仍只是 supportive readout

本轮唯一要回答的是：

> top7 supervision 是否本身可行，只是需要一个 transfer-compatible curriculum/path，才能把 direct branch 训练到 replace-compatible basin？

---

## 2. E2-A design / controls / invariants

### 2.1 Fixed assay

与 E0/E1 完全对齐：

- **host replace context**: `coadapt_allrot_interface_bestlr_longer_4x_20260406`
- **transplant-compatible target**: 同 host + baseline replace 7-module direct-branch transplant
- **mode**: deterministic / single-step / first-forward
- **offset**: `45`
- **contacts**: baseline replace native same-entry `contacts_in_t`

### 2.2 Single new arm

本轮只新增一个 arm：

- `E2A-R`: matched curriculum arm
  - early warmup = `top3`
  - mid ramp = `top5`
  - late target = `top7`

### 2.3 Strict controls / invariants

保持不变：

- same basetrain pipeline
- same stage schedule surface（只改 `rot_local_tail_k`）
- same optimizer family / LR / weight decay
- same data / seed / init policy
- same stage6 tailfix config
- same 70a config
- same `save_fit_ckpt_epochs = 12-15`
- same `rot_local_tail_reduce = rank_linear_mix`
- same `rot_local_tail_uniform_mix = 0.4`
- same `rot_local_tail_rank_mix = 0.6`

### 2.4 No degraded variant

本轮 **没有**降级到 `top3 -> top7` late switch：

- `3 -> 5 -> 7` 可以用现有 schedule surface 干净表达
- 不需要新 scheduler machinery
- `degraded_e2a_variant = false`

---

## 3. Arm inventory

| arm | provenance | support schedule | basetrain `epoch014` | final `70a` |
|---|---|---|---|---|
| `E1-top7` | reuse existing | `7 -> 7 -> 7` | `models/cp015_phasecd_tailk_probe_20260331/exp_phase_DirectBranch_v1_d1_cp015_tailk7_rankmix_tw020_corridor_hold_tail15_phasea050_fixedsched_ep014center_control_denseckpt_seed2024_20260401/ckpt_epoch_014.pth` | `models/__tmp_cp015_tailk7_stage70a_from_tailfix_20260402/ckpt_last_WalkF_stage7_70a_lr3e4_from_cp015_tailk7_stage6tailfix_20260402.pth` |
| `E1-top3` | reuse existing | `3 -> 3 -> 3` | `models/cp015_phasecd_tailk_probe_20260331/exp_phase_DirectBranch_v1_d1_cp015_tailk3_rankmix_tw020_corridor_hold_tail15_phasea050_fixedsched_ep014center_control_denseckpt_seed2024_20260408/ckpt_epoch_014.pth` | `models/__tmp_cp015_tailk3_rankmix_tw020_stage70a_from_tailfix_e1_20260408/ckpt_last_WalkF_stage7_70a_lr3e4_from_cp015_tailk3_rankmix_tw020_stage6tailfix_e1_20260408.pth` |
| `E2A-R` | new curriculum arm | `3 -> 5 -> 7` | `models/cp015_phasecd_tailk_probe_20260331/exp_phase_DirectBranch_v1_d1_cp015_tailk357ramp_rankmix_tw020_corridor_hold_tail15_phasea050_fixedsched_ep014center_control_denseckpt_seed2024_20260408/ckpt_epoch_014.pth` | `models/__tmp_cp015_tailk357ramp_stage70a_from_tailfix_e2a_20260408/ckpt_last_WalkF_stage7_70a_lr3e4_from_cp015_tailk357ramp_stage6tailfix_e2a_20260408.pth` |

E2A 新增中间点：

- stage6 tailfix final: `models/__tmp_cp015_tailk357ramp_stage6_tailfix_e2a_20260408/lr3e4_e8x60_wd1e4_reinit1/ckpt_last_lr3e4_e8x60_wd1e4_reinit1_stage6_tailfix_tailk357ramp_e2a_20260408.pth`
- stage6 eval: `debug_output/_tmp_cp015_tailk_curriculum_e2a_20260408/stage6_tailfix/stage6_freerun/Walk_F_freerun_cycles.json`
- final `70a` eval: `debug_output/_tmp_cp015_tailk_curriculum_e2a_20260408/stage70a/eval_model_source/Walk_F_freerun_cycles.json`

---

## 4. Config diff table

| field | `E1-top7` | `E2A-R` | note |
|---|---:|---:|---|
| `rot_local_tail_k` | `7` | `3` | top-level / early effective support |
| `phase_b.core.rot_local_tail_k` | `7` | `3` | early warmup |
| `phase_c.core.rot_local_tail_k` | `7` | `5` | mid ramp |
| `phase_d.core.rot_local_tail_k` | `7` | `7` | late target |
| `rot_local_tail_reduce` | `rank_linear_mix` | `rank_linear_mix` | fixed |
| `rot_local_tail_uniform_mix` | `0.4` | `0.4` | fixed |
| `rot_local_tail_rank_mix` | `0.6` | `0.6` | fixed |
| `save_fit_ckpt_epochs` | `12-15` | `12-15` | fixed |
| `seed` | `2024` | `2024` | fixed |
| `epochs` | `15` | `15` | fixed |
| `lr` | `0.001` | `0.001` | fixed |

结论上，这轮的唯一新增实质变量就是：

- support curriculum path：`3 -> 5 -> 7`

---

## 5. Basetrain support schedule table

| stage | epoch range | `rot_local_tail_k` | meaning |
|---|---|---:|---|
| top-level | global default | `3` | early effective support |
| `phase_b_corridor_entry` | `6-9` | `3` | top3 warmup |
| `phase_c_corridor_hold` | `10-11` | `5` | top5 ramp |
| `phase_d_short_late_tail` | `12-15` | `7` | full top7 target |

对应 config：

- `config/exp_phase_DirectBranch_v1_d1_cp015_tailk357ramp_rankmix_tw020_corridor_hold_tail15_phasea050_fixedsched_ep014center_control_denseckpt_seed2024_20260408.json`

---

## 6. Stage6 tailfix / final `70a` result summary

### 6.1 E2A-R own-chain summary

| checkpoint | `out_direct` gap | `dir_base` gap | `dir_leg` gap | `dir_nonleg` gap | aggregate transfer score |
|---|---:|---:|---:|---:|---:|
| `E2A-R stage6 tailfix final` | `0.556670` | `1.024364` | `2.500365` | `0.705229` | `0.039916` |
| `E2A-R final 70a` | `0.461871` | `0.960091` | `2.021634` | `0.730569` | `0.156738` |

链内读法：

- `70a` 相比本 arm 的 stage6 确实继续修复了 transferability
- 但最终 `70a` 仍没有超过 `E1-top3` 的 `0.232733`

### 6.2 Native freerun group summary

| checkpoint | `all_ex_root` | `leg` | `nonleg` | `arm` | `else` |
|---|---:|---:|---:|---:|---:|
| `E2A-R stage6 tailfix final` | `0.494559` | `1.197252` | `0.342626` | `0.399557` | `0.208060` |
| `E2A-R final 70a` | `0.400640` | `1.069717` | `0.255975` | `0.289740` | `0.176167` |

这说明 `70a` 对 arm 内 native 指标是继续改善的；但主判据仍是 fixed replace-transferability。

---

## 7. Fixed transfer assay table

### 7.1 Final `70a` raw gaps

| arm | `out_direct` gap | `dir_base` gap | `dir_leg` gap | `dir_nonleg` gap |
|---|---:|---:|---:|---:|
| `E1-top7` | `0.469847` | `1.293004` | `2.010747` | `1.137816` |
| `E1-top3` | `0.475508` | `0.794442` | `1.931073` | `0.548684` |
| `E2A-R` | `0.461871` | `0.960091` | `2.021634` | `0.730569` |

### 7.2 Final `70a` closure ratios vs transplant-compatible target

| arm | `out_direct` closure | `dir_base` closure | `dir_leg` closure | `dir_nonleg` closure | aggregate |
|---|---:|---:|---:|---:|---:|
| `E1-top7` | `0.000000` | `0.000000` | `0.000000` | `0.000000` | `0.000000` |
| `E1-top3` | `-0.012050` | `0.385584` | `0.039624` | `0.517774` | `0.232733` |
| `E2A-R` | `0.016976` | `0.257472` | `-0.005414` | `0.357920` | `0.156738` |

### 7.3 Delta summary

`E2A-R - E1-top7`:

- `aggregate_transfer_score`: `+0.156738`
- `out_direct_closure_ratio`: `+0.016976`
- `dir_base_closure_ratio`: `+0.257472`
- `dir_leg_closure_ratio`: `-0.005414`
- `dir_nonleg_closure_ratio`: `+0.357920`

`E2A-R - E1-top3`:

- `aggregate_transfer_score`: `-0.075995`
- `out_direct_closure_ratio`: `+0.029025`
- `dir_base_closure_ratio`: `-0.128112`
- `dir_leg_closure_ratio`: `-0.045038`
- `dir_nonleg_closure_ratio`: `-0.159854`

最关键的读法是：

- `E2A-R` **明显优于 `E1-top7`**
- 但 **没有超过 `E1-top3`**
- 改善主要集中在 `dir_base` / `dir_nonleg`
- `dir_leg` 反而略差于 `E1-top7` 和 `E1-top3`

---

## 8. Replace-normality summary

### 8.1 Fixed readout result

| case | `plan/direct` sensitivity | `plan zero Δ` (deg) | `direct zero Δ` (deg) | `meas zero Δ` (deg) | label |
|---|---:|---:|---:|---:|---|
| host-native bad reference | `0.385554` | `0.145226` | `3.951890` | `0.085150` | `plan_compensatory` |
| baseline-transplant target | `0.385554` | `0.145226` | `3.951890` | `0.085150` | `plan_compensatory` |
| `E1-top7` | `0.385554` | `0.145226` | `3.951890` | `0.085150` | `plan_compensatory` |
| `E1-top3` | `0.385554` | `0.145226` | `3.951890` | `0.085150` | `plan_compensatory` |
| `E2A-R` | `0.385554` | `0.145226` | `3.951890` | `0.085150` | `plan_compensatory` |

### 8.2 Interpretation

这轮必须明确写成：

- `normality_probe_non_discriminative`

原因不是 “E2A-R 没改善” 这么简单，而是：

- 在当前口径下，probe **连 bad reference / transplant target 都分不出来**
- 五个 case 的数值和 label 都完全相同
- 所以本轮关于 “是否更正常地进入 replace” 的判断，**只能保守依赖 fixed transferability**
- 不能把这条 probe 硬解读成支持或反对 E2A-R

---

## 9. Proxy telemetry summary

### 9.1 `direct_pose_head.0` input-block statistics

| arm | `plan` /dim | `direct` /dim | `meas` /dim | `plan/direct` | `plan/meas` | `plan/(direct+meas)` |
|---|---:|---:|---:|---:|---:|---:|
| `E1-top7` | `2.011084` | `2.038038` | `1.948133` | `0.986775` | `1.032313` | `0.504515` |
| `E1-top3` | `2.009971` | `2.033001` | `2.028098` | `0.988672` | `0.991062` | `0.494933` |
| `E2A-R` | `2.009031` | `2.034409` | `2.024326` | `0.987525` | `0.992445` | `0.494989` |

### 9.2 Proxy deltas

`E2A-R - E1-top7`:

- `plan/direct`: `+0.000751`
- `plan/meas`: `-0.039869`
- `plan/(direct+meas)`: `-0.009526`

`E2A-R - E1-top3`:

- `plan/direct`: `-0.001146`
- `plan/meas`: `+0.001382`
- `plan/(direct+meas)`: `+0.000057`

### 9.3 Interpretation

proxy 的角色没有升级：

- 它继续只是 **supportive readout**
- `E2A-R` 的 proxy 比 `E1-top7` 更接近 `E1-top3`
- 但这种接近程度非常温和，不足以主导判读

---

## 10. Interpretation

### 10.1 Case label

本轮最稳妥的判法是：

- **Case 4 leaning Case 2**

也就是：

- transferability 上，curriculum/path-shaping **有帮助**
- 但 normality probe 在当前口径下完全不区分任何 arm
- 同时 `E2A-R` **没有明显超过 `E1-top3`**

### 10.2 Required judgments

1. **`E2A-R` 的 final `70a` 是否比 `E1-top7` 更 replace-transferable？**  
   **是。** aggregate transfer score 从 `0.000000` 提升到 `0.156738`。

2. **`E2A-R` 的 final `70a` 是否比 `E1-top3` 更 replace-transferable？**  
   **否。** `E2A-R = 0.156738`，`E1-top3 = 0.232733`。

3. **`E2A-R` 是否让 produced checkpoint 更正常地进入 replace？**  
   **本轮不能判成“是”。** 当前 normality probe 完全 non-discriminative，所以只能保守回答 **inconclusive / no positive evidence**。

4. **是否可以判断 “top7 viable under transfer-compatible path”？**  
   **还不能。**  
   原因不是 E2A-R 完全失败，而是它只部分修复：
   - 它已经优于 `E1-top7`
   - 但没有优于 `E1-top3`
   - normality 也没有新信息

5. **下一步最该开的是 `E2-B/C` 还是 `E3`？**  
   **优先 `E2-B/C`。**

### 10.3 Why not jump directly to E3

本轮不支持立刻转 `E3` 的原因是：

- ramp 本身已经带来了一定 transfer gain
- 说明 path-shaping 仍然有信息增益
- collapse 还没有强到足以说明 “support-ramp alone 完全无用，主要问题已是 widening 时的 co-adaptation allocation 崩塌”

更贴切的口径是：

- curriculum/path-shaping **helpful but insufficient**
- 更像要继续试 **readout-first / leg-first** 这种更定向的 path shaping

---

## 11. Next-step recommendation

下一步应优先：

- **`E2-B/C`**

优先顺序建议：

1. `E2-B`: readout-first / full-branch-later
2. `E2-C`: leg-first / nonleg expansion

理由：

- `E2A-R` 已经证明 widening 不是完全不可能
- 但 support-ramp 单独还不够把 donor 拉回 `E1-top3` 以上
- 下一步最该继续收缩的是 **joint contract formation path**
- 而不是立刻转去更强的 `E3` co-adaptation allocation 主线

一句话总结：

> E2-A 证明了 curriculum/path-shaping 对 top7 replace-transferability 确实有帮助，但在当前口径下它还只是部分修复；因此最优先 follow-up 仍是 `E2-B/C`，而不是直接转 `E3`。
