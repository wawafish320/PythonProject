# 2026-04-08 CP015 tailk support scope isolation E1 record

> Archived on 2026-04-12.  
> Current role: historical old-boundary `top7` transferability record inside the archived `E0/E1/E2A/E2C/E3A/A1-S1..S5` family, not current design policy.  
> Reader guidance: any `主线` / `推荐` / `默认下一步` / `canonical` wording below is preserved as family-local historical language.

> Last updated: 2026-04-08  
> Scope: **E1 only** / `top7` vs `top3` support isolation / fixed replace context / deterministic first-forward  
> Machine summary: `debug_output/_tmp_cp015_tailk_support_scope_isolation_e1_20260408/summary.json`

## 1. Scope / inherited conclusions

本轮只执行 **E1: support scope isolation (top7 vs top3)**，直接继承以下结论，不重复证明：

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
- E0 已显示：`direct_pose_head.0` input-block allocation proxy 只是 useful coarse readout，不是 leading indicator，也不是 root cause

本轮的唯一问题是：

> 在尽量保持其他训练条件不变的前提下，`effective direct-pose support scope` 是否是 upstream path 产出 replace-transfer incompatible direct branch 的强杠杆？

---

## 2. E1 design / controls / invariants

### 2.1 Fixed assay

与 E0 对齐的固定 replace-transfer assay：

- **host replace context**: `coadapt_allrot_interface_bestlr_longer_4x_20260406`
- **transplant-compatible target**: 同 host + baseline replace 的 7-module direct-branch transplant
- **mode**: deterministic / single-step / first-forward
- **offset**: `45`
- **contacts**: baseline replace native same-entry `contacts_in_t`

### 2.2 Arm design

只比较两个 arm：

- `E1-A`: current `top7` support path
- `E1-B`: matched-control `top3` support path

### 2.3 Strict controls / invariants

本轮强制保持一致：

- same basetrain pipeline
- same freerun stage schedule
- same optimizer family / LR schedule
- same data and seed policy
- same init policy
- same checkpoint cadence (`save_fit_ckpt_epochs=12-15`)
- same stage6 tailfix config
- same 70a config

唯一实质改动面是：

- `rot_local_tail_k`: `7 -> 3`

并且沿 schedule 内各阶段的 `rot_local_tail_k` 同步改为 `3`。

### 2.4 Reuse decision

- `top7` arm 直接复用现有 canonical chain，不重跑。
- 已存在的 historical `top3 control_denseckpt` **不复用**，因为它使用默认 `rot_local_tail_reduce=flat`，与当前 `top7 rank_linear_mix tw020` lane 不是 strict matched control。
- 因此补了一个最小 matched top3 basetrain config：  
  `config/exp_phase_DirectBranch_v1_d1_cp015_tailk3_rankmix_tw020_corridor_hold_tail15_phasea050_fixedsched_ep014center_control_denseckpt_seed2024_20260408.json`

### 2.5 Operational note

为让 matched top3 basetrain 在当前代码头部正常运行，本轮加入了一个**单点兼容补丁**：

- `train/models.py` 中 `MotionJointLoss` 的 `direct_pose_factorized_readout_enable` 访问改为 `getattr(..., False)` fallback

这个补丁只修复旧 config 在当前代码上的缺省属性访问，不改变 E1 的训练设计、pipeline 或 assay 口径。

---

## 3. Arm inventory

| arm | provenance | basetrain `epoch014` | stage6 tailfix | final `70a` |
|---|---|---|---|---|
| `top7` | reuse existing | `models/cp015_phasecd_tailk_probe_20260331/exp_phase_DirectBranch_v1_d1_cp015_tailk7_rankmix_tw020_corridor_hold_tail15_phasea050_fixedsched_ep014center_control_denseckpt_seed2024_20260401/ckpt_epoch_014.pth` | `models/__tmp_cp015_tailk7_stage6_tailfix_20260401/lr3e4_e8x60_wd1e4_reinit1/ckpt_last_lr3e4_e8x60_wd1e4_reinit1_stage6_tailfix_tailk7_stage6_tailfix_20260401.pth` | `models/__tmp_cp015_tailk7_stage70a_from_tailfix_20260402/ckpt_last_WalkF_stage7_70a_lr3e4_from_cp015_tailk7_stage6tailfix_20260402.pth` |
| `top3` | new matched control | `models/cp015_phasecd_tailk_probe_20260331/exp_phase_DirectBranch_v1_d1_cp015_tailk3_rankmix_tw020_corridor_hold_tail15_phasea050_fixedsched_ep014center_control_denseckpt_seed2024_20260408/ckpt_epoch_014.pth` | `models/__tmp_cp015_tailk3_rankmix_tw020_stage6_tailfix_e1_20260408/lr3e4_e8x60_wd1e4_reinit1/ckpt_last_lr3e4_e8x60_wd1e4_reinit1_stage6_tailfix_tailk3_rankmix_tw020_e1_20260408.pth` | `models/__tmp_cp015_tailk3_rankmix_tw020_stage70a_from_tailfix_e1_20260408/ckpt_last_WalkF_stage7_70a_lr3e4_from_cp015_tailk3_rankmix_tw020_stage6tailfix_e1_20260408.pth` |

对应 final `70a` eval：

- `top7`: `debug_output/_tmp_cp015_tailk7_stage70a_from_tailfix_20260402/eval_model_source/Walk_F_freerun_cycles.json`
- `top3`: `debug_output/_tmp_cp015_tailk3_rankmix_tw020_stage70a_from_tailfix_e1_20260408/eval_model_source/Walk_F_freerun_cycles.json`

---

## 4. Config diff table (`top7` vs `top3`)

| field | `top7` | `top3` | note |
|---|---:|---:|---|
| `rot_local_tail_k` | `7` | `3` | only intended top-level change |
| `phase_b.core.rot_local_tail_k` | `7` | `3` | matched schedule, scope-isolated |
| `phase_c.core.rot_local_tail_k` | `7` | `3` | matched schedule, scope-isolated |
| `phase_d.core.rot_local_tail_k` | `7` | `3` | matched schedule, scope-isolated |
| `rot_local_tail_reduce` | `rank_linear_mix` | `rank_linear_mix` | identical |
| `rot_local_tail_uniform_mix` | `0.4` | `0.4` | identical |
| `rot_local_tail_rank_mix` | `0.6` | `0.6` | identical |
| `save_fit_ckpt_epochs` | `12-15` | `12-15` | identical |
| `seed` | `2024` | `2024` | identical |
| `epochs` | `15` | `15` | identical |
| `lr` | `0.001` | `0.001` | identical |

结论上，这轮对照是干净的 matched-control：

- `top7` vs `top3` 的主差异就是 effective support scope
- 不是 `flat` vs `rankmix`
- 不是 schedule / optimizer / seed / cadence 差异

---

## 5. Final result summary table

| arm | transfer score | normality label | main readout |
|---|---:|---|---|
| `top7` | `0.000000` | `plan_compensatory` | final `70a` 仍是 fixed transfer assay 的 host-native bad endpoint |
| `top3` | `0.232733` | `plan_compensatory` | transfer 明显好于 `top7`，但 replace-normality probe 未显示改善 |

最直接的 top3-vs-top7 delta：

- `aggregate_transfer_score`: `+0.232733`
- `dir_base_closure_ratio`: `+0.385584`
- `dir_leg_closure_ratio`: `+0.039624`
- `dir_nonleg_closure_ratio`: `+0.517774`
- `out_direct_closure_ratio`: `-0.012050`
- replace-normality label: `plan_compensatory -> plan_compensatory`

---

## 6. Fixed transfer assay table

### 6.1 Raw gaps

| arm | `out_direct` gap | `dir_base` gap | `dir_leg` gap | `dir_nonleg` gap |
|---|---:|---:|---:|---:|
| `top7` | `0.469847` | `1.293004` | `2.010747` | `1.137816` |
| `top3` | `0.475508` | `0.794442` | `1.931073` | `0.548684` |

### 6.2 Closure ratios vs transplant-compatible target

| arm | `out_direct` closure | `dir_base` closure | `dir_leg` closure | `dir_nonleg` closure | aggregate |
|---|---:|---:|---:|---:|---:|
| `top7` | `0.000000` | `0.000000` | `0.000000` | `0.000000` | `0.000000` |
| `top3` | `-0.012050` | `0.385584` | `0.039624` | `0.517774` | `0.232733` |

### 6.3 Transfer readout

`top3` 的 fixed transferability 改善是真实存在的，但它不是全维恢复：

- **明显改善**：`dir_base`、`dir_nonleg`
- **轻微改善**：`dir_leg`
- **未改善**：`out_direct`

因此这轮更像：

- support scope 对 donor contract formation **有帮助**
- 但 scope isolation **不足以单独把 final 70a 修回 transplant-compatible basin**

---

## 7. Replace-normality summary

### 7.1 Fixed readout definition

本轮沿用一个最小 single-step contract-normality probe：

- `plan_over_direct_sensitivity`
- `plan_zero_delta_geolocal_deg`
- `direct_zero_delta_geolocal_deg`
- `meas_zero_delta_geolocal_deg`
- label: `nonplan_owned` / `mixed` / `plan_compensatory`

### 7.2 Result table

| arm | `plan/direct` sensitivity | `plan zero Δ` (deg) | `direct zero Δ` (deg) | `meas zero Δ` (deg) | label |
|---|---:|---:|---:|---:|---|
| `top7` | `0.385554` | `0.145226` | `3.951890` | `0.085150` | `plan_compensatory` |
| `top3` | `0.385554` | `0.145226` | `3.951890` | `0.085150` | `plan_compensatory` |

### 7.3 Normality readout interpretation

本轮**没有**看到 `top3` 在这条 fixed readout 上比 `top7` 更正常地进入 replace path。

而且需要明确说明：

- 在这次 E1 里，`host-native` bad reference、baseline-transplant target、`top7`、`top3` 在这条 probe 上都没有被拉开
- 因此这条 minimal normality probe 在本轮是 **non-discriminative**
- 它足以支撑“**没有观察到** replace-normality 改善”，但**不足以单独支持更细的 E1 follow-up**

所以 Q2 的最稳妥回答是：

- **在本轮 fixed readout 下，没有。**

---

## 8. Proxy telemetry summary (`direct_pose_head.0`)

### 8.1 Input-block statistics

| arm | `plan` /dim | `direct` /dim | `meas` /dim | `plan/direct` | `plan/meas` | `plan/(direct+meas)` |
|---|---:|---:|---:|---:|---:|---:|
| `top7` | `2.011084` | `2.038038` | `1.948133` | `0.986775` | `1.032313` | `0.504515` |
| `top3` | `2.009971` | `2.033001` | `2.028098` | `0.988672` | `0.991062` | `0.494933` |

### 8.2 Proxy delta (`top3 - top7`)

| metric | delta |
|---|---:|
| `plan` /dim | `-0.001113` |
| `direct` /dim | `-0.005037` |
| `meas` /dim | `+0.079965` |
| `plan/direct` | `+0.001897` |
| `plan/meas` | `-0.041251` |
| `plan/(direct+meas)` | `-0.009582` |

### 8.3 Proxy interpretation

E1 里这个 proxy 仍然有一定解释价值，但角色没有升级：

- 它更像 **supportive readout**
- 不是 leading indicator
- 更不是 root cause

这里的主要原因是：

- transfer 确实改善了
- 但 proxy 变化是温和的，而且方向也不是单一 clean monotonic
- 它可以作为辅助 concurrent readout
- 但不足以决定 “replace 是否更正常”

---

## 9. Interpretation

### 9.1 Four required judgments

1. **`top3` 是否比 `top7` 产生更 replace-transferable 的 final checkpoint？**  
   **是。** final `70a` 的 aggregate transfer score 从 `0.000000` 提升到 `0.232733`，改善主要集中在 `dir_base` 和 `dir_nonleg`。

2. **`top3` 是否让 produced checkpoint 更正常地进入 replace？**  
   **没有观察到。** 在本轮固定 normality probe 下，`top3` 与 `top7` 完全同标签、同数值；该 probe 在本轮也基本没有拉开 bad-vs-good 参考。

3. **support scope 应被判断为？**  
   **`partial lever`**。`top3` 明显改善 transferability，但没有同时改善 replace-normality，也没有把 final donor 直接拉回 transplant-compatible basin。

4. **下一步最该开什么？**  
   **直接做 `E2 curriculum/path-shaping`。**  
   这轮 scope isolation 已经回答了最关键的问题：support scope 有作用，但单独不够。

### 9.2 Causal strength wording

本轮最稳妥的口径是：

- support scope 是 **causal contributor / partial lever**
- 不是 standalone sufficient explanation
- 更不能写成 “top7 support scope 就是 root cause”

### 9.3 Why not continue finer E1 isolation first

不建议继续把预算优先砸在更细 E1 isolation 上，理由有三条：

1. `top3` 已经证明 support scope **确实有杠杆**
2. 但它没有把 final `70a` 修回 replace-normal
3. 当前最缺的信息已经不是 “scope 有无作用”，而是：
   - **什么 path / curriculum 能把 top7 或 widened support 带进 transfer-compatible basin**

---

## 10. E2 recommendation

下一步主线建议直接转：

- **E2 curriculum / path-shaping**

优先级排序：

1. `top3 -> top7` warmup / ramp
2. readout-first / full-branch-later
3. leg-first / nonleg expansion

原因不是因为 “scope 不重要”，而是因为 E1 已经显示：

- **scope 会改变最终 donor contract**
- 但 **scope isolation alone 还不够解释 replace incompatibility**

因此下一轮最该问的是：

> 什么 upstream path 能把 widened support 训练到 replace-compatible basin？
