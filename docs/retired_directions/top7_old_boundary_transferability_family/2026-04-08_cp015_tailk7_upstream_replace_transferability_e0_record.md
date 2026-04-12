# 2026-04-08 CP015 tailk7 upstream replace-transferability E0 record

> Archived on 2026-04-12.  
> Current role: historical old-boundary `top7` transferability record inside the archived `E0/E1/E2A/E2C/E3A/A1-S1..S5` family, not current design policy.  
> Reader guidance: any `主线` / `推荐` / `默认下一步` / `canonical` wording below is preserved as family-local historical language.

> Last updated: 2026-04-08  
> Scope: analysis-only / no new training / deterministic single-step / fixed replace context

## 1. Scope / inherited conclusions

本轮只执行 **E0: checkpoint archaeology / transfer curve**，直接继承以下前提，不重复证明：

- root cause 不在 planner semantics 主线
- root cause 不在 replace entry 外部 rollout state
- root cause 不在 `contacts_in_t`
- earliest semantic split 在 `direct_pose_head` boundary
- `direct_pose_head` 是 earliest boundary / necessary anchor，但不是 standalone sufficient module
- 高 closure 需要 **7-module direct branch joint contract**
- baseline 的 7-module direct branch transplant 到 coadapt context 后可以 work，所以问题不是 “top7 impossible”，而是 **当前 upstream path 产出了 replace-transfer incompatible direct branch**

本轮固定 assay：

- **host context**: `coadapt_allrot_interface_bestlr_longer_4x_20260406`
- **transplant-compatible target**: 在同一 host 上 transplant baseline replace 的 7-module direct branch
- **inputs**: deterministic first-forward，offset=`45`
- **contacts**: 固定为 baseline replace native same-entry `contacts_in_t`

因此，本轮的 `transferability` 口径不是 native stage6/70a 指标，而是：

- donor ckpt 的 7-module direct branch transplant 到固定 replace host 后，
- 与 transplant-compatible target 之间的 gap / closure

---

## 2. Checkpoint inventory

### 2.1 Assayed checkpoints

| label | family | run | phase | epoch/step | included | path |
|---|---|---|---|---|---|---|
| `baseline_stage6_fromfresh` | top3_reference | `posttrain_pipeline_from_bestfree_20260317` | stage6 | final | yes | `models/__tmp_posttrain_pipeline_from_bestfree_20260317/stage6/ckpt_last_WalkF_stage6_fromfresh_20260317.pth` |
| `baseline_70a_fromfresh` | top3_reference | `posttrain_pipeline_from_bestfree_20260317` | 70a | final | yes | `models/__tmp_posttrain_pipeline_from_bestfree_20260317/70a/ckpt_last_WalkF_stage7_70a_fromfresh_20260317.pth` |
| `ep014center_70a_lr3e4` | top3_reference_historical | `ep014center_70a_lowlr_sweep_20260328` | 70a | final | yes | `models/__tmp_ep014center_70a_lowlr_sweep_20260328/lr3e4/ckpt_last_WalkF_stage7_70a_lr3e4_from_ep014center_stage6winner_20260328.pth` |
| `tailk7_stage6_exact_epoch013` | top7_current | `cp015_tailk7_rankmix_tw020_stage6_20260401` | stage6 | epoch013 | yes | `models/__tmp_cp015_tailk7_rankmix_tw020_stage6_20260401/epoch013/ckpt_last_epoch013_stage6_exact_tailk7_rankmix_tw020_20260401.pth` |
| `tailk7_stage6_exact_epoch014` | top7_current | `cp015_tailk7_rankmix_tw020_stage6_20260401` | stage6 | epoch014 | yes | `models/__tmp_cp015_tailk7_rankmix_tw020_stage6_20260401/epoch014/ckpt_last_epoch014_stage6_exact_tailk7_rankmix_tw020_20260401.pth` |
| `tailk7_stage6_exact_epoch015` | top7_current | `cp015_tailk7_rankmix_tw020_stage6_20260401` | stage6 | epoch015 | yes | `models/__tmp_cp015_tailk7_rankmix_tw020_stage6_20260401/epoch015/ckpt_last_epoch015_stage6_exact_tailk7_rankmix_tw020_20260401.pth` |
| `tailk7_stage6_tailfix_lr3e4_reinit1` | top7_current | `cp015_tailk7_stage6_tailfix_20260401` | stage6 | tailfix_final | yes | `models/__tmp_cp015_tailk7_stage6_tailfix_20260401/lr3e4_e8x60_wd1e4_reinit1/ckpt_last_lr3e4_e8x60_wd1e4_reinit1_stage6_tailfix_tailk7_stage6_tailfix_20260401.pth` |
| `tailk7_70a_from_tailfix` | top7_current | `cp015_tailk7_stage70a_from_tailfix_20260402` | 70a | final | yes | `models/__tmp_cp015_tailk7_stage70a_from_tailfix_20260402/ckpt_last_WalkF_stage7_70a_lr3e4_from_cp015_tailk7_stage6tailfix_20260402.pth` |

### 2.2 Available but not assayed

这些 checkpoint 已盘点，但本轮不纳入主 sweep：

- `tailk7_stage6_tailfix_lr1e4_reinit1`
- `tailk7_stage6_tailfix_lr3e4_reinit0`
- `tailk5_only_epoch012/013/014/015`
- `tailk5_rankmix_epoch013/014/015`
- `tailk5_rankmix_tw025_epoch013/014/015`
- `top3_control_denseckpt_rerun_epoch010~015_legacy`

其中 `top3_control_denseckpt_rerun` 这组为 **legacy payload (`config` only, no `posttrain_cfg`)**，本轮只做 inventory，不把它硬塞进统一 loader。

---

## 3. Assay definition

### 3.1 Primary transfer metrics

对每个 donor ckpt：

1. 将其 **7-module direct branch** transplant 到固定 `coadapt` host
2. 与 transplant-compatible target 比较，记录：
   - `out_direct gap`: `out_direct` tensor 的 normalized L2 gap
   - `dir_base gap`: donor-vs-target 的 local geodesic mean（all non-root joints）
   - `dir_leg gap`: donor-vs-target 的 local geodesic mean（leg joints）
   - `dir_nonleg gap`: donor-vs-target 的 local geodesic mean（non-leg, non-root joints）
3. 再用 host-native -> target 的原始 gap 做 closure：
   - `closure = 1 - candidate_gap / host_gap`

固定 host-native 到 target 的原始 gap 为：

| metric | host gap |
|---|---:|
| `out_direct` | `0.469847` |
| `dir_base` | `1.293004` |
| `dir_leg` | `2.010747` |
| `dir_nonleg` | `1.137816` |

### 3.2 Auxiliary proxy telemetry

`direct_pose_head.0` 的 proxy 口径与前序 audit 对齐：

- block statistic = **first-layer block Frobenius norm / sqrt(block_input_dim)**
- blocks:
  - `direct`
  - `plan`
  - `meas`
- ratios:
  - `plan/direct`
  - `plan/meas`
  - `plan/(direct+meas)`

---

## 4. Transfer curve summary

### 4.1 Top7 current path

| checkpoint | transfer score | `out_direct` closure | `dir_base` closure | `dir_leg` closure | `dir_nonleg` closure |
|---|---:|---:|---:|---:|---:|
| `tailk7_stage6_exact_epoch013` | `0.259345` | `0.192734` | `0.356846` | `-0.008731` | `0.496532` |
| `tailk7_stage6_exact_epoch014` | `0.472428` | `0.197016` | `0.609487` | `0.389773` | `0.693438` |
| `tailk7_stage6_exact_epoch015` | `0.460737` | `0.267946` | `0.551892` | `0.421332` | `0.601779` |
| `tailk7_stage6_tailfix_lr3e4_reinit1` | `-0.055965` | `-0.185658` | `0.027878` | `-0.169296` | `0.103217` |
| `tailk7_70a_from_tailfix` | `0.000000` | `0.000000` | `0.000000` | `0.000000` | `0.000000` |

### 4.2 Reference anchors

| checkpoint | family | transfer score | `out_direct gap` | `dir_base gap` | `dir_leg gap` | `dir_nonleg gap` |
|---|---|---:|---:|---:|---:|---:|
| `baseline_stage6_fromfresh` | top3 reference | `0.652214` | `0.326013` | `0.279600` | `0.592797` | `0.211881` |
| `baseline_70a_fromfresh` | top3 reference | `0.892159` | `0.018527` | `0.173541` | `0.235004` | `0.160251` |
| `ep014center_70a_lr3e4` | historical ref | `0.151502` | `0.359424` | `1.149432` | `1.664211` | `1.038129` |

### 4.3 Readout

最重要的事实有三条：

1. **当前 top7 path 在最早可见的 stage6 exact epoch013 就已经不“好”**  
   transfer score 只有 `0.259`，并不是 final 才坏。

2. **存在明显优于 final 70a 的中间 checkpoint**  
   `epoch014` (`0.472`) 和 `epoch015` (`0.461`) 都显著优于 final `70a` (`0.000`)。

3. **最大的坏转折出现在 `epoch015 -> stage6_tailfix`**  
   transfer score 单步下降 `-0.5167`，这是整个 top7 curve 上最清楚的 bad turn。

---

## 5. Proxy telemetry summary

### 5.1 Top7 current path

| checkpoint | `plan` /dim | `direct` /dim | `meas` /dim | `plan/direct` | `plan/(direct+meas)` |
|---|---:|---:|---:|---:|---:|
| `tailk7_stage6_exact_epoch013` | `2.043250` | `2.072380` | `1.985849` | `0.985943` | `0.503483` |
| `tailk7_stage6_exact_epoch014` | `2.036589` | `2.067830` | `1.979513` | `0.984892` | `0.503192` |
| `tailk7_stage6_exact_epoch015` | `2.040293` | `2.064855` | `1.978323` | `0.988105` | `0.504626` |
| `tailk7_stage6_tailfix_lr3e4_reinit1` | `2.013417` | `2.030278` | `1.962285` | `0.991695` | `0.504292` |
| `tailk7_70a_from_tailfix` | `2.011084` | `2.038038` | `1.948133` | `0.986775` | `0.504515` |

### 5.2 Reference anchors

| checkpoint | `plan/direct` |
|---|---:|
| `baseline_stage6_fromfresh` | `0.968220` |
| `baseline_70a_fromfresh` | `0.961782` |
| `ep014center_70a_lr3e4` | `0.982132` |

### 5.3 Proxy readout

- top7 path 内部的 `plan/direct` 变化幅度其实很小：约 `0.9849 ~ 0.9917`
- 但 **最坏的 transfer drop (`epoch015 -> tailfix`) 恰好也对应 proxy 最明显的坏向移动**
- baseline good references 处于更低的 `plan/direct` 区间（约 `0.962 ~ 0.968`）

所以这个 proxy **不是 root cause**，但也不是完全没信息：

- 它更像 **coarse concurrent readout / window locator**
- 不像强 leading indicator

---

## 6. Transferability vs proxy synchronicity judgement

### 6.1 Classification

本轮判为：

- **`synchronous_inflection`**

但需要加一句约束：

- **largest later bad-turn and largest proxy worsening happen on the same edge (`epoch015 -> tailfix`)**
- 同时，**transferability 在 earliest available stage6 checkpoint (`epoch013`) 就已经偏低**

因此更精确的说法是：

- **problem 已在早期形成**
- **proxy 对“后面的 bad turn edge”有同步定位价值**
- **但它没有把 earliest formation 本身提前暴露成一个更早的领先指标**

### 6.2 Plan weight 是否仍是 useful proxy

结论：

- **是，仍然是 useful proxy**
- 但角色应降级为：
  - **concurrent readout**
  - **useful coarse window locator**
  - **not root cause**

本轮不支持把它升级成：

- leading indicator
- standalone mechanism

---

## 7. Best checkpoint / bad-turn checkpoint / earliest divergence window

### 7.1 Best checkpoint

- **overall best assayed checkpoint**: `baseline_70a_fromfresh`
- **within current top7 path**: `tailk7_stage6_exact_epoch014`
  - transfer score = `0.472428`

### 7.2 Bad-turn checkpoint

- **sharpest bad-turn edge**: `tailk7_stage6_exact_epoch015 -> tailk7_stage6_tailfix_lr3e4_reinit1`
  - transfer delta = `-0.516702`
  - proxy `plan/direct` delta = `+0.003591`

### 7.3 Earliest divergence window

- **problem is already present by the earliest available top7 stage6 exact checkpoint**
  - earliest visible bad state: `tailk7_stage6_exact_epoch013`
- 但 **largest additional degradation** 发生在：
  - `epoch015 -> tailfix`

因此本轮最稳妥的两层结论是：

1. **formation**: at-or-before `epoch013`
2. **later collapse / worsening edge**: `epoch015 -> tailfix`

---

## 8. E1 recommendation

本轮建议：

- **下一步最该开的是 `E1 scope isolation`**

理由：

1. 当前 available E0 已经足够回答“问题不是 final-only 才出现”  
   它在 earliest visible stage6 exact 就已经存在。

2. 同时又看到一个明确的 later bad turn (`epoch015 -> tailfix`)  
   这说明问题不是单纯 final 70a 才崩，而是 upstream path 内部会进一步把 donor 推向更差 basin。

3. proxy 已经足够做 coarse window locator  
   再继续做更细的 checkpoint archaeology，边际收益大概率不如直接做 causal isolation。

### 8.1 E1 应该怎么落

建议优先测试：

- **同一 tailk7 pipeline，只改 direct pose effective support scope**
  - current `top7`
  - isolated `top3`

并继续沿用本轮固定的 **replace-transferability single-step assay** 做验收，而不是只看 native stage6 / 70a loss。

---

## 9. Final explicit answers

1. **E0 显示 transferability 是从什么时候开始形成问题的？**  
   - 已经在 earliest available `tailk7_stage6_exact_epoch013` 时偏低；  
   - 中间有 partial repair (`epoch014/015`)；  
   - 之后在 `epoch015 -> tailfix` 出现 sharp bad turn。

2. **是否存在比最终 checkpoint 更好的中间 checkpoint？**  
   - 有。`tailk7_stage6_exact_epoch014` 和 `epoch015` 都明显优于 final `70a`。

3. **`direct_pose_head.0` 的 block allocation proxy 是否对窗口定位有帮助？**  
   - 有，但只是 **coarse concurrent readout**；  
   - 它能帮助锁定 later bad-turn edge；  
   - 不足以单独解释 earliest formation。

4. **下一步最该开的是？**  
   - **`E1 scope isolation`**  
   - 不是先补更多 archaeology  
   - 也不是直接跳到 `E2 curriculum/path-shaping`
