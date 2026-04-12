# Basetrain Handoff Selector 设计（以 GRU / contact-plan 为主）

> Status: archived legacy upstream / handoff / control record
> Reader note: this file belongs to the old-boundary upstream-control investigation; any `current`, `default`, `canonical`, `recommend`, or `mainline` wording below is historical context, not present-tense repo policy.
> Current entry points: `docs/posttrain_pipeline.md`, `docs/train_design/2026-04-11_donor_contract_basin_universal_debug_synthesis.md`, `docs/retired_directions/legacy_upstream_handoff_control_family/README.md`

> Last updated: 2026-03-19
> 目标：将 `basetrain -> posttrain(Stage6)` 的 checkpoint 选点逻辑，从当前以 pose/free-run 全局指标为主，调整为更贴近 downstream 实际依赖的 `GRU/contact-plan` 锚点质量。

关联输入文档：
- `docs/Problems/active/2026-03-06_basetrain_to_posttrain_startline_diagnostic.md`
- `docs/Problems/active/2026-03-06_trainbase_stage6_presplit_only_scope.md`
- `docs/Problems/active/2026-03-06_trainbase_stage6_presplit_phaseA_checklist.md`
- `docs/posttrain_pipeline.md`
- `docs/trainbase_design/2026-03-02_trainbase_v2_core_patch_flow.md`

---

## 0) TL;DR（结论先行）

1. 当前问题重点不是继续讨论 `split`，而是 `basetrain` 的 selector 与 downstream handoff 目标不一致。
2. 现有 `best_free` 主要由 `GeoDriftSlopeProxy` 驱动，更像“pose/global scalar 更优”。
3. 但 Stage6 以及后续 posttrain 真正依赖的是 `contact-plan` 这条独立锚点链是否稳定、是否对下游有用。
4. 因此应新增一个并行 selector：`best_handoff`，按 `contact-plan + downstream proxy` 选点，而不是直接替换现有 `best_free`。
5. 不建议直接按 `plan_z` hidden state 选；应按 `contacts_plan` 的外显行为和 downstream proxy 指标选。

---

## 1) 问题定义

### 1.1 当前 basetrain 的选点逻辑

`train/training_MPL.py` 当前逻辑是：

- `best_teacher`：按 `GeoLocalDeg / KeyBoneGeoLocalDegMean` 选；
- `best_free`：按 `GeoDriftSlope` 选。

对应代码：

- `train/training_MPL.py:3219`
- `train/training_MPL.py:3232`

这意味着当前 ckpt selector 仍然主要围绕：

1. teacher 几何误差；
2. freerun 全局漂移斜率。

### 1.2 但 Stage6 handoff 直接使用 `ckpt_best_free_*`

已有诊断文档明确写明：

- Stage6 config 的 `ckpt_in` 直接指向 `ckpt_best_free_*`

对应：

- `docs/Problems/active/2026-03-06_basetrain_to_posttrain_startline_diagnostic.md:17`

因此当前真实 handoff 链路是：

`basetrain(best_free by GeoDriftSlope) -> Stage6`

### 1.3 downstream 真正依赖的不是单纯 pose，而是 contact-plan 锚点

`EventMotionModel` 中，`contacts_plan` 被设计为独立锚点，而不是 pose 附属量：

- `contacts_plan` 只看 `cond + plan_z`；
- `contacts_err = contacts_plan - contacts_meas`；
- `direct_pose_head` 直接消费 `contacts_plan`。

对应代码：

- `train/models.py:1301`
- `train/models.py:2314`
- `train/models.py:3477`
- `train/models.py:3480`

因此 downstream 真正用到的是：

1. `contact-plan` 是否稳定；
2. `contacts_err` 是否有判别力；
3. 这些信号是否能给 Stage6/70R/71/72/lambda 提供可用锚点。

---

## 2) 为什么当前 selector 会错配

已有 Phase A 文档已经把问题定性为 `selector mismatch`，而不是单纯 Stage6 起跑坏掉：

- `docs/Problems/active/2026-03-06_trainbase_stage6_presplit_phaseA_checklist.md:1036`
- `docs/Problems/active/2026-03-06_trainbase_stage6_presplit_phaseA_checklist.md:1129`
- `docs/Problems/active/2026-03-06_trainbase_stage6_presplit_phaseA_checklist.md:1161`
- `docs/Problems/active/2026-03-06_trainbase_stage6_presplit_phaseA_checklist.md:1294`

其中关键结论是：

- `best_free` 更像“global scalar 更优”；
- `last` 更像“lower-body 风险更低”；
- 若 selector 仍只看 `GeoDriftSlopeProxy / global scalar`，会系统性低估 downstream 真正关心的收益。

这与 `docs/posttrain_pipeline.md` 当前链路的选择原则也是一致的：

- current chain selection is driven by aggregate leg-side behavior

对应：

- `docs/posttrain_pipeline.md:75`
- `docs/posttrain_pipeline.md:79`

结论：

> 当前 basetrain selector 的优化目标，与 posttrain handoff 的真实目标不一致。

---

## 3) 新 selector 的设计原则

### 3.1 不直接按 `plan_z` hidden state 选

虽然问题本质与 GRU / contact-plan 相关，但不建议直接把 `plan_z` 作为 selector 指标：

1. `plan_z` 是隐状态，数值基底不稳定；
2. 不同 epoch / seed / hidden dim 下可比性差；
3. downstream 真正关心的是它的外显行为，而不是 hidden 本身。

因此 selector 应按：

- `contacts_plan` 的可观测输出；
- `contacts_err` 的质量；
- downstream proxy 指标；
- 全局 pose 指标只保留为 guardrail。

### 3.2 selector 应是 downstream-aware，而不是 pose-only

目标不是挑“最会 freerun 的 pose ckpt”，而是挑“最适合作为 posttrain 起点的 ckpt”。

因此新增 selector 的名字建议明确区分：

- `best_teacher`
- `best_free`
- `best_handoff`

其中：

- `best_teacher`：保留现有 teacher 语义；
- `best_free`：保留现有 free-run 漂移语义；
- `best_handoff`：专门用于 `basetrain -> Stage6` handoff。

---

## 4) 建议的 handoff selector 指标

## 4.1 一级指标：contact-plan 主指标

这些指标应作为 `best_handoff` 的主排序项。

### A. `ContactErrAbsMean`

含义：

- `|contacts_plan - contacts_meas|` 的平均误差

代码导出：

- `train/validate/run_freerun_cycles.py:6197`
- `train/validate/run_freerun_cycles.py:10312`

意义：

- 衡量 `contact-plan` 与当前闭环 meas 的一致性；
- 对 event-clock、lambda reliability、direct hint 都有直接意义。

### B. `ContactPlanGtAbsMean`

含义：

- `contacts_plan` 相对 GT contacts 的误差

代码导出：

- `train/validate/run_freerun_cycles.py:6200`
- `train/validate/run_freerun_cycles.py:10310`

意义：

- 直接衡量 contact-plan 本身是否接近目标接触时序。

### C. phase/shift 对齐质量

看 `contacts_plan` 与 GT contacts 的 best circular shift 是否稳定。

代码位置：

- `train/validate/run_freerun_cycles.py:8572`

意义：

- 比单纯 pointwise MSE 更接近“相位是否对”的问题；
- 对周期动作 handoff 特别重要。

## 4.2 二级指标：downstream proxy

这些指标不是主因，但决定 handoff 是否对 Stage6/后链友好。

建议纳入：

1. `leg8_mean`
2. `SIC12-15 + {foot_l, ball_l}`
3. `calf_r global`
4. 必要时加 `calf_r SIC2-4`

这与已有 Phase A 文档中给出的 selector 对齐建议一致：

- `docs/Problems/active/2026-03-06_trainbase_stage6_presplit_phaseA_checklist.md:1161`

## 4.3 三级指标：全局 guardrail

这些指标继续保留，但只作为 guardrail / tiebreaker：

1. `GeoLocalDegWeighted`
2. `GeoDriftSlopeProxy`
3. `KeyBoneGeoLocalDegMean`

用途：

- 防止为了 contact-plan 指标牺牲整体 pose 质量太多；
- 但不再单独决定 handoff 最佳 ckpt。

---

## 5) 建议打分方式

为避免不同量纲直接线性相加，第一版建议使用 rank-based selector，而不是原值线性加权。

### 5.1 候选集

每个 run 不只比较：

- `best_free`
- `last`

还应比较：

- `best_free` 附近的 top-k epoch
- 例如：`best_free-2` 到 `best_free+2`

目的是避免 selector 只在单个 epoch 上做二选一。

### 5.2 rank-based handoff score

对每个候选 ckpt，在固定口径下跑统一 handoff eval：

- `pretrain_contact + affine_mix08`
- `cycle>=1`
- `drop_wrap=true`

然后定义：

```text
handoff_score =
  0.30 * rank(ContactErrAbsMean)
+ 0.25 * rank(ContactPlanGtAbsMean)
+ 0.20 * rank(leg8_mean)
+ 0.15 * rank(SIC12-15_footl_balll_mean)
+ 0.10 * rank(calf_r_global_mean)
```

lowest wins。

### 5.3 guardrail 条件

只有满足以下条件的候选，才允许进入排序：

1. `GeoLocalDegWeighted` 不劣于 baseline 太多（例如不超过 +3% ~ +5%）
2. `SourceMatchRate = 1.0`
3. contact source 固定为 `pretrain_contact + affine_mix08`
4. teacher / rounds / masking 口径固定

---

## 6) 与现有训练代码的关系

### 6.1 不是说模型没学 contact-plan，而是 selector 没按它选

`basetrain` 当前已经有 contact-plan supervision：

- `w_contact_plan`
- `contact_plan_bce`
- `contact_plan_mse`

对应代码：

- `train/models.py:5854`
- `train/models.py:5868`

这说明当前问题不是：

- “GRU/contact-plan 没训练起来”

而是：

- “虽然训练了，但 checkpoint selector 没用它来选点”

### 6.2 当前 `contacts_err` 也已经进入闭环语义

`contacts_err` 已被训练/推理链路消费，例如：

- reliability mode 可用 `contacts_err`

对应代码：

- `train/training_MPL.py:2593`
- `train/training_MPL.py:3593`

这进一步说明：

> 如果目标是 posttrain handoff 友好，selector 不该继续只围绕 pose/global drift。

---

## 7) 推荐实施顺序

## Phase 0：先不改主训练流程，只做离线 selector A/B

目标：

- 证明 `best_handoff` 比当前 `best_free` 更适合 downstream

做法：

1. 保持 `train/training_MPL.py` 原样；
2. 每个 run 额外收集：
   - `best_free`
   - `last`
   - `best_free` 附近 top-k epoch
3. 用统一 handoff eval 计算上述指标；
4. 比较：
   - 当前 `best_free`
   - 候选 `best_handoff`

只有当新 selector 稳定优于旧 selector，才进入 Phase 1。

## Phase 1：新增并行 selector，不替换旧 selector

在 `train/training_MPL.py` 中新增：

- `best_handoff`
- `best_handoff_payload`
- `ckpt_best_handoff_{run_name}.pth`

保持兼容：

- `best_teacher` 不动
- `best_free` 不动
- `best_handoff` 仅供 Stage6 / downstream A/B 使用

## Phase 2：Stage6 默认从 `best_handoff` 起跑

仅当以下条件都成立时，才建议把 Stage6 默认入口从 `best_free` 切到 `best_handoff`：

1. 多个 run / seed 上趋势稳定；
2. downstream full-chain 指标优于当前 `best_free` 路线；
3. 不引入全局 pose 质量明显回退。

---

## 8) 本轮明确不做的事

本方案当前不建议一起改：

1. 不继续把问题重新定义成 `split` 问题；
2. 不直接改 GRU 结构；
3. 不直接用 `plan_z` hidden state 做 selector；
4. 不在 selector 改造前继续做大量 Stage6-only 小步调权；
5. 不改 contact source / event-clock 主语义。

原因：

- 已有证据表明当前更像 `selector mismatch`；
- 先把 handoff 选对，比继续局部调 Stage6 更划算。

---

## 9) 验收标准

若新 selector 要进入主链，至少满足：

1. 相对当前 `best_free`：
   - `ContactErrAbsMean` 更低或持平；
   - `ContactPlanGtAbsMean` 更低或持平；
2. downstream proxy：
   - `leg8_mean`
   - `SIC12-15 + {foot_l, ball_l}`
   - `calf_r global`
   至少 2/3 改善；
3. 全局 guardrail：
   - `GeoLocalDegWeighted` 不出现明显回退；
4. 稳定性：
   - 不是单 run 偶然现象；
   - 在 control / experiment 两类上游都能复现趋势。

---

## 10) 建议新增产物

建议每个 basetrain run 目录里新增：

- `handoff_selector_candidates.json`
- `handoff_selector_summary.json`

至少包含：

- epoch
- `GeoLocalDegWeighted`
- `GeoDriftSlopeProxy`
- `ContactErrAbsMean`
- `ContactPlanGtAbsMean`
- `leg8_mean`
- `SIC12-15_footl_balll_mean`
- `calf_r_global_mean`
- `handoff_score`
- `selected_by`: `best_free | best_handoff | last`

---

## 11) 最终建议

当前更合理的方向不是：

- “继续把 Stage6 split-aware 前置一点”

而是：

- “把 basetrain 的 handoff selector 改成 contact-plan / downstream-aware”

更准确地说：

> 如果 basetrain 后续改动的目标是提升 posttrain 收益，那么 pose/global drift 应降级为 guardrail，
> `contact-plan usefulness` 应成为 handoff selector 的主目标。

