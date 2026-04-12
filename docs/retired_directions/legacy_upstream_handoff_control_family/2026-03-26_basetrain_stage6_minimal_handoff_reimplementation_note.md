# CP015 upstream：basetrain -> Stage6 minimal handoff reimplementation note

> Status: archived legacy upstream / handoff / control record
> Reader note: this file belongs to the old-boundary upstream-control investigation; any `current`, `default`, `canonical`, `recommend`, or `mainline` wording below is historical context, not present-tense repo policy.
> Current entry points: `docs/posttrain_pipeline.md`, `docs/train_design/2026-04-11_donor_contract_basin_universal_debug_synthesis.md`, `docs/retired_directions/legacy_upstream_handoff_control_family/README.md`

> Last updated: 2026-03-26  
> 目的：把 exploratory worktree 中已经验证过、但不适合直接合并的 handoff selector 结论整理到主工作区，供后续 clean reimplementation 使用。

## TL;DR

本轮应沉淀到主工作区的，不是探索期脚本本身，而是以下已经验证清楚的语义：

- `run_basetrain_handoff_selector.py` 适合作为 `basetrain -> Stage6` 的 **candidate discovery / coarse prefilter**
- 它**不应**被描述为已经能最终选出最佳 `basetrain -> posttrain` handoff
- final exact handoff ranking 需要一个轻量但 exact 的 **Stage6-only probe gate**
- 主工作区建议按最小 contract 重新实现，不直接搬运 exploratory branch 里的临时代码和杂项实验脚本

## 已验证结论

### 1. 主杠杆在 upstream handoff，而不是继续压 downstream tuning

当前更值得投入的是 `basetrain -> Stage6` handoff 选点，而不是继续把主时间预算放在 downstream `71/72` recipe tuning 上。

更准确的判断是：

- downstream 在当前锁定语义下已进入明显的 diminishing returns
- 更大的剩余 gap 更像来自 upstream handoff quality
- 问题重点不是 late checkpoint 的 basetrain scalar 再压一点，而是能否稳定选到更 `Stage6-friendly` 的 checkpoint basin

### 2. `run_basetrain_handoff_selector.py` 的合理定位是 prefilter

这轮最重要的语义澄清之一是：

- `run_basetrain_handoff_selector.py` 可以负责 candidate discovery
- 它可以负责 locked-boundary 下的 basetrain-side coarse prefilter
- 它不适合单独承担 final exact ranking

因此主工作区里不应把它讲成：

> “已经能最终选出最优 basetrain -> posttrain handoff 的 selector”

更合适的表述是：

> “它提供候选池与 coarse reject，final exact handoff 由 Stage6-only probe gate 补判。”

### 3. final exact handoff 需要 `Stage6-only probe gate`

只看 broad `all_ex_root / leg / nonleg` mean 不够。

final exact handoff 最少还需要显式纳入：

- localized hotspot
- tail guardrail
- best / reject / near-tie 的 summary 语义

也就是说，主工作区里应该存在一个轻量的 `Stage6-only probe`，它不跑 full chain，但能在 canonical Stage6 语义下判断：

- 谁是 Stage6-only best
- 谁被 hotspot / tail guardrail 明显淘汰
- 哪些只是 near-tie，需要 exact downstream replay 再判

## 建议冻结的最小 contract

### A. init diagnostics

保留以下 init diagnostics，仅用于看 interface skew，不替代 final rank：

- `step1 leg_over_nonleg`
- `head20 leg_over_nonleg`

### B. Stage6-only gate 主表

主表至少包含：

- `all_ex_root`
- `leg`
- `nonleg`
- `foot_l/ball_l@SIC12-15`
- `calf_r@SIC2-4`

理由：

- 前三项保留 broad exit quality
- 后两项把当前最 downstream-sensitive 的 hotspot 直接纳入 gate
- 避免只看 broad mean，把 localized pocket 放掉

### C. tail guardrail

guardrail 至少包含：

- `leg true_p95`
- `nonleg true_p95`
- `arm true_p95`

如实现顺手，可带上：

- `leg true_trim_top10`
- `nonleg true_trim_top10`
- `arm true_trim_top10`

### D. summary 语义

最终 summary 至少必须能回答：

1. 谁是本轮 `Stage6-only` best
2. 谁被 hotspot / tail guardrail 明显淘汰
3. 哪些 candidate 只是 near-tie，需要 exact downstream replay 再判

## 推荐的实现分层

### Layer 0：candidate discovery

继续复用 `run_basetrain_handoff_selector.py` 的现有能力：

- `best_free`
- `best_teacher`
- `last`
- 如存在 dense checkpoints，则保留 mid-training 窗口（优先 `ep12-15`）

### Layer 1：basetrain-side coarse prefilter

`run_basetrain_handoff_selector.py` 继续承担：

- candidate discovery
- locked-boundary basetrain eval runner
- obvious broad degradation reject
- obvious hotspot concentration worsening reject

但这一层仍然只是 prefilter，不做 near-tie final rank。

### Layer 2：Stage6-only exact gate

新增或 clean reimplement 一个轻量 `Stage6-only probe gate`：

- canonical Stage6 config
- canonical eval contract
- 只跑 Stage6
- 输出最小 gate 指标与 guardrail
- 给出 `best / reject / near-tie`

### Layer 3：optional downstream replay

只有在 `Stage6-only` 仍 near-tie 时，再进入 exact downstream replay。

## 本次不建议直接合并的内容

以下内容不建议直接从 exploratory branch 搬到主工作区：

- 临时实验脚本
- exploratory replay / backfill 代码
- 只服务于本轮诊断的中间产物生成逻辑
- 尚未收敛的 selector 阈值实现细节
- 把 `run_basetrain_handoff_selector.py` 直接扩成 final selector 的尝试

原因：

- 对验证过程有帮助
- 但不适合作为主工作区长期维护入口
- 更适合在主工作区按最小 contract clean reimplementation

## 建议主工作区后续动作

1. 不直接搬运 exploratory code
2. 先把 `run_basetrain_handoff_selector.py` 定位冻结为 prefilter
3. 再 clean reimplement 一个最小 `Stage6-only probe gate`
4. 只冻结最小输出表头与 summary 语义
5. 等 `Stage6-only gate` 在更多窗口上稳定后，再决定是否把部分指标正式并回 basetrain-side selector

## 一句话结论

主工作区应实现的核心语义是：

- `run_basetrain_handoff_selector.py` = prefilter
- `Stage6-only probe gate` = final exact handoff ranker

而不是继续把两者混成一个“单脚本 final selector”。
