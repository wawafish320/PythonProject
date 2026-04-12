# 2026-04-10 DSN-style auxiliary leg supervision research spec

> Status: archived / retired aux-family mechanism record
> Reader note: this aux / shared-trunk family did **not** become current repo mainline; any `recommend`, `default`, `ship`, `mainline`, or `current` wording below is historical family-local language only.
> Current entry points: `docs/posttrain_pipeline.md`, `docs/train_design/2026-04-11_donor_contract_basin_universal_debug_synthesis.md`, `docs/retired_directions/aux_shared_trunk_family/2026-04-11_aux_detach_downstream_reevaluation_record.md`

> Status: research spec / proposal only / no-train  
> Scope: `stage6 -> 70a -> 70b replace` redesign idea  
> Purpose: 用 training-only auxiliary supervision 替代 permanent expanded direct branch，在不破坏 baseline downstream contract 的前提下增强 trunk 的 leg-discriminative feature

## 1. Why this spec exists

当前主线困境已经足够清楚：

- `expanded direct branch` 一类方案的最大问题，不只是某个 metric 没调顺，而是它会把 **permanent structure / donor contract** 带进 downstream replace。
- 一旦 upstream donor contract 偏离 current baseline-compatible manifold，`70b replace` 的 handoff 就会变成主矛盾。
- 现有证据并不支持把 `"heavy-tail -> EMA 爆炸"` 这条具体机制升级为主线根因；但这不影响一个更高层次的判断：
  - **如果结构本身改变了 replace contract，就不应该继续把它当 mainline。**

因此，本 spec 的出发点不是：

- “如何继续修 expanded branch”

而是：

- **如何在保持 baseline inference architecture 不变的前提下，给 trunk 更强的 leg supervision。**

这正是 DSN-style companion objective 最有价值的地方：

- 辅助分支只在训练期存在；
- 推理期 / downstream export 时被完全丢弃；
- 最终模型仍是 baseline contract。

---

## 2. Core design principle

### 2.1 One-line summary

> 不再把 leg-focused capacity 做成 permanent physical split；只把它做成 training-time ghost head。

### 2.2 What we keep

- 保留 baseline 主干与主输出路径
- 保留 baseline `stage6 -> 70a -> 70b replace` downstream contract
- 保留 current inference/export structure

### 2.3 What we add

- 在 `stage6` 训练期，从 shared trunk / `direct_pose_head` 输入边界附近接一个 **lightweight auxiliary leg head**
- 只对 leg subset 施加 auxiliary objective
- auxiliary head 在训练结束后 **不参与 70a/70b**

### 2.4 What we explicitly forbid

- 不允许 auxiliary path 成为 permanent inference branch
- 不允许 auxiliary path 引入新的 downstream-required interface
- 不允许 auxiliary path 带来新的 persistent normalization state 依赖
- 不允许 auxiliary path 的输出回灌 main output contract

---

## 3. Corrected causal position

本方案成立，不依赖下面这条强机制被先证明：

- `expanded branch heavy-tail -> group norm EMA seed 被击穿 -> replace 爆炸`

当前更稳的因果表述应是：

1. permanent expanded branch 改变了 donor architecture / handoff contract  
2. changed contract 让 downstream replace 接住 donor 的难度显著上升  
3. 因此最优先的 intervention 应该是 **preserve baseline contract**  
4. 若还需要更强 leg signal，应优先使用 **training-only auxiliary supervision**

一句话：

- **DSN 方案的主要价值是 contract preservation，不是 pathology repair。**

---

## 4. Main hypothesis and sub-hypotheses

## 4.1 Main hypothesis

> 在 baseline architecture 不变的条件下，training-only auxiliary leg supervision 能让 trunk 学到更好的 leg-relevant feature，并把这种收益保留下游到 `70a -> 70b replace`。

## 4.2 H1: feature-learning hypothesis

- auxiliary leg objective 的梯度会进入 shared trunk
- trunk 的 intermediate representation 会变得更 leg-discriminative
- 即使删除 auxiliary head，main path 仍保留部分收益

## 4.3 H2: contract-preservation hypothesis

- 因为 inference/export 时 auxiliary head 被删除
- final checkpoint 的 effective architecture 仍等价于 baseline
- downstream replace 不再面对 expanded-branch transplant / absorb 问题

## 4.4 H3: downstream-transfer hypothesis

- 若 auxiliary supervision 真正帮助 trunk，而不是只让一个 side head 单独变好
- 则收益应体现在：
  - `stage6 native`
  - `70a native`
  - `70b replace`

而不是仅体现在训练期 auxiliary loss 本身下降。

---

## 5. Minimal model change

## 5.1 Structural sketch

```text
input
  -> shared trunk h
      -> main output path (unchanged)
      -> aux_leg_head(h)   # train-time only
```

其中：

- `shared trunk h`：优先选择 `direct_pose_head` 输入边界附近的 shared representation
- `main output path`：完全保持现有 baseline
- `aux_leg_head`：轻量、低容量、无额外 persistent contract

## 5.2 Recommended attachment point

首选 attachment point：

- `trunk output`
- 或者更具体地说：`direct_pose_head` 的输入边界

原因：

- 这是最接近 shared representation、且最容易把梯度回注到 trunk 的位置
- 同时避免把 auxiliary 头接在太晚的、已经 branch-specific 的位置

## 5.3 Recommended aux head form

优先级从高到低：

1. `Linear`
2. `Linear -> GELU -> Linear`
3. 非常小的 `MLP`

明确不推荐：

- 深 auxiliary tower
- 带额外 `BN/GN` 的 auxiliary tower
- 任何可能引入新 EMA/stat dependency 的设计

原则：

- auxiliary head 的职责是 **提供 companion gradient**
- 不是变成第二个主模型

---

## 6. Objective design

总损失定义：

```text
L_total = L_main + lambda_aux(t) * L_aux_leg
```

其中：

- `L_main`：现有 baseline 主目标，不改语义
- `L_aux_leg`：只看 leg subset 的 auxiliary objective
- `lambda_aux(t)`：随训练步数变化的 schedule，不建议常数全程硬压

## 6.1 Auxiliary target

`L_aux_leg` 应尽量与主任务语义一致，只是 supervision scope 更聚焦：

- leg subset 的 rotation / geodesic objective
- 或 leg-only 版本的 direct-pose objective

不建议：

- 重新造一个与主任务语义无关的新 proxy target
- 引入很难解释的 heuristic-only target

## 6.2 Lambda schedule

推荐使用：

- `warmup -> plateau -> decay-to-zero`

理由：

- 早期：帮助 trunk 快速长出 leg-sensitive feature
- 中期：维持足够梯度信号
- 后期：逐步把主导权交还 `L_main`，避免 auxiliary objective 绑架最终收敛

不推荐：

- 全程大常数
- 后期不衰减

---

## 7. Required implementation constraints

为保证这是一个 **compatibility-first** intervention，需要把约束写死：

1. final exported checkpoint 必须仍可按 baseline path 直接加载  
2. auxiliary modules 不参与 inference graph  
3. auxiliary modules 不写入 downstream required config contract  
4. auxiliary modules 不新增 planner / meas / replace-side boundary semantic  
5. auxiliary modules 不改变 main head output tensor schema  
6. auxiliary loss 不应依赖未来不在 inference 可用的信息  
7. auxiliary head 删除后，main path 行为必须可独立运行  

这几条中，最重要的是：

- **训练期存在，不等于部署期存在。**

---

## 8. Minimal experiment matrix

本 spec 的目标不是一次性做大矩阵，而是先做最小可解释验证。

## 8.1 Arm A: baseline control

- 现有 baseline `stage6`
- 无 auxiliary head
- 无 auxiliary loss

## 8.2 Arm B: sham head control

- 挂上 auxiliary head
- 但 `lambda_aux = 0`

目的：

- 排除“只是多了个头/多了点参数结构”带来的假阳性

## 8.3 Arm C: DSN auxiliary leg supervision

- 挂 auxiliary leg head
- `lambda_aux(t) > 0`
- 其余 training recipe 尽量与 baseline 一致

这是本 spec 的真正测试臂。

## 8.4 Optional Arm D: weak-vs-medium lambda

若 `Arm C` 有正信号，再开一个 very small follow-up：

- `C1`: weak lambda
- `C2`: medium lambda

不建议第一轮就开太大 sweep。

---

## 9. Evaluation pipeline

必须按完整链路看，而不是只看 `stage6` train loss。

## 9.1 Native checks

- `stage6 native`
- `70a native`

检查问题：

- trunk 是否真的学到更好的 leg-relevant feature
- native aggregate 是否退化

## 9.2 Downstream transfer checks

- `70a -> 70b replace`

这是本 spec 最关键的判据，因为该方案的核心 claim 是：

- **帮助 trunk，但不破坏 replace contract**

## 9.3 Whitebox / probe checks

建议保留以下轻量 probe：

- direct head boundary activation health
- `near_dead / low_diversity / scale_outlier`
- early `dir_leg_base / dir_nonleg_base`
- group-norm early seed trajectory

注意这里 probe 的角色是：

- **supporting evidence**

而不是：

- 决定主结论的唯一标准

---

## 10. Success criteria

只有同时满足 contract 与收益，才能判定成功。

## 10.1 Minimum success

- `70a native` 不劣于 baseline
- `70b replace` 优于 baseline
- auxiliary head 删除后收益仍保留

## 10.2 Strong success

- `stage6 native` 已显著改善 leg-side closure
- `70a native` 改善或至少不退化
- `70b replace` 在 aggregate 与 leg 指标上都有正增益
- whitebox/probe 未出现更严重 distribution pathology

## 10.3 Failure modes

### F1. Aux-only improvement

表现：

- auxiliary loss 好看
- 但 `70a/70b` 无改善

解释：

- aux head 学会了任务
- trunk 没有真正受益

### F2. Native up, replace flat/down

表现：

- `stage6/70a native` 好
- `70b replace` 不好

解释：

- trunk 变强了
- 但仍未形成更好的 downstream-compatible contract

### F3. Replace up, native down badly

表现：

- `70b replace` 变好
- 但 native 退化明显

解释：

- auxiliary objective 可能把 trunk 拉向过度 leg-biased representation

### F4. Sham ~= DSN

表现：

- `Arm B` 与 `Arm C` 接近

解释：

- companion objective 本身贡献不显著
- 观察到的提升可能不是 auxiliary supervision 带来的

---

## 11. How to interpret outcomes

## 11.1 If DSN arm wins cleanly

可支持的结论：

- current系统确实需要更强 leg gradient
- 但这个梯度不应通过 permanent branch 提供
- `training-only scaffold + baseline inference contract` 是有效方向

后续可进入：

- lambda schedule 微调
- aux head attach point 微调
- aux target 细化

## 11.2 If DSN arm helps stage6/70a but not 70b

可支持的结论：

- auxiliary supervision 确实能改善 upstream feature
- 但 main bottleneck 仍在 `70a -> 70b replace`

后续方向应偏向：

- replace-side absorbability redesign
- 而不是继续扩大 auxiliary head

## 11.3 If DSN arm completely无效

可支持的结论：

- 当前系统的主要瓶颈并不是 trunk 缺少 leg supervision
- companion objective 不是主矛盾

那就应把主线重新压回：

- replace boundary / handoff contract redesign

---

## 12. Why this is better than physical split

相对于 permanent expanded branch，本方案至少有三点结构优势：

1. **No transplant problem**  
   不需要把 donor 的额外永久结构移交给 downstream

2. **No new inference contract**  
   最终模型仍是 baseline architecture

3. **Cleaner falsifiability**  
   如果失败，可以明确归因给 companion objective 不足；不会混入 transplant compatibility 问题

一句话：

- **physical split 改的是模型身份；DSN auxiliary 改的是训练信号。**

---

## 13. Recommendation

当前建议把这个 spec 作为一个高优先级、低结构风险的 research line：

- 它不要求先解释清楚所有 pathology
- 它直接避开 permanent branch contract 风险
- 它可以最干净地回答：
  - “给 trunk 更强 leg gradient” 是否本身就足够有用

推荐执行顺序：

1. `baseline`
2. `sham aux-head`
3. `DSN aux-leg`
4. 如有正信号，再做小范围 lambda / attach-point follow-up

## 14. Final one-sentence bet

> 如果 current system 的核心需求真的是 “更强 leg supervision，但不能破坏 replace contract”，那么 DSN-style auxiliary leg head 很可能比 permanent expanded direct branch 更接近正确解。
