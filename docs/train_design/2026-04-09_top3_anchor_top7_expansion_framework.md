# 2026-04-09 Top3-anchor / Top7-expansion experiment framework

> Last updated: 2026-04-09  
> Status: framework memo / post-E3-A pivot  
> Purpose: 把主问题从 “怎么把 top7 调顺” 改写为 “什么 donor contract 更适合 downstream replace，以及 replace 应如何接住 upstream expansion”
> Historical evidence bundle: `docs/retired_directions/top7_old_boundary_transferability_family/README.md`

## 1. Why this memo exists

到 `E3A-RF` 为止，当前实验画面已经出现了一个足够强、不能再回避的 pattern：

- `E1-top3` 是唯一明确有效的干预
- 所有 **late/full `top7` scope** 变体都差于 `E1-top3`
- `dir_leg` 在这些变体里始终没有形成稳定 closure，甚至持续恶化
- 新增的 path / curriculum / allocation 并没有把 top7 拉回 replace-compatible basin，反而呈现近似单调退化

因此，主问题不该再表述成：

- “还没找到对的 top7 ordering / tuning”

而应改写为：

- **当前 pipeline 里，什么样的 upstream donor contract 才是 downstream replace-compatible 的？**
- **如果 top7 需要更宽的语义范围，应该由 upstream 负责长成兼容形态，还是由 downstream replace 负责吸收这个 expansion？**

**Related A1 evidence chain**

这个 framework memo 现在不再只是 proposal；`Family A` 的第一轮已经在 `A1` 系列里落地，因此这里把已执行记录双引用挂上：

- `A1-S1` partial-transplant boundary scout  
  summary: `debug_output/_tmp_partial_transplant_boundary_a1s1_20260409/summary.json`  
  record: `docs/retired_directions/top7_old_boundary_transferability_family/2026-04-09_partial_transplant_boundary_a1s1_record.md`
- `A1-S2` mixed-contract absorbability scout  
  summary: `debug_output/_tmp_mixed_contract_a1s2_20260409/summary.json`  
  record: `docs/retired_directions/top7_old_boundary_transferability_family/2026-04-09_mixed_contract_a1s2_record.md`
- `A1-S3` replace-side absorb boundary scout  
  summary: `debug_output/_tmp_replace_absorb_boundary_a1s3_20260409/summary.json`  
  record: `docs/retired_directions/top7_old_boundary_transferability_family/2026-04-09_replace_absorb_boundary_a1s3_record.md`

它们目前给出的最直接补充是：

- `A1-S1` 支持：`anchor_only` 不够，shared head 更像 already compromised
- `A1-S2` 支持：`top7 nonleg` 更像 partially absorbable，但还不 decisive
- `A1-S3` 支持：plain replace-side split 仍不足以 clear absorb；最多只有对 `host nonleg out side` 的 weak lean

---

## 2. Current evidence summary

截至目前，可以把证据压缩成下面几条：

### 2.1 `top3` 不是普通 control，而是 current compatible anchor

`E1-top3` 是当前唯一被证明能：

- 提升 aggregate transferability
- 维持正向 `dir_leg` closure
- 同时不给出不可接受的 nonleg giveback

因此 `top3` 不应再被视为一个临时 baseline；它更像：

- **current pipeline 的 operating point**
- **current replace-compatible anchor contract**

### 2.2 问题不只是 path，而是 late/full `top7` contract

`E2-A`、`E2-C`、`E3-A` 虽然分别改了：

- support/path shaping
- leg/nonleg formation path
- head/readout allocation ordering

但它们有一个共同点：

- late phase 最终都回到了 **full `top7` scope**

而一旦回到这个状态，final `70a` 都重新朝更差的 replace-transferability 收缩。

### 2.3 当前最像的 failure mode：shared compromise contract

最符合数据的解释不是：

- “某个 schedule 还没调对”

而是：

- **shared head 同时服务 leg 与 nonleg 时，会形成一个 native-self-consistent 但 replace-incompatible 的 compromise representation**

### 2.4 已知不能直接押注的方向

当前已有信息表明，下面两条都不能作为主假设直接推进：

- **naive dedicated head**：已有负结果，不能简单假定“给 top7 一个独立 head 就会变好”
- **explicit transfer-compatibility scalar signal**：replace 是 multi-task，下游 compatibility 很难压缩成一个干净、稳定、可回传的单目标

---

## 3. Reframed main question

后续主问题建议正式改写成：

> 如何让 `top7` 成为对 `top3-compatible anchor` 的可吸收 expansion，而不是一个需要完全替代 current compatible contract 的 monolithic donor？

这个问题天然分成两半：

1. **Upstream / donor design**
   - basetrain 应该产出什么样的 contract？
   - shared head 应该承载什么，不应该承载什么？

2. **Downstream / replace adaptation**
   - replace 应该把 donor 的哪些部分视为“必须对齐的 anchor”
   - 哪些部分应视为“可渐进吸收的 expansion”

---

## 4. What the shared head should be

这是本 memo 最重要的结构性判断。

### 4.1 不应该是什么

shared head 不应该继续承担：

- full `top7` final semantics
- leg + nonleg 的完整 joint compromise
- transplant boundary 前的唯一语义载体

因为目前所有证据都指向：

- 一旦 shared head 承担 full `top7` compromise，replace compatibility 就会恶化

### 4.2 更合理的目标形态

shared head 更应该是：

- **anchor carrier**
- **small, stable, replace-compatible trunk**
- **只负责当前已验证兼容的公共 contract**

而 `top7` 多出来的那部分职责，应尽量由 shared head 外的 **private / residual expansion paths** 承担。

换句话说，更合理的结构语义应是：

```text
input
  -> shared anchor trunk
      -> anchor-aligned output path
      -> leg residual expansion path
      -> nonleg residual expansion path
```

其中：

- `shared anchor trunk` 负责保住 `top3-compatible` contract
- `leg residual expansion path` 只表达新增 leg-side expansion
- `nonleg residual expansion path` 只表达新增 nonleg-side expansion

### 4.3 一句话版本

- **shared-head 应该是 anchor carrier，不应该是 full top7 compromise carrier。**

---

## 5. Working hypothesis

当前最值得下注的主假设是：

> 不是 `top7` 本身不行，而是 `top7` 不能以 current shared-head monolithic contract 的形式进入 replace。

进一步拆开就是：

### H-A. Anchor hypothesis

- `top3` 对应的是一个 replace-compatible anchor basin
- downstream replace 目前只在这个 basin 周围建立过稳定 contract

### H-B. Expansion hypothesis

- `top7` 新增的信息不是不能学
- 但它不能直接覆盖 / 重写 anchor
- 它需要以 residual expansion 的方式附着在 anchor 上

### H-C. Downstream adaptation hypothesis

- downstream replace 不应要求 donor 自己完全长成最终兼容 contract
- replace 应该有能力吸收 donor 的 expansion，而不是只接受一种 monolithic donor shape

---

## 6. Current bets

既然用户已经明确提到 “现在存在一个 bet 某个方面”，这里把 bet 明写出来。

### Bet-1. `top3` 是 current anchor，而不是旧版本残留

如果这个 bet 对：

- 后续所有设计都应尽量 preserve `top3-compatible` contract

如果这个 bet 错：

- 后续 boundary / replace redesign 不会从保留 `top3` anchor 中受益

### Bet-2. top7 应以 expansion 形式出现，而不是覆盖 anchor

如果这个 bet 对：

- 最优 donor 形态应是 `anchor + residual expansion`

如果这个 bet 错：

- 说明真正问题不在 monolithic compromise，而在别的 interface / optimization source

### Bet-3. 下游 replace 比上游 basetrain 更适合承担 “吸收 expansion” 的责任

如果这个 bet 对：

- 优先级应放在 replace-side adaptation / transplant boundary redesign

如果这个 bet 错：

- 说明 upstream 仍需先显式地产生更结构化的 donor contract

---

## 7. Hard stop rules

为了避免继续在低信息量空间里打转，建议把 stop rule 先写死。

### 7.1 Allocation-family hard stop

`E3-B` 可以做，但它应被降级为：

- **allocation family closing falsifier**

而不是主希望臂。

若 `E3-B final70a` 满足：

- `aggregate < E1-top3`
- 且 `dir_leg` 仍不明显优于 `E1-top3`
- 且仍存在明显 `dir_base` / `dir_nonleg` giveback

则判定：

- **current `top7` scope + current shared-head architecture/training recipe 不具备 replace-compatibility**

在这个判定下，不再优先继续：

- `E3-C`
- `E3-D`
- `E4`

### 7.2 Monolithic-top7 hard stop

如果任何后续实验仍然保持：

- full `top7` late/full monolithic shared-head donor

那么除非它明确超过 `E1-top3`，否则不再视为主线成功候选。

---

## 8. New design direction

### 8.1 From monolithic donor to factorized donor

后续设计方向，不应再是：

- “如何让一个 full top7 monolithic donor 更 compatible”

而更应是：

- “如何把 donor factorize 成 anchor + expansion，并让 replace 有机会渐进吸收 expansion”

### 8.2 Proposed contract decomposition

建议把 donor contract 概念上拆成三层：

1. **Anchor contract**
   - 当前由 `top3` 所代表
   - replace-compatible
   - 应尽量稳定、窄、可复用

2. **Leg expansion**
   - 扩展到 `top7` 时新增的 leg-side burden
   - 不应直接压进 anchor latent

3. **Nonleg expansion**
   - 扩展到 `top7` 时新增的 nonleg-side burden
   - 同样不应直接把 anchor 覆盖掉

这三层在训练与 transplant 上都不应该被视为完全同质的对象。

---

## 9. New experiment families

后续框架不再按 `E1/E2/E3/E4` 的老思路单纯延长，而改成三大实验家族。

### Family A. Boundary experiments

目标：

- 先回答 transplant boundary 是否需要从 “7-module 整包” 改成 “anchor + expansion” 分层

关键问题：

- replace 到底是在拒绝 full donor，还是只是在拒绝 donor 的某一部分 expansion？

优先候选：

#### A1. Partial-transplant boundary grid

不再只测整包 transplant，而是测：

- `anchor only`
- `anchor + nonleg expansion`
- `anchor + leg expansion`
- `expansion only`

目的不是 sweep，而是判断：

- compatibility 主要卡在 leg expansion
- 还是卡在 nonleg expansion
- 还是只要 expansion 一进来就破坏 anchor

对应已执行记录：

- `A1-S1` 已经覆盖了最小 partial boundary scout，结论不是 “anchor only 就够”
- `A1-S2` 已经覆盖了 preserved-anchor mixed-contract scout，结论是 nonleg 侧有 partial absorbability signal
- `A1-S3` 已经把 nonleg 再拆成 `proj side / out side` 做了 tri-donor absorb boundary assay，当前仍不支持把 simple replace-side split 当成已解决主线

#### A2. Host-preserved sub-contract transplant

保留 host 某个子合同不替换，只换另一个子合同：

- 例如保留 host leg path，只替 nonleg
- 或保留 host nonleg，只替 leg

这类实验直接服务于：

- boundary 是否应从 full donor 改成 mixed donor-host composition

### Family B. Replace-side adaptation experiments

目标：

- 让 replace 自己去 bridge donor contract，而不是要求 donor 先天长成旧 basin

优先候选：

#### B1. Donor-aware replace adaptation

以 top7 donor 为起点，放宽 replace 的适配自由度，例如：

- 更长 adaptation horizon
- 更大的 trainable scope
- 更分阶段的 unfreeze
- 更 donor-aware 的 warmstart

注意：

- 这不是再去把 top7 donor 逼回 top3
- 而是让 replace 有机会吸收 expansion

#### B2. Anchor-preserving replace adaptation

replace 端显式保住 anchor，再逐步吸收 expansion：

- early replace 先锁 anchor-sensitive 部分
- late replace 再吸收 expansion-sensitive 部分

这类设计与 `E3-A/B` 的不同在于：

- `E3-A/B` 是 upstream allocation
- 这里是 downstream adaptation allocation

### Family C. Upstream donor redesign experiments

目标：

- 如果 boundary / replace 侧给出了明确信号，再回头让 basetrain 显式产出 `anchor + expansion` donor

优先候选：

#### C1. Anchor-preserving basetrain

让 basetrain 明确区分：

- anchor path
- expansion path

而不是 late phase full shared compromise。

#### C2. Residual top7 expansion basetrain

以 `top3` anchor 为主干，`top7` 只作为 residual add-on 学习。

这类实验的判据不是 native loss 是否最低，而是：

- produced donor 是否更适合 downstream replace 吸收

---

## 10. Priority order

在当前信息下，建议优先级如下：

### Priority-0

- 如果资源允许，做 `E3-B` 作为 allocation-family closing falsifier

### Priority-1

- 转向 **Boundary experiments**

因为它们最直接回答：

- donor contract 应该整体换，还是分层换
- 哪个 sub-contract 才是真正的 interface 冲突源

### Priority-2

- 做 **Replace-side adaptation experiments**

因为现有证据更支持：

- current replace recipe 在约束 donor
- 而不是 donor 只要多调几轮就能自然落进旧 basin

### Priority-3

- 只有在 Boundary / Replace-side 给出清晰方向后，再做 **Upstream donor redesign**

这样可以避免盲目重做 upstream architecture。

---

## 11. How to evaluate success

后续任何新家族实验，都不能只看 native freerun。

### 11.1 Primary readout

仍然必须以 fixed replace-transferability 为主：

- `out_direct`
- `dir_base`
- `dir_leg`
- `dir_nonleg`
- closure ratio
- aggregate transfer score

### 11.2 Strong success criterion

只有当某个新方案同时满足：

- 明显优于 `E1-top3`
- `dir_leg` 明确改善
- 没有不可接受的 nonleg giveback

才可视为真正突破 current anchor boundary。

### 11.3 Partial success criterion

如果某方案不能超过 `E1-top3`，但能清楚回答：

- boundary 应该怎么改
- replace 应该吸收哪一类 expansion

它仍然是高信息量成功。

---

## 12. What not to do next

在当前证据下，不建议把主线继续押在：

- 单纯的 `E3-C/D`
- 局部 `E4` tuning
- 再做一轮 “也许这次 top7 ordering 会对”

原因不是这些永远没用，而是：

- 现在它们的信息增益已经明显低于 boundary / replace-side experiments

---

## 13. Short answer to the design question

如果只用一句话回答 “shared-head 应该是什么形态”：

> shared-head 应该是一个小而稳的 `top3-compatible anchor trunk`，而不是 full `top7` compromise trunk；`top7` 应该以 leg/nonleg residual expansion 的形式附着其上，并由 downstream replace 决定如何吸收这些 expansion。

---

## 14. Immediate next-step recommendation

建议把下一阶段工作定义成：

1. **可选**：做 `E3-B` 作为 allocation-family closing falsifier  
2. **主线**：开一个新的 boundary / replace-side framework，而不是继续延长 upstream ordering family  
3. 明确把 `E1-top3` 视为 current compatible anchor  
4. 所有新设计都围绕：
   - preserve anchor
   - isolate expansion
   - downstream absorb expansion

结合已执行的 `A1-S1 / A1-S2 / A1-S3`，这条 recommendation 现在应更具体地理解为：

- `A1-S1/S2/S3` 已经基本完成了第一轮 `Boundary experiments` 的探路
- 因此“主线转向 boundary / replace-side framework”这句话，现在不应再读成“先去做最初级 A1 scout”
- 更准确的当前落点是：**在 A1 现有证据下，先往更早 boundary / stronger boundary guard 收缩，再决定是否值得进入更明确的 replace-side absorb-expansion design**

这个 memo 的实质含义是：

- 主线已经从 “寻找更好的 top7 basetrain trick”
- 转成了 “重新定义 donor contract 与 replace interface 的关系”
