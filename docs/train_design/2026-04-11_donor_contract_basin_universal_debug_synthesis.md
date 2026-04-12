# 2026-04-11 donor-contract / basin / universal-debug synthesis

> Status: synthesis memo / debug reframing  
> Purpose: 把当前 `top7 / capacity / replace handoff` 问题从“局部调参失败”整理成一条更清晰的 debug 主线，并明确区分 **locked-contract 局部避险** 与 **通解候选**。

---

## 0. TL;DR

当前最准确的问题表述不是：

- `Stage6` / `direct-branch` 缺 branch-side parameter capacity，所以加头 / 加 branch / 加 aux 应该变好。

而是：

- 当前 `70a -> downstream replace` 更像一个 **donor-basin-dependent locked contract**。
- `top7` / widened support / extra branch / aux supervision 带来的额外自由度，经常落到 **shared interface variables** 上，改写 downstream replace 正在消费的隐式 donor contract。
- 因此更多 capacity / plasticity 不是天然有害；有害的是 **capacity landed on the wrong interface variables**。
- 如果目标是 debug 出通解，就不能把 `top3 anchor` / freeze / stronger boundary guard 当最终方案；它们只能作为 **diagnostic scaffolding**，用于定位 break boundary 和 current locked contract 的可行域。
- 但 `top3` 不应在 redesign 一开始就被 deprecate；它还应承担 **benchmark / acceptance gate** 角色：任何 universal corrector 至少要在 `top3 donor` 上持平当前 locked contract。
- 真正的通解方向应该回到 `universal downstream correction`：让 downstream correction 尽量不依赖 donor 内部 basin / parameter-space local geometry，而是转向 frozen-donor、behavior-space observable、fresh residual corrector。

---

## 1. Terminology / framing

### 1.1 Two meanings of "capacity problem"

这里必须拆开两个完全不同的含义：

1. **Parameter-count shortage**
   - 含义：branch-side 参数量不够，模型学不到需要的信息。
   - 预期：加 head / 加 branch / 加 aux / 加 allocation freedom 应该至少有一条稳定变好。
   - 当前证据：基本不支持。

2. **Useful plasticity / basin budget under a locked contract**
   - 含义：在现有 downstream locked contract 下，direct branch 只有有限的“可安全改变”的自由度；超过这个范围会进入 replace-incompatible basin。
   - 预期：更多自由度可能让 upstream donor 更 self-consistent，但更不 transfer-compatible。
   - 当前证据：更支持这一版。

### 1.2 Three classes of interventions

后续实验和文档应明确标注属于哪一类：

1. **Diagnostic scaffolding**
   - 例如 `top3 anchor`、freeze、boundary guard、partial transplant、moment-match guard。
   - 目标：定位接口边界 / 判断哪一层开始破 contract。
   - 不能当最终解法。

2. **Locked-contract local optimum**
   - 例如继续找能进入旧 replace-compatible basin 的 donor path。
   - 目标：在当前 locked replace 下找最稳局部路线。
   - 只能解决当前产品/当前 contract，不能自动称为通解。

3. **Universal-solution candidate**
   - 例如 frozen-donor behavior-space residual correction。
   - 目标：donor basin 改变时，同一个 downstream correction 仍成立。
   - 必须用 multi-donor / held-out donor 验证。

补充一个单独角色：

4. **Benchmark / acceptance gate**
   - `top3 donor` 不只是诊断锚点。
   - 它还是 universal redesign 的最低验收基线。
   - 若一个 universal corrector 在 `top3 donor` 上都打不过当前 locked contract，它更像换了一种 donor-dependence，而不是得到更强通解。

---

## 2. Evidence chain

### 2.1 Early handoff framing: selector / basin rather than late scalar

Historical evidence bundle:

- `docs/retired_directions/legacy_upstream_handoff_control_family/README.md`

早期已经把问题从单纯 late scalar 转向 handoff / basin：

- `docs/retired_directions/legacy_upstream_handoff_control_family/2026-03-26_basetrain_stage6_minimal_handoff_reimplementation_note.md`
  - 结论：downstream 在当前锁定语义下已进入 diminishing returns；更大的剩余 gap 更像 upstream handoff quality；重点不是 late checkpoint 的 basetrain scalar 再压一点，而是能否稳定选到更 `Stage6-friendly` 的 checkpoint basin。
- `docs/retired_directions/legacy_upstream_handoff_control_family/2026-03-26_basetrain_to_stage6_minimal_handoff_probe_and_export_contract.md`
  - 结论：不是“没有更好的 basin”，而是已有 `Stage6-friendly mid-training basin`，但 selector / proxy 选不稳、解释不稳。
  - 同文还强调：现有 proxy 能缩小候选池，但不能可靠区分 `mid-training Stage6-friendly basin` 与 late broad/tail failure 伪装出的候选。

这说明问题很早就不该被压缩成“继续提高某个 scalar 指标”。

### 2.2 Universal replace redesign memo: current replace is donor-local continuation

`docs/retired_directions/replace_redesign_and_falsifier_family/2026-04-02_cp015_replace_universal_downstream_correction_redesign.md` 是一份重要的 historical redesign prior，但不应再被读成当前 canonical 方向本身。

关键结论：

- 当前 `replace` 的真实建模对象不是“通用 arm correction”，而是 **donor-local parameter-space continuation**。
- 它天然依赖 donor 的局部 state geometry；一旦 donor basin 改变，即使 freerun 更好、梯度更大，下游 replace 仍可能显著失效。
- 当前问题应理解为 `70a -> downstream` stage interface 的结构性问题：它把 donor basin 当成隐式前提。
- 新目标不是“让 replace 更适应某类 donor 形态”，而是让 downstream correction 尽量不把 donor 内部形态当作输入。
- 推荐 v1：`frozen donor + behavior-space observable input + freshly initialized ArmResidualCorrector`。

这篇 memo 应当作为后续“通解 debug”的主锚点。

### 2.3 Closed-loop falsifier: local optizability / warmstart surgery not enough

`docs/retired_directions/replace_redesign_and_falsifier_family/2026-04-04_cp015_tailk7_replace_closed_loop_stability_falsifier.md` 把若干局部解释降级：

- `tailk7 baseline-style adapted warmstart` 仍没有进入 baseline low-plan basin，更支持问题在 donor-state / `70a` exit basin，而不是 warmstart surgery 本身。
- 如果保留 C，只能写成 **higher-order exit-basin / multi-step trajectory geometry**。
- 不能再写成：
  - raw70a zero-plan readiness；
  - replace step0/1 local optizability；
  - tailk7 raw70a 的 non-plan path 在 step0/1 上更难被优化出来。

这一步很重要：它排除了“只要 local replace entry 更好优化就能救”的强版本。

### 2.4 Top7 transferability chain: capacity / scope widening is not the missing lever

Historical evidence bundle:

- `docs/retired_directions/top7_old_boundary_transferability_family/README.md`

从 `E0 -> E3A`，负证据非常一致。

#### E0: bad state begins early

`docs/retired_directions/top7_old_boundary_transferability_family/2026-04-08_cp015_tailk7_upstream_replace_transferability_e0_record.md`

- 最早可见坏状态：`tailk7_stage6_exact_epoch013`。
- 最大 additional degradation：`epoch015 -> tailfix`。
- 结论：更像 formation at-or-before `epoch013` + later collapse / worsening edge，而不是 late final 才出现容量不够。

#### E1: top3 helps, but not full solution

`docs/retired_directions/top7_old_boundary_transferability_family/2026-04-08_cp015_tailk_support_scope_isolation_e1_record.md`

- `top3` 的 fixed transferability 改善真实存在。
- 明显改善：`dir_base`、`dir_nonleg`。
- 轻微改善：`dir_leg`。
- 未改善：`out_direct`。
- 结论：support scope 对 donor contract formation 有帮助，但 scope isolation 不足以单独把 final `70a` 修回 transplant-compatible basin。

这里已经和 “top7 缺容量” 相反：缩小 support scope 反而更 transfer-compatible。

#### E2A: curriculum helpful but insufficient

`docs/retired_directions/top7_old_boundary_transferability_family/2026-04-08_cp015_tailk_curriculum_e2a_record.md`

- `E2A-R` 比 `E1-top7` 更 replace-transferable。
- 但 `E2A-R` 没有明显超过 `E1-top3`。
- normality probe 当前 non-discriminative。
- 结论：curriculum/path-shaping 有帮助，但不足以证明 top7 viable under transfer-compatible path。

这说明 widening 不是完全不可做，但 current path 一 widen 就容易被拉离兼容 basin。

#### E2C: leg-first is not the missing lever

`docs/retired_directions/top7_old_boundary_transferability_family/2026-04-08_cp015_tailk_legfirst_e2c_record.md`

- `E2C-L` final `70a` 优于 `E1-top7`，但没有优于 `E1-top3`，也没有优于 `E2A-R`。
- `dir_leg` 没有明确改善，反而比三条比较臂都更差。
- 还发生了明确 nonleg giveback。
- 结论：不支持 `leg-targeted path-shaping is the missing lever`。

#### E3A: allocation ordering is not enough

`docs/retired_directions/top7_old_boundary_transferability_family/2026-04-08_cp015_tailk_allocation_e3a_record.md`

- `E3A-RF` 明显优于 `E1-top7`，但不如 `E1-top3`、`E2A-R`、`E2C-L`。
- `dir_leg` 不仅没有明确抬升，反而比 `E2A-R` / `E2C-L` 更差。
- 同时发生不可接受的 nonleg giveback。
- 结论：没有证据支持“只要 readout-first staged allocation，top7 就可行”。

### 2.5 Distribution pathology probe: downgrade, do not promote

`docs/retired_directions/top7_old_boundary_transferability_family/2026-04-09_replace_handoff_distribution_pathology_probe.md`

该文档应保留为 negative evidence / ruled-out-mainline：

- `step0/step1` raw loss 与 EMA seed 确实分开。
- 但 activation pathology 的共享形态不成立：
  - baseline 反而有更多 `near_dead / low_diversity / scale_outlier`；
  - donor 侧额外增加的主要只是 `heavy_tail`；
  - `E1-top3` 与 `notail` 的 dominant family / grad alignment 不一致。
- 结论：`overall distribution pathology support: False`。
- recommendation：`do_not_promote_distribution_pathology_as_mainline`。

因此不建议把 `robust normalization / clipping / branchwise rescale / whitening` 作为主线修正；最多保留极轻量 follow-up smoke。

### 2.6 Top3-anchor / Top7-expansion framework: correct problem rewrite

`docs/train_design/2026-04-09_top3_anchor_top7_expansion_framework.md`

这是当前问题重写的关键文档：

- 主问题不应再是“还没找到对的 top7 ordering / tuning”。
- 应改写为：
  - 当前 pipeline 里，什么样的 upstream donor contract 才是 downstream replace-compatible？
  - 如果 top7 需要更宽语义范围，应该由 upstream 负责长成兼容形态，还是由 downstream replace 负责吸收 expansion？

核心假设：

- `top3` 对应 replace-compatible anchor basin。
- downstream replace 目前只在这个 basin 周围建立过稳定 contract。
- `top7` 新增信息不是不能学，但不能直接覆盖 / 重写 anchor；它需要以 residual expansion 方式附着在 anchor 上。
- downstream replace 不应要求 donor 自己完全长成最终兼容 contract，而应有能力吸收 donor expansion。

注意：这不是说 `top3 anchor` 是最终通解；它是当前 locked contract 的诊断锚点。

### 2.7 A1 boundary chain: anchor/guard as diagnostic, not solution

#### A1-S1: shared head already compromised

`docs/retired_directions/top7_old_boundary_transferability_family/2026-04-09_partial_transplant_boundary_a1s1_record.md`

- 判例：Case 3。
- 主判断：`shared_head_already_compromised`。
- `anchor_only` 不比 `E2A-R full7` 更 replace-transferable。
- `dir_leg` worsening boundary 更像 earlier shared-head boundary。
- 建议进入 A1-S2，优先测 `E1-top3 anchor + top7 nonleg expansion`。

含义：break 比单侧 expansion 更早，不能简单写成某个 side branch 容量不足。

#### A1-S2: nonleg partially absorbable but not decisive

`docs/retired_directions/top7_old_boundary_transferability_family/2026-04-09_mixed_contract_a1s2_record.md`

- `A1S2-mix-nonleg` 比 `E2A-R full7` 更好，但不够 clear-win。
- `A1S2-mix-nonleg` aggregate 上接近 `E1-top3 full7`。
- preserved anchor 下，`top7_nonleg_partially_absorbable_but_not_yet_decisive`。
- 不建议更明确转向 replace-side absorb-expansion；推荐 `nonleg_absorb_expansion_only_with_stronger_replace_side_absorb_or_boundary_guard`。

含义：mixed transplant 有诊断价值，但不是通解。

#### A1-S3: plain split not decisive

`docs/retired_directions/top7_old_boundary_transferability_family/2026-04-09_replace_absorb_boundary_a1s3_record.md`

- A1-S3 判例：Case C。
- 两个 split arms 都没有明显优于 `A1S2-mix-nonleg`。
- host absorb boundary：`no_clear_winner; weak lean host nonleg out side`。
- main incompatibility boundary：`no_clear_single_boundary; weak lean downstream nonleg readout contract`。
- 推荐：`shrink_back_to_earlier_boundary_or_stronger_boundary_guard`。

含义：plain replace-side splitting 不足以成为 decisive absorb 路线。

#### A1-S4: direct head input guard ruled out

`docs/retired_directions/top7_old_boundary_transferability_family/2026-04-09_direct_head_input_guard_a1s4_record.md`

- `direct_pose_head.0` input pairwise divergence = 0。
- 相同 host + weight transplant 口径下，所有 arms 在 direct head input 收到完全相同 activation。
- moment-matching input guard 只有数值噪声级别效果，不是有效 lever。
- 结论倾向：bottleneck 不在 head 上游，而更像在 head 内部 / head output -> downstream expansion contract。

含义：继续在 head input 前做 normalization / matching guard 不是主线。

#### A1-S5: head output / nonleg consumer entry confirms downstream contract work

`docs/retired_directions/top7_old_boundary_transferability_family/2026-04-09_direct_head_output_guard_a1s5_record.md`

- A1-S4 已排除 head input，因此 A1-S5 移到 shared head output / nonleg consumer entry。
- head output / nonleg consumer entry 出现非零 divergence。
- consumer-only guard 的 spillover 有限，但不是 decisive solution。
- 推荐下一步：shift effort to more-downstream contract work or training-side recipe constraints。

含义：更支持“head output -> downstream expansion contract”这一层的接口问题，而不是上游输入分布问题。

### 2.8 Adapter sham audit: branch positives can be structural perturbation

`docs/retired_directions/replace_redesign_and_falsifier_family/2026-04-10_branch_sham_audit_replace_adapter_record.md`

该文档给了一个重要方法学约束：

- `replace zero-init residual adapter` 的历史强结论需要降级。
- mean 指标上，sham 已吃掉大部分 branch gap：
  - `all_ex_root`: 82.8%
  - `nonleg`: 91.3%
  - `arm`: 89.0%
  - `else`: 93.8%
- 原始最关键的 `all_ex_root p95 / arm p95 / else p95` 改善，sham 已解释到约等于甚至超过全 branch gap。
- adapter 仍有更窄真实净效应，主要集中在 `leg p95`。
- 方法学 takeaway：凡是 `branch / auxiliary / adapter / replace-side structural module`，只要历史结论依赖 `baseline -> branch` gap，就必须默认带 matched sham control。

含义：今后不能轻易把“加一个结构后变好/变坏”读成该结构 objective 本体效果；要先排除 structural perturbation / optimizer trajectory fork。

### 2.9 Aux-family chain: interface-contract mechanism evidence, not a direct mainline solution

Historical evidence bundle:

- `docs/retired_directions/aux_shared_trunk_family/README.md`

`docs/retired_directions/aux_shared_trunk_family/2026-04-11_shared_trunk_mechanism_chain_closure_note.md`

final mechanism hierarchy：

1. Primary near mechanism: gradient-path-mediated harm when aux pressure lands on the shared trunk。
2. Most plausible local form: shared-trunk plasticity / capacity sink。
3. attach mismatch can redirect the sink into a less costly parameter pool。
4. per-step sign conflict is not primary。
5. rollout/objective mismatch downgraded as main explanation。

`docs/retired_directions/aux_shared_trunk_family/2026-04-11_aux_detach_downstream_reevaluation_record.md`

- `aux_detach` 没有形成 clean downstream positive case。
- `stage6` 不 beat baseline。
- `70a` 明显 worse than baseline。
- final `70b replace` only mixed：部分 nonleg/arm improvement，但 leg regress，aggregate improvement 太小。
- final decision：
  - `global default = baseline`
  - `conditional aux-family default = aux_detach`
  - feature not justified for mainline。

含义应更精确地写成两层：

1. **它不是 aux-family mainline 通解**
   - `aux_detach` 没有 clean downstream positive case；
   - 因此不能把 aux-family 本身升格成 mainline solution。

2. **但它是 interface-contract rewrite 的重要微观机制证据**
   - 它直接支持：当额外梯度压力落到 shared trunk / shared interface variables 时，会出现 plasticity sink，并改写下游正在消费的 contract。
   - 因而在 future universal redesign 里，aux chain 不应只被记为 side observation，而应被记为：
     - **interface-contract fragility evidence**
     - **why frozen-donor / decoupled-corrector may help 的近机制论据**

不过这里仍保留一个 transport caution：

- aux chain 提供的是 **近机制证据**，不是对 top3/top7 全链条的一步到位证明；
- 它强化“shared interface variables 对额外 plasticity 很敏感”，但不自动证明所有 widened-support failure 都由完全相同的微观机制单独造成。

---

## 3. Falsified / downgraded explanations

### 3.1 Simple branch-side parameter capacity shortage

不支持。

原因：

- `top3` 比 late/full `top7` 变体更 transfer-compatible。
- `3 -> 5 -> 7` curriculum helpful but insufficient，仍不如 `top3`。
- leg-first 没救 `dir_leg`，且出现 nonleg giveback。
- readout-first / allocation 也没有建立更好的 transfer-compatible basin。
- aux-family 不形成 mainline positive case。

### 3.2 Single leg-capacity / leg-ordering root cause

不支持。

原因：

- `E2C-L` 没把 `dir_leg` 稳定抬起来。
- `E3A-RF` 也让 `dir_leg` 更差。
- `A1-S1` 把 boundary 更早地指向 shared head，而不是单纯 leg expansion candidate。

### 3.3 Shared activation distribution pathology mainline

不支持作为主线。

原因：

- `E1-top3` 与 `notail` 的 dominant family / grad alignment 不一致。
- baseline 在某些 pathology 计数上反而更多。
- donor 侧额外增加主要是 heavy-tail，但没有 shared dominant family。

### 3.4 Warmstart surgery as main cause

不支持作为主因。

原因：

- `tailk7 baseline-style adapted warmstart` 能改善绝对指标，但仍没有把局部几何推向 baseline-style non-plan basin。
- 当前更像 donor-state / `70a` exit basin。

### 3.5 Step0/1 local optizability as main cause

不支持。

原因：

- `tailk7 raw 70a` 没表现出比 baseline raw 70a 更差的 local non-plan optizability。
- 若保留 C，只能保留 higher-order exit-basin / multi-step geometry 版本。

### 3.6 Adapter / branch structural module as clean global solution

需要降级。

原因：

- matched sham 吃掉了 adapter 大多数 mean / p95 gap。
- adapter 的真实净效应主要集中在 `leg p95`，不是全局 interface translation 解。

---

## 4. Current working hypothesis

当前最简洁的工作假设：

> Widened support / extra capacity 不一定学不到东西；它能学，但在 current locked replace contract 下，这些新增自由度经常改写 shared interface contract，把 donor 推进 self-consistent but replace-incompatible basin。

展开：

1. `top3` 更像 current replace 已经能消费的 anchor basin。
2. `top7` 新增信息不是不能学，而是不能以 monolithic overwrite 的方式进入 current shared-head contract。
3. 当前 `replace` 更像 weight-space continuation，而不是 donor-robust downstream correction。
4. 因此 “越验证越差” 是合理现象：更多 freedom 增加了 contract drift / basin mismatch 风险。
5. `aux` 和 adapter/sham 记录共同提醒：extra structure / extra path 的 readout 很容易混入 structural perturbation、optimizer trajectory fork、shared-trunk plasticity sink，不能直接当 objective gain。

---

## 5. What counts as escaping the problem

以下做法如果被当成最终方案，就属于逃避通解问题：

- 固定 `top3` anchor，然后宣布 top7 问题 solved。
- freeze / guard shared head，只为了不让 donor 离开旧 basin。
- 继续寻找某个 donor path，使其恰好落回 current replace-compatible basin。
- 把 boundary guard / partial transplant 的正读数当成 universal downstream correction。
- 用 normalization / clipping / whitening 修局部 distribution readout，然后宣布 root cause 解决。

但这些做法仍可作为 debug instrument：

- 用 `top3 anchor` 定义 current locked contract 的参考点。
- 用 freeze / guard 定位哪一层改动会破 contract。
- 用 partial transplant 找 earliest break boundary。
- 用 moment matching / activation taps 排除输入分布解释。
- 用 matched sham 控制 structural perturbation。

换句话说：

- **允许作为 diagnostic scaffolding**
- **不允许作为 universal-solution candidate**

补充：

- **允许作为 benchmark / acceptance gate**
  - 尤其是 `top3 donor`。
  - 不能把 “top3 不是最终解” 误写成 “top3 在 redesign 中不再重要”。

---

## 6. Universal-debug direction

### 6.1 Main objective

后续如果目标是“通解”，主问题应写成：

> 如何让 downstream correction 在 donor basin / donor contract 改变时仍然成立？

而不是：

> 如何把 widened top7 donor 训练回旧的 top3-like replace-compatible basin？

### 6.2 Candidate design axis

与 2026-04-02 redesign memo 保持一致，v1 通解候选应优先探索：

- frozen donor；
- behavior-space observable input；
- fresh init；
- zero-correction start；
- residual corrector；
- 尽量不读取 donor internal parameter-space shape；
- 多 donor / held-out donor 验证。

但在直接进入 full redesign 之前，应先加一个 **R0 minimal interface-decoupling probe**：

- frozen donor；
- fresh-init residual corrector；
- zero-correction start；
- 只跑 `top3 donor`；
- 目标不是证明 universal，而是先 falsify 当前最小 `behavior-space observable` 假设。

R0 的解释规则：

- 若 `top3 donor` 上都明显打不过当前 locked contract：
  - 不能直接推进 full redesign；
  - 需要先回头修正 `behavior-space observable` 通道、input choice、objective 或最小容量假设。
- 若 `top3 donor` 上至少持平或更好：
  - 才值得扩到 `E2A-R` / bad top7 / mid-training / mixed donors。

因此 `top3 donor` 在 redesign 初期不是被 deprecate，而是：

- **first benchmark**
- **acceptance gate**
- **minimal falsification target**

### 6.3 Required evaluation change

若继续只在 single donor / locked contract 上看结果，就会不断把 local optimum 和通解混在一起。

更合理的验证矩阵，且应先被写成一个**封闭 donor scope contract**：

1. Donor scope（建议当前版本锁定为）:
   - `E1-top3`
   - `E2A-R`
   - bad late/full `top7`
   - mid-training donor（当前建议显式落到 `epoch013/014/015` 家族）
   - mixed-contract donor（当前建议以 `A1S2-mix-nonleg` 为起点）

2. Acceptance rule:
   - 任一 universal candidate 必须先在 `E1-top3` 上至少持平当前 locked contract；
   - 否则不进入更大 donor scope。

3. Evaluation rule:
   - train corrector on subset of donors；
   - hold out at least one donor family；
   - held-out donor 仍有效，才接近 universal candidate；
   - 只在 seen donor 有效，则降级为 donor-specific adaptation。

4. Control rule:
   - any branch / adapter / aux structure must include matched sham；
   - no baseline -> branch gap can be promoted without sham decomposition。

这里特别修正一点：

- donor scope 不应长期保持开放式表述；
- 如果一年后 donor set 继续漂移，几乎无法判断 redesign 是真的成立，还是 scope 在不断重定义。

---

## 7. Suggested next actions

### U0. Write the experimental taxonomy before running more arms

新增实验记录必须标注：

- `diagnostic scaffolding`
- `locked-contract local optimum`
- `universal-solution candidate`

如果一个 arm 只是 freeze / guard / anchor preservation，它不能被写成通解候选。

### U1. Build a minimal multi-donor robustness table

先不重训大模型，整理现有 donors：

- `E1-top3`
- `E2A-R`
- `E1-top7` / bad full top7
- `epoch013/014/015` stage6 exact snapshots
- `A1S2-mix-nonleg`

目标：定义一个固定 donor family matrix，后续所有 downstream correction 都必须在这张表上汇报。

这张表不只是 tracking sheet，还应成为 redesign memo 的 **scope contract**。

### U1.5 Run R0 before full redesign

先不一次性 commit 整个 universal redesign。

最小新增实验建议：

- `R0 = frozen donor + fresh-init residual corrector + zero-correction start + top3 donor only`

判读：

- 若 `R0` 在 `top3 donor` 上都打不过 current locked contract：
  - 当前最小 `behavior-space observable` 假设先不成立；
  - redesign 先修输入通道 / objective / minimal capacity；
  - 暂不扩 donor family。
- 若 `R0` 在 `top3 donor` 上打平或更好：
  - 再扩到 `E2A-R`、bad top7、mid-training、mixed donors。

### U2. Prototype behavior-space frozen-donor residual corrector

优先验证最小 v1：

- donor frozen；
- corrector fresh init；
- zero-correction start；
- 输入只用 behavior-space observable / standardized outputs；
- 不把 donor internal hidden / parameter shape 当主要输入；
- 先做 small smoke，而不是 full recipe sweep。

### U3. Use anchor / guard only to localize interface variables

如果继续做 `top3 anchor`、freeze、guard：

- 只回答 boundary 问题；
- 不把结果写成 final solution；
- 每次记录都明确说明它是否改变了通解设计。

### U4. Keep distribution pathology and aux-family as downgraded branches

- distribution pathology：只保留极轻量 smoke，不再作为 mainline root cause。
- aux-family：`aux_detach` 只作为 conditional aux-family default；global mainline 仍是 baseline。
- adapter/branch family：必须带 matched sham。

---

## 8. Current bottom line

当前最应该固定的结论：

1. 这不是简单的 capacity shortage。
2. `top7` / widened support 暴露的是 current downstream interface 的 donor-basin sensitivity。
3. `top3 anchor` 不是最终通解，但它仍是 redesign 的 benchmark / acceptance gate。
4. freeze / guard / boundary preservation 是 debug scaffolding，不是 solution。
5. 在 full redesign 前，应先跑 `top3 donor` 上的最小 interface-decoupling probe。
6. universal scope 应先锁定成封闭 donor 集合，而不是长期开放。
7. aux-family chain 不只是 aux-only finding；它还是 interface-contract fragility 的重要微观机制证据。
8. 任何新 structural branch / aux / adapter 结论必须配 matched sham control。

一句话：

> 当前不是在 debug “怎么把 top7 调顺”，而是在 debug **为什么 current downstream interface 只吃某一类 donor basin，以及怎样把它改成真正 donor-robust 的接口**。
