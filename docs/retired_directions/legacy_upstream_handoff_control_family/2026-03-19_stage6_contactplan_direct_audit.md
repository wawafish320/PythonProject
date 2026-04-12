# Stage6 设计审计（contact-plan 锚点 -> direct expert）

> Status: archived legacy upstream / handoff / control record
> Reader note: this file belongs to the old-boundary upstream-control investigation; any `current`, `default`, `canonical`, `recommend`, or `mainline` wording below is historical context, not present-tense repo policy.
> Current entry points: `docs/posttrain_pipeline.md`, `docs/train_design/2026-04-11_donor_contract_basin_universal_debug_synthesis.md`, `docs/retired_directions/legacy_upstream_handoff_control_family/README.md`

> Last updated: 2026-03-19
> 目标：回答一个更基础的问题：`Stage6` 采用“基于 `contact-plan` 锚点去特训 `direct expert`”这条路线，本身是否正确；如果有问题，问题更可能出在设计理念、输入契约，还是上游 handoff/selector。

关联输入文档：
- `docs/retired_directions/legacy_upstream_handoff_control_family/2026-03-19_basetrain_handoff_selector_contactplan.md`
- `docs/Problems/active/2026-03-06_trainbase_stage6_presplit_phaseA_checklist.md`
- `docs/posttrain_pipeline.md`

关联代码：
- `train/models.py:1301`
- `train/models.py:2314`
- `train/models.py:3477`
- `train/models.py:3480`
- `train/training_MPL.py:3219`
- `train/training_MPL.py:3232`

---

## 0) TL;DR

1. **方向本身是对的**：对 gait / lower-body / phase-sensitive motion 来说，`contact-plan` 作为 `direct expert` 的主锚点，比直接围绕 pose 做特训更合理。
2. **它不是全解**：`contact-plan` 很适合 disambiguate lower-body mode，但不足以单独决定 full-body 的 arm/trunk/style 细节。
3. **Stage6 的主问题更像 contract mismatch，不像设计错误**：
   - `Stage6` 假设上游给进来的 ckpt 已经具备“可用的 contact-plan 语义”；
   - 但当前 `basetrain` 选点更偏 `pose/global drift`，不保证这一点。
4. 因此，当前优先级应是：
   - 先修 `basetrain -> Stage6` 的 handoff selector；
   - 再看 `Stage6` 本体是否仍然存在结构性缺陷。

---

## 1) 直接结论

对于当前这条 old-d1 / gait 主链，我的判断是：

> “基于 `contact-plan` 锚点去特训 `direct expert`”本身是正确选择，而且在 lower-body / phase disambiguation 这一层面，比“继续基于 pose 误差去硬调 direct head”更符合问题结构。

但这句话需要补完整：

> `contact-plan` 应该是 `Stage6 direct expert` 的 **primary anchor**，而不是 **only anchor**。

也就是说：

- 它应主导“当前属于哪一种接触/相位模式”的判别；
- 但 direct pose 的最终细节，仍需要 `cond / hidden / context` 提供额外带宽。

---

## 2) 为什么这个设计方向是对的

## 2.1 因果顺序更合理：contact-plan 比 pose 更接近 mode

对 walking / gait 这类任务，语义顺序更接近：

`cond -> contact/phase mode -> pose realization`

而不是：

`cond -> pose -> 再反推出 contact`

因此，如果 `direct expert` 的关键困难是：

- 同一段 `cond` 下，到底处在哪个 phase / contact mode；
- 当前左/右脚的支撑切换和步态时相是否已经对齐；

那么最自然的锚点不是 pose 本身，而是：

- `contact-plan`
- 或与之等价的 phase/contact latent

这也是当前模型设计中 `contacts_plan` 的定位：

- `train/models.py:1301`：`contacts_plan` 被定义为 independent anchor
- `train/models.py:2314`：它通过 GRU(cond) 产生，并保持对 pose 独立

## 2.2 它能避开 autoregressive pose drift 的污染

如果把 `direct expert` 的主要锚点放在 pose/history 本身，会有一个天然问题：

- freerun 时 pose 已经被前序误差污染；
- 再拿这个被污染的 pose 去指导 direct head，容易把错相位当成真相位。

而 `contact-plan` 的价值在于：

- 它是从 `cond + plan_z` 递推出来的独立锚点；
- 它不直接依赖当前预测 pose；
- 因而在 freerun / long-chain 中更适合作为“纠错参考系”。

这也是当前实现里 `contacts_err = contacts_plan - contacts_meas` 还能持续有意义的原因：

- `train/models.py:3477`

## 2.3 它与后链模块语义一致

`Stage6` 不是孤立模块。它后面接的是：

- event-clock / correction logic
- direct-vs-inc fusion logic
- Stage70R/71/72/lambda 这些 downstream 适配层

这些模块能否工作，很大程度上取决于：

1. `contacts_plan` 是否稳定
2. `contacts_err` 是否有判别力
3. direct head 是否围绕这个锚点形成清晰模式

所以从系统设计上看：

> 用 `contact-plan` 作为 `Stage6 direct expert` 的主锚点，是与后链契约一致的。

---

## 3) 为什么它不是全解

## 3.1 lower-body 对，full-body 不够

`contact-plan` 对以下内容非常有效：

1. 左/右支撑切换
2. 步态 phase disambiguation
3. lower-body mode selection
4. 触地窗口附近的 leg/foot pose 选择

但以下内容并不完全由 `contact-plan` 决定：

1. arm swing 细节
2. trunk/upper-body style
3. 非步态主导的局部姿态变化
4. 一些 long-range style / motion texture

因此如果把 `contact-plan` 当作唯一锚点，会有两个风险：

- 它会过强地约束 lower-body，但对 non-leg 解释不足；
- direct head 可能把“不由 contact 决定的部分”也硬塞到 contact pattern 里。

## 3.2 它应该回答“mode 是什么”，不应该单独回答“全部 pose 是什么”

比较准确的职责分工应是：

- `contact-plan`：决定 mode / phase / support pattern
- `cond / hidden / context`：决定同一 mode 下的具体 pose realization

因此 `Stage6` 如果继续走这条路线，最健康的形态不是：

- `contact-plan only`

而是：

- `contact-plan anchored`
- `cond/context completed`

这与当前 direct head 的输入设计也是一致的：`direct_pose_head` 并不是只吃 plan，而是吃 `cond + contacts_plan (+ optional hints)`。

## 3.3 `contact-plan` 的信息论角色：不是单一“锚点”，而是双重职责

在当前 `Stage6` 实现里，`contact-plan` 实际承担两个不同的信息职责：

1. **mode selector**
   - 告诉 `direct expert` 当前属于哪一种接触/相位模式；
   - 例如：左支撑、右支撑、双支撑、过渡区间等。

2. **continuous control signal**
   - 作为连续值直接进入 `direct_pose_head`，参与 pose realization；
   - 因而它不仅决定“是哪一类 mode”，也会影响“该 mode 下的输出落在什么位置”。

这一区分很重要，因为两种职责对锚点质量的要求并不相同：

- 若只承担 `mode selector`，则 `contact-plan` 只要相位大致对、切换时机大致准，downstream 往往可以容忍一定的幅度噪声；
- 若同时作为 `continuous control signal` 被 direct head 直接消费，则其数值幅度、校准误差、transition sharpness 都会更直接地传导到 pose 误差。

从当前实现看，`Stage6` 更接近第二种，而不是纯第一种：

- `contacts_plan` 由 `cond -> GRUCell -> logits -> sigmoid` 生成，是一个连续值信号；
- `direct_pose_head` 默认路径会直接消费 `contacts_plan`，而不是先把它离散成 mode token 再使用。

对应代码：

- `train/models.py:2799`
- `train/models.py:3221`
- `train/models.py:3627`

因此，`Stage6` 对 `contact-plan` 的真实要求，并不只是：

- “当前 mode 有没有大致选对”

而更接近：

- “当前 mode 选对了”
- “contact-plan 的连续幅度也足够可解释、可校准”
- “这种质量在 freerun 下仍能维持”

这意味着，当前系统对 `contact-plan` 的敏感度，比“锚点”这一表述通常给人的直觉更高。

---

## 4) Stage6 设计本体有没有明显问题

## 4.1 没有看到“理念错误”

从现有证据看，我不认为 `Stage6` 的核心理念有错。

更像是：

- `Stage6` 假设自己拿到的是“已经有可用 contact-plan 语义的上游 ckpt”；
- 但实际上，当前上游 `basetrain` 的 selector 并不是按这个目标选的。

所以 `Stage6` 常常要先做一件本不该由它主导的事：

- 先把 handoff 分布纠回到 contact-plan-friendly 区域

这会让它看起来像“设计不稳”，但问题未必在 `Stage6` 自身。

## 4.2 它的真正前提条件是：上游 handoff 必须 contact-plan-friendly

当前 `basetrain` 的 ckpt 选择仍主要按：

- teacher pose 指标
- freerun drift/global scalar

对应：

- `train/training_MPL.py:3219`
- `train/training_MPL.py:3232`

而 `Stage6` 真正想要的上游是：

- `contacts_plan` 有清晰语义
- `contacts_err` 有足够判别力
- lower-body risk 不被 global scalar 掩盖

如果 selector 不保证这些条件，那么 `Stage6` 的表现会被系统性低估。

## 4.3 repo 现有结论也更支持“selector mismatch”，而不是“Stage6 理念错”

现有 Phase A 结论已经把问题定性为：

- 更像 `selector mismatch`
- 不只是 Stage6 起跑就坏了
- 不建议继续在 trainbase / Stage6 做小步调权

对应：

- `docs/Problems/active/2026-03-06_trainbase_stage6_presplit_phaseA_checklist.md:1036`
- `docs/Problems/active/2026-03-06_trainbase_stage6_presplit_phaseA_checklist.md:1129`
- `docs/Problems/active/2026-03-06_trainbase_stage6_presplit_phaseA_checklist.md:1161`

这与本审计的结论一致：

> 现阶段更该怀疑的是 handoff selector，而不是 `contact-plan anchored direct expert` 这条设计本身。

## 4.4 `Stage6` 真正依赖的 handoff contract：不是一层，而是三层

若要更准确地定义 `Stage6` 需要什么样的上游 ckpt，不能只说“有可用的 contact-plan 语义”，而应拆成三层契约。

### A. 语义层（semantic alignment）

要求：

- 左/右支撑顺序正确；
- 相位切换时机基本对齐；
- `contacts_plan` 与 GT / meas 的 circular shift 不大；
- 关键 gait mode 不发生系统性左右翻转或整段错位。

它回答的问题是：

> “当前属于哪个 gait/contact mode？”

### B. 校准层（continuous calibration）

要求：

- `contacts_plan` 不塌缩到长期高熵中间值；
- stance / swing 之间有足够 separation；
- transition 区间的宽度、斜率、峰谷形状处于合理范围；
- channel 间相对幅度具有稳定语义，而不是随机漂动。

它回答的问题是：

> “即使 mode 大体对了，这个 mode 在连续空间中的位置是否也对？”

### C. 稳定层（freerun stability）

要求：

- 上述 A / B 两层质量在 freerun 下仍能维持；
- 不因 `cond` 分布偏移而快速恶化；
- `contacts_err` 在 `cycle>=1` 下仍保有判别力；
- contact-plan 不只在 teacher 下好看，而是在闭环时仍可作为 downstream 锚点。

它回答的问题是：

> “这个锚点在真正运行时是否仍可靠？”

因此，`Stage6` 的 handoff contract 更准确的说法应是：

> 上游进入 `Stage6` 的 ckpt，不仅要有 `contact-plan` 语义，还要同时满足 semantic alignment、continuous calibration、freerun stability 三层要求。

如果 selector 只保证其中一层，或者只保证 teacher pose/global scalar，而不保证这三层，那么 `Stage6` 的真实起点就会被系统性高估或低估。

---

## 5) Stage6 这条设计的真正短板是什么

如果要说 `Stage6` 有什么短板，我会说有三个，而且都不是“锚点选错”。

## 5.1 锚点带宽有限

`contact-plan` 可以很好地告诉 direct head：

- “现在是哪个接触/相位模式”

但不能单独告诉它：

- “当前 upper-body 的具体 realization 应该是什么”

所以它天然更偏 lower-body / phase 友好，而不是全身统一最优。

## 5.2 对上游 ckpt 语义质量敏感

如果上游给进来的 ckpt：

- `contacts_plan` 质量一般
- 或虽然 pose/global drift 很好，但 contact-plan 语义并不强

那 `Stage6` 的起点会变差。

这不是因为 Stage6 选错了锚点，而是因为 handoff 契约没有被 selector 保证。

## 5.3 容易让人误判成“pose 问题”

因为最终观测到的退化往往仍然表现为：

- direct pose error
- lower-body pocket
- global scalar 变化

所以很容易把它误判成：

- “Stage6 direct 结构不对”
- “split 还不够”
- “direct loss 还要再调”

但如果根因是：

- contact-plan 锚点质量未被上游 selector 保证

那继续调 Stage6 只是在下游补锅。

## 5.4 当前架构的结构性张力：anchor 语义与 continuous consumption 之间存在耦合

`Stage6` 当前路线的理念是正确的：对 gait / lower-body 问题，先抓 `contact-plan` 比先抓 pose 更符合因果结构。

但当前实现里存在一个需要正视的结构性张力：

- 设计语言上，`contact-plan` 被当作“anchor”；
- 但在实现上，它又被 direct head 当作连续输入直接消费。

这带来一个后果：

- 系统对 `contact-plan` 的敏感度，不只是“mode 有没有选对”；
- 还包括“连续幅度是否校准”“transition 是否合理”“中间概率是否携带了错误的置信度语义”。

换句话说，当前 handoff contract 不是：

- “phase roughly right 就够了”

而更接近：

- “phase/timing 对”
- “连续 contact 基底也别太偏”
- “而且这种质量要能在 freerun 下保持”

这也解释了为什么 selector mismatch 的影响会这么大：

- 不只是 mode 选偏了；
- 也是 direct head 所依赖的连续基底一起被带偏了。

因此，当前 `Stage6` 的脆弱性不在于“锚点选错”，而在于：

> 这个锚点在实现上不只是做 disambiguation，还承担了连续控制信号的职责。

这不是对当前路线的否定，但它意味着：

> 如果 selector 对齐之后，系统仍然对上游微小 contact-plan 偏差异常敏感，那么下一步更值得怀疑的是 anchor contract 过硬，而不是继续先调 pose loss。

## 5.5 `contact-plan` 的主要 error modes

`contact-plan` 作为独立锚点，确实能避免直接继承 autoregressive pose drift；
但它并不是没有误差模式，而是拥有一组与 pose drift 不同的、属于自己链路的 error modes。

### 1. semantic mismatch

表现：

- 左右支撑顺序错；
- 相位切换时机偏；
- circular shift 明显；
- contact-plan 的峰谷与真实 gait 事件存在稳定错位。

后果：

- direct head 进入错误 gait mode；
- lower-body 直接出现错相位 realization；
- 后链 correction / lambda 可能只能“局部补偿”，无法真正纠正根模式错误。

### 2. calibration mismatch

表现：

- `contacts_plan` 长时间停留在高熵中间值；
- 峰谷不够分离；
- stance/swing 之间 separation 不稳定；
- transition 过宽、过窄、或不同片段中标度不一致。

后果：

- direct head 学到的是模糊的 phase basis；
- 即使 mode 大体没错，pose realization 也会持续偏；
- 系统表面上看起来像“direct 回归不稳”，实则是 continuous hint 质量不够。

### 3. freerun stability failure

表现：

- teacher 下 `contact-plan` 看起来不错；
- freerun / `cycle>=1` 下迅速失真；
- `contacts_err` 早期仍有判别力，后期却失去区分度。

后果：

- handoff 时看似“起点没问题”，闭环一跑就不耐久；
- downstream 容易误以为是 `Stage6 / 70R / 71 / 72 / lambda` 在放大误差；
- 实际上更早的锚点稳定性已经先失效。

### 4. downstream over-consumption

表现：

- `contacts_plan` 指标本身不一定差；
- 但 direct 或下游模块对其连续值过度敏感；
- 小的 calibration noise 就能引出明显的 lower-body regression。

后果：

- 小误差被结构放大；
- 表面现象像“Stage6 direct 结构不稳”，本质则是 input contract 对 anchor 质量要求过硬。

因此，在看 `Stage6` / downstream 回退时，应优先区分：

- 是 `contact-plan` 自己就坏了；
- 还是 `contact-plan` 尚可，但 downstream 对其消费方式过于敏感。

## 5.6 `contact-plan` 的独立性是相对的，不是绝对的

`contact-plan` 的“独立”含义，需要更精确地表述。

它的独立性主要体现在：

- 独立于当前预测 pose；
- 独立于被 autoregressive 污染的 pose history；
- 因而不会像 pose-anchored 方案那样，直接把前序姿态误差重新喂回主锚点。

但它并不独立于：

- `cond` 的质量；
- `cond` 在 freerun 中的分布偏移；
- `plan_z` / GRU 递推本身的累计误差；
- 与 contact timing 相关的长期时序偏移。

因此，更准确的表述应是：

> `contact-plan` 独立于 pose drift，但不独立于 cond drift。

这意味着：

> `contact-plan` 的可靠性上界，受 `cond` 在 freerun 中的稳定性约束。

这也说明为什么 handoff selector 不能只看 teacher 下的 `contact-plan` 质量；
对于 `Stage6` 真正重要的是：

- 在 freerun 条件下，`cond -> contact-plan` 这条链是否仍然稳定；
- 在 `cycle>=1` 时，contact-plan 是否仍足以支撑 downstream 的 mode / correction / fusion 逻辑。

---

## 6) 对这个设计的建议结论

## 6.1 应保留，不建议推翻

当前我不建议把 `Stage6` 重新改回：

- pose-centered direct expert training
- 或 purely hidden-centered direct tuning

因为对于 gait/lower-body 主问题来说，`contact-plan` 仍是更合理的主锚点。

## 6.2 应明确成“primary anchor, not sole source”

后续设计判断应统一为：

1. `contact-plan` 是 primary anchor
2. `cond / hidden / context` 是 completion path
3. downstream-aware selector 要保证 handoff 进来的 ckpt 已经满足这个前提

## 6.3 先修 selector，再审 Stage6

更合理的排查顺序是：

1. 先把 `basetrain -> Stage6` selector 改成 contact-plan/downstream-aware
2. 再看 `Stage6` 是否还存在结构性退化
3. 只有在 selector 对齐后问题仍然持续，才考虑：
   - Stage6 direct input 带宽不足
   - direct head 过度依赖 contact-plan
   - lower-body / non-leg 解耦方式还不够好

## 6.4 selector 对齐之后，何时才值得做结构升级

更合理的判断顺序，不应是“先看到 pose 回退，再立刻改 Stage6 结构”，而应是分阶段推进。

### 第一阶段：先修 selector / handoff contract

目标是先保证：

- 上游 ckpt 在 `contact-plan` 的 semantic alignment 上达标；
- 在 continuous calibration 上不过度塌缩；
- 在 freerun stability 上足够可靠。

在这个阶段完成之前，不应贸然把问题归咎于 `Stage6` 本体结构。

### 第二阶段：selector 对齐后再看剩余问题属于哪一类

若 selector 对齐后问题仍存在，再区分：

1. **semantic 仍差**
   - 优先继续修 handoff / contact-plan 训练；
   - 不应先改 direct 结构。

2. **semantic 基本对，但 calibration 敏感**
   - 说明当前 direct 对连续 `contacts_plan` 过于敏感；
   - 此时才值得考虑把 `mode-selection` 与 `continuous hint` 更显式地拆开。

3. **teacher 好、freerun 差**
   - 优先怀疑 `cond -> plan` 链的闭环稳定性；
   - 不应只看 teacher contact-plan 指标得出结论。

4. **contact-plan 指标好，但 downstream 仍差**
   - 再回头定位 `Stage6 -> 70R -> 71 -> 72 -> lambda` 中哪一级对新上游分布最敏感。

因此，结构升级不应是第一反应，而应是：

> selector 对齐之后，对残余误差模式做归因，再决定是否需要改 `Stage6` 本体。

## 6.5 若必须做结构升级，推荐顺序

只有在 selector 已对齐、且 `contact-plan` 三层契约已基本达标后，才值得考虑更进一步的结构升级。

推荐顺序如下：

### 1. 先做“角色分离”，不要先做大改

更值得优先考虑的是：

- 把 `contact-plan` 的 `mode-selection` 角色
- 与它的 `continuous residual hint` 角色

做更显式的分离。

原因是，当前最值得怀疑的结构性问题，不是“有没有 contact-plan”，而是：

- 同一个信号同时承担了离散分流与连续控制两种职责；
- 这使 downstream 对 anchor calibration noise 更敏感。

### 2. 优先考虑 mode-gated direct，而不是继续加 pose loss

如果残余问题本质仍是：

- gait mode ambiguity
- phase ambiguity
- support pattern ambiguity

那么继续在 pose 侧加 loss、加 split、加局部权重，往往只是下游补锅。

相比之下，更有针对性的方向是：

- 让 direct 明确按 mode 分流；
- 再让连续 phase/contact 信号只承担 residual hint 的角色。

### 3. 高带宽 phase hint 应视为第二阶段选项

repo 中已经存在更高带宽的 phase hint 方向（如 `phase_z_in`）；
这说明系统也已经意识到 2D `contacts_*` 连续概率可能带宽不足。

对应更早的历史设计分析：

但这个方向应被视为：

- selector 对齐后的第二阶段增强项；
- 而不是在 handoff contract 尚未对齐时就提前大改的主线方案。

### 4. 不建议回退到 pose-centered 作为默认解

即使当前 `contact-plan` 路线存在敏感度问题，也不意味着应该回退到：

- pose-centered direct tuning
- 或重新把 pose/history 拉回主锚点

因为这样做会把 autoregressive pose drift 再次放回系统主参考系，等于回到原来更不稳定的问题结构中。

因此，如果未来需要升级，推荐理解为：

> 保留 `contact-plan anchored` 这条主线，只对其“被 direct 如何消费”这件事做更精细的因子化，而不是推翻主锚点本身。

## 6.6 建议的后续实验路线

后续实验不建议一上来就做大结构改造，而应按“归因确定性 / 预算成本”从低到高推进。

### Phase 1：先做 selector / handoff 对齐，不改 `Stage6` 语义

目标：

- 先确认当前问题里有多少是 handoff 造成的；
- 避免把“上游进错 ckpt”的问题误判成“Stage6 结构本身有问题”。

建议做法：

1. 按 `docs/retired_directions/legacy_upstream_handoff_control_family/2026-03-19_basetrain_handoff_selector_contactplan.md` 增加 `best_handoff` 候选选择；
2. 候选集不要只比较 `best_free` 和 `last`，还应包含 `best_free` 附近的 top-k epoch；
3. 固定评估口径为：
   - `pretrain_contact + affine_mix08`
   - `cycle>=1`
   - `drop_wrap=true`
4. 主排序先看：
   - `ContactErrAbsMean`
   - `ContactPlanGtAbsMean`
   - phase / circular shift 对齐
   - `leg8_mean`
   - `SIC12-15 + {foot_l, ball_l}`
   - `calf_r global`
5. `GeoLocalDegWeighted / GeoDriftSlopeProxy / KeyBoneGeoLocalDegMean` 只作为 guardrail / tiebreaker。

若 selector 对齐后，当前 `Stage6` 在 downstream 上已明显改善，则应优先把 handoff 收敛为主结论，而不是立即做结构改造。

### Phase 2：固定 selector-aligned 上游，只做 `Stage6` consumption 诊断

这一阶段的目标不是“找一个最终最优结构”，而是回答三个更基础的问题：

1. `direct` 到底有没有真正依赖 `contact-plan`？
2. 它依赖的是“mode 语义”，还是“连续幅度基底”？
3. 2D `contacts_*` 带宽是否已经成为瓶颈？

建议原则：

- 固定上游为同一个 selector-aligned ckpt；
- 先只看 `Stage6` 出口，不急着把每个变体都接进 `70a -> new70b_replace_lowdrift -> 70R -> 71 -> 72 -> lambda` 全链；
- 保持 downstream canonical chain 不变，只把它作为通过 gate 后的 promotion 路径。

这一步的意义是降低 attribution 混淆：

- 若上游和 `Stage6` 同时变，无法知道收益来自哪里；
- 若 `Stage6` 和 downstream 同时变，也无法知道 sensitivity 出在 direct 还是后链吸收。

### Phase 3：只有通过 `Stage6-only gate` 的变体，才接入 canonical full chain

这一阶段才把候选变体接入当前 canonical chain：

- `Stage6 -> 70a -> new70b_replace_lowdrift -> 70R -> 71(lr=3e-4) -> 72(lr=1e-4) -> lambda`

对应运行契约见：

- `docs/posttrain_pipeline.md`

这样做的原因有两个：

1. 现有 Phase A 结论已经证明，`Stage6-only gate` 能有效淘汰不值得继续追的方向；
2. 在 `70R/71/72/lambda` 已锁定为 canonical downstream 的情况下，保持后链不变更利于归因。

对应历史经验：

- `docs/Problems/active/2026-03-06_trainbase_stage6_presplit_phaseA_checklist.md:1166`
- `docs/Problems/active/2026-03-06_trainbase_stage6_presplit_phaseA_checklist.md:1303`

## 6.7 推荐的 ablation gate

围绕“`contact-plan` 到底在 direct 中扮演什么角色”这个问题，建议把 ablation 组织成一套固定 gate，而不是零散试配方。

### Gate A：anchor necessity（它到底是不是必要锚点）

目标：

- 判断当前 direct 是否真的依赖 `contact-plan`；
- 避免出现“文档上说 anchored，实际上 head 已经学会 mostly ignore plan”的情况。

优先用现有实现中已经存在的开关：

- `direct_pose_detach_plan`
- `direct_pose_plan_drop_prob`

对应代码：

- `train/models.py:594`
- `train/models.py:604`

若评估 harness 支持 debug override，还可以补充：

- `direct_pose_plan_override='zero'`

对应代码：

- `train/models.py:3493`

判读：

- 若去掉 / 弱化 plan 后几乎不掉点，说明当前 direct 并没有真正把它当主锚点；
- 若轻微 corruption 就显著恶化，说明系统对连续 `contact-plan` 质量过于敏感，存在 over-consumption 风险。

### Gate B：mode-vs-continuous（它更像 mode selector 还是连续控制信号）

目标：

- 判断 direct 的主要收益来自离散 disambiguation，还是来自连续 contact 基底。

建议对比：

1. baseline：`direct_pose_meas_mode='concat'`
2. mode-select：`direct_pose_meas_mode='mode_select'`

对应代码：

- `train/models.py:600`
- `train/models.py:3635`

判读：

- 若 `mode_select` 在 lower-body 指标上更稳，且对 contact amplitude 噪声更不敏感，说明“角色分离”方向是对的；
- 若 `mode_select` 明显吃亏，说明当前问题未必在离散/连续耦合，而可能在更早的语义层或带宽层。

### Gate C：phase bandwidth（2D `contacts_*` 带宽是否已不够）

目标：

- 判断 direct 的瓶颈是否来自 `contacts_plan / contacts_meas` 的低带宽或高熵塌缩。

建议对比：

1. baseline：`direct_pose_use_phase_z=false`
2. append：`direct_pose_use_phase_z=true`, `direct_pose_phase_z_mode='concat'`
3. replace：`direct_pose_use_phase_z=true`, `direct_pose_phase_z_mode='replace_contacts'`

对应代码：

- `train/models.py:618`
- `train/models.py:623`
- `train/models.py:3617`

判读：

- 若 `replace_contacts` 优于 baseline，说明 2D `contacts_*` 的连续基底可能已成为 direct 的主要瓶颈；
- 若 `concat` 优于 `replace_contacts`，说明 contact-plan 仍有独立信息，phase_z 更适合作为补充而非替代。

### Gate D：hybrid factorization（离散分流 + 高带宽 phase hint）

目标：

- 检查更明确的“mode 负责分流、phase 负责 residual hint”是否比当前纯连续 contact 输入更稳。

在现有代码可直接尝试的最接近近似是：

1. `direct_pose_meas_mode='mode_select'`
2. 同时启用 `direct_pose_use_phase_z=true`

这样虽然还不是完全重构版的双层结构，但已经能回答一个关键问题：

> 如果先让 direct 按 mode 分流，再补充更高带宽 phase hint，是否能降低对 raw contact amplitude 的敏感度？

### Gate E：calibration instrumentation（补齐当前缺失的中间诊断）

除已有的 `ContactErrAbsMean / ContactPlanGtAbsMean / circular shift` 外，建议补几类中间诊断量，专门评估 calibration：

1. `midrange occupancy`
   - 统计 `contacts_plan` 落在 `[0.35, 0.65]` 的占比；
   - 用于识别长期高熵中间值塌缩。

2. `peak-trough gap`
   - 统计每个 contact channel 的峰谷差；
   - 用于识别 stance/swing separation 是否足够。

3. `transition width`
   - 统计 contact 从低到高 / 高到低穿过阈值区间的平均宽度；
   - 用于识别切换边界过宽或过窄。

4. `teacher-free gap`
   - 对同一 ckpt 比较 teacher 与 freerun 的上述指标差值；
   - 用于识别“teacher 好看但闭环不稳”的情况。

这类量的价值在于：

- 它们能把“semantic 问题”和“calibration 问题”拆开；
- 也能把“contact-plan 本身不行”和“downstream 对它过敏”拆开。

### Gate 的 promote / kill 规则

为避免再次把预算耗在 attribution 不清的方向上，建议固定 promote / kill 规则。

**Promote 到 full chain 的条件：**

1. 在 selector-aligned 上游下，`Stage6-only` 出口已出现稳定收益；
2. 收益至少同时体现在：
   - `contact-plan` 主面板中的一部分
   - downstream-aware lower-body 面板中的一部分
3. `nonleg / arm / global scalar` 没有出现不可接受的 guardrail 退化。

**直接 kill 的条件：**

1. 只改善 teacher，不改善 `cycle>=1` freerun；
2. 只改善 global scalar，却让 `leg8 / SIC12-15{foot_l,ball_l} / calf_r` 变差；
3. `Stage6-only gate` 已经不改善，却还想继续接 `70R/71/72/lambda` 复赌一次；
4. 需要同时更换 selector、`Stage6` 结构、downstream 配方才能“看起来变好”。

最后，围绕当前主题，最值得优先回答的不是“新结构能不能再涨一点”，而是三个更基础的问题：

1. `contact-plan` 是否真的被 direct 当成主锚点使用？
2. 当前收益主要来自 mode disambiguation，还是来自连续 contact 基底？
3. selector 对齐之后，剩余问题主要是 semantic、calibration、freerun stability，还是 downstream over-consumption？

只要这三个问题没有先回答清楚，任何更大规模的 `Stage6` 升级都容易再次掉回 attribution 不清的循环里。

---

## 7) 最终结论

关于“基于 `contact-plan` 锚点去特训 `direct expert` 这个选择本身是否正确”，本审计的最终答案是：

> **正确。**

更完整地说：

> 对当前 gait / lower-body 主链来说，这是结构上正确的主锚点选择；它的问题不在理念本身，而在于当前上游 handoff selector 还没有围绕这个目标对齐。

因此，当前优先级不应是：

- 先怀疑 `Stage6` 设计是否错了

而应是：

- 先把 `basetrain` 的 handoff selector 改成 `contact-plan / downstream-aware`
- 然后再评估 `Stage6` 是否还有剩余结构问题
