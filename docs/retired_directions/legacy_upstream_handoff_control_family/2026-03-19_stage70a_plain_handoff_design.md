# 70a 设计分析（plain handoff buffer / replace warmstart）

> Status: archived legacy upstream / handoff / control record
> Reader note: this file belongs to the old-boundary upstream-control investigation; any `current`, `default`, `canonical`, `recommend`, or `mainline` wording below is historical context, not present-tense repo policy.
> Current entry points: `docs/posttrain_pipeline.md`, `docs/train_design/2026-04-11_donor_contract_basin_universal_debug_synthesis.md`, `docs/retired_directions/legacy_upstream_handoff_control_family/README.md`

> Last updated: 2026-03-19
> 目标：整理当前主链中 `70a` 的设计职责、边界条件，以及后续更合理的改进路径。

关联文档：
- `docs/posttrain_pipeline.md`
- `docs/retired_directions/legacy_upstream_handoff_control_family/2026-03-19_stage6_contactplan_direct_audit.md`
- `docs/retired_directions/legacy_upstream_handoff_control_family/2026-03-19_basetrain_handoff_selector_contactplan.md`
- `docs/Problems/active/2026-03-14_oldd1_newflow_leg_regression_handoff.md`
- `docs/Problems/active/2026-03-14_oldd1_skip70b_replace_lowdrift_experiment.md`

---

## 0) TL;DR

1. `70a` 的核心定位不是“新语义阶段”，而是 **plain upstream cleanup / handoff buffer**。
2. 它的主要价值是：
   - 保持 `Stage6` 的 direct contract 基本不变；
   - 先把 plain 路线收口；
   - 为后续 `new70b_replace_lowdrift` 提供稳定 warmstart；
   - 给 `phase_z` / `replace_contacts` 改动提供清晰 attribution baseline。
3. 从当前证据看，`70a` 本身是有用的 cleanup step：`Stage6 -> 70a` 在 direct-path 的 leg / nonleg / arm 上都改善。
4. `70a` 不负责解决两类更上游或更下游的问题：
   - 它**不解决** `basetrain -> Stage6` selector mismatch；
   - 它**也不解决** 2D `contacts_*` hint 的带宽不足。
5. 因此后续更合理的方向不是“把 70a 做得更激进”，而是：
   - 先把上游 handoff selector 改成 contact-plan / downstream-aware；
   - 再把高带宽 phase hint、mode-vs-continuous factorization 等结构升级放在 `70a` 之后验证。

---

## 1) 当前链路里，70a 的准确位置是什么

当前 canonical chain 是：

`Stage6 -> 70a -> new70b_replace_lowdrift -> 70R -> 71(lr=3e-4) -> 72(lr=1e-4) -> lambda`

其中：

- `docs/posttrain_pipeline.md` 明确把 `70a` 定义为：
  - `last plain upstream stage`
- 同一文档又明确：
  - operational replace handoff 不是 raw `70b`
  - 而是 `70a -> new70b_replace_lowdrift`

这两个约束放在一起，说明 `70a` 的真实职责不是：

- 提前引入更强的 phase-hint 语义
- 或把 replace 逻辑前移到自己身上

而是：

> 在进入真正的 replace / phase-hint 改造之前，先保留 plain direct contract，做一次有控制的 cleanup，并输出一个更适合 downstream replace 使用的 warmstart。

---

## 2) 为什么 70a 需要保持 plain，而不是一开始就切 phase_z

## 2.1 Stage6 的主线本来就是 contact-plan anchored，不需要 70a 再改主题

`Stage6` 审计文档已经给出比较明确的结论：

- 对 gait / lower-body 主问题来说，`contact-plan` 作为 direct 的 primary anchor 是正确方向；
- 但它应是 `primary anchor, not sole source`；
- 当前更大的问题不像是 `Stage6` 理念错误，而像是 handoff selector 没有围绕这个目标对齐。

因此，在这个阶段如果 `70a` 立刻去做更强的 `phase_z` 路由改造，会带来一个分析问题：

- 你无法区分当前问题到底来自：
  - 上游 `Stage6` 起点质量不够；
  - 还是新引入的 phase-hint routing 改变了 direct 消费方式。

所以 `70a` 保持 plain 的意义，是先把下面这件事做干净：

> 在不改变 direct 主输入语义的前提下，确认 `Stage6` 这条 anchored plain 路线本身还能继续收敛多少、清理多少。

## 2.2 70a 的 config 设计，本质上是在延续 Stage6 contract

从 active config 看：

- `Stage6`:
  - `direct_pose_use_phase_z=False`
  - `direct_pose_phase_z_mode=concat`
  - 保留 split direct / arm split / leg-enable / proj256 / so3
- `70a`:
  - 同样保持 `direct_pose_use_phase_z=False`
  - 其余 direct 主结构与 `Stage6` 连续

也就是说，`70a` 不是另起炉灶，而是：

- 继续沿用 `Stage6` 的 plain direct contract；
- 不提前引入 `phase_z_in`；
- 不提前切 `replace_contacts`；
- 不把自己变成结构试验场。

这是一个非常强的设计信号：

> `70a` 的目标是 contract-preserving cleanup，而不是 semantic transition。

---

## 3) 70a 在系统里实际承担哪三种职责

## 3.1 Cleanup step：先把 Stage6 的 direct-path 噪声收一轮

从 `docs/Problems/active/2026-03-14_oldd1_newflow_leg_regression_handoff.md` 的 stage readout 看：

| stage | all_ex_root | leg | nonleg | arm |
|---|---:|---:|---:|---:|
| `Stage6` | 0.315735 | 0.865450 | 0.196877 | 0.222863 |
| `70a` | 0.275083 | 0.730911 | 0.176525 | 0.203549 |

对应增量：

| transition | d_all_ex_root | d_leg | d_nonleg | d_arm |
|---|---:|---:|---:|---:|
| `Stage6 -> 70a` | -0.040652 | -0.134539 | -0.020352 | -0.019313 |

这说明 `70a` 不是中性的占位阶段，而是一个真正有价值的 cleanup step：

- leg 明显改善；
- nonleg / arm 也同步改善；
- 并没有引入明显的局部 tradeoff。

所以从功能上说，`70a` 是：

> 先把 `Stage6` 的 plain direct 输出整理到一个更干净、更易接后续 phase-hint 改造的位置。

## 3.2 Attribution baseline：给后续 routing 改动提供可解释的对照组

紧接着的 raw `70b` 是：

- `direct_pose_use_phase_z=True`
- `direct_pose_phase_z_mode=concat`

而再后面的 replace 系列则切到：

- `direct_pose_phase_z_mode=replace_contacts`

如果没有 `70a` 这个 plain baseline，那么后面一旦出现：

- leg 变好但 nonleg / arm 变坏
- 或 phase 变稳但 calibration 漂了

就很难判断问题出在：

- 上游起点就不干净；
- 还是 `phase_z` 的 direct routing 本身带来了 tradeoff。

因此，`70a` 的第二个职责是：

> 在所有更激进的相位提示改造之前，提供一个语义不变、只做 plain cleanup 的对照节点。

## 3.3 Warmstart donor：作为 operational replace 的真实起点

当前链路已经明确：

- raw `70b` 是 diagnostic-only；
- real optimization handoff 是 `70a -> new70b_replace`；
- 现在 canonical pipeline 进一步锁成 `70a -> new70b_replace_lowdrift`。

这说明 `70a` 的第三个职责不是“自己完成 phase 升级”，而是：

> 作为 downstream replace stage 的真实 warmstart donor。

也就是说，系统默认认为：

- 直接从 `Stage6` 跳到 replace，不够稳；
- 直接把 raw `70b` 当 handoff，也不对；
- 更合适的方式是先经由一个 plain cleanup 节点，再进入 replace。

---

## 4) 70a 不是什么

为了避免后续设计讨论时把 `70a` 的责任范围说大，有必要明确它不承担什么。

## 4.1 它不是 selector 修复层

`Stage6` 审计已经指出：当前更大的系统问题像是 `basetrain -> Stage6` handoff selector mismatch，而不是 `Stage6` 的 anchor 理念错了。

这意味着：

- 如果上游 ckpt 没有满足 `semantic alignment + continuous calibration + freerun stability` 这三层契约；
- 那么 `70a` 只能在一个不够理想的起点上继续细调。

它能改善，但不能从根本上替代 selector 对齐。

## 4.2 它不是 phase-bandwidth 升级层

更早的 phase-hint 分析已经说明：

- 2D `contacts_plan / contacts_meas` 可能塌缩到高熵中间值；
- `phase_z_in` 作为更高带宽的 phase hint，在 `replace_contacts` 路由下可能更适合 direct 做 fine phase locking。

但这些都属于：

- 更强的 phase hint routing；
- 更偏后续阶段的结构升级。

`70a` 刻意不承担这个角色，正是为了：

- 不把 contract cleanup 和 bandwidth 升级混为一谈。

## 4.3 它不是最终 tradeoff 求解器

从当前链路读数看：

- `70a` 能把 `Stage6` 清理得更干净；
- 但真正大的 nonleg cleanup 在 `70R`；
- 真正大的 leg cleanup 在 `71`；
- replace 阶段还涉及明显的 leg / nonleg / arm tradeoff。

因此，`70a` 不是“把整条链一次解决”的阶段，而是：

- 稳定起点；
- 清晰边界；
- 可解释 handoff。

---

## 5) 为什么说 70a 的设计是合理的

## 5.1 它符合“先修 contract，再加高带宽 hint”的顺序

`Stage6` 审计给出的优先级是：

1. 先把 `basetrain -> Stage6` selector 改成 contact-plan / downstream-aware；
2. 再看 `Stage6` 是否还有结构性问题；
3. 只有在 selector 对齐后问题仍持续，才进一步考虑：
   - direct input 带宽不足
   - direct 对 contact-plan 过度敏感
   - mode / continuous 职责耦合过强

把这套优先级映射到 Stage7 入口，就会得到一个很自然的设计原则：

- `70a` 不应该抢先承担“带宽升级”任务；
- `70a` 更应该保持 plain，扮演 contract-preserving handoff buffer。

## 5.2 它降低了结构分析的耦合度

如果 `70a` 也同步引入：

- `phase_z_in`
- `replace_contacts`
- 或更强的 mode routing

那么 `Stage6 -> 70a` 这一步就同时混入了：

- 起点质量变化
- 训练继续收敛
- 输入路由变化
- 带宽变化

这种设计虽然未必一定更差，但会显著提高 root-cause attribution 的难度。

而当前 `70a` 的设计恰恰相反：

> 它主动把“plain cleanup”和“phase-hint rerouting”拆到两个阶段里。

## 5.3 它已经被现有链路结果验证为有实际收益

从 old d1 newflow 的结果看：

- `Stage6 -> 70a` 是稳定改善；
- raw `70b` 才是第一个明显 regression 点；
- 真实 operational replace 又是从 `70a` warmstart，而不是从 raw `70b` 接下去。

这表明 `70a` 不是历史遗留的冗余节点，而是：

- 对当前链路确实有实际价值；
- 并且已经形成了清晰的 downstream 依赖关系。

---

## 6) 当前 70a 设计的短板和边界

虽然我认为 `70a` 的设计方向是对的，但它也有天然边界。

## 6.1 它改善的是 plain path，不是 handoff quality definition 本身

如果 `best_free` 选出来的 basetrain ckpt 在 pose/global scalar 上好看，但 contact-plan 语义一般，那么：

- `Stage6` 的起点会带偏；
- `70a` 只能在这个带偏起点上继续清理；
- 最终仍可能把 downstream 的 replace / recovery 压力放大。

所以 `70a` 的收益上限，受上游 selector 定义直接限制。

## 6.2 它无法回答“direct 主要依赖的到底是 mode 还是 continuous contact amplitude”

`Stage6` 审计特别强调了一个结构性张力：

- `contact-plan` 在设计语言里是 anchor；
- 但在实现上又被 direct 当作连续输入直接消费。

`70a` 由于刻意保持 plain，不会主动去拆这个问题。

所以如果未来要继续进化结构，真正该做的是：

- 在 `70a` 之后做 mode-vs-continuous 的 gate ablation；
- 而不是让 `70a` 自己承担所有结构实验。

## 6.3 它不是永久不可替代的

如果未来出现下面两个条件：

1. `basetrain -> Stage6` selector 已经按 contact-plan / downstream-aware 对齐；
2. `Stage6` 直接进入 low-drift replace 也能稳定，不再需要额外 plain cleanup 才能形成好 handoff；

那么 `70a` 是可以被重新评估是否需要保留的。

也就是说：

> `70a` 当前是合理且有用的，但它的存在仍然应被视为一个系统层的设计选择，而不是不可触碰的永久定律。

---

## 7) 对后续改进思路的建议

## 7.1 第一优先级：先修 `basetrain -> Stage6` handoff selector

这应该仍然是最高优先级，因为它决定 `70a` 能拿到什么起点。

建议：

1. 引入与 `best_free` 并行的 `best_handoff` 选择口径；
2. 用 `contact-plan + downstream proxy` 作为主要依据，而不是只看 pose/global scalar；
3. 最少覆盖以下三层指标：
   - semantic alignment
   - continuous calibration
   - freerun stability

如果这一步没做，后面对 `70a` 或 replace 的很多争论都会混入起点偏差。

## 7.2 第二优先级：把 70a 的验收标准从“只看 direct 指标”扩成“plain handoff readiness”

目前 `70a` 的直观收益已经能从 direct-path metrics 看出来，但如果要把它作为更明确的设计节点，建议补一个固定 readout：

### A. direct cleanup 面板

- `all_ex_root`
- `leg`
- `nonleg`
- `arm`
- 关键 hotspot（如 calf / foot / ball 的 SIC 区间）

### B. contact-plan readiness 面板

- `contacts_plan` 相对 GT/meas 的 phase alignment
- `contacts_plan` 的 mid-range collapse 占比
- `|contacts_plan - contacts_meas|` 的均值和 cycle>=1 保持情况
- freerun 下的 shift / amplitude stability

这样可以把“70a 是个好 cleanup step”进一步升级为：

- “70a 是个好 handoff step”

## 7.3 第三优先级：把结构升级放到 70a 之后，而不是塞进 70a 本身

在 selector 对齐之后，后续更值得做的是 `70a` 之后的固定 gate ablation，而不是继续把 `70a` 本体做复杂。

建议优先顺序：

### Gate A：anchor necessity

检查 direct 是否真的依赖 `contact-plan`：

- `direct_pose_detach_plan`
- `direct_pose_plan_drop_prob`
- 若 harness 支持，再加 debug override

目的：

- 判断当前 anchor 是不是实锚点；
- 判断系统是否对连续 contact-plan 过度敏感。

### Gate B：mode-vs-continuous

比较：

- `direct_pose_meas_mode='concat'`
- `direct_pose_meas_mode='mode_select'`

目的：

- 看 direct 真正需要的是 mode disambiguation，还是连续 contact amplitude。

### Gate C：phase bandwidth

比较：

1. baseline：`direct_pose_use_phase_z=false`
2. append：`direct_pose_use_phase_z=true`, `direct_pose_phase_z_mode='concat'`
3. replace：`direct_pose_use_phase_z=true`, `direct_pose_phase_z_mode='replace_contacts'`

目的：

- 判断 2D `contacts_*` 是否已经成为带宽瓶颈；
- 判断 `phase_z_in` 是补充更好，还是替代更好。

### Gate D：hybrid factorization

如果 Gate B / C 结果都支持，可以进一步做：

- mode 负责分流
- high-bandwidth phase hint 负责 residual hint

也就是把：

- `mode-selection`
- `continuous residual control`

这两种职责显式拆开，而不是继续让单个 `contacts_plan` 同时承担两件事。

## 7.4 第四优先级：继续保持 `70a` 的 plain 语义，不建议把它改成“小 70b”

我不建议下一轮改进把 `70a` 做成：

- 半 phase-z 化
- 半 replace 化
- 或加很多只在 70a 出现的临时开关

因为这样会直接破坏它当前最有价值的性质：

> 作为 plain cleanup / attribution baseline / replace warmstart donor 的角色清晰性。

---

## 8) 建议的后续决策规则

为了避免今后反复讨论“70a 到底要不要继续保留”，建议把判断规则写清楚。

### 保留 70a 的条件

如果满足以下任一情况，应继续保留 `70a`：

1. `Stage6 -> 70a` 在 direct cleanup 上仍有稳定收益；
2. downstream replace 仍明显依赖 `70a` warmstart 才能更稳；
3. `70a` 仍是后续结构变体最清晰的 attribution baseline。

### 重新评估 70a 的条件

如果同时满足以下条件，可重新评估是否压缩 `70a`：

1. 上游 selector 已按 contact-plan / downstream-aware 对齐；
2. `Stage6` 直接进入 low-drift replace 也能稳定；
3. `70a` 相对 `Stage6` 的 plain cleanup 收益已经接近零；
4. 去掉 `70a` 不会让 downstream 的 attribution 明显变糊。

这时才值得讨论：

- `Stage6 -> lowdrift replace` 直连；
- 或把 `70a` 吸收回更短的 upstream schedule。

---

## 9) 最终结论

关于 `70a` 的设计思路，我的结论是：

> `70a` 的正确理解，不是“一个过渡性的普通 stage”，而是当前主链里有明确职责的 plain handoff buffer。

更完整地说：

- `Stage6` 负责把系统带入 `contact-plan anchored direct` 的主线；
- `70a` 负责在不改变主输入语义的情况下，把这条 plain 路线先清理干净；
- `new70b_replace_lowdrift` 之后的阶段，才开始承接更强的 phase-hint routing 与下游恢复。

因此，当前更合理的策略不是：

- 质疑 `70a` 为什么没更激进

而是：

- 保留 `70a` 的 plain 角色清晰性；
- 先修上游 selector；
- 再把更激进的 phase / mode 因子化实验放到 `70a` 之后做。

这条顺序更符合当前 repo 已有证据，也更利于后续做可解释的结构升级。
