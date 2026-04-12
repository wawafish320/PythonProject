# 2026-04-08 cp015 tailk7 upstream replace-transferability experiments

> Status: archived / superseded planning memo
> Reader note: this pre-framework experiment memo was superseded by `docs/train_design/2026-04-09_top3_anchor_top7_expansion_framework.md` and later execution records; any `mainline` / `recommend` / `current` wording below is historical.
> Current entry points: `docs/posttrain_pipeline.md`, `docs/train_design/2026-04-09_top3_anchor_top7_expansion_framework.md`, `docs/train_design/2026-04-12_top7_clean_stage6_stepc_causality_record.md`

## 1. Purpose

### Follow-up framework

- Post-E3-A framework memo:
  `docs/train_design/2026-04-09_top3_anchor_top7_expansion_framework.md`

这份文档是一个新的 upstream experiment memo。

它的目标不是继续细化 downstream attribution，而是把当前问题正式改写为：

- **哪些 stage6 / 70a 训练条件，能让 full top7 direct branch 落到 replace-compatible basin？**

换句话说：

- 不是去追问 “正确的 direct-branch geometry 长什么样”
- 也不是默认 “top7 scope 本体一定错”
- 而是要找出：
  - 哪些 supervision scope
  - 哪些 trainable-scope allocation
  - 哪些 optimization path
  - 会把 direct branch 训练到 **native stage6 自洽但 replace-transfer 不兼容** 的解


## 2. Inherited conclusions

本 memo 直接继承前序结论，不重复证明：

- root cause 不在 planner semantics 主线
- root cause 不在 replace entry 外部 rollout state
- root cause 不在 `contacts_in_t`
- earliest semantic split 在 `direct_pose_head` boundary
- current first-step split 最像 **whole direct-branch contract mismatch**
- current dominant readout 是 leg block，但 leg-dominant readout != leg root cause
- `direct_pose_head` 是 earliest boundary / necessary anchor，但不是 standalone sufficient module
- 只有 7-module direct branch joint swap 才能在 weight-space 把 closure 关到 near-1

本 memo 还继承一个非常关键的 existence proof：

- **baseline 的 7-module direct branch transplant 到 coadapt context 后，可以把 replace 指标从约 `0.213` 修到约 `0.150`，且 pose-side 其他部分不动**

这个结果的含义是：

- coadapt trunk / planner / interface **不是**根因
- full 7-module direct branch **不是不可能 work**
- 当前问题不是 “top7 branch impossible”
- 而是：
  - **当前 stage6 / 70a 训练路径把 top7 direct branch 带进了一个 self-consistent but non-transferable basin**


## 3. Why pivot now

继续做 attribution 的边际收益已经明显下降。

原因：

1. 当前定位已经足够精确  
   已经知道问题位于 7-module direct branch 的 joint weight geometry / joint contract，而不是某个 isolated single module。

2. downstream transplant 已经证明了 existence  
   所以现在最该问的不是 “哪里坏”，而是 “上游怎样才能训出兼容的 direct branch”。

3. 还未解的核心问题只剩一个  
   - **为什么 top7 的 stage6 / 70a direct branch 在 replace 阶段不 work，而 top3 的 work？**

因此主线应从：

- downstream attribution / falsifier

切换到：

- upstream replace-transferability experiment


## 4. Reframed question

后续实验的主问题应表述为：

- **What upstream training path produces a replace-compatible direct-branch contract?**

而不是：

- “哪种形态才是正确 geometry”
- “top7 scope 本体是不是错”

更细一点，可以拆成三个候选机制：

### H1. Support problem

- top7 effective support 一上来就把训练轨迹推离 replace-compatible basin

### H2. Path problem

- top7 本身并不错误
- 但 `from-step0 top7` 的训练路径错误
- 可能需要 `top3 -> top7` curriculum / ramp 才能落到 compatible basin

### H3. Co-adaptation allocation problem

- 不是 supervision target 错
- 而是 `head / adapters / readouts` 同时自由 co-adapt，导致形成 native-stage6 自洽、但 replace-transfer 不兼容的 joint contract


## 5. Core metric: replace-transferability

后续 upstream 实验不能只看 native stage6 / 70a loss。

必须引入一个固定的 **replace-transferability assay**，作为每个 candidate checkpoint 的共同评测。

### Required assay

对每个 stage6 / 70a candidate checkpoint：

- 放到固定 replace context
- 保持 deterministic single-step / first-forward
- 评测以下量：
  - `out_direct` gap
  - `dir_base` gap
  - `dir_leg` gap
  - `dir_nonleg` gap
  - 对 baseline transplant target 的 closure ratio

### Why this is the right metric

因为当前已知：

- native stage6 上可以 self-consistent
- 但 replace 阶段仍然不 work

所以 downstream compatibility 必须变成 upstream model selection 的正式目标，而不是事后解释。


## 6. Minimal upstream experiment ladder

这里给出推荐的最小实验顺序。

重点不是一次性开很多 lane，而是尽快回答：

- **scope 问题？path 问题？还是 co-adaptation allocation 问题？**


### E0. Checkpoint archaeology / transfer curve

这是最优先的实验，且如果已有 stage6 / 70a 中间 checkpoint，成本最低。

#### Record

- E0 record:
  `docs/retired_directions/top7_old_boundary_transferability_family/2026-04-08_cp015_tailk7_upstream_replace_transferability_e0_record.md`

#### Goal

不新训练，先看当前 stage6 / 70a 过程中：

- replace-transferability 是从一开始就差
- 还是训练后期才崩
- 还是先好后坏

#### Method

对现有 stage6 / 70a 中间 checkpoints 做统一 assay：

- fixed replace context
- deterministic first-forward / single-step
- 记录：
  - `out_direct`
  - `dir_base`
  - `dir_leg`
  - `dir_nonleg`
  - closure to transplant-compatible target

同时增加一组**低成本 proxy telemetry**，但明确降级为 auxiliary panel，而不是主判据：

- 对每个中间 checkpoint 同步记录 7-module direct branch 的 weight statistics
- 最低要求至少包含 `direct_pose_head.0` 的输入 block statistics：
  - `plan` block norm per dim
  - `direct` block norm per dim
  - `meas` block norm per dim
- 以及派生比值：
  - `plan/direct`
  - `plan/meas`
  - `plan/(direct+meas)`

这里必须强调两点：

1. **口径必须与前序 audit 保持一致**  
   尤其是 `norm per dim` 的定义不能临时改写，否则时间序列不可比。

2. **这些量不是新的主因判据**  
   当前已知 `plan weight` 更像 surface indicator / cheap proxy，而不是 root cause 本体。

因此 E0 的每个 checkpoint 应同时产出两层结果：

- **Primary transfer metrics**  
  `out_direct / dir_base / dir_leg / dir_nonleg / closure`

- **Auxiliary proxy metrics**  
  `direct_pose_head.0` 的 `plan/direct/meas` block norm-per-dim 与其 ratio curve

#### What it tells us

- 如果 compatibility 从 early stage 就差：更像 support / initialization 问题
- 如果 early compatibility 好、later 才坏：更像 optimization path / co-adaptation drift
- 如果有明显 best epoch：后续可以把 stage6 model selection 改成 transfer-aware selection

再进一步，加入 proxy telemetry 后，E0 还能回答下面这个更窄的问题：

- **transferability 崩坏的时间点，是否与 head input-block allocation 的变化同步？**

最有价值的几种读法是：

- **同步拐点**  
  transferability 变差，同时 `plan` block norm-per-dim 或其 ratio 明显跳变  
  → 说明该 proxy 至少能帮助缩小问题形成窗口

- **proxy 先变，transfer 后坏**  
  → 说明它可能是 leading indicator

- **transfer 先坏，proxy 后变**  
  → 说明它更像滞后 readout

- **长期不同步**  
  → 说明 `plan weight` 连 proxy 都不是，应进一步降级

#### Decision value

这是最该先做的 upstream 实验，因为它先回答“问题从什么时候形成”。

而且加上这组 telemetry 后，E0 不只是给出一条 compatibility curve，还会给出一条：

- **head input-block allocation curve**

这样即使最后证明 `plan weight` 不是 root cause，它仍然可能是一个：

- 廉价窗口定位器
- 或被正式证伪的 proxy 指标


### E1. Scope isolation

这是最直接的最小新训练实验。

#### Record

- E1 record:
  `docs/retired_directions/top7_old_boundary_transferability_family/2026-04-08_cp015_tailk_support_scope_isolation_e1_record.md`

#### Goal

隔离 “effective direct-pose support scope” 是否是强因果杠杆。

#### Design

在 tailk7 的 stage6 / 70a pipeline 上：

- 保持其余配置尽量不变
- 只把 direct pose loss 的 effective support 从 `top7` 收回 `top3`

也就是比较：

- `E1-A`: current `top7` support
- `E1-B`: same pipeline but `top3` support

#### Readout

不是只看 native 训练 loss，而是看：

- 产出的 70a checkpoint 进入 replace 阶段后是否正常优化
- 以及 fixed transfer assay 的 single-step closure

#### Interpretation

如果 `E1-B` 显著恢复 transferability：

- 只能说明 **support/scope 是强因果杠杆**
- **不能**直接说 “top7 scope 本体就是 root cause”

因为它也可能说明：

- top7 需要不同的 curriculum
- top7 需要不同的 trainable-scope allocation
- top7 需要不同的 optimizer path


### E2. Path-shaping / curriculum experiment

如果 E1 指向 support/path 问题，下一步优先做这个，而不是马上扫一堆超参。

#### Goal

测试：

- top7 supervision 是否本身可行
- 只是需要更平滑的 path 才能到 replace-compatible basin

#### Recommended variants

**E2-A. top3 warmup -> top7 ramp**

- E2-A record:
  `docs/retired_directions/top7_old_boundary_transferability_family/2026-04-08_cp015_tailk_curriculum_e2a_record.md`

- early stage: `top3`
- mid stage: gradually expand to `top7`
- late stage: full `top7`

**E2-B. readout-first -> full-branch**

- 先限制更小的 trainable direct subset
- 再逐步解冻 `head / adapters / readouts`

**E2-C. leg-first -> nonleg expansion**

- E2-C record:
  `docs/retired_directions/top7_old_boundary_transferability_family/2026-04-08_cp015_tailk_legfirst_e2c_record.md`

- 先让 dominant leg readout block 稳住
- 再扩到 arm / else path

#### Why this is preferred over immediate LR sweeps

当前问题更像：

- joint contract formation problem

而不是：

- 单纯的 optimizer magnitude 太大 / 太小

因此 path / curriculum 的优先级应高于纯 LR 微调。


### E3. Co-adaptation allocation

如果 E2 仍然不 work，再测 trainable-scope allocation。

#### Record

- E3-A record:
  `docs/retired_directions/top7_old_boundary_transferability_family/2026-04-08_cp015_tailk_allocation_e3a_record.md`

#### Goal

测试是不是因为 head / adapters / readouts 同时自由 co-adapt，导致 geometry drift 到 non-transferable solution。

#### Candidate variants

**E3-A. freeze head, train readouts/adapters first**

**E3-B. freeze readouts, adapt head first**

**E3-C. head + adapters first, readouts later**

**E3-D. leg path first, nonleg path later**

#### What it addresses

这个实验不是改 supervision target，而是在改：

- 哪些模块允许共同形成 contract
- 哪些模块需要分阶段 co-adapt


### E4. Local optimization dynamics

只有在 E2 / E3 已经表明 path 对了但 still unstable，才值得做这一层。

#### Goal

调节局部 training dynamics，而不是主线结构。

#### Candidate knobs

- direct head LR scale
- branch-specific LR scale
- leg / nonleg loss weighting
- adapter / readout regularization
- gradient clipping / norm balancing

#### Warning

这一步应视为 second-order tuning。

如果还没回答：

- scope 是否强因果
- path 是否强因果
- co-adaptation allocation 是否强因果

就过早扫超参，信息价值会很低。


## 7. Recommended execution order

建议严格按下面顺序走：

1. **E0 checkpoint archaeology**
2. **E1 scope isolation**
3. **E2 curriculum / path-shaping**
4. **E3 trainable-scope allocation**
5. **E4 local optimization tuning**

这样做的好处是：

- 先最大化单次实验的信息密度
- 避免在 attribution 已经充分时继续做低边际收益 probing
- 避免把问题过早表述成 “调参问题”


## 8. Decision rules

为了避免后续实验越做越散，先把判定规则写死。

### If E0 shows early good / late bad

主线转向：

- path drift / co-adaptation drift

优先做：

- E2
- E3

### If E1 top3-support restores transferability

主线转向：

- support / path interaction

优先做：

- E2-A `top3 -> top7 ramp`
- E3-D `leg first -> nonleg later`

### If E1 top3-support still does not restore transferability

主线转向：

- stage6 / 70a 里还有其他 config 差异在起作用

优先做：

- config diff audit
- E0 + config-factor ablation

### If E2 yields a working top7 run

结论应写成：

- **top7 is viable, but only under a transfer-compatible training path**

而不是：

- “top7 原来没问题，只是运气不好”


## 9. What not to do next

以下方向暂时不应再回到主线：

- planner semantics 主线
- 外部 rollout state 主线
- `contacts_in_t` 主线
- per-SIC / per-joint 主分析
- 再做更细的 direct-branch attribution tree search

这些内容当前最多作为 side diagnostic，不应再吃掉主实验预算。


## 10. Current best statement of the problem

到目前为止，最合适的问题表述不是：

- “top7 direct branch 必须长成某种指定形态”

而是：

- **current stage6 / 70a training path can produce a top7 direct branch that is native-stage self-consistent but replace-transfer incompatible**

因此 downstream transplant + attribution 的任务已经基本完成。

后续真正要做的是：

- **design an upstream training path that reliably produces replace-compatible direct-branch contract**


## 11. Immediate next action

如果只选一个最值得立刻开的上游动作：

- **先做 E0：checkpoint archaeology / transfer curve**

如果现有中间 checkpoint 不足以支持 E0，再开：

- **E1：same pipeline, top7 -> top3 support isolation**

这是当前信息增益最高、同时最不容易把问题表述错的 upstream 起点。
