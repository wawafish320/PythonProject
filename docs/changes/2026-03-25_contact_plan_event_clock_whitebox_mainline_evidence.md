# 2026-03-25 Contact-Plan / Event-Clock 白盒调试主线数据论证

> Status: companion evidence memo for `main`
> Goal: 给“为什么主线只需要保留最小 white-box 观测量与 probe”提供独立的数据支撑。

Companion playbook:

- `docs/changes/2026-03-25_contact_plan_event_clock_whitebox_mainline_playbook.md`

---

## 1. 这份文档要回答什么

上一份文档 `docs/changes/2026-03-25_contact_plan_event_clock_whitebox_mainline_playbook.md` 讲的是：

- 主线应该保留哪些 debug hooks；
- 哪些实验性训练逻辑不该带回 `main`；
- 为什么 `plan_z_prev / plan_z_raw / lambda_corr / delta_z / meas-derived inputs` 这批量最有归因价值。

这份文档单独回答一个更具体的问题：

**这些判断不是拍脑袋得出的，而是哪些历史数据把它们支撑起来的？**

本备忘录只做“证据 -> 结论”整理，不引入新的实验设计。

---

## 2. 证据来源

本页结论主要整理自下面两份历史记录：

- `docs/gait_speed_scaling_whitebox_evaluation.md`
- `docs/Problems/active/2026-03-15_72_lowlr_to_lambda.md`

其中最关键的是三轮 white-box follow-up：

1. planner speed-zero probe
2. planner state-path probe
3. planner write-back probe

这三轮 probe 的价值在于：

- intervention 很小；
- 每轮只改一条路径；
- 可以直接把“当前步显式输入”“carry state”“corrector content”“meas-driven signals”分开看。

---

## 3. 核心数据结论

### 3.1 结论 A：显式 speed -> planner input 是主要耦合路径，但不是唯一来源

证据见：

- `docs/gait_speed_scaling_whitebox_evaluation.md`
  - `15.7 planner speed-zero probe`
- `docs/Problems/active/2026-03-15_72_lowlr_to_lambda.md`
  - `Gait speed white-box plan speed-zero probe`

关键读数：

- speed-zero 后，off-scale mean-abs delta 平均下降：
  - raw：`0.572`
  - base：`0.575`
  - final：`0.575`
- base mean-logit span across scales 显著压缩：
  - baseline：left/right=`0.959 / 0.821`
  - speed-zero：left/right=`0.377 / 0.219`

这说明：

- `speed` 直接喂入 `contact_plan_cell` 的确是主要 coupling path；
- 所以主线必须能看到 planner 输入之后、corrector 之前的状态与读出。

但 speed-zero 后仍然有明显残余：

- raw overall mean-abs delta 仍有 `0.107 ~ 0.265`
- base overall mean-abs delta 仍有 `0.104 ~ 0.259`

所以这轮数据也同时说明：

- 仅看当前步显式输入不够；
- residual 不可能只来自 “当前步 speed 通道直连”；
- 还必须去看 stateful path，特别是 `plan_z_prev` carry。

**对主线 observability 的含义**：

- 不能只导出最终 `contacts_plan_logits`；
- 至少还要保留：
  - `event_clock_plan_z_raw`
  - `contacts_plan_logits_base`
  - `plan_z_prev`

---

### 3.2 结论 B：speed-zero 后剩余 drift 的主因是 carry path，而且是被 Event-Clock 写回后的 carry

证据见：

- `docs/gait_speed_scaling_whitebox_evaluation.md`
  - `15.8 planner state-path probe`
- `docs/Problems/active/2026-03-15_72_lowlr_to_lambda.md`
  - state-path probe readout / decision

关键读数：

- speed-zero 之后，如果关掉当前步 Event-Clock correction：
  - raw：`0.1667 -> 0.0000`
  - base：`0.1610 -> 0.0000`
  - final：`0.1610 -> 0.0000`
- speed-zero 之后，如果 donor `plan_z_prev`：
  - raw drift：直接到 `0`
  - base / final drift：降到约 `0.0199`
  - 相对 speed-zero residual，base / final 再降约 `87.7%`

这些数字最重要的含义不是“哪个 intervention 更强”，而是：

- 一旦把 entering planner 的 `plan_z_prev` 换成 `1.0x` donor，
  - **同样的当前步 planner input 下，raw drift 就消失了**；
- 所以 residual 主因不在当前步显式 speed 通道，而在 carry path；
- 更准确地说，是 **Event-Clock correction 写回 `plan_z` 后形成的 carry**。

文档里还明确给了更细的解释：

- donor 后残余大约 `12.3%` 的 final drift 才是“纯当前步 correction residual”
- 主矛盾是 **write-back 后再被 recurrent carry 放大**

**对主线 observability 的含义**：

- `plan_z_prev` 不是可有可无的字段，而是核心字段；
- 只看 `event_clock_plan_z_raw` 还不够，必须知道它是从什么 carry state 进来的；
- 主线应该优先保留能区分：
  - `plan_z_prev`
  - `event_clock_plan_z_raw`
  - `event_clock_plan_z_t`

否则无法回答“问题是当前步产生的，还是上一步带进来的”。

---

### 3.3 结论 C：donor residual 的主因不是 `lambda_corr` 标量，而是 `delta_z` / corrector write-back content

证据见：

- `docs/gait_speed_scaling_whitebox_evaluation.md`
  - `15.9 planner write-back probe`
- `docs/Problems/active/2026-03-15_72_lowlr_to_lambda.md`
  - write-back probe readout / decision

关键读数：

- 以 speed-zero + donor `plan_z_prev` 为 baseline，off-scale final drift 约 `0.019869`
- donor `lambda_corr` only：
  - final：`0.019869 -> 0.019816`
  - 只再降约 `0.27%`
- donor `delta_z` only：
  - final：`0.019869 -> 0.000708`
  - 再降约 `96.43%`

这组数字的解释很直接：

- `lambda_corr` mismatch 不是 donor residual 的主因；
- 剩余问题主要在 `delta_z`，更准确地说是 **current-step corrector write-back content**。

这也是为什么上一份文档强调：

- 单独看 `lambda_corr` 很容易错怪 gate；
- `delta_z` 比 `lambda_corr` 更接近问题内容本身；
- 最值得额外保留的派生量，是：

```text
applied_correction = lambda_corr * delta_z
```

**对主线 observability 的含义**：

- `event_clock_lambda_corr` 要留；
- 但更重要的是 `event_clock_delta_z` 必须留；
- 最好还能离线计算 `lambda_corr * delta_z`；
- 如果只能保留一个“corrector内容量”，优先级应是 `delta_z` 而不是 `lambda_logit`。

---

### 3.4 结论 D：`time_term` 不是当前 residual 的主要来源

证据见：

- `docs/gait_speed_scaling_whitebox_evaluation.md:1458` 之后的 write-back probe readout
- `docs/Problems/active/2026-03-15_72_lowlr_to_lambda.md:390` 附近的 runtime facts / readout

关键读数：

- no time term only：
  - final：`0.019869 -> 0.019853`
  - 只再降约 `0.08%`

并且 runtime facts 明确写了：

- `time_term` 只加在 final logits 上；
- 它不参与 `plan_z` write-back；
- 它也不参与下一步 carry。

所以当前这批问题里：

- `time_term` 不是主矛盾；
- 它更像一个 readout residual，而不是 planner carry residual 的源头。

**对主线 observability 的含义**：

- `event_clock_time_term` / `contacts_plan_logits_time` 有诊断价值；
- 但它们不是主线最小集合的第一优先级；
- 如果主线需要先做“最小回收”，完全可以先不把这批量当强依赖。

---

### 3.5 结论 E：当前 donor residual 主要由 meas-derived signals 驱动

证据见：

- `docs/gait_speed_scaling_whitebox_evaluation.md`
  - `15.9.2 exported tensors 给出的主结论`
  - `15.9.3 如何回答这轮最关键的问题`
- `docs/Problems/active/2026-03-15_72_lowlr_to_lambda.md`
  - write-back probe readout / decision

关键读数：

- donor meas-derived inputs 直接把 residual 压到 `0`
- 并且：
  - `lambda_corr`
  - `delta_z`
  - `dynamic_prior`
  - final logits
  的 off-scale delta 也一起归零

同时 runtime facts 还给了一个重要约束：

- 当前 checkpoint 下 `period_feat_dim = 0`
- 也就是说这轮路径里没有独立 `period_feat` 分支

因此更强的解释是：

- residual 主要不是“抽象 periodic prior 自己漂了”；
- 而是 **meas-derived drive** 触发了 gate / corrector 的错位响应。

**对主线 observability 的含义**：

- 不能只保留 `lambda_corr` 和 `delta_z`；
- 还必须保留这批 corrector 输入量：
  - `event_clock_contacts_meas`
  - `event_clock_delta_meas`
  - `event_clock_err_raw`
  - `event_clock_lr_diff`

否则你只能看到“corrector 做错了”，却看不到“它是被什么输入驱动着做错的”。

---

## 4. 为什么这些数据支持“只回收最小 debug hooks，而不是整包训练逻辑”

这三轮数据有一个很一致的特征：

- 它们的关键结论都来自 **diagnostic-only / inference-time only intervention**；
- 它们并不依赖新的训练损失真的优化成功；
- 它们靠的是“导出足够中间量 + 做最小 donor/override”。

换句话说，真正让结论成立的并不是：

- `touchdown_guard`
- `fixed_ref`
- `joint_consistency`
- `lambda_train_planner_scope`
- `lambda_grad_probe_*`

而是：

- 是否能看到 `plan_z_prev -> plan_z_raw -> delta_z -> plan_z_t -> logits_base` 这条链；
- 是否能替换其中 1~2 个关键量做 counterfactual；
- 是否能把所有 scale 写成统一 export 结构。

因此，从数据上看，最合理的主线回收策略就是：

1. 先回收最小 observability
2. 不要先回收实验性训练配件

这不是“保守”，而是和现有证据最一致的做法。

---

## 5. 最终落到主线时，哪些字段最值钱

结合上面所有证据，按信息价值排序，我会把主线 white-box 字段分成三层：

### 第一层：必须有

- `plan_z_prev`
- `event_clock_plan_z_raw`
- `event_clock_err_raw`
- `event_clock_lambda_corr`
- `event_clock_delta_z`
- `contacts_plan_logits_base`

这层已经足够回答：

- residual 是 carry 还是 current-step；
- 主因在 gate 还是 corrector content；
- planner 行为是从 hidden 还是从 readout 才开始失真。

### 第二层：强烈建议有

- `event_clock_contacts_meas`
- `event_clock_delta_meas`
- `event_clock_plan_z_t`
- `contacts_plan_logits`

这层能把 meas path、corrector 输出、最终行为连接起来。

### 第三层：可选增强

- `event_clock_lr_diff`
- `event_clock_lambda_logit`
- `event_clock_dynamic_prior`
- `event_clock_time_term`
- `contacts_plan_logits_phase`
- `contacts_plan_logits_time`

这层更适合做进一步分支排查，不是当前最小主线回收的阻塞项。

---

## 6. 一句话总结

现有数据最稳定地支持了三件事：

1. **主矛盾不是最终输出本身，而是 planner hidden state 的 write-back / carry**
2. **主因不是 `lambda_corr` 标量，而是 meas-driven 的 `delta_z` / corrector content**
3. **因此主线最值得回收的是“能观察并干预这条链”的最小 white-box hooks，而不是实验性训练逻辑**

这也是为什么，把“主线落地说明”和“数据论证”拆成两份文档，是一个比较干净、也比较适合提交到 `main` 的做法。
