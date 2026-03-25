# 2026-03-25 Contact-Plan / Event-Clock 白盒调试主线落地说明

> Status: proposal for `main`
> Goal: 只把真正有复用价值的 white-box debug 观测能力带回主分支；不把当前分支里的实验性 posttrain/loss/scaffold 一起带回去。

Companion evidence memo:

- `docs/changes/2026-03-25_contact_plan_event_clock_whitebox_mainline_evidence.md`

---

## 1. 结论先行

当前工作树里最有价值的部分，不是新的 `lambda` 训练逻辑本身，而是：

- 能把 `contact_plan` / `event_clock` 的中间状态白盒导出来；
- 能在 rollout 里做最小 counterfactual override；
- 能用统一 JSON 导出去做 donor / ablation 对比。

因此，如果目标是给 `main` 一个稳定、可复现、低维护成本的 debug 能力，建议：

1. **先只提交文档**
2. 后续如果真的要落地代码，只回收最小 debug hooks
3. **不要**把当前分支的实验性 loss、训练开关、grad probe、临时工具整包带回 `main`

---

## 2. 本次建议带回主线的东西

### 2.1 必须保留的观测量

主线如果要复现这套诊断，最小需要导出这些 per-step 张量：

- `plan_z_prev`
- `contacts_plan`
- `contacts_plan_logits`
- `contacts_plan_logits_base`
- `event_clock_plan_z_raw`
- `event_clock_plan_z_t`
- `event_clock_lambda_corr`
- `event_clock_delta_z`
- `event_clock_contacts_meas`
- `event_clock_delta_meas`
- `event_clock_err_raw`

可选但推荐：

- `contacts_plan_logits_raw`
- `contacts_plan_logits_phase`
- `contacts_plan_logits_time`
- `event_clock_lambda_logit`
- `event_clock_dynamic_prior`
- `event_clock_lr_diff`
- `event_clock_time_term`

这批量足够回答下面几个核心问题：

1. off-scale drift 是当前步 planner input 造成的，还是 carry state 造成的？
2. current-step residual 主要来自 `lambda_corr`，还是来自 `delta_z` / corrector writeback content？
3. residual 是 meas-driven 还是 hidden-state-driven？

### 2.1.1 为什么这些观测量有用：按因果链拆解

这些字段之所以值得留，不是因为“它们都在 planner 里”，而是因为它们分别卡在不同的因果位置上。

推荐用下面这条链理解：

```text
cond_t + plan_z_prev
    -> contact_plan_cell
    -> plan_z_raw
    -> (gate/corrector; inputs = meas, delta_meas, err_raw, lr_diff)
    -> plan_z_read_t
    -> logits_base
    -> + phase/time residual
    -> logits_final
```

如果某个字段不能帮你把这条链切开，它对主线归因价值就有限。

### 2.1.2 核心观测量与归因职责

| 观测量 | 所在位置 | 最主要回答的问题 | 为什么有用 |
|---|---|---|---|
| `plan_z_prev` | carry 输入 | 问题是不是上一步带进来的 | 这是区分 **carry-path** 和 **current-step** 的第一证据；没有它，很难解释 donor `plan_z_prev` 为什么有效 |
| `event_clock_plan_z_raw` | planner 输出 / corrector 前 | planner 本身是否已经 off-scale | 如果它已经漂了，问题更靠前；如果它稳定，问题多半在 corrector 或 readout |
| `event_clock_contacts_meas` | corrector 输入 | meas 路径是否在注入错误 | 很多 residual 不是 hidden 自己坏了，而是 meas path 把偏差喂给了 corrector |
| `event_clock_delta_meas` | corrector 输入 | 问题是不是出在事件边沿 / 变化量 | touchdown/liftoff 常常是边沿信号错，而不是绝对值错 |
| `event_clock_err_raw` | corrector 输入 | innovation 本身是否 off-scale | 这是最重要的解释变量之一；gate 和 corrector 基本都绕着它工作 |
| `event_clock_lambda_corr` | gate 输出 | correction 强度是不是异常 | 它回答“门开多大”，但不能独立回答“写回内容是什么” |
| `event_clock_delta_z` | corrector 输出 | corrector 实际想写回什么 | 比 `lambda_corr` 更接近问题内容本身；很多 residual 的主因在这里 |
| `event_clock_plan_z_t` | read path hidden | correction 后 hidden 是否已经被拉偏 | 它把“内部校正”与“最终 logits”连接起来 |
| `contacts_plan_logits_base` | planner readout | planner lane 的对外行为是否稳定 | 这是最好的 planner 行为代理量，比 hidden 更贴近结果，但还没混入 phase/time 分支 |
| `contacts_plan_logits` | 最终读出 | 最终行为有没有失真 | 用于最终确认，但不适合单独做根因定位 |

### 2.1.3 一个特别值得留的派生量

除了原始字段，推荐在离线分析时始终额外看：

- `applied_correction = event_clock_lambda_corr * event_clock_delta_z`

原因很简单：

- 单看 `lambda_corr`，你只知道 gate 开多大；
- 单看 `delta_z`，你只知道 corrector 想写什么；
- **只有两者乘起来，才是当前步真正施加到 hidden 上的校正量**。

如果主线只允许保留一个派生统计，我最推荐这个。

### 2.1.4 最小集合 vs 推荐集合

如果主线只能保留最小集合，建议至少保留下面 6 个：

- `plan_z_prev`
- `event_clock_plan_z_raw`
- `event_clock_err_raw`
- `event_clock_lambda_corr`
- `event_clock_delta_z`
- `contacts_plan_logits_base`

这 6 个已经足够覆盖三类核心归因：

1. 是不是 carry 带来的；
2. 是 gate 强度问题，还是 corrector content 问题；
3. planner 对外行为从哪一步开始失真。

如果允许再加 4 个，优先补这组：

- `event_clock_contacts_meas`
- `event_clock_delta_meas`
- `event_clock_plan_z_t`
- `contacts_plan_logits`

加上这 4 个后，就可以把 meas path、corrector path、最终 readout 三段更清楚地切开。

### 2.1.5 哪些量不要单独拿来“责怪”

下面这些量不是没用，而是**单独看很容易误判**：

- 只看最终 pose / final drift
  - 太下游，混合了 planner、corrector、phase/time 和 rollout carry 的共同结果。
- 只看 `event_clock_lambda_corr`
  - 容易把问题错怪到 gate；但大 gate 配小 `delta_z` 可能几乎没影响。
- 只看 `contacts_plan`
  - 经过 sigmoid 后容易饱和，隐藏了 logits 漂移。
- 只看 `1.0x` 与 `1.2x` 的最终差值
  - 能看出“坏了”，但看不出“为什么坏”。

### 2.1.6 这些观测量如何支持责任归因

实际归因时，建议按下面顺序判断：

1. 先看 `event_clock_plan_z_raw`
   - 若它已经 off-scale，优先怀疑 `carry path`、planner 输入或 speed 注入。
2. 再看 `event_clock_delta_z`
   - 若 `plan_z_raw` 还稳、但 `delta_z` 漂了，优先怀疑 corrector content。
3. 再看 `event_clock_lambda_corr`
   - 若 `delta_z` 稳、但 `lambda_corr` 漂了，才优先怀疑 gate 标量。
4. 再比 `contacts_plan_logits_base` 与 `contacts_plan_logits`
   - 若 base 稳、final 不稳，则更多是 phase/time residual 在搅局。

对应 donor probe 的解读也应该统一：

- donor `plan_z_prev` 有效
  - 说明主因在 carry path。
- donor meas-derived inputs 有效
  - 说明主因在 meas-driven path。
- donor `delta_z` 比 donor `lambda_corr` 更有效
  - 说明主因在 corrector content，而不是 gate 强度。

### 2.2 必须保留的干预能力

主线只需要保留极少数 override 能力：

- `plan_z_prev` donor override
- `lambda_corr` donor/zero override
- `delta_z` donor override
- `contacts_meas` donor override
- `delta_meas` donor override
- `lr_diff` donor override
- planner 输入侧的 `speed-zero` probe

这些 override 只需要存在于 **validate / debug lane**，不需要进入训练主路径。

### 2.3 必须保留的导出形态

推荐保留单一 JSON export contract：

```json
{
  "clip": "Walk_F",
  "scales": ["0.8", "0.9", "1.0", "1.1", "1.2"],
  "series": {
    "1.0": {
      "contacts_plan_logits_base": "...",
      "event_clock_plan_z_raw": "...",
      "event_clock_plan_z_t": "...",
      "event_clock_lambda_corr": "...",
      "event_clock_delta_z": "..."
    }
  }
}
```

关键不是字段名长得多漂亮，而是：

- **同一 probe** 在不同 scale 下字段一致；
- `1.0x` donor 可以直接拿来给其它 scale 做 counterfactual；
- validate lane 和后处理脚本共享同一 export 结构。

---

## 3. 明确不要带回主线的东西

以下内容当前更像 experiment scaffolding，不建议进 `main`：

- `touchdown_guard`
- `fixed_ref`
- `joint_consistency`
- `lambda_train_planner_stack`
- `lambda_train_planner_scope`
- `lambda_grad_probe_*`
- 针对单个 case 的 step snapshot / step probe 训练逻辑
- 依赖 `debug_output/_tmp_*`、`models/__tmp_*` 的临时 runner
- 带强实验口径默认值的脚本入口

一句话概括：

**主线保留“看得见、替得动、导得出”的 debug 能力；不要保留“为了这次实验方便而加的训练配件”。**

---

## 4. 建议的最小主线实现边界

### 4.1 `train/models.py`

主线只需要两类变化：

- 在 planner / event-clock forward 中，把关键中间量放进返回字典；
- 保留 read/write split 的可观测语义：
  - `plan_z_raw`
  - `plan_z_read_t`
  - `plan_z_next_t`

这里最重要的不是多一个功能，而是把下面这段语义固定下来：

```text
plan_z_raw --(read path + correction)--> plan_z_read_t --(logits)
plan_z_raw --(writeback rule)---------> plan_z_next_t --(carry to next step)
```

如果主线里看不到 read/write split，后续很多 donor 诊断都会失真。

### 4.2 `train/training_MPL.py`

主线只需要：

- rollout buffer 能累计上述关键字段；
- debug lane 能做 per-step override；
- 最终能把这些序列写进 `preds` / export。

不需要把 posttrain 的新 loss 逻辑搬进去。

### 4.3 `train/validate/*`

主线只需要一个稳定入口：

- baseline export
- speed-zero probe
- donor `plan_z_prev`
- donor `delta_z`
- donor meas-derived input

换句话说，主线只要有一个 “white-box export + override” 骨架就够了，没必要把所有当前分支里的分析脚本都带回去。

---

## 5. 推荐的 probe 套餐

如果主线以后只保留最小一套 probe，建议固定为下面 5 个：

### Probe A: baseline

目的：

- 导出所有 scale 的原始 planner/event-clock 序列；
- 建立后续 donor 对照基线。

### Probe B: planner speed-zero

目的：

- 只把 planner 输入中的 speed 通道置零；
- 判断 speed 信息是否显式进入 `contact_plan_cell` 当前步路径。

### Probe C: donor `plan_z_prev`

目的：

- 把每一步 incoming `plan_z_prev` 换成 `1.0x` donor；
- 判断 off-scale drift 主要是不是 carry state 放大的。

### Probe D: donor `delta_z`

目的：

- 固定当前步 corrector writeback content；
- 区分 residual 是 `lambda_corr` 问题，还是 `delta_z` / corrector content 问题。

### Probe E: donor meas-derived inputs

目的：

- 同步替换 `contacts_meas / delta_meas / err_raw`；
- 判断 residual 是否本质上是 meas-driven。

这 5 个 probe 足够覆盖当前最有价值的诊断问题，不需要额外上更多花样。

---

## 6. 推荐的主线验收口径

最小主线版本不要求“自动给出结论”，但必须能稳定回答以下三个判断题：

### Q1. drift 主要在当前步还是 carry 路径？

判据：

- donor `plan_z_prev` 后，若 raw drift 大幅下降，则 carry path 是主因。

### Q2. current-step residual 主要在 `lambda_corr` 还是 `delta_z`？

判据：

- donor `lambda_corr` 变化小；
- donor `delta_z` 变化大；
- 则主因是 `delta_z` / corrector content，而不是 gate 标量本身。

### Q3. residual 是否 meas-driven？

判据：

- donor meas-derived inputs 后，corrector mismatch 明显收敛或归零；
- 则 residual 主要来自 meas path，而不是抽象 hidden dynamics。

只要主线能稳定复现这三类判断，这批 debug hooks 就已经值回票价。

---

## 7. 与当前分支代码的关系

如果后续需要回收实现，可把当前分支当作参考来源，但不要原样搬运。

建议优先参考：

- `train/models.py`
  - planner / event-clock 的返回字段
  - read/write split 语义
- `train/training_MPL.py`
  - rollout buffer
  - per-step override
  - export 拼装
- `train/validate/run_gait_speed_scaling_whitebox.py`
  - donor probe 的组织方式

不建议直接照搬：

- `train/posttrain.py` 里的新增训练损失
- 针对单次问题排查堆出来的 CLI 开关
- 临时 `tools/*probe*` / `tools/*diagnose*` runner

---

## 8. 主线落地顺序建议

如果后续真要把这套能力放回 `main`，建议按下面顺序：

1. **先提交本文档**
2. 再提交“只增观测、不改训练行为”的最小 model + rollout hook
3. 最后补一个单一 validate export 入口

不要反过来先合并训练改动。  
否则 `main` 很容易得到一套难以维护、又夹带实验语义的半成品。

---

## 9. 当前建议

对本次合并请求，建议就是：

- **只提交这份文档到主分支**
- 暂不提交 `train/posttrain.py` 的实验性训练逻辑
- 暂不提交大批临时 debug 工具和实验日志

这样做的好处是：

- `main` 先拿到稳定结论和明确边界；
- 后续真正补代码时可以按文档做“最小实现”；
- 避免把当前分支的一次性实验结构永久沉淀进主线。
