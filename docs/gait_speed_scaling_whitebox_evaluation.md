# 步态速度扩展白盒评测设计文档

## 文档信息
- **创建时间**: 2026-03-11
- **版本**: v1.0
- **状态**: 设计方案
- **适用范围**: Walk gait 的 `0.8x / 0.9x / 1.1x / 1.2x` speed scaling，后续可扩展到变速段与走跑过渡

---

## 零、2026-05-05 现状对齐（按 runbook 0504 后缀 71 实验 final 产物）

本节只做“当前评估锚点”对齐，不改正文的 evaluator 设计。

对齐依据：

- runbook：`docs/basetrain_to_posttrain_top7_fresh_chain_runbook.md` §15
- 默认 final lambda（0504 71 实验）：  
  `debug_output/_tmp_71_lr1e4_lowlr_downstream_20260504/evals/lambda/group_summary.json`
- 首版 downstream 对照：  
  `debug_output/_tmp_70r_dense_to_lambda_20260504_r2/evals/lambda/group_summary.json`
- 同代码线 healthy final anchor：  
  `debug_output/_tmp_tail_top7_fresh_chain_20260418_074813/lambda_clean/eval_model_source_group_summary.json`

三条线关键指标（mask: `cycle_gte=1`, `drop_wrap=true`, `kept_steps=344`, `total_steps=434`）：

| lane | all_ex_root mean / p95 | leg mean / p95 | nonleg mean / p95 |
|---|---:|---:|---:|
| healthy final (2026-04-18) | `0.131884 / 0.439573` | `0.173059 / 0.474623` | `0.122981 / 0.434047` |
| dense downstream r2 (`71 lr=3e-4`) | `0.096919 / 0.311584` | `0.160309 / 0.392986` | `0.083213 / 0.287950` |
| **0504 71 实验 final** (`71 lr=1e-4`, lowlr) | `0.093424 / 0.309132` | `0.158343 / 0.417908` | `0.079388 / 0.274671` |

当前结论（仅基于 group summary）：

- 0504 final 相对 dense r2：`all_ex_root` 和 `nonleg` 继续改善。
- `leg` 呈现 mean 更好、p95 更差的 trade-off（`0.392986 -> 0.417908`）。
- 在 `debug_output/_tmp_71_lr1e4_lowlr_downstream_20260504/run_result.json` 中，`72` 与 `lambda` 指标完全一致；当前 lane 下 `lambda` 仍是 downstream no-op。

边界说明：

- 本节评估只覆盖 pose group 指标（`all_ex_root/leg/nonleg/arm/else`）。
- `E_speed / E_skate_weighted / E_phase_mono / E_cycle_speed_consistency` 等白盒速度扩展指标，本节未复跑，不应从上述数字外推。

---

## 一、问题背景

当前速度扩展任务不是传统的 supervised benchmark。

现阶段没有真实的“变速后 GT 动画”可以逐帧对比，因此不能继续沿用：

```text
pred vs GT
```

这一套评估范式。

当前更合理的做法是建立一个**白盒评测工具（white-box evaluator）**，直接评估：

1. 目标速度是否真的被实现
2. 相位 / 周期时钟是否稳定
3. 接触事件和落脚时序是否合理
4. 输出运动是否满足基本物理约束
5. 相对 `1.0x` 基线是否平滑退化，而不是结构崩坏

这套评测工具服务于当前路线：

```text
Step 3a: 先做 walk 内 speed scaling
Step 3b: 再看 contact anchor 是否必要
Step 4 : 最后再扩到 gait mixing / walk-run transition
```

---

## 二、评测目标

### 2.1 当前阶段目标

当前阶段只评估：

- 单一 walk gait
- 不考虑 walk-run transition
- 只考虑游戏常见轻中度速度缩放：`0.8x / 0.9x / 1.1x / 1.2x`

### 2.2 评测工具需要回答的问题

白盒评测工具必须回答以下核心问题：

1. **Speed tracking 是否成立**
   - 模型输出的 root speed 是否接近目标倍率

2. **Timing 是否稳定**
   - phase clock 是否单调、平滑、无异常跳变

3. **Contact 是否合理**
   - touchdown/liftoff 是否保持稳定交替
   - contact 段内是否出现明显 foot skating

4. **Cycle 语义是否仍成立**
   - speed scaling 后重新归一化到 87-frame cycle，是否仍接近基线 gait template

5. **误差是否可接受**
   - 相比 `1.0x`，`leg / non-leg` 是否平滑退化

---

## 三、设计原则

### 3.1 先数值、后视觉

当前阶段不依赖 UE 里的主观视觉评估，而是先建立纯数值准则。

原因：

- 视觉判断成本高，且难以稳定复现
- 当前还要继续做 forward -> turning 等后续工作
- 现阶段更需要一个可脚本化、可批量跑、可回归对比的指标体系

### 3.1.1 当前阶段暂不把 root trajectory 作为主评测对象

当前阶段默认采用 **pose/timing-first** 策略：

- 输入侧速度条件允许按目标倍率直接缩放
- rollout 时下一步继续喂缩放后的速度条件
- root trajectory 暂不作为当前阶段的 hard gate

原因：

- 当前评估主目标是验证 `speed scaling` 对 `pose + phase + contact` 子系统是否稳定
- 现有观察表明 root/trajectory 误差相对较小，不是当前主瓶颈
- 若过早把 trajectory 一并纳入主评测，会增加问题定位难度，掩盖 leg timing 的真实退化来源

因此当前阶段的结论边界应明确写为：

```text
当前数值评测结论主要针对 pose/timing subsystem，
不等价于完整 locomotion system（含 root trajectory）已完全验证。
```

### 3.2 先相对基线，再谈绝对正确

由于没有真实变速 GT，当前不追求“绝对误差”，而采用：

```text
scaled result relative to 1.0x baseline
```

即：

- `1.0x` 作为 reference
- 其他倍率和 `1.0x` 比较退化趋势
- 重点看是否**平滑退化**而非**突然失稳**

### 3.3 白盒优先于黑盒

当前任务属于“受控 gait speed scaling”，不是通用生成任务。

因此评测应优先使用系统内部可解释信号：

- target speed scale
- root speed
- phase increment
- contacts / ttc_td / ttc_td_events
- foot world velocity
- leg / non-leg error

而不是只看最终动画是否“像”。

---

## 四、评测输入与输出

### 4.1 输入

对白盒评测工具，建议每次评测输入以下信息：

- 基准 clip（如 `Walk_F`）
- 推理倍率 `s ∈ {0.8, 0.9, 1.0, 1.1, 1.2}`
- 推理输出序列
- 可选中间状态：
  - phase / cycle clock
  - resampling mapping
  - contacts plan / contacts meas
  - touchdown events

说明：

- 第一版 evaluator 可先不把 root trajectory fidelity 作为必算项
- 但仍建议保留 root speed、root planar displacement 等基础统计，供后续 trajectory 联调时复用

### 4.2 输出

建议统一输出一个结构化结果文件（如 JSON）：

```json
{
  "clip": "Walk_F",
  "scale": 1.1,
  "metrics": {
    "speed_tracking_rel": 0.03,
    "foot_skate_weighted": 0.014,
    "cycle_speed_consistency": 0.05,
    "freq_monotonic_ok": true,
    "stride_monotonic_ok": true,
    "phase_mono_violation": 0.0,
    "phase_jitter": 0.012,
    "cycle_consistency_leg": 0.08,
    "cycle_consistency_nonleg": 0.02,
    "leg_ratio_vs_ref": 1.12,
    "nonleg_ratio_vs_ref": 1.03
  },
  "status": "pass"
}
```

这样后续可以：

- 按 scale 批量比较
- 做回归测试
- 做 D-only vs D+C 的对比

---

## 五、核心指标设计

以下指标构成当前版本的最小白盒评测集。

### 5.1 指标 A：速度跟踪误差

这是主指标，用来确认“要求的倍率是否真的被实现”。

定义：

```text
v_ref  = mean planar root speed at 1.0x
v_pred = mean planar root speed at scale s
v_tgt  = s * v_ref

E_speed = |v_pred - v_tgt| / max(v_tgt, eps)
```

解释：

- `E_speed` 越小越好
- 若该值过大，说明 speed scaling 本身没有真正成立

备注：

- 当前阶段这里的 `speed tracking` 主要指 **条件速度与输出 pose/timing 响应的一致性**
- 不强制等价为“完整 root trajectory 已达到最终游戏运行质量”

### 5.2 指标 B：Foot skating 误差

这是最关键的物理一致性指标之一。

对于每只脚，在接触置信度高的帧上统计足底世界平面速度：

```text
E_skate = mean( ||v_foot_xy|| | contact_score > thr )
```

更推荐使用加权形式：

```text
E_skate_weighted = sum( contact_score * ||v_foot_xy|| ) / sum(contact_score)
```

解释：

- 接触时脚应近似静止
- 该指标升高通常意味着相位或接触边界出了问题

### 5.3 指标 C：Cycle decomposition consistency

这一项不再预设 `T_tgt = T_ref / s`。

原因是速度满足：

```text
v = stride_length × stride_frequency
```

而 `T_tgt = T_ref / s` 等价于假设 stride length 不变、全部速度变化都由频率承担。
在当前没有变速 GT、训练数据也没有显式告诉模型“正确分解比例”的前提下，这个假设过强，容易把：

- 真正的 timing 失稳
- 合理但不同的 stride/frequency 分解

混在一个指标里。

因此这里改为三层：先检查内部一致性，再检查单调性与平滑性，最后把分解比例作为诊断量记录。

#### C1：内部一致性校验（P0 主指标）

对每个 cycle，定义：

```text
T_i(s) = cycle period
       = same-side touchdown interval

L_i(s) = stride length per cycle
       = || root_xy(td_{i+1}) - root_xy(td_i) ||

v_cycle_i(s) = L_i(s) / max(T_i(s), eps)
```

同时，从整段输出中计算：

```text
v_pred(s) = mean planar root speed at scale s
```

定义一致性误差：

```text
E_cycle_speed_consistency(s)
  = mean_i( |v_cycle_i(s) - v_pred(s)| / max(v_pred(s), eps) )
```

解释：

- 这是纯运动学恒等式，不依赖任何 stride/frequency 分解假设
- 若 `L/T` 与 `v_pred` 明显不一致，说明 touchdown timing、root motion、reported speed 已经脱耦
- 这类问题比“周期没有贴近 `T_ref / s`”更严重，因为它说明系统内部已不自洽

建议实现：

- 不只记录 mean，还记录 cycle-wise `std / count / outlier_ratio`
- 避免均值正常但个别 cycle 跳变被掩盖
- P0 touchdown gate 建议拆成两类：
  - `td_count_unstable`：best 通道 count 与预期偏离超 `±1`（hard fail）
  - `td_channel_diverge`：左右脚通道计数分歧（默认 warn，但仅在 best 通道本身 clean 时成立）
- best 通道 clean 的建议条件：`count_error <= 1` 且 `interval_cv` 有限并在合理范围内（当前实现阈值 `<= 0.50`）
- `contact_source=auto` 在 gate 模式下只在 `meas / plan` 之间排序；`teacher` 只作为 reference 统计导出，不参与选源
- touchdown 统计建议同时导出 raw / smoothed 两路（如 3-frame majority vote）：
  - gate 用 raw
  - smoothed 只用于诊断，不参与 status

> **Known limitation（2026-05-05）**：`interval_cv <= 0.50` 当前是固定阈值，没有用 healthy anchor 在 1.0x 的 cycle-wise interval 分布做标定。
> 在 v4 评测下（`debug_output/_tmp_speed_eval_20260505/final_lambda_0504_71_whitebox_v4_*`），plan 全 5 个 scale `clean=True`、meas 在 `0.8/1.0/1.2x` 正确触发 `clean=False`，目前不漏不杀，因此不动。
> 后续若 healthy anchor (`_tmp_tail_top7_fresh_chain_20260418_074813`) 的 whitebox 复跑可用，再考虑改为 per-source 自适应阈值（按各源 1.0x 的 interval_cv 分位数定）。届时本节应同步更新阈值口径。

#### C2：单调性校验（P0 主指标）

这里采用最弱假设：速度升高时，频率不应降低、步幅不应缩短。

定义：

```text
f(s) = 1 / T(s)
L(s) = mean cycle stride length
```

检查：

- `f(s)` 关于 `s` 单调递增
- `L(s)` 关于 `s` 单调递增

实现建议：

- 由于当前只有 `0.8 / 0.9 / 1.0 / 1.1 / 1.2` 这 5 个点，不建议只看点估计均值
- 更稳的做法是基于 cycle-wise 样本同时记录 `mean ± 1 std`
- 只有当相邻 scale 的区间已明显反转时，才记为真正的 monotonicity violation

解释：

- 这不要求模型遵循“纯时间缩放”
- 只要求更快时不要出现更低步频或更短步幅
- 若 `1.1x` 的频率反而低于 `1.0x`，或 stride length 反而下降，就说明结构已经不稳定

#### C3：平滑性校验（P1 诊断指标）

除单调性外，还应检查 `f(s)` 与 `L(s)` 随倍率变化是否平滑。

建议记录：

```text
E_freq_smooth  = local second-difference or neighbor jump score on f(s)
E_stride_smooth = local second-difference or neighbor jump score on L(s)
```

解释：

- 目标不是拟合某条先验曲线
- 而是避免 `0.9 -> 1.0 -> 1.1` 之间出现跳变式重分配

#### C4：分解比例诊断量 `rho(s)`（P1，仅诊断）

定义频率贡献比例：

```text
f_ref = 1 / T_ref
v_ref = mean planar root speed at 1.0x

rho(s) = ((f(s) - f_ref) / max(f_ref, eps))
       / ((v_pred(s) - v_ref) / max(v_ref, eps))
```

解释：

- `rho ≈ 1`：接近“纯频率承担速度变化”
- `rho ≈ 0`：接近“纯步幅承担速度变化”
- `0 < rho < 1`：频率 / 步幅混合分解

数值稳定性说明：

- 当 `s ≈ 1.0` 时，`v_pred(s) - v_ref` 很小，逐点 `rho(s)` 容易被噪声放大
- 因此不建议把 `0.9x / 1.1x` 的逐点 `rho` 当成强结论
- 更稳的做法有两种：
  1. 仅在 `0.8x / 1.2x` 上记录逐点 `rho`
  2. 在全部 scale 点上拟合 `f(s) = a * v_pred(s) + b`，把斜率 `a` 作为全局 frequency-contribution estimate

这一项不作为 pass/fail 门槛，只作为诊断信号：

- 若 `rho ≈ 1` 且 `E_cycle_consistency` 变差，偏向 pathological timing scaling
- 若 `rho < 1` 且 `E_cycle_consistency` 变差，可能是 stride length 改变导致 gait shape 合理偏移

### 5.4 指标 D：Phase 连续性误差

如果系统中存在显式 phase clock / resampling clock，则必须测这一项。

建议至少记录两个数：

```text
E_phase_mono   = ratio(Δphase < -eps)
E_phase_jitter = std(Δphase - mean(Δphase))
```

解释：

- `E_phase_mono` 检查 phase 是否违反单调递增
- `E_phase_jitter` 检查 phase 增量是否出现明显毛刺

当 `D-only` 在变速段不稳定时，这两个指标通常会先坏。

### 5.5 指标 E：相对退化比值

由于你已有 `1.0x` 的正常基线，因此建议持续跟踪 `leg / non-leg` 相对涨幅。

定义：

```text
R_leg(s)    = leg_metric(s) / max(leg_metric(1.0), eps)
R_nonleg(s) = nonleg_metric(s) / max(nonleg_metric(1.0), eps)
```

解释：

- `R_leg` 主要反映腿部 timing 扩展是否稳定
- `R_nonleg` 主要监控系统是否整体失稳
- 正常情况下：`R_leg` 允许缓慢上升，`R_nonleg` 应保持稳定

### 5.6 指标 F：Cycle self-consistency

这是当前任务里最有价值的“无 GT 指标”。

逻辑：

- 把 `scale = s` 的输出重新归一化到 87-frame cycle
- 与 `1.0x` 输出在归一化相位空间里做对比

定义：

```text
E_cycle_consistency(s) = dist( normalize_to_87(pred_s), pred_1.0 )
```

这个 `dist` 可以分成：

- `E_cycle_leg`
- `E_cycle_nonleg`
- 可选 `E_cycle_root_dir`

解释：

- 如果 speed scaling 是合理的，那么同一 gait 在归一化周期空间里应仍接近基线模板
- 若该指标明显变差，说明 speed scaling 已开始改变 gait shape，而不仅仅是时间尺度

---

## 六、推荐验收标准

当前阶段先采用“分级通过”的策略，而不是一刀切绝对阈值。

### 6.1 对 `0.9x / 1.1x`

目标：应接近基线，可视为低风险区。

建议标准：

- `E_speed` 保持小幅误差
- `E_skate_weighted` 接近 `1.0x` 基线
- `E_cycle_speed_consistency` 保持小幅误差
- `f(s)` / `L(s)` 相对 `1.0x` 单调且无明显跳变
- `E_phase_mono ≈ 0`
- `R_leg` 仅小幅上升
- `R_nonleg` 基本稳定
- `E_cycle_consistency` 维持低值

结论口径：

```text
若 0.9x / 1.1x 已出现明显 phase 抖动或 foot skating，则 D-only 不足。
```

### 6.2 对 `0.8x / 1.2x`

目标：允许一定退化，但不能结构性失稳。

建议标准：

- 允许 `R_leg` 明显高于 `1.0x`
- 但 `R_nonleg` 不应同步大幅恶化
- `E_skate_weighted` 不应出现跳变式上升
- `E_cycle_speed_consistency` 不应明显恶化
- `f(s)` / `L(s)` 不应出现反单调或跳变
- `E_phase_mono` 仍应接近零

结论口径：

```text
0.8x / 1.2x 可以有腿部误差增大，但不能出现明显时钟失控、接触错乱或周期边界抖动。
```

---

## 七、异常模式与诊断含义

白盒评测工具不仅要给出分数，还要给出“异常意味着什么”。

### 7.1 若 `E_speed` 大

说明：

- 重采样倍率没有真正作用到最终 motion
- root motion / cond speed / physical timing 之间存在不一致

优先检查：

- speed scale 注入位置
- root speed 计算方式
- resampling mapping 是否正确

### 7.2 若 `E_skate_weighted` 大

说明：

- 接触边界和实际足底时序错位
- phase mapping 出问题，脚在 contact 段仍被推进

优先判断：

- 是否需要引入 `contact anchor`
- touchdown 事件是否被正确检测

### 7.3 若 `E_phase_mono / E_phase_jitter` 坏

说明：

- timing clock 自身不稳定
- D-only 对该速度区间或该变速强度不再成立

优先判断：

- 是否需要在 TD 事件处做硬重置
- 是否进入“单周期内速度变化过大”的失败区间

### 7.4 若 `E_cycle_speed_consistency` 坏

说明：

- `L / T` 与 `v_pred` 已经不能彼此解释
- touchdown timing、root displacement、reported speed 三者出现脱耦

优先检查：

- touchdown 事件是否稳定
- root planar position / velocity 的单位与对齐方式
- cycle 切分是否跨越异常事件边界

结论：

```text
这不是“分解比例不同”这么简单，而是系统内部运动学不自洽。
```

### 7.5 若 `f(s)` 或 `L(s)` 反单调 / 不平滑

说明：

- 模型在倍率切换时没有形成稳定的速度分解策略
- 即便平均速度达标，内部 cycle 组织也可能已经失真

优先判断：

- 是否某个 scale 的 touchdown 检测不稳定
- 是否 timing estimator 在该区间出现局部翻转
- 是否存在特定 scale 的 contact / resampling 边界问题

### 7.6 若 `R_leg` 坏但 `R_nonleg` 稳定

说明：

- 这是典型的腿部 timing 问题
- 通常还没到系统整体失稳

结论：

```text
优先补 contact / phase 校正，不必先怀疑 non-leg trunk。
```

### 7.7 若 `E_cycle_consistency` 明显变差

说明：

- gait shape 本身开始发生明显变化
- 已经不只是时间缩放问题，而是 motion manifold 在偏移

联动解释：

- 若同时 `rho ≈ 1`，更像纯 timing scaling 导致的 pathological 变形
- 若同时 `rho < 1`，则可能是步幅增加带来的可解释 gait shape 改变

结论：

```text
该速度范围可能需要更强建模，或后续进入 gait mixing / transition 问题范畴。
```

---

## 八、评测流程建议

建议把第一版评测流程固定成如下顺序：

### Step 1：建立 `1.0x` 基线

记录：

- `leg_metric(1.0)`
- `nonleg_metric(1.0)`
- `v_ref`
- `T_ref`
- `f_ref = 1 / T_ref`
- `L_ref`
- `E_skate_weighted(1.0)`

### Step 2：跑 4 个倍率点

对以下 scale 单独评测：

- `0.8x`
- `0.9x`
- `1.1x`
- `1.2x`

### Step 3：输出统一报告

建议每次生成一张汇总表：

| scale | E_speed | E_cycle_speed_consistency | freq_mono | stride_mono | E_phase_mono | R_leg | R_nonleg | E_cycle_leg | status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0.8 | ... | ... | ok/warn | ok/warn | ... | ... | ... | ... | pass/warn/fail |
| 0.9 | ... | ... | ok/warn | ok/warn | ... | ... | ... | ... | pass/warn/fail |
| 1.1 | ... | ... | ok/warn | ok/warn | ... | ... | ... | ... | pass/warn/fail |
| 1.2 | ... | ... | ok/warn | ok/warn | ... | ... | ... | ... | pass/warn/fail |

### Step 4：做 D-only vs D+C 对比

当 D-only 出现问题时，再跑：

- D-only
- D + contact re-anchor

比较哪一项指标被修复：

- `E_skate`
- `E_cycle_speed_consistency`
- `E_phase_jitter`

这一步用于判断 contact anchor 是否值得引入。

---

## 九、与后续工作关系

这套白盒评测工具不是一次性工具，而是后续扩展的基础设施。

### 9.1 对 Step 3a 的意义

用于判断：

- `D-only` 是否足够覆盖 `+-10% / +-20%`
- 何时必须补 `contact anchor`
- 在暂不引入 trajectory 主评测的前提下，先把 pose/timing 子系统单独验证干净

### 9.2 对 Step 3b 的意义

用于分析：

- contact plan 在变速边界是否稳定
- 周期边界重置是否真的生效

### 9.3 对 Step 4/5 的意义

未来走跑过渡时，该工具还能帮助区分：

- 问题出在 timing estimator
- 还是出在 gait mixing / 双模板切换

即：

```text
先用白盒指标分离 timing 问题与 gait topology 问题。
```

---

## 十、第一版 Evaluator 实现顺序

当前建议不要一开始就把所有指标一次性实现完，而是按“先复用现有信号、先建立回归闭环”的顺序推进。

### 10.1 P0：先打通最小闭环

第一批只实现最关键、最容易落地的 4 项：

1. `E_speed`
2. `R_leg / R_nonleg`
3. `E_cycle_speed_consistency` + `f(s) / L(s)` 单调性
4. `E_cycle_consistency`

原因：

- 这 4 项已经能回答当前最核心问题：`D-only 在 +-10% / +-20% 下是否稳定`
- 所需张量大多已在现有 loader / freerun / validate 工具中存在
- 可以先建立 `1.0x / 0.9x / 1.1x / 0.8x / 1.2x` 的自动化回归表

### 10.2 P1：补 timing 白盒指标

第二批加入：

5. `E_phase_mono`
6. `E_phase_jitter`
7. `E_skate_weighted`
8. `f(s) / L(s)` 平滑性
9. `rho(s)`
10. 可选 `E_td_asymmetry`

原因：

- 这几项更依赖中间状态或 FK / foot world velocity 计算
- 适合作为 D-only 出现边界问题时的定向诊断器

### 10.3 P2：做 D-only vs D+C 对比

当 P0/P1 已稳定后，再增加配置对比：

- `D-only`
- `D + contact anchor`

重点比较：

- `E_cycle_speed_consistency`
- `f(s) / L(s)` 单调性与平滑性
- `E_phase_jitter`
- `E_skate_weighted`

这样可以直接回答：

```text
contact anchor 到底是在修 timing 边界，还是只是改变了输出分布。
```

---

## 十一、指标与现有代码的字段映射

这一节的目标是：尽量复用你当前代码里已经存在的张量、日志和导出接口，避免第一版 evaluator 重新发明一整套数据流。

### 11.1 `E_speed` 对应字段

推荐来源：

- `RootVelocity` in state X / denorm 后的 `rootvel_x_slice`
- `train/eval_utils.py:487`
- `train/eval_utils.py:515`
- `train/layout.py:425`

说明：

- 当前 freerun 调试记录已经有 `root_vel_mae`
- `DataNormalizer.denorm_x()` 已处理 `root_vel` 的 inverse-tanh，物理单位较可信
- 第一版 evaluator 可以直接从 denorm 后的 root velocity 计算 `v_ref` 和 `v_pred`

建议实现：

```text
ref: scale=1.0 的 mean ||root_vel_xy||
pred: scale=s 的 mean ||root_vel_xy||
```

### 11.2 `R_leg / R_nonleg` 对应字段

推荐来源：

- `dir_leg_base`
- `dir_nonleg_base`
- `leg_over_nonleg`

对应代码：

- `train/models.py:5627`
- `train/models.py:5726`
- `train/training_MPL.py:3839`
- `train/posttrain.py:2968`

说明：

- 你当前系统里已经显式维护 leg / non-leg 分组指标
- 第一版 evaluator 不需要重新定义一套腿/非腿误差
- 只需统一在 `scale=1.0` 上取 reference，再计算 ratio

### 11.3 `E_cycle_speed_consistency / f(s) / L(s)` 对应字段

推荐来源：

- `contacts`
- `ttc_td`
- `ttc_td_events`
- root planar position / denorm 后 root translation

对应代码：

- `train/dataset.py:524`
- `train/dataset.py:1007`
- `train/dataset.py:1013`

说明：

- 当前 dataset 已能在窗口级别重算 touchdown TTC 与事件掩码
- 第一版 evaluator 可先直接基于 `ttc_td_events` 统计 same-side touchdown interval，得到 `T_i(s)`
- `L_i(s)` 定义为同侧脚连续两次 touchdown 时刻 root planar position 差的范数
- 这一项不必一开始就依赖 model 内部的 `contacts_plan`
- `v_pred(s)` 仍建议复用 denorm 后的 root planar speed
- 若 best 通道 touchdown count 与预期 cycle 数偏离超过 `±1`，建议直接打上 `td_count_unstable`（hard fail）
- 若只出现左右脚计数分歧，建议打 `td_channel_diverge`；只有在 best 通道 clean 时可降为 warn
- 若 `contact_source=auto`（gate mode），建议只在 `contacts_meas / contacts_plan` 里做稳定性排序；
  `teacher contacts` 仅导出 `td_stats_teacher` 供 diff，不参与 gate 选源

建议优先顺序：

1. gate 模式先在 `meas/plan` 里选 touchdown 稳定性更好的源
2. `teacher` 单独保留 reference 统计，不进入 gate status
3. 配合 root planar position 重建 cycle-wise `T_i(s)` 与 `L_i(s)`
4. 先做 touchdown sanity check，再计算 C1 / C2
5. 后续再补 `contacts_plan` / `contacts_meas` 的对比诊断

### 11.4 `E_phase_mono / E_phase_jitter` 对应字段

推荐来源：

- `phase_z_next`
- `phase_event_age_next`

对应代码：

- `train/eval_utils.py:427`
- `train/eval_utils.py:430`
- `train/models.py:2348`
- `train/models.py:2349`

说明：

- freerun 已在 rollout 中维护 `phase_z` 和 `phase_event_age`
- 但第一版默认未统一导出为时间序列
- 因此这一项建议放在 P1，再补充导出或调试记录

建议实现：

- 每步保存 `phase_z_next`
- 从 `sin/cos` 恢复 phase angle
- 使用 circular difference 计算 `Δphase`，避免 `atan2` 在 `±pi` 处 wraparound 造成假性 phase mono violation
- 再计算 `Δphase` 的单调性与 jitter

### 11.5 `E_cycle_consistency` 对应字段

推荐来源：

- `predY`
- `gtY`
- 可选 `per_step_direct_geolocal_deg`

对应代码：

- `train/eval_utils.py:531`
- `train/eval_utils.py:539`
- `train/validate/run_freerun_cycles.py:8772`

说明：

- 当前已经支持导出逐步、逐关节的 `DirectGeoLocalDeg`
- 这非常适合做 cycle-normalized 后的 leg / non-leg 一致性比较
- 第一版可先用整体或分组均值，后续再细化到 per-joint curve

### 11.6 `E_skate_weighted` 对应字段

推荐来源：

- `contacts`
- `bone_rot6d`
- skeleton offsets / parents

对应代码：

- `train/dataset.py:598`
- `train/dataset.py:640`
- `raw_data/Walk_F.json:43`

说明：

- 当前 repo 已有骨骼层级、offset、rot6d 等构造信息
- 但还没有现成的“足底世界速度”统一评测接口
- 因此第一版不建议先上这一项；放到 P1 更合适

---

## 十二、第一版 Evaluator 建议输入输出

### 12.1 建议输入

第一版脚本建议只接受最少参数：

- `--clip Walk_F`
- `--scales 0.8,0.9,1.0,1.1,1.2`
- `--mode d_only`
- `--ckpt <path>`
- `--config <path>`
- `--out <json>`

后续再扩展：

- `--mode d_plus_anchor`
- `--export_phase_series`
- `--export_contact_series`

### 12.2 建议输出结构

建议输出三层：

1. `summary`
2. `per_scale`
3. `optional_series`

示例：

```json
{
  "summary": {
    "clip": "Walk_F",
    "mode": "d_only",
    "ref_scale": 1.0,
    "touchdown_source_policy": "stable_touchdown_v1"
  },
  "per_scale": {
    "0.8": {
      "E_speed": 0.06,
      "R_leg": 1.18,
      "R_nonleg": 1.05,
      "E_cycle_speed_consistency": 0.04,
      "freq_hz": 1.72,
      "stride_length": 0.81,
      "freq_monotonic_ok": true,
      "stride_monotonic_ok": true,
      "touchdown_source": "teacher",
      "touchdown_count": 5,
      "td_unstable": false,
      "E_cycle_leg": 0.11,
      "status": "warn"
    }
  },
  "optional_series": {}
}
```

---

## 十三、第一版通过/告警/失败规则

第一版不建议定死非常激进的阈值，更适合先用规则型判定。

### 13.1 `pass`

满足：

- `E_speed` 小
- `R_leg` 仅平滑上升
- `R_nonleg` 基本稳定
- `E_cycle_speed_consistency` 小
- `f(s)` / `L(s)` 保持单调且无异常跳变
- `E_cycle_consistency` 无异常跳变

### 13.2 `warn`

满足：

- `R_leg` 上升明显，但 `R_nonleg` 仍稳定
- 或某个 scale 出现 `td_channel_diverge=true`，且 best 通道仍满足 clean 条件
- 或 `E_cycle_speed_consistency` 开始偏大
- 或 `freq_std_hz / freq_hz > 0.10`
- 或 `stride_std_length / stride_length > 0.10`
- 或 `f(s)` / `L(s)` 出现轻微反单调或局部跳变
- 或 `E_cycle_consistency` 有可解释退化

解释：

```text
当前倍率已接近 D-only 的安全边界，但尚未完全失控。
```

plan `1.2x` mild CV fray 于 2026-05-05 收口为 known accepted boundary，不构成阻塞合并失败模式；读数与 reopen 条件见 evidence §2.2（`docs/changes/2026-03-25_contact_plan_event_clock_whitebox_mainline_evidence.md`）。

### 13.3 `fail`

满足任一：

- `E_speed` 明显失真
- `R_nonleg` 也同步恶化
- 或 `td_count_unstable=true`
- 或 `td_channel_diverge=true` 且 best 通道不 clean
- `E_cycle_speed_consistency` 明显失配
- 或 `freq_std_hz / freq_hz > 0.20`
- 或 `stride_std_length / stride_length > 0.20`
- `f(s)` / `L(s)` 与倍率关系反单调或跳变式失稳
- `E_cycle_consistency` 出现跳变式恶化

解释：

```text
说明当前倍率下已不是“平滑退化”，而是结构性失稳；应考虑 contact anchor 或更强建模。
```

---

## 十四、当前结论

当前阶段应优先建设一个**步态速度扩展白盒评测工具**，而不是先依赖 UE 视觉判断。

同时，当前阶段默认采用：

```text
先验证 pose/timing subsystem，
暂不把 root trajectory fidelity 作为主验收门槛。
```

这意味着：

- 现在的 speed scaling 评测可以先聚焦 leg/non-leg、phase、contact、cycle consistency
- trajectory 相关问题可在后续 forward -> turning 与完整 locomotion 联调阶段再纳入主评测

第一版工具至少应覆盖以下 6 个核心指标：

1. `E_speed`
2. `E_skate_weighted`
3. `E_cycle_speed_consistency`
4. `E_phase_mono / E_phase_jitter`
5. `R_leg / R_nonleg`
6. `E_cycle_consistency`

其中 `f(s)` / `L(s)` 的单调性建议作为 P0 同步输出，`rho(s)` 作为 P1 诊断量输出。

这套工具的目标不是给出“绝对真实度”，而是回答：

```text
在没有变速 GT 的条件下，当前 speed scaling 是否数值上稳定、物理上合理、并且相对 1.0x 平滑退化。
```

只要这套数值体系先建立起来，后续无论是：

- walk 内 speed scaling
- forward -> turning
- 还是未来的 walk-run transition

都可以在进入 UE 视觉验证前，先做稳定的数值回归。

---

## 十五、P0 落地快照（2026-03-15）

一次性实验快照已拆分到：

- `docs/changes/2026-03-15_gait_speed_scaling_whitebox_p0_snapshot.md`
- `docs/Problems/active/2026-03-15_72_lowlr_to_lambda.md`

本文保留 **evaluator 规范与验收口径**；具体实验读数、teacher/plan/meas 三路对比、当日结论请以上述快照/问题单为准。
