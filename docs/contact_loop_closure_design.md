# Contact Loop Closure（软接触闭环）设计与本次修复说明

## 1. 这次改进要解决什么问题？

项目现象（症状）：

- **Teacher forcing 下**单步误差很小（单帧预测 OK）。
- **Free-run（自回归）长序列**误差会逐步累积（drift），并在 `freerun_cycles` 这类“多轮循环”测试中明显放大（第二轮/第三轮更差）。
- 曾出现开启 `--so3_corr_apply` 后 **step=0 直接爆炸**（`GeoDeg` 从 ~0.x° 跳到 50°+）。

核心目标：

- 让模型在 free-run 时具备**闭环纠错（closed-loop correction）**能力：当状态逐渐偏离真实运动流形时，有一个“独立锚点”能持续提供可用的反馈信号，驱动纠偏而不是继续漂移。

## 2. 根因拆解（从 debug JSON 得到的结论）

### 2.1 step=0 爆炸的根因（已修）

开启 `--so3_corr_apply` 的爆炸并非 GRU 本身导致，而是 **SO(3) apply 的 rot6d 表示处理错误**：

- 将“delta 表示（residual）”误当作“absolute 6D rotation”写回，导致第一步就把旋转写成错误分布，从而 step=0 爆炸。
- 修复后：`so3_corr_apply` 不再导致首帧 50°+ 的异常，首帧会回到与 baseline 同量级（~0.x°）。

> 这类 bug 的典型特征是：`*_apply*.json` step=0 巨大，且与 gate/GRU 无关。

### 2.2 “闭环没闭合”的根因（当前主要问题）

闭环设计里有两个接触信号：

- `contacts_plan`：GRU（cond-only）预测的“计划/锚点接触”（独立于当前预测姿态）
- `contacts_meas`：由当前 pose_history / angvel 推断的“观测接触”（依赖当前姿态）
- `e_t = contacts_plan - contacts_meas`：闭环误差信号，用来驱动 `omega_hat`（SO(3) corrector）

在 `debug_output/freerun_cycles/_clgate2/Walk_F_freerun_cycles.json` 中可以看到：

- `ContactMeasMean ≈ 0.49` 长期接近常数（≈ sigmoid(0)）
- 同时 `ContactMeasGtAbsMean` 很大（与 GT soft contacts 偏差显著）

结论：

- `contact_meas_head` **没有学成一个可用的“观测”**，导致 `e_t` 失真；
- 于是你会感觉“GRU 好像没起作用、主要是 so3 在拉回”，本质原因是：闭环反馈（plan-meas）这条通路缺了“meas 是真的 meas”这一环。

## 3. 参考论文（docs/basketball.pdf）的关键思想：我们应该借用什么？

论文核心不是“相位拟合本身”，而是**用独立锚点消除自回归误差累积**，并通过**双路径**让模型“必须使用锚点”。

### 3.1 论文里最有价值的 3 个机制

1) **独立更新的相位锚点（phase anchor）**  
相位每帧重新定位，不从预测结果递推，因此不会像自回归状态那样累积漂移。

2) **双路径（direct vs incremental）误差正交**  
论文利用两条预测路径的失败模式不同：

- 直接预测：易高频抖动/跳变
- 增量预测：易低频漂移/长期偏离

最终输出通过插值融合，两种误差互相抵消一部分。

3) **“网络不完全信任自己”**  
对未来控制量也做插值，避免预测误差自我放大。

### 3.2 为什么原论文的“正弦相位拟合”不适合 ARPG

论文的相位提取（正弦拟合）隐含了强周期性假设：动作可以被稳定的频率/相位描述。  
ARPG 里大量动作是非周期/强事件驱动（受击、闪避、技能、combo），单纯正弦拟合会失效或变得脆弱。

## 4. 适合当前项目的改造：用“软接触”替代 Local Motion Phases

### 4.1 软接触信号的定位：phase 的“超集”

我们把每个接触通道的 soft contact `c(t)∈[0,1]` 视为：

- 周期动作：它天然呈周期波形（等价提供相位信息）
- 非周期动作：它仍能编码“进度/稳定性变化”（不是相位，但仍是有用的低维锚点）

关键在于：**这个锚点必须是独立的（不从预测姿态递推）**。

### 4.2 当前闭环结构（推荐保留的主线）

这条路线与论文“独立相位锚点 + 容错融合”同构，只是把相位换成软接触：

- `contacts_plan = GRU(cond-only)`：独立锚点（对应论文相位更新）
- `contacts_meas = MLP(pose_history, angvel)`：从当前预测姿态提取“观测”
- `e_t = plan - meas`：闭环误差（类似控制系统的 innovation）
- `omega_hat = f(h_final, e_t)`：SO(3) corrector 预测纠偏（小角度）
- `ΔR_used = Exp(omega_used) * ΔR_pred`：在 SO(3) 流形上应用纠偏

**注意：**仅仅把 `contacts_plan` 作为额外输入并不能保证模型使用它；闭环必须在结构上“强制参与输出”，这点与论文一致。

## 5. 本次落地的工程改动（以及为什么这么做）

### 5.1 修复 `--so3_corr_apply` 的首帧爆炸

修复点：SO(3) apply 中 rot6d residual/absolute 的转换逻辑错误（见 `BUG_REPORT_SO3_CORRECTOR.md` 记录）。  
效果：开启 apply 后不再出现 step=0 50°+ 的异常。

### 5.2 增强诊断：把“闭环是不是闭合”变成可观测量

在 freerun cycles 输出中增加：

- `ContactPlanMean`
- `ContactMeasMean`
- `ContactErrAbsMean`
- `ContactMeasGtAbsMean`（关键：meas 是否在学）

用于直接判断：

- meas 是否是“接近 0.5 的常数头”
- e_t 是否会随 drift 增大而增大（能否作为控制信号）

### 5.3 新增 posttrain 能力：可选 finetune `contact_meas_head`

文件：`train/posttrain.py`

新增两个能力：

- `--train_contact_meas true`：解冻 `contact_meas_head`，做监督训练
- `--contact_meas_weight <float>`：监督损失权重（MSE(meas, GT_contacts)）

设计原则：

- **不把 GT contacts 喂进 model forward**（保持 train/infer 一致），只用来计算监督 loss；
- 目标是把 `contacts_meas` 训练成一个“像传感器/观测”的估计器，使 `plan-meas` 成为有意义的闭环误差。

## 6. 推荐的训练/验证流程（针对当前项目）

### Step A：先把 `contacts_meas` 训成真的 meas（闭环先闭合）

只训练 meas head（不动 so3）：

```bash
python -m train.posttrain \
  --config config/posttrain_contactloop_corr.json \
  --run_name posttrain_meas_only \
  --train_so3_corrector false \
  --train_contact_meas true \
  --contact_meas_weight 1.0
```

然后跑 freerun cycles 验证（关注 contact 指标）：

- `ContactMeasGtAbsMean` 应显著下降
- `ContactErrAbsMean` 应更能反映 drift（第二轮漂移更大时误差更大）

### Step B：再联合微调 so3 corrector（用更可靠的 e_t 驱动纠偏）

```bash
python -m train.posttrain \
  --config config/posttrain_contactloop_corr.json \
  --run_name posttrain_meas_plus_corr \
  --train_so3_corrector true \
  --train_contact_meas true \
  --contact_meas_weight 0.5
```

## 7. 后续扩展思路（面向 ARPG 的通用性）

### 7.1 多锚点（强烈推荐）

仅脚接触对“上半身动、下半身平台段”的动作会出现多对一（phase 不可辨）。  
建议逐步扩展 anchor 通道：

- 手（握持/击打/触地）
- 武器（命中/碰撞/挥动阶段）
- 身体关键接触（盾牌、膝盖、背部触地等）

仍保持原则：**plan 是 cond-only 的独立锚点**；meas 是从预测姿态估计的“观测”。

### 7.2 双路径融合（借用论文最核心的容错结构）

论文的“Θ_{i+1}（direct） vs Θ_i+ΔΘ（incremental）”在本项目可以映射为：

- 路径1：直接预测下一帧 absolute（或更强的直接分支）
- 路径2：增量预测（当前项目已有 Δ 输出）
- 用 gate/λ 融合输出（并让 gate 依赖 anchor 信号）

这能让误差模式更“正交”，提高长序列稳定性。

### 7.3 频域/傅立叶特征：作为“抗多对一”的辅助，而不是替代

你提出的“对软接触做傅立叶建模”最主要解决的是：

- 单帧 contact 向量多对一（phase 不可辨）

建议定位为：

- **辅助特征**（给 plan GRU 或给 gate），而不是强制把动作解释成单一周期；
- 对变速跑/切换动作，可用“多频 + 幅度”或短窗 STFT/滤波器组，让频率随时间变动；
- 仍保持闭环结构：频域特征用于提高锚点辨识度，而不是替代闭环本身。

### 7.4 可靠性建模（让系统学会“什么时候信 contacts”）

对非周期动作，contact 的信息量会变化。可以增加一个 `reliability r_t∈[0,1]`：

- 由 plan/meas 的一致性、cond 的动作类型、速度等预测
- 用于调制：`e_t := r_t * (plan - meas)`

这与论文“网络不完全信任自己”的哲学一致。

---

## TL;DR（当前阶段的结论）

- 你的“软接触 + GRU（cond-only）作为独立锚点”这条主线是对的，且比正弦相位拟合更适合 ARPG。
- 目前闭环效果不稳的关键不在 GRU，而在 `contacts_meas` 没学好（输出接近常数），导致 `plan-meas` 不是有效反馈。
- 先把 meas head 训到像“观测”，闭环才会真正闭合；之后再谈 so3 gate、双路径融合与多锚点扩展。

