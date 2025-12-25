# Contact Loop Closure 训练迭代复盘（2025-12-24）

本文档记录本轮“Contact Loop Closure（软接触闭环）”训练迭代要解决的问题、实际偏差（drift 指标）、与 `docs/basketball.pdf`（Local Motion Phases）论文思想的对接方式、为何尚未解决问题，以及下一步改进路线。

> 对应验证输出（baseline）：`debug_output/freerun_cycles/posttrain_meas_plus_corr/Walk_F_freerun_cycles.json`
>
> 补充验证输出（本轮追加）：
> - time-PE（relative t）：`debug_output/freerun_cycles/timepe16_planhead/Walk_F_freerun_cycles.json`
> - time-PE（abs start / global t，会在 multi-cycle 下 OOD）：`debug_output/freerun_cycles/timepe16_planhead_absstart/Walk_F_freerun_cycles.json`
> - time-PE（abs start + phase time_index=t%cycle_len，multi-cycle 对齐 ✅）：`debug_output/freerun_cycles/timepe16_planhead_absstart_cycleidx/Walk_F_freerun_cycles.json`
> - meas/err 时序曲线（global vs cycle、apply vs no-apply）：`debug_output/freerun_cycles/compare_contacts_meas_err_timeseries.png`
> - knob test（强行开增益）：
>   - gate=0.2：`debug_output/freerun_cycles/knobtest_posttrain_meas_plus_corr_gate02_max10/Walk_F_freerun_cycles.json`
>   - gate=0.3：`debug_output/freerun_cycles/knobtest_posttrain_meas_plus_corr_gate03_max10/Walk_F_freerun_cycles.json`

---

## 1. 本轮要解决的问题是什么？

目标症状（现象）：

- **Teacher forcing 下单步误差小**（单帧/短 horizon OK）
- **Free-run 自回归长序列 drift 累积**：随着时间推进误差变大；在 `freerun_cycles`（多轮循环）中，第二轮/第三轮明显更差

问题本质（与自回归数学困境一致）：

- 推理时输入来自模型上一帧输出，误差被递推并在非线性动力学中放大；
- 训练时（teacher forcing）模型很少见到“带误差的输入分布”，导致推理分布偏移；
- 高维输出（多关节 × 6D）存在大量漂移自由度，没有外部锚点就容易“迷航”。

本轮希望通过 **闭环纠错（closed-loop correction）** 引入一个“独立锚点”，让模型在 free-run 时能持续获得纠错信号，而不是只能被动漂移。

---

## 2. 参考论文（`docs/basketball.pdf`）的核心思想是什么？

论文（Local Motion Phases）的关键不在“正弦拟合”本身，而在 **用独立于预测姿态的外部时钟/锚点（phase anchor）** 破除自回归 drift：

- 相位/接触信号由外部测量计算（不依赖 `y_{t-1}`），因此**不会随着预测漂移**；
- 模型同时拥有两条路径（direct vs incremental），误差模式不同，通过插值融合互相抵消：
  - direct：不累积但可能高频抖动
  - incremental：连续但会低频漂移

对我们的映射：不强行做周期正弦相位，而是把 **soft contact（软接触）** 当作“相位的超集/替代锚点”。

---

## 3. 本项目如何对接论文思路（当前实现）

当前闭环结构（与论文“独立锚点 + 纠错融合”同构）：

1) **独立锚点（anchor）**：`contacts_plan`

- `contacts_plan = GRU(cond-only)`：只看 `cond`（控制信号）与自身隐藏状态，不看预测姿态；
- 目的是让锚点不受 drift 污染（即使 `y_t` 错了，`contacts_plan` 仍应有可用信号）。

2) **观测（measurement）**：`contacts_meas`

- `contacts_meas = MLP(pose_history, angvel)`：从当前预测状态的派生量（pose history / angular velocity）估计“接触观测”；
- 这是闭环的“传感器”，用于计算 innovation。

3) **闭环误差（innovation）**：`contacts_err`

- `e_t = contacts_plan - contacts_meas`

4) **SO(3) 小角度纠偏（corrector）**

- `omega_hat = f(h_final, e_t)`
- 在合成时做：`ΔR_used = Exp(gate * omega_hat) @ ΔR_pred`（并限制 `so3_corr_max_deg`）

训练流程（本轮）：

- Stage A：只训 `contact_meas_head`（把 meas 训成“真的 meas”）
- Stage B：训 SO(3) corrector（可选继续训 meas），得到 `ckpt_last_posttrain_meas_plus_corr.pth`

---

## 4. 实际偏差是什么？（本轮 freerun_cycles 数据结论）

验证文件：`debug_output/freerun_cycles/posttrain_meas_plus_corr/Walk_F_freerun_cycles.json`

### 4.1 Drift（多轮循环）仍然明显

`Walk_F`：`rounds=2`，`cycle_len=87`

Round 0（第 1 轮）：

- `GeoDeg = 8.45`
- `GeoLocalDeg = 14.01`
- `RootPosErrMean = 0.199`，`RootPosErrEnd = 1.359`
- `RootGeoDeg = 9.50`

Round 1（第 2 轮）：

- `GeoDeg = 17.43`（相比 Round0 明显变差）
- `GeoLocalDeg = 31.80`
- `RootPosErrMean = 1.544`，`RootPosErrEnd = 1.713`
- `RootGeoDeg = 25.05`

结论：**闭环尚未把“多轮误差累积”压住**。

### 4.2 闭环信号本身是“有信息量”的

从 per-step 统计看：

- Round0：`ContactErrAbsMean` 平均约 `0.365`
- Round1：`ContactErrAbsMean` 平均约 `0.508`
- 并且 `corr(GeoDeg, ContactErrAbsMean) ≈ 0.77`（整体相关性很高）

结论：`contacts_plan - contacts_meas` 这条通路在“反映 drift”，并非完全无效。

---

## 5. 为什么没有解决对应问题？（根因解释 + 本轮追加修改反馈）

本轮失败点不是“闭环想法不对”，而是 **闭环纠偏稳定性/尺度校准不足（不是只差增益） + 锚点时序信息不足**，导致 drift 无法被压住。

### 5.1 SO(3) gate 很小，但问题并不是“只差开大增益”

在 `ckpt_last_posttrain_meas_plus_corr.pth` 中：

- `so3_corr_gate_logit ≈ -3.99`
- `sigmoid(gate) ≈ 0.018`

直觉上这会把 correction 缩得很小；但本轮的 **knob test** 显示：**推理时强行把 gate 开大不会立刻改善，反而会炸掉**。

强行开 gate（并保持 `so3_corr_max_deg=10`）：

- baseline（learned gate≈0.018）：Round0 `GeoDeg ≈ 8.45`；Round1 `GeoDeg ≈ 17.43`
- `--so3_corr_gate_force 0.2`：Round0 `GeoDeg ≈ 61.40`；Round1 `GeoDeg ≈ 95.81`
- `--so3_corr_gate_force 0.3`：Round0 `GeoDeg ≈ 70.38`；Round1 `GeoDeg ≈ 92.94`

同时 `RootPosErrMean` 在三者中完全一致（Round0 `≈0.199`，Round1 `≈1.544`），原因是 **SO(3) corrector 只改旋转，不改 root 平移**。

结论：

- gate 小确实“限制了纠偏幅度”，但 **corrector 目前不具备在大增益下稳定工作的能力**（方向/尺度/输入对齐可能有问题，或者训练时从未见过大开度）。

### 5.2 `contacts_plan` 在本 clip 上退化为接近常数（锚点缺少“相位推进”）

在 `Walk_F` 的 `cond_in`（7 维）中，绝大部分维度几乎常数，只有少数维度有变化；因此 cond-only GRU 很容易收敛到固定点输出。

在 freerun log 里也能看到（baseline / Round1=第二轮）：

- `ContactPlanMean mean ≈ 0.606` 且 `std ≈ 0.004`（近似常数）

这会带来两个后果：

- 锚点本应提供“现在处于哪一段动作”的信息，但常数 plan 几乎不携带 phase；
- `e_t` 变成“常数 - 常数”的差，虽然会随 meas 漂移变大，但它缺少“把动作拉回正确节律”的能力。

#### 5.2.1 修复：给 `contacts_plan` 加 multi-frequency time positional encoding（time-PE）

本轮已做的改动（核心点）：

- `train/models.py`：对 `contacts_plan` 的 logits 叠加可学习的 `time_head(PE(t))`（Transformer-style 多频率 sin/cos，不依赖 `cycle_len`），并在 `EventMotionModel.forward(...)` 增加 `time_index` 输入。
- `train/validate/run_freerun_cycles.py`：free-run 每步传 `time_index`（支持 `global/cycle/auto`；multi-cycle 默认 `auto→cycle` 以避免 OOD），并从 ckpt 自动推断 `contact_plan_time_pe_dim`。

本节对应 ckpt / 输出：

- ckpt：`models/MLPL2_uncertainty_v2/ckpt_last_posttrain_meas_plus_corr_timepe16_planhead.pth`
- freerun log：`debug_output/freerun_cycles/timepe16_planhead/Walk_F_freerun_cycles.json`

直接效果（`Walk_F` / Round1=第二轮）：

- baseline：`ContactPlanMean std ≈ 0.004`（几乎常数）
- time-PE（relative t）：`ContactPlanMean std ≈ 0.054`（明显非零 ✅）

结论：**“plan 缺少相位推进 → 退化为常数”这个症状已部分修复**（至少 plan 输出开始随时间变化）。

#### 5.2.2 新问题：multi-cycle 的时间跨度 > 训练可见范围，abs time_index 会 OOD（“上下文太长”）

为了让训练/推理一致，我们进一步把训练端也接上 `time_index`：

- `train/training_MPL.py`：在 `Trainer._rollout_sequence(...)` 里传 `time_index = batch['start'] + t`（并在 freerun 子窗口里传 `batch['start'] + start + t`）。

然后只 finetune `contact_plan_time_head` 得到新 ckpt：

- `models/MLPL2_uncertainty_v2/ckpt_last_posttrain_meas_plus_corr_timepe16_planhead_absstart.pth`

现象非常“二极管”：

- （旧版 `freerun_cycles` / `time_index=global t` / legacy round slicing）  
  - Round0（t∈[0,86]）：`ContactPlanGtAbsMean ≈ 0.0615`（大幅下降 ✅）  
  - Round1（t∈[87,172]）：`ContactPlanGtAbsMean ≈ 0.4817`（比 baseline `0.4046` 更差 ❌）

**注：** `freerun_cycles` 的 `step=t` 本质是 transition（t→t+1）的预测步数。对 `Walk_F` 这种 87 帧 clip，**每个 cycle 内部只有 86 个“可比”的 step（t=0..85）**；旧脚本 Round0 把 wrap boundary（t=86，86→87=下一轮的起点）算进 Round0，导致 Round0=87 steps、Round1=86 steps（统计口径不对齐）。

修复（phase time_index）后 multi-cycle 不再退化：

- （新版 `freerun_cycles` / `time_index=t%cycle_len` / `round_seg_mode=intra`，输出：`debug_output/freerun_cycles/timepe16_planhead_absstart_cycleidx/Walk_F_freerun_cycles.json`）  
  - Round0（t∈[0,85]）：`ContactPlanGtAbsMean ≈ 0.0627`（保持很低 ✅）  
  - Round1（t∈[87,172]）：`ContactPlanGtAbsMean ≈ 0.0621`（不再跑偏 ✅）

根因（就是“上下文/时域太长”）：

- `Walk_F` 单 clip 长度只有 87 帧；训练阶段的 `time_index` 天花板也就到 `86`（因为 `time_index = start + t`）。
- `freerun_cycles` 的 Round1 会继续用全局步数 `t=87..172`；
- multi-frequency time-PE **不会在 87 帧处自动对齐回到“同一个 phase”**，因此 Round1 进入训练域外的时间索引（OOD），plan 直接跑偏。

结论：time-PE 想在 **多轮循环** 下当 phase anchor，需要把时间输入变成 “phase-like”（例如 `t % cycle_len` / `t % clip_len`），或者让训练覆盖更大的 time_index 范围（单 clip 长度限制下做不到，只能通过“重复拼接/合成更长序列”或引入外部 phase 信号）。

### 5.3 `contacts_meas` 仍偏向均值解：输入与 GT 生成机制不匹配（隐式 FK 过难）

把 GT 的 “soft contact” 生成路径先写清楚（来自原始 JSON 的 `FootEvidence.*` 字段）：

- `pose` → FK → `foot_pos_world`
- `foot_pos_world` → `foot_height_world_m` / `dist_to_ground_m`
- `foot_pos_world` 的时间差分 → `vxy_mps` / `vz_mps`（或 `||v||`）
- 一套门控/阈值/平滑（通常是 sigmoid/clip 的组合）→ `soft_contact_score ∈ [0, 1]`

而当前 meas 是：

- `pose_history, angvel` → `contact_meas_head`(MLP) → `contacts_meas`

这会迫使 MLP **隐式学会一个“FK + 高度/速度证据”的复合测量函数**：

- FK 是强非线性的链式传递（parent→child accumulation），并且对骨骼拓扑/骨长敏感；
- `foot_height_world_m` 依赖世界系与 ground plane（若输入里没有显式的 root height / ground_z 信息，问题本质上是欠定的）；
- 监督目标本身是 soft label（带噪/阈值平滑），再叠加 contact 分布不均衡时，最容易收敛到 **预测常数/均值** 来最小化期望误差。

所以你给出的判断基本成立：**“MLP 需要隐式学 FK → 难 → 退化成均值解”是一个非常合理的根因解释**。

现象上也符合：

- `ContactMeasMean` 在 time-series 上非常平（std 很小），更像“均值解”而不是随动作 phase 起伏的传感器输出：
  - global/no-apply：mean≈`0.423`，std≈`0.016`
  - global/apply10：mean≈`0.474`，std≈`0.014`
  - cycle/no-apply：mean≈`0.422`，std≈`0.016`
  - cycle/apply10：mean≈`0.451`，std≈`0.019`
- `ContactMeasGtAbsMean` 仍较大（mean≈`0.43~0.48`），且方差明显（GT 在变、meas 近似常数），见图：`debug_output/freerun_cycles/compare_contacts_meas_err_timeseries.png`

这会让 `e_t = plan - meas` 的“测量侧”不够可靠：

- 在 5.2 用 `time_index=t%cycle_len` 把 plan 的 multi-cycle OOD 修掉后，`ContactPlanGtAbsMean` 两轮都能维持在 `~0.06`；但由于 meas 仍接近均值解，`ContactErrAbsMean` 反而变成“近似常数偏置”，与 `GeoLocalDeg` 的相关性很弱（cycle/apply10 下 per-step corr≈`0.09`），因此很难形成有效的“随 drift 增长而增强”的负反馈去拉回姿态。

#### 建议：把 `contacts_meas` 改成白盒 measurement function（可解释、无拟合误差）

思路：直接复刻 FootEvidence 的逻辑：`pose → FK → foot_height/velocity → contact`，避免让 MLP 去学 FK。

```python
import torch
import torch.nn.functional as F

def contacts_meas(
    dist_to_ground_m,      # (...,)  ≥0
    vxy_mps,               # (...,)
    vz_mps,                # (...,)
    hit_flag=None,         # (...,) bool (可选，等价 FootEvidence.hit_flag)
    *,
    # 来自 meta['foot_evidence']['soft_score_spec']
    dist0_cm=0.5,
    alpha_dist=2.0,
    vxy0_cmps=96.0,
    alpha_vxy=0.2,
    vz0_cmps=40.0,
    alpha_vz=0.5,
    # 与 JSON 的数值范围对齐（hit=1 时约在 [1e-4, 0.9]）
    min_score=1e-4,
    max_score=0.9,
    scale=0.92,
):
    """
    White-box contact measurement (FootEvidence-style):
      pose -> FK -> dist_to_ground / (vxy, vz) -> soft_contact_score

    关键点：
      - dist gate 用 sigmoid（并归一化到 dist=0 时为 1）
      - velocity gate 用 hinge + exp（低于阈值不惩罚）
    """
    dist_cm = dist_to_ground_m * 100.0
    vxy_cmps = vxy_mps * 100.0
    vz_cmps = vz_mps * 100.0

    # distance gate (normalized so dist=0 -> 1.0)
    dist0_cm = max(1e-6, float(dist0_cm))
    alpha_dist = max(1e-6, float(alpha_dist))
    dist_raw = torch.sigmoid(alpha_dist * (dist0_cm - dist_cm) / dist0_cm)
    dist_score = (dist_raw / torch.sigmoid(torch.tensor(alpha_dist, device=dist_raw.device))).clamp(0.0, 1.0)

    # velocity gates (no penalty when below thresholds)
    vz0_cmps = max(1e-6, float(vz0_cmps))
    alpha_vz = max(1e-6, float(alpha_vz))
    vz_score = torch.exp(-F.relu(vz_cmps - vz0_cmps) / (alpha_vz * vz0_cmps))

    vxy0_cmps = max(1e-6, float(vxy0_cmps))
    alpha_vxy = max(1e-6, float(alpha_vxy))
    vxy_score = torch.exp(-F.relu(vxy_cmps - vxy0_cmps) / (alpha_vxy * vxy0_cmps))

    contacts = dist_score * vz_score * vxy_score
    contacts = contacts * float(scale)
    contacts = contacts.clamp(0.0, float(max_score))

    if hit_flag is not None:
        contacts = contacts * hit_flag.to(dtype=contacts.dtype)
        # hit=1 时 clamp 到 min_score；hit=0 时保持 0
        if float(min_score) > 0.0:
            contacts = torch.where(hit_flag, contacts.clamp_min(float(min_score)), contacts)
    elif float(min_score) > 0.0:
        contacts = contacts.clamp_min(float(min_score))

    return contacts
```

好处（vs MLP）：

| 对比 | MLP meas | 白盒 meas |
|------|----------|-----------|
| 是否需要训练 | 要 | 不要（或只学少量标定参数） |
| 是否有拟合误差 | 有（均值解风险） | 无（确定性映射） |
| 可解释性 | 黑盒 | 完全可解释 |
| 推理开销 | MLP forward | 几行数学运算 |

对应闭环结构变成：

```
contacts_plan = GRU(cond, h_{t-1})                       ← 学习的：提供“期望节律/phase anchor”
contacts_meas = WhiteBox(FK(pose) → height/velocity)     ← 确定的：提供“实际观测/sensor”
e_t = plan - meas                                        ← 干净的误差信号
```

实现落点（代码层面）：

- `train/training_MPL.py:Trainer._contact_meas_whitebox()`：用 `rot6d → FK → foot_pos` 直接算 `contacts_meas`
- `train/validate/run_freerun_cycles.py` / `train/posttrain.py`：在 `contact_plan_enable=True` 的闭环路径里调用该白盒 meas

备注：

- 若担心阈值不准，可把 `dist0_cm/vxy0_cmps/vz0_cmps/alpha_*` 做成可学习标量（per-foot / per-action），但仍保持 “FK + 公式” 的白盒骨架；
- 这不会自动解决所有 drift（corrector 仍需要稳定训练），但会显著提升 `contacts_err` 的时序信息量与与姿态误差的相关性，是后续纠偏稳定性的前提。

---

## 6. 下一步改进思路（按优先级）

### 6.1 先用验证脚本确认：如果把 gate 打开，corrector 是否“有能力”压 drift？

已验证（见上面的 knob test 输出）：推理时强行开 gate 并不会改善，反而会让 `GeoDeg` 直接爆炸到 `60~95°` 量级；并且 `RootPosErr` 完全不变（corrector 不影响平移）。

因此这里的结论更新为：

- **结构不是只差增益**，不建议推理时硬开 gate。
- 如果要走 corrector 路线，需要在 posttrain 中让模型见过更大的 gate（curriculum/warmup），并配合更强的稳定性约束（例如更严格的 `omega_hat` 正则、误差对齐、或更保守的 gate 调制逻辑）。

### 6.2 调整 posttrain Stage B：别让 gate 从一开始就卡在 ~0

目前的配置里 `so3_corr_gate_logit_reset=-4`（约 0.018），再叠加 warmup/短训练步数，会导致 gate 很难学开。

建议：

- 将 `so3_corr_gate_logit_reset` 调到 `-2`（gate≈0.12）或 `-1`（gate≈0.27）；
- 缩短或提高 warmup（例如 warmup value 0.1，而不是 0.02）；
- 增加 Stage B 训练步数（例如 2k+ steps），否则 gate 刚开始能学就结束了。

### 6.3 强化锚点：让 `contacts_plan` 真正携带“相位推进”

如果 `cond` 在某些 clip 近似常数，cond-only GRU 很容易输出常数；time-PE 能让它“动起来”，但要注意 multi-cycle 下如果用 global/abs time_index 会 OOD（见 5.2.2；用 `t%cycle_len` 可对齐回训练域）。

可选方案：

- 给 plan 输入增加“时间推进相关”的 cond 特征（速度/轨迹方向/动作进度等）；
- 为 plan 引入显式的“相位状态”（类似论文的 phase accumulator），但仍保持它不依赖预测姿态；
- 如果目标是 locomotion loop：让 `time_index` 变成 **phase-like**（例如 `t % cycle_len` / `t % clip_len`），避免跨轮 OOD；
- 增加锚点通道（不仅脚接触，加入手/球/武器等事件型接触），减少多对一。

### 6.4 强化测量：让 meas 更像“传感器”而不是均值解

当前 meas 仅由旋转历史推断接触，可能信息不足。可选增强：

- **白盒 meas**：用 `pose → FK → foot_height/velocity → contact` 直接算 `contacts_meas`（见 5.3），避免 MLP 隐式学 FK；
- 给 meas 增加更接触相关的输入：root/bone velocity、foot height proxy、骨骼末端速度/加速度等；
- 对 meas 做不确定性/可靠性建模（输出 `reliability r_t`），用 `e_t := r_t*(plan-meas)` 调制纠偏强度；
- 在训练中加入“带漂移输入”的 meas 强化（对 pose_history 做 delay/noise/drop 已有雏形，可继续增强）。

### 6.5 借用论文“双路径融合”的真正收益

目前我们主要在 “ΔR_pred” 上做 on-manifold correction，但没有 direct vs incremental 的显式融合结构。

后续可以：

- 增加 direct branch（预测 absolute pose 或更强的 anchor-conditioned 分支）；
- 与 incremental（现有 Δ 输出）做 gate/λ 融合；
- 让 gate 依赖 anchor 可信度/contacts_err，形成更稳健的容错结构。

---

## 7. 本轮结论（TL;DR）

- `freerun_cycles` 的 Round1 明显比 Round0 差，drift 仍在累积。
- 闭环误差信号 `contacts_err` 与姿态误差高度相关，说明“反馈信号有信息量”。
- SO(3) gate 虽然很小（≈0.018），但 **knob test 显示推理时强行开大只会爆炸**；结构不是只差增益。
- `contacts_plan` 的“退化常数”问题通过 time-PE 已部分解决：`ContactPlanMeanStd` 从 `~0.004` 提升到 `~0.09~0.11` 量级（明显非零）。
- multi-cycle 下如果用 **global/abs time_index** 会触发 OOD，导致 Round1 的 `ContactPlanGtAbsMean` 跑偏到 `~0.48`；把 time_index 改成 **phase-like**（`t % cycle_len`）后，Round0/1 都能回到 `~0.06`（见 5.2.2）。
- `contacts_meas` 仍很像均值解：`ContactMeasMean` 的 std 只有 `~0.014~0.019`，导致 `contacts_err` 在 plan 修好后更像“常数偏置”而不是 drift 反馈（见 `debug_output/freerun_cycles/compare_contacts_meas_err_timeseries.png`）。
- 下一步：锚点 time_index 的 OOD 已能通过 phase-like 输入规避；优先把 meas 做成更像传感器/带可靠度的测量，再回到 corrector 的稳定性训练（curriculum 开 gate，而不是推理时硬开）。
