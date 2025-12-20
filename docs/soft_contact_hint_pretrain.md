# Soft Contact Hint Pretraining (No Phase / No Rising-Edge)

## TL;DR

- 预训练不再做 `soft_contacts → 上升沿检测/相位插值 → sin/cos phase`。
- 直接用 `soft_contacts`（物理信号）作为预训练 hint/target，并映射到 `[-1, 1]` 以匹配 `tanh` 输出。
- 这避免了“周期性假设 + 非周期 fallback 造假相位”的分布偏差，对非周期/过渡动作泛化更稳。

---

## 背景：为什么移除 phase 中间层？

旧流程（已移除）：

`soft_contacts → [rising edge + phase interpolation] → soft period [sin, cos] → pretrain target`

主要问题：

1. **强周期性假设**：上升沿检测默认动作必须可重复（周期性）。
2. **非周期动作退化**：上升沿不足时会 fallback 到“线性相位”，这不是物理信号。
3. **学到的是人为构造目标**：预训练会偏向拟合“如何预测假相位”，而不是学习接触本身的动力学关联。

新流程（当前）：

`soft_contacts → pretrain target`

soft contacts 的性质：

- 周期动作：左右脚接触会自然交替变化（可隐式提供节奏信息）。
- 非周期动作：接触会稳定或单次变化（仍然有明确物理意义，不会被强行扭成周期相位）。

---

## 数据来源与字段约定

### JSON

软接触来自每帧：

- `Frames[i].FootEvidence.L.soft_contact_score`
- `Frames[i].FootEvidence.R.soft_contact_score`

加载函数：`train/io.py:27` 的 `load_soft_contacts_from_json()`，返回 `soft_contacts[T, 2]`，范围约定为 `[0, 1]`。

### NPZ

`raw_data/processed_data/*.npz` 通过 `source_json` 回指原 JSON；预训练数据集会对齐 `npz` 与 `json` 的最短长度，并使用 `Frames[1:]` 与角速度差分对齐（见 `train/pretrain_mpl_min.py`）。

---

## 预训练目标定义（contact hint）

因为预训练的 `period_head` 输出经过 `tanh`：

```python
contact_hint = 2.0 * soft_contacts - 1.0   # [T, 2] in [-1, 1]
```

训练时约束：

- `soft_period = tanh(period_head(h_period))`，shape `[B, T, period_dim]`
- 只对齐前 2 维：`soft_period[..., :2] ≈ contact_hint`

其余维度（`period_dim - 2`）不强加周期性结构，可用于编码更丰富的 motion hint（供 pose/ang 解码器使用）。

对应实现：

- 数据集生成：`train/pretrain_mpl_min.py`（`period_hint` 现在固定为 `contacts_tanh`）
- hint 损失：`loss_hint = MSE(soft_period[..., :2], period_hint)`

---

## 与主训练的衔接（frozen encoder → hint injection）

主训练通过 `EventMotionModel.attach_motion_encoder()` 加载并冻结预训练的：

- `MotionEncoder`（frozen）
- `PeriodHead`（frozen）

然后在 `EventMotionModel.forward()` 中：

1. 用 `contacts/angvel/pose_history` 拼出 `encoder_input`
2. `frozen_encoder(encoder_input)` 得到 `enc_hidden`
3. `tanh(frozen_period_head(enc_hidden))` 得到 `soft_period`（现在是 **contact-hint embedding**）
4. 通过 `period_encoder` 投影到主干 hidden 并注入

注意：代码里仍沿用 `period_*` 命名是历史兼容，但语义已变为 **soft hint**（第一性约束是接触，而不是相位）。

---

## 诊断指标（训练/评估日志）

主训练侧保留 embedding 对齐的诊断（如果模型同时产出 period_pred）：

- `Period/EmbedL2`：预测 embedding 与 frozen embedding 的 L2
- `Period/EmbedCos`：预测 embedding 与 frozen embedding 的 cosine similarity

新增 contact-hint 诊断（始终基于 `2*contacts-1`）：

- `Period/ContactHintMAE`：`period_pred[..., :2]` 与 `contact_hint` 的 MAE
- `Period/ContactHintGTMAE`：`period_gt[..., :2]` 与 `contact_hint` 的 MAE（用于 sanity check）

旧的 `PhaseMAE / PhaseCosSim / phase-bin 曲线` 已弃用。

---

## 迁移与使用建议

1. 重新跑预训练生成新的 `encoder_path`（不要继续用旧 phase-based bundle）。
2. 在 `config/exp_phase_mpl.json` 更新：
   - `encoder_path`: 指向新的预训练产物
3. 观察主训练日志：
   - `Period/ContactHintGTMAE` 应该较小（代表 frozen hint 的前 2 维确实在表达接触）
   - `Period/ContactHintMAE` 随训练逐步下降（代表主模型的 hint 分支在跟随）

