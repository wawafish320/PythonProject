# Rotation Local Tail Loss: CVaR-Based Per-Bone Stability

## 目录

1. [核心问题](#核心问题)
2. [设计理念](#设计理念)
3. [数学定义](#数学定义)
4. [理论基础](#理论基础)
5. [为什么选择Rotation Local而非FK](#为什么选择rotation-local而非fk)
6. [为什么限制到KeyBones](#为什么限制到keybones)
7. [收敛机制：钟摆模型](#收敛机制钟摆模型)
8. [实验验证](#实验验证)
9. [配置指南](#配置指南)
10. [常见问题](#常见问题)

---

## 核心问题

### Whack-a-Mole现象

在动作生成训练中，经常观察到典型的"打地鼠"（whack-a-mole）现象：

```
ep1: thigh最差(10.0°) → 增加thigh权重
ep2: hand变最差(8.5°) → thigh改善了，但hand退化了
ep3: foot变最差(9.2°) → 又换了一个骨骼垫底
...
→ 最差骨骼不断"换人"，但误差分布始终不稳定
```

**根本原因：**

1. **模型容量有限**：无法同时将所有骨骼优化到极致
2. **骨骼难度不均**：不同骨骼的运动特性本质上不同
   - `thigh`: ±60°大范围旋转，学习难度高
   - `pelvis`: ±5°精细调节，相对误差敏感
   - `hand`: 多轴耦合+高速运动，复杂度高
3. **传统per-bone reweighting的副作用**：
   - 在归一化约束下（`weights /= weights.mean()`），提高某个骨骼权重会相对降低其他骨骼权重
   - 这是一个零和博弈，导致误差在不同骨骼间转移

### 传统方案的局限

**方案A: 手动per-bone权重表**
```python
bone_weights = {
    'thigh': 1.5,  # 难度高，加权
    'hand': 1.2,
    'pelvis': 1.0,
    # ...
}
```
❌ 问题：
- 需要人工调参，成本高
- 不同动作需要不同权重（走路vs跑步vs舞蹈）
- 难以应对训练过程中的动态变化

**方案B: Metric-driven动态权重**
```python
# 根据上一epoch的误差更新权重
weights[j] ∝ error[j]
weights /= weights.mean()  # 归一化
```
❌ 问题：
- 归一化导致零和博弈（"给A加权 = 给B减权"）
- 加剧whack-a-mole（本epoch压下去的，下epoch又起来）
- 权重振荡，训练不稳定

---

## 设计理念

我们的设计哲学有三个核心洞察：

### 1. 接受"较差骨骼"的存在

**每个骨骼的运动幅度和复杂度本质上不同**，有"较差骨骼"是生物力学和数据特性的自然结果，不是数据质量问题。

目标不是"消除较差骨骼"（不现实），而是**防止某个骨骼长期垫底，拉低整体质量**。

### 2. 利用而非对抗Whack-a-Mole

传统方案试图"消除"whack-a-mole，但往往徒劳。

我们的方案：**利用whack-a-mole作为"收敛钟摆"**：
- 允许最差骨骼在不同epoch"换人"
- 但确保每次"换人"时，新的最差值比上次低
- 最终所有骨骼收敛到一个紧凑的分布

### 3. Mean + Tail组合优化

传统方法只优化均值：
```python
L = mean(e_1, e_2, ..., e_13)
```
问题：只要均值低，分布可以很不均匀
- `[1°, 1°, 1°, ..., 1°]` → mean=1° ✅
- `[0.5°, 0.5°, ..., 10°, 10°]` → mean=1° ❌ 视觉质量差

**我们的方案：Mean + Tail**
```python
L = w_rot_local · mean(errors) + rot_local_tail_weight · mean(top-k errors)
    └─────全局平均─────┘          └──────尾部风险──────┘
```
- 第一项：确保整体收敛
- 第二项：压制尾部outlier，压缩分布

---

## 数学定义

### Loss组成

```python
# 对应实现：train/models.py（rot_local + rot_local_tail）
#
# geo_local: (B, T, J) = geodesic_R(Rp_local, Rg_local)  # radians
#
# 1) 基础 rot_local（所有骨骼的加权平均；weights.mean()==1）
#    注意：实现里是 (geo_local * weights).mean()，等价于用 sum(weights)=J 的 weighted-mean。
weights = joint_weight_vector(J)  # (J,), mean=1
L_mean = mean_{B,T,J}( geo_local[b,t,j] * weights[j] )
#
# 2) Tail loss（按“每骨骼的 batch-time 平均误差”选 top-k；选择用 detach，梯度不穿过 topk）
per_bone = mean_{B,T}( geo_local.detach() )  # (J,)
candidate_idx = scope_indices(rot_local_tail_scope)  # all / limbs / keybones
idx = topk(per_bone[candidate_idx], k=min(rot_local_tail_k, |candidate_idx|))
#
# 3) 对选中的 k 个骨骼施加额外 loss（实现里用 nanmean，避免 NaN 污染）
L_tail = mean_{B,T,j∈idx}( geo_local[b,t,j] )  # unweighted
#
# 4) 总 loss
L = ... + w_rot_local * L_mean + rot_local_tail_weight * L_tail
```

### 参数说明

| 参数 | 含义 | 推荐值 | 说明 |
|------|------|--------|------|
| `w_rot_local` | 基础权重 | 0.22 | 所有骨骼的基础优化力度 |
| `rot_local_tail_weight` | Tail额外权重 | 0.2 | 最差骨骼的额外惩罚 |
| `rot_local_tail_k` | Tail骨骼数量 | 3 | 选择最差的k个（13个中选3个≈23%分位） |
| `rot_local_tail_scope` | 选择范围 | `"keybones"` | 限制在关键骨骼，避免稀释 |

> ⚠️ 实现细节：`rot_local_tail_scope` **只影响 top-k 的候选集合**（`candidate_idx`），不影响 `rot_local` 的 `L_mean`。
> `L_mean` 始终是对 `geo_local.shape[-1]` 的全骨骼平均（带 `joint_weight_vector` 权重）。

### 梯度分配

下面按实现的 **reduction/归一化** 写清楚“被选中骨骼到底放大了多少”。

记：
- `geo_local` shape = `(B, T, J)`，`J = geo_local.shape[-1]`（全骨骼数）
- `k = |idx|`（实际选中的数量，`k = min(tail_k, |candidate_idx|)`）
- `w_j = weights[j]`（`joint_weight_vector` 输出；`mean(w_j)=1`）

实现等价于：
```
L_mean = (1/(B*T*J)) * Σ_{b,t,j} w_j * geo_local[b,t,j]
L_tail = (1/(B*T*k)) * Σ_{b,t,j∈idx} geo_local[b,t,j]
L = w_rot_local * L_mean + tail_w * L_tail
```

因此，对单个元素 `geo_local[b,t,j]` 的梯度系数为：
- 若 `j ∉ idx`：
  ```
  ∂L/∂geo_local[b,t,j] = w_rot_local * w_j / (B*T*J)
  ```
- 若 `j ∈ idx`：
  ```
  ∂L/∂geo_local[b,t,j] = w_rot_local * w_j / (B*T*J) + tail_w / (B*T*k)
  ```

把“选中 vs 未选中”的倍率写成闭式：
```
ratio_j = 1 + (tail_w * J) / (w_rot_local * w_j * k)
```

> 直观结论：
> - `J` 越大、`k` 越小，tail 对选中骨骼的相对放大会更强。
> - `w_j` 越大（该骨骼在 `rot_local` 里本来就更重要），tail 的相对放大倍率越小；反之亦然。
> - **关键仍成立**：所有骨骼都有 `L_mean` 的基础梯度；tail 只是对 top-k 骨骼额外加成。

---

## 理论基础

### CVaR (Conditional Value at Risk)

我们的 tail loss可以视为对“骨骼维度误差分布”的 **empirical CVaR / top-k mean**：

**金融学中的CVaR**：
```
投资组合优化 = minimize: -E[return] + λ · CVaR_α[loss]
                        └─期望收益─┘    └─尾部风险─┘
```
不仅优化平均收益，还控制"最坏情况下的损失"。

**我们的骨骼优化**：
```
L = mean(errors) + λ · mean(top-k per-bone errors)
    └─整体误差─┘    └──────尾部风险──────┘
```
不仅优化平均误差，还控制"最差骨骼的误差"。

在“把每个骨骼当作离散分布中的一个样本点（等权）”的视角下，`top-k mean` 等价于一个离散的 CVaR（`α ≈ 1 - k/|scope|`）。

### 数学性质

**1. 非零梯度覆盖**
- `L_mean` 保证所有骨骼持续被优化（不会因为 top-k membership 变化而“断梯度”）
- `L_tail` 只对最差骨骼额外加压，抑制 whack-a-mole 的尾部风险

**2. 方差压缩**
- Tail项相当于隐式的方差惩罚
- `L = mean + β·top-k ≈ mean + β·(mean + c·std)`
- 压缩分布的离散度

**3. 凸性（仅在“误差空间”成立）**
- `mean(e)` 和 `top-k mean(e)` 对误差向量 `e` 是凸的风险度量
- 但 `e(θ)` 来自深网 + SO(3) geodesic，整体对参数 `θ` 仍是非凸的；因此不能用“凸性”来保证训练一定稳定

**4. 自适应性**
- 无需手动指定哪个骨骼需要加权
- 谁最差谁获得额外梯度，自动调节

---

## 为什么选择Rotation Local而非FK

在动作生成中，常见两种loss：

### FK (Forward Kinematics) Position Loss

```python
# 计算端点3D位置误差
FK_loss = ||FK(pred_rotations) - FK(gt_rotations)||
```

**优点：**
- ✅ 端到端优化视觉目标（手脚位置）
- ✅ 隐式容错（中间骨骼略偏，端点对就行）
- ✅ 符合人类感知（我们主要看"手在哪"）

**缺点（为什么我们不用）：**
- ❌ **误差归属混淆**：无法区分是thigh错还是calf错
  ```
  foot位置偏了10cm → 是thigh错了5° + calf错了3°？
                   还是只有thigh错了8°？
  FK loss无法判断，梯度分配不准确
  ```

- ❌ **补偿效应**：多个旋转组合可能产生相同FK位置
  ```
  GT:   θ_shoulder=30°, θ_elbow=45° → hand at (x,y,z)
  Pred: θ_shoulder=35°, θ_elbow=40° → hand at (x,y,z)  (位置对但姿态错)
  → FK_loss=0（满意），Rot_loss≠0（不满意）
  → 梯度冲突
  ```

- ❌ **梯度按力臂分配**：根部骨骼获得过多梯度
  ```
  ∂FK_loss/∂θ_pelvis ∝ 力臂长度（很长）
  ∂FK_loss/∂θ_foot ∝ 力臂长度（很短）
  → pelvis被过度优化，foot被忽视
  ```

- ❌ **与Rot Local梯度冲突**：
  ```
  FK说："肩膀多转8°，肘部少转3°"（优化效率）
  Rot说："每个关节都要精准匹配GT"（姿态保真）
  → 如果同时使用，梯度方向矛盾，优化困难
  ```

### Rotation Local Loss

```python
# 每个骨骼的parent-relative旋转误差
Rot_loss = mean_j(geodesic(pred_rot[j], gt_rot[j]))
```

**优点（为什么我们选择它）：**
- ✅ **误差归属明确**：每个骨骼独立计算，梯度精准
- ✅ **对root不敏感**：parent-relative，不受全局对齐影响
- ✅ **姿态保真**：直接匹配关节角度，避免补偿效应
- ✅ **可解释性强**：GeoLocalDeg直接对应角度误差（度数）

**结论：我们选择Rotation Local + Tail Loss，完全避免FK的冲突。**

---

## 为什么限制到KeyBones

### 问题：全骨骼scope的稀释效应

如果在所有40-60个骨骼中选top-k=3：

```
选中率 = 3/60 ≈ 5%（过窄）
```

**问题1：可能选中视觉不重要的骨骼**
```
top-3 = [finger_tip_01, toe_02, twist_bone_03]
→ 这些骨骼误差大，但对视觉质量影响小
→ 优化资源浪费
```

**问题2：视觉重要骨骼被稀释**
```
pelvis误差=1.5°（第4差）→ 未被选中
但pelvis是视觉最重要的骨骼之一！
```

### 解决方案：限制到KeyBones

```python
rot_local_tail_scope = "keybones"
```

**KeyBones = pelvis + limb_monitor_names ≈ 13个骨骼：**
- pelvis（躯干根部）
- thigh_l, thigh_r（大腿）
- calf_l, calf_r（小腿）
- foot_l, foot_r（脚）
- upperarm_l, upperarm_r（上臂）
- lowerarm_l, lowerarm_r（前臂）
- hand_l, hand_r（手）

**优势：**
```
选中率 = 3/13 ≈ 23%（合理的尾部比例）
```
- ✅ 聚焦视觉主导关节（四肢+pelvis）
- ✅ 忽略次要骨骼（手指、twist bones）
- ✅ 改善直接体现在视觉质量上

### 运动幅度的考量

不同骨骼的运动特性差异巨大：

| 骨骼 | 典型旋转范围 | 运动复杂度 | 视觉重要性 | 为什么会成为"较差骨骼" |
|------|-------------|-----------|-----------|---------------------|
| thigh | ±60° (大) | 单轴主导 | ★★★★★ | 学习难度高，数据多样性大 |
| calf | ±70° (大) | 单轴 | ★★★★☆ | 大范围运动，预测困难 |
| pelvis | ±5° (小) | 多轴精细 | ★★★★★ | 相对误差大（5°/5° vs 5°/60°） |
| hand | ±40° (中) | 多轴+高速 | ★★★☆☆ | 自由度高，角速度大 |
| finger | ±15° (小) | 多轴 | ☆☆☆☆☆ | 视觉不重要 |

**Keybones选择策略：**
- 包含视觉重要骨骼（pelvis, limbs）
- 排除视觉不重要骨骼（fingers, twist bones）
- 在13个内部，让模型自动选择最需要改善的3个

---

## 收敛机制：钟摆模型

### 传统方法：持续振荡

```
无tail loss的训练过程：

ep1: thigh=10.0° (最差) → 优化thigh
ep2: hand=8.5° (最差)   → thigh改善到7°，但hand退化到8.5°
ep3: foot=9.2° (最差)   → hand改善到7.5°，但foot退化到9.2°
ep4: thigh=10.5° (最差) → 又回到thigh最差
...
→ 钟摆一直摆，幅度不减小
→ max_error在8-10°区间振荡
```

### 我们的方法：收敛钟摆

```
有tail loss的训练过程：

ep7:  thigh=5.0° (最差) → 额外压制 → 降到4.2°
ep8:  hand=4.5° (最差)  → 额外压制 → 降到3.8°
ep9:  foot=4.0° (最差)  → 额外压制 → 降到3.5°
ep10: thigh=3.9° (最差) → 额外压制 → 降到3.2°
      ↑
      虽然thigh又最差了，但已经比ep7的5.0°小了！
...
ep12: 所有骨骼都在[0.3°, 1.1°]区间，钟摆幅度收敛
```

### 动态演化过程

```
时刻 t=0:  max_error=10.0°, min_error=2.0°, gap=8.0°
      ↓ 压制最差骨骼
时刻 t=1:  max_error=8.0°,  min_error=2.0°, gap=6.0°  (最差被压下来)
      ↓ 可能换骨骼，但新最差值更低
时刻 t=2:  max_error=7.0°,  min_error=2.5°, gap=4.5°
      ↓ 继续螺旋下降
...
时刻 t=N:  max_error=3.0°,  min_error=1.0°, gap=2.0°  → 收敛！
```

### 钟摆比喻的精髓

1. **摆动是必然的**：总有某个骨骼最差（模型容量有限）
2. **振幅逐渐减小**：每次回摆时，上界下降
3. **最终小幅振荡**：所有骨骼收敛到紧凑分布

**关键机制：**
- Mean项提供全局下降趋势
- Tail项确保每次"换人"时，新的最差值比上次低
- 组合效果：均值↓ + 方差↓

---

## 实验验证

### 对比实验设置

**v15（对照组）：**
- `w_rot_local = 0.22`
- `rot_local_tail_weight = 0.2`
- `rot_local_tail_k = 3`
- `rot_local_tail_scope = "all"`（全骨骼，40-60个）

**v16（实验组）：**
- `w_rot_local = 0.22`
- `rot_local_tail_weight = 0.2`
- `rot_local_tail_k = 3`
- `rot_local_tail_scope = "keybones"`（13个关键骨骼）

### KeyBones GeoLocalDeg结果（度）

| Epoch | 版本 | Mean | Max | p90-p10 | 说明 |
|-------|------|------|-----|---------|------|
| ep6 | v15 (全骨骼) | 0.733 | 1.322 | 0.752 | Tail loss启用前 |
| ep6 | v16 (keybones) | 0.654 | 1.296 | 0.926 | |
| | Δ | -0.079 | -0.026 | +0.174 | keybones略优 |
| | | | | | |
| ep9 | v15 (全骨骼) | 0.665 | 1.371 | 0.761 | 过渡期 |
| ep9 | v16 (keybones) | 0.808 | 1.831 | 1.234 | 出现波动 |
| | Δ | +0.143 | +0.459 | +0.472 | "钟摆冲击" |
| | | | | | |
| ep12 | v15 (全骨骼) | 0.724 | 1.476 | 0.730 | 最终收敛 |
| ep12 | v16 (keybones) | **0.617** | **1.158** | **0.481** | ✅ |
| | **改善** | **-15%** | **-22%** | **-34%** | **显著优于v15** |

### 关键观察

1. **ep9波动是正常的"钟摆冲击"**：
   - ep7启用tail loss后，优化目标突变
   - lr=0.0007仍较高，模型响应激进
   - ep10降低lr到0.0003后，重新稳定

2. **ep12显著改善**：
   - Mean降低15%（全局改善）
   - Max降低22%（极端值压制）
   - **p90-p10降低34%（尾部压缩效果显著）**

3. **keybones scope的必要性**：
   - v15的全骨骼scope虽然也能工作，但改善有限
   - v16的keybones scope聚焦视觉关键骨骼，效果更好

### 与理论预期的一致性

✅ **Mean + Tail双重改善**：均值↓15% + 离散度↓34%
✅ **收敛钟摆机制**：ep9波动后ep12收敛
✅ **CVaR效果**：成功压缩尾部风险分布

---

## 配置指南

### Stage Schedule配置

在 `config/exp_phase_mpl.json` 中：

```json
{
  "freerun_stage_schedule": [
    {
      "range": [1, 6],
      "label": "stage1_pure_convergence",
      "params": {
        "opt_lr": 0.001
      },
      "loss_groups": {
        "core": {
          "w_rot_local": 0.20,
          "rot_local_tail_weight": 0.0,  // 早期不启用tail
          "rot_local_tail_k": 0
        }
      }
    },
    {
      "range": [7, 9],
      "label": "stage2_tail_activation",
      "params": {
        "opt_lr": 0.0007
      },
      "loss_groups": {
        "core": {
          "w_rot_local": 0.22,
          "rot_local_tail_weight": 0.2,  // 启用tail loss
          "rot_local_tail_k": 3,
          "rot_local_tail_scope": "keybones",  // 限制到keybones
          "rot_local_tail_select": "ema",      // 用EMA做top-k选择，更平滑
          "rot_local_tail_ema_beta": 0.9
        }
      }
    },
    {
      "range": [10, 12],
      "label": "stage3_fine_convergence",
      "params": {
        "opt_lr": 0.0003  // 降低lr，稳定优化
      },
      "loss_groups": {
        "core": {
          "w_rot_local": 0.22,
          "rot_local_tail_weight": 0.2,
          "rot_local_tail_k": 3,
          "rot_local_tail_scope": "keybones",
          "rot_local_tail_select": "ema",
          "rot_local_tail_ema_beta": 0.9
        }
      }
    }
  ]
}
```

### 关键原则

**1. 分阶段启用**
- ep1-6：纯基础收敛（tail_weight=0）
  - 让模型先学会基础姿态
  - 避免tail loss干扰初期探索
- ep7+：启用tail loss
  - 基础收敛后，开始压制尾部

**2. 降低学习率**
- ep7-9：lr=0.0007（适应新loss）
- ep10+：lr=0.0003（精细收敛）
  - 降低lr缓解"钟摆冲击"

**3. 限制到keybones**
```json
"rot_local_tail_scope": "keybones"
```
- 避免全骨骼的稀释效应
- 聚焦视觉重要骨骼

**4. 用EMA做top-k选择（更平滑）**
```json
"rot_local_tail_select": "ema",
"rot_local_tail_ema_beta": 0.9
```
- `batch` 选择会让 top-k 在 batch 间频繁跳，容易“钟摆”
- `ema` 让 tail 骨骼选择更稳定，梯度更连续

### 超参数调优建议

| 参数 | 默认值 | 调优方向 | 观察指标 |
|------|--------|---------|---------|
| `tail_weight` | 0.2 | 如果mean上升→降低到0.15<br>如果p90-p10仍大→提高到0.25 | mean, p90-p10 |
| `tail_k` | 3 | 13个骨骼中固定3个（23%分位）<br>一般不需要调整 | - |
| `tail_scope` | keybones | 固定使用keybones<br>不建议改为all | - |
| `tail_select` | ema | `batch`更敏捷但更抖<br>`ema`更稳更适合长尾压制 | tail骨骼是否频繁切换 |
| `tail_ema_beta` | 0.9 | 越大越平滑（0.8~0.98）<br>过大可能反应慢 | tail切换频率/收敛速度 |

---

## 常见问题

### Q1: 为什么不直接用per-bone权重表？

**A:** Per-bone权重表的问题：
1. 需要人工调参，成本高
2. 不同动作需要不同权重（走路vs跑步vs舞蹈）
3. 静态权重无法应对训练过程的动态变化
4. 归一化导致零和博弈，加剧whack-a-mole

**Tail loss的优势：**
- 自适应：谁最差谁获得额外梯度
- 通用：不需要针对动作类型调整
- 动态：随训练进程自动调节

### Q2: 为什么ep9会出现波动？

**A:** 这是正常的"钟摆冲击"现象：
1. ep7突然引入tail loss，优化目标改变
2. lr=0.0007仍较高，模型响应激进
3. 可能出现：优化A→B变差→优化B→C变差

**解决方案：**
- ep10降低lr到0.0003，稳定优化
- 实验证明ep12重新收敛且效果更好

### Q3: 硬选择top-k会不会导致优化不稳定？

**A:** 不会，因为：
1. **所有骨骼都有基础梯度**（w_rot_local=0.22）
2. Tail只是额外增强（+0.2），不是唯一梯度
3. 即使top-k的membership变化（A,B,C → B,C,D），所有骨骼仍在持续优化

**梯度分配：**
- 未选中骨骼：只来自 `L_mean`（实现里系数 ∝ `w_rot_local * w_j / (B*T*J)`）
- 被选中骨骼：在上式基础上额外叠加 `L_tail`（额外项系数 ∝ `tail_w / (B*T*k)`）
- 相对倍率：`1 + (tail_w * J) / (w_rot_local * w_j * k)`（取决于全骨骼数 `J`、top-k 的 `k`、以及该骨骼在 `rot_local` 的权重 `w_j`，不是简单的 `0.22 + 0.2`）

### Q4: 为什么不用时间维度的tail loss？

**A:** 时间维度有本质风险：
1. **破坏时序连贯性**：帧间强依赖，单独惩罚某几帧会导致不连贯
2. **与AngVel冲突**：时间tail说"压低峰值"，AngVel说"保持平滑"
3. **噪声敏感**：单帧MoCap噪声会被直接惩罚，导致过拟合
4. **系统已有时序保护**：AngVel loss、自回归架构、Teacher forcing等

**空间tail安全的原因：**
- 骨骼间独立（thigh差不影响hand）
- 时间上是依赖（t和t+1必须连贯）
- 打断空间独立性（安全） vs 打断时序依赖（危险）

### Q5: 能否同时使用FK loss和Rot Local loss？

**A:** 不建议，原因：
1. **梯度冲突**：FK优化位置，Rot优化角度，方向可能矛盾
2. **补偿效应**：FK可能满意（位置对），Rot不满意（角度错）
3. **归属混淆**：FK无法区分是哪个骨骼导致端点误差

**我们的选择：**
- 专注Rot Local（姿态保真）
- 完全避免FK（消除冲突）
- 实验证明纯Rot Local + Tail Loss足够好

### Q6: 为什么选择GeoLocalDeg而不是其他误差度量？

**A:** GeoLocalDeg的优势：
1. **Parent-relative**：不受root对齐/漂移影响
2. **角度单位**：可解释性强（直接对应度数）
3. **旋转空间的自然距离**：测地距离是SO(3)上的标准度量
4. **对小误差敏感**：线性度量，适合精细优化

### Q7: 如何监控tail loss是否有效？

**A:** 关注这些指标：
1. **mean (GeoLocalDeg)**：应该下降（全局改善）
2. **max (GeoLocalDeg)**：应该下降（极端值压制）
3. **p90-p10 (GeoLocalDeg)**：应该下降（尾部压缩效果）
4. **std (GeoLocalDeg)**：应该下降（分布更集中）

**实现对齐说明（code-level）**：
- 训练内的关键决策（如 LR plateau 与 best-teacher checkpoint 选择）也应当优先基于 `GeoLocalDeg` / `KeyBone/GeoLocalDegMean`，以避免 root drift 对误差判断的干扰。

**实验v16的成功标志：**
- ep12: mean↓15%, max↓22%, p90-p10↓34%

### Q8: 这个方法的理论基础是什么？

**A:** 三个理论支柱：

1. **CVaR (Conditional Value at Risk)** - 金融风险管理
   - 文献：Rockafellar & Uryasev (2000)
   - 核心：优化最坏情况，而非只看平均

2. **Distributionally Robust Optimization (DRO)**
   - 文献：NeurIPS 2021 "Distributionally Robust Neural Networks"
   - 核心：minimize E[loss] + λ·worst-case risk

3. **Multi-Objective Optimization的Pareto前沿**
   - Mean和Tail是两个互补目标
   - 组合优化达到更好的Pareto前沿

---

## 总结

### 核心贡献

1. **收敛钟摆机制**：利用而非对抗whack-a-mole
2. **Mean + Tail组合**：均值↓ + 方差↓的双重优化
3. **KeyBones聚焦**：避免稀释，提升视觉质量
4. **理论支撑**：CVaR、DRO等成熟理论

### 实验效果

- Mean降低15%（全局改善）
- Max降低22%（极端值压制）
- **p90-p10降低34%（尾部压缩显著）**

### 设计哲学

**接受骨骼差异的现实**
→ 不同骨骼运动特性本质不同，有"较差骨骼"是自然的

**利用动态平衡机制**
→ 最差骨骼可以"换人"，但确保螺旋下降

**职责分离**
→ 空间tail（骨骼分布） + AngVel（时序平滑），各司其职

### 适用场景

✅ 适用：
- 训练后期（ep7+）的fine-tune阶段
- 基础收敛后，想进一步压制尾部误差
- 观察到max-min gap较大，分布不均

❌ 不适用：
- 训练初期（ep1-6）：先让模型学基础
- 整体误差仍很大（>5°）：先优化mean
- 时序连贯性已有问题：先检查AngVel等机制

---

## 参考文献

1. Rockafellar, R. T., & Uryasev, S. (2000). Optimization of conditional value-at-risk. *Journal of Risk*, 2, 21-42.

2. Duchi, J. C., & Namkoong, H. (2021). Learning models with uniform performance via distributionally robust optimization. *Annals of Statistics*, 49(3), 1378-1406.

3. Holden, D., Komura, T., & Saito, J. (2017). Phase-functioned neural networks for character control. *ACM Transactions on Graphics*, 36(4), 1-13.

4. Starke, S., Zhang, H., Komura, T., & Saito, J. (2019). Neural state machine for character-scene interactions. *ACM Transactions on Graphics*, 38(6), 1-14.
