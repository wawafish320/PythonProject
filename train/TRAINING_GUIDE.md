# 运动生成模型训练指南

## 目录

- [项目概述](#项目概述)
- [环境配置](#环境配置)
- [数据准备](#数据准备)
- [配置说明](#配置说明)
- [训练设计哲学与注意事项](#训练设计哲学与注意事项)
- [训练流程](#训练流程)
- [高级功能](#高级功能)
- [模型评估](#模型评估)
- [常见问题](#常见问题)

---

## 项目概述

本项目是一个基于深度学习的**角色运动生成与预测系统**，主要用于生成高质量的骨骼动画序列。系统支持：

- **6D旋转表示**：避免欧拉角奇异性问题
- **软脚接触预测**：连续的接触概率而非二值标签
- **分阶段训练**：从Teacher Forcing逐步过渡到自由运行
- **自适应优化**：支持贝叶斯超参优化和自适应损失权重
- **多任务学习**：同时优化位置、旋转、速度等多个目标

> **2025-11-16 更新**：原先用于“前瞻”约束的 lookahead loss 已彻底移除，阶段调度仅围绕 freerun 及相关超参展开。下文的示例和配置都已改为 `freerun_*` 命名。

### 核心模型架构

```
MotionEncoder (无状态MLP)
  ↓
[B, T, 512] 隐藏状态
  ↓
FiLM 条件调制
  ↓
多任务预测头 (周期、步态、姿态)
```

---

## 环境配置

### 系统要求

- Python 3.8+
- PyTorch 1.10+
- CUDA 11.x (推荐使用GPU)
- 8GB+ GPU显存 (对于较大模型)

### 安装依赖

```bash
pip install torch torchvision torchaudio
pip install numpy scipy
pip install tqdm
pip install optuna  # 用于贝叶斯优化
```

### 项目结构

```
PythonProject/
├── config/                 # 配置文件
│   ├── exp_phase_mpl.json # 主训练配置
│   ├── FullHierarchy.json # 骨骼层级
│   └── FullOffsets.json   # 骨骼偏移量
├── train/                  # 训练代码
│   ├── training_MPL.py    # 主训练脚本
│   ├── pretrain_mpl_min.py # 预训练脚本
│   ├── train_configurator.py # 配置管理器
│   └── configuration/      # 配置工具模块
├── raw_data/              # 原始数据目录
├── models/                # 模型输出目录
└── docs/                  # 文档目录
```

---

## 数据准备

### 数据格式

训练数据使用JSON格式，包含骨骼运动的完整信息：

```json
{
  "FPS": 60,
  "Frames": [
    {
      "RootYaw": 0.0,
      "RootVelocityXY": [0.5, 0.0],
      "BoneRotations": [
        [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],  // 6D旋转表示
        ...
      ],
      "FootEvidence": {
        "L": {"soft_contact_score": 0.8},
        "R": {"soft_contact_score": 0.2}
      }
    }
  ]
}
```

### 数据转换

将JSON格式转换为NPZ格式（训练使用）：

```bash
cd train
python convert_json_to_npz.py \
  --input ../raw_data/Walk_F.json \
  --output ../raw_data/processed_data/Walk_F.npz
```

转换后的NPZ文件包含：
- `rot6d`: 骨骼6D旋转 [T, N, 6]
- `angvel`: 角速度 [T, N, 3]
- `root_yaw`: 根节点朝向 [T]
- `root_vel`: 根节点速度 [T, 2]
- `soft_contacts`: 脚接触分数 [T, 2]

### 数据集分析

在训练前，建议先分析数据集统计信息：

```bash
cd train
python train_configurator.py \
  --base-config ../config/exp_phase_mpl.json \
  --output ../config/exp_phase_mpl.json \
  --profile
```

这会生成数据集的统计信息并保存到配置文件中：
- 片段数量 (n_clips)
- 总帧数 (total_frames)
- 平均序列长度 (avg_seq_len)
- 平均朝向角度 (yaw_mean_deg)
- 平均速度 (speed_mean)
- 复杂度评分 (complexity)

---

## 配置说明

### 主配置文件：`config/exp_phase_mpl.json`

```json
{
  // 数据和输出路径
  "data": "./raw_data/processed_data",
  "bundle_json": "./raw_data/processed_data/norm_template.json",
  "out": "./models/exp_phase_e2e_sc",

  // 基础训练参数
  "epochs": 30,
  "batch": 8,
  "lr": 0.00031732900585132803,
  "seq_len": 60,

  // Teacher Forcing 配置
  "tf_mode": "epoch_linear",      // 模式: epoch_linear, step_linear, constant
  "tf_start_epoch": 1,            // 开始衰减的epoch
  "tf_end_epoch": 19,             // 结束衰减的epoch
  "tf_max": 1.0,                  // 最大teacher forcing比例
  "tf_min": 0.0,                  // 最小teacher forcing比例

  // 损失权重
  "w_fk_pos": 0.07,               // FK位置损失权重
  "w_rot_local": 0.07,            // 局部旋转损失权重
  "w_cond_yaw": 0,                // 已弃用：cond yaw 损失关闭
  "w_rot_log": 0.2,               // 旋转增量对数域损失
  "w_limb_geo": 0.0,              // 四肢辅助损失
  "w_latent_consistency": 0.03,   // 潜在一致性损失权重

  // 分阶段训练配置
  "freerun_stage_schedule": [
    {
      "range": [1, 9],
      "label": "stage1_teacher",
      "trainer": {
        "freerun_weight": 0.0,
        "freerun_horizon": 8,
        "w_latent_consistency": 0.03
      },
      "loss_groups": {
        "core": {"w_fk_pos": 0.07, "w_rot_local": 0.07},
        "aux": {"w_limb_geo": 0.0}
      }
    },
    {
      "range": [10, 21],
      "label": "stage2_mixed",
      "trainer": {
        "freerun_weight": 0.175,
        "freerun_horizon": 14,
        "w_latent_consistency": 0.12
      },
      "loss_groups": {
        "core": {"w_fk_pos": 0.2275, "w_rot_local": 0.2275},
        "aux": {"w_limb_geo": 0.05}
      }
    },
    {
      "range": [22, 30],
      "label": "stage3_freerun",
      "trainer": {
        "freerun_weight": 0.325,
        "freerun_horizon": 18,
        "w_latent_consistency": 0.288
      },
      "loss_groups": {
        "core": {"w_fk_pos": 0.4025, "w_rot_local": 0.4025},
        "aux": {"w_limb_geo": 0.08}
      }
    }
  ]
}
```

### 配置参数说明

#### 基础参数

| 参数 | 说明 | 推荐值 |
|------|------|--------|
| `epochs` | 训练轮数 | 30-50 |
| `batch` | 批大小 | 4-16 (取决于GPU显存) |
| `lr` | 学习率 | 1e-4 ~ 5e-4 |
| `seq_len` | 序列长度（帧） | 60-120 |

#### Teacher Forcing 参数

Teacher Forcing是一种训练策略，在训练初期使用真实数据作为输入，后期逐步使用模型自身的预测：

- **`tf_mode`**: 衰减模式
  - `epoch_linear`: 按epoch线性衰减
  - `step_linear`: 按训练步线性衰减
  - `constant`: 保持恒定

- **`tf_start_epoch` / `tf_end_epoch`**: 衰减的起止epoch
- **`tf_max` / `tf_min`**: Teacher forcing比例的最大值和最小值

#### 分阶段训练

训练分为3个阶段，每个阶段有不同的训练策略：

**Stage 1: Teacher Forcing (Epoch 1-9)**
- 主要使用真实数据监督
- 仅保留极短 horizon（8 帧）用于 freerun 状态同步
- `freerun_weight=0`，即不计入自由运行损失

**Stage 2: Mixed (Epoch 10-21)**
- 混合使用真实数据和模型预测
- 开始启用 freerun 损失（horizon≈14 帧，对应自由滚动 ~0.23 秒）
- `freerun_weight=0.175`

**Stage 3: Free Run (Epoch 22-30)**
- 主要使用模型自身预测
- 进一步拉长 freerun horizon（≈18 帧）并提高权重
- `freerun_weight=0.325`

---

## 训练设计哲学与注意事项

### 核心原则：泛化性优先

本项目的训练设计遵循一个核心原则：**优先考虑模型的泛化能力，而非过度拟合特定数据集**。这意味着我们要让模型学习运动的本质规律（动力学、物理约束、姿态协调），而不是依赖过多的显式约束和先验知识。

> **设计哲学**：如果模型的动力学学习得足够好，以及姿态学习得足够准确，那么很多显式的约束实际上是不必要的，甚至会适得其反。

---

### 1. 软周期 vs 显式周期

#### 为什么使用软周期？

**本项目采用的方法**：从软脚接触（soft foot contacts）中提取软周期（soft period），而不是使用显式的周期标注。

**设计考虑**：

| 方法 | 优点 | 缺点 | 泛化性 |
|------|------|------|--------|
| **软周期（本项目采用）** | 从数据中隐式学习周期模式<br>自适应不同运动风格<br>对不规则步态更鲁棒 | 需要高质量的脚接触标注<br>训练初期可能不稳定 | ⭐⭐⭐⭐⭐ |
| 显式周期 | 训练稳定<br>收敛快 | 过度依赖人工标注<br>泛化到新动作困难<br>限制模型创造力 | ⭐⭐ |

**实现方式**：
```python
# 从软接触提取周期特征，而不是直接使用周期标签
soft_contacts = data['soft_contacts']  # [T, 2] 连续值 [0, 1]
period_signal = extract_soft_period(soft_contacts)  # 从接触模式中推断

# 而不是：
# period = data['explicit_period']  # 显式标注的周期值
```

**优势说明**：
1. **自适应性**：软周期可以自动适应走、跑、跳等不同步态的节奏变化
2. **鲁棒性**：对于不规则或过渡性运动（如转向、加速），软周期更加灵活
3. **泛化能力**：模型学习到的是"如何从接触模式推断周期"，而不是记忆固定的周期值

---

### 2. 显式约束的权衡

#### 脚接触约束（Foot Contact Constraints）

**⚠️ 重要注意事项**：谨慎使用显式的脚接触约束（如IK固定、硬约束等）。

**为什么要谨慎？**

虽然添加显式的foot约束可以快速改善训练初期的脚滑动问题，但会带来以下负面影响：

1. **降低泛化性**
   - 模型过度依赖约束，而不是学习正确的动力学
   - 在约束之外的场景（如不平整地面、斜坡）表现变差
   - 限制了模型对新动作类型的适应能力

2. **损失权重竞争**
   ```python
   # 多个损失项会互相"抢夺"梯度权重
   total_loss = (
       w_fk_pos * loss_fk_pos +           # 位置损失
       w_rot_local * loss_rot_local +     # 旋转损失
       w_rot_geo * loss_rot_geo +         # 测地线约束
       w_rot_delta * loss_rot_delta +     # 增量平滑
       # cond_yaw 已移除
       w_latent_consistency * loss_latent_consistency +  # 编码器一致性
       freerun_weight * loss_freerun
   )
   # 问题：过多的foot相关损失会压制其他重要损失的学习
   ```

3. **优化困难**
   - 损失项之间可能存在冲突（例如：旋转损失希望优化姿态，但foot约束要求固定位置）
   - 需要仔细调整多个权重，增加超参数调优的复杂度
   - 自适应损失平衡算法（GradNorm等）也难以完美处理过多的约束

4. **物理不一致性**
   - 显式约束可能与物理规律冲突
   - 例如：强制固定脚部位置可能导致不自然的关节角度或扭曲

**推荐做法**：

```python
# ✅ 好的做法：让模型学习软接触和物理一致的运动
loss = (
    w_fk_pos * loss_fk_pos +              # 前向运动学位置
    w_rot_local * loss_rot_local +        # 局部旋转
    w_rot_geo * loss_rot_geo +            # 测地线误差
    w_rot_delta * loss_rot_delta +        # 平滑性
    # cond_yaw 已移除
    w_latent_consistency * loss_latent_consistency +  # 潜在一致性
    freerun_weight * loss_freerun         # 自由运行稳定性
)

# ❌ 避免的做法：在 Trainer 中额外堆叠大量足部硬约束（以下名称仅作说明）
loss = (
    ... +
    w_foot_fixed * loss_foot_fixed +      # 强制固定脚部
    w_foot_penetration * loss_penetration + # 防止穿透
    w_foot_slide * loss_foot_slide +      # 防止滑动
    w_foot_velocity * loss_foot_velocity  # 接触时速度为0
    # 太多约束！模型失去了学习动力学的机会
)
```
> 以上 `w_foot_*` 名称只是示例，当前代码库并没有这些参数；只有在你自行扩展 Trainer 时才可能出现。

---

### 3. 何时可以使用显式约束？

虽然我们强调泛化性，但在某些情况下，适度的显式约束是合理的：

#### ✅ 可以使用显式约束的场景

1. **训练早期作为辅助**
   ```python
   # 如果你在自定义分支里实现了额外的足部约束，限定其作用周期
   extra_foot_constraint = (epoch < 5)
   ```

2. **数据质量差**
   - 如果训练数据本身存在严重的脚滑动或不物理的运动
   - 可以用约束作为"数据清洗"的手段

3. **特定应用场景**
   - 如果最终应用只需要在平地上行走（非常受限的场景）
   - 可以适度使用约束来优化这个特定场景的表现

4. **物理硬约束**
   - 明确的物理规律（如重力方向、地面穿透）
   - 这类约束不会降低泛化性，反而有助于物理一致性

#### ❌ 不应该使用显式约束的场景

1. **模型已经收敛良好**
   - 如果旋转、位置、速度等指标已经达标
   - 不要为了微小提升而添加约束

2. **需要泛化到多种场景**
   - 如果希望模型能处理各种地形、步态、过渡动作
   - 显式约束会严重限制泛化能力

3. **数据集多样性高**
   - 如果训练数据包含多种运动类型
   - 约束可能只对部分数据有效，反而增加冲突

---

### 4. 本项目的损失设计：层次化与分组

#### 实际损失结构

本项目采用**层次化的损失设计**，针对运动生成任务的多个方面进行优化：

```python
# 完整的损失结构 (training_MPL.py:2436-2568)
total_loss = (
    # === 核心运动损失 (MotionJointLoss) ===
    # Core组 (直接影响运动质量)
    w_rot_geo * loss_rot_geo +          # 旋转测地线误差
    w_rot_delta * loss_rot_delta +      # 旋转增量损失
    w_rot_ortho * loss_rot_ortho +      # 旋转正交性
    # cond_yaw 已移除
    w_fk_pos * loss_fk_pos +            # FK位置误差
    w_rot_local * loss_rot_local +      # 父子相对旋转

    # Aux组 (辅助优化)
    w_attn_reg * loss_attn +            # 注意力正则化
    w_rot_delta_root * loss_rot_delta_root +  # 根节点增量
    w_rot_log * loss_rot_log +          # SO(3)对数空间损失
    w_limb_geo * loss_limb_geo +        # 四肢旋转hinge损失

    # === 训练策略损失 ===
    w_latent * loss_latent_consistency + # 编码器一致性
    w_freerun * loss_freerun            # 自由运行损失
)
```

**为什么需要这么多损失项？**

运动生成不是简单的回归问题，需要同时满足多个约束：
1. **几何约束**：旋转矩阵正交性 (rot_ortho)、父子关节关系 (rot_local)
2. **物理约束**：FK位置一致性 (fk_pos)、朝向控制 (cond_yaw)
3. **平滑性约束**：旋转增量 (rot_delta)、四肢平滑 (limb_geo)
4. **一致性约束**：编码器特征 (latent_consistency)、注意力 (attn_reg)
5. **策略约束**：自由运行稳定性 (freerun)

**每个损失项都有明确的职责**，移除任何一项都会导致某方面的性能下降。

#### 损失分组逻辑

本项目将损失分为三组 (training_MPL.py:1700-1711)：

| 组别 | 损失项 | 作用 | 权重范围 |
|------|--------|------|----------|
| **core** | rot_geo, rot_delta, rot_ortho<br>fk_pos, rot_local | 直接决定运动质量 | 0.01 ~ 0.2 |
| **aux** | attn, rot_delta_root<br>rot_log, limb_geo | 辅助优化、正则化 | 0.001 ~ 0.05 |
| **long** | latent_consistency<br>freerun | 长期稳定性、泛化性 | 0.01 ~ 0.3 |

**这种分组的意义**：
- **Core损失**：调优的重点，直接影响输出质量
- **Aux损失**：起辅助作用，权重较小，保持稳定
- **Long损失**：影响泛化能力，按训练阶段动态调整

#### 权重平衡策略

**1. 不要盲目减少损失项数量**

虽然损失项多达10+个，但这是运动生成任务的必然选择：
- ❌ 错误做法：为了"简化"而移除某些损失
- ✅ 正确做法：保持完整的损失结构，通过**分组管理**和**权重分配**来平衡

**2. 采用分层调优策略**

```python
# 第一优先级：Core组 (占总权重的60-70%)
w_fk_pos = 0.07        # 位置准确性
w_rot_local = 0.07     # 旋转准确性
w_rot_geo = 0.01       # 测地线距离
w_rot_delta = 1.0      # 增量平滑性
w_rot_ortho = 0.001    # 正交性约束

# 第二优先级：Long组 (占总权重的20-30%)
w_latent_consistency = 0.03  # 编码器一致性
freerun_weight = 0.0 → 0.325  # 逐步增加

# 第三优先级：Aux组 (占总权重的<10%)
w_attn_reg = 0.01      # 注意力正则
w_limb_geo = 0.0       # 四肢辅助（可选）
```

**3. 监控分组贡献**

训练脚本自动计算每组的总贡献 (training_MPL.py:2577-2601)：

```python
# 训练日志中会输出
stats = {
    'loss_group/core': 0.125,    # Core组总贡献
    'loss_group/aux': 0.015,     # Aux组总贡献
    'loss_group/long': 0.08,     # Long组总贡献
}

# 理想的组间比例
core : aux : long ≈ 65% : 5% : 30%
```

**如果比例失衡**：
- Core组占比过高（>80%）→ 增加Long组权重，提高泛化性
- Aux组占比过高（>15%）→ 减小Aux权重，避免过拟合正则化
- Long组占比过低（<15%）→ 增加freerun权重

---

### 5. 实战建议：渐进式训练策略

#### 核心原则：从Teacher Forcing到Free Run的平滑过渡

训练应该**渐进式地**从完全监督（teacher forcing）过渡到自由运行（free run）。阶段的数量和划分可以灵活调整，关键是保证权重的**平滑变化**。

**本项目当前采用3阶段配置**（仅供参考，可根据需求调整）：

#### 示例阶段1：训练初期（Epoch 1-9, Teacher Forcing主导）

- Trainer：`freerun_weight=0`、`freerun_horizon=8`、`w_latent_consistency=0.03`。
- Loss 组：`core` 以 `w_fk_pos=0.07`、`w_rot_local=0.07` 为主，`aux` 中 `w_limb_geo` 关闭；全局 `w_cond_yaw=0.1` 不变。
- 目的：让模型通过高 teacher forcing 比例稳定学习基本姿态，仅做最小化的 freerun 状态同步。

#### 示例阶段2：训练中期（Epoch 10-21, Mixed模式）

- Trainer：`freerun_weight=0.175`、`freerun_horizon=14`，`w_latent_consistency` 提升到 0.12。
- Loss 组：`core` 动态放大至 `w_fk_pos=0.2275`、`w_rot_local=0.2275`，`aux` 打开 `w_limb_geo=0.05`，让四肢姿态更平滑。
- 目的：在 teacher/free 混合模式下训练，逐渐降低 teacher forcing（`tf_max` 变为 0.75），同时让 freerun 损失开始发挥作用。

#### 示例阶段3：训练后期（Epoch 22-30, Free Run主导）

- Trainer：`freerun_weight=0.325`、`freerun_horizon=18`，并将 `w_latent_consistency` 增加到 0.288。
- Loss 组：`core` 提升至 `w_fk_pos=0.4025`、`w_rot_local=0.4025`，`aux` 中 `w_limb_geo=0.08` 维持稳定正则；teacher forcing 最终衰减到 0。
- 目的：完全聚焦自由运行表现，确保长序列稳定性成为主要训练信号。

**关键提示**
- 不同阶段的 `loss_groups` 只会覆盖显式声明的权重，其他如 `w_rot_log=0.2` 仍按全局配置执行（`w_cond_yaw` 已弃用）。
- 如果需要关闭某个损失（例如探索无 `cond_yaw` 的 ablation），必须在相应阶段的 `loss_groups` 中显式写入该权重，而不是假设脚本会自动切换。

---

#### 关于阶段划分的灵活性

**重要说明**：上述3阶段配置只是一个示例，实际使用时完全可以根据需求调整：

**更细粒度的划分（例如5阶段）**：
```json
{
  "freerun_stage_schedule": [
    {"range": [1, 5], "label": "warmup"},           // 热身阶段
    {"range": [6, 12], "label": "early_mixed"},     // 早期混合
    {"range": [13, 20], "label": "mid_mixed"},      // 中期混合
    {"range": [21, 27], "label": "late_mixed"},     // 后期混合
    {"range": [28, 35], "label": "pure_freerun"}    // 纯自由运行
  ]
}
// 优势：更平滑的权重过渡，适合复杂任务
```

**更粗粒度的划分（例如2阶段）**：
```json
{
  "freerun_stage_schedule": [
    {"range": [1, 15], "label": "supervised"},      // 监督学习
    {"range": [16, 30], "label": "self_supervised"} // 自监督学习
  ]
}
// 优势：配置简单，适合快速实验
```

**阶段数量的选择原则**：
1. **数据集规模小**（<1000帧）：建议2-3个阶段，避免过拟合
2. **数据集规模中等**（1000-10000帧）：建议3-4个阶段（当前配置）
3. **数据集规模大**（>10000帧）：可以使用5+个阶段，更精细调控
4. **任务复杂度高**（多种动作类型）：增加阶段数，逐步提升难度
5. **快速迭代实验**：使用较少阶段（2-3个），加快调试速度

**核心不变的原则**：
- ✅ **渐进式过渡**：freerun 权重应该单调递增
- ✅ **平滑变化**：相邻阶段的权重变化不宜过大（建议<0.15）
- ✅ **后期强化freerun**：最终阶段的freerun权重应该>0.3
- ❌ **避免突变**：不要在某个阶段突然大幅调整权重
- ❌ **避免回退**：不要在后期阶段降低freerun权重（会降低泛化性）

**阶段数量不影响泛化性**，关键是**渐进式策略**的正确执行。

---

### 6. 检查清单：评估训练设计质量

在开始训练或调整配置前，检查以下关键点：

#### 损失结构检查

- [ ] **损失分组**：是否明确Core/Aux/Long三组的职责？
- [ ] **组间平衡**：Core组占比是否在60-70%，Aux组<10%，Long组20-30%？
- [ ] **关键损失**：`w_fk_pos`、`w_rot_local`、`w_rot_delta`是否已配置？
- [ ] **显式约束**：是否添加了foot固定、IK约束、高度约束等？❌ 这些都不应该存在！

#### 泛化性检查

- [ ] **软周期**：是否从软接触中提取周期信息，而非显式标注？
- [ ] **Freerun权重**：后期freerun权重是否达到0.3以上？
- [ ] **Teacher Forcing衰减**：是否有明确的衰减计划（1.0→0.5→0.0）？
- [ ] **监控指标**：是否同时监控Teacher和Free-Run模式？两者差距是否<2x？

#### 训练策略检查

- [ ] **分阶段训练**：是否定义了明确的训练阶段（渐进式从Teacher到FreeRun）？
- [ ] **权重渐进**：freerun 权重是否单调递增、平滑变化？
- [ ] **阶段合理性**：阶段数量是否与数据集规模和任务复杂度匹配？
- [ ] **数据多样性**：训练数据是否包含多种运动模式（走、跑、转向等）？
- [ ] **超参优化**：是否使用了贝叶斯优化或自适应调整（可选但推荐）？

#### 避免的错误模式

**❌ 错误示例1：过度依赖显式约束**
```json
{
  "w_foot_fixed": 0.1,        // ❌ 强制固定脚部
  "w_foot_height": 0.05,      // ❌ 高度约束
  "w_foot_slide": 0.08,       // ❌ 防滑动约束
  "w_foot_velocity": 0.03     // ❌ 速度约束
}
// 问题：4个foot约束权重总和0.26，过度压制了模型学习动力学
```
> 同样地，`w_foot_*` 只是说明性的变量，当前实现里没有暴露这些开关。

**❌ 错误示例2：忽略Long组权重**
```json
{
  "freerun_weight": 0.05,     // ❌ 太低！应该>0.3
  "freerun_horizon": 6        // ❌ 全程只有短窗口，无法学习长期依赖
}
// 问题：模型只学会拟合训练集，泛化能力差
```

**❌ 错误示例3：Core组权重失衡**
```json
{
  "w_fk_pos": 0.5,            // ❌ 过高！碾压其他损失
  "w_rot_local": 0.01,        // ❌ 过低！旋转学不好
  "w_rot_delta": 0.001        // ❌ 过低！运动不平滑
}
// 问题：位置准确但姿态和平滑性很差
```

#### ✅ 正确的训练设计

**本项目的推荐配置**（参见config/exp_phase_mpl.json）：
```
✅ 10+个损失项，但通过分组(core/aux/long)清晰管理
✅ 使用软周期（从软接触提取）而非显式周期标注
✅ 零显式foot约束，完全依赖动力学学习
✅ 渐进式多阶段训练（当前采用3阶段，可根据需求调整）
✅ Freerun权重从0.0逐步增加到0.325（单调递增）
✅ 重点监控Free-Run vs Teacher的差距
✅ Teacher Forcing从1.0衰减到0.0（平滑衰减）
```

**如果遇到问题**：
1. **Teacher模式好，Free-Run差**：增加freerun_weight（必要时同时拉长 freerun_horizon）
2. **整体损失不降**：检查Core组权重平衡，尝试贝叶斯优化
3. **运动不平滑**：检查`w_rot_delta`是否足够高（推荐1.0）
4. **脚滑动严重**：❌ 不要添加foot约束！→ ✅ 增加训练时间，检查soft_contacts标注质量
5. **四肢姿态差**：可以启用`w_limb_geo`（权重<0.05），但优先检查`w_rot_local`

---

### 7. 总结：泛化性 > 训练集性能

**核心信念**：

> 一个在训练集上表现完美但泛化能力差的模型，远不如一个训练集上表现良好且泛化能力强的模型。

**实践中的体现**：

1. **宁可让训练损失略高，也不要过度约束**
   - 训练损失 = 0.015（无过多约束）> 0.008（大量约束）

2. **Free-Run模式是最重要的评估指标**
   - 如果Teacher模式很好但Free-Run差距大 → 模型在作弊
   - 如果两者差距小 → 模型真正学会了运动规律

3. **少即是多**
   - 更少的损失项 + 更多的训练时间 > 更多的约束 + 更快的收敛

4. **相信模型的学习能力**
   - 给模型足够的时间和数据，它能学会正确的动力学
   - 不要因为训练初期表现不好就急于添加约束

---

## 训练流程

### 1. 基础训练

最简单的训练方式：

```bash
cd train
python training_MPL.py \
  --config_json ../config/exp_phase_mpl.json
```

训练过程中会输出：
```
Epoch 1/30
  Train Loss: 0.0125 | Val Loss: 0.0142
  GeoDeg: 2.34 | YawAbsDeg: 1.87 | RootVelMAE: 0.012
Saving checkpoint to models/exp_phase_e2e_sc/checkpoint_epoch_1.pt
```

### 2. 从检查点恢复训练

```bash
python training_MPL.py \
  --config_json ../config/exp_phase_mpl.json \
  --resume ../models/exp_phase_e2e_sc/checkpoint_epoch_10.pt
```

### 3. 预训练运动编码器

在主训练之前，可以先预训练运动编码器：

```bash
python pretrain_mpl_min.py \
  --data ../raw_data/processed_data \
  --out ../models/motion_encoder_pretrained \
  --epochs 50 \
  --batch 16 \
  --lr 0.0005
```

预训练的编码器可以在主训练中加载：

```bash
python training_MPL.py \
  --config_json ../config/exp_phase_mpl.json \
  --encoder ../models/motion_encoder_pretrained/best_encoder.pt
```

### 4. 监控训练进度

训练过程中会保存以下信息：

**模型检查点**
```
models/exp_phase_e2e_sc/
├── checkpoint_epoch_1.pt
├── checkpoint_epoch_5.pt
├── ...
└── best_model.pt
```

**训练指标**
```
models/exp_phase_e2e_sc/exp_phase_AllDebug/metrics/
├── teacher/
│   ├── GeoDeg.txt
│   ├── YawAbsDeg.txt
│   └── RootVelMAE.txt
└── valfree/
    ├── GeoDeg.txt
    └── ...
```

---

## 高级功能

### 1. 贝叶斯超参数优化

使用贝叶斯优化自动搜索最佳超参数：

```bash
cd train
python train_configurator.py \
  --base-config ../config/exp_phase_mpl.json \
  --output ../config/exp_phase_mpl.json \
  --bayes-opt \
  --trials 50
```

优化的参数包括：
- 学习率 (lr)
- 损失权重 (w_fk_pos, w_rot_local等)
- 训练策略 (freerun_weight、freerun_horizon)
- Teacher forcing参数

优化历史保存在 `train/bayes_history.json`

### 2. 指标驱动的配置调整

根据验证集指标自动调整训练参数：

```bash
cd train
python train_configurator.py \
  --base-config ../config/exp_phase_mpl.json \
  --output ../config/exp_phase_mpl.json \
  --metrics-root ../models/exp_phase_e2e_sc/exp_phase_AllDebug/metrics \
  --tune
```

调整逻辑：
- `YawAbsDeg` 偏高 → 自动提高 `freerun_weight`、`freerun_horizon`
- `RootVelMAE` 偏高 → 增加 `freerun_weight`
- `InputRotGeoDeg` 偏高 → 升高 `w_latent_consistency`、`w_fk_pos`、`w_rot_local`

### 3. 自适应损失权重

通过 CLI 即可启用自适应损失模块：

```bash
python training_MPL.py \
  --config_json ../config/exp_phase_mpl.json \
  --adaptive_loss_method gradnorm \
  --adaptive_loss_terms fk_pos,rot_local,rot_delta \
  --adaptive_loss_alpha 1.5 \
  --adaptive_loss_temperature 2.0
```

支持的方法：
- **GradNorm**: 基于梯度范数平衡损失
- **Uncertainty**: 基于不确定性学习权重
- **DWA**: 动态权重平均

Trainer 会在内部调用 `build_adaptive_loss` 并把统计写入 `adaptive_loss/*` 指标，无需修改源码。

### 4. 自定义训练阶段

编辑配置文件中的 `freerun_stage_schedule` 来自定义训练阶段：

```json
{
  "freerun_stage_schedule": [
    {
      "range": [1, 10],
      "label": "warmup",
      "trainer": {
        "freerun_weight": 0.0,
        "freerun_horizon": 8
      }
    },
    {
      "range": [11, 30],
      "label": "main_training",
      "trainer": {
        "freerun_weight": 0.3,
        "freerun_horizon": 20
      }
    }
  ]
}
```

---

## 模型评估

### 记录的指标

训练脚本会在 `models/<run_name>/exp_phase_AllDebug/metrics/teacher_epXXX.json` 与 `valfree_epXXX.json` 中写入：

- `GeoDeg`、`YawAbsDeg`、`RootVelMAE`：主指标，Teacher 和 Free-Run 各一份。
- `loss_group/<core|aux|long>`：Loss 贡献比例，用于诊断权重平衡。
- `FootContact`：基于预测角速度的接触比分布（不是分类准确率）。
- `KeyBone/*`：针对躯干/四肢的角速度对齐与局部测地线统计。
- `adaptive_loss/*`：若启用自适应损失，会输出各项权重与基准 loss。

指标目录下的 `*.txt` 会记录同名指标的历史曲线，可直接画图或导入 TensorBoard。

### Teacher vs Free-Run

Trainer 在每个 epoch 末尾分别调用 `evaluate_teacher()` 与 `evaluate_freerun()`：

- Teacher：Teacher forcing 推理，衡量有监督条件下的单步误差。
- Free-Run (`valfree`): 自由运行滚动窗口，观察误差积累情况，重点参看 `FreeRun/GeoDeg` 与 `FreeRun/RootVelMAE`。

### 如何离线复用评估函数

`train/eval_utils.py` 只提供函数而非 CLI。如需单独评估，可在脚本中显式构建 Trainer 和数据加载器：

```python
from train import training_MPL
from train.eval_utils import evaluate_teacher, evaluate_freerun, FreeRunSettings

trainer = training_MPL.Trainer(... )  # 复用和训练时相同的模型、loss、normalizer
teacher_stats = evaluate_teacher(trainer, val_loader, mode='teacher')
free_stats = evaluate_freerun(
    trainer,
    val_loader,
    settings=FreeRunSettings(warmup_steps=4, horizon=12)
)
print(teacher_stats['GeoDeg'], free_stats['FreeRun/GeoDeg'])
```

记得在脚本中加载与训练一致的 `config_json`、normalizer bundle 以及 checkpoints，否则评估结果不可比。

---

## 常见问题

### Q1: 训练损失不下降？

**可能原因**：
1. 学习率过高或过低
2. 批大小不合适
3. 损失权重不平衡

**解决方案**：
```bash
# 尝试贝叶斯优化找到更好的超参数
python train_configurator.py \
  --base-config ../config/exp_phase_mpl.json \
  --output ../config/exp_phase_mpl.json \
  --bayes-opt \
  --trials 20
```

### Q2: Free-Run模式下误差累积严重？

**可能原因**：
1. 过早进入Free-Run阶段
2. Teacher Forcing衰减过快
3. Lookahead steps设置不当

**解决方案**：
- 延长Stage 1的训练时间
- 调整 `tf_end_epoch` 到更晚的epoch
- 在Stage 2增加更多的epoch

### Q3: 模型过拟合小数据集？

**可能原因**：
- 数据集过小（本项目只有5个片段，375帧）
- 模型容量过大

**解决方案**：
1. 增加数据增强：
   ```python
   # 在配置中添加
   "augmentation": {
     "noise_std": 0.01,
     "speed_scale_range": [0.8, 1.2]
   }
   ```

2. 使用Dropout和正则化：
   ```python
   "dropout": 0.2,
   "weight_decay": 1e-5
   ```

3. 减小模型规模：
   ```json
   "hidden_dim": 256,  // 从512减小到256
   "num_layers": 2      // 从3减小到2
   ```

### Q4: GPU显存不足？

**解决方案**：
1. 减小批大小：`"batch": 4`
2. 减小序列长度：`"seq_len": 40`
3. 使用梯度累积：
   ```python
   # 在训练脚本中添加
   accumulation_steps = 4
   ```

### Q5: 如何添加新的运动数据？

步骤：
1. 准备JSON格式的数据（参考 `raw_data/Walk_F.json`）
2. 转换为NPZ：
   ```bash
   python convert_json_to_npz.py --input new_motion.json --output processed_data/new_motion.npz
   ```
3. 重新分析数据集：
   ```bash
   python train_configurator.py \
     --base-config ../config/exp_phase_mpl.json \
     --output ../config/exp_phase_mpl.json \
     --profile
   ```
4. 开始训练

### Q6: 如何导出训练好的模型用于推理？

```python
import torch

# 加载检查点
checkpoint = torch.load('models/exp_phase_e2e_sc/best_model.pt')

# 提取模型权重
model_state = checkpoint['model_state_dict']

# 保存为单独的权重文件
torch.save(model_state, 'exported_model.pt')
```

在推理时：
```python
from train.training_MPL import EventMotionModel

model = EventMotionModel(config)
model.load_state_dict(torch.load('exported_model.pt'))
model.eval()

# 生成运动
with torch.no_grad():
    output = model(input_state, conditions)
```

---

## 训练最佳实践

### 1. 推荐的训练流程

```bash
# Step 1: 数据准备和分析
python train_configurator.py \
  --base-config config.json \
  --output config.json \
  --profile

# Step 2: 预训练编码器（可选）
python pretrain_mpl_min.py --data data --epochs 50

# Step 3: 贝叶斯超参优化（推荐）
python train_configurator.py \
  --base-config config.json \
  --output config.json \
  --bayes-opt \
  --trials 30

# Step 4: 主训练
python training_MPL.py --config_json config.json --encoder pretrained_encoder.pt

# Step 5: 根据指标调优
python train_configurator.py \
  --base-config config.json \
  --output config.json \
  --metrics-root models/exp_phase_e2e_sc/exp_phase_AllDebug/metrics \
  --tune
```

### 2. 超参数调优建议

**小数据集 (< 1000帧)**:
```json
{
  "lr": 0.0001,
  "batch": 4,
  "seq_len": 40,
  "dropout": 0.3,
  "weight_decay": 1e-4
}
```

**中等数据集 (1000-10000帧)**:
```json
{
  "lr": 0.0003,
  "batch": 8,
  "seq_len": 60,
  "dropout": 0.1,
  "weight_decay": 1e-5
}
```

**大数据集 (> 10000帧)**:
```json
{
  "lr": 0.0005,
  "batch": 16,
  "seq_len": 120,
  "dropout": 0.05,
  "weight_decay": 0
}
```

### 3. 监控关键指标

在训练过程中重点关注：

1. **收敛性**: Train Loss vs Val Loss
   - 两者都应该下降
   - 差距不应该太大（防止过拟合）

2. **旋转质量**: GeoDeg < 5度
   - 如果 > 10度，增加 `w_rot_local`

3. **位置准确性**: RootVelMAE < 0.02
   - 如果过高，增加 `w_fk_pos`

4. **Free-Run稳定性**: Valfree指标不应该远高于Teacher
   - 如果差距 > 2x，延长Stage 1训练

---

## 参考资料

### 相关论文

1. **6D旋转表示**: "On the Continuity of Rotation Representations in Neural Networks" (CVPR 2019)
2. **Teacher Forcing**: "Scheduled Sampling for Sequence Prediction with Recurrent Neural Networks" (NeurIPS 2015)
3. **自适应损失**: "GradNorm: Gradient Normalization for Adaptive Loss Balancing" (ICML 2018)

### 代码文件索引

| 功能 | 文件路径 |
|------|----------|
| 主训练脚本 | `train/training_MPL.py` |
| 预训练脚本 | `train/pretrain_mpl_min.py` |
| 配置管理 | `train/train_configurator.py` |
| 贝叶斯优化 | `train/configuration/bayes.py` |
| 阶段配置 | `train/configuration/stages.py` |
| 数据集分析 | `train/configuration/profile.py` |
| 几何运算 | `train/geometry.py` |
| 评估工具 | `train/eval_utils.py` |
| 数据加载 | `train/io.py` |

---
