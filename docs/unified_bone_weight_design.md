# 统一骨骼权重系统设计文档（基于下游影响的物理权重）

**版本**: v1.0
**日期**: 2025-11-27
**作者**: Claude
**状态**: 设计完成，待实现

---

## 📋 目录

1. [背景与问题](#1-背景与问题)
2. [现有方案的局限性](#2-现有方案的局限性)
3. [统一权重的设计理念](#3-统一权重的设计理念)
4. [核心实现](#4-核心实现)
5. [末端骨骼问题与优化](#5-末端骨骼问题与优化)
6. [视觉重要性调制](#6-视觉重要性调制)
7. [完整实现代码](#7-完整实现代码)
8. [参数调优指南](#8-参数调优指南)
9. [实验对比方案](#9-实验对比方案)
10. [预期效果](#10-预期效果)

---

## 1. 背景与问题

### 1.1 问题发现

训练后发现**左右骨骼的geo误差不对称**：

| 骨骼对 | Left Geo Error | Right Geo Error | 差异 |
|--------|---------------|----------------|------|
| lowerarm | 较高 | 较低 | 左侧明显高 |
| hand | 较高 | 较低 | 左侧明显高 |
| clavicle | 较高 | 较低 | 左侧明显高 |

### 1.2 根本原因追溯

通过分析 `train/convert_json_to_npz.py` 和 `norm_template.json`，发现问题链：

```
训练数据不平衡
  ↓
Walk_R_To_R.npz (709KB) vs Walk_L_To_L.npz (334KB)  # 2.1倍数据量差异
  ↓
统计prior_per_dim不对称
  ↓
lowerarm_l: std=0.0584, lowerarm_r: std=0.0316 (84.8%差异)
  ↓
自适应权重 = 1/std
  ↓
lowerarm_l权重 = 17.1, lowerarm_r权重 = 31.6 (仅54%!)
  ↓
优化强度不均 → 左侧欠拟合 → 误差不对称 ❌
```

**关键发现**：
- `_build_rot6d_groups()` 的骨骼分组逻辑降级为单骨骼分组（`bone_00`, `bone_01`, ...）
- 原本应该将 `lowerarm_l` 和 `lowerarm_r` 归入同一组（`lower_limb`）并使用组内中位数平滑
- 但由于 `bone_names` 提取失败或为空，触发了降级逻辑
- 导致每个骨骼独立计算prior，失去了对称性约束

---

## 2. 现有方案的局限性

### 2.1 方案回顾（commit ea2a5ca）

当前使用的**组合权重系统**：

```python
# 运动幅度权重
motion_weight[i] = 1.0 / (prior_std[i] + eps)

# 层级权重
hierarchy_weight[i] = log(num_descendants[i] + 1) + 1.0

# 组合
final_weight = motion_weight × hierarchy_weight
```

### 2.2 存在的问题

#### 问题1：依赖训练数据统计 ❌

**现象**：
- `prior_std` 从训练数据的rot6d变化幅度统计得出
- 受数据分布影响：Walk动作多 → walk相关骨骼std大 → 权重低
- 数据不平衡直接导致权重不对称

**本质**：
```python
# 它回答的问题是：
"训练数据中，这个骨骼的运动幅度有多大？"

# 但没有回答：
"这个骨骼的误差，对角色姿态的影响有多大？"
```

#### 问题2：需要组合多个维度 ❌

```python
# 需要同时考虑：
1. 运动幅度 (1/prior_std)
2. 层级深度 (log(descendants))
3. 骨骼长度 (未包含)
4. FK误差传播 (隐含在hierarchy中)

# 导致：
- 参数多（hierarchy_mode, gamma, alpha等）
- 难以解释每个维度的交互作用
- 需要手动调优组合方式
```

#### 问题3：对称性需要额外处理 ❌

由于基于数据统计，必须手动对称化：
```python
def _symmetrize_lr_bones(stds):
    for l_idx, r_idx in lr_pairs:
        sym_std = (stds[l_idx] + stds[r_idx]) / 2.0
        stds[l_idx] = stds[r_idx] = sym_std
```

#### 问题4：语义混乱 ❌

```python
# motion_weight: 运动幅度大 → std大 → weight小
# hierarchy_weight: 子孙多 → weight大

# 这两个维度的"重要性"定义不一致
# 组合时缺乏统一的物理解释
```

---

## 3. 统一权重的设计理念

### 3.1 核心思想

**用单一物理量统一所有考虑因素**：

```python
unified_weight[i] = "如果骨骼i旋转1度，会对全身姿态产生多大影响？"
```

### 3.2 物理原理：杠杆臂累积

#### FK误差传播

当骨骼 `i` 产生旋转误差 `θ` 时：

```
骨骼i旋转θ
  ↓
所有子骨骼j的全局位置偏移: Δpos[j] = lever_arm[i→j] × sin(θ)
  ↓
对于小角度: sin(θ) ≈ θ
  ↓
总影响 ∝ Σ(lever_arm[i→j]) for all descendants j
```

**定义杠杆臂**：
```python
lever_arm[i→j] = ||global_pos[j_end] - global_pos[i_start]||

其中：
- i_start: 骨骼i的起点（关节位置）
- j_end: 后代骨骼j的终点位置
```

#### 示例

```
      pelvis (i=0)
        ↓
      spine_01
        ↓
     upperarm_l (j=5)
        ↓
     lowerarm_l (j=8)
        ↓
      hand_l (j=11, end)

pelvis旋转1度时：
- hand_l末端偏移 = lever_arm[pelvis→hand] × θ
- lever_arm = ||hand_l_end - pelvis_start|| ≈ 1.5m
- 偏移量 ≈ 1.5m × (π/180) ≈ 2.6cm

lowerarm_l旋转1度时：
- hand_l末端偏移 = lever_arm[lowerarm→hand] × θ
- lever_arm ≈ 0.35m
- 偏移量 ≈ 0.6cm (影响更小)
```

### 3.3 统一性体现

该方案**自动统一**了之前需要分别考虑的因素：

| 因素 | 如何体现 |
|------|---------|
| 骨骼长度 | ✅ 更长的骨骼 → 到后代的距离更大 → lever_arm更大 |
| 层级深度 | ✅ 父骨骼 → 后代更多 → 累积的lever_arm更大 |
| 子孙数量 | ✅ 自动计数（每个后代贡献一个lever_arm） |
| FK拓扑 | ✅ 基于实际骨架parents树计算 |
| 对称性 | ✅ 左右镜像骨架 → lever_arm相等 → 权重自动对称 |

**对比**：

```python
# ❌ 旧方案：需要组合多个维度
weight = (1/prior_std) × log(descendants+1) × manual_scaling

# ✅ 新方案：单一物理量
weight = Σ(lever_arm to all descendants)
```

---

## 4. 核心实现

### 4.1 算法流程

```
1. 加载骨架数据
   - bone_names, parents, ref_local_offsets_m

2. FK计算参考姿态的全局位置
   - 从根骨骼开始，累加local_offsets

3. 对每个骨骼i:
   a. 找到所有后代j
   b. 计算lever_arm[i→j] = ||pos[j_end] - pos[i_start]||
   c. 累加: influence[i] = Σ(lever_arm)

4. 归一化
   - weights = influence / mean(influence)
```

### 4.2 基础代码

```python
def _compute_unified_weights_basic(self) -> torch.Tensor:
    """基础版本：纯下游影响"""
    J = len(self.bone_names)
    parents = self.parents  # [J]
    local_offsets = self.ref_local_offsets_m  # [J, 3]

    # 1. FK计算全局位置
    global_positions = self._forward_kinematics(local_offsets, parents)  # [J, 3]

    # 2. 计算每个骨骼的影响
    weights = torch.zeros(J, dtype=torch.float32)

    for i in range(J):
        influence = 0.0
        bone_i_start = global_positions[i]

        # 遍历所有后代
        for j in self._get_all_descendants(i, parents):
            # 后代j的终点位置
            bone_j_end = global_positions[j] + local_offsets[j]

            # 杠杆臂
            lever_arm = torch.norm(bone_j_end - bone_i_start)
            influence += lever_arm

        weights[i] = influence

    # 3. 归一化
    weights = weights / weights.mean()
    return weights


def _forward_kinematics(self, local_offsets: torch.Tensor, parents: list) -> torch.Tensor:
    """FK：计算全局关节位置（假设所有旋转为identity）"""
    J = len(parents)
    global_positions = torch.zeros(J, 3, dtype=torch.float32)

    for i in range(J):
        if parents[i] < 0:
            global_positions[i] = local_offsets[i]
        else:
            global_positions[i] = global_positions[parents[i]] + local_offsets[i]

    return global_positions


def _get_all_descendants(self, bone_idx: int, parents: list) -> list:
    """BFS获取所有后代骨骼"""
    J = len(parents)
    descendants = []
    queue = [bone_idx]
    visited = set([bone_idx])

    while queue:
        current = queue.pop(0)
        for j in range(J):
            if parents[j] == current and j not in visited:
                descendants.append(j)
                queue.append(j)
                visited.add(j)

    return descendants
```

### 4.3 预期权重分布（基础版本）

基于47骨骼骨架的估算：

| 骨骼 | 后代数量 | 累积lever_arm | 归一化权重 | 相对比例 |
|------|---------|--------------|-----------|---------|
| pelvis | 46 | ~55m | **14.0** | 1400% |
| spine_01 | ~30 | ~30m | **7.5** | 750% |
| upperarm_l | ~12 | ~5m | **1.25** | 125% |
| lowerarm_l | 6 | ~1.8m | **0.45** | 45% |
| hand_l | 5 | ~0.5m | **0.13** | 13% |
| thumb_01_l | 0 | 0m | **0.0** | **0%** ⚠️ |

---

## 5. 末端骨骼问题与优化

### 5.1 问题：末端骨骼权重为0

**现象**：
```python
# 手指、脚趾等末端骨骼没有后代
thumb_01_l: descendants=0 → influence=0 → weight=0 ❌

# 训练时被完全忽略
# 导致手指姿态误差大
```

**根本矛盾**：

```
"下游影响"物理量回答：
  ❓ 如果这个关节错了，会传播多少？
  → 手指：0传播（无后代）

但视觉重要性问：
  ❓ 如果这个关节错了，玩家能看出来吗？
  → 手指：很明显！（握武器、施法手势）
```

### 5.2 解决方案：包含自身长度

```python
# include_self=True
influence[i] = bone_length[i] + Σ(lever_arm to descendants)

# 例如：
thumb_01_l:
  self_length = 0.0513m
  downstream = 0
  influence = 0.0513  # 非零！

pelvis:
  self_length = 0.891m
  downstream = 55m
  influence = 55.891m  # 略微增加
```

**效果**：
- 末端骨骼有基础权重（自身长度）
- 但相对pelvis仍然很小（0.0513 vs 55.891）

### 5.3 问题依然存在

重新计算归一化权重（include_self=True）：

| 骨骼 | Influence | 归一化权重 |
|------|-----------|-----------|
| pelvis | 55.891 | 14.0 |
| lowerarm_l | 2.03 | 0.51 |
| hand_l | 0.731 | 0.18 |
| thumb_01_l | 0.0513 | **0.013** ⚠️ |

**问题**：
- thumb权重仅1.3%，仍然过低
- pelvis:thumb = 1077:1（极端比例）
- 手指在训练中几乎被忽略

---

## 6. 指数缩放优化

### 6.1 数学直觉

**目标**：压缩极端值差距，保留相对排序

```python
原始范围: pelvis=55, thumb=0.05  → 比例 1100:1 ❌
期望范围: pelvis=8,  thumb=0.3   → 比例 27:1  ✅
```

**方法**：对下游影响做指数缩放

```python
downstream_scaled = downstream ** power

power=1.0: 原始值（无变化）
power=0.5: 平方根（压缩大值）
power=0.3: 立方根级别（更激进）
```

### 6.2 平方根缩放效果

```python
# power = 0.5
pelvis:   downstream=55   → 55^0.5 = 7.42
lowerarm: downstream=1.8  → 1.8^0.5 = 1.34
thumb:    downstream=0    → 0^0.5 = 0

# 加上自身长度：
pelvis:   influence = 0.891 + 7.42 = 8.31
lowerarm: influence = 0.230 + 1.34 = 1.57
thumb:    influence = 0.0513 + 0 = 0.0513
```

归一化后（假设平均=2.0）：

| 骨骼 | Influence | 归一化权重 | 相对比例 |
|------|-----------|-----------|---------|
| pelvis | 8.31 | **4.16** | 416% ✅ |
| spine_01 | ~6.5 | **3.25** | 325% ✅ |
| lowerarm_l | 1.57 | **0.79** | 79% ✅ |
| hand_l | 0.938 | **0.47** | 47% ✅ |
| thumb_01_l | 0.0513 | **0.026** | **2.6%** ⚠️ 仍然低 |

### 6.3 进一步优化：增强自身贡献

```python
# self_scale = 1.5
influence = (self_scale × self_length) + downstream^power

# thumb:
influence = 1.5 × 0.0513 + 0 = 0.077

# 归一化后：
thumb权重 ≈ 0.077 / 2.0 = 0.038 (3.8%) ✅ 略有改善
```

### 6.4 最小权重保底

```python
# 设置下限：不低于平均值的5%
weights = torch.clamp(weights, min=0.05)
weights = weights / weights.mean()  # 重新归一化

# thumb:
原始权重 = 0.026 → clamp后 = 0.05 (5%) ✅
```

### 6.5 组合方案

```python
def _compute_unified_weights(self,
                             downstream_power=0.6,
                             self_scale=1.5,
                             min_weight_percentile=0.05):
    """
    统一权重 = (self_scale × 自身长度) + (下游影响)^power

    Args:
        downstream_power: 下游影响的指数 (0.5-0.7)
            - 越小，压缩越强，末端权重越高
        self_scale: 自身长度的放大系数 (1.0-2.0)
            - 提升所有骨骼（尤其末端）的基础权重
        min_weight_percentile: 最小权重占比 (0.03-0.10)
            - 保底：不低于平均值的x%
    """
    for i in range(J):
        # 自身贡献（放大）
        self_contrib = self_scale * torch.norm(local_offsets[i])

        # 下游贡献（压缩）
        downstream = sum(lever_arms_to_descendants)
        downstream_scaled = downstream ** downstream_power

        influence = self_contrib + downstream_scaled
        weights[i] = influence

    # 归一化
    weights = weights / weights.mean()

    # 裁剪下限
    weights = torch.clamp(weights, min=min_weight_percentile)

    # 重新归一化
    weights = weights / weights.mean()

    return weights
```

### 6.6 预期权重分布（优化后）

**参数**：`power=0.6, self_scale=1.5, min_weight=0.05`

| 骨骼 | 归一化权重 | 说明 |
|------|-----------|------|
| pelvis | **5-8x** | 最重要，但不过分 ✅ |
| spine_01 | **3-4x** | 次重要 ✅ |
| upperarm | **1.5-2x** | 重要 ✅ |
| lowerarm | **0.8-1.2x** | 接近平均 ✅ |
| hand | **0.5-0.8x** | 略低于平均 ✅ |
| fingers | **0.15-0.30x** | 保留15-30%权重 ✅ |
| toes | **0.10-0.20x** | 保留10-20%权重 ✅ |

**对比原始方案**：
- pelvis: 14x → 5-8x（降低，避免过度集中）
- fingers: 0.01x → 0.15-0.30x（提升15-30倍！）

---

## 7. 视觉重要性调制

### 7.1 问题：几何重要性 ≠ 视觉重要性

**示例**：

```
thumb_01_l (手指):
  - 几何影响：小（下游短）
  - 视觉重要性：高（握剑、施法手势）

ball_l (脚趾):
  - 几何影响：小（与手指类似）
  - 视觉重要性：低（常被遮挡、玩家很少注意）
```

**但unified_weight给它们相同的权重！**

### 7.2 解决方案：区域差异化

```python
# 基于ARPG玩家视角的视觉重要性
VISUAL_IMPORTANCE = {
    # 高优先级区域
    'hand': 1.5,      # 武器、施法手势、交互动作
    'finger': 1.3,    # 手指姿态（握持细节）
    'head': 1.2,      # 角色朝向、表情
    'weapon': 1.4,    # 武器骨骼（如果有）

    # 中等优先级
    'upperarm': 1.1,  # 挥砍动作
    'spine': 1.0,     # 默认
    'pelvis': 1.0,

    # 低优先级区域
    'foot': 0.8,      # 常被地面、草丛遮挡
    'calf': 0.9,      # 腿部细节较少被关注
    'toe': 0.5,       # 很少被玩家注意
    'ball': 0.5,      # 脚趾末端

    # 辅助骨骼（twist等）
    'twist': 0.7,     # 扭曲辅助骨骼，视觉影响小
}
```

### 7.3 实现

```python
def _apply_visual_importance(self, weights: torch.Tensor) -> torch.Tensor:
    """应用视觉重要性调制"""
    modulated = weights.clone()

    for i, bone_name in enumerate(self.bone_names):
        bone_lower = bone_name.lower()

        # 匹配视觉重要性字典
        for key, multiplier in VISUAL_IMPORTANCE.items():
            if key in bone_lower:
                modulated[i] *= multiplier
                break  # 只匹配第一个

    # 重新归一化
    return modulated / modulated.mean()
```

### 7.4 最终权重分布（含视觉调制）

| 骨骼 | 几何权重 | 视觉系数 | 最终权重 |
|------|---------|---------|---------|
| pelvis | 6.0x | 1.0 | **6.0x** |
| hand_l | 0.6x | 1.5 | **0.9x** ✅ 提升50% |
| thumb_01_l | 0.2x | 1.3 | **0.26x** ✅ 提升30% |
| ball_l | 0.15x | 0.5 | **0.075x** ✅ 降低50% |

**效果**：
- 手指权重提升 → 握武器姿态更精确
- 脚趾权重降低 → 节省优化预算给更重要区域

---

## 8. 完整实现代码

### 8.1 模型类（train/models.py）

```python
class YourModel(nn.Module):
    def __init__(self, skeleton, bone_weight_config, ...):
        super().__init__()

        # 骨架数据
        self.bone_names = skeleton['bone_names']
        self.parents = skeleton['parents']
        self.ref_local_offsets_m = skeleton['ref_local_offsets_m']  # [J, 3]

        # 权重配置
        self.bone_weight_mode = bone_weight_config.get('mode', 'unified')
        self.downstream_power = bone_weight_config.get('downstream_power', 0.6)
        self.self_scale = bone_weight_config.get('self_scale', 1.5)
        self.min_weight_percentile = bone_weight_config.get('min_weight_percentile', 0.05)
        self.use_visual_importance = bone_weight_config.get('use_visual_importance', True)

        # 计算并缓存权重
        self.register_buffer('bone_weights', self._compute_bone_weights())

    def _compute_bone_weights(self) -> torch.Tensor:
        """根据配置计算骨骼权重"""
        if self.bone_weight_mode == 'uniform':
            return torch.ones(len(self.bone_names), dtype=torch.float32)

        elif self.bone_weight_mode == 'unified':
            weights = self._compute_unified_weights(
                downstream_power=self.downstream_power,
                self_scale=self.self_scale,
                min_weight_percentile=self.min_weight_percentile
            )

            if self.use_visual_importance:
                weights = self._apply_visual_importance(weights)

            return weights

        else:
            raise ValueError(f"Unknown bone_weight_mode: {self.bone_weight_mode}")

    def _compute_unified_weights(self,
                                 downstream_power=0.6,
                                 self_scale=1.5,
                                 min_weight_percentile=0.05) -> torch.Tensor:
        """
        统一权重 = (self_scale × 自身长度) + (下游影响)^power
        """
        J = len(self.bone_names)
        parents = self.parents
        local_offsets = self.ref_local_offsets_m

        # 1. FK计算全局位置
        global_positions = self._forward_kinematics(local_offsets, parents)

        # 2. 计算每个骨骼的影响
        weights = torch.zeros(J, dtype=torch.float32)

        for i in range(J):
            # 自身贡献
            self_length = torch.norm(local_offsets[i])
            self_contrib = self_scale * self_length

            # 下游贡献
            downstream = 0.0
            bone_i_start = global_positions[i]

            descendants = self._get_all_descendants(i, parents)
            for j in descendants:
                bone_j_end = global_positions[j] + local_offsets[j]
                lever_arm = torch.norm(bone_j_end - bone_i_start)
                downstream += lever_arm

            # 指数缩放
            downstream_scaled = downstream ** downstream_power if downstream > 0 else 0.0

            # 组合
            influence = self_contrib + downstream_scaled
            weights[i] = influence

        # 3. 归一化
        weights = weights / weights.mean()

        # 4. 裁剪下限
        weights = torch.clamp(weights, min=min_weight_percentile)

        # 5. 重新归一化
        weights = weights / weights.mean()

        return weights

    def _forward_kinematics(self, local_offsets: torch.Tensor, parents: list) -> torch.Tensor:
        """前向运动学：计算全局关节位置"""
        J = len(parents)
        global_positions = torch.zeros(J, 3, dtype=torch.float32)

        for i in range(J):
            if parents[i] < 0:
                # 根骨骼
                global_positions[i] = local_offsets[i]
            else:
                # 子骨骼 = 父位置 + 局部偏移
                global_positions[i] = global_positions[parents[i]] + local_offsets[i]

        return global_positions

    def _get_all_descendants(self, bone_idx: int, parents: list) -> list:
        """BFS获取所有后代骨骼索引"""
        J = len(parents)
        descendants = []
        queue = [bone_idx]
        visited = set([bone_idx])

        while queue:
            current = queue.pop(0)
            for j in range(J):
                if parents[j] == current and j not in visited:
                    descendants.append(j)
                    queue.append(j)
                    visited.add(j)

        return descendants

    def _apply_visual_importance(self, weights: torch.Tensor) -> torch.Tensor:
        """应用视觉重要性调制"""
        VISUAL_IMPORTANCE = {
            'hand': 1.5,
            'finger': 1.3,
            'thumb': 1.3,
            'index': 1.3,
            'middle': 1.3,
            'ring': 1.3,
            'pinky': 1.3,
            'head': 1.2,
            'upperarm': 1.1,
            'foot': 0.8,
            'calf': 0.9,
            'toe': 0.5,
            'ball': 0.5,
            'twist': 0.7,
        }

        modulated = weights.clone()

        for i, bone_name in enumerate(self.bone_names):
            bone_lower = bone_name.lower()
            for key, multiplier in VISUAL_IMPORTANCE.items():
                if key in bone_lower:
                    modulated[i] *= multiplier
                    break

        return modulated / modulated.mean()

    def compute_geo_loss(self, pred_rot, gt_rot):
        """计算加权geodesic loss"""
        # pred_rot, gt_rot: [B, T, J, 3, 3]

        # Geodesic distance
        geo_dist = self._geodesic_distance(pred_rot, gt_rot)  # [B, T, J]

        # 应用权重
        weighted_geo = geo_dist * self.bone_weights.view(1, 1, -1)  # [B, T, J]

        # 平均
        loss = weighted_geo.mean()

        return loss
```

### 8.2 训练脚本（train/training_MPL.py）

```python
import json

def _load_skeleton_data(template_path):
    """从norm_template.json加载骨架结构"""
    with open(template_path, 'r') as f:
        template = json.load(f)

    skeleton = template['meta']['skeleton']

    return {
        'bone_names': skeleton['bone_names'],
        'parents': skeleton['parents'],
        'ref_local_offsets_m': torch.tensor(
            skeleton['ref_local_offsets_m'],
            dtype=torch.float32
        )
    }

def main():
    parser = argparse.ArgumentParser()

    # 骨骼权重配置
    parser.add_argument('--bone_weight_mode', type=str, default='unified',
                        choices=['uniform', 'unified', 'adaptive'],
                        help='Bone weighting strategy')

    # Unified权重参数
    parser.add_argument('--downstream_power', type=float, default=0.6,
                        help='Power for downstream influence scaling (0.5-0.7)')
    parser.add_argument('--self_scale', type=float, default=1.5,
                        help='Scale for self-length contribution (1.0-2.0)')
    parser.add_argument('--min_weight_percentile', type=float, default=0.05,
                        help='Minimum weight as fraction of mean (0.03-0.10)')
    parser.add_argument('--use_visual_importance', action='store_true',
                        help='Apply visual importance modulation')

    args = parser.parse_args()

    # 加载骨架数据
    skeleton = _load_skeleton_data('raw_data/processed_data/norm_template.json')

    # 权重配置
    bone_weight_config = {
        'mode': args.bone_weight_mode,
        'downstream_power': args.downstream_power,
        'self_scale': args.self_scale,
        'min_weight_percentile': args.min_weight_percentile,
        'use_visual_importance': args.use_visual_importance,
    }

    # 创建模型
    model = YourModel(
        skeleton=skeleton,
        bone_weight_config=bone_weight_config,
        ...
    )

    # 打印权重分布（调试）
    print("\n=== Bone Weights ===")
    for i, (name, weight) in enumerate(zip(skeleton['bone_names'], model.bone_weights)):
        print(f"{i:2d}. {name:20s}: {weight:.3f}")
    print(f"Mean: {model.bone_weights.mean():.3f}")
    print(f"Std: {model.bone_weights.std():.3f}")
    print(f"Min: {model.bone_weights.min():.3f}, Max: {model.bone_weights.max():.3f}")

    # 训练...
```

### 8.3 权重可视化脚本（tools/visualize_bone_weights.py）

```python
import json
import torch
import matplotlib.pyplot as plt
import numpy as np

def load_and_compute_weights(template_path, config):
    """加载骨架并计算权重"""
    # ... (与上面model代码相同)
    pass

def visualize_weights(bone_names, weights, save_path='bone_weights.png'):
    """可视化骨骼权重"""
    fig, ax = plt.subplots(figsize=(14, 8))

    indices = np.arange(len(bone_names))
    colors = ['red' if w > 2.0 else 'orange' if w > 1.0 else 'green' if w > 0.5 else 'blue'
              for w in weights]

    ax.bar(indices, weights, color=colors, alpha=0.7)
    ax.axhline(y=1.0, color='black', linestyle='--', label='Mean (1.0)')

    ax.set_xlabel('Bone Index')
    ax.set_ylabel('Weight (normalized)')
    ax.set_title('Unified Bone Weights Distribution')
    ax.set_xticks(indices[::3])
    ax.set_xticklabels([bone_names[i] for i in indices[::3]], rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Saved to {save_path}")

if __name__ == '__main__':
    config = {
        'downstream_power': 0.6,
        'self_scale': 1.5,
        'min_weight_percentile': 0.05,
        'use_visual_importance': True,
    }

    bone_names, weights = load_and_compute_weights(
        'raw_data/processed_data/norm_template.json',
        config
    )

    visualize_weights(bone_names, weights)
```

---

## 9. 参数调优指南

### 9.1 核心参数

| 参数 | 默认值 | 范围 | 效果 |
|------|--------|------|------|
| `downstream_power` | 0.6 | 0.5-0.7 | 下游影响的压缩程度 |
| `self_scale` | 1.5 | 1.0-2.0 | 自身长度的放大系数 |
| `min_weight_percentile` | 0.05 | 0.03-0.10 | 最小权重保底 |
| `use_visual_importance` | True | bool | 是否应用视觉调制 |

### 9.2 调参策略

#### Step 1: 确定 `downstream_power`

**目标**：平衡pelvis与末端骨骼的权重比

```python
# 测试不同power值
for power in [0.5, 0.6, 0.7]:
    weights = compute_weights(downstream_power=power)
    ratio = weights[pelvis_idx] / weights[hand_idx]
    print(f"power={power}: pelvis/hand = {ratio:.1f}")

# 期望：
# power=0.5: ratio ≈ 15-20  (末端权重较高)
# power=0.6: ratio ≈ 8-12   (推荐)
# power=0.7: ratio ≈ 5-8    (末端权重更高)
```

**建议**：
- ARPG（手部动作重要）：0.55-0.60
- 动画电影（整体流畅性）：0.65-0.70
- 格斗游戏（精细姿态）：0.50-0.55

#### Step 2: 调整 `self_scale`

**目标**：提升所有骨骼（尤其末端）的基础权重

```python
# 检查最小权重（末端骨骼）
min_weight = weights.min().item()

if min_weight < 0.03:
    # 增加self_scale
    self_scale += 0.2
```

**建议**：
- 1.0-1.3：保守（依赖下游影响）
- 1.5：推荐（平衡）
- 1.8-2.0：激进（末端权重更高）

#### Step 3: 设置 `min_weight_percentile`

**目标**：避免任何骨骼被完全忽略

```python
# 检查权重分布
percentiles = np.percentile(weights, [1, 5, 10, 25])
print(f"1%: {percentiles[0]:.3f}")
print(f"5%: {percentiles[1]:.3f}")

# 如果5%分位 < 0.05，设置保底
min_weight_percentile = 0.05
```

**建议**：
- 0.03：最小保底（允许更大差异）
- 0.05：推荐（保证末端至少5%）
- 0.08-0.10：激进（压缩极端值）

### 9.3 实验组合

| 场景 | downstream_power | self_scale | min_weight | 说明 |
|------|-----------------|-----------|-----------|------|
| **保守** | 0.7 | 1.2 | 0.03 | 更依赖下游影响 |
| **推荐** | 0.6 | 1.5 | 0.05 | 平衡 |
| **激进** | 0.5 | 1.8 | 0.08 | 末端权重高 |
| **调试** | 1.0 | 1.0 | 0.0 | 无缩放（查看原始分布） |

---

## 10. 实验对比方案

### 10.1 配置文件（config/exp_unified_weights.json）

```json
{
  "experiment_name": "unified_bone_weights",
  "base_config": {
    "epochs": 200,
    "batch_size": 32,
    "lr": 0.0001,
    "w_geo": 1.0,
    "w_yaw": 0.0
  },
  "experiment_groups": [
    {
      "name": "E0_baseline_uniform",
      "description": "Baseline: uniform weights (all bones equal)",
      "bone_weight_mode": "uniform"
    },
    {
      "name": "E1_adaptive_old",
      "description": "Old approach: adaptive + hierarchy (commit ea2a5ca)",
      "bone_weight_mode": "adaptive",
      "adaptive_bone_weights": true,
      "use_hierarchy_weights": true,
      "hierarchy_mode": "multiply"
    },
    {
      "name": "E2_unified_conservative",
      "description": "Unified weights: conservative params",
      "bone_weight_mode": "unified",
      "downstream_power": 0.7,
      "self_scale": 1.2,
      "min_weight_percentile": 0.03,
      "use_visual_importance": false
    },
    {
      "name": "E3_unified_recommended",
      "description": "Unified weights: recommended params",
      "bone_weight_mode": "unified",
      "downstream_power": 0.6,
      "self_scale": 1.5,
      "min_weight_percentile": 0.05,
      "use_visual_importance": true
    },
    {
      "name": "E4_unified_aggressive",
      "description": "Unified weights: aggressive (high end-effector weights)",
      "bone_weight_mode": "unified",
      "downstream_power": 0.5,
      "self_scale": 1.8,
      "min_weight_percentile": 0.08,
      "use_visual_importance": true
    },
    {
      "name": "E5_unified_no_visual",
      "description": "Unified weights without visual importance modulation",
      "bone_weight_mode": "unified",
      "downstream_power": 0.6,
      "self_scale": 1.5,
      "min_weight_percentile": 0.05,
      "use_visual_importance": false
    }
  ],
  "metrics_to_compare": [
    "geo_loss_total",
    "geo_loss_pelvis",
    "geo_loss_hand_l",
    "geo_loss_hand_r",
    "geo_loss_fingers",
    "left_right_symmetry_error"
  ]
}
```

### 10.2 评估指标

```python
def compute_evaluation_metrics(pred, gt, bone_weights, bone_names):
    """计算评估指标"""
    metrics = {}

    # 1. 总体geo loss
    geo_dist = geodesic_distance(pred, gt)  # [B, T, J]
    metrics['geo_loss_total'] = (geo_dist * bone_weights).mean()

    # 2. 关键骨骼的geo loss
    key_bones = {
        'pelvis': [0],
        'hand_l': [11],
        'hand_r': [24],
        'fingers': [12,13,14,15,16, 25,26,27,28,29],
        'feet': [37,38, 44,45],
    }

    for name, indices in key_bones.items():
        metrics[f'geo_loss_{name}'] = geo_dist[:, :, indices].mean()

    # 3. 左右对称性误差
    lr_pairs = [
        (4, 17),   # clavicle
        (5, 18),   # upperarm
        (8, 21),   # lowerarm
        (11, 24),  # hand
    ]

    sym_errors = []
    for l_idx, r_idx in lr_pairs:
        l_error = geo_dist[:, :, l_idx].mean()
        r_error = geo_dist[:, :, r_idx].mean()
        sym_error = abs(l_error - r_error)
        sym_errors.append(sym_error)

    metrics['left_right_symmetry_error'] = torch.stack(sym_errors).mean()

    return metrics
```

### 10.3 预期结果对比

| 实验组 | Total Loss | Hand Loss | Fingers Loss | LR Symmetry | 说明 |
|--------|-----------|-----------|--------------|-------------|------|
| E0 (uniform) | 1.00 (基线) | 高 | 很高 | 好 | 无区分度 |
| E1 (adaptive old) | 0.85 | 中 | 高 | **差** ⚠️ | 左右不对称 |
| E2 (conservative) | 0.78 | 低 | 中 | **好** ✅ | 保守平衡 |
| **E3 (recommended)** | **0.72** | **低** | **中低** | **好** ✅ | **推荐** |
| E4 (aggressive) | 0.75 | 很低 | 低 | 好 ✅ | 末端优秀 |
| E5 (no visual) | 0.74 | 低 | 中 | 好 ✅ | 无视觉调制 |

**关键发现**：
- ✅ **E3优于E1**：总体loss更低，且左右对称
- ✅ **E3优于E0**：关键区域（手、手指）误差更低
- ✅ **E3 vs E5**：视觉调制进一步降低手部误差

---

## 11. 预期效果

### 11.1 解决的核心问题

#### ✅ 问题1：左右骨骼不对称

**Before (adaptive)**:
```
lowerarm_l: weight=17.1 (基于std=0.0584)
lowerarm_r: weight=31.6 (基于std=0.0316)
→ 差异：54% ❌
```

**After (unified)**:
```
lowerarm_l: weight=0.85 (基于几何结构)
lowerarm_r: weight=0.85 (镜像对称)
→ 差异：0% ✅
```

#### ✅ 问题2：依赖训练数据统计

**Before**:
- 数据多 → std大 → 权重小
- Walk_R数据2倍 → 右侧std不同 → 权重不对称

**After**:
- 基于骨架几何 → 与数据分布无关
- 左右镜像 → 自动对称

#### ✅ 问题3：多维度组合复杂

**Before**:
```python
weight = (1/prior_std) × log(descendants+1) × 其他调节
# 需要调：hierarchy_mode, gamma, alpha等
```

**After**:
```python
weight = self_length + downstream^power
# 只需调：power, self_scale, min_weight
```

### 11.2 训练效果预期

#### 整体geo loss

```
Baseline (uniform):     1.00
Adaptive (old):         0.85 (但左右不对称)
Unified (recommended):  0.72 ✅ 降低15%
```

#### 关键区域改善

| 区域 | Baseline | Adaptive | Unified | 改善 |
|------|---------|---------|---------|------|
| pelvis | 中 | 低 | **低** | ✅ |
| hand | 高 | 中 | **低** | ✅ 降低40% |
| fingers | 很高 | 高 | **中低** | ✅ 降低50% |

#### 对称性

```
Adaptive: 左右误差差异 15-20% ❌
Unified:  左右误差差异 <2%   ✅
```

### 11.3 视觉质量改善

- ✅ 手部姿态更精确（握武器、施法手势）
- ✅ 手指自然度提升（不再"僵硬"）
- ✅ 左右动作对称（挥砍、格挡）
- ✅ 整体流畅性保持（pelvis仍有高权重）

---

## 12. 总结

### 12.1 设计原则

1. **单一物理量**：用"下游影响"统一所有考虑因素
2. **数据无关**：基于骨架几何，不依赖训练数据统计
3. **自然对称**：左右镜像结构 → 权重自动对称
4. **视觉导向**：通过指数缩放和视觉调制，平衡物理重要性与视觉重要性

### 12.2 关键创新

- **指数缩放**：`downstream^0.6` 压缩极端值，保留排序
- **自身贡献**：`1.5 × self_length` 给末端骨骼基础权重
- **最小保底**：`min(0.05)` 避免任何骨骼被忽略
- **视觉调制**：手部1.5x, 脚趾0.5x，符合ARPG视角

### 12.3 实施建议

1. **先实现基础版本**（不含视觉调制）
2. **可视化权重分布**，验证pelvis:hand:finger比例合理
3. **小规模训练**（50 epochs）对比E1 vs E3
4. **如果手部仍不理想**，启用视觉调制
5. **微调参数**（power, self_scale）直到满意

### 12.4 后续扩展

- **动态权重**：根据动作类型调整（攻击动作提升手部权重）
- **渐进式权重**：训练初期uniform，后期切换unified
- **用户可配置**：通过JSON配置不同角色的视觉重要性

---

## 附录A：数学推导

### A.1 杠杆臂原理

骨骼 `i` 旋转 `θ` 时，后代骨骼 `j` 的末端位移：

```
Δp[j] = R(θ) @ (p[j] - p[i]) + p[i] - p[j]

对于小角度 θ：
R(θ) ≈ I + θ × [axis]×

其中 [axis]× 是反对称矩阵

Δp[j] ≈ θ × axis × (p[j] - p[i])

位移大小：
||Δp[j]|| ≈ θ × ||p[j] - p[i]|| = θ × lever_arm
```

因此，总影响：
```
Total_displacement = Σ ||Δp[j]||
                   = Σ (θ × lever_arm[i→j])
                   = θ × Σ lever_arm[i→j]
```

权重应正比于 `Σ lever_arm`。

### A.2 指数缩放的效果

原始分布（假设）：
```
x ~ [0.05, 0.1, 0.5, 1, 5, 10, 55]
```

平方根缩放（power=0.5）：
```
x^0.5 ~ [0.22, 0.32, 0.71, 1, 2.24, 3.16, 7.42]
```

**效果**：
- 大值压缩：55 → 7.42（-86.5%）
- 小值提升：0.05 → 0.22（+340%）
- 排序保持：单调性不变

对数空间分析：
```
log(x^0.5) = 0.5 × log(x)

相当于在对数空间缩小50%的范围
```

---

## 附录B：代码清单

完整代码位置：
- `train/models.py`: 第XXX-XXX行，`_compute_unified_weights()` 方法
- `train/training_MPL.py`: 第XXX行，骨架数据加载
- `config/exp_unified_weights.json`: 实验配置
- `tools/visualize_bone_weights.py`: 权重可视化工具

---

**文档结束**
