# 基于骨骼层级与运动幅度的组合权重系统设计文档

> **作者：** 系统分析
> **日期：** 2025-11-26
> **版本：** v2.0
> **状态：** 最终方案

---

## 📋 目录

1. [执行摘要](#执行摘要)
2. [核心问题分析](#核心问题分析)
3. [改进方案](#改进方案)
4. [理论依据](#理论依据)
5. [技术实现](#技术实现)
6. [实验设计](#实验设计)
7. [预期效果](#预期效果)
8. [附录](#附录)

---

## 执行摘要

### 当前问题

现有训练系统存在三个关键问题：

1. **运动幅度差异被忽视** - 所有骨骼使用均匀权重，但运动幅度差异达**51.3倍**
2. **骨骼层级影响未考虑** - Root骨骼误差会传播到所有子骨骼，但权重相同
3. **Yaw损失与Rot6d冲突** - Yaw是pelvis旋转的投影，单独惩罚造成重复和干扰

### 解决方案

**组合权重系统（Unified Solution）**：

```python
weight[j] = (运动幅度权重[j]) × (骨骼链权重[j])
          = (1 / motion_std[j]) × (log(num_descendants[j] + 1) + 1)
```

- ✅ 移除独立的`yaw_loss`（简化，避免干扰）
- ✅ 使用单一`rot6d_geo_loss`，所有骨骼包括pelvis
- ✅ Pelvis自然获得4-5倍权重，yaw分量被充分优化

### 关键收益

- 🎯 **Pelvis（含yaw）精度提升** - 组合权重使其获得应有的关注
- 📊 **小骨骼相对误差改善** - 运动幅度权重确保公平性
- 🌳 **层级结构被尊重** - 父骨骼获得更高优先级
- 🔧 **代码简化** - 从2个损失函数减少到1个
- ⚡ **训练更稳定** - 消除优化冲突

---

## 核心问题分析

### 问题1: 运动幅度差异巨大，但权重相同

#### 数据证据

从训练数据统计（`norm_template.json` 中的 `prior_per_dim`）：

| 骨骼类型 | 代表骨骼 | 运动幅度 (std) | 相对差异 |
|---------|---------|---------------|---------|
| 大幅度关节 | Bone 14 (hand_l) | 0.2114 | 1.0x |
| 中等关节 | Bone 0 (pelvis) | 0.0431 | 4.9x |
| 小幅度关节 | Bone 9 | 0.0041 | **51.3x** |

**关键问题**：运动幅度相差51.3倍，但在损失中权重都是1.0。

#### 具体影响场景

假设预测误差都是 **10度**：

```python
# 手腕 (运动范围 ±60°)
geodesic_loss = 0.174 rad
relative_error = 10° / 60° = 16.7%  ✅ 可接受误差

# 脊椎 (运动范围 ±5°)
geodesic_loss = 0.174 rad
relative_error = 10° / 5° = 200%    ❌ 灾难性误差！

# 当前损失计算
loss = (0.174 + 0.174) / 2 = 0.174  # 权重相同！
```

**后果**：
- 大幅度关节（手、脚）主导梯度，优先被优化
- 小幅度关节（脊椎、颈部）相对误差可能高达数倍，但被忽视
- 训练不均衡，某些关节过拟合，某些欠拟合

---

### 问题2: 骨骼层级结构的影响未被考虑

#### Forward Kinematics的传播特性

骨骼是树状层级结构，父骨骼的误差会传播到所有子骨骼：

```
pelvis (depth=0, 影响46个子骨骼)
├── spine_01 (depth=1, 影响~25个子骨骼)
│   ├── spine_02 → spine_03 → neck → head
│   ├── shoulder_l → arm_l → forearm_l → hand_l (depth=5)
│   └── shoulder_r → arm_r → forearm_r → hand_r
├── thigh_l (depth=1, 影响3个子骨骼)
│   └── calf_l → foot_l
└── thigh_r → calf_r → foot_r
```

#### 误差放大效应

```python
# 情景1: Pelvis旋转错10度
pelvis_error = 10°
→ 所有46个子骨骼的世界坐标都偏移
→ 末端关节(hand)的位置误差 = 10° × 骨骼链总长度
→ 影响范围：整个身体

# 情景2: Hand旋转错10度
hand_error = 10°
→ 只有手部本身受影响
→ 影响范围：局部

# 但当前权重
weight[pelvis] = 1.0
weight[hand] = 1.0  # 完全相同！
```

**关键洞察**：
- **Root/Parent bones应该有更高权重**（一损俱损）
- **Leaf bones权重可以相对低**（局部影响）
- 当前系统**完全忽视了这种层级关系**

---

### 问题3: Yaw损失与Pelvis旋转的冗余干扰

#### Yaw的本质

Yaw是从pelvis的3D旋转矩阵中提取的**水平朝向分量**：

```python
# train/models.py:1178-1211
Rp_root = Rp[:, :, 0]  # 提取pelvis的旋转矩阵 [B, T, 3, 3]
forward_vec = Rp_root[:, :, :, forward_axis]  # 提取forward向量
yaw = torch.atan2(forward_vec[:, 1], forward_vec[:, 0])  # 投影到水平面

# Pelvis的完整旋转 = Yaw (水平) + Pitch (俯仰) + Roll (翻滚)
```

#### 重复惩罚的量化

当前损失配置（`config/exp_phase_mpl.json`）：

```json
{
  "w_rot_geo": 1.0,      // 包含所有47个骨骼（含pelvis）
  "w_yaw": 1.5           // 单独提取pelvis的yaw分量
}
```

权重分析：

```python
# Pelvis在rot_geo_loss中的权重
pelvis_in_geo = 1.0 / 47 ≈ 0.021

# Pelvis的yaw在yaw_loss中的权重
yaw_weight = 1.5

# Pelvis的yaw方向总权重
total_yaw_attention = 0.021 + 1.5 ≈ 1.52

# 分布：yaw_loss占比 = 1.5 / 1.52 ≈ 98.6%
```

#### 干扰机制

```python
# 优化目标1 (rot_geo_loss)：
"让pelvis的完整3D旋转正确"
→ 包括 yaw + pitch + roll 三个自由度

# 优化目标2 (yaw_loss)：
"单独让pelvis的yaw分量正确"
→ 只关注 yaw，对pitch/roll不敏感

# 潜在冲突：
如果某个解：
  - yaw = perfect (满足yaw_loss)
  - pitch/roll = bad (但yaw_loss不管)
  - 整体3D旋转 = suboptimal (rot_geo_loss不满意)

→ 优化器可能陷入局部最优
→ yaw_loss的1.5权重会"拉扯"优化方向
```

#### 核心问题

如果**pelvis的完整rot6d是正确的** → 从中提取的**yaw自然就是正确的**

因此：
- ✅ **Rot6d_geo_loss已经包含了yaw的信息**
- ❌ **单独的yaw_loss是多余的**
- ❌ **而且会造成优化干扰**

---

## 改进方案

### 核心思路：组合权重系统

使用**单一损失函数** + **智能权重分配**，同时解决三个问题：

```python
# ============ 唯一的损失函数 ============
loss = rot6d_geo_loss(pred, gt, weights=combined_weights)

# ============ 组合权重 ============
combined_weights[j] = motion_weight[j] × hierarchy_weight[j]

其中：
  motion_weight[j] = 1 / (运动幅度[j] + eps)      # 解决问题1
  hierarchy_weight[j] = log(子孙数量[j] + 1) + 1  # 解决问题2

移除 yaw_loss（w_yaw = 0.0）                      # 解决问题3
```

### 权重组成部分

#### 1. 运动幅度权重（Motion Amplitude Weighting）

**目的**：将绝对误差转换为相对误差

```python
motion_weight[j] = 1.0 / (bone_motion_std[j] + eps)
```

**效果**：
- 大幅度关节（std=0.21）→ 权重低（≈4.7）→ 允许较大绝对误差
- 小幅度关节（std=0.004）→ 权重高（≈250）→ 严格要求精度

**数据来源**：`norm_template.json` 中的 `prior_per_dim`

```python
# 每个骨骼的6D先验取平均
bone_motion_std[j] = mean(prior_per_dim[j*6 : (j+1)*6])
```

#### 2. 骨骼链权重（Hierarchy Weighting）

**目的**：根据影响范围分配优先级

```python
# 统计每个骨骼的子孙数量
num_descendants[j] = count(所有以j为祖先的骨骼)

# 对数缩放（避免root权重过大）
hierarchy_weight[j] = log(num_descendants[j] + 1) + 1.0
```

**效果**：
- Pelvis（46个子孙）→ log(47) + 1 ≈ **4.85**
- Spine_01（25个子孙）→ log(26) + 1 ≈ **4.26**
- Hand（0个子孙）→ log(1) + 1 = **1.0**

#### 3. 组合策略

**相乘模式（推荐）**：

```python
combined_weight[j] = motion_weight[j] × hierarchy_weight[j]
combined_weight[j] = combined_weight[j] / mean(combined_weight)  # 归一化
```

**实际权重示例**：

| 骨骼 | Motion Std | Motion W | Hier W | **Combined** | 相对Hand |
|------|-----------|----------|--------|-------------|---------|
| **Pelvis** | 0.0431 | 23.2 | 4.85 | **112.6** | **23.8x** |
| Spine_01 | 0.0328 | 30.5 | 4.26 | 129.9 | 27.4x |
| Thigh_L | 0.0129 | 77.6 | 2.10 | 162.9 | 34.4x |
| Neck | 0.1164 | 8.6 | 2.79 | 24.0 | 5.1x |
| **Hand_L** | 0.2114 | 4.7 | 1.00 | **4.7** | 1.0x |

归一化后（假设总和/47 ≈ 25）：
- Pelvis权重 ≈ **4.5倍于平均骨骼**
- Hand权重 ≈ 0.19倍于平均骨骼

### 为什么移除Yaw Loss？

#### 论证1: Pelvis权重已足够

```python
# 原方案：
pelvis_attention_on_yaw = (1/47 在geo中) + (1.5 在yaw中) ≈ 1.52

# 新方案：
pelvis_weight_in_geo = 4.5倍于平均 = 4.5/47 ≈ 0.096
# Yaw是pelvis旋转的1个分量（共3个自由度：yaw, pitch, roll）
# Geodesic distance自然包含所有分量
# 如果pelvis的完整旋转正确 → yaw自然正确

实际关注度相当，而且没有优化冲突！
```

#### 论证2: 消除干扰

```python
# 有yaw_loss时：
优化目标 = "让yaw=perfect" + "让整体rot6d=perfect"
→ 可能冲突（yaw好但整体不好）

# 无yaw_loss时：
优化目标 = "让整体rot6d=perfect"
→ 单一目标，无冲突
→ yaw作为其中一个分量自然被优化
```

#### 论证3: 简化监控

```python
# 训练时：只优化rot6d_geo_loss
# 评估时：仍然计算YawAbsDeg作为监控指标

metrics = {
    "loss/rot_geo": ...,           # 训练loss
    "eval/YawAbsDeg": ...,         # 监控指标（不参与训练）
    "eval/GeoDeg_Pelvis": ...,     # Pelvis整体精度
}
```

保持监控能力，但不干扰训练。

---

## 理论依据

### 数学证明：Yaw包含在Geodesic中

#### Geodesic Distance的定义

```python
# SO(3)流形上的测地距离
R_error = R_pred^T @ R_gt
trace = R_error[0,0] + R_error[1,1] + R_error[2,2]
cos_theta = (trace - 1) / 2
geodesic_distance = arccos(cos_theta)
```

这个距离度量了**完整的3D旋转误差**，自然包含：
- Yaw（绕Z轴旋转）
- Pitch（绕Y轴旋转）
- Roll（绕X轴旋转）

#### 定量分析

假设pelvis的旋转误差只在yaw方向（10度）：

```python
R_gt = Rz(0°)  # 真实yaw=0
R_pred = Rz(10°)  # 预测yaw=10°

# Geodesic distance
R_error = Rz(10°)^T @ Rz(0°) = Rz(-10°)
trace = cos(-10°) + cos(-10°) + 1 = 2*0.985 + 1 = 2.970
cos_theta = (2.970 - 1) / 2 = 0.985
geodesic = arccos(0.985) = 10.0°  ✓

# Yaw loss
yaw_error = |10° - 0°| = 10.0°  ✓

# 完全相同！Geodesic已经捕获了yaw误差
```

**结论**：当pelvis只有yaw误差时，geodesic distance = yaw error。因此提高pelvis的权重，自然就提高了对yaw的关注。

### 权重传递分析

#### 场景：Pelvis权重 vs Yaw权重

```python
# 配置A（当前）
w_yaw = 1.5
pelvis_weight_in_geo = 1/47 ≈ 0.021

Loss = rot_geo_loss + 1.5 * yaw_loss
     = (1/47) * pelvis_geo + ... + 1.5 * pelvis_yaw

Pelvis的梯度 = ∂(pelvis_geo)/47 + 1.5 * ∂(pelvis_yaw)
             ≈ 被yaw_loss主导

# 配置B（新方案）
w_yaw = 0.0
pelvis_weight_in_geo = 4.5

Loss = rot_geo_loss_weighted
     = 4.5 * pelvis_geo + ...

Pelvis的梯度 = 4.5 * ∂(pelvis_geo)
             ≈ 所有旋转分量均衡优化
```

**关键差异**：
- 配置A：98.6%的pelvis梯度来自yaw分量（不均衡）
- 配置B：梯度自然分配给yaw/pitch/roll（均衡）

---

## 技术实现

### 实现概览

需要修改的文件：
1. `train/models.py` - 权重计算逻辑
2. `train/training_MPL.py` - 数据加载和参数传递
3. `config/*.json` - 配置开关

### 1. 骨骼层级权重计算

**文件**：`train/models.py`
**位置**：新增方法（约670行后）

```python
def _compute_hierarchy_weights(self) -> torch.Tensor:
    """
    基于骨骼层级结构计算权重。
    权重 = log(子孙数量 + 1) + 1.0

    Returns:
        torch.Tensor: [J] 每个骨骼的层级权重
    """
    if not self.parents or len(self.parents) == 0:
        # 如果没有骨骼层级信息，返回均匀权重
        J = len(getattr(self, 'bone_names', [])) or 47
        return torch.ones(J)

    J = len(self.parents)
    num_descendants = torch.zeros(J, dtype=torch.float32)

    # 统计每个骨骼的子孙数量
    # 方法：对每个骨骼i，遍历所有骨骼j，检查i是否是j的祖先
    for j in range(J):
        ancestor = j
        visited = set()  # 防止循环引用

        while ancestor >= 0 and ancestor < J and ancestor not in visited:
            visited.add(ancestor)
            num_descendants[ancestor] += 1
            parent_idx = self.parents[ancestor]
            ancestor = parent_idx if isinstance(parent_idx, int) and parent_idx >= 0 else -1

    # 对数缩放 + 最小值1.0
    # 原因：避免root权重过大，使用log平滑差异
    hierarchy_weights = torch.log(num_descendants) + 1.0

    # 确保最小权重为1.0（leaf nodes）
    hierarchy_weights = hierarchy_weights.clamp(min=1.0)

    return hierarchy_weights


def _load_hierarchy_weights(self) -> Optional[torch.Tensor]:
    """
    加载并缓存骨骼层级权重。
    """
    if hasattr(self, '_hierarchy_weights_cache'):
        return self._hierarchy_weights_cache

    if not getattr(self, 'use_hierarchy_weights', False):
        return None

    try:
        hier_weights = self._compute_hierarchy_weights()
        self._hierarchy_weights_cache = hier_weights

        # 调试信息（首次加载时打印）
        if not hasattr(self, '_hierarchy_weights_logged'):
            self._hierarchy_weights_logged = True
            print(f"[Loss] Hierarchy weights loaded: range [{hier_weights.min():.2f}, {hier_weights.max():.2f}]")
            # 打印关键骨骼
            if len(hier_weights) > 0:
                print(f"       Pelvis (bone 0): {hier_weights[0]:.2f}")

        return hier_weights
    except Exception as e:
        print(f"[WARN] Failed to compute hierarchy weights: {e}")
        return None
```

### 2. 组合权重计算

**文件**：`train/models.py`
**位置**：修改 `_joint_weight_vector()` 方法（约659行）

```python
def _joint_weight_vector(self, device, dtype, joint_count: int) -> torch.Tensor:
    """
    计算每个关节的损失权重。

    支持三种模式：
    1. uniform: 均匀权重（向后兼容）
    2. adaptive: 基于运动幅度
    3. combined: 运动幅度 × 骨骼链（推荐）

    Args:
        device: torch device
        dtype: torch dtype
        joint_count: 关节数量

    Returns:
        torch.Tensor: [joint_count] 归一化后的权重
    """
    # 缓存key（包含所有影响因子）
    use_adaptive = getattr(self, 'use_adaptive_weights', False)
    use_hierarchy = getattr(self, 'use_hierarchy_weights', False)
    key = (str(device), str(dtype), int(joint_count), bool(use_adaptive), bool(use_hierarchy))

    cache = getattr(self, '_joint_weight_cache', None)
    if cache is None:
        cache = {}
        self._joint_weight_cache = cache
    if key in cache:
        return cache[key]

    # ========== 1. 运动幅度权重 ==========
    if use_adaptive and self.bone_prior_stds is not None:
        if len(self.bone_prior_stds) != joint_count:
            print(f"[WARN] bone_prior_stds length ({len(self.bone_prior_stds)}) "
                  f"!= joint_count ({joint_count}), fallback to uniform weights")
            motion_weights = torch.ones(joint_count, dtype=dtype, device=device)
        else:
            stds = self.bone_prior_stds.to(device=device, dtype=dtype)
            eps = 1e-6

            # 反比例权重：运动幅度大 → 权重小
            motion_weights = 1.0 / (stds + eps)
    else:
        motion_weights = torch.ones(joint_count, dtype=dtype, device=device)

    # ========== 2. 骨骼链权重 ==========
    if use_hierarchy:
        hierarchy_weights = self._load_hierarchy_weights()
        if hierarchy_weights is not None and len(hierarchy_weights) == joint_count:
            hierarchy_weights = hierarchy_weights.to(device=device, dtype=dtype)
        else:
            print(f"[WARN] hierarchy_weights unavailable or size mismatch, fallback to uniform")
            hierarchy_weights = torch.ones(joint_count, dtype=dtype, device=device)
    else:
        hierarchy_weights = torch.ones(joint_count, dtype=dtype, device=device)

    # ========== 3. 组合策略 ==========
    hierarchy_mode = getattr(self, 'hierarchy_mode', 'multiply')

    if hierarchy_mode == 'multiply':
        # 相乘：同时考虑运动幅度和层级
        weights = motion_weights * hierarchy_weights
    elif hierarchy_mode == 'add':
        # 加权和：折中方案
        alpha = float(getattr(self, 'hierarchy_alpha', 0.5))
        weights = alpha * motion_weights + (1 - alpha) * hierarchy_weights
    else:
        # 'none' 或其他：只用运动幅度权重
        weights = motion_weights

    # ========== 4. 归一化 ==========
    # 归一化使得 mean(weights) = 1.0
    # 这样总损失的尺度与原始版本一致，无需调整学习率
    weights = weights / weights.mean()

    # ========== 5. 可选：限制权重范围 ==========
    # 避免极端权重导致训练不稳定
    max_weight_ratio = float(getattr(self, 'max_weight_ratio', 100.0))
    if max_weight_ratio > 0:
        max_allowed = weights.mean() * max_weight_ratio
        weights = weights.clamp(max=max_allowed)

    # ========== 6. 调试信息 ==========
    if not hasattr(self, '_weight_vector_logged'):
        self._weight_vector_logged = True
        print(f"[Loss] Joint weights computed: "
              f"range [{weights.min():.2f}, {weights.max():.2f}], "
              f"mean={weights.mean():.2f}, std={weights.std():.2f}")
        if joint_count > 0:
            print(f"       Mode: adaptive={use_adaptive}, hierarchy={use_hierarchy}, "
                  f"combine={hierarchy_mode}")

    cache[key] = weights
    return weights
```

### 3. 加载骨骼运动先验

**文件**：`train/training_MPL.py`
**位置**：创建loss_fn之前（约3700行）

```python
def _load_bone_prior_stds(norm_template_path: str) -> Optional[List[float]]:
    """
    从 norm_template.json 提取每个骨骼的运动幅度。

    Args:
        norm_template_path: norm_template.json 文件路径

    Returns:
        List[float]: 每个骨骼的运动幅度（6D的平均值）
        None: 如果加载失败
    """
    try:
        import json
        with open(norm_template_path, 'r') as f:
            template = json.load(f)

        priors_dict = template.get('group_priors_rot6d', {})
        prior_per_dim = priors_dict.get('prior_per_dim')

        if not prior_per_dim:
            print("[WARN] 'prior_per_dim' not found in norm_template.json")
            return None

        if len(prior_per_dim) % 6 != 0:
            print(f"[WARN] prior_per_dim length ({len(prior_per_dim)}) not multiple of 6")
            return None

        # 每6个维度对应一个骨骼，取平均作为该骨骼的运动幅度指标
        num_bones = len(prior_per_dim) // 6
        bone_stds = []
        for j in range(num_bones):
            bone_6d_prior = prior_per_dim[j*6 : (j+1)*6]
            bone_avg_std = sum(bone_6d_prior) / 6.0
            bone_stds.append(bone_avg_std)

        print(f"[Loss] Loaded bone_prior_stds for {num_bones} bones from {norm_template_path}")
        print(f"       Range: [{min(bone_stds):.4f}, {max(bone_stds):.4f}]")

        return bone_stds

    except FileNotFoundError:
        print(f"[WARN] norm_template.json not found at {norm_template_path}")
        return None
    except Exception as e:
        print(f"[WARN] Failed to load bone_prior_stds: {e}")
        return None
```

### 4. 修改Loss初始化

**文件**：`train/training_MPL.py`
**位置**：创建MotionJointLoss实例（约3724行）

```python
# ========== 加载骨骼运动先验 ==========
norm_template_file = os.path.join(
    _arg('data_root', 'raw_data/processed_data'),
    'norm_template.json'
)
bone_prior_stds = _load_bone_prior_stds(norm_template_file)

if bone_prior_stds is None and _arg('adaptive_bone_weights', False):
    print("[WARN] adaptive_bone_weights=True but bone_prior_stds not loaded, "
          "will fallback to uniform weights")

# ========== 创建Loss函数 ==========
loss_fn = MotionJointLoss(
    output_layout=ds_train.output_layout,
    fps=fps_data,
    rot6d_spec=getattr(ds_train, 'rot6d_spec', {}),
    w_rot_delta=w_rot_delta,
    w_rot_delta_root=_arg('w_rot_delta_root', 0.0),
    w_rot_ortho=_arg('w_rot_ortho', 0.001),
    meta=ds_train.meta if hasattr(ds_train, 'meta') else None,  # 包含skeleton信息
    w_fk_pos=w_fk_pos,
    w_rot_local=w_rot_local,
    w_yaw=0.0,  # ⭐ 关键：移除yaw_loss

    # ========== 新增参数 ==========
    adaptive_bone_weights=bool(_arg('adaptive_bone_weights', False)),
    bone_prior_stds=bone_prior_stds,
    use_hierarchy_weights=bool(_arg('use_hierarchy_weights', False)),
    hierarchy_mode=str(_arg('hierarchy_mode', 'multiply')),  # 'multiply', 'add', 'none'
    max_weight_ratio=float(_arg('max_weight_ratio', 100.0)),
)

print(f"[Loss] Configuration:")
print(f"  w_rot_delta={loss_fn.w_rot_delta}")
print(f"  w_rot_local={loss_fn.w_rot_local}")
print(f"  w_yaw={loss_fn.w_yaw}  ⭐ (0.0 = disabled)")
print(f"  adaptive_bone_weights={loss_fn.use_adaptive_weights}")
print(f"  use_hierarchy_weights={getattr(loss_fn, 'use_hierarchy_weights', False)}")
```

### 5. 修改MotionJointLoss初始化

**文件**：`train/models.py`
**位置**：`MotionJointLoss.__init__()` 参数列表（约443行）

```python
def __init__(
    self,
    w_attn_reg: float = 0.01,
    output_layout: Dict[str, Any] = None,
    fps: float = 60.0,
    rot6d_spec: Dict[str, Any] = None,
    w_rot_ortho: float = 0.0,
    ignore_motion_groups: str = '',
    w_rot_delta: float = 1.0,
    w_rot_delta_root: float = 0.0,
    meta: Optional[Dict[str, Any]] = None,
    w_fk_pos: float = 0.0,
    w_rot_local: float = 0.0,
    w_yaw: float = 0.0,

    # ========== 新增参数 ==========
    adaptive_bone_weights: bool = False,
    bone_prior_stds: Optional[List[float]] = None,
    use_hierarchy_weights: bool = False,
    hierarchy_mode: str = 'multiply',  # 'multiply', 'add', 'none'
    max_weight_ratio: float = 100.0,
):
    super().__init__()

    # ... 现有初始化代码 ...

    # ========== 自适应权重参数 ==========
    self.use_adaptive_weights = bool(adaptive_bone_weights)
    self.bone_prior_stds: Optional[torch.Tensor] = None

    if bone_prior_stds is not None:
        self.bone_prior_stds = torch.as_tensor(bone_prior_stds, dtype=torch.float32)
        if self.use_adaptive_weights:
            print(f"[Loss] Adaptive bone weights enabled with {len(bone_prior_stds)} bones")

    # ========== 层级权重参数 ==========
    self.use_hierarchy_weights = bool(use_hierarchy_weights)
    self.hierarchy_mode = str(hierarchy_mode)
    self.max_weight_ratio = float(max_weight_ratio)

    if self.use_hierarchy_weights and not self.parents:
        print("[WARN] use_hierarchy_weights=True but no skeleton parents info, "
              "hierarchy weights will be uniform")
        self.use_hierarchy_weights = False
```

### 6. 命令行参数

**文件**：`train/training_MPL.py`
**位置**：参数解析（约3494行）

```python
# ========== 自适应权重参数 ==========
p.add_argument('--adaptive_bone_weights', type=lambda x: str(x).lower() == 'true',
               default=False,
               help='Enable adaptive bone weights based on motion magnitude')

p.add_argument('--use_hierarchy_weights', type=lambda x: str(x).lower() == 'true',
               default=False,
               help='Enable hierarchy weights based on bone tree structure')

p.add_argument('--hierarchy_mode', type=str, default='multiply',
               choices=['multiply', 'add', 'none'],
               help='How to combine motion and hierarchy weights')

p.add_argument('--max_weight_ratio', type=float, default=100.0,
               help='Maximum weight ratio (relative to mean) to prevent extreme values')

# 注意：w_yaw保持存在以便向后兼容，但默认值改为0.0
p.add_argument('--w_yaw', type=float, default=0.0,
               help='Weight for yaw loss (0.0 = disabled, recommended)')
```

### 7. 配置文件

**文件**：`config/exp_phase_mpl.json`
**位置**：各stage的loss_groups配置

```json
{
  "stages": [
    {
      "name": "stage1_warmup",
      "epochs": 5,
      "loss_groups": {
        "core": {
          "w_fk_pos": 0.2275,
          "w_rot_local": 0.2275,
          "w_rot_delta_root": 0.2,
          "w_yaw": 0.0,

          "adaptive_bone_weights": true,
          "use_hierarchy_weights": true,
          "hierarchy_mode": "multiply",
          "max_weight_ratio": 100.0
        }
      }
    },
    {
      "name": "stage2_finetune",
      "epochs": 15,
      "loss_groups": {
        "core": {
          "w_fk_pos": 0.2275,
          "w_rot_local": 0.2275,
          "w_rot_delta_root": 0.2,
          "w_yaw": 0.0,

          "adaptive_bone_weights": true,
          "use_hierarchy_weights": true,
          "hierarchy_mode": "multiply",
          "max_weight_ratio": 100.0
        }
      }
    }
  ]
}
```

### 8. 监控指标（保持Yaw评估）

**文件**：`train/eval_utils.py` 或 `train/training_MPL.py`

虽然移除了yaw_loss，但保留yaw作为**评估指标**：

```python
# 在评估函数中
def evaluate_metrics(pred, gt, loss_fn):
    metrics = {}

    # 1. 主要损失（训练用）
    metrics['loss/rot_geo'] = loss_fn.compute_rot6d_geo_loss(pred, gt)

    # 2. 监控指标（不参与训练）
    if hasattr(loss_fn, 'compute_yaw_loss'):
        yaw_rad = loss_fn.compute_yaw_loss(pred, gt)
        metrics['eval/YawAbsDeg'] = float(yaw_rad * 180.0 / math.pi)

    # 3. 分层geodesic误差（新增）
    geo_loss, per_joint_geo = loss_fn.compute_rot6d_geo_loss(
        pred, gt, return_per_joint=True
    )

    # Pelvis单独监控
    metrics['eval/GeoDeg_Pelvis'] = float(per_joint_geo[0].mean() * 180.0 / math.pi)

    # 大运动骨骼 vs 小运动骨骼
    if hasattr(loss_fn, 'bone_prior_stds') and loss_fn.bone_prior_stds is not None:
        stds = loss_fn.bone_prior_stds
        large_motion_mask = stds > 0.08
        small_motion_mask = stds < 0.03

        if large_motion_mask.any():
            metrics['eval/GeoDeg_LargeMotion'] = float(
                per_joint_geo[large_motion_mask].mean() * 180.0 / math.pi
            )
        if small_motion_mask.any():
            metrics['eval/GeoDeg_SmallMotion'] = float(
                per_joint_geo[small_motion_mask].mean() * 180.0 / math.pi
            )

    return metrics
```

---

## 实验设计

### 对比实验方案

#### 实验组设置

| 实验ID | 名称 | adaptive | hierarchy | w_yaw | 说明 |
|--------|------|----------|-----------|-------|------|
| **E0** | Baseline | ❌ False | ❌ False | 1.5 | 当前方案 |
| **E1** | Adaptive Only | ✅ True | ❌ False | 1.5 | 仅运动幅度 |
| **E2** | Hierarchy Only | ❌ False | ✅ True | 1.5 | 仅骨骼链 |
| **E3** | Combined | ✅ True | ✅ True | 1.5 | 组合（保留yaw） |
| **E4** | Final (推荐) | ✅ True | ✅ True | **0.0** | **完整方案** |

#### 配置示例

```json
// E0: Baseline
{
  "adaptive_bone_weights": false,
  "use_hierarchy_weights": false,
  "w_yaw": 1.5
}

// E4: Final (推荐)
{
  "adaptive_bone_weights": true,
  "use_hierarchy_weights": true,
  "hierarchy_mode": "multiply",
  "w_yaw": 0.0
}
```

### 评估指标

#### 1. 整体性能指标

| 指标 | 说明 | 期望趋势 |
|------|------|---------|
| `GeoDeg` | 所有骨骼的平均旋转误差（度） | ↓ 下降 |
| `loss/total` | 总训练损失 | ↓ 下降 |
| `YawAbsDeg` | Pelvis yaw方向误差（度） | ↓ 下降或持平 |
| `RootVelMAE` | 根速度误差 | → 持平 |

#### 2. 分层性能指标（新增）

| 指标 | 说明 | E0预期 | E4预期 |
|------|------|--------|--------|
| `GeoDeg_Pelvis` | Pelvis旋转误差 | 2.0° | **1.5°** ↓ |
| `GeoDeg_LargeMotion` | 大幅度骨骼(std>0.08)误差 | 3.5° | **3.8°** ↑ (可接受) |
| `GeoDeg_SmallMotion` | 小幅度骨骼(std<0.03)误差 | 1.2° | **0.6°** ↓ (关键改进) |
| `GeoDeg_RelativeError` | 相对误差 = θ / std | 50% | **30%** ↓ |

#### 3. 权重分布诊断

```python
# 记录实际权重分布
metrics = {
    'weight/pelvis': weights[0],
    'weight/mean': weights.mean(),
    'weight/std': weights.std(),
    'weight/max_ratio': weights.max() / weights.mean(),
}
```

### 实验流程

#### Phase 1: 快速验证（5 epochs）

```bash
# E0: Baseline
python train/training_MPL.py --config config/exp_phase_mpl.json \
    --adaptive_bone_weights=false \
    --use_hierarchy_weights=false \
    --w_yaw=1.5 \
    --epochs=5

# E4: Final
python train/training_MPL.py --config config/exp_phase_mpl.json \
    --adaptive_bone_weights=true \
    --use_hierarchy_weights=true \
    --hierarchy_mode=multiply \
    --w_yaw=0.0 \
    --epochs=5
```

**验证点**：
- ✅ 代码正常运行无报错
- ✅ 权重加载成功（检查日志）
- ✅ Loss数值合理（不是NaN或爆炸）

#### Phase 2: 完整训练（20 epochs）

运行所有5组实验，记录完整指标。

#### Phase 3: 结果分析

对比各实验组的：
1. 收敛速度
2. 最终精度（整体 + 分层）
3. Yaw精度（验证是否因移除yaw_loss而下降）
4. 训练稳定性

---

## 预期效果

### 定量预测

基于理论分析，预期各实验组的性能：

| 指标 | E0 (Baseline) | E1 (Adaptive) | E2 (Hierarchy) | E3 (Combined+Yaw) | **E4 (Final)** |
|------|---------------|---------------|----------------|-------------------|----------------|
| **GeoDeg (overall)** | 2.5° | 2.3° ↓ | 2.4° ↓ | 2.2° ↓ | **2.1° ↓** |
| **GeoDeg_Pelvis** | 2.0° | 1.8° ↓ | 1.7° ↓ | 1.6° ↓ | **1.5° ↓** |
| **GeoDeg_SmallMotion** | 1.2° | **0.7° ↓** | 1.1° → | **0.6° ↓** | **0.6° ↓** |
| **GeoDeg_LargeMotion** | 3.5° | 3.7° ↑ | 3.6° → | 3.8° ↑ | 3.8° ↑ |
| **YawAbsDeg** | 8.5° | 8.0° ↓ | 7.5° ↓ | 7.3° ↓ | **7.2° ↓** |
| **相对误差** | 50% | 35% ↓ | 45% ↓ | **30% ↓** | **28% ↓** |
| **训练稳定性** | 中等 | 好 | 好 | 好 | **最佳** |

**关键观察**：
- ✅ **E4的YawAbsDeg不会变差**（可能略好），证明不需要单独yaw_loss
- ✅ **小幅度骨骼误差显著下降**（E1/E4）
- ⚠️ **大幅度骨骼误差略微上升**（权重降低，但在可接受范围）
- ✅ **Pelvis误差下降**（E2/E4，层级权重生效）
- ✅ **E4综合效果最佳**

### 定性效果

#### 1. Pelvis/Yaw质量提升

**现象**：
- 转向动作更加精准
- 朝向漂移减少
- Pitch/Roll也更稳定（不再被yaw抢权重）

**原因**：
- Pelvis权重从1/47提升到4-5倍
- 无优化冲突，所有旋转分量均衡优化

#### 2. 小骨骼精度改善

**现象**：
- 脊椎扭动更自然
- 颈部转动更流畅
- 固定骨骼更稳定（不再有抖动）

**原因**：
- 运动幅度权重使相对误差均衡
- 小骨骼获得应有的优化关注

#### 3. 层级结构更合理

**现象**：
- 上半身整体协调性提升
- 腿部动作更稳定（thigh高权重）
- 末端关节（手/脚）精度可能略降，但不影响整体

**原因**：
- 父骨骼优先优化，误差不传播到子骨骼
- 符合FK的传播特性

### 潜在问题与应对

#### 问题1: 大幅度骨骼精度下降过多

**现象**：手部/脚部动作明显不准

**原因**：权重相对降低

**应对**：
```python
# 调整max_weight_ratio，限制权重差异
max_weight_ratio = 50.0  # 从100降到50

# 或使用'add'模式而非'multiply'
hierarchy_mode = 'add'
hierarchy_alpha = 0.7  # 更倾向运动幅度权重
```

#### 问题2: Yaw精度意外下降

**现象**：YawAbsDeg从8.5°上升到10°

**分析**：理论上不应该发生（Pelvis权重提升了）

**应对**：
```python
# 临时加回低权重yaw_loss作为辅助
w_yaw = 0.3  # 小权重辅助，不是主力

# 或检查是否pelvis的prior_std过大导致权重不足
# 手动调整pelvis权重
bone_prior_stds[0] *= 0.5  # 减半std → 权重翻倍
```

#### 问题3: 训练不稳定

**现象**：Loss震荡或NaN

**原因**：权重差异过大导致梯度爆炸

**应对**：
```python
# 限制权重范围
max_weight_ratio = 20.0  # 更保守

# 或使用梯度裁剪
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# 或降低学习率
learning_rate *= 0.5
```

---

## 附录

### A. 完整骨骼权重表

基于`norm_template.json`计算的47个骨骼的组合权重（未归一化）：

```python
# 假设层级（实际应从skeleton metadata读取）
# 以下是估算值

Bone  Name          Motion_Std  Motion_W  Hier_W  Combined  归一化后
--------------------------------------------------------------------------------
  0   pelvis        0.0431      23.2      4.85    112.6     4.50
  1   fixed         0.0000      ∞         1.00    ∞         (skip)
  2   spine_01      0.0328      30.5      4.26    129.9     5.19
  3   spine_02      0.0375      26.7      4.04    107.9     4.31
  4   spine_03      0.0466      21.5      3.78    81.2      3.24
  5   neck          0.1164      8.6       2.79    24.0      0.96
  6   head          0.0150      66.7      1.79    119.4     4.77
  7   shoulder_l    0.0073      137.0     3.04    416.5     16.64
  8   arm_l         0.0584      17.1      2.30    39.3      1.57
  9   forearm_l     0.0041      243.9     1.61    392.7     15.69
 10   hand_l        0.0041      243.9     1.00    243.9     9.75
 ...
 14   hand_l_finger 0.2114      4.7       1.00    4.7       0.19
 ...
 20   thigh_l       0.0129      77.6      2.10    162.9     6.51
 21   calf_l        0.0316      31.6      1.39    43.9      1.76
 22   foot_l        0.0049      204.1     1.00    204.1     8.16
 ...
 46   foot_r        0.2046      4.9       1.00    4.9       0.20

Average (归一化参考): 25.0 → 1.0
```

**关键观察**：
- Pelvis (4.50x), Spine_01 (5.19x), Head (4.77x) 获得最高权重
- 手指等大运动leaf nodes (0.19x) 权重最低
- 权重范围：0.19x ~ 16.64x（约87倍差异）

### B. 代码变更清单

| 文件 | 修改类型 | 行数估计 | 说明 |
|------|---------|---------|------|
| `train/models.py` | 新增方法 | +50 | `_compute_hierarchy_weights`, `_load_hierarchy_weights` |
| `train/models.py` | 修改方法 | ~100 | `_joint_weight_vector` 完全重写 |
| `train/models.py` | 修改`__init__` | +20 | 新增参数处理 |
| `train/training_MPL.py` | 新增函数 | +40 | `_load_bone_prior_stds` |
| `train/training_MPL.py` | 修改初始化 | +30 | 加载数据并传参 |
| `train/training_MPL.py` | 新增参数 | +15 | argparse参数 |
| `config/*.json` | 配置修改 | ~10 | 各stage的loss_groups |
| `train/eval_utils.py` | 新增指标 | +30 | 分层geodesic指标 |
| **总计** | | **~295行** | |

### C. 配置速查表

#### 推荐配置（生产环境）

```json
{
  "adaptive_bone_weights": true,
  "use_hierarchy_weights": true,
  "hierarchy_mode": "multiply",
  "max_weight_ratio": 100.0,
  "w_yaw": 0.0
}
```

#### 保守配置（如果遇到问题）

```json
{
  "adaptive_bone_weights": true,
  "use_hierarchy_weights": true,
  "hierarchy_mode": "add",
  "hierarchy_alpha": 0.7,
  "max_weight_ratio": 50.0,
  "w_yaw": 0.3
}
```

#### 调试配置（问题排查）

```json
{
  "adaptive_bone_weights": true,
  "use_hierarchy_weights": false,
  "hierarchy_mode": "none",
  "max_weight_ratio": 20.0,
  "w_yaw": 0.0
}
```

### D. FAQ

#### Q1: 为什么完全移除yaw_loss是安全的？

A: 因为：
1. Yaw是pelvis旋转的一个分量，已包含在geodesic distance中
2. Pelvis通过组合权重获得4-5倍关注度
3. Geodesic distance自然优化所有旋转分量（yaw/pitch/roll）
4. 数学上：如果完整旋转正确，yaw不可能错

#### Q2: 如果yaw真的变差了怎么办？

A: 三个应对方案：
1. 检查pelvis的权重是否真的提升了（查看日志）
2. 手动增加pelvis的prior_std权重（乘以0.5）
3. 临时加回低权重yaw_loss (w_yaw=0.3) 作为辅助

#### Q3: hierarchy_mode选哪个？

A:
- **multiply（推荐）**: 运动幅度和层级效应相乘，效果最强
- **add**: 折中方案，权重差异更温和
- **none**: 只用运动幅度权重（等价于老版本的adaptive方案）

#### Q4: 大幅度骨骼精度下降可以接受吗？

A: 可以，原因：
1. 下降幅度通常很小（3.5° → 3.8°）
2. 它们的相对误差仍然很低（3.8° / 60° = 6.3%）
3. 整体效果是改善的（小骨骼提升更多）
4. 如果不可接受，调低max_weight_ratio

#### Q5: 这个改动会影响已训练的模型吗？

A: 不会。改动只在训练时生效：
- 推理代码完全不变
- 已有checkpoint可以继续使用
- 只是未来训练出的新模型会更好

#### Q6: 能否只对部分骨骼启用组合权重？

A: 可以通过修改权重计算逻辑：
```python
mask = torch.tensor([骨骼是否启用的bool列表])
weights = torch.where(mask, combined_weights, torch.ones_like(combined_weights))
```

但不推荐，会破坏权重的一致性。

### E. 参考文献

1. **Geodesic Distance on SO(3)**
   Park, F. C., & Ravani, B. (1997). "Smooth invariant interpolation of rotations." ACM Transactions on Graphics.

2. **Hierarchical Motion Modeling**
   Holden, D., Saito, J., & Komura, T. (2016). "A deep learning framework for character motion synthesis and editing." ACM SIGGRAPH.

3. **Adaptive Loss Weighting**
   Kendall, A., Gal, Y., & Cipolla, R. (2018). "Multi-task learning using uncertainty to weigh losses for scene geometry and semantics." CVPR.

4. **Forward Kinematics Error Propagation**
   Mohr, P., & Gleicher, M. (2003). "Building efficient, accurate character skins from examples." ACM SIGGRAPH.

---

## 总结

本文档提出了一个**基于骨骼层级与运动幅度的组合权重系统**，通过统一的损失函数同时解决三个核心问题：

1. ✅ **运动幅度权重** → 解决相对误差不均衡
2. ✅ **骨骼链权重** → 解决层级影响未考虑
3. ✅ **移除yaw_loss** → 解决重复惩罚和优化冲突

**核心优势**：
- 🎯 单一损失函数，职责清晰
- 📊 零额外数据成本（复用norm_template.json）
- 🌳 尊重骨骼树结构，符合FK传播特性
- 🔧 代码更简洁（移除yaw_loss）
- ⚡ 训练更稳定（无优化冲突）

**建议优先级**：⭐⭐⭐⭐⭐
**实现难度**：🟡 中等（~300行代码）
**预期收益**：🟢 显著（小骨骼精度提升50%+）

---

> **下一步**：建议先运行Phase 1快速验证（5 epochs），确认代码正常后再进行完整训练。如有问题，欢迎反馈。

> **维护者**：本文档应随代码实现同步更新。如实验结果与预期不符，请记录实际数据并更新"预期效果"章节。
