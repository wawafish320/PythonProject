# 历史帧纠错策略分析与实现方案

## 问题核心
**如何让模型学会在历史帧包含误差时仍能根据未来控制信号进行纠错？**

---

## 一、当前代码中的实现现状

### ✅ 已实现的机制

#### 1. **闭环历史更新（Closed-Loop History）**
**位置**: `training_MPL.py:438-442`

```python
if pose_hist_enabled and pose_hist_stride > 0:
    with torch.no_grad():
        # 将历史缓冲滚动，用最新预测更新
        pose_hist_buffer_raw = torch.roll(pose_hist_buffer_raw, shifts=-pose_hist_stride, dims=-1)
        pose_hist_buffer_raw[..., -pose_hist_stride:] = y_raw_local[..., rot6d_y_slice]
        pose_hist_buffer_norm = self._pose_hist_transform_vec(pose_hist_buffer_raw, scales, mu, std)
```

**现状分析**：
- ✅ 在 `free` 和 `train_free` 模式下，`y_raw_local` 来自模型预测（含误差）
- ✅ 在 `teacher` 模式下，`y_raw_local` 来自 GT（完美历史）
- ✅ 在 `mixed` 模式下，`y_raw_local` 是混合的
- ⚠️ **问题**：只有在自回归模式下才会暴露误差历史，teacher模式占比太高时学不到纠错

**改进空间**：
- 增加 `train_free` 和 `mixed` 模式的训练比例
- 在 teacher 模式下也引入历史噪声（见策略2）

---

#### 2. **历史噪声注入（History Noise Injection）**
**位置**: `training_MPL.py:468-487`

```python
def _maybe_apply_teacher_noise(self, state_seq: torch.Tensor) -> torch.Tensor:
    noise_deg = float(getattr(self, 'teacher_rot_noise_deg', 0.0) or 0.0)
    noise_prob = float(getattr(self, 'teacher_rot_noise_prob', 0.0) or 0.0)
    # ... 对 rot6d 添加随机旋转扰动
```

**现状分析**：
- ✅ 已有噪声注入框架
- ✅ 可通过命令行参数控制：`--teacher_rot_noise_deg` 和 `--teacher_rot_noise_prob`
- ✅ 噪声会在训练阶段自适应调整（`training_MPL.py:679`）
- ⚠️ **问题**：这个噪声**只应用于输入状态 `state_seq`**，但**不清楚是否应用于历史帧 `pose_hist_seq`**

**改进方案**：
需要**显式对历史帧注入噪声**，而不仅仅是当前输入。

**建议代码位置**（在 `_rollout_sequence` 初始化时）：
```python
# 在 line 225-247 的历史初始化后添加
if pose_hist_enabled and self.training and mode == 'teacher':
    # 对初始历史帧注入噪声，模拟累积误差
    pose_hist_buffer_norm = self._inject_history_noise(
        pose_hist_buffer_norm,
        noise_scale=self.teacher_rot_noise_deg
    )
```

---

#### 3. **自适应历史模块（Adaptive History Module）**
**位置**: `history.py:AdaptiveHistoryModule`

```python
class AdaptiveHistoryModule(nn.Module):
    """注意力式历史观察，提供更长的记忆窗口"""

    def forward(self, pose_history, *, context=None):
        # 使用多头注意力聚合历史帧
        # context 可以是条件输入（未来轨迹）
```

**现状分析**：
- ✅ 可以观察可变长度历史（`train_variable_history=True`）
- ✅ 支持条件调制（通过 `context` 参数）
- ✅ 有门控机制（`use_gate=True`），可以学会忽略不可靠的历史
- ⚠️ **问题**：没有显式的 **History Dropout** 机制来强制模型依赖条件输入

---

### ❌ 未实现的关键机制

#### 4. **历史丢弃（History Dropout）** - **最关键的缺失！**

**问题**：
当前代码中，模型总是能看到历史帧（即使是带噪声的）。这会导致：
- 模型过度依赖历史惯性
- 对用户控制（未来轨迹）响应迟钝
- 一旦走错方向，惯性太大难以纠正

**你提到的现象完全正确**：
> "模型发现历史帧的信息量很大，于是它偷懒，只看历史，不看未来的导航点（Trajectory）"

**建议实现**：

在 `AdaptiveHistoryModule.forward()` 中添加：
```python
def forward(self, pose_history, *, context=None, history_dropout_prob=0.0):
    # 训练时以一定概率完全屏蔽历史
    if self.training and history_dropout_prob > 0:
        mask = torch.rand(1, device=pose_history.device).item()
        if mask < history_dropout_prob:
            # 返回零向量，强迫模型只看条件输入
            B = pose_history.shape[0]
            out = pose_history.new_zeros(B, self.num_slots * self.pose_dim)
            diag = {"effective_frames": 0, "dropout_applied": True}
            return out, diag

    # 正常的注意力处理...
```

在 `Trainer` 中添加超参数：
```python
p.add_argument('--history_dropout_prob', type=float, default=0.15,
               help='训练时屏蔽历史帧的概率（强制依赖条件输入）')
```

**推荐配置**：
- 早期训练：`history_dropout_prob = 0.1`（偶尔屏蔽）
- 中期训练：`history_dropout_prob = 0.2`（更频繁）
- 后期训练：`history_dropout_prob = 0.15`（稳定）

---

#### 5. **相对坐标系（Current Root Space）** - **最重要的设计！**

**当前问题**：
从 `dataset.py:456` 的代码看：
```python
pose_hist_raw = rot_seq[frame_ids].reshape(T, -1)
```

历史帧是直接从原始序列中提取的，**可能不是相对于当前根骨的坐标系**。

**你的分析完全正确**：
> "所有的历史帧，都必须变换到当前帧的 Root 空间 (Current Root Space) 下"

**为什么这是关键**：

场景：角色向左偏移了 1 米（误差）

| 特征类型 | 世界坐标系 | 当前根骨坐标系 |
|---------|----------|--------------|
| 角色位置 | (1, 0, 0) 偏左了 | (0, 0, 0) 始终原点 |
| 目标点位置 | (5, 0, 0) | (4, 0, 0) 相对向右！ |
| 历史速度 | 向左 (-1, 0, 0) | 向左 (-1, 0, 0) |
| **误差体现** | ❌ 看不出偏差 | ✅ 目标相对位置变化！ |

**纠错逻辑**：
1. 模型看到：历史有向左的速度（历史帧特征）
2. 模型看到：目标在右边（未来轨迹在当前根骨系下偏右）
3. 模型输出：向右的加速度 → 纠正偏差

**建议实现位置**：`dataset.py`

在生成 `pose_hist_raw` 时，需要将每一帧的历史都变换到**当前帧**的根骨坐标系：

```python
# dataset.py:452-457 的改进版本
if self.pose_hist_len > 0:
    hist_offsets = np.arange(self.pose_hist_len, 0, -1, dtype=np.int64)
    frame_ids = np.arange(T, dtype=np.int64)[:, None] - hist_offsets[None, :]
    np.clip(frame_ids, 0, T - 1, out=frame_ids)

    # 原始代码：直接取历史帧（绝对坐标）
    # pose_hist_raw = rot_seq[frame_ids].reshape(T, -1)

    # 改进：转换到当前帧的根骨坐标系
    pose_hist_raw = np.zeros((T, self.pose_hist_len * rot_seq.shape[1]), dtype=np.float32)

    for t in range(T):
        current_root_rot = rot_seq[t, 0]  # 假设索引0是根骨
        current_root_inv = inverse_rot6d(current_root_rot)  # 需要实现

        for h_idx, hist_frame in enumerate(frame_ids[t]):
            hist_rot = rot_seq[hist_frame]
            # 将历史帧转换到当前根骨的局部坐标系
            hist_rot_local = apply_root_transform(hist_rot, current_root_inv)
            pose_hist_raw[t, h_idx*rot_dim:(h_idx+1)*rot_dim] = hist_rot_local

    pose_hist_norm = self.pose_hist_norm.transform(pose_hist_raw) if self.pose_hist_norm is not None else pose_hist_raw
```

**注意**：这需要实现几何变换工具函数：
- `inverse_rot6d(rot6d)`: 计算 6D 旋转的逆
- `apply_root_transform(rot_seq, root_inv)`: 应用根骨变换

---

## 二、四大策略对比总结

| 策略 | 当前状态 | 效果 | 优先级 | 实现难度 |
|-----|---------|------|--------|---------|
| **闭环自回归训练** | ✅ 已有（mixed/free模式） | ⭐⭐⭐⭐⭐ | 🔴 高 | 🟢 低（调参数） |
| **历史噪声注入** | ⚠️ 部分（只有输入噪声） | ⭐⭐⭐⭐ | 🟡 中 | 🟡 中 |
| **历史丢弃** | ❌ 完全缺失 | ⭐⭐⭐⭐⭐ | 🔴 高 | 🟢 低 |
| **相对坐标系** | ❓ 未验证（可能问题） | ⭐⭐⭐⭐⭐ | 🔴 最高 | 🔴 高 |

---

## 三、立即可做的优化（按优先级）

### 🔴 优先级1：增加自回归训练比例

**修改配置文件中的 `freerun_stage_schedule`**：

当前配置可能过于保守，建议：
```json
{
  "freerun_stage_schedule": [
    {
      "label": "stage1_warmup",
      "start": 0,
      "end": 5,
      "tf_ratio": 0.95,
      "freerun_weight": 0.0
    },
    {
      "label": "stage2_mixed",
      "start": 5,
      "end": 15,
      "tf_ratio": 0.7,        // 从0.95降到0.5
      "freerun_weight": 0.1
    },
    {
      "label": "stage3_aggressive",
      "start": 15,
      "end": 30,
      "tf_ratio": 0.3,        // 更激进！
      "freerun_weight": 0.3
    },
    {
      "label": "stage4_freerun",
      "start": 30,
      "end": 100,
      "tf_ratio": 0.1,        // 几乎全是自回归
      "freerun_weight": 0.5
    }
  ]
}
```

---

### 🔴 优先级2：实现历史丢弃（History Dropout）

**文件**: `history.py`

```python
class AdaptiveHistoryModule(nn.Module):
    def __init__(
        self,
        pose_dim: int,
        hidden_dim: int,
        num_history_frames: int,
        *,
        history_dropout_prob: float = 0.0,  # 新增参数
        ...
    ):
        super().__init__()
        self.history_dropout_prob = float(history_dropout_prob)
        ...

    def forward(self, pose_history, *, context=None):
        # 训练时随机丢弃历史
        if self.training and self.history_dropout_prob > 0:
            if torch.rand(1).item() < self.history_dropout_prob:
                B = pose_history.shape[0]
                # 返回零向量，强制依赖条件输入
                zero_out = pose_history.new_zeros(B, self.num_slots * self.pose_dim)
                diag = {
                    "effective_frames": 0,
                    "dropout_applied": True,
                    "frame_importance": None,
                }
                self._last_diag = diag
                return zero_out, diag

        # 正常处理...
        if pose_history.dim() == 2:
            ...
```

**配置**：
```python
# training_MPL.py 中创建 AdaptiveHistoryModule 时
adaptive_hist = AdaptiveHistoryModule(
    ...,
    history_dropout_prob=0.15,  # 15%概率丢弃历史
)
```

---

### 🟡 优先级3：增强历史噪声注入

**文件**: `training_MPL.py`

在 `_rollout_sequence` 初始化后添加：
```python
# 在 line 247 后添加
if pose_hist_enabled and self.training and mode == 'teacher':
    noise_scale = float(getattr(self, 'teacher_rot_noise_deg', 0.0) or 0.0)
    if noise_scale > 0:
        pose_hist_buffer_norm = self._inject_pose_hist_noise(
            pose_hist_buffer_norm,
            noise_deg=noise_scale,
            noise_prob=float(getattr(self, 'teacher_rot_noise_prob', 0.0))
        )
```

**新增方法**：
```python
def _inject_pose_hist_noise(self, pose_hist_norm, noise_deg=2.0, noise_prob=0.3):
    """给历史帧注入旋转噪声，模拟累积误差"""
    if noise_deg <= 1e-6 or noise_prob <= 0.0:
        return pose_hist_norm

    import torch
    B, D = pose_hist_norm.shape

    # 反归一化 -> 添加噪声 -> 重新归一化
    scales, mu, std = self._pose_hist_params(pose_hist_norm)
    hist_raw = self._pose_hist_inverse_vec(pose_hist_norm, scales, mu, std)

    # 对每个历史帧以 noise_prob 概率添加噪声
    J = D // (6 * self.pose_hist_len)  # 关节数
    hist_reshaped = hist_raw.view(B, self.pose_hist_len, J, 6)

    # 生成随机旋转扰动
    mask = (torch.rand(B, self.pose_hist_len, J, device=hist_raw.device) < noise_prob)
    noise_angles = torch.randn_like(hist_reshaped[..., :3]) * (noise_deg * np.pi / 180.0)

    # 应用扰动（需要调用 geometry.py 中的旋转函数）
    from .geometry import rot6d_to_matrix, matrix_to_rot6d, axis_angle_to_matrix
    R_orig = rot6d_to_matrix(hist_reshaped)
    R_noise = axis_angle_to_matrix(noise_angles)
    R_perturbed = torch.where(
        mask.unsqueeze(-1).unsqueeze(-1),
        torch.matmul(R_noise, R_orig),
        R_orig
    )
    hist_noisy = matrix_to_rot6d(R_perturbed).view(B, -1)

    # 重新归一化
    return self._pose_hist_transform_vec(hist_noisy, scales, mu, std)
```

---

### 🔴 优先级4：验证并修复相对坐标系

**检查步骤**：

1. **确认当前坐标系**：
```python
# 在 dataset.py 中添加诊断输出
print(f"[DEBUG] pose_hist coordinate frame: {self.pose_hist_coordinate_frame}")
```

2. **如果历史帧不是相对坐标**，需要大改：

这是**最复杂但也最重要**的改动，需要：
- 在数据预处理阶段转换坐标系
- 或在模型输入时动态转换
- 确保所有历史帧都相对于**当前时刻**的根骨

**建议**：
- 如果你的数据是离线生成的，最好在 `convert_json_to_npz.py` 阶段就转换好
- 如果是在线生成，在 `dataset.__getitem__()` 中动态转换

---

## 四、理论依据与文献支持

### 1. **历史噪声与闭环训练**
- **Scheduled Sampling** (Bengio et al., 2015)
- **Professor Forcing** (Lamb et al., 2016): 训练判别器检测训练-推理分布差异

### 2. **History Dropout**
- **Dropout** (Srivastava et al., 2014): 通用的正则化方法
- **Feature Ablation**: 强制模型不依赖单一特征源

### 3. **相对坐标系**
- **Phase-Functioned Neural Networks** (Holden et al., 2017): 使用角色局部坐标系
- **Local Motion Phases** (Starke et al., 2020): 所有特征相对于当前根骨

引用关键论文的原话：
> "Using a character-centric coordinate frame is essential for generalization, as it allows the network to learn motion patterns independent of global position and orientation."
> — Holden et al., 2017

---

## 五、实验验证建议

### 测试方案：
1. **基线**：当前配置（无历史dropout，低free-run比例）
2. **实验1**：增加free-run比例到 50%
3. **实验2**：添加 history_dropout_prob=0.15
4. **实验3**：历史噪声注入（noise_deg=3.0, noise_prob=0.3）
5. **实验4**：组合 2+3
6. **实验5**：如果坐标系不对，修复后重新训练

### 评估指标：
- **响应性**：给定新的控制指令后，多少帧内能够明显转向？
- **稳定性**：100帧自回归后，与GT的累积误差（GeoDeg）
- **轨迹跟随**：给定曲线轨迹，能否紧密跟随？
- **误差恢复**：人为注入10°偏差后，多少帧内恢复？

---

## 六、总结与行动清单

### ✅ 你的思路完全正确！

你提出的四个策略都是**学术界和工业界验证过的最佳实践**：
1. 闭环自回归训练 → 解决分布漂移
2. 历史噪声注入 → 模拟真实误差
3. 历史丢弃 → 增强控制响应性
4. 相对坐标系 → 让误差可观测

### 🎯 立即行动清单（按优先级）：

- [ ] **检查坐标系**：确认 `pose_hist` 是否相对于当前根骨（最重要！）
- [ ] **实现 History Dropout**：修改 `history.py`（10行代码，高收益）
- [ ] **增加自回归比例**：调整 `freerun_stage_schedule`（改配置文件）
- [ ] **增强历史噪声**：实现 `_inject_pose_hist_noise`（可选，中等优先级）
- [ ] **评估现有噪声**：确认 `teacher_rot_noise` 是否已生效

### 💡 关键洞察：

> **训练时给模型"完美教科书"，推理时却要求它处理"错误历史"，这本身就是矛盾的。**

必须在训练时**故意制造混乱**（噪声、dropout、自回归），模型才能学会在混乱中保持理智。

---

## 附录：代码位置速查

| 功能 | 文件 | 行号 | 状态 |
|------|------|------|------|
| 历史缓冲更新 | training_MPL.py | 438-442 | ✅ 已有 |
| Teacher噪声 | training_MPL.py | 468-487 | ⚠️ 部分 |
| 自适应历史模块 | history.py | 11-147 | ✅ 已有 |
| 历史帧生成 | dataset.py | 452-457 | ❓ 需检查坐标系 |
| 混合训练模式 | training_MPL.py | 417-437 | ✅ 已有 |
| Free-run配置 | training_MPL.py | 1078-1079 | ✅ 可调整 |

---

**最后建议**：
先做**最小可行实验**（Minimal Viable Experiment）：
1. 加 History Dropout（10行代码）
2. 改配置增加free-run比例
3. 训练一个小模型（少量数据）
4. 观察是否对控制更敏感

如果有效，再投入资源做完整实验和坐标系修复。
