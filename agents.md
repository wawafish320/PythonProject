# Motion Prediction Training System - Agent Guide

## 概述

本文档描述了运动预测训练系统的核心组件、设计模式和最佳实践。本项目实现了一个基于深度学习的人体动作生成系统，包含数据归一化、模型训练、损失计算和自适应调度等模块。

## 核心组件

### 1. 数据处理 Agents

#### 1.1 LayoutCenter
**职责**: 单一数据源，管理归一化参数和数据布局

**核心功能**:
- 从 bundle JSON 加载 μ/σ 统计量
- 严格验证数据维度和布局
- 提供统一的数据规格（state_layout, output_layout）

**使用示例**:
```python
# 加载并验证布局
center = LayoutCenter(bundle_path="configs/bundle.json")
center.strict_validate(Dx=512, Dy=384)

# 应用到数据集
center.apply_to_dataset(dataset)

# 创建 Normalizer（自动注入到 Trainer）
normalizer = center.apply_to_trainer(trainer, dataset=dataset)
```

**最佳实践**:
- ✅ 在训练开始前一次性加载和验证
- ✅ 使用 `strict_validate()` 确保维度匹配
- ✅ 避免运行时修改布局参数

---

#### 1.2 DataNormalizer
**职责**: 执行数据归一化和反归一化操作

**核心功能**:
- 标准化输入状态 (X) 和输出动作 (Y)
- 特殊处理：Yaw（绝对值）、RootVelocity（tanh）、AngularVelocity（tanh/standardize）
- 自动缓存设备/类型转换的张量

**使用示例**:
```python
# 归一化
x_normalized = normalizer.norm(x_raw)

# 反归一化
x_raw = normalizer.denorm_x(x_normalized)
y_raw = normalizer.denorm_y(y_normalized)

# Y→X 映射（用于自回归）
x_next = normalizer.apply_y_to_x_map(x_current, y_pred)
```

**性能优化**:
- 内置 `_match_tensor` 缓存机制，避免重复转换
- 自动处理设备（CPU/GPU）和数据类型（float32/float16）

---

### 2. 模型 Agents

#### 2.1 MotionEncoder
**职责**: 无状态帧级编码器，用于预训练特征提取

**架构**:
```
Input [B, T, D_in]
  ↓
MLP (num_layers × [Linear → GELU → Dropout])
  ↓
Hidden [B, T, hidden_dim]
  ↓ (optional)
Summary [B, z_dim]
```

**使用示例**:
```python
encoder = MotionEncoder(
    input_dim=128,
    hidden_dim=256,
    z_dim=64,          # 摘要维度（可选）
    num_layers=3,
    dropout=0.1
)

# 获取隐藏状态
hidden = encoder(x)  # [B, T, 256]

# 获取摘要向量
summary, hidden = encoder(x, return_summary=True)  # [B, 64], [B, T, 256]
```

**集成模式**:
- 可冻结后作为 `EventMotionModel` 的特征提取器
- 支持从 checkpoint 加载预训练权重

---

#### 2.2 EventMotionModel
**职责**: 主动作生成模型，基于状态和条件生成动作增量

**架构**:
```
State + Condition
  ↓
Shared Encoder (MLP + Residual)
  ↓
Self-Attention (with FiLM conditioning)
  ↓
Motion Head → Delta Output
  ↓ (optional)
Frozen Encoder → Period Prediction
```

**使用示例**:
```python
model = EventMotionModel(
    in_state_dim=512,
    out_motion_dim=384,
    cond_dim=64,
    period_dim=8,        # 周期预测维度
    hidden_dim=256,
    num_heads=4,
    contact_dim=32,      # 接触力输入
    angvel_dim=48,       # 角速度输入
    pose_hist_dim=128    # 历史姿态输入
)

# 前向传播
result = model(
    state=state,         # [B, T, 512]
    cond=cond,          # [B, T, 64]
    contacts=contacts,   # [B, T, 32]
    angvel=angvel,      # [B, T, 48]
    pose_history=hist   # [B, T, 128]
)

# 输出
delta = result['delta']          # 动作增量
period_pred = result.get('period_pred')  # 周期预测（可选）
attn = result['attn']           # 注意力权重
```

**附加编码器**:
```python
# 加载并冻结预训练编码器
model.attach_motion_encoder(
    bundle="pretrained/encoder.pt",
    map_location='cuda'
)
```

---

### 3. 损失函数 Agents

#### 3.1 MotionJointLoss
**职责**: 计算多组分损失（几何、正交、FK、局部旋转等）

**核心损失项**:
| 损失项 | 描述 | 权重参数 | 分组 |
|--------|------|----------|------|
| rot_geo | Rot6D 几何距离（geodesic） | w_rot_geo | core |
| rot_delta | Rot6D 增量损失 | w_rot_delta | core |
| rot_ortho | Rot6D 正交性约束 | w_rot_ortho | core |
| rot_log | Rot6D 对数映射损失 | w_rot_log | aux |
| fk_pos | FK 位置损失 | w_fk_pos | core |
| rot_local | 局部旋转损失 | w_rot_local | core |
| cond_yaw | 条件方向损失 | w_cond_yaw | core |
| limb_geo | 肢体几何约束 | w_limb_geo | aux |
| attn | 注意力正则化 | w_attn_reg | aux |

**使用示例**:
```python
loss_fn = MotionJointLoss(
    w_rot_geo=1.0,
    w_rot_delta=1.0,
    w_rot_ortho=0.1,
    w_fk_pos=0.5,
    w_limb_geo=0.01,
    output_layout=dataset.output_layout,
    fps=60.0,
    rot6d_spec={'columns': ['X', 'Z']},
    meta={'skeleton': skeleton_config}
)

# 计算损失
loss, stats = loss_fn(
    pred_motion=model_output,
    gt_motion=ground_truth,
    attn_weights=attn,
    batch=batch_data
)
```

**自适应损失**:
```python
# 获取自适应调度 payload
payload = loss_fn.adaptive_loss_payload()
if payload:
    core_loss = payload['core_loss']
    component_losses = payload['losses']     # {'fk_pos': ..., 'rot_delta': ...}
    component_weights = payload['weights']
```

---

#### 3.2 AdaptiveLossManager
**职责**: 分离自适应损失追踪与调度逻辑

**核心功能**:
- 注册可自适应调整的损失分量
- 累积损失分组统计（core/aux/long）
- 提供自适应调度所需的 payload

**使用示例**:
```python
# 创建管理器
manager = AdaptiveLossManager(
    adaptive_terms=("fk_pos", "rot_local", "rot_delta"),
    group_alias={
        'rot_geo': 'core',
        'limb_geo': 'aux',
        'rot_delta': 'core'
    }
)

# 每个 forward 开始时重置
manager.reset()

# 注册损失分量
manager.register('rot_delta', l_delta, weight=1.0)
manager.accumulate_group('rot_delta', l_delta, weight=1.0)

# 计算核心损失
manager.finalize(total_loss)

# 获取统计
payload = manager.get_payload()
stats = manager.get_group_stats()  # {'loss_group/core': 0.5, ...}
```

**设计优势**:
- ✅ 职责分离：损失计算 vs 自适应追踪
- ✅ 易于扩展：添加新损失项只需修改一处
- ✅ 可复用：可用于其他损失函数

---

### 4. 训练 Agents

#### 4.1 Trainer
**职责**: 管理训练循环、优化器、学习率调度和自回归展开

**核心功能**:
- Teacher forcing / Free-run 混合训练
- 自适应超参数调度
- Lookahead 损失（长期预测）
- AMP（混合精度训练）
- Gradient accumulation

**使用示例**:
```python
trainer = Trainer(
    model=model,
    loss_fn=loss_fn,
    lr=1e-4,
    grad_clip=1.0,
    weight_decay=0.01,
    tf_warmup_steps=5000,    # Teacher forcing 预热
    use_amp=True,            # 启用混合精度
    accum_steps=4            # 梯度累积
)

# 训练一个 epoch
metrics = trainer.train_epoch(
    train_loader,
    epoch=0,
    max_grad_norm=1.0
)

# 评估
eval_metrics = trainer.evaluate(
    val_loader,
    mode='freerun',          # 'teacher' / 'freerun' / 'mixed'
    rollout_steps=128
)
```

**自回归展开**:
```python
# Roll-out 序列预测
outputs = trainer._rollout_sequence(
    state_seq=state,         # [B, T, Dx]
    cond_seq=cond,          # [B, T, Dcond]
    mode='mixed',           # 混合模式
    tf_ratio=0.8            # Teacher forcing 比例
)
```

---

## 工具函数

### build_mlp
**职责**: 通用 MLP 构建器，避免重复定义

**签名**:
```python
def build_mlp(
    input_dim: int,
    output_dim: int,
    hidden_dims: Optional[List[int]] = None,
    *,
    activation: str = 'relu',      # 'relu' / 'gelu' / 'tanh'
    use_layer_norm: bool = False,  # 是否使用 LayerNorm
    dropout: float = 0.0,          # Dropout 比例
    final_activation: bool = False # 最后一层是否激活
) -> nn.Sequential
```

**使用示例**:
```python
# 简单两层 MLP
mlp = build_mlp(128, 64, [256], activation='relu', dropout=0.1)

# 三层 MLP with LayerNorm
mlp = build_mlp(
    128, 64,
    hidden_dims=[256, 256],
    activation='gelu',
    use_layer_norm=True,
    dropout=0.1
)

# 直接连接（无中间层）
mlp = build_mlp(128, 64, activation='relu')
```

---

## 数据流

### 训练流程

```
1. 数据加载
   └─ LayoutCenter.apply_to_dataset(dataset)

2. 初始化
   ├─ normalizer = LayoutCenter.apply_to_trainer(trainer, dataset)
   └─ loss_fn.normalizer = normalizer

3. 训练循环
   for epoch in epochs:
       for batch in dataloader:
           ├─ x_norm = normalizer.norm(x_raw)
           ├─ output = model(x_norm, cond)
           ├─ loss, stats = loss_fn(output, gt)
           ├─ optimizer.step()
           └─ log(stats)

4. 自适应调度（可选）
   └─ payload = loss_fn.adaptive_loss_payload()
       └─ adjust_weights(payload)
```

### 推理流程

```
1. 加载模型
   ├─ model.load_state_dict(checkpoint)
   └─ normalizer = create_normalizer(bundle)

2. 自回归生成
   state = initial_state
   for t in range(T):
       ├─ state_norm = normalizer.norm(state)
       ├─ output = model(state_norm, cond[t])
       ├─ delta_raw = normalizer.denorm_y(output['delta'])
       ├─ y_raw = compose_delta(state_raw, delta_raw)
       └─ state = normalizer.apply_y_to_x_map(state_raw, y_raw)
```

---

## 最佳实践

### 1. 数据归一化
✅ **DO**:
- 始终使用 `LayoutCenter` 作为单一数据源
- 在训练前调用 `strict_validate()` 确保维度匹配
- 利用 `DataNormalizer` 的缓存机制（避免重复转换）

❌ **DON'T**:
- 不要手动计算 μ/σ，使用预计算的 bundle
- 不要在运行时修改归一化参数
- 不要绕过 normalizer 直接操作原始数据

### 2. 模型训练
✅ **DO**:
- 使用 Teacher forcing warmup 稳定初期训练
- 启用 AMP 节省显存和加速训练
- 使用 gradient accumulation 模拟大 batch size
- 定期保存 checkpoint 和 best model

❌ **DON'T**:
- 不要在 free-run 模式下直接训练（容易不稳定）
- 不要跳过梯度裁剪（避免梯度爆炸）
- 不要忽略 NaN/Inf 检查

### 3. 损失函数
✅ **DO**:
- 使用分组权重（core/aux）区分主要和辅助损失
- 启用自适应损失调度（如果适用）
- 监控各损失分量的统计（通过 stats）

❌ **DON'T**:
- 不要一次性启用所有损失项（逐步增加）
- 不要设置过大的权重（导致训练不稳定）
- 不要忽略 limb_geo 警告（可能表示姿态异常）

### 4. 调试
✅ **DO**:
- 使用 `_run_one_epoch` 的诊断输出检查数据分布
- 检查 `loss_group` 统计确认损失平衡
- 可视化 attention 权重理解模型关注点
- 使用 tensorboard 或 wandb 记录训练曲线

❌ **DON'T**:
- 不要忽略 `[FATAL]` 或 `[WARN]` 日志
- 不要在出现 NaN 时继续训练
- 不要跳过验证集评估

---

## 常见问题

### Q1: 如何添加新的损失项？
```python
# 1. 在 MotionJointLoss.__init__ 中添加权重参数
def __init__(self, ..., w_my_loss: float = 0.0):
    self.w_my_loss = float(w_my_loss)

# 2. 实现损失计算方法
def compute_my_loss(self, pred, gt):
    return F.mse_loss(pred, gt)

# 3. 在 forward 中调用
if self.w_my_loss > 0:
    my_loss = self.compute_my_loss(pm, gm)
    loss = loss + self.w_my_loss * my_loss
    self.adaptive_mgr.accumulate_group('my_loss', my_loss, self.w_my_loss)
    stats['my_loss'] = float(my_loss.detach().cpu())

# 4. （可选）添加到自适应追踪
self.adaptive_mgr.register('my_loss', my_loss, self.w_my_loss)
```

### Q2: 如何更改归一化方式？
修改 `DataNormalizer.norm()` 和 `denorm_x()` 方法：
```python
# 例如：添加新的特殊处理
if isinstance(self.my_slice, slice):
    x_proc[..., self.my_slice] = custom_transform(x_raw[..., self.my_slice])
```

### Q3: 训练时出现 NaN 怎么办？
1. 检查数据：`assert torch.isfinite(x).all()`
2. 降低学习率：`lr = 1e-5`
3. 启用梯度裁剪：`grad_clip = 1.0`
4. 检查损失权重：确保不会过大
5. 使用 AMP 的 GradScaler

### Q4: 如何实现自定义的自回归策略？
重写 `Trainer._rollout_sequence` 或修改 `mixed_state_mode`：
```python
trainer.mixed_state_mode = 'rot6d'  # 只混合旋转
trainer.mixed_state_mode = 'full'   # 混合整个状态
```

---

## 版本历史

### v1.2.0 (2025-01-16)
- 提取 `build_mlp` 工具函数
- 新增 `AdaptiveLossManager` 类
- 重构 MotionEncoder、EventMotionModel
- 消除 mu/std 重复存储

### v1.1.0
- 统一 LayoutCenter 和 DataNormalizer
- 移除推理端代码
- 添加 Rot6D 规格支持

### v1.0.0
- 初始版本

---

## 参考资料

- [Rot6D 论文](https://arxiv.org/abs/1812.07035) - On the Continuity of Rotation Representations
- [Teacher Forcing](https://machinelearningmastery.com/teacher-forcing-for-recurrent-neural-networks/)
- [Mixed Precision Training](https://pytorch.org/docs/stable/amp.html)

---

**维护者**: Motion AI Lab
**最后更新**: 2025-01-16
**许可证**: MIT
