# 训练系统架构与思路详解

> 本文档详细分析了整个训练系统的架构、数据流程、模型设计和训练策略
> 生成日期: 2025-12-25

---

## 目录

1. [系统概览](#系统概览)
2. [数据层架构](#数据层架构)
3. [模型层设计](#模型层设计)
4. [几何工具库](#几何工具库)
5. [训练流程](#训练流程)
6. [自适应机制](#自适应机制)
7. [评估系统](#评估系统)
8. [配置系统](#配置系统)
9. [关键创新点](#关键创新点)
10. [Contact Loop Closure & Stage2 λ Fusion](contact_loop_closure_design.md)

---

## 系统概览

### 核心设计理念

本训练系统是一个**基于事件驱动的无状态运动生成框架**，主要特点：

- **无状态架构**: 通过显式历史缓冲替代隐式RNN状态
- **分阶段训练**: Teacher Forcing → Mixed Mode → Free-Run 渐进式训练
- **多头监督**: 主运动头 + 接触规划头 + 直接姿态头 + SO(3)修正头
- **自适应学习**: 历史长度适配（AdaptiveHistory）

### 目录结构

```
train/
├── dataset.py              # 数据集与数据增强
├── models.py               # 核心模型定义
├── geometry.py             # 几何数学工具
├── training_MPL.py         # 主训练循环
├── eval_utils.py           # 评估工具
├── train_configurator.py   # 配置CLI入口（wrapper）
├── configuration/
│   ├── cli.py              # 配置CLI实现
│   ├── stages.py           # 分阶段配置
│   ├── profile.py          # 数据集profiling
│   └── io.py               # JSON IO
├── history.py              # 自适应历史模块
├── normalizers.py          # 归一化器
├── layout.py               # 数据布局工具
└── validate/               # 验证与可视化
```

---

## 数据层架构

### MotionEventDataset (`dataset.py`)

#### 核心职责

1. **加载预处理的NPZ数据**
   - 读取 `X_norm` (状态), `Y_norm` (输出), `cond_in` (条件)
   - 解析 `state_layout` 和 `output_layout`
   - 加载元数据 (骨骼名、FPS、坐标系等)

2. **提供每帧特征**
   ```python
   {
       'motion': X,              # [T, Dx] 状态 (已归一化)
       'gt_motion': Y,           # [T, Dy] 目标 (已归一化)
       'cond_in': C_in,          # [T, Dc] 条件输入 (窗口归一化)
       'cond_tgt': C_tgt,        # [T, Dc] 条件目标 (移位1帧)
       'contacts': contacts,     # [T, 2] 软接触标签
       'angvel': angvel,         # [T, J*3] 角速度 (tanh归一化)
       'pose_hist': pose_hist,   # [T, hist_len*J*6] 姿态历史
   }
   ```

3. **窗口级归一化**
   - 对条件 `C` 在窗口内做鲁棒均值/方差归一化
   - 避免全局统计量导致的分布偏移

#### 数据增强 (`MotionAugmentation`)

1. **时间扭曲** (Time Warping)
   - 等长重采样，反射边界
   - 保持可微性，适合训练时增强时序多样性

2. **Yaw旋转增强**
   - 随机旋转 `[-yaw_aug_deg, +yaw_aug_deg]`
   - 同步旋转:
     - 骨骼位置 `BonePositions`
     - 骨骼速度 `BoneVelocities`
     - 骨骼旋转 `BoneRotations6D`
     - 轨迹位置/方向 `TrajectoryPos/Dir`
     - 条件方向 `cond_in[:, -2:]`

3. **高斯噪声**
   - 可选的加性噪声

#### 特征计算

- **角速度 (`angvel_norm`)**:
  ```python
  # 从旋转序列计算
  R_seq = rot6d_to_matrix(bone_rot6d)  # [T, J, 3, 3]
  omega = angvel_vec_from_R_seq(R_seq, fps)  # [T-1, J, 3]
  angvel_norm = VectorTanhNormalizer.transform(omega)
  ```

- **姿态历史 (`pose_hist_norm`)**:
  ```python
  # 取前 pose_hist_len 帧的 rot6d
  hist_offsets = [1, 2, ..., pose_hist_len]
  pose_hist = rot6d[frame_ids - hist_offsets]  # [T, hist_len, J*6]
  pose_hist_norm = VectorTanhNormalizer.transform(pose_hist)
  ```

#### Forward Axis推断

- 自动推断骨盆前向轴 (`forward_axis`)
- 通过比对 `pelvis_yaw`（由 rot6d 推回）/ `velocity_yaw` / `cond_tgt` 找到最佳匹配轴
- 用于Yaw旋转增强和推理时的方向对齐

---

## 模型层设计

### EventMotionModel (`models.py`)

#### 整体架构

```
Input: [state, cond] → Shared Encoder → PASA (FiLM-modulated Attention)
                                          ↓
        ┌────────────────────────────────────────────┐
        │  Motion Head (主输出: delta_rot6d)         │
        │  + Bone Residual Adapters (per-bone微调)   │
        └────────────────────────────────────────────┘
                        ↓
        ┌───────────────┬─────────────────┬──────────────────┐
   Contact Plan      Direct Pose      Contact Meas       SO(3) Corrector
  (cond-only GRU)    (cond+plan)    (pose-derived)     (omega_hat修正)
```

#### 1. Shared Encoder

- **输入拼接**: `[state, cond, (opt: plan_inject)]`
- **结构**:
  - 基础: 2层MLP (baseline)
  - 可选: + N层ResidualMLPBlock (深度增强)
- **Residual Connection**: `h = encoder(x) + residual_proj(x)`

#### 2. PASA (Past-Attended Self-Attention)

```python
# FiLM调制
g, b = CondFiLM(cond)  # gamma, beta

# Multi-head Self-Attention
Q = LayerNorm(h) @ W_q
K, V = h @ W_k, h @ W_v
attn = softmax(Q @ K^T / sqrt(d_head))
ctx = attn @ V

# FiLM + Residual
h_final = LayerNorm((h + ctx) * (1 + g) + b)
```

- **目的**: 捕捉时序依赖，同时让条件能调制attention强度

#### 3. Motion Head

- **输出**: `delta_rot6d` (旋转增量)
- **Bone Adapters**:
  ```python
  # 为关键骨骼 (thigh_l/r, calf_l/r, foot_l/r) 添加残差适配
  for bone_slice, adapter in zip(slices, adapters):
      delta_full[..., bone_slice] += adapter(h_final)
  ```
  - 初始输出为0 (zero-init)
  - 允许后训练微调特定骨骼

#### 4. Contact Plan (独立锚点)

**设计动机**: 提供与姿态漂移无关的接触预测

```python
# GRUCell: 只看 cond 历史
for t in range(T):
    plan_z = GRUCell(cond[t], plan_z)
    logits = ContactPlanHead(plan_z)
    contacts_plan[t] = sigmoid(logits)
```

- **时间位置编码** (可选):
  ```python
  PE(t) = [sin(t/10000^(2i/D)), cos(t/10000^(2i/D))]
  logits += TimeHead(PE(t))
  ```
- **注入策略** (`contact_plan_inject`):
  - `"none"`: 不注入主encoder
  - `"contacts"`: 注入预测的contacts
  - `"plan_z"`: 注入GRU隐状态

#### 5. Direct Pose Head

**目的**: 从 `cond + contacts_plan` 直接预测绝对姿态

```python
direct_pose = DirectHead([cond, contacts_plan])  # [B, T, Dy]
```

- **监督**: `L_direct = Geo(direct_pose, Y_abs)` (rot6d→SO(3)测地线距离)
- **用途**: 提供额外的绝对姿态监督，辅助主delta分支

#### 6. Contact Meas (姿态衍生)

**白盒接触测量**: 从姿态特征预测接触

```python
meas_input = [pose_hist, angvel]  # 历史姿态 + 角速度
contacts_meas = sigmoid(ContactMeasHead(meas_input))
```

- **误差信号**: `e_t = contacts_plan - contacts_meas`
- **用于SO(3)修正**: 见下节

#### 7. SO(3) Delta Corrector

**目的**: 基于接触误差在流形上修正旋转

```python
corr_input = [h_final, e_t]  # 主隐状态 + 接触误差
omega_hat = SO3Corrector(corr_input)  # [B, T, J, 3] axis-angle

# 应用修正 (在训练/后训练中可选使用)
dR_corr = so3_exp_map(omega_hat)
R_corrected = dR_corr @ R_pred
```

- **Gate机制**: 通过 `so3_corr_gate_logit` 控制修正强度
- **初始化**: 输出初始为0，不影响baseline

#### Frozen Encoder (预训练hint)

```python
# 加载预训练的 MotionEncoder + PeriodHead
encoder_hidden = FrozenEncoder([contacts, angvel, pose_hist])
soft_period = tanh(FrozenPeriodHead(encoder_hidden))

# 注入到主encoder
period_emb = PeriodEncoder(soft_period)
h = h + period_emb
```

- **目的**: 提供软接触提示，稳定训练初期

---

### MotionJointLoss (`models.py`)

#### 损失项组成

1. **旋转损失** (`w_rot_local`)
   - 局部旋转测地线距离
   ```python
   R_pred = rot6d_to_matrix(pred_rot6d)
   R_gt = rot6d_to_matrix(gt_rot6d)
   loss_rot = geodesic_R(R_pred, R_gt).mean()
   ```

2. **接触规划损失** (`w_contact_plan`)
   ```python
   # soft contacts in [0,1]
   loss_plan = MSE(contacts_plan, contacts_gt)
   ```

3. **接触测量损失** (`w_contact_meas`)
   ```python
   # soft contacts in [0,1]
   loss_meas = MSE(contacts_meas, contacts_gt)
   ```

4. **直接姿态损失** (`w_direct_pose`)
   ```python
   # rot6d→SO(3)测地线距离
   loss_direct = Geo(out_direct, Y_abs)
   ```

5. **Omega正则化** (`w_omega_l2`)
   ```python
   loss_omega = ||omega_hat||^2
   ```

6. **Attention正则化** (`w_attn_reg`)
   ```python
   # 鼓励attention分布均匀
   loss_attn = -entropy(attn_weights)
   ```

7. **根速度/速度损失** (`w_root_vel`, `w_root_speed`)

8. **Ortho正则化** (`w_rot_ortho`)
   ```python
   # 鼓励旋转矩阵正交
   loss_ortho = ||R^T @ R - I||^2
   ```

#### 几何感知骨骼权重

- **当前主线**使用统一几何权重
- **统一权重公式**:
  ```python
  influence = self_scale * ||offset|| + (sum_lever_arm_to_descendants) ** power
  bone_weights = normalize(clamp_min(influence, min_w))
  ```
- **评估口径**:
  ```python
  GeoLocalDegWeighted = weighted_mean(geo_local_per_joint, bone_weights)
  ```
- **混合空间**: log-space或linear-space混合

---

## 几何工具库

### geometry.py

#### 1. 旋转表示转换

- **6D → Matrix**:
  ```python
  R = rot6d_to_matrix(xJ6, columns=("X", "Z"))
  # 取两列 → Gram-Schmidt正交化 → 派生第三列 → det修正
  ```

- **Matrix → 6D**:
  ```python
  rot6d = matrix_to_rot6d(R, columns=("X", "Z"))
  ```

- **重投影** (Gram-Schmidt):
  ```python
  rot6d_clean = reproject_rot6d(rot6d)  # 修正数值误差
  ```

- **SVD正交化**:
  ```python
  R_ortho = orthogonalize_rotation_matrix(R)
  # 使用SVD投影到SO(3)流形
  ```

#### 2. Delta合成

```python
# 增量合成: R_next = ΔR @ R_prev
rot6d_next = compose_rot6d_delta(
    prev_rot6d,
    delta_rot6d,
    reproject_result=True  # SVD重投影
)
```

#### 3. SO(3) Log/Exp Map

- **Exp**: `omega → R`
  ```python
  R = so3_exp_map(omega)  # axis-angle → matrix
  # 使用稳定的Rodrigues系数 + Taylor展开
  ```

- **Log**: `R → omega`
  ```python
  omega = so3_log_map(R)  # matrix → standard axis-angle / rotvec
  ```

#### 4. 角速度计算

```python
# 从旋转序列计算角速度
omega = angvel_vec_from_R_seq(R_seq, fps)
# omega[t] = log(R[t+1] @ R[t]^T) * fps
```

#### 5. Forward Kinematics

```python
# 从局部旋转 + 偏移计算全局位置
pos = fk_positions_from_rot6d(
    rot6d,       # [B, J, 6]
    parents,     # [J] parent indices
    offsets,     # [J, 3] local offsets
    root_pos,    # [B, 3]
)
```

---

## 训练流程

> 说明：关于 **Contact-loop / Stage2（λ Gate + SO(3) on-manifold 融合，把 `out_direct` 真正接入 rollout）** 的完整设计、诊断与验收指标，见 `docs/contact_loop_closure_design.md`。

### training_MPL.py: Trainer类

#### 核心循环

```python
for epoch in range(1, epochs + 1):
    # 1. 更新阶段配置
    stage_cfg = get_current_stage(epoch, freerun_stage_schedule)
    update_trainer_params(stage_cfg)

    # 2. 训练
    for batch in train_loader:
        # Teacher forcing + optional freerun mix
        preds, attn = _rollout_sequence(
            state_seq, cond_seq, ...,
            mode='mixed',
            tf_ratio=current_tf_ratio
        )

        # 计算损失
        loss, loss_dict = loss_fn(preds, gt_seq, attn, batch)

        # 自适应损失权重
        if adaptive_loss_module:
            loss, weights = adaptive_loss_module(
                loss_dict, model, epoch
            )

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

    # 3. 验证
    teacher_metrics = evaluate_teacher(val_loader, mode='teacher')
    freerun_metrics = evaluate_freerun(val_loader, FreeRunSettings)

    # 4. 保存checkpoint
    save_checkpoint(epoch, model, optimizer, metrics)
```

#### 分阶段训练策略

**Stage 1: Teacher (30%)**
- `freerun_weight = 0.0`
- `tf_max = 1.0` (100% teacher forcing)
- 目标: 学习单步预测

**Stage 2: Mixed (40%)**
- `freerun_weight = 0.35`
- `tf_max = 0.75` (75% → 线性衰减)
- 目标: 混合teacher和autoregressive

**Stage 3: Free-Run (30%)**
- `freerun_weight = 0.65`
- `tf_max = 0.5` (50% → 线性衰减)
- 目标: 强化自回归稳定性

#### Rollout实现

```python
def _rollout_sequence(
    state_seq, cond_seq, ...,
    mode='mixed', tf_ratio=1.0
):
    motion = state_seq[:, 0]  # 初始状态
    motion_raw = normalizer.denorm_x(motion)

    outs = []
    for t in range(T):
        # 前向传播
        result = model(
            motion.unsqueeze(1),  # [B, 1, Dx]
            cond_seq[:, t:t+1],
            contacts=contacts_seq[:, t:t+1],
            angvel=angvel_seq[:, t:t+1],
            pose_hist=pose_hist_buffer,
            plan_z=plan_z,
        )

        delta_norm = result['delta']  # 归一化空间的delta
        delta_raw = normalizer.denorm_y(delta_norm)

        # 合成下一帧
        y_raw_next = compose_rot6d_delta(
            y_raw_prev, delta_raw,
            reproject_result=True
        )

        # Teacher forcing vs Free-run
        if mode == 'mixed':
            use_gt = (torch.rand(B) < tf_ratio)
            y_raw_next = torch.where(
                use_gt[:, None],
                gt_y_raw[:, t],
                y_raw_next
            )

        # 更新状态
        motion_raw = update_state(motion_raw, y_raw_next)
        motion = normalizer.norm_x(motion_raw)
        y_raw_prev = y_raw_next

        # 更新历史缓冲
        update_pose_hist_buffer(y_raw_next)

        outs.append(delta_norm)

    return {'out': torch.stack(outs, dim=1)}, last_attn
```

#### Teacher Forcing比例调度

```python
def compute_tf_ratio(epoch, stage_cfg):
    if tf_mode == 'epoch_linear':
        # 线性从 tf_max → tf_min
        progress = (epoch - tf_start_epoch) / (tf_end_epoch - tf_start_epoch)
        progress = max(0, min(1, progress))
        return tf_max - (tf_max - tf_min) * progress
    return 1.0
```

---

## 自适应机制

### 自适应历史模块 (`train/history.py`)

#### 设计目标

- **训练时**: 可变长度历史 (增强泛化)
- **推理时**: 固定长度历史 (便于部署)

#### 实现

```python
class AdaptiveHistoryModule:
    def forward(self, pose_history, context):
        # 1. 动态采样有效长度
        if training and train_variable_history:
            eff = randint(num_slots, max_history_frames)
        else:
            eff = num_slots

        hist_slice = pose_history[:, -eff:, :]

        # 2. Attention聚合
        queries = query_tokens + context_proj(context)
        K, V = k_proj(hist_slice), v_proj(hist_slice)
        Q = q_proj(queries)

        attn = softmax(Q @ K^T)
        out = attn @ V

        # 3. Gate (optional)
        if use_gate:
            g = sigmoid(gate_proj(queries))
            out = g * out + (1 - g) * queries

        return out_proj(out)  # [B, num_slots * pose_dim]
```

#### History Dropout

```python
# 训练时随机丢弃历史，强制模型依赖其他线索
if training and rand() < history_dropout_prob:
    return zeros(B, num_slots * pose_dim)
```

---

## 评估系统

### eval_utils.py

#### 1. Teacher Forcing评估

```python
def evaluate_teacher(trainer, loader, mode='teacher'):
    for batch in loader:
        preds, attn = trainer._rollout_sequence(
            state_seq, cond_seq, ...,
            mode='teacher',
            tf_ratio=1.0
        )
        loss = loss_fn(preds, gt_seq, attn, batch)

        # 诊断统计
        diag = trainer._diagnose_free_run(
            predY, gtY, ...
        )
        # diag包含: MSEnormY, GeoDeg, ContactAccuracy等

    return aggregate_metrics(diag_list)
```

#### 2. Free-Run评估

```python
def evaluate_freerun(trainer, loader, settings):
    warmup = settings.warmup_steps
    horizon = settings.horizon

    for batch in loader:
        # 初始化
        motion = state_seq[:, warmup]

        # Autoregressive rollout
        for t in range(warmup, warmup + horizon):
            result = model(motion, cond[t], ...)
            delta = result['delta']

            # 合成下一帧 (无GT混合)
            y_raw_next = compose_rot6d_delta(y_raw_prev, delta_raw)
            motion_raw = update_state(motion_raw, y_raw_next)
            motion = normalizer.norm_x(motion_raw)

        # 诊断
        diag = diagnose_free_run(predsY, gtY)

    return aggregate_metrics(diag_list)
```

#### 关键指标

- **MSEnormY**: 归一化空间MSE
- **GeoDeg**: 平均测地线角度误差 (度)
- **ContactAccuracy**: 接触预测准确率
- **KeyBoneDetails**: 关键骨骼的详细误差
- **RootSpeed/Yaw误差**: 根速度和朝向误差

---

## 配置系统

### configuration/stages.py

#### TrainingConfigBuilder

**输入**: 数据集profile (总帧数、平均序列长度、复杂度等)

**输出**: 完整训练配置

```python
cfg = TrainingConfigBuilder(base_cfg).build(profile)

# 包含:
cfg['epochs'] = compute_total_epochs(total_frames)
cfg['batch'] = compute_batch_size(avg_seq_len)
cfg['lr'] = compute_base_lr(total_frames, complexity, batch)
cfg['freerun_stage_schedule'] = [
    {
        'range': [1, 30],
        'label': 'stage1_teacher',
        'trainer': {'freerun_weight': 0.0, 'freerun_horizon': 8},
        'loss': {'w_rot_local': 0.2},
        'tf': {'max': 1.0},
    },
    ...
]
```

#### 动态调整机制

```python
def update_trainer_from_stage(trainer, stage_cfg):
    # 更新损失权重
    trainer.loss_fn.w_rot_local = stage_cfg['loss']['w_rot_local']

    # 更新freerun参数
    trainer.freerun_weight = stage_cfg['trainer']['freerun_weight']
    trainer.freerun_horizon = stage_cfg['trainer']['freerun_horizon']

    # 更新TF参数
    trainer.tf_max = stage_cfg['tf']['max']
```

---

## 关键创新点

### 1. 无状态架构 + 显式历史

**传统RNN问题**:
- 隐状态难以解释
- 长序列梯度消失/爆炸
- 不便于调试和可视化

**本方案**:
- 显式传递 `pose_hist`, `angvel`, `contacts`
- 每帧可独立计算 (便于并行和缓存)
- 自适应历史长度 (训练时增强，推理时固定)

### 2. 接触规划 - 测量 - 误差闭环

```
Cond → ContactPlan (独立锚点, GRU)
          ↓
       e_t = plan - meas
          ↓
    SO(3) Corrector (omega_hat)
          ↓
    修正旋转增量
```

**优势**:
- `ContactPlan` 不依赖姿态，不受漂移影响
- `ContactMeas` 提供白盒姿态反馈
- `e_t` 作为误差信号引导SO(3)修正

### 3. 分阶段渐进训练

| 阶段 | Teacher比例 | FreeRun权重 | 目标 |
|------|------------|------------|------|
| Stage1 | 100% | 0.0 | 学习单步准确性 |
| Stage2 | 75%→50% | 0.35 | 混合模式平滑过渡 |
| Stage3 | 50%→0% | 0.65 | 强化自回归稳定性 |

**避免**:
- 一开始就freerun → 模型崩溃
- 始终teacher → 暴露偏差 (exposure bias)

### 4. 多头监督协同

- **Motion Head**: 主delta输出
- **Direct Head**: 绝对姿态辅助
- **Contact Plan**: 独立接触预测
- **Contact Meas**: 姿态衍生接触
- **SO(3) Corrector**: 流形上修正

每个头提供不同视角的监督，互相补充。

### 5. 自适应机制全覆盖

- **损失权重**: GradNorm/DWA自适应平衡多任务
- **超参调度**: 根据训练状态动态调整lr/clip_norm
- **历史长度**: 训练时可变，推理时固定
- **骨骼权重**: 基于先验方差和层级动态加权

### 6. 6D旋转表示 + SO(3)流形操作

**6D优势**:
- 连续可微 (优于欧拉角/四元数)
- 无奇异点
- 易于正交化 (Gram-Schmidt)

**流形操作**:
- Delta合成在SO(3)上: `R_next = ΔR @ R_prev`
- SVD重投影避免数值漂移
- Exp/Log map用于旋转向量 ↔ 矩阵转换

### 7. 窗口级条件归一化

**传统全局归一化**:
- 容易过拟合训练集统计量
- 测试时分布偏移

**窗口归一化**:
```python
mu, std = robust_mean_std(C_window)  # 鲁棒IQR统计
C_norm = (C - mu) / std
```
- 每个窗口自适应
- 更好的泛化性

---

## 数据流图

```
NPZ文件
  ↓
MotionEventDataset
  ├─ X_norm (state)
  ├─ Y_norm (target)
  ├─ cond_in (条件)
  ├─ contacts (软接触)
  ├─ angvel_norm (角速度)
  └─ pose_hist_norm (姿态历史)
  ↓
DataLoader (batch)
  ↓
Trainer._rollout_sequence
  ├─ mode='mixed', tf_ratio=0.75
  ├─ 初始化: motion = state[:, 0]
  └─ for t in range(T):
       ├─ result = model(motion, cond[t], contacts[t], angvel[t], pose_hist)
       ├─ delta_norm = result['delta']
       ├─ delta_raw = denorm_y(delta_norm)
       ├─ y_raw_next = compose_rot6d_delta(y_raw_prev, delta_raw)
       ├─ if rand() < tf_ratio: y_raw_next = gt_y_raw[t]
       ├─ motion_raw = update_state(motion_raw, y_raw_next)
       ├─ motion = norm_x(motion_raw)
       └─ update_pose_hist_buffer(y_raw_next)
  ↓
loss_fn(preds, gt, attn, batch)
  ├─ loss_rot_local
  ├─ loss_contact_plan
  ├─ loss_direct_pose
  ├─ loss_omega_l2
  └─ ...
  ↓
loss.backward()
  ↓
optimizer.step()
```

---

## 总结

本训练系统通过以下设计实现了**高精度、稳定、可控的运动生成**:

1. **数据层**: 窗口归一化 + 多模态特征 + Yaw增强
2. **模型层**: 无状态架构 + 多头监督 + SO(3)流形操作
3. **训练层**: 分阶段渐进 + 混合TF/Free
4. **评估层**: Teacher/FreeRun双模式 + 丰富诊断指标
5. **配置层**: 数据驱动的自动配置 + 灵活的阶段调度

**核心哲学**:
- **显式优于隐式**: 历史缓冲、接触规划都显式建模
- **多视角监督**: 不依赖单一损失，多头协同
- **渐进式学习**: 从单步到多步，从监督到自回归
- **流形上操作**: 尊重旋转的几何结构

---

**文档版本**: v1.0
**最后更新**: 2025-12-25
**维护者**: Training System Team
