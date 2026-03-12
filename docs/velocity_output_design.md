# 运动模型速度输入输出优化设计文档

## 文档信息
- **创建时间**: 2025-11-28
- **版本**: v1.0
- **状态**: 设计方案
- **作者**: 运动生成系统重构
> **2025-12-09 更新：** 代码中已移除 `w_rot_delta`/`w_rot_delta_root` 及相关分支，文档出现的旧权重仅保留为历史记录，不再可用。

---

## 一、问题背景

### 1.1 当前系统架构

#### 数据布局
```
State (X): [419维]
├─ RootPosition [0:3]           # 3D位置
├─ RootVelocity [3:5]           # 2D速度 (XY平面)
├─ BoneRotations6D [5:281]      # 276维 = 46骨骼 × 6
└─ BoneAngularVelocities [281:419]  # 138维

Output (Y): [276维]
└─ BoneRotations6D [0:276]      # ❌ 仅骨骼旋转，无速度输出

Condition (C): [7维]
├─ ActionOneHot [0:4]           # 4个动作类别
├─ Direction [4:6]              # 2D单位方向向量
└─ Speed [6:7]                  # ❌ 绝对速度值
```

#### 代码位置
- 数据处理: `train/convert_json_to_npz.py:1206-1248`
- 模型输出: `train/models.py:198-205` (EventMotionModel.motion_head)
- 布局定义: `raw_data/processed_data/norm_template.json:1909-1932`

### 1.2 核心问题

#### **问题1: 模型输出缺少速度预测**

**现状**:
- 模型仅输出 BoneRotations6D (276维)
- RootVelocity 存在于 State(X) 中，但不在 Output(Y) 中
- 速度变化**隐式**通过骨骼旋转变化体现

**影响**:
| 方面 | 影响 |
|------|------|
| ✅ 优点 | 简化输出维度，避免速度与姿态不一致 |
| ❌ 缺点 | 无法直接获取速度预测，必须通过位置差分计算 |
| ❌ 缺点 | 速度控制能力弱，难以精确控制加速/减速 |
| ❌ 缺点 | 推理时速度累积误差大 |

#### **问题2: Cond中速度语义不合理**

**现状**: 使用绝对速度值 (单位: m/s)
```python
# convert_json_to_npz.py:1247-1248
cond_speed = speed[:-1]  # 原始速度值
cond_in = np.concatenate([act_oh, cond_dir, cond_speed], axis=-1)
```

**问题场景**:

| 动作 | 典型速度范围 (m/s) | speed=1.0 的语义 | 问题 |
|------|-------------------|-----------------|------|
| idle | 0.0 - 0.1 | 过快 (超过10倍正常速度) | ❌ 语义错误 |
| walk | 0.5 - 1.2 | 正常行走速度 | ✅ 合理 |
| run | 2.0 - 3.5 | 过慢 (不到一半速度) | ❌ 语义错误 |
| jump | 0.3 - 2.0 | 不确定 | ⚠️ 混乱 |

**游戏控制场景的问题**:
```python
# 场景1: 用户想让角色"加速10%"
current_speed = get_current_speed()  # walk时0.85, run时2.5
new_speed = current_speed * 1.1      # walk→0.935, run→2.75

# ❌ 问题: 需要知道当前动作的实际速度值，游戏逻辑复杂
```

```python
# 场景2: 不同动作的"加速"效果不一致
walk_acceleration = 0.85 * 0.1  # 加速0.085 m/s
run_acceleration = 2.5 * 0.1    # 加速0.25 m/s (3倍差异!)

# ❌ 问题: 相同的加速操作，对不同动作影响完全不同
```

**根本原因**:
- 不同动作有**不同的固有速度范围**
- 使用绝对速度值混淆了"动作类型"和"速度控制"两个正交维度
- 游戏端需要维护复杂的动作-速度映射表

---

## 二、解决方案设计

### 2.1 设计原则

#### 核心思路: **分阶段渐进式优化**

**阶段1 (本次)**: 解决速度输入输出的基础问题
- ✅ 添加速度输出 (Output扩展)
- ✅ 改用相对速度倍率 (Cond优化)
- ⏸️ 暂不处理动作切换 (速度过低自动从run切换到walk)

**阶段2 (后续)**: 智能动作切换与混合
- 根据速度自动选择合适的动作
- 动作间的平滑过渡
- 多动作混合 (blend)

### 2.2 技术方案

#### **方案A: 扩展Output - 添加速度输出**

##### 修改1: 扩展Output Layout

```python
# 原来 (276维)
output_layout = {
    "BoneRotations6D": {"start": 0, "size": 276}
}

# 修改后 (278维)
output_layout = {
    "BoneRotations6D": {"start": 0, "size": 276},
    "RootVelocity": {"start": 276, "size": 2}  # 新增: 2D速度向量
}
```

**说明**:
- `RootVelocity [276:278]`: 2D速度向量 (vx, vy)，单位 m/s
- 与 State(X) 中的 RootVelocity[3:5] 对应
- **不添加** RootSpeed 标量 (可从向量计算: `sqrt(vx^2 + vy^2)`)

**优点**:
- ✅ 显式预测速度，控制精度高
- ✅ 可添加速度损失项，提高预测准确性
- ✅ 与Cond中的速度形成闭环控制
- ✅ 推理时直接使用，无需差分计算

**代价**:
- ⚠️ 需要重新训练模型
- ⚠️ 输出维度增加2维 (开销极小)

##### 修改2: 模型输出维度

**文件**: `train/models.py`

```python
# 原来
class EventMotionModel(nn.Module):
    def __init__(
        self,
        in_state_dim: int,      # 419
        out_motion_dim: int,    # 276 ← 修改为 278
        ...
    ):
        ...
        self.motion_head = build_mlp(
            hidden_dim,
            hidden_dim,
            num_layers=1,
            activation=nn.ReLU,
            dropout=dropout,
            final_dim=out_motion_dim,  # 输出278维
        )
```

##### 修改3: 损失函数扩展

**文件**: `train/models.py` (MotionJointLoss)

新增速度相关损失项:

```python
class MotionJointLoss(nn.Module):
    def __init__(
        self,
        ...
        w_root_vel: float = 0.1,      # 新增: 速度向量L2损失权重
        w_root_speed: float = 0.05,   # 新增: 速度大小MAE损失权重
    ):
        ...

    def forward(self, pred, target, ...):
        losses = {}

        # 原有损失: BoneRotations6D (276维)
        pred_rot = pred[:, :, :276]
        target_rot = target[:, :, :276]
        losses['rot_local'] = self._compute_rotation_loss(pred_rot, target_rot)

        # 新增: RootVelocity损失 (278-276=2维)
        if pred.shape[-1] >= 278:
            pred_vel = pred[:, :, 276:278]      # [B, T, 2]
            target_vel = target[:, :, 276:278]  # [B, T, 2]

            # L2损失: 速度向量误差
            vel_l2 = F.mse_loss(pred_vel, target_vel)
            losses['root_vel_l2'] = vel_l2

            # MAE损失: 速度大小误差
            pred_speed = torch.norm(pred_vel, dim=-1)    # [B, T]
            target_speed = torch.norm(target_vel, dim=-1)
            speed_mae = F.l1_loss(pred_speed, target_speed)
            losses['root_speed_mae'] = speed_mae

            # 加权到总损失
            total_loss += self.w_root_vel * vel_l2
            total_loss += self.w_root_speed * speed_mae

        return total_loss, losses
```

**损失权重建议**:
- `w_root_vel = 0.1`: 速度向量的重要性约为旋转损失的1/7 (因为旋转有46个骨骼)
- `w_root_speed = 0.05`: 速度大小的重要性稍低，用于辅助约束

##### 修改4: 数据处理

**文件**: `train/convert_json_to_npz.py`

```python
# 原来: Y只包含骨骼旋转
y_out_features = bone_rot_6d  # [T, 276]

# 修改后: Y包含骨骼旋转 + 速度
root_vel_2d = clip["root_vel"][:, :2].astype(np.float32)  # [T, 2]
y_out_features = np.concatenate([
    bone_rot_6d,      # [T, 276]
    root_vel_2d,      # [T, 2]
], axis=-1)           # [T, 278]

# 更新output_layout
output_layout_json = json.dumps({
    'BoneRotations6D': {'start': 0, 'size': 276},
    'RootVelocity': {'start': 276, 'size': 2},
}, ensure_ascii=False)
```

---

#### **方案B: 优化Cond - 相对速度倍率**

##### 核心思想: 使用相对于动作正常速度的倍率

**语义定义**:
```
speed_multiplier = 1.0  →  正常速度 (无论什么动作)
speed_multiplier = 1.1  →  加速10%
speed_multiplier = 0.9  →  减速10%
speed_multiplier = 0.0  →  静止
```

##### 修改1: 预计算动作速度统计

**文件**: `train/convert_json_to_npz.py` (新增函数)

```python
def compute_action_speed_stats(json_files: list[str]) -> dict:
    """
    统计每个动作的速度分布

    Returns:
        {
            "idle": {"mean": 0.05, "std": 0.03, "p50": 0.04, "p95": 0.10},
            "walk": {"mean": 0.85, "std": 0.15, "p50": 0.83, "p95": 1.15},
            "run":  {"mean": 2.50, "std": 0.40, "p50": 2.45, "p95": 3.20},
            "jump": {"mean": 1.20, "std": 0.60, "p50": 1.10, "p95": 2.10},
        }
    """
    from collections import defaultdict
    import numpy as np

    action_speeds = defaultdict(list)

    for json_path in json_files:
        with open(json_path, 'r') as f:
            clip = json.load(f)

        action = str(clip.get("action", "unknown")).strip().lower()

        # 计算速度
        if "root_vel" in clip and clip["root_vel"] is not None:
            vel = np.asarray(clip["root_vel"], dtype=np.float32)[:, :2]
        else:
            fps = float(clip.get("FPS", 60.0))
            pos = np.asarray(clip["root_pos"], dtype=np.float32)[:, :2]
            vel = np.zeros_like(pos)
            vel[1:] = (pos[1:] - pos[:-1]) * fps

        speeds = np.linalg.norm(vel, axis=1)
        action_speeds[action].extend(speeds.tolist())

    # 统计
    stats = {}
    for action, speeds in action_speeds.items():
        speeds = np.array(speeds)
        stats[action] = {
            "mean": float(np.mean(speeds)),
            "std": float(np.std(speeds)),
            "p50": float(np.percentile(speeds, 50)),
            "p95": float(np.percentile(speeds, 95)),
            "count": len(speeds),
        }

    return stats
```

##### 修改2: 构造相对速度倍率

**文件**: `train/convert_json_to_npz.py:1206-1248`

```python
# 原来
vel_dir, speed = _compute_planar_vel_dir_and_speed(clip)
cond_speed = speed[:-1]  # 绝对速度值
cond_in = np.concatenate([act_oh, cond_dir, cond_speed], axis=-1)

# 修改后
vel_dir, speed = _compute_planar_vel_dir_and_speed(clip)

# 获取当前动作的基准速度
action_name = str(clip.get("action", "unknown")).strip().lower()
if action_name in ACTION_SPEED_STATS:
    base_speed = ACTION_SPEED_STATS[action_name]["mean"]
else:
    # 未知动作: 使用全局平均或当前片段平均
    base_speed = np.mean(speed) + 1e-6

# 计算相对倍率
speed_multiplier = speed / np.clip(base_speed, 1e-3, None)  # [T, 1]
speed_multiplier = np.clip(speed_multiplier, 0.0, 5.0)     # 限制在合理范围

# 构造cond
cond_speed_mult = speed_multiplier[:-1]  # [T-1, 1]
cond_in = np.concatenate([act_oh, cond_dir, cond_speed_mult], axis=-1)
```

**关键点**:
- `base_speed`: 从统计数据获取，如 walk=0.85, run=2.5
- `speed_multiplier`: 相对倍率，1.0表示正常速度
- `clip(0.0, 5.0)`: 限制倍率范围，避免极端值

##### 修改3: 保存速度统计到配置

**文件**: `raw_data/processed_data/norm_template.json` (新增字段)

```json
{
  "state_layout": {...},
  "output_layout": {...},
  "action_speed_stats": {
    "idle": {"mean": 0.05, "std": 0.03, "p50": 0.04, "p95": 0.10},
    "walk": {"mean": 0.85, "std": 0.15, "p50": 0.83, "p95": 1.15},
    "run":  {"mean": 2.50, "std": 0.40, "p50": 2.45, "p95": 3.20},
    "jump": {"mean": 1.20, "std": 0.60, "p50": 1.10, "p95": 2.10}
  }
}
```

##### 修改4: 推理时的速度控制

**游戏端伪代码**:
```python
# 用户控制接口
def set_character_motion(action: str, direction: Vec2, speed_multiplier: float):
    """
    action: "walk", "run", "jump", "idle"
    direction: 2D单位向量 (归一化的朝向)
    speed_multiplier: 速度倍率 (1.0=正常, 1.1=加速10%, 0.9=减速10%)
    """
    # 构造cond
    action_onehot = encode_action(action)  # [4]
    direction_2d = normalize(direction)    # [2]

    cond = np.concatenate([
        action_onehot,      # [4]
        direction_2d,       # [2]
        [speed_multiplier], # [1] - 直接传入倍率!
    ])  # [7]

    # 调用模型
    output = model.forward(state, cond)

    # 解析输出
    bone_rotations = output[:276]
    root_velocity = output[276:278]  # 预测的速度向量

    return bone_rotations, root_velocity


# 示例: 加速10%
set_character_motion("walk", direction=[1, 0], speed_multiplier=1.1)

# 示例: 减速20%
set_character_motion("run", direction=[0, 1], speed_multiplier=0.8)

# 示例: 静止
set_character_motion("idle", direction=[0, 0], speed_multiplier=0.0)
```

**优点对比**:

| 场景 | 原方案 (绝对速度) | 新方案 (相对倍率) |
|------|------------------|------------------|
| 加速10% | ❌ `speed = get_speed() * 1.1`<br>需要查询当前速度 | ✅ `multiplier = 1.1`<br>直接传入 |
| 不同动作一致性 | ❌ walk加速0.085, run加速0.25<br>效果差3倍 | ✅ 所有动作加速10%<br>效果一致 |
| 代码复杂度 | ❌ 需要维护动作速度映射表 | ✅ 无需额外映射 |
| 语义清晰度 | ❌ speed=1.5含义模糊 | ✅ multiplier=1.5表示1.5倍速 |

---

### 2.3 数据流完整对比

#### 原来的流程
```
训练数据处理:
  root_vel (m/s) → speed = norm(vel) → cond[6] = speed (绝对值)

模型输入:
  X[419维] = [RootPos(3), RootVel(2), BoneRot(276), AngVel(138)]
  C[7维] = [ActionOH(4), Direction(2), Speed(1)]  ← 绝对速度

模型输出:
  Y[276维] = [BoneRot(276)]  ← 无速度输出

推理:
  next_pos = current_pos + root_vel * dt  ← root_vel从哪来? 差分或FK计算
```

**问题**:
1. ❌ 输出无速度，需要额外计算
2. ❌ Cond中的绝对速度语义不统一
3. ❌ 速度控制需要查表

#### 修改后的流程
```
训练数据处理:
  root_vel (m/s) → speed = norm(vel)
  action_base_speed (从统计获取) → multiplier = speed / base_speed
  cond[6] = multiplier (相对倍率)

模型输入:
  X[419维] = [RootPos(3), RootVel(2), BoneRot(276), AngVel(138)]
  C[7维] = [ActionOH(4), Direction(2), Multiplier(1)]  ← 相对倍率

模型输出:
  Y[278维] = [BoneRot(276), RootVel(2)]  ← 显式速度输出

推理:
  next_root_vel = output[276:278]  ← 直接获取
  next_pos = current_pos + next_root_vel * dt
```

**改进**:
1. ✅ 输出包含速度，直接使用
2. ✅ Cond中的倍率语义统一 (1.0=正常速度)
3. ✅ 速度控制简化 (直接传倍率)

---

## 三、实施步骤

### 3.1 阶段1: 数据统计与验证 (优先)

#### Step 1.1: 统计动作速度分布

**目标**: 了解当前数据中各动作的速度特征

**任务**:
```bash
# convert_json_to_npz.py 当前没有 analyze_speed CLI；可直接调用脚本里的函数：
python - <<'PY'
from pathlib import Path
import json

from train.convert_json_to_npz import find_json_files, compute_action_speed_stats

stats = compute_action_speed_stats(find_json_files("raw_data/source_json"))
out_path = Path("raw_data/processed_data/action_speed_stats.json")
out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
print("[OK] wrote", out_path)
PY
```

**产出**: `raw_data/processed_data/action_speed_stats.json`
```json
{
  "idle": {"mean": 0.05, "std": 0.03, "count": 1250},
  "walk": {"mean": 0.85, "std": 0.15, "count": 3200},
  "run":  {"mean": 2.50, "std": 0.40, "count": 2800},
  "jump": {"mean": 1.20, "std": 0.60, "count": 1100}
}
```

**验证点**:
- [ ] 每个动作至少有100帧数据
- [ ] mean值符合物理常识 (walk < run)
- [ ] std不会过大 (std/mean < 0.5)

#### Step 1.2: 可视化速度分布

**目标**: 确认问题严重程度

**任务**:
```python
# 绘制速度分布直方图
import matplotlib.pyplot as plt

for action, stats in action_speed_stats.items():
    plt.hist(stats['data'], bins=50, alpha=0.5, label=action)
plt.legend()
plt.xlabel('Speed (m/s)')
plt.ylabel('Frequency')
plt.title('Speed Distribution by Action')
plt.savefig('speed_distribution.png')
```

**验证点**:
- [ ] 不同动作的速度分布有明显分离
- [ ] idle的速度集中在接近0的区域
- [ ] walk和run的速度范围不重叠或重叠很少

---

### 3.2 阶段2: 修改数据处理 (核心)

#### Step 2.1: 实现相对速度倍率计算

**文件**: `train/convert_json_to_npz.py`

**修改点**:
1. ✅ 新增 `compute_action_speed_stats()` 函数
2. ✅ 新增 `load_action_speed_stats()` 函数
3. ✅ 修改 cond_in 构造逻辑 (1206-1248行)
4. ✅ 保存统计数据到 norm_template.json

**测试**:
```bash
# 重新处理一个JSON文件
python train/convert_json_to_npz.py raw_data/source_json/walk_01.json --out test_output

# 检查输出
python -c "
import numpy as np
data = np.load('test_output/walk_01.npz')
print('Cond shape:', data['cond_in'].shape)  # 应该是 (T-1, 7)
print('Speed multiplier range:', data['cond_in'][:, 6].min(), data['cond_in'][:, 6].max())
# 应该在 0.8-1.2 左右 (正常walk的倍率应该接近1.0)
"
```

#### Step 2.2: 扩展Output布局

**文件**: `train/convert_json_to_npz.py`

**修改点**:
```python
# 行号约1100-1150 (构造y_out_features的地方)

# 原来
y_out_features = bone_rot_6d  # [T, 276]

# 修改后
root_vel_2d = clip["root_vel"][:, :2].astype(np.float32)  # [T, 2]
y_out_features = np.concatenate([bone_rot_6d, root_vel_2d], axis=-1)  # [T, 278]

# 更新output_layout (行号约1259)
output_layout_json = json.dumps({
    'BoneRotations6D': {'start': 0, 'size': 276},
    'RootVelocity': {'start': 276, 'size': 2},
}, ensure_ascii=False)
```

**测试**:
```bash
python -c "
import numpy as np
data = np.load('test_output/walk_01.npz')
print('Y shape:', data['y_out_features'].shape)  # 应该是 (T-1, 278)
print('RootVel in Y:', data['y_out_features'][0, 276:278])  # 应该是合理的速度值
"
```

#### Step 2.3: 批量重新处理数据

**任务**:
```bash
# 备份旧数据
cp -r raw_data/processed_data raw_data/processed_data.backup_v1

# 重新处理所有数据
python train/convert_json_to_npz.py raw_data/source_json --out raw_data/processed_data
```

**验证点**:
- [ ] 所有 .npz 文件的 y_out_features 维度为 278
- [ ] 所有 .npz 文件的 cond_in[:,6] 值在合理范围 (0-5)
- [ ] norm_template.json 包含 action_speed_stats 字段

---

### 3.3 阶段3: 修改模型与训练 (重训练)

#### Step 3.1: 更新模型输出维度

**文件**: `train/models.py`

**修改点**:
1. ✅ EventMotionModel.__init__ 的 `out_motion_dim` 参数
2. ✅ motion_head 输出维度

**代码**:
```python
# 行号约 140-156
class EventMotionModel(nn.Module):
    def __init__(
        self,
        in_state_dim: int,
        out_motion_dim: int,  # 从配置读取,应该是278
        ...
    ):
        # 无需修改,只需确保传入的参数是278
```

**配置文件**: `config/exp_phase_mpl.json`

不需要修改 (out_motion_dim 从 norm_template.json 自动读取)

#### Step 3.2: 扩展损失函数

**文件**: `train/models.py`

**修改位置**: `MotionJointLoss` 类

```python
# 行号约 420-550

class MotionJointLoss(nn.Module):
    def __init__(
        self,
        ...
        w_root_vel: float = 0.0,      # 新增参数
        w_root_speed: float = 0.0,    # 新增参数
    ):
        super().__init__()
        self.w_root_vel = float(w_root_vel)
        self.w_root_speed = float(w_root_speed)
        ...

    def forward(
        self,
        pred: torch.Tensor,      # [B, T, 278]
        target: torch.Tensor,    # [B, T, 278]
        ...
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        B, T, D = pred.shape
        losses = {}
        total_loss = 0.0

        # 原有损失: BoneRotations6D [0:276]
        pred_rot = pred[:, :, :276]
        target_rot = target[:, :, :276]
        # ... 原有的旋转损失计算 ...

        # 新增: RootVelocity损失 [276:278]
        if D >= 278 and (self.w_root_vel > 0 or self.w_root_speed > 0):
            pred_vel = pred[:, :, 276:278]      # [B, T, 2]
            target_vel = target[:, :, 276:278]  # [B, T, 2]

            # 速度向量L2损失
            if self.w_root_vel > 0:
                vel_mse = F.mse_loss(pred_vel, target_vel)
                losses['root_vel_mse'] = vel_mse.item()
                total_loss = total_loss + self.w_root_vel * vel_mse

            # 速度大小MAE损失
            if self.w_root_speed > 0:
                pred_speed = torch.norm(pred_vel, dim=-1)    # [B, T]
                target_speed = torch.norm(target_vel, dim=-1)
                speed_mae = F.l1_loss(pred_speed, target_speed)
                losses['root_speed_mae'] = speed_mae.item()
                total_loss = total_loss + self.w_root_speed * speed_mae

        return total_loss, losses
```

**配置权重**: `config/exp_phase_mpl.json`

在各阶段的 `loss_groups.core` 中添加:
```json
{
  "label": "stage1_teacher",
  "loss_groups": {
    "core": {
      "w_fk_pos": 0.07,
      "w_rot_local": 0.07,
      "w_root_vel": 0.05,      // 新增
      "w_root_speed": 0.03     // 新增
    }
  }
}
```

**权重调优建议**:
- 初始: `w_root_vel=0.05, w_root_speed=0.03` (相对较小)
- 观察训练: 如果速度误差偏大,逐步增加到 0.1, 0.05
- 最终目标: RootVelMAE < 0.02 (见配置中的goal指标)

#### Step 3.3: 更新训练脚本

**文件**: `train/training_MPL.py`

**修改点**: 确保从配置正确读取 loss 权重

```python
# 行号约 200-250 (损失函数初始化)

criterion = MotionJointLoss(
    state_layout=state_layout,
    output_layout=output_layout,
    w_fk_pos=config.get('w_fk_pos', 0.07),
    w_rot_local=config.get('w_rot_local', 0.07),
    w_root_vel=config.get('w_root_vel', 0.05),      # 新增
    w_root_speed=config.get('w_root_speed', 0.03),  # 新增
    ...
)
```

#### Step 3.4: 开始训练

**命令**:
```bash
# 清空旧模型 (可选,建议备份)
mv models/exp_phase_e2e_sc models/exp_phase_e2e_sc.backup_v1

# 开始训练
python train/training_MPL.py --config=config/exp_phase_mpl.json
```

**监控指标**:
| 指标 | 阶段1目标 | 阶段3目标 | 说明 |
|------|----------|----------|------|
| GeoDeg | < 2.5 | < 2.0 | 旋转误差 (度) |
| MSEnormY | < 0.12 | < 0.10 | 整体输出MSE |
| **root_vel_mse** | **< 0.01** | **< 0.005** | **速度向量MSE** |
| **root_speed_mae** | **< 0.05** | **< 0.02** | **速度大小MAE** |
| RootVelMAE (freerun) | - | < 0.02 | 自由运行速度误差 |

**验证点**:
- [ ] 损失正常下降 (不发散)
- [ ] root_vel_mse 在前5个epoch降到 < 0.02
- [ ] root_speed_mae 在前5个epoch降到 < 0.1
- [ ] GeoDeg 不因新增速度损失而变差

---

### 3.4 阶段4: 推理验证 (集成测试)

#### Step 4.1: 单帧推理测试

**测试脚本**: `test_velocity_output.py` (新建)

```python
import torch
import numpy as np
from train.models import EventMotionModel

# 加载模型
model = EventMotionModel(...)
model.load_state_dict(torch.load('models/exp_phase_e2e_sc/model_epoch_10.pt'))
model.eval()

# 构造输入
state = torch.randn(1, 1, 419)  # [B, T, 419]

# 测试不同速度倍率
for speed_mult in [0.5, 0.8, 1.0, 1.2, 1.5]:
    cond = torch.zeros(1, 1, 7)
    cond[0, 0, 0] = 1.0  # action=walk
    cond[0, 0, 4] = 1.0  # direction x
    cond[0, 0, 6] = speed_mult  # speed multiplier

    with torch.no_grad():
        output = model(state, cond)['out']  # [1, 1, 278]

    # 提取速度
    pred_vel = output[0, 0, 276:278].numpy()
    pred_speed = np.linalg.norm(pred_vel)

    print(f"Multiplier={speed_mult:.1f} → Predicted speed={pred_speed:.3f} m/s")

# 预期结果 (假设walk的base_speed=0.85):
# Multiplier=0.5 → Predicted speed≈0.42
# Multiplier=1.0 → Predicted speed≈0.85
# Multiplier=1.5 → Predicted speed≈1.28
```

**验证点**:
- [ ] 速度倍率与预测速度成正比
- [ ] multiplier=1.0 时预测速度接近动作的base_speed
- [ ] 速度向量方向与direction一致

#### Step 4.2: 自由运行测试 (Freerun)

**目标**: 验证速度在长序列中的累积误差

**命令**:
```bash
python train/validate/run_teacher_rollout.py \
  --model_path=models/exp_phase_e2e_sc/model_epoch_10.pt \
  --horizon=60 \
  --output=debug_output/freerun_velocity_test.pt
```

**分析脚本**:
```python
import torch
import matplotlib.pyplot as plt

data = torch.load('debug_output/freerun_velocity_test.pt')

# 提取速度
pred_vel = data['pred'][:, :, 276:278]  # [B, T, 2]
target_vel = data['target'][:, :, 276:278]

# 计算速度大小
pred_speed = torch.norm(pred_vel, dim=-1).numpy()  # [B, T]
target_speed = torch.norm(target_vel, dim=-1).numpy()

# 绘制对比
plt.figure(figsize=(12, 4))
for i in range(min(3, len(pred_speed))):
    plt.plot(target_speed[i], label=f'Target {i}', linestyle='--')
    plt.plot(pred_speed[i], label=f'Pred {i}')
plt.xlabel('Frame')
plt.ylabel('Speed (m/s)')
plt.legend()
plt.title('Speed Prediction vs Target (Freerun)')
plt.savefig('speed_freerun_comparison.png')

# 计算累积误差
mae = np.mean(np.abs(pred_speed - target_speed))
print(f"Freerun Speed MAE: {mae:.4f} m/s")
```

**验证点**:
- [ ] Freerun MAE < 0.05 (前30帧)
- [ ] Freerun MAE < 0.10 (60帧)
- [ ] 速度不会无限增大或减小 (稳定性)

#### Step 4.3: 游戏引擎集成测试 (如UE/Unity)

**伪代码**:
```python
# 游戏端
class MotionController:
    def update(self, action: str, direction: Vec2, speed_mult: float):
        # 构造cond
        cond = self.build_cond(action, direction, speed_mult)

        # 模型推理
        output = self.model.forward(self.state, cond)

        # 解析输出
        bone_rotations = output[:276]
        root_velocity = output[276:278]  # 关键: 直接获取速度!

        # 更新状态
        self.state[0:3] += root_velocity * dt  # 位置更新
        self.state[3:5] = root_velocity        # 速度更新
        self.state[5:281] = bone_rotations     # 骨骼更新

        return bone_rotations, root_velocity

# 测试场景1: 匀速行走
controller.update("walk", direction=[1, 0], speed_mult=1.0)
# → 预期速度稳定在 0.85 m/s 左右

# 测试场景2: 加速
for t in range(30):
    mult = 1.0 + 0.5 * (t / 30)  # 从1.0线性加速到1.5
    controller.update("walk", direction=[1, 0], speed_mult=mult)
# → 预期速度平滑增加

# 测试场景3: 急停
controller.update("walk", direction=[1, 0], speed_mult=0.1)
# → 预期速度快速降低
```

**验证点**:
- [ ] 速度倍率控制生效
- [ ] 加速/减速过渡平滑
- [ ] 无抖动或异常跳变

---

## 四、配置文件修改清单

### 4.1 数据配置

**文件**: `raw_data/processed_data/norm_template.json`

```json
{
  "state_layout": {
    "RootPosition": {"start": 0, "size": 3},
    "RootVelocity": {"start": 3, "size": 2},
    "BoneRotations6D": {"start": 5, "size": 276},
    "BoneAngularVelocities": {"start": 281, "size": 138}
  },
  "output_layout": {
    "BoneRotations6D": {"start": 0, "size": 276},
    "RootVelocity": {"start": 276, "size": 2}     // 新增
  },
  "action_speed_stats": {                         // 新增
    "idle": {"mean": 0.05, "std": 0.03},
    "walk": {"mean": 0.85, "std": 0.15},
    "run":  {"mean": 2.50, "std": 0.40},
    "jump": {"mean": 1.20, "std": 0.60}
  }
}
```

### 4.2 训练配置

**文件**: `config/exp_phase_mpl.json`

在各训练阶段的 `loss_groups.core` 中添加速度损失权重:

```json
{
  "freerun_stage_schedule": [
    {
      "label": "stage1_teacher",
      "loss_groups": {
        "core": {
          "w_fk_pos": 0.07,
          "w_rot_local": 0.07,
          "w_root_vel": 0.05,        // 新增
          "w_root_speed": 0.03       // 新增
        }
      }
    },
    {
      "label": "stage2_mixed_warmup",
      "loss_groups": {
        "core": {
          "w_fk_pos": 0.2275,
          "w_rot_local": 0.2275,
          "w_rot_delta_root": 0.2,
          "w_root_vel": 0.08,        // 新增 (稍微提高)
          "w_root_speed": 0.05       // 新增
        }
      }
    },
    {
      "label": "stage3_freerun",
      "loss_groups": {
        "core": {
          "w_fk_pos": 0.4025,
          "w_rot_local": 0.4025,
          "w_rot_delta_root": 0.25,
          "w_root_vel": 0.15,        // 新增 (freerun阶段提高权重)
          "w_root_speed": 0.08       // 新增
        }
      },
      "goal": {
        "metrics": {
          "GeoDeg": {"ref": 2.0, "hi_ratio": 1.02},
          "YawAbsDeg": {"ref": 10.0, "hi_ratio": 0.9},
          "FreeRun/RootVelMAE": {"hi": 0.02}  // 已有,验证速度误差
        }
      }
    }
  ]
}
```

**权重调优策略**:
- 阶段1 (teacher): 较低权重,优先学习旋转
- 阶段2 (mixed): 中等权重,开始关注速度
- 阶段3 (freerun): 较高权重,强化速度预测准确性

---

## 五、风险与应对

### 5.1 潜在风险

| 风险 | 影响 | 概率 | 应对措施 |
|------|------|------|---------|
| **速度损失权重过高** | 旋转质量下降 | 中 | 逐步增加权重,监控GeoDeg指标 |
| **动作速度统计不准** | 倍率语义错误 | 低 | 人工检查统计结果,可视化分布 |
| **数据中速度噪声大** | 训练不稳定 | 中 | 对速度做平滑处理 (SG滤波) |
| **模型容量不足** | 新增输出效果差 | 低 | 当前hidden_dim=512足够,必要时增加到768 |
| **推理速度变慢** | 游戏帧率下降 | 极低 | 新增2维输出几乎无影响 |

### 5.2 回滚方案

如果新方案效果不佳,可快速回滚:

```bash
# 恢复旧数据
rm -rf raw_data/processed_data
mv raw_data/processed_data.backup_v1 raw_data/processed_data

# 恢复旧模型
rm -rf models/exp_phase_e2e_sc
mv models/exp_phase_e2e_sc.backup_v1 models/exp_phase_e2e_sc

# 恢复旧配置
git checkout config/exp_phase_mpl.json
git checkout train/models.py
```

---

## 六、关键注意事项与陷阱 ⚠️

在实施过程中，以下几个细节问题需要特别注意，否则可能导致训练成功但推理失效，或速度控制不符合预期。

### 6.1 坐标系一致性问题 🔴 **高优先级**

#### 问题描述
文档默认 `RootVelocity` 为平面 2D 向量 `[vx, vy]`，但**未明确是在世界坐标还是角色朝向坐标**。

#### 风险
- 如果 `RootVelocity` 与 `cond Direction` 的坐标系不一致
- 推理时会出现**方向偏转/漂移**
- 例如：cond给定向右移动，但速度向量在旋转后的局部坐标系，导致实际运动方向错误

#### 解决方案

**当前系统使用的坐标系** (需确认):
```python
# 检查 convert_json_to_npz.py 中的处理
# RootVelocity 应该是 UE 世界坐标系 (右手系, Z-up)
# Direction 也应该是世界坐标系

# 确保在预处理时统一坐标系
```

**推荐方案: 统一使用世界坐标系**
1. `RootVelocity [vx, vy]`: XY 平面的世界坐标速度
2. `Direction [dx, dy]`: XY 平面的世界坐标单位方向
3. 在 `norm_template.json` 中明确注明:

```json
{
  "coordinate_system": "world",  // 新增字段
  "coordinate_convention": {
    "handedness": "right",
    "up_axis": "Z",
    "forward_axis": "X"
  },
  "state_layout": {
    "RootVelocity": {
      "start": 3,
      "size": 2,
      "description": "2D velocity in world XY plane (m/s)",  // 新增
      "coordinate_system": "world"  // 新增
    }
  }
}
```

**验证代码**:
```python
# 在数据预处理后检查
def verify_coordinate_consistency(clip):
    root_vel = clip["root_vel"][:, :2]  # [T, 2]
    cond_dir = clip["cond_direction"]  # [T, 2]

    # 速度方向应该与 cond 方向大致一致
    vel_norm = np.linalg.norm(root_vel, axis=1, keepdims=True)
    vel_dir = root_vel / np.maximum(vel_norm, 1e-6)

    # 余弦相似度应该接近 1.0
    cos_sim = np.sum(vel_dir * cond_dir, axis=1)
    print(f"Direction consistency: mean={cos_sim.mean():.3f}, min={cos_sim.min():.3f}")
    # 期望: mean > 0.9, min > 0.7
```

---

### 6.2 动作基准速度的鲁棒性 🟡 **中优先级**

#### 问题描述
当前方案使用**均值 (mean)** 作为动作基准速度，但如果数据存在以下问题：
- **长尾分布**: walk 中混入少量快速移动片段
- **标注混淆**: walk 片段被错误标注为 run
- **异常值**: 数据采集中的噪声/抖动

则均值会被拉高/拉低，导致倍率语义偏移。

#### 示例
```python
# 假设 walk 数据
speeds = [0.8, 0.85, 0.82, 0.9, 0.88, 2.5, 3.0]  # 最后两个是异常值
mean = 1.26  # ❌ 被异常值拉高
p50  = 0.85  # ✅ 鲁棒

# 使用 mean 时
speed_mult = 0.85 / 1.26 = 0.67  # ❌ 正常速度被识别为减速
# 使用 p50 时
speed_mult = 0.85 / 0.85 = 1.0   # ✅ 正确
```

#### 解决方案

**修改基准速度计算** (`train/convert_json_to_npz.py`):

```python
def compute_action_speed_stats(json_files: list[str]) -> dict:
    # ... (前面的代码不变) ...

    # 统计 - 改用中位数和百分位数
    stats = {}
    for action, speeds in action_speeds.items():
        speeds = np.array(speeds)

        # 过滤异常值 (可选: 使用 IQR 方法)
        q25, q75 = np.percentile(speeds, [25, 75])
        iqr = q75 - q25
        lower_bound = q25 - 1.5 * iqr
        upper_bound = q75 + 1.5 * iqr
        speeds_filtered = speeds[(speeds >= lower_bound) & (speeds <= upper_bound)]

        stats[action] = {
            "mean": float(np.mean(speeds_filtered)),      # 过滤后的均值
            "std": float(np.std(speeds_filtered)),
            "p50": float(np.percentile(speeds_filtered, 50)),  # ✅ 使用中位数作为基准
            "p95": float(np.percentile(speeds_filtered, 95)),  # 上限
            "p05": float(np.percentile(speeds_filtered, 5)),   # 下限
            "count": len(speeds),
            "count_filtered": len(speeds_filtered),
        }

    return stats
```

**使用 p50 作为基准**:
```python
# 修改 cond 构造逻辑
action_name = str(clip.get("action", "unknown")).strip().lower()
if action_name in ACTION_SPEED_STATS:
    base_speed = ACTION_SPEED_STATS[action_name]["p50"]  # ✅ 改为 p50
    max_speed = ACTION_SPEED_STATS[action_name]["p95"]   # 用于上限裁剪
else:
    base_speed = np.percentile(speed, 50) + 1e-6

# 计算倍率并裁剪到合理范围
speed_multiplier = speed / np.clip(base_speed, 1e-3, None)
# 使用 p95 而不是固定的 5.0
max_mult = max_speed / base_speed
speed_multiplier = np.clip(speed_multiplier, 0.0, max(5.0, max_mult))
```

**训练和推理必须使用同一份统计**:
- 统计数据保存到 `norm_template.json` 的 `action_speed_stats` 字段
- 推理时加载相同的配置文件
- **禁止**推理时重新计算统计

---

### 6.3 损失项权重重复性 🟡 **中优先级**

#### 问题描述
当前方案同时使用:
- `root_vel_mse`: 速度向量的 L2 损失 (MSE)
- `root_speed_mae`: 速度大小的 L1 损失 (MAE)

这两个损失项会对**速度部分重复加权**，容易压过旋转损失。

#### 分析
```python
# root_vel_mse
loss1 = MSE([vx, vy], [vx_gt, vy_gt])
      = (vx - vx_gt)^2 + (vy - vy_gt)^2

# root_speed_mae
speed = sqrt(vx^2 + vy^2)
speed_gt = sqrt(vx_gt^2 + vy_gt^2)
loss2 = |speed - speed_gt|

# 两者都在惩罚速度误差，且 loss1 已经隐含了 speed 的约束
```

#### 推荐方案

**阶段1: 只使用 root_vel_mse**
```python
# train/models.py - MotionJointLoss.__init__
w_root_vel = 0.03    # 小权重开始
w_root_speed = 0.0   # ❌ 先不使用
```

**阶段2: 根据训练情况决定是否叠加**
观察训练后:
- 如果 `GeoDeg` 正常 (< 2.5) 且 `RootVelMAE` 较大 (> 0.05)
  - 可以增加 `w_root_vel` 到 0.05-0.08
- 如果速度**方向正确但大小不准** (方向余弦相似度高，但 MAE 高)
  - 再启用 `w_root_speed` (权重极小，如 0.01-0.02)

**监控指标**:
```python
# 训练日志中应该分别记录
metrics = {
    "root_vel_mse": ...,      # 向量误差
    "root_speed_mae": ...,    # 大小误差
    "root_dir_cos": ...,      # 方向余弦相似度 (新增)
    "GeoDeg": ...,            # 旋转误差
}
```

---

### 6.4 输入泄漏/捷径风险 🔴 **高优先级**

#### 问题核心
**当前 State(X) 中已包含 `RootVelocity[3:5]`，Output(Y) 也要预测 `RootVelocity[276:278]`**

模型可能学会**"复制上一帧速度"**而不是根据 `cond` 预测，即:
```python
# 模型的捷径学习
next_velocity = current_velocity  # ❌ 忽略 cond 中的 speed_multiplier
```

#### 验证是否存在捷径
训练后测试:
```python
# 测试1: 改变 cond 中的 speed_multiplier
state[3:5] = [1.0, 0.0]  # 当前速度向右 1.0 m/s
cond[6] = 0.5            # 要求减速到 0.5 倍

output = model(state, cond)
pred_vel = output[276:278]

# 如果 pred_vel ≈ [1.0, 0.0] (复制输入) 而不是 [0.5, 0.0] (根据 cond)
# 说明存在捷径
```

#### 解决方案

**方案1: Free-run 阶段屏蔽 RootVelocity 输入** (推荐)
```python
# train/dataset.py - 在 freerun 模式下
if is_freerun:
    # 将 State 中的 RootVelocity 置零或使用历史平均
    X[:, 3:5] = 0.0  # 强制模型依赖 cond
```

**方案2: 添加噪声干扰** (备选)
```python
# train/dataset.py - 在训练时
if self.is_train:
    # 对 RootVelocity 添加噪声
    noise_scale = 0.1  # 10% 噪声
    X[:, 3:5] += np.random.randn(*X[:, 3:5].shape) * noise_scale
```

**方案3: 延迟一帧** (较复杂)
```python
# 输入使用 t-1 时刻的速度，预测 t 时刻的速度
X_vel = X[:-1, 3:5]  # t-1 时刻
Y_vel = Y[1:, 276:278]  # t 时刻
```

**推荐实施**:
1. **Teacher-forcing 阶段**: 保留真实 RootVelocity，允许模型快速收敛
2. **Free-run 阶段**: 屏蔽或加噪，强制模型学习从 cond 预测

---

### 6.5 归一化模板一致性 🟡 **中优先级**

#### 问题描述
新增 2 维输出后，必须同步更新:
1. `norm_template.json` 的 `output_layout`
2. `DataNormalizer` 的统计信息 (均值/方差)
3. 推理时的反归一化逻辑

否则会出现:
- 训练正常 (归一化后的速度)
- 推理失败 (速度被错误缩放)

#### 检查清单

**Step 1: 确认 output_layout 更新**
```json
// norm_template.json
{
  "output_layout": {
    "BoneRotations6D": {"start": 0, "size": 276},
    "RootVelocity": {"start": 276, "size": 2}  // ✅ 必须存在
  }
}
```

**Step 2: 重新计算归一化统计**
```python
# 数据处理完成后
python train/dataset.py --recompute_norm_stats \
  --data_dir=raw_data/processed_data \
  --output=raw_data/processed_data/norm_template.json
```

**Step 3: 检查 DataNormalizer**
```python
# train/layout.py 或相关文件
normalizer = DataNormalizer(...)
normalizer.update_output_stats(...)  # 确保包含新增的 2 维

# 检查统计维度
assert normalizer.Y_mu.shape[0] == 278, f"Expected 278, got {normalizer.Y_mu.shape[0]}"
assert normalizer.Y_std.shape[0] == 278
```

**Step 4: 推理时反归一化**
```python
# 推理代码
output_normalized = model(state, cond)['out']  # [B, T, 278]
output = normalizer.denormalize_Y(output_normalized)

# 提取速度
root_vel = output[:, :, 276:278]  # 已反归一化,单位 m/s
```

**验证**:
```python
# 检查反归一化后的速度范围是否合理
print(f"RootVel range: [{root_vel.min():.3f}, {root_vel.max():.3f}]")
# 期望: [-5.0, 5.0] 左右 (m/s)
# 如果是 [-0.1, 0.1] 说明归一化有问题
```

---

### 6.6 垂直速度/跳跃场景 🟢 **低优先级 (可选)**

#### 问题描述
当前方案仅输出 2D 速度 `[vx, vy]`，对于有明显垂直运动的场景 (跳跃、落地、上下楼梯)，Z 轴速度仍需靠骨骼姿态间接表达。

#### 影响场景
- **跳跃动作**: 起跳瞬间 vz 很大，落地时 vz < 0
- **上下楼梯**: 持续的垂直运动
- **地形起伏**: 角色在斜坡上移动

#### 当前方案的限制
- 跳跃高度由骨骼姿态隐式控制
- 无法直接控制"跳多高"
- 落地速度不可控

#### 扩展方案 (阶段2)

如果需要显式控制垂直速度:

**扩展 Output 到 279 维**:
```python
output_layout = {
    "BoneRotations6D": {"start": 0, "size": 276},
    "RootVelocity": {"start": 276, "size": 3},  # 改为 3D [vx, vy, vz]
}
```

**扩展 Cond**:
```python
# 对于跳跃动作,添加垂直速度控制
cond = [
    action_onehot,     # [4]
    direction_2d,      # [2]
    speed_mult,        # [1]
    vertical_mult,     # [1] - 新增: 垂直速度倍率 (仅跳跃时非零)
]  # [8维]
```

**当前建议**:
- **阶段1**: 仅处理 2D 速度，跳跃高度由姿态控制
- **阶段2**: 如果游戏需要精确控制跳跃高度，再扩展到 3D

---

### 6.7 推理端速度上限 🟡 **中优先级**

#### 问题描述
当前倍率上限裁剪到固定值 `5.0`:
```python
speed_multiplier = np.clip(speed_multiplier, 0.0, 5.0)
```

但不同游戏的最大速度差异很大:
- 慢节奏游戏: 最大 2 倍速就够
- 快节奏游戏: 可能需要 10 倍速 (冲刺/瞬移)

#### 风险
- 如果游戏需要更高速度但被裁剪,角色会"跑不快"
- 如果训练数据中没有覆盖高倍率,模型外推效果差

#### 解决方案

**Step 1: 确认游戏最大速度需求**
```python
# 与游戏设计沟通
MAX_SPEED_MULTIPLIER = 5.0  # 或更高

# 示例场景
walk_normal = 0.85 m/s, multiplier = 1.0
walk_sprint = 4.25 m/s, multiplier = 5.0  # ← 确认这是否够用
```

**Step 2: 检查训练数据覆盖范围**
```python
# 统计脚本
for action, stats in action_speed_stats.items():
    max_mult = stats["p95"] / stats["p50"]
    print(f"{action}: max_multiplier ≈ {max_mult:.2f}")

# 如果 max_mult < 目标倍率,需要数据增强或放宽上限
```

**Step 3: 动态上限 (推荐)**
```python
# 根据动作的 p95/p50 动态设置上限
if action_name in ACTION_SPEED_STATS:
    base_speed = ACTION_SPEED_STATS[action_name]["p50"]
    max_speed = ACTION_SPEED_STATS[action_name]["p95"]
    max_mult = max(5.0, max_speed / base_speed * 1.2)  # 留 20% 余量
else:
    max_mult = 5.0

speed_multiplier = np.clip(speed_multiplier, 0.0, max_mult)
```

**Step 4: 训练时数据增强**
```python
# 如果需要更高倍率但训练数据不足
# 可以在 teacher-forcing 阶段对 cond 做数据增强
if self.is_train and np.random.rand() < 0.1:  # 10% 概率
    # 随机缩放 speed_multiplier
    C[:, 6] *= np.random.uniform(0.8, 1.5)
    C[:, 6] = np.clip(C[:, 6], 0.0, MAX_MULTIPLIER)
```

---

## 七、调整后的实施顺序 (最小风险方案)

基于以上注意事项,**推荐的实施顺序**调整为:

### 🔵 Phase 0: 预检查 (1天)

1. **坐标系确认** (6.1)
   ```bash
   python scripts/verify_coordinate_system.py --data_dir=raw_data/processed_data
   ```
   - [ ] 确认 RootVelocity 和 Direction 在同一坐标系
   - [ ] 检查方向一致性 (余弦相似度 > 0.9)

2. **速度统计分析** (6.2)
```bash
python - <<'PY'
from pathlib import Path
import json

from train.convert_json_to_npz import find_json_files, compute_action_speed_stats

stats = compute_action_speed_stats(find_json_files("raw_data/source_json"))
out_path = Path("raw_data/processed_data/action_speed_stats.json")
out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
print("[OK] wrote", out_path)
PY
```
   - [ ] 绘制每个动作的速度分布图
   - [ ] 检查异常值/长尾
   - [ ] 决定使用 mean 还是 p50

### 🟢 Phase 1: 最小修改 - 仅输出扩展 (2-3天)

**目标**: 先验证输出速度是否可行,不修改 cond

1. **数据处理**:
   - ✅ 使用 **p50** 作为基准速度
   - ✅ 扩展 Output 到 278 维
   - ✅ 修改 cond 为相对倍率
   - ⚠️ 统一坐标系并在 norm_template.json 注明

2. **损失函数**:
   - ✅ **仅开启 `w_root_vel`** (0.03-0.05)
   - ❌ **不使用 `w_root_speed`**

3. **训练配置**:
   - Teacher-forcing 阶段: 保留真实 RootVelocity 输入
   - **不**做 freerun (快速验证)

4. **验证**:
   - [ ] 训练 3-5 epoch 后检查 `root_vel_mse` 是否下降
   - [ ] 检查 `GeoDeg` 是否正常 (不变差)
   - [ ] **捷径检测**: 改变 cond 看速度是否响应

### 🟡 Phase 2: 处理捷径问题 (1-2天)

**仅在 Phase 1 发现捷径时执行**

1. **检测捷径**:
   ```python
   python scripts/test_shortcut.py --model=models/exp_phase_e2e_sc/model_epoch_05.pt
   ```

2. **如果存在捷径**:
   - ✅ Free-run 阶段屏蔽 State 中的 RootVelocity
   - ✅ 或添加噪声 (noise_scale=0.1)

3. **重新训练并验证**

### 🟢 Phase 3: 完整训练 (7-10天)

1. **完整训练流程**:
   - 所有训练阶段 (teacher → mixed → freerun)
   - 监控 GeoDeg, RootVelMAE, 方向一致性

2. **权重调优**:
   - 如果速度误差偏大,逐步增加 `w_root_vel`
   - 仅在方向对、大小错时启用 `w_root_speed` (极小权重)

3. **验证点**:
   - [ ] GeoDeg < 2.0
   - [ ] RootVelMAE < 0.02
   - [ ] 方向余弦相似度 > 0.95

### 🔵 Phase 4: 推理验证 (1-2天)

1. **归一化检查** (6.5):
   - [ ] 确认 norm_template.json 更新
   - [ ] 检查反归一化后速度范围合理

2. **游戏集成测试**:
   - [ ] 单帧测试: 倍率 0.5/1.0/1.5 线性响应
   - [ ] 方向测试: 无偏转/漂移
   - [ ] 加速/减速平滑性

---

## 八、后续优化方向 (阶段2)

### 8.1 智能动作切换

**场景**: 用户控制run动作,但speed_multiplier=0.3 (很慢)

**当前行为**:
- 模型输出"很慢的跑步动作" (可能不自然)

**优化方向**:
- 自动检测: 当 `actual_speed = base_speed * multiplier` 进入其他动作的速度范围时
- 自动切换: run(mult=0.3) → walk(mult=0.7)
- 平滑混合: 使用blend过渡

**实现思路**:
```python
def auto_switch_action(action: str, speed_mult: float) -> Tuple[str, float]:
    actual_speed = ACTION_SPEED_STATS[action]["mean"] * speed_mult

    # 检查是否需要切换
    for candidate_action, stats in ACTION_SPEED_STATS.items():
        if abs(actual_speed - stats["mean"]) < abs(actual_speed - ACTION_SPEED_STATS[action]["mean"]):
            # 切换到更合适的动作
            new_mult = actual_speed / stats["mean"]
            return candidate_action, new_mult

    return action, speed_mult

# 示例
auto_switch_action("run", 0.3)  # → ("walk", 0.88)
auto_switch_action("walk", 2.0)  # → ("run", 0.68)
```

### 8.2 多动作混合 (Blend)

**场景**: walk和run之间的平滑过渡

**实现**:
```python
def blend_actions(action1: str, action2: str, blend_weight: float, speed_mult: float):
    # 同时调用两个动作的模型
    output1 = model.forward(state, build_cond(action1, direction, speed_mult))
    output2 = model.forward(state, build_cond(action2, direction, speed_mult))

    # 线性混合
    blended_output = output1 * (1 - blend_weight) + output2 * blend_weight

    return blended_output
```

### 8.3 加速度约束

**目标**: 限制速度变化率,避免不物理的突变

**实现**:
```python
# 在损失函数中添加加速度惩罚
accel = (pred_vel[:, 1:] - pred_vel[:, :-1]) * fps  # [B, T-1, 2]
accel_magnitude = torch.norm(accel, dim=-1)         # [B, T-1]
accel_loss = torch.mean(torch.clamp(accel_magnitude - max_accel, min=0.0))
```

---

## 九、总结

### 9.1 核心改进

| 方面 | 原方案 | 新方案 | 收益 |
|------|--------|--------|------|
| **模型输出** | 276维 (仅旋转) | 278维 (旋转+速度) | ✅ 显式速度预测 |
| **Cond速度** | 绝对速度 (m/s) | 相对倍率 (无量纲) | ✅ 语义统一 |
| **速度控制** | 需查表映射 | 直接传倍率 | ✅ 控制简化 |
| **加速/减速** | 不同动作效果不一致 | 所有动作一致 | ✅ 用户体验一致 |

### 9.2 关键指标

**训练指标**:
- GeoDeg < 2.0 (旋转精度不下降)
- root_vel_mse < 0.005 (速度向量误差)
- root_speed_mae < 0.02 (速度大小误差)

**游戏集成指标**:
- 速度倍率控制误差 < 5%
- 加速/减速响应时间 < 3帧
- 无抖动或异常跳变

### 9.3 实施优先级

**P0 (本次必须完成)**:
- [x] 统计动作速度分布
- [x] 修改Cond为相对倍率
- [x] 扩展Output增加速度输出
- [ ] 重新处理数据
- [ ] 修改模型和损失函数
- [ ] 重新训练

**P1 (短期优化)**:
- [ ] 自由运行验证
- [ ] 游戏引擎集成测试
- [ ] 性能优化

**P2 (长期规划)**:
- [ ] 智能动作切换
- [ ] 多动作混合
- [ ] 加速度约束

---

## 十、附录

### 10.1 相关文件清单

| 文件路径 | 修改内容 | 优先级 |
|---------|---------|--------|
| `train/convert_json_to_npz.py` | 速度统计 + 相对倍率 + Output扩展 | P0 |
| `train/models.py` | 损失函数扩展 | P0 |
| `train/training_MPL.py` | 损失权重读取 | P0 |
| `config/exp_phase_mpl.json` | 速度损失权重配置 | P0 |
| `raw_data/processed_data/norm_template.json` | 布局 + 速度统计 | P0 |
| `test_velocity_output.py` | 新建测试脚本 | P1 |

### 10.2 参考资料

- 旋转表示: `train/geometry.py` (rot6d, geodesic distance)
- 数据归一化: `train/layout.py` (DataNormalizer)
- 训练流程: `docs/` (如有相关文档)

### 10.3 问题反馈

如果在实施过程中遇到问题,请检查:

1. **数据维度不匹配**: 检查 state_layout, output_layout 是否一致
2. **损失异常**: 检查速度损失权重是否过大 (建议从小开始)
3. **速度倍率范围**: 检查是否在 0-5 之间,避免极端值
4. **统计数据缺失**: 检查 action_speed_stats.json 是否生成

---

**文档版本历史**:
- v1.0 (2025-11-28): 初始版本,定义阶段1方案
- v1.1 (2025-11-28): 新增"六、关键注意事项与陷阱"章节,包含7个高价值技术细节;调整实施顺序为最小风险方案
