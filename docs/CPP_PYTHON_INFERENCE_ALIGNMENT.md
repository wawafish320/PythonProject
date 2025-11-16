# C++ 推理与 Python 训练对接文档

## 概述

本文档详细说明 Unreal Engine C++ 推理代码与 Python 训练代码之间的数据处理流程对接，特别是骨骼旋转的 6D 表示、归一化/反归一化、以及 delta 组合的关键步骤。

**重要性：** 任何对 C++ 推理或 Python 训练流程的修改都必须保持两者的一致性，否则会导致推理结果与训练不匹配。

---

## 1. 模型训练策略

### 1.1 残差学习 (Residual Learning)

模型学习的是**残差 delta**，而不是绝对旋转：

```python
# Python 训练时的归一化 (train/geometry.py)
Y_norm = (Y_abs - MuY) / StdY
```

其中：
- `Y_abs`: 下一帧的绝对骨骼旋转
- `MuY`: 训练数据的均值（保存在 `norm_template.json`）
- `StdY`: 训练数据的标准差
- `Y_norm`: 归一化后的值（模型的输出目标）

### 1.2 关键发现：MuY 不是单位旋转

**重要：** `MuY` 包含的是训练数据的平均姿态，**不是**标准单位旋转 `[1,0,0, 0,0,1]`。

例如，第一个骨骼的 MuY（来自 `norm_template.json`）：
```json
"MuY": [
  -0.207665, 0.977394, -0.000387,  // X 列（约 90 度旋转）
  -0.019582, -0.002247, 0.998585   // Z 列
]
```

这意味着训练数据的平均姿态相对于单位旋转有约 90 度的偏移。

---

## 2. Python 推理流程

### 2.1 完整流程 (run_teacher_rollout.py)

```python
# 步骤 1: 部分反归一化（只乘标准差，不加均值）
if std_y is not None:
    delta_raw = delta_norm * std_y  # 只乘 std，不加 mean！
else:
    delta_raw = delta_norm

# 步骤 2: 使用 compose_rot6d_delta 组合
y_raw = compose_rot6d_delta(y_raw_local, delta_raw[:, rot_y_slice])
```

### 2.2 Delta 组合细节 (train/geometry.py)

```python
def compose_rot6d_delta(prev_rot6d, delta_rot6d, *, columns=("X", "Z")):
    """
    组合旋转 delta：next = delta @ prev
    关键：delta 需要先 normalize（加 identity + reproject）
    """
    # 步骤 2a: normalize delta（加单位旋转 + 投影到 SO(3)）
    delta_normalized = normalize_rot6d_delta(delta_rot6d, columns=columns)

    # 步骤 2b: 转成矩阵并组合
    prev_mat = rot6d_to_matrix(prev_rot6d)
    delta_mat = rot6d_to_matrix(delta_normalized)
    next_mat = delta_mat @ prev_mat  # 矩阵乘法

    # 步骤 2c: 转回 6D
    return matrix_to_rot6d(next_mat)
```

### 2.3 Delta Normalization 关键步骤

```python
def normalize_rot6d_delta(delta_flat, *, columns=("X", "Z")):
    """
    将残差 delta 转换为有效的旋转 delta
    """
    orig = delta_flat.shape
    J = orig[-1] // 6
    residual = delta_flat.view(*orig[:-1], J, 6)

    # 步骤 A: 加单位旋转
    # identity = [1,0,0, 0,0,1] for columns=("X", "Z")
    delta_with_identity = residual + _rot6d_identity_like(residual, columns=columns)

    delta_flat = delta_with_identity.view(*orig[:-1], 6 * J)

    # 步骤 B: Reproject 到 SO(3) 流形（关键！）
    # 加 identity 后的数据不一定是有效的 6D 旋转，需要投影
    delta_proj = reproject_rot6d(delta_flat)  # Line 139 - 非常关键！

    return delta_proj.view(*orig[:-1], J, 6)
```

**关键点：**
1. 加 identity：`residual + [1,0,0, 0,0,1]`
2. **Reproject（第 139 行）：** 使用 Gram-Schmidt 正交化投影到 SO(3) 流形，确保结果是有效的旋转

---

## 3. C++ 推理流程

### 3.1 完整流程对应

C++ 必须**精确匹配** Python 的每一步：

| Python 步骤 | C++ 对应函数 | 代码位置 |
|------------|-------------|---------|
| 部分反归一化（delta_raw = Y_norm * std） | `DenormY_Z_To_Raw` | `EnemyAnimInstance.cpp:1498-1519` |
| 加 identity | `ComposeDeltaWithPrev` | `EnemyAnimInstance.cpp:2092-2095` |
| **Reproject delta** | `ComposeDeltaWithPrev` | `EnemyAnimInstance.cpp:2097-2107` |
| 矩阵组合（delta @ prev） | `ComposeDeltaWithPrev` | `EnemyAnimInstance.cpp:2113-2114` |

### 3.2 C++ 实现细节

#### 3.2.1 部分反归一化

```cpp
// reasoning/EnemyAnimInstance.cpp:1498-1519
void UEnemyAnimInstance::DenormY_Z_To_Raw(const TArray<float>& Y_norm, TArray<float>& OutRaw) const
{
    OutRaw.SetNum(OutputDim);

    // 关键：BoneRotations6D 特殊处理
    const FStateSlice* RotSlice = OutputLayout.Find(TEXT("BoneRotations6D"));

    for (int32 i=0; i<OutputDim; ++i)
    {
        const float sd = StdY.IsValidIndex(i) ? StdY[i] : 1.f;
        const float zn = Y_norm.IsValidIndex(i) ? Y_norm[i] : 0.f;

        // BoneRotations6D: 只乘 std (residual)，后续会在 compose 时加 identity
        if (RotSlice && i >= RotSlice->Start && i < RotSlice->Start + RotSlice->Size)
        {
            OutRaw[i] = zn * sd;  // 只乘 std，不加 MuY！
        }
        else
        {
            // 其他通道：标准反归一化 (乘 std + 加 mean)
            const float mu = MuY.IsValidIndex(i) ? MuY[i] : 0.f;
            OutRaw[i] = zn * sd + mu;
        }
    }
    // ... 其他通道的特殊处理（RootYaw、RootVelocity 等）
}
```

**关键：** `BoneRotations6D` 只做 `OutRaw[i] = Y_norm[i] * std[i]`，**不加** `MuY[i]`。

#### 3.2.2 Delta 组合（加 Identity + Reproject + 矩阵乘法）

```cpp
// reasoning/EnemyAnimInstance.cpp:2064-2120
auto ComposeDeltaWithPrev = [&]() -> void
{
    const FStateSlice* SX_rot = StateLayout.Find(TEXT("BoneRotations6D"));
    const int32 NBlocks = SY_rot->Size / 6;

    for (int32 b = 0; b < NBlocks; ++b)
    {
        const int32 PrevIdx = SX_rot->Start + b * 6;
        const int32 DeltaIdx = SY_rot->Start + b * 6;
        float DeltaRaw[6];
        for (int32 k = 0; k < 6; ++k)
        {
            DeltaRaw[k] = MotionDenorm.IsValidIndex(DeltaIdx + k) ? MotionDenorm[DeltaIdx + k] : 0.f;
        }

        // 步骤 1: 加 identity（对应 Python geometry.py:137）
        // identity 6D = [1,0,0, 0,0,1] for columns=(X,Z)
        DeltaRaw[0] += 1.0f;  // X column: x component
        DeltaRaw[5] += 1.0f;  // Z column: z component

        // 步骤 2: Reproject delta（对应 Python geometry.py:139）
        // 关键！加 identity 后的数据不一定是有效的 6D，需要投影到 SO(3) 流形
        FVector X(DeltaRaw[0], DeltaRaw[1], DeltaRaw[2]);
        FVector Z(DeltaRaw[3], DeltaRaw[4], DeltaRaw[5]);

        // Gram-Schmidt 正交化
        X = X.GetSafeNormal();
        if (X.IsNearlyZero()) X = FVector::ForwardVector;

        Z = (Z - FVector::DotProduct(Z, X)*X).GetSafeNormal();
        if (Z.IsNearlyZero()) Z = FVector::UpVector;

        FVector Y = FVector::CrossProduct(Z, X).GetSafeNormal();
        if (FVector::DotProduct(X, FVector::CrossProduct(Y, Z)) < 0.f) Y *= -1.f;
        Z = FVector::CrossProduct(X, Y).GetSafeNormal();

        // 写回 DeltaRaw
        DeltaRaw[0] = X.X; DeltaRaw[1] = X.Y; DeltaRaw[2] = X.Z;
        DeltaRaw[3] = Z.X; DeltaRaw[4] = Z.Y; DeltaRaw[5] = Z.Z;

        // 步骤 3: 矩阵组合 next = delta @ prev
        FMatrix PrevM, DeltaM;
        DecodeRot6DToMatrix(&Prev_X_raw[PrevIdx], PrevM);
        DecodeRot6DToMatrix(DeltaRaw, DeltaM);

        const FMatrix NextM = DeltaM * PrevM;
        EncodeMatrixToRot6D(NextM, &MotionDenorm[DeltaIdx]);
    }
};
```

---

## 4. 常见错误和调试

### 4.1 错误 1：对 BoneRotations6D 做了完整反归一化

**症状：** 骨骼旋转有约 90 度的偏移（例如 spine_01 Yaw=88 度）

**错误代码：**
```cpp
// 错误！
OutRaw[i] = Y_norm[i] * std[i] + MuY[i];  // 对 BoneRotations6D 加了 MuY
```

**原因：** MuY 包含训练数据的平均姿态（约 90 度旋转），不是单位旋转。加了 MuY 会导致双重旋转。

**正确做法：**
```cpp
// 正确！
if (是 BoneRotations6D)
    OutRaw[i] = Y_norm[i] * std[i];  // 只乘 std
else
    OutRaw[i] = Y_norm[i] * std[i] + MuY[i];  // 其他通道正常
```

### 4.2 错误 2：缺少 Reproject 步骤

**症状：** 骨骼旋转仍有轻微偏移，或者在某些帧出现抖动

**错误代码：**
```cpp
// 错误！缺少 reproject
DeltaRaw[0] += 1.0f;
DeltaRaw[5] += 1.0f;

// 直接使用 DeltaRaw 解码（错误！）
DecodeRot6DToMatrix(DeltaRaw, DeltaM);
```

**原因：** 加 identity 后的数据（例如 `[residual[0]+1, residual[1], residual[2], ...]`）不一定满足 6D 旋转的约束（X, Z 列正交、单位长度）。必须投影到 SO(3) 流形。

**正确做法：**
```cpp
// 正确！加 identity 后立即 reproject
DeltaRaw[0] += 1.0f;
DeltaRaw[5] += 1.0f;

// Gram-Schmidt 正交化（reproject）
FVector X(DeltaRaw[0], DeltaRaw[1], DeltaRaw[2]);
FVector Z(DeltaRaw[3], DeltaRaw[4], DeltaRaw[5]);
X = X.GetSafeNormal();
Z = (Z - FVector::DotProduct(Z, X)*X).GetSafeNormal();
FVector Y = FVector::CrossProduct(Z, X).GetSafeNormal();
// ... 重新构建正交归一化的 X, Z
```

### 4.3 错误 3：矩阵乘法顺序错误

**症状：** 骨骼旋转完全错误，或者反向

**错误代码：**
```cpp
// 错误！顺序反了
const FMatrix NextM = PrevM * DeltaM;  // 错误
```

**正确做法：**
```cpp
// 正确！delta 在左，prev 在右
const FMatrix NextM = DeltaM * PrevM;  // 正确
```

**原因：** Python 使用 `delta @ prev`（矩阵乘法），C++ 必须匹配。

---

## 5. 验证方法

### 5.1 对比 Teacher 数据

使用 Teacher Forcing 模式（`bUseTeacherForcingEval=true`）时，模型推理结果应该与 `teacher_predictions/*.json` 中的数据匹配。

```cpp
// 对比代码示例（StepModelFused 中已有）
if (bTeacherDriving && TeacherFrames.IsValidIndex(TeacherFrameCursor))
{
    const FTeacherFrame& TeacherFrame = TeacherFrames[TeacherFrameCursor];
    // 对比 MotionReprojected 和 TeacherFrame.RawState 的 BoneRotations6D
}
```

### 5.2 检查关键骨骼的旋转

```cpp
// 检查 spine_01 (骨骼索引 1) 的旋转
// 注意：原始训练数据中 spine_01 的 Yaw 约为 ±90 度（这是正常的！）
```

### 5.3 预期结果

**重要：** 某些骨骼（如 spine_01）在训练数据中本身就有约 90 度的旋转。这**不是 bug**，而是动画数据的特性。

例如，Walk_F 数据中：
- `spine_01` Yaw ≈ -90° (Python) 或 +90° (UE，符号取决于欧拉角约定)
- 这是正常现象，说明代码工作正确

---

## 6. 相关文件

### 6.1 Python 关键文件

| 文件 | 说明 |
|------|------|
| `train/geometry.py:121-140` | `normalize_rot6d_delta` 函数（加 identity + reproject） |
| `train/geometry.py:139` | **关键的 reproject 调用** |
| `train/validate/run_teacher_rollout.py:687-690` | 推理时的部分反归一化 |
| `raw_data/processed_data/norm_template.json` | MuY 和 StdY 数据 |

### 6.2 C++ 关键文件

| 文件 | 代码行 | 说明 |
|------|--------|------|
| `reasoning/EnemyAnimInstance.cpp` | 1498-1519 | `DenormY_Z_To_Raw`（部分反归一化） |
| `reasoning/EnemyAnimInstance.cpp` | 2064-2120 | `ComposeDeltaWithPrev`（加 identity + reproject + 组合） |
| `reasoning/EnemyAnimInstance.cpp` | 2092-2095 | 加 identity |
| `reasoning/EnemyAnimInstance.cpp` | 2097-2107 | **Reproject 步骤（关键！）** |
| `reasoning/EnemyAnimInstance.cpp` | 2113-2114 | 矩阵组合 |

---

## 7. 修改历史

### 最新修复（Commit 49c7341, 45a8b54）

**问题：** C++ 推理的骨骼旋转有约 90 度偏移

**根因：**
1. C++ 对 `BoneRotations6D` 做了完整反归一化（加了 MuY），但 MuY 包含训练数据的平均姿态（约 90 度旋转）
2. C++ 缺少关键的 reproject 步骤

**修复：**
1. 修改 `DenormY_Z_To_Raw`：对 `BoneRotations6D` 只乘 std，不加 MuY
2. 修改 `ComposeDeltaWithPrev`：加 identity 后立即 reproject（Gram-Schmidt 正交化）

**参考 Commits：**
- `45a8b54`: FIX: Align C++ denormalization with Python - partial denorm + identity
- `49c7341`: FIX: Reproject delta before composition (match Python geometry.py:139)

---

## 8. 维护指南

### 8.1 修改 Python 训练代码时

如果修改了以下任何部分，**必须**同步更新 C++：

- [ ] `normalize_rot6d_delta` 的实现
- [ ] `compose_rot6d_delta` 的实现
- [ ] `run_teacher_rollout.py` 中的反归一化逻辑
- [ ] 归一化策略（MuY/StdY 的使用）

### 8.2 修改 C++ 推理代码时

修改前，**必须**确认：

- [ ] 对应的 Python 代码是什么
- [ ] Python 的每一步在 C++ 中都有对应实现
- [ ] 测试：对比 C++ 推理结果和 Python `teacher_predictions/*.json`

### 8.3 添加新的状态通道时

如果在状态中添加新通道（例如新的骨骼属性）：

1. 确认是否需要特殊的归一化/反归一化
2. 更新 `DenormY_Z_To_Raw` 函数
3. 在 `norm_template.json` 中添加对应的 MuY/StdY

---

## 9. 联系人

如果对对接逻辑有疑问，请参考：
- Python 训练代码：`train/geometry.py`, `train/validate/run_teacher_rollout.py`
- C++ 推理代码：`reasoning/EnemyAnimInstance.cpp`
- 本文档维护者：[添加维护者信息]

---

**最后更新：** 2025-11-16
**版本：** 1.0
**状态：** ✅ 已验证正确
