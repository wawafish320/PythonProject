# C++ ↔️ Python 推理对接手册

这份文档用于约束 **Python 训练导出物** 与 **Unreal Engine (`UEnemyAnimInstance`) 运行时代码** 的接口契约。无论训练算法如何升级、损失项如何增删，只要遵守本文档定义的约束，双方即可保持兼容。

---

## 1. 产物 & 路径

| 产物 | 生成于 Python 端 | 被谁消费 | 备注 |
| --- | --- | --- | --- |
| `raw_data/processed_data/norm_template.json` | `train/convert_json_to_npz.py` + `train/training_MPL.py` | `UEnemyAnimInstance::LoadSchemaAndStats` (`SchemaJsonPath`) | 携带 `state_layout / output_layout / Mu* / Std* / tanh_scales / rot6d_spec / cond_norm_config` |
| ONNX 模型 (`*.onnx` 或 `UNNEModelData`) | `train/training_MPL.py` 导出的 checkpoint/ONNX | `FFusedEventMotionModel::Initialize` | 输入顺序固定：State, Cond, Contacts, AngVel Aux, PoseHist Aux |
| Teacher 参考 (`validate/teacher_predictions/*.json`) | `train/validate/run_teacher_rollout.py` | `UEnemyAnimInstance`（可选） | 用于 Teacher 模式验证 & 回退 |

> **约束**：任何改动布局/特征维度的训练改动，必须同步更新 `norm_template.json` 并在 UE 中重新指定 `SchemaJsonPath`。

---

## 2. 状态/输出/条件切片（Schema 摘要）

以下数据来自 `norm_template.json` 的 `meta.state_layout` / `meta.output_layout`，必须保持命名一致：

| 名称 | 角色 | Start | Size | 备注 |
| --- | --- | --- | --- | --- |
| `RootPosition` | State | 0 | 3 | 仅用于初始化 / 调试，推理循环通常从 `MuX` 复制 |
| `RootVelocity` | State | 3 | 2 | 先除以 `tanh_scales_rootvel`（长度 2），再标准化 |
| `RootYaw` | State | 5 | 1 | `wrap_to_pi` / π |
| `BoneRotations6D` | State | 6 | 276 | 46 骨 × 6D；训练端通过 `geometry.rot6d_to_matrix`，运行时 `DecodeRot6DToMatrix` |
| `BoneAngularVelocities` | State | 282 | 138 | 46 骨 × 3D；先 `tanh(raw / tanh_scales_angvel)` |
| `Contacts` | State (运行时追加) | - | 2 | 左右脚接触标记；由 UE 在 `UpdateFootContacts` 中写入 |
| `PoseHistory` | State (运行时追加) | - | `PoseHistoryLen * tracked_bones * 6` | UE 在运行期维护，统计来自 `PoseHistoryScales/Mu/Std` |
| `Cond` | 条件输入 | 0 | `CondDim`（当前 7） | 4 × action one-hot + 2 × 平面方向 + 1 × 速度，支持 `ECondNormMode` |
| `Output.BoneRotations6D` | 输出 | 0 | 276 | `DenormY_Z_To_Raw` → reproject → 写入 `PredictedLocal` |

> 若未来增加/删除 slice，请同步修改 `train/layout.py` 里的 key、重新生成 schema，并在本文档补充更新。

---

## 3. 推理张量契约

UE C++ 端在 `FFusedEventMotionModel::Forward` 中严格要求 5 个输入张量；其要点如下：

| 输入序号 | 含义 | 维度 | 生产方 |
| --- | --- | --- | --- |
| 0 | `State` | `Dx`（420） | `CarryX_Norm`（上一帧 X_norm，经 `NormalizeXRaw_To_Z` 和自回归更新） |
| 1 | `Cond` | `CondDim`（默认 7） | `BuildCondVector`（Teacher 模式可直接复制 `CurrentTeacherCondRaw`） |
| 2 | `ContactsAux` | 2 | `UpdateFootContacts` 输出，按 `MuX/StdX` 重新白化 |
| 3 | `AngVelAux` | `BoneAngularVelocities` size（138） | 来自 `CurrentTeacherAngVelNorm` 或运行时估计 |
| 4 | `PoseHistAux` | `PoseHistoryLen * tracked_bones * 6` | `BuildPoseHistoryFeature`（维度由 schema 决定） |

输出张量只有 1 个：`Y_norm`（长度 276），对应骨骼 6D 旋转增量。

**强制步骤**：
1. Python 端导出的 ONNX 必须以 `State, Cond, Contacts, AngVel, PoseHist` 的顺序声明输入；否则 UE 无法正确绑定。
2. 运行时在每次 `RunSync` 之后执行：
   - `DenormY_Z_To_Raw`: `Y_norm * StdY + MuY`
   - `reproject_rot6d`: 将 6D 列向量重新正交；det < 0 时翻转派生列
   - `compose_rot6d_delta`: `R_next = ΔR @ R_prev`（Python 同名函数在 `train/geometry.py`）

---

## 4. 训练端要点（可引用现有训练文档）

1. **集中化 Normalizer**：`train/training_MPL.py` 中的 `LayoutCenter` 负责加载 schema、验证 `Dx/Dy`、注入 tanh scales 与 rot6d 配置。若任意 key 缺失，`strict_validate` 会直接抛错，避免“猜测”。
2. **6D 旋转约束**：训练与推理均使用 `columns=("X","Z")`，第三列由 `Z × X` 计算。任何列顺序、手性或 reproject 规则的改动都必须写入 `meta.rot6d_spec` 并在 UE 中读取。
3. **Teacher Rollout**：`train/validate/run_teacher_rollout.py` 读取教师批次（`validate/teacher_batches/*.json`），输出 `validate/teacher_predictions/*.json`。UE 侧在 `bEnableTeacherPlayback` / `bUseTeacherForcingEval` 为真时读取这些 JSON 并通过 `FPoseMailbox` 播放，用于调试或对齐。

---

## 5. 验证流程（建议）

1. **Teacher Forcing 检查**  
   - 运行 `python train/validate/run_teacher_rollout.py --teacher <clip> --model <ckpt> --bundle raw_data/processed_data/norm_template.json --out validate/teacher_predictions`.  
   - UE 中启用 `bEnableTeacherPlayback = true`、`bUseTeacherForcingEval = true`，加载同一 `SchemaJsonPath` 与 `TeacherClipJsonPath`。  
   - 确认 `PoseMailbox` 广播的 `Y_denorm` 与 Python JSON 匹配（可利用日志或自定义 diff 工具）。
2. **Free-run 稳定性**  
   - 禁用 Teacher，确保 `ResetARState` 以最新 schema 初始化。  
   - 监控 `CurrentMotionStateNorm` 是否出现 NaN；若有，检查 `MuX/StdX` 长度或 tanh scale 是否与训练一致。

---

## 6. 变更流程

1. 任何训练侧对 **特征维度**、**归一化策略**、**rot6d 逻辑** 的修改，都必须：  
   - 重新生成 `norm_template.json`（或等价 schema bundle）  
   - 更新 UE 侧资源并验证 `LoadSchemaAndStats` 日志  
   - 在本文件中补充相应章节（例如新的 slice、Cond 维度）
2. 如果仅调整损失权重或训练策略，没有改变输入/输出布局，可在“变更记录”中简单注明版本与影响范围。

---

## 7. 参考实现位置

- Python：`train/training_MPL.py`、`train/geometry.py`、`train/validate/run_teacher_rollout.py`
- Unreal：`Source/Test/Public/Anim/EnemyAnimInstance.h`、`Source/Test/Private/Anim/EnemyAnimInstance.cpp`

保持上述文件的处理逻辑与本文档一致，即可确保 C++ 与 Python 之间的推理结果对齐。
