# [2026-03-16] EventMotionModel refactor Phase C3 执行结果

路线图来源：`docs/changes/2026-03-16_event_motion_model_refactor_roadmap.md`

执行日期：2026-03-17（Asia/Shanghai）

## C3. override / direct input 入口收敛

已在 `train/models.py` 内新增并复用以下入口：

- `_prepare_direct_pose_seq_input(...)`
- `_prepare_direct_pose_features(...)`
- `_blend_direct_pose_mode_outputs(...)`
- `_forward_direct_pose_main(...)`

本轮收敛范围：

- `plan_in` 与 `meas_in` 的 detach / drop / noise / override / `(B,T,C)` 对齐改为单一 helper
- `phase_z_in_direct` 的 zero-fallback 与 `(B,T,C)` 对齐收敛到同一入口
- `concat` / `mode_select` / `replace_contacts` 三条 direct input 组装路径收敛到单一 forward helper

## 收敛结果

### 1. direct override 单一入口

- `direct_pose_plan_override` 与 `direct_pose_meas_override` 不再各自手写 canonicalization / zero-ignore / tensor override 逻辑。
- plan override 继续保持：
  - base 可选 detach
  - train-time drop
  - tensor override 支持 `(C)` / `(B,C)` / `(B,T,C)`
- meas override 继续保持：
  - concat 路径 `"ignore"/"zero"` => zeros
  - mode_select 路径 `"ignore"/"zero"` => `None`，后续保持 uniform blend 退化语义

### 2. direct feature / phase hint 入口收敛

- direct head 的 `cond/hidden/hidden_pre/concat` feature 选择改由 `_prepare_direct_pose_features(...)` 统一处理。
- `phase_z_in_direct` 的 zero-fallback、expand、dtype/device 对齐不再散落在 `forward` 主体中。

### 3. direct main path 单 helper

- `concat`
- `mode_select`
- `direct_pose_phase_z_mode='replace_contacts'`

以上三条 direct main path 的输入拼接与 readout/blend 现在都从 `_forward_direct_pose_main(...)` 进入。

## 结构变化

相对 Phase C2 报告口径：

- `EventMotionModel.forward` 长度：`1742 -> 1548`

说明：

- 本轮主要把 direct override 与 input assembly 收敛到 helper；输出 shape、默认配置行为、checkpoint key 未改。

## 验证

| label | command | exit | 说明 |
|---|---|---:|---|
| `py_compile` | `python -m py_compile train/models.py` | 0 | Phase C3 语法/导入检查 |
| `debug_contact_loop_module_path` | `python -m train.debug_contact_loop` | 0 | contact-plan 主链 smoke 通过 |
| `direct_override_smoke` | 内联 smoke：覆盖 `concat + tensor override`、`mode_select + ignore meas`、`replace_contacts + phase_z` | 0 | 验证 plan/meas override 与 direct input 三条主路径保持可用 |

## 验收结论

- [x] `plan_in` override 与 `meas_in` override 已统一到单一 helper
- [x] `concat` / `mode_select` / phase-z replace 路径行为保持不变
- [x] `EventMotionModel.forward` 行数进一步下降
- [x] `python -m py_compile train/models.py` 通过
- [x] `python -m train.debug_contact_loop` 通过
- [x] direct override targeted smoke 通过

结论：Phase C3 已完成；下一步可进入 Phase D，清理旧重复实现并补 focused regression check。
