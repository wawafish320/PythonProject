# [2026-03-16] EventMotionModel refactor Phase C1 执行结果

路线图来源：`docs/changes/2026-03-16_event_motion_model_refactor_roadmap.md`

执行日期：2026-03-17（Asia/Shanghai）

## C1. direct-pose builder 收敛

已在 `train/models.py` 内新增并复用以下入口：

- `_direct_pose_trunk_layers(...)`
- `_init_last_linear(...)`
- `_build_direct_pose_trunk(...)`
- `_build_direct_pose_head(...)`

本轮替换范围：

- split direct-pose trunk（保留 legacy trunk 的两层 `Linear-ReLU-Dropout` 结构）
- non-split `direct_pose_head`
- `direct_pose_leg_head`
- `direct_pose_leg_gate_head`
- `direct_pose_leg_head_shared`
- `direct_pose_leg_gate_head_shared`
- `direct_pose_leg_side_sign_gate_head` 的 last-layer safe init

## 收敛结果

### 1. trunk builder 单一入口

- split / non-split direct-pose 不再分别手写两层 trunk。
- helper 继续返回扁平 `nn.Sequential`，保持现有 `direct_pose_head.0/3/6` 这类 state_dict key 语义不变。

### 2. head builder 单一入口

- 通用 head builder 支持：
  - 标准输出层
  - zero-init last layer
  - bias-init last layer
- learned gate 继续保持 `bias=2.0`
- scale gate 继续保持 `bias=0.0`
- leg residual / shared leg residual 继续保持 zero residual cold start

### 3. direct-pose leg family 收敛

- `direct_pose_leg_head` / `direct_pose_leg_gate_head` 改为共享同一套 MLP builder
- `direct_pose_leg_head_shared` / `direct_pose_leg_gate_head_shared` 改为共享同一套 MLP builder
- side sign gate 复用公共 last-layer init helper，避免再次手写 zero/bias init

## 验证

| label | command | exit | 说明 |
|---|---|---:|---|
| `py_compile` | `python -m py_compile train/models.py` | 0 | Phase C1 语法/导入检查 |
| `debug_contact_loop_module_path` | `python -m train.debug_contact_loop` | 0 | 保持 package-form smoke baseline 正常 |
| `direct_pose_phase_c1_smoke` | 内联 smoke：覆盖 split/non-split、leg/shared-leg/gate/sign-gate init 与最小 forward | 0 | 验证 builder 收敛后 shape、finite、init 语义正常 |

补充说明：

- 本轮未修改默认配置、输出 tensor shape、checkpoint key 兼容策略。
- `direct_pose_head` 仍保持扁平 sequential 布局；split/non-split 路径的读权重与 legacy upgrade 逻辑未被改写。

## 验收结论

- [x] direct-pose trunk 已收敛到公共 builder
- [x] 通用 head builder 已覆盖 zero-init / bias-init 语义
- [x] `direct_pose_leg_head` / `direct_pose_leg_gate_head` 已去除重复手写
- [x] `direct_pose_leg_head_shared` / `direct_pose_leg_gate_head_shared` 已去除重复手写
- [x] `python -m py_compile train/models.py` 通过
- [x] `python -m train.debug_contact_loop` 通过
- [x] direct-pose 最小 init/forward smoke 通过

结论：Phase C1 已完成；下一步应继续执行 Phase C2，收敛 `forward` 内 phase/contact 的双份 scaffold。
