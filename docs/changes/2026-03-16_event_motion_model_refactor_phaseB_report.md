# [2026-03-16] EventMotionModel refactor Phase B 执行结果

路线图来源：`docs/changes/2026-03-16_event_motion_model_refactor_roadmap.md`

执行日期：2026-03-17（Asia/Shanghai）

## B1. tensor canonicalization helper 收敛

已在 `train/models.py` 内新增并复用以下入口：

- `_match_last_dim(...)`
- `_canonicalize_contact_seq_tensor(...)`
- `_canonicalize_contact_step_tensor(...)`

本轮已收敛到单一入口的场景：

- `contacts_input -> (B,T,C)`
- `meas_logits_prev -> (B,C)`
- direct-pose `plan_in` override -> `(B,T,C)`
- direct-pose `meas_in` override -> `(B,T,C)`
- forward 尾部 `contacts_meas` 导出路径的 shape 对齐

结果：

- event-clock on/off 两条路径不再各自维护一套 `contacts_input` / `meas_logits_prev` pad-expand-trim 骨架。
- direct-pose override 与 contact-plan 主链共用同一类 shape 对齐语义，降低静默漂移风险。

## B2. phase-state init helper 收敛

已在 `train/models.py` 内新增并复用以下入口：

- `_canonicalize_phase_z_input(...)`
- `_init_phase_state(...)`

统一覆盖内容：

- 外部 `phase_z` 规范化到 `(B, 2*C)`
- `phase=0 -> [sin=0, cos=1]` anchor fallback
- `obs` / `learnable+obs` 初始化分支
- `phase_event_age -> (B,C)` 规范化与 `clamp_min(0)`
- `min_interval` 默认初始化
- `leg_side_cue_mode == "phase_event_age"` 时的 stateful age fallback

复用位置：

- event-clock on 路径的 phase-state init
- event-clock off 路径的 phase-state init

结果：

- `phase_z` / `phase_event_age` 初始化只保留一套 source of truth。
- Phase A 标记的双份初始化骨架已收敛到公共 helper。

## B3. rot6d joint-count 单一入口

已将以下分支改为复用 `_resolve_rot6d_joint_count(...)`：

- `lambda_fusion_joint_count`
- `so3_corr_joint_count`

结果：

- `BoneRotations6D` slice 与 `bone_names` fallback 的解析逻辑不再各自手写。
- 后续若 rot6d layout 规则变动，只需维护单一 helper。

## 验证

| label | command | exit | 说明 |
|---|---|---:|---|
| `py_compile` | `python -m py_compile train/models.py` | 0 | Phase B 基本语法/导入检查 |
| `debug_contact_loop_script_path` | `python train/debug_contact_loop.py` | 1 | 维持 Phase A 已冻结现状；repo root 下仍因 `ModuleNotFoundError: No module named 'train'` 失败，不是本轮回归 |
| `debug_contact_loop_module_path` | `python -m train.debug_contact_loop` | 0 | 可复跑 package-form smoke baseline |
| `direct_phase_override_smoke` | 内联 smoke：`direct_pose + phase_state + override`，覆盖 `use_event_clock=False/True` | 0 | 同 seed / 同输入下关键输出 shape 与 finite 性正常 |

补充说明：

- 本轮没有修改默认配置、输出 tensor shape、checkpoint key。
- 额外 smoke 覆盖了 `phase_z`、`phase_event_age`、`meas_logits_prev`、plan/meas override 的 helper 收敛后主链行为。

## 验收结论

- [x] `contacts_input` / `meas_logits_prev` / override 三类 canonicalization 已收敛到单一入口
- [x] phase-state init 已收敛到单一 helper，并被 event-clock on/off 共同复用
- [x] `lambda_fusion_joint_count` / `so3_corr_joint_count` 已统一到 `_resolve_rot6d_joint_count(...)`
- [x] `python -m py_compile train/models.py` 通过
- [x] 同 seed / 同输入下关键输出 shape 不变（经 targeted smoke 验证）
- [ ] `python train/debug_contact_loop.py` 仍保持 Phase A 已冻结的脚本路径失败现状；当前可复跑等价 smoke 仍为 `python -m train.debug_contact_loop`

结论：Phase B 的代码级目标已完成；当前唯一未勾选项是 Phase A 已确认的脚本路径导入问题，未在本轮 refactor 中处理。
