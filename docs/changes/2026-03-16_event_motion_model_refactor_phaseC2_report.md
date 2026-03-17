# [2026-03-16] EventMotionModel refactor Phase C2 执行结果

路线图来源：`docs/changes/2026-03-16_event_motion_model_refactor_roadmap.md`

执行日期：2026-03-17（Asia/Shanghai）

## C2. forward 中 phase/contact scaffold 收敛

已在 `train/models.py` 内新增并复用以下入口：

- `_prepare_contact_plan_observations(...)`
- `_capture_direct_pose_aux_step(...)`
- `_update_phase_state_step(...)`

本轮收敛范围：

- event-clock on/off 共用一套 `contacts_meas / delta_meas / lr_diff` precompute
- event-clock on/off 共用一套 phase-state init 入口后的 per-step update
- contact-plan rollout 改为单一 `for _t in range(Tq)` scaffold
- event-clock 特有逻辑压缩为 loop 内局部 hook：`gate + corrector + lambda/delta_z cache`

## 收敛结果

### 1. phase/contact precompute 单一入口

- `contacts_input` 与 `meas_logits_prev` 不再在 event-clock on/off 两条路径各自计算一遍。
- `contacts_meas_obs`、`delta_meas_obs`、`lr_diff_obs` 统一由 `_prepare_contact_plan_observations(...)` 产出。
- phase-state init 继续走 Phase B 的 `_init_phase_state(...)`，本轮不再在 on/off 两侧复制预处理骨架。

### 2. contact-plan loop 单 scaffold

- `plan_z_raw -> (optional gate/corrector) -> logits_base -> phase/time residual -> sigmoid(logits)` 现在只保留一套主循环。
- event-clock off 时直接走 `plan_z_t = plan_z_raw`。
- event-clock on 时仅额外执行：
  - `logits_raw`
  - `event_clock_gate(...)`
  - `event_clock_corrector(...)`
  - `lambda/logit/dynamic_prior/delta_z` 缓存

### 3. phase-state update 单一 helper

- `phase_z` 推进、event reset、`phase_event_age` 更新统一收敛到 `_update_phase_state_step(...)`。
- direct-pose 的 `phase_z_in_direct` / `leg_side_cue_in` 每步缓存统一收敛到 `_capture_direct_pose_aux_step(...)`。

## 结构变化

相对 Phase A 固化基线：

- `EventMotionModel.forward` 长度：`2130 -> 1742`
- `train/models.py` 总 LOC：`6177 -> 6053`

说明：

- 本轮主要是将 on/off 双份 scaffold 收敛到 helper 与单 loop，目标是降低维护分叉，而不是大规模改数学逻辑。

## 验证

| label | command | exit | 说明 |
|---|---|---:|---|
| `py_compile` | `python -m py_compile train/models.py` | 0 | Phase C2 语法/导入检查 |
| `debug_contact_loop_module_path` | `python -m train.debug_contact_loop` | 0 | contact-plan 主链 smoke 通过 |
| `phase_contact_scaffold_smoke` | 内联 smoke：覆盖 `use_event_clock=False/True` + `contact_phase_state_enable=True` + direct-pose phase hint | 0 | 验证单 loop scaffold 后 shape、finite、phase/event-clock cache 正常 |

补充说明：

- 本轮未修改默认配置、输出 tensor shape、checkpoint key 兼容策略。
- `python train/debug_contact_loop.py` 仍保持 Phase A 已冻结的脚本路径失败现状，不属于本轮回归。

## 验收结论

- [x] phase/contact precompute 已收敛到公共 helper
- [x] contact-plan per-step loop 已收敛到单一 scaffold
- [x] event-clock on/off 差异已压缩到局部 hook
- [x] `EventMotionModel.forward` 行数已显著下降
- [x] `python -m py_compile train/models.py` 通过
- [x] `python -m train.debug_contact_loop` 通过
- [x] phase/contact targeted smoke 通过

结论：Phase C2 已完成；下一步应继续执行 Phase C3，收敛 direct-pose 的 `plan_in` / `meas_in` override 入口。
