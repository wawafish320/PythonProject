# [2026-03-16] EventMotionModel refactor Phase D 执行结果

路线图来源：`docs/changes/2026-03-16_event_motion_model_refactor_roadmap.md`

执行日期：2026-03-17（Asia/Shanghai）

## D. 清理与防回归

本轮在 `train/models.py` 内新增并复用以下清理入口：

- `_resolve_direct_pose_contact_layout(...)`
- `_ablate_direct_pose_contact_channel(...)`

并新增 focused regression check：

- `tests/train/test_event_motion_model_refactor_phase_d.py`

## 收敛结果

### 1. 删除旧重复块

- non-routed leg head 内 layout A / layout B 的两版局部 `_ablate(...)` 内联实现已删除。
- direct leg cross-leg ablation 现在统一走：
  - contact feature layout 解析
  - 单 channel ablation 执行

这避免后续新增 ablation mode 或调整 direct input layout 时再维护两份近似实现。

### 2. `_ablate` layout A/B 已统一

- `plan(C) + meas(C?) + phase(2C?)`
- `phase-only (replace_contacts)`

以上两种 layout 现在共用同一套 region 解析与 zero / roll_batch / roll_time 逻辑。

当前代码中已不再保留旧的内联 `_ablate(...)` 双实现；`train/models.py` 中只剩统一 helper 入口。

### 3. focused regression check 已补齐

新增 `tests/train/test_event_motion_model_refactor_phase_d.py`，覆盖：

- event-clock on/off
- direct override path
- split / non-split direct head
- split checkpoint compatibility upgrade
- unified cross-leg ablation helper（concat / replace_contacts 两种 layout）

## 结构变化

相对 Phase C3 报告口径：

- `EventMotionModel.forward` 长度：`1548 -> 1506`
- `train/models.py` 总 LOC：`6146 -> 6193`

说明：

- LOC 略有回升是因为把旧重复块替换成可复用 helper + focused regression check；`forward` 主体继续下降。

## 验证

| label | command | exit | 说明 |
|---|---|---:|---|
| `py_compile` | `python -m py_compile train/models.py tests/train/test_event_motion_model_refactor_phase_d.py` | 0 | Phase D 语法检查 |
| `unit_regression` | `python -m unittest tests.train.test_event_motion_model_refactor_phase_d -v` | 0 | focused regression check 全通过 |
| `debug_contact_loop_module_path` | `python -m train.debug_contact_loop` | 0 | 主链 smoke 复跑通过 |

## 验收结论

- [x] 旧重复块已删除，不再与 helper 双实现并存
- [x] `_ablate` layout A/B 已统一到单一 helper
- [x] focused regression check 已覆盖 event-clock on/off、split/non-split、override path
- [x] split checkpoint compatibility upgrade 未破坏
- [x] 文档与 smoke 路径可复跑

结论：Phase D 已完成，本轮 roadmap 闭环。
