# meas contact head 在 speed-scaling whitebox 下的非平滑抖动

> Created: 2026-05-05
> Status: open
> Owner: TBD
> Scope: contact_meas head 输出稳定性，独立于 speed-scaling evaluator gate 校准

---

## 0. 一句话

在 `final_lambda_0504_71` ckpt 上，meas 路径的 touchdown best-channel 在 `0.8 / 1.0 / 1.2x` 上反复触发 `td_best_channel_clean=False`，但在 `0.9 / 1.1x` 上又恢复正常。这种**跨 scale 不平滑的失稳模式**指向 contact_meas head 输出抖动，而不是 speed-scaling 本身的问题。

---

## 1. 来源 / 触发评测

evaluator：`train/validate/run_gait_speed_scaling_whitebox.py`（v4）

ckpt：`debug_output/_tmp_71_lr1e4_lowlr_downstream_20260504/`（runbook §15 默认 final lambda）

产物：

- `debug_output/_tmp_speed_eval_20260505/final_lambda_0504_71_whitebox_v4_meas.json`
- `debug_output/_tmp_speed_eval_20260505/final_lambda_0504_71_whitebox_v4_plan.json`（对照）

---

## 2. 现象

### 2.1 meas vs plan 同 ckpt 同 scale 对比

| scale | meas freq_cv | plan freq_cv | meas stride_cv | plan stride_cv | meas E_cycle | plan E_cycle | meas clean | plan clean | meas status | plan status |
|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|
| 0.8 | 0.762 | 0.339 | 0.570 | 0.320 | 0.033 | 0.026 | False | True  | fail | pass |
| 0.9 | 0.336 | 0.347 | 0.328 | 0.324 | 0.021 | 0.027 | True  | True  | pass | pass |
| 1.0 | 1.037 | 0.345 | 0.552 | 0.321 | 0.044 | 0.027 | False | True  | fail | pass |
| 1.1 | 0.343 | 0.351 | 0.327 | 0.321 | 0.021 | 0.026 | True  | True  | warn | pass |
| 1.2 | 0.954 | 0.458 | 0.694 | 0.368 | 0.030 | 0.009 | False | True  | fail | warn |

注：`v4` 已修正 evaluator 校准（`auto` 不再回退 teacher、CV 改为 per-source 相对 1.0x、`td_best_channel_clean=False` 直接 fail）。所以此处 fail 不是 evaluator bug。

### 2.2 关键观察

1. **plan 全程稳**：所有 scale `clean=True`、`freq_cv ≈ 0.32 ~ 0.46`、`E_cycle ≤ 0.03`，和 baseline 1.0x 同口径。
2. **meas 在 0.8 / 1.0 / 1.2 上跳变**：`freq_cv` 飙到 `0.76 ~ 1.04`、`stride_cv` 到 `0.55 ~ 0.69`，**比 plan 高 2~3×**。
3. **meas 在 0.9 / 1.1 上恢复**：`freq_cv ≈ 0.34`、`stride_cv ≈ 0.33`，几乎和 plan 一致。
4. **失稳模式不是 speed scaling 单调函数**：如果是 speed scaling 把 meas 推出工作区，应该是 `|s - 1| ↑` 越坏；
   现在却是 `0.8 / 1.0 / 1.2x` 一组坏、`0.9 / 1.1x` 一组好，这种隔点失稳更像是 contact_meas head 自身的离散抖动被 best-channel 选择放大。
5. **`E_cycle_speed_consistency` 在 fail 时也只升到 `0.03 ~ 0.04`**：cycle 内部 L/T 与 v_pred 仍能解释，所以失稳不在运动学层面，
   集中在 touchdown 事件检测层面。

### 2.3 evaluator 端读数解释

- 触发 fail 的是 `td_best_channel_clean=False`，即 best-channel 选完之后 `count_error > 1` 或 `interval_cv > 0.50` 仍命中。
- best-channel 选择已经在 `meas / plan` 之间排过序（`auto` 模式），但本 issue 是 **`--contact-source meas` 显式指定**，
  所以选源策略不参与；问题落在 meas channel 自己。

---

## 3. 假设与排除

### 3.1 不是 speed-scaling evaluator 校准问题

v4 已经把 evaluator 校准过：
- `auto` 不再回退 teacher
- CV 改为 per-source 相对 1.0x（warn ratio>1.3 / fail ratio>1.6）
- `td_best_channel_clean=False` 直接 fail
- `td_channel_diverge` 仅在 `clean=True` 时降级为 warn

同 ckpt 同 rollout 下 plan 全 pass、meas 在等距 scale 上隔点失稳，**问题不在 evaluator**。

### 3.2 不是 meas head 在变速下整体退化

如果 meas head 在 0.8 / 1.2x 整体退化，应该看到平滑单调的 CV 上升曲线；
现在 0.9 / 1.1x 反而完全正常，这不符合"模型对变速 robustness 不足"的图像。

### 3.3 假设 A：meas head 输出在 0.5 阈值附近毛刺，best-channel 选择放大

- meas 输出是 contact 概率/逻辑值，在 `touchdown_threshold = 0.5` 附近的小幅噪声会触发 rising-edge 抖动。
- 不同 scale 下 phase / cycle alignment 会让噪声落在阈值不同侧 → 隔点失稳。
- 验证手段：series 文件里看 contacts_meas raw 时序；用 smoothed 通道（已加 raw / smoothed 双路）复算 td_count，
  看 fail 的 scale 是否被 smoothed 修复。

### 3.4 假设 B：meas head 在某些 phase alignment 下产生双跳变

- 不是阈值噪声，而是模型在特定 phase 下输出双 bump（比如 left/right channel 在同一窗口都 rising）。
- 若如此，smoothed 不能修复，需要回到 head 训练侧。

### 3.5 假设 C：与 dataset 内的 walk_F clip phase coverage 相关

- `0.9 / 1.1x` 落在 clip 数据更密集的 phase 区段，meas head 在该区段内插稳定；
- `0.8 / 1.0 / 1.2x` 落在 phase 边缘或外推区段，meas head 输出毛刺。
- 验证手段：用其他 walk clip（若有 forward / turning 变体）复跑相同 scale 集合，看抖动 scale 模式是否一致。

---

## 4. 下一步建议

按成本由低到高：

1. **看 series 直接证据**：从 `final_lambda_0504_71_whitebox_meas_s11_series.json`（已存在）提 `contacts_meas` 时序，
   在 fail 的 scale 上画 raw vs threshold 跨越曲线。30 分钟工作量。
2. **跑 raw vs smoothed 对比**：v4 已经导出 smoothed 通道（3-frame majority vote）。
   对比 `td_count_smoothed` 是否在 `0.8 / 1.0 / 1.2` 上回到合理区间。
   - 如果是 → 假设 A 成立，问题在 deploy 侧 contact debounce，不是模型问题。
   - 如果不是 → 走假设 B，需要回 head 训练侧。
3. **跨 ckpt 对照**：用 healthy anchor `_tmp_tail_top7_fresh_chain_20260418_074813` 同口径跑 meas whitebox v4，
   看是历史一直如此还是 0504 71 lowlr 引入的回退。
4. **跨 clip 对照**：若有可用的非 walk_F clip（forward / turning），同 scale 集复跑，验证假设 C。

---

## 5. 与 speed-scaling 主线的关系

**正交，不阻塞 speed-scaling 收口**：

- `docs/changes/2026-03-25_contact_plan_event_clock_whitebox_mainline_evidence.md` §2.2 已把 v4 plan 路径作为 D-only ±10% 稳定的支撑数据。
- 该 issue 处理的 meas 抖动只在以 meas 为 contact_source 显式评测时显现；
- gate 模式 `auto` 已经会选 plan，因此对当前 lambda gate 决策不构成阻塞。
- 但若未来要把 meas 作为 deploy-time 默认 contact 源（而非 plan），这个抖动必须先处理。

---

## 6. 关联文档

| 文档 | 关系 |
|---|---|
| `docs/gait_speed_scaling_whitebox_evaluation.md` §5.3 / §11.3 / §13 | evaluator 规范，已与本 issue 触发的 v4 改动对齐 |
| `docs/changes/2026-03-25_contact_plan_event_clock_whitebox_mainline_evidence.md` §2.2 | 给出 v4 数据全景 |
| `train/validate/run_gait_speed_scaling_whitebox.py` | evaluator 实现（v4） |
| `docs/basetrain_to_posttrain_top7_fresh_chain_runbook.md` §15 | 默认 final lambda 产物源 |
