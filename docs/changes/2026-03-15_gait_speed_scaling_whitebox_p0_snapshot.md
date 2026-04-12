# 2026-03-15 Gait Speed Scaling White-Box P0 快照

> Status: snapshot extracted from `docs/gait_speed_scaling_whitebox_evaluation.md`
> Goal: 保留一次性实验结果与读数，不再把这些结果混在 evaluator 规范正文里。

---

## 1. 背景

当前已基于现有 `FreeRunCycleRunner / dataset / model stack` 跑通 P0。

本次验证使用：

- ckpt：`models/__tmp_72_lowlr_to_lambda_20260315/lambda/ckpt_last_WalkF_stage7_lambda_from_lowlr72lr1e4_20260315.pth`
- auto 结果：`debug_output/_tmp_72_lowlr_to_lambda_20260315/eval_lambda_model/Walk_F_gait_speed_scaling_whitebox_auto_fixed.json`
- teacher-touchdown 对照：`debug_output/_tmp_72_lowlr_to_lambda_20260315/eval_lambda_model/Walk_F_gait_speed_scaling_whitebox_teacher_td.json`

说明：

- `auto` 现已改为 `stable_touchdown_v1`
- 对该 `lambda` ckpt，`auto` 会稳定选择 `teacher` 作为 touchdown source
- 因而当前 `auto_fixed` 与 `teacher_td` 的主指标一致

---

## 2. Teacher-Touchdown 指标

| scale | E_speed | R_leg | R_nonleg | E_cycle_speed_consistency | freq_hz | stride_length | E_cycle_all | E_cycle_leg | E_cycle_nonleg | touchdown_source | touchdown_count | td_unstable | status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---|---|
| 0.8 | 0.000000 | 0.995343 | 0.991658 | 0.000267 | 0.689655 | 1.086907 | 0.469487 | 0.695287 | 0.420665 | teacher | 5 | false | pass |
| 0.9 | 0.000000 | 0.998189 | 0.997249 | 0.000267 | 0.689655 | 1.222770 | 0.210071 | 0.322117 | 0.185845 | teacher | 5 | false | pass |
| 1.0 | 0.000000 | 1.000000 | 1.000000 | 0.000267 | 0.689655 | 1.358633 | 0.000000 | 0.000000 | 0.000000 | teacher | 5 | false | pass |
| 1.1 | 0.000000 | 1.004159 | 1.005359 | 0.000267 | 0.689655 | 1.494497 | 0.255337 | 0.347441 | 0.235422 | teacher | 5 | false | pass |
| 1.2 | 0.000000 | 1.008707 | 1.010768 | 0.000268 | 0.689655 | 1.630362 | 0.507458 | 0.682710 | 0.469566 | teacher | 5 | false | pass |

解读：

- `E_speed` 基本为 `0`
- `R_leg / R_nonleg` 现已做真实分组；到 `1.2x` 时分别约为 `1.0087 / 1.0108`，都保持平稳
- `E_cycle_leg` 在 `±0.1x` 为 `0.322 / 0.347`，在 `±0.2x` 为 `0.695 / 0.683`，呈平滑退化而非结构崩坏
- `E_cycle_speed_consistency` 约为 `2.67e-4`，说明 cycle-wise `L/T` 与 `v_pred` 基本一致
- 所有 scale 都无 `td_unstable`，因此当前 `pass` 反映的是 speed scaling 本身稳定，而不是被 touchdown heuristic 误伤
- 在这组 teacher-touchdown 边界下，`freq_hz` 固定在 `0.689655`（period 约 `1.45s`），主要变化体现在 `stride_length`
- 因此这组结果应视为 teacher-anchored upper bound；真正的 `freq / stride` 分解仍需在 model-predicted contacts 下单独复测

---

## 3. `meas vs plan vs teacher` 三路对比

补充产物：

- `meas`: `debug_output/_tmp_72_lowlr_to_lambda_20260315/eval_lambda_model/Walk_F_gait_speed_scaling_whitebox_meas.json`
- `plan`: `debug_output/_tmp_72_lowlr_to_lambda_20260315/eval_lambda_model/Walk_F_gait_speed_scaling_whitebox_plan.json`
- `teacher`: `debug_output/_tmp_72_lowlr_to_lambda_20260315/eval_lambda_model/Walk_F_gait_speed_scaling_whitebox_teacher_td.json`

| source | touchdown 行为 | `freq_hz` 行为 | `E_cycle_speed_consistency` | 结论 |
|---|---|---|---|---|
| `teacher` | 全 scale `td_unstable=false`, `touchdown_count=5` | 固定 `0.689655` | 约 `0.000267` | 干净 upper bound；周期被 teacher TD 锁定 |
| `plan` | 仅 `1.0x` 通过；其余 scale `td_unstable=true`, `touchdown_count=4/4/6/7/7` | `1.04 ~ 3.82`，波动明显 | `0.020 ~ 0.040` | planner anchor 比 `meas` 更干净，但 off-scale 仍不稳定 |
| `meas` | 全 scale `td_unstable=true`, `touchdown_count=26 ~ 31` | `7.78 ~ 12.08`，明显 chattery | `0.086 ~ 0.112` | 直接暴露 model-predicted contacts 的真实问题，是当前主瓶颈 |

细化观察：

- `teacher`
  - 这一路说明 speed scaling 主体是通的
  - 但 `freq_hz` 恒定更多反映 teacher boundary 锁定，而不是模型自发的 `freq / stride` 分解
- `plan`
  - `1.0x` 可过，但 `0.8/0.9/1.1/1.2x` 都因 touchdown 不稳失败
  - `E_cycle_leg` 在非 `1.0x` 约 `1.71 ~ 3.11`，明显高于 teacher
- `meas`
  - 左右脚 rising-edge 数严重失衡，典型如 `0.8x=[44,30]`, `1.2x=[38,26]`
  - `E_cycle_leg` 在高倍率到 `1.53 / 1.62`，说明 cycle boundary 噪声已开始污染腿部 cycle 指标

当前判断：

- `teacher` 给出的是可解释、可复现的 upper bound baseline
- `plan` 还不足以替代 teacher 作为稳定评测边界
- `meas` 才是判断 D-only 是否足够的真正测试点，而它当前明确暴露了 contact/touchdown 稳定性问题
- 对固定 rollout 而言，`R_leg / R_nonleg` 基本不随 touchdown source 改变；真正被 boundary source 改写的是 `freq_hz / stride_length / E_cycle_speed_consistency / E_cycle_leg`
