# 72 lr=1e-4 -> lambda

Artifacts:

- runner: `tools/run_72_lowlr_to_lambda.py`
- machine summary: `debug_output/_tmp_72_lowlr_to_lambda_20260315/summary.json`
- readable summary: `debug_output/_tmp_72_lowlr_to_lambda_20260315/summary.md`
- source `72` ckpt: `models/__tmp_72_lowlr_sweep_20260314/lr1e4/ckpt_last_WalkF_stage7_72_lr1e4_from_lowlr71_20260314.pth`
- output `lambda` ckpt: `models/__tmp_72_lowlr_to_lambda_20260315/lambda/ckpt_last_WalkF_stage7_lambda_from_lowlr72lr1e4_20260315.pth`

Scope guard:

- start lane is `candidate 71 (lr=3e-4) -> 72 (lr=1e-4)`
- `lambda` semantics stayed unchanged
- eval contract is model-source only

## Short conclusion

- 已从 `72 lr=1e-4` 成功继续跑到 `lambda`
- 在当前 model-source 口径下，`lambda` 没有引入任何可见变化
  - `candidate lambda` 与输入的 `candidate 72 (lr=1e-4)` **逐项完全一致**
- 这意味着 lower-LR `72` 修复的 aggregate / leg / hotspot 优势被完整保留下来，没有在 `lambda` 再次吐回去
- 相对 current `lambda` / current `72`，candidate lane 仍明显更优

## End-state table

| lane | DirectGeoLocalDeg | all_ex_root | leg | nonleg | arm | legs_main | arms_main | foot_l/ball_l@SIC12-15 | calf_r@SIC2-4 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| candidate `72` (`lr=1e-4`) | 0.101969 | 0.101969 | 0.186385 | 0.083717 | 0.091849 | 0.186385 | 0.091849 | 0.385267 | 0.042300 |
| current `lambda` | 0.112074 | 0.112074 | 0.296389 | 0.072222 | 0.082665 | 0.296389 | 0.082665 | 0.812663 | 0.288880 |
| candidate `lambda` | 0.101969 | 0.101969 | 0.186385 | 0.083717 | 0.091849 | 0.186385 | 0.091849 | 0.385267 | 0.042300 |

## Key deltas

candidate `lambda` vs current `lambda`:

- `DirectGeoLocalDeg=-0.010104`
- `all_ex_root=-0.010104`
- `leg=-0.110005`
- `nonleg=+0.011496`
- `arm=+0.009184`
- `legs_main=-0.110005`
- `arms_main=+0.009184`
- `foot_l/ball_l@SIC12-15=-0.427396`
- `calf_r@SIC2-4=-0.246580`

candidate `lambda` vs candidate `72 (lr=1e-4)`:

- all tracked metrics are exactly `0.000000`

## Readout

- 这条 downstream continuation 没有破坏 `72 lr=1e-4` 的收益
- 在当前评估合同下，`lambda` 对这条 lane 基本是 no-op
- 因此当前最强链路可以直接记为：
  - `candidate 71 (lr=3e-4) -> 72 (lr=1e-4) -> lambda`

## Gait speed white-box follow-up

Artifacts:

- auto-fixed: `debug_output/_tmp_72_lowlr_to_lambda_20260315/eval_lambda_model/Walk_F_gait_speed_scaling_whitebox_auto_fixed.json`
- teacher TD: `debug_output/_tmp_72_lowlr_to_lambda_20260315/eval_lambda_model/Walk_F_gait_speed_scaling_whitebox_teacher_td.json`
- plan TD: `debug_output/_tmp_72_lowlr_to_lambda_20260315/eval_lambda_model/Walk_F_gait_speed_scaling_whitebox_plan.json`
- meas TD: `debug_output/_tmp_72_lowlr_to_lambda_20260315/eval_lambda_model/Walk_F_gait_speed_scaling_whitebox_meas.json`

Short readout:

- `teacher`:
  - 全 scale `pass`
  - `td_unstable=false`, `touchdown_count=5`
  - `freq_hz=0.689655` 固定，`E_cycle_speed_consistency≈2.67e-4`
  - 这是 clean upper bound baseline；周期由 teacher touchdown 锁定
- `plan`:
  - 仅 `1.0x` `pass`
  - 其余 scale 因 `td_unstable=true` 失败，`touchdown_count=4/4/6/7/7`
  - `freq_hz` 在 `1.04 ~ 3.82` 间波动，planner anchor off-scale 还不稳定
- `meas`:
  - 全 scale `fail`
  - `td_unstable=true`, `touchdown_count=26 ~ 31`
  - `freq_hz` 在 `7.78 ~ 12.08`，`E_cycle_speed_consistency=0.086 ~ 0.112`
  - 直接暴露 model-predicted contacts / touchdown 的 chatter 问题

Compact comparison:

| source | td stability | freq behavior | cycle consistency | interpretation |
|---|---|---|---|---|
| `teacher` | stable on all scales | locked at `0.689655` | best (`~2.67e-4`) | upper bound only |
| `plan` | only `1.0x` stable | noisy | medium | planner anchor not yet robust enough |
| `meas` | unstable on all scales | very noisy | worst | current bottleneck / true D-only test |

Takeaway:

- `lambda` lane 本身在 teacher-anchored P0 下是稳定的
- 真正阻塞 D-only 可用性的不是 pose 主体，而是 model-source touchdown stability
- 下一步优先级应转到 `contacts_meas` / touchdown post-process，而不是继续纠结 teacher baseline

## One-sentence answer

- 从 `72 lr=1e-4` 继续跑 `lambda` 后，aggregate 优势没有再回退；在 model-source 口径下，`lambda` 基本不改结果，等于把 `72 lr=1e-4` 的改进原样保留到了最终 `lambda`。
