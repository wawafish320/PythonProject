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

## One-sentence answer

- 从 `72 lr=1e-4` 继续跑 `lambda` 后，aggregate 优势没有再回退；在 model-source 口径下，`lambda` 基本不改结果，等于把 `72 lr=1e-4` 的改进原样保留到了最终 `lambda`。
