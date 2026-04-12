# `control-neighborhood boundary probe` Round A 结案

> Status: archived legacy upstream / handoff / control record
> Reader note: this file belongs to the old-boundary upstream-control investigation; any `current`, `default`, `canonical`, `recommend`, or `mainline` wording below is historical context, not present-tense repo policy.
> Current entry points: `docs/posttrain_pipeline.md`, `docs/train_design/2026-04-11_donor_contract_basin_universal_debug_synthesis.md`, `docs/retired_directions/legacy_upstream_handoff_control_family/README.md`

## 1. Scope

本文件只记录 `2026-03-31` 这轮 `control-neighborhood boundary probe` 的 Round A 执行结果与结论。

- primary baseline: `control_denseckpt`
- Round A candidates:
  - `dplus1_orig`
  - `d_lr_hold`
- 不执行：
  - `cplus1/cplus2 keepd`
  - `tail relax`
  - `Stage7/downstream`

source-of-truth:

- `docs/retired_directions/legacy_upstream_handoff_control_family/2026-03-31_phasecd_family_rootcause_and_corig_stage6_validation.md`
- `docs/retired_directions/legacy_upstream_handoff_control_family/2026-03-31_control_boundary_probe_execution_plan.md`

## 2. Executed Commands

本轮没有重跑长 basetrain；直接复用已有 `20260330` basetrain ckpt，并按 runbook 刷新同口径 handoff summary。

```bash
debug_output/_tmp_phasecd_min_ablation_20260330/cpu_nomps_exec.py \
  debug_output/_tmp_phasecd_min_ablation_20260330/run_exact_handoff_epoch_sweep.py \
  --exp-dir models/cp015_phasecd_min_ablation_20260330/exp_phase_DirectBranch_v1_d1_cp015_tailk3_corridor_hold_tail15_phasea050_fixedsched_ep014center_control_denseckpt_seed2024_20260330 \
  --out-root debug_output/_tmp_cp015_control_boundary_probe_20260331/control \
  --epoch-start 10 \
  --epoch-end 15
```

```bash
debug_output/_tmp_phasecd_min_ablation_20260330/cpu_nomps_exec.py \
  debug_output/_tmp_phasecd_min_ablation_20260330/run_exact_handoff_epoch_sweep.py \
  --exp-dir models/cp015_phasecd_min_ablation_20260330/exp_phase_DirectBranch_v1_d1_cp015_tailk3_corridor_hold_tail15_phasea050_fixedsched_ep014center_dplus1_orig_seed2024_20260330 \
  --out-root debug_output/_tmp_cp015_control_boundary_probe_20260331/round_a/dplus1_orig \
  --epoch-start 10 \
  --epoch-end 16
```

```bash
debug_output/_tmp_phasecd_min_ablation_20260330/cpu_nomps_exec.py \
  debug_output/_tmp_phasecd_min_ablation_20260330/run_exact_handoff_epoch_sweep.py \
  --exp-dir models/cp015_phasecd_min_ablation_20260330/exp_phase_DirectBranch_v1_d1_cp015_tailk3_corridor_hold_tail15_phasea050_fixedsched_ep014center_d_lr_hold_seed2024_20260330 \
  --out-root debug_output/_tmp_cp015_control_boundary_probe_20260331/round_a/d_lr_hold \
  --epoch-start 10 \
  --epoch-end 15
```

## 3. Artifacts

- control:
  - `debug_output/_tmp_cp015_control_boundary_probe_20260331/control/handoff_selector_summary.json`
  - `debug_output/_tmp_cp015_control_boundary_probe_20260331/control/epoch015/handoff_eval/Walk_F_freerun_cycles.json`
- `dplus1_orig`:
  - `debug_output/_tmp_cp015_control_boundary_probe_20260331/round_a/dplus1_orig/handoff_selector_summary.json`
  - `debug_output/_tmp_cp015_control_boundary_probe_20260331/round_a/dplus1_orig/epoch016/handoff_eval/Walk_F_freerun_cycles.json`
- `d_lr_hold`:
  - `debug_output/_tmp_cp015_control_boundary_probe_20260331/round_a/d_lr_hold/handoff_selector_summary.json`
  - `debug_output/_tmp_cp015_control_boundary_probe_20260331/round_a/d_lr_hold/epoch015/handoff_eval/Walk_F_freerun_cycles.json`

## 4. Primary Gate Table

runbook primary normalized shape gate:

- `calf_r@SIC2-4 / leg`: lower is better
- `ratio12_24/57_70`: lower is better
- `ratio20_24+49_52/57_70`: lower is better
- `leg SIC57-70 / leg`: higher is better

broad safety check:

- `leg broad mean`
- `all_ex_root`

comparison row:

- control reference = `control_denseckpt epoch015`
- `dplus1_orig` row = `epoch016`
- `d_lr_hold` row = `epoch015`

| candidate | eval json | calf_r@SIC2-4 / leg | ratio12_24/57_70 | ratio20_24+49_52/57_70 | leg SIC57-70 / leg | leg broad mean | all_ex_root | primary improvement count |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `control_denseckpt` | `debug_output/_tmp_cp015_control_boundary_probe_20260331/control/epoch015/handoff_eval/Walk_F_freerun_cycles.json` | 2.024100 | 1.671357 | 2.786622 | 0.727270 | 10.897732 | 6.225274 | reference |
| `dplus1_orig` | `debug_output/_tmp_cp015_control_boundary_probe_20260331/round_a/dplus1_orig/epoch016/handoff_eval/Walk_F_freerun_cycles.json` | 2.092373 | 1.708062 | 2.831470 | 0.715371 | 10.941655 | 6.219251 | 0 / 4 |
| `d_lr_hold` | `debug_output/_tmp_cp015_control_boundary_probe_20260331/round_a/d_lr_hold/epoch015/handoff_eval/Walk_F_freerun_cycles.json` | 2.105219 | 1.803201 | 2.970428 | 0.685673 | 11.082560 | 6.218422 | 0 / 4 |

secondary hotspot tie-break:

- `foot_l/ball_l@SIC12-15 / leg`
  - `control_denseckpt`: `1.344173`
  - `dplus1_orig`: `1.340776`
  - `d_lr_hold`: `1.331214`

这个 secondary 指标不足以覆盖 primary normalized shape 的系统性恶化。

## 5. Round A Verdict

### `dplus1_orig`

判定：`reject`

理由：

- 4 个 primary normalized shape 指标是 `0 / 4` 改善；
- 同时复现 `cplus1` 签名：
  - `ratio12_24/57_70` 上升
  - `ratio20_24+49_52/57_70` 上升
  - `leg SIC57-70 / leg` 下降
- `leg broad mean` 也更差：`10.941655 > 10.897732`

### `d_lr_hold`

判定：`reject`

理由：

- 终点对比 `control` 是 `0 / 4` primary improvement；
- `epoch12-15` 全段都复现 `cplus1` 式前移签名；
- `leg broad mean` 明显更差：`11.082560 > 10.897732`

### Round B Trigger

判定：**不触发**

runbook 条件要求：

- 至少一个 Round A candidate 没有复现 `cplus1` 式前移签名
- 且相对 `control` 在 primary normalized shape gate 上有清晰改善

本轮两个条件都未满足，因此不补：

- `d_cp_hold`
- `d_entry_bridge`

## 6. Final Conclusion

本轮结论可以直接固化为：

> `control_denseckpt` 在当前 fixed boundary 邻域上没有出现值得继续 schedule 微调的正信号。`dplus1_orig` 与 `d_lr_hold` 都是 hard reject，Round B 不触发，因此这条 basetrain 主线进入 `control boundary local optimum / 暂不继续微调` 结论。

由此带来的执行约束：

- 不继续跑 `d_cp_hold`
- 不继续跑 `d_entry_bridge`
- 不让任何 Round A candidate 进入 exact `Stage6-only`
- 后续如果继续推进，应切换到 `control-fixed` 的非 schedule 诊断线
