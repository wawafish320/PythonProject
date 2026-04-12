# CP015 `phase_c/phase_d` family：root-cause 判别与 `corig_dtailrelax` Stage6 验证

> Status: archived legacy upstream / handoff / control record
> Reader note: this file belongs to the old-boundary upstream-control investigation; any `current`, `default`, `canonical`, `recommend`, or `mainline` wording below is historical context, not present-tense repo policy.
> Current entry points: `docs/posttrain_pipeline.md`, `docs/train_design/2026-04-11_donor_contract_basin_universal_debug_synthesis.md`, `docs/retired_directions/legacy_upstream_handoff_control_family/README.md`

> Last updated: 2026-03-31
> 目标：把 `2026-03-30` 这轮 `phase_c/phase_d` family 的 basetrain root-cause 分析和唯一补跑的 `corig_dtailrelax_final` full Stage6 验证沉淀到 `docs/train_design`。
>
> 本轮严格遵守以下约束：
>
> - `control_denseckpt` 作为 primary family baseline
> - `old bestfree / old Stage6 exit` 只作 secondary anchor，不主导 family 内判断
> - 优先回答 root-cause，不先扩展 downstream
> - 复用已有 basetrain final ckpt 与 handoff eval；只在缺失处补跑
> - 所有 Python 命令都通过 `debug_output/_tmp_phasecd_min_ablation_20260330/cpu_nomps_exec.py`
> - 本轮不进入 `Stage7/downstream`

---

## 0. TL;DR

这轮最关键的结论有五条：

1. `control_denseckpt` 应继续作为这组 family 的 primary baseline，而不是被旧 baseline 或 `cplus2_keepd` 的单点 Stage6 表现带偏。
2. basetrain handoff 的 family-relative 对照已经足够说明：`cplus1` 方向整体把 entry shape 往更前、更偏的方向推，`phase_c +1` 不是稳定改善 downstream 的 basetrain 方向。
3. `corig_dtailrelax` 在 basetrain handoff 上几乎是 shape-neutral，对 `control` 没有给出清晰的根因级修复信号。
4. `corig_dtailrelax_final` 的 full Stage6 已补跑完成；它虽然在 broad scalar 上略好于 `control_denseckpt_final`，但 normalized shape 在 4 个关键口径里有 3 个更差，因此不能作为更 downstream-friendly 的替代。
5. 当前最稳的 root-cause verdict 是：
   - `phase_c length` 不是应继续推进的主方向；
   - `boundary timing` 仍比 tail weight 更像 primary locus；
   - `phase_d tail weight` 最多只是 weak secondary amplifier，而不是 primary root cause。

因此，本轮不建议推进 `cplus1 + keepd + tailrelax`，也不建议基于这轮结果直接走 `Stage7/downstream`。如果要继续做新的 basetrain schedule，应回到 `control` 邻域，做更细粒度的 boundary-side probe，而不是继续延长 `phase_c`。

---

## 1. 输入与结果文件

### 1.1 Primary family handoff summary

- `debug_output/_tmp_phasecd_min_ablation_20260330/priority_family_final_metrics_20260330.md`

### 1.2 Existing Stage6 full rerun summary

- `debug_output/_tmp_phasecd_stage6_trend_top3_fullrerun_20260330/stage6_trend_top3_fullrerun_summary_20260330.json`
- `debug_output/_tmp_phasecd_stage6_trend_top3_fullrerun_20260330/stage6_trend_top3_fullrerun_summary_20260330.md`

### 1.3 New `corig_dtailrelax_final` full Stage6 outputs

- `debug_output/_tmp_phasecd_stage6_corig_fullrerun_20260330/stage6_corig_fullrerun_report.json`
- `debug_output/_tmp_phasecd_stage6_corig_fullrerun_20260330/stage6_corig_fullrerun_report.md`
- `debug_output/_tmp_phasecd_stage6_corig_fullrerun_20260330/corig_dtailrelax_final/lane.log`
- `debug_output/_tmp_phasecd_stage6_corig_fullrerun_20260330/corig_dtailrelax_final/stage6_freerun/Walk_F_freerun_cycles.json`
- `debug_output/_tmp_phasecd_stage6_corig_fullrerun_20260330/corig_dtailrelax_final/stage6_group_summary.json`
- `models/__tmp_phasecd_stage6_corig_fullrerun_20260330/corig_dtailrelax_final/ckpt_last_corig_dtailrelax_final_stage6_fullrerun_fromscratch_20260330.pth`

### 1.4 Secondary anchor only

- `debug_output/_tmp_stage6_basetrain_compare_20260313/old_bestfree/stage6_freerun/Walk_F_freerun_cycles.json`

---

## 2. 本轮 scope

本轮没有新开 basetrain schedule，也没有把全部 family candidate 推到完整 Stage6。

实际执行的是：

1. 复用 priority 5 个 candidate 的已有 basetrain final ckpt 与 handoff eval，先做 family-relative root-cause 判断。
2. 只对 control-side 的尾权重 probe `corig_dtailrelax_final` 补跑 full Stage6，用来回答：
   - `phase_d tail weight` 是否本身就是主因；
   - 或者它只是放大器。
3. 不推进 `Stage7/downstream`。

这个取舍是有意的，因为在 basetrain handoff 上，`cplus1*` 方向已经没有给出“值得优先消耗 Stage6 算力”的信号。

---

## 3. Family Design Map

### 3.1 Priority 5 candidates

| candidate | config | 相对 `control_denseckpt` 的变化 | test axis | 本轮优先级 |
|---|---|---|---|---|
| `control_denseckpt` | `config/exp_phase_DirectBranch_v1_d1_cp015_tailk3_corridor_hold_tail15_phasea050_fixedsched_ep014center_control_denseckpt_seed2024_20260330.json` | 基线：`phase_c=[10,11]`, `phase_d=[12,15]`, `phase_d rot_local_tail_weight=0.2` | primary baseline | primary |
| `corig_dtailrelax` | `config/exp_phase_DirectBranch_v1_d1_cp015_tailk3_corridor_hold_tail15_phasea050_fixedsched_ep014center_corig_dtailrelax_seed2024_20260330.json` | 保持 `phase_c=[10,11]`, `phase_d=[12,15]`，只把 `phase_d rot_local_tail_weight 0.2 -> 0.1` | `phase_d` tail lock / amplifier | primary |
| `cplus1_dorig` | `config/exp_phase_DirectBranch_v1_d1_cp015_tailk3_corridor_hold_tail15_phasea050_fixedsched_ep014center_cplus1_dorig_seed2024_20260330.json` | `phase_c [10,11] -> [10,12]`，`phase_d [12,15] -> [13,15]`，tail weight 仍为 `0.2` | boundary timing | primary |
| `cplus1_keepd` | `config/exp_phase_DirectBranch_v1_d1_cp015_tailk3_corridor_hold_tail15_phasea050_fixedsched_ep014center_cplus1_keepd_seed2024_20260330.json` | `phase_c [10,11] -> [10,12]`，`phase_d [12,15] -> [13,16]`，保留 `phase_d` duration，tail weight 仍为 `0.2` | `phase_c` length + keep `phase_d` duration | primary |
| `cplus1_dtailrelax` | `config/exp_phase_DirectBranch_v1_d1_cp015_tailk3_corridor_hold_tail15_phasea050_fixedsched_ep014center_cplus1_dtailrelax_seed2024_20260330.json` | `cplus1_dorig` 的 shifted boundary 上再做 `phase_d rot_local_tail_weight 0.2 -> 0.1` | boundary timing + tail relax | primary |

### 3.2 Deferred candidates

| candidate | config | 相对 `control_denseckpt` 的变化 | test axis | 本轮处理 |
|---|---|---|---|---|
| `cplus2_keepd` | `config/exp_phase_DirectBranch_v1_d1_cp015_tailk3_corridor_hold_tail15_phasea050_fixedsched_ep014center_cplus2_keepd_seed2024_20260330.json` | `phase_c [10,11] -> [10,13]`，`phase_d [12,15] -> [14,17]`，保留 `phase_d` duration | 更强的 `phase_c` 延长 probe | deferred |
| `dplus1_orig` | `config/exp_phase_DirectBranch_v1_d1_cp015_tailk3_corridor_hold_tail15_phasea050_fixedsched_ep014center_dplus1_orig_seed2024_20260330.json` | 保持 boundary，`phase_d [12,15] -> [12,16]` | pure `phase_d` duration | deferred |
| `d_lr_hold` | `config/exp_phase_DirectBranch_v1_d1_cp015_tailk3_corridor_hold_tail15_phasea050_fixedsched_ep014center_d_lr_hold_seed2024_20260330.json` | 保持 boundary / duration，`phase_d opt_lr 1e-4 -> 2e-4` | pure `phase_d` LR | deferred |

这张 design map 的重点是：

- `control` vs `cplus1_dorig` 回答 boundary 是否太早；
- `cplus1_dorig` vs `cplus1_keepd` 回答 boundary 延后后，是否还必须保留原 `phase_d` duration；
- `control` vs `corig_dtailrelax` 回答 tail weight 是否独立放大 late-tail 问题；
- `cplus1_dorig` vs `cplus1_dtailrelax` 回答 shifted boundary 上 tail relax 是否有额外价值。

---

## 4. Basetrain Handoff Table

口径说明：

- primary shape metrics：
  - `calf_r@SIC2-4 / leg`
  - `ratio12_24/57_70`
  - `ratio20_24+49_52/57_70`
  - `leg SIC57-70 / leg`
  - `foot_l/ball_l@SIC12-15 / leg`
- secondary raw metrics：
  - `leg broad mean`
  - `leg SIC57-70`
  - `foot_l/ball_l@SIC12-15`
  - `all_ex_root`

数据源统一来自 `debug_output/_tmp_phasecd_min_ablation_20260330/priority_family_final_metrics_20260330.md`。

| candidate | config | final ckpt | handoff json | calf_r@SIC2-4 / leg | ratio12_24/57_70 | ratio20_24+49_52/57_70 | leg SIC57-70 / leg | foot_l/ball_l@SIC12-15 / leg | leg broad mean | all_ex_root | missing |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| `control_denseckpt` | `config/exp_phase_DirectBranch_v1_d1_cp015_tailk3_corridor_hold_tail15_phasea050_fixedsched_ep014center_control_denseckpt_seed2024_20260330.json` | `models/cp015_phasecd_min_ablation_20260330/exp_phase_DirectBranch_v1_d1_cp015_tailk3_corridor_hold_tail15_phasea050_fixedsched_ep014center_control_denseckpt_seed2024_20260330/ckpt_epoch_015.pth` | `debug_output/_tmp_phasecd_min_ablation_20260330/handoff/control_denseckpt/epoch015/handoff_eval/Walk_F_freerun_cycles.json` | 2.024100 | 1.671357 | 2.786622 | 0.727270 | 1.344173 | 10.897732 | 6.225274 | `-` |
| `corig_dtailrelax` | `config/exp_phase_DirectBranch_v1_d1_cp015_tailk3_corridor_hold_tail15_phasea050_fixedsched_ep014center_corig_dtailrelax_seed2024_20260330.json` | `models/cp015_phasecd_min_ablation_20260330/exp_phase_DirectBranch_v1_d1_cp015_tailk3_corridor_hold_tail15_phasea050_fixedsched_ep014center_corig_dtailrelax_seed2024_20260330/ckpt_epoch_015.pth` | `debug_output/_tmp_phasecd_min_ablation_20260330/handoff/corig_dtailrelax/epoch015/handoff_eval/Walk_F_freerun_cycles.json` | 1.998290 | 1.671801 | 2.786535 | 0.726573 | 1.344006 | 10.898305 | 6.228215 | `-` |
| `cplus1_dorig` | `config/exp_phase_DirectBranch_v1_d1_cp015_tailk3_corridor_hold_tail15_phasea050_fixedsched_ep014center_cplus1_dorig_seed2024_20260330.json` | `models/cp015_phasecd_min_ablation_20260330/exp_phase_DirectBranch_v1_d1_cp015_tailk3_corridor_hold_tail15_phasea050_fixedsched_ep014center_cplus1_dorig_seed2024_20260330/ckpt_epoch_015.pth` | `debug_output/_tmp_phasecd_min_ablation_20260330/handoff/cplus1_dorig/epoch015/handoff_eval/Walk_F_freerun_cycles.json` | 2.051922 | 1.706581 | 2.826163 | 0.715694 | 1.340572 | 10.951514 | 6.219819 | `-` |
| `cplus1_keepd` | `config/exp_phase_DirectBranch_v1_d1_cp015_tailk3_corridor_hold_tail15_phasea050_fixedsched_ep014center_cplus1_keepd_seed2024_20260330.json` | `models/cp015_phasecd_min_ablation_20260330/exp_phase_DirectBranch_v1_d1_cp015_tailk3_corridor_hold_tail15_phasea050_fixedsched_ep014center_cplus1_keepd_seed2024_20260330/ckpt_epoch_016.pth` | `debug_output/_tmp_phasecd_min_ablation_20260330/handoff/cplus1_keepd/epoch016/handoff_eval/Walk_F_freerun_cycles.json` | 2.090073 | 1.739323 | 2.868825 | 0.705467 | 1.337789 | 10.986076 | 6.232385 | `-` |
| `cplus1_dtailrelax` | `config/exp_phase_DirectBranch_v1_d1_cp015_tailk3_corridor_hold_tail15_phasea050_fixedsched_ep014center_cplus1_dtailrelax_seed2024_20260330.json` | `models/cp015_phasecd_min_ablation_20260330/exp_phase_DirectBranch_v1_d1_cp015_tailk3_corridor_hold_tail15_phasea050_fixedsched_ep014center_cplus1_dtailrelax_seed2024_20260330/ckpt_epoch_015.pth` | `debug_output/_tmp_phasecd_min_ablation_20260330/handoff/cplus1_dtailrelax/epoch015/handoff_eval/Walk_F_freerun_cycles.json` | 2.019538 | 1.706524 | 2.834087 | 0.715358 | 1.340618 | 10.946705 | 6.237389 | `-` |

从 handoff table 看，当前 family 的主要结构性现象不是“谁 broad 最低”，而是：

- `control` 的 normalized shape 仍然是最稳的 reference；
- `cplus1_dorig / cplus1_keepd / cplus1_dtailrelax` 整体都把 `ratio12_24/57_70`、`ratio20_24+49_52/57_70` 推高，同时把 `leg SIC57-70 / leg` 压低；
- 这更像是 boundary shift 造成的 entry 结构前移，而不是单纯 tail weight 锁出来的假象。

---

## 5. Pairwise Diagnosis

### 5.1 `control_denseckpt` vs `cplus1_dorig`

对照文件：

- `debug_output/_tmp_phasecd_min_ablation_20260330/priority_family_final_metrics_20260330.md`

具体变化：

- 恶化：
  - `calf_r@SIC2-4 / leg`: `2.024100 -> 2.051922`
  - `ratio12_24/57_70`: `1.671357 -> 1.706581`
  - `ratio20_24+49_52/57_70`: `2.786622 -> 2.826163`
  - `leg SIC57-70 / leg`: `0.727270 -> 0.715694`
- 改善：
  - `foot_l/ball_l@SIC12-15 / leg`: `1.344173 -> 1.340572`
  - `all_ex_root`: `6.225274 -> 6.219819`

诊断：

- boundary 延后一拍后，broad scalar 没有塌，但 shape 主指标整体更偏。
- 这不支持“当前 `phase_c -> phase_d` 边界太早，往后挪一拍会更 downstream-friendly”这个假设。
- 支持的反而是：`cplus1` 这类 boundary shift 会把 family entry 进一步往前推。

### 5.2 `cplus1_dorig` vs `cplus1_keepd`

对照文件：

- `debug_output/_tmp_phasecd_min_ablation_20260330/priority_family_final_metrics_20260330.md`

具体变化：

- 恶化：
  - `calf_r@SIC2-4 / leg`: `2.051922 -> 2.090073`
  - `ratio12_24/57_70`: `1.706581 -> 1.739323`
  - `ratio20_24+49_52/57_70`: `2.826163 -> 2.868825`
  - `leg SIC57-70 / leg`: `0.715694 -> 0.705467`
  - `leg broad mean`: `10.951514 -> 10.986076`
  - `all_ex_root`: `6.219819 -> 6.232385`
- 改善：
  - `foot_l/ball_l@SIC12-15 / leg`: `1.340572 -> 1.337789`

诊断：

- 在已经做了 `phase_c +1` 的前提下，再保留原 `phase_d` duration 并没有把 shape 拉回去，反而让前移更明显。
- 这说明 `phase_c +1` 之后，保留 `phase_d` duration 不是必要条件，甚至更像是在延续偏移。
- 这直接反对“只要 keepd 就能把 `cplus1` 救回来”的解释。

### 5.3 `control_denseckpt` vs `corig_dtailrelax`

对照文件：

- basetrain handoff:
  - `debug_output/_tmp_phasecd_min_ablation_20260330/priority_family_final_metrics_20260330.md`
- full Stage6:
  - `debug_output/_tmp_phasecd_stage6_trend_top3_fullrerun_20260330/stage6_trend_top3_fullrerun_summary_20260330.json`
  - `debug_output/_tmp_phasecd_stage6_corig_fullrerun_20260330/stage6_corig_fullrerun_report.json`
  - `debug_output/_tmp_phasecd_stage6_corig_fullrerun_20260330/corig_vs_control_stage6_comparison_20260330.md`

basetrain handoff 具体变化：

- 改善：
  - `calf_r@SIC2-4 / leg`: `2.024100 -> 1.998290`
  - `ratio20_24+49_52/57_70`: `2.786622 -> 2.786535`
  - `foot_l/ball_l@SIC12-15 / leg`: `1.344173 -> 1.344006`
- 恶化：
  - `ratio12_24/57_70`: `1.671357 -> 1.671801`
  - `leg SIC57-70 / leg`: `0.727270 -> 0.726573`
  - `leg broad mean`: `10.897732 -> 10.898305`
  - `all_ex_root`: `6.225274 -> 6.228215`

full Stage6 具体变化：

- broad 改善：
  - `all_ex_root`: `0.395504 -> 0.394552`
  - `leg`: `1.023086 -> 0.988743`
- normalized shape 恶化：
  - `calf_r/leg`: `1.416013 -> 3.045000`
  - `ratio20_24+49_52/57_70`: `3.142430 -> 3.304236`
  - `foot_l/ball_l@SIC12-15`: `1.505868 -> 1.639808`
- normalized shape 改善：
  - `ratio12_24/57_70`: `1.710777 -> 1.439108`

诊断：

- basetrain handoff 上，tail relax 几乎是 shape-neutral，没有给出“主因就在 tail weight”的信号。
- full Stage6 上，tail relax 只能换来一点 broad scalar 改善，但 4 个关键 shape 口径中有 3 个更差。
- 这不支持“`phase_d tail weight` 是 root cause”。
- 更合理的解释是：`phase_d tail weight` 最多只是 weak amplifier，能轻微动 broad scale 或个别 ratio，但不能修掉主结构偏移。

secondary anchor only：

- 相对旧 Stage6 exit `debug_output/_tmp_stage6_basetrain_compare_20260313/old_bestfree/stage6_freerun/Walk_F_freerun_cycles.json`，`corig_dtailrelax_final` 的 `final_blended_curve_distance_to_old=0.245792`，而 `control_denseckpt_final` 在 `debug_output/_tmp_phasecd_stage6_trend_top3_fullrerun_20260330/stage6_trend_top3_fullrerun_summary_20260330.json` 中对应 `old_exit_blended_l1=0.244619`。这说明即便只拿旧 exit 当 secondary anchor，`corig` 也没有比 `control` 更靠近旧 clean basin。

### 5.4 `cplus1_dorig` vs `cplus1_dtailrelax`

对照文件：

- `debug_output/_tmp_phasecd_min_ablation_20260330/priority_family_final_metrics_20260330.md`

具体变化：

- 改善：
  - `calf_r@SIC2-4 / leg`: `2.051922 -> 2.019538`
  - `ratio12_24/57_70`: `1.706581 -> 1.706524`
  - `leg broad mean`: `10.951514 -> 10.946705`
- 恶化：
  - `ratio20_24+49_52/57_70`: `2.826163 -> 2.834087`
  - `leg SIC57-70 / leg`: `0.715694 -> 0.715358`
  - `foot_l/ball_l@SIC12-15 / leg`: `1.340572 -> 1.340618`
  - `all_ex_root`: `6.219819 -> 6.237389`

诊断：

- 在 shifted boundary 上再做 tail relax，只能带来很局部的 `calf_r` 改善，但不能把 late-tail ratio 拉回，也不能让整体 entry 重新变得更干净。
- 这支持“boundary shift 才是主要问题，tail relax 只是在边缘处调形”的解释。
- 也就是说，`boundary shift + tail relax` 并没有比单纯 boundary shift 更 downstream-friendly。

---

## 6. `corig_dtailrelax_final` Full Stage6 验证

### 6.1 执行事实

本轮新增的唯一完整 Stage6 运行是 `corig_dtailrelax_final`。

- 运行脚本：
  - `debug_output/_tmp_phasecd_stage6_corig_fullrerun_20260330/run_stage6_corig_fullrerun.py`
- 强制 wrapper：
  - `debug_output/_tmp_phasecd_min_ablation_20260330/cpu_nomps_exec.py`
- 日志：
  - `debug_output/_tmp_phasecd_stage6_corig_fullrerun_20260330/corig_dtailrelax_final/lane.log`

`lane.log` 显示三步都成功结束：

- `train.posttrain`: `exit_code 0`
- `train.validate.run_freerun_cycles`: `exit_code 0`
- `tools/phasea_group_summary.py`: `exit_code 0`

### 6.2 结果摘要

结果文件：

- eval json:
  - `debug_output/_tmp_phasecd_stage6_corig_fullrerun_20260330/corig_dtailrelax_final/stage6_freerun/Walk_F_freerun_cycles.json`
- summary json:
  - `debug_output/_tmp_phasecd_stage6_corig_fullrerun_20260330/corig_dtailrelax_final/stage6_group_summary.json`
- consolidated report:
  - `debug_output/_tmp_phasecd_stage6_corig_fullrerun_20260330/stage6_corig_fullrerun_report.json`

关键数值：

| candidate | all_ex_root | leg | nonleg | arm | else | calf_r/leg | ratio12_24/57_70 | ratio20_24+49_52/57_70 | foot_l/ball_l@SIC12-15 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `control_denseckpt_final` | 0.395504 | 1.023086 | 0.259811 | 0.304031 | 0.155292 | 1.416013 | 1.710777 | 3.142430 | 1.505868 |
| `corig_dtailrelax_final` | 0.394552 | 0.988743 | 0.266078 | 0.289119 | 0.211619 | 3.045000 | 1.439108 | 3.304236 | 1.639808 |

### 6.3 解释

这组 full Stage6 对照的含义非常直接：

- `corig` 没有在 family 内 cleanly beat `control`
- 它只是把 broad scalar 稍微压低了一点，却把最关键的 normalized shape 再次推偏

因此，本轮 `corig` full Stage6 的正确使用方式是：

- 它不是新的 basetrain winner
- 它只是一个 control-side falsification probe
- probe 结果支持：`phase_d tail weight` 不是 primary root cause

---

## 7. Root-Cause Verdict

### 7.1 `phase_c length` 是不是主因

结论：**不是当前最支持的主因方向，而且不值得继续沿 `cplus1/cplus2 keepd` 线性外推。**

依据：

- `control -> cplus1_dorig -> cplus1_keepd` 的 handoff primary shape 指标整体单调变差：
  - `calf_r@SIC2-4 / leg`: `2.024100 -> 2.051922 -> 2.090073`
  - `ratio12_24/57_70`: `1.671357 -> 1.706581 -> 1.739323`
  - `ratio20_24+49_52/57_70`: `2.786622 -> 2.826163 -> 2.868825`
  - `leg SIC57-70 / leg`: `0.727270 -> 0.715694 -> 0.705467`

这组趋势不支持“延长 `phase_c` 会更 downstream-friendly”。

### 7.2 boundary timing 是不是主因

结论：**仍然更像 primary locus。**

这里的含义不是“当前 boundary 必须后移”，而是：

- family 内最明显的 shape shift 是由 boundary-side 改动触发的；
- 一旦把 `phase_c -> phase_d` boundary 往后推，entry 结构马上出现系统性前移；
- 因此真正需要继续精修的，是 boundary 邻域本身，而不是 tail weight 或更长 `phase_c`。

### 7.3 `phase_d tail weight` 是不是放大器

结论：**是 weak secondary amplifier，不是 primary root cause。**

依据：

- `control` vs `corig_dtailrelax` 在 handoff 上几乎不动；
- `cplus1_dorig` vs `cplus1_dtailrelax` 也只出现局部小修正；
- full Stage6 `corig` 仍然没有 cleanly beat `control`。

### 7.4 Primary / Secondary

- primary factor:
  - boundary-side schedule structure
- secondary amplifier:
  - `phase_d tail weight`

---

## 8. Next Action

### 8.1 下一条最值得推进的 basetrain schedule

不是 `cplus1_keepd`，也不是 `corig_dtailrelax`。

本轮更合理的推进方向是：

- 回到 `control_denseckpt` 邻域；
- 设计更细粒度的 boundary-side probe；
- 不再把“继续延长 `phase_c`”当成默认主线。

### 8.2 是否值得再做 `cplus1 + keepd + tailrelax`

结论：**当前不值得。**

理由：

- `cplus1_keepd` 已经显示 `phase_c +1` 后 keep `phase_d` duration 会继续把 shape 往前推；
- `cplus1_dtailrelax` 也没有把 shifted boundary 拉回 clean basin；
- 把两者继续组合只会增加算力成本，不会更快分清 root-cause。

### 8.3 是否还需要回头看 `cplus2_keepd`

结论：**本轮不需要。**

理由：

- `cplus1` 方向在 handoff 上已经给出负面趋势；
- 在没出现相反证据前，继续做更强的 `cplus2` 延长只会强化同一方向的风险；
- `cplus2_keepd` 可以保留为 secondary anchor，但不应主导 family 判断。

### 8.4 是否值得把 winner 推到完整 Stage6

本轮已经完成的 full Stage6 只有：

- existing:
  - `control_denseckpt_final`
- new:
  - `corig_dtailrelax_final`

结论：

- `corig` 不值得继续推更深链路；
- 在出现新的 control-neighborhood boundary probe 之前，不建议额外把当前 `cplus1*` 推到 full Stage6。

### 8.5 本轮禁止项

本轮结论不支持以下动作：

- 直接推进 `Stage7/downstream`
- 把 `cplus2_keepd` 重新当作 basetrain winner
- 把 `tail relax` 误判成 root-cause solution

---

## 9. Final Conclusion

如果只保留一句最关键的话，这轮 family 分析支持的是：

> `cplus2_keepd` 的 Stage6 exit 虽然曾表现出更 downstream-friendly 的一面，但它并不能反推“延长 `phase_c` 是更好的 basetrain schedule 方向”；在 `control_denseckpt` 为 primary baseline 的 family-relative 分析下，真正更像主问题的是 boundary-side schedule structure，而 `phase_d tail weight` 只像次级放大器。

因此，本轮应把 `control_denseckpt` 留作基准，把 `corig_dtailrelax` 视为已完成的 falsification probe，并把下一步工作收敛到新的 control-neighborhood boundary probe，而不是继续把算力投向 `cplus1/cplus2 keepd` 或 downstream。

---

## 10. Addendum：`control-neighborhood boundary probe` 设计

### 10.1 为什么下一轮必须锁回 `control` 邻域

`2026-03-30` 这轮 family 已经回答了一个关键问题：

- `cplus1/cplus2` 这类 probe 不只是“多给 `phase_c` 一拍”
- 它们实际上改变了哪些 epoch 落在 `phase_c`、哪些 epoch 落在 `phase_d`
- 因而也一起改变了 boundary 两侧的局部优化条件

把 `control_denseckpt` 和 `cplus1_dorig` 的 schedule 拆开看，`phase_c -> phase_d` 在 `control` 里的真实差异并不大：

- 不变：
  - `tf_max/tf_min = 0.5/0.5`
  - `ss_chunk_len = 1`
  - `w_rot_local = 0.22`
  - `rot_local_tail_weight = 0.2`
- 真正发生跳变的只有：
  - `opt_lr: 2e-4 -> 1e-4`
  - `w_contact_plan: 0.13 -> 0.10`

这意味着本轮看到的 `cplus1` 前移签名，不能再被粗略表述成“`phase_c` 变长所以变差”。更准确的说法是：

- family 当前最敏感的是 `phase_c -> phase_d` 邻域的局部切换语义；
- 下一轮应优先回答 boundary-local 的 `entry contract` 是否过硬；
- 而不是继续沿 `phase_c +1/+2` 做更大的外推。

### 10.2 新 probe 要回答的不是“boundary 要不要后移”

下一轮 `control-neighborhood boundary probe` 的目标，应从“boundary 往后推一拍会不会更好”改成：

> 在保持 `control_denseckpt` 的 boundary 位置不动时，当前 `phase_d` entry 的局部切换到底是
> 1) residence 不足，
> 2) LR drop 太硬，
> 3) contact-plan release 太硬，
> 还是
> 4) entry 当拍本身需要一个 bridge。

换句话说，下一轮不再测试大步长的 schedule relocation，而是测试：

- fixed boundary 下的 `phase_d` residence
- fixed boundary 下的 `phase_d` LR
- fixed boundary 下的 `phase_d` aux release
- fixed boundary 下的 one-step bridge

并且必须把 `tail weight` 固定住，不再让 `tail relax` 重新混入主判断。

### 10.3 Probe Design Principles

- `control_denseckpt` 继续作为唯一 primary baseline。
- 新 probe 全部保持 `phase_c=[10,11] -> phase_d=[12,*]` 的 boundary 起点不变。
- 新 probe 全部固定：
  - `w_rot_local = 0.22`
  - `rot_local_tail_weight = 0.2`
  - `tf_max/tf_min = 0.5/0.5`
  - `ss_chunk_len = 1`
- 只允许改动 boundary 邻域真正发生跳变的局部字段：
  - `phase_d` duration
  - `phase_d opt_lr`
  - `phase_d w_contact_plan`
  - `epoch12` entry bridge
- 仍然只做 basetrain + handoff；不因为某个 candidate broad scalar 好一点就直接走 `Stage7/downstream`。

### 10.4 Minimal Probe Matrix

建议把下一轮 matrix 收敛成 `control + 2 existing + 2 new` 的最小集合。

| candidate | status | 相对 `control_denseckpt` 的变化 | test axis | 目标问题 |
|---|---|---|---|---|
| `control_denseckpt` | existing baseline | 无变化 | reference | 锁定 primary baseline |
| `dplus1_orig` | existing config | 保持 boundary 在 `12`，只把 `phase_d [12,15] -> [12,16]` | residence length | 当前 boundary 不变时，`phase_d` residence 是否偏短 |
| `d_lr_hold` | existing config | 保持 boundary 与 duration，不把 `phase_d opt_lr` 从 `2e-4` 降到 `1e-4`，而是维持 `2e-4` | LR drop hardness | 当前 entry 是否因为 LR drop 过快而提前锁形 |
| `d_cp_hold` | new | 保持 boundary 与 duration，不把 `phase_d w_contact_plan` 从 `0.13` 降到 `0.10`，而是在整个 `phase_d` 维持 `0.13` | aux release hardness | 当前 entry 是否因为 contact-plan release 过快而前移 |
| `d_entry_bridge` | new | 保持 `phase_d=[12,15]`，但把 `epoch12` 单独拆成 bridge：`opt_lr=2e-4`、`w_contact_plan=0.13`；`13-15` 再回到标准 `phase_d` | entry-local bridge | 问题是不是只出在 boundary 当拍过硬，而不是整个 `phase_d` 语义有误 |

这组矩阵故意不包含：

- `cplus1_keepd`
- `cplus2_keepd`
- 任意 `tailrelax`
- 任意 `downstream` 追加链路

因为这些方向在本轮已经不足以提供更高价值的信息增量。

### 10.5 为什么这 4 个 probe 足够回答 boundary 邻域问题

这四个 probe 对应的是四种互斥程度较高的解释：

1. 如果 `dplus1_orig` 改善而 `d_lr_hold / d_cp_hold` 不改善：
   - 更像是 `phase_d` residence 偏短；
   - 问题不在 boundary 当拍的切换过硬，而在 fixed boundary 后停留不够。
2. 如果 `d_lr_hold` 改善而 `dplus1_orig` 不改善：
   - 更像是 `phase_d` 一进场就降 LR 太快；
   - 问题是 adaptation 不足，而不是 `phase_c` 长度本身。
3. 如果 `d_cp_hold` 改善而 `d_lr_hold` 不改善：
   - 更像是 aux release 太快，contact-plan 支撑在 `phase_d` 初期松得过急。
4. 如果只有 `d_entry_bridge` 改善：
   - 更像是 steady-state `phase_d` 本身没问题；
   - 真正的问题只集中在 `epoch12` 这一拍的 boundary shock。
5. 如果四者都不改善：
   - 当前 `control` boundary 很可能已经处在 local optimum；
   - 这条 basetrain 主线应停止继续微调 boundary 邻域。

### 10.6 推荐执行顺序

不建议一次把全部新 config 都推完。更稳的顺序是两轮：

#### Round A：先跑已有 single-factor probe

- `dplus1_orig`
- `d_lr_hold`

原因：

- 两者配置已存在；
- 都保持 `control` boundary 起点不动；
- 能最快回答“residence”还是“LR drop”更值得继续。

#### Round B：只在 Round A 给出正信号时补新 probe

补跑条件建议是：

- 至少有一个 Round A candidate 没有复现 `cplus1` 式前移签名；
- 且在 primary shape 指标上相对 `control` 有清晰改善。

若满足，再补：

- `d_cp_hold`
- `d_entry_bridge`

这让新增 config 的责任非常清楚：

- `d_cp_hold` 用来回答 aux release；
- `d_entry_bridge` 用来回答 boundary 当拍。

### 10.7 Handoff Promote / Reject 规则

下一轮不应再被单点 broad scalar 或 secondary old anchor 带偏。推荐把 handoff 结论固定成三层：

#### A. Primary normalized shape gate

以下 4 项作为 primary boundary-shape gate：

- `calf_r@SIC2-4 / leg`：越低越好
- `ratio12_24/57_70`：越低越好
- `ratio20_24+49_52/57_70`：越低越好
- `leg SIC57-70 / leg`：越高越好

`foot_l/ball_l@SIC12-15 / leg` 继续作为 secondary hotspot tie-break，而不是单独主导 promote。

#### B. Broad safety check

以下两项只承担“不允许明显塌”的职责：

- `leg broad mean`
- `all_ex_root`

也就是说：

- broad scalar 可以辅助确认 candidate 没塌；
- 但它不能单独覆盖 normalized shape 的恶化。

#### C. 决策语义

- `promote to exact Stage6`：
  - 相对 `control` 改善至少 `3/4` 个 primary normalized shape 指标；
  - 且 broad safety check 没有出现清晰恶化。
- `reject`：
  - 复现 `cplus1` 签名，即至少同时出现：
    - `ratio12_24/57_70` 上升
    - `ratio20_24+49_52/57_70` 上升
    - `leg SIC57-70 / leg` 下降
- `near-tie / hold`：
  - primary shape 改善不足以 cleanly beat `control`；
  - 但也没有明显 broad collapse。

只有 `promote` 候选才值得补 exact `Stage6-only`。其余 candidate 不应越级进入 full downstream。

### 10.8 建议的 exact Stage6 使用方式

下一轮 exact `Stage6-only` 的使用规则也应收紧：

- 最多只让 `control` 与一个 boundary winner 进入 exact `Stage6-only`
- 不再把 `old bestfree / old exit` 当 primary promote 依据
- 仍然使用当前 canonical contract：
  - `teacher = validate/teacher_batches/Walk_F_teacher.json`
  - `contacts_meas_source = pretrain_contact`
  - `phase_reset_source = none`
  - `time_index_mode = cycle`
  - `event_clock = auto`
  - `mask = cycle>=1 && drop_wrap=true`

换句话说，exact `Stage6-only` 在这轮只承担：

- `control` vs boundary winner 的 final arbitration

而不再承担：

- 帮一个 handoff 上已经没有 shape 优势的 candidate 翻案。

### 10.9 产物与命名建议

为了避免和 `2026-03-30` family 混淆，建议单独开一套更收敛的命名：

- out dir:
  - `models/cp015_control_boundary_probe_20260331`
- debug dir:
  - `debug_output/_tmp_cp015_control_boundary_probe_20260331`
- family label:
  - `control_boundary_probe_20260331`

建议命名：

- existing:
  - `control_denseckpt`
  - `dplus1_orig`
  - `d_lr_hold`
- new:
  - `d_cp_hold`
  - `d_entry_bridge`

其中 `d_entry_bridge` 的 config 语义建议明确写进 `strategy_meta.notes`：

> keep boundary at epoch12; epoch12 uses bridge settings copied from phase_c (`opt_lr=2e-4`, `w_contact_plan=0.13`), epoch13-15 revert to canonical phase_d.

### 10.10 一句话设计结论

下一轮最合理的 basetrain 工作，不是再问“`phase_c` 要不要继续加长”，而是：

> 在 `control_denseckpt` 的固定 boundary 上，精确分辨 `phase_d` 的 residence、LR drop、contact-plan release 与 `epoch12` boundary shock 哪一个才是真正的 local cause。

只有把这个 `control-neighborhood boundary probe` 做完，才有资格决定这条 upstream basetrain 线是否还值得继续优化。
