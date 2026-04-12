# `control-fixed diagnostics` 执行计划

> Status: archived legacy upstream / handoff / control record
> Reader note: this file belongs to the old-boundary upstream-control investigation; any `current`, `default`, `canonical`, `recommend`, or `mainline` wording below is historical context, not present-tense repo policy.
> Current entry points: `docs/posttrain_pipeline.md`, `docs/train_design/2026-04-11_donor_contract_basin_universal_debug_synthesis.md`, `docs/retired_directions/legacy_upstream_handoff_control_family/README.md`

## 1. Scope

- family label: `control_fixed_diagnostics_20260331`
- fixed upstream baseline: `control_denseckpt`
- primary question:
  - 在 `control_denseckpt` 的 basetrain schedule 固定不变时，downstream 偏移更像是 `Stage6 entry contract / contact-plan measurement / phase-shift` 侧问题，还是已经没有足够清晰的单因子抓手
- explicit non-goals:
  - 不再改 `phase_c/phase_d` schedule
  - 不再开新的 basetrain family
  - 不进入 `Stage7/downstream`
  - 不把 `old bestfree / old exit` 当成 primary promote 依据

## 2. Fixed Inputs

- basetrain config:
  - `config/exp_phase_DirectBranch_v1_d1_cp015_tailk3_corridor_hold_tail15_phasea050_fixedsched_ep014center_control_denseckpt_seed2024_20260330.json`
- basetrain ckpt:
  - `models/cp015_phasecd_min_ablation_20260330/exp_phase_DirectBranch_v1_d1_cp015_tailk3_corridor_hold_tail15_phasea050_fixedsched_ep014center_control_denseckpt_seed2024_20260330/ckpt_epoch_015.pth`
- canonical Stage6 config:
  - `config/posttrain_WalkF_stage6_direct_cond_anchor_splitfirst_3way_armchain_pe32h512_20260227.json`
- wrapper:
  - `debug_output/_tmp_phasecd_min_ablation_20260330/cpu_nomps_exec.py`
- teacher:
  - `validate/teacher_batches/Walk_F_teacher.json`
- encoder bundle:
  - `models/motion_encoder_equiv.pt.best.pt`
- affine stats:
  - `debug_output/_tmp_phaseb_affine_20260304/affine_fit_mix08/affine_stats.json`

## 3. Existing Evidence To Reuse

source-of-truth:

- `docs/retired_directions/legacy_upstream_handoff_control_family/2026-03-31_control_boundary_probe_rounda_closeout.md`
- `docs/retired_directions/legacy_upstream_handoff_control_family/2026-03-31_control_fixed_nextline_recommendation.md`
- `docs/retired_directions/legacy_upstream_handoff_control_family/2026-03-19_stage6_contactplan_direct_audit.md`
- `docs/retired_directions/legacy_upstream_handoff_control_family/2026-03-26_basetrain_to_stage6_minimal_handoff_probe_and_export_contract.md`

existing artifacts to reuse before任何新运行:

- control handoff reference:
  - `debug_output/_tmp_cp015_control_boundary_probe_20260331/control/epoch015/handoff_eval/Walk_F_freerun_cycles.json`
- control exact Stage6 reference:
  - `debug_output/_tmp_phasecd_stage6_trend_top3_fullrerun_20260330/control_denseckpt_final/stage6_freerun/Walk_F_freerun_cycles.json`
  - `debug_output/_tmp_phasecd_stage6_trend_top3_fullrerun_20260330/control_denseckpt_final/stage6_group_summary.json`
- control same-contract entry/exit summary:
  - `debug_output/_tmp_phasecd_stage6_trend_top3_fullrerun_20260330/stage6_trend_top3_fullrerun_summary_20260330.md`
  - 只读取 `control_denseckpt_final` 行，不借机重开 `dplus1/cplus2` 讨论
- control entry-contract matrix:
  - `debug_output/_tmp_stage6_entry_contract_matrix_20260330/stage6_entry_contract_matrix_report.md`
  - 已知 `control_denseckpt_rerun` 最接近 old exit 的是 `epoch011`，但仍然 off-basin，因此这条线不再回到 basetrain boundary 微调
- secondary anchor:
  - `debug_output/_tmp_stage6_basetrain_compare_20260313/old_bestfree/stage6_freerun/Walk_F_freerun_cycles.json`
- plan-stack whitebox precedent:
  - `debug_output/_tmp_stage6_plantransplant_20260314/summary.md`
  - 已有信号支持 `plan-stack / contact-plan` 侧值得优先排查

## 4. Naming And Directory Convention

建议本轮新增产物统一落在：

- debug root: `debug_output/_tmp_control_fixed_diagnostics_20260331`
- model root: `models/__tmp_control_fixed_diagnostics_20260331`
- family label: `control_fixed_diagnostics_20260331`

目录建议：

- same-contract refresh:
  - `debug_output/_tmp_control_fixed_diagnostics_20260331/same_contract/`
- contact-plan whitebox:
  - `debug_output/_tmp_control_fixed_diagnostics_20260331/contact_plan/`
- window tables:
  - `debug_output/_tmp_control_fixed_diagnostics_20260331/window_tables/`
- only-if-needed Stage6-only arbitration:
  - `debug_output/_tmp_control_fixed_diagnostics_20260331/stage6_only/`

## 5. Recommended Execution Order

### Step 0. Read-only reuse first

先复核现成证据，不重跑训练：

1. `control` handoff vs `control_denseckpt_final` exact Stage6 的 same-contract 对照
2. `control_denseckpt_rerun` entry-contract matrix 的 `epoch010-015` 结论
3. `planstack transplant` 的既有 whitebox 结论

目标不是再找新的 basetrain winner，而是先确认：

- `control` 的 downstream 偏移是否主要发生在 `Stage6 exact` 之后
- 这个偏移是否已经有明显的 `plan-stack / contact-plan / phase-shift` 指向

### Step 1. 仅补轻量 eval refresh

只对 fixed control exact lane 和 old exit secondary anchor 补 `logged freerun eval`，不重跑 basetrain，也默认不重跑 `train.posttrain`。

refresh lanes:

- `control_denseckpt_final` existing Stage6 ckpt
- `old_bestfree` existing Stage6 ckpt

refresh 目的：

- 给 exact Stage6 eval JSON 补齐 `ContactGTPerC / ContactPlanPerC / ContactMeasPerC / ContactErrPerC`
- 给 whitebox 留下同口径 `phase-shift` / window table 输入

### Step 2. Contact-plan / phase-shift whitebox

对以下三份 JSON 做同口径检查：

- control handoff
- refreshed control exact Stage6 eval
- refreshed old exit exact Stage6 eval

whitebox 工具：

- `tools/analyze_freerun_contact_plan.py`
- `tools/tabulate_freerun_window.py`

固定窗口：

- `2-4`
- `12-24`
- `49-52`
- `57-70`
- `83-0`

解释原则：

- `2-4`：看 `calf_r` 早段 pocket 是否在 exact Stage6 被重新放大
- `12-24` / `49-52`：看 forward-shift pocket 是否在 exact Stage6 被继续推高
- `57-70`：看 late support basin 是否在 exact Stage6 被继续掏空
- `83-0`：看 seam / wrap 区域是否带来额外 contact-phase mismatch

### Step 3. Only-if-needed Stage6-only arbitration

只有同时满足以下条件，才允许开一个新的 `Stage6-only` 单因子 probe：

1. same-contract 对照显示：`control` handoff 并不差，但 exact Stage6 明显把 normalized shape 推偏
2. whitebox 指向单一且稳定的 culprit：
   - `contact-plan amplitude / collapse`
   - `contact meas calibration`
   - `phase-shift / seam mismatch`
3. 这个 culprit 不能仅靠现有 artifact reuse 得到结论

约束：

- 一次只允许一个 Stage6-only lane
- probe 必须是 single-factor
- 不进入 `Stage7/downstream`
- probe spec 需要另开文档，不在本文件里继续发散

## 6. Decision Rule

### `contract-side likely`

满足大部分以下信号：

- control handoff 的 normalized shape 明显干净于 control exact Stage6
- control exact Stage6 的 logged eval 出现更明显的 `plan/meas` shift、幅度塌缩或 seam mismatch
- old exit logged eval 的对应症状更轻，或 pattern 明显不同

动作：

- 允许写一个单因子 `Stage6-only` arbitration spec

### `measurement-side likely`

满足大部分以下信号：

- `ContactPlanPerC` 仍然有可解释 amplitude / 左右切换
- `ContactMeasPerC` 或 `ContactErrPerC` 的 shift / calibration 才是主要异常
- handoff 没有同等级异常，但 exact Stage6 后异常放大

动作：

- 优先把后续 probe 限定在 measurement / contract 侧
- 不回去改 basetrain schedule

### `no clean single-factor culprit`

满足任一情况：

- same-contract delta 不稳定
- old exit 只能提供弱 secondary anchor
- whitebox 信号互相冲突，无法收敛到一个 culprit

动作：

- 结论写成 `control-fixed downstream issue not isolated`
- 不开新的 Stage6-only probe
- 保持 `control boundary local optimum / 暂不继续微调`

## 7. Command Sketch

### A. same-contract existing references

只刷新现有 report，不重开新 family：

```bash
python3 tools/phasecd_stage6_trend_top3_fullrerun_report.py
```

只把 `control_denseckpt_final` 作为 fixed-control 证据读取；其他行不用于 reopening schedule。

### B. refresh logged exact Stage6 eval for fixed control

```bash
debug_output/_tmp_phasecd_min_ablation_20260330/cpu_nomps_exec.py -m train.validate.run_freerun_cycles \
  --teacher validate/teacher_batches/Walk_F_teacher.json \
  --model models/__tmp_phasecd_stage6_trend_top3_fullrerun_20260330/control_denseckpt_final/ckpt_last_control_denseckpt_final_stage6_trend_fullrerun_20260330.pth \
  --rounds 5 \
  --depth 3 \
  --time-index-mode cycle \
  --event_clock auto \
  --phase_reset_source none \
  --contacts_meas_source pretrain_contact \
  --contacts_meas_pretrain_clamp 1.0 \
  --contacts_meas_pretrain_affine_stats debug_output/_tmp_phaseb_affine_20260304/affine_fit_mix08/affine_stats.json \
  --encoder-bundle models/motion_encoder_equiv.pt.best.pt \
  --log_contacts \
  --analyze_phase_shift \
  --export_joint_direct_geolocal_series \
  --out debug_output/_tmp_control_fixed_diagnostics_20260331/same_contract/control_denseckpt_final_logged_eval \
  --force
```

```bash
debug_output/_tmp_phasecd_min_ablation_20260330/cpu_nomps_exec.py \
  tools/phasea_group_summary.py \
  debug_output/_tmp_control_fixed_diagnostics_20260331/same_contract/control_denseckpt_final_logged_eval/Walk_F_freerun_cycles.json \
  --cycle_gte 1 \
  --drop_wrap \
  --out debug_output/_tmp_control_fixed_diagnostics_20260331/same_contract/control_denseckpt_final_logged_group_summary.json
```

### C. refresh logged exact Stage6 eval for old exit secondary anchor

```bash
debug_output/_tmp_phasecd_min_ablation_20260330/cpu_nomps_exec.py -m train.validate.run_freerun_cycles \
  --teacher validate/teacher_batches/Walk_F_teacher.json \
  --model models/__tmp_stage6_basetrain_compare_20260313/old_bestfree/ckpt_last_old_bestfree_stage6_cmp_20260313.pth \
  --rounds 5 \
  --depth 3 \
  --time-index-mode cycle \
  --event_clock auto \
  --phase_reset_source none \
  --contacts_meas_source pretrain_contact \
  --contacts_meas_pretrain_clamp 1.0 \
  --contacts_meas_pretrain_affine_stats debug_output/_tmp_phaseb_affine_20260304/affine_fit_mix08/affine_stats.json \
  --encoder-bundle models/motion_encoder_equiv.pt.best.pt \
  --log_contacts \
  --analyze_phase_shift \
  --export_joint_direct_geolocal_series \
  --out debug_output/_tmp_control_fixed_diagnostics_20260331/same_contract/old_bestfree_logged_eval \
  --force
```

```bash
debug_output/_tmp_phasecd_min_ablation_20260330/cpu_nomps_exec.py \
  tools/phasea_group_summary.py \
  debug_output/_tmp_control_fixed_diagnostics_20260331/same_contract/old_bestfree_logged_eval/Walk_F_freerun_cycles.json \
  --cycle_gte 1 \
  --drop_wrap \
  --out debug_output/_tmp_control_fixed_diagnostics_20260331/same_contract/old_bestfree_logged_group_summary.json
```

### D. contact-plan whitebox

control handoff:

```bash
python3 tools/analyze_freerun_contact_plan.py \
  --json debug_output/_tmp_cp015_control_boundary_probe_20260331/control/epoch015/handoff_eval/Walk_F_freerun_cycles.json \
  --pred-source plan \
  --exclude-round0 \
  --cycle-len 87
```

```bash
python3 tools/analyze_freerun_contact_plan.py \
  --json debug_output/_tmp_cp015_control_boundary_probe_20260331/control/epoch015/handoff_eval/Walk_F_freerun_cycles.json \
  --pred-source meas \
  --exclude-round0 \
  --cycle-len 87
```

control exact Stage6:

```bash
python3 tools/analyze_freerun_contact_plan.py \
  --json debug_output/_tmp_control_fixed_diagnostics_20260331/same_contract/control_denseckpt_final_logged_eval/Walk_F_freerun_cycles.json \
  --pred-source plan \
  --exclude-round0 \
  --cycle-len 87
```

```bash
python3 tools/analyze_freerun_contact_plan.py \
  --json debug_output/_tmp_control_fixed_diagnostics_20260331/same_contract/control_denseckpt_final_logged_eval/Walk_F_freerun_cycles.json \
  --pred-source meas \
  --exclude-round0 \
  --cycle-len 87
```

old exit secondary anchor:

```bash
python3 tools/analyze_freerun_contact_plan.py \
  --json debug_output/_tmp_control_fixed_diagnostics_20260331/same_contract/old_bestfree_logged_eval/Walk_F_freerun_cycles.json \
  --pred-source plan \
  --exclude-round0 \
  --cycle-len 87
```

```bash
python3 tools/analyze_freerun_contact_plan.py \
  --json debug_output/_tmp_control_fixed_diagnostics_20260331/same_contract/old_bestfree_logged_eval/Walk_F_freerun_cycles.json \
  --pred-source meas \
  --exclude-round0 \
  --cycle-len 87
```

### E. window-table whitebox

对 `control handoff / control exact Stage6 / old exit exact Stage6` 三份 JSON，各跑同一组窗口：

```bash
for window in 2-4 12-24 49-52 57-70 83-0; do
  python3 tools/tabulate_freerun_window.py \
    --json debug_output/_tmp_control_fixed_diagnostics_20260331/same_contract/control_denseckpt_final_logged_eval/Walk_F_freerun_cycles.json \
    --cycles 1-4 \
    --window "$window" \
    --out "debug_output/_tmp_control_fixed_diagnostics_20260331/window_tables/control_denseckpt_final_${window//[^0-9]/_}.json"
done
```

同样方式分别替换成：

- `debug_output/_tmp_cp015_control_boundary_probe_20260331/control/epoch015/handoff_eval/Walk_F_freerun_cycles.json`
- `debug_output/_tmp_control_fixed_diagnostics_20260331/same_contract/old_bestfree_logged_eval/Walk_F_freerun_cycles.json`

## 8. Final Handoff Rule

本轮执行完成后，只允许输出以下三种结论之一：

1. `contract-side likely -> worth exactly one Stage6-only arbitration lane`
2. `measurement-side likely -> keep schedule fixed, write measurement-side single-factor spec`
3. `control-fixed downstream issue not isolated -> no new probe`

无论哪一种，都不允许直接跳进 `Stage7/downstream`。
