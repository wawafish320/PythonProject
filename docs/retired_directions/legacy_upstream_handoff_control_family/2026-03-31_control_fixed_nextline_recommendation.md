# `control-fixed` 下一条线建议

> Status: archived legacy upstream / handoff / control record
> Reader note: this file belongs to the old-boundary upstream-control investigation; any `current`, `default`, `canonical`, `recommend`, or `mainline` wording below is historical context, not present-tense repo policy.
> Current entry points: `docs/posttrain_pipeline.md`, `docs/train_design/2026-04-11_donor_contract_basin_universal_debug_synthesis.md`, `docs/retired_directions/legacy_upstream_handoff_control_family/README.md`

## 1. Premise

`control-neighborhood boundary probe` 的 Round A 已经给出清晰负信号：

- `dplus1_orig`: reject
- `d_lr_hold`: reject
- Round B: not triggered

因此下一步不应该再继续做 basetrain schedule 微调，而应该把 `control_denseckpt` 固定成 upstream baseline，转入 `control-fixed` 的新问题。

## 2. Fixed Baseline

后续所有新工作都以以下 baseline 为唯一起点：

- basetrain config:
  - `config/exp_phase_DirectBranch_v1_d1_cp015_tailk3_corridor_hold_tail15_phasea050_fixedsched_ep014center_control_denseckpt_seed2024_20260330.json`
- basetrain ckpt:
  - `models/cp015_phasecd_min_ablation_20260330/exp_phase_DirectBranch_v1_d1_cp015_tailk3_corridor_hold_tail15_phasea050_fixedsched_ep014center_control_denseckpt_seed2024_20260330/ckpt_epoch_015.pth`

不再改动：

- `phase_c/phase_d` boundary
- `phase_d` duration / LR / contact-plan release
- 任意 `cplus1/cplus2 keepd`
- 任意 `tail relax`

## 3. Recommended Next Question

下一条线建议改成：

> 在 basetrain schedule 固定为 `control_denseckpt` 的前提下，问题是否主要出在 posttrain / Stage6 entry contract / contact-plan measurement 这一侧，而不是 upstream schedule 本身。

这条线的目标不是再找新的 basetrain winner，而是回答：

- `control` 的 basetrain handoff 为什么已经是 local optimum，但更 downstream 的表现仍然不够理想
- 问题更像是 Stage6 entry contract、contact-plan 支撑、phase-shift 对齐，还是其他 measurement / loss 侧因素

## 4. Minimal Scope

建议把下一条线收敛成 `control-fixed diagnostics`，不要再开 schedule family。

优先级顺序：

1. `control` basetrain handoff vs `control_denseckpt_final` Stage6 的 same-contract 对照
2. contact-plan / phase-shift 的 whitebox 对照
3. 只在有明确证据时，再开新的 Stage6-only 单因子 probe

不建议直接做：

- 新一轮 basetrain schedule matrix
- `Stage7/downstream`
- 与 `old bestfree / old exit` 的大范围翻案式对比

## 5. Working Rule

这条新线建议固定以下 rule：

- `control_denseckpt` 是唯一 upstream baseline
- 所有新 probe 都必须写清楚“不是 schedule 问题，而是在 fixed control baseline 上检查 downstream-side cause”
- exact `Stage6-only` 只作为 contract / whitebox arbitration 工具，而不是 schedule 翻案工具

## 6. Concrete Recommendation

如果要立刻开下一个任务，我建议这样写：

> 以 `control_denseckpt` 为固定 basetrain baseline，不再改 schedule。请只围绕 `control` 的 posttrain / Stage6 entry contract / contact-plan whitebox 做最小诊断设计，回答 downstream 偏移到底是不是 contract-side 问题。

这比继续补 `d_cp_hold` 或 `d_entry_bridge` 更有价值，也更符合当前证据。
