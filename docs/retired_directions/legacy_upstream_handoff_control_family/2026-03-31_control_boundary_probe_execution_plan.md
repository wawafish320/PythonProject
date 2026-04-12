# `control-neighborhood boundary probe` 执行计划

> Status: archived legacy upstream / handoff / control record
> Reader note: this file belongs to the old-boundary upstream-control investigation; any `current`, `default`, `canonical`, `recommend`, or `mainline` wording below is historical context, not present-tense repo policy.
> Current entry points: `docs/posttrain_pipeline.md`, `docs/train_design/2026-04-11_donor_contract_basin_universal_debug_synthesis.md`, `docs/retired_directions/legacy_upstream_handoff_control_family/README.md`

## 1. Scope

- family label: `control_boundary_probe_20260331`
- primary baseline: `control_denseckpt`
- scope: `Stage6-only` 之前的 basetrain + handoff
- explicit non-goals:
  - 不沿 `cplus1/cplus2 keepd` 继续外推
  - 不把 `tail relax` 重新带回主线
  - 不直接进入 `Stage7/downstream`

## 2. Boundary-Local Contract

`control_denseckpt` 的固定 boundary 是：

- `phase_c=[10,11]`
- `phase_d=[12,15]`

在这个 fixed boundary 上，`phase_c -> phase_d` 真正发生跳变的局部字段只有：

- `opt_lr: 0.0002 -> 0.0001`
- `w_contact_plan: 0.13 -> 0.10`

保持不变的字段：

- `tf_max/tf_min = 0.5/0.5`
- `ss_chunk_len = 1`
- `w_rot_local = 0.22`
- `rot_local_tail_weight = 0.2`

因此本轮 probe 只允许围绕以下 4 个 boundary-local 解释展开：

- `phase_d` residence
- `phase_d` LR drop
- `phase_d` contact-plan release
- `epoch12` one-step bridge

## 3. Probe Matrix

| candidate | status | config | test axis | expected answer |
|---|---|---|---|---|
| `control_denseckpt` | existing baseline | `config/exp_phase_DirectBranch_v1_d1_cp015_tailk3_corridor_hold_tail15_phasea050_fixedsched_ep014center_control_denseckpt_seed2024_20260330.json` | reference | 锁定 primary baseline |
| `dplus1_orig` | existing | `config/exp_phase_DirectBranch_v1_d1_cp015_tailk3_corridor_hold_tail15_phasea050_fixedsched_ep014center_dplus1_orig_seed2024_20260330.json` | residence length | `phase_d` residence 是否偏短 |
| `d_lr_hold` | existing | `config/exp_phase_DirectBranch_v1_d1_cp015_tailk3_corridor_hold_tail15_phasea050_fixedsched_ep014center_d_lr_hold_seed2024_20260330.json` | LR drop hardness | `phase_d` 入场时 LR drop 是否过硬 |
| `d_cp_hold` | new | `config/exp_phase_DirectBranch_v1_d1_cp015_tailk3_corridor_hold_tail15_phasea050_fixedsched_ep014center_d_cp_hold_seed2024_20260331.json` | aux release hardness | `phase_d` 入场时 contact-plan release 是否过硬 |
| `d_entry_bridge` | new | `config/exp_phase_DirectBranch_v1_d1_cp015_tailk3_corridor_hold_tail15_phasea050_fixedsched_ep014center_d_entry_bridge_seed2024_20260331.json` | entry-local bridge | 问题是否集中在 `epoch12` boundary shock |

## 4. Naming And Directory Convention

推荐本轮统一使用：

- out dir: `models/cp015_control_boundary_probe_20260331`
- debug dir: `debug_output/_tmp_cp015_control_boundary_probe_20260331`
- family label: `control_boundary_probe_20260331`

说明：

- 两个 new config 已直接落到 `models/cp015_control_boundary_probe_20260331`
- 两个 existing config 保持原语义不改，仍指向它们的 `20260330` out dir
- 如果后续需要把 existing 也重跑到同一 out family，应新建 rerun wrapper config，而不是原地改写 existing config

建议 handoff/debug 目录约定：

- Round A:
  - `debug_output/_tmp_cp015_control_boundary_probe_20260331/round_a/<candidate>/`
- Round B:
  - `debug_output/_tmp_cp015_control_boundary_probe_20260331/round_b/<candidate>/`
- exact `Stage6-only`:
  - `debug_output/_tmp_cp015_control_boundary_probe_20260331/stage6_only/<winner>/`

## 5. Recommended Run Order

### Round A

先跑 existing single-factor probe：

- `dplus1_orig`
- `d_lr_hold`

理由：

- 不改 fixed boundary 起点 `12`
- 已有 config，可直接复现
- 能优先区分 `residence` 与 `LR drop` 两个解释

### Round B

只有在 Round A 满足以下条件时才补 new probe：

- 至少有一个 Round A candidate 没有复现 `cplus1` 式前移签名
- 且相对 `control` 在 primary normalized shape gate 上有清晰改善

满足后再补：

- `d_cp_hold`
- `d_entry_bridge`

## 6. Handoff Decision Rules

### Primary normalized shape gate

主判断只看以下 4 项：

- `calf_r@SIC2-4 / leg`：越低越好
- `ratio12_24/57_70`：越低越好
- `ratio20_24+49_52/57_70`：越低越好
- `leg SIC57-70 / leg`：越高越好

`foot_l/ball_l@SIC12-15 / leg` 只作为 secondary hotspot tie-break。

### Broad safety check

只承担“不允许明显塌”的职责：

- `leg broad mean`
- `all_ex_root`

这些 broad scalar 不允许单独覆盖 normalized shape 的恶化。

### Promote / Reject / Near-Tie

- `promote`:
  - 相对 `control` 改善至少 `3/4` 个 primary normalized shape 指标
  - 且 broad safety check 没有清晰恶化
- `reject`:
  - 复现 `cplus1` 签名，即至少同时出现：
    - `ratio12_24/57_70` 上升
    - `ratio20_24+49_52/57_70` 上升
    - `leg SIC57-70 / leg` 下降
- `near-tie`:
  - 没有 cleanly beat `control`
  - 但也没有 broad collapse

## 7. Exact `Stage6-only` Handoff Rule

- 只允许 `control_denseckpt` 对一个 boundary winner
- 不做更深 downstream
- old exit / old bestfree 只作 secondary anchor，不作 primary promote 依据

如果 Round A 和 Round B 都没有 clean `promote` winner，则这条 basetrain 主线停在 fixed-boundary local optimum verdict，不再继续微调。

## 8. Handoff Checklist

执行 handoff 前后，逐项确认：

- boundary 起点仍然是 `phase_c=[10,11] -> phase_d=[12,*]`
- `w_rot_local = 0.22`
- `rot_local_tail_weight = 0.2`
- `tf_max/tf_min = 0.5/0.5`
- `ss_chunk_len = 1`
- 没有把 `tail relax` 混入任何本轮 candidate
- 没有把 `cplus1/cplus2 keepd` 重新带回主判断
- 只有 `promote` winner 才允许进入 exact `Stage6-only`

## 9. Command Sketch

以下命令模式统一通过：

- wrapper: `debug_output/_tmp_phasecd_min_ablation_20260330/cpu_nomps_exec.py`
- teacher: `validate/teacher_batches/Walk_F_teacher.json`
- summary tool: `debug_output/_tmp_phasecd_min_ablation_20260330/run_exact_handoff_epoch_sweep.py`

Round A basetrain:

```bash
debug_output/_tmp_phasecd_min_ablation_20260330/cpu_nomps_exec.py -m train.training_MPL \
  --config_json config/exp_phase_DirectBranch_v1_d1_cp015_tailk3_corridor_hold_tail15_phasea050_fixedsched_ep014center_dplus1_orig_seed2024_20260330.json

debug_output/_tmp_phasecd_min_ablation_20260330/cpu_nomps_exec.py -m train.training_MPL \
  --config_json config/exp_phase_DirectBranch_v1_d1_cp015_tailk3_corridor_hold_tail15_phasea050_fixedsched_ep014center_d_lr_hold_seed2024_20260330.json
```

Round A handoff summary:

```bash
debug_output/_tmp_phasecd_min_ablation_20260330/cpu_nomps_exec.py \
  debug_output/_tmp_phasecd_min_ablation_20260330/run_exact_handoff_epoch_sweep.py \
  --exp-dir models/cp015_phasecd_min_ablation_20260330/exp_phase_DirectBranch_v1_d1_cp015_tailk3_corridor_hold_tail15_phasea050_fixedsched_ep014center_dplus1_orig_seed2024_20260330 \
  --out-root debug_output/_tmp_cp015_control_boundary_probe_20260331/round_a/dplus1_orig \
  --epoch-start 10 \
  --epoch-end 16

debug_output/_tmp_phasecd_min_ablation_20260330/cpu_nomps_exec.py \
  debug_output/_tmp_phasecd_min_ablation_20260330/run_exact_handoff_epoch_sweep.py \
  --exp-dir models/cp015_phasecd_min_ablation_20260330/exp_phase_DirectBranch_v1_d1_cp015_tailk3_corridor_hold_tail15_phasea050_fixedsched_ep014center_d_lr_hold_seed2024_20260330 \
  --out-root debug_output/_tmp_cp015_control_boundary_probe_20260331/round_a/d_lr_hold \
  --epoch-start 10 \
  --epoch-end 15
```

Round B basetrain:

```bash
debug_output/_tmp_phasecd_min_ablation_20260330/cpu_nomps_exec.py -m train.training_MPL \
  --config_json config/exp_phase_DirectBranch_v1_d1_cp015_tailk3_corridor_hold_tail15_phasea050_fixedsched_ep014center_d_cp_hold_seed2024_20260331.json

debug_output/_tmp_phasecd_min_ablation_20260330/cpu_nomps_exec.py -m train.training_MPL \
  --config_json config/exp_phase_DirectBranch_v1_d1_cp015_tailk3_corridor_hold_tail15_phasea050_fixedsched_ep014center_d_entry_bridge_seed2024_20260331.json
```

Round B handoff summary:

```bash
debug_output/_tmp_phasecd_min_ablation_20260330/cpu_nomps_exec.py \
  debug_output/_tmp_phasecd_min_ablation_20260330/run_exact_handoff_epoch_sweep.py \
  --exp-dir models/cp015_control_boundary_probe_20260331/exp_phase_DirectBranch_v1_d1_cp015_tailk3_corridor_hold_tail15_phasea050_fixedsched_ep014center_d_cp_hold_seed2024_20260331 \
  --out-root debug_output/_tmp_cp015_control_boundary_probe_20260331/round_b/d_cp_hold \
  --epoch-start 10 \
  --epoch-end 15

debug_output/_tmp_phasecd_min_ablation_20260330/cpu_nomps_exec.py \
  debug_output/_tmp_phasecd_min_ablation_20260330/run_exact_handoff_epoch_sweep.py \
  --exp-dir models/cp015_control_boundary_probe_20260331/exp_phase_DirectBranch_v1_d1_cp015_tailk3_corridor_hold_tail15_phasea050_fixedsched_ep014center_d_entry_bridge_seed2024_20260331 \
  --out-root debug_output/_tmp_cp015_control_boundary_probe_20260331/round_b/d_entry_bridge \
  --epoch-start 10 \
  --epoch-end 15
```
