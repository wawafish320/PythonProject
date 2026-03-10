# 2026-03-10 标准 axis-angle / rotvec 全仓迁移复核

## 1) 结论

基于 2026-03-10 当前 debug worktree 的代码审计与本地验证，仓内这次 `SO(3)` 语义迁移已经形成统一口径：

- `train/geometry.py` 里的 `so3_log_map(R)` / `_matrix_log_map(R)` 当前都返回标准 `axis * angle`。
- `angvel_vec_from_delta_R` / `angvel_vec_from_R_seq` 已随之统一为标准 rotvec 角速度语义。
- `lambda blend`、freerun/export、诊断脚本、帮助文本、模板/encoder 资产都已经按新语义对齐。
- 在 `train/`、`tools/`、`docs/` 范围内，未再搜到残留的 `so3_log_map(...)*2` 或 “half-angle 补偿” 逻辑。
- `models/pretrain_template.json` 已恢复为 canonical 模板路径，并与 `temp_pretrain_template.json` 保持同步。

本次复核实际跑通了：

```bash
python tools/check_standard_rotvec_semantics.py
python tools/check_lambda_fusion_blend_geometry.py
```

两条命令都通过。

## 2) 问题原因

这次问题的根因不是 `lambda blend` 本身，而是历史 `SO(3)` log-map 语义不一致：

- 旧实现把 `vee(0.5 * (R - R^T))` 当成了最终 rotvec 口径的一部分，导致 `so3_log_map` 实际返回的是 legacy half-angle 语义。
- 直接后果是 `so3_log_map(so3_exp_map(omega))` 的范数只有 `0.5 * ||omega||`，于是 `lambda blend` 主路径实际只走了 `0.5 * lambda` 的 residual rotation。
- 同一个语义偏差还会继续污染 angvel、rotvec 导出、omega 导出、模板尺度解释，以及依赖这些输入的 frozen pretrain 资产。

因此这次修复的正确方向不是在单个调用点外层补 `*2`，而是把 log-map 本体语义改正，再把所有下游调用点、模板、bundle、文档一起迁到统一标准。

## 3) 已确认的修改位置

### 3.1 几何核心

- `train/geometry.py`
  - `_matrix_log_map`
  - `so3_log_map`
  - `angvel_vec_from_delta_R`
  - `angvel_vec_from_R_seq`

当前实现满足标准 round-trip 口径：`so3_log_map(so3_exp_map([0,0,1]))` 范数应为 `1.0`，不再是 `0.5`。

### 3.2 语义标记与 fail-fast

- `train/rotvec_semantics.py`
  - 定义 `standard_axis_angle_v1`
  - 定义 `standard_axis_angle_times_fps_v1`
  - 提供 `stamp_standard_rotvec_spec`
  - 提供 `require_standard_rotvec_spec`
  - 提供 `require_standard_rotvec_bundle`

### 3.3 runtime / loader / normalizer / eval 路径

- `train/layout.py`
- `train/dataset.py`
- `train/normalizers.py`
- `train/models.py`
- `train/training_MPL.py`
- `train/posttrain.py`
- `train/validate/run_freerun_cycles.py`
- `train/validate/run_teacher_rollout.py`

这些路径当前都显式要求标准 rotvec 语义模板或 bundle；repo 外旧资产若未迁移到标准语义，应视为 legacy incompatible。

### 3.4 未来生成资产的写出路径

- `train/convert_json_to_npz.py`
- `train/pretrain_mpl_min.py`

这两处已经会给新生成模板/资产写入标准 rotvec / angvel 语义标记。

### 3.5 文档与说明文本

- `docs/rotvec_semantics_standardization.md`
- `docs/train_architecture_overview.md`
- `train/TRAINING_GUIDE.md`
- `train/validate/run_freerun_cycles.py` 内导出说明文本

当前文案已经统一为“直接输出标准 rotvec / omega”，不再保留“外层乘 `*2` 才是 full axis-angle”的说法。

### 3.6 已审计的调试脚本调用点

当前 worktree 里直接调用 `so3_log_map` 的调试脚本，已确认不再依赖外层 `*2` 补偿，至少包括：

- `tools/check_standard_rotvec_semantics.py`
- `tools/check_lambda_fusion_blend_geometry.py`
- `tools/test_hinge_delta_target.py`
- `tools/run_h1_10p2_audits.py`
- `tools/run_lrflip_strict_funnel.py`
- `tools/report_sic_hotspots_vs_gt_angvel.py`
- `tools/analyze_freerun_joint_so3_error.py`
- `tools/run_h1_10p3a_gate.py`
- `tools/diagnose_direct_error_footr_axis_gate.py`

## 4) 资产状态与兼容性口径

### 4.1 已迁移并已打标的 canonical 资产

- `raw_data/processed_data/norm_template.json`
- `temp_pretrain_template.json`
- `models/pretrain_template.json`
- `models/motion_encoder_equiv_stageA.pt`
- `models/motion_encoder_equiv.pt.best.pt`

本地检查已确认这些资产都带有：

- `rotvec_semantics = standard_axis_angle_v1`
- `angvel_semantics = standard_axis_angle_times_fps_v1`

### 4.2 angvel 归一化口径

这次迁移后，原始 angvel 的几何语义变成标准 rotvec，因此模板侧同步迁的是“原始量尺度”：

- `tanh_scales_angvel` 已按新 raw magnitude 调整
- `MuAngVel` / `StdAngVel` 保持 tanh 域统计口径

对应解释见：

- `docs/rotvec_semantics_standardization.md`
- `raw_data/processed_data/norm_template.json`
- `models/pretrain_template.json`
- `temp_pretrain_template.json`

### 4.3 与历史结果的可比性

迁移后，下列标量 geodesic 指标仍可继续横向参考：

- geodesic angle
- `DirectGeoLocalDeg`
- `BlendGeoLocalDeg`
- `GeoLocalDeg`

但旧历史导出里如果这些字段是按旧 half-angle 口径写出的，则不应再和新导出的向量量直接混比：

- `rotvec_deg_xyz`
- `omega_deg_xyz`
- keybone omega / keybone state 中的 rotvec 向量字段

换言之，旧导出报表里的“向量值”不可直接横比；标量角度指标可以继续作为参考。

## 5) 本次复核实际执行的检查

### 5.1 代码审计

已审计/检索：

- `so3_log_map(`
- `_matrix_log_map(`
- 与 `so3_log_map` 相关的 `*2` 补偿
- `half-angle`
- `theta/2`
- `full axis-angle`
- `so3_log_map(...)*2`

结果：

- 在 `train/`、`tools/`、`docs/` 范围内，没有再发现残留的 `so3_log_map(...)*2` 代码模式。
- `half-angle` 只在 `docs/rotvec_semantics_standardization.md` 中作为历史迁移背景被提及，不再是当前实现约定。

### 5.2 几何与 lambda 行为

`python tools/check_standard_rotvec_semantics.py` 覆盖并通过了：

- `so3_log_map(so3_exp_map([0,0,1])) ~= [0,0,1]`
- 多角度点 round-trip：`0.1 / 0.5 / 1.0 / pi/2`
- `exp(log(R))` geodesic 误差检查
- `angvel_vec_from_delta_R` / `angvel_vec_from_R_seq` 语义检查
- canonical 模板与 encoder bundle 语义标签检查

`python tools/check_lambda_fusion_blend_geometry.py` 覆盖并通过了：

- `lambda=0 -> blend == incremental`
- `lambda=1 -> blend == direct`
- `lambda=0.5 -> residual geodesic = 0.5 * full residual`

## 6) 当前可直接使用的验证命令

```bash
python tools/check_standard_rotvec_semantics.py
python tools/check_lambda_fusion_blend_geometry.py
```

freerun / teacher 如需走 canonical pretrain 模板，使用：

```bash
--pretrain-template models/pretrain_template.json
```

如需走 frozen encoder 路径，当前 canonical bundle 可直接使用：

```bash
--encoder-bundle models/motion_encoder_equiv_stageA.pt
--encoder-bundle models/motion_encoder_equiv.pt.best.pt
```

## 7) 备注

- 本文档记录的是 2026-03-10 当前 debug worktree 的复核结论。
- 当前工作区还有大量与本问题无关的未提交改动/未跟踪文件；本记录仅覆盖 rotvec 语义迁移相关路径。
