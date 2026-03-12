# 2026-03-05 Pretrain Contact Route 清理前就绪基线（Cleanup Readiness）

Last updated: 2026-03-05

## 1) 目的

本文件用于回答两个问题：

1. 当前是否已经证明 `pretrain_contact(+affine)` 是可行方向，可以替代 `basetrain` 中 contact 相关职责？
2. 在真正执行清理（移除 basetrain contact 相关代码）前，需要满足哪些量化准线（gate）？

结论先行：
- **方向可行**：接触残差指标显著优于 whitebox 基线，且 source 路由稳定命中。
- **暂不建议立即硬清理**：存在局部回退点（尤其是特定 SIC/骨骼），应先定位根因并收敛到 gate 以内。

---

## 2) 本轮执行范围（已完成）

1. 文档口径更新（该条为 2026-03-05 当时状态；当前主线已固定为 `pretrain_contact`）：
   - `docs/posttrain_pipeline.md`
2. 完整 8-stage posttrain 主链重跑（pretrain_contact + clamp1 + affine）：
   - Stage6 -> 70a -> 70b -> 70c -> 70R -> 71 -> 72 -> lambda_final
3. 最终 ckpt 5-clip freerun 评估：
   - `lambda_fusion_apply=off`
   - `lambda_fusion_apply=on`
4. 额外生成 `DirectGeoLocalDeg` 的 old/new 对照统计（含你需要的 overall/region/pointwise 口径）。

---

## 3) 关键产物路径

### 3.1 新链路训练产物

- 目录：`models/__tmp_phaseD_posttrain_pipeline_pretrain_20260305_fullchain`
- 最终模型：
  - `ckpt_last_WalkF_stage7_lambda_final_calib_20260227_fromarmchain_fullcompat_pcsrc20260305.pth`

### 3.2 聚合指标对比（5 clips）

- 汇总（off/on 双口径）：
  - `debug_output/_tmp_phaseD_posttrain_pipeline_pretrain_20260305_fullchain/summary_fullchain_pretrain_vs_baseline.md`
  - `debug_output/_tmp_phaseD_posttrain_pipeline_pretrain_20260305_fullchain/summary_fullchain_pretrain_vs_baseline.json`

### 3.3 DirectGeoLocalDeg 对照（Walk_F）

- rounds=5 对照目录：
  - `debug_output/_tmp_phaseD_direct_geolocal_compare_20260305_r5/old_whitebox_vs_new_fullchain_pretrain`
- 核心文件：
  - `global_signal_summary.txt`
  - `summary_metrics.txt`

---

## 4) 5-clip 聚合结果（round>=1）

### 4.1 `lambda_fusion_apply=off`

当前 fullchain pretrain vs baseline lambda_whitebox：

- `GeoLocalDegWeighted`: `34.433958` vs `34.363971` -> `+0.069986`
- `ContactErrAbsMean`: `0.123358` vs `0.289088` -> `-0.165730`
- `ContactMeasGtAbsMean`: `0.421413` vs `0.354103` -> `+0.067310`
- `source_match_rate`: `1.000000` vs `1.000000`

说明：
- contact 残差显著改善；
- pose 主指标在该口径有轻微回退（`+0.069986`）。

### 4.2 `lambda_fusion_apply=on`

当前 fullchain pretrain vs baseline lambda_whitebox：

- `GeoLocalDegWeighted`: `7.336871` vs `7.313585` -> `+0.023287`
- `ContactErrAbsMean`: `0.133698` vs `0.291946` -> `-0.158248`
- `ContactMeasGtAbsMean`: `0.422625` vs `0.445916` -> `-0.023291`
- `source_match_rate`: `1.000000` vs `1.000000`

说明：
- 在实际常用（`lambda_fusion_apply=on`）口径下，contact 与 pose 同时较稳。

---

## 5) DirectGeoLocalDeg（Walk_F, rounds=5, cycle>=1 drop_wrap）

来源：
- old: `phaseD_ab_lambda_whitebox`
- new: `fullchain_pretrain_contact_affine`

### 5.1 你要求的整体口径

```text
[overall]
mean_old=0.159976
mean_new=0.147802
mean_delta=-0.012174
bones_excl_root=45
bones_regress_by_mean=18
bones_improve_by_mean=27

[region_split]
leg8_mean_old=0.332883
leg8_mean_new=0.313692
leg8_mean_delta=-0.019191
non_leg_mean_old=0.122591
non_leg_mean_new=0.111934
non_leg_mean_delta=-0.010657

[pointwise_signal]
points=15480
improved_ratio=0.535465
worse_ratio=0.464535
median_delta=-0.002178
```

### 5.2 分位数（Global, all bones excl root）

- `p50_deg`: `0.084800 -> 0.082417` (`-0.002383`)
- `p90_deg`: `0.417071 -> 0.371342` (`-0.045729`)
- `p99_deg`: `0.950540 -> 0.924327` (`-0.026213`)
- `max_deg`: `2.654812 -> 1.712817` (`-0.941995`)

### 5.3 当前主要回退热点（用于 debug）

1. `SIC12-15 + {foot_l, ball_l}`：`mean 0.498373 -> 0.551165`（`+0.052792`）
2. top regressions by mean（示例）：
   - `lowerarm_l`: `+0.061063`
   - `thumb_01_r`: `+0.043992`
   - `pinky_01_r`: `+0.033368`
   - `ball_l`: `+0.016819`

---

## 6) 阶段性判断

1. **可以确认方向成立**：pretrain contact route 在核心 contact 残差上有稳定且显著收益。
2. **尚不建议立刻删除 basetrain contact 逻辑**：有局部回退点，尤其是 SIC12-15 的左脚区域与少量上肢骨骼。
3. **建议策略**：先针对热点做最小闭环 debug，达标后执行清理。

---

## 7) 清理前准线（Gate）

建议把以下作为执行 cleanup 的硬门槛：

1. 路由可追溯性：
   - `source_match_rate == 1.0`（`ContactsMeasSourceApplied` 命中预期分支）。
2. contact 主指标：
   - 相对 `lambda_whitebox`：`ContactErrAbsMean <= -0.15`（delta）。
3. pose 主指标：
   - `GeoLocalDegWeighted` 相对基线不劣化超过 `+0.05`（至少在生产采用口径上满足）。
4. DirectGeoLocalDeg 全局分布：
   - `mean_delta <= 0`
   - `p90/p99/max` 不劣化（`delta <= 0`）
5. 局部热点：
   - `SIC12-15 + {foot_l, ball_l}` 的 `mean_delta <= +0.02`
6. 回退骨骼规模控制：
   - `bones_regress_by_mean <= 20`

当前状态（2026-03-05，按第 10 节最佳链路 `rC -> stage71 -> s72a -> lam_from_s72a`，`lambda_fusion_apply=on`）：
- Gate 1：通过（`source_match_rate=1.0`）
- Gate 2：未通过（`ContactErrAbsMean delta=-0.138092`，阈值 `<= -0.15`）
- Gate 3：通过（`GeoLocalDegWeighted delta=+0.009835 <= +0.05`）
- Gate 4：通过（`mean/p90/max delta = -0.053393/-0.151608/-1.134904`）
- Gate 5：通过（`hot_foot_ball_delta=+0.001043 <= +0.02`）
- Gate 6：通过（`bones_regress_by_mean=4 <= 20`）

注：第 4/5 节中较早的“当前链路”数值对应旧 ckpt 快照；最新调参结论以第 10 节为准。

---

## 8) 建议的下一步 debug 顺序

1. 优先排查 `SIC12-15 + left foot/ball` 回退根因（phase/contact 对齐与局部权重耦合）。
2. 针对 `lowerarm_l / thumb_01_r / pinky_01_r` 做最小 ablation，确认是否是 source 变更引发的副作用。
3. 每次修复后复跑：
   - `Walk_F` rounds=5（含 `--export_joint_direct_geolocal_series`）
   - 以及 5-clip 聚合门槛（off/on 各一版）
4. 所有 gate 通过后，执行 basetrain contact 清理并在主链文档同步 contract 变更。

---

## 9) 最小复现实验命令（每次 debug 后直接复核）

### 9.1 生成 old/new 的 Walk_F 对照（rounds=5）

```bash
OUT=debug_output/_tmp_phaseD_direct_geolocal_compare_YYYYMMDD
mkdir -p "${OUT}"

# old: lambda_whitebox baseline
# Historical note: this baseline was evaluated with the now-retired whitebox validate/control lane.
# Keep the archived JSON/output as reference; do not rerun the whitebox source on current mainline.

# new: current pretrain-contact fullchain lambda_final
PYTHONUNBUFFERED=1 PYTHONPATH=. python -m train.validate.run_freerun_cycles \
  --teacher validate/teacher_batches/Walk_F_teacher.json \
  --model models/__tmp_phaseD_posttrain_pipeline_pretrain_20260305_fullchain/ckpt_last_WalkF_stage7_lambda_final_calib_20260227_fromarmchain_fullcompat_pcsrc20260305.pth \
  --rounds 5 --depth 3 --time-index-mode cycle --phase_reset_source none \
  --contacts_meas_source pretrain_contact \
  --contacts_meas_pretrain_clamp 1.0 \
  --contacts_meas_pretrain_affine_stats debug_output/_tmp_phaseb_affine_20260304/affine_fit_mix08/affine_stats.json \
  --encoder-bundle models/motion_encoder_equiv.pt.best.pt \
  --lambda_fusion_apply \
  --log_contacts --export_joint_direct_geolocal_series \
  --out "${OUT}/new_fullchain_pretrain" --force
```

### 9.2 生成你要的 `[overall]/[region_split]/[pointwise_signal]` 与分位数

```bash
PYTHONPATH=. python tools/build_stage7_old_new_summary.py \
  --old-json "${OUT}/old_lambda_whitebox/Walk_F_freerun_cycles.json" \
  --new-json "${OUT}/new_fullchain_pretrain/Walk_F_freerun_cycles.json" \
  --out-dir "${OUT}/old_whitebox_vs_new_fullchain_pretrain"

cat "${OUT}/old_whitebox_vs_new_fullchain_pretrain/global_signal_summary.txt"
cat "${OUT}/old_whitebox_vs_new_fullchain_pretrain/summary_metrics.txt"
```

### 9.3 快速 gate 检查（建议粘贴运行）

```bash
python - <<'PY'
import re
from pathlib import Path
p = Path("debug_output/_tmp_phaseD_direct_geolocal_compare_YYYYMMDD/old_whitebox_vs_new_fullchain_pretrain/global_signal_summary.txt")
t = p.read_text(encoding="utf-8")
def g(k):
    m = re.search(rf"^{k}=([+-]?[0-9]*\\.?[0-9]+)$", t, flags=re.M)
    return float(m.group(1)) if m else None
mean_delta = g("mean_delta")
leg_delta = g("leg8_mean_delta")
nonleg_delta = g("non_leg_mean_delta")
regress = g("bones_regress_by_mean")
print("mean_delta", mean_delta)
print("leg8_mean_delta", leg_delta)
print("non_leg_mean_delta", nonleg_delta)
print("bones_regress_by_mean", regress)
print("gate_mean_delta<=0", mean_delta is not None and mean_delta <= 0.0)
print("gate_leg8_delta<=+0.02", leg_delta is not None and leg_delta <= 0.02)
print("gate_nonleg_delta<=+0.015", nonleg_delta is not None and nonleg_delta <= 0.015)
print("gate_regress_bones<=20", regress is not None and regress <= 20)
PY
```

---

## 10) 2026-03-05 追加轮：71/72 调参与 λ 复核（新增）

### 10.1 目标与起点

- 目标：不改 `phaseB affine`，直接在 `rC` 路线上继续调 `71/72`，优先压 `Gate5`（`SIC12-15 + {foot_l, ball_l}`）。
- 起点 ckpt（70R）：
  - `models/__tmp_phaseD_70R_ablation_20260305/ckpt_last_rC_nonlegFalse_lr3e4_e1.pth`

### 10.2 本轮产物路径

1. `rC -> 71 -> 72 -> lambda` 串跑产物：
   - `models/__tmp_phaseD_70R_ablation_20260305_chain`
   - `debug_output/_tmp_phaseD_70R_ablation_20260305_chain_eval`
2. Stage72 参数轮调（5 个变体）：
   - `models/__tmp_phaseD_72_tune_20260305`
   - `debug_output/_tmp_phaseD_72_tune_20260305_eval`
   - 排名汇总：`debug_output/_tmp_phaseD_72_tune_20260305_eval/s72_sweep_metrics_ranked.md`
3. 基于最佳 72 的 λ 复核（2 个变体）：
   - `models/__tmp_phaseD_72_tune_20260305_lambda`
   - `debug_output/_tmp_phaseD_72_tune_20260305_lambda_eval`
   - 汇总：`debug_output/_tmp_phaseD_72_tune_20260305_lambda_eval/lambda_tune_metrics.md`

### 10.3 关键结果（Walk_F, rounds=5, cycle>=1 drop_wrap）

#### A) Stage72 调参排名（按 Gate5 hotspot 从低到高）

来自 `s72_sweep_metrics_ranked.md`：

1. `s72a_w20_proj_max45_e1_lr3e4`：`hot_foot_ball_delta_vs_old=+0.001043`
2. `s72b_w5_proj_max45_e1_lr3e4`：`+0.006665`
3. `s72c_w0_cos_max45_e1_lr3e4`：`+0.006927`
4. `s72d_w10_proj_max20_e1_lr3e4`：`+0.540503`
5. `s72e_w0_cos_max20_e1_lr3e4`：`+0.544672`

结论：
- `max_deg=45` 组（a/b/c）可把 Gate5 压到阈值内（`<= +0.02`）。
- `max_deg=20` 组（d/e）会显著恶化 hotspot。

#### B) λ 复核（基于调优后 72）

来自 `lambda_tune_metrics.md`：

`lam_from_s72a`（推荐）：
- `GeoLocalDegWeighted` 相对 old: `+0.009835`
- `ContactErrAbsMean` 相对 old: `-0.138092`
- `ContactMeasGtAbsMean` 相对 old: `+0.208499`
- `overall_mean_delta`: `-0.053393`
- `hot_foot_ball_delta`: `+0.001043`
- `source_match_rate`: `1.0`

`lam_from_s72b`：
- 与 `s72a` 接近，`hot_foot_ball_delta=+0.006665`，略差于 `s72a`。

### 10.4 本轮最佳配置（截至 2026-03-05）

- **最佳链路**：`rC -> stage71 -> s72a -> lam_from_s72a`
- `s72a` 关键覆盖参数：
  - `--epochs 1`
  - `--lr 0.0003`
  - `--direct_pose_leg_align_weight 20`
  - `--direct_pose_leg_align_mode proj`
  - `--direct_pose_leg_max_deg 45`

Gate 视角（按第 7 节原定义）：
- Gate5（hotspot）在本配置下可通过（`+0.001043 <= +0.02`）。
- Gate3（pose 主指标）可通过（`+0.009835 <= +0.05`）。
- Gate2 在当前阈值 `<= -0.15` 下仍略差（当前 `-0.138092`）。

### 10.5 最小复现命令（本轮最佳链路）

```bash
ENCODER_BUNDLE=models/motion_encoder_equiv.pt.best.pt
AFFINE_STATS=debug_output/_tmp_phaseb_affine_20260304/affine_fit_mix08/affine_stats.json

# 1) stage71（从 rC 70R 起跑）
PYTHONPATH=. python -m train.posttrain \
  --config config/posttrain_WalkF_stage7_71_legonly_after_nonlegproj256_20260227_fromarmchain.json \
  --ckpt_in models/__tmp_phaseD_70R_ablation_20260305/ckpt_last_rC_nonlegFalse_lr3e4_e1.pth \
  --out_dir models/__tmp_phaseD_70R_ablation_20260305_chain \
  --run_name rCchain_stage71_from_rC70R \
  --posttrain_contacts_source pretrain_contact \
  --encoder_bundle "${ENCODER_BUNDLE}" \
  --posttrain_contacts_pretrain_affine_stats "${AFFINE_STATS}"

# 2) stage72（s72a）
PYTHONPATH=. python -m train.posttrain \
  --config config/posttrain_WalkF_stage7_72_legomega_after_nonlegproj256_20260227_fromarmchain.json \
  --ckpt_in models/__tmp_phaseD_70R_ablation_20260305_chain/ckpt_last_rCchain_stage71_from_rC70R.pth \
  --out_dir models/__tmp_phaseD_72_tune_20260305 \
  --run_name s72a_w20_proj_max45_e1_lr3e4 \
  --epochs 1 --lr 0.0003 \
  --direct_pose_leg_align_weight 20 \
  --direct_pose_leg_align_mode proj \
  --direct_pose_leg_max_deg 45 \
  --posttrain_contacts_source pretrain_contact \
  --encoder_bundle "${ENCODER_BUNDLE}" \
  --posttrain_contacts_pretrain_affine_stats "${AFFINE_STATS}"

# 3) lambda（从 s72a 起跑）
PYTHONPATH=. python -m train.posttrain \
  --config config/posttrain_WalkF_stage7_lambda_final_calib_20260227_fromarmchain_fullcompat.json \
  --ckpt_in models/__tmp_phaseD_72_tune_20260305/ckpt_last_s72a_w20_proj_max45_e1_lr3e4.pth \
  --out_dir models/__tmp_phaseD_72_tune_20260305_lambda \
  --run_name lam_from_s72a \
  --posttrain_contacts_source pretrain_contact \
  --encoder_bundle "${ENCODER_BUNDLE}" \
  --posttrain_contacts_pretrain_affine_stats "${AFFINE_STATS}"

# 4) freerun 评估（on 口径）
PYTHONPATH=. python -m train.validate.run_freerun_cycles \
  --teacher validate/teacher_batches/Walk_F_teacher.json \
  --model models/__tmp_phaseD_72_tune_20260305_lambda/ckpt_last_lam_from_s72a.pth \
  --rounds 5 --depth 3 --time-index-mode cycle --phase_reset_source none \
  --contacts_meas_source pretrain_contact \
  --contacts_meas_pretrain_clamp 1.0 \
  --contacts_meas_pretrain_affine_stats "${AFFINE_STATS}" \
  --encoder-bundle "${ENCODER_BUNDLE}" \
  --lambda_fusion_apply \
  --log_contacts --export_joint_direct_geolocal_series \
  --out debug_output/_tmp_phaseD_72_tune_20260305_lambda_eval/lam_from_s72a --force
```

---

## 11) 2026-03-05 追加轮：禁用 `contact_plan GRU` 全链 A/B（新增）

### 11.1 实验目的

- 目标：在不改 `posttrain_contacts_source=pretrain_contact + affine` 主链口径的前提下，验证“禁用 `contact_plan GRU`”对全链结果的影响。
- 对照：
  1. A（ref）：`fullchain_pretrain`（现有 2026-03-05 主链，`contact_plan GRU` 保持启用）
  2. B（new）：`fullchain_pretrain_noplan_gru`（禁用 `contact_plan GRU` 后重跑 8-stage）

### 11.2 禁用实现口径（本轮）

起始 ckpt 采用“权重清零禁用”：

- 基础 ckpt：
  - `models/MLPL2_DirectBranch_v1__base_to_stage7_lambdafinal_20260226_cpu/exp_phase_DirectBranch_v1_d1_fromscratch_20260226/ckpt_best_free_exp_phase_DirectBranch_v1_d1_fromscratch_20260226.pth`
- 生成禁用版 bootstrap：
  - `models/__tmp_phaseD_contact_plan_gru_off_bootstrap_20260305/ckpt_base_contact_plan_gru_off.pth`
- 清零范围：
  - `contact_plan_cell.*`
  - `contact_plan_init_head.*`
  - `contact_plan_head.*`（并将 `contact_plan_head.4.bias=-20`）
  - `contact_plan_time_head.*`
  - `contact_plan_phase_head.*`
  - `event_clock_gate.*`
  - `event_clock_corrector.*`
  - `contact_phase_state_delta_head.*`
  - `contact_plan_init_z`、`contact_phase_state_init`

### 11.3 产物路径

1. 全链训练产物（8-stage）：
   - `models/__tmp_phaseD_posttrain_pipeline_pretrain_20260305_fullchain_noplan_gru`
   - 最终 ckpt：
     - `ckpt_last_WalkF_stage7_lambda_final_calib_20260227_fromarmchain_fullcompat_pcsrc20260305_noplanGRU.pth`
2. 5-clip 评估产物：
   - off：`debug_output/_tmp_phaseD_posttrain_pipeline_pretrain_20260305_fullchain_noplan_gru_eval`
   - on：`debug_output/_tmp_phaseD_posttrain_pipeline_pretrain_20260305_fullchain_noplan_gru_eval_lambda_apply`
3. 汇总：
   - `debug_output/_tmp_phaseD_posttrain_pipeline_pretrain_20260305_fullchain_noplan_gru/summary_fullchain_noplan_gru_vs_refs.md`
   - `debug_output/_tmp_phaseD_posttrain_pipeline_pretrain_20260305_fullchain_noplan_gru/summary_fullchain_noplan_gru_vs_refs.json`
4. Walk_F rounds=5 DirectGeoLocal 对照：
   - `debug_output/_tmp_phaseD_direct_geolocal_compare_20260305_noplan_gru_r5/old_whitebox_vs_new_noplan_gru`
   - `debug_output/_tmp_phaseD_direct_geolocal_compare_20260305_noplan_gru_r5/old_fullchain_vs_new_noplan_gru`

### 11.4 关键结果（5-clip 聚合，round>=1）

#### A) `lambda_fusion_apply=off`

`new_noplan_gru` vs `ref_fullchain_plan_on`：

- `GeoLocalDegWeighted`: `37.116532` vs `34.433958` -> `+2.682574`
- `ContactErrAbsMean`: `0.473937` vs `0.123358` -> `+0.350579`
- `ContactMeasGtAbsMean`: `0.422050` vs `0.421413` -> `+0.000636`
- `source_match_rate`: `1.000000` vs `1.000000`

#### B) `lambda_fusion_apply=on`

`new_noplan_gru` vs `ref_fullchain_plan_on`：

- `GeoLocalDegWeighted`: `7.351717` vs `7.336871` -> `+0.014845`
- `ContactErrAbsMean`: `0.484717` vs `0.133698` -> `+0.351018`
- `ContactMeasGtAbsMean`: `0.423701` vs `0.422625` -> `+0.001076`
- `source_match_rate`: `1.000000` vs `1.000000`

### 11.5 DirectGeoLocal（Walk_F, rounds=5, cycle>=1 drop_wrap）

1. 对 `old_whitebox`：
   - `mean_delta=-0.005673`
   - `leg8_mean_delta=-0.012996`
   - `non_leg_mean_delta=-0.004089`
   - `bones_regress_by_mean=21`
2. 对 `old_fullchain_pretrain(plan_on)`：
   - `mean_delta=+0.006501`
   - `leg8_mean_delta=+0.006195`
   - `non_leg_mean_delta=+0.006567`
   - `bones_regress_by_mean=27`

### 11.6 结论

1. 禁用 `contact_plan GRU` 后，`ContactErrAbsMean` 在 off/on 两个口径均显著恶化（约 `+0.35`），该路线不满足 cleanup gate。
2. `lambda_fusion_apply=on` 下 pose 主指标仅轻微变化（`GeoLocalDegWeighted +0.014845`），但 contact 主指标退化幅度过大，整体不可接受。
3. 现阶段不建议推进“禁用 `contact_plan GRU`”作为主线；建议保持 GRU 路径并继续围绕 Gate2（contact 残差）做针对性优化。
