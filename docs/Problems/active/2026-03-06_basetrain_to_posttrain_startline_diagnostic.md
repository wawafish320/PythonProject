# 2026-03-06 Basetrain -> Posttrain 起跑前诊断固定化（leg/nonleg）

Last updated: 2026-03-06

## 1) 结论（确认）

- 这个思路是对的：把“`basetrain ckpt` 的 group 分布 + Stage6 step1 `leg/nonleg` 起点比值”作为固定起跑前诊断，能在进入长链 posttrain 前提前识别 leg 风险。
- 当前链路已经支持该诊断，不需要改主流程代码即可落地执行。
- 建议将其纳入 Stage6 前的标准检查项（先作为 soft gate，后续可升级为 hard gate）。

---

## 2) 证据核对（本次数据）

1. Basetrain -> Posttrain 起点链路支持：
   - 主链推荐顺序以 Stage6 开始（`docs/posttrain_pipeline.md` 第 68 行）。
   - Stage6 config 的 `ckpt_in` 直接指向 `ckpt_best_free_*`（`config/posttrain_WalkF_stage6_direct_cond_anchor_splitfirst_3way_armchain_pe32h512_20260227.json` 第 2 行）。
2. Freerun 支持全关节 DirectGeoLocalDeg 序列导出：
   - CLI 参数 `--export_joint_direct_geolocal_series` 已提供（`train/validate/run_freerun_cycles.py` 第 10156 行）。
   - 导出字段 `per_step_direct_geolocal_deg.DirectGeoLocalDeg` 已实现（`train/validate/run_freerun_cycles.py` 第 8200 行）。
3. 本次 group validation 分布（`cycle>=1`, `drop_wrap=true`）：
   - 来源：`debug_output/__tmp_basetrain_bestfree_groupdist_20260305/group_summary.json`
   - `leg mean = 10.6688°`
   - `arm mean = 6.6670°`
   - `else mean = 2.0487°`
   - `nonleg mean = 5.2940°`
4. Stage6 初始 loss 起点（step1 + head20）：
   - 来源：`debug_output/__tmp_basetrain_bestfree_groupdist_20260305/posttrain_stage6_init_stats.json`
   - step1: `dir_leg_base=0.1919`, `dir_nonleg_base=0.0624`, `leg/nonleg=3.075x`
   - head20 mean: `leg/nonleg=3.550x`

---

## 3) 解读

- `leg_mean/nonleg_mean = 10.6688/5.2940 = 2.02x`，且 Stage6 step1 直接出现 `3.07x` gap，与“leg 在 basetrain 终点相对 nonleg 偏高”一致。
- gap 出现在 Stage6 起点（step1），说明问题主要来自输入 ckpt 状态，而不是长链 posttrain 后段才新引入。
- 工程上可将该状态标记为“leg 欠拟合风险高”。
- 注：这属于强诊断信号，不等于唯一因果证明；后续仍建议结合 targeted ablation 做机制确认。

---

## 4) 固定“起跑前诊断”SOP（v0）

### 4.1 输入

- ckpt：`ckpt_best_free_*`
- teacher：`validate/teacher_batches/Walk_F_teacher.json`
- Stage6 config：主链配置文件

### 4.2 步骤 A：导出全关节 DirectGeoLocalDeg 序列

```bash
PYTHONPATH=. python -m train.validate.run_freerun_cycles \
  --teacher validate/teacher_batches/Walk_F_teacher.json \
  --model <ckpt_best_free> \
  --rounds 5 --depth 3 --time-index-mode cycle --phase_reset_source none \
  --contacts_meas_source pretrain_contact \
  --contacts_meas_pretrain_clamp 1.0 \
  --contacts_meas_pretrain_affine_stats <affine_stats.json> \
  --encoder-bundle <encoder_bundle> \
  --export_joint_direct_geolocal_series \
  --out debug_output/__tmp_prestart_diag_<date> --force
```

### 4.3 步骤 B：按 leg/arm/else/nonleg 聚合（`cycle>=1`, `drop_wrap=true`）

- 产出：`group_summary.json`
- 最低输出要求：`mean/p50/p90/p95/samples`

### 4.4 步骤 C：Stage6 短跑并读取起点 gap

- 运行 Stage6（至少前 20 步日志），从 `posttrain_log_*.json` 提取：
  - `dir_leg_base`
  - `dir_nonleg_base`
  - `leg_over_nonleg`（step1 + head20 mean）
- 产出：`posttrain_stage6_init_stats.json`

### 4.5 判定规则（建议阈值）

- `Alert-A`: freerun `leg_mean/nonleg_mean >= 1.8`
- `Alert-B`: Stage6 step1 `leg_over_nonleg >= 2.5`
- `Alert-C`: Stage6 head20 `leg_over_nonleg >= 3.0`
- 若 A+B 同时触发，标记为“高风险起点”，优先处理 leg 侧问题，再进入全链。

---

## 5) 本次样例判定（2026-03-05 数据）

- A 触发：`10.67 / 5.29 = 2.02`（是）
- B 触发：`3.07`（是）
- C 触发：`3.55`（是）

结论：你的判断成立；该方法值得固定为 Stage6 前“起跑前诊断”。

---

## 6) 文档与产物关联

- `docs/Problems/active/2026-03-05_pretrain_contact_route_cleanup_readiness.md`
- `docs/posttrain_pipeline.md`
- `config/posttrain_WalkF_stage6_direct_cond_anchor_splitfirst_3way_armchain_pe32h512_20260227.json`
- `debug_output/__tmp_basetrain_bestfree_groupdist_20260305/group_summary.json`
- `debug_output/__tmp_basetrain_bestfree_groupdist_20260305/posttrain_stage6_init_stats.json`
