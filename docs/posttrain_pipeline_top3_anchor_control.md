# Posttrain Pipeline (Top3 Anchor / Legacy Control)

> Last updated: 2026-04-13  
> Status: legacy-compatible anchor / control  
> Caveat: `N=5 / limited-N`

这份文档现在补充一个关键约束：

- `top3` 不是一条独立的“canonical downstream pipeline”
- 它首先是 **anchor / donor contract**
- current default 下，它应该接到 `Stage6-StepC handoff` 再进入 downstream chain

---

## 1) TL;DR

现在对 `top3` 最准确的写法是：

- `top3 = donor anchor / control`
- `top3 = compatibility reference range`
- `top3 = anchor contract`, not a universal natural optimum

如果进入 current default posttrain，链路应写成：

- `top3 donor -> Stage6-StepC handoff -> 70a -> replace -> 70R -> 71 -> 72 -> lambda`

只有复现 legacy control 时，才写成：

- `top3 donor -> legacy Stage6-old-boundary exit -> 70a -> ...`

不要再写：

- `top3 天然最优`
- `top3 is the final semantic scope`
- `top3 自己就等于一整条 canonical pipeline`

---

## 2) What `top3` Is Operationally

`top3` 在当前文档系统里应该拆成两层：

### 2.1 As an upstream anchor

它定义的是：

- donor compatibility 的 anchor range
- historical old-boundary 最容易 absorb 的 reference donor

### 2.2 As a downstream reference

它后面可以挂两种不同阅读方式：

- **legacy old-boundary control**  
  见：`docs/posttrain_pipeline_legacy_old_boundary.md`
- **StepC / expansion-family comparison reference**  
  见：`docs/posttrain_pipeline.md` 与 `docs/posttrain_pipeline_top7_clean_stepc.md`

也就是说，`top3` 本身不是“下游链条名”。它只是 donor anchor；current default 的 Stage6 出口仍然是 `Stage6-StepC handoff`。

---

## 3) Concrete Anchor Artifacts

这里把文档里最常用的两个 `top3` anchor 层级固定下来，避免后面再只说概念、不说 artifact。

### 3.1 Historical fixed-control anchor

这个层级主要用于：

- old-boundary absorbability
- fixed-control stage6 exact reference

Artifacts:

- basetrain config: `config/exp_phase_DirectBranch_v1_d1_cp015_tailk3_corridor_hold_tail15_phasea050_fixedsched_ep014center_control_denseckpt_seed2024_20260330.json`
- basetrain ckpt: `models/cp015_phasecd_min_ablation_20260330/exp_phase_DirectBranch_v1_d1_cp015_tailk3_corridor_hold_tail15_phasea050_fixedsched_ep014center_control_denseckpt_seed2024_20260330/ckpt_epoch_015.pth`
- stage6 config: `config/posttrain_WalkF_stage6_direct_cond_anchor_splitfirst_3way_armchain_pe32h512_20260227.json`
- stage6 exact summary: `debug_output/_tmp_phasecd_stage6_trend_top3_fullrerun_20260330/stage6_trend_top3_fullrerun_summary_20260330.md`
- stage6 exact eval: `debug_output/_tmp_phasecd_stage6_trend_top3_fullrerun_20260330/control_denseckpt_final/stage6_freerun/Walk_F_freerun_cycles.json`

### 3.2 `E1-top3` anchor (expansion-framework reference)

这个层级主要用于：

- top3-vs-top7 support-scope / expansion-family discussion
- April 8/9 之后的 anchor-vs-expansion narrative

Artifacts:

- basetrain config: `config/exp_phase_DirectBranch_v1_d1_cp015_tailk3_rankmix_tw020_corridor_hold_tail15_phasea050_fixedsched_ep014center_control_denseckpt_seed2024_20260408.json`
- basetrain ckpt: `models/cp015_phasecd_tailk_probe_20260331/exp_phase_DirectBranch_v1_d1_cp015_tailk3_rankmix_tw020_corridor_hold_tail15_phasea050_fixedsched_ep014center_control_denseckpt_seed2024_20260408/ckpt_epoch_014.pth`
- stage6 tailfix config: `config/posttrain_WalkF_stage6_direct_cond_anchor_splitfirst_3way_armchain_pe32h512_tailfix_lr3e4_e8x60_wd1e4_reinit1_20260401.json`
- stage6 tailfix ckpt: `models/__tmp_cp015_tailk3_rankmix_tw020_stage6_tailfix_e1_20260408/lr3e4_e8x60_wd1e4_reinit1/ckpt_last_lr3e4_e8x60_wd1e4_reinit1_stage6_tailfix_tailk3_rankmix_tw020_e1_20260408.pth`
- stage70a config: `debug_output/_tmp_ep014center_70a_lowlr_sweep_20260328/configs/posttrain_70a_lr3e4_from_ep014center_20260328.json`
- stage70a eval: `debug_output/_tmp_cp015_tailk3_rankmix_tw020_stage70a_from_tailfix_e1_20260408/eval_model_source/Walk_F_freerun_cycles.json`
- reference summary: `debug_output/_tmp_cp015_tailk_curriculum_e2a_20260408/summary.json`

---

## 4) Minimal Reproduction

### 4.1 Basetrain anchor command

对这类 `exp_phase_*` anchor config，主训练入口是：

```bash
PYTHONPATH=. python -m train.training_MPL \
  --config_json config/exp_phase_DirectBranch_v1_d1_cp015_tailk3_corridor_hold_tail15_phasea050_fixedsched_ep014center_control_denseckpt_seed2024_20260330.json
```

如果你要复现 `E1-top3` 而不是 March 30 fixed-control anchor，把 `--config_json` 换成：

- `config/exp_phase_DirectBranch_v1_d1_cp015_tailk3_rankmix_tw020_corridor_hold_tail15_phasea050_fixedsched_ep014center_control_denseckpt_seed2024_20260408.json`

### 4.2 Stage6 anchor command

固定-control anchor 的 stage6 exact reference 是 historical old-boundary/control 口径，可以这样跑：

```bash
PYTHONPATH=. debug_output/_tmp_phasecd_min_ablation_20260330/cpu_nomps_exec.py -m train.posttrain \
  --config config/posttrain_WalkF_stage6_direct_cond_anchor_splitfirst_3way_armchain_pe32h512_20260227.json \
  --ckpt_in models/cp015_phasecd_min_ablation_20260330/exp_phase_DirectBranch_v1_d1_cp015_tailk3_corridor_hold_tail15_phasea050_fixedsched_ep014center_control_denseckpt_seed2024_20260330/ckpt_epoch_015.pth \
  --out_dir models/__tmp_phasecd_stage6_trend_top3_fullrerun_20260330/control_denseckpt_final \
  --run_name control_denseckpt_final_stage6_trend_fullrerun_20260330 \
  --posttrain_contacts_source pretrain_contact \
  --posttrain_contacts_pretrain_clamp 1.0 \
  --encoder_bundle models/motion_encoder_equiv.pt.best.pt \
  --posttrain_contacts_pretrain_affine_stats \
    debug_output/_tmp_phaseb_affine_20260304/affine_fit_mix08/affine_stats.json
```

### 4.3 `E1-top3` stage6/70a concrete config roots

如果你要追 `E1-top3` 的更现代 anchor，而不是 March 30 fixed-control exact lane，至少要锁住这组 config / io：

| Link | Config | Input ckpt | Output root |
|---|---|---|---|
| `E1-top3 basetrain -> stage6 tailfix` | `config/posttrain_WalkF_stage6_direct_cond_anchor_splitfirst_3way_armchain_pe32h512_tailfix_lr3e4_e8x60_wd1e4_reinit1_20260401.json` | `models/cp015_phasecd_tailk_probe_20260331/exp_phase_DirectBranch_v1_d1_cp015_tailk3_rankmix_tw020_corridor_hold_tail15_phasea050_fixedsched_ep014center_control_denseckpt_seed2024_20260408/ckpt_epoch_014.pth` | `models/__tmp_cp015_tailk3_rankmix_tw020_stage6_tailfix_e1_20260408/lr3e4_e8x60_wd1e4_reinit1` |
| `E1-top3 stage6 tailfix -> stage70a` | `debug_output/_tmp_ep014center_70a_lowlr_sweep_20260328/configs/posttrain_70a_lr3e4_from_ep014center_20260328.json` | `models/__tmp_cp015_tailk3_rankmix_tw020_stage6_tailfix_e1_20260408/lr3e4_e8x60_wd1e4_reinit1/ckpt_last_lr3e4_e8x60_wd1e4_reinit1_stage6_tailfix_tailk3_rankmix_tw020_e1_20260408.pth` | `models/__tmp_cp015_tailk3_rankmix_tw020_stage70a_from_tailfix_e1_20260408` |

如果目标是 current default，而不是复现旧 anchor/control，则把这层 donor anchor 接到 `Stage6-StepC handoff` 再继续 downstream。

---

## 5) Why `top3` Still Matters

`top3` 仍然有用，因为它稳定地锚住了下面这三个问题：

- old boundary 当时到底还能 absorb 什么 donor range
- donor burden 和 boundary mismatch 应该怎么拆开看
- top7 应该被理解为 expansion family，而不是必须直接取代 anchor 的 monolithic donor

最准确的一句话：

> `top3` 是 anchor/control，不是 metaphysical optimum。

---

## 6) Relationship to `top7`

最准确的配对关系仍然是：

- `top3` = anchor/control
- `top7` = expansion family

推荐的阅读规则：

- 判断 `top7` 时，要看它在 clean StepC handoff 下是否仍然 absorbable
- 不要只看它是否能在 legacy old-boundary 条件下直接复制 `top3`

见：`docs/posttrain_pipeline_top7_clean_stepc.md`

---

## 7) Caveats

- `top3` 不是当前 global canonical 本身
- `top3` 也不是一条独立的 downstream chain name
- 讨论 `top3` 时必须同时说明你说的是哪一层 artifact：
  - fixed-control exact anchor
  - `E1-top3` expansion-framework anchor
- `N=5 / limited-N`
