# Posttrain Pipeline (Legacy Old-Boundary Chain)

> Last updated: 2026-04-13  
> Status: legacy control / former canonical  
> Caveat: `N=5 / limited-N`

这份文档现在只保留一件事：

- **把 old-boundary chain 作为可复现的 legacy control 固定下来**

它不再是当前默认 source of truth。当前默认文档是：

- `docs/posttrain_pipeline.md`

---

## 1) TL;DR

legacy old-boundary chain 应该被理解为一条连续链：

- `legacy Stage6-old-boundary exit -> 70a -> new70b_replace_lowdrift -> 70R -> 71(lr=3e-4) -> 72(lr=1e-4) -> lambda`

这条链现在只用于：

- historical reproduction
- old-boundary control
- StepC promotion 的比较基线

同样也不要再把它写成：

- “一个 stage6 故事 + 一个后面随便接的 stage7 故事”

在 legacy 语境里，真正的语义是：

- `stage6 -> 70a -> replace -> 70R -> 71 -> 72 -> lambda`
  构成了旧 boundary contract 下的一条完整 downstream absorbability chain

但这只是 historical control。当前正常的 `Stage6` 出口语义是：

- `Stage6-StepC handoff`

---

## 2) Source of Truth

### 2.1 Historical decision roots

- `docs/Problems/active/2026-03-14_oldd1_newflow_leg_regression_handoff.md`
- `docs/Problems/active/2026-03-14_oldd1_skip70b_replace_lowdrift_experiment.md`
- `docs/Problems/active/2026-03-14_71_regression_attribution.md`
- `docs/Problems/active/2026-03-14_72_loss_curve_attribution.md`
- `docs/Problems/active/2026-03-14_72_lowlr_sweep.md`
- `docs/Problems/active/2026-03-15_72_lowlr_to_lambda.md`

### 2.2 Comparison-compatible legacy bundle

当你需要和当前 StepC canonical 做 **同口径比较** 时，优先使用 April 12 audit 里锁定的 legacy bundle：

- `debug_output/_tmp_posttrain_pipeline_from_bestfree_20260317/configs/posttrain_70b_replace_lowdrift_fromfresh_20260317.json`
- `debug_output/_tmp_posttrain_pipeline_from_bestfree_20260317/configs/posttrain_70R_fromfresh_20260317.json`
- `debug_output/_tmp_71_lowlr_sweep_20260314/configs/posttrain_71_lr3e4_20260314.json`
- `debug_output/_tmp_72_lowlr_sweep_20260314/configs/posttrain_72_lr1e4_20260314.json`
- `config/posttrain_WalkF_stage7_lambda_final_calib_20260227_fromarmchain_fullcompat.json`

换句话说：

- March problem docs 解释这条链怎么形成
- April comparison bundle 负责把它固定成今天仍可复现的 legacy control

---

## 3) Preferred Reproduction

### 3.1 Shared runtime contract

这条 legacy control 绑定的是下面这套运行面：

- contacts source: `pretrain_contact`
- clamp: `1.0`
- encoder bundle: `models/motion_encoder_equiv.pt.best.pt`
- affine stats: `debug_output/_tmp_phaseb_affine_20260304/affine_fit_mix08/affine_stats.json`
- eval contract: historical `model-source`

**注意**：很多 JSON 里自带的 `encoder_bundle` 是旧值或 base-config 默认值。  
真正 accepted run 应该看 **实际 lane command / decision artifact**，而不是只看 base config。

### 3.2 Manual stage map

| Link | Launcher | Config | Input ckpt | Locked recipe |
|---|---|---|---|---|
| `legacy Stage6-old-boundary exit -> 70a` | `python -m train.posttrain` | `config/posttrain_WalkF_stage7_70a_splitB2_pe32h512_20260227_fromarmchain.json` | `models/__tmp_posttrain_pipeline_from_bestfree_20260317/stage6/ckpt_last_WalkF_stage6_fromfresh_20260317.pth` | `epochs=5`, `steps_per_epoch=60`, `lr=1e-3` |
| `70a -> new70b_replace_lowdrift` | `python -m train.posttrain` | `debug_output/_tmp_posttrain_pipeline_from_bestfree_20260317/configs/posttrain_70b_replace_lowdrift_fromfresh_20260317.json` | `models/__tmp_posttrain_pipeline_from_bestfree_20260317/warmstart/ckpt_last_70a_replace_zerophase_20260317.pth` | `epochs=1`, `steps_per_epoch=60`, `lr=3e-4`, raw `70b` is not accepted |
| `replace -> 70R` | `python tools/run_posttrain_nonleg_trunk_ablation.py --trunk-mode full` | `debug_output/_tmp_posttrain_pipeline_from_bestfree_20260317/configs/posttrain_70R_fromfresh_20260317.json` | `models/__tmp_posttrain_pipeline_from_bestfree_20260317/70b_replace_lowdrift/ckpt_last_WalkF_stage7_70b_replace_lowdrift_fromfresh_20260317.pth` | `epochs=1`, `steps_per_epoch=180`, save steps `0,1,5,20,60,180` |
| `70R -> 71(lr=3e-4)` | `python -m train.posttrain` | `debug_output/_tmp_71_lowlr_sweep_20260314/configs/posttrain_71_lr3e4_20260314.json` | `models/__tmp_oldd1_skip70b_lowdrift_to71_20260314/70R/ckpt_last_WalkF_stage7_70R_from_oldd1_lowdrift_replace_20260314.pth` | `epochs=3`, `steps_per_epoch=60`, `lr=3e-4` |
| `71 -> 72(lr=1e-4)` | `python -m train.posttrain` | `debug_output/_tmp_72_lowlr_sweep_20260314/configs/posttrain_72_lr1e4_20260314.json` | `models/__tmp_71_lowlr_sweep_20260314/lr3e4/ckpt_last_WalkF_stage7_71_lr3e4_from_candidate70R_20260314.pth` | `epochs=3`, `steps_per_epoch=60`, `lr=1e-4` |
| `72 -> lambda` | `python -m train.posttrain` | `config/posttrain_WalkF_stage7_lambda_final_calib_20260227_fromarmchain_fullcompat.json` | `models/__tmp_72_lowlr_sweep_20260314/lr1e4/ckpt_last_WalkF_stage7_72_lr1e4_from_lowlr71_20260314.pth` | `epochs=1`, `steps_per_epoch=200`, `lr=2e-4`, `train_lambda_head=true` |

### 3.3 Shared `train.posttrain` template

```bash
PYTHONPATH=. python -m train.posttrain \
  --config <config_json> \
  --ckpt_in <input_ckpt> \
  --out_dir <out_dir> \
  --run_name <run_name> \
  --posttrain_contacts_source pretrain_contact \
  --posttrain_contacts_pretrain_clamp 1.0 \
  --encoder_bundle models/motion_encoder_equiv.pt.best.pt \
  --posttrain_contacts_pretrain_affine_stats \
    debug_output/_tmp_phaseb_affine_20260304/affine_fit_mix08/affine_stats.json
```

### 3.4 `70R` special launcher

```bash
PYTHONPATH=. python tools/run_posttrain_nonleg_trunk_ablation.py \
  --config debug_output/_tmp_posttrain_pipeline_from_bestfree_20260317/configs/posttrain_70R_fromfresh_20260317.json \
  --trunk-mode full \
  --out-dir models/__tmp_posttrain_pipeline_from_bestfree_20260317/70R \
  --run-name WalkF_stage7_70R_fromfresh_s180_20260317 \
  --epochs 1 \
  --steps-per-epoch 180 \
  --save-step-ckpts 0,1,5,20,60,180
```

### 3.5 `lambda` exact accepted command

legacy `lambda` 的 accepted lane 在 `debug_output/_tmp_72_lowlr_to_lambda_20260315/lane.log` 中已经固定，repo-relative 形式如下：

```bash
PYTHONPATH=. python -m train.posttrain \
  --config config/posttrain_WalkF_stage7_lambda_final_calib_20260227_fromarmchain_fullcompat.json \
  --ckpt_in models/__tmp_72_lowlr_sweep_20260314/lr1e4/ckpt_last_WalkF_stage7_72_lr1e4_from_lowlr71_20260314.pth \
  --out_dir models/__tmp_72_lowlr_to_lambda_20260315/lambda \
  --run_name WalkF_stage7_lambda_from_lowlr72lr1e4_20260315 \
  --posttrain_contacts_source pretrain_contact \
  --posttrain_contacts_pretrain_clamp 1.0 \
  --encoder_bundle models/motion_encoder_equiv.pt.best.pt \
  --posttrain_contacts_pretrain_affine_stats \
    debug_output/_tmp_phaseb_affine_20260304/affine_fit_mix08/affine_stats.json
```

---

## 4) Stage Semantics

### 4.1 `70a`

- old-boundary 下最后一个 plain upstream continuation
- 是 legacy `Stage6-old-boundary exit` 的第一个 downstream consumer，而不是孤立 probe

### 4.2 `new70b_replace_lowdrift`

- 这是 accepted replace handoff
- raw `70b` 只是 archive / diagnostic，不是 accepted downstream lane

### 4.3 `70R`

- 历史上用来做 nonleg recovery
- 也是 old-boundary chain 里最后一个真正决定能否继续下游的 stage

### 4.4 `71 / 72 / lambda`

- `71` = lower-LR leg continuation
- `72` = lower-LR legomega continuation
- `lambda` = chain closure；在 legacy 叙事里它更接近 final calibration / closure，而不是新机制本身

---

## 5) Why This Doc Is Still Kept

这份文档还要保留，因为它仍然是理解下面几个问题的最短路径：

- old-boundary downstream absorbability 到底是什么
- `top3` 为什么会变成 old-boundary-compatible anchor/control
- StepC promotion 到底替换了 legacy chain 的哪一层 boundary mismatch

最准确的 status：

- 它不再是默认
- 但它仍然是 legacy-control baseline

---

## 6) Relationship to Top3

在当前文档系统里，`top3` 应该被理解成：

- the donor range that this old boundary could still absorb
- legacy-compatible anchor/control

而不是：

- universal optimum

见：`docs/posttrain_pipeline_top3_anchor_control.md`

---

## 7) Caveats

- 不要把这份文件当成 current global canonical
- 不要把 old-boundary 语言继续当成 present-tense default interpretation；当前 `Stage6` 正常出口是 `Stage6-StepC handoff`
- `N=5 / limited-N` 仍然适用
- legacy chain 解释的是 **control / history / comparison baseline**，不是今天要推广的新默认
