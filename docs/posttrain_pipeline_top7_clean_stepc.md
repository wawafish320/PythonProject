# Posttrain Pipeline (Top7 Clean StepC Family Default)

> Last updated: 2026-04-13  
> Status: top7-family default under clean StepC  
> Caveat: `N=5 / limited-N`

这份文档这次补的重点有两个：

1. 把 `stage6` 和 `stage7` 明确写成一条连续链  
2. 把对应的 **运行入口 / config / recipe** 固定下来

---

## 1) TL;DR

对 `top7` 最准确的当前写法是：

- `top7` 不是“天然太 aggressive”
- 它在 legacy old-boundary 下看起来更糟，主要是 boundary contract 本身不兼容
- 一旦切到 **clean StepC handoff**，整个 family 的 downstream chain 变成可行

当前应当把 `top7` family default 明确写成：

- `top7 donor(epoch014) -> Stage6 clean-StepC handoff -> 70a clean-StepC -> replace clean-StepC -> 70R clean-StepC -> 71 clean-StepC -> 72 clean-StepC -> lambda clean-StepC`

这里和 `docs/posttrain_pipeline.md` 的差别是：

- global canonical 文档强调的是 **boundary contract promotion**
- 本文强调的是 **这个 promoted contract 在 top7 family 里的落地默认链**

---

## 2) Source of Truth

### 2.1 Decision artifacts

- `debug_output/_tmp_top7_clean_stage6_stepc_chain_20260412/summary.md`
- `debug_output/_tmp_top7_clean_stage6_stepc_chain_20260412/decision.md`
- `docs/train_design/2026-04-12_top7_clean_stage6_stepc_causality_record.md`

### 2.2 Entry script

- `tools/run_top7_clean_stage6_stepc_chain.py`

### 2.3 Reference lanes

这个脚本内部固定比较三条 lane：

- `O`: old stage6 handoff -> old-cut downstream
- `P`: old stage6 handoff -> downstream StepC compatibility
- `C`: clean stage6-StepC handoff -> clean-StepC downstream

本文默认指的 family default 是：

- `C lane`

---

## 3) Preferred Reproduction

### 3.1 One-command entrypoint

```bash
PYTHONPATH=. python tools/run_top7_clean_stage6_stepc_chain.py
```

这个脚本会从 canonical `top7 donor` 自动串起：

- `Stage6 clean-StepC handoff`
- `70a clean`
- `replace clean`
- `70R clean`
- `71 clean`
- `72 clean`
- `lambda clean`

### 3.2 Locked runtime contract

这条 top7-family default 绑定的是：

- contacts source: `pretrain_contact`
- clamp: `1.0`
- encoder bundle: `models/motion_encoder_equiv.pt.best.pt`
- affine stats: `debug_output/_tmp_phaseb_affine_20260304/affine_fit_mix08/affine_stats.json`

---

## 4) Stage6 Clean-StepC Is The Entry

`top7 clean-StepC` 是“当前 Stage6 正常出口就是 StepC handoff”这条原则的显式训练版。

### 4.1 正确的写法

不要再写成：

- “先有一个 stage6 probe，然后 stage7 另开一条线”

这里更准确的写法是：

- `top7 donor -> Stage6 clean-StepC handoff -> 70a -> replace -> 70R -> 71 -> 72 -> lambda`

### 4.2 为什么必须这么写

因为在这条链里：

1. `Stage6 clean-StepC handoff` 本身就是 StepC-compatible Stage6 出口
2. `70a` 是这个 clean handoff 的第一个 downstream consumer
3. `replace / 70R / 71 / 72 / lambda` 看的也不是“孤立 stage7 recipe”，而是这个 clean handoff 能不能被整条 downstream chain absorb

这也是为什么本文和 `docs/posttrain_pipeline.md` 可以互相补充：

- 前者讲 **family-scoped concrete chain**
- 后者讲 **global boundary promotion**

---

## 5) Manual Stage Map

| Link | Launcher | Config | Input ckpt | Locked recipe |
|---|---|---|---|---|
| `top7 donor -> Stage6 clean-StepC handoff` | `python -m train.posttrain` | `debug_output/_tmp_top7_clean_stage6_stepc_chain_20260412/configs/posttrain_stage6_tailfix_top7_clean_stepc_20260412.json` | `models/cp015_phasecd_tailk_probe_20260331/exp_phase_DirectBranch_v1_d1_cp015_tailk7_rankmix_tw020_corridor_hold_tail15_phasea050_fixedsched_ep014center_control_denseckpt_seed2024_20260401/ckpt_epoch_014.pth` | `epochs=8`, `steps_per_epoch=60`, `lr=3e-4`, `weight_decay=1e-4`, `direct_pose_reinit=true`, canonical split leg terminal |
| `Stage6 clean-StepC handoff -> 70a clean` | `python -m train.posttrain` | `debug_output/_tmp_top7_clean_stage6_stepc_chain_20260412/configs/posttrain_70a_from_top7_stage6_clean_stepc_20260412.json` | `models/__tmp_top7_clean_stage6_stepc_chain_20260412/stage6_stepc_handoff/ckpt_last_lr3e4_e8x60_wd1e4_reinit1_stage6_tailfix_tailk7_stepc_clean_20260412.pth` | `epochs=5`, `steps_per_epoch=60`, `lr=3e-4` |
| `70a clean -> replace clean` | `python -m train.posttrain` | `debug_output/_tmp_top7_clean_stage6_stepc_chain_20260412/configs/posttrain_replace_from_top7_70a_clean_stepc_20260412.json` | `models/__tmp_top7_clean_stage6_stepc_chain_20260412/warmstart_clean/ckpt_last_cp015_tailk7_70a_replace_zerophase_cleanstepc_20260412.pth` | `epochs=3`, `steps_per_epoch=60`, `lr=5e-5` |
| `replace clean -> 70R clean` | `python tools/run_posttrain_nonleg_trunk_ablation.py --trunk-mode full` | `debug_output/_tmp_top7_clean_stage6_stepc_chain_20260412/configs/posttrain_70R_from_top7_replace_clean_stepc_20260412.json` | `models/__tmp_top7_clean_stage6_stepc_chain_20260412/replace_clean/ckpt_last_WalkF_stage7_70b_replace_lowdrift_e3x60_lr5e5_from_cp015_tailk7_70a_cleanstepc_20260412.pth` | `epochs=1`, `steps_per_epoch=180`, full nonleg recovery |
| `70R clean -> 71 clean` | `python -m train.posttrain` | `debug_output/_tmp_top7_clean_stage6_stepc_chain_20260412/configs/posttrain_71_from_top7_70R_clean_stepc_20260412.json` | `models/__tmp_top7_clean_stage6_stepc_chain_20260412/70R_clean/ckpt_last_WalkF_stage7_70R_from_cp015_tailk7_replace_cleanstepc_s180_20260412.pth` | `epochs=3`, `steps_per_epoch=60`, `lr=3e-4` |
| `71 clean -> 72 clean` | `python -m train.posttrain` | `debug_output/_tmp_top7_clean_stage6_stepc_chain_20260412/configs/posttrain_72_from_top7_71_clean_stepc_20260412.json` | `models/__tmp_top7_clean_stage6_stepc_chain_20260412/71_clean/ckpt_last_WalkF_stage7_71_from_top7_70R_cleanstepc_lr3e4_20260412.pth` | `epochs=3`, `steps_per_epoch=60`, `lr=1e-4` |
| `72 clean -> lambda clean` | `python -m train.posttrain` | `debug_output/_tmp_top7_clean_stage6_stepc_chain_20260412/configs/posttrain_lambda_from_top7_72_clean_stepc_20260412.json` | `models/__tmp_top7_clean_stage6_stepc_chain_20260412/72_clean/ckpt_last_WalkF_stage7_72_from_top7_71_cleanstepc_lr1e4_20260412.pth` | `epochs=1`, `steps_per_epoch=200`, `lr=2e-4`, lambda-only closure |

### 5.1 Shared `train.posttrain` template

除 `70R clean` 外，其它 stage 都可以按下面模板展开：

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

### 5.2 `70R clean` special launcher

```bash
PYTHONPATH=. python tools/run_posttrain_nonleg_trunk_ablation.py \
  --config debug_output/_tmp_top7_clean_stage6_stepc_chain_20260412/configs/posttrain_70R_from_top7_replace_clean_stepc_20260412.json \
  --trunk-mode full \
  --out-dir models/__tmp_top7_clean_stage6_stepc_chain_20260412/70R_clean \
  --run-name WalkF_stage7_70R_from_cp015_tailk7_replace_cleanstepc_s180_20260412 \
  --epochs 1 \
  --steps-per-epoch 180 \
  --save-step-ckpts 0,1,5,20,60,180
```

### 5.3 Warmstart caveat

`replace clean` 前同样包含一段 `warmstart_clean`：

- 这不是一个普通训练 stage
- 它是 clean-StepC handoff 和 replace 之间的精确 handoff adapter

因此如果不是在做底层调试，直接用：

- `tools/run_top7_clean_stage6_stepc_chain.py`

---

## 6) Stage-Level Read

### 6.1 `70a clean`

最准确的表述不是“已经完全 clean”，而是：

- 相比 pseudo-StepC 有改善
- 相比 old-cut 仍有 residual mixed signal

所以：

- boundary mismatch 是主因
- 但不是唯一残余来源

### 6.2 `replace clean`

- `replace` 是第一段清楚显示 downstream absorbability 的 stage
- 也是 clean-StepC handoff 开始稳定传递到后续链路的地方

### 6.3 `70R clean`

- `70R` 是 decisive rescue stage
- clean-StepC 并不是“避免 collapse”而已，而是已经稳定优于 `O` 与 `P`

### 6.4 `71 -> 72 -> lambda clean`

- gain survives full downstream continuation
- `lambda` 更像 chain closure，而不是额外引入全新机制意义上的 win

---

## 7) Why This Matters

当前被接受的 high-level causal read 是：

- legacy `stage6` handoff / fragmented boundary contract 是 dominant early drag
- `top7` 确实还带着一些 residual donor / recipe burden
- 但 clean StepC handoff 已经足够把整个 family 拉回可吸收区间

最准确的一句话：

> `top7` 不是天然不可用；它主要是超过了 legacy old-boundary contract 的吸收能力，而 clean StepC handoff 让它重新变成 viable family default。

---

## 8) Relationship to `top3`

最准确的 pairing 仍然是：

- `top3` = anchor/control
- `top7` = expansion family

推荐的文档语气：

- `top3` 提供 anchor
- `top7` 在 clean StepC 下提供 expansion default
- 这不是 winner-take-all donor ideology

见：`docs/posttrain_pipeline_top3_anchor_control.md`

---

## 9) Relationship to Global Canonical

`docs/posttrain_pipeline.md` promote 的是：

- StepC boundary contract itself

而这份文档解释的是：

- 这份 promoted contract 在 `top7` family 里怎么具体落地

所以：

- 本文是 family-scoped
- `docs/posttrain_pipeline.md` 仍然是 global default doc

---

## 10) Caveats

- `N=5 / limited-N`
- raw `70a clean` 仍然不是完全无残差
- 某些 hotspot 仍可能 mixed
- 不要把本文写成“top7 单独定义了 global canonical”
- 也不要再把 `stage6 clean-StepC` 和后面的 `70a/replace/70R/...` 拆成两个互不相关的故事
