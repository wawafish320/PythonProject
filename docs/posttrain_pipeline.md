# Posttrain Pipeline (Global Canonical / StepC Boundary)

> Last updated: 2026-04-13  
> Status: current global canonical posttrain flow  
> Caveat: `N=5 / limited-N`

这份文档只定义当前 default mental model：

- `Stage6` 的正常出口是 `StepC handoff`
- `70a -> replace -> 70R -> 71 -> 72 -> lambda` 是消费这个 handoff 的 downstream chain

---

## 1) TL;DR

当前默认的全局 canonical 不是“某个 donor family”，而是 **StepC unified-leg-terminal boundary contract**。

最清楚的写法是两层：

- boundary outlet: `Stage6 -> StepC handoff`
- downstream continuation: `StepC handoff -> 70a -> replace -> 70R -> 71(lr=3e-4) -> 72(lr=1e-4) -> lambda`

合起来就是：

- `Stage6-StepC handoff -> 70a -> replace -> 70R -> 71(lr=3e-4) -> 72(lr=1e-4) -> lambda`

关键语义：

- `Stage6` 的正常/current 出口已经是 StepC-compatible
- `70a` 是第一个 downstream consumer，不是 StepC repair stage
- 证据看的是这个 handoff 能否被整条 downstream chain absorb

---

## 2) Source of Truth

### 2.1 Canonical decision artifacts

- `debug_output/_tmp_stage6_stepc_canonical_chain_20260412/summary.md`
- `debug_output/_tmp_stage6_stepc_canonical_chain_20260412/decision.md`
- `debug_output/_tmp_stage6_stepc_70r_to_lambda_20260412/summary.md`
- `debug_output/_tmp_stage6_stepc_70r_to_lambda_20260412/decision.md`

### 2.2 Canonical entry scripts

- phase-1 (`70a -> replace -> 70R`): `tools/run_stage6_stepc_canonical_chain.py`
- phase-2 (`70R -> 71 -> 72 -> lambda`): `tools/run_stage6_stepc_70r_to_lambda.py`

### 2.3 Related interpretation docs

- legacy old-boundary control: `docs/posttrain_pipeline_legacy_old_boundary.md`
- top3 anchor/control: `docs/posttrain_pipeline_top3_anchor_control.md`
- top7 family default under clean StepC: `docs/posttrain_pipeline_top7_clean_stepc.md`

---

## 3) Preferred Reproduction

### 3.1 One-command entrypoints

优先使用 repo 内已经固定好的 orchestration script，而不是手抄零散阶段。

```bash
PYTHONPATH=. python tools/run_stage6_stepc_canonical_chain.py
PYTHONPATH=. python tools/run_stage6_stepc_70r_to_lambda.py
```

说明：

- 第一个脚本固定 `StepC handoff -> 70a -> replace -> 70R`
- 第二个脚本固定 `70R -> 71 -> 72 -> lambda`
- 第二个脚本默认读取第一个脚本产出的 `70R_stepc` ckpt

### 3.2 Locked runtime contract

这条 canonical 绑定的是下面这套 runtime / reporting contract：

- contacts source: `pretrain_contact`
- clamp: `1.0`
- encoder bundle: `models/motion_encoder_equiv_20260317.pt.best.pt`
- affine stats: `debug_output/_tmp_phaseb_affine_20260304/affine_fit_mix08/affine_stats.json`
- eval contract: `model-source`
- Step A gate: necessary-but-not-sufficient
- Step B' primary: `all_ex_root_mean`
- tie-break1: `all_ex_root_p95`
- tie-break2: `leg_mean`
- hard reject: fixed-incumbent `nonleg_p95` threshold

**注意**：这些不是“随便换掉也算同一条 canonical chain”的可选项。

---

## 4) Stage6 Normal Outlet = StepC Handoff

当前 posttrain 入口不要拆成“Stage6 一段、Stage7 另一段”。正确 mental model 是：

1. `Stage6` 产出 StepC-compatible boundary。
2. `70a` 从这个 boundary 开始做 downstream continuation。
3. `replace / 70R / 71 / 72 / lambda` 继续验证这个 boundary 是否能被整条链吸收。

所以文档中的主语应当始终是：

- `Stage6-StepC handoff`

`70a` 只是在消费这个 handoff，并把它继续传给 `replace / 70R / 71 / 72 / lambda`。

April 12 canonical audit 里，第一段为了和 historical old-boundary artifact 做严格同源比较，使用了 legacy `Stage6` ckpt，并在 `70a` load 点执行 StepC-compatible `partial_load + tensor_upgrade`。这只是复现实验的 artifact materialization；概念上仍然写作 `Stage6-StepC handoff -> 70a`。

---

## 5) Manual Stage Map

如果你不走 orchestration script，而是要人工逐段复跑，下面这张表是当前 canonical 的最小 runbook。

| Link | Launcher | Config | Input ckpt | Locked recipe |
|---|---|---|---|---|
| `Stage6-StepC handoff -> 70a` | `python -m train.posttrain` | `debug_output/_tmp_stage6_stepc_canonical_chain_20260412/configs/posttrain_70a_fromfresh_stepc_20260412.json` | `models/__tmp_posttrain_pipeline_from_bestfree_20260317/stage6/ckpt_last_WalkF_stage6_fromfresh_20260317.pth` | `epochs=5`, `steps_per_epoch=60`, `lr=1e-3`, April 12 exact audit materializes StepC via `partial_load + tensor_upgrade`, `direct_pose_stepc_unified_leg_terminal=true` |
| `70a -> replace` | `python -m train.posttrain` | `debug_output/_tmp_stage6_stepc_canonical_chain_20260412/configs/posttrain_70b_replace_lowdrift_fromfresh_stepc_20260412.json` | `models/__tmp_stage6_stepc_canonical_chain_20260412/warmstart/ckpt_last_70a_replace_zerophase_stepc_20260412.pth` | `epochs=1`, `steps_per_epoch=60`, `lr=3e-4`, exact canonical zerophase warmstart delta |
| `replace -> 70R` | `python tools/run_posttrain_nonleg_trunk_ablation.py --trunk-mode full` | `debug_output/_tmp_stage6_stepc_canonical_chain_20260412/configs/posttrain_70R_fromfresh_stepc_20260412.json` | `models/__tmp_stage6_stepc_canonical_chain_20260412/replace_stepc/ckpt_last_WalkF_stage7_70b_replace_lowdrift_fromfresh_stepc_20260412.pth` | `epochs=1`, `steps_per_epoch=180`, save steps `0,1,5,20,60,180` |
| `70R -> 71` | `python -m train.posttrain` | `debug_output/_tmp_stage6_stepc_70r_to_lambda_20260412/configs/posttrain_71_from_70R_stepc_lr3e4_20260412.json` | `models/__tmp_stage6_stepc_canonical_chain_20260412/70R_stepc/ckpt_last_WalkF_stage7_70R_fromfresh_stepc_s180_20260412.pth` | `epochs=3`, `steps_per_epoch=60`, `lr=3e-4`, leg-only continuation |
| `71 -> 72` | `python -m train.posttrain` | `debug_output/_tmp_stage6_stepc_70r_to_lambda_20260412/configs/posttrain_72_from_71_stepc_lr1e4_20260412.json` | `models/__tmp_stage6_stepc_70r_to_lambda_20260412/71_stepc/ckpt_last_WalkF_stage7_71_from_70R_stepc_lr3e4_20260412.pth` | `epochs=3`, `steps_per_epoch=60`, `lr=1e-4` |
| `72 -> lambda` | `python -m train.posttrain` | `debug_output/_tmp_stage6_stepc_70r_to_lambda_20260412/configs/posttrain_lambda_from_72_stepc_20260412.json` | `models/__tmp_stage6_stepc_70r_to_lambda_20260412/72_stepc/ckpt_last_WalkF_stage7_72_from_71_stepc_lr1e4_20260412.pth` | `epochs=1`, `steps_per_epoch=200`, `lr=2e-4`, `train_lambda_head=true`, `train_direct_pose=false` |

### 5.1 Shared manual command template

除 `70R` 外，其它 stage 都是同一套 `train.posttrain` 调用面：

```bash
PYTHONPATH=. python -m train.posttrain \
  --config <config_json> \
  --ckpt_in <input_ckpt> \
  --out_dir <out_dir> \
  --run_name <run_name> \
  --posttrain_contacts_source pretrain_contact \
  --posttrain_contacts_pretrain_clamp 1.0 \
  --encoder_bundle models/motion_encoder_equiv_20260317.pt.best.pt \
  --posttrain_contacts_pretrain_affine_stats \
    debug_output/_tmp_phaseb_affine_20260304/affine_fit_mix08/affine_stats.json
```

### 5.2 `70R` special launcher

```bash
PYTHONPATH=. python tools/run_posttrain_nonleg_trunk_ablation.py \
  --config debug_output/_tmp_stage6_stepc_canonical_chain_20260412/configs/posttrain_70R_fromfresh_stepc_20260412.json \
  --trunk-mode full \
  --out-dir models/__tmp_stage6_stepc_canonical_chain_20260412/70R_stepc \
  --run-name WalkF_stage7_70R_fromfresh_stepc_s180_20260412 \
  --epochs 1 \
  --steps-per-epoch 180 \
  --save-step-ckpts 0,1,5,20,60,180
```

### 5.3 Warmstart caveat

`replace` 前的 warmstart 不是“普通 config 能自己表达出来”的单独训练阶段。  
它要求：

- 先拿到从 `Stage6-StepC handoff` 训练出的 `70a`
- 再把 **canonical 70a->replace zerophase delta** 精确贴回这个 handoff 分支

所以如果你不是在做底层调试，**不要手工跳过这一步**；直接用
`tools/run_stage6_stepc_canonical_chain.py`。

---

## 6) Why This Is Canonical

当前被接受的证据链是：

1. `Stage6-StepC handoff -> 70a` 已经优于 old-boundary `Stage6 -> 70a`
2. `replace` 保留了这个 handoff gain
3. `70R` 继续优于 canonical old-cut `70R`
4. `71 / 72 / lambda` 没有把 StepC gain 洗掉，只是逐步衰减
5. final `lambda` 仍然在锁定 Step B' policy 下优于 old-cut `lambda`

最准确的一句话：

> StepC unified-leg-terminal 现在是默认 global posttrain boundary contract，因为它在真实 canonical downstream chain 上一直保留到 final `lambda`。

---

## 7) Relationship to Legacy / Top3 / Top7

### 7.1 Legacy old-boundary chain

现在只保留为：

- historical reproduction
- legacy control
- comparison baseline

见：`docs/posttrain_pipeline_legacy_old_boundary.md`

### 7.2 Top3

`top3` 现在应当写成：

- donor anchor/control
- historical old-boundary compatibility reference

而不是：

- universal natural optimum

见：`docs/posttrain_pipeline_top3_anchor_control.md`

### 7.3 Top7

`top7` 现在应当写成：

- clean-StepC 下的 expansion family default
- family-scoped corroborating evidence

它支持同一个 boundary-causality 故事，但它本身不是全局 canonical 的唯一来源。

见：`docs/posttrain_pipeline_top7_clean_stepc.md`

---

## 8) Caveats

- `N=5 / limited-N`
- promotion 绑定在 locked Step B' policy 上
- 某些 local hotspot 仍然 mixed
- 某些 downstream stage 仍可能有小的 `leg_p95` tradeoff

不要再写：

- `stage6 修完了，stage7 再看`
- `top3 天然最优`
- `top7 太 aggressive`

更准确的写法是：

- real bottleneck 在 `stage6 -> downstream` 的 boundary contract
- StepC 定义的是 **真实 handoff interface**
- donor-family residual burden 仍可存在，但不推翻 canonical promotion
