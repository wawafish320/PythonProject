# 补丁：推荐 dropout schedule 配置（70R=0.03、71/72/lambda 关）

这是 doc 第 4 节**推荐 schedule** 的配置补丁，供后续**重新跑**。本目录**只含配置，不含权重**——因为这条精确链从未被端到端训练过（见下文 lineage 说明）。

| stage | dropout route | p | 相对 C 全链 |
|---|---|---:|---|
| stage6 | hidden | 0.10 | 同 C |
| 70a | hidden | 0.10 | 同 C |
| replace | hidden | 0.10 | 同 C |
| **70R** | hidden | **0.03** | ← 0.10 改 0.03 |
| **71** | **off** | **0.00** | ← 关 |
| **72** | **off** | **0.00** | ← 关 |
| **lambda** | **off** | **0.00** | ← 关 |

## 配置如何确认的（重要）

- 基底：上级目录 `../config/` 的 **C 全链配置**（lineage 自洽、真实端到端跑过）。
- 只覆盖 dropout 三个字段：`posttrain_contacts_pretrain_dropout_injection_mode`、`posttrain_contacts_pretrain_dropout_prob`，以及本流水线配置里的镜像字段 `dropout`。其余 148 项超参一律不动。
- **双向交叉验证**（见 `schedule.json` 的 `field_overrides`）：
  1. 与 C 链配置 diff：实质差异**仅** dropout 字段；
  2. 与产出 doc 结论的真实实验配置 diff（70R 取 `g1_70r_p003`，71/72/lambda 取 offsplit `C_off_from_71`）：dropout 字段**逐一相等**，且**无任何其它超参差异**（只剩 ckpt_in/run_name 这类路径）。

也就是说本补丁 = 「C 的连贯 lineage」+「真实实验验证过的 dropout 取值」。

## ⚠️ 为什么必须重跑（lineage 说明）

推荐表是**逐阶段最优拼出来的合成结论**，并不存在一条端到端训练到 lambda 的真实链：

- `g1_70r_p003` 做了 70R=0.03，但**没有 lambda**，且其下游 71/72 并非 off；
- offsplit `C_off_from_71` 做了 71/72/lambda=off，但其 71 是**从 70R=0.10** 训练来的（`ckpt_in` 指向 injecttrue_full 的 0.10 70R），**不是** 0.03 的 70R。

所以要得到精确链，必须按下列 lineage **重新训练** 70R→lambda：

```
00_basetrain_donor
  └─ 01_stage6      (hidden 0.10)   ┐ 这三段权重可直接复用 ../model/ 里的 C 链产物
     └─ 02_70a      (hidden 0.10)   │ （schedule 与 C 完全相同），无需重训
        └─ 03_replace (hidden 0.10) ┘
           └─ 04_70R   (hidden 0.03)  ← 从 03_replace 重训
              └─ 05_71  (off)         ← 从【新的 0.03】70R 重训（关键：不是 0.10 的）
                 └─ 06_72 (off)       ← 从新的 71 重训
                    └─ 07_lambda (off)← 从新的 72 重训
```

## 重跑时要改的路径

每个配置仍保留着 C 链（injecttrue_full）的 `ckpt_in` / handoff 路径。重跑时把每个 stage 的输入指向**上一 stage 的本次新产物**，尤其：

- `04_stage7_70R.json` 的 `ckpt_in` → 复用 C 的 `03_stage7_replace` 输出（replace 阶段 schedule 与 C 相同）。
- `05_stage7_71.json` 的 `ckpt_in` → **本次新训的 70R(0.03) 输出**（务必不要用 0.10 的 70R）。
- `06`、`07` 依次指向上一新产物。
- `run_name` / 输出目录按你的新 run 命名。

## 评估对照

复用上级 `../eval/ablations/` 的逐阶段消融证据：
- `70R_p_sweep_partial_compare_5x5.json`、`70R_p001_p002_p005_vs_p003_compare_5x5.json`：70R 各 p 对比，0.03 为当前最强。
- `offsplit_71_72_clean_compare_5x5.json`：71/72 关闭的收益。

跑完后建议按 doc 第 9 节的 5×5 口径，与 `../eval/full_chain_abc/`（C 链）对比。
