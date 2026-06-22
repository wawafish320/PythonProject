# Walk_F posttrain 全链 baseline（basetrain → stage6 → stage7 → lambda）

完整、连贯、可直接使用的参考 posttrain baseline，覆盖整条流水线：**basetrain donor → stage6 → stage7(70a/replace/70R/71/72) → lambda(final)**。供其它环境（含 Windows）`git clone` 后直接取用。

- 来源 run：`stage6_to_lambda_dropout_hidden010_injecttrue_full_20260510_4`（同一次端到端跑，lineage 一致）
- 依据文档：`docs/analysis/2026-05-10_dropout_branch_summary.md`
- 数据集：Walk_F，eval mask `344/434`，deterministic eval（eval/free-run 不注入 dropout）
- **dropout schedule：全链 hidden p=0.10**（即 doc 第 5 节的 “C” 链）

## 为什么是全链 0.10（而不是 doc 第 4 节那张推荐表）

doc 第 4 节的推荐表（stage6/70a/replace=0.10、70R=0.03、71/72/lambda=off）是**逐阶段最优拼出来的合成结果**，并不存在一条端到端、连贯训练到 lambda 的真实链（例如 70R=0.03 的 followup 没有 lambda）。

本包选用**全链 hidden 0.10（C）**，因为它是唯一满足以下全部条件的链：

1. 单次端到端跑、lineage 内部一致（每个 stage 都从上一 stage 的 checkpoint 训练而来）；
2. 跑到了 lambda（final）；
3. **评估与权重字节级匹配**——`eval/full_chain_abc/` 的 C 列就是用本包这条链评估出来的。

如需逐阶段最优 schedule（70R=0.03、71+ off），见 `eval/ablations/` 里的消融数据，可基于本链权重重新跑对应 schedule。

## 目录结构

```
baselines/walk_f_posttrain_chain/
├── README.md
├── model/                          # 8 个 checkpoint，按链顺序编号
│   ├── 00_basetrain_donor.pth      # stage6 的起点（from0504 strict, direct-pose reinit donor）
│   ├── 01_stage6.pth
│   ├── 02_stage7_70a.pth
│   ├── 03_stage7_replace.pth
│   ├── 04_stage7_70R.pth
│   ├── 05_stage7_71.pth
│   ├── 06_stage7_72.pth
│   ├── 07_stage7_lambda_final.pth  # 最终模型
│   └── MANIFEST.sha256.json        # 每个 ckpt 的 sha256 / 大小 / dropout 设置 / manifest hash
├── config/                         # 对应每个 stage 的训练配置（与权重同一次 run）
│   ├── 00_basetrain_donor_report.json
│   ├── 01_stage6.json
│   ├── 02_stage7_70a.json
│   ├── 03_stage7_replace.json
│   ├── 04_stage7_70R.json
│   ├── 05_stage7_71.json
│   ├── 06_stage7_72.json
│   └── 07_stage7_lambda.json
└── eval/
    ├── full_chain_abc/             # 全链 ABC 对比评估（A=新baseline, B=0504 ref, C=本链）
    │   ├── quick_summary.md        ← 先看这个
    │   ├── metrics_abc_merged.csv
    │   ├── metrics_long.csv
    │   ├── diffs_long.csv
    │   ├── full_chain_metrics_rows.json
    │   ├── compare_full_chain_abc_with_intermediates.json
    │   └── artifact_overview.csv
    └── ablations/                  # 逐阶段消融（doc 第 6/7 节）
        ├── offsplit_71_72_clean_compare_5x5.json
        ├── 70R_p_sweep_partial_compare_5x5.json
        ├── 70R_p001_p002_p005_vs_p003_compare_5x5.json
        └── selected_cases_manifest.json
```

## 链与 dropout schedule（本包实际值，已从 checkpoint 内嵌 posttrain_cfg 校验）

| # | stage | 文件 | dropout route | p |
|---|---|---|---|---:|
| 00 | basetrain donor | `00_basetrain_donor.pth` | (起点, 无 posttrain) | — |
| 01 | stage6 | `01_stage6.pth` | hidden | 0.10 |
| 02 | stage7 70a | `02_stage7_70a.pth` | hidden | 0.10 |
| 03 | stage7 replace | `03_stage7_replace.pth` | hidden | 0.10 |
| 04 | stage7 70R | `04_stage7_70R.pth` | hidden | 0.10 |
| 05 | stage7 71 | `05_stage7_71.pth` | hidden | 0.10 |
| 06 | stage7 72 | `06_stage7_72.pth` | hidden | 0.10 |
| 07 | stage7 lambda (final) | `07_stage7_lambda_final.pth` | hidden | 0.10 |

完整 sha256 见 `model/MANIFEST.sha256.json`。

## 评估口径

- 5×5：groups = `all_ex_root / leg / nonleg / arm / else`，metrics = `mean / p50 / p90 / p95 / max`。
- ABC 含义：A=调整输入后的 baseline，B=0504 reference，**C=本链（hidden 0.10）**。
- 不能只看 `all_ex_root mean`；`else` 指标可能与 tail 指标方向不一致。详见 doc 第 9 节。
- 先读 `eval/full_chain_abc/quick_summary.md`。

## Windows 取用与校验

```powershell
git clone <repo>
git checkout feat/action-handoff-middle-state-feasibility
# 全部在 baselines\walk_f_posttrain_chain\ 下
# 校验最终模型：
Get-FileHash -Algorithm SHA256 baselines\walk_f_posttrain_chain\model\07_stage7_lambda_final.pth
# 与 model\MANIFEST.sha256.json 中对应条目比对
```

`.gitattributes` 已将 `*.pth` 标为 binary，避免 Windows checkout 时 line-ending 转换损坏权重。

## 备注

- 这些源文件原本在 `.gitignore` 排除的 `models/`、`debug_output/` 下（仓库约定不提交实验产物）。本包是有意复制到受跟踪的 `baselines/` 目录，未改动 `.gitignore`，也未改动原始文件。
- 8 个 checkpoint 共约 164MB，普通 git 提交（未用 LFS），以便 Windows 裸 clone 直接获取。
