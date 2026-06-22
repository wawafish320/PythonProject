# stage6-final posttrain baseline（dropout hidden p=0.10）

这是一个**参考用 posttrain baseline**，打包了 stage6 最终模型权重、解析后的训练配置，以及对应的评估汇总数据，供在其它环境（含 Windows）`git clone` 后直接取用。

- 来源分支：`feat/action-handoff-middle-state-feasibility`
- 依据文档：`docs/analysis/2026-05-10_dropout_branch_summary.md`（2026-05-10 dropout 分支总结）
- 训练 run：`stage6_from0504_strictcfg_dropout_hidden010_20260510_2`（`ckpt_last`）
- dropout 路线：`hidden`，`p=0.10`（stage6 阶段的推荐 baseline，见下文）

## 目录结构

```
baselines/stage6_final/
├── README.md
├── model/
│   ├── stage6_final.pth          # stage6 最终权重（ckpt_last，已剥离 optimizer state）
│   └── stage6_final.meta.json    # sha256 / manifest hash / dropout 设置
├── config/
│   ├── stage6_final_resolved_config.json   # 解析后的 148 项完整训练配置（可迁移的关键产物）
│   └── stage6_final_posttrain_log.json     # 完整 posttrain 日志（config + 逐 step 训练曲线）
└── eval/
    ├── full_chain_abc/           # 全链 ABC 对比评估汇总（doc 第 5 节数字来源）
    │   ├── quick_summary.md
    │   ├── metrics_abc_merged.csv
    │   ├── metrics_long.csv
    │   ├── diffs_long.csv
    │   ├── full_chain_metrics_rows.json
    │   ├── compare_full_chain_abc_with_intermediates.json
    │   └── artifact_overview.csv
    └── ablations/                # 消融评估（doc 第 6/7 节）
        ├── offsplit_71_72_clean_compare_5x5.json
        ├── 70R_p_sweep_partial_compare_5x5.json
        ├── 70R_p001_p002_p005_vs_p003_compare_5x5.json
        └── selected_cases_manifest.json
```

## 模型校验

| 项 | 值 |
|---|---|
| 文件 | `model/stage6_final.pth` |
| sha256 | `bc7209a5463dd93632dcddebbe2856c9c3dbabf6bbb0bc69a4b80e35ea2bc2c8` |
| 大小 | 21,162,382 bytes |
| `resolved_build_manifest_hash` | `6bf22f36981e84a183df8b585c64500057102fb62714ed55601e308852181753` |
| checkpoint 顶层键 | `model`, `posttrain_cfg`, `checkpoint_contract`, `build_cfg`, `fingerprints`, `manifest_summary`, `resolved_build_manifest`, `resolved_build_manifest_hash`, `strict_current_model_build` |

Windows 下 clone 后校验：

```powershell
Get-FileHash -Algorithm SHA256 baselines\stage6_final\model\stage6_final.pth
# 应等于 bc7209a5463dd93632dcddebbe2856c9c3dbabf6bbb0bc69a4b80e35ea2bc2c8
```

## 分阶段 dropout 推荐 baseline（来自 doc 第 4 节）

| stage | dropout route | p |
|---|---:|---:|
| stage6 | hidden | 0.10 |
| 70a | hidden | 0.10 |
| replace | hidden | 0.10 |
| 70R | hidden | 0.03 |
| 71 | off | 0.00 |
| 72 | off | 0.00 |
| lambda | off | 0.00 |

本包提供的是表中 **stage6** 那一行（hidden p=0.10）的最终权重。下游 70a/replace/70R/71/72/lambda 不在本包内。

## 评估口径

- 评估按 5×5：groups = `all_ex_root / leg / nonleg / arm / else`，metrics = `mean / p50 / p90 / p95 / max`。
- 数据集 Walk_F，eval mask `344/434`，deterministic（eval/free-run 不注入 dropout，硬约束）。
- 详细解读见 `docs/analysis/2026-05-10_dropout_branch_summary.md`。先看 `eval/full_chain_abc/quick_summary.md`。

## ⚠️ 重要：模型与评估的 rerun 耦合说明

- 本包的**权重**来自 run `..._dropout_hidden010_20260510_2` 的 `ckpt_last`。
- `eval/full_chain_abc/` 里的 headline 数字（doc 第 5 节的 C 列）来自**同配置的另一次重跑** `..._dropout_hidden010_injecttrue_full_20260510_4`。
- 两次 run 使用**完全相同的配置**（hidden dropout 0.10、8 epochs、同一 `from0504` strict 起点），仅为不同 rerun，权重二进制不完全相同。
- 因此评估汇总应视为**该配置的代表性 baseline 结果**，而非与本包权重字节级一一对应。配置层面完全一致，作为参考 baseline 足够；若需要权重与数字严格配套，请基于本权重重新跑评估。

## 训练配置要点（完整见 `config/stage6_final_resolved_config.json`）

- `run_name = stage6_from0504_strictcfg_dropout_hidden010_20260510_2`
- `posttrain_contacts_pretrain_dropout_injection_mode = hidden`
- `posttrain_contacts_pretrain_dropout_prob = 0.1`
- `epochs = 8`
- 起点 ckpt（基座 donor）：`..._tail_top7_stage6_eval_from0504_strictcfg_20260510_003735/migrated_ckpts/basetrain_donor_strict_contract_r2.pth`
