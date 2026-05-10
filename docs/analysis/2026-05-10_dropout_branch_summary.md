# 当前 dropout 分支总结

日期：2026-05-10

## 1. 背景与问题

当前要解决的问题是 teacher/free-run gap 和 downstream free-run robustness。现有 posttrain / rollout 中的 frozen contact/pretrain route 会把当前 motion step 和 pose history 重新打包成 pretrain contact encoder input，再经过 `frozen_encoder -> frozen_contact_head` 得到 `contacts_in_t`。这个 route 在训练中可能形成 hidden shortcut：模型依赖 frozen route 的稳定 hidden/contact 输入，而不是学到对 free-run drift、contact perturbation 和 pose-history 误差更稳的 downstream basin。

本分支的尝试是只在训练路径对 frozen contact/pretrain route 做 dropout/corruption，增强 route robustness。eval/deploy 必须 deterministic，所以 eval/free-run 与 route contract probe 不能注入 dropout；这是硬约束，不是可调实验项。

## 2. 代码改动概要

以下基于当前工作区 diff 与当前源码行号整理。

- `train/configuration/model_build.py:67` 增加 trainbase contact-pretrain dropout 默认值，`DEFAULT_TRAINBASE_CONTACTS_PRETRAIN_DROPOUT_INJECTION_MODE="off"`、`DEFAULT_TRAINBASE_CONTACTS_PRETRAIN_DROPOUT_PROB=0.0`；`train/configuration/model_build.py:71` 增加 posttrain 对应默认值。
- `train/configuration/model_build.py:814` 在 `resolve_train_trainer_runtime_config` 中把 `trainbase_contacts_pretrain_dropout_injection_mode` 和 `trainbase_contacts_pretrain_dropout_prob` 传入 `resolve_contact_pretrain_runtime`；prob 先经 `[0,1)` range check，见 `train/configuration/model_build.py:831`。
- `train/configuration/model_build.py:912` 在 `resolve_posttrain_trainer_runtime_config` 中解析 `posttrain_contacts_pretrain_dropout_*`，同样经 `[0,1)` range check，见 `train/configuration/model_build.py:929`。
- `train/configuration/norm_spec.py:31` 扩展 `ContactPretrainRuntime`，新增 `dropout_injection_mode: str = "off"` 和 `dropout_prob: float = 0.0`。
- `train/configuration/norm_spec.py:162` 在 `resolve_contact_pretrain_runtime` 统一校验 `dropout_injection_mode` 只能是 `off|encoder_input|hidden`，并校验 `dropout_prob` 必须是 finite float 且在 `[0,1)`；返回值写回 normalized mode/prob，见 `train/configuration/norm_spec.py:204`。
- `train/runtime_attach.py:108` 的 `apply_contacts_pretrain_runtime` 同时绑定 neutral attrs 与 owner-prefixed attrs：`contacts_pretrain_dropout_*` 和 `{owner}_contacts_pretrain_dropout_*`，见 `train/runtime_attach.py:121` 和 `train/runtime_attach.py:127`。
- `train/training_MPL.py:536` 的 `Trainer._predict_pretrain_contacts_from_frozen` 增加 `inject_dropout: bool = False`。该函数要求 runtime attach 后 neutral attrs 完整存在，见 `train/training_MPL.py:561`；同时对 mode/prob fail-fast 校验，见 `train/training_MPL.py:587`。
- `train/training_MPL.py:608` 构造 `encoder_input`，tensor shape 是 `[B,Denc]`，dtype/device 跟 `motion_step_t` 对齐；`encoder_input` dropout 注入点在 frozen encoder 之前，见 `train/training_MPL.py:616`。
- `train/training_MPL.py:620` 在 `torch.no_grad()` 下调用 frozen encoder；`hidden` dropout 注入点在 frozen encoder 后、frozen contact head 前，见 `train/training_MPL.py:622`。`hidden` tensor shape 由 frozen encoder 返回，当前测试覆盖 `[B,1,Denc]`，dtype/device 继承 encoder input。
- `train/training_MPL.py:1106` basetrain rollout 通过 `resolve_rollout_step_inputs(..., inject_dropout=True)` 打开训练路径 dropout。
- `train/training_MPL.py:2633` CLI 增加 `--trainbase_contacts_pretrain_dropout_injection_mode`，choices 为 `off|encoder_input|hidden`；`train/training_MPL.py:2640` 增加 `--trainbase_contacts_pretrain_dropout_prob`。
- `train/rollout_kernel.py:250` 的 `RolloutModelStepRequest` 增加 `inject_dropout: bool = False`。
- `train/rollout_kernel.py:711` 的 `prepare_rollout_contacts_input` 接收 `inject_dropout` 并传给 `_predict_pretrain_contacts_from_frozen`；输入 `motion_t` shape 是 `[B,Dx]`，`pose_hist_t` shape 是 `None` 或 `[B,Dh]`，dtype/device 与 rollout state 对齐。
- `train/rollout_kernel.py:869` 的 `execute_rollout_model_step` 传递 request 上的 `inject_dropout`；`train/rollout_kernel.py:914` 的 `resolve_rollout_step_inputs` 也显式透传。
- `train/eval_utils.py:346` 的 `evaluate_freerun` 显式调用 `_predict_pretrain_contacts_from_frozen(..., inject_dropout=False)`，保证 eval/free-run deterministic。
- `train/posttrain.py:84` 引入 `resolve_contact_pretrain_runtime`，`train/posttrain.py:90` 引入 diagnostics route contract probe。
- `train/posttrain.py:429` 的 `PostTrainConfig` 增加 `posttrain_contacts_pretrain_dropout_injection_mode` 和 `posttrain_contacts_pretrain_dropout_prob`。
- `train/posttrain.py:907` parser/config 读取 `posttrain_contacts_pretrain_dropout_*`；`train/posttrain.py:968` 复用 `norm_spec.resolve_contact_pretrain_runtime` 做统一校验，并把错误名替换成 posttrain owner key，见 `train/posttrain.py:976`。
- `train/posttrain.py:1096` 构造 posttrain rollout request 时写入 `inject_dropout=True`，即 posttrain rollout 训练路径启用 dropout。
- `train/posttrain.py:2199` 在 contact plan enabled 但 `contacts_in_t is None` 时 fail-fast，禁止 cached hidden 或 fallback route。
- `train/posttrain.py:3370` 训练 loop 首个 batch 调用 `verify_posttrain_contact_route_contract`，结果记录到 `trainer._posttrain_route_contract_evidence`。
- `train/posttrain.py:4299` posttrain CLI 增加 `--posttrain_contacts_pretrain_dropout_injection_mode`，choices 为 `off|encoder_input|hidden`；`train/posttrain.py:4307` 增加 `--posttrain_contacts_pretrain_dropout_prob`。
- `train/diagnostics.py:103` 增加 `verify_posttrain_contact_route_contract`。probe 从 `batch["motion"]` 读取 `[B,T,Dx]` tensor，转成 `motion_step_t: [B,Dx]`，dtype=`model_dtype`，device=`trainer.device`，见 `train/diagnostics.py:126`。如果有 `pose_hist`，接受 `[B,T,Dh]` 或 `[B,Dh]` 并转成 `[B,Dh]`，见 `train/diagnostics.py:135`。
- `train/diagnostics.py:162` route probe 固定 `inject_dropout=False`。它通过 hook 记录 frozen encoder/head 的实际输入 shape/dtype/device，见 `train/diagnostics.py:228`；并用 shared builder 重建 encoder input，校验 `encoder_input_max_abs_diff <= 1e-6`，见 `train/diagnostics.py:207` 和 `train/diagnostics.py:220`。
- `tests/train/test_trainer_runtime_config.py:32` 覆盖 trainbase defaults，`tests/train/test_trainer_runtime_config.py:67` 覆盖 trainbase explicit override，`tests/train/test_trainer_runtime_config.py:112` 覆盖 posttrain defaults，`tests/train/test_trainer_runtime_config.py:141` 覆盖 posttrain explicit override，`tests/train/test_trainer_runtime_config.py:166` 覆盖 invalid prob/mode fail-fast。
- `tests/train/test_contacts_pretrain_runtime_attach.py:14` 覆盖 runtime attach 同时写 neutral 和 owner-prefixed dropout attrs。
- `tests/train/test_posttrain_local_runtime_overlay.py:73` 覆盖 posttrain local runtime overlay 保留 dropout runtime；`tests/train/test_posttrain_local_runtime_overlay.py:94` 覆盖 overlay apply 到 trainer 后 owner-prefixed attrs 与 `contacts_pretrain_runtime_attached`。
- `tests/train/test_predict_pretrain_contacts_from_frozen_dropout.py:73` 覆盖 `inject_dropout=False` deterministic route；`tests/train/test_predict_pretrain_contacts_from_frozen_dropout.py:89` 覆盖 `encoder_input` dropout；`tests/train/test_predict_pretrain_contacts_from_frozen_dropout.py:108` 覆盖 `hidden` dropout；`tests/train/test_predict_pretrain_contacts_from_frozen_dropout.py:128` 覆盖 invalid mode/prob fail-fast。测试 tensor 为 `motion: [2,4]`、`pose_hist: [2,2]`、dtype=`torch.float32`、device=`cpu`。

## 3. 训练/评估语义

- train/basetrain rollout：`inject_dropout=True`。
- posttrain rollout：`inject_dropout=True`。
- eval/free-run：`inject_dropout=False`。
- route contract probe：`inject_dropout=False`。
- `dropout_injection_mode=off`：不注入 corruption，即使调用方传 `inject_dropout=True`。
- `dropout_injection_mode=encoder_input`：对 frozen encoder 前的 `encoder_input: [B,Denc]` 注入 dropout，dtype/device 与当前 rollout tensor 一致。
- `dropout_injection_mode=hidden`：对 frozen encoder 输出的 `hidden` 注入 dropout，再送 frozen contact head；dtype/device 继承 frozen encoder 输出。
- `dropout_prob` 范围是 `[0,1)`，必须 finite。
- eval/probe deterministic 是硬约束。任何 eval/free-run/route probe 注入 dropout 都会污染 deploy contract 与实验归因。

## 4. 当前分阶段 dropout 基线配置

当前推荐 baseline：

| stage | dropout route | p |
|---|---:|---:|
| stage6 | hidden | 0.10 |
| 70a | hidden | 0.10 |
| replace | hidden | 0.10 |
| 70R | hidden | 0.03 |
| 71 | off | 0.00 |
| 72 | off | 0.00 |
| lambda | off | 0.00 |

说明：

- 原始 C 是全链 hidden dropout p=0.10。
- offsplit 证明 71+ 关闭更适合 leg refinement。
- 70R p sweep 的部分结果证明 70R 不能直接 p=0；p=0.03 是当前最强的阶段性结果。
- 这个 baseline 是当前阶段性推荐，不是最终定论。

## 5. 实验依据一：原始 C hidden dropout=0.1 全链

artifact：

- `debug_output/_tmp_stage6_lambda_compare_all_20260510_dropout_hidden010_70R180_last/quick_summary.md`
- `debug_output/_tmp_stage6_lambda_compare_all_20260510_dropout_hidden010_70R180_last/metrics_abc_merged.csv`
- `debug_output/_tmp_stage6_lambda_compare_all_20260510_dropout_hidden010_70R180_last/diffs_long.csv`

实验定义：

- A 是调整训练/评估输入后的 baseline。
- B 是原始 0504 reference。
- C 是 hidden dropout=0.1。
- eval mask 全部是 `344/434`，见 `quick_summary.md` 的 stage blocks。

stage6/70a 指标明显退化：

- `stage6/step_000360`：C 相对 A 的 `all_ex_root mean/p90/p95/max` 为 `+0.056689 / +0.127148 / +0.246289 / +1.530520`；`leg mean/p90/p95/max` 为 `+0.152169 / +0.425825 / +0.567923 / +0.669890`。
- `70a/last`：C 相对 A 的 `all_ex_root mean/p90/p95` 为 `+0.074943 / +0.195180 / +0.338740`；`leg mean/p90/p95` 为 `+0.199009 / +0.552896 / +0.676781`；`nonleg mean` 为 `+0.048118`，`arm mean` 为 `+0.056523`。

downstream 后开始反超：

- `70R/last`：C 相对 A 的 `all_ex_root mean/p90/p95` 为 `-0.008948 / -0.025402 / -0.007561`；`nonleg mean/p90/p95` 为 `-0.007560 / -0.030451 / -0.039420`；`arm mean/p90/p95` 为 `-0.011347 / -0.044902 / -0.043998`。
- `72/step_000150`：`C_72_step_000150` 对 `A_72_last` 在 5 groups × 5 metrics 中赢 `19/25`。未赢的是 `all_ex_root.max +0.025204`、`nonleg.max +0.025204`、`arm.max +0.025204`、`else.mean +0.001391`、`else.p50 +0.004531`、`else.p90 +0.008176`。
- `C_72_step_000150` 的核心值：`all_ex_root mean/p90/p95 = 0.091486 / 0.208042 / 0.299739`，`leg mean/p90/p95/max = 0.169520 / 0.335453 / 0.429920 / 1.358040`，`nonleg mean/p90/p95 = 0.074613 / 0.175399 / 0.246177`，`arm mean/p90/p95 = 0.084513 / 0.203226 / 0.283779`。

C 从 `72/step_000150` 到 `72/last` 出现 leg 回退：

- `C_72_last - C_72_step_000150`：`all_ex_root mean/p90/p95/max = +0.003686 / +0.020722 / +0.023827 / +0.041970`。
- `C_72_last - C_72_step_000150`：`leg mean/p50/p90/p95/max = +0.020731 / +0.016287 / +0.050330 / +0.071657 / +0.121346`。
- nonleg/arm/else 在该对比中为 `0.000000`，说明回退主要来自 leg refinement 继续训练后的 tail/leg 变差。

解释：dropout 不是让当前 stage 指标更好；stage6/70a 的局部指标变差与 downstream 更稳并不矛盾。当前效果更像把模型推离 frozen route shortcut，使后续 replace/70R/72 进入更好的 free-run basin。

## 6. 实验依据二：offsplit 71/72 关闭 dropout

artifact：

- `debug_output/_tmp_stage6_to_lambda_dropout_hidden010_offsplit_20260510/offsplit_eval_compare_5x5.json`

实验定义：

- `C_off_from_72`：72 起关闭 dropout。
- `C_off_from_71`：71 起关闭 dropout。

结果：

- `C_off_from_71_72_last` 对 `A_72_last` 是 `19/25` 更优；对 `C_72_step_000150` 是 `6/25` 更优、`16/25` 持平、`3/25` 更差。
- `C_off_from_71_72_last` 相对 `C_72_step_000150` 的 leg 改善：`leg mean -0.007989`，`leg p90 -0.016808`，`leg p95 -0.041631`，`leg max -0.193673`。`leg p50` 轻微变差 `+0.001200`，因此后续仍要单独看 p50。
- `C_off_from_71_72_last` 的 leg 核心值为 `mean/p90/p95/max = 0.161530 / 0.318645 / 0.388289 / 1.164367`。
- nonleg/arm 相对 `C_72_step_000150` 没有退化：`nonleg mean/p90/p95` delta 全部 `+0.000000`，`arm mean/p90/p95` delta 全部 `+0.000000`，保留了 C 的 nonleg/arm 优势。
- `C_off_from_71_lambda_last - C_off_from_71_72_last` 在 5×5 全部是 `+0.000000`，即 lambda/last 与 72/last 基本一致。

解释：71/72 是 leg refinement，后期继续 route dropout 会干扰 leg tail/refinement；从 71 开始 clean 更合适。

## 7. 实验依据三：70R p sweep 部分结果

artifact：

- `debug_output/_tmp_stage6_dropout_followups_20260510/partial_compare_5x5.json`

实验定义：

- `g1_p000`：70R p=0。
- `g1_p003`：70R p=0.03。

70R p=0：

- `g1_p000_72_last` 对 `A_72_last` 是 `0/25` 更优、`25/25` 持平，基本回到 A。
- `g1_p000_72_step_000150` 对 `A_72_last` 是 `8/25` 更优、`16/25` 持平、`1/25` 更差；更差项只有 `all_ex_root.p50 +0.000010`。
- `g1_p000_72_step_000150` 的 leg 稍好：相对 `A_72_last`，`leg mean/p50/p90/p95/max = -0.009825 / -0.003976 / -0.021985 / -0.013079 / -0.117442`。
- 但 `g1_p000_72_step_000150` 的 nonleg/arm/else 全部回到 A：nonleg、arm、else 的 5×5 delta 均为 `+0.000000`。这说明 p=0 会丢掉 C 的 nonleg/arm robustness。

70R p=0.03：

- `g1_p003_72_step_000150` 对 `A_72_last` 是 `25/25` 全优；对 `C_72_step_000150` 是 `24/25` 更优，仅 `all_ex_root.p90 +0.000906`。
- `g1_p003_72_last` 对 `A_72_last` 是 `25/25` 全优；对 `C_72_step_000150` 是 `24/25` 更优，仅 `leg.max +0.016555`。
- `g1_p003_72_last` 核心数字：
  - `all_ex_root mean/p90/p95 = 0.085410 / 0.203379 / 0.280491`
  - `leg mean/p90/p95/max = 0.156929 / 0.296849 / 0.386820 / 1.374595`
  - `nonleg mean/p90/p95 = 0.069947 / 0.167618 / 0.234126`
  - `arm mean/p90/p95 = 0.079185 / 0.194405 / 0.274976`
  - `else mean/p90/p95/max = 0.048111 / 0.110665 / 0.140741 / 0.468583`
- `g1_p003_72_last - C_72_step_000150`：`all_ex_root mean/p90/p95/max = -0.006075 / -0.004663 / -0.019248 / -0.034124`；`leg mean/p90/p95/max = -0.012590 / -0.038604 / -0.043101 / +0.016555`；`nonleg mean/p90/p95 = -0.004667 / -0.007781 / -0.012051`；`arm mean/p90/p95 = -0.005328 / -0.008821 / -0.008804`。

解释：70R 仍需要弱 dropout 维持 nonleg/arm robustness，不能直接 off；但 p=0.03 明显优于全强度 p=0.10，更像合理的 bridge regularization。

## 8. 机制解释

- stage6/70a 指标退化但 downstream 更好并不矛盾。dropout 在前段引入 route corruption，会提高当前 stage 拟合难度，但也减少模型对 frozen contact/pretrain route 的 shortcut 依赖。
- 当前 dropout 的作用更像 shortcut breaking / route robustness regularizer，而不是直接优化当前 stage 的 teacher/free-run metric。
- stage6/70a/replace 使用 p=0.10，是为了扩大输入/hidden route 覆盖，让模型在较早阶段见到更宽的 frozen-route perturbation。
- 70R p=0 会回到 A，说明 70R 仍需要 dropout 维持 nonleg/arm robustness。
- 70R p=0.03 优于 p=0.10，说明 70R 应退火，而不是继续全强度 regularization。
- 71+ clean 让 leg tail/refinement 更好，说明后期 dropout 会干扰 leg refinement。
- 暂不优先 trajectory scheduled sampling / dimension scheduled sampling / professor forcing，因为它们会和 memory、carry、pose history 耦合，归因更差；当前 dropout schedule 仍有明确可优化空间。

## 9. 评估标准

当前分支所有实验按以下 5×5 评估：

- groups = `all_ex_root`, `leg`, `nonleg`, `arm`, `else`
- metrics = `mean`, `p50`, `p90`, `p95`, `max`

不能只看 `all_ex_root mean`。当前结果里 `else mean/p50/p90` 可能和 tail 指标方向不一致，例如 `C_72_step_000150` 对 `A_72_last` 的 `else mean/p50/p90` 是 `+0.001391 / +0.004531 / +0.008176`，但 `else p95/max` 是 `-0.003996 / -0.029493`。

推荐主判据：

- 对 `A_72_last` 尽量 `25/25` 全优。
- 对 `C_72_step_000150` 尽量 `>=24/25`。
- `leg mean/p90/p95/max` 不能退化。
- `nonleg/arm mean/p90/p95` 不能丢掉 C 优势。
- `else mean/p50/p90` 需要单独记录，因为当前它可能和 tail 指标方向不一致。

## 10. 已验证

实际运行结果：

- `python3 -m py_compile train/posttrain.py train/diagnostics.py`：通过，exit code 0。
- `python3 -m pytest tests/train/test_posttrain_local_runtime_overlay.py tests/train/test_predict_pretrain_contacts_from_frozen_dropout.py tests/train/test_contacts_pretrain_runtime_attach.py tests/train/test_trainer_runtime_config.py`：未运行成功，当前 Python 3.12 环境缺少 pytest，错误是 `No module named pytest`。这是测试环境缺口，未显示与本次 dropout 改动耦合。
- `python3 -m unittest tests.train.test_posttrain_local_runtime_overlay tests.train.test_predict_pretrain_contacts_from_frozen_dropout tests.train.test_contacts_pretrain_runtime_attach tests.train.test_trainer_runtime_config`：通过，`Ran 24 tests in 0.006s`，`OK`。附带 warning：`tqdm not found`，不影响这些 unittest。

覆盖到的行为：

- posttrain local runtime overlay test：覆盖 `ContactPretrainRuntime` 中 dropout mode/prob 通过 overlay 进入 trainer。
- `predict_pretrain_contacts_from_frozen_dropout` test：覆盖 `inject_dropout=False` deterministic route、`encoder_input` 注入、`hidden` 注入、invalid mode/prob fail-fast。测试 tensor 是 `motion: [2,4]`、`pose_hist: [2,2]`、dtype=`torch.float32`、device=`cpu`。
- runtime attach test：覆盖 neutral attrs 与 owner-prefixed attrs 双写。
- trainer runtime config test：覆盖 trainbase/posttrain defaults、explicit override、invalid dropout prob/mode。

## 11. 未完成与风险

- 70R p sweep 当前只确认 p=0 和 p=0.03，p=0.01 / 0.02 / 0.05 尚未补齐。
- 当前实验依赖 Walk_F 和 eval mask `344/434`。
- lambda 当前没有额外收益：offsplit 中 `C_off_from_71_lambda_last - C_off_from_71_72_last` 全部 `+0.000000`，后续不应优先优化 lambda。
- `g1_p003` 是当前阶段性强基线，但还需要更多 p 和 possibly replace schedule 复核。
- 如果迁移到其他 motion/action，dropout schedule 可能需要重调。
- pytest 环境当前不可用；虽然 unittest 覆盖了核心行为，但 CI/pytest runner 下仍需补一次同命令验证。
