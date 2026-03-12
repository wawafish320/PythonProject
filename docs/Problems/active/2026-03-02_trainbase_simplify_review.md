# 2026-03-02 train(base) 流程简化审核记录（Stage6→Stage7）

Last updated: 2026-03-02

## 1) 审核结论（先给结论）

当前口径切换为：**先评估与 `inc/direct + λ fusion` 的强弱耦合，再决定拆分/移除顺序**。

当前工作假设（基于 2026-03-02 实测）：

1. `contact_phase_state`（原 Step C）是**弱耦合候选**
2. `contact_meas provider`（原 Step B）是**弱耦合候选**
3. `event_clock`（原 Step A，最后做）是**相对强耦合候选**，需要先做观测归因

关键补充（修订）：**在 active whitelist 的 8 个 config 对应 `ckpt_in` 上，这三块不是“未激活”，而是“非训练目标头但仍参与 rollout 运行”**。因此简化时要按“运行依赖迁移”处理，而不是按“死代码删除”处理。

---

## 2) 事实核查（对应你给的关注点）

### 2.1 phase-z / replace_contacts 路径

- 70b 已使用 phase-z（`direct_pose_use_phase_z=true`）：
  - `config/posttrain_WalkF_stage7_70b_phasezin_splitB2_pe32h512_20260227_fromarmchain.json:84`
- 70c 采用 `replace_contacts`：
  - `config/posttrain_WalkF_stage7_70c_replacecontacts_splitB2_pe32h512_20260227_fromarmchain.json:85`
- 实现入口在 direct head 里（`replace_contacts` 分支）：
  - `train/models.py:4136`

### 2.2 event_clock

- active config 里是 `event_clock="auto"`（例：70a/70b/70c）：
  - `config/posttrain_WalkF_stage7_70a_splitB2_pe32h512_20260227_fromarmchain.json:43`
- `auto/on/off` 的行为分支在：
  - `train/posttrain.py:4718`
- 明确存在安全提示：当 `off` 且 ckpt 含 event_clock 权重时，保存会丢弃该分支权重：
  - `train/posttrain.py:4726`

### 2.3 contact_meas（whitebox vs learned）

- 主链 config 的 `contact_meas_weight=0.0`（不做监督）：
  - `config/posttrain_WalkF_stage7_70b_phasezin_splitB2_pe32h512_20260227_fromarmchain.json:130`
- rollout 中 whitebox provider 路径已存在：
  - `train/posttrain.py:1354`
- learned meas 是否启用由 ckpt 是否含 `contact_meas_head.*` 自动推断：
  - `train/posttrain.py:4692`

### 2.4 contact_phase_state

- 是否启用同样由 ckpt 中是否含 phase-state 权重自动推断：
  - `train/posttrain.py:4831`
- 运行期确有 phase-state 分支入口：
  - `train/models.py:2679`
- 因此“删实现”风险不只在历史回放兼容，也会影响当前链路 rollout 行为。

---

## 3) 本地快照核对（2026-03-02，按 active whitelist）

对 `docs/delete/2026-03-01_posttrain_active_whitelist_runtime.txt` 中 8 个 config 逐个读取 `ckpt_in` 做 key 级检查，结果一致：

- `contact_meas_head.*`：存在
- `event_clock_gate.*` / `event_clock_corrector.*`：存在
- `contact_phase_state_init` / `contact_phase_state_delta_head.*`：存在

结论：当前主链上，这三块并非“未激活”，而是**由 ckpt 权重驱动自动启用**的 rollout 运行模块。

对应自动启用逻辑：
- meas：`train/posttrain.py:4692`
- event_clock：`train/posttrain.py:4718`
- phase_state：`train/posttrain.py:4831`

---

## 4) 安全逐步移除建议（按风险低→高）

基于 2026-03-02 的 Step A/B/C 实测，建议执行顺序改为：

1. `contact_phase_state`（原 Step C）
2. `contact_meas provider`（原 Step B）
3. `event_clock`（原 Step A，最后做）

### Step C（弱耦合候选）: 先处理 `contact_phase_state`

目标：先迁移/可选关闭相位状态分支，观察是否影响主质量指标。

- 依据：在当前 `phase_reset_source=none` 口径下，主指标变化接近 0（见第 8 节）。
- 执行：先做 runtime 开关化（默认可关），保留兼容路径一版。
- 验收：`GeoLocalDegWeighted` / `GeoDeg` 与 baseline 基本一致。

### Step B（弱耦合候选）: 再处理 `contact_meas` provider

目标：把当前“隐式（由 ckpt 推断）”改成“显式（配置可读）”。

建议新增配置语义（不一定立即删代码）：
- `contact_meas_provider: whitebox | learned | auto`
- train(base) 主链默认 `whitebox`
- `learned` 仅用于专项实验

验收：
- 在当前 active chain 上，`auto` vs `whitebox` 输出差异受控，并可解释
- `learned` 路径保留可跑（用于历史/研究复现）

### Step A（相对强耦合候选）: 最后处理 `event_clock`

目标：解释并量化 `event_clock` 的收益来源，再决定是否保留/降级/替代。

- 依据：`event_clock=off` 在主链上有明显质量回退（已在实测中确认）。
- 执行：先做 attribution（对 `plan_z` / `lambda` / per-step 误差做对齐分析），再考虑结构简化。
- 验收：若要默认关闭，需先证明目标场景下质量回退可接受。

---

## 5) 我对你原方案的补充（避免踩坑）

1. `event_clock=off` 在当前 active chain 上不是 no-op，且回退幅度明显；不适合作为第一步简化项。
2. `contact_meas_weight=0` 只代表“不监督”，不等于“不用 meas provider”；应先把 provider 显式化再决定默认值。
3. 在当前 `phase_reset_source=none` 口径下，`contact_phase_state` 更接近低风险候选，可优先做开关化迁移。

---

## 6) 推荐文档同步动作

本记录建议与以下文档联动维护：

- 主流程：`docs/posttrain_pipeline.md`
- 事实交接：`docs/Problems/active/2026-02-26_stage6_stage7_newflow_handoff.md`
- trainbase 分层治理：`docs/trainbase_design/2026-03-02_trainbase_v2_core_patch_flow.md`

建议把本页作为“优化点12拆分与归因记录”，与 trainbase v2 分层文档配套维护，避免讨论分叉。

---

## 7) Step B 实测（2026-03-02）

执行口径：
- 固定 `event_clock=auto`、`phase_reset_source=none`
- 对 active chain 8 个 ckpt 做 `contacts_meas_source=model` vs `whitebox` A/B freerun
- Teacher: `validate/teacher_batches/Walk_F_teacher.json`
- 产物：
  - `debug_output/stepB_contactmeas_ab_20260302/summary.md`
  - `debug_output/stepB_contactmeas_ab_20260302/stepB_model_vs_whitebox.csv`

结论（`round>=1` 平均）：
- 质量层面 whitebox 与 model 基本接近：
  - `GeoLocalDegWeighted` 均值差（whitebox-model）约 `-0.0154`
  - `GeoDeg` 均值差约 `+0.0797`
  - `DirectGeoLocalDegWeighted` 均值差约 `+0.0083`
- 推理耗时（单次实测，噪声较大）whitebox 侧平均约 `+7.35%`，需要按目标部署环境做重复 benchmark 再定最终默认值。

---

## 8) Step C 实测（contact_phase_state，2026-03-02）

执行口径：
- 固定 `event_clock=auto`、`contacts_meas_source=model`、`phase_reset_source=none`
- 对 active chain 8 个 ckpt 做 A/B：
  - ON：原始 ckpt
  - OFF：移除 `contact_phase_state_init` 和 `contact_phase_state_delta_head.*` 后的临时 ckpt
- 产物：
  - `debug_output/stepC_contactphasestate_ab_20260302/summary.md`
  - `debug_output/stepC_contactphasestate_ab_20260302/stepC_phase_state_on_vs_off.csv`
  - 临时 ckpt：`debug_output/stepC_contactphasestate_ab_20260302/ckpt_phase_state_off/`

结论（`round>=1` 平均，OFF-ON）：
- `GeoLocalDegWeighted`：约 `-0.000036`（几乎无差异）
- `GeoDeg`：约 `-0.000116`（几乎无差异）
- `DirectGeoLocalDegWeighted`：约 `+0.004206`（轻微变差）
- 推理耗时：平均约 `-4.07%`（单次测量噪声较大，建议重复 benchmark）

当前 `phase_reset_source=none` 口径下，`contact_phase_state` 对主质量指标影响很小，但对 direct-local 指标有轻微影响；可作为后续“可选关闭/迁移”的候选模块继续评估。

---

## 9) A/B 快速验证：contact / event_clock 对 `inc/direct + λ fusion` 的耦合（2026-03-02）

### 9.1 验证思路（最小可复现）

目标：验证“影响是否主要经由 contact 链路作用到 direct 或 λ fusion”。

- E1（direct-hint 路径）：
  - 固定 `--lambda_fusion_apply` 关闭（避免闭环状态改写干扰）。
  - 仅改 `--direct_pose_meas_source model -> zero`。
- E2（event_clock -> λ 路径）：
  - 开启 `--lambda_fusion_apply`。
  - 仅改 `--event_clock auto -> off`。
- E3（contact-meas -> λ 路径）：
  - 开启 `--lambda_fusion_apply`。
  - 仅改 `--contacts_meas_source model -> zero`。

统一口径：
- ckpt：`models/MLPL2_DirectBranch_v1/ckpt_last_posttrain_DirectBranch_v1_d1_phaseclk_lambda_cycles2_after_direct_pose.pth`
- teacher：`Walk_F_teacher.json` + `Walk_R_To_L_teacher.json`
- `rounds=5`，统计 `cycle>=1`（即 `round>=1`）均值
- 其他固定：`--time-index-mode cycle --depth 3 --phase_reset_source none --log_contacts`

### 9.2 执行指令（本次实际运行）

```bash
# E1-A: direct baseline (no apply)
PYTHONPATH=. python -m train.validate.run_freerun_cycles \
  --teacher validate/teacher_batches/Walk_F_teacher.json validate/teacher_batches/Walk_R_To_L_teacher.json \
  --model models/MLPL2_DirectBranch_v1/ckpt_last_posttrain_DirectBranch_v1_d1_phaseclk_lambda_cycles2_after_direct_pose.pth \
  --rounds 5 --time-index-mode cycle --depth 3 \
  --event_clock auto --phase_reset_source none \
  --contacts_meas_source model --direct_pose_meas_source model \
  --log_contacts \
  --out debug_output/trainbase_ab_20260302/direct_noapply_base --force

# E1-B: cut direct meas hint (no apply)
PYTHONPATH=. python -m train.validate.run_freerun_cycles \
  --teacher validate/teacher_batches/Walk_F_teacher.json validate/teacher_batches/Walk_R_To_L_teacher.json \
  --model models/MLPL2_DirectBranch_v1/ckpt_last_posttrain_DirectBranch_v1_d1_phaseclk_lambda_cycles2_after_direct_pose.pth \
  --rounds 5 --time-index-mode cycle --depth 3 \
  --event_clock auto --phase_reset_source none \
  --contacts_meas_source model --direct_pose_meas_source zero \
  --log_contacts \
  --out debug_output/trainbase_ab_20260302/direct_noapply_cut --force

# E2-A: apply baseline
PYTHONPATH=. python -m train.validate.run_freerun_cycles \
  --teacher validate/teacher_batches/Walk_F_teacher.json validate/teacher_batches/Walk_R_To_L_teacher.json \
  --model models/MLPL2_DirectBranch_v1/ckpt_last_posttrain_DirectBranch_v1_d1_phaseclk_lambda_cycles2_after_direct_pose.pth \
  --rounds 5 --time-index-mode cycle --depth 3 \
  --event_clock auto --phase_reset_source none \
  --contacts_meas_source model --direct_pose_meas_source model \
  --lambda_fusion_apply --log_contacts \
  --out debug_output/trainbase_ab_20260302/apply_base --force

# E2-B: event_clock off (apply)
PYTHONPATH=. python -m train.validate.run_freerun_cycles \
  --teacher validate/teacher_batches/Walk_F_teacher.json validate/teacher_batches/Walk_R_To_L_teacher.json \
  --model models/MLPL2_DirectBranch_v1/ckpt_last_posttrain_DirectBranch_v1_d1_phaseclk_lambda_cycles2_after_direct_pose.pth \
  --rounds 5 --time-index-mode cycle --depth 3 \
  --event_clock off --phase_reset_source none \
  --contacts_meas_source model --direct_pose_meas_source model \
  --lambda_fusion_apply --log_contacts \
  --out debug_output/trainbase_ab_20260302/apply_eventclock_off --force

# E3-B: contacts_meas zero (apply)
PYTHONPATH=. python -m train.validate.run_freerun_cycles \
  --teacher validate/teacher_batches/Walk_F_teacher.json validate/teacher_batches/Walk_R_To_L_teacher.json \
  --model models/MLPL2_DirectBranch_v1/ckpt_last_posttrain_DirectBranch_v1_d1_phaseclk_lambda_cycles2_after_direct_pose.pth \
  --rounds 5 --time-index-mode cycle --depth 3 \
  --event_clock auto --phase_reset_source none \
  --contacts_meas_source zero --direct_pose_meas_source model \
  --lambda_fusion_apply --log_contacts \
  --out debug_output/trainbase_ab_20260302/apply_contacts_zero --force
```

### 9.3 产物

- 汇总：
  - `debug_output/trainbase_ab_20260302/summary.md`
  - `debug_output/trainbase_ab_20260302/ab_metrics_round_ge1.csv`
- 单实验 JSON：
  - `debug_output/trainbase_ab_20260302/direct_noapply_base/*.json`
  - `debug_output/trainbase_ab_20260302/direct_noapply_cut/*.json`
  - `debug_output/trainbase_ab_20260302/apply_base/*.json`
  - `debug_output/trainbase_ab_20260302/apply_eventclock_off/*.json`
  - `debug_output/trainbase_ab_20260302/apply_contacts_zero/*.json`

### 9.4 结果（`cycle>=1` 聚合，cand-base）

- E1（direct-hint，noapply，`direct_pose_meas_source: model -> zero`）：
  - `ΔDirectGeoLocalDegWeighted = +0.0181`
  - `ΔGeoLocalDegWeighted = +0.0000`
  - 结论：对 direct 分支有轻微影响；在 noapply 口径下整体误差几乎不变。

- E2（event_clock，apply，`event_clock: auto -> off`）：
  - `ΔBlendGeoLocalDegWeighted = +0.2392`
  - `ΔGeoLocalDegWeighted = +0.2397`
  - `ΔDirectGeoLocalDegWeighted = +0.1665`
  - 结论：在 apply 闭环口径下，event_clock 影响明显（相对强耦合）。

- E3（contact-meas，apply，`contacts_meas_source: model -> zero`）：
  - `ΔBlendGeoLocalDegWeighted = +0.1670`
  - `ΔGeoLocalDegWeighted = +0.1605`
  - `ΔDirectGeoLocalDegWeighted = +0.1016`
  - `ΔContactErrAbsMean = +0.3200`
  - 结论：contact 链路对闭环质量有明显影响；其影响可以传导到 direct/λ 相关指标。

### 9.5 对当前假设的结论

本次快速 A/B 支持以下判断：

1. `contact_phase_state` / `contact_meas provider` 作为“可拆分对象”仍可按弱耦合候选推进（见第 7/8 节已有结果）。
2. 但“contact 链路本体”并非无关：当直接切断 meas（`source=zero`）时，对 apply 闭环有明显回退。
3. `event_clock` 在当前 ckpt+口径下表现为相对强耦合模块，维持“最后做、先归因”的策略不变。

---

## 10) 优化点12拆分矩阵（2×2×2×2）补充实测（2026-03-02）

### 10.1 目的与矩阵定义

目的：把“谁在影响 `inc/direct + λ fusion`”从单点 A/B 扩展为完整拆分矩阵。

矩阵维度：
- `lambda_fusion_apply`：`off` / `on`
- `event_clock`：`auto` / `off`
- `contacts_meas_source`：`model` / `zero`
- `direct_pose_meas_source`：`model` / `zero`

固定条件：
- ckpt：`models/MLPL2_DirectBranch_v1/ckpt_last_posttrain_DirectBranch_v1_d1_phaseclk_lambda_cycles2_after_direct_pose.pth`
- teacher：`Walk_F_teacher.json`、`Walk_R_To_L_teacher.json`
- `rounds=5`，统计 `cycle>=1`（=`round>=1`）

### 10.2 执行命令（本次实际运行）

```bash
python3 - <<'PY'
import subprocess
from pathlib import Path

root = Path('/Users/xingzhaorui/PycharmProjects/PythonProject')
ckpt = root / 'models/MLPL2_DirectBranch_v1/ckpt_last_posttrain_DirectBranch_v1_d1_phaseclk_lambda_cycles2_after_direct_pose.pth'
teachers = [
    root / 'validate/teacher_batches/Walk_F_teacher.json',
    root / 'validate/teacher_batches/Walk_R_To_L_teacher.json',
]
out_root = root / 'debug_output/trainbase_ab_20260302_matrix'
out_root.mkdir(parents=True, exist_ok=True)

base_cmd = [
    'python', '-m', 'train.validate.run_freerun_cycles',
    '--teacher', *(str(p) for p in teachers),
    '--model', str(ckpt),
    '--rounds', '5',
    '--time-index-mode', 'cycle',
    '--depth', '3',
    '--phase_reset_source', 'none',
    '--log_contacts',
    '--force',
]

for apply in [False, True]:
    for ec in ['auto', 'off']:
        for csrc in ['model', 'zero']:
            for dsrc in ['model', 'zero']:
                tag = f"apply_{int(apply)}__ec_{ec}__c_{csrc}__d_{dsrc}"
                cmd = list(base_cmd)
                cmd += ['--event_clock', ec]
                cmd += ['--contacts_meas_source', csrc]
                cmd += ['--direct_pose_meas_source', dsrc]
                if apply:
                    cmd += ['--lambda_fusion_apply']
                cmd += ['--out', str(out_root / tag)]
                subprocess.run(cmd, cwd=str(root), check=True)
PY
```

### 10.3 产物

- 矩阵汇总：
  - `debug_output/trainbase_ab_20260302_matrix/summary.md`
  - `debug_output/trainbase_ab_20260302_matrix/matrix_metrics_cycle_ge1.csv`
- 16 组运行 JSON：
  - `debug_output/trainbase_ab_20260302_matrix/apply_0__ec_*__c_*__d_*/*.json`
  - `debug_output/trainbase_ab_20260302_matrix/apply_1__ec_*__c_*__d_*/*.json`

### 10.4 关键结果（围绕 baseline: `apply=?, ec=auto, c=model, d=model`）

`apply=0`（noapply）下的单因素变化：
- `d:model->zero`：`ΔDirectGeoLocalDegWeighted=+0.0181`，`ΔBlendGeoLocalDegWeighted=+0.0000`
- `c:model->zero`：`ΔDirectGeoLocalDegWeighted=-0.1205`，`ΔBlendGeoLocalDegWeighted=-1.2093`，`ΔContactErrAbsMean=+0.3147`
- `ec:auto->off`：`ΔDirectGeoLocalDegWeighted=+0.0416`，`ΔBlendGeoLocalDegWeighted=+2.2281`

`apply=1`（apply）下的单因素变化：
- `d:model->zero`：`ΔDirectGeoLocalDegWeighted=+0.0402`，`ΔBlendGeoLocalDegWeighted=+0.0362`
- `c:model->zero`：`ΔDirectGeoLocalDegWeighted=+0.1016`，`ΔBlendGeoLocalDegWeighted=+0.1670`，`ΔContactErrAbsMean=+0.3200`
- `ec:auto->off`：`ΔDirectGeoLocalDegWeighted=+0.1665`，`ΔBlendGeoLocalDegWeighted=+0.2392`

### 10.5 本轮结论（用于后续拆分决策）

1. `direct_pose_meas_source` 对 direct 分支有轻微影响（noapply/apply 都是小幅退化），属于相对弱耦合路径。
2. `event_clock` 在 apply 下回退最大（`ΔBlend≈+0.239`），支持“相对强耦合、最后处理”的判断。
3. `contacts_meas` 链路在 apply 下也有明显影响（`ΔBlend≈+0.167`，`ΔContactErr≈+0.320`），说明“contact 链路本体”不是弱耦合。
4. 与第 7 节（model vs whitebox 差异小）不矛盾：**provider 替换可弱耦合，但 contact 链路硬切断（`source=zero`）是强扰动**。
