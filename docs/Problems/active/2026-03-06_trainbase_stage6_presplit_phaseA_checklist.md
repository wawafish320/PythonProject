# 2026-03-06 TrainBase 前置 Stage6 3-way armchain 实施清单（Phase A）

Last updated: 2026-03-07

## 0) TL;DR（2026-03-07）

- `H4.2 best_free / H4.2 last / phaseA_ctrl` 的 downstream full-chain 已按 `pretrain_contact + affine_mix08` 跑完。
- trainbase 侧的 `+0.145°` 级 endpoint gap 没有在 downstream 中 1:1 放大；在最终 `lambda_apply on` 口径下，相对 `phaseA_ctrl` 只剩：
  - `H4.2 best_free = +0.0122`
  - `H4.2 last = +0.0147`
- `best_free` 与 `last` 在 downstream 上代表两种不同最优：
  - `best_free` 略优于单一 global scalar；
  - `last` 明显更优于 lower-body pocket（`leg8 / SIC12-15{foot_l,ball_l} / calf_r`）；
  - 因而当前问题更像 **情形 C + selector mismatch**，而不是 B，也不是纯 D。
- 为钉死“只改 Stage6 split-aware 是否值得继续”这一分支，已额外完成 `Stage6-only gate`：只改 `Stage6` split-aware group norm，`70R/71/72/lambda` 全不动。
- gate 结果是 **Stage6 出口全组回退**：
  - `all_ex_root = +0.14473`
  - `leg = +0.34268`
  - `nonleg = +0.10193`
  因此机器判定为 `stop_after_stage6`。
- 当前可直接收口的结论是：
  - **“只靠 Stage6 split-aware 适配去救 H4.2 best_free downstream” 这条方向关闭**；
  - **Phase A 在这一分支上可视为 resolved**；
  - 后续若还要继续推进，优先级应放在 **selector 对齐**，而不是继续做 trainbase / Stage6 的小步调权。

## 1) 目标与范围

本清单用于落地如下方向：

- 将 `Stage6 split-first 3-way armchain` 的 direct branch 形态前置到 trainbase。
- 先做 **Phase A：基线冻结 + 最小实现 + 指标对齐**，不在本轮直接切换整条下游主链结论。

本轮只回答两件事：

1. trainbase 是否已经具备与 `Stage6 3-way armchain` 一致的 direct head / direct loss 口径；
2. 当前 trainbase 产物进入下游 posttrain 后，回退究竟发生在：
   - basetrain 终点分布；
   - Stage6 起跑初始化；
   - Stage7 中段适配；
   - 还是 lambda final 收尾。

本轮明确不做：

- 不前置 `70R/71/72` 的 train-only / freeze 策略。
- 不迁移 lambda reliability / gate supervision / boundary weighting。
- 不改 `event_clock` 机制。
- 不在本轮额外追加 phase/provider 历史清理动作。

---

## 2) 当前下游线路确认（先钉死）

Phase A 之后的 downstream 诊断链路，统一按下面口径固定：

- `posttrain_contacts_source = pretrain_contact`
- `posttrain_contacts_pretrain_affine_stats = affine_mix08`

也就是说：

- **当前这条问题链路里，应把 `pretrain_contact + affine_mix08` 视为下游主诊断线路；**
- `whitebox` 只保留为历史 control / validate 对照，不再作为本问题默认假设。

确认依据：

1. `train/posttrain.py` 当前 runtime 已只接受 `posttrain_contacts_source=pretrain_contact`。
2. `debug_output/_tmp_phaseD_posttrain_ab_20260305/summary_phaseD_posttrain_ab.md` 中，
   `pretrain_contact + affine_mix08` 相对 `whitebox` 已满足门槛：
   - `GeoLocalDegWeighted`: `+0.033678 <= +0.05`
   - `ContactErrAbsMean`: `-0.154467 <= -0.02`
3. 因此本问题后续所有 trainbase -> posttrain 回退定位，都应在相同下游 source 口径下比较，避免把 `source` 切换误判为“上游前置 split”回退。

固定资产：

- affine：`debug_output/_tmp_phaseb_affine_20260304/affine_fit_mix08/affine_stats.json`
- encoder bundle：必须在全部 A/B 中保持同一份；建议优先沿用已验证资产，并在运行记录中显式落盘

---

## 3) Phase A 要冻结的基线产物

### 3.1 Trainbase 侧基线

固定以下文件为本轮 baseline：

- `debug_output/__tmp_basetrain_bestfree_groupdist_20260305/Walk_F_freerun_cycles.json`
- `debug_output/__tmp_basetrain_bestfree_groupdist_20260305/group_summary.json`
- `debug_output/__tmp_basetrain_bestfree_groupdist_20260305/basetrain_keybone_group_summary.json`
- `debug_output/__tmp_basetrain_bestfree_groupdist_20260305/posttrain_stage6_init_stats.json`

### 3.2 Posttrain 侧对照基线

固定以下日志为 downstream 对照链路：

- `models/MLPL2_DirectBranch_v1__base_to_stage7_lambdafinal_20260226_cpu/posttrain_log_WalkF_stage6_direct_cond_anchor_splitfirst_3way_armchain_pe32h512_20260227.json`
- `models/MLPL2_DirectBranch_v1__base_to_stage7_lambdafinal_20260226_cpu/posttrain_log_WalkF_stage7_70R_nonleg_recovery_proj256_preleg_20260227_fromarmchain.json`
- `models/MLPL2_DirectBranch_v1__base_to_stage7_lambdafinal_20260226_cpu/posttrain_log_WalkF_stage7_71_legonly_after_nonlegproj256_20260227_fromarmchain.json`
- `models/MLPL2_DirectBranch_v1__base_to_stage7_lambdafinal_20260226_cpu/posttrain_log_WalkF_stage7_72_legomega_after_nonlegproj256_20260227_fromarmchain.json`
- `models/MLPL2_DirectBranch_v1__base_to_stage7_lambdafinal_20260226_cpu/posttrain_log_WalkF_stage7_lambda_final_calib_20260227_fromarmchain_fullcompat.json`
- `debug_output/_tmp_phaseD_posttrain_ab_20260305/summary_phaseD_posttrain_ab.json`

---

## 4) 当前基线快照（写死，便于后续回归比对）

### 4.1 Basetrain 终点 group 分布（`cycle>=1`, `drop_wrap=true`）

来源：`debug_output/__tmp_basetrain_bestfree_groupdist_20260305/group_summary.json`

- `leg mean = 10.6688°`
- `arm mean = 6.6670°`
- `else mean = 2.0487°`
- `nonleg mean = 5.2940°`

### 4.2 Stage6 起跑初始化成本

来源：`debug_output/__tmp_basetrain_bestfree_groupdist_20260305/posttrain_stage6_init_stats.json`

- step1: `dir_leg_base = 0.1919`
- step1: `dir_nonleg_base = 0.0624`
- step1: `leg_over_nonleg = 3.075x`
- step1: `direct_grad_norm_out_arm / direct_grad_norm_out_else = 7.709x`
- head20 mean: `leg_over_nonleg = 3.550x`

### 4.3 下游 source A/B 已有结论

来源：`debug_output/_tmp_phaseD_posttrain_ab_20260305/summary_phaseD_posttrain_ab.md`

- `pretrain_contact + affine_mix08` 相对 `whitebox`：
  - `GeoLocalDegWeighted = +0.033678`
  - `ContactErrAbsMean = -0.154467`
  - `SourceMatchRate = 1.0`

注：这组数值不用于证明“前置 split 一定收益”，只用于固定本轮 downstream 诊断口径。

---

## 5) Phase A 实施清单

### A0. 冻结实验边界

- 固定 teacher / clips / rounds / depth / `time_index_mode` / `phase_reset_source`。
- 固定 downstream source 为 `pretrain_contact + affine_mix08`。
- 固定 encoder bundle，不允许 control / experiment 混用不同 bundle。
- control 与 experiment 除 trainbase 前置 split 差异外，其余配置全部保持一致。

### A1. 补齐 trainbase 配置入口（必须）

trainbase 至少要显式支持并落盘以下键：

- `direct_pose_split_enable`
- `direct_pose_arm_split_enable`
- `direct_pose_arm_bones`
- `direct_pose_nonleg_proj_dim`
- `direct_pose_loss_leg_split`

建议同时补齐（用于 parity 与定位）：

- `direct_pose_grad_monitor_enable`
- `direct_pose_grad_ratio_gate`
- `direct_pose_loss_group_norm_enable`
- `direct_pose_loss_group_norm_w_leg`
- `direct_pose_loss_group_norm_w_nonleg`
- `direct_pose_loss_group_norm_ema_beta`
- `direct_pose_loss_group_norm_ratio_min`
- `direct_pose_loss_group_norm_ratio_max`
- `direct_pose_loss_group_norm_eps`

说明：

- 本轮目标不是只开一个 `split_enable` 壳；
- 目标是让 trainbase 的 direct branch 在结构、loss 统计、日志可观测性上，足够接近 `Stage6 3-way armchain`。

### A2. 补齐 trainbase -> model 接线（必须）

需要确保 trainbase 在实例化 `EventMotionModel` 时，把 A1 中所有必要键传入模型，而不是只传 `direct_pose_split_enable`。

最小要求：

1. `direct_pose_split_enable=true` 时，模型实际创建 leg/non-leg split 输出头；
2. `direct_pose_arm_split_enable=true` 时，模型实际创建 arm/else 分头；
3. `direct_pose_arm_bones` 解析结果与当前 Stage6 armchain 口径一致；
4. `direct_pose_nonleg_proj_dim` 在 trainbase 中可见、可解析、可保存。

### A3. trainbase direct loss 改造（必须）

本轮至少做到：

1. direct loss 支持 split-aware 统计；
2. 不只保留总 `direct`，要能区分：
   - `leg`
   - `nonleg`
   - `arm`
   - `else`
3. 日志中至少能回答：
   - 现在 trainbase 是否真正启用了 3-way armchain；
   - 当前竞争主要发生在 `leg vs nonleg`，还是 `arm vs else`。

建议新增的最小日志字段：

- `dir_leg_base`
- `dir_nonleg_base`
- `dir_arm_base`
- `dir_else_base`
- `leg_over_nonleg`
- `arm_over_else`
- `direct_grad_norm_out_leg`
- `direct_grad_norm_out_arm`
- `direct_grad_norm_out_else`
- `direct_grad_norm_trunk`

### A4. Phase A 固定导出面板（必须）

每次 control / experiment 都必须产出以下四层面板：

1. **Basetrain 终点分布面板**
   - 来源：`group_summary.json`
   - 指标：`leg/arm/else/nonleg` 的 `mean/p50/p90/p95/samples`

2. **Basetrain 训练过程面板**
   - 来源：`basetrain_keybone_group_summary.json`
   - 指标：`GeoLocalDeg`、`KeyBoneGeoLocalDegMean`、`group_mean.leg/arm/trunk`、`GeoDriftSlopeProxy`

3. **Stage6 起跑成本面板**
   - 来源：`posttrain_stage6_init_stats.json`
   - 指标：step1 / head20 的 `dir_leg_base`、`dir_nonleg_base`、`leg_over_nonleg`、`grad_arm_over_else`

4. **Posttrain 分阶段适配面板**
   - 来源：`posttrain_log_*`
   - 关注阶段：`Stage6`, `70R`, `71`, `72`, `lambda_final`
   - 指标：`dir_leg_base`、`dir_nonleg_base`、group norm、grad norm、grad ratio、最终 `GeoLocalDegWeighted` / `ContactErrAbsMean`

### A5. Control / Experiment 运行定义（必须）

- **control**：当前 unsplit trainbase 产物 + 固定 downstream 路由
- **experiment**：前置 `Stage6 3-way armchain` 的 trainbase 产物 + 同一条 downstream 路由

运行要求：

1. 同 seed；
2. 同 teacher / 同 5 clips；
3. 同 `pretrain_contact + affine_mix08`；
4. 同 posttrain 配置链；
5. `whitebox` 仅可作为额外 reference，不得混入主判断。

### A6. 2026-03-06 本轮实跑记录（已完成）

本节记录 2026-03-06 这轮实际执行的 Phase A A/B。

#### A6.1 先决检查

静态检查：

```bash
python3 -m py_compile train/models.py train/training_MPL.py train/eval_utils.py
```

参数入口核对：

```bash
python3 -m train.training_MPL --help | rg "direct_pose_arm_split_enable|direct_pose_arm_bones|direct_pose_nonleg_proj_dim|direct_pose_loss_leg_split|direct_pose_grad_monitor_enable|direct_pose_loss_group_norm_enable"
```

核对结果：以上 6 个 Phase A 需要的 trainbase 入口均已存在。

#### A6.2 本轮使用的 trainbase baseline 配置

原计划希望直接使用“当前 trainbase config_json”；但当前 runtime 下，历史配置
`config/exp_phase_DirectBranch_v1_d1_noreset.json` 与直接裸用 `config/exp_phase_mpl.json`
都包含已被主入口移除/拒收的旧键（例如 `contact_meas_enable` / `contact_meas_hidden`
等），无法直接作为 Phase A A/B 基线。

因此本轮实际做法是：

1. 以 `config/exp_phase_mpl.json` 为底；
2. 移除当前入口不再接受的旧键；
3. 用 2026-03-03 baseline 口径补齐 trainbase 主链公共参数；
4. 仅让 control / experiment 在 split 相关键上不同。

实际生成命令：

```bash
python3 - <<'PY'
import json
from pathlib import Path
src = Path('config/exp_phase_mpl.json')
out = Path('debug_output/__tmp_phaseA_trainbase_config_20260306.json')
obj = json.loads(src.read_text())
for k in ['contact_meas_dropout', 'contact_meas_enable', 'contact_meas_hidden']:
    obj.pop(k, None)
obj.update({
    'out': './runs',
    'depth': 3,
    'encoder_path': './models/motion_encoder_equiv_stageA.pt',
    'contact_plan_enable': True,
    'contact_plan_inject': 'plan_z',
    'contact_plan_init_mode': 'learnable+obs',
    'contact_plan_init_hidden': 128,
    'phase_reset_source': 'none',
    'direct_pose_enable': True,
    'w_direct_pose': 0.2,
    'contact_plan_time_pe_dim': 16,
    'direct_pose_meas_mode': 'concat',
    'direct_pose_meas_drop_prob': 0.05,
    'direct_pose_plan_drop_prob': 0.05,
    'direct_pose_meas_noise_std': 0.02,
    'use_event_clock': True,
    'event_clock_max_delta': 0.5,
    'event_clock_hidden_dim': 64,
    'event_clock_gate_hidden_dim': 32,
    'event_clock_lambda_entropy_weight': 0.01,
    'event_clock_lambda_prior_weight': 0.01,
    'event_clock_delta_z_l2_weight': 0.001,
})
out.write_text(json.dumps(obj, ensure_ascii=False, indent=2))
print(out)
PY
```

本轮约定：

- `BASE_TRAIN_CFG=debug_output/__tmp_phaseA_trainbase_config_20260306.json`
- seed 固定为 `0`
- `out=./runs`

#### A6.3 本轮实际 trainbase A/B 指令

arm bones 常量：

```bash
ARM_BONES=$(python3 - <<'PY'
from train.models import STAGE6_3WAY_ARMCHAIN_BONES_CSV
print(STAGE6_3WAY_ARMCHAIN_BONES_CSV)
PY
)
```

由于 `training_MPL` 当前无显式 `--seed` 参数，本轮用 Python wrapper 在进入
`train.training_MPL` 前固定：`random.seed(0)`、`np.random.seed(0)`、`torch.manual_seed(0)`。

control：

```bash
PYTHONPATH=. python3 - <<'PY'
import sys, runpy, random
import numpy as np
import torch
seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
sys.argv = [
    'train.training_MPL',
    '--config_json', 'debug_output/__tmp_phaseA_trainbase_config_20260306.json',
    '--run_name', 'phaseA_ctrl',
    '--config_override', 'direct_pose_grad_monitor_enable=True',
    '--config_override', 'direct_pose_split_enable=False',
    '--config_override', 'direct_pose_arm_split_enable=False',
    '--config_override', 'direct_pose_nonleg_proj_dim=0',
    '--config_override', 'direct_pose_loss_leg_split=False',
    '--config_override', 'direct_pose_loss_group_norm_enable=False',
]
runpy.run_module('train.training_MPL', run_name='__main__')
PY
```

experiment：

```bash
PYTHONPATH=. python3 - <<'PY'
import sys, runpy, random
import numpy as np
import torch
from train.models import STAGE6_3WAY_ARMCHAIN_BONES_CSV
seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
arm_bones = f'direct_pose_arm_bones="{STAGE6_3WAY_ARMCHAIN_BONES_CSV}"'
sys.argv = [
    'train.training_MPL',
    '--config_json', 'debug_output/__tmp_phaseA_trainbase_config_20260306.json',
    '--run_name', 'phaseA_exp',
    '--config_override', 'direct_pose_grad_monitor_enable=True',
    '--config_override', 'direct_pose_split_enable=True',
    '--config_override', 'direct_pose_arm_split_enable=True',
    '--config_override', arm_bones,
    '--config_override', 'direct_pose_nonleg_proj_dim=256',
    '--config_override', 'direct_pose_loss_leg_split=True',
    '--config_override', 'direct_pose_loss_group_norm_enable=False',
]
runpy.run_module('train.training_MPL', run_name='__main__')
PY
```

实际产物目录：

- control：`runs/phaseA_ctrl`
- experiment：`runs/phaseA_exp`

补充：两边训练结束后 ONNX 导出都报了同一个 legacy contract 错误：

- `ONNX export expects fixed mainchain contract plan_z+phase_z, got plan_dim=64, phase_dim=0`

但这发生在 `ckpt_best_*`、`metrics/*.json`、`config_resolved.json` 全部落盘之后，
不影响本轮 Phase A 的 basetrain / Stage6 结论。

#### A6.4 本轮实际 basetrain 产物检查

1. `config_resolved.json` 已满足 A1/A2：

   - `runs/phaseA_ctrl/config_resolved.json`
   - `runs/phaseA_exp/config_resolved.json`

   其中 experiment 侧已正确保存：

   - `direct_pose_split_enable=true`
   - `direct_pose_arm_split_enable=true`
   - `direct_pose_arm_bones=<Stage6 3-way armchain csv>`
   - `direct_pose_nonleg_proj_dim=256`
   - `direct_pose_loss_leg_split=true`

2. `basetrain_keybone_group_summary.json` 已生成：

   - `runs/phaseA_ctrl/basetrain_keybone_group_summary.json`
   - `runs/phaseA_exp/basetrain_keybone_group_summary.json`

3. `metrics/train_ep001.json` 的一个 runtime 现状需要单独记录：

   - `train_ep001` 只包含最小 train 标量，尚未包含 split/base/grad 字段；
   - 从 `train_ep002` 开始，这些字段稳定出现；
   - 因而本轮针对 A3/A4 的 train-side 字段核对，实际以 `train_ep002.json` 为首个有效 epoch。

4. `phaseA_exp` 自 `train_ep002` 起，已能观测到：

   - `dir_leg_base`
   - `dir_nonleg_base`
   - `dir_arm_base`
   - `dir_else_base`
   - `leg_over_nonleg`
   - `arm_over_else`
   - `direct_grad_norm_out_arm`
   - `direct_grad_norm_out_else`

5. `phaseA_ctrl` 自 `train_ep002` 起，已有 split-aware base 字段，但因为 unsplit 无独立
   `arm/else` 输出头，grad 侧只稳定记录 trunk，不应强行要求 `direct_grad_norm_out_arm`
   与 `direct_grad_norm_out_else` 必须存在。

#### A6.5 本轮 actual group_summary 生成命令

control：

```bash
PYTHONPATH=. python -m train.validate.run_freerun_cycles \
  --teacher validate/teacher_batches/Walk_F_teacher.json \
  --model runs/phaseA_ctrl/ckpt_best_free_phaseA_ctrl.pth \
  --rounds 5 --depth 3 --time-index-mode cycle --phase_reset_source none \
  --contacts_meas_source pretrain_contact \
  --contacts_meas_pretrain_clamp 1.0 \
  --contacts_meas_pretrain_affine_stats <affine_stats.json> \
  --encoder-bundle <encoder_bundle> \
  --export_joint_direct_geolocal_series \
  --out debug_output/phaseA_ctrl_freerun --force
```

experiment：

```bash
PYTHONPATH=. python -m train.validate.run_freerun_cycles \
  --teacher validate/teacher_batches/Walk_F_teacher.json \
  --model runs/phaseA_exp/ckpt_best_free_phaseA_exp.pth \
  --rounds 5 --depth 3 --time-index-mode cycle --phase_reset_source none \
  --contacts_meas_source pretrain_contact \
  --contacts_meas_pretrain_clamp 1.0 \
  --contacts_meas_pretrain_affine_stats <affine_stats.json> \
  --encoder-bundle <encoder_bundle> \
  --export_joint_direct_geolocal_series \
  --out debug_output/phaseA_exp_freerun --force
```

聚合脚本：

```bash
python3 - <<'PY'
import json, numpy as np
from pathlib import Path
from train.models import DEFAULT_DIRECT_POSE_LEG_BONES, STAGE6_3WAY_ARMCHAIN_BONES
for name in ['phaseA_ctrl','phaseA_exp']:
    src = Path(f'debug_output/{name}_freerun/Walk_F_freerun_cycles.json')
    out = Path(f'debug_output/{name}_freerun/group_summary.json')
    obj = json.loads(src.read_text())
    bones = obj['per_step_direct_geolocal_deg']['bone_names']
    root_idx = int(obj['per_step_direct_geolocal_deg'].get('root_idx', 0))
    series = obj['per_step_direct_geolocal_deg']['DirectGeoLocalDeg']
    meta = obj['metrics_per_step']
    idx = {n:i for i,n in enumerate(bones)}
    leg = [idx[n] for n in DEFAULT_DIRECT_POSE_LEG_BONES if n in idx]
    arm = [idx[n] for n in STAGE6_3WAY_ARMCHAIN_BONES if n in idx]
    all_ex_root = [i for i in range(len(bones)) if i != root_idx]
    nonleg = [i for i in all_ex_root if i not in set(leg)]
    else_idx = [i for i in nonleg if i not in set(arm)]
    keep = [i for i, row in enumerate(meta) if int(row.get('cycle', 0)) >= 1 and not bool(row.get('wrap_boundary_step', False))]
    def stats(indices):
        vals = np.asarray([[series[t][j] for j in indices] for t in keep], dtype=float).reshape(-1)
        return {
            'j': len(indices),
            'samples': int(vals.size),
            'mean': float(vals.mean()),
            'p50': float(np.percentile(vals, 50)),
            'p90': float(np.percentile(vals, 90)),
            'p95': float(np.percentile(vals, 95)),
        }
    payload = {
        'source': str(src),
        'mask': {'cycle_gte': 1, 'drop_wrap': True, 'kept_steps': len(keep), 'total_steps': len(meta)},
        'groups': {'leg': stats(leg), 'nonleg': stats(nonleg), 'arm': stats(arm), 'else': stats(else_idx), 'all_ex_root': stats(all_ex_root)},
        'group_names': {'leg': [bones[i] for i in leg], 'arm': [bones[i] for i in arm], 'else': [bones[i] for i in else_idx]},
    }
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    print(out)
PY
```

#### A6.6 本轮 actual Stage6 起跑命令

公共变量：

```bash
ENCODER_BUNDLE=models/motion_encoder_equiv.pt.best.pt
AFFINE_STATS=debug_output/_tmp_phaseb_affine_20260304/affine_fit_mix08/affine_stats.json
PRETRAIN_CLAMP=1.0
```

control：

```bash
PYTHONPATH=. python -m train.posttrain \
  --config config/posttrain_WalkF_stage6_direct_cond_anchor_splitfirst_3way_armchain_pe32h512_20260227.json \
  --ckpt_in runs/phaseA_ctrl/ckpt_best_free_phaseA_ctrl.pth \
  --out_dir models/phaseA_ctrl_posttrain \
  --run_name phaseA_ctrl_stage6 \
  --posttrain_contacts_source pretrain_contact \
  --posttrain_contacts_pretrain_clamp "${PRETRAIN_CLAMP}" \
  --encoder_bundle "${ENCODER_BUNDLE}" \
  --posttrain_contacts_pretrain_affine_stats "${AFFINE_STATS}"
```

experiment：

```bash
PYTHONPATH=. python -m train.posttrain \
  --config config/posttrain_WalkF_stage6_direct_cond_anchor_splitfirst_3way_armchain_pe32h512_20260227.json \
  --ckpt_in runs/phaseA_exp/ckpt_best_free_phaseA_exp.pth \
  --out_dir models/phaseA_exp_posttrain \
  --run_name phaseA_exp_stage6 \
  --posttrain_contacts_source pretrain_contact \
  --posttrain_contacts_pretrain_clamp "${PRETRAIN_CLAMP}" \
  --encoder_bundle "${ENCODER_BUNDLE}" \
  --posttrain_contacts_pretrain_affine_stats "${AFFINE_STATS}"
```

初始化统计提取脚本与本清单前文定义一致，实际产物为：

- `models/phaseA_ctrl_posttrain/posttrain_stage6_init_stats.json`
- `models/phaseA_exp_posttrain/posttrain_stage6_init_stats.json`

#### A6.7 本轮结果（2026-03-06）

1. **Basetrain 终点 group_summary**

   来源：

   - `debug_output/phaseA_ctrl_freerun/group_summary.json`
   - `debug_output/phaseA_exp_freerun/group_summary.json`

   对比（experiment - control）：

   - `leg mean`: `10.8805° -> 11.2746°`（`+0.3941°`）
   - `nonleg mean`: `4.9734° -> 5.4008°`（`+0.4275°`）
   - `arm mean`: `6.2560° -> 6.7690°`（`+0.5130°`）
   - `else mean`: `1.9417° -> 2.1670°`（`+0.2253°`）
   - `all_ex_root mean`: `6.0235° -> 6.4450°`（`+0.4215°`）

   结论：experiment 的 basetrain 终点分布整体更差，不是单一 limb 的偶发回退。

2. **Basetrain 训练过程面板**

   来源：

   - `runs/phaseA_ctrl/basetrain_keybone_group_summary.json`
   - `runs/phaseA_exp/basetrain_keybone_group_summary.json`

   `best_teacher_by_GeoLocalDeg`：

   - control：`GeoLocalDeg=0.059239`，`KeyBoneGeoLocalDegMean=0.076709`
   - experiment：`GeoLocalDeg=0.070370`，`KeyBoneGeoLocalDegMean=0.086441`

   `best_free_by_GeoDriftSlopeProxy`：

   - control：`GeoLocalDeg=3.678026`，`KeyBoneGeoLocalDegMean=6.435477`，`GeoDriftSlopeProxy=0.509611`
   - experiment：`GeoLocalDeg=3.768759`，`KeyBoneGeoLocalDegMean=6.446567`，`GeoDriftSlopeProxy=0.517855`

   group mean（best_free row）：

   - control：`leg=7.200680`，`arm=4.121953`，`trunk=1.877380`
   - experiment：`leg=7.246432`，`arm=4.567642`，`trunk=2.037460`

   结论：teacher 与 freerun 过程面板都没有显示 experiment 优于 control。

3. **Stage6 起跑成本面板**

   来源：

   - `models/phaseA_ctrl_posttrain/posttrain_stage6_init_stats.json`
   - `models/phaseA_exp_posttrain/posttrain_stage6_init_stats.json`

   step1：

   - control：`dir_leg_base=0.170177`，`dir_nonleg_base=0.064913`，`leg_over_nonleg=2.621607`，`grad_arm_over_else=7.572005`
   - experiment：`dir_leg_base=0.170038`，`dir_nonleg_base=0.064921`，`leg_over_nonleg=2.619131`，`grad_arm_over_else=7.514850`

   head20 mean：

   - control：`leg_over_nonleg=3.441878`
   - experiment：`leg_over_nonleg=3.417696`

   结论：Stage6 起跑几乎一致，experiment 未出现额外恶化；若只看 Stage6 起点，
   两边可以视作同一量级。

#### A6.8 本轮判定（只落在 trainbase / Stage6 前半链）

依据本清单第 6 节决策树，本轮应归入：

- **情形 A：Basetrain 终点分布已变差**

原因：

1. `group_summary.json` 中 experiment 相比 control 在 `leg/arm/else/nonleg` 上均变差；
2. `basetrain_keybone_group_summary.json` 的 teacher / freerun 过程面板也同步偏差；
3. `posttrain_stage6_init_stats.json` 并未显示 experiment 在 Stage6 起跑继续恶化。

因此本轮结论是：

- 问题主要落在 **trainbase 本体**；
- 目前没有证据支持把责任先归到 `trainbase -> Stage6` 接口分布；
- 更不应在这一轮先怪 `70R/71/72`、lambda 或 event_clock。

#### A6.9 本轮为跑通 A/B 暴露并修复的 trainbase runtime blocker

这轮在正式 A/B 起跑前暴露出两个与 Phase A 无关、但会阻塞 trainbase 评估/落盘的
runtime bug；已先做最小修复，以保证 A/B 命令可跑通：

1. `train/eval_utils.py`
   - `evaluate_teacher()` 内错误引用未定义的 `_avg_simple_dict` / `_avg_nested_dict`
   - `evaluate_freerun()` 末尾缺失 `_avg_dict_recursive`
2. `train/training_MPL.py`
   - `Trainer.fit()` 的 `forced_valfree_metrics` 在异常路径下未预先初始化，
     会触发 `UnboundLocalError`

这些修复只影响 trainbase 训练期评估聚合与日志落盘，不涉及 Stage7 / lambda /
event_clock 语义变更。

#### A6.10 曲线形状证据链与假设优先级（2026-03-06 更新）

基于 `runs/phaseA_ctrl/metrics/teacher_ep*.json`、
`runs/phaseA_exp/metrics/teacher_ep*.json`、
`runs/phaseA_exp/metrics/train_ep*.json` 的逐 epoch 对比，本轮对 H1/H2/H3 的优先级更新为：

- `H3（split 后 loss/竞争结构失衡）` > `H1（trunk 梯度被 split 改写）` >> `H2（纯训练 budget 不足）`

证据链如下：

1. **teacher 曲线形状不像“从 step 0 就全面更难学”**

   以 teacher pose 指标而不是 raw loss 为主看：

   - early（ep1-4）平均：experiment - control
     - `GeoLocalDeg = -0.024668`
     - `KeyBoneGeoLocalDegMean = -0.057696`
   - mid（ep5-8）平均：
     - `GeoLocalDeg = +0.007993`
     - `KeyBoneGeoLocalDegMean = +0.017428`
   - late（ep9-18）平均：
     - `GeoLocalDeg = +0.010205`
     - `KeyBoneGeoLocalDegMean = +0.004989`

   解释：experiment 不是从 step 0 就在 pose 指标上全面更差；更像是前 1-2 个 epoch
   仍能建立表征，但从 ep3 以后逐步进入更差的平台。因此“纯 H1：早期 trunk 碎梯度导致
   一开始就学不动”不是当前首要解释。

2. **late gap 主要集中在 arm，不在 leg / trunk**

   late（ep9-18）平均：experiment - control

   - `leg = +0.004535`
   - `arm = +0.044840`
   - `trunk = -0.000813`

   终点 ep18 也一致：

   - control：`leg=0.129494`，`arm=0.280505`，`trunk=0.109474`
   - experiment：`leg=0.138050`，`arm=0.315976`，`trunk=0.111065`

   解释：本轮退化的主导项是 arm，而不是 leg 或 trunk 同步全面恶化；这更符合
   `H3`，即 split 后子头之间的优化竞争/权重结构失衡。

3. **train-side split 统计里，arm/else 的梯度竞争长期失衡**

   在 `phaseA_exp` 的 `train_ep002` 到 `train_ep018` 中：

   - `arm_over_else` 长期在 `2.37 ~ 2.65`
   - `direct_grad_ratio_arm_over_else` 长期在 `7.78 ~ 16.60`

   解释：这说明 arm 分支不仅误差基线更高，梯度量级也持续强于 else；当前 split 结构下，
   arm/else 的竞争并不平衡。

4. **trunk 梯度确实被放大，但当前更像 H3 的结果，而不是 H1 的独立证据**

   按 `direct_grad_norm_trunk` 比较，experiment / control 在 ep2-18 的平均倍率约为 `2.24x`。

   但由于当前 arm 子头已经明显主导梯度流，trunk 被拉高本身就是自然结果；因此这个 `2.24x`
   目前不能单独作为“H1 已证实”的充分证据。更合理的顺序应是：

   - 先修 H3；
   - 再复测 trunk gradient 倍率；
   - 若修完 H3 后 trunk 仍显著偏高，再提升 H1 的优先级。

5. **H2 当前证据最弱**

   如果只是“split 参数更多、budget 不够”，更常见的形状应是：experiment 后期仍持续追近。
   但本轮 ep9-18 更像平行但更差的 late plateau，而不是单纯的收敛滞后。

#### A6.11 group norm 现状核对（trainbase vs Stage6）

这个问题需要单独钉死，因为它决定 H3 ablation 的第一步该做什么。

1. **本轮 Phase A A/B 实跑里，trainbase group norm 实际是关闭的**

   本轮 control / experiment 都显式使用了：

   - `direct_pose_loss_group_norm_enable=False`

   因此本轮观测到的 `arm/else` 失衡，并不能解读为“group norm 已启用但没压住”；
   更准确地说，是“本轮 A/B 设计里根本没有启用这条约束”。

2. **trainbase 当前 argparse 默认值与 Stage6 explicit config 并不完全相同**

   trainbase 当前默认值：

   - `direct_pose_loss_group_norm_enable=False`
   - `direct_pose_loss_group_norm_w_leg=1.0`
   - `direct_pose_loss_group_norm_w_nonleg=1.0`
   - `direct_pose_loss_group_norm_ema_beta=0.9`
   - `direct_pose_loss_group_norm_ratio_min=0.2`
   - `direct_pose_loss_group_norm_ratio_max=5.0`
   - `direct_pose_loss_group_norm_eps=1e-6`
   - `direct_pose_grad_ratio_gate=0.35`

   Stage6 当前 config 显式值：

   - `direct_pose_loss_group_norm_enable=true`
   - `direct_pose_loss_group_norm_w_leg=1.0`
   - `direct_pose_loss_group_norm_w_nonleg=1.0`
   - `direct_pose_loss_group_norm_ema_beta=0.95`
   - `direct_pose_loss_group_norm_ratio_min=0.2`
   - `direct_pose_loss_group_norm_ratio_max=5.0`
   - `direct_pose_loss_group_norm_eps=1e-6`
   - `direct_pose_grad_ratio_gate=0.35`

   结论：

   - **enable 位不同**：trainbase 默认关，Stage6 显式开；
   - **EMA beta 不同**：trainbase 默认 `0.9`，Stage6 用 `0.95`；
   - 其余数值在当前主链上是对齐的。

3. **当前 group norm 的作用对象只是 leg/nonleg，不是 arm/else**

   现实现中，group norm objective 只对：

   - `dir_leg_base / ema_leg_prev`
   - `dir_nonleg_base / ema_non_prev`

   做 clamp 和加权；并不会直接对 `arm/else` 比值施加约束。

   因而：

   - `arm_over_else` 与 `direct_grad_ratio_arm_over_else` 可以作为 H3 失衡信号；
   - 但不能把它们直接解释成“现有 group norm clamp 范围没压住 arm/else”；
   - 若后续确认 arm/else 是主矛盾，最终可能需要的是 **arm/else-specific rebalance**，
     而不是仅复用 leg/nonleg group norm。

4. **H3 ablation 的第一步不是盲调权重，而是先确认 Stage6 那组 norm 参数在 trainbase from-scratch 场景下到底有没有“真正生效”**

   用户补充的关键判断成立：Stage6 是在一个已收敛 checkpoint 上微调，而 trainbase 是从随机初始化开始。
   即便 `ratio_min / ratio_max / ema_beta` 数值看起来接近，同一组 clamp 参数在这两个场景中的有效约束力也可能完全不同。

   结合本轮核对结果：

   - trainbase 本轮 A/B 中 `direct_pose_loss_group_norm_enable=False`，所以这轮还不能下“group norm 已开但没压住”的结论；
   - 但 trainbase 默认值与 Stage6 explicit config 的确不完全相同：
     - trainbase 默认：`enable=False`，`ema_beta=0.9`，`ratio_min=0.2`，`ratio_max=5.0`
     - Stage6 显式：`enable=True`，`ema_beta=0.95`，`ratio_min=0.2`，`ratio_max=5.0`
   - 这意味着后续 H3 ablation 至少要把“是否启用 + from-scratch 下 clamp 实际命中情况”一起看，而不是只照搬 Stage6 数值。

   因此 H3-first 的最低成本、最高信号做法不是先调权重，而是：

   - 先在 trainbase 开启 group norm 观测；
   - 看 raw ratio / clamped ratio / clamp hit rate；
   - 再决定是调 `ratio_min/max`、调 `ema_beta`，还是需要走 arm/else-specific rebalance。

5. **teacher 最佳值已差，但曲线形状更像“中后程竞争失衡”，不是方向性否定 split**

   本轮 best teacher 确实已经显示 experiment 劣于 control：

   - control：`GeoLocalDeg=0.059239`
   - experiment：`GeoLocalDeg=0.070370`

   但结合 A6.10 的 early / mid / late 形状证据，当前更像：

   - split head 从 basetrain 一开始就介入；
   - 在当前训练量/信号密度下，中后期 competition / weighting 更快失衡；
   - 因而实现路径还不对，但 split 方向本身暂不应直接否定。

#### A6.12 下一步方向（优先级更新）

> 注：以下 1-5 是 A6.12 当时的初始建议。后续已实际完成 `H3-0 / H3.1 / H4.0 / H4.1 / H4.2`，更新后的执行结果与当前优先级见后文“**A6.12 执行回填（2026-03-07）**”。

同意把下一步收口为 **H3-first**，且第一步先做低成本、强信号的运行时诊断，而不是盲调权重。

推荐顺序：

1. **H3-0：先做 group norm runtime clamp 命中率诊断**

   目标：回答在 trainbase from-scratch 场景下，当前 group norm range 是否真正起作用；如果继续看到
   `arm_over_else` 与 `direct_grad_ratio_arm_over_else` 明显失衡，再判断这是“现有 norm 对 scratch 场景几乎不工作”，
   还是“主矛盾已经转成 arm/else，需单独 rebalance”。

   执行口径：

   - 下一轮 H3 ablation 应显式开启 `direct_pose_loss_group_norm_enable=True`；
   - 保留 `direct_pose_grad_monitor_enable=True`；
   - 先不盲调 loss weight，先补齐 clamp observability。

   最少应新增的运行时统计：

   - `dir_group_norm_leg_raw`
   - `dir_group_norm_nonleg_raw`
   - `dir_group_norm_leg_clamped`
   - `dir_group_norm_nonleg_clamped`
   - `dir_group_norm_leg_hit_min`
   - `dir_group_norm_leg_hit_max`
   - `dir_group_norm_nonleg_hit_min`
   - `dir_group_norm_nonleg_hit_max`

   建议同时输出 epoch 聚合：

   - `leg clamp hit rate`
   - `nonleg clamp hit rate`

   判读原则：

   - 若长期几乎不 hit，说明该 range 在 trainbase 场景下约束力太弱；
   - 若大部分 step 都打在 clamp 边界，说明当前允许区间对 from-scratch 动态仍过松，边界本身没有真正压住竞争；
   - raw ratio、clamped ratio 与 hit rate 三者必须一起看，不能只看命中率。

2. **H3-1：在 clamp hit 结果出来后，再定向调整 range / EMA，而不是先盲调权重**

   这一步优先于“多跑更多 epoch”。

3. **H1 作为第二优先级，且建议用 competition-driven trigger 而不是固定 epoch**

   若后续要做 curriculum split，优先考虑基于竞争指标触发，而不是固定 epoch：

   - 例如在 unified head 下，当 `grad_arm_over_else` 首次稳定超过某阈值（如 `4x`）时再 split；
   - 这样比硬编码 epoch 对 seed / clip 组合更稳健。

4. **H1 的 trunk 诊断要放在 H3 rebalance 之后复测**

   当前 `direct_grad_norm_trunk` 的 experiment / control 平均倍率约为 `2.24x`，但这很可能只是 arm 子头主导梯度后的伴随现象。

   更合理的判读顺序是：

   - 先做 H3 rebalance；
   - 再复测 trunk gradient 倍率是否自然回落；
   - 只有在 H3 修完后 trunk 仍显著偏高，H1 才具备独立诊断价值。

5. **H2 放在最后，只作为排除项**

   在完成 H3-first 与必要的 H1 验证前，不建议先用“加预算”解释当前回退。


#### A6.12 执行回填（2026-03-07）

按上述路线实际执行后的结果如下。结论上，A6.12 原先的 **H3-first** 判断在当时是合理的，但执行结果表明：

- `H3` 只解释了问题的一部分；
- `curriculum split` 本身是有效方向；
- 当前真正更强的信号已经转成 **late-stage trunk-level tradeoff + selector mismatch**。

1. **H3-0 证实：现有 group norm 在 trainbase from-scratch 下几乎没有真正生效**

   `phaseA_h30_gnorm` 的 epoch tail 显示：

   - `GroupNormLegClampHitRate = 0.0`
   - `GroupNormNonlegClampHitRate = 0.0`
   - `dir_group_norm_leg_raw = 0.9792`
   - `dir_group_norm_nonleg_raw = 0.9762`

   说明当前 `ratio_min/max` 在 scratch basetrain 动态下几乎不 hit；因此 A6.12 当时先做 observability 的判断是对的，但结果是：
   **这组 leg/nonleg clamp 并不是当前瓶颈。**

2. **H3.1 arm/else rebalance 方向正确，但幅度有限**

   `phaseA_h31_armelse_eq` 相对 `phaseA_h30_gnorm`：

   - `direct_grad_ratio_arm_over_else: 9.47 -> 4.10`
   - endpoint `all_ex_root = -0.0690°`
   - endpoint `arm = -0.1237°`
   - endpoint `else = -0.0283°`
   - 但 `leg = +0.0529°`
   - `best_free` 反而几乎不变/微差：`GeoLocalDeg = +0.0021°`

   解释：rebalance 确实改善了 nonleg 内部失衡，但只追回了退化的一部分，且没有传导到 checkpoint 选择层。

3. **H4.0 curriculum split 证明 split 路线仍成立，但问题从“H3-only”转成了“late-stage tradeoff”**

   `phaseA_h40_curr_stageB_split_h31` 相对原始 `phaseA_ctrl`：

   - `best_teacher`: `GeoLocalDeg = -0.0091°`
   - `best_free`: `GeoLocalDeg = -0.0684°`
   - 但 `best_free` endpoint 仍有明显 group gap：
     - `all_ex_root = +0.3179°`
     - `arm = +0.5786°`
     - `nonleg = +0.4250°`
     - `leg = -0.1774°`

   更关键的是 `best_free(ep8)` 到 `last(ep14)` 的变化：

   - `all_ex_root = -0.0565°`
   - `arm = -0.2479°`
   - `nonleg = -0.1706°`
   - `leg = +0.4713°`

   且 `last(ep14)` 相对 `ctrl(best_free)` 为：

   - `all_ex_root = +0.2614°`
   - `arm = +0.3307°`
   - `nonleg = +0.2543°`
   - `leg = +0.2939°`

   即后半段不是“全面变差”，而是 **arm/nonleg 继续改善、leg 明显恶化**。因此主问题不再像单纯 arm/else reweight，而更像
   **split 下 trunk 层的 leg/nonleg 竞争**。

4. **H4.1 late else-upweight 基本排除了“arm/else 仍是主矛盾”**

   `phaseA_h41_curr_stageB_lateelse` 相对 `H4.0`：

   - `direct_grad_ratio_arm_over_else: 4.24 -> 1.38`
   - 但 `best_teacher` 完全相同
   - `best_free` 完全相同
   - endpoint 还微差：`all_ex_root = +0.0098°`

   这说明在 curriculum split 之后，**继续压 arm/else ratio 已不能改变最优 checkpoint 的位置**；arm/else 竞争已不是当前 dominant bottleneck。

5. **原始 unsplit ctrl 也存在轻微 tradeoff，但 split 把它放大了**

   `phaseA_ctrl_last` 相对 `phaseA_ctrl(best_free)`：

   - `leg = +0.0870°`
   - `nonleg = -0.0254°`
   - `arm = -0.0376°`
   - `all_ex_root = -0.0054°`

   方向与 split 相同，但幅度远小于 `H4.0` 的 `leg +0.4713°`。因此：

   - tradeoff 本身并非 split 独有；
   - 但 split 明显放大了共享 trunk 下的 leg/nonleg 竞争。

6. **H4.2（ep9 之后 freeze trunk）首次直接支持 trunk-level 假设**

   `phaseA_h42_curr_stageB_trunkfreeze_ep9` 的 Stage B 设定为：

   - `ep1-8` 沿用 `H4.0`；
   - `ep9-14` 保持 split head 学习，但冻结 `direct_pose_head` trunk。

   注意：冻结阶段 `direct_grad_norm_trunk` 变为 `NaN/None` 属预期，不应再与未冻结 run 的 trunk grad 数值直接比较。

   结果分两层看：

   **(a) best checkpoint 仍未赢 ctrl**

   相对 `phaseA_ctrl`：

   - `best_teacher`: `GeoLocalDeg = -0.00735°`
   - `best_free`: `GeoLocalDeg = +0.0460°`
   - `best_free` endpoint：
     - `all_ex_root = +0.2511°`
     - `arm = +0.4624°`
     - `nonleg = +0.3408°`
     - `leg = -0.1640°`

   且相对 `H4.0 best_free`，`H4.2 best_free` 反而更差：

   - `GeoLocalDeg = +0.1145°`
   - `KeyBoneGeoLocalDegMean = +0.1101°`

   **(b) 但 endpoint tradeoff 明显收敛**

   `H4.2 last` 相对 `phaseA_ctrl(best_free)`：

   - `all_ex_root = +0.1446°`
   - `arm = +0.2304°`
   - `nonleg = +0.1761°`
   - `leg = -0.0011°`

   相对 `H4.0 last`：

   - `all_ex_root = -0.1168°`
   - `arm = -0.1002°`
   - `nonleg = -0.0783°`
   - `leg = -0.2950°`

   且 `H4.2 best -> last` 的 tradeoff 幅度也显著缩小：

   - `all_ex_root = -0.1065°`
   - `arm = -0.2320°`
   - `nonleg = -0.1648°`
   - `leg = +0.1629°`

   对比 `H4.0 best -> last` 的 `leg +0.4713°`，可以认为 **freeze trunk 明显压住了 late-stage leg collapse**。
   这意味着 trunk-level 竞争确实是当前主矛盾之一。

7. **由 H4.2 进一步暴露出的新问题是 selector mismatch**

   `H4.2` 的 `best_free epoch = 7`，仍然落在 freeze 之前；但 freeze 之后 `last` 的 group endpoint 已明显优于 `H4.0 last`，也明显更接近 `ctrl`。

   这说明当前 `best_free` 选择依据（`GeoDriftSlopeProxy`）与我们真正关心的
   `cycle>=1 + drop_wrap + group endpoint` 目标并不对齐。现阶段若只看 `best_free`，会低估 trunk-level 修正的收益。

#### A6.12 下一步方向（2026-03-07 再次更新）

基于上述回填，A6.12 的优先级需要从原来的 **H3-first** 更新为：

1. **H1/H4 trunk-level late-stage control + selector 对齐** 为第一优先级

   当前最强信号已经不是 arm/else ratio，而是：

   - split 后半段的 trunk 更新会放大 leg/nonleg tradeoff；
   - 现有 `best_free` selector 又不能正确挑出更好的 endpoint。

2. **H3 arm/else reweight 降为次优先级**

   它能改善梯度比，但对 best checkpoint 选择几乎没有影响，收益上限已比较清楚。

3. **后续低成本验证优先顺序**

   - `freeze trunk` 的软化版：`trunk lr << head lr`，验证是否能保留 `H4.2` 的 endpoint 改善，同时把 `best_free` 拉回；
   - 增加/替换 checkpoint selector：把 `leg/nonleg/arm/else/all_ex_root` 的 endpoint 聚合纳入选点，而不是只看 `GeoDriftSlopeProxy`；
   - 若 selector 对齐后仍无解，再考虑更细的 competition-driven split trigger。

4. **H2 仍放最后**

   在完成 trunk-level 诊断与 selector 对齐前，仍不建议先用“加预算”解释当前回退。

#### A6.13 downstream 主链回填（2026-03-07）

按 `docs/posttrain_pipeline.md` 的当前主线口径，已完成以下三条 full-chain：

- `phaseA_ctrl`：复用现有 `models/phaseA_ctrl_posttrain/ckpt_last_phaseA_ctrl_stage6.pth`，继续跑 `70a -> 70b -> 70c -> 70R -> 71 -> 72 -> lambda_final`；
- `phaseA_h42_curr_stageB_trunkfreeze_ep9 best_free`：从 `runs/phaseA_h42_curr_stageB_trunkfreeze_ep9/ckpt_best_free_phaseA_h42_curr_stageB_trunkfreeze_ep9.pth` 起跑，跑同一条链；
- `phaseA_h42_curr_stageB_trunkfreeze_ep9 ckpt_last`：从 `runs/phaseA_h42_curr_stageB_trunkfreeze_ep9/ckpt_last_phaseA_h42_curr_stageB_trunkfreeze_ep9.pth` 起跑，跑同一条链。

固定约束：

- `posttrain_contacts_source = pretrain_contact`
- `posttrain_contacts_pretrain_affine_stats = affine_mix08`
- 不引入额外 `event_clock / provider / source` 变化。

本轮集中产物：

- 汇总读数：`debug_output/_tmp_phaseA_downstream_20260307/phaseA_downstream_readout.json`
- final ckpt 清单：`debug_output/_tmp_phaseA_downstream_20260307/finals.env`
- 关键对照：
  - `debug_output/_tmp_phaseA_downstream_20260307/ctrl_vs_h42_bestfree_apply/`
  - `debug_output/_tmp_phaseA_downstream_20260307/ctrl_vs_h42_last_apply/`
  - `debug_output/_tmp_phaseA_downstream_20260307/h42_bestfree_vs_last_apply/`

回填结论如下：

1. **trainbase 的 `+0.145°` 级 endpoint gap 没有在 downstream 中 1:1 放大，反而大部分被吸收**

   在最终 `lambda_apply on` 口径下，`round>=1` 的 `GeoLocalDegWeighted`：

   - `phaseA_ctrl = 0.9254`
   - `H4.2 best_free = 0.9376`（相对 ctrl 仅 `+0.0122`）
   - `H4.2 last = 0.9400`（相对 ctrl 仅 `+0.0147`）

   换句话说，trainbase 侧的明显差距，并没有原样传导到最终 downstream scalar。

   但这不等于“完全无差别”：在 masked direct signal（`cycle>=1 + drop_wrap`）上，
   `H4.2 best_free / last` 的 global mean 反而都优于 `ctrl`：

   - `ctrl -> H4.2 best_free`：`0.15606 -> 0.14594`（`-6.48%`）
   - `ctrl -> H4.2 last`：`0.15606 -> 0.14751`（`-5.48%`）

   说明 downstream 确实吸收了大量全局误差，但 residual 主要表现为局部 pocket，而不是全局均匀恶化。

2. **`best_free` 与 `last` 在 downstream 上代表的是两种不同的“更优”**

   若只看单一 global scalar，`best_free` 仍然略优于 `last`：

   - `H4.2 best_free on = 0.9376`
   - `H4.2 last on = 0.9400`

   但若看我们真正关心的 lower-body pocket，则 `last` 更友好：

   - 相对 `best_free`，`last` 的 `leg8_mean` 继续下降 `-0.00818`
   - `SIC12-15 + {foot_l, ball_l}` 从 `0.5912` 降到 `0.4399`
   - `calf_r global` 从 `0.4578` 降到 `0.3215`

   同时，`last` 也不是无代价更好：

   - `non_leg_mean` 相对 `best_free` 回升 `+0.00367`
   - `calf_r_sic2_4` 比 `best_free` 更高。

   因此，`H4.2` 在 downstream 上进一步坐实了 **selector mismatch**：

   - `best_free` 更像是“global scalar 更优”；
   - `last` 更像是“lower-body 风险更低”。

   若 selector 仍只看单一 `GeoDriftSlopeProxy` / global scalar，会系统性低估 freeze-trunk 带来的 lower-body endpoint 收益。

3. **当前 full-chain 更接近命中“情形 C + selector mismatch”，而不是 B，也不是纯 D**

   先排除 `B`：

   - `Stage6 init` 没有出现足够强的起跑恶化；
   - `leg_over_nonleg / grad_arm_over_else` 相对 `ctrl` 只有极小扰动，不支持“trainbase -> Stage6 接口明显坏掉”的解释。

   再看 `D`：

   - `best_free` 在 `lambda off` 下相对 `ctrl` 仍更好；
   - `last` 在 `lambda off` 下则反而更差。

   这说明问题并不只是 `lambda_final` 单层决定，而是 `70R/71/72 -> lambda` 的组合路径对两类上游分布的响应不同。

   综合判断：

   - **不是** “Stage6 起跑就坏了”；
   - **更像** “中段适配 + 选点目标不一致” 共同造成了表面回退；
   - 因此当前 Phase A 的主任务，不应再继续在 trainbase 层面做小步调权，而应先完成 downstream-aware selector 对齐。

4. **对下一步优先级的影响**

   结合本轮 full-chain，优先级再收敛为：

   - 先暂停 trainbase 侧 `split 时机 / arm-else 权重 / trunk freeze` 这类局部迭代；
   - 第一优先级改为 **selector 对齐**：至少同时纳入
     `global scalar + leg8 + SIC12-15{foot_l,ball_l} + calf_r`；
   - 第二优先级才是 `freeze trunk` 的软化版（如 `trunk lr << head lr`），目标是保留 `last` 的 lower-body 收益，同时尽量不损失 `best_free` 的 global scalar；
   - 若 selector 对齐后 downstream 仍不稳，再回头定位 `70R/71/72` 哪一级对新上游分布最敏感。

#### A6.14 Stage6-only split-aware gate（2026-03-07）

为把“是否还值得继续沿 Stage6 split-aware 适配往下试”这个问题一次性钉死，本轮增加了一个更小的 gate 实验：

- 固定上游为 `H4.2 best_free`
- 只改 `Stage6` 的 split-aware 配置；
- `70a/70b/70c/70R/71/72/lambda` **全部不动、也不继续跑**；
- 先看 `Stage6` 出口的分 group 统计是否改善，再决定是否值得把变体接进完整 downstream 链。

本轮产物：

- runner：`debug_output/_tmp_phaseA_h42_stage6_splitaware_gate_20260307/run_stage6_gate.sh`
- 决策输出：`debug_output/_tmp_phaseA_h42_stage6_splitaware_gate_20260307/stage6_gate_decision.json`
- group 对比：`debug_output/_tmp_phaseA_h42_stage6_splitaware_gate_20260307/stage6_group_compare.json`
- baseline `Stage6` 出口：`debug_output/_tmp_phaseA_h42_stage6_splitaware_gate_20260307/baseline_stage6_group_summary.json`
- variant `Stage6` 出口：`debug_output/_tmp_phaseA_h42_stage6_splitaware_gate_20260307/variant_stage6_group_summary.json`

实验设置：

- baseline：直接使用已跑完的 `H4.2 best_free -> Stage6` 产物
  `models/__tmp_phaseA_downstream_20260307/phaseA_h42_bestfree/ckpt_last_phaseA_h42_bestfree_stage6_pcsrc20260307.pth`
- variant：仍从
  `runs/phaseA_h42_curr_stageB_trunkfreeze_ep9/ckpt_best_free_phaseA_h42_curr_stageB_trunkfreeze_ep9.pth`
  起跑 `Stage6`，但只覆盖 split-aware group norm 相关参数：
  - `direct_pose_loss_group_norm_enable=true`
  - `direct_pose_loss_group_norm_ema_beta=0.9`
  - `direct_pose_loss_group_norm_ratio_min=0.7`
  - `direct_pose_loss_group_norm_ratio_max=1.4`

也就是：**只动 Stage6 的 split-aware 约束强度，不动后续链路配置**。

判定标准（本轮实际执行版）：

- 只有当 `Stage6` 出口同时满足：
  - `all_ex_root` 有可见改善；
  - `leg` 不回退；
  - `nonleg` 不回退；
  才允许继续把该变体接入 `70R/71/72/lambda`。

结果如下（variant - baseline）：

- `all_ex_root = +0.14473`
- `leg = +0.34268`
- `nonleg = +0.10193`
- `arm = +0.12747`
- `else = +0.04158`

也就是说，**Stage6 出口不是“没明显变化”，而是全组显著变差**：

- baseline `all_ex_root = 0.31006` → variant `0.45479`
- baseline `leg = 0.81364` → variant `1.15632`
- baseline `nonleg = 0.20117` → variant `0.30311`

因此本轮 gate 的机器判定为：

- `decision = stop_after_stage6`

结论：

1. **这条“只靠 Stage6 split-aware 适配去救 H4.2 best_free downstream”的方向可以直接关闭**。

   因为它在 `Stage6` 出口就已经明显回退，
   没有任何继续浪费预算去试 `70R/71/72/lambda` 的必要。

2. 这也从反面支持了一个更强结论：

   - 当前主链并不是“缺一个 Stage6 split-aware 小修正”就能翻负；
   - downstream 对现有 `H4.2 best_free` 上游分布，已经有相当强的自动吸收能力；
   - 继续在这条局部适配线上深挖，信号/成本比已经很差。

3. 按本轮用户事先约定的 gate 语义，应视为：

   - **该方向关闭**；
   - **Phase A 在“Stage6 split-aware downstream 适配”这一分支上可视为 resolved**。

---

## 6) 回退定位决策树（本清单的核心用途）

### 情形 A：Basetrain 终点分布已变差

表现：

- `group_summary.json` 中 `leg/arm/else/nonleg` 分布本身就比 baseline 更差；
- `basetrain_keybone_group_summary.json` 的 teacher 曲线也同步恶化。

判定：

- 问题在 trainbase 本体；
- 不应先怪 Stage6 / Stage7。

### 情形 B：Basetrain 分布稳定，但 Stage6 起跑成本变差

表现：

- trainbase 终点面板接近 baseline；
- 但 `posttrain_stage6_init_stats.json` 的 `leg_over_nonleg` / `grad_arm_over_else` 明显升高。

判定：

- 问题在 **trainbase -> Stage6 接口分布**；
- 说明前置 split 后的表征，与当前 Stage6 初始化习惯仍存在 mismatch。

### 情形 C：Stage6 起跑稳定，但 70R/71/72 中段变差

表现：

- Stage6 初始化信号近似不变；
- 但 `70R/71/72` 中的 direct/base/grad 统计开始偏离。

判定：

- 问题在 downstream 对新上游分布的敏感性；
- 重点排查 `70R/71/72` 的适配路径，而不是回退 trainbase split 本身。

### 情形 D：前段稳定，lambda final 才回退

表现：

- `Stage6/70R/71/72` 均无明显恶化；
- 最终 `lambda_final` 或 apply-on 指标单独回退。

判定：

- 问题主要属于 lambda 路由 / 融合，而非 direct split head 前置本身。

### A6.13 当前实际命中（2026-03-07）

基于本轮 `H4.2 best_free / H4.2 last / phaseA_ctrl` 的 full-chain 回填：

- **不支持情形 B**：`Stage6 init` 未见显著恶化；
- **不属于纯情形 D**：差异并非只在 `lambda_final` 才出现；
- **当前更接近情形 C**：`70R/71/72 -> lambda` 对新上游分布的吸收与放大方式不同；
- 同时伴随一个额外问题：**selector mismatch**。也即，当前“最佳 ckpt”的选法，与 downstream 真正关心的 lower-body 风险并不对齐。

补充：

- `A6.14` 的 `Stage6-only split-aware gate` 已证明：单独增强 `Stage6` split-aware 约束不会改善出口，反而会让 `leg/nonleg/all_ex_root` 同时变差；
- 因而当前不应再继续沿“只改 Stage6 split-aware，然后再接 70R/71/72/lambda 复赌一次”的方向投入预算。

---

## 7) Phase A 验收条件

最小验收：

1. trainbase 运行时能识别、保存并回显 `3-way armchain` 相关配置；
2. trainbase direct 日志中能区分 `leg/nonleg/arm/else`；
3. 可稳定导出本文件第 5.4 节定义的四层面板；
4. downstream 评估固定跑在 `pretrain_contact + affine_mix08`；
5. 不引入 `70R/71/72`、lambda、event_clock、phase/provider 的额外行为变化。

建议 gate：

- 若 `Phase A` 只完成参数壳，未完成 split-aware 统计与导出面板，则视为 **未完成**；
- 若 source 未固定、或 control / experiment 使用了不同 contact 路由，则该轮结果 **无效**。

---

## 8) 文档分工

- scope freeze：`docs/Problems/active/2026-03-06_trainbase_stage6_presplit_only_scope.md`
- 起跑前诊断：`docs/Problems/active/2026-03-06_basetrain_to_posttrain_startline_diagnostic.md`
- posttrain 历史接入与 A/B：`docs/Problems/active/2026-03-04_pretrain_contact_route_debug_handoff.md`
- 当前主链说明（注意其中 whitebox 默认描述已落后于当前 runtime）：`docs/posttrain_pipeline.md`

注：本清单优先服从“当前 runtime + 已验证产物”的事实口径；如文档与代码冲突，以当前 runtime 行为为准。
