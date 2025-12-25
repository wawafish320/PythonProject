# SO(3) Delta Corrector：用 Lie Algebra 纠正 Free‑Run Drift

## 目录

1. [背景与目标](#背景与目标)
2. [核心思路：在 SO(3) 上做 Delta Correction](#核心思路在-so3-上做-delta-correction)
3. [训练信号：闭环定义，避免推理端不可得信息](#训练信号闭环定义避免推理端不可得信息)
4. [代码落点（最小侵入）](#代码落点最小侵入)
5. [开关与推荐配置](#开关与推荐配置)
6. [对训练的影响与常见坑](#对训练的影响与常见坑)
7. [调试与验证清单](#调试与验证清单)

---

## 背景与目标

我们希望在 **不引入“推理端不可得信息”** 的前提下：

- 把 free‑run 的累计误差从“越滚越偏”变成“偏了能拉回”（降低 drift slope）。
- 同时尽量保持单步精度（teacher forcing / 单帧 geodesic）不掉。

当前系统已具备长序列信号（scheduled sampling rollout + train_free window loss），但仅靠 `Δrot6d residual + I → reproject` 往往只能保证 **ΔR 合法**，仍可能出现 **系统性 bias**（长期 yaw/phase drift、特定骨骼持续偏转）导致误差斜率上升。

---

## 核心思路：在 SO(3) 上做 Delta Correction

### 问题：旋转不适合“欧氏加法残差”

虽然我们已有“残差 + I 再投影”的 rot6d delta（`normalize_rot6d_delta`），但本质仍是在欧氏空间上回归一个残差，再投影到 SO(3)。当模型进入 autoregressive 模式时，误差会通过链式合成累积，并出现偏差放大。

### 方案：SO(3) 上的 Delta Corrector

在每一步，我们在模型预测的增量旋转 `ΔR_pred` 之上，再乘一个 **群上的修正旋转**：

```
ΔR_used = Exp(ω̂) @ ΔR_pred
R_next  = ΔR_used @ R_prev
```

其中 `ω̂ ∈ so(3)` 用旋转向量（axis‑angle vector，shape `(B,J,3)`）表示，`Exp` 用 Rodrigues 稳定实现（见 `train/geometry.py` 的 `so3_exp_map`）。

直觉：
- `ΔR_pred` 负责“主动力学/趋势”；
- `Exp(ω̂)` 负责“纠偏/拉回”（可限幅、可 gate、可正则、可解释）。

---

## 训练信号：闭环定义，避免推理端不可得信息

关键约束：corrector 监督必须 **只依赖推理时也可获得的闭环状态**。

我们的 compose 约定是：

```
R_next = ΔR @ R_prev
```

因此在 step t（从 `R_prev_pred` 出发）构造闭环 target：

```
ΔR_target = R_gt_next @ (R_prev_pred)^T
R_err     = ΔR_target @ (ΔR_pred)^T
```

corrector 只需要拟合 `R_err`：

```
loss_corr = geodesic_R( Exp(ω̂), R_err )
```

优势：
- 用 SO(3) geodesic 度量（坐标系/参数化更鲁棒）；
- 监督与推理一致：仅依赖 `R_prev_pred`（来自 rollout）、`R_gt_next`（训练可得）、`ΔR_pred`（模型输出）。

### “只训练 corrector” 的梯度隔离

为避免 `loss_corr` 沿 rollout 的 `y_prev_raw` 链路回传到基模，我们提供开关：

- `so3_corr_detach_apply`: detach correction 的应用路径（避免 base loss 通过 `y_t` 回传到 corrector）
- `so3_corr_detach_target`: detach 监督 target 的状态（避免 `loss_corr` 通过 `y_prev_raw` 回传到基模）

默认推荐两者都为 `true`，这样 **base model** 与 **corrector** 基本解耦，只通过 `ΔR_pred.detach()` 对齐误差（让 corrector 学“补多少”）。

---

## 代码落点（最小侵入）

### 1) 新增轻量 head（非 RNN/GRU）

- 模型输出 `omega_hat`：`train/models.py`（`EventMotionModel.forward` 返回 dict 增加 `omega_hat`）
- head 采用 zero‑init（初始 `omega_hat == 0`），确保 baseline 行为严格不变。

### 2) rollout 中插入 correction（不改数据管线）

在 Trainer rollout 内：

1. `delta_norm → delta_raw`（std 反归一化）
2. `apply_so3_delta_correction`：只改 rot6d slice（把 `ΔR_used` 写回 residual rot6d）
3. `compose_rot6d_delta`：按既定约定 `R_next = ΔR @ R_prev` 合成

对应函数：
- `train/training_MPL.py`: `_delta_norm_to_raw` / `_compose_delta_raw`
- `train/training_MPL.py`: `_apply_so3_delta_correction` / `_so3_corr_prepare_omega`

### 3) train_free window 同步训练 corrector

为了让 corrector 更贴近长期 free‑run 的分布，`_freerun_loss_window` 内也会把 `_so3_corr_loss` 加进 `free_loss`（与主训练保持一致权重开关）。

---

## 开关与推荐配置

### 关键参数（CLI / config_json 同名）

| 参数 | 含义 | 推荐 |
|---|---|---|
| `so3_corr_enable` | 启用 correction（rollout/eval 应用） | `true` |
| `so3_corr_gate_force` | 强制 gate（sanity：0=完全不生效） | 初期 `0.0`，训练阶段 `null` |
| `so3_corr_loss_weight` | corrector geodesic 监督权重 | `0.01~0.05` |
| `so3_corr_omega_l2_weight` | `||ω||^2` 正则 | `1e-4~1e-3` |
| `so3_corr_max_deg` | 每步限幅（deg） | `10~30` |
| `so3_corr_detach_apply` | detach 应用链路（base loss 不训 corrector） | `true` |
| `so3_corr_detach_target` | detach target（corr loss 不训 base） | `true` |
| `so3_corr_apply_in_mixed` | mixed rollout 也应用 correction | `true` |
| `so3_corr_topk` | ValFree 打印 Top‑K joints | `8` |

### 推荐 schedule（exp_phase_mpl.json）

`config/exp_phase_mpl.json` 已内置一套“安全启用 → 后期开启训练”的策略：

- 前期：`so3_corr_enable=true` 但 `so3_corr_gate_force=0.0`  
  → 路径开启但严格不改变输出（用于 sanity）。
- stage2b（更强调 free‑run）：`so3_corr_gate_force=null` 且设置 `so3_corr_loss_weight / omega_l2`  
  → 开始训练并实际纠偏。

---

## 对训练的影响与常见坑

### 1) corrector 训练与基模“打架”

如果你希望 **corrector only**，务必开启：

- `so3_corr_detach_apply=true`
- `so3_corr_detach_target=true`

否则 corr loss 可能沿 `rollout state` 或 `compose` 链路对基模产生隐式约束，导致单帧精度波动。

### 2) 分布偏移：只在 mixed 监督不够贴近长期 free-run

我们已在 `train_free` window 中加入 corr loss，使 corrector 的训练分布更贴近长期 free‑run；若仍观察到 drift slope 不降，可：

- 提高 freerun window 的 horizon 或权重（你已有 `freerun_weight/horizon` 机制）
- 或将 stage2b 的 `so3_corr_loss_weight` 上调一点（先从 `0.03` 试）

### 3) 数值与过修正

- `so3_corr_max_deg` 是第一道保险（优先保守）。
- `so3_corr_omega_l2_weight` 用于抑制振荡/正反馈。
- 若出现“末端关节 correction 很大”，优先检查是否 `joint_count`/`rot6d slice` 对齐错误。

---

## 调试与验证清单

### A. baseline sanity

1. `so3_corr_enable=true` 且 `so3_corr_gate_force=0.0`  
   - teacher/free 指标应与 baseline 一致（或数值误差级别差异）。

#### 快速跑命令（`run_freerun_cycles`）

把下面命令里的 `--model` 和 `--out` 换成你自己的路径即可（其它参数保持与训练时一致，如 `--depth/--encoder-bundle`）。

**A1) baseline sanity（路径开启但严格不生效）**
```bash
python -m train.validate.run_freerun_cycles \
  --model YOUR_CKPT.pth \
  --teacher validate/teacher_batches/Walk_F_teacher.json \
  --bundle raw_data/processed_data/norm_template.json \
  --pretrain-template models/pretrain_template.json \
  --npz-root raw_data/processed_data \
  --encoder-bundle models/motion_encoder_equiv_stageA.pt \
  --out debug_output/freerun_cycles/YOUR_OUT_DIR \
  --rounds 2 \
  --depth 3 \
  --so3_corr_enable \
  --so3_corr_gate_force 0.0 \
  --force
```

### B. 启用纠偏（带限幅）

**B1) 启用 correction + 限幅（推荐先从 `20°` 试）**
```bash
python -m train.validate.run_freerun_cycles \
  --model models/MLPL2_uncertainty_v2/exp_phase_MLPL2_uncertainty_v1/ckpt_best_free_exp_phase_MLPL2_uncertainty_v1.pth \
  --teacher validate/teacher_batches/Walk_F_teacher.json \
  --bundle raw_data/processed_data/norm_template.json \
  --pretrain-template models/pretrain_template.json \
  --npz-root raw_data/processed_data \
  --encoder-bundle models/motion_encoder_equiv_stageA.pt \
  --out debug_output/freerun_cycles/corrv1 \
  --rounds 2 \
  --depth 3 \
  --so3_corr_enable \
  --so3_corr_max_deg 20 \
  --force
```

**B2) 固定 gate 复现实验（推荐先从 `0.1` 开始）**

当你怀疑 `gate_logit` 还没学会、或者担心不同 ckpt 的默认 gate 初始化导致结果不可比时，
建议用 **强制 gate** 做对照（更“实验可控”）：

```bash
python -m train.validate.run_freerun_cycles \
  --model YOUR_CKPT.pth \
  --teacher validate/teacher_batches/Walk_F_teacher.json \
  --bundle raw_data/processed_data/norm_template.json \
  --pretrain-template models/pretrain_template.json \
  --npz-root raw_data/processed_data \
  --encoder-bundle models/motion_encoder_equiv_stageA.pt \
  --out debug_output/freerun_cycles/YOUR_OUT_DIR_gate01 \
  --rounds 2 \
  --depth 3 \
  --so3_corr_enable \
  --so3_corr_gate_force 0.1 \
  --so3_corr_max_deg 20 \
  --force
```

> 产物 JSON（`*_freerun_cycles.json`）的 `metrics_per_step` 里会包含：
> `So3OmegaDegMean` / `So3GateMean`（便于确认纠偏是否真的在 rollout 中生效）。

### B. free-run drift 指标

关注已有指标：
- `GeoDegCurve` 斜率（drift slope）
- `RootVelMAE`

新增的 corrector 诊断（ValFree 会打印/保存到 metrics JSON）：
- `So3OmegaDegCurve` / `So3OmegaDegCurveMax`
- `So3OmegaDegEnd`
- `So3OmegaDegByJoint`（用于 Top‑K）

解释：
- `So3OmegaDegEnd` 越大不一定坏：可能是模型“在努力拉回”；关键看 `GeoDegCurve` slope 是否下降、是否出现振荡。

### C. Top‑K joints 定位

ValFree 日志会追加：
- `So3TopK=bone:deg,...`

经验上，若 drift 与脚/髋相关，Top‑K 常集中在 `foot/calf/thigh/pelvis` 一类；若 Top‑K 很分散或异常固定在某个关节，优先排查：
- bone_names/输出 layout 对齐
- `J` 推断是否正确

---

## 相关代码入口

- `train/geometry.py`：`so3_exp_map`
- `train/models.py`：`EventMotionModel` 输出 `omega_hat`（zero‑init head）
- `train/training_MPL.py`：rollout correction + corr loss + `so3_corr_detach_target`
- `train/validate/run_freerun_cycles.py`：多 cycle free-run 验证脚本（支持 `--so3_corr_*`）
- `train/eval_utils.py`：ValFree 统计 `So3OmegaDegCurve` / `So3OmegaDegByJoint`
- `config/exp_phase_mpl.json`：推荐启用策略（early sanity + stage2b 激活）
