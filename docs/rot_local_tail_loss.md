## rot_local tail loss (CVaR / top-k) for per-bone stability

### Motivation

在训练中经常会出现典型的 *whack-a-mole* 现象：当我们通过 per-bone reweighting 把某个差骨骼压下去时，下一轮最差骨骼会“换人”。
这在均值归一化的权重更新（“给 A 加权就会相对压低 B”）下尤其明显，导致误差分布不稳定、尾部波动大。

本项目引入一个更可解释的目标：在不牺牲整体收敛的情况下，显式压低“最差骨骼尾部风险”。

### Definition

我们在 `rot_local`（parent-relative 的 geodesic / GeoLocalDeg）损失中增加 tail 项：

- 先计算每个骨骼的局部测地误差 `geo_local[..., j]`（rad）
- 计算 per-bone mean：`e_j = mean_{B,T}(geo_local[..., j])`
- 取 top-k 最差骨骼索引 `S = topk(e, k)`
- tail loss：
  - `L_tail = mean_{B,T,j∈S}(geo_local[..., j])`
- 最终：
  - `L = ... + w_rot_local * L_local + rot_local_tail_weight * L_tail`

其中 `L_local` 仍然保留（全骨骼平均），tail 只是对最差骨骼额外施压，因此更接近“CVaR / top-k 风格的尾部风险最小化”，不依赖手工 per-bone 权重表。

### Why use GeoLocalDeg (local) for tail?

`GeoLocalDeg` 基于 parent-relative rotation，能最大限度排除 root alignment / root drift 的干扰，更贴近“逐骨骼姿态质量”的改善目标。

### Recommended setting (KeyBone=13)

对于 KeyBone 数量约为 13 的 rig，推荐：

- `rot_local_tail_k = 3`（≈ 23% tail，避免 top-1 的 membership 抖动）
- `rot_local_tail_weight = 0.1 ~ 0.3`（从小到大试；若观测到整体均值变差则减小）

并建议将 tail 的选择范围限制在 KeyBones 上（而不是全骨骼），避免 `k=3` 在 40~60 骨骼时变成“过窄尾部”：

- `rot_local_tail_scope = keybones`（pelvis + limb_monitor_names，合计 13 左右）

### Interactions with metric-driven per-bone reweighting

本项目中还存在一套“根据 teacher 指标跨 epoch 更新骨骼权重”的机制（`update_bone_weights_from_metrics`）。
该机制在 `alpha /= alpha.mean()` 的归一化下可能加剧 whack-a-mole。

当启用 tail loss 时，建议先关闭 metric-driven reweighting 来隔离变量：

- CLI：`--disable_bone_metric_reweight`
- 或在 stage schedule 的 `params` 中设置 `disable_bone_metric_reweight=true`

### How to enable via stage schedule

在 `config/exp_phase_mpl.json` 的 `freerun_stage_schedule[*].loss_groups.core` 中加入：

- `rot_local_tail_weight`
- `rot_local_tail_k`
- `rot_local_tail_scope`（推荐 `keybones`）

并在对应 stage 的 `params` 中加入：

- `disable_bone_metric_reweight: true`

建议在早期纯收敛阶段（例如 ep1-6）不启用 tail loss，避免干扰基础收敛；在后续 fine-tune / expose 阶段启用以压低尾部误差和抑制分布振荡。
