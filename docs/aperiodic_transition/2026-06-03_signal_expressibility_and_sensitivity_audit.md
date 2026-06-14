> TRIAGE: KEEP-AS-PROVENANCE. Historical evidence / anti-relitigation note; preserve to explain why this path was rejected or bounded. Do not use as a live plan unless a new decision record explicitly reopens it.

# Signal Expressibility and Sensitivity Audit

Date: 2026-06-03

Status: GT/read-only audit + representation contract note. No model training, no production
Trainer/runtime/gate forward or edit, no checkpoint mutation, no residual head, no endpoint/yaw
or discriminator continuation.

Primary artifact:
- `debug_output/_tmp_action_handoff_signal_representation_audit_20260603/signal_representation_audit_summary.md`
- `debug_output/_tmp_action_handoff_signal_representation_audit_20260603/signal_representation_audit_summary.json`
- `debug_output/_tmp_action_handoff_signal_representation_audit_20260603/signal_representation_audit_rows.csv`

## 1. Scope / Non-goals

本轮只回答 action-handoff inbetween 的 signal expressibility 和 perturbation sensitivity
gate，不训练 generator，不继续调 flat loss，不把 flat decoder failure 写成 diffusion required，
也不预设 support-foot-anchor 是正确答案。

Audit 输入是现有 GT windows 和现有 debug acceptance helper。评估张量均在 `cpu` 上以
`float32` 处理：`state281 [B,H,281]`、soft contact `[B,H,2]`、FK foot world pos
`[B,H,2,3]`、`bone_angvel [B,H,138]`。本轮默认 `B=188` matched windows、`H=16`，
其中 support switch windows `100`。

非目标：
- 不改 `train/` production owner。
- 不 forward production runtime/trainer/gate。
- 不改 checkpoint。
- 不做 residual head。
- 不继续 endpoint / yaw / discriminator 仪器。
- yaw/`cond_dir` 只作为 commanded cue。
- hidden/carry/latent 只允许作为 witness，不作为 success metric。

## 2. Current Evidence Reframe

当前证据应重写为 conditioning / perturbation sensitivity 问题，而不是 expressibility
问题。

- `flat_only` 单窗 exact + acceptance pass：`state mse=0.0`、train accept `1.0000`，
  说明 optimizer、standardizer、MLP 本身不是根因。
- `flat + root_vel/root_pos/foot_vel` 单项失败：`flat_plus_root_vel`、`flat_plus_root_pos`、
  `flat_plus_foot_vel` 单窗 train accept 都是 `0.0000`，冲突集中在 root/foot grounding
  派生路径。
- Oracle-schedule smoke 的 reconstructed GT guard：matched windows `188`，
  reconstructed GT train/test accept `1.0000`；flat decoder train/test accept `0.0000`。
- support topology 仍是 coverage/granularity-bound：`16` 个 split-topology rows 都是
  `granularity_fragment`，unique unseen topologies `12`，`true_new_support_mode=0`。
- learner ablation 不是 diffusion 证据：true learner train top1 多数 `0.9915..1.0000`，
  blocked/leave-clip test 仍低，decision 是
  `data_coverage_insufficient_expand_clips_no_generator`。

结论：flat 和 anchored/lifted 理论上都可以是无损重参数化；当前 perturbation 数只作为
per-frame independent Gaussian / high-frequency sensitivity diagnostic，不是 anchored-vs-flat
conditioning verdict。真正需要先隔离的是 lifted signal 到 acceptance path 的 exactness contract；
fair perturbation gate 需要 native-space correlated/bias noise、equal reconstructed-`state281`
MSE 标定，以及 position-side / velocity-side 双侧指标。

## 3. Signal Taxonomy

Signals 分四类：

- Commanded / available cue：`cond_dir`、yaw-rate command、start ctx。它们可作为条件，不是预测成功目标。
- Oracle / coverage-bound schedule：soft contact、support label/topology/timing、event phase。
  本轮可用作 GT oracle；runtime causal availability 仍受 coverage/granularity 限制。
- Derived diagnostic：FK foot world pos、foot slip、support-side features、`bone_angvel` rate witness。
  这些用于 acceptance，不应作为 future GT condition 泄漏给 decoder。
- Candidate representation variable：root pos、root vel、root-relative-to-anchor、support-foot anchor
  transform、inter-anchor / footstep placement。它们必须先能 reconstruct acceptance-grade seq，
  再谈 decoder。

## 4. Signal Ledger

| signal | source | shape / dtype / device | causal availability | derived from what | acceptance role | leakage risk | schema status | reconstructability expectation | perturbation sensitivity expectation |
|---|---|---|---|---|---|---|---|---|---|
| soft contact | GT `state281[:,279:281]` / oracle schedule | `[B,H,2] float32 cpu` | available as oracle in audit; runtime schedule remains coverage-bound | state281 contact channels | support_honesty, support token equality, endpoint proxy | future contact as condition is oracle leakage | canonical state281 field | reconstructed-domain pass expected | threshold crossings can change support masks |
| support label / topology / timing | support contract over soft contact | token `[B,H]` or one-hot `[B,H,4] float32 cpu` | predictable-but-coverage-bound: `16` granularity fragments, `12` unseen topologies, true new mode `0` | normalized contact labels and runs | support_side_correctness, switch contract | future topology/timing leaks if consumed as condition | debug contract / candidate layer-1 output | must preserve first/last and timing | off-by-one can flip anchor side |
| FK foot world pos | FK helper over pose/root | `[B,H,2,3] float32 cpu` | diagnostic-only after reconstruction | rot6d `[B,H,276]` + root_pos `[B,H,3]` + skeleton | foot slip p95/ratio, support-side bands | future FK target condition leaks | derived diagnostic | must use same FK path as acceptance | root/pose errors directly move planted foot |
| support-foot anchor transform | candidate anchored representation | `[B,R,3] + [B,H] side`, float32/int64 cpu | oracle in audit; runtime requires predicted schedule/placement | support side + FK foot world | root/support grounding | footstep copied from future GT is leakage | candidate lifted contract | should pass only with enough root/anchor fields | root-relative perturb can be as sensitive as root-position |
| inter-anchor / footstep placement | candidate anchor deltas | `[B,R-1,3] float32 cpu` | oracle in audit; must be predicted/conditioned causally later | successive support anchors | cross-switch root path | future foot placement leakage | candidate contract field if drop fails | dropping should expose switch root-path error | drop-arm should create root displacement |
| root pos / root vel / root-relative-to-anchor | raw processed + lifted transforms | `[B,H,3]`, `[B,H,2]`, `[B,H,3] float32 cpu` | root_vel canonical output; root_pos/root-relative are lifted candidates | state281 ego_vel integration or anchor + relative root | command_response, rate_budget, FK support | root path copied from GT is oracle | root_vel canonical; lifted candidate fields | must pass acceptance, not just root MSE | per-frame position noise creates derivative sensitivity |
| local pose rot6d / pose delta | state281 pose prefix | `[B,H,276]`, `[B,H-1,276] float32 cpu` | decoder output variable; GT-only here | `bone_rot6d.reshape(H,46*6)` | pose_continuity and FK local foot offset | future pose condition leaks | canonical state281 field | finite rot6d/FK path required | pose noise can move support foot unless compensated |
| event phase | run phase over support labels | `[B,H,2] float32 cpu` | oracle if from future schedule; coverage-bound if predicted | normalized support runs | switch timing condition | future event timing leak | candidate condition, not metric | must cover switch windows | phase shift can select wrong anchor |
| bone_angvel | raw `bone_ang_vel` | `[B,H,138] float32 cpu` | diagnostic/aux witness | processed skeleton angular velocity | regime/rate witness | future dynamics condition leaks | optional aux/witness, not state281 | needed by acceptance family, not standalone success | rate spikes can fail rate_budget |
| GRU hidden/carry / latent | model internals / z probes | runtime-dependent float32 device | diagnostic-only unless causal + stable + runtime-recoverable | recurrent state or latent probes | witness only | can package hidden proximity as fake success | not representation contract | cannot reconstruct seq by itself | excluded from gate metric |

Soft contact、support tokens、GRU hidden/carry 是三类不同对象，不能混为 schedule success。

## 5. Acceptance-grade Reconstructability

Reconstructability 不是 MSE-grade。本轮每个 representation 都执行：
signal -> reconstruct state/seq -> existing reconstructed-domain acceptance path -> pass/fail。

| representation | n | switch n | max abs state delta | accept | support_side | support_honesty | command | pose | rate | foot ratio | root p95 err m | support foot disp p95 m |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| flat_state281 | 188 | 100 | 0.000000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.7356 | 0.000000 | 0.000000 |
| root_position_lifted | 188 | 100 | 0.097438 | 0.0000 | 0.0053 | 1.0000 | 0.0000 | 1.0000 | 0.9681 | 0.7356 | 0.000000 | 0.000000 |
| support_anchor_keep_inter_anchor | 188 | 100 | 0.097438 | 0.0000 | 0.0053 | 0.9840 | 0.0000 | 1.0000 | 0.9574 | 0.7356 | 0.000000 | 0.000000 |
| support_anchor_drop_inter_anchor | 188 | 100 | 47.067451 | 0.0000 | 0.0053 | 0.9468 | 0.0000 | 1.0000 | 0.7500 | 0.9581 | 0.145956 | 0.145557 |

解读：

- `flat_state281` 是 acceptance-grade reconstructability：accept `1.0000`，switch accept
  `1.0000`，max delta `0.0`。
- `root_position_lifted` 和 `support_anchor_keep_inter_anchor` 的 root path / support-foot world
  displacement 都近似 0，但 acceptance 仍为 `0.0000`。失败主因不是 root path expressibility，
  而是 root-position-to-root-velocity reconstruction 后 `command_response=0.0000`、
  `support_side_correctness=0.0053`。
- `support_anchor_drop_inter_anchor` 产生 cross-switch root path error：root p95 error
  `0.145956` m，support-foot displacement p95 `0.145557` m，说明去掉 inter-anchor / footstep
  placement 会破坏跨 switch root path；但由于 keep-arm 还没有 acceptance-grade，这个字段尚不能作为
  clean pass/fail contract 宣告 solved。

## 6. Perturbation Sensitivity

Noise levels 是 per-channel MSE level，注入 Gaussian std=`sqrt(level)`。本轮扰动对象：
flat 扰动 `state281` ego velocity；root-position lifted 扰动 per-frame root XY；anchored 扰动
per-frame root-relative-to-anchor XY；drop-arm 移除 inter-anchor placement。

| representation | mse | std | accept | support_honesty | rate | foot ratio | root p95 err m | support foot disp p95 m |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| flat_velocity_state281 | 1e-4 | 0.010000 | 0.0000 | 0.9060 | 0.7323 | 0.7360 | 0.000914 | 0.000866 |
| flat_velocity_state281 | 1e-3 | 0.031623 | 0.0000 | 0.8759 | 0.0142 | 0.7375 | 0.002885 | 0.002764 |
| flat_velocity_state281 | 1e-2 | 0.100000 | 0.0000 | 0.8316 | 0.0000 | 0.7470 | 0.009282 | 0.008839 |
| root_position_lifted | 1e-4 | 0.010000 | 0.0000 | 0.3475 | 0.0000 | 1.0716 | 0.028863 | 0.027859 |
| root_position_lifted | 1e-3 | 0.031623 | 0.0000 | 0.0213 | 0.0000 | 2.2342 | 0.092092 | 0.089033 |
| root_position_lifted | 1e-2 | 0.100000 | 0.0000 | 0.0018 | 0.0000 | 6.3251 | 0.291450 | 0.282238 |
| support_anchor_keep_inter_anchor | 1e-4 | 0.010000 | 0.0000 | 0.3493 | 0.0000 | 1.0651 | 0.029181 | 0.028150 |
| support_anchor_keep_inter_anchor | 1e-3 | 0.031623 | 0.0000 | 0.0230 | 0.0000 | 2.2470 | 0.092942 | 0.089601 |
| support_anchor_keep_inter_anchor | 1e-2 | 0.100000 | 0.0000 | 0.0053 | 0.0000 | 6.4791 | 0.293723 | 0.285371 |
| support_anchor_drop_inter_anchor | 1e-4 | 0.010000 | 0.0000 | 0.3156 | 0.0000 | 1.2890 | 0.170969 | 0.169756 |
| support_anchor_drop_inter_anchor | 1e-3 | 0.031623 | 0.0000 | 0.0177 | 0.0000 | 2.4355 | 0.226018 | 0.223213 |
| support_anchor_drop_inter_anchor | 1e-2 | 0.100000 | 0.0000 | 0.0018 | 0.0000 | 6.6301 | 0.401961 | 0.394235 |

Perturbation 结论：

- 所有 perturbed arms 的 acceptance_proxy_pass_rate 都是 `0.0000`，这说明当前 acceptance
  exactness / calibration 过于敏感，不能直接拿来批准 decoder 训练。
- 本表的 noise 是 per-frame independent Gaussian，是高频噪声模型。它被 flat 的
  velocity->root_pos 积分路径低通，因此 flat root/support-foot displacement 较小：
  `1e-3` 为 `0.002885` m / `0.002764` m。
- 同一高频噪声会被 root-position / root-relative-to-anchor 的 finite-diff root_vel
  重建路径放大，因此 `1e-3` 时 foot ratio 是 `2.2342` / `2.2470`，support_honesty
  只有 `0.0213` / `0.0230`。
- 这些数字不能作为 anchored conditioning 的反证，也不能作为 anchored 稳定性的证明。
  它们只说明当前 independent high-frequency noise model 会放大 lifted 的 velocity-side
  acceptance 量。公平 conditioning gate 仍未测。

这不是“anchored 不能表达”或“flat 能表达”的结论；它只说明在当前重建和扰动定义下，
acceptance-critical 量对 lifted/root-relative high-frequency perturbation 更敏感，并且
command/support-side exactness 阻断了 reconstructability gate。真正的 perturbation gate 必须先等
GT-only lifted reconstructability 达到 acceptance-grade，再用 native-space correlated/bias noise 和
equal reconstructed-`state281` MSE calibration 重测，同时报告 position-side 与 velocity-side 指标。

## 7. Cross-switch / Inter-anchor Requirement

跨 switch 覆盖是本轮必须项：matched windows `188`，switch windows `100`。

Inter-anchor drop 负控给出明确 root path 破坏：

- keep-inter-anchor: root p95 error `0.000000` m，support-foot displacement p95 `0.000000` m，
  但 accept `0.0000`，原因是 root_vel/command/support-side exactness。
- drop-inter-anchor: root p95 error `0.145956` m，support-foot displacement p95 `0.145557` m，
  max abs state delta `47.067451`。

因此 inter-anchor / footstep placement 必须进入候选 representation contract；但在 root-position-to-root-velocity
acceptance exactness 未解决前，不能把 keep/drop 结果写成 anchored decoder 已通过。

## 8. Causal Availability Notes

Support topology/timing 仍不能写成 solved：

- topology coverage audit：`16` split-topology rows 都是 `granularity_fragment`，unique unseen
  topologies `12`，`true_new_support_mode=0`，当前不允许降 topology granularity。
- learner ablation：train top1 多数接近 `1.0`，但 contiguous/leave-clip tests 低且 unseen topology
  fraction 高；primary decision 是 `data_coverage_insufficient_expand_clips_no_generator`。
- Layer 1 结论不是 diffusion required，也不是 topology solved。

GRU hidden/carry/latent 仍只能是 witness：没有 causal + stable + runtime-recoverable 证明时，不作为
representation success metric。`bone_angvel [B,H,138] float32 cpu` 是 regime/rate witness，不是
state281 schema 字段。Yaw/`cond_dir` 只保留 commanded cue 身份。

## 9. Decision Boundary

本轮判据应用如下：

- flat 和 anchored 都 acceptance-grade reconstruct GT：不满足。flat 是 `1.0000`，anchored keep 是
  `0.0000`，但 root path error 约 0，暴露的是 acceptance exactness / root_vel reconstruction contract。
- anchored 需要 inter-anchor placement 且去掉后 fail：root path 层面满足，drop root p95 error
  `0.145956` m；但 keep-arm acceptance 未过，不能作为完整 contract pass。
- anchored 在同等扰动下比 flat 稳：未测试。当前 `1e-3` independent high-frequency noise 下
  flat foot ratio `0.7375`，anchored keep `2.2470`；flat support-foot displacement `0.002764` m，
  anchored keep `0.089601` m。但这不是 equal-state-MSE / correlated-noise apples-to-apples gate。
- 所有 representation 对 acceptance perturbation 极敏感：满足。应先审 acceptance exactness /
  calibration，不训练 decoder。
- support topology/timing causal availability 仍 coverage-bound：满足。Layer 1 不得降粒度或宣称 solved。

## 10. Final Position

不允许进入 anchored/lifted decoder toy smoke。

本轮证据支持的最小下一步不是训练 decoder，而是先定义并验证 lifted representation 的
root-position-to-root-velocity / command_response / support_side_correctness exactness contract：
同一个 root path 在 reconstructed-domain 下应当如何携带或派生 root_vel，哪些 feature bands 应该对
finite-difference root_vel 保持 acceptance-grade。只有 flat 和 lifted/anchored 都能 reconstruct GT 到
acceptance-grade，perturbation sensitivity 对比才有资格作为 decoder gate。

更新边界：后续 perturbation gate 必须使用时间相关/有偏噪声，注入各自 native prediction space，
并按 equal reconstructed-`state281` MSE 标定幅度；否则 independent high-frequency noise 会系统性
偏向 flat 的积分路径、惩罚 lifted 的 finite-diff 路径，不能作为 conditioning 判决。
