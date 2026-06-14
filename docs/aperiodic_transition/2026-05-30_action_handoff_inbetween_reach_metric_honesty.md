> TRIAGE: KEEP-AS-PROVENANCE. Historical evidence / anti-relitigation note; preserve to explain why this path was rejected or bounded. Do not use as a live plan unless a new decision record explicitly reopens it.

# Action-Handoff In-Betweening Reach Metric Honesty (W0)

Date: 2026-05-30

Status: **UPDATED by W1a**, read-only honesty probes. W0 ran no training; W1a reran the same
PHASE2 hyperparams only to add state persistence, then replayed exact trained B/C from the
saved state.

Probe artifact:

- JSON: `debug_output/_tmp_action_handoff_inbetween_reach_honesty_20260530/reach_honesty_probe_summary.json`
- MD: `debug_output/_tmp_action_handoff_inbetween_reach_honesty_20260530/reach_honesty_probe_summary.md`
- Tool: `tools/run_action_handoff_inbetween_reach_honesty_probe.py`

W1a exact trained artifacts:

- PHASE2 state rerun:
  `debug_output/_tmp_action_handoff_inbetween_phase2_guarded_finetune_20260530_w1a_state/`
- Saved state:
  `debug_output/_tmp_action_handoff_inbetween_phase2_guarded_finetune_20260530_w1a_state/phase2_trained_state.pt`
- Exact trained replay:
  `debug_output/_tmp_action_handoff_inbetween_reach_honesty_20260530_w1a_exact_trained/reach_honesty_probe_summary.json`

Important limitation correction: W0 B/C below are **not PHASE2 trained-model evidence**. They
are no-goal base-checkpoint旁证 only. A is the PHASE2 aggregate翻案证据. The exact PHASE2
trained free rollout has now been measured by W1a from saved `model.state_dict` +
`goal_head.state_dict`; its result is reported in the W1a addendum below.

## W1a Exact PHASE2 Trained Replay Addendum

G0 state round-trip passed exactly: loading
`phase2_trained_state.pt` and replaying the PHASE2 pinned context-AR §6 gate reproduced
`phase2_guarded_finetune_summary.json` with `abs_tol=1e-5`, `max_abs_delta=0.0`.
`Walk_L_To_R` matched `reach_min_norm_min=1.326910`, `reach_min_norm_mean=1.429873`,
`reach_rate=0.75`, `pop_safe_rate=0.0`, `mean_best_pose_d=0.116589`.

G1 trained self-reach calibration passed: each turn clip still self-reaches its own anchor
under the fine-tuned base. Context-window self min_norm / self abs:

| target | context self min_norm | context self abs | reached |
|---|---:|---:|---|
| Walk_L_To_L | 1.491 | 0.004947 | True |
| Walk_L_To_R | 0.584 | 0.006823 | True |
| Walk_R_To_L | 0.820 | 0.004081 | True |
| Walk_R_To_R | 1.130 | 0.008546 | True |

G2 exact trained free-rollout result (`pinned` = Walk_F cond/contact + trained goal;
`free` = target cond + model `contacts_plan` self-carry + trained goal):

| target | pinned k2/k3/k5 | free k2/k3/k5 | pinned radius reach | free radius reach | free heading MAE deg | free corr | free pop_safe |
|---|---:|---:|---:|---:|---:|---:|---:|
| Walk_L_To_L | 0.00 / 0.00 / 1.00 | 0.00 / 0.00 / 1.00 | 0.00 | 0.00 | 5.5 | -0.27 | 0.00 |
| Walk_L_To_R | 0.00 / 1.00 / 1.00 | 0.00 / 0.00 / 1.00 | 0.75 | 0.00 | 39.6 | -0.48 | 0.00 |
| Walk_R_To_L | 0.00 / 0.55 / 1.00 | 0.00 / 0.00 / 1.00 | 0.00 | 0.00 | 32.6 | 0.96 | 0.00 |
| Walk_R_To_R | 0.00 / 0.85 / 1.00 | 0.00 / 1.00 / 1.00 | 0.00 | 0.00 | 20.0 | -0.87 | 0.00 |

G3裁决: hidden_pre self-reach 是必要非充分，因为 injection 仍直写 hidden_pre。
`Walk_L_To_R` 在 exact trained target-cond/self-contact free rollout 下没有满足动作层条件:
`k=3` self-reach rate 从 pinned `1.00` 掉到 free `0.00`; realized yaw corr `-0.48`;
heading MAE 只从 `43.5°` 到 `39.6°`，没有显著下降；`pop_safe_rate=0.00`。
因此 **B4/seam remains blocked**.

## A. Absolute Self-Reach Gate

New gate: pass iff `generated_abs_cos <= k * self_reach_abs_cos`, where `self_reach_abs_cos = PHASE2 fullseq_self_min_norm * anchor_radius`. Reported for `k in {2,3,5}`. Since PHASE2 did not save per-start hidden arrays, this table uses the most favorable available PHASE2 aggregate: best-start `generated_abs_cos_min`. A per-start self-reach rate remains unavailable from the saved artifact.

| target | old radius reach_rate | old pass | generated abs min | self abs floor | k=2 | k=3 | k=5 | k=5 margin | pop_safe |
|---|---:|---|---:|---:|---|---|---|---:|---:|
| Walk_L_To_L | 0.00 | False | 0.01566 | 0.00035 | False | False | False | 8.834 | 0.00 |
| Walk_L_To_R | 0.75 | True | 0.01551 | 0.00309 | False | False | False | 1.003 | 0.00 |
| Walk_R_To_L | 0.00 | False | 0.01051 | 0.00118 | False | False | False | 1.775 | 0.00 |
| Walk_R_To_R | 0.00 | False | 0.02115 | 0.00233 | False | False | False | 1.814 | 0.00 |

判定: `Walk_L_To_R` 在旧 radius gate 下 pass (`reach_rate=0.75 >= 0.70`)，但在 `k=2/3/5` 的 absolute self-reach gate 下全部 fail。结论翻转: **True**。尤其 `k=5` 已是宽松门，L_R 仍以 margin `1.003 > 1` 失败；这说明旧正结果依赖 anchor radius，而不是达到该 clip 自己的 self-reach floor 附近。

## B. Pinned vs Free Reach

This is the read-only base-checkpoint/no-goal-head rollout:

- pinned: Walk_F cond + Walk_F contact, no goal;
- free: target turn cond trajectory, future contacts from model `contacts_plan` self-carry, no goal;
- seed context still uses Walk_F motion/contact history because the task starts from arbitrary Walk_F phase.

| target | pinned min_norm mean/min | free min_norm mean/min | free/pinned mean | pinned pop_safe | free pop_safe |
|---|---:|---:|---:|---:|---:|
| Walk_L_To_L | 13.82 / 9.57 | 15.27 / 13.42 | 1.10 | 0.00 | 0.00 |
| Walk_L_To_R | 6.77 / 5.04 | 8.26 / 5.22 | 1.22 | 0.00 | 0.00 |
| Walk_R_To_L | 16.65 / 11.39 | 18.86 / 14.39 | 1.13 | 0.00 | 0.00 |
| Walk_R_To_R | 7.37 / 5.46 | 8.06 / 5.82 | 1.09 | 0.00 | 0.00 |

判定: no-goal base 下，free target-cond/self-contact reach 没有改善，反而比 pinned 更远 (`free/pinned mean = 1.09..1.22`)，且 `pop_safe=0/4`。这不是 exact PHASE2 goal-head 证据；它只说明在不训练/不加载 goal head 的只读条件下，目标 cond + generated contact 本身不能让 base 进入 turn hidden_pre anchor。

## C. Realized Yaw / Heading

Yaw realization uses generated `root_vel[:, 0:2]` heading integral as the realized motion heading. Target yaw is from the target clip `cond_dir` heading integral. The free rollout's command yaw equals the target command by construction; the question is whether generated root velocity follows it.

| target | target final yaw deg | free command final yaw deg | free realized final yaw deg | free heading MAE deg | free yaw-rate MAE rad/s | free corr |
|---|---:|---:|---:|---:|---:|---:|
| Walk_L_To_L | -6.7 | -6.7 | -38.7 | 8.7 | 0.54 | 0.83 |
| Walk_L_To_R | 76.8 | 76.8 | -60.4 | 53.2 | 2.05 | -0.76 |
| Walk_R_To_L | -59.6 | -59.6 | -47.4 | 19.3 | 0.86 | 0.95 |
| Walk_R_To_R | 16.1 | 16.1 | -49.4 | 22.4 | 0.96 | -0.90 |

判定: motion-space turn statistic does not support the PHASE2 L_R positive. L_R is the clearest failure: command/GT asks for `+76.8 deg`, generated root-velocity heading integrates to `-60.4 deg`, heading MAE `53.2 deg`, correlation `-0.76`. R_L has a better yaw correlation but still has no hidden reach and no pop-safe seam in B.

## Verdict

PHASE2 的 `Walk_L_To_R` 旧 radius 门正结果在诚实度量下 **不成立**。A flips L_R from
pass to fail under the original fullseq absolute self-reach-relative check (`k=2/3/5` all
fail). W0 B/C are retained only as no-goal base旁证, not PHASE2 trained-model evidence.

W1a closes the remaining exact-trained口子: the saved PHASE2 state round-trips exactly, the
fine-tuned model still self-reaches its own anchors, but target-cond/self-contact free rollout
does not carry the L_R latent reach or the motion. L_R free `k=3` self-reach rate is `0.00`
vs pinned `1.00`, realized-yaw corr is `-0.48`, heading MAE remains `39.6°`, and
`pop_safe_rate=0.00`.

Decision for B4/seam: **blocked**. Hidden_pre self-reach is necessary but not sufficient; the
trained free rollout fails the required action-layer checks.
