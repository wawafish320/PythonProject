> TRIAGE: KEEP-AS-PROVENANCE. Historical evidence / anti-relitigation note; preserve to explain why this path was rejected or bounded. Do not use as a live plan unless a new decision record explicitly reopens it.

# Action Handoff z Probe Closeout / Decision Record (P0-P6)

Date: 2026-05-26

Status: closeout decision record

Scope: research/probe closeout, not production sign-off

本记录覆盖 P0-P6 全链路。它关闭的是 research/probe 阶段的判断，不是 runtime route 的 production sign-off。v1 设计原文已经声明初始设计不是 normative contract，预期随 P0-P6 结果修订（docs/aperiodic_transition/2026-05-24_action_handoff_predictive_contrastive_z_probe_design.md:3, docs/aperiodic_transition/2026-05-24_action_handoff_predictive_contrastive_z_probe_design.md:4）。

主要依据：

- v1 probe 设计、P0-P6 gate 定义、P4-alt 重新定义和 P6 priority：docs/aperiodic_transition/2026-05-24_action_handoff_predictive_contrastive_z_probe_design.md:145, docs/aperiodic_transition/2026-05-24_action_handoff_predictive_contrastive_z_probe_design.md:156, docs/aperiodic_transition/2026-05-24_action_handoff_predictive_contrastive_z_probe_design.md:210, docs/aperiodic_transition/2026-05-24_action_handoff_predictive_contrastive_z_probe_design.md:240
- P0 preflight / MM oracle / energy baseline：debug_output/_tmp_action_handoff_p0_preflight_20260524/preflight_summary.md:54, debug_output/_tmp_action_handoff_p0_preflight_20260524/preflight_summary.md:89
- z v1 feature extraction、P1、legacy P4：debug_output/_tmp_action_handoff_z_probe_v1_20260524/summary.md:7, debug_output/_tmp_action_handoff_z_probe_v1_20260524/summary.md:48
- P2/P3 internal structure v2：debug_output/_tmp_action_handoff_z_probe_v1_internal_structure_v2_20260524/internal_structure_v2_summary.md:7, debug_output/_tmp_action_handoff_z_probe_v1_internal_structure_v2_20260524/internal_structure_v2_summary.md:18
- P4-alt stability sweep：debug_output/_tmp_action_handoff_z_probe_v1_p4_alt_sweep_20260524/p4_alt_sweep_summary.md:7, debug_output/_tmp_action_handoff_z_probe_v1_p4_alt_sweep_20260524/p4_alt_sweep_summary.md:53
- Walk_L_To_R weak-source analysis：debug_output/_tmp_action_handoff_z_probe_v1_walk_l_to_r_failure_analysis_20260524/walk_l_to_r_failure_analysis.md:5, debug_output/_tmp_action_handoff_z_probe_v1_walk_l_to_r_failure_analysis_20260524/walk_l_to_r_failure_analysis.md:69
- P6 full-matrix injected smoke：debug_output/_tmp_action_handoff_p6_synthetic_boundary_eval_20260525_runner_injected_full_matrix_v3/p6_synthetic_boundary_eval_summary.md:15, debug_output/_tmp_action_handoff_p6_synthetic_boundary_eval_20260525_runner_injected_full_matrix_v3/p6_synthetic_boundary_eval_summary.md:51
- P6 provisional acceptance eval：debug_output/_tmp_action_handoff_p6_acceptance_eval_20260526/p6_acceptance_eval_summary.md:3, debug_output/_tmp_action_handoff_p6_acceptance_eval_20260526/p6_acceptance_eval_summary.md:42
- P6 threshold/acceptance contract note 和 JSON：docs/aperiodic_transition/2026-05-26_action_handoff_z_p6_threshold_acceptance_contract_note.md:3, docs/aperiodic_transition/2026-05-26_action_handoff_z_p6_threshold_acceptance_contract_note.md:114, docs/aperiodic_transition/2026-05-26_action_handoff_z_p6_threshold_acceptance_contract.json:1, docs/aperiodic_transition/2026-05-26_action_handoff_z_p6_threshold_acceptance_contract.json:44

## 1. Title + Status

- Date: 2026-05-26
- Status: closeout decision record
- Scope: research/probe closeout, not production sign-off

## 2. Executive Decision

v1 route 不进入 production-approved 状态。当前证据支持继续下一实现阶段，但只能在 fallback-aware P6 planning 下推进。

H3 is partially supported under the recalibrated P4-alt future-equivalence yardstick.

P1 point-regression remains a magnitude-sufficiency diagnostic risk, not an H3 blocker.

weak rows are classified as known-risk fallback_required, not global framework rejection.

下一阶段 blockers：

- fallback policy implementation：P6 weak rows 已暴露需要 fallback 的 known-risk 行，不能在无 fallback 策略下推进。
- production threshold contract not calibrated on current smoke only：当前阈值是 provisional smoke thresholds / calibration-on-current-smoke，不能作为 production threshold（docs/aperiodic_transition/2026-05-26_action_handoff_z_p6_threshold_acceptance_contract_note.md:52, docs/aperiodic_transition/2026-05-26_action_handoff_z_p6_threshold_acceptance_contract_note.md:72）。
- Walk_L_To_R weak-source handling：该 source 在 P4-alt 中不是 near-random，但有 long-horizon degradation 和 target-specific weakness（debug_output/_tmp_action_handoff_z_probe_v1_walk_l_to_r_failure_analysis_20260524/walk_l_to_r_failure_analysis.md:61, debug_output/_tmp_action_handoff_z_probe_v1_walk_l_to_r_failure_analysis_20260524/walk_l_to_r_failure_analysis.md:69）。
- P1 magnitude diagnostic remains unresolved：z_bottleneck 的 point-regression Huber 弱于 energy 和 raw_hidden_pre（debug_output/_tmp_action_handoff_z_probe_v1_20260524/summary.md:21, debug_output/_tmp_action_handoff_z_probe_v1_20260524/summary.md:39）。

## 3. Probe-by-probe Summary Table

| Probe | original intent | final interpretation | key evidence | decision / status |
|---|---|---|---|---|
| P0 preflight | 建立非 neural Motion Matching / energy lower-bound 和 cross-clip oracle reference。 | energy mostly basin-level；overlap-region entry ranking weak。P0 仍是 lower-bound/context，不足以直接解决 handoff entry selection。 | energy same-clip top1=0.343243/top3=0.654054；cross-clip top1=0.343243 over 1480 queries；overlap runtime=0.237838；overlap aggregate=0.126446 over 1210 scored, 270 dropped（debug_output/_tmp_action_handoff_p0_preflight_20260524/preflight_summary.md:66, debug_output/_tmp_action_handoff_p0_preflight_20260524/preflight_summary.md:89）。 | preflight complete；supports moving to z probes but does not certify route. |
| P1 predictive point-regression | 用 matched readout 比较 energy/raw_hidden_pre/z 的 future_desc point-regression magnitude sufficiency。 | z loses magnitude point-regression；recalibration 后这是 diagnostic risk，不是 H3 blocker。 | energy=0.005601，raw_hidden_pre=0.005108，z_bottleneck=0.010920（debug_output/_tmp_action_handoff_z_probe_v1_20260524/summary.md:21, debug_output/_tmp_action_handoff_z_probe_v1_20260524/summary.md:39）。 | unresolved magnitude-sufficiency diagnostic risk. |
| P2 internal phase/structure | 检查 z 是否有 phase locality、cycle closure、low-dimensional structure。 | structured_but_mixed；single-cycle/transient data 下 old closure hard gate 有 measurement bias。 | cycle_closure_ratio=1.002566，knn.mean=0.048989，pca_2d_explained_variance=0.783396（debug_output/_tmp_action_handoff_z_probe_v1_internal_structure_v2_20260524/internal_structure_v2_summary.md:7, debug_output/_tmp_action_handoff_z_probe_v1_internal_structure_v2_20260524/internal_structure_v2_summary.md:10）。 | diagnostic pass-like on structure; not hard gate closure. |
| P3 turn end structure | 检查 turn/end monotonic convergence 和 cross-turn end tightness。 | structured_but_mixed；end convergence strong，但 end-vs-mid 仍是 diagnostic-only weakness。 | monotonic 4/4，slope<0 4/4，end_tightness_ratio=0.648635，end_vs_mid=1.351456（debug_output/_tmp_action_handoff_z_probe_v1_internal_structure_v2_20260524/internal_structure_v2_summary.md:12, debug_output/_tmp_action_handoff_z_probe_v1_internal_structure_v2_20260524/internal_structure_v2_summary.md:15）。 | payoff diagnostic supportive; not production approval. |
| legacy P4 MM/P0 overlap agreement | 原先作为 z 与 MM/P0 overlap-priority ranking 的 agreement gate。 | downgraded to secondary diagnostic；legacy overlap agreement 不能作为 H3 main gate。 | z global=0.400000 vs gate>=0.343243 pass；overlap runtime=0.258784 vs gate>=0.400000 fail；overlap aggregate=0.176033 vs gate>=0.300000 fail（debug_output/_tmp_action_handoff_z_probe_v1_20260524/summary.md:41, debug_output/_tmp_action_handoff_z_probe_v1_20260524/summary.md:45）。设计后续也明确 MM/P0-overlap agreement 是 secondary diagnostic（docs/aperiodic_transition/2026-05-24_action_handoff_predictive_contrastive_z_probe_design.md:153, docs/aperiodic_transition/2026-05-24_action_handoff_predictive_contrastive_z_probe_design.md:154）。 | rejected as H3 main gate; retained as safety-priority sanity diagnostic. |
| P4-alt future-equivalence | 用 z-neighborhood 是否预测 GT future-equivalence 作为 H3 main gate。 | future-equivalence signal is real；source-specific weakness remains。 | global 7/10 configs pass-like；N=6/12 ratio 0.798845/0.821149；hit_lift +0.336217/+0.341058；N=12 Spearman 0.592137（debug_output/_tmp_action_handoff_z_probe_v1_p4_alt_sweep_20260524/p4_alt_sweep_summary.md:7, debug_output/_tmp_action_handoff_z_probe_v1_p4_alt_sweep_20260524/p4_alt_sweep_summary.md:20）。 | H3 partially supported under recalibrated yardstick; not full approval. |
| P5 | 原设计为 horizon set、feature set、Dz、tau 的 residual stability / robustness gate。 | 未形成 standalone full P5 sign-off。当前只看到 P4-alt 维度上的 stability sweep，不能替代 feature/Dz/tau robustness。 | 设计中 P5 要求 sweep horizon set、feature set、Dz、tau（docs/aperiodic_transition/2026-05-24_action_handoff_predictive_contrastive_z_probe_design.md:155）；现有 P4-alt sweep total configs=10，主要变化 N/q/top_k（debug_output/_tmp_action_handoff_z_probe_v1_p4_alt_sweep_20260524/p4_alt_sweep_summary.md:5, debug_output/_tmp_action_handoff_z_probe_v1_p4_alt_sweep_20260524/p4_alt_sweep_summary.md:20）。 | partial stability evidence only; remains next-phase robustness requirement. |
| P6 injected full-matrix smoke | 在 synthetic boundary stress 中检查 execution coverage、canonical safety metrics、normal vs weak 区分和 fallback classification。 | smoke acceptance supports tool route and fallback-aware planning；weak rows require fallback；production threshold/pass not established。 | 8/8 rows executed；canonical metric completeness 8/8；identical_digest_pairs=0；acceptance overall_status=p6_smoke_accept_with_known_weak_fallback_required（debug_output/_tmp_action_handoff_p6_synthetic_boundary_eval_20260525_runner_injected_full_matrix_v3/p6_synthetic_boundary_eval_summary.md:15, debug_output/_tmp_action_handoff_p6_synthetic_boundary_eval_20260525_runner_injected_full_matrix_v3/p6_synthetic_boundary_eval_summary.md:51, debug_output/_tmp_action_handoff_p6_acceptance_eval_20260526/p6_acceptance_eval_summary.md:36, debug_output/_tmp_action_handoff_p6_acceptance_eval_20260526/p6_acceptance_eval_summary.md:42）。 | provisional smoke acceptance with known weak fallback_required; not production-approved. |

## 4. P0 Summary

P0 preflight 使用 5 个 locked turn-cycle clips，导出了 pose/root/contact/energy/FK diagnostic features。典型 feature shape/dtype/device 已在 preflight 中固定，例如 Walk_F pose shape=[87, 276] dtype=float32 device=cpu，root shape=[87, 2] dtype=float32 device=cpu，contact shape=[87, 2] dtype=float32 device=cpu，energy shape=[87, 1] dtype=float32 device=cpu（debug_output/_tmp_action_handoff_p0_preflight_20260524/preflight_summary.md:12, debug_output/_tmp_action_handoff_p0_preflight_20260524/preflight_summary.md:20）。其余 clips 分别为 Walk_L_To_L frames_used=54、Walk_L_To_R frames_used=50、Walk_R_To_L frames_used=86、Walk_R_To_R frames_used=93，pose/root/contact/energy 均为 float32 cpu（debug_output/_tmp_action_handoff_p0_preflight_20260524/preflight_summary.md:21, debug_output/_tmp_action_handoff_p0_preflight_20260524/preflight_summary.md:52）。

关键数字：

- energy same-clip top1=0.343243, top3=0.654054（debug_output/_tmp_action_handoff_p0_preflight_20260524/preflight_summary.md:66, debug_output/_tmp_action_handoff_p0_preflight_20260524/preflight_summary.md:69）。
- energy cross-clip global top1=0.343243 over 1480 queries（debug_output/_tmp_action_handoff_p0_preflight_20260524/preflight_summary.md:75, debug_output/_tmp_action_handoff_p0_preflight_20260524/preflight_summary.md:78）。
- overlap_restricted_runtime=0.237838（debug_output/_tmp_action_handoff_p0_preflight_20260524/preflight_summary.md:84, debug_output/_tmp_action_handoff_p0_preflight_20260524/preflight_summary.md:88）。
- overlap_restricted_aggregate=0.126446 over 1210 scored queries, 270 dropped（debug_output/_tmp_action_handoff_p0_preflight_20260524/preflight_summary.md:84, debug_output/_tmp_action_handoff_p0_preflight_20260524/preflight_summary.md:87）。

解释：energy mostly basin-level; overlap-region entry ranking weak. 它能提供 basin-level separability 和 P0/MM 参考，但 overlap-region entry selection 不够强，不能单独支撑 cross-action handoff route。

## 5. P1 Summary

z v1 使用 frozen lambda checkpoint 的 `hidden_pre` 作为 primary frozen feature，feature shapes 为 Walk_F [87, 512]、Walk_L_To_L [54, 512]、Walk_L_To_R [50, 512]、Walk_R_To_L [86, 512]、Walk_R_To_R [93, 512]，dtype=float32 device=cpu（debug_output/_tmp_action_handoff_z_probe_v1_20260524/summary.md:7, debug_output/_tmp_action_handoff_z_probe_v1_20260524/summary.md:14）。z head 为 MLP(512->256->128->32) + LayerNorm + GELU，loss 为 L_predict + 0.25*L_InfoNCE，tau=0.07，horizons=[1, 3, 6, 12, 24]（debug_output/_tmp_action_handoff_z_probe_v1_20260524/summary.md:16, debug_output/_tmp_action_handoff_z_probe_v1_20260524/summary.md:19）。

关键数字：

- energy=0.005601
- raw_hidden_pre=0.005108
- z_bottleneck=0.010920

以上均为 `test_weighted_huber` point-regression 指标；对应 repr_dim 分别为 1、512、32（debug_output/_tmp_action_handoff_z_probe_v1_20260524/summary.md:21, debug_output/_tmp_action_handoff_z_probe_v1_20260524/summary.md:39）。

解释：z loses magnitude point-regression, but this does not falsify H3 after recalibration. P1 point-regression remains a magnitude-sufficiency diagnostic risk, not an H3 blocker.

## 6. P2/P3 Recalibration Summary

P2：

- cycle_closure_ratio=1.002566
- knn.mean=0.048989
- pca_2d_explained_variance=0.783396

这些数字来自 internal_structure_v2：closure weak，但 phase locality 和 low-dimensionality strong（debug_output/_tmp_action_handoff_z_probe_v1_internal_structure_v2_20260524/internal_structure_v2_summary.md:7, debug_output/_tmp_action_handoff_z_probe_v1_internal_structure_v2_20260524/internal_structure_v2_summary.md:10）。

P3：

- monotonic 4/4
- slope<0 4/4
- end_tightness_ratio=0.648635
- end_vs_mid=1.351456

P3 表明 turn monotonic convergence 和 cross-turn end tightness strong，但 end-window variance vs mid-window 仍 weaker_than_mid（debug_output/_tmp_action_handoff_z_probe_v1_internal_structure_v2_20260524/internal_structure_v2_summary.md:12, debug_output/_tmp_action_handoff_z_probe_v1_internal_structure_v2_20260524/internal_structure_v2_summary.md:15）。

解释：structured_but_mixed; old closure/stable-cluster gates were measurement-biased for single-cycle/transient data. 因此 P2/P3 被保留为 diagnostic / precondition signal，而不是 hard reject gate。

## 7. P4 Recalibration Summary

legacy MM/P0 agreement downgraded to secondary diagnostic。原 legacy P4 中 z global agreement 通过，但 overlap_restricted_runtime 和 overlap_restricted_aggregate 均未达 gate，因此不能作为 H3 主判断（debug_output/_tmp_action_handoff_z_probe_v1_20260524/summary.md:41, debug_output/_tmp_action_handoff_z_probe_v1_20260524/summary.md:45）。设计记录也明确 P4-alt 才是 P4 main definition，MM/P0-overlap agreement 仅保留为 secondary diagnostic（docs/aperiodic_transition/2026-05-24_action_handoff_predictive_contrastive_z_probe_design.md:210, docs/aperiodic_transition/2026-05-24_action_handoff_predictive_contrastive_z_probe_design.md:217）。

P4-alt aggregate：

- global stability 7/10 configs pass-like
- top1_future_distance_vs_random_ratio around 0.80-0.82 at N=6/12：0.798845 at N=6，0.821149 at N=12
- hit_lift global around +0.29 to +0.38：configs 中从 +0.286733 到 +0.387294
- Spearman around 0.59 for N=12 anchor：0.592137

这些来自 10-config global table（debug_output/_tmp_action_handoff_z_probe_v1_p4_alt_sweep_20260524/p4_alt_sweep_summary.md:7, debug_output/_tmp_action_handoff_z_probe_v1_p4_alt_sweep_20260524/p4_alt_sweep_summary.md:20）。P4-alt design 的判断语义是 z-nearest target frames 是否有显著优于 random/oracle-expectation 的 GT future-equivalence，且 P6 仍是 integration boundary gate（docs/aperiodic_transition/2026-05-24_action_handoff_predictive_contrastive_z_probe_design.md:214, docs/aperiodic_transition/2026-05-24_action_handoff_predictive_contrastive_z_probe_design.md:239）。

Walk_L_To_R：

- pass_like_count=7/10
- near_random=0/10
- mean_ratio=0.940338
- mean_spearman=0.232160
- long horizon degradation short spearman=0.326221 -> long spearman=0.012683
- weak target pairs Walk_L_To_R->Walk_R_To_L and Walk_L_To_R->Walk_R_To_R

P4-alt summary 标记 Walk_L_To_R pass_like_count=7/10 且 near_random_count=0/10（debug_output/_tmp_action_handoff_z_probe_v1_p4_alt_sweep_20260524/p4_alt_sweep_summary.md:22, debug_output/_tmp_action_handoff_z_probe_v1_p4_alt_sweep_20260524/p4_alt_sweep_summary.md:35）。failure analysis 显示 long_horizon_degradation=yes、target_specific_weakness=yes、short_horizon_mean spearman=0.326221、long_horizon_mean spearman=0.012683，并建议后续聚焦 Walk_L_To_R->Walk_R_To_L 和 Walk_L_To_R->Walk_R_To_R（debug_output/_tmp_action_handoff_z_probe_v1_walk_l_to_r_failure_analysis_20260524/walk_l_to_r_failure_analysis.md:50, debug_output/_tmp_action_handoff_z_probe_v1_walk_l_to_r_failure_analysis_20260524/walk_l_to_r_failure_analysis.md:69）。

解释：future-equivalence signal is real but source-specific weakness remains. H3 is partially supported under the recalibrated P4-alt future-equivalence yardstick.

## 8. P6 Summary

P6 当前是 runner-invoke full-matrix injected smoke，artifact 自身声明这是 execution coverage + metric completeness check only, not pass gate（debug_output/_tmp_action_handoff_p6_synthetic_boundary_eval_20260525_runner_injected_full_matrix_v3/p6_synthetic_boundary_eval_summary.md:3, debug_output/_tmp_action_handoff_p6_synthetic_boundary_eval_20260525_runner_injected_full_matrix_v3/p6_synthetic_boundary_eval_summary.md:8）。provisional acceptance eval 的 scope 是 `provisional_smoke_acceptance`，contract_version 是 `p6_smoke_v0.1`（debug_output/_tmp_action_handoff_p6_acceptance_eval_20260526/p6_acceptance_eval_summary.md:3, debug_output/_tmp_action_handoff_p6_acceptance_eval_20260526/p6_acceptance_eval_summary.md:5）。

Execution / completeness：

- full-matrix injected smoke executed 8/8 rows
- canonical metric completeness complete 8/8
- stress differentiability observed, identical_digest_pairs=0

对应 artifact counts：total_rows=8、runner_invoke_executed_rows=8、rows_with_complete_canonical_metrics=8、rows_with_missing_canonical_metrics=0、rows_with_proxy_metrics=0（debug_output/_tmp_action_handoff_p6_synthetic_boundary_eval_20260525_runner_injected_full_matrix_v3/p6_synthetic_boundary_eval_summary.md:15, debug_output/_tmp_action_handoff_p6_synthetic_boundary_eval_20260525_runner_injected_full_matrix_v3/p6_synthetic_boundary_eval_summary.md:39）。stress audit 显示 differentiable_trace_observed、checked_pairs=4、identical_digest_pairs=0（debug_output/_tmp_action_handoff_p6_synthetic_boundary_eval_20260525_runner_injected_full_matrix_v3/p6_synthetic_boundary_eval_summary.md:47, debug_output/_tmp_action_handoff_p6_synthetic_boundary_eval_20260525_runner_injected_full_matrix_v3/p6_synthetic_boundary_eval_summary.md:51）。

normal vs weak means：

- ContactMismatchRate normal=0.38904899, weak=0.74246231
- FootSlipBallL normal=1.95297538, weak=1.05049640
- FootSlipBallR normal=1.47022423, weak=2.12872073
- RootStepDispErr normal=0.00421452, weak=0.00414479
- GeoLocalDeg normal=0.49716126, weak=11.30985872

这些均来自 P6 full-matrix smoke 的 normal/weak comparison（debug_output/_tmp_action_handoff_p6_synthetic_boundary_eval_20260525_runner_injected_full_matrix_v3/p6_synthetic_boundary_eval_summary.md:40, debug_output/_tmp_action_handoff_p6_synthetic_boundary_eval_20260525_runner_injected_full_matrix_v3/p6_synthetic_boundary_eval_summary.md:45）。

provisional acceptance：

- 4/4 normal rows normal_accept
- 4/4 weak rows weak_fallback_required_known_risk
- overall_status=p6_smoke_accept_with_known_weak_fallback_required
- production_pass_established=false

Row table 中 4 个 normal rows 均为 normal_accept，4 个 weak_stress rows 均为 weak_fallback_required_known_risk（debug_output/_tmp_action_handoff_p6_acceptance_eval_20260526/p6_acceptance_eval_summary.md:17, debug_output/_tmp_action_handoff_p6_acceptance_eval_20260526/p6_acceptance_eval_summary.md:28）。classification summary 为 normal_accept=4/4、weak_fallback_required_known_risk=4/4，overall_status 为 `p6_smoke_accept_with_known_weak_fallback_required`（debug_output/_tmp_action_handoff_p6_acceptance_eval_20260526/p6_acceptance_eval_summary.md:30, debug_output/_tmp_action_handoff_p6_acceptance_eval_20260526/p6_acceptance_eval_summary.md:42）。

阈值边界：当前 thresholds 来自 normal rows 的 max*1.10，且 contract note 明确当前状态未发现 signed threshold contract、采用当前 smoke 推导；有效性边界是仅用于 smoke 分类，不可作为 production pass 阈值（docs/aperiodic_transition/2026-05-26_action_handoff_z_p6_threshold_acceptance_contract_note.md:52, docs/aperiodic_transition/2026-05-26_action_handoff_z_p6_threshold_acceptance_contract_note.md:72）。JSON contract 同样标注 threshold_derivation 为 calibration-on-current-smoke（docs/aperiodic_transition/2026-05-26_action_handoff_z_p6_threshold_acceptance_contract.json:1, docs/aperiodic_transition/2026-05-26_action_handoff_z_p6_threshold_acceptance_contract.json:5）。

weak rows are classified as known-risk fallback_required, not global framework rejection.

## 9. What Was Rejected / What Was Not Rejected

Rejected：

- original P1 as necessary H3 gate。P1 仍暴露 magnitude-sufficiency risk，但 recalibration 后不再作为 H3 blocker。
- legacy MM/P0 agreement as H3 main gate。legacy overlap agreement 失败说明它不适合作为主 gate，但可保留为 safety-priority sanity diagnostic。
- treating single-cycle closure failure as hard P2 fail。P2 closure weak 与 phase locality / PCA structure strong 并存，说明 single-cycle/transient 数据上的 closure hard gate 有 measurement bias。
- proceeding without fallback for known weak-source rows。P6 weak rows 需要 weak_fallback_required_known_risk 分类和 runtime fallback policy。

Not rejected：

- causal-state / PSR principle。设计原文保留的是 principle：state 是能产生相同 future distribution 的 histories equivalence class，PSR 下为 action-conditioned causal states（docs/aperiodic_transition/2026-05-24_action_handoff_predictive_contrastive_z_probe_design.md:49, docs/aperiodic_transition/2026-05-24_action_handoff_predictive_contrastive_z_probe_design.md:53）。
- frozen lambda checkpoint substrate。v1 明确使用 2026-05-14 lambda-applied ckpt 作为 frozen feature substrate，z v1 artifact 也使用同一 checkpoint 提取 hidden_pre（docs/aperiodic_transition/2026-05-24_action_handoff_predictive_contrastive_z_probe_design.md:87, debug_output/_tmp_action_handoff_z_probe_v1_20260524/summary.md:7）。
- z direction as future-equivalence representation。P4-alt 显示 global pass-like stability 和 positive future-equivalence signal，但只支持 partial H3，不支持 production-approved。
- P6 standalone tool route。P6 runner-invoke full-matrix injected smoke 能执行 8/8 行并完整产出 canonical metrics；这支持 standalone tool route 继续作为下一阶段验证载体，不等于 production sign-off。

## 10. Next Phase Requirements

- fallback policy implementation for weak rows：weak_stress / Walk_L_To_R-source rows 必须有显式 fallback_required runtime policy，不能只记录 warning。
- production threshold contract not derived only from current smoke：生产阈值需要独立校准集、更多 source/target rows、signed contract 或等价可追溯依据；当前 contract 只能用于 provisional smoke classification。
- Walk_L_To_R source-specific handling：优先覆盖 Walk_L_To_R->Walk_R_To_L 和 Walk_L_To_R->Walk_R_To_R，并分别处理 long-horizon degradation 与 target-specific weakness。
- P1 magnitude-risk tracking：后续实现需要继续跟踪 point-regression magnitude risk，尤其是 z_bottleneck 对 future_desc magnitude 的损失；该风险不阻塞 H3，但阻塞“无监控上线”。
- optional future z objective improvements only after fallback-aware P6 planning, not as immediate blocker：例如 Dz/feature/tau/objective 改进应在 fallback-aware P6 planning 之后排期，不能替代 fallback policy 和 production threshold contract。

## 11. Three-binary Framing

- Was the original rollout-eval pilot sufficient to answer the cross-action handoff research question? No.
- Does the recalibrated causal-state / P4-alt direction have substantive empirical support? Yes.
- Has v1 been validated to production-ready under independent thresholds and full source coverage? No.

The final decision follows from this triple.

## 12. Methodology Lessons

本 spike 最有长期价值的发现不是单个 v1 数字，而是 measurement-bias 校正。P0-P6 的价值在于把原先看似失败的 gate 拆成了不同测量问题：magnitude regression、geometry heuristic agreement、single-cycle structure、transient end behavior、overlap-region ranking 和 boundary smoke safety。

1. P1 point-prediction loss ≠ predictive sufficiency

Crutchfield 的 minimal sufficient 是 future-equivalence / equivalence relation，不是 magnitude regression。P1 失败说明 z_bottleneck 在 point-regression Huber 上有 magnitude-sufficiency risk，但不单独证伪 H3。

2. MM oracle agreement ≠ causal-state ground truth

MM oracle 是 contact/foot/root/pose lexicographic 几何启发式。它是 safety-priority diagnostic，不是 future-distribution equivalence 主真值；因此 legacy MM/P0 agreement 应保留为 secondary diagnostic，而不是 H3 main gate。

3. Cycle closure 在 single-cycle data 上结构性不可达

只有一个 Walk_F cycle 时，没有监督信号教模型把 cycle 2 同 phase 映射回 cycle 1。cycle_closure_ratio 只能诊断 phase/locality 与 embedding geometry 的局限，不能硬判 P2 fail。

4. Transient turn end 不应要求 end-window variance < mid-window variance

transient 末段可在向 attractor 移动；此时 end-window variance 大于 mid-window variance 不必然否定结构。monotonic convergence / cross-turn tightness 更适合作为 P3 的主要 payoff diagnostic。

5. Overlap is signal, not noise

energy 区分不开 overlap region 不代表 overlap 是 noise。overlap 是 handoff 的物理前提；真正问题是 overlap 内 entry ranking，因此 P0/P4 应区分 basin-level separability 与 overlap-region ranking。

Methodology principle：

- Probe gate 的数字阈值应在 baseline run 后校准，不应在 design 阶段拍死。
- 一次 baseline/probe run 通常比继续设计讨论更可信。

## 13. Scope Guard For Next Phase

以下内容不是下一阶段 critical path，用于防止 scope creep：

- β/Dz/τ further ablation on v1 z architecture：这些是 z-objective optimization，不是 fallback-aware P6 blocker。应 defer until fallback policy / production threshold / weak-source handling 更清楚。
- multi-cycle data acquisition：不是 v1 fallback-aware P6 blocker。但它仍是未来验证 phase-invariance 的好实验，不要永久排除。
- Walk_L_To_R extra deep-dive before fallback-aware P6 stress planning：当前已知道 failure mode 是 long-horizon degradation + target-specific weakness。不需要在 P6 stress planning 前再阻塞；但 P6 stress run 必须继续显式观测它。

The original rollout-eval pilot remains useful for in-basin drift diagnostics, but it is not sufficient to answer the cross-action handoff H3 question.

## 14. Final Decision Statement

The v1 route is not production-approved, but the probe direction is justified for a next implementation phase under fallback-aware P6 planning.
