# MLPL2_uncertainty_adapt5 yaw diagnostics (generated 2025-11-23)

## Valfree metrics
| epoch | file | YawAbsDeg | RootVelMAE | GeoDeg | KeyBoneGeo | tf_ratio |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | valfree_ep001.json | 27.655 | 0.187 | 7.282 | 14.471 | 1.000 |
| 2 | valfree_ep002.json | 25.926 | 0.172 | 5.022 | 9.209 | 0.944 |
| 3 | valfree_ep003.json | 24.928 | 0.171 | 3.220 | 5.767 | 0.889 |
| 4 | valfree_ep004.json | 27.799 | 0.193 | 3.367 | 5.653 | 0.775 |
| 5 | valfree_ep005.json | 28.778 | 0.182 | 2.756 | 4.952 | 0.767 |
| 6 | valfree_ep006.json | 22.974 | 0.183 | 2.970 | 5.342 | 0.758 |

## Freerun debug summary
聚焦核心的预测贴合度与稳定性指标：

| file | YawAbsDeg | GeoDeg | RootVelMAE | mean_delta_norm |
| --- | --- | --- | --- | --- |
| freerun_diag_ep002_exp_phase_MLPL2_uncertainty_adapt5_ep001.pt | 28.443 | 2.808 | 0.676 | 0.298 |
| freerun_diag_ep002_exp_phase_MLPL2_uncertainty_adapt5_ep002.pt | 28.956 | 2.732 | 0.701 | 0.338 |
| freerun_diag_ep002_exp_phase_MLPL2_uncertainty_adapt5_ep003.pt | 24.225 | 2.517 | 0.644 | 0.341 |
| freerun_diag_ep002_exp_phase_MLPL2_uncertainty_adapt5_ep004.pt | 32.964 | 3.083 | 0.709 | 0.347 |
| freerun_diag_ep002_exp_phase_MLPL2_uncertainty_adapt5_ep005.pt | 33.206 | 2.808 | 0.676 | 0.343 |
| freerun_diag_ep002_exp_phase_MLPL2_uncertainty_adapt5_ep006.pt | 25.189 | 2.521 | 0.601 | 0.340 |

> **注**：GeoDeg、RootVelMAE 来源于自由回圈诊断摘要，`mean_delta_norm` 用于衡量 carry 输出能量。如需更细粒度的稳定性指标（如 `Diag/YawSlope`、`Diag/DeltaEnergy*`），可直接查阅对应的 `debug_output/*.pt`。
