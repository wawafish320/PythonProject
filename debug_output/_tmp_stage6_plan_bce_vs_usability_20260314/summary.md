# 2026-03-14 Stage6 contacts_plan BCE vs usability quick check

Primary mask: `cycle>=1 && !wrap_boundary_step`

| lane | plan_bce | plan_mae | stance_acc@0.7 | |logit| mean | |Δprob| mean | Stage6 all_ex_root | leg | nonleg |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `cp015_with_old_planstack` | 0.611586 | 0.362840 | 0.800000 | 0.369525 | 0.008810 | 0.295533 | 0.740703 | 0.199280 |
| `rollback_planner_core` | 0.633419 | 0.366485 | 0.530000 | 0.505457 | 0.006499 | 0.305250 | 0.766829 | 0.205449 |
| `old_bestfree` | 0.611716 | 0.362915 | 0.803333 | 0.372866 | 0.009323 | 0.313279 | 0.874230 | 0.191993 |
| `cp015_bestfree` | 0.877855 | 0.390883 | 0.506667 | 1.566154 | 0.003717 | 0.431377 | 1.167171 | 0.272286 |

Quick reads:

- `old_bestfree` vs `cp015_bestfree`: old is better on both BCE (0.611716 < 0.877855) and Stage6 usability (0.313279 < 0.431377).
- `old_bestfree` vs `rollback_planner_core`: old has lower BCE (0.611716 < 0.633419) but worse Stage6 all_ex_root (0.313279 > 0.305250).
- `old_bestfree` vs `cp015_with_old_planstack`: BCE is almost tied (0.611716 vs 0.611586), but Stage6 all_ex_root/leg still favor `cp015_with_old_planstack` (0.295533/0.740703 vs 0.313279/0.874230).
- `cp015_bestfree` keeps much larger |logit| (1.566154) but much smaller |Δprob| (0.003717), which is consistent with a badly calibrated / overly sticky plan signal.
