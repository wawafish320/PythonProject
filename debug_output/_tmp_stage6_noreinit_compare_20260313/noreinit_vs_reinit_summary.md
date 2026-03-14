# Stage6 no-reinit vs reinit

- config delta: `direct_pose_reinit=true -> false` only
- note: runtime log still reports `dropped 10 direct_pose_* tensors from checkpoint (reinit/override)` for both lanes

| lane | exit all_ex_root delta | exit leg delta | exit nonleg delta | step1 grad arm/else delta | result |
|---|---:|---:|---:|---:|---|
| old_bestfree | +0.000000 | +0.000000 | +0.000000 | +0.000000 | identical |
| cp015_bestfree | +0.000000 | +0.000000 | +0.000000 | +0.000000 | identical |

