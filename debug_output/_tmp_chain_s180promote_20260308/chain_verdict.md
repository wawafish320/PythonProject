# s180 promote downstream chain verdict

## Key outcome

- Promoting the `s180 low-LR trunkfull` 70R candidate into the downstream chain **holds up cleanly**.
- Against the existing `new` chain, stage71 already improves both legs and arms, and the arm gains remain through stage72 and lambda_final.
- The downstream stages do not erase the promoted upstream geometry: `trunk_hidden` cosine stays ~`0.998` and `shared_encoder.0.weight` cosine stays ~`1.000` in all three compares.
- At `lambda_final`, the blend-aware freerun metrics also improve, so this is not just a direct-head-only artifact.

## Stage71: new71 -> candidate71

- overall: `legs_main=-0.065894`, `arms_main=-0.103466`, `left_arm_main=-0.097107`, `right_arm_main=-0.109825`
- A window (52-59): `legs_main=-0.154649`, `arms_main=-0.236356`
- B window (76-80): `legs_main=-0.075964`, `arms_main=-0.115982`

## Stage72: new72 -> candidate72

- overall: `legs_main=-0.026292`, `arms_main=-0.103466`, `left_arm_main=-0.097107`, `right_arm_main=-0.109825`
- A window (52-59): `legs_main=-0.189506`, `arms_main=-0.236356`
- B window (76-80): `legs_main=-0.005523`, `arms_main=-0.115982`

## LambdaFinal direct-path: newlambda -> candidatelambda

- overall: `legs_main=-0.026292`, `arms_main=-0.103466`, `left_arm_main=-0.097107`, `right_arm_main=-0.109825`
- A window (52-59): `legs_main=-0.189506`, `arms_main=-0.236356`
- B window (76-80): `legs_main=-0.005523`, `arms_main=-0.115982`

## LambdaFinal blend-aware summary

- `BlendGeoLocalDeg`: `-0.020124`
- `BlendGeoLocalDegWeighted`: `-0.033764`
- `GeoLocalDeg`: `-0.014627`
- `GeoLocalDegWeighted`: `-0.027642`
- `DirectGeoLocalDeg`: `-0.032109`
- `DirectGeoLocalDegWeighted`: `-0.039647`
- `LambdaMean`: `+0.000702` (essentially unchanged)

## Interpretation

- The promoted 70R fix survives the full `71 -> 72 -> lambda_final` chain instead of being washed out downstream.
- The largest downstream benefit is on upper body: overall `arms_main` improves by about `-0.103` at every downstream stage relative to the previous `new` chain.
- Legs also improve downstream: strongest at stage71 (`-0.065894` overall), then still remain slightly better at stage72/lambda (`-0.026292` overall).
- This supports promoting the s180 checkpoint as the new default 70R upstream for the chain.
