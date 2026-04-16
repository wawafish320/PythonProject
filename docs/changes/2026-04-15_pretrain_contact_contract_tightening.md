# 2026-04-15 mainline contact contract tightening

- Removed `posttrain_contacts_source`.
- Removed `trainbase_contacts_source`.
- Mainline posttrain and basetrain rollout contacts are now fixed to `pretrain_contact`.
- `whitebox` / `auto` are no longer exposed as mainline entry points.
