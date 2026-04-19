from __future__ import annotations

from typing import Any, Mapping, MutableMapping, Optional

STANDARD_ROTVEC_SEMANTICS = "standard_axis_angle_v1"
STANDARD_ANGVEL_SEMANTICS = "standard_axis_angle_times_fps_v1"
LEGACY_ROTVEC_SEMANTICS = "legacy_half_angle_v0"


def _meta_dict(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    meta = payload.get("meta", None)
    return meta if isinstance(meta, Mapping) else {}


def get_rotvec_semantics(payload: Mapping[str, Any]) -> Optional[str]:
    for cand in (
        payload.get("rotvec_semantics", None),
        _meta_dict(payload).get("rotvec_semantics", None),
    ):
        if isinstance(cand, str) and cand.strip():
            return cand.strip()
    return None


def get_angvel_semantics(payload: Mapping[str, Any]) -> Optional[str]:
    for cand in (
        payload.get("angvel_semantics", None),
        _meta_dict(payload).get("angvel_semantics", None),
    ):
        if isinstance(cand, str) and cand.strip():
            return cand.strip()
    return None


def stamp_standard_rotvec_spec(
    payload: MutableMapping[str, Any],
    *,
    asset_kind: str,
    source: Optional[str] = None,
) -> MutableMapping[str, Any]:
    meta_raw = payload.get("meta", None)
    meta = dict(meta_raw) if isinstance(meta_raw, Mapping) else {}
    geom_raw = meta.get("geometry_contract", None)
    geom = dict(geom_raw) if isinstance(geom_raw, Mapping) else {}
    geom["so3_log_map"] = "axis_times_angle_rad"
    geom["angvel_vec_from_delta_R"] = "axis_times_angle_rad_per_second"
    geom["angvel_vec_from_R_seq"] = "axis_times_angle_rad_per_second"
    meta["geometry_contract"] = geom
    meta["rotvec_semantics"] = STANDARD_ROTVEC_SEMANTICS
    meta["angvel_semantics"] = STANDARD_ANGVEL_SEMANTICS
    meta["rotvec_asset_kind"] = str(asset_kind)
    if source is not None:
        meta["rotvec_semantics_source"] = str(source)
    payload["meta"] = meta
    payload["rotvec_semantics"] = STANDARD_ROTVEC_SEMANTICS
    payload["angvel_semantics"] = STANDARD_ANGVEL_SEMANTICS
    return payload


def require_standard_rotvec_spec(payload: Mapping[str, Any], *, context: str) -> None:
    rotvec_sem = get_rotvec_semantics(payload)
    angvel_sem = get_angvel_semantics(payload)
    if rotvec_sem != STANDARD_ROTVEC_SEMANTICS:
        got = rotvec_sem or "missing"
        raise RuntimeError(
            f"[FATAL][ROTSEM] {context} requires rotvec_semantics={STANDARD_ROTVEC_SEMANTICS}, got {got}. "
            "Migrate the repo asset to standard axis-angle semantics before using it."
        )
    if angvel_sem is not None and angvel_sem != STANDARD_ANGVEL_SEMANTICS:
        raise RuntimeError(
            f"[FATAL][ROTSEM] {context} requires angvel_semantics={STANDARD_ANGVEL_SEMANTICS}, "
            f"got {angvel_sem}."
        )


def require_standard_rotvec_bundle(payload: Mapping[str, Any], *, context: str) -> None:
    require_standard_rotvec_spec(payload, context=context)
