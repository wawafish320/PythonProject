#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

try:
    from train.models import DEFAULT_DIRECT_POSE_LEG_BONES, STAGE6_3WAY_ARMCHAIN_BONES
except Exception:
    DEFAULT_DIRECT_POSE_LEG_BONES = (
        'thigh_r', 'calf_r', 'foot_r', 'ball_r', 'thigh_l', 'calf_l', 'foot_l', 'ball_l',
    )
    STAGE6_3WAY_ARMCHAIN_BONES = (
        'clavicle_l', 'upperarm_l', 'RUpArmTwist_l_01', 'RUpArmTwist_l_02', 'lowerarm_l', 'L_ForeTwist_01',
        'L_ForeTwist_02', 'hand_l', 'index_01_l', 'middle_01_l', 'ring_01_l', 'pinky_01_l', 'thumb_01_l',
        'clavicle_r', 'upperarm_r', 'RUpArmTwist_r_01', 'RUpArmTwist_r_02', 'lowerarm_r', 'R_ForeTwist_01',
        'R_ForeTwist_02', 'hand_r', 'index_01_r', 'middle_01_r', 'ring_01_r', 'pinky_01_r', 'thumb_01_r',
    )


def _finite(values: Sequence[Any]) -> List[float]:
    out: List[float] = []
    for value in values:
        try:
            x = float(value)
        except Exception:
            continue
        if math.isfinite(x):
            out.append(x)
    return out


def _mean(values: Sequence[Any]) -> float:
    vals = _finite(values)
    if not vals:
        return float('nan')
    return float(sum(vals) / len(vals))


def _quantile(values: Sequence[Any], q: float) -> float:
    vals = sorted(_finite(values))
    if not vals:
        return float('nan')
    qq = min(1.0, max(0.0, float(q)))
    idx = int(round(qq * (len(vals) - 1)))
    idx = max(0, min(len(vals) - 1, idx))
    return float(vals[idx])


def _load_json(path: Path) -> Mapping[str, Any]:
    return json.loads(path.read_text())


def _build_mask(steps: Sequence[Mapping[str, Any]], *, cycle_gte: int, drop_wrap: bool) -> List[bool]:
    mask: List[bool] = []
    for step in steps:
        try:
            cycle = int(step.get('cycle', 0) or 0)
        except Exception:
            cycle = 0
        if cycle < int(cycle_gte):
            mask.append(False)
            continue
        if drop_wrap and bool(step.get('wrap_boundary_step', False)):
            mask.append(False)
            continue
        mask.append(True)
    return mask


def _pick_group_indices(names: Sequence[str], root_idx: int) -> Dict[str, List[int]]:
    leg_set = {str(x) for x in DEFAULT_DIRECT_POSE_LEG_BONES}
    arm_set = {str(x) for x in STAGE6_3WAY_ARMCHAIN_BONES}
    idx_all = [i for i in range(len(names)) if int(i) != int(root_idx)]
    idx_leg = [i for i, name in enumerate(names) if int(i) != int(root_idx) and str(name) in leg_set]
    idx_arm = [i for i, name in enumerate(names) if int(i) != int(root_idx) and str(name) in arm_set]
    idx_nonleg = [i for i in idx_all if i not in set(idx_leg)]
    idx_else = [i for i in idx_nonleg if i not in set(idx_arm)]
    return {
        'leg': idx_leg,
        'nonleg': idx_nonleg,
        'arm': idx_arm,
        'else': idx_else,
        'all_ex_root': idx_all,
    }


def build_summary(path: Path, *, cycle_gte: int, drop_wrap: bool) -> Dict[str, Any]:
    obj = _load_json(path)
    steps = obj.get('metrics_per_step', [])
    per = obj.get('per_step_direct_geolocal_deg', {})
    if not isinstance(steps, list) or not isinstance(per, Mapping):
        raise SystemExit('[FATAL] invalid freerun json: missing metrics_per_step/per_step_direct_geolocal_deg')
    names = per.get('bone_names', [])
    mat = per.get('DirectGeoLocalDeg', [])
    if not isinstance(names, list) or not isinstance(mat, list) or not names or not mat:
        raise SystemExit('[FATAL] invalid per_step_direct_geolocal_deg payload')
    try:
        root_idx = int(per.get('root_idx', 0) or 0)
    except Exception:
        root_idx = 0
    names = [str(x) for x in names]
    step_mask = _build_mask(steps, cycle_gte=int(cycle_gte), drop_wrap=bool(drop_wrap))
    group_idx = _pick_group_indices(names, root_idx)

    groups: Dict[str, Dict[str, Any]] = {}
    for group_name, indices in group_idx.items():
        values: List[float] = []
        for step_i, keep in enumerate(step_mask):
            if not keep or step_i >= len(mat):
                continue
            row = mat[step_i]
            if not isinstance(row, list):
                continue
            for joint_i in indices:
                if joint_i >= len(row):
                    continue
                try:
                    value = float(row[joint_i])
                except Exception:
                    continue
                if math.isfinite(value):
                    values.append(value)
        groups[group_name] = {
            'j': int(len(indices)),
            'samples': int(len(values)),
            'mean': _mean(values),
            'p50': _quantile(values, 0.50),
            'p90': _quantile(values, 0.90),
            'p95': _quantile(values, 0.95),
        }

    return {
        'source': str(path),
        'mask': {
            'cycle_gte': int(cycle_gte),
            'drop_wrap': bool(drop_wrap),
            'kept_steps': int(sum(1 for keep in step_mask if keep)),
            'total_steps': int(len(step_mask)),
        },
        'groups': groups,
        'group_names': {
            'leg': [names[i] for i in group_idx['leg']],
            'arm': [names[i] for i in group_idx['arm']],
            'else': [names[i] for i in group_idx['else']],
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('json', type=str, help='run_freerun_cycles output json')
    ap.add_argument('--cycle_gte', type=int, default=1)
    ap.add_argument('--drop_wrap', action='store_true')
    ap.add_argument('--out', type=str, default=None)
    args = ap.parse_args()

    in_path = Path(args.json).expanduser().resolve()
    out_path = Path(args.out).expanduser().resolve() if args.out else (in_path.parent / 'group_summary.json')
    payload = build_summary(in_path, cycle_gte=int(args.cycle_gte), drop_wrap=bool(args.drop_wrap))
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    print(out_path)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
