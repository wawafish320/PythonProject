#!/usr/bin/env python3
"""Auto-adjust stage schedule parameters based on latest validation metrics.

Usage (run after each epoch or when new metrics are saved):
  python tools/stage_auto_tuner.py \
      --config config/exp_phase_mpl.json \
      --metrics models/exp_phase_e2e_sc/exp_phase_MLP/metrics

The script reads valfree metrics (yaw drift, root velocity MAE, rot6d geodesic
error) per epoch, aggregates them for each configured stage, and nudges
`lookahead_weight`, `freerun_weight`, `freerun_horizon`, and
`w_latent_consistency` toward stage-specific targets.

Use `--dry-run` to preview adjustments without writing the config file.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


DEFAULT_THRESHOLDS = {
    'yaw': {'lo': 12.0, 'hi': 18.0},
    'root': {'lo': 0.014, 'hi': 0.018},
    'rot': {'lo': 1.9, 'hi': 2.2},
}


Metric = Dict[str, float]


def load_val_metrics(metrics_dir: Path) -> Dict[int, Metric]:
    epochs: Dict[int, Metric] = {}
    for path in sorted(metrics_dir.glob('valfree_ep*.json')):
        try:
            data = json.loads(path.read_text())
        except Exception as exc:
            print(f"[WARN] skip {path.name}: {exc}")
            continue
        epoch = int(data.get('epoch') or 0)
        metrics = data.get('metrics') or {}
        epochs[epoch] = {
            'YawAbsDeg': float(metrics.get('YawAbsDeg')) if metrics.get('YawAbsDeg') is not None else None,
            'RootVelMAE': float(metrics.get('RootVelMAE')) if metrics.get('RootVelMAE') is not None else None,
            'InputRotGeoDeg': float(metrics.get('InputRotGeoDeg')) if metrics.get('InputRotGeoDeg') is not None else None,
        }
    return epochs


def summarize_stage_metrics(stage: Dict[str, Any], val_metrics: Dict[int, Metric]) -> Metric:
    rng = stage.get('range') or [stage.get('start'), stage.get('end')]
    if not rng or len(rng) < 1:
        return {}
    start = int(rng[0])
    end = int(rng[-1] if len(rng) > 1 else rng[0])
    samples: List[Metric] = [val_metrics[e] for e in range(start, end + 1) if e in val_metrics]
    if not samples:
        return {}
    agg: Dict[str, float] = {}
    for key in ('YawAbsDeg', 'RootVelMAE', 'InputRotGeoDeg'):
        values = [m[key] for m in samples if m.get(key) is not None]
        if values:
            agg[key] = sum(values) / len(values)
    return agg


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _get_bounds(stage: Dict[str, Any], key: str) -> Tuple[float, float]:
    target = stage.get('targets') or {}
    spec = target.get(key) or {}
    defaults = DEFAULT_THRESHOLDS[key]
    lo = float(spec.get('lo', defaults['lo']))
    hi = float(spec.get('hi', defaults['hi']))
    if lo > hi:
        lo, hi = hi, lo
    return lo, hi


def adjust_stage(stage: Dict[str, Any], metrics: Metric) -> Dict[str, float]:
    trainer = stage.setdefault('trainer', {})
    loss_cfg = stage.setdefault('loss', {})
    changed: Dict[str, float] = {}

    if not metrics:
        return changed

    yaw = metrics.get('YawAbsDeg')
    root_mae = metrics.get('RootVelMAE')
    rot_geo = metrics.get('InputRotGeoDeg')

    yaw_lo, yaw_hi = _get_bounds(stage, 'yaw')
    root_lo, root_hi = _get_bounds(stage, 'root')
    rot_lo, rot_hi = _get_bounds(stage, 'rot')

    lookahead_w = float(trainer.get('lookahead_weight', 0.0) or 0.0)
    freerun_w = float(trainer.get('freerun_weight', 0.0) or 0.0)
    freerun_h = int(trainer.get('freerun_horizon', 0) or 0)
    latent_w = float(trainer.get('w_latent_consistency', 0.0) or 0.0)
    w_fk = float(loss_cfg.get('w_fk_pos', 0.0) or 0.0)
    w_rot_local = float(loss_cfg.get('w_rot_local', 0.0) or 0.0)

    if yaw is not None:
        if yaw > yaw_hi:
            new = clamp(lookahead_w + 0.05, 0.0, 0.5)
            if new != lookahead_w:
                trainer['lookahead_weight'] = round(new, 4)
                changed['lookahead_weight'] = trainer['lookahead_weight']
                lookahead_w = new
            new_h = clamp(freerun_h + 2, 0, 32)
            if new_h != freerun_h:
                trainer['freerun_horizon'] = int(new_h)
                changed['freerun_horizon'] = trainer['freerun_horizon']
                freerun_h = new_h
        elif yaw < yaw_lo:
            new = clamp(lookahead_w - 0.05, 0.0, 0.5)
            if new != lookahead_w:
                trainer['lookahead_weight'] = round(new, 4)
                changed['lookahead_weight'] = trainer['lookahead_weight']
                lookahead_w = new
            new_h = clamp(freerun_h - 2, 4, 32)
            if new_h != freerun_h:
                trainer['freerun_horizon'] = int(new_h)
                changed['freerun_horizon'] = trainer['freerun_horizon']
                freerun_h = new_h

    if root_mae is not None:
        if root_mae > root_hi:
            new = clamp(freerun_w + 0.05, 0.0, 0.6)
            if new != freerun_w:
                trainer['freerun_weight'] = round(new, 4)
                changed['freerun_weight'] = trainer['freerun_weight']
                freerun_w = new
        elif root_mae < root_lo:
            new = clamp(freerun_w - 0.02, 0.0, 0.6)
            if new != freerun_w:
                trainer['freerun_weight'] = round(new, 4)
                changed['freerun_weight'] = trainer['freerun_weight']
                freerun_w = new

    if rot_geo is not None:
        if rot_geo > rot_hi:
            new = clamp(latent_w + 0.05, 0.0, 0.5)
            if new != latent_w:
                trainer['w_latent_consistency'] = round(new, 4)
                changed['w_latent_consistency'] = trainer['w_latent_consistency']
                latent_w = new
            new_fk = clamp(w_fk + 0.05, 0.0, 0.6)
            if new_fk != w_fk:
                loss_cfg['w_fk_pos'] = round(new_fk, 4)
                changed['loss.w_fk_pos'] = loss_cfg['w_fk_pos']
                w_fk = new_fk
            new_rot_local = clamp(w_rot_local + 0.05, 0.0, 0.6)
            if new_rot_local != w_rot_local:
                loss_cfg['w_rot_local'] = round(new_rot_local, 4)
                changed['loss.w_rot_local'] = loss_cfg['w_rot_local']
                w_rot_local = new_rot_local
        elif rot_geo < rot_lo:
            new = clamp(latent_w - 0.02, 0.0, 0.5)
            if new != latent_w:
                trainer['w_latent_consistency'] = round(new, 4)
                changed['w_latent_consistency'] = trainer['w_latent_consistency']
                latent_w = new
            new_fk = clamp(w_fk - 0.02, 0.0, 0.6)
            if new_fk != w_fk:
                loss_cfg['w_fk_pos'] = round(new_fk, 4)
                changed['loss.w_fk_pos'] = loss_cfg['w_fk_pos']
                w_fk = new_fk
            new_rot_local = clamp(w_rot_local - 0.02, 0.0, 0.6)
            if new_rot_local != w_rot_local:
                loss_cfg['w_rot_local'] = round(new_rot_local, 4)
                changed['loss.w_rot_local'] = loss_cfg['w_rot_local']
                w_rot_local = new_rot_local

    return changed


def main() -> None:
    ap = argparse.ArgumentParser(description="Auto-tune stage schedule based on metrics")
    ap.add_argument('--config', default='config/exp_phase_mpl.json', help='Path to config JSON to update')
    ap.add_argument('--metrics', default='models/exp_phase_e2e_sc/exp_phase_MLP/metrics', help='Directory containing valfree metrics json files')
    ap.add_argument('--dry-run', action='store_true', help='Preview adjustments without writing the config file')
    args = ap.parse_args()

    config_path = Path(args.config)
    metrics_dir = Path(args.metrics)
    if not config_path.is_file():
        ap.error(f'config file not found: {config_path}')
    if not metrics_dir.is_dir():
        ap.error(f'metrics dir not found: {metrics_dir}')

    config = json.loads(config_path.read_text())
    schedule = config.get('lookahead_stage_schedule')
    if schedule is None:
        ap.error('config missing lookahead_stage_schedule')

    val_metrics = load_val_metrics(metrics_dir)
    if not val_metrics:
        ap.error('no valfree metrics found to guide tuning')

    total_changes = 0
    for stage in schedule:
        label = stage.get('label') or f"{stage.get('range')}"
        summary = summarize_stage_metrics(stage, val_metrics)
        if not summary:
            print(f"[INFO] stage {label}: no matching epochs yet, skip")
            continue
        delta = adjust_stage(stage, summary)
        print(f"[STAGE] {label}: metrics={summary} adjustments={delta or 'no change'}")
        total_changes += len(delta)

    if total_changes == 0:
        print('[INFO] no parameter updates needed')
        return

    if args.dry_run:
        print('[INFO] dry-run enabled; config not written')
        return

    config['lookahead_stage_schedule'] = schedule
    config_path.write_text(json.dumps(config, indent=2, ensure_ascii=False))
    print(f'[OK] saved updated schedule to {config_path}')


if __name__ == '__main__':
    main()
