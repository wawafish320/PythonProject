from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass(frozen=True)
class WindowSpec:
    entry_window_pre_k: int
    entry_window_post_k: int
    recovery_window_k: int


def _validate_non_negative(value: int, *, name: str) -> int:
    out = int(value)
    if out < 0:
        raise ValueError(f"{name} must be >= 0")
    return out


def compute_window_bounds(
    *,
    inject_at_step: int,
    total_steps: int,
    spec: WindowSpec,
) -> dict[str, dict]:
    inject_step = int(inject_at_step)
    steps = int(total_steps)
    if steps <= 0:
        raise ValueError("total_steps must be > 0")
    if inject_step < 0 or inject_step >= steps:
        raise ValueError(f"inject_at_step out of range: {inject_step} for total_steps={steps}")

    pre_k = _validate_non_negative(spec.entry_window_pre_k, name="entry_window_pre_k")
    post_k = _validate_non_negative(spec.entry_window_post_k, name="entry_window_post_k")
    rec_k = _validate_non_negative(spec.recovery_window_k, name="recovery_window_k")

    entry_start = max(0, inject_step - pre_k)
    entry_end = min(steps - 1, inject_step + post_k)
    rec_start = inject_step
    rec_end = min(steps - 1, inject_step + rec_k)

    return {
        "entry_window": {
            "window_label": "entry_window",
            "rel_origin_step": inject_step,
            "rel_key": "step_rel_entry",
            "t_start_requested": inject_step - pre_k,
            "t_end_requested": inject_step + post_k,
            "t_start": entry_start,
            "t_end": entry_end,
            "window_steps": int(entry_end - entry_start + 1),
            "entry_window_pre_k": pre_k,
            "entry_window_post_k": post_k,
        },
        "post_inject_recovery": {
            "window_label": "post_inject_recovery",
            "rel_origin_step": inject_step,
            "rel_key": "step_rel_inject",
            "t_start_requested": inject_step,
            "t_end_requested": inject_step + rec_k,
            "t_start": rec_start,
            "t_end": rec_end,
            "window_steps": int(rec_end - rec_start + 1),
            "recovery_window_k": rec_k,
        },
    }


def summarize_window_metrics(
    *,
    per_step_metrics: List[Dict[str, Any]],
    bounds: dict[str, dict],
    required_metrics: list[str],
) -> dict[str, Any]:
    if not isinstance(per_step_metrics, list):
        raise ValueError("per_step_metrics must be a list")
    req = [str(k) for k in required_metrics]
    if not req:
        raise ValueError("required_metrics cannot be empty")

    out: dict[str, Any] = {}
    for win_name in ("entry_window", "post_inject_recovery"):
        win = bounds.get(win_name)
        if not isinstance(win, dict):
            raise ValueError(f"bounds missing window: {win_name}")
        t0 = int(win["t_start"])
        t1 = int(win["t_end"])
        if t0 < 0 or t1 < t0:
            raise ValueError(f"invalid window bounds for {win_name}: t_start={t0} t_end={t1}")
        idxs = list(range(t0, min(t1 + 1, len(per_step_metrics))))
        per_step = []
        for i in idxs:
            rec = per_step_metrics[i]
            if not isinstance(rec, dict):
                continue
            rec2 = dict(rec)
            rec2["step"] = int(i)
            rec2[str(win["rel_key"])] = int(i - int(win["rel_origin_step"]))
            per_step.append(rec2)

        metric_summary: dict[str, dict] = {}
        for key in req:
            vals: list[tuple[int, float]] = []
            for rec in per_step:
                v = rec.get(key)
                try:
                    fv = float(v)
                except (TypeError, ValueError):
                    continue
                if fv != fv or fv in (float("inf"), float("-inf")):
                    continue
                vals.append((int(rec["step"]), fv))
            if not vals:
                metric_summary[key] = {
                    "n": 0,
                    "mean": None,
                    "max_abs": None,
                    "end": None,
                    "peak_step_rel": None,
                }
                continue
            only_vals = [x[1] for x in vals]
            mean_v = sum(only_vals) / float(len(only_vals))
            max_abs_v = max(abs(v) for v in only_vals)
            end_step, end_v = vals[-1]
            peak_step, _ = max(vals, key=lambda p: abs(p[1]))
            metric_summary[key] = {
                "n": int(len(only_vals)),
                "mean": float(mean_v),
                "max_abs": float(max_abs_v),
                "end": float(end_v),
                "peak_step_rel": int(peak_step - int(win["rel_origin_step"])),
            }

        out[win_name] = {
            **win,
            "per_step": per_step,
            "metric_summary": metric_summary,
        }
    return out
