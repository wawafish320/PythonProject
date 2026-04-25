from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any, Callable, Dict, Optional

import torch

_POSTTRAIN_REDUCE_ERRORS = (RuntimeError, ValueError, TypeError)


def reduce_optional_term_totals(
    *,
    zero: torch.Tensor,
    named_terms: Mapping[str, Sequence[torch.Tensor]],
) -> Dict[str, torch.Tensor]:
    totals: Dict[str, torch.Tensor] = {}
    for name, terms in named_terms.items():
        totals[name] = torch.stack(tuple(terms)).sum() if terms else zero
    return totals


def safe_float_scalar(value: Any, default: float = float("nan")) -> float:
    if torch.is_tensor(value):
        return float(value.detach().cpu())
    if isinstance(value, (float, int)):
        resolved = float(value)
        return resolved if math.isfinite(resolved) else default
    return default


def summarize_optional_term_values(
    *,
    terms: Sequence[torch.Tensor],
    reduction: str,
    record_soft_fail: Optional[Callable[[str], None]] = None,
    soft_fail_key: Optional[str] = None,
) -> Optional[float]:
    if not terms:
        return None
    try:
        stacked = torch.stack(tuple(terms))
        if reduction == "mean":
            value = stacked.mean()
        elif reduction == "sum":
            value = stacked.sum()
        else:
            raise ValueError(f"Unsupported reduction: {reduction}")
        return safe_float_scalar(value)
    except _POSTTRAIN_REDUCE_ERRORS:
        if record_soft_fail is not None and soft_fail_key:
            record_soft_fail(soft_fail_key)
        return None


def summarize_lambda_finalize_stats(
    *,
    lam_vals: Sequence[torch.Tensor],
    lam_eff_vals: Sequence[torch.Tensor],
    lam_rel_vals: Sequence[torch.Tensor],
    record_soft_fail: Optional[Callable[[str], None]] = None,
) -> Dict[str, float]:
    lam_mean = lam_std = lam_eff_mean = lam_eff_std = lam_rel_mean = None
    try:
        flat = torch.cat([x.reshape(-1) for x in lam_vals], dim=0)
        lam_mean = safe_float_scalar(flat.mean())
        lam_std = safe_float_scalar(flat.std(unbiased=False))
    except _POSTTRAIN_REDUCE_ERRORS:
        if record_soft_fail is not None:
            record_soft_fail("finalize_lambda_stats_raw")
    try:
        flat = torch.cat([x.reshape(-1) for x in lam_eff_vals], dim=0)
        lam_eff_mean = safe_float_scalar(flat.mean())
        lam_eff_std = safe_float_scalar(flat.std(unbiased=False))
    except _POSTTRAIN_REDUCE_ERRORS:
        if record_soft_fail is not None:
            record_soft_fail("finalize_lambda_stats_eff")
    try:
        if lam_rel_vals:
            lam_rel_mean = safe_float_scalar(torch.cat([x.reshape(-1) for x in lam_rel_vals], dim=0).mean())
    except _POSTTRAIN_REDUCE_ERRORS:
        if record_soft_fail is not None:
            record_soft_fail("finalize_lambda_stats_rel")
    return {
        "lambda_mean": lam_mean if lam_mean is not None else float("nan"),
        "lambda_std": lam_std if lam_std is not None else float("nan"),
        "lambda_eff_mean": lam_eff_mean if lam_eff_mean is not None else float("nan"),
        "lambda_eff_std": lam_eff_std if lam_eff_std is not None else float("nan"),
        "lambda_rel_mean": lam_rel_mean if lam_rel_mean is not None else float("nan"),
    }
