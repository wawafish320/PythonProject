from __future__ import annotations

from typing import Dict, List, Optional

from .loss import AdaptiveLossWeighting, build_adaptive_loss
from .scheduler import AdaptiveHyperparamScheduler


class AdaptiveLossManager:
    """
    组合自适应损失权重与超参调度的薄封装，方便在 Trainer 中一次性注入。
    逻辑保持幂等，不改变原有 build_adaptive_loss / AdaptiveHyperparamScheduler 的行为。
    """

    def __init__(
        self,
        loss_terms: List[str],
        loss_method: str = "none",
        *,
        loss_alpha: float = 1.5,
        loss_temperature: float = 2.0,
        scheduler_params: Optional[Dict[str, float]] = None,
    ):
        self.loss_module: Optional[AdaptiveLossWeighting] = build_adaptive_loss(
            loss_terms, loss_method, alpha=loss_alpha, dwa_temperature=loss_temperature
        )
        self.scheduler: Optional[AdaptiveHyperparamScheduler] = (
            AdaptiveHyperparamScheduler(scheduler_params or {}) if scheduler_params else None
        )

    def step_scheduler(self, loss: float, grad_norm: float):
        if self.scheduler:
            self.scheduler.step(loss, grad_norm)

    def get_sched_params(self) -> Dict[str, float]:
        return self.scheduler.get_params() if self.scheduler else {}


__all__ = ["AdaptiveLossManager"]
