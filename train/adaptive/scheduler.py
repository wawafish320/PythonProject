from __future__ import annotations

from collections import deque
from typing import Dict

import numpy as np


class AdaptiveHyperparamScheduler:
    """在线调整训练超参数（freerun horizon / teacher forcing 等）。"""

    def __init__(
        self,
        initial_params: Dict[str, float],
        loss_spike_threshold: float = 1.5,
        convergence_threshold: float = 0.02,
        adjustment_rate: float = 0.1,
        check_interval: int = 50,
    ):
        self.params = dict(initial_params)
        self.loss_history = deque(maxlen=max(check_interval, 50))
        self.grad_norm_history = deque(maxlen=max(check_interval, 50))
        self.loss_spike_threshold = float(loss_spike_threshold)
        self.convergence_threshold = float(convergence_threshold)
        self.adjustment_rate = float(adjustment_rate)
        self.check_interval = int(max(1, check_interval))
        self._last_loss_ratio: float = 1.0
        self._last_cv: float = 1.0

    def step(self, loss: float, grad_norm: float):
        self.loss_history.append(float(loss))
        self.grad_norm_history.append(float(grad_norm))
        if len(self.loss_history) % self.check_interval == 0:
            self._auto_adjust()

    def _auto_adjust(self):
        if self._detect_loss_spike():
            self._reduce_difficulty()
        elif self._detect_convergence():
            self._increase_difficulty()
        elif self._detect_vanishing_gradients():
            self._boost_gradients()

    def _detect_loss_spike(self) -> bool:
        if len(self.loss_history) < self.check_interval * 2:
            return False
        recent = np.mean(list(self.loss_history)[-self.check_interval :])
        baseline = np.mean(
            list(self.loss_history)[-self.check_interval * 2 : -self.check_interval]
        )
        if baseline <= 0:
            return False
        self._last_loss_ratio = recent / baseline
        return self._last_loss_ratio > self.loss_spike_threshold

    def _detect_convergence(self) -> bool:
        if len(self.loss_history) < self.check_interval:
            return False
        recent = list(self.loss_history)[-self.check_interval :]
        mean = np.mean(recent)
        std = np.std(recent)
        if mean <= 1e-8:
            return False
        self._last_cv = std / mean
        return self._last_cv < self.convergence_threshold

    def _detect_vanishing_gradients(self) -> bool:
        if len(self.grad_norm_history) < self.check_interval * 2:
            return False
        recent = np.mean(list(self.grad_norm_history)[-self.check_interval :])
        baseline = np.mean(
            list(self.grad_norm_history)[
                -self.check_interval * 2 : -self.check_interval
            ]
        )
        if baseline <= 1e-8:
            return False
        return recent < 0.1 * baseline

    def _reduce_difficulty(self):
        horizon = int(self.params.get("freerun_horizon", 0))
        min_h = int(self.params.get("freerun_min", 6))
        spike_scale = max(1.0, self._last_loss_ratio / max(self.loss_spike_threshold, 1e-6))
        if horizon > 0:
            step = self.adjustment_rate * spike_scale
            new_h = max(min_h, int(horizon * max(0.5, 1.0 - step)))
            self.params["freerun_horizon"] = new_h
        tf_ratio = float(self.params.get("teacher_forcing_ratio", 0.5))
        tf_ratio = min(1.0, tf_ratio + self.adjustment_rate * spike_scale)
        self.params["teacher_forcing_ratio"] = tf_ratio

    def _increase_difficulty(self):
        horizon = int(self.params.get("freerun_horizon", 0))
        max_h = int(self.params.get("freerun_max", max(horizon, 0)))
        if max_h <= 0:
            return
        # 收敛越平稳（cv 越小），提升幅度越大，但控制在 2x 调整率以内
        cv = max(self._last_cv, 1e-4)
        scale = min(2.0, max(1.0, self.convergence_threshold / cv))
        new_h = min(max_h, max(1, int(horizon * (1.0 + self.adjustment_rate * scale))))
        self.params["freerun_horizon"] = new_h
        tf_ratio = float(self.params.get("teacher_forcing_ratio", 0.5))
        tf_ratio = max(0.0, tf_ratio - self.adjustment_rate * scale)
        self.params["teacher_forcing_ratio"] = tf_ratio

    def _boost_gradients(self):
        tf_ratio = float(self.params.get("teacher_forcing_ratio", 0.5))
        tf_ratio = min(1.0, tf_ratio + 0.2)
        self.params["teacher_forcing_ratio"] = tf_ratio

    def get_params(self) -> Dict[str, float]:
        return dict(self.params)


__all__ = ["AdaptiveHyperparamScheduler"]
