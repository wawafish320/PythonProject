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

    # --- heuristics -------------------------------------------------

    def _detect_loss_spike(self) -> bool:
        if len(self.loss_history) < self.check_interval:
            return False
        recent = np.mean(list(self.loss_history)[-self.check_interval :])
        baseline = np.mean(
            list(self.loss_history)[-self.check_interval * 2 : -self.check_interval]
        )
        if baseline <= 0:
            return False
        return recent > baseline * self.loss_spike_threshold

    def _detect_convergence(self) -> bool:
        if len(self.loss_history) < self.check_interval:
            return False
        recent = list(self.loss_history)[-self.check_interval :]
        mean = np.mean(recent)
        std = np.std(recent)
        if mean <= 1e-8:
            return False
        return (std / mean) < self.convergence_threshold

    def _detect_vanishing_gradients(self) -> bool:
        if len(self.grad_norm_history) < self.check_interval:
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

    # --- adjustment actions ----------------------------------------

    def _reduce_difficulty(self):
        horizon = int(self.params.get("freerun_horizon", 0))
        min_h = int(self.params.get("freerun_min", 6))
        if horizon > 0:
            new_h = max(min_h, int(horizon * (1.0 - self.adjustment_rate)))
            self.params["freerun_horizon"] = new_h
        tf_ratio = float(self.params.get("teacher_forcing_ratio", 0.5))
        tf_ratio = min(1.0, tf_ratio + self.adjustment_rate)
        self.params["teacher_forcing_ratio"] = tf_ratio

    def _increase_difficulty(self):
        horizon = int(self.params.get("freerun_horizon", 0))
        max_h = int(self.params.get("freerun_max", max(horizon, 0)))
        if max_h <= 0:
            return
        new_h = min(max_h, max(1, int(horizon * (1.0 + self.adjustment_rate))))
        self.params["freerun_horizon"] = new_h
        tf_ratio = float(self.params.get("teacher_forcing_ratio", 0.5))
        tf_ratio = max(0.0, tf_ratio - self.adjustment_rate)
        self.params["teacher_forcing_ratio"] = tf_ratio

    def _boost_gradients(self):
        tf_ratio = float(self.params.get("teacher_forcing_ratio", 0.5))
        tf_ratio = min(1.0, tf_ratio + 0.2)
        self.params["teacher_forcing_ratio"] = tf_ratio

    # ----------------------------------------------------------------

    def get_params(self) -> Dict[str, float]:
        return dict(self.params)
