import math
from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch import autograd


class AdaptiveLossWeighting(nn.Module):
    """
    在线自适应损失权重，支持 GradNorm / Uncertainty / DWA。
    """

    def __init__(
        self,
        loss_names: List[str],
        method: str = "gradnorm",
        alpha: float = 1.5,
        dwa_temperature: float = 2.0,
    ):
        super().__init__()
        self.loss_names = list(loss_names)
        self.method = (method or "gradnorm").lower()
        self.alpha = float(alpha)
        self.T = float(dwa_temperature)

        if self.method == "uncertainty":
            self.log_vars = nn.Parameter(torch.zeros(len(self.loss_names)))
        else:
            self.log_vars = None

        if self.method == "dwa":
            self.loss_history: Dict[str, deque] = {
                name: deque(maxlen=2) for name in self.loss_names
            }
        else:
            self.loss_history = {}

    def forward(
        self,
        losses: Dict[str, torch.Tensor],
        model: Optional[nn.Module] = None,
        epoch: int = 1,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        if not losses:
            raise ValueError("losses 字典不能为空")

        # 只保留需要的 loss
        filtered: Dict[str, torch.Tensor] = {
            name: loss
            for name, loss in losses.items()
            if name in self.loss_names and loss is not None
        }
        if not filtered:
            raise ValueError("提供的 losses 中不包含需要自适应的项目")

        if self.method == "gradnorm":
            return self._gradnorm_weighting(filtered, model)
        if self.method == "uncertainty":
            return self._uncertainty_weighting(filtered)
        if self.method == "dwa":
            return self._dwa_weighting(filtered, epoch)
        # 默认均分
        weight = 1.0 / len(filtered)
        weights = {name: weight for name in filtered}
        total = sum(weight * filtered[name] for name in filtered)
        return total, weights

    # --- 不同策略实现 ---

    def _gradnorm_weighting(
        self, losses: Dict[str, torch.Tensor], model: Optional[nn.Module]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        if model is None:
            raise ValueError("GradNorm 需要传入 model 以获取梯度")

        params = [p for p in model.parameters() if p.requires_grad]
        if not params:
            raise ValueError("模型没有需要求梯度的参数")
        last_layer = params[-1]

        grad_norms: Dict[str, float] = {}
        for name, loss in losses.items():
            grad = autograd.grad(
                loss,
                last_layer,
                retain_graph=True,
                create_graph=False,
                allow_unused=True,
            )[0]
            if grad is None:
                grad_norms[name] = 0.0
            else:
                grad_norms[name] = float(grad.norm().detach().cpu().item())

        total_norm = sum(grad_norms.values()) + 1e-8
        weights = {
            name: (total_norm / (norm + 1e-8)) ** self.alpha
            for name, norm in grad_norms.items()
        }
        sum_weights = sum(weights.values()) + 1e-8
        weights = {name: weight / sum_weights for name, weight in weights.items()}
        weighted_loss = sum(weights[name] * losses[name] for name in weights)
        return weighted_loss, weights

    def _uncertainty_weighting(
        self, losses: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        if self.log_vars is None:
            raise ValueError("未初始化 uncertainty 参数")
        if len(losses) != len(self.loss_names):
            # 只取两者交集
            valid_names = [n for n in self.loss_names if n in losses]
        else:
            valid_names = self.loss_names

        weighted_terms = []
        weights_display: Dict[str, float] = {}
        for idx, name in enumerate(valid_names):
            precision = torch.exp(-self.log_vars[idx])
            term = precision * losses[name] + self.log_vars[idx]
            weighted_terms.append(term)
            weights_display[name] = float(precision.detach().cpu().item())

        sum_w = sum(max(v, 1e-8) for v in weights_display.values())
        if sum_w > 0:
            weights_display = {k: v / sum_w for k, v in weights_display.items()}

        return sum(weighted_terms), weights_display

    def _dwa_weighting(
        self, losses: Dict[str, torch.Tensor], epoch: int
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        for name, loss in losses.items():
            if name in self.loss_history:
                self.loss_history[name].append(float(loss.detach().cpu().item()))

        ready = all(len(h) == 2 for h in self.loss_history.values())
        if not ready or epoch <= 1:
            equal = 1.0 / len(losses)
            weights = {name: equal for name in losses}
        else:
            rates: Dict[str, float] = {}
            for name, history in self.loss_history.items():
                prev, last = history[0], history[1]
                rates[name] = last / (prev + 1e-8)
            exp_rates = {n: math.exp(rate / self.T) for n, rate in rates.items()}
            denom = sum(exp_rates.values()) + 1e-8
            weights = {n: exp_rates.get(n, 0.0) / denom for n in losses}

        weighted_loss = sum(weights[name] * losses[name] for name in weights)
        return weighted_loss, weights


def build_adaptive_loss(
    loss_names: List[str],
    method: str,
    alpha: float = 1.5,
    dwa_temperature: float = 2.0,
) -> Optional[AdaptiveLossWeighting]:
    method = (method or "none").lower()
    if method in ("", "none"):
        return None
    return AdaptiveLossWeighting(
        loss_names=loss_names,
        method=method,
        alpha=alpha,
        dwa_temperature=dwa_temperature,
    )
