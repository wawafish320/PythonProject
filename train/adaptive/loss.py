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
        loss_names: List[str] | None,
        method: str = "gradnorm",
        alpha: float = 1.5,
        dwa_temperature: float = 2.0,
        *,
        scales: Optional[Dict[str, float]] = None,
        weight_ema_beta: float = 0.0,
        logvar_clip: float = 4.0,
    ):
        super().__init__()
        # allow loss_names为空，运行时自动使用 payload 中的所有 loss
        self.loss_names = list(loss_names or [])
        self.method = (method or "gradnorm").lower()
        self.alpha = float(alpha)
        self.T = float(dwa_temperature)
        self.scales = {
            k: float(v) for k, v in (scales.items() if isinstance(scales, dict) else [])
        }
        self.weight_ema_beta = float(max(0.0, min(0.999, weight_ema_beta)))
        self.logvar_clip = float(max(0.0, logvar_clip))

        if self.method == "uncertainty":
            self.log_vars = nn.Parameter(torch.zeros(len(self.loss_names)))
        else:
            self.log_vars = None

        # 供权重可视化平滑用
        self._ema_weights: Dict[str, float] = {}

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

        filtered: Dict[str, torch.Tensor] = {
            name: loss
            for name, loss in losses.items()
            if (not self.loss_names or name in self.loss_names) and loss is not None
        }
        if not filtered:
            raise ValueError("提供的 losses 中不包含需要自适应的项目")

        if self.method == "gradnorm":
            return self._gradnorm_weighting(filtered, model)
        if self.method == "uncertainty":
            return self._uncertainty_weighting(filtered)
        if self.method == "dwa":
            return self._dwa_weighting(filtered, epoch)
        weight = 1.0 / len(filtered)
        weights = {name: weight for name in filtered}
        total = sum(weight * filtered[name] for name in filtered)
        return total, weights

    def _maybe_scale(self, name: str, loss: torch.Tensor) -> torch.Tensor:
        scale = float(self.scales.get(name, 1.0)) if hasattr(self, "scales") else 1.0
        if scale <= 0 or not math.isfinite(scale):
            return loss
        return loss / scale

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
            loss_scaled = self._maybe_scale(name, loss)
            grad = autograd.grad(
                loss_scaled,
                last_layer,
                retain_graph=True,
                create_graph=False,
                allow_unused=True,
            )[0]
            grad_norms[name] = 0.0 if grad is None else float(grad.norm().detach().cpu().item())

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
            valid_names = [n for n in self.loss_names if n in losses]
        else:
            valid_names = self.loss_names

        weighted_terms = []
        weights_display: Dict[str, float] = {}
        log_vars = torch.clamp(self.log_vars, -self.logvar_clip, self.logvar_clip)
        for idx, name in enumerate(valid_names):
            precision = torch.exp(-log_vars[idx])
            loss_scaled = self._maybe_scale(name, losses[name])
            term = precision * loss_scaled + log_vars[idx]
            weighted_terms.append(term)
            weights_display[name] = float(precision.detach().cpu().item())

        sum_w = sum(max(v, 1e-8) for v in weights_display.values())
        if sum_w > 0:
            weights_display = {k: v / sum_w for k, v in weights_display.items()}

        if self.weight_ema_beta > 0.0:
            ema = {}
            for k, v in weights_display.items():
                prev = self._ema_weights.get(k, v)
                cur = self.weight_ema_beta * prev + (1.0 - self.weight_ema_beta) * v
                ema[k] = cur
            self._ema_weights.update(ema)
            weights_display = ema

        return sum(weighted_terms), weights_display

    def _dwa_weighting(
        self, losses: Dict[str, torch.Tensor], epoch: int
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        for name, loss in losses.items():
            if name in self.loss_history:
                self.loss_history[name].append(float(loss.detach().cpu().item()))
            elif not self.loss_names:
                # 如果运行时发现新的 loss 且未指定名单，则动态跟踪
                self.loss_history[name] = deque(maxlen=2)
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


__all__ = ["AdaptiveLossWeighting", "build_adaptive_loss"]
