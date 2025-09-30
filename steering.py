from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Optional, Literal
import torch
import torch.nn as nn

SteerMode = Literal["none", "shift", "weight", "topk"]


@dataclass
class SteeringConfig:
    mode: SteerMode = "none"
    tau: float = 0.05
    topk_frac: float = 0.25
    mix_lambda: float = 1.0
    distill_gamma: float = 0.0


class SteeringEngine:
    def __init__(
        self,
        cfg: SteeringConfig,
        predict_fn: Optional[Callable[[nn.Module, torch.Tensor], torch.Tensor]] = None,
        loss_per_example: Optional[
            Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
        ] = None,
    ):
        self.cfg = cfg
        self.predict_fn = predict_fn or (lambda m, X: m(X))
        if loss_per_example is None:

            def _mse_per_example(p, y):
                if p.ndim > 1:
                    axes = tuple(range(1, p.ndim))
                    return torch.mean((p - y) ** 2, dim=axes)
                return (p - y) ** 2

            self.loss_per_example = _mse_per_example
        else:
            self.loss_per_example = loss_per_example

    @torch.no_grad()
    def _ref_forward(self, ref_model: nn.Module, X: torch.Tensor) -> torch.Tensor:
        return self.predict_fn(ref_model, X)

    def compute_loss(
        self,
        model: nn.Module,
        X: torch.Tensor,
        y_loss: torch.Tensor,
        ref_model: Optional[nn.Module] = None,
        extra_reduce: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> torch.Tensor:
        preds = self.predict_fn(model, X)
        per_ex = self.loss_per_example(preds, y_loss)

        kd_term = 0.0
        if ref_model is not None and self.cfg.distill_gamma > 0:
            with torch.no_grad():
                preds_ref = self._ref_forward(ref_model, X)
            kd_term = self.cfg.distill_gamma * torch.mean((preds - preds_ref) ** 2)

        if ref_model is None or self.cfg.mode == "none":
            base = per_ex.mean()
            return base + (kd_term if isinstance(kd_term, torch.Tensor) else 0.0)

        with torch.no_grad():
            preds_ref = self._ref_forward(ref_model, X)
            per_ex_ref = self.loss_per_example(preds_ref, y_loss)
        rho = per_ex - per_ex_ref

        mode = self.cfg.mode
        if mode == "shift":
            loss_main = per_ex.mean()
            loss_rho = rho.mean()
            return (
                (1 - self.cfg.mix_lambda) * loss_main
                + self.cfg.mix_lambda * loss_rho
                + kd_term
            )

        if mode == "weight":
            w = torch.softmax(rho / max(self.cfg.tau, 1e-8), dim=0).detach()
            return (w * per_ex).sum() + kd_term

        if mode == "topk":
            k = max(1, int(self.cfg.topk_frac * per_ex.shape[0]))
            idx = torch.topk(rho, k=k, largest=True).indices
            return per_ex[idx].mean() + kd_term

        raise ValueError(f"Unknown steering mode: {mode}")
