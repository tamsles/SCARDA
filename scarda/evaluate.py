"""Final evaluation on T_eval (SPEC §7). Together with oracle.py, the only code that reads true labels.

The training loop only receives the callback built by `make_evaluator`; its output is logged and
never used for any decision.
"""
from __future__ import annotations

from typing import Callable

import torch
import torch.nn.functional as F

from scarda.models import predict_logits


def accuracy(pred: torch.Tensor, y: torch.Tensor) -> float:
    return float((pred == y).double().mean().item())


def balanced_accuracy(pred: torch.Tensor, y: torch.Tensor, K: int) -> float:
    recalls = [(pred[y == k] == k).double().mean().item() for k in range(K) if bool((y == k).any())]
    return float(sum(recalls) / len(recalls))


def make_evaluator(eval_x: torch.Tensor, eval_y: torch.Tensor, K: int, device) -> Callable[[torch.nn.Module], dict]:
    x, y = eval_x.to(device), eval_y.to(device)

    def evaluate(model: torch.nn.Module) -> dict:
        pred = predict_logits(model, x, device).argmax(dim=1)
        return {"target_acc": accuracy(pred, y), "target_bal_acc": balanced_accuracy(pred, y, K)}

    return evaluate


def true_ovr_losses(model: torch.nn.Module, eval_x: torch.Tensor, eval_y: torch.Tensor, device) -> torch.Tensor:
    """Per-sample true-label OVR loss on T_eval, ell(f_y) + sum_{k!=y} ell(-f_k), float64 [n].
    Used only by the risk-identity check (scripts/check_identity.py), never in training."""
    logits = predict_logits(model, eval_x, device).double().cpu()
    on = F.one_hot(eval_y.long(), logits.shape[1]).double()
    return (on * F.softplus(-logits) + (1.0 - on) * F.softplus(logits)).sum(dim=1)
