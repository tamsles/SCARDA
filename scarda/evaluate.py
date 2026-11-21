"""Final evaluation on T_eval (SPEC §7). One of the two modules allowed to read true labels.

The training loop receives only the opaque callback built by `make_evaluator`; it never sees
the vault, the T_eval inputs or labels.
"""
from __future__ import annotations

from typing import Callable

import torch

from scarda.data import LabelVault


def accuracy(pred: torch.Tensor, y: torch.Tensor) -> float:
    return float((pred == y).double().mean().item())


def balanced_accuracy(pred: torch.Tensor, y: torch.Tensor, K: int) -> float:
    recalls = [(pred[y == k] == k).double().mean().item() for k in range(K) if bool((y == k).any())]
    return float(sum(recalls) / len(recalls))


def make_evaluator(vault: LabelVault, K: int, device, bs: int = 4096) -> Callable[[torch.nn.Module], dict]:
    x, y = vault.eval_set()
    x, y = x.to(device), y.to(device)

    @torch.no_grad()
    def evaluate(model: torch.nn.Module) -> dict:
        was_training = model.training
        model.eval()
        pred = torch.cat([model(x[i:i + bs]).argmax(dim=1) for i in range(0, len(x), bs)])
        model.train(was_training)
        return {"target_acc": accuracy(pred, y), "target_bal_acc": balanced_accuracy(pred, y, K)}

    return evaluate


@torch.no_grad()
def true_ovr_losses(model: torch.nn.Module, vault: LabelVault, device, bs: int = 4096) -> torch.Tensor:
    """Per-sample true-label OVR loss on T_eval, ell(f_y) + sum_{k!=y} ell(-f_k), float64 [n].
    Used only by the risk-identity check (scripts/check_identity.py), never in training."""
    x, y = vault.eval_set()
    was_training = model.training
    model.eval()
    logits = torch.cat([model(x[i:i + bs].to(device)) for i in range(0, len(x), bs)]).double().cpu()
    model.train(was_training)
    on = torch.nn.functional.one_hot(y.long(), logits.shape[1]).double()
    sp = torch.nn.functional.softplus
    return (on * sp(-logits) + (1.0 - on) * sp(logits)).sum(dim=1)
