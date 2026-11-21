"""Generalized SCAR-OVR empirical risk (SPEC §3.1-3.2).

Notation (SPEC §2): ell(z) = log(1 + exp(-z)); f_k is the k-th logit;
ybar[i, k] = 1 means k is a complementary label of sample i (multi-hot, all-zero rows allowed).

    M+_k(g) = sum_{i: ybar_ik=1} w+_ik g_ik / sum_{i: ybar_ik=1} w+_ik
    M-_k(g) = sum_{i: ybar_ik=0} w-_ik g_ik / sum_{i: ybar_ik=0} w-_ik
    N_k     = (1 - pi_k) M+_k(ell(-f_k))
    P_k     = (pibar_k + pi_k - 1) M+_k(ell(f_k)) + (1 - pibar_k) M-_k(ell(f_k))
    R       = sum_k [ N_k + corr(P_k) ]

A group whose weight sum is 0 (in particular an empty group) has mean 0 and is counted,
matching the denominator-set-to-1 convention of SCARCE_loss (utils_algo.py L22-23).
Self-normalization makes this a ratio estimator, biased in finite samples.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import torch
import torch.nn.functional as F

Correction = Literal["abs", "relu", "none"]
PibarMode = Literal["batch", "fixed"]


@dataclass
class RiskOut:
    total: torch.Tensor  # scalar, keeps the autograd graph
    N: torch.Tensor  # [K], detached
    P_raw: torch.Tensor  # [K], before correction, detached
    neg_rate: float  # fraction of the K classes with P_raw < 0 (strict)
    empty_pos: int  # number of empty (k,+) groups (weight sum == 0)
    empty_neg: int  # number of empty (k,-) groups (weight sum == 0)


def logistic_ell(z: torch.Tensor) -> torch.Tensor:
    """ell(z) = log(1 + exp(-z)); identical to utils_algo.logistic_loss."""
    return F.softplus(-z)


def apply_correction(P: torch.Tensor, correction: Correction) -> torch.Tensor:
    if correction == "abs":
        return P.abs()
    if correction == "relu":
        return P.clamp_min(0.0)
    if correction == "none":
        return P
    raise ValueError(f"unknown correction: {correction!r}")


def _group_mean(weights: torch.Tensor, values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Column-wise weighted mean; returns (mean[K], empty_mask[K]). Empty columns get mean 0."""
    s = weights.sum(dim=0)
    empty = s == 0
    denom = torch.where(empty, torch.ones_like(s), s)
    return (weights * values).sum(dim=0) / denom, empty


def scar_ovr_risk(
    logits: torch.Tensor,
    ybar: torch.Tensor,
    pi: torch.Tensor,
    pibar: Optional[torch.Tensor] = None,
    w_pos: Optional[torch.Tensor] = None,
    w_neg: Optional[torch.Tensor] = None,
    correction: Correction = "abs",
    pibar_mode: PibarMode = "batch",
) -> RiskOut:
    if logits.dim() != 2:
        raise ValueError(f"logits must be [n,K], got {tuple(logits.shape)}")
    n, K = logits.shape
    if tuple(ybar.shape) != (n, K):
        raise ValueError(f"ybar shape {tuple(ybar.shape)} != logits shape {(n, K)}")
    if not bool(((ybar == 0) | (ybar == 1)).all()):
        raise ValueError("ybar must be a {0,1} matrix")
    if correction not in ("abs", "relu", "none"):
        raise ValueError(f"unknown correction: {correction!r}")

    dtype, device = logits.dtype, logits.device
    Yp = ybar.to(dtype=dtype, device=device)
    Yn = 1.0 - Yp

    if pibar_mode == "batch":
        if pibar is not None:
            raise ValueError("pibar_mode='batch' computes pibar from ybar; pibar must be None")
        if w_pos is not None or w_neg is not None:
            raise ValueError("pibar_mode='batch' is only for unweighted risks (SPEC §3.2)")
        pibar_v = Yp.mean(dim=0)
    elif pibar_mode == "fixed":
        if pibar is None:
            raise ValueError("pibar_mode='fixed' requires pibar")
        pibar_v = torch.as_tensor(pibar, dtype=dtype, device=device)
    else:
        raise ValueError(f"unknown pibar_mode: {pibar_mode!r}")

    pi_v = torch.as_tensor(pi, dtype=dtype, device=device)
    for name, v in (("pi", pi_v), ("pibar", pibar_v)):
        if tuple(v.shape) != (K,):
            raise ValueError(f"{name} must have shape [K]={K}, got {tuple(v.shape)}")

    def _eff(w: Optional[torch.Tensor], mask: torch.Tensor, name: str) -> torch.Tensor:
        if w is None:
            return mask
        w = torch.as_tensor(w, dtype=dtype, device=device)
        if tuple(w.shape) != (n, K):
            raise ValueError(f"{name} must be [n,K], got {tuple(w.shape)}")
        if bool((w < 0).any()):
            raise ValueError(f"{name} must be non-negative")
        return w * mask  # entries outside the group are ignored

    Ap = _eff(w_pos, Yp, "w_pos")
    An = _eff(w_neg, Yn, "w_neg")

    loss_f = logistic_ell(logits)  # ell(f_k)
    loss_mf = logistic_ell(-logits)  # ell(-f_k)

    Mp_mf, empty_p = _group_mean(Ap, loss_mf)  # M+_k(ell(-f_k))
    Mp_f, _ = _group_mean(Ap, loss_f)  # M+_k(ell(f_k))
    Mn_f, empty_n = _group_mean(An, loss_f)  # M-_k(ell(f_k))

    N = (1.0 - pi_v) * Mp_mf
    P_raw = (pibar_v + pi_v - 1.0) * Mp_f + (1.0 - pibar_v) * Mn_f
    total = (N + apply_correction(P_raw, correction)).sum()

    return RiskOut(
        total=total,
        N=N.detach(),
        P_raw=P_raw.detach(),
        neg_rate=float((P_raw.detach() < 0).to(dtype).mean().item()),
        empty_pos=int(empty_p.sum().item()),
        empty_neg=int(empty_n.sum().item()),
    )
