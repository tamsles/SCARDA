"""Oracle importance weights under class-prior shift (SPEC §5).

Privileged module: besides evaluate.py, the only code allowed to read true labels, and only the
S_tr labels (``LabelVault.source_labels``).

With p(x|y) shared across domains and SCAR with a class-independent c_d in each domain,

    p_d(y | Ybar_k=1) = pi_{d,y} 1[y!=k] / (1 - pi_{d,k})
    p_d(y | Ybar_k=0) = pi_{d,y} (1 - c_d 1[y!=k]) / (1 - pibar_{d,k}),  pibar_{d,k} = c_d (1 - pi_{d,k})

and the weight of a source sample with label y in group (k,s) is p_te(y|g) / p_tr(y|g):

    w+_k(y) = [pi_te,y / (1 - pi_te,k)] / [pi_tr,y / (1 - pi_tr,k)]                       (y != k)
    w-_k(y) = [pi_te,y (1 - c_te 1[y!=k]) / (1 - pibar_te,k)]
              / [pi_tr,y (1 - c_tr 1[y!=k]) / (1 - pibar_tr,k)]

All of pi_tr, pi_te, c_tr, c_te and pibar_{d,k} are population values (SPEC §5). The weights are
not renormalized to sample mean 1; the self-normalization of scar_ovr_risk (§3.1) still applies,
under which the per-group constants (1 - pi_{d,k}), (1 - pibar_{d,k}) cancel.
Entries outside the group (w_pos where ybar=0, w_neg where ybar=1) are set to 0.
"""
from __future__ import annotations

import numpy as np
import torch

from scarda.data import LabelVault, WeakSet, gen_ybar, target_prior, uniform_prior


def group_weight_tables(pi_tr, pi_te, c_tr: float, c_te: float) -> tuple[np.ndarray, np.ndarray]:
    """(W+[k, y], W-[k, y]) as float64 [K, K]. W+[k, k] is set to 0 (impossible: ybar_k=1 => y!=k)."""
    pi_tr = np.asarray(pi_tr, dtype=np.float64)
    pi_te = np.asarray(pi_te, dtype=np.float64)
    K = len(pi_tr)
    if pi_te.shape != (K,):
        raise ValueError("pi_tr and pi_te must have the same shape [K]")
    if (pi_tr <= 0).any() or (pi_te < 0).any():
        raise ValueError("priors must be positive (pi_tr) / non-negative (pi_te)")
    off = 1.0 - np.eye(K)  # off[k, y] = 1[y != k]
    pibar_tr = c_tr * (1.0 - pi_tr)
    pibar_te = c_te * (1.0 - pi_te)
    p_pos_te = pi_te[None, :] * off / (1.0 - pi_te)[:, None]
    p_pos_tr = pi_tr[None, :] * off / (1.0 - pi_tr)[:, None]
    p_neg_te = pi_te[None, :] * (1.0 - c_te * off) / (1.0 - pibar_te)[:, None]
    p_neg_tr = pi_tr[None, :] * (1.0 - c_tr * off) / (1.0 - pibar_tr)[:, None]
    W_pos = np.divide(p_pos_te, p_pos_tr, out=np.zeros((K, K)), where=off > 0)
    W_neg = p_neg_te / p_neg_tr
    return W_pos, W_neg


def oracle_weights(y, ybar, pi_tr, pi_te, c_tr: float, c_te: float,
                   dtype: torch.dtype = torch.float32) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-sample oracle weights (w_pos[n,K], w_neg[n,K]) for scar_ovr_risk."""
    y = torch.as_tensor(y).long().cpu()
    yb = torch.as_tensor(ybar).cpu()
    n, K = yb.shape
    if y.shape != (n,):
        raise ValueError(f"y must have shape [{n}], got {tuple(y.shape)}")
    if bool(yb[torch.arange(n), y].any()):
        raise ValueError("ybar contains the true class (violates SCAR)")
    W_pos, W_neg = group_weight_tables(pi_tr, pi_te, c_tr, c_te)
    Wp = torch.from_numpy(W_pos).T[y]  # [n, K]: entry (i, k) = W+[k, y_i]
    Wn = torch.from_numpy(W_neg).T[y]
    mask = yb.to(torch.float64)
    return (Wp * mask).to(dtype), (Wn * (1.0 - mask)).to(dtype)


def oracle_weights_from_vault(vault: LabelVault, s_tr: WeakSet, cfg) -> tuple[torch.Tensor, torch.Tensor]:
    """Oracle weights for the S_tr pool of a run (population pi_tr, pi_te, c_tr, c_te)."""
    if s_tr.name != "S_tr":
        raise ValueError("oracle weights are defined for the S_tr pool only")
    c_tr, c_te = cfg.effective_c
    return oracle_weights(vault.source_labels(), s_tr.ybar, uniform_prior(cfg.K), target_prior(cfg.K, cfg.rho),
                          c_tr, c_te)


def fresh_source_pool(vault: LabelVault, s_tr: WeakSet, cfg, rng: np.random.Generator
                      ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    """Independent redraw of the S_tr complementary labels, with matching oracle weights.

    For the risk-identity check only (scripts/check_identity.py), never for training: a classifier
    trained on S_tr.ybar is not independent of S_tr.ybar, so its weak risk on those labels is biased
    low. The redraw keeps x and y of S_tr, uses the same generator (cfg.gen, c_tr) and `rng`, which
    the caller must derive independently of the training label stream.
    Returns (ybar uint8 [n,K], (w_pos, w_neg)).
    """
    if s_tr.name != "S_tr":
        raise ValueError("the fresh redraw is defined for the S_tr pool only")
    y = vault.source_labels()
    if len(y) != len(s_tr):
        raise ValueError("vault source labels and S_tr pool have different lengths")
    c_tr, c_te = cfg.effective_c
    ybar = torch.from_numpy(gen_ybar(y.numpy(), cfg.K, cfg.gen, c_tr, rng))
    weights = oracle_weights(y, ybar, uniform_prior(cfg.K), target_prior(cfg.K, cfg.rho), c_tr, c_te)
    return ybar, weights
