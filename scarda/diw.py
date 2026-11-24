"""DIW weights: per-mini-batch KMM on the hidden representation, per complementary-label group (SPEC §6).

For every group (k, s), s in {+, -}:  G_tr = {i in B_tr : ybar_ik = s},  G_te = {j in B_te : ybar_jk = s},
m = |G_tr|, m' = |G_te|, and

    min_w  1/2 w^T K w - kappa^T w,   K_ij = k(z_i, z_j) (i, j in G_tr),
                                      kappa_i = (m / m') sum_{j in G_te} k(z_i, z_j)
    s.t.   0 <= w_i <= B,  |sum_i w_i - m| <= m * eps.

k(a, b) = exp(-||a - b||^2 / (2 sigma^2)), sigma = median pairwise distance over B_tr ∪ B_te of the step.
If |G_tr| < min_group or |G_te| < min_group (empty groups included) the group gets weight 1 and is
counted as a fallback.

Solver: projected gradient, started from w = 1 at every step (no warm start across steps), step size
1 / lambda_max(K), run until the relative objective change is < 1e-6 or 1000 iterations. The
projection onto {box} ∩ {sum band} is w = clip(v - lambda, 0, B) with the scalar shift lambda found
by bisection. All solvable groups of a step are padded and solved as one batch; each group stops
on its own criterion. Computation is in float64.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

KMM_TOL = 1e-6  # relative objective change
KMM_MAX_ITER = 1000
N_BISECT = 60  # bracket width shrinks by 2^-60: below float64 resolution for these magnitudes


@dataclass
class KMMOut:
    w: torch.Tensor  # [G, M] (0 on padding)
    iters: torch.Tensor  # [G] int64, PGD iterations performed per group
    obj: torch.Tensor  # [G] final objective


@dataclass
class DIWOut:
    w_pos: torch.Tensor  # [n_tr, K]: weight of sample i in group (k,+), 0 outside the group
    w_neg: torch.Tensor  # [n_tr, K]: weight of sample i in group (k,-), 0 outside the group
    fallback: int  # number of (k,s) groups with |G_tr| < min_group or |G_te| < min_group
    iters: torch.Tensor  # [n_solved] PGD iterations of the solved groups
    sigma: float


def median_bandwidth(z: torch.Tensor) -> float:
    """Median of the pairwise distances ||z_i - z_j||, i < j."""
    n = z.shape[0]
    iu = torch.triu_indices(n, n, offset=1, device=z.device)
    return float(torch.cdist(z, z)[iu[0], iu[1]].median().item())


def rbf_kernel(a: torch.Tensor, b: torch.Tensor, sigma: float) -> torch.Tensor:
    return torch.exp(-torch.cdist(a, b).pow(2) / (2.0 * sigma * sigma))


def project_box_sum(v: torch.Tensor, mask: torch.Tensor, m: torch.Tensor, B: float, eps: float) -> torch.Tensor:
    """Euclidean projection of each row of v (entries where mask=1) onto
    {0 <= w <= B, m(1-eps) <= sum w <= m(1+eps)}: w = clip(v - lam, 0, B), lam by bisection.

    s(lam) = sum clip(v - lam, 0, B) is non-increasing; lam = 0 if s(0) is inside the band, otherwise
    lam solves s(lam) = nearest band edge, bracketed by [min v - B, max v] (s = m*B and 0 there).
    v, mask: [G, M]; m: [G] (group sizes, float). Padding entries come back as 0.
    """
    lo_t, hi_t = m * (1.0 - eps), m * (1.0 + eps)
    s0 = (v.clamp(0.0, B) * mask).sum(dim=1)
    need = (s0 > hi_t) | (s0 < lo_t)
    w = v.clamp(0.0, B) * mask
    if not bool(need.any()):
        return w
    target = torch.where(s0 > hi_t, hi_t, lo_t)
    big = torch.finfo(v.dtype).max
    lo = torch.where(mask > 0, v, torch.full_like(v, big)).amin(dim=1) - B  # s(lo) = m*B >= target
    hi = torch.where(mask > 0, v, torch.full_like(v, -big)).amax(dim=1)  # s(hi) = 0 <= target
    for _ in range(N_BISECT):
        mid = 0.5 * (lo + hi)
        s = ((v - mid[:, None]).clamp(0.0, B) * mask).sum(dim=1)
        above = s > target
        lo = torch.where(above, mid, lo)
        hi = torch.where(above, hi, mid)
    lam = torch.where(need, 0.5 * (lo + hi), torch.zeros_like(lo))
    return (v - lam[:, None]).clamp(0.0, B) * mask


def kmm_objective(Kmat: torch.Tensor, kappa: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    return 0.5 * (w * torch.bmm(Kmat, w[:, :, None])[:, :, 0]).sum(dim=1) - (kappa * w).sum(dim=1)


def kmm_pgd(Kmat: torch.Tensor, kappa: torch.Tensor, mask: torch.Tensor, B: float, eps: float,
            tol: float = KMM_TOL, max_iter: int = KMM_MAX_ITER) -> KMMOut:
    """Batched projected gradient for the KMM QP. Kmat [G,M,M], kappa/mask [G,M] (padding: 0).

    Starts from w = 1 (feasible). Group g stops after the first iteration t with
    |f_t - f_{t-1}| < tol * |f_{t-1}|, or after max_iter iterations.
    """
    G = Kmat.shape[0]
    m = mask.sum(dim=1)
    L = torch.linalg.eigvalsh(Kmat)[:, -1]  # lambda_max; zero padding adds only zero eigenvalues
    w = mask.clone()
    f = kmm_objective(Kmat, kappa, w)
    iters = torch.zeros(G, dtype=torch.int64, device=Kmat.device)
    active = torch.ones(G, dtype=torch.bool, device=Kmat.device)
    for _ in range(max_iter):
        grad = torch.bmm(Kmat, w[:, :, None])[:, :, 0] - kappa
        w_new = project_box_sum(w - grad / L[:, None], mask, m, B, eps)
        f_new = kmm_objective(Kmat, kappa, w_new)
        w = torch.where(active[:, None], w_new, w)
        iters += active.long()
        converged = (f_new - f).abs() < tol * f.abs()
        f = torch.where(active, f_new, f)
        active = active & ~converged
        if not bool(active.any()):
            break
    return KMMOut(w=w, iters=iters, obj=f)


@torch.no_grad()
def diw_weights(z_tr: torch.Tensor, ybar_tr: torch.Tensor, z_te: torch.Tensor, ybar_te: torch.Tensor,
                B: float, eps: float, min_group: int) -> DIWOut:
    """DIW weights of one step (SPEC §6.3-6.7). z_* are the (detached) representations of B_tr / B_te."""
    n_tr, K = ybar_tr.shape
    z_tr, z_te = z_tr.detach().double(), z_te.detach().double()
    if not (bool(torch.isfinite(z_tr).all()) and bool(torch.isfinite(z_te).all())):
        raise FloatingPointError("DIW: non-finite representation")
    sigma = median_bandwidth(torch.cat([z_tr, z_te]))
    if not (sigma > 0.0):
        raise FloatingPointError(f"DIW: degenerate RBF bandwidth sigma={sigma}")

    # group g = 2k + (0 for s=+, 1 for s=-); membership matrices [n, 2K]
    yt, ye = ybar_tr.double(), ybar_te.double()
    Mtr = torch.stack([yt, 1.0 - yt], dim=2).reshape(n_tr, 2 * K)
    Mte = torch.stack([ye, 1.0 - ye], dim=2).reshape(ye.shape[0], 2 * K)
    m_tr, m_te = Mtr.sum(dim=0), Mte.sum(dim=0)
    solve = (m_tr >= min_group) & (m_te >= min_group)
    W = Mtr.clone()  # fallback groups (and groups solved below get overwritten): weight 1 in group
    fallback = int((~solve).sum().item())

    gs = torch.nonzero(solve).flatten()
    iters = torch.zeros(0, dtype=torch.int64, device=z_tr.device)
    if gs.numel() > 0:
        K_tr = rbf_kernel(z_tr, z_tr, sigma)
        C = rbf_kernel(z_tr, z_te, sigma) @ Mte[:, gs]  # C[i, g] = sum_{j in G_te(g)} k(z_i, z_j)
        sizes = m_tr[gs].long()
        M = int(sizes.max().item())
        # padded member indices of each solved group, in ascending sample order
        member = Mtr[:, gs].T > 0  # [G, n_tr]
        order = torch.argsort((~member).to(torch.int8), dim=1, stable=True)[:, :M]  # members first
        mask = (torch.arange(M, device=z_tr.device)[None, :] < sizes[:, None]).double()
        Kmat = K_tr[order[:, :, None], order[:, None, :]] * mask[:, :, None] * mask[:, None, :]
        kappa = torch.gather(C.T, 1, order) * (m_tr[gs] / m_te[gs])[:, None] * mask
        res = kmm_pgd(Kmat, kappa, mask, B, eps)
        Wg = torch.zeros(len(gs), n_tr, dtype=torch.float64, device=z_tr.device)
        Wg.scatter_(1, order, res.w)  # padding writes 0 (res.w is 0 there) onto non-members
        W[:, gs] = (Wg * member).T
        iters = res.iters
    if not bool(torch.isfinite(W).all()):
        raise FloatingPointError("DIW: non-finite weights")
    W = W.reshape(n_tr, K, 2)
    return DIWOut(w_pos=W[:, :, 0], w_neg=W[:, :, 1], fallback=fallback, iters=iters, sigma=sigma)
