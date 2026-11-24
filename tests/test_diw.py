"""T7: DIW / KMM solver (SPEC §6, TASKS M3.2)."""
from __future__ import annotations

import numpy as np
import pytest
import torch
from scipy.optimize import minimize

from scarda.diw import diw_weights, kmm_pgd, median_bandwidth, rbf_kernel

B, EPS = 10.0, 0.05
# (dim, m, m', target shift); several sizes so that the batch is padded
PROBLEMS = [(2, 15, 10, 0.5), (5, 30, 20, 0.7), (5, 24, 31, 0.0), (50, 40, 30, 0.3)]


def _problem(d, m, mp, shift, seed):
    g = torch.Generator().manual_seed(seed)
    zt = torch.randn(m, d, generator=g, dtype=torch.float64)
    ze = torch.randn(mp, d, generator=g, dtype=torch.float64) + shift
    s = median_bandwidth(torch.cat([zt, ze]))
    return rbf_kernel(zt, zt, s), m / mp * rbf_kernel(zt, ze, s).sum(dim=1)


def _batch(problems):
    """Pad the problems into one batch as diw_weights does."""
    M = max(K.shape[0] for K, _ in problems)
    G = len(problems)
    Kb, kb, mask = (torch.zeros(G, M, M, dtype=torch.float64), torch.zeros(G, M, dtype=torch.float64),
                    torch.zeros(G, M, dtype=torch.float64))
    for g, (K, kap) in enumerate(problems):
        m = K.shape[0]
        Kb[g, :m, :m], kb[g, :m], mask[g, :m] = K, kap, 1.0
    return Kb, kb, mask


def _scipy(K, kap):
    Kn, kn = K.numpy(), kap.numpy()
    m = len(kn)
    cons = [{"type": "ineq", "fun": lambda w: m * (1 + EPS) - w.sum(), "jac": lambda w: -np.ones(m)},
            {"type": "ineq", "fun": lambda w: w.sum() - m * (1 - EPS), "jac": lambda w: np.ones(m)}]
    res = minimize(lambda w: 0.5 * w @ Kn @ w - kn @ w, np.ones(m), jac=lambda w: Kn @ w - kn,
                   bounds=[(0.0, B)] * m, constraints=cons, method="SLSQP",
                   options={"ftol": 1e-12, "maxiter": 5000})
    assert res.success, res.message
    return float(res.fun)


def _check_feasible(w, mask):
    m = mask.sum(dim=1)
    assert bool(((w >= 0) & (w <= B)).all())
    assert bool((w[mask == 0] == 0).all())  # padding stays 0
    s = (w * mask).sum(dim=1)
    tol = 1e-9 * m
    assert bool((s >= m * (1 - EPS) - tol).all()) and bool((s <= m * (1 + EPS) + tol).all())


def _rel_gaps(res, problems):
    gaps = []
    for g, (K, kap) in enumerate(problems):
        ref = _scipy(K, kap)
        gaps.append(abs(float(res.obj[g]) - ref) / abs(ref))
    return gaps


@pytest.mark.parametrize("seed", [0, 1])
def test_T7_pgd_converged_matches_scipy(seed):
    """Algorithm and projection: run to convergence, PGD reaches the SLSQP optimum."""
    problems = [_problem(*p, seed=100 * seed + i) for i, p in enumerate(PROBLEMS)]
    Kb, kb, mask = _batch(problems)
    res = kmm_pgd(Kb, kb, mask, B, EPS, tol=0.0, max_iter=5000)
    _check_feasible(res.w, mask)
    assert max(_rel_gaps(res, problems)) < 1e-4


@pytest.mark.parametrize("seed", [0, 1])
def test_T7_pgd_spec_rule_matches_scipy(seed):
    """T7 with the SPEC §6 stopping rule (solver defaults). Tolerance relaxed from 1e-4 to 1e-3 by
    human decision (M3): the rule stops early, gaps observed 4.3e-5..6.3e-4 (see TASKS M3 record)."""
    problems = [_problem(*p, seed=100 * seed + i) for i, p in enumerate(PROBLEMS)]
    Kb, kb, mask = _batch(problems)
    res = kmm_pgd(Kb, kb, mask, B, EPS)
    assert max(_rel_gaps(res, problems)) < 1e-3


@pytest.mark.parametrize("seed", [0, 1])
def test_T7_pgd_spec_rule_feasible(seed):
    """Constraints hold for the solver defaults as well (every iterate is a projection)."""
    problems = [_problem(*p, seed=100 * seed + i) for i, p in enumerate(PROBLEMS)]
    Kb, kb, mask = _batch(problems)
    res = kmm_pgd(Kb, kb, mask, B, EPS)
    _check_feasible(res.w, mask)
    assert bool((res.iters >= 1).all()) and bool((res.iters <= 1000).all())


def test_T7_same_samples_weight_one():
    """G_te = G_tr (same samples): w = 1 satisfies K w = kappa and is the optimum."""
    g = torch.Generator().manual_seed(0)
    n, K = 64, 4
    z = torch.randn(n, 20, generator=g).relu()
    yb = (torch.rand(n, K, generator=g) < 0.4).float()
    out = diw_weights(z, yb, z.clone(), yb.clone(), B, EPS, min_group=4)
    assert out.fallback == 0
    torch.testing.assert_close(out.w_pos, yb.double(), rtol=0, atol=1e-8)
    torch.testing.assert_close(out.w_neg, 1.0 - yb.double(), rtol=0, atol=1e-8)


def test_T7_fallback_empty_and_small_groups():
    """Groups with |G_tr| < min_group or |G_te| < min_group get weight 1 and are counted."""
    g = torch.Generator().manual_seed(0)
    n_tr, n_te, K = 20, 20, 4
    z_tr, z_te = torch.randn(n_tr, 8, generator=g), torch.randn(n_te, 8, generator=g) + 0.3
    ytr, yte = torch.zeros(n_tr, K), torch.zeros(n_te, K)
    # class 0: (0,+) empty in B_tr -> fallback; (0,-) has 20/20 -> solved
    yte[:10, 0] = 1  # (0,-) in B_te has 10
    # class 1: (1,+) has 3 in B_te (< 4) -> fallback; (1,-) 10/17 -> solved
    ytr[:10, 1] = 1
    yte[:3, 1] = 1
    # class 2: (2,+) exactly 4/4 -> solved (boundary); (2,-) 16/16 -> solved
    ytr[:4, 2] = 1
    yte[:4, 2] = 1
    # class 3: (3,+) 17/20 -> solved; (3,-) has 3 in B_tr and 0 in B_te -> fallback
    ytr[:17, 3] = 1
    yte[:, 3] = 1
    out = diw_weights(z_tr, ytr, z_te, yte, B, EPS, min_group=4)
    assert out.fallback == 3
    assert out.iters.numel() == 2 * K - 3
    # fallback groups: weight 1 on members (class 0 '+' has none), 0 outside the group
    assert bool((out.w_pos[:, 0] == 0).all())
    torch.testing.assert_close(out.w_pos[:, 1], ytr[:, 1].double(), rtol=0, atol=0)
    torch.testing.assert_close(out.w_neg[:, 3], 1.0 - ytr[:, 3].double(), rtol=0, atol=0)
    # solved groups: zero outside the group, feasible inside
    for k, s in ((0, "-"), (1, "-"), (2, "+"), (2, "-"), (3, "+")):
        member = ytr[:, k] if s == "+" else 1.0 - ytr[:, k]
        w = out.w_pos[:, k] if s == "+" else out.w_neg[:, k]
        m = float(member.sum())
        assert bool((w[member == 0] == 0).all())
        assert bool(((w >= 0) & (w <= B)).all())
        assert m * (1 - EPS) - 1e-9 <= float(w.sum()) <= m * (1 + EPS) + 1e-9


def test_T7_fallback_counts_with_larger_min_group():
    """Same data, min_group = 5: the (2,+) group (4/4) now falls back as well."""
    g = torch.Generator().manual_seed(0)
    n, K = 20, 4
    z_tr, z_te = torch.randn(n, 8, generator=g), torch.randn(n, 8, generator=g)
    ytr, yte = torch.zeros(n, K), torch.zeros(n, K)
    ytr[:4, 2] = 1
    yte[:4, 2] = 1
    # every '+' group except class 2 is empty in both batches -> fallback; all '-' groups solved
    out = diw_weights(z_tr, ytr, z_te, yte, B, EPS, min_group=5)
    assert out.fallback == K
    assert out.iters.numel() == K
    torch.testing.assert_close(out.w_pos[:, 2], ytr[:, 2].double(), rtol=0, atol=0)


@pytest.mark.parametrize("n_tr,n_te,K,seed", [(96, 60, 5, 0), (32, 256, 10, 1)])
def test_T7_diw_weights_match_per_group_reference(n_tr, n_te, K, seed):
    """diw_weights vs an independent per-group construction (SPEC §6.3-6.5) with m != m': sigma from
    B_tr ∪ B_te, K over G_tr, kappa_i = (m/m') sum_{j in G_te} k(z_i, z_j). Guards the m/m' factor and
    the bandwidth choice, which the scipy comparison above (kappa built by the test) does not reach."""
    rng = np.random.default_rng(seed)
    z_tr = np.maximum(rng.normal(size=(n_tr, 30)), 0.0)
    z_te = np.maximum(rng.normal(size=(n_te, 30)) + 0.4, 0.0)
    y_tr = (rng.random((n_tr, K)) < 0.27).astype(np.float64)
    y_te = (rng.random((n_te, K)) < 0.18).astype(np.float64)
    out = diw_weights(torch.tensor(z_tr), torch.tensor(y_tr), torch.tensor(z_te), torch.tensor(y_te), B, EPS, 4)

    z_all = np.concatenate([z_tr, z_te])
    d = np.sqrt(((z_all[:, None, :] - z_all[None, :, :]) ** 2).sum(-1))[np.triu_indices(len(z_all), 1)]
    sigma = np.sort(d)[(len(d) - 1) // 2]  # torch.median convention: lower middle value
    assert abs(out.sigma - sigma) <= 1e-12 * sigma

    def k(a, b):
        return np.exp(-((a[:, None, :] - b[None, :, :]) ** 2).sum(-1) / (2.0 * sigma ** 2))

    solved = 0
    for kk in range(K):
        for s, W in ((1.0, out.w_pos), (0.0, out.w_neg)):
            gt, ge = y_tr[:, kk] == s, y_te[:, kk] == s
            w = W[:, kk].numpy()
            assert (w[~gt] == 0).all()
            if gt.sum() < 4 or ge.sum() < 4:
                assert (w[gt] == 1).all()
                continue
            m, mp = gt.sum(), ge.sum()
            Kg = torch.tensor(k(z_tr[gt], z_tr[gt]))[None]
            kap = torch.tensor(m / mp * k(z_tr[gt], z_te[ge]).sum(axis=1))[None]
            ref = kmm_pgd(Kg, kap, torch.ones(1, int(m), dtype=torch.float64), B, EPS).w[0].numpy()
            np.testing.assert_allclose(w[gt], ref, rtol=0, atol=1e-9)
            solved += 1
    assert solved == 2 * K - out.fallback and solved > 0
