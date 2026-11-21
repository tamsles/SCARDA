"""T5: oracle weights (SPEC §5) on synthetic 3-class Gaussian mixture + class-prior shift."""
import numpy as np
import pytest
import torch

import _synth as S
from scarda.config import Config
from scarda.data import build_from_arrays, gen_ybar, target_prior, uniform_prior
from scarda.oracle import group_weight_tables, oracle_weights, oracle_weights_from_vault
from scarda.train import fit, method_setup


def _groups(w_pos, w_neg, ybar):
    """Yield (k, s, in-group weights) for the 2K groups."""
    yb = torch.as_tensor(ybar).bool()
    for k in range(yb.shape[1]):
        yield k, "+", w_pos[yb[:, k], k]
        yield k, "-", w_neg[~yb[:, k], k]


def test_T5_group_mean_weight_is_one_large_sample():
    """Each (k,s) group mean of the oracle weights is within 3 SE of 1 (SE = sd/sqrt(m))."""
    rng = np.random.default_rng(20261121)
    n = 200_000
    _, y, ybar = S.sample(n, S.PI_TR, S.C_TR, rng)
    w_pos, w_neg = oracle_weights(y, ybar, S.PI_TR, S.PI_TE, S.C_TR, S.C_TE, dtype=torch.float64)
    for k, s, w in _groups(w_pos, w_neg, ybar):
        m = len(w)
        assert m > 1000, (k, s, m)
        mean, se = float(w.mean()), float(w.std(unbiased=True)) / np.sqrt(m)
        assert abs(mean - 1.0) <= 3 * se, f"group ({k},{s}): mean {mean:.5f}, se {se:.5f}, m {m}"


@pytest.mark.parametrize("rho", [1.0, 5.0, 20.0])
@pytest.mark.parametrize("K,c_tr,c_te", [(3, 0.3, 0.2), (10, 0.3, 0.2), (10, 1 / 9, 1 / 9), (4, 0.0, 0.5)])
def test_T5_population_group_mean_exact(K, c_tr, c_te, rho):
    """sum_y p_tr(y | g) w_g(y) = 1 exactly for every group g=(k,s) (analytic, not sampled)."""
    pi_tr, pi_te = uniform_prior(K), target_prior(K, rho)
    W_pos, W_neg = group_weight_tables(pi_tr, pi_te, c_tr, c_te)
    off = 1.0 - np.eye(K)
    if c_tr > 0:
        p_pos_tr = pi_tr[None, :] * off / (1 - pi_tr)[:, None]  # p_tr(y | Ybar_k=1), rows k
        np.testing.assert_allclose((p_pos_tr * W_pos).sum(axis=1), 1.0, rtol=0, atol=1e-12)
    p_neg_tr = pi_tr[None, :] * (1 - c_tr * off) / (1 - c_tr * (1 - pi_tr))[:, None]
    np.testing.assert_allclose(p_neg_tr.sum(axis=1), 1.0, atol=1e-12)
    np.testing.assert_allclose((p_neg_tr * W_neg).sum(axis=1), 1.0, rtol=0, atol=1e-12)


def test_weights_are_one_without_shift():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 10, 500)
    ybar = gen_ybar(y, 10, "bernoulli", 0.3, rng)
    w_pos, w_neg = oracle_weights(y, ybar, uniform_prior(10), target_prior(10, 1.0), 0.3, 0.3, dtype=torch.float64)
    yb = torch.as_tensor(ybar).bool()
    torch.testing.assert_close(w_pos[yb], torch.ones(int(yb.sum()), dtype=torch.float64), rtol=0, atol=1e-12)
    torch.testing.assert_close(w_neg[~yb], torch.ones(int((~yb).sum()), dtype=torch.float64), rtol=0, atol=1e-12)


def test_weights_zero_outside_group_and_match_formula():
    rng = np.random.default_rng(1)
    K, c_tr, c_te = 4, 0.3, 0.2
    pi_tr, pi_te = uniform_prior(K), target_prior(K, 20.0)
    y = rng.integers(0, K, 300)
    ybar = gen_ybar(y, K, "bernoulli", c_tr, rng)
    w_pos, w_neg = oracle_weights(y, ybar, pi_tr, pi_te, c_tr, c_te, dtype=torch.float64)
    yb = torch.as_tensor(ybar).bool()
    assert bool((w_pos[~yb] == 0).all()) and bool((w_neg[yb] == 0).all())
    pb_tr, pb_te = c_tr * (1 - pi_tr), c_te * (1 - pi_te)
    for i in range(30):
        for k in range(K):
            yi = int(y[i])
            if ybar[i, k]:
                ref = (pi_te[yi] / (1 - pi_te[k])) / (pi_tr[yi] / (1 - pi_tr[k]))
                assert float(w_pos[i, k]) == pytest.approx(ref, rel=1e-12)
            else:
                a = 0.0 if yi == k else 1.0
                ref = (pi_te[yi] * (1 - c_te * a) / (1 - pb_te[k])) / (pi_tr[yi] * (1 - c_tr * a) / (1 - pb_tr[k]))
                assert float(w_neg[i, k]) == pytest.approx(ref, rel=1e-12)


def test_rejects_ybar_containing_true_class():
    y = np.array([0, 1])
    ybar = np.array([[1, 0, 0], [0, 0, 1]], dtype=np.uint8)
    with pytest.raises(ValueError):
        oracle_weights(y, ybar, uniform_prior(3), uniform_prior(3), 0.3, 0.3)


# ----------------------------------------------------------------------------- oracle_iw training path
def _tiny_pools(method):
    rng = np.random.default_rng(5)
    K = 10
    y_tr = rng.permutation(np.repeat(np.arange(K), 700))
    y_te = rng.permutation(np.repeat(np.arange(K), 150))
    x_tr = torch.from_numpy(rng.standard_normal((len(y_tr), 1, 2, 2)).astype(np.float32))
    x_te = torch.from_numpy(rng.standard_normal((len(y_te), 1, 2, 2)).astype(np.float32))
    cfg = Config(method=method, rho=20, n_tr=1500, n_te=100, n_te_max=400, n_val=400, n_eval=300, seed=0,
                 hidden=8, device="cpu")
    pools, vault, _ = build_from_arrays(x_tr, y_tr, x_te, y_te, cfg)
    return cfg, pools, vault


def test_oracle_iw_trains_and_logs_weight_diagnostics():
    cfg, pools, vault = _tiny_pools("oracle_iw")
    sw = oracle_weights_from_vault(vault, pools["S_tr"], cfg)
    _, _, pibar = method_setup(cfg, pools, sw)
    torch.testing.assert_close(pibar, pools["T_tr"].pibar_hat())
    model, hist = fit(cfg, pools, lambda m: {"target_acc": 0.0, "target_bal_acc": 0.0}, epochs=2, source_weights=sw)
    assert len(hist) == 2 and isinstance(model, torch.nn.Module)
    for r in hist:
        assert np.isfinite(r["train_loss_mean"])
        assert 0 < r["ess_pos_mean"] <= 1 and 0 < r["ess_neg_mean"] <= 1
        assert r["w_max"] >= 1
        assert np.isnan(r["fallback"]) and np.isnan(r["kmm_iters_mean"])


def test_source_weights_only_for_oracle_iw():
    cfg, pools, vault = _tiny_pools("oracle_iw")
    with pytest.raises(ValueError):
        method_setup(cfg, pools, None)
    cfg2, pools2, vault2 = _tiny_pools("source_only")
    sw = oracle_weights_from_vault(vault2, pools2["S_tr"], cfg2)
    with pytest.raises(ValueError):
        method_setup(cfg2, pools2, sw)
