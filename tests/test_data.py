"""T4: data protocol (SPEC §4) on synthetic arrays (no MNIST download)."""
import numpy as np
import pytest
import torch

from scarda.config import Config, load_config
from scarda.data import POOLS, WEAK_POOLS, build_from_arrays, largest_remainder, target_prior, uniform_prior

K = 10


def _base(seed=0, per_class_tr=700, per_class_te=150):
    rng = np.random.default_rng(1000 + seed)
    y_tr = rng.permutation(np.repeat(np.arange(K), per_class_tr))
    y_te = rng.permutation(np.repeat(np.arange(K), per_class_te))
    # x encodes the base index so that rows can be traced back in tests
    x_tr = torch.arange(len(y_tr), dtype=torch.float32).view(-1, 1, 1).expand(-1, 2, 2).clone()
    x_te = torch.arange(len(y_te), dtype=torch.float32).view(-1, 1, 1).expand(-1, 2, 2).clone()
    return x_tr, y_tr, x_te, y_te


def _cfg(**kw):
    d = dict(rho=20, n_tr=1500, n_te=100, n_te_max=400, n_val=400, n_eval=300, seed=0)
    d.update(kw)
    return Config(**d)


def _build(cfg, base=None):
    x_tr, y_tr, x_te, y_te = base if base is not None else _base()
    pools, labels, info = build_from_arrays(x_tr, y_tr, x_te, y_te, cfg)
    return pools, labels, info, (x_tr, y_tr, x_te, y_te)


# ----------------------------------------------------------------------------- largest remainder
@pytest.mark.parametrize("n", [0, 1, 7, 300, 500, 2000, 3000, 20000])
@pytest.mark.parametrize("rho", [1, 5, 20])
def test_largest_remainder_sums_and_is_close(n, rho):
    p = target_prior(K, rho)
    c = largest_remainder(n, p)
    assert c.sum() == n and c.dtype == np.int64
    assert np.all(np.abs(c - n * p) < 1.0)


def test_largest_remainder_ties_and_float_noise():
    assert largest_remainder(500, uniform_prior(K)).tolist() == [50] * K
    assert largest_remainder(3, np.full(4, 0.25)).tolist() == [1, 1, 1, 0]  # ties -> smaller index
    assert largest_remainder(10, np.array([0.55, 0.45])).tolist() == [6, 4]  # 5.5/4.5 -> first wins
    with pytest.raises(ValueError):
        largest_remainder(10, np.array([0.5, 0.6]))


# ----------------------------------------------------------------------------- counts / disjointness
@pytest.mark.parametrize("rho", [1, 5, 20])
def test_pool_counts_exact(rho):
    cfg = _cfg(rho=rho)
    pools, labels, rec, (_, y_tr, _, y_te) = _build(cfg)
    pi_te = target_prior(K, rho)
    expected = {
        "S_tr": largest_remainder(cfg.n_tr, uniform_prior(K)),
        "T_tr": largest_remainder(cfg.n_te, pi_te),
        "T_val": largest_remainder(cfg.n_val, pi_te),
        "T_eval": largest_remainder(cfg.n_eval, pi_te),
    }
    for p in POOLS:
        y = y_te if p == "T_eval" else y_tr
        actual = np.bincount(y[rec["indices"][p]], minlength=K)
        assert actual.tolist() == expected[p].tolist() == rec["target_counts"][p] == rec["actual_counts"][p]
    assert [len(pools[p]) for p in WEAK_POOLS] == [cfg.n_tr, cfg.n_te, cfg.n_val]
    assert len(labels.eval_y) == len(labels.eval_x) == cfg.n_eval


def test_pools_pairwise_disjoint_and_rows_match():
    cfg = _cfg(n_te=400)
    pools, labels, rec, (x_tr, y_tr, x_te, y_te) = _build(cfg)
    sets = {p: set(rec["indices"][p].tolist()) for p in WEAK_POOLS}
    for p in WEAK_POOLS:
        assert len(sets[p]) == len(rec["indices"][p])  # no duplicates
        for q in WEAK_POOLS:
            if p < q:
                assert not (sets[p] & sets[q]), (p, q)
    assert len(set(rec["indices"]["T_eval"].tolist())) == cfg.n_eval
    for p in WEAK_POOLS:  # x rows come from the recorded indices
        assert torch.equal(pools[p].x.flatten(1), x_tr[torch.from_numpy(rec["indices"][p])].flatten(1))
    ex, ey = labels.eval_x, labels.eval_y
    assert torch.equal(ex.flatten(1), x_te[torch.from_numpy(rec["indices"]["T_eval"])].flatten(1))
    assert ey.tolist() == y_te[rec["indices"]["T_eval"]].tolist()
    assert labels.source_y.tolist() == y_tr[rec["indices"]["S_tr"]].tolist()


def test_insufficient_samples_raises():
    base = _base(per_class_tr=200, per_class_te=150)
    with pytest.raises(ValueError, match="train class 0"):
        _build(_cfg(), base)
    with pytest.raises(ValueError, match="test class 0"):
        _build(_cfg(n_eval=600), _base(per_class_te=100))


# ----------------------------------------------------------------------------- complementary labels
@pytest.mark.parametrize("gen", ["bernoulli", "single_uniform"])
def test_ybar_never_contains_true_class(gen):
    pools, _, rec, (_, y_tr, _, _) = _build(_cfg(gen=gen))
    for p in WEAK_POOLS:
        yb = pools[p].ybar.numpy()
        assert yb.dtype == np.uint8 and set(np.unique(yb)) <= {0, 1}
        assert (yb[np.arange(len(yb)), y_tr[rec["indices"][p]]] == 0).all()
        if gen == "single_uniform":
            assert (yb.sum(1) == 1).all()


def test_empirical_c_within_3_se():
    cfg = _cfg(n_te=400)
    pools, _, rec, (_, y_tr, _, _) = _build(cfg)
    for p, c in (("S_tr", cfg.c_tr), ("T_tr", cfg.c_te), ("T_val", cfg.c_te)):
        yb = pools[p].ybar.numpy().astype(bool)
        y = y_tr[rec["indices"][p]]
        offtrue = np.ones_like(yb)
        offtrue[np.arange(len(y)), y] = False
        n_pairs = offtrue.sum()
        c_hat = yb[offtrue].mean()
        se = np.sqrt(c * (1 - c) / n_pairs)
        assert abs(c_hat - c) < 3 * se, (p, c_hat, c, se)


# ----------------------------------------------------------------------------- nesting / determinism
def test_t_tr_nested_and_other_pools_invariant():
    small = _build(_cfg(n_te=100))
    big = _build(_cfg(n_te=400))
    ps, _, rs, _ = small
    pb, _, rb, _ = big
    pos = {int(i): j for j, i in enumerate(rb["indices"]["T_tr"])}
    rows = np.array([pos[int(i)] for i in rs["indices"]["T_tr"]])  # raises KeyError if not nested
    assert torch.equal(ps["T_tr"].ybar, pb["T_tr"].ybar[torch.from_numpy(rows)])
    for p in ("S_tr", "T_val", "T_eval"):
        assert np.array_equal(rs["indices"][p], rb["indices"][p]), p
    for p in ("S_tr", "T_val"):
        assert torch.equal(ps[p].ybar, pb[p].ybar), p


def test_same_seed_same_pools_different_seed_differs():
    a, b, c = (_build(_cfg(seed=s)) for s in (3, 3, 4))
    for p in POOLS:
        assert np.array_equal(a[2]["indices"][p], b[2]["indices"][p]), p
    for p in WEAK_POOLS:
        assert torch.equal(a[0][p].ybar, b[0][p].ybar), p
    assert not np.array_equal(a[2]["indices"]["S_tr"], c[2]["indices"]["S_tr"])
    assert not torch.equal(a[0]["S_tr"].ybar, c[0]["S_tr"].ybar)


def test_config_overrides_and_validation(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("method: target_only\nrho: 5\n")
    cfg = load_config(p, ["n_te=500", "seed=2"])
    assert (cfg.method, cfg.rho, cfg.n_te, cfg.seed) == ("target_only", 5.0, 500, 2)
    with pytest.raises(ValueError):
        load_config(p, ["bogus=1"])
    with pytest.raises(ValueError):
        load_config(p, ["method=diw"])
    with pytest.raises(ValueError):
        load_config(p, ["n_te=3000"])
