import math

import pytest
import torch

from scarda.risk import scar_ovr_risk
from utils_algo import SCARCE_loss

CPU = torch.device("cpu")


def _onehot(idx: torch.Tensor, K: int) -> torch.Tensor:
    out = torch.zeros(idx.shape[0], K)
    out[torch.arange(idx.shape[0]), idx.long()] = 1.0
    return out


def _uniform_pi(K, dtype):
    return torch.full((K,), 1.0 / K, dtype=dtype)


def _reference_risk(logits, ybar, pi, pibar, w_pos, w_neg, correction):
    """Naive per-element loop implementation of SPEC §3.1 (float64, python scalars)."""
    n, K = logits.shape
    ell = lambda z: math.log1p(math.exp(-z)) if z > -30 else -z  # noqa: E731
    total, P_list = 0.0, []
    for k in range(K):
        sp = sn = 0.0
        a_mf = a_f = b_f = 0.0
        for i in range(n):
            f = float(logits[i, k])
            if ybar[i, k] == 1:
                w = 1.0 if w_pos is None else float(w_pos[i, k])
                sp += w
                a_mf += w * ell(-f)
                a_f += w * ell(f)
            else:
                w = 1.0 if w_neg is None else float(w_neg[i, k])
                sn += w
                b_f += w * ell(f)
        Mp_mf = a_mf / sp if sp > 0 else 0.0
        Mp_f = a_f / sp if sp > 0 else 0.0
        Mn_f = b_f / sn if sn > 0 else 0.0
        N = (1 - float(pi[k])) * Mp_mf
        P = (float(pibar[k]) + float(pi[k]) - 1) * Mp_f + (1 - float(pibar[k])) * Mn_f
        P_list.append(P)
        corr = {"abs": abs(P), "relu": max(P, 0.0), "none": P}[correction]
        total += N + corr
    return total, P_list


# ---------------------------------------------------------------- T1: reproduce SCARCE_loss

T1_CASES = [(0, 256, 10), (1, 256, 10), (2, 64, 10), (3, 17, 5), (4, 1000, 10), (5, 3, 4)]


@pytest.mark.parametrize("seed,n,K", T1_CASES)
def test_t1_reproduces_scarce_float64(seed, n, K):
    g = torch.Generator().manual_seed(seed)
    logits = 3.0 * torch.randn(n, K, generator=g, dtype=torch.float64)
    labels = torch.randint(0, K, (n,), generator=g)
    ref = SCARCE_loss(logits, labels, CPU)
    out = scar_ovr_risk(logits, _onehot(labels, K), _uniform_pi(K, torch.float64),
                        correction="abs", pibar_mode="batch")
    assert abs(out.total.item() - ref.item()) < 1e-6


@pytest.mark.parametrize("seed,n,K", T1_CASES)
def test_t1_reproduces_scarce_float32(seed, n, K):
    g = torch.Generator().manual_seed(100 + seed)
    logits = 3.0 * torch.randn(n, K, generator=g, dtype=torch.float32)
    labels = torch.randint(0, K, (n,), generator=g)
    ref = SCARCE_loss(logits, labels, CPU)
    out = scar_ovr_risk(logits, _onehot(labels, K), _uniform_pi(K, torch.float32),
                        correction="abs", pibar_mode="batch")
    assert abs(out.total.item() - ref.item()) / abs(ref.item()) < 1e-5


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
@pytest.mark.parametrize("missing", [[3], [0, 7, 9]])
def test_t1_with_empty_class(dtype, missing):
    K, n = 10, 128
    g = torch.Generator().manual_seed(7)
    present = torch.tensor([k for k in range(K) if k not in missing])
    labels = present[torch.randint(0, len(present), (n,), generator=g)]
    logits = 3.0 * torch.randn(n, K, generator=g, dtype=dtype)
    ref = SCARCE_loss(logits, labels, CPU)
    out = scar_ovr_risk(logits, _onehot(labels, K), _uniform_pi(K, dtype),
                        correction="abs", pibar_mode="batch")
    if dtype == torch.float64:
        assert abs(out.total.item() - ref.item()) < 1e-6
    else:
        assert abs(out.total.item() - ref.item()) / abs(ref.item()) < 1e-5
    assert out.empty_pos == len(missing)
    assert out.empty_neg == 0


def test_t1_gradients_match_scarce():
    K, n = 10, 64
    g = torch.Generator().manual_seed(11)
    base = 3.0 * torch.randn(n, K, generator=g, dtype=torch.float64)
    labels = torch.randint(0, K, (n,), generator=g)
    a = base.clone().requires_grad_(True)
    b = base.clone().requires_grad_(True)
    SCARCE_loss(a, labels, CPU).backward()
    # SCARCE builds its prior in float32 (utils_algo.py L34-36): ccp = float32(1 - float32(1/K)),
    # which differs from 0.9 by ~2.4e-8. Feed the bit-identical prior so gradients are comparable.
    pi_scarce = 1.0 - (1.0 - 1.0 / K * torch.ones(K)).double()
    scar_ovr_risk(b, _onehot(labels, K), pi_scarce).total.backward()
    assert torch.allclose(a.grad, b.grad, atol=1e-10, rtol=0)


# ---------------------------------------------------------------- T2: unit weights == None


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("correction", ["abs", "relu", "none"])
def test_t2_unit_weights_equal_none(seed, correction):
    K, n = 10, 200
    g = torch.Generator().manual_seed(seed)
    logits = 2.0 * torch.randn(n, K, generator=g, dtype=torch.float64)
    ybar = (torch.rand(n, K, generator=g) < 0.25).double()  # multi-hot, includes all-zero rows
    ybar[:5] = 0.0
    pi = torch.softmax(torch.randn(K, generator=g, dtype=torch.float64), 0)
    pibar = torch.rand(K, generator=g, dtype=torch.float64) * 0.5
    ones = torch.ones(n, K, dtype=torch.float64)
    a = scar_ovr_risk(logits, ybar, pi, pibar, None, None, correction, "fixed")
    b = scar_ovr_risk(logits, ybar, pi, pibar, ones, ones, correction, "fixed")
    c = scar_ovr_risk(logits, ybar, pi, pibar, ones, None, correction, "fixed")
    for o in (b, c):
        assert torch.equal(a.total, o.total)
        assert torch.equal(a.N, o.N) and torch.equal(a.P_raw, o.P_raw)
        assert (a.neg_rate, a.empty_pos, a.empty_neg) == (o.neg_rate, o.empty_pos, o.empty_neg)


def test_t2_fixed_with_batch_pibar_equals_batch_mode():
    K, n = 10, 150
    g = torch.Generator().manual_seed(3)
    logits = torch.randn(n, K, generator=g, dtype=torch.float64)
    ybar = (torch.rand(n, K, generator=g) < 0.3).double()
    pi = _uniform_pi(K, torch.float64)
    a = scar_ovr_risk(logits, ybar, pi, pibar_mode="batch")
    b = scar_ovr_risk(logits, ybar, pi, ybar.mean(0), torch.ones(n, K, dtype=torch.float64),
                      torch.ones(n, K, dtype=torch.float64), "abs", "fixed")
    assert torch.allclose(a.total, b.total, atol=1e-12, rtol=0)


# ---------------------------------------------------------------- T3: correction branches


def _negative_P_input():
    """Classes 0..5 get P_k < 0, classes 6..9 get P_k > 0 (single-hot, batch pibar, uniform pi).

    With pibar_k ~ 0.1 and pi_k = 0.1, the coefficient (pibar+pi-1) < 0. Making ell(f_k) large on
    the (k,+) group and small on (k,-) gives P_k < 0; the reverse gives P_k > 0.
    """
    K, per = 10, 20
    labels = torch.arange(K).repeat_interleave(per)  # every class has exactly `per` comp. labels
    ybar = _onehot(labels, K).double()
    logits = torch.empty(K * per, K, dtype=torch.float64)
    for k in range(K):
        pos = ybar[:, k] == 1
        if k < 6:
            logits[pos, k], logits[~pos, k] = -8.0, 8.0
        else:
            logits[pos, k], logits[~pos, k] = 8.0, -8.0
    return logits, ybar, _uniform_pi(K, torch.float64), 6 / K


@pytest.mark.parametrize("correction", ["abs", "relu", "none"])
def test_t3_correction_branches(correction):
    logits, ybar, pi, expected_neg_rate = _negative_P_input()
    out = scar_ovr_risk(logits, ybar, pi, correction=correction, pibar_mode="batch")
    assert (out.P_raw[:6] < 0).all() and (out.P_raw[6:] > 0).all()
    assert out.neg_rate == pytest.approx(expected_neg_rate)
    corr = {"abs": out.P_raw.abs(), "relu": out.P_raw.clamp_min(0), "none": out.P_raw}[correction]
    assert out.total.item() == pytest.approx((out.N + corr).sum().item(), abs=1e-12)
    ref_total, ref_P = _reference_risk(logits, ybar, pi, ybar.mean(0), None, None, correction)
    assert out.total.item() == pytest.approx(ref_total, abs=1e-9)
    assert out.P_raw.tolist() == pytest.approx(ref_P, abs=1e-9)


def test_t3_branch_ordering():
    logits, ybar, pi, _ = _negative_P_input()
    t = {c: scar_ovr_risk(logits, ybar, pi, correction=c).total.item() for c in ("abs", "relu", "none")}
    assert t["none"] < t["relu"] < t["abs"]


# ---------------------------------------------------------------- weighted reference / empty groups


def test_weighted_matches_reference():
    K, n = 6, 40
    g = torch.Generator().manual_seed(5)
    logits = 2.0 * torch.randn(n, K, generator=g, dtype=torch.float64)
    ybar = (torch.rand(n, K, generator=g) < 0.3).double()
    w_pos = torch.rand(n, K, generator=g, dtype=torch.float64) * 3
    w_neg = torch.rand(n, K, generator=g, dtype=torch.float64) * 3
    pi = torch.softmax(torch.randn(K, generator=g, dtype=torch.float64), 0)
    pibar = torch.rand(K, generator=g, dtype=torch.float64) * 0.4
    for corr in ("abs", "relu", "none"):
        out = scar_ovr_risk(logits, ybar, pi, pibar, w_pos, w_neg, corr, "fixed")
        ref_total, ref_P = _reference_risk(logits, ybar, pi, pibar, w_pos, w_neg, corr)
        assert out.total.item() == pytest.approx(ref_total, abs=1e-9)
        assert out.P_raw.tolist() == pytest.approx(ref_P, abs=1e-9)


def test_zero_weight_sum_group_counts_as_empty():
    K, n = 4, 12
    g = torch.Generator().manual_seed(9)
    logits = torch.randn(n, K, generator=g, dtype=torch.float64)
    ybar = torch.zeros(n, K, dtype=torch.float64)
    ybar[:4, 0] = 1  # (0,+) non-empty but zero weight
    ybar[:, 1] = 1  # (1,-) empty
    ybar[4:8, 2] = 1
    # class 3: (3,+) empty
    w_pos = torch.ones(n, K, dtype=torch.float64)
    w_pos[:, 0] = 0.0
    pi = _uniform_pi(K, torch.float64)
    pibar = torch.full((K,), 0.2, dtype=torch.float64)
    out = scar_ovr_risk(logits, ybar, pi, pibar, w_pos, None, "none", "fixed")
    assert out.empty_pos == 2  # (0,+) zero-sum, (3,+) empty
    assert out.empty_neg == 1  # (1,-)
    assert out.N[0].item() == 0.0 and out.N[3].item() == 0.0
    assert torch.isfinite(out.total)


def test_argument_validation():
    K, n = 3, 5
    logits = torch.zeros(n, K)
    ybar = torch.zeros(n, K)
    pi = torch.full((K,), 1 / K)
    with pytest.raises(ValueError):
        scar_ovr_risk(logits, ybar, pi, pibar=torch.zeros(K), pibar_mode="batch")
    with pytest.raises(ValueError):
        scar_ovr_risk(logits, ybar, pi, w_pos=torch.ones(n, K), pibar_mode="batch")
    with pytest.raises(ValueError):
        scar_ovr_risk(logits, ybar, pi, pibar_mode="fixed")
    with pytest.raises(ValueError):
        scar_ovr_risk(logits, ybar + 2, pi)
    with pytest.raises(ValueError):
        scar_ovr_risk(logits, ybar, pi, correction="clip")
    with pytest.raises(ValueError):
        scar_ovr_risk(logits, ybar, pi, torch.zeros(K), -torch.ones(n, K), None, "abs", "fixed")
