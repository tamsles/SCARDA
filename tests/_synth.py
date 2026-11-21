"""Synthetic 3-class Gaussian mixture with class-prior shift, for T5 / T6.

x | y ~ N(mu_y, I_2), mu_k = 2.5 (cos 2πk/3, sin 2πk/3), shared by both domains (p(x|y) invariant).
pi_tr = uniform, pi_te = target_prior(3, rho=5) ≈ (0.609, 0.272, 0.122).
Complementary labels: data.gen_ybar(bernoulli) with c_tr = 0.3, c_te = 0.2.
y is drawn i.i.d. from pi (no stratification) so that plain standard errors apply.
"""
import numpy as np

from scarda.data import gen_ybar, target_prior, uniform_prior

K = 3
D = 2
MU = 2.5 * np.stack([np.cos(2 * np.pi * np.arange(K) / K), np.sin(2 * np.pi * np.arange(K) / K)], axis=1)
PI_TR = uniform_prior(K)
PI_TE = target_prior(K, 5.0)
C_TR, C_TE = 0.3, 0.2


def sample(n: int, pi: np.ndarray, c: float, rng: np.random.Generator):
    """(x[n,2] float64, y[n] int64, ybar[n,K] uint8)."""
    y = rng.choice(K, size=n, p=pi)
    x = MU[y] + rng.standard_normal((n, D))
    ybar = gen_ybar(y, K, "bernoulli", c, rng)
    return x, y, ybar
