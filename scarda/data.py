"""Data protocol (SPEC §4): pools, shifted priors, complementary labels.

`build_from_arrays` is the only code that sees x and y together. It returns
  * ``pools``: {S_tr, T_tr, T_val} -> ``WeakSet`` (x, ybar only) for training,
  * ``labels``: S_tr true labels (oracle weights only) and T_eval (x, y) (final evaluation only),
  * ``info``: indices, class counts and empirical pibar, written to meta.json.
True labels of T_tr / T_val are dropped after generating ybar.

Pool carving: within each class the indices are shuffled once (one generator per purpose,
derived from ``SeedSequence([seed, tag])``) and consecutive blocks are cut in the fixed order
S_tr -> T_tr parent (n_te_max) -> T_val; T_eval is cut from the test set. The order does not
depend on n_te, so S_tr / T_val are identical across n_te, and the T_tr of a smaller n_te is a
per-class prefix of the parent pool (indices and ybar rows). Pool rows are class-major.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
import torch

POOLS = ("S_tr", "T_tr", "T_val", "T_eval")
WEAK_POOLS = ("S_tr", "T_tr", "T_val")
SEED_TAGS = {"split_train": 1, "split_test": 2, "ybar_S_tr": 3, "ybar_T_tr": 4, "ybar_T_val": 5, "loader": 6,
             "target_batch": 7}


def rng_for(seed: int, purpose: str) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([int(seed), SEED_TAGS[purpose]]))


def int_seed_for(seed: int, purpose: str) -> int:
    return int(np.random.SeedSequence([int(seed), SEED_TAGS[purpose]]).generate_state(1)[0])


# ----------------------------------------------------------------------------- priors / counts
def target_prior(K: int, rho: float) -> np.ndarray:
    """pi_k ∝ rho^{-k/(K-1)} (SPEC §4.2); rho=1 gives the uniform prior."""
    p = float(rho) ** (-np.arange(K, dtype=np.float64) / (K - 1))
    return p / p.sum()


def uniform_prior(K: int) -> np.ndarray:
    return np.full(K, 1.0 / K)


def largest_remainder(n: int, p: np.ndarray) -> np.ndarray:
    """Integer counts summing exactly to n, proportional to p (Hamilton method).

    q = n*p is rounded to 1e-9 before flooring so that e.g. 0.1*500 = 49.999999... floors to 50.
    Remaining units go to the largest fractional parts; ties -> smaller class index.
    """
    p = np.asarray(p, dtype=np.float64)
    if n < 0 or (p < 0).any() or not np.isclose(p.sum(), 1.0):
        raise ValueError("largest_remainder needs n >= 0 and a probability vector p")
    q = np.round(n * p, 9)
    base = np.floor(q).astype(np.int64)
    r = int(n - base.sum())
    order = np.argsort(-(q - base), kind="stable")
    base[order[:r]] += 1
    assert int(base.sum()) == n
    return base


def target_counts(cfg) -> dict[str, np.ndarray]:
    """Per-class target counts of every pool; T_tr_parent is the n_te_max pool T_tr is nested in."""
    pi_tr, pi_te = uniform_prior(cfg.K), target_prior(cfg.K, cfg.rho)
    return {
        "S_tr": largest_remainder(cfg.n_tr, pi_tr),
        "T_tr_parent": largest_remainder(cfg.n_te_max, pi_te),
        "T_tr": largest_remainder(cfg.n_te, pi_te),
        "T_val": largest_remainder(cfg.n_val, pi_te),
        "T_eval": largest_remainder(cfg.n_eval, pi_te),
    }


# ----------------------------------------------------------------------------- carving
def carve_indices(y_train: np.ndarray, y_test: np.ndarray, tc: dict[str, np.ndarray], seed: int):
    """Cut the pools from per-class permutations. Returns (idx {pool: int64[n]}, parent_idx, sub_pos),
    where T_tr = parent_idx[sub_pos]. Raises ValueError if a class has too few samples."""
    K = len(tc["S_tr"])
    if (tc["T_tr"] > tc["T_tr_parent"]).any():
        bad = np.flatnonzero(tc["T_tr"] > tc["T_tr_parent"]).tolist()
        raise ValueError(f"T_tr is not nested in the n_te_max parent pool for classes {bad}")

    need_tr = tc["S_tr"] + tc["T_tr_parent"] + tc["T_val"]
    avail_tr = np.bincount(y_train, minlength=K)
    avail_te = np.bincount(y_test, minlength=K)
    msgs = [f"train class {k}: need {need_tr[k]} > available {avail_tr[k]}" for k in range(K) if need_tr[k] > avail_tr[k]]
    msgs += [f"test class {k}: need {tc['T_eval'][k]} > available {avail_te[k]}" for k in range(K) if tc["T_eval"][k] > avail_te[k]]
    if msgs:
        raise ValueError("insufficient samples for the requested pools: " + "; ".join(msgs))

    rng_tr, rng_te = rng_for(seed, "split_train"), rng_for(seed, "split_test")
    blocks = {p: [] for p in ("S_tr", "T_tr_parent", "T_val", "T_eval")}
    for k in range(K):
        perm = rng_tr.permutation(np.flatnonzero(y_train == k))
        a, b, c = tc["S_tr"][k], tc["T_tr_parent"][k], tc["T_val"][k]
        blocks["S_tr"].append(perm[:a])
        blocks["T_tr_parent"].append(perm[a:a + b])
        blocks["T_val"].append(perm[a + b:a + b + c])
        blocks["T_eval"].append(rng_te.permutation(np.flatnonzero(y_test == k))[:tc["T_eval"][k]])

    offsets = np.concatenate([[0], np.cumsum(tc["T_tr_parent"])[:-1]])
    sub_pos = np.concatenate([offsets[k] + np.arange(tc["T_tr"][k]) for k in range(K)]).astype(np.int64)
    parent = np.concatenate(blocks["T_tr_parent"]).astype(np.int64)
    idx = {p: np.concatenate(blocks[p]).astype(np.int64) for p in ("S_tr", "T_val", "T_eval")}
    idx["T_tr"] = parent[sub_pos]
    return idx, parent, sub_pos


# ----------------------------------------------------------------------------- complementary labels
def gen_ybar(y: np.ndarray, K: int, mode: str, c: float, rng: np.random.Generator) -> np.ndarray:
    """SCAR complementary labels as a uint8 [n,K] matrix; never contains the true class.

    bernoulli: each k != y independently with probability c (all-zero rows possible).
    single_uniform: exactly one label, uniform over the K-1 non-true classes (SCARCE 'random').
    """
    y = np.asarray(y, dtype=np.int64)
    n = len(y)
    if mode == "bernoulli":
        yb = rng.random((n, K)) < c
        yb[np.arange(n), y] = False
    elif mode == "single_uniform":
        r = rng.integers(0, K - 1, size=n)
        yb = np.zeros((n, K), dtype=bool)
        yb[np.arange(n), r + (r >= y)] = True
    else:
        raise ValueError(f"unknown gen mode {mode!r}")
    return yb.astype(np.uint8)


# ----------------------------------------------------------------------------- containers
@dataclass(frozen=True)
class WeakSet:
    """Training-side view of a pool: inputs and complementary labels only."""

    name: str
    x: torch.Tensor  # float32 [n,1,28,28] (or [n,d])
    ybar: torch.Tensor  # uint8 [n,K]

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def pibar_hat(self) -> torch.Tensor:
        return self.ybar.double().mean(dim=0)


class Labels(NamedTuple):
    """True labels: S_tr labels for oracle.py, T_eval (x, y) for evaluate.py. Never passed to training."""

    source_y: torch.Tensor
    eval_x: torch.Tensor
    eval_y: torch.Tensor


# ----------------------------------------------------------------------------- builders
def _to_float_images(x) -> torch.Tensor:
    x = torch.as_tensor(x)
    if x.dtype == torch.uint8:
        x = x.float().div(255.0)  # identical to transforms.ToTensor() for uint8 images
    x = x.float()
    if x.dim() == 3:
        x = x.unsqueeze(1)
    return x.contiguous()


def build_from_arrays(x_tr, y_tr, x_te, y_te, cfg) -> tuple[dict[str, WeakSet], Labels, dict]:
    """Build the four pools from base arrays. `cfg` needs K, rho, n_tr, n_te, n_te_max, n_val,
    n_eval, gen, seed and `effective_c`."""
    K = cfg.K
    y_tr = np.asarray(y_tr, dtype=np.int64)
    y_te = np.asarray(y_te, dtype=np.int64)
    tc = target_counts(cfg)
    idx, parent_idx, sub_pos = carve_indices(y_tr, y_te, tc, cfg.seed)
    c_tr, c_te = cfg.effective_c

    ybar = {
        "S_tr": gen_ybar(y_tr[idx["S_tr"]], K, cfg.gen, c_tr, rng_for(cfg.seed, "ybar_S_tr")),
        # generated on the parent pool, then the n_te prefix rows are taken (nested)
        "T_tr": gen_ybar(y_tr[parent_idx], K, cfg.gen, c_te, rng_for(cfg.seed, "ybar_T_tr"))[sub_pos],
        "T_val": gen_ybar(y_tr[idx["T_val"]], K, cfg.gen, c_te, rng_for(cfg.seed, "ybar_T_val")),
    }

    x_tr_t, x_te_t = torch.as_tensor(x_tr), torch.as_tensor(x_te)
    pools = {p: WeakSet(p, _to_float_images(x_tr_t[torch.from_numpy(idx[p])]), torch.from_numpy(ybar[p]))
             for p in WEAK_POOLS}
    labels = Labels(
        source_y=torch.from_numpy(y_tr[idx["S_tr"]].copy()),
        eval_x=_to_float_images(x_te_t[torch.from_numpy(idx["T_eval"])]),
        eval_y=torch.from_numpy(y_te[idx["T_eval"]].copy()),
    )

    actual = {p: np.bincount(y_tr[idx[p]], minlength=K).tolist() for p in WEAK_POOLS}
    actual["T_eval"] = np.bincount(y_te[idx["T_eval"]], minlength=K).tolist()
    info = {
        "indices": idx,
        "target_counts": {p: tc[p].tolist() for p in POOLS},
        "actual_counts": actual,
        "pibar_hat": {p: ybar[p].mean(axis=0).tolist() for p in WEAK_POOLS},
        "pi_tr": uniform_prior(K).tolist(), "pi_te": target_prior(K, cfg.rho).tolist(),
        "c_tr": c_tr, "c_te": c_te,
    }
    return pools, labels, info


def build_mnist(cfg) -> tuple[dict[str, WeakSet], Labels, dict]:
    import torchvision.datasets as dsets

    tr = dsets.MNIST(root=cfg.data_root, train=True, download=True)
    te = dsets.MNIST(root=cfg.data_root, train=False, download=True)
    return build_from_arrays(tr.data, tr.targets.numpy(), te.data, te.targets.numpy(), cfg)


def format_pool_report(info: dict) -> str:
    K = len(info["target_counts"]["S_tr"])
    lines = ["class counts as target/actual",
             "pool    | " + " ".join(f"{k:>9d}" for k in range(K)) + " |     total"]
    for p in POOLS:
        t, a = info["target_counts"][p], info["actual_counts"][p]
        cells = " ".join(f"{ti:>4d}/{ai:<4d}" for ti, ai in zip(t, a))
        lines.append(f"{p:<7} | {cells} | {sum(t):>4d}/{sum(a):<4d}")
    lines.append("pibar: empirical (column mean of ybar) vs population c_d*(1-pi_d)")
    for p in WEAK_POOLS:
        lines.append(f"{p:<7} | " + " ".join(f"{v:>9.4f}" for v in info["pibar_hat"][p]))
    for name, c, pi in (("pop_S", info["c_tr"], info["pi_tr"]), ("pop_T", info["c_te"], info["pi_te"])):
        lines.append(f"{name:<7} | " + " ".join(f"{v:>9.4f}" for v in c * (1 - np.asarray(pi))))
    return "\n".join(lines)
