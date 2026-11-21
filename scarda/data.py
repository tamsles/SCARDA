"""Data protocol (SPEC §4): pools, shifted priors, complementary labels, isolation, split hashes.

Isolation (SPEC §4.4): the builder below is the only code that sees true labels together with x.
It returns
  * ``WeakSet`` objects (x, ybar only; no labels, no indices) -> training side,
  * a ``LabelVault`` holding S_tr labels (for oracle.py) and T_eval (x, y) (for evaluate.py),
  * a ``SplitRecord`` (indices, class counts, hashes) -> written to meta.json only.
True labels of T_tr / T_val are discarded after generating ybar.

Pool carving: within each class the indices are shuffled once (one generator per purpose,
derived from ``SeedSequence([seed, tag])``) and consecutive blocks are cut in the fixed order
S_tr -> T_tr parent (n_te_max) -> T_val; T_eval is cut from the test set. The order does not
depend on n_te, so S_tr / T_val are identical across n_te, and the T_tr of a smaller n_te is a
per-class prefix of the parent pool (indices and ybar rows). Pool rows are class-major.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch

POOLS = ("S_tr", "T_tr", "T_val", "T_eval")
WEAK_POOLS = ("S_tr", "T_tr", "T_val")
SEED_TAGS = {
    "split_train": 1,
    "split_test": 2,
    "ybar_S_tr": 3,
    "ybar_T_tr": 4,
    "ybar_T_val": 5,
    "loader": 6,
}


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
    if r < 0 or r > len(p):
        raise AssertionError(f"remainder {r} out of range")
    order = np.argsort(-(q - base), kind="stable")
    base[order[:r]] += 1
    assert int(base.sum()) == n
    return base


# ----------------------------------------------------------------------------- hashing
def sha256_array(arr: np.ndarray) -> str:
    a = np.ascontiguousarray(arr)
    if a.dtype.byteorder == ">":
        a = a.astype(a.dtype.newbyteorder("<"))
    h = hashlib.sha256(f"{a.dtype.str}|{a.shape}|".encode())
    h.update(a.tobytes())
    return h.hexdigest()


def combine_hashes(hashes: dict[str, str]) -> str:
    lines = "\n".join(f"{k}:{hashes[k]}" for k in sorted(hashes))
    return hashlib.sha256(lines.encode()).hexdigest()


# ----------------------------------------------------------------------------- carving
@dataclass(frozen=True)
class PoolSpec:
    K: int
    pi_tr: np.ndarray
    pi_te: np.ndarray
    n_tr: int
    n_te: int
    n_te_max: int
    n_val: int
    n_eval: int

    def target_counts(self) -> dict[str, np.ndarray]:
        return {
            "S_tr": largest_remainder(self.n_tr, self.pi_tr),
            "T_tr_parent": largest_remainder(self.n_te_max, self.pi_te),
            "T_tr": largest_remainder(self.n_te, self.pi_te),
            "T_val": largest_remainder(self.n_val, self.pi_te),
            "T_eval": largest_remainder(self.n_eval, self.pi_te),
        }


def carve_indices(y_train: np.ndarray, y_test: np.ndarray, spec: PoolSpec, seed: int) -> dict:
    """Return {"idx": {pool: int64[n]}, "parent_idx": int64[n_te_max], "sub_pos": int64[n_te],
    "target": {pool: counts}}.  Raises ValueError if a class has too few samples."""
    K = spec.K
    y_train = np.asarray(y_train, dtype=np.int64)
    y_test = np.asarray(y_test, dtype=np.int64)
    tc = spec.target_counts()
    if (tc["T_tr"] > tc["T_tr_parent"]).any():
        bad = np.flatnonzero(tc["T_tr"] > tc["T_tr_parent"]).tolist()
        raise ValueError(f"T_tr(n_te={spec.n_te}) is not nested in T_tr(n_te_max={spec.n_te_max}) for classes {bad}")

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
        perm_te = rng_te.permutation(np.flatnonzero(y_test == k))
        blocks["T_eval"].append(perm_te[:tc["T_eval"][k]])

    offsets = np.concatenate([[0], np.cumsum(tc["T_tr_parent"])[:-1]])
    sub_pos = np.concatenate([offsets[k] + np.arange(tc["T_tr"][k]) for k in range(K)]).astype(np.int64)
    parent = np.concatenate(blocks["T_tr_parent"]).astype(np.int64)
    idx = {p: np.concatenate(blocks[p]).astype(np.int64) for p in ("S_tr", "T_val", "T_eval")}
    idx["T_tr"] = parent[sub_pos]
    return {"idx": idx, "parent_idx": parent, "sub_pos": sub_pos, "target": tc}


# ----------------------------------------------------------------------------- complementary labels
def gen_ybar(y: np.ndarray, K: int, mode: str, c: float, rng: np.random.Generator) -> np.ndarray:
    """SCAR complementary labels as a uint8 [n,K] matrix; never contains the true class.

    bernoulli: each k != y independently with probability c (all-zero rows possible).
    single_uniform: exactly one label, uniform over the K-1 non-true classes (SCARCE 'random').
    """
    y = np.asarray(y, dtype=np.int64)
    n = len(y)
    if mode == "bernoulli":
        u = rng.random((n, K))
        yb = u < c
        yb[np.arange(n), y] = False
    elif mode == "single_uniform":
        r = rng.integers(0, K - 1, size=n)
        comp = r + (r >= y)
        yb = np.zeros((n, K), dtype=bool)
        yb[np.arange(n), comp] = True
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


class LabelVault:
    """Privileged labels. Only evaluate.py (T_eval) and oracle.py (S_tr labels) may read it."""

    __slots__ = ("_source_y", "_eval_x", "_eval_y")

    def __init__(self, source_y: torch.Tensor, eval_x: torch.Tensor, eval_y: torch.Tensor):
        self._source_y = source_y
        self._eval_x = eval_x
        self._eval_y = eval_y

    def source_labels(self) -> torch.Tensor:
        return self._source_y

    def eval_set(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self._eval_x, self._eval_y

    def __repr__(self) -> str:
        return f"LabelVault(n_source={len(self._source_y)}, n_eval={len(self._eval_y)})"


@dataclass(frozen=True)
class SplitRecord:
    indices: dict  # pool -> int64 indices into the base train (S_tr,T_tr,T_val) / test (T_eval) set
    target_counts: dict  # pool -> per-class target counts (list[int])
    actual_counts: dict  # pool -> per-class realized counts (list[int])
    pibar_hat: dict  # weak pool -> empirical complementary-label rate per class (list[float])
    hashes: dict  # "pool/idx", "pool/ybar" -> sha256
    split_hash: str
    base_hash: Optional[str] = None
    extra: dict = field(default_factory=dict)

    def summary(self) -> dict:
        return {
            "target_counts": self.target_counts,
            "actual_counts": self.actual_counts,
            "pibar_hat": self.pibar_hat,
            "hashes": self.hashes,
            "split_hash": self.split_hash,
            "base_hash": self.base_hash,
            **self.extra,
        }


# ----------------------------------------------------------------------------- builders
def _to_float_images(x) -> torch.Tensor:
    x = torch.as_tensor(x)
    if x.dtype == torch.uint8:
        x = x.float().div(255.0)  # identical to transforms.ToTensor() for uint8 images
    x = x.float()
    if x.dim() == 3:
        x = x.unsqueeze(1)
    return x.contiguous()


def build_from_arrays(x_tr, y_tr, x_te, y_te, cfg) -> tuple[dict[str, WeakSet], LabelVault, SplitRecord]:
    """Build the four pools from base arrays. `cfg` needs K, rho, n_tr, n_te, n_te_max, n_val,
    n_eval, gen, seed and `effective_c`."""
    K = cfg.K
    y_tr = np.asarray(y_tr, dtype=np.int64)
    y_te = np.asarray(y_te, dtype=np.int64)
    spec = PoolSpec(K, uniform_prior(K), target_prior(K, cfg.rho), cfg.n_tr, cfg.n_te, cfg.n_te_max,
                    cfg.n_val, cfg.n_eval)
    carved = carve_indices(y_tr, y_te, spec, cfg.seed)
    idx = carved["idx"]
    c_tr, c_te = cfg.effective_c

    ybar = {
        "S_tr": gen_ybar(y_tr[idx["S_tr"]], K, cfg.gen, c_tr, rng_for(cfg.seed, "ybar_S_tr")),
        # generated on the parent pool, then the n_te prefix rows are taken (nested)
        "T_tr": gen_ybar(y_tr[carved["parent_idx"]], K, cfg.gen, c_te, rng_for(cfg.seed, "ybar_T_tr"))[carved["sub_pos"]],
        "T_val": gen_ybar(y_tr[idx["T_val"]], K, cfg.gen, c_te, rng_for(cfg.seed, "ybar_T_val")),
    }

    x_tr_t = torch.as_tensor(x_tr)
    x_te_t = torch.as_tensor(x_te)
    pools = {
        p: WeakSet(p, _to_float_images(x_tr_t[torch.from_numpy(idx[p])]), torch.from_numpy(ybar[p]))
        for p in WEAK_POOLS
    }
    vault = LabelVault(
        source_y=torch.from_numpy(y_tr[idx["S_tr"]].copy()),
        eval_x=_to_float_images(x_te_t[torch.from_numpy(idx["T_eval"])]),
        eval_y=torch.from_numpy(y_te[idx["T_eval"]].copy()),
    )

    actual = {p: np.bincount(y_tr[idx[p]], minlength=K).tolist() for p in WEAK_POOLS}
    actual["T_eval"] = np.bincount(y_te[idx["T_eval"]], minlength=K).tolist()
    hashes = {f"{p}/idx": sha256_array(idx[p]) for p in POOLS}
    hashes.update({f"{p}/ybar": sha256_array(ybar[p]) for p in WEAK_POOLS})
    record = SplitRecord(
        indices=idx,
        target_counts={p: carved["target"][p].tolist() for p in POOLS},
        actual_counts=actual,
        pibar_hat={p: ybar[p].mean(axis=0).tolist() for p in WEAK_POOLS},
        hashes=hashes,
        split_hash=combine_hashes(hashes),
        base_hash=combine_hashes({"train/y": sha256_array(y_tr), "test/y": sha256_array(y_te)}),
        extra={"pi_tr": spec.pi_tr.tolist(), "pi_te": spec.pi_te.tolist(), "c_tr": c_tr, "c_te": c_te},
    )
    return pools, vault, record


def build_mnist(cfg) -> tuple[dict[str, WeakSet], LabelVault, SplitRecord]:
    import torchvision.datasets as dsets

    tr = dsets.MNIST(root=cfg.data_root, train=True, download=True)
    te = dsets.MNIST(root=cfg.data_root, train=False, download=True)
    return build_from_arrays(tr.data, tr.targets.numpy(), te.data, te.targets.numpy(), cfg)


def format_pool_report(record: SplitRecord) -> str:
    K = len(record.target_counts["S_tr"])
    lines = []
    head = "pool    | " + " ".join(f"{k:>9d}" for k in range(K)) + " |     total"
    lines.append("class counts as target/actual")
    lines.append(head)
    for p in POOLS:
        t, a = record.target_counts[p], record.actual_counts[p]
        cells = " ".join(f"{ti:>4d}/{ai:<4d}" for ti, ai in zip(t, a))
        lines.append(f"{p:<7} | {cells} | {sum(t):>4d}/{sum(a):<4d}")
    lines.append("pibar: empirical (column mean of ybar) vs population c_d*(1-pi_d)")
    ex = record.extra
    pop = {"S_tr": ex["c_tr"] * (1 - np.asarray(ex["pi_tr"])), "T": ex["c_te"] * (1 - np.asarray(ex["pi_te"]))}
    for p in WEAK_POOLS:
        lines.append(f"{p:<7} | " + " ".join(f"{v:>9.4f}" for v in record.pibar_hat[p]))
    lines.append(f"{'pop_S':<7} | " + " ".join(f"{v:>9.4f}" for v in pop["S_tr"]))
    lines.append(f"{'pop_T':<7} | " + " ".join(f"{v:>9.4f}" for v in pop["T"]))
    return "\n".join(lines)
