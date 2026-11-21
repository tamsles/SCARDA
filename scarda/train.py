"""Training loop (SPEC §3.3, §7). Supports `source_only`, `target_only` and `oracle_iw`.

This module only receives WeakSets (x, ybar), an opaque evaluator callback and, for `oracle_iw`,
precomputed per-sample weights for S_tr (built by oracle.py); it never sees true labels.
The evaluator's output is logged and never used for any decision.
"""
from __future__ import annotations

import math
import time
from typing import Callable, Iterator, Optional

import numpy as np
import torch

from scarda.data import WeakSet, int_seed_for, target_prior, uniform_prior
from scarda.models import MLP
from scarda.risk import scar_ovr_risk

METRIC_COLUMNS = [
    "epoch", "step", "train_loss_mean", "val_weak_risk", "target_acc", "target_bal_acc",
    "neg_rate", "empty_pos", "empty_neg", "ess_pos_mean", "ess_neg_mean", "w_max", "fallback",
    "kmm_iters_mean", "epoch_time",
]


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def cycle_batches(n: int, bs: int, gen: torch.Generator) -> Iterator[torch.Tensor]:
    """Endless stream of index batches: reshuffle at each pass, keep the last partial batch
    (same as a shuffled DataLoader with drop_last=False, looped)."""
    while True:
        perm = torch.randperm(n, generator=gen)
        for i in range(0, n, bs):
            yield perm[i:i + bs]


@torch.no_grad()
def weak_val_risk(model: torch.nn.Module, val: WeakSet, pi_te: torch.Tensor, device, bs: int = 4096) -> float:
    """Weak-label validation risk on T_val: correction=none, true pi_te, pibar = T_val frequency."""
    was_training = model.training
    model.eval()
    x = val.x.to(device)
    logits = torch.cat([model(x[i:i + bs]) for i in range(0, len(x), bs)])
    out = scar_ovr_risk(logits, val.ybar.to(device), pi_te.to(device, logits.dtype),
                        pibar=val.pibar_hat().to(device, logits.dtype),
                        correction="none", pibar_mode="fixed")
    model.train(was_training)
    return float(out.total.item())


Weights = tuple[torch.Tensor, torch.Tensor]


def method_setup(cfg, pools: dict[str, WeakSet], source_weights: Optional[Weights] = None
                 ) -> tuple[WeakSet, np.ndarray, Optional[torch.Tensor]]:
    """(training pool, prior used in the risk, fixed pibar or None for batch mode) per method (SPEC §3.3)."""
    if (cfg.method == "oracle_iw") != (source_weights is not None):
        raise ValueError("source_weights must be given for oracle_iw and only for oracle_iw")
    if cfg.method == "source_only":
        return pools["S_tr"], uniform_prior(cfg.K), None
    if cfg.method == "target_only":
        return pools["T_tr"], target_prior(cfg.K, cfg.rho), None  # oracle prior (SPEC §3.3)
    if cfg.method == "oracle_iw":
        # weighted -> pibar_mode=fixed with the observable T_tr frequency (SPEC §3.2)
        return pools["S_tr"], target_prior(cfg.K, cfg.rho), pools["T_tr"].pibar_hat()
    raise ValueError(f"method {cfg.method!r} not implemented")


def weight_diagnostics(w_pos: torch.Tensor, w_neg: torch.Tensor, ybar: torch.Tensor) -> tuple[float, float, float]:
    """(ess_pos_mean, ess_neg_mean, w_max) of one batch. ESS ratio (sum w)^2 / (m sum w^2) per
    non-empty (k,s) group (SPEC §8), averaged over groups; w_max over in-group entries."""
    def ess(w, mask):
        m = mask.sum(dim=0)
        s1, s2 = (w * mask).sum(dim=0), (w * w * mask).sum(dim=0)
        ok = (m > 0) & (s2 > 0)
        return float((s1[ok] ** 2 / (m[ok] * s2[ok])).mean().item()) if bool(ok.any()) else float("nan")

    Yp = ybar.to(w_pos.dtype)
    Yn = 1.0 - Yp
    w_max = float(torch.maximum((w_pos * Yp).max(), (w_neg * Yn).max()).item())
    return ess(w_pos, Yp), ess(w_neg, Yn), w_max


def train(
    cfg,
    pools: dict[str, WeakSet],
    evaluator: Callable[[torch.nn.Module], dict],
    on_epoch: Optional[Callable[[dict], None]] = None,
    epochs: Optional[int] = None,
    source_weights: Optional[Weights] = None,
) -> list[dict]:
    return fit(cfg, pools, evaluator, on_epoch=on_epoch, epochs=epochs, source_weights=source_weights)[1]


def fit(
    cfg,
    pools: dict[str, WeakSet],
    evaluator: Callable[[torch.nn.Module], dict],
    on_epoch: Optional[Callable[[dict], None]] = None,
    epochs: Optional[int] = None,
    source_weights: Optional[Weights] = None,
) -> tuple[torch.nn.Module, list[dict]]:
    """Train and return (model, per-epoch history). `source_weights` = (w_pos, w_neg) over the
    full S_tr pool, required for oracle_iw only."""
    device = resolve_device(cfg.device)
    epochs = cfg.epochs if epochs is None else epochs
    train_set, pi, pibar_fixed = method_setup(cfg, pools, source_weights)
    pi_t = torch.as_tensor(pi, dtype=torch.float32, device=device)
    weighted = source_weights is not None
    if weighted:
        w_pos_all = source_weights[0].to(device=device, dtype=torch.float32)
        w_neg_all = source_weights[1].to(device=device, dtype=torch.float32)
        if w_pos_all.shape != train_set.ybar.shape or w_neg_all.shape != train_set.ybar.shape:
            raise ValueError("source_weights must match the S_tr ybar shape")
        pibar_t = pibar_fixed.to(device=device, dtype=torch.float32)
    pi_te = torch.as_tensor(target_prior(cfg.K, cfg.rho), dtype=torch.float32)

    torch.manual_seed(cfg.seed)  # init_seed = seed (SPEC §4.5)
    model = MLP(input_dim=int(np.prod(train_set.x.shape[1:])), hidden_dim=cfg.hidden, output_dim=cfg.K).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.wd)

    x_all = train_set.x.to(device)
    yb_all = train_set.ybar.to(device=device, dtype=torch.float32)
    gen = torch.Generator().manual_seed(int_seed_for(cfg.seed, "loader"))
    batches = cycle_batches(len(train_set), cfg.bs, gen)
    steps_per_epoch = math.ceil(cfg.n_tr / cfg.bs)  # same budget for all methods (SPEC §7)

    history, step = [], 0
    for epoch in range(1, epochs + 1):
        t0 = time.perf_counter()
        model.train()
        loss_sum = neg_sum = 0.0
        ep_sum = en_sum = 0
        ess_p_sum = ess_n_sum = 0.0
        w_max = float("-inf")
        for _ in range(steps_per_epoch):
            idx = next(batches).to(device)
            if weighted:
                wp, wn = w_pos_all[idx], w_neg_all[idx]
                out = scar_ovr_risk(model(x_all[idx]), yb_all[idx], pi_t, pibar=pibar_t, w_pos=wp, w_neg=wn,
                                    correction=cfg.correction, pibar_mode="fixed")
                e_p, e_n, w_m = weight_diagnostics(wp, wn, yb_all[idx])
                ess_p_sum += e_p
                ess_n_sum += e_n
                w_max = max(w_max, w_m)
            else:
                out = scar_ovr_risk(model(x_all[idx]), yb_all[idx], pi_t,
                                    correction=cfg.correction, pibar_mode="batch")
            opt.zero_grad(set_to_none=True)
            out.total.backward()
            opt.step()
            step += 1
            loss_sum += out.total.item()
            neg_sum += out.neg_rate
            ep_sum += out.empty_pos
            en_sum += out.empty_neg
        val = weak_val_risk(model, pools["T_val"], pi_te, device)
        ev = evaluator(model)
        if device.type == "cuda":
            torch.cuda.synchronize()
        row = {
            "epoch": epoch, "step": step,
            "train_loss_mean": loss_sum / steps_per_epoch,
            "val_weak_risk": val,
            "target_acc": ev["target_acc"], "target_bal_acc": ev["target_bal_acc"],
            "neg_rate": neg_sum / steps_per_epoch,
            "empty_pos": ep_sum / steps_per_epoch, "empty_neg": en_sum / steps_per_epoch,
            # weight diagnostics: nan for unweighted methods; no KMM for oracle_iw
            "ess_pos_mean": ess_p_sum / steps_per_epoch if weighted else float("nan"),
            "ess_neg_mean": ess_n_sum / steps_per_epoch if weighted else float("nan"),
            "w_max": w_max if weighted else float("nan"),
            "fallback": float("nan"), "kmm_iters_mean": float("nan"),
            "epoch_time": time.perf_counter() - t0,
        }
        history.append(row)
        if on_epoch is not None:
            on_epoch(row)
    return model, history


def final_metrics(history: list[dict]) -> dict:
    last = history[-10:]
    return {
        "acc_last10": float(np.mean([r["target_acc"] for r in last])),
        "acc_last": float(history[-1]["target_acc"]),
        "bal_acc_last10": float(np.mean([r["target_bal_acc"] for r in last])),
        "bal_acc_last": float(history[-1]["target_bal_acc"]),
    }
