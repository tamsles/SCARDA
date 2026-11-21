"""Training loop (SPEC §3.3, §7). Supports `source_only`, `target_only` and `oracle_iw`.

Receives only WeakSets (x, ybar), an evaluator callback (logged, never used for decisions) and,
for `oracle_iw`, precomputed per-sample S_tr weights from oracle.py.
"""
from __future__ import annotations

import math
import time
from typing import Callable, Iterator, Optional

import numpy as np
import torch

from scarda.data import WeakSet, int_seed_for, target_prior, uniform_prior
from scarda.models import build_model, predict_logits
from scarda.risk import scar_ovr_risk

METRIC_COLUMNS = [
    "epoch", "step", "train_loss_mean", "val_weak_risk", "target_acc", "target_bal_acc",
    "neg_rate", "empty_pos", "empty_neg", "ess_pos_mean", "ess_neg_mean", "w_max", "fallback",
    "kmm_iters_mean", "epoch_time",
]
Weights = tuple[torch.Tensor, torch.Tensor]


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


def weak_val_risk(model: torch.nn.Module, val: WeakSet, pi_te: torch.Tensor, device) -> float:
    """Weak-label validation risk on T_val: correction=none, true pi_te, pibar = T_val frequency."""
    logits = predict_logits(model, val.x, device)
    out = scar_ovr_risk(logits, val.ybar.to(device), pi_te.to(device, logits.dtype),
                        pibar=val.pibar_hat().to(device, logits.dtype), correction="none", pibar_mode="fixed")
    return float(out.total.item())


def method_setup(cfg, pools: dict[str, WeakSet]) -> tuple[WeakSet, np.ndarray, Optional[torch.Tensor]]:
    """(training pool, prior used in the risk, fixed pibar or None for batch mode) per method (SPEC §3.3)."""
    if cfg.method == "source_only":
        return pools["S_tr"], uniform_prior(cfg.K), None
    if cfg.method == "target_only":
        return pools["T_tr"], target_prior(cfg.K, cfg.rho), None  # oracle prior (SPEC §3.3)
    if cfg.method == "oracle_iw":
        # weighted -> pibar_mode=fixed with the observable T_tr frequency (SPEC §3.2)
        return pools["S_tr"], target_prior(cfg.K, cfg.rho), pools["T_tr"].pibar_hat()
    raise ValueError(f"method {cfg.method!r} not implemented")


def ess_ratio(w: torch.Tensor, mask: torch.Tensor) -> float:
    """ESS ratio (sum w)^2 / (m sum w^2) per non-empty group (column), averaged over groups (SPEC §8)."""
    m = mask.sum(dim=0)
    s1, s2 = (w * mask).sum(dim=0), (w * w * mask).sum(dim=0)
    ok = (m > 0) & (s2 > 0)
    return float((s1[ok] ** 2 / (m[ok] * s2[ok])).mean().item()) if bool(ok.any()) else float("nan")


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
    weighted = cfg.method == "oracle_iw"
    if weighted != (source_weights is not None):
        raise ValueError("source_weights must be given for oracle_iw and only for oracle_iw")
    device = resolve_device(cfg.device)
    epochs = cfg.epochs if epochs is None else epochs
    train_set, pi, pibar_fixed = method_setup(cfg, pools)
    pi_t = torch.as_tensor(pi, dtype=torch.float32, device=device)
    pi_te = torch.as_tensor(target_prior(cfg.K, cfg.rho), dtype=torch.float32)
    if weighted:
        w_pos_all, w_neg_all = (w.to(device=device, dtype=torch.float32) for w in source_weights)
        if w_pos_all.shape != train_set.ybar.shape or w_neg_all.shape != train_set.ybar.shape:
            raise ValueError("source_weights must match the S_tr ybar shape")
        pibar_t = pibar_fixed.to(device=device, dtype=torch.float32)

    model = build_model(train_set.x.shape[1:], cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.wd)

    x_all = train_set.x.to(device)
    yb_all = train_set.ybar.to(device=device, dtype=torch.float32)
    batches = cycle_batches(len(train_set), cfg.bs, torch.Generator().manual_seed(int_seed_for(cfg.seed, "loader")))
    steps_per_epoch = math.ceil(cfg.n_tr / cfg.bs)  # same budget for all methods (SPEC §7)

    history, step = [], 0
    for epoch in range(1, epochs + 1):
        t0 = time.perf_counter()
        model.train()
        sums = dict.fromkeys(("loss", "neg_rate", "empty_pos", "empty_neg", "ess_pos", "ess_neg"), 0.0)
        w_max = float("-inf")
        for _ in range(steps_per_epoch):
            idx = next(batches).to(device)
            yb = yb_all[idx]
            if weighted:
                wp, wn = w_pos_all[idx], w_neg_all[idx]
                out = scar_ovr_risk(model(x_all[idx]), yb, pi_t, pibar=pibar_t, w_pos=wp, w_neg=wn,
                                    correction=cfg.correction, pibar_mode="fixed")
                sums["ess_pos"] += ess_ratio(wp, yb)
                sums["ess_neg"] += ess_ratio(wn, 1.0 - yb)
                w_max = max(w_max, float(torch.maximum((wp * yb).max(), (wn * (1.0 - yb)).max()).item()))
            else:
                out = scar_ovr_risk(model(x_all[idx]), yb, pi_t, correction=cfg.correction, pibar_mode="batch")
            opt.zero_grad(set_to_none=True)
            out.total.backward()
            opt.step()
            step += 1
            sums["loss"] += out.total.item()
            sums["neg_rate"] += out.neg_rate
            sums["empty_pos"] += out.empty_pos
            sums["empty_neg"] += out.empty_neg
        val = weak_val_risk(model, pools["T_val"], pi_te, device)
        ev = evaluator(model)
        if device.type == "cuda":
            torch.cuda.synchronize()
        mean = {k: v / steps_per_epoch for k, v in sums.items()}
        nan = float("nan")
        row = {
            "epoch": epoch, "step": step,
            "train_loss_mean": mean["loss"], "val_weak_risk": val,
            "target_acc": ev["target_acc"], "target_bal_acc": ev["target_bal_acc"],
            "neg_rate": mean["neg_rate"], "empty_pos": mean["empty_pos"], "empty_neg": mean["empty_neg"],
            # weight diagnostics: nan for unweighted methods; no KMM for oracle_iw
            "ess_pos_mean": mean["ess_pos"] if weighted else nan,
            "ess_neg_mean": mean["ess_neg"] if weighted else nan,
            "w_max": w_max if weighted else nan,
            "fallback": nan, "kmm_iters_mean": nan,
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
