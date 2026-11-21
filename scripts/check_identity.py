"""Risk identity check on MNIST (TASKS M2.4).

    python scripts/check_identity.py --config configs/base.yaml [rho=20 n_te=2000 seed=0] [--boot 2000]

For two fixed classifiers -- (1) the random initialization used by train.fit (models.build_model),
(2) source_only trained for 1 epoch -- computes, all with correction=none:

  (a) true-label target OVR risk on T_eval:   mean_i [ell(f_y) + sum_{k!=y} ell(-f_k)]
  (b) target weak SCAR risk on T_val:          pi_te (config), pibar = T_val frequency, unweighted
  (c) oracle-weighted source risk on S_tr:     pi_te, pibar = T_tr frequency, oracle weights (SPEC §5)
  control: unweighted source risk on S_tr:     pi_tr, pibar = S_tr frequency (expected to differ)

(c) and control are evaluated on an independent redraw of the S_tr complementary labels
(oracle.fresh_source_ybar, generator SeedSequence([seed, 101])), not on S_tr.ybar: classifier (2) is
trained on S_tr.ybar, so its weak risk on those labels is biased low (the identity needs a classifier
independent of the evaluation labels). The values on the training labels are kept as a diagnostic
("in_sample_training_ybar"), without standard errors.

Standard errors: nonparametric bootstrap (B resamples with replacement of the rows of each pool;
(c) resamples S_tr and T_tr jointly so the pibar estimate is included). The pools are disjoint and
independently labeled, so the SE of a difference is sqrt(SE_u^2 + SE_v^2). The bootstrap ignores the
per-class stratification of the pools (slightly conservative).
Output: results/checks/identity.json.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scarda.config import load_config  # noqa: E402
from scarda.data import build_mnist, target_prior, uniform_prior  # noqa: E402
from scarda.evaluate import make_evaluator, true_ovr_losses  # noqa: E402
from scarda.io_utils import now_iso, run_info  # noqa: E402
from scarda.models import build_model, predict_logits  # noqa: E402
from scarda.oracle import fresh_source_ybar, source_oracle_weights  # noqa: E402
from scarda.risk import scar_ovr_risk  # noqa: E402
from scarda.train import fit, resolve_device  # noqa: E402

BOOT_SEED_TAG = 100  # bootstrap generator: SeedSequence([seed, 100]); not used anywhere else
FRESH_YBAR_SEED_TAG = 101  # S_tr ybar redraw: SeedSequence([seed, 101]); not used anywhere else


def weak_risk_counts(lf, lmf, Yp, Ap, An, pi, pibar, C):
    """SPEC §3.1 risk (correction=none) for each row of the count matrix C [b, n].

    lf = ell(f), lmf = ell(-f), Yp = ybar, Ap / An = in-group weights (ybar*w+, (1-ybar)*w-), all [n, K];
    pibar: [K] or [b, K]. Returns [b]. With C = ones(1, n) this is exactly scar_ovr_risk."""
    sp, sn = C @ Ap, C @ An
    Mp_mf = np.divide(C @ (Ap * lmf), sp, out=np.zeros_like(sp), where=sp > 0)
    Mp_f = np.divide(C @ (Ap * lf), sp, out=np.zeros_like(sp), where=sp > 0)
    Mn_f = np.divide(C @ (An * lf), sn, out=np.zeros_like(sn), where=sn > 0)
    N = (1 - pi) * Mp_mf
    P = (pibar + pi - 1) * Mp_f + (1 - pibar) * Mn_f
    return (N + P).sum(axis=1)


def boot_counts(rng, n, b):
    return rng.multinomial(n, np.full(n, 1.0 / n), size=b).astype(np.float64)


def logits_of(model, x, device):
    return predict_logits(model, x, device).double().cpu()


def evaluate_classifier(model, pools, labels, eval_ybar_S, weights, train_ybar_S, train_weights, cfg, device, B,
                        rng, chunk=100):
    """(c) and control use `eval_ybar_S` / `weights` (the independent redraw); `train_ybar_S` /
    `train_weights` (S_tr.ybar) only give the in-sample diagnostic point values."""
    pi_te, pi_tr = target_prior(cfg.K, cfg.rho), uniform_prior(cfg.K)
    S, Tt, V = pools["S_tr"], pools["T_tr"], pools["T_val"]
    fS, fV = logits_of(model, S.x, device), logits_of(model, V.x, device)
    a_loss = true_ovr_losses(model, labels.eval_x, labels.eval_y, device).numpy()

    def parts(f, ybar, wp=None, wn=None):
        Yp = ybar.double().numpy()
        Ap = Yp if wp is None else wp.double().numpy() * Yp
        An = (1 - Yp) if wn is None else wn.double().numpy() * (1 - Yp)
        sp = torch.nn.functional.softplus
        return sp(-f).numpy(), sp(f).numpy(), Yp, Ap, An

    lfS, lmfS, YS, ApS, AnS = parts(fS, eval_ybar_S, *weights)
    _, _, _, ApS0, AnS0 = parts(fS, eval_ybar_S)
    lfV, lmfV, YV, ApV, AnV = parts(fV, V.ybar)
    YT = Tt.ybar.double().numpy()

    # point estimates, cross-checked against scar_ovr_risk
    pibar_T, pibar_V, pibar_S = YT.mean(0), YV.mean(0), YS.mean(0)
    pt = {
        "a": float(a_loss.mean()),
        "b": float(weak_risk_counts(lfV, lmfV, YV, ApV, AnV, pi_te, pibar_V, np.ones((1, len(YV))))[0]),
        "c": float(weak_risk_counts(lfS, lmfS, YS, ApS, AnS, pi_te, pibar_T, np.ones((1, len(YS))))[0]),
        "control": float(weak_risk_counts(lfS, lmfS, YS, ApS0, AnS0, pi_tr, pibar_S, np.ones((1, len(YS))))[0]),
    }
    ref = {
        "b": scar_ovr_risk(fV, V.ybar, torch.as_tensor(pi_te), pibar=torch.as_tensor(pibar_V),
                           correction="none", pibar_mode="fixed").total.item(),
        "c": scar_ovr_risk(fS, eval_ybar_S, torch.as_tensor(pi_te), pibar=torch.as_tensor(pibar_T),
                           w_pos=weights[0].double(), w_neg=weights[1].double(),
                           correction="none", pibar_mode="fixed").total.item(),
        "control": scar_ovr_risk(fS, eval_ybar_S, torch.as_tensor(pi_tr), correction="none",
                                 pibar_mode="batch").total.item(),
    }
    for k, v in ref.items():
        assert abs(pt[k] - v) < 1e-9, (k, pt[k], v)

    # diagnostic only: the same quantities on the labels the classifier may have been trained on
    in_sample = {
        "c": scar_ovr_risk(fS, train_ybar_S, torch.as_tensor(pi_te), pibar=torch.as_tensor(pibar_T),
                           w_pos=train_weights[0].double(), w_neg=train_weights[1].double(),
                           correction="none", pibar_mode="fixed").total.item(),
        "control": scar_ovr_risk(fS, train_ybar_S, torch.as_tensor(pi_tr), correction="none",
                                 pibar_mode="batch").total.item(),
    }

    boots = {k: [] for k in pt}
    for start in range(0, B, chunk):
        b = min(chunk, B - start)
        CA, CV = boot_counts(rng, len(a_loss), b), boot_counts(rng, len(YV), b)
        CS, CT = boot_counts(rng, len(YS), b), boot_counts(rng, len(YT), b)
        boots["a"].append(CA @ a_loss / len(a_loss))
        boots["b"].append(weak_risk_counts(lfV, lmfV, YV, ApV, AnV, pi_te, CV @ YV / len(YV), CV))
        boots["c"].append(weak_risk_counts(lfS, lmfS, YS, ApS, AnS, pi_te, CT @ YT / len(YT), CS))
        boots["control"].append(weak_risk_counts(lfS, lmfS, YS, ApS0, AnS0, pi_tr, CS @ YS / len(YS), CS))
    se = {k: float(np.concatenate(v).std(ddof=1)) for k, v in boots.items()}

    risks = {k: {"value": pt[k], "se": se[k]} for k in pt}
    diffs = {}
    for u, v in (("a", "b"), ("a", "c"), ("b", "c"), ("a", "control")):
        d, s = pt[u] - pt[v], float(np.hypot(se[u], se[v]))
        diffs[f"{u}-{v}"] = {"diff": d, "se": s, "z": d / s, "within_3se": bool(abs(d) <= 3 * s)}
    return {"risks": risks, "diffs": diffs, "in_sample_training_ybar": in_sample, "n": {"T_eval": len(a_loss), "T_val": len(YV), "S_tr": len(YS),
                                                  "T_tr": len(YT)}}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--out", default=str(ROOT / "results" / "checks" / "identity.json"))
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args(argv)
    cfg = load_config(args.config, args.overrides)
    device = resolve_device(cfg.device)

    pools, labels, _ = build_mnist(cfg)
    train_weights = source_oracle_weights(labels.source_y, pools["S_tr"].ybar, cfg)
    eval_ybar_S, weights = fresh_source_ybar(
        labels.source_y, cfg, np.random.default_rng(np.random.SeedSequence([cfg.seed, FRESH_YBAR_SEED_TAG])))
    rng = np.random.default_rng(np.random.SeedSequence([cfg.seed, BOOT_SEED_TAG]))

    init_model = build_model(pools["S_tr"].x.shape[1:], cfg).to(device)  # same initialization as train.fit
    src_cfg = dataclasses.replace(cfg, method="source_only")
    trained, hist = fit(src_cfg, pools, make_evaluator(labels.eval_x, labels.eval_y, cfg.K, device), epochs=1)

    result = {
        "description": "Risk identity check (TASKS M2.4): (a) true-label target OVR risk on T_eval; "
                       "(b) target weak SCAR risk on T_val (pi_te, pibar=T_val freq); "
                       "(c) oracle-weighted source risk on S_tr (pi_te, pibar=T_tr freq); "
                       "control = unweighted source risk (pi_tr). correction=none. "
                       "(c) and control use an independent redraw of the S_tr complementary labels "
                       "(not the training labels); in_sample_training_ybar = same on S_tr.ybar, diagnostic only. "
                       "SE: nonparametric bootstrap per pool; diff SE = sqrt(SE_u^2+SE_v^2).",
        "fresh_ybar_seed": f"SeedSequence([{cfg.seed}, {FRESH_YBAR_SEED_TAG}])",
        "config": {"rho": cfg.rho, "n_te": cfg.n_te, "seed": cfg.seed, "gen": cfg.gen,
                   "c_tr": cfg.effective_c[0], "c_te": cfg.effective_c[1]},
        "bootstrap": {"B": args.boot, "seed": f"SeedSequence([{cfg.seed}, {BOOT_SEED_TAG}])"},
        **run_info(device), "time": now_iso(),
        "classifiers": {},
    }
    for name, model in (("random_init", init_model), ("source_only_1epoch", trained)):
        res = evaluate_classifier(model, pools, labels, eval_ybar_S, weights, pools["S_tr"].ybar, train_weights,
                                  cfg, device, args.boot, rng)
        result["classifiers"][name] = res
        print(f"== {name}")
        for k, v in res["risks"].items():
            print(f"  {k:<8} {v['value']:.5f}  (se {v['se']:.5f})")
        for k, v in res["in_sample_training_ybar"].items():
            print(f"  {k + ' (on training ybar, diagnostic)':<8} {v:.5f}")
        for k, v in res["diffs"].items():
            print(f"  {k:<10} {v['diff']:+.5f}  (se {v['se']:.5f}, z {v['z']:+.2f}, within 3se: {v['within_3se']})")
    result["source_only_1epoch_train_loss"] = hist[0]["train_loss_mean"]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"[written] {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
