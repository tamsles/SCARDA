"""M3 freeze: choose B (SPEC §7), choose the matrix plan (SPEC §9.2), write frozen.yaml / matrix_main.yaml.

    python scripts/freeze.py --config configs/base.yaml --bench-dir DIR --P 8 --H 172800 [key=value ...]

Inputs
  - the two selection runs: method=diw, rho=20, n_te=2000, seed=0, B in {5, 10}, located through the
    same config resolution as run.py (base config + trailing key=value overrides + run fields);
  - DIR/{source_only,diw,diw_combined}.json written by `bench_time.py --json` (DIW benched with
    diw_warmup_epochs=0 so that every timed epoch solves KMM).

Selection (SPEC §7): smallest mean `val_weak_risk` over the last 10 epochs. Only the columns listed in
READ_COLUMNS are read from metrics.csv; `target_acc` / `target_bal_acc` are never read. A tie goes to
the smaller B.

Plan (SPEC §9.2), t = max(per-epoch time of diw, diw_combined), E = epochs:
    25*E*t/P <= H  -> full (56 runs)
    13*E*t/P <= H  -> drop_rho5 (29 runs)
    otherwise      -> stop and report (abort)

Abort (exit code 2, reasons printed and written to the record) when: a selection run is missing or
incomplete; a NaN appears (loss / val risk / risk counters in any epoch, DIW diagnostics after warm-up);
in either run, ess_pos_mean or ess_neg_mean is < 0.05 in more than half of the post-warm-up epochs;
a bench file is missing or invalid; or §9.2 says stop. On abort neither yaml file is written.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scarda.config import load_config  # noqa: E402
from scarda.io_utils import now_iso, run_dir  # noqa: E402

B_CANDIDATES = (5.0, 10.0)
SELECTION_RUN = {"method": "diw", "rho": 20, "n_te": 2000, "seed": 0}  # SPEC §7
LAST = 10
ESS_MIN = 0.05
ALWAYS_FINITE = ("train_loss_mean", "val_weak_risk", "neg_rate", "empty_pos", "empty_neg")
DIW_DIAG = ("ess_pos_mean", "ess_neg_mean", "w_max", "fallback", "kmm_iters_mean")
READ_COLUMNS = ("epoch",) + ALWAYS_FINITE + DIW_DIAG  # never target_acc / target_bal_acc
N_DIW = {"full": 25, "drop_rho5": 13}  # DIW-type runs in the matrix (SPEC §9.2)
N_OTHER = {"full": 31, "drop_rho5": 16}  # non-DIW runs, recorded for information only
BENCH_METHODS = ("source_only", "diw", "diw_combined")


class Abort(Exception):
    pass


def ov(d: dict) -> list[str]:
    return [f"{k}={v}" for k, v in d.items()]


def read_metrics(path: Path) -> list[dict]:
    """Rows of metrics.csv restricted to READ_COLUMNS (floats)."""
    with open(path, newline="", encoding="utf-8") as f:
        rd = csv.reader(f)
        header = next(rd)
        missing = [c for c in READ_COLUMNS if c not in header]
        if missing:
            raise Abort(f"{path}: missing columns {missing}")
        pos = {c: header.index(c) for c in READ_COLUMNS}
        return [{c: float(row[i]) for c, i in pos.items()} for row in rd if row]


def check_selection_run(cfg, out: Path) -> dict:
    """Completeness, NaN and ESS checks of one selection run; returns its record."""
    rec = {"run_dir": out.name, "B": cfg.B}
    if not (out / "final.json").exists():
        raise Abort(f"selection run B={cfg.B}: {out.name} has no final.json (failed or not run)")
    rows = read_metrics(out / "metrics.csv")
    if len(rows) != cfg.epochs or [int(r["epoch"]) for r in rows] != list(range(1, cfg.epochs + 1)):
        raise Abort(f"selection run B={cfg.B}: expected epochs 1..{cfg.epochs}, found {len(rows)} rows")
    post = [r for r in rows if r["epoch"] > cfg.diw_warmup_epochs]
    if not post:
        raise Abort(f"selection run B={cfg.B}: no post-warm-up epoch (epochs={cfg.epochs}, "
                    f"diw_warmup_epochs={cfg.diw_warmup_epochs})")
    bad = [f"epoch {int(r['epoch'])} {c}" for r in rows for c in ALWAYS_FINITE if math.isnan(r[c])]
    bad += [f"epoch {int(r['epoch'])} {c}" for r in post for c in DIW_DIAG if math.isnan(r[c])]
    if bad:
        raise Abort(f"selection run B={cfg.B}: NaN in {', '.join(bad[:10])}" + (" ..." if len(bad) > 10 else ""))
    for c in ("ess_pos_mean", "ess_neg_mean"):
        low = sum(r[c] < ESS_MIN for r in post)
        rec[f"{c}_below_{ESS_MIN}"] = f"{low}/{len(post)}"
        if low > len(post) / 2:
            raise Abort(f"selection run B={cfg.B}: {c} < {ESS_MIN} in {low}/{len(post)} post-warm-up epochs "
                        f"(weight degeneration, TASKS M3 stop-loss)")
    last = rows[-LAST:]
    rec["val_weak_risk_last10"] = sum(r["val_weak_risk"] for r in last) / len(last)
    rec["ess_pos_mean_post"] = sum(r["ess_pos_mean"] for r in post) / len(post)
    rec["ess_neg_mean_post"] = sum(r["ess_neg_mean"] for r in post) / len(post)
    rec["fallback_mean_post"] = sum(r["fallback"] for r in post) / len(post)
    meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
    rec.update({k: meta.get(k) for k in ("git_commit", "gpu", "elapsed_sec")})
    return rec


def read_bench(bench_dir: Path) -> dict:
    res = {}
    for m in BENCH_METHODS:
        p = bench_dir / f"{m}.json"
        if not p.exists():
            raise Abort(f"bench file {p} missing")
        b = json.loads(p.read_text(encoding="utf-8"))
        if b.get("method") != m:
            raise Abort(f"bench file {p}: method {b.get('method')!r} != {m!r}")
        if m != "source_only" and b.get("diw_warmup_epochs") != 0:
            raise Abort(f"bench file {p}: diw_warmup_epochs={b.get('diw_warmup_epochs')} (must be 0, "
                        "otherwise only warm-up epochs are timed)")
        t = b.get("mean_rest")
        if not (isinstance(t, (int, float)) and math.isfinite(t) and t > 0):
            raise Abort(f"bench file {p}: invalid mean_rest {t!r}")
        res[m] = {"mean_rest": t, "std_rest": b.get("std_rest"), "epochs": b.get("epochs"), "gpu": b.get("gpu"),
                  "git_commit": b.get("git_commit")}
    return res


def choose_plan(t: float, t_src: float, E: int, P: int, H: float) -> dict:
    need = {p: N_DIW[p] * E * t / P for p in N_DIW}
    incl = {p: (N_DIW[p] * t + N_OTHER[p] * t_src) * E / P for p in N_DIW}
    if need["full"] <= H:
        plan = "full"
    elif need["drop_rho5"] <= H:
        plan = "drop_rho5"
    else:
        plan = "stop"
    return {"t": t, "t_source_only": t_src, "E": E, "P": P, "H": H, "need_sec": need,
            "need_sec_incl_non_diw": incl, "plan": plan}


def matrix_runs(plan: str) -> list[dict]:
    """SPEC §9 matrix, longest runs first (DIW), then the rest; rho=1 controls (seed 0, n_te=2000)."""
    rhos = [5, 20] if plan == "full" else [20]
    grid = [(r, n, s) for r in rhos for n in (500, 2000) for s in (0, 1, 2)]
    runs = []
    for m in ("diw_combined", "diw"):
        runs += [{"method": m, "rho": r, "n_te": n, "seed": s} for r, n, s in grid]
    runs.append({"method": "diw", "rho": 1, "n_te": 2000, "seed": 0})
    for m in ("oracle_iw", "target_only"):
        runs += [{"method": m, "rho": r, "n_te": n, "seed": s} for r, n, s in grid]
    # source_only does not depend on n_te (SPEC §9): one run per (rho, seed), at n_te=2000
    runs += [{"method": "source_only", "rho": r, "n_te": 2000, "seed": s} for r in rhos for s in (0, 1, 2)]
    runs.append({"method": "source_only", "rho": 1, "n_te": 2000, "seed": 0})
    return runs


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--bench-dir", required=True)
    ap.add_argument("--P", type=int, required=True, help="concurrent runs (SPEC §9.2)")
    ap.add_argument("--H", type=float, required=True, help="M4 wall-clock budget in seconds (SPEC §9.2)")
    ap.add_argument("--out-frozen", default="configs/frozen.yaml")
    ap.add_argument("--out-matrix", default="configs/matrix_main.yaml")
    ap.add_argument("--out-record", default="results/checks/freeze.json")
    ap.add_argument("overrides", nargs="*", help="key=value applied to every run (local tests only)")
    args = ap.parse_args(argv)
    if args.P < 1 or not args.H > 0:
        ap.error("need P >= 1 and H > 0")

    record = {"time": now_iso(), "config": args.config, "overrides": args.overrides, "status": None}
    try:
        sel = {}
        for b in B_CANDIDATES:
            cfg = load_config(args.config, args.overrides + ov(SELECTION_RUN) + [f"B={b}"])
            sel[b] = check_selection_run(cfg, run_dir(cfg))
        E = cfg.epochs
        record["selection"] = {"rule": "argmin mean val_weak_risk over the last 10 epochs (SPEC §7)",
                               "runs": {str(b): r for b, r in sel.items()}}
        chosen = min(B_CANDIDATES, key=lambda b: (sel[b]["val_weak_risk_last10"], b))
        record["selection"]["B"] = chosen
        print(f"[freeze] selection: " + ", ".join(f"B={b:g}: val_last10={sel[b]['val_weak_risk_last10']:.6f}"
                                                  for b in B_CANDIDATES) + f" -> B={chosen:g}")

        bench = read_bench(Path(args.bench_dir))
        record["bench"] = bench
        t = max(bench["diw"]["mean_rest"], bench["diw_combined"]["mean_rest"])
        budget = choose_plan(t, bench["source_only"]["mean_rest"], E, args.P, args.H)
        record["budget"] = budget
        print(f"[freeze] t={t:.2f}s/epoch E={E} P={args.P} H={args.H:.0f}s; "
              f"need full={budget['need_sec']['full']:.0f}s drop_rho5={budget['need_sec']['drop_rho5']:.0f}s "
              f"-> {budget['plan']}")
        if budget["plan"] == "stop":
            raise Abort("SPEC §9.2: 13*E*t/P > H even after dropping rho=5 -> stop and report "
                        "(human decides: lower E or speed up the solver; no ADIW-style warm start)")
    except Abort as e:
        record.update(status="aborted", reason=str(e))
        _write_json(Path(args.out_record), record)
        print(f"[freeze] ABORT: {e}", file=sys.stderr, flush=True)
        print(f"[freeze] ABORT: {e}", flush=True)
        return 2

    frozen_overrides = {"B": chosen, "epochs": E}
    runs = matrix_runs(budget["plan"])
    commits = {sel[b]["git_commit"] for b in B_CANDIDATES} | {bench[m]["git_commit"] for m in BENCH_METHODS}
    frozen = {"B": chosen, "epochs": E, "plan": budget["plan"], "n_runs": len(runs),
              "selection": {str(b): round(sel[b]["val_weak_risk_last10"], 8) for b in B_CANDIDATES},
              "t_diw_epoch_sec": round(t, 3), "P": args.P, "H": args.H,
              "git_commit": sorted(c for c in commits if c) or None,
              "frozen_at": record["time"]}
    matrix = {"base_config": args.config, "overrides": ov(frozen_overrides) + list(args.overrides),
              "plan": budget["plan"], "runs": runs}
    record.update(status="ok", frozen=frozen, n_runs=len(runs))
    _write_yaml(Path(args.out_frozen), frozen, "# M3 freeze (scripts/freeze.py). Matrix runs use base config + B, epochs.\n")
    _write_yaml(Path(args.out_matrix), matrix, "# M4 matrix (scripts/freeze.py, SPEC §9). Read by scripts/run_matrix.py.\n")
    _write_json(Path(args.out_record), record)
    print(f"[freeze] OK: B={chosen:g}, plan={budget['plan']}, {len(runs)} runs -> {args.out_frozen}, {args.out_matrix}")
    return 0


def _path(p: Path) -> Path:
    return p if p.is_absolute() else ROOT / p


def _write_yaml(p: Path, obj: dict, header: str) -> None:
    p = _path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        f.write(header)
        yaml.safe_dump(obj, f, sort_keys=False, allow_unicode=True)


def _write_json(p: Path, obj: dict) -> None:
    p = _path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
