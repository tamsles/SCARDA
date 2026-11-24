"""Single-run entry point.

    python scripts/run.py --config configs/base.yaml method=source_only rho=20 n_te=2000 seed=0

Skips the run if final.json already exists (SPEC §8).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scarda.config import load_config  # noqa: E402
from scarda.data import build_mnist, format_pool_report  # noqa: E402
from scarda.evaluate import make_evaluator  # noqa: E402
from scarda.io_utils import MetricsWriter, now_iso, run_dir, run_info, write_config, write_json  # noqa: E402
from scarda.oracle import source_oracle_weights  # noqa: E402
from scarda.config import DIW_METHODS  # noqa: E402
from scarda.train import METRIC_COLUMNS, diw_schedule, final_metrics, fit, resolve_device  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("overrides", nargs="*", help="key=value")
    args = ap.parse_args(argv)
    cfg = load_config(args.config, args.overrides)

    out = run_dir(cfg)
    if (out / "final.json").exists():
        print(f"[skip] {out} already has final.json")
        return 0
    out.mkdir(parents=True, exist_ok=True)
    device = resolve_device(cfg.device)

    pools, labels, info = build_mnist(cfg)
    print(format_pool_report(info))

    write_config(out, cfg)
    meta = {"seed": cfg.seed, **run_info(device),
            "pools": {k: v for k, v in info.items() if k != "indices"}, "start_time": now_iso()}
    if cfg.method in DIW_METHODS:
        meta["diw"] = diw_schedule(cfg, pools)  # warm-up switch point (SPEC §6.8), B_te size, eta
    write_json(out / "meta.json", meta)

    evaluator = make_evaluator(labels.eval_x, labels.eval_y, cfg.K, device)
    source_weights = (source_oracle_weights(labels.source_y, pools["S_tr"].ybar, cfg)
                      if cfg.method == "oracle_iw" else None)
    writer = MetricsWriter(out / "metrics.csv", METRIC_COLUMNS)

    def on_epoch(row):
        writer.write(row)
        print(f"ep {row['epoch']:>3d} loss {row['train_loss_mean']:.4f} val {row['val_weak_risk']:.4f} "
              f"acc {row['target_acc']:.4f} bal {row['target_bal_acc']:.4f} neg {row['neg_rate']:.3f} "
              f"t {row['epoch_time']:.2f}s", flush=True)

    t0 = time.perf_counter()
    _, history = fit(cfg, pools, evaluator, on_epoch=on_epoch, source_weights=source_weights)
    meta.update({"end_time": now_iso(), "elapsed_sec": time.perf_counter() - t0})
    write_json(out / "meta.json", meta)
    final = final_metrics(history)
    write_json(out / "final.json", final)
    print(f"[done] {out}\n{final}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
