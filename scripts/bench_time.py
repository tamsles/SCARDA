"""Per-epoch wall time of a method on real MNIST (TASKS M1.4).

    python scripts/bench_time.py --config configs/base.yaml [--epochs 5] [--json out.json] [key=value ...]

The first epoch is reported separately (CUDA / allocator warm-up); the mean and std are over the
remaining epochs. epoch_time includes the T_val weak risk and the T_eval accuracy pass.
For DIW methods pass diw_warmup_epochs=0, otherwise short benchmarks time only warm-up epochs.
`--json` also writes the numbers (read by scripts/freeze.py).
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scarda.config import load_config  # noqa: E402
from scarda.data import build_mnist  # noqa: E402
from scarda.evaluate import make_evaluator  # noqa: E402
from scarda.io_utils import run_info  # noqa: E402
from scarda.oracle import source_oracle_weights  # noqa: E402
from scarda.train import fit, resolve_device  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--json", default=None, help="also write the timings to this JSON file")
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args(argv)
    cfg = load_config(args.config, args.overrides)
    device = resolve_device(cfg.device)
    print(run_info(device))

    t0 = time.perf_counter()
    pools, labels, _ = build_mnist(cfg)
    t_data = time.perf_counter() - t0
    evaluator = make_evaluator(labels.eval_x, labels.eval_y, cfg.K, device)
    sw = source_oracle_weights(labels.source_y, pools["S_tr"].ybar, cfg) if cfg.method == "oracle_iw" else None
    _, hist = fit(cfg, pools, evaluator, epochs=args.epochs, source_weights=sw)
    times = [r["epoch_time"] for r in hist]
    rest = times[1:] or times
    print(f"method={cfg.method} steps/epoch={hist[0]['step']} data_build={t_data:.2f}s")
    print("epoch times: " + ", ".join(f"{t:.3f}" for t in times))
    mean_rest, std_rest = statistics.mean(rest), statistics.stdev(rest) if len(rest) > 1 else 0.0
    print(f"first={times[0]:.3f}s  mean(rest)={mean_rest:.3f}s  std(rest)={std_rest:.3f}s")
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump({"method": cfg.method, "overrides": args.overrides, "epochs": args.epochs,
                       "diw_warmup_epochs": cfg.diw_warmup_epochs, "steps_per_epoch": hist[0]["step"],
                       "data_build_sec": t_data, "epoch_times": times, "first": times[0],
                       "mean_rest": mean_rest, "std_rest": std_rest, **run_info(device)}, f, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
