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

from scarda.config import config_hash, load_config  # noqa: E402
from scarda.data import build_mnist, format_pool_report  # noqa: E402
from scarda.evaluate import make_evaluator  # noqa: E402
from scarda.io_utils import (MetricsWriter, device_info, env_info, git_info, now_iso, run_dir,  # noqa: E402
                             write_config, write_json)
from scarda.train import METRIC_COLUMNS, final_metrics, resolve_device, train  # noqa: E402


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

    pools, vault, record = build_mnist(cfg)
    print(format_pool_report(record))
    print(f"split_hash={record.split_hash}")

    write_config(out, cfg)
    meta = {
        **git_info(), **env_info(), **device_info(device),
        "config_hash": config_hash(cfg),
        "split_hash": record.split_hash,
        "split_hashes": record.hashes,
        "base_hash": record.base_hash,
        "pools": {k: v for k, v in record.summary().items() if k not in ("hashes", "split_hash", "base_hash")},
        "start_time": now_iso(),
    }
    write_json(out / "meta.json", meta)

    evaluator = make_evaluator(vault, cfg.K, device)
    writer = MetricsWriter(out / "metrics.csv", METRIC_COLUMNS)

    def on_epoch(row):
        writer.write(row)
        print(f"ep {row['epoch']:>3d} loss {row['train_loss_mean']:.4f} val {row['val_weak_risk']:.4f} "
              f"acc {row['target_acc']:.4f} bal {row['target_bal_acc']:.4f} neg {row['neg_rate']:.3f} "
              f"t {row['epoch_time']:.2f}s", flush=True)

    t0 = time.perf_counter()
    history = train(cfg, pools, evaluator, on_epoch=on_epoch)
    meta.update({"end_time": now_iso(), "elapsed_sec": time.perf_counter() - t0})
    write_json(out / "meta.json", meta)
    final = final_metrics(history)
    write_json(out / "final.json", final)
    print(f"[done] {out}\n{final}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
