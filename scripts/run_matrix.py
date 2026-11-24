"""Run the M4 matrix (TASKS M4.1) or single runs, one subprocess of scripts/run.py per run.

    python scripts/run_matrix.py --matrix configs/matrix_main.yaml --list
    python scripts/run_matrix.py --matrix configs/matrix_main.yaml --index i         # e.g. one array element
    python scripts/run_matrix.py --matrix configs/matrix_main.yaml --worker --claim-dir DIR
    python scripts/run_matrix.py --matrix configs/matrix_main.yaml --status          # after the runs
    python scripts/run_matrix.py --config configs/base.yaml --run key=value ...      # one ad hoc run

Each run is base_config + matrix overrides (frozen B, epochs) + its own (method, rho, n_te, seed).
A run whose directory already has final.json is skipped (SPEC §8). The output of every attempt is
appended to <run_dir>/run.log, so a failed run keeps its log.

--worker: loops over the matrix in order and claims each pending run by creating DIR/<index> with
O_EXCL, so several workers (one per GPU, possibly in different jobs on the shared file system) can
share one matrix. A failed run is not retried within the same claim directory.
--status: prints done / failed / pending per run and the set of git commits and GPUs among the done
runs; exit code 1 if anything is not done or commits / GPUs differ.
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scarda.config import load_config  # noqa: E402
from scarda.io_utils import run_dir  # noqa: E402

RUN_KEYS = ("method", "rho", "n_te", "seed")


def load_matrix(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        m = yaml.safe_load(f)
    for r in m["runs"]:
        if set(r) != set(RUN_KEYS):
            raise ValueError(f"matrix run {r} must have exactly the keys {RUN_KEYS}")
    return m


def run_overrides(matrix: dict, i: int) -> list[str]:
    r = matrix["runs"][i]
    return list(matrix.get("overrides", [])) + [f"{k}={r[k]}" for k in RUN_KEYS]


def execute(config: str, overrides: list[str], tag: str = "") -> int:
    """Run one configuration unless it is complete; returns the exit code (0 also for skipped)."""
    out = run_dir(load_config(config, overrides))
    if (out / "final.json").exists():
        print(f"[skip] {tag}{out.name}", flush=True)
        return 0
    out.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(ROOT / "scripts" / "run.py"), "--config", config, *overrides]
    with open(out / "run.log", "a", encoding="utf-8") as log:
        log.write(f"\n=== {datetime.now().astimezone().isoformat(timespec='seconds')} host={socket.gethostname()} "
                  f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}\n=== {' '.join(cmd)}\n")
        log.flush()
        print(f"[run] {tag}{out.name}", flush=True)
        rc = subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT).returncode
        log.write(f"=== exit code {rc}\n")
    print(f"[{'done' if rc == 0 else 'FAIL'}] {tag}{out.name} (exit {rc})", flush=True)
    return rc


def worker(matrix: dict, claim_dir: Path) -> int:
    claim_dir.mkdir(parents=True, exist_ok=True)
    failed = []
    for i in range(len(matrix["runs"])):
        ovs = run_overrides(matrix, i)
        if (run_dir(load_config(matrix["base_config"], ovs)) / "final.json").exists():
            continue
        try:
            fd = os.open(claim_dir / f"{i:03d}", os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            continue  # another worker has it
        os.write(fd, f"{socket.gethostname()} pid={os.getpid()} gpu={os.environ.get('CUDA_VISIBLE_DEVICES')}\n".encode())
        os.close(fd)
        if execute(matrix["base_config"], ovs, tag=f"#{i} ") != 0:
            failed.append(i)
    if failed:
        print(f"[worker] failed indices: {failed}", flush=True)
    return 1 if failed else 0


def status(matrix: dict) -> int:
    rows, commits, gpus = [], {}, {}
    for i in range(len(matrix["runs"])):
        out = run_dir(load_config(matrix["base_config"], run_overrides(matrix, i)))
        if (out / "final.json").exists():
            st = "done"
            meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
            commits.setdefault(meta.get("git_commit"), []).append(i)
            gpus.setdefault(meta.get("gpu"), []).append(i)
        elif out.exists():
            st = "failed/incomplete"
        else:
            st = "pending"
        rows.append((i, st, out.name))
    for i, st, name in rows:
        print(f"{i:>3d}  {st:<18s} {name}")
    n_done = sum(st == "done" for _, st, _ in rows)
    print(f"done {n_done}/{len(rows)}")
    print(f"git commits (done runs): { {k: len(v) for k, v in commits.items()} }")
    print(f"GPUs (done runs): { {k: len(v) for k, v in gpus.items()} }")
    return 0 if n_done == len(rows) and len(commits) == 1 and len(gpus) == 1 else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix")
    ap.add_argument("--config", help="base config for --run")
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--list", action="store_true")
    mode.add_argument("--index", type=int)
    mode.add_argument("--worker", action="store_true")
    mode.add_argument("--status", action="store_true")
    mode.add_argument("--run", action="store_true", help="one ad hoc run: --config + key=value overrides")
    ap.add_argument("--claim-dir")
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args(argv)

    if args.run:
        if not args.config:
            ap.error("--run needs --config")
        return execute(args.config, args.overrides)
    if not args.matrix or args.overrides:
        ap.error("--list/--index/--worker/--status need --matrix and take no overrides")
    matrix = load_matrix(args.matrix)
    n = len(matrix["runs"])
    if args.list:
        for i in range(n):
            out = run_dir(load_config(matrix["base_config"], run_overrides(matrix, i)))
            print(f"{i:>3d}  {'done' if (out / 'final.json').exists() else '-':<5s} {out.name}")
        print(f"{n} runs, plan={matrix.get('plan')}")
        return 0
    if args.index is not None:
        if not 0 <= args.index < n:
            ap.error(f"--index must be in [0, {n})")
        return execute(matrix["base_config"], run_overrides(matrix, args.index), tag=f"#{args.index} ")
    if args.worker:
        if not args.claim_dir:
            ap.error("--worker needs --claim-dir")
        return worker(matrix, Path(args.claim_dir))
    return status(matrix)


if __name__ == "__main__":
    raise SystemExit(main())
