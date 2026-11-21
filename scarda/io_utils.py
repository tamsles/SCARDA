"""Run directories and result files (SPEC §8)."""
from __future__ import annotations

import csv
import json
import math
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
import yaml

from scarda.config import Config, config_dict, config_hash

REPO_ROOT = Path(__file__).resolve().parents[1]


def run_name(cfg: Config) -> str:
    rho = int(cfg.rho) if float(cfg.rho).is_integer() else cfg.rho
    return f"{cfg.method}_rho{rho}_nte{cfg.n_te}_seed{cfg.seed}_{config_hash(cfg)[:8]}"


def run_dir(cfg: Config) -> Path:
    root = Path(cfg.out_root)
    if not root.is_absolute():
        root = REPO_ROOT / root
    return root / run_name(cfg)


def git_info() -> dict:
    def _git(*args):
        return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True).stdout.strip()

    try:
        commit = _git("rev-parse", "HEAD")
        status = _git("status", "--porcelain", "--untracked-files=no")
        return {"git_commit": commit, "git_dirty": bool(status), "git_dirty_files": status.splitlines()}
    except OSError:
        return {"git_commit": None, "git_dirty": None, "git_dirty_files": []}


def device_info(device: torch.device) -> dict:
    info = {"device": str(device), "cpu": platform.processor() or platform.machine()}
    if device.type == "cuda":
        info["gpu"] = torch.cuda.get_device_name(device)
        info["cuda"] = torch.version.cuda
    return info


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def env_info() -> dict:
    return {"python": sys.version.split()[0], "torch": torch.__version__, "platform": platform.platform()}


def write_config(path: Path, cfg: Config) -> None:
    with open(path / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump({**config_dict(cfg), "config_hash": config_hash(cfg)}, f, sort_keys=False)


def write_json(path: Path, obj: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


class MetricsWriter:
    def __init__(self, path: Path, columns: list[str]):
        self.path, self.columns = path, columns
        with open(path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(columns)

    def write(self, row: dict) -> None:
        def fmt(v):
            if isinstance(v, float):
                return "nan" if math.isnan(v) else f"{v:.6g}"
            return v

        with open(self.path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([fmt(row[c]) for c in self.columns])
