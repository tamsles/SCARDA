"""Run configuration: YAML + dataclass, with `key=value` CLI overrides."""
from __future__ import annotations

import dataclasses
import hashlib
from dataclasses import dataclass, fields
from pathlib import Path

import yaml

METHODS = ("source_only", "target_only", "oracle_iw")
# Fields that do not affect results and are excluded from the config hash.
NON_SEMANTIC = ("device", "data_root", "out_root")


@dataclass(frozen=True)
class Config:
    method: str = "source_only"
    rho: float = 20.0
    n_te: int = 2000
    seed: int = 0
    # data protocol (SPEC §4)
    K: int = 10
    n_tr: int = 20000
    n_te_max: int = 2000  # size of the T_tr parent pool; smaller n_te are per-class prefixes
    n_val: int = 2000
    n_eval: int = 3000
    gen: str = "bernoulli"  # bernoulli | single_uniform
    c_tr: float = 0.3
    c_te: float = 0.2
    # training (SPEC §7)
    epochs: int = 100
    bs: int = 256
    lr: float = 1e-3
    wd: float = 1e-5
    hidden: int = 500
    correction: str = "abs"
    # environment (not hashed)
    device: str = "auto"
    data_root: str = "dataset/mnist"
    out_root: str = "results/runs"

    def __post_init__(self):
        if self.method not in METHODS:
            raise ValueError(f"method {self.method!r} not supported (available: {METHODS})")
        if self.gen not in ("bernoulli", "single_uniform"):
            raise ValueError(f"unknown gen {self.gen!r}")
        if self.correction not in ("abs", "relu", "none"):
            raise ValueError(f"unknown correction {self.correction!r}")
        if not (0 < self.n_te <= self.n_te_max):
            raise ValueError(f"n_te={self.n_te} must be in (0, n_te_max={self.n_te_max}]")
        if self.rho <= 0:
            raise ValueError("rho must be positive")

    @property
    def effective_c(self) -> tuple[float, float]:
        """(c_tr, c_te) actually used by the label generator; single_uniform fixes c = 1/(K-1)."""
        if self.gen == "single_uniform":
            return 1.0 / (self.K - 1), 1.0 / (self.K - 1)
        return self.c_tr, self.c_te


def _coerce(name: str, raw, typ):
    if typ in (int, "int"):
        if isinstance(raw, float) and not raw.is_integer():
            raise ValueError(f"{name} must be an integer, got {raw}")
        return int(raw)
    if typ in (float, "float"):
        return float(raw)
    if typ in (str, "str"):
        return str(raw)
    return raw


def load_config(path: str | Path | None, overrides: list[str] | None = None) -> Config:
    data: dict = {}
    if path is not None:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    for ov in overrides or []:
        if "=" not in ov:
            raise ValueError(f"override must be key=value, got {ov!r}")
        k, v = ov.split("=", 1)
        data[k.strip()] = yaml.safe_load(v)
    known = {f.name: f.type for f in fields(Config)}
    unknown = set(data) - set(known)
    if unknown:
        raise ValueError(f"unknown config keys: {sorted(unknown)}")
    return Config(**{k: _coerce(k, v, known[k]) for k, v in data.items()})


def config_dict(cfg: Config) -> dict:
    return dataclasses.asdict(cfg)


def config_hash(cfg: Config) -> str:
    d = {k: v for k, v in config_dict(cfg).items() if k not in NON_SEMANTIC}
    return hashlib.sha256(yaml.safe_dump(d, sort_keys=True).encode("utf-8")).hexdigest()
