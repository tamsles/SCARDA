# SCARDA

SCARDA studies **complementary-label learning under label-prior shift with a selected-completely-at-random (SCAR) labelling process**.
It extends the SCARCE risk estimator (Wang et al., ICML 2024) to a train/test domain-shift setting. The test-domain OVR risk is estimated
from training-domain data re-weighted by conditional density ratios, and the ratios are estimated by **DIW** (dynamic importance weighting,
per-mini-batch KMM on the hidden representation, one problem per complementary-label group).

SCARDA 研究类先验偏移下、SCAR 互补标签设定的学习问题：在 SCARCE 风险估计器的基础上，用条件密度比加权把训练域数据转换成测试域风险的估计，密度比由 DIW 在每个 mini-batch 内按互补标签分组估计。

## Repository layout

| Path | Content |
|---|---|
| `scarda/` | Main package: `risk.py` (SCAR OVR risk), `diw.py` (DIW weights), `oracle.py` (oracle weights), `data.py` (data protocol), `models.py`, `train.py`, `evaluate.py`, `io_utils.py`, `config.py` |
| `scripts/` | `run.py` (single run), `run_matrix.py` (experiment matrix), `freeze.py` (freeze config), `bench_time.py`, `check_identity.py`, `wisteria/` (UTokyo Wisteria/BDEC-01 job scripts) |
| `configs/` | YAML configs (`base.yaml`) |
| `tests/` | pytest suite (risk, data, DIW, oracle) |
| `specs/` | `SPEC.md` (math and data protocol, authoritative), `TASKS.md` (milestones), `AGENTS.md` (working rules) |
| `results/checks/` | Recorded sanity checks (e.g. numerical identity with the original SCARCE loss) |
| `main.py`, `utils_algo.py`, `utils_data.py`, `models.py` (repo root) | Original SCARCE code, kept as the numerical reference |

## Setup

Python >= 3.10 (tested with 3.10.11).

```bash
# GPU (CUDA 12.4 wheels)
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu124
# CPU only
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

## Usage

```bash
# tests
pytest -q

# one run: config + key=value overrides
python scripts/run.py --config configs/base.yaml

# one ad hoc run through the matrix runner
python scripts/run_matrix.py --run --config configs/base.yaml key=value

# freeze the config and build the main matrix (see specs/SPEC.md section 9.2)
python scripts/freeze.py --config configs/base.yaml --bench-dir <dir> --P <n> --H <seconds>
```

Raw runs are written to `results/runs/` (not tracked). Timing benchmarks and the full matrix are meant to run on Wisteria/BDEC-01;
a local GPU is for development only. See `scripts/wisteria/` (copy `env.example.sh` to `env.sh` and fill it in).

## Original SCARCE demo

The original implementation is still runnable from the repository root:

```bash
python main.py -ds mnist -gen random -me SCARCE -mo mlp -op adam -lr 1e-3 -wd 1e-5 -bs 256 -ep 200 -seed 0 -gpu 0
```

## Acknowledgement and citation

Built on the official SCARCE implementation (https://github.com/wwangwitsel/SCARCE).

```
@inproceedings{wang2024learning,
    author = {Wang, Wei and Ishida, Takashi and Zhang, Yu-Jie and Niu, Gang and Sugiyama, Masashi},
    title = {Learning with complementary labels revisited: The selected-completely-at-random setting is more practical},
    booktitle = {Proceedings of the 41st International Conference on Machine Learning},
    year = {2024}
}
```
