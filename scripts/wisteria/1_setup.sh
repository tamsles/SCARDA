#!/bin/bash
# Step 1: .venv + requirements + MNIST download, GPU check, unit tests.
# Idempotent; if compute nodes have no internet access, run it once on a login node first
# (REPO=$PWD LOGDIR=/tmp bash scripts/wisteria/1_setup.sh), then the job step only re-checks.
source "$REPO/scripts/wisteria/common.sh"
echo "[setup] $(date) host=$(hostname) repo=$REPO"
# all runs must come from one committed code version: refuse modified tracked files
if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
    echo "[abort] tracked files are modified:"
    git status --porcelain --untracked-files=no
    exit 3
fi
git rev-parse HEAD | tee "$LOGDIR/git_commit.txt"
[ -d .venv ] || "$PYTHON_BIN" -m venv .venv
activate_venv
python -m pip install --upgrade pip
python -m pip install -r requirements.txt --extra-index-url "$TORCH_INDEX"
python scripts/wisteria/check_env.py
python -m pytest -q
mark_ok setup
