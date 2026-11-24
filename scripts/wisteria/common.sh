# Shared by the job scripts: environment, repo root, GPU list, step markers.
# Expects REPO and LOGDIR (exported by submit.sh through pjsub -x).
set -euo pipefail
: "${REPO:?REPO not set}"
: "${LOGDIR:?LOGDIR not set}"
cd "$REPO"
[ -f scripts/wisteria/env.sh ] || { echo "scripts/wisteria/env.sh missing: copy env.example.sh and fill it in"; exit 3; }
source scripts/wisteria/env.sh
if [ -n "${MODULES// /}" ] && [[ "$MODULES" != *"<"* ]] && command -v module >/dev/null 2>&1; then
    module load $MODULES
fi
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export PYTHONUNBUFFERED=1
EXTRA="${EXTRA:-}"   # key=value overrides for every run; local tests only, empty on Wisteria

activate_venv() {
    if [ -f "$REPO/.venv/bin/activate" ]; then source "$REPO/.venv/bin/activate"
    else source "$REPO/.venv/Scripts/activate"; fi   # Windows venv layout (local tests)
}

# GPUs visible to this job -> bash array GPUS (values to put into CUDA_VISIBLE_DEVICES)
gpu_list() {
    if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
        IFS=',' read -r -a GPUS <<< "$CUDA_VISIBLE_DEVICES"
    else
        mapfile -t GPUS < <(nvidia-smi --query-gpu=index --format=csv,noheader | tr -d ' \r')
    fi
    echo "[gpus] ${GPUS[*]}"
}

# The previous step must have left its marker (second guard besides the PJM step dependency).
require_ok() {
    if [ ! -f "$LOGDIR/$1.ok" ]; then
        echo "[abort] $LOGDIR/$1.ok missing: step '$1' did not succeed"
        exit 3
    fi
}
mark_ok() { date '+%Y-%m-%dT%H:%M:%S%z' > "$LOGDIR/$1.ok"; echo "[ok] step $1"; }
