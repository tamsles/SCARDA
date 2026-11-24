#!/bin/bash
# Step 2 (3 GPUs), in parallel:
#   GPU 0, 1: DIW selection runs B=5 / B=10 (rho=20, n_te=2000, seed 0, base-config epochs = 100; SPEC §7)
#   GPU 2:    bench_time.py, 5 epochs each of source_only, diw, diw_combined (DIW with diw_warmup_epochs=0,
#             so that every timed epoch solves KMM; t of SPEC §9.2 = the slower DIW method)
# The selection run whose B is frozen has the same config as the matrix run (diw, 20, 2000, 0) and is reused there.
source "$REPO/scripts/wisteria/common.sh"
require_ok setup
activate_venv
gpu_list
if [ "${#GPUS[@]}" -lt 3 ]; then echo "[abort] need 3 GPUs, have ${#GPUS[@]}"; exit 3; fi
SEL="method=diw rho=20 n_te=2000 seed=0"
mkdir -p "$LOGDIR/bench"

CUDA_VISIBLE_DEVICES=${GPUS[0]} python scripts/run_matrix.py --config configs/base.yaml --run $EXTRA $SEL B=5 \
    > "$LOGDIR/select_B5.out" 2>&1 &
p5=$!
CUDA_VISIBLE_DEVICES=${GPUS[1]} python scripts/run_matrix.py --config configs/base.yaml --run $EXTRA $SEL B=10 \
    > "$LOGDIR/select_B10.out" 2>&1 &
p10=$!
(
    set -e
    export CUDA_VISIBLE_DEVICES=${GPUS[2]}
    python scripts/bench_time.py --config configs/base.yaml --epochs 5 --json "$LOGDIR/bench/source_only.json" \
        $EXTRA method=source_only
    for m in diw diw_combined; do
        python scripts/bench_time.py --config configs/base.yaml --epochs 5 --json "$LOGDIR/bench/$m.json" \
            $EXTRA method=$m diw_warmup_epochs=0
    done
) > "$LOGDIR/bench.out" 2>&1 &
pb=$!

rc=0
wait $p5 || { echo "[fail] selection run B=5: see $LOGDIR/select_B5.out and the run's run.log"; rc=1; }
wait $p10 || { echo "[fail] selection run B=10: see $LOGDIR/select_B10.out and the run's run.log"; rc=1; }
wait $pb || { echo "[fail] bench: see $LOGDIR/bench.out"; rc=1; }
cat "$LOGDIR/bench.out"
if [ $rc -ne 0 ]; then exit $rc; fi
mark_ok select
