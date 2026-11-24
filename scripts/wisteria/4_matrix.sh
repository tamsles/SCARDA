#!/bin/bash
# Step 4: the M4 matrix. P workers, one per GPU, share configs/matrix_main.yaml through a claim directory
# (run_matrix.py --worker). Runs with final.json are skipped, so submitting this script again
# (pjsub ... -x REPO=...,LOGDIR=... scripts/wisteria/4_matrix.sh) resumes the matrix and retries only
# runs without final.json. Every run's output is in <run_dir>/run.log.
source "$REPO/scripts/wisteria/common.sh"
require_ok freeze
activate_venv
gpu_list
if [ "${#GPUS[@]}" -lt "$P" ]; then echo "[abort] P=$P workers but only ${#GPUS[@]} GPUs"; exit 3; fi
MATRIX="${OUT_MATRIX:-configs/matrix_main.yaml}"
CLAIM="$LOGDIR/claims_${PJM_JOBID:-$$}"
python scripts/run_matrix.py --matrix "$MATRIX" --list
pids=()
for ((w = 0; w < P; w++)); do
    CUDA_VISIBLE_DEVICES=${GPUS[$w]} python scripts/run_matrix.py --matrix "$MATRIX" --worker --claim-dir "$CLAIM" \
        > "$LOGDIR/matrix_worker$w.out" 2>&1 &
    pids+=($!)
done
rc=0
for p in "${pids[@]}"; do wait "$p" || rc=1; done
python scripts/run_matrix.py --matrix "$MATRIX" --status | tee "$LOGDIR/matrix_status.txt" || rc=1
if [ $rc -eq 0 ]; then mark_ok matrix; fi
exit $rc
