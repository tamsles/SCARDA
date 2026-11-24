#!/bin/bash
# Step 3: scripts/freeze.py -> configs/frozen.yaml, configs/matrix_main.yaml, results/checks/freeze.json.
# freeze.py exits 2 on abort (NaN, ESS degeneration, missing / incomplete run, bad bench file,
# SPEC §9.2 "stop and report"); the reason is printed into this step's log and into freeze.json.
source "$REPO/scripts/wisteria/common.sh"
require_ok select
activate_venv
rc=0
python scripts/freeze.py --config configs/base.yaml --bench-dir "$LOGDIR/bench" --P "$P" --H "$H" \
    --out-frozen "${OUT_FROZEN:-configs/frozen.yaml}" --out-matrix "${OUT_MATRIX:-configs/matrix_main.yaml}" \
    --out-record "${OUT_RECORD:-results/checks/freeze.json}" $EXTRA || rc=$?
cp "${OUT_RECORD:-results/checks/freeze.json}" "$LOGDIR/" || true
if [ $rc -ne 0 ]; then echo "[abort] freeze.py exit code $rc: matrix will not run"; exit $rc; fi
cp "${OUT_FROZEN:-configs/frozen.yaml}" "${OUT_MATRIX:-configs/matrix_main.yaml}" "$LOGDIR/"
mark_ok freeze
