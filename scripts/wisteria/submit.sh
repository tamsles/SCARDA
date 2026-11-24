#!/bin/bash
# Submit the M3 -> M4 pipeline as one PJM step job with 4 sub-jobs. On a Wisteria login node, in the repo root:
#     bash scripts/wisteria/submit.sh
# A sub-job that exits non-zero deletes the remaining sub-jobs (sd=ec!=0:all); in addition every step
# refuses to run without the previous step's marker file in LOGDIR.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO"
[ -f scripts/wisteria/env.sh ] || { echo "scripts/wisteria/env.sh missing: copy env.example.sh and fill it in"; exit 3; }
source scripts/wisteria/env.sh
for v in GROUP MODULES P H RSCGRP_SETUP RSCGRP_SELECT RSCGRP_FREEZE RSCGRP_MATRIX RES_MATRIX; do
    case "${!v}" in *"<"*) echo "fill in $v in scripts/wisteria/env.sh"; exit 1 ;; esac
done
LOGDIR="$REPO/results/logs/pipeline_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOGDIR"
if [ -z "$ELAPSE_MATRIX" ]; then
    H_INT=${H%.*}
    ELAPSE_MATRIX=$(printf '%02d:%02d:%02d' $((H_INT / 3600)) $((H_INT % 3600 / 60)) $((H_INT % 60)))
fi
JID=""
sub() {  # step-number rscgrp resources elapse script
    local sparam="sn=$1"
    if [ "$1" -gt 1 ]; then sparam="jid=$JID,sn=$1,sd=ec!=0:all"; fi
    pjsub --step --sparam "$sparam" -g "$GROUP" -L "rscgrp=$2,$3,elapse=$4" -j \
        -o "$LOGDIR/step$1_$(basename "$5" .sh).log" -x "REPO=$REPO,LOGDIR=$LOGDIR" "scripts/wisteria/$5"
}
out=$(sub 1 "$RSCGRP_SETUP" "$RES_SETUP" "$ELAPSE_SETUP" 1_setup.sh)
echo "$out"
JID=$(echo "$out" | sed -n 's/.*Job \([0-9][0-9]*\)_[0-9]* submitted.*/\1/p')
if [ -z "$JID" ]; then echo "could not parse the job id from pjsub output"; exit 1; fi
sub 2 "$RSCGRP_SELECT" "$RES_SELECT" "$ELAPSE_SELECT" 2_select_bench.sh
sub 3 "$RSCGRP_FREEZE" "$RES_FREEZE" "$ELAPSE_FREEZE" 3_freeze.sh
sub 4 "$RSCGRP_MATRIX" "$RES_MATRIX" "$ELAPSE_MATRIX" 4_matrix.sh
echo "$JID" > "$LOGDIR/jobid.txt"
echo "submitted step job $JID (matrix elapse $ELAPSE_MATRIX); logs in $LOGDIR"
