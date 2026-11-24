# Template: cp scripts/wisteria/env.example.sh scripts/wisteria/env.sh, then fill in every <...> in env.sh (untracked).
# Sourced by submit.sh and by every job script.

GROUP="<project code>"                         # pjsub -g
MODULES="<modules to load, space separated>"   # a python >= 3.10 module; the torch wheels bring their own CUDA runtime
PYTHON_BIN="python3"                           # interpreter (after module load) used to create .venv

# SPEC §9.2: P = runs executed at the same time (= GPU workers in the matrix job), H = M4 wall budget [s]
P="<P>"
H="<H seconds>"

# Resource group and resources per step, passed as -L rscgrp=<RSCGRP>,<RES>,elapse=<ELAPSE>.
# Node groups (e.g. regular-a) take node=1 (8 GPUs); share groups take gpu=N.
RSCGRP_SETUP="<rscgrp>";   RES_SETUP="gpu=1";   ELAPSE_SETUP="01:00:00"
RSCGRP_SELECT="<rscgrp>";  RES_SELECT="gpu=3";  ELAPSE_SELECT="08:00:00"   # 2 selection runs (E=100) + bench, in parallel
RSCGRP_FREEZE="<rscgrp>";  RES_FREEZE="gpu=1";  ELAPSE_FREEZE="00:10:00"
RSCGRP_MATRIX="<rscgrp>";  RES_MATRIX="gpu=<P>"                             # needs >= P GPUs
ELAPSE_MATRIX=""                                                            # empty: H converted to hh:mm:ss

# pip index for the torch CUDA wheels (see requirements.txt header)
TORCH_INDEX="https://download.pytorch.org/whl/cu124"
