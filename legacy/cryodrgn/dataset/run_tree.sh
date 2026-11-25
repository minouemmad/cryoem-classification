#!/usr/bin/env bash
# Extra recon tree for a dataset after run_pipeline.sh: waits for its "all done", takes a free GPU, runs one tree.
# Usage: run_tree.sh NAME [AXIS=pc1|latent] [KRULE=best|incremental]      Log: results_cryodrgn/NAME/pipeline.log
set -u
source "$(dirname "$(readlink -f "$0")")/lib.sh"
NAME=$1; AX=${2:-pc1}; KR=${3:-best}; LOG=$RES/$NAME/pipeline.log
until grep -q "all done" $LOG; do sleep 600; done
gpu_lock ${NAME}_${AX}_$KR
run_tree $NAME $AX $KR
