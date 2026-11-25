#!/usr/bin/env bash
# Consensus poses + cryoDRGN inputs for one dataset (after prepare.py NAME, write_star.py NAME and cs_pose_jobs.py inside cryosparcm cli):
#   cryoSPARC: import -> homogeneous refinement -> backprojection -> NU refinement, each queued once its parent completes
#   cryoDRGN:  ctf.pkl and the 128 px stack (in parallel) -> NU poses in particle order -> poses.pkl
# Usage: consensus.sh NAME CRYOSPARC_GPU     -> $RES/NAME/inputs/{particles.128.mrcs, ctf.pkl, poses.pkl}; log $RES/NAME/consensus.log
set -u; source "$(dirname "$(readlink -f "$0")")/../../config/env.sh"; H=$ROOT/pipeline/0_consensus
NAME=$1; GPU=$2; R=$RES/$NAME; N=$R/nu_consensus; I=$R/inputs; J=$N/cryosparc_job.json; LOG=$R/consensus.log; mkdir -p $I
log() { echo "$(date '+%m-%d %H:%M') $*" >> $LOG; }
jget() { python3 -c "import json; print(json.load(open('$1'))['$2'])"; }
status() { $CSM cli "api.jobs.get_status('$1','$2').value" 2>/dev/null | tail -n 1 | tr -d "'\""; }
cs_run() {   # PROJECT JOB [GPU]: enqueue if building, wait; non-zero if it fails or is killed
  [ "$(status $1 $2)" = building ] && { $CSM cli "api.jobs.enqueue('$1','$2', hostname='$CS_HOST'${3:+, gpus=[$3]})" > /dev/null 2>&1; log "enqueued $1 $2"; }
  while true; do case $(status $1 $2) in completed) log "$1 $2 completed"; return 0;; failed|killed) log "$1 $2 failed"; return 1;; esac; sleep 120; done; }
P=$(jget $J project); BOX=$(jget $N/prep.json box); APIX=$(jget $N/prep.json apix)
( [ -s $I/ctf.pkl ] || $CRYODRGN parse_ctf_csparc $N/${NAME}_particles.cs -D $BOX --Apix $APIX -o $I/ctf.pkl > $I/parse_ctf.log 2>&1
  [ -f $I/particles.128.done ] || { $CRYODRGN downsample $N/${NAME}_particles.cs -D 128 -o $I/particles.128.mrcs > $I/downsample_128.log 2>&1 && touch $I/particles.128.done; } ) &
for j in import_volume import_particles; do cs_run $P $(jget $J $j) || exit 1; done
for j in homo backproject nu; do cs_run $P $(jget $J $j) $GPU || exit 1; done
$PY $H/reorder_poses.py $NAME >> $LOG 2>&1 || { log "pose reorder failed"; exit 1; }
$CRYODRGN parse_pose_csparc $N/${NAME}_particles_NUposes.cs -D $BOX -o $N/poses.pkl > $N/parse_pose.log 2>&1 || { log "parse_pose failed"; exit 1; }
ln -sf $N/poses.pkl $I/poses.pkl; wait
[ -f $I/particles.128.done ] && [ -s $I/ctf.pkl ] && log "inputs ready: $I" || { log "cryoDRGN inputs failed (see $I/*.log)"; exit 1; }
