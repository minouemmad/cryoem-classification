#!/usr/bin/env bash
# One dataset end to end, after prepare.py NAME, write_star.py NAME and cs_pose_jobs.py (in cryosparcm cli):
#   cryoSPARC: import -> homogeneous refinement -> backprojection -> NU refinement (poses), each queued when its parent is done
#   cryoDRGN:  ctf.pkl + 128 px stack (in parallel) -> NU poses -> VAE (zdim 10, 1024x3, 50 epochs, seed 44688) on a free GPU
#              -> decode 2000 centres -> 64 px mask -> recon trees along volume PC1 and latent PC1 (incremental K rule)
# Usage: run_pipeline.sh NAME CRYOSPARC_GPU      Log: results_cryodrgn/NAME/pipeline.log
set -u
source "$(dirname "$(readlink -f "$0")")/lib.sh"
NAME=$1; CSGPU=$2
R=$RES/$NAME; N=$R/nu_consensus; I=$R/inputs; T=$R/train_fullset_128_nuposes; J=$N/cryosparc_job.json; LOG=$R/pipeline.log
mkdir -p $I
P=$(jget $J project); BOX=$(jget $N/prep.json box); APIX=$(jget $N/prep.json apix); APIX128=$(jget $N/prep.json apix_128)
log "start $NAME ($P, cryoSPARC GPU $CSGPU)"

(   # cryoDRGN inputs (CPU, background)
  [ -s $I/ctf.pkl ] || $CD parse_ctf_csparc $N/${NAME}_particles.cs -D $BOX --Apix $APIX -o $I/ctf.pkl > $I/parse_ctf.log 2>&1
  [ -f $I/particles.128.done ] || { $CD downsample $N/${NAME}_particles.cs -D 128 -o $I/particles.128.mrcs > $I/downsample_128.log 2>&1 && touch $I/particles.128.done; }
) &
DS_PID=$!

for j in import_volume import_particles; do cs_run $P $(jget $J $j) || exit 1; done
for j in homo backproject nu; do cs_run $P $(jget $J $j) $CSGPU || exit 1; done

$PY $SC/dataset/reorder_poses.py $NAME >> $LOG 2>&1 || { log "pose reorder failed"; exit 1; }
$CD parse_pose_csparc $N/${NAME}_particles_NUposes.cs -D $BOX -o $N/poses.pkl > $N/parse_pose.log 2>&1 || { log "parse_pose failed"; exit 1; }
ln -sf $N/poses.pkl $I/poses.pkl
wait $DS_PID
[ -f $I/particles.128.done ] && [ -s $I/ctf.pkl ] || { log "cryoDRGN inputs failed (see $I/*.log)"; exit 1; }
log "inputs ready: $I"

gpu_lock $NAME
log "training on GPU $GPU -> $T"
if [ ! -f $T/weights.50.pkl ]; then
  CUDA_VISIBLE_DEVICES=$GPU $CD train_vae $I/particles.128.mrcs --poses $I/poses.pkl --ctf $I/ctf.pkl --zdim 10 --enc-dim 1024 \
    --enc-layers 3 --dec-dim 1024 --dec-layers 3 --no-amp -n 50 --lazy --num-workers 4 --batch-size 64 --checkpoint 5 \
    --log-interval 5000 --seed 44688 -o $T > $T.log 2>&1 || { log "training failed"; exit 1; }
fi
log "training done"

DEC=$R/recon_tree/decoded2000_nuposes_e50; mkdir -p $R/recon_tree
if [ ! -f $DEC/sample.json ]; then
  CUDA_VISIBLE_DEVICES=$GPU $PY $SC/decode.py --run $T --epoch 50 --samples 2000 --box 64 --native-apix $APIX128 \
    --device 0 -o $DEC > $DEC.log 2>&1 || { log "decode failed"; exit 1; }
fi
$PY $SC/dataset/make_mask.py $DEC $R/recon_tree/mask64.mrc >> $LOG 2>&1 || { log "mask failed"; exit 1; }
for AX in pc1 latent; do run_tree $NAME $AX incremental; done
log "all done"
