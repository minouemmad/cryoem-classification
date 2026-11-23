#!/usr/bin/env bash
# hudson lane: for each spec  DS:TAG:SCOREFILE[:K[:HI]]  run the residual-axis model (residual_fit.py, no-evaluation mode),...
# residual_lane.sh GPU SPEC [SPEC ...]       log rv/lane_<GPU>.log, per-tag output rv/ppca_<TAG>.txt / _z.npy; skips a step...
SCR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
R=/data/hCFTR/mae2183/cryoem-classification/results_cryodrgn; RV=$R/J4001_gP25_WT/method_rank/rv; PY=/home/mae2183/miniconda3/envs/cryodrgn/bin/python; SC=$SCR/lib; GPU=$1; shift
log() { echo "$(date '+%m-%d %H:%M') [GPU $GPU] $*" >> $RV/lane_$GPU.log; }
for spec in "$@"; do IFS=: read ds tag sc K HI <<< "$spec"; K=${K:-8}; HI=${HI:-7.0}
  if [ $ds = J4001 ]; then W=$R/J4001_gP25_WT; EN=$RV/env_J4001.npy; else W=$R/J4624_eP37_WT_T2a; EN=$W/recon_tree/env128_for_ppca.npy; fi
  mkdir -p $RV/$tag; cp -n $EN $RV/$tag/env.npy
  if [ ! -f $RV/$tag/ppca_${tag}_z.npy ]; then log "ppca $ds $tag (scores $sc, K $K, band 50-$HI A)"
    CUDA_VISIBLE_DEVICES=$GPU OMP_NUM_THREADS=8 MCR_W=$W MCR_A=$RV MCR_SC=$SCR/lib MCR_NOEVAL=1 MCR_SCORES=$sc $PY $SCR/03_find_volumes/residual_axis/residual_fit.py $RV/$tag $tag real $K 30 $HI > $RV/$tag/ppca.stdout 2>&1 || { log "FAILED ppca $tag"; continue; }; fi
  log "split + submit $ds $tag"; $PY $SCR/03_find_volumes/residual_axis/residual_split.py $ds $tag $sc $RV/$tag/ppca_${tag}_z.npy > $RV/$tag/mkset.out 2>&1 && bash $SCR/04_refine_cryosparc/submit_split.sh $ds $tag || log "FAILED split/submit $tag"; log "done $ds $tag"
done
log "lane finished"
