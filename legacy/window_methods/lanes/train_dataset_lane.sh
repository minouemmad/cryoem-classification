#!/usr/bin/env bash
# RAE (same recipe as J4001 base RAE: z10, 1024x3 enc/dec, beta 0.1, lambda 0.00243, seed 30112, 50 ep, 128 px, NU poses) for...
# train_dataset_lane.sh GPU DS...   DS = J399|J2708|J207.
# train -> decode 2,000 -> label-free volume-PC scores.
SCR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
R=/data/hCFTR/mae2183/cryoem-classification/results_cryodrgn; RV=$R/J4001_gP25_WT/method_rank/rv; PY=/home/mae2183/miniconda3/envs/cryodrgn/bin/python; SC=$SCR/lib; GPU=$1; shift
log() { echo "$(date '+%m-%d %H:%M') [GPU $GPU] $*" >> $RV/train_ds_$GPU.log; }
for ds in "$@"; do
  case $ds in J399) W=$R/J399_hP7_E1371Q_ATP;; J2708) W=$R/J2708_eP30_F508del_E1371W;; J207) W=$R/J207_hP12_IDOR4;; *) log "unknown $ds"; continue;; esac
  CFG=$W/train_fullset_128_nuposes/config.yaml
  ST=$(grep -E '^  particles:' $CFG | awk '{print $2}'); PO=$(grep -E '^  poses:' $CFG | awk '{print $2}'); CT=$(grep -E '^  ctf:' $CFG | awk '{print $2}')
  OUT=$W/rv_models/rae_base; DEC=$W/recon_tree/decoded2000_rae_base_e50; SCP=$W/recon_tree/pc_scan_${ds}_rae_base_e50; MASK=$W/recon_tree/mask64.mrc; mkdir -p $OUT
  LOADARG=; ls $OUT/weights.*.pkl > /dev/null 2>&1 && LOADARG="--load latest"
  if [ ! -f $OUT/weights.50.pkl ]; then log "train $ds ($ST | $PO | $CT)"
    (CUDA_VISIBLE_DEVICES=$GPU OMP_NUM_THREADS=4 $PY $SCR/01_train/train_rae.py $ST --poses $PO --ctf $CT --zdim 10 --enc-dim 1024 --enc-layers 3 --dec-dim 1024 --dec-layers 3 --no-amp -n 50 --lazy --num-workers 4 \
      --batch-size 64 --checkpoint 5 --log-interval 5000 --seed 30112 --no-analysis --rae-beta 0.1 --rae-lambda 0.00243 $LOADARG -o $OUT >> $OUT/run.log 2>&1) || { log "FAILED train $ds"; continue; }; fi
  if [ ! -f $DEC/sample.json ]; then log "decode $ds"; (CUDA_VISIBLE_DEVICES=$GPU $PY $SCR/02_decode_and_pc/decode.py --run $OUT --epoch 50 --samples 2000 --box 64 --device 0 -o $DEC > $DEC.log 2>&1) || { log "FAILED decode $ds"; continue; }; fi
  if [ ! -f $SCP.scores.npy ]; then log "scores $ds"; OMP_NUM_THREADS=8 $PY $SCR/02_decode_and_pc/volume_pc_scores.py $DEC $MASK $SCP 10 > $SCP.log 2>&1 || { log "FAILED scores $ds"; continue; }; fi
  log "done $ds"
done
log "ds lane finished"
