#!/bin/bash
# Full label-free pipeline for ONE trained model (RAE / VAE / AE), any dataset in config/datasets.tsv; every step is skipped if its output exists:
#   decode medoid maps -> volume-PC scores -> residual model -> sharpened main coordinate -> fully conditioned search -> classes (-> reassignment)
# Usage: run_model.sh GPU DS NAME MODELDIR EPOCH [OPTS]          -> $WORK/pipe/DS/NAME/ (DONE when finished; log $WORK/pipe.log)
#   OPTS = comma list KEY=VALUE exported to every step.  Knobs (defaults):
#     MP_DECBOX 64, MP_NMED 2000 (decode) | MP_SCFW 8 (score smoothing FWHM, A) | MP_K 8, MP_HI 7, MP_LO 50 (residual model: directions, band A)
#     MP_LP 10, MP_GRID 64 (map low-pass A / grid for sharpen, search, classes) | MP_NC_SH 200, MP_NC_IT 400, MP_NC_CL 300 (cells per step)
#     MP_KEEP (classes on the first KEEP coordinates only) | MP_REASSIGN=1 (+ MP_RE_K MP_RE_LO MP_RE_HI MP_RE_IT MP_RE_NULL)
#   Sharing upstream steps of the same model: REUSE_DEC=DS/NAME (decode) | REUSE_SC (decode + scores) | REUSE (+ first residual model) | REUSE_IT (all but classes)
source "$(dirname "$(readlink -f "$0")")/../config/env.sh"; H=$ROOT/pipeline
GPU=$1; DS=$2; NM=$3; MD=$4; EP=$5
read _ STK POS CTF ENV64 MASK AP64 MAXR <<< "$(awk -v d=$DS '$1==d' $DATASETS)"; [ -n "$STK" ] || { echo "unknown dataset $DS"; exit 1; }
OUT=$WORK/pipe/$DS/$NM; mkdir -p $OUT; echo "${6:-default}" > $OUT/opts.txt
link() { local U=$WORK/pipe/$1 wait=$2; shift 2; until [ -e $U/$wait ]; do sleep 300; done; for f in "$@"; do [ -e $OUT/$f ] || ln -s $U/$f $OUT/$f; done; }
for kv in $(echo "${6:-}" | tr ',' ' '); do v=${kv#*=}; case ${kv%%=*} in
  REUSE_DEC) link $v dec/sample.json dec;;  REUSE_SC) link $v pc.scores.npy dec pc.scores.npy;;  REUSE) link $v ppca/ppca_base_z.npy dec pc.scores.npy ppca;;
  REUSE_IT) link $v iter2/S_final.npy dec pc.scores.npy ppca e2e_s.npy iter2;;  *) export "$kv";; esac; done
DBOX=${MP_DECBOX:-64}; SMASK=$MASK
if [ "$DBOX" = 128 ]; then SMASK=$OUT/mask128.mrc; [ -f $SMASK ] || $PY -c "
import mrcfile, numpy as np
v = np.asarray(mrcfile.open('$MASK', permissive=True).data, dtype=np.float32); v = np.repeat(np.repeat(np.repeat(v, 2, 0), 2, 1), 2, 2)
with mrcfile.new('$SMASK', overwrite=True) as o: o.set_data(v); o.voxel_size = $AP64 / 2"; fi
log() { echo "$(date '+%m-%d %H:%M') [gpu$GPU] $DS/$NM $*" >> $WORK/pipe.log; }
step() { local name=$1 out=$2; shift 2; [ -e "$out" ] && return 0; log "$name"; "$@" > $OUT/$name.log 2>&1 || { log "FAILED $name (see $OUT/$name.log)"; exit 1; }; }
export MP_OUT=$OUT MP_ENV64=$ENV64 MP_STACK=$STK MP_POSES=$POS MP_CTF=$CTF MP_AP64=$AP64 CUDA_VISIBLE_DEVICES=$GPU OMP_NUM_THREADS=8
step decode $OUT/dec/sample.json $PY $H/3_scores/decode.py --run $MD --epoch $EP --samples ${MP_NMED:-2000} --box $DBOX --native-apix $(awk -v a=$AP64 'BEGIN{print a/2}') --device 0 -o $OUT/dec
step scores $OUT/pc.scores.npy $PY $H/3_scores/scores.py $OUT/dec $SMASK $OUT/pc 10
mkdir -p $OUT/ppca; [ -f $OUT/ppca/env.npy ] || $PY -c "import numpy as np; e = np.load('$ENV64').astype(bool); np.save('$OUT/ppca/env.npy', np.repeat(np.repeat(np.repeat(e, 2, 0), 2, 1), 2, 2))"
step ppca $OUT/ppca/ppca_base_z.npy env MCR_STACK=$STK MCR_POSES=$POS MCR_CTF=$CTF MCR_SCORES=$OUT/pc.scores.npy $PY $H/4_residual/residual_ppca.py $OUT/ppca base ${MP_K:-8} 30 ${MP_HI:-7.0} ${MP_LO:-50.0}
step sharpen $OUT/e2e_s.npy env MP_SCORES=$OUT/pc.scores.npy MP_Z=$OUT/ppca/ppca_base_z.npy $PY $H/4_residual/sharpen.py
step search $OUT/iter2/S_final.npy $PY $H/5_search/search.py $MAXR
step classes $OUT/mc/MP.json $PY $H/6_classes/classes.py
[ -n "${MP_REASSIGN:-}" ] && step reassign $OUT/reassign$([ "${MP_RE_NULL:-}" = 1 ] && echo _null)/done.json $PY $H/6_classes/reassign.py
touch $OUT/DONE; log "done: $(grep -o '"elbow": [0-9]*\|"offblend_halving": [0-9]*' $OUT/classes.log | tr '\n' ' ')"
