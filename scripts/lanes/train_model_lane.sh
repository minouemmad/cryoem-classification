#!/usr/bin/env bash
# hudson lane: train a RAE variant (whole 128 px stack, NU poses) -> decode 2,000 maps -> label-free volume-PC scores ->...
# train_model_lane.sh GPU SPEC...    SPEC = DS:NAME:SEED:ZDIM:BETA:LAMBDA:BOX     DS J4001|J4624, BOX 128 / 96 / 64 (stack...
# Skips every step whose output exists (so a model trained elsewhere can be copied to <DS>/rv_models/<NAME> and run through...
SCR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
R=/data/hCFTR/mae2183/cryoem-classification/results_cryodrgn; RV=$R/J4001_gP25_WT/method_rank/rv; PY=/home/mae2183/miniconda3/envs/cryodrgn/bin/python; SC=$SCR/lib; GPU=$1; shift
log() { echo "$(date '+%m-%d %H:%M') [GPU $GPU] $*" >> $RV/train_$GPU.log; }
for spec in "$@"; do IFS=: read ds nm seed z beta lam box <<< "$spec"; box=${box:-128}
  if [ $ds = J4001 ]; then W=$R/J4001_gP25_WT; G=$W/candidate_groups_epoch50; DEC=$G/decoded2000/${nm}_e50; SCP=$G/axis_survey/pc_scan_J4001_${nm}_e50; MASK=$G/fullset/mask.mrc; LAB=$W/method_rank/j2923_labels.npy
  else W=$R/J4624_eP37_WT_T2a; DEC=$W/recon_tree/decoded2000_${nm}_e50; SCP=$W/recon_tree/pc_scan_J4624_${nm}_e50; MASK=$W/recon_tree/mask64.mrc; LAB=$W/recon_tree/heteroref_J4624_classes/leaves.npy; fi
  OUT=$W/rv_models/$nm; mkdir -p $OUT $(dirname $DEC); ST=$W/inputs/particles.128.mrcs; CT=$W/inputs/ctf.pkl; AP128=$($PY -c "import pickle;c=pickle.load(open('$CT','rb'));print(c[0,1])"); NAT=
  if [ $box != 128 ]; then ST=$W/inputs/particles.$box.mrcs; CT=$W/inputs/ctf.$box.pkl; NAT=$($PY -c "print($AP128*128/$box)")
    [ -f $ST ] || { log "downsample $ds -> $box px"; $PY -c "import sys;sys.argv=['x']" ; /home/mae2183/miniconda3/envs/cryodrgn/bin/cryodrgn downsample $W/inputs/particles.128.mrcs -D $box -o $ST --chunk 50000 >> $RV/train_$GPU.log 2>&1 || { log "FAILED downsample"; continue; }; }
    [ -f $CT ] || $PY -c "import pickle;c=pickle.load(open('$W/inputs/ctf.pkl','rb')).copy();c[:,0]=$box;c[:,1]=$AP128*128/$box;pickle.dump(c,open('$CT','wb'))"; fi
  LOADARG=; ls $OUT/weights.*.pkl > /dev/null 2>&1 && LOADARG="--load latest"
  if [ ! -f $OUT/weights.50.pkl ]; then log "train $ds $nm (seed $seed z $z beta $beta lambda $lam box $box)"
    (CUDA_VISIBLE_DEVICES=$GPU OMP_NUM_THREADS=4 $PY $SCR/01_train/train_rae.py $ST --poses $W/nu_consensus/poses.pkl --ctf $CT --zdim $z --enc-dim 1024 --enc-layers 3 --dec-dim 1024 --dec-layers 3 --no-amp -n 50 --lazy --num-workers 4 \
      --batch-size 64 --checkpoint 5 --log-interval 5000 --seed $seed --no-analysis --rae-beta $beta --rae-lambda $lam $LOADARG -o $OUT >> $OUT/run.log 2>&1) || { log "FAILED train $nm"; continue; }; fi
  if [ ! -f $DEC/sample.json ]; then log "decode $nm"; (CUDA_VISIBLE_DEVICES=$GPU $PY $SCR/02_decode_and_pc/decode.py --run $OUT --epoch 50 --samples 2000 --box 64 ${NAT:+--native-apix $NAT} --device 0 -o $DEC > $DEC.log 2>&1) || { log "FAILED decode $nm"; continue; }; fi
  if [ ! -f $SCP.scores.npy ]; then log "scores $nm"; OMP_NUM_THREADS=8 $PY $SCR/02_decode_and_pc/volume_pc_scores.py $DEC $MASK $SCP 10 $LAB > $SCP.log 2>&1 || { log "FAILED scores $nm"; continue; }; fi
  bash $SCR/lanes/residual_lane.sh $GPU $ds:$nm:$SCP.scores.npy; log "done $ds $nm"
done
log "train lane finished"
