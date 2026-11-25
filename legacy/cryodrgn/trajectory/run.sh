#!/usr/bin/env bash
# Decode the K=20 bin means (prepare.py) -> mean -> ChimeraX align to the J1497 class-8 frame -> protein mask ->
# render at 35,000 A^3. Usage: run.sh DATASET_DIR APIX128 GPU
D=$1; APIX=$2; G=$3; T=$D/traj20; RUN=$D/train_fullset_128_nuposes; S=$(dirname $(readlink -f $0))
PY=~/miniconda3/envs/cryodrgn/bin/python; CX=/data/hCFTR/mae2183/tools/chimerax/usr/bin/chimerax; P=/home/mae2183/miniconda3/pkgs
export LD_LIBRARY_PATH=$P/mesalib-24.3.4-h0b126fc_1/lib:$P/libllvm19-19.1.7-hf7376ad_2/lib:/home/mae2183/miniconda3/lib
for d in lat vol; do CUDA_VISIBLE_DEVICES=$G ~/miniconda3/envs/cryodrgn/bin/cryodrgn eval_vol $RUN/weights.50.pkl -c $RUN/config.yaml \
  --zfile $T/z_$d.txt -o $T/dec_$d --Apix $APIX > $T/dec_$d.log 2>&1 || exit 1; done
$PY -c "
import mrcfile, numpy as np, glob
fs=sorted(glob.glob('$T/dec_lat/*.mrc'))+sorted(glob.glob('$T/dec_vol/*.mrc'))
A=np.mean([mrcfile.read(f) for f in fs],0).astype(np.float32)
with mrcfile.new('$T/dec_mean.mrc',overwrite=True) as o: o.set_data(A); o.voxel_size=$APIX"
TRAJ=$T $CX --nogui --offscreen --exit --script $S/cx_align.py > $T/cx_align.log 2>&1
$PY $S/premask.py $T
TRAJ=$T $CX --nogui --offscreen --exit --script $S/cx_render.py > $T/cx_render.log 2>&1
echo "$(date) $D: $(grep -c SAVED $T/cx_align.log) aligned, $(grep -c RENDER $T/cx_render.log) rendered, fit $(cat $T/aligned/fit.json)"
